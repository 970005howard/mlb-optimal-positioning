# 檔案位置: src/modeling/step_03_train_catch_model.py

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from pathlib import Path
import glob
from sklearn.preprocessing import StandardScaler
import joblib # 用於儲存 scaler 物件

# 從 config 匯入專案路徑
from config import PROCESSED_DATA_DIR, MODELS_DIR

# --- 1. 常數定義區 ---
# 輸入欄位
COL_CAUGHT = "caught"
COL_PLAYER_NAME = "player_name"
COL_FIELDER_DIST = "fielder_distance_to_ball"
COL_FLIGHT_TIME = "flight_time_s"

# 模型超參數
RANDOM_SEED = 42
DRAWS = 2000
TUNE = 1500
CHAINS = 4
TARGET_ACCEPT = 0.9
CORES = 4 # 根據您的 CPU 核心數設定

# --- 2. 主模型訓練函式區 ---
def define_and_run_model(position_code: str):
    """
    對指定守備位置的資料進行完整的階層式貝氏回歸模型訓練，
    使用標準化 (Standardization) 對特徵進行縮放，並儲存 Scaler。
    """
    print(f"--- 開始訓練守備位置: {position_code} 的接殺機率模型 ---")

    # 1. 動態建立路徑
    input_dir = PROCESSED_DATA_DIR / f"{position_code}_modified_data"
    output_dir = MODELS_DIR / position_code
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 載入並準備資料
    file_list = glob.glob(str(input_dir / f"*_with_all.csv"))
    if not file_list:
        print(f"[警告] 在 {input_dir} 中找不到任何由 step_02 產生的檔案。已跳過 {position_code} 的訓練。")
        return

    df_list = [pd.read_csv(f, encoding='utf-8') for f in file_list]
    df = pd.concat(df_list, ignore_index=True)

    required_cols = [COL_CAUGHT, COL_PLAYER_NAME, COL_FIELDER_DIST, COL_FLIGHT_TIME]
    df_model = df[required_cols].dropna().copy()
    
    if df_model.empty:
        print(f"[警告] 清理 NaN 後，沒有可用於訓練 {position_code} 模型的數據。")
        return
        
    print(f"  - 資料載入完成，共 {len(df_model)} 筆有效數據，{df_model[COL_PLAYER_NAME].nunique()} 位球員。")

    # 3. 使用 StandardScaler 進行標準化
    print("  - 正在對特徵進行標準化 (Standardization)...")
    features_to_scale = [COL_FIELDER_DIST, COL_FLIGHT_TIME]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_model[features_to_scale])
    scaled_feature_names = [f + '_scaled' for f in features_to_scale]
    df_model[scaled_feature_names] = scaled_features
    print(f"    - '{COL_FIELDER_DIST}' 縮放後: mean={df_model[scaled_feature_names[0]].mean():.2f}, std={df_model[scaled_feature_names[0]].std():.2f}")
    print(f"    - '{COL_FLIGHT_TIME}' 縮放後: mean={df_model[scaled_feature_names[1]].mean():.2f}, std={df_model[scaled_feature_names[1]].std():.2f}")

    # 4. 將 scaler 物件儲存起來，供後續步驟使用
    scaler_path = output_dir / f"{position_code}_scaler.joblib"
    try:
        joblib.dump(scaler, scaler_path)
        print(f"    - 標準化參數 (Scaler) 已儲存至: {scaler_path}")
    except Exception as e:
        print(f"❌ [錯誤] 儲存 Scaler 失敗: {e}")
        return
        
    # 將球員姓名轉換為整數索引
    player_idx, players = pd.factorize(df_model[COL_PLAYER_NAME])
    n_players = len(players)

    # 5. 定義 PyMC 階層模型
    print("  - 開始定義 PyMC 模型...")
    with pm.Model(coords={"player": players}) as model:
        mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=1)
        sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1)
        mu_beta_dist = pm.Normal('mu_beta_dist', mu=0, sigma=1)
        sigma_beta_dist = pm.HalfNormal('sigma_beta_dist', sigma=1)
        mu_beta_time = pm.Normal('mu_beta_time', mu=0, sigma=1)
        sigma_beta_time = pm.HalfNormal('sigma_beta_time', sigma=1)

        alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, dims="player")
        beta_dist = pm.Normal('beta_dist', mu=mu_beta_dist, sigma=sigma_beta_dist, dims="player")
        beta_time = pm.Normal('beta_time', mu=mu_beta_time, sigma=sigma_beta_time, dims="player")
        
        logit_p = (
            alpha[player_idx] + 
            beta_dist[player_idx] * df_model[scaled_feature_names[0]] + 
            beta_time[player_idx] * df_model[scaled_feature_names[1]]
        )
        y_obs = pm.Bernoulli('y_obs', logit_p=logit_p, observed=df_model[COL_CAUGHT])

    # 6. 執行模型推論
    print(f"  - 模型定義完成，開始使用 {CHAINS} 條鏈進行抽樣 (Draws={DRAWS}, Tune={TUNE}, Cores={CORES})...")
    with model:
        trace = pm.sample(
            draws=DRAWS, 
            tune=TUNE, 
            chains=CHAINS, 
            target_accept=TARGET_ACCEPT, 
            random_seed=RANDOM_SEED,
            cores=CORES 
        )

    # 7. 儲存結果
    print("  - 抽樣完成，正在儲存結果...")
    try:
        summary = az.summary(trace)
        summary_path = output_dir / f"{position_code}_posterior_summary.csv"
        summary.to_csv(summary_path)
        print(f"  - 模型參數摘要已儲存至: {summary_path}")

        trace_path = output_dir / f"{position_code}_model_trace.nc"
        trace.to_netcdf(trace_path)
        print(f"  - 完整的模型訓練 Trace 已儲存至: {trace_path}")
    except Exception as e:
        print(f"❌ [錯誤] 儲存模型結果時發生問題: {e}")
    
    print(f"--- {position_code} 模型訓練完成 ---\n")

def run_all_modeling():
    positions_to_process = ["CF", "LF", "RF"]
    print("==========================================")
    print("開始執行所有模型訓練任務...")
    print(f"目標守備位置: {positions_to_process}")
    print("==========================================")
    for pos in positions_to_process:
        define_and_run_model(pos)
    print("所有模型訓練任務已全部完成！")

if __name__ == "__main__":
    run_all_modeling()