# 檔案位置: src/optimization/step_04_find_optimal_position.py (使用 SLSQP)

import pandas as pd
import numpy as np
import arviz as az
from pathlib import Path
import time
import json
from scipy.optimize import minimize # 導入最佳化工具
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names, but StandardScaler was fitted with feature names")

# 從 config 匯入專案路徑
from config import MODELS_DIR, INPUTS_DATA_DIR, RESULTS_DIR
# 從「中央廚房」導入共用函式和常數
from src.utils.feature_engineering import calculate_batted_ball_features, COL_X_COORD, COL_Y_COORD, COL_FLIGHT_TIME, COL_PLAYER_NAME, COL_FIELDER_DIST

# --- 1. 常數定義區 ---
# 定義扇形約束的邊界 (請根據您的球場實際情況調整)
MIN_RADIUS = 100.0    # 最小半徑 (例如：內外野草皮交界)
MAX_RADIUS = 420.0   # 最大半徑 (例如：接近全壘打牆)
MIN_ANGLE_DEG = -45.0 # 最小角度 (例如：左外野邊線，0度朝向中外野)
MAX_ANGLE_DEG = 45.0  # 最大角度 (例如：右外野邊線)

# --- 2. 輔助函式區 ---
# ... (load_model_scaler_and_params, load_player_params, predict_catch_probability_scaled 等函式維持不變) ...
# (為求簡潔，此處省略未變動的程式碼)
def load_model_scaler_and_params(position_code: str) -> tuple:
    model_dir = MODELS_DIR / position_code
    trace_path = model_dir / f"{position_code}_model_trace.nc"
    scaler_path = model_dir / f"{position_code}_scaler.joblib"
    if not trace_path.exists(): raise FileNotFoundError(f"找不到 {position_code} 的模型 Trace 檔案: {trace_path}")
    if not scaler_path.exists(): raise FileNotFoundError(f"找不到 {position_code} 的 Scaler 檔案: {scaler_path}")
    trace = az.from_netcdf(trace_path)
    scaler = joblib.load(scaler_path)
    params = {'alpha': trace.posterior['alpha'].mean(dim=('chain', 'draw')).values,
              'beta_dist': trace.posterior['beta_dist'].mean(dim=('chain', 'draw')).values,
              'beta_time': trace.posterior['beta_time'].mean(dim=('chain', 'draw')).values,
              'players': trace.posterior['player'].values.tolist()}
    return scaler, params

def load_player_params(params: dict, player_name: str) -> dict:
    try:
        player_idx = params['players'].index(player_name)
        player_params = {'alpha': params['alpha'][player_idx],
                         'beta_dist': params['beta_dist'][player_idx],
                         'beta_time': params['beta_time'][player_idx],}
        return player_params
    except ValueError: raise ValueError(f"在模型參數中找不到球員 '{player_name}'。")
    except KeyError: raise KeyError("載入的參數字典格式不正確。")

def predict_catch_probability_scaled(fielder_distance_scaled, flight_time_scaled, player_params):
    logit_p = player_params['alpha'] + (player_params['beta_dist'] * fielder_distance_scaled) + (player_params['beta_time'] * flight_time_scaled)
    logit_p_clipped = np.clip(logit_p, -700, 700)
    return 1 / (1 + np.exp(-logit_p_clipped))

def objective_function_team(positions, batter_df, scaler_lf, scaler_cf, scaler_rf, lf_params, cf_params, rf_params):
    """目標函式，在內部應用標準化。"""
    lf_x, lf_y, cf_x, cf_y, rf_x, rf_y = positions
    ball_x = batter_df[COL_X_COORD].to_numpy()
    ball_y = batter_df[COL_Y_COORD].to_numpy()
    flight_time_raw = batter_df[COL_FLIGHT_TIME].to_numpy()
    dist_lf_raw = np.sqrt((ball_x - lf_x)**2 + (ball_y - lf_y)**2)
    dist_cf_raw = np.sqrt((ball_x - cf_x)**2 + (ball_y - cf_y)**2)
    dist_rf_raw = np.sqrt((ball_x - rf_x)**2 + (ball_y - rf_y)**2)
    features_lf = np.stack([dist_lf_raw, flight_time_raw], axis=1)
    features_cf = np.stack([dist_cf_raw, flight_time_raw], axis=1)
    features_rf = np.stack([dist_rf_raw, flight_time_raw], axis=1)
    scaled_features_lf = scaler_lf.transform(features_lf)
    scaled_features_cf = scaler_cf.transform(features_cf)
    scaled_features_rf = scaler_rf.transform(features_rf)
    dist_lf_scaled, time_lf_scaled = scaled_features_lf[:, 0], scaled_features_lf[:, 1]
    dist_cf_scaled, time_cf_scaled = scaled_features_cf[:, 0], scaled_features_cf[:, 1]
    dist_rf_scaled, time_rf_scaled = scaled_features_rf[:, 0], scaled_features_rf[:, 1]
    prob_lf = predict_catch_probability_scaled(dist_lf_scaled, time_lf_scaled, lf_params)
    prob_cf = predict_catch_probability_scaled(dist_cf_scaled, time_cf_scaled, cf_params)
    prob_rf = predict_catch_probability_scaled(dist_rf_scaled, time_rf_scaled, rf_params)
    prob_team_per_ball = 1 - (1 - prob_lf) * (1 - prob_cf) * (1 - prob_rf)
    total_team_catch_prob = np.sum(prob_team_per_ball)
    return -total_team_catch_prob

# 定義約束條件的函式
# =======================================================
def get_constraints():
    """定義扇形約束條件，應用於每個守備員。"""
    constraints = []
    # 遍歷三個守備員 (LF, CF, RF)，每個守備員有兩個座標 (x, y)
    for i in range(3):
        # 獲取該守備員的 x 和 y 座標在 6 維陣列中的索引
        x_idx, y_idx = i * 2, i * 2 + 1
        
        # 1. 最小半徑約束: sqrt(x^2 + y^2) >= MIN_RADIUS
        constraints.append({
            'type': 'ineq', # 不等式約束 (inequality)
            'fun': lambda pos, idx=y_idx, jdx=x_idx: np.sqrt(pos[idx]**2 + pos[jdx]**2) - MIN_RADIUS
        })
        
        # 2. 最大半徑約束: sqrt(x^2 + y^2) <= MAX_RADIUS
        constraints.append({
            'type': 'ineq',
            'fun': lambda pos, idx=y_idx, jdx=x_idx: MAX_RADIUS - np.sqrt(pos[idx]**2 + pos[jdx]**2)
        })
        
        # 3. 最小角度約束: atan2(x, y) >= MIN_ANGLE_DEG (角度以弧度計算)
        # 注意：使用 atan2(x, y) 得到的是以 Y 軸 (中外野) 為 0 度的角度
        min_angle_rad = np.radians(MIN_ANGLE_DEG)
        constraints.append({
            'type': 'ineq',
            'fun': lambda pos, idx=y_idx, jdx=x_idx: np.arctan2(pos[jdx], pos[idx]) - min_angle_rad
        })
        
        # 4. 最大角度約束: atan2(x, y) <= MAX_ANGLE_DEG
        max_angle_rad = np.radians(MAX_ANGLE_DEG)
        constraints.append({
            'type': 'ineq',
            'fun': lambda pos, idx=y_idx, jdx=x_idx: max_angle_rad - np.arctan2(pos[jdx], pos[idx])
        })
        
    return constraints
# =======================================================

# --- 3. 主流程函式 ---
def run_team_optimization(batter_name: str, fielder_names: dict):
    """主執行函式，執行使用 SLSQP 的團隊最佳化。"""
    print("==========================================")
    print(f"開始為打者 [{batter_name}] 和指定團隊尋找最佳防守佈陣 (使用 SLSQP)...")
    print("==========================================")
    
    # ... (載入打者數據 batter_df 和球員參數 lf_player_params 等的邏輯維持不變) ...
    # (為求簡潔，此處省略未變動的程式碼)
    batter_file = INPUTS_DATA_DIR / "batter_spray_charts" / f"{batter_name}.csv"
    batter_df_raw = pd.read_csv(batter_file, encoding='utf-8')
    batter_df_processed = calculate_batted_ball_features(batter_df_raw)
    batter_df = batter_df_processed.dropna(subset=[COL_X_COORD, COL_Y_COORD, COL_FLIGHT_TIME])
    print(f"  - 已載入並處理 [{batter_name}] 的 {len(batter_df)} 筆有效擊球數據。")
    try:
        scaler_lf, params_lf_all = load_model_scaler_and_params("LF")
        scaler_cf, params_cf_all = load_model_scaler_and_params("CF")
        scaler_rf, params_rf_all = load_model_scaler_and_params("RF")
        lf_player_params = load_player_params(params_lf_all, fielder_names["LF"])
        cf_player_params = load_player_params(params_cf_all, fielder_names["CF"])
        rf_player_params = load_player_params(params_rf_all, fielder_names["RF"])
        print("  - 所有 Scaler 和球員模型參數載入成功。")
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"❌ [錯誤] 載入模型或 Scaler 或提取參數失敗: {e}")
        return

    # 3. 獲取約束條件列表，並移除邊界 (Bounds)
    constraints = get_constraints()
    initial_guess = np.array([-150, 220, 0, 250, 150, 220]) # 初始猜測點

    # 4. 執行最佳化，使用 SLSQP 方法和約束
    print("\n  - 開始執行 6 維團隊最佳化 (使用 SLSQP)...")
    start_time = time.time()
    result = minimize(
        objective_function_team,
        x0=initial_guess,
        args=(batter_df , scaler_lf, scaler_cf, scaler_rf, lf_player_params, cf_player_params, rf_player_params), 
        method='SLSQP', # 指定使用 SLSQP 方法
        constraints=constraints, # 傳入約束條件
        options={'disp': True, 'maxiter': 200} # 增加最大迭代次數，顯示收斂過程
    )
    end_time = time.time()
    print(f"\n--- 總最佳化耗時: {end_time - start_time:.2f} 秒 ---")

    # 5. 輸出並儲存結果
    if result.success:
        optimal_pos_array = result.x
        # ✨ [新增] 檢查結果是否真的在約束內 (作為驗證)
        final_lf_r = np.sqrt(optimal_pos_array[0]**2 + optimal_pos_array[1]**2)
        final_lf_a = np.degrees(np.arctan2(optimal_pos_array[0], optimal_pos_array[1]))
        print(f"  - 驗證 LF: r={final_lf_r:.1f}, a={final_lf_a:.1f}°") # 依此類推驗證 CF, RF

        optimal_positions = {
            "LF": [optimal_pos_array[0], optimal_pos_array[1]],
            "CF": [optimal_pos_array[2], optimal_pos_array[3]],
            "RF": [optimal_pos_array[4], optimal_pos_array[5]]
        }

        print("\n🎉 [結論] 找到的最佳團隊防守佈陣如下：")
        for pos_code, position in optimal_positions.items():
            print(f"  - {pos_code} ({fielder_names[pos_code]}):  X = {position[0]:.2f}, Y = {position[1]:.2f}")
        
        # ... (儲存 JSON 的邏輯維持不變) ...
        output_dir = RESULTS_DIR / "optimizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        team_str = f"LF_{fielder_names['LF']}_CF_{fielder_names['CF']}_RF_{fielder_names['RF']}".replace(" ", "_").replace(",", "")
        batter_str = batter_name.replace(" ", "_").replace(",", "")
        output_filename = f"{batter_str}_vs_{team_str}_optimal.json"
        output_path = output_dir / output_filename
        with open(output_path, 'w') as f:
            json_compatible_positions = {k: [float(coord) for coord in v] for k, v in optimal_positions.items()}
            json.dump(json_compatible_positions, f, indent=4)
        print(f"\n💾 最佳站位已儲存至: {output_path}")

    else:
        print("❌ [錯誤] SLSQP 最佳化程序未能成功收斂。")
        print(f"  - 狀態: {result.status}")
        print(f"  - 訊息: {result.message}")
        # 有時即使未完全收斂，result.x 也是一個可用的近似解
        if hasattr(result, 'x'):
             print(f"  - (近似解): {result.x}")

    print("\n所有團隊最佳化任務已全部完成！")

if __name__ == "__main__":
    example_fielders = { "LF": "Profar, Jurickson", "CF": "Harris II, Michael", "RF": "Acuña Jr., Ronald" }
    run_team_optimization("Kwan, Steven", example_fielders)