# 檔案位置: src/evaluation/step_07_compare_initial_vs_optimal.py

import pandas as pd
import numpy as np
import json
from pathlib import Path
import joblib

# --- 導入我們在專案中已經建立好的工具 ---
from config import INPUTS_DATA_DIR, RESULTS_DIR, MODELS_DIR, RAW_DATA_DIR # 導入 RAW_DATA_DIR
from src.utils.feature_engineering import (
    calculate_batted_ball_features, convert_positioning_to_xy,
    COL_X_COORD, COL_Y_COORD, COL_FLIGHT_TIME, COL_PLAYER_NAME,
    COL_FIELDER_NAME, COL_FIELDER_X, COL_FIELDER_Y
)
from src.optimization.step_04_find_optimal_position import (
    load_model_scaler_and_params, load_player_params,
    predict_catch_probability_scaled
)

# --- 1. 輔助函式：載入球員的「初始」站位 ---
# (此函式維持不變)
def load_initial_positions(fielder_names: dict) -> dict:
    """
    從 positioning.csv 檔案中讀取指定球員的平均站位，並轉換為 XY 座標。
    """
    initial_positions = {}
    for pos_code, player_name in fielder_names.items():
        positioning_file = RAW_DATA_DIR / f"{pos_code}_positioning.csv"
        if not positioning_file.exists():
            raise FileNotFoundError(f"找不到初始站位檔案: {positioning_file}")
            
        df_pos_original = pd.read_csv(positioning_file, encoding='utf-8')
        df_pos_xy = convert_positioning_to_xy(df_pos_original)
        
        player_pos_data = df_pos_xy[df_pos_xy[COL_FIELDER_NAME] == player_name]
        
        if player_pos_data.empty:
            print(f"[警告] 在 {positioning_file.name} 中找不到球員 '{player_name}' 的初始站位。將使用該位置的平均站位。")
            avg_x = df_pos_xy[COL_FIELDER_X].mean()
            avg_y = df_pos_xy[COL_FIELDER_Y].mean()
            if pd.isna(avg_x) or pd.isna(avg_y):
                 avg_x, avg_y = {'LF': (-150, 220), 'CF': (0, 250), 'RF': (150, 220)}.get(pos_code, (0,0))
            initial_positions[pos_code] = [avg_x, avg_y]
        else:
            initial_positions[pos_code] = [
                player_pos_data.iloc[0][COL_FIELDER_X],
                player_pos_data.iloc[0][COL_FIELDER_Y]
            ]
    return initial_positions

# --- 2. 輔助函式：計算給定站位下的團隊表現 ---
# (此函式維持不變)
def calculate_team_performance(positions: dict, batter_df: pd.DataFrame, scalers: dict, player_params: dict, fielder_names: dict) -> tuple:
    """
    計算在給定站位下，團隊的總接殺分數和平均接殺機率。
    """
    lf_x, lf_y = positions['LF']
    cf_x, cf_y = positions['CF']
    rf_x, rf_y = positions['RF']

    ball_x = batter_df[COL_X_COORD].to_numpy()
    ball_y = batter_df[COL_Y_COORD].to_numpy()
    flight_time_raw = batter_df[COL_FLIGHT_TIME].to_numpy()

    # 計算原始距離
    dist_lf_raw = np.sqrt((ball_x - lf_x)**2 + (ball_y - lf_y)**2)
    dist_cf_raw = np.sqrt((ball_x - cf_x)**2 + (ball_y - cf_y)**2)
    dist_rf_raw = np.sqrt((ball_x - rf_x)**2 + (ball_y - rf_y)**2)
    
    # 應用標準化
    features_lf = np.stack([dist_lf_raw, flight_time_raw], axis=1)
    features_cf = np.stack([dist_cf_raw, flight_time_raw], axis=1)
    features_rf = np.stack([dist_rf_raw, flight_time_raw], axis=1)
    
    scaled_features_lf = scalers["LF"].transform(features_lf)
    scaled_features_cf = scalers["CF"].transform(features_cf)
    scaled_features_rf = scalers["RF"].transform(features_rf)

    dist_lf_scaled, time_lf_scaled = scaled_features_lf[:, 0], scaled_features_lf[:, 1]
    dist_cf_scaled, time_cf_scaled = scaled_features_cf[:, 0], scaled_features_cf[:, 1]
    dist_rf_scaled, time_rf_scaled = scaled_features_rf[:, 0], scaled_features_rf[:, 1]

    # 預測個人機率
    prob_lf = predict_catch_probability_scaled(dist_lf_scaled, time_lf_scaled, player_params["LF"])
    prob_cf = predict_catch_probability_scaled(dist_cf_scaled, time_cf_scaled, player_params["CF"])
    prob_rf = predict_catch_probability_scaled(dist_rf_scaled, time_rf_scaled, player_params["RF"])

    # 計算團隊機率
    prob_team_per_ball = 1 - (1 - prob_lf) * (1 - prob_cf) * (1 - prob_rf)
    
    total_score = np.sum(prob_team_per_ball)
    avg_prob = np.mean(prob_team_per_ball) * 100
    
    return total_score, avg_prob

# --- 3. 主流程函式 (返回一個結果字典) ---
def compare_initial_vs_optimal(batter_name: str, fielder_names: dict) -> dict:
    """
    比較初始站位和最佳站位下的團隊接殺表現。
    [修改] 此版本返回一個包含結果的字典，而不是列印它們。
    """
    print("=== 開始比較初始站位 vs. 最佳站位的團隊表現 ===")
    print(f"打者: {batter_name}")
    print(f"團隊: LF={fielder_names['LF']}, CF={fielder_names['CF']}, RF={fielder_names['RF']}")

    # ✨ [新增] 初始化結果字典
    results = {
        "batter": batter_name,
        "fielders": fielder_names,
        "num_batted_balls": 0,
        "initial": {},
        "optimal": {},
        "summary": {}
    }
    
    # --- 步驟 A: 載入所有必要的數據 ---
    print("\n--- 步驟 A: 載入資料 ---")
    try:
        # 1. 載入打者原始數據
        batter_file = INPUTS_DATA_DIR / "batter_spray_charts" / f"{batter_name}.csv"
        batter_df_raw = pd.read_csv(batter_file, encoding='utf-8')
        
        # 2. 載入「最佳」站位座標
        team_str = f"LF_{fielder_names['LF']}_CF_{fielder_names['CF']}_RF_{fielder_names['RF']}".replace(" ", "_").replace(",", "")
        batter_str = batter_name.replace(" ", "_").replace(",", "")
        positions_filename = f"{batter_str}_vs_{team_str}_optimal.json"
        positions_file = RESULTS_DIR / "optimizations" / positions_filename
        with open(positions_file, 'r') as f:
            optimal_positions = json.load(f)
            
        # 3. 載入「初始」站位座標
        initial_positions = load_initial_positions(fielder_names)

        # 4. 載入 Scalers 和 個人化模型參數
        scalers = {}
        player_params = {}
        for pos_code in ["LF", "CF", "RF"]:
            scalers[pos_code], params_all = load_model_scaler_and_params(pos_code)
            player_params[pos_code] = load_player_params(params_all, fielder_names[pos_code])

        print("  - 所有必要資料載入成功。")
        
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"❌ [錯誤] 載入資料失敗: {e}")
        print("請確認您已成功執行了對應的 `--optimize` 指令，並且 positioning.csv 檔案存在且包含指定球員。")
        return None # 發生錯誤時返回 None

    # --- 步驟 B: 預處理打者數據 ---
    print("\n--- 步驟 B: 處理擊球特徵 ---")
    batter_df_processed = calculate_batted_ball_features(batter_df_raw)
    batter_df = batter_df_processed.dropna(subset=[COL_X_COORD, COL_Y_COORD, COL_FLIGHT_TIME])
    num_batted_balls = len(batter_df)
    results["num_batted_balls"] = num_batted_balls # ✨ [新增] 儲存擊球總數
    print(f"  - 處理完成，共 {num_batted_balls} 筆有效擊球數據。")

    # --- 步驟 C: 計算兩種情境下的表現 ---
    print("\n--- 步驟 C: 計算表現指標 ---")
    # 1. 計算初始站位下的表現
    initial_score, initial_avg_prob = calculate_team_performance(
        initial_positions, batter_df, scalers, player_params, fielder_names
    )
    # 儲存初始站位結果
    results["initial"] = {
        "positions": initial_positions,
        "score": initial_score,
        "avg_prob": initial_avg_prob
    }
    print("  - 初始站位表現計算完成。")
    
    # 2. 計算最佳站位下的表現
    optimal_score, optimal_avg_prob = calculate_team_performance(
        optimal_positions, batter_df, scalers, player_params, fielder_names
    )
    # 儲存最佳站位結果
    results["optimal"] = {
        "positions": optimal_positions,
        "score": optimal_score,
        "avg_prob": optimal_avg_prob
    }
    print("  - 最佳站位表現計算完成。")

    # --- 步驟 D: 匯總結果 ---
    print("\n--- 步驟 D: 匯總效益 ---")
    score_diff = optimal_score - initial_score
    prob_diff = optimal_avg_prob - initial_avg_prob
    
    # 儲存總結
    results["summary"] = {
        "score_diff": score_diff,
        "prob_diff": prob_diff
    }

    # 移除所有原本在終端機顯示結果的 print() 敘述
    
    print("=======================================================")
    print(f"      針對打者 [{batter_name}] 的效益評估計算完成      ")
    print("=======================================================")
    
    # 返回完整的結果字典
    return results


if __name__ == "__main__":
    # 範例：請替換為您想分析的組合
    example_batter = "Kwan, Steven"
    example_fielders = {
        "LF": "Profar, Jurickson",
        "CF": "Harris II, Michael",
        "RF": "Acuña Jr., Ronald"
    }
    
    # 測試新的返回格式
    analysis_results = compare_initial_vs_optimal(example_batter, example_fielders)
    
    if analysis_results:
        print("\n--- 函式返回結果 (測試) ---")
        import json
        print(json.dumps(analysis_results, indent=2))