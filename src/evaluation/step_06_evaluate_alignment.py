# 檔案位置: src/evaluation/step_06_evaluate_alignment.py

import pandas as pd
import numpy as np
import json
from pathlib import Path
import joblib # ✨ [新增] 用於載入 scaler 物件

# --- 導入我們在專案中已經建立好的工具 ---
from config import INPUTS_DATA_DIR, RESULTS_DIR, MODELS_DIR # ✨ [新增] 導入 MODELS_DIR
from src.utils.feature_engineering import calculate_batted_ball_features, COL_X_COORD, COL_Y_COORD, COL_FLIGHT_TIME, COL_FIELDER_DIST # 確保導入所需常數
from src.optimization.step_04_find_optimal_position import load_player_params, predict_catch_probability_scaled # ✨ [修改] 導入縮放版的預測函式

def evaluate_team_alignment(batter_name: str, fielder_names: dict):
    """
    主執行函式，計算在最佳站位下，指定團隊對指定打者的接殺機率，
    並在預測前應用標準化。
    """
    print("=== 開始評估最佳站位的團隊接殺機率 ===")
    
    # --- 步驟 A: 載入所有必要的數據 ---
    print("\n--- 步驟 A: 載入資料 ---")
    try:
        # 1. 載入打者原始數據
        batter_file = INPUTS_DATA_DIR / "batter_spray_charts" / f"{batter_name}.csv"
        batter_df_raw = pd.read_csv(batter_file, encoding='utf-8')
        
        # 2. 載入這個情境對應的最佳站位座標
        team_str = f"LF_{fielder_names['LF']}_CF_{fielder_names['CF']}_RF_{fielder_names['RF']}".replace(" ", "_").replace(",", "")
        batter_str = batter_name.replace(" ", "_").replace(",", "")
        positions_filename = f"{batter_str}_vs_{team_str}_optimal.json"
        positions_file = RESULTS_DIR / "optimizations" / positions_filename
        with open(positions_file, 'r') as f:
            optimal_positions = json.load(f)
            
        # 3. ✨ [修改] 載入三位外野手各自的 Scaler 和 個人化模型參數
        scalers = {}
        player_params = {}
        for pos_code in ["LF", "CF", "RF"]:
            model_dir = MODELS_DIR / pos_code
            scaler_path = model_dir / f"{pos_code}_scaler.joblib"
            
            if not scaler_path.exists():
                 raise FileNotFoundError(f"找不到 {pos_code} 的 Scaler 檔案: {scaler_path}")
            scalers[pos_code] = joblib.load(scaler_path)
            
            # 假設 load_player_params 能正確處理參數字典
            # 我們需要先載入完整的參數字典
            from src.optimization.step_04_find_optimal_position import load_model_scaler_and_params 
            _, params_all = load_model_scaler_and_params(pos_code) 
            player_params[pos_code] = load_player_params(params_all, fielder_names[pos_code])

        print("  - 打者數據、最佳站位、Scaler、球員模型參數均已成功載入。")
        
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"❌ [錯誤] 載入資料失敗: {e}")
        return

    # --- 步驟 B: 預處理打者數據 ---
    print("\n--- 步驟 B: 處理擊球特徵 ---")
    batter_df_processed = calculate_batted_ball_features(batter_df_raw)
    # 我們需要保留原始的距離和時間，因為縮放必須在計算完距離後進行
    required_cols_eval = [COL_X_COORD, COL_Y_COORD, COL_FLIGHT_TIME]
    batter_df = batter_df_processed.dropna(subset=required_cols_eval).copy()
    print(f"  - 處理完成，共 {len(batter_df)} 筆有效擊球數據。")

    # --- 步驟 C: 在最佳站位下，重新計算所有機率 (包含縮放) ---
    print("\n--- 步驟 C: 計算接殺機率 ---")
    lf_x, lf_y = optimal_positions['LF']
    cf_x, cf_y = optimal_positions['CF']
    rf_x, rf_y = optimal_positions['RF']

    ball_x = batter_df[COL_X_COORD].to_numpy()
    ball_y = batter_df[COL_Y_COORD].to_numpy()
    flight_time_raw = batter_df[COL_FLIGHT_TIME].to_numpy()

    # 計算「原始」距離
    dist_lf_raw = np.sqrt((ball_x - lf_x)**2 + (ball_y - lf_y)**2)
    dist_cf_raw = np.sqrt((ball_x - cf_x)**2 + (ball_y - cf_y)**2)
    dist_rf_raw = np.sqrt((ball_x - rf_x)**2 + (ball_y - rf_y)**2)
    
    # ✨ [核心修改] 使用載入的 Scalers 對數據進行標準化
    features_lf = np.stack([dist_lf_raw, flight_time_raw], axis=1)
    features_cf = np.stack([dist_cf_raw, flight_time_raw], axis=1)
    features_rf = np.stack([dist_rf_raw, flight_time_raw], axis=1)
    
    scaled_features_lf = scalers["LF"].transform(features_lf)
    scaled_features_cf = scalers["CF"].transform(features_cf)
    scaled_features_rf = scalers["RF"].transform(features_rf)

    dist_lf_scaled, time_lf_scaled = scaled_features_lf[:, 0], scaled_features_lf[:, 1]
    dist_cf_scaled, time_cf_scaled = scaled_features_cf[:, 0], scaled_features_cf[:, 1]
    dist_rf_scaled, time_rf_scaled = scaled_features_rf[:, 0], scaled_features_rf[:, 1]

    # 使用「縮放後」的數據進行預測
    prob_lf = predict_catch_probability_scaled(dist_lf_scaled, time_lf_scaled, player_params["LF"])
    prob_cf = predict_catch_probability_scaled(dist_cf_scaled, time_cf_scaled, player_params["CF"])
    prob_rf = predict_catch_probability_scaled(dist_rf_scaled, time_rf_scaled, player_params["RF"])

    # 計算團隊機率
    prob_team_per_ball = 1 - (1 - prob_lf) * (1 - prob_cf) * (1 - prob_rf)
    
    print("  - 所有擊球的個人及團隊接殺機率計算完成。")

    # --- 步驟 D: 匯總並呈現結果 ---
    # ... (這部分邏輯完全不變) ...
    print("\n==================================================")
    print(f"      針對打者 [{batter_name}] 的最佳化防守佈陣評估      ")
    print("==================================================")
    print("最佳站位:")
    for pos, coords in optimal_positions.items():
        print(f"  - {pos} ({fielder_names[pos]}): (X={coords[0]:.2f}, Y={coords[1]:.2f})")
    print("\n預期表現:")
    total_team_catch_score = np.sum(prob_team_per_ball)
    average_team_catch_prob = np.mean(prob_team_per_ball) * 100
    print(f"  - 團隊總接殺分數 (期望出局數): {total_team_catch_score:.2f} / {len(batter_df)} 顆球")
    print(f"  - 平均團隊接殺機率: {average_team_catch_prob:.2f}%")
    print("\n個人貢獻預期 (期望出局數):")
    print(f"  - LF ({fielder_names['LF']}): {np.sum(prob_lf):.2f}")
    print(f"  - CF ({fielder_names['CF']}): {np.sum(prob_cf):.2f}")
    print(f"  - RF ({fielder_names['RF']}): {np.sum(prob_rf):.2f}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    example_batter = "Kwan, Steven"
    example_fielders = { "LF": "Profar, Jurickson", "CF": "Harris II, Michael", "RF": "Acuña Jr., Ronald" }
    evaluate_team_alignment(example_batter, example_fielders)