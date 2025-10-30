# 檔案位置: src/utils/feature_engineering.py

import pandas as pd
import numpy as np

# --- 1. 常數定義區 ---
# 統一管理所有欄位名稱和常數，方便所有模組共用
# 輸入欄位 (from original_data)
COL_EVENTS = "events"
COL_HC_X = "hc_x"
COL_HC_Y = "hc_y"
COL_HIT_DISTANCE = "hit_distance_sc"
COL_PLAYER_NAME = "player_name"
COL_LAUNCH_SPEED = "launch_speed"
COL_LAUNCH_ANGLE = "launch_angle"

# 輸入欄位 (from positioning.csv)
COL_FIELDER_NAME = "name_fielder"
COL_AVG_DIST = "avg_norm_start_distance"
COL_AVG_ANGLE = "avg_norm_start_angle"

# 輸出欄位
COL_CAUGHT = "caught"
COL_X_COORD = "x_coord" # 球的落點 X
COL_Y_COORD = "y_coord" # 球的落點 Y
COL_FLIGHT_TIME = "flight_time_s"
COL_FIELDER_X = "fielder_x" # 守備員的站位 X
COL_FIELDER_Y = "fielder_y" # 守備員的站位 Y
COL_FIELDER_DIST = "fielder_distance_to_ball"

# 物理座標常數
X0, Y0 = 125.42, 198.27

# --- 2. 輔助函式區 ---
def compute_flight_time(launch_speed_mph, launch_angle_deg, g=32.174, h0=3.0, h1=0.0):
    """根據擊出速度和仰角計算球的飛行時間。"""
    MPH_TO_FPS = 1.4666666667
    theta = np.radians(launch_angle_deg)
    v0_fps = launch_speed_mph * MPH_TO_FPS
    vy0_fps = v0_fps * np.sin(theta)
    under_sqrt = vy0_fps**2 + 2 * g * (h0 - h1)
    with np.errstate(invalid="ignore"):
        T = (vy0_fps + np.sqrt(under_sqrt)) / g
    T = np.where(np.isfinite(T) & (T >= 0), T, np.nan)
    return T

def convert_positioning_to_xy(df_pos: pd.DataFrame) -> pd.DataFrame:
    """
    根據使用者定義的座標系（0度朝向中外野），
    將守備員站位從極座標轉為直角座標。
    """
    df_out = df_pos.copy()
    
    # 抓取距離 (r) 和 原始角度 (θ_original)
    r = df_out[COL_AVG_DIST].astype(float)
    angle_original_deg = df_out[COL_AVG_ANGLE].astype(float)
    angle_original_rad = np.radians(angle_original_deg)
    
    # 目標 X 軸 (本壘右側) = r * sin(原始角度)
    # 目標 Y 軸 (中外野)   = r * cos(原始角度)
    df_out[COL_FIELDER_X] = r * np.sin(angle_original_rad)
    df_out[COL_FIELDER_Y] = r * np.cos(angle_original_rad)
    
    return df_out

# --- 3. 核心特徵計算函式 (已拆分) ---

def calculate_batted_ball_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    【共用函式】接收原始擊球數據，只計算與「球本身」相關的特徵。
    這個函式會被 step_02 和 step_04 共同使用。
    """
    df_out = df.copy()

    # 計算 'caught' 欄位
    catch_conditions = ["field_out", "sac_fly", "double_play", "triple_play"]
    if COL_EVENTS in df_out.columns:
        df_out[COL_CAUGHT] = df_out[COL_EVENTS].isin(catch_conditions).astype(int)
    else:
        df_out[COL_CAUGHT] = np.nan

    # 計算球的落點座標
    required_coord_cols = [COL_HC_X, COL_HC_Y, COL_HIT_DISTANCE]
    if all(c in df_out.columns for c in required_coord_cols):
        hc_x = df_out[COL_HC_X].astype(float)
        hc_y = df_out[COL_HC_Y].astype(float)
        r = df_out[COL_HIT_DISTANCE].astype(float)
        
        phi_deg = np.degrees(np.arctan2((hc_x - X0), (Y0 - hc_y)))
        phi_rad = np.radians(phi_deg)

        df_out[COL_X_COORD] = r * np.sin(phi_rad)
        df_out[COL_Y_COORD] = r * np.cos(phi_rad)
    else:
        df_out[[COL_X_COORD, COL_Y_COORD]] = np.nan

    # 計算飛行時間
    required_flight_cols = [COL_LAUNCH_SPEED, COL_LAUNCH_ANGLE]
    if all(c in df_out.columns for c in required_flight_cols):
        df_out[COL_FLIGHT_TIME] = compute_flight_time(
            df_out[COL_LAUNCH_SPEED].astype(float).to_numpy(),
            df_out[COL_LAUNCH_ANGLE].astype(float).to_numpy()
        )
    else:
        df_out[COL_FLIGHT_TIME] = np.nan
        
    return df_out

def add_fielder_features(df_batted_ball: pd.DataFrame, df_pos_xy: pd.DataFrame) -> pd.DataFrame:
    """
    【step_02 專用函式】接收已處理好的擊球數據，
    合併守備員站位，並計算相關距離。
    """
    # 合併守備員的站位座標
    if not df_pos_xy.empty:
        # 使用 pd.merge 將站位資料加到每一筆擊球數據上
        df_merged = pd.merge(df_batted_ball, df_pos_xy[[COL_FIELDER_NAME, COL_FIELDER_X, COL_FIELDER_Y]], 
                             left_on=COL_PLAYER_NAME, right_on=COL_FIELDER_NAME, how='left')
        # 合併後移除多餘的 'name_fielder' 欄位
        if COL_FIELDER_NAME in df_merged.columns:
            df_merged = df_merged.drop(columns=[COL_FIELDER_NAME])
    else:
        # 如果站位資料不存在，則手動新增空欄位以保持結構一致
        df_merged = df_batted_ball.copy()
        df_merged[[COL_FIELDER_X, COL_FIELDER_Y]] = np.nan

    # 計算守備員到球落點的距離
    dx = df_merged[COL_X_COORD] - df_merged[COL_FIELDER_X]
    dy = df_merged[COL_Y_COORD] - df_merged[COL_FIELDER_Y]
    df_merged[COL_FIELDER_DIST] = np.sqrt(dx**2 + dy**2)
    
    return df_merged