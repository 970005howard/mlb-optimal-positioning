# 檔案位置: src/data/step_02_preprocess_batted_balls.py

import pandas as pd
from pathlib import Path
import glob

# 從 config 匯入專案路徑
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
# 從我們建立的「中央廚房」導入所有需要的工具函式
from src.utils.feature_engineering import (
    calculate_batted_ball_features, 
    add_fielder_features,
    convert_positioning_to_xy
)

def preprocess_position_data(position_code: str):
    """
    對指定守備位置的資料進行完整的預處理流程。
    """
    print(f"--- 開始預處理守備位置: {position_code} ---")

    # 1. 設定路徑
    input_dir = PROCESSED_DATA_DIR / f"{position_code}_original_data"
    output_dir = PROCESSED_DATA_DIR / f"{position_code}_modified_data"
    positioning_file = RAW_DATA_DIR / f"{position_code}_positioning.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 讀取並預處理 positioning.csv，在迴圈外只執行一次
    if not positioning_file.exists():
        print(f"[警告] 找不到站位檔案: {positioning_file}。守備員相關特徵將為空。")
        df_pos_xy = pd.DataFrame() # 建立一個空的 DataFrame
    else:
        try:
            df_pos_original = pd.read_csv(positioning_file, encoding='utf-8')
            df_pos_xy = convert_positioning_to_xy(df_pos_original)
        except Exception as e:
            print(f"[錯誤] 讀取或處理站位檔案 {positioning_file} 失敗: {e}")
            return

    # 3. 獲取所有要處理的球員檔案列表
    file_list = glob.glob(str(input_dir / f"*_{position_code}.csv"))
    if not file_list:
        print(f"  - 在 {input_dir} 中沒有找到任何要處理的檔案。")
        return
        
    print(f"  - 找到 {len(file_list)} 個球員檔案，開始遍歷處理...")
    
    # 4. 遍歷並處理每一個球員檔案
    for file_path_str in file_list:
        file_path = Path(file_path_str)
        try:
            df_original = pd.read_csv(file_path, encoding='utf-8')
            
            # 流程第一步：呼叫共用函式，計算擊球本身的特徵
            df_batted_ball = calculate_batted_ball_features(df_original)
            
            # 流程第二步：呼叫專用函式，添加守備員相關特徵
            df_final = add_fielder_features(df_batted_ball, df_pos_xy)
            
            # 儲存最終的完整結果
            output_filename = file_path.name.replace(f"_{position_code}.csv", f"_{position_code}_with_all.csv")
            output_path = output_dir / output_filename
            df_final.to_csv(output_path, index=False, encoding='utf-8')

        except Exception as e:
            print(f"  ❗️ [錯誤] 處理檔案 {file_path.name} 時發生未知錯誤: {e}")
            continue

    print(f"--- {position_code} 預處理完成 ---\n")


def run_all_preprocessing():
    """
    這是 main.py 要呼叫的主函式，負責調度所有守備位置的處理。
    """
    positions_to_process = ["CF", "LF", "RF"]
    
    print("==========================================")
    print("開始執行所有資料預處理任務...")
    print(f"目標守備位置: {positions_to_process}")
    print("==========================================")

    for pos in positions_to_process:
        preprocess_position_data(pos)
        
    print("所有資料預處理任務已全部完成！")

# 讓這個腳本也可以被單獨執行，方便獨立測試
if __name__ == "__main__":
    run_all_preprocessing()