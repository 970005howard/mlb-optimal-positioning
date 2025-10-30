# 檔案位置: src/data/step_00_split_batter_data.py
# (✨ 已修改為可處理多個輸入檔案)

import sys
import pandas as pd
from pathlib import Path

# 將專案根目錄加到 Python 的搜尋路徑中
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# 從 config 匯入我們需要的路徑
from config import RAW_DATA_DIR, INPUTS_DATA_DIR

# --- 1. 設定 ---

# ✨ [修改] 將 INPUT_FILE 改為 INPUT_FILES 列表
# ⚠️ 請將這 3 個檔案都放到 data/raw/ 資料夾中
INPUT_FILES = [
    RAW_DATA_DIR / "batter_67_data.csv",
    RAW_DATA_DIR / "batter_89_data.csv",
    RAW_DATA_DIR / "batter_345_data.csv"
]

# ⚠️ 請確認：這是您總表中代表「打者姓名」的欄位名稱
# (在 Statcast 資料中，這通常是 "player_name"，請您再次確認)
BATTER_COL_NAME = "player_name"

# 輸出位置 (此路徑是 step_04, step_05 等腳本預期的)
OUTPUT_DIR = INPUTS_DATA_DIR / "batter_spray_charts"

# --- 2. 主程式 ---
def split_batter_data():
    """
    讀取多個包含打者資料的總表，將它們合併，
    然後按打者姓名分割成多個獨立的 CSV 檔案。
    """
    
    # 確保輸出資料夾存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df_list = [] # 用來存放讀取到的
    
    print("--- 開始讀取多個打者資料檔案 ---")
    
    # ✨ [修改] 遍歷所有輸入檔案
    for file_path in INPUT_FILES:
        if not file_path.exists():
            print(f"⚠️ [警告] 找不到輸入檔案: {file_path}，已跳過。")
            continue
            
        print(f"  - 正在讀取: {file_path.name}")
        try:
            temp_df = pd.read_csv(file_path, encoding='utf-8')
            df_list.append(temp_df)
        except Exception as e:
            print(f"  - ❌ [錯誤] 讀取檔案 {file_path.name} 失敗: {e}")
    
    if not df_list:
        print("❌ [錯誤] 沒有成功讀取到任何資料檔案。請檢查 INPUT_FILES 設定。")
        return

    # ✨ [修改] 將所有讀取到的 DataFrame 合併成一個
    print("  - 正在合併所有資料...")
    df = pd.concat(df_list, ignore_index=True)
    print(f"  - 所有檔案合併完成，共 {len(df)} 筆擊球資料。")

    # --- (以下邏輯與之前版本相同) ---
        
    if BATTER_COL_NAME not in df.columns:
        print(f"❌ [錯誤] 在合併後的資料中找不到指定的打者欄位: '{BATTER_COL_NAME}'")
        print(f"  > 檔案中包含的欄位有: {df.columns.tolist()}")
        return

    print(f"\n開始按 '{BATTER_COL_NAME}' 分割檔案並儲存至: {OUTPUT_DIR}")
    
    grouped = df.groupby(BATTER_COL_NAME, dropna=False)
    total_files = len(grouped)
    print(f"  - 偵測到 {total_files} 位獨立的打者。")
    
    # 遍歷所有打者
    for i, (player_name, g) in enumerate(grouped):
        
        # 處理空值或空字串
        if pd.isna(player_name) or str(player_name).strip() == "":
            print(f"  - 警告: (進度 {i+1}/{total_files}) 發現空的打者姓名，已略過。")
            continue
        
        # 確保檔名安全
        safe_name = str(player_name).replace("/", "_").replace("\\", "_")
        out_path = OUTPUT_DIR / f"{safe_name}.csv"
        
        try:
            # 儲存該打者的所有擊球資料
            g.to_csv(out_path, index=False, encoding='utf-8')
            
            # 每 100 個檔案報告一次進度
            if (i+1) % 100 == 0 or (i+1) == total_files:
                print(f"  - (進度 {i+1}/{total_files}) 已儲存: {out_path.name}")
                
        except Exception as e:
            print(f"  - ❗️ (進度 {i+1}/{total_files}) 儲存 '{safe_name}.csv' 失敗: {e}")

    print(f"\n--- ✅ 所有打者資料分割完成，共處理 {total_files} 位球員 ---")

# 讓這個腳本可以直接被執行
if __name__ == "__main__":
    split_batter_data() 