# 檔案位置: src/data/01_split_player_data.py

import sys
from pathlib import Path
import pandas as pd

# 將專案根目錄加到 Python 的搜尋路徑中，以便匯入 config
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# 從 config 匯入我們需要的路徑
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def split_data_for_position(position_code: str):
    """
    根據指定的守備位置代碼 (例如 "CF", "LF")，讀取對應的原始資料，
    並將其按球員姓名分割成多個 CSV 檔案。

    Args:
        position_code (str): 守備位置的縮寫，例如 "CF", "LF", "RF"。
    """
    print(f"--- 開始處理守備位置: {position_code} ---")
    
    # 1. 動態建立輸入和輸出路徑
    # 使用 f-string 來根據 position_code 組合檔名和資料夾名稱
    input_file = RAW_DATA_DIR / f"{position_code}_data.csv"
    output_dir = PROCESSED_DATA_DIR / f"{position_code}_original_data"
    
    # 確保輸出資料夾存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 檢查輸入檔案是否存在
    if not input_file.exists():
        print(f"[錯誤] 找不到輸入檔案: {input_file}")
        return # 如果檔案不存在，就跳過這個位置

    # 2. 執行與之前完全相同的資料處理邏輯
    print(f"讀取資料: {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8')
    
    print(f"開始按球員姓名分割檔案並儲存至: {output_dir}")
    for player, g in df.groupby("player_name", dropna=False):
        if pd.isna(player) or player.strip() == "":
            print("  - 警告: 發現空的球員姓名，已略過。")
            continue
            
        # 清理檔名，避免特殊字元問題
        base = player.replace(" ", "_").replace(".", "")
        out_path = output_dir / f"{base}_{position_code}.csv"
        g.to_csv(out_path, index=False, encoding='utf-8')
        
    print(f"--- {position_code} 處理完成 ---\n")

def run_all_splits():
    """
    執行所有指定守備位置的資料分割流程。
    """
    # ✨ 未來的擴充點！只需要修改這個列表！ ✨
    positions_to_process = ["CF", "LF", "RF"]
    
    print("======================================")
    print("開始執行所有資料分割任務...")
    print(f"目標守備位置: {positions_to_process}")
    print("======================================")

    for pos in positions_to_process:
        split_data_for_position(pos)
        
    print("所有資料分割任務已全部完成！")

# 讓這個腳本可以直接被執行
if __name__ == "__main__":
    run_all_splits()