# 檔案位置: setup_project.py (更新版)
from pathlib import Path

# --- 1. 定義專案根目錄 ---
PROJECT_ROOT = Path(__file__).resolve().parent
print(f"專案根目錄設定在: {PROJECT_ROOT}")

# --- 2. 定義所有需要建立的資料夾路徑列表 ---
dirs_to_create = [
    # 數據資料夾 (分階段)
    PROJECT_ROOT / "data" / "01_raw",
    PROJECT_ROOT / "data" / "02_processed",
    PROJECT_ROOT / "data" / "03_inputs" / "batter_spray_charts",
    
    # 產出結果資料夾
    PROJECT_ROOT / "results" / "figures",       # 存放最終圖表 (.png)
    PROJECT_ROOT / "results" / "models",        # 存放訓練好的模型 (.nc)
    PROJECT_ROOT / "results" / "optimizations", # ✨ [更新] 新增，存放最佳化結果的 JSON 檔案
    
    # 原始碼資料夾 (按步驟和功能)
    PROJECT_ROOT / "src" / "data",          # Step 1 & 2: 資料處理
    PROJECT_ROOT / "src" / "modeling",      # Step 3: 模型訓練
    PROJECT_ROOT / "src" / "optimization",  # Step 4: 最佳化
    PROJECT_ROOT / "src" / "visualization", # ✨ [更新] 新增，Step 5: 視覺化
    PROJECT_ROOT / "src" / "evaluation",    # ✨ [更新] 新增，Step 6: 評估
    PROJECT_ROOT / "src" / "utils",         # 共用工具函式 (中央廚房)
    
    # 其他資料夾
    PROJECT_ROOT / "notebooks",             # (可選) 用於探索性分析
]

# --- 3. 遍歷列表並建立資料夾 ---
print("\n開始建立專案目錄結構...")
for dir_path in dirs_to_create:
    # 使用 .mkdir() 來建立資料夾
    # parents=True: 如果父目錄不存在，會一併建立。
    # exist_ok=True: 如果資料夾已經存在，不要拋出錯誤，直接跳過。
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"  - 已建立或確認存在: {dir_path}")

# --- 4. 建立 __init__.py 檔案，讓 src 下的資料夾成為 Python 套件 ---
init_files_to_create = [
    PROJECT_ROOT / "src" / "__init__.py",
    PROJECT_ROOT / "src" / "data" / "__init__.py",
    PROJECT_ROOT / "src" / "modeling" / "__init__.py",
    PROJECT_ROOT / "src" / "optimization" / "__init__.py",
    PROJECT_ROOT / "src" / "visualization" / "__init__.py", # ✨ [更新] 新增
    PROJECT_ROOT / "src" / "evaluation" / "__init__.py",    # ✨ [更新] 新增
    PROJECT_ROOT / "src" / "utils" / "__init__.py",
]

print("\n開始建立 __init__.py 檔案...")
for file_path in init_files_to_create:
    # 使用 .touch() 來建立一個空檔案
    file_path.touch(exist_ok=True)
    print(f"  - 已建立或確認存在: {file_path}")

print("\n專案結構初始化完成！")