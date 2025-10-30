# config.py
from pathlib import Path

# 專案根目錄
PROJECT_ROOT = Path(__file__).resolve().parent

# --- 資料夾路徑 ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "01_raw"
PROCESSED_DATA_DIR = DATA_DIR / "02_processed"
INPUTS_DATA_DIR = DATA_DIR / "03_inputs"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

SRC_DIR = PROJECT_ROOT / "src"

# --- 確保資料夾存在 ---
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- 檔案路徑範例 ---
CF_RAW_DATA_PATH = RAW_DATA_DIR / "CF_data.csv"
CF_POSITIONING_PATH = RAW_DATA_DIR / "CF_positioning.csv"