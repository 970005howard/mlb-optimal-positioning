import streamlit as st
import pandas as pd  # <-- 新增
from pathlib import Path # <-- 新增
import os              # <-- 新增
import sys             # <-- 新增

# --- 關鍵設定：將專案根目錄加入 Python 路徑 ---
# 這能確保 streamlit 能找到 'src' 資料夾
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# ---------------------------------------------

try:
    # 現在我們可以從 src 導入了
    from src.visualization.step_05_visualize_alignment import visualize_team_alignment
    # (我們將不再使用 step_00 的 get_available_batters，改用更動態的方式)
    # from src.data.step_00_split_batter_data import get_available_batters
except ImportError:
    st.error(
        "**啟動失敗**：找不到必要的 'src' 模組。\n"
        "請確認您的 `dashboard.py` 檔案是放在專案的根目錄中 "
        "(與 `src` 和 `data` 資料夾在同一層)。"
    )
    st.stop() # 停止執行

# --- 網頁標題 ---
st.set_page_config(page_title="MLB Optimal Positioning", layout="wide")
st.title("⚾ MLB 外野防守最佳化分析儀")


# (範例守備球員 - 之後您可以讓使用者自行選擇)
FIELDERS = {
    "LF": "Profar, Jurickson",
    "CF": "Harris II, Michael",
    "RF": "Acuña Jr., Ronald"
}

# --- 1. 準備球員選單 (從 'processed' 資料夾動態載入) ---
try:
    processed_data_dir = PROJECT_ROOT / "data" / "processed"
    
    # 掃描 'data/processed' 資料夾，找出所有已處理過的打者 .csv 檔案
    available_batters = [
        f.stem.replace('_batted_balls', '') 
        for f in processed_data_dir.glob('*_batted_balls.csv')
    ]
    
    if not available_batters:
        st.warning("在 'data/processed' 中找不到已處理的打者檔案 (*_batted_balls.csv)。")
        st.stop()
    
    available_batters.sort() # 排序

except Exception as e:
    st.error(f"讀取可用的打者列表時出錯：{e}")
    st.stop()


# --- 2. 建立 Streamlit 介面 ---
st.header("選擇分析對象")
selected_batter = st.selectbox(
    "選擇打者:",
    options=available_batters,
    index=0
)

if selected_batter:
    
    # --- ▼▼▼ 【新功能】顯示統計數據 ▼▼▼ ---
    
    st.subheader(f"📊 打者擊球統計 ({selected_batter})")
    
    # 1. 建立該打者的 .csv 檔案路徑
    batted_ball_path = PROJECT_ROOT / "data" / "processed" / f"{selected_batter}_batted_balls.csv"
    
    # 2. 讀取檔案並計算
    if not batted_ball_path.exists():
        st.error(f"找不到打者 {selected_batter} 的統計檔案。")
    else:
        try:
            batted_ball_df = pd.read_csv(batted_ball_path)
            
            # !! 請確認 'field_out' 是您資料中代表「接殺」的正確值 !!
            actual_catches = batted_ball_df[batted_ball_df['events'] == 'field_out'].shape[0]
            total_balls = batted_ball_df.shape[0]

            # 3. 使用 st.metric 顯示統計數字
            col1, col2 = st.columns(2)
            col1.metric(
                label="實際接殺球數 (Actual Catches)", 
                value=f"{actual_catches} 球"
            )
            col2.metric(
                label="總擊球數 (Total Batted Balls)", 
                value=f"{total_balls} 球"
            )
            
        except Exception as e:
            st.error(f"讀取統計檔案時出錯: {e}")

    # --- ▲▲▲ 【新功能】結束 ▲▲▲ ---
    
    
    # --- 3. 產生並顯示視覺化圖表 (這部分與您原本的程式碼相同) ---
    st.subheader(f"🛡️ 防守站位視覺化 ({selected_batter})")
    
    try:
        with st.spinner(f"正在為 {selected_batter} 產生防守站位熱力圖..."):
            
            # 呼叫 step_05 的函式來產生圖表
            fig = visualize_team_alignment(selected_batter, FIELDERS)
            
            # 顯示圖表
            st.pyplot(fig)
            
    except FileNotFoundError as e:
        st.error(f"**產生圖表失敗**：找不到必要的檔案。 {e}")
        st.warning(f"請確認您已經為 {selected_batter} 執行了最佳化流程 (step_04)，"
                   f"並且 {selected_batter}_optimal_positions.json 檔案已存在。")
    except Exception as e:
        st.error(f"產生圖表時發生未預期的錯誤: {e}")