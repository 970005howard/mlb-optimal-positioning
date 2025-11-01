# 檔案位置: dashboard.py

import streamlit as st
from pathlib import Path
import pandas as pd  # <-- 新增
import sys           # <-- 新增

# --- 關鍵設定：將專案根目錄加入 Python 路徑 ---
# 這能確保 streamlit 能找到 'src' 和 'utils' 資料夾
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# ---------------------------------------------

# 導入您的主函式
try:
    from src.optimization.step_04_find_optimal_position import run_team_optimization
    from src.visualization.step_05_visualize_alignment import visualize_team_alignment
    from src.evaluation.step_07_compare_initial_vs_optimal import compare_initial_vs_optimal
    # 導入輔助工具
    from src.utils.dashboard_utils import get_player_lists
except ImportError as e:
    st.error(f"**啟動失敗**：找不到必要的 'src' 或 'utils' 模組。\n錯誤: {e}")
    st.warning("請確認您的 `dashboard.py` 檔案是放在專案的根目錄中 (與 `src` 和 `data` 資料夾在同一層)。")
    st.stop() # 停止執行

# --- 1. 頁面配置 & 標題 ---
st.set_page_config(layout="wide") # 讓介面使用寬螢幕
st.title("MLB 外野手防守站位最佳化分析")

# --- 2. 側邊欄 (Sidebar) 用於放置控制項 ---
st.sidebar.header("分析參數選擇")

# 載入球員列表
try:
    # 假設 get_player_lists() 函式知道如何從
    # 'data/03_inputs/batter_spray_charts/' 或 'data/raw' 獲取列表
    batters, lfs, cfs, rfs = get_player_lists()
except FileNotFoundError as e:
    st.sidebar.error(f"載入球員列表失敗: {e}")
    st.sidebar.warning("請確認 'data/raw' 或 'data/03_inputs' 資料夾中已包含必要的球員資料檔案。")
    st.stop()
except Exception as e:
    st.sidebar.error(f"載入球員列表時發生錯誤: {e}")
    st.stop()


# 建立下拉選單
selected_batter = st.sidebar.selectbox("選擇打者:", [""] + batters)
selected_lf = st.sidebar.selectbox("選擇左外野手 (LF):", [""] + lfs)
selected_cf = st.sidebar.selectbox("選擇中外野手 (CF):", [""] + cfs)
selected_rf = st.sidebar.selectbox("選擇右外野手 (RF):", [""] + rfs)

# 執行按鈕
run_button = st.sidebar.button("執行分析")

# --- 3. 主頁面 (用於顯示結果) ---
if run_button:
    # 檢查是否所有選項都已選擇
    if not all([selected_batter, selected_lf, selected_cf, selected_rf]):
        st.error("請選擇一位打者和三位外野手。")
    else:
        fielder_names = {
            "LF": selected_lf,
            "CF": selected_cf,
            "RF": selected_rf
        }
        
        # 【新功能】為實際統計數據準備變數
        actual_catches = "N/A"
        total_balls = "N/A"
        stats_error_message = ""
        
        # 使用 st.spinner 顯示載入動畫
        with st.spinner("正在執行分析... (這可能需要 1-2 分鐘)"):
            try:
                # --- 【新功能】載入並計算實際統計數據 ---
                # **【路徑已更新】** # 使用您提供的 'data/03_inputs/...' 路徑
                batted_ball_path = PROJECT_ROOT / "data" / "03_inputs" / "batter_spray_charts" / f"{selected_batter}.csv"
                
                if not batted_ball_path.exists():
                    # **【路徑已更新】** # 更新錯誤訊息中的路徑
                    stats_error_message = f"找不到實際統計檔案: data/03_inputs/batter_spray_charts/{selected_batter}.csv"
                else:
                    batted_ball_df = pd.read_csv(batted_ball_path)
                    if 'events' not in batted_ball_df.columns:
                        stats_error_message = f"檔案 {selected_batter}.csv 中找不到 'events' 欄位。"
                    else:
                        # 您的邏輯: 'field_out' 或 'field_error' 都算接殺
                        catch_events = ['field_out', 'field_error']
                        actual_catches = batted_ball_df[batted_ball_df['events'].isin(catch_events)].shape[0]
                        total_balls = batted_ball_df.shape[0]
                # --- 【新功能】結束 ---

                
                # 1. 執行最佳化 (Step 4) - 確保 .json 檔案存在
                # 附註: 這會依賴 step_03 的輸出
                run_team_optimization(selected_batter, fielder_names)
                
                # 2. 執行效益比較 (Step 7) - 獲取數據字典
                # 附註: 這會依賴 step_02 和 step_06 的輸出
                results_data = compare_initial_vs_optimal(selected_batter, fielder_names)
                
                # 3. 執行視覺化 (Step 5) - 獲取圖表物件
                # 附註: 這會依賴 step_04 的輸出 (.json)
                fig = visualize_team_alignment(selected_batter, fielder_names)
                
                st.success("分析完成！")

                # --- 4. 顯示結果 (使用雙欄位佈局) ---
                col1, col2 = st.columns([1, 2]) # 建立兩個欄位，右邊是左邊的 2 倍寬

                # --- 在左側欄位 (col1) 顯示 Step 07 的結果 ---
                with col1:
                    
                    # --- 【新功能】顯示實際統計數據 ---
                    st.header("實際擊球統計")
                    if stats_error_message:
                        st.warning(stats_error_message)
                        st.info(f"提醒：請確認 {selected_batter}.csv 檔案存在於 'data/03_inputs/batter_spray_charts/' 資料夾中。")
                    else:
                        st.metric(label="實際接殺球數 (Actual Catches)", value=f"{actual_catches} 球")
                        st.metric(label="總擊球數 (Total Batted Balls)", value=f"{total_balls} 球")
                    # --- 【新功能】結束 ---

                    st.header("調整站位後評估")
                    
                    st.subheader("初始站位:")
                    for pos, coords in results_data["initial"]["positions"].items():
                        st.text(f"{pos}: (X={coords[0]:.2f}, Y={coords[1]:.2f})")
                    st.text(f"預期總接殺: {results_data['initial']['score']:.2f} / {results_data['num_batted_balls']} 球")
                    st.metric(label="平均團隊接殺機率", value=f"{results_data['initial']['avg_prob']:.2f}%")

                    st.subheader("最佳化站位:")
                    for pos, coords in results_data["optimal"]["positions"].items():
                        st.text(f"{pos}: (X={coords[0]:.2f}, Y={coords[1]:.2f})")
                    st.text(f"預期總接殺: {results_data['optimal']['score']:.2f} / {results_data['num_batted_balls']} 球")
                    st.metric(label="平均團隊接殺機率", value=f"{results_data['optimal']['avg_prob']:.2f}%")

                    st.subheader("總結:")
                    st.metric(label="預期額外增加的出局數", value=f"{results_data['summary']['score_diff']:.2f}")
                    st.metric(label="平均團隊接殺機率提升", value=f"{results_data['summary']['prob_diff']:.2f}%")

                # --- 在右側欄位 (col2) 顯示 Step 05 的圖表 ---
                with col2:
                    st.pyplot(fig) # 使用 st.pyplot() 來顯示 Matplotlib 圖表

            except FileNotFoundError as e:
                 st.error(f"分析過程中發生錯誤：找不到檔案。 {e}")
                 st.warning("請確認您已為此打者和外野手**完整**執行過 `main.py` 的**所有前置處理步驟** (step_00 到 step_03)，"
                            "以確保 'data/processed' 和 'models' 資料夾中有所需的檔案。")
            except Exception as e:
                st.error(f"分析過程中發生錯誤: {e}")