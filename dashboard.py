# 檔案位置: dashboard.py

import streamlit as st
from pathlib import Path

# 導入您的主函式
from src.optimization.step_04_find_optimal_position import run_team_optimization
from src.visualization.step_05_visualize_alignment import visualize_team_alignment
from src.evaluation.step_07_compare_initial_vs_optimal import compare_initial_vs_optimal
# 導入我們剛剛建立的輔助工具
from src.utils.dashboard_utils import get_player_lists

# --- 1. 頁面配置 & 標題 ---
st.set_page_config(layout="wide") # 讓介面使用寬螢幕
st.title("⚾ MLB 外野手防守站位最佳化分析")

# --- 2. 側邊欄 (Sidebar) 用於放置控制項 ---
st.sidebar.header("分析參數選擇")

# 載入球員列表
batters, lfs, cfs, rfs = get_player_lists()

# 建立下拉選單
selected_batter = st.sidebar.selectbox("選擇打者:", [""] + batters)
selected_lf = st.sidebar.selectbox("選擇左外野手 (LF):", [""] + lfs)
selected_cf = st.sidebar.selectbox("選擇中外野手 (CF):", [""] + cfs)
selected_rf = st.sidebar.selectbox("選擇右外野手 (RF):", [""] + rfs)

# 執行按鈕
run_button = st.sidebar.button("🚀 執行分析")

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
        
        # 使用 st.spinner 顯示載入動畫
        with st.spinner("正在執行分析... (這可能需要 1-2 分鐘)"):
            try:
                # 1. 執行最佳化 (Step 4) - 確保 .json 檔案存在
                run_team_optimization(selected_batter, fielder_names)
                
                # 2. 執行效益比較 (Step 7) - 獲取數據字典
                results_data = compare_initial_vs_optimal(selected_batter, fielder_names)
                
                # 3. 執行視覺化 (Step 5) - 獲取圖表物件
                fig = visualize_team_alignment(selected_batter, fielder_names)
                
                st.success("分析完成！")

                # --- 4. 顯示結果 (使用雙欄位佈局) ---
                col1, col2 = st.columns([1, 2]) # 建立兩個欄位，右邊是左邊的 2 倍寬

                # --- 在左側欄位 (col1) 顯示 Step 07 的結果 ---
                with col1:
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

            except Exception as e:
                st.error(f"分析過程中發生錯誤: {e}")