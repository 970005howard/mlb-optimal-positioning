# 檔案位置: src/visualization/step_05_visualize_alignment.py

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import matplotlib.transforms as transforms # 確保導入 transforms

# 從 config 匯入專案路徑
from config import INPUTS_DATA_DIR, RESULTS_DIR, FIGURES_DIR, RAW_DATA_DIR
# 從 utils 導入必要的函式和常數
from src.utils.feature_engineering import (
    calculate_batted_ball_features, convert_positioning_to_xy,
    COL_X_COORD, COL_Y_COORD, COL_FLIGHT_TIME,
    COL_FIELDER_NAME, COL_FIELDER_X, COL_FIELDER_Y, COL_PLAYER_NAME # 確保導入 COL_PLAYER_NAME
)

# --- 1. 繪圖輔助函式 ---
def draw_baseball_field_v2(ax):
    """繪製匹配風格的棒球場。"""
    infield_bases = np.array([(0, 0), (63.6, 63.6), (0, 127.3), (-63.6, 63.6), (0, 0)])
    ax.plot(infield_bases[:, 0], infield_bases[:, 1], color="black", lw=2)
    ax.plot([0, -250], [0, 250], color="black", lw=2)
    ax.plot([0, 250], [0, 250], color="black", lw=2)
    outfield_wall = patches.Arc((0, 0), 800, 800, theta1=45, theta2=135, linestyle='--', color="black", lw=2)
    ax.add_patch(outfield_wall)
    bases = np.array([(63.6, 63.6), (0, 127.3), (-63.6, 63.6)])
    ax.scatter(bases[:, 0], bases[:, 1], c='white', ec='black', s=100, zorder=5)

# --- 2. 載入初始站位的輔助函式 ---
def load_initial_positions(fielder_names: dict) -> dict:
    """從 positioning.csv 檔案中讀取指定球員的平均站位，並轉換為 XY 座標。"""
    initial_positions = {}
    for pos_code, player_name in fielder_names.items():
        positioning_file = RAW_DATA_DIR / f"{pos_code}_positioning.csv"
        if not positioning_file.exists():
            print(f"[警告] 找不到初始站位檔案: {positioning_file}，無法繪製初始 {pos_code} 位置。")
            initial_positions[pos_code] = [np.nan, np.nan]
            continue
            
        df_pos_original = pd.read_csv(positioning_file, encoding='utf-8')
        df_pos_xy = convert_positioning_to_xy(df_pos_original)
        
        player_pos_data = df_pos_xy[df_pos_xy[COL_FIELDER_NAME] == player_name]
        
        if player_pos_data.empty:
            print(f"[警告] 在 {positioning_file.name} 中找不到球員 '{player_name}' 的初始站位。將使用該位置的平均站位。")
            avg_x = df_pos_xy[COL_FIELDER_X].mean()
            avg_y = df_pos_xy[COL_FIELDER_Y].mean()
            if pd.isna(avg_x) or pd.isna(avg_y):
                 avg_x, avg_y = {'LF': (-150, 220), 'CF': (0, 250), 'RF': (150, 220)}.get(pos_code, (0,0))
            initial_positions[pos_code] = [avg_x, avg_y]
        else:
            initial_positions[pos_code] = [
                player_pos_data.iloc[0][COL_FIELDER_X],
                player_pos_data.iloc[0][COL_FIELDER_Y]
            ]
    return initial_positions

# --- 3. 主流程函式 ---
def visualize_team_alignment(batter_name: str, fielder_names: dict):
    """
    為指定的打者和外野手團隊，讀取結果並視覺化初始站位與最佳站位。
    """
    print("==========================================")
    print(f"開始為打者 [{batter_name}] 和指定團隊繪製佈陣對比圖...")
    print(f"  - LF: {fielder_names['LF']}, CF: {fielder_names['CF']}, RF: {fielder_names['RF']}")
    print("==========================================")

    # 1. 載入資料
    print("  - 正在載入資料...")
    try:
        team_str = f"LF_{fielder_names['LF']}_CF_{fielder_names['CF']}_RF_{fielder_names['RF']}".replace(" ", "_").replace(",", "")
        batter_str = batter_name.replace(" ", "_").replace(",", "")
        positions_filename = f"{batter_str}_vs_{team_str}_optimal.json"
        
        batter_file = INPUTS_DATA_DIR / "batter_spray_charts" / f"{batter_name}.csv"
        positions_file = RESULTS_DIR / "optimizations" / positions_filename
        
        if not batter_file.exists() or not positions_file.exists():
            print(f"❌ [錯誤] 缺少必要的輸入檔案 (打者數據或最佳站位 JSON)。請先執行優化步驟。")
            return

        batter_df_raw = pd.read_csv(batter_file, encoding='utf-8')
        with open(positions_file, 'r') as f:
            optimal_positions = json.load(f)
            
        initial_positions = load_initial_positions(fielder_names)
            
        print("  - 資料載入完成。")
        
    except Exception as e:
        print(f"❌ [錯誤] 載入資料時發生問題: {e}")
        return

    # 2. 預處理打者資料
    batter_df_processed = calculate_batted_ball_features(batter_df_raw)
    batter_df = batter_df_processed.dropna(subset=[COL_X_COORD, COL_Y_COORD, COL_FLIGHT_TIME])

    # 3. 設定繪圖視窗
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 9))
    draw_baseball_field_v2(ax)
    print("  - 棒球場繪製完成。")

    # 4. 繪製擊球密度圖 (KDE)
    sns.kdeplot(x=batter_df[COL_X_COORD], y=batter_df[COL_Y_COORD], fill=True, cmap="Blues", ax=ax, alpha=0.6, levels=8)
    print("  - 擊球密度圖繪製完成。")
    
    # ✨ [核心修正] 取消註解，並恢復原始散點樣式 (黑色, s=20, alpha=0.5)
    ax.scatter(batter_df[COL_X_COORD], batter_df[COL_Y_COORD], s=20, alpha=0.5, label='Batted Ball Locations', color='black', edgecolors='black', linewidths=0.5, zorder=3) # 設定 zorder 確保在 KDE 之上
    print("  - 擊球落點散點繪製完成。")

    # 5. 繪製兩種站位
    print("  - 正在繪製守備站位...")
    initial_pos_arr = np.array(list(initial_positions.values()))
    ax.scatter(initial_pos_arr[:, 0], initial_pos_arr[:, 1], c='blue', s=100, marker='o', label='Initial Positions', zorder=5, edgecolors='white')

    optimal_pos_arr = np.array(list(optimal_positions.values()))
    ax.scatter(optimal_pos_arr[:, 0], optimal_pos_arr[:, 1], c='red', s=200, marker='*', label='Optimal Positions', zorder=5)

    positions_combined = {'Initial': initial_positions, 'Optimal': optimal_positions}
    colors = {'Initial': 'blue', 'Optimal': 'red'}
    offsets = {'Initial': 12, 'Optimal': -12} 

    for label_type, positions in positions_combined.items():
        for pos_code, coords in positions.items():
            if not np.isnan(coords).any():
                ax.text(coords[0], coords[1] + offsets[label_type], f"{pos_code}", 
                        color='white', ha='center', va='center', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc=colors[label_type], ec='none', alpha=0.8))

    print("  - 守備站位繪製完成。")

    # 6. 圖表美化與設定
    ax.set_title(f"Initial vs Optimal Outfield Alignment for {batter_name}\nTeam: {', '.join(fielder_names.values())}", fontsize=14)
    ax.set_xlabel("X coordinate (ft) ", fontsize=12)
    ax.set_ylabel("Y coordinate (ft) ", fontsize=12)
    ax.set_xlim(-280, 280)
    ax.set_ylim(0, 420)
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    legend = ax.legend(loc='upper left')
    plt.setp(legend.get_texts(), color='black')
    
    plt.tight_layout()
    
    # 7. 儲存與顯示
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    team_str = f"LF_{fielder_names['LF']}_CF_{fielder_names['CF']}_RF_{fielder_names['RF']}".replace(" ", "_").replace(",", "")
    batter_str = batter_name.replace(" ", "_").replace(",", "")
    output_filename = f"{batter_str}_vs_{team_str}_alignment_comparison.png"
    output_path = FIGURES_DIR / output_filename
    
    plt.savefig(output_path, dpi=300, facecolor='white')
    print(f"\n🎉 [成功] 對比圖表已儲存至: {output_path}")
    
    return fig

if __name__ == "__main__":
    example_batter = "Kwan, Steven"
    example_fielders = {
        "LF": "Profar, Jurickson",
        "CF": "Harris II, Michael",
        "RF": "Acuña Jr., Ronald"
    }
    visualize_team_alignment(example_batter, example_fielders)