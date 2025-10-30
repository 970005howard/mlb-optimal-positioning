# 檔案位置: src/utils/dashboard_utils.py

import streamlit as st
import arviz as az
import pandas as pd
from pathlib import Path
from config import MODELS_DIR, INPUTS_DATA_DIR

@st.cache_data # Streamlit 的快取功能，避免重複載入
def get_player_lists():
    """
    掃描模型和資料檔案，獲取所有可選的球員和打者列表。
    """
    
    # 1. 獲取打者列表
    batter_dir = INPUTS_DATA_DIR / "batter_spray_charts"
    batter_files = list(batter_dir.glob("*.csv"))
    # 從檔名 "Kwan, Steven.csv" 中提取 "Kwan, Steven"
    batters = sorted([f.stem for f in batter_files])

    # 2. 獲取外野手列表
    fielder_lists = {}
    for pos_code in ["LF", "CF", "RF"]:
        trace_path = MODELS_DIR / pos_code / f"{pos_code}_model_trace.nc"
        if trace_path.exists():
            trace = az.from_netcdf(trace_path)
            # 從模型的 'player' 維度獲取球員姓名列表
            players = sorted(trace.posterior['player'].values.tolist())
            fielder_lists[pos_code] = players
        else:
            fielder_lists[pos_code] = [] # 如果模型不存在，返回空列表

    return batters, fielder_lists.get("LF", []), fielder_lists.get("CF", []), fielder_lists.get("RF", [])