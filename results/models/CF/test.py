import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# --- 1. 檔案路徑設定 ---
# 請確保這個路徑指向您上傳的 .nc 檔案
TRACE_FILE_NAME = "CF_model_trace.nc" 
file_path = Path(TRACE_FILE_NAME)

print(f"--- 開始讀取 MCMC 追蹤檔案: {TRACE_FILE_NAME} ---")

# --- 2. 讀取 MCMC 追蹤數據 ---
try:
    # 使用 arviz.from_netcdf 讀取數據，轉換為 InferenceData 物件
    idata = az.from_netcdf(file_path)
    print("檔案讀取成功。")
except FileNotFoundError:
    print(f"錯誤：找不到檔案 {TRACE_FILE_NAME}。請確認檔案路徑是否正確。")
    exit()
except Exception as e:
    print(f"讀取檔案時發生錯誤: {e}")
    exit()


# --- 3. 輸出收斂與統計摘要 ---
print("\n--- 1. 模型參數摘要與收斂診斷 (R-hat & ESS) ---")

# 使用 az.summary 輸出摘要，包括平均值、標準差以及收斂指標
summary = az.summary(idata, round_to=3) 

# 打印摘要
print(summary)

print("\n--- 摘要解讀 ---")
print("1. Rhat (Gelman-Rubin 統計量)：應該接近 1.0 (通常 < 1.05)。如果 Rhat 很高，表示 MCMC 鏈尚未收斂。")
print("2. ess_bulk (Effective Sample Size, 有效樣本數)：表示您擁有的獨立樣本數量。值越高越好。")
print("3. mean：參數的後驗平均值，即最佳估計值。")


# --- 4. 視覺化診斷 ---
print("\n--- 2. 視覺化追蹤診斷圖 ---")

# 繪製追蹤圖 (Trace Plot)
# 左側是後驗分佈的直方圖，右側是每條鏈的採樣歷史軌跡圖
# 通過軌跡圖可檢查鏈的混合性（Mixing）和暖機期（Burn-in）
try:
    az.plot_trace(idata)
    plt.tight_layout()
    plt.show()

    print("追蹤圖 (Trace Plot) 已顯示。請觀察右側軌跡：")
    print("- 軌跡是否看起來像『毛毛蟲』一樣充分混合？")
    print("- 軌跡是否在整個採樣過程中都保持穩定（沒有向上或向下的漂移）？")

    # 繪製自相關圖 (Autocorrelation Plot)
    # 用於檢查樣本的獨立性
    az.plot_autocorr(idata)
    plt.tight_layout()
    plt.show()
    print("自相關圖已顯示。理想情況下，曲線應該迅速下降至零，表示樣本的自相關性低。")

except Exception as e:
    print(f"繪製圖表時發生錯誤: {e}")