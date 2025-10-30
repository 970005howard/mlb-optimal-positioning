# 檔案位置: main.py (最終修正版)

import argparse

# 導入各個步驟的主執行函式
from src.data.step_01_split_player_data import run_all_splits
from src.data.step_02_preprocess_batted_balls import run_all_preprocessing
from src.modeling.step_03_train_catch_model import run_all_modeling
from src.optimization.step_04_find_optimal_position import run_team_optimization
from src.visualization.step_05_visualize_alignment import visualize_team_alignment
# 假設 step_07 在 src/evaluation/step_07... 且主函式為 compare_initial_vs_optimal
from src.evaluation.step_07_compare_initial_vs_optimal import compare_initial_vs_optimal 

def main():
    """
    定義並解析命令列參數，根據使用者的指令執行對應的專案流程。
    """
    parser = argparse.ArgumentParser(
        description="MLB 外野手防守站位最佳化專案",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- 流程控制參數 ---
    parser.add_argument('--split', action='store_true', 
                        help='(步驟 1) 執行資料分割')
    parser.add_argument('--preprocess', action='store_true', 
                        help='(步驟 2) 執行資料預處理與特徵工程')
    parser.add_argument('--train', action='store_true', 
                        help='(步驟 3) 訓練階層式貝氏回歸模型')
    parser.add_argument('--optimize', action='store_true',
                        help='(步驟 4) 執行「指定團隊」站位最佳化。\n'
                             '必須同時提供 --batter, --lf-player, --cf-player, --rf-player')
    parser.add_argument('--visualize', action='store_true',
                        help='(步驟 5) 將指定團隊的最佳站位視覺化。\n'
                             '必須同時提供 --batter, --lf-player, --cf-player, --rf-player')
    # ✨ [確認] 指令名稱是 --compare
    parser.add_argument('--compare', action='store_true', 
                        help='(步驟 7) 比較初始站位與最佳站位的效益。\n'
                             '必須同時提供 --batter, --lf-player, --cf-player, --rf-player')

    # --- 執行所需參數 ---
    parser.add_argument('--batter', type=str, help='指定目標打者姓名')
    parser.add_argument('--lf-player', type=str, help='指定左外野手姓名')
    parser.add_argument('--cf-player', type=str, help='指定中外野手姓名')
    parser.add_argument('--rf-player', type=str, help='指定右外野手姓名')

    args = parser.parse_args()

    # --- 根據參數執行對應的任務 ---
    if args.split:
        print("\n--- 任務: 執行資料分割 ---")
        run_all_splits()

    if args.preprocess:
        print("\n--- 任務: 執行資料預處理 ---")
        run_all_preprocessing()

    if args.train:
        print("\n--- 任務: 執行模型訓練 ---")
        run_all_modeling()

    if args.optimize:
        required_args = [args.batter, args.lf_player, args.cf_player, args.rf_player]
        if not all(required_args):
            print("\n❌ [錯誤] 使用 --optimize 時，必須同時提供所有球員姓名。")
        else:
            print("\n--- 任務: 執行團隊站位最佳化 ---")
            fielder_names = {"LF": args.lf_player, "CF": args.cf_player, "RF": args.rf_player}
            run_team_optimization(batter_name=args.batter, fielder_names=fielder_names)

    if args.visualize:
        required_args = [args.batter, args.lf_player, args.cf_player, args.rf_player]
        if not all(required_args):
            print("\n❌ [錯誤] 使用 --visualize 時，必須同時提供所有球員姓名。")
        else:
            print("\n--- 任務: 執行團隊站位視覺化 ---")
            fielder_names = {"LF": args.lf_player, "CF": args.cf_player, "RF": args.rf_player}
            visualize_team_alignment(batter_name=args.batter, fielder_names=fielder_names)
            
    # ✨ [確認] 判斷條件是 args.compare
    if args.compare:
        required_args = [args.batter, args.lf_player, args.cf_player, args.rf_player]
        if not all(required_args):
            print("\n❌ [錯誤] 使用 --compare 時，必須同時提供所有球員姓名。")
        else:
            print("\n--- 任務: 比較初始站位 vs. 最佳站位 ---")
            fielder_names = {"LF": args.lf_player, "CF": args.cf_player, "RF": args.rf_player}
            compare_initial_vs_optimal(batter_name=args.batter, fielder_names=fielder_names)

    # --- 完整流程執行 ---
    # ✨ [核心修正] 確保 active_flags 列表包含所有正確的旗標
    active_flags = [args.split, args.preprocess, args.train, args.optimize, args.visualize, args.compare] 
    if not any(active_flags):
        print("=== 未指定特定任務，將執行預設的基礎流程 (步驟 1-3) ===")
        print("\n--- 任務: 執行資料分割 ---")
        run_all_splits()
        print("\n--- 任務: 執行資料預處理 ---")
        run_all_preprocessing()
        print("\n--- 任務: 執行模型訓練 ---")
        run_all_modeling()
        print("\n✅ [提示] 基礎流程的前三步已完成。")
        print("若要執行後續步驟 (4-7)，請使用特定指令。")

if __name__ == "__main__":
    main()