from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from signal_trade_perf.source_backtest import dataframe_to_csv_with_retry


COMPARE_METRICS = [
    "avgAllExecRet",
    "totalExecPnl",
    "matchedClientSellAmt",
    "clientAmtMatchRate",
    "totalTradeCount",
    "avgAllHoldMinutes",
    "notionalWeightedExecRet",
    "totalEodCloseCount",
]


def _load_summary(path: Path, hold_bars: int) -> pd.DataFrame:
    # compare 现在改成长表，每个 holdBars 单独一行，只保留最核心的结果指标。
    summary_df = pd.read_csv(path)
    required_cols = ["poolName", "variantTag", *COMPARE_METRICS]
    selected_cols = [col for col in required_cols if col in summary_df.columns]
    result_df = summary_df[selected_cols].copy()
    result_df["holdBars"] = int(hold_bars)

    first_columns = ["poolName", "variantTag", "holdBars"]
    remaining_cols = [col for col in result_df.columns if col not in first_columns]
    return result_df[first_columns + remaining_cols]


def main() -> None:
    # 这个脚本专门把 hold5 / hold20 汇总整理成一张长表，便于直接按行比较。
    parser = argparse.ArgumentParser(description="Build long-format internalization compare CSV for two hold-bar summaries.")
    parser.add_argument("--date", default="20260105")
    parser.add_argument("--open-threshold", type=float, default=6.0)
    parser.add_argument("--close-threshold", type=float, default=4.0)
    parser.add_argument("--left-hold-bars", type=int, default=5)
    parser.add_argument("--right-hold-bars", type=int, default=20)
    args = parser.parse_args()

    base_dir = PROJECT_ROOT / "results" / "internalization_backtest" / "all_pools_single_day"
    left_tag = f"open_{int(args.open_threshold)}_close_{int(args.close_threshold)}_hold_{args.left_hold_bars}"
    right_tag = f"open_{int(args.open_threshold)}_close_{int(args.close_threshold)}_hold_{args.right_hold_bars}"
    left_path = base_dir / f"{args.date}_{left_tag}" / "all_pool_summary.csv"
    right_path = base_dir / f"{args.date}_{right_tag}" / "all_pool_summary.csv"

    compare_df = pd.concat(
        [
            _load_summary(left_path, args.left_hold_bars),
            _load_summary(right_path, args.right_hold_bars),
        ],
        ignore_index=True,
    ).sort_values(["poolName", "variantTag", "holdBars"]).reset_index(drop=True)

    output_path = base_dir / f"{args.date}_hold{args.left_hold_bars}_vs_hold{args.right_hold_bars}_compare.csv"
    dataframe_to_csv_with_retry(compare_df, output_path, index=False)
    print(f"outputPath={output_path}")


if __name__ == "__main__":
    main()
