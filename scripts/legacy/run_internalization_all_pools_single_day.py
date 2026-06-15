from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from signal_trade_perf import BacktestParams
from signal_trade_perf.internalization import get_default_ims_roots, run_internalization_single_day
from signal_trade_perf.source_backtest import dataframe_to_csv_with_retry, mkdir_with_retry


POOL_NAMES = ["hs300", "zz500", "zz1000", "zz2000_1", "zz2000_2", "zz2000_3", "other"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-day internalization backtest across all pools.")
    parser.add_argument("--date", default="20260105")
    parser.add_argument("--open-threshold", type=float, default=6.0)
    parser.add_argument("--close-threshold", type=float, default=4.0)
    parser.add_argument("--min-hold-bars", type=int, default=20)
    parser.add_argument("--match-window-seconds", type=int, default=10)
    args = parser.parse_args()

    params = BacktestParams(
        open_threshold=args.open_threshold,
        close_threshold=args.close_threshold,
        min_hold_bars=args.min_hold_bars,
    )
    result_dir = (
        PROJECT_ROOT
        / "results"
        / "internalization_all_pools_single_day"
        / f"{args.date}_{params.param_tag}"
    )
    mkdir_with_retry(result_dir)

    datetime_format = "%Y-%m-%d %H:%M:%S.%f"
    summary_frames: list[pd.DataFrame] = []
    timing_rows: list[dict[str, object]] = []

    for idx, pool_name in enumerate(POOL_NAMES, start=1):
        pool_dir = result_dir / pool_name
        mkdir_with_retry(pool_dir)
        start = perf_counter()
        order_events_df, trades_df, security_summary_df, pool_summary_df = run_internalization_single_day(
            trade_date=args.date,
            pool_name=pool_name,
            params=params,
            ims_roots=get_default_ims_roots(PROJECT_ROOT),
            match_window_seconds=args.match_window_seconds,
        )
        elapsed = perf_counter() - start

        dataframe_to_csv_with_retry(order_events_df, pool_dir / "order_events.csv", index=False, date_format=datetime_format)
        dataframe_to_csv_with_retry(trades_df, pool_dir / "trades.csv", index=False, date_format=datetime_format)
        dataframe_to_csv_with_retry(security_summary_df, pool_dir / "security_summary.csv", index=False, date_format=datetime_format)
        dataframe_to_csv_with_retry(pool_summary_df, pool_dir / "pool_summary.csv", index=False, date_format=datetime_format)

        if not pool_summary_df.empty:
            pool_summary_with_pool = pool_summary_df.copy()
            pool_summary_with_pool["poolName"] = pool_name
            summary_frames.append(pool_summary_with_pool)

        timing_rows.append(
            {
                "poolName": pool_name,
                "elapsedSeconds": elapsed,
                "orderEventCount": int(len(order_events_df)),
                "tradeCount": int(len(trades_df)),
                "variantCount": int(pool_summary_df["variantTag"].nunique()) if not pool_summary_df.empty else 0,
            }
        )
        print(f"[{idx}/{len(POOL_NAMES)}] {pool_name} elapsed={elapsed:.2f}s trades={len(trades_df)}")

    all_pool_summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    timing_df = pd.DataFrame(timing_rows)
    dataframe_to_csv_with_retry(all_pool_summary_df, result_dir / "all_pool_summary.csv", index=False, date_format=datetime_format)
    dataframe_to_csv_with_retry(timing_df, result_dir / "pool_timing.csv", index=False, date_format=datetime_format)

    print(f"resultDir={result_dir}")


if __name__ == "__main__":
    main()
