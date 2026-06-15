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


def list_cached_trade_dates(pool_name: str, start_date: str, end_date: str) -> list[str]:
    pool_dir = PROJECT_ROOT / "cache" / "source_day_cache" / pool_name
    dates = sorted({path.name.split("_")[0] for path in pool_dir.glob("*_signal_quote.pkl.gz")})
    return [trade_date for trade_date in dates if start_date <= trade_date <= end_date]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run internalization backtest for a date range.")
    parser.add_argument("--pool", default="hs300")
    parser.add_argument("--start-date", default="20260101")
    parser.add_argument("--end-date", default="20260430")
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
    trade_dates = list_cached_trade_dates(args.pool, args.start_date, args.end_date)
    if not trade_dates:
        raise ValueError(f"No cached trade dates found for {args.pool} in [{args.start_date}, {args.end_date}]")

    result_dir = (
        PROJECT_ROOT
        / "results"
        / "internalization_date_range"
        / f"{args.pool}_{trade_dates[0]}_{trade_dates[-1]}_{params.param_tag}"
    )
    mkdir_with_retry(result_dir)

    all_pool_summaries: list[pd.DataFrame] = []
    timing_rows: list[dict[str, object]] = []
    total_start = perf_counter()

    for idx, trade_date in enumerate(trade_dates, start=1):
        day_start = perf_counter()
        order_events_df, trades_df, security_summary_df, pool_summary_df = run_internalization_single_day(
            trade_date=trade_date,
            pool_name=args.pool,
            params=params,
            ims_roots=get_default_ims_roots(PROJECT_ROOT),
            match_window_seconds=args.match_window_seconds,
        )
        elapsed = perf_counter() - day_start

        if not pool_summary_df.empty:
            pool_summary_df = pool_summary_df.copy()
            pool_summary_df["tradeDate"] = trade_date
            all_pool_summaries.append(pool_summary_df)

        timing_rows.append(
            {
                "tradeDate": trade_date,
                "elapsedSeconds": elapsed,
                "orderEventCount": int(len(order_events_df)),
                "tradeCount": int(len(trades_df)),
                "variantCount": int(pool_summary_df["variantTag"].nunique()) if not pool_summary_df.empty else 0,
            }
        )
        print(f"[{idx}/{len(trade_dates)}] {trade_date} elapsed={elapsed:.2f}s trades={len(trades_df)}")

    total_elapsed = perf_counter() - total_start
    range_pool_summary_df = pd.concat(all_pool_summaries, ignore_index=True) if all_pool_summaries else pd.DataFrame()
    timing_df = pd.DataFrame(timing_rows)
    total_timing_df = pd.DataFrame(
        [
            {
                "poolName": args.pool,
                "startDate": trade_dates[0],
                "endDate": trade_dates[-1],
                "tradeDateCount": len(trade_dates),
                "paramTag": params.param_tag,
                "totalElapsedSeconds": total_elapsed,
            }
        ]
    )

    datetime_format = "%Y-%m-%d %H:%M:%S.%f"
    dataframe_to_csv_with_retry(range_pool_summary_df, result_dir / "range_pool_summary.csv", index=False, date_format=datetime_format)
    dataframe_to_csv_with_retry(timing_df, result_dir / "date_timing.csv", index=False, date_format=datetime_format)
    dataframe_to_csv_with_retry(total_timing_df, result_dir / "total_timing.csv", index=False, date_format=datetime_format)

    print(f"tradeDateCount={len(trade_dates)}")
    print(f"totalElapsedSeconds={total_elapsed:.2f}")
    print(f"resultDir={result_dir}")


if __name__ == "__main__":
    main()
