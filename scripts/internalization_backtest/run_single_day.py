from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from signal_trade_perf.internalization_backtest import BacktestParams, get_default_ims_roots, run_internalization_single_day
from signal_trade_perf.source_backtest import dataframe_to_csv_with_retry, mkdir_with_retry


def main() -> None:
    # 这是 internalization 链路最简的一日入口，适合先验证单池结果。
    parser = argparse.ArgumentParser(description="Run one-day internalization backtest.")
    parser.add_argument("--date", default="20260105")
    parser.add_argument("--pool", default="hs300")
    parser.add_argument("--open-threshold", type=float, default=6.0)
    parser.add_argument("--close-threshold", type=float, default=4.0)
    parser.add_argument("--min-hold-bars", type=int, default=5)
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
        / "internalization_backtest"
        / "single_day"
        / f"{args.pool}_{args.date}_{params.param_tag}"
    )
    mkdir_with_retry(result_dir)

    order_events_df, trades_df, security_summary_df, pool_summary_df = run_internalization_single_day(
        trade_date=args.date,
        pool_name=args.pool,
        params=params,
        ims_roots=get_default_ims_roots(PROJECT_ROOT),
        match_window_seconds=args.match_window_seconds,
    )

    datetime_format = "%Y-%m-%d %H:%M:%S.%f"
    dataframe_to_csv_with_retry(order_events_df, result_dir / "order_events.csv", index=False, date_format=datetime_format)
    dataframe_to_csv_with_retry(trades_df, result_dir / "trades.csv", index=False, date_format=datetime_format)
    dataframe_to_csv_with_retry(security_summary_df, result_dir / "security_summary.csv", index=False, date_format=datetime_format)
    dataframe_to_csv_with_retry(pool_summary_df, result_dir / "pool_summary.csv", index=False, date_format=datetime_format)

    print(pool_summary_df.to_string(index=False))
    print(f"tradeCount={len(trades_df)}")
    print(f"resultDir={result_dir}")


if __name__ == "__main__":
    main()
