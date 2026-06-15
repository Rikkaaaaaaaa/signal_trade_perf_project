from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from signal_trade_perf.configs import build_signal_table_name
from signal_trade_perf.io_utils import load_prev_signal_date_mysql
from signal_trade_perf.signal_backtest import (
    BacktestParams,
    MysqlConfig,
    SourceBacktestRunner,
    aggregate_param_summary,
    simulate_signal_day,
)
from signal_trade_perf.source_backtest import dataframe_to_csv_with_retry, mkdir_with_retry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-day legacy signal backtest.")
    parser.add_argument("--date", default="20260105")
    parser.add_argument("--pool", default="hs300")
    parser.add_argument("--open-threshold", type=float, default=6.0)
    parser.add_argument("--close-threshold", type=float, default=4.0)
    parser.add_argument("--min-hold-bars", type=int, default=20)
    args = parser.parse_args()

    params = BacktestParams(
        open_threshold=args.open_threshold,
        close_threshold=args.close_threshold,
        min_hold_bars=args.min_hold_bars,
    )
    result_dir = (
        PROJECT_ROOT
        / "results"
        / "signal_backtest"
        / "single_day"
        / f"{args.pool}_{args.date}_{params.param_tag}"
    )
    cache_dir = PROJECT_ROOT / "cache" / "source_day_cache"
    mkdir_with_retry(result_dir)
    mkdir_with_retry(cache_dir)

    mysql_config = MysqlConfig()
    table_name = build_signal_table_name(args.pool)
    prev_trade_date = load_prev_signal_date_mysql(args.date, table_name, mysql_config)

    runner = SourceBacktestRunner(mysql_config=mysql_config, day_cache_dir=cache_dir)
    try:
        signal_df, meta_df, prep = runner.prepare_signal_mid_day(
            trade_date=args.date,
            prev_trade_date=prev_trade_date,
            table_name=table_name,
            force_rebuild=False,
        )
    finally:
        runner.close()

    if signal_df.empty:
        print("signal_df is empty")
        return

    trade_df, security_summary_df = simulate_signal_day(
        signal_df=signal_df,
        meta_df=meta_df,
        pool_name=args.pool,
        trade_date=signal_df["tradeDate"].iloc[0],
        params=params,
    )
    pool_summary_df = aggregate_param_summary(security_summary_df, params, args.pool)

    dataframe_to_csv_with_retry(trade_df, result_dir / "trades.csv", index=False)
    dataframe_to_csv_with_retry(security_summary_df, result_dir / "security_summary.csv", index=False)
    dataframe_to_csv_with_retry(pool_summary_df, result_dir / "pool_summary.csv", index=False)
    dataframe_to_csv_with_retry(
        pd.DataFrame([{"tradeDate": args.date, **prep}]),
        result_dir / "prepare_timing.csv",
        index=False,
    )

    print(pool_summary_df.to_string(index=False))
    print(f"tradeCount={len(trade_df)}")
    print(f"resultDir={result_dir}")


if __name__ == "__main__":
    main()
