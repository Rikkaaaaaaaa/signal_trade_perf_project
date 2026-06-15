from __future__ import annotations

from itertools import product
from pathlib import Path
from time import perf_counter
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from signal_trade_perf import BacktestParams
from signal_trade_perf.source_backtest import (
    MysqlConfig,
    SourceBacktestRunner,
    build_signal_table_name,
    dataframe_to_csv_with_retry,
    load_signal_dates_mysql,
    mkdir_with_retry,
)


def main() -> None:
    results_dir = PROJECT_ROOT / "results" / "hs300_all_dates"
    cache_dir = PROJECT_ROOT / "cache" / "source_day_cache"
    mkdir_with_retry(results_dir)
    mkdir_with_retry(cache_dir)

    pool_name = "hs300"
    table_name = build_signal_table_name(pool_name)
    mysql_config = MysqlConfig()
    trade_dates = load_signal_dates_mysql(table_name, mysql_config)

    params_list = [
        BacktestParams(open_threshold=open_threshold, close_threshold=close_threshold, min_hold_bars=min_hold_bars)
        for open_threshold, close_threshold, min_hold_bars in product([6.0, 7.5], [4.0, 4.5], [20, 40])
    ]

    runner = SourceBacktestRunner(mysql_config=mysql_config, day_cache_dir=cache_dir)
    try:
        total_start = perf_counter()
        pool_summary_df, price_summary_df, vol_summary_df, prepare_timing_df, param_timing_df, vol_cutoffs = runner.run_param_sweep(
            trade_dates=trade_dates,
            table_name=table_name,
            params_list=params_list,
            num_vol_bins=10,
            force_rebuild_cache=False,
        )
        total_elapsed = perf_counter() - total_start
    finally:
        runner.close()

    dataframe_to_csv_with_retry(pool_summary_df, results_dir / "pool_summary.csv", index=False)
    dataframe_to_csv_with_retry(price_summary_df, results_dir / "price_summary.csv", index=False)
    dataframe_to_csv_with_retry(vol_summary_df, results_dir / "vol_summary.csv", index=False)
    dataframe_to_csv_with_retry(prepare_timing_df, results_dir / "prepare_timing.csv", index=False)
    dataframe_to_csv_with_retry(param_timing_df, results_dir / "param_timing.csv", index=False)
    dataframe_to_csv_with_retry(pd.DataFrame({"tradeDate": trade_dates}), results_dir / "trade_dates.csv", index=False)
    dataframe_to_csv_with_retry(pd.DataFrame({"prevVolCutoff": vol_cutoffs}), results_dir / "vol_cutoffs.csv", index=False)
    dataframe_to_csv_with_retry(pd.DataFrame(
        [{"totalElapsedSeconds": total_elapsed, "tradeDateCount": len(trade_dates), "paramCount": len(params_list)}]
    ), results_dir / "total_timing.csv", index=False)

    print(pool_summary_df.to_string(index=False))
    print(f"tradeDateCount={len(trade_dates)}")
    print(f"paramCount={len(params_list)}")
    print(f"totalElapsedSeconds={total_elapsed:.3f}")


if __name__ == "__main__":
    main()
