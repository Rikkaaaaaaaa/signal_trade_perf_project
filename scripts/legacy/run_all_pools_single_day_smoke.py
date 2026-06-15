from __future__ import annotations

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


POOL_NAMES = ["hs300", "zz500", "zz1000", "zz2000_1", "zz2000_2", "zz2000_3", "other"]


def find_common_trade_date(mysql_config: MysqlConfig) -> str:
    date_sets = []
    for pool_name in POOL_NAMES:
        table_name = build_signal_table_name(pool_name)
        date_sets.append(set(load_signal_dates_mysql(table_name, mysql_config)))
    common_dates = sorted(set.intersection(*date_sets))
    if not common_dates:
        raise RuntimeError("No common trade date found across all pools.")
    return common_dates[0]


def main() -> None:
    results_dir = PROJECT_ROOT / "results" / "smoke_all_pools_single_day"
    cache_dir = PROJECT_ROOT / "cache" / "source_day_cache"
    mkdir_with_retry(results_dir)
    mkdir_with_retry(cache_dir)
    for pool_name in POOL_NAMES:
        mkdir_with_retry(cache_dir / pool_name)

    mysql_config = MysqlConfig()
    smoke_date = find_common_trade_date(mysql_config)
    params = BacktestParams(open_threshold=6.0, close_threshold=4.0, min_hold_bars=20)

    rows = []
    runner = SourceBacktestRunner(mysql_config=mysql_config, day_cache_dir=cache_dir)
    try:
        for pool_name in POOL_NAMES:
            table_name = build_signal_table_name(pool_name)
            start = perf_counter()
            _, security_summary_df, pool_summary_df, timing_df = runner.run_single_param([smoke_date], table_name, params)
            elapsed = perf_counter() - start
            if pool_summary_df.empty:
                rows.append(
                    {
                        "poolName": pool_name,
                        "tradeDate": smoke_date,
                        "status": "empty",
                        "elapsedSeconds": elapsed,
                    }
                )
                continue

            pool_row = pool_summary_df.iloc[0].to_dict()
            timing_row = timing_df.iloc[0].to_dict() if not timing_df.empty else {}
            rows.append(
                {
                    "poolName": pool_name,
                    "tradeDate": smoke_date,
                    "status": "ok",
                    "elapsedSeconds": elapsed,
                    "securityCount": int(pool_row["securityCount"]),
                    "securityDayCount": int(pool_row["securityDayCount"]),
                    "totalTradeCount": int(pool_row["totalTradeCount"]),
                    "avgAllExecRet": float(pool_row["avgAllExecRet"]),
                    "maxConcurrentTotalCount": int(pool_row["maxConcurrentTotalCount"]),
                    "avgMaxConcurrentTotalCount": float(pool_row["avgMaxConcurrentTotalCount"]),
                    "p95MaxConcurrentTotalCount": float(pool_row["p95MaxConcurrentTotalCount"]),
                    "signalRowCount": int(timing_row.get("signalRowCount", 0)),
                }
            )
    finally:
        runner.close()

    result_df = pd.DataFrame(rows)
    dataframe_to_csv_with_retry(result_df, results_dir / "smoke_summary.csv", index=False)
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
