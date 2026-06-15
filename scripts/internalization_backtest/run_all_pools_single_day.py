from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from signal_trade_perf.internalization_backtest import BacktestParams, get_default_ims_roots, run_internalization_single_day
from signal_trade_perf.source_backtest import dataframe_to_csv_with_retry, mkdir_with_retry


POOL_NAMES = ["hs300", "zz500", "zz1000", "zz2000_1", "zz2000_2", "zz2000_3", "other"]
# POOL_NAMES = ["zz2000_3"]


REPORT_VARIANT_ORDER = ["all", "lt1000", "lt2000", "liqcap5tick", "poscap_min5", "poscap_avg5"]
CORE_METRIC_COLUMNS = [
    "totalTradeCount",
    "totalExecPnl",
    "clientAmtMatchRate",
    "notionalWeightedExecRet",
    "totalMatchedNotional",
]


def _variant_metric_rows(pool_summary_df: pd.DataFrame) -> list[dict[str, object]]:
    if pool_summary_df.empty:
        return []

    variant_order = {variant_tag: idx for idx, variant_tag in enumerate(REPORT_VARIANT_ORDER)}
    rows = pool_summary_df[pool_summary_df["variantTag"].isin(REPORT_VARIANT_ORDER)].to_dict(orient="records")
    rows.sort(key=lambda row: variant_order.get(str(row["variantTag"]), len(variant_order)))
    return rows


def _print_core_metric_report(pool_name: str, trade_date: str, row: dict[str, object]) -> None:
    print("")
    print(f"pool: {pool_name}")
    print(f"date: {trade_date}")
    print(f"variantTag: {row['variantTag']}")
    print(f"trades: {int(row['totalTradeCount'])}")
    for column in CORE_METRIC_COLUMNS[1:]:
        print(f"{column}: {row[column]}")


def _build_all_pool_core_summaries(all_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_df = pd.DataFrame(all_rows)
    all_pool_rows: list[dict[str, object]] = []
    for variant_tag in REPORT_VARIANT_ORDER:
        variant_df = summary_df[summary_df["variantTag"] == variant_tag]
        if variant_df.empty:
            continue

        total_client_amt = float(variant_df["totalClientAmt"].astype(float).sum())
        matched_client_amt = float(variant_df["matchedClientAmt"].astype(float).sum())
        total_exec_pnl = float(variant_df["totalExecPnl"].astype(float).sum())
        total_matched_notional = float(variant_df["totalMatchedNotional"].astype(float).sum())
        all_pool_rows.append(
            {
                "variantTag": variant_tag,
                "totalTradeCount": int(variant_df["totalTradeCount"].astype(int).sum()),
                "totalExecPnl": total_exec_pnl,
                "clientAmtMatchRate": float("nan") if total_client_amt == 0 else matched_client_amt / total_client_amt,
                "notionalWeightedExecRet": (
                    float("nan") if total_matched_notional == 0 else total_exec_pnl / total_matched_notional
                ),
                "totalMatchedNotional": total_matched_notional,
            }
        )
    return all_pool_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-day internalization backtest across all pools.")
    parser.add_argument("--date", default="20260105")
    parser.add_argument("--open-threshold", type=float, default=6.0)
    parser.add_argument("--close-threshold", type=float, default=4.0)
    parser.add_argument("--min-hold-bars", type=int, default=5)
    parser.add_argument("--match-window-seconds", type=int, default=10)
    parser.add_argument("--profile", action="store_true", help="Print stage timing diagnostics to stdout.")
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
        / "all_pools_single_day"
        / f"{args.date}_{params.param_tag}"
    )
    mkdir_with_retry(result_dir)

    datetime_format = "%Y-%m-%d %H:%M:%S.%f"
    summary_frames: list[pd.DataFrame] = []
    core_metric_rows: list[dict[str, object]] = []
    total_start = perf_counter()

    for idx, pool_name in enumerate(POOL_NAMES, start=1):
        pool_start = perf_counter()
        pool_dir = result_dir / pool_name
        mkdir_with_retry(pool_dir)
        order_events_df, trades_df, security_summary_df, pool_summary_df = run_internalization_single_day(
            trade_date=args.date,
            pool_name=pool_name,
            params=params,
            ims_roots=get_default_ims_roots(PROJECT_ROOT),
            match_window_seconds=args.match_window_seconds,
            profile=args.profile,
        )
        pool_elapsed = perf_counter() - pool_start

        if order_events_df.empty and trades_df.empty and security_summary_df.empty and pool_summary_df.empty:
            print(f"[{idx}/{len(POOL_NAMES)}] {pool_name} skipped_empty_result poolElapsedSeconds={pool_elapsed:.2f}")
            continue

        dataframe_to_csv_with_retry(order_events_df, pool_dir / "order_events.csv", index=False, date_format=datetime_format)
        dataframe_to_csv_with_retry(trades_df, pool_dir / "trades.csv", index=False, date_format=datetime_format)
        dataframe_to_csv_with_retry(security_summary_df, pool_dir / "security_summary.csv", index=False, date_format=datetime_format)
        dataframe_to_csv_with_retry(pool_summary_df, pool_dir / "pool_summary.csv", index=False, date_format=datetime_format)

        if not pool_summary_df.empty:
            pool_summary_with_pool = pool_summary_df.copy()
            pool_summary_with_pool["poolName"] = pool_name
            pool_summary_with_pool = pool_summary_with_pool[
                ["poolName", *[col for col in pool_summary_with_pool.columns if col != "poolName"]]
            ]
            summary_frames.append(pool_summary_with_pool)
            variant_rows = _variant_metric_rows(pool_summary_with_pool)
            core_metric_rows.extend(variant_rows)
            for variant_row in variant_rows:
                _print_core_metric_report(pool_name, args.date, variant_row)

        print(f"[{idx}/{len(POOL_NAMES)}] {pool_name} trades={len(trades_df)} poolElapsedSeconds={pool_elapsed:.2f}")

    all_pool_summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    if not all_pool_summary_df.empty:
        dataframe_to_csv_with_retry(all_pool_summary_df, result_dir / "all_pool_summary.csv", index=False, date_format=datetime_format)
    else:
        print("all_pool_summary skipped: no non-empty pool summaries.")

    if core_metric_rows:
        for all_pool_row in _build_all_pool_core_summaries(core_metric_rows):
            _print_core_metric_report("ALL_POOLS", args.date, all_pool_row)

    total_elapsed = perf_counter() - total_start
    print(f"totalElapsedSeconds: {total_elapsed:.2f}")
    print(f"resultDir={result_dir}")


if __name__ == "__main__":
    main()
