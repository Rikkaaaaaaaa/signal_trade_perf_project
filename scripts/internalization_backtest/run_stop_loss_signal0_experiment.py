from __future__ import annotations

import argparse
import gzip
import multiprocessing as mp
import pickle
import sys
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from signal_trade_perf.internalization import _variant_mask
from signal_trade_perf.internalization_backtest import BacktestParams, run_internalization_prepared_day
from signal_trade_perf.source_backtest import dataframe_to_csv_with_retry, mkdir_with_retry


POOL_NAMES = ["hs300", "zz500", "zz1000", "zz2000_1", "zz2000_2", "zz2000_3", "other"]
VARIANT_TAGS = ["all", "liqcap5tick"]
CORE_SUM_COLUMNS = [
    "totalTradeCount",
    "totalExecPnl",
    "totalMatchedNotional",
    "totalClientAmt",
    "matchedClientAmt",
]
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "results" / "internalization_backtest" / "data_cache"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "internalization_backtest" / "stop_loss_hard_tail_experiment"
DEFAULT_BASELINE_TOTAL = (
    PROJECT_ROOT
    / "results"
    / "internalization_backtest"
    / "param_sweep_date_mp"
    / "20260105_20260331"
    / "open_6_close_6_hold_30_match_unlimited"
    / "total_all_dates_summary.csv"
)


def _parse_pools(raw: str) -> list[str]:
    token = raw.strip().lower()
    if token in {"all", "*"}:
        return POOL_NAMES
    pools = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = sorted(set(pools) - set(POOL_NAMES))
    if unknown:
        raise ValueError(f"Unknown pools: {unknown}")
    return pools


def _parse_match_window(raw: str) -> int | None:
    token = raw.strip().lower()
    if token in {"none", "unlimited", "all", "不限"}:
        return None
    return int(token)


def _match_window_value(match_window_seconds: int | None) -> str:
    return "unlimited" if match_window_seconds is None else str(match_window_seconds)


def _cache_path(cache_root: Path, trade_date: str, pool_name: str) -> Path:
    return cache_root / trade_date / f"{pool_name}.pkl.gz"


def _load_cached_inputs(path: Path) -> dict[str, object]:
    with gzip.open(path, "rb") as file:
        return pickle.load(file)


def _discover_trade_dates(cache_root: Path, start_date: str, end_date: str) -> list[str]:
    if not cache_root.exists():
        return []
    return sorted(
        path.name
        for path in cache_root.iterdir()
        if path.is_dir() and start_date <= path.name <= end_date
    )


def _aggregate_core_rows(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(group_cols, dropna=False, as_index=False)[CORE_SUM_COLUMNS].sum()
    grouped["clientAmtMatchRate"] = grouped["matchedClientAmt"] / grouped["totalClientAmt"]
    grouped["notionalWeightedExecRet"] = grouped["totalExecPnl"] / grouped["totalMatchedNotional"]
    return grouped


def _run_one_pool(task: dict[str, Any]) -> dict[str, Any]:
    trade_date = str(task["tradeDate"])
    pool_name = str(task["poolName"])
    cache_root = Path(str(task["cacheRoot"]))
    params = BacktestParams(
        open_threshold=float(task["openThreshold"]),
        close_threshold=float(task["closeThreshold"]),
        min_hold_bars=int(task["minHoldBars"]),
        stop_loss_mid_ret_threshold=float(task["stopLossMidRetThreshold"]),
        stop_loss_signal_threshold=float(task["StopLossSignalThreshold"]),
    )
    match_window_seconds = task["matchWindowSeconds"]
    start = perf_counter()
    try:
        cache_file = _cache_path(cache_root, trade_date, pool_name)
        if not cache_file.exists():
            return {
                "tradeDate": trade_date,
                "poolName": pool_name,
                "status": "missing_cache",
                "elapsedSeconds": perf_counter() - start,
                "poolRows": [],
                "closeTypeRows": [],
            }
        prepared_inputs = _load_cached_inputs(cache_file)
        _, trades_df, _, pool_summary_df = run_internalization_prepared_day(
            prepared_inputs=prepared_inputs,
            params=params,
            match_window_seconds=match_window_seconds,
            profile=False,
        )

        pool_rows: list[dict[str, Any]] = []
        if not pool_summary_df.empty:
            pool_summary_df = pool_summary_df[pool_summary_df["variantTag"].isin(VARIANT_TAGS)].copy()
            pool_summary_df["tradeDate"] = trade_date
            pool_summary_df["poolName"] = pool_name
            pool_summary_df["matchWindowSeconds"] = _match_window_value(match_window_seconds)
            pool_rows = pool_summary_df.to_dict(orient="records")

        close_type_rows: list[dict[str, Any]] = []
        if not trades_df.empty:
            for variant_tag in VARIANT_TAGS:
                variant_trades_df = trades_df[_variant_mask(trades_df, variant_tag)]
                if variant_trades_df.empty:
                    continue
                close_stats = (
                    variant_trades_df.groupby("closeType", dropna=False)
                    .agg(
                        tradeCount=("execPnl", "size"),
                        totalExecPnl=("execPnl", "sum"),
                        totalMatchedNotional=("openNotional", "sum"),
                    )
                    .reset_index()
                )
                close_stats["tradeDate"] = trade_date
                close_stats["poolName"] = pool_name
                close_stats["variantTag"] = variant_tag
                close_type_rows.extend(close_stats.to_dict(orient="records"))

        return {
            "tradeDate": trade_date,
            "poolName": pool_name,
            "status": "ok",
            "elapsedSeconds": perf_counter() - start,
            "poolRows": pool_rows,
            "closeTypeRows": close_type_rows,
        }
    except Exception:
        return {
            "tradeDate": trade_date,
            "poolName": pool_name,
            "status": "error",
            "elapsedSeconds": perf_counter() - start,
            "error": traceback.format_exc(),
            "poolRows": [],
            "closeTypeRows": [],
        }


def _build_comparison(stop_total_df: pd.DataFrame, baseline_path: Path) -> pd.DataFrame:
    if stop_total_df.empty or not baseline_path.exists():
        return pd.DataFrame()
    baseline_df = pd.read_csv(baseline_path)
    baseline_df = baseline_df[baseline_df["variantTag"].isin(VARIANT_TAGS)].copy()
    merge_cols = ["variantTag"]
    keep_cols = [
        "variantTag",
        "totalTradeCount",
        "totalExecPnl",
        "totalMatchedNotional",
        "clientAmtMatchRate",
        "notionalWeightedExecRet",
    ]
    baseline_df = baseline_df[keep_cols].rename(columns={col: f"{col}Baseline" for col in keep_cols if col not in merge_cols})
    stop_df = stop_total_df[keep_cols].rename(columns={col: f"{col}StopLoss" for col in keep_cols if col not in merge_cols})
    compare_df = baseline_df.merge(stop_df, on=merge_cols, how="outer")
    compare_df["deltaTotalExecPnl"] = compare_df["totalExecPnlStopLoss"] - compare_df["totalExecPnlBaseline"]
    compare_df["deltaNotionalWeightedExecRet"] = (
        compare_df["notionalWeightedExecRetStopLoss"] - compare_df["notionalWeightedExecRetBaseline"]
    )
    compare_df["deltaTradeCount"] = compare_df["totalTradeCountStopLoss"] - compare_df["totalTradeCountBaseline"]
    return compare_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adverse mid move + opposite signal hard-tail stop-loss experiment from cached inputs.")
    parser.add_argument("--start-date", default="20260105")
    parser.add_argument("--end-date", default="20260331")
    parser.add_argument("--pools", default="all")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--baseline-total-path", type=Path, default=DEFAULT_BASELINE_TOTAL)
    parser.add_argument("--open-threshold", type=float, default=6.0)
    parser.add_argument("--close-threshold", type=float, default=6.0)
    parser.add_argument("--min-hold-bars", type=int, default=30)
    parser.add_argument("--match-window-seconds", default="unlimited")
    parser.add_argument("--stop-loss-bp", type=float, default=150.0)
    parser.add_argument("--stop-loss-signal-threshold", type=float, default=2.0)
    parser.add_argument("--processes", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pools = _parse_pools(args.pools)
    match_window_seconds = _parse_match_window(args.match_window_seconds)
    stop_loss_threshold = args.stop_loss_bp / 10000.0
    trade_dates = _discover_trade_dates(args.cache_root, args.start_date, args.end_date)
    if not trade_dates:
        raise SystemExit(f"No cached trade dates found under {args.cache_root}")

    tasks = [
        {
            "tradeDate": trade_date,
            "poolName": pool_name,
            "cacheRoot": str(args.cache_root),
            "openThreshold": args.open_threshold,
            "closeThreshold": args.close_threshold,
            "minHoldBars": args.min_hold_bars,
            "matchWindowSeconds": match_window_seconds,
            "stopLossMidRetThreshold": stop_loss_threshold,
            "StopLossSignalThreshold": args.stop_loss_signal_threshold,
        }
        for trade_date in trade_dates
        for pool_name in pools
    ]

    start = perf_counter()
    pool_rows: list[dict[str, Any]] = []
    close_type_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    processes = max(1, min(args.processes, len(tasks)))
    print(
        f"[start] tasks={len(tasks)} processes={processes} "
        f"stopLossBp={args.stop_loss_bp:g} stopLossSignalThreshold={args.stop_loss_signal_threshold:g}"
    )

    if processes == 1:
        result_iter = map(_run_one_pool, tasks)
        pool = None
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=processes)
        result_iter = pool.imap_unordered(_run_one_pool, tasks)

    try:
        completed = 0
        for result in result_iter:
            completed += 1
            pool_rows.extend(result.get("poolRows", []))
            close_type_rows.extend(result.get("closeTypeRows", []))
            status_rows.append({key: value for key, value in result.items() if key not in {"poolRows", "closeTypeRows"}})
            if completed % 20 == 0 or completed == len(tasks):
                print(f"[progress] completed={completed}/{len(tasks)} elapsedSeconds={perf_counter() - start:.1f}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    pool_df = pd.DataFrame(pool_rows)
    close_type_df = pd.DataFrame(close_type_rows)
    status_df = pd.DataFrame(status_rows)

    daily_all_pool_df = _aggregate_core_rows(
        pool_df,
        [
            "tradeDate",
            "variantTag",
            "openThreshold",
            "closeThreshold",
            "minHoldBars",
            "matchWindowSeconds",
        ],
    )
    total_df = _aggregate_core_rows(
        pool_df,
        [
            "variantTag",
            "openThreshold",
            "closeThreshold",
            "minHoldBars",
            "matchWindowSeconds",
        ],
    )

    close_type_total_df = pd.DataFrame()
    if not close_type_df.empty:
        close_type_total_df = (
            close_type_df.groupby(["variantTag", "closeType"], dropna=False, as_index=False)
            .agg(
                tradeCount=("tradeCount", "sum"),
                totalExecPnl=("totalExecPnl", "sum"),
                totalMatchedNotional=("totalMatchedNotional", "sum"),
            )
            .sort_values(["variantTag", "totalExecPnl"])
        )
        close_type_total_df["notionalWeightedExecRet"] = (
            close_type_total_df["totalExecPnl"] / close_type_total_df["totalMatchedNotional"]
        )

    compare_df = _build_comparison(total_df, args.baseline_total_path)

    elapsed = perf_counter() - start
    tag = (
        f"{args.start_date}_{args.end_date}_open{args.open_threshold:g}_close{args.close_threshold:g}_"
        f"hold{args.min_hold_bars}_match{_match_window_value(match_window_seconds)}_"
        f"stoploss{args.stop_loss_bp:g}bp_signal{args.stop_loss_signal_threshold:g}_hold{args.min_hold_bars}"
    )
    out_dir = args.output_root / tag
    mkdir_with_retry(out_dir)
    dataframe_to_csv_with_retry(pool_df, out_dir / "daily_pool_summary.csv", index=False)
    dataframe_to_csv_with_retry(daily_all_pool_df, out_dir / "daily_all_pool_summary.csv", index=False)
    dataframe_to_csv_with_retry(total_df, out_dir / "total_all_dates_summary.csv", index=False)
    dataframe_to_csv_with_retry(close_type_df, out_dir / "daily_pool_close_type_summary.csv", index=False)
    dataframe_to_csv_with_retry(close_type_total_df, out_dir / "close_type_total_summary.csv", index=False)
    dataframe_to_csv_with_retry(compare_df, out_dir / "baseline_vs_stop_loss_summary.csv", index=False)
    dataframe_to_csv_with_retry(status_df, out_dir / "task_status.csv", index=False)
    dataframe_to_csv_with_retry(
        pd.DataFrame(
            [
                {
                    "startDate": args.start_date,
                    "endDate": args.end_date,
                    "processes": processes,
                    "taskCount": len(tasks),
                    "elapsedSeconds": elapsed,
                    "stopLossBp": args.stop_loss_bp,
                    "stopLossSignalThreshold": args.stop_loss_signal_threshold,
                    "stopLossMinHoldBars": args.min_hold_bars,
                    "baselineTotalPath": str(args.baseline_total_path),
                }
            ]
        ),
        out_dir / "run_timing.csv",
        index=False,
    )

    print(f"[done] elapsedSeconds={elapsed:.1f}")
    print(f"[output] {out_dir}")
    if not compare_df.empty:
        print(compare_df.to_string(index=False))
    if not close_type_total_df.empty:
        print(close_type_total_df.to_string(index=False))


if __name__ == "__main__":
    main()
