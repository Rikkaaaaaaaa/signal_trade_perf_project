from __future__ import annotations

import argparse
import gzip
import heapq
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

from signal_trade_perf.internalization_backtest import BacktestParams, run_internalization_prepared_day
from signal_trade_perf.source_backtest import dataframe_to_csv_with_retry, mkdir_with_retry


POOL_NAMES = ["hs300", "zz500", "zz1000", "zz2000_1", "zz2000_2", "zz2000_3", "other"]
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "results" / "internalization_backtest" / "data_cache"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "internalization_backtest" / "diagnostics"


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


def _match_window_tag(match_window_seconds: int | None) -> str:
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


def _select_worst(df: pd.DataFrame, top_n: int) -> list[dict[str, Any]]:
    if df.empty or "execPnl" not in df.columns:
        return []
    worst_df = df.nsmallest(top_n, "execPnl").copy()
    return worst_df.to_dict(orient="records")


def _run_one_pool(task: dict[str, Any]) -> dict[str, Any]:
    trade_date = str(task["tradeDate"])
    pool_name = str(task["poolName"])
    cache_root = Path(str(task["cacheRoot"]))
    params = BacktestParams(
        open_threshold=float(task["openThreshold"]),
        close_threshold=float(task["closeThreshold"]),
        min_hold_bars=int(task["minHoldBars"]),
    )
    match_window_seconds = task["matchWindowSeconds"]
    top_n = int(task["topN"])

    start = perf_counter()
    try:
        cache_file = _cache_path(cache_root, trade_date, pool_name)
        if not cache_file.exists():
            return {
                "tradeDate": trade_date,
                "poolName": pool_name,
                "status": "missing_cache",
                "elapsedSeconds": perf_counter() - start,
                "rows": [],
            }

        prepared_inputs = _load_cached_inputs(cache_file)
        _, trades_df, _, pool_summary_df = run_internalization_prepared_day(
            prepared_inputs=prepared_inputs,
            params=params,
            match_window_seconds=match_window_seconds,
            profile=False,
        )
        if trades_df.empty:
            return {
                "tradeDate": trade_date,
                "poolName": pool_name,
                "status": "empty_trades",
                "elapsedSeconds": perf_counter() - start,
                "rows": [],
            }

        trades_df = trades_df.copy()
        trades_df["tradeDate"] = trade_date
        trades_df["poolName"] = pool_name
        trades_df["variantTag"] = "all"
        trades_df["openThreshold"] = params.open_threshold
        trades_df["closeThreshold"] = params.close_threshold
        trades_df["minHoldBars"] = params.min_hold_bars
        trades_df["matchWindowSeconds"] = _match_window_tag(match_window_seconds)

        if not pool_summary_df.empty:
            all_summary = pool_summary_df[pool_summary_df["variantTag"] == "all"]
            if not all_summary.empty:
                trades_df["poolTotalExecPnl"] = float(all_summary.iloc[0]["totalExecPnl"])
                trades_df["poolTotalTradeCount"] = int(all_summary.iloc[0]["totalTradeCount"])

        return {
            "tradeDate": trade_date,
            "poolName": pool_name,
            "status": "ok",
            "elapsedSeconds": perf_counter() - start,
            "rows": _select_worst(trades_df, top_n),
        }
    except Exception:
        return {
            "tradeDate": trade_date,
            "poolName": pool_name,
            "status": "error",
            "elapsedSeconds": perf_counter() - start,
            "error": traceback.format_exc(),
            "rows": [],
        }


def _push_worst(heap: list[tuple[float, int, dict[str, Any]]], rows: list[dict[str, Any]], top_n: int, counter_start: int) -> int:
    counter = counter_start
    for row in rows:
        exec_pnl = row.get("execPnl")
        if pd.isna(exec_pnl):
            continue
        item = (float(exec_pnl), counter, row)
        counter += 1
        if len(heap) < top_n:
            heapq.heappush(heap, (-item[0], item[1], item[2]))
            continue
        if item[0] < -heap[0][0]:
            heapq.heapreplace(heap, (-item[0], item[1], item[2]))
    return counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export worst losing internalization trades from cached day/pool inputs.")
    parser.add_argument("--start-date", default="20260105")
    parser.add_argument("--end-date", default="20260331")
    parser.add_argument("--pools", default="all")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--open-threshold", type=float, default=6.0)
    parser.add_argument("--close-threshold", type=float, default=6.0)
    parser.add_argument("--min-hold-bars", type=int, default=30)
    parser.add_argument("--match-window-seconds", default="unlimited")
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--processes", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pools = _parse_pools(args.pools)
    match_window_seconds = _parse_match_window(args.match_window_seconds)
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
            "topN": args.top_n,
        }
        for trade_date in trade_dates
        for pool_name in pools
    ]

    start = perf_counter()
    heap: list[tuple[float, int, dict[str, Any]]] = []
    counter = 0
    status_rows: list[dict[str, Any]] = []
    processes = max(1, min(args.processes, len(tasks)))
    print(f"[start] tasks={len(tasks)} processes={processes} topN={args.top_n}")

    if processes == 1:
        result_iter = map(_run_one_pool, tasks)
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=processes)
        result_iter = pool.imap_unordered(_run_one_pool, tasks)

    try:
        completed = 0
        for result in result_iter:
            completed += 1
            status_rows.append({key: value for key, value in result.items() if key != "rows"})
            counter = _push_worst(heap, result.get("rows", []), args.top_n, counter)
            if completed % 20 == 0 or completed == len(tasks):
                elapsed = perf_counter() - start
                print(f"[progress] completed={completed}/{len(tasks)} elapsedSeconds={elapsed:.1f}")
    finally:
        if processes != 1:
            pool.close()
            pool.join()

    worst_rows = [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]
    worst_df = pd.DataFrame(worst_rows)
    status_df = pd.DataFrame(status_rows)

    out_dir = args.output_dir
    mkdir_with_retry(out_dir)
    tag = (
        f"{args.start_date}_{args.end_date}_"
        f"open{args.open_threshold:g}_close{args.close_threshold:g}_hold{args.min_hold_bars}_"
        f"match{_match_window_tag(match_window_seconds)}"
    )
    worst_path = out_dir / f"{tag}_worst_loss_trades_top{args.top_n}.csv"
    status_path = out_dir / f"{tag}_worst_loss_trades_status.csv"
    dataframe_to_csv_with_retry(worst_df, worst_path, index=False)
    dataframe_to_csv_with_retry(status_df, status_path, index=False)

    display_cols = [
        "tradeDate",
        "poolName",
        "securityCode",
        "side",
        "clientSide",
        "clientQty",
        "openTime",
        "closeTime",
        "openSignal",
        "closeSignal",
        "matchDelaySeconds",
        "openMid",
        "closeBid1",
        "closeAsk1",
        "execPnl",
        "execRet",
        "openNotional",
        "closeType",
        "holdBars",
        "poolTotalExecPnl",
    ]
    existing_cols = [col for col in display_cols if col in worst_df.columns]
    print(f"[done] elapsedSeconds={perf_counter() - start:.1f}")
    print(f"[output] {worst_path}")
    print(f"[status] {status_path}")
    if not worst_df.empty:
        print(worst_df[existing_cols].head(min(30, len(worst_df))).to_string(index=False))


if __name__ == "__main__":
    main()
