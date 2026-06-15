from __future__ import annotations

import argparse
import gzip
import itertools
import multiprocessing as mp
import os
import pickle
from pathlib import Path
from time import perf_counter
import sys
import traceback
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from signal_trade_perf.internalization_backtest import (
    BacktestParams,
    get_default_ims_roots,
    load_internalization_day_inputs,
    run_internalization_prepared_day,
)
from signal_trade_perf.source_backtest import dataframe_to_csv_with_retry, mkdir_with_retry


POOL_NAMES = ["hs300", "zz500", "zz1000", "zz2000_1", "zz2000_2", "zz2000_3", "other"]
VARIANT_ORDER = ["all", "lt1000", "lt2000", "liqcap5tick", "poscap_min5", "poscap_avg5"]
CORE_SUM_COLUMNS = [
    "totalTradeCount",
    "totalExecPnl",
    "totalMatchedNotional",
    "totalClientAmt",
    "matchedClientAmt",
]
CORE_REPORT_COLUMNS = [
    "totalTradeCount",
    "totalExecPnl",
    "clientAmtMatchRate",
    "notionalWeightedExecRet",
    "byDateWinRate",
    "byDateRetMean",
    "byDateRetStd",
    "totalMatchedNotional",
]


def _parse_float_list(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _parse_match_windows(raw: str) -> list[int | None]:
    windows: list[int | None] = []
    for value in raw.split(","):
        token = value.strip().lower()
        if not token:
            continue
        if token in {"none", "unlimited", "all", "不限"}:
            windows.append(None)
        else:
            windows.append(int(token))
    return windows


def _parse_relaxed_close_specs(raw: str) -> list[tuple[int | None, int | None, float | None]]:
    token = raw.strip().lower()
    if token in {"", "none", "off"}:
        return [(None, None, None)]

    specs: list[tuple[int | None, int | None, float | None]] = []
    for value in raw.split(","):
        item = value.strip()
        if not item:
            continue
        if item.lower() in {"none", "off"}:
            specs.append((None, None, None))
            continue
        parts = item.split(":")
        if len(parts) == 3:
            hold_bars, after_bars, threshold = parts
            specs.append((int(hold_bars), int(after_bars), float(threshold)))
        elif len(parts) == 2:
            after_bars, threshold = parts
            specs.append((None, int(after_bars), float(threshold)))
        else:
            raise ValueError(f"Invalid relaxed close spec: {item}. Use afterBars:threshold or holdBars:afterBars:threshold.")
    return specs


def _relaxed_close_tag(after_bars: int | None, threshold: float | None) -> str:
    if after_bars is None or threshold is None:
        return ""
    return f"_relax_after_{after_bars}_close_{threshold:g}"


def _parse_pools(raw: str) -> list[str]:
    token = raw.strip().lower()
    if token in {"all", "*"}:
        return POOL_NAMES
    pools = [value.strip() for value in raw.split(",") if value.strip()]
    unknown_pools = sorted(set(pools) - set(POOL_NAMES))
    if unknown_pools:
        raise ValueError(f"Unknown pools: {unknown_pools}")
    return pools


def _parse_path_list(raw: str) -> list[Path]:
    return [Path(value.strip()) for value in raw.split(",") if value.strip()]


def _match_window_tag(match_window_seconds: int | None) -> str:
    return "match_unlimited" if match_window_seconds is None else f"match_{match_window_seconds}"


def _match_window_value(match_window_seconds: int | None) -> str:
    return "unlimited" if match_window_seconds is None else str(match_window_seconds)


def _combo_tag(params: BacktestParams, match_window_seconds: int | None) -> str:
    return f"{params.param_tag}_{_match_window_tag(match_window_seconds)}"


def _discover_ims_trade_dates(ims_roots: list[Path], start_date: str, end_date: str) -> list[str]:
    trade_dates: set[str] = set()
    for ims_root in ims_roots:
        if not ims_root.exists():
            continue
        trade_dates.update(path.name for path in ims_root.iterdir() if path.is_dir() and start_date <= path.name <= end_date)
    return sorted(trade_dates)


def _empty_result(trade_date: str, params: BacktestParams, match_window_seconds: int | None, elapsed: float, error: str) -> dict[str, Any]:
    return {
        "tradeDate": trade_date,
        "paramTag": params.param_tag,
        "openThreshold": params.open_threshold,
        "closeThreshold": params.close_threshold,
        "minHoldBars": params.min_hold_bars,
        "relaxedCloseAfterBars": params.relaxed_close_after_bars,
        "relaxedCloseThreshold": params.relaxed_close_threshold,
        "matchWindowSeconds": _match_window_value(match_window_seconds),
        "elapsedSeconds": elapsed,
        "status": "error",
        "error": error,
        "cacheHits": 0,
        "cacheMisses": 0,
        "cacheWrites": 0,
        "poolRows": [],
    }


def _cache_path(cache_root: Path, trade_date: str, pool_name: str) -> Path:
    return cache_root / trade_date / f"{pool_name}.pkl.gz"


def _load_cached_inputs(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with gzip.open(path, "rb") as file:
        return pickle.load(file)


def _write_cached_inputs(path: Path, prepared_inputs: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with gzip.open(tmp_path, "wb") as file:
        pickle.dump(prepared_inputs, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _run_one_date(task: dict[str, Any]) -> dict[str, Any]:
    trade_date = str(task["tradeDate"])
    params = BacktestParams(
        open_threshold=float(task["openThreshold"]),
        close_threshold=float(task["closeThreshold"]),
        min_hold_bars=int(task["minHoldBars"]),
        relaxed_close_after_bars=(
            None if task.get("relaxedCloseAfterBars") in {None, "", "none"} else int(task["relaxedCloseAfterBars"])
        ),
        relaxed_close_threshold=(
            None if task.get("relaxedCloseThreshold") in {None, "", "none"} else float(task["relaxedCloseThreshold"])
        ),
    )
    match_window_seconds = task["matchWindowSeconds"]
    if match_window_seconds == "unlimited":
        match_window_seconds = None
    else:
        match_window_seconds = int(match_window_seconds)

    start = perf_counter()
    pool_rows: list[dict[str, Any]] = []
    skipped_pools: list[str] = []
    cache_hits = 0
    cache_misses = 0
    cache_writes = 0
    try:
        ims_roots = [Path(path) for path in task["imsRoots"]]
        cache_mode = str(task["cacheMode"])
        cache_root = Path(str(task["cacheRoot"]))
        for pool_name in task["pools"]:
            prepared_inputs = None
            cache_file = _cache_path(cache_root, trade_date, pool_name)
            if cache_mode != "none" and cache_mode != "refresh":
                prepared_inputs = _load_cached_inputs(cache_file)
                if prepared_inputs is not None:
                    cache_hits += 1

            if prepared_inputs is None:
                cache_misses += 1
                prepared_inputs = load_internalization_day_inputs(
                    trade_date=trade_date,
                    pool_name=pool_name,
                    ims_roots=ims_roots,
                    profile=False,
                )
                if prepared_inputs is not None and cache_mode in {"readwrite", "refresh"}:
                    _write_cached_inputs(cache_file, prepared_inputs)
                    cache_writes += 1

            if prepared_inputs is None:
                skipped_pools.append(pool_name)
                continue

            _, _, _, pool_summary_df = run_internalization_prepared_day(
                prepared_inputs=prepared_inputs,
                params=params,
                match_window_seconds=match_window_seconds,
                profile=False,
            )
            if pool_summary_df.empty:
                skipped_pools.append(pool_name)
                continue

            pool_summary_df = pool_summary_df.copy()
            pool_summary_df["tradeDate"] = trade_date
            pool_summary_df["poolName"] = pool_name
            pool_summary_df["matchWindowSeconds"] = _match_window_value(match_window_seconds)
            pool_summary_df["relaxedCloseAfterBars"] = params.relaxed_close_after_bars
            pool_summary_df["relaxedCloseThreshold"] = params.relaxed_close_threshold
            pool_rows.extend(pool_summary_df.to_dict(orient="records"))

        return {
            "tradeDate": trade_date,
            "paramTag": params.param_tag,
            "openThreshold": params.open_threshold,
            "closeThreshold": params.close_threshold,
            "minHoldBars": params.min_hold_bars,
            "relaxedCloseAfterBars": params.relaxed_close_after_bars,
            "relaxedCloseThreshold": params.relaxed_close_threshold,
            "matchWindowSeconds": _match_window_value(match_window_seconds),
            "elapsedSeconds": perf_counter() - start,
            "status": "ok" if pool_rows else "skipped_empty_result",
            "skippedPools": ",".join(skipped_pools),
            "cacheHits": cache_hits,
            "cacheMisses": cache_misses,
            "cacheWrites": cache_writes,
            "poolRows": pool_rows,
        }
    except Exception:
        return _empty_result(
            trade_date=trade_date,
            params=params,
            match_window_seconds=match_window_seconds,
            elapsed=perf_counter() - start,
            error=traceback.format_exc(),
        )


def _aggregate_core_rows(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for key, group_df in df.groupby(group_cols, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_values))
        total_client_amt = float(group_df["totalClientAmt"].astype(float).sum())
        matched_client_amt = float(group_df["matchedClientAmt"].astype(float).sum())
        total_exec_pnl = float(group_df["totalExecPnl"].astype(float).sum())
        total_matched_notional = float(group_df["totalMatchedNotional"].astype(float).sum())
        row.update(
            {
                "totalTradeCount": int(group_df["totalTradeCount"].astype(int).sum()),
                "totalExecPnl": total_exec_pnl,
                "clientAmtMatchRate": float("nan") if total_client_amt == 0 else matched_client_amt / total_client_amt,
                "notionalWeightedExecRet": (
                    float("nan") if total_matched_notional == 0 else total_exec_pnl / total_matched_notional
                ),
                "totalMatchedNotional": total_matched_notional,
                "totalClientAmt": total_client_amt,
                "matchedClientAmt": matched_client_amt,
            }
        )
        rows.append(row)

    result_df = pd.DataFrame(rows)
    if "variantTag" in result_df.columns:
        variant_order = {variant_tag: idx for idx, variant_tag in enumerate(VARIANT_ORDER)}
        result_df["_variantOrder"] = result_df["variantTag"].map(variant_order).fillna(len(variant_order))
        sort_cols = [col for col in group_cols if col != "variantTag"] + ["_variantOrder"]
        result_df = result_df.sort_values(sort_cols).drop(columns=["_variantOrder"]).reset_index(drop=True)
    return result_df


def _add_by_date_return_metrics(total_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    if total_df.empty or daily_df.empty or "tradeDate" not in daily_df.columns:
        return total_df

    merge_cols = [col for col in total_df.columns if col in daily_df.columns and col not in CORE_SUM_COLUMNS]
    merge_cols = [
        col
        for col in merge_cols
        if col in {
            "variantTag",
            "paramTag",
            "openThreshold",
            "closeThreshold",
            "minHoldBars",
            "relaxedCloseAfterBars",
            "relaxedCloseThreshold",
            "matchWindowSeconds",
        }
    ]
    metric_rows: list[dict[str, Any]] = []
    for key, group_df in daily_df.groupby(merge_cols, sort=False, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        daily_ret = pd.to_numeric(group_df["notionalWeightedExecRet"], errors="coerce").dropna()
        row = dict(zip(merge_cols, key_values))
        row.update(
            {
                "byDateWinRate": float((daily_ret > 0).mean()) if len(daily_ret) else float("nan"),
                "byDateRetMean": float(daily_ret.mean()) if len(daily_ret) else float("nan"),
                "byDateRetStd": float(daily_ret.std(ddof=1)) if len(daily_ret) > 1 else float("nan"),
            }
        )
        metric_rows.append(row)

    metric_df = pd.DataFrame(metric_rows)
    if metric_df.empty:
        return total_df
    enriched_df = total_df.merge(metric_df, on=merge_cols, how="left")
    first_cols = [
        "variantTag",
        "paramTag",
        "openThreshold",
        "closeThreshold",
        "minHoldBars",
        "relaxedCloseAfterBars",
        "relaxedCloseThreshold",
        "matchWindowSeconds",
        "totalTradeCount",
        "totalExecPnl",
        "clientAmtMatchRate",
        "notionalWeightedExecRet",
        "byDateWinRate",
        "byDateRetMean",
        "byDateRetStd",
        "totalMatchedNotional",
    ]
    ordered_first = [col for col in first_cols if col in enriched_df.columns]
    return enriched_df[ordered_first + [col for col in enriched_df.columns if col not in ordered_first]]


def _print_daily_report(daily_summary_df: pd.DataFrame) -> None:
    if daily_summary_df.empty:
        return
    report_cols = ["tradeDate", "variantTag", *[col for col in CORE_REPORT_COLUMNS if col in daily_summary_df.columns]]
    print(daily_summary_df[report_cols].to_string(index=False))


def _write_combo_outputs(
    combo_dir: Path,
    pool_summary_df: pd.DataFrame,
    date_timing_df: pd.DataFrame,
    combo_timing_row: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mkdir_with_retry(combo_dir)
    daily_all_pool_df = _aggregate_core_rows(
        pool_summary_df,
        [
            "tradeDate",
            "variantTag",
            "paramTag",
            "openThreshold",
            "closeThreshold",
            "minHoldBars",
            "relaxedCloseAfterBars",
            "relaxedCloseThreshold",
            "matchWindowSeconds",
        ],
    )
    total_all_dates_df = _aggregate_core_rows(
        pool_summary_df,
        [
            "variantTag",
            "paramTag",
            "openThreshold",
            "closeThreshold",
            "minHoldBars",
            "relaxedCloseAfterBars",
            "relaxedCloseThreshold",
            "matchWindowSeconds",
        ],
    )
    total_all_dates_df = _add_by_date_return_metrics(total_all_dates_df, daily_all_pool_df)
    combo_timing_df = pd.DataFrame([combo_timing_row])

    dataframe_to_csv_with_retry(pool_summary_df, combo_dir / "daily_pool_summary.csv", index=False)
    dataframe_to_csv_with_retry(daily_all_pool_df, combo_dir / "daily_all_pool_summary.csv", index=False)
    dataframe_to_csv_with_retry(total_all_dates_df, combo_dir / "total_all_dates_summary.csv", index=False)
    dataframe_to_csv_with_retry(date_timing_df, combo_dir / "date_timing.csv", index=False)
    dataframe_to_csv_with_retry(combo_timing_df, combo_dir / "combo_timing.csv", index=False)
    return daily_all_pool_df, total_all_dates_df


def _write_date_checkpoint(combo_dir: Path, result: dict[str, Any], pool_rows: list[dict[str, Any]]) -> None:
    checkpoint_dir = combo_dir / "daily_checkpoints"
    mkdir_with_retry(checkpoint_dir)
    trade_date = str(result["tradeDate"])
    pool_summary_df = pd.DataFrame(pool_rows)
    daily_all_pool_df = _aggregate_core_rows(
        pool_summary_df,
        [
            "tradeDate",
            "variantTag",
            "paramTag",
            "openThreshold",
            "closeThreshold",
            "minHoldBars",
            "relaxedCloseAfterBars",
            "relaxedCloseThreshold",
            "matchWindowSeconds",
        ],
    )
    timing_df = pd.DataFrame([{key: value for key, value in result.items() if key != "poolRows"}])
    dataframe_to_csv_with_retry(pool_summary_df, checkpoint_dir / f"{trade_date}_pool_summary.csv", index=False)
    dataframe_to_csv_with_retry(daily_all_pool_df, checkpoint_dir / f"{trade_date}_all_pool_summary.csv", index=False)
    dataframe_to_csv_with_retry(timing_df, checkpoint_dir / f"{trade_date}_timing.csv", index=False)


def _write_progress_status(output_root: Path, progress_rows: list[dict[str, Any]]) -> None:
    if not progress_rows:
        return
    progress_df = pd.DataFrame(progress_rows)
    first_cols = [
        "comboIndex",
        "comboCount",
        "comboTag",
        "status",
        "completedDateCount",
        "tradeDateCount",
        "latestTradeDate",
        "latestWriteTime",
        "okDateCount",
        "errorDateCount",
        "elapsedSeconds",
    ]
    ordered_first = [col for col in first_cols if col in progress_df.columns]
    dataframe_to_csv_with_retry(
        progress_df[ordered_first + [col for col in progress_df.columns if col not in ordered_first]],
        output_root / "progress_status.csv",
        index=False,
    )


def _read_date_checkpoint(combo_dir: Path, trade_date: str) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    checkpoint_dir = combo_dir / "daily_checkpoints"
    pool_summary_path = checkpoint_dir / f"{trade_date}_pool_summary.csv"
    timing_path = checkpoint_dir / f"{trade_date}_timing.csv"
    if not pool_summary_path.exists() or not timing_path.exists():
        return None

    pool_summary_df = pd.read_csv(pool_summary_path)
    timing_df = pd.read_csv(timing_path)
    timing_row = timing_df.iloc[0].to_dict() if not timing_df.empty else {"tradeDate": trade_date, "status": "checkpoint"}
    return pool_summary_df.to_dict(orient="records"), timing_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run date-parallel internalization parameter sweep.")
    parser.add_argument("--start-date", default="20260105")
    parser.add_argument("--end-date", default="20260331")
    parser.add_argument("--processes", type=int, default=10)
    parser.add_argument("--open-thresholds", default="6,7.5")
    parser.add_argument("--close-thresholds", default="4,6")
    parser.add_argument("--min-hold-bars-list", default="20,30")
    parser.add_argument(
        "--relaxed-close-specs",
        default="none",
        help="Comma list of afterBars:threshold or holdBars:afterBars:threshold. Use none/off to disable.",
    )
    parser.add_argument("--match-window-seconds-list", default="5,10,unlimited")
    parser.add_argument("--pools", default="all")
    parser.add_argument(
        "--ims-roots",
        default="",
        help="Comma-separated IMS root directories. Default: all subdirectories under PROJECT_ROOT/ims_backtest_data.",
    )
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results" / "internalization_backtest" / "param_sweep_date_mp"))
    parser.add_argument("--cache-mode", choices=["none", "readwrite", "refresh"], default="readwrite")
    parser.add_argument("--cache-root", default=str(PROJECT_ROOT / "results" / "internalization_backtest" / "data_cache"))
    parser.add_argument("--resume", action="store_true", help="Reuse completed daily checkpoints for the same combo output directory.")
    args = parser.parse_args()

    ims_roots = _parse_path_list(args.ims_roots) if args.ims_roots.strip() else get_default_ims_roots(PROJECT_ROOT)
    ims_roots = [path.resolve() for path in ims_roots]
    trade_dates = _discover_ims_trade_dates(ims_roots, args.start_date, args.end_date)
    if not trade_dates:
        raise ValueError(f"No IMS trade dates found in [{args.start_date}, {args.end_date}]")

    output_root = Path(args.output_root) / f"{trade_dates[0]}_{trade_dates[-1]}"
    mkdir_with_retry(output_root)
    pools = _parse_pools(args.pools)

    relaxed_specs = _parse_relaxed_close_specs(args.relaxed_close_specs)
    param_grid = []
    for open_threshold, close_threshold, min_hold_bars, match_window_seconds in itertools.product(
        _parse_float_list(args.open_thresholds),
        _parse_float_list(args.close_thresholds),
        _parse_int_list(args.min_hold_bars_list),
        _parse_match_windows(args.match_window_seconds_list),
    ):
        for spec_hold_bars, relaxed_after_bars, relaxed_threshold in relaxed_specs:
            if spec_hold_bars is not None and int(spec_hold_bars) != int(min_hold_bars):
                continue
            param_grid.append(
                (
                    open_threshold,
                    close_threshold,
                    min_hold_bars,
                    match_window_seconds,
                    relaxed_after_bars,
                    relaxed_threshold,
                )
            )

    print(f"tradeDateCount={len(trade_dates)}")
    print(f"paramComboCount={len(param_grid)}")
    print(f"processes={args.processes}")
    print(f"pools={','.join(pools)}")
    print(f"imsRoots={','.join(str(path) for path in ims_roots)}")
    print(f"cacheMode={args.cache_mode}")
    print(f"cacheRoot={args.cache_root}")
    print(f"outputRoot={output_root}")

    all_daily_summaries: list[pd.DataFrame] = []
    all_total_summaries: list[pd.DataFrame] = []
    all_combo_timing_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    total_start = perf_counter()

    context = mp.get_context("spawn")
    for combo_idx, (
        open_threshold,
        close_threshold,
        min_hold_bars,
        match_window_seconds,
        relaxed_after_bars,
        relaxed_threshold,
    ) in enumerate(param_grid, start=1):
        params = BacktestParams(
            open_threshold=open_threshold,
            close_threshold=close_threshold,
            min_hold_bars=min_hold_bars,
            relaxed_close_after_bars=relaxed_after_bars,
            relaxed_close_threshold=relaxed_threshold,
        )
        combo_tag = _combo_tag(params, match_window_seconds)
        combo_dir = output_root / combo_tag
        combo_start = perf_counter()
        print(f"[combo {combo_idx}/{len(param_grid)}] start {combo_tag}")
        progress_row = {
            "comboIndex": combo_idx,
            "comboCount": len(param_grid),
            "comboTag": combo_tag,
            "status": "running",
            "completedDateCount": 0,
            "tradeDateCount": len(trade_dates),
            "latestTradeDate": "",
            "latestWriteTime": "",
            "okDateCount": 0,
            "errorDateCount": 0,
            "elapsedSeconds": 0.0,
        }
        progress_rows.append(progress_row)
        _write_progress_status(output_root, progress_rows)

        pool_rows: list[dict[str, Any]] = []
        timing_rows: list[dict[str, Any]] = []
        tasks = [
            {
                "tradeDate": trade_date,
                "openThreshold": open_threshold,
                "closeThreshold": close_threshold,
                "minHoldBars": min_hold_bars,
                "relaxedCloseAfterBars": relaxed_after_bars,
                "relaxedCloseThreshold": relaxed_threshold,
                "matchWindowSeconds": _match_window_value(match_window_seconds),
                "imsRoots": [str(path) for path in ims_roots],
                "pools": pools,
                "cacheMode": args.cache_mode,
                "cacheRoot": args.cache_root,
            }
            for trade_date in trade_dates
        ]
        if args.resume:
            pending_tasks: list[dict[str, Any]] = []
            for task in tasks:
                trade_date = str(task["tradeDate"])
                checkpoint = _read_date_checkpoint(combo_dir, trade_date)
                if checkpoint is None:
                    pending_tasks.append(task)
                    continue
                checkpoint_pool_rows, checkpoint_timing_row = checkpoint
                pool_rows.extend(checkpoint_pool_rows)
                timing_rows.append(checkpoint_timing_row)
                progress_row["completedDateCount"] = len(timing_rows)
                progress_row["latestTradeDate"] = trade_date
                progress_row["latestWriteTime"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                progress_row["okDateCount"] = sum(1 for row in timing_rows if row.get("status") == "ok")
                progress_row["errorDateCount"] = sum(1 for row in timing_rows if row.get("status") == "error")
                progress_row["elapsedSeconds"] = perf_counter() - combo_start
                _write_progress_status(output_root, progress_rows)
                print(f"[combo {combo_idx}/{len(param_grid)}] [resume] {trade_date} loaded checkpoint")
            tasks = pending_tasks

        with context.Pool(processes=args.processes) as pool:
            for result in pool.imap_unordered(_run_one_date, tasks):
                date_pool_rows = result["poolRows"]
                _write_date_checkpoint(combo_dir, result, date_pool_rows)
                pool_rows.extend(date_pool_rows)
                result = {key: value for key, value in result.items() if key != "poolRows"}
                timing_rows.append(result)
                progress_row["completedDateCount"] = len(timing_rows)
                progress_row["latestTradeDate"] = result["tradeDate"]
                progress_row["latestWriteTime"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                progress_row["okDateCount"] = sum(1 for row in timing_rows if row.get("status") == "ok")
                progress_row["errorDateCount"] = sum(1 for row in timing_rows if row.get("status") == "error")
                progress_row["elapsedSeconds"] = perf_counter() - combo_start
                _write_progress_status(output_root, progress_rows)
                print(
                    f"[combo {combo_idx}/{len(param_grid)}] "
                    f"[date {len(timing_rows)}/{len(trade_dates)}] {result['tradeDate']} "
                    f"status={result['status']} elapsed={float(result['elapsedSeconds']):.2f}s "
                    f"cacheHits={result['cacheHits']} cacheMisses={result['cacheMisses']} cacheWrites={result['cacheWrites']}"
                )

        pool_summary_df = pd.DataFrame(pool_rows)
        date_timing_df = pd.DataFrame(timing_rows).sort_values("tradeDate").reset_index(drop=True)
        combo_elapsed = perf_counter() - combo_start
        combo_timing_row = {
            "comboTag": combo_tag,
            "paramTag": params.param_tag,
            "openThreshold": open_threshold,
            "closeThreshold": close_threshold,
            "minHoldBars": min_hold_bars,
            "relaxedCloseAfterBars": relaxed_after_bars,
            "relaxedCloseThreshold": relaxed_threshold,
            "matchWindowSeconds": _match_window_value(match_window_seconds),
            "cacheMode": args.cache_mode,
            "cacheRoot": args.cache_root,
            "tradeDateCount": len(trade_dates),
            "okDateCount": int((date_timing_df["status"] == "ok").sum()) if not date_timing_df.empty else 0,
            "errorDateCount": int((date_timing_df["status"] == "error").sum()) if not date_timing_df.empty else 0,
            "elapsedSeconds": combo_elapsed,
        }
        daily_all_pool_df, total_all_dates_df = _write_combo_outputs(
            combo_dir=combo_dir,
            pool_summary_df=pool_summary_df,
            date_timing_df=date_timing_df,
            combo_timing_row=combo_timing_row,
        )
        all_daily_summaries.append(daily_all_pool_df)
        all_total_summaries.append(total_all_dates_df.assign(comboTag=combo_tag))
        all_combo_timing_rows.append(combo_timing_row)

        print(f"[combo {combo_idx}/{len(param_grid)}] done {combo_tag} elapsedSeconds={combo_elapsed:.2f}")
        progress_row["status"] = "done"
        progress_row["completedDateCount"] = len(timing_rows)
        progress_row["latestWriteTime"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        progress_row["okDateCount"] = int((date_timing_df["status"] == "ok").sum()) if not date_timing_df.empty else 0
        progress_row["errorDateCount"] = int((date_timing_df["status"] == "error").sum()) if not date_timing_df.empty else 0
        progress_row["elapsedSeconds"] = combo_elapsed
        _write_progress_status(output_root, progress_rows)
        _print_daily_report(total_all_dates_df.assign(tradeDate="ALL_DATES"))

    combined_daily_df = pd.concat(all_daily_summaries, ignore_index=True) if all_daily_summaries else pd.DataFrame()
    combined_total_df = pd.concat(all_total_summaries, ignore_index=True) if all_total_summaries else pd.DataFrame()
    combo_timing_df = pd.DataFrame(all_combo_timing_rows)
    run_timing_df = pd.DataFrame(
        [
            {
                "startDate": trade_dates[0],
                "endDate": trade_dates[-1],
                "tradeDateCount": len(trade_dates),
                "paramComboCount": len(param_grid),
                "processes": args.processes,
                "cacheMode": args.cache_mode,
                "cacheRoot": args.cache_root,
                "elapsedSeconds": perf_counter() - total_start,
            }
        ]
    )

    dataframe_to_csv_with_retry(combined_daily_df, output_root / "combined_daily_all_pool_summary.csv", index=False)
    dataframe_to_csv_with_retry(combined_total_df, output_root / "combined_total_all_dates_summary.csv", index=False)
    dataframe_to_csv_with_retry(combo_timing_df, output_root / "combined_combo_timing.csv", index=False)
    dataframe_to_csv_with_retry(run_timing_df, output_root / "run_timing.csv", index=False)

    print(f"totalElapsedSeconds={float(run_timing_df['elapsedSeconds'].iloc[0]):.2f}")
    print(f"resultDir={output_root}")


if __name__ == "__main__":
    main()
