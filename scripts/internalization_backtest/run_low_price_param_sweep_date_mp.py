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

import numpy as np
import pandas as pd

from signal_trade_perf.internalization_backtest import get_default_ims_roots
from signal_trade_perf.low_price_internalization import LOW_PRICE_VARIANTS, LowPriceBacktestParams, run_low_price_prepared_day
from signal_trade_perf.low_price_internalization_data import load_low_price_day_inputs
from signal_trade_perf.source_backtest import dataframe_to_csv_with_retry, mkdir_with_retry


# 低价股挂单策略扫参入口：日期级 multiprocessing，每个日期任务内部串行跑多个 pool。
# 这样和高价股 run_param_sweep_date_mp.py 的并行粒度保持一致，避免多个进程同时写同一天/同参数文件。
VARIANT_ORDER = LOW_PRICE_VARIANTS
POOL_NAMES = ["hs300", "zz500", "zz1000", "zz2000_1", "zz2000_2", "zz2000_3", "other"]
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
    "maxCapitalUsed",
    "p95CapitalUsedByEvent",
    "maxDailyCapitalUsed",
    "p95DailyCapitalUsed",
    "avgDailyCapitalUsed",
    "capitalAdjustedReturn",
    "clientAmtMatchRate",
    "notionalWeightedExecRet",
    "byDateWinRate",
    "byDateRetMean",
    "byDateRetStd",
    "yTestWinRate",
    "totalMatchedNotional",
]


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


def _parse_float_list(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _parse_signal_rank_sets(raw: str) -> list[tuple[int, ...]]:
    # 分号分隔多组参数，逗号分隔同一组 exact ranks：例如 "1;1,2"。
    rank_sets: list[tuple[int, ...]] = []
    for item in raw.split(";"):
        token = item.strip()
        if not token:
            continue
        rank_sets.append(tuple(int(value.strip()) for value in token.split(",") if value.strip()))
    return rank_sets


def _parse_path_list(raw: str) -> list[Path]:
    return [Path(value.strip()) for value in raw.split(",") if value.strip()]


def _parse_pools(raw: str) -> list[str]:
    token = raw.strip().lower()
    if token in {"all", "*"}:
        return POOL_NAMES
    pools = [value.strip() for value in raw.split(",") if value.strip()]
    unknown_pools = sorted(set(pools) - set(POOL_NAMES))
    if unknown_pools:
        raise ValueError(f"Unknown pools: {unknown_pools}")
    return pools


def _match_window_value(match_window_seconds: int | None) -> str:
    return "unlimited" if match_window_seconds is None else str(match_window_seconds)


def _combo_tag(params: LowPriceBacktestParams) -> str:
    return params.param_tag


def _discover_ims_trade_dates(ims_roots: list[Path], start_date: str, end_date: str) -> list[str]:
    trade_dates: set[str] = set()
    for ims_root in ims_roots:
        if not ims_root.exists():
            continue
        trade_dates.update(path.name for path in ims_root.iterdir() if path.is_dir() and start_date <= path.name <= end_date)
    return sorted(trade_dates)


def _cache_path(cache_root: Path, trade_date: str, pool_name: str) -> Path:
    # cache 按 date/pool 存放，因为这些数据和 signal rank、match window、spread 参数无关。
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


def _empty_result(trade_date: str, params: LowPriceBacktestParams, elapsed: float, error: str) -> dict[str, Any]:
    return {
        "tradeDate": trade_date,
        "paramTag": params.param_tag,
        "signalRanks": ",".join(str(rank) for rank in params.signal_ranks),
        "matchWindowSeconds": _match_window_value(params.match_window_seconds),
        "spread": params.spread,
        "elapsedSeconds": elapsed,
        "status": "error",
        "error": error,
        "cacheHits": 0,
        "cacheMisses": 0,
        "cacheWrites": 0,
        "skippedPools": "",
        "summaryRows": [],
    }


def _run_one_date(task: dict[str, Any]) -> dict[str, Any]:
    # worker 粒度是单个交易日；同一天内循环所有 pool，再返回给主进程统一写 checkpoint。
    trade_date = str(task["tradeDate"])
    params = LowPriceBacktestParams(
        signal_ranks=tuple(int(value) for value in task["signalRanks"]),
        match_window_seconds=(None if task["matchWindowSeconds"] == "unlimited" else int(task["matchWindowSeconds"])),
        spread=float(task["spread"]),
    )
    start = perf_counter()
    cache_hits = 0
    cache_misses = 0
    cache_writes = 0
    summary_rows: list[dict[str, Any]] = []
    skipped_pools: list[str] = []
    try:
        cache_mode = str(task["cacheMode"])
        cache_root = Path(str(task["cacheRoot"]))
        for pool_name in task["pools"]:
            # 每个 pool 的 prepared input 单独缓存，方便之后只删除某个 pool 或某天缓存。
            cache_file = _cache_path(cache_root, trade_date, pool_name)
            prepared_inputs = None
            if cache_mode not in {"none", "refresh"}:
                prepared_inputs = _load_cached_inputs(cache_file)
                if prepared_inputs is not None:
                    cache_hits += 1

            if prepared_inputs is None:
                cache_misses += 1
                prepared_inputs = load_low_price_day_inputs(
                    trade_date=trade_date,
                    ims_roots=[Path(path) for path in task["imsRoots"]],
                    pool_name=pool_name,
                )
                if prepared_inputs is not None and cache_mode in {"readwrite", "refresh"}:
                    _write_cached_inputs(cache_file, prepared_inputs)
                    cache_writes += 1

            if prepared_inputs is None:
                skipped_pools.append(pool_name)
                continue

            _, _, summary_df = run_low_price_prepared_day(prepared_inputs=prepared_inputs, params=params)
            pool_summary_rows = summary_df.to_dict(orient="records") if not summary_df.empty else []
            if not pool_summary_rows:
                skipped_pools.append(pool_name)
                continue

            for row in pool_summary_rows:
                row["tradeDate"] = trade_date
                row["poolName"] = pool_name
            summary_rows.extend(pool_summary_rows)

        status = "ok" if summary_rows else "skipped_empty_result"
        return {
            "tradeDate": trade_date,
            "paramTag": params.param_tag,
            "signalRanks": ",".join(str(rank) for rank in params.signal_ranks),
            "matchWindowSeconds": _match_window_value(params.match_window_seconds),
            "spread": params.spread,
            "elapsedSeconds": perf_counter() - start,
            "status": status,
            "cacheHits": cache_hits,
            "cacheMisses": cache_misses,
            "cacheWrites": cache_writes,
            "skippedPools": ",".join(skipped_pools),
            "summaryRows": summary_rows,
        }
    except Exception:
        return _empty_result(trade_date=trade_date, params=params, elapsed=perf_counter() - start, error=traceback.format_exc())


def _aggregate_total(daily_df: pd.DataFrame) -> pd.DataFrame:
    # total_summary 聚合所有日期和所有 pool；资金占用先按日算 maxCapitalUsed，再对日度值做 max/p95/mean。
    if daily_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["variantTag", "paramTag", "signalRanks", "matchWindowSeconds", "spread"]
    for key, group_df in daily_df.groupby(group_cols, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_values))
        total_pnl = float(group_df["totalExecPnl"].astype(float).sum())
        total_notional = float(group_df["totalMatchedNotional"].astype(float).sum())
        total_client_amt = float(group_df["totalClientAmt"].astype(float).sum())
        matched_client_amt = float(group_df["matchedClientAmt"].astype(float).sum())
        daily_capital = pd.to_numeric(group_df["maxCapitalUsed"], errors="coerce").dropna()
        max_daily_capital = float(daily_capital.max()) if len(daily_capital) else 0.0
        row.update(
            {
                "tradeDateCount": int(group_df["tradeDate"].nunique()),
                "totalTradeCount": int(group_df["totalTradeCount"].astype(int).sum()),
                "totalExecPnl": total_pnl,
                "clientAmtMatchRate": np.nan if total_client_amt == 0 else matched_client_amt / total_client_amt,
                "notionalWeightedExecRet": np.nan if total_notional == 0 else total_pnl / total_notional,
                "yTestWinRate": float(np.average(group_df["yTestWinRate"].fillna(0), weights=group_df["totalTradeCount"].clip(lower=0))) if int(group_df["totalTradeCount"].sum()) > 0 else np.nan,
                "maxDailyCapitalUsed": max_daily_capital,
                "p95DailyCapitalUsed": float(daily_capital.quantile(0.95)) if len(daily_capital) else 0.0,
                "avgDailyCapitalUsed": float(daily_capital.mean()) if len(daily_capital) else 0.0,
                "capitalAdjustedReturn": np.nan if max_daily_capital == 0 else total_pnl / max_daily_capital,
                "totalMatchedNotional": total_notional,
                "totalClientAmt": total_client_amt,
                "matchedClientAmt": matched_client_amt,
            }
        )
        rows.append(row)
    result_df = pd.DataFrame(rows)
    return _add_by_date_return_metrics(result_df, daily_df)


def _add_by_date_return_metrics(total_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    if total_df.empty or daily_df.empty or "tradeDate" not in daily_df.columns:
        return total_df

    merge_cols = [
        col
        for col in ["variantTag", "paramTag", "signalRanks", "matchWindowSeconds", "spread"]
        if col in total_df.columns and col in daily_df.columns
    ]
    metric_rows: list[dict[str, Any]] = []
    for key, group_df in daily_df.groupby(merge_cols, sort=False, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        daily_ret = pd.to_numeric(group_df["notionalWeightedExecRet"], errors="coerce").dropna()
        row = dict(zip(merge_cols, key_values))
        row.update(
            {
                "byDateWinRate": float((daily_ret > 0).mean()) if len(daily_ret) else np.nan,
                "byDateRetMean": float(daily_ret.mean()) if len(daily_ret) else np.nan,
                "byDateRetStd": float(daily_ret.std(ddof=1)) if len(daily_ret) > 1 else np.nan,
            }
        )
        metric_rows.append(row)

    metric_df = pd.DataFrame(metric_rows)
    if metric_df.empty:
        return total_df
    enriched_df = total_df.merge(metric_df, on=merge_cols, how="left")
    return _order_report_columns(enriched_df, id_cols=merge_cols)


def _order_report_columns(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    ordered_first = [col for col in [*id_cols, *CORE_REPORT_COLUMNS] if col in df.columns]
    return df[ordered_first + [col for col in df.columns if col not in ordered_first]]


def _print_total_report(total_df: pd.DataFrame) -> None:
    if total_df.empty:
        return
    report_cols = ["variantTag", *[col for col in CORE_REPORT_COLUMNS if col in total_df.columns]]
    print(total_df[report_cols].to_string(index=False))


def _write_date_checkpoint(combo_dir: Path, result: dict[str, Any]) -> None:
    # 每跑完一天立刻写 checkpoint，长任务中断后可以用 --resume 继续。
    checkpoint_dir = combo_dir / "daily_checkpoints"
    mkdir_with_retry(checkpoint_dir)
    trade_date = str(result["tradeDate"])
    summary_df = pd.DataFrame(result.get("summaryRows", []))
    timing_df = pd.DataFrame([{key: value for key, value in result.items() if key != "summaryRows"}])
    dataframe_to_csv_with_retry(summary_df, checkpoint_dir / f"{trade_date}_summary.csv", index=False)
    dataframe_to_csv_with_retry(timing_df, checkpoint_dir / f"{trade_date}_timing.csv", index=False)


def _read_date_checkpoint(combo_dir: Path, trade_date: str) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    checkpoint_dir = combo_dir / "daily_checkpoints"
    summary_path = checkpoint_dir / f"{trade_date}_summary.csv"
    timing_path = checkpoint_dir / f"{trade_date}_timing.csv"
    if not summary_path.exists() or not timing_path.exists():
        return None
    summary_df = pd.read_csv(summary_path)
    timing_df = pd.read_csv(timing_path)
    timing_row = timing_df.iloc[0].to_dict() if not timing_df.empty else {"tradeDate": trade_date, "status": "checkpoint"}
    return summary_df.to_dict(orient="records"), timing_row


def _write_progress_status(output_root: Path, progress_rows: list[dict[str, Any]]) -> None:
    if not progress_rows:
        return
    dataframe_to_csv_with_retry(pd.DataFrame(progress_rows), output_root / "progress_status.csv", index=False)


def main() -> None:
    # 主入口只负责任务编排和 CSV 落盘；真实撮合逻辑在 low_price_internalization.py。
    parser = argparse.ArgumentParser(description="Run date-parallel low-price internalization parameter sweep.")
    parser.add_argument("--start-date", default="20260105")
    parser.add_argument("--end-date", default="20260331")
    parser.add_argument("--processes", type=int, default=10)
    parser.add_argument("--pools", default="all")
    parser.add_argument("--signal-ranks-list", default="1,2", help='Semicolon-separated rank sets, e.g. "1,2;1;1,2,10".')
    parser.add_argument("--match-window-seconds-list", default="10")
    parser.add_argument("--spreads", default="0.01")
    parser.add_argument("--ims-roots", default="")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results" / "internalization_backtest" / "low_price_param_sweep_date_mp"))
    parser.add_argument("--cache-mode", choices=["none", "readwrite", "refresh"], default="readwrite")
    parser.add_argument("--cache-root", default=str(PROJECT_ROOT / "results" / "internalization_backtest" / "low_price_data_cache"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    ims_roots = _parse_path_list(args.ims_roots) if args.ims_roots.strip() else get_default_ims_roots(PROJECT_ROOT)
    ims_roots = [path.resolve() for path in ims_roots]
    trade_dates = _discover_ims_trade_dates(ims_roots, args.start_date, args.end_date)
    if not trade_dates:
        raise ValueError(f"No IMS trade dates found in [{args.start_date}, {args.end_date}]")

    output_root = Path(args.output_root) / f"{trade_dates[0]}_{trade_dates[-1]}"
    mkdir_with_retry(output_root)
    pools = _parse_pools(args.pools)

    param_grid = list(itertools.product(
        _parse_signal_rank_sets(args.signal_ranks_list),
        _parse_match_windows(args.match_window_seconds_list),
        _parse_float_list(args.spreads),
    ))
    print(f"tradeDateCount={len(trade_dates)}")
    print(f"paramComboCount={len(param_grid)}")
    print(f"processes={args.processes}")
    print(f"pools={','.join(pools)}")
    print(f"cacheMode={args.cache_mode}")
    print(f"outputRoot={output_root}")

    context = mp.get_context("spawn")
    all_daily_frames: list[pd.DataFrame] = []
    all_total_frames: list[pd.DataFrame] = []
    combo_timing_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    total_start = perf_counter()

    for combo_idx, (signal_ranks, match_window_seconds, spread) in enumerate(param_grid, start=1):
        params = LowPriceBacktestParams(signal_ranks=signal_ranks, match_window_seconds=match_window_seconds, spread=spread)
        combo_tag = _combo_tag(params)
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
            "elapsedSeconds": 0.0,
        }
        progress_rows.append(progress_row)
        _write_progress_status(output_root, progress_rows)

        summary_rows: list[dict[str, Any]] = []
        timing_rows: list[dict[str, Any]] = []
        tasks = [
            {
                "tradeDate": trade_date,
                "pools": pools,
                "signalRanks": signal_ranks,
                "matchWindowSeconds": _match_window_value(match_window_seconds),
                "spread": spread,
                "imsRoots": [str(path) for path in ims_roots],
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
                checkpoint_summary_rows, checkpoint_timing_row = checkpoint
                summary_rows.extend(checkpoint_summary_rows)
                timing_rows.append(checkpoint_timing_row)
                progress_row["completedDateCount"] = len(timing_rows)
                progress_row["latestTradeDate"] = trade_date
                progress_row["elapsedSeconds"] = perf_counter() - combo_start
                _write_progress_status(output_root, progress_rows)
            tasks = pending_tasks

        with context.Pool(processes=args.processes) as pool:
            for result in pool.imap_unordered(_run_one_date, tasks):
                _write_date_checkpoint(combo_dir, result)
                summary_rows.extend(result.get("summaryRows", []))
                timing_row = {key: value for key, value in result.items() if key != "summaryRows"}
                timing_rows.append(timing_row)
                progress_row["completedDateCount"] = len(timing_rows)
                progress_row["latestTradeDate"] = result["tradeDate"]
                progress_row["elapsedSeconds"] = perf_counter() - combo_start
                _write_progress_status(output_root, progress_rows)
                print(
                    f"[combo {combo_idx}/{len(param_grid)}] [date {len(timing_rows)}/{len(trade_dates)}] "
                    f"{result['tradeDate']} status={result['status']} elapsed={float(result['elapsedSeconds']):.2f}s "
                    f"cacheHits={result['cacheHits']} cacheMisses={result['cacheMisses']} cacheWrites={result['cacheWrites']}"
                )

        daily_df = pd.DataFrame(summary_rows)
        timing_df = pd.DataFrame(timing_rows).sort_values("tradeDate").reset_index(drop=True) if timing_rows else pd.DataFrame()
        total_df = _aggregate_total(daily_df)
        combo_elapsed = perf_counter() - combo_start
        combo_timing_row = {
            "comboTag": combo_tag,
            "signalRanks": ",".join(str(rank) for rank in signal_ranks),
            "matchWindowSeconds": _match_window_value(match_window_seconds),
            "spread": spread,
            "tradeDateCount": len(trade_dates),
            "okDateCount": int((timing_df["status"] == "ok").sum()) if not timing_df.empty else 0,
            "errorDateCount": int((timing_df["status"] == "error").sum()) if not timing_df.empty else 0,
            "elapsedSeconds": combo_elapsed,
        }
        dataframe_to_csv_with_retry(daily_df, combo_dir / "daily_summary.csv", index=False)
        dataframe_to_csv_with_retry(total_df, combo_dir / "total_summary.csv", index=False)
        dataframe_to_csv_with_retry(timing_df, combo_dir / "date_timing.csv", index=False)
        dataframe_to_csv_with_retry(pd.DataFrame([combo_timing_row]), combo_dir / "combo_timing.csv", index=False)
        all_daily_frames.append(daily_df)
        all_total_frames.append(total_df.assign(comboTag=combo_tag))
        combo_timing_rows.append(combo_timing_row)

        progress_row["status"] = "done"
        progress_row["completedDateCount"] = len(timing_rows)
        progress_row["elapsedSeconds"] = combo_elapsed
        _write_progress_status(output_root, progress_rows)
        print(f"[combo {combo_idx}/{len(param_grid)}] done {combo_tag} elapsedSeconds={combo_elapsed:.2f}")
        if not total_df.empty:
            _print_total_report(total_df)

    combined_daily_df = pd.concat(all_daily_frames, ignore_index=True) if all_daily_frames else pd.DataFrame()
    combined_total_df = pd.concat(all_total_frames, ignore_index=True) if all_total_frames else pd.DataFrame()
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
                "pools": ",".join(pools),
                "elapsedSeconds": perf_counter() - total_start,
            }
        ]
    )
    dataframe_to_csv_with_retry(combined_daily_df, output_root / "combined_daily_summary.csv", index=False)
    dataframe_to_csv_with_retry(combined_total_df, output_root / "combined_total_summary.csv", index=False)
    dataframe_to_csv_with_retry(pd.DataFrame(combo_timing_rows), output_root / "combined_combo_timing.csv", index=False)
    dataframe_to_csv_with_retry(run_timing_df, output_root / "run_timing.csv", index=False)
    print(f"totalElapsedSeconds={float(run_timing_df['elapsedSeconds'].iloc[0]):.2f}")
    print(f"resultDir={output_root}")


if __name__ == "__main__":
    main()
