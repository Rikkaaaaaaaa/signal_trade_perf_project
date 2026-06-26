from __future__ import annotations

import argparse
import gzip
import hashlib
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

from signal_trade_perf.core import BacktestParams
from signal_trade_perf.internalization_backtest import get_default_ims_roots
from signal_trade_perf.merged_internalization import MergedBacktestParams, run_merged_prepared_day
from signal_trade_perf.merged_internalization_data import load_merged_day_inputs
from signal_trade_perf.source_backtest import dataframe_to_csv_with_retry, mkdir_with_retry


POOL_NAMES = ["hs300", "zz500", "zz1000", "zz2000_1", "zz2000_2", "zz2000_3", "other"]


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
            raise ValueError(f"Invalid relaxed close spec: {item}")
    return specs


def _parse_signal_rank_sets(raw: str) -> list[tuple[int, ...]]:
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


def _discover_ims_trade_dates(ims_roots: list[Path], start_date: str, end_date: str) -> list[str]:
    trade_dates: set[str] = set()
    for ims_root in ims_roots:
        if not ims_root.exists():
            continue
        trade_dates.update(path.name for path in ims_root.iterdir() if path.is_dir() and start_date <= path.name <= end_date)
    return sorted(trade_dates)


def _cache_namespace(prediction_signal_table: str | None, fill_rate_signal_table: str | None) -> str:
    raw = f"{prediction_signal_table or ''}||{fill_rate_signal_table or ''}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _cache_path(
    cache_root: Path,
    trade_date: str,
    pool_name: str,
    prediction_signal_table: str | None,
    fill_rate_signal_table: str | None,
) -> Path:
    namespace = _cache_namespace(prediction_signal_table, fill_rate_signal_table)
    return cache_root / namespace / trade_date / f"{pool_name}.pkl.gz"


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


def _cached_inputs_missing_close_prices(prepared_inputs: dict[str, object] | None) -> bool:
    if not prepared_inputs:
        return False
    close_price_map = prepared_inputs.get("closePriceMap")
    return isinstance(close_price_map, dict) and len(close_price_map) == 0


def _match_window_value(match_window_seconds: int | None) -> str:
    return "unlimited" if match_window_seconds is None else str(match_window_seconds)


def _empty_result(trade_date: str, params: MergedBacktestParams, elapsed: float, error: str) -> dict[str, Any]:
    return {
        "tradeDate": trade_date,
        "paramTag": params.param_tag,
        "elapsedSeconds": elapsed,
        "status": "error",
        "error": error,
        "cacheHits": 0,
        "cacheMisses": 0,
        "cacheWrites": 0,
        "mergedRows": [],
        "predictionRows": [],
        "fillRateRows": [],
        "routeRows": [],
        "predictionOrderEventRows": [],
        "predictionTradeRows": [],
        "fillRateOrderEventRows": [],
        "fillRateTradeRows": [],
    }


def _run_one_date(task: dict[str, Any]) -> dict[str, Any]:
    trade_date = str(task["tradeDate"])
    params = MergedBacktestParams(
        prediction_params=BacktestParams(
            open_threshold=float(task["openThreshold"]),
            close_threshold=float(task["closeThreshold"]),
            min_hold_bars=int(task["minHoldBars"]),
            relaxed_close_after_bars=(
                None if task.get("relaxedCloseAfterBars") in {None, "", "none"} else int(task["relaxedCloseAfterBars"])
            ),
            relaxed_close_threshold=(
                None if task.get("relaxedCloseThreshold") in {None, "", "none"} else float(task["relaxedCloseThreshold"])
            ),
        ),
        fill_rate_signal_ranks=tuple(int(value) for value in task["fillSignalRanks"]),
        fill_rate_support_threshold=float(task["fillSupportThreshold"]),
        match_window_seconds=(None if task["matchWindowSeconds"] == "unlimited" else int(task["matchWindowSeconds"])),
        fill_rate_spread=float(task["fillRateSpread"]),
    )
    start = perf_counter()
    cache_hits = 0
    cache_misses = 0
    cache_writes = 0
    merged_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fill_rate_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    prediction_order_event_rows: list[dict[str, Any]] = []
    prediction_trade_rows: list[dict[str, Any]] = []
    fill_rate_order_event_rows: list[dict[str, Any]] = []
    fill_rate_trade_rows: list[dict[str, Any]] = []
    try:
        cache_mode = str(task["cacheMode"])
        cache_root = Path(str(task["cacheRoot"]))
        prediction_signal_table = task.get("predictionSignalTableName")
        fill_rate_signal_table = task.get("fillRateSignalTableName")
        for pool_name in task["pools"]:
            cache_file = _cache_path(
                cache_root=cache_root,
                trade_date=trade_date,
                pool_name=pool_name,
                prediction_signal_table=prediction_signal_table,
                fill_rate_signal_table=fill_rate_signal_table,
            )
            prepared_inputs = None
            if cache_mode not in {"none", "refresh"}:
                prepared_inputs = _load_cached_inputs(cache_file)
                if _cached_inputs_missing_close_prices(prepared_inputs):
                    prepared_inputs = None
                elif prepared_inputs is not None:
                    cache_hits += 1

            if prepared_inputs is None:
                cache_misses += 1
                prepared_inputs = load_merged_day_inputs(
                    trade_date=trade_date,
                    pool_name=pool_name,
                    ims_roots=[Path(path) for path in task["imsRoots"]],
                    prediction_signal_table_name=prediction_signal_table,
                    fill_rate_signal_table_name=fill_rate_signal_table,
                )
                if prepared_inputs is not None and cache_mode in {"readwrite", "refresh"}:
                    _write_cached_inputs(cache_file, prepared_inputs)
                    cache_writes += 1

            if prepared_inputs is None:
                continue

            (
                route_df,
                prediction_order_events_df,
                prediction_trades_df,
                fill_rate_order_events_df,
                fill_rate_trades_df,
                fill_rate_summary_df,
                merged_summary_df,
                prediction_pool_summary_df,
            ) = run_merged_prepared_day(
                prepared_inputs=prepared_inputs,
                params=params,
            )
            if not merged_summary_df.empty:
                merged_rows.extend(merged_summary_df.to_dict(orient="records"))
            if not prediction_pool_summary_df.empty:
                prediction_rows.extend(prediction_pool_summary_df.to_dict(orient="records"))
            if not fill_rate_summary_df.empty:
                fill_rate_rows.extend(fill_rate_summary_df.to_dict(orient="records"))
            if not route_df.empty:
                route_rows.extend(route_df.to_dict(orient="records"))
            if not prediction_order_events_df.empty:
                prediction_order_event_rows.extend(prediction_order_events_df.to_dict(orient="records"))
            if not prediction_trades_df.empty:
                prediction_trade_rows.extend(prediction_trades_df.to_dict(orient="records"))
            if not fill_rate_order_events_df.empty:
                fill_rate_order_event_rows.extend(fill_rate_order_events_df.to_dict(orient="records"))
            if not fill_rate_trades_df.empty:
                fill_rate_trade_rows.extend(fill_rate_trades_df.to_dict(orient="records"))

        return {
            "tradeDate": trade_date,
            "paramTag": params.param_tag,
            "elapsedSeconds": perf_counter() - start,
            "status": "ok" if merged_rows else "skipped_empty_result",
            "cacheHits": cache_hits,
            "cacheMisses": cache_misses,
            "cacheWrites": cache_writes,
            "mergedRows": merged_rows,
            "predictionRows": prediction_rows,
            "fillRateRows": fill_rate_rows,
            "routeRows": route_rows,
            "predictionOrderEventRows": prediction_order_event_rows,
            "predictionTradeRows": prediction_trade_rows,
            "fillRateOrderEventRows": fill_rate_order_event_rows,
            "fillRateTradeRows": fill_rate_trade_rows,
        }
    except Exception:
        return _empty_result(trade_date=trade_date, params=params, elapsed=perf_counter() - start, error=traceback.format_exc())


def _aggregate_total(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()
    group_cols = [
        "variantTag",
        "predictionVariantTag",
        "fillRateVariantTag",
        "paramTag",
        "openThreshold",
        "closeThreshold",
        "minHoldBars",
        "relaxedCloseAfterBars",
        "relaxedCloseThreshold",
        "fillRateSignalRanks",
        "fillRateSupportThreshold",
        "matchWindowSeconds",
        "fillRateSpread",
    ]
    rows: list[dict[str, Any]] = []
    for key, group_df in daily_df.groupby(group_cols, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_values))
        total_exec_pnl = float(group_df["totalExecPnl"].astype(float).sum())
        total_notional = float(group_df["totalMatchedNotional"].astype(float).sum())
        total_client_amt = float(group_df["totalClientAmt"].astype(float).sum())
        matched_client_amt = float(group_df["matchedClientAmt"].astype(float).sum())
        merged_daily_capital = pd.to_numeric(group_df["mergedMaxCapitalUsed"], errors="coerce").dropna()
        prediction_daily_capital = pd.to_numeric(group_df["predictionMaxCapitalUsed"], errors="coerce").dropna()
        fill_rate_daily_capital = pd.to_numeric(group_df["fillRateMaxCapitalUsed"], errors="coerce").dropna()
        prediction_exec_pnl = float(group_df["predictionExecPnl"].astype(float).sum())
        fill_rate_exec_pnl = float(group_df["fillRateExecPnl"].astype(float).sum())
        prediction_notional = float(group_df["predictionMatchedNotional"].astype(float).sum())
        fill_rate_notional = float(group_df["fillRateMatchedNotional"].astype(float).sum())
        by_date_ret = pd.to_numeric(group_df["notionalWeightedExecRet"], errors="coerce").dropna()
        row.update(
            {
                "tradeDateCount": int(group_df["tradeDate"].nunique()),
                "totalTradeCount": int(group_df["totalTradeCount"].astype(int).sum()),
                "totalExecPnl": total_exec_pnl,
                "predictionExecPnl": prediction_exec_pnl,
                "fillRateExecPnl": fill_rate_exec_pnl,
                "totalMatchedNotional": total_notional,
                "predictionMatchedNotional": prediction_notional,
                "fillRateMatchedNotional": fill_rate_notional,
                "predictionMaxDailyCapitalUsed": float(prediction_daily_capital.max()) if len(prediction_daily_capital) else 0.0,
                "fillRateMaxDailyCapitalUsed": float(fill_rate_daily_capital.max()) if len(fill_rate_daily_capital) else 0.0,
                "predictionNotionalWeightedExecRet": np.nan if prediction_notional == 0 else prediction_exec_pnl / prediction_notional,
                "fillRateNotionalWeightedExecRet": np.nan if fill_rate_notional == 0 else fill_rate_exec_pnl / fill_rate_notional,
                "predictionCapitalAdjustedReturn": (
                    np.nan if not len(prediction_daily_capital) or float(prediction_daily_capital.max()) == 0 else prediction_exec_pnl / float(prediction_daily_capital.max())
                ),
                "fillRateCapitalAdjustedReturn": (
                    np.nan if not len(fill_rate_daily_capital) or float(fill_rate_daily_capital.max()) == 0 else fill_rate_exec_pnl / float(fill_rate_daily_capital.max())
                ),
                "clientAmtMatchRate": np.nan if total_client_amt == 0 else matched_client_amt / total_client_amt,
                "notionalWeightedExecRet": np.nan if total_notional == 0 else total_exec_pnl / total_notional,
                "mergedMaxDailyCapitalUsed": float(merged_daily_capital.max()) if len(merged_daily_capital) else 0.0,
                "mergedP95DailyCapitalUsed": float(merged_daily_capital.quantile(0.95)) if len(merged_daily_capital) else 0.0,
                "mergedAvgDailyCapitalUsed": float(merged_daily_capital.mean()) if len(merged_daily_capital) else 0.0,
                "mergedCapitalAdjustedReturn": (
                    np.nan if not len(merged_daily_capital) or float(merged_daily_capital.max()) == 0 else total_exec_pnl / float(merged_daily_capital.max())
                ),
                "byDateWinRate": float((by_date_ret > 0).mean()) if len(by_date_ret) else np.nan,
                "byDateRetMean": float(by_date_ret.mean()) if len(by_date_ret) else np.nan,
                "byDateRetStd": float(by_date_ret.std(ddof=1)) if len(by_date_ret) > 1 else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _print_combo_report(total_df: pd.DataFrame) -> None:
    if total_df.empty:
        return
    report_df = total_df[total_df["predictionVariantTag"] == "poscap_avg5x5_partial"].copy()
    if report_df.empty:
        return
    report_cols = [
        "predictionVariantTag",
        "fillRateVariantTag",
        "totalTradeCount",
        "totalExecPnl",
        "predictionExecPnl",
        "fillRateExecPnl",
        "predictionMaxDailyCapitalUsed",
        "predictionNotionalWeightedExecRet",
        "fillRateMaxDailyCapitalUsed",
        "fillRateNotionalWeightedExecRet",
        "mergedMaxDailyCapitalUsed",
        "mergedCapitalAdjustedReturn",
        "clientAmtMatchRate",
        "notionalWeightedExecRet",
    ]
    report_cols = [col for col in report_cols if col in report_df.columns]
    print(report_df[report_cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run date-parallel merged internalization parameter sweep.")
    parser.add_argument("--start-date", default="20260401")
    parser.add_argument("--end-date", default="20260430")
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--open-thresholds", default="6")
    parser.add_argument("--close-thresholds", default="4")
    parser.add_argument("--min-hold-bars-list", default="20")
    parser.add_argument("--relaxed-close-specs", default="none")
    parser.add_argument("--match-window-seconds-list", default="10")
    parser.add_argument("--fill-signal-ranks-list", default="1,2")
    parser.add_argument("--fill-support-thresholds", default="0")
    parser.add_argument("--fill-spreads", default="0.01")
    parser.add_argument("--pools", default="all")
    parser.add_argument("--ims-roots", default="")
    parser.add_argument("--prediction-signal-table", default="")
    parser.add_argument("--fill-rate-signal-table", default="")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "results" / "internalization_backtest" / "merged_param_sweep_date_mp"),
    )
    parser.add_argument("--cache-mode", choices=["none", "readwrite", "refresh"], default="readwrite")
    parser.add_argument(
        "--cache-root",
        default=str(PROJECT_ROOT / "results" / "internalization_backtest" / "merged_data_cache"),
    )
    args = parser.parse_args()

    ims_roots = _parse_path_list(args.ims_roots) if args.ims_roots.strip() else get_default_ims_roots(PROJECT_ROOT)
    ims_roots = [path.resolve() for path in ims_roots]
    trade_dates = _discover_ims_trade_dates(ims_roots, args.start_date, args.end_date)
    if not trade_dates:
        raise ValueError(f"No IMS trade dates found in [{args.start_date}, {args.end_date}]")

    pools = _parse_pools(args.pools)
    output_root = Path(args.output_root) / f"{trade_dates[0]}_{trade_dates[-1]}"
    mkdir_with_retry(output_root)

    relaxed_specs = _parse_relaxed_close_specs(args.relaxed_close_specs)
    param_grid = []
    for open_threshold, close_threshold, min_hold_bars, match_window_seconds, fill_signal_ranks, fill_support_threshold, fill_spread in itertools.product(
        _parse_float_list(args.open_thresholds),
        _parse_float_list(args.close_thresholds),
        _parse_int_list(args.min_hold_bars_list),
        _parse_match_windows(args.match_window_seconds_list),
        _parse_signal_rank_sets(args.fill_signal_ranks_list),
        _parse_float_list(args.fill_support_thresholds),
        _parse_float_list(args.fill_spreads),
    ):
        for spec_hold_bars, relaxed_after_bars, relaxed_threshold in relaxed_specs:
            if spec_hold_bars is not None and int(spec_hold_bars) != int(min_hold_bars):
                continue
            param_grid.append(
                {
                    "openThreshold": open_threshold,
                    "closeThreshold": close_threshold,
                    "minHoldBars": min_hold_bars,
                    "relaxedCloseAfterBars": relaxed_after_bars,
                    "relaxedCloseThreshold": relaxed_threshold,
                    "matchWindowSeconds": _match_window_value(match_window_seconds),
                    "fillSignalRanks": fill_signal_ranks,
                    "fillSupportThreshold": fill_support_threshold,
                    "fillRateSpread": fill_spread,
                }
            )

    print(f"tradeDateCount={len(trade_dates)}")
    print(f"paramComboCount={len(param_grid)}")
    print(f"processes={args.processes}")
    print(f"pools={','.join(pools)}")
    print(f"cacheMode={args.cache_mode}")
    print(f"outputRoot={output_root}")

    all_daily_frames: list[pd.DataFrame] = []
    all_total_frames: list[pd.DataFrame] = []
    combo_timing_rows: list[dict[str, Any]] = []
    total_start = perf_counter()
    context = mp.get_context("spawn")

    for combo_idx, combo in enumerate(param_grid, start=1):
        params = MergedBacktestParams(
            prediction_params=BacktestParams(
                open_threshold=float(combo["openThreshold"]),
                close_threshold=float(combo["closeThreshold"]),
                min_hold_bars=int(combo["minHoldBars"]),
                relaxed_close_after_bars=combo["relaxedCloseAfterBars"],
                relaxed_close_threshold=combo["relaxedCloseThreshold"],
            ),
            fill_rate_signal_ranks=tuple(int(value) for value in combo["fillSignalRanks"]),
            fill_rate_support_threshold=float(combo["fillSupportThreshold"]),
            match_window_seconds=(None if combo["matchWindowSeconds"] == "unlimited" else int(combo["matchWindowSeconds"])),
            fill_rate_spread=float(combo["fillRateSpread"]),
        )
        combo_tag = params.param_tag
        combo_dir = output_root / combo_tag
        combo_start = perf_counter()
        print(f"[combo {combo_idx}/{len(param_grid)}] start {combo_tag}")

        tasks = [
            {
                "tradeDate": trade_date,
                "openThreshold": combo["openThreshold"],
                "closeThreshold": combo["closeThreshold"],
                "minHoldBars": combo["minHoldBars"],
                "relaxedCloseAfterBars": combo["relaxedCloseAfterBars"],
                "relaxedCloseThreshold": combo["relaxedCloseThreshold"],
                "matchWindowSeconds": combo["matchWindowSeconds"],
                "fillSignalRanks": combo["fillSignalRanks"],
                "fillSupportThreshold": combo["fillSupportThreshold"],
                "fillRateSpread": combo["fillRateSpread"],
                "imsRoots": [str(path) for path in ims_roots],
                "pools": pools,
                "cacheMode": args.cache_mode,
                "cacheRoot": args.cache_root,
                "predictionSignalTableName": args.prediction_signal_table.strip() or None,
                "fillRateSignalTableName": args.fill_rate_signal_table.strip() or None,
            }
            for trade_date in trade_dates
        ]

        merged_rows: list[dict[str, Any]] = []
        route_rows: list[dict[str, Any]] = []
        prediction_order_event_rows: list[dict[str, Any]] = []
        prediction_trade_rows: list[dict[str, Any]] = []
        fill_rate_order_event_rows: list[dict[str, Any]] = []
        fill_rate_trade_rows: list[dict[str, Any]] = []
        timing_rows: list[dict[str, Any]] = []
        with context.Pool(processes=args.processes) as pool:
            for result in pool.imap_unordered(_run_one_date, tasks):
                merged_rows.extend(result.get("mergedRows", []))
                route_rows.extend(result.get("routeRows", []))
                prediction_order_event_rows.extend(result.get("predictionOrderEventRows", []))
                prediction_trade_rows.extend(result.get("predictionTradeRows", []))
                fill_rate_order_event_rows.extend(result.get("fillRateOrderEventRows", []))
                fill_rate_trade_rows.extend(result.get("fillRateTradeRows", []))
                timing_rows.append({key: value for key, value in result.items() if key not in {"mergedRows", "predictionRows", "fillRateRows", "routeRows", "predictionOrderEventRows", "predictionTradeRows", "fillRateOrderEventRows", "fillRateTradeRows"}})
                print(
                    f"[combo {combo_idx}/{len(param_grid)}] [date {len(timing_rows)}/{len(trade_dates)}] "
                    f"{result['tradeDate']} status={result['status']} elapsed={float(result['elapsedSeconds']):.2f}s "
                    f"cacheHits={result['cacheHits']} cacheMisses={result['cacheMisses']} cacheWrites={result['cacheWrites']}"
                )

        daily_df = pd.DataFrame(merged_rows)
        route_df = pd.DataFrame(route_rows)
        prediction_order_event_df = pd.DataFrame(prediction_order_event_rows)
        prediction_trade_df = pd.DataFrame(prediction_trade_rows)
        fill_rate_order_event_df = pd.DataFrame(fill_rate_order_event_rows)
        fill_rate_trade_df = pd.DataFrame(fill_rate_trade_rows)
        merged_trade_df = pd.concat([prediction_trade_df, fill_rate_trade_df], ignore_index=True, sort=False) if not prediction_trade_df.empty or not fill_rate_trade_df.empty else pd.DataFrame()
        timing_df = pd.DataFrame(timing_rows).sort_values("tradeDate").reset_index(drop=True) if timing_rows else pd.DataFrame()
        total_df = _aggregate_total(daily_df)
        combo_elapsed = perf_counter() - combo_start
        dataframe_to_csv_with_retry(daily_df, combo_dir / "daily_summary.csv", index=False)
        dataframe_to_csv_with_retry(route_df, combo_dir / "route_decisions.csv", index=False)
        dataframe_to_csv_with_retry(prediction_order_event_df, combo_dir / "prediction_order_events.csv", index=False)
        dataframe_to_csv_with_retry(prediction_trade_df, combo_dir / "prediction_trades.csv", index=False)
        dataframe_to_csv_with_retry(fill_rate_order_event_df, combo_dir / "fill_rate_order_events.csv", index=False)
        dataframe_to_csv_with_retry(fill_rate_trade_df, combo_dir / "fill_rate_trades.csv", index=False)
        dataframe_to_csv_with_retry(merged_trade_df, combo_dir / "merged_trades.csv", index=False)
        dataframe_to_csv_with_retry(total_df, combo_dir / "total_summary.csv", index=False)
        dataframe_to_csv_with_retry(timing_df, combo_dir / "date_timing.csv", index=False)
        dataframe_to_csv_with_retry(
            pd.DataFrame(
                [
                    {
                        "comboTag": combo_tag,
                        "tradeDateCount": len(trade_dates),
                        "elapsedSeconds": combo_elapsed,
                    }
                ]
            ),
            combo_dir / "combo_timing.csv",
            index=False,
        )
        all_daily_frames.append(daily_df)
        all_total_frames.append(total_df.assign(comboTag=combo_tag))
        combo_timing_rows.append(
            {
                "comboTag": combo_tag,
                "tradeDateCount": len(trade_dates),
                "elapsedSeconds": combo_elapsed,
            }
        )
        print(f"[combo {combo_idx}/{len(param_grid)}] done {combo_tag} elapsedSeconds={combo_elapsed:.2f}")
        _print_combo_report(total_df)

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
