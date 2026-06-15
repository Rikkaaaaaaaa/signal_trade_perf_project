from __future__ import annotations

import argparse
import gzip
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from signal_trade_perf.internalization_backtest import BacktestParams, run_internalization_prepared_day
from signal_trade_perf.source_backtest import dataframe_to_csv_with_retry, mkdir_with_retry


DEFAULT_CACHE_ROOT = PROJECT_ROOT / "results" / "internalization_backtest" / "data_cache"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "internalization_backtest" / "diagnostics"


def _parse_match_window(raw: str) -> int | None:
    token = raw.strip().lower()
    if token in {"none", "unlimited", "all", "不限"}:
        return None
    return int(token)


def _cache_path(cache_root: Path, trade_date: str, pool_name: str) -> Path:
    return cache_root / trade_date / f"{pool_name}.pkl.gz"


def _load_cached_inputs(path: Path) -> dict[str, object]:
    with gzip.open(path, "rb") as file:
        return pickle.load(file)


def _identity_cols(df: pd.DataFrame) -> list[str]:
    candidates = [
        "securityCode",
        "strategySource",
        "parentOrderId",
        "clientOrderId",
        "clientOrderTime",
        "openTime",
        "side",
        "clientSide",
        "clientQty",
    ]
    return [col for col in candidates if col in df.columns]


def _prepare_trades(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    if df.empty:
        return df
    keep_cols = _identity_cols(df) + [
        "openSignal",
        "closeSignal",
        "matchDelaySeconds",
        "openMid",
        "stopLossMidPrice",
        "closeMid",
        "closeBid1",
        "closeAsk1",
        "execPnl",
        "execRet",
        "openNotional",
        "closeType",
        "holdBars",
        "holdMinutes",
        "liquidityCapQty",
    ]
    keep_cols = [col for col in keep_cols if col in df.columns]
    return df[keep_cols].rename(
        columns={col: f"{col}_{suffix}" for col in keep_cols if col not in _identity_cols(df)}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs stop-loss trades for one date/pool.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--open-threshold", type=float, default=6.0)
    parser.add_argument("--close-threshold", type=float, default=6.0)
    parser.add_argument("--min-hold-bars", type=int, default=30)
    parser.add_argument("--match-window-seconds", default="unlimited")
    parser.add_argument("--stop-loss-bp", type=float, default=150.0)
    parser.add_argument("--stop-loss-signal-threshold", type=float, default=2.0)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    prepared_inputs = _load_cached_inputs(_cache_path(args.cache_root, args.date, args.pool))
    match_window_seconds = _parse_match_window(args.match_window_seconds)
    base_params = BacktestParams(
        open_threshold=args.open_threshold,
        close_threshold=args.close_threshold,
        min_hold_bars=args.min_hold_bars,
    )
    stop_params = BacktestParams(
        open_threshold=args.open_threshold,
        close_threshold=args.close_threshold,
        min_hold_bars=args.min_hold_bars,
        stop_loss_mid_ret_threshold=args.stop_loss_bp / 10000.0,
        stop_loss_signal_threshold=args.stop_loss_signal_threshold,
    )

    _, base_trades, _, base_pool = run_internalization_prepared_day(
        prepared_inputs=prepared_inputs,
        params=base_params,
        match_window_seconds=match_window_seconds,
        profile=False,
    )
    _, stop_trades, _, stop_pool = run_internalization_prepared_day(
        prepared_inputs=prepared_inputs,
        params=stop_params,
        match_window_seconds=match_window_seconds,
        profile=False,
    )

    key_cols = _identity_cols(base_trades)
    merged = _prepare_trades(base_trades, "base").merge(
        _prepare_trades(stop_trades, "stop"),
        on=key_cols,
        how="outer",
        indicator=True,
    )
    merged["deltaExecPnl"] = merged["execPnl_stop"].fillna(0.0) - merged["execPnl_base"].fillna(0.0)
    merged["deltaExecRet"] = merged["execRet_stop"].fillna(0.0) - merged["execRet_base"].fillna(0.0)
    merged["wasStopLoss"] = merged["closeType_stop"].astype(str).str.startswith("STOP_LOSS")

    tag = (
        f"{args.date}_{args.pool}_open{args.open_threshold:g}_close{args.close_threshold:g}_"
        f"hold{args.min_hold_bars}_match{args.match_window_seconds}_"
        f"stoploss{args.stop_loss_bp:g}bp_signal{args.stop_loss_signal_threshold:g}"
    )
    mkdir_with_retry(args.output_dir)
    all_path = args.output_dir / f"{tag}_trade_impact_all.csv"
    worst_path = args.output_dir / f"{tag}_trade_impact_worst.csv"
    summary_path = args.output_dir / f"{tag}_summary.csv"
    dataframe_to_csv_with_retry(merged.sort_values("deltaExecPnl"), all_path, index=False)
    dataframe_to_csv_with_retry(merged.sort_values("deltaExecPnl").head(args.top_n), worst_path, index=False)

    base_all = base_pool[base_pool["variantTag"] == "all"].iloc[0].to_dict() if not base_pool.empty else {}
    stop_all = stop_pool[stop_pool["variantTag"] == "all"].iloc[0].to_dict() if not stop_pool.empty else {}
    summary = pd.DataFrame(
        [
            {
                "tradeDate": args.date,
                "poolName": args.pool,
                "baseTotalExecPnl": base_all.get("totalExecPnl"),
                "stopTotalExecPnl": stop_all.get("totalExecPnl"),
                "deltaTotalExecPnl": stop_all.get("totalExecPnl", 0.0) - base_all.get("totalExecPnl", 0.0),
                "stopLossTradeCount": int(merged["wasStopLoss"].sum()),
                "stopLossDeltaExecPnl": float(merged.loc[merged["wasStopLoss"], "deltaExecPnl"].sum()),
                "allTradeCount": len(merged),
                "matchedBothCount": int((merged["_merge"] == "both").sum()),
            }
        ]
    )
    dataframe_to_csv_with_retry(summary, summary_path, index=False)

    display_cols = [
        "securityCode",
        "side",
        "clientQty",
        "openTime",
        "openSignal_base",
        "closeSignal_base",
        "closeSignal_stop",
        "openMid_base",
        "stopLossMidPrice_stop",
        "closeMid_base",
        "closeMid_stop",
        "execPnl_base",
        "execPnl_stop",
        "deltaExecPnl",
        "execRet_base",
        "execRet_stop",
        "openNotional_base",
        "closeType_base",
        "closeType_stop",
        "holdBars_base",
        "holdBars_stop",
    ]
    display_cols = [col for col in display_cols if col in merged.columns]
    print(f"[output] {all_path}")
    print(f"[output] {worst_path}")
    print(summary.to_string(index=False))
    print(merged.sort_values("deltaExecPnl")[display_cols].head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
