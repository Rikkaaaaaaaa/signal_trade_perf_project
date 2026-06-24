from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from signal_trade_perf.core import BacktestParams
from signal_trade_perf.internalization_backtest import get_default_ims_roots
from signal_trade_perf.merged_internalization import MergedBacktestParams, run_merged_prepared_day
from signal_trade_perf.merged_internalization_data import load_merged_day_inputs
from signal_trade_perf.source_backtest import dataframe_to_csv_with_retry, mkdir_with_retry


def _parse_match_window(raw: str) -> int | None:
    token = str(raw).strip().lower()
    if token in {"none", "unlimited", "all", "不限"}:
        return None
    return int(token)


def _parse_signal_ranks(raw: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-day merged internalization backtest.")
    parser.add_argument("--date", default="20260401")
    parser.add_argument("--pool", default="hs300")
    parser.add_argument("--open-threshold", type=float, default=6.0)
    parser.add_argument("--close-threshold", type=float, default=4.0)
    parser.add_argument("--min-hold-bars", type=int, default=20)
    parser.add_argument("--match-window-seconds", default="10")
    parser.add_argument("--fill-signal-ranks", default="1,2")
    parser.add_argument("--fill-support-threshold", type=float, default=0.0)
    parser.add_argument("--fill-spread", type=float, default=0.01)
    parser.add_argument("--prediction-signal-table", default="")
    parser.add_argument("--fill-rate-signal-table", default="")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "results" / "internalization_backtest" / "merged_single_day"),
    )
    args = parser.parse_args()

    params = MergedBacktestParams(
        prediction_params=BacktestParams(
            open_threshold=args.open_threshold,
            close_threshold=args.close_threshold,
            min_hold_bars=args.min_hold_bars,
        ),
        fill_rate_signal_ranks=_parse_signal_ranks(args.fill_signal_ranks),
        fill_rate_support_threshold=args.fill_support_threshold,
        match_window_seconds=_parse_match_window(args.match_window_seconds),
        fill_rate_spread=args.fill_spread,
    )
    result_dir = Path(args.output_root) / f"{args.date}_{args.pool}_{params.param_tag}"
    mkdir_with_retry(result_dir)

    start = perf_counter()
    prepared_inputs = load_merged_day_inputs(
        trade_date=args.date,
        pool_name=args.pool,
        ims_roots=get_default_ims_roots(PROJECT_ROOT),
        prediction_signal_table_name=args.prediction_signal_table.strip() or None,
        fill_rate_signal_table_name=args.fill_rate_signal_table.strip() or None,
    )
    if prepared_inputs is None:
        print("empty prepared inputs")
        return

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

    datetime_format = "%Y-%m-%d %H:%M:%S.%f"
    dataframe_to_csv_with_retry(route_df, result_dir / "route_decisions.csv", index=False, date_format=datetime_format)
    dataframe_to_csv_with_retry(
        prediction_order_events_df,
        result_dir / "prediction_order_events.csv",
        index=False,
        date_format=datetime_format,
    )
    dataframe_to_csv_with_retry(
        prediction_trades_df,
        result_dir / "prediction_trades.csv",
        index=False,
        date_format=datetime_format,
    )
    dataframe_to_csv_with_retry(
        prediction_pool_summary_df,
        result_dir / "prediction_pool_summary.csv",
        index=False,
        date_format=datetime_format,
    )
    dataframe_to_csv_with_retry(
        fill_rate_order_events_df,
        result_dir / "fill_rate_order_events.csv",
        index=False,
        date_format=datetime_format,
    )
    dataframe_to_csv_with_retry(
        fill_rate_trades_df,
        result_dir / "fill_rate_trades.csv",
        index=False,
        date_format=datetime_format,
    )
    dataframe_to_csv_with_retry(
        fill_rate_summary_df,
        result_dir / "fill_rate_summary.csv",
        index=False,
        date_format=datetime_format,
    )
    dataframe_to_csv_with_retry(
        merged_summary_df,
        result_dir / "merged_summary.csv",
        index=False,
        date_format=datetime_format,
    )

    if merged_summary_df.empty:
        print("empty merged summary")
    else:
        report_cols = [
            "predictionVariantTag",
            "fillRateVariantTag",
            "totalTradeCount",
            "totalExecPnl",
            "predictionExecPnl",
            "fillRateExecPnl",
            "clientAmtMatchRate",
            "notionalWeightedExecRet",
            "mergedMaxCapitalUsed",
        ]
        print(merged_summary_df[report_cols].to_string(index=False))

    print(f"totalElapsedSeconds={perf_counter() - start:.2f}")
    print(f"resultDir={result_dir}")


if __name__ == "__main__":
    main()
