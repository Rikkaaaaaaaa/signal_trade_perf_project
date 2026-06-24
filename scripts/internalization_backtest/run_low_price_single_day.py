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

from signal_trade_perf.internalization_backtest import get_default_ims_roots
from signal_trade_perf.low_price_internalization import (
    LOW_PRICE_VARIANTS,
    LowPriceBacktestParams,
    run_low_price_single_day,
)
from signal_trade_perf.source_backtest import dataframe_to_csv_with_retry, mkdir_with_retry


def _parse_match_window(raw: str) -> int | None:
    # 单日入口支持 unlimited/none，方便对比“客户单必须在 signal 后 N 秒内到达”和“不限制窗口”。
    token = str(raw).strip().lower()
    if token in {"none", "unlimited", "all", "不限"}:
        return None
    return int(token)


def _parse_signal_ranks(raw: str) -> tuple[int, ...]:
    # signal-ranks 是 exact 集合；例如 1,2 不包含 10。
    return tuple(int(value.strip()) for value in raw.split(',') if value.strip())

def _print_summary(row: dict[str, object]) -> None:
    # 控制台只打印最常看的核心指标，完整明细见 result_dir 下的 CSV。
    print("")
    print(f"variantTag: {row['variantTag']}")
    print(f"trades: {int(row['totalTradeCount'])}")
    for col in [
        "totalExecPnl",
        "clientAmtMatchRate",
        "notionalWeightedExecRet",
        "yTestWinRate",
        "maxCapitalUsed",
        "p95CapitalUsedByEvent",
        "capitalAdjustedReturn",
        "totalMatchedNotional",
    ]:
        print(f"{col}: {row[col]}")


def main() -> None:
    # 单日入口用于 debug：会额外落 order_events/trades，便于检查某只股票或某个客户单的撮合细节。
    parser = argparse.ArgumentParser(description="Run one-day low-price internalization backtest.")
    parser.add_argument("--date", default="20260105")
    parser.add_argument("--pool", default="hs300")
    parser.add_argument("--signal-ranks", default="", help="Comma-separated exact signal ranks, for example 1,2,10.")
    parser.add_argument("--signal-rank-threshold", type=int, default=2, help="Backward compatible shorthand for 1..N when --signal-ranks is empty.")
    parser.add_argument("--match-window-seconds", default="10")
    parser.add_argument("--spread", type=float, default=0.01)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results" / "internalization_backtest" / "low_price_single_day"))
    args = parser.parse_args()

    # 如果显式给 --signal-ranks，就按 exact 集合跑；否则兼容旧的 threshold 写法 1..N。
    signal_ranks = (
        _parse_signal_ranks(args.signal_ranks)
        if args.signal_ranks.strip()
        else tuple(range(1, int(args.signal_rank_threshold) + 1))
    )

    params = LowPriceBacktestParams(
        signal_ranks=signal_ranks,
        match_window_seconds=_parse_match_window(args.match_window_seconds),
        spread=args.spread,
    )
    result_dir = Path(args.output_root) / f"{args.date}_{args.pool}_{params.param_tag}"
    mkdir_with_retry(result_dir)

    start = perf_counter()
    events_df, trades_df, summary_df = run_low_price_single_day(
        trade_date=args.date,
        params=params,
        ims_roots=get_default_ims_roots(PROJECT_ROOT),
        pool_name=args.pool,
    )

    datetime_format = "%Y-%m-%d %H:%M:%S.%f"
    # order_events 记录客户单是否找到 signal、是否被 cap 拦住；trades 记录最终执行和 PnL。
    dataframe_to_csv_with_retry(events_df, result_dir / "order_events.csv", index=False, date_format=datetime_format)
    dataframe_to_csv_with_retry(trades_df, result_dir / "trades.csv", index=False, date_format=datetime_format)
    dataframe_to_csv_with_retry(summary_df, result_dir / "summary.csv", index=False, date_format=datetime_format)
    dataframe_to_csv_with_retry(
        summary_df[["tradeDate", "variantTag", "maxCapitalUsed", "p95CapitalUsedByEvent", "capitalAdjustedReturn"]],
        result_dir / "daily_capital_summary.csv",
        index=False,
        date_format=datetime_format,
    )

    if summary_df.empty:
        print("empty result")
    else:
        for row in summary_df[summary_df["variantTag"].isin(LOW_PRICE_VARIANTS)].to_dict(orient="records"):
            _print_summary(row)

    print(f"totalElapsedSeconds: {perf_counter() - start:.2f}")
    print(f"resultDir={result_dir}")


if __name__ == "__main__":
    main()

