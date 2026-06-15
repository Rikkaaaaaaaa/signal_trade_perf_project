from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from signal_trade_perf import BacktestParams
from signal_trade_perf.configs import MysqlConfig, build_signal_table_name
from signal_trade_perf.internalization import (
    TickMidCache,
    build_internalization_meta,
    get_default_ims_roots,
    load_aligned_price_df,
    load_ims_child_orders,
    load_signal_day_for_internalization,
    simulate_internalization_day,
)
from signal_trade_perf.io_utils import load_prev_signal_date_mysql
from signal_trade_perf.runner import SourceBacktestRunner


def _build_trade_record_old(
    pool_name: str,
    security_code: str,
    trade_date: pd.Timestamp,
    position: dict[str, object],
    close_row: pd.Series,
    hold_bars: int,
    close_type: str,
    price_bucket_low: int,
    price_bucket_high: int,
    prev_day_vol: float | None,
) -> dict[str, object]:
    open_mid = float(position["openMid"])
    close_mid = float(close_row.closeMidAligned if not pd.isna(close_row.closeMidAligned) else close_row.midPrice15s)
    close_bid1 = float(close_row.closeBid1Aligned if not pd.isna(close_row.closeBid1Aligned) else close_row.bid1_15s)
    close_ask1 = float(close_row.closeAsk1Aligned if not pd.isna(close_row.closeAsk1Aligned) else close_row.ask1_15s)
    qty = int(position["clientQty"])
    side = str(position["inventorySide"])

    if side == "LONG":
        mid_ret = (close_mid - open_mid) / open_mid
        exec_ret = (close_bid1 - open_mid) / open_mid
        mid_pnl = (close_mid - open_mid) * qty
        exec_pnl = (close_bid1 - open_mid) * qty
    else:
        mid_ret = (open_mid - close_mid) / open_mid
        exec_ret = (open_mid - close_ask1) / open_mid
        mid_pnl = (open_mid - close_mid) * qty
        exec_pnl = (open_mid - close_ask1) * qty

    return {
        "poolName": pool_name,
        "tradeDate": trade_date,
        "securityCode": security_code,
        "side": side,
        "strategySource": position["strategySource"],
        "parentOrderId": position["parentOrderId"],
        "clientOrderId": position["clientOrderId"],
        "clientOrderTime": position["clientOrderTime"],
        "clientSide": position["clientSide"],
        "openTime": position["openTime"],
        "openTickTime": position.get("openTickTime", pd.NaT),
        "closeTime": close_row.barTime,
        "openSignalTime": int(position["openSignalTime"]),
        "closeSignalTime": int(close_row.signalTime),
        "openSignal": float(position["openSignal"]),
        "closeSignal": float(close_row.merge_signal),
        "matchDelaySeconds": float(position["matchDelaySeconds"]),
        "clientQty": qty,
        "clientOrderPrice": float(position["clientOrderPrice"]) if not pd.isna(position["clientOrderPrice"]) else np.nan,
        "clientExecPrice": float(position["clientExecPrice"]) if not pd.isna(position["clientExecPrice"]) else np.nan,
        "openMid": open_mid,
        "closeMid": close_mid,
        "closeBid1": close_bid1,
        "closeAsk1": close_ask1,
        "closePriceSource": "ddb_returns" if not pd.isna(close_row.closeBid1Aligned) or not pd.isna(close_row.closeAsk1Aligned) else "tick_15s",
        "holdBars": int(hold_bars),
        "holdMinutes": hold_bars * 0.25,
        "midRet": mid_ret,
        "execRet": exec_ret,
        "midPnl": mid_pnl,
        "execPnl": exec_pnl,
        "openNotional": open_mid * qty,
        "closeType": close_type,
        "priceBucketLow": int(price_bucket_low),
        "priceBucketHigh": int(price_bucket_high),
        "prevDayVol": prev_day_vol,
    }


def simulate_internalization_day_old(
    signal_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    client_order_df: pd.DataFrame,
    pool_name: str,
    params: BacktestParams,
    tick_mid_cache: TickMidCache,
    match_window_seconds: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signal_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    trade_date = pd.Timestamp(signal_df["tradeDate"].iloc[0])
    meta_map = meta_df.set_index("securityCode").to_dict("index")
    event_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []

    signal_by_security = {
        security_code: sec_df.sort_values("barTime").reset_index(drop=True)
        for security_code, sec_df in signal_df.groupby("securityCode", sort=True)
    }
    orders_by_security = {
        security_code: sec_df.sort_values("clientOrderTime").reset_index(drop=True)
        for security_code, sec_df in client_order_df.groupby("securityCode", sort=True)
    }

    security_codes = sorted(set(signal_by_security) | set(orders_by_security))

    for security_code in security_codes:
        sec_signal_df = signal_by_security.get(security_code, pd.DataFrame())
        sec_orders_df = orders_by_security.get(security_code, pd.DataFrame())
        meta = meta_map.get(
            security_code,
            {
                "priceBucketLow": np.nan,
                "priceBucketHigh": np.nan,
                "prevDayVol": np.nan,
            },
        )

        if sec_signal_df.empty:
            continue

        signal_times = sec_signal_df["barTime"].to_numpy(dtype="datetime64[ns]")
        pending_match_rows: list[dict[str, object]] = []
        positions_by_open_idx: dict[int, list[dict[str, object]]] = {}

        for order in sec_orders_df.to_dict(orient="records"):
            order_time = pd.Timestamp(order["clientOrderTime"])
            window_start = order_time - pd.Timedelta(seconds=match_window_seconds)
            matched_idx = int(signal_times.searchsorted(order_time.to_datetime64(), side="right") - 1)
            matched_row = None
            if matched_idx >= 0:
                probe_row = sec_signal_df.iloc[matched_idx]
                if probe_row.barTime >= window_start:
                    if order["clientSide"] == "B" and probe_row.merge_signal < -params.open_threshold:
                        matched_row = probe_row
                    elif order["clientSide"] == "S" and probe_row.merge_signal > params.open_threshold:
                        matched_row = probe_row

            if matched_row is None:
                continue

            pending_match_rows.append(
                {
                    **order,
                    "matchSignalTime": matched_row.barTime,
                    "matchSignalTimeInt": int(matched_row.signalTime),
                    "matchSignal": float(matched_row.merge_signal),
                    "matchDelaySeconds": float((order_time - matched_row.barTime).total_seconds()),
                    "matchedSignalRowIdx": matched_idx,
                    "inventorySide": "SHORT" if order["clientSide"] == "B" else "LONG",
                }
            )

        if pending_match_rows:
            matched_order_df = pd.DataFrame(pending_match_rows)
            matched_order_df = pd.merge_asof(
                matched_order_df.sort_values("clientOrderTime"),
                tick_mid_cache.get(security_code)
                .sort_values("tickTime")
                .rename(columns={"tickTime": "openTickTime", "midPriceTick": "openMid"}),
                by="securityCode",
                left_on="clientOrderTime",
                right_on="openTickTime",
                direction="backward",
            )

            for row in matched_order_df.to_dict(orient="records"):
                if pd.isna(row.get("openMid")):
                    continue
                position = {
                    **row,
                    "openRowIdx": int(row["matchedSignalRowIdx"]),
                    "openTime": row["clientOrderTime"],
                    "openSignalTime": int(row["matchSignalTimeInt"]),
                    "openSignal": float(row["matchSignal"]),
                    "openMid": float(row["openMid"]),
                }
                positions_by_open_idx.setdefault(int(row["matchedSignalRowIdx"]), []).append(position)
                event_rows.append(row)

        long_open_positions: list[dict[str, object]] = []
        short_open_positions: list[dict[str, object]] = []

        for idx, row in sec_signal_df.iterrows():
            for pos in positions_by_open_idx.get(idx, []):
                if pos["inventorySide"] == "LONG":
                    long_open_positions.append(pos)
                else:
                    short_open_positions.append(pos)

            if row.merge_signal >= params.close_threshold and short_open_positions:
                eligible = [pos for pos in short_open_positions if pos["openRowIdx"] <= idx - params.min_hold_bars]
                if eligible:
                    for pos in eligible:
                        trade_rows.append(
                            _build_trade_record_old(
                                pool_name=pool_name,
                                security_code=security_code,
                                trade_date=trade_date,
                                position=pos,
                                close_row=row,
                                hold_bars=idx - int(pos["openRowIdx"]),
                                close_type="SIGNAL",
                                price_bucket_low=int(meta["priceBucketLow"]) if not pd.isna(meta["priceBucketLow"]) else -1,
                                price_bucket_high=int(meta["priceBucketHigh"]) if not pd.isna(meta["priceBucketHigh"]) else -1,
                                prev_day_vol=float(meta["prevDayVol"]) if not pd.isna(meta["prevDayVol"]) else np.nan,
                            )
                        )
                    short_open_positions = [pos for pos in short_open_positions if pos["openRowIdx"] > idx - params.min_hold_bars]

            if row.merge_signal <= -params.close_threshold and long_open_positions:
                eligible = [pos for pos in long_open_positions if pos["openRowIdx"] <= idx - params.min_hold_bars]
                if eligible:
                    for pos in eligible:
                        trade_rows.append(
                            _build_trade_record_old(
                                pool_name=pool_name,
                                security_code=security_code,
                                trade_date=trade_date,
                                position=pos,
                                close_row=row,
                                hold_bars=idx - int(pos["openRowIdx"]),
                                close_type="SIGNAL",
                                price_bucket_low=int(meta["priceBucketLow"]) if not pd.isna(meta["priceBucketLow"]) else -1,
                                price_bucket_high=int(meta["priceBucketHigh"]) if not pd.isna(meta["priceBucketHigh"]) else -1,
                                prev_day_vol=float(meta["prevDayVol"]) if not pd.isna(meta["prevDayVol"]) else np.nan,
                            )
                        )
                    long_open_positions = [pos for pos in long_open_positions if pos["openRowIdx"] > idx - params.min_hold_bars]

        if not sec_signal_df.empty:
            last_row = sec_signal_df.iloc[-1]
            last_idx = len(sec_signal_df) - 1
            for pos in long_open_positions + short_open_positions:
                trade_rows.append(
                    _build_trade_record_old(
                        pool_name=pool_name,
                        security_code=security_code,
                        trade_date=trade_date,
                        position=pos,
                        close_row=last_row,
                        hold_bars=last_idx - int(pos["openRowIdx"]),
                        close_type="EOD",
                        price_bucket_low=int(meta["priceBucketLow"]) if not pd.isna(meta["priceBucketLow"]) else -1,
                        price_bucket_high=int(meta["priceBucketHigh"]) if not pd.isna(meta["priceBucketHigh"]) else -1,
                        prev_day_vol=float(meta["prevDayVol"]) if not pd.isna(meta["prevDayVol"]) else np.nan,
                    )
                )

    return pd.DataFrame(event_rows), pd.DataFrame(trade_rows)


def compare_trades(old_trades: pd.DataFrame, new_trades: pd.DataFrame) -> pd.DataFrame:
    join_cols = ["securityCode", "strategySource", "parentOrderId", "clientOrderId", "clientOrderTime", "side"]
    old_df = old_trades.copy()
    new_df = new_trades.copy()
    old_df["sourceVersion"] = "old"
    new_df["sourceVersion"] = "new"

    compare_df = old_df.merge(new_df, on=join_cols, how="outer", suffixes=("_old", "_new"), indicator=True)
    if compare_df.empty:
        return compare_df

    compare_df["sameCloseTime"] = compare_df["closeTime_old"] == compare_df["closeTime_new"]
    compare_df["sameCloseSignalTime"] = compare_df["closeSignalTime_old"] == compare_df["closeSignalTime_new"]
    compare_df["sameClosePrice"] = (
        np.isclose(compare_df["closeBid1_old"], compare_df["closeBid1_new"], equal_nan=True)
        & np.isclose(compare_df["closeAsk1_old"], compare_df["closeAsk1_new"], equal_nan=True)
        & np.isclose(compare_df["closeMid_old"], compare_df["closeMid_new"], equal_nan=True)
    )
    compare_df["execPnlDiff"] = compare_df["execPnl_new"].fillna(0.0) - compare_df["execPnl_old"].fillna(0.0)
    compare_df["execRetDiff"] = compare_df["execRet_new"].fillna(0.0) - compare_df["execRet_old"].fillna(0.0)
    return compare_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260105")
    parser.add_argument("--pool", default="hs300")
    parser.add_argument("--variant", default="all", choices=["all", "lt1000", "lt2000"])
    parser.add_argument("--open-threshold", type=float, default=6.0)
    parser.add_argument("--close-threshold", type=float, default=4.0)
    parser.add_argument("--min-hold-bars", type=int, default=20)
    parser.add_argument("--match-window-seconds", type=int, default=10)
    parser.add_argument("--topn", type=int, default=8)
    args = parser.parse_args()

    params = BacktestParams(
        open_threshold=args.open_threshold,
        close_threshold=args.close_threshold,
        min_hold_bars=args.min_hold_bars,
    )
    mysql_config = MysqlConfig()
    ims_roots = get_default_ims_roots(PROJECT_ROOT)

    table_name = build_signal_table_name(args.pool)
    prev_trade_date = load_prev_signal_date_mysql(args.date, table_name, mysql_config)

    runner = SourceBacktestRunner(mysql_config=mysql_config, day_cache_dir=PROJECT_ROOT / "cache" / "source_day_cache")
    try:
        old_signal_df, old_meta_df, _ = runner.prepare_signal_mid_day(
            trade_date=args.date,
            prev_trade_date=prev_trade_date,
            table_name=table_name,
            force_rebuild=False,
        )
    finally:
        runner.close()

    new_signal_df = load_signal_day_for_internalization(args.date, args.pool, mysql_config=mysql_config)
    current_price_df = load_aligned_price_df(args.date, args.pool, new_signal_df["securityCode"].unique().tolist())
    prev_price_df = (
        load_aligned_price_df(prev_trade_date, args.pool, new_signal_df["securityCode"].unique().tolist())
        if prev_trade_date is not None
        else pd.DataFrame(columns=["barTime", "securityCode", "closeMidAligned", "closeAsk1Aligned", "closeBid1Aligned"])
    )
    new_meta_df = build_internalization_meta(
        signal_df=new_signal_df,
        pool_name=args.pool,
        trade_date=args.date,
        current_price_df=current_price_df,
        prev_price_df=prev_price_df,
    )
    close_price_map = {
        security_code: sec_df.reset_index(drop=True)
        for security_code, sec_df in current_price_df.groupby("securityCode", sort=True)
    }

    base_order_df = load_ims_child_orders(args.date, ims_roots)
    base_order_df = base_order_df[base_order_df["securityCode"].isin(new_signal_df["securityCode"].unique())].reset_index(drop=True)
    if args.variant == "lt1000":
        base_order_df = base_order_df[base_order_df["clientQty"] < 1000].reset_index(drop=True)
    elif args.variant == "lt2000":
        base_order_df = base_order_df[base_order_df["clientQty"] < 2000].reset_index(drop=True)

    tick_mid_cache = TickMidCache(trade_date=args.date)
    try:
        _, old_trades = simulate_internalization_day_old(
            signal_df=old_signal_df.merge(current_price_df, on=["securityCode", "barTime"], how="left"),
            meta_df=old_meta_df,
            client_order_df=base_order_df,
            pool_name=args.pool,
            params=params,
            tick_mid_cache=tick_mid_cache,
            match_window_seconds=args.match_window_seconds,
        )
        _, new_trades, _ = simulate_internalization_day(
            signal_df=new_signal_df,
            meta_df=new_meta_df,
            client_order_df=base_order_df,
            pool_name=args.pool,
            params=params,
            tick_mid_cache=tick_mid_cache,
            close_price_map=close_price_map,
            match_window_seconds=args.match_window_seconds,
        )
    finally:
        tick_mid_cache.close()

    compare_df = compare_trades(old_trades, new_trades)

    print(f"variant={args.variant}")
    print(f"oldTradeCount={len(old_trades)}")
    print(f"newTradeCount={len(new_trades)}")
    if compare_df.empty:
        return

    print(compare_df["_merge"].value_counts(dropna=False).to_string())
    both_df = compare_df[compare_df["_merge"] == "both"].copy()
    if not both_df.empty:
        print(f"commonTradeCount={len(both_df)}")
        print(f"sameCloseTimeCount={int(both_df['sameCloseTime'].sum())}")
        print(f"sameCloseSignalTimeCount={int(both_df['sameCloseSignalTime'].sum())}")
        print(f"sameClosePriceCount={int(both_df['sameClosePrice'].sum())}")

        changed_df = both_df[
            (~both_df["sameCloseTime"]) | (~both_df["sameCloseSignalTime"]) | (~both_df["sameClosePrice"])
        ].copy()
        changed_df = changed_df.reindex(
            columns=[
                "securityCode",
                "strategySource",
                "parentOrderId",
                "clientOrderId",
                "side",
                "clientQty_old",
                "openTime_old",
                "closeTime_old",
                "closeTime_new",
                "closeSignalTime_old",
                "closeSignalTime_new",
                "closeBid1_old",
                "closeBid1_new",
                "closeAsk1_old",
                "closeAsk1_new",
                "execPnl_old",
                "execPnl_new",
                "execPnlDiff",
                "holdBars_old",
                "holdBars_new",
                "holdMinutes_old",
                "holdMinutes_new",
                "sameCloseTime",
                "sameCloseSignalTime",
                "sameClosePrice",
            ]
        ).sort_values("execPnlDiff", key=lambda s: s.abs(), ascending=False)
        print("changedCasesTop:")
        print(changed_df.head(args.topn).to_string(index=False))

    only_old_df = compare_df[compare_df["_merge"] == "left_only"].copy()
    if not only_old_df.empty:
        print("oldOnlyTop:")
        print(
            only_old_df[
                [
                    "securityCode",
                    "strategySource",
                    "parentOrderId",
                    "clientOrderId",
                    "side",
                    "clientQty_old",
                    "openTime_old",
                    "closeTime_old",
                    "closeSignalTime_old",
                    "execPnl_old",
                    "holdBars_old",
                    "holdMinutes_old",
                ]
            ].head(args.topn).to_string(index=False)
        )

    only_new_df = compare_df[compare_df["_merge"] == "right_only"].copy()
    if not only_new_df.empty:
        print("newOnlyTop:")
        print(
            only_new_df[
                [
                    "securityCode",
                    "strategySource",
                    "parentOrderId",
                    "clientOrderId",
                    "side",
                    "clientQty_new",
                    "openTime_new",
                    "closeTime_new",
                    "closeSignalTime_new",
                    "execPnl_new",
                    "holdBars_new",
                    "holdMinutes_new",
                ]
            ].head(args.topn).to_string(index=False)
        )


if __name__ == "__main__":
    main()
