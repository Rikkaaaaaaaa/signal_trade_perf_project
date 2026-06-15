from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestParams:
    open_threshold: float
    close_threshold: float
    min_hold_bars: int
    relaxed_close_threshold: float | None = None
    relaxed_close_after_bars: int | None = None
    stop_loss_mid_ret_threshold: float | None = None
    stop_loss_signal_threshold: float = 0.0

    @property
    def param_tag(self) -> str:
        tag = f"open_{self.open_threshold:g}_close_{self.close_threshold:g}_hold_{self.min_hold_bars}"
        if self.relaxed_close_threshold is not None and self.relaxed_close_after_bars is not None:
            tag += f"_relax_after_{self.relaxed_close_after_bars}_close_{self.relaxed_close_threshold:g}"
        if self.stop_loss_mid_ret_threshold is not None:
            tag += (
                f"_stoploss_mid_{self.stop_loss_mid_ret_threshold * 10000:g}bp_"
                f"signal_{self.stop_loss_signal_threshold:g}"
            )
        return tag

    @property
    def min_hold_minutes(self) -> float:
        return self.min_hold_bars * 0.25


def _build_trade_record(
    pool_name: str,
    trade_date: pd.Timestamp,
    security_code: str,
    side: str,
    open_pos: dict,
    close_row: pd.Series,
    hold_bars: int,
    close_type: str,
    price_bucket_low: int,
    price_bucket_high: int,
    prev_day_vol: float | None,
) -> dict:
    open_mid = float(open_pos["openMid"])
    close_mid = float(close_row.midPrice15s)
    close_bid1 = float(close_row.bid1_15s)
    close_ask1 = float(close_row.ask1_15s)

    if side == "LONG":
        mid_ret = (close_mid - open_mid) / open_mid
        exec_ret = (close_bid1 - open_mid) / open_mid
    else:
        mid_ret = (open_mid - close_mid) / open_mid
        exec_ret = (open_mid - close_ask1) / open_mid

    return {
        "poolName": pool_name,
        "tradeDate": trade_date,
        "securityCode": security_code,
        "side": side,
        "openTime": open_pos["openTime"],
        "closeTime": close_row.barTime,
        "openSignalTime": int(open_pos["openSignalTime"]),
        "closeSignalTime": int(close_row.signalTime),
        "openSignal": float(open_pos["openSignal"]),
        "closeSignal": float(close_row.merge_signal),
        "openMid": open_mid,
        "closeMid": close_mid,
        "closeBid1": close_bid1,
        "closeAsk1": close_ask1,
        "holdBars": int(hold_bars),
        "holdMinutes": hold_bars * 0.25,
        "midRet": mid_ret,
        "execRet": exec_ret,
        "closeType": close_type,
        "priceBucketLow": int(price_bucket_low),
        "priceBucketHigh": int(price_bucket_high),
        "prevDayVol": prev_day_vol,
    }


def simulate_signal_day(
    signal_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    pool_name: str,
    trade_date: pd.Timestamp,
    params: BacktestParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades: list[dict] = []
    summaries: list[dict] = []

    meta_map = meta_df.set_index("securityCode").to_dict("index")

    for security_code, sec_df in signal_df.groupby("securityCode", sort=True):
        sec_df = sec_df.sort_values("barTime").reset_index(drop=True)
        meta = meta_map[security_code]
        price_bucket_low = int(meta["priceBucketLow"])
        price_bucket_high = int(meta["priceBucketHigh"])
        prev_day_vol = meta["prevDayVol"]
        if pd.isna(prev_day_vol):
            prev_day_vol = np.nan

        long_open_positions: list[dict] = []
        short_open_positions: list[dict] = []
        max_concurrent_long = 0
        max_concurrent_short = 0
        max_concurrent_total = 0

        long_signal_count = int((sec_df["merge_signal"] > params.open_threshold).sum())
        short_signal_count = int((sec_df["merge_signal"] < -params.open_threshold).sum())
        total_bar_count = int(len(sec_df))

        for idx, row in sec_df.iterrows():
            if row.merge_signal >= params.close_threshold and short_open_positions:
                eligible = [pos for pos in short_open_positions if pos["openRowIdx"] <= idx - params.min_hold_bars]
                if eligible:
                    for pos in eligible:
                        trades.append(
                            _build_trade_record(
                                pool_name,
                                trade_date,
                                security_code,
                                "SHORT",
                                pos,
                                row,
                                idx - pos["openRowIdx"],
                                "SIGNAL",
                                price_bucket_low,
                                price_bucket_high,
                                prev_day_vol,
                            )
                        )
                    short_open_positions = [pos for pos in short_open_positions if pos["openRowIdx"] > idx - params.min_hold_bars]

            if row.merge_signal <= -params.close_threshold and long_open_positions:
                eligible = [pos for pos in long_open_positions if pos["openRowIdx"] <= idx - params.min_hold_bars]
                if eligible:
                    for pos in eligible:
                        trades.append(
                            _build_trade_record(
                                pool_name,
                                trade_date,
                                security_code,
                                "LONG",
                                pos,
                                row,
                                idx - pos["openRowIdx"],
                                "SIGNAL",
                                price_bucket_low,
                                price_bucket_high,
                                prev_day_vol,
                            )
                        )
                    long_open_positions = [pos for pos in long_open_positions if pos["openRowIdx"] > idx - params.min_hold_bars]

            if row.merge_signal > params.open_threshold:
                long_open_positions.append(
                    {
                        "openTime": row.barTime,
                        "openSignalTime": int(row.signalTime),
                        "openSignal": float(row.merge_signal),
                        "openMid": float(row.midPrice15s),
                        "openRowIdx": idx,
                    }
                )

            if row.merge_signal < -params.open_threshold:
                short_open_positions.append(
                    {
                        "openTime": row.barTime,
                        "openSignalTime": int(row.signalTime),
                        "openSignal": float(row.merge_signal),
                        "openMid": float(row.midPrice15s),
                        "openRowIdx": idx,
                    }
                )

            max_concurrent_long = max(max_concurrent_long, len(long_open_positions))
            max_concurrent_short = max(max_concurrent_short, len(short_open_positions))
            max_concurrent_total = max(max_concurrent_total, len(long_open_positions) + len(short_open_positions))

        last_row = sec_df.iloc[-1]
        last_idx = len(sec_df) - 1

        for pos in long_open_positions:
            trades.append(
                _build_trade_record(
                    pool_name,
                    trade_date,
                    security_code,
                    "LONG",
                    pos,
                    last_row,
                    last_idx - pos["openRowIdx"],
                    "EOD",
                    price_bucket_low,
                    price_bucket_high,
                    prev_day_vol,
                )
            )

        for pos in short_open_positions:
            trades.append(
                _build_trade_record(
                    pool_name,
                    trade_date,
                    security_code,
                    "SHORT",
                    pos,
                    last_row,
                    last_idx - pos["openRowIdx"],
                    "EOD",
                    price_bucket_low,
                    price_bucket_high,
                    prev_day_vol,
                )
            )

        summaries.append(
            {
                "poolName": pool_name,
                "tradeDate": trade_date,
                "securityCode": security_code,
                "totalBarCount": total_bar_count,
                "longOpenCount": long_signal_count,
                "shortOpenCount": short_signal_count,
                "longSignalRatio": np.nan if total_bar_count == 0 else long_signal_count / total_bar_count,
                "shortSignalRatio": np.nan if total_bar_count == 0 else short_signal_count / total_bar_count,
                "maxConcurrentLongCount": max_concurrent_long,
                "maxConcurrentShortCount": max_concurrent_short,
                "maxConcurrentTotalCount": max_concurrent_total,
            }
        )

    trade_df = pd.DataFrame(trades)
    summary_df = pd.DataFrame(summaries)
    if trade_df.empty:
        return trade_df, summary_df

    long_summary = (
        trade_df[trade_df["side"] == "LONG"]
        .groupby(["poolName", "tradeDate", "securityCode"], as_index=False)
        .agg(
            closedLongCount=("side", "size"),
            avgLongHoldMinutes=("holdMinutes", "mean"),
            avgLongMidRet=("midRet", "mean"),
            avgLongExecRet=("execRet", "mean"),
            totalLongMidRet=("midRet", "sum"),
            totalLongExecRet=("execRet", "sum"),
            longMidWinRate=("midRet", lambda s: (s > 0).mean()),
            longExecWinRate=("execRet", lambda s: (s > 0).mean()),
            longEodCloseCount=("closeType", lambda s: (s == "EOD").sum()),
        )
    )
    short_summary = (
        trade_df[trade_df["side"] == "SHORT"]
        .groupby(["poolName", "tradeDate", "securityCode"], as_index=False)
        .agg(
            closedShortCount=("side", "size"),
            avgShortHoldMinutes=("holdMinutes", "mean"),
            avgShortMidRet=("midRet", "mean"),
            avgShortExecRet=("execRet", "mean"),
            totalShortMidRet=("midRet", "sum"),
            totalShortExecRet=("execRet", "sum"),
            shortMidWinRate=("midRet", lambda s: (s > 0).mean()),
            shortExecWinRate=("execRet", lambda s: (s > 0).mean()),
            shortEodCloseCount=("closeType", lambda s: (s == "EOD").sum()),
        )
    )

    summary_df = summary_df.merge(long_summary, on=["poolName", "tradeDate", "securityCode"], how="left")
    summary_df = summary_df.merge(short_summary, on=["poolName", "tradeDate", "securityCode"], how="left")

    for col in ["closedLongCount", "closedShortCount", "longEodCloseCount", "shortEodCloseCount"]:
        summary_df[col] = summary_df[col].fillna(0).astype(int)

    total_trade_count = summary_df["closedLongCount"] + summary_df["closedShortCount"]
    summary_df["avgAllHoldMinutes"] = np.where(
        total_trade_count > 0,
        (
            summary_df["avgLongHoldMinutes"].fillna(0) * summary_df["closedLongCount"]
            + summary_df["avgShortHoldMinutes"].fillna(0) * summary_df["closedShortCount"]
        )
        / total_trade_count,
        np.nan,
    )
    summary_df["avgAllMidRet"] = np.where(
        total_trade_count > 0,
        (summary_df["totalLongMidRet"].fillna(0) + summary_df["totalShortMidRet"].fillna(0)) / total_trade_count,
        np.nan,
    )
    summary_df["avgAllExecRet"] = np.where(
        total_trade_count > 0,
        (summary_df["totalLongExecRet"].fillna(0) + summary_df["totalShortExecRet"].fillna(0)) / total_trade_count,
        np.nan,
    )
    return trade_df, summary_df


def _weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    valid = df[[value_col, weight_col]].dropna()
    if valid.empty or valid[weight_col].sum() == 0:
        return np.nan
    return float(np.average(valid[value_col], weights=valid[weight_col]))


def _series_p95(df: pd.DataFrame, value_col: str) -> float:
    valid = df[value_col].dropna()
    if valid.empty:
        return np.nan
    return float(valid.quantile(0.95))


def aggregate_param_summary(summary_df: pd.DataFrame, params: BacktestParams, scope: str) -> pd.DataFrame:
    total_bar_count = int(summary_df["totalBarCount"].sum())
    total_long_open_count = int(summary_df["longOpenCount"].sum())
    total_short_open_count = int(summary_df["shortOpenCount"].sum())
    total_closed_long_count = int(summary_df["closedLongCount"].sum())
    total_closed_short_count = int(summary_df["closedShortCount"].sum())
    total_trade_count = total_closed_long_count + total_closed_short_count
    total_long_mid_ret = summary_df["totalLongMidRet"].fillna(0).sum()
    total_short_mid_ret = summary_df["totalShortMidRet"].fillna(0).sum()
    total_long_exec_ret = summary_df["totalLongExecRet"].fillna(0).sum()
    total_short_exec_ret = summary_df["totalShortExecRet"].fillna(0).sum()

    row = {
        "scope": scope,
        "paramTag": params.param_tag,
        "openThreshold": params.open_threshold,
        "closeThreshold": params.close_threshold,
        "minHoldBars": params.min_hold_bars,
        "minHoldMinutes": params.min_hold_minutes,
        "tradeDateCount": int(summary_df["tradeDate"].nunique()),
        "securityCount": int(summary_df["securityCode"].nunique()),
        "securityDayCount": int(len(summary_df)),
        "totalBarCount": total_bar_count,
        "totalLongOpenCount": total_long_open_count,
        "totalShortOpenCount": total_short_open_count,
        "longSignalRatio": np.nan if total_bar_count == 0 else total_long_open_count / total_bar_count,
        "shortSignalRatio": np.nan if total_bar_count == 0 else total_short_open_count / total_bar_count,
        "totalClosedLongCount": total_closed_long_count,
        "totalClosedShortCount": total_closed_short_count,
        "totalTradeCount": total_trade_count,
        "avgLongHoldMinutes": _weighted_mean(summary_df, "avgLongHoldMinutes", "closedLongCount"),
        "avgShortHoldMinutes": _weighted_mean(summary_df, "avgShortHoldMinutes", "closedShortCount"),
        "avgLongMidRet": np.nan if total_closed_long_count == 0 else total_long_mid_ret / total_closed_long_count,
        "avgShortMidRet": np.nan if total_closed_short_count == 0 else total_short_mid_ret / total_closed_short_count,
        "avgLongExecRet": np.nan if total_closed_long_count == 0 else total_long_exec_ret / total_closed_long_count,
        "avgShortExecRet": np.nan if total_closed_short_count == 0 else total_short_exec_ret / total_closed_short_count,
        "totalLongMidRet": total_long_mid_ret,
        "totalShortMidRet": total_short_mid_ret,
        "totalMidRet": total_long_mid_ret + total_short_mid_ret,
        "totalLongExecRet": total_long_exec_ret,
        "totalShortExecRet": total_short_exec_ret,
        "totalExecRet": total_long_exec_ret + total_short_exec_ret,
        "longMidWinRate": _weighted_mean(summary_df, "longMidWinRate", "closedLongCount"),
        "shortMidWinRate": _weighted_mean(summary_df, "shortMidWinRate", "closedShortCount"),
        "longExecWinRate": _weighted_mean(summary_df, "longExecWinRate", "closedLongCount"),
        "shortExecWinRate": _weighted_mean(summary_df, "shortExecWinRate", "closedShortCount"),
        "totalEodCloseCount": int(summary_df["longEodCloseCount"].sum() + summary_df["shortEodCloseCount"].sum()),
        "maxConcurrentLongCount": int(summary_df["maxConcurrentLongCount"].max()),
        "maxConcurrentShortCount": int(summary_df["maxConcurrentShortCount"].max()),
        "maxConcurrentTotalCount": int(summary_df["maxConcurrentTotalCount"].max()),
        "avgMaxConcurrentLongCount": float(summary_df["maxConcurrentLongCount"].mean()),
        "avgMaxConcurrentShortCount": float(summary_df["maxConcurrentShortCount"].mean()),
        "avgMaxConcurrentTotalCount": float(summary_df["maxConcurrentTotalCount"].mean()),
        "p95MaxConcurrentLongCount": _series_p95(summary_df, "maxConcurrentLongCount"),
        "p95MaxConcurrentShortCount": _series_p95(summary_df, "maxConcurrentShortCount"),
        "p95MaxConcurrentTotalCount": _series_p95(summary_df, "maxConcurrentTotalCount"),
    }
    row["avgAllHoldMinutes"] = (
        np.nan
        if total_trade_count == 0
        else (
            np.nan_to_num(row["avgLongHoldMinutes"]) * total_closed_long_count
            + np.nan_to_num(row["avgShortHoldMinutes"]) * total_closed_short_count
        )
        / total_trade_count
    )
    row["avgAllMidRet"] = np.nan if total_trade_count == 0 else row["totalMidRet"] / total_trade_count
    row["avgAllExecRet"] = np.nan if total_trade_count == 0 else row["totalExecRet"] / total_trade_count
    return pd.DataFrame([row])
