from __future__ import annotations

import numpy as np
import pandas as pd

from .core import BacktestParams


def get_default_price_bin_edges() -> list[float]:
    return [0.0, 10.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 1000000.0]


def calc_prev_day_vol(quote_df: pd.DataFrame) -> pd.DataFrame:
    if quote_df.empty:
        return pd.DataFrame(columns=["securityCode", "prevDayVol"])

    ret_df = quote_df.sort_values(["securityCode", "barTime"]).copy()
    ret_df["ret_15s"] = ret_df.groupby("securityCode")["midPrice15s"].shift(-1).sub(ret_df["midPrice15s"]).div(ret_df["midPrice15s"])
    return (
        ret_df.groupby("securityCode", as_index=False)["ret_15s"]
        .std()
        .rename(columns={"ret_15s": "prevDayVol"})
    )


def assign_price_bins(meta_df: pd.DataFrame, price_bin_edges: list[float]) -> pd.DataFrame:
    result = meta_df.copy()
    intervals = pd.cut(result["openMidPrice"], bins=price_bin_edges, right=False, include_lowest=True)
    result["priceBucketLow"] = intervals.apply(lambda x: int(x.left) if pd.notna(x) else -1)
    result["priceBucketHigh"] = intervals.apply(lambda x: int(x.right) if pd.notna(x) else -1)
    return result


def build_security_meta(
    signal_quote_df: pd.DataFrame,
    prev_vol_df: pd.DataFrame,
    pool_name: str,
    trade_date: str,
    price_bin_edges: list[float],
) -> pd.DataFrame:
    if signal_quote_df.empty:
        return pd.DataFrame(columns=["poolName", "tradeDate", "securityCode", "openMidPrice", "priceBucketLow", "priceBucketHigh", "prevDayVol"])

    meta_df = (
        signal_quote_df.sort_values(["securityCode", "barTime"])
        .groupby("securityCode", as_index=False)
        .agg(openMidPrice=("midPrice15s", "first"))
    )
    meta_df["poolName"] = pool_name
    meta_df["tradeDate"] = pd.Timestamp(trade_date)
    meta_df = assign_price_bins(meta_df, price_bin_edges)
    meta_df = meta_df.merge(prev_vol_df, on="securityCode", how="left")
    return meta_df[["poolName", "tradeDate", "securityCode", "openMidPrice", "priceBucketLow", "priceBucketHigh", "prevDayVol"]]


def calc_vol_cutoffs(meta_frames: list[pd.DataFrame], num_bins: int = 10) -> list[float]:
    non_empty_frames = [frame for frame in meta_frames if not frame.empty]
    if not non_empty_frames or num_bins <= 1:
        return []

    meta_union = pd.concat(non_empty_frames, ignore_index=True)
    valid = meta_union.loc[meta_union["prevDayVol"].notna() & (meta_union["prevDayVol"] > 0), "prevDayVol"]
    if valid.empty:
        return []
    return [float(valid.quantile(i / num_bins)) for i in range(1, num_bins)]


def assign_vol_bucket(trade_df: pd.DataFrame, vol_cutoffs: list[float]) -> pd.DataFrame:
    result = trade_df.copy()
    if not vol_cutoffs:
        result["prevVolBucket"] = np.where(result["prevDayVol"].fillna(0) > 0, 1, -1)
        return result

    cutoffs = np.array(vol_cutoffs)
    bucket = np.searchsorted(cutoffs, result["prevDayVol"].fillna(-1).to_numpy(), side="left") + 1
    bucket = np.where(result["prevDayVol"].fillna(0) > 0, bucket, -1)
    result["prevVolBucket"] = bucket.astype(int)
    return result


def summarize_price_bucket_contrib(trade_df: pd.DataFrame, params: BacktestParams) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame()

    grouped = (
        trade_df.groupby(["priceBucketLow", "priceBucketHigh"], as_index=False)
        .agg(
            totalTradeCount=("side", "size"),
            holdMinutesSum=("holdMinutes", "sum"),
            midRetSum=("midRet", "sum"),
            execRetSum=("execRet", "sum"),
            midWinCount=("midRet", lambda s: int((s > 0).sum())),
            execWinCount=("execRet", lambda s: int((s > 0).sum())),
            eodCloseCount=("closeType", lambda s: int((s == "EOD").sum())),
        )
    )
    grouped["paramTag"] = params.param_tag
    grouped["openThreshold"] = params.open_threshold
    grouped["closeThreshold"] = params.close_threshold
    grouped["minHoldBars"] = params.min_hold_bars
    grouped["minHoldMinutes"] = params.min_hold_minutes
    return grouped


def finalize_price_bucket_summary(price_contrib_df: pd.DataFrame) -> pd.DataFrame:
    if price_contrib_df.empty:
        return pd.DataFrame()

    grouped = (
        price_contrib_df.groupby(
            ["paramTag", "openThreshold", "closeThreshold", "minHoldBars", "minHoldMinutes", "priceBucketLow", "priceBucketHigh"],
            as_index=False,
        )
        .agg(
            totalTradeCount=("totalTradeCount", "sum"),
            holdMinutesSum=("holdMinutesSum", "sum"),
            midRetSum=("midRetSum", "sum"),
            execRetSum=("execRetSum", "sum"),
            midWinCount=("midWinCount", "sum"),
            execWinCount=("execWinCount", "sum"),
            eodCloseCount=("eodCloseCount", "sum"),
        )
    )
    grouped["avgHoldMinutes"] = grouped["holdMinutesSum"] / grouped["totalTradeCount"]
    grouped["avgMidRet"] = grouped["midRetSum"] / grouped["totalTradeCount"]
    grouped["avgExecRet"] = grouped["execRetSum"] / grouped["totalTradeCount"]
    grouped["midWinRate"] = grouped["midWinCount"] / grouped["totalTradeCount"]
    grouped["execWinRate"] = grouped["execWinCount"] / grouped["totalTradeCount"]
    grouped = grouped.rename(columns={"midRetSum": "totalMidRet", "execRetSum": "totalExecRet"})
    return grouped[
        [
            "paramTag",
            "openThreshold",
            "closeThreshold",
            "minHoldBars",
            "minHoldMinutes",
            "priceBucketLow",
            "priceBucketHigh",
            "totalTradeCount",
            "avgHoldMinutes",
            "avgMidRet",
            "avgExecRet",
            "totalMidRet",
            "totalExecRet",
            "midWinRate",
            "execWinRate",
            "eodCloseCount",
        ]
    ].sort_values(["priceBucketLow", "openThreshold", "closeThreshold", "minHoldBars"]).reset_index(drop=True)


def summarize_vol_bucket_contrib(trade_df: pd.DataFrame, params: BacktestParams, vol_cutoffs: list[float]) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame()

    vol_df = assign_vol_bucket(trade_df, vol_cutoffs)
    vol_df = vol_df[vol_df["prevVolBucket"] > 0].copy()
    if vol_df.empty:
        return pd.DataFrame()

    grouped = (
        vol_df.groupby("prevVolBucket", as_index=False)
        .agg(
            totalTradeCount=("side", "size"),
            holdMinutesSum=("holdMinutes", "sum"),
            midRetSum=("midRet", "sum"),
            execRetSum=("execRet", "sum"),
            midWinCount=("midRet", lambda s: int((s > 0).sum())),
            execWinCount=("execRet", lambda s: int((s > 0).sum())),
            eodCloseCount=("closeType", lambda s: int((s == "EOD").sum())),
            prevDayVolSum=("prevDayVol", "sum"),
        )
    )
    grouped["paramTag"] = params.param_tag
    grouped["openThreshold"] = params.open_threshold
    grouped["closeThreshold"] = params.close_threshold
    grouped["minHoldBars"] = params.min_hold_bars
    grouped["minHoldMinutes"] = params.min_hold_minutes
    return grouped


def finalize_vol_bucket_summary(vol_contrib_df: pd.DataFrame) -> pd.DataFrame:
    if vol_contrib_df.empty:
        return pd.DataFrame()

    grouped = (
        vol_contrib_df.groupby(
            ["paramTag", "openThreshold", "closeThreshold", "minHoldBars", "minHoldMinutes", "prevVolBucket"],
            as_index=False,
        )
        .agg(
            totalTradeCount=("totalTradeCount", "sum"),
            holdMinutesSum=("holdMinutesSum", "sum"),
            midRetSum=("midRetSum", "sum"),
            execRetSum=("execRetSum", "sum"),
            midWinCount=("midWinCount", "sum"),
            execWinCount=("execWinCount", "sum"),
            eodCloseCount=("eodCloseCount", "sum"),
            prevDayVolSum=("prevDayVolSum", "sum"),
        )
    )
    grouped["avgPrevDayVol"] = grouped["prevDayVolSum"] / grouped["totalTradeCount"]
    grouped["avgHoldMinutes"] = grouped["holdMinutesSum"] / grouped["totalTradeCount"]
    grouped["avgMidRet"] = grouped["midRetSum"] / grouped["totalTradeCount"]
    grouped["avgExecRet"] = grouped["execRetSum"] / grouped["totalTradeCount"]
    grouped["midWinRate"] = grouped["midWinCount"] / grouped["totalTradeCount"]
    grouped["execWinRate"] = grouped["execWinCount"] / grouped["totalTradeCount"]
    grouped = grouped.rename(columns={"midRetSum": "totalMidRet", "execRetSum": "totalExecRet"})
    return grouped[
        [
            "paramTag",
            "openThreshold",
            "closeThreshold",
            "minHoldBars",
            "minHoldMinutes",
            "prevVolBucket",
            "avgPrevDayVol",
            "totalTradeCount",
            "avgHoldMinutes",
            "avgMidRet",
            "avgExecRet",
            "totalMidRet",
            "totalExecRet",
            "midWinRate",
            "execWinRate",
            "eodCloseCount",
        ]
    ].sort_values(["prevVolBucket", "openThreshold", "closeThreshold", "minHoldBars"]).reset_index(drop=True)


def compare_summary_to_baseline(
    current_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    numeric_tolerance: float = 1e-12,
) -> pd.DataFrame:
    current_row = current_df.iloc[0].to_dict()
    baseline_row = baseline_df.iloc[0].to_dict()
    fields = [
        "paramTag",
        "openThreshold",
        "closeThreshold",
        "minHoldBars",
        "minHoldMinutes",
        "totalLongOpenCount",
        "totalShortOpenCount",
        "totalClosedLongCount",
        "totalClosedShortCount",
        "totalTradeCount",
        "avgLongHoldMinutes",
        "avgShortHoldMinutes",
        "avgAllHoldMinutes",
        "avgLongMidRet",
        "avgShortMidRet",
        "avgAllMidRet",
        "avgLongExecRet",
        "avgShortExecRet",
        "avgAllExecRet",
        "totalLongMidRet",
        "totalShortMidRet",
        "totalMidRet",
        "totalLongExecRet",
        "totalShortExecRet",
        "totalExecRet",
        "longMidWinRate",
        "shortMidWinRate",
        "longExecWinRate",
        "shortExecWinRate",
        "totalEodCloseCount",
    ]

    rows = []
    for field in fields:
        baseline_value = baseline_row[field]
        current_value = current_row[field]
        try:
            diff_value = float(current_value) - float(baseline_value)
            equal_numeric = abs(diff_value) <= numeric_tolerance
        except Exception:
            diff_value = None
            equal_numeric = str(baseline_value) == str(current_value)
        rows.append(
            {
                "field": field,
                "baseline": baseline_value,
                "current": current_value,
                "diff": diff_value,
                "equalString": str(baseline_value) == str(current_value),
                "equalNumeric": equal_numeric,
            }
        )
    return pd.DataFrame(rows)
