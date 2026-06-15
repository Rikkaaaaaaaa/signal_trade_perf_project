from __future__ import annotations

import numpy as np
import pandas as pd


def _prune_and_reorder_columns(df: pd.DataFrame, first_columns: list[str], drop_columns: list[str]) -> pd.DataFrame:
    # 导出时只保留当前真正关心的字段，并把核心指标提前，避免 CSV 越跑越臃肿。
    if df.empty:
        return df

    trimmed_df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors="ignore").copy()
    ordered_first = [col for col in first_columns if col in trimmed_df.columns]
    remaining_cols = [col for col in trimmed_df.columns if col not in ordered_first]
    return trimmed_df[ordered_first + remaining_cols]


def format_internalization_security_summary_for_output(security_summary_df: pd.DataFrame) -> pd.DataFrame:
    # 单票汇总输出口径：保留 debug 常用字段，隐藏 qty/mid/win-rate/totalRet 等噪音字段。
    if security_summary_df.empty:
        return security_summary_df

    formatted_df = security_summary_df.copy()
    formatted_df["totalTradeCount"] = formatted_df["closedLongCount"] + formatted_df["closedShortCount"]
    formatted_df["totalExecPnl"] = formatted_df["totalLongExecPnl"] + formatted_df["totalShortExecPnl"]
    formatted_df["avgAllExecRet"] = np.where(
        formatted_df["totalTradeCount"] == 0,
        np.nan,
        (formatted_df["totalLongExecRet"] + formatted_df["totalShortExecRet"]) / formatted_df["totalTradeCount"],
    )
    formatted_df["clientAmtMatchRate"] = np.where(
        formatted_df["clientAmt"] == 0,
        np.nan,
        formatted_df["matchedClientAmt"] / formatted_df["clientAmt"],
    )
    formatted_df["totalEodCloseCount"] = formatted_df["longEodCloseCount"] + formatted_df["shortEodCloseCount"]
    formatted_df["maxConcurrentTotalCount"] = formatted_df["maxConcurrentLongCount"] + formatted_df["maxConcurrentShortCount"]
    if "maxConcurrentLongQty" in formatted_df.columns and "maxConcurrentShortQty" in formatted_df.columns:
        formatted_df["maxConcurrentTotalQty"] = formatted_df["maxConcurrentLongQty"] + formatted_df["maxConcurrentShortQty"]
    formatted_df["avgAllHoldMinutes"] = np.where(
        formatted_df["totalTradeCount"] == 0,
        np.nan,
        (
            formatted_df["avgLongHoldMinutes"].fillna(0) * formatted_df["closedLongCount"]
            + formatted_df["avgShortHoldMinutes"].fillna(0) * formatted_df["closedShortCount"]
        ) / formatted_df["totalTradeCount"],
    )

    first_columns = [
        "poolName",
        "tradeDate",
        "securityCode",
        "variantTag",
        "avgAllExecRet",
        "totalExecPnl",
        "matchedClientSellAmt",
        "clientAmtMatchRate",
        "totalTradeCount",
        "longOpenCount",
        "shortOpenCount",
        "closedLongCount",
        "closedShortCount",
        "totalEodCloseCount",
        "clientAmt",
        "clientBuyAmt",
        "clientSellAmt",
        "matchedClientAmt",
        "matchedClientBuyAmt",
        "clientChildCount",
        "clientBuyChildCount",
        "clientSellChildCount",
        "matchedClientChildCount",
        "matchedClientBuyChildCount",
        "matchedClientSellChildCount",
        "unmatchedClientChildCount",
        "avgAllHoldMinutes",
        "avgLongHoldMinutes",
        "avgShortHoldMinutes",
        "avgLongExecRet",
        "avgShortExecRet",
        "totalLongExecPnl",
        "totalShortExecPnl",
        "totalMatchedNotional",
        "notionalWeightedExecRet",
        "totalBarCount",
        "longSignalCount",
        "shortSignalCount",
        "maxConcurrentLongCount",
        "maxConcurrentShortCount",
        "maxConcurrentTotalCount",
        "maxConcurrentLongQty",
        "maxConcurrentShortQty",
        "maxConcurrentTotalQty",
    ]
    drop_columns = [
        "clientBuyQty",
        "clientSellQty",
        "matchedClientQty",
        "matchedClientBuyQty",
        "matchedClientSellQty",
        "avgLongMidRet",
        "avgShortMidRet",
        "totalLongMidRet",
        "totalShortMidRet",
        "totalLongExecRet",
        "totalShortExecRet",
        "totalLongMidPnl",
        "totalShortMidPnl",
        "longMidWinRate",
        "shortMidWinRate",
        "longExecWinRate",
        "shortExecWinRate",
        "longEodCloseCount",
        "shortEodCloseCount",
        "notionalWeightedMidRet",
    ]
    return _prune_and_reorder_columns(formatted_df, first_columns, drop_columns)


def format_internalization_pool_summary_for_output(pool_summary_df: pd.DataFrame) -> pd.DataFrame:
    # Pool 级输出只保留金额口径 match rate，不再导出 child/qty/mid/win 这些噪音指标。
    if pool_summary_df.empty:
        return pool_summary_df

    first_columns = [
        "scope",
        "variantTag",
        "paramTag",
        "openThreshold",
        "closeThreshold",
        "minHoldBars",
        "minHoldMinutes",
        "relaxedCloseThreshold",
        "relaxedCloseAfterBars",
        "avgAllExecRet",
        "totalExecPnl",
        "matchedClientSellAmt",
        "clientAmtMatchRate",
        "totalTradeCount",
        "totalLongOpenCount",
        "totalShortOpenCount",
        "totalClosedLongCount",
        "totalClosedShortCount",
        "totalEodCloseCount",
        "totalClientAmt",
        "totalClientBuyAmt",
        "totalClientSellAmt",
        "matchedClientAmt",
        "matchedClientBuyAmt",
        "totalClientChildCount",
        "totalClientBuyChildCount",
        "totalClientSellChildCount",
        "matchedClientChildCount",
        "matchedClientBuyChildCount",
        "matchedClientSellChildCount",
        "unmatchedClientChildCount",
        "unmatchedClientBuyChildCount",
        "unmatchedClientSellChildCount",
        "avgAllHoldMinutes",
        "avgLongHoldMinutes",
        "avgShortHoldMinutes",
        "avgLongExecRet",
        "avgShortExecRet",
        "totalLongExecPnl",
        "totalShortExecPnl",
        "totalMatchedNotional",
        "notionalWeightedExecRet",
        "tradeDateCount",
        "securityCount",
        "securityDayCount",
        "totalBarCount",
        "totalLongSignalCount",
        "totalShortSignalCount",
        "maxConcurrentLongCount",
        "maxConcurrentShortCount",
        "maxConcurrentTotalCount",
        "maxConcurrentLongQty",
        "maxConcurrentShortQty",
        "maxConcurrentTotalQty",
        "avgMaxConcurrentLongCount",
        "avgMaxConcurrentShortCount",
        "avgMaxConcurrentTotalCount",
        "avgMaxConcurrentLongQty",
        "avgMaxConcurrentShortQty",
        "avgMaxConcurrentTotalQty",
        "p95MaxConcurrentLongCount",
        "p95MaxConcurrentShortCount",
        "p95MaxConcurrentTotalCount",
        "p95MaxConcurrentLongQty",
        "p95MaxConcurrentShortQty",
        "p95MaxConcurrentTotalQty",
    ]
    drop_columns = [
        "clientChildMatchRate",
        "clientBuyMatchRate",
        "totalClientBuyQty",
        "totalClientSellQty",
        "totalClientQty",
        "matchedClientQty",
        "matchedClientBuyQty",
        "matchedClientSellQty",
        "clientQtyMatchRate",
        "clientBuyQtyMatchRate",
        "clientSellQtyMatchRate",
        "clientBuyAmtMatchRate",
        "clientSellAmtMatchRate",
        "avgLongMidRet",
        "avgShortMidRet",
        "avgAllMidRet",
        "totalLongMidRet",
        "totalShortMidRet",
        "totalMidRet",
        "totalLongExecRet",
        "totalShortExecRet",
        "totalExecRet",
        "totalLongMidPnl",
        "totalShortMidPnl",
        "totalMidPnl",
        "longMidWinRate",
        "shortMidWinRate",
        "longExecWinRate",
        "shortExecWinRate",
        "notionalWeightedMidRet",
    ]
    return _prune_and_reorder_columns(pool_summary_df, first_columns, drop_columns)
