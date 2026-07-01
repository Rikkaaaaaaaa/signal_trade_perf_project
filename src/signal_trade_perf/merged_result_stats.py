from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .low_price_internalization import LOW_PRICE_VARIANTS


PREDICTION_VARIANTS = [
    "all",
    "lt1000",
    "lt2000",
    "liqcap5tick",
    "poscap_min5",
    "poscap_avg5",
    "poscap_avg5x5_partial",
]

PRICE_BUCKET_SPECS: list[tuple[float, float | None, str]] = [
    (5.0, 10.0, "5-10"),
    (10.0, 15.0, "10-15"),
    (15.0, 20.0, "15-20"),
    (20.0, 30.0, "20-30"),
    (30.0, 50.0, "30-50"),
    (50.0, 100.0, "50-100"),
    (100.0, None, "100+"),
]


def default_merged_variant_tags() -> list[str]:
    return [
        f"{prediction_variant}__{fill_variant}"
        for prediction_variant in PREDICTION_VARIANTS
        for fill_variant in LOW_PRICE_VARIANTS
    ]


def normalize_variant_tags(variant_tags: Iterable[str] | None) -> list[str]:
    valid_tags = set(default_merged_variant_tags())
    tags = list(variant_tags or [])
    if not tags:
        return sorted(valid_tags)
    unknown_tags = sorted(set(tags) - valid_tags)
    if unknown_tags:
        raise ValueError(f"Unknown merged variant tags: {unknown_tags}")
    return tags


def assign_price_bucket_fields(open_mid_price: pd.Series) -> pd.DataFrame:
    numeric_price = pd.to_numeric(open_mid_price, errors="coerce")
    labels: list[str] = []
    orders: list[int] = []
    for value in numeric_price:
        if pd.isna(value):
            labels.append("unknown")
            orders.append(-1)
            continue
        matched = False
        for idx, (lower, upper, label) in enumerate(PRICE_BUCKET_SPECS):
            if value >= lower and (upper is None or value < upper):
                labels.append(label)
                orders.append(idx)
                matched = True
                break
        if not matched:
            labels.append("other")
            orders.append(len(PRICE_BUCKET_SPECS))
    return pd.DataFrame({"priceBucketLabel": labels, "priceBucketOrder": orders})


def _build_base_security_frame(
    pool_name: str,
    trade_date: str,
    meta_df: pd.DataFrame,
    client_order_df: pd.DataFrame,
) -> pd.DataFrame:
    client_amt_df = (
        client_order_df.groupby("securityCode", as_index=False)
        .agg(totalClientAmt=("clientFilledAmt", "sum"))
        if not client_order_df.empty
        else pd.DataFrame(columns=["securityCode", "totalClientAmt"])
    )
    meta_cols = [
        col for col in ["securityCode", "openMidPrice", "priceBucketLow", "priceBucketHigh", "prevDayVol"]
        if col in meta_df.columns
    ]
    meta_slice_df = meta_df[meta_cols].drop_duplicates("securityCode") if meta_cols else pd.DataFrame(columns=["securityCode"])
    base_df = client_amt_df.merge(meta_slice_df, on="securityCode", how="left")
    if base_df.empty and not meta_slice_df.empty:
        base_df = meta_slice_df.copy()
        base_df["totalClientAmt"] = 0.0
    if base_df.empty:
        base_df = pd.DataFrame(
            columns=["securityCode", "totalClientAmt", "openMidPrice", "priceBucketLow", "priceBucketHigh", "prevDayVol"]
        )
    bucket_df = assign_price_bucket_fields(base_df.get("openMidPrice", pd.Series(dtype=float)))
    base_df = pd.concat([base_df.reset_index(drop=True), bucket_df], axis=1)
    base_df["poolName"] = pool_name
    base_df["tradeDate"] = trade_date
    return base_df


def _build_prediction_security_metrics(prediction_security_summary_df: pd.DataFrame) -> pd.DataFrame:
    if prediction_security_summary_df.empty:
        return pd.DataFrame(columns=["securityCode", "variantTag"])
    keep_cols = [
        "securityCode",
        "variantTag",
        "totalTradeCount",
        "totalExecPnl",
        "totalMatchedNotional",
        "matchedClientAmt",
    ]
    available_cols = [col for col in keep_cols if col in prediction_security_summary_df.columns]
    result_df = prediction_security_summary_df[available_cols].copy()
    return result_df.rename(
        columns={
            "variantTag": "predictionVariantTag",
            "totalTradeCount": "predictionTradeCount",
            "totalExecPnl": "predictionExecPnl",
            "totalMatchedNotional": "predictionMatchedNotional",
            "matchedClientAmt": "predictionMatchedClientAmt",
        }
    )


def _build_fill_rate_security_metrics(fill_rate_trades_df: pd.DataFrame) -> pd.DataFrame:
    if fill_rate_trades_df.empty:
        return pd.DataFrame(columns=["securityCode", "fillRateVariantTag"])
    grouped = (
        fill_rate_trades_df.groupby(["securityCode", "variantTag"], as_index=False)
        .agg(
            fillRateTradeCount=("variantTag", "size"),
            fillRateExecPnl=("execPnl", "sum"),
            fillRateMatchedNotional=("openNotional", "sum"),
            fillRateMatchedClientAmt=("clientFilledAmt", "sum"),
        )
    )
    return grouped.rename(columns={"variantTag": "fillRateVariantTag"})


def build_merged_security_summary(
    pool_name: str,
    trade_date: str,
    meta_df: pd.DataFrame,
    client_order_df: pd.DataFrame,
    prediction_security_summary_df: pd.DataFrame,
    fill_rate_trades_df: pd.DataFrame,
    variant_tags: Iterable[str] | None = None,
) -> pd.DataFrame:
    requested_variant_tags = normalize_variant_tags(variant_tags)
    if not requested_variant_tags:
        return pd.DataFrame()

    base_df = _build_base_security_frame(
        pool_name=pool_name,
        trade_date=trade_date,
        meta_df=meta_df,
        client_order_df=client_order_df,
    )
    if base_df.empty:
        return pd.DataFrame()

    prediction_metrics_df = _build_prediction_security_metrics(prediction_security_summary_df)
    fill_rate_metrics_df = _build_fill_rate_security_metrics(fill_rate_trades_df)
    rows: list[pd.DataFrame] = []

    for merged_variant_tag in requested_variant_tags:
        prediction_variant_tag, fill_rate_variant_tag = merged_variant_tag.split("__", maxsplit=1)
        prediction_variant_df = (
            prediction_metrics_df[prediction_metrics_df["predictionVariantTag"] == prediction_variant_tag].drop(columns=["predictionVariantTag"])
            if not prediction_metrics_df.empty
            else pd.DataFrame(columns=["securityCode"])
        )
        fill_rate_variant_df = (
            fill_rate_metrics_df[fill_rate_metrics_df["fillRateVariantTag"] == fill_rate_variant_tag].drop(columns=["fillRateVariantTag"])
            if not fill_rate_metrics_df.empty
            else pd.DataFrame(columns=["securityCode"])
        )
        merged_df = base_df.merge(prediction_variant_df, on="securityCode", how="left").merge(fill_rate_variant_df, on="securityCode", how="left")
        fill_zero_cols = [
            "predictionTradeCount",
            "predictionExecPnl",
            "predictionMatchedNotional",
            "predictionMatchedClientAmt",
            "fillRateTradeCount",
            "fillRateExecPnl",
            "fillRateMatchedNotional",
            "fillRateMatchedClientAmt",
        ]
        for col in fill_zero_cols:
            if col not in merged_df.columns:
                merged_df[col] = 0.0
        merged_df[fill_zero_cols] = merged_df[fill_zero_cols].fillna(0.0)
        merged_df["predictionTradeCount"] = merged_df["predictionTradeCount"].astype(int)
        merged_df["fillRateTradeCount"] = merged_df["fillRateTradeCount"].astype(int)
        merged_df["variantTag"] = merged_variant_tag
        merged_df["predictionVariantTag"] = prediction_variant_tag
        merged_df["fillRateVariantTag"] = fill_rate_variant_tag
        merged_df["matchedClientAmt"] = merged_df["predictionMatchedClientAmt"] + merged_df["fillRateMatchedClientAmt"]
        merged_df["clientAmtMatchRate"] = np.where(
            pd.to_numeric(merged_df["totalClientAmt"], errors="coerce").fillna(0.0) == 0,
            np.nan,
            merged_df["matchedClientAmt"] / merged_df["totalClientAmt"],
        )
        merged_df["totalTradeCount"] = merged_df["predictionTradeCount"] + merged_df["fillRateTradeCount"]
        merged_df["totalExecPnl"] = merged_df["predictionExecPnl"] + merged_df["fillRateExecPnl"]
        merged_df["totalMatchedNotional"] = merged_df["predictionMatchedNotional"] + merged_df["fillRateMatchedNotional"]
        merged_df["notionalWeightedExecRet"] = np.where(
            pd.to_numeric(merged_df["totalMatchedNotional"], errors="coerce").fillna(0.0) == 0,
            np.nan,
            merged_df["totalExecPnl"] / merged_df["totalMatchedNotional"],
        )
        merged_df["predictionNotionalWeightedExecRet"] = np.where(
            pd.to_numeric(merged_df["predictionMatchedNotional"], errors="coerce").fillna(0.0) == 0,
            np.nan,
            merged_df["predictionExecPnl"] / merged_df["predictionMatchedNotional"],
        )
        merged_df["fillRateNotionalWeightedExecRet"] = np.where(
            pd.to_numeric(merged_df["fillRateMatchedNotional"], errors="coerce").fillna(0.0) == 0,
            np.nan,
            merged_df["fillRateExecPnl"] / merged_df["fillRateMatchedNotional"],
        )
        rows.append(merged_df)

    result_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if result_df.empty:
        return result_df
    ordered_cols = [
        "poolName",
        "tradeDate",
        "securityCode",
        "variantTag",
        "predictionVariantTag",
        "fillRateVariantTag",
        "openMidPrice",
        "priceBucketLabel",
        "priceBucketOrder",
        "priceBucketLow",
        "priceBucketHigh",
        "prevDayVol",
        "totalClientAmt",
        "matchedClientAmt",
        "clientAmtMatchRate",
        "totalTradeCount",
        "totalExecPnl",
        "totalMatchedNotional",
        "notionalWeightedExecRet",
        "predictionTradeCount",
        "predictionExecPnl",
        "predictionMatchedNotional",
        "predictionMatchedClientAmt",
        "predictionNotionalWeightedExecRet",
        "fillRateTradeCount",
        "fillRateExecPnl",
        "fillRateMatchedNotional",
        "fillRateMatchedClientAmt",
        "fillRateNotionalWeightedExecRet",
    ]
    keep_cols = [col for col in ordered_cols if col in result_df.columns]
    return result_df[keep_cols].sort_values(["variantTag", "poolName", "tradeDate", "priceBucketOrder", "securityCode"]).reset_index(drop=True)


def aggregate_merged_security_by_price_bucket(
    merged_security_df: pd.DataFrame,
    variant_tags: Iterable[str] | None = None,
    include_trade_date: bool = False,
) -> pd.DataFrame:
    if merged_security_df.empty:
        return pd.DataFrame()
    filtered_df = merged_security_df.copy()
    requested_variant_tags = normalize_variant_tags(variant_tags)
    if requested_variant_tags:
        filtered_df = filtered_df[filtered_df["variantTag"].isin(requested_variant_tags)].copy()
    if filtered_df.empty:
        return filtered_df
    group_cols = [
        "poolName",
        "variantTag",
        "predictionVariantTag",
        "fillRateVariantTag",
        "priceBucketOrder",
        "priceBucketLabel",
    ]
    if include_trade_date and "tradeDate" in filtered_df.columns:
        group_cols.insert(1, "tradeDate")
    grouped = (
        filtered_df.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            securityCount=("securityCode", "nunique"),
            totalClientAmt=("totalClientAmt", "sum"),
            matchedClientAmt=("matchedClientAmt", "sum"),
            totalTradeCount=("totalTradeCount", "sum"),
            totalExecPnl=("totalExecPnl", "sum"),
            totalMatchedNotional=("totalMatchedNotional", "sum"),
            predictionTradeCount=("predictionTradeCount", "sum"),
            predictionExecPnl=("predictionExecPnl", "sum"),
            predictionMatchedNotional=("predictionMatchedNotional", "sum"),
            fillRateTradeCount=("fillRateTradeCount", "sum"),
            fillRateExecPnl=("fillRateExecPnl", "sum"),
            fillRateMatchedNotional=("fillRateMatchedNotional", "sum"),
        )
    )
    grouped["clientAmtMatchRate"] = np.where(
        grouped["totalClientAmt"] == 0,
        np.nan,
        grouped["matchedClientAmt"] / grouped["totalClientAmt"],
    )
    grouped["notionalWeightedExecRet"] = np.where(
        grouped["totalMatchedNotional"] == 0,
        np.nan,
        grouped["totalExecPnl"] / grouped["totalMatchedNotional"],
    )
    grouped["predictionNotionalWeightedExecRet"] = np.where(
        grouped["predictionMatchedNotional"] == 0,
        np.nan,
        grouped["predictionExecPnl"] / grouped["predictionMatchedNotional"],
    )
    grouped["fillRateNotionalWeightedExecRet"] = np.where(
        grouped["fillRateMatchedNotional"] == 0,
        np.nan,
        grouped["fillRateExecPnl"] / grouped["fillRateMatchedNotional"],
    )
    sort_cols = [col for col in ["variantTag", "poolName", "tradeDate", "priceBucketOrder"] if col in grouped.columns]
    return grouped.sort_values(sort_cols).reset_index(drop=True)
