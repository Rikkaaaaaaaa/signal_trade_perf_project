from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


BASE_VARIANT_FLAG_COLUMNS = {
    "all": None,
    "lt1000": "variantLt1000",
    "lt2000": "variantLt2000",
    "liqcap5tick": "variantLiqcap5tick",
}


def _truthy_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def select_variant_trades(
    trades_df: pd.DataFrame,
    position_cap_trades_df: pd.DataFrame,
    variant_tag: str,
) -> pd.DataFrame:
    if variant_tag in BASE_VARIANT_FLAG_COLUMNS:
        if trades_df.empty:
            return pd.DataFrame()
        flag_column = BASE_VARIANT_FLAG_COLUMNS[variant_tag]
        if flag_column is None:
            return trades_df.copy()
        if flag_column not in trades_df.columns:
            return pd.DataFrame()
        return trades_df[_truthy_mask(trades_df[flag_column])].copy()

    if position_cap_trades_df.empty or "variantTag" not in position_cap_trades_df.columns:
        return pd.DataFrame()
    return position_cap_trades_df[position_cap_trades_df["variantTag"] == variant_tag].copy()


def build_capital_event_rows(
    trades_df: pd.DataFrame,
    position_cap_trades_df: pd.DataFrame,
    variant_tags: Iterable[str],
) -> list[dict[str, Any]]:
    event_rows: list[dict[str, Any]] = []
    for variant_tag in variant_tags:
        variant_trades_df = select_variant_trades(
            trades_df=trades_df,
            position_cap_trades_df=position_cap_trades_df,
            variant_tag=variant_tag,
        )
        if variant_trades_df.empty:
            continue

        required_cols = {"openTime", "closeTime", "openNotional"}
        if not required_cols.issubset(variant_trades_df.columns):
            continue

        notional = pd.to_numeric(variant_trades_df["openNotional"], errors="coerce").abs()
        open_time = pd.to_datetime(variant_trades_df["openTime"], errors="coerce")
        close_time = pd.to_datetime(variant_trades_df["closeTime"], errors="coerce")

        valid_open = open_time.notna() & notional.notna()
        for event_time, value in zip(open_time[valid_open], notional[valid_open]):
            event_rows.append(
                {
                    "variantTag": variant_tag,
                    "eventTime": event_time,
                    "eventOrder": 0,
                    "capitalDelta": float(value),
                }
            )

        valid_close = close_time.notna() & notional.notna()
        for event_time, value in zip(close_time[valid_close], notional[valid_close]):
            event_rows.append(
                {
                    "variantTag": variant_tag,
                    "eventTime": event_time,
                    "eventOrder": 1,
                    "capitalDelta": -float(value),
                }
            )

    return event_rows


def capital_metrics_from_events(event_df: pd.DataFrame) -> dict[str, float]:
    if event_df.empty:
        return {"maxCapitalUsed": 0.0, "p95CapitalUsedByEvent": 0.0}

    sorted_df = event_df.sort_values(["eventTime", "eventOrder"]).reset_index(drop=True)
    cumulative = pd.to_numeric(sorted_df["capitalDelta"], errors="coerce").fillna(0.0).cumsum()
    cumulative = cumulative.clip(lower=0.0)
    if cumulative.empty:
        return {"maxCapitalUsed": 0.0, "p95CapitalUsedByEvent": 0.0}

    return {
        "maxCapitalUsed": float(cumulative.max()),
        "p95CapitalUsedByEvent": float(np.percentile(cumulative.to_numpy(dtype=float), 95)),
    }


def capital_metrics_by_variant(
    event_rows: list[dict[str, Any]],
    base_row: dict[str, Any],
    variant_tags: Iterable[str],
) -> list[dict[str, Any]]:
    event_df = pd.DataFrame(event_rows)
    rows: list[dict[str, Any]] = []
    for variant_tag in variant_tags:
        variant_event_df = (
            event_df[event_df["variantTag"] == variant_tag]
            if not event_df.empty and "variantTag" in event_df.columns
            else pd.DataFrame()
        )
        rows.append(
            {
                **base_row,
                "variantTag": variant_tag,
                **capital_metrics_from_events(variant_event_df),
            }
        )
    return rows
