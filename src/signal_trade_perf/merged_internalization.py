from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .core import BacktestParams
from .internalization import run_internalization_prepared_day
from .internalization_capital import capital_metrics_from_events, select_variant_trades
from .low_price_internalization import LOW_PRICE_VARIANTS, LowPriceBacktestParams, run_low_price_prepared_day


PREDICTION_VARIANTS = [
    "all",
    "lt1000",
    "lt2000",
    "liqcap5tick",
    "poscap_min5",
    "poscap_avg5",
    "poscap_avg5x5_partial",
]


@dataclass(frozen=True)
class MergedBacktestParams:
    prediction_params: BacktestParams
    fill_rate_signal_ranks: tuple[int, ...] = (1, 2)
    fill_rate_support_threshold: float = 0.0
    match_window_seconds: int | None = 10
    fill_rate_spread: float = 0.01

    @property
    def fill_rate_params(self) -> LowPriceBacktestParams:
        return LowPriceBacktestParams(
            signal_ranks=self.fill_rate_signal_ranks,
            match_window_seconds=self.match_window_seconds,
            spread=self.fill_rate_spread,
        )

    @property
    def param_tag(self) -> str:
        return (
            f"{self.prediction_params.param_tag}_"
            f"fill_rank_{'_'.join(str(rank) for rank in self.fill_rate_signal_ranks)}_"
            f"support_{self.fill_rate_support_threshold:g}_"
            f"{'match_unlimited' if self.match_window_seconds is None else f'match_{self.match_window_seconds}'}_"
            f"fill_spread_{self.fill_rate_spread:g}"
        )


def _prediction_opens_position(signal_value: float, client_side: str, open_threshold: float) -> bool:
    if pd.isna(signal_value):
        return False
    if client_side == "B":
        return float(signal_value) < -float(open_threshold)
    return float(signal_value) > float(open_threshold)


def _prediction_supports_fill(signal_value: float, client_side: str, support_threshold: float) -> bool:
    if pd.isna(signal_value):
        return False
    if client_side == "B":
        return float(signal_value) < -float(support_threshold)
    return float(signal_value) > float(support_threshold)


def _eligible_fill_rank_mask(signal_df: pd.DataFrame, client_side: str, ranks: tuple[int, ...]) -> pd.Series:
    rank_values = {abs(int(rank)) for rank in ranks}
    if client_side == "S":
        return signal_df["bsFlag"].eq("s") & signal_df["mergeSignal"].abs().isin(rank_values) & signal_df["mergeSignal"].lt(0)
    return signal_df["bsFlag"].eq("b") & signal_df["mergeSignal"].abs().isin(rank_values) & signal_df["mergeSignal"].gt(0)


def _latest_prediction_signal(
    sec_signal_df: pd.DataFrame,
    order_time: pd.Timestamp,
    match_window_seconds: int | None,
) -> tuple[pd.Series | None, int]:
    if sec_signal_df.empty:
        return None, -1
    signal_times = sec_signal_df["barTime"].to_numpy(dtype="datetime64[ns]")
    matched_idx = int(signal_times.searchsorted(order_time.to_datetime64(), side="right") - 1)
    if matched_idx < 0:
        return None, -1
    matched_row = sec_signal_df.iloc[matched_idx]
    if match_window_seconds is not None:
        window_start = order_time - pd.Timedelta(seconds=match_window_seconds)
        if pd.Timestamp(matched_row.barTime) < window_start:
            return None, -1
    return matched_row, matched_idx


def _latest_fill_signal(
    sec_signal_df: pd.DataFrame,
    order_time: pd.Timestamp,
    client_side: str,
    ranks: tuple[int, ...],
    match_window_seconds: int | None,
) -> tuple[pd.Series | None, pd.Timestamp | None]:
    if sec_signal_df.empty:
        return None, None
    signal_times = sec_signal_df["barTime"].drop_duplicates().to_numpy(dtype="datetime64[ns]")
    matched_pos = int(signal_times.searchsorted(order_time.to_datetime64(), side="right") - 1)
    if matched_pos < 0:
        return None, None
    matched_time = pd.Timestamp(signal_times[matched_pos])
    if match_window_seconds is not None and matched_time < order_time - pd.Timedelta(seconds=match_window_seconds):
        return None, None
    same_bar_df = sec_signal_df[sec_signal_df["barTime"].eq(matched_time)]
    eligible_df = same_bar_df[_eligible_fill_rank_mask(same_bar_df, client_side, ranks)]
    if eligible_df.empty:
        return None, matched_time
    return eligible_df.iloc[0], matched_time


def route_merged_orders(
    prediction_signal_df: pd.DataFrame,
    fill_rate_signal_df: pd.DataFrame,
    client_order_df: pd.DataFrame,
    params: MergedBacktestParams,
) -> pd.DataFrame:
    route_rows: list[dict[str, Any]] = []
    prediction_by_security = {
        security_code: sec_df.sort_values("barTime").reset_index(drop=True)
        for security_code, sec_df in prediction_signal_df.groupby("securityCode", sort=True)
    }
    fill_by_security = {
        security_code: sec_df.sort_values(["barTime", "bsFlag"]).reset_index(drop=True)
        for security_code, sec_df in fill_rate_signal_df.groupby("securityCode", sort=True)
    }
    for security_code, sec_orders_df in client_order_df.groupby("securityCode", sort=True):
        sec_prediction_df = prediction_by_security.get(security_code, pd.DataFrame())
        sec_fill_df = fill_by_security.get(security_code, pd.DataFrame())
        prediction_by_bar_time = {
            pd.Timestamp(row.barTime): row
            for row in sec_prediction_df.itertuples(index=False)
        }
        for order in sec_orders_df.sort_values("clientOrderTime").to_dict(orient="records"):
            order_time = pd.Timestamp(order["clientOrderTime"])
            client_side = str(order["clientSide"]).upper()
            prediction_row, prediction_idx = _latest_prediction_signal(
                sec_signal_df=sec_prediction_df,
                order_time=order_time,
                match_window_seconds=params.match_window_seconds,
            )
            prediction_signal = float(prediction_row.merge_signal) if prediction_row is not None else np.nan
            prediction_open = (
                prediction_row is not None
                and _prediction_opens_position(
                    signal_value=prediction_signal,
                    client_side=client_side,
                    open_threshold=params.prediction_params.open_threshold,
                )
            )

            fill_row, fill_bar_time = _latest_fill_signal(
                sec_signal_df=sec_fill_df,
                order_time=order_time,
                client_side=client_side,
                ranks=params.fill_rate_signal_ranks,
                match_window_seconds=params.match_window_seconds,
            )
            fill_prediction_row = prediction_by_bar_time.get(fill_bar_time) if fill_bar_time is not None else None
            fill_prediction_signal = (
                float(fill_prediction_row.merge_signal)
                if fill_prediction_row is not None
                else np.nan
            )
            fill_supported = (
                fill_row is not None
                and fill_prediction_row is not None
                and not _prediction_opens_position(
                    signal_value=fill_prediction_signal,
                    client_side=client_side,
                    open_threshold=params.prediction_params.open_threshold,
                )
                and _prediction_supports_fill(
                    signal_value=fill_prediction_signal,
                    client_side=client_side,
                    support_threshold=params.fill_rate_support_threshold,
                )
            )

            if prediction_open:
                route_source = "prediction"
                route_status = "prediction_open_signal"
            elif fill_supported:
                route_source = "fill_rate"
                route_status = "fill_rate_supported_signal"
            else:
                route_source = "unmatched"
                route_status = "no_route_signal"

            route_rows.append(
                {
                    **order,
                    "routeSource": route_source,
                    "routeStatus": route_status,
                    "predictionSignalTime": pd.Timestamp(prediction_row.barTime) if prediction_row is not None else pd.NaT,
                    "predictionSignalTimeInt": int(prediction_row.signalTime) if prediction_row is not None else np.nan,
                    "predictionSignal": prediction_signal,
                    "predictionSignalRowIdx": prediction_idx,
                    "fillRateSignalTime": pd.Timestamp(fill_row.barTime) if fill_row is not None else pd.NaT,
                    "fillRateSignalTimeInt": int(fill_row.signalTime) if fill_row is not None else np.nan,
                    "fillRateSignal": float(fill_row.mergeSignal) if fill_row is not None else np.nan,
                    "fillRateBsFlag": str(fill_row.bsFlag) if fill_row is not None else "",
                    "fillRateYTest": int(fill_row.yTest) if fill_row is not None else np.nan,
                    "fillRateSupportSignal": fill_prediction_signal,
                    "matchWindowSeconds": (
                        "unlimited" if params.match_window_seconds is None else str(params.match_window_seconds)
                    ),
                }
            )
    return pd.DataFrame(route_rows)


def _empty_prediction_summary_row(
    pool_name: str,
    trade_date: str,
    variant_tag: str,
    params: MergedBacktestParams,
) -> dict[str, Any]:
    return {
        "poolName": pool_name,
        "tradeDate": trade_date,
        "variantTag": variant_tag,
        "paramTag": params.prediction_params.param_tag,
        "totalTradeCount": 0,
        "totalExecPnl": 0.0,
        "totalMatchedNotional": 0.0,
        "matchedClientAmt": 0.0,
        "maxCapitalUsed": 0.0,
        "p95CapitalUsedByEvent": 0.0,
        "maxDailyCapitalUsed": 0.0,
        "p95DailyCapitalUsed": 0.0,
        "avgDailyCapitalUsed": 0.0,
        "capitalAdjustedReturn": np.nan,
        "clientAmtMatchRate": np.nan,
        "notionalWeightedExecRet": np.nan,
    }


def _normalize_prediction_summary(
    pool_name: str,
    trade_date: str,
    params: MergedBacktestParams,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant_tag in PREDICTION_VARIANTS:
        if summary_df.empty or "variantTag" not in summary_df.columns:
            rows.append(_empty_prediction_summary_row(pool_name, trade_date, variant_tag, params))
            continue
        match_df = summary_df[summary_df["variantTag"] == variant_tag]
        if match_df.empty:
            rows.append(_empty_prediction_summary_row(pool_name, trade_date, variant_tag, params))
            continue
        row = _empty_prediction_summary_row(pool_name, trade_date, variant_tag, params)
        for key in row:
            if key in match_df.columns:
                row[key] = match_df.iloc[0][key]
        rows.append(row)
    return pd.DataFrame(rows)


def _trade_df_to_capital_metrics(trade_df: pd.DataFrame) -> dict[str, float]:
    if trade_df.empty:
        return {"maxCapitalUsed": 0.0, "p95CapitalUsedByEvent": 0.0}
    event_rows: list[dict[str, Any]] = []
    open_time = pd.to_datetime(trade_df["openTime"], errors="coerce")
    close_time = pd.to_datetime(trade_df["closeTime"], errors="coerce")
    notional = pd.to_numeric(trade_df["openNotional"], errors="coerce").abs()
    valid_open = open_time.notna() & notional.notna()
    valid_close = close_time.notna() & notional.notna()
    for event_time, value in zip(open_time[valid_open], notional[valid_open]):
        event_rows.append({"eventTime": event_time, "eventOrder": 0, "capitalDelta": float(value)})
    for event_time, value in zip(close_time[valid_close], notional[valid_close]):
        event_rows.append({"eventTime": event_time, "eventOrder": 1, "capitalDelta": -float(value)})
    return capital_metrics_from_events(pd.DataFrame(event_rows))


def _capital_metric_trade_slice(trade_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame(columns=["openTime", "closeTime", "openNotional"])
    keep_cols = [col for col in ["openTime", "closeTime", "openNotional"] if col in trade_df.columns]
    if len(keep_cols) < 3:
        missing_cols = [col for col in ["openTime", "closeTime", "openNotional"] if col not in keep_cols]
        return trade_df[keep_cols].assign(**{col: np.nan for col in missing_cols})[["openTime", "closeTime", "openNotional"]]
    return trade_df[["openTime", "closeTime", "openNotional"]].copy()


def _select_prediction_variant_trades(
    trades_df: pd.DataFrame,
    variant_tag: str,
) -> pd.DataFrame:
    position_cap_trades_df = trades_df.attrs.get("position_cap_trades", pd.DataFrame())
    return select_variant_trades(
        trades_df=trades_df,
        position_cap_trades_df=position_cap_trades_df,
        variant_tag=variant_tag,
    )


def build_merged_summary(
    pool_name: str,
    trade_date: str,
    client_order_df: pd.DataFrame,
    params: MergedBacktestParams,
    prediction_summary_df: pd.DataFrame,
    prediction_trades_df: pd.DataFrame,
    fill_rate_summary_df: pd.DataFrame,
    fill_rate_trades_df: pd.DataFrame,
) -> pd.DataFrame:
    total_client_amt = float(client_order_df["clientFilledAmt"].sum()) if not client_order_df.empty else 0.0
    prediction_summary_df = _normalize_prediction_summary(pool_name, trade_date, params, prediction_summary_df)
    fill_rows = {
        row["variantTag"]: row
        for row in fill_rate_summary_df.to_dict(orient="records")
    } if not fill_rate_summary_df.empty else {}
    merged_rows: list[dict[str, Any]] = []
    for prediction_variant in PREDICTION_VARIANTS:
        prediction_row = prediction_summary_df[prediction_summary_df["variantTag"] == prediction_variant].iloc[0].to_dict()
        prediction_variant_trades = _select_prediction_variant_trades(prediction_trades_df, prediction_variant)
        for fill_variant in LOW_PRICE_VARIANTS:
            fill_row = fill_rows.get(
                fill_variant,
                {
                    "variantTag": fill_variant,
                    "totalTradeCount": 0,
                    "totalExecPnl": 0.0,
                    "totalMatchedNotional": 0.0,
                    "matchedClientAmt": 0.0,
                    "maxCapitalUsed": 0.0,
                    "p95CapitalUsedByEvent": 0.0,
                    "capitalAdjustedReturn": np.nan,
                    "clientAmtMatchRate": np.nan,
                    "notionalWeightedExecRet": np.nan,
                    "yTestWinRate": np.nan,
                },
            )
            fill_variant_trades = (
                fill_rate_trades_df[fill_rate_trades_df["variantTag"] == fill_variant].copy()
                if not fill_rate_trades_df.empty and "variantTag" in fill_rate_trades_df.columns
                else pd.DataFrame()
            )
            combined_trades_df = pd.concat(
                [
                    _capital_metric_trade_slice(prediction_variant_trades),
                    _capital_metric_trade_slice(fill_variant_trades),
                ],
                ignore_index=True,
            ) if not prediction_variant_trades.empty or not fill_variant_trades.empty else pd.DataFrame(columns=["openTime", "closeTime", "openNotional"])
            merged_capital = _trade_df_to_capital_metrics(combined_trades_df)
            total_exec_pnl = float(prediction_row["totalExecPnl"]) + float(fill_row["totalExecPnl"])
            total_matched_notional = float(prediction_row["totalMatchedNotional"]) + float(fill_row["totalMatchedNotional"])
            matched_client_amt = float(prediction_row["matchedClientAmt"]) + float(fill_row["matchedClientAmt"])
            merged_rows.append(
                {
                    "poolName": pool_name,
                    "tradeDate": trade_date,
                    "variantTag": f"{prediction_variant}__{fill_variant}",
                    "predictionVariantTag": prediction_variant,
                    "fillRateVariantTag": fill_variant,
                    "paramTag": params.param_tag,
                    "openThreshold": params.prediction_params.open_threshold,
                    "closeThreshold": params.prediction_params.close_threshold,
                    "minHoldBars": params.prediction_params.min_hold_bars,
                    "relaxedCloseAfterBars": params.prediction_params.relaxed_close_after_bars,
                    "relaxedCloseThreshold": params.prediction_params.relaxed_close_threshold,
                    "fillRateSignalRanks": ",".join(str(rank) for rank in params.fill_rate_signal_ranks),
                    "fillRateSupportThreshold": params.fill_rate_support_threshold,
                    "matchWindowSeconds": (
                        "unlimited" if params.match_window_seconds is None else str(params.match_window_seconds)
                    ),
                    "fillRateSpread": params.fill_rate_spread,
                    "totalClientAmt": total_client_amt,
                    "matchedClientAmt": matched_client_amt,
                    "clientAmtMatchRate": np.nan if total_client_amt == 0 else matched_client_amt / total_client_amt,
                    "totalTradeCount": int(prediction_row["totalTradeCount"]) + int(fill_row["totalTradeCount"]),
                    "totalExecPnl": total_exec_pnl,
                    "totalMatchedNotional": total_matched_notional,
                    "notionalWeightedExecRet": (
                        np.nan if total_matched_notional == 0 else total_exec_pnl / total_matched_notional
                    ),
                    "predictionTradeCount": int(prediction_row["totalTradeCount"]),
                    "predictionExecPnl": float(prediction_row["totalExecPnl"]),
                    "predictionMatchedNotional": float(prediction_row["totalMatchedNotional"]),
                    "predictionMatchedClientAmt": float(prediction_row["matchedClientAmt"]),
                    "predictionMaxCapitalUsed": float(prediction_row["maxCapitalUsed"]),
                    "predictionP95CapitalUsedByEvent": float(prediction_row["p95CapitalUsedByEvent"]),
                    "predictionCapitalAdjustedReturn": prediction_row.get("capitalAdjustedReturn", np.nan),
                    "fillRateTradeCount": int(fill_row["totalTradeCount"]),
                    "fillRateExecPnl": float(fill_row["totalExecPnl"]),
                    "fillRateMatchedNotional": float(fill_row["totalMatchedNotional"]),
                    "fillRateMatchedClientAmt": float(fill_row["matchedClientAmt"]),
                    "fillRateMaxCapitalUsed": float(fill_row["maxCapitalUsed"]),
                    "fillRateP95CapitalUsedByEvent": float(fill_row["p95CapitalUsedByEvent"]),
                    "fillRateCapitalAdjustedReturn": fill_row.get("capitalAdjustedReturn", np.nan),
                    "fillRateYTestWinRate": fill_row.get("yTestWinRate", np.nan),
                    "mergedMaxCapitalUsed": float(merged_capital["maxCapitalUsed"]),
                    "mergedP95CapitalUsedByEvent": float(merged_capital["p95CapitalUsedByEvent"]),
                    "mergedCapitalAdjustedReturn": (
                        np.nan if float(merged_capital["maxCapitalUsed"]) == 0 else total_exec_pnl / float(merged_capital["maxCapitalUsed"])
                    ),
                }
            )
    return pd.DataFrame(merged_rows)


def run_merged_prepared_day(
    prepared_inputs: dict[str, object],
    params: MergedBacktestParams,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trade_date = str(prepared_inputs["tradeDate"])
    pool_name = str(prepared_inputs["poolName"])
    client_order_df = prepared_inputs["clientOrderDf"]
    prediction_signal_df = prepared_inputs["predictionSignalDf"]
    fill_rate_signal_df = prepared_inputs["fillRateSignalDf"]

    route_df = route_merged_orders(
        prediction_signal_df=prediction_signal_df,
        fill_rate_signal_df=fill_rate_signal_df,
        client_order_df=client_order_df,
        params=params,
    )
    prediction_orders_df = route_df[route_df["routeSource"] == "prediction"].copy()
    fill_rate_orders_df = route_df[route_df["routeSource"] == "fill_rate"].copy()

    prediction_order_events_df, prediction_trades_df, prediction_security_summary_df, prediction_pool_summary_df = run_internalization_prepared_day(
        prepared_inputs={
            "tradeDate": trade_date,
            "poolName": pool_name,
            "signalDf": prediction_signal_df,
            "metaDf": prepared_inputs["metaDf"],
            "clientOrderDf": prediction_orders_df,
            "closePriceMap": prepared_inputs["closePriceMap"],
            "limitMap": prepared_inputs["limitMap"],
        },
        params=params.prediction_params,
        match_window_seconds=params.match_window_seconds,
    )
    fill_rate_order_events_df, fill_rate_trades_df, fill_rate_summary_df = run_low_price_prepared_day(
        prepared_inputs={
            "tradeDate": trade_date,
            "poolName": pool_name,
            "signalDf": fill_rate_signal_df,
            "clientOrderDf": fill_rate_orders_df,
        },
        params=params.fill_rate_params,
    )

    if not prediction_order_events_df.empty:
        prediction_order_events_df = prediction_order_events_df.copy()
        prediction_order_events_df["signalSource"] = "prediction"
    if not prediction_trades_df.empty:
        prediction_trades_df = prediction_trades_df.copy()
        prediction_trades_df["signalSource"] = "prediction"
    if not prediction_security_summary_df.empty:
        prediction_security_summary_df = prediction_security_summary_df.copy()
        prediction_security_summary_df["signalSource"] = "prediction"
    if not prediction_pool_summary_df.empty:
        prediction_pool_summary_df = prediction_pool_summary_df.copy()
        prediction_pool_summary_df["signalSource"] = "prediction"

    if not fill_rate_order_events_df.empty:
        fill_rate_order_events_df = fill_rate_order_events_df.copy()
        fill_rate_order_events_df["signalSource"] = "fill_rate"
    if not fill_rate_trades_df.empty:
        fill_rate_trades_df = fill_rate_trades_df.copy()
        fill_rate_trades_df["signalSource"] = "fill_rate"
    if not fill_rate_summary_df.empty:
        fill_rate_summary_df = fill_rate_summary_df.copy()
        fill_rate_summary_df["signalSource"] = "fill_rate"

    merged_summary_df = build_merged_summary(
        pool_name=pool_name,
        trade_date=trade_date,
        client_order_df=client_order_df,
        params=params,
        prediction_summary_df=prediction_pool_summary_df,
        prediction_trades_df=prediction_trades_df,
        fill_rate_summary_df=fill_rate_summary_df,
        fill_rate_trades_df=fill_rate_trades_df,
    )
    return (
        route_df,
        prediction_order_events_df,
        prediction_trades_df,
        fill_rate_order_events_df,
        fill_rate_trades_df,
        fill_rate_summary_df,
        merged_summary_df,
        prediction_pool_summary_df,
    )
