from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .configs import DdbConfig, MysqlConfig
from .core import BacktestParams
from .internalization_data import (
    discover_ims_security_codes,
    get_default_ims_roots,
    load_a_share_limit_df,
    load_aligned_price_df,
    load_ims_child_orders,
    load_pool_universe_mysql,
    load_previous_aligned_price_df,
    load_signal_day_for_internalization,
    validate_a_share_limit_data,
)
from .internalization_output import (
    format_internalization_pool_summary_for_output,
    format_internalization_security_summary_for_output,
)
from .io_utils import connect_ddb, fetch_tick_mid_ddb


CONTINUOUS_AUCTION_WINDOWS_A_SHARE = [("09:30:00", "11:30:00"), ("13:00:00", "14:57:00")]
A_SHARE_LIMIT_BUFFER_TICKS = 5
A_SHARE_TICK_SIZE = 0.01

# 文件分工说明：
# - internalization_data.py：负责 IMS / MySQL / DDB 的数据读取和代码映射。
# - internalization_output.py：负责最终 CSV 输出列的裁剪和排序。
# - 当前文件：只保留 meta 构造、tick 缓存、开平仓模拟、variant 汇总和单日入口。


def _weighted_average(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    valid = df[[value_col, weight_col]].dropna()
    if valid.empty or valid[weight_col].sum() == 0:
        return np.nan
    return float(np.average(valid[value_col], weights=valid[weight_col]))


def _series_p95(df: pd.DataFrame, value_col: str) -> float:
    valid = df[value_col].dropna()
    if valid.empty:
        return np.nan
    return float(valid.quantile(0.95))


def _position_identity(position: dict[str, object]) -> tuple[str, str, str, pd.Timestamp]:
    # 平仓后要把仓位从 open list 里准确移除，不能只靠 clientOrderId，
    # 否则不同母单下如果子单号碰巧重复，会误删别的仓位。
    return (
        str(position["strategySource"]),
        str(position["parentOrderId"]),
        str(position["clientOrderId"]),
        pd.Timestamp(position["clientOrderTime"]),
    )


def _derive_min_hold_signal_bars(params: BacktestParams) -> int:
    # Internalization uses raw MySQL signal rows as the bar sequence.
    return max(1, int(params.min_hold_bars))


def _limit_buffer_price() -> float:
    return A_SHARE_LIMIT_BUFFER_TICKS * A_SHARE_TICK_SIZE


def _get_limit_prices(limit_map: dict[str, dict[str, float]], security_code: str) -> tuple[float, float]:
    limit_row = limit_map.get(security_code, {})
    return float(limit_row.get("highLimit", np.nan)), float(limit_row.get("lowLimit", np.nan))


def _limit_status(row: pd.Series) -> str:
    # ask 为空代表涨停，bid 为空代表跌停；两边都有盘口时视为普通状态。
    if pd.isna(row.get("closeAsk1Aligned", np.nan)) and not pd.isna(row.get("closeBid1Aligned", np.nan)):
        return "UP_LIMIT"
    if pd.isna(row.get("closeBid1Aligned", np.nan)) and not pd.isna(row.get("closeAsk1Aligned", np.nan)):
        return "DOWN_LIMIT"
    return "NORMAL"


def _is_near_high_limit(row: pd.Series, high_limit: float) -> bool:
    if pd.isna(high_limit) or pd.isna(row.get("closeBid1Aligned", np.nan)):
        return False
    return float(row["closeBid1Aligned"]) >= high_limit - _limit_buffer_price()


def _is_near_low_limit(row: pd.Series, low_limit: float) -> bool:
    if pd.isna(low_limit) or pd.isna(row.get("closeAsk1Aligned", np.nan)):
        return False
    return float(row["closeAsk1Aligned"]) <= low_limit + _limit_buffer_price()


def _should_block_open(row: pd.Series, inventory_side: str, high_limit: float, low_limit: float) -> str | None:
    # 接近涨停时不再开空仓，接近跌停时不再开多仓，避免后续无法买回/卖出。
    if inventory_side == "SHORT" and _is_near_high_limit(row, high_limit):
        return "near_high_limit_no_short_open"
    if inventory_side == "LONG" and _is_near_low_limit(row, low_limit):
        return "near_low_limit_no_long_open"
    return None


def _stop_loss_mid_price(open_mid: float, inventory_side: str, threshold: float | None) -> float:
    if threshold is None:
        return np.nan
    if inventory_side == "LONG":
        return open_mid * (1.0 - threshold)
    return open_mid * (1.0 + threshold)


def _close_threshold_for_position(params: BacktestParams, hold_signal_count: int) -> float:
    if (
        params.relaxed_close_threshold is not None
        and params.relaxed_close_after_bars is not None
        and hold_signal_count > int(params.relaxed_close_after_bars)
    ):
        return float(params.relaxed_close_threshold)
    return float(params.close_threshold)


def _position_open_qty(positions: list[dict[str, object]]) -> int:
    return int(sum(int(pos["clientQty"]) for pos in positions))


def _position_remaining_qty(position: dict[str, object]) -> int:
    return int(position.get("remainingQty", position["clientQty"]))


def _positions_remaining_qty(positions: list[dict[str, object]]) -> int:
    return int(sum(_position_remaining_qty(pos) for pos in positions))


def _scaled_client_filled_amt(position: dict[str, object], executed_qty: int, original_qty: int) -> float:
    if original_qty <= 0:
        return 0.0
    return float(position["clientFilledAmt"]) * executed_qty / original_qty


def _merge_close_tick_volumes(sec_signal_df: pd.DataFrame, tick_df: pd.DataFrame) -> pd.DataFrame:
    if sec_signal_df.empty:
        return sec_signal_df

    result_df = sec_signal_df.copy()
    if tick_df.empty or "tickTime" not in tick_df.columns:
        result_df["closeBidVol1Tick"] = np.nan
        result_df["closeAskVol1Tick"] = np.nan
        return result_df

    left_df = result_df.reset_index().rename(columns={"index": "_signalRowIdx"}).sort_values("barTime")
    right_df = (
        tick_df[["tickTime", "bidVol1Tick", "askVol1Tick"]]
        .dropna(subset=["tickTime"])
        .sort_values("tickTime")
        .reset_index(drop=True)
    )
    if right_df.empty:
        result_df["closeBidVol1Tick"] = np.nan
        result_df["closeAskVol1Tick"] = np.nan
        return result_df

    merged_df = pd.merge_asof(
        left_df,
        right_df,
        left_on="barTime",
        right_on="tickTime",
        direction="backward",
    )
    merged_df = merged_df.sort_values("_signalRowIdx").drop(columns=["_signalRowIdx", "tickTime"])
    merged_df = merged_df.rename(
        columns={
            "bidVol1Tick": "closeBidVol1Tick",
            "askVol1Tick": "closeAskVol1Tick",
        }
    )
    return merged_df.reset_index(drop=True)


def _clip_position_by_cap(
    position: dict[str, object],
    open_positions_same_side: list[dict[str, object]],
    cap_column: str,
    cap_mode: str,
    cap_multiplier: float = 1.0,
) -> tuple[dict[str, object], dict[str, object] | None]:
    original_qty = int(position["clientQty"])
    raw_cap = position.get(cap_column, np.nan)
    base_cap_qty = int(np.floor(float(raw_cap))) if not pd.isna(raw_cap) else 0
    cap_qty = int(np.floor(base_cap_qty * cap_multiplier))
    current_qty = _positions_remaining_qty(open_positions_same_side)
    available_qty = max(0, cap_qty - current_qty)
    executed_qty = min(original_qty, available_qty)

    if executed_qty <= 0:
        reason = "position_cap_full" if cap_qty > 0 else "missing_liquidity_cap"
    elif executed_qty < original_qty and current_qty > 0:
        reason = "clipped_by_position_cap"
    elif executed_qty < original_qty:
        reason = "client_qty_clipped_by_cap"
    else:
        reason = "not_clipped"

    event = {
        **position,
        "clientQtyOriginal": original_qty,
        "clientQty": executed_qty,
        "clientFilledAmtOriginal": float(position["clientFilledAmt"]),
        "clientFilledAmt": _scaled_client_filled_amt(position, executed_qty, original_qty),
        "matched": executed_qty > 0,
        "matchStatus": "matched" if executed_qty > 0 else reason,
        "liquidityClipReason": reason,
        "positionCapMode": cap_mode,
        "positionCapMultiplier": float(cap_multiplier),
        "positionCapBaseQty": base_cap_qty,
        "positionCapQty": cap_qty,
        "positionCapAvailableQty": available_qty,
        "positionCapCurrentQtyBefore": current_qty,
    }
    if executed_qty <= 0:
        return event, None

    clipped_position = {
        **position,
        "clientQtyOriginal": original_qty,
        "clientQty": executed_qty,
        "clientFilledAmtOriginal": float(position["clientFilledAmt"]),
        "clientFilledAmt": _scaled_client_filled_amt(position, executed_qty, original_qty),
        "liquidityClipReason": reason,
        "positionCapMode": cap_mode,
        "positionCapMultiplier": float(cap_multiplier),
        "positionCapBaseQty": base_cap_qty,
        "positionCapQty": cap_qty,
        "positionCapAvailableQty": available_qty,
        "positionCapCurrentQtyBefore": current_qty,
        "remainingQty": executed_qty,
        "closeFillCount": 0,
        "closedQty": 0,
    }
    return event, clipped_position


def _close_positions(
    positions: list[dict[str, object]],
    row: pd.Series,
    pool_name: str,
    security_code: str,
    trade_date: pd.Timestamp,
    close_type: str,
    price_bucket_low: int,
    price_bucket_high: int,
    prev_day_vol: float | None,
) -> list[dict[str, object]]:
    # 涨跌停风控触发时会一次性平掉同方向所有可平仓位，这里统一生成交易记录。
    trade_rows: list[dict[str, object]] = []
    for pos in positions:
        close_qty = _position_remaining_qty(pos)
        if close_qty <= 0:
            continue
        closed_before = int(pos.get("closedQty", 0))
        closed_after = closed_before + close_qty
        close_position = {
            **pos,
            "clientQty": close_qty,
            "clientFilledAmt": _scaled_client_filled_amt(pos, close_qty, int(pos.get("clientQtyOriginal", pos["clientQty"]))),
            "closeFillSeq": int(pos.get("closeFillCount", 0)) + 1,
            "positionClosedQtyBefore": closed_before,
            "positionClosedQtyAfter": closed_after,
            "positionRemainingQtyAfter": 0,
            "isPositionFullyClosed": True,
        }
        trade_rows.append(
            _build_trade_record(
            pool_name=pool_name,
            security_code=security_code,
            trade_date=trade_date,
            position=close_position,
            close_row=row,
            hold_signal_count=int(row.name) - int(close_position["openRowIdx"]),
            close_type=close_type,
            price_bucket_low=price_bucket_low,
            price_bucket_high=price_bucket_high,
            prev_day_vol=prev_day_vol,
        )
        )
    return trade_rows


def _close_positions_with_volume_cap(
    positions: list[dict[str, object]],
    row: pd.Series,
    pool_name: str,
    security_code: str,
    trade_date: pd.Timestamp,
    close_type: str,
    price_bucket_low: int,
    price_bucket_high: int,
    prev_day_vol: float | None,
    close_volume: float,
    force_close_all: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    trade_rows: list[dict[str, object]] = []
    if not positions:
        return trade_rows, positions

    remaining_close_qty = float("inf") if force_close_all else int(np.floor(close_volume)) if not pd.isna(close_volume) else 0
    if remaining_close_qty <= 0:
        return trade_rows, positions

    still_open_positions: list[dict[str, object]] = []
    for pos in positions:
        pos_remaining_qty = _position_remaining_qty(pos)
        if pos_remaining_qty <= 0:
            continue
        if remaining_close_qty <= 0:
            still_open_positions.append(pos)
            continue

        fill_qty = pos_remaining_qty if force_close_all else min(pos_remaining_qty, int(remaining_close_qty))
        if fill_qty <= 0:
            still_open_positions.append(pos)
            continue

        closed_before = int(pos.get("closedQty", 0))
        closed_after = closed_before + int(fill_qty)
        remaining_after = pos_remaining_qty - int(fill_qty)
        fill_seq = int(pos.get("closeFillCount", 0)) + 1
        trade_position = {
            **pos,
            "clientQty": int(fill_qty),
            "clientFilledAmt": _scaled_client_filled_amt(pos, int(fill_qty), int(pos.get("clientQtyOriginal", pos["clientQty"]))),
            "closeFillSeq": fill_seq,
            "positionClosedQtyBefore": closed_before,
            "positionClosedQtyAfter": closed_after,
            "positionRemainingQtyAfter": remaining_after,
            "isPositionFullyClosed": remaining_after <= 0,
        }
        trade_rows.append(
            _build_trade_record(
                pool_name=pool_name,
                security_code=security_code,
                trade_date=trade_date,
                position=trade_position,
                close_row=row,
                hold_signal_count=int(row.name) - int(pos["openRowIdx"]),
                close_type=close_type if remaining_after <= 0 else f"{close_type}_PARTIAL",
                price_bucket_low=price_bucket_low,
                price_bucket_high=price_bucket_high,
                prev_day_vol=prev_day_vol,
            )
        )

        if not force_close_all:
            remaining_close_qty -= int(fill_qty)
        if remaining_after > 0:
            still_open_positions.append(
                {
                    **pos,
                    "remainingQty": remaining_after,
                    "closedQty": closed_after,
                    "closeFillCount": fill_seq,
                }
            )

    return trade_rows, still_open_positions


def _build_position_close_summary(partial_trade_df: pd.DataFrame, variant_tag: str) -> pd.DataFrame:
    if partial_trade_df.empty:
        return pd.DataFrame()

    group_cols = [
        "poolName",
        "tradeDate",
        "securityCode",
        "side",
        "strategySource",
        "parentOrderId",
        "clientOrderId",
        "clientOrderTime",
        "openTime",
        "openSignalTime",
    ]
    rows: list[dict[str, object]] = []
    for key, group_df in partial_trade_df.groupby(group_cols, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_values))
        qty = pd.to_numeric(group_df["clientQty"], errors="coerce").fillna(0)
        open_notional = pd.to_numeric(group_df["openNotional"], errors="coerce").fillna(0.0)
        exec_pnl = pd.to_numeric(group_df["execPnl"], errors="coerce").fillna(0.0)
        mid_pnl = pd.to_numeric(group_df["midPnl"], errors="coerce").fillna(0.0)
        close_price = np.where(
            group_df["side"].iloc[0] == "LONG",
            pd.to_numeric(group_df["closeBid1"], errors="coerce"),
            pd.to_numeric(group_df["closeAsk1"], errors="coerce"),
        )
        weighted_close_price = (
            float(np.nansum(close_price * qty) / qty.sum())
            if float(qty.sum()) > 0
            else np.nan
        )
        row.update(
            {
                "variantTag": variant_tag,
                "openMid": float(group_df["openMid"].iloc[0]),
                "openSignal": float(group_df["openSignal"].iloc[0]),
                "clientQtyOriginal": int(group_df["clientQtyOriginal"].iloc[0]),
                "filledCloseQty": int(qty.sum()),
                "closeFillCount": int(len(group_df)),
                "firstCloseTime": group_df["closeTime"].min(),
                "lastCloseTime": group_df["closeTime"].max(),
                "weightedClosePrice": weighted_close_price,
                "totalExecPnl": float(exec_pnl.sum()),
                "totalMidPnl": float(mid_pnl.sum()),
                "openNotional": float(open_notional.sum()),
                "notionalWeightedExecRet": (
                    float(exec_pnl.sum() / open_notional.sum())
                    if float(open_notional.sum()) != 0
                    else np.nan
                ),
                "isPositionFullyClosed": bool(group_df["isPositionFullyClosed"].fillna(False).iloc[-1]),
                "lastCloseType": group_df["closeType"].iloc[-1],
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _assign_price_bins(open_mid_price: pd.Series) -> tuple[pd.Series, pd.Series]:
    edges = [0.0, 10.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 1000000.0]
    intervals = pd.cut(open_mid_price, bins=edges, right=False, include_lowest=True)
    return (
        intervals.apply(lambda x: int(x.left) if pd.notna(x) else -1),
        intervals.apply(lambda x: int(x.right) if pd.notna(x) else -1),
    )


def _calc_prev_day_vol_from_price_df(price_df: pd.DataFrame) -> pd.DataFrame:
    # 这里不再依赖旧的 15s quote cache，而是直接用 DDB_Returns 对齐价格
    # 计算前一交易日逐信号 bar 的收益波动，供 meta 分桶时使用。
    if price_df.empty:
        return pd.DataFrame(columns=["securityCode", "prevDayVol"])

    ret_df = price_df.sort_values(["securityCode", "barTime"]).copy()
    ret_df["ret"] = (
        ret_df.groupby("securityCode")["closeMidAligned"].shift(-1).sub(ret_df["closeMidAligned"]).div(ret_df["closeMidAligned"])
    )
    return ret_df.groupby("securityCode", as_index=False)["ret"].std().rename(columns={"ret": "prevDayVol"})


def build_internalization_meta(
    signal_df: pd.DataFrame,
    pool_name: str,
    trade_date: str,
    current_price_df: pd.DataFrame,
    prev_price_df: pd.DataFrame,
) -> pd.DataFrame:
    # meta 只保留 internalization 汇总真正会用到的字段：
    # - 开盘附近价格分桶
    # - 前一日波动率
    if signal_df.empty:
        return pd.DataFrame(columns=["poolName", "tradeDate", "securityCode", "openMidPrice", "priceBucketLow", "priceBucketHigh", "prevDayVol"])

    if current_price_df.empty:
        open_price_df = signal_df.groupby("securityCode", as_index=False).agg(openMidPrice=("securityCode", lambda _: np.nan))
    else:
        open_price_df = (
            current_price_df.sort_values(["securityCode", "barTime"])
            .groupby("securityCode", as_index=False)
            .agg(openMidPrice=("closeMidAligned", "first"))
        )

    meta_df = signal_df[["securityCode"]].drop_duplicates().merge(open_price_df, on="securityCode", how="left")
    meta_df["poolName"] = pool_name
    meta_df["tradeDate"] = pd.Timestamp(trade_date)
    meta_df["priceBucketLow"], meta_df["priceBucketHigh"] = _assign_price_bins(meta_df["openMidPrice"])
    meta_df = meta_df.merge(_calc_prev_day_vol_from_price_df(prev_price_df), on="securityCode", how="left")
    return meta_df[["poolName", "tradeDate", "securityCode", "openMidPrice", "priceBucketLow", "priceBucketHigh", "prevDayVol"]]


def _load_latest_tick_mid_for_orders(
    matched_order_df: pd.DataFrame,
    tick_df: pd.DataFrame,
) -> pd.DataFrame:
    # 开仓价使用“客户单到来前最近一笔 tick 的 mid”，
    # 同时把过去 5 个 tick 的买一/卖一量均值一并带出来，供流动性过滤使用。
    if matched_order_df.empty:
        return matched_order_df.assign(
            openMid=np.nan,
            openTickTime=pd.NaT,
            avgBidVol1Last5=np.nan,
            avgAskVol1Last5=np.nan,
            liquidityCapQty=np.nan,
        )

    if tick_df.empty:
        return matched_order_df.assign(
            openMid=np.nan,
            openTickTime=pd.NaT,
            avgBidVol1Last5=np.nan,
            avgAskVol1Last5=np.nan,
            liquidityCapQty=np.nan,
        )

    left_df = matched_order_df.sort_values("clientOrderTime").reset_index(drop=True)
    right_df = tick_df.sort_values("tickTime").reset_index(drop=True)
    enriched_df = pd.merge_asof(
        left_df,
        right_df.rename(
            columns={
                "tickTime": "openTickTime",
                "midPriceTick": "openMid",
            }
        ),
        by="securityCode",
        left_on="clientOrderTime",
        right_on="openTickTime",
        direction="backward",
    )
    return enriched_df


def _filter_tick_to_trading_windows(
    tick_df: pd.DataFrame,
    windows: list[tuple[str, str]],
) -> pd.DataFrame:
    # 只保留连续竞价时段，避免集合竞价或盘后 tick 影响最近 5 tick 盘口量。
    if tick_df.empty or not windows:
        return tick_df

    tick_time = tick_df["tickTime"].dt.time
    mask = pd.Series(False, index=tick_df.index)
    for start_time, end_time in windows:
        start = pd.Timestamp(start_time).time()
        end = pd.Timestamp(end_time).time()
        mask |= (tick_time >= start) & (tick_time <= end)
    return tick_df.loc[mask].copy()


class TickMidCache:
    def __init__(
        self,
        trade_date: str,
        ddb_config: DdbConfig | None = None,
        continuous_auction_windows: list[tuple[str, str]] | None = None,
    ):
        self.trade_date = trade_date
        self.ddb_config = ddb_config or DdbConfig()
        self.continuous_auction_windows = continuous_auction_windows or CONTINUOUS_AUCTION_WINDOWS_A_SHARE
        self.session = connect_ddb(self.ddb_config)
        self.cache: dict[str, pd.DataFrame] = {}
        self.load_count = 0
        self.load_seconds = 0.0

    def get(self, security_code: str) -> pd.DataFrame:
        if security_code not in self.cache:
            # 按股票懒加载 tick，避免一次把整天整池 tick 全拉进内存。
            start = perf_counter()
            tick_df = fetch_tick_mid_ddb(self.session, self.trade_date, [security_code], chunk_size=1)
            self.load_seconds += perf_counter() - start
            self.load_count += 1
            if not tick_df.empty:
                tick_df = tick_df.sort_values("tickTime").reset_index(drop=True)
                tick_df = _filter_tick_to_trading_windows(tick_df, self.continuous_auction_windows)
                # 盘口量约束使用“最近 5 个 tick 的一档量均值”。
                tick_df["avgBidVol1Last5"] = tick_df["bidVol1Tick"].rolling(window=5, min_periods=1).mean()
                tick_df["avgAskVol1Last5"] = tick_df["askVol1Tick"].rolling(window=5, min_periods=1).mean()
                tick_df["liquidityCapQtyMin5"] = np.floor(
                    np.minimum(tick_df["avgBidVol1Last5"], tick_df["avgAskVol1Last5"])
                )
                tick_df["liquidityCapQtyAvg5"] = np.floor(
                    (tick_df["avgBidVol1Last5"] + tick_df["avgAskVol1Last5"]) / 2
                )
                tick_df["liquidityCapQty"] = tick_df["liquidityCapQtyMin5"]
                self.cache[security_code] = tick_df
            else:
                self.cache[security_code] = pd.DataFrame()
        return self.cache[security_code]

    def close(self) -> None:
        self.session.close()


def _build_trade_record(
    pool_name: str,
    security_code: str,
    trade_date: pd.Timestamp,
    position: dict[str, object],
    close_row: pd.Series,
    hold_signal_count: int,
    close_type: str,
    price_bucket_low: int,
    price_bucket_high: int,
    prev_day_vol: float | None,
) -> dict[str, object]:
    # 这里把单笔 internalization 仓位在开平仓两端的核心字段全部固化下来，
    # 方便后面既能做汇总，也能回看单笔 case。
    open_mid = float(position["openMid"])
    close_mid = float(close_row.closeMidAligned) if not pd.isna(close_row.closeMidAligned) else np.nan
    close_bid1 = float(close_row.closeBid1Aligned) if not pd.isna(close_row.closeBid1Aligned) else np.nan
    close_ask1 = float(close_row.closeAsk1Aligned) if not pd.isna(close_row.closeAsk1Aligned) else np.nan
    qty = int(position["clientQty"])
    side = str(position["inventorySide"])
    hold_minutes = (pd.Timestamp(close_row.barTime) - pd.Timestamp(position["openTime"])).total_seconds() / 60.0

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
        "clientQtyOriginal": int(position.get("clientQtyOriginal", qty)),
        "clientQty": qty,
        "clientOrderPrice": float(position["clientOrderPrice"]) if not pd.isna(position["clientOrderPrice"]) else np.nan,
        "clientExecPrice": float(position["clientExecPrice"]) if not pd.isna(position["clientExecPrice"]) else np.nan,
        "openMid": open_mid,
        "stopLossMidPrice": float(position.get("stopLossMidPrice", np.nan)),
        "avgBidVol1Last5": float(position["avgBidVol1Last5"]) if not pd.isna(position.get("avgBidVol1Last5", np.nan)) else np.nan,
        "avgAskVol1Last5": float(position["avgAskVol1Last5"]) if not pd.isna(position.get("avgAskVol1Last5", np.nan)) else np.nan,
        "liquidityCapQtyMin5": float(position["liquidityCapQtyMin5"]) if not pd.isna(position.get("liquidityCapQtyMin5", np.nan)) else np.nan,
        "liquidityCapQtyAvg5": float(position["liquidityCapQtyAvg5"]) if not pd.isna(position.get("liquidityCapQtyAvg5", np.nan)) else np.nan,
        "liquidityCapQty": float(position["liquidityCapQty"]) if not pd.isna(position.get("liquidityCapQty", np.nan)) else np.nan,
        "liquidityClipReason": position.get("liquidityClipReason", "not_clipped"),
        "positionCapMode": position.get("positionCapMode", "none"),
        "positionCapMultiplier": float(position.get("positionCapMultiplier", np.nan)),
        "positionCapBaseQty": float(position.get("positionCapBaseQty", np.nan)),
        "positionCapQty": float(position.get("positionCapQty", np.nan)),
        "positionCapAvailableQty": float(position.get("positionCapAvailableQty", np.nan)),
        "positionCapCurrentQtyBefore": float(position.get("positionCapCurrentQtyBefore", np.nan)),
        "closeFillSeq": int(position.get("closeFillSeq", 1)),
        "closeFilledQty": qty,
        "positionClosedQtyBefore": int(position.get("positionClosedQtyBefore", 0)),
        "positionClosedQtyAfter": int(position.get("positionClosedQtyAfter", qty)),
        "positionRemainingQtyAfter": int(position.get("positionRemainingQtyAfter", 0)),
        "isPositionFullyClosed": bool(position.get("isPositionFullyClosed", True)),
        "closeMid": close_mid,
        "closeBid1": close_bid1,
        "closeAsk1": close_ask1,
        "closeBidVol1Tick": float(close_row.get("closeBidVol1Tick", np.nan)) if not pd.isna(close_row.get("closeBidVol1Tick", np.nan)) else np.nan,
        "closeAskVol1Tick": float(close_row.get("closeAskVol1Tick", np.nan)) if not pd.isna(close_row.get("closeAskVol1Tick", np.nan)) else np.nan,
        "closePriceSource": "ddb_returns" if not pd.isna(close_row.closeBid1Aligned) or not pd.isna(close_row.closeAsk1Aligned) else "missing",
        "holdSignalCount": int(hold_signal_count),
        "holdBars": int(hold_signal_count),
        "holdMinutes": hold_minutes,
        "midRet": mid_ret,
        "execRet": exec_ret,
        "midPnl": mid_pnl,
        "execPnl": exec_pnl,
        "openNotional": open_mid * qty,
        "closeType": close_type,
        "closeLimitStatus": _limit_status(close_row),
        "priceBucketLow": int(price_bucket_low),
        "priceBucketHigh": int(price_bucket_high),
        "prevDayVol": prev_day_vol,
    }


def _simulate_position_cap_lifecycle(
    candidate_positions_by_open_idx: dict[int, list[dict[str, object]]],
    sec_signal_df: pd.DataFrame,
    pool_name: str,
    security_code: str,
    trade_date: pd.Timestamp,
    params: BacktestParams,
    high_limit: float,
    low_limit: float,
    price_bucket_low: int,
    price_bucket_high: int,
    prev_day_vol: float | None,
    cap_column: str,
    cap_mode: str,
    variant_column: str,
    cap_multiplier: float = 1.0,
    partial_close_by_tick_volume: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cap_event_rows: list[dict[str, object]] = []
    cap_trade_rows: list[dict[str, object]] = []
    if not candidate_positions_by_open_idx:
        return pd.DataFrame(), pd.DataFrame()

    min_hold_signal_bars = _derive_min_hold_signal_bars(params)
    long_open_positions: list[dict[str, object]] = []
    short_open_positions: list[dict[str, object]] = []
    position_signal_df = sec_signal_df.iloc[min(candidate_positions_by_open_idx) :]

    for idx, row in position_signal_df.iterrows():
        for candidate in candidate_positions_by_open_idx.get(idx, []):
            same_side_positions = long_open_positions if candidate["inventorySide"] == "LONG" else short_open_positions
            cap_event, clipped_position = _clip_position_by_cap(
                position=candidate,
                open_positions_same_side=same_side_positions,
                cap_column=cap_column,
                cap_mode=cap_mode,
                cap_multiplier=cap_multiplier,
            )
            cap_event[variant_column] = True
            cap_event_rows.append(cap_event)
            if clipped_position is None:
                continue
            clipped_position[variant_column] = True
            if clipped_position["inventorySide"] == "LONG":
                long_open_positions.append(clipped_position)
            else:
                short_open_positions.append(clipped_position)

        limit_status = _limit_status(row)
        can_close_long_by_limit = limit_status == "UP_LIMIT" or (
            limit_status != "DOWN_LIMIT" and _is_near_low_limit(row, low_limit)
        )
        can_close_short_by_limit = limit_status == "DOWN_LIMIT" or (
            limit_status != "UP_LIMIT" and _is_near_high_limit(row, high_limit)
        )

        if long_open_positions and can_close_long_by_limit:
            close_type = "LIMIT_UP_CLOSE_LONG" if limit_status == "UP_LIMIT" else "NEAR_LOW_LIMIT_CLOSE_LONG"
            eligible = list(long_open_positions)
            cap_trade_rows.extend(
                _close_positions(
                    positions=eligible,
                    row=row,
                    pool_name=pool_name,
                    security_code=security_code,
                    trade_date=trade_date,
                    close_type=close_type,
                    price_bucket_low=price_bucket_low,
                    price_bucket_high=price_bucket_high,
                    prev_day_vol=prev_day_vol,
                )
            )
            eligible_ids = {_position_identity(pos) for pos in eligible}
            long_open_positions = [pos for pos in long_open_positions if _position_identity(pos) not in eligible_ids]

        if short_open_positions and can_close_short_by_limit:
            close_type = "LIMIT_DOWN_CLOSE_SHORT" if limit_status == "DOWN_LIMIT" else "NEAR_HIGH_LIMIT_CLOSE_SHORT"
            eligible = list(short_open_positions)
            cap_trade_rows.extend(
                _close_positions(
                    positions=eligible,
                    row=row,
                    pool_name=pool_name,
                    security_code=security_code,
                    trade_date=trade_date,
                    close_type=close_type,
                    price_bucket_low=price_bucket_low,
                    price_bucket_high=price_bucket_high,
                    prev_day_vol=prev_day_vol,
                )
            )
            eligible_ids = {_position_identity(pos) for pos in eligible}
            short_open_positions = [pos for pos in short_open_positions if _position_identity(pos) not in eligible_ids]

        stop_loss_threshold = params.stop_loss_mid_ret_threshold
        stop_loss_signal_threshold = float(params.stop_loss_signal_threshold)
        if stop_loss_threshold is not None and row.merge_signal >= stop_loss_signal_threshold and short_open_positions:
            close_mid = float(row.closeMidAligned) if not pd.isna(row.closeMidAligned) else np.nan
            eligible = [
                pos
                for pos in short_open_positions
                if pos["openRowIdx"] <= idx - min_hold_signal_bars
                and not pd.isna(close_mid)
                and not pd.isna(row.closeAsk1Aligned)
                and close_mid >= float(pos["stopLossMidPrice"])
            ]
            if eligible:
                for pos in eligible:
                    cap_trade_rows.append(
                        _build_trade_record(
                            pool_name=pool_name,
                            security_code=security_code,
                            trade_date=trade_date,
                            position=pos,
                            close_row=row,
                            hold_signal_count=idx - int(pos["openRowIdx"]),
                            close_type="STOP_LOSS_HARD_TAIL",
                            price_bucket_low=price_bucket_low,
                            price_bucket_high=price_bucket_high,
                            prev_day_vol=prev_day_vol,
                        )
                    )
                eligible_ids = {_position_identity(pos) for pos in eligible}
                short_open_positions = [pos for pos in short_open_positions if _position_identity(pos) not in eligible_ids]

        if stop_loss_threshold is not None and row.merge_signal <= -stop_loss_signal_threshold and long_open_positions:
            close_mid = float(row.closeMidAligned) if not pd.isna(row.closeMidAligned) else np.nan
            eligible = [
                pos
                for pos in long_open_positions
                if pos["openRowIdx"] <= idx - min_hold_signal_bars
                and not pd.isna(close_mid)
                and not pd.isna(row.closeBid1Aligned)
                and close_mid <= float(pos["stopLossMidPrice"])
            ]
            if eligible:
                for pos in eligible:
                    cap_trade_rows.append(
                        _build_trade_record(
                            pool_name=pool_name,
                            security_code=security_code,
                            trade_date=trade_date,
                            position=pos,
                            close_row=row,
                            hold_signal_count=idx - int(pos["openRowIdx"]),
                            close_type="STOP_LOSS_HARD_TAIL",
                            price_bucket_low=price_bucket_low,
                            price_bucket_high=price_bucket_high,
                            prev_day_vol=prev_day_vol,
                        )
                    )
                eligible_ids = {_position_identity(pos) for pos in eligible}
                long_open_positions = [pos for pos in long_open_positions if _position_identity(pos) not in eligible_ids]

        if short_open_positions:
            eligible = [
                pos
                for pos in short_open_positions
                if pos["openRowIdx"] <= idx - min_hold_signal_bars
                and row.merge_signal >= _close_threshold_for_position(params, idx - int(pos["openRowIdx"]))
            ]
            if eligible:
                if partial_close_by_tick_volume:
                    partial_trades, updated_eligible = _close_positions_with_volume_cap(
                        positions=eligible,
                        row=row,
                        pool_name=pool_name,
                        security_code=security_code,
                        trade_date=trade_date,
                        close_type="SIGNAL",
                        price_bucket_low=price_bucket_low,
                        price_bucket_high=price_bucket_high,
                        prev_day_vol=prev_day_vol,
                        close_volume=row.get("closeAskVol1Tick", np.nan),
                    )
                    cap_trade_rows.extend(partial_trades)
                    eligible_ids = {_position_identity(pos) for pos in eligible}
                    remaining_by_id = {_position_identity(pos): pos for pos in updated_eligible}
                    short_open_positions = [
                        remaining_by_id.get(_position_identity(pos), pos)
                        for pos in short_open_positions
                        if _position_identity(pos) not in eligible_ids or _position_identity(pos) in remaining_by_id
                    ]
                else:
                    for pos in eligible:
                        cap_trade_rows.append(
                            _build_trade_record(
                                pool_name=pool_name,
                                security_code=security_code,
                                trade_date=trade_date,
                                position=pos,
                                close_row=row,
                                hold_signal_count=idx - int(pos["openRowIdx"]),
                                close_type="SIGNAL",
                                price_bucket_low=price_bucket_low,
                                price_bucket_high=price_bucket_high,
                                prev_day_vol=prev_day_vol,
                            )
                        )
                    eligible_ids = {_position_identity(pos) for pos in eligible}
                    short_open_positions = [pos for pos in short_open_positions if _position_identity(pos) not in eligible_ids]

        if long_open_positions:
            eligible = [
                pos
                for pos in long_open_positions
                if pos["openRowIdx"] <= idx - min_hold_signal_bars
                and row.merge_signal <= -_close_threshold_for_position(params, idx - int(pos["openRowIdx"]))
            ]
            if eligible:
                if partial_close_by_tick_volume:
                    partial_trades, updated_eligible = _close_positions_with_volume_cap(
                        positions=eligible,
                        row=row,
                        pool_name=pool_name,
                        security_code=security_code,
                        trade_date=trade_date,
                        close_type="SIGNAL",
                        price_bucket_low=price_bucket_low,
                        price_bucket_high=price_bucket_high,
                        prev_day_vol=prev_day_vol,
                        close_volume=row.get("closeBidVol1Tick", np.nan),
                    )
                    cap_trade_rows.extend(partial_trades)
                    eligible_ids = {_position_identity(pos) for pos in eligible}
                    remaining_by_id = {_position_identity(pos): pos for pos in updated_eligible}
                    long_open_positions = [
                        remaining_by_id.get(_position_identity(pos), pos)
                        for pos in long_open_positions
                        if _position_identity(pos) not in eligible_ids or _position_identity(pos) in remaining_by_id
                    ]
                else:
                    for pos in eligible:
                        cap_trade_rows.append(
                            _build_trade_record(
                                pool_name=pool_name,
                                security_code=security_code,
                                trade_date=trade_date,
                                position=pos,
                                close_row=row,
                                hold_signal_count=idx - int(pos["openRowIdx"]),
                                close_type="SIGNAL",
                                price_bucket_low=price_bucket_low,
                                price_bucket_high=price_bucket_high,
                                prev_day_vol=prev_day_vol,
                            )
                        )
                    eligible_ids = {_position_identity(pos) for pos in eligible}
                    long_open_positions = [pos for pos in long_open_positions if _position_identity(pos) not in eligible_ids]

    if not sec_signal_df.empty:
        last_row = sec_signal_df.iloc[-1]
        last_idx = len(sec_signal_df) - 1
        for pos in long_open_positions:
            cap_trade_rows.append(
                _build_trade_record(
                    pool_name=pool_name,
                    security_code=security_code,
                    trade_date=trade_date,
                    position=pos,
                    close_row=last_row,
                    hold_signal_count=last_idx - int(pos["openRowIdx"]),
                    close_type="EOD",
                    price_bucket_low=price_bucket_low,
                    price_bucket_high=price_bucket_high,
                    prev_day_vol=prev_day_vol,
                )
            )
        for pos in short_open_positions:
            cap_trade_rows.append(
                _build_trade_record(
                    pool_name=pool_name,
                    security_code=security_code,
                    trade_date=trade_date,
                    position=pos,
                    close_row=last_row,
                    hold_signal_count=last_idx - int(pos["openRowIdx"]),
                    close_type="EOD",
                    price_bucket_low=price_bucket_low,
                    price_bucket_high=price_bucket_high,
                    prev_day_vol=prev_day_vol,
                )
            )

    for trade_row in cap_trade_rows:
        trade_row[variant_column] = True
    return pd.DataFrame(cap_event_rows), pd.DataFrame(cap_trade_rows)


def simulate_internalization_day(
    signal_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    client_order_df: pd.DataFrame,
    pool_name: str,
    params: BacktestParams,
    tick_mid_cache: TickMidCache,
    close_price_map: dict[str, pd.DataFrame] | None = None,
    limit_map: dict[str, dict[str, float]] | None = None,
    match_window_seconds: int | None = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 单票内部的主流程：
    # 1. 客户子单在 signal 窗口内匹配开仓信号
    # 2. 用 tick mid 生成开仓价
    # 3. 按 signal bar 序列管理持仓并在满足平仓阈值时出场
    if signal_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    trade_date = pd.Timestamp(signal_df["tradeDate"].iloc[0])
    close_price_map = close_price_map or {}
    limit_map = limit_map or {}
    meta_map = meta_df.set_index("securityCode").to_dict("index")
    event_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    position_cap_summary_frames: list[pd.DataFrame] = []
    position_cap_trade_frames: list[pd.DataFrame] = []
    position_cap_position_frames: list[pd.DataFrame] = []

    # 先按股票拆分 signal 和客户子单。只有两边都有数据的股票才进入逐票模拟，避免无意义拉 tick。
    signal_by_security = {
        security_code: sec_df.sort_values("barTime").reset_index(drop=True)
        for security_code, sec_df in signal_df.groupby("securityCode", sort=True)
    }
    orders_by_security = {
        security_code: sec_df.sort_values("clientOrderTime").reset_index(drop=True)
        for security_code, sec_df in client_order_df.groupby("securityCode", sort=True)
    }

    security_codes = sorted(set(signal_by_security) & set(orders_by_security))

    for security_code in security_codes:
        # 每只股票独立模拟：signal/订单/Returns 价格/tick 价格都只取当前股票，控制内存占用。
        sec_signal_df = signal_by_security.get(security_code, pd.DataFrame())
        sec_orders_df = orders_by_security.get(security_code, pd.DataFrame())
        sec_close_df = close_price_map.get(security_code, pd.DataFrame())
        high_limit, low_limit = _get_limit_prices(limit_map, security_code)
        meta = meta_map.get(
            security_code,
            {
                "priceBucketLow": np.nan,
                "priceBucketHigh": np.nan,
                "prevDayVol": np.nan,
            },
        )

        if sec_signal_df.empty:
            for order in sec_orders_df.to_dict(orient="records"):
                event_rows.append(
                    {
                        **order,
                        "poolName": pool_name,
                        "matchStatus": "no_signal_data",
                        "matched": False,
                        "matchSignalTime": pd.NaT,
                        "matchSignalTimeInt": np.nan,
                        "matchSignal": np.nan,
                        "matchMidPrice": np.nan,
                        "matchDelaySeconds": np.nan,
                    }
                )
            continue

        if not sec_close_df.empty:
            # 平仓时直接按 signal barTime 去对齐 Returns 盘口。
            sec_signal_df = sec_signal_df.merge(
                sec_close_df,
                on=["securityCode", "barTime"],
                how="left",
            )
        else:
            sec_signal_df = sec_signal_df.copy()
            sec_signal_df["closeMidAligned"] = np.nan
            sec_signal_df["closeAsk1Aligned"] = np.nan
            sec_signal_df["closeBid1Aligned"] = np.nan

        signal_times = sec_signal_df["barTime"].to_numpy(dtype="datetime64[ns]")
        min_hold_signal_bars = _derive_min_hold_signal_bars(params)
        pending_match_rows: list[dict[str, object]] = []
        positions_by_open_idx: dict[int, list[dict[str, object]]] = {}
        sec_tick_df = pd.DataFrame()

        # 第一段：逐笔客户子单寻找开仓 signal。信号必须先到，客户单必须在窗口内跟上。
        for order in sec_orders_df.to_dict(orient="records"):
            order_time = pd.Timestamp(order["clientOrderTime"])
            window_start = (
                pd.Timestamp.min
                if match_window_seconds is None
                else order_time - pd.Timedelta(seconds=match_window_seconds)
            )
            # 信号先来、客户单后到：
            # 在客户单之前最近的一条 signal 上找开仓机会，并要求落在匹配窗口内。
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
                event_rows.append(
                    {
                        **order,
                        "poolName": pool_name,
                        "matchStatus": "no_qualifying_signal",
                        "matched": False,
                        "matchSignalTime": pd.NaT,
                        "matchSignalTimeInt": np.nan,
                        "matchSignal": np.nan,
                        "matchMidPrice": np.nan,
                        "matchDelaySeconds": np.nan,
                    }
                )
                continue

            inventory_side = "SHORT" if order["clientSide"] == "B" else "LONG"
            limit_block_status = _should_block_open(matched_row, inventory_side, high_limit, low_limit)
            if limit_block_status is not None:
                event_rows.append(
                    {
                        **order,
                        "poolName": pool_name,
                        "matchStatus": limit_block_status,
                        "matched": False,
                        "matchSignalTime": matched_row.barTime,
                        "matchSignalTimeInt": int(matched_row.signalTime),
                        "matchSignal": float(matched_row.merge_signal),
                        "matchMidPrice": float(matched_row.closeMidAligned) if not pd.isna(matched_row.closeMidAligned) else np.nan,
                        "matchDelaySeconds": (order_time - matched_row.barTime).total_seconds(),
                        "limitStatus": _limit_status(matched_row),
                    }
                )
                continue

            match_delay = (order_time - matched_row.barTime).total_seconds()
            pending_match_rows.append(
                {
                    **order,
                    "poolName": pool_name,
                    "matchStatus": "matched",
                    "matched": True,
                    "matchSignalTime": matched_row.barTime,
                    "matchSignalTimeInt": int(matched_row.signalTime),
                    "matchSignal": float(matched_row.merge_signal),
                    "matchMidPrice": float(matched_row.closeMidAligned) if not pd.isna(matched_row.closeMidAligned) else np.nan,
                    "matchDelaySeconds": match_delay,
                    "matchedSignalRowIdx": matched_idx,
                    "inventorySide": inventory_side,
                    "limitStatus": _limit_status(matched_row),
                }
            )

        # 第二段：给已经通过 signal 检查的子单补开仓 tick mid；如果最近 tick mid 为空，则不能开仓。
        if pending_match_rows:
            sec_tick_df = tick_mid_cache.get(security_code)
            matched_order_df = _load_latest_tick_mid_for_orders(
                matched_order_df=pd.DataFrame(pending_match_rows),
                tick_df=sec_tick_df,
            )

            for row in matched_order_df.to_dict(orient="records"):
                if pd.isna(row.get("openMid")):
                    event_rows.append(
                        {
                            **row,
                            "matchStatus": "no_tick_mid",
                            "matched": False,
                        }
                    )
                    continue

                position = {
                    **row,
                    "clientQtyOriginal": int(row["clientQty"]),
                    "clientFilledAmtOriginal": float(row["clientFilledAmt"]),
                    "liquidityClipReason": "not_clipped",
                    "positionCapMode": "none",
                    "openRowIdx": int(row["matchedSignalRowIdx"]),
                    "openTime": row["clientOrderTime"],
                    "openTickTime": row["openTickTime"],
                    "openSignalTime": int(row["matchSignalTimeInt"]),
                    "openSignal": float(row["matchSignal"]),
                    "openMid": float(row["openMid"]),
                }
                position["stopLossMidPrice"] = _stop_loss_mid_price(
                    open_mid=float(position["openMid"]),
                    inventory_side=str(position["inventorySide"]),
                    threshold=params.stop_loss_mid_ret_threshold,
                )
                positions_by_open_idx.setdefault(int(row["matchedSignalRowIdx"]), []).append(position)
                event_rows.append(
                    {
                        **row,
                        "clientQtyOriginal": int(row["clientQty"]),
                        "clientFilledAmtOriginal": float(row["clientFilledAmt"]),
                        "liquidityClipReason": "not_clipped",
                        "positionCapMode": "none",
                        "matched": True,
                    }
                )

        if positions_by_open_idx:
            sec_signal_df = _merge_close_tick_volumes(sec_signal_df, sec_tick_df)

        matched_order_df = pd.DataFrame(
            [row for row in event_rows if row["securityCode"] == security_code and bool(row["matched"])]
        )

        long_open_positions: list[dict[str, object]] = []
        short_open_positions: list[dict[str, object]] = []
        max_concurrent_long = 0
        max_concurrent_short = 0
        sec_trade_rows: list[dict[str, object]] = []
        price_bucket_low = int(meta["priceBucketLow"]) if not pd.isna(meta["priceBucketLow"]) else -1
        price_bucket_high = int(meta["priceBucketHigh"]) if not pd.isna(meta["priceBucketHigh"]) else -1
        prev_day_vol = float(meta["prevDayVol"]) if not pd.isna(meta["prevDayVol"]) else np.nan

        # 第三段：按 signal bar 时间推进持仓。先加入本 bar 新开的仓，再处理涨跌停风控和平仓信号。
        # 没有任何可开仓子单时，不需要扫完整 signal 序列；有开仓时从第一笔开仓候选所在 bar 开始扫。
        position_signal_df = (
            sec_signal_df.iloc[min(positions_by_open_idx) :]
            if positions_by_open_idx
            else sec_signal_df.iloc[0:0]
        )
        for idx, row in position_signal_df.iterrows():
            for pos in positions_by_open_idx.get(idx, []):
                if pos["inventorySide"] == "LONG":
                    long_open_positions.append(pos)
                else:
                    short_open_positions.append(pos)

            limit_status = _limit_status(row)
            can_close_long_by_limit = limit_status == "UP_LIMIT" or (
                limit_status != "DOWN_LIMIT" and _is_near_low_limit(row, low_limit)
            )
            can_close_short_by_limit = limit_status == "DOWN_LIMIT" or (
                limit_status != "UP_LIMIT" and _is_near_high_limit(row, high_limit)
            )
            if long_open_positions and can_close_long_by_limit:
                close_type = "LIMIT_UP_CLOSE_LONG" if limit_status == "UP_LIMIT" else "NEAR_LOW_LIMIT_CLOSE_LONG"
                eligible = list(long_open_positions)
                sec_trade_rows.extend(
                    _close_positions(
                        positions=eligible,
                        row=row,
                        pool_name=pool_name,
                        security_code=security_code,
                        trade_date=trade_date,
                        close_type=close_type,
                        price_bucket_low=price_bucket_low,
                        price_bucket_high=price_bucket_high,
                        prev_day_vol=prev_day_vol,
                    )
                )
                eligible_ids = {_position_identity(pos) for pos in eligible}
                long_open_positions = [pos for pos in long_open_positions if _position_identity(pos) not in eligible_ids]

            if short_open_positions and can_close_short_by_limit:
                close_type = "LIMIT_DOWN_CLOSE_SHORT" if limit_status == "DOWN_LIMIT" else "NEAR_HIGH_LIMIT_CLOSE_SHORT"
                eligible = list(short_open_positions)
                sec_trade_rows.extend(
                    _close_positions(
                        positions=eligible,
                        row=row,
                        pool_name=pool_name,
                        security_code=security_code,
                        trade_date=trade_date,
                        close_type=close_type,
                        price_bucket_low=price_bucket_low,
                        price_bucket_high=price_bucket_high,
                        prev_day_vol=prev_day_vol,
                    )
                )
                eligible_ids = {_position_identity(pos) for pos in eligible}
                short_open_positions = [pos for pos in short_open_positions if _position_identity(pos) not in eligible_ids]

            stop_loss_threshold = params.stop_loss_mid_ret_threshold
            stop_loss_signal_threshold = float(params.stop_loss_signal_threshold)
            if stop_loss_threshold is not None and row.merge_signal >= stop_loss_signal_threshold and short_open_positions:
                close_mid = float(row.closeMidAligned) if not pd.isna(row.closeMidAligned) else np.nan
                eligible = [
                    pos
                    for pos in short_open_positions
                    if pos["openRowIdx"] <= idx - min_hold_signal_bars
                    and not pd.isna(close_mid)
                    and not pd.isna(row.closeAsk1Aligned)
                    and close_mid >= float(pos["stopLossMidPrice"])
                ]
                if eligible:
                    for pos in eligible:
                        sec_trade_rows.append(
                            _build_trade_record(
                                pool_name=pool_name,
                                security_code=security_code,
                                trade_date=trade_date,
                                position=pos,
                                close_row=row,
                                hold_signal_count=idx - int(pos["openRowIdx"]),
                                close_type="STOP_LOSS_HARD_TAIL",
                                price_bucket_low=price_bucket_low,
                                price_bucket_high=price_bucket_high,
                                prev_day_vol=prev_day_vol,
                            )
                        )
                    eligible_ids = {_position_identity(pos) for pos in eligible}
                    short_open_positions = [pos for pos in short_open_positions if _position_identity(pos) not in eligible_ids]

            if stop_loss_threshold is not None and row.merge_signal <= -stop_loss_signal_threshold and long_open_positions:
                close_mid = float(row.closeMidAligned) if not pd.isna(row.closeMidAligned) else np.nan
                eligible = [
                    pos
                    for pos in long_open_positions
                    if pos["openRowIdx"] <= idx - min_hold_signal_bars
                    and not pd.isna(close_mid)
                    and not pd.isna(row.closeBid1Aligned)
                    and close_mid <= float(pos["stopLossMidPrice"])
                ]
                if eligible:
                    for pos in eligible:
                        sec_trade_rows.append(
                            _build_trade_record(
                                pool_name=pool_name,
                                security_code=security_code,
                                trade_date=trade_date,
                                position=pos,
                                close_row=row,
                                hold_signal_count=idx - int(pos["openRowIdx"]),
                                close_type="STOP_LOSS_HARD_TAIL",
                                price_bucket_low=price_bucket_low,
                                price_bucket_high=price_bucket_high,
                                prev_day_vol=prev_day_vol,
                            )
                        )
                    eligible_ids = {_position_identity(pos) for pos in eligible}
                    long_open_positions = [pos for pos in long_open_positions if _position_identity(pos) not in eligible_ids]

            if short_open_positions:
                # SHORT 仓位需要等正向平仓信号。
                eligible = [
                    pos
                    for pos in short_open_positions
                    if pos["openRowIdx"] <= idx - min_hold_signal_bars
                    and row.merge_signal >= _close_threshold_for_position(params, idx - int(pos["openRowIdx"]))
                ]
                if eligible:
                    for pos in eligible:
                        sec_trade_rows.append(
                            _build_trade_record(
                                pool_name=pool_name,
                                security_code=security_code,
                                trade_date=trade_date,
                                position=pos,
                                close_row=row,
                                hold_signal_count=idx - int(pos["openRowIdx"]),
                                close_type="SIGNAL",
                                price_bucket_low=price_bucket_low,
                                price_bucket_high=price_bucket_high,
                                prev_day_vol=prev_day_vol,
                            )
                        )
                    eligible_ids = {_position_identity(pos) for pos in eligible}
                    short_open_positions = [pos for pos in short_open_positions if _position_identity(pos) not in eligible_ids]

            if long_open_positions:
                # LONG 仓位需要等负向平仓信号。
                eligible = [
                    pos
                    for pos in long_open_positions
                    if pos["openRowIdx"] <= idx - min_hold_signal_bars
                    and row.merge_signal <= -_close_threshold_for_position(params, idx - int(pos["openRowIdx"]))
                ]
                if eligible:
                    for pos in eligible:
                        sec_trade_rows.append(
                            _build_trade_record(
                                pool_name=pool_name,
                                security_code=security_code,
                                trade_date=trade_date,
                                position=pos,
                                close_row=row,
                                hold_signal_count=idx - int(pos["openRowIdx"]),
                                close_type="SIGNAL",
                                price_bucket_low=price_bucket_low,
                                price_bucket_high=price_bucket_high,
                                prev_day_vol=prev_day_vol,
                            )
                        )
                    eligible_ids = {_position_identity(pos) for pos in eligible}
                    long_open_positions = [pos for pos in long_open_positions if _position_identity(pos) not in eligible_ids]

            max_concurrent_long = max(max_concurrent_long, len(long_open_positions))
            max_concurrent_short = max(max_concurrent_short, len(short_open_positions))

        if not sec_signal_df.empty:
            # 第四段：收盘仍未平掉的仓位，按最后一条 signal 对齐盘口做 EOD 平仓。
            # 还有没平掉的仓位就按当日最后一条 signal 对应的盘口强制 EOD 平仓。
            last_row = sec_signal_df.iloc[-1]
            last_idx = len(sec_signal_df) - 1
            for pos in long_open_positions:
                sec_trade_rows.append(
                    _build_trade_record(
                        pool_name=pool_name,
                        security_code=security_code,
                        trade_date=trade_date,
                        position=pos,
                        close_row=last_row,
                        hold_signal_count=last_idx - int(pos["openRowIdx"]),
                        close_type="EOD",
                        price_bucket_low=price_bucket_low,
                        price_bucket_high=price_bucket_high,
                        prev_day_vol=prev_day_vol,
                    )
                )

            for pos in short_open_positions:
                sec_trade_rows.append(
                    _build_trade_record(
                        pool_name=pool_name,
                        security_code=security_code,
                        trade_date=trade_date,
                        position=pos,
                        close_row=last_row,
                        hold_signal_count=last_idx - int(pos["openRowIdx"]),
                        close_type="EOD",
                        price_bucket_low=price_bucket_low,
                        price_bucket_high=price_bucket_high,
                        prev_day_vol=prev_day_vol,
                    )
                )

        # 第五段：把逐笔交易和客户订单事件汇总成单票级指标，后续 variant 只替换分子口径。
        trade_rows.extend(sec_trade_rows)
        sec_trade_df = pd.DataFrame(sec_trade_rows)
        matched_buy_order_df = matched_order_df[matched_order_df["clientSide"] == "B"] if not matched_order_df.empty else pd.DataFrame()
        matched_sell_order_df = matched_order_df[matched_order_df["clientSide"] == "S"] if not matched_order_df.empty else pd.DataFrame()
        sec_trade_long_df = sec_trade_df[sec_trade_df["side"] == "LONG"] if not sec_trade_df.empty else pd.DataFrame()
        sec_trade_short_df = sec_trade_df[sec_trade_df["side"] == "SHORT"] if not sec_trade_df.empty else pd.DataFrame()
        security_summary = {
            "poolName": pool_name,
            "tradeDate": trade_date,
            "securityCode": security_code,
            "totalBarCount": int(len(sec_signal_df)),
            "longSignalCount": int((sec_signal_df["merge_signal"] > params.open_threshold).sum()),
            "shortSignalCount": int((sec_signal_df["merge_signal"] < -params.open_threshold).sum()),
            "clientChildCount": int(len(sec_orders_df)),
            "matchedClientChildCount": int(len(matched_order_df)),
            "unmatchedClientChildCount": int(len(sec_orders_df) - len(matched_order_df)),
            "clientBuyChildCount": int((sec_orders_df["clientSide"] == "B").sum()) if not sec_orders_df.empty else 0,
            "clientSellChildCount": int((sec_orders_df["clientSide"] == "S").sum()) if not sec_orders_df.empty else 0,
            "matchedClientBuyChildCount": int(len(matched_buy_order_df)),
            "matchedClientSellChildCount": int(len(matched_sell_order_df)),
            "clientBuyQty": int(sec_orders_df.loc[sec_orders_df["clientSide"] == "B", "clientQty"].sum()) if not sec_orders_df.empty else 0,
            "clientSellQty": int(sec_orders_df.loc[sec_orders_df["clientSide"] == "S", "clientQty"].sum()) if not sec_orders_df.empty else 0,
            "clientAmt": float(sec_orders_df["clientFilledAmt"].sum()) if not sec_orders_df.empty else 0.0,
            "clientBuyAmt": float(sec_orders_df.loc[sec_orders_df["clientSide"] == "B", "clientFilledAmt"].sum()) if not sec_orders_df.empty else 0.0,
            "clientSellAmt": float(sec_orders_df.loc[sec_orders_df["clientSide"] == "S", "clientFilledAmt"].sum()) if not sec_orders_df.empty else 0.0,
            "matchedClientQty": int(matched_order_df["clientQty"].sum()) if not matched_order_df.empty else 0,
            "matchedClientBuyQty": int(matched_buy_order_df["clientQty"].sum()) if not matched_buy_order_df.empty else 0,
            "matchedClientSellQty": int(matched_sell_order_df["clientQty"].sum()) if not matched_sell_order_df.empty else 0,
            "matchedClientAmt": float(matched_order_df["clientFilledAmt"].sum()) if not matched_order_df.empty else 0.0,
            "matchedClientBuyAmt": float(matched_buy_order_df["clientFilledAmt"].sum()) if not matched_buy_order_df.empty else 0.0,
            "matchedClientSellAmt": float(matched_sell_order_df["clientFilledAmt"].sum()) if not matched_sell_order_df.empty else 0.0,
            "longOpenCount": int(len(matched_sell_order_df)),
            "shortOpenCount": int(len(matched_buy_order_df)),
            "maxConcurrentLongCount": int(max_concurrent_long),
            "maxConcurrentShortCount": int(max_concurrent_short),
            "maxConcurrentLongQty": _max_concurrent_qty_from_trades(sec_trade_df, "LONG") if not sec_trade_df.empty else 0,
            "maxConcurrentShortQty": _max_concurrent_qty_from_trades(sec_trade_df, "SHORT") if not sec_trade_df.empty else 0,
        }

        if sec_trade_df.empty:
            security_summary.update(
                {
                    "closedLongCount": 0,
                    "closedShortCount": 0,
                    "avgLongHoldMinutes": np.nan,
                    "avgShortHoldMinutes": np.nan,
                    "avgLongMidRet": np.nan,
                    "avgShortMidRet": np.nan,
                    "avgLongExecRet": np.nan,
                    "avgShortExecRet": np.nan,
                    "totalLongMidRet": 0.0,
                    "totalShortMidRet": 0.0,
                    "totalLongExecRet": 0.0,
                    "totalShortExecRet": 0.0,
                    "totalLongMidPnl": 0.0,
                    "totalShortMidPnl": 0.0,
                    "totalLongExecPnl": 0.0,
                    "totalShortExecPnl": 0.0,
                    "longMidWinRate": np.nan,
                    "shortMidWinRate": np.nan,
                    "longExecWinRate": np.nan,
                    "shortExecWinRate": np.nan,
                    "longEodCloseCount": 0,
                    "shortEodCloseCount": 0,
                    "totalMatchedNotional": 0.0,
                    "notionalWeightedMidRet": np.nan,
                    "notionalWeightedExecRet": np.nan,
                }
            )
        else:
            security_summary.update(
                {
                    "closedLongCount": int(len(sec_trade_long_df)),
                    "closedShortCount": int(len(sec_trade_short_df)),
                    "avgLongHoldMinutes": float(sec_trade_long_df["holdMinutes"].mean()) if not sec_trade_long_df.empty else np.nan,
                    "avgShortHoldMinutes": float(sec_trade_short_df["holdMinutes"].mean()) if not sec_trade_short_df.empty else np.nan,
                    "avgLongMidRet": float(sec_trade_long_df["midRet"].mean()) if not sec_trade_long_df.empty else np.nan,
                    "avgShortMidRet": float(sec_trade_short_df["midRet"].mean()) if not sec_trade_short_df.empty else np.nan,
                    "avgLongExecRet": float(sec_trade_long_df["execRet"].mean()) if not sec_trade_long_df.empty else np.nan,
                    "avgShortExecRet": float(sec_trade_short_df["execRet"].mean()) if not sec_trade_short_df.empty else np.nan,
                    "totalLongMidRet": float(sec_trade_long_df["midRet"].sum()) if not sec_trade_long_df.empty else 0.0,
                    "totalShortMidRet": float(sec_trade_short_df["midRet"].sum()) if not sec_trade_short_df.empty else 0.0,
                    "totalLongExecRet": float(sec_trade_long_df["execRet"].sum()) if not sec_trade_long_df.empty else 0.0,
                    "totalShortExecRet": float(sec_trade_short_df["execRet"].sum()) if not sec_trade_short_df.empty else 0.0,
                    "totalLongMidPnl": float(sec_trade_long_df["midPnl"].sum()) if not sec_trade_long_df.empty else 0.0,
                    "totalShortMidPnl": float(sec_trade_short_df["midPnl"].sum()) if not sec_trade_short_df.empty else 0.0,
                    "totalLongExecPnl": float(sec_trade_long_df["execPnl"].sum()) if not sec_trade_long_df.empty else 0.0,
                    "totalShortExecPnl": float(sec_trade_short_df["execPnl"].sum()) if not sec_trade_short_df.empty else 0.0,
                    "longMidWinRate": float((sec_trade_long_df["midRet"] > 0).mean()) if not sec_trade_long_df.empty else np.nan,
                    "shortMidWinRate": float((sec_trade_short_df["midRet"] > 0).mean()) if not sec_trade_short_df.empty else np.nan,
                    "longExecWinRate": float((sec_trade_long_df["execRet"] > 0).mean()) if not sec_trade_long_df.empty else np.nan,
                    "shortExecWinRate": float((sec_trade_short_df["execRet"] > 0).mean()) if not sec_trade_short_df.empty else np.nan,
                    "longEodCloseCount": int((sec_trade_long_df["closeType"] == "EOD").sum()) if not sec_trade_long_df.empty else 0,
                    "shortEodCloseCount": int((sec_trade_short_df["closeType"] == "EOD").sum()) if not sec_trade_short_df.empty else 0,
                    "totalMatchedNotional": float(sec_trade_df["openNotional"].sum()),
                    "notionalWeightedMidRet": (
                        float(sec_trade_df["midPnl"].sum() / sec_trade_df["openNotional"].sum())
                        if float(sec_trade_df["openNotional"].sum()) != 0
                        else np.nan
                    ),
                    "notionalWeightedExecRet": (
                        float(sec_trade_df["execPnl"].sum() / sec_trade_df["openNotional"].sum())
                        if float(sec_trade_df["openNotional"].sum()) != 0
                        else np.nan
                    ),
                }
            )

        summary_rows.append(security_summary)

        if positions_by_open_idx:
            base_sec_summary_df = pd.DataFrame([security_summary])
            for variant_tag, cap_column, cap_mode, variant_column, cap_multiplier, partial_close_by_tick_volume in [
                ("poscap_min5", "liquidityCapQtyMin5", "min5", "variantPoscapMin5", 1.0, False),
                ("poscap_avg5", "liquidityCapQtyAvg5", "avg5", "variantPoscapAvg5", 1.0, False),
                ("poscap_avg5x5_partial", "liquidityCapQtyAvg5", "avg5x5_partial", "variantPoscapAvg5x5Partial", 5.0, True),
            ]:
                cap_events_df, cap_trades_df = _simulate_position_cap_lifecycle(
                    candidate_positions_by_open_idx=positions_by_open_idx,
                    sec_signal_df=sec_signal_df,
                    pool_name=pool_name,
                    security_code=security_code,
                    trade_date=trade_date,
                    params=params,
                    high_limit=high_limit,
                    low_limit=low_limit,
                    price_bucket_low=price_bucket_low,
                    price_bucket_high=price_bucket_high,
                    prev_day_vol=prev_day_vol,
                    cap_column=cap_column,
                    cap_mode=cap_mode,
                    variant_column=variant_column,
                    cap_multiplier=cap_multiplier,
                    partial_close_by_tick_volume=partial_close_by_tick_volume,
                )
                if not cap_trades_df.empty:
                    cap_trades_df = cap_trades_df.copy()
                    cap_trades_df["variantTag"] = variant_tag
                    position_cap_trade_frames.append(cap_trades_df)
                    if partial_close_by_tick_volume:
                        position_summary_df = _build_position_close_summary(cap_trades_df, variant_tag=variant_tag)
                        if not position_summary_df.empty:
                            position_cap_position_frames.append(position_summary_df)
                cap_summary_df = build_variant_security_summary(
                    base_security_summary_df=base_sec_summary_df,
                    order_events_df=cap_events_df,
                    trades_df=cap_trades_df,
                    variant_tag=variant_tag,
                )
                if not cap_summary_df.empty:
                    cap_summary_df = cap_summary_df.copy()
                    cap_summary_df["variantTag"] = variant_tag
                    position_cap_summary_frames.append(cap_summary_df)

    order_events_df = pd.DataFrame(event_rows).sort_values(
        ["securityCode", "clientOrderTime", "strategySource", "parentOrderId", "clientOrderId"]
    ).reset_index(drop=True) if event_rows else pd.DataFrame()
    trades_df = pd.DataFrame(trade_rows).sort_values(
        ["securityCode", "openTime", "clientOrderId"]
    ).reset_index(drop=True) if trade_rows else pd.DataFrame()
    security_summary_df = pd.DataFrame(summary_rows).sort_values(["securityCode"]).reset_index(drop=True) if summary_rows else pd.DataFrame()
    security_summary_df.attrs["position_cap_summaries"] = (
        pd.concat(position_cap_summary_frames, ignore_index=True)
        if position_cap_summary_frames
        else pd.DataFrame()
    )
    security_summary_df.attrs["position_cap_trades"] = (
        pd.concat(position_cap_trade_frames, ignore_index=True)
        if position_cap_trade_frames
        else pd.DataFrame()
    )
    security_summary_df.attrs["position_close_summaries"] = (
        pd.concat(position_cap_position_frames, ignore_index=True)
        if position_cap_position_frames
        else pd.DataFrame()
    )
    return order_events_df, trades_df, security_summary_df


VARIANT_TAGS = ["all", "lt1000", "lt2000", "liqcap5tick", "poscap_min5", "poscap_avg5", "poscap_avg5x5_partial"]


def _add_variant_flags(df: pd.DataFrame) -> pd.DataFrame:
    # 每条子单/交易只生成一次，再用这些布尔标签切出不同观察口径。
    if df.empty:
        return df

    flagged_df = df.copy()
    flagged_df["variantAll"] = True
    flagged_df["variantLt1000"] = flagged_df["clientQty"] < 1000
    flagged_df["variantLt2000"] = flagged_df["clientQty"] < 2000
    if "liquidityCapQty" in flagged_df.columns:
        flagged_df["variantLiqcap5tick"] = (
            flagged_df["liquidityCapQty"].notna()
            & (flagged_df["clientQty"].astype(float) <= flagged_df["liquidityCapQty"].astype(float))
        )
    else:
        flagged_df["variantLiqcap5tick"] = False
    if "variantPoscapMin5" not in flagged_df.columns:
        flagged_df["variantPoscapMin5"] = False
    if "variantPoscapAvg5" not in flagged_df.columns:
        flagged_df["variantPoscapAvg5"] = False
    if "variantPoscapAvg5x5Partial" not in flagged_df.columns:
        flagged_df["variantPoscapAvg5x5Partial"] = False
    return flagged_df


def _variant_mask(df: pd.DataFrame, variant_tag: str) -> pd.Series:
    if variant_tag == "all":
        return pd.Series(True, index=df.index)
    if variant_tag == "lt1000":
        return df["variantLt1000"].fillna(False)
    if variant_tag == "lt2000":
        return df["variantLt2000"].fillna(False)
    if variant_tag == "liqcap5tick":
        return df["variantLiqcap5tick"].fillna(False)
    if variant_tag == "poscap_min5":
        return df["variantPoscapMin5"].fillna(False)
    if variant_tag == "poscap_avg5":
        return df["variantPoscapAvg5"].fillna(False)
    if variant_tag == "poscap_avg5x5_partial":
        return df["variantPoscapAvg5x5Partial"].fillna(False)
    raise ValueError(f"Unknown variant_tag: {variant_tag}")


def _max_concurrent_from_trades(sec_trade_df: pd.DataFrame, side: str) -> int:
    # variant 汇总不再重跑持仓循环，因此从交易开平时间反推该口径下的最大并发持仓。
    side_trade_df = sec_trade_df[sec_trade_df["side"] == side] if not sec_trade_df.empty else pd.DataFrame()
    if side_trade_df.empty:
        return 0

    events: list[tuple[pd.Timestamp, int, int]] = []
    for row in side_trade_df.to_dict(orient="records"):
        events.append((pd.Timestamp(row["openTime"]), 0, 1))
        events.append((pd.Timestamp(row["closeTime"]), 1, -1))

    concurrent = 0
    max_concurrent = 0
    for _, _, delta in sorted(events):
        concurrent += delta
        max_concurrent = max(max_concurrent, concurrent)
    return int(max_concurrent)


def _max_concurrent_qty_from_trades(sec_trade_df: pd.DataFrame, side: str) -> int:
    side_trade_df = sec_trade_df[sec_trade_df["side"] == side] if not sec_trade_df.empty else pd.DataFrame()
    if side_trade_df.empty:
        return 0

    events: list[tuple[pd.Timestamp, int, int]] = []
    for row in side_trade_df.to_dict(orient="records"):
        qty = int(row["clientQty"])
        events.append((pd.Timestamp(row["openTime"]), 0, qty))
        events.append((pd.Timestamp(row["closeTime"]), 1, -qty))

    concurrent_qty = 0
    max_concurrent_qty = 0
    for _, _, delta in sorted(events):
        concurrent_qty += delta
        max_concurrent_qty = max(max_concurrent_qty, concurrent_qty)
    return int(max_concurrent_qty)


def _apply_variant_trade_metrics(summary_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    # 将某个 variant 的交易收益写回 security summary；没有交易的股票保持 0 / NaN。
    for security_code, sec_trade_df in trades_df.groupby("securityCode", sort=False):
        row_mask = summary_df["securityCode"] == security_code
        sec_trade_long_df = sec_trade_df[sec_trade_df["side"] == "LONG"]
        sec_trade_short_df = sec_trade_df[sec_trade_df["side"] == "SHORT"]
        total_notional = float(sec_trade_df["openNotional"].sum())

        summary_df.loc[row_mask, "closedLongCount"] = int(len(sec_trade_long_df))
        summary_df.loc[row_mask, "closedShortCount"] = int(len(sec_trade_short_df))
        summary_df.loc[row_mask, "avgLongHoldMinutes"] = float(sec_trade_long_df["holdMinutes"].mean()) if not sec_trade_long_df.empty else np.nan
        summary_df.loc[row_mask, "avgShortHoldMinutes"] = float(sec_trade_short_df["holdMinutes"].mean()) if not sec_trade_short_df.empty else np.nan
        summary_df.loc[row_mask, "avgLongMidRet"] = float(sec_trade_long_df["midRet"].mean()) if not sec_trade_long_df.empty else np.nan
        summary_df.loc[row_mask, "avgShortMidRet"] = float(sec_trade_short_df["midRet"].mean()) if not sec_trade_short_df.empty else np.nan
        summary_df.loc[row_mask, "avgLongExecRet"] = float(sec_trade_long_df["execRet"].mean()) if not sec_trade_long_df.empty else np.nan
        summary_df.loc[row_mask, "avgShortExecRet"] = float(sec_trade_short_df["execRet"].mean()) if not sec_trade_short_df.empty else np.nan
        summary_df.loc[row_mask, "totalLongMidRet"] = float(sec_trade_long_df["midRet"].sum()) if not sec_trade_long_df.empty else 0.0
        summary_df.loc[row_mask, "totalShortMidRet"] = float(sec_trade_short_df["midRet"].sum()) if not sec_trade_short_df.empty else 0.0
        summary_df.loc[row_mask, "totalLongExecRet"] = float(sec_trade_long_df["execRet"].sum()) if not sec_trade_long_df.empty else 0.0
        summary_df.loc[row_mask, "totalShortExecRet"] = float(sec_trade_short_df["execRet"].sum()) if not sec_trade_short_df.empty else 0.0
        summary_df.loc[row_mask, "totalLongMidPnl"] = float(sec_trade_long_df["midPnl"].sum()) if not sec_trade_long_df.empty else 0.0
        summary_df.loc[row_mask, "totalShortMidPnl"] = float(sec_trade_short_df["midPnl"].sum()) if not sec_trade_short_df.empty else 0.0
        summary_df.loc[row_mask, "totalLongExecPnl"] = float(sec_trade_long_df["execPnl"].sum()) if not sec_trade_long_df.empty else 0.0
        summary_df.loc[row_mask, "totalShortExecPnl"] = float(sec_trade_short_df["execPnl"].sum()) if not sec_trade_short_df.empty else 0.0
        summary_df.loc[row_mask, "longMidWinRate"] = float((sec_trade_long_df["midRet"] > 0).mean()) if not sec_trade_long_df.empty else np.nan
        summary_df.loc[row_mask, "shortMidWinRate"] = float((sec_trade_short_df["midRet"] > 0).mean()) if not sec_trade_short_df.empty else np.nan
        summary_df.loc[row_mask, "longExecWinRate"] = float((sec_trade_long_df["execRet"] > 0).mean()) if not sec_trade_long_df.empty else np.nan
        summary_df.loc[row_mask, "shortExecWinRate"] = float((sec_trade_short_df["execRet"] > 0).mean()) if not sec_trade_short_df.empty else np.nan
        summary_df.loc[row_mask, "longEodCloseCount"] = int((sec_trade_long_df["closeType"] == "EOD").sum()) if not sec_trade_long_df.empty else 0
        summary_df.loc[row_mask, "shortEodCloseCount"] = int((sec_trade_short_df["closeType"] == "EOD").sum()) if not sec_trade_short_df.empty else 0
        summary_df.loc[row_mask, "totalMatchedNotional"] = total_notional
        summary_df.loc[row_mask, "notionalWeightedMidRet"] = float(sec_trade_df["midPnl"].sum() / total_notional) if total_notional != 0 else np.nan
        summary_df.loc[row_mask, "notionalWeightedExecRet"] = float(sec_trade_df["execPnl"].sum() / total_notional) if total_notional != 0 else np.nan
        summary_df.loc[row_mask, "maxConcurrentLongCount"] = _max_concurrent_from_trades(sec_trade_df, "LONG")
        summary_df.loc[row_mask, "maxConcurrentShortCount"] = _max_concurrent_from_trades(sec_trade_df, "SHORT")
        summary_df.loc[row_mask, "maxConcurrentLongQty"] = _max_concurrent_qty_from_trades(sec_trade_df, "LONG")
        summary_df.loc[row_mask, "maxConcurrentShortQty"] = _max_concurrent_qty_from_trades(sec_trade_df, "SHORT")

    return summary_df


def build_variant_security_summary(
    base_security_summary_df: pd.DataFrame,
    order_events_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    variant_tag: str,
) -> pd.DataFrame:
    # denominator 保持全局客户流，只替换该 variant 真正贡献的 matched/trade 分子。
    if base_security_summary_df.empty:
        return base_security_summary_df
    if variant_tag == "all":
        return base_security_summary_df.copy()

    summary_df = base_security_summary_df.copy()
    matched_count_cols = [
        "matchedClientChildCount",
        "matchedClientBuyChildCount",
        "matchedClientSellChildCount",
        "matchedClientQty",
        "matchedClientBuyQty",
        "matchedClientSellQty",
        "longOpenCount",
        "shortOpenCount",
    ]
    matched_amt_cols = [
        "matchedClientAmt",
        "matchedClientBuyAmt",
        "matchedClientSellAmt",
    ]
    trade_count_cols = [
        "closedLongCount",
        "closedShortCount",
        "longEodCloseCount",
        "shortEodCloseCount",
        "maxConcurrentLongCount",
        "maxConcurrentShortCount",
        "maxConcurrentLongQty",
        "maxConcurrentShortQty",
    ]
    trade_float_cols = [
        "totalLongMidRet",
        "totalShortMidRet",
        "totalLongExecRet",
        "totalShortExecRet",
        "totalLongMidPnl",
        "totalShortMidPnl",
        "totalLongExecPnl",
        "totalShortExecPnl",
        "totalMatchedNotional",
    ]
    trade_nan_cols = [
        "avgLongHoldMinutes",
        "avgShortHoldMinutes",
        "avgLongMidRet",
        "avgShortMidRet",
        "avgLongExecRet",
        "avgShortExecRet",
        "longMidWinRate",
        "shortMidWinRate",
        "longExecWinRate",
        "shortExecWinRate",
        "notionalWeightedMidRet",
        "notionalWeightedExecRet",
    ]
    summary_df[matched_count_cols + trade_count_cols] = 0
    summary_df[matched_amt_cols + trade_float_cols] = 0.0
    summary_df[trade_nan_cols] = np.nan

    matched_events_df = order_events_df[
        order_events_df["matched"].fillna(False) & _variant_mask(order_events_df, variant_tag)
    ]
    for security_code, sec_event_df in matched_events_df.groupby("securityCode", sort=False):
        row_mask = summary_df["securityCode"] == security_code
        matched_buy_df = sec_event_df[sec_event_df["clientSide"] == "B"]
        matched_sell_df = sec_event_df[sec_event_df["clientSide"] == "S"]
        summary_df.loc[row_mask, "matchedClientChildCount"] = int(len(sec_event_df))
        summary_df.loc[row_mask, "matchedClientBuyChildCount"] = int(len(matched_buy_df))
        summary_df.loc[row_mask, "matchedClientSellChildCount"] = int(len(matched_sell_df))
        summary_df.loc[row_mask, "matchedClientQty"] = int(sec_event_df["clientQty"].sum())
        summary_df.loc[row_mask, "matchedClientBuyQty"] = int(matched_buy_df["clientQty"].sum()) if not matched_buy_df.empty else 0
        summary_df.loc[row_mask, "matchedClientSellQty"] = int(matched_sell_df["clientQty"].sum()) if not matched_sell_df.empty else 0
        summary_df.loc[row_mask, "matchedClientAmt"] = float(sec_event_df["clientFilledAmt"].sum())
        summary_df.loc[row_mask, "matchedClientBuyAmt"] = float(matched_buy_df["clientFilledAmt"].sum()) if not matched_buy_df.empty else 0.0
        summary_df.loc[row_mask, "matchedClientSellAmt"] = float(matched_sell_df["clientFilledAmt"].sum()) if not matched_sell_df.empty else 0.0
        summary_df.loc[row_mask, "longOpenCount"] = int(len(matched_sell_df))
        summary_df.loc[row_mask, "shortOpenCount"] = int(len(matched_buy_df))

    summary_df["unmatchedClientChildCount"] = summary_df["clientChildCount"] - summary_df["matchedClientChildCount"]

    variant_trades_df = trades_df[_variant_mask(trades_df, variant_tag)] if not trades_df.empty else trades_df
    return _apply_variant_trade_metrics(summary_df, variant_trades_df)


def aggregate_internalization_summary(
    security_summary_df: pd.DataFrame,
    params: BacktestParams,
    scope: str,
    variant_tag: str = "all",
) -> pd.DataFrame:
    # 这里汇总的是 internalization 语义下最关心的指标：
    # 匹配率、成交量/金额覆盖率、开仓次数、收益、并发持仓等。
    if security_summary_df.empty:
        return pd.DataFrame()

    total_client_buy_count = int(security_summary_df["clientBuyChildCount"].sum())
    total_client_sell_count = int(security_summary_df["clientSellChildCount"].sum())
    total_client_child_count = int(security_summary_df["clientChildCount"].sum())
    matched_client_child_count = int(security_summary_df["matchedClientChildCount"].sum())
    matched_client_buy_count = int(security_summary_df["matchedClientBuyChildCount"].sum())
    matched_client_sell_count = int(security_summary_df["matchedClientSellChildCount"].sum())
    total_client_buy_qty = int(security_summary_df["clientBuyQty"].sum())
    total_client_sell_qty = int(security_summary_df["clientSellQty"].sum())
    total_client_qty = total_client_buy_qty + total_client_sell_qty
    total_client_amt = float(security_summary_df["clientAmt"].sum())
    total_client_buy_amt = float(security_summary_df["clientBuyAmt"].sum())
    total_client_sell_amt = float(security_summary_df["clientSellAmt"].sum())
    matched_client_qty = int(security_summary_df["matchedClientQty"].sum())
    matched_client_buy_qty = int(security_summary_df["matchedClientBuyQty"].sum())
    matched_client_sell_qty = int(security_summary_df["matchedClientSellQty"].sum())
    matched_client_amt = float(security_summary_df["matchedClientAmt"].sum())
    matched_client_buy_amt = float(security_summary_df["matchedClientBuyAmt"].sum())
    matched_client_sell_amt = float(security_summary_df["matchedClientSellAmt"].sum())
    total_closed_long_count = int(security_summary_df["closedLongCount"].sum())
    total_closed_short_count = int(security_summary_df["closedShortCount"].sum())
    total_long_mid_ret = float(security_summary_df["totalLongMidRet"].sum())
    total_short_mid_ret = float(security_summary_df["totalShortMidRet"].sum())
    total_long_exec_ret = float(security_summary_df["totalLongExecRet"].sum())
    total_short_exec_ret = float(security_summary_df["totalShortExecRet"].sum())
    total_long_mid_pnl = float(security_summary_df["totalLongMidPnl"].sum())
    total_short_mid_pnl = float(security_summary_df["totalShortMidPnl"].sum())
    total_long_exec_pnl = float(security_summary_df["totalLongExecPnl"].sum())
    total_short_exec_pnl = float(security_summary_df["totalShortExecPnl"].sum())
    total_matched_notional = float(security_summary_df["totalMatchedNotional"].sum())
    total_trade_count = total_closed_long_count + total_closed_short_count
    total_long_open_count = int(security_summary_df["longOpenCount"].sum())
    total_short_open_count = int(security_summary_df["shortOpenCount"].sum())

    row = {
        "scope": scope,
        "variantTag": variant_tag,
        "paramTag": params.param_tag,
        "openThreshold": params.open_threshold,
        "closeThreshold": params.close_threshold,
        "minHoldBars": params.min_hold_bars,
        "minHoldMinutes": params.min_hold_minutes,
        "relaxedCloseThreshold": params.relaxed_close_threshold,
        "relaxedCloseAfterBars": params.relaxed_close_after_bars,
        "tradeDateCount": int(security_summary_df["tradeDate"].nunique()),
        "securityCount": int(security_summary_df["securityCode"].nunique()),
        "securityDayCount": int(len(security_summary_df)),
        "totalBarCount": int(security_summary_df["totalBarCount"].sum()),
        "totalLongSignalCount": int(security_summary_df["longSignalCount"].sum()),
        "totalShortSignalCount": int(security_summary_df["shortSignalCount"].sum()),
        "totalClientChildCount": total_client_child_count,
        "matchedClientChildCount": matched_client_child_count,
        "unmatchedClientChildCount": total_client_child_count - matched_client_child_count,
        "clientChildMatchRate": np.nan if total_client_child_count == 0 else matched_client_child_count / total_client_child_count,
        "totalClientBuyChildCount": total_client_buy_count,
        "totalClientSellChildCount": total_client_sell_count,
        "matchedClientBuyChildCount": matched_client_buy_count,
        "matchedClientSellChildCount": matched_client_sell_count,
        "unmatchedClientBuyChildCount": total_client_buy_count - matched_client_buy_count,
        "unmatchedClientSellChildCount": total_client_sell_count - matched_client_sell_count,
        "clientBuyMatchRate": np.nan if total_client_buy_count == 0 else matched_client_buy_count / total_client_buy_count,
        "totalClientBuyQty": total_client_buy_qty,
        "totalClientSellQty": total_client_sell_qty,
        "totalClientQty": total_client_qty,
        "totalClientAmt": total_client_amt,
        "totalClientBuyAmt": total_client_buy_amt,
        "totalClientSellAmt": total_client_sell_amt,
        "matchedClientQty": matched_client_qty,
        "matchedClientBuyQty": matched_client_buy_qty,
        "matchedClientSellQty": matched_client_sell_qty,
        "matchedClientAmt": matched_client_amt,
        "matchedClientBuyAmt": matched_client_buy_amt,
        "matchedClientSellAmt": matched_client_sell_amt,
        "clientQtyMatchRate": np.nan if total_client_qty == 0 else matched_client_qty / total_client_qty,
        "clientBuyQtyMatchRate": np.nan if total_client_buy_qty == 0 else matched_client_buy_qty / total_client_buy_qty,
        "clientSellQtyMatchRate": np.nan if total_client_sell_qty == 0 else matched_client_sell_qty / total_client_sell_qty,
        "clientAmtMatchRate": np.nan if total_client_amt == 0 else matched_client_amt / total_client_amt,
        "clientBuyAmtMatchRate": np.nan if total_client_buy_amt == 0 else matched_client_buy_amt / total_client_buy_amt,
        "clientSellAmtMatchRate": np.nan if total_client_sell_amt == 0 else matched_client_sell_amt / total_client_sell_amt,
        "totalLongOpenCount": total_long_open_count,
        "totalShortOpenCount": total_short_open_count,
        "totalClosedLongCount": total_closed_long_count,
        "totalClosedShortCount": total_closed_short_count,
        "totalTradeCount": total_trade_count,
        "avgLongHoldMinutes": _weighted_average(security_summary_df, "avgLongHoldMinutes", "closedLongCount"),
        "avgShortHoldMinutes": _weighted_average(security_summary_df, "avgShortHoldMinutes", "closedShortCount"),
        "avgAllHoldMinutes": (
            np.nan
            if total_trade_count == 0
            else (
                np.nan_to_num(_weighted_average(security_summary_df, "avgLongHoldMinutes", "closedLongCount")) * total_closed_long_count
                + np.nan_to_num(_weighted_average(security_summary_df, "avgShortHoldMinutes", "closedShortCount")) * total_closed_short_count
            ) / total_trade_count
        ),
        "avgLongMidRet": np.nan if total_closed_long_count == 0 else total_long_mid_ret / total_closed_long_count,
        "avgShortMidRet": np.nan if total_closed_short_count == 0 else total_short_mid_ret / total_closed_short_count,
        "avgAllMidRet": np.nan if total_trade_count == 0 else (total_long_mid_ret + total_short_mid_ret) / total_trade_count,
        "avgLongExecRet": np.nan if total_closed_long_count == 0 else total_long_exec_ret / total_closed_long_count,
        "avgShortExecRet": np.nan if total_closed_short_count == 0 else total_short_exec_ret / total_closed_short_count,
        "avgAllExecRet": np.nan if total_trade_count == 0 else (total_long_exec_ret + total_short_exec_ret) / total_trade_count,
        "totalLongMidRet": total_long_mid_ret,
        "totalShortMidRet": total_short_mid_ret,
        "totalMidRet": total_long_mid_ret + total_short_mid_ret,
        "totalLongExecRet": total_long_exec_ret,
        "totalShortExecRet": total_short_exec_ret,
        "totalExecRet": total_long_exec_ret + total_short_exec_ret,
        "totalLongMidPnl": total_long_mid_pnl,
        "totalShortMidPnl": total_short_mid_pnl,
        "totalMidPnl": total_long_mid_pnl + total_short_mid_pnl,
        "totalLongExecPnl": total_long_exec_pnl,
        "totalShortExecPnl": total_short_exec_pnl,
        "totalExecPnl": total_long_exec_pnl + total_short_exec_pnl,
        "longMidWinRate": _weighted_average(security_summary_df, "longMidWinRate", "closedLongCount"),
        "shortMidWinRate": _weighted_average(security_summary_df, "shortMidWinRate", "closedShortCount"),
        "longExecWinRate": _weighted_average(security_summary_df, "longExecWinRate", "closedLongCount"),
        "shortExecWinRate": _weighted_average(security_summary_df, "shortExecWinRate", "closedShortCount"),
        "totalEodCloseCount": int(security_summary_df["longEodCloseCount"].sum() + security_summary_df["shortEodCloseCount"].sum()),
        "maxConcurrentLongCount": int(security_summary_df["maxConcurrentLongCount"].max()),
        "maxConcurrentShortCount": int(security_summary_df["maxConcurrentShortCount"].max()),
        "maxConcurrentTotalCount": int((security_summary_df["maxConcurrentLongCount"] + security_summary_df["maxConcurrentShortCount"]).max()),
        "maxConcurrentLongQty": int(security_summary_df["maxConcurrentLongQty"].max()),
        "maxConcurrentShortQty": int(security_summary_df["maxConcurrentShortQty"].max()),
        "maxConcurrentTotalQty": int((security_summary_df["maxConcurrentLongQty"] + security_summary_df["maxConcurrentShortQty"]).max()),
        "avgMaxConcurrentLongCount": float(security_summary_df["maxConcurrentLongCount"].mean()),
        "avgMaxConcurrentShortCount": float(security_summary_df["maxConcurrentShortCount"].mean()),
        "avgMaxConcurrentTotalCount": float((security_summary_df["maxConcurrentLongCount"] + security_summary_df["maxConcurrentShortCount"]).mean()),
        "avgMaxConcurrentLongQty": float(security_summary_df["maxConcurrentLongQty"].mean()),
        "avgMaxConcurrentShortQty": float(security_summary_df["maxConcurrentShortQty"].mean()),
        "avgMaxConcurrentTotalQty": float((security_summary_df["maxConcurrentLongQty"] + security_summary_df["maxConcurrentShortQty"]).mean()),
        "p95MaxConcurrentLongCount": _series_p95(security_summary_df, "maxConcurrentLongCount"),
        "p95MaxConcurrentShortCount": _series_p95(security_summary_df, "maxConcurrentShortCount"),
        "p95MaxConcurrentTotalCount": _series_p95(
            security_summary_df.assign(maxConcurrentTotalCount=security_summary_df["maxConcurrentLongCount"] + security_summary_df["maxConcurrentShortCount"]),
            "maxConcurrentTotalCount",
        ),
        "p95MaxConcurrentLongQty": _series_p95(security_summary_df, "maxConcurrentLongQty"),
        "p95MaxConcurrentShortQty": _series_p95(security_summary_df, "maxConcurrentShortQty"),
        "p95MaxConcurrentTotalQty": _series_p95(
            security_summary_df.assign(maxConcurrentTotalQty=security_summary_df["maxConcurrentLongQty"] + security_summary_df["maxConcurrentShortQty"]),
            "maxConcurrentTotalQty",
        ),
        "totalMatchedNotional": total_matched_notional,
        "notionalWeightedMidRet": np.nan if total_matched_notional == 0 else (total_long_mid_pnl + total_short_mid_pnl) / total_matched_notional,
        "notionalWeightedExecRet": np.nan if total_matched_notional == 0 else (total_long_exec_pnl + total_short_exec_pnl) / total_matched_notional,
    }
    return pd.DataFrame([row])


def load_internalization_day_inputs(
    trade_date: str,
    pool_name: str,
    ims_roots: list[Path | str],
    mysql_config: MysqlConfig | None = None,
    ddb_config: DdbConfig | None = None,
    profile: bool = False,
    signal_table_name: str | None = None,
) -> dict[str, object] | None:
    # Data here is independent of strategy parameters and can be cached per date/pool.
    mysql_config = mysql_config or MysqlConfig()

    def record_stage(stage_name: str, start_time: float, **extra: object) -> None:
        if not profile:
            return
        extra_text = " ".join(f"{key}={value}" for key, value in extra.items())
        suffix = f" {extra_text}" if extra_text else ""
        print(f"[profile] {pool_name} {stage_name} seconds={perf_counter() - start_time:.2f}{suffix}")

    stage_start = perf_counter()
    ims_security_codes = discover_ims_security_codes(trade_date, ims_roots)
    record_stage("discover_ims_security_codes", stage_start, securityCount=len(ims_security_codes))
    if not ims_security_codes:
        return None

    stage_start = perf_counter()
    pool_universe_codes = load_pool_universe_mysql(
        trade_date=trade_date,
        pool_name=pool_name,
        mysql_config=mysql_config,
    )
    record_stage("load_pool_universe_mysql", stage_start, securityCount=len(pool_universe_codes))
    signal_filter_codes = sorted(set(ims_security_codes) & set(pool_universe_codes)) if pool_universe_codes else ims_security_codes
    if not signal_filter_codes:
        return None

    stage_start = perf_counter()
    signal_df = load_signal_day_for_internalization(
        trade_date=trade_date,
        pool_name=pool_name,
        mysql_config=mysql_config,
        security_codes=signal_filter_codes,
        signal_table_name=signal_table_name,
    )
    record_stage("load_signal_mysql", stage_start, rowCount=len(signal_df), securityCount=len(signal_filter_codes))
    if signal_df.empty:
        return None

    security_codes = signal_df["securityCode"].unique().tolist()
    stage_start = perf_counter()
    base_client_order_df = load_ims_child_orders(
        trade_date,
        ims_roots,
        security_codes=security_codes,
    )
    record_stage("load_ims_child_orders", stage_start, rowCount=len(base_client_order_df))
    stage_start = perf_counter()
    current_price_df = load_aligned_price_df(
        trade_date=trade_date,
        pool_name=pool_name,
        security_codes=security_codes,
        ddb_config=ddb_config,
    )
    record_stage("load_current_returns_price", stage_start, rowCount=len(current_price_df))
    close_price_map = {
        security_code: sec_df.reset_index(drop=True)
        for security_code, sec_df in current_price_df.groupby("securityCode", sort=True)
    } if not current_price_df.empty else {}

    stage_start = perf_counter()
    limit_df = load_a_share_limit_df(
        trade_date=trade_date,
        security_codes=security_codes,
        mysql_config=mysql_config,
    )
    record_stage("load_static_limit_mysql", stage_start, rowCount=len(limit_df))
    if not validate_a_share_limit_data(trade_date, security_codes, limit_df):
        return None
    limit_map = limit_df.set_index("securityCode")[["highLimit", "lowLimit"]].to_dict("index") if not limit_df.empty else {}

    stage_start = perf_counter()
    prev_price_df = load_previous_aligned_price_df(
        trade_date=trade_date,
        pool_name=pool_name,
        security_codes=security_codes,
        ddb_config=ddb_config,
    )
    record_stage("load_previous_returns_price", stage_start, rowCount=len(prev_price_df))
    stage_start = perf_counter()
    meta_df = build_internalization_meta(
        signal_df=signal_df,
        pool_name=pool_name,
        trade_date=trade_date,
        current_price_df=current_price_df,
        prev_price_df=prev_price_df,
    )
    record_stage("build_meta", stage_start, rowCount=len(meta_df))

    return {
        "tradeDate": trade_date,
        "poolName": pool_name,
        "signalDf": signal_df,
        "metaDf": meta_df,
        "clientOrderDf": base_client_order_df,
        "closePriceMap": close_price_map,
        "limitMap": limit_map,
    }


def run_internalization_prepared_day(
    prepared_inputs: dict[str, object],
    params: BacktestParams,
    ddb_config: DdbConfig | None = None,
    match_window_seconds: int | None = 10,
    profile: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trade_date = str(prepared_inputs["tradeDate"])
    pool_name = str(prepared_inputs["poolName"])

    def record_stage(stage_name: str, start_time: float, **extra: object) -> None:
        if not profile:
            return
        extra_text = " ".join(f"{key}={value}" for key, value in extra.items())
        suffix = f" {extra_text}" if extra_text else ""
        print(f"[profile] {pool_name} {stage_name} seconds={perf_counter() - start_time:.2f}{suffix}")

    signal_df = prepared_inputs["signalDf"]
    meta_df = prepared_inputs["metaDf"]
    base_client_order_df = prepared_inputs["clientOrderDf"]
    close_price_map = prepared_inputs["closePriceMap"]
    limit_map = prepared_inputs["limitMap"]

    all_security_summaries: list[pd.DataFrame] = []
    pool_summary_frames: list[pd.DataFrame] = []

    tick_mid_cache = TickMidCache(trade_date=trade_date, ddb_config=ddb_config)
    try:
        stage_start = perf_counter()
        order_events_df, trades_df, base_security_summary_df = simulate_internalization_day(
            signal_df=signal_df,
            meta_df=meta_df,
            client_order_df=base_client_order_df,
            pool_name=pool_name,
            params=params,
            tick_mid_cache=tick_mid_cache,
            close_price_map=close_price_map,
            limit_map=limit_map,
            match_window_seconds=match_window_seconds,
        )
        record_stage(
            "simulate_internalization",
            stage_start,
            orderEventCount=len(order_events_df),
            tradeCount=len(trades_df),
            securityCount=len(base_security_summary_df),
            tickLoadCount=tick_mid_cache.load_count,
            tickLoadSeconds=f"{tick_mid_cache.load_seconds:.2f}",
        )
        position_cap_trades_df = base_security_summary_df.attrs.get("position_cap_trades", pd.DataFrame())
        position_close_summary_df = base_security_summary_df.attrs.get("position_close_summaries", pd.DataFrame())
        stage_start = perf_counter()
        order_events_df = _add_variant_flags(order_events_df)
        trades_df = _add_variant_flags(trades_df)

        for variant_tag in [tag for tag in VARIANT_TAGS if not tag.startswith("poscap_")]:
            security_summary_df = build_variant_security_summary(
                base_security_summary_df=base_security_summary_df,
                order_events_df=order_events_df,
                trades_df=trades_df,
                variant_tag=variant_tag,
            )
            if not security_summary_df.empty:
                security_summary_df = security_summary_df.copy()
                security_summary_df["variantTag"] = variant_tag

            all_security_summaries.append(security_summary_df)
            pool_summary_frames.append(
                aggregate_internalization_summary(
                    security_summary_df=security_summary_df,
                    params=params,
                    scope=pool_name,
                    variant_tag=variant_tag,
                )
            )
        position_cap_summary_df = base_security_summary_df.attrs.get("position_cap_summaries", pd.DataFrame())
        if not position_cap_summary_df.empty:
            for variant_tag in ["poscap_min5", "poscap_avg5", "poscap_avg5x5_partial"]:
                cap_variant_summary_df = position_cap_summary_df[
                    position_cap_summary_df["variantTag"] == variant_tag
                ].copy()
                all_security_summaries.append(cap_variant_summary_df)
                pool_summary_frames.append(
                    aggregate_internalization_summary(
                        security_summary_df=cap_variant_summary_df,
                        params=params,
                        scope=pool_name,
                        variant_tag=variant_tag,
                    )
                )
        record_stage("build_variants", stage_start, variantCount=len(VARIANT_TAGS))
    finally:
        tick_mid_cache.close()

    security_summary_df = pd.concat(all_security_summaries, ignore_index=True) if all_security_summaries else pd.DataFrame()
    pool_summary_df = pd.concat(pool_summary_frames, ignore_index=True) if pool_summary_frames else pd.DataFrame()
    trades_df.attrs["position_cap_trades"] = position_cap_trades_df
    trades_df.attrs["position_close_summaries"] = position_close_summary_df
    security_summary_df = format_internalization_security_summary_for_output(security_summary_df)
    pool_summary_df = format_internalization_pool_summary_for_output(pool_summary_df)
    trades_df.attrs["position_cap_trades"] = position_cap_trades_df
    trades_df.attrs["position_close_summaries"] = position_close_summary_df
    return order_events_df, trades_df, security_summary_df, pool_summary_df


def run_internalization_single_day(
    trade_date: str,
    pool_name: str,
    params: BacktestParams,
    ims_roots: list[Path | str],
    mysql_config: MysqlConfig | None = None,
    ddb_config: DdbConfig | None = None,
    match_window_seconds: int | None = 10,
    profile: bool = False,
    signal_table_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 单日入口负责把数据源拼齐，然后分别跑不同 variant。
    prepared_inputs = load_internalization_day_inputs(
        trade_date=trade_date,
        pool_name=pool_name,
        ims_roots=ims_roots,
        mysql_config=mysql_config,
        ddb_config=ddb_config,
        profile=profile,
        signal_table_name=signal_table_name,
    )
    if prepared_inputs is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return run_internalization_prepared_day(
        prepared_inputs=prepared_inputs,
        params=params,
        ddb_config=ddb_config,
        match_window_seconds=match_window_seconds,
        profile=profile,
    )
