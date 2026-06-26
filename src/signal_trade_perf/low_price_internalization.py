from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .configs import DdbConfig, MysqlConfig
from .internalization import TickMidCache
from .internalization_capital import build_capital_event_rows, capital_metrics_by_variant
from .internalization_data import get_default_ims_roots, load_aligned_price_df
from .low_price_internalization_data import load_low_price_day_inputs


# 低价股挂单策略的容量约束分组：
# - lowcap_min5：过去 5 个 tick 的 bid/ask 一档量均值取 min，更保守；
# - lowcap_avg5_half：bid/ask 一档量均值的平均值 * 0.5；
# - lowcap_avg5_075：bid/ask 一档量均值的平均值 * 0.75。
LOW_PRICE_VARIANTS = ["lowcap_min5", "lowcap_avg5_half", "lowcap_avg5_075"]


@dataclass(frozen=True)
class LowPriceBacktestParams:
    # signal_ranks 是精确 rank 集合，不是阈值区间；(1, 2) 只使用 top10/top20。
    signal_ranks: tuple[int, ...] = (1, 2)
    match_window_seconds: int | None = 10
    spread: float = 0.01

    @property
    def signal_rank_threshold(self) -> int:
        return max(self.signal_ranks) if self.signal_ranks else 0

    @property
    def param_tag(self) -> str:
        match_tag = "match_unlimited" if self.match_window_seconds is None else f"match_{self.match_window_seconds}"
        rank_tag = "_".join(str(rank) for rank in self.signal_ranks)
        return f"rank_{rank_tag}_{match_tag}_spread_{self.spread:g}"


def _match_window_start(order_time: pd.Timestamp, match_window_seconds: int | None) -> pd.Timestamp:
    # None 表示不限制客户单和 signal 的时间差；否则只允许客户单前 N 秒内的 signal。
    if match_window_seconds is None:
        return pd.Timestamp.min
    return order_time - pd.Timedelta(seconds=match_window_seconds)


def _eligible_signal_mask(signal_df: pd.DataFrame, client_side: str, ranks: tuple[int, ...]) -> pd.Series:
    # 客户卖出时，我们接走客户卖单形成 LONG，需要 bs_flag=s 的卖出挂单成交概率信号；
    # 客户买入时，我们接走客户买单形成 SHORT，需要 bs_flag=b 的买入挂单成交概率信号。
    rank_values = {abs(int(rank)) for rank in ranks}
    if client_side == "S":
        return signal_df["bsFlag"].eq("s") & signal_df["mergeSignal"].abs().isin(rank_values) & signal_df["mergeSignal"].lt(0)
    return signal_df["bsFlag"].eq("b") & signal_df["mergeSignal"].abs().isin(rank_values) & signal_df["mergeSignal"].gt(0)


def _find_latest_eligible_signal_row(
    sec_signal_df: pd.DataFrame,
    order_time: pd.Timestamp,
    client_side: str,
    params: LowPriceBacktestParams,
) -> tuple[pd.Series | None, int]:
    # 低价股同一个 15s bar 有 b/s 两行；先定位最近 bar，再在这个 bar 内找客户方向对应的 signal。
    if sec_signal_df.empty:
        return None, -1
    signal_times = sec_signal_df["barTime"].drop_duplicates().to_numpy(dtype="datetime64[ns]")
    matched_pos = int(signal_times.searchsorted(order_time.to_datetime64(), side="right") - 1)
    if matched_pos < 0:
        return None, -1

    matched_time = pd.Timestamp(signal_times[matched_pos])
    if matched_time < _match_window_start(order_time, params.match_window_seconds):
        return None, -1

    same_bar_df = sec_signal_df[sec_signal_df["barTime"].eq(matched_time)]
    eligible_df = same_bar_df[_eligible_signal_mask(same_bar_df, client_side, params.signal_ranks)]
    if eligible_df.empty:
        return None, -1
    matched_row = eligible_df.iloc[0]
    return matched_row, int(matched_row.name)


def _merge_tick_to_signal(sec_signal_df: pd.DataFrame, tick_df: pd.DataFrame) -> pd.DataFrame:
    # 给每个 signal bar 回填“这个 bar 之前最近一个 tick”的盘口价量。
    # direction="backward" 保证不会偷看到 signal 之后的 tick。
    if sec_signal_df.empty:
        return sec_signal_df
    if tick_df.empty:
        return sec_signal_df.assign(
            bidPrice1Tick=np.nan,
            askPrice1Tick=np.nan,
            midPriceTick=np.nan,
            bidVol1Tick=np.nan,
            askVol1Tick=np.nan,
            avgBidVol1Last5=np.nan,
            avgAskVol1Last5=np.nan,
        )

    left_df = sec_signal_df.reset_index().rename(columns={"index": "_signalRowIdx"}).sort_values("barTime")
    right_cols = [
        "tickTime",
        "bidPrice1Tick",
        "askPrice1Tick",
        "midPriceTick",
        "bidVol1Tick",
        "askVol1Tick",
        "avgBidVol1Last5",
        "avgAskVol1Last5",
    ]
    right_df = tick_df[right_cols].dropna(subset=["tickTime"]).sort_values("tickTime").reset_index(drop=True)
    merged_df = pd.merge_asof(left_df, right_df, left_on="barTime", right_on="tickTime", direction="backward")
    return merged_df.sort_values("_signalRowIdx").drop(columns=["_signalRowIdx"]).reset_index(drop=True)


def _merge_tick_to_orders(matched_order_df: pd.DataFrame, tick_df: pd.DataFrame) -> pd.DataFrame:
    # 客户单开仓价和挂单价都用客户单到达前最新 tick，同样用 backward asof 防止取到未来 tick。
    if matched_order_df.empty:
        return matched_order_df
    if tick_df.empty:
        return matched_order_df.assign(openMid=np.nan, openBidPrice1=np.nan, openAskPrice1=np.nan, openTickTime=pd.NaT)

    left_df = matched_order_df.sort_values("clientOrderTime").reset_index(drop=True)
    right_df = (
        tick_df[["tickTime", "midPriceTick", "bidPrice1Tick", "askPrice1Tick"]]
        .dropna(subset=["tickTime"])
        .sort_values("tickTime")
        .reset_index(drop=True)
    )
    if right_df.empty:
        return matched_order_df.assign(openMid=np.nan, openBidPrice1=np.nan, openAskPrice1=np.nan, openTickTime=pd.NaT)
    enriched_df = pd.merge_asof(
        left_df,
        right_df,
        left_on="clientOrderTime",
        right_on="tickTime",
        direction="backward",
    )
    return enriched_df.rename(
        columns={
            "midPriceTick": "openMid",
            "bidPrice1Tick": "openBidPrice1",
            "askPrice1Tick": "openAskPrice1",
            "tickTime": "openTickTime",
        }
    )


def _merge_aligned_to_signal(sec_signal_df: pd.DataFrame, aligned_price_df: pd.DataFrame) -> pd.DataFrame:
    # ?????? tick ????????????????? signal ?????? Prices_* ???
    if sec_signal_df.empty:
        return sec_signal_df
    if aligned_price_df.empty:
        return sec_signal_df.assign(
            closeMidAligned=np.nan,
            closeAsk1Aligned=np.nan,
            closeBid1Aligned=np.nan,
        )

    right_df = aligned_price_df[["securityCode", "barTime", "closeMidAligned", "closeAsk1Aligned", "closeBid1Aligned"]]
    return sec_signal_df.merge(right_df, on=["securityCode", "barTime"], how="left")


def _cap_qty(row: pd.Series, variant_tag: str) -> int:
    # 低价股不希望在盘口挂出过大量，所以容量按一档量滚动均值限制。
    # 这里返回的是同一价格周期内、同一敞口方向允许累计打开的最大股数。
    avg_bid = float(row["avgBidVol1Last5"]) if not pd.isna(row.get("avgBidVol1Last5", np.nan)) else np.nan
    avg_ask = float(row["avgAskVol1Last5"]) if not pd.isna(row.get("avgAskVol1Last5", np.nan)) else np.nan
    if pd.isna(avg_bid) or pd.isna(avg_ask):
        return 0
    if variant_tag == "lowcap_min5":
        return max(0, int(np.floor(min(avg_bid, avg_ask))))
    if variant_tag == "lowcap_avg5_half":
        return max(0, int(np.floor(0.5 * ((avg_bid + avg_ask) / 2.0))))
    if variant_tag == "lowcap_avg5_075":
        return max(0, int(np.floor(0.75 * ((avg_bid + avg_ask) / 2.0))))
    raise ValueError(f"Unknown low-price variant: {variant_tag}")


def _next_price_move_indices(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    # 对每个 signal bar 预计算“未来第一次价格上移/下移”的行号，后续逐笔查表即可 O(1)。
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    n = len(arr)
    next_up = np.full(n, -1, dtype=int)
    next_down = np.full(n, -1, dtype=int)
    for idx in range(n):
        current = arr[idx]
        if np.isnan(current):
            continue
        for future_idx in range(idx + 1, n):
            future = arr[future_idx]
            if np.isnan(future) or future == current:
                continue
            if future > current:
                next_up[idx] = future_idx
            else:
                next_down[idx] = future_idx
            break
    return next_up, next_down


def _add_next_price_move_cols(sec_signal_df: pd.DataFrame) -> pd.DataFrame:
    # ???? PnL ??????????????LONG ? ask1?SHORT ? bid1?
    if sec_signal_df.empty:
        return sec_signal_df
    result_df = sec_signal_df.copy()
    ask_series = result_df["closeAsk1Aligned"].where(result_df["closeAsk1Aligned"].notna(), result_df["askPrice1Tick"])
    bid_series = result_df["closeBid1Aligned"].where(result_df["closeBid1Aligned"].notna(), result_df["bidPrice1Tick"])
    result_df["nextAskUpIdx"], result_df["nextAskDownIdx"] = _next_price_move_indices(ask_series)
    result_df["nextBidUpIdx"], result_df["nextBidDownIdx"] = _next_price_move_indices(bid_series)
    return result_df


def _log_price_path_fallback(message: str) -> None:
    # y_test=0 但价格先向有利方向变化时，说明标签可能偏保守；同时打印和落日志方便复盘。
    # print(message)
    log_path = Path.cwd() / "results" / "internalization_backtest" / "logs" / "low_price_price_path_fallback.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as file:
            file.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {message}\n")
    except OSError:
        pass


def _low_price_price_path_pnl(
    sec_signal_df: pd.DataFrame,
    signal_idx: int,
    signal_row: pd.Series,
    side: str,
    open_mid: float,
    posted_price: float,
    y_test: int,
    params: LowPriceBacktestParams,
    context: dict[str, object],
) -> dict[str, object]:
    # B 版真实价格路径 PnL：开仓挂单价取客户到达前最新 tick，后续仍用 signal 序列判断价格路径。
    # 这里只改变 PnL 和 pnlSettleTime，不改变低价股容量 reset/资金占用释放口径。
    signal_time = pd.Timestamp(signal_row.barTime)
    posted_price = float(posted_price) if not pd.isna(posted_price) else np.nan
    old_penalty = -2.0 * float(params.spread)

    if pd.isna(posted_price):
        return {
            "pnlPerShare": old_penalty if int(y_test) == 0 else np.nan,
            "pnlSettleTime": signal_time,
            "postedPrice": np.nan,
            "pricePathClosePrice": np.nan,
            "pnlModel": "price_path_missing_posted_price",
            "pricePathReason": "missing_posted_price",
        }

    if int(y_test) == 1:
        pnl_per_share = (posted_price - open_mid) if side == "LONG" else (open_mid - posted_price)
        return {
            "pnlPerShare": pnl_per_share,
            "pnlSettleTime": signal_time,
            "postedPrice": posted_price,
            "pricePathClosePrice": posted_price,
            "pnlModel": "price_path_y_test_fill",
            "pricePathReason": "y_test_1",
        }

    if side == "LONG":
        next_up_idx = int(signal_row.get("nextAskUpIdx", -1))
        next_down_idx = int(signal_row.get("nextAskDownIdx", -1))
        if next_up_idx >= 0:
            future_row = sec_signal_df.iloc[next_up_idx]
            future_time = pd.Timestamp(future_row.barTime)
            message = (
                "[low_price_price_path_fallback] "
                f"tradeDate={context.get('tradeDate')} pool={context.get('poolName')} security={context.get('securityCode')} "
                f"side={side} signalTime={context.get('signalTime')} postedPrice={posted_price} futureTime={future_time} "
                f"reason=y_test0_but_ask_up_first"
            )
            _log_price_path_fallback(message)
            return {
                "pnlPerShare": posted_price - open_mid,
                "pnlSettleTime": future_time,
                "postedPrice": posted_price,
                "pricePathClosePrice": posted_price,
                "pnlModel": "price_path_y_test0_favorable_fill",
                "pricePathReason": "ask_up_before_ask_down",
            }
        if next_down_idx >= 0:
            future_row = sec_signal_df.iloc[next_down_idx]
            future_time = pd.Timestamp(future_row.barTime)
            bid_t = future_row.get("closeBid1Aligned", np.nan)
            if pd.isna(bid_t):
                bid_t = future_row.get("bidPrice1Tick", np.nan)
            bid_t = float(bid_t) if not pd.isna(bid_t) else np.nan
            return {
                "pnlPerShare": (bid_t - open_mid) if not pd.isna(bid_t) else old_penalty,
                "pnlSettleTime": future_time,
                "postedPrice": posted_price,
                "pricePathClosePrice": bid_t,
                "pnlModel": "price_path_y_test0_adverse_close",
                "pricePathReason": "ask_down_first",
            }
    else:
        next_down_idx = int(signal_row.get("nextBidDownIdx", -1))
        next_up_idx = int(signal_row.get("nextBidUpIdx", -1))
        if next_down_idx >= 0:
            future_row = sec_signal_df.iloc[next_down_idx]
            future_time = pd.Timestamp(future_row.barTime)
            message = (
                "[low_price_price_path_fallback] "
                f"tradeDate={context.get('tradeDate')} pool={context.get('poolName')} security={context.get('securityCode')} "
                f"side={side} signalTime={context.get('signalTime')} postedPrice={posted_price} futureTime={future_time} "
                f"reason=y_test0_but_bid_down_first"
            )
            _log_price_path_fallback(message)
            return {
                "pnlPerShare": open_mid - posted_price,
                "pnlSettleTime": future_time,
                "postedPrice": posted_price,
                "pricePathClosePrice": posted_price,
                "pnlModel": "price_path_y_test0_favorable_fill",
                "pricePathReason": "bid_down_before_bid_up",
            }
        if next_up_idx >= 0:
            future_row = sec_signal_df.iloc[next_up_idx]
            future_time = pd.Timestamp(future_row.barTime)
            ask_t = future_row.get("closeAsk1Aligned", np.nan)
            if pd.isna(ask_t):
                ask_t = future_row.get("askPrice1Tick", np.nan)
            ask_t = float(ask_t) if not pd.isna(ask_t) else np.nan
            return {
                "pnlPerShare": (open_mid - ask_t) if not pd.isna(ask_t) else old_penalty,
                "pnlSettleTime": future_time,
                "postedPrice": posted_price,
                "pricePathClosePrice": ask_t,
                "pnlModel": "price_path_y_test0_adverse_close",
                "pricePathReason": "bid_up_first",
            }

    eod_time = pd.Timestamp(sec_signal_df.iloc[-1].barTime) if not sec_signal_df.empty else signal_time
    return {
        "pnlPerShare": old_penalty,
        "pnlSettleTime": eod_time,
        "postedPrice": posted_price,
        "pricePathClosePrice": np.nan,
        "pnlModel": "price_path_no_move_spread_penalty",
        "pricePathReason": "no_price_move_before_eod",
    }


def _empty_summary(pool_name: str, trade_date: str, variant_tag: str, params: LowPriceBacktestParams) -> dict[str, Any]:
    return {
        "scope": pool_name,
        "tradeDate": trade_date,
        "variantTag": variant_tag,
        "paramTag": params.param_tag,
        "signalRanks": ",".join(str(rank) for rank in params.signal_ranks),
        "matchWindowSeconds": "unlimited" if params.match_window_seconds is None else str(params.match_window_seconds),
        "spread": params.spread,
        "totalTradeCount": 0,
        "totalExecPnl": 0.0,
        "totalMatchedNotional": 0.0,
        "totalClientAmt": 0.0,
        "matchedClientAmt": 0.0,
        "clientAmtMatchRate": np.nan,
        "notionalWeightedExecRet": np.nan,
        "yTestWinRate": np.nan,
        "maxCapitalUsed": 0.0,
        "p95CapitalUsedByEvent": 0.0,
        "capitalAdjustedReturn": np.nan,
    }


def _build_summary(
    pool_name: str,
    trade_date: str,
    variant_tag: str,
    params: LowPriceBacktestParams,
    trade_df: pd.DataFrame,
    client_order_df: pd.DataFrame,
    capital_row: dict[str, Any] | None,
) -> dict[str, Any]:
    # summary 统一输出 PnL、成交金额占比、按 openNotional 加权收益和资金占用指标。
    row = _empty_summary(pool_name, trade_date, variant_tag, params)
    total_client_amt = float(client_order_df["clientFilledAmt"].sum()) if not client_order_df.empty else 0.0
    row["totalClientAmt"] = total_client_amt
    if trade_df.empty:
        return row

    total_exec_pnl = float(trade_df["execPnl"].sum())
    total_notional = float(trade_df["openNotional"].sum())
    matched_client_amt = float(trade_df["clientFilledAmt"].sum())
    max_capital = float(capital_row.get("maxCapitalUsed", 0.0)) if capital_row else 0.0
    row.update(
        {
            "totalTradeCount": int(len(trade_df)),
            "totalExecPnl": total_exec_pnl,
            "totalMatchedNotional": total_notional,
            "matchedClientAmt": matched_client_amt,
            "clientAmtMatchRate": np.nan if total_client_amt == 0 else matched_client_amt / total_client_amt,
            "notionalWeightedExecRet": np.nan if total_notional == 0 else total_exec_pnl / total_notional,
            "yTestWinRate": float(pd.to_numeric(trade_df["yTest"], errors="coerce").mean()),
            "maxCapitalUsed": max_capital,
            "p95CapitalUsedByEvent": float(capital_row.get("p95CapitalUsedByEvent", 0.0)) if capital_row else 0.0,
            "capitalAdjustedReturn": np.nan if max_capital == 0 else total_exec_pnl / max_capital,
        }
    )
    return row


def _simulate_security_variant(
    sec_signal_df: pd.DataFrame,
    sec_order_df: pd.DataFrame,
    security_code: str,
    pool_name: str,
    trade_date: str,
    variant_tag: str,
    params: LowPriceBacktestParams,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # 单票单 variant 模拟：先拿到已经匹配 signal 的客户单，再按低价股容量和价格 reset 规则生成 trade。
    if {"matchedSignalRowIdx", "openMid"}.issubset(sec_order_df.columns):
        event_rows = [
            row
            for row in sec_order_df.to_dict(orient="records")
            if not bool(row.get("matched", False))
        ]
        matched_df = sec_order_df[sec_order_df["matched"].astype(bool)].copy()
        if matched_df.empty:
            return event_rows, []
    else:
        matched_rows: list[dict[str, Any]] = []
        event_rows: list[dict[str, Any]] = []

        for order in sec_order_df.to_dict(orient="records"):
            client_side = str(order["clientSide"]).upper()
            if client_side not in {"B", "S"}:
                continue

            order_time = pd.Timestamp(order["clientOrderTime"])
            matched_row, matched_idx = _find_latest_eligible_signal_row(
                sec_signal_df=sec_signal_df,
                order_time=order_time,
                client_side=client_side,
                params=params,
            )

            if matched_row is None:
                event_rows.append({**order, "variantTag": variant_tag, "matched": False, "matchStatus": "no_eligible_signal"})
                continue

            matched_rows.append(
                {
                    **order,
                    "variantTag": variant_tag,
                    "matched": True,
                    "matchStatus": "matched_signal",
                    "matchedSignalRowIdx": matched_idx,
                    "matchSignalTime": matched_row.barTime,
                    "matchSignalTimeInt": int(matched_row.signalTime),
                    "bsFlag": str(matched_row.bsFlag),
                    "mergeSignal": float(matched_row.mergeSignal),
                    "yTest": int(matched_row.yTest),
                    "inventorySide": "LONG" if client_side == "S" else "SHORT",
                }
            )

        if not matched_rows:
            return event_rows, []

        matched_df = pd.DataFrame(matched_rows)
    trade_rows: list[dict[str, Any]] = []
    long_qty = 0
    short_qty = 0
    long_active_trade_idxs: list[int] = []
    short_active_trade_idxs: list[int] = []
    prev_ask = np.nan
    prev_bid = np.nan

    orders_by_signal_idx = {
        int(idx): rows.to_dict(orient="records")
        for idx, rows in matched_df.groupby("matchedSignalRowIdx", sort=True)
    }

    def release_active(side: str, close_time: pd.Timestamp, reset_reason: str) -> None:
        # 低价股挂单 PnL 在 y_test 处立即结算，但资金占用要等一档价格变化或 EOD 才释放。
        nonlocal long_qty, short_qty, long_active_trade_idxs, short_active_trade_idxs
        active_idxs = long_active_trade_idxs if side == "LONG" else short_active_trade_idxs
        for trade_idx in active_idxs:
            trade_rows[trade_idx]["closeTime"] = close_time
            trade_rows[trade_idx]["resetReason"] = reset_reason
        if side == "LONG":
            long_qty = 0
            long_active_trade_idxs = []
        else:
            short_qty = 0
            short_active_trade_idxs = []

    for idx, signal_row in sec_signal_df.iterrows():
        ask_price = signal_row.get("askPrice1Tick", np.nan)
        bid_price = signal_row.get("bidPrice1Tick", np.nan)
        # LONG 对应在 ask1 挂卖平仓；ask1 变化表示这一档挂单周期结束，累计开仓量清零。
        if not pd.isna(prev_ask) and not pd.isna(ask_price) and float(ask_price) != float(prev_ask):
            release_active("LONG", pd.Timestamp(signal_row.barTime), "ask_price_changed")
        # SHORT 对应在 bid1 挂买平仓；bid1 变化同样触发 reset。
        if not pd.isna(prev_bid) and not pd.isna(bid_price) and float(bid_price) != float(prev_bid):
            release_active("SHORT", pd.Timestamp(signal_row.barTime), "bid_price_changed")

        cap_qty = _cap_qty(signal_row, variant_tag)
        for order in orders_by_signal_idx.get(idx, []):
            original_qty = int(order["clientQty"])
            side = str(order["inventorySide"])
            current_qty = long_qty if side == "LONG" else short_qty
            available_qty = max(0, cap_qty - current_qty)
            exec_qty = min(original_qty, available_qty)
            reason = "not_clipped" if exec_qty == original_qty else "clipped_by_low_price_cap"
            if exec_qty <= 0:
                event_rows.append({**order, "matched": False, "matchStatus": "low_price_cap_full", "capQty": cap_qty})
                continue
            if pd.isna(order.get("openMid")):
                event_rows.append({**order, "matched": False, "matchStatus": "no_tick_mid", "capQty": cap_qty})
                continue

            open_mid = float(order["openMid"])
            posted_price = order.get("openAskPrice1") if side == "LONG" else order.get("openBidPrice1")
            price_path = _low_price_price_path_pnl(
                sec_signal_df=sec_signal_df,
                signal_idx=int(idx),
                signal_row=signal_row,
                side=side,
                open_mid=open_mid,
                posted_price=posted_price,
                y_test=int(order["yTest"]),
                params=params,
                context={
                    "tradeDate": trade_date,
                    "poolName": pool_name,
                    "securityCode": security_code,
                    "signalTime": int(order["matchSignalTimeInt"]),
                },
            )
            pnl_per_share = float(price_path["pnlPerShare"])
            client_filled_amt = float(order["clientFilledAmt"]) * exec_qty / original_qty if original_qty > 0 else 0.0
            trade_row = {
                "poolName": pool_name,
                "tradeDate": pd.Timestamp(trade_date),
                "securityCode": security_code,
                "variantTag": variant_tag,
                "side": side,
                "strategySource": order["strategySource"],
                "parentOrderId": order["parentOrderId"],
                "clientOrderId": order["clientOrderId"],
                "clientOrderTime": order["clientOrderTime"],
                "clientSide": order["clientSide"],
                "openTime": order["clientOrderTime"],
                "openTickTime": order.get("openTickTime", pd.NaT),
                "closeTime": pd.NaT,
                "pnlSettleTime": price_path["pnlSettleTime"],
                "openSignalTime": int(order["matchSignalTimeInt"]),
                "bsFlag": order["bsFlag"],
                "mergeSignal": float(order["mergeSignal"]),
                "yTest": int(order["yTest"]),
                "clientQtyOriginal": original_qty,
                "clientQty": exec_qty,
                "clientFilledAmt": client_filled_amt,
                "openMid": open_mid,
                "openBidPrice1": order.get("openBidPrice1", np.nan),
                "openAskPrice1": order.get("openAskPrice1", np.nan),
                "spread": params.spread,
                "postedPrice": price_path["postedPrice"],
                "pricePathClosePrice": price_path["pricePathClosePrice"],
                "pnlModel": price_path["pnlModel"],
                "pricePathReason": price_path["pricePathReason"],
                "pnlPerShare": pnl_per_share,
                "execRet": pnl_per_share / open_mid if open_mid else np.nan,
                "execPnl": pnl_per_share * exec_qty,
                "openNotional": open_mid * exec_qty,
                "capQty": cap_qty,
                "capUsedQtyBefore": current_qty,
                "capAvailableQty": available_qty,
                "liquidityClipReason": reason,
                "askPrice1AtSignal": ask_price,
                "bidPrice1AtSignal": bid_price,
                "avgBidVol1Last5": signal_row.get("avgBidVol1Last5", np.nan),
                "avgAskVol1Last5": signal_row.get("avgAskVol1Last5", np.nan),
                "resetReason": "eod",
            }
            trade_rows.append(trade_row)
            trade_idx = len(trade_rows) - 1
            if side == "LONG":
                long_qty += exec_qty
                long_active_trade_idxs.append(trade_idx)
            else:
                short_qty += exec_qty
                short_active_trade_idxs.append(trade_idx)
            event_rows.append({**order, "matched": True, "matchStatus": "matched", "clientQty": exec_qty, "capQty": cap_qty})

        prev_ask = ask_price
        prev_bid = bid_price

    if not sec_signal_df.empty:
        eod_time = pd.Timestamp(sec_signal_df.iloc[-1].barTime)
        release_active("LONG", eod_time, "eod")
        release_active("SHORT", eod_time, "eod")

    return event_rows, trade_rows


def _match_orders_to_signals(
    sec_signal_df: pd.DataFrame,
    sec_order_df: pd.DataFrame,
    variant_tag: str,
    params: LowPriceBacktestParams,
) -> list[dict[str, Any]]:
    # 只看客户单到达前最近一条 signal，再判断它是否符合方向/rank 和 match window。
    # 不能往前翻历史 eligible signal；否则 unlimited 会变成“历史上曾经出现过即可匹配”。
    event_rows: list[dict[str, Any]] = []
    for order in sec_order_df.to_dict(orient="records"):
        client_side = str(order["clientSide"]).upper()
        if client_side not in {"B", "S"}:
            continue

        order_time = pd.Timestamp(order["clientOrderTime"])
        matched_row, matched_idx = _find_latest_eligible_signal_row(
            sec_signal_df=sec_signal_df,
            order_time=order_time,
            client_side=client_side,
            params=params,
        )

        if matched_row is None:
            event_rows.append({**order, "variantTag": variant_tag, "matched": False, "matchStatus": "no_eligible_signal"})
            continue

        event_rows.append(
            {
                **order,
                "variantTag": variant_tag,
                "matched": True,
                "matchStatus": "matched_signal",
                "matchedSignalRowIdx": matched_idx,
                "matchSignalTime": matched_row.barTime,
                "matchSignalTimeInt": int(matched_row.signalTime),
                "bsFlag": str(matched_row.bsFlag),
                "mergeSignal": float(matched_row.mergeSignal),
                "yTest": int(matched_row.yTest),
                "inventorySide": "LONG" if client_side == "S" else "SHORT",
            }
        )
    return event_rows


def run_low_price_prepared_day(
    prepared_inputs: dict[str, object],
    params: LowPriceBacktestParams,
    ddb_config: DdbConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # prepared 入口只做参数相关模拟；数据读取已在 low_price_internalization_data.py 完成。
    trade_date = str(prepared_inputs["tradeDate"])
    pool_name = str(prepared_inputs["poolName"])
    signal_df = prepared_inputs["signalDf"]
    client_order_df = prepared_inputs["clientOrderDf"]

    all_events: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []

    signal_by_security = {
        security_code: sec_df.sort_values(["barTime", "bsFlag"]).reset_index(drop=True)
        for security_code, sec_df in signal_df.groupby("securityCode", sort=True)
    }
    orders_by_security = {
        security_code: sec_df.sort_values("clientOrderTime").reset_index(drop=True)
        for security_code, sec_df in client_order_df.groupby("securityCode", sort=True)
    }
    security_codes = sorted(set(signal_by_security) & set(orders_by_security))

    aligned_price_df = load_aligned_price_df(
        trade_date=trade_date,
        pool_name=pool_name,
        security_codes=security_codes,
        ddb_config=ddb_config,
    )
    aligned_by_security = {
        security_code: sec_df.reset_index(drop=True)
        for security_code, sec_df in aligned_price_df.groupby("securityCode", sort=True)
    } if not aligned_price_df.empty else {}


    # TickMidCache 懒加载单票 tick，避免一次性读全市场 tick；每只票只读一次后在三个 variant 间复用。
    tick_cache = TickMidCache(trade_date=trade_date, ddb_config=ddb_config)
    try:
        for security_code in security_codes:
            tick_df = tick_cache.get(security_code)
            sec_signal_df = signal_by_security[security_code]
            sec_signal_df = _merge_aligned_to_signal(sec_signal_df, aligned_by_security.get(security_code, pd.DataFrame()))
            sec_signal_df = _add_next_price_move_cols(_merge_tick_to_signal(sec_signal_df, tick_df))
            sec_order_df = orders_by_security[security_code]
            for variant_tag in LOW_PRICE_VARIANTS:
                raw_events = _match_orders_to_signals(
                    sec_signal_df=sec_signal_df,
                    sec_order_df=sec_order_df,
                    variant_tag=variant_tag,
                    params=params,
                )
                matched_events = [row for row in raw_events if bool(row.get("matched"))]
                if matched_events:
                    matched_df = _merge_tick_to_orders(pd.DataFrame(matched_events), tick_df)
                    matched_by_id = {
                        (row["variantTag"], row["clientOrderId"], row["clientOrderTime"]): row
                        for row in matched_df.to_dict(orient="records")
                    }
                    enriched_events = [
                        {
                            **row,
                            **{
                                key: value
                                for key, value in matched_by_id.get((row["variantTag"], row["clientOrderId"], row["clientOrderTime"]), {}).items()
                                if key in {"openMid", "openTickTime", "openBidPrice1", "openAskPrice1"}
                            },
                        }
                        for row in raw_events
                    ]
                else:
                    enriched_events = raw_events

                sec_events, sec_trades = _simulate_security_variant(
                    sec_signal_df=sec_signal_df,
                    sec_order_df=pd.DataFrame(enriched_events),
                    security_code=security_code,
                    pool_name=pool_name,
                    trade_date=trade_date,
                    variant_tag=variant_tag,
                    params=params,
                )
                all_events.extend(sec_events)
                all_trades.extend(sec_trades)
    finally:
        tick_cache.close()

    events_df = pd.DataFrame(all_events)
    trades_df = pd.DataFrame(all_trades)
    # 资金占用用 openNotional 的事件流计算；同一笔 trade 不会因为 partial/event 行重复计入。
    capital_rows = capital_metrics_by_variant(
        event_rows=build_capital_event_rows(
            trades_df=pd.DataFrame(),
            position_cap_trades_df=trades_df,
            variant_tags=LOW_PRICE_VARIANTS,
        ),
        base_row={"tradeDate": trade_date},
        variant_tags=LOW_PRICE_VARIANTS,
    )
    capital_by_variant = {row["variantTag"]: row for row in capital_rows}
    summary_rows = [
        _build_summary(
            pool_name=pool_name,
            trade_date=trade_date,
            variant_tag=variant_tag,
            params=params,
            trade_df=trades_df[trades_df["variantTag"] == variant_tag] if not trades_df.empty else pd.DataFrame(),
            client_order_df=client_order_df,
            capital_row=capital_by_variant.get(variant_tag),
        )
        for variant_tag in LOW_PRICE_VARIANTS
    ]
    summary_df = pd.DataFrame(summary_rows)
    return events_df, trades_df, summary_df


def run_low_price_single_day(
    trade_date: str,
    params: LowPriceBacktestParams,
    ims_roots: list[Path | str] | None = None,
    pool_name: str = "hs300",
    mysql_config: MysqlConfig | None = None,
    ddb_config: DdbConfig | None = None,
    signal_table: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 单日调试入口，方便先看某一天/某个 pool 的 order_events、trades、summary 明细。
    roots = ims_roots if ims_roots is not None else get_default_ims_roots(Path.cwd())
    prepared_inputs = load_low_price_day_inputs(
        trade_date=trade_date,
        ims_roots=roots,
        pool_name=pool_name,
        mysql_config=mysql_config,
        signal_table=signal_table,
    )
    if prepared_inputs is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return run_low_price_prepared_day(prepared_inputs=prepared_inputs, params=params, ddb_config=ddb_config)






