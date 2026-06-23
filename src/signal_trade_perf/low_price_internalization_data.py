from __future__ import annotations

from pathlib import Path

import pandas as pd
import pymysql

from .configs import MysqlConfig
from .internalization_data import (
    discover_ims_security_codes,
    load_ims_child_orders,
    load_pool_universe_mysql,
    time_int_to_timestamp,
)


# 低价股“挂单等成交”策略的默认 signal 表。
# 这张表和高价股开平仓 signal 不同：
# - bs_flag=b/s 表示挂买/挂卖方向；
# - merge_signal=1/2 或 -1/-2 表示成交概率 rank；
# - y_test=1/0 是事后真实是否成交，用于直接估算挂单 PnL。
LOW_PRICE_SIGNAL_TABLE = "signal_hs300_low_price_70_pct"


def build_low_price_signal_table_name(pool_name: str) -> str:
    # 低价股挂单 signal 表按股票池命名，例如 hs300 -> signal_hs300_low_price_70_pct。
    return f"signal_{pool_name}_low_price_70_pct"


def load_low_price_signal_day_mysql(
    trade_date: str,
    table_name: str = LOW_PRICE_SIGNAL_TABLE,
    mysql_config: MysqlConfig | None = None,
    security_codes: list[str] | set[str] | None = None,
) -> pd.DataFrame:
    # 只读取低价股挂单策略需要的字段，并统一改成模拟层使用的列名。
    # 这里按 security_codes 预过滤，是为了避免全表扫 signal，尤其多日期扫参时会明显省时间。
    config = mysql_config or MysqlConfig()
    security_code_list = sorted(set(security_codes)) if security_codes is not None else []
    if security_codes is not None and not security_code_list:
        return pd.DataFrame(columns=["securityCode", "tradeDate", "signalTime", "barTime", "bsFlag", "mergeSignal", "yTest"])

    ticker_filter_sql = ""
    params: list[object] = [int(trade_date)]
    if security_code_list:
        placeholders = ",".join(["%s"] * len(security_code_list))
        ticker_filter_sql = f" and ticker in ({placeholders})"
        params.extend(security_code_list)

    query = f"""
        select ticker, date, time, bs_flag, merge_signal, y_test
        from {table_name}
        where date = %s{ticker_filter_sql}
    """

    with pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database,
        charset=config.charset,
        read_timeout=config.read_timeout,
        write_timeout=config.write_timeout,
        connect_timeout=config.connect_timeout,
        cursorclass=pymysql.cursors.DictCursor,
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

    if not rows:
        return pd.DataFrame(columns=["securityCode", "tradeDate", "signalTime", "barTime", "bsFlag", "mergeSignal", "yTest"])

    # 标准化列名和类型：后续撮合只认 securityCode/barTime/bsFlag/mergeSignal/yTest。
    signal_df = pd.DataFrame(rows).rename(
        columns={
            "ticker": "securityCode",
            "time": "signalTime",
            "bs_flag": "bsFlag",
            "merge_signal": "mergeSignal",
            "y_test": "yTest",
        }
    )
    signal_df["securityCode"] = signal_df["securityCode"].astype(str)
    signal_df["tradeDate"] = pd.Timestamp(trade_date)
    signal_df["barTime"] = signal_df["signalTime"].apply(lambda value: time_int_to_timestamp(trade_date, value))
    signal_df["bsFlag"] = signal_df["bsFlag"].astype(str).str.lower()
    signal_df["mergeSignal"] = pd.to_numeric(signal_df["mergeSignal"], errors="coerce")
    signal_df["yTest"] = pd.to_numeric(signal_df["yTest"], errors="coerce").fillna(0).astype(int)
    return signal_df[["securityCode", "tradeDate", "signalTime", "barTime", "bsFlag", "mergeSignal", "yTest"]]


def load_low_price_day_inputs(
    trade_date: str,
    ims_roots: list[Path | str],
    pool_name: str = "hs300",
    mysql_config: MysqlConfig | None = None,
    signal_table: str | None = None,
) -> dict[str, object] | None:
    # 单日 prepared inputs 只放“和参数无关”的数据，方便多参数扫参时按 date/pool 缓存。
    # 这里先用 pool universe ∩ IMS 目录 ticker 缩小 signal 查询范围。
    mysql_config = mysql_config or MysqlConfig()
    pool_universe_codes = load_pool_universe_mysql(trade_date=trade_date, pool_name=pool_name, mysql_config=mysql_config)
    ims_security_codes = discover_ims_security_codes(trade_date=trade_date, ims_roots=ims_roots)
    security_codes = sorted(set(pool_universe_codes) & set(ims_security_codes))
    if not security_codes:
        return None

    signal_df = load_low_price_signal_day_mysql(
        trade_date=trade_date,
        table_name=signal_table or build_low_price_signal_table_name(pool_name),
        mysql_config=mysql_config,
        security_codes=security_codes,
    )
    if signal_df.empty:
        return None

    # 客户子单仍复用高价股 IMS 读取逻辑，保证客户方向、数量、到达时间口径一致。
    client_order_df = load_ims_child_orders(
        trade_date=trade_date,
        ims_roots=ims_roots,
        security_codes=sorted(signal_df["securityCode"].unique()),
    )
    if client_order_df.empty:
        return None

    return {
        "tradeDate": trade_date,
        "poolName": pool_name,
        "signalDf": signal_df,
        "clientOrderDf": client_order_df,
    }

