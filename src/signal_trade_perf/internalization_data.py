from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql

from .configs import DdbConfig, MysqlConfig, build_signal_table_name
from .io_utils import connect_ddb, fetch_aligned_prices_ddb, fetch_previous_aligned_prices_ddb
from .io_utils import load_signal_day_mysql


def get_default_ims_roots(project_root: Path | str) -> list[Path]:
    # 默认把 ims_backtest_data 下所有策略目录纳入 internalization 候选池。
    base_dir = Path(project_root) / "ims_backtest_data"
    if not base_dir.exists():
        return []
    return sorted(path for path in base_dir.iterdir() if path.is_dir())


def discover_ims_security_codes(trade_date: str, ims_roots: list[Path | str]) -> list[str]:
    # 只扫描目录名，不读取 CSV；用于提前缩小 MySQL signal 查询范围。
    security_codes: set[str] = set()
    for ims_root in [Path(path) for path in ims_roots]:
        date_dir = ims_root / trade_date
        if not date_dir.exists():
            continue
        security_codes.update(path.name for path in date_dir.iterdir() if path.is_dir())
    return sorted(security_codes)


def load_pool_universe_mysql(
    trade_date: str,
    pool_name: str,
    mysql_config: MysqlConfig | None = None,
) -> list[str]:
    # 优先用月度股票池成分表缩小 signal 查询范围，比拿 IMS 全目录 ticker 去查 signal 更精确。
    config = mysql_config or MysqlConfig()
    table_name = f"static_data_price_{pool_name}_history"
    query = f"""
        select distinct ticker
        from {table_name}
        where test_month = %s
    """
    try:
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
                cursor.execute(query, (int(trade_date[:6]),))
                rows = cursor.fetchall()
    except pymysql.MySQLError as exc:
        log_internalization_issue(f"[WARN] {trade_date} {pool_name} failed to load pool universe: {exc}")
        return []

    return sorted(str(row["ticker"]) for row in rows if row.get("ticker"))


def time_int_to_timestamp(trade_date: str, value: int | float | str) -> pd.Timestamp:
    # MySQL / IMS 里大量时间字段都用 HHMMSSmmm 的整数表示，这里统一转成带日期的 Timestamp。
    raw = int(float(value))
    milliseconds = raw % 1000
    raw //= 1000
    seconds = raw % 100
    raw //= 100
    minutes = raw % 100
    hours = raw // 100
    base = pd.Timestamp(f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}")
    return base + pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)


def normalize_price(value: float | int | None) -> float:
    # 不同来源的价格精度不完全一致；这里统一转成实际小数价格。
    if value is None or pd.isna(value):
        return np.nan
    return float(value) / 10000.0


def normalize_static_limit_price(value: float | int | None) -> float:
    # 静态表涨跌停价固定是实际价格 * 10000，这里不做尺度猜测，直接还原为实际价格。
    if value is None or pd.isna(value):
        return np.nan
    return float(value) / 10000.0


def is_a_share(security_code: str) -> bool:
    return str(security_code).endswith((".SH", ".SZ"))


def static_market_suffix(market_type: object) -> str | None:
    # 静态表里 securityCode 不带交易所后缀，需要用 marketType 还原，避免沪深重名代码误匹配。
    if pd.isna(market_type):
        return None
    market_type_int = int(market_type)
    if market_type_int == 101:
        return ".SH"
    if market_type_int == 102:
        return ".SZ"
    return None


def month_static_table_name(trade_date: str) -> str:
    return f"test_static_demo_{trade_date[:6]}"


def log_internalization_issue(message: str, log_path: Path | None = None) -> None:
    # 数据源缺失或权限问题统一写到 internalization 日志，便于长批量回测后排查。
    print(message)
    if log_path is None:
        log_path = Path.cwd() / "results" / "internalization_backtest" / "logs" / "internalization_errors.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {message}\n")


def load_a_share_limit_df(
    trade_date: str,
    security_codes: list[str],
    mysql_config: MysqlConfig | None = None,
) -> pd.DataFrame:
    # A 股涨跌停价来自 dev 库静态表；非 A 股不需要涨跌停逻辑。
    a_share_codes = sorted(code for code in set(security_codes) if is_a_share(code))
    if not a_share_codes:
        return pd.DataFrame(columns=["securityCode", "highLimit", "lowLimit"])

    config = replace(mysql_config or MysqlConfig(), database="dev")
    table_name = month_static_table_name(trade_date)
    query = f"""
        select securityCode, marketType, highLimit, lowLimit
        from {table_name}
        where date = %s
    """
    try:
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
                cursor.execute(query, (int(trade_date),))
                rows = cursor.fetchall()
    except pymysql.MySQLError as exc:
        log_internalization_issue(f"[ERROR] {trade_date} failed to load A-share static limit data: {exc}")
        return pd.DataFrame(columns=["securityCode", "highLimit", "lowLimit"])

    if not rows:
        return pd.DataFrame(columns=["securityCode", "highLimit", "lowLimit"])

    limit_df = pd.DataFrame(rows)
    limit_df["securityCode"] = limit_df["securityCode"].astype(str)
    limit_df["marketSuffix"] = limit_df["marketType"].apply(static_market_suffix)
    limit_df = limit_df[limit_df["marketSuffix"].notna()].copy()
    limit_df["securityCode"] = limit_df["securityCode"] + limit_df["marketSuffix"]
    limit_df = limit_df[limit_df["securityCode"].isin(a_share_codes)].copy()
    limit_df["highLimit"] = limit_df["highLimit"].apply(normalize_static_limit_price)
    limit_df["lowLimit"] = limit_df["lowLimit"].apply(normalize_static_limit_price)
    return limit_df[["securityCode", "highLimit", "lowLimit"]]


def validate_a_share_limit_data(trade_date: str, security_codes: list[str], limit_df: pd.DataFrame) -> bool:
    # 如果当天 A 股静态数据不完整，直接跳过当天，避免涨跌停风控在缺数据时误成交。
    a_share_codes = {code for code in set(security_codes) if is_a_share(code)}
    if not a_share_codes:
        return True
    if limit_df.empty:
        log_internalization_issue(f"[ERROR] {trade_date} missing A-share static limit data for all securities.")
        return False
    available_codes = set(limit_df["securityCode"].astype(str))
    missing_codes = sorted(a_share_codes - available_codes)
    if missing_codes:
        preview = ",".join(missing_codes[:20])
        log_internalization_issue(
            f"[ERROR] {trade_date} missing A-share static limit data for {len(missing_codes)} securities: {preview}"
        )
        return False
    return True


def build_prices_table_name(pool_name: str) -> str:
    return f"Prices_{pool_name}"


def load_signal_day_for_internalization(
    trade_date: str,
    pool_name: str,
    mysql_config: MysqlConfig | None = None,
    security_codes: list[str] | set[str] | None = None,
    signal_table_name: str | None = None,
) -> pd.DataFrame:
    # internalization 链路直接使用 MySQL 原始 signal 时间，不再做 15s 聚合。
    mysql_config = mysql_config or MysqlConfig()
    table_name = signal_table_name or build_signal_table_name(pool_name)
    signal_df = load_signal_day_mysql(trade_date, table_name, mysql_config, security_codes=security_codes)
    if signal_df.empty:
        return signal_df

    signal_df = signal_df.copy()
    signal_df["barTime"] = signal_df["signalTime"].apply(lambda value: time_int_to_timestamp(trade_date, value))
    return signal_df.sort_values(["securityCode", "barTime"]).reset_index(drop=True)


def load_aligned_price_df(
    trade_date: str,
    pool_name: str,
    security_codes: list[str],
    ddb_config: DdbConfig | None = None,
    chunk_size: int = 40,
) -> pd.DataFrame:
    # 平仓价读取 DDB_Returns 的 Prices_{pool}，这些时间戳和 signal 链是一致的。
    security_codes = sorted(set(security_codes))
    if not security_codes:
        return pd.DataFrame(columns=["barTime", "securityCode", "closeMidAligned", "closeAsk1Aligned", "closeBid1Aligned"])

    ddb_config = ddb_config or DdbConfig()
    session = connect_ddb(ddb_config)
    try:
        price_df = fetch_aligned_prices_ddb(
            session=session,
            trade_date=trade_date,
            table_name=build_prices_table_name(pool_name),
            ticker_list=security_codes,
            chunk_size=chunk_size,
        )
    finally:
        session.close()

    if price_df.empty:
        return pd.DataFrame(columns=["barTime", "securityCode", "closeMidAligned", "closeAsk1Aligned", "closeBid1Aligned"])

    return price_df.sort_values(["securityCode", "barTime"]).reset_index(drop=True)


def load_previous_aligned_price_df(
    trade_date: str,
    pool_name: str,
    security_codes: list[str],
    ddb_config: DdbConfig | None = None,
    chunk_size: int = 40,
) -> pd.DataFrame:
    # 前一交易日价格只用于计算 prevDayVol，直接从 DDB_Returns 找最近历史日期并读取价格。
    security_codes = sorted(set(security_codes))
    if not security_codes:
        return pd.DataFrame(columns=["barTime", "securityCode", "closeMidAligned", "closeAsk1Aligned", "closeBid1Aligned"])

    ddb_config = ddb_config or DdbConfig()
    session = connect_ddb(ddb_config)
    try:
        price_df = fetch_previous_aligned_prices_ddb(
            session=session,
            trade_date=trade_date,
            table_name=build_prices_table_name(pool_name),
            ticker_list=security_codes,
            chunk_size=chunk_size,
        )
    finally:
        session.close()

    if price_df.empty:
        return pd.DataFrame(columns=["barTime", "securityCode", "closeMidAligned", "closeAsk1Aligned", "closeBid1Aligned"])

    return price_df.sort_values(["securityCode", "barTime"]).reset_index(drop=True)


def load_ims_child_orders(
    trade_date: str,
    ims_roots: list[Path | str],
    security_codes: list[str] | set[str] | None = None,
) -> pd.DataFrame:
    # 每个 *_statistics.csv 对应一个母单，文件里是子单级别记录。
    # 如果上游已经知道当天只需要哪些股票，就在读文件阶段先过滤，避免先全量加载再二次筛选。
    order_rows: list[dict[str, object]] = []
    security_code_set = set(security_codes) if security_codes is not None else None

    for ims_root in [Path(path) for path in ims_roots]:
        date_dir = ims_root / trade_date
        if not date_dir.exists():
            continue

        for ticker_dir in sorted(path for path in date_dir.iterdir() if path.is_dir()):
            if security_code_set is not None and ticker_dir.name not in security_code_set:
                continue
            for csv_path in sorted(ticker_dir.glob("*_statistics.csv")):
                child_df = pd.read_csv(csv_path)
                if child_df.empty:
                    continue

                price_col = "price" if "price" in child_df.columns else "orderPrice"
                qty_col = "filledQty" if "filledQty" in child_df.columns else "orderQty"

                for row in child_df.to_dict(orient="records"):
                    side = str(row.get("side", "")).upper()
                    client_qty = row.get(qty_col)
                    if side not in {"B", "S"} or pd.isna(client_qty) or float(client_qty) <= 0:
                        continue

                    time_value = row.get("time", row.get("order_time_seconds"))
                    if pd.isna(time_value):
                        continue

                    order_rows.append(
                        {
                            "strategySource": ims_root.name,
                            "tradeDate": pd.Timestamp(trade_date),
                            "securityCode": ticker_dir.name,
                            "parentOrderId": csv_path.stem.removesuffix("_statistics"),
                            "clientOrderId": row.get("orderId", csv_path.stem),
                            "clientOrderTimeInt": int(float(time_value)),
                            "clientOrderTime": time_int_to_timestamp(trade_date, time_value),
                            "clientSide": side,
                            "clientQty": int(float(client_qty)),
                            "clientOrderPrice": normalize_price(row.get("orderPrice")),
                            "clientExecPrice": normalize_price(row.get(price_col)),
                            "clientFilledAmt": float(row.get("filledAmt", np.nan)) if not pd.isna(row.get("filledAmt", np.nan)) else np.nan,
                            "clientFillTime": (
                                time_int_to_timestamp(trade_date, row["fillTime"])
                                if "fillTime" in row and not pd.isna(row["fillTime"])
                                else pd.NaT
                            ),
                        }
                    )

    if not order_rows:
        return pd.DataFrame(
            columns=[
                "strategySource",
                "tradeDate",
                "securityCode",
                "parentOrderId",
                "clientOrderId",
                "clientOrderTimeInt",
                "clientOrderTime",
                "clientSide",
                "clientQty",
                "clientOrderPrice",
                "clientExecPrice",
                "clientFilledAmt",
                "clientFillTime",
            ]
        )

    return pd.DataFrame(order_rows).sort_values(
        ["securityCode", "clientOrderTime", "strategySource", "parentOrderId", "clientOrderId"]
    ).reset_index(drop=True)
