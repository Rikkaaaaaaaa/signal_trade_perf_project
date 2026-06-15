from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time

import dolphindb as ddb
import pandas as pd
import pymysql

from .configs import DdbConfig, MysqlConfig, format_trade_date


def mkdir_with_retry(path: Path, retries: int = 5, sleep_seconds: float = 0.5) -> None:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            path.mkdir(parents=True, exist_ok=True)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(sleep_seconds)
    if last_error is not None:
        raise last_error


def dataframe_to_csv_with_retry(
    df: pd.DataFrame,
    path: Path,
    retries: int = 5,
    sleep_seconds: float = 0.5,
    **kwargs,
) -> None:
    mkdir_with_retry(path.parent, retries=retries, sleep_seconds=sleep_seconds)
    last_error: Exception | None = None
    for _ in range(retries):
        tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
        try:
            df.to_csv(tmp_path, **kwargs)
            os.replace(tmp_path, path)
            return
        except PermissionError as exc:
            last_error = exc
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            time.sleep(sleep_seconds)
    if last_error is None:
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_pickle(tmp_path)
        child_code = """
from pathlib import Path
import pandas as pd
import sys

pickle_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
tmp_csv_path = csv_path.with_name(f"{csv_path.stem}.tmp{csv_path.suffix}")
df = pd.read_pickle(pickle_path)
df.to_csv(tmp_csv_path, index=False)
import os
os.replace(tmp_csv_path, csv_path)
"""
        result = subprocess.run(
            [sys.executable, "-c", child_code, str(tmp_path), str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise PermissionError(
                f"Fallback CSV writer failed for {path}: {result.stderr.strip() or result.stdout.strip()}"
            ) from last_error
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def dataframe_to_pickle_with_retry(
    df: pd.DataFrame,
    path: Path,
    compression: str = "gzip",
    retries: int = 5,
    sleep_seconds: float = 0.5,
) -> None:
    mkdir_with_retry(path.parent, retries=retries, sleep_seconds=sleep_seconds)
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            df.to_pickle(path, compression=compression)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(sleep_seconds)
    if last_error is not None:
        raise last_error


def connect_ddb(ddb_config: DdbConfig) -> ddb.session:
    session = ddb.session()
    session.connect(ddb_config.host, ddb_config.port, ddb_config.user, ddb_config.password)
    return session


def load_signal_dates_mysql(table_name: str, mysql_config: MysqlConfig) -> list[str]:
    query = f"select distinct date from {table_name} order by date"
    with pymysql.connect(
        host=mysql_config.host,
        port=mysql_config.port,
        user=mysql_config.user,
        password=mysql_config.password,
        database=mysql_config.database,
        charset=mysql_config.charset,
        read_timeout=mysql_config.read_timeout,
        write_timeout=mysql_config.write_timeout,
        connect_timeout=mysql_config.connect_timeout,
        cursorclass=pymysql.cursors.DictCursor,
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
    return [str(row["date"]) for row in rows]


def load_signal_day_mysql(
    signal_date: str,
    table_name: str,
    mysql_config: MysqlConfig,
    security_codes: list[str] | set[str] | None = None,
) -> pd.DataFrame:
    # 直接从 MySQL 取原始 signal 序列，午休时间段过滤掉。
    security_code_list = sorted(set(security_codes)) if security_codes is not None else []
    if security_codes is not None and not security_code_list:
        return pd.DataFrame(columns=["securityCode", "tradeDate", "signalTime", "merge_signal"])

    ticker_filter_sql = ""
    params: list[object] = [int(signal_date)]
    if security_code_list:
        placeholders = ",".join(["%s"] * len(security_code_list))
        ticker_filter_sql = f" and ticker in ({placeholders})"
        params.extend(security_code_list)

    query = f"""
        select ticker, date, time, merge_signal
        from {table_name}
        where date = %s{ticker_filter_sql}
    """

    with pymysql.connect(
        host=mysql_config.host,
        port=mysql_config.port,
        user=mysql_config.user,
        password=mysql_config.password,
        database=mysql_config.database,
        charset=mysql_config.charset,
        read_timeout=mysql_config.read_timeout,
        write_timeout=mysql_config.write_timeout,
        connect_timeout=mysql_config.connect_timeout,
        cursorclass=pymysql.cursors.DictCursor,
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

    if not rows:
        return pd.DataFrame(columns=["securityCode", "tradeDate", "signalTime", "merge_signal"])

    signal_df = pd.DataFrame(rows)
    signal_df = signal_df.rename(columns={"ticker": "securityCode", "time": "signalTime"})
    signal_df["securityCode"] = signal_df["securityCode"].astype(str)
    signal_df["tradeDate"] = pd.Timestamp(signal_date)
    return signal_df[["securityCode", "tradeDate", "signalTime", "merge_signal"]]


def load_prev_signal_date_mysql(signal_date: str, table_name: str, mysql_config: MysqlConfig) -> str | None:
    # internalization 只需要“上一交易日”这一件事，不需要把所有日期都扫出来。
    query = f"""
        select max(date) as prev_date
        from {table_name}
        where date < %s
    """

    with pymysql.connect(
        host=mysql_config.host,
        port=mysql_config.port,
        user=mysql_config.user,
        password=mysql_config.password,
        database=mysql_config.database,
        charset=mysql_config.charset,
        read_timeout=mysql_config.read_timeout,
        write_timeout=mysql_config.write_timeout,
        connect_timeout=mysql_config.connect_timeout,
        cursorclass=pymysql.cursors.DictCursor,
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (int(signal_date),))
            row = cursor.fetchone()

    if not row or row["prev_date"] is None:
        return None
    return str(row["prev_date"])


def fetch_quote_15s_ddb(
    session: ddb.session,
    trade_date: str,
    ticker_list: list[str],
    chunk_size: int = 40,
) -> pd.DataFrame:
    if not ticker_list:
        return pd.DataFrame(columns=["securityCode", "barTime", "signalTime", "bid1_15s", "ask1_15s", "midPrice15s"])

    trade_dt = format_trade_date(trade_date)
    script = f"""
obs = select string(ticker) as securityCode,
        time,
        iif(bidPrice[0].isNull() || askPrice[0].isNull() || bidPrice[0] == 0 || askPrice[0] == 0, NULL, round(bidPrice[0], 4)) as bid1,
        iif(bidPrice[0].isNull() || askPrice[0].isNull() || bidPrice[0] == 0 || askPrice[0] == 0, NULL, round(askPrice[0], 4)) as ask1,
        iif(bidPrice[0].isNull() || askPrice[0].isNull() || bidPrice[0] == 0 || askPrice[0] == 0, NULL, round((bidPrice[0] + askPrice[0]) \\ 2, 4)) as midPrice
    from loadTable("dfs://hdb", "tick")
    where date = {trade_dt}, ticker in symbol(pyTickers)

if(size(obs) == 0){{
    return table(1:0, `securityCode`barTime`signalTime`bid1_15s`ask1_15s`midPrice15s, [STRING, TIMESTAMP, INT, DOUBLE, DOUBLE, DOUBLE])
}}

temp = select last(bid1) as bid1_15s,
        last(ask1) as ask1_15s,
        last(midPrice) as mid_15s
    from obs
    group by securityCode, interval(time, 15s, NULL, closed='right', label='right')

temp = select * from temp
    where second(interval_time) >= 09:30:00
        and (second(interval_time) < 11:30:00 or second(interval_time) >= 13:00:00)
        and second(interval_time) < 14:57:00

select securityCode,
        concatDateTime({trade_dt}, interval_time) as barTime,
        int(string(interval_time).strReplace(":", "").strReplace(".", "")) as signalTime,
        round(bid1_15s, 4) as bid1_15s,
        round(ask1_15s, 4) as ask1_15s,
        round(mid_15s, 4) as midPrice15s
    from temp
"""

    frames: list[pd.DataFrame] = []
    for start_idx in range(0, len(ticker_list), chunk_size):
        chunk = ticker_list[start_idx : start_idx + chunk_size]
        session.upload({"pyTickers": chunk})
        chunk_df = session.run(script)
        if isinstance(chunk_df, pd.DataFrame) and not chunk_df.empty:
            chunk_df["securityCode"] = chunk_df["securityCode"].astype(str)
            frames.append(chunk_df)

    if not frames:
        return pd.DataFrame(columns=["securityCode", "barTime", "signalTime", "bid1_15s", "ask1_15s", "midPrice15s"])

    return pd.concat(frames, ignore_index=True).sort_values(["securityCode", "barTime"]).reset_index(drop=True)


def fetch_tick_mid_ddb(
    session: ddb.session,
    trade_date: str,
    ticker_list: list[str],
    chunk_size: int = 40,
) -> pd.DataFrame:
    # 开仓端需要用到 tick 级别的 mid，同时为了做盘口量约束，
    # 这里一并取买一/卖一量。
    if not ticker_list:
        return pd.DataFrame(columns=["securityCode", "tickTime", "bidPrice1Tick", "askPrice1Tick", "midPriceTick", "bidVol1Tick", "askVol1Tick"])

    trade_dt = format_trade_date(trade_date)
    script = f"""
obs = select string(ticker) as securityCode,
        concatDateTime({trade_dt}, time) as tickTime,
        iif(bidPrice[0].isNull() || bidPrice[0] == 0, NULL, round(bidPrice[0], 4)) as bidPrice1Tick,
        iif(askPrice[0].isNull() || askPrice[0] == 0, NULL, round(askPrice[0], 4)) as askPrice1Tick,
        iif(
            bidPrice[0].isNull() || askPrice[0].isNull() || bidPrice[0] == 0 || askPrice[0] == 0,
            NULL,
            round((bidPrice[0] + askPrice[0]) \\ 2, 4)
        ) as midPriceTick,
        iif(bidVolume[0].isNull(), NULL, int(bidVolume[0])) as bidVol1Tick,
        iif(askVolume[0].isNull(), NULL, int(askVolume[0])) as askVol1Tick
    from loadTable("dfs://hdb", "tick")
    where date = {trade_dt}, ticker in symbol(pyTickers)

select securityCode, tickTime, bidPrice1Tick, askPrice1Tick, midPriceTick, bidVol1Tick, askVol1Tick
    from obs
"""

    frames: list[pd.DataFrame] = []
    for start_idx in range(0, len(ticker_list), chunk_size):
        chunk = ticker_list[start_idx : start_idx + chunk_size]
        session.upload({"pyTickers": chunk})
        chunk_df = session.run(script)
        if isinstance(chunk_df, pd.DataFrame) and not chunk_df.empty:
            chunk_df["securityCode"] = chunk_df["securityCode"].astype(str)
            frames.append(chunk_df)

    if not frames:
        return pd.DataFrame(columns=["securityCode", "tickTime", "bidPrice1Tick", "askPrice1Tick", "midPriceTick", "bidVol1Tick", "askVol1Tick"])

    return pd.concat(frames, ignore_index=True).sort_values(["securityCode", "tickTime"]).reset_index(drop=True)


def fetch_aligned_prices_ddb(
    session: ddb.session,
    trade_date: str,
    table_name: str,
    ticker_list: list[str],
    chunk_size: int = 40,
) -> pd.DataFrame:
    # 平仓端读取和 signal 时间严格对齐的 Returns 价格。
    if not ticker_list:
        return pd.DataFrame(columns=["securityCode", "barTime", "closeMidAligned", "closeAsk1Aligned", "closeBid1Aligned"])

    trade_dt = format_trade_date(trade_date)
    script = f"""
obs = select time as barTime,
        string(securityCode) as securityCode,
        round(midPrice / 10000.0, 4) as closeMidAligned,
        round(askPrice1 / 10000.0, 4) as closeAsk1Aligned,
        round(bidPrice1 / 10000.0, 4) as closeBid1Aligned
    from loadTable("dfs://DDB_Returns", "{table_name}")
    where date(time) = {trade_dt}, securityCode in symbol(pyTickers)

select barTime, securityCode, closeMidAligned, closeAsk1Aligned, closeBid1Aligned
    from obs
"""

    frames: list[pd.DataFrame] = []
    for start_idx in range(0, len(ticker_list), chunk_size):
        chunk = ticker_list[start_idx : start_idx + chunk_size]
        session.upload({"pyTickers": chunk})
        chunk_df = session.run(script)
        if isinstance(chunk_df, pd.DataFrame) and not chunk_df.empty:
            chunk_df["securityCode"] = chunk_df["securityCode"].astype(str)
            frames.append(chunk_df)

    if not frames:
        return pd.DataFrame(columns=["securityCode", "barTime", "closeMidAligned", "closeAsk1Aligned", "closeBid1Aligned"])

    return pd.concat(frames, ignore_index=True).sort_values(["securityCode", "barTime"]).reset_index(drop=True)


def fetch_previous_aligned_prices_ddb(
    session: ddb.session,
    trade_date: str,
    table_name: str,
    ticker_list: list[str],
    chunk_size: int = 40,
) -> pd.DataFrame:
    # 前一日波动率只依赖 DDB_Returns 里的价格序列，直接从同一张 Prices 表找最近历史日期。
    if not ticker_list:
        return pd.DataFrame(columns=["securityCode", "barTime", "closeMidAligned", "closeAsk1Aligned", "closeBid1Aligned"])

    trade_dt = format_trade_date(trade_date)
    script = f"""
select max(date(time)) as prevTradeDate
    from loadTable("dfs://DDB_Returns", "{table_name}")
    where date(time) < {trade_dt}
"""
    prev_date_df = session.run(script)
    if not isinstance(prev_date_df, pd.DataFrame) or prev_date_df.empty or pd.isna(prev_date_df["prevTradeDate"].iloc[0]):
        return pd.DataFrame(columns=["securityCode", "barTime", "closeMidAligned", "closeAsk1Aligned", "closeBid1Aligned"])

    prev_trade_date = pd.Timestamp(prev_date_df["prevTradeDate"].iloc[0]).strftime("%Y%m%d")
    return fetch_aligned_prices_ddb(
        session=session,
        trade_date=prev_trade_date,
        table_name=table_name,
        ticker_list=ticker_list,
        chunk_size=chunk_size,
    )
