from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MysqlConfig:
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "quantStrat"
    password: str = "eqalgo_2024"
    database: str = "strategy"
    charset: str = "utf8mb4"
    read_timeout: int = 600
    write_timeout: int = 600
    connect_timeout: int = 10


@dataclass(frozen=True)
class DdbConfig:
    host: str = "localhost"
    port: int = 8902
    user: str = "admin"
    password: str = "123456"


def build_signal_table_name(pool_name: str) -> str:
    return f"ensemble_signal_{pool_name}_highprice_lgbm_batch3_pct2_mix_am_pm"


def get_pool_name(table_name: str) -> str:
    return table_name.removeprefix("ensemble_signal_").removesuffix("_highprice_lgbm_batch3_pct2_mix_am_pm")


def format_trade_date(date_yyyymmdd: str) -> str:
    return f"{date_yyyymmdd[:4]}.{date_yyyymmdd[4:6]}.{date_yyyymmdd[6:8]}"
