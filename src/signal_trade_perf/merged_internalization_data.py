from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from .configs import (
    DdbConfig,
    MysqlConfig,
    build_high_price_fill_rate_signal_table_name,
    build_signal_table_name,
)
from .internalization import load_internalization_day_inputs
from .low_price_internalization_data import load_low_price_signal_day_mysql


def load_merged_day_inputs(
    trade_date: str,
    pool_name: str,
    ims_roots: list[Path | str],
    mysql_config: MysqlConfig | None = None,
    ddb_config: DdbConfig | None = None,
    prediction_signal_table_name: str | None = None,
    fill_rate_signal_table_name: str | None = None,
    profile: bool = False,
) -> dict[str, object] | None:
    mysql_config = mysql_config or MysqlConfig()
    prediction_table_name = prediction_signal_table_name or build_signal_table_name(pool_name)
    fill_table_name = fill_rate_signal_table_name or build_high_price_fill_rate_signal_table_name(pool_name)

    def record_stage(stage_name: str, start_time: float, **extra: object) -> None:
        if not profile:
            return
        extra_text = " ".join(f"{key}={value}" for key, value in extra.items())
        suffix = f" {extra_text}" if extra_text else ""
        print(f"[profile] {pool_name} {stage_name} seconds={perf_counter() - start_time:.2f}{suffix}")

    stage_start = perf_counter()
    prediction_inputs = load_internalization_day_inputs(
        trade_date=trade_date,
        pool_name=pool_name,
        ims_roots=ims_roots,
        mysql_config=mysql_config,
        ddb_config=ddb_config,
        profile=profile,
        signal_table_name=prediction_table_name,
    )
    record_stage("load_prediction_inputs", stage_start)
    if prediction_inputs is None:
        return None

    prediction_signal_df = prediction_inputs["signalDf"]
    security_codes = sorted(prediction_signal_df["securityCode"].astype(str).unique())
    stage_start = perf_counter()
    fill_rate_signal_df = load_low_price_signal_day_mysql(
        trade_date=trade_date,
        table_name=fill_table_name,
        mysql_config=mysql_config,
        security_codes=security_codes,
    )
    record_stage("load_fill_rate_signal", stage_start, rowCount=len(fill_rate_signal_df), securityCount=len(security_codes))

    return {
        "tradeDate": trade_date,
        "poolName": pool_name,
        "predictionSignalTableName": prediction_table_name,
        "fillRateSignalTableName": fill_table_name,
        "predictionSignalDf": prediction_signal_df,
        "fillRateSignalDf": fill_rate_signal_df,
        "metaDf": prediction_inputs["metaDf"],
        "clientOrderDf": prediction_inputs["clientOrderDf"],
        "closePriceMap": prediction_inputs["closePriceMap"],
        "limitMap": prediction_inputs["limitMap"],
    }
