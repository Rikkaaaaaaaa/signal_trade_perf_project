from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from .analytics import (
    build_security_meta,
    calc_prev_day_vol,
    calc_vol_cutoffs,
    finalize_price_bucket_summary,
    finalize_vol_bucket_summary,
    get_default_price_bin_edges,
    summarize_price_bucket_contrib,
    summarize_vol_bucket_contrib,
)
from .configs import DdbConfig, MysqlConfig, get_pool_name
from .core import BacktestParams, aggregate_param_summary, simulate_signal_day
from .io_utils import (
    connect_ddb,
    dataframe_to_pickle_with_retry,
    fetch_quote_15s_ddb,
    load_signal_day_mysql,
    mkdir_with_retry,
)


class SourceBacktestRunner:
    def __init__(
        self,
        mysql_config: MysqlConfig | None = None,
        ddb_config: DdbConfig | None = None,
        quote_chunk_size: int = 40,
        price_bin_edges: list[float] | None = None,
        day_cache_dir: Path | str | None = None,
    ):
        self.mysql_config = mysql_config or MysqlConfig()
        self.ddb_config = ddb_config or DdbConfig()
        self.quote_chunk_size = quote_chunk_size
        self.price_bin_edges = price_bin_edges or get_default_price_bin_edges()
        self.day_cache_dir = Path(day_cache_dir) if day_cache_dir is not None else None
        self.quote_cache: dict[tuple[str, tuple[str, ...]], pd.DataFrame] = {}
        self.ddb_session = connect_ddb(self.ddb_config)

    def close(self) -> None:
        self.ddb_session.close()

    def _get_day_cache_paths(self, pool_name: str, trade_date: str) -> tuple[Path, Path]:
        if self.day_cache_dir is None:
            raise ValueError("day_cache_dir is not configured")
        pool_dir = self.day_cache_dir / pool_name
        mkdir_with_retry(pool_dir)
        return pool_dir / f"{trade_date}_signal_quote.pkl.gz", pool_dir / f"{trade_date}_meta.pkl.gz"

    def load_prepared_day_cache(self, pool_name: str, trade_date: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        if self.day_cache_dir is None:
            return None
        signal_path, meta_path = self._get_day_cache_paths(pool_name, trade_date)
        if not signal_path.exists() or not meta_path.exists():
            return None
        return pd.read_pickle(signal_path), pd.read_pickle(meta_path)

    def save_prepared_day_cache(
        self,
        pool_name: str,
        trade_date: str,
        signal_quote_df: pd.DataFrame,
        meta_df: pd.DataFrame,
    ) -> None:
        if self.day_cache_dir is None:
            return
        signal_path, meta_path = self._get_day_cache_paths(pool_name, trade_date)
        dataframe_to_pickle_with_retry(signal_quote_df, signal_path, compression="gzip")
        dataframe_to_pickle_with_retry(meta_df, meta_path, compression="gzip")

    def get_quote_day(self, trade_date: str, ticker_list: list[str]) -> pd.DataFrame:
        cache_key = (trade_date, tuple(ticker_list))
        if cache_key not in self.quote_cache:
            self.quote_cache[cache_key] = fetch_quote_15s_ddb(
                self.ddb_session,
                trade_date,
                ticker_list,
                chunk_size=self.quote_chunk_size,
            )
        return self.quote_cache[cache_key].copy()

    def prepare_signal_mid_day(
        self,
        trade_date: str,
        prev_trade_date: str | None,
        table_name: str,
        force_rebuild: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
        timings: dict[str, float | int] = {}
        pool_name = get_pool_name(table_name)

        if not force_rebuild:
            cached = self.load_prepared_day_cache(pool_name, trade_date)
            if cached is not None:
                signal_quote_df, meta_df = cached
                timings["signalRowCount"] = int(len(signal_quote_df))
                timings["metaRowCount"] = int(len(meta_df))
                return signal_quote_df, meta_df, timings

        t0 = perf_counter()
        signal_df = load_signal_day_mysql(trade_date, table_name, self.mysql_config)
        timings["loadSignalSeconds"] = perf_counter() - t0
        if signal_df.empty:
            timings["signalRowCount"] = 0
            timings["metaRowCount"] = 0
            return signal_df, pd.DataFrame(), timings

        ticker_list = sorted(signal_df["securityCode"].unique().tolist())

        t0 = perf_counter()
        quote_df = self.get_quote_day(trade_date, ticker_list)
        timings["loadQuoteSeconds"] = perf_counter() - t0
        if quote_df.empty:
            timings["signalRowCount"] = 0
            timings["metaRowCount"] = 0
            return pd.DataFrame(), pd.DataFrame(), timings

        t0 = perf_counter()
        signal_quote_df = signal_df.merge(quote_df, on=["securityCode", "signalTime"], how="left")
        signal_quote_df = signal_quote_df[signal_quote_df["midPrice15s"].notna()].sort_values(["securityCode", "barTime"]).reset_index(drop=True)
        timings["mergeSignalQuoteSeconds"] = perf_counter() - t0
        if signal_quote_df.empty:
            timings["signalRowCount"] = 0
            timings["metaRowCount"] = 0
            return signal_quote_df, pd.DataFrame(), timings

        t0 = perf_counter()
        if prev_trade_date is None:
            prev_vol_df = pd.DataFrame(columns=["securityCode", "prevDayVol"])
        else:
            prev_quote_df = self.get_quote_day(prev_trade_date, ticker_list)
            prev_vol_df = calc_prev_day_vol(prev_quote_df)
        timings["calcPrevVolSeconds"] = perf_counter() - t0

        t0 = perf_counter()
        meta_df = build_security_meta(signal_quote_df, prev_vol_df, pool_name, trade_date, self.price_bin_edges)
        timings["buildMetaSeconds"] = perf_counter() - t0

        t0 = perf_counter()
        self.save_prepared_day_cache(pool_name, trade_date, signal_quote_df, meta_df)
        timings["saveDayCacheSeconds"] = perf_counter() - t0
        timings["signalRowCount"] = int(len(signal_quote_df))
        timings["metaRowCount"] = int(len(meta_df))
        return signal_quote_df, meta_df, timings

    def run_single_param(
        self,
        trade_dates: list[str],
        table_name: str,
        params: BacktestParams,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        trade_frames: list[pd.DataFrame] = []
        summary_frames: list[pd.DataFrame] = []
        timing_rows: list[dict[str, float | int | str]] = []
        pool_name = get_pool_name(table_name)

        for idx, trade_date in enumerate(trade_dates):
            prev_trade_date = trade_dates[idx - 1] if idx > 0 else None
            day_t0 = perf_counter()
            signal_quote_df, meta_df, prep = self.prepare_signal_mid_day(trade_date, prev_trade_date, table_name)
            if signal_quote_df.empty:
                timing_rows.append({"tradeDate": trade_date, "tradeCount": 0, "elapsedSeconds": perf_counter() - day_t0, **prep})
                continue

            t0 = perf_counter()
            trade_df, security_summary = simulate_signal_day(signal_quote_df, meta_df, pool_name, pd.Timestamp(trade_date), params)
            simulate_seconds = perf_counter() - t0

            trade_frames.append(trade_df)
            summary_frames.append(security_summary)
            timing_rows.append(
                {
                    "tradeDate": trade_date,
                    "tradeCount": int(len(trade_df)),
                    "simulateSeconds": simulate_seconds,
                    "elapsedSeconds": perf_counter() - day_t0,
                    **prep,
                }
            )

        all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        all_summaries = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
        pool_summary = aggregate_param_summary(all_summaries, params, pool_name) if not all_summaries.empty else pd.DataFrame()
        return all_trades, all_summaries, pool_summary, pd.DataFrame(timing_rows)

    def run_param_sweep(
        self,
        trade_dates: list[str],
        table_name: str,
        params_list: list[BacktestParams],
        num_vol_bins: int = 10,
        force_rebuild_cache: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[float]]:
        pool_name = get_pool_name(table_name)
        prepare_rows: list[dict[str, float | int | str]] = []
        param_rows: list[dict[str, float | int | str]] = []
        prepared_days: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
        meta_frames: list[pd.DataFrame] = []

        for idx, trade_date in enumerate(trade_dates):
            prev_trade_date = trade_dates[idx - 1] if idx > 0 else None
            t0 = perf_counter()
            signal_quote_df, meta_df, prep = self.prepare_signal_mid_day(trade_date, prev_trade_date, table_name, force_rebuild=force_rebuild_cache)
            prepare_rows.append({"tradeDate": trade_date, "elapsedSeconds": perf_counter() - t0, **prep})
            if signal_quote_df.empty:
                continue
            prepared_days.append((trade_date, signal_quote_df, meta_df))
            meta_frames.append(meta_df)

        vol_cutoffs = calc_vol_cutoffs(meta_frames, num_bins=num_vol_bins)
        summary_by_param: dict[str, list[pd.DataFrame]] = {params.param_tag: [] for params in params_list}
        price_contrib_frames: list[pd.DataFrame] = []
        vol_contrib_frames: list[pd.DataFrame] = []

        for trade_date, signal_quote_df, meta_df in prepared_days:
            for params in params_list:
                t0 = perf_counter()
                trade_df, security_summary = simulate_signal_day(signal_quote_df, meta_df, pool_name, pd.Timestamp(trade_date), params)
                param_rows.append(
                    {
                        "tradeDate": trade_date,
                        "paramTag": params.param_tag,
                        "openThreshold": params.open_threshold,
                        "closeThreshold": params.close_threshold,
                        "minHoldBars": params.min_hold_bars,
                        "tradeCount": int(len(trade_df)),
                        "simulateSeconds": perf_counter() - t0,
                    }
                )
                summary_by_param[params.param_tag].append(security_summary)
                if not trade_df.empty:
                    price_contrib_frames.append(summarize_price_bucket_contrib(trade_df, params))
                    vol_contrib_frames.append(summarize_vol_bucket_contrib(trade_df, params, vol_cutoffs))

        pool_summary_frames: list[pd.DataFrame] = []
        for params in params_list:
            frames = summary_by_param[params.param_tag]
            if not frames:
                continue
            summary_df = pd.concat(frames, ignore_index=True)
            pool_summary_frames.append(aggregate_param_summary(summary_df, params, pool_name))

        pool_summary_df = pd.concat(pool_summary_frames, ignore_index=True) if pool_summary_frames else pd.DataFrame()
        if not pool_summary_df.empty:
            pool_summary_df = pool_summary_df.sort_values(
                ["totalExecRet", "avgAllExecRet", "totalTradeCount"],
                ascending=[False, False, False],
            ).reset_index(drop=True)
        price_summary_df = finalize_price_bucket_summary(
            pd.concat(price_contrib_frames, ignore_index=True) if price_contrib_frames else pd.DataFrame()
        )
        vol_summary_df = finalize_vol_bucket_summary(
            pd.concat(vol_contrib_frames, ignore_index=True) if vol_contrib_frames else pd.DataFrame()
        )
        return pool_summary_df, price_summary_df, vol_summary_df, pd.DataFrame(prepare_rows), pd.DataFrame(param_rows), vol_cutoffs
