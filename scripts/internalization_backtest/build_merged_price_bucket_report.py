from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from signal_trade_perf.merged_result_stats import aggregate_merged_security_by_price_bucket, normalize_variant_tags
from signal_trade_perf.source_backtest import dataframe_to_csv_with_retry, mkdir_with_retry


def _parse_variant_tag_list(raw: str) -> list[str]:
    # 支持逗号或分号分隔，方便命令行里直接贴 variantTag。
    tokens = [token.strip() for token in raw.replace(";", ",").split(",")]
    return [token for token in tokens if token]


def main() -> None:
    parser = argparse.ArgumentParser(description="从 by_ticker_daily_summary 生成价格分组总表。")
    parser.add_argument("--input", required=True, help="by_ticker_daily_summary.csv 路径")
    parser.add_argument("--variant-tags", required=True, help="逗号分隔的 merged variant tags")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output-prefix", default="")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    variant_tags = normalize_variant_tags(_parse_variant_tag_list(args.variant_tags))
    by_ticker_df = pd.read_csv(input_path)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    mkdir_with_retry(output_dir)

    # 这里只保留目标策略组合，避免输出无关 variant 的汇总结果。
    filtered_df = by_ticker_df[by_ticker_df["variantTag"].isin(variant_tags)].copy()
    if filtered_df.empty:
        raise ValueError(f"No rows found for variant tags: {variant_tags}")

    # 价格分组统计是后处理逻辑，不回写单票明细，只生成最终总表。
    total_df = aggregate_merged_security_by_price_bucket(
        filtered_df,
        variant_tags=variant_tags,
        include_trade_date=False,
    )

    prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    dataframe_to_csv_with_retry(total_df, output_dir / f"{prefix}by_price_bucket_total_summary.csv", index=False)
    print(f"input={input_path}")
    print(f"variantTags={','.join(variant_tags)}")
    print(f"outputDir={output_dir}")


if __name__ == "__main__":
    main()
