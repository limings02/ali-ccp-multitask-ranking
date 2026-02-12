from __future__ import annotations

import argparse


def _add_canonical_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--config",
        type=str,
        default="configs/dataset/aliccp.yaml",
        help="Path to dataset config YAML.",
    )
    subparser.add_argument("--skeleton-path", type=str, help="Override raw skeleton CSV path.")
    subparser.add_argument(
        "--common-features-path", type=str, help="Override raw common features CSV path."
    )
    subparser.add_argument("--out-dir", type=str, help="Override canonical output root directory.")
    subparser.add_argument("--sqlite-path", type=str, help="Override SQLite output path.")
    subparser.add_argument("--samples-path", type=str, help="Override samples parquet path.")
    subparser.add_argument("--tokens-dir", type=str, help="Override tokens output directory.")
    subparser.add_argument("--nrows", type=int, help="Limit number of skeleton rows to process.")
    subparser.add_argument(
        "--chunksize-sk",
        type=int,
        help="Chunk size for reading skeleton CSV.",
    )
    subparser.add_argument(
        "--chunksize-cf",
        type=int,
        help="Chunk size for reading common features CSV.",
    )
    subparser.add_argument(
        "--buffer-max-tokens",
        type=int,
        help="Max tokens to hold before flushing a parquet part.",
    )
    subparser.add_argument(
        "--rebuild-sqlite",
        action="store_true",
        help="Force rebuild of the SQLite index even if it already exists.",
    )
    subparser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing canonical outputs (samples/tokens parts).",
    )
    subparser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Set log level (INFO/DEBUG/WARN). Overrides LOGLEVEL env.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI for aliccp-mtl-ranker utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    canonical_parser = subparsers.add_parser(
        "canonical", help="Build Ali-CCP canonical samples/tokens."
    )
    _add_canonical_args(canonical_parser)

    compact_parser = subparsers.add_parser(
        "compact", help="Compact canonical tokens parquet parts."
    )
    compact_parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset/aliccp.yaml",
        help="Path to dataset config YAML.",
    )
    compact_parser.add_argument("--manifest-path", type=str, help="Path to manifest.json.")
    compact_parser.add_argument("--in-tokens-dir", type=str, help="Input tokens directory override.")
    compact_parser.add_argument("--out-tokens-dir", type=str, help="Output tokens directory.")
    compact_parser.add_argument("--target-parts", type=int, help="Target number of output parts.")
    compact_parser.add_argument(
        "--max-rows-per-file", type=int, help="Max rows per output parquet (overrides target-parts)."
    )
    compact_parser.add_argument(
        "--target-chunk-rows",
        type=int,
        default=None,
        help="Rows to accumulate before writing a row group.",
    )
    compact_parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic runs.")
    compact_parser.add_argument(
        "--inplace",
        action="store_true",
        help="Replace existing tokens_train with compacted output (backs up to tokens_train.bak).",
    )
    compact_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output tokens directory.",
    )

    split_parser = subparsers.add_parser(
        "split", help="Split canonical samples into train/valid (hash by entity_id)."
    )
    split_parser.add_argument(
        "--config", type=str, default="configs/dataset/aliccp.yaml", help="Path to dataset config."
    )
    split_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing split outputs."
    )
    split_parser.add_argument("--seed", type=int, help="Override split seed.")
    split_parser.add_argument("--train-ratio", type=float, help="Override train ratio.")
    split_parser.add_argument("--valid-ratio", type=float, help="Override valid ratio.")

    split_tokens_parser = subparsers.add_parser(
        "split-tokens", help="Filter tokens into train/valid using split row_id membership."
    )
    split_tokens_parser.add_argument(
        "--config", type=str, default="configs/dataset/aliccp.yaml", help="Path to dataset config."
    )
    split_tokens_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing tokens split output."
    )
    split_tokens_parser.add_argument(
        "--flush-rows", type=int, help="Rows threshold per output shard before flushing."
    )

    eda_parser = subparsers.add_parser("eda", help="Run EDA on a prepared split + tokens.")
    eda_parser.add_argument(
        "--config", type=str, default="configs/dataset/aliccp.yaml", help="Path to dataset config."
    )
    eda_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing EDA outputs.")
    eda_parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "duckdb", "pyarrow"],
        help="Backend for aggregations.",
    )
    eda_parser.add_argument("--topk-n", dest="topk_n", type=int, help="Top-N for field fid ranking.")
    eda_parser.add_argument(
        "--min-support", dest="min_support", type=int, help="Minimum rows for fid lift."
    )
    eda_parser.add_argument(
        "--topk-jaccard-n", dest="topk_jaccard_n", type=int, help="Top-K size for drift Jaccard."
    )

    eda_extra = subparsers.add_parser("eda-extra", help="Run refined EDA for FeatureMap decisions.")
    eda_extra.add_argument(
        "--config", type=str, default="configs/dataset/aliccp.yaml", help="Path to dataset config."
    )
    eda_extra.add_argument(
        "--in-stats",
        type=str,
        default="data/stats/eda_v1",
        help="Input directory containing base EDA outputs (field_stats_train etc.).",
    )
    eda_extra.add_argument(
        "--out",
        type=str,
        default="data/stats/eda_extra_v1",
        help="Output directory for structured parquet/json results.",
    )
    eda_extra.add_argument(
        "--plots",
        type=str,
        default="reports/eda/eda_extra_v1",
        help="Output directory for plots and featuremap patch/rationale.",
    )
    eda_extra.add_argument(
        "--debug-sample",
        action="store_true",
        help="Use data/raw/fieldwise_100k.parquet for quick validation instead of full tokens.",
    )
    eda_extra.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    eda_extra.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Set log level (INFO/DEBUG/WARN). Overrides LOGLEVEL env.",
    )

    process_parser = subparsers.add_parser(
        "process", help="Convert split (samples+tokens) into processed training data."
    )
    process_parser.add_argument(
        "--config", type=str, default="configs/dataset/featuremap.yaml", help="Path to featuremap yaml."
    )
    process_parser.add_argument(
        "--split-dir", type=str, default="data/splits/aliccp_entity_hash_v1", help="Input split directory."
    )
    process_parser.add_argument(
        "--out", type=str, default="data/processed/aliccp_entity_hash_v1", help="Output processed root directory."
    )
    process_parser.add_argument(
        "--batch-size", type=int, default=500_000, help="Rows per processing batch."
    )
    process_parser.add_argument(
        "--log-level", type=str, default=None, help="Set log level (INFO/DEBUG/WARN). Overrides LOGLEVEL env."
    )

    train_parser = subparsers.add_parser("train", help="Train DeepFM SharedBottom model.")
    train_parser.add_argument(
        "--config", type=str, default="configs/experiments/deepfm_sharedbottom_train.yaml", help="Experiment config."
    )
    train_parser.add_argument(
        "--resume-path", type=str, default=None, help="Path to checkpoint for resume training."
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained checkpoint.")
    eval_parser.add_argument(
        "--config", type=str, default="configs/experiments/deepfm_sharedbottom_train.yaml", help="Experiment config."
    )
    eval_parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file.")
    eval_parser.add_argument("--split", type=str, default="valid", help="Dataset split to evaluate.")
    eval_parser.add_argument("--run-dir", type=str, default=None, help="Run directory to place eval outputs.")
    eval_parser.add_argument(
        "--save-preds",
        action="store_true",
        help="If set, write predictions parquet (preds_<split>.parquet) alongside eval.json.",
    )
    eval_parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on number of batches for quick smoke evaluation.",
    )

    return parser
