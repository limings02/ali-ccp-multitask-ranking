from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict
from datetime import datetime

from src.cli.args import build_parser
from src.core.logging import get_logger
from src.data.canonical import build_canonical_aliccp, load_aliccp_config
from src.data.compact_tokens import compact_tokens_dir
from src.data.split import materialize_entity_hash_split
from src.data.token_filter import build_rowid_membership, filter_tokens_by_rowids
from src.data.processed_builder import build_processed_dataset
from src.eda.aliccp_eda import run_eda_aliccp
from src.eda.extra import run_eda_extra
from src.eval.run_eval import run_eval
from src.train.trainer import Trainer
from src.utils.config import load_yaml


def _collect_overrides(args) -> Dict[str, object]:
    overrides: Dict[str, object] = {}
    mapping = {
        "skeleton_path": "raw.skeleton_path",
        "common_features_path": "raw.common_features_path",
        "out_dir": "canonical.out_dir",
        "sqlite_path": "canonical.sqlite_path",
        "samples_path": "canonical.samples_path",
        "tokens_dir": "canonical.tokens_dir",
        "nrows": "params.nrows",
        "chunksize_sk": "params.chunksize_sk",
        "chunksize_cf": "params.chunksize_cf",
        "buffer_max_tokens": "params.buffer_max_tokens",
    }
    for arg_name, cfg_key in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[cfg_key] = value
    if getattr(args, "rebuild_sqlite", False):
        overrides["params.rebuild_sqlite"] = True
    if getattr(args, "overwrite", False):
        overrides["params.overwrite"] = True
    return overrides


def _collect_eda_overrides(args) -> Dict[str, object]:
    overrides: Dict[str, object] = {}
    if getattr(args, "overwrite", False):
        overrides["eda.overwrite"] = True
    if getattr(args, "backend", None):
        overrides["eda.backend"] = args.backend
    if getattr(args, "topk_n", None) is not None:
        overrides["eda.topk_n"] = args.topk_n
    if getattr(args, "min_support", None) is not None:
        overrides["eda.min_support"] = args.min_support
    if getattr(args, "topk_jaccard_n", None) is not None:
        overrides["eda.topk_jaccard_n"] = args.topk_jaccard_n
    return overrides


def main():
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "log_level", None):
        os.environ["LOGLEVEL"] = args.log_level
        logging.getLogger().setLevel(args.log_level.upper())

    logger = get_logger(__name__)

    if args.command == "canonical":
        overrides = _collect_overrides(args)
        manifest = build_canonical_aliccp(args.config, overrides=overrides)
        manifest_path = (
            os.path.join(manifest["output_dir"], "manifest.json")
            if manifest.get("output_dir")
            else None
        )
        logger.info(
            "Canonical build completed. samples_rows=%s tokens_rows=%s manifest=%s",
            manifest.get("samples_rows"),
            manifest.get("tokens_total_rows"),
            manifest_path,
        )
        print(json.dumps({"manifest": manifest_path, "summary": manifest}, indent=2))
    elif args.command == "compact":
        config = load_aliccp_config(args.config)
        canonical_cfg = config.get("canonical", {})
        compact_cfg = config.get("compact", {})

        manifest_path = args.manifest_path or compact_cfg.get("manifest_path")
        if not manifest_path:
            manifest_path = os.path.join(
                canonical_cfg.get("out_dir", "data/interim/aliccp_canonical"), "manifest.json"
            )

        out_tokens_dir = args.out_tokens_dir or compact_cfg.get("out_tokens_dir")
        in_tokens_dir = args.in_tokens_dir or compact_cfg.get("in_tokens_dir")
        target_parts = args.target_parts or compact_cfg.get("target_parts", 100)
        max_rows_per_file = args.max_rows_per_file or compact_cfg.get("max_rows_per_file")
        target_chunk_rows = (
            args.target_chunk_rows or compact_cfg.get("target_chunk_rows") or 2_000_000
        )
        inplace = bool(args.inplace or compact_cfg.get("inplace", False))
        overwrite = bool(args.overwrite or compact_cfg.get("overwrite", False))
        seed = args.seed or compact_cfg.get("seed", 20260121)

        result = compact_tokens_dir(
            manifest_path=manifest_path,
            in_tokens_dir=in_tokens_dir,
            out_tokens_dir=out_tokens_dir,
            target_parts=target_parts,
            max_rows_per_file=max_rows_per_file,
            target_chunk_rows=target_chunk_rows,
            overwrite=overwrite,
            inplace=inplace,
            seed=seed,
        )
        manifest_compact_path = Path(result.get("output_dir", Path(manifest_path).parent)) / (
            "manifest.compact.json"
        )
        logger.info(
            "Compaction completed. output_tokens_dir=%s manifest=%s",
            result.get("tokens_dir_compact"),
            manifest_compact_path,
        )
        print(
            json.dumps(
                {
                    "manifest_compact": str(manifest_compact_path),
                    "tokens_dir_compact": result.get("tokens_dir_compact"),
                },
                indent=2,
            )
        )
    elif args.command == "split":
        config = load_aliccp_config(args.config)
        canonical_cfg = config.get("canonical", {})
        split_cfg = config.get("split", {})
        samples_path = Path(canonical_cfg.get("samples_path", "data/interim/aliccp_canonical/samples_train.parquet"))
        out_dir = Path(split_cfg.get("out_dir", "data/splits/aliccp_entity_hash_v1"))
        seed = args.seed if getattr(args, "seed", None) is not None else split_cfg.get("seed", 20260121)
        ratios = split_cfg.get("ratios", {"train": 0.95, "valid": 0.05})
        chunksize = split_cfg.get("chunksize", 2_000_000)
        overwrite = bool(args.overwrite or split_cfg.get("overwrite", False))
        result = materialize_entity_hash_split(
            samples_path=samples_path,
            out_dir=out_dir,
            seed=seed,
            ratios=ratios,
            overwrite=overwrite,
            chunksize=chunksize,
        )
        logger.info("Split completed. spec=%s stats=%s", result.get("spec"), result.get("stats"))
        print(json.dumps({"split_spec": result.get("spec"), "split_stats": result.get("stats")}, indent=2))
    elif args.command == "split-tokens":
        config = load_aliccp_config(args.config)
        canonical_cfg = config.get("canonical", {})
        tokens_split_cfg = config.get("tokens_split", {})
        split_cfg = config.get("split", {})

        split_dir = Path(tokens_split_cfg.get("split_dir", split_cfg.get("out_dir", "data/splits/aliccp_entity_hash_v1")))
        tokens_dir = tokens_split_cfg.get("tokens_dir") or canonical_cfg.get(
            "tokens_dir", "data/interim/aliccp_canonical/tokens_train"
        )
        tokens_dir = Path(tokens_dir)
        out_dir = Path(tokens_split_cfg.get("out_dir", "data/interim/aliccp_tokens_split_v1"))
        flush_rows = args.flush_rows or tokens_split_cfg.get("flush_rows", 400_000_000)
        overwrite = bool(args.overwrite or tokens_split_cfg.get("overwrite", False))

        train_samples = split_dir / "samples_train.parquet"
        valid_samples = split_dir / "samples_valid.parquet"
        if not train_samples.exists() or not valid_samples.exists():
            raise FileNotFoundError("Split samples not found. Run split first.")

        logger.info("Building train membership from %s", train_samples)
        train_mem = build_rowid_membership(train_samples)
        logger.info("Building valid membership from %s", valid_samples)
        valid_mem = build_rowid_membership(valid_samples)

        stats = filter_tokens_by_rowids(
            tokens_dir=tokens_dir,
            train_membership=train_mem,
            valid_membership=valid_mem,
            out_dir=out_dir,
            flush_rows=flush_rows,
            overwrite=overwrite,
        )

        # Read split seed for manifest reuse
        split_spec_path = split_dir / "split_spec.json"
        split_seed = None
        if split_spec_path.exists():
            with open(split_spec_path, "r", encoding="utf-8") as f:
                split_seed = json.load(f).get("seed")

        canonical_manifest_path = Path(canonical_cfg.get("out_dir", "data/interim/aliccp_canonical")) / "manifest.json"
        expected_total = None
        if canonical_manifest_path.exists():
            with open(canonical_manifest_path, "r", encoding="utf-8") as f:
                expected_total = json.load(f).get("tokens_total_rows")
        observed_total = stats["train_token_rows"] + stats["valid_token_rows"]

        split_manifest = {
            "input_tokens_dir": str(tokens_dir),
            "split_dir": str(split_dir),
            "out_dir": str(out_dir),
            "train_token_rows": stats["train_token_rows"],
            "valid_token_rows": stats["valid_token_rows"],
            "train_files_count": stats["train_files_count"],
            "valid_files_count": stats["valid_files_count"],
            "created_at": datetime.utcnow().isoformat() + "Z",
            "seed": split_seed,
            "expected_tokens_total_rows": expected_total,
            "observed_tokens_total_rows": observed_total,
            "diff": None if expected_total is None else observed_total - expected_total,
        }
        manifest_path = out_dir / "tokens_split_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(split_manifest, f, ensure_ascii=False, indent=2)
        if expected_total is not None and expected_total != observed_total:
            logger.warning(
                "Tokens split row mismatch: expected=%s observed=%s diff=%s",
                expected_total,
                observed_total,
                split_manifest["diff"],
            )
        logger.info("Tokens split completed. manifest=%s", manifest_path)
        print(json.dumps({"tokens_split_manifest": str(manifest_path)}, indent=2))
    elif args.command == "eda":
        overrides = _collect_eda_overrides(args)
        outputs = run_eda_aliccp(args.config, overrides=overrides)
        logger.info("EDA completed. sanity=%s", outputs.get("sanity"))
        print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2))
    elif args.command == "eda-extra":
        if getattr(args, "log_level", None):
            os.environ["LOGLEVEL"] = args.log_level
            logging.getLogger().setLevel(args.log_level.upper())
        outputs = run_eda_extra(
            config_path=args.config,
            in_stats_dir=args.in_stats,
            out_stats_dir=args.out,
            plots_dir=args.plots,
            debug_sample=bool(args.debug_sample),
            overwrite=bool(args.overwrite),
        )
        logger.info("EDA-extra completed. outputs=%s", outputs)
        print(json.dumps(outputs, indent=2))
    elif args.command == "process":
        if getattr(args, "log_level", None):
            os.environ["LOGLEVEL"] = args.log_level
            logging.getLogger().setLevel(args.log_level.upper())
        result = build_processed_dataset(
            config_path=args.config,
            split_dir=args.split_dir,
            out_root=args.out,
            batch_size=args.batch_size,
        )
        logger.info("Process completed. output=%s", result.get("processed_root"))
        print(json.dumps(result, indent=2))
    elif args.command == "train":
        cfg = load_yaml(args.config)
        # ===== Support --resume-path command line argument =====
        if args.resume_path:
            cfg.setdefault("runtime", {})["resume_path"] = args.resume_path
        trainer = Trainer(cfg, config_path=args.config)
        trainer.run()
    elif args.command == "eval":
        cfg = load_yaml(args.config)
        result = run_eval(
            cfg=cfg,
            split=args.split,
            ckpt_path=args.ckpt,
            run_dir=args.run_dir,
            save_preds=bool(args.save_preds),
            max_batches=args.max_batches,
            logger=logger,
            config_path=args.config,
        )
        print(
            json.dumps(
                {"eval_json": str(Path(result["run_dir"]) / "eval.json"), "summary": result},
                indent=2,
            )
        )
    else:  # pragma: no cover - argparse guards this
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
