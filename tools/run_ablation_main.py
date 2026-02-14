#!/usr/bin/env python3
"""
Run the main MTL ablation suite (baseline + 7 ablations) sequentially.

Usage:
    python tools/run_ablation_main.py
    python tools/run_ablation_main.py --only mmoe_baseline,single_task_ctr
    python tools/run_ablation_main.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("run_ablation_main")

RUNS_ROOT = PROJECT_ROOT / "runs"
REPORT_ROOT = PROJECT_ROOT / "reports" / "main_ablation"
MANIFEST_PATH = REPORT_ROOT / "main_ablation_runs.json"


@dataclass(frozen=True)
class Job:
    name: str
    config_rel: str
    desc: str


JOBS: list[Job] = [
    Job("mmoe_baseline", "configs/experiments/main/DeepFM_ctr.yaml", "DeepFM CTR"),
    Job("single_task_ctr", "configs/experiments/main/single_task_ctr.yaml", "A1: single-task CTR"),
    Job("single_task_cvr", "configs/experiments/main/single_task_cvr.yaml", "A2: single-task CVR (click-only masked)"),
    Job("single_task_ctcvr", "configs/experiments/main/single_task_ctcvr.yaml", "A3: single-task CTCVR (ESMM, lambda_ctr=0)"),
    Job("shared_bottom", "configs/experiments/main/shared_bottom.yaml", "B1: SharedBottom + legacy ESMM branch"),
    Job("shared_bottom_esmm", "configs/experiments/main/shared_bottom_esmm.yaml", "B2: SharedBottom + ESMM v2"),
    Job("esmm_mmoe_nogate", "configs/experiments/main/esmm_mmoe_nogate.yaml", "B3: ESMM + MMoE (no gate stabilize)"),
    Job("esmm_ple", "configs/experiments/main/esmm_ple.yaml", "B4: ESMM + PLE"),
]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_jobs(only: str | None) -> list[Job]:
    if not only:
        return JOBS
    requested = {x.strip() for x in only.split(",") if x.strip()}
    by_name = {j.name: j for j in JOBS}
    missing = sorted([x for x in requested if x not in by_name])
    if missing:
        raise ValueError(f"Unknown experiment names in --only: {missing}")
    return [job for job in JOBS if job.name in requested]


def _list_matching_runs(exp_name: str) -> set[str]:
    if not RUNS_ROOT.exists():
        return set()
    return {p.name for p in RUNS_ROOT.glob(f"{exp_name}_*") if p.is_dir()}


def _find_new_run_dir(exp_name: str, before: set[str], t0: float) -> Path | None:
    candidates = []
    for p in RUNS_ROOT.glob(f"{exp_name}_*"):
        if not p.is_dir():
            continue
        if p.name in before:
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        # guard against clock skew by allowing a small back-off
        if mtime >= (t0 - 5.0):
            candidates.append((mtime, p))
    if not candidates:
        # fallback: pick latest existing run for this exp_name
        all_runs = []
        for p in RUNS_ROOT.glob(f"{exp_name}_*"):
            if p.is_dir():
                try:
                    all_runs.append((p.stat().st_mtime, p))
                except OSError:
                    pass
        if not all_runs:
            return None
        all_runs.sort(key=lambda x: x[0], reverse=True)
        return all_runs[0][1]
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _load_seeds(config_path: Path) -> tuple[Any, Any]:
    cfg = load_yaml(config_path)
    runtime_seed = (cfg.get("runtime") or {}).get("seed")
    data_seed = (cfg.get("data") or {}).get("seed")
    return runtime_seed, data_seed


def _run_one(job: Job, dry_run: bool = False) -> dict[str, Any]:
    cfg_path = PROJECT_ROOT / job.config_rel
    if not cfg_path.exists():
        return {
            "name": job.name,
            "config": str(cfg_path),
            "desc": job.desc,
            "success": False,
            "error": f"config not found: {cfg_path}",
            "started_at": _now_utc(),
            "finished_at": _now_utc(),
            "run_dir": None,
        }

    runtime_seed, data_seed = _load_seeds(cfg_path)
    cmd = [sys.executable, "-m", "src.cli.main", "train", "--config", str(cfg_path)]
    before = _list_matching_runs(job.name)
    t0 = time.time()
    started_at = _now_utc()

    LOG.info("=" * 90)
    LOG.info("[%s] %s", job.name, job.desc)
    LOG.info("config=%s", cfg_path)
    LOG.info("seed(runtime)=%s seed(data)=%s", runtime_seed, data_seed)
    LOG.info("cmd=%s", " ".join(cmd))

    if dry_run:
        return {
            "name": job.name,
            "config": str(cfg_path),
            "desc": job.desc,
            "success": True,
            "dry_run": True,
            "return_code": 0,
            "started_at": started_at,
            "finished_at": _now_utc(),
            "duration_sec": 0.0,
            "run_dir": None,
            "runtime_seed": runtime_seed,
            "data_seed": data_seed,
        }

    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        rc = int(proc.returncode)
    except Exception as exc:  # noqa: BLE001
        rc = 999
        LOG.exception("[%s] failed to execute: %s", job.name, exc)

    run_dir = _find_new_run_dir(job.name, before, t0)
    finished_at = _now_utc()
    duration_sec = round(time.time() - t0, 2)
    success = (rc == 0)
    if success:
        LOG.info("[%s] done rc=%d run_dir=%s", job.name, rc, run_dir)
    else:
        LOG.error("[%s] failed rc=%d run_dir=%s", job.name, rc, run_dir)

    return {
        "name": job.name,
        "config": str(cfg_path),
        "desc": job.desc,
        "success": success,
        "dry_run": False,
        "return_code": rc,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": duration_sec,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "runtime_seed": runtime_seed,
        "data_seed": data_seed,
    }


def _write_manifest(results: list[dict[str, Any]]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": _now_utc(),
        "project_root": str(PROJECT_ROOT),
        "runs_root": str(RUNS_ROOT),
        "experiments": results,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run baseline+main ablation suite.")
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated experiment names to run (default: all 8).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print plan without launching training.")
    args = ap.parse_args()

    jobs = _resolve_jobs(args.only)
    LOG.info("project_root=%s", PROJECT_ROOT)
    LOG.info("total_jobs=%d", len(jobs))

    results: list[dict[str, Any]] = []
    for idx, job in enumerate(jobs, start=1):
        LOG.info("[%d/%d] start", idx, len(jobs))
        results.append(_run_one(job, dry_run=args.dry_run))

    _write_manifest(results)

    ok = sum(1 for r in results if r.get("success"))
    fail = len(results) - ok
    LOG.info("=" * 90)
    LOG.info("completed ok=%d fail=%d", ok, fail)
    LOG.info("manifest=%s", MANIFEST_PATH)
    LOG.info("next: python tools/summarize_main_ablation.py --manifest %s", MANIFEST_PATH)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
