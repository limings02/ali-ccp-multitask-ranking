#!/usr/bin/env python3
"""
Summarize the main ablation suite into CSV + Markdown.

Usage:
    python tools/summarize_main_ablation.py
    python tools/summarize_main_ablation.py --manifest reports/main_ablation/main_ablation_runs.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ORDER = [
    "mmoe_baseline",
    "single_task_ctr",
    "single_task_cvr",
    "single_task_ctcvr",
    "shared_bottom",
    "shared_bottom_esmm",
    "esmm_mmoe_nogate",
    "esmm_ple",
]

ANALYSIS_ONLY_EXPERIMENTS = {
    "single_task_ctr",
    "single_task_cvr",
    "single_task_ctcvr",
    "shared_bottom",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isfinite(f):
            return f
        return None
    return None


def _safe_json_load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _safe_jsonl_load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            try:
                obj = json.loads(t)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _discover_latest_run(runs_root: Path, exp_name: str) -> Path | None:
    cands: list[tuple[float, Path]] = []
    for p in runs_root.glob(f"{exp_name}_*"):
        if not p.is_dir():
            continue
        try:
            cands.append((p.stat().st_mtime, p))
        except OSError:
            continue
    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]


def _load_manifest_map(manifest_path: Path) -> dict[str, dict[str, Any]]:
    payload = _safe_json_load(manifest_path)
    if not payload:
        return {}
    rows = payload.get("experiments", [])
    if not isinstance(rows, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = r.get("name")
        if isinstance(name, str):
            out[name] = r
    return out


def _get_split_rows(metrics_rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    out = []
    for r in metrics_rows:
        if str(r.get("split", "")).lower() == split.lower():
            out.append(r)
    return out


def _select_last_train_row(metrics_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    train_rows = _get_split_rows(metrics_rows, "train")
    if not train_rows:
        return None
    train_rows.sort(key=lambda r: (_to_float(r.get("global_step")) or -1.0, _to_float(r.get("epoch")) or -1.0))
    return train_rows[-1]


def _score_valid_row(row: dict[str, Any]) -> tuple[float, float, float, float]:
    ctcvr = _to_float(row.get("auc_ctcvr")) or float("-inf")
    primary = _to_float(row.get("auc_primary")) or float("-inf")
    ctr = _to_float(row.get("auc_ctr")) or float("-inf")
    cvr = _to_float(row.get("auc_cvr")) or float("-inf")
    return (ctcvr, primary, ctr, cvr)


def _select_best_valid_row(metrics_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid_rows = _get_split_rows(metrics_rows, "valid")
    if not valid_rows:
        return None

    decision_rows = _get_split_rows(metrics_rows, "valid_decision")
    decision_rows = [r for r in decision_rows if bool(r.get("should_update", False))]
    if decision_rows:
        decision_rows.sort(key=lambda r: (_to_float(r.get("global_step")) or -1.0))
        gstep = _to_float(decision_rows[-1].get("global_step"))
        if gstep is not None:
            exact = [r for r in valid_rows if (_to_float(r.get("global_step")) == gstep)]
            if exact:
                return exact[-1]
            # fallback: nearest previous valid row
            prev = [r for r in valid_rows if (_to_float(r.get("global_step")) or -1.0) <= gstep]
            if prev:
                prev.sort(key=lambda r: (_to_float(r.get("global_step")) or -1.0))
                return prev[-1]

    valid_rows.sort(key=_score_valid_row, reverse=True)
    return valid_rows[0]


def _extract_eval_metrics(eval_obj: dict[str, Any] | None) -> dict[str, Any]:
    if not eval_obj:
        return {}

    ctr = eval_obj.get("ctr", {}) if isinstance(eval_obj.get("ctr"), dict) else {}
    cvr = eval_obj.get("cvr_masked", {}) if isinstance(eval_obj.get("cvr_masked"), dict) else {}
    ctcvr = eval_obj.get("ctcvr", {}) if isinstance(eval_obj.get("ctcvr"), dict) else {}

    return {
        "ctr_auc": _to_float(eval_obj.get("ctr_auc")) or _to_float(ctr.get("auc")),
        "ctr_logloss": _to_float(eval_obj.get("ctr_logloss")) or _to_float(ctr.get("logloss")),
        "ctr_pos_rate": _to_float(ctr.get("pos_rate")),
        "ctr_pred_mean": _to_float(ctr.get("pred_mean")),
        "cvr_auc": _to_float(eval_obj.get("cvr_auc_masked")) or _to_float(cvr.get("auc")),
        "cvr_logloss": _to_float(eval_obj.get("cvr_logloss_masked")) or _to_float(cvr.get("logloss")),
        "cvr_pos_rate": _to_float(cvr.get("pos_rate")),
        "cvr_pred_mean": _to_float(cvr.get("pred_mean")),
        "ctcvr_auc": _to_float(eval_obj.get("ctcvr_auc")) or _to_float(ctcvr.get("auc")),
        "ctcvr_logloss": _to_float(eval_obj.get("ctcvr_logloss")) or _to_float(ctcvr.get("logloss")),
        "ctcvr_pos_rate": _to_float(ctcvr.get("pos_rate")),
        "ctcvr_pred_mean": _to_float(ctcvr.get("pred_mean")),
    }


def _max_from_list_like(v: Any) -> float | None:
    if not isinstance(v, list) or len(v) == 0:
        return None
    vals = [_to_float(x) for x in v]
    vals = [x for x in vals if x is not None]
    return max(vals) if vals else None


def _mean(vals: list[float | None]) -> float | None:
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _extract_gate_metrics(
    best_valid_row: dict[str, Any] | None,
    train_row: dict[str, Any] | None,
    expert_diag_rows: list[dict[str, Any]],
) -> tuple[float | None, float | None, float | None]:
    # gate entropy (prefer valid health metrics; fallback to train gate regularization metric)
    gate_entropy = None
    if best_valid_row is not None:
        gate_entropy = _mean(
            [
                _to_float(best_valid_row.get("gate_ctr_entropy_mean")),
                _to_float(best_valid_row.get("gate_cvr_entropy_mean")),
            ]
        )
    if gate_entropy is None and train_row is not None:
        gate_entropy = _to_float(train_row.get("gate_entropy_mean"))

    # top1 share concentration
    top1_share = None
    if best_valid_row is not None:
        top1_share = _mean(
            [
                _max_from_list_like(best_valid_row.get("gate_ctr_top1_share")),
                _max_from_list_like(best_valid_row.get("gate_cvr_top1_share")),
            ]
        )

    # gini from expert_health_diag valid record (if available)
    gini = None
    if expert_diag_rows:
        valid_rows = [r for r in expert_diag_rows if str(r.get("phase", "")).lower() == "valid"]
        target = valid_rows[-1] if valid_rows else expert_diag_rows[-1]
        util_ctr = target.get("utilization_ctr", {}) if isinstance(target.get("utilization_ctr"), dict) else {}
        util_cvr = target.get("utilization_cvr", {}) if isinstance(target.get("utilization_cvr"), dict) else {}
        gini = _mean(
            [
                _to_float(util_ctr.get("gini_coefficient")),
                _to_float(util_cvr.get("gini_coefficient")),
            ]
        )
        if top1_share is None:
            ctr_top1 = util_ctr.get("expert_top1_share", {})
            cvr_top1 = util_cvr.get("expert_top1_share", {})
            ctr_max = max([_to_float(v) or -1.0 for v in ctr_top1.values()], default=-1.0) if isinstance(ctr_top1, dict) else -1.0
            cvr_max = max([_to_float(v) or -1.0 for v in cvr_top1.values()], default=-1.0) if isinstance(cvr_top1, dict) else -1.0
            vals = [v for v in [ctr_max, cvr_max] if v >= 0]
            top1_share = _mean(vals) if vals else None

    return gate_entropy, top1_share, gini


def _fmt(v: Any, ndigits: int = 6) -> str:
    fv = _to_float(v)
    if fv is None:
        return "NA"
    return f"{fv:.{ndigits}f}"


def _write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    columns = [
        "experiment",
        "run_dir",
        "success",
        "online_candidate",
        "seed_runtime",
        "seed_data",
        "ctr_auc",
        "ctr_logloss",
        "ctr_pos_rate",
        "ctr_pred_mean",
        "cvr_auc",
        "cvr_logloss",
        "cvr_pos_rate",
        "cvr_pred_mean",
        "ctcvr_auc",
        "ctcvr_logloss",
        "ctcvr_pos_rate",
        "ctcvr_pred_mean",
        "grad_cosine_p10",
        "grad_cosine_p50",
        "grad_cosine_p90",
        "conflict_rate",
        "neg_conflict_strength_mean",
        "gate_entropy",
        "expert_top1_share",
        "gini",
        "ctcvr_auc_delta_vs_baseline",
        "note",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_md(
    rows: list[dict[str, Any]],
    out_path: Path,
    signif_delta: float,
) -> None:
    baseline = next((r for r in rows if r["experiment"] == "mmoe_baseline"), None)
    baseline_ctcvr = _to_float(baseline.get("ctcvr_auc")) if baseline else None

    sortable = list(rows)
    sortable.sort(
        key=lambda r: (_to_float(r.get("ctcvr_auc")) if _to_float(r.get("ctcvr_auc")) is not None else float("-inf")),
        reverse=True,
    )

    better = []
    failures = []
    for r in rows:
        if not bool(r.get("success", False)):
            failures.append((r["experiment"], str(r.get("note") or "run failed")))
            continue
        d = _to_float(r.get("ctcvr_auc_delta_vs_baseline"))
        if d is not None and d >= signif_delta:
            better.append((r["experiment"], d))

    lines: list[str] = []
    lines.append("# Main Ablation Report")
    lines.append("")
    lines.append(f"- generated_at: `{_now_utc()}`")
    lines.append(f"- sort_key: `ctcvr_auc (desc)`")
    lines.append(f"- significance_rule: `ctcvr_auc_delta_vs_baseline >= {signif_delta:.4f}`")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| experiment | ctcvr_auc | delta_vs_base | ctr_auc | cvr_auc | grad_cosine_p50 | conflict_rate | gate_entropy | top1_share | gini | online_candidate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in sortable:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.get("experiment")),
                    _fmt(r.get("ctcvr_auc")),
                    _fmt(r.get("ctcvr_auc_delta_vs_baseline")),
                    _fmt(r.get("ctr_auc")),
                    _fmt(r.get("cvr_auc")),
                    _fmt(r.get("grad_cosine_p50")),
                    _fmt(r.get("conflict_rate")),
                    _fmt(r.get("gate_entropy")),
                    _fmt(r.get("expert_top1_share")),
                    _fmt(r.get("gini")),
                    "yes" if bool(r.get("online_candidate")) else "no",
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Better Than Baseline")
    if better:
        for name, delta in sorted(better, key=lambda x: x[1], reverse=True):
            lines.append(f"- `{name}`: `+{delta:.6f}` ctcvr_auc vs baseline")
    else:
        lines.append("- None under current significance rule.")

    lines.append("")
    lines.append("## Failed / Incomplete Runs")
    if failures:
        for name, reason in failures:
            lines.append(f"- `{name}`: {reason}")
    else:
        lines.append("- None.")

    lines.append("")
    lines.append("## CVR Single-Task Note")
    lines.append("- `single_task_cvr` uses click-only CVR with `click_mask`; no synthetic negatives are introduced.")
    lines.append("- This CVR single-task result is not equal to real online full-exposure CVR; it is a representation/gradient-behavior control experiment.")

    lines.append("")
    lines.append("## Online Suitability")
    lines.append("- Analysis-only (not recommended for direct online serving):")
    for name in sorted(ANALYSIS_ONLY_EXPERIMENTS):
        lines.append(f"  - `{name}`")
    lines.append("- Candidate for online consideration (after additional validation):")
    for name in EXPERIMENT_ORDER:
        if name not in ANALYSIS_ONLY_EXPERIMENTS:
            lines.append(f"  - `{name}`")

    if baseline_ctcvr is not None:
        lines.append("")
        lines.append(f"- baseline_ctcvr_auc: `{baseline_ctcvr:.6f}`")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize main ablation suite from run artifacts.")
    ap.add_argument("--manifest", type=Path, default=Path("reports/main_ablation/main_ablation_runs.json"))
    ap.add_argument("--runs-root", type=Path, default=Path("runs"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/main_ablation"))
    ap.add_argument("--signif-delta", type=float, default=0.002, help="CTCVR AUC delta threshold vs baseline.")
    args = ap.parse_args()

    manifest_map = _load_manifest_map(args.manifest)
    rows: list[dict[str, Any]] = []

    for exp in EXPERIMENT_ORDER:
        manifest_row = manifest_map.get(exp, {})
        run_dir = manifest_row.get("run_dir")
        if isinstance(run_dir, str) and run_dir:
            rd = Path(run_dir)
        else:
            rd = _discover_latest_run(args.runs_root, exp)

        success = bool(manifest_row.get("success", rd is not None))
        note_parts: list[str] = []
        if rd is None:
            rows.append(
                {
                    "experiment": exp,
                    "run_dir": None,
                    "success": False,
                    "online_candidate": exp not in ANALYSIS_ONLY_EXPERIMENTS,
                    "seed_runtime": manifest_row.get("runtime_seed"),
                    "seed_data": manifest_row.get("data_seed"),
                    "note": "run_dir not found",
                }
            )
            continue

        eval_obj = _safe_json_load(rd / "eval.json")
        if eval_obj is None:
            success = False
            note_parts.append("missing eval.json")

        metrics_rows = _safe_jsonl_load(rd / "metrics.jsonl")
        if not metrics_rows:
            success = False
            note_parts.append("missing metrics.jsonl")

        expert_diag_rows = _safe_jsonl_load(rd / "expert_health_diag.jsonl")
        train_row = _select_last_train_row(metrics_rows) if metrics_rows else None
        best_valid_row = _select_best_valid_row(metrics_rows) if metrics_rows else None

        eval_metrics = _extract_eval_metrics(eval_obj)
        gate_entropy, top1_share, gini = _extract_gate_metrics(best_valid_row, train_row, expert_diag_rows)

        row = {
            "experiment": exp,
            "run_dir": str(rd),
            "success": success,
            "online_candidate": exp not in ANALYSIS_ONLY_EXPERIMENTS,
            "seed_runtime": manifest_row.get("runtime_seed"),
            "seed_data": manifest_row.get("data_seed"),
            "ctr_auc": eval_metrics.get("ctr_auc"),
            "ctr_logloss": eval_metrics.get("ctr_logloss"),
            "ctr_pos_rate": eval_metrics.get("ctr_pos_rate"),
            "ctr_pred_mean": eval_metrics.get("ctr_pred_mean"),
            "cvr_auc": eval_metrics.get("cvr_auc"),
            "cvr_logloss": eval_metrics.get("cvr_logloss"),
            "cvr_pos_rate": eval_metrics.get("cvr_pos_rate"),
            "cvr_pred_mean": eval_metrics.get("cvr_pred_mean"),
            "ctcvr_auc": eval_metrics.get("ctcvr_auc"),
            "ctcvr_logloss": eval_metrics.get("ctcvr_logloss"),
            "ctcvr_pos_rate": eval_metrics.get("ctcvr_pos_rate"),
            "ctcvr_pred_mean": eval_metrics.get("ctcvr_pred_mean"),
            "grad_cosine_p10": _to_float((train_row or {}).get("grad_cosine_p10")),
            "grad_cosine_p50": _to_float((train_row or {}).get("grad_cosine_p50")),
            "grad_cosine_p90": _to_float((train_row or {}).get("grad_cosine_p90")),
            "conflict_rate": _to_float((train_row or {}).get("conflict_rate")),
            "neg_conflict_strength_mean": _to_float((train_row or {}).get("neg_conflict_strength_mean")),
            "gate_entropy": gate_entropy,
            "expert_top1_share": top1_share,
            "gini": gini,
            "ctcvr_auc_delta_vs_baseline": None,
            "note": " | ".join(note_parts) if note_parts else "",
        }
        rows.append(row)

    baseline = next((r for r in rows if r["experiment"] == "mmoe_baseline"), None)
    baseline_ctcvr = _to_float(baseline.get("ctcvr_auc")) if baseline else None
    if baseline_ctcvr is not None:
        for r in rows:
            ctcvr = _to_float(r.get("ctcvr_auc"))
            r["ctcvr_auc_delta_vs_baseline"] = (ctcvr - baseline_ctcvr) if ctcvr is not None else None

    out_csv = args.out_dir / "main_ablation_report.csv"
    out_md = args.out_dir / "main_ablation_report.md"
    _write_csv(rows, out_csv)
    _write_md(rows, out_md, signif_delta=args.signif_delta)

    print("Main ablation summary generated:")
    print(f"- csv: {out_csv}")
    print(f"- md:  {out_md}")


if __name__ == "__main__":
    main()
