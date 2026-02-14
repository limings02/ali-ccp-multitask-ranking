#!/usr/bin/env python3
"""
Scan `runs/` for classic_mmoe lambda sweeps and generate:
- reports/lambda_sweep/lambda_sweep_summary.md
- reports/lambda_sweep/lambda_sweep_runs.csv
- reports/lambda_sweep/lambda_sweep_lambdas.csv
- reports/lambda_sweep/lambda_sweep_summary.html (optional, if pandas is available)

Example:
python tools/report_lambda_sweep.py --runs_dir runs --pattern classic_mmoe_lambda --lambda_min 1 --lambda_max 13 --window 500 --out_dir reports/lambda_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

MAX_EVAL_BYTES = 50 * 1024 * 1024
TOP_K = 6
NA = "NA"


def to_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isfinite(f):
            return f
    return None


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def std(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    m = mean(xs)
    if m is None:
        return None
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    ys = sorted(xs)
    pos = (len(ys) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    w = pos - lo
    return ys[lo] * (1 - w) + ys[hi] * w


def corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx = mean(xs)
    my = mean(ys)
    if mx is None or my is None:
        return None
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    if den == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / den


def fmt(v: float | None, n: int = 6) -> str:
    return NA if v is None else f"{v:.{n}f}"


def fmt_ms(m: float | None, s: float | None) -> str:
    if m is None:
        return NA
    return f"{m:.6f}" if s is None else f"{m:.6f} ± {s:.6f}"


def grad_key(k: str) -> bool:
    lk = k.lower()
    return lk.startswith("grad_") or ("conflict" in lk) or ("cosine" in lk)


def parse_ts(name: str, mtime: float) -> str:
    m = re.search(r"(\d{8}_\d{6})$", name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")


def norm_phase(row: dict[str, Any]) -> str:
    # Prefer explicit phase; fallback to split to support old logs.
    x = row.get("phase", row.get("split"))
    if x is None:
        return "no_phase"
    s = str(x).lower()
    if "train" in s:
        return "train"
    if "valid" in s or s == "val":
        return "valid"
    return "other"


def eval_candidate(d: dict[str, Any]) -> bool:
    has_ctr = ("ctr_auc" in d) or ("auc_ctr" in d)
    has_ctcvr = any(k in d for k in ("ctcvr_auc", "ctcvr_auc_mul", "auc_ctcvr", "auc_ctcvr_mul"))
    return has_ctr and has_ctcvr


def find_eval_dict(obj: Any) -> dict[str, Any] | None:
    cand: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        cand.append(obj)
        for v in obj.values():
            if isinstance(v, dict):
                cand.append(v)
    elif isinstance(obj, list):
        cand.extend(v for v in obj if isinstance(v, dict))
    for d in cand:
        if eval_candidate(d):
            return d
    return None


def discover_eval(run_dir: Path) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    best: tuple[int, float, Path, dict[str, Any]] | None = None
    errs: list[str] = []
    for p in run_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in {".json", ".jsonl"}:
            continue
        try:
            if p.stat().st_size > MAX_EVAL_BYTES:
                continue
        except OSError:
            continue
        found = None
        try:
            if p.suffix.lower() == ".json":
                obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                found = find_eval_dict(obj)
            else:
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f):
                        if i >= 200:
                            break
                        t = line.strip()
                        if not t:
                            continue
                        try:
                            obj = json.loads(t)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, dict):
                            found = find_eval_dict(obj)
                            if found is not None:
                                break
        except Exception as e:  # noqa: BLE001
            errs.append(f"{p.name}: {e}")
            continue
        if found is None:
            continue
        name = p.name.lower()
        score = (5 if "eval" in name else 0) + (3 if "best" in name else 0) + (2 if "summary" in name else 0)
        if p.suffix.lower() == ".json":
            score += 1
        cand = (score, p.stat().st_mtime, p, found)
        if best is None or (cand[0], cand[1]) > (best[0], best[1]):
            best = cand
    if best is None:
        err = None if not errs else "eval parse failed: " + "; ".join(errs[:3])
        return None, None, err
    return best[2], best[3], None


def sample_metrics_hits(p: Path) -> int:
    hits = 0
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= 60:
                break
            t = line.strip()
            if not t:
                continue
            try:
                row = json.loads(t)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            for k, v in row.items():
                if grad_key(k) and to_float(v) is not None:
                    hits += 1
    return hits


def discover_metrics(run_dir: Path) -> tuple[Path | None, str | None]:
    cands = [p for p in run_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jsonl"]
    if not cands:
        return None, "no jsonl file found"
    best: tuple[int, float, Path] | None = None
    for p in cands:
        score = 5 if "metrics" in p.name.lower() else 0
        hits = sample_metrics_hits(p)
        score += 4 if hits > 0 else 0
        score += min(hits, 5)
        cand = (score, p.stat().st_mtime, p)
        if best is None or (cand[0], cand[1]) > (best[0], best[1]):
            best = cand
    assert best is not None
    return best[2], None


@dataclass
class Bucket:
    count: int = 0
    rows: deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=500))
    last: dict[str, float] = field(default_factory=dict)
    step: float | None = None
    gstep: float | None = None
    epoch: float | None = None


def parse_metrics(p: Path, window: int) -> tuple[dict[str, dict[str, float | int | None]], dict[str, Any]]:
    buckets: dict[str, Bucket] = {
        "train": Bucket(rows=deque(maxlen=window)),
        "valid": Bucket(rows=deque(maxlen=window)),
        "other": Bucket(rows=deque(maxlen=window)),
        "no_phase": Bucket(rows=deque(maxlen=window)),
    }
    line_n = 0
    err_n = 0
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_n += 1
            t = line.strip()
            if not t:
                continue
            try:
                row = json.loads(t)
            except json.JSONDecodeError:
                err_n += 1
                continue
            if not isinstance(row, dict):
                continue
            b = buckets[norm_phase(row)]
            packed: dict[str, float] = {}
            s = to_float(row.get("step"))
            gs = to_float(row.get("global_step"))
            ep = to_float(row.get("epoch"))
            if s is not None:
                packed["__step"] = s
                b.step = s
            if gs is not None:
                packed["__global_step"] = gs
                b.gstep = gs
            if ep is not None:
                packed["__epoch"] = ep
                b.epoch = ep
            for k, v in row.items():
                if not grad_key(k):
                    continue
                fv = to_float(v)
                if fv is None:
                    continue
                packed[k] = fv
                b.last[k] = fv
            b.rows.append(packed)
            b.count += 1

    used = "train" if buckets["train"].count > 0 else "no_phase" if buckets["no_phase"].count > 0 else "valid" if buckets["valid"].count > 0 else "other"
    b = buckets[used]
    keys = set(b.last.keys())
    for r in b.rows:
        for k in r:
            if not k.startswith("__"):
                keys.add(k)
    stats: dict[str, dict[str, float | int | None]] = {}
    for k in sorted(keys):
        vals = [r[k] for r in b.rows if k in r]
        stats[k] = {
            "last_value": b.last.get(k),
            "mean_last_window": mean(vals),
            "p10": quantile(vals, 0.1) if len(vals) >= 20 else None,
            "p50": quantile(vals, 0.5) if len(vals) >= 20 else None,
            "p90": quantile(vals, 0.9) if len(vals) >= 20 else None,
            "window_n": len(vals),
        }
    meta = {
        "phase_used": used,
        "phase_counts": {k: v.count for k, v in buckets.items()},
        "selected_row_count": b.count,
        "last_step": b.step,
        "last_global_step": b.gstep,
        "last_epoch": b.epoch,
        "line_count": line_n,
        "parse_error_count": err_n,
    }
    return stats, meta


@dataclass
class Run:
    lam: int
    run_dir: Path | None
    ts: str | None
    eval: dict[str, Any] = field(default_factory=dict)
    eval_file: str | None = None
    metrics_file: str | None = None
    grad: dict[str, dict[str, float | int | None]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    warn: list[str] = field(default_factory=list)


def run_to_row(r: Run) -> dict[str, Any]:
    row: dict[str, Any] = {
        "lambda": r.lam,
        "run_dir": str(r.run_dir) if r.run_dir else None,
        "timestamp": r.ts,
        "ctr_auc": r.eval.get("ctr_auc"),
        "cvr_auc_masked": r.eval.get("cvr_auc_masked"),
        "ctcvr_auc": r.eval.get("ctcvr_auc"),
        "ctr_logloss": r.eval.get("ctr_logloss"),
        "cvr_logloss_masked": r.eval.get("cvr_logloss_masked"),
        "ctcvr_logloss": r.eval.get("ctcvr_logloss"),
        "ckpt_path": r.eval.get("ckpt_path"),
        "eval_file": r.eval_file,
        "metrics_file": r.metrics_file,
        "phase_used": r.meta.get("phase_used"),
        "selected_row_count": r.meta.get("selected_row_count"),
        "last_step": r.meta.get("last_step"),
        "last_global_step": r.meta.get("last_global_step"),
        "last_epoch": r.meta.get("last_epoch"),
        "warn": " | ".join(r.warn) if r.warn else None,
    }
    for k, st in r.grad.items():
        p = f"grad::{k}"
        row[f"{p}::last"] = st.get("last_value")
        row[f"{p}::mean_last_window"] = st.get("mean_last_window")
        row[f"{p}::p10"] = st.get("p10")
        row[f"{p}::p50"] = st.get("p50")
        row[f"{p}::p90"] = st.get("p90")
        row[f"{p}::window_n"] = st.get("window_n")
    return row


def write_csv(rows: list[dict[str, Any]], p: Path) -> None:
    head = ["lambda", "run_dir", "timestamp", "ctr_auc", "cvr_auc_masked", "ctcvr_auc", "ctr_logloss", "cvr_logloss_masked", "ctcvr_logloss", "ckpt_path", "eval_file", "metrics_file", "phase_used", "selected_row_count", "last_step", "last_global_step", "last_epoch", "warn"]
    cols = set().union(*(r.keys() for r in rows)) if rows else set(head)
    tail = sorted(k for k in cols if k not in head)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=head + tail, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Lambda sweep report generator")
    ap.add_argument("--runs_dir", type=Path, default=Path("runs"))
    ap.add_argument("--pattern", type=str, default="classic_mmoe_lambda")
    ap.add_argument("--lambda_min", type=int, default=1)
    ap.add_argument("--lambda_max", type=int, default=13)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--out_dir", type=Path, default=Path("reports/lambda_sweep"))
    ap.add_argument("--include_all_grad_keys", type=int, choices=[0, 1], default=0)
    a = ap.parse_args()
    if a.lambda_min > a.lambda_max:
        raise ValueError("lambda_min must be <= lambda_max")
    if a.window <= 0:
        raise ValueError("window must be > 0")

    lam_re = re.compile(r".*lambda(\d+)_")
    lam_range = list(range(a.lambda_min, a.lambda_max + 1))
    found: dict[int, list[Path]] = defaultdict(list)
    for p in a.runs_dir.iterdir():
        if not p.is_dir() or a.pattern not in p.name:
            continue
        m = lam_re.match(p.name)
        if not m:
            continue
        lam = int(m.group(1))
        if a.lambda_min <= lam <= a.lambda_max:
            found[lam].append(p)
    for k in found:
        found[k].sort(key=lambda x: x.stat().st_mtime)

    runs: list[Run] = []
    for lam in lam_range:
        for rd in found.get(lam, []):
            rr = Run(lam=lam, run_dir=rd, ts=parse_ts(rd.name, rd.stat().st_mtime))
            ep, ed, ee = discover_eval(rd)
            if ee:
                rr.warn.append(ee)
            if ep is not None and ed is not None:
                rr.eval_file = str(ep)
                rr.eval = {
                    "ctr_auc": to_float(ed.get("ctr_auc", ed.get("auc_ctr"))),
                    "cvr_auc_masked": to_float(ed.get("cvr_auc_masked", ed.get("auc_cvr_click"))),
                    "ctcvr_auc": to_float(ed.get("ctcvr_auc", ed.get("ctcvr_auc_mul", ed.get("auc_ctcvr", ed.get("auc_ctcvr_mul"))))),
                    "ctr_logloss": to_float(ed.get("ctr_logloss")),
                    "cvr_logloss_masked": to_float(ed.get("cvr_logloss_masked")),
                    "ctcvr_logloss": to_float(ed.get("ctcvr_logloss")),
                    "ckpt_path": ed.get("ckpt_path"),
                }
            else:
                rr.warn.append("eval summary not found")
            mp, me = discover_metrics(rd)
            if me:
                rr.warn.append(me)
            if mp is not None:
                rr.metrics_file = str(mp)
                rr.grad, rr.meta = parse_metrics(mp, a.window)
            else:
                rr.warn.append("metrics file not found")
            runs.append(rr)

    missing = [lam for lam in lam_range if lam not in found]
    for lam in missing:
        runs.append(Run(lam=lam, run_dir=None, ts=None, warn=["No run directory found for this lambda."]))
    runs.sort(key=lambda r: (r.lam, r.ts or "", str(r.run_dir) if r.run_dir else ""))

    run_rows = [run_to_row(r) for r in runs]
    freq = Counter()
    for r in runs:
        if r.run_dir is None:
            continue
        for k, st in r.grad.items():
            if int(st.get("window_n") or 0) > 0:
                freq[k] += 1
    focus = [k for k, _ in freq.most_common(TOP_K)]

    # lambda-level aggregation
    by_lam: dict[int, list[Run]] = defaultdict(list)
    for r in runs:
        if r.run_dir is not None:
            by_lam[r.lam].append(r)
    grad_all = sorted({k for r in runs for k in r.grad})
    lam_rows: list[dict[str, Any]] = []
    for lam in lam_range:
        rs = by_lam.get(lam, [])
        row: dict[str, Any] = {"lambda": lam, "n_runs": len(rs)}
        for m in ("ctr_auc", "cvr_auc_masked", "ctcvr_auc", "ctr_logloss", "cvr_logloss_masked", "ctcvr_logloss"):
            vals = [to_float(r.eval.get(m)) for r in rs]
            vs = [v for v in vals if v is not None]
            row[f"{m}_mean"] = mean(vs)
            row[f"{m}_std"] = std(vs)
        best = None
        best_v = None
        for r in rs:
            v = to_float(r.eval.get("ctcvr_auc"))
            if v is not None and (best_v is None or v > best_v):
                best_v, best = v, r
        row["best_ctcvr_auc"] = best_v
        row["best_ctcvr_run_dir"] = str(best.run_dir) if best and best.run_dir else None
        for gk in grad_all:
            lv = [to_float(r.grad.get(gk, {}).get("last_value")) for r in rs]
            mv = [to_float(r.grad.get(gk, {}).get("mean_last_window")) for r in rs]
            lvs = [x for x in lv if x is not None]
            mvs = [x for x in mv if x is not None]
            row[f"grad::{gk}::last_mean"] = mean(lvs)
            row[f"grad::{gk}::last_std"] = std(lvs)
            row[f"grad::{gk}::mean_last_window_mean"] = mean(mvs)
            row[f"grad::{gk}::mean_last_window_std"] = std(mvs)
        lam_rows.append(row)

    out = a.out_dir
    md = out / "lambda_sweep_summary.md"
    rcsv = out / "lambda_sweep_runs.csv"
    lcsv = out / "lambda_sweep_lambdas.csv"
    html = out / "lambda_sweep_summary.html"
    write_csv(run_rows, rcsv)
    write_csv(lam_rows, lcsv)

    # Build markdown report.
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines += [
        "# Lambda Sweep Summary",
        "",
        "## Scan Scope",
        f"- generated_at: `{now}`",
        f"- runs_dir: `{a.runs_dir}`",
        f"- pattern: `{a.pattern}`",
        f"- lambda_range: `{a.lambda_min}..{a.lambda_max}`",
        f"- window: `{a.window}`",
        f"- missing_lambdas: `{missing}`",
        "",
        "## Run-level Summary",
    ]
    cols = ["lambda", "run_dir", "ctr_auc", "ctcvr_auc", "cvr_auc_masked", "ctr_logloss", "ctcvr_logloss", "phase_used"]
    for k in focus:
        cols += [f"{k}(last)", f"{k}(mean)"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for lam in lam_range:
        rs = [r for r in runs if r.lam == lam and r.run_dir is not None]
        if not rs:
            row = [str(lam), NA, NA, NA, NA, NA, NA, NA]
            for _ in focus:
                row += [NA, NA]
            lines.append("| " + " | ".join(row) + " |")
            continue
        for r in rs:
            row = [str(lam), str(r.run_dir), fmt(to_float(r.eval.get("ctr_auc"))), fmt(to_float(r.eval.get("ctcvr_auc"))), fmt(to_float(r.eval.get("cvr_auc_masked"))), fmt(to_float(r.eval.get("ctr_logloss"))), fmt(to_float(r.eval.get("ctcvr_logloss"))), str(r.meta.get("phase_used", NA))]
            for k in focus:
                st = r.grad.get(k, {})
                row += [fmt(to_float(st.get("last_value"))), fmt(to_float(st.get("mean_last_window")))]
            lines.append("| " + " | ".join(row) + " |")
    lines += ["", f"Main table uses top-{len(focus)} frequent grad keys. Full grad keys are in CSV.", "", "## Lambda-level Aggregation"]
    lc = ["lambda", "n_runs", "ctr_auc(mean±std)", "ctcvr_auc(mean±std)", "cvr_auc_masked(mean±std)", "best_ctcvr_auc", "best_ctcvr_run_dir"]
    lines.append("| " + " | ".join(lc) + " |")
    lines.append("|" + "|".join(["---"] * len(lc)) + "|")
    for r in lam_rows:
        lines.append("| " + " | ".join([str(r["lambda"]), str(r.get("n_runs", 0)), fmt_ms(to_float(r.get("ctr_auc_mean")), to_float(r.get("ctr_auc_std"))), fmt_ms(to_float(r.get("ctcvr_auc_mean")), to_float(r.get("ctcvr_auc_std"))), fmt_ms(to_float(r.get("cvr_auc_masked_mean")), to_float(r.get("cvr_auc_masked_std"))), fmt(to_float(r.get("best_ctcvr_auc"))), str(r.get("best_ctcvr_run_dir") or NA)]) + " |")

    # Conclusions
    lines += ["", "## Top Conclusions"]
    real_runs = [r for r in runs if r.run_dir is not None]
    best = None
    best_v = None
    for r in real_runs:
        v = to_float(r.eval.get("ctcvr_auc"))
        if v is not None and (best_v is None or v > best_v):
            best_v, best = v, r
    if best is not None and best_v is not None:
        lines.append(f"- best ctcvr_auc: `{best_v:.6f}` at lambda `{best.lam}` (`{best.run_dir}`), ctr_auc=`{fmt(to_float(best.eval.get('ctr_auc')))}`.")
    else:
        lines.append("- best ctcvr_auc: NA")
    l1 = [r for r in real_runs if r.lam == 1]
    if l1 and best is not None:
        l1_ctr = max((to_float(r.eval.get("ctr_auc")) for r in l1), default=None)
        b_ctr = to_float(best.eval.get("ctr_auc"))
        if l1_ctr is not None and b_ctr is not None:
            lines.append(f"- ctr_auc delta (lambda=1 -> best_ctcvr_lambda={best.lam}): `{(b_ctr-l1_ctr):+.6f}` (`{l1_ctr:.6f}` -> `{b_ctr:.6f}`).")
    # conflict/cosine trends
    gkeys = sorted({k for r in real_runs for k in r.grad})
    ckey = "conflict_rate" if "conflict_rate" in gkeys else ("conflict_rate_ema" if "conflict_rate_ema" in gkeys else next((k for k in gkeys if "conflict" in k.lower()), None))
    skey = "grad_cosine_shared_mean" if "grad_cosine_shared_mean" in gkeys else ("grad_cosine_shared_dense_mean" if "grad_cosine_shared_dense_mean" in gkeys else next((k for k in gkeys if "cosine" in k.lower()), None))
    for name, key in [("conflict", ckey), ("cosine", skey)]:
        if key is None:
            lines.append(f"- {name} trend: NA (no key found).")
            continue
        xs, ys = [], []
        for r in lam_rows:
            if int(r.get("n_runs", 0)) <= 0:
                continue
            v = to_float(r.get(f"grad::{key}::mean_last_window_mean"))
            if v is None:
                continue
            xs.append(float(r["lambda"]))
            ys.append(v)
        c = corr(xs, ys)
        if c is None:
            lines.append(f"- {name} trend `{key}`: NA (insufficient points={len(xs)}).")
        else:
            lines.append(f"- {name} trend corr(lambda, `{key}` mean_last_window): `{c:+.4f}` (n={len(xs)}).")
    lines.append(f"- missing lambdas: `{missing}`.")

    # Appendix
    lines += ["", "## Appendix: Per-lambda Grad Diagnostics", ""]
    for lam in lam_range:
        rs = [r for r in real_runs if r.lam == lam]
        lines.append(f"### lambda={lam}")
        if not rs:
            lines += ["- No run found.", ""]
            continue
        keys = sorted({k for r in rs for k in r.grad})
        show = keys if a.include_all_grad_keys == 1 else [k for k in focus if k in keys]
        if a.include_all_grad_keys == 0 and len(keys) > len(show):
            lines.append(f"- Showing focus keys only ({len(show)}/{len(keys)}). Use `--include_all_grad_keys 1` to show all.")
        lines.append("| key | last(mean over runs) | mean_last_window(mean over runs) | runs_with_key |")
        lines.append("|---|---|---|---|")
        for k in show:
            lv = [to_float(r.grad.get(k, {}).get("last_value")) for r in rs]
            mv = [to_float(r.grad.get(k, {}).get("mean_last_window")) for r in rs]
            lvs = [x for x in lv if x is not None]
            mvs = [x for x in mv if x is not None]
            n = sum(1 for r in rs if k in r.grad)
            lines.append(f"| {k} | {fmt(mean(lvs))} | {fmt(mean(mvs))} | {n} |")
        lines.append("")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(lines), encoding="utf-8")

    html_ok = False
    try:
        import pandas as pd  # type: ignore
        h = [
            "<html><head><meta charset='utf-8'><title>Lambda Sweep</title></head><body>",
            "<h1>Lambda Sweep Summary</h1>",
            "<h2>Run-level</h2>",
            pd.DataFrame(run_rows).to_html(index=False),
            "<h2>Lambda-level</h2>",
            pd.DataFrame(lam_rows).to_html(index=False),
            "</body></html>",
        ]
        html.write_text("\n".join(h), encoding="utf-8")
        html_ok = True
    except Exception:
        html_ok = False

    print("Lambda sweep report generated:")
    print(f"- markdown: {md}")
    print(f"- run csv: {rcsv}")
    print(f"- lambda csv: {lcsv}")
    print(f"- html: {html if html_ok else 'skipped (pandas unavailable)'}")


if __name__ == "__main__":
    main()
