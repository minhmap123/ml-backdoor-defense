#!/usr/bin/env python3
"""
Post-hoc re-scoring of NC và NC-CSO bằng ratio test thay MAD.
Không chạy lại optimization — chỉ đọc mask_norms từ optimization_trace.json đã lưu.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK_ROOT = Path("results/benchmark_runs")
MAD_THRESHOLDS   = [2.0, 1.0]
RATIO_THRESHOLDS = [0.3, 0.5]


# ── Decision rules ──────────────────────────────────────────────────────────

def mad_flag(norms, threshold):
    norms = np.array(norms, dtype=float)
    median = np.median(norms)
    mad = 1.4826 * np.median(np.abs(norms - median))
    if mad < 1e-7:
        return False, None
    flagged = [i for i, v in enumerate(norms) if v < median and abs(v - median) / mad > threshold]
    flagged.sort(key=lambda i: norms[i])
    return (True, flagged[0]) if flagged else (False, None)


def ratio_flag(norms, threshold):
    norms = np.array(norms, dtype=float)
    median = np.median(norms)
    if median < 1e-7:
        return False, None
    flagged = [i for i, v in enumerate(norms) if v / median < threshold]
    flagged.sort(key=lambda i: norms[i])
    return (True, flagged[0]) if flagged else (False, None)


# ── Collect runs ─────────────────────────────────────────────────────────────

rows = []

for trace_path in sorted(BENCHMARK_ROOT.glob(
    "*/*/artifacts/detection/*/detection/*/optimization_trace.json"
)):
    parts = trace_path.parts
    dataset   = parts[-8]   # cic_ids2018 / iotid20
    run_dir   = parts[-4]   # mode-attacked__attack-...
    det_name  = parts[-2]   # neural_cleanse / nc_cso

    if det_name not in ("neural_cleanse", "nc_cso"):
        continue

    # Parse run name: mode-attacked__attack-badnets__seed7__model-mlp__det-neural_cleanse
    fields = dict(p.split("-", 1) for p in run_dir.split("__") if "-" in p)
    attack = fields.get("attack", "?")
    model  = fields.get("model", "?")

    with open(trace_path) as f:
        trace = json.load(f)
    with open(trace_path.parent / "summary.json") as f:
        summary = json.load(f)

    norms       = trace["mask_norms"]
    ctx         = summary.get("context", {})
    true_target = ctx.get("attack_target_label")
    candidate   = summary.get("candidate_target_class")

    row = {
        "dataset":           dataset,
        "attack":            attack,
        "model":             model,
        "detector":          det_name,
        "K":                 len(norms),
        "true_target":       true_target,
        "candidate":         candidate,
        "candidate_correct": candidate == true_target,
        "min/med":           min(norms) / (np.median(norms) + 1e-9),
    }

    for t in MAD_THRESHOLDS:
        flag, pred = mad_flag(norms, t)
        row[f"MAD>{t}"] = flag and pred == true_target

    for t in RATIO_THRESHOLDS:
        flag, pred = ratio_flag(norms, t)
        row[f"ratio<{t}"] = flag and pred == true_target

    rows.append(row)

df = pd.DataFrame(rows)
print(f"Runs loaded: {len(df)}\n")

rule_cols = [f"MAD>{t}" for t in MAD_THRESHOLDS] + [f"ratio<{t}" for t in RATIO_THRESHOLDS]


# ── Aggregate table ──────────────────────────────────────────────────────────

print("=== Detection rate (correct) by detector × dataset ===")
agg = df.groupby(["detector", "dataset"])[rule_cols].agg(["sum", "count"])
for (det, ds), grp in df.groupby(["detector", "dataset"]):
    n = len(grp)
    k = grp["K"].iloc[0]
    parts = [f"K={k}"]
    for col in rule_cols:
        c = grp[col].sum()
        parts.append(f"{col}: {c}/{n} ({100*c/n:.0f}%)")
    print(f"  {det:<16} {ds:<16}  " + "   ".join(parts))

print()


# ── Candidate target accuracy ─────────────────────────────────────────────────

print("=== Candidate target accuracy (smallest-mask = true target) ===")
for (det, ds), grp in df.groupby(["detector", "dataset"]):
    n = len(grp)
    c = grp["candidate_correct"].sum()
    print(f"  {det:<16} {ds:<16}  {c}/{n} ({100*c/n:.0f}%)")

print()


# ── Per-run detail ────────────────────────────────────────────────────────────

print("=== Per-run detail ===")
cols_show = ["dataset", "attack", "model", "detector", "K", "min/med"] + rule_cols
print(df.sort_values(["dataset", "detector", "attack", "model"])[cols_show].to_string(index=False))

print()


# ── Primary delta ─────────────────────────────────────────────────────────────

print("=== Delta: ratio<0.3 vs MAD>2.0 ===")
for (det, ds), grp in df.groupby(["detector", "dataset"]):
    n    = len(grp)
    orig = grp["MAD>2.0"].sum()
    new  = grp["ratio<0.3"].sum()
    print(f"  {det:<16} {ds:<16}  {orig}/{n} → {new}/{n}  (Δ={new-orig:+d})")
