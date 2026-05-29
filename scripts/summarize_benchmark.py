#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BENCHMARK_ROOT = Path("results/benchmark_runs")

# ── Helpers ──────────────────────────────────────────────────────────────────

def num(s, *keys):
    for src in (s.get("summary_metrics") or {}, s):
        if not isinstance(src, dict):
            continue
        for k in keys:
            v = src.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
    return None

def class_label(summary, idx):
    if idx is None:
        return None
    names = (summary.get("context") or {}).get("class_names")
    try:
        i = int(idx)
        return names[i] if isinstance(names, list) and 0 <= i < len(names) else str(i)
    except (TypeError, ValueError):
        return str(idx)

def dataset_of(s):
    for d in (s, s.get("context") or {}, s.get("resolved_cfg") or {}, s.get("dataset_metadata") or {}):
        for k in ("dataset_name", "dataset", "data_name"):
            v = d.get(k)
            if v not in (None, ""):
                return str(v)
    data = (s.get("resolved_cfg") or {}).get("data")
    v = data.get("name") if isinstance(data, dict) else data
    return str(v) if v else None

def agg_detection(df, group_cols):
    """Aggregate detection runs by group_cols. Adds std across seeds when multi-seed data exists.
    Sorted by attack_detect_rate desc (best first)."""
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        attacked = g[g["true_is_infected"] == True]
        detected = int(attacked["predicted_is_infected"].fillna(False).astype(bool).sum())
        # Per-seed TPR -> std reflects stability across seeds, not across (model, run) within a seed.
        if not attacked.empty and "seed" in attacked.columns:
            per_seed = (attacked.groupby("seed", dropna=False)
                                .apply(lambda x: x["predicted_is_infected"].fillna(False).astype(bool).mean()))
            tpr_std = float(per_seed.std()) if len(per_seed) > 1 else None
            n_seeds = int(per_seed.count())
        else:
            tpr_std, n_seeds = None, 0
        rows.append({
            **dict(zip(group_cols, keys)),
            "runs":                     len(g),
            "n_seeds":                  n_seeds,
            "total_attacks":            len(attacked),
            "detected_attacks":         detected,
            "attack_detect_rate":       detected / len(attacked) if len(attacked) else None,
            "attack_detect_rate_std":   tpr_std,
            "detection_accuracy":       g["detection_correct"].dropna().astype(bool).mean() if g["detection_correct"].notna().any() else None,
            "candidate_target_accuracy": attacked["candidate_target_class_correct"].dropna().astype(bool).mean() if not attacked.empty else None,
            "backdoor_asr_mean":         g["backdoor_asr"].mean(),
            "detector_runtime_sec_mean": g["detector_runtime_sec"].mean(),
        })
    return (pd.DataFrame(rows)
              .sort_values("attack_detect_rate", ascending=False, na_position="last")
              .reset_index(drop=True))

def clean_fpr_summary(detection_df):
    """FPR per detector on clean baseline runs (attack='none'). Adds std across seeds.
    Sorted by fpr asc (best first)."""
    clean = detection_df[detection_df["attack"].fillna("") == "none"]
    if clean.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in clean.groupby(["dataset_name", "detector"], dropna=False):
        false_pos = int(g["predicted_is_infected"].fillna(False).astype(bool).sum())
        if "seed" in g.columns:
            per_seed = (g.groupby("seed", dropna=False)
                         .apply(lambda x: x["predicted_is_infected"].fillna(False).astype(bool).mean()))
            fpr_std = float(per_seed.std()) if len(per_seed) > 1 else None
            n_seeds = int(per_seed.count())
        else:
            fpr_std, n_seeds = None, 0
        rows.append({
            "dataset_name":    keys[0],
            "detector":        keys[1],
            "runs":            len(g),
            "n_seeds":         n_seeds,
            "false_positives": false_pos,
            "fpr":             false_pos / len(g) if len(g) else None,
            "fpr_std":         fpr_std,
        })
    return (pd.DataFrame(rows)
              .sort_values("fpr", ascending=True, na_position="last")
              .reset_index(drop=True))

def detector_attack_pivot(detection_df):
    """Wide pivot: rows=detector, cols=attack, cells=TPR. Adds a 'mean' column. Sorted by mean desc."""
    backdoor = detection_df[detection_df["attack"].fillna("") != "none"]
    backdoor = backdoor[backdoor["attack"].notna()]
    if backdoor.empty:
        return pd.DataFrame()
    pivot = (backdoor.groupby(["detector", "attack"])
                     .apply(lambda x: x["predicted_is_infected"].fillna(False).astype(bool).mean())
                     .unstack(fill_value=float("nan")))
    pivot["mean"] = pivot.mean(axis=1, skipna=True)
    return pivot.sort_values("mean", ascending=False)

def _round_floats(df, decimals=4):
    """Round all float columns to N decimal places. Int columns untouched."""
    if df.empty:
        return df
    return df.round({c: decimals for c in df.select_dtypes("float").columns})

# ── Per-run processing ──────────────────────────────────────────────────────

def process_run(run_dir):
    """Scan one run-dir, write CSVs into <run_dir>/reports/. Returns (training_df, detection_df)."""
    artifact_root = run_dir / "artifacts"
    out_dir = run_dir / "reports"

    # Load training runs
    training_rows = []
    for p in sorted((artifact_root / "attack_train").glob("*/summary.json")):
        s = json.load(open(p))
        training_rows.append({
            "dataset_name":              dataset_of(s),
            "attack_train_run_id":       s.get("run_id"),
            "attack_train_artifact_dir": str(Path(s.get("artifact_dir") or p.parent).resolve()),
            "attack":                    s.get("attack"),
            "model":                     s.get("model"),
            "seed":                      s.get("seed"),
            "clean_test_accuracy":       num(s, "clean/test/accuracy"),
            "clean_test_f1":             num(s, "clean/test/f1", "clean/test/f1_macro"),
            "backdoor_asr":              num(s, "backdoor/asr"),
            "backdoor_accuracy":         num(s, "backdoor/accuracy"),
            "train_runtime_sec":         num(s, "runtime_sec"),
        })
    training_df = pd.DataFrame(training_rows)

    # Build attack index for join
    by_path, by_run_id = {}, {}
    for _, row in training_df.iterrows():
        if pd.notna(row.get("attack_train_run_id")):
            by_run_id[str(row["attack_train_run_id"])] = row
        if pd.notna(row.get("attack_train_artifact_dir")):
            by_path[str(Path(str(row["attack_train_artifact_dir"])).resolve())] = row

    # Load detection runs
    detection_rows = []
    for p in sorted((artifact_root / "detection").glob("*/summary.json")):
        s = json.load(open(p))
        ctx = s.get("context") or {}

        attack = None
        src = s.get("source_attack_train_artifact_dir")
        if src:
            attack = by_path.get(str(Path(str(src)).resolve()))
        if attack is None:
            run_id = s.get("run_id", "")
            prefix = run_id.split("__det-", 1)[0] if "__det-" in run_id else None
            if prefix:
                attack = by_run_id.get(prefix)

        true_infected = ctx.get("true_is_infected")
        if true_infected is None and attack is not None:
            true_infected = bool(attack.get("backdoor_asr") or 0)
        elif true_infected is not None:
            true_infected = bool(true_infected)

        pred_infected = s.get("predicted_is_infected")
        if pred_infected is not None:
            pred_infected = bool(pred_infected)

        true_target = ctx.get("true_target_class", ctx.get("attack_target_label"))
        pred_target = s.get("predicted_target_class")
        cand_target = s.get("candidate_target_class")

        row = {
            "dataset_name":    dataset_of(s),
            "detector":        s.get("detector_name") or (s.get("resolved_cfg") or {}).get("name"),
            "seed":            s.get("seed"),
            "attack":          None,
            "model":           None,
            "predicted_is_infected":          pred_infected,
            "true_is_infected":               true_infected,
            "detection_correct":              None if pred_infected is None or true_infected is None else pred_infected == true_infected,
            "predicted_target_class":         class_label(s, pred_target),
            "candidate_target_class":         class_label(s, cand_target),
            "true_target_class":              class_label(s, true_target),
            "target_class_correct":           None if pred_target is None or true_target is None else int(pred_target) == int(true_target),
            "candidate_target_class_correct": None if cand_target is None or true_target is None else int(cand_target) == int(true_target),
            "decision_score":                 num(s, "detection/decision_score", "decision_score"),
            "decision_threshold":             num(s, "detection/decision_threshold", "decision_threshold"),
            "detector_runtime_sec":           num(s, "detection/runtime_sec", "runtime_sec"),
            "clean_test_accuracy":            None,
            "backdoor_asr":                   None,
        }
        if attack is not None:
            for k in ("attack", "model", "clean_test_accuracy", "backdoor_asr"):
                row[k] = attack.get(k)
        detection_rows.append(row)

    detection_df = pd.DataFrame(detection_rows)

    # Training summary — sort by attack then backdoor_asr_mean desc (best model per attack first).
    train_summary_rows = []
    for keys, g in training_df.groupby(["dataset_name", "attack", "model"], dropna=False):
        train_summary_rows.append({
            "dataset_name": keys[0], "attack": keys[1], "model": keys[2],
            "runs":                     len(g),
            "clean_test_accuracy_mean": g["clean_test_accuracy"].mean(),
            "clean_test_f1_mean":       g["clean_test_f1"].mean(),
            "backdoor_asr_mean":        g["backdoor_asr"].mean(),
            "train_runtime_sec_mean":   g["train_runtime_sec"].mean(),
        })
    train_summary_df = (pd.DataFrame(train_summary_rows)
                          .sort_values(["dataset_name", "attack", "backdoor_asr_mean"],
                                       ascending=[True, True, False],
                                       na_position="last")
                          .reset_index(drop=True))

    det_cols     = [c for c in ["dataset_name", "detector"]           if c in detection_df and detection_df[c].notna().any()] or ["detector"]
    det_atk_cols = [c for c in ["dataset_name", "detector", "attack"] if c in detection_df and detection_df[c].notna().any()] or ["detector", "attack"]

    # Write CSVs -- round floats before saving for readability.
    out_dir.mkdir(parents=True, exist_ok=True)
    _round_floats(training_df).to_csv(out_dir / "training_runs.csv", index=False)
    _round_floats(detection_df).to_csv(out_dir / "detection_runs.csv", index=False)
    _round_floats(train_summary_df).to_csv(out_dir / "training_summary_by_attack_model.csv", index=False)
    _round_floats(agg_detection(detection_df, det_cols)).to_csv(out_dir / "detection_summary_by_detector.csv", index=False)
    _round_floats(agg_detection(detection_df, det_atk_cols)).to_csv(out_dir / "detection_summary_by_detector_attack.csv", index=False)

    fpr_df = clean_fpr_summary(detection_df)
    if not fpr_df.empty:
        _round_floats(fpr_df).to_csv(out_dir / "clean_baseline_fpr.csv", index=False)

    pivot_df = detector_attack_pivot(detection_df)
    if not pivot_df.empty:
        _round_floats(pivot_df).to_csv(out_dir / "detector_x_attack_tpr.csv")  # keep detector index

    return training_df, detection_df

# ── Console summary ─────────────────────────────────────────────────────────

def print_summary(training, detection):
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    print(f"Datasets:        {training['dataset_name'].nunique() if not training.empty else 0}")
    print(f"Training runs:   {len(training)}")
    print(f"Detection runs:  {len(detection)}")
    print()

    if training.empty or detection.empty:
        return

    print("Per dataset:  (clean_acc = trained on clean baseline; asr = ASR of backdoor models)")
    print(f"  {'dataset':<18s}  {'clean':>5s}  {'attack':>6s}  {'clean_acc':>9s}  {'asr':>5s}")
    fmt = lambda v: f"{v:.3f}" if pd.notna(v) else "    —"
    for ds, g in training.groupby("dataset_name", dropna=False):
        n_clean        = (g["attack"] == "none").sum()
        n_backdoor     = (g["attack"] != "none").sum()
        avg_clean_acc  = g.loc[g["attack"] == "none", "clean_test_accuracy"].mean()
        avg_asr        = g.loc[g["attack"] != "none", "backdoor_asr"].mean()
        print(f"  {str(ds):<18s}  {n_clean:>5d}  {n_backdoor:>6d}  {fmt(avg_clean_acc):>9s}  {fmt(avg_asr):>5s}")
    print()

    backdoor = detection[detection["attack"].fillna("") != "none"]
    if not backdoor.empty:
        # Per-detector mean + std across seeds (each seed has 5 models x 3 attacks = 15 runs)
        per_seed = (backdoor.groupby(["detector", "seed"])
                            .apply(lambda g: g["predicted_is_infected"].fillna(False).astype(bool).mean()))
        det_stats = (per_seed.groupby("detector")
                             .agg(["mean", "std", "count"])
                             .sort_values("mean", ascending=False))
        print("Detection TPR (backdoor, mean ± std across seeds):")
        for det, row in det_stats.iterrows():
            std_str = f"± {row['std']:.3f}" if pd.notna(row["std"]) else "(1 seed)"
            print(f"  {str(det):<18s}  {row['mean']:.3f}  {std_str}")
        print()

    fpr = clean_fpr_summary(detection)
    if not fpr.empty:
        # fpr is already aggregated per (dataset, detector). Average across datasets and report its std.
        det_fpr = (fpr.groupby("detector")
                      .agg(fpr_mean=("fpr", "mean"), fpr_std=("fpr", "std"))
                      .sort_values("fpr_mean"))
        print("Detection FPR (clean baseline, mean ± std across seeds, sorted asc):")
        for det, row in det_fpr.iterrows():
            std_str = f"± {row['fpr_std']:.3f}" if pd.notna(row["fpr_std"]) else "(1 dataset)"
            print(f"  {str(det):<18s}  {row['fpr_mean']:.3f}  {std_str}")
        print()

    pivot = detector_attack_pivot(detection)
    if not pivot.empty:
        print("Detection TPR (detector × attack, sorted by mean desc):")
        attacks = list(pivot.columns)
        print(f"  {'detector':<18s}" + "".join(f"{a:>10s}" for a in attacks))
        for det, row in pivot.iterrows():
            cells = "".join(f"{row[a]:>10.4f}" if pd.notna(row[a]) else f"{'—':>10s}" for a in attacks)
            print(f"  {str(det):<18s}{cells}")
        print()

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark runs into CSV reports + console table")
    parser.add_argument(
        "--run-dir", action="append", default=None,
        help="Specific run dir to summarize (e.g. results/benchmark_runs/iotid20/20260101_120000). "
             "Can repeat. Default: scan all run dirs under results/benchmark_runs/.",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_dirs = [Path(d) for d in args.run_dir]
    else:
        run_dirs = sorted(
            [p.parent for p in BENCHMARK_ROOT.glob("*/*/artifacts")],
            key=lambda p: p.stat().st_mtime,
        )

    if not run_dirs:
        raise SystemExit(f"No benchmark run found under {BENCHMARK_ROOT}")

    all_training, all_detection = [], []
    for run_dir in run_dirs:
        if not (run_dir / "artifacts").exists():
            print(f"[WARN] Skipping {run_dir}: no artifacts/", file=sys.stderr)
            continue
        training_df, detection_df = process_run(run_dir)
        all_training.append(training_df)
        all_detection.append(detection_df)
        print(f"Processed: {run_dir} → {run_dir / 'reports'}")

    combined_training  = pd.concat(all_training,  ignore_index=True) if all_training  else pd.DataFrame()
    combined_detection = pd.concat(all_detection, ignore_index=True) if all_detection else pd.DataFrame()

    # Cross-dataset rollup when multiple datasets
    if not combined_training.empty and combined_training["dataset_name"].nunique() > 1:
        cross_dir = BENCHMARK_ROOT / "_cross_dataset"
        cross_dir.mkdir(parents=True, exist_ok=True)
        _round_floats(agg_detection(combined_detection, ["detector", "attack"])).to_csv(
            cross_dir / "detection_by_detector_attack.csv", index=False)
        fpr_cross = clean_fpr_summary(combined_detection)
        if not fpr_cross.empty:
            _round_floats(fpr_cross).to_csv(cross_dir / "clean_baseline_fpr.csv", index=False)
        pivot_cross = detector_attack_pivot(combined_detection)
        if not pivot_cross.empty:
            _round_floats(pivot_cross).to_csv(cross_dir / "detector_x_attack_tpr.csv")
        print(f"Cross-dataset rollup: {cross_dir}")

    print_summary(combined_training, combined_detection)


if __name__ == "__main__":
    main()
