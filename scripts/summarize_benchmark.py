#!/usr/bin/env python3
import json
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
    for d in (s, s.get("context") or {}, s.get("resolved_cfg") or {}):
        for k in ("dataset_name", "dataset", "data_name"):
            v = d.get(k)
            if v not in (None, ""):
                return str(v)
    data = (s.get("resolved_cfg") or {}).get("data")
    v = data.get("name") if isinstance(data, dict) else data
    return str(v) if v else None

def agg_detection(df, group_cols):
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        attacked = g[g["true_is_infected"] == True]
        detected = int(attacked["predicted_is_infected"].fillna(False).astype(bool).sum())
        rows.append({
            **dict(zip(group_cols, keys)),
            "runs":               len(g),
            "total_attacks":      len(attacked),
            "detected_attacks":   detected,
            "attack_detect_rate": detected / len(attacked) if len(attacked) else None,
            "detection_accuracy": g["detection_correct"].dropna().astype(bool).mean() if g["detection_correct"].notna().any() else None,
            "candidate_target_accuracy": attacked["candidate_target_class_correct"].dropna().astype(bool).mean() if not attacked.empty else None,
            "backdoor_asr_mean":         g["backdoor_asr"].mean(),
            "detector_runtime_sec_mean": g["detector_runtime_sec"].mean(),
        })
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)

# ── Process each benchmark run ───────────────────────────────────────────────

artifact_roots = sorted(BENCHMARK_ROOT.glob("*/*/artifacts"), key=lambda p: p.stat().st_mtime)
if not artifact_roots:
    raise SystemExit(f"No benchmark run found under {BENCHMARK_ROOT}")

for artifact_root in artifact_roots:
    run_dir = artifact_root.parent
    out_dir = run_dir / "reports"

    # Load training runs
    training_rows = []
    for p in sorted((artifact_root / "attack_train").glob("*/summary.json")):
        s = json.load(open(p))
        training_rows.append({
            "dataset_name":            dataset_of(s),
            "attack_train_run_id":     s.get("run_id"),
            "attack_train_artifact_dir": str(Path(s.get("artifact_dir") or p.parent).resolve()),
            "attack":                  s.get("attack"),
            "model":                   s.get("model"),
            "seed":                    s.get("seed"),
            "clean_test_accuracy":     num(s, "clean/test/accuracy"),
            "clean_test_f1":           num(s, "clean/test/f1", "clean/test/f1_macro"),
            "backdoor_asr":            num(s, "backdoor/asr"),
            "backdoor_accuracy":       num(s, "backdoor/accuracy"),
            "train_runtime_sec":       num(s, "runtime_sec"),
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
        ctx  = s.get("context") or {}

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
        pred_target  = s.get("predicted_target_class")
        cand_target  = s.get("candidate_target_class")

        row = {
            "dataset_name":    dataset_of(s),
            "detector":        s.get("detector_name") or (s.get("resolved_cfg") or {}).get("name"),
            "seed":            s.get("seed"),
            "attack":          None,
            "model":           None,
            "predicted_is_infected":           pred_infected,
            "true_is_infected":                true_infected,
            "detection_correct":               None if pred_infected is None or true_infected is None else pred_infected == true_infected,
            "predicted_target_class":          class_label(s, pred_target),
            "candidate_target_class":          class_label(s, cand_target),
            "true_target_class":               class_label(s, true_target),
            "target_class_correct":            None if pred_target is None or true_target is None else int(pred_target) == int(true_target),
            "candidate_target_class_correct":  None if cand_target is None or true_target is None else int(cand_target) == int(true_target),
            "decision_score":                  num(s, "detection/decision_score", "decision_score"),
            "decision_threshold":              num(s, "detection/decision_threshold", "decision_threshold"),
            "detector_runtime_sec":            num(s, "detection/runtime_sec", "runtime_sec"),
            "clean_test_accuracy":             None,
            "backdoor_asr":                    None,
        }
        if attack is not None:
            for k in ("attack", "model", "clean_test_accuracy", "backdoor_asr"):
                row[k] = attack.get(k)
        detection_rows.append(row)

    detection_df = pd.DataFrame(detection_rows)

    # Training summary
    train_summary_rows = []
    for keys, g in training_df.groupby(["dataset_name", "attack", "model"], dropna=False):
        train_summary_rows.append({
            "dataset_name": keys[0], "attack": keys[1], "model": keys[2],
            "runs":                    len(g),
            "clean_test_accuracy_mean": g["clean_test_accuracy"].mean(),
            "clean_test_f1_mean":       g["clean_test_f1"].mean(),
            "backdoor_asr_mean":        g["backdoor_asr"].mean(),
            "train_runtime_sec_mean":   g["train_runtime_sec"].mean(),
        })
    train_summary_df = pd.DataFrame(train_summary_rows)

    det_cols     = [c for c in ["dataset_name", "detector"] if c in detection_df and detection_df[c].notna().any()] or ["detector"]
    det_atk_cols = [c for c in ["dataset_name", "detector", "attack"] if c in detection_df and detection_df[c].notna().any()] or ["detector", "attack"]

    # Write CSVs
    out_dir.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(out_dir / "training_runs.csv", index=False)
    detection_df.to_csv(out_dir / "detection_runs.csv", index=False)
    train_summary_df.to_csv(out_dir / "training_summary_by_attack_model.csv", index=False)
    agg_detection(detection_df, det_cols).to_csv(out_dir / "detection_summary_by_detector.csv", index=False)
    agg_detection(detection_df, det_atk_cols).to_csv(out_dir / "detection_summary_by_detector_attack.csv", index=False)

    print(f"Source: {run_dir}")
    print(f"CSV output: {out_dir}")
    print()
