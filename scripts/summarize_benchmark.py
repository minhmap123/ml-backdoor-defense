#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_ROOT = ROOT_DIR / "results" / "benchmark_runs"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    return None


def num(summary: dict[str, Any], *keys: str) -> float | None:
    for source in (summary.get("summary_metrics", {}), summary):
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = as_float(source.get(key))
            if value is not None:
                return value
    return None


def text(summary: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = summary.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def resolve_paths(input_path: Path) -> tuple[Path, Path]:
    input_path = input_path.expanduser().resolve()
    if (input_path / "artifacts").exists():
        return input_path, input_path / "artifacts"
    if (input_path / "attack_train").exists() or (input_path / "detection").exists():
        return input_path.parent, input_path
    candidates = sorted((path for path in input_path.glob("*/artifacts") if path.is_dir()), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No benchmark run found under {input_path}")
    artifact_root = candidates[-1]
    return artifact_root.parent, artifact_root


def dataset_name(summary: dict[str, Any]) -> str | None:
    context = summary.get("context", {}) if isinstance(summary.get("context"), dict) else {}
    resolved_cfg = summary.get("resolved_cfg", {}) if isinstance(summary.get("resolved_cfg"), dict) else {}
    data_cfg = resolved_cfg.get("data")
    data_name = data_cfg.get("name") if isinstance(data_cfg, dict) else data_cfg
    return text(summary, "dataset_name", "dataset", "data_name") or text(context, "dataset_name") or text(resolved_cfg, "dataset_name") or (str(data_name) if data_name else None)


def class_name(summary: dict[str, Any], class_index: Any) -> str | None:
    if class_index is None:
        return None
    context = summary.get("context", {}) if isinstance(summary.get("context"), dict) else {}
    class_names = context.get("class_names") if isinstance(context.get("class_names"), list) else None
    try:
        index = int(class_index)
    except (TypeError, ValueError):
        return str(class_index)
    if class_names is not None and 0 <= index < len(class_names):
        return str(class_names[index])
    return str(index)


def attack_row(summary_path: Path) -> dict[str, Any]:
    summary = load_json(summary_path)
    return {
        "dataset_name": dataset_name(summary),
        "attack_train_run_id": summary.get("run_id"),
        "attack_train_artifact_dir": str(Path(summary.get("artifact_dir") or summary_path.parent).resolve()),
        "attack_train_summary_json": str(summary_path),
        "attack": summary.get("attack"),
        "model": summary.get("model"),
        "seed": summary.get("seed"),
        "train_status": summary.get("status"),
        "clean_test_accuracy": num(summary, "clean/test/accuracy"),
        "clean_test_f1": num(summary, "clean/test/f1", "clean/test/f1_macro"),
        "clean_val_accuracy_best": num(summary, "clean/val_accuracy_best"),
        "clean_val_f1_best": num(summary, "clean/val_f1_best"),
        "backdoor_asr": num(summary, "backdoor/asr"),
        "backdoor_accuracy": num(summary, "backdoor/accuracy"),
        "poisoned_train_samples": num(summary, "poisoned_train_samples"),
        "train_runtime_sec": num(summary, "runtime_sec"),
        "model_total_parameters": num(summary, "model/total_parameters", "total_params"),
    }


def infer_attack_run_id(detection_run_id: Any) -> str | None:
    if detection_run_id is None:
        return None
    text = str(detection_run_id)
    return text.split("__det-", 1)[0] if "__det-" in text else None


def make_attack_index(training_df: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    by_path: dict[str, pd.Series] = {}
    by_run_id: dict[str, pd.Series] = {}
    for _, row in training_df.iterrows():
        if pd.notna(row.get("attack_train_run_id")):
            by_run_id[str(row.get("attack_train_run_id"))] = row
        if pd.notna(row.get("attack_train_artifact_dir")):
            by_path[str(Path(str(row.get("attack_train_artifact_dir"))).expanduser().resolve())] = row
    return by_path, by_run_id


def lookup_attack_row(summary: dict[str, Any], by_path: dict[str, pd.Series], by_run_id: dict[str, pd.Series]) -> pd.Series | None:
    source_path = summary.get("source_attack_train_artifact_dir")
    if source_path is not None:
        found = by_path.get(str(Path(str(source_path)).expanduser().resolve()))
        if found is not None:
            return found
    run_id = infer_attack_run_id(summary.get("run_id"))
    return by_run_id.get(run_id) if run_id is not None else None


def detection_row(summary_path: Path, by_path: dict[str, pd.Series], by_run_id: dict[str, pd.Series]) -> dict[str, Any]:
    summary = load_json(summary_path)
    attack = lookup_attack_row(summary, by_path, by_run_id)

    context = summary.get("context", {}) if isinstance(summary.get("context"), dict) else {}
    attack_metadata = context.get("attack_metadata", {}) if isinstance(context.get("attack_metadata"), dict) else {}
    resolved_cfg = summary.get("resolved_cfg", {}) if isinstance(summary.get("resolved_cfg"), dict) else {}

    true_is_infected = as_bool(context.get("true_is_infected"))
    if true_is_infected is None and attack is not None:
        true_is_infected = bool(as_float(attack.get("poisoned_train_samples")) or 0.0)

    predicted_is_infected = as_bool(summary.get("predicted_is_infected"))
    true_target = context.get("true_target_class", context.get("attack_target_label"))
    predicted_target = summary.get("predicted_target_class")
    candidate_target = summary.get("candidate_target_class")

    row = {
        "dataset_name": dataset_name(summary),
        "detection_run_id": summary.get("run_id"),
        "detector": summary.get("detector_name") or resolved_cfg.get("name"),
        "detection_status": summary.get("status"),
        "seed": summary.get("seed"),
        "attack_train_run_id": None,
        "attack": None,
        "model": None,
        "predicted_is_infected": predicted_is_infected,
        "true_is_infected": true_is_infected,
        "detection_correct": None if predicted_is_infected is None or true_is_infected is None else bool(predicted_is_infected == true_is_infected),
        "predicted_target_class": class_name(summary, predicted_target),
        "candidate_target_class": class_name(summary, candidate_target),
        "true_target_class": class_name(summary, true_target),
        "target_class_correct": None if predicted_target is None or true_target is None else int(predicted_target) == int(true_target),
        "candidate_target_class_correct": None if candidate_target is None or true_target is None else int(candidate_target) == int(true_target),
        "candidate_target_score": num(summary, "detection/candidate_target_score", "candidate_target_score"),
        "decision_score": num(summary, "detection/decision_score", "decision_score"),
        "decision_threshold": num(summary, "detection/decision_threshold", "decision_threshold"),
        "decision_margin": num(summary, "detection/decision_margin", "decision_margin"),
        "detector_runtime_sec": num(summary, "detection/runtime_sec", "runtime_sec"),
        "detection_stage_runtime_sec": num(summary, "stage_runtime_sec"),
        "clean_test_accuracy": None,
        "clean_test_f1": None,
        "backdoor_asr": None,
        "backdoor_accuracy": None,
        "poisoned_train_samples": None,
        "train_runtime_sec": None,
        "model_total_parameters": None,
        "source_attack_train_artifact_dir": str(Path(str(summary.get("source_attack_train_artifact_dir", ""))).expanduser().resolve()) if summary.get("source_attack_train_artifact_dir") else None,
        "detection_summary_json": str(summary_path),
        "attack_target_label": class_name(summary, attack_metadata.get("target_label")),
        "observed_backdoor_asr": as_float(attack_metadata.get("observed_backdoor_asr")),
    }

    if attack is not None:
        for key in ("attack_train_run_id", "attack", "model", "clean_test_accuracy", "clean_test_f1", "backdoor_asr", "backdoor_accuracy", "poisoned_train_samples", "train_runtime_sec", "model_total_parameters"):
            row[key] = attack.get(key)

    return row


def load_training(artifact_root: Path) -> pd.DataFrame:
    return pd.DataFrame([attack_row(path) for path in sorted((artifact_root / "attack_train").glob("*/summary.json"))])


def load_detection(artifact_root: Path, training_df: pd.DataFrame) -> pd.DataFrame:
    by_path, by_run_id = make_attack_index(training_df)
    return pd.DataFrame([detection_row(path, by_path, by_run_id) for path in sorted((artifact_root / "detection").glob("*/summary.json"))])


def mean_bool(series: pd.Series) -> float | None:
    values = series.dropna()
    return None if values.empty else float(values.astype(bool).mean())


def present_columns(df: pd.DataFrame, preferred: list[str]) -> list[str]:
    return [column for column in preferred if column in df.columns and df[column].notna().any()]


def summarize_training(training_df: pd.DataFrame) -> pd.DataFrame:
    if training_df.empty:
        return pd.DataFrame()
    group_cols = present_columns(training_df, ["dataset_name", "attack", "model"]) or ["attack", "model"]
    rows = []
    for keys, group in training_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "runs": int(len(group)),
                "clean_test_accuracy_mean": group["clean_test_accuracy"].mean(),
                "clean_test_f1_mean": group["clean_test_f1"].mean(),
                "backdoor_asr_mean": group["backdoor_asr"].mean(),
                "backdoor_accuracy_mean": group["backdoor_accuracy"].mean(),
                "train_runtime_sec_mean": group["train_runtime_sec"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def summarize_detection(detection_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if detection_df.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in detection_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        true_attacks = group[group["true_is_infected"] == True]
        detected_attacks = true_attacks["predicted_is_infected"].fillna(False).astype(bool).sum()
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "runs": int(len(group)),
                "total_attacks": int(len(true_attacks)),
                "detected_attacks": int(detected_attacks),
                "attack_detect_rate": float(detected_attacks / max(len(true_attacks), 1)) if len(true_attacks) else None,
                "detection_accuracy": mean_bool(group["detection_correct"]),
                "predicted_target_accuracy": mean_bool(true_attacks["target_class_correct"]) if not true_attacks.empty else None,
                "candidate_target_accuracy": mean_bool(true_attacks["candidate_target_class_correct"]) if not true_attacks.empty else None,
                "clean_test_accuracy_mean": group["clean_test_accuracy"].mean(),
                "clean_test_f1_mean": group["clean_test_f1"].mean(),
                "backdoor_asr_mean": group["backdoor_asr"].mean(),
                "decision_margin_mean": group["decision_margin"].mean(),
                "detector_runtime_sec_mean": group["detector_runtime_sec"].mean(),
                "detector_runtime_sec_sum": group["detector_runtime_sec"].sum(),
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def write_outputs(
    out_dir: Path,
    training_df: pd.DataFrame,
    detection_df: pd.DataFrame,
    training_summary: pd.DataFrame,
    detector_summary: pd.DataFrame,
    detector_attack_summary: pd.DataFrame,
    ) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(out_dir / "training_runs.csv", index=False)
    detection_df.to_csv(out_dir / "detection_runs.csv", index=False)
    training_summary.to_csv(out_dir / "training_summary_by_attack_model.csv", index=False)
    detector_summary.to_csv(out_dir / "detection_summary_by_detector.csv", index=False)
    detector_attack_summary.to_csv(out_dir / "detection_summary_by_detector_attack.csv", index=False)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_dir", nargs="?", default=str(DEFAULT_BENCHMARK_ROOT))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-rows", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_dir, artifact_root = resolve_paths(Path(args.benchmark_dir))
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    training_df = load_training(artifact_root)
    detection_df = load_detection(artifact_root, training_df)
    training_summary = summarize_training(training_df)
    detector_cols = present_columns(detection_df, ["dataset_name", "detector"]) or ["detector"]
    detector_attack_cols = present_columns(detection_df, ["dataset_name", "detector", "attack"]) or ["detector", "attack"]
    detector_summary = summarize_detection(detection_df, detector_cols)
    detector_attack_summary = summarize_detection(detection_df, detector_attack_cols)
    out_dir = write_outputs(
        out_dir=Path(args.out_dir) if args.out_dir else run_dir / "reports",
        training_df=training_df,
        detection_df=detection_df,
        training_summary=training_summary,
        detector_summary=detector_summary,
        detector_attack_summary=detector_attack_summary,
    )

    print(f"Source: {run_dir}")
    print(f"CSV output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
