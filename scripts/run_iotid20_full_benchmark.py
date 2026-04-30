#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON_BIN = ROOT_DIR / ".venv" / "bin" / "python"

DEVICE = "cuda"
TRAIN_EPOCHS = 100
TRAIN_BATCH_SIZE = 1024
WANDB_ENABLED = True
WANDB_SUMMARY_PROFILE = "compact"
WANDB_PROJECT = "iotid20_full_backdoor_unlearning_benchmark"
SEED = 7
SHOW_OUTPUT = True
SKIP_EXISTING = True
OUT_DIR = ROOT_DIR / "results" / "benchmark_runs" / time.strftime("%Y%m%d_%H%M%S")

ATTACKS = ["badnets", "tabdoor", "catback"]
MODELS = ["mlp", "resnet", "tabnet", "ft_transformer", "saint"]
DETECTIONS = ["none", "spectral_signatures", "mlbd", "mm_bd", "neural_cleanse", "mlbd_cso", "mmbd_cso", "nc_cso"]
UNLEARNINGS = ["oracle_retrain", "retrain", "detected_retrain", "bad_teaching", "abl", "rnp"]

TARGET_LABEL = 3
TRIGGER_FEATURES = "34,15,1"
BADNETS_TRIGGER_VALUE = 3.0
BADNETS_POISON_RATE = 0.05
TABDOOR_TRIGGER_SIZE = 3
TABDOOR_POISON_RATE = 0.02
CATBACK_POISON_RATE = 0.02


def has_sample_output(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts", {}) or {}
    suspect = artifacts.get("suspect_indices_npy")
    if suspect and Path(str(suspect)).exists():
        return True
    raw_scores = artifacts.get("raw_scores_csv")
    if raw_scores and Path(str(raw_scores)).exists():
        header = Path(str(raw_scores)).read_text(encoding="utf-8", errors="ignore").splitlines()[0]
        return "flag" in header.split(",")
    return False


def run_stage(
    stage: str,
    run_id: str,
    artifact_dir: Path,
    log_file: Path,
    attack: str,
    model: str,
    seed: int,
    poison_rate: float,
    detection: str,
    unlearning: str,
    attack_train_artifact_dir: Path | None = None,
    detection_artifact_dir: Path | None = None,
) -> bool:
    overrides = [
        "data=iotid20",
        f"model={model}",
        f"attack={attack}",
        f"attack.poison_rate={poison_rate}",
        f"attack.target_label={TARGET_LABEL}",
        f"attack.seed={seed}",
        f"attack.trigger_features=[{TRIGGER_FEATURES}]",
        f"detection={detection}",
        f"unlearning={unlearning}",
        f"seed={seed}",
        f"train.device={DEVICE}",
        f"train.epochs={TRAIN_EPOCHS}",
        f"train.batch_size={TRAIN_BATCH_SIZE}",
        f"pipeline.stage={stage}",
        f"pipeline.run_id={run_id}",
        f"pipeline.artifact_dir={artifact_dir}",
        f"pipeline.skip_existing={'true' if SKIP_EXISTING else 'false'}",
        f"wandb.enabled={'true' if WANDB_ENABLED else 'false'}",
        f"wandb.project={WANDB_PROJECT}",
        f"wandb.summary_profile={WANDB_SUMMARY_PROFILE}",
        f"hydra.run.dir={OUT_DIR / 'hydra' / stage / run_id}",
    ]

    if attack_train_artifact_dir is not None:
        overrides.append(f"pipeline.attack_train_artifact_dir={attack_train_artifact_dir}")
    if detection_artifact_dir is not None:
        overrides.append(f"pipeline.detection_artifact_dir={detection_artifact_dir}")
    if attack == "badnets":
        overrides.append(f"attack.trigger_value={BADNETS_TRIGGER_VALUE}")
    elif attack == "tabdoor":
        overrides.append(f"attack.trigger_size={TABDOOR_TRIGGER_SIZE}")
    elif attack == "catback":
        overrides.append(f"attack.device={DEVICE}")

    cmd = [str(PYTHON_BIN), str(ROOT_DIR / "run.py"), *overrides]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Stage {stage}: {run_id}")

    if SHOW_OUTPUT:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        with log_file.open("w", encoding="utf-8") as handle:
            for line in proc.stdout:
                sys.stdout.write(line)
                handle.write(line)
        return proc.wait() == 0

    with log_file.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    return completed.returncode == 0


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = OUT_DIR / "logs"
    artifact_root = OUT_DIR / "artifacts"
    hydra_root = OUT_DIR / "hydra"
    log_dir.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    hydra_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output dir: {OUT_DIR}")
    print("[INFO] Clean controls: deferred; benchmark main mode is attacked")
    print(f"[INFO] Seed: {SEED}")

    started = time.time()
    for attack in ATTACKS:
        if attack == "badnets":
            poison_rate = BADNETS_POISON_RATE
        elif attack == "tabdoor":
            poison_rate = TABDOOR_POISON_RATE
        elif attack == "catback":
            poison_rate = CATBACK_POISON_RATE
        else:
            print(f"[WARN] Unknown or unsupported ATTACK '{attack}', skipping")
            continue

        for model in MODELS:
            attack_train_run_id = f"mode-attacked__attack-{attack}__seed{SEED}__model-{model}"
            attack_train_artifact_dir = artifact_root / "attack_train" / attack_train_run_id
            attack_train_log = log_dir / f"{attack_train_run_id}__stage-attack_train.log"

            if not run_stage(
                "attack_train",
                attack_train_run_id,
                attack_train_artifact_dir,
                attack_train_log,
                attack,
                model,
                SEED,
                poison_rate,
                "none",
                "none",
            ):
                print(f"[WARN] Attack/train failed for {attack_train_run_id}")
                continue

            for detection in DETECTIONS:
                detection_run_id = f"{attack_train_run_id}__det-{detection}"
                detection_artifact_dir = artifact_root / "detection" / detection_run_id
                detection_log = log_dir / f"{detection_run_id}__stage-detection.log"

                if not run_stage(
                    "detection",
                    detection_run_id,
                    detection_artifact_dir,
                    detection_log,
                    attack,
                    model,
                    SEED,
                    poison_rate,
                    detection,
                    "none",
                    attack_train_artifact_dir,
                ):
                    print(f"[WARN] Detection failed for {detection_run_id}")
                    continue

                for unlearning in UNLEARNINGS:
                    if unlearning == "rnp" and model not in {"mlp", "resnet"}:
                        continue
                    if unlearning in {"oracle_retrain", "retrain", "abl", "rnp"} and detection != "none":
                        continue
                    if unlearning in {"detected_retrain", "bad_teaching"} and detection == "none":
                        continue
                    if unlearning in {"detected_retrain", "bad_teaching"} and not has_sample_output(detection_artifact_dir / "summary.json"):
                        continue

                    unlearning_run_id = f"{detection_run_id}__unlearn-{unlearning}"
                    unlearning_artifact_dir = artifact_root / "unlearning" / unlearning_run_id
                    unlearning_log = log_dir / f"{unlearning_run_id}__stage-unlearning.log"

                    if not run_stage(
                        "unlearning",
                        unlearning_run_id,
                        unlearning_artifact_dir,
                        unlearning_log,
                        attack,
                        model,
                        SEED,
                        poison_rate,
                        detection,
                        unlearning,
                        attack_train_artifact_dir,
                        detection_artifact_dir,
                    ):
                        print(f"[WARN] Unlearning failed for {unlearning_run_id}")

    elapsed = int(time.time() - started)
    print(f"[INFO] Benchmark completed in {elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())