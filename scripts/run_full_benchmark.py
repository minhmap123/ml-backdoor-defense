#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any


ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON_BIN = ROOT_DIR / ".venv" / "bin" / "python"

# Shared config 
SEED = 7
DEVICE = "cuda"
SHOW_OUTPUT = True
SKIP_EXISTING = True

# Dataset configurations - extend this dict to add new datasets
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "iotid20": {
        "data_config": "iotid20",
        "train_epochs": 100,
        "train_batch_size": 1024,
        "wandb_project": "iotid20_full_detection_benchmark",
        "target_label": 3,
        "trigger_features": "34,15,1",
        "badnets_trigger_value": 3.0,
        "badnets_poison_rate": 0.05,
        "tabdoor_trigger_size": 3,
        "tabdoor_poison_rate": 0.02,
        "catback_poison_rate": 0.02,
    },
    "cic_ids2018": {
        "data_config": "cic_ids2018",
        "train_epochs": 100,
        "train_batch_size": 4096,
        "wandb_project": "cic_ids2018_full_detection_benchmark",
        "target_label": 0,           # Benign — attacker misclassifies attacks as benign
        "trigger_features": "1,15,30",  # Flow Duration, Flow IAT Mean, Pkt Len Mean (0-based)
        "badnets_trigger_value": 3.0,
        "badnets_poison_rate": 0.05,
        "tabdoor_trigger_size": 3,
        "tabdoor_poison_rate": 0.02,
        "catback_poison_rate": 0.02,
    },
    "cic_ids2017": {
        "data_config": "cic_ids2017",
        "train_epochs": 100,
        "train_batch_size": 4096,
        "wandb_project": "cic_ids2017_full_detection_benchmark",
        "target_label": 0,           # Benign — attacker misclassifies attacks as benign
        "trigger_features": "0,13,33",  # Flow Duration, Flow IAT Mean, Pkt Len Mean (post-FS)
        "badnets_trigger_value": 3.0,
        "badnets_poison_rate": 0.05,
        "tabdoor_trigger_size": 3,
        "tabdoor_poison_rate": 0.02,
        "catback_poison_rate": 0.02,
    },
    "cic_iot_2023": {
        "data_config": "cic_iot_2023",
        "train_epochs": 100,
        "train_batch_size": 4096,
        "wandb_project": "cic_iot_2023_full_detection_benchmark",
        "target_label": 0,           # Benign — attacker misclassifies attacks as benign
        "trigger_features": "3,30,28",  # Rate, IAT, AVG packet size (post-FS 33 features)
        "badnets_trigger_value": 3.0,
        "badnets_poison_rate": 0.05,
        "tabdoor_trigger_size": 3,
        "tabdoor_poison_rate": 0.02,
        "catback_poison_rate": 0.02,
    },
}

DEFAULT_DATASET = "iotid20"

ATTACKS = ["catback", "tabdoor", "badnets"]
MODELS = ["mlp", "resnet", "tabnet", "ft_transformer", "saint"]
DETECTIONS = ["mlbd", "mm_bd", "neural_cleanse", "pt_red", "mlbd_cso", "mmbd_cso", "nc_cso", "pt_red_cso"]


def run_stage(
    stage: str,
    run_id: str,
    artifact_dir: Path,
    log_file: Path,
    attack: str,
    model: str,
    poison_rate: float,
    detection: str,
    dataset_cfg: Dict[str, Any],
    attack_train_artifact_dir: Path | None = None,
) -> bool:
    overrides = [
        f"data={dataset_cfg['data_config']}",
        f"model={model}",
        f"attack={attack}",
        f"attack.poison_rate={poison_rate}",
        f"attack.target_label={dataset_cfg['target_label']}",
        f"attack.seed={SEED}",
        f"detection={detection}",
        f"seed={SEED}",
        f"train.device={DEVICE}",
        f"train.epochs={dataset_cfg['train_epochs']}",
        f"train.batch_size={dataset_cfg['train_batch_size']}",
        "train.class_weight_mode=balanced",
        "train.selection_metric=clean/val/f1",
        f"pipeline.stage={stage}",
        f"pipeline.run_id={run_id}",
        f"pipeline.artifact_dir={artifact_dir}",
        f"pipeline.skip_existing={'true' if SKIP_EXISTING else 'false'}",
        f"wandb.enabled=true",
        f"wandb.project={dataset_cfg['wandb_project']}",
        f"wandb.summary_profile=compact",
        f"hydra.run.dir={artifact_dir.parent.parent / 'hydra' / stage / run_id}",
    ]

    if attack_train_artifact_dir is not None:
        overrides.append(f"pipeline.attack_train_artifact_dir={attack_train_artifact_dir}")
    if attack == "badnets":
        overrides.append(f"attack.trigger_features=[{dataset_cfg['trigger_features']}]")
        overrides.append(f"attack.trigger_value={dataset_cfg['badnets_trigger_value']}")
    elif attack == "tabdoor":
        overrides.append(f"attack.trigger_features=[{dataset_cfg['trigger_features']}]")
        overrides.append(f"attack.trigger_size={dataset_cfg['tabdoor_trigger_size']}")
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
        ok = proc.wait() == 0
    else:
        with log_file.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
        ok = completed.returncode == 0

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Full Detection Benchmark Runner")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        choices=list(DATASET_CONFIGS.keys()),
        help=f"Dataset to benchmark (default: {DEFAULT_DATASET})",
    )
    args = parser.parse_args()

    if args.dataset not in DATASET_CONFIGS:
        print(f"[ERROR] Unknown dataset: {args.dataset}")
        print(f"[ERROR] Available: {', '.join(DATASET_CONFIGS.keys())}")
        return 1

    dataset_cfg = DATASET_CONFIGS[args.dataset]
    out_dir = Path(ROOT_DIR) / "results" / "benchmark_runs" / args.dataset / time.strftime("%Y%m%d_%H%M%S")
    log_dir = out_dir / "logs"
    artifact_root = out_dir / "artifacts"
    log_dir.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Dataset: {args.dataset}")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Seed: {SEED}")

    started = time.time()

    # Build poison rates mapping
    poison_rates = {
        "badnets": dataset_cfg["badnets_poison_rate"],
        "tabdoor": dataset_cfg["tabdoor_poison_rate"],
        "catback": dataset_cfg["catback_poison_rate"],
    }

    for attack in ATTACKS:
        poison_rate = poison_rates.get(attack)
        if poison_rate is None:
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
                poison_rate,
                "none",
                dataset_cfg,
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
                    poison_rate,
                    detection,
                    dataset_cfg,
                    attack_train_artifact_dir,
                ):
                    print(f"[WARN] Detection failed for {detection_run_id}")

    elapsed = int(time.time() - started)
    print(f"[INFO] Benchmark completed in {elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
