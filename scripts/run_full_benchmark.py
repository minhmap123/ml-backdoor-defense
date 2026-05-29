#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON_BIN = ROOT_DIR / ".venv" / "bin" / "python"
DATA_CONF_DIR = ROOT_DIR / "conf" / "data"

# Shared config
# Use [7] for quick first-pass benchmark; switch to [7, 13, 42] for full multi-seed runs.
SEEDS = [7]
DEVICE = "cuda"
SHOW_OUTPUT = True
SKIP_EXISTING = True

# Artifacts safe to delete AFTER all 8 detections of a (seed, attack, model) tuple finish:
#   - data/            : datasets.npz (1-2GB for cic_iot_2023). Loaded only by same-tuple detections.
#   - checkpoint/      : model_state_dict.pt. Loaded only by same-tuple detections.
#   - attack_result/   : AttackResult pickle. Read only by same-tuple detections.
# Summarize reads only summary.json + metadata.json -> NEVER touches cleanup targets.
CLEANUP_TARGETS = ("data", "checkpoint", "attack_result")

DEFAULT_DATASET = "iotid20"

# Discover datasets from conf/data/*.yaml -- only files that contain a `benchmark:` key.
DATASET_CONFIGS = {}
for _p in sorted(DATA_CONF_DIR.glob("*.yaml")):
    _cfg = yaml.safe_load(_p.read_text())
    if "benchmark" in _cfg:
        DATASET_CONFIGS[_p.stem] = _cfg

# "none" = clean baseline (no poisoning). Placed first so clean runs finish before backdoor runs.
ATTACKS = ["none", "catback", "tabdoor", "badnets"]
MODELS = ["mlp", "resnet", "tabnet", "ft_transformer", "saint"]
DETECTIONS = ["mlbd", "mm_bd", "neural_cleanse", "pt_red", "mlbd_cso", "mmbd_cso", "nc_cso", "pt_red_cso"]


def run_stage(stage, run_id, artifact_dir, log_file, attack, model, poison_rate, detection, dataset_cfg, seed, attack_train_artifact_dir=None):
    bench = dataset_cfg["benchmark"]

    overrides = [
        f"data={dataset_cfg['name']}",
        f"model={model}",
        f"attack={attack}",
        f"attack.poison_rate={poison_rate}",
        f"attack.target_label={bench['target_label']}",
        f"attack.seed={seed}",
        f"detection={detection}",
        f"seed={seed}",
        f"train.device={DEVICE}",
        f"train.epochs={bench['train_epochs']}",
        f"train.batch_size={bench['train_batch_size']}",
        "train.class_weight_mode=balanced",
        "train.selection_metric=clean/val/f1",
        f"pipeline.stage={stage}",
        f"pipeline.run_id={run_id}",
        f"pipeline.artifact_dir={artifact_dir}",
        f"pipeline.skip_existing={'true' if SKIP_EXISTING else 'false'}",
        f"wandb.enabled=true",
        f"wandb.project={bench['wandb_project']}",
        f"wandb.summary_profile=compact",
        f"hydra.run.dir={artifact_dir.parent.parent / 'hydra' / stage / run_id}",
    ]

    if attack_train_artifact_dir is not None:
        overrides.append(f"pipeline.attack_train_artifact_dir={attack_train_artifact_dir}")
    if attack == "badnets":
        trigger_features_str = ",".join(str(x) for x in bench["badnets"]["trigger_features"])
        overrides.append(f"attack.trigger_features=[{trigger_features_str}]")
        overrides.append(f"attack.trigger_value={bench['badnets']['trigger_value']}")
    elif attack == "tabdoor":
        tabdoor_features = bench["tabdoor"].get("trigger_features")
        if tabdoor_features is None:
            overrides.append("attack.trigger_features=null")
        else:
            trigger_features_str = ",".join(str(x) for x in tabdoor_features)
            overrides.append(f"attack.trigger_features=[{trigger_features_str}]")
        overrides.append(f"attack.trigger_size={bench['tabdoor']['trigger_size']}")
    elif attack == "catback":
        overrides.append(f"attack.device={DEVICE}")
    # attack == "none": clean baseline, no trigger overrides needed

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


def cleanup_tuple_artifacts(attack_train_dir):
    """Delete heavy artifacts once all 8 detections of a (seed, attack, model) tuple are done.
    Kept files: summary.json, metadata.json, model_cfg.json, train_cfg.json, model_metrics.json."""
    for target in CLEANUP_TARGETS:
        path = attack_train_dir / target
        if path.exists():
            shutil.rmtree(path)
    print(f"[CLEANUP] purged {CLEANUP_TARGETS} in {attack_train_dir.name}")


def run_dataset(ds_name, dataset_cfg, cleanup_artifacts=False):
    bench = dataset_cfg["benchmark"]
    out_dir = Path(ROOT_DIR) / "results" / "benchmark_runs" / ds_name / time.strftime("%Y%m%d_%H%M%S")
    log_dir = out_dir / "logs"
    artifact_root = out_dir / "artifacts"
    log_dir.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Dataset: {ds_name}")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Seeds: {SEEDS}  cleanup_artifacts={cleanup_artifacts}")

    poison_rates = {
        "none":    0.0,
        "badnets": bench["badnets"]["poison_rate"],
        "tabdoor": bench["tabdoor"]["poison_rate"],
        "catback": bench["catback"]["poison_rate"],
    }

    for seed in SEEDS:
        for attack in ATTACKS:
            poison_rate = poison_rates.get(attack)
            if poison_rate is None:
                print(f"[WARN] Unknown ATTACK '{attack}', skipping")
                continue

            for model in MODELS:
                if attack == "none":
                    attack_train_run_id = f"mode-clean__seed{seed}__model-{model}"
                else:
                    attack_train_run_id = f"mode-attacked__attack-{attack}__seed{seed}__model-{model}"
                attack_train_artifact_dir = artifact_root / "attack_train" / attack_train_run_id
                attack_train_log = log_dir / f"{attack_train_run_id}__stage-attack_train.log"

                if not run_stage("attack_train", attack_train_run_id, attack_train_artifact_dir,
                                 attack_train_log, attack, model, poison_rate, "none", dataset_cfg, seed):
                    print(f"[WARN] Attack/train failed for {attack_train_run_id}")
                    continue

                for detection in DETECTIONS:
                    detection_run_id = f"{attack_train_run_id}__det-{detection}"
                    detection_artifact_dir = artifact_root / "detection" / detection_run_id
                    detection_log = log_dir / f"{detection_run_id}__stage-detection.log"

                    if not run_stage("detection", detection_run_id, detection_artifact_dir,
                                     detection_log, attack, model, poison_rate, detection,
                                     dataset_cfg, seed, attack_train_artifact_dir):
                        print(f"[WARN] Detection failed for {detection_run_id}")

                # All 8 detections done for this (seed, attack, model) tuple.
                # attack_train heavy artifacts are no longer needed anywhere in this benchmark.
                if cleanup_artifacts:
                    cleanup_tuple_artifacts(attack_train_artifact_dir)

    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Full Detection Benchmark Runner")
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET,
        choices=list(DATASET_CONFIGS.keys()),
        help=f"Dataset to benchmark (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all datasets sequentially (overrides --dataset).",
    )
    parser.add_argument(
        "--no-summarize", action="store_true",
        help="Skip auto-triggering summarize_benchmark.py at the end.",
    )
    parser.add_argument(
        "--cleanup-artifacts", action="store_true",
        help="Delete data/datasets.npz + checkpoint/ + attack_result/ after all 8 detections "
             "for each (seed, attack, model) tuple have finished. Saves ~80%% disk space. "
             "Keeps summary.json + metadata. Loses the ability to re-run detection on old models.",
    )
    args = parser.parse_args()

    datasets = list(DATASET_CONFIGS) if args.all else [args.dataset]
    print(f"[INFO] Datasets: {datasets}")
    print(f"[INFO] Seeds: {SEEDS}")
    print(f"[INFO] Cleanup artifacts after each tuple: {args.cleanup_artifacts}")

    started = time.time()
    run_dirs = [run_dataset(ds, DATASET_CONFIGS[ds], args.cleanup_artifacts) for ds in datasets]

    elapsed = int(time.time() - started)
    print(f"[INFO] Benchmark completed in {elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}")

    if args.no_summarize:
        return

    summarize_cmd = [str(PYTHON_BIN), str(ROOT_DIR / "scripts" / "summarize_benchmark.py")]
    for d in run_dirs:
        summarize_cmd += ["--run-dir", str(d)]
    print(f"[INFO] Running summarize_benchmark.py on {len(run_dirs)} run dir(s)")
    subprocess.run(summarize_cmd, check=False)


if __name__ == "__main__":
    main()
