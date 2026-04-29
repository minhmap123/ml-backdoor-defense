#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Fixed constants - customize by directly editing these values.
PYTHON_BIN="./.venv/bin/python"
DEVICE="cuda"
TRAIN_EPOCHS="100"
TRAIN_BATCH_SIZE="1024"
WANDB_ENABLED="true"
WANDB_SUMMARY_PROFILE="compact"
WANDB_PROJECT="iotid20_full_backdoor_unlearning_benchmark"
SEEDS="1 2 3"
SHOW_OUTPUT="1"
SKIP_EXISTING="1"

# IoTID20-specific attack constants.
# LabelEncoder mapping in the prepared IoTID20 split:
# 0=DoS, 1=MITM ARP Spoofing, 2=Mirai, 3=Normal, 4=Scan.
IOTID20_TARGET_LABEL="3"

# Raw numeric columns (trigger is injected before preprocessing):
# 34=Bwd_Pkts/s, 15=Flow_Pkts/s, 1=Flow_Duration.
IOTID20_TRIGGER_FEATURES_CSV="34,15,1"

# Attack-specific parameters for BadNets, TabDoor, and CatBack.
IOTID20_BADNETS_TRIGGER_VALUE="3.0"
IOTID20_BADNETS_POISON_RATE="0.05"
IOTID20_TABDOOR_TRIGGER_SIZE="3"
IOTID20_TABDOOR_POISON_RATE="0.02"
IOTID20_CATBACK_POISON_RATE="0.02"

# Main benchmark configuration. Clean controls are intentionally deferred.
ATTACK_MODES="attacked"
MODELS=(mlp resnet tabnet ft_transformer saint)
ATTACKS=(badnets tabdoor catback)
DETECTIONS=(none spectral_signatures mlbd mm_bd neural_cleanse mlbd_cso mmbd_cso nc_cso)
UNLEARNINGS=(oracle_retrain retrain detected_retrain bad_teaching abl rnp)

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-results/benchmark_runs/${STAMP}}"
LOG_DIR="${OUT_DIR}/logs"
ART_DIR="${OUT_DIR}/artifacts"
HYDRA_DIR="${OUT_DIR}/hydra"
mkdir -p "$LOG_DIR" "$ART_DIR" "$HYDRA_DIR"

SUMMARY_CSV="${OUT_DIR}/summarize.csv"
SUMMARY_COLUMNS="mode,attack,poison_rate,target_label,trigger_features,seed,model,detection,unlearning,status,attack_train_run_id,detection_run_id,unlearning_run_id,attack_train_artifact_dir,detection_artifact_dir,unlearning_artifact_dir,clean_test_accuracy,clean_val_best_accuracy,backdoor_asr,backdoor_accuracy,poisoned_train_samples,params_total,params_active,params_inactive,epochs,batch_size,learning_rate,detection_status,detection_track_type,predicted_is_infected,predicted_target_class,true_is_infected,true_target_class,detection_precision,detection_recall,detection_f1,detection_auroc,detection_average_precision,detection_topk_recall,suspect_count,detection_runtime_sec,unlearning_status,unlearning_track_type,unlearning_forget_source,unlearning_num_removed,unlearning_num_retained,unlearning_forget_precision,unlearning_forget_recall,unlearning_forget_f1,unlearning_clean_test_accuracy_before,unlearning_backdoor_asr_before,unlearning_clean_test_accuracy_after,unlearning_backdoor_asr_after,unlearning_delta_clean_accuracy,unlearning_delta_asr,unlearning_runtime_sec,runtime_attack_train_sec,runtime_detection_sec,runtime_unlearning_sec,total_runtime_sec,log_file"
printf "%s\n" "$SUMMARY_COLUMNS" > "$SUMMARY_CSV"
AGGREGATE_CSV="${OUT_DIR}/aggregate_summary.csv"

format_seconds_hms() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

as_hydra_bool() {
  if [[ "$1" == "1" || "$1" == "true" || "$1" == "TRUE" ]]; then
    echo "true"
  else
    echo "false"
  fi
}

attack_poison_rate() {
  local attack="$1"
  case "$attack" in
    badnets) echo "$IOTID20_BADNETS_POISON_RATE" ;;
    tabdoor) echo "$IOTID20_TABDOOR_POISON_RATE" ;;
    catback) echo "$IOTID20_CATBACK_POISON_RATE" ;;
    *) echo "0.0" ;;
  esac
}

unlearning_requires_detector_flags() {
  local method="$1"
  [[ "$method" == "detected_retrain" || "$method" == "bad_teaching" ]]
}

unlearning_independent_of_detection() {
  local method="$1"
  [[ "$method" == "oracle_retrain" || "$method" == "retrain" || "$method" == "abl" || "$method" == "rnp" ]]
}

unlearning_supports_model() {
  local method="$1"
  local model="$2"
  if [[ "$method" == "rnp" ]]; then
    [[ "$model" == "mlp" || "$model" == "resnet" ]]
    return
  fi
  return 0
}

detection_has_sample_forget_output() {
  local summary_path="$1"
  "$PYTHON_BIN" - "$summary_path" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
if not summary_path.exists():
    sys.exit(1)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
artifacts = summary.get("artifacts", {}) or {}

def resolve(path_text):
    if not path_text:
        return None
    path = Path(str(path_text))
    if path.exists():
        return path
    candidate = summary_path.parent / path
    if candidate.exists():
        return candidate
    return path

suspect_path = resolve(artifacts.get("suspect_indices_npy"))
if suspect_path is not None and suspect_path.exists():
    sys.exit(0)

raw_scores = resolve(artifacts.get("raw_scores_csv"))
if raw_scores is not None and raw_scores.exists():
    header = raw_scores.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    if "flag" in header.split(","):
        sys.exit(0)

sys.exit(1)
PY
}

run_pipeline_stage() {
  local stage="$1"
  local run_id="$2"
  local artifact_dir="$3"
  local log_file="$4"
  local stage_detection="$5"
  local stage_unlearning="$6"
  local attack_train_artifact_dir="${7:-}"
  local detection_artifact_dir="${8:-}"
  local skip_existing_bool
  skip_existing_bool="$(as_hydra_bool "$SKIP_EXISTING")"

  mkdir -p "$(dirname "$log_file")" "$artifact_dir"

  local hydra_overrides=(
    "data=iotid20"
    "model=$model"
    "attack=$attack"
    "attack.poison_rate=$poison_rate"
    "attack.target_label=$IOTID20_TARGET_LABEL"
    "attack.seed=$seed"
    "attack.trigger_features=[$IOTID20_TRIGGER_FEATURES_CSV]"
    "detection=$stage_detection"
    "unlearning=$stage_unlearning"
    "seed=$seed"
    "train.device=$DEVICE"
    "pipeline.stage=$stage"
    "pipeline.run_id=$run_id"
    "pipeline.artifact_dir=$artifact_dir"
    "pipeline.skip_existing=$skip_existing_bool"
    "wandb.enabled=$WANDB_ENABLED"
    "wandb.project=$WANDB_PROJECT"
    "wandb.summary_profile=$WANDB_SUMMARY_PROFILE"
    "hydra.run.dir=${HYDRA_DIR}/${stage}/${run_id}"
  )

  if [[ -n "$attack_train_artifact_dir" ]]; then
    hydra_overrides+=("pipeline.attack_train_artifact_dir=$attack_train_artifact_dir")
  fi
  if [[ -n "$detection_artifact_dir" ]]; then
    hydra_overrides+=("pipeline.detection_artifact_dir=$detection_artifact_dir")
  fi
  if [[ -n "$TRAIN_EPOCHS" ]]; then
    hydra_overrides+=("train.epochs=$TRAIN_EPOCHS")
  fi
  if [[ -n "$TRAIN_BATCH_SIZE" ]]; then
    hydra_overrides+=("train.batch_size=$TRAIN_BATCH_SIZE")
  fi

  if [[ "$attack" == "badnets" ]]; then
    hydra_overrides+=("attack.trigger_value=$IOTID20_BADNETS_TRIGGER_VALUE")
  elif [[ "$attack" == "tabdoor" ]]; then
    hydra_overrides+=("attack.trigger_size=$IOTID20_TABDOOR_TRIGGER_SIZE")
  elif [[ "$attack" == "catback" ]]; then
    hydra_overrides+=("attack.device=$DEVICE")
  fi

  echo "[INFO] Stage $stage: $run_id"
  set +e
  if [[ "$SHOW_OUTPUT" == "1" ]]; then
    "$PYTHON_BIN" run.py "${hydra_overrides[@]}" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
  else
    "$PYTHON_BIN" run.py "${hydra_overrides[@]}" >"$log_file" 2>&1
    local exit_code=$?
  fi
  set -e
  return "$exit_code"
}

append_summary_row() {
  SUMMARY_CSV="$SUMMARY_CSV" \
  SUMMARY_COLUMNS="$SUMMARY_COLUMNS" \
  MODE="$mode" \
  ATTACK="$attack" \
  POISON_RATE="$poison_rate" \
  TARGET_LABEL="$IOTID20_TARGET_LABEL" \
  TRIGGER_FEATURES="$IOTID20_TRIGGER_FEATURES_CSV" \
  SEED="$seed" \
  MODEL="$model" \
  DETECTION="$detection" \
  UNLEARNING="$unlearning" \
  STATUS="$status" \
  ATTACK_TRAIN_RUN_ID="$attack_train_run_id" \
  DETECTION_RUN_ID="$detection_run_id" \
  UNLEARNING_RUN_ID="$unlearning_run_id" \
  ATTACK_TRAIN_ARTIFACT_DIR="$attack_train_artifact_dir" \
  DETECTION_ARTIFACT_DIR="$detection_artifact_dir" \
  UNLEARNING_ARTIFACT_DIR="$unlearning_artifact_dir" \
  LOG_FILE="$log_file" \
  "$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path

columns = os.environ["SUMMARY_COLUMNS"].split(",")

row = {column: "" for column in columns}
row.update(
    {
        "mode": os.environ["MODE"],
        "attack": os.environ["ATTACK"],
        "poison_rate": os.environ["POISON_RATE"],
        "target_label": os.environ["TARGET_LABEL"],
        "trigger_features": os.environ["TRIGGER_FEATURES"],
        "seed": os.environ["SEED"],
        "model": os.environ["MODEL"],
        "detection": os.environ["DETECTION"],
        "unlearning": os.environ["UNLEARNING"],
        "status": os.environ["STATUS"],
        "attack_train_run_id": os.environ["ATTACK_TRAIN_RUN_ID"],
        "detection_run_id": os.environ["DETECTION_RUN_ID"],
        "unlearning_run_id": os.environ["UNLEARNING_RUN_ID"],
        "attack_train_artifact_dir": os.environ["ATTACK_TRAIN_ARTIFACT_DIR"],
        "detection_artifact_dir": os.environ["DETECTION_ARTIFACT_DIR"],
        "unlearning_artifact_dir": os.environ["UNLEARNING_ARTIFACT_DIR"],
        "log_file": os.environ["LOG_FILE"],
    }
)

def load_summary(artifact_dir: str):
    path = Path(artifact_dir) / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

attack_summary = load_summary(os.environ["ATTACK_TRAIN_ARTIFACT_DIR"])
detection_summary = load_summary(os.environ["DETECTION_ARTIFACT_DIR"])
unlearning_summary = load_summary(os.environ["UNLEARNING_ARTIFACT_DIR"])

if attack_summary:
    metrics = attack_summary.get("train_metrics", {}) or attack_summary.get("summary_metrics", {}) or {}
    row["clean_test_accuracy"] = metrics.get("clean/test/accuracy", "")
    row["clean_val_best_accuracy"] = metrics.get("clean/val_accuracy_best", "")
    row["backdoor_asr"] = metrics.get("backdoor/asr", "")
    row["backdoor_accuracy"] = metrics.get("backdoor/accuracy", "")
    row["poisoned_train_samples"] = attack_summary.get("poisoned_train_samples", "")
    row["params_total"] = attack_summary.get("total_params", metrics.get("model/total_parameters", ""))
    row["params_active"] = attack_summary.get("active_params", metrics.get("model/active_parameters", ""))
    row["params_inactive"] = attack_summary.get("inactive_params", metrics.get("model/inactive_parameters", ""))
    row["runtime_attack_train_sec"] = attack_summary.get("runtime_sec", "")
    train_cfg_path = attack_summary.get("train_cfg_json")
    if train_cfg_path and Path(train_cfg_path).exists():
        train_cfg = json.loads(Path(train_cfg_path).read_text(encoding="utf-8"))
        row["epochs"] = train_cfg.get("epochs", "")
        row["batch_size"] = train_cfg.get("batch_size", "")
        row["learning_rate"] = train_cfg.get("learning_rate", "")

if detection_summary:
    det_metrics = detection_summary.get("summary_metrics", {}) or {}
    context = detection_summary.get("context", {}) or {}
    row["detection_status"] = detection_summary.get("status", "")
    row["detection_track_type"] = detection_summary.get("track_type", "")
    row["predicted_is_infected"] = detection_summary.get("predicted_is_infected", "")
    row["predicted_target_class"] = detection_summary.get("predicted_target_class", "")
    row["true_is_infected"] = context.get("true_is_infected", "")
    row["true_target_class"] = context.get("true_target_class", "")
    row["detection_precision"] = det_metrics.get("detection/precision", "")
    row["detection_recall"] = det_metrics.get("detection/recall", "")
    row["detection_f1"] = det_metrics.get("detection/f1", "")
    row["detection_auroc"] = det_metrics.get("detection/auroc", "")
    row["detection_average_precision"] = det_metrics.get("detection/average_precision", "")
    row["suspect_count"] = detection_summary.get("suspect_indices_count", "")
    row["detection_runtime_sec"] = detection_summary.get("runtime_sec", "")
    row["runtime_detection_sec"] = detection_summary.get("stage_runtime_sec", detection_summary.get("runtime_sec", ""))
    for key, value in det_metrics.items():
        if str(key).startswith("detection/topk_recall@"):
            row["detection_topk_recall"] = value
            break

if unlearning_summary:
    unl_metrics = unlearning_summary.get("summary_metrics", {}) or {}
    row["unlearning_status"] = unlearning_summary.get("status", "")
    row["unlearning_track_type"] = unlearning_summary.get("track_type", "")
    row["unlearning_forget_source"] = unlearning_summary.get("forget_set_source", "")
    row["unlearning_num_removed"] = unlearning_summary.get("num_removed", "")
    row["unlearning_num_retained"] = unlearning_summary.get("num_retained", "")
    row["unlearning_forget_precision"] = unl_metrics.get("unlearning/forget_precision", "")
    row["unlearning_forget_recall"] = unl_metrics.get("unlearning/forget_recall", "")
    row["unlearning_forget_f1"] = unl_metrics.get("unlearning/forget_f1", "")
    row["unlearning_clean_test_accuracy_before"] = unl_metrics.get("clean/test/accuracy_before", "")
    row["unlearning_backdoor_asr_before"] = unl_metrics.get("backdoor/asr_before", "")
    row["unlearning_clean_test_accuracy_after"] = unl_metrics.get("clean/test/accuracy_after", "")
    row["unlearning_backdoor_asr_after"] = unl_metrics.get("backdoor/asr_after", "")
    row["unlearning_delta_clean_accuracy"] = unl_metrics.get("unlearning/delta_clean_accuracy", "")
    row["unlearning_delta_asr"] = unl_metrics.get("unlearning/delta_asr", "")
    row["unlearning_runtime_sec"] = unlearning_summary.get("runtime_sec", "")
    row["runtime_unlearning_sec"] = unlearning_summary.get("stage_runtime_sec", unlearning_summary.get("runtime_sec", ""))

def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

row["total_runtime_sec"] = (
    to_float(row["runtime_attack_train_sec"])
    + to_float(row["runtime_detection_sec"])
    + to_float(row["runtime_unlearning_sec"])
)

with Path(os.environ["SUMMARY_CSV"]).open("a", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writerow(row)
PY
}

write_aggregate_summary() {
  SUMMARY_CSV="$SUMMARY_CSV" \
  AGGREGATE_CSV="$AGGREGATE_CSV" \
  "$PYTHON_BIN" - <<'PY'
import csv
from collections import defaultdict
from pathlib import Path
import statistics
import os

summary_path = Path(os.environ["SUMMARY_CSV"])
aggregate_path = Path(os.environ["AGGREGATE_CSV"])

group_keys = ["mode", "attack", "poison_rate", "target_label", "trigger_features", "model", "detection", "unlearning"]
metric_columns = [
  "clean_test_accuracy",
  "clean_val_best_accuracy",
  "backdoor_asr",
  "backdoor_accuracy",
  "poisoned_train_samples",
  "params_total",
  "params_active",
  "params_inactive",
  "epochs",
  "batch_size",
  "learning_rate",
  "detection_precision",
  "detection_recall",
  "detection_f1",
  "detection_auroc",
  "detection_average_precision",
  "detection_topk_recall",
  "detection_runtime_sec",
  "unlearning_forget_precision",
  "unlearning_forget_recall",
  "unlearning_forget_f1",
  "unlearning_clean_test_accuracy_before",
  "unlearning_backdoor_asr_before",
  "unlearning_clean_test_accuracy_after",
  "unlearning_backdoor_asr_after",
  "unlearning_delta_clean_accuracy",
  "unlearning_delta_asr",
  "unlearning_runtime_sec",
  "runtime_attack_train_sec",
  "runtime_detection_sec",
  "runtime_unlearning_sec",
  "total_runtime_sec",
]

rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8")))
groups = defaultdict(list)
for row in rows:
  key = tuple(row.get(column, "") for column in group_keys)
  groups[key].append(row)

aggregate_columns = group_keys + ["num_runs", "num_success_runs"]
for metric in metric_columns:
  aggregate_columns.extend([f"{metric}_mean", f"{metric}_std"])

def to_float(value: str):
  if value is None:
    return None
  text = str(value).strip()
  if text == "":
    return None
  try:
    return float(text)
  except ValueError:
    return None

def mean_std(values):
  if not values:
    return "", ""
  if len(values) == 1:
    return values[0], 0.0
  return statistics.fmean(values), statistics.pstdev(values)

with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
  writer = csv.DictWriter(handle, fieldnames=aggregate_columns)
  writer.writeheader()
  for key, group_rows in sorted(groups.items()):
    success_rows = [row for row in group_rows if str(row.get("status", "")).strip().lower() == "ok"]
    source_rows = success_rows if success_rows else group_rows
    agg_row = {column: value for column, value in zip(group_keys, key)}
    agg_row["num_runs"] = len(group_rows)
    agg_row["num_success_runs"] = len(success_rows)

    for metric in metric_columns:
      values = [to_float(row.get(metric, "")) for row in source_rows]
      values = [value for value in values if value is not None]
      mean_value, std_value = mean_std(values)
      agg_row[f"{metric}_mean"] = mean_value
      agg_row[f"{metric}_std"] = std_value

    writer.writerow(agg_row)
PY
}

echo "[INFO] Output dir: $OUT_DIR"
echo "[INFO] Attacks: ${ATTACKS[*]}"
echo "[INFO] Models: ${MODELS[*]}"
echo "[INFO] Detections: ${DETECTIONS[*]}"
echo "[INFO] Unlearning methods: ${UNLEARNINGS[*]}"
echo "[INFO] Seeds: $SEEDS"
echo "[INFO] Modes: $ATTACK_MODES"
echo "[INFO] W&B project: $WANDB_PROJECT"
echo "[INFO] IoTID20 target label: $IOTID20_TARGET_LABEL"
echo "[INFO] IoTID20 trigger features: [$IOTID20_TRIGGER_FEATURES_CSV]"
echo "[INFO] Clean controls: deferred; benchmark main mode is attacked"

benchmark_started_epoch="$(date +%s)"
benchmark_started_human="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] Benchmark start time: $benchmark_started_human"

for attack in "${ATTACKS[@]}"; do
  if [[ "$attack" != "badnets" && "$attack" != "tabdoor" && "$attack" != "catback" ]]; then
    echo "[WARN] Unknown or unsupported ATTACK '$attack', skipping"
    continue
  fi

  for seed in $SEEDS; do
    for mode in $ATTACK_MODES; do
      if [[ "$mode" != "attacked" ]]; then
        echo "[WARN] Mode '$mode' is deferred in the staged benchmark, skipping"
        continue
      fi

      poison_rate="$(attack_poison_rate "$attack")"

      for model in "${MODELS[@]}"; do
        attack_train_run_id="mode-${mode}__attack-${attack}__seed${seed}__model-${model}"
        attack_train_artifact_dir="${ART_DIR}/attack_train/${attack_train_run_id}"
        attack_train_log="${LOG_DIR}/${attack_train_run_id}__stage-attack_train.log"

        set +e
        run_pipeline_stage "attack_train" "$attack_train_run_id" "$attack_train_artifact_dir" "$attack_train_log" "none" "none"
        attack_train_exit=$?
        set -e
        if [[ $attack_train_exit -ne 0 ]]; then
          echo "[WARN] Attack/train failed for $attack_train_run_id; skipping dependent stages"
          continue
        fi

        for detection in "${DETECTIONS[@]}"; do
          detection_run_id="${attack_train_run_id}__det-${detection}"
          detection_artifact_dir="${ART_DIR}/detection/${detection_run_id}"
          detection_log="${LOG_DIR}/${detection_run_id}__stage-detection.log"

          set +e
          run_pipeline_stage "detection" "$detection_run_id" "$detection_artifact_dir" "$detection_log" "$detection" "none" "$attack_train_artifact_dir"
          detection_exit=$?
          set -e
          if [[ $detection_exit -ne 0 ]]; then
            echo "[WARN] Detection failed for $detection_run_id; skipping dependent unlearning rows"
            continue
          fi

          for unlearning in "${UNLEARNINGS[@]}"; do
            if ! unlearning_supports_model "$unlearning" "$model"; then
              echo "[INFO] Skip unsupported combo: model=$model unlearning=$unlearning"
              continue
            fi

            if unlearning_independent_of_detection "$unlearning" && [[ "$detection" != "none" ]]; then
              echo "[INFO] Skip duplicate independent unlearning combo: detection=$detection unlearning=$unlearning"
              continue
            fi

            if unlearning_requires_detector_flags "$unlearning"; then
              if ! detection_has_sample_forget_output "${detection_artifact_dir}/summary.json"; then
                echo "[INFO] Skip incompatible combo: detection=$detection unlearning=$unlearning has no sample-level forget output"
                continue
              fi
            fi

            unlearning_run_id="${detection_run_id}__unlearn-${unlearning}"
            unlearning_artifact_dir="${ART_DIR}/unlearning/${unlearning_run_id}"
            log_file="${LOG_DIR}/${unlearning_run_id}__stage-unlearning.log"

            set +e
            run_pipeline_stage "unlearning" "$unlearning_run_id" "$unlearning_artifact_dir" "$log_file" "$detection" "$unlearning" "$attack_train_artifact_dir" "$detection_artifact_dir"
            unlearning_exit=$?
            set -e

            status="ok"
            if [[ $unlearning_exit -ne 0 ]]; then
              status="failed"
            fi
            append_summary_row

            if [[ "$status" == "ok" ]]; then
              echo "[INFO] Done $unlearning_run_id"
            else
              echo "[WARN] Failed $unlearning_run_id (see $log_file)"
            fi
          done
        done
      done
    done
  done
done

benchmark_finished_epoch="$(date +%s)"
benchmark_elapsed_sec=$((benchmark_finished_epoch - benchmark_started_epoch))
benchmark_elapsed_hms="$(format_seconds_hms "$benchmark_elapsed_sec")"

echo "[INFO] Benchmark completed."
echo "[INFO] Total elapsed: $benchmark_elapsed_hms"
echo "[INFO] Summary: $SUMMARY_CSV"
write_aggregate_summary
echo "[INFO] Aggregate summary: $AGGREGATE_CSV"
