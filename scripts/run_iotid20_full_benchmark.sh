#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
DEVICE="cuda"
TRAIN_EPOCHS="100"
TRAIN_BATCH_SIZE="1024"
WANDB_ENABLED="true"
WANDB_SUMMARY_PROFILE="compact"
WANDB_PROJECT="iotid20_full_backdoor_unlearning_benchmark"
SEEDS="1 2 3"
SHOW_OUTPUT="1"

# IoTID20-specific attack constants.
# LabelEncoder mapping in the prepared IoTID20 split:
# 0=DoS, 1=MITM ARP Spoofing, 2=Mirai, 3=Normal, 4=Scan.
IOTID20_TARGET_LABEL="3"

# Raw numeric columns (trigger is injected before preprocessing):
# 34=Bwd_Pkts/s, 15=Flow_Pkts/s, 1=Flow_Duration.
IOTID20_TRIGGER_FEATURES_CSV="34,15,1"
IOTID20_BADNETS_TRIGGER_VALUE="3.0"
IOTID20_BADNETS_POISON_RATE="0.05"
IOTID20_TABDOOR_POISON_RATE="0.02"
IOTID20_CATBACK_POISON_RATE="0.02"

# Main benchmark defaults.
ATTACK_MODES="attacked"
MODELS=(mlp resnet tabnet ft_transformer saint)
ATTACKS=(badnets tabdoor catback)
DETECTIONS=(none spectral_signatures mlbd mm_bd neural_cleanse mlbd_cso mmbd_cso nc_cso)
UNLEARNINGS=(oracle_retrain detected_retrain retrain bad_teaching abl rnp)
INCLUDE_NONE_DETECTION="1"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results/benchmark_runs/${STAMP}"
LOG_DIR="${OUT_DIR}/logs"
ART_DIR="${OUT_DIR}/artifacts"
mkdir -p "$LOG_DIR" "$ART_DIR"

SUMMARY_CSV="${OUT_DIR}/summarize.csv"
SUMMARY_COLUMNS="mode,attack,poison_rate,target_label,trigger_features,seed,model,detection,unlearning,status,clean_test_accuracy,clean_val_best_accuracy,backdoor_asr,backdoor_accuracy,detection_status,detection_track_type,predicted_is_infected,predicted_target_class,true_is_infected,true_target_class,poisoned_train_samples,detection_precision,detection_recall,detection_f1,detection_auroc,detection_average_precision,detection_topk_recall,suspect_count,detection_runtime_sec,unlearning_status,unlearning_track_type,unlearning_num_removed,unlearning_forget_recall,unlearning_clean_test_accuracy_after,unlearning_backdoor_asr_after,unlearning_delta_clean_accuracy,unlearning_delta_asr,unlearning_runtime_sec,run_elapsed_sec,log_file"
printf "%s\n" "$SUMMARY_COLUMNS" > "$SUMMARY_CSV"
AGGREGATE_CSV="${OUT_DIR}/aggregate_summary.csv"

format_seconds_hms() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

latest_dir_since() {
  local base="$1"
  local start_ts="$2"
  if [[ ! -d "$base" ]]; then
    echo ""
    return
  fi
  find "$base" -mindepth 1 -maxdepth 1 -type d -printf "%T@ %p\n" | \
    awk -v start="$start_ts" '$1 >= start {print $0}' | \
    sort -nr | head -n1 | awk '{print $2}'
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
  [[ "$method" == "retrain" || "$method" == "detected_retrain" || "$method" == "bad_teaching" ]]
}

detection_supports_sample_level_forget() {
  local det="$1"
  [[ "$det" == "spectral_signatures" ]]
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
  RUN_ART_DIR="$run_art_dir" \
  RUN_ELAPSED_SEC="$run_elapsed_sec" \
  LOG_FILE="$log_file" \
  "$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path

columns = os.environ["SUMMARY_COLUMNS"].split(",")
run_art_dir = Path(os.environ["RUN_ART_DIR"])

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
        "run_elapsed_sec": os.environ["RUN_ELAPSED_SEC"],
        "log_file": os.environ["LOG_FILE"],
    }
)


def load_json(name: str):
    path = run_art_dir / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


model_metrics = load_json("model_metrics.json")
row["clean_test_accuracy"] = model_metrics.get("clean/test/accuracy", "")
row["clean_val_best_accuracy"] = model_metrics.get("clean/val_accuracy_best", "")
row["backdoor_asr"] = model_metrics.get("backdoor/asr", "")
row["backdoor_accuracy"] = model_metrics.get("backdoor/accuracy", "")

detection_summary = load_json("detection_summary.json")
if detection_summary:
    det_metrics = detection_summary.get("summary_metrics", {}) or {}
    context = detection_summary.get("context", {}) or {}
    attack_metadata = context.get("attack_metadata", {}) or {}
    row["detection_status"] = detection_summary.get("status", "")
    row["detection_track_type"] = detection_summary.get("track_type", "")
    row["predicted_is_infected"] = detection_summary.get("predicted_is_infected", "")
    row["predicted_target_class"] = detection_summary.get("predicted_target_class", "")
    row["true_is_infected"] = context.get("true_is_infected", "")
    row["true_target_class"] = context.get("true_target_class", "")
    row["poisoned_train_samples"] = attack_metadata.get("num_poisoned_train_samples", "")
    row["detection_precision"] = det_metrics.get("detection/precision", "")
    row["detection_recall"] = det_metrics.get("detection/recall", "")
    row["detection_f1"] = det_metrics.get("detection/f1", "")
    row["detection_auroc"] = det_metrics.get("detection/auroc", "")
    row["detection_average_precision"] = det_metrics.get("detection/average_precision", "")
    row["suspect_count"] = detection_summary.get("suspect_indices_count", "")
    row["detection_runtime_sec"] = detection_summary.get("runtime_sec", "")
    for key, value in det_metrics.items():
        if str(key).startswith("detection/topk_recall@"):
            row["detection_topk_recall"] = value
            break

unlearning_summary = load_json("unlearning_summary.json")
if unlearning_summary:
    unl_metrics = unlearning_summary.get("summary_metrics", {}) or {}
    row["unlearning_status"] = unlearning_summary.get("status", "")
    row["unlearning_track_type"] = unlearning_summary.get("track_type", "")
    row["unlearning_num_removed"] = unlearning_summary.get("num_removed", "")
    row["unlearning_forget_recall"] = unl_metrics.get("unlearning/forget_recall", "")
    row["unlearning_clean_test_accuracy_after"] = unl_metrics.get("clean/test/accuracy_after", "")
    row["unlearning_backdoor_asr_after"] = unl_metrics.get("backdoor/asr_after", "")
    row["unlearning_delta_clean_accuracy"] = unl_metrics.get("unlearning/delta_clean_accuracy", "")
    row["unlearning_delta_asr"] = unl_metrics.get("unlearning/delta_asr", "")
    row["unlearning_runtime_sec"] = unlearning_summary.get("runtime_sec", "")

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

if not summary_path.exists():
  raise FileNotFoundError(f"Missing summary CSV: {summary_path}")

group_keys = ["mode", "attack", "poison_rate", "target_label", "trigger_features", "model", "detection", "unlearning"]
metric_columns = [
  "clean_test_accuracy",
  "clean_val_best_accuracy",
  "backdoor_asr",
  "backdoor_accuracy",
  "detection_precision",
  "detection_recall",
  "detection_f1",
  "detection_auroc",
  "detection_average_precision",
  "detection_topk_recall",
  "detection_runtime_sec",
  "unlearning_forget_recall",
  "unlearning_clean_test_accuracy_after",
  "unlearning_backdoor_asr_after",
  "unlearning_delta_clean_accuracy",
  "unlearning_delta_asr",
  "unlearning_runtime_sec",
  "run_elapsed_sec",
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
      poison_rate="$(attack_poison_rate "$attack")"
      if [[ "$mode" == "clean" ]]; then
        poison_rate="0.0"
      elif [[ "$mode" != "attacked" ]]; then
        echo "[WARN] Unknown ATTACK_MODE '$mode', skipping"
        continue
      fi

      for model in "${MODELS[@]}"; do
        for detection in "${DETECTIONS[@]}"; do
          for unlearning in "${UNLEARNINGS[@]}"; do
            if unlearning_requires_detector_flags "$unlearning" && ! detection_supports_sample_level_forget "$detection"; then
              echo "[INFO] Skip incompatible combo: detection=$detection unlearning=$unlearning requires sample-level flags"
              continue
            fi

            if ! unlearning_supports_model "$unlearning" "$model"; then
              echo "[INFO] Skip unsupported combo: model=$model unlearning=$unlearning"
              continue
            fi

            run_name="mode-${mode}__attack-${attack}__seed${seed}__model-${model}__det-${detection}__unlearn-${unlearning}"
            log_file="${LOG_DIR}/${run_name}.log"
            run_started_epoch="$(date +%s)"
            run_started_at="$("$PYTHON_BIN" - <<'PY'
import time
print(time.time())
PY
)"

            hydra_overrides=(
              "data=iotid20"
              "model=$model"
              "attack=$attack"
              "attack.poison_rate=$poison_rate"
              "attack.target_label=$IOTID20_TARGET_LABEL"
              "attack.seed=$seed"
              "attack.trigger_features=[$IOTID20_TRIGGER_FEATURES_CSV]"
              "detection=$detection"
              "unlearning=$unlearning"
              "seed=$seed"
              "train.device=$DEVICE"
              "wandb.enabled=$WANDB_ENABLED"
              "wandb.project=$WANDB_PROJECT"
              "wandb.summary_profile=$WANDB_SUMMARY_PROFILE"
            )

            if [[ -n "$TRAIN_EPOCHS" ]]; then
              hydra_overrides+=("train.epochs=$TRAIN_EPOCHS")
            fi
            if [[ -n "$TRAIN_BATCH_SIZE" ]]; then
              hydra_overrides+=("train.batch_size=$TRAIN_BATCH_SIZE")
            fi

            if [[ "$attack" == "badnets" ]]; then
              hydra_overrides+=("attack.trigger_value=$IOTID20_BADNETS_TRIGGER_VALUE")
            elif [[ "$attack" == "tabdoor" ]]; then
              hydra_overrides+=("attack.trigger_size=3")
            elif [[ "$attack" == "catback" ]]; then
              hydra_overrides+=("attack.device=$DEVICE")
            fi

            echo "[INFO] Running $run_name"
            set +e
            if [[ "$SHOW_OUTPUT" == "1" ]]; then
              "$PYTHON_BIN" run.py "${hydra_overrides[@]}" 2>&1 | tee "$log_file"
              exit_code=${PIPESTATUS[0]}
            else
              "$PYTHON_BIN" run.py "${hydra_overrides[@]}" >"$log_file" 2>&1
              exit_code=$?
            fi
            set -e

            run_finished_epoch="$(date +%s)"
            run_elapsed_sec=$((run_finished_epoch - run_started_epoch))
            run_elapsed_hms="$(format_seconds_hms "$run_elapsed_sec")"

            status="ok"
            if [[ $exit_code -ne 0 ]]; then
              status="failed"
            fi

            run_art_dir="${ART_DIR}/${run_name}"
            mkdir -p "$run_art_dir"

            model_metrics_candidate="artifacts/models/metrics.json"
            model_metrics_is_new="$("$PYTHON_BIN" - <<PY
from pathlib import Path
p = Path(r"${model_metrics_candidate}")
print(int(p.exists() and p.stat().st_mtime >= float("${run_started_at}")))
PY
)"
            if [[ "$model_metrics_is_new" == "1" ]]; then
              cp "$model_metrics_candidate" "$run_art_dir/model_metrics.json"
            fi

            det_dir="$(latest_dir_since artifacts/detection "$run_started_at")"
            if [[ -n "$det_dir" && -f "$det_dir/summary.json" ]]; then
              cp "$det_dir/summary.json" "$run_art_dir/detection_summary.json"
            fi

            unlearn_dir="$(latest_dir_since artifacts/unlearning "$run_started_at")"
            if [[ -n "$unlearn_dir" && -f "$unlearn_dir/summary.json" ]]; then
              cp "$unlearn_dir/summary.json" "$run_art_dir/unlearning_summary.json"
            fi

            append_summary_row

            if [[ "$status" == "ok" ]]; then
              echo "[INFO] Done $run_name (elapsed=$run_elapsed_hms)"
            else
              echo "[WARN] Failed $run_name (elapsed=$run_elapsed_hms, see $log_file)"
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
