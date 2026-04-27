#!/usr/bin/env bash
set -euo pipefail

# Full IoTID20 benchmark runner:
# - attack: badnets
# - unlearning: all conf/unlearning/*.yaml (optionally skip none)
# - models: all conf/model/*.yaml
# - detections: all conf/detection/*.yaml (optionally skip none)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
DEVICE="${DEVICE:-cuda}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
SEEDS="${SEEDS:-42}"
SHOW_OUTPUT="${SHOW_OUTPUT:-1}"
INCLUDE_NONE_DETECTION="${INCLUDE_NONE_DETECTION:-1}"
ATTACK_MODES="${ATTACK_MODES:-attacked clean}"
ATTACKED_POISON_RATE="${ATTACKED_POISON_RATE:-0.1}"
MODELS_OVERRIDE="${MODELS_OVERRIDE:-}"
DETECTIONS_OVERRIDE="${DETECTIONS_OVERRIDE:-}"
UNLEARNINGS_OVERRIDE="${UNLEARNINGS_OVERRIDE:-}"
INCLUDE_NONE_UNLEARNING="${INCLUDE_NONE_UNLEARNING:-1}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results/benchmark_runs/${STAMP}"
LOG_DIR="${OUT_DIR}/logs"
ART_DIR="${OUT_DIR}/artifacts"
mkdir -p "$LOG_DIR" "$ART_DIR"

SUMMARY_CSV="${OUT_DIR}/summary.csv"
printf "mode,poison_rate,seed,model,detection,unlearning,status,test_accuracy,val_best_accuracy,backdoor_asr,backdoor_accuracy,detection_status,det_track_type,predicted_is_infected,predicted_target_class,true_is_infected,true_target_class,poisoned_train_samples,det_num_candidates,det_infected_acc,det_target_acc,det_source_acc,det_fpr,det_tpr,det_precision,det_recall,det_f1,det_auroc,det_average_precision,det_topk_recall,suspect_count,smallest_mask_target_class,anomaly_index,num_flagged_labels,detection_runtime_sec,unlearning_status,unlearning_track_type,unlearning_num_removed,unlearning_forget_recall,unlearning_delta_asr,unlearning_runtime_sec,log_file\n" > "$SUMMARY_CSV"

# Fixed benchmark matrix (override via *_OVERRIDE env vars when needed)
MODELS=(tabnet saint ft_transformer)
DETECTIONS=(spectral_signatures mlbd mm_bd neural_cleanse mlbd_cso mmbd_cso nc_cso)
UNLEARNINGS=(none oracle_retrain detected_retrain retrain bad_teaching abl)

if [[ -n "$MODELS_OVERRIDE" ]]; then
  read -r -a MODELS <<< "$MODELS_OVERRIDE"
fi

if [[ -n "$DETECTIONS_OVERRIDE" ]]; then
  read -r -a DETECTIONS <<< "$DETECTIONS_OVERRIDE"
fi

if [[ -n "$UNLEARNINGS_OVERRIDE" ]]; then
  read -r -a UNLEARNINGS <<< "$UNLEARNINGS_OVERRIDE"
fi

if [[ "$INCLUDE_NONE_DETECTION" != "1" ]]; then
  echo "[WARN] INCLUDE_NONE_DETECTION is ignored because detection=none was removed from this benchmark matrix."
fi

if [[ "$INCLUDE_NONE_UNLEARNING" != "1" ]]; then
  FILTERED=()
  for u in "${UNLEARNINGS[@]}"; do
    if [[ "$u" != "none" ]]; then
      FILTERED+=("$u")
    fi
  done
  UNLEARNINGS=("${FILTERED[@]}")
fi

echo "[INFO] Output dir: $OUT_DIR"
echo "[INFO] Models: ${MODELS[*]}"
echo "[INFO] Detections: ${DETECTIONS[*]}"
echo "[INFO] Unlearning methods: ${UNLEARNINGS[*]}"
echo "[INFO] Seeds: $SEEDS"
echo "[INFO] Modes: $ATTACK_MODES"

latest_dir() {
  local base="$1"
  if [[ ! -d "$base" ]]; then
    echo ""
    return
  fi
  find "$base" -mindepth 1 -maxdepth 1 -type d -printf "%T@ %p\n" | sort -nr | head -n1 | awk '{print $2}'
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

format_seconds_hms() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

benchmark_started_epoch="$(date +%s)"
benchmark_started_human="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] Benchmark start time: $benchmark_started_human"

unlearning_requires_detector_flags() {
  local method="$1"
  [[ "$method" == "retrain" || "$method" == "detected_retrain" || "$method" == "bad_teaching" ]]
}

detection_supports_sample_level_forget() {
  local det="$1"
  # Current sample-level detector in this benchmark path.
  [[ "$det" == "spectral_signatures" ]]
}

for seed in $SEEDS; do
  for mode in $ATTACK_MODES; do
    poison_rate="$ATTACKED_POISON_RATE"
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
          echo "[WARN] Skipping incompatible combo: detection=$detection unlearning=$unlearning (unlearning requires sample-level detector flags)"
          continue
        fi

        run_name="mode-${mode}__seed${seed}__model-${model}__det-${detection}__unlearn-${unlearning}"
        log_file="${LOG_DIR}/${run_name}.log"
        run_started_epoch="$(date +%s)"
        run_started_at="$("$PYTHON_BIN" - <<'PY'
import time
print(time.time())
PY
)"

        echo "[INFO] Running $run_name"

        set +e
        if [[ "$SHOW_OUTPUT" == "1" ]]; then
          "$PYTHON_BIN" run.py \
            data=iotid20 \
            model="$model" \
            attack=badnets \
            attack.poison_rate="$poison_rate" \
            detection="$detection" \
            unlearning="$unlearning" \
            seed="$seed" \
            train.epochs="$EPOCHS" \
            train.batch_size="$BATCH_SIZE" \
            train.device="$DEVICE" \
            wandb.enabled="$WANDB_ENABLED" 2>&1 | tee "$log_file"
          exit_code=${PIPESTATUS[0]}
        else
          "$PYTHON_BIN" run.py \
            data=iotid20 \
            model="$model" \
            attack=badnets \
            attack.poison_rate="$poison_rate" \
            detection="$detection" \
            unlearning="$unlearning" \
            seed="$seed" \
            train.epochs="$EPOCHS" \
            train.batch_size="$BATCH_SIZE" \
            train.device="$DEVICE" \
            wandb.enabled="$WANDB_ENABLED" >"$log_file" 2>&1
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

        model_metrics_json=""
        model_metrics_candidate="artifacts/models/metrics.json"
        model_metrics_is_new="$("$PYTHON_BIN" - <<PY
from pathlib import Path
p = Path(r"${model_metrics_candidate}")
print(int(p.exists() and p.stat().st_mtime >= float("${run_started_at}")))
PY
)"
        if [[ "$model_metrics_is_new" == "1" ]]; then
          model_metrics_json="$model_metrics_candidate"
        fi

        det_dir="$(latest_dir_since artifacts/detection "$run_started_at")"
        det_summary_json=""
        if [[ -n "$det_dir" && -f "$det_dir/summary.json" ]]; then
          det_summary_json="$det_dir/summary.json"
        fi

        run_art_dir="${ART_DIR}/${run_name}"
        mkdir -p "$run_art_dir"
        if [[ -n "$model_metrics_json" && -f "$model_metrics_json" ]]; then
          cp "$model_metrics_json" "$run_art_dir/model_metrics.json"
        fi
        if [[ -n "$det_summary_json" ]]; then
          cp "$det_summary_json" "$run_art_dir/detection_summary.json"
        fi

        unlearn_dir="$(latest_dir_since artifacts/unlearning "$run_started_at")"
        unlearn_summary_json=""
        if [[ -n "$unlearn_dir" && -f "$unlearn_dir/summary.json" ]]; then
          unlearn_summary_json="$unlearn_dir/summary.json"
          cp "$unlearn_summary_json" "$run_art_dir/unlearning_summary.json"
        fi

        eval_line="$($PYTHON_BIN - <<PY
import json
from pathlib import Path
model_metrics = Path(r"${run_art_dir}/model_metrics.json")
det_summary = Path(r"${run_art_dir}/detection_summary.json")
unlearning_summary = Path(r"${run_art_dir}/unlearning_summary.json")
def as_csv(value):
    if value is None:
        return ""
    return str(value)

values = {
    "status": "${status}",
    "acc": "",
    "val_best": "",
    "backdoor_asr": "",
    "backdoor_accuracy": "",
    "det_status": "",
    "det_track_type": "",
    "pred_is_infected": "",
    "pred_target_class": "",
    "true_is_infected": "",
    "true_target_class": "",
    "poisoned_train_samples": "",
    "det_num_candidates": "",
    "det_infected_acc": "",
    "det_target_acc": "",
    "det_source_acc": "",
    "det_fpr": "",
    "det_tpr": "",
    "det_precision": "",
    "det_recall": "",
    "det_f1": "",
    "det_auroc": "",
    "det_average_precision": "",
    "det_topk_recall": "",
    "suspects": "",
    "smallest_mask_target_class": "",
    "anomaly_index": "",
    "num_flagged_labels": "",
    "det_runtime": "",
    "unl_status": "",
    "unl_track_type": "",
    "unl_num_removed": "",
    "unl_forget_recall": "",
    "unl_delta_asr": "",
    "unl_runtime": "",
}
if model_metrics.exists():
    d = json.loads(model_metrics.read_text(encoding="utf-8"))
    values["acc"] = d.get("clean/test/accuracy", "")
    values["val_best"] = d.get("clean/val_accuracy_best", "")
    values["backdoor_asr"] = d.get("backdoor/asr", "")
    values["backdoor_accuracy"] = d.get("backdoor/accuracy", "")
if det_summary.exists():
    d = json.loads(det_summary.read_text(encoding="utf-8"))
    metrics = d.get("summary_metrics", {}) or {}
    values["det_status"] = d.get("status", "")
    values["det_track_type"] = d.get("track_type", "")
    values["pred_is_infected"] = d.get("predicted_is_infected", "")
    values["pred_target_class"] = d.get("predicted_target_class", "")
    values["true_is_infected"] = d.get("context", {}).get("true_is_infected", "")
    values["true_target_class"] = d.get("context", {}).get("true_target_class", "")
    values["poisoned_train_samples"] = d.get("context", {}).get("attack_metadata", {}).get("num_poisoned_train_samples", "")
    values["det_num_candidates"] = metrics.get("detection/num_candidates", "")
    values["det_infected_acc"] = metrics.get("detection/is_infected_accuracy", "")
    values["det_target_acc"] = metrics.get("detection/target_class_accuracy", "")
    values["det_source_acc"] = metrics.get("detection/source_class_accuracy", "")
    values["det_fpr"] = metrics.get("detection/false_positive_rate", "")
    values["det_tpr"] = metrics.get("detection/true_positive_rate", "")
    values["det_precision"] = metrics.get("detection/precision", "")
    values["det_recall"] = metrics.get("detection/recall", "")
    values["det_f1"] = metrics.get("detection/f1", "")
    values["det_auroc"] = metrics.get("detection/auroc", "")
    values["det_average_precision"] = metrics.get("detection/average_precision", "")
    values["suspects"] = d.get("suspect_indices_count", "")
    values["smallest_mask_target_class"] = metrics.get("detection/smallest_mask_target_class", "")
    values["anomaly_index"] = metrics.get("detection/anomaly_index", "")
    values["num_flagged_labels"] = metrics.get("detection/num_flagged_labels", "")
    values["det_runtime"] = d.get("runtime_sec", "")
    for key, value in metrics.items():
        if str(key).startswith("detection/topk_recall@"):
            values["det_topk_recall"] = value
            break

if unlearning_summary.exists():
    d = json.loads(unlearning_summary.read_text(encoding="utf-8"))
    metrics = d.get("summary_metrics", {}) or {}
    values["unl_status"] = d.get("status", "")
    values["unl_track_type"] = d.get("track_type", "")
    values["unl_num_removed"] = d.get("num_removed", "")
    values["unl_forget_recall"] = metrics.get("unlearning/forget_recall", "")
    values["unl_delta_asr"] = metrics.get("unlearning/delta_asr", "")
    values["unl_runtime"] = d.get("runtime_sec", "")

ordered = [
    "status",
    "acc",
    "val_best",
    "backdoor_asr",
    "backdoor_accuracy",
    "det_status",
    "det_track_type",
    "pred_is_infected",
    "pred_target_class",
    "true_is_infected",
    "true_target_class",
    "poisoned_train_samples",
    "det_num_candidates",
    "det_infected_acc",
    "det_target_acc",
    "det_source_acc",
    "det_fpr",
    "det_tpr",
    "det_precision",
    "det_recall",
    "det_f1",
    "det_auroc",
    "det_average_precision",
    "det_topk_recall",
    "suspects",
    "smallest_mask_target_class",
    "anomaly_index",
    "num_flagged_labels",
    "det_runtime",
    "unl_status",
    "unl_track_type",
    "unl_num_removed",
    "unl_forget_recall",
    "unl_delta_asr",
    "unl_runtime",
]
print(",".join(as_csv(values[key]) for key in ordered))
PY
)"

        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
          "$mode" "$poison_rate" "$seed" "$model" "$detection" "$unlearning" \
          "${eval_line%%,*}" \
          "$(echo "$eval_line" | cut -d',' -f2)" \
          "$(echo "$eval_line" | cut -d',' -f3)" \
          "$(echo "$eval_line" | cut -d',' -f4)" \
          "$(echo "$eval_line" | cut -d',' -f5)" \
          "$(echo "$eval_line" | cut -d',' -f6)" \
          "$(echo "$eval_line" | cut -d',' -f7)" \
          "$(echo "$eval_line" | cut -d',' -f8)" \
          "$(echo "$eval_line" | cut -d',' -f9)" \
          "$(echo "$eval_line" | cut -d',' -f10)" \
          "$(echo "$eval_line" | cut -d',' -f11)" \
          "$(echo "$eval_line" | cut -d',' -f12)" \
          "$(echo "$eval_line" | cut -d',' -f13)" \
          "$(echo "$eval_line" | cut -d',' -f14)" \
          "$(echo "$eval_line" | cut -d',' -f15)" \
          "$(echo "$eval_line" | cut -d',' -f16)" \
          "$(echo "$eval_line" | cut -d',' -f17)" \
          "$(echo "$eval_line" | cut -d',' -f18)" \
          "$(echo "$eval_line" | cut -d',' -f19)" \
          "$(echo "$eval_line" | cut -d',' -f20)" \
          "$(echo "$eval_line" | cut -d',' -f21)" \
          "$(echo "$eval_line" | cut -d',' -f22)" \
          "$(echo "$eval_line" | cut -d',' -f23)" \
          "$(echo "$eval_line" | cut -d',' -f24)" \
          "$(echo "$eval_line" | cut -d',' -f25)" \
          "$(echo "$eval_line" | cut -d',' -f26)" \
          "$(echo "$eval_line" | cut -d',' -f27)" \
          "$(echo "$eval_line" | cut -d',' -f28)" \
          "$(echo "$eval_line" | cut -d',' -f29)" \
          "$(echo "$eval_line" | cut -d',' -f30)" \
          "$(echo "$eval_line" | cut -d',' -f31)" \
          "$(echo "$eval_line" | cut -d',' -f32)" \
          "$(echo "$eval_line" | cut -d',' -f33)" \
          "$(echo "$eval_line" | cut -d',' -f34)" \
          "$(echo "$eval_line" | cut -d',' -f35)" \
          "$log_file" \
          >> "$SUMMARY_CSV"

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

benchmark_finished_epoch="$(date +%s)"
benchmark_elapsed_sec=$((benchmark_finished_epoch - benchmark_started_epoch))
benchmark_elapsed_hms="$(format_seconds_hms "$benchmark_elapsed_sec")"

echo "[INFO] Benchmark completed."
echo "[INFO] Total elapsed: $benchmark_elapsed_hms"
echo "[INFO] Summary: $SUMMARY_CSV"
