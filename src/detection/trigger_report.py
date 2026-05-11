from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..attacks.base import AttackResult
from ..data import IoTID20Dataset
from .types import DetectorResult, FeatureMetadata


def _class_name(bundle: Dict[str, Any], class_index: Any) -> str | None:
    if class_index is None:
        return None
    class_names = bundle.get("metadata", {}).get("classes") if isinstance(bundle.get("metadata", {}), dict) else None
    try:
        index = int(class_index)
    except (TypeError, ValueError):
        return str(class_index)
    if isinstance(class_names, list) and 0 <= index < len(class_names):
        return str(class_names[index])
    return str(index)


def build_model_input_feature_metadata(
    datasets: Dict[str, Any],
    attack_result: AttackResult,
    metadata: Dict[str, Any] | None = None,
) -> FeatureMetadata:
    train_x = np.asarray(datasets["train"]["x"], dtype=np.float32)
    num_feats = int(train_x.shape[1])
    feature_names = list(attack_result.poisoned_features.columns)
    lower = _metadata_vector(metadata, "model_input_min", num_feats, 0.0)
    upper = _metadata_vector(metadata, "model_input_max", num_feats, 1.0)
    return FeatureMetadata(
        feature_names=feature_names,
        feature_bounds_min=lower,
        feature_bounds_max=upper,
        num_numeric_features=num_feats,
        num_categorical_features=0,
    )


def print_reversed_trigger_if_available(
    *,
    seed: int,
    bundle: Dict[str, Any],
    detection_result: DetectorResult,
) -> None:
    if detection_result.estimated_trigger is None:
        return
    trigger_model_input = np.asarray(detection_result.estimated_trigger, dtype=np.float32).reshape(1, -1)
    mask = None if detection_result.estimated_mask is None else np.asarray(detection_result.estimated_mask, dtype=np.float32)
    feature_names = [str(col) for col in bundle["attack_result"].poisoned_features.columns]
    payload: Dict[str, Any] = {
        "detector": detection_result.detector_name,
        "predicted_target_class": _class_name(bundle, detection_result.predicted_target_class),
        "smallest_mask_target_class": detection_result.summary_metrics.get("detection/smallest_mask_target_class"),
        "feature_names": feature_names,
        "estimated_trigger_model_input": _round_float_list(trigger_model_input[0]),
        "estimated_mask": None if mask is None else _round_float_list(mask),
        "attack_feature_space": bundle["metadata"].get("attack_feature_space"),
        "true_trigger_metadata": _true_trigger_payload(_attack_metadata_from_bundle(bundle)),
    }
    try:
        scaler = reconstruct_iotid20_model_input_scaler(seed=seed, bundle=bundle)
        trigger_raw = scaler.inverse_transform(trigger_model_input)[0]
        payload["estimated_trigger_raw"] = _round_float_list(trigger_raw)
        payload["raw_feature_values"] = [
            {
                "index": int(idx),
                "feature": feature_names[idx] if idx < len(feature_names) else str(idx),
                "mask": None if mask is None else round(float(mask[idx]), 6),
                "model_input_value": round(float(trigger_model_input[0, idx]), 6),
                "raw_value": round(float(trigger_raw[idx]), 6),
            }
            for idx in range(trigger_model_input.shape[1])
        ]
    except Exception as exc:
        payload["raw_inverse_error"] = f"{exc.__class__.__name__}: {exc}"
    print("[detection] reversed_trigger")
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


def reconstruct_iotid20_model_input_scaler(*, seed: int, bundle: Dict[str, Any]):
    schema = replace(IoTID20Dataset.schema, random_state=int(seed))
    dataset = IoTID20Dataset(schema=schema)
    metadata = bundle["metadata"]
    attack_feature_space = str(metadata.get("attack_feature_space", ""))
    attack_injection_stage = str(metadata.get("attack_injection_stage", ""))
    if attack_feature_space == "scaled_model_input" or attack_injection_stage == "post_scaler_model_feature_space":
        return dataset.prepare_clean_partitions()["scaler"]
    scaler = dataset._make_scaler()
    scaler.fit(bundle["attack_result"].poisoned_features)
    return scaler


def _metadata_vector(metadata: Dict[str, Any] | None, key: str, size: int, default: float) -> np.ndarray:
    if metadata is not None and key in metadata:
        values = np.asarray(metadata[key], dtype=np.float32)
        if values.shape[0] >= size:
            return values[:size]
    return np.full(size, float(default), dtype=np.float32)


def _attack_metadata_from_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    attack_result = bundle["attack_result"]
    attack_metadata = dict(attack_result.metadata)
    train_metrics = bundle.get("model_metrics", {}) or {}
    if "backdoor/asr" in train_metrics:
        attack_metadata["observed_backdoor_asr"] = float(train_metrics["backdoor/asr"])
    if "backdoor/accuracy" in train_metrics:
        attack_metadata["observed_backdoor_accuracy"] = float(train_metrics["backdoor/accuracy"])
    attack_metadata["num_poisoned_train_samples"] = int(np.asarray(attack_result.poison_indices, dtype=np.int64).size)
    return attack_metadata


def _true_trigger_payload(attack_metadata: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "trigger_features",
        "trigger_value",
        "selected_features",
        "selected_values",
        "configured_trigger_features",
        "effective_feature_ranking",
        "trigger_size",
        "trigger_mode",
    )
    return {key: attack_metadata.get(key) for key in keys if key in attack_metadata}


def _round_float_list(values: np.ndarray, digits: int = 6) -> list[float | None]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return [None if not np.isfinite(value) else round(float(value), digits) for value in flat]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value
