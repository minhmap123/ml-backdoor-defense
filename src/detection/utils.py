from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..models.utils import parse_torch_batch, resolve_device as model_resolve_device, set_seed as model_set_seed, split_to_dataloader
from .types import ArtifactIndex, DetectorContext, DetectorResult, _to_jsonable


def set_seed(seed: Optional[int]) -> None:
    model_set_seed(seed)


def resolve_device(device: Optional[str]) -> torch.device:
    return model_resolve_device(device)


def measure_runtime(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    start = perf_counter()
    result = fn(*args, **kwargs)
    return result, float(perf_counter() - start)


def extract_model_features(
    model: torch.nn.Module,
    split: Any,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = split_to_dataloader(split, batch_size=int(batch_size), shuffle=False)
    model = model.to(device)
    model.eval()

    features_list = []
    labels_list = []
    with torch.no_grad():
        for batch in loader:
            model_input, y = parse_torch_batch(batch, device)
            features = model.forward_features(model_input)
            features_list.append(features.detach().cpu().numpy())
            labels_list.append(y.detach().cpu().numpy())

    x = np.concatenate(features_list, axis=0) if features_list else np.empty((0, 0), dtype=np.float32)
    y = np.concatenate(labels_list, axis=0) if labels_list else np.empty((0,), dtype=np.int64)
    return x, y


def extract_model_logits(
    model: torch.nn.Module,
    split: Any,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = split_to_dataloader(split, batch_size=int(batch_size), shuffle=False)
    model = model.to(device)
    model.eval()

    logits_list = []
    labels_list = []
    with torch.no_grad():
        for batch in loader:
            model_input, y = parse_torch_batch(batch, device)
            logits = model(model_input)
            logits_list.append(logits.detach().cpu().numpy())
            labels_list.append(y.detach().cpu().numpy())

    logits = np.concatenate(logits_list, axis=0) if logits_list else np.empty((0, 0), dtype=np.float32)
    labels = np.concatenate(labels_list, axis=0) if labels_list else np.empty((0,), dtype=np.int64)
    return logits, labels


def normalize_detector_cfg(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if hasattr(cfg, "items"):
        try:
            return {str(k): v for k, v in cfg.items()}
        except Exception:
            pass
    if hasattr(cfg, "__dict__"):
        return {str(k): v for k, v in vars(cfg).items() if not str(k).startswith("_")}
    return {"value": cfg}


def compute_class_detection_metrics(
    *,
    predicted_is_infected: Optional[bool],
    true_is_infected: Optional[bool],
    predicted_target_class: Optional[int],
    true_target_class: Optional[int],
    predicted_source_class: Optional[int],
    true_source_class: Optional[int],
    runtime_sec: float,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {"detection/runtime_sec": float(runtime_sec)}
    if predicted_is_infected is not None and true_is_infected is not None:
        metrics["detection/is_infected_accuracy"] = float(bool(predicted_is_infected) == bool(true_is_infected))
        if not bool(true_is_infected) and bool(predicted_is_infected):
            metrics["detection/false_positive_rate"] = 1.0
            metrics["detection/true_positive_rate"] = 0.0
        elif bool(true_is_infected) and bool(predicted_is_infected):
            metrics["detection/false_positive_rate"] = 0.0
            metrics["detection/true_positive_rate"] = 1.0
        elif bool(true_is_infected) and not bool(predicted_is_infected):
            metrics["detection/false_positive_rate"] = 0.0
            metrics["detection/true_positive_rate"] = 0.0
        else:
            metrics["detection/false_positive_rate"] = 0.0
            metrics["detection/true_positive_rate"] = 0.0

    if predicted_target_class is not None and true_target_class is not None:
        metrics["detection/target_class_accuracy"] = float(int(predicted_target_class) == int(true_target_class))
    if predicted_source_class is not None and true_source_class is not None:
        metrics["detection/source_class_accuracy"] = float(int(predicted_source_class) == int(true_source_class))
    return metrics


def enrich_class_decision_result(result: DetectorResult) -> None:
    scores = None if result.class_scores is None else np.asarray(result.class_scores, dtype=np.float32).reshape(-1)
    candidate = result.candidate_target_class
    if candidate is None and result.predicted_target_class is not None:
        candidate = int(result.predicted_target_class)
    if candidate is None and scores is not None and scores.size > 0 and np.any(np.isfinite(scores)):
        candidate = int(np.nanargmax(scores))

    candidate_score = result.candidate_target_score
    if candidate_score is None and candidate is not None and scores is not None:
        if 0 <= int(candidate) < int(scores.shape[0]):
            candidate_score = float(scores[int(candidate)])

    thresholds = result.thresholds or {}
    decision_score = result.decision_score
    if decision_score is None:
        value = thresholds.get("decision_score")
        if value is not None:
            decision_score = float(value)

    decision_threshold = result.decision_threshold
    if decision_threshold is None:
        value = thresholds.get("decision_threshold")
        if value is not None:
            decision_threshold = float(value)

    greater_is_infected = result.decision_greater_is_infected
    if greater_is_infected is None and "decision_greater_is_infected" in thresholds:
        greater_is_infected = bool(thresholds["decision_greater_is_infected"])

    decision_margin = result.decision_margin
    if (
        decision_margin is None
        and decision_score is not None
        and decision_threshold is not None
        and greater_is_infected is not None
    ):
        if bool(greater_is_infected):
            decision_margin = float(decision_score - decision_threshold)
        else:
            decision_margin = float(decision_threshold - decision_score)

    result.candidate_target_class = None if candidate is None else int(candidate)
    result.candidate_target_score = candidate_score
    result.decision_score = decision_score
    result.decision_threshold = decision_threshold
    result.decision_margin = decision_margin
    result.decision_greater_is_infected = greater_is_infected

    updates: Dict[str, Any] = {}
    if result.candidate_target_class is not None:
        updates["detection/candidate_target_class"] = float(result.candidate_target_class)
    if candidate_score is not None:
        updates["detection/candidate_target_score"] = float(candidate_score)
    if decision_score is not None:
        updates["detection/decision_score"] = float(decision_score)
    if decision_threshold is not None:
        updates["detection/decision_threshold"] = float(decision_threshold)
    if decision_margin is not None:
        updates["detection/decision_margin"] = float(decision_margin)
    if greater_is_infected is not None:
        updates["detection/decision_greater_is_infected"] = float(bool(greater_is_infected))
    result.summary_metrics = merge_metric_dicts(result.summary_metrics, updates)


def merge_metric_dicts(*metric_dicts: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for metric_dict in metric_dicts:
        merged.update(metric_dict)
    return merged


def _write_json(path: Path, payload: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return str(path)


def write_summary(
    *,
    output_dir: str,
    result: DetectorResult,
    context: DetectorContext,
    resolved_cfg: Dict[str, Any],
) -> str:
    result.artifacts.summary_json = str(Path(output_dir) / "summary.json")
    payload = {
        **result.to_summary_dict(),
        "context": context.to_jsonable(),
        "resolved_cfg": _to_jsonable(resolved_cfg),
    }
    return _write_json(Path(output_dir) / "summary.json", payload)


def write_class_scores(output_dir: str, result: DetectorResult) -> Optional[str]:
    if result.class_scores is None:
        return None
    scores = np.asarray(result.class_scores, dtype=np.float32)
    frame = pd.DataFrame({"class_index": np.arange(scores.shape[0], dtype=np.int64), "score": scores})
    path = Path(output_dir) / "class_scores.csv"
    frame.to_csv(path, index=False)
    return str(path)


def write_class_details(output_dir: str, result: DetectorResult) -> Optional[str]:
    if result.class_details is None:
        return None
    frame = pd.DataFrame(result.class_details)
    path = Path(output_dir) / "class_details.csv"
    frame.to_csv(path, index=False)
    return str(path)


def write_pair_scores(output_dir: str, result: DetectorResult) -> Optional[str]:
    if result.pair_scores is None:
        return None
    frame = pd.DataFrame(result.pair_scores)
    path = Path(output_dir) / "pair_scores.csv"
    frame.to_csv(path, index=False)
    return str(path)


def write_optimization_trace(output_dir: str, result: DetectorResult) -> Optional[str]:
    if result.optimization_trace is None:
        return None
    return _write_json(Path(output_dir) / "optimization_trace.json", result.optimization_trace)


def write_estimated_pattern(output_dir: str, result: DetectorResult) -> Optional[str]:
    array = result.estimated_trigger
    if array is None:
        array = result.estimated_mask
    if array is None:
        array = result.estimated_perturbation
    if array is None:
        return None
    path = Path(output_dir) / "estimated_pattern.npy"
    np.save(path, np.asarray(array))
    return str(path)


def write_estimated_trigger(output_dir: str, result: DetectorResult) -> Optional[str]:
    if result.estimated_trigger is None:
        return None
    path = Path(output_dir) / "estimated_trigger.npy"
    np.save(path, np.asarray(result.estimated_trigger))
    return str(path)


def write_estimated_mask(output_dir: str, result: DetectorResult) -> Optional[str]:
    if result.estimated_mask is None:
        return None
    path = Path(output_dir) / "estimated_mask.npy"
    np.save(path, np.asarray(result.estimated_mask))
    return str(path)


def write_optimized_inputs(output_dir: str, result: DetectorResult) -> Optional[str]:
    if result.optimized_inputs is None:
        return None
    path = Path(output_dir) / "optimized_inputs.npy"
    np.save(path, np.asarray(result.optimized_inputs))
    return str(path)


def write_candidate_margin_vectors(output_dir: str, result: DetectorResult) -> Optional[str]:
    if result.candidate_margin_vectors is None:
        return None
    path = Path(output_dir) / "candidate_margin_vectors.npy"
    np.save(path, np.asarray(result.candidate_margin_vectors))
    return str(path)


def write_candidate_objective_vectors(output_dir: str, result: DetectorResult) -> Optional[str]:
    if result.candidate_objective_vectors is None:
        return None
    path = Path(output_dir) / "candidate_objective_vectors.npy"
    np.save(path, np.asarray(result.candidate_objective_vectors))
    return str(path)


def save_detection_artifacts(
    *,
    output_dir: str,
    result: DetectorResult,
    context: DetectorContext,
    resolved_cfg: Dict[str, Any],
) -> ArtifactIndex:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    index = result.artifacts
    index.class_scores_csv = write_class_scores(str(out), result)
    class_details_csv = write_class_details(str(out), result)
    if class_details_csv is not None:
        index.extra_files["class_details_csv"] = class_details_csv
    pair_scores_csv = write_pair_scores(str(out), result)
    if pair_scores_csv is not None:
        index.extra_files["pair_scores_csv"] = pair_scores_csv
    index.optimization_trace_json = write_optimization_trace(str(out), result)
    index.estimated_pattern_npy = write_estimated_pattern(str(out), result)
    estimated_trigger_npy = write_estimated_trigger(str(out), result)
    if estimated_trigger_npy is not None:
        index.extra_files["estimated_trigger_npy"] = estimated_trigger_npy
    estimated_mask_npy = write_estimated_mask(str(out), result)
    if estimated_mask_npy is not None:
        index.extra_files["estimated_mask_npy"] = estimated_mask_npy
    optimized_inputs_npy = write_optimized_inputs(str(out), result)
    if optimized_inputs_npy is not None:
        index.extra_files["optimized_inputs_npy"] = optimized_inputs_npy
    candidate_objective_vectors_npy = write_candidate_objective_vectors(str(out), result)
    if candidate_objective_vectors_npy is not None:
        index.extra_files["candidate_objective_vectors_npy"] = candidate_objective_vectors_npy
    candidate_margin_vectors_npy = write_candidate_margin_vectors(str(out), result)
    if candidate_margin_vectors_npy is not None:
        index.extra_files["candidate_margin_vectors_npy"] = candidate_margin_vectors_npy
    index.summary_json = write_summary(output_dir=str(out), result=result, context=context, resolved_cfg=resolved_cfg)
    return index
