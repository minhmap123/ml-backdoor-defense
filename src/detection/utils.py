from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from ..models.utils import parse_torch_batch, resolve_device as model_resolve_device, set_seed as model_set_seed, split_to_dataloader
from .types import ArtifactIndex, DetectorContext, DetectorResult, _to_jsonable


def set_seed(seed: Optional[int]) -> None:
    model_set_seed(seed)


def resolve_device(device: Optional[str]) -> torch.device:
    return model_resolve_device(device)


def ensure_run_dir(path: str | Path) -> str:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


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


def clamp_numeric_input(x: np.ndarray, lower: Optional[np.ndarray], upper: Optional[np.ndarray]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if lower is not None:
        x = np.maximum(x, np.asarray(lower, dtype=np.float32))
    if upper is not None:
        x = np.minimum(x, np.asarray(upper, dtype=np.float32))
    return x


def rank_desc(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    return np.argsort(scores)[::-1].astype(np.int64)


def rank_asc(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    return np.argsort(scores).astype(np.int64)


def derive_restart_seeds(seed: int, num_restarts: int) -> np.ndarray:
    base = int(seed)
    return np.asarray([base + rid for rid in range(int(num_restarts))], dtype=np.int64)


def count_split_samples(split: Any) -> int:
    if isinstance(split, dict):
        if "y" in split:
            return int(len(split["y"]))
        if "x" in split:
            return int(len(split["x"]))
    if isinstance(split, (tuple, list)) and split:
        return int(len(split[0]))
    raise ValueError(f"Unsupported split type: {type(split)}")


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


def compute_sample_detection_metrics(
    *,
    sample_scores: Optional[np.ndarray],
    sample_flags: Optional[np.ndarray],
    poisoned_indices: Optional[np.ndarray],
    num_candidates: int,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {"detection/num_candidates": float(num_candidates)}
    if poisoned_indices is None or num_candidates <= 0:
        return metrics

    poisoned_indices = np.asarray(poisoned_indices, dtype=np.int64)
    y_true = np.zeros(int(num_candidates), dtype=np.int64)
    valid_poisoned = poisoned_indices[(poisoned_indices >= 0) & (poisoned_indices < num_candidates)]
    y_true[valid_poisoned] = 1

    if sample_flags is not None:
        y_pred = np.asarray(sample_flags, dtype=np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        metrics.update(
            {
                "detection/precision": float(precision),
                "detection/recall": float(recall),
                "detection/f1": float(f1),
            }
        )

    if sample_scores is not None:
        scores = np.asarray(sample_scores, dtype=np.float32)
        if len(np.unique(y_true)) > 1:
            metrics["detection/auroc"] = float(roc_auc_score(y_true, scores))
            metrics["detection/average_precision"] = float(average_precision_score(y_true, scores))

    return metrics


def compute_topk_recall(sample_ranking: np.ndarray, poisoned_indices: np.ndarray, k: int) -> float:
    ranking = np.asarray(sample_ranking, dtype=np.int64)
    poisoned = set(np.asarray(poisoned_indices, dtype=np.int64).tolist())
    if not poisoned or k <= 0:
        return 0.0
    topk = set(ranking[: int(k)].tolist())
    return float(len(topk & poisoned) / max(len(poisoned), 1))


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


def write_sample_scores(output_dir: str, result: DetectorResult, context: DetectorContext) -> Optional[str]:
    if result.sample_scores is None:
        return None
    scores = np.asarray(result.sample_scores)
    ranking = None if result.sample_ranking is None else np.asarray(result.sample_ranking, dtype=np.int64)
    flags = None if result.sample_flags is None else np.asarray(result.sample_flags, dtype=np.int64)
    labels = None if result.sample_labels is None else np.asarray(result.sample_labels)
    sample_indices = None if context.sample_indices is None else np.asarray(context.sample_indices, dtype=np.int64)

    rank_positions = np.full(scores.shape[0], -1, dtype=np.int64)
    if ranking is not None:
        rank_positions[ranking] = np.arange(ranking.shape[0], dtype=np.int64)

    payload = {
        "sample_index": np.arange(scores.shape[0], dtype=np.int64),
        "score": scores.astype(np.float32),
        "rank_position": rank_positions,
        "flag": None if flags is None else flags,
    }
    if sample_indices is not None:
        payload["original_index"] = sample_indices
    if labels is not None:
        payload["label"] = labels

    frame = pd.DataFrame(payload)
    path = Path(output_dir) / "sample_scores.csv"
    frame.to_csv(path, index=False)
    return str(path)


def write_raw_sample_scores(output_dir: str, result: DetectorResult, context: DetectorContext) -> Optional[str]:
    if result.raw_sample_scores is None:
        return None

    raw_scores = np.asarray(result.raw_sample_scores)
    decision_scores = None if result.sample_scores is None else np.asarray(result.sample_scores)
    flags = None if result.sample_flags is None else np.asarray(result.sample_flags, dtype=np.int64)
    labels = None if result.sample_labels is None else np.asarray(result.sample_labels)
    sample_indices = None if context.sample_indices is None else np.asarray(context.sample_indices, dtype=np.int64)

    payload = {
        "sample_index": np.arange(raw_scores.shape[0], dtype=np.int64),
        "raw_score": raw_scores.astype(np.float32),
    }
    if decision_scores is not None:
        payload["decision_score"] = decision_scores.astype(np.float32)
    if flags is not None:
        payload["flag"] = flags
    if sample_indices is not None:
        payload["original_index"] = sample_indices
    if labels is not None:
        payload["label"] = labels

    frame = pd.DataFrame(payload)
    path = Path(output_dir) / "raw_sample_scores.csv"
    frame.to_csv(path, index=False)
    return str(path)


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


def resolve_suspect_indices(result: DetectorResult, context: DetectorContext) -> Optional[np.ndarray]:
    local_indices: Optional[np.ndarray]
    if result.suspect_indices is not None:
        local_indices = np.asarray(result.suspect_indices, dtype=np.int64)
    elif result.sample_flags is not None:
        local_indices = np.flatnonzero(np.asarray(result.sample_flags, dtype=np.int64)).astype(np.int64)
    else:
        return None
    if context.sample_indices is None:
        return local_indices

    sample_indices = np.asarray(context.sample_indices, dtype=np.int64)
    if local_indices.size == 0:
        return local_indices
    if int(local_indices.max()) >= int(sample_indices.shape[0]):
        raise ValueError("sample_flags reference indices outside DetectorContext.sample_indices.")
    return sample_indices[local_indices].astype(np.int64, copy=False)


def write_suspect_indices(output_dir: str, result: DetectorResult, context: DetectorContext) -> Optional[str]:
    suspect_indices = resolve_suspect_indices(result, context)
    if suspect_indices is None:
        return None
    path = Path(output_dir) / "suspect_indices.npy"
    np.save(path, np.asarray(suspect_indices, dtype=np.int64))
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
    index.raw_scores_csv = write_sample_scores(str(out), result, context)
    raw_sample_scores_csv = write_raw_sample_scores(str(out), result, context)
    if raw_sample_scores_csv is not None:
        index.extra_files["raw_sample_scores_csv"] = raw_sample_scores_csv
    index.class_scores_csv = write_class_scores(str(out), result)
    class_details_csv = write_class_details(str(out), result)
    if class_details_csv is not None:
        index.extra_files["class_details_csv"] = class_details_csv
    pair_scores_csv = write_pair_scores(str(out), result)
    if pair_scores_csv is not None:
        index.extra_files["pair_scores_csv"] = pair_scores_csv
    index.suspect_indices_npy = write_suspect_indices(str(out), result, context)
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
