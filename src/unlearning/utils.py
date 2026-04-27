from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ..models.train import compute_backdoor_metrics
from ..models.utils import parse_torch_batch, resolve_device as model_resolve_device, set_seed as model_set_seed, split_to_dataloader
from .types import ForgetSet, UnlearningArtifacts, UnlearningContext, UnlearningResult, _to_jsonable


def set_seed(seed: Optional[int]) -> None:
    model_set_seed(seed)


def resolve_device(device: Optional[str]) -> torch.device:
    return model_resolve_device(device)

def measure_runtime(fn, *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    start = perf_counter()
    result = fn(*args, **kwargs)
    return result, float(perf_counter() - start)


def normalize_cfg(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if hasattr(cfg, "items"):
        return {str(k): _to_jsonable(v) for k, v in cfg.items()}
    if hasattr(cfg, "__dict__"):
        return {str(k): _to_jsonable(v) for k, v in vars(cfg).items() if not str(k).startswith("_")}
    return {"value": _to_jsonable(cfg)}


def get_obj_field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if hasattr(obj, "get"):
        return obj.get(name, default)
    return getattr(obj, name, default)


def count_split_samples(split: Any) -> int:
    if isinstance(split, dict):
        if "y" in split:
            return int(len(split["y"]))
        if "x" in split:
            return int(len(split["x"]))
        if "x_num" in split:
            return int(len(split["x_num"]))
    if isinstance(split, (tuple, list)) and split:
        return int(len(split[0]))
    raise ValueError(f"Unsupported split type: {type(split)}")


def stable_unique_indices(indices: Any, *, upper_bound: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    if upper_bound is not None:
        arr = arr[(arr >= 0) & (arr < int(upper_bound))]
    if arr.size == 0:
        return np.empty((0,), dtype=np.int64)

    seen = set()
    ordered = []
    for index in arr.tolist():
        if index in seen:
            continue
        seen.add(index)
        ordered.append(index)
    return np.asarray(ordered, dtype=np.int64)


def complement_indices(num_samples: int, removed_indices: Any) -> np.ndarray:
    removed = stable_unique_indices(removed_indices, upper_bound=int(num_samples))
    mask = np.ones(int(num_samples), dtype=bool)
    mask[removed] = False
    return np.flatnonzero(mask).astype(np.int64)


def normalize_forget_set(
    forget_set: ForgetSet,
    *,
    num_train: int,
    train_sample_indices: Any = None,
) -> ForgetSet:
    raw_indices = np.asarray(forget_set.indices, dtype=np.int64).reshape(-1)
    input_index_space = str(getattr(forget_set, "index_space", "train_local") or "train_local")
    index_space = input_index_space.lower()

    if index_space in {"train_local", "local", "train"}:
        local_candidates = raw_indices
        raw_positions = np.arange(raw_indices.shape[0], dtype=np.int64)
    elif index_space in {"original", "dataset", "train_original", "original_dataset"}:
        assert train_sample_indices is not None, (
            f"Forget set index_space='{input_index_space}' requires context.train_sample_indices."
        )
        train_ids = np.asarray(train_sample_indices, dtype=np.int64).reshape(-1)
        assert train_ids.shape[0] == int(num_train), (
            "context.train_sample_indices must have the same length as the train split."
        )
        local_by_original = {int(original): int(local) for local, original in enumerate(train_ids.tolist())}
        local_values = []
        kept_raw_positions = []
        for raw_pos, original_index in enumerate(raw_indices.tolist()):
            local_index = local_by_original.get(int(original_index))
            if local_index is None:
                continue
            local_values.append(local_index)
            kept_raw_positions.append(raw_pos)
        local_candidates = np.asarray(local_values, dtype=np.int64)
        raw_positions = np.asarray(kept_raw_positions, dtype=np.int64)
    else:
        raise AssertionError(f"Unsupported forget_set.index_space: {input_index_space}")

    seen = set()
    local_indices = []
    selected_raw_positions = []
    for candidate, raw_pos in zip(local_candidates.tolist(), raw_positions.tolist()):
        candidate = int(candidate)
        if candidate < 0 or candidate >= int(num_train) or candidate in seen:
            continue
        seen.add(candidate)
        local_indices.append(candidate)
        selected_raw_positions.append(int(raw_pos))

    selected = np.asarray(local_indices, dtype=np.int64)
    selected_positions = np.asarray(selected_raw_positions, dtype=np.int64)

    scores = None
    if forget_set.scores is not None:
        raw_scores = np.asarray(forget_set.scores, dtype=np.float32).reshape(-1)
        if raw_scores.shape[0] == raw_indices.shape[0]:
            scores = raw_scores[selected_positions].astype(np.float32, copy=False)

    flags = None
    if forget_set.flags is not None:
        raw_flags = np.asarray(forget_set.flags, dtype=np.int64).reshape(-1)
        if raw_flags.shape[0] == raw_indices.shape[0]:
            flags = raw_flags[selected_positions].astype(np.int64, copy=False)

    metadata = dict(getattr(forget_set, "metadata", {}) or {})
    metadata.update(
        {
            "input_index_space": input_index_space,
            "output_index_space": "train_local",
            "num_raw_indices": int(raw_indices.shape[0]),
            "num_valid_unique_indices": int(selected.shape[0]),
        }
    )
    return ForgetSet(
        indices=selected,
        scores=scores,
        flags=flags,
        source=forget_set.source,
        index_space="train_local",
        notes=forget_set.notes,
        metadata=metadata,
    )


def _subset_value(value: Any, indices: np.ndarray) -> Any:
    if value is None:
        return None
    arr = np.asarray(value)
    return arr[indices].copy()


def subset_split(split: Any, indices: Any) -> Any:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)

    if isinstance(split, dict):
        out: Dict[str, Any] = {}
        for key, value in split.items():
            if key in {"x", "y", "x_num", "x_cat"}:
                out[key] = _subset_value(value, idx)
            else:
                out[key] = value
        return out

    if isinstance(split, tuple):
        return tuple(_subset_value(value, idx) for value in split)

    if isinstance(split, list):
        return [_subset_value(value, idx) for value in split]

    raise ValueError(f"Unsupported split type: {type(split)}")


def build_retain_datasets(datasets: Dict[str, Any], retain_indices: Any) -> Dict[str, Any]:
    retained = dict(datasets)
    retained["train"] = subset_split(datasets["train"], retain_indices)
    return retained


def split_train_forget_retain(train_split: Any, forget_indices: Any) -> Tuple[Any, Any, np.ndarray, np.ndarray]:
    num_train = count_split_samples(train_split)
    forget_indices_np = stable_unique_indices(forget_indices, upper_bound=num_train)
    retain_indices_np = complement_indices(num_train, forget_indices_np)
    forget_split = subset_split(train_split, forget_indices_np)
    retain_split = subset_split(train_split, retain_indices_np)
    return forget_split, retain_split, forget_indices_np, retain_indices_np


def resolve_target_label(context: UnlearningContext) -> Optional[int]:
    if context.target_label is not None:
        return int(context.target_label)

    attack_result = context.attack_result
    if attack_result is not None:
        target_label = get_obj_field(attack_result, "target_label", None)
        if target_label is not None:
            return int(target_label)

    for cfg in (context.method_cfg, context.train_cfg, context.model_cfg):
        if not isinstance(cfg, dict):
            continue
        target_label = cfg.get("target_label")
        if target_label is not None:
            return int(target_label)

    return None


def _evaluate_split(
    model: nn.Module,
    *,
    split: Any,
    device: torch.device,
    batch_size: int,
    prefix: str,
    per_class: bool = False,
) -> Dict[str, float]:
    loader = split_to_dataloader(split, batch_size=int(batch_size), shuffle=False)
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0.0
    total_batches = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            model_input, y = parse_torch_batch(batch, device)
            logits = model(model_input)
            loss = model.compute_eval_loss(logits, y) if hasattr(model, "compute_eval_loss") else criterion(logits, y)
            preds = torch.argmax(logits, dim=1)

            y_true.append(y.detach().cpu().numpy())
            y_pred.append(preds.detach().cpu().numpy())
            total_loss += float(loss.item())
            total_batches += 1

    y_true_np = np.concatenate(y_true) if y_true else np.empty((0,), dtype=np.int64)
    y_pred_np = np.concatenate(y_pred) if y_pred else np.empty((0,), dtype=np.int64)
    avg_loss = total_loss / max(total_batches, 1)
    accuracy = float(accuracy_score(y_true_np, y_pred_np)) if y_true_np.size > 0 else 0.0
    metrics = {
        f"{prefix}/loss": float(avg_loss),
        f"{prefix}/accuracy": float(accuracy),
    }
    if per_class and y_true_np.size > 0:
        for class_id in sorted(np.unique(y_true_np).astype(np.int64).tolist()):
            mask = y_true_np == int(class_id)
            support = int(mask.sum())
            if support <= 0:
                continue
            metrics[f"{prefix}/class_{class_id}_accuracy"] = float((y_pred_np[mask] == y_true_np[mask]).mean())
            metrics[f"{prefix}/class_{class_id}_support"] = float(support)
    return metrics


def evaluate_model(
    model: nn.Module,
    *,
    datasets: Dict[str, Any],
    target_label: Optional[int],
    device: Any,
    batch_size: int = 512,
    per_class: bool = False,
) -> Dict[str, Any]:
    device_resolved = resolve_device(None if device is None else str(device))
    model = model.to(device_resolved)

    metrics: Dict[str, Any] = {}
    split_prefixes = (
        ("train", "clean/train"),
        ("val", "clean/val"),
        ("test", "clean/test"),
        ("forget", "forget/train"),
        ("retain", "retain/train"),
    )
    for split_name, prefix in split_prefixes:
        if split_name not in datasets:
            continue
        metrics.update(
            _evaluate_split(
                model,
                split=datasets[split_name],
                device=device_resolved,
                batch_size=batch_size,
                prefix=prefix,
                per_class=per_class,
            )
        )

    if target_label is not None and "test_triggered" in datasets and "test_clean_labels" in datasets:
        loader = split_to_dataloader(datasets["test_triggered"], batch_size=int(batch_size), shuffle=False)
        preds = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                model_input, _ = parse_torch_batch(batch, device_resolved)
                logits = model(model_input)
                preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())

        if preds:
            y_pred_triggered = np.concatenate(preds)
            y_clean_reference = np.asarray(datasets["test_clean_labels"], dtype=np.int64)
            metrics.update(
                compute_backdoor_metrics(
                    y_pred_triggered=y_pred_triggered,
                    y_clean_reference=y_clean_reference,
                    target_label=int(target_label),
                )
            )
            if per_class:
                for source_label in sorted(np.unique(y_clean_reference).astype(np.int64).tolist()):
                    if int(source_label) == int(target_label):
                        continue
                    mask = y_clean_reference == int(source_label)
                    support = int(mask.sum())
                    if support <= 0:
                        continue
                    metrics[f"backdoor/asr_source_{source_label}"] = float(
                        (y_pred_triggered[mask] == int(target_label)).mean()
                    )
                    metrics[f"backdoor/source_{source_label}_support"] = float(support)

    return metrics


def phase_metrics(metrics: Dict[str, Any], phase: str) -> Dict[str, Any]:
    return {f"{key}_{phase}": value for key, value in metrics.items()}


def compute_delta_metrics(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    pairs = {
        "unlearning/delta_clean_accuracy": "clean/test/accuracy",
        "unlearning/delta_clean_val_accuracy": "clean/val/accuracy",
        "unlearning/delta_clean_train_accuracy": "clean/train/accuracy",
        "unlearning/delta_asr": "backdoor/asr",
        "unlearning/delta_backdoor_accuracy": "backdoor/accuracy",
        "unlearning/delta_forget_accuracy": "forget/train/accuracy",
        "unlearning/delta_retain_accuracy": "retain/train/accuracy",
    }
    for out_key, metric_key in pairs.items():
        if metric_key in before and metric_key in after:
            deltas[out_key] = float(after[metric_key]) - float(before[metric_key])
    return deltas


def compute_forget_metrics(
    *,
    removed_indices: Any,
    poisoned_indices: Any,
    num_candidates: int,
) -> Dict[str, float]:
    removed = stable_unique_indices(removed_indices, upper_bound=int(num_candidates))
    metrics: Dict[str, float] = {
        "unlearning/forget_size": float(len(removed)),
        "unlearning/retain_size": float(max(int(num_candidates) - len(removed), 0)),
        "unlearning/remove_fraction": float(len(removed) / max(int(num_candidates), 1)),
    }
    if poisoned_indices is None:
        return metrics

    poisoned = stable_unique_indices(poisoned_indices, upper_bound=int(num_candidates))
    y_true = np.zeros(int(num_candidates), dtype=np.int64)
    y_pred = np.zeros(int(num_candidates), dtype=np.int64)
    y_true[poisoned] = 1
    y_pred[removed] = 1

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    metrics.update(
        {
            "unlearning/forget_precision": float(precision),
            "unlearning/forget_recall": float(recall),
            "unlearning/forget_f1": float(f1),
        }
    )
    return metrics


def merge_metric_dicts(*metric_dicts: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for metric_dict in metric_dicts:
        merged.update(metric_dict)
    return merged


def scalar_metrics_only(metrics: Dict[str, Any]) -> Dict[str, float]:
    scalar_metrics: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            scalar_metrics[key] = float(value)
    return scalar_metrics


def _write_json(path: Path, payload: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return str(path)


def write_forget_scores(output_dir: str, forget_set: ForgetSet) -> Optional[str]:
    if forget_set.scores is None:
        return None

    path = Path(output_dir) / "forget_scores.csv"
    flags = None if forget_set.flags is None else np.asarray(forget_set.flags, dtype=np.int64).reshape(-1)
    scores = np.asarray(forget_set.scores, dtype=np.float32).reshape(-1)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["index", "score"]
        if flags is not None:
            header.append("flag")
        writer.writerow(header)
        for row_id, index in enumerate(forget_set.indices.astype(np.int64).tolist()):
            row = [index, float(scores[row_id])]
            if flags is not None:
                row.append(int(flags[row_id]))
            writer.writerow(row)
    return str(path)


def save_unlearning_artifacts(
    *,
    output_dir: str,
    result: UnlearningResult,
    context: UnlearningContext,
    forget_set: ForgetSet,
    resolved_cfg: Dict[str, Any],
) -> UnlearningArtifacts:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    existing_artifacts = result.artifacts
    artifacts = UnlearningArtifacts()
    artifacts.extra_files.update(dict(existing_artifacts.extra_files))
    if existing_artifacts.checkpoint_dir is not None:
        artifacts.checkpoint_dir = str(existing_artifacts.checkpoint_dir)
    if existing_artifacts.summary_json is not None:
        artifacts.summary_json = str(existing_artifacts.summary_json)
    if existing_artifacts.metrics_before_json is not None:
        artifacts.metrics_before_json = str(existing_artifacts.metrics_before_json)
    if existing_artifacts.metrics_after_json is not None:
        artifacts.metrics_after_json = str(existing_artifacts.metrics_after_json)
    if existing_artifacts.forget_indices_npy is not None:
        artifacts.forget_indices_npy = str(existing_artifacts.forget_indices_npy)
    if existing_artifacts.retain_indices_npy is not None:
        artifacts.retain_indices_npy = str(existing_artifacts.retain_indices_npy)

    artifacts.metrics_before_json = _write_json(out / "metrics_before.json", _to_jsonable(result.metrics_before))
    artifacts.metrics_after_json = _write_json(out / "metrics_after.json", _to_jsonable(result.metrics_after))

    forget_indices = result.removed_indices if result.removed_indices is not None else forget_set.indices
    np.save(out / "forget_indices.npy", np.asarray(forget_indices, dtype=np.int64))
    artifacts.forget_indices_npy = str(out / "forget_indices.npy")

    retain_indices = result.retain_indices if result.retain_indices is not None else np.empty((0,), dtype=np.int64)
    np.save(out / "retain_indices.npy", np.asarray(retain_indices, dtype=np.int64))
    artifacts.retain_indices_npy = str(out / "retain_indices.npy")

    if result.checkpoint_dir is not None:
        artifacts.checkpoint_dir = str(result.checkpoint_dir)

    forget_scores_csv = write_forget_scores(output_dir, forget_set)
    if forget_scores_csv is not None:
        artifacts.extra_files["forget_scores_csv"] = forget_scores_csv

    result.artifacts = artifacts
    payload = {
        **result.to_summary_dict(),
        "context": context.to_jsonable(),
        "forget_set": forget_set.to_jsonable(),
        "resolved_cfg": _to_jsonable(resolved_cfg),
    }
    artifacts.summary_json = _write_json(out / "summary.json", payload)
    result.artifacts = artifacts
    return artifacts
