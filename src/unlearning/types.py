from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "to_jsonable"):
        return value.to_jsonable()
    if isinstance(value, DictConfig):
        # Convert Hydra DictConfig to regular dict
        return _to_jsonable(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, ListConfig):
        return _to_jsonable(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
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
    return value


@dataclass
class ForgetSet:
    indices: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    scores: Optional[np.ndarray] = None
    flags: Optional[np.ndarray] = None
    source: str = "none"
    index_space: str = "train_local"
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.indices = np.asarray(self.indices, dtype=np.int64).reshape(-1)
        if self.scores is not None:
            self.scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        if self.flags is not None:
            self.flags = np.asarray(self.flags, dtype=np.int64).reshape(-1)

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "indices": _to_jsonable(self.indices),
            "scores": _to_jsonable(self.scores),
            "flags": _to_jsonable(self.flags),
            "source": self.source,
            "index_space": self.index_space,
            "notes": self.notes,
            "metadata": _to_jsonable(self.metadata),
        }


@dataclass
class UnlearningContext:
    model: Any
    datasets: Dict[str, Any]
    attack_result: Any = None
    detection_result: Any = None
    model_cfg: Optional[Dict[str, Any]] = None
    train_cfg: Optional[Dict[str, Any]] = None
    seed: int = 42
    device: Any = "cpu"
    num_classes: Optional[int] = None
    class_names: Optional[List[str]] = None
    run_dir: Optional[str] = None
    method_cfg: Optional[Dict[str, Any]] = None
    model_name: Optional[str] = None
    model_family: Optional[str] = None
    target_label: Optional[int] = None
    train_sample_indices: Optional[np.ndarray] = None
    detection_sample_indices: Optional[np.ndarray] = None
    feature_metadata: Any = None
    attack_metadata: Optional[Dict[str, Any]] = None
    clean_support_split: Any = None

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "seed": int(self.seed),
            "device": str(self.device),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "run_dir": self.run_dir,
            "target_label": self.target_label,
            "train_sample_indices": _to_jsonable(self.train_sample_indices),
            "detection_sample_indices": _to_jsonable(self.detection_sample_indices),
            "feature_metadata": _to_jsonable(self.feature_metadata),
            "attack_metadata": _to_jsonable(self.attack_metadata),
            "model_cfg": _to_jsonable(self.model_cfg),
            "train_cfg": _to_jsonable(self.train_cfg),
            "method_cfg": _to_jsonable(self.method_cfg),
        }


@dataclass
class UnlearningArtifacts:
    summary_json: Optional[str] = None
    metrics_before_json: Optional[str] = None
    metrics_after_json: Optional[str] = None
    forget_indices_npy: Optional[str] = None
    retain_indices_npy: Optional[str] = None
    train_sample_indices_npy: Optional[str] = None
    detection_sample_indices_npy: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    extra_files: Dict[str, str] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "summary_json": self.summary_json,
            "metrics_before_json": self.metrics_before_json,
            "metrics_after_json": self.metrics_after_json,
            "forget_indices_npy": self.forget_indices_npy,
            "retain_indices_npy": self.retain_indices_npy,
            "train_sample_indices_npy": self.train_sample_indices_npy,
            "detection_sample_indices_npy": self.detection_sample_indices_npy,
            "checkpoint_dir": self.checkpoint_dir,
            "extra_files": dict(self.extra_files),
        }


@dataclass
class UnlearningResult:
    method_name: str
    track_type: str
    status: str
    seed: int
    runtime_sec: float
    forget_set_source: Optional[str] = None
    num_removed: int = 0
    num_retained: int = 0
    metrics_before: Dict[str, Any] = field(default_factory=dict)
    metrics_after: Dict[str, Any] = field(default_factory=dict)
    summary_metrics: Dict[str, Any] = field(default_factory=dict)
    removed_indices: Optional[np.ndarray] = None
    retain_indices: Optional[np.ndarray] = None
    artifacts: UnlearningArtifacts = field(default_factory=UnlearningArtifacts)
    deviation_note: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    model_after: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.removed_indices is not None:
            self.removed_indices = np.asarray(self.removed_indices, dtype=np.int64).reshape(-1)
        if self.retain_indices is not None:
            self.retain_indices = np.asarray(self.retain_indices, dtype=np.int64).reshape(-1)

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "track_type": self.track_type,
            "status": self.status,
            "seed": int(self.seed),
            "runtime_sec": float(self.runtime_sec),
            "forget_set_source": self.forget_set_source,
            "num_removed": int(self.num_removed),
            "num_retained": int(self.num_retained),
            "summary_metrics": _to_jsonable(self.summary_metrics),
            "removed_indices_count": 0 if self.removed_indices is None else int(len(self.removed_indices)),
            "retain_indices_count": 0 if self.retain_indices is None else int(len(self.retain_indices)),
            "checkpoint_dir": self.checkpoint_dir,
            "artifacts": self.artifacts.to_jsonable(),
            "deviation_note": self.deviation_note,
        }

    def save_summary(self, output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_summary_dict(), handle, indent=2, sort_keys=True)
        self.artifacts.summary_json = str(path)
        return str(path)
