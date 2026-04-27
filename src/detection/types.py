from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import numpy as np
import pandas as pd


def _to_jsonable(value: Any) -> Any:
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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_list()
    return value


@dataclass
class FeatureMetadata:
    feature_names: Optional[List[str]] = None
    feature_bounds_min: Optional[np.ndarray] = None
    feature_bounds_max: Optional[np.ndarray] = None
    num_numeric_features: Optional[int] = None
    num_categorical_features: int = 0
    cat_cardinalities: List[int] = field(default_factory=list)

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "feature_bounds_min": _to_jsonable(self.feature_bounds_min),
            "feature_bounds_max": _to_jsonable(self.feature_bounds_max),
            "num_numeric_features": self.num_numeric_features,
            "num_categorical_features": self.num_categorical_features,
            "cat_cardinalities": list(self.cat_cardinalities),
        }


@dataclass
class DetectorContext:
    model: Any
    model_name: str
    model_family: str
    num_classes: int
    detection_split: Any
    seed: int
    device: Any = "cpu"
    clean_support_split: Optional[Any] = None
    poisoned_indices: Optional[np.ndarray] = None
    attack_target_label: Optional[int] = None
    attack_source_labels: Optional[List[int]] = None
    class_names: Optional[List[str]] = None
    feature_metadata: Optional[FeatureMetadata] = None
    model_metadata: Optional[Dict[str, Any]] = None
    attack_metadata: Optional[Dict[str, Any]] = None
    detector_cfg: Optional[Dict[str, Any]] = None
    run_dir: Optional[str] = None
    sample_indices: Optional[np.ndarray] = None
    true_is_infected: Optional[bool] = None
    true_target_class: Optional[int] = None
    true_source_class: Optional[int] = None
    evaluation_split: Optional[Any] = None

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "num_classes": int(self.num_classes),
            "seed": int(self.seed),
            "device": str(self.device),
            "poisoned_indices": _to_jsonable(self.poisoned_indices),
            "attack_target_label": self.attack_target_label,
            "attack_source_labels": _to_jsonable(self.attack_source_labels),
            "class_names": self.class_names,
            "feature_metadata": None if self.feature_metadata is None else self.feature_metadata.to_jsonable(),
            "model_metadata": _to_jsonable(self.model_metadata),
            "attack_metadata": _to_jsonable(self.attack_metadata),
            "detector_cfg": _to_jsonable(self.detector_cfg),
            "run_dir": self.run_dir,
            "sample_indices": _to_jsonable(self.sample_indices),
            "true_is_infected": self.true_is_infected,
            "true_target_class": self.true_target_class,
            "true_source_class": self.true_source_class,
        }


@dataclass
class ArtifactIndex:
    summary_json: Optional[str] = None
    raw_scores_csv: Optional[str] = None
    class_scores_csv: Optional[str] = None
    suspect_indices_npy: Optional[str] = None
    optimization_trace_json: Optional[str] = None
    estimated_pattern_npy: Optional[str] = None
    plots: List[str] = field(default_factory=list)
    extra_files: Dict[str, str] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "summary_json": self.summary_json,
            "raw_scores_csv": self.raw_scores_csv,
            "class_scores_csv": self.class_scores_csv,
            "suspect_indices_npy": self.suspect_indices_npy,
            "optimization_trace_json": self.optimization_trace_json,
            "estimated_pattern_npy": self.estimated_pattern_npy,
            "plots": list(self.plots),
            "extra_files": dict(self.extra_files),
        }


@dataclass
class DetectorResult:
    detector_name: str
    track_type: str
    status: str
    seed: int
    runtime_sec: float
    summary_metrics: Dict[str, Any] = field(default_factory=dict)
    raw_sample_scores: Optional[np.ndarray] = None
    sample_scores: Optional[np.ndarray] = None
    sample_ranking: Optional[np.ndarray] = None
    sample_flags: Optional[np.ndarray] = None
    sample_labels: Optional[np.ndarray] = None
    suspect_indices: Optional[np.ndarray] = None
    class_scores: Optional[np.ndarray] = None
    class_details: Optional[pd.DataFrame] = None
    pair_scores: Optional[pd.DataFrame] = None
    predicted_is_infected: Optional[bool] = None
    predicted_target_class: Optional[int] = None
    predicted_source_class: Optional[int] = None
    thresholds: Dict[str, Any] = field(default_factory=dict)
    artifacts: ArtifactIndex = field(default_factory=ArtifactIndex)
    deviation_note: Optional[str] = None
    optimization_trace: Optional[Dict[str, Any]] = None
    feature_layer_name: Optional[str] = None
    estimated_trigger: Optional[np.ndarray] = None
    estimated_mask: Optional[np.ndarray] = None
    estimated_perturbation: Optional[np.ndarray] = None
    optimized_inputs: Optional[np.ndarray] = None
    candidate_objective_vectors: Optional[np.ndarray] = None
    candidate_margin_vectors: Optional[np.ndarray] = None

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "track_type": self.track_type,
            "status": self.status,
            "seed": int(self.seed),
            "runtime_sec": float(self.runtime_sec),
            "summary_metrics": _to_jsonable(self.summary_metrics),
            "suspect_indices_count": 0 if self.suspect_indices is None else int(len(self.suspect_indices)),
            "predicted_is_infected": self.predicted_is_infected,
            "predicted_target_class": self.predicted_target_class,
            "predicted_source_class": self.predicted_source_class,
            "thresholds": _to_jsonable(self.thresholds),
            "artifacts": self.artifacts.to_jsonable(),
            "deviation_note": self.deviation_note,
            "feature_layer_name": self.feature_layer_name,
        }

    def save_summary(self, output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_summary_dict(), handle, indent=2, sort_keys=True)
        self.artifacts.summary_json = str(path)
        return str(path)

    @classmethod
    def load_summary(cls, summary_path: str) -> Dict[str, Any]:
        with Path(summary_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @classmethod
    def load_suspect_indices(cls, summary_path: str) -> np.ndarray:
        summary = cls.load_summary(summary_path)
        artifact_path = summary.get("artifacts", {}).get("suspect_indices_npy")
        if not artifact_path:
            return np.empty((0,), dtype=np.int64)

        path = Path(artifact_path)
        if not path.is_absolute():
            summary_dir = Path(summary_path).resolve().parent
            path = (summary_dir / path).resolve() if not path.exists() else path
        return np.load(path).astype(np.int64, copy=False)
