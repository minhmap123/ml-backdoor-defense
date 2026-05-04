from __future__ import annotations

from math import ceil
from typing import Any, Dict, List

import numpy as np

from .base import BaseDetector
from .types import DetectorContext, DetectorResult
from .utils import extract_model_features, measure_runtime, rank_desc


class SpectralSignaturesDetector(BaseDetector):
    """
    Spectral Signatures detector that follows the original paper's per-class removal flow.

    Research references:
    - Paper: https://papers.nips.cc/paper_files/paper/2018/file/280cf18baf4311c92aa5a042336587d3-Paper.pdf
    - Official repo audited for deviations: https://github.com/MadryLab/backdoor_data_poisoning

    Local research assumptions:
    - The detector does not know the attacked target class, so it always scans
      all observed classes in the detection split.
    - Feature representations are taken from `model.forward_features()`, so
      fidelity depends on each local model wrapper exposing the intended
      representation layer.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.feature_batch_size = int(getattr(cfg, "feature_batch_size", 512))
        self.poison_rate_upper_bound = getattr(cfg, "poison_rate_upper_bound", None)
        self.removed_multiplier = float(getattr(cfg, "removed_multiplier", 1.5))
        self.num_top_singular_vectors = int(getattr(cfg, "num_top_singular_vectors", 1))
        self.score_on_centered_features = bool(getattr(cfg, "score_on_centered_features", True))
        self.min_samples_per_class = int(getattr(cfg, "min_samples_per_class", 2))
        self.save_clean_reference_stats = bool(getattr(cfg, "save_clean_reference_stats", True))

    def _validate_context(self, context: DetectorContext) -> None:
        super()._validate_context(context)
        # Paper-mode per-class removal requires a poison-rate estimate.
        _ = self._resolve_poison_rate_upper_bound(context)

    def _run_impl(self, context: DetectorContext) -> DetectorResult:
        (features, labels), runtime_sec = measure_runtime(
            extract_model_features,
            context.model,
            context.detection_split,
            device=context.device,
            batch_size=self.feature_batch_size,
        )

        num_samples = int(features.shape[0])
        active_labels = np.unique(np.asarray(labels, dtype=np.int64))
        raw_scores = np.zeros(num_samples, dtype=np.float64)
        decision_scores = np.zeros(num_samples, dtype=np.float64)
        global_flags = np.zeros(num_samples, dtype=np.int64)
        class_scores = np.zeros(int(context.num_classes), dtype=np.float32)
        per_class_records: List[Dict[str, Any]] = []

        for label in active_labels:
            class_indices = np.flatnonzero(labels == label).astype(np.int64)
            class_stat: Dict[str, Any] = {
                "label": int(label),
                "num_samples": int(class_indices.size),
                "status": "ok",
            }

            if class_indices.size < self.min_samples_per_class:
                class_stat["status"] = "skipped_too_few_samples"
                per_class_records.append(
                    {
                        "label": int(label),
                        "class_indices": class_indices,
                        "sample_scores": np.empty((0,), dtype=np.float64),
                        "class_stat": class_stat,
                    }
                )
                continue

            class_features = np.asarray(features[class_indices], dtype=np.float64)
            class_mean = np.mean(class_features, axis=0, keepdims=True)
            centered = class_features - class_mean

            try:
                _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
            except np.linalg.LinAlgError:
                class_stat["status"] = "svd_failed"
                per_class_records.append(
                    {
                        "label": int(label),
                        "class_indices": class_indices,
                        "sample_scores": np.empty((0,), dtype=np.float64),
                        "class_stat": class_stat,
                    }
                )
                continue

            top_vectors = right_vectors[: self.num_top_singular_vectors]
            score_source = centered if self.score_on_centered_features else class_features
            correlations = np.matmul(top_vectors, np.transpose(score_source))
            sample_scores = np.sum(np.square(correlations), axis=0)

            raw_scores[class_indices] = sample_scores
            class_scores[int(label)] = float(singular_values[0]) if singular_values.size > 0 else 0.0
            class_stat["top_singular_values"] = singular_values[:7].astype(np.float32).tolist()

            if self.save_clean_reference_stats and context.poisoned_indices is not None:
                poisoned_set = set(np.asarray(context.poisoned_indices, dtype=np.int64).tolist())
                clean_mask = np.asarray([idx not in poisoned_set for idx in class_indices], dtype=bool)
                if int(clean_mask.sum()) >= self.min_samples_per_class:
                    clean_features = class_features[clean_mask]
                    clean_centered = clean_features - np.mean(clean_features, axis=0, keepdims=True)
                    clean_singular_values = np.linalg.svd(
                        clean_centered,
                        full_matrices=False,
                        compute_uv=False,
                    )
                    class_stat["clean_top_singular_values"] = clean_singular_values[:7].astype(np.float32).tolist()

            per_class_records.append(
                {
                    "label": int(label),
                    "class_indices": class_indices,
                    "sample_scores": sample_scores,
                    "class_stat": class_stat,
                }
            )

        decision_scores = raw_scores.copy()
        for record in per_class_records:
            class_indices = record["class_indices"]
            sample_scores = record["sample_scores"]
            class_stat = record["class_stat"]
            if sample_scores.size == 0:
                continue
            flagged_local, threshold, num_to_remove = self._select_per_class_flags(
                sample_scores=sample_scores,
                class_size=int(class_indices.size),
                context=context,
            )
            flagged_indices = class_indices[flagged_local]
            global_flags[flagged_indices] = 1
            class_stat.update(
                {
                    "threshold": threshold,
                    "num_to_remove": int(num_to_remove),
                    "num_flagged": int(flagged_local.size),
                    "flagged_indices_local": flagged_local.tolist(),
                    "flagged_indices_global": flagged_indices.tolist(),
                }
            )

        suspect_indices = np.flatnonzero(global_flags).astype(np.int64)
        ranking = rank_desc(decision_scores.astype(np.float32))

        return DetectorResult(
            detector_name=self.name,
            track_type="sample",
            status="ok",
            seed=int(context.seed),
            runtime_sec=float(runtime_sec),
            raw_sample_scores=raw_scores.astype(np.float32),
            sample_scores=decision_scores.astype(np.float32),
            sample_ranking=ranking,
            sample_flags=global_flags,
            sample_labels=labels,
            suspect_indices=suspect_indices,
            class_scores=class_scores,
            predicted_is_infected=None,
            predicted_target_class=None,
            thresholds=self._build_thresholds(context, num_samples),
            deviation_note=self._build_deviation_note(),
            optimization_trace={
                "num_top_singular_vectors": self.num_top_singular_vectors,
                "score_on_centered_features": self.score_on_centered_features,
                "per_class_stats": [record["class_stat"] for record in per_class_records],
            },
            feature_layer_name="forward_features",
        )

    def _resolve_poison_rate_upper_bound(self, context: DetectorContext) -> float:
        epsilon = self.poison_rate_upper_bound
        if epsilon is None and context.attack_metadata is not None:
            epsilon = context.attack_metadata.get("poison_rate")
        assert epsilon is not None, (
            "paper-mode requires poison_rate_upper_bound or attack_metadata['poison_rate']"
        )
        epsilon = float(epsilon)
        assert epsilon >= 0.0, f"poison_rate_upper_bound must be >= 0, got {epsilon}"
        return epsilon

    def _select_per_class_flags(
        self,
        *,
        sample_scores: np.ndarray,
        class_size: int,
        context: DetectorContext,
    ) -> tuple[np.ndarray, float, int]:
        scores = np.asarray(sample_scores, dtype=np.float64)
        epsilon = self._resolve_poison_rate_upper_bound(context)
        num_to_remove = int(ceil(self.removed_multiplier * epsilon * float(class_size)))
        num_to_remove = max(0, min(num_to_remove, int(class_size)))
        if num_to_remove == 0:
            return np.empty((0,), dtype=np.int64), float("inf"), 0
        order = np.argsort(scores)[::-1].astype(np.int64)
        flagged_local = order[:num_to_remove]
        threshold = float(scores[flagged_local[-1]])
        return flagged_local, threshold, num_to_remove

    def _build_thresholds(self, context: DetectorContext, num_samples: int) -> Dict[str, Any]:
        thresholds: Dict[str, Any] = {}
        thresholds["poison_rate_upper_bound"] = self._resolve_poison_rate_upper_bound(context)
        thresholds["removed_multiplier"] = self.removed_multiplier
        return thresholds

    def _build_deviation_note(self) -> str | None:
        notes: List[str] = []
        if not self.score_on_centered_features:
            notes.append("Local extension: score_on_centered_features=False.")
        if self.num_top_singular_vectors != 1:
            notes.append("Local extension: num_top_singular_vectors != 1.")
        return " ".join(notes) if notes else None
