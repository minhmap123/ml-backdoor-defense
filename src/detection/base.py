from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ..utils.logging import get_logger
from .types import DetectorContext, DetectorResult
from .utils import (
    compute_class_detection_metrics,
    compute_sample_detection_metrics,
    compute_topk_recall,
    count_split_samples,
    merge_metric_dicts,
    normalize_detector_cfg,
    save_detection_artifacts,
    set_seed,
)


LOGGER = get_logger("detection.base")


class BaseDetector(ABC):
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.name = str(getattr(cfg, "name", self.__class__.__name__.lower()))
        self.seed = int(getattr(cfg, "seed", 42))
        self.auto_save_result = bool(getattr(cfg, "auto_save_result", True))
        self.artifact_root = str(getattr(cfg, "detection_artifact_root", "artifacts/detection"))
        self.topk = int(getattr(cfg, "topk", 0))
        self.requires_model = bool(getattr(cfg, "requires_model", True))
        self.requires_detection_split = bool(getattr(cfg, "requires_detection_split", True))
        self.requires_clean_support_split = bool(getattr(cfg, "requires_clean_support_split", False))

    def run(self, context: DetectorContext) -> DetectorResult:
        self._validate_context(context)
        set_seed(context.seed)
        LOGGER.info("Detection start: name=%s model=%s", self.name, context.model_name)

        result = self._run_impl(context)
        result.detector_name = self.name
        result.seed = int(context.seed)

        sample_metrics = self._compute_sample_metrics(result, context)
        class_metrics = compute_class_detection_metrics(
            predicted_is_infected=result.predicted_is_infected,
            true_is_infected=context.true_is_infected,
            predicted_target_class=result.predicted_target_class,
            true_target_class=context.true_target_class,
            predicted_source_class=result.predicted_source_class,
            true_source_class=context.true_source_class,
            runtime_sec=result.runtime_sec,
        )
        result.summary_metrics = merge_metric_dicts(result.summary_metrics, sample_metrics, class_metrics)

        if self.auto_save_result:
            output_dir = self._build_output_dir(context)
            result.artifacts = save_detection_artifacts(
                output_dir=output_dir,
                result=result,
                context=context,
                resolved_cfg=normalize_detector_cfg(self.cfg),
            )

        LOGGER.info("Detection done: name=%s status=%s runtime=%.4fs", self.name, result.status, result.runtime_sec)
        return result

    def _compute_sample_metrics(self, result: DetectorResult, context: DetectorContext) -> Dict[str, Any]:
        num_candidates = 0
        if result.sample_scores is not None:
            num_candidates = int(len(result.sample_scores))
        else:
            try:
                num_candidates = count_split_samples(context.detection_split)
            except Exception:
                num_candidates = 0

        metrics = compute_sample_detection_metrics(
            sample_scores=result.sample_scores,
            sample_flags=result.sample_flags,
            poisoned_indices=context.poisoned_indices,
            num_candidates=num_candidates,
        )
        if (
            self.topk > 0
            and result.sample_ranking is not None
            and context.poisoned_indices is not None
        ):
            metrics[f"detection/topk_recall@{self.topk}"] = compute_topk_recall(
                result.sample_ranking,
                context.poisoned_indices,
                self.topk,
            )
        return metrics

    def _validate_context(self, context: DetectorContext) -> None:
        if self.requires_model and context.model is None:
            raise ValueError("DetectorContext.model must not be None.")
        if self.requires_detection_split and context.detection_split is None:
            raise ValueError("DetectorContext.detection_split must not be None.")
        if self.requires_clean_support_split and context.clean_support_split is None:
            raise ValueError("DetectorContext.clean_support_split must not be None.")
        if int(context.num_classes) <= 0:
            raise ValueError(f"num_classes must be > 0, got {context.num_classes}")
        if context.sample_indices is not None and context.detection_split is not None:
            num_candidates = count_split_samples(context.detection_split)
            if int(len(context.sample_indices)) != int(num_candidates):
                raise ValueError(
                    "DetectorContext.sample_indices must have the same length as detection_split. "
                    f"Got {len(context.sample_indices)} vs {num_candidates}."
                )

    def _build_output_dir(self, context: DetectorContext) -> str:
        if context.run_dir:
            out = Path(context.run_dir) / "detection" / self.name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out = Path(self.artifact_root) / f"{self.name}_{timestamp}"
        out.mkdir(parents=True, exist_ok=True)
        return str(out)

    @abstractmethod
    def _run_impl(self, context: DetectorContext) -> DetectorResult:
        raise NotImplementedError
