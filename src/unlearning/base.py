from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..utils.logging import get_logger
from .types import ForgetSet, UnlearningContext, UnlearningResult
from .utils import (
    complement_indices,
    compute_delta_metrics,
    compute_forget_metrics,
    count_split_samples,
    evaluate_model,
    get_obj_field,
    measure_runtime,
    merge_metric_dicts,
    normalize_forget_set,
    normalize_cfg,
    phase_metrics,
    resolve_target_label,
    save_unlearning_artifacts,
    scalar_metrics_only,
    set_seed,
    stable_unique_indices,
    subset_split,
)


LOGGER = get_logger("unlearning.base")


class BaseUnlearner(ABC):
    def __init__(self, cfg: Any) -> None:
        self.resolved_cfg = normalize_cfg(cfg)
        self.name = str(self.resolved_cfg.get("name", self.__class__.__name__.lower()))
        self.track_type = str(self.resolved_cfg.get("track_type", self.name))
        self.seed = int(self.resolved_cfg.get("seed", 42))
        self.auto_save_result = bool(self.resolved_cfg.get("auto_save_result", True))
        self.artifact_root = str(self.resolved_cfg.get("artifact_root", "artifacts/unlearning"))
        self.evaluation_batch_size = int(self.resolved_cfg.get("evaluation_batch_size", 512))
        self.allow_empty_forget_set = bool(self.resolved_cfg.get("allow_empty_forget_set", False))
        self.min_retained_samples = int(self.resolved_cfg.get("min_retained_samples", 1))
        forget_cfg = self.resolved_cfg.get("forget_set", {})
        self.forget_set_source = str(forget_cfg.get("source", self.resolved_cfg.get("forget_set_source", "none")))
        self.forget_set_topk = int(forget_cfg.get("topk", self.resolved_cfg.get("forget_set_topk", 0)))
        self.forget_set_index_space = str(forget_cfg.get("index_space", self.resolved_cfg.get("forget_set_index_space", "train_local")))
        self.forget_set_score_threshold = forget_cfg.get(
            "score_threshold",
            self.resolved_cfg.get("forget_set_score_threshold", None),
        )
        self.forget_set_score_quantile = forget_cfg.get(
            "score_quantile",
            self.resolved_cfg.get("forget_set_score_quantile", None),
        )
        self.forget_set_remove_fraction = float(
            forget_cfg.get("remove_fraction", self.resolved_cfg.get("forget_set_remove_fraction", 0.0)) or 0.0
        )
        self.forget_set_max_remove = int(
            forget_cfg.get("max_remove", self.resolved_cfg.get("forget_set_max_remove", 0)) or 0
        )
        self.forget_set_larger_scores_are_more_suspicious = bool(
            forget_cfg.get(
                "larger_scores_are_more_suspicious",
                self.resolved_cfg.get("larger_scores_are_more_suspicious", True),
            )
        )
        eval_cfg = self.resolved_cfg.get("evaluation", {})
        self.evaluate_train_split = bool(eval_cfg.get("train_split", self.resolved_cfg.get("evaluate_train_split", False)))
        self.evaluate_forget_split = bool(eval_cfg.get("forget_split", self.resolved_cfg.get("evaluate_forget_split", True)))
        self.evaluate_retain_split = bool(eval_cfg.get("retain_split", self.resolved_cfg.get("evaluate_retain_split", False)))
        self.evaluate_per_class = bool(eval_cfg.get("per_class", self.resolved_cfg.get("evaluate_per_class", True)))

    def run(self, *args: Any, **kwargs: Any) -> UnlearningResult:
        if len(args) == 1 and isinstance(args[0], UnlearningContext):
            context = args[0]
        else:
            context = self._build_context_from_kwargs(**kwargs)

        self._validate_context(context)
        context.run_dir = self._build_output_dir(context)
        set_seed(context.seed)
        LOGGER.info("Unlearning start: name=%s track=%s", self.name, self.track_type)

        num_train = count_split_samples(context.datasets["train"])
        target_label = resolve_target_label(context)
        candidate_forget_set = normalize_forget_set(
            self._resolve_forget_set(context),
            num_train=num_train,
            train_sample_indices=context.train_sample_indices,
        )
        forget_set = normalize_forget_set(
            self._select_forget_set(context, candidate_forget_set),
            num_train=num_train,
            train_sample_indices=context.train_sample_indices,
        )
        self._validate_forget_set(context, forget_set)
        eval_datasets = self._build_evaluation_datasets(context, forget_set)
        metrics_before, eval_before_sec = measure_runtime(
            self._evaluate_model,
            context.model,
            eval_datasets,
            target_label,
            context.device,
        )

        result, repair_runtime_sec = measure_runtime(self._run_impl, context, forget_set)
        result.method_name = self.name
        result.track_type = str(result.track_type or self.track_type)
        result.seed = int(context.seed)
        result.runtime_sec = float(repair_runtime_sec)

        removed_indices = forget_set.indices
        retain_indices = complement_indices(num_train, removed_indices)

        result.forget_set_source = result.forget_set_source or forget_set.source
        result.removed_indices = removed_indices if result.removed_indices is None else stable_unique_indices(result.removed_indices, upper_bound=num_train)
        result.retain_indices = retain_indices if result.retain_indices is None else stable_unique_indices(result.retain_indices, upper_bound=num_train)
        result.num_removed = int(len(result.removed_indices))
        result.num_retained = int(len(result.retain_indices))
        result.metrics_before = merge_metric_dicts(metrics_before, result.metrics_before)

        eval_after_sec = 0.0
        if result.model_after is not None:
            metrics_after_eval, eval_after_sec = measure_runtime(
                self._evaluate_model,
                result.model_after,
                eval_datasets,
                target_label,
                context.device,
            )
            result.metrics_after = merge_metric_dicts(metrics_after_eval, result.metrics_after)
        elif not result.metrics_after and result.status == "skipped":
            result.metrics_after = dict(result.metrics_before)
        eval_runtime_sec = float(eval_before_sec) + float(eval_after_sec)

        result.summary_metrics = merge_metric_dicts(
            result.summary_metrics,
            phase_metrics(scalar_metrics_only(result.metrics_before), "before"),
            phase_metrics(scalar_metrics_only(result.metrics_after), "after"),
            compute_forget_metrics(
                removed_indices=result.removed_indices,
                poisoned_indices=get_obj_field(context.attack_result, "poison_indices", None),
                num_candidates=num_train,
            ),
            compute_delta_metrics(result.metrics_before, result.metrics_after),
            {
                "unlearning/runtime_sec": float(repair_runtime_sec),
                "unlearning/repair_runtime_sec": float(repair_runtime_sec),
                "unlearning/eval_runtime_sec": float(eval_runtime_sec),
                "unlearning/total_runtime_sec": float(repair_runtime_sec + eval_runtime_sec),
            },
        )

        if self.auto_save_result:
            result.artifacts = save_unlearning_artifacts(
                output_dir=context.run_dir,
                result=result,
                context=context,
                forget_set=forget_set,
                resolved_cfg=self.resolved_cfg,
            )

        LOGGER.info("Unlearning done: name=%s status=%s runtime=%.4fs", self.name, result.status, result.runtime_sec)
        return result

    def _build_context_from_kwargs(self, **kwargs: Any) -> UnlearningContext:
        model = kwargs.get("model")
        datasets = kwargs.get("datasets")
        attack_result = kwargs.get("attack_result")
        detection_result = kwargs.get("detection_result")
        model_cfg = kwargs.get("model_cfg")
        train_cfg = kwargs.get("train_cfg")
        seed = int(kwargs.get("seed", self.seed))
        device = kwargs.get("device", get_obj_field(train_cfg, "device", "cpu"))
        num_classes = kwargs.get("num_classes")
        class_names = kwargs.get("class_names")
        run_dir = kwargs.get("run_dir")
        target_label = kwargs.get("target_label")
        train_sample_indices = kwargs.get("train_sample_indices")
        detection_sample_indices = kwargs.get("detection_sample_indices")
        feature_metadata = kwargs.get("feature_metadata")
        attack_metadata = kwargs.get("attack_metadata")
        clean_support_split = kwargs.get("clean_support_split")

        model_name = kwargs.get("model_name")
        if model_name is None and model is not None:
            model_name = model.__class__.__name__

        model_family = kwargs.get("model_family")
        if model_family is None and model is not None:
            model_family = getattr(model, "model_family", None)

        if num_classes is None and model is not None:
            num_classes = int(getattr(model, "d_out", 0) or 0) or None

        return UnlearningContext(
            model=model,
            datasets=datasets,
            attack_result=attack_result,
            detection_result=detection_result,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            seed=seed,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            run_dir=run_dir,
            method_cfg=self.resolved_cfg,
            model_name=model_name,
            model_family=model_family,
            target_label=target_label,
            train_sample_indices=train_sample_indices,
            detection_sample_indices=detection_sample_indices,
            feature_metadata=feature_metadata,
            attack_metadata=attack_metadata,
            clean_support_split=clean_support_split,
        )

    def _validate_context(self, context: UnlearningContext) -> None:
        assert context.model is not None, "UnlearningContext.model must not be None."
        assert context.datasets is not None, "UnlearningContext.datasets must not be None."
        for split_name in ("train", "val", "test"):
            assert split_name in context.datasets, f"datasets must contain '{split_name}' for unlearning."

    def _validate_forget_set(self, context: UnlearningContext, forget_set: ForgetSet) -> None:
        num_train = count_split_samples(context.datasets["train"])
        forget_indices = stable_unique_indices(forget_set.indices, upper_bound=num_train)
        retained = complement_indices(num_train, forget_indices)

        assert forget_set.index_space == "train_local", "ForgetSet must be normalized to train_local before validation."
        assert (
            forget_set.source == "none"
            or forget_indices.size > 0
            or self.allow_empty_forget_set
        ), f"Forget set source '{forget_set.source}' produced no samples."
        assert retained.size > 0, "Forget set removes every training sample."
        assert retained.size >= int(self.min_retained_samples), (
            f"Retained split is too small: {retained.size} < {self.min_retained_samples}."
        )

    def _build_output_dir(self, context: UnlearningContext) -> str:
        if context.run_dir:
            out = Path(context.run_dir) / "unlearning" / self.name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out = Path(self.artifact_root) / f"{self.name}_{timestamp}"
        out.mkdir(parents=True, exist_ok=True)
        return str(out)

    def _resolve_forget_set(self, context: UnlearningContext) -> ForgetSet:
        source = self.forget_set_source
        num_train = count_split_samples(context.datasets["train"])
        if source == "none":
            return ForgetSet(source="none", index_space="train_local", notes="No forgetting requested.")

        if source == "oracle_poison":
            poison_indices = get_obj_field(context.attack_result, "poison_indices", None)
            assert poison_indices is not None, "oracle_poison requested but attack_result.poison_indices is missing."
            return self._build_forget_set(
                indices=poison_indices,
                source="oracle_poison",
                index_space=self.forget_set_index_space,
                num_candidates=num_train,
            )

        detection_result = context.detection_result
        if source == "detector_flags":
            assert detection_result is not None, "detector_flags requested but detection_result is missing."
            suspect_indices = get_obj_field(detection_result, "suspect_indices", None)
            if suspect_indices is not None and len(np.asarray(suspect_indices).reshape(-1)) > 0:
                scores = get_obj_field(detection_result, "sample_scores", None)
                if scores is not None:
                    suspect_indices_np = np.asarray(suspect_indices, dtype=np.int64).reshape(-1)
                    score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
                    if (
                        suspect_indices_np.size > 0
                        and int(suspect_indices_np.min()) >= 0
                        and int(suspect_indices_np.max()) < int(score_values.shape[0])
                    ):
                        scores = score_values[suspect_indices_np]
                    else:
                        scores = None
                return self._build_forget_set(
                    indices=suspect_indices,
                    scores=scores,
                    source="detector_flags",
                    index_space=self.forget_set_index_space,
                    num_candidates=num_train,
                )

            sample_flags = get_obj_field(detection_result, "sample_flags", None)
            assert sample_flags is not None, "detector_flags requested but detector has neither suspect_indices nor sample_flags."
            flags = np.asarray(sample_flags, dtype=np.int64).reshape(-1)
            indices = np.flatnonzero(flags).astype(np.int64)
            scores = get_obj_field(detection_result, "sample_scores", None)
            if scores is not None:
                scores = np.asarray(scores, dtype=np.float32)[indices]
            return self._build_forget_set(
                indices=indices,
                scores=scores,
                flags=np.ones(indices.shape[0], dtype=np.int64),
                source="detector_flags",
                index_space=self.forget_set_index_space,
                num_candidates=num_train,
            )

        if source == "detector_topk":
            assert detection_result is not None, "detector_topk requested but detection_result is missing."
            ranking = get_obj_field(detection_result, "sample_ranking", None)
            assert ranking is not None, "detector_topk requested but detection_result.sample_ranking is missing."
            assert self.forget_set_topk > 0, "detector_topk requested but forget_set.topk <= 0."
            ranking = np.asarray(ranking, dtype=np.int64).reshape(-1)
            scores = get_obj_field(detection_result, "sample_scores", None)
            if scores is not None:
                score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
                if ranking.size > 0 and int(ranking.min()) >= 0 and int(ranking.max()) < int(score_values.shape[0]):
                    scores = score_values[ranking]
                else:
                    scores = None
            return self._build_forget_set(
                indices=ranking,
                scores=scores,
                source="detector_topk",
                index_space=self.forget_set_index_space,
                num_candidates=num_train,
            )

        if source == "manual":
            manual_indices = self._load_manual_indices()
            return self._build_forget_set(
                indices=manual_indices,
                source="manual",
                index_space=self.forget_set_index_space,
                num_candidates=num_train,
            )

        raise AssertionError(f"Unknown forget_set.source: {source}")

    def _select_forget_set(self, context: UnlearningContext, candidate_set: ForgetSet) -> ForgetSet:
        """
        Select the final forget set from all candidates exposed by detection.

        Base behavior applies only unlearner-level budget/score knobs. Subclasses
        can override this hook for method-specific policies while still receiving
        the full normalized candidate set from `_resolve_forget_set`.
        """
        return self._apply_configured_candidate_selection(candidate_set, num_candidates=count_split_samples(context.datasets["train"]))

    def _evaluate_model(self, model: Any, datasets: Dict[str, Any], target_label: Any, device: Any) -> Dict[str, Any]:
        return evaluate_model(
            model,
            datasets=datasets,
            target_label=target_label,
            device=device,
            batch_size=self.evaluation_batch_size,
            per_class=self.evaluate_per_class,
        )

    def _build_evaluation_datasets(self, context: UnlearningContext, forget_set: ForgetSet) -> Dict[str, Any]:
        datasets: Dict[str, Any] = {}
        for split_name in ("val", "test", "test_triggered", "test_clean_labels"):
            if split_name in context.datasets:
                datasets[split_name] = context.datasets[split_name]

        train_split = context.datasets["train"]
        num_train = count_split_samples(train_split)
        retain_indices = complement_indices(num_train, forget_set.indices)

        if self.evaluate_train_split:
            datasets["train"] = train_split
        if self.evaluate_forget_split and len(forget_set.indices) > 0:
            datasets["forget"] = subset_split(train_split, forget_set.indices)
        if self.evaluate_retain_split and len(retain_indices) > 0:
            datasets["retain"] = subset_split(train_split, retain_indices)
        return datasets

    def _load_manual_indices(self) -> Any:
        forget_cfg = self.resolved_cfg.get("forget_set", {})
        if "indices" in forget_cfg:
            return forget_cfg.get("indices", [])
        manual_path = forget_cfg.get("indices_path", self.resolved_cfg.get("manual_indices_path", None))
        if manual_path:
            path = Path(str(manual_path))
            if path.suffix == ".npy":
                return np.load(path)
            return np.loadtxt(path, delimiter=",", dtype=np.int64)
        return self.resolved_cfg.get("manual_indices", [])

    def _build_forget_set(
        self,
        *,
        indices: Any,
        source: str,
        num_candidates: int,
        scores: Any = None,
        flags: Any = None,
        index_space: str | None = None,
        notes: str | None = None,
    ) -> ForgetSet:
        indices_np = np.asarray(indices, dtype=np.int64).reshape(-1)
        scores_np = None if scores is None else np.asarray(scores, dtype=np.float32).reshape(-1)
        flags_np = None if flags is None else np.asarray(flags, dtype=np.int64).reshape(-1)

        if scores_np is not None and scores_np.shape[0] != indices_np.shape[0]:
            scores_np = None
        if flags_np is not None and flags_np.shape[0] != indices_np.shape[0]:
            flags_np = None

        metadata = {
            "num_candidate_indices": int(indices_np.shape[0]),
        }
        return ForgetSet(
            indices=indices_np,
            scores=scores_np,
            flags=flags_np,
            source=source,
            index_space=str(index_space or self.forget_set_index_space),
            notes=notes,
            metadata=metadata,
        )

    def _apply_configured_candidate_selection(self, candidate_set: ForgetSet, *, num_candidates: int) -> ForgetSet:
        indices_np = np.asarray(candidate_set.indices, dtype=np.int64).reshape(-1)
        scores_np = None if candidate_set.scores is None else np.asarray(candidate_set.scores, dtype=np.float32).reshape(-1)
        flags_np = None if candidate_set.flags is None else np.asarray(candidate_set.flags, dtype=np.int64).reshape(-1)

        if scores_np is not None and scores_np.shape[0] != indices_np.shape[0]:
            scores_np = None
        if flags_np is not None and flags_np.shape[0] != indices_np.shape[0]:
            flags_np = None

        selected = np.arange(indices_np.shape[0], dtype=np.int64)
        if scores_np is not None:
            if self.forget_set_score_threshold is not None:
                threshold = float(self.forget_set_score_threshold)
                if self.forget_set_larger_scores_are_more_suspicious:
                    selected = selected[scores_np[selected] >= threshold]
                else:
                    selected = selected[scores_np[selected] <= threshold]
            if self.forget_set_score_quantile is not None and selected.size > 0:
                quantile = float(self.forget_set_score_quantile)
                threshold = float(np.quantile(scores_np[selected], quantile))
                if self.forget_set_larger_scores_are_more_suspicious:
                    selected = selected[scores_np[selected] >= threshold]
                else:
                    selected = selected[scores_np[selected] <= threshold]

        cap = None
        if self.forget_set_topk > 0:
            cap = self.forget_set_topk
        if self.forget_set_remove_fraction > 0.0:
            fraction_cap = int(np.ceil(float(num_candidates) * self.forget_set_remove_fraction))
            cap = fraction_cap if cap is None else min(cap, fraction_cap)
        if self.forget_set_max_remove > 0:
            cap = self.forget_set_max_remove if cap is None else min(cap, self.forget_set_max_remove)

        if cap is not None and selected.size > int(cap):
            if scores_np is not None:
                order = np.argsort(scores_np[selected])
                if self.forget_set_larger_scores_are_more_suspicious:
                    order = order[::-1]
                selected = selected[order[: int(cap)]]
            else:
                selected = selected[: int(cap)]

        metadata = dict(candidate_set.metadata)
        metadata.update(
            {
                "selection_policy": "configured_unlearner_selection",
                "num_candidate_indices": int(indices_np.shape[0]),
                "num_selected_indices": int(selected.shape[0]),
                "selection_topk": int(self.forget_set_topk),
                "selection_remove_fraction": float(self.forget_set_remove_fraction),
                "selection_max_remove": int(self.forget_set_max_remove),
                "selection_score_threshold": self.forget_set_score_threshold,
                "selection_score_quantile": self.forget_set_score_quantile,
            }
        )
        return ForgetSet(
            indices=indices_np[selected],
            scores=None if scores_np is None else scores_np[selected],
            flags=None if flags_np is None else flags_np[selected],
            source=candidate_set.source,
            index_space=candidate_set.index_space,
            notes=candidate_set.notes,
            metadata=metadata,
        )

    def _selection_metadata(self) -> Dict[str, Any]:
        return {
            "selection_topk": int(self.forget_set_topk),
            "selection_remove_fraction": float(self.forget_set_remove_fraction),
            "selection_max_remove": int(self.forget_set_max_remove),
            "selection_score_threshold": self.forget_set_score_threshold,
            "selection_score_quantile": self.forget_set_score_quantile,
        }

    @abstractmethod
    def _run_impl(self, context: UnlearningContext, forget_set: ForgetSet) -> UnlearningResult:
        raise NotImplementedError


class NoOpUnlearner(BaseUnlearner):
    def _resolve_forget_set(self, context: UnlearningContext) -> ForgetSet:
        return ForgetSet(source="none", notes="No forgetting requested.")

    def _run_impl(self, context: UnlearningContext, forget_set: ForgetSet) -> UnlearningResult:
        return UnlearningResult(
            method_name=self.name,
            track_type="noop",
            status="skipped",
            seed=int(context.seed),
            runtime_sec=0.0,
            forget_set_source=forget_set.source,
            deviation_note="No unlearning step executed.",
            model_after=context.model,
        )
