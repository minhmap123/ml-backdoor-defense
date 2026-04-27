from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import gamma, median_abs_deviation

from ..models.utils import split_to_numpy
from .base import BaseDetector
from .types import DetectorContext, DetectorResult
from .utils import measure_runtime, rank_desc, resolve_device


class MMBDDetector(BaseDetector):
    """
    Paper- and official-code-guided MM-BD detector adapted to numeric IDS inputs.

    Research references:
    - Paper: https://arxiv.org/abs/2205.06900
    - Official repo: https://github.com/wanghangpsu/MM-BD

    Local research assumptions:
    - The active repo path is numeric-only IDS, so optimization happens in a
      bounded continuous feature space instead of image space.
    - Bounds come from `feature_metadata` by default. Falling back to the
      observed min/max of `detection_split` is a local escape hatch that must
      be enabled explicitly via `allow_split_bounds_fallback=true`.
    - The core MM-BD statistic is preserved: per class, optimize synthetic
      inputs to maximize class margin, then apply a gamma-tail test to the
      maximum margin statistic vector.

    Suggested deviation note for reporting:
    - We adapt MM-BD from image inputs to numeric tabular IDS inputs by
      optimizing bounded continuous feature vectors instead of images.
      The optimization objective, per-class maximum-margin statistic, and
      gamma-based infected/target decision rule follow the official method.
      The main deviation is the search domain: valid feature bounds are taken
      from dataset preprocessing metadata rather than the image range [0, 1].
      This preserves the detector core while making it compatible with
      all-numeric IDS datasets.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.num_candidates = int(getattr(cfg, "num_candidates", 30))
        self.num_steps = int(getattr(cfg, "num_steps", 300))
        self.lr = float(getattr(cfg, "lr", 1e-2))
        self.sgd_momentum = float(getattr(cfg, "sgd_momentum", 0.2))
        self.early_stop_rel_tol = float(getattr(cfg, "early_stop_rel_tol", 1e-5))
        self.significance_level = float(getattr(cfg, "significance_level", 0.05))
        self.allow_split_bounds_fallback = bool(getattr(cfg, "allow_split_bounds_fallback", False))
        self.requires_clean_support_split = False

    def _validate_context(self, context: DetectorContext) -> None:
        super()._validate_context(context)
        self._assert_numeric_only_context(context)

    def _run_impl(self, context: DetectorContext) -> DetectorResult:
        result, runtime_sec = measure_runtime(self._optimize_all_classes, context)
        result.runtime_sec = float(runtime_sec)
        return result

    def _optimize_all_classes(self, context: DetectorContext) -> DetectorResult:
        device = resolve_device(context.device)
        model = context.model.to(device)
        model.eval()

        reference_x = self._extract_reference_matrix(context)
        lower, upper = self._resolve_bounds(context, reference_x)
        lower_t = torch.as_tensor(lower, dtype=torch.float32, device=device)
        upper_t = torch.as_tensor(upper, dtype=torch.float32, device=device)

        class_scores = np.full(int(context.num_classes), -np.inf, dtype=np.float32)
        optimized_inputs = np.zeros((int(context.num_classes), lower.shape[0]), dtype=np.float32)
        candidate_vectors = np.full((int(context.num_classes), self.num_candidates), np.nan, dtype=np.float32)
        per_class_stats = []

        for target_class in range(int(context.num_classes)):
            best_score, best_input, final_scores, class_stat = self._optimize_one_class(
                model=model,
                target_class=int(target_class),
                num_classes=int(context.num_classes),
                lower_t=lower_t,
                upper_t=upper_t,
                device=device,
            )
            class_scores[target_class] = float(best_score)
            optimized_inputs[target_class] = best_input
            candidate_vectors[target_class, : final_scores.shape[0]] = final_scores
            per_class_stats.append(class_stat)

        anomaly_scores = self._compute_robust_anomaly_scores(class_scores)
        predicted_target_class = int(np.argmax(class_scores))
        p_value, gamma_params, gamma_status = self._compute_gamma_pvalue(class_scores, predicted_target_class)
        predicted_is_infected = bool(p_value <= self.significance_level)

        return DetectorResult(
            detector_name=self.name,
            track_type="class",
            status="ok",
            seed=int(context.seed),
            runtime_sec=0.0,
            summary_metrics={
                "detection/p_value": float(p_value),
                "detection/significance_level": float(self.significance_level),
                f"detection/max_{self._objective_name()}": float(np.max(class_scores)),
                "detection/max_anomaly_score": float(np.max(anomaly_scores)),
            },
            sample_scores=None,
            class_scores=class_scores,
            predicted_is_infected=predicted_is_infected,
            predicted_target_class=predicted_target_class if predicted_is_infected else None,
            thresholds={
                "threshold_source": "gamma_pvalue",
                "significance_level": self.significance_level,
            },
            deviation_note=None,
            optimization_trace={
                "num_candidates": self.num_candidates,
                "num_steps": self.num_steps,
                "lr": self.lr,
                "sgd_momentum": self.sgd_momentum,
                "early_stop_rel_tol": self.early_stop_rel_tol,
                "lower_bounds_source": self._resolve_bounds_source(context),
                "allow_split_bounds_fallback": self.allow_split_bounds_fallback,
                "objective_name": self._objective_name(),
                "gamma_fit_status": gamma_status,
                "gamma_params": gamma_params,
                "class_anomaly_scores": anomaly_scores.astype(np.float32).tolist(),
                "per_class_stats": per_class_stats,
            },
            optimized_inputs=optimized_inputs,
            candidate_objective_vectors=candidate_vectors,
            candidate_margin_vectors=candidate_vectors if self._objective_name() == "margin" else None,
        )

    def _extract_reference_matrix(self, context: DetectorContext) -> np.ndarray:
        self._assert_numeric_only_context(context)
        x, _ = split_to_numpy(context.detection_split)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        d_in = self._resolve_input_dim(context, x)
        return x[:, :d_in]

    def _resolve_input_dim(self, context: DetectorContext, reference_x: np.ndarray) -> int:
        if context.feature_metadata is not None and context.feature_metadata.num_numeric_features is not None:
            return int(context.feature_metadata.num_numeric_features)
        if context.model_metadata is not None:
            if "num_numeric_features" in context.model_metadata:
                return int(context.model_metadata["num_numeric_features"])
            if "d_in" in context.model_metadata:
                return int(context.model_metadata["d_in"])
        if hasattr(context.model, "d_in"):
            return int(context.model.d_in)
        return int(reference_x.shape[1])

    def _resolve_bounds(self, context: DetectorContext, reference_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d_in = int(reference_x.shape[1])
        if context.feature_metadata is not None:
            lower = context.feature_metadata.feature_bounds_min
            upper = context.feature_metadata.feature_bounds_max
            if lower is not None and upper is not None:
                lower = np.asarray(lower, dtype=np.float32)[:d_in]
                upper = np.asarray(upper, dtype=np.float32)[:d_in]
                return lower, upper

        assert self.allow_split_bounds_fallback, (
            "MM-BD requires explicit feature bounds in feature_metadata unless "
            "`allow_split_bounds_fallback=true` is set."
        )
        lower = np.min(reference_x, axis=0).astype(np.float32)
        upper = np.max(reference_x, axis=0).astype(np.float32)
        return lower, upper

    def _resolve_bounds_source(self, context: DetectorContext) -> str:
        if context.feature_metadata is not None:
            if context.feature_metadata.feature_bounds_min is not None and context.feature_metadata.feature_bounds_max is not None:
                return "feature_metadata"
        return "detection_split_minmax_fallback"

    def _assert_numeric_only_context(self, context: DetectorContext) -> None:
        if context.feature_metadata is not None:
            assert int(context.feature_metadata.num_categorical_features) == 0, (
                "MM-BD local IDS path expects numeric-only features."
            )
        if context.model_metadata is not None and "num_categorical_features" in context.model_metadata:
            assert int(context.model_metadata["num_categorical_features"]) == 0, (
                "MM-BD local IDS path expects numeric-only models."
            )
        if hasattr(context.model, "num_categorical_features"):
            assert int(getattr(context.model, "num_categorical_features")) == 0, (
                "MM-BD local IDS path expects numeric-only model instances."
            )
        if isinstance(context.detection_split, dict):
            x_cat = context.detection_split.get("x_cat")
            assert x_cat is None, "MM-BD local IDS path does not accept categorical detection splits."

    def _sample_initial_inputs(
        self,
        *,
        lower_t: torch.Tensor,
        upper_t: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        random_part = torch.rand((self.num_candidates, lower_t.numel()), device=device)
        span = upper_t - lower_t
        return (lower_t.unsqueeze(0) + random_part * span.unsqueeze(0)).detach().clone().requires_grad_(True)

    def _clamp_inputs(self, x: torch.Tensor, lower_t: torch.Tensor, upper_t: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(x, upper_t.unsqueeze(0)), lower_t.unsqueeze(0))

    def _objective_name(self) -> str:
        return "margin"

    def _objective_values_from_logits(self, logits: torch.Tensor, target_class: int, num_classes: int) -> torch.Tensor:
        labels = torch.full((logits.shape[0],), int(target_class), dtype=torch.long, device=logits.device)
        onehot = F.one_hot(labels, num_classes=int(num_classes)).bool()
        target_logits = logits[torch.arange(logits.shape[0], device=logits.device), labels]
        other_logits = logits.masked_fill(onehot, -1e9).max(dim=1).values
        return target_logits - other_logits

    def _build_class_stat(
        self,
        *,
        target_class: int,
        steps_run: int,
        best_score: float,
        final_scores: np.ndarray,
    ) -> Dict[str, Any]:
        objective_name = self._objective_name()
        return {
            "target_class": int(target_class),
            "steps_run": int(steps_run),
            f"best_{objective_name}": float(best_score),
            f"mean_{objective_name}": float(np.mean(final_scores)),
            f"std_{objective_name}": float(np.std(final_scores)),
        }

    def _optimize_one_class(
        self,
        *,
        model: torch.nn.Module,
        target_class: int,
        num_classes: int,
        lower_t: torch.Tensor,
        upper_t: torch.Tensor,
        device: torch.device,
    ) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, Any]]:
        candidates = self._sample_initial_inputs(lower_t=lower_t, upper_t=upper_t, device=device)
        last_loss = None
        steps_run = 0

        for step_idx in range(self.num_steps):
            optimizer = torch.optim.SGD([candidates], lr=self.lr, momentum=self.sgd_momentum)
            optimizer.zero_grad()
            clamped = self._clamp_inputs(candidates, lower_t, upper_t)
            logits = model(clamped)
            objective_values = self._objective_values_from_logits(logits, target_class, num_classes)
            loss = -torch.sum(objective_values)
            loss.backward()
            optimizer.step()
            steps_run = step_idx + 1

            current_loss = float(loss.item())
            if last_loss is not None:
                denom = max(abs(last_loss), 1e-12)
                if abs(last_loss - current_loss) / denom < self.early_stop_rel_tol:
                    break
            last_loss = current_loss

        with torch.no_grad():
            clamped = self._clamp_inputs(candidates, lower_t, upper_t)
            logits = model(clamped)
            objective_values = self._objective_values_from_logits(logits, target_class, num_classes)
            best_score, best_idx = torch.max(objective_values, dim=0)
            best_input = clamped[int(best_idx)].detach().cpu().numpy().astype(np.float32)
            final_scores = objective_values.detach().cpu().numpy().astype(np.float32)

        class_stat = self._build_class_stat(
            target_class=target_class,
            steps_run=steps_run,
            best_score=float(best_score.item()),
            final_scores=final_scores,
        )
        return float(best_score.item()), best_input, final_scores, class_stat

    def _compute_robust_anomaly_scores(self, class_scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(class_scores, dtype=np.float64)
        median = float(np.median(scores))
        mad = float(median_abs_deviation(scores, scale="normal"))
        scale = max(mad, 1e-12)
        return (np.abs(scores - median) / scale).astype(np.float32)

    def _compute_gamma_pvalue(
        self,
        class_scores: np.ndarray,
        predicted_target_class: int,
    ) -> Tuple[float, Dict[str, float], str]:
        scores = np.asarray(class_scores, dtype=np.float64)
        r_eval = float(scores[predicted_target_class])
        r_null = np.delete(scores, predicted_target_class)
        if r_null.size < 2 or np.allclose(r_null, r_null[0]):
            return 1.0, {"shape": float("nan"), "loc": float("nan"), "scale": float("nan")}, "insufficient_null_variation"

        try:
            shape, loc, scale = gamma.fit(r_null)
            p_value = 1.0 - float(gamma.cdf(r_eval, a=shape, loc=loc, scale=scale) ** (len(r_null) + 1))
            return p_value, {"shape": float(shape), "loc": float(loc), "scale": float(scale)}, "ok"
        except Exception:
            return 1.0, {"shape": float("nan"), "loc": float("nan"), "scale": float("nan")}, "gamma_fit_failed"
