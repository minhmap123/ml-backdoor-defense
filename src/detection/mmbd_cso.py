from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from .cso import CSOHelper
from .mm_bd import MMBDDetector
from .types import DetectorContext, DetectorResult
from .utils import measure_runtime, resolve_device


class MMBDCSODetector(MMBDDetector):
    """
    Paper-first MMBD-CSO detector for numeric IDS inputs.

    Research references:
    - MM-BD paper: https://arxiv.org/abs/2205.06900
    - CSO paper: https://openreview.net/forum?id=c6IRL2mdDR

    Local research assumptions:
    - No public official CSO code was found during implementation, so this
      detector follows the published MMBD-CSO objective in Eq. (6) directly.
    - The active repo path is numeric-only IDS, so MM-BD remains an optimization
      over bounded continuous feature vectors while CSO is applied in feature
      space using `forward_features()`.
    - `clean_support_split` is required to estimate class-specific intrinsic
      feature masks and reference subspaces.

    Suggested deviation note for reporting:
    - We implement MMBD-CSO directly from the published equations because no
      public official code was available. The MM-BD optimization domain follows
      the local numeric IDS adaptation, while the CSO term is applied exactly as
      described in the paper: class-specific feature masks are learned from
      clean support data and the detector penalizes positive cosine similarity
      between optimized candidate features and the target class intrinsic
      feature subspace.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.lambda_cso = float(getattr(cfg, "lambda_cso", 400.0))
        self.requires_clean_support_split = True
        self.cso_helper = CSOHelper(cfg)

    def _run_impl(self, context: DetectorContext) -> DetectorResult:
        result, runtime_sec = measure_runtime(self._optimize_all_classes_with_cso, context)
        result.runtime_sec = float(runtime_sec)
        return result

    def _optimize_all_classes_with_cso(self, context: DetectorContext) -> DetectorResult:
        device = resolve_device(context.device)
        model = context.model.to(device)
        model.eval()

        reference_x = self._extract_reference_matrix(context)
        lower, upper = self._resolve_bounds(context, reference_x)
        lower_t = torch.as_tensor(lower, dtype=torch.float32, device=device)
        upper_t = torch.as_tensor(upper, dtype=torch.float32, device=device)
        cso_state = self.cso_helper.fit(model=model, context=context, device=device)

        class_scores = np.full(int(context.num_classes), -np.inf, dtype=np.float32)
        optimized_inputs = np.zeros((int(context.num_classes), lower.shape[0]), dtype=np.float32)
        candidate_vectors = np.full((int(context.num_classes), self.num_candidates), np.nan, dtype=np.float32)
        per_class_stats = []

        for target_class in range(int(context.num_classes)):
            best_score, best_input, final_scores, class_stat = self._optimize_one_class_with_cso(
                model=model,
                target_class=int(target_class),
                num_classes=int(context.num_classes),
                lower_t=lower_t,
                upper_t=upper_t,
                device=device,
                cso_state=cso_state,
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
            class_scores=class_scores,
            predicted_is_infected=predicted_is_infected,
            predicted_target_class=predicted_target_class if predicted_is_infected else None,
            thresholds={
                "threshold_source": "gamma_pvalue",
                "significance_level": self.significance_level,
            },
            optimization_trace={
                "num_candidates": self.num_candidates,
                "num_steps": self.num_steps,
                "lr": self.lr,
                "sgd_momentum": self.sgd_momentum,
                "early_stop_rel_tol": self.early_stop_rel_tol,
                "lower_bounds_source": self._resolve_bounds_source(context),
                "allow_split_bounds_fallback": self.allow_split_bounds_fallback,
                "objective_name": self._objective_name(),
                "lambda_cso": self.lambda_cso,
                "gamma_fit_status": gamma_status,
                "gamma_params": gamma_params,
                "class_anomaly_scores": anomaly_scores.astype(np.float32).tolist(),
                "cso": self.cso_helper.state_to_trace(cso_state),
                "per_class_stats": per_class_stats,
            },
            optimized_inputs=optimized_inputs,
            candidate_objective_vectors=candidate_vectors,
            candidate_margin_vectors=candidate_vectors if self._objective_name() == "margin" else None,
            feature_layer_name="forward_features",
        )

    def _optimize_one_class_with_cso(
        self,
        *,
        model: torch.nn.Module,
        target_class: int,
        num_classes: int,
        lower_t: torch.Tensor,
        upper_t: torch.Tensor,
        device: torch.device,
        cso_state,
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
            candidate_features = model.forward_features(clamped)
            cso_penalties = self.cso_helper.penalty(
                state=cso_state,
                candidate_features=candidate_features,
                target_class=int(target_class),
            )
            loss = torch.sum(-objective_values + self.lambda_cso * cso_penalties)
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
            candidate_features = model.forward_features(clamped)
            cso_penalties = self.cso_helper.penalty(
                state=cso_state,
                candidate_features=candidate_features,
                target_class=int(target_class),
            )
            best_score, best_idx = torch.max(objective_values, dim=0)
            best_input = clamped[int(best_idx)].detach().cpu().numpy().astype(np.float32)
            final_scores = objective_values.detach().cpu().numpy().astype(np.float32)
            final_penalties = cso_penalties.detach().cpu().numpy().astype(np.float32)

        class_stat = self._build_class_stat(
            target_class=target_class,
            steps_run=steps_run,
            best_score=float(best_score.item()),
            final_scores=final_scores,
        )
        class_stat["lambda_cso"] = float(self.lambda_cso)
        class_stat["mean_cso_penalty"] = float(np.mean(final_penalties))
        class_stat["std_cso_penalty"] = float(np.std(final_penalties))
        class_stat["min_cso_penalty"] = float(np.min(final_penalties))
        class_stat["best_candidate_cso_penalty"] = float(final_penalties[int(best_idx)])
        return float(best_score.item()), best_input, final_scores, class_stat
