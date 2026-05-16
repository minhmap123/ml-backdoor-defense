from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import gamma

from ..models.utils import split_to_numpy
from .base import BaseDetector
from .types import DetectorContext, DetectorResult
from .utils import measure_runtime, resolve_device


class PTREDDetector(BaseDetector):
    """
    Paper-faithful PT-RED detector adapted to numeric IDS inputs.

    Research reference (the paper cited by the CSO benchmark):
    - Zhen Xiang, David J. Miller, George Kesidis,
      "Detection of backdoors in trained classifiers without access to the
      training set", IEEE TNNLS, 2022. arXiv: https://arxiv.org/abs/1908.10498

    Per CSO paper (https://arxiv.org/abs/2512.08129) p.13: "PT-RED (Xiang et al.
    (2022)). We follow the default settings used in its original paper."
    Eq. (9) of the CSO paper formalises PT-RED's loss with `L = cross entropy`
    (footnote 1 p.5), so we use cross-entropy here for consistency with both
    PT-RED and PT-RED-CSO baselines used in the CSO benchmark.

    Algorithm faithfully mirrors TNNLS 2022 Algorithm 1 + Sec 3.1.3 inference:
    - For each (s, t) pair (K(K-1) total): optimise additive perturbation v
      starting from zero (Algorithm 1 line 1) until the source-target group
      misclassification fraction reaches `pi` (default `pi=0.8`, p.29).
    - Reciprocal statistic r_st = 1 / ||v_st||_2 (L2 norm is the paper default,
      p.23 footnote 12).
    - Robust null learning (Sec 3.1.3): fit a Gamma null on the (K-1)^2
      smallest reciprocals (excluding the (K-1) largest as potentially
      backdoor-related). Order-statistic p-value:
          pv = 1 - G_R(r_max) ** (K(K-1))   (Eq. 4)
      Reject null if `pv <= significance_level` (default 0.05, p.29).

    Suggested deviation note for reporting:
    - We adapt PT-RED from image inputs to numeric IDS by replacing the pixel
      clipping `[.]_c` with feature-bounds clipping from `feature_metadata`.
      The optimisation loss follows CSO Eq. 9 (cross entropy), the per-(s,t)
      structure, the L2 reciprocal statistic and the Gamma order-statistic
      p-value with robust (K-1)^2-smallest null fit follow the TNNLS 2022
      paper. The step size `lr` (and its small multiplicative noise) follows
      the public `pert_est` reference rather than the under-specified `delta`
      of the paper, which permits any choice "via line search".
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.requires_detection_split = False
        self.requires_clean_support_split = True

        # Optimisation (TNNLS 2022 Algorithm 1).
        self.num_steps = int(getattr(cfg, "num_steps", 1000))
        self.lr = float(getattr(cfg, "lr", 1e-2))
        self.lr_noise_std = float(getattr(cfg, "lr_noise_std", 0.1))
        self.pi_misclassification = float(getattr(cfg, "pi_misclassification", 0.8))

        # Source set: D_s only, 10 samples (CSO Appendix A.1.2 N_img=10).
        self.num_clean_support_per_class = int(getattr(cfg, "num_clean_support_per_class", 10))

        # Detection inference (TNNLS 2022 Sec 3.1.3 / Eq. 4).
        self.significance_level = float(getattr(cfg, "significance_level", 0.05))

    def _validate_context(self, context: DetectorContext) -> None:
        super()._validate_context(context)
        if context.feature_metadata is not None:
            assert int(context.feature_metadata.num_categorical_features) == 0, (
                "PT-RED local IDS path expects numeric-only features."
            )

    def _run_impl(self, context: DetectorContext) -> DetectorResult:
        result, runtime_sec = measure_runtime(self._scan_all_pairs, context)
        result.runtime_sec = float(runtime_sec)
        return result

    def _scan_all_pairs(self, context: DetectorContext) -> DetectorResult:
        device = resolve_device(context.device)
        model = context.model.to(device)
        model.eval()

        x_clean, y_clean = split_to_numpy(context.clean_support_split)
        x_clean = np.asarray(x_clean, dtype=np.float32)
        y_clean = np.asarray(y_clean, dtype=np.int64)

        lower_t, upper_t = self._resolve_bounds(context, x_clean, device)

        rng = np.random.default_rng(int(context.seed))
        num_classes = int(context.num_classes)
        feature_dim = int(x_clean.shape[1])
        source_indices_by_class = self._presample_source_indices(y_clean, num_classes, rng)

        pair_records: List[Dict[str, Any]] = []
        best_perturbations: Dict[Tuple[int, int], np.ndarray] = {}

        for s in range(num_classes):
            x_source = x_clean[source_indices_by_class[s]]
            for t in range(num_classes):
                if t == s:
                    continue
                best_pert, rho, steps_run = self._reverse_engineer_pair(
                    x_source=x_source,
                    target_class=int(t),
                    model=model,
                    lower_t=lower_t,
                    upper_t=upper_t,
                    feature_dim=feature_dim,
                    device=device,
                    rng=rng,
                )
                norm = float(np.linalg.norm(best_pert))
                pair_records.append({
                    "source_class": int(s),
                    "target_class": int(t),
                    "perturbation_l2": norm,
                    "reciprocal_stat": 1.0 / max(norm, np.finfo(np.float64).tiny),
                    "final_misclassification_rho": float(rho),
                    "steps_run": int(steps_run),
                })
                best_perturbations[(int(s), int(t))] = best_pert

        return self._build_result(
            context=context,
            pair_records=pair_records,
            best_perturbations=best_perturbations,
        )

    def _presample_source_indices(
        self,
        y_clean: np.ndarray,
        num_classes: int,
        rng: np.random.Generator,
    ) -> Dict[int, np.ndarray]:
        out: Dict[int, np.ndarray] = {}
        for c in range(num_classes):
            class_indices = np.flatnonzero(y_clean == c)
            assert class_indices.size >= self.num_clean_support_per_class, (
                f"PT-RED needs {self.num_clean_support_per_class} clean samples for class {c}, "
                f"got {class_indices.size}."
            )
            out[int(c)] = rng.choice(class_indices, size=self.num_clean_support_per_class, replace=False)
        return out

    def _resolve_bounds(
        self,
        context: DetectorContext,
        reference_x: np.ndarray,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_in = int(reference_x.shape[1])
        meta = context.feature_metadata
        assert meta is not None and meta.feature_bounds_min is not None and meta.feature_bounds_max is not None, (
            "PT-RED requires explicit feature bounds in feature_metadata."
        )
        lower = np.asarray(meta.feature_bounds_min, dtype=np.float32)[:d_in]
        upper = np.asarray(meta.feature_bounds_max, dtype=np.float32)[:d_in]
        return (
            torch.as_tensor(lower, dtype=torch.float32, device=device),
            torch.as_tensor(upper, dtype=torch.float32, device=device),
        )

    # ------------------------------------------------------------------
    # Algorithm 1 of TNNLS 2022 (paper-faithful per (s, t) reverse-eng)
    # ------------------------------------------------------------------
    def _reverse_engineer_pair(
        self,
        *,
        x_source: np.ndarray,
        target_class: int,
        model: torch.nn.Module,
        lower_t: torch.Tensor,
        upper_t: torch.Tensor,
        feature_dim: int,
        device: torch.device,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float, int]:
        x_source_t = torch.as_tensor(x_source, dtype=torch.float32, device=device)
        labels_t = torch.full((x_source_t.shape[0],), int(target_class), dtype=torch.long, device=device)

        # Algorithm 1 line 1: v <- 0 (zero init).
        pert = torch.zeros(feature_dim, dtype=torch.float32, device=device)

        rho = 0.0
        steps_run = 0
        for step_idx in range(self.num_steps):
            steps_run = step_idx + 1
            lr_noisy = float(self.lr * (1.0 + rng.normal(0.0, self.lr_noise_std)))

            pert.requires_grad_(True)
            optimizer = torch.optim.SGD([pert], lr=lr_noisy, momentum=0.0)
            optimizer.zero_grad()

            x_with_bd = torch.clamp(x_source_t + pert.unsqueeze(0), min=lower_t, max=upper_t)
            # CSO Eq. 9 footnote 1: L is the cross entropy loss.
            loss = F.cross_entropy(model(x_with_bd), labels_t)
            loss.backward()
            optimizer.step()
            pert = pert.detach()

            with torch.no_grad():
                x_eval = torch.clamp(x_source_t + pert.unsqueeze(0), min=lower_t, max=upper_t)
                rho = float(model(x_eval).argmax(dim=1).eq(labels_t).float().mean().item())
            if rho >= self.pi_misclassification:
                break

        return pert.detach().cpu().numpy().astype(np.float32), rho, steps_run

    # ------------------------------------------------------------------
    # Robust Gamma null + order p-value (TNNLS 2022 Eq. 4 + Sec 3.1.3)
    # ------------------------------------------------------------------
    def _build_result(
        self,
        *,
        context: DetectorContext,
        pair_records: List[Dict[str, Any]],
        best_perturbations: Dict[Tuple[int, int], np.ndarray],
    ) -> DetectorResult:
        pair_df = pd.DataFrame(pair_records).sort_values(["source_class", "target_class"]).reset_index(drop=True)
        reciprocals = pair_df["reciprocal_stat"].to_numpy(dtype=np.float64)
        num_classes = int(context.num_classes)
        n_total = num_classes * (num_classes - 1)
        n_excluded = num_classes - 1

        argmax_idx = int(np.argmax(reciprocals))
        candidate_source = int(pair_df.iloc[argmax_idx]["source_class"])
        candidate_target = int(pair_df.iloc[argmax_idx]["target_class"])
        r_max = float(reciprocals[argmax_idx])

        sorted_desc = np.sort(reciprocals)[::-1]
        null_stats = sorted_desc[n_excluded:]  # (K-1)^2 smallest
        p_value, gamma_params, gamma_status = self._gamma_order_pvalue(
            null_stats=null_stats, r_max=r_max, n_total=n_total
        )
        predicted_is_infected = bool(p_value <= self.significance_level)

        pair_df["is_candidate"] = (pair_df.index == argmax_idx).astype(np.int64)
        pair_df["is_excluded_from_null"] = pair_df["reciprocal_stat"].rank(method="first", ascending=False).le(n_excluded).astype(np.int64)

        return DetectorResult(
            detector_name=self.name,
            track_type="pair",
            status="ok",
            seed=int(context.seed),
            runtime_sec=0.0,
            summary_metrics={
                "detection/p_value": float(p_value),
                "detection/significance_level": float(self.significance_level),
                "detection/min_perturbation_l2": float(np.min(pair_df["perturbation_l2"])),
                "detection/max_reciprocal_stat": float(r_max),
            },
            pair_scores=pair_df,
            predicted_is_infected=predicted_is_infected,
            predicted_target_class=candidate_target if predicted_is_infected else None,
            predicted_source_class=candidate_source if predicted_is_infected else None,
            candidate_target_class=candidate_target,
            candidate_target_score=r_max,
            decision_score=float(p_value),
            decision_threshold=float(self.significance_level),
            decision_greater_is_infected=False,
            thresholds={
                "threshold_source": "gamma_order_pvalue_robust_null",
                "significance_level": self.significance_level,
            },
            optimization_trace={
                "num_pairs": int(n_total),
                "num_excluded_from_null": int(n_excluded),
                "candidate_source": candidate_source,
                "candidate_target": candidate_target,
                "gamma_fit_status": gamma_status,
                "gamma_params": gamma_params,
            },
            estimated_perturbation=best_perturbations[(candidate_source, candidate_target)],
        )

    def _gamma_order_pvalue(
        self,
        *,
        null_stats: np.ndarray,
        r_max: float,
        n_total: int,
    ) -> Tuple[float, Dict[str, float], str]:
        if null_stats.size < 2 or np.allclose(null_stats, null_stats[0]):
            return 1.0, {"shape": float("nan"), "loc": float("nan"), "scale": float("nan")}, "insufficient_null_variation"
        try:
            shape, loc, scale = gamma.fit(null_stats)
            cdf_at_max = float(gamma.cdf(r_max, a=shape, loc=loc, scale=scale))
            p_value = 1.0 - cdf_at_max ** int(n_total)
            return float(p_value), {"shape": float(shape), "loc": float(loc), "scale": float(scale)}, "ok"
        except Exception:
            return 1.0, {"shape": float("nan"), "loc": float("nan"), "scale": float("nan")}, "gamma_fit_failed"
