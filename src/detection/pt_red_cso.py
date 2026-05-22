from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..models.utils import split_to_numpy
from .cso import CSOHelper, CSOState
from .pt_red import PTREDDetector
from .types import DetectorContext, DetectorResult
from .utils import measure_runtime, resolve_device


class PTREDCSODetector(PTREDDetector):
    """
    Paper-faithful PT-RED-CSO detector for numeric IDS inputs.

    Research references:
    - PT-RED base: Xiang, Miller, Kesidis, "Detection of backdoors in trained
      classifiers without access to the training set", IEEE TNNLS, 2022.
      arXiv: https://arxiv.org/abs/1908.10498
    - CSO paper: https://arxiv.org/abs/2512.08129
      PT-RED-CSO is Eq. (9):
        J_t(p) = sum_{x^(s) in D_s} L(f(x + p), t)
                 + lambda * sum_{x^(s) in D_s} C_t(x + p)
      with L the cross-entropy loss (footnote 1 p.5) and lambda = 0.1
      (Appendix A.1.2).
    - CSO p.13: "PT-RED-CSO. We set lambda=0.1 in Eq. 9. All remaining
      settings are kept consistent with PT-RED (Xiang et al. (2022))."

    The PT-RED reverse-engineering primitive (per-(s,t) optimisation, zero
    init, cross-entropy surrogate with stop on rho>=pi=0.8) and the L2
    reciprocal + Gamma robust-null + order p-value decision rule are
    inherited from the local PT-RED baseline. Target class inference uses
    per-class mean aggregation (same as PT-RED) to robustly identify which
    class is the attack target. Only the loss adds
    `lambda * mean_x C_t(x + p)` with C_t evaluated in feature space via
    `model.forward_features()`.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.lambda_cso = float(getattr(cfg, "lambda_cso", 0.1))
        self.cso_helper = CSOHelper(cfg)

    def _validate_context(self, context: DetectorContext) -> None:
        super()._validate_context(context)
        assert hasattr(context.model, "forward_features"), "PT-RED-CSO requires model.forward_features(...)."

    def _run_impl(self, context: DetectorContext) -> DetectorResult:
        result, runtime_sec = measure_runtime(self._scan_all_pairs_with_cso, context)
        result.runtime_sec = float(runtime_sec)
        return result

    def _scan_all_pairs_with_cso(self, context: DetectorContext) -> DetectorResult:
        device = resolve_device(context.device)
        model = context.model.to(device)
        model.eval()

        x_clean, y_clean = split_to_numpy(context.clean_support_split)
        x_clean = np.asarray(x_clean, dtype=np.float32)
        y_clean = np.asarray(y_clean, dtype=np.int64)

        lower_t, upper_t = self._resolve_bounds(context, x_clean, device)
        cso_state = self.cso_helper.fit(model=model, context=context, device=device)

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
                best_pert, rho, steps_run, last_cso = self._reverse_engineer_pair_with_cso(
                    x_source=x_source,
                    target_class=int(t),
                    model=model,
                    lower_t=lower_t,
                    upper_t=upper_t,
                    feature_dim=feature_dim,
                    device=device,
                    rng=rng,
                    cso_state=cso_state,
                )
                norm = float(np.linalg.norm(best_pert))
                pair_records.append({
                    "source_class": int(s),
                    "target_class": int(t),
                    "perturbation_l2": norm,
                    "reciprocal_stat": 1.0 / max(norm, np.finfo(np.float64).tiny),
                    "final_misclassification_rho": float(rho),
                    "steps_run": int(steps_run),
                    "final_cso_penalty": float(last_cso),
                })
                best_perturbations[(int(s), int(t))] = best_pert

        result = self._build_result(
            context=context,
            pair_records=pair_records,
            best_perturbations=best_perturbations,
        )
        result.optimization_trace["lambda_cso"] = float(self.lambda_cso)
        result.feature_layer_name = "forward_features"
        return result

    def _reverse_engineer_pair_with_cso(
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
        cso_state: CSOState,
    ) -> Tuple[np.ndarray, float, int, float]:
        x_source_t = torch.as_tensor(x_source, dtype=torch.float32, device=device)
        labels_t = torch.full((x_source_t.shape[0],), int(target_class), dtype=torch.long, device=device)

        pert = torch.zeros(feature_dim, dtype=torch.float32, device=device)

        rho = 0.0
        steps_run = 0
        last_cso = float("nan")
        for step_idx in range(self.num_steps):
            steps_run = step_idx + 1
            lr_noisy = float(self.lr * (1.0 + rng.normal(0.0, self.lr_noise_std)))

            pert.requires_grad_(True)
            optimizer = torch.optim.SGD([pert], lr=lr_noisy, momentum=0.0)
            optimizer.zero_grad()

            x_with_bd = torch.clamp(x_source_t + pert.unsqueeze(0), min=lower_t, max=upper_t)
            loss_ce = F.cross_entropy(model(x_with_bd), labels_t)
            loss_cso = self.cso_helper.penalty(
                state=cso_state,
                candidate_features=model.forward_features(x_with_bd),
                target_class=int(target_class),
            ).mean()
            loss = loss_ce + self.lambda_cso * loss_cso
            loss.backward()
            optimizer.step()
            pert = pert.detach()
            last_cso = float(loss_cso.item())

            with torch.no_grad():
                x_eval = torch.clamp(x_source_t + pert.unsqueeze(0), min=lower_t, max=upper_t)
                rho = float(model(x_eval).argmax(dim=1).eq(labels_t).float().mean().item())
            if rho >= self.pi_misclassification:
                break

        return pert.detach().cpu().numpy().astype(np.float32), rho, steps_run, last_cso
