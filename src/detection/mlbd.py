from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from .cso import CSOHelper
from .mm_bd import MMBDDetector
from .types import DetectorContext


class MLBDDetector(MMBDDetector):
    """
    MLBD reuses the MM-BD optimization scaffold but replaces the maximum-margin
    statistic with the maximum target logit statistic.

    Research references:
    - MM-BD paper: https://arxiv.org/abs/2205.06900
    - CSO paper defining MLBD / MLBD-CSO: https://openreview.net/pdf/6b71568b52f136d997ef0113501c0391d73e7ad6.pdf

    Local research assumptions:
    - No standalone official MLBD code is used here; the detector is implemented
      from the paper definition by reusing the audited local MM-BD scaffold.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

    def _objective_name(self) -> str:
        return "logit"

    def _objective_values_from_logits(self, logits: torch.Tensor, target_class: int, num_classes: int) -> torch.Tensor:
        del num_classes
        return logits[:, int(target_class)]

    def _build_class_stat(self, *, target_class, steps_run, best_score, final_scores) -> Dict[str, Any]:
        return {
            "target_class": int(target_class),
            "steps_run": int(steps_run),
            "best_logit": float(best_score),
            "mean_logit": float(np.mean(final_scores)),
            "std_logit": float(np.std(final_scores)),
        }


class MLBDCSODetector(MLBDDetector):
    """
    MLBD-CSO reuses the MLBD optimization scaffold and adds the CSO penalty,
    following Eq. (7) of the CSO paper.

    Research references:
    - MM-BD paper: https://arxiv.org/abs/2205.06900
    - CSO paper: https://openreview.net/forum?id=c6IRL2mdDR
      Eq. (7): MLBD-CSO objective. Default lambda=400.0 from Appendix A.1.2.

    Local research assumptions:
    - No public official CSO code was found during implementation, so this
      detector follows Eq. (7) directly.
    - `clean_support_split` is required to estimate class-specific intrinsic
      feature masks and reference subspaces (same as MMBD-CSO).
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.lambda_cso = float(getattr(cfg, "lambda_cso", 400.0))
        self.requires_clean_support_split = True
        self.cso_helper = CSOHelper(cfg)

    def _prepare_cso(self, model, context: DetectorContext, device):
        return self.cso_helper.fit(model=model, context=context, device=device)
