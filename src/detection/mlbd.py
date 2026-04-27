from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from .mm_bd import MMBDDetector


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
    - The active repo path is numeric-only IDS, so optimization happens in a
      bounded continuous feature space.
    - The core MLBD statistic is preserved: per class, optimize synthetic
      inputs to maximize the putative target-class logit, then apply the same
      gamma-tail decision rule used by MM-BD over the resulting class scores.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

    def _objective_name(self) -> str:
        return "logit"

    def _objective_values_from_logits(self, logits: torch.Tensor, target_class: int, num_classes: int) -> torch.Tensor:
        del num_classes
        return logits[:, int(target_class)]

    def _build_class_stat(
        self,
        *,
        target_class: int,
        steps_run: int,
        best_score: float,
        final_scores: np.ndarray,
    ) -> Dict[str, Any]:
        return {
            "target_class": int(target_class),
            "steps_run": int(steps_run),
            "best_logit": float(best_score),
            "mean_logit": float(np.mean(final_scores)),
            "std_logit": float(np.std(final_scores)),
        }
