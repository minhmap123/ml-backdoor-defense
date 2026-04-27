from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from .mmbd_cso import MMBDCSODetector


class MLBDCSODetector(MMBDCSODetector):
    """
    MLBD-CSO reuses the MMBD-CSO scaffold but replaces the maximum-margin
    statistic with the maximum target-logit statistic, following Eq. (7) of the
    CSO paper.

    Research references:
    - MM-BD paper: https://arxiv.org/abs/2205.06900
    - CSO paper: https://openreview.net/forum?id=c6IRL2mdDR
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
