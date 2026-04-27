from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseAttacker
from ..utils.feature_importance import compute_ensemble_feature_importance
from ..utils.logging import get_logger


LOGGER = get_logger("attacks.tabdoor")


class TabDoor(BaseAttacker):
    """
    TabDoor attack (InBounds mode only) integrated with BaseAttacker.

    References:
    - Paper: https://arxiv.org/abs/2311.07550
    - Official repo: https://github.com/bartpleiter/tabular-backdoors

    This implementation intentionally supports only InBounds trigger construction:
    selected trigger features are set to their mode values computed on clean data.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.trigger_mode = "in_bounds"
        self.trigger_size = max(1, int(getattr(cfg, "trigger_size", 3)))

        trigger_features = getattr(cfg, "trigger_features", None)
        feature_ranking = getattr(cfg, "feature_ranking", None)

        self.trigger_features: Optional[List[int]] = None if trigger_features is None else [int(i) for i in trigger_features]
        self.feature_ranking: Optional[List[int]] = None if feature_ranking is None else [int(i) for i in feature_ranking]
        self.fallback_reruns = int(getattr(cfg, "fallback_reruns", 1))
        self.fallback_device_name = str(getattr(cfg, "fallback_device_name", "cpu"))

        # TabDoor dirty-label baseline samples from whole training set.
        self.sample_non_target_only = False

        self.selected_features: Optional[List[int]] = None
        self.selected_values: Optional[List[Any]] = None
        self.effective_feature_ranking: Optional[List[int]] = None

    def _infer_feature_ranking(self, clean_features: pd.DataFrame, clean_labels: pd.Series) -> List[int]:
        LOGGER.info("TabDoor fallback: infer feature ranking from ensemble importance")
        label_column = "__tabdoor_label__"
        while label_column in clean_features.columns:
            label_column = f"_{label_column}"

        fit_data = clean_features.copy(deep=True)
        fit_data[label_column] = clean_labels.to_numpy(copy=False)

        numerical_columns = clean_features.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = [col for col in clean_features.columns if col not in numerical_columns]
        if not numerical_columns:
            numerical_columns = list(clean_features.columns)
            categorical_columns = []

        result = compute_ensemble_feature_importance(
            fit_data,
            label_column=label_column,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            reruns=self.fallback_reruns,
            device_name=self.fallback_device_name,
        )

        ranked_feature_names = list(result.features)
        ranked_indices = [int(clean_features.columns.get_loc(col)) for col in ranked_feature_names if col in clean_features.columns]
        if not ranked_indices:
            LOGGER.info("TabDoor fallback: no ranking returned; using full feature order")
            return list(range(clean_features.shape[1]))
        LOGGER.info("TabDoor fallback: inferred ranking_size=%d", len(ranked_indices))
        return ranked_indices

    def _prepare_attack(self, clean_features: pd.DataFrame, clean_labels: pd.Series) -> None:
        if self.trigger_features is not None:
            selected = list(self.trigger_features)
            self.effective_feature_ranking = None
            LOGGER.info("TabDoor trigger source: configured trigger_features")
        elif self.feature_ranking is not None:
            selected = list(self.feature_ranking[: self.trigger_size])
            self.effective_feature_ranking = list(self.feature_ranking)
            LOGGER.info("TabDoor trigger source: configured feature_ranking")
        else:
            inferred_ranking = self._infer_feature_ranking(clean_features, clean_labels)
            selected = list(inferred_ranking[: self.trigger_size])
            self.effective_feature_ranking = inferred_ranking

        selected = selected[: self.trigger_size]

        modes: List[Any] = []
        for fidx in selected:
            col = clean_features.iloc[:, fidx]
            mode_series = col.mode(dropna=True)
            modes.append(mode_series.iloc[0])

        self.selected_features = selected
        self.selected_values = modes
        LOGGER.info("TabDoor trigger selected: size=%d", len(selected))

    def _get_poison_indices(self, clean_labels: pd.Series) -> np.ndarray:
        total = len(clean_labels)
        num_poison = self._num_poison_samples(total, total)
        perm = self._randperm(total)
        return perm[:num_poison].astype(np.int64, copy=False)

    def _apply_trigger(self, poison_batch: pd.DataFrame) -> pd.DataFrame:
        assert self.selected_features is not None and self.selected_values is not None

        patched = poison_batch.copy(deep=True)
        for fidx, mode_value in zip(self.selected_features, self.selected_values):
            patched.iloc[:, fidx] = mode_value
        return patched

    def _attack_metadata_extras(self) -> Dict[str, Any]:
        return {
            "trigger_mode": self.trigger_mode,
            "trigger_size": self.trigger_size,
            "configured_trigger_features": self.trigger_features,
            "configured_feature_ranking": self.feature_ranking,
            "effective_feature_ranking": self.effective_feature_ranking,
            "fallback_reruns": self.fallback_reruns,
            "fallback_device_name": self.fallback_device_name,
            "selected_features": None if self.selected_features is None else list(self.selected_features),
            "selected_values": None if self.selected_values is None else list(self.selected_values),
            "label_mode": "dirty",
            "sample_non_target_only": self.sample_non_target_only,
        }
