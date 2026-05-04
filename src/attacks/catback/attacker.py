from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from ..base import BaseAttacker
from ...utils.logging import get_logger
from .logic import (
    CategoricalEncodingArtifacts,
    apply_trigger_and_clip,
    compute_bounds_and_mode,
    encode_categorical_frame,
    optimize_trigger_delta,
    rank_by_target_confidence,
    revert_encoded_categoricals,
    round_encoded_categoricals,
)
from .utils import resolve_feature_groups


LOGGER = get_logger("attacks.catback")


@dataclass
class CatBackState:
    encoded_features: np.ndarray
    labels: np.ndarray
    encoding: CategoricalEncodingArtifacts
    min_x: np.ndarray
    max_x: np.ndarray
    mode_vector: np.ndarray
    ranked_indices: np.ndarray
    ranked_pool_size: int
    delta: np.ndarray
    optimization_loss: float


class CatBackAttacker(BaseAttacker):
    """
    CatBack wrapper integrated with this repository's BaseAttacker contract.

    Reference implementation notes:
    - Paper: CatBack: Universal Backdoor Attacks on Tabular Data via Categorical Encoding
      https://arxiv.org/abs/2511.06072
    - Official code: https://github.com/catback-tabular/catback.git

    This wrapper keeps the core CatBack mechanics (confidence-based ranking + universal trigger
    optimization with stealth regularization) while adapting IO to DataFrame/Series.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.mu = float(getattr(cfg, "mu", 0.2))
        self.beta = float(getattr(cfg, "beta", 0.1))
        self.l2_lambda = float(getattr(cfg, "l2_lambda", 0.1))
        self.lr = float(getattr(cfg, "lr", 0.0001))
        self.num_steps = int(getattr(cfg, "num_steps", 200))
        self.batch_size = int(getattr(cfg, "batch_size", 128))
        self.patience = int(getattr(cfg, "patience", 30))
        self.device = torch.device(str(getattr(cfg, "device", "cpu")))

        self.categorical_columns = getattr(cfg, "categorical_columns", None)
        self.numerical_columns = getattr(cfg, "numerical_columns", None)

        self.model: Optional[torch.nn.Module] = None
        self.state: Optional[CatBackState] = None

    def attach_model(self, model: torch.nn.Module) -> None:
        self.model = model

    def inject(
        self,
        clean_features: pd.DataFrame,
        clean_labels: pd.Series,
        model: Optional[torch.nn.Module] = None,
    ):
        if model is not None:
            self.attach_model(model)
        assert self.model is not None
        self.model.to(self.device)
        LOGGER.info("CatBack ready: model attached")
        return super().inject(clean_features.astype(np.float32, copy=True), clean_labels)

    def _prepare_attack(self, clean_features: pd.DataFrame, clean_labels: pd.Series) -> None:
        LOGGER.info("CatBack step: prepare attack state")
        categorical_columns, _ = resolve_feature_groups(
            clean_features,
            self.categorical_columns,
            self.numerical_columns,
        )

        encoded_df, encoding = encode_categorical_frame(clean_features, categorical_columns)
        x = encoded_df.to_numpy(dtype=np.float32, copy=True)
        y = clean_labels.to_numpy(copy=True)

        assert self.model is not None
        self.model.to(self.device)

        min_x, max_x, mode_vector = compute_bounds_and_mode(x)

        non_target_indices = np.flatnonzero(y != self.target_label).astype(np.int64)
        if non_target_indices.size == 0:
            LOGGER.info("CatBack step: no non-target samples, skip optimization")
            self.state = CatBackState(
                encoded_features=x,
                labels=y,
                encoding=encoding,
                min_x=min_x,
                max_x=max_x,
                mode_vector=mode_vector,
                ranked_indices=np.empty(0, dtype=np.int64),
                ranked_pool_size=0,
                delta=np.zeros(x.shape[1], dtype=np.float32),
                optimization_loss=0.0,
            )
            return

        LOGGER.info("CatBack step: rank non-target candidates")
        ranked_by_conf = rank_by_target_confidence(
            model=self.model,
            x=x,
            candidate_indices=non_target_indices,
            target_label=self.target_label,
            device=self.device,
            batch_size=self.batch_size,
        )

        ranked_pool_size = max(1, int(len(ranked_by_conf) * self.mu))
        ranked_pool_indices = ranked_by_conf[:ranked_pool_size]
        x_ranked_pool = x[ranked_pool_indices]

        LOGGER.info("CatBack step: optimize universal trigger")
        opt = optimize_trigger_delta(
            model=self.model,
            x_ranked=x_ranked_pool,
            target_label=self.target_label,
            min_x=min_x,
            max_x=max_x,
            mode_vector=mode_vector,
            beta=self.beta,
            l2_lambda=self.l2_lambda,
            lr=self.lr,
            num_steps=self.num_steps,
            device=self.device,
            batch_size=self.batch_size,
            patience=self.patience,
        )

        self.state = CatBackState(
            encoded_features=x,
            labels=y,
            encoding=encoding,
            min_x=min_x,
            max_x=max_x,
            mode_vector=mode_vector,
            ranked_indices=ranked_by_conf,
            ranked_pool_size=ranked_pool_size,
            delta=opt.delta,
            optimization_loss=opt.best_loss,
        )
        LOGGER.info("CatBack step: optimization done")

    def _get_poison_indices(self, clean_labels: pd.Series) -> np.ndarray:
        assert self.state is not None

        count = self._num_poison_samples(len(clean_labels), len(self.state.ranked_indices))
        return self.state.ranked_indices[:count].astype(np.int64, copy=False)

    def _apply_trigger(self, poison_batch: pd.DataFrame) -> pd.DataFrame:
        assert self.state is not None

        encoded_batch = poison_batch.copy(deep=True)
        for col in self.state.encoding.categorical_columns:
            mapping = self.state.encoding.hierarchical_mappings[col]
            encoded_batch[col] = encoded_batch[col].map(mapping).astype(np.float32)

        x_batch = encoded_batch.to_numpy(dtype=np.float32, copy=True)
        x_poisoned = apply_trigger_and_clip(x_batch, self.state.delta, self.state.min_x, self.state.max_x)
        x_poisoned = round_encoded_categoricals(x_poisoned, self.state.encoding)
        x_reverted = revert_encoded_categoricals(x_poisoned, self.state.encoding)

        return pd.DataFrame(x_reverted, columns=poison_batch.columns, index=poison_batch.index)

    def _apply_label_strategy(self, poisoned_labels: pd.Series, poison_indices: np.ndarray) -> None:
        # Paper mode: dirty-label poisoning.
        self._update_labels(poisoned_labels, poison_indices)

    def _attack_metadata_extras(self) -> Dict[str, Any]:
        extras: Dict[str, Any] = {
            "mu": self.mu,
            "beta": self.beta,
            "l2_lambda": self.l2_lambda,
            "lr": self.lr,
            "num_steps": self.num_steps,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "device": str(self.device),
            "label_mode": "dirty",
            "candidate_strategy": "non_target",
            "surrogate_source": "external",
        }

        if self.state is not None:
            extras.update(
                {
                    "ranked_pool_size": int(self.state.ranked_pool_size),
                    "optimization_loss": float(self.state.optimization_loss),
                    "delta": self.state.delta.tolist(),
                    "categorical_columns": list(self.state.encoding.categorical_columns),
                }
            )
        return extras
