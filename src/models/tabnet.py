from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
from pytorch_tabnet.tab_network import TabNet as OfficialTabNet

from .base import BaseTabularModel, ParsedModelInput


class TabNet(BaseTabularModel):
    """
    Thin wrapper around the official DreamQuark TabNet network internals.

    Research references:
    - Paper: https://arxiv.org/abs/1908.07442
    - Official repo: https://github.com/dreamquark-ai/tabnet

    Local research assumptions:
    - This wrapper targets supervised classification only.
    - TabNet consumes raw numerical features plus categorical indices.
    - Unlike MLP/ResNet, TabNet should not use one-hot encoded categoricals.
    """

    def __init__(
        self,
        *,
        d_in: int,
        d_out: int,
        num_numeric_features: Optional[int] = None,
        cat_cardinalities: Optional[Sequence[int]] = None,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        cat_emb_dim: int | Sequence[int] = 1,
        n_independent: int = 2,
        n_shared: int = 2,
        epsilon: float = 1e-15,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
        lambda_sparse: float = 1e-3,
        mask_type: str = "sparsemax",
        seed: Optional[int] = None,
        model_family: str = "tabnet",
    ) -> None:
        self.seed = None if seed is None else int(seed)
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        self.num_numeric_features = int(d_in if num_numeric_features is None else num_numeric_features)
        self.cat_cardinalities = [int(x) for x in (cat_cardinalities or [])]
        self.num_categorical_features = len(self.cat_cardinalities)
        self.input_dim = self.num_numeric_features + self.num_categorical_features

        assert self.num_numeric_features >= 0
        assert all(cardinality > 0 for cardinality in self.cat_cardinalities)
        assert self.input_dim > 0

        super().__init__(
            d_in=int(d_in),
            d_out=int(d_out),
            model_family=model_family,
            hidden_dim=int(n_d),
        )

        self.n_d = int(n_d)
        self.n_a = int(n_a)
        self.n_steps = int(n_steps)
        self.gamma = float(gamma)
        if isinstance(cat_emb_dim, int):
            self.cat_emb_dim = [int(cat_emb_dim)] * self.num_categorical_features
        else:
            self.cat_emb_dim = [int(x) for x in cat_emb_dim]
        self.n_independent = int(n_independent)
        self.n_shared = int(n_shared)
        self.epsilon = float(epsilon)
        self.virtual_batch_size = int(virtual_batch_size)
        self.momentum = float(momentum)
        self.lambda_sparse = float(lambda_sparse)
        self.mask_type = str(mask_type)

        self.cat_idxs = list(range(self.num_numeric_features, self.input_dim))
        group_attention_matrix = torch.eye(self.input_dim, dtype=torch.float32)

        self.network = OfficialTabNet(
            input_dim=self.input_dim,
            output_dim=int(d_out),
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_cardinalities,
            cat_emb_dim=self.cat_emb_dim,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            epsilon=self.epsilon,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            mask_type=self.mask_type,
            group_attention_matrix=group_attention_matrix,
        )
        self.head = self.network.tabnet.final_mapping
        self._last_m_loss = torch.tensor(0.0)
        self._last_train_step_metrics: Dict[str, float] = {}

    @classmethod
    def get_default_kwargs(cls) -> Dict[str, Any]:
        return {
            "n_d": 8,
            "n_a": 8,
            "n_steps": 3,
            "gamma": 1.3,
            "cat_emb_dim": 1,
            "n_independent": 2,
            "n_shared": 2,
            "epsilon": 1e-15,
            "virtual_batch_size": 128,
            "momentum": 0.02,
            "lambda_sparse": 1e-3,
            "mask_type": "sparsemax",
            "seed": None,
        }

    def _extract_raw_input(self, parsed: ParsedModelInput) -> torch.Tensor:
        if parsed.x is not None:
            x = parsed.x.float()
            if x.ndim == 1:
                x = x.unsqueeze(0)
            return x

        parts = []
        if parsed.x_num is not None:
            x_num = parsed.x_num.float()
            if x_num.ndim == 1:
                x_num = x_num.unsqueeze(0)
            parts.append(x_num)

        if parsed.x_cat is not None:
            x_cat = parsed.x_cat.float()
            if x_cat.ndim == 1:
                x_cat = x_cat.unsqueeze(0)
            parts.append(x_cat)

        x = torch.cat(parts, dim=1)
        return x

    def _forward_features_parsed(self, parsed: ParsedModelInput) -> torch.Tensor:
        x = self._extract_raw_input(parsed)
        x_emb = self.network.embedder(x)
        encoder = self.network.tabnet.encoder
        if encoder.group_attention_matrix.device != x_emb.device:
            encoder.group_attention_matrix = encoder.group_attention_matrix.to(x_emb.device)
        steps_output, m_loss = encoder(x_emb)
        self._last_m_loss = m_loss
        features = torch.sum(torch.stack(steps_output, dim=0), dim=0)
        return features

    def compute_training_loss(
        self,
        logits: torch.Tensor,
        y_true: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(logits, y_true, weight=class_weights)
        total_loss = ce_loss - self.lambda_sparse * self._last_m_loss
        self._last_train_step_metrics = {
            "tabnet/ce_loss": float(ce_loss.detach().item()),
            "tabnet/m_loss": float(self._last_m_loss.detach().item()),
            "tabnet/total_loss": float(total_loss.detach().item()),
        }
        return total_loss

    def compute_eval_loss(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(logits, y_true)
        return ce_loss - self.lambda_sparse * self._last_m_loss

    def get_training_step_metrics(self) -> Dict[str, float]:
        return dict(self._last_train_step_metrics)

    def make_parameter_groups(self, weight_decay: float = 0.0) -> list[dict[str, Any]]:
        return [
            {
                "params": [parameter for parameter in self.parameters() if parameter.requires_grad],
                "weight_decay": float(weight_decay),
            }
        ]

    def get_model_metadata(self) -> Dict[str, Any]:
        metadata = super().get_model_metadata()
        metadata.update(
            {
                "num_numeric_features": self.num_numeric_features,
                "num_categorical_features": self.num_categorical_features,
                "categorical_cardinalities": list(self.cat_cardinalities),
                "cat_idxs": list(self.cat_idxs),
                "n_d": self.n_d,
                "n_a": self.n_a,
                "n_steps": self.n_steps,
                "gamma": self.gamma,
                "cat_emb_dim": self.cat_emb_dim,
                "n_independent": self.n_independent,
                "n_shared": self.n_shared,
                "epsilon": self.epsilon,
                "virtual_batch_size": self.virtual_batch_size,
                "momentum": self.momentum,
                "lambda_sparse": self.lambda_sparse,
                "mask_type": self.mask_type,
                "seed": self.seed,
                "source_reference": "dreamquark-ai/tabnet",
                "source_repository_url": "https://github.com/dreamquark-ai/tabnet",
                "paper_url": "https://arxiv.org/abs/1908.07442",
                "implementation_style": "official-package-wrapper",
                "input_assumption": "raw_structured_tabular_input",
            }
        )
        return metadata


TabNetClassifier = TabNet

__all__ = ["TabNet", "TabNetClassifier"]
