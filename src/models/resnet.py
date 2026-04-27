from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn

from .base import BaseTabularModel, ParsedModelInput


class ResNet(BaseTabularModel):
    """ResNet for tabular data from RTDL (Revisiting Deep Learning Models for Tabular Data).
    
    Paper: https://arxiv.org/abs/2106.11959
    Official code: https://github.com/yandex-research/rtdl-revisiting-models
    
    Architecture: Linear projection -> Residual blocks -> Output head.
    Each block: BN -> Linear -> ReLU -> Dropout -> Linear -> Dropout + Residual skip connection.
    """

    def __init__(
        self,
        *,
        d_in: int,
        d_out: int,
        n_blocks: int = 2,
        d_block: int = 192,
        d_hidden: Optional[int] = None,
        d_hidden_multiplier: float = 2.0,
        dropout1: float = 0.15,
        dropout2: float = 0.0,
        model_family: str = "resnet",
    ) -> None:
        d_hidden = int(d_block * d_hidden_multiplier) if d_hidden is None else int(d_hidden)

        super().__init__(
            d_in=int(d_in),
            d_out=int(d_out),
            model_family=model_family,
            hidden_dim=int(d_block),
        )

        self.input_projection = nn.Linear(int(d_in), int(d_block))

        self.blocks = nn.ModuleList()
        for _ in range(int(n_blocks)):
            block = nn.Sequential(
                nn.BatchNorm1d(int(d_block)),
                nn.ReLU(inplace=True),
                nn.Linear(int(d_block), int(d_hidden)),
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout1)),
                nn.Linear(int(d_hidden), int(d_block)),
                nn.Dropout(float(dropout2)),
            )
            self.blocks.append(block)

        self.output = nn.Sequential(
            nn.BatchNorm1d(int(d_block)),
            nn.ReLU(inplace=True),
            nn.Linear(int(d_block), int(d_out)),
        )

    def _forward_features_parsed(self, parsed: ParsedModelInput) -> torch.Tensor:
        x = self._flatten_numeric_and_categorical(parsed)
        x = self.input_projection(x)

        # Apply residual blocks
        for block in self.blocks:
            x = x + block(x)

        return x

    def forward_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Override to apply output head to features."""
        logits = self.output(features)
        return self.normalize_logits(logits)


ResNetClassifier = ResNet

__all__ = ["ResNet", "ResNetClassifier"]
