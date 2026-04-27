from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import BaseTabularModel, ParsedModelInput


class MLP(BaseTabularModel):
    """MLP from RTDL (Revisiting Deep Learning Models for Tabular Data).
    
    Paper: https://arxiv.org/abs/2106.11959
    Official code: https://github.com/yandex-research/rtdl-revisiting-models
    
    Architecture: Linear-BN-ReLU blocks stacked, final linear head.
    """

    def __init__(
        self,
        *,
        d_in: int,
        d_out: int,
        n_blocks: int = 2,
        d_block: int = 256,
        dropout: float = 0.0,
        model_family: str = "mlp",
    ) -> None:
        super().__init__(
            d_in=int(d_in),
            d_out=int(d_out),
            model_family=model_family,
            hidden_dim=int(d_block),
        )

        blocks = []
        current = int(d_in)
        for _ in range(int(n_blocks)):
            blocks.append(
                nn.Sequential(
                    nn.Linear(current, int(d_block)),
                    nn.BatchNorm1d(int(d_block)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(float(dropout)),
                )
            )
            current = int(d_block)

        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(current, int(d_out))

    def _forward_features_parsed(self, parsed: ParsedModelInput) -> torch.Tensor:
        x = self._flatten_numeric_and_categorical(parsed)
        return self.backbone(x)


MLPClassifier = MLP

__all__ = ["MLP", "MLPClassifier"]
