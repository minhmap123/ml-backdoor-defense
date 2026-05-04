from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ParsedModelInput:
    x: Optional[torch.Tensor] = None
    x_num: Optional[torch.Tensor] = None
    x_cat: Optional[torch.Tensor] = None


class BaseTabularModel(nn.Module, ABC):
    """Base contract for tabular models used by train/detect/unlearn pipelines."""

    def __init__(
        self,
        *,
        d_in: int,
        d_out: int,
        model_family: str,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.model_family = str(model_family)
        self.hidden_dim = None if hidden_dim is None else int(hidden_dim)

    @staticmethod
    def _to_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype)
        return torch.tensor(np.array(value, copy=True), dtype=dtype)

    @staticmethod
    def parse_input(x: Any) -> ParsedModelInput:
        """Normalize accepted input formats into a unified parsed object."""
        if isinstance(x, ParsedModelInput):
            return x

        if isinstance(x, dict):
            if "x" in x:
                return ParsedModelInput(x=BaseTabularModel._to_tensor(x["x"], dtype=torch.float32))
            x_num = x.get("x_num")
            x_cat = x.get("x_cat")
            return ParsedModelInput(
                x_num=None if x_num is None else BaseTabularModel._to_tensor(x_num, dtype=torch.float32),
                x_cat=None if x_cat is None else BaseTabularModel._to_tensor(x_cat, dtype=torch.long),
            )

        if isinstance(x, (tuple, list)):
            if len(x) == 2:
                first = BaseTabularModel._to_tensor(x[0], dtype=torch.float32)
                second = BaseTabularModel._to_tensor(x[1], dtype=torch.long)
                # Dataloader format (x, y): ignore y for model input parsing
                if first.ndim >= 2 and second.ndim == 1:
                    return ParsedModelInput(x=first)
                # Mixed tabular format (x_num, x_cat)
                return ParsedModelInput(x_num=first, x_cat=second)
            if len(x) == 1:
                return ParsedModelInput(x=BaseTabularModel._to_tensor(x[0], dtype=torch.float32))

        return ParsedModelInput(x=BaseTabularModel._to_tensor(x, dtype=torch.float32))

    @staticmethod
    def _flatten_numeric_and_categorical(parsed: ParsedModelInput) -> torch.Tensor:
        """Helper for flat-input models that concatenate numeric and categorical parts."""
        if parsed.x is not None:
            x = parsed.x
            if x.ndim == 1:
                x = x.unsqueeze(0)
            if x.ndim > 2:
                x = x.view(x.size(0), -1)
            return x.float()

        parts = []
        if parsed.x_num is not None:
            x_num = parsed.x_num
            if x_num.ndim == 1:
                x_num = x_num.unsqueeze(0)
            if x_num.ndim > 2:
                x_num = x_num.view(x_num.size(0), -1)
            parts.append(x_num.float())

        if parsed.x_cat is not None:
            x_cat = parsed.x_cat
            if x_cat.ndim == 1:
                x_cat = x_cat.unsqueeze(0)
            if x_cat.ndim > 2:
                x_cat = x_cat.view(x_cat.size(0), -1)
            parts.append(x_cat.float())

        return torch.cat(parts, dim=1)

    @staticmethod
    def normalize_logits(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 1:
            return logits.unsqueeze(1)
        return logits

    @abstractmethod
    def _forward_features_parsed(self, parsed: ParsedModelInput) -> torch.Tensor:
        raise NotImplementedError

    def _forward_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward_features(self, x: Any) -> torch.Tensor:
        parsed = self.parse_input(x)
        return self._forward_features_parsed(parsed)

    def forward_logits(self, features: torch.Tensor) -> torch.Tensor:
        logits = self._forward_logits_from_features(features)
        return self.normalize_logits(logits)

    def forward(self, x: Any) -> torch.Tensor:
        features = self.forward_features(x)
        return self.forward_logits(features)

    def get_model_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "model_family": self.model_family,
            "d_in": self.d_in,
            "d_out": self.d_out,
            "hidden_dim": self.hidden_dim,
            "num_parameters": int(sum(p.numel() for p in self.parameters())),
        }
