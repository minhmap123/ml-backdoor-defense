from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTabularModel, ParsedModelInput


class ReGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.d_token = int(d_token)
        self.weight = nn.Parameter(torch.empty(self.n_features, self.d_token))
        self.bias = nn.Parameter(torch.empty(self.n_features, self.d_token))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.weight.shape[1] ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)
        nn.init.uniform_(self.bias, -d_rsqrt, d_rsqrt)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        return x_num[..., None] * self.weight[None] + self.bias[None]


class CategoricalFeatureTokenizer(nn.Module):
    def __init__(self, cardinalities: Sequence[int], d_token: int) -> None:
        super().__init__()
        self.cardinalities = [int(x) for x in cardinalities]
        self.d_token = int(d_token)
        total = int(sum(self.cardinalities))
        self.embeddings = nn.Embedding(total, self.d_token) if total > 0 else None
        self.bias = nn.Parameter(torch.empty(len(self.cardinalities), self.d_token))
        offsets = torch.tensor([0] + self.cardinalities[:-1], dtype=torch.long).cumsum(0)
        self.register_buffer("offsets", offsets, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.d_token ** -0.5
        if self.embeddings is not None:
            nn.init.uniform_(self.embeddings.weight, -d_rsqrt, d_rsqrt)
        nn.init.uniform_(self.bias, -d_rsqrt, d_rsqrt)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        return self.embeddings(x_cat + self.offsets[None]) + self.bias[None]


class CLSToken(nn.Module):
    def __init__(self, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_token))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.weight.shape[-1] ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls = self.weight.view(1, 1, -1).expand(x.shape[0], 1, -1)
        return torch.cat([cls, x], dim=1)


class MultiheadAttention(nn.Module):
    def __init__(self, d_token: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_token % n_heads == 0

        self.d_token = int(d_token)
        self.n_heads = int(n_heads)
        self.d_head = self.d_token // self.n_heads
        self.scale = self.d_head**-0.5
        self.dropout = nn.Dropout(float(dropout))
        self.W_q = nn.Linear(self.d_token, self.d_token)
        self.W_k = nn.Linear(self.d_token, self.d_token)
        self.W_v = nn.Linear(self.d_token, self.d_token)
        self.W_out = nn.Linear(self.d_token, self.d_token) if self.n_heads > 1 else None

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_tokens, _ = x.shape
        return x.view(batch_size, n_tokens, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        q = self._reshape(self.W_q(x_q))
        k = self._reshape(self.W_k(x_kv))
        v = self._reshape(self.W_v(x_kv))

        attention_logits = q @ k.transpose(-1, -2) * self.scale
        attention_probs = self.dropout(attention_logits.softmax(dim=-1))
        x = attention_probs @ v
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], self.d_token)
        return x if self.W_out is None else self.W_out(x)


class FTTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        d_token: int,
        attention_n_heads: int,
        attention_dropout: float,
        ffn_d_hidden: int,
        ffn_dropout: float,
        residual_dropout: float,
        first_prenormalization: bool = False,
        last_layer_query_idx: Optional[slice] = None,
    ) -> None:
        super().__init__()
        self.last_layer_query_idx = last_layer_query_idx
        self.attention_norm = nn.Identity() if first_prenormalization else nn.LayerNorm(d_token)
        self.ffn_norm = nn.LayerNorm(d_token)
        self.attention = MultiheadAttention(d_token, attention_n_heads, attention_dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, 2 * ffn_d_hidden),
            ReGLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_d_hidden, d_token),
        )
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_residual = self.attention_norm(x)
        x_query = x_residual[:, self.last_layer_query_idx] if self.last_layer_query_idx is not None else x_residual
        x_attention = self.attention(x_query, x_residual)
        x_query_base = x[:, self.last_layer_query_idx] if self.last_layer_query_idx is not None else x
        x = x_query_base + self.residual_dropout(x_attention)
        x = x + self.residual_dropout(self.ffn(self.ffn_norm(x)))
        return x


class FTTransformer(BaseTabularModel):
    """FT-Transformer derived from the official RTDL implementation.

    Paper: https://arxiv.org/abs/2106.11959
    Official code: https://github.com/yandex-research/rtdl-revisiting-models

    This implementation keeps the feature tokenizer + transformer + [CLS] head
    structure close to RTDL while adapting the public API to this repository's
    BaseTabularModel contract.
    """

    def __init__(
        self,
        *,
        d_in: int,
        d_out: int,
        num_numeric_features: Optional[int] = None,
        cat_cardinalities: Optional[Sequence[int]] = None,
        n_blocks: int = 3,
        d_token: int = 192,
        attention_n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_d_hidden: Optional[int] = None,
        ffn_d_hidden_multiplier: float = 4.0 / 3.0,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        model_family: str = "ft_transformer",
    ) -> None:
        self.num_numeric_features = int(d_in if num_numeric_features is None else num_numeric_features)
        self.cat_cardinalities = [int(x) for x in (cat_cardinalities or [])]
        self.num_categorical_features = len(self.cat_cardinalities)
        self.embedding_dim = int(d_token)
        assert self.num_numeric_features >= 0
        assert all(cardinality > 0 for cardinality in self.cat_cardinalities)
        assert self.num_numeric_features > 0 or self.cat_cardinalities

        super().__init__(
            d_in=int(d_in),
            d_out=int(d_out),
            model_family=model_family,
            hidden_dim=int(d_token),
        )

        ffn_d_hidden = (
            int(self.embedding_dim * float(ffn_d_hidden_multiplier))
            if ffn_d_hidden is None
            else int(ffn_d_hidden)
        )

        self.numeric_tokenizer = (
            NumericalFeatureTokenizer(self.num_numeric_features, self.embedding_dim)
            if self.num_numeric_features > 0
            else None
        )
        self.categorical_tokenizer = (
            CategoricalFeatureTokenizer(self.cat_cardinalities, self.embedding_dim)
            if self.cat_cardinalities
            else None
        )
        self.cls_token = CLSToken(self.embedding_dim)
        self.blocks = nn.ModuleList(
            [
                FTTransformerBlock(
                    d_token=self.embedding_dim,
                    attention_n_heads=int(attention_n_heads),
                    attention_dropout=float(attention_dropout),
                    ffn_d_hidden=ffn_d_hidden,
                    ffn_dropout=float(ffn_dropout),
                    residual_dropout=float(residual_dropout),
                    first_prenormalization=i == 0,
                    last_layer_query_idx=slice(0, 1) if i == int(n_blocks) - 1 else None,
                )
                for i in range(int(n_blocks))
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, int(d_out)),
        )

    @classmethod
    def get_default_kwargs(cls, n_blocks: int = 3) -> Dict[str, Any]:
        assert 1 <= int(n_blocks) <= 6
        return {
            "n_blocks": int(n_blocks),
            "d_token": [96, 128, 192, 256, 320, 384][int(n_blocks) - 1],
            "attention_n_heads": 8,
            "attention_dropout": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35][int(n_blocks) - 1],
            "ffn_d_hidden_multiplier": 4.0 / 3.0,
            "ffn_dropout": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25][int(n_blocks) - 1],
            "residual_dropout": 0.0,
        }

    def reset_parameters(self) -> None:
        for module in self.modules():
            if module is self:
                continue
            if isinstance(
                module,
                (
                    NumericalFeatureTokenizer,
                    CategoricalFeatureTokenizer,
                    CLSToken,
                    nn.Linear,
                    nn.LayerNorm,
                ),
            ):
                module.reset_parameters()

    def _extract_structured_input(
        self, parsed: ParsedModelInput
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        x_num = parsed.x_num
        x_cat = parsed.x_cat

        if parsed.x is not None:
            x_flat = parsed.x.float()
            if x_flat.ndim == 1:
                x_flat = x_flat.unsqueeze(0)
            assert self.num_categorical_features == 0
            x_num = x_flat

        if x_num is not None:
            x_num = x_num.float()
            if x_num.ndim == 1:
                x_num = x_num.unsqueeze(0)

        if x_cat is not None:
            x_cat = x_cat.long()
            if x_cat.ndim == 1:
                x_cat = x_cat.unsqueeze(0)

        return x_num, x_cat

    def _tokenize(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        tokens = []
        if self.numeric_tokenizer is not None and x_num is not None:
            tokens.append(self.numeric_tokenizer(x_num))
        if self.categorical_tokenizer is not None and x_cat is not None:
            tokens.append(self.categorical_tokenizer(x_cat))
        x = torch.cat(tokens, dim=1)
        return self.cls_token(x)

    def _forward_features_parsed(self, parsed: ParsedModelInput) -> torch.Tensor:
        x_num, x_cat = self._extract_structured_input(parsed)
        x = self._tokenize(x_num, x_cat)
        for block in self.blocks[:-1]:
            x = block(x)
        x = self.blocks[-1](x)
        if x.ndim == 3:
            x = x[:, 0]
        return x

    def make_parameter_groups(self, weight_decay: float = 0.0) -> list[dict[str, Any]]:
        def get_parameters(module: Optional[nn.Module]) -> Iterable[torch.nn.Parameter]:
            return () if module is None else module.parameters()

        zero_wd = set()
        zero_wd.update(get_parameters(self.cls_token))
        zero_wd.update(get_parameters(self.numeric_tokenizer))
        zero_wd.update(get_parameters(self.categorical_tokenizer))
        zero_wd.update(
            parameter
            for block in self.blocks
            for name, module in block.named_children()
            if name.endswith("norm")
            for parameter in module.parameters()
        )
        zero_wd.update(
            parameter
            for name, parameter in self.named_parameters()
            if name.endswith(".bias") or name == "head.2.bias"
        )
        all_parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        decay = [parameter for parameter in all_parameters if parameter not in zero_wd]
        no_decay = [parameter for parameter in all_parameters if parameter in zero_wd]
        return [
            {"params": decay, "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def make_default_optimizer(self) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            self.make_parameter_groups(weight_decay=1e-5),
            lr=1e-4,
            weight_decay=0.0,
        )

    def get_model_metadata(self) -> Dict[str, Any]:
        metadata = super().get_model_metadata()
        metadata.update(
            {
                "num_numeric_features": self.num_numeric_features,
                "num_categorical_features": self.num_categorical_features,
                "categorical_cardinalities": list(self.cat_cardinalities),
                "embedding_dim": self.embedding_dim,
                "source_reference": "yandex-research/rtdl-revisiting-models",
                "source_repository_url": "https://github.com/yandex-research/rtdl-revisiting-models",
                "paper_url": "https://arxiv.org/abs/2106.11959",
                "implementation_style": "official-code-derived-wrapper",
            }
        )
        return metadata


FTTransformerClassifier = FTTransformer

__all__ = ["FTTransformer", "FTTransformerClassifier"]
