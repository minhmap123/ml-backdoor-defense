from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch

from .base import BaseTabularModel, ParsedModelInput
from .saint_lib import SAINTAuthorCore


class SAINT(BaseTabularModel):
    """
    Thin wrapper around an author-code-derived SAINT core.

    Research references:
    - Paper: https://arxiv.org/abs/2106.01342
    - Official repo: https://github.com/somepago/saint

    Local research assumptions:
    - This wrapper is intended for supervised SAINT baselines only.
    - The active local pipeline assumes fully observed features after preprocessing.
    - Therefore `cat_mask` and `con_mask` are synthesized as all-ones instead of
      being provided by a dataset-level missingness pipeline as in the author repo.
    """

    def __init__(
        self,
        *,
        d_in: int,
        d_out: int,
        num_numeric_features: Optional[int] = None,
        cat_cardinalities: Optional[Sequence[int]] = None,
        d_token: int = 32,
        n_blocks: int = 1,
        attention_n_heads: int = 4,
        attention_dim_head: int = 16,
        attention_dropout: float = 0.8,
        ff_dropout: float = 0.8,
        cont_embeddings: str = "MLP",
        attention_type: str = "colrow",
        final_mlp_style: str = "sep",
        mlp_hidden_mults: Sequence[int] = (4, 2),
        model_family: str = "saint",
    ) -> None:
        self.num_numeric_features = int(d_in if num_numeric_features is None else num_numeric_features)
        self.cat_cardinalities = [int(x) for x in (cat_cardinalities or [])]
        self.num_categorical_features = len(self.cat_cardinalities)
        self.embedding_dim = int(d_token)
        self.attention_type = str(attention_type)
        self.cont_embeddings = str(cont_embeddings)
        self.final_mlp_style = str(final_mlp_style)

        assert self.num_numeric_features >= 0
        assert all(cardinality > 0 for cardinality in self.cat_cardinalities)
        assert self.cont_embeddings == "MLP"
        assert self.attention_type in {"col", "row", "colrow"}
        assert self.final_mlp_style in {"common", "sep"}

        super().__init__(
            d_in=int(d_in),
            d_out=int(d_out),
            model_family=model_family,
            hidden_dim=int(d_token),
        )

        self.core = SAINTAuthorCore(
            categories=[1, *self.cat_cardinalities],
            num_continuous=self.num_numeric_features,
            dim=self.embedding_dim,
            depth=int(n_blocks),
            heads=int(attention_n_heads),
            dim_head=int(attention_dim_head),
            dim_out=1,
            mlp_hidden_mults=mlp_hidden_mults,
            attn_dropout=float(attention_dropout),
            ff_dropout=float(ff_dropout),
            cont_embeddings=self.cont_embeddings,
            attentiontype=self.attention_type,
            final_mlp_style=self.final_mlp_style,
            y_dim=int(d_out),
        )
        self.head = self.core.mlpfory

    @classmethod
    def get_default_kwargs(cls) -> Dict[str, Any]:
        return {
            "d_token": 32,
            "n_blocks": 1,
            "attention_n_heads": 4,
            "attention_dim_head": 16,
            "attention_dropout": 0.8,
            "ff_dropout": 0.8,
            "cont_embeddings": "MLP",
            "attention_type": "colrow",
            "final_mlp_style": "sep",
        }

    def _extract_structured_input(self, parsed: ParsedModelInput) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
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

        if self.num_numeric_features > 0:
            assert x_num is not None
        if self.num_categorical_features > 0:
            assert x_cat is not None

        return x_num, x_cat

    def _build_author_style_inputs(
        self,
        x_num: Optional[torch.Tensor],
        x_cat: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cls = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        x_categ = cls if x_cat is None else torch.cat([cls, x_cat.long()], dim=1)
        # Local research assumption: SAINT runs on fully observed features.
        # If dataset-level missingness support is added later, replace these
        # synthesized masks with `cat_mask` and `con_mask` from the batch.
        cat_mask = torch.ones_like(x_categ, dtype=torch.long, device=device)
        con_mask = torch.ones(batch_size, self.num_numeric_features, dtype=torch.long, device=device)
        x_cont = (
            torch.empty(batch_size, 0, dtype=torch.float32, device=device)
            if x_num is None
            else x_num.float()
        )
        return x_categ, x_cont, cat_mask, con_mask

    def _forward_features_parsed(self, parsed: ParsedModelInput) -> torch.Tensor:
        x_num, x_cat = self._extract_structured_input(parsed)
        batch_size = (
            x_num.shape[0]
            if x_num is not None
            else (x_cat.shape[0] if x_cat is not None else 1)
        )
        device = (
            x_num.device
            if x_num is not None
            else (x_cat.device if x_cat is not None else self.core.embeds.weight.device)
        )
        x_categ, x_cont, cat_mask, con_mask = self._build_author_style_inputs(
            x_num,
            x_cat,
            batch_size=batch_size,
            device=device,
        )
        reps = self.core.encode(x_categ, x_cont, cat_mask, con_mask, vision_dset=False)
        return reps[:, 0, :]

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
                "embedding_dim": self.embedding_dim,
                "attention_type": self.attention_type,
                "cont_embeddings": self.cont_embeddings,
                "final_mlp_style": self.final_mlp_style,
                "implementation_style": "author-code-derived-core-plus-wrapper",
                "missingness_assumption": "fully_observed_features",
            }
        )
        return metadata


SAINTClassifier = SAINT

__all__ = ["SAINT", "SAINTClassifier"]
