from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


# Author-code-derived SAINT core.
# Provenance:
# - Paper: https://arxiv.org/abs/2106.01342
# - Official repo: https://github.com/somepago/saint
# - Official commit used for local derivation: e288e84c77a54cfd2ffb55a53678fb7cbbb16630


class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_tokens, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.view(batch_size, n_tokens, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch_size, n_tokens, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch_size, n_tokens, self.heads, self.dim_head).transpose(1, 2)
        sim = (q @ k.transpose(-1, -2)) * self.scale
        # Keep the dropout module for structural parity with the author code.
        # The official forward path does not apply it to attention weights.
        attn = sim.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, n_tokens, self.heads * self.dim_head)
        return self.to_out(out)


class RowColTransformer(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        nfeats: int,
        depth: int,
        heads: int,
        dim_head: int,
        attn_dropout: float,
        ff_dropout: float,
        style: str = "col",
    ) -> None:
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        self.mask_embed = nn.Embedding(nfeats, dim)
        self.style = style
        for _ in range(depth):
            if self.style == "colrow":
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                            PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                            PreNorm(dim * nfeats, Residual(Attention(dim * nfeats, heads=heads, dim_head=64, dropout=attn_dropout))),
                            PreNorm(dim * nfeats, Residual(FeedForward(dim * nfeats, dropout=ff_dropout))),
                        ]
                    )
                )
            else:
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(dim * nfeats, Residual(Attention(dim * nfeats, heads=heads, dim_head=64, dropout=attn_dropout))),
                            PreNorm(dim * nfeats, Residual(FeedForward(dim * nfeats, dropout=ff_dropout))),
                        ]
                    )
                )

    def forward(self, x: torch.Tensor, x_cont: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        del mask
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        _, n_tokens, d_token = x.shape
        if self.style == "colrow":
            for attn1, ff1, attn2, ff2 in self.layers:
                x = attn1(x)
                x = ff1(x)
                x = x.reshape(1, x.shape[0], n_tokens * d_token)
                x = attn2(x)
                x = ff2(x)
                x = x.reshape(x.shape[1], n_tokens, d_token)
        else:
            for attn1, ff1 in self.layers:
                x = x.reshape(1, x.shape[0], n_tokens * d_token)
                x = attn1(x)
                x = ff1(x)
                x = x.reshape(x.shape[1], n_tokens, d_token)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        attn_dropout: float,
        ff_dropout: float,
    ) -> None:
        super().__init__()
        del num_tokens
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                        PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, x_cont: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class MLP(nn.Module):
    def __init__(self, dims: Sequence[int], act: Optional[nn.Module] = None) -> None:
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            layers.append(nn.Linear(dim_in, dim_out))
            if is_last:
                continue
            if act is not None:
                layers.append(act)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SimpleMLP(nn.Module):
    def __init__(self, dims: Sequence[int]) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.view(x.size(0), -1)
        return self.layers(x)


class SepMLP(nn.Module):
    def __init__(self, dim: int, len_feats: int, categories: Sequence[int]) -> None:
        super().__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(SimpleMLP([dim, 5 * dim, categories[i]]))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        y_pred = []
        for i in range(self.len_feats):
            pred = self.layers[i](x[:, i, :])
            y_pred.append(pred)
        return y_pred


def embed_data_mask(
    x_categ: torch.Tensor,
    x_cont: torch.Tensor,
    cat_mask: torch.Tensor,
    con_mask: torch.Tensor,
    model: "SAINTAuthorCore",
    vision_dset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = x_cont.device
    x_categ = x_categ + model.categories_offset.type_as(x_categ)
    x_categ_enc = model.embeds(x_categ)
    n1, n2 = x_cont.shape
    if model.cont_embeddings == "MLP":
        x_cont_enc = torch.empty(n1, n2, model.dim)
        for i in range(model.num_continuous):
            x_cont_enc[:, i, :] = model.simple_MLP[i](x_cont[:, i])
    else:
        raise Exception("This case should not work!")

    x_cont_enc = x_cont_enc.to(device)
    cat_mask_temp = cat_mask + model.cat_mask_offset.type_as(cat_mask)
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)
    cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)
    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

    if vision_dset:
        pos = torch.arange(x_categ.shape[-1], device=device).repeat(x_categ.shape[0], 1)
        pos_enc = model.pos_encodings(pos)
        x_categ_enc = x_categ_enc + pos_enc

    return x_categ, x_categ_enc, x_cont_enc


class SAINTAuthorCore(nn.Module):
    def __init__(
        self,
        *,
        categories: Sequence[int],
        num_continuous: int,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int = 16,
        dim_out: int = 1,
        mlp_hidden_mults: Sequence[int] = (4, 2),
        mlp_act: Optional[nn.Module] = None,
        num_special_tokens: int = 0,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        cont_embeddings: str = "MLP",
        scalingfactor: int = 10,
        attentiontype: str = "col",
        final_mlp_style: str = "common",
        y_dim: int = 2,
    ) -> None:
        super().__init__()
        del scalingfactor
        assert all(map(lambda n: n > 0, categories))

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        self.register_buffer("categories_offset", categories_offset)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

        if self.cont_embeddings == "MLP":
            self.simple_MLP = nn.ModuleList([SimpleMLP([1, 100, self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        elif self.cont_embeddings == "pos_singleMLP":
            self.simple_MLP = nn.ModuleList([SimpleMLP([1, 100, self.dim]) for _ in range(1)])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories

        if attentiontype == "col":
            self.transformer = Transformer(
                num_tokens=self.total_tokens,
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
            )
        elif attentiontype in ["row", "colrow"]:
            self.transformer = RowColTransformer(
                num_tokens=self.total_tokens,
                dim=dim,
                nfeats=nfeats,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                style=attentiontype,
            )
        else:
            raise ValueError(f"Unsupported attentiontype: {attentiontype}")

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        self.mlp = MLP(all_dimensions, act=mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim)

        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value=0)
        cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]
        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value=0)
        con_mask_offset = con_mask_offset.cumsum(dim=-1)[:-1]
        self.register_buffer("cat_mask_offset", cat_mask_offset)
        self.register_buffer("con_mask_offset", con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories * 2, self.dim)
        self.mask_embeds_cont = nn.Embedding(max(self.num_continuous * 2, 1), self.dim)
        self.single_mask = nn.Embedding(2, self.dim)
        self.pos_encodings = nn.Embedding(self.num_categories + self.num_continuous, self.dim)

        if self.final_mlp_style == "common":
            self.mlp1 = SimpleMLP([dim, (self.total_tokens) * 2, self.total_tokens])
            self.mlp2 = SimpleMLP([dim, self.num_continuous, 1])
        else:
            self.mlp1 = SepMLP(dim, self.num_categories, categories)
            self.mlp2 = SepMLP(dim, self.num_continuous, torch.ones(self.num_continuous, dtype=torch.int64).tolist())

        self.mlpfory = SimpleMLP([dim, 1000, y_dim])
        self.pt_mlp = SimpleMLP(
            [
                dim * (self.num_continuous + self.num_categories),
                6 * dim * (self.num_continuous + self.num_categories) // 5,
                dim * (self.num_continuous + self.num_categories) // 2,
            ]
        )
        self.pt_mlp2 = SimpleMLP(
            [
                dim * (self.num_continuous + self.num_categories),
                6 * dim * (self.num_continuous + self.num_categories) // 5,
                dim * (self.num_continuous + self.num_categories) // 2,
            ]
        )

    def encode(
        self,
        x_categ: torch.Tensor,
        x_cont: torch.Tensor,
        cat_mask: torch.Tensor,
        con_mask: torch.Tensor,
        *,
        vision_dset: bool = False,
    ) -> torch.Tensor:
        _, x_categ_enc, x_cont_enc = embed_data_mask(
            x_categ,
            x_cont,
            cat_mask,
            con_mask,
            self,
            vision_dset=vision_dset,
        )
        return self.transformer(x_categ_enc, x_cont_enc)


__all__ = [
    "Attention",
    "FeedForward",
    "GEGLU",
    "MLP",
    "PreNorm",
    "Residual",
    "RowColTransformer",
    "SAINTAuthorCore",
    "SepMLP",
    "SimpleMLP",
    "Transformer",
    "embed_data_mask",
]
