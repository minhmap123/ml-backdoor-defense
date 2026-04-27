from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ...models import get_model
from ...models.utils import parse_torch_batch
from ..utils import count_split_samples


def _build_membership_sample(split: Any, index: int, membership_label: int) -> Dict[str, Any]:
    label = np.int64(membership_label)

    if isinstance(split, dict):
        sample: Dict[str, Any] = {"y": label}
        if "x" in split:
            sample["x"] = np.asarray(split["x"])[index]
            return sample

        sample["x_num"] = np.asarray(split["x_num"])[index]
        x_cat = split.get("x_cat")
        if x_cat is not None:
            sample["x_cat"] = np.asarray(x_cat)[index]
        return sample

    if isinstance(split, (tuple, list)):
        return {"x": np.asarray(split[0])[index], "y": label}

    raise ValueError(f"Unsupported split type: {type(split)}")


class BadTeachingDataset(torch.utils.data.Dataset):
    def __init__(self, forget_split: Any, retain_split: Any) -> None:
        super().__init__()
        self.forget_split = forget_split
        self.retain_split = retain_split
        self.forget_len = count_split_samples(forget_split)
        self.retain_len = count_split_samples(retain_split)

    def __len__(self) -> int:
        return int(self.forget_len + self.retain_len)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < self.forget_len:
            return _build_membership_sample(self.forget_split, index, membership_label=1)
        return _build_membership_sample(self.retain_split, index - self.forget_len, membership_label=0)


def build_bad_teaching_dataset(forget_split: Any, retain_split: Any) -> BadTeachingDataset:
    return BadTeachingDataset(forget_split=forget_split, retain_split=retain_split)


def parse_membership_batch(batch: Any, device: torch.device) -> Tuple[Any, torch.Tensor]:
    return parse_torch_batch(batch, device)


def clone_model(model_cfg: Dict[str, Any], source_model: Optional[nn.Module] = None) -> nn.Module:
    cloned = get_model(dict(model_cfg))
    if source_model is not None:
        cloned.load_state_dict(source_model.state_dict())
    return cloned


def freeze_model(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model
