from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _to_torch(value: Any, *, dtype: torch.dtype, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype, device=device)
    return torch.tensor(np.array(value, copy=True), dtype=dtype, device=device)


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: Optional[str]) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def split_to_numpy(split: Any) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(split, (tuple, list)):
        x = np.asarray(split[0], dtype=np.float32)
        y = np.asarray(split[1], dtype=np.int64)
        return x, y

    if isinstance(split, dict):
        y = np.asarray(split["y"], dtype=np.int64)
        if "x" in split:
            x = np.asarray(split["x"], dtype=np.float32)
            return x, y
        x_num = np.asarray(split.get("x_num"), dtype=np.float32)
        x_cat = split.get("x_cat")
        if x_cat is None:
            return x_num, y
        x_cat = np.asarray(x_cat, dtype=np.float32)
        if x_cat.ndim == 1:
            x_cat = x_cat[:, None]
        return np.concatenate([x_num, x_cat], axis=1), y

    raise ValueError(f"Unsupported split type: {type(split)}")


def split_to_dataloader(split: Any, *, batch_size: int, shuffle: bool, drop_last: bool = False) -> DataLoader:
    if isinstance(split, dict):
        y = _to_torch(split["y"], dtype=torch.long)
        if "x" in split:
            x = _to_torch(split["x"], dtype=torch.float32)
            dataset = TensorDataset(x, y)
            return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), drop_last=bool(drop_last))

        x_num = _to_torch(split.get("x_num"), dtype=torch.float32)
        x_cat_raw = split.get("x_cat")
        if x_cat_raw is None:
            dataset = TensorDataset(x_num, y)
            return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), drop_last=bool(drop_last))

        x_cat = _to_torch(x_cat_raw, dtype=torch.long)
        dataset = TensorDataset(x_num, x_cat, y)
        return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), drop_last=bool(drop_last))

    if isinstance(split, (tuple, list)):
        x = _to_torch(split[0], dtype=torch.float32)
        y = _to_torch(split[1], dtype=torch.long)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), drop_last=bool(drop_last))

    raise ValueError(f"Unsupported split type: {type(split)}")


def parse_torch_batch(batch: Any, device: torch.device) -> Tuple[Any, torch.Tensor]:
    if isinstance(batch, dict):
        y = _to_torch(batch["y"], dtype=torch.long, device=device)
        if "x" in batch:
            x = _to_torch(batch["x"], dtype=torch.float32, device=device)
            return x, y
        x_num = _to_torch(batch["x_num"], dtype=torch.float32, device=device)
        x_cat = batch.get("x_cat")
        if x_cat is None:
            return {"x_num": x_num}, y
        x_cat = _to_torch(x_cat, dtype=torch.long, device=device)
        return {"x_num": x_num, "x_cat": x_cat}, y

    if isinstance(batch, (tuple, list)):
        if len(batch) == 3:
            x_num = _to_torch(batch[0], dtype=torch.float32, device=device)
            x_cat = _to_torch(batch[1], dtype=torch.long, device=device)
            y = _to_torch(batch[2], dtype=torch.long, device=device)
            return {"x_num": x_num, "x_cat": x_cat}, y
        x = _to_torch(batch[0], dtype=torch.float32, device=device)
        y = _to_torch(batch[1], dtype=torch.long, device=device)
        return x, y

    raise ValueError(f"Unsupported batch type: {type(batch)}")


def save_model(
    model: torch.nn.Module,
    output_dir: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out / "model_state_dict.pt")

    if config is not None:
        with (out / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)

    if metadata is not None:
        with (out / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    if metrics is not None:
        with (out / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)

    if optimizer is not None:
        torch.save(optimizer.state_dict(), out / "optimizer.pt")

    return str(out)


def load_model(model_class: type, checkpoint_dir: str, *, strict: bool = True) -> torch.nn.Module:
    ckpt = Path(checkpoint_dir)

    model_config: Dict[str, Any] = {}
    config_path = ckpt / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            model_config = loaded.get("model_kwargs", loaded)
    if isinstance(model_config, dict) and "name" in model_config:
        model_config = {k: v for k, v in model_config.items() if k != "name"}

    model = model_class(**model_config)
    state_dict = torch.load(ckpt / "model_state_dict.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=bool(strict))
    return model
