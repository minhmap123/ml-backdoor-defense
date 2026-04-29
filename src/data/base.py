from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    raw_path: Path
    target_column: str
    drop_columns: tuple[str, ...] = ()
    protected_label_columns: tuple[str, ...] = ("Label", "Cat", "Sub_Cat")
    near_constant_threshold: float = 0.999
    test_size: float = 0.2
    val_size_within_train: float = 0.2
    random_state: int = 42


class NumericIDSDataset(ABC):
    schema: DatasetSchema

    def __init__(self, schema: DatasetSchema | None = None) -> None:
        if schema is not None:
            self.schema = schema
        elif not hasattr(self, "schema"):
            raise ValueError("NumericIDSDataset subclasses must define a schema")

    def load_raw(self, data_path: str | Path | None = None) -> pd.DataFrame:
        path = Path(data_path or self.schema.raw_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        return df

    def clean(self, df: pd.DataFrame, *, near_constant_threshold: float | None = None) -> tuple[pd.DataFrame, dict]:
        schema = self.schema if near_constant_threshold is None else replace(self.schema, near_constant_threshold=near_constant_threshold)
        df = df.copy().drop_duplicates().reset_index(drop=True)
        drop_cols = [c for c in schema.drop_columns if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

        top_ratio = pd.Series({c: df[c].value_counts(normalize=True, dropna=False).iloc[0] for c in df.columns})
        protected = set(schema.protected_label_columns)
        constant_like = [c for c in top_ratio[top_ratio > schema.near_constant_threshold].index.tolist() if c not in protected]
        df = df.drop(columns=constant_like, errors="ignore")

        return df, {
            "drop_columns": drop_cols,
            "dropped_constant_like_columns": constant_like,
            "clean_shape": list(df.shape),
        }

    @abstractmethod
    def prepare(self, *args, **kwargs):
        """Prepare a dataset in its dataset-specific way."""
        raise NotImplementedError

    def export_prepared(self, prepared: dict, *, output_dir: str | Path | None = None, target_column: str | None = None) -> Path:
        out = Path(output_dir or self.schema.raw_path.parents[2] / "1_processed" / self.schema.name)
        out.mkdir(parents=True, exist_ok=True)
        label = target_column or self.schema.target_column
        encoded = f"{label}_encoded"

        for split in ("train", "val", "test"):
            prepared[split]["x"].to_csv(out / f"X_{split}.csv", index=False)
            pd.DataFrame({encoded: prepared[split]["y"]}).to_csv(out / f"y_{split}.csv", index=False)

        with (out / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(prepared["metadata"], f, indent=2, ensure_ascii=True)
        return out

    def load_processed(self, *, data_root: str | Path | None = None, label_column: str | None = None) -> dict:
        root = Path(data_root or self.schema.raw_path.parents[2] / "1_processed" / self.schema.name)
        label = label_column or f"{self.schema.target_column}_encoded"
        splits = {}
        for split in ("train", "val", "test"):
            x = pd.read_csv(root / f"X_{split}.csv")
            y = pd.read_csv(root / f"y_{split}.csv")
            if label not in y.columns:
                raise ValueError(f"label_column '{label}' not found in y_{split}.csv")
            splits[split] = {"x": x.to_numpy(dtype=np.float32, copy=False), "y": y[label].to_numpy(dtype=np.int64, copy=False)}
        return splits
