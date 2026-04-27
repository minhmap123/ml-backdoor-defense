from __future__ import annotations

import json
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


class NumericIDSDataset:
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

    def prepare(
        self,
        *,
        data_path: str | Path | None = None,
        target_column: str | None = None,
        test_size: float | None = None,
        val_size_within_train: float | None = None,
        random_state: int | None = None,
    ) -> dict:
        schema = replace(
            self.schema,
            raw_path=Path(data_path) if data_path is not None else self.schema.raw_path,
            target_column=target_column or self.schema.target_column,
            test_size=self.schema.test_size if test_size is None else test_size,
            val_size_within_train=self.schema.val_size_within_train if val_size_within_train is None else val_size_within_train,
            random_state=self.schema.random_state if random_state is None else random_state,
        )
        df, cleaning = self.clean(self.load_raw(schema.raw_path), near_constant_threshold=schema.near_constant_threshold)

        labels = [c for c in schema.protected_label_columns if c in df.columns]
        if schema.target_column not in df.columns:
            raise ValueError(f"target_column '{schema.target_column}' not found in cleaned data")

        y = df[schema.target_column].copy()
        X = df.drop(columns=labels, errors="ignore").copy()
        if X.select_dtypes(include=["object"]).shape[1]:
            raise ValueError("Expected numeric-only features")

        x_train_val, x_test, y_train_val, y_test = train_test_split(X, y, test_size=schema.test_size, random_state=schema.random_state, stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val,
            y_train_val,
            test_size=schema.val_size_within_train,
            random_state=schema.random_state,
            stratify=y_train_val,
        )

        scaler = StandardScaler()
        x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
        x_val = pd.DataFrame(scaler.transform(x_val), columns=x_val.columns)
        x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_val = encoder.transform(y_val)
        y_test = encoder.transform(y_test.reset_index(drop=True))

        metadata = {
            "dataset": schema.name,
            "source_path": str(schema.raw_path),
            "target_column": schema.target_column,
            "classes": encoder.classes_.tolist(),
            "label_mapping": {name: int(i) for i, name in enumerate(encoder.classes_)},
            "num_features": int(x_train.shape[1]),
            "train_shape": list(x_train.shape),
            "val_shape": list(x_val.shape),
            "test_shape": list(x_test.shape),
            "scaler": "StandardScaler",
            "cleaning": cleaning,
        }

        return {"train": {"x": x_train, "y": y_train}, "val": {"x": x_val, "y": y_val}, "test": {"x": x_test, "y": y_test}, "metadata": metadata}

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
