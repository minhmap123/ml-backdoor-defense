from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .base import DatasetSchema, NumericIDSDataset


class IoTID20Dataset(NumericIDSDataset):
    SCALER_TYPE = "QuantileTransformer->MinMaxScaler"
    
    schema = DatasetSchema(
        name="IoTID20",
        raw_path=Path("data/0_raw/IoTID20/IoT Network Intrusion Dataset.csv"),
        target_column="Cat",
        drop_columns=("Flow_ID", "Src_IP", "Dst_IP", "Src_Port", "Dst_Port", "Timestamp"),
        test_size=0.15,
        val_size_within_train=0.15 / (1.0 - 0.15),
    )

    @staticmethod
    def _class_counts(labels: pd.Series | np.ndarray) -> dict[int, int]:
        counts = pd.Series(np.asarray(labels, dtype=np.int64)).value_counts().sort_index()
        return {int(k): int(v) for k, v in counts.items()}

    @staticmethod
    def _make_provenance(split: str, row_ids: np.ndarray, labels: pd.Series | np.ndarray) -> pd.DataFrame:
        labels = np.asarray(labels, dtype=np.int64)
        return pd.DataFrame(
            {
                "split": split,
                "original_row_id": np.asarray(row_ids, dtype=np.int64),
                "train_local_index" if split == "train" else "split_local_index": np.arange(len(labels), dtype=np.int64),
                "source_label": labels,
                "poison_flag": np.zeros(len(labels), dtype=np.int64),
                "target_label": np.full(len(labels), -1, dtype=np.int64),
                "final_label": labels,
            }
        )

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        threshold = 0.99999
        df = df.copy()
        df = df.drop_duplicates().reset_index(drop=True)

        drop_cols = [col for col in self.schema.drop_columns if col in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

        top_value_ratios = {col: df[col].value_counts(normalize=True, dropna=False).iloc[0] for col in df.columns}
        constant_like_cols = [
            col
            for col, ratio in top_value_ratios.items()
            if ratio > threshold and col not in self.schema.protected_label_columns
        ]
        df = df.drop(columns=constant_like_cols, errors="ignore")

        return df

    def prepare_clean_partitions(self) -> dict:
        df = self.load_raw(self.schema.raw_path)
        df = self.clean(df)

        label_columns = [col for col in self.schema.protected_label_columns if col in df.columns]
        assert self.schema.target_column in df.columns, f"Missing target column: {self.schema.target_column}"
        x = df.drop(columns=label_columns, errors="ignore").copy()
        y = df[self.schema.target_column].copy()
        assert x.select_dtypes(include=["object"]).empty, "IoTID20 pipeline expects numeric-only features."

        row_ids = np.arange(len(df), dtype=np.int64)
        x_train_val, x_test_raw, y_train_val, y_test_raw, row_train_val, row_test = train_test_split(
            x,
            y,
            row_ids,
            test_size=self.schema.test_size,
            random_state=self.schema.random_state,
            stratify=y,
        )
        x_train_raw, x_val_raw, y_train_raw, y_val_raw, row_train, row_val = train_test_split(
            x_train_val,
            y_train_val,
            row_train_val,
            test_size=self.schema.val_size_within_train,
            random_state=self.schema.random_state,
            stratify=y_train_val,
        )

        encoder = LabelEncoder()
        y_train = pd.Series(encoder.fit_transform(y_train_raw), dtype=np.int64).reset_index(drop=True)
        y_val = pd.Series(encoder.transform(y_val_raw), dtype=np.int64).reset_index(drop=True)
        y_test = pd.Series(encoder.transform(y_test_raw), dtype=np.int64).reset_index(drop=True)

        # Scale clean data for CatBack surrogate model training
        scaler = self._make_scaler()
        x_train_scaled = scaler.fit_transform(x_train_raw)
        x_val_scaled = scaler.transform(x_val_raw)
        x_test_scaled = scaler.transform(x_test_raw)
        
        clean_datasets = {
            "train": {
                "x": x_train_scaled.astype(np.float32),
                "y": y_train.to_numpy(dtype=np.int64, copy=False),
            },
            "val": {
                "x": x_val_scaled.astype(np.float32),
                "y": y_val.to_numpy(dtype=np.int64, copy=False),
            },
            "test": {
                "x": x_test_scaled.astype(np.float32),
                "y": y_test.to_numpy(dtype=np.int64, copy=False),
            },
            "train_class_weight_labels": y_train.to_numpy(dtype=np.int64, copy=False),
        }

        split_counts = {
            "train": self._class_counts(y_train),
            "val": self._class_counts(y_val),
            "test": self._class_counts(y_test),
        }



        return {
            "encoder": encoder,
            "x_train_raw": x_train_raw.reset_index(drop=True),
            "x_val_raw": x_val_raw.reset_index(drop=True),
            "x_test_raw": x_test_raw.reset_index(drop=True),
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "row_train": np.asarray(row_train, dtype=np.int64),
            "row_val": np.asarray(row_val, dtype=np.int64),
            "row_test": np.asarray(row_test, dtype=np.int64),
            "split_counts": split_counts,
            "clean_datasets": clean_datasets,
        }

    def prepare(self, attack, prepared: dict | None = None) -> tuple[dict, dict, object]:
        prepared = self.prepare_clean_partitions() if prepared is None else prepared

        attack_result = attack.inject(
            clean_features=prepared["x_train_raw"],
            clean_labels=prepared["y_train"].copy(deep=True),
        )

        x_val_triggered_raw = attack.apply_trigger_to_features(prepared["x_val_raw"])
        x_test_triggered_raw = attack.apply_trigger_to_features(prepared["x_test_raw"])

        scaler = self._make_scaler()
        x_train_poisoned = pd.DataFrame(scaler.fit_transform(attack_result.poisoned_features), columns=prepared["x_train_raw"].columns)
        x_train_clean_reference = pd.DataFrame(scaler.transform(prepared["x_train_raw"]), columns=prepared["x_train_raw"].columns)
        x_val = pd.DataFrame(scaler.transform(prepared["x_val_raw"]), columns=prepared["x_val_raw"].columns)
        x_test = pd.DataFrame(scaler.transform(prepared["x_test_raw"]), columns=prepared["x_test_raw"].columns)
        x_val_triggered = pd.DataFrame(scaler.transform(x_val_triggered_raw), columns=x_val_triggered_raw.columns)
        x_test_triggered = pd.DataFrame(scaler.transform(x_test_triggered_raw), columns=x_test_triggered_raw.columns)

        poisoned_labels = attack_result.get_poisoned_labels(prepared["y_train"].copy(deep=True))
        train_provenance = self._make_provenance("train", prepared["row_train"], prepared["y_train"])
        poison_flags = np.zeros(len(train_provenance), dtype=np.int64)
        poison_flags[np.asarray(attack_result.poison_indices, dtype=np.int64)] = 1
        train_provenance["poison_flag"] = poison_flags
        train_provenance["target_label"] = np.where(poison_flags == 1, int(attack.target_label), -1)
        train_provenance["final_label"] = poisoned_labels.to_numpy(dtype=np.int64, copy=False)

        datasets = {
            "train": {
                "x": x_train_poisoned.to_numpy(dtype=np.float32, copy=False),
                "y": poisoned_labels.to_numpy(dtype=np.int64, copy=False),
            },
            "train_clean_reference": {
                "x": x_train_clean_reference.to_numpy(dtype=np.float32, copy=False),
                "y": prepared["y_train"].to_numpy(dtype=np.int64, copy=False),
            },
            "train_class_weight_labels": prepared["y_train"].to_numpy(dtype=np.int64, copy=False),
            "val": {
                "x": x_val.to_numpy(dtype=np.float32, copy=False),
                "y": prepared["y_val"].to_numpy(dtype=np.int64, copy=False),
            },
            "val_triggered": {
                "x": x_val_triggered.to_numpy(dtype=np.float32, copy=False),
                "y": prepared["y_val"].to_numpy(dtype=np.int64, copy=False),
            },
            "val_clean_labels": prepared["y_val"].to_numpy(dtype=np.int64, copy=False),
            "test": {
                "x": x_test.to_numpy(dtype=np.float32, copy=False),
                "y": prepared["y_test"].to_numpy(dtype=np.int64, copy=False),
            },
            "test_triggered": {
                "x": x_test_triggered.to_numpy(dtype=np.float32, copy=False),
                "y": prepared["y_test"].to_numpy(dtype=np.int64, copy=False),
            },
            "test_clean_labels": prepared["y_test"].to_numpy(dtype=np.int64, copy=False),
            "sample_provenance": {
                "train": train_provenance,
                "val": self._make_provenance("val", prepared["row_val"], prepared["y_val"]),
                "test": self._make_provenance("test", prepared["row_test"], prepared["y_test"]),
            },
        }

        minmax = scaler.named_steps["minmax_scaler"]
        metadata = {
            "dataset": self.schema.name,
            "classes": prepared["encoder"].classes_.tolist(),
            "label_mapping": {name: int(i) for i, name in enumerate(prepared["encoder"].classes_)},
            "train_shape": list(x_train_poisoned.shape),
            "val_shape": list(x_val.shape),
            "test_shape": list(x_test.shape),
            "scaler": self.SCALER_TYPE,
            "scaler_min": np.asarray(minmax.data_min_, dtype=np.float32).tolist(),
            "scaler_max": np.asarray(minmax.data_max_, dtype=np.float32).tolist(),
            "attack_injection_stage": "preprocess_before_scaler",
            "imbalance_protocol": "balanced_cross_entropy_from_train_labels",
            "dataset_random_state": int(self.schema.random_state),
            "class_counts": prepared["split_counts"],
            "sample_provenance_saved": True,
        }

        return datasets, metadata, attack_result
