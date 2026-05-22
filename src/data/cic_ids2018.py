from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .base import DatasetSchema, NumericIDSDataset

LABELS_TO_DROP = {
    "Infilteration",            # label corruption — Liu et al. 2022 IEEE CNS
    "Brute Force -Web",         # 570 samples → F1=0.04
    "Brute Force -XSS",         # 229 samples → F1=0.16
    "DoS attacks-SlowHTTPTest", #  55 samples → F1=0.30
    "FTP-BruteForce",           #  53 samples → F1=0.00
    "SQL Injection",            #  85 samples → F1=0.01
    "Label",                    # header artifact
}

CLASS_LIST = [
    "Benign",
    "Bot",
    "DDOS attack-HOIC",
    "DDOS attack-LOIC-UDP",
    "DDoS attacks-LOIC-HTTP",
    "DoS attacks-GoldenEye",
    "DoS attacks-Hulk",
    "DoS attacks-Slowloris",
    "SSH-Bruteforce",
]

BENIGN_CAP          = 950_000
NEAR_CONSTANT_THRESH = 0.99999
CORRELATION_THRESH   = 0.99


class CSECICIDS2018Dataset(NumericIDSDataset):
    SCALER_TYPE = "QuantileTransformer->MinMaxScaler"

    schema = DatasetSchema(
        name="CSE-CIC-IDS2018",
        raw_path=Path("data/0_raw/CSE-CIC-IDS2018"),
        target_column="Label",
        drop_columns=("Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp"),
        test_size=0.15,
        val_size_within_train=0.15 / (1.0 - 0.15),
    )

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        df = df.drop(columns=list(self.schema.drop_columns), errors="ignore").drop_duplicates()
        for col in df.columns:
            if col != self.schema.target_column:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        df[self.schema.target_column] = df[self.schema.target_column].astype(str).str.strip()
        return df[~df[self.schema.target_column].isin(LABELS_TO_DROP)].reset_index(drop=True)

    def _load_all(self) -> pd.DataFrame:
        csvs = sorted(Path(self.schema.raw_path).glob("*.csv"))
        return pd.concat(
            [self.clean(pd.read_csv(p, low_memory=False)) for p in csvs],
            ignore_index=True,
        )

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        # Step 1: near-constant columns
        drop1 = [c for c in X if X[c].value_counts(normalize=True, dropna=False).iloc[0] > NEAR_CONSTANT_THRESH]
        X = X.drop(columns=drop1)
        # Step 2: duplicate-value columns (hash comparison)
        seen, drop2 = {}, []
        for col, h in {c: pd.util.hash_pandas_object(X[c]).sum() for c in X}.items():
            if h in seen and X[col].equals(X[seen[h]]):
                drop2.append(col)
            else:
                seen[h] = col
        X = X.drop(columns=drop2)
        # Step 3: high Pearson correlation (greedy upper-triangle)
        upper = X.corr(numeric_only=True).abs().where(
            np.triu(np.ones((X.shape[1],) * 2), k=1).astype(bool)
        )
        drop3 = [c for c in upper if (upper[c] > CORRELATION_THRESH).any()]
        return X.drop(columns=drop3)

    @staticmethod
    def _class_counts(labels: pd.Series | np.ndarray) -> dict[int, int]:
        counts = pd.Series(np.asarray(labels, dtype=np.int64)).value_counts().sort_index()
        return {int(k): int(v) for k, v in counts.items()}

    def prepare_clean_partitions(self) -> dict:
        df = self._load_all()
        X_raw = self._select_features(df.drop(columns=[self.schema.target_column]))
        y = df[self.schema.target_column]
        feature_names = list(X_raw.columns)

        row_ids = np.arange(len(df), dtype=np.int64)
        X_tv, X_test, y_tv, y_test, row_tv, row_test = train_test_split(
            X_raw, y, row_ids,
            test_size=self.schema.test_size,
            stratify=y,
            random_state=self.schema.random_state,
        )
        X_train, X_val, y_train, y_val, row_train, row_val = train_test_split(
            X_tv, y_tv, row_tv,
            test_size=self.schema.val_size_within_train,
            stratify=y_tv,
            random_state=self.schema.random_state,
        )

        # Cap Benign only in train — val/test keep natural distribution
        n_cap = min(BENIGN_CAP, (y_train == "Benign").sum())
        keep = (
            y_train[y_train == "Benign"].sample(n=n_cap, random_state=self.schema.random_state).index
            .union(y_train[y_train != "Benign"].index)
        )
        X_train = X_train.loc[keep].sample(frac=1, random_state=self.schema.random_state)
        y_train = y_train.loc[X_train.index]
        row_train = X_train.index.to_numpy(dtype=np.int64)

        encoder = LabelEncoder()
        encoder.fit(CLASS_LIST)
        y_train_enc = pd.Series(encoder.transform(y_train), dtype=np.int64).reset_index(drop=True)
        y_val_enc   = pd.Series(encoder.transform(y_val),   dtype=np.int64).reset_index(drop=True)
        y_test_enc  = pd.Series(encoder.transform(y_test),  dtype=np.int64).reset_index(drop=True)

        scaler = self._make_scaler()
        x_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train).astype(np.float32), columns=feature_names
        ).reset_index(drop=True)
        x_val_scaled = pd.DataFrame(
            scaler.transform(X_val).astype(np.float32), columns=feature_names
        ).reset_index(drop=True)
        x_test_scaled = pd.DataFrame(
            scaler.transform(X_test).astype(np.float32), columns=feature_names
        ).reset_index(drop=True)

        split_counts = {
            "train": self._class_counts(y_train_enc),
            "val":   self._class_counts(y_val_enc),
            "test":  self._class_counts(y_test_enc),
        }

        return {
            "encoder":        encoder,
            "x_train_raw":    X_train.reset_index(drop=True),
            "x_val_raw":      X_val.reset_index(drop=True),
            "x_test_raw":     X_test.reset_index(drop=True),
            "x_train_scaled": x_train_scaled,
            "x_val_scaled":   x_val_scaled,
            "x_test_scaled":  x_test_scaled,
            "scaler":         scaler,
            "feature_names":  feature_names,
            "y_train":        y_train_enc,
            "y_val":          y_val_enc,
            "y_test":         y_test_enc,
            "row_train":      row_train,
            "row_val":        np.asarray(row_val, dtype=np.int64),
            "row_test":       np.asarray(row_test, dtype=np.int64),
            "split_counts":   split_counts,
            "clean_datasets": {
                "train": {
                    "x": x_train_scaled.to_numpy(dtype=np.float32, copy=False),
                    "y": y_train_enc.to_numpy(dtype=np.int64, copy=False),
                },
                "val": {
                    "x": x_val_scaled.to_numpy(dtype=np.float32, copy=False),
                    "y": y_val_enc.to_numpy(dtype=np.int64, copy=False),
                },
                "test": {
                    "x": x_test_scaled.to_numpy(dtype=np.float32, copy=False),
                    "y": y_test_enc.to_numpy(dtype=np.int64, copy=False),
                },
                "train_class_weight_labels": y_train_enc.to_numpy(dtype=np.int64, copy=False),
            },
        }

    def prepare(self, attack, prepared: dict | None = None) -> tuple[dict, dict, object]:
        prepared = self.prepare_clean_partitions() if prepared is None else prepared
        attack_name = str(getattr(attack, "name", "")).lower()

        if attack_name == "catback":
            attack_result    = attack.inject(
                clean_features=prepared["x_train_scaled"].copy(deep=True),
                clean_labels=prepared["y_train"].copy(deep=True),
            )
            x_train_poisoned  = attack_result.poisoned_features.copy(deep=True)
            x_train_clean_ref = prepared["x_train_scaled"].copy(deep=True)
            x_val             = prepared["x_val_scaled"].copy(deep=True)
            x_test            = prepared["x_test_scaled"].copy(deep=True)
            x_val_triggered   = attack.apply_trigger_to_features(prepared["x_val_scaled"])
            x_test_triggered  = attack.apply_trigger_to_features(prepared["x_test_scaled"])
            output_scaler          = prepared["scaler"]
            attack_injection_stage = "post_scaler_model_feature_space"
        else:
            attack_result = attack.inject(
                clean_features=prepared["x_train_raw"],
                clean_labels=prepared["y_train"].copy(deep=True),
            )
            x_val_triggered_raw  = attack.apply_trigger_to_features(prepared["x_val_raw"])
            x_test_triggered_raw = attack.apply_trigger_to_features(prepared["x_test_raw"])

            cols = prepared["feature_names"]
            output_scaler = self._make_scaler()
            x_train_poisoned  = pd.DataFrame(output_scaler.fit_transform(attack_result.poisoned_features), columns=cols)
            x_train_clean_ref = pd.DataFrame(output_scaler.transform(prepared["x_train_raw"]),             columns=cols)
            x_val             = pd.DataFrame(output_scaler.transform(prepared["x_val_raw"]),               columns=cols)
            x_test            = pd.DataFrame(output_scaler.transform(prepared["x_test_raw"]),              columns=cols)
            x_val_triggered   = pd.DataFrame(output_scaler.transform(x_val_triggered_raw),                columns=x_val_triggered_raw.columns)
            x_test_triggered  = pd.DataFrame(output_scaler.transform(x_test_triggered_raw),               columns=x_test_triggered_raw.columns)
            attack_injection_stage = "preprocess_before_scaler"

        poisoned_labels = attack_result.get_poisoned_labels(prepared["y_train"].copy(deep=True))

        datasets = {
            "train": {
                "x": x_train_poisoned.to_numpy(dtype=np.float32, copy=False),
                "y": poisoned_labels.to_numpy(dtype=np.int64, copy=False),
            },
            "train_clean_reference": {
                "x": x_train_clean_ref.to_numpy(dtype=np.float32, copy=False),
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
        }

        num_features = int(x_train_poisoned.shape[1])
        metadata = {
            "dataset":       self.schema.name,
            "classes":       prepared["encoder"].classes_.tolist(),
            "train_shape":   list(x_train_poisoned.shape),
            "val_shape":     list(x_val.shape),
            "test_shape":    list(x_test.shape),
            "model_input_min": np.zeros(num_features, dtype=np.float32).tolist(),
            "model_input_max": np.ones(num_features, dtype=np.float32).tolist(),
            "attack_injection_stage": attack_injection_stage,
            "attack_feature_space": (
                "scaled_model_input" if attack_name == "catback" else "raw_before_preprocessing"
            ),
            "imbalance_protocol": "balanced_cross_entropy_from_train_labels",
            "dataset_random_state": int(self.schema.random_state),
            "class_counts": prepared["split_counts"],
        }

        return datasets, metadata, attack_result
