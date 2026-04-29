from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .base import DatasetSchema, NumericIDSDataset


class IoTID20Dataset(NumericIDSDataset):
    imbalance_handling_mode = "hybrid"

    schema = DatasetSchema(
        name="IoTID20",
        raw_path=Path("data/0_raw/IoTID20/IoT Network Intrusion Dataset.csv"),
        target_column="Cat",
        drop_columns=("Flow_ID", "Src_IP", "Dst_IP", "Src_Port", "Dst_Port", "Timestamp"),
    )

    def _balance_train_split(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        mode: str,
        row_ids: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict]:
        row_ids = (
            pd.Series(np.arange(len(y_train), dtype=np.int64))
            if row_ids is None
            else row_ids.reset_index(drop=True).astype(np.int64)
        )
        provenance = pd.DataFrame(
            {
                "split": "train",
                "original_row_id": row_ids.to_numpy(dtype=np.int64, copy=False),
                "pre_balance_index": np.arange(len(y_train), dtype=np.int64),
                "balance_source_index": np.arange(len(y_train), dtype=np.int64),
                "duplicate_of_local_index": np.full(len(y_train), -1, dtype=np.int64),
                "balance_stage": "original",
            }
        )

        mode_normalized = str(mode or "none").strip().lower()
        if mode_normalized in {"none", "off", "false", "0"}:
            x_out = x_train.reset_index(drop=True)
            y_out = y_train.reset_index(drop=True)
            provenance = provenance.reset_index(drop=True)
            provenance["train_local_index"] = np.arange(len(provenance), dtype=np.int64)
            return x_out, y_out, provenance, {
                "imbalance_handling": "none",
            }

        if mode_normalized not in {"hybrid", "mild_hybrid", "notebook"}:
            raise ValueError(
                f"Unsupported imbalance_handling='{mode}'. Expected one of: none, hybrid, mild_hybrid, notebook."
            )

        x_bal = x_train.reset_index(drop=True).copy(deep=True)
        y_bal = y_train.reset_index(drop=True).copy(deep=True)
        provenance_bal = provenance.reset_index(drop=True).copy(deep=True)

        train_counts = pd.Series(y_bal).value_counts().sort_index()
        median_count = int(train_counts.median())
        low_target = max(1, int(round(median_count * 0.75)))
        high_target = max(low_target + 1, int(round(median_count * 1.25)))

        rng = np.random.default_rng(int(self.schema.random_state))

        balanced_parts_x = [x_bal]
        balanced_parts_y = [y_bal]
        balanced_parts_provenance = [provenance_bal]

        for cls, count in train_counts.items():
            if int(count) >= low_target:
                continue
            cls_mask = y_bal == cls
            cls_x = x_bal.loc[cls_mask].reset_index(drop=True)
            cls_y = y_bal.loc[cls_mask].reset_index(drop=True)
            cls_provenance = provenance_bal.loc[cls_mask].reset_index(drop=True)
            extra_count = low_target - int(count)
            extra_indices = rng.integers(0, len(cls_x), size=extra_count)
            extra_provenance = cls_provenance.iloc[extra_indices].reset_index(drop=True).copy(deep=True)
            extra_provenance["duplicate_of_local_index"] = extra_provenance["balance_source_index"].astype(np.int64)
            extra_provenance["balance_stage"] = "oversampled"
            balanced_parts_x.append(cls_x.iloc[extra_indices].reset_index(drop=True))
            balanced_parts_y.append(cls_y.iloc[extra_indices].reset_index(drop=True))
            balanced_parts_provenance.append(extra_provenance)

        x_bal = pd.concat(balanced_parts_x, ignore_index=True)
        y_bal = pd.concat(balanced_parts_y, ignore_index=True)
        provenance_bal = pd.concat(balanced_parts_provenance, ignore_index=True)

        train_counts_after_over = pd.Series(y_bal).value_counts().sort_index()
        keep_indices: list[int] = []
        for cls, count in train_counts_after_over.items():
            cls_indices = pd.Index(np.flatnonzero(y_bal.to_numpy(copy=False) == cls))
            if int(count) > high_target:
                sampled_positions = rng.choice(cls_indices.to_numpy(), size=high_target, replace=False)
                keep_indices.extend(sorted(int(i) for i in sampled_positions))
            else:
                keep_indices.extend(int(i) for i in cls_indices.to_numpy())

        keep_indices = sorted(keep_indices)
        x_bal = x_bal.iloc[keep_indices].reset_index(drop=True)
        y_bal = y_bal.iloc[keep_indices].reset_index(drop=True)
        provenance_bal = provenance_bal.iloc[keep_indices].reset_index(drop=True)

        perm = rng.permutation(len(y_bal))
        x_bal = x_bal.iloc[perm].reset_index(drop=True)
        y_bal = y_bal.iloc[perm].reset_index(drop=True)
        provenance_bal = provenance_bal.iloc[perm].reset_index(drop=True)
        provenance_bal["train_local_index"] = np.arange(len(provenance_bal), dtype=np.int64)

        balancing_metadata = {
            "imbalance_handling": "mild_hybrid",
            "train_counts_before": {int(k): int(v) for k, v in train_counts.items()},
            "median_train_count": int(median_count),
            "low_target": int(low_target),
            "high_target": int(high_target),
            "train_counts_after_over": {int(k): int(v) for k, v in train_counts_after_over.items()},
            "train_counts_after_balance": {
                int(k): int(v) for k, v in pd.Series(y_bal).value_counts().sort_index().items()
            },
            "num_oversampled_duplicates": int((provenance_bal["balance_stage"] == "oversampled").sum()),
            "final_train_permutation_saved": True,
            "sample_provenance_saved": True,
        }
        return x_bal, y_bal, provenance_bal, balancing_metadata

    def prepare(self, attack) -> tuple[dict, dict, object]:
        """
        Prepare IoTID20 with raw-space attack injection followed by preprocessing.

        The attack is applied before StandardScaler so the threat model stays
        aligned with a raw-data attacker.
        """
        schema = self.schema
        raw_df = self.load_raw(schema.raw_path)
        cleaned_df, cleaning = self.clean(raw_df, near_constant_threshold=schema.near_constant_threshold)

        protected_labels = [c for c in schema.protected_label_columns if c in cleaned_df.columns]
        if schema.target_column not in cleaned_df.columns:
            raise ValueError(f"target_column '{schema.target_column}' not found in cleaned data")

        y = cleaned_df[schema.target_column].copy()
        x = cleaned_df.drop(columns=protected_labels, errors="ignore").copy()
        if x.select_dtypes(include=["object"]).shape[1]:
            raise ValueError("Expected numeric-only features")

        all_row_ids = pd.Series(np.arange(len(x), dtype=np.int64), index=x.index)
        x_train_val, x_test, y_train_val, y_test, row_train_val, row_test = train_test_split(
            x,
            y,
            all_row_ids,
            test_size=schema.test_size,
            random_state=schema.random_state,
            stratify=y,
        )
        x_train, x_val, y_train_raw, y_val_raw, row_train, row_val = train_test_split(
            x_train_val,
            y_train_val,
            row_train_val,
            test_size=schema.val_size_within_train,
            random_state=schema.random_state,
            stratify=y_train_val,
        )

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train_raw)
        y_val = encoder.transform(y_val_raw)
        y_test = encoder.transform(y_test.reset_index(drop=True))

        x_train_bal, y_train_bal, train_provenance, balancing_metadata = self._balance_train_split(
            x_train,
            pd.Series(y_train, dtype=np.int64),
            mode=self.imbalance_handling_mode,
            row_ids=pd.Series(row_train.to_numpy(dtype=np.int64)),
        )

        attack_result = attack.inject(
            clean_features=x_train_bal,
            clean_labels=pd.Series(np.asarray(y_train_bal, dtype=np.int64)),
        )
        triggered_val_features_raw = attack.apply_trigger_to_features(x_val.reset_index(drop=True))
        triggered_test_features_raw = attack.apply_trigger_to_features(x_test.reset_index(drop=True))

        scaler = StandardScaler()
        x_train_poisoned = pd.DataFrame(
            scaler.fit_transform(attack_result.poisoned_features),
            columns=attack_result.poisoned_features.columns,
        )
        x_train_clean_reference = pd.DataFrame(
            scaler.transform(x_train_bal),
            columns=x_train_bal.columns,
        )
        x_val_scaled = pd.DataFrame(scaler.transform(x_val.reset_index(drop=True)), columns=x_val.columns)
        x_test_scaled = pd.DataFrame(scaler.transform(x_test.reset_index(drop=True)), columns=x_test.columns)
        x_val_triggered_scaled = pd.DataFrame(
            scaler.transform(triggered_val_features_raw),
            columns=triggered_val_features_raw.columns,
        )
        x_test_triggered_scaled = pd.DataFrame(
            scaler.transform(triggered_test_features_raw),
            columns=triggered_test_features_raw.columns,
        )

        poisoned_labels = attack_result.get_poisoned_labels(pd.Series(np.asarray(y_train_bal, dtype=np.int64)))
        poison_flags = np.zeros(len(train_provenance), dtype=np.int64)
        poison_flags[np.asarray(attack_result.poison_indices, dtype=np.int64)] = 1
        train_provenance = train_provenance.copy(deep=True)
        train_provenance["poison_flag"] = poison_flags
        train_provenance["source_label"] = np.asarray(y_train_bal, dtype=np.int64)
        train_provenance["target_label"] = np.where(poison_flags == 1, int(attack.target_label), -1)
        train_provenance["final_label"] = poisoned_labels.to_numpy(dtype=np.int64, copy=False)

        val_provenance = pd.DataFrame(
            {
                "split": "val",
                "original_row_id": row_val.to_numpy(dtype=np.int64),
                "split_local_index": np.arange(len(y_val), dtype=np.int64),
                "source_label": np.asarray(y_val, dtype=np.int64),
                "poison_flag": np.zeros(len(y_val), dtype=np.int64),
                "target_label": np.full(len(y_val), -1, dtype=np.int64),
                "final_label": np.asarray(y_val, dtype=np.int64),
            }
        )
        test_provenance = pd.DataFrame(
            {
                "split": "test",
                "original_row_id": row_test.to_numpy(dtype=np.int64),
                "split_local_index": np.arange(len(y_test), dtype=np.int64),
                "source_label": np.asarray(y_test, dtype=np.int64),
                "poison_flag": np.zeros(len(y_test), dtype=np.int64),
                "target_label": np.full(len(y_test), -1, dtype=np.int64),
                "final_label": np.asarray(y_test, dtype=np.int64),
            }
        )

        datasets = {
            "train": {
                "x": x_train_poisoned.to_numpy(dtype=np.float32, copy=False),
                "y": poisoned_labels.to_numpy(dtype=np.int64, copy=False),
            },
            "train_clean_reference": {
                "x": x_train_clean_reference.to_numpy(dtype=np.float32, copy=False),
                "y": np.asarray(y_train_bal, dtype=np.int64),
            },
            "val": {
                "x": x_val_scaled.to_numpy(dtype=np.float32, copy=False),
                "y": np.asarray(y_val, dtype=np.int64),
            },
            "val_triggered": {
                "x": x_val_triggered_scaled.to_numpy(dtype=np.float32, copy=False),
                "y": np.asarray(y_val, dtype=np.int64),
            },
            "val_clean_labels": np.asarray(y_val, dtype=np.int64),
            "test": {
                "x": x_test_scaled.to_numpy(dtype=np.float32, copy=False),
                "y": np.asarray(y_test, dtype=np.int64),
            },
            "test_triggered": {
                "x": x_test_triggered_scaled.to_numpy(dtype=np.float32, copy=False),
                "y": np.asarray(y_test, dtype=np.int64),
            },
            "test_clean_labels": np.asarray(y_test, dtype=np.int64),
            "sample_provenance": {
                "train": train_provenance,
                "val": val_provenance,
                "test": test_provenance,
            },
        }

        metadata = {
            "dataset": schema.name,
            "source_path": str(schema.raw_path),
            "target_column": schema.target_column,
            "classes": encoder.classes_.tolist(),
            "label_mapping": {name: int(i) for i, name in enumerate(encoder.classes_)},
            "num_features": int(x_train_poisoned.shape[1]),
            "train_shape": list(x_train_poisoned.shape),
            "val_shape": list(x_val_scaled.shape),
            "test_shape": list(x_test_scaled.shape),
            "scaler": "StandardScaler",
            "scaler_mean": scaler.mean_.astype(float).tolist(),
            "scaler_scale": scaler.scale_.astype(float).tolist(),
            "scaler_var": scaler.var_.astype(float).tolist(),
            "cleaning": cleaning,
            "attack_injection_stage": "preprocess_before_scaler",
            "dataset_random_state": int(schema.random_state),
            "seed_semantics": "single pipeline seed controls split, balancing, poison sampling, model init, detection, and unlearning",
            "sample_provenance_columns": {
                split: list(frame.columns)
                for split, frame in datasets["sample_provenance"].items()
            },
            **balancing_metadata,
        }
        return datasets, metadata, attack_result
