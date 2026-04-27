from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger


LOGGER = get_logger("attacks.base")


@dataclass
class AttackResult:
    attack_name: str
    poisoned_features: pd.DataFrame
    poison_indices: np.ndarray
    target_label: int
    poisoned_labels: Optional[pd.Series] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifact_dir: Optional[str] = None

    def get_poisoned_labels(self, clean_labels: pd.Series) -> pd.Series:
        if self.poisoned_labels is not None:
            return self.poisoned_labels.copy(deep=True)
        labels = clean_labels.copy(deep=True)
        labels.iloc[self.poison_indices] = self.target_label
        return labels

    def save(self, output_dir: str) -> str:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self.poisoned_features.to_pickle(out / "poisoned_features.pkl")
        np.save(out / "poison_indices.npy", self.poison_indices.astype(np.int64, copy=False))

        payload = {
            "attack_name": self.attack_name,
            "target_label": int(self.target_label),
            "metadata": self._to_jsonable(self.metadata),
        }

        if self.poisoned_labels is not None:
            self.poisoned_labels.to_pickle(out / "poisoned_labels.pkl")
            payload["has_poisoned_labels"] = True
        else:
            payload["has_poisoned_labels"] = False

        with (out / "result.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

        self.artifact_dir = str(out)
        return self.artifact_dir

    @classmethod
    def load(cls, output_dir: str) -> "AttackResult":
        out = Path(output_dir)
        with (out / "result.json").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        poisoned_features = pd.read_pickle(out / "poisoned_features.pkl")
        poison_indices = np.load(out / "poison_indices.npy")

        poisoned_labels = None
        if bool(payload.get("has_poisoned_labels", False)):
            poisoned_labels = pd.read_pickle(out / "poisoned_labels.pkl")

        return cls(
            attack_name=str(payload["attack_name"]),
            poisoned_features=poisoned_features,
            poison_indices=poison_indices.astype(np.int64, copy=False),
            target_label=int(payload["target_label"]),
            poisoned_labels=poisoned_labels,
            metadata=dict(payload.get("metadata", {})),
            artifact_dir=str(out),
        )

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): AttackResult._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [AttackResult._to_jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return value


class BaseAttacker(ABC):
    def __init__(self, cfg):
        self.name = cfg.name
        self.poison_rate = float(cfg.poison_rate)
        self.target_label = int(cfg.target_label)
        self.seed = getattr(cfg, "seed", None)
        self.dataset_name = getattr(cfg, "dataset_name", None)
        self.sample_non_target_only = bool(getattr(cfg, "sample_non_target_only", True))
        self.store_poisoned_labels = bool(getattr(cfg, "store_poisoned_labels", False))
        self.auto_save_result = bool(getattr(cfg, "auto_save_result", True))
        self.attack_artifact_root = str(getattr(cfg, "attack_artifact_root", "artifacts/attacks"))

        if self.poison_rate < 0.0 or self.poison_rate > 1.0:
            raise ValueError(f"poison_rate must be in [0, 1], got {self.poison_rate}")

    def inject(self, clean_features: pd.DataFrame, clean_labels: pd.Series) -> AttackResult:
        self._validate_inputs(clean_features, clean_labels)
        LOGGER.info(
            "Attack start: name=%s samples=%d",
            self.name,
            len(clean_labels),
        )

        poisoned_features = clean_features.copy(deep=True)
        poisoned_labels = clean_labels.copy(deep=True)

        # Hook 1: prepare attack state (feature stats, etc.)
        self._prepare_attack(clean_features, clean_labels)

        # Hook 2: get poison indices (sampling strategy)
        poison_indices = self._get_poison_indices(clean_labels)

        if poison_indices.size == 0:
            LOGGER.info("Attack done: name=%s poisoned=0/%d ratio=0.0000", self.name, len(clean_labels))
            return self._build_result(
                poisoned_features=poisoned_features,
                poison_indices=poison_indices,
                poisoned_labels=poisoned_labels,
            )

        poison_batch = poisoned_features.iloc[poison_indices].copy(deep=True)
        poisoned_features.iloc[poison_indices] = self._apply_trigger(poison_batch)

        # Hook 3: apply label strategy (overwrite or keep original)
        self._apply_label_strategy(poisoned_labels, poison_indices)

        LOGGER.info(
            "Attack done: name=%s poisoned=%d/%d ratio=%.4f",
            self.name,
            poison_indices.size,
            len(clean_labels),
            poison_indices.size / max(len(clean_labels), 1),
        )
        return self._build_result(
            poisoned_features=poisoned_features,
            poison_indices=poison_indices,
            poisoned_labels=poisoned_labels,
        )

    def apply_trigger_to_features(self, clean_features: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the learned/configured trigger to every row without changing labels.

        Intended use:
        - build triggered test sets for ASR evaluation,
        - inspect trigger behavior after `inject(...)` has prepared any attack state.
        """
        if not isinstance(clean_features, pd.DataFrame):
            raise TypeError("clean_features must be a pandas.DataFrame for trigger-only application.")
        if clean_features.empty:
            return clean_features.copy(deep=True)
        return self._apply_trigger(clean_features.copy(deep=True))

    def _build_result(
        self,
        *,
        poisoned_features: pd.DataFrame,
        poison_indices: np.ndarray,
        poisoned_labels: Optional[pd.Series],
    ) -> AttackResult:
        stored_labels = poisoned_labels.copy(deep=True) if (self.store_poisoned_labels and poisoned_labels is not None) else None
        result = AttackResult(
            attack_name=self.name,
            poisoned_features=poisoned_features.copy(deep=True),
            poison_indices=np.asarray(poison_indices, dtype=np.int64),
            target_label=self.target_label,
            poisoned_labels=stored_labels,
            metadata=self._attack_metadata(),
        )
        if self.auto_save_result:
            run_dir = self._build_artifact_output_dir()
            result.save(run_dir)
        return result

    def _build_artifact_output_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        folder = f"{self.name}_{timestamp}"
        return str(Path(self.attack_artifact_root) / folder)

    def _attack_metadata(self) -> Dict[str, Any]:
        metadata = {
            "poison_rate": self.poison_rate,
            "target_label": self.target_label,
            "seed": self.seed,
            "dataset_name": self.dataset_name,
            "sample_non_target_only": self.sample_non_target_only,
            "store_poisoned_labels": self.store_poisoned_labels,
            "auto_save_result": self.auto_save_result,
            "attack_artifact_root": self.attack_artifact_root,
        }
        # Hook 4: add attack-specific metadata
        metadata.update(self._attack_metadata_extras())
        return metadata

    def _validate_inputs(self, clean_features: pd.DataFrame, clean_labels: pd.Series) -> None:
        if not isinstance(clean_features, pd.DataFrame):
            raise TypeError("clean_features must be a pandas.DataFrame for this attack interface.")
        if not isinstance(clean_labels, pd.Series):
            raise TypeError("clean_labels must be a pandas.Series for this attack interface.")
        if len(clean_features) != len(clean_labels):
            raise ValueError("clean_features and clean_labels must have the same number of samples.")

    def _prepare_attack(self, clean_features: pd.DataFrame, clean_labels: pd.Series) -> None:
        """
        Hook: prepare attack state before poisoning.
        Override to compute feature statistics, setup encodings, etc.
        Default: no-op.
        """
        pass

    def _get_poison_indices(self, clean_labels: pd.Series) -> np.ndarray:
        """
        Hook: determine which samples to poison.
        Override for custom sampling logic (e.g., ranked by confidence, clean-label filtering).
        Default: random sample from candidate pool.
        """
        candidate_indices = self._candidate_indices(clean_labels)
        num_poison = self._num_poison_samples(len(clean_labels), len(candidate_indices))
        return self._sample_poison_indices(candidate_indices, num_poison)

    def _apply_label_strategy(self, poisoned_labels: pd.Series, poison_indices: np.ndarray) -> None:
        """
        Hook: decide how to update labels.
        Override for conditional or no-op label updates (e.g., clean-label attacks).
        Default: overwrite poisoned samples to target_label.
        """
        self._update_labels(poisoned_labels, poison_indices)

    def _attack_metadata_extras(self) -> Dict[str, Any]:
        """
        Hook: add attack-specific metadata fields.
        Override to include trigger parameters, feature rankings, etc.
        Default: empty dict.
        """
        return {}

    def _candidate_indices(self, clean_labels: pd.Series) -> np.ndarray:
        if self.sample_non_target_only:
            return np.flatnonzero(clean_labels.to_numpy(copy=False) != self.target_label).astype(np.int64)
        return np.arange(len(clean_labels), dtype=np.int64)

    def _num_poison_samples(self, total_samples: int, candidate_count: int) -> int:
        return min(int(total_samples * self.poison_rate), candidate_count)

    def _sample_poison_indices(self, candidate_indices: np.ndarray, num_poison: int) -> np.ndarray:
        if num_poison <= 0 or candidate_indices.size == 0:
            return np.empty(0, dtype=np.int64)
        permutation = self._randperm(candidate_indices.size)
        return candidate_indices[permutation[:num_poison]]

    def _update_labels(self, poisoned_labels: pd.Series, poison_indices: np.ndarray) -> None:
        poisoned_labels.iloc[poison_indices] = self.target_label

    def _randperm(self, n: int) -> np.ndarray:
        if self.seed is None:
            return np.random.default_rng().permutation(n)
        return np.random.default_rng(int(self.seed)).permutation(n)

    @abstractmethod
    def _apply_trigger(self, poison_batch: pd.DataFrame) -> pd.DataFrame:
        "Apply attack trigger to the poisoned feature batch."
        pass
