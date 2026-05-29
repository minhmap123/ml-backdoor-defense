import numpy as np
import pandas as pd

from .base import BaseAttacker


class NoneAttack(BaseAttacker):
    """Pass-through "attack" — model trained on 100% clean data.

    Used as the clean baseline in the benchmark: skips poisoning entirely
    (empty poison_indices) so BaseAttacker.inject() returns clean features
    and labels unchanged. apply_trigger_to_features() is also identity, so
    val_triggered / test_triggered equal val_clean / test_clean — ASR on
    such a clean model is just clean accuracy on target_label.
    """

    def _get_poison_indices(self, clean_labels: pd.Series) -> np.ndarray:
        return np.empty(0, dtype=np.int64)

    def _apply_trigger(self, poison_batch: pd.DataFrame) -> pd.DataFrame:
        return poison_batch
