import pandas as pd

from .base import BaseAttacker


class BadNets(BaseAttacker):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.trigger_features = list(cfg.trigger_features)
        self.trigger_value = cfg.trigger_value

    def _apply_trigger(self, poison_batch: pd.DataFrame):
        poison_batch.iloc[:, self.trigger_features] = self.trigger_value
        return poison_batch

    def _attack_metadata_extras(self):
        return {
            "trigger_features": list(self.trigger_features),
            "trigger_value": self.trigger_value,
        }
