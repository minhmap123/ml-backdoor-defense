from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..models import get_model, train_torch_model
from .base import BaseUnlearner
from .types import ForgetSet, UnlearningContext, UnlearningResult
from .utils import build_retain_datasets, resolve_target_label


class RetrainFromScratchUnlearner(BaseUnlearner):
    def _run_impl(self, context: UnlearningContext, forget_set: ForgetSet) -> UnlearningResult:
        assert context.model_cfg is not None, "RetrainFromScratchUnlearner requires context.model_cfg."

        retain_indices = self._resolve_retain_indices(context, forget_set)
        retain_datasets = build_retain_datasets(context.datasets, retain_indices)
        fresh_model = get_model(dict(context.model_cfg))

        train_cfg = self._build_train_cfg(context)
        target_label = resolve_target_label(context)
        if target_label is not None:
            train_cfg["target_label"] = int(target_label)

        checkpoint_dir = str(Path(context.run_dir) / "checkpoint")
        train_cfg["save_dir"] = checkpoint_dir

        retrained_model, train_metrics = train_torch_model(
            fresh_model,
            datasets=retain_datasets,
            model_cfg=dict(context.model_cfg),
            train_cfg=train_cfg,
        )

        return UnlearningResult(
            method_name=self.name,
            track_type=self.track_type,
            status="ok",
            seed=int(context.seed),
            runtime_sec=0.0,
            forget_set_source=forget_set.source,
            metrics_after=train_metrics,
            checkpoint_dir=train_metrics.get("checkpoint_dir", checkpoint_dir),
            model_after=retrained_model,
        )

    def _resolve_retain_indices(self, context: UnlearningContext, forget_set: ForgetSet):
        from .utils import complement_indices, count_split_samples

        num_train = count_split_samples(context.datasets["train"])
        return complement_indices(num_train, forget_set.indices)

    def _build_train_cfg(self, context: UnlearningContext) -> Dict[str, Any]:
        train_cfg = dict(context.train_cfg or {})
        train_overrides = self.resolved_cfg.get("train_overrides", {})
        train_cfg.update(train_overrides)
        train_cfg["seed"] = int(context.seed)
        if context.device is not None:
            train_cfg["device"] = str(context.device)
        return train_cfg
