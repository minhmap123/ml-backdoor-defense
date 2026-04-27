from __future__ import annotations

from typing import Any

from .abl import ABLUnlearner
from .bad_teaching import BadTeachingUnlearner
from .base import BaseUnlearner, NoOpUnlearner
from .retrain import RetrainFromScratchUnlearner
from .types import ForgetSet, UnlearningArtifacts, UnlearningContext, UnlearningResult


UNLEARNING_REGISTRY = {
    "abl": ABLUnlearner,
    "bad_teaching": BadTeachingUnlearner,
    "badteaching": BadTeachingUnlearner,
    "none": NoOpUnlearner,
    "noop": NoOpUnlearner,
    "retrain": RetrainFromScratchUnlearner,
    "oracle_retrain": RetrainFromScratchUnlearner,
    "detected_retrain": RetrainFromScratchUnlearner,
}


def get_unlearning(cfg: Any, unlearning_name: str | None = None):
    cfg_name = cfg.get("name") if isinstance(cfg, dict) else getattr(cfg, "name", None)
    name = str(unlearning_name or cfg_name or "none").lower()
    try:
        return UNLEARNING_REGISTRY[name](cfg)
    except KeyError as exc:
        available = ", ".join(sorted(UNLEARNING_REGISTRY))
        raise ValueError(f"Unknown unlearning: {name}. Available methods: {available}") from exc


__all__ = [
    "ABLUnlearner",
    "BadTeachingUnlearner",
    "BaseUnlearner",
    "ForgetSet",
    "NoOpUnlearner",
    "RetrainFromScratchUnlearner",
    "UNLEARNING_REGISTRY",
    "UnlearningArtifacts",
    "UnlearningContext",
    "UnlearningResult",
    "get_unlearning",
]
