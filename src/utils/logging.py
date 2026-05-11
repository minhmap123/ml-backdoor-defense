import logging
from typing import Optional, Dict, Any, List

import numpy as np
import wandb


DEFAULT_WANDB_SUMMARY_PROFILE = "compact"
_SUMMARY_HISTORY_METRICS = (
    ("epoch", {"hidden": True}),
    ("clean/train_loss", {}),
    ("clean/val/*", {}),
    ("clean/test/*", {}),
    ("clean/val_accuracy_best", {}),
    ("clean/val_f1_best", {}),
    ("backdoor/*", {}),
    ("tabnet/*", {}),
)

def get_logger(name: str) -> logging.Logger:
    """Create/reuse a simple stdout logger for experiment progress logs."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _resolve_component_name(config: Dict[str, Any], key: str) -> Optional[str]:
    section = config.get(key)
    if isinstance(section, dict):
        value = section.get("name")
        if value is not None:
            return str(value)
    if section is not None and not isinstance(section, (dict, list, tuple)):
        return str(section)
    return None


def _derive_wandb_tags(config: Dict[str, Any]) -> List[str]:
    wandb_cfg = config.get("wandb", {})
    configured_tags = wandb_cfg.get("tags", [])
    if configured_tags is None:
        configured_tags = []
    elif not isinstance(configured_tags, (list, tuple, set)):
        configured_tags = [configured_tags]
    tags: List[str] = []
    seen = set()

    for raw_tag in configured_tags:
        tag = str(raw_tag).strip()
        if tag and tag not in seen:
            tags.append(tag)
            seen.add(tag)

    # Determine which component tags to include based on the pipeline stage.
    # For staged benchmarks, only tag components that are active in the current stage.
    stage = str(config.get("pipeline", {}).get("stage", "attack_train")).strip().lower()
    
    # Define which components are active in each stage.
    if stage in ("attack_train", "attack-train"):
        active_components = {"attack", "model"}
    elif stage == "detection":
        active_components = {"attack", "model", "detection"}
    else:
        # Fallback: include all components for unknown stages.
        active_components = {"attack", "model", "detection"}

    for key in active_components:
        component_name = _resolve_component_name(config, key)
        if component_name is None:
            continue
        tag = f"{key}:{component_name}"
        if tag not in seen:
            tags.append(tag)
            seen.add(tag)

    return tags


def get_wandb_summary_profile(config: Dict[str, Any]) -> str:
    wandb_cfg = config.get("wandb", {})
    profile = wandb_cfg.get("summary_profile", DEFAULT_WANDB_SUMMARY_PROFILE)
    return str(profile).strip().lower() or DEFAULT_WANDB_SUMMARY_PROFILE


def _configure_wandb_summary_metrics(config: Dict[str, Any]) -> None:
    if wandb.run is None:
        return

    if get_wandb_summary_profile(config) == "full":
        return

    for metric_name, extra_kwargs in _SUMMARY_HISTORY_METRICS:
        wandb.define_metric(metric_name, summary="none", **extra_kwargs)


def filter_wandb_summary_metrics(summary_metrics: Dict[str, Any], *, prefix: str, profile: str) -> Dict[str, Any]:
    _ = prefix
    profile_normalized = str(profile).strip().lower() or DEFAULT_WANDB_SUMMARY_PROFILE
    if profile_normalized == "full":
        return dict(summary_metrics)

    filtered: Dict[str, Any] = {}
    for key, value in summary_metrics.items():
        filtered[str(key)] = value
    return filtered


def as_wandb_summary_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return None


def as_wandb_history_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, float, bool)):
        return value
    return None


def update_wandb_summary_from_stage_payload(payload: Dict[str, Any], prefix: str, *, summary_profile: str) -> None:
    if wandb.run is None or payload is None:
        return

    summary_updates = {
        f"{prefix}/stage": payload.get("stage"),
        f"{prefix}/run_id": payload.get("run_id"),
        f"{prefix}/status": payload.get("status"),
        f"{prefix}/track_type": payload.get("track_type"),
        f"{prefix}/runtime_sec": payload.get("runtime_sec"),
        f"{prefix}/stage_runtime_sec": payload.get("stage_runtime_sec"),
        f"{prefix}/error_type": payload.get("error_type"),
        f"{prefix}/error": payload.get("error"),
    }

    for attr in ("predicted_is_infected", "predicted_target_class", "predicted_source_class"):
        if attr in payload:
            summary_updates[f"{prefix}/{attr}"] = payload.get(attr)

    summary_metrics = payload.get("summary_metrics", {}) or {}
    summary_metrics = filter_wandb_summary_metrics(
        summary_metrics,
        prefix=prefix,
        profile=summary_profile,
    )
    for key, value in summary_metrics.items():
        summary_updates[str(key)] = value

    try:
        for key, value in summary_updates.items():
            scalar_value = as_wandb_summary_scalar(value)
            if scalar_value is not None:
                wandb.run.summary[key] = scalar_value
    except Exception as exc:
        print(f"[W&B] Failed to update staged summary for {prefix}: {exc}")


def log_wandb_stage_metrics_from_payload(payload: Dict[str, Any], prefix: str) -> None:
    if wandb.run is None or payload is None:
        return

    metric_updates: Dict[str, Any] = {}
    for key in ("runtime_sec", "stage_runtime_sec", "predicted_is_infected", "predicted_target_class", "predicted_source_class"):
        scalar_value = as_wandb_history_scalar(payload.get(key))
        if scalar_value is not None:
            metric_updates[f"{prefix}/{key}"] = scalar_value

    summary_metrics = payload.get("summary_metrics", {}) or {}
    for key, value in summary_metrics.items():
        scalar_value = as_wandb_history_scalar(value)
        if scalar_value is not None:
            metric_updates[str(key)] = scalar_value

    if metric_updates:
        log_metrics(metric_updates)


def init_wandb(config: Dict[str, Any]) -> bool:
    """
    Initialize W&B with graceful degradation.
    Returns True if W&B is initialized, False if disabled or failed.
    """
    try:
        wandb_cfg = config.get("wandb", {})
        if not wandb_cfg.get("enabled", True):
            return False

        project = wandb_cfg.get("project", "backdoor_detection_benchmark")
        entity = wandb_cfg.get("entity", None)
        tags = _derive_wandb_tags(config)

        wandb.init(
            project=project,
            entity=entity,
            config=dict(config),
            notes=wandb_cfg.get("notes"),
            tags=tags,
        )
        _configure_wandb_summary_metrics(config)
        print(f"[W&B] Logging initialized successfully. tags={tags}")
        return True
    except Exception as e:
        print(f"[W&B] Failed to initialize: {e}. Continuing without W&B logging.")
        return False


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """
    Log metrics to W&B if initialized.
    Safe to call even if W&B is not initialized.
    """
    try:
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception:
        pass  # Gracefully skip logging if W&B fails


def update_wandb_config(config: Dict[str, Any]) -> None:
    """
    Merge extra configuration into the active W&B run.
    Safe to call even if W&B is not initialized.
    """
    try:
        if wandb.run is not None:
            wandb.config.update(config, allow_val_change=True)
    except Exception:
        pass
