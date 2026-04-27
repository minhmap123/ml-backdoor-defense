from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import wandb

from src.attacks import get_attack
from src.data import IoTID20Dataset
from src.detection import get_detection
from src.detection.types import DetectorContext, FeatureMetadata
from src.models import get_model, train_torch_model
from src.unlearning import get_unlearning
from src.utils.logging import filter_wandb_summary_metrics, get_wandb_summary_profile, init_wandb


TRAIN_OVERRIDE_KEYS = (
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "patience",
    "save_dir",
    "device",
    "seed",
)


def _load_default_train_cfg() -> dict:
    default_path = Path(__file__).resolve().parent / "conf" / "train" / "default.yaml"
    return dict(OmegaConf.to_container(OmegaConf.load(default_path), resolve=True))


def _split_model_and_train_cfg(cfg: DictConfig) -> tuple[dict, dict]:
    raw_model_cfg = dict(OmegaConf.to_container(cfg.model, resolve=True))
    train_cfg = dict(OmegaConf.to_container(cfg.train, resolve=True))
    default_train_cfg = _load_default_train_cfg()

    for key in TRAIN_OVERRIDE_KEYS:
        if key not in raw_model_cfg:
            continue
        default_value = default_train_cfg.get(key)
        current_train_value = train_cfg.get(key)
        if key not in train_cfg or current_train_value == default_value:
            train_cfg[key] = raw_model_cfg[key]

    model_cfg = {k: v for k, v in raw_model_cfg.items() if k not in TRAIN_OVERRIDE_KEYS}
    return model_cfg, train_cfg


def _as_wandb_summary_scalar(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return None


def _update_wandb_summary_from_result(result, prefix: str, *, summary_profile: str) -> None:
    if wandb.run is None or result is None:
        return

    summary_updates = {
        f"{prefix}/status": getattr(result, "status", None),
        f"{prefix}/track_type": getattr(result, "track_type", None),
        f"{prefix}/runtime_sec": getattr(result, "runtime_sec", None),
    }

    for attr in ("predicted_is_infected", "predicted_target_class", "predicted_source_class"):
        if hasattr(result, attr):
            summary_updates[f"{prefix}/{attr}"] = getattr(result, attr)

    summary_metrics = getattr(result, "summary_metrics", None) or {}
    summary_metrics = filter_wandb_summary_metrics(
        summary_metrics,
        prefix=prefix,
        profile=summary_profile,
    )
    for key, value in summary_metrics.items():
        summary_updates[str(key)] = value

    try:
        for key, value in summary_updates.items():
            scalar_value = _as_wandb_summary_scalar(value)
            if scalar_value is not None:
                wandb.run.summary[key] = scalar_value
    except Exception as exc:
        print(f"[W&B] Failed to update summary for {prefix}: {exc}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== Active Configuration ===")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    summary_profile = get_wandb_summary_profile(cfg_dict)
    init_wandb(cfg_dict)

    if str(cfg.data.name).lower() != "iotid20":
        raise ValueError("Only data.name=iotid20 is currently wired in run.py")

    dataset = IoTID20Dataset()
    prepared = dataset.prepare()

    x_train = prepared["train"]["x"]
    y_train = prepared["train"]["y"]
    x_val = prepared["val"]["x"]
    y_val = prepared["val"]["y"]
    x_test = prepared["test"]["x"]
    y_test = prepared["test"]["y"]
    num_classes = int(len(prepared["metadata"]["classes"]))

    attack = get_attack(cfg.attack)
    print(f"Built attack: {attack.__class__.__name__}")

    attack_result = attack.inject(
        clean_features=pd.DataFrame(x_train),
        clean_labels=pd.Series(np.asarray(y_train, dtype=np.int64)),
    )
    triggered_test_features = attack.apply_trigger_to_features(pd.DataFrame(x_test))

    poisoned_labels = attack_result.get_poisoned_labels(pd.Series(np.asarray(y_train, dtype=np.int64)))
    train_split = {
        "x": attack_result.poisoned_features.to_numpy(dtype=np.float32, copy=False),
        "y": poisoned_labels.to_numpy(dtype=np.int64, copy=False),
    }

    datasets = {
        "train": train_split,
        "val": {
            "x": x_val.to_numpy(dtype=np.float32, copy=False),
            "y": np.asarray(y_val, dtype=np.int64),
        },
        "test": {
            "x": x_test.to_numpy(dtype=np.float32, copy=False),
            "y": np.asarray(y_test, dtype=np.int64),
        },
        "test_triggered": {
            "x": triggered_test_features.to_numpy(dtype=np.float32, copy=False),
            "y": np.asarray(y_test, dtype=np.int64),
        },
        "test_clean_labels": np.asarray(y_test, dtype=np.int64),
    }

    model_cfg, train_cfg = _split_model_and_train_cfg(cfg)
    model_cfg["d_in"] = int(datasets["train"]["x"].shape[1])
    model_cfg["d_out"] = int(num_classes)
    model_cfg["num_numeric_features"] = int(datasets["train"]["x"].shape[1])
    model_cfg["cat_cardinalities"] = []

    train_cfg["seed"] = int(cfg.seed)
    train_cfg["target_label"] = int(cfg.attack.target_label)

    model = get_model(model_cfg)
    model, train_metrics = train_torch_model(
        model,
        datasets=datasets,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )
    print("Training finished. clean/test/accuracy=", train_metrics.get("clean/test/accuracy"))

    detector = get_detection(cfg.detection)
    print(f"Built detection: {detector.__class__.__name__}")

    train_x = np.asarray(train_split["x"], dtype=np.float32)
    feature_names = list(attack_result.poisoned_features.columns)
    feature_metadata = FeatureMetadata(
        feature_names=feature_names,
        feature_bounds_min=np.min(train_x, axis=0).astype(np.float32),
        feature_bounds_max=np.max(train_x, axis=0).astype(np.float32),
        num_numeric_features=int(train_x.shape[1]),
        num_categorical_features=0,
    )

    poisoned_train_indices = np.asarray(attack_result.poison_indices, dtype=np.int64)
    is_infected = bool(poisoned_train_indices.size > 0)
    attack_target_label = int(cfg.attack.target_label)
    attack_source_labels = [class_idx for class_idx in range(num_classes) if class_idx != attack_target_label]
    attack_metadata = dict(attack_result.metadata)
    if "backdoor/asr" in train_metrics:
        attack_metadata["observed_backdoor_asr"] = float(train_metrics["backdoor/asr"])
    if "backdoor/accuracy" in train_metrics:
        attack_metadata["observed_backdoor_accuracy"] = float(train_metrics["backdoor/accuracy"])
    attack_metadata["num_poisoned_train_samples"] = int(poisoned_train_indices.size)

    model_metadata = (
        model.get_model_metadata()
        if hasattr(model, "get_model_metadata")
        else {"d_in": int(model_cfg["d_in"])}
    )

    detection_context = DetectorContext(
        model=model,
        model_name=model.__class__.__name__,
        model_family=str(getattr(model, "model_family", "unknown")),
        num_classes=num_classes,
        detection_split=datasets["train"],
        seed=int(cfg.seed),
        device=str(cfg.train.device),
        clean_support_split=datasets["val"],
        poisoned_indices=poisoned_train_indices,
        attack_target_label=attack_target_label,
        attack_source_labels=attack_source_labels,
        attack_metadata=attack_metadata,
        detector_cfg=OmegaConf.to_container(cfg.detection, resolve=True),
        model_metadata=model_metadata,
        feature_metadata=feature_metadata,
        class_names=[str(x) for x in prepared["metadata"]["classes"]],
        sample_indices=np.arange(int(datasets["train"]["x"].shape[0]), dtype=np.int64),
        true_is_infected=is_infected,
        true_target_class=attack_target_label if is_infected else None,
        evaluation_split=datasets["test"],
    )

    try:
        detection_result = detector.run(detection_context)
        print("Detection finished. status=", detection_result.status)
    except Exception as exc:
        detection_result = None
        print(f"Detection stage skipped/failed: {exc}")
    _update_wandb_summary_from_result(
        detection_result,
        prefix="detection",
        summary_profile=summary_profile,
    )

    unlearner = get_unlearning(cfg.unlearning)
    print(f"Built unlearning: {unlearner.__class__.__name__}")
    train_sample_indices = np.arange(int(datasets["train"]["x"].shape[0]), dtype=np.int64)
    unlearning_result = unlearner.run(
        model=model,
        datasets=datasets,
        attack_result=attack_result,
        detection_result=detection_result,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        seed=int(cfg.seed),
        device=str(cfg.train.device),
        num_classes=num_classes,
        class_names=[str(x) for x in prepared["metadata"]["classes"]],
        target_label=attack_target_label,
        train_sample_indices=train_sample_indices,
        detection_sample_indices=train_sample_indices,
        feature_metadata=feature_metadata,
        attack_metadata=attack_metadata,
        clean_support_split=datasets["val"],
    )
    print("Unlearning finished.", unlearning_result)
    _update_wandb_summary_from_result(
        unlearning_result,
        prefix="unlearning",
        summary_profile=summary_profile,
    )


if __name__ == "__main__":
    main()
