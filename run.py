import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from src.attacks import get_attack
from src.attacks.base import AttackResult
from src.data import IoTID20Dataset
from src.detection import get_detection
from src.detection.types import ArtifactIndex, DetectorContext, DetectorResult, FeatureMetadata
from src.models import get_model, train_torch_model
from src.unlearning import get_unlearning
from src.utils.logging import (
    get_wandb_summary_profile,
    init_wandb,
    log_wandb_stage_metrics_from_payload,
    update_wandb_summary_from_stage_payload,
)


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


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return _to_jsonable(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value


def _write_json(path: str | Path, payload: Dict[str, Any]) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2, sort_keys=True)
    return str(out)


def _write_stage_status(stage_dir: Path, status: str) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "status.txt").write_text(str(status), encoding="utf-8")


def _save_resolved_config(stage_dir: Path, cfg: DictConfig) -> str:
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / "resolved_config.yaml"
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    return str(path)


def _stage_dir(cfg: DictConfig, stage: str) -> Path:
    pipeline_cfg = cfg.get("pipeline", {})
    explicit_dir = pipeline_cfg.get("artifact_dir")
    if explicit_dir:
        return Path(str(explicit_dir))
    run_id = str(pipeline_cfg.get("run_id") or "adhoc")
    return Path("artifacts/pipeline") / run_id / stage


def _should_skip_stage(stage_dir: Path, cfg: DictConfig) -> bool:
    pipeline_cfg = cfg.get("pipeline", {})
    if not bool(pipeline_cfg.get("skip_existing", False)):
        return False
    summary_path = stage_dir / "summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return str(summary.get("status", "")).lower() in {"ok", "skipped"}


def _make_iotid20_dataset(seed: int) -> IoTID20Dataset:
    schema = replace(IoTID20Dataset.schema, random_state=int(seed))
    return IoTID20Dataset(schema=schema)


def _prepare_model_train_cfg(
    cfg: DictConfig,
    datasets: Dict[str, Any],
    metadata: Dict[str, Any],
) -> tuple[dict, dict]:
    model_cfg, train_cfg = _split_model_and_train_cfg(cfg)
    num_classes = int(len(metadata["classes"]))
    num_features = int(datasets["train"]["x"].shape[1])
    model_cfg["d_in"] = num_features
    model_cfg["d_out"] = num_classes
    model_cfg["num_numeric_features"] = num_features
    model_cfg["cat_cardinalities"] = []
    train_cfg["seed"] = int(cfg.seed)
    train_cfg["target_label"] = int(cfg.attack.target_label)
    return model_cfg, train_cfg


def _parameter_counts(model: torch.nn.Module) -> Dict[str, int]:
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    return {
        "total_params": total,
        "active_params": trainable,
        "trainable_params": trainable,
        "inactive_params": int(total - trainable),
    }


def _save_datasets(stage_dir: Path, datasets: Dict[str, Any]) -> Dict[str, Any]:
    data_dir = stage_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "train_x": np.asarray(datasets["train"]["x"], dtype=np.float32),
        "train_y": np.asarray(datasets["train"]["y"], dtype=np.int64),
        "train_clean_reference_x": np.asarray(datasets["train_clean_reference"]["x"], dtype=np.float32),
        "train_clean_reference_y": np.asarray(datasets["train_clean_reference"]["y"], dtype=np.int64),
        "train_class_weight_labels": np.asarray(
            datasets.get("train_class_weight_labels", datasets["train"]["y"]),
            dtype=np.int64,
        ),
        "val_x": np.asarray(datasets["val"]["x"], dtype=np.float32),
        "val_y": np.asarray(datasets["val"]["y"], dtype=np.int64),
        "val_triggered_x": np.asarray(datasets["val_triggered"]["x"], dtype=np.float32),
        "val_triggered_y": np.asarray(datasets["val_triggered"]["y"], dtype=np.int64),
        "val_clean_labels": np.asarray(datasets["val_clean_labels"], dtype=np.int64),
        "test_x": np.asarray(datasets["test"]["x"], dtype=np.float32),
        "test_y": np.asarray(datasets["test"]["y"], dtype=np.int64),
        "test_triggered_x": np.asarray(datasets["test_triggered"]["x"], dtype=np.float32),
        "test_triggered_y": np.asarray(datasets["test_triggered"]["y"], dtype=np.int64),
        "test_clean_labels": np.asarray(datasets["test_clean_labels"], dtype=np.int64),
    }
    dataset_npz = data_dir / "datasets.npz"
    np.savez_compressed(dataset_npz, **arrays)

    provenance_files: Dict[str, str] = {}
    provenance = datasets.get("sample_provenance", {}) or {}
    for split_name, frame in provenance.items():
        if not isinstance(frame, pd.DataFrame):
            continue
        path = data_dir / f"{split_name}_sample_provenance.csv"
        frame.to_csv(path, index=False)
        provenance_files[str(split_name)] = str(path)

    return {
        "datasets_npz": str(dataset_npz),
        "sample_provenance": provenance_files,
    }


def _load_datasets(artifact_dir: str | Path) -> Dict[str, Any]:
    path = Path(artifact_dir) / "data" / "datasets.npz"
    arrays = np.load(path)
    def arr(name: str) -> np.ndarray:
        return np.array(arrays[name], copy=True)

    return {
        "train": {"x": arr("train_x"), "y": arr("train_y")},
        "train_clean_reference": {
            "x": arr("train_clean_reference_x"),
            "y": arr("train_clean_reference_y"),
        },
        "train_class_weight_labels": arr("train_class_weight_labels") if "train_class_weight_labels" in arrays else arr("train_y"),
        "val": {"x": arr("val_x"), "y": arr("val_y")},
        "val_triggered": {"x": arr("val_triggered_x"), "y": arr("val_triggered_y")},
        "val_clean_labels": arr("val_clean_labels"),
        "test": {"x": arr("test_x"), "y": arr("test_y")},
        "test_triggered": {"x": arr("test_triggered_x"), "y": arr("test_triggered_y")},
        "test_clean_labels": arr("test_clean_labels"),
    }


def _load_json_file(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_attack_train_artifact(artifact_dir: str | Path) -> Dict[str, Any]:
    root = Path(artifact_dir)
    summary = _load_json_file(root / "summary.json")
    return {
        "root": root,
        "summary": summary,
        "datasets": _load_datasets(root),
        "metadata": _load_json_file(root / "metadata.json"),
        "attack_result": AttackResult.load(str(root / "attack_result")),
        "model_cfg": _load_json_file(root / "model_cfg.json"),
        "train_cfg": _load_json_file(root / "train_cfg.json"),
        "model_metrics": _load_json_file(root / "model_metrics.json"),
        "checkpoint_dir": root / "checkpoint",
    }


def _load_model_from_attack_artifact(bundle: Dict[str, Any], device: str) -> torch.nn.Module:
    model = get_model(dict(bundle["model_cfg"]))
    checkpoint_dir = Path(bundle["checkpoint_dir"])
    state_dict = torch.load(checkpoint_dir / "model_state_dict.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(torch.device(str(device)))
    model.eval()
    return model


def _build_feature_metadata(
    datasets: Dict[str, Any], attack_result: AttackResult, metadata: Dict[str, Any] | None = None
) -> FeatureMetadata:
    
    train_x = np.asarray(datasets["train"]["x"], dtype=np.float32)
    feature_names = list(attack_result.poisoned_features.columns)
    num_feats = int(train_x.shape[1])

    lower = np.asarray(metadata["scaler_min"], dtype=np.float32)
    upper = np.asarray(metadata["scaler_max"], dtype=np.float32)

    return FeatureMetadata(
        feature_names=feature_names,
        feature_bounds_min=lower,
        feature_bounds_max=upper,
        num_numeric_features=num_feats,
        num_categorical_features=0,
    )


def _attack_metadata_from_bundle(bundle: Dict[str, Any], train_metrics: Dict[str, Any] | None = None) -> Dict[str, Any]:
    train_metrics = train_metrics or bundle.get("model_metrics", {})
    attack_result = bundle["attack_result"]
    poisoned_train_indices = np.asarray(attack_result.poison_indices, dtype=np.int64)
    attack_metadata = dict(attack_result.metadata)
    if "backdoor/asr" in train_metrics:
        attack_metadata["observed_backdoor_asr"] = float(train_metrics["backdoor/asr"])
    if "backdoor/accuracy" in train_metrics:
        attack_metadata["observed_backdoor_accuracy"] = float(train_metrics["backdoor/accuracy"])
    attack_metadata["num_poisoned_train_samples"] = int(poisoned_train_indices.size)
    return attack_metadata


def _build_detection_context(
    *,
    cfg: DictConfig,
    bundle: Dict[str, Any],
    model: torch.nn.Module,
    run_dir: Path,
) -> DetectorContext:
    datasets = bundle["datasets"]
    metadata = bundle["metadata"]
    attack_result = bundle["attack_result"]
    num_classes = int(len(metadata["classes"]))
    poisoned_train_indices = np.asarray(attack_result.poison_indices, dtype=np.int64)
    is_infected = bool(poisoned_train_indices.size > 0)
    attack_target_label = int(attack_result.target_label)
    attack_source_labels = [class_idx for class_idx in range(num_classes) if class_idx != attack_target_label]
    model_metadata = model.get_model_metadata() if hasattr(model, "get_model_metadata") else {"d_in": int(bundle["model_cfg"]["d_in"])}

    return DetectorContext(
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
        attack_metadata=_attack_metadata_from_bundle(bundle),
        detector_cfg=OmegaConf.to_container(cfg.detection, resolve=True),
        model_metadata=model_metadata,
        feature_metadata=_build_feature_metadata(datasets, attack_result, metadata),
        class_names=[str(x) for x in metadata["classes"]],
        sample_indices=np.arange(int(datasets["train"]["x"].shape[0]), dtype=np.int64),
        true_is_infected=is_infected,
        true_target_class=attack_target_label if is_infected else None,
        evaluation_split=datasets["test"],
        run_dir=str(run_dir),
    )


def _resolve_artifact_path(path_text: str | None, summary_path: Path) -> Path | None:
    if not path_text:
        return None
    path = Path(str(path_text))
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = summary_path.parent / path
        if candidate.exists():
            return candidate
    return path


def _load_detector_result_from_stage(detection_artifact_dir: str | Path) -> DetectorResult | None:
    summary_path = Path(detection_artifact_dir) / "summary.json"
    if not summary_path.exists():
        return None
    summary = _load_json_file(summary_path)
    artifacts = summary.get("artifacts", {}) or {}
    result = DetectorResult(
        detector_name=str(summary.get("detector_name", "unknown")),
        track_type=str(summary.get("track_type", "unknown")),
        status=str(summary.get("status", "unknown")),
        seed=int(summary.get("seed", 0)),
        runtime_sec=float(summary.get("runtime_sec", 0.0)),
        summary_metrics=dict(summary.get("summary_metrics", {}) or {}),
        predicted_is_infected=summary.get("predicted_is_infected"),
        predicted_target_class=summary.get("predicted_target_class"),
        predicted_source_class=summary.get("predicted_source_class"),
        thresholds=dict(summary.get("thresholds", {}) or {}),
        deviation_note=summary.get("deviation_note"),
        artifacts=ArtifactIndex(
            summary_json=str(summary_path),
            raw_scores_csv=artifacts.get("raw_scores_csv"),
            class_scores_csv=artifacts.get("class_scores_csv"),
            suspect_indices_npy=artifacts.get("suspect_indices_npy"),
            optimization_trace_json=artifacts.get("optimization_trace_json"),
            estimated_pattern_npy=artifacts.get("estimated_pattern_npy"),
            plots=list(artifacts.get("plots", []) or []),
            extra_files=dict(artifacts.get("extra_files", {}) or {}),
        ),
    )

    suspect_path = _resolve_artifact_path(artifacts.get("suspect_indices_npy"), summary_path)
    if suspect_path is not None and suspect_path.exists():
        result.suspect_indices = np.load(suspect_path).astype(np.int64, copy=False)

    scores_path = _resolve_artifact_path(artifacts.get("raw_scores_csv"), summary_path)
    if scores_path is not None and scores_path.exists():
        frame = pd.read_csv(scores_path)
        if "score" in frame.columns:
            result.sample_scores = frame["score"].to_numpy(dtype=np.float32)
        if "flag" in frame.columns:
            result.sample_flags = frame["flag"].to_numpy(dtype=np.int64)
        if "rank_position" in frame.columns:
            rank_positions = frame["rank_position"].to_numpy(dtype=np.int64)
            valid = np.flatnonzero(rank_positions >= 0)
            result.sample_ranking = valid[np.argsort(rank_positions[valid])].astype(np.int64, copy=False)

    class_scores_path = _resolve_artifact_path(artifacts.get("class_scores_csv"), summary_path)
    if class_scores_path is not None and class_scores_path.exists():
        frame = pd.read_csv(class_scores_path)
        if "score" in frame.columns:
            result.class_scores = frame["score"].to_numpy(dtype=np.float32)

    return result


def _write_stage_summary(stage_dir: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    _write_json(stage_dir / "summary.json", payload)
    _write_stage_status(stage_dir, str(payload.get("status", "unknown")))
    return payload


def _run_stage_or_record_failure(cfg: DictConfig, stage: str, runner) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        return runner(cfg)
    except Exception as exc:
        stage_dir = _stage_dir(cfg, stage)
        run_id = str(cfg.get("pipeline", {}).get("run_id") or stage_dir.name)
        _write_stage_summary(
            stage_dir,
            {
                "stage": stage,
                "run_id": run_id,
                "status": "failed",
                "seed": int(cfg.seed),
                "runtime_sec": float(time.perf_counter() - started),
                "artifact_dir": str(stage_dir),
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
        )
        raise


def run_attack_train_stage(cfg: DictConfig) -> Dict[str, Any]:
    stage_dir = _stage_dir(cfg, "attack_train")
    run_id = str(cfg.pipeline.get("run_id") or stage_dir.name)
    if _should_skip_stage(stage_dir, cfg):
        print(f"[pipeline] skip_existing hit for attack_train: {stage_dir}")
        return _load_json_file(stage_dir / "summary.json")

    start = time.perf_counter()
    stage_dir.mkdir(parents=True, exist_ok=True)
    _write_stage_status(stage_dir, "running")
    resolved_config_path = _save_resolved_config(stage_dir, cfg)

    attack_cfg = OmegaConf.create(OmegaConf.to_container(cfg.attack, resolve=True))
    with open_dict(attack_cfg):
        attack_cfg.auto_save_result = False
        attack_cfg.attack_artifact_root = str(stage_dir / "attack_result")

    attack = get_attack(attack_cfg)
    print(f"Built attack: {attack.__class__.__name__}")
    dataset = _make_iotid20_dataset(int(cfg.seed))
    prepared = None
    catback_surrogate = None
    if str(cfg.attack.name).lower() == "catback":
        prepared = dataset.prepare_clean_partitions()
        surrogate_datasets = prepared["clean_datasets"]
        surrogate_metadata = {"classes": prepared["encoder"].classes_.tolist()}
        surrogate_model_cfg, surrogate_train_cfg = _prepare_model_train_cfg(
            cfg,
            surrogate_datasets,
            surrogate_metadata,
        )
        surrogate_train_cfg["save_dir"] = str(stage_dir / "catback_surrogate_checkpoint")
        surrogate_model = get_model(surrogate_model_cfg)
        surrogate_model, surrogate_metrics = train_torch_model(
            surrogate_model,
            datasets=surrogate_datasets,
            model_cfg=surrogate_model_cfg,
            train_cfg=surrogate_train_cfg,
        )
        if hasattr(attack, "attach_model"):
            attack.attach_model(surrogate_model)
        catback_surrogate = {
            "checkpoint_dir": surrogate_metrics.get("checkpoint_dir"),
            "clean/val_accuracy_best": surrogate_metrics.get("clean/val_accuracy_best"),
            "clean/val_f1_best": surrogate_metrics.get("clean/val_f1_best"),
            "clean/test/accuracy": surrogate_metrics.get("clean/test/accuracy"),
            "clean/test/f1": surrogate_metrics.get("clean/test/f1"),
        }
    datasets, metadata, attack_result = dataset.prepare(attack, prepared=prepared)
    attack_result.save(str(stage_dir / "attack_result"))

    model_cfg, train_cfg = _prepare_model_train_cfg(cfg, datasets, metadata)
    train_cfg["save_dir"] = str(stage_dir / "checkpoint")
    model = get_model(model_cfg)
    model, train_metrics = train_torch_model(
        model,
        datasets=datasets,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )
    parameter_counts = _parameter_counts(model)

    dataset_artifacts = _save_datasets(stage_dir, datasets)
    _write_json(stage_dir / "metadata.json", metadata)
    _write_json(stage_dir / "model_cfg.json", model_cfg)
    _write_json(stage_dir / "train_cfg.json", train_cfg)
    _write_json(stage_dir / "model_metrics.json", train_metrics)

    runtime_sec = float(time.perf_counter() - start)
    payload = {
        "stage": "attack_train",
        "run_id": run_id,
        "status": "ok",
        "seed": int(cfg.seed),
        "runtime_sec": runtime_sec,
        "attack": str(cfg.attack.name),
        "model": str(cfg.model.name),
        "mode": "attacked",
        "resolved_config_yaml": resolved_config_path,
        "artifact_dir": str(stage_dir),
        "checkpoint_dir": str(stage_dir / "checkpoint"),
        "attack_result_dir": str(stage_dir / "attack_result"),
        "dataset_artifacts": dataset_artifacts,
        "metadata_json": str(stage_dir / "metadata.json"),
        "model_cfg_json": str(stage_dir / "model_cfg.json"),
        "train_cfg_json": str(stage_dir / "train_cfg.json"),
        "model_metrics_json": str(stage_dir / "model_metrics.json"),
        "catback_surrogate": catback_surrogate,
        "train_metrics": train_metrics,
        "summary_metrics": {
            **{k: v for k, v in train_metrics.items() if isinstance(v, (int, float, np.integer, np.floating))},
            "model/total_parameters": float(parameter_counts["total_params"]),
            "model/active_parameters": float(parameter_counts["active_params"]),
            "model/trainable_parameters": float(parameter_counts["trainable_params"]),
            "model/inactive_parameters": float(parameter_counts["inactive_params"]),
        },
        **parameter_counts,
        "poisoned_train_samples": int(np.asarray(attack_result.poison_indices, dtype=np.int64).size),
        "dataset_metadata": {
            "dataset": metadata.get("dataset"),
            "train_shape": metadata.get("train_shape"),
            "val_shape": metadata.get("val_shape"),
            "test_shape": metadata.get("test_shape"),
            "attack_injection_stage": metadata.get("attack_injection_stage"),
            "dataset_random_state": metadata.get("dataset_random_state"),
            "sample_provenance_saved": metadata.get("sample_provenance_saved"),
            "imbalance_protocol": metadata.get("imbalance_protocol"),
            "class_counts": metadata.get("class_counts"),
        },
    }
    return _write_stage_summary(stage_dir, payload)


def run_detection_stage(cfg: DictConfig) -> Dict[str, Any]:
    stage_dir = _stage_dir(cfg, "detection")
    run_id = str(cfg.pipeline.get("run_id") or stage_dir.name)
    if _should_skip_stage(stage_dir, cfg):
        print(f"[pipeline] skip_existing hit for detection: {stage_dir}")
        return _load_json_file(stage_dir / "summary.json")

    attack_train_dir = cfg.pipeline.get("attack_train_artifact_dir")
    if not attack_train_dir:
        raise ValueError("pipeline.attack_train_artifact_dir is required for pipeline.stage=detection")

    start = time.perf_counter()
    stage_dir.mkdir(parents=True, exist_ok=True)
    _write_stage_status(stage_dir, "running")
    resolved_config_path = _save_resolved_config(stage_dir, cfg)

    bundle = _load_attack_train_artifact(attack_train_dir)
    model = _load_model_from_attack_artifact(bundle, str(cfg.train.device))
    detector = get_detection(cfg.detection)
    print(f"Built detection: {detector.__class__.__name__}")
    context = _build_detection_context(cfg=cfg, bundle=bundle, model=model, run_dir=stage_dir)
    detection_result = detector.run(context)
    runtime_sec = float(time.perf_counter() - start)

    nested_summary = detection_result.artifacts.summary_json
    if nested_summary:
        shutil.copyfile(nested_summary, stage_dir / "detector_summary.json")

    context_payload = context.to_jsonable()
    context_payload.pop("poisoned_indices", None)
    context_payload.pop("sample_indices", None)

    payload = {
        **detection_result.to_summary_dict(),
        "stage": "detection",
        "run_id": run_id,
        "status": detection_result.status,
        "seed": int(cfg.seed),
        "runtime_sec": float(detection_result.runtime_sec or runtime_sec),
        "stage_runtime_sec": runtime_sec,
        "artifact_dir": str(stage_dir),
        "source_attack_train_artifact_dir": str(attack_train_dir),
        "resolved_config_yaml": resolved_config_path,
        "detector_summary_json": str(stage_dir / "detector_summary.json") if nested_summary else None,
        "context": context_payload,
        "resolved_cfg": OmegaConf.to_container(cfg.detection, resolve=True),
    }
    return _write_stage_summary(stage_dir, payload)


def run_unlearning_stage(cfg: DictConfig) -> Dict[str, Any]:
    stage_dir = _stage_dir(cfg, "unlearning")
    run_id = str(cfg.pipeline.get("run_id") or stage_dir.name)
    if _should_skip_stage(stage_dir, cfg):
        print(f"[pipeline] skip_existing hit for unlearning: {stage_dir}")
        return _load_json_file(stage_dir / "summary.json")

    attack_train_dir = cfg.pipeline.get("attack_train_artifact_dir")
    if not attack_train_dir:
        raise ValueError("pipeline.attack_train_artifact_dir is required for pipeline.stage=unlearning")

    start = time.perf_counter()
    stage_dir.mkdir(parents=True, exist_ok=True)
    _write_stage_status(stage_dir, "running")
    resolved_config_path = _save_resolved_config(stage_dir, cfg)

    bundle = _load_attack_train_artifact(attack_train_dir)
    model = _load_model_from_attack_artifact(bundle, str(cfg.train.device))
    detection_result = None
    detection_artifact_dir = cfg.pipeline.get("detection_artifact_dir")
    if detection_artifact_dir:
        detection_result = _load_detector_result_from_stage(detection_artifact_dir)

    unlearner = get_unlearning(cfg.unlearning)
    print(f"Built unlearning: {unlearner.__class__.__name__}")
    datasets = bundle["datasets"]
    metadata = bundle["metadata"]
    attack_result = bundle["attack_result"]
    train_sample_indices = np.arange(int(datasets["train"]["x"].shape[0]), dtype=np.int64)
    attack_target_label = int(attack_result.target_label)
    feature_metadata = _build_feature_metadata(datasets, attack_result, metadata)
    attack_metadata = _attack_metadata_from_bundle(bundle)

    unlearning_result = unlearner.run(
        model=model,
        datasets=datasets,
        attack_result=attack_result,
        detection_result=detection_result,
        model_cfg=bundle["model_cfg"],
        train_cfg=bundle["train_cfg"],
        seed=int(cfg.seed),
        device=str(cfg.train.device),
        num_classes=int(len(metadata["classes"])),
        class_names=[str(x) for x in metadata["classes"]],
        target_label=attack_target_label,
        train_sample_indices=train_sample_indices,
        detection_sample_indices=train_sample_indices,
        feature_metadata=feature_metadata,
        attack_metadata=attack_metadata,
        clean_support_split=datasets["val"],
        run_dir=str(stage_dir),
    )
    runtime_sec = float(time.perf_counter() - start)

    nested_summary = unlearning_result.artifacts.summary_json
    if nested_summary:
        shutil.copyfile(nested_summary, stage_dir / "unlearner_summary.json")

    payload = {
        **unlearning_result.to_summary_dict(),
        "stage": "unlearning",
        "run_id": run_id,
        "status": unlearning_result.status,
        "seed": int(cfg.seed),
        "runtime_sec": float(unlearning_result.runtime_sec or runtime_sec),
        "stage_runtime_sec": runtime_sec,
        "artifact_dir": str(stage_dir),
        "source_attack_train_artifact_dir": str(attack_train_dir),
        "source_detection_artifact_dir": None if not detection_artifact_dir else str(detection_artifact_dir),
        "resolved_config_yaml": resolved_config_path,
        "unlearner_summary_json": str(stage_dir / "unlearner_summary.json") if nested_summary else None,
    }
    return _write_stage_summary(stage_dir, payload)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== Active Configuration ===")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    summary_profile = get_wandb_summary_profile(cfg_dict)
    init_wandb(cfg_dict)

    if str(cfg.data.name).lower() != "iotid20":
        raise ValueError("Only data.name=iotid20 is currently wired in run.py")

    stage = str(cfg.get("pipeline", {}).get("stage", "attack_train")).strip().lower()
    stage_runners = {
        "attack_train": ("attack_train", run_attack_train_stage),
        "attack-train": ("attack_train", run_attack_train_stage),
        "detection": ("detection", run_detection_stage),
        "unlearning": ("unlearning", run_unlearning_stage),
    }
    resolved_stage = stage_runners.get(stage)
    if resolved_stage is None:
        raise ValueError(
            f"Unknown pipeline.stage={stage!r}. Expected attack_train, detection, or unlearning."
        )

    stage_name, runner = resolved_stage
    try:
        payload = _run_stage_or_record_failure(cfg, stage_name, runner)
    except Exception:
        summary_path = _stage_dir(cfg, stage_name) / "summary.json"
        if summary_path.exists():
            try:
                payload = _load_json_file(summary_path)
            except Exception:
                payload = None
            if payload is not None:
                log_wandb_stage_metrics_from_payload(payload, prefix=stage_name)
                update_wandb_summary_from_stage_payload(
                    payload,
                    prefix=stage_name,
                    summary_profile=summary_profile,
                )
        raise

    log_wandb_stage_metrics_from_payload(payload, prefix=stage_name)
    update_wandb_summary_from_stage_payload(payload, prefix=stage_name, summary_profile=summary_profile)


if __name__ == "__main__":
    main()
