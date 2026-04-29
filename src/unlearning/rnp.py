from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from ..models.train import compute_backdoor_metrics
from ..models.utils import parse_torch_batch, resolve_device, save_model, split_to_dataloader
from ..utils.logging import get_logger
from .bad_teaching.utils import clone_model
from .base import BaseUnlearner
from .types import ForgetSet, UnlearningArtifacts, UnlearningContext, UnlearningResult
from .utils import count_split_samples, subset_split


LOGGER = get_logger("unlearning.rnp")


class MaskBatchNorm1d(nn.BatchNorm1d):
    """
    BatchNorm1d variant adapted from the official RNP mask_batchnorm.py.

    Upstream provenance for this wrapper:
    - repo: https://github.com/bboylyg/RNP
    - commit: eeae192e5eab974d8b3002964cfb62d00388d36f
    - main source: main.py
    - mask layer source: models/mask_batchnorm.py

    Local deviations:
    - this port targets BatchNorm1d-based tabular models only;
    - prune-point selection uses validation-side dynamic thresholding to avoid
      test leakage while staying close to the paper's pruning spirit;
    - if the official clean-threshold stop is never reached, the last unlearned
      checkpoint is used so the recovering/pruning stages remain defined.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__(
            int(num_features),
            eps=float(eps),
            momentum=float(momentum),
            affine=bool(affine),
            track_running_stats=bool(track_running_stats),
        )
        self.neuron_mask = Parameter(torch.ones(int(num_features)))
        self.neuron_noise = Parameter(torch.zeros(int(num_features)))
        self.neuron_noise_bias = Parameter(torch.zeros(int(num_features)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)

        coeff_weight = self.neuron_mask
        coeff_bias = 1.0

        return F.batch_norm(
            input,
            self.running_mean if (not self.training or self.track_running_stats) else None,
            self.running_var if (not self.training or self.track_running_stats) else None,
            self.weight * coeff_weight if self.weight is not None else None,
            self.bias * coeff_bias if self.bias is not None else None,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


def _copy_batchnorm1d_state(source: nn.BatchNorm1d, target: MaskBatchNorm1d) -> None:
    target.load_state_dict(source.state_dict(), strict=False)
    if source.affine:
        assert target.weight is not None
        assert target.bias is not None
        assert source.weight is not None
        assert source.bias is not None
        target.weight.data.copy_(source.weight.data)
        target.bias.data.copy_(source.bias.data)
    if source.running_mean is not None:
        target.running_mean.data.copy_(source.running_mean.data)
    if source.running_var is not None:
        target.running_var.data.copy_(source.running_var.data)
    if source.num_batches_tracked is not None:
        target.num_batches_tracked.data.copy_(source.num_batches_tracked.data)
    target.neuron_mask.data.fill_(1.0)
    target.neuron_noise.data.zero_()
    target.neuron_noise_bias.data.zero_()


def _replace_batchnorm1d_with_mask_batchnorm1d(module: nn.Module) -> int:
    replaced = 0
    for child_name, child_module in list(module.named_children()):
        if isinstance(child_module, nn.BatchNorm1d) and not isinstance(child_module, MaskBatchNorm1d):
            masked = MaskBatchNorm1d(
                num_features=int(child_module.num_features),
                eps=float(child_module.eps),
                momentum=0.1 if child_module.momentum is None else float(child_module.momentum),
                affine=bool(child_module.affine),
                track_running_stats=bool(child_module.track_running_stats),
            )
            _copy_batchnorm1d_state(child_module, masked)
            masked = masked.to(device=child_module.weight.device if child_module.weight is not None else next(child_module.parameters(), torch.empty(0)).device)
            setattr(module, child_name, masked)
            replaced += 1
        else:
            replaced += _replace_batchnorm1d_with_mask_batchnorm1d(child_module)
    return replaced


def _clip_mask(model: nn.Module, lower: float = 0.0, upper: float = 1.0) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "neuron_mask" in name:
                parameter.clamp_(float(lower), float(upper))


def _train_step_unlearning(
    model: nn.Module,
    *,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_correct = 0
    total_loss = 0.0
    total_samples = 0

    for batch in data_loader:
        model_input, y = parse_torch_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(model_input)
        loss = criterion(logits, y)

        preds = logits.argmax(dim=1)
        total_correct += int((preds == y).sum().item())
        total_loss += float(loss.item())
        total_samples += int(y.shape[0])

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0, norm_type=2)
        (-loss).backward()
        optimizer.step()

    mean_loss = total_loss / max(len(data_loader), 1)
    accuracy = total_correct / max(total_samples, 1)
    return float(mean_loss), float(accuracy)


def _train_step_recovering(
    model: nn.Module,
    *,
    criterion: nn.Module,
    mask_optimizer: torch.optim.Optimizer,
    data_loader,
    device: torch.device,
    alpha: float,
) -> Tuple[float, float]:
    model.train()
    total_correct = 0
    total_loss = 0.0
    total_samples = 0

    for batch in data_loader:
        model_input, y = parse_torch_batch(batch, device)
        mask_optimizer.zero_grad(set_to_none=True)
        logits = model(model_input)
        loss = float(alpha) * criterion(logits, y)

        preds = logits.argmax(dim=1)
        total_correct += int((preds == y).sum().item())
        total_loss += float(loss.item())
        total_samples += int(y.shape[0])

        loss.backward()
        mask_optimizer.step()
        _clip_mask(model)

    mean_loss = total_loss / max(len(data_loader), 1)
    accuracy = total_correct / max(total_samples, 1)
    return float(mean_loss), float(accuracy)


def _evaluate_split(
    model: nn.Module,
    *,
    split: Any,
    device: torch.device,
    batch_size: int,
) -> Tuple[float, float]:
    loader = split_to_dataloader(split, batch_size=int(batch_size), shuffle=False)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_correct = 0
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            model_input, y = parse_torch_batch(batch, device)
            logits = model(model_input)
            loss = criterion(logits, y)
            preds = logits.argmax(dim=1)
            total_loss += float(loss.item())
            total_correct += int((preds == y).sum().item())
            total_samples += int(y.shape[0])

    mean_loss = total_loss / max(len(loader), 1)
    accuracy = total_correct / max(total_samples, 1)
    return float(mean_loss), float(accuracy)


def _predict_labels(
    model: nn.Module,
    *,
    split: Any,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = split_to_dataloader(split, batch_size=int(batch_size), shuffle=False)
    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            model_input, _ = parse_torch_batch(batch, device)
            logits = model(model_input)
            preds.append(logits.argmax(dim=1).detach().cpu().numpy())
    if not preds:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(preds, axis=0).astype(np.int64, copy=False)


def _collect_candidate_metrics(
    model: nn.Module,
    *,
    datasets: Dict[str, Any],
    target_label: int,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    val_clean_loss, val_clean_acc = _evaluate_split(
        model,
        split=datasets["val"],
        device=device,
        batch_size=batch_size,
    )
    test_clean_loss, test_clean_acc = _evaluate_split(
        model,
        split=datasets["test"],
        device=device,
        batch_size=batch_size,
    )

    val_triggered_preds = _predict_labels(
        model,
        split=datasets["val_triggered"],
        device=device,
        batch_size=batch_size,
    )
    test_triggered_preds = _predict_labels(
        model,
        split=datasets["test_triggered"],
        device=device,
        batch_size=batch_size,
    )
    val_backdoor = compute_backdoor_metrics(
        y_pred_triggered=val_triggered_preds,
        y_clean_reference=np.asarray(datasets["val_clean_labels"], dtype=np.int64),
        target_label=int(target_label),
    )
    test_backdoor = compute_backdoor_metrics(
        y_pred_triggered=test_triggered_preds,
        y_clean_reference=np.asarray(datasets["test_clean_labels"], dtype=np.int64),
        target_label=int(target_label),
    )
    return {
        "clean_val_loss": float(val_clean_loss),
        "clean_val_accuracy": float(val_clean_acc),
        "clean_test_loss": float(test_clean_loss),
        "clean_test_accuracy": float(test_clean_acc),
        "val_backdoor_asr": float(val_backdoor["backdoor/asr"]),
        "val_backdoor_accuracy": float(val_backdoor["backdoor/accuracy"]),
        "test_backdoor_asr": float(test_backdoor["backdoor/asr"]),
        "test_backdoor_accuracy": float(test_backdoor["backdoor/accuracy"]),
    }


def _snapshot_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _load_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state_dict, strict=True)


def _select_defense_split(clean_reference_split: Any, *, ratio: float, seed: int) -> Tuple[Any, np.ndarray]:
    total = count_split_samples(clean_reference_split)
    keep = max(1, int(total * float(ratio)))
    keep = min(keep, total)
    rng = np.random.default_rng(int(seed))
    positions = np.sort(rng.choice(total, size=keep, replace=False)).astype(np.int64)
    return subset_split(clean_reference_split, positions), positions


def _save_rows_as_csv(rows: Sequence[Dict[str, Any]], output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(output_path)


def _extract_mask_values(state_dict: Dict[str, torch.Tensor]) -> List[Tuple[str, int, float]]:
    mask_values: List[Tuple[str, int, float]] = []
    for name, param in state_dict.items():
        if "neuron_mask" not in name:
            continue
        layer_name = ".".join(name.split(".")[:-1])
        for idx in range(int(param.shape[0])):
            mask_values.append((layer_name, int(idx), float(param[idx].item())))
    return mask_values


def _save_mask_scores(
    state_dict: Dict[str, torch.Tensor],
    *,
    txt_path: Path,
    csv_path: Path,
) -> List[Tuple[str, int, float]]:
    mask_values = _extract_mask_values(state_dict)
    lines = ["No \t Layer Name \t Neuron Idx \t Mask Score \n"]
    csv_rows = []
    for count, (layer_name, neuron_idx, score) in enumerate(mask_values):
        lines.append(f"{count} \t {layer_name} \t {neuron_idx} \t {score:.4f} \n")
        csv_rows.append(
            {
                "row_index": int(count),
                "layer_name": str(layer_name),
                "neuron_idx": int(neuron_idx),
                "mask_score": float(score),
            }
        )
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.writelines(lines)
    _save_rows_as_csv(csv_rows, csv_path)
    return mask_values


def _prune_neuron(model: nn.Module, neuron: Tuple[str, int, float]) -> None:
    layer_name, neuron_idx, _ = neuron
    module = model.get_submodule(str(layer_name))
    assert hasattr(module, "weight"), f"Pruning target '{layer_name}' has no weight parameter."
    with torch.no_grad():
        module.weight[int(neuron_idx)] = 0.0


def _apply_first_n_prunes(
    model: nn.Module,
    *,
    sorted_mask_values: Sequence[Tuple[str, int, float]],
    num_pruned: int,
) -> None:
    capped = min(int(num_pruned), len(sorted_mask_values))
    for neuron in sorted_mask_values[:capped]:
        _prune_neuron(model, neuron)


def _select_best_pruning_row(
    rows: Sequence[Dict[str, Any]],
    *,
    baseline_clean_val_accuracy: float,
    max_clean_accuracy_drop: float,
) -> Dict[str, Any]:
    """
    Select the pruning candidate in a dynamic-threshold style.

    Following the paper's pruning spirit, keep candidates whose validation clean
    accuracy stays within an acceptable drop of the attacked baseline, then pick
    the most aggressive pruning level among them. Validation ASR is only a
    tie-breaker inside the acceptable clean-accuracy region.
    """

    assert rows, "RNP pruning requires at least one candidate row."
    floor = float(baseline_clean_val_accuracy) - float(max_clean_accuracy_drop)
    admissible = [row for row in rows if float(row["clean_val_accuracy"]) >= floor]
    if not admissible:
        admissible = [max(rows, key=lambda row: float(row["clean_val_accuracy"]))]

    return min(
        admissible,
        key=lambda row: (
            -int(row["num_pruned"]),
            float(row["val_backdoor_asr"]),
            -float(row["clean_val_accuracy"]),
            float(row["selection_value"]),
        ),
    )


class RNPUnlearner(BaseUnlearner):
    """
    Reconstructive Neuron Pruning with official-code-guided unlearn/recover/prune.

    Upstream provenance for this wrapper:
    - repo: https://github.com/bboylyg/RNP
    - commit: eeae192e5eab974d8b3002964cfb62d00388d36f
    - main source: main.py
    - mask layer source: models/mask_batchnorm.py

    Local deviations that are intentional and recorded in the result summary:
    - defense data comes from the clean train reference split already available
      in the local runtime, not from the upstream CIFAR loader stack;
    - this baseline is only benchmark-ready for BatchNorm1d-based `mlp` and
      `resnet` families in the current IDS path;
    - prune-point selection uses validation-side dynamic thresholding rather
      than test metrics: choose the most aggressive pruning candidate whose
      clean validation accuracy stays within a configurable tolerance of the
      attacked baseline.
    """

    SUPPORTED_MODEL_FAMILIES = {"mlp", "resnet"}

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.batch_size = int(self.resolved_cfg.get("batch_size", 128))
        self.momentum = float(self.resolved_cfg.get("momentum", 0.9))
        self.weight_decay = float(self.resolved_cfg.get("weight_decay", 5e-4))
        self.ratio = float(self.resolved_cfg.get("ratio", 0.01))
        self.alpha = float(self.resolved_cfg.get("alpha", 0.2))
        self.clean_threshold = float(self.resolved_cfg.get("clean_threshold", 0.2))
        self.unlearning_lr = float(self.resolved_cfg.get("unlearning_lr", 0.01))
        self.recovering_lr = float(self.resolved_cfg.get("recovering_lr", 0.2))
        self.unlearning_epochs = int(self.resolved_cfg.get("unlearning_epochs", 20))
        self.recovering_epochs = int(self.resolved_cfg.get("recovering_epochs", 20))
        self.schedule = [int(x) for x in self.resolved_cfg.get("schedule", [10, 20])]
        self.pruning_by = str(self.resolved_cfg.get("pruning_by", "threshold")).lower()
        self.pruning_max = float(self.resolved_cfg.get("pruning_max", 0.9))
        self.pruning_step = float(self.resolved_cfg.get("pruning_step", 0.05))
        self.max_clean_accuracy_drop = float(self.resolved_cfg.get("max_clean_accuracy_drop", 0.02))
        self.save_checkpoint = bool(self.resolved_cfg.get("save_checkpoint", True))
        self.save_stage_checkpoints = bool(self.resolved_cfg.get("save_stage_checkpoints", True))

    def _resolve_forget_set(self, context: UnlearningContext) -> ForgetSet:
        return ForgetSet(
            source="none",
            index_space="train_local",
            notes="RNP is a model-edit defense and does not remove a sample-level forget set.",
        )

    def _run_impl(self, context: UnlearningContext, forget_set: ForgetSet) -> UnlearningResult:
        assert context.model_cfg is not None, "RNPUnlearner requires context.model_cfg."
        assert self.batch_size > 0, "rnp batch_size must be > 0."
        assert self.ratio > 0.0, "rnp ratio must be > 0."
        assert self.unlearning_epochs >= 0, "rnp unlearning_epochs must be >= 0."
        assert self.recovering_epochs > 0, "rnp recovering_epochs must be > 0."
        assert self.pruning_by in {"threshold", "number"}, "rnp pruning_by must be threshold or number."
        assert self.max_clean_accuracy_drop >= 0.0, "rnp max_clean_accuracy_drop must be >= 0."

        model_family = str(context.model_family or getattr(context.model, "model_family", "unknown")).lower()
        if model_family not in self.SUPPORTED_MODEL_FAMILIES:
            return UnlearningResult(
                method_name=self.name,
                track_type=self.track_type,
                status="skipped",
                seed=int(context.seed),
                runtime_sec=0.0,
                forget_set_source="rnp_model_edit",
                summary_metrics={
                    "rnp/supported_model_family": 0.0,
                    "rnp/requested_model_family": str(model_family),
                },
                deviation_note=(
                    "RNP currently supports only BatchNorm1d-based mlp/resnet families in the local IDS path; "
                    f"received model_family='{model_family}'."
                ),
            )

        clean_reference_split = context.datasets.get("train_clean_reference")
        assert clean_reference_split is not None, "RNP requires datasets['train_clean_reference'] in the local runtime."
        assert "val_triggered" in context.datasets and "val_clean_labels" in context.datasets, (
            "RNP requires validation triggered data for prune-point selection."
        )

        device = resolve_device(None if context.device is None else str(context.device))
        model_cfg = dict(context.model_cfg)
        target_label = int(context.target_label if context.target_label is not None else 0)

        defense_split, defense_positions = _select_defense_split(
            clean_reference_split,
            ratio=self.ratio,
            seed=int(context.seed),
        )
        defense_loader = split_to_dataloader(
            defense_split,
            batch_size=self.batch_size,
            shuffle=True,
        )

        attacked_model = clone_model(model_cfg, source_model=context.model).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        unlearning_optimizer = torch.optim.SGD(
            attacked_model.parameters(),
            lr=self.unlearning_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            unlearning_optimizer,
            milestones=list(self.schedule),
            gamma=0.1,
        )

        unlearning_trace = []
        selected_unlearned_state: Optional[Dict[str, torch.Tensor]] = None
        clean_threshold_hit = False
        unlearning_epochs_completed = 0
        selected_unlearning_epoch = 0

        for epoch in range(0, self.unlearning_epochs + 1):
            lr = float(unlearning_optimizer.param_groups[0]["lr"])
            defense_loss, defense_acc = _train_step_unlearning(
                attacked_model,
                criterion=criterion,
                optimizer=unlearning_optimizer,
                data_loader=defense_loader,
                device=device,
            )
            clean_val_loss, clean_val_acc = _evaluate_split(
                attacked_model,
                split=context.datasets["val"],
                device=device,
                batch_size=self.batch_size,
            )
            clean_test_loss, clean_test_acc = _evaluate_split(
                attacked_model,
                split=context.datasets["test"],
                device=device,
                batch_size=self.batch_size,
            )
            test_triggered_preds = _predict_labels(
                attacked_model,
                split=context.datasets["test_triggered"],
                device=device,
                batch_size=self.batch_size,
            )
            test_backdoor = compute_backdoor_metrics(
                y_pred_triggered=test_triggered_preds,
                y_clean_reference=np.asarray(context.datasets["test_clean_labels"], dtype=np.int64),
                target_label=target_label,
            )
            scheduler.step()

            clean_threshold_hit = bool(defense_acc <= self.clean_threshold)
            unlearning_epochs_completed = epoch + 1
            selected_unlearning_epoch = epoch
            unlearning_trace.append(
                {
                    "epoch": int(epoch),
                    "lr": float(lr),
                    "defense_loss": float(defense_loss),
                    "defense_accuracy": float(defense_acc),
                    "clean_val_loss": float(clean_val_loss),
                    "clean_val_accuracy": float(clean_val_acc),
                    "clean_test_loss": float(clean_test_loss),
                    "clean_test_accuracy": float(clean_test_acc),
                    "test_backdoor_asr": float(test_backdoor["backdoor/asr"]),
                    "test_backdoor_accuracy": float(test_backdoor["backdoor/accuracy"]),
                    "clean_threshold_hit": float(clean_threshold_hit),
                }
            )
            LOGGER.info(
                "RNP unlearning epoch=%d/%d lr=%.5f defense_loss=%.6f defense_acc=%.4f test_asr=%.4f test_clean_acc=%.4f",
                epoch,
                self.unlearning_epochs,
                lr,
                defense_loss,
                defense_acc,
                float(test_backdoor["backdoor/asr"]),
                clean_test_acc,
            )

            if clean_threshold_hit:
                selected_unlearned_state = _snapshot_state_dict(attacked_model)
                break

        if selected_unlearned_state is None:
            selected_unlearned_state = _snapshot_state_dict(attacked_model)

        unlearned_model = clone_model(model_cfg).to(device)
        _load_state_dict(unlearned_model, selected_unlearned_state)

        artifacts = UnlearningArtifacts()
        unlearning_trace_path = Path(context.run_dir) / "unlearning_trace.json"
        with unlearning_trace_path.open("w", encoding="utf-8") as handle:
            json.dump(unlearning_trace, handle, indent=2)
        artifacts.extra_files["unlearning_trace_json"] = str(unlearning_trace_path)

        unlearned_checkpoint_dir = None
        if self.save_stage_checkpoints:
            unlearned_checkpoint_dir = save_model(
                unlearned_model,
                str(Path(context.run_dir) / "unlearned_model_checkpoint"),
                config={"model_kwargs": model_cfg},
                metadata={
                    "method": "rnp",
                    "phase": "unlearning",
                    "selected_unlearning_epoch": int(selected_unlearning_epoch),
                    "clean_threshold_hit": bool(clean_threshold_hit),
                },
                optimizer=unlearning_optimizer,
                metrics={
                    "rnp/defense_ratio": float(self.ratio),
                    "rnp/defense_size": float(count_split_samples(defense_split)),
                    "rnp/unlearning_epochs_completed": float(unlearning_epochs_completed),
                    "rnp/clean_threshold_hit": float(clean_threshold_hit),
                },
            )
            artifacts.extra_files["unlearned_checkpoint_dir"] = str(unlearned_checkpoint_dir)

        recovering_model = clone_model(model_cfg, source_model=unlearned_model).to(device)
        masked_bn_layers = _replace_batchnorm1d_with_mask_batchnorm1d(recovering_model)
        if masked_bn_layers <= 0:
            return UnlearningResult(
                method_name=self.name,
                track_type=self.track_type,
                status="skipped",
                seed=int(context.seed),
                runtime_sec=0.0,
                forget_set_source="rnp_model_edit",
                artifacts=artifacts,
                summary_metrics={
                    "rnp/supported_model_family": 0.0,
                    "rnp/masked_bn_layers": 0.0,
                },
                deviation_note="RNP requires BatchNorm1d layers, but no replaceable BatchNorm1d module was found.",
            )

        mask_params = [param for name, param in recovering_model.named_parameters() if "neuron_mask" in name]
        assert mask_params, "RNP recovering stage produced no neuron_mask parameters."
        mask_optimizer = torch.optim.SGD(mask_params, lr=self.recovering_lr, momentum=self.momentum)

        recovering_trace = []
        for epoch in range(1, self.recovering_epochs + 1):
            defense_loss, defense_acc = _train_step_recovering(
                recovering_model,
                criterion=criterion,
                mask_optimizer=mask_optimizer,
                data_loader=defense_loader,
                device=device,
                alpha=self.alpha,
            )
            candidate_metrics = _collect_candidate_metrics(
                recovering_model,
                datasets=context.datasets,
                target_label=target_label,
                device=device,
                batch_size=self.batch_size,
            )
            recovering_trace.append(
                {
                    "epoch": int(epoch),
                    "lr": float(mask_optimizer.param_groups[0]["lr"]),
                    "defense_loss": float(defense_loss),
                    "defense_accuracy": float(defense_acc),
                    **candidate_metrics,
                }
            )
            LOGGER.info(
                "RNP recovering epoch=%d/%d defense_loss=%.6f defense_acc=%.4f val_asr=%.4f val_clean_acc=%.4f",
                epoch,
                self.recovering_epochs,
                defense_loss,
                defense_acc,
                float(candidate_metrics["val_backdoor_asr"]),
                float(candidate_metrics["clean_val_accuracy"]),
            )

        recovering_trace_path = Path(context.run_dir) / "recovering_trace.json"
        with recovering_trace_path.open("w", encoding="utf-8") as handle:
            json.dump(recovering_trace, handle, indent=2)
        artifacts.extra_files["recovering_trace_json"] = str(recovering_trace_path)

        recovered_checkpoint_dir = None
        if self.save_stage_checkpoints:
            recovered_checkpoint_dir = save_model(
                recovering_model,
                str(Path(context.run_dir) / "recovering_model_checkpoint"),
                config={"model_kwargs": model_cfg},
                metadata={
                    "method": "rnp",
                    "phase": "recovering",
                    "alpha": float(self.alpha),
                    "masked_bn_layers": int(masked_bn_layers),
                },
                optimizer=mask_optimizer,
                metrics={
                    "rnp/recovering_epochs_completed": float(self.recovering_epochs),
                    "rnp/masked_bn_layers": float(masked_bn_layers),
                },
            )
            artifacts.extra_files["recovering_checkpoint_dir"] = str(recovered_checkpoint_dir)

        mask_txt_path = Path(context.run_dir) / "mask_values.txt"
        mask_csv_path = Path(context.run_dir) / "mask_values.csv"
        mask_values = _save_mask_scores(
            recovering_model.state_dict(),
            txt_path=mask_txt_path,
            csv_path=mask_csv_path,
        )
        artifacts.extra_files["mask_values_txt"] = str(mask_txt_path)
        artifacts.extra_files["mask_values_csv"] = str(mask_csv_path)
        sorted_mask_values = sorted(mask_values, key=lambda value: float(value[2]))

        pruning_rows: List[Dict[str, Any]] = []
        baseline_model = clone_model(model_cfg, source_model=context.model).to(device)
        baseline_metrics = _collect_candidate_metrics(
            baseline_model,
            datasets=context.datasets,
            target_label=target_label,
            device=device,
            batch_size=self.batch_size,
        )
        pruning_rows.append(
            {
                "pruning_by": str(self.pruning_by),
                "selection_value": 0.0,
                "num_pruned": 0,
                "last_layer_name": "",
                "last_neuron_idx": -1,
                "last_mask_score": 0.0,
                **baseline_metrics,
            }
        )

        if self.pruning_by == "threshold":
            pruning_model = clone_model(model_cfg, source_model=context.model).to(device)
            num_pruned = 0
            thresholds = np.arange(0.0, self.pruning_max + self.pruning_step, self.pruning_step, dtype=np.float64)
            for threshold in thresholds:
                while num_pruned < len(sorted_mask_values) and float(sorted_mask_values[num_pruned][2]) <= float(threshold):
                    _prune_neuron(pruning_model, sorted_mask_values[num_pruned])
                    num_pruned += 1
                last_neuron = sorted_mask_values[num_pruned - 1] if num_pruned > 0 else ("", -1, 0.0)
                candidate_metrics = _collect_candidate_metrics(
                    pruning_model,
                    datasets=context.datasets,
                    target_label=target_label,
                    device=device,
                    batch_size=self.batch_size,
                )
                pruning_rows.append(
                    {
                        "pruning_by": "threshold",
                        "selection_value": float(threshold),
                        "num_pruned": int(num_pruned),
                        "last_layer_name": str(last_neuron[0]),
                        "last_neuron_idx": int(last_neuron[1]),
                        "last_mask_score": float(last_neuron[2]),
                        **candidate_metrics,
                    }
                )
        else:
            pruning_model = clone_model(model_cfg, source_model=context.model).to(device)
            step = max(1, int(np.ceil(self.pruning_step)))
            max_pruned = min(len(sorted_mask_values), int(np.ceil(self.pruning_max)))
            num_pruned = 0
            while num_pruned < max_pruned:
                next_pruned = min(num_pruned + step, max_pruned)
                for prune_idx in range(num_pruned, next_pruned):
                    _prune_neuron(pruning_model, sorted_mask_values[prune_idx])
                num_pruned = next_pruned
                last_neuron = sorted_mask_values[num_pruned - 1]
                candidate_metrics = _collect_candidate_metrics(
                    pruning_model,
                    datasets=context.datasets,
                    target_label=target_label,
                    device=device,
                    batch_size=self.batch_size,
                )
                pruning_rows.append(
                    {
                        "pruning_by": "number",
                        "selection_value": float(num_pruned),
                        "num_pruned": int(num_pruned),
                        "last_layer_name": str(last_neuron[0]),
                        "last_neuron_idx": int(last_neuron[1]),
                        "last_mask_score": float(last_neuron[2]),
                        **candidate_metrics,
                    }
                )

        pruning_curve_path = Path(context.run_dir) / "pruning_curve.csv"
        _save_rows_as_csv(pruning_rows, pruning_curve_path)
        artifacts.extra_files["pruning_curve_csv"] = str(pruning_curve_path)

        baseline_clean_val_accuracy = float(pruning_rows[0]["clean_val_accuracy"])
        selected_row = _select_best_pruning_row(
            pruning_rows,
            baseline_clean_val_accuracy=baseline_clean_val_accuracy,
            max_clean_accuracy_drop=self.max_clean_accuracy_drop,
        )
        selected_prune_point_path = Path(context.run_dir) / "selected_prune_point.json"
        with selected_prune_point_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    **selected_row,
                    "selection_rule": "validation_dynamic_threshold",
                    "baseline_clean_val_accuracy": float(baseline_clean_val_accuracy),
                    "max_clean_accuracy_drop": float(self.max_clean_accuracy_drop),
                    "selected_clean_val_drop": float(
                        baseline_clean_val_accuracy - float(selected_row["clean_val_accuracy"])
                    ),
                },
                handle,
                indent=2,
            )
        artifacts.extra_files["selected_prune_point_json"] = str(selected_prune_point_path)

        final_model = clone_model(model_cfg, source_model=context.model).to(device)
        _apply_first_n_prunes(
            final_model,
            sorted_mask_values=sorted_mask_values,
            num_pruned=int(selected_row["num_pruned"]),
        )

        checkpoint_dir = None
        if self.save_checkpoint:
            checkpoint_dir = save_model(
                final_model,
                str(Path(context.run_dir) / "checkpoint"),
                config={"model_kwargs": model_cfg},
                metadata={
                    "method": "rnp",
                    "phase": "pruned_final",
                    "pruning_by": self.pruning_by,
                },
                metrics={
                    "rnp/selected_num_pruned": float(selected_row["num_pruned"]),
                    "rnp/selected_selection_value": float(selected_row["selection_value"]),
                    "rnp/selected_val_asr": float(selected_row["val_backdoor_asr"]),
                    "rnp/selected_val_clean_accuracy": float(selected_row["clean_val_accuracy"]),
                },
            )

        deviation_parts = [
            "Official RNP unlearn/recover/prune flow mirrored from author code.",
            "Local wrapper uses clean train reference data from the shared runtime instead of the upstream CIFAR loader stack.",
            "This baseline is benchmark-ready only for BatchNorm1d-based mlp/resnet families in the current IDS path.",
            "Prune-point selection uses validation-side dynamic thresholding with a clean-accuracy drop tolerance to stay close to the paper's pruning protocol while avoiding test leakage.",
        ]
        if not clean_threshold_hit:
            deviation_parts.append(
                "The official clean-threshold stop was not reached, so the last unlearned checkpoint was used for recovering."
            )

        summary_metrics: Dict[str, Any] = {
            "rnp/defense_size": float(count_split_samples(defense_split)),
            "rnp/defense_ratio": float(self.ratio),
            "rnp/clean_reference_size": float(count_split_samples(clean_reference_split)),
            "rnp/clean_threshold": float(self.clean_threshold),
            "rnp/clean_threshold_hit": float(clean_threshold_hit),
            "rnp/unlearning_epochs_budget": float(self.unlearning_epochs),
            "rnp/unlearning_epochs_completed": float(unlearning_epochs_completed),
            "rnp/recovering_epochs_completed": float(self.recovering_epochs),
            "rnp/unlearning_lr": float(self.unlearning_lr),
            "rnp/recovering_lr": float(self.recovering_lr),
            "rnp/alpha": float(self.alpha),
            "rnp/masked_bn_layers": float(masked_bn_layers),
            "rnp/mask_values_count": float(len(sorted_mask_values)),
            "rnp/pruning_rows_evaluated": float(len(pruning_rows)),
            "rnp/pruning_mode_is_threshold": float(self.pruning_by == "threshold"),
            "rnp/pruning_max": float(self.pruning_max),
            "rnp/pruning_step": float(self.pruning_step),
            "rnp/pruning_selection_baseline_clean_val_accuracy": float(baseline_clean_val_accuracy),
            "rnp/pruning_selection_max_clean_accuracy_drop": float(self.max_clean_accuracy_drop),
            "rnp/selected_num_pruned": float(selected_row["num_pruned"]),
            "rnp/selected_selection_value": float(selected_row["selection_value"]),
            "rnp/selected_val_asr": float(selected_row["val_backdoor_asr"]),
            "rnp/selected_val_backdoor_accuracy": float(selected_row["val_backdoor_accuracy"]),
            "rnp/selected_val_clean_accuracy": float(selected_row["clean_val_accuracy"]),
            "rnp/selected_val_clean_accuracy_drop": float(
                baseline_clean_val_accuracy - float(selected_row["clean_val_accuracy"])
            ),
            "rnp/selected_test_asr_curve": float(selected_row["test_backdoor_asr"]),
            "rnp/selected_test_backdoor_accuracy_curve": float(selected_row["test_backdoor_accuracy"]),
            "rnp/selected_test_clean_accuracy_curve": float(selected_row["clean_test_accuracy"]),
        }

        defense_indices_path = Path(context.run_dir) / "defense_subset_indices.npy"
        np.save(defense_indices_path, defense_positions.astype(np.int64, copy=False))
        artifacts.extra_files["defense_subset_indices_npy"] = str(defense_indices_path)

        return UnlearningResult(
            method_name=self.name,
            track_type=self.track_type,
            status="ok",
            seed=int(context.seed),
            runtime_sec=0.0,
            forget_set_source="rnp_model_edit",
            removed_indices=np.empty((0,), dtype=np.int64),
            summary_metrics=summary_metrics,
            checkpoint_dir=checkpoint_dir,
            model_after=final_model,
            artifacts=artifacts,
            deviation_note=" ".join(deviation_parts),
        )


__all__ = ["MaskBatchNorm1d", "RNPUnlearner"]
