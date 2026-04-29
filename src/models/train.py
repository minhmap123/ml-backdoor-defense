from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report

from ..utils.logging import get_logger, log_metrics, update_wandb_config
from .utils import parse_torch_batch, resolve_device, save_model, set_seed, split_to_dataloader


LOGGER = get_logger(__name__)


def _scalar_metrics_only(metrics: Dict[str, Any]) -> Dict[str, float]:
    scalar_metrics: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            scalar_metrics[key] = float(value)
    return scalar_metrics


def _evaluate_torch_model(
    model: nn.Module,
    *,
    loader,
    device: torch.device,
    prefix: str,
) -> Dict[str, Any]:
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0.0
    total_batches = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            model_input, y = parse_torch_batch(batch, device)
            logits = model(model_input)
            if hasattr(model, "compute_eval_loss"):
                loss = model.compute_eval_loss(logits, y)
            else:
                loss = criterion(logits, y)
            preds = torch.argmax(logits, dim=1)

            y_true.append(y.detach().cpu().numpy())
            y_pred.append(preds.detach().cpu().numpy())
            total_loss += float(loss.item())
            total_batches += 1

    y_true_np = np.concatenate(y_true) if y_true else np.array([], dtype=np.int64)
    y_pred_np = np.concatenate(y_pred) if y_pred else np.array([], dtype=np.int64)
    avg_loss = total_loss / max(total_batches, 1)
    acc = float(accuracy_score(y_true_np, y_pred_np)) if y_true_np.size > 0 else 0.0

    metrics: Dict[str, Any] = {
        f"{prefix}/loss": float(avg_loss),
        f"{prefix}/accuracy": float(acc),
    }

    if prefix.startswith("clean") and y_true_np.size > 0:
        report = classification_report(y_true_np, y_pred_np, output_dict=True, zero_division=0)
        metrics[f"{prefix}/classification_report"] = report

    return metrics


def compute_backdoor_metrics(
    *,
    y_pred_triggered: np.ndarray,
    y_clean_reference: np.ndarray,
    target_label: int,
) -> Dict[str, float]:
    y_pred_triggered = np.asarray(y_pred_triggered, dtype=np.int64)
    y_clean_reference = np.asarray(y_clean_reference, dtype=np.int64)

    non_target_mask = y_clean_reference != int(target_label)
    non_target_count = int(non_target_mask.sum())

    if non_target_count == 0:
        asr = 0.0
    else:
        asr = float((y_pred_triggered[non_target_mask] == int(target_label)).mean())

    backdoor_acc = float((y_pred_triggered == int(target_label)).mean()) if y_pred_triggered.size > 0 else 0.0
    return {
        "backdoor/asr": asr,
        "backdoor/accuracy": backdoor_acc,
    }


def train_torch_model(
    model: nn.Module,
    *,
    datasets: Dict[str, Any],
    model_cfg: Dict[str, Any],
    train_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    train_cfg = train_cfg or {}

    device = resolve_device(train_cfg.get("device", model_cfg.get("device", "auto")))
    model = model.to(device)

    epochs = int(model_cfg.get("epochs", train_cfg.get("epochs", 10)))
    batch_size = int(model_cfg.get("batch_size", train_cfg.get("batch_size", 256)))
    learning_rate = float(model_cfg.get("learning_rate", train_cfg.get("learning_rate", 1e-3)))
    weight_decay = float(model_cfg.get("weight_decay", train_cfg.get("weight_decay", 0.0)))
    patience = int(model_cfg.get("patience", train_cfg.get("patience", 0)))
    save_dir = str(model_cfg.get("save_dir", train_cfg.get("save_dir", "artifacts/models")))

    seed = model_cfg.get("seed", train_cfg.get("seed", None))
    set_seed(seed)

    train_loader = split_to_dataloader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = split_to_dataloader(datasets["val"], batch_size=batch_size, shuffle=False)
    test_loader = split_to_dataloader(datasets["test"], batch_size=batch_size, shuffle=False)

    if hasattr(model, "make_parameter_groups"):
        optimizer = torch.optim.AdamW(
            model.make_parameter_groups(weight_decay=weight_decay),
            lr=learning_rate,
            weight_decay=0.0,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    metadata = model.get_model_metadata() if hasattr(model, "get_model_metadata") else {"name": model.__class__.__name__}
    total_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable_parameters = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    inactive_parameters = int(total_parameters - trainable_parameters)

    update_wandb_config(
        {
            "train_runtime": {
                "device": str(device),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "patience": patience,
                "seed": seed,
            },
            "model_runtime": {
                "name": model.__class__.__name__,
                "total_parameters": total_parameters,
                "active_parameters": trainable_parameters,
                "trainable_parameters": trainable_parameters,
                "inactive_parameters": inactive_parameters,
                **metadata,
            },
        }
    )
    LOGGER.info(
        "Training %s on %s for %d epochs (batch_size=%d, lr=%g, weight_decay=%g, params=%d)",
        model.__class__.__name__,
        device,
        epochs,
        batch_size,
        learning_rate,
        weight_decay,
        total_parameters,
    )
    if metadata.get("model_family") == "saint":
        LOGGER.info(
            "SAINT assumption: fully observed features in local pipeline; dataset-level missing masks are not passed into the wrapper."
        )
    elif metadata.get("model_family") == "tabnet":
        LOGGER.info(
            "TabNet config: n_d=%s, n_a=%s, n_steps=%s, gamma=%s, lambda_sparse=%s, mask_type=%s",
            metadata.get("n_d"),
            metadata.get("n_a"),
            metadata.get("n_steps"),
            metadata.get("gamma"),
            metadata.get("lambda_sparse"),
            metadata.get("mask_type"),
        )

    best_state = None
    best_val_acc = -1.0
    bad_epochs = 0
    completed_epochs = 0

    for epoch in range(epochs):
        completed_epochs = epoch + 1
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            model_input, y = parse_torch_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(model_input)
            if hasattr(model, "compute_training_loss"):
                loss = model.compute_training_loss(logits, y)
            else:
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)

        val_metrics = _evaluate_torch_model(model, loader=val_loader, device=device, prefix="clean/val")
        val_acc = float(val_metrics["clean/val/accuracy"])

        epoch_metrics = {
            "epoch": epoch + 1,
            "clean/train_loss": train_loss,
            **_scalar_metrics_only(val_metrics),
        }
        if hasattr(model, "get_training_step_metrics"):
            epoch_metrics.update(_scalar_metrics_only(model.get_training_step_metrics()))
        log_metrics(epoch_metrics, step=epoch + 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = _evaluate_torch_model(model, loader=test_loader, device=device, prefix="clean/test")

    final_metrics: Dict[str, Any] = {
        "clean/val_accuracy_best": float(best_val_acc),
        **test_metrics,
    }

    # Optional backdoor evaluation if provided by caller
    if "test_triggered" in datasets and "test_clean_labels" in datasets:
        trig_loader = split_to_dataloader(datasets["test_triggered"], batch_size=batch_size, shuffle=False)
        model.eval()
        trig_preds = []
        with torch.no_grad():
            for batch in trig_loader:
                model_input, _ = parse_torch_batch(batch, device)
                logits = model(model_input)
                trig_preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        if trig_preds:
            y_pred_triggered = np.concatenate(trig_preds)
            y_clean_reference = np.asarray(datasets["test_clean_labels"], dtype=np.int64)
            target_label = int(model_cfg.get("target_label", train_cfg.get("target_label", 0)))
            final_metrics.update(
                compute_backdoor_metrics(
                    y_pred_triggered=y_pred_triggered,
                    y_clean_reference=y_clean_reference,
                    target_label=target_label,
                )
            )

    log_metrics(
        {
            "train/epochs_completed": float(completed_epochs),
            "model/total_parameters": float(total_parameters),
            "model/active_parameters": float(trainable_parameters),
            "model/trainable_parameters": float(trainable_parameters),
            "model/inactive_parameters": float(inactive_parameters),
            **_scalar_metrics_only(final_metrics),
        },
        step=completed_epochs or None,
    )

    final_metrics["model/total_parameters"] = float(total_parameters)
    final_metrics["model/active_parameters"] = float(trainable_parameters)
    final_metrics["model/trainable_parameters"] = float(trainable_parameters)
    final_metrics["model/inactive_parameters"] = float(inactive_parameters)
    final_metrics["train/epochs_completed"] = float(completed_epochs)

    checkpoint_dir = save_model(
        model,
        save_dir,
        config={"model_kwargs": dict(model_cfg)},
        metadata=metadata,
        optimizer=optimizer,
        metrics=final_metrics,
    )
    final_metrics["checkpoint_dir"] = checkpoint_dir

    return model, final_metrics
