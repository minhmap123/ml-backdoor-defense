from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from ..models.utils import parse_torch_batch, resolve_device, save_model, split_to_dataloader
from ..utils.logging import get_logger
from .bad_teaching.utils import clone_model
from .base import BaseUnlearner
from .types import ForgetSet, UnlearningArtifacts, UnlearningContext, UnlearningResult
from .utils import complement_indices, count_split_samples, subset_split


LOGGER = get_logger("unlearning.abl")


class ABLUnlearner(BaseUnlearner):
    """
    Anti-Backdoor Learning with official-code-guided isolation + ascent repair.

    Upstream provenance for this wrapper:
    - repo: https://github.com/bboylyg/ABL
    - commit: 3fd736ce0906f24b0672b471ae20016362eefde4
    - stage-1 source: backdoor_isolation.py
    - stage-2 source: backdoor_unlearning.py

    Upstream behavior mirrored here:
    - stage 1: train an isolation model with gradient-ascent-style tuning on the
      attacked training split, then rank examples by ascending per-sample loss;
    - stage 2: continue from the isolation model, optionally fine-tune on the
      non-isolated pool, then run gradient ascent on the isolated subset to
      suppress backdoor behavior.

    Local deviations that are intentional and recorded in the result summary:
    - local tabular datasets / model wrappers replace the upstream CIFAR stack;
    - stage2_init="attacked_model" is still exposed only as a local ablation,
      but the default now follows the official two-stage isolation-model flow.
    - when the upstream stage-2 scripts disagree, this wrapper treats
      backdoor_unlearning.py as the canonical method structure and treats
      quick_unlearning_demo.py as a demo-only variant;
    - the main script's epoch-0 eval-only quirk is not copied literally:
      local unlearning_epochs means actual ascent-update epochs.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.batch_size = int(self.resolved_cfg.get("batch_size", 128))
        self.per_example_loss_batch_size = int(
            self.resolved_cfg.get("per_example_loss_batch_size", self.batch_size)
        )
        self.optimizer_name = str(self.resolved_cfg.get("optimizer", "sgd")).lower()
        self.momentum = float(self.resolved_cfg.get("momentum", 0.9))
        self.weight_decay = float(self.resolved_cfg.get("weight_decay", 1e-4))
        self.num_workers = int(self.resolved_cfg.get("num_workers", 0))
        self.pin_memory = bool(self.resolved_cfg.get("pin_memory", False))

        self.tuning_epochs = int(self.resolved_cfg.get("tuning_epochs", 10))
        self.tuning_lr = float(self.resolved_cfg.get("tuning_lr", self.resolved_cfg.get("lr", 0.1)))
        self.gradient_ascent_type = str(
            self.resolved_cfg.get("gradient_ascent_type", "Flooding")
        ).lower()
        self.gamma = float(self.resolved_cfg.get("gamma", 0.5))
        self.flooding = float(self.resolved_cfg.get("flooding", 0.5))
        self.isolation_ratio = float(self.resolved_cfg.get("isolation_ratio", 0.01))
        self.stage1_init = str(self.resolved_cfg.get("stage1_init", "fresh_model")).lower()

        self.finetuning_ascent_model = bool(self.resolved_cfg.get("finetuning_ascent_model", True))
        self.finetuning_epochs = int(self.resolved_cfg.get("finetuning_epochs", 60))
        self.finetuning_lr_init = float(
            self.resolved_cfg.get("finetuning_lr_init", self.resolved_cfg.get("lr_finetuning_init", 0.1))
        )
        self.unlearning_epochs = int(self.resolved_cfg.get("unlearning_epochs", 5))
        self.unlearning_lr = float(
            self.resolved_cfg.get("unlearning_lr", self.resolved_cfg.get("lr_unlearning_init", 5e-4))
        )
        self.stage2_init = str(self.resolved_cfg.get("stage2_init", "isolation_model")).lower()

        self.save_checkpoint = bool(self.resolved_cfg.get("save_checkpoint", True))
        self.save_isolation_checkpoint = bool(self.resolved_cfg.get("save_isolation_checkpoint", True))

    def _resolve_forget_set(self, context: UnlearningContext) -> ForgetSet:
        return ForgetSet(
            source="none",
            index_space="train_local",
            notes="ABL isolates suspicious samples internally from per-sample loss ranking.",
        )

    def _run_impl(self, context: UnlearningContext, forget_set: ForgetSet) -> UnlearningResult:
        assert context.model_cfg is not None, "ABLUnlearner requires context.model_cfg."
        assert self.tuning_epochs > 0, "abl tuning_epochs must be > 0."
        assert self.unlearning_epochs > 0, "abl unlearning_epochs must be > 0."
        assert 0.0 < self.isolation_ratio < 1.0, "abl isolation_ratio must be in (0, 1)."
        assert self.stage1_init in {"fresh_model", "attacked_model"}
        assert self.stage2_init in {"attacked_model", "isolation_model"}
        assert self.gradient_ascent_type in {"flooding", "lga"}

        device = resolve_device(None if context.device is None else str(context.device))
        model_cfg = dict(context.model_cfg)
        train_split = context.datasets["train"]
        num_train = count_split_samples(train_split)

        isolation_model = clone_model(
            model_cfg,
            source_model=context.model if self.stage1_init == "attacked_model" else None,
        ).to(device)
        isolation_optimizer = self._build_optimizer(isolation_model, lr=self.tuning_lr)
        train_loader = split_to_dataloader(
            train_split,
            batch_size=self.batch_size,
            shuffle=False,
        )

        isolation_trace = []
        for epoch in range(self.tuning_epochs):
            self._set_optimizer_lr(isolation_optimizer, self.tuning_lr)
            epoch_metrics = self._run_epoch(
                model=isolation_model,
                data_loader=train_loader,
                optimizer=isolation_optimizer,
                device=device,
                mode=self.gradient_ascent_type,
            )
            isolation_trace.append(
                {
                    "epoch": epoch + 1,
                    "phase": "isolation_tuning",
                    "lr": float(self.tuning_lr),
                    **epoch_metrics,
                }
            )
            LOGGER.info(
                "ABL isolation epoch=%d/%d objective=%s ce_loss=%.6f acc=%.4f",
                epoch + 1,
                self.tuning_epochs,
                self.gradient_ascent_type,
                float(epoch_metrics["ce_loss"]),
                float(epoch_metrics["accuracy"]),
            )

        sample_losses = self._compute_per_example_losses(
            model=isolation_model,
            split=train_split,
            device=device,
        )
        loss_ranking = np.argsort(sample_losses, kind="stable").astype(np.int64)
        isolated_indices = self._select_isolated_indices(loss_ranking, num_train=num_train)
        other_indices = complement_indices(num_train, isolated_indices)
        isolated_split = subset_split(train_split, isolated_indices)
        other_split = subset_split(train_split, other_indices)

        if self.stage2_init == "isolation_model":
            repair_model = clone_model(model_cfg, source_model=isolation_model).to(device)
        else:
            repair_model = clone_model(model_cfg, source_model=context.model).to(device)

        repair_trace = []
        finetuning_epochs_completed = 0
        if self.finetuning_ascent_model and self.finetuning_epochs > 0 and len(other_indices) > 0:
            finetune_optimizer = self._build_optimizer(repair_model, lr=self.finetuning_lr_init)
            finetune_loader = split_to_dataloader(
                other_split,
                batch_size=self.batch_size,
                shuffle=True,
            )
            for epoch in range(self.finetuning_epochs):
                finetune_lr = self._finetuning_lr(epoch)
                self._set_optimizer_lr(finetune_optimizer, finetune_lr)
                epoch_metrics = self._run_epoch(
                    model=repair_model,
                    data_loader=finetune_loader,
                    optimizer=finetune_optimizer,
                    device=device,
                    mode="standard",
                )
                repair_trace.append(
                    {
                        "epoch": epoch + 1,
                        "phase": "repair_finetuning",
                        "lr": float(finetune_lr),
                        **epoch_metrics,
                    }
                )
                finetuning_epochs_completed = epoch + 1
                LOGGER.info(
                    "ABL finetune epoch=%d/%d ce_loss=%.6f acc=%.4f",
                    epoch + 1,
                    self.finetuning_epochs,
                    float(epoch_metrics["ce_loss"]),
                    float(epoch_metrics["accuracy"]),
                )

        unlearning_optimizer = self._build_optimizer(repair_model, lr=self.unlearning_lr)
        isolated_loader = split_to_dataloader(
            isolated_split,
            batch_size=self.batch_size,
            shuffle=True,
        )
        # The main upstream script evaluates once before the first ascent step,
        # while the quick demo trains immediately. We resolve that ambiguity by
        # using unlearning_epochs as the number of actual ascent-update epochs.
        unlearning_epochs_completed = 0
        for epoch in range(self.unlearning_epochs):
            self._set_optimizer_lr(unlearning_optimizer, self.unlearning_lr)
            epoch_metrics = self._run_epoch(
                model=repair_model,
                data_loader=isolated_loader,
                optimizer=unlearning_optimizer,
                device=device,
                mode="negative_ce",
            )
            repair_trace.append(
                {
                    "epoch": epoch + 1,
                    "phase": "repair_unlearning",
                    "lr": float(self.unlearning_lr),
                    **epoch_metrics,
                }
            )
            unlearning_epochs_completed = epoch + 1
            LOGGER.info(
                "ABL unlearning epoch=%d/%d ce_loss=%.6f acc=%.4f",
                epoch + 1,
                self.unlearning_epochs,
                float(epoch_metrics["ce_loss"]),
                float(epoch_metrics["accuracy"]),
            )

        artifacts = UnlearningArtifacts()
        loss_path = Path(context.run_dir) / "sample_losses.npy"
        ranking_path = Path(context.run_dir) / "loss_ranking.npy"
        isolation_trace_path = Path(context.run_dir) / "isolation_trace.json"
        repair_trace_path = Path(context.run_dir) / "repair_trace.json"
        np.save(loss_path, sample_losses.astype(np.float32, copy=False))
        np.save(ranking_path, loss_ranking)
        with isolation_trace_path.open("w", encoding="utf-8") as handle:
            json.dump(isolation_trace, handle, indent=2)
        with repair_trace_path.open("w", encoding="utf-8") as handle:
            json.dump(repair_trace, handle, indent=2)
        artifacts.extra_files.update(
            {
                "sample_losses_npy": str(loss_path),
                "loss_ranking_npy": str(ranking_path),
                "isolation_trace_json": str(isolation_trace_path),
                "repair_trace_json": str(repair_trace_path),
            }
        )

        isolation_checkpoint_dir = None
        if self.save_isolation_checkpoint:
            isolation_checkpoint_dir = save_model(
                isolation_model,
                str(Path(context.run_dir) / "isolation_model_checkpoint"),
                config={"model_kwargs": model_cfg},
                metadata={
                    "method": "abl",
                    "phase": "isolation",
                    "gradient_ascent_type": self.gradient_ascent_type,
                    "stage1_init": self.stage1_init,
                },
                optimizer=isolation_optimizer,
                metrics={
                    "abl/num_isolated": float(len(isolated_indices)),
                    "abl/num_other": float(len(other_indices)),
                    "abl/isolation_ratio": float(self.isolation_ratio),
                },
            )
            artifacts.extra_files["isolation_checkpoint_dir"] = isolation_checkpoint_dir

        checkpoint_dir = None
        if self.save_checkpoint:
            checkpoint_dir = save_model(
                repair_model,
                str(Path(context.run_dir) / "checkpoint"),
                config={"model_kwargs": model_cfg},
                metadata={
                    "method": "abl",
                    "phase": "repair",
                    "stage2_init": self.stage2_init,
                    "gradient_ascent_type": self.gradient_ascent_type,
                },
                optimizer=unlearning_optimizer,
                metrics={
                    "abl/num_isolated": float(len(isolated_indices)),
                    "abl/num_other": float(len(other_indices)),
                    "abl/tuning_epochs_completed": float(self.tuning_epochs),
                    "abl/finetuning_epochs_completed": float(finetuning_epochs_completed),
                    "abl/unlearning_epochs_completed": float(unlearning_epochs_completed),
                },
            )

        summary_metrics: Dict[str, Any] = {
            "abl/num_isolated": float(len(isolated_indices)),
            "abl/num_other": float(len(other_indices)),
            "abl/isolation_ratio": float(self.isolation_ratio),
            "abl/tuning_epochs_completed": float(self.tuning_epochs),
            "abl/finetuning_epochs_completed": float(finetuning_epochs_completed),
            "abl/unlearning_epochs_completed": float(unlearning_epochs_completed),
            "abl/tuning_lr": float(self.tuning_lr),
            "abl/unlearning_lr": float(self.unlearning_lr),
            "abl/finetuning_lr_init": float(self.finetuning_lr_init),
            "abl/flooding": float(self.flooding),
            "abl/gamma": float(self.gamma),
            "abl/stage1_init_is_attacked_model": float(self.stage1_init == "attacked_model"),
            "abl/stage2_init_is_attacked_model": float(self.stage2_init == "attacked_model"),
            "abl/finetuning_enabled": float(bool(self.finetuning_ascent_model)),
            "abl/uses_plain_cross_entropy": 1.0,
            "abl/min_isolated_loss": float(sample_losses[isolated_indices[0]]) if len(isolated_indices) > 0 else 0.0,
            "abl/max_isolated_loss": float(sample_losses[isolated_indices[-1]]) if len(isolated_indices) > 0 else 0.0,
            "abl/min_other_loss": float(sample_losses[other_indices].min()) if len(other_indices) > 0 else 0.0,
            "abl/mean_isolated_loss": float(sample_losses[isolated_indices].mean()) if len(isolated_indices) > 0 else 0.0,
            "abl/mean_other_loss": float(sample_losses[other_indices].mean()) if len(other_indices) > 0 else 0.0,
        }
        if isolation_checkpoint_dir is not None:
            summary_metrics["abl/isolation_checkpoint_saved"] = 1.0

        deviation_note = (
            "Official ABL isolation and ascent logic mirrored from author code. "
            "Local wrapper uses repo tabular datasets/models instead of the upstream CIFAR pipeline, "
            f"optional stage-2 initialization from '{self.stage2_init}' is exposed through the shared runtime config, "
            "and unlearning_epochs counts actual ascent-update epochs despite the upstream stage-2 warmup-loop ambiguity."
        )

        return UnlearningResult(
            method_name=self.name,
            track_type=self.track_type,
            status="ok",
            seed=int(context.seed),
            runtime_sec=0.0,
            forget_set_source="abl_internal_isolation",
            removed_indices=isolated_indices,
            retain_indices=other_indices,
            summary_metrics=summary_metrics,
            checkpoint_dir=checkpoint_dir,
            model_after=repair_model,
            artifacts=artifacts,
            deviation_note=deviation_note,
        )

    def _build_optimizer(self, model: nn.Module, *, lr: float) -> torch.optim.Optimizer:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(
                parameters,
                lr=float(lr),
                momentum=float(self.momentum),
                weight_decay=float(self.weight_decay),
                nesterov=True,
            )
        if self.optimizer_name == "adam":
            return torch.optim.Adam(parameters, lr=float(lr), weight_decay=float(self.weight_decay))
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(parameters, lr=float(lr), weight_decay=float(self.weight_decay))
        raise AssertionError(f"Unsupported abl optimizer: {self.optimizer_name}")

    def _set_optimizer_lr(self, optimizer: torch.optim.Optimizer, lr: float) -> None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = float(lr)

    def _finetuning_lr(self, epoch: int) -> float:
        if epoch < 40:
            return float(self.finetuning_lr_init)
        if epoch < 60:
            return 0.01
        return 0.001

    def _run_epoch(
        self,
        *,
        model: nn.Module,
        data_loader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        mode: str,
    ) -> Dict[str, float]:
        criterion = nn.CrossEntropyLoss()
        model.train()
        total_ce_loss = 0.0
        total_objective_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in data_loader:
            model_input, y = parse_torch_batch(batch, device)
            logits = model(model_input)
            ce_loss = criterion(logits, y)

            if mode == "flooding":
                objective_loss = (ce_loss - float(self.flooding)).abs() + float(self.flooding)
                backward_loss = objective_loss
            elif mode == "lga":
                objective_loss = torch.sign(ce_loss.detach() - float(self.gamma)) * ce_loss
                backward_loss = objective_loss
            elif mode == "negative_ce":
                objective_loss = -ce_loss
                backward_loss = objective_loss
            elif mode == "standard":
                objective_loss = ce_loss
                backward_loss = objective_loss
            else:
                raise AssertionError(f"Unsupported abl epoch mode: {mode}")

            optimizer.zero_grad()
            backward_loss.backward()
            optimizer.step()

            batch_size = int(y.shape[0])
            total_samples += batch_size
            total_ce_loss += float(ce_loss.detach().item()) * batch_size
            total_objective_loss += float(objective_loss.detach().item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == y).sum().item())

        denom = max(total_samples, 1)
        return {
            "ce_loss": total_ce_loss / denom,
            "objective_loss": total_objective_loss / denom,
            "accuracy": total_correct / denom,
        }

    def _compute_per_example_losses(
        self,
        *,
        model: nn.Module,
        split,
        device: torch.device,
    ) -> np.ndarray:
        loader = split_to_dataloader(
            split,
            batch_size=self.per_example_loss_batch_size,
            shuffle=False,
        )
        criterion = nn.CrossEntropyLoss(reduction="none")
        model.eval()

        losses = []
        with torch.no_grad():
            for batch in loader:
                model_input, y = parse_torch_batch(batch, device)
                logits = model(model_input)
                batch_losses = criterion(logits, y)
                losses.append(batch_losses.detach().cpu().numpy())

        if not losses:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(losses, axis=0).astype(np.float32, copy=False)

    def _select_isolated_indices(self, loss_ranking: np.ndarray, *, num_train: int) -> np.ndarray:
        num_isolated = max(1, int(num_train * self.isolation_ratio))
        num_isolated = min(num_isolated, num_train)
        return np.sort(loss_ranking[:num_isolated].astype(np.int64, copy=False))


__all__ = ["ABLUnlearner"]
