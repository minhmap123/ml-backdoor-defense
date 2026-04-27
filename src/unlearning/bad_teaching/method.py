from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...models.utils import resolve_device, save_model
from ...utils.logging import get_logger
from ..base import BaseUnlearner
from ..types import ForgetSet, UnlearningArtifacts, UnlearningContext, UnlearningResult
from ..utils import split_train_forget_retain, subset_split
from .utils import (
    build_bad_teaching_dataset,
    clone_model,
    freeze_model,
    parse_membership_batch,
)


LOGGER = get_logger("unlearning.bad_teaching")


def UnlearnerLoss(
    output: torch.Tensor,
    labels: torch.Tensor,
    full_teacher_logits: torch.Tensor,
    unlearn_teacher_logits: torch.Tensor,
    KL_temperature: float,
) -> torch.Tensor:
    labels = torch.unsqueeze(labels, dim=1).float()

    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    # Upstream semantics: label 1 means forget sample, label 0 means retain sample.
    overall_teacher_out = labels * u_teacher_out + (1.0 - labels) * f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="reduction: 'mean' divides the total loss.*")
        return F.kl_div(student_out, overall_teacher_out, reduction="mean")


def unlearning_step(
    model: torch.nn.Module,
    unlearning_teacher: torch.nn.Module,
    full_trained_teacher: torch.nn.Module,
    unlearn_data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    KL_temperature: float,
) -> float:
    losses = []
    model.train()

    for batch in unlearn_data_loader:
        x, y = parse_membership_batch(batch, device)
        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)

        output = model(x)
        optimizer.zero_grad(set_to_none=True)
        loss = UnlearnerLoss(
            output=output,
            labels=y,
            full_teacher_logits=full_teacher_logits,
            unlearn_teacher_logits=unlearn_teacher_logits,
            KL_temperature=KL_temperature,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return float(sum(losses) / max(len(losses), 1))


class BadTeachingUnlearner(BaseUnlearner):
    """
    Bad Teaching unlearner using the official blindspot distillation loop.

    Upstream provenance for this wrapper:
    - repo: https://github.com/vikram2000b/bad-teaching-unlearning
    - commit: f1aa988f71cccf1be6d50e0c6f7b2b905e4c9126
    - source file mirrored locally: third_party/bad-teaching-unlearning/unlearn.py

    Local research assumptions:
    - This repo uses sample-wise forgetting, not class-wise forgetting.
    - The student starts from the attacked model weights already present in the
      benchmark pipeline.
    - The competent teacher is a frozen copy of the attacked model.
    - The incompetent teacher is the same local architecture with fresh random
      initialization, matching the author notebook.
    - The author notebook samples 30% of the retain pool before building the
      mixed unlearning dataset.
    - `retain_fraction`, `max_retain_samples`, and `retain_to_forget_ratio`
      remain local controls for IDS/backdoor ablations.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.epochs = int(self.resolved_cfg.get("epochs", 1))
        self.batch_size = int(self.resolved_cfg.get("batch_size", 256))
        self.learning_rate = float(self.resolved_cfg.get("learning_rate", 1e-4))
        self.weight_decay = float(self.resolved_cfg.get("weight_decay", 0.0))
        self.optimizer_name = str(self.resolved_cfg.get("optimizer", "adam")).lower()
        self.num_workers = int(self.resolved_cfg.get("num_workers", 0))
        self.pin_memory = bool(self.resolved_cfg.get("pin_memory", False))
        self.kl_temperature = float(self.resolved_cfg.get("kl_temperature", 1.0))
        self.save_checkpoint = bool(self.resolved_cfg.get("save_checkpoint", True))
        self.retain_fraction = float(self.resolved_cfg.get("retain_fraction", 0.3))
        self.max_retain_samples = int(self.resolved_cfg.get("max_retain_samples", 0) or 0)
        self.retain_to_forget_ratio = float(self.resolved_cfg.get("retain_to_forget_ratio", 0.0) or 0.0)

    def _run_impl(self, context: UnlearningContext, forget_set: ForgetSet) -> UnlearningResult:
        assert context.model_cfg is not None, "BadTeachingUnlearner requires context.model_cfg."
        assert self.epochs > 0, "bad_teaching epochs must be > 0."
        assert self.batch_size > 0, "bad_teaching batch_size must be > 0."
        assert self.kl_temperature > 0.0, "bad_teaching kl_temperature must be > 0."

        device = resolve_device(None if context.device is None else str(context.device))
        model_cfg = dict(context.model_cfg)

        forget_split, retain_split, forget_indices, retain_indices = split_train_forget_retain(
            context.datasets["train"],
            forget_set.indices,
        )
        retain_used_indices = retain_indices
        if len(retain_indices) > 0:
            retain_cap = len(retain_indices)
            if 0.0 < self.retain_fraction < 1.0:
                retain_cap = min(retain_cap, int(max(1, round(self.retain_fraction * len(retain_indices)))))
            if self.retain_to_forget_ratio > 0.0 and len(forget_indices) > 0:
                retain_cap = min(retain_cap, int(max(1, round(self.retain_to_forget_ratio * len(forget_indices)))))
            if self.max_retain_samples > 0:
                retain_cap = min(retain_cap, int(self.max_retain_samples))
            if retain_cap < len(retain_indices):
                rng = np.random.default_rng(int(context.seed))
                selected_positions = np.sort(rng.choice(len(retain_indices), size=retain_cap, replace=False))
                retain_split = subset_split(retain_split, selected_positions)
                retain_used_indices = retain_indices[selected_positions]

        student_model = clone_model(model_cfg, source_model=context.model).to(device)
        full_trained_teacher = freeze_model(clone_model(model_cfg, source_model=context.model)).to(device)
        unlearning_teacher = freeze_model(clone_model(model_cfg)).to(device)

        unlearning_data = build_bad_teaching_dataset(
            forget_split=forget_split,
            retain_split=retain_split,
        )
        unlearning_loader = DataLoader(
            unlearning_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        optimizer = self._build_optimizer(student_model)

        history = []
        best_loss = float("inf")
        last_loss = float("inf")
        for epoch in range(self.epochs):
            last_loss = unlearning_step(
                model=student_model,
                unlearning_teacher=unlearning_teacher,
                full_trained_teacher=full_trained_teacher,
                unlearn_data_loader=unlearning_loader,
                optimizer=optimizer,
                device=device,
                KL_temperature=self.kl_temperature,
            )
            best_loss = min(best_loss, last_loss)
            history.append({"epoch": epoch + 1, "bad_teaching/unlearning_loss": float(last_loss)})
            LOGGER.info("BadTeaching epoch=%d/%d loss=%.6f", epoch + 1, self.epochs, last_loss)

        artifacts = UnlearningArtifacts()
        trace_path = Path(context.run_dir) / "optimization_trace.json"
        with trace_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        artifacts.extra_files["optimization_trace_json"] = str(trace_path)

        checkpoint_dir = None
        if self.save_checkpoint:
            checkpoint_dir = save_model(
                student_model,
                str(Path(context.run_dir) / "checkpoint"),
                config={"model_kwargs": model_cfg},
                metadata={
                    "method": "bad_teaching",
                    "student_init": "attacked_model_weights",
                    "competent_teacher_init": "attacked_model_weights",
                    "incompetent_teacher_init": "fresh_random_model",
                },
                optimizer=optimizer,
                metrics={
                    "bad_teaching/unlearning_loss_last": float(last_loss),
                    "bad_teaching/unlearning_loss_best": float(best_loss),
                    "bad_teaching/epochs_completed": float(self.epochs),
                },
            )

        summary_metrics: Dict[str, Any] = {
            "bad_teaching/unlearning_loss_last": float(last_loss),
            "bad_teaching/unlearning_loss_best": float(best_loss),
            "bad_teaching/epochs_completed": float(self.epochs),
            "bad_teaching/forget_train_size": float(len(forget_indices)),
            "bad_teaching/retain_train_size": float(len(retain_indices)),
            "bad_teaching/retain_used_size": float(len(retain_used_indices)),
            "bad_teaching/retain_fraction": float(self.retain_fraction),
            "bad_teaching/max_retain_samples": float(self.max_retain_samples),
            "bad_teaching/retain_to_forget_ratio": float(self.retain_to_forget_ratio),
            "bad_teaching/kl_temperature": float(self.kl_temperature),
        }

        return UnlearningResult(
            method_name=self.name,
            track_type=self.track_type,
            status="ok",
            seed=int(context.seed),
            runtime_sec=0.0,
            forget_set_source=forget_set.source,
            removed_indices=forget_indices,
            summary_metrics=summary_metrics,
            checkpoint_dir=checkpoint_dir,
            model_after=student_model,
            artifacts=artifacts,
        )

    def _build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        if self.optimizer_name == "adam":
            return torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        raise AssertionError(f"Unsupported bad_teaching optimizer: {self.optimizer_name}")
