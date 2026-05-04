from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..models.utils import split_to_numpy
from .base import BaseDetector
from .types import DetectorContext, DetectorResult
from .utils import measure_runtime, resolve_device


class NeuralCleanseDetector(BaseDetector):
    """
    Paper- and official-code-guided Neural Cleanse detector adapted to numeric IDS inputs.

    Research references:
    - Paper: https://doi.org/10.1109/SP.2019.00031
    - Official repo: https://github.com/bolunwang/backdoor

    Local research assumptions:
    - The official method reverse-engineers a spatial mask and pattern in image
      space. The active repo path adapts this to numeric IDS by using feature-wise
      mask and pattern vectors on bounded continuous inputs.
    - The official optimization schedule is preserved as closely as practical:
      optimize one target class at a time, use a dynamic cost balancing CE loss
      and mask regularization, and detect anomalous target classes using MAD on
      mask norms.
    - The clean support split is used as the reverse-engineering data pool,
      matching the official sample script behavior of optimizing over a pooled
      clean dataset rather than a target-filtered subset.

    Suggested deviation note for reporting:
    - We adapt Neural Cleanse from image-domain spatial triggers to numeric IDS
      inputs by replacing the 2D mask and pattern with feature-wise mask and
      pattern vectors defined over bounded continuous features. The target-wise
      optimization schedule, dynamic cost adjustment, and MAD-based anomaly
      detection follow the official method, while the trigger parameterization
      is modified to fit all-numeric tabular data.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.requires_detection_split = False
        self.requires_clean_support_split = True

        self.regularization = str(getattr(cfg, "regularization", "l1")).lower()
        self.init_cost = float(getattr(cfg, "init_cost", 1e-3))
        self.num_steps = int(getattr(cfg, "num_steps", 1000))
        self.num_samples_per_step = int(getattr(cfg, "num_samples_per_step", 1000))
        self.batch_size = int(getattr(cfg, "batch_size", 32))
        self.lr = float(getattr(cfg, "lr", 0.1))
        self.attack_succ_threshold = float(getattr(cfg, "attack_succ_threshold", 0.99))
        self.patience = int(getattr(cfg, "patience", 5))
        self.cost_multiplier = float(getattr(cfg, "cost_multiplier", 2.0))
        self.reset_cost_to_zero = bool(getattr(cfg, "reset_cost_to_zero", True))
        self.early_stop = bool(getattr(cfg, "early_stop", True))
        self.early_stop_threshold = float(getattr(cfg, "early_stop_threshold", 1.0))
        self.early_stop_patience = int(getattr(cfg, "early_stop_patience", 25))
        self.allow_clean_support_bounds_fallback = bool(
            getattr(cfg, "allow_clean_support_bounds_fallback", False)
        )
        self.epsilon = float(getattr(cfg, "epsilon", 1e-7))
        self.mad_threshold = float(getattr(cfg, "mad_threshold", 2.0))
        self.scan_priority_label = getattr(cfg, "scan_priority_label", None)

    def _validate_context(self, context: DetectorContext) -> None:
        super()._validate_context(context)
        self._assert_numeric_only_context(context)
        assert self.regularization in {"l1", "l2"}, f"Unsupported regularization: {self.regularization}"

    def _run_impl(self, context: DetectorContext) -> DetectorResult:
        result, runtime_sec = measure_runtime(self._scan_all_targets, context)
        result.runtime_sec = float(runtime_sec)
        return result

    def _scan_all_targets(self, context: DetectorContext) -> DetectorResult:
        device = resolve_device(context.device)
        model = context.model.to(device)
        model.eval()

        x_clean, _ = self._extract_clean_support(context)
        assert x_clean.shape[0] > 0, "Neural Cleanse requires a non-empty clean_support_split."

        lower, upper = self._resolve_bounds(context, x_clean)
        lower_t = torch.as_tensor(lower, dtype=torch.float32, device=device)
        upper_t = torch.as_tensor(upper, dtype=torch.float32, device=device)

        target_order = self._resolve_target_order(context)
        anomaly_scores = np.zeros(int(context.num_classes), dtype=np.float32)
        mask_norms = np.full(int(context.num_classes), np.nan, dtype=np.float32)
        per_target_stats: List[Dict[str, Any]] = []
        best_masks: Dict[int, np.ndarray] = {}
        best_patterns: Dict[int, np.ndarray] = {}

        for target_class in target_order:
            pattern_best, mask_best, target_stats = self._optimize_one_target(
                x_clean=x_clean,
                target_class=int(target_class),
                model=model,
                lower_t=lower_t,
                upper_t=upper_t,
                device=device,
            )
            mask_norm = float(np.sum(np.abs(mask_best)))
            mask_norms[int(target_class)] = mask_norm
            best_masks[int(target_class)] = mask_best.astype(np.float32, copy=False)
            best_patterns[int(target_class)] = pattern_best.astype(np.float32, copy=False)
            per_target_stats.append(target_stats)

        anomaly_index, flagged_labels, anomaly_scores = self._mad_outlier_detection(mask_norms)
        smallest_mask_target_class = int(np.nanargmin(mask_norms))
        predicted_is_infected = bool(len(flagged_labels) > 0)
        predicted_target_class = int(flagged_labels[0]) if predicted_is_infected else None
        artifact_mask = best_masks[smallest_mask_target_class]
        artifact_pattern = best_patterns[smallest_mask_target_class]

        class_details = self._build_class_details(
            mask_norms=mask_norms,
            anomaly_scores=anomaly_scores,
            flagged_labels=flagged_labels,
            smallest_mask_target_class=smallest_mask_target_class,
            predicted_target_class=predicted_target_class,
        )

        return DetectorResult(
            detector_name=self.name,
            track_type="class",
            status="ok",
            seed=int(context.seed),
            runtime_sec=0.0,
            summary_metrics={
                "detection/anomaly_index": float(anomaly_index),
                "detection/num_flagged_labels": float(len(flagged_labels)),
                "detection/min_mask_norm": float(np.nanmin(mask_norms)),
                "detection/median_mask_norm": float(np.nanmedian(mask_norms)),
                "detection/smallest_mask_target_class": float(smallest_mask_target_class),
            },
            class_scores=anomaly_scores,
            class_details=class_details,
            predicted_is_infected=predicted_is_infected,
            predicted_target_class=predicted_target_class,
            thresholds={
                "threshold_source": "mad_lower_tail",
                "mad_threshold": self.mad_threshold,
            },
            optimization_trace={
                "regularization": self.regularization,
                "init_cost": self.init_cost,
                "num_steps": self.num_steps,
                "num_samples_per_step": self.num_samples_per_step,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "attack_succ_threshold": self.attack_succ_threshold,
                "patience": self.patience,
                "cost_multiplier": self.cost_multiplier,
                "reset_cost_to_zero": self.reset_cost_to_zero,
                "early_stop": self.early_stop,
                "early_stop_threshold": self.early_stop_threshold,
                "early_stop_patience": self.early_stop_patience,
                "lower_bounds_source": self._resolve_bounds_source(context),
                "allow_clean_support_bounds_fallback": self.allow_clean_support_bounds_fallback,
                "scan_priority_label": self.scan_priority_label,
                "target_order": target_order,
                "mask_norms": mask_norms.astype(np.float32).tolist(),
                "anomaly_scores": anomaly_scores.astype(np.float32).tolist(),
                "flagged_labels": [int(x) for x in flagged_labels],
                "anomaly_index": float(anomaly_index),
                "smallest_mask_target_class": int(smallest_mask_target_class),
                "predicted_target_class_if_flagged": predicted_target_class,
                "per_target_stats": per_target_stats,
            },
            estimated_trigger=artifact_pattern,
            estimated_mask=artifact_mask,
        )

    def _extract_clean_support(self, context: DetectorContext) -> Tuple[np.ndarray, np.ndarray]:
        self._assert_numeric_only_context(context)
        x, y = split_to_numpy(context.clean_support_split)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if x.ndim == 1:
            x = x[:, None]
        return x, y

    def _resolve_target_order(self, context: DetectorContext) -> List[int]:
        labels = list(range(int(context.num_classes)))
        priority = self.scan_priority_label
        if priority is not None and int(priority) in labels:
            labels.remove(int(priority))
            labels = [int(priority)] + labels
        return labels

    def _resolve_bounds(self, context: DetectorContext, reference_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d_in = int(reference_x.shape[1])
        if context.feature_metadata is not None:
            lower = context.feature_metadata.feature_bounds_min
            upper = context.feature_metadata.feature_bounds_max
            if lower is not None and upper is not None:
                return np.asarray(lower, dtype=np.float32)[:d_in], np.asarray(upper, dtype=np.float32)[:d_in]

        assert self.allow_clean_support_bounds_fallback, (
            "Neural Cleanse requires explicit feature bounds in feature_metadata unless "
            "`allow_clean_support_bounds_fallback=true` is set."
        )
        return (
            np.min(reference_x, axis=0).astype(np.float32),
            np.max(reference_x, axis=0).astype(np.float32),
        )

    def _resolve_bounds_source(self, context: DetectorContext) -> str:
        if context.feature_metadata is not None:
            if context.feature_metadata.feature_bounds_min is not None and context.feature_metadata.feature_bounds_max is not None:
                return "feature_metadata"
        return "clean_support_minmax_fallback"

    def _assert_numeric_only_context(self, context: DetectorContext) -> None:
        if context.feature_metadata is not None:
            assert int(context.feature_metadata.num_categorical_features) == 0, (
                "Neural Cleanse local IDS path expects numeric-only features."
            )
        if context.model_metadata is not None and "num_categorical_features" in context.model_metadata:
            assert int(context.model_metadata["num_categorical_features"]) == 0, (
                "Neural Cleanse local IDS path expects numeric-only models."
            )
        if hasattr(context.model, "num_categorical_features"):
            assert int(getattr(context.model, "num_categorical_features")) == 0, (
                "Neural Cleanse local IDS path expects numeric-only model instances."
            )
        for split_name in ("clean_support_split", "detection_split"):
            split = getattr(context, split_name)
            if isinstance(split, dict):
                x_cat = split.get("x_cat")
                assert x_cat is None, f"Neural Cleanse local IDS path does not accept categorical {split_name}."

    def _optimize_one_target(
        self,
        *,
        x_clean: np.ndarray,
        target_class: int,
        model: torch.nn.Module,
        lower_t: torch.Tensor,
        upper_t: torch.Tensor,
        device: torch.device,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        d_in = int(x_clean.shape[1])
        # The official sample script uses `MINI_BATCH = NB_SAMPLE // BATCH_SIZE`.
        # Keep the same floor-division semantics here for fidelity.
        mini_batch = max(int(self.num_samples_per_step // self.batch_size), 1)

        mask_init = np.random.random(d_in).astype(np.float32)
        pattern_init = np.random.uniform(low=lower_t.cpu().numpy(), high=upper_t.cpu().numpy()).astype(np.float32)
        mask_tanh = torch.nn.Parameter(
            torch.as_tensor(self._to_tanh_space(mask_init, min_v=0.0, max_v=1.0), dtype=torch.float32, device=device)
        )
        pattern_tanh = torch.nn.Parameter(
            torch.as_tensor(
                self._to_tanh_space(pattern_init, min_v=lower_t.cpu().numpy(), max_v=upper_t.cpu().numpy()),
                dtype=torch.float32,
                device=device,
            )
        )
        optimizer = torch.optim.Adam([pattern_tanh, mask_tanh], lr=self.lr, betas=(0.5, 0.9))

        cost = 0.0 if self.reset_cost_to_zero else float(self.init_cost)
        mask_best = None
        pattern_best = None
        reg_best = float("inf")
        logs: List[Dict[str, Any]] = []
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False
        early_stop_counter = 0
        early_stop_reg_best = reg_best

        target_tensor_full = torch.full((self.batch_size,), int(target_class), dtype=torch.long, device=device)

        for step in range(self.num_steps):
            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            loss_acc_list = []

            for _ in range(mini_batch):
                batch_idx = np.random.choice(x_clean.shape[0], size=min(self.batch_size, x_clean.shape[0]), replace=False)
                x_batch = torch.as_tensor(x_clean[batch_idx], dtype=torch.float32, device=device)
                y_target = target_tensor_full[: x_batch.shape[0]]

                optimizer.zero_grad()
                mask, pattern = self._decode_parameters(mask_tanh=mask_tanh, pattern_tanh=pattern_tanh, lower_t=lower_t, upper_t=upper_t)
                x_adv = self._apply_trigger(x_batch, mask, pattern)
                logits = model(x_adv)
                loss_ce = F.cross_entropy(logits, y_target)
                loss_reg = self._mask_regularization(mask)
                loss = loss_ce + loss_reg * cost
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)
                    loss_acc = float(pred.eq(y_target).float().mean().item())

                loss_ce_list.append(float(loss_ce.item()))
                loss_reg_list.append(float(loss_reg.item()))
                loss_list.append(float(loss.item()))
                loss_acc_list.append(float(loss_acc))

            avg_loss_ce = float(np.mean(loss_ce_list))
            avg_loss_reg = float(np.mean(loss_reg_list))
            avg_loss = float(np.mean(loss_list))
            avg_loss_acc = float(np.mean(loss_acc_list))

            if avg_loss_acc >= self.attack_succ_threshold and avg_loss_reg < reg_best:
                with torch.no_grad():
                    mask_best, pattern_best = self._decode_numpy(
                        mask_tanh=mask_tanh,
                        pattern_tanh=pattern_tanh,
                        lower_t=lower_t,
                        upper_t=upper_t,
                    )
                reg_best = avg_loss_reg

            logs.append(
                {
                    "step": int(step),
                    "avg_loss_ce": avg_loss_ce,
                    "avg_loss_reg": avg_loss_reg,
                    "avg_loss": avg_loss,
                    "avg_loss_acc": avg_loss_acc,
                    "reg_best": float(reg_best),
                    "cost": float(cost),
                }
            )

            if self.early_stop:
                if reg_best < float("inf"):
                    if reg_best >= self.early_stop_threshold * early_stop_reg_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_reg_best = min(reg_best, early_stop_reg_best)
                if cost_down_flag and cost_up_flag and early_stop_counter >= self.early_stop_patience:
                    break

            if cost == 0 and avg_loss_acc >= self.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.patience:
                    cost = float(self.init_cost)
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
            else:
                cost_set_counter = 0

            if avg_loss_acc >= self.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                cost *= self.cost_multiplier
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                cost /= self.cost_multiplier ** 1.5
                cost_down_flag = True

        if mask_best is None or pattern_best is None:
            with torch.no_grad():
                mask_best, pattern_best = self._decode_numpy(
                    mask_tanh=mask_tanh,
                    pattern_tanh=pattern_tanh,
                    lower_t=lower_t,
                    upper_t=upper_t,
                )

        target_stats = {
            "target_class": int(target_class),
            "best_mask_norm": float(np.sum(np.abs(mask_best))),
            "best_attack_success": float(max((entry["avg_loss_acc"] for entry in logs), default=0.0)),
            "steps_run": int(len(logs)),
            "final_cost": float(cost),
            "reg_best": float(reg_best),
            "logs": logs,
        }
        return pattern_best, mask_best, target_stats

    def _decode_parameters(
        self,
        *,
        mask_tanh: torch.Tensor,
        pattern_tanh: torch.Tensor,
        lower_t: torch.Tensor,
        upper_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.tanh(mask_tanh) / (2.0 - self.epsilon) + 0.5
        mask = torch.clamp(mask, min=0.0, max=1.0)
        pattern_unit = torch.tanh(pattern_tanh) / (2.0 - self.epsilon) + 0.5
        pattern = lower_t + pattern_unit * (upper_t - lower_t)
        return mask, pattern

    def _decode_numpy(
        self,
        *,
        mask_tanh: torch.Tensor,
        pattern_tanh: torch.Tensor,
        lower_t: torch.Tensor,
        upper_t: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mask, pattern = self._decode_parameters(
            mask_tanh=mask_tanh,
            pattern_tanh=pattern_tanh,
            lower_t=lower_t,
            upper_t=upper_t,
        )
        return (
            mask.detach().cpu().numpy().astype(np.float32),
            pattern.detach().cpu().numpy().astype(np.float32),
        )

    def _apply_trigger(self, x: torch.Tensor, mask: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        return x * (1.0 - mask.unsqueeze(0)) + pattern.unsqueeze(0) * mask.unsqueeze(0)

    def _mask_regularization(self, mask: torch.Tensor) -> torch.Tensor:
        if self.regularization == "l1":
            return torch.sum(torch.abs(mask))
        return torch.sqrt(torch.sum(mask.square()) + self.epsilon)

    def _to_tanh_space(self, value: np.ndarray, *, min_v: np.ndarray | float, max_v: np.ndarray | float) -> np.ndarray:
        min_v_arr = np.asarray(min_v, dtype=np.float32)
        max_v_arr = np.asarray(max_v, dtype=np.float32)
        span = np.maximum(max_v_arr - min_v_arr, self.epsilon)
        scaled = (np.asarray(value, dtype=np.float32) - min_v_arr) / span
        scaled = np.clip(scaled, self.epsilon, 1.0 - self.epsilon)
        return np.arctanh((scaled - 0.5) * (2.0 - self.epsilon))

    def _mad_outlier_detection(self, mask_norms: np.ndarray) -> Tuple[float, List[int], np.ndarray]:
        norms = np.asarray(mask_norms, dtype=np.float64)
        if np.allclose(norms, norms[0]):
            return 0.0, [], np.zeros_like(norms, dtype=np.float32)

        consistency_constant = 1.4826
        median = float(np.median(norms))
        mad = float(consistency_constant * np.median(np.abs(norms - median)))
        if mad < self.epsilon:
            return 0.0, [], np.zeros_like(norms, dtype=np.float32)

        anomaly_scores = np.zeros_like(norms, dtype=np.float32)
        flagged = []
        for target_class in range(len(norms)):
            if norms[target_class] > median:
                continue
            score = float(abs(norms[target_class] - median) / mad)
            anomaly_scores[target_class] = score
            if score > self.mad_threshold:
                flagged.append(int(target_class))

        flagged = sorted(flagged, key=lambda label: norms[label])
        anomaly_index = float(abs(np.min(norms) - median) / mad)
        return anomaly_index, flagged, anomaly_scores.astype(np.float32)

    def _build_class_details(
        self,
        mask_norms: np.ndarray,
        anomaly_scores: np.ndarray,
        flagged_labels: List[int],
        smallest_mask_target_class: int,
        predicted_target_class: Optional[int],
    ) -> pd.DataFrame:
        flagged_set = set(int(x) for x in flagged_labels)
        predicted_target = None if predicted_target_class is None else int(predicted_target_class)
        return pd.DataFrame(
            {
                "class_index": np.arange(mask_norms.shape[0], dtype=np.int64),
                "mask_norm": np.asarray(mask_norms, dtype=np.float32),
                "anomaly_score": np.asarray(anomaly_scores, dtype=np.float32),
                "flagged": np.asarray(
                    [1 if idx in flagged_set else 0 for idx in range(mask_norms.shape[0])],
                    dtype=np.int64,
                ),
                "is_smallest_mask": np.asarray(
                    [1 if idx == int(smallest_mask_target_class) else 0 for idx in range(mask_norms.shape[0])],
                    dtype=np.int64,
                ),
                "is_predicted_target": np.asarray(
                    [1 if predicted_target is not None and idx == predicted_target else 0 for idx in range(mask_norms.shape[0])],
                    dtype=np.int64,
                ),
            }
        )
