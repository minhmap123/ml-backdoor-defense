from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .cso import CSOHelper
from .neural_cleanse import NeuralCleanseDetector
from .types import DetectorContext, DetectorResult
from .utils import measure_runtime, resolve_device


class NCCSODetector(NeuralCleanseDetector):
    """
    Paper-first NC-CSO detector for numeric IDS inputs.

    Research references:
    - Neural Cleanse paper: https://doi.org/10.1109/SP.2019.00031
    - Neural Cleanse official repo: https://github.com/bolunwang/backdoor
    - CSO paper: https://openreview.net/forum?id=c6IRL2mdDR

    Local research assumptions:
    - No public official CSO code was found during implementation, so this
      detector follows the published NC-CSO objective in Eq. (8) directly.
    - The locked local Neural Cleanse baseline keeps the paper/official
      target-wise reverse-engineering schedule and adapts only the trigger
      parameterization to numeric IDS inputs.
    - NC-CSO reuses that same input-space mask/pattern optimization and adds
      the CSO penalty in feature space using `forward_features()`.

    Suggested deviation note for reporting:
    - We implement NC-CSO directly from the published CSO objective because no
      public official code was available. The detector inherits the local
      numeric IDS adaptation of Neural Cleanse by optimizing feature-wise mask
      and pattern vectors over bounded continuous inputs, while the additional
      CSO term penalizes positive cosine similarity between reverse-engineered
      candidate features and the target class intrinsic feature subspace
      estimated from clean support data.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.lambda_cso = float(getattr(cfg, "lambda_cso", 0.01))
        self.cso_helper = CSOHelper(cfg)

    def _validate_context(self, context: DetectorContext) -> None:
        super()._validate_context(context)
        assert hasattr(context.model, "forward_features"), "NC-CSO requires model.forward_features(...)."
        assert hasattr(context.model, "forward_logits"), "NC-CSO requires model.forward_logits(...)."

    def _run_impl(self, context: DetectorContext) -> DetectorResult:
        result, runtime_sec = measure_runtime(self._scan_all_targets_with_cso, context)
        result.runtime_sec = float(runtime_sec)
        return result

    def _scan_all_targets_with_cso(self, context: DetectorContext) -> DetectorResult:
        device = resolve_device(context.device)
        model = context.model.to(device)
        model.eval()

        x_clean, _ = self._extract_clean_support(context)
        assert x_clean.shape[0] > 0, "NC-CSO requires a non-empty clean_support_split."

        lower, upper = self._resolve_bounds(context, x_clean)
        lower_t = torch.as_tensor(lower, dtype=torch.float32, device=device)
        upper_t = torch.as_tensor(upper, dtype=torch.float32, device=device)

        target_order = self._resolve_target_order(context)
        anomaly_scores = np.zeros(int(context.num_classes), dtype=np.float32)
        mask_norms = np.full(int(context.num_classes), np.nan, dtype=np.float32)
        per_target_stats: List[Dict[str, Any]] = []
        best_masks: Dict[int, np.ndarray] = {}
        best_patterns: Dict[int, np.ndarray] = {}
        cso_state = self.cso_helper.fit(model=model, context=context, device=device)

        for target_class in target_order:
            pattern_best, mask_best, target_stats = self._optimize_one_target_with_cso(
                x_clean=x_clean,
                target_class=int(target_class),
                model=model,
                lower_t=lower_t,
                upper_t=upper_t,
                device=device,
                cso_state=cso_state,
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
                "lambda_cso": self.lambda_cso,
                "target_order": target_order,
                "mask_norms": mask_norms.astype(np.float32).tolist(),
                "anomaly_scores": anomaly_scores.astype(np.float32).tolist(),
                "flagged_labels": [int(x) for x in flagged_labels],
                "anomaly_index": float(anomaly_index),
                "smallest_mask_target_class": int(smallest_mask_target_class),
                "predicted_target_class_if_flagged": predicted_target_class,
                "cso": self.cso_helper.state_to_trace(cso_state),
                "per_target_stats": per_target_stats,
            },
            estimated_trigger=artifact_pattern,
            estimated_mask=artifact_mask,
            feature_layer_name="forward_features",
        )

    def _optimize_one_target_with_cso(
        self,
        *,
        x_clean: np.ndarray,
        target_class: int,
        model: torch.nn.Module,
        lower_t: torch.Tensor,
        upper_t: torch.Tensor,
        device: torch.device,
        cso_state,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        d_in = int(x_clean.shape[1])
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
        best_cso_penalty = float("nan")
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
            loss_cso_list = []
            loss_list = []
            loss_acc_list = []

            for _ in range(mini_batch):
                batch_idx = np.random.choice(x_clean.shape[0], size=min(self.batch_size, x_clean.shape[0]), replace=False)
                x_batch = torch.as_tensor(x_clean[batch_idx], dtype=torch.float32, device=device)
                y_target = target_tensor_full[: x_batch.shape[0]]

                optimizer.zero_grad()
                mask, pattern = self._decode_parameters(
                    mask_tanh=mask_tanh,
                    pattern_tanh=pattern_tanh,
                    lower_t=lower_t,
                    upper_t=upper_t,
                )
                x_adv = self._apply_trigger(x_batch, mask, pattern)
                logits = model(x_adv)
                loss_ce = F.cross_entropy(logits, y_target)
                loss_reg = self._mask_regularization(mask)
                candidate_features = model.forward_features(x_adv)
                loss_cso = self.cso_helper.penalty(
                    state=cso_state,
                    candidate_features=candidate_features,
                    target_class=int(target_class),
                ).mean()
                loss = loss_ce + loss_reg * cost + self.lambda_cso * loss_cso
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)
                    loss_acc = float(pred.eq(y_target).float().mean().item())

                loss_ce_list.append(float(loss_ce.item()))
                loss_reg_list.append(float(loss_reg.item()))
                loss_cso_list.append(float(loss_cso.item()))
                loss_list.append(float(loss.item()))
                loss_acc_list.append(float(loss_acc))

            avg_loss_ce = float(np.mean(loss_ce_list))
            avg_loss_reg = float(np.mean(loss_reg_list))
            avg_loss_cso = float(np.mean(loss_cso_list))
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
                best_cso_penalty = avg_loss_cso

            logs.append(
                {
                    "step": int(step),
                    "avg_loss_ce": avg_loss_ce,
                    "avg_loss_reg": avg_loss_reg,
                    "avg_loss_cso": avg_loss_cso,
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

        if not np.isfinite(best_cso_penalty):
            best_cso_penalty = float(logs[-1]["avg_loss_cso"]) if logs else 0.0

        target_stats = {
            "target_class": int(target_class),
            "best_mask_norm": float(np.sum(np.abs(mask_best))),
            "best_attack_success": float(max((entry["avg_loss_acc"] for entry in logs), default=0.0)),
            "steps_run": int(len(logs)),
            "final_cost": float(cost),
            "reg_best": float(reg_best),
            "lambda_cso": float(self.lambda_cso),
            "best_candidate_cso_penalty": float(best_cso_penalty),
            "logs": logs,
        }
        return pattern_best, mask_best, target_stats
