from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .types import DetectorContext
from .utils import extract_model_features


@dataclass
class CSOState:
    class_masks: np.ndarray
    class_counts: np.ndarray
    class_mask_norms: np.ndarray
    selected_indices_by_class: Dict[int, np.ndarray]
    masked_reference_features: Dict[int, torch.Tensor]


class CSOHelper:
    """
    Paper-first helper for Class Subspace Orthogonalization (CSO).

    Research references:
    - Paper: https://openreview.net/forum?id=c6IRL2mdDR
    - arXiv HTML with equations:
      - Eq. (4): class-specific intrinsic mask learning
      - Eq. (5): CSO penalty
      - Eq. (6): MMBD-CSO
      - Eq. (7): MLBD-CSO

    Local research assumptions:
    - No public official code was found during implementation, so this helper is
      a direct paper-guided implementation of the published equations.
    - The active repo path is numeric-only IDS; class-specific intrinsic feature
      masks are learned from `forward_features()` on `clean_support_split`.
    - The class-specific mask optimization is performed in feature space with a
      sigmoid-parameterized soft mask and full-batch Adam by default.
    """

    def __init__(self, cfg: Any) -> None:
        _n = getattr(cfg, "num_clean_support_per_class", 10)
        self.num_clean_support_per_class = int(_n) if _n is not None else None
        self.min_clean_support_per_class = int(getattr(cfg, "min_clean_support_per_class", 10))
        self.mask_opt_steps = int(getattr(cfg, "cso_mask_opt_steps", 200))
        self.mask_opt_lr = float(getattr(cfg, "cso_mask_opt_lr", 0.1))
        self.mask_batch_size = int(getattr(cfg, "cso_mask_batch_size", 0))
        self.feature_batch_size = int(getattr(cfg, "cso_feature_batch_size", 512))
        self.epsilon = float(getattr(cfg, "cso_epsilon", 1e-8))

    def fit(
        self,
        *,
        model: torch.nn.Module,
        context: DetectorContext,
        device: torch.device,
    ) -> CSOState:
        assert context.clean_support_split is not None, "CSO requires clean_support_split."
        assert hasattr(model, "forward_features"), "CSO requires model.forward_features(...)."
        assert hasattr(model, "forward_logits"), "CSO requires model.forward_logits(...)."
        if isinstance(context.clean_support_split, dict):
            x_cat = context.clean_support_split.get("x_cat")
            assert x_cat is None, "CSO local IDS path does not accept categorical clean_support_split."

        clean_features, clean_labels = extract_model_features(
            model,
            context.clean_support_split,
            device=device,
            batch_size=self.feature_batch_size,
        )
        clean_features = np.asarray(clean_features, dtype=np.float32)
        clean_labels = np.asarray(clean_labels, dtype=np.int64)
        assert clean_features.ndim == 2 and clean_features.shape[0] > 0, (
            "CSO requires non-empty 2D clean features from forward_features()."
        )

        rng = np.random.default_rng(int(context.seed))
        num_classes = int(context.num_classes)
        feature_dim = int(clean_features.shape[1])
        class_masks = np.zeros((num_classes, feature_dim), dtype=np.float32)
        class_counts = np.zeros((num_classes,), dtype=np.int64)
        class_mask_norms = np.zeros((num_classes,), dtype=np.float32)
        selected_indices_by_class: Dict[int, np.ndarray] = {}
        masked_reference_features: Dict[int, torch.Tensor] = {}

        for class_idx in range(num_classes):
            class_indices = np.flatnonzero(clean_labels == class_idx).astype(np.int64)
            assert class_indices.size >= self.min_clean_support_per_class, (
                f"CSO requires at least {self.min_clean_support_per_class} clean support samples "
                f"for class {class_idx}, got {class_indices.size}."
            )

            if self.num_clean_support_per_class is not None:
                assert class_indices.size >= self.num_clean_support_per_class, (
                    f"CSO requested {self.num_clean_support_per_class} clean support samples for class "
                    f"{class_idx}, got {class_indices.size}."
                )
                selected = np.sort(rng.choice(class_indices, size=self.num_clean_support_per_class, replace=False))
            else:
                selected = class_indices

            selected_indices_by_class[int(class_idx)] = selected
            class_counts[int(class_idx)] = int(selected.shape[0])

            class_features = torch.as_tensor(clean_features[selected], dtype=torch.float32, device=device)
            class_mask = self._optimize_class_mask(
                model=model,
                class_features=class_features,
                class_idx=int(class_idx),
            )
            class_masks[int(class_idx)] = class_mask.detach().cpu().numpy().astype(np.float32)
            class_mask_norms[int(class_idx)] = float(torch.sum(torch.abs(class_mask)).item())

            masked_features = class_features * class_mask.unsqueeze(0)
            masked_reference_features[int(class_idx)] = F.normalize(
                masked_features,
                p=2,
                dim=1,
                eps=self.epsilon,
            ).detach()

        return CSOState(
            class_masks=class_masks,
            class_counts=class_counts,
            class_mask_norms=class_mask_norms,
            selected_indices_by_class=selected_indices_by_class,
            masked_reference_features=masked_reference_features,
        )

    def penalty(
        self,
        *,
        state: CSOState,
        candidate_features: torch.Tensor,
        target_class: int,
    ) -> torch.Tensor:
        references = state.masked_reference_features[int(target_class)]
        if references.numel() == 0:
            return torch.zeros((candidate_features.shape[0],), dtype=candidate_features.dtype, device=candidate_features.device)

        candidate_norm = F.normalize(candidate_features, p=2, dim=1, eps=self.epsilon)
        reference_norm = references.to(candidate_features.device)
        cosine = candidate_norm @ reference_norm.T
        return F.relu(cosine).mean(dim=1)

    def state_to_trace(self, state: CSOState) -> Dict[str, Any]:
        return {
            "num_clean_support_per_class": self.num_clean_support_per_class,
            "min_clean_support_per_class": self.min_clean_support_per_class,
            "mask_opt_steps": self.mask_opt_steps,
            "mask_opt_lr": self.mask_opt_lr,
            "mask_batch_size": self.mask_batch_size,
            "feature_batch_size": self.feature_batch_size,
            "class_counts": state.class_counts.astype(np.int64).tolist(),
            "class_mask_norms": state.class_mask_norms.astype(np.float32).tolist(),
            "class_masks": state.class_masks.astype(np.float32).tolist(),
            "selected_indices_by_class": {
                str(class_idx): indices.astype(np.int64).tolist()
                for class_idx, indices in state.selected_indices_by_class.items()
            },
        }

    def _optimize_class_mask(
        self,
        *,
        model: torch.nn.Module,
        class_features: torch.Tensor,
        class_idx: int,
    ) -> torch.Tensor:
        feature_dim = int(class_features.shape[1])
        labels = torch.full((class_features.shape[0],), int(class_idx), dtype=torch.long, device=class_features.device)
        mask_logits = torch.nn.Parameter(torch.zeros(feature_dim, dtype=torch.float32, device=class_features.device))
        optimizer = torch.optim.Adam([mask_logits], lr=self.mask_opt_lr)

        batch_size = int(self.mask_batch_size)
        if batch_size <= 0:
            batch_size = int(class_features.shape[0])

        for _ in range(self.mask_opt_steps):
            perm = torch.randperm(class_features.shape[0], device=class_features.device)
            for start in range(0, int(class_features.shape[0]), batch_size):
                batch_indices = perm[start : start + batch_size]
                features_batch = class_features[batch_indices]
                labels_batch = labels[batch_indices]

                optimizer.zero_grad()
                mask = torch.sigmoid(mask_logits)
                intrinsic_logits = model.forward_logits(features_batch * mask.unsqueeze(0))
                complement_logits = model.forward_logits(features_batch * (1.0 - mask).unsqueeze(0))
                loss = F.cross_entropy(intrinsic_logits, labels_batch) - F.cross_entropy(complement_logits, labels_batch)
                loss.backward()
                optimizer.step()

        return torch.sigmoid(mask_logits).detach()
