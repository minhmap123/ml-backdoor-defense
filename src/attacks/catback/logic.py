from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from ...utils.logging import get_logger


LOGGER = get_logger("attacks.catback.logic")


# NOTE:
# This module follows CatBack paper/repo logic and keeps the core mechanics close to official code.
# Paper: https://arxiv.org/abs/2511.06072
# Official repo: https://github.com/catback-tabular/catback.git
# Adaptation scope is limited to DataFrame-based IO for this repository's attack framework.


@dataclass
class CategoricalEncodingArtifacts:
    categorical_columns: List[str]
    feature_names: List[str]
    primary_mappings: Dict[str, Dict[Any, float]]
    delta_r_values: Dict[str, float]
    hierarchical_mappings: Dict[str, Dict[Any, float]]
    lookup_tables: Dict[str, Dict[float, Any]]
    largest_p: int


@dataclass
class TriggerOptimizationResult:
    delta: np.ndarray
    best_loss: float


def compute_primary_frequency_mapping(df: pd.DataFrame, categorical_columns: List[str]) -> Dict[str, Dict[Any, float]]:
    primary_mappings: Dict[str, Dict[Any, float]] = {}
    for col in categorical_columns:
        freq_counts = df[col].value_counts().sort_index()
        c_max_j = int(freq_counts.max())
        if c_max_j == 1:
            primary_mappings[col] = {category: 1.0 for category in freq_counts.index}
            continue

        r_jl: Dict[Any, float] = {}
        for category, count in freq_counts.items():
            r_value = (c_max_j - int(count)) / (c_max_j - 1)
            r_jl[category] = round(float(r_value), 5)
        primary_mappings[col] = r_jl
    return primary_mappings


def compute_adaptive_delta(primary_mappings: Dict[str, Dict[Any, float]], categorical_columns: List[str]) -> Tuple[Dict[str, float], int]:
    delta_r_values: Dict[str, float] = {}
    largest_p = 0

    for col in categorical_columns:
        r_values = list(primary_mappings[col].values())
        if not r_values:
            delta_r_values[col] = 0.0
            continue

        if all(float(r) == 1.0 for r in r_values):
            delta_r_values[col] = 0.0
            continue

        unique_r = sorted(set(float(v) for v in r_values))
        if len(unique_r) < 2:
            delta_r_values[col] = 0.0
            continue

        delta_r_min = min(unique_r[i + 1] - unique_r[i] for i in range(len(unique_r) - 1))
        decimal_part = f"{delta_r_min:.10f}".split(".")[1]

        p = None
        for idx, digit in enumerate(decimal_part, start=1):
            if digit != "0":
                p = idx
                break
        if p is None:
            p = 0

        delta_r = 10 ** (-(p + 1))
        largest_p = max(largest_p, p)
        delta_r_values[col] = float(delta_r)

    return delta_r_values, largest_p


def compute_hierarchical_mapping(
    primary_mappings: Dict[str, Dict[Any, float]],
    delta_r_values: Dict[str, float],
    categorical_columns: List[str],
    largest_p: int,
) -> Tuple[Dict[str, Dict[Any, float]], Dict[str, Dict[float, Any]]]:
    hierarchical_mappings: Dict[str, Dict[Any, float]] = {}
    lookup_tables: Dict[str, Dict[float, Any]] = {col: {} for col in categorical_columns}

    for col in categorical_columns:
        primary_mapping = primary_mappings[col]
        delta_r = float(delta_r_values.get(col, 0.0))

        if delta_r == 0.0:
            mapping = {category: round(float(r), largest_p + 1) for category, r in primary_mapping.items()}
            hierarchical_mappings[col] = mapping
            for category, r_prime in mapping.items():
                lookup_tables[col][float(r_prime)] = category
            continue

        inverted: Dict[float, List[Any]] = {}
        for category, r_value in primary_mapping.items():
            inverted.setdefault(float(r_value), []).append(category)

        hierarchical: Dict[Any, float] = {}
        for r_value, categories in inverted.items():
            if len(categories) == 1:
                category = categories[0]
                hierarchical[category] = round(float(r_value), largest_p + 1)
                continue

            for idx, category in enumerate(sorted(categories), start=1):
                r_prime = round(float(r_value) + (idx - 1) * delta_r, largest_p + 1)
                hierarchical[category] = float(r_prime)

        hierarchical_mappings[col] = hierarchical
        for category, r_prime in hierarchical.items():
            lookup_tables[col][float(r_prime)] = category

    return hierarchical_mappings, lookup_tables


def encode_categorical_frame(df: pd.DataFrame, categorical_columns: List[str]) -> Tuple[pd.DataFrame, CategoricalEncodingArtifacts]:
    converted = df.copy(deep=True)
    primary_mappings = compute_primary_frequency_mapping(converted, categorical_columns)
    delta_r_values, largest_p = compute_adaptive_delta(primary_mappings, categorical_columns)
    hierarchical_mappings, lookup_tables = compute_hierarchical_mapping(
        primary_mappings,
        delta_r_values,
        categorical_columns,
        largest_p,
    )

    for col in categorical_columns:
        if col in converted.columns:
            converted[col] = converted[col].map(hierarchical_mappings[col]).astype(np.float32)

    artifacts = CategoricalEncodingArtifacts(
        categorical_columns=list(categorical_columns),
        feature_names=[str(c) for c in converted.columns],
        primary_mappings=primary_mappings,
        delta_r_values=delta_r_values,
        hierarchical_mappings=hierarchical_mappings,
        lookup_tables=lookup_tables,
        largest_p=largest_p,
    )
    return converted, artifacts


def round_encoded_categoricals(x: np.ndarray, artifacts: CategoricalEncodingArtifacts) -> np.ndarray:
    rounded = np.asarray(x, dtype=np.float32).copy()
    for col in artifacts.categorical_columns:
        col_idx = artifacts.feature_names.index(col)
        valid_values = np.array(sorted(artifacts.lookup_tables[col].keys()), dtype=np.float32)
        if valid_values.size == 0:
            continue
        feature_values = rounded[:, col_idx]
        diff = np.abs(feature_values[:, np.newaxis] - valid_values[np.newaxis, :])
        nearest_idx = np.argmin(diff, axis=1)
        rounded[:, col_idx] = valid_values[nearest_idx]
    return rounded


def revert_encoded_categoricals(x: np.ndarray, artifacts: CategoricalEncodingArtifacts) -> np.ndarray:
    reverted = np.asarray(x, dtype=np.float32).copy()
    for col in artifacts.categorical_columns:
        col_idx = artifacts.feature_names.index(col)
        for i in range(reverted.shape[0]):
            encoded_value = round(float(reverted[i, col_idx]), artifacts.largest_p + 1)
            category = artifacts.lookup_tables[col].get(encoded_value)
            if category is None:
                keys = np.array(list(artifacts.lookup_tables[col].keys()), dtype=np.float32)
                if keys.size == 0:
                    continue
                nearest = float(keys[np.argmin(np.abs(keys - encoded_value))])
                category = artifacts.lookup_tables[col][nearest]
            reverted[i, col_idx] = float(category)
    return reverted

def rank_by_target_confidence(
    model: torch.nn.Module,
    x: np.ndarray,
    candidate_indices: np.ndarray,
    target_label: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if candidate_indices.size == 0:
        return np.empty(0, dtype=np.int64)

    x_candidates = torch.tensor(x[candidate_indices], dtype=torch.float32)
    dataset = TensorDataset(x_candidates)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)

    model.eval()
    confidence_scores: List[float] = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            confidence_scores.extend(probs[:, int(target_label)].detach().cpu().numpy().tolist())

    order = np.argsort(np.asarray(confidence_scores, dtype=np.float32))[::-1]
    return candidate_indices[order].astype(np.int64)


def compute_bounds_and_mode(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_x = np.min(x, axis=0).astype(np.float32)
    max_x = np.max(x, axis=0).astype(np.float32)
    mode_vector = pd.DataFrame(x).mode().iloc[0].to_numpy(dtype=np.float32)
    return min_x, max_x, mode_vector


def optimize_trigger_delta(
    model: torch.nn.Module,
    x_ranked: np.ndarray,
    target_label: int,
    min_x: np.ndarray,
    max_x: np.ndarray,
    mode_vector: np.ndarray,
    beta: float,
    l2_lambda: float,
    lr: float,
    num_steps: int,
    device: torch.device,
    batch_size: int,
    patience: int,
) -> TriggerOptimizationResult:
    if x_ranked.shape[0] == 0:
        return TriggerOptimizationResult(delta=np.zeros(x_ranked.shape[1], dtype=np.float32), best_loss=0.0)

    x_tensor = torch.tensor(x_ranked, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tensor), batch_size=int(batch_size), shuffle=True)

    min_t = torch.tensor(min_x, dtype=torch.float32, device=device)
    max_t = torch.tensor(max_x, dtype=torch.float32, device=device)
    mode_t = torch.tensor(mode_vector, dtype=torch.float32, device=device)

    delta = torch.zeros(x_ranked.shape[1], dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=float(lr))

    model.eval()
    best_loss = float("inf")
    best_delta = delta.detach().clone()
    stale_steps = 0

    for step in range(int(num_steps)):
        epoch_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()

            x_hat = xb + delta.unsqueeze(0)
            x_hat = torch.clamp(x_hat, min=min_t, max=max_t)

            logits = model(x_hat)
            probs = torch.softmax(logits, dim=1)
            f_t = probs[:, int(target_label)]

            nll_loss = -torch.log(f_t + 1e-8).mean()
            l1_loss = torch.norm(x_hat - mode_t, p=1)
            l2_loss = torch.norm(x_hat - mode_t, p=2) ** 2
            loss = nll_loss + float(beta) * l1_loss + float(l2_lambda) * l2_loss

            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(len(loader), 1)
        LOGGER.info("CatBack optimize step %d/%d avg_loss=%.6f best_loss=%.6f", step + 1, int(num_steps), avg_loss, float(best_loss))
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_delta = delta.detach().clone()
            stale_steps = 0
        else:
            stale_steps += 1
            if stale_steps >= int(patience):
                break

    return TriggerOptimizationResult(delta=best_delta.detach().cpu().numpy().astype(np.float32), best_loss=float(best_loss))


def apply_trigger_and_clip(x: np.ndarray, delta: np.ndarray, min_x: np.ndarray, max_x: np.ndarray) -> np.ndarray:
    x_hat = np.asarray(x, dtype=np.float32) + np.asarray(delta, dtype=np.float32)[np.newaxis, :]
    return np.clip(x_hat, np.asarray(min_x, dtype=np.float32), np.asarray(max_x, dtype=np.float32)).astype(np.float32)
