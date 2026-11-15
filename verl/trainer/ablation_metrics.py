"""Diagnostics for the matched scalar-GRPO versus FARPO ablation."""

from collections import defaultdict
from typing import Any

import numpy as np


def _pair_sign(value: float, eps: float) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def compute_component_ablation_metrics(
    reward_metrics: dict[str, Any],
    group_ids: np.ndarray,
    component_keys: list[str],
    component_weights: list[float],
    eps: float = 1e-6,
    prefix: str = "ablation",
) -> dict[str, float]:
    """Compare both estimator geometries on the same rollout reward vectors.

    The returned values are diagnostics only and never affect the learning
    signal. Population standard deviations match the component estimators.
    """
    if len(component_keys) != len(component_weights):
        raise ValueError("component_keys and component_weights must have the same length")
    if len(component_keys) == 0:
        return {}

    arrays = []
    for key in component_keys:
        if key not in reward_metrics:
            raise KeyError(f"Missing reward component for ablation metrics: {key}")
        values = np.asarray(reward_metrics[key], dtype=np.float64).reshape(-1)
        arrays.append(values)

    sample_count = len(group_ids)
    if any(values.size != sample_count for values in arrays):
        raise ValueError("Reward component lengths must match group_ids")

    reward_matrix = np.stack(arrays, axis=1)
    weights = np.asarray(component_weights, dtype=np.float64)
    group_to_indices: dict[Any, list[int]] = defaultdict(list)
    for sample_idx, group_id in enumerate(group_ids):
        group_to_indices[group_id].append(sample_idx)

    standardized_components = np.zeros_like(reward_matrix)
    standardized_scalar = np.zeros(sample_count, dtype=np.float64)
    raw_scalar = reward_matrix @ weights
    component_stds: list[list[float]] = [[] for _ in component_keys]
    scalar_stds = []

    for indices in group_to_indices.values():
        if len(indices) <= 1:
            raise ValueError("Ablation metrics require more than one rollout per prompt")
        group_values = reward_matrix[indices]
        means = np.mean(group_values, axis=0)
        stds = np.std(group_values, axis=0)
        for component_idx, std in enumerate(stds):
            component_stds[component_idx].append(float(std))
            if std > eps:
                standardized_components[indices, component_idx] = (
                    group_values[:, component_idx] - means[component_idx]
                ) / (std + eps)

        group_scalar = raw_scalar[indices]
        scalar_mean = float(np.mean(group_scalar))
        scalar_std = float(np.std(group_scalar))
        scalar_stds.append(scalar_std)
        if scalar_std > eps:
            standardized_scalar[indices] = (group_scalar - scalar_mean) / (scalar_std + eps)

    farpo_prewhitened = standardized_components @ weights
    metrics: dict[str, float] = {
        f"{prefix}/group_count": float(len(group_to_indices)),
        f"{prefix}/sample_count": float(sample_count),
        f"{prefix}/scalar/group_std_mean": float(np.mean(scalar_stds)),
        f"{prefix}/scalar/active_group_ratio": float(np.mean(np.asarray(scalar_stds) > eps)),
        f"{prefix}/prewhiten/scalar_std": float(np.std(standardized_scalar)),
        f"{prefix}/prewhiten/farpo_std": float(np.std(farpo_prewhitened)),
    }

    for component_idx, key in enumerate(component_keys):
        std_values = np.asarray(component_stds[component_idx], dtype=np.float64)
        weighted_contribution = component_weights[component_idx] * standardized_components[:, component_idx]
        metrics[f"{prefix}/component/{key}/group_std_mean"] = float(np.mean(std_values))
        metrics[f"{prefix}/component/{key}/active_group_ratio"] = float(np.mean(std_values > eps))
        metrics[f"{prefix}/component/{key}/weighted_abs_contribution_mean"] = float(
            np.mean(np.abs(weighted_contribution))
        )

    distinct_vector_pairs = 0
    scalar_collisions = 0
    rank_changes = 0
    for indices in group_to_indices.values():
        for left_offset, left_idx in enumerate(indices):
            for right_idx in indices[left_offset + 1 :]:
                if np.max(np.abs(reward_matrix[left_idx] - reward_matrix[right_idx])) <= eps:
                    continue
                distinct_vector_pairs += 1
                scalar_delta = float(raw_scalar[left_idx] - raw_scalar[right_idx])
                farpo_delta = float(farpo_prewhitened[left_idx] - farpo_prewhitened[right_idx])
                scalar_sign = _pair_sign(scalar_delta, eps)
                farpo_sign = _pair_sign(farpo_delta, eps)
                if scalar_sign == 0:
                    scalar_collisions += 1
                if scalar_sign != farpo_sign:
                    rank_changes += 1

    denominator = max(distinct_vector_pairs, 1)
    metrics[f"{prefix}/distinct_reward_pair_count"] = float(distinct_vector_pairs)
    metrics[f"{prefix}/scalar_collision_ratio"] = scalar_collisions / denominator
    metrics[f"{prefix}/scalar_farpo_rank_change_ratio"] = rank_changes / denominator
    return metrics
