# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from ..protocol import DataProto


def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    return {key: np.mean(value) for key, value in metrics.items()}


def compute_length_metrics(batch: DataProto) -> dict[str, Any]:
    max_response_length = batch.batch["responses"].size(-1)
    max_prompt_length = batch.batch["attention_mask"].size(-1) - max_response_length

    prompt_length = batch.batch["attention_mask"][:, :-max_response_length].sum(-1).float()
    response_length = batch.batch["attention_mask"][:, -max_response_length:].sum(-1).float()

    return {
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.eq(response_length, max_response_length).float().mean().detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.eq(prompt_length, max_prompt_length).float().mean().detach().item(),
    }


def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> dict[str, Any]:
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].size(-1)
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    return {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        **compute_length_metrics(batch),
    }


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    num_response_tokens = torch.sum(batch.batch["response_mask"]).item()
    num_overall_tokens = sum(batch.meta_info["global_token_num"])
    num_tokens_of_section = {
        **dict.fromkeys(["gen", "reward"], num_response_tokens),
        **dict.fromkeys(["ref", "old", "values", "adv", "update_critic", "update_actor"], num_overall_tokens),
    }
    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], num_gpus: int) -> dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * num_gpus),
    }


def _compute_entropy_metrics(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    """Compute entropy-related metrics from log probabilities.

    Uses ``-log_prob(sampled_token)`` as an unbiased estimator of the token-level
    Shannon entropy ``H(p) = -∑ p(v) log p(v)`` (valid when the sampled token is
    drawn from the policy distribution itself).

    Args:
        log_probs:     shape (batch, response_length). Token log-probs under the
                       policy being measured (e.g. old_log_probs or ref_log_probs).
        response_mask: Boolean/float mask of the same shape – True for response tokens.
        prefix:        Metric name prefix, e.g. ``"policy"`` or ``"ref"``.

    Returns:
        Dictionary with the following keys (all floats):

        ``{prefix}/entropy/mean``
            Mean per-token entropy estimate over all valid (non-masked) tokens.
            Primary indicator of **entropy collapse**: monotonically decreasing →
            the model is becoming over-confident and losing diversity.

        ``{prefix}/entropy/std``
            Std of per-token entropy values across valid tokens.
            Low std = uniform confidence; high std = mixed confident/uncertain.

        ``{prefix}/entropy/min``
            Minimum per-token entropy (most confident token position across batch).

        ``{prefix}/entropy/max``
            Maximum per-token entropy (least confident token position across batch).

        ``{prefix}/entropy/per_seq_mean``
            Average of each sequence's mean token entropy (mean-of-means).
            Normalises out sequence-length effects; useful to compare across
            batches with different length distributions.

        ``{prefix}/entropy/per_seq_std``
            Std across sequences of their per-sequence entropy.
            Low → model has similar confidence on all prompts;
            High → some prompts still uncertain while others are already collapsed.
    """
    # neg_log_probs[i,t] ≈ H(p_{i,t})  (token-level entropy estimator)
    neg_lp = -log_probs  # (batch, response_length)
    mask = response_mask.bool()

    valid_neg_lp = neg_lp[mask]  # 1-D tensor of valid token entropies

    # per-sequence mean entropy
    seq_lengths = mask.sum(dim=-1).float().clamp(min=1)           # (batch,)
    seq_entropy = (neg_lp * mask.float()).sum(dim=-1) / seq_lengths  # (batch,)

    return {
        f"{prefix}/entropy/mean":         valid_neg_lp.mean().item(),
        f"{prefix}/entropy/std":          valid_neg_lp.std().item(),
        f"{prefix}/entropy/min":          valid_neg_lp.min().item(),
        f"{prefix}/entropy/max":          valid_neg_lp.max().item(),
        f"{prefix}/entropy/per_seq_mean": seq_entropy.mean().item(),
        f"{prefix}/entropy/per_seq_std":  seq_entropy.std().item(),
    }


def compute_entropy_metrics(batch: DataProto) -> dict[str, Any]:
    """Compute entropy metrics for training rollout.

    Computes entropy from old_log_probs (policy at rollout time) and optionally
    ref_log_probs (reference policy) to detect entropy collapse.

    Args:
        batch: Training batch with old_log_probs and optionally ref_log_probs.

    Returns:
        Dictionary of entropy metrics with prefixes:
        - "policy": entropy from old_log_probs (policy at rollout time)
        - "ref_policy": entropy from ref_log_probs (reference policy)
        - "policy/entropy/gap_vs_ref": difference between policy and ref entropy
    """
    entropy_metrics: dict[str, float] = {}

    if "old_log_probs" in batch.batch:
        # response_mask must align with old_log_probs
        old_lp = batch.batch["old_log_probs"]   # (batch, response_length)
        max_response_length = batch.batch["responses"].size(-1)
        response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()
        resp_mask = batch.batch.get("response_mask", response_mask)
        # make sure the mask matches log_probs
        if resp_mask.shape != old_lp.shape:
            resp_mask = response_mask
        entropy_metrics.update(_compute_entropy_metrics(old_lp, resp_mask, "policy"))

        if "ref_log_probs" in batch.batch:
            ref_lp = batch.batch["ref_log_probs"]  # (batch, response_length)
            entropy_metrics.update(_compute_entropy_metrics(ref_lp, resp_mask, "ref_policy"))

            # entropy gap: policy - ref (negative -> the policy is more collapsed than the reference)
            neg_old_lp = -old_lp
            neg_ref_lp = -ref_lp
            mask_f = resp_mask.bool()
            valid_gap = (neg_old_lp - neg_ref_lp)[mask_f]
            entropy_metrics["policy/entropy/gap_vs_ref_mean"] = valid_gap.mean().item()
            entropy_metrics["policy/entropy/gap_vs_ref_std"]  = valid_gap.std().item()

    return entropy_metrics


def _compute_group_stats(
    uid2scores: dict[str, list[float]],
    prefix: str,
    success_threshold: float = 0.9,
    zero_var_eps: float = 1e-4,
    action_type_weight: float = 0.5,
) -> dict[str, Any]:
    """Core computation of rollout-group statistics.

    Given a mapping from prompt-uid to the list of rollout scores in that group,
    computes a comprehensive set of metrics covering:
    ① Backward-compatible basics (all_fail / all_success / avg_success_rate …)
    ② Zero-variance detection — the *true* GRPO dead zone (advantage = 0)
    ③ Fine-grained all-above / all-below threshold breakdowns
    ④ Group distribution stats (min / std per group)
    ⑤ Rollout-level effective-training-signal ratio
    ⑥ Individual score distribution (semantically binned by action_type_weight)

    Args:
        uid2scores: Dict mapping uid → list of per-rollout scores in that group.
        prefix:     WandB metric key prefix (e.g. ``"rollout"`` or ``"val"``).
        success_threshold: Score strictly above this is considered "successful".
        zero_var_eps: Groups whose score std is below this are treated as zero-variance.
        action_type_weight: Weight assigned to the action-type sub-score.

    Returns:
        Dictionary of all group metrics.
    """
    total_prompts = len(uid2scores)
    if total_prompts == 0:
        return {}

    # score_dist bin thresholds (weight-aware)
    _dist_lo = action_type_weight * 0.5
    _dist_hi = action_type_weight * 1.1

    # group-level counters
    all_fail_count = 0
    all_success_count = 0
    zero_var_count = 0
    all_zero_count = 0
    all_perfect_count = 0
    all_type_only_max_count = 0

    # zero-variance internal 4-way breakdown
    zv_wrong_count = 0
    zv_type_only_count = 0
    zv_partial_count = 0
    zv_perfect_count = 0

    group_max_scores: list[float] = []
    group_min_scores: list[float] = []
    group_mean_scores: list[float] = []
    group_std_scores: list[float] = []
    group_success_rates: list[float] = []

    # rollout-level accumulators
    all_rollout_scores: list[float] = []
    effective_rollout_count = 0
    total_rollout_count = 0

    for scores in uid2scores.values():
        n_r = len(scores)
        total_rollout_count += n_r
        sa = np.array(scores, dtype=np.float64)

        max_s = float(sa.max())
        min_s = float(sa.min())
        mean_s = float(sa.mean())
        std_s = float(sa.std())

        group_max_scores.append(max_s)
        group_min_scores.append(min_s)
        group_mean_scores.append(mean_s)
        group_std_scores.append(std_s)
        group_success_rates.append(float((sa > success_threshold).sum()) / n_r)
        all_rollout_scores.extend(scores)

        # zero variance
        if std_s < zero_var_eps:
            zero_var_count += 1
            if mean_s < _dist_lo:
                zv_wrong_count += 1
            elif mean_s <= _dist_hi:
                zv_type_only_count += 1
            elif mean_s < 0.99:
                zv_partial_count += 1
            else:
                zv_perfect_count += 1
        else:
            effective_rollout_count += n_r

        # backward-compatible all_fail / all_success
        if max_s <= success_threshold:
            all_fail_count += 1
        if min_s > success_threshold:
            all_success_count += 1

        # semantic extremes
        if max_s < 0.10:
            all_zero_count += 1
        if min_s >= 0.99:
            all_perfect_count += 1

        # no-params-credit group
        if max_s <= _dist_hi:
            all_type_only_max_count += 1

    p = total_prompts
    r = max(total_rollout_count, 1)
    all_scores_arr = np.array(all_rollout_scores, dtype=np.float64)

    return {
        # backward-compatible basics
        f"{prefix}/total_prompts":          p,
        f"{prefix}/all_fail_count":         all_fail_count,
        f"{prefix}/all_fail_ratio":         all_fail_count / p,
        f"{prefix}/all_success_count":      all_success_count,
        f"{prefix}/all_success_ratio":      all_success_count / p,
        f"{prefix}/any_success_ratio":      1.0 - all_fail_count / p,
        f"{prefix}/avg_success_rate":       float(np.mean(group_success_rates)),
        f"{prefix}/group_max_score/mean":   float(np.mean(group_max_scores)),
        f"{prefix}/group_mean_score/mean":  float(np.mean(group_mean_scores)),

        # true GRPO dead zones (zero variance)
        f"{prefix}/zero_variance_count":    zero_var_count,
        f"{prefix}/zero_variance_ratio":    zero_var_count / p,
        f"{prefix}/has_gradient_ratio":     1.0 - zero_var_count / p,

        # zero-variance internal breakdown
        f"{prefix}/zv_wrong_count":         zv_wrong_count,
        f"{prefix}/zv_wrong_ratio":         zv_wrong_count / p,
        f"{prefix}/zv_type_only_count":     zv_type_only_count,
        f"{prefix}/zv_type_only_ratio":     zv_type_only_count / p,
        f"{prefix}/zv_partial_count":       zv_partial_count,
        f"{prefix}/zv_partial_ratio":       zv_partial_count / p,
        f"{prefix}/zv_perfect_count":      zv_perfect_count,
        f"{prefix}/zv_perfect_ratio":       zv_perfect_count / p,

        # zero-variance internal relative ratio (within zero_variance groups)
        **({
            f"{prefix}/zv_composition/all_wrong":    zv_wrong_count      / zero_var_count,
            f"{prefix}/zv_composition/type_only":    zv_type_only_count  / zero_var_count,
            f"{prefix}/zv_composition/partial_stuck": zv_partial_count   / zero_var_count,
            f"{prefix}/zv_composition/all_perfect":  zv_perfect_count    / zero_var_count,
        } if zero_var_count > 0 else {}),

        # fine-grained dead-zone semantic breakdown
        f"{prefix}/all_zero_count":               all_zero_count,
        f"{prefix}/all_zero_ratio":               all_zero_count / p,
        f"{prefix}/all_perfect_count":            all_perfect_count,
        f"{prefix}/all_perfect_ratio":            all_perfect_count / p,
        f"{prefix}/all_type_only_max_count":      all_type_only_max_count,
        f"{prefix}/all_type_only_max_ratio":      all_type_only_max_count / p,

        # group distribution stats
        f"{prefix}/group_min_score/mean":   float(np.mean(group_min_scores)),
        f"{prefix}/group_std/mean":         float(np.mean(group_std_scores)),

        # rollout-level effective training-signal ratio
        f"{prefix}/effective_rollout_ratio": effective_rollout_count / r,

        # weight-aware 4-bin score distribution
        f"{prefix}/score_dist/type_wrong":  float(np.mean(all_scores_arr < _dist_lo)),
        f"{prefix}/score_dist/type_only":   float(np.mean((all_scores_arr >= _dist_lo) & (all_scores_arr <= _dist_hi))),
        f"{prefix}/score_dist/partial":     float(np.mean((all_scores_arr > _dist_hi)  & (all_scores_arr < 0.99))),
        f"{prefix}/score_dist/perfect":     float(np.mean(all_scores_arr >= 0.99)),
    }


def compute_rollout_group_metrics(
    batch: DataProto,
    n: int,
    success_threshold: float = 0.9,
    prefix: str = "rollout",
    action_type_weight: float = 0.5,
) -> dict[str, Any]:
    """Compute per-prompt group metrics for n rollouts (training).

    Args:
        batch: Training batch after rewards have been computed.
        n: Number of rollouts per prompt.
        success_threshold: Score strictly above this counts as "successful".
        prefix: WandB metric key prefix.
        action_type_weight: Weight for the action-type sub-score.

    Returns:
        Dictionary of group metrics.
    """
    sequence_scores = batch.batch["token_level_scores"].sum(-1)
    uids = batch.non_tensor_batch["uid"]

    uid2scores: dict[str, list[float]] = defaultdict(list)
    for uid, score in zip(uids, sequence_scores.cpu().tolist()):
        uid2scores[uid].append(score)

    return _compute_group_stats(
        uid2scores,
        prefix=prefix,
        success_threshold=success_threshold,
        action_type_weight=action_type_weight,
    )


def compute_rollout_group_metrics_from_scores(
    scores: np.ndarray,
    n: int,
    success_threshold: float = 0.9,
    prefix: str = "val",
    action_type_weight: float = 0.5,
) -> dict[str, Any]:
    """Compute rollout group metrics from a flat score array (validation).

    Assumes scores are ordered as n consecutive rollouts per prompt.

    Args:
        scores: 1-D array of per-rollout scores.
        n: Number of rollouts per prompt.
        success_threshold: Score strictly above this counts as "successful".
        prefix: WandB metric key prefix.
        action_type_weight: Weight for the action-type sub-score.

    Returns:
        Dictionary of group metrics.
    """
    n_groups = len(scores) // n
    if n_groups == 0:
        return {}

    grouped = scores[: n_groups * n].reshape(n_groups, n)
    uid2scores: dict[str, list[float]] = {
        str(i): grouped[i].tolist() for i in range(n_groups)
    }
    return _compute_group_stats(
        uid2scores,
        prefix=prefix,
        success_threshold=success_threshold,
        action_type_weight=action_type_weight,
    )


# MemGUI-specific action types for monitoring
MEMGUI_ACTION_TYPES = [
    "click", "long_press", "swipe", "type", "system_button",
    "wait", "terminate", "answer", "open_app",
    "memory_add", "memory_update", "memory_delete", "no_action"
]


def compute_action_type_metrics(
    reward_metrics: dict[str, list[float]],
    group_keys: dict[str, np.ndarray],
    prefix: str = "reward",
) -> dict[str, Any]:
    """Compute per-action-type reward metrics for MemGUI.

    Args:
        reward_metrics: Dict of metric_name -> list[float] from reward computation.
            Expected keys: "overall", "action_type", "action_params", "format", "folding".
        group_keys: Dict containing grouping arrays:
            - "action_type": np.ndarray of action types per sample
        prefix: Metric key prefix (e.g. "reward" or "val").

    Returns:
        Dictionary of per-action-type metrics ready for WandB logging:
        - ``{prefix}/action/{action_type}/overall``: mean overall score per action type
        - ``{prefix}/action/{action_type}/count``: count of samples per action type
        - ``{prefix}/action/memory_*``: memory-specific metrics
    """
    metrics = {}

    overall_scores = reward_metrics.get("overall", [])
    if np.isscalar(overall_scores):
        overall_scores = [float(overall_scores)]
    else:
        overall_scores = list(overall_scores)
    if not overall_scores:
        return metrics

    n_samples = len(overall_scores)
    action_types = group_keys.get("action_type", None)

    if action_types is not None and len(action_types) == n_samples:
        action2scores = defaultdict(lambda: defaultdict(list))

        # Separate memory actions from UI actions
        memory_actions = ["memory_add", "memory_update", "memory_delete"]
        ui_action_scores = []
        memory_action_scores = []

        for i in range(n_samples):
            action = str(action_types[i])
            action2scores[action]["overall"].append(overall_scores[i])

            # Track memory vs UI actions separately
            if action in memory_actions:
                memory_action_scores.append(overall_scores[i])
            else:
                ui_action_scores.append(overall_scores[i])

        # Per-action-type metrics
        for action, scores_dict in action2scores.items():
            for metric_name, values in scores_dict.items():
                metrics[f"{prefix}/action/{action}/{metric_name}"] = float(np.mean(values))
            metrics[f"{prefix}/action/{action}/count"] = len(scores_dict["overall"])

        # Aggregate: UI vs Memory
        if ui_action_scores:
            metrics[f"{prefix}/action/ui/overall_mean"] = float(np.mean(ui_action_scores))
            metrics[f"{prefix}/action/ui/count"] = len(ui_action_scores)
        if memory_action_scores:
            metrics[f"{prefix}/action/memory/overall_mean"] = float(np.mean(memory_action_scores))
            metrics[f"{prefix}/action/memory/count"] = len(memory_action_scores)
            # Memory action breakdown
            for action in memory_actions:
                action_specific = [s for i, s in enumerate(overall_scores) if str(action_types[i]) == action]
                if action_specific:
                    metrics[f"{prefix}/action/{action}/overall_mean"] = float(np.mean(action_specific))

    return metrics


def compute_folding_metrics(
    reward_metrics: dict[str, list[float]],
    group_keys: dict[str, np.ndarray],
    prefix: str = "reward",
) -> dict[str, Any]:
    """Compute folding-specific metrics for MemGUI.

    Monitors:
    - Fold depth distribution (how many steps are folded at once)
    - Folding instruction accuracy when GT has folding
    - Fold vs no-fold ratio

    Args:
        reward_metrics: Dict of metric_name -> list[float] from reward computation.
            Expected keys: "folding", "overall".
        group_keys: Dict containing grouping arrays:
            - "has_folding_gt": np.ndarray of bools indicating if GT has folding
            - "fold_depth": np.ndarray of fold depths (optional)
        prefix: Metric key prefix (e.g. "reward" or "val").

    Returns:
        Dictionary of folding metrics ready for WandB logging:
        - ``{prefix}/folding/accuracy``: folding instruction accuracy
        - ``{prefix}/folding/with_gt_ratio``: ratio of samples with folding GT
        - ``{prefix}/folding/depth_mean``: mean fold depth
    """
    metrics = {}

    def _as_float_array(values) -> np.ndarray:
        if values is None:
            return np.array([], dtype=np.float64)
        if isinstance(values, (int, float, np.floating)):
            return np.array([float(values)], dtype=np.float64)
        if isinstance(values, np.ndarray):
            return values.astype(np.float64)
        if isinstance(values, list):
            return np.array(values, dtype=np.float64)
        return np.array(list(values), dtype=np.float64) if hasattr(values, "__iter__") else np.array([float(values)])

    def _add_depth_stats(metric_prefix: str, depths: np.ndarray) -> None:
        valid_depths = depths[depths > 0]
        metrics[f"{metric_prefix}/depth_count"] = int(valid_depths.size)
        if valid_depths.size == 0:
            return

        metrics[f"{metric_prefix}/depth_mean"] = float(np.mean(valid_depths))
        metrics[f"{metric_prefix}/depth_min"] = float(np.min(valid_depths))
        metrics[f"{metric_prefix}/depth_max"] = float(np.max(valid_depths))
        metrics[f"{metric_prefix}/depth_std"] = float(np.std(valid_depths))
        metrics[f"{metric_prefix}/depth_p50"] = float(np.percentile(valid_depths, 50))
        metrics[f"{metric_prefix}/depth_p90"] = float(np.percentile(valid_depths, 90))
        metrics[f"{metric_prefix}/depth_p95"] = float(np.percentile(valid_depths, 95))

        for depth in [1, 2, 3, 4, 5]:
            depth_count = int(np.sum(valid_depths == depth))
            metrics[f"{metric_prefix}/depth_{depth}_count"] = depth_count
            metrics[f"{metric_prefix}/depth_{depth}_ratio"] = depth_count / valid_depths.size

        deep_fold_count = int(np.sum(valid_depths > 5))
        metrics[f"{metric_prefix}/depth_5+_count"] = deep_fold_count
        metrics[f"{metric_prefix}/depth_5+_ratio"] = deep_fold_count / valid_depths.size

    # New MemGUI folding observability metrics. These are emitted by
    # examples/reward_function/r1gui_memgui.py for each rollout response.
    gt_has = _as_float_array(reward_metrics.get("gt_has_folding", None))
    pred_has = _as_float_array(reward_metrics.get("pred_has_folding", None))
    gt_depths = _as_float_array(reward_metrics.get("gt_fold_depth", None))
    pred_depths = _as_float_array(reward_metrics.get("pred_fold_depth", None))

    if gt_has.size > 0:
        n_samples = gt_has.size
        gt_mask = gt_has > 0.5
        pred_mask = pred_has > 0.5 if pred_has.size == n_samples else np.zeros(n_samples, dtype=bool)
        no_gt_mask = ~gt_mask
        gt_stats_mask_base = np.ones(n_samples, dtype=bool)
        uids = group_keys.get("uid", None)
        if uids is not None and len(uids) == n_samples:
            gt_stats_mask_base = np.zeros(n_samples, dtype=bool)
            seen_uids = set()
            for idx, uid in enumerate(uids):
                if uid not in seen_uids:
                    gt_stats_mask_base[idx] = True
                    seen_uids.add(uid)

        pred_metric_prefix = {
            "train": "train/rollout_folding",
            "val": "val/inference_folding",
        }.get(prefix, f"{prefix}/pred_folding")

        gt_count = int(np.sum(gt_mask & gt_stats_mask_base))
        gt_denominator = int(np.sum(gt_stats_mask_base))
        gt_response_count = int(np.sum(gt_mask))
        pred_count = int(np.sum(pred_mask))
        no_gt_count = int(np.sum(no_gt_mask))
        both_present = int(np.sum(gt_mask & pred_mask))
        missing_count = int(np.sum(gt_mask & ~pred_mask))
        unexpected_count = int(np.sum(no_gt_mask & pred_mask))

        metrics[f"{prefix}/gt_folding/has_ratio"] = gt_count / gt_denominator if gt_denominator > 0 else 0.0
        metrics[f"{prefix}/gt_folding/count"] = gt_count
        metrics[f"{pred_metric_prefix}/has_ratio"] = pred_count / n_samples
        metrics[f"{pred_metric_prefix}/count"] = pred_count
        metrics[f"{prefix}/folding/pred_when_gt_ratio"] = both_present / gt_response_count if gt_response_count > 0 else 0.0
        metrics[f"{prefix}/folding/missing_count"] = missing_count
        metrics[f"{prefix}/folding/missing_ratio"] = missing_count / n_samples
        metrics[f"{prefix}/folding/missing_given_gt_ratio"] = (
            missing_count / gt_response_count if gt_response_count > 0 else 0.0
        )
        metrics[f"{prefix}/folding/unexpected_count"] = unexpected_count
        metrics[f"{prefix}/folding/unexpected_ratio"] = unexpected_count / n_samples
        metrics[f"{prefix}/folding/unexpected_given_no_gt_ratio"] = (
            unexpected_count / no_gt_count if no_gt_count > 0 else 0.0
        )

        if gt_depths.size == n_samples:
            gt_stats_mask = gt_mask & gt_stats_mask_base
            _add_depth_stats(f"{prefix}/gt_folding", gt_depths[gt_stats_mask])
            gt_valid = _as_float_array(reward_metrics.get("gt_fold_range_valid", None))
            if gt_valid.size == n_samples and gt_count > 0:
                metrics[f"{prefix}/gt_folding/invalid_range_ratio"] = float(
                    np.sum(gt_stats_mask & (gt_valid <= 0.5)) / gt_count
                )

        if pred_depths.size == n_samples:
            _add_depth_stats(pred_metric_prefix, pred_depths[pred_mask])
            pred_valid = _as_float_array(reward_metrics.get("pred_fold_range_valid", None))
            if pred_valid.size == n_samples and pred_count > 0:
                metrics[f"{pred_metric_prefix}/invalid_range_ratio"] = float(
                    np.sum(pred_mask & (pred_valid <= 0.5)) / pred_count
                )

        both_valid = _as_float_array(reward_metrics.get("fold_both_valid_range", None))
        if both_valid.size == n_samples:
            both_valid_mask = both_valid > 0.5
            both_valid_count = int(np.sum(both_valid_mask))
            metrics[f"{prefix}/folding/both_valid_range_count"] = both_valid_count
            metrics[f"{prefix}/folding/both_valid_range_ratio"] = both_valid_count / n_samples

            if both_valid_count > 0:
                range_iou = _as_float_array(reward_metrics.get("fold_range_iou", None))
                exact_match = _as_float_array(reward_metrics.get("fold_exact_range_match", None))
                abs_error = _as_float_array(reward_metrics.get("fold_depth_abs_error", None))
                signed_error = _as_float_array(reward_metrics.get("fold_depth_signed_error", None))
                deeper_or_equal = _as_float_array(reward_metrics.get("fold_pred_deeper_or_equal", None))

                if range_iou.size == n_samples:
                    metrics[f"{prefix}/folding/range_iou_mean"] = float(np.mean(range_iou[both_valid_mask]))
                    metrics[f"{prefix}/folding/range_iou_min"] = float(np.min(range_iou[both_valid_mask]))
                if exact_match.size == n_samples:
                    metrics[f"{prefix}/folding/exact_range_match_ratio"] = float(np.mean(exact_match[both_valid_mask]))
                if abs_error.size == n_samples:
                    metrics[f"{prefix}/folding/depth_abs_error_mean"] = float(np.mean(abs_error[both_valid_mask]))
                    metrics[f"{prefix}/folding/depth_abs_error_max"] = float(np.max(abs_error[both_valid_mask]))
                if signed_error.size == n_samples:
                    metrics[f"{prefix}/folding/depth_signed_error_mean"] = float(np.mean(signed_error[both_valid_mask]))
                if deeper_or_equal.size == n_samples:
                    metrics[f"{prefix}/folding/pred_deeper_or_equal_ratio"] = float(
                        np.mean(deeper_or_equal[both_valid_mask])
                    )

    folding_scores = _as_float_array(reward_metrics.get("folding", []))

    if folding_scores.size == 0:
        return metrics

    n_samples = len(folding_scores)

    # Folding accuracy (when GT has folding)
    has_folding_gt = group_keys.get("has_folding_gt", None)
    if has_folding_gt is not None and len(has_folding_gt) == n_samples:
        folding_with_gt = [folding_scores[i] for i in range(n_samples) if has_folding_gt[i]]
        if folding_with_gt:
            metrics[f"{prefix}/folding/accuracy"] = float(np.mean(folding_with_gt))
        metrics[f"{prefix}/folding/with_gt_ratio"] = float(np.mean([1 if x else 0 for x in has_folding_gt]))

    # Fold depth metrics
    fold_depths = group_keys.get("fold_depth", None)
    if fold_depths is not None and len(fold_depths) == n_samples:
        valid_depths = [d for d in fold_depths if d is not None and d > 0]
        if valid_depths:
            metrics[f"{prefix}/folding/depth_mean"] = float(np.mean(valid_depths))
            metrics[f"{prefix}/folding/depth_std"] = float(np.std(valid_depths))
            metrics[f"{prefix}/folding/depth_max"] = float(np.max(valid_depths))
            # Depth distribution
            for depth in [1, 2, 3, 4, 5]:
                depth_count = sum(1 for d in valid_depths if d == depth)
                if depth_count > 0:
                    metrics[f"{prefix}/folding/depth_{depth}_count"] = depth_count
                    metrics[f"{prefix}/folding/depth_{depth}_ratio"] = depth_count / len(valid_depths)
            # Depth > 5 (deep fold)
            deep_fold_count = sum(1 for d in valid_depths if d > 5)
            if deep_fold_count > 0:
                metrics[f"{prefix}/folding/depth_5+_count"] = deep_fold_count
                metrics[f"{prefix}/folding/depth_5+_ratio"] = deep_fold_count / len(valid_depths)

    return metrics
