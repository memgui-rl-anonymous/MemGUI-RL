import numpy as np
import pytest
import torch

from verl.trainer.core_algos import (
    compute_gdpo_outcome_advantage,
    compute_scalar_grpo_matched_outcome_advantage,
)


def _inputs(second_scale: float = 1.0):
    response_mask = torch.ones((3, 2), dtype=torch.float32)
    token_rewards = torch.zeros_like(response_mask)
    index = np.array(["prompt", "prompt", "prompt"], dtype=object)
    components = {
        "first": torch.tensor([0.0, 1.0, 0.0]),
        "second": second_scale * torch.tensor([0.0, 0.0, 0.2]),
    }
    weights = {"first": 0.5, "second": 0.5}
    return token_rewards, response_mask, index, components, weights


def test_normalize_then_aggregate_is_invariant_to_positive_component_rescaling():
    original = _inputs(second_scale=1.0)
    rescaled = _inputs(second_scale=10.0)

    farpo_original, _ = compute_gdpo_outcome_advantage(
        token_level_rewards=original[0],
        response_mask=original[1],
        index=original[2],
        reward_components=original[3],
        gdpo_weights=original[4],
        eps=0.0,
    )
    farpo_rescaled, _ = compute_gdpo_outcome_advantage(
        token_level_rewards=rescaled[0],
        response_mask=rescaled[1],
        index=rescaled[2],
        reward_components=rescaled[3],
        gdpo_weights=rescaled[4],
        eps=0.0,
    )

    torch.testing.assert_close(farpo_original, farpo_rescaled)


def test_matched_scalar_baseline_remains_sensitive_to_component_rescaling():
    original = _inputs(second_scale=1.0)
    rescaled = _inputs(second_scale=10.0)

    scalar_original, _ = compute_scalar_grpo_matched_outcome_advantage(
        token_level_rewards=original[0],
        response_mask=original[1],
        index=original[2],
        reward_components=original[3],
        gdpo_weights=original[4],
        eps=0.0,
    )
    scalar_rescaled, _ = compute_scalar_grpo_matched_outcome_advantage(
        token_level_rewards=rescaled[0],
        response_mask=rescaled[1],
        index=rescaled[2],
        reward_components=rescaled[3],
        gdpo_weights=rescaled[4],
        eps=0.0,
    )

    with pytest.raises(AssertionError):
        torch.testing.assert_close(scalar_original, scalar_rescaled)


def test_estimators_are_identical_when_only_one_component_is_used():
    token_rewards, response_mask, index, components, _ = _inputs()
    one_component = {"first": components["first"]}
    weights = {"first": 0.4}

    scalar, _ = compute_scalar_grpo_matched_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=index,
        reward_components=one_component,
        gdpo_weights=weights,
    )
    farpo, _ = compute_gdpo_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=index,
        reward_components=one_component,
        gdpo_weights=weights,
    )

    torch.testing.assert_close(scalar, farpo)


@pytest.mark.parametrize(
    "estimator",
    [compute_scalar_grpo_matched_outcome_advantage, compute_gdpo_outcome_advantage],
)
def test_component_estimators_refuse_unmatched_overall_fallback(estimator):
    token_rewards, response_mask, index, _, weights = _inputs()

    with pytest.raises(ValueError, match="requires raw reward_components"):
        estimator(
            token_level_rewards=token_rewards,
            response_mask=response_mask,
            index=index,
            reward_components=None,
            gdpo_weights=weights,
        )
