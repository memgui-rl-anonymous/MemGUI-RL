import numpy as np
import pytest

from verl.trainer.ablation_metrics import compute_component_ablation_metrics


def test_component_ablation_metrics_report_group_activity_and_collisions():
    metrics = compute_component_ablation_metrics(
        reward_metrics={
            "first": [1.0, 0.0, 0.0, 1.0],
            "second": [0.0, 1.0, 0.0, 1.0],
        },
        group_ids=np.array(["a", "a", "b", "b"], dtype=object),
        component_keys=["first", "second"],
        component_weights=[0.5, 0.5],
        eps=0.0,
    )

    assert metrics["ablation/group_count"] == 2.0
    assert metrics["ablation/sample_count"] == 4.0
    assert metrics["ablation/component/first/active_group_ratio"] == 1.0
    assert metrics["ablation/component/second/active_group_ratio"] == 1.0
    assert metrics["ablation/scalar_collision_ratio"] == pytest.approx(0.5)


def test_component_ablation_metrics_require_every_configured_component():
    with pytest.raises(KeyError, match="folding"):
        compute_component_ablation_metrics(
            reward_metrics={"format": [0.0, 1.0]},
            group_ids=np.array(["a", "a"], dtype=object),
            component_keys=["format", "folding"],
            component_weights=[0.5, 0.5],
        )
