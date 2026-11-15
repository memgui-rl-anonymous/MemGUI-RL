import pytest

from verl.trainer.config import AlgorithmConfig, DataConfig


def test_dataloader_worker_defaults_preserve_framework_behavior():
    config = DataConfig()

    assert config.train_dataloader_num_workers == 8
    assert config.val_dataloader_num_workers == 8
    assert config.dataloader_prefetch_factor == 2
    assert config.dataloader_persistent_workers is False


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("train_dataloader_num_workers", -1, "train_dataloader_num_workers"),
        ("val_dataloader_num_workers", -1, "val_dataloader_num_workers"),
        ("dataloader_prefetch_factor", 0, "dataloader_prefetch_factor"),
    ],
)
def test_invalid_dataloader_worker_config_is_rejected(field_name, value, message):
    config = DataConfig(**{field_name: value})

    with pytest.raises(ValueError, match=message):
        config.post_init()


@pytest.mark.parametrize("estimator", ["gdpo", "scalar_grpo_matched"])
def test_component_estimator_config_accepts_matched_keys_and_weights(estimator):
    config = AlgorithmConfig(
        adv_estimator=estimator,
        gdpo_reward_keys="format,action_type,action_params,folding",
        gdpo_reward_weights="0.1,0.4,0.4,0.1",
    )

    config.post_init()


def test_component_estimator_config_rejects_mismatched_keys_and_weights():
    config = AlgorithmConfig(
        adv_estimator="scalar_grpo_matched",
        gdpo_reward_keys="format,folding",
        gdpo_reward_weights="1.0",
    )

    with pytest.raises(ValueError, match="same length"):
        config.post_init()
