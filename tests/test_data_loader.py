import pytest

from verl.trainer.data_loader import _dataloader_worker_kwargs


def test_single_process_dataloader_omits_multiprocessing_only_options():
    kwargs = _dataloader_worker_kwargs(
        num_workers=0,
        prefetch_factor=1,
        persistent_workers=True,
    )

    assert kwargs == {"num_workers": 0}


def test_multiprocess_dataloader_applies_prefetch_options():
    kwargs = _dataloader_worker_kwargs(
        num_workers=2,
        prefetch_factor=1,
        persistent_workers=True,
    )

    assert kwargs == {
        "num_workers": 2,
        "prefetch_factor": 1,
        "persistent_workers": True,
    }


@pytest.mark.parametrize(
    ("num_workers", "prefetch_factor"),
    [(-1, 1), (1, 0)],
)
def test_invalid_dataloader_worker_options_are_rejected(num_workers, prefetch_factor):
    with pytest.raises(ValueError):
        _dataloader_worker_kwargs(
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=False,
        )
