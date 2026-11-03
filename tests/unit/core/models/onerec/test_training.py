"""Optimizer-update schedules and accumulation through the real CPU training loop."""

import math

import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from genrecsys.core.config import Config, Scheduler, Training
from genrecsys.core.models.onerec.training import _scheduler
from genrecsys.core.models.training import run_training


def test_cosine_schedule_counts_updates_and_reaches_peak_and_floor():
    config = Config(
        training=Training(
            num_epochs=2,
            gradient_accumulation_steps=2,
            scheduler=Scheduler(
                "cosine_warmup", {"warmup_ratio": 1 / 3, "min_lr_ratio": 0.1}
            ),
        )
    )
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=4e-4)
    scheduler = _scheduler(optimizer, config, steps_per_epoch=5)
    ratios = [optimizer.param_groups[0]["lr"] / 4e-4]
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        ratios.append(optimizer.param_groups[0]["lr"] / 4e-4)
    expected = [0, 0.5, 1] + [
        0.1 + 0.9 * (1 + math.cos(math.pi * step / 4)) / 2 for step in range(1, 5)
    ]
    assert ratios == pytest.approx(expected)


@pytest.mark.parametrize("accumulation", [1, 2, 3])
def test_training_steps_scheduler_once_per_update_including_partial_window(
    accumulation, fp32_runtime
):
    config = Config(
        training=Training(
            num_epochs=2,
            gradient_accumulation_steps=accumulation,
            log_every_n_train_steps=10,
            scheduler=Scheduler(
                "cosine_warmup", {"warmup_ratio": 0.1, "min_lr_ratio": 0.1}
            ),
        )
    )
    accelerator = Accelerator(
        mixed_precision="no", gradient_accumulation_steps=accumulation
    )
    loader = accelerator.prepare_data_loader(
        DataLoader(
            [{"target": torch.tensor([float(i)])} for i in range(9)], batch_size=2
        )
    )

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, batch, labels):
            output = self.linear(labels)
            return (output - labels).square().mean(), output

    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = _scheduler(optimizer, config, steps_per_epoch=len(loader))
    model, optimizer = accelerator.prepare(model, optimizer)
    _, history, _ = run_training(
        model,
        optimizer,
        scheduler,
        loader,
        config.training,
        accelerator,
        None,
        dict,
        to_device=lambda batch, device: batch,
    )
    assert scheduler.last_epoch == 2 * math.ceil(5 / accumulation)
    assert history[-1]["learning_rate"] == pytest.approx(0.001)


@pytest.mark.parametrize(
    "kind,params",
    [
        ("constant", {"warmup_ratio": 0.1}),
        ("cosine_warmup", {"warmup_ratio": 1, "min_lr_ratio": 0.1}),
        ("cosine_warmup", {"warmup_ratio": 0.1, "min_lr_ratio": 1.1}),
        ("cosine_warmup", {"warmup_ratio": 0.1, "min_lr_ratio": -0.1}),
        ("cosine_warmup", {"warmup_ratio": 0.1}),
        ("unknown", {}),
    ],
)
def test_invalid_scheduler_settings_fail_during_config_validation(kind, params):
    config = Config.load("tests/fixtures/framework/sequence.yaml")
    config.training.scheduler = Scheduler(kind, params)
    with pytest.raises(ValueError):
        config.validate()
