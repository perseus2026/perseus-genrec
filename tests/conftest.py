"""Shared deterministic settings for the GenRecSys test suite."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    """Keep model initialization and sampling reproducible across tests."""

    torch.manual_seed(0)


@pytest.fixture
def fp32_runtime(monkeypatch):
    """Isolate Accelerate state and verify FP32 despite a host FP16 setting."""
    from accelerate.state import AcceleratorState

    AcceleratorState._reset_state(reset_partial_state=True)
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "fp16")
    try:
        yield
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)
