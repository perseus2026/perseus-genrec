"""Memory-bounded Torch K-means tests."""

import numpy as np
import torch

from item_representations.semantic_ids.models.torch_kmeans import (
    TorchKMeans,
    pairwise_squared_distances,
)


def test_pairwise_squared_distances_matches_torch_cdist() -> None:
    generator = torch.Generator(device="cpu").manual_seed(17)
    values = torch.randn((31, 7), generator=generator)
    centers = torch.randn((5, 7), generator=generator)
    expected = torch.cdist(values, centers).square()

    actual = pairwise_squared_distances(values, centers)
    cached = pairwise_squared_distances(
        values,
        centers,
        x_squared_norms=torch.linalg.vector_norm(values, dim=1).square(),
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(cached, expected, rtol=1e-5, atol=1e-5)


def test_torch_kmeans_fit_does_not_call_cdist(monkeypatch) -> None:
    values = np.asarray(
        [
            [-4.1, -4.0],
            [-4.0, -3.9],
            [-3.9, -4.1],
            [4.1, 4.0],
            [4.0, 3.9],
            [3.9, 4.1],
        ],
        dtype=np.float32,
    )

    def fail_cdist(*args, **kwargs):
        raise AssertionError("memory-heavy torch.cdist must not be used")

    monkeypatch.setattr(torch, "cdist", fail_cdist)
    model = TorchKMeans(
        n_clusters=2,
        n_init=1,
        max_iter=20,
        random_state=42,
        device="cpu",
    )
    labels = model.fit_predict(values)

    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert labels[0] != labels[3]
    assert model.inertia_ is not None
    assert model.inertia_ < 0.2


def test_torch_kmeans_assignment_respects_fit_batch_size(monkeypatch) -> None:
    from item_representations.semantic_ids.models import torch_kmeans as implementation

    values = np.arange(42, dtype=np.float32).reshape(14, 3)
    centers = np.asarray([values[0], values[-1]], dtype=np.float32)
    observed_rows: list[int] = []
    original = implementation.pairwise_squared_distances

    def record_rows(X, active_centers, **kwargs):
        observed_rows.append(len(X))
        return original(X, active_centers, **kwargs)

    monkeypatch.setattr(implementation, "pairwise_squared_distances", record_rows)
    TorchKMeans(
        n_clusters=2,
        init=centers,
        max_iter=2,
        device="cpu",
        assignment_batch_size=4,
    ).fit(values)

    assert observed_rows
    assert max(observed_rows) <= 4
