"""Torch implementation of K-means used by semantic-ID representations."""

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch


def pairwise_squared_distances(
    X: torch.Tensor,
    centers: torch.Tensor,
    *,
    x_squared_norms: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute squared L2 distances from row norms and a matrix product.

    Use ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x c^T to avoid torch.cdist input copies.
    """
    if X.ndim != 2 or centers.ndim != 2:
        raise ValueError("X and centers must both be two-dimensional")
    if X.shape[1] != centers.shape[1]:
        raise ValueError(
            f"Feature dimensions differ: {X.shape[1]} != {centers.shape[1]}"
        )
    if x_squared_norms is None:
        x_squared_norms = torch.linalg.vector_norm(X, dim=1).square_()
    elif x_squared_norms.shape != (X.shape[0],):
        raise ValueError(
            f"Expected x_squared_norms shape {(X.shape[0],)}, "
            f"got {tuple(x_squared_norms.shape)}"
        )
    center_squared_norms = torch.linalg.vector_norm(centers, dim=1).square_()
    distances = torch.mm(X, centers.transpose(0, 1))
    distances.mul_(-2.0)
    distances.add_(x_squared_norms.unsqueeze(1))
    distances.add_(center_squared_norms.unsqueeze(0))
    return distances.clamp_min_(0.0)


class TorchKMeans:
    """Fit Lloyd KMeans on CPU or GPU, optionally using balanced assignment.

    Support k-means++, random, or explicit centers; retain the lowest-inertia run.
    Expose cluster_centers_, labels_, inertia_, and n_iter_. Copy all input vectors
    to the device; assignment batching bounds distance buffers only.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: Union[str, np.ndarray, torch.Tensor] = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: int = 0,
        device: Optional[str] = None,
        balanced: bool = False,
        assignment_batch_size: int | None = None,
    ):
        """Configure Lloyd/balanced KMeans and assignment batch limits on the target device."""

        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.balanced = balanced
        if assignment_batch_size is not None and assignment_batch_size <= 0:
            raise ValueError("assignment_batch_size must be positive")
        self.assignment_batch_size = assignment_batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._random_gen = self._make_generator(random_state)

        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None
        self.stop_reason_: str | None = None
        self.center_shift_sq_: float | None = None

    def _make_generator(self, seed: Optional[int]) -> Optional[torch.Generator]:
        """Create a seeded device generator, or use the shared RNG."""

        if seed is None:
            return None
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        return gen

    def _kmeans_plus_plus(
        self,
        X: torch.Tensor,
        gen: Optional[torch.Generator],
        x_squared_norms: torch.Tensor,
    ) -> torch.Tensor:
        """Select k-means++ centers with local trials and squared-distance sampling."""

        n_samples, n_features = X.shape
        n_local_trials = 2 + int(math.log(self.n_clusters))

        rand_kwargs = {"device": self.device}
        if gen is not None:
            rand_kwargs["generator"] = gen

        first_idx = torch.randint(n_samples, (1,), **rand_kwargs).item()
        centers = torch.empty(
            (self.n_clusters, n_features), dtype=X.dtype, device=self.device
        )
        centers[0] = X[first_idx]

        closest_dist_sq = pairwise_squared_distances(
            X,
            centers[:1],
            x_squared_norms=x_squared_norms,
        ).squeeze(1)

        for c in range(1, self.n_clusters):
            total = closest_dist_sq.sum()
            if total.item() == 0:
                candidate_ids = torch.randint(
                    n_samples, (n_local_trials,), **rand_kwargs
                )
            else:
                probs = closest_dist_sq / total
                if gen is not None:
                    candidate_ids = torch.multinomial(
                        probs, n_local_trials, replacement=True, generator=gen
                    )
                else:
                    candidate_ids = torch.multinomial(
                        probs, n_local_trials, replacement=True
                    )

            candidates = X[candidate_ids]
            dist_sq_to_cand = pairwise_squared_distances(
                X,
                candidates,
                x_squared_norms=x_squared_norms,
            )
            new_closest = torch.minimum(closest_dist_sq.unsqueeze(1), dist_sq_to_cand)
            candidate_potentials = new_closest.sum(dim=0)

            best = torch.argmin(candidate_potentials).item()
            centers[c] = candidates[best]
            # Clone the selected column so it does not retain the complete
            # n_samples x n_local_trials allocation through the next step.
            closest_dist_sq = new_closest[:, best].clone()
            del candidates, dist_sq_to_cand, new_closest, candidate_potentials
            if self.verbose and (
                (c + 1) % max(1, self.n_clusters // 16) == 0 or c + 1 == self.n_clusters
            ):
                print(
                    f"K-means++ initialized {c + 1}/{self.n_clusters} centers.",
                    flush=True,
                )

        return centers

    def _kmeans_single(
        self,
        X: torch.Tensor,
        centers: torch.Tensor,
        x_squared_norms: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, int, str, float]:
        """Run one Lloyd fit and return centers, labels, and convergence diagnostics."""

        _, n_features = X.shape
        K = centers.shape[0]
        labels_prev: Optional[torch.Tensor] = None
        completed_iter = 0
        strict_convergence = False
        stop_reason = "max_iter"
        center_shift_sq = float("inf")

        for i in range(self.max_iter):
            labels, point_distances = self._assign(
                X,
                centers,
                x_squared_norms,
            )

            strict_convergence = labels_prev is not None and torch.equal(
                labels, labels_prev
            )

            new_centers = torch.zeros_like(centers)
            new_centers.scatter_add_(0, labels.unsqueeze(1).expand(-1, n_features), X)
            counts = torch.zeros(K, device=self.device, dtype=X.dtype)
            counts.scatter_add_(0, labels, torch.ones_like(labels, dtype=X.dtype))

            new_centers = new_centers / counts.clamp(min=1.0).unsqueeze(1)

            empty = counts == 0
            if empty.any():
                _, sorted_idx = torch.sort(point_distances, descending=True)
                empty_indices = torch.nonzero(empty).flatten().tolist()
                for j, empty_idx in enumerate(empty_indices):
                    new_centers[empty_idx] = X[sorted_idx[j]]

            center_shift_sq = ((new_centers - centers) ** 2).sum().item()

            if self.verbose:
                cur_inertia = point_distances.sum().item()
                print(f"Iteration {i}, inertia {cur_inertia}.")

            centers = new_centers
            completed_iter = i + 1

            if strict_convergence:
                stop_reason = "strict_convergence"
                if self.verbose:
                    print(f"Converged at iteration {i}: strict convergence.")
                break
            if center_shift_sq <= self.tol:
                stop_reason = "center_shift_tolerance"
                if self.verbose:
                    print(
                        f"Converged at iteration {i}: center shift "
                        f"{center_shift_sq} within tolerance {self.tol}."
                    )
                break

            labels_prev = labels
            del point_distances

        if not strict_convergence:
            # Recompute final labels after the last center update, using the
            # same bounded assignment path as every Lloyd iteration.
            labels, point_distances = self._assign(X, centers, x_squared_norms)
        inertia = point_distances.sum().item()
        return (
            centers,
            labels,
            inertia,
            completed_iter,
            stop_reason,
            center_shift_sq,
        )

    def _assign(
        self,
        X: torch.Tensor,
        centers: torch.Tensor,
        x_squared_norms: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assign rows with a bounded ``batch x clusters`` H100 allocation."""

        batch_size = self.assignment_batch_size or len(X)
        labels: list[torch.Tensor] = []
        distances: list[torch.Tensor] = []
        for start in range(0, len(X), batch_size):
            stop = min(start + batch_size, len(X))
            part = pairwise_squared_distances(
                X[start:stop],
                centers,
                x_squared_norms=x_squared_norms[start:stop],
            )
            part_distances, part_labels = torch.min(part, dim=1)
            labels.append(part_labels)
            distances.append(part_distances)
            del part
        return torch.cat(labels), torch.cat(distances)

    def _kmeans_single_balanced(
        self,
        X: torch.Tensor,
        centers: torch.Tensor,
        x_squared_norms: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, int, str, float]:
        """Run Balanced K-means with an approximately fixed cluster size."""
        n_samples = X.shape[0]
        K = centers.shape[0]
        w = n_samples // K

        labels_prev: Optional[torch.Tensor] = None
        completed_iter = 0
        strict_convergence = False
        stop_reason = "max_iter"
        center_shift_sq = float("inf")

        for i in range(self.max_iter):
            centers_prev = centers.clone()
            unassigned_mask = torch.ones(
                n_samples,
                dtype=torch.bool,
                device=self.device,
            )
            labels = torch.empty(n_samples, dtype=torch.long, device=self.device)

            for k in range(K):
                unassigned_idx = torch.nonzero(unassigned_mask).flatten()
                if len(unassigned_idx) == 0:
                    break

                unassigned = X[unassigned_idx]
                dists = pairwise_squared_distances(
                    unassigned,
                    centers[k : k + 1],
                    x_squared_norms=x_squared_norms[unassigned_idx],
                ).squeeze(1)

                # Last cluster absorbs all remaining points to handle non-divisible n
                assigned_count = w if k < K - 1 else len(unassigned_idx)
                assigned_count = min(assigned_count, len(unassigned_idx))

                sorted_pos = torch.argsort(dists)[:assigned_count]
                assigned_idx = unassigned_idx[sorted_pos]

                labels[assigned_idx] = k
                centers[k] = X[assigned_idx].mean(dim=0)
                unassigned_mask[assigned_idx] = False

            strict_convergence = labels_prev is not None and torch.equal(
                labels, labels_prev
            )
            center_shift_sq = ((centers - centers_prev) ** 2).sum().item()

            if self.verbose:
                cur_inertia = self._assigned_inertia(X, centers, labels)
                print(f"Iteration {i}, inertia {cur_inertia}.")

            completed_iter = i + 1

            if strict_convergence:
                stop_reason = "strict_convergence"
                if self.verbose:
                    print(f"Converged at iteration {i}: strict convergence.")
                break
            if center_shift_sq <= self.tol:
                stop_reason = "center_shift_tolerance"
                if self.verbose:
                    print(
                        f"Converged at iteration {i}: center shift "
                        f"{center_shift_sq} within tolerance {self.tol}."
                    )
                break

            labels_prev = labels

        inertia = self._assigned_inertia(X, centers, labels)
        return (
            centers,
            labels,
            inertia,
            completed_iter,
            stop_reason,
            center_shift_sq,
        )

    def _assigned_inertia(
        self,
        X: torch.Tensor,
        centers: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Compute fixed-label inertia without an ``n_samples x clusters`` tensor."""

        batch_size = self.assignment_batch_size or len(X)
        total = 0.0
        for start in range(0, len(X), batch_size):
            stop = min(start + batch_size, len(X))
            residual = X[start:stop] - centers[labels[start:stop]]
            total += residual.square_().sum().item()
            del residual
        return total

    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> "TorchKMeans":
        # A fitted catalog normally arrives as a read-only mmap.  Make the
        # full FP32 training allocation explicit on both CPU and CUDA.
        """Fit KMeans and retain the lowest-inertia run in fitted attributes.

        Copy all input vectors to the device as FP32. Assignment batching limits distance
        buffers but does not eliminate the rows * dim input allocation.
        """

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        if X_t.ndim != 2:
            raise ValueError(f"Expected 2D input, got shape {tuple(X_t.shape)}")
        if X_t.shape[0] < self.n_clusters:
            raise ValueError(f"n_samples={X_t.shape[0]} < n_clusters={self.n_clusters}")
        x_squared_norms = torch.linalg.vector_norm(X_t, dim=1).square_()

        explicit_init = isinstance(self.init, (np.ndarray, torch.Tensor))
        n_init = 1 if explicit_init else self.n_init

        best_inertia = float("inf")
        best_centers: Optional[torch.Tensor] = None
        best_labels: Optional[torch.Tensor] = None
        best_n_iter = 0
        best_stop_reason: str | None = None
        best_center_shift_sq: float | None = None

        for trial in range(n_init):
            if self.verbose:
                print(
                    f"Starting K-means initialization {trial + 1}/{n_init}.",
                    flush=True,
                )

            if explicit_init:
                centers = torch.as_tensor(
                    self.init, dtype=torch.float32, device=self.device
                ).clone()
                if centers.shape != (self.n_clusters, X_t.shape[1]):
                    raise ValueError(
                        f"init array shape {tuple(centers.shape)} != "
                        f"({self.n_clusters}, {X_t.shape[1]})"
                    )
            elif self.init == "k-means++":
                centers = self._kmeans_plus_plus(
                    X_t,
                    self._random_gen,
                    x_squared_norms,
                )
            elif self.init == "random":
                if self._random_gen is not None:
                    perm = torch.randperm(
                        X_t.shape[0], generator=self._random_gen, device=self.device
                    )
                else:
                    perm = torch.randperm(X_t.shape[0], device=self.device)
                centers = X_t[perm[: self.n_clusters]].clone()
            else:
                raise ValueError(f"Unknown init: {self.init!r}")

            if self.verbose:
                print("Initialization complete", flush=True)

            if self.balanced:
                (
                    centers,
                    labels,
                    inertia,
                    n_iter,
                    stop_reason,
                    center_shift_sq,
                ) = self._kmeans_single_balanced(X_t, centers, x_squared_norms)
            else:
                (
                    centers,
                    labels,
                    inertia,
                    n_iter,
                    stop_reason,
                    center_shift_sq,
                ) = self._kmeans_single(X_t, centers, x_squared_norms)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                best_n_iter = n_iter
                best_stop_reason = stop_reason
                best_center_shift_sq = center_shift_sq

        assert best_centers is not None and best_labels is not None
        self.cluster_centers_ = best_centers.cpu().numpy()
        self.labels_ = best_labels.cpu().numpy()
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        self.stop_reason_ = best_stop_reason
        self.center_shift_sq_ = best_center_shift_sq
        return self

    def fit_predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Fit KMeans and return labels_ as a NumPy array."""

        self.fit(X)
        return self.labels_
