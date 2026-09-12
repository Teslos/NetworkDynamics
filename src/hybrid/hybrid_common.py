"""Shared pieces for the hybrid FHN-reservoir / EP-readout studies.

``compare_readouts.py`` remains the script that reproduces Table 5 of the
manuscript. This module factors out the parts that the follow-up studies
(``relaxation_study.py``, ``readout_ablation.py``) need to share, so that every
readout in every study sees byte-identical cached reservoir features:

* the frozen FHN feature cache for a given seed,
* the relaxation time budget expressed as a physical time rather than a step
  count, so that Euler at different step sizes can be compared fairly,
* a training loop that periodically checkpoints the readout.

Nothing here changes the model; it only makes the existing one reusable.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from compare_readouts import (
    make_raw_digits_loaders,
    precompute_features,
    set_seed,
    standardize_cached_features,
    train_energy_epoch,
    train_linear_epoch,
    evaluate_energy,
    evaluate_linear,
)
from energy_head import EnergyOscillatorHead
from linear_head import LinearReadout
from reservoir import FrozenFHNReservoir

# Relaxation time budgets of the published configuration, expressed as physical
# times: the paper used free_steps=200 and nudged_steps=80 at phase_dt=0.08.
# Holding T fixed while varying the step size is what makes the Euler step-size
# sweep a discretisation comparison rather than a different amount of relaxation.
T_FREE = 16.0
T_NUDGED = 6.4
PHASE_DT = 0.08


def steps_for(time_budget: float, phase_dt: float) -> int:
    """Number of fixed steps covering ``time_budget`` at step ``phase_dt``."""
    return max(1, int(round(time_budget / phase_dt)))


@dataclass
class FeatureCache:
    train_features: Tensor
    train_labels: Tensor
    test_features: Tensor
    test_labels: Tensor
    seed: int
    reservoir_size: int

    @property
    def feature_dim(self) -> int:
        return self.train_features.shape[1]


def build_feature_cache(
    seed: int,
    reservoir_size: int = 64,
    batch_size: int = 64,
    device: torch.device | None = None,
    cache_path: Path | None = None,
) -> FeatureCache:
    """Simulate the frozen FHN reservoir once and cache its features.

    The cache is what every readout is trained on, so differences in accuracy
    are attributable to the readout and not to reservoir variability. If
    ``cache_path`` exists it is loaded instead of recomputed.
    """
    device = device or torch.device("cpu")

    if cache_path is not None and cache_path.exists():
        blob = torch.load(cache_path, weights_only=True)
        if blob["seed"] != seed or blob["reservoir_size"] != reservoir_size:
            raise RuntimeError(
                f"cached features at {cache_path} were built with "
                f"seed={blob['seed']}, reservoir_size={blob['reservoir_size']}, "
                f"but seed={seed}, reservoir_size={reservoir_size} was requested"
            )
        return FeatureCache(
            blob["train_features"], blob["train_labels"],
            blob["test_features"], blob["test_labels"],
            seed, reservoir_size,
        )

    set_seed(seed)
    raw_train_loader, raw_test_loader = make_raw_digits_loaders(batch_size, seed)
    reservoir = FrozenFHNReservoir(
        input_dim=64, n_reservoir=reservoir_size, seed=seed
    ).to(device)

    train_features, train_labels = precompute_features(
        reservoir, raw_train_loader, device
    )
    test_features, test_labels = precompute_features(
        reservoir, raw_test_loader, device
    )
    train_features, test_features = standardize_cached_features(
        train_features, test_features
    )

    cache = FeatureCache(
        train_features, train_labels, test_features, test_labels,
        seed, reservoir_size,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "train_features": train_features,
                "train_labels": train_labels,
                "test_features": test_features,
                "test_labels": test_labels,
                "seed": seed,
                "reservoir_size": reservoir_size,
            },
            cache_path,
        )
    return cache


def make_loaders(cache: FeatureCache, batch_size: int = 64):
    train_loader = DataLoader(
        TensorDataset(cache.train_features, cache.train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cache.seed),
    )
    test_loader = DataLoader(
        TensorDataset(cache.test_features, cache.test_labels),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def make_energy_head(
    feature_dim: int,
    coupling_enabled: bool = True,
    use_ode: bool = True,
    phase_dt: float = PHASE_DT,
    relax_tol: float | None = 1e-3,
    ode_rtol: float = 1e-3,
    ode_atol: float = 1e-5,
) -> EnergyOscillatorHead:
    """Energy readout with the published settings, at a chosen step size.

    ``free_steps`` / ``nudged_steps`` are derived from the fixed time budgets so
    that halving ``phase_dt`` doubles the step count rather than halving the
    relaxation time.
    """
    return EnergyOscillatorHead(
        feature_dim,
        10,
        phase_dt=phase_dt,
        free_steps=steps_for(T_FREE, phase_dt),
        nudged_steps=steps_for(T_NUDGED, phase_dt),
        use_ode=use_ode,
        relax_tol=relax_tol,
        ode_rtol=ode_rtol,
        ode_atol=ode_atol,
        coupling_enabled=coupling_enabled,
    )


def train_energy_readout(
    model: EnergyOscillatorHead,
    cache: FeatureCache,
    epochs: int,
    device: torch.device,
    learning_rate: float = 1e-3,
    coupling_regularization: float = 1e-5,
    batch_size: int = 64,
    checkpoint_epochs: tuple[int, ...] = (),
    checkpoint_dir: Path | None = None,
    checkpoint_tag: str = "energy",
    verbose: bool = True,
):
    """Train an energy readout, recording history and optional checkpoints.

    Checkpoints store the raw ``state_dict`` only. The relaxation study reloads
    them and re-runs inference with the weights held fixed, so it measures the
    solver rather than the training trajectory.
    """
    train_loader, test_loader = make_loaders(cache, batch_size)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )

    history = []
    best_accuracy = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_energy_epoch(
            model, train_loader, optimizer, device, coupling_regularization
        )
        metrics, _, _, residual = evaluate_energy(model, test_loader, device)
        history.append(
            {
                "epoch": epoch,
                "test_accuracy": metrics.accuracy,
                "test_cross_entropy": metrics.cross_entropy,
                "equilibrium_residual": residual,
            }
        )
        if metrics.accuracy > best_accuracy:
            best_accuracy = metrics.accuracy
            best_state = copy.deepcopy(model.state_dict())

        if epoch in checkpoint_epochs and checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "test_accuracy": metrics.accuracy,
                    "equilibrium_residual": residual,
                    "seed": cache.seed,
                    "coupling_enabled": model.coupling_enabled,
                },
                checkpoint_dir / f"{checkpoint_tag}_epoch{epoch:03d}.pt",
            )

        if verbose:
            print(
                f"  epoch {epoch:03d} | acc={metrics.accuracy:.4f} "
                f"| residual={residual:.6f}",
                flush=True,
            )

    return history, best_state, best_accuracy


def train_linear_readout(
    model: LinearReadout,
    cache: FeatureCache,
    epochs: int,
    device: torch.device,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
):
    train_loader, test_loader = make_loaders(cache, batch_size)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )

    history = []
    best_accuracy = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_linear_epoch(model, train_loader, optimizer, device)
        metrics, _, _ = evaluate_linear(model, test_loader, device)
        history.append(
            {
                "epoch": epoch,
                "test_accuracy": metrics.accuracy,
                "test_cross_entropy": metrics.cross_entropy,
            }
        )
        if metrics.accuracy > best_accuracy:
            best_accuracy = metrics.accuracy
            best_state = copy.deepcopy(model.state_dict())

    return history, best_state, best_accuracy
