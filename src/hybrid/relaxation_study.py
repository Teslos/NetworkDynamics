"""How accurately must the readout relax for EP inference to be reliable?

The manuscript reports that fixed-step Euler relaxation destabilises training
while adaptive RK45 does not, and quotes a stopping criterion R < 1e-3 alongside
mean residuals of 0.016-0.070 -- an order of magnitude above the threshold that
was supposedly enforced. This script measures, rather than asserts, what the
solver actually delivers.

Method. Training checkpoints (early / middle / final) are reloaded and the
weights and inputs are held **fixed**; only the relaxation scheme varies. For
each checkpoint and each solver we relax the test set and record the terminal
residual per sample, the fraction of solves meeting the criterion, accuracy,
right-hand-side evaluations and wall time. Free and nudged phases are measured
separately, since the nudged phase starts from the free equilibrium and relaxes
on a different (beta-tilted) landscape.

The relaxation itself is performed by ``EnergyOscillatorHead.relax`` -- the same
method used in training -- with only its solver attributes changed. Nothing is
reimplemented here, so the numbers describe the published code path.

Usage (from src/hybrid):
    python relaxation_study.py --output-dir ../../results/hybrid/relaxation
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from energy_head import EnergyOscillatorHead
from hybrid_common import (
    PHASE_DT,
    T_FREE,
    T_NUDGED,
    build_feature_cache,
    make_energy_head,
    steps_for,
    train_energy_readout,
)

# Residual thresholds at which the fraction of converged solves is reported.
REPORT_THRESHOLDS = (1e-2, 1e-3, 1e-4)

# (label, kwargs for the solver attributes of EnergyOscillatorHead)
#
# The Euler entries hold the relaxation *time* fixed at T_FREE / T_NUDGED and
# vary only the step size, so they isolate discretisation error rather than
# comparing different amounts of relaxation. The RK45 entries tighten the local
# error tolerance and the terminal convergence threshold together, since it is
# pointless to demand a residual the local error control cannot resolve.
# ``time_scale`` multiplies the published relaxation budget. A first pass showed
# that tightening the RK45 tolerance from 1e-2 to 1e-4 leaves the terminal
# residual unchanged, which says the binding constraint is how long the network
# is allowed to relax rather than how accurately each step is taken. The last
# three entries vary that budget so the claim can be tested instead of assumed.
SOLVERS: tuple[tuple[str, dict], ...] = (
    ("euler_dt0.08", dict(use_ode=False, phase_dt=0.08, relax_tol=1e-3)),
    ("euler_dt0.02", dict(use_ode=False, phase_dt=0.02, relax_tol=1e-3)),
    ("euler_dt0.005", dict(use_ode=False, phase_dt=0.005, relax_tol=1e-3)),
    ("rk45_tol1e-2", dict(use_ode=True, relax_tol=1e-2, ode_rtol=1e-2, ode_atol=1e-4)),
    ("rk45_tol1e-3", dict(use_ode=True, relax_tol=1e-3, ode_rtol=1e-3, ode_atol=1e-5)),
    ("rk45_tol1e-4", dict(use_ode=True, relax_tol=1e-4, ode_rtol=1e-4, ode_atol=1e-6)),
    ("rk45_T4x", dict(use_ode=True, relax_tol=1e-4, ode_rtol=1e-4, ode_atol=1e-6,
                      time_scale=4.0)),
    ("rk45_T16x", dict(use_ode=True, relax_tol=1e-4, ode_rtol=1e-4, ode_atol=1e-6,
                       time_scale=16.0)),
    ("rk45_T64x", dict(use_ode=True, relax_tol=1e-4, ode_rtol=1e-4, ode_atol=1e-6,
                       time_scale=64.0)),
)


@dataclass
class PhaseReport:
    """Measurements for one relaxation phase (free or nudged)."""

    phase: str
    residual_mean: float
    residual_median: float
    residual_p95: float
    residual_max: float
    batch_mean_residual: float
    fraction_below: dict
    fraction_batches_meeting_own_tol: float
    rhs_evaluations: int
    seconds: float
    accuracy: float | None = None


@contextmanager
def counted_rhs(model: EnergyOscillatorHead):
    """Count calls to ``phase_gradient`` for the duration of the block.

    Every right-hand-side evaluation and every convergence-event evaluation goes
    through ``phase_gradient``, so this counts the total work the solver asked
    for, which is the cost measure that transfers across solvers.
    """
    original = model.phase_gradient
    counter = {"calls": 0}

    def wrapper(*args, **kwargs):
        counter["calls"] += 1
        return original(*args, **kwargs)

    model.phase_gradient = wrapper  # type: ignore[method-assign]
    try:
        yield counter
    finally:
        model.phase_gradient = original  # type: ignore[method-assign]


def apply_solver(model: EnergyOscillatorHead, config: dict) -> EnergyOscillatorHead:
    """Point the head at a different relaxation scheme, weights untouched.

    Step counts are rederived from the fixed time budgets so that a smaller
    Euler step integrates for the same physical time.
    """
    model.use_ode = config["use_ode"]
    model.relax_tol = config["relax_tol"]
    if config["use_ode"]:
        model.ode_rtol = config["ode_rtol"]
        model.ode_atol = config["ode_atol"]
        model.phase_dt = PHASE_DT
    else:
        model.phase_dt = config["phase_dt"]
    scale = config.get("time_scale", 1.0)
    model.free_steps = steps_for(T_FREE * scale, model.phase_dt)
    model.nudged_steps = steps_for(T_NUDGED * scale, model.phase_dt)
    return model


def summarise(
    phase: str,
    residuals: Tensor,
    batch_means: list[float],
    own_tol: float | None,
    evaluations: int,
    seconds: float,
    accuracy: float | None = None,
) -> PhaseReport:
    values = residuals.detach().cpu().numpy()
    batch_mean = float(np.mean(batch_means)) if batch_means else float("nan")
    met = (
        float(np.mean([m <= own_tol * 1.01 for m in batch_means]))
        if own_tol is not None and batch_means
        else float("nan")
    )
    return PhaseReport(
        phase=phase,
        residual_mean=float(values.mean()),
        residual_median=float(np.median(values)),
        residual_p95=float(np.percentile(values, 95)),
        residual_max=float(values.max()),
        batch_mean_residual=batch_mean,
        fraction_below={
            f"{t:.0e}": float((values < t).mean()) for t in REPORT_THRESHOLDS
        },
        fraction_batches_meeting_own_tol=met,
        rhs_evaluations=evaluations,
        seconds=seconds,
        accuracy=accuracy,
    )


@torch.no_grad()
def measure(
    model: EnergyOscillatorHead,
    features: Tensor,
    labels: Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[PhaseReport, PhaseReport, Tensor]:
    """Relax the given set once, measuring the free and nudged phases.

    The nudged phase starts from the free equilibrium reached by the *same*
    solver, mirroring training, so its cost and residual describe the scheme as
    it is actually used rather than an idealised restart.
    """
    n = features.shape[0]
    free_residuals, nudged_residuals = [], []
    free_batch_means, nudged_batch_means = [], []
    correct = 0

    free_evaluations = nudged_evaluations = 0
    free_seconds = nudged_seconds = 0.0

    for start in range(0, n, batch_size):
        z = features[start : start + batch_size].to(device)
        y = labels[start : start + batch_size].to(device)
        phi0 = torch.zeros(z.shape[0], model.n_classes, device=device, dtype=z.dtype)

        with counted_rhs(model) as counter:
            tic = time.perf_counter()
            free_phi = model.relax(z=z, initial_phi=phi0, steps=model.free_steps)
            free_seconds += time.perf_counter() - tic
        free_evaluations += counter["calls"]

        residual = model.residual_per_sample(z, free_phi)
        free_residuals.append(residual.cpu())
        free_batch_means.append(float(residual.mean()))
        correct += int((model.logits(free_phi).argmax(dim=1) == y).sum())

        with counted_rhs(model) as counter:
            tic = time.perf_counter()
            nudged_phi = model.relax(
                z=z,
                initial_phi=free_phi,
                steps=model.nudged_steps,
                labels=y,
                beta=model.beta,
            )
            nudged_seconds += time.perf_counter() - tic
        nudged_evaluations += counter["calls"]

        nudged_residual = model.residual_per_sample(
            z, nudged_phi, labels=y, beta=model.beta
        )
        nudged_residuals.append(nudged_residual.cpu())
        nudged_batch_means.append(float(nudged_residual.mean()))

    free_residual_values = torch.cat(free_residuals)
    free = summarise(
        "free",
        free_residual_values,
        free_batch_means,
        model.relax_tol,
        free_evaluations,
        free_seconds,
        accuracy=correct / n,
    )
    nudged = summarise(
        "nudged",
        torch.cat(nudged_residuals),
        nudged_batch_means,
        model.relax_tol,
        nudged_evaluations,
        nudged_seconds,
    )
    return free, nudged, free_residual_values


@torch.no_grad()
def field_diagnostic(
    model: EnergyOscillatorHead,
    features: Tensor,
    residuals: Tensor,
    time_budget: float,
) -> dict:
    """Relate the terminal residual to the strength of the driving field.

    Linearising the uncoupled energy about its minimum gives
    d(delta phi_c)/dt = -r_c delta phi_c, so r_c is the relaxation *rate* of
    class c and 1/r_c its time constant. The slowest class of a sample,
    min_c r_c, therefore predicts which samples can reach equilibrium inside a
    budget T: those with r_c T >> 1 converge, those with r_c T <~ 1 cannot,
    however accurately the trajectory is integrated. With coupling present this
    is an approximation, but K is small compared with the fields here.
    """
    a = model.field_cos(features)
    b = model.field_sin(features)
    r = torch.sqrt(a * a + b * b)
    slowest = r.min(dim=1).values.cpu().numpy()
    residual_values = residuals.cpu().numpy()

    order_r = slowest.argsort().argsort().astype(float)
    order_residual = residual_values.argsort().argsort().astype(float)
    spearman = float(np.corrcoef(order_r, order_residual)[0, 1])

    converged = residual_values < 1e-3
    return {
        "slowest_rate_mean": float(slowest.mean()),
        "slowest_rate_median": float(np.median(slowest)),
        "slowest_rate_p05": float(np.percentile(slowest, 5)),
        "spearman_slowest_rate_vs_residual": spearman,
        "slowest_rate_of_converged_median": (
            float(np.median(slowest[converged])) if converged.any() else float("nan")
        ),
        "slowest_rate_of_unconverged_median": (
            float(np.median(slowest[~converged])) if (~converged).any() else float("nan")
        ),
        "fraction_with_rate_times_T_below_10": float((slowest * time_budget < 10).mean()),
        "time_budget": time_budget,
    }


def load_checkpoint(
    path: Path, feature_dim: int, device: torch.device
) -> tuple[EnergyOscillatorHead, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = make_energy_head(
        feature_dim, coupling_enabled=blob.get("coupling_enabled", True)
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    return model, blob


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--reservoir-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device", default="cpu", help="device for the relaxation sweep"
    )
    parser.add_argument(
        "--feature-device",
        default=None,
        help="device for the FHN feature cache; defaults to cuda when "
        "available, which reproduces the stored cache bit-for-bit",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("../../results/hybrid/relaxation")
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        nargs="+",
        default=None,
        help="epochs to checkpoint; default is early/middle/final",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"

    checkpoint_epochs = tuple(
        args.checkpoint_epochs or (1, args.epochs // 2, args.epochs)
    )

    feature_device = torch.device(
        args.feature_device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cache = build_feature_cache(
        seed=args.seed,
        reservoir_size=args.reservoir_size,
        device=feature_device,
        cache_path=args.output_dir / f"features_seed{args.seed}.pt",
    )

    # --- obtain checkpoints -------------------------------------------------
    missing = [
        e
        for e in checkpoint_epochs
        if not (checkpoint_dir / f"energy_epoch{e:03d}.pt").exists()
    ]
    if missing:
        print(
            f"training reference energy readout ({args.epochs} epochs, "
            f"checkpoints at {checkpoint_epochs})",
            flush=True,
        )
        torch.manual_seed(args.seed)
        model = make_energy_head(cache.feature_dim, coupling_enabled=True).to(device)
        history, _, best = train_energy_readout(
            model,
            cache,
            epochs=args.epochs,
            device=device,
            checkpoint_epochs=checkpoint_epochs,
            checkpoint_dir=checkpoint_dir,
            checkpoint_tag="energy",
        )
        (args.output_dir / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(f"best test accuracy during training: {best:.4f}", flush=True)

    # --- solver sweep at fixed weights --------------------------------------
    records = []
    for epoch in checkpoint_epochs:
        path = checkpoint_dir / f"energy_epoch{epoch:03d}.pt"
        model, blob = load_checkpoint(path, cache.feature_dim, device)
        stage = (
            "early"
            if epoch == checkpoint_epochs[0]
            else "final"
            if epoch == checkpoint_epochs[-1]
            else "middle"
        )
        print(
            f"\ncheckpoint {stage} (epoch {epoch}, "
            f"training-time accuracy {blob['test_accuracy']:.4f})",
            flush=True,
        )
        for label, config in SOLVERS:
            apply_solver(model, config)
            free, nudged, _ = measure(
                model, cache.test_features, cache.test_labels, args.batch_size, device
            )
            for report in (free, nudged):
                records.append(
                    {
                        "checkpoint_epoch": epoch,
                        "stage": stage,
                        "solver": label,
                        "time_scale": config.get("time_scale", 1.0),
                        "free_time_budget": T_FREE * config.get("time_scale", 1.0),
                        "batch_size": args.batch_size,
                        **asdict(report),
                    }
                )
            print(
                f"  {label:14s} acc={free.accuracy:.4f} "
                f"free R(mean/p95)={free.residual_mean:.2e}/{free.residual_p95:.2e} "
                f"nudged R(mean)={nudged.residual_mean:.2e} "
                f"evals={free.rhs_evaluations + nudged.rhs_evaluations:6d} "
                f"t={free.seconds + nudged.seconds:6.1f}s",
                flush=True,
            )

    # --- batch-size probe ---------------------------------------------------
    # The stopping criterion is imposed on the batch-mean residual, so a batch
    # can stop while individual samples are far from equilibrium. Re-running the
    # final checkpoint one sample at a time turns the same criterion into a
    # per-sample one and isolates that effect.
    final_epoch = checkpoint_epochs[-1]
    model, _ = load_checkpoint(
        checkpoint_dir / f"energy_epoch{final_epoch:03d}.pt", cache.feature_dim, device
    )
    apply_solver(model, dict(SOLVERS)["rk45_tol1e-3"])
    print("\nbatch-size probe on the final checkpoint (rk45_tol1e-3)", flush=True)
    probe = []
    mechanism_residuals = None
    for batch_size in (args.batch_size, 1):
        free, nudged, free_residuals = measure(
            model, cache.test_features, cache.test_labels, batch_size, device
        )
        # Keep the standard-batch residuals for the mechanism diagnostic below.
        # Reading whatever the loop ended on would silently describe the
        # batch-size-1 run, whose stopping behaviour differs.
        if batch_size == args.batch_size:
            mechanism_residuals = free_residuals
        probe.append(
            {"batch_size": batch_size, **asdict(free)},
        )
        print(
            f"  batch={batch_size:3d} acc={free.accuracy:.4f} "
            f"free R(mean)={free.residual_mean:.2e} "
            f"batch-mean R={free.batch_mean_residual:.2e} "
            f"frac<1e-3={free.fraction_below['1e-03']:.3f} "
            f"t={free.seconds:.1f}s",
            flush=True,
        )

    mechanism = field_diagnostic(
        model,
        cache.test_features.to(device),
        mechanism_residuals,
        time_budget=T_FREE,
    )
    print()
    print(
        "why samples stall: Spearman(slowest field rate, residual) = "
        f"{mechanism['spearman_slowest_rate_vs_residual']:+.3f}; "
        f"median slowest rate converged="
        f"{mechanism['slowest_rate_of_converged_median']:.3f} vs unconverged="
        f"{mechanism['slowest_rate_of_unconverged_median']:.3f}",
        flush=True,
    )

    payload = {
        "seed": args.seed,
        "mechanism": mechanism,
        "epochs": args.epochs,
        "checkpoint_epochs": list(checkpoint_epochs),
        "feature_dim": cache.feature_dim,
        "test_samples": int(cache.test_features.shape[0]),
        "time_budgets": {"free": T_FREE, "nudged": T_NUDGED},
        "solver_sweep": records,
        "batch_size_probe": probe,
    }
    (args.output_dir / "relaxation_study.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {args.output_dir / 'relaxation_study.json'}")


if __name__ == "__main__":
    main()
