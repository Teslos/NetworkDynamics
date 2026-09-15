"""Does the relaxation scheme change learning, or only inference?

``relaxation_study.py`` holds the weights fixed and varies the solver, which
diagnoses inference and gradient reliability. It cannot show that the solver
changes what the network *learns*, because the checkpoints it reloads were all
produced by one training run.

This script supplies the missing control. The same readout is trained from the
same initialisation on the same cached features under several relaxation
schemes, and the accuracy and residual trajectories are recorded per epoch:

* ``euler_dt0.08`` -- the published fixed-step scheme, which the manuscript
  reports collapsing after epoch 7;
* ``euler_dt0.02`` / ``euler_dt0.005`` -- smaller steps at the same relaxation
  time, separating "Euler is unstable" from "the step was too large";
* ``rk45_tol1e-3`` -- the published adaptive fix;
* ``rk45_T4x`` -- adaptive, with the relaxation time budget that the fixed-weight
  sweep found sufficient to actually meet the stated stopping criterion.

Usage (from src/hybrid):
    python training_controls.py --output-dir ../../results/hybrid/relaxation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hybrid_common import (
    T_FREE,
    T_NUDGED,
    build_feature_cache,
    make_energy_head,
    steps_for,
    train_energy_readout,
)

# (label, solver kwargs, time_scale)
CONTROLS: tuple[tuple[str, dict, float], ...] = (
    ("euler_dt0.08", dict(use_ode=False, phase_dt=0.08, relax_tol=1e-3), 1.0),
    ("euler_dt0.02", dict(use_ode=False, phase_dt=0.02, relax_tol=1e-3), 1.0),
    ("euler_dt0.005", dict(use_ode=False, phase_dt=0.005, relax_tol=1e-3), 1.0),
    ("rk45_tol1e-3", dict(use_ode=True, relax_tol=1e-3, ode_rtol=1e-3, ode_atol=1e-5), 1.0),
    ("rk45_T4x", dict(use_ode=True, relax_tol=1e-4, ode_rtol=1e-4, ode_atol=1e-6), 4.0),
    # The EnergyOscillatorHead class defaults are free_steps=80, nudged_steps=30
    # at phase_dt=0.08, i.e. a relaxation budget of T=6.4 -- 0.4x the budget the
    # RK45 driver sets explicitly. The manuscript's collapsing Euler run predates
    # that driver, so this control reproduces the *original* configuration and
    # separates "Euler is unstable" from "the original run relaxed for 6.4 time
    # units instead of 16".
    ("euler_original_T6.4", dict(use_ode=False, phase_dt=0.08, relax_tol=1e-3), 0.4),
)


def build_head(feature_dim: int, config: dict, time_scale: float, seed: int):
    """Construct the readout with a given solver, from a fixed initialisation.

    Seeding immediately before construction means every control starts from
    byte-identical weights, so trajectories diverge only because the relaxation
    differs.
    """
    torch.manual_seed(seed)
    model = make_energy_head(
        feature_dim,
        coupling_enabled=True,
        use_ode=config["use_ode"],
        phase_dt=config.get("phase_dt", 0.08),
        relax_tol=config["relax_tol"],
        ode_rtol=config.get("ode_rtol", 1e-3),
        ode_atol=config.get("ode_atol", 1e-5),
    )
    model.free_steps = steps_for(T_FREE * time_scale, model.phase_dt)
    model.nudged_steps = steps_for(T_NUDGED * time_scale, model.phase_dt)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--reservoir-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only", default=None, help="run a single control by label")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("../../results/hybrid/relaxation")
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = build_feature_cache(
        seed=args.seed,
        reservoir_size=args.reservoir_size,
        device=feature_device,
        cache_path=args.output_dir / f"features_seed{args.seed}.pt",
    )

    controls = [c for c in CONTROLS if args.only is None or c[0] == args.only]
    if not controls:
        raise SystemExit(f"no control named {args.only!r}")

    results = {}
    for label, config, time_scale in controls:
        print(f"\n=== {label} (time budget x{time_scale:g}) ===", flush=True)
        model = build_head(cache.feature_dim, config, time_scale, args.seed).to(device)
        history, _, best = train_energy_readout(
            model,
            cache,
            epochs=args.epochs,
            device=device,
            verbose=True,
        )
        accuracies = [h["test_accuracy"] for h in history]
        residuals = [h["equilibrium_residual"] for h in history]
        peak_epoch = int(max(range(len(accuracies)), key=accuracies.__getitem__)) + 1
        results[label] = {
            "solver": config,
            "time_scale": time_scale,
            "history": history,
            "best_accuracy": best,
            "peak_epoch": peak_epoch,
            "final_accuracy": accuracies[-1],
            "accuracy_drop_from_peak": best - accuracies[-1],
            "residual_first": residuals[0],
            "residual_max": max(residuals),
            "residual_final": residuals[-1],
        }
        print(
            f"{label}: best={best:.4f} at epoch {peak_epoch}, "
            f"final={accuracies[-1]:.4f}, drop={best - accuracies[-1]:+.4f}, "
            f"residual {residuals[0]:.4f} -> {residuals[-1]:.4f} "
            f"(max {max(residuals):.4f})",
            flush=True,
        )

        suffix = f"_{args.only}" if args.only else ""
        (args.output_dir / f"training_controls{suffix}.json").write_text(
            json.dumps(
                {"seed": args.seed, "epochs": args.epochs, "controls": results},
                indent=2,
            ),
            encoding="utf-8",
        )

    print("\n--- summary ---")
    print(f"{'control':16s} {'best':>7s} {'peak@':>6s} {'final':>7s} {'drop':>7s} {'Rmax':>8s}")
    for label, r in results.items():
        print(
            f"{label:16s} {r['best_accuracy']:7.4f} {r['peak_epoch']:6d} "
            f"{r['final_accuracy']:7.4f} {r['accuracy_drop_from_peak']:+7.4f} "
            f"{r['residual_max']:8.4f}"
        )


if __name__ == "__main__":
    main()
