"""Track the weak-field relaxation mechanism across training checkpoints.

``relaxation_study.py`` evaluates the field-rate diagnostic once, on the final
checkpoint. The strength of the relationship changes over training: early on the
coupling K is small and the uncoupled linearisation that makes r_c the relaxation
rate of class c is accurate, so the correlation between the slowest rate and the
terminal residual is strong; it weakens as K grows. This script evaluates the
same diagnostic at every checkpoint so that trend is measured rather than
asserted.

All checkpoints are swept with one solver (`rk45_tol1e-3`, the published
setting) at the standard batch size, so the only thing varying between rows is
the trained weights.

Usage (from src/hybrid, after relaxation_study.py has produced checkpoints):
    python mechanism_by_checkpoint.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from hybrid_common import T_FREE, build_feature_cache
from relaxation_study import (
    SOLVERS,
    apply_solver,
    field_diagnostic,
    load_checkpoint,
    measure,
)

CHECKPOINTS = ((1, "early"), (25, "middle"), (50, "final"))
BATCH_SIZE = 64


def main() -> None:
    out = Path("../../results/hybrid/relaxation")
    cache = build_feature_cache(
        seed=42,
        reservoir_size=64,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        cache_path=out / "features_seed42.pt",
    )
    device = torch.device("cpu")
    config = dict(SOLVERS)["rk45_tol1e-3"]

    rows = {}
    for epoch, stage in CHECKPOINTS:
        model, _ = load_checkpoint(
            out / "checkpoints" / f"energy_epoch{epoch:03d}.pt",
            cache.feature_dim,
            device,
        )
        apply_solver(model, config)
        _, _, residuals = measure(
            model, cache.test_features, cache.test_labels, BATCH_SIZE, device
        )
        entry = field_diagnostic(
            model, cache.test_features.to(device), residuals, T_FREE
        )
        rows[stage] = entry
        print(
            f"{stage:7s} epoch {epoch:2d}: "
            f"spearman={entry['spearman_slowest_rate_vs_residual']:+.3f} "
            f"converged={entry['slowest_rate_of_converged_median']:.3f} "
            f"unconverged={entry['slowest_rate_of_unconverged_median']:.3f} "
            f"frac rT<10 = {entry['fraction_with_rate_times_T_below_10']:.3f}",
            flush=True,
        )

    path = out / "mechanism_by_checkpoint.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
