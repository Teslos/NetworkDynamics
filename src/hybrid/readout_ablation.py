"""Do the interactions between output oscillators actually help classification?

Table 5 of the manuscript compares a linear soft-max readout with a *coupled*
phase readout and finds them within one test sample of each other. That
comparison cannot say whether the coupling K contributes anything, because it
changes two things at once: the phase encoding/decoding and the interaction
between class oscillators. This script inserts the missing middle term.

Three readouts are trained on byte-identical cached FHN features:

* ``linear``            -- soft-max on the reservoir features (baseline);
* ``phase_uncoupled``   -- the same energy readout with K == 0, so the ten class
                           oscillators are independent (isolates the nonlinear
                           phase encoding and decoding);
* ``phase_coupled``     -- the published readout (adds the interactions).

With K == 0 the energy separates per class and its minimum is known in closed
form, phi_c* = atan2(b_c, a_c), so the uncoupled model doubles as a correctness
check on the numerical relaxation: the solver's answer is compared against the
exact one. Samples with a vanishing field r_c = 0 have an undetermined
equilibrium and are counted separately rather than scored.

Two methodological differences from ``compare_readouts.py``, both deliberate:

* epochs are selected on a held-out **validation** split carved from the
  training partition, not on the test set the result is then reported on;
* every configuration is run over paired seeds sharing the split and the
  feature cache, so the readouts can be compared per seed.

Usage (from src/hybrid):
    python readout_ablation.py --seeds 42 43 44 45 46 \
        --output-dir ../../results/hybrid/ablation
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from compare_readouts import train_energy_epoch, train_linear_epoch
from energy_head import EnergyOscillatorHead
from hybrid_common import build_feature_cache, make_energy_head
from linear_head import LinearReadout

VALIDATION_FRACTION = 0.2


def stratified_split(labels: Tensor, fraction: float, seed: int):
    """Indices for a stratified hold-out of ``fraction`` of the samples."""
    generator = np.random.default_rng(seed)
    held, kept = [], []
    label_values = labels.numpy()
    for value in np.unique(label_values):
        index = np.flatnonzero(label_values == value)
        generator.shuffle(index)
        cut = max(1, int(round(fraction * len(index))))
        held.extend(index[:cut].tolist())
        kept.extend(index[cut:].tolist())
    return sorted(kept), sorted(held)


@torch.no_grad()
def evaluate(model, features: Tensor, labels: Tensor, device, batch_size=64):
    """Accuracy, cross-entropy and wall time for any of the three readouts."""
    is_energy = isinstance(model, EnergyOscillatorHead)
    total_loss = correct = 0
    predictions = []
    tic = time.perf_counter()
    for start in range(0, features.shape[0], batch_size):
        z = features[start : start + batch_size].to(device)
        y = labels[start : start + batch_size].to(device)
        if is_energy:
            phi0 = torch.zeros(z.shape[0], model.n_classes, device=device, dtype=z.dtype)
            phi = model.relax(z=z, initial_phi=phi0, steps=model.free_steps)
            logits = model.logits(phi)
        else:
            logits = model(z)
        total_loss += float(F.cross_entropy(logits, y, reduction="sum"))
        predicted = logits.argmax(dim=1)
        correct += int((predicted == y).sum())
        predictions.append(predicted.cpu())
    seconds = time.perf_counter() - tic
    n = features.shape[0]
    return {
        "accuracy": correct / n,
        "cross_entropy": total_loss / n,
        "errors": n - correct,
        "samples": n,
        "seconds_per_sample": seconds / n,
    }, torch.cat(predictions)


@torch.no_grad()
def analytic_check(model: EnergyOscillatorHead, features: Tensor, device, batch_size=64):
    """Compare the relaxed phases with the closed-form minimum (K == 0 only).

    Reports the circular distance between numerical and exact equilibria, and
    whether the two give the same prediction. Entries whose field magnitude is
    below ``field_floor`` have no determined minimum and are excluded from the
    error statistics but reported as a count.
    """
    errors, rates, undetermined, total = [], [], 0, 0
    agree = 0
    n = features.shape[0]
    for start in range(0, n, batch_size):
        z = features[start : start + batch_size].to(device)
        phi0 = torch.zeros(z.shape[0], model.n_classes, device=device, dtype=z.dtype)
        numeric = model.relax(z=z, initial_phi=phi0, steps=model.free_steps)
        exact, r, determined = model.analytic_free_equilibrium(z)

        difference = torch.atan2(
            torch.sin(numeric - exact), torch.cos(numeric - exact)
        ).abs()
        errors.append(difference[determined].cpu())
        rates.append(r[determined].cpu())
        undetermined += int((~determined).sum())
        total += determined.numel()

        agree += int(
            (
                model.logits(numeric).argmax(dim=1)
                == model.logits(exact).argmax(dim=1)
            ).sum()
        )

    values = torch.cat(errors).numpy()
    rate_values = torch.cat(rates).numpy()
    # r_c is the linear relaxation rate of class c about its minimum, so r_c * T
    # is the number of e-foldings available. Entries that disagree with the
    # closed form should be the ones that ran out of relaxation time, not ones
    # the integrator got wrong.
    failed = values > 0.1
    return {
        "phase_error_median_rad": float(np.median(values)),
        "phase_error_p95_rad": float(np.percentile(values, 95)),
        "phase_error_max_rad": float(values.max()),
        "fraction_error_above_0p1rad": float(failed.mean()),
        "median_rate_all": float(np.median(rate_values)),
        "median_rate_of_failures": (
            float(np.median(rate_values[failed])) if failed.any() else float("nan")
        ),
        "median_rate_of_matches": (
            float(np.median(rate_values[~failed])) if (~failed).any() else float("nan")
        ),
        "prediction_agreement": agree / n,
        "undetermined_entries": undetermined,
        "total_entries": total,
    }


def effective_parameters(model) -> tuple[int, int]:
    """(stored, effective) trainable parameter counts.

    The coupling is stored as a full C x C matrix but enters the energy only
    through its symmetric, zero-diagonal part, so C(C-1)/2 of its C^2 entries
    are effective and the rest are null directions of the energy.
    """
    stored = sum(p.numel() for p in model.parameters() if p.requires_grad)
    effective = stored
    if isinstance(model, EnergyOscillatorHead) and model.coupling_enabled:
        c = model.n_classes
        effective = stored - c * c + c * (c - 1) // 2
    return stored, effective


def train_with_validation(
    model,
    train_features,
    train_labels,
    validation_features,
    validation_labels,
    epochs,
    device,
    learning_rate=1e-3,
    batch_size=64,
    coupling_regularization=1e-5,
    seed=0,
    verbose=True,
):
    """Train, selecting the reported epoch on validation accuracy."""
    is_energy = isinstance(model, EnergyOscillatorHead)
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )

    best_accuracy, best_state, best_epoch = -1.0, copy.deepcopy(model.state_dict()), 0
    history = []
    for epoch in range(1, epochs + 1):
        if is_energy:
            train_energy_epoch(
                model, loader, optimizer, device, coupling_regularization
            )
        else:
            train_linear_epoch(model, loader, optimizer, device)
        metrics, _ = evaluate(model, validation_features, validation_labels, device)
        history.append({"epoch": epoch, "validation_accuracy": metrics["accuracy"]})
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"    epoch {epoch:03d} val={metrics['accuracy']:.4f}", flush=True)

    model.load_state_dict(best_state)
    return history, best_epoch, best_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--reservoir-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--readouts",
        nargs="+",
        default=["linear", "phase_uncoupled", "phase_coupled"],
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("../../results/hybrid/ablation")
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    feature_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_seed = {}
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        cache = build_feature_cache(
            seed=seed,
            reservoir_size=args.reservoir_size,
            device=feature_device,
            cache_path=args.output_dir / f"features_seed{seed}.pt",
        )
        train_index, validation_index = stratified_split(
            cache.train_labels, VALIDATION_FRACTION, seed
        )
        x_train = cache.train_features[train_index]
        y_train = cache.train_labels[train_index]
        x_validation = cache.train_features[validation_index]
        y_validation = cache.train_labels[validation_index]
        print(
            f"  train={len(train_index)} val={len(validation_index)} "
            f"test={cache.test_features.shape[0]}",
            flush=True,
        )

        seed_results = {}
        predictions = {}
        for name in args.readouts:
            torch.manual_seed(seed)
            if name == "linear":
                model = LinearReadout(cache.feature_dim, 10).to(device)
            elif name == "phase_uncoupled":
                model = make_energy_head(
                    cache.feature_dim, coupling_enabled=False
                ).to(device)
            elif name == "phase_coupled":
                model = make_energy_head(
                    cache.feature_dim, coupling_enabled=True
                ).to(device)
            else:
                raise SystemExit(f"unknown readout {name!r}")

            print(f"  -- {name}", flush=True)
            _, best_epoch, best_validation = train_with_validation(
                model,
                x_train,
                y_train,
                x_validation,
                y_validation,
                epochs=args.epochs,
                device=device,
                seed=seed,
            )
            test_metrics, test_predictions = evaluate(
                model, cache.test_features, cache.test_labels, device
            )
            stored, effective = effective_parameters(model)
            entry = {
                **test_metrics,
                "selected_epoch": best_epoch,
                "validation_accuracy": best_validation,
                "stored_parameters": stored,
                "effective_parameters": effective,
            }
            if name == "phase_uncoupled":
                entry["analytic_check"] = analytic_check(
                    model, cache.test_features, device
                )
            seed_results[name] = entry
            predictions[name] = test_predictions
            print(
                f"     test={test_metrics['accuracy']:.4f} "
                f"({test_metrics['errors']} errors), epoch {best_epoch}, "
                f"params {stored}/{effective}, "
                f"{test_metrics['seconds_per_sample'] * 1e3:.2f} ms/sample",
                flush=True,
            )
            if name == "phase_uncoupled":
                check = entry["analytic_check"]
                print(
                    f"     analytic check: median |dphi|="
                    f"{check['phase_error_median_rad']:.2e} rad, "
                    f"max={check['phase_error_max_rad']:.2e}, "
                    f"agreement={check['prediction_agreement']:.4f}, "
                    f"undetermined={check['undetermined_entries']}"
                    f"/{check['total_entries']}; "
                    f"err>0.1rad on "
                    f"{check['fraction_error_above_0p1rad'] * 100:.2f}% "
                    f"(median rate {check['median_rate_of_failures']:.3f} vs "
                    f"{check['median_rate_of_matches']:.3f} for matches)",
                    flush=True,
                )

        # Paired disagreement structure between readouts on this seed.
        targets = cache.test_labels
        pairs = {}
        names = list(predictions)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                a = predictions[first].eq(targets)
                b = predictions[second].eq(targets)
                pairs[f"{first}_vs_{second}"] = {
                    "both_correct": int((a & b).sum()),
                    "both_wrong": int((~a & ~b).sum()),
                    f"{first}_only": int((a & ~b).sum()),
                    f"{second}_only": int((~a & b).sum()),
                    "disagreements": int(
                        predictions[first].ne(predictions[second]).sum()
                    ),
                }
        seed_results["_pairs"] = pairs
        per_seed[seed] = seed_results

        (args.output_dir / "readout_ablation.json").write_text(
            json.dumps({"seeds": args.seeds, "per_seed": per_seed}, indent=2),
            encoding="utf-8",
        )

    # --- aggregate ---------------------------------------------------------
    print("\n--- mean +/- std over seeds ---")
    print(f"{'readout':17s} {'accuracy':>16s} {'errors':>8s} {'params':>13s} {'ms/sample':>10s}")
    summary = {}
    for name in args.readouts:
        accuracies = [per_seed[s][name]["accuracy"] for s in args.seeds]
        errors = [per_seed[s][name]["errors"] for s in args.seeds]
        milliseconds = [
            per_seed[s][name]["seconds_per_sample"] * 1e3 for s in args.seeds
        ]
        stored = per_seed[args.seeds[0]][name]["stored_parameters"]
        effective = per_seed[args.seeds[0]][name]["effective_parameters"]
        summary[name] = {
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies, ddof=1)),
            "accuracies": accuracies,
            "errors_mean": float(np.mean(errors)),
            "stored_parameters": stored,
            "effective_parameters": effective,
            "milliseconds_per_sample_mean": float(np.mean(milliseconds)),
        }
        print(
            f"{name:17s} {np.mean(accuracies):.4f} +/- {np.std(accuracies, ddof=1):.4f} "
            f"{np.mean(errors):8.1f} {stored:6d}/{effective:<6d} "
            f"{np.mean(milliseconds):10.2f}"
        )

    # Paired differences: the seeds share splits and features, so differences
    # are paired and the per-seed spread is the relevant scale.
    print("\n--- paired differences (per seed) ---")
    paired = {}
    for i, first in enumerate(args.readouts):
        for second in args.readouts[i + 1 :]:
            differences = [
                per_seed[s][second]["accuracy"] - per_seed[s][first]["accuracy"]
                for s in args.seeds
            ]
            mean = float(np.mean(differences))
            std = float(np.std(differences, ddof=1))
            paired[f"{second}_minus_{first}"] = {
                "mean": mean,
                "std": std,
                "per_seed": differences,
                "wins": int(sum(d > 0 for d in differences)),
                "losses": int(sum(d < 0 for d in differences)),
            }
            print(
                f"{second} - {first}: {mean * 100:+.2f} +/- {std * 100:.2f} pp "
                f"(wins {sum(d > 0 for d in differences)}/{len(differences)})"
            )

    (args.output_dir / "readout_ablation.json").write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "epochs": args.epochs,
                "validation_fraction": VALIDATION_FRACTION,
                "per_seed": per_seed,
                "summary": summary,
                "paired_differences": paired,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.output_dir / 'readout_ablation.json'}")


if __name__ == "__main__":
    main()
