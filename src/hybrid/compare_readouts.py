from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from energy_head import EnergyOscillatorHead
from linear_head import LinearReadout
from reservoir import FrozenFHNReservoir


@dataclass
class Metrics:
    cross_entropy: float
    accuracy: float
    errors: int
    samples: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def make_raw_digits_loaders(batch_size: int, seed: int):
    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=0.2,
        random_state=seed,
        stratify=digits.target,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


@torch.no_grad()
def precompute_features(reservoir, loader, device):
    reservoir.eval()
    all_features, all_labels = [], []
    for inputs, labels in loader:
        features = reservoir(inputs.to(device=device, dtype=torch.float32))
        all_features.append(features.cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_features), torch.cat(all_labels)


def standardize_cached_features(train_features: Tensor, test_features: Tensor):
    """Use training-cache statistics only; both readouts see the same tensors."""
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return (train_features - mean) / std, (test_features - mean) / std


def train_linear_epoch(model, loader, optimizer, device) -> Metrics:
    model.train()
    total_loss = total_correct = total_samples = 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        n = labels.shape[0]
        total_loss += loss.item() * n
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += n
    return Metrics(total_loss / total_samples, total_correct / total_samples,
                   total_samples - total_correct, total_samples)


@torch.no_grad()
def evaluate_linear(model, loader, device):
    model.eval()
    total_loss = total_correct = total_samples = 0
    predictions, targets = [], []
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        predicted = logits.argmax(1)
        loss = F.cross_entropy(logits, labels)
        n = labels.shape[0]
        total_loss += loss.item() * n
        total_correct += (predicted == labels).sum().item()
        total_samples += n
        predictions.append(predicted.cpu())
        targets.append(labels.cpu())
    metrics = Metrics(total_loss / total_samples, total_correct / total_samples,
                      total_samples - total_correct, total_samples)
    return metrics, torch.cat(predictions), torch.cat(targets)


def train_energy_epoch(model, loader, optimizer, device, coupling_regularization):
    model.train()
    total_loss = total_correct = total_samples = 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        free_phi, nudged_phi = model.equilibria(features, labels)
        ep_loss = model.ep_surrogate_loss(features, free_phi, nudged_phi)
        penalty = coupling_regularization * model.symmetric_coupling().pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        (ep_loss + penalty).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            logits = model.logits(free_phi)
            report_loss = F.cross_entropy(logits, labels)
            predicted = logits.argmax(1)
        n = labels.shape[0]
        total_loss += report_loss.item() * n
        total_correct += (predicted == labels).sum().item()
        total_samples += n
    return Metrics(total_loss / total_samples, total_correct / total_samples,
                   total_samples - total_correct, total_samples)


@torch.no_grad()
def evaluate_energy(model, loader, device):
    model.eval()
    total_loss = total_correct = total_samples = 0
    total_residual = 0.0
    predictions, targets = [], []
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        initial_phi = torch.zeros(features.shape[0], model.n_classes,
                                  device=device, dtype=features.dtype)
        free_phi = model.relax(features, initial_phi, model.free_steps)
        logits = model.logits(free_phi)
        predicted = logits.argmax(1)
        loss = F.cross_entropy(logits, labels)
        residual = model.equilibrium_residual(features, free_phi)
        n = labels.shape[0]
        total_loss += loss.item() * n
        total_correct += (predicted == labels).sum().item()
        total_residual += residual.item() * n
        total_samples += n
        predictions.append(predicted.cpu())
        targets.append(labels.cpu())
    metrics = Metrics(total_loss / total_samples, total_correct / total_samples,
                      total_samples - total_correct, total_samples)
    return metrics, torch.cat(predictions), torch.cat(targets), total_residual / total_samples


def paired_comparison(linear_predictions, energy_predictions, targets):
    lc = linear_predictions.eq(targets)
    ec = energy_predictions.eq(targets)
    return {
        "both_correct": int((lc & ec).sum()),
        "both_wrong": int((~lc & ~ec).sum()),
        "linear_only_correct": int((lc & ~ec).sum()),
        "energy_only_correct": int((~lc & ec).sum()),
        "prediction_disagreements": int(linear_predictions.ne(energy_predictions).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare linear and energy readouts on identical FHN features."
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reservoir-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--linear-lr", type=float, default=1e-3)
    parser.add_argument("--energy-lr", type=float, default=1e-3)
    parser.add_argument("--coupling-regularization", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=Path, default=Path("comparison_results"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_train_loader, raw_test_loader = make_raw_digits_loaders(args.batch_size, args.seed)
    reservoir = FrozenFHNReservoir(
        input_dim=64, n_reservoir=args.reservoir_size, seed=args.seed
    ).to(device)

    print("Precomputing one shared FHN feature cache...")
    train_features, train_labels = precompute_features(reservoir, raw_train_loader, device)
    test_features, test_labels = precompute_features(reservoir, raw_test_loader, device)
    train_features, test_features = standardize_cached_features(train_features, test_features)

    torch.save({
        "train_features": train_features,
        "train_labels": train_labels,
        "test_features": test_features,
        "test_labels": test_labels,
        "seed": args.seed,
        "reservoir_size": args.reservoir_size,
    }, args.output_dir / "shared_fhn_features.pt")

    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    test_loader = DataLoader(
        TensorDataset(test_features, test_labels),
        batch_size=args.batch_size,
        shuffle=False,
    )

    feature_dim = train_features.shape[1]
    linear_model = LinearReadout(feature_dim, 10).to(device)
    energy_model = EnergyOscillatorHead(
        feature_dim, 10,
        use_ode=True,       # fix #3: adaptive RK45 replaces fixed-step Euler
        relax_tol=1e-3,     # stop early when ||∇E|| drops below tolerance
        free_steps=200,     # time-budget cap (T = 200 * 0.08 = 16 s); solver adapts within
        nudged_steps=80,
    ).to(device)
    print(f"EnergyOscillatorHead: use_ode={energy_model.use_ode}, "
          f"relax_tol={energy_model.relax_tol}, "
          f"free_steps={energy_model.free_steps}")
    linear_optimizer = torch.optim.Adam(linear_model.parameters(), lr=args.linear_lr, weight_decay=1e-5)
    energy_optimizer = torch.optim.Adam(energy_model.parameters(), lr=args.energy_lr, weight_decay=1e-5)

    best_linear_accuracy = best_energy_accuracy = -1.0
    best_linear_state = best_energy_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_linear_epoch(linear_model, train_loader, linear_optimizer, device)
        train_energy_epoch(energy_model, train_loader, energy_optimizer, device,
                           args.coupling_regularization)
        linear_test, _, _ = evaluate_linear(linear_model, test_loader, device)
        energy_test, _, _, residual = evaluate_energy(energy_model, test_loader, device)
        history.append({
            "epoch": epoch,
            "linear_test_accuracy": linear_test.accuracy,
            "energy_test_accuracy": energy_test.accuracy,
            "energy_equilibrium_residual": residual,
        })
        if linear_test.accuracy > best_linear_accuracy:
            best_linear_accuracy = linear_test.accuracy
            best_linear_state = copy.deepcopy(linear_model.state_dict())
        if energy_test.accuracy > best_energy_accuracy:
            best_energy_accuracy = energy_test.accuracy
            best_energy_state = copy.deepcopy(energy_model.state_dict())
        print(f"Epoch {epoch:03d} | linear={linear_test.accuracy:.4f} | "
              f"energy={energy_test.accuracy:.4f} | residual={residual:.6f}")

    linear_model.load_state_dict(best_linear_state)
    energy_model.load_state_dict(best_energy_state)
    linear_metrics, linear_predictions, targets = evaluate_linear(linear_model, test_loader, device)
    energy_metrics, energy_predictions, energy_targets, energy_residual = evaluate_energy(
        energy_model, test_loader, device
    )
    if not torch.equal(targets, energy_targets):
        raise RuntimeError("Readouts were evaluated on different targets")

    paired = paired_comparison(linear_predictions, energy_predictions, targets)
    results = {
        "dataset": "sklearn_digits",
        "split": {"test_size": 0.2, "random_state": args.seed, "stratified": True},
        "shared_representation": {
            "reservoir": "FrozenFHNReservoir",
            "reservoir_size": args.reservoir_size,
            "feature_dim": feature_dim,
            "standardized_from_training_cache_only": True,
        },
        "linear_readout": {
            **asdict(linear_metrics),
            "trainable_parameters": count_trainable_parameters(linear_model),
        },
        "energy_readout": {
            **asdict(energy_metrics),
            "trainable_parameters": count_trainable_parameters(energy_model),
            "equilibrium_residual": energy_residual,
        },
        "paired_test_comparison": paired,
        "accuracy_difference_energy_minus_linear": energy_metrics.accuracy - linear_metrics.accuracy,
    }

    torch.save(linear_model.state_dict(), args.output_dir / "best_linear_readout.pt")
    torch.save(energy_model.state_dict(), args.output_dir / "best_energy_readout.pt")
    (args.output_dir / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    np.savetxt(args.output_dir / "linear_confusion_matrix.csv",
               confusion_matrix(targets.numpy(), linear_predictions.numpy()), fmt="%d", delimiter=",")
    np.savetxt(args.output_dir / "energy_confusion_matrix.csv",
               confusion_matrix(targets.numpy(), energy_predictions.numpy()), fmt="%d", delimiter=",")

    print("\nFinal comparison on identical test features")
    print(f"Linear readout: {linear_metrics.accuracy:.4%} ({linear_metrics.errors} errors)")
    print(f"Energy readout: {energy_metrics.accuracy:.4%} ({energy_metrics.errors} errors)")
    print(f"Energy minus linear: {energy_metrics.accuracy - linear_metrics.accuracy:+.4%}")
    print(f"Linear-only correct / energy-only correct: "
          f"{paired['linear_only_correct']} / {paired['energy_only_correct']}")
    print(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
