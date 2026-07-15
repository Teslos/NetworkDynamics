# Monostable Duffing on real MNIST (28x28)

Script: `scripts/duffing_mnist_monostable.jl`
Run: `julia -t auto --project=. scripts/duffing_mnist_monostable.jl` (2026-07-15)

## Question

Our best Duffing digit result (0.96) was on the *scikit-learn* 8x8 `load_digits`
set. Does the same monostable single-basin setup transfer to **genuine MNIST**
(28x28 handwriting, loaded via `MLDatasets`, no PyCall)?

## Setup

Real MNIST, 2x2 average-pooled 28x28 -> **14x14 (196 inputs)** to keep the
second-order Duffing relaxations tractable (N=246); balanced subset of 100
train / 50 test per class (1000 / 500). Otherwise identical to the sklearn full-res
run: monostable hidden (a>0), linear/softmax readout, symmetric +-beta gradient,
Landau annealing, best-checkpoint, layered input->hidden->output. Baselines: logreg
and MLP on the same 14x14 features.

## Results

| model                | train | test  |
|----------------------|------:|------:|
| Duffing mono (MNIST) | 0.931 | 0.850 |
| logreg (14x14)       | —     | 0.868 |
| MLP (14x14)          | —     | 0.884 |

Chance 0.10. 300 iters in 100 s. Test peaked at 0.854; CE 2.3 -> ~0.3 (some bounce).
Reference: monostable Duffing on sklearn 8x8 digits = 0.96.

## Conclusion — the substrate transfers to real MNIST

The monostable Duffing **generalizes to genuine MNIST**, reaching 0.85 test while
fitting the train set (0.93). It lands ~2 points below logreg (0.868) and ~3 below
MLP (0.884) on the *same* 14x14 features -- close, but not matching, unlike the
sklearn set where it tied logreg at 0.96.

Two honest points:
1. **MNIST is genuinely harder** than sklearn digits: logreg itself falls from 0.96
   (sklearn 8x8) to 0.87 (MNIST 14x14). Absolute accuracy is lower for everyone.
2. **The gap to the linear baseline is small and looks optimization/resolution-
   limited**, not a substrate failure: 300 iterations, a 1000-image subset, 14x14
   pooling, and visible test bounce -- the same signature that a v1->v2 style
   retune (more iterations/data, lower LR, early stopping) closed on the sklearn
   set. More resolution / samples would likely narrow it.

**Caveat:** this is a *tractable* MNIST test (14x14-pooled, balanced 1000/500
subset), not the full 28x28 / 60k benchmark. Full-resolution MNIST (784 input
cells, N~800) is impractical for the second-order Duffing relaxation without a
faster solver or a convolutional/patch front-end.

## Takeaway

Combined with the sklearn result (0.96 = logreg) and the XOR result (layered Duffing
95%), this shows the monostable Duffing is a real classifier that **transfers across
digit datasets**, competitive with linear/MLP baselines on the same features. The
bistable regime remains the memory/single-bit tool; the monostable/smooth regime
carries multi-class classification, on sklearn digits *and* real MNIST.
