# Parity encoder experiments

Do the trained oscillator (UDE Kuramoto) dynamics act as a genuine **learned
encoder** that a random reservoir cannot match — and what limits it?

These scripts isolate that question on a controlled **k-parity** task (label =
parity of `k` of `d` bits, the rest nuisance), where random features provably
struggle but the task is learnable. Reserved for a future paper (not the current
manuscript).

Run with the project env, e.g. `julia --project=. scripts/parity_encoder/<script>.jl`.
Gradients use Enzyme reverse-mode (parameter-count-independent), so training the
input projection is free.

## Scripts

| Script | What it does |
|---|---|
| `gapcheck_parity.jl` | Finds the Goldilocks difficulty: linear vs random-features vs MLP, swept over parity order `k`. Confirms `k=3` (d=20) gives a clean gap. |
| `parity_ude.jl` | The main experiment: UDE encoder with **fixed random** vs **trainable** input projection `W_IN`, against random-features (and MLP reference). |
| `esn_ceiling_sweep.jl` | Ceiling control for the *digit* model: sweeps ESN size (5 seeds), holding the trained UDE fixed, to show the ESN+UDE gain vs headroom. Requires `results/models/ude_subreservoir.jld2` (run `src/models/UDE-SubReservoir.jl` first). |

## Headline results

**Gap-check** (d=20; linear / random-features / MLP):
`k=2` → 51 / 100 / 100 (random saturates), **`k=3` → 51 / 71 / 100 (clean gap)**, `k=4` → 50 / 57 / 100.

**Parity UDE** (k=3, d=20, N_OSC=8), test accuracy:

| Method | Acc |
|---|---|
| Random features | 78.5% |
| UDE, **fixed random** `W_IN` | **50.8%** (chance — projection destroys parity) |
| UDE, **trainable** `W_IN` | **100.0%** (loss → 0 in ~150 iters) |
| MLP (reference) | ~100% |

**Takeaway:** the oscillator UDE is a capable learned encoder — it solves a task
random reservoirs cannot — **but only if the input projection is trainable**. A
fixed random projection linearly mixes the parity bits into the nuisance dims and
destroys the signal before the dynamics see it. This also explains the small
ESN+UDE gain on optdigits: an easy dataset (ESN saturates, see the ceiling sweep)
*and* a fixed random `W_IN_UDE` in the digit model.
