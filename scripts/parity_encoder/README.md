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
| `esn_parity_baseline.jl` | Direct "learned encoder vs random reservoir" test: a random recurrent ESN (bits fed sequentially) on the same k=3 parity, swept over reservoir size to show failure is not a capacity issue. |
| `esn_ceiling_sweep.jl` | Ceiling control for the *digit* model: sweeps ESN size (5 seeds), holding the trained UDE fixed, to show the ESN+UDE gain vs headroom. Requires `results/models/ude_subreservoir.jld2` (run `src/models/UDE-SubReservoir.jl` first). |

## Headline results

**Gap-check** (d=20; linear / random-features / MLP):
`k=2` → 51 / 100 / 100 (random saturates), **`k=3` → 51 / 71 / 100 (clean gap)**, `k=4` → 50 / 57 / 100.

**Parity UDE** (k=3, d=20, N_OSC=8), test accuracy — full substrate comparison:

| Substrate | Acc |
|---|---|
| Linear (logreg) | ~50% (chance) |
| Random recurrent ESN (sequential, ≤2000 nodes) | **~65%** (plateaus — see below) |
| Random static features | 78.5% |
| UDE, **fixed random** `W_IN` | **50.8%** (chance — projection destroys parity) |
| UDE, **trainable** `W_IN` | **100.0%** (loss → 0 in ~150 iters) |
| MLP (reference) | ~100% |

**Random ESN plateau** (`esn_parity_baseline.jl`, k=3, sequential input):
N_RES 100 → 57.5%, 500 → 62.8%, 1000 → 64.1%, 2000 → 65.0% (±0.2, 2 seeds).
The reservoir saturates ~65% regardless of size, so its failure is *fundamental*
(random dynamics can't implement the XOR-accumulation parity needs), not a
capacity limit. Notably it does *worse* than random static features (78.5%):
parity is order-invariant, so the recurrence adds nothing and the one-bit-per-step
input is just a bottleneck.

**Takeaway:** the oscillator UDE is a capable learned encoder — it solves a task
**no random substrate can** (neither static random features nor a recurrent ESN up
to 2000 nodes get near it) — **but only if the input projection is trainable**. A
fixed random projection linearly mixes the parity bits into the nuisance dims and
destroys the signal before the dynamics see it. This also explains the small
ESN+UDE gain on optdigits: an easy dataset (ESN saturates, see the ceiling sweep)
*and* a fixed random `W_IN_UDE` in the digit model.
