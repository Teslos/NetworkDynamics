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
| `dynamics_bypass_control.jl` | **Critical control**: removes the RK4 dynamics entirely (phases = `W_IN·x`) and asks whether trainable-`W_IN` + sin/cos still solves parity. It does (100%) — so the dynamics are inert here (see caveat below). |
| `temporal_parity.jl` | **Temporal/delayed parity** — k bits presented *sequentially* to transient input nodes, output = parity. Memory is required (memoryless model = chance by construction), so the dynamics are load-bearing here (bypass cannot apply). Compares trained coupling (UDE) vs fixed-random coupling (reservoir). Uses hand-rolled fixed-step RK4 + Enzyme. |
| `temporal_parity_adaptive.jl` | Same temporal task with an **adaptive solver** (`OrdinaryDiffEq` Tsit5 + `SciMLSensitivity` adjoint + Zygote). Shows the trained-coupling instability seen under fixed-step RK4 was an *integrator artifact* — training is stable here. |
| `adjoint_vjp_backends.jl` | Diagnostic: which adjoint VJP backend works for the Kuramoto-UDE on this stack. `ZygoteVJP` = correct; `EnzymeVJP` = broken (silent zero gradient / segfault) on Julia 1.12 + Windows. |
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

### Critical caveat: on parity, the dynamics are inert (dynamics-bypass control)

`dynamics_bypass_control.jl` removes the RK4 settling entirely (phases = `W_IN·x`,
no oscillators) and trains `W_IN` + the sin/cos readout:

| Model | Acc |
|---|---|
| tanh random features (H=1024) | 78.5% |
| random Fourier features (sin/cos, H=1024) | 75.8% |
| bypass, **fixed** random `W_IN` (sin/cos, ridge) | 54.4% |
| bypass, **trainable** `W_IN` (sin/cos, **no dynamics**) | **100.0%** |
| UDE, trainable `W_IN` (**with** dynamics) | 100.0% |

Removing the dynamics changes nothing. Parity of 3 ±1 bits is a **single Fourier
mode** (`sin(π/2·Σxᵢ)` separates the classes), so a trainable linear projection +
sin/cos readout solves it on its own; the oscillator dynamics do **not**
contribute. Therefore **this parity result does NOT show that the UDE dynamics are
a capable encoder** — it shows that a *trainable input projection* + *periodic
readout* solves parity, which the UDE happens to contain.

**Honest takeaway:** the only thing that matters on parity is whether `W_IN` is
trainable (fixed → chance; trainable → 100%), for *any* model with a periodic
readout. To test whether the **oscillator dynamics themselves** add value, you need
a task where settling/recurrence is essential — a temporal / working-memory task
(e.g. the temporal-XOR, a delayed task, or NARMA) — not static parity, where they
are provably redundant. The random-ESN and random-features baselines here only
establish that *random static/recurrent substrates* fail parity; a trivial
trainable-linear + sine model beats them too.

The optdigits story is unaffected in direction: an easy dataset (ESN saturates,
see the ceiling sweep) *and* a fixed random `W_IN_UDE` in the digit model.

## Temporal / delayed parity (PRELIMINARY — where the dynamics *are* load-bearing)

Static parity is the wrong task to judge the dynamics (they're inert, above). The
temporal version presents the k bits **sequentially to transient input nodes** and
reads parity at the end, so the network must *remember* — a memoryless model is at
chance by construction, and the dynamics-bypass cannot apply.

Preliminary K=3 result (8 patterns, single seed — a **memorization** test, not a
generalization claim):

| Model | Acc |
|---|---|
| memoryless (by construction) | 4/8 (chance) |
| random reservoir (fixed coupling, trained readout) | 7/8 (stable) |
| UDE (trained coupling), **fixed-step RK4 + lr 0.01** | unstable: 8→6→7→5→6/8 |
| UDE (trained coupling), **adaptive Tsit5 + adjoint + lr 1e-3** | stable, monotone: 4→5→6→7/8 |

**Integrator lesson (important):** the trained-coupling instability under hand-rolled
fixed-step RK4 was an **artifact of the integrator + too-high LR** (the same
Euler-blows-up failure fixed elsewhere by RK45), *not* evidence that UDE training is
unstable. With an adaptive solver the training is stable and monotone. So: distrust
instability seen under fixed-step RK4; verify with an adaptive solver.

**AD-backend lesson:** for adaptive-solver gradients on this stack (Julia 1.12 +
Windows), use `InterpolatingAdjoint(autojacvec=ZygoteVJP())`. `EnzymeVJP` is broken
here — `InterpolatingAdjoint`+`EnzymeVJP` silently returns a **zero gradient** and
`GaussAdjoint`+`EnzymeVJP` **segfaults**. Reserve Enzyme for the fast fixed-step /
hand-rolled RK4 path; the adjoint route is ~200× slower per gradient.

**Caveats / not yet a result:** K=3 is only 8 patterns (memorization, single seed);
the trained-vs-random-*dynamics* comparison is not settled. A real study needs a
stable integrator, larger k (16–32 patterns), and multiple seeds.
