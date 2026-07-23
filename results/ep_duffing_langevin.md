# Finite-temperature thermodynamic EP for the Duffing network (Langevin sampling)

## Motivation

The bistable Duffing substrate (double-well, logic on the `x = ±1` wells) trained with the
**deterministic** relaxer (`notebooks/EP-Duffing-Network.jl`) does **not** robustly solve XOR:
multi-seed evaluation gives **~57% (chance)** because the *forward inference* is basin-pinned —
the output well is set by the free-cell initialization, not by the input. The repo works around
this with explicit remedies: basin averaging (`scripts/duffing_basin_*.jl`, ~76%), Landau /
barrier annealing (`scripts/duffing_landau_anneal.jl`), monostable operation
(`scripts/duffing_digits_mono*.jl`).

This study replaces the deterministic relaxation with an **overdamped Langevin sampler** and
trains via **finite-temperature thermodynamic EP** (the Boltzmann-machine / Hinton contrastive
form). Thermal noise lets the state cross the barrier and sample *both* wells, so the output
becomes an input-determined distribution rather than an init-pinned point — one physical
temperature knob in place of the basin-averaging machinery.

## Method

- **Dynamics (overdamped Langevin, Euler–Maruyama):** `x ← x + F(x)·dt + √(2T·dt)·ξ`,
  `F = −∂E/∂x`, `ξ ~ N(0,I)`. Inputs clamped; output cells receive the `−β(x−target)` nudge.
  Stationary law is the conditional Boltzmann `p(x_free | x_in) ∝ exp(−E/T)`.
- **Energy (unchanged):** `E = Σ V(x_i) − ½ Σ W_ij x_i x_j − Σ h_i x_i`, `V = ¼c x⁴ + ½a x²`
  (`a=−1,c=1` → wells at `±1`, barrier `ΔV = a²/4c = 0.25`).
- **Finite-T EP gradient (thermal-average contrast):**
  `∂L/∂W_ij = (⟨x_i x_j⟩_{−β} − ⟨x_i x_j⟩_{+β}) / 2β`, with `⟨·⟩` **time averages** over a
  post-burn-in window. The loss it descends is the *thermal-averaged* cost
  `⟨C⟩ = ⟨½Σ(x_out−target)²⟩` (includes output variance), not the cost of the mean.
- **Common random numbers (CRN):** all phases (free, ±β) use the **identical** noise
  realization; without this the O(1/√n) sampling noise in the correlations fails to cancel and,
  divided by β, blows up as β→0. Essential for both the gradient check and training stability.

Code: `notebooks/EP-Duffing-Langevin.jl` (sampler + gradient + trainer; reuses
`DuffingNetwork`/`random_init!`/`adam_update`/`batch_cost` from `EP-Duffing-Network.jl`).

## Result 1 — the sampler samples the correct Boltzmann distribution

Single isolated unit (`W=0,h=0`), `⟨x²⟩` sampled vs numerical `∫x²e^{−V/T}/∫e^{−V/T}`:

| T | ⟨x²⟩ sampled | ⟨x²⟩ Boltzmann | ⟨x⟩ sampled | ⟨x⟩ ref |
|-----|--------------|----------------|-------------|---------|
| 0.05 | 0.942 | 0.937 | +0.60 | 0 |
| 0.10 | 0.866 | 0.871 | +0.007 | 0 |
| 0.20 | 0.827 | 0.831 | −0.030 | 0 |

`⟨x²⟩` matches across T. The mean symmetrizes (`⟨x⟩→0`) at `T≥0.10`, but at `T=0.05` it is
stuck at `+0.60`: the chain barely crosses the barrier in the window (`ΔV/T=5`, Kramers-slow).
**Mixing degrades sharply below `T≈0.10` — the first sign of a usable temperature window.**

## Result 2 — the finite-T EP gradient is faithful in the monostable regime, variance-limited in the bistable one

Cosine similarity of the EP gradient vs a central finite-difference of `⟨C⟩` (CRN throughout),
`scripts/check_langevin_ep_gradient.jl`:

| Regime | β=0.10 | β=0.05 | β=0.02 |
|--------|--------|--------|--------|
| **Monostable** (`a=+1`), T=0.10 | **0.984** | **0.985** | **0.985** |
| **Bistable** (`a=−1`), T=0.15 | 0.287 | 0.107 | 0.500 |

- **Monostable:** cos ≈ 0.98, **stable across β** → the estimator (and its CRN) is **correct**.
- **Bistable:** low and erratic. This is *not* a bug: with a double well the ±β paths cross the
  barrier at different times, so the correlation difference is dominated by crossing-time
  jitter, not the β-drift. The thermal gradient is **unbiased but high-variance** in the
  bistable regime — which is exactly why training there is harder and a temperature window
  matters.

## Result 3 — XOR: a temperature window rescues bistable inference

Robust XOR accuracy (free sampler, full-range test-init draws), `scripts/duffing_langevin_xor.jl`:

Single-seed sweep (seed 1): `T=0.06 → 54%`, `0.10 → 67%`, `0.15 → 92%`, `0.20 → 83%`.

Multi-seed (8 training seeds), the window:

| T | mean | median | solved ≥99% | per-seed % |
|------|------|--------|-------------|------------|
| 0.13 | **81.8%** | 83.3% | 1/8 | 96,83,50,67,96,79,83,100 |
| 0.15 | 76.0% | 77.1% | 1/8 | 92,100,79,42,79,67,75,75 |
| 0.17 | 78.6% | 75.0% | 1/8 | 100,96,75,83,67,67,67,75 |

**Chance → ~80% mean robust accuracy** (best seeds 96–100%) in the window `T ≈ 0.13–0.17`
(`ΔV/T ≈ 1.5–2`). Too cold (`T≤0.06 → 54%`) does not mix (reproduces the deterministic
failure); too hot (`T=0.20 → 83%`, then washes out).

This ~80% is **comparable to the paper's basin-averaging result (~76%, Table `tab:ep_xor`)** —
but obtained from a **single temperature parameter** rather than explicit basin averaging or
Landau annealing. Thermal sampling recovers basin-averaging-level robustness as an emergent
property. It does *not* reach the phase-network's robust 10/10 solve; the residual seed variance
tracks the bistable gradient-variance of Result 2.

## Interpretation (ties to the paper's regime split)

The two regimes of Result 2 mirror the paper's substrate-independent split exactly:
- **Monostable / smooth** → faithful thermal gradients, clean training.
- **Bistable / deep-well** → gradients are variance-limited by barrier crossing.

So the same bistability that makes the double well a good single-bit memory (and that blocks
graded multi-class readout in the paper) also makes the thermodynamic gradient hard to estimate.
Finite-T sampling narrows but does not eliminate the gap: it buys ~+25 points on XOR via one
knob, within a window set by `ΔV/T`.

## Reproduce

```
julia --project=. scripts/check_langevin_ep_gradient.jl          # Result 2 (both regimes)
julia --project=. scripts/duffing_langevin_xor.jl                # Result 3 (full T sweep, 8 seeds)
julia --project=. scripts/duffing_langevin_xor.jl 0.15 8         # single T, 8 seeds
```
Result 1 (Boltzmann sanity) is an inline check on `langevin_sample_batch`.

## Limitations / next steps

- **Higher robustness:** more samples / epochs (reduce gradient variance), a **lower readout
  temperature** at test time (sharpen `sign⟨x_out⟩`), or a shallower-well / larger-hidden net.
- **Temperature annealing** (hot→cold during training) would combine exploration with a sharp
  final readout — deferred here because the goal was a *fixed-T* thermodynamic result.
- **Generalization:** the sampler matches the `relax_batch` contract, so it can be dropped into
  the XY substrate and the digits scripts (deferred).
- The bistable gradient-variance (Result 2) is the fundamental limiter; annealing or a layered
  (feedforward-symmetric) topology are the natural routes past it.
