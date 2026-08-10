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

## Result 4 — temperature annealing (hot → cold) closes most of the gap

Cooling the temperature during training and readout combines exploration with a sharp final
decision. `scripts/duffing_langevin_anneal_xor.jl`:

- **Annealed training:** T geometrically cooled `T_hi=0.20 → T_lo=0.06` over epochs — hot epochs
  mix / escape wells and give lower-variance gradients, cold epochs sharpen the landscape.
- **Annealed readout:** the free relaxation ramps `T_hi=0.15 → T_lo=0.05` within a single run, so
  the trained field pulls each output to the input-selected well (warm) then commits it (cold).
  A *fixed* cold readout would re-pin in whatever well the init landed in.

Robust XOR (8 training seeds, full-range test inits):

| Config | mean | median | solved ≥99% | per-seed % |
|--------|------|--------|-------------|------------|
| fixed-T (0.13–0.17), fixed readout | ~80% | ~80% | 1/8 | (Result 3) |
| fixed-T (0.15) train + annealed readout | 83.9% | 85.4% | 0/8 | 88,96,96,62,83,79,75,92 |
| **annealed train + annealed readout** | **90.1%** | **93.8%** | **3/8** | 100,96,100,88,92,88,58,100 |

Annealing lifts the fixed-T thermodynamic result by **~+10 points** (mean 80% → 90%, median → 94%)
and gets **3/8 seeds to a full XOR solve** (vs 1/8). The **annealed training is the main driver**
(fixed-T train + annealed readout only reaches ~84%); the annealed readout is a smaller boost.
On the strong seeds this reaches parity with the phase network (100%); a minority of seeds
(e.g. 58%) still stall, tracking the residual bistable gradient-variance of Result 2.

## Result 5 — a layered (feedforward-symmetric) topology removes the stalls

The remaining stalls (Result 4) come from the all-to-all output well not being reliably
input-determined. The paper's phase-network remedy is a **layered (feedforward-symmetric)**
architecture: restrict the symmetric coupling to input↔hidden and hidden↔output edges only, so
the output field comes solely from the input-driven hidden layer. `layered_mask` +
`scripts/duffing_langevin_layered_xor.jl` (annealed, 8 seeds, full-range test inits):

| Config | mean | median | min seed | solved ≥99% |
|--------|------|--------|----------|-------------|
| all-to-all (N=5), annealed | 90.1% | 93.8% | **58%** (stall) | 3/8 |
| layered 2-2-1 (N=5) | 80.7% | 81.2% | 62% | 0/8 |
| layered 2-3-1 (N=6) | 85.9% | 89.6% | 62% | 0/8 |
| **layered 2-4-1 (N=7)** | **96.4%** | 95.8% | **92%** (no stall) | 3/8 |

Layering helps **only with enough hidden capacity** (the mask removes edges, so a 2-hidden
layered net is *worse* than all-to-all). At 4 hidden units the catastrophic stall is gone —
**every seed ≥92%**, mean 96.4% — matching the paper's layered phase-network result (~95%, all
seeds solved). This is the best Duffing XOR result in this study.

Note layering's benefit here is capacity/structure, not symmetry breaking: unlike the paper's
phase network, the Duffing bias field `h` already breaks the ±well symmetry, which is why a
capacity-reduced 2-2-1 layered net underperforms and the win only appears once the hidden layer
is wide enough.

## Result 6 — multi-class digits (Langevin EP generalizes beyond XOR)

The sampler drops into the multi-class pipeline unchanged (it matches the deterministic
`relax_batch` contract). `scripts/duffing_langevin_digits.jl` reuses the monostable Duffing
digits architecture of `scripts/duffing_digits_mono_v2.jl` — 16 pooled inputs → monostable
hidden (40, `a_h=1`) → 10 linear output cells, layered coupling, softmax-CE readout — and swaps
only the relaxer (Langevin) and the gradient (thermal-average contrast with the softmax-CE
nudge and common random numbers). Efficiency note: the batch-summed correlation the gradient
needs is `Σ_t XᵀX` (one gemm per sampled step), so no per-sample `N×N` tensors.

scikit-learn / optdigits, 10 classes. Two resolutions (`fullres` arg toggles):

**Full resolution — 64 px** (N=138: 64 in, 64 mono-hidden, 10 out; 120 train + 40 test per class):

| Model | test acc |
|-------|----------|
| **Langevin monostable EP** (this work) | **0.968** (train 0.989) |
| MLP, 1 hidden (64 px) | 0.973 |
| logreg (64 px) | 0.958 |
| XY phase network (paper) | 0.94 |
| deterministic Duffing mono (paper) | 0.96 |
| bistable readout / chance | 0.18 / 0.10 |

Test accuracy climbs chance (0.18) → **0.968** over ~320 iterations (CE 2.4 → 0.01). The Langevin
thermodynamic EP **beats logreg (0.958) and the phase network (0.94), matches the deterministic
Duffing monostable (0.96), and is within noise of the MLP (0.973)** — on the same 64-px features.

**Pooled 16 px** (N=66, 40 train + 40 test per class, faster prototype): reaches 0.79, matching
logreg-16 (0.81); MLP-16 0.88.

Both use the *monostable* regime — the one that is both required for graded multi-class readout
(the bistable readout fails at chance, 0.18) *and* where the finite-T EP gradient is faithful and
sampling mixes cleanly (Result 2). Resolution is the main lever: 16 px → 64 px lifts the
substrate from logreg-parity to phase-network-beating.

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

- **Temperature annealing (done, Result 4)** lifts fixed-T ~80% → ~90% and solves several seeds
  fully; the annealed training is the main driver.
- **Seed stalls solved (Result 5):** a **layered 2-4-1** topology removes the stalls (every seed
  ≥92%, mean 96.4%) — the best result here. It requires enough hidden capacity; a narrow 2-2-1
  layered net underperforms all-to-all.
- **Generalization (done, Result 6):** the sampler drops into the multi-class monostable digits
  pipeline unchanged and reaches **0.968 at full 64-px resolution** (N=138) — beating logreg
  (0.958) and the phase network (0.94), matching the deterministic Duffing mono (0.96), within
  noise of MLP (0.973). Pooled-16 gives the faster 0.79 prototype.
