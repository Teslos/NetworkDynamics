# Introduction
We are trying to get NetworkDynamics to run the general nonlinear oscillators.
By creating the graph we can create different enviroments and train different
cases. 
# Examples
**Duffing.jl** - calculates the Duffing oscillators on the network graph

**FitzHug-Nagumo** - calculates the FitzHugh-Nagumo model for the nodes in the network graph.

**Diffusion-optimiz.jl** - Optimization of the network dynamics for the diffusion constant.

# How to use

# Results
The oscillations in Duffing network using the random eigen frequencies chosen randomly around given value $\omega$:


![Duffing oscillators](./figs/duffing_barabasi_albert.png)

Using the larger deviation from the average frequency produces this result:

![Duffing oscillator 0.5](./figs/duffing_barabasi_albert_0_5.png)

This are solution to the network graph of the brain system:

![Duffing oscillator brain](./figs/duffing_brain_graph.png)

# Learning using reservoir computing
The learning procedure for the cases of Drybean classification uses the reservoir compute by network of FHN oscillators.
The network state is set between equilibrium and chaotic state. We expect highest computing ability in this particular state. 
Additional things to be done and tested are:
- Training efficiency with different network topologies
- Parallel solution of the reservoir using the GPU and maybe ModelingToolkit library instead of NetworkDynamics
- Interactive demo to set up the active nodes in the graph (using JavaScript)

# Equilibrium Propagation on oscillator networks

Beyond reservoir computing (fixed dynamics + trained readout), this repo trains
oscillator networks **end-to-end** with **equilibrium propagation (EP)** — a
physics-based, local-update alternative to backpropagation for energy-based systems
that relax to a fixed point. We study two substrates and find they obey the same
regime split. (Implements/extends Wang et al. 2024; see also Berneman & Hexner 2026
for EP on dissipative *dynamical* systems.)

**Two substrates**
- **XY / Kuramoto** (`notebooks/EP-XY-Network-Claude.jl`) — phase oscillators;
  logic encoded in a *phase* (smooth, on the circle).
- **Duffing** (`notebooks/EP-Duffing-Network.jl`) — damped double-well oscillators;
  logic encoded in *position* at the ±1 wells (bistable).

**XOR** (`scripts/run_ep_xor.jl`, `scripts/multiseed_convergence.jl`,
`scripts/duffing_layered*.jl`)

| substrate | result |
|---|---|
| XY / Kuramoto | robust, 10/10 seeds |
| Duffing (naive) | ~chance — bistable forward inference (output basin chosen by init, not input) |
| Duffing (basin-averaging + layered) | **95%, 6/6 seeds** |

The Duffing failure is the standard EP multistability problem, fixed by (a)
**basin-averaging** (Wang's remedy: full-range init + gradient averaged over basins)
and (b) a **layered** feedforward-symmetric architecture (`results/ep_duffing_layered.md`).
Gradient-fidelity and mechanism studies: `scripts/check_ep_gradient_fidelity.jl`,
`results/ep_duffing_basin_tracking.md`.

**Handwritten digits (sklearn 8×8, `data/digits/`)**

| model | test acc |
|---|---|
| XY (Wang protocol, full 64px) | 0.94 (`results/xy_digits_stage3.md`) |
| Duffing, bistable readout | ~0.18 (chance) |
| Duffing, **monostable** hidden + stable training, full 64px | **0.96** = logreg / MLP (`results/ep_duffing_digits_mono_fullres.md`) |

The deep-double-well (bistable) Duffing **cannot** do multi-class — 10 independent
bistable output cells (2¹⁰ basins, no competition) plus multistable hidden features.
Making the cells **monostable** (single-well, smooth saturating nonlinearity) +
graded/softmax readout + Landau-style annealing recovers **full** multi-class
capability: at full resolution the monostable Duffing reaches 0.96, matching the
logreg/MLP baselines and edging out XY (both substrates ≈ baseline in their smooth
regime).

**Unified conclusion.** For *both* substrates: the **bistable / deep-double-well
regime** suits **single-bit / memory** tasks (XOR), while the **monostable / smooth /
phase regime** is required for **rich multi-class** tasks (digits). EP trains
oscillator networks well provided the substrate operates in its smooth regime.

The XY scale-up (`scripts/xy_digits_stage*.jl`) and the full Duffing arc
(`scripts/duffing_*.jl`) are documented stage-by-stage under `results/ep_*.md` and
`results/xy_*.md`. Reference papers are in `docs/`.

# References

Chaos 31, 013108 (2021)

Wang, Wanjura & Marquardt, "Training coupled phase oscillators as a neuromorphic
platform using equilibrium propagation," *Neuromorph. Comput. Eng.* 4 034014 (2024).

Berneman & Hexner, "Equilibrium Propagation for Dissipative Dynamics,"
*Advanced Intelligent Systems* (2026).