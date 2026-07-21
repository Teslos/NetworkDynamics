# EP-XY-XOR on FPGA (SystemVerilog) — for EDA Playground

Hardware (fixed-point SystemVerilog) implementation of **Equilibrium Propagation
(EP)** training for an **XY / Kuramoto phase-oscillator** network that learns the
**XOR** gate. Ported from `notebooks/EP-XY-Network-Claude.jl` (the code behind the
paper's `ep_xor_cost_history.png`).

- **`design.sv`** — the synthesizable-style fixed-point DUT `ep_xy_xor_top`. Runs
  the **full EP training loop on-chip**: free relaxation → +β / −β nudged
  relaxations → EP gradient → **SGD + gradient clipping**, over epochs and the 4
  XOR patterns; then evaluates the XOR truth table.
- **`testbench.sv`** — self-checking testbench. Includes a behavioral **`real`
  golden reference** (a faithful transcription using `$sin`/`$cos`) that trains
  the same algorithm and prints its cost history + XOR truth table, then runs the
  DUT and checks its XOR output (PASS/FAIL per pattern). EPWave dump included.

## Run it on EDA Playground

1. Go to https://edaplayground.com and sign in.
2. Left sidebar:
   - **Languages & Libraries → Testbench + Design:** `SystemVerilog/Verilog`
   - **Tools & Simulators:** **Aldec Riviera-PRO** (best `real`/`$sin` support).
     *Icarus Verilog* also works as a free fallback.
   - Tick **Open EPWave after run**.
3. Paste `design.sv` into the **Design** pane (right) and `testbench.sv` into the
   **Testbench** pane (left). (Top module is `tb`.)
4. Click **Run**.

## What you should see

```
---- GOLDEN REFERENCE (real) : EP-XY-XOR ----
[REF]  epoch 1      cost = 0.5xxxx
[REF]  epoch 250    cost = 0.xxxxx
 ...
---- GOLDEN REFERENCE : XOR truth table ----
  pat 0  in=(F,F)  phi_out=-1.55 rad -> F (expect F)
  pat 1  in=(F,T)  phi_out=+1.5x rad -> T (expect T)
  pat 2  in=(T,F)  phi_out=+1.5x rad -> T (expect T)
  pat 3  in=(T,T)  phi_out=-1.5x rad -> F (expect F)

---- FIXED-POINT DUT : training ----
[DUT]  epoch 0    cost = ...
 ...
---- FIXED-POINT DUT : XOR truth table ----
  pat 0 -> F (expect F) PASS
  ...
*** DUT XOR: ALL 4 PATTERNS PASS ***
```

Open **EPWave** to watch the phases relax, the cost trace, and weights evolve.

## Status: validated on EDA Playground

Confirmed on **Aldec Riviera-PRO**: **both** the fixed-point DUT and the `real`
golden reference train and pass the full XOR truth table (4/4). The golden
reference converges to cost ~1e-3; the fixed-point DUT to ~0.01–0.07 (it jitters
because of SGD + the per-epoch LFSR re-init noise, but stays well on the correct
side of the ±90° decision boundary). The two agreeing is the cross-check that the
fixed-point implementation is correct, not merely plausible.

After training, the testbench also prints the **learned parameters** (coupling `W`,
field `h`, preferred phases `psi`) via the DUT's `W_o`/`h_o`/`psi_o` readout ports.

Notes:
- **The golden reference is the ground-truth demonstrator** (full-precision `real`).
  If you retune and something breaks, use it to tell an *algorithm* problem (both
  wrong) from a *quantization/scaling* problem (only the DUT wrong).
- Two deliberate simplifications vs. the Julia reference (both flagged in code):
  **SGD + clip** instead of Adam (avoids a fixed-point 1/sqrt), and fixed-step
  **Euler** relaxation instead of the adaptive `Tsit5` solver.

## Tuning the fixed-point DUT (parameters at the top of `ep_xy_xor_top`)

If the DUT cost doesn't fall or XOR fails while the golden reference works, adjust:

| Param | Meaning | If cost diverges / stalls |
|---|---|---|
| `STEP_SH` / `STEP_MUL` | Euler step size (`dφ = F·STEP_MUL >> STEP_SH`) | diverges → **increase** `STEP_SH` (smaller step); stalls → decrease |
| `MAX_ITERS` | Euler steps per relaxation | not reaching equilibrium → increase |
| `STEADY` | convergence threshold on `max|dφ|` (brad) | too coarse → decrease |
| `GSC` | gradient → Q6.12 scale | no learning → increase; unstable → decrease |
| `ETA_SH` | SGD learning-rate shift (η ≈ 2^−`ETA_SH`) | too slow → **decrease** `ETA_SH`; unstable → increase |
| `PSI_SH` | ψ (bias phase) update step | ψ often dominates XOR; tune first |
| `NUDGE_SH` | aligns β·tanhalf to force units | wrong nudge magnitude |
| `N_EPOCH` | training epochs | raise for more convergence (slower sim) |

Recommended order: get relaxation stable (`STEP_SH`, `MAX_ITERS`, `STEADY`) so the
DUT's *free-phase* output tracks the golden reference, **then** tune the update
(`GSC`, `ETA_SH`, `PSI_SH`) so cost falls.

**Runtime:** the reference used 5000 epochs; the DUT default is `N_EPOCH=400` to fit
the emulator's time budget. Each epoch = 4 patterns × 3 relaxations × up to
`MAX_ITERS` Euler steps, so raising these lengthens the run.

## Algorithm mapping (from `notebooks/EP-XY-Network-Claude.jl`)

- Energy `E = −½ Σ k_ij cos(φ_i−φ_j) − Σ h_i cos(φ_i−ψ_i)`; force `dφ/dt = −∂E/∂φ`.
- Nudge on the output node: `−β·sin(d)/(1+cos(d))` = `−β·tan(d/2)`, `d=φ_out−target`
  (implemented via the saturated `tanhalf` ROM).
- Symmetric EP: free (β=0) → φ⁰; then from φ⁰: +β → φ⁺, −β → φ⁻; `scale=2β`.
- `gW_jk = (cos(φ⁻_j−φ⁻_k) − cos(φ⁺_j−φ⁺_k))/(2β)`; analogous `gh`, `gψ` (biases
  **are** trained).
- N=5, inputs {0,1} clamped, output {4} (nudged/read), hidden {2,3}, all-to-all
  symmetric W (zero diagonal). Encoding FALSE=−π/2, TRUE=+π/2. β=0.01.

These files are for pasting into EDA Playground; they are not wired into the Julia
build.
