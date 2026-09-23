# The XY gradient-fidelity check was validating the wrong objective

Scripts: `scripts/check_ep_gradient_fidelity.jl xy`, `notebooks/EP-XY-Network-Claude.jl`
Run: 2026-09-12, `XY_FD_COST=log` (corrected) vs `XY_FD_COST=cos` (previous behaviour)

## The mismatch

The XY nudge adds, on each output cell,

```
F_j  -=  beta * sin(d) / (1 + cos(d)),      d = phi_j - target_j
```

which is `-beta * dC/dphi_j` for **Wang's logarithmic phase cost**

```
C_log(d) = -log((1 + cos d)/2)         (0 at d=0, divergent at d=pi)
```

because `d/dd[-log(1+cos d)] = sin d/(1+cos d)`. The EP parameter gradient the
notebook returns therefore estimates `dC_log/dtheta`.

`batch_costs`, however, reports the **cosine deviation** `C_cos(d) = (1-cos d)/2`,
and `check_ep_gradient_fidelity.jl` differenced *that* to build its "true
gradient" reference. The two objectives share a minimiser and agree to first
order about `d = 0` (both `~ d/2`), but their derivatives differ by a per-output
factor

```
dC_log/dd  =  sec^2(d/2) * dC_cos/dd
```

which is 1 only at `d = 0` and diverges at `d = pi`. Since the parameter gradient
sums `(dC/dd_i)(dd_i/dtheta)` over outputs and samples with *different* `d_i`, the
two parameter gradients are not proportional, and the check was comparing EP
against an objective nobody was descending.

The Duffing arm of the same script has no such mismatch: its nudge `-beta(x-t)`
is exactly the derivative of its reported `batch_cost` `(1/2)(x-t)^2`, which is
why that arm reports cosine similarity 0.99999 while the XY arm did not.

## Fix

* `xy_force!` and `batch_costs` now state which objective each represents.
* `batch_log_cost(equilibria, target, output_index)` evaluates `C_log`; it is
  the reference any finite-difference check must use.
* `EP_param_gradient` returns `log_cost` as a fifth value (callers unpacking
  four are unaffected); the logged `cost` stays the bounded cosine diagnostic so
  that recorded training curves remain comparable.
* The fidelity script differences `C_log` by default; `XY_FD_COST=cos`
  reproduces the old, mismatched comparison.

## What changes (XOR, N=5, symmetric estimator, beta = 0.01)

| operating point | metric | old (cos reference) | corrected (log reference) |
|---|---|---:|---:|
| B: lightly trained | cos(g_EP, g_FD) | 0.9808 | **0.99962** |
| B | \|\|g_EP\|\|/\|\|g_FD\|\| | 1.184 | **1.124** |
| B | weights block cos | 0.9412 | 0.9401 |
| B | h block cos | 0.9921 | **1.0000** |
| A: random init | cos | 0.624 | 0.775 |
| A | \|\|g_EP\|\|/\|\|g_FD\|\| | 19.04 | **1.67** |
| A | h block cos | 0.9628 | **0.9994** |
| C: strongly coupled | cos | -0.354 | 0.067 |
| C | \|\|g_EP\|\|/\|\|g_FD\|\| | 4706 | 0.0019 |

Magnitude ratio across the beta sweep at the random init (A):

| beta | 0.2 | 0.1 | 0.05 | 0.02 | 0.01 |
|---|---:|---:|---:|---:|---:|
| old (cos reference) | 6.07 | 14.54 | 32.46 | 26.21 | 19.04 |
| corrected (log reference) | 0.08 | 0.42 | 0.76 | 0.90 | 1.67 |

## Consequences

1. **At a well-conditioned operating point the EP gradient is far more faithful
   than the repo recorded.** Point B goes from 0.981 to 0.9996 cosine similarity
   — the residual 2% was the objective mismatch, not EP bias.
2. **The "weight gradient grows as 1/beta" claim does not survive.** It rested on
   the magnitude ratios at the random init, which reached 19-32x under the
   mismatched reference. Against the objective actually being descended the ratio
   is O(1) and *approaches* 1 as beta falls (0.08 -> 0.90 at beta=0.02), i.e. the
   estimator converges rather than blowing up. The inflation came from the
   denominator: at random init the outputs sit at large |d|, where
   `sec^2(d/2)` makes the true (log-cost) gradient ~29x larger in norm than the
   cosine-cost gradient it was being compared against (\|\|g_FD\|\| 0.289 -> 8.40).
3. **The basin-sensitivity claim does survive, in weaker form.** The weight block
   remains the least faithful (cos 0.78-0.94 vs 0.999-1.000 for the bias fields),
   and the bias blocks still dominate the gradient norm, so the explanation for
   why XOR trains is unchanged.
4. **Point C exposes a property of the log cost, not of EP.** With strong
   coupling an output parks near `d = pi`, where `C_log` diverges; the true
   gradient norm is then ~10^3 and EP's O(beta) probe sees ~0.2% of it. This is
   the cost function's singularity, and it is a reason to prefer the bounded
   cosine cost as a *training monitor* even though the log cost is what the
   dynamics descend.

## Pending

`paper/main.tex:491-500` still states the coupling gradient "can grow as
$1/\beta$"; that sentence needs to be rewritten against these numbers, and the
near-exactness of the bias blocks stated with the corrected figures.
