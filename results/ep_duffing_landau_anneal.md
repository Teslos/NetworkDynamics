# Landau / deterministic-annealing EP-Duffing: cool through the phase transition

Script: `scripts/duffing_landau_anneal.jl`
Run: `julia --project=. scripts/duffing_landau_anneal.jl` (2026-07-01)

## Idea

Phase-field / Landau free energy `F(x) = ½ a(T) x² + ¼ b x⁴`, with the quadratic
coefficient set by temperature (`a(T) = a₀(T − T_c)`, `b` fixed). Above `T_c`
(`a>0`) the well is single — monostable, equilibrium uniquely set by the field
(bias + input) → input-determined inference, no basin ambiguity. Below `T_c`
(`a<0`) it's a double well (wells at ±√(−a/b)). Cooling through `T_c` is a
supercritical pitchfork; a nonzero symmetry-breaking field (the input) unfolds it
(cusp), so the state slides continuously into the input-selected well and is frozen
in as the barrier grows — deterministic annealing / graduated non-convexity.

Tests the refinement over earlier annealing: **start above `T_c`** (`a>0`,
genuinely monostable) vs starting at a shallow double well (`a<0`, already
bistable). Only `a` is annealed; `b=c=1` fixed, so the minima *emerge* from 0 and
grow to ±1 (a true order-parameter transition). Both with basin-averaging (Minit=40,
full-range init), 6 seeds, 1200 epochs, honest full-range test.

## Results — robust XOR under full-range test init

| schedule                    | deep a=−1 (mean/max/min) | solved | a=−0.5 | med cost |
|-----------------------------|--------------------------|--------|--------|----------|
| below-T_c start a:−0.3→−1   | 75 / 98 / 67             | 1/6    | 82     | 0.749    |
| **Landau a:+0.5→−1 (thru T_c)** | **79 / 95 / 65**     | **2/6**| 83     | 0.754    |

## Conclusion — the phase-transition insight is validated, modestly

Starting above `T_c` and cooling through the pitchfork beats starting
already-bistable: mean **75→79%**, solved seeds **1/6→2/6**. Cooling from the
monostable phase (where the equilibrium is uniquely input-determined) and freezing
that state in does select more correct basins — the predicted mechanism works,
directionally.

The predicted caveat also appears: the **floor does not move** (67→65). The gain is
in the mean and in more seeds crossing the solve threshold, not the worst case —
because the XOR-hard corner `(+1,+1)→−1` sits near the pitchfork singularity (its
symmetry-breaking field ≈ 0), so cooling cannot reliably choose its well. Separable
patterns benefit from cooling; the cusp-singular corner remains the bottleneck.

## Meta-finding: a consistent plateau across all mechanisms

Every principled mechanism tried on the N=5 / 2-hidden bistable Duffing lands in
the same band:

| mechanism | mean | solved | best seed |
|-----------|-----:|-------:|----------:|
| plain basin-averaging (Minit=20)      | 76% | —   | 100% |
| + annealing (Minit=50)                | 84% | 3/6 | ~100% |
| + symmetric gradient                  | 84% | 3/6 | ~100% |
| + learnable potential shape           | 80% | 0/5 | 88% |
| Landau anneal (cool through T_c)       | 79% | 2/6 | 95% |

Five distinct, principled approaches all plateau at ~79–84% mean, 2–3 of 6 seeds
fully solved, best seed ~100%. The convergence is the signal: at this minimal size
the residual limit is **capacity**, not the specific training trick. The best seeds
*do* fully solve (existence proof), so it's possible; the tiny network just lacks
the room to make *every* random-weight seed's basins agree for all four patterns
including the cusp-singular corner.

## Where the Duffing investigation stands

"Can the bistable Duffing be addressed?" — **yes**: from chance (51% under the
honest full-range test) to ~80–84% mean with best seeds fully solving, via the
standard EP multistability remedy (basin-averaging), optionally refined by
annealing (Landau, cool through T_c), a symmetric gradient, or a learnable well
shape. The remaining gap to an all-seed solve is capacity/architecture. The clean
next lever is a **larger / layered network** (which lifted XY); the faithful
"smooth input-following" endpoint remains the 2-D phase oscillator — i.e. XY.
