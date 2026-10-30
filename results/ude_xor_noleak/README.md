# UDE XOR without the answer leak, and why EP does better

The UDE XOR results in the paper's first version came from
`src/models/Wang-Kuramoto.jl` passing each row of its XOR table — answer
column included — as the field phases of every oscillator. The output node got
`h = 0.5` at the XOR answer, so a network with zero coupling already scored
−2.746 against the −2.773 optimum at t = 5 and was exact at T = 200. Every N
"solved" XOR. The N = 5 version (commit 632483d) has the same leak.

Everything here is re-run with fields on the inputs only, a zero-coupling
control, evaluation on the settled state (Tsit5, reltol 1e-10, T = 200), and a
strict test: a pattern is correct when the output is nearer its own target than
the opposite one. "Within π/4" additionally requires all four outputs within
π/4 of target. Ten seeds per row.

## Network size (baseline: shared coupling NN, field on inputs only, loss t = 5–20)

`summary.txt`, `N{3,4,5}_complete_seed*.log`, `N8_ws_seed*.log`

| Network | Correct | Within π/4 |
|---|---|---|
| N = 3 all-to-all | 8/10 | 5/10 |
| N = 4 all-to-all | 7/10 | 1/10 |
| N = 5 all-to-all | 3/10 | 1/10 |
| N = 8 Watts–Strogatz | 2/10 | 0/10 |

## Against EP at N = 5 (inputs clamped)

Entries are correct / within π/4. Window = loss over t = 5–20; settled = loss
over t = 50–200 (`--tend=200`); implicit = `scripts/ude_xor_implicit.jl`.

| Coupling | Fields on | Window | Settled | Implicit |
|---|---|---|---|---|
| shared NN, 16 units, inputs not clamped | none | 3/1 | — | — |
| shared NN, 16 units, clamped | none | 2/0 | — | — |
| shared NN, 16 units | output | 4/2 | — | — |
| shared NN + edge identity | output | 5/4 | — | — |
| per-edge, directed | output | 2/2 | 8/4 | 3/1 |
| per-edge, symmetric | output | 0/0 | 2/0 | 0/0 |
| per-edge, directed | all free | 4/4 | 10/6 | 9/9 |
| per-edge, symmetric | all free | 5/3 | **10/10** | **10/10** |
| EP-trained XY (`ep_xy_multiseed.log`) | all free | 10/10 (≤ 0.04π, 20 random starts per pattern) | | |

Files: `summary_epcompare.txt` (clamp, clamp + output field),
`summary_coupling.txt` (edge identity, per-edge; the 64-unit arm was stopped
after 5 h of ForwardDiff without finishing, hence its FAILED lines),
`summary_tend.txt` (settled, output field), `summary_hidfields.txt` (window and
settled, all fields), `implicit_*.log` (implicit gradient; `allfields` = all free
oscillators). `summary_settle.txt` holds the five runs of a |dφ/dt|² settling
penalty that was stopped early: none solved XOR (best 3/4), consistent with the
penalty being satisfiable at an unstable rest point.

## What it shows

- With a field on the output only, the short window optimises a transient: the
  output passes through the target by t = 20 and then leaves it — sliding to
  another fixed point, or rotating for directed coupling (`diag_pair.jl`,
  `diag_sympair.jl`).
- Two ingredients close the gap to EP, and both are needed: a learned field on
  every free oscillator (as EP's XY network has), and a cost measured on the
  settled state. With both, the gradient method no longer matters.
- Capacity is not the limit: 27 per-edge weights reach near-optimal loss.

Runs were made on thl06710 (Julia 1.12.6) and thl06712 (`julia +release`,
1.12.5; its default 1.11 cannot load this manifest). The scripts there were
copies of `src/models/Wang-Kuramoto.jl` placed in this directory, which is why
the launch scripts call `results/ude_xor_noleak/Wang-Kuramoto.jl`.
