# The FHN digit classifier is transductive — and does not need to be

Scripts: `scripts/run_fhn_digits.jl` (now with `--mode`), ablations in the session
scratchpad. Runs: 2026-09-13.

## The problem

`run_fhn_digits.jl` — and the original it reproduces,
`src/classification/FitzHug-Nagumo-MNIST-Ridge.jl:107`, which builds
`create_complete_graph(1797)` — assigns **one sample per node** and couples all
of them in a single ODE solve, applying the train/test split only afterwards.
Consequences:

* No labels leak: labels enter only at the ridge/logistic readout, on train
  indices.
* But every sample's features depend on every other sample's drive, test samples
  included, so the representation is a function of the whole batch.
* And there is no map from a single new sample to features: classifying one digit
  requires re-simulating a network containing it. That is a transductive system,
  not a deployable classifier, and it was being compared against inductive
  baselines (logreg, MLP, ESN) without qualification.

## What the coupling is actually doing (N=300, 3 seeds)

| variant | test accuracy | inductive? |
|---|---:|---|
| transductive (as shipped, random ICs) | 0.894 ± 0.041 | no |
| transductive, identical ICs | 0.855 ± 0.008 | no |
| **fixed reservoir (directed)** | **0.849 ± 0.002** | **yes** |
| uncoupled (σ=0), identical ICs | 0.497 ± 0.076 | yes |
| uncoupled (σ=0), random ICs | 0.498 ± 0.057 | yes |
| train/test networks solved separately | 0.424 ± 0.092 | yes |
| raw-pixel logistic regression, same split | ≈0.933 | yes |

These variants were computed inside one process from a single spike encoding per
seed, so they are paired on the encoding; they were, however, run before the
readout initialisation was seeded, so each variant drew a different readout init.
The readout is a convex full-batch fit and the effects below are 30-40 points, so
this does not threaten the conclusions, but the comparison was not as controlled
as it should have been.

Two hypotheses were tested and one survived:

* **Initial conditions (rejected).** Every node starts at a random `z0`; the
  conjecture was that strong all-to-all coupling merely washes this out. Fixing
  the ICs does nothing for the uncoupled network (0.497 vs 0.498), so the
  coupling is not just cancelling a self-inflicted nuisance.
* **Common frame (supported).** Train and test features are only commensurable
  when measured against the same mean field and the same coupling draw. Solving
  the halves separately puts them in different coordinate systems and scores
  *worse* (0.424) than having no network at all (0.498).

So the network coupling carries the accuracy — 0.85 coupled against 0.50
uncoupled — which supports the paper's thesis that the dynamics compute. What it
does **not** need is the test set inside the network.

## The repair: the training batch is the reservoir

`--mode fixed_reservoir` zeroes the columns of the coupling matrix belonging to
test nodes, so **no node receives from a test node**. Each test node then evolves
under its own drive plus the training nodes only. This is not an approximation:
since train dynamics are independent of the test set, it is exactly equivalent to
inserting each test sample *on its own* into a reservoir made of the training
batch. No test sample influences the training representation or any other test
sample, so the system is inductive.

It is also cheap to deploy. The train trajectories do not depend on the query, so
they are computed once and cached; a new sample then costs a single-node solve
driven by its own input plus the cached field, not a 1797-node solve.

## Full resolution (N=1797, 3 seeds per mode)

| seed label | transductive | fixed reservoir |
|---|---:|---:|
| 1 | 0.9415 | 0.9387 |
| 2 | 0.9331 | 0.9359 |
| 3 | 0.9248 | 0.9164 |
| **mean** | **0.9331 ± 0.0084** | **0.9303 ± 0.0121** |

The inductive variant matches the transductive one within run-to-run noise. At
full resolution the transduction buys nothing measurable; the earlier N=300 gap
(0.855 vs 0.849) likewise sat inside the noise. Solve time ≈ 39–83 min per run.

**Correction (2026-09-14).** This table was first published as a *paired*
comparison with a per-seed difference of −0.28 ± 0.56 pp. It is not paired. The
spike encoder drew its Bernoulli spikes from the global RNG (see
`src/utils/spikerate.jl`), so each of these six runs — separate process
invocations — encoded its data differently, and rows sharing a seed label do not
share an encoding. The means above remain valid estimates of each mode, but the
per-seed differences are not pairs and the ±0.56 pp precision was unearned. The
encoder now takes an explicit `rng`, and a second unseeded source has been fixed
alongside it: Flux's `Dense` initialised the readout from the global RNG even
when `train_logreg` was handed a seed, so `rng=` was a no-op for every model in
`src/baselines/baseline_models.jl`. Two runs of the same command now agree
exactly. These numbers should be regenerated under the seeded pipeline before
being used anywhere load-bearing.

Caveat on the comparison: `fixed_reservoir` also changes the *training* nodes
slightly, because their row sums no longer include the test columns (~20% fewer
neighbours). It is therefore a different model that is inductive, not a pure
"delete the leakage" ablation. Given that the two agree to within 0.3 pp, the
distinction does not matter here, but a σ rescaled to match row sums would be the
clean control if it ever did.

## Recommendation

1. Report `fixed_reservoir` as the FHN digit result. The number is unchanged
   within noise (0.930 ± 0.012 vs the transductive 0.933 ± 0.008), and it is a
   deployable classifier that can honestly be compared with inductive baselines.
2. Describe what the reservoir *is*: the training population, with a query
   coupled into it — not a reservoir in the usual sense of fixed internal state
   driven by one sample at a time.
3. Keep the claim that the coupling does the computational work; it is now
   supported by a direct ablation (0.85 coupled vs ≈0.50 uncoupled) rather than
   assumed. Note that this ablation exists only at **N=300**. Because the
   coupling is not degree-normalised, the aggregate drive on a node grows with
   N, so N=300 and N=1797 are different dynamical regimes at the same σ=0.72;
   the uncoupled arm has not been measured at full resolution, and the claim
   should not be quoted as if it had.
4. Note for the physical-realizability argument: the directed coupling breaks the
   symmetry of `W`. A hardware implementation needs either directed coupling or a
   frozen reference population.
