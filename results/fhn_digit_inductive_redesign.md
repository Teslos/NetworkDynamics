# FHN digit classifier: true inductive redesign

Branch: `fhn-digit-inductive-redesign`

This branch is an experimental, paper-oriented replacement for the historical
joint train/test simulation. It is intentionally separate so the published
baseline can be reproduced unchanged and compared with the redesign.

## What changed

1. `fit_fhn_digit_classifier(X_train, y_train)` integrates **only training
   oscillators** and caches their trajectories. No test values, test labels, or
   test population size enter fitting.
2. `predict_fhn_digits(model, X_query)` integrates each query as an independent
   two-state FHN oscillator against the immutable cached training trajectory.
   Queries cannot affect training states or one another, including indirectly
   through an adaptive solver's global error control.
3. One fixed query coupling vector and initial state are generated during fit
   and stored in the model. The classifier therefore applies the same fitted
   dynamical map to every sample. The readout is trained on training examples
   passed through that same virtual-query map, rather than on differently
   distributed internal reservoir-node states.
4. The default `row_total` coupling normalization gives every node total
   incoming weight `sigma`. Thus `sigma` keeps the same interpretation when the
   number of training samples changes. The historical per-edge scaling remains
   available as the explicit `per_edge` ablation.
5. Production solver tolerances and return codes are explicit, and regression
   tests call the production API. The tests require exact invariance to query
   batching and ordering.

The implementation lives in `src/baselines/fhn_digit_classifier.jl`; the
command-line entry point is `scripts/run_fhn_digits.jl`.

## Comparison protocol

Run the same seeds and split sizes for all arms. Record accuracy, macro-F1,
fit time, and query time.

```text
# On branch fhn-digits-seed-sweep: historical masked joint solve
julia --project=. scripts/run_fhn_digits.jl --mode fixed_reservoir --seed 1

# On this branch: isolate the fit/predict redesign while retaining old scaling
julia --project=. scripts/run_fhn_digits.jl --normalization per_edge --seed 1

# On this branch: complete redesign with size-independent total coupling
julia --project=. scripts/run_fhn_digits.jl --normalization row_total --seed 1
```

Use multiple seeds before changing manuscript headline numbers. The old
full-resolution results were generated under a pipeline that still contained
unseeded random sources, so they are useful context but not a paired benchmark.
This file deliberately contains no new full-resolution claim until those runs
have completed.

## Interpreting the arms

| Arm | Fit/predict boundary | Coupling convention | Purpose |
|---|---|---|---|
| historical `fixed_reservoir` | joint adaptive solve | per edge | current baseline |
| redesigned `per_edge` | separate cached solves | per edge | isolates numerical/API repair |
| redesigned `row_total` | separate cached solves | fixed total per node | recommended model |

The `per_edge` arm is an ablation, not the recommended production setting:
aggregate coupling grows roughly linearly with reservoir size, so changing the
train/test split or dataset size changes the dynamical regime.
