# Shared evaluation protocol for the classification experiments.
#
# The digit/MNIST scripts in scripts/ originally each (a) ran one seed and
# (b) picked their reported checkpoint by scoring the TEST set every EVAL_EVERY
# iterations and keeping the maximum. That is selection on the test set, and the
# manuscript states that every reported accuracy is a mean over random seeds.
# This module holds the three pieces needed to do it properly, so that every
# script does it the same way:
#
#   * `class_split`  -- a stratified train / validation / test split, with the
#     validation part carved out of the training partition (the approach of
#     src/hybrid/readout_ablation.py, VALIDATION_FRACTION = 0.2). The checkpoint
#     is selected on validation; the test partition is evaluated once, at the
#     end, on checkpoints fixed in advance.
#   * `msfmt` / `mean_std` -- mean +/- std formatting over seeds.
#   * `write_seed_record` -- a JSON record of the per-seed numbers, so a result
#     table can be regenerated without rerunning the experiment.
#
# Usage:
#   include(joinpath(@__DIR__, "..", "src", "utils", "eval_protocol.jl"))
#   using .EvalProtocol
module EvalProtocol

using Random, Statistics, Printf, JSON

export class_split, msfmt, mean_std, write_seed_record, frozen_init

"""
    class_split(y, classes, n_train_pc, n_test_pc; val_frac=0.2, seed=1)

Stratified per-class split into `(train, validation, test)` index vectors.

`n_train_pc` images per class form the training partition and the next
`n_test_pc` the test partition (the convention of Wang et al. 2024, which the
digit scripts follow). A stratified `val_frac` of the *training* partition is
then held out for model selection, so training sees `(1 - val_frac) *
n_train_pc` images per class. The seed moves the split, so that averaging over
seeds averages over splits as well as over initialisations.
"""
function class_split(y, classes, n_train_pc::Int, n_test_pc::Int;
                     val_frac::Float64=0.2, seed::Int=1)
    rng = MersenneTwister(1000 + seed)
    n_val = max(1, round(Int, val_frac * n_train_pc))
    train = Int[]; validation = Int[]; test = Int[]
    for c in classes
        idx = shuffle(rng, findall(==(c), y))
        length(idx) >= n_train_pc + n_test_pc ||
            error("class $c has only $(length(idx)) samples, need $(n_train_pc + n_test_pc)")
        append!(validation, idx[1:n_val])
        append!(train, idx[n_val+1:n_train_pc])
        append!(test, idx[n_train_pc+1:n_train_pc+n_test_pc])
    end
    return train, validation, test
end

"""
    frozen_init(seed, n, d; scale=1.0, uniform=false)

One fixed draw of free-cell initial conditions per (seed, split), reused at
every checkpoint so that a validation curve tracks the weights rather than the
initialisation noise. `uniform=true` draws from `scale * [-pi, pi)` (phase
networks), otherwise from `scale * randn` (position-encoded networks).
"""
frozen_init(seed::Int, n::Int, d::Int; scale::Float64=1.0, uniform::Bool=false) =
    uniform ? scale .* (2π .* rand(MersenneTwister(seed), n, d) .- π) :
              scale .* randn(MersenneTwister(seed), n, d)

"""mean and (corrected) std of a vector; std is NaN for a single seed."""
mean_std(v) = (mean(v), length(v) > 1 ? std(v) : NaN)

"""Format a vector of per-seed values as "mean +/- std" (or just the value)."""
msfmt(v; digits::Int=3) =
    length(v) > 1 ? Printf.format(Printf.Format("%.$(digits)f +/- %.$(digits)f"),
                                  mean(v), std(v)) :
                    Printf.format(Printf.Format("%.$(digits)f"), only(v))

"""
    write_seed_record(path, meta, results)

Write `meta` (configuration) plus the per-seed `results` (a vector of
NamedTuples) to `path` as JSON, alongside mean/std of every numeric field.
"""
function write_seed_record(path, meta::Dict, results::Vector)
    isempty(results) && error("no results to write")
    numeric = [k for k in keys(results[1]) if results[1][k] isa Number]
    summary = Dict{String,Any}()
    for k in numeric
        v = [r[k] for r in results]
        m, s = mean_std(v)
        summary[string(k, "_mean")] = m
        summary[string(k, "_std")] = s
    end
    payload = merge(Dict{String,Any}(string(k) => v for (k, v) in meta),
                    Dict("per_seed" => [Dict(string(k) => r[k] for k in keys(r))
                                        for r in results],
                         "summary" => summary))
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, payload, 2)
    end
    return path
end

end # module
