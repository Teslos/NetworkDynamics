# Copilot Instructions — NetworkDynamics

## Project Overview

This is a **Julia research project** simulating nonlinear coupled oscillator networks for reservoir computing and classification tasks. The core idea: networks of FitzHugh-Nagumo (FHN), Duffing, or Kuramoto oscillators are operated near the edge of chaos and used as feature extractors for ML classification (MNIST, DryBean). The primary library is [`NetworkDynamics.jl`](https://github.com/pik-copan/NetworkDynamics.jl), which assembles graph-based ODE systems.

---

## Commands

### Run all tests
```
julia --project=. tests/runtests.jl
```

### Run a single test file
```julia
# From REPL with project activated
include("tests/test_graph_utils.jl")
include("tests/test_ode_models.jl")
include("tests/test_spikerate.jl")
```

### Run a single @testset by name
```
julia --project=. -e 'using Test; include("tests/runtests.jl")' 
# Or filter with: @testset with matching name in the file directly
```

### Generate figures
```
julia --project=. scripts/generate_figures.jl
```
Feature flags at the top of `generate_figures.jl` control what is generated (`GEN_FHN_DYNAMICS`, `GEN_NETWORK_TOPOLOGY`, etc.). Heavy training runs are disabled by default.

### Julia version
The project uses **Julia 1.12**. The manifest was resolved for Julia 1.12. Use `julia --project=.` from the repo root.

---

## Architecture

### Directory layout
```
src/
  models/        # ODE model definitions (FHN, Duffing, Kuramoto, ChaosFHN)
  classification/ # Reservoir computing classification pipelines
  networks/       # Graph creation, topology utilities
  utils/          # Spike rate encoding, dataset loaders
scripts/          # generate_figures.jl (figure generation entry point)
tests/            # Test suite (runtests.jl + test_*.jl)
results/          # Output: figures/, confusion_matrices/
data/             # Input datasets
```

### How files connect
- **No central module file.** Each script/model includes what it needs via `include("../utils/spikerate.jl")` etc.
- The **explicitly defined modules** are: `ChaosFHN` (chaos analysis), `graph_utils` (graph generators), `spikerate` (spike encoding).
- Classification scripts in `src/classification/` are self-contained pipelines that include models + utilities directly.
- `scripts/generate_figures.jl` is the only top-level runner; it `include`s model files and calls their functions.

### Data flow for reservoir computing
1. **Input encoding** (`src/utils/spikerate.jl`): raw signal → Poisson spike train rates → spline interpolation → time-varying input current `g(t)` per node
2. **ODE simulation** (`NetworkDynamics.jl`): `ODEProblem` on the network, solved with `Tsit5()` or `RK4()`
3. **Readout**: final or sampled state vector fed to linear classifier (Ridge regression or logistic)

---

## Key Conventions

### NetworkDynamics vertex/edge function signatures
Always `(output, state, edges_or_neighbors, p, t)` with `nothing` return and `@inline Base.@propagate_inbounds`:

```julia
@inline Base.@propagate_inbounds function fhn_vertex!(dv, v, edges, p, t)
    dv[1] = v[1] - v[1]^3 / 3 - v[2]
    dv[2] = ϵ * (v[1] - a)
    for e in edges[1]; dv[1] -= e[1]; end  # incoming
    for e in edges[2]; dv[1] += e[1]; end  # outgoing
    nothing
end

@inline Base.@propagate_inbounds function fhn_edge!(e, v_s, v_d, p, t)
    e[1] = σ * (v_s[1] - v_d[1])  # diffusive coupling
    nothing
end
```

### Parameter passing to ODE
- **The `p` argument is the entire parameter vector passed to `ODEProblem`** — it is NOT automatically split per-component by NetworkDynamics (v0.6.x).
- Coupling constants accessed inside edge/vertex functions should be **global `const` values**, not taken from `p` unless carefully indexed.
- When edge functions don't need parameters: pass `nothing` to `ODEProblem` — e.g. `ODEProblem(nd, x0, tspan, nothing)`.
- Typical compound parameter tuple: `p = (g_input_splines, σ * edge_weights)` where index 1 is vertex params, index 2 is edge weights.

### Graph construction
Use `graph_utils` module functions (not raw Graphs.jl calls in scripts):
```julia
include("src/networks/graph_utils.jl")
g = create_barabasi_albert_graph(N, k)
g = create_complete_graph(N)
g = create_erdos_renyi_graph(N, p)
g = create_watts_strogatz_graph(N, k, β)
```
All return `SimpleDiGraph` (directed), as required by NetworkDynamics.

### ODEVertex / StaticEdge registration
```julia
vertex = ODEVertex(f=my_vertex!, dim=2, sym=[:u, :v])
edge   = StaticEdge(f=my_edge!, dim=1, coupling=:directed)
nd     = network_dynamics(vertex, edge, g)
```

### Paths inside scripts
Use `joinpath(@__DIR__, "..")` to construct paths relative to the script — do **not** use `@__DIR__ * "/"` (parse error in Julia 1.12).

### Julia 1.12 compatibility notes
- `ConstructionBase` must be ≥ v1.6.0 (fixes `@generated` function issues with `SimpleNonlinearSolve`)
- `CairoMakie` / `Makie` precompile workloads have issues in v0.13.x on Julia 1.12; `Figure(size=...)` is correct (not `resolution=`)
- `const` bindings must be defined at top-level before any function references them

### Output conventions
- Figures saved to `results/figures/` as `.png`
- Confusion matrices saved to `results/confusion_matrices/`
- Trained model weights saved to `results/` as `.bson` or `.jld2`
