# daemon_start.jl
# Starts a persistent Julia kernel via DaemonMode.jl.
# Run once in a terminal:
#   julia --project=. scripts/daemon_start.jl
#
# Then send any script to it from another terminal:
#   julia --project=. scripts/daemon_run.jl src/models/Wang-Kuramoto.jl
#
# The kernel keeps all packages loaded between runs — no cold start.

using DaemonMode
println("Julia daemon listening on port 3000  (Ctrl-C to stop)")
serve(3000; shared=true)
