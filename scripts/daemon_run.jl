# daemon_run.jl
# Sends a script to the running Julia daemon (started with daemon_start.jl).
# Usage:
#   julia --project=. scripts/daemon_run.jl <script.jl> [args...]
#
# Example:
#   julia --project=. scripts/daemon_run.jl src/models/Wang-Kuramoto.jl
#   julia --project=. scripts/daemon_run.jl src/models/Wang-Kuramoto.jl --nd

using DaemonMode
runargs(3000)
