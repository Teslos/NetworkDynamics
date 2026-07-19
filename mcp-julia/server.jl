# Persistent Julia eval server for the MCP-Julia bridge.
#
# Loads the reservoir-computing project's heavy packages once, then serves
# arbitrary code evaluations over a local TCP socket for the lifetime of the
# process. State (variables, `using`d packages) persists in Main across calls,
# exactly like a REPL. See mcp-julia/index.js for the Node side that spawns
# and talks to this process.
#
# Wire protocol: each message (both directions) is a 4-byte big-endian UInt32
# length prefix followed by that many bytes of UTF-8 JSON.

using Sockets, JSON

const PORT = parse(Int, get(ENV, "JULIA_MCP_PORT", "27182"))
const START_TIME = time()

println("Loading packages (first run may take several minutes)...")
flush(stdout)
@time begin
    using LinearAlgebra, Random, Statistics, Printf
    using Graphs
    using OrdinaryDiffEq
    using Flux
    using Zygote
    using CairoMakie
    using Interpolations
    using CSV, DataFrames, Distributions
    using MLUtils, MLDataUtils, MLDatasets
    using Lux
    using ComponentArrays, StableRNGs, JLD2
    using Optimization, OptimizationOptimisers, OptimizationOptimJL
end

const BASELINE_NAMES = Set(names(Main; all = true))

function capture_eval(code::String)
    result = nothing
    ok = true
    err_str = ""
    t0 = time()
    old_stdout = stdout
    rd, wr = redirect_stdout()
    reader = @async read(rd, String)
    try
        result = Base.include_string(Main, code, "mcp_eval")
    catch e
        ok = false
        err_str = sprint(showerror, e, catch_backtrace())
    finally
        redirect_stdout(old_stdout)
        close(wr)
    end
    captured_stdout = fetch(reader)
    close(rd)
    result_str = nothing
    result_type = "Nothing"
    if ok && result !== nothing
        result_type = string(typeof(result))
        try
            result_str = sprint(
                (io, v) -> show(IOContext(io, :limit => true, :displaysize => (40, 120)),
                                MIME("text/plain"), v),
                result,
            )
        catch
            result_str = "<unshowable $(result_type)>"
        end
    end
    (; ok, result = result_str, result_type, stdout = captured_stdout, error = err_str,
       elapsed_ms = round((time() - t0) * 1000, digits = 1))
end

function dispatch(req)
    id = get(req, "id", nothing)
    action = req["action"]
    if action == "eval"
        r = capture_eval(req["code"])
        Dict("id" => id, "ok" => r.ok, "result" => r.result, "result_type" => r.result_type,
             "stdout" => r.stdout, "error" => r.error, "elapsed_ms" => r.elapsed_ms)
    elseif action == "ping"
        nvars = length(setdiff(Set(names(Main; all = true)), BASELINE_NAMES))
        Dict("id" => id, "ok" => true, "pid" => getpid(),
             "uptime_s" => round(time() - START_TIME, digits = 1), "nvars" => nvars)
    else
        Dict("id" => id, "ok" => false, "error" => "unknown action: $action")
    end
end

function handle_client(sock)
    try
        while isopen(sock)
            len_bytes = read(sock, 4)
            isempty(len_bytes) && break
            len = ntoh(reinterpret(UInt32, len_bytes)[1])
            req = JSON.parse(String(read(sock, len)))
            resp_bytes = Vector{UInt8}(JSON.json(dispatch(req)))
            write(sock, hton(UInt32(length(resp_bytes))))
            write(sock, resp_bytes)
            flush(sock)
        end
    catch e
        @error "client handler error" exception = (e, catch_backtrace())
    finally
        close(sock)
    end
end

server_sock = listen(IPv4("127.0.0.1"), PORT)
println("MCP_JULIA_READY port=$PORT")
flush(stdout)

while true
    sock = accept(server_sock)
    @async handle_client(sock)
end
