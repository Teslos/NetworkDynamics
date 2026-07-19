import { spawn } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

// ---- paths (resolved relative to this file, not process.cwd()) -----------

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.dirname(HERE);
const SERVER_JL_PATH = path.join(HERE, "server.jl");

const PROJECT_DIR = path.resolve(
  REPO_ROOT,
  process.env.JULIA_MCP_PROJECT_DIR || "Reservoir-Computing-in-Julia",
);
const CWD = path.resolve(REPO_ROOT, process.env.JULIA_MCP_CWD || PROJECT_DIR);
const PORT = parseInt(process.env.JULIA_MCP_PORT || "27182", 10);

// On Windows, "julia" resolved via PATH commonly hits the WindowsApps
// app-execution-alias shim (C:\...\WindowsApps\julia.exe) before a real
// juliaup/Julia-X.Y.Z install. That shim launches the real interpreter as an
// unrelated child process outside our process tree, so killing the PID we
// spawned does not kill the actual interpreter — it gets orphaned. Prefer a
// non-WindowsApps julia.exe on PATH so `.kill()` actually terminates it.
function resolveJuliaBin() {
  if (process.env.JULIA_MCP_JULIA_BIN) return process.env.JULIA_MCP_JULIA_BIN;
  if (process.platform === "win32") {
    const exeName = "julia.exe";
    for (const dir of (process.env.PATH || "").split(path.delimiter)) {
      if (!dir || /WindowsApps/i.test(dir)) continue;
      const candidate = path.join(dir, exeName);
      if (fs.existsSync(candidate)) return candidate;
    }
  }
  return "julia";
}
const JULIA_BIN = resolveJuliaBin();
const READY_TIMEOUT_MS = parseInt(process.env.JULIA_MCP_READY_TIMEOUT_MS || "600000", 10);
const EVAL_TIMEOUT_MS = parseInt(process.env.JULIA_MCP_EVAL_TIMEOUT_MS || "120000", 10);
const STDOUT_LIMIT = parseInt(process.env.JULIA_MCP_STDOUT_LIMIT || "4000", 10);
const RESULT_LIMIT = parseInt(process.env.JULIA_MCP_RESULT_LIMIT || "2000", 10);
const ERROR_LIMIT = 8000;

// ---- deferred helper -------------------------------------------------------

function createDeferred() {
  let resolve, reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

// ---- Julia child process + socket state -----------------------------------

let state = "starting"; // "starting" | "ready" | "crashed"
let juliaProcess = null;
let socket = null;
let recvBuf = Buffer.alloc(0);
let nextId = 1;
const pendingResolvers = new Map();
let readyDeferred = null;
let lastCrashMessage = "";
let startingSince = Date.now();

function encodeFrame(obj) {
  const body = Buffer.from(JSON.stringify(obj), "utf8");
  const len = Buffer.alloc(4);
  len.writeUInt32BE(body.length, 0);
  return Buffer.concat([len, body]);
}

function onSocketData(chunk) {
  recvBuf = Buffer.concat([recvBuf, chunk]);
  while (recvBuf.length >= 4) {
    const len = recvBuf.readUInt32BE(0);
    if (recvBuf.length < 4 + len) break;
    const body = recvBuf.subarray(4, 4 + len);
    recvBuf = recvBuf.subarray(4 + len);
    let msg;
    try {
      msg = JSON.parse(body.toString("utf8"));
    } catch (e) {
      continue; // malformed frame, drop it
    }
    const pending = pendingResolvers.get(msg.id);
    if (pending) {
      pendingResolvers.delete(msg.id);
      clearTimeout(pending.timer);
      pending.resolve(msg);
    }
  }
}

function connectSocket() {
  return new Promise((resolve, reject) => {
    let attempts = 0;
    const tryConnect = () => {
      attempts += 1;
      const s = net.connect(PORT, "127.0.0.1");
      s.once("connect", () => {
        socket = s;
        recvBuf = Buffer.alloc(0);
        socket.on("data", onSocketData);
        socket.on("error", () => {});
        resolve(socket);
      });
      s.once("error", () => {
        s.destroy();
        if (attempts >= 10) {
          reject(new Error("could not connect to julia server socket"));
        } else {
          setTimeout(tryConnect, 200);
        }
      });
    };
    tryConnect();
  });
}

// Julia's own package precompilation spawns further julia.exe worker
// processes as grandchildren of the process we spawn. On Windows, chaining
// anonymous pipes across that many process generations (MCP client -> this
// process -> julia -> julia's precompile workers) has been observed to
// stall indefinitely (the child never seems to make progress past package
// loading, even though the exact same script runs in seconds standalone).
// Redirecting the julia child's stdout/stderr to plain log files instead of
// pipes sidesteps the multi-generation handle-inheritance chain entirely;
// we detect readiness by polling the log file's contents instead.
const LOG_DIR = path.join(HERE, ".logs");
fs.mkdirSync(LOG_DIR, { recursive: true });
const STDOUT_LOG = path.join(LOG_DIR, "julia-stdout.log");
const STDERR_LOG = path.join(LOG_DIR, "julia-stderr.log");
const READY_POLL_MS = 300;

function tailFile(filePath, maxChars) {
  try {
    const text = fs.readFileSync(filePath, "utf8");
    return text.length > maxChars ? text.slice(-maxChars) : text;
  } catch {
    return "";
  }
}

function spawnJulia() {
  state = "starting";
  readyDeferred = createDeferred();
  startingSince = Date.now();

  const stdoutFd = fs.openSync(STDOUT_LOG, "w");
  const stderrFd = fs.openSync(STDERR_LOG, "w");

  const child = spawn(
    JULIA_BIN,
    [`--project=${PROJECT_DIR}`, "--startup-file=no", SERVER_JL_PATH],
    {
      cwd: CWD,
      env: { ...process.env, JULIA_MCP_PORT: String(PORT) },
      stdio: ["ignore", stdoutFd, stderrFd],
      // IMPORTANT: never shell:true — on Windows that makes kill() only
      // terminate the cmd.exe wrapper, orphaning julia.exe.
    },
  );
  juliaProcess = child;

  let readyHandled = false;
  let pollTimer = null;

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    try {
      fs.closeSync(stdoutFd);
    } catch {}
    try {
      fs.closeSync(stderrFd);
    } catch {}
  }

  pollTimer = setInterval(() => {
    if (readyHandled) return;
    const out = tailFile(STDOUT_LOG, 8192);
    if (out.includes("MCP_JULIA_READY")) {
      readyHandled = true;
      clearInterval(pollTimer);
      pollTimer = null;
      connectSocket()
        .then(() => {
          state = "ready";
          readyDeferred.resolve();
        })
        .catch((e) => {
          state = "crashed";
          lastCrashMessage = e.message;
          readyDeferred.reject(e);
        });
    }
  }, READY_POLL_MS);

  child.on("exit", (code, signal) => {
    stopPolling();
    // A restart may have already superseded this child with a new one before
    // this (async) exit event arrives — don't let a stale event stomp the
    // new child's state.
    if (juliaProcess !== child) return;
    state = "crashed";
    lastCrashMessage = `julia process exited (code=${code}, signal=${signal}). stderr tail:\n${tailFile(STDERR_LOG, 2000)}`;
    socket = null;
    if (!readyHandled) {
      readyDeferred.reject(new Error(lastCrashMessage));
    }
  });
  child.on("error", (e) => {
    stopPolling();
    if (juliaProcess !== child) return;
    state = "crashed";
    lastCrashMessage = `failed to spawn julia: ${e.message}`;
    if (!readyHandled) {
      readyDeferred.reject(e);
    }
  });

  const timer = setTimeout(() => {
    if (state === "starting") {
      lastCrashMessage = "timed out waiting for julia to become ready";
      readyDeferred.reject(new Error(lastCrashMessage));
    }
  }, READY_TIMEOUT_MS);
  readyDeferred.promise.finally(() => clearTimeout(timer)).catch(() => {});

  return readyDeferred.promise;
}

function sendRequest(action, extra, timeoutMs) {
  return new Promise((resolve, reject) => {
    if (!socket) {
      reject(new Error("julia socket not connected"));
      return;
    }
    const id = nextId++;
    const timer = setTimeout(() => {
      pendingResolvers.delete(id);
      reject(new Error(`request timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    pendingResolvers.set(id, { resolve, timer });
    socket.write(encodeFrame({ id, action, ...extra }));
  });
}

// Serialize evals: only one in flight at a time.
let queue = Promise.resolve();
function enqueue(fn) {
  const run = queue.then(fn, fn);
  queue = run.then(
    () => {},
    () => {},
  );
  return run;
}

// ---- output formatting ------------------------------------------------------

function truncateHeadTail(text, limit) {
  if (text.length <= limit) return text;
  const headLen = Math.round(limit * 0.85);
  const tailLen = limit - headLen;
  const omitted = text.length - headLen - tailLen;
  return `${text.slice(0, headLen)}\n...[truncated, ${omitted} chars omitted]...\n${text.slice(text.length - tailLen)}`;
}

function formatEvalResult(r) {
  const lines = [];
  lines.push(
    `[${r.ok ? "ok" : "error"}] elapsed=${r.elapsed_ms}ms  type=${r.result_type}`,
  );
  if (r.stdout) {
    lines.push("stdout:");
    lines.push(truncateHeadTail(r.stdout, STDOUT_LIMIT));
  }
  if (r.ok) {
    if (r.result !== null && r.result !== undefined) {
      lines.push("result:");
      const truncated = truncateHeadTail(r.result, RESULT_LIMIT);
      lines.push(truncated);
      if (truncated.length !== r.result.length) {
        lines.push(
          "(truncated — use size/summary/first(x,5) to inspect further, or save(...) to a file and Read it)",
        );
      }
    }
  } else {
    lines.push("error:");
    lines.push(truncateHeadTail(r.error, ERROR_LIMIT));
  }
  return lines.join("\n");
}

// ---- MCP server + tools -----------------------------------------------------

const server = new McpServer({ name: "julia-reservoir", version: "1.0.0" });

server.registerTool(
  "julia_eval",
  {
    title: "Julia Eval",
    description:
      "Evaluate Julia code in a persistent, pre-warmed Julia session for the " +
      "reservoir-computing project (packages like CairoMakie/Flux/OrdinaryDiffEq/Lux " +
      "are already loaded). Variables and `using` statements persist across calls, " +
      "like a REPL. Returns stdout and the value of the last top-level expression.",
    inputSchema: { code: z.string().describe("Julia source code to evaluate") },
  },
  async ({ code }) => {
    if (state === "starting") {
      // Never block here: cold package-compile can take minutes, well past
      // typical MCP client-side tool-call timeouts. Fail fast with a status
      // the caller (Claude) can act on by retrying shortly instead of
      // risking the whole request timing out mid-wait.
      const elapsedS = Math.round((Date.now() - startingSince) / 1000);
      return {
        isError: true,
        content: [
          {
            type: "text",
            text: `Julia is still starting (loading packages, ${elapsedS}s elapsed). Retry julia_eval in a bit, or call julia_status to check progress.`,
          },
        ],
      };
    }
    if (state === "crashed") {
      return {
        isError: true,
        content: [
          {
            type: "text",
            text: `Julia session is not running (${lastCrashMessage}). Call julia_restart to recover.`,
          },
        ],
      };
    }
    try {
      const r = await enqueue(() => sendRequest("eval", { code }, EVAL_TIMEOUT_MS));
      return {
        isError: !r.ok,
        content: [{ type: "text", text: formatEvalResult(r) }],
      };
    } catch (e) {
      return { isError: true, content: [{ type: "text", text: `eval failed: ${e.message}` }] };
    }
  },
);

server.registerTool(
  "julia_status",
  {
    title: "Julia Status",
    description:
      "Check whether the persistent Julia session is starting up, ready, or crashed. " +
      "Does not block on startup.",
    inputSchema: {},
  },
  async () => {
    if (state === "starting") {
      const elapsedS = Math.round((Date.now() - startingSince) / 1000);
      return {
        content: [
          {
            type: "text",
            text: `starting: Julia is still loading packages (${elapsedS}s elapsed).`,
          },
        ],
      };
    }
    if (state === "crashed") {
      return {
        content: [
          {
            type: "text",
            text: `crashed: ${lastCrashMessage}`,
          },
        ],
      };
    }
    try {
      const r = await sendRequest("ping", {}, 5000);
      return {
        content: [
          {
            type: "text",
            text: `ready: pid=${r.pid} uptime=${r.uptime_s}s nvars=${r.nvars}`,
          },
        ],
      };
    } catch (e) {
      return {
        isError: true,
        content: [{ type: "text", text: `ping failed: ${e.message}` }],
      };
    }
  },
);

server.registerTool(
  "julia_restart",
  {
    title: "Julia Restart",
    description:
      "Kill and respawn the persistent Julia process. All session state (variables) " +
      "is lost. Use this if julia_eval is stuck or the session has crashed.",
    inputSchema: {},
  },
  async () => {
    if (juliaProcess) {
      try {
        juliaProcess.kill();
      } catch {}
    }
    if (socket) {
      try {
        socket.destroy();
      } catch {}
      socket = null;
    }
    // Don't await full readiness here — package loading can take minutes,
    // well past typical MCP client-side tool-call timeouts (same reasoning
    // as julia_eval above). Kick off the respawn and let the caller poll
    // julia_status / retry julia_eval.
    spawnJulia().catch(() => {});
    return {
      content: [
        {
          type: "text",
          text: "Julia restart initiated (all session state cleared). Call julia_status to check when it's ready.",
        },
      ],
    };
  },
);

// ---- cleanup ------------------------------------------------------------------

function shutdown() {
  try {
    juliaProcess?.kill();
  } catch {}
  process.exit(0);
}
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
process.on("exit", () => {
  try {
    juliaProcess?.kill();
  } catch {}
});

// ---- start ----------------------------------------------------------------

spawnJulia().catch(() => {}); // don't block MCP startup on Julia readiness

const transport = new StdioServerTransport();
await server.connect(transport);
