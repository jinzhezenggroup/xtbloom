/* xtbloom web demo worker.
 *
 * The target-width module (xtbloom + web adapter + preloaded Eigen LAPACKE
 * side module) runs here, so the long synchronous calls (single-point and,
 * especially, the multi-iteration geometry optimization) never block the UI
 * thread.
 *
 * Protocol (all messages JSON-ish, transfer lists for binary payloads):
 *   main -> worker {type:"init", wasmBinary, dataBinary, moduleUrl, helpersUrl}
 *   worker -> main {type:"ready", version: string} | {type:"error", error}
 *   main -> worker {type:"call", id, cmd: "compute"|"optimize", args: [...]}
 *   worker -> main {type:"result", id, ok, raw?: string, error?: string}
 */
let Module = null;
let stepFn = null;
let onStep = null;
let copyFloat64FromMemory = null;
let initializeDownloadedEngineModule = null;

function withPhase(error, phase) {
  const wrapped = error instanceof Error ? error : new Error(String(error || "init failed"));
  wrapped.phase = phase;
  return wrapped;
}

function serializedError(error) {
  const phase = String(error?.phase || "engine-initialize");
  const name = String(error?.name || "Error");
  return {
    name,
    message: String(error?.message || error || "init failed"),
    phase,
    // Dynamic imports are the only remaining startup network operations in
    // the Worker. Payload/module validation and WebAssembly linking are
    // deterministic and should not consume all retry attempts.
    retryable: (phase === "module-import" || phase === "helpers-import") && name === "TypeError",
  };
}

function ensureStepCallback() {
  if (stepFn !== null || !Module || typeof Module.addFunction !== "function") return;
  try {
    stepFn = Module.addFunction((iter, natoms, ptr, energy, fmax) => {
      // wasm32 passes a Number and wasm64 passes a BigInt for this pointer.
      // Copy the frame immediately so it stays valid after memory growth.
      const mem = Module.wasmMemory || Module.memory;
      const coords = copyFloat64FromMemory(mem, ptr, natoms * 3);
      if (onStep) onStep(iter, natoms, coords, energy, fmax);
    }, "viipdd");
    Module.ccall("xtbloom_web_set_optimize_step_cb", "void", ["pointer"], [stepFn]);
  } catch (err) {
    stepFn = null; /* animation is optional; core optimize still works */
  }
}

self.onmessage = async (event) => {
  const msg = event.data;

  if (msg.type === "init") {
    try {
      if (!(msg.wasmBinary instanceof Uint8Array) || msg.wasmBinary.byteLength === 0) {
        throw withPhase(new TypeError("missing wasm payload"), "payload-validation");
      }
      if (!(msg.dataBinary instanceof Uint8Array) || msg.dataBinary.byteLength === 0) {
        throw withPhase(new TypeError("missing data payload"), "payload-validation");
      }

      let moduleNamespace;
      try {
        moduleNamespace = await import(msg.moduleUrl);
      } catch (error) {
        throw withPhase(error, "module-import");
      }
      let helpers;
      try {
        helpers = await import(msg.helpersUrl);
      } catch (error) {
        throw withPhase(error, "helpers-import");
      }
      const createXTBloomModule = moduleNamespace?.default;
      copyFloat64FromMemory = helpers?.copyFloat64FromMemory;
      initializeDownloadedEngineModule = helpers?.initializeDownloadedEngineModule;
      if (
        typeof createXTBloomModule !== "function" ||
        typeof copyFloat64FromMemory !== "function" ||
        typeof initializeDownloadedEngineModule !== "function"
      ) {
        throw withPhase(new TypeError("invalid engine module exports"), "module-validation");
      }

      // Both binary payloads were counted by the UI loader. Supplying the data
      // package here prevents Emscripten from issuing an invisible second fetch
      // before it can load the Eigen LAPACKE side module from the virtual
      // filesystem.
      Module = await initializeDownloadedEngineModule(
        createXTBloomModule,
        msg.wasmBinary,
        msg.dataBinary,
      );
      const version = Module.ccall("xtbloom_web_version", "string", [], []);
      self.postMessage({ type: "ready", version });
    } catch (err) {
      self.postMessage({
        type: "error",
        error: serializedError(err),
      });
    }
    return;
  }

  if (msg.type === "call") {
    if (!Module) {
      self.postMessage({
        type: "result",
        id: msg.id,
        ok: false,
        error: "engine not ready",
      });
      return;
    }
    try {
      let raw;
      if (msg.cmd === "compute") {
        raw = Module.ccall(
          "xtbloom_web_compute", "string",
          ["string", "number", "number", "number", "number", "number", "number", "number"],
          msg.args,
        );
      } else if (msg.cmd === "optimize") {
        onStep = (iter, natoms, coords, energy, fmax) => {
          self.postMessage({ type: "step", id: msg.id, iter, natoms, coords, energy, fmax });
        };
        ensureStepCallback();
        try {
          raw = Module.ccall(
            "xtbloom_web_optimize", "string",
            ["string", "number", "number", "number", "number", "number", "number", "number", "number", "number"],
            msg.args,
          );
        } finally {
          onStep = null;
        }
      } else {
        throw new Error("unknown command: " + msg.cmd);
      }
      self.postMessage({ type: "result", id: msg.id, ok: true, raw });
    } catch (err) {
      self.postMessage({
        type: "result",
        id: msg.id,
        ok: false,
        error: String((err && (err.message || err)) || "call failed"),
      });
    }
    return;
  }
};
