/* xtbloom web demo worker.
 *
 * The target-width module (xtbloom + web adapter + preloaded LAPACK side module)
 * runs here, so the long synchronous calls (single-point and, especially,
 * the multi-iteration geometry optimization) never block the UI thread.
 *
 * Protocol (all messages JSON-ish, transfer lists for binary payloads):
 *   main -> worker {type:"init", wasmBinary: Uint8Array}
 *   worker -> main {type:"ready", version: string} | {type:"error", error}
 *   main -> worker {type:"call", id, cmd: "compute"|"optimize", args: [...]}
 *   worker -> main {type:"result", id, ok, raw?: string, error?: string}
 */
import createXTBloomModule from "./xtbloom_web.js";
import { copyFloat64FromMemory } from "./app_helpers.js";

let Module = null;
let stepFn = null;
let onStep = null;

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
      // The main thread already downloaded the wasm with a progress bar;
      // pass those bytes in so the glue does not fetch it again. The small
      // .data payload (preloaded LAPACK side module in the virtual FS) is
      // fetched by the glue itself.
      Module = await createXTBloomModule({ wasmBinary: msg.wasmBinary });
      const version = Module.ccall("xtbloom_web_version", "string", [], []);
      self.postMessage({ type: "ready", version });
    } catch (err) {
      self.postMessage({
        type: "error",
        error: String((err && (err.message || err)) || "init failed"),
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
