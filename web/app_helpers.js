/* Pure browser-engine helpers kept separate so units and readiness semantics
 * are executable under Node without constructing the full page DOM. */

export const BOHR_PER_ANGSTROM = 1.8897261254578281;

export function angstromToBohr(value) {
  return value * BOHR_PER_ANGSTROM;
}

/* Fetch exposes decoded response bytes, while Content-Length may describe a
 * compressed representation. Only compare streamed bytes with a declared
 * length when the response is explicitly identity encoded. */
export function comparableContentLength(headers) {
  const encoding = String(headers?.get?.("content-encoding") || "")
    .trim()
    .toLowerCase();
  if (encoding && encoding !== "identity") return 0;
  const total = Number(headers?.get?.("content-length"));
  return Number.isFinite(total) && total > 0 ? total : 0;
}

/* Keep every visible progress consumer on the same bounded value. This is a
 * final defense against proxies with inconsistent response metadata. */
export function clampProgressPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.min(100, Math.max(0, numeric));
}

/* Reserve 100% for a completed read. This avoids claiming completion early
 * when a server reports a slightly smaller identity length than the stream. */
export function downloadProgressPercent(loaded, total, previous = 0, complete = false) {
  if (complete) return 100;
  const prior = Math.min(99, clampProgressPercent(previous));
  if (!Number.isFinite(loaded) || loaded < 0 || !Number.isFinite(total) || total <= 0) {
    return prior;
  }
  const current = Math.min(99, clampProgressPercent((loaded / total) * 100));
  return Math.max(prior, current);
}

/* URLSearchParams performs the required percent decoding. Literal '+' in a
 * charged SMILES must therefore be encoded as %2B, as required by URL syntax. */
export function readSmilesQuery(url, maxLength = 2048) {
  const value = new URL(url).searchParams.get("smiles");
  if (value === null || value.trim() === "") return null;
  const trimmed = value.trim();
  if (trimmed.length > maxLength) {
    const error = new RangeError(`SMILES exceeds ${maxLength} characters`);
    error.code = "smiles_err_too_long";
    throw error;
  }
  return trimmed;
}

/* The URL workflow is intentionally one-shot and begins only after both
 * independent workers are idle and ready. Keeping the predicate pure makes
 * the race-prevention contract executable without constructing the page DOM. */
export function canStartUrlSmiles({
  smiles,
  started,
  engineState,
  smilesState,
  engineBusy,
  smilesBusy,
}) {
  return Boolean(smiles) && !started && engineState === "ready" &&
    smilesState === "ready" && !engineBusy && !smilesBusy;
}

/* Keep UI callers from ever dereferencing a null or partially initialized
 * Worker. postMessage can itself throw (for example after termination), so the
 * caller still owns cleanup of any request bookkeeping around this helper. */
export function postToReadyWorker(worker, engineState, message) {
  if (
    engineState !== "ready" ||
    worker === null ||
    typeof worker !== "object" ||
    typeof worker.postMessage !== "function"
  ) {
    throw new Error("engine not ready");
  }
  worker.postMessage(message);
}

/* Emscripten callback pointers are Numbers for wasm32 and BigInts for wasm64.
 * Normalize only safe, aligned offsets and copy immediately so a later memory
 * growth cannot invalidate the optimizer animation frame. */
export function copyFloat64FromMemory(memory, pointer, length) {
  if (!Number.isSafeInteger(length) || length < 0) {
    throw new RangeError("invalid Float64 element count");
  }
  let byteOffset;
  if (typeof pointer === "bigint") {
    if (pointer < 0n || pointer > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new RangeError("WASM pointer is outside JavaScript's safe integer range");
    }
    byteOffset = Number(pointer);
  } else {
    byteOffset = pointer;
  }
  if (!Number.isSafeInteger(byteOffset) || byteOffset < 0 || byteOffset % 8 !== 0) {
    throw new RangeError("WASM Float64 pointer is invalid or misaligned");
  }
  if (!memory || !(memory.buffer instanceof ArrayBuffer)) {
    throw new TypeError("WASM memory is unavailable");
  }
  const byteLength = length * 8;
  if (!Number.isSafeInteger(byteLength) || byteOffset > memory.buffer.byteLength - byteLength) {
    throw new RangeError("WASM Float64 slice is outside memory");
  }
  return Array.from(new Float64Array(memory.buffer, byteOffset, length));
}

/* Transfer the downloaded main module and resolve only after the worker has
 * instantiated Emscripten, fetched its .data payload, and loaded the LAPACK
 * side module. Runtime result/step messages continue through onMessage. */
export function initializeWorker(worker, wasmBinary, onMessage = () => {}) {
  return new Promise((resolve, reject) => {
    let ready = false;
    worker.onmessage = (event) => {
      const message = event.data;
      if (message.type === "ready") {
        ready = true;
        resolve(message);
      } else if (message.type === "error" && !ready) {
        reject(new Error(message.error || "worker initialization failed"));
      } else {
        onMessage(message);
      }
    };
    worker.onerror = (event) => {
      const error = new Error((event && event.message) || "worker error");
      if (ready) {
        onMessage({ type: "error", error: error.message });
      } else {
        reject(error);
      }
    };
    worker.postMessage({ type: "init", wasmBinary }, [wasmBinary.buffer]);
  });
}

/* Apply one timeout to the complete asynchronous startup chain. onTimeout is
 * responsible for aborting fetch and terminating any partially ready worker. */
export function withTimeout(promise, timeoutMs, onTimeout = () => {}) {
  let timer = null;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      onTimeout();
      reject(new Error("TIME_OUT"));
    }, timeoutMs);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}
