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

/* Aggregate a concurrently downloaded resource set without inventing a byte
 * total. File completion remains exact even when a proxy compresses one asset
 * and therefore makes Content-Length incomparable with decoded fetch chunks. */
export function aggregateResourceProgress(resources) {
  const states = Array.from(resources || []);
  const totalFiles = states.length;
  let completedFiles = 0;
  let loadedBytes = 0;
  let totalBytes = 0;
  let byteTotalKnown = totalFiles > 0;

  for (const state of states) {
    const loaded = Number(state?.loaded);
    const total = Number(state?.total);
    if (state?.complete) completedFiles++;
    if (Number.isFinite(loaded) && loaded > 0) loadedBytes += loaded;
    if (Number.isFinite(total) && total > 0) {
      totalBytes += total;
    } else {
      byteTotalKnown = false;
    }
  }

  const complete = totalFiles > 0 && completedFiles === totalFiles;
  const percent = byteTotalKnown
    ? downloadProgressPercent(loadedBytes, totalBytes, 0, complete)
    : null;
  const barPercent = percent ?? downloadProgressPercent(
    completedFiles,
    totalFiles,
    0,
    complete,
  );
  return {
    totalFiles,
    completedFiles,
    loadedBytes,
    totalBytes: byteTotalKnown ? totalBytes : null,
    percent,
    barPercent,
    complete,
  };
}

/* The build manifest is the cache-coherency contract between app.js and the
 * generated Emscripten artifacts. Reject partial or path-injecting manifests
 * before any resource URL is constructed. */
export function validateEngineManifest(manifest, requiredIds = []) {
  if (!manifest || manifest.schema_version !== 1) {
    throw new TypeError("unsupported engine manifest schema");
  }
  if (!/^[0-9a-f]{64}$/i.test(String(manifest.version || ""))) {
    throw new TypeError("invalid engine manifest version");
  }
  if (!Array.isArray(manifest.assets) || manifest.assets.length === 0) {
    throw new TypeError("engine manifest has no assets");
  }
  const ids = new Set();
  const assets = manifest.assets.map((asset) => {
    const id = String(asset?.id || "");
    const path = String(asset?.path || "");
    const bytes = Number(asset?.bytes);
    const sha256 = String(asset?.sha256 || "");
    if (!/^[a-z][a-z0-9_-]*$/.test(id) || ids.has(id)) {
      throw new TypeError("engine manifest has an invalid or duplicate asset id");
    }
    if (!/^[A-Za-z0-9_.-]+$/.test(path) || path === "." || path === "..") {
      throw new TypeError(`engine manifest has an unsafe path for ${id}`);
    }
    if (!Number.isSafeInteger(bytes) || bytes <= 0) {
      throw new TypeError(`engine manifest has an invalid size for ${id}`);
    }
    if (!/^[0-9a-f]{64}$/i.test(sha256)) {
      throw new TypeError(`engine manifest has an invalid digest for ${id}`);
    }
    ids.add(id);
    return { id, path, bytes, sha256: sha256.toLowerCase() };
  });
  for (const id of requiredIds) {
    if (!ids.has(id)) throw new TypeError(`engine manifest is missing ${id}`);
  }
  return { schemaVersion: 1, version: String(manifest.version).toLowerCase(), assets };
}

function resourceLoadError(message, resource, extras = {}) {
  const error = new Error(message);
  error.name = "ResourceLoadError";
  error.resourceId = resource?.id || "";
  error.resourceUrl = resource?.url || "";
  Object.assign(error, extras);
  return error;
}

/* Fetch all engine resources as one generation. Headers are collected before
 * bodies are consumed so a trustworthy aggregate byte total is available from
 * the first streamed chunk. The returned bytes also let the caller hand large
 * wasm/data payloads to the Worker without a second network request. */
export async function fetchResourceBatch(
  resources,
  {
    signal,
    cache = "default",
    fetchImpl = globalThis.fetch,
    cryptoImpl = globalThis.crypto,
    onProgress = () => {},
  } = {},
) {
  const descriptors = Array.from(resources || []);
  if (!descriptors.length) throw new TypeError("resource batch is empty");
  if (typeof fetchImpl !== "function") throw new TypeError("fetch is unavailable");

  const opened = await Promise.all(descriptors.map(async (resource) => {
    let response;
    try {
      response = await fetchImpl(resource.url, { signal, cache });
    } catch (cause) {
      const error = resourceLoadError(
        `Network error while loading ${resource.id || resource.url}`,
        resource,
        { cause, retryable: true },
      );
      throw error;
    }
    if (!response.ok) {
      const status = Number(response.status) || 0;
      throw resourceLoadError(
        `HTTP ${status} while loading ${resource.id || resource.url}`,
        resource,
        { status, retryable: status === 408 || status === 425 || status === 429 || status >= 500 },
      );
    }
    return {
      resource,
      response,
      state: {
        id: resource.id,
        loaded: 0,
        total: Number.isSafeInteger(resource.bytes) && resource.bytes > 0
          ? resource.bytes
          : comparableContentLength(response.headers),
        complete: false,
      },
    };
  }));

  const states = opened.map((entry) => entry.state);
  onProgress(aggregateResourceProgress(states));
  const results = new Map();

  await Promise.all(opened.map(async ({ resource, response, state }) => {
    const chunks = [];
    const reader = response.body?.getReader?.();
    if (reader) {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!value || value.byteLength === 0) continue;
        chunks.push(value);
        state.loaded += value.byteLength;
        onProgress(aggregateResourceProgress(states));
      }
    } else {
      const value = new Uint8Array(await response.arrayBuffer());
      if (value.byteLength) chunks.push(value);
      state.loaded = value.byteLength;
      onProgress(aggregateResourceProgress(states));
    }

    if (state.total > 0 && state.loaded !== state.total) {
      throw resourceLoadError(
        `Incomplete response while loading ${resource.id || resource.url}`,
        resource,
        { expectedBytes: state.total, receivedBytes: state.loaded, retryable: true },
      );
    }
    const bytes = new Uint8Array(state.loaded);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
    if (resource.sha256) {
      if (!cryptoImpl?.subtle?.digest) {
        throw resourceLoadError(
          `SHA-256 verification is unavailable for ${resource.id || resource.url}`,
          resource,
          { retryable: false },
        );
      }
      const digest = new Uint8Array(await cryptoImpl.subtle.digest("SHA-256", bytes));
      const actual = Array.from(digest, (value) => value.toString(16).padStart(2, "0")).join("");
      if (actual !== String(resource.sha256).toLowerCase()) {
        throw resourceLoadError(
          `Digest mismatch while loading ${resource.id || resource.url}`,
          resource,
          { expectedSha256: resource.sha256, actualSha256: actual, retryable: true },
        );
      }
    }
    results.set(resource.id, bytes);
    state.complete = true;
    onProgress(aggregateResourceProgress(states));
  }));

  return results;
}

/* Safari commonly reports a transient module/Worker fetch as TypeError:
 * "Load failed". Do not retry deterministic syntax, security, or WebAssembly
 * validation failures, but allow bounded retries for network/status/timeout
 * failures and opaque pre-ready Worker load errors. */
export function isRetryableLoadError(error) {
  if (!error) return false;
  if (typeof error.retryable === "boolean") return error.retryable;
  const name = String(error.name || "");
  const message = String(error.message || error);
  const status = Number(error.status);
  if (status) return status === 408 || status === 425 || status === 429 || status >= 500;
  if (["SyntaxError", "SecurityError", "CompileError", "LinkError"].includes(name)) return false;
  if (name === "AbortError") return false;
  if (name === "TypeError" || message === "TIME_OUT") return true;
  if (/failed to fetch|load failed|network error|network request|timed? out/i.test(message)) {
    return true;
  }
  return error.phase === "worker-bootstrap";
}

export function retryDelayMs(failedAttempt, baseDelayMs = 500, maxDelayMs = 4000) {
  const attempt = Math.max(1, Math.trunc(Number(failedAttempt) || 1));
  return Math.min(maxDelayMs, baseDelayMs * (2 ** (attempt - 1)));
}

export function delayWithSignal(delayMs, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, Math.max(0, delayMs));
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

export async function runWithRetries(
  operation,
  {
    maxAttempts = 3,
    shouldRetry = isRetryableLoadError,
    delay = delayWithSignal,
    signal,
    onRetry = () => {},
  } = {},
) {
  const attempts = Math.max(1, Math.trunc(Number(maxAttempts) || 1));
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      return await operation(attempt);
    } catch (error) {
      if (attempt >= attempts || !shouldRetry(error)) throw error;
      const waitMs = retryDelayMs(attempt);
      onRetry({ error, failedAttempt: attempt, nextAttempt: attempt + 1, maxAttempts: attempts, waitMs });
      await delay(waitMs, signal);
    }
  }
  throw new Error("retry loop exhausted");
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

function workerMessageError(message, fallback = "worker initialization failed") {
  const detail = message && typeof message.error === "object"
    ? message.error
    : { message: message?.error };
  const error = new Error(String(detail?.message || fallback));
  if (detail?.name) error.name = String(detail.name);
  if (detail?.phase) error.phase = String(detail.phase);
  if (typeof detail?.retryable === "boolean") error.retryable = detail.retryable;
  return error;
}

function initializationError(error, phase) {
  const wrapped = error instanceof Error
    ? error
    : new Error(String(error || "engine initialization failed"));
  if (!wrapped.phase) wrapped.phase = phase;
  return wrapped;
}

/* Initialize Emscripten from the exact wasm and .data bytes already verified
 * by the aggregate loader. This shared path is used by both the browser Worker
 * and the executable Node smoke test so a future Emscripten change cannot
 * silently restore hidden binary fetches in production only. */
export async function initializeDownloadedEngineModule(
  createModule,
  wasmBinary,
  dataBinary,
) {
  if (typeof createModule !== "function") {
    throw initializationError(new TypeError("invalid engine module factory"), "module-validation");
  }
  if (!(wasmBinary instanceof Uint8Array) || wasmBinary.byteLength === 0) {
    throw initializationError(new TypeError("missing wasm payload"), "payload-validation");
  }
  if (!(dataBinary instanceof Uint8Array) || dataBinary.byteLength === 0) {
    throw initializationError(new TypeError("missing data payload"), "payload-validation");
  }

  let compiledWasm;
  try {
    compiledWasm = await WebAssembly.compile(wasmBinary);
  } catch (error) {
    throw initializationError(error, "wasm-compile");
  }

  // Uint8Array views (notably Node Buffers) may cover only part of their
  // backing allocation. Give Emscripten an exact package ArrayBuffer.
  const packageBytes = dataBinary.slice().buffer;
  try {
    return await createModule({
      instantiateWasm: (imports, receiveInstance) => {
        const instance = new WebAssembly.Instance(compiledWasm, imports);
        receiveInstance(instance, compiledWasm);
        return instance.exports;
      },
      getPreloadedPackage: (_packageName, expectedSize) => {
        if (Number.isFinite(expectedSize) && expectedSize !== dataBinary.byteLength) {
          throw new Error(
            `data payload size mismatch: expected ${expectedSize}, got ${dataBinary.byteLength}`,
          );
        }
        return packageBytes;
      },
    });
  } catch (error) {
    throw initializationError(error, "engine-initialize");
  }
}

/* Transfer private copies of the downloaded wasm/data payloads and resolve
 * only after Emscripten has loaded the prepackaged LAPACK side module. Keeping
 * the caller's original bytes attached makes a failed Worker safely retryable. */
export function initializeWorker(
  worker,
  { wasmBinary, dataBinary, moduleUrl, helpersUrl },
  onMessage = () => {},
) {
  if (!(wasmBinary instanceof Uint8Array) || !(dataBinary instanceof Uint8Array)) {
    throw new TypeError("worker initialization requires wasm and data bytes");
  }
  const workerWasm = wasmBinary.slice();
  const workerData = dataBinary.slice();
  return new Promise((resolve, reject) => {
    let ready = false;
    worker.onmessage = (event) => {
      const message = event.data;
      if (message.type === "ready") {
        ready = true;
        resolve(message);
      } else if (message.type === "error" && !ready) {
        reject(workerMessageError(message));
      } else {
        onMessage(message);
      }
    };
    worker.onerror = (event) => {
      const source = event?.error;
      const error = new Error(source?.message || event?.message || "worker error");
      error.name = source?.name || "Error";
      error.phase = ready ? "worker-runtime" : "worker-bootstrap";
      if (ready) {
        onMessage({
          type: "error",
          error: { name: error.name, message: error.message, phase: error.phase },
        });
      } else {
        reject(error);
      }
    };
    try {
      worker.postMessage({
        type: "init",
        wasmBinary: workerWasm,
        dataBinary: workerData,
        moduleUrl,
        helpersUrl,
      }, [workerWasm.buffer, workerData.buffer]);
    } catch (error) {
      reject(error);
    }
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
