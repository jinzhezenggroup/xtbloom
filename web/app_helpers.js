/* Pure browser-engine helpers kept separate so units and readiness semantics
 * are executable under Node without constructing the full page DOM. */

export const BOHR_PER_ANGSTROM = 1.8897261254578281;

export function angstromToBohr(value) {
  return value * BOHR_PER_ANGSTROM;
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
