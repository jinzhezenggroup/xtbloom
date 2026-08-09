import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  BOHR_PER_ANGSTROM,
  angstromToBohr,
  copyFloat64FromMemory,
  initializeWorker,
  postToReadyWorker,
  withTimeout,
} from "../app_helpers.js";

class FakeWorker {
  constructor() {
    this.messages = [];
    this.terminated = false;
    this.onmessage = null;
    this.onerror = null;
  }

  postMessage(message, transfer) {
    this.messages.push({ message, transfer });
  }

  emit(message) {
    this.onmessage({ data: message });
  }

  terminate() {
    this.terminated = true;
  }
}

test("angstrom controls are converted to optimizer bohr", () => {
  assert.equal(angstromToBohr(1), BOHR_PER_ANGSTROM);
  assert.ok(Math.abs(angstromToBohr(0.4) - 0.7558904501831313) < 1e-15);
});

test("worker initialization remains pending until ready", async () => {
  const worker = new FakeWorker();
  const wasmBinary = new Uint8Array([0, 1, 2]);
  let settled = false;
  const initialized = initializeWorker(worker, wasmBinary).then((message) => {
    settled = true;
    return message;
  });

  await Promise.resolve();
  assert.equal(settled, false);
  assert.equal(worker.messages[0].message.type, "init");
  assert.equal(worker.messages[0].transfer[0], wasmBinary.buffer);

  worker.emit({ type: "ready", version: "test" });
  assert.deepEqual(await initialized, { type: "ready", version: "test" });
});

test("worker initialization errors reject", async () => {
  const worker = new FakeWorker();
  const initialized = initializeWorker(worker, new Uint8Array([0]));
  worker.emit({ type: "error", error: "side module failed" });
  await assert.rejects(initialized, /side module failed/);
});

test("calls require a live ready worker", () => {
  const worker = new FakeWorker();
  assert.throws(() => postToReadyWorker(null, "ready", { type: "call" }), /not ready/);
  assert.throws(() => postToReadyWorker(worker, "loading", { type: "call" }), /not ready/);

  postToReadyWorker(worker, "ready", { type: "call", id: 7 });
  assert.deepEqual(worker.messages[0].message, { type: "call", id: 7 });
});

test("Float64 callback slices accept wasm32 Numbers and wasm64 BigInts", () => {
  const memory = { buffer: new ArrayBuffer(64) };
  new Float64Array(memory.buffer).set([1.25, -2.5, 3.75], 2);
  assert.deepEqual(copyFloat64FromMemory(memory, 16, 3), [1.25, -2.5, 3.75]);
  assert.deepEqual(copyFloat64FromMemory(memory, 16n, 3), [1.25, -2.5, 3.75]);
  assert.throws(() => copyFloat64FromMemory(memory, 4, 1), /misaligned/);
  assert.throws(() => copyFloat64FromMemory(memory, 56, 2), /outside memory/);
  assert.throws(
    () => copyFloat64FromMemory(memory, BigInt(Number.MAX_SAFE_INTEGER) + 1n, 1),
    /safe integer range/,
  );
});

test("deployed page labels the universal browser artifact as wasm32", async () => {
  const [appSource, indexSource] = await Promise.all([
    readFile(new URL("../app.js", import.meta.url), "utf8"),
    readFile(new URL("../index.html", import.meta.url), "utf8"),
  ]);
  assert.doesNotMatch(appSource, /C\+\+17 WASM64/i);
  assert.doesNotMatch(indexSource, /C\+\+17 WASM64/i);
  assert.match(appSource, /C\+\+17 WASM32/i);
  assert.match(indexSource, /C\+\+17 WASM32/i);
});

test("timeout covers a worker that never becomes ready", async () => {
  const worker = new FakeWorker();
  const initialized = initializeWorker(worker, new Uint8Array([0]));
  await assert.rejects(
    withTimeout(initialized, 5, () => worker.terminate()),
    /TIME_OUT/,
  );
  assert.equal(worker.terminated, true);
});
