import assert from "node:assert/strict";
import test from "node:test";

import {
  BOHR_PER_ANGSTROM,
  angstromToBohr,
  initializeWorker,
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

test("timeout covers a worker that never becomes ready", async () => {
  const worker = new FakeWorker();
  const initialized = initializeWorker(worker, new Uint8Array([0]));
  await assert.rejects(
    withTimeout(initialized, 5, () => worker.terminate()),
    /TIME_OUT/,
  );
  assert.equal(worker.terminated, true);
});
