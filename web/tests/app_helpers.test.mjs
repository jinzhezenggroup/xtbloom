import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  BOHR_PER_ANGSTROM,
  angstromToBohr,
  canStartUrlSmiles,
  clampProgressPercent,
  comparableContentLength,
  copyFloat64FromMemory,
  downloadProgressPercent,
  initializeWorker,
  postToReadyWorker,
  readSmilesQuery,
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

test("encoded response lengths are not compared with decoded fetch chunks", () => {
  const headers = new Headers({
    "content-encoding": "gzip",
    "content-length": "737252",
  });
  assert.equal(comparableContentLength(headers), 0);
  headers.set("content-encoding", "identity");
  assert.equal(comparableContentLength(headers), 737252);
});

test("download progress is monotonic, bounded, and completes explicitly", () => {
  assert.equal(downloadProgressPercent(50, 100), 50);
  assert.equal(downloadProgressPercent(1138031, 737252), 99);
  assert.equal(downloadProgressPercent(20, 100, 75), 75);
  assert.equal(downloadProgressPercent(0, 0, 42), 42);
  assert.equal(downloadProgressPercent(0, 0, 42, true), 100);
  assert.equal(clampProgressPercent(154.361), 100);
  assert.equal(clampProgressPercent(-4), 0);
  assert.equal(clampProgressPercent(Number.NaN), 0);
});

test("SMILES query parsing is percent-decoded and length bounded", () => {
  assert.equal(readSmilesQuery("https://example.test/?smiles=CCO"), "CCO");
  assert.equal(readSmilesQuery("https://example.test/?smiles=%5BNH4%2B%5D"), "[NH4+]");
  assert.equal(readSmilesQuery("https://example.test/?other=CCO"), null);
  assert.throws(
    () => readSmilesQuery("https://example.test/?smiles=CCCC", 3),
    /exceeds 3 characters/,
  );
});

test("URL SMILES starts once only when both workers are ready and idle", () => {
  const ready = {
    smiles: "CCO",
    started: false,
    engineState: "ready",
    smilesState: "ready",
    engineBusy: false,
    smilesBusy: false,
  };
  assert.equal(canStartUrlSmiles(ready), true);
  for (const override of [
    { smiles: null },
    { started: true },
    { engineState: "loading" },
    { smilesState: "error" },
    { engineBusy: true },
    { smilesBusy: true },
  ]) {
    assert.equal(canStartUrlSmiles({ ...ready, ...override }), false);
  }
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

test("page bootstraps pinned SMILES loading and applies URL-optimized geometry", async () => {
  const [appSource, indexSource] = await Promise.all([
    readFile(new URL("../app.js", import.meta.url), "utf8"),
    readFile(new URL("../index.html", import.meta.url), "utf8"),
  ]);
  assert.match(appSource, /startSmilesWorker\(\);/);
  assert.match(appSource, /readSmilesQuery\(window\.location\.href\)/);
  assert.match(appSource, /applyFinalGeometry:\s*true/);
  assert.match(appSource, /\$\("xyz"\)\.value = d\.geometry/);
  assert.doesNotMatch(appSource, /smiles-alert/);
  assert.match(indexSource, /id="smiles"/);
  assert.match(indexSource, /id="smiles-generate"/);
  assert.doesNotMatch(indexSource, /id="smiles-alert"/);
});

test("every literal app DOM lookup exists in the deployed HTML", async () => {
  const [appSource, indexSource] = await Promise.all([
    readFile(new URL("../app.js", import.meta.url), "utf8"),
    readFile(new URL("../index.html", import.meta.url), "utf8"),
  ]);
  const referenced = new Set(
    Array.from(appSource.matchAll(/\$\("([^"]+)"\)/g), (match) => match[1]),
  );
  const declared = new Set(
    Array.from(indexSource.matchAll(/\bid="([^"]+)"/g), (match) => match[1]),
  );
  assert.deepEqual([...referenced].filter((id) => !declared.has(id)).sort(), []);
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
