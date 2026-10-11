import assert from "node:assert/strict";
import test from "node:test";

import {
  parseRetainedSamples,
  summarize,
  validateResult,
} from "./wasm_benchmark.mjs";

test("summarize retains raw order and reports odd and even medians", () => {
  assert.deepEqual(summarize([3, 1, 2]), {
    median_ms: 2,
    range_ms: [1, 3],
    samples_ms: [3, 1, 2],
  });
  assert.equal(summarize([4, 1, 3, 2]).median_ms, 2.5);
});

test("benchmark sample counts require at least three complete samples", () => {
  assert.equal(parseRetainedSamples(), 5);
  assert.equal(parseRetainedSamples("3"), 3);
  for (const invalid of ["2", "3.5", "three", "3samples"]) {
    assert.throws(() => parseRetainedSamples(invalid), /at least 3/);
  }
});

test("water validation rejects converged reference mismatches", () => {
  const waterResult = {
    ok: 1,
    model: 2,
    energy_Eh: -5.06262145,
    scc_iterations: 9,
    scc_converged: 1,
    charges: Array(3).fill(0),
    forces: Array.from({ length: 3 }, () => [0, 0, 0]),
  };
  assert.doesNotThrow(() => validateResult("water", waterResult));
  assert.throws(
    () => validateResult("water", { ...waterResult, energy_Eh: -4.0 }),
    /water energy differs from the reference/,
  );
  assert.throws(
    () => validateResult("water", { ...waterResult, scc_iterations: 8 }),
    /water SCC iteration count differs from the reference/,
  );
});
