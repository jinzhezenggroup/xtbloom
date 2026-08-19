import assert from "node:assert/strict";
import test from "node:test";

import { parseRetainedSamples, summarize } from "./wasm_benchmark.mjs";

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
