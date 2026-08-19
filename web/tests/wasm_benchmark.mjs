import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { initializeDownloadedEngineModule } from "../app_helpers.js";
import { C60_REFERENCE, C60_XYZ } from "../c60_case.js";

const MODEL_GFN2_XTB = 2;
const WATER_XYZ = "O 0 0 0\nH 0 0 0.9572\nH 0 0.75718 -0.58552";
const WATER_REFERENCE = {
  energyEh: -5.06262145,
  sccIterations: 9,
};

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

export function summarize(samples) {
  assert.ok(samples.length > 0, "at least one timing sample is required");
  const sorted = [...samples].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
  return {
    median_ms: median,
    range_ms: [sorted[0], sorted.at(-1)],
    samples_ms: samples,
  };
}

export function parseRetainedSamples(value = "5") {
  const retainedSamples = Number(value);
  assert.ok(
    Number.isInteger(retainedSamples) && retainedSamples >= 3,
    "retained sample count must be an integer of at least 3",
  );
  return retainedSamples;
}

async function loadEngine(sitePath) {
  const site = path.resolve(sitePath);
  const [wasmBinary, dataBinary] = await Promise.all([
    readFile(path.join(site, "xtbloom_web.wasm")),
    readFile(path.join(site, "xtbloom_web.side.wasm")),
  ]);
  const createModule = (
    await import(pathToFileURL(path.join(site, "xtbloom_web.js")).href)
  ).default;
  const Module = await initializeDownloadedEngineModule(
    createModule,
    wasmBinary,
    dataBinary,
  );
  return { Module, wasmBinary, dataBinary };
}

function makeCompute(Module) {
  return (xyz) => {
    const raw = Module.ccall(
      "xtbloom_web_compute",
      "string",
      [
        "string",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
      ],
      [xyz, MODEL_GFN2_XTB, 0, 0, 0, 1e-8, 1e-5, 250, 1],
    );
    return JSON.parse(raw);
  };
}

export function validateResult(name, result) {
  assert.equal(result.ok, 1, `${name} calculation failed`);
  assert.equal(result.model, MODEL_GFN2_XTB);
  assert.ok(Number.isFinite(result.energy_Eh));
  assert.ok(Number.isInteger(result.scc_iterations));
  assert.ok(result.scc_iterations > 0);
  assert.equal(result.scc_converged, 1);
  const expectedAtoms = name === "c60" ? 60 : 3;
  assert.equal(result.charges.length, expectedAtoms);
  assert.equal(result.forces.length, expectedAtoms);
  const reference = name === "c60" ? C60_REFERENCE : WATER_REFERENCE;
  assert.equal(
    result.scc_iterations,
    reference.sccIterations,
    `${name} SCC iteration count differs from the reference`,
  );
  assert.ok(
    Math.abs(result.energy_Eh - reference.energyEh) < 1e-6,
    `${name} energy differs from the reference`,
  );
}

function benchmark(name, xyz, compute, retainedSamples) {
  validateResult(name, compute(xyz));
  const samples = [];
  let result;
  for (let sample = 0; sample < retainedSamples; sample += 1) {
    const start = process.hrtime.bigint();
    result = compute(xyz);
    const elapsed = process.hrtime.bigint() - start;
    validateResult(name, result);
    samples.push(Number(elapsed) / 1e6);
  }
  return {
    ...summarize(samples),
    result: {
      energy_Eh: result.energy_Eh,
      scc_iterations: result.scc_iterations,
      scc_converged: result.scc_converged,
    },
  };
}

export async function runBenchmark({
  sitePath,
  label,
  retainedSamples,
  sourceRevision = "unknown",
}) {
  const { Module, wasmBinary, dataBinary } = await loadEngine(sitePath);
  const compute = makeCompute(Module);
  return {
    label,
    source_revision: sourceRevision,
    site: path.resolve(sitePath),
    runtime: {
      node: process.version,
      v8: process.versions.v8,
      platform: process.platform,
      os_release: os.release(),
      arch: process.arch,
      cpu: os.cpus()[0]?.model || "unknown",
      logical_cpus: os.cpus().length,
    },
    warmup_samples: 1,
    retained_samples: retainedSamples,
    timed_scope: "synchronous xtbloom_web_compute ccall plus JSON parsing",
    artifacts: {
      main_wasm: {
        bytes: wasmBinary.byteLength,
        sha256: sha256(wasmBinary),
      },
      side_wasm: {
        bytes: dataBinary.byteLength,
        sha256: sha256(dataBinary),
      },
    },
    workloads: {
      water: benchmark("water", WATER_XYZ, compute, retainedSamples),
      c60: benchmark("c60", C60_XYZ, compute, retainedSamples),
    },
  };
}

if (fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  const report = await runBenchmark({
    sitePath: process.argv[2] || "build/wasm32-web/web/site",
    label: process.argv[3] || "web-build",
    retainedSamples: parseRetainedSamples(process.argv[4]),
    sourceRevision: process.argv[5] || "unknown",
  });
  console.log(JSON.stringify(report, null, 2));
}
