import assert from "node:assert/strict";
import path from "node:path";
import { pathToFileURL } from "node:url";

const site = path.resolve(process.argv[2] || "build/wasm-web/web/site");
const moduleUrl = pathToFileURL(path.join(site, "gpuxtb_web.js")).href;
const createModule = (await import(moduleUrl)).default;
// Emscripten resolves the packaged data file from the process working directory.
process.chdir(site);
const Module = await createModule();

const water = "O 0 0 0\nH 0 0 0.9572\nH 0 0.75718 -0.58552";
const stretchedWater = "O 0 0 0\nH 0 0 1.15\nH 0 0.9 -0.7";

function compute(maxIterations, forces) {
  const raw = Module.ccall(
    "gpuxtb_web_compute", "string",
    ["string", "number", "number", "number", "number", "number", "number", "number"],
    [water, 0, 0, 0, 1e-8, 1e-5, maxIterations, forces ? 1 : 0],
  );
  return { raw, result: JSON.parse(raw) };
}

const version = Module.ccall("gpuxtb_web_version", "string", [], []);
const withForces = compute(250, true).result;
assert.equal(withForces.ok, 1);
assert.ok(Number.isFinite(withForces.energy_Eh));
assert.equal(withForces.forces.length, 3);

const withoutForces = compute(250, false).result;
assert.equal(withoutForces.ok, 1);
assert.equal(Object.hasOwn(withoutForces, "forces"), false);

const failedCompute = compute(1, true).result;
assert.equal(failedCompute.ok, 0);
assert.equal(failedCompute.error_code, "err_compute");

const failedOptimizeRaw = Module.ccall(
  "gpuxtb_web_optimize", "string",
  ["string", "number", "number", "number", "number", "number", "number", "number", "number", "number"],
  [water, 0, 0, 0, 1e-8, 1e-5, 1, 2, 4.5e-4, 0.4],
);
const failedOptimize = JSON.parse(failedOptimizeRaw);
assert.equal(failedOptimize.ok, 0);
assert.equal(failedOptimize.error_code, "err_initial_calc");

const optimizeRaw = Module.ccall(
  "gpuxtb_web_optimize", "string",
  ["string", "number", "number", "number", "number", "number", "number", "number", "number", "number"],
  [stretchedWater, 0, 0, 0, 1e-8, 1e-5, 250, 2, 1e-12, 0.4],
);
const optimized = JSON.parse(optimizeRaw);
assert.equal(optimized.ok, 1);
assert.equal(optimized.trajectory.length, 3);
for (let i = 1; i < optimized.trajectory.length; i += 1) {
  assert.ok(Number.isFinite(optimized.trajectory[i]));
  assert.ok(optimized.trajectory[i] <= optimized.trajectory[i - 1]);
}

console.log(
  "smoke ok, version", version,
  "energy", withForces.energy_Eh,
  "optimized", optimized.energy_final_Eh,
);
