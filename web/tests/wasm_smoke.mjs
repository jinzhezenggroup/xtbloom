import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { copyFloat64FromMemory } from "../app_helpers.js";

const water = "O 0 0 0\nH 0 0 0.9572\nH 0 0.75718 -0.58552";
const stretchedWater = "O 0 0 0\nH 0 0 1.15\nH 0 0.9 -0.7";

export async function runWebCases(sitePath) {
  const site = path.resolve(sitePath);
  const moduleUrl = pathToFileURL(path.join(site, "gpuxtb_web.js")).href;
  const createModule = (await import(moduleUrl)).default;
  const Module = await createModule({
    // Use absolute paths so two pointer-width modules can be instantiated and
    // compared in one Node process without changing global working directory.
    locateFile: (name) => path.join(site, name),
  });

  function compute(maxIterations, forces) {
    const raw = Module.ccall(
      "gpuxtb_web_compute", "string",
      ["string", "number", "number", "number", "number", "number", "number", "number"],
      [water, 0, 0, 0, 1e-8, 1e-5, maxIterations, forces ? 1 : 0],
    );
    return JSON.parse(raw);
  }

  const failedOptimize = JSON.parse(Module.ccall(
    "gpuxtb_web_optimize", "string",
    ["string", "number", "number", "number", "number", "number", "number", "number", "number", "number"],
    [water, 0, 0, 0, 1e-8, 1e-5, 1, 2, 4.5e-4, 0.4],
  ));
  const callbackFrames = [];
  const stepFn = Module.addFunction((iter, natoms, pointer, energy, fmax) => {
    callbackFrames.push({
      iter,
      natoms,
      coords: copyFloat64FromMemory(Module.wasmMemory, pointer, natoms * 3),
      energy,
      fmax,
    });
  }, "viipdd");
  Module.ccall("gpuxtb_web_set_optimize_step_cb", "void", ["pointer"], [stepFn]);
  let optimized;
  try {
    optimized = JSON.parse(Module.ccall(
      "gpuxtb_web_optimize", "string",
      ["string", "number", "number", "number", "number", "number", "number", "number", "number", "number"],
      [stretchedWater, 0, 0, 0, 1e-8, 1e-5, 250, 2, 1e-12, 0.4],
    ));
  } finally {
    Module.ccall("gpuxtb_web_set_optimize_step_cb", "void", ["pointer"], [0]);
  }

  return {
    version: Module.ccall("gpuxtb_web_version", "string", [], []),
    withForces: compute(250, true),
    withoutForces: compute(250, false),
    failedCompute: compute(1, true),
    failedOptimize,
    optimized,
    callbackFrames,
  };
}

export function validateWebCases(cases) {
  assert.equal(cases.withForces.ok, 1);
  assert.ok(Number.isFinite(cases.withForces.energy_Eh));
  assert.equal(cases.withForces.forces.length, 3);
  assert.equal(cases.withForces.charges.length, 3);
  assert.equal(cases.withoutForces.ok, 1);
  assert.equal(Object.hasOwn(cases.withoutForces, "forces"), false);
  assert.equal(cases.failedCompute.ok, 0);
  assert.equal(cases.failedCompute.error_code, "err_compute");
  assert.equal(cases.failedOptimize.ok, 0);
  assert.equal(cases.failedOptimize.error_code, "err_initial_calc");
  assert.equal(cases.optimized.ok, 1);
  assert.equal(cases.optimized.trajectory.length, 3);
  assert.equal(cases.callbackFrames.length, 2);
  assert.equal(cases.callbackFrames[0].coords.length, 9);
  assert.ok(cases.callbackFrames[0].coords.every(Number.isFinite));
  for (let i = 1; i < cases.optimized.trajectory.length; i += 1) {
    assert.ok(Number.isFinite(cases.optimized.trajectory[i]));
    assert.ok(cases.optimized.trajectory[i] <= cases.optimized.trajectory[i - 1]);
  }
}

const invokedAsScript = process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (invokedAsScript) {
  const cases = await runWebCases(process.argv[2] || "build/wasm32-web/web/site");
  validateWebCases(cases);
  console.log(
    "smoke ok, version", cases.version,
    "energy", cases.withForces.energy_Eh,
    "optimized", cases.optimized.energy_final_Eh,
  );
}
