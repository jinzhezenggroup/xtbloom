import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  copyFloat64FromMemory,
  initializeDownloadedEngineModule,
} from "../app_helpers.js";
import { C60_REFERENCE, C60_XYZ } from "../c60_case.js";

const water = "O 0 0 0\nH 0 0 0.9572\nH 0 0.75718 -0.58552";
const stretchedWater = "O 0 0 0\nH 0 0 1.15\nH 0 0.9 -0.7";

export async function runWebCases(sitePath) {
  const site = path.resolve(sitePath);
  const moduleUrl = pathToFileURL(path.join(site, "xtbloom_web.js")).href;
  const createModule = (await import(moduleUrl)).default;
  const [wasmBinary, dataBinary] = await Promise.all([
    readFile(path.join(site, "xtbloom_web.wasm")),
    readFile(path.join(site, "xtbloom_web.data")),
  ]);
  const Module = await initializeDownloadedEngineModule(
    createModule,
    wasmBinary,
    dataBinary,
  );

  function compute(xyz, maxIterations, forces) {
    const raw = Module.ccall(
      "xtbloom_web_compute", "string",
      ["string", "number", "number", "number", "number", "number", "number", "number"],
      [xyz, 0, 0, 0, 1e-8, 1e-5, maxIterations, forces ? 1 : 0],
    );
    return JSON.parse(raw);
  }

  const failedOptimize = JSON.parse(Module.ccall(
    "xtbloom_web_optimize", "string",
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
  Module.ccall("xtbloom_web_set_optimize_step_cb", "void", ["pointer"], [stepFn]);
  let optimized;
  try {
    optimized = JSON.parse(Module.ccall(
      "xtbloom_web_optimize", "string",
      ["string", "number", "number", "number", "number", "number", "number", "number", "number", "number"],
      [stretchedWater, 0, 0, 0, 1e-8, 1e-5, 250, 2, 1e-12, 0.4],
    ));
  } finally {
    Module.ccall("xtbloom_web_set_optimize_step_cb", "void", ["pointer"], [0]);
  }

  return {
    version: Module.ccall("xtbloom_web_version", "string", [], []),
    withForces: compute(water, 250, true),
    withoutForces: compute(water, 250, false),
    failedCompute: compute(water, 1, true),
    c60: compute(C60_XYZ, 250, true),
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
  assert.equal(cases.c60.ok, 1);
  assert.equal(cases.c60.scc_converged, 1);
  assert.equal(cases.c60.scc_iterations, C60_REFERENCE.sccIterations);
  assert.ok(Math.abs(cases.c60.energy_Eh - C60_REFERENCE.energyEh) < 1e-6);
  assert.equal(cases.c60.charges.length, 60);
  assert.equal(cases.c60.forces.length, 60);
  const c60Charges = cases.c60.charges.map(({ element, q }) => {
    assert.equal(element, 6);
    assert.ok(Number.isFinite(q));
    return q;
  });
  const c60Forces = cases.c60.forces.map(
    ({ element, fx_eh_bohr: fx, fy_eh_bohr: fy, fz_eh_bohr: fz }) => {
      assert.equal(element, 6);
      assert.ok([fx, fy, fz].every(Number.isFinite));
      return [fx, fy, fz];
    },
  );
  assert.ok(Math.abs(c60Charges.reduce((sum, value) => sum + value, 0)) < 2e-8);
  for (let axis = 0; axis < 3; axis += 1) {
    const sum = c60Forces.reduce((total, force) => total + force[axis], 0);
    assert.ok(Math.abs(sum - C60_REFERENCE.forceSum[axis]) < 2e-8);
  }
  const maxAbsForce = Math.max(...c60Forces.flat().map(Math.abs));
  assert.ok(Math.abs(maxAbsForce - C60_REFERENCE.maxAbsForce) < 2e-8);
  for (const checkpoint of C60_REFERENCE.checkpoints) {
    assert.ok(Math.abs(c60Charges[checkpoint.index] - checkpoint.charge) < 2e-9);
    for (let axis = 0; axis < 3; axis += 1) {
      assert.ok(
        Math.abs(c60Forces[checkpoint.index][axis] - checkpoint.force[axis]) < 2e-8,
      );
    }
  }
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
