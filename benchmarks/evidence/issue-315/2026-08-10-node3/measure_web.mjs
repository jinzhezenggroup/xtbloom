import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, readdir, stat, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { pathToFileURL } from "node:url";

import { initializeDownloadedEngineModule } from
  "/home/jzzeng/codes/xtbloom-web-eigen/web/app_helpers.js";
import { C60_REFERENCE, C60_XYZ } from
  "/home/jzzeng/codes/xtbloom-web-eigen/web/c60_case.js";

if (process.argv.length !== 7) {
  throw new Error(
    "usage: node measure-web.mjs <label> <revision> <site> " +
      "<expect-c60-success> <output>",
  );
}

const [, , label, revision, siteArgument, expectArgument, output] = process.argv;
const site = path.resolve(siteArgument);
const expectC60Success = expectArgument === "true";
const warmups = 1;
const repetitions = 5;
const water = "O 0 0 0\nH 0 0 0.9572\nH 0 0.75718 -0.58552";

async function sha256(file) {
  return createHash("sha256").update(await readFile(file)).digest("hex");
}

async function directorySize(root) {
  let total = 0;
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const candidate = path.join(root, entry.name);
    total += entry.isDirectory() ? await directorySize(candidate) : (await stat(candidate)).size;
  }
  return total;
}

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

function compute(xyz) {
  return JSON.parse(
    Module.ccall(
      "xtbloom_web_compute",
      "string",
      ["string", "number", "number", "number", "number", "number", "number", "number"],
      [xyz, 0, 0, 0, 1e-8, 1e-5, 250, 1],
    ),
  );
}

function summarize(result, c60) {
  const summary = {
    ok: result.ok,
    error_code: result.error_code ?? null,
    message: result.message ?? null,
    energy_Eh: result.energy_Eh ?? null,
    scc_converged: result.scc_converged ?? null,
    scc_iterations: result.scc_iterations ?? null,
  };
  if (!result.ok) {
    summary.failure_payload = result;
    return summary;
  }
  if (!c60) return summary;

  const charges = result.charges.map(({ q }) => q);
  const forces = result.forces.map(
    ({ fx_eh_bohr: x, fy_eh_bohr: y, fz_eh_bohr: z }) => [x, y, z],
  );
  summary.charge_sum_e = charges.reduce((sum, value) => sum + value, 0);
  summary.force_sum_Eh_per_bohr = [0, 1, 2].map((axis) =>
    forces.reduce((sum, force) => sum + force[axis], 0),
  );
  summary.max_abs_force_Eh_per_bohr = Math.max(...forces.flat().map(Math.abs));
  summary.max_checkpoint_charge_error_e = Math.max(
    ...C60_REFERENCE.checkpoints.map(({ index, charge }) =>
      Math.abs(charges[index] - charge),
    ),
  );
  summary.max_checkpoint_force_error_Eh_per_bohr = Math.max(
    ...C60_REFERENCE.checkpoints.flatMap(({ index, force }) =>
      force.map((value, axis) => Math.abs(forces[index][axis] - value)),
    ),
  );
  return summary;
}

function validate(result, c60) {
  if (!c60) {
    assert.equal(result.ok, 1);
    assert.equal(result.scc_converged, 1);
    assert.ok(Number.isFinite(result.energy_Eh));
    return;
  }
  if (!expectC60Success) {
    assert.equal(result.ok, 0);
    assert.equal(result.error_code, "err_compute");
    return;
  }
  assert.equal(result.ok, 1);
  assert.equal(result.scc_converged, 1);
  assert.equal(result.scc_iterations, C60_REFERENCE.sccIterations);
  assert.ok(Math.abs(result.energy_Eh - C60_REFERENCE.energyEh) < 1e-6);
  const summary = summarize(result, true);
  assert.ok(Math.abs(summary.charge_sum_e) < 2e-8);
  assert.ok(summary.max_checkpoint_charge_error_e < 2e-9);
  assert.ok(summary.max_checkpoint_force_error_Eh_per_bohr < 2e-8);
}

async function measure(name, xyz, c60) {
  const warmupResults = [];
  for (let index = 0; index < warmups; index += 1) {
    const start = performance.now();
    const result = compute(xyz);
    const milliseconds = performance.now() - start;
    validate(result, c60);
    warmupResults.push({ milliseconds, result: summarize(result, c60) });
  }

  const samples = [];
  for (let index = 0; index < repetitions; index += 1) {
    const start = performance.now();
    const result = compute(xyz);
    const milliseconds = performance.now() - start;
    validate(result, c60);
    samples.push({ milliseconds, result: summarize(result, c60) });
  }
  return { name, warmupResults, samples, correctness_qualified: true };
}

const wasmPath = path.join(site, "xtbloom_web.wasm");
const dataPath = path.join(site, "xtbloom_web.data");
const sideModulePath = path.join(site, "..", "libscipy_openblas.so");
const record = {
  schema_version: 1,
  issue: 315,
  label,
  source_revision: revision,
  source_dirty_at_start: false,
  measured_at_utc: new Date().toISOString(),
  runtime: {
    node: process.version,
    platform: `${process.platform}-${process.arch}`,
    cpu_model: os.cpus()[0]?.model ?? "unknown",
    logical_cpu_count: os.cpus().length,
    process_affinity: "CPU 0 via taskset -c 0",
  },
  protocol: {
    public_entry_point: "xtbloom_web_compute",
    backend: "single-threaded wasm32 CPU",
    descriptors: "host-resident Web adapter inputs and outputs",
    requested_properties: "energy, charges, analytic forces",
    charge: 0,
    unpaired_electrons: 0,
    electronic_temperature: "300 K default",
    energy_tolerance_Eh: 1e-8,
    charge_tolerance_e: 1e-5,
    maximum_scc_iterations: 250,
    warmups,
    repetitions,
    synchronization_boundary: "synchronous return from Emscripten ccall",
  },
  artifacts: {
    site_path: site,
    site_payload_bytes: await directorySize(site),
    wasm: {
      bytes: (await stat(wasmPath)).size,
      sha256: await sha256(wasmPath),
    },
    data: {
      bytes: (await stat(dataPath)).size,
      sha256: await sha256(dataPath),
    },
    side_module: {
      bytes: (await stat(sideModulePath)).size,
      sha256: await sha256(sideModulePath),
    },
  },
  workloads: [
    await measure("water-3-atoms", water, false),
    await measure("c60-60-atoms-240-orbitals", C60_XYZ, true),
  ],
};

await writeFile(output, `${JSON.stringify(record, null, 2)}\n`, "utf8");
console.log(`wrote ${output}`);
