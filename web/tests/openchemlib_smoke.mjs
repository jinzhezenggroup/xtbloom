import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import {
  OPEN_CHEMLIB_MODULE_URL,
  OPEN_CHEMLIB_RESOURCES_URL,
  OPEN_CHEMLIB_VERSION,
  smilesToGeometry,
} from "../smiles_helpers.js";

const EXPECTED_MODULE_SHA256 =
  "5978967b12e938208e8d36222370f88fd615a2b5ec83f02e435caab26f3f4cb3";
const EXPECTED_RESOURCES_SHA256 =
  "d2741130d5a5546aeebebc43eb3dac937881b04755fefe5925e4b228a56bee14";

async function fetchPinned(url) {
  const response = await fetch(url, { signal: AbortSignal.timeout(60000) });
  assert.equal(response.ok, true, `${url}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function inspectGeometry(result, expectedAtoms, expectedCharge) {
  assert.equal(result.atomCount, expectedAtoms);
  assert.equal(result.formalCharge, expectedCharge);
  assert.equal(result.mmffStatus, 0);
  const lines = result.xyz.split("\n");
  assert.equal(lines.length, expectedAtoms);
  const symbols = [];
  for (const line of lines) {
    const [symbol, ...coordinateText] = line.trim().split(/\s+/);
    symbols.push(symbol);
    assert.equal(coordinateText.length, 3);
    assert.equal(coordinateText.map(Number).every(Number.isFinite), true);
  }
  return symbols;
}

const [moduleBytes, resourcesBytes] = await Promise.all([
  fetchPinned(OPEN_CHEMLIB_MODULE_URL),
  fetchPinned(OPEN_CHEMLIB_RESOURCES_URL),
]);
assert.equal(sha256(moduleBytes), EXPECTED_MODULE_SHA256);
assert.equal(sha256(resourcesBytes), EXPECTED_RESOURCES_SHA256);

const resources = JSON.parse(new TextDecoder().decode(resourcesBytes));
assert.equal(Object.keys(resources).length, 35);

const temporaryDirectory = await mkdtemp(join(tmpdir(), "xtbloom-openchemlib-"));
try {
  const modulePath = join(temporaryDirectory, "openchemlib.mjs");
  await writeFile(modulePath, moduleBytes);
  const OCL = await import(pathToFileURL(modulePath).href);
  assert.equal(String(OCL.version), OPEN_CHEMLIB_VERSION);
  assert.equal(typeof OCL.Resources.registerFromUrl, "function");
  OCL.Resources.register(resources);

  const ethanol = smilesToGeometry(OCL, "CCO");
  assert.equal(inspectGeometry(ethanol, 9, 0).filter((x) => x === "H").length, 6);
  assert.equal(
    smilesToGeometry(OCL, "CCO").xyz,
    ethanol.xyz,
    "the pinned conformer seed must produce reproducible coordinates",
  );

  const benzene = smilesToGeometry(OCL, "c1ccccc1");
  assert.equal(inspectGeometry(benzene, 12, 0).filter((x) => x === "H").length, 6);

  const ammonium = smilesToGeometry(OCL, "[NH4+]");
  assert.equal(inspectGeometry(ammonium, 5, 1).filter((x) => x === "H").length, 4);

  const acetate = smilesToGeometry(OCL, "CC(=O)[O-]");
  assert.equal(inspectGeometry(acetate, 7, -1).filter((x) => x === "H").length, 3);

  assert.throws(
    () => smilesToGeometry(OCL, "not-a-smiles"),
    (error) => error && error.code === "smiles_err_parse",
  );
  assert.throws(
    () => smilesToGeometry(OCL, "C.C"),
    (error) => error && error.code === "smiles_err_fragments",
  );
  assert.throws(
    () => smilesToGeometry(OCL, "[CH3]"),
    (error) => error && error.code === "smiles_err_radical",
  );
  assert.throws(
    () => smilesToGeometry(OCL, "[Fr]"),
    (error) => error && error.code === "smiles_err_element",
  );
} finally {
  await rm(temporaryDirectory, { recursive: true, force: true });
}

console.log("OpenChemLib pinned-CDN SMILES smoke passed");
