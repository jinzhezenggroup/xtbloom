import assert from "node:assert/strict";
import test from "node:test";

import {
  CDN_REGION_GLOBAL,
  CDN_REGION_MAINLAND_CHINA,
  OPEN_CHEMLIB_CDN_URLS,
  OPEN_CHEMLIB_VERSION,
  loadOpenChemLibRuntime,
  normalizeCdnProviderOrder,
  openChemLibUrlsForProviders,
  smilesToGeometry,
} from "../smiles_helpers.js";

function fakeOcl(atoms, options = {}) {
  class Molecule {
    static fromSmiles(smiles) {
      if (smiles === "bad") throw new Error("parse failed");
      const molecule = new Molecule();
      molecule.smiles = smiles;
      return molecule;
    }

    validate() {}
    getFragmentNumbers() { return this.smiles === "C.C" ? 2 : 1; }
    getAllAtoms() { return atoms.length; }
    getAtomicNo(index) { return atoms[index].z; }
    getAtomCharge(index) { return atoms[index].charge || 0; }
    getAtomRadical(index) { return atoms[index].radical || 0; }
    getAtomX(index) { return atoms[index].xyz[0]; }
    getAtomY(index) { return atoms[index].xyz[1]; }
    getAtomZ(index) { return atoms[index].xyz[2]; }
  }

  class ConformerGenerator {
    constructor(seed) { this.seed = seed; }
    getOneConformerAsMolecule(molecule) {
      molecule.seed = this.seed;
      return options.noConformer ? null : molecule;
    }
  }

  class ForceFieldMMFF94 {
    static MMFF94 = "MMFF94";
    minimise() { return options.mmffStatus || 0; }
  }

  return { Molecule, ConformerGenerator, ForceFieldMMFF94 };
}

test("OpenChemLib runtime URLs pin byte-identical reviewed CDN releases", () => {
  assert.equal(OPEN_CHEMLIB_VERSION, "9.21.0");
  for (const [provider, urls] of Object.entries(OPEN_CHEMLIB_CDN_URLS)) {
    assert.match(urls.module, /openchemlib@9\.21\.0\/dist\/openchemlib\.js$/, provider);
    assert.match(urls.resources, /openchemlib@9\.21\.0\/dist\/resources\.json$/, provider);
    assert.doesNotMatch(urls.module, /latest|\+esm/, provider);
  }
  assert.equal(new URL(OPEN_CHEMLIB_CDN_URLS.jsdmirror.module).hostname, "cdn.jsdmirror.com");
  assert.equal(new URL(OPEN_CHEMLIB_CDN_URLS.jsdelivr.module).hostname, "cdn.jsdelivr.net");
});

test("measured providers are normalized before regional fallback defaults", () => {
  assert.deepEqual(normalizeCdnProviderOrder(
    ["jsdelivr", "unknown", "jsdelivr"],
    CDN_REGION_MAINLAND_CHINA,
  ), ["jsdelivr", "jsdmirror"]);
  assert.deepEqual(normalizeCdnProviderOrder([], CDN_REGION_MAINLAND_CHINA), [
    "jsdmirror",
    "jsdelivr",
  ]);
  assert.deepEqual(normalizeCdnProviderOrder([], CDN_REGION_GLOBAL), [
    "jsdelivr",
    "jsdmirror",
  ]);
  assert.deepEqual(
    openChemLibUrlsForProviders(["jsdmirror"], CDN_REGION_GLOBAL).providers,
    ["jsdmirror", "jsdelivr"],
  );
});

test("OpenChemLib retries the complete pinned provider pair after one artifact fails", async () => {
  const attempts = [];
  const registrations = [];
  const OCL = {
    version: OPEN_CHEMLIB_VERSION,
    Resources: { register: (resources) => registrations.push(resources) },
  };
  const resourcesBytes = new TextEncoder().encode(JSON.stringify({ "/resource": "value" }));
  const runtime = await loadOpenChemLibRuntime(["jsdmirror", "jsdelivr"], {
    attemptTimeoutMs: 1000,
    fetchPinnedBytesImpl: async (url) => {
      attempts.push(url);
      if (url.includes("jsdmirror") && url.endsWith("resources.json")) {
        throw new Error("mirror resource failed");
      }
      return url.endsWith("resources.json")
        ? resourcesBytes.buffer
        : new Uint8Array([1]).buffer;
    },
    importModule: async (url) => {
      assert.equal(url, "blob:verified-openchemlib");
      return OCL;
    },
    createModuleUrl: () => ({
      url: "blob:verified-openchemlib",
      revoke: () => {},
    }),
  });
  assert.equal(runtime.provider, "jsdelivr");
  assert.deepEqual(registrations, [{ "/resource": "value" }]);
  const attemptedHosts = new Set(attempts.map((url) => new URL(url).hostname));
  assert.equal(attemptedHosts.has("cdn.jsdmirror.com"), true);
  assert.equal(attemptedHosts.has("cdn.jsdelivr.net"), true);
});

test("SMILES conversion publishes explicit XYZ and total formal charge", () => {
  const OCL = fakeOcl([
    { z: 7, charge: 1, xyz: [0, 0.25, -0.5] },
    { z: 1, xyz: [1, 0, 0] },
  ]);
  const result = smilesToGeometry(OCL, "[NH4+]", { seed: 17 });
  assert.equal(result.atomCount, 2);
  assert.equal(result.formalCharge, 1);
  assert.equal(result.seed, 17);
  assert.equal(result.mmffStatus, 0);
  assert.equal(result.xyz, "N 0.0000000000 0.2500000000 -0.5000000000\nH 1.0000000000 0.0000000000 0.0000000000");
});

test("SMILES conversion rejects unsafe or scientifically ambiguous inputs", () => {
  assert.throws(() => smilesToGeometry(fakeOcl([]), "C"), /no atoms/);
  assert.throws(
    () => smilesToGeometry(fakeOcl([{ z: 87, xyz: [0, 0, 0] }]), "[Fr]"),
    /atomic number 87/,
  );
  assert.throws(
    () => smilesToGeometry(fakeOcl([{ z: 8, radical: 2, xyz: [0, 0, 0] }]), "[O]"),
    /unpaired-electron count/,
  );
  assert.throws(
    () => smilesToGeometry(fakeOcl([{ z: 6, xyz: [Number.NaN, 0, 0] }]), "C"),
    /non-finite coordinates/,
  );
  assert.throws(
    () => smilesToGeometry(fakeOcl([{ z: 6, xyz: [0, 0, 0] }], { mmffStatus: 2 }), "C"),
    /status 2/,
  );
  assert.throws(() => smilesToGeometry(fakeOcl([]), "bad"), /could not parse/);
  assert.throws(() => smilesToGeometry(fakeOcl([{ z: 6, xyz: [0, 0, 0] }]), "C.C"), /relative placement/);
});
