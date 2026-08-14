import assert from "node:assert/strict";
import test from "node:test";

import {
  CDN_REGION_GLOBAL,
  CDN_REGION_MAINLAND_CHINA,
  OPEN_CHEMLIB_CDN_URLS,
  OPEN_CHEMLIB_MODULE_BYTES,
  OPEN_CHEMLIB_MODULE_SHA256,
  OPEN_CHEMLIB_RESOURCES_BYTES,
  OPEN_CHEMLIB_RESOURCES_SHA256,
  OPEN_CHEMLIB_VERSION,
  loadOpenChemLibRuntime,
  normalizeCdnProviderOrder,
  openChemLibUrlsForProviders,
  smilesToGeometry,
} from "../smiles_helpers.js";

function digestBytes(hexDigest) {
  return Uint8Array.from(Buffer.from(hexDigest, "hex")).buffer;
}

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

test("OpenChemLib verifies pinned byte counts and digests before Blob import", async () => {
  const moduleBytes = new Uint8Array(OPEN_CHEMLIB_MODULE_BYTES).buffer;
  const resourcesPrefix = '{"padding":"';
  const resourcesSuffix = '"}';
  const resourcesText = resourcesPrefix + "x".repeat(
    OPEN_CHEMLIB_RESOURCES_BYTES - resourcesPrefix.length - resourcesSuffix.length,
  ) + resourcesSuffix;
  const resourcesBytes = new TextEncoder().encode(resourcesText);
  const signals = [];
  const registrations = [];
  const runtime = await loadOpenChemLibRuntime(["jsdmirror"], {
    fetchImpl: async (url, options) => {
      signals.push(options.signal);
      return {
        ok: true,
        status: 200,
        arrayBuffer: async () => url.endsWith("resources.json")
          ? resourcesBytes.buffer
          : moduleBytes,
      };
    },
    cryptoImpl: {
      subtle: {
        digest: async (_algorithm, bytes) => digestBytes(
          bytes.byteLength === OPEN_CHEMLIB_MODULE_BYTES
            ? OPEN_CHEMLIB_MODULE_SHA256
            : OPEN_CHEMLIB_RESOURCES_SHA256,
        ),
      },
    },
    importModule: async (url) => {
      assert.match(url, /^blob:/);
      return {
        version: OPEN_CHEMLIB_VERSION,
        Resources: { register: (resources) => registrations.push(resources) },
      };
    },
  });
  assert.equal(runtime.provider, "jsdmirror");
  assert.equal(registrations.length, 1);
  assert.equal(registrations[0].padding.length, OPEN_CHEMLIB_RESOURCES_BYTES - 14);
  assert.equal(signals.length, 2);
  assert.equal(signals.every((signal) => signal.aborted), true);
});

test("OpenChemLib rejects unavailable fetch and unverified provider bytes", async () => {
  await assert.rejects(
    loadOpenChemLibRuntime([], { fetchImpl: null }),
    /fetch is unavailable/,
  );

  for (const fixture of [
    {
      name: "HTTP failure",
      fetchImpl: async () => ({ ok: false, status: 503 }),
      message: /HTTP 503/,
    },
    {
      name: "wrong size",
      fetchImpl: async () => ({
        ok: true,
        status: 200,
        arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer,
      }),
      message: /expected \d+ bytes, received 3/,
    },
    {
      name: "missing SHA-256",
      fetchImpl: async (url) => ({
        ok: true,
        status: 200,
        arrayBuffer: async () => new Uint8Array(
          url.endsWith("resources.json")
            ? OPEN_CHEMLIB_RESOURCES_BYTES
            : OPEN_CHEMLIB_MODULE_BYTES,
        ).buffer,
      }),
      cryptoImpl: {},
      message: /SHA-256 verification is unavailable/,
    },
    {
      name: "SHA-256 mismatch",
      fetchImpl: async (url) => ({
        ok: true,
        status: 200,
        arrayBuffer: async () => new Uint8Array(
          url.endsWith("resources.json")
            ? OPEN_CHEMLIB_RESOURCES_BYTES
            : OPEN_CHEMLIB_MODULE_BYTES,
        ).buffer,
      }),
      cryptoImpl: {
        subtle: {
          digest: async () => new Uint8Array(32).buffer,
        },
      },
      message: /SHA-256 mismatch/,
    },
  ]) {
    await assert.rejects(
      loadOpenChemLibRuntime(["jsdelivr"], {
        fetchImpl: fixture.fetchImpl,
        cryptoImpl: fixture.cryptoImpl,
      }),
      (error) => {
        assert.equal(error instanceof AggregateError, true, fixture.name);
        assert.equal(error.errors.length, 2, fixture.name);
        assert.equal(error.errors.every((inner) => fixture.message.test(inner.message)), true);
        return true;
      },
    );
  }
});

test("OpenChemLib rejects incompatible modules and malformed resource registries", async () => {
  const validResources = new TextEncoder().encode(JSON.stringify({ value: 17 })).buffer;
  const cases = [
    {
      name: "version",
      OCL: { version: "0.0.0", Resources: { register() {} } },
      resourcesBytes: validResources,
      message: /unexpected OpenChemLib version/,
    },
    {
      name: "registry API",
      OCL: { version: OPEN_CHEMLIB_VERSION, Resources: {} },
      resourcesBytes: validResources,
      message: /resource registry API is unavailable/,
    },
    {
      name: "resource JSON",
      OCL: { version: OPEN_CHEMLIB_VERSION, Resources: { register() {} } },
      resourcesBytes: new TextEncoder().encode("not JSON").buffer,
      message: /JSON/,
    },
  ];

  for (const fixture of cases) {
    let revocations = 0;
    await assert.rejects(
      loadOpenChemLibRuntime(["jsdmirror"], {
        fetchImpl: async () => {
          throw new Error("injected fetch helper should be used");
        },
        fetchPinnedBytesImpl: async (url) => url.endsWith("resources.json")
          ? fixture.resourcesBytes
          : new Uint8Array([1]).buffer,
        importModule: async () => fixture.OCL,
        createModuleUrl: () => ({
          url: "blob:verified-openchemlib",
          revoke: () => { revocations += 1; },
        }),
      }),
      (error) => {
        assert.equal(error instanceof AggregateError, true, fixture.name);
        assert.equal(error.errors.length, 2, fixture.name);
        assert.equal(error.errors.every((inner) => fixture.message.test(inner.message)), true);
        return true;
      },
    );
    assert.equal(revocations, 2, fixture.name);
  }
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
