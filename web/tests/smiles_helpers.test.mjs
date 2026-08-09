import assert from "node:assert/strict";
import test from "node:test";

import {
  OPEN_CHEMLIB_MODULE_URL,
  OPEN_CHEMLIB_RESOURCES_URL,
  OPEN_CHEMLIB_VERSION,
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

test("OpenChemLib runtime URLs pin the reviewed static jsDelivr release", () => {
  assert.equal(OPEN_CHEMLIB_VERSION, "9.21.0");
  assert.match(OPEN_CHEMLIB_MODULE_URL, /openchemlib@9\.21\.0\/dist\/openchemlib\.js$/);
  assert.match(OPEN_CHEMLIB_RESOURCES_URL, /openchemlib@9\.21\.0\/dist\/resources\.json$/);
  assert.doesNotMatch(OPEN_CHEMLIB_MODULE_URL, /latest|\+esm/);
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
