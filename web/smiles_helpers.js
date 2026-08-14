/* OpenChemLib browser integration kept DOM-free so conversion semantics are
 * testable under Node. The exact jsDelivr package version is part of the
 * deployed dependency contract; do not replace these URLs with latest/+esm. */

export const OPEN_CHEMLIB_VERSION = "9.21.0";
export const OPEN_CHEMLIB_MODULE_URL =
  "https://cdn.jsdelivr.net/npm/openchemlib@9.21.0/dist/openchemlib.js";
export const OPEN_CHEMLIB_RESOURCES_URL =
  "https://cdn.jsdelivr.net/npm/openchemlib@9.21.0/dist/resources.json";
export const MAX_SMILES_LENGTH = 2048;
export const MAX_WEB_ATOMS = 512;

/* The browser-exposed GFN1 and GFN2 parameter sets share the same supported
 * element range (H through Rn). Keep SMILES validation model-neutral. */
const GFN_XTB_SYMBOLS = [
  "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
  "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
  "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
  "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
  "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
];

function codedError(code, message, cause) {
  const error = new Error(message, cause === undefined ? undefined : { cause });
  error.code = code;
  return error;
}

function requireApi(OCL) {
  if (
    !OCL || typeof OCL.Molecule?.fromSmiles !== "function" ||
    typeof OCL.ConformerGenerator !== "function" ||
    typeof OCL.ForceFieldMMFF94 !== "function"
  ) {
    throw codedError("smiles_err_library", "OpenChemLib API is incomplete");
  }
}

/* Convert one ordinary closed-shell SMILES into a fully explicit, MMFF94-
 * relaxed XYZ geometry in angstrom. Radical SMILES are rejected rather than
 * guessing xtbloom's unpaired-electron input. */
export function smilesToGeometry(OCL, smiles, options = {}) {
  requireApi(OCL);
  const trimmed = String(smiles || "").trim();
  const maxLength = options.maxLength ?? MAX_SMILES_LENGTH;
  const maxAtoms = options.maxAtoms ?? MAX_WEB_ATOMS;
  const seed = options.seed ?? 42;
  if (!trimmed) throw codedError("smiles_err_empty", "SMILES is empty");
  if (trimmed.length > maxLength) {
    throw codedError("smiles_err_too_long", `SMILES exceeds ${maxLength} characters`);
  }

  let molecule;
  try {
    molecule = OCL.Molecule.fromSmiles(trimmed);
  } catch (cause) {
    throw codedError("smiles_err_parse", "OpenChemLib could not parse the SMILES", cause);
  }
  /* Molecule.validate() rejects legitimate non-neutral molecules as
   * "unbalanced atom charge" in OpenChemLib 9.21.0. Parsing plus the explicit
   * structural checks below preserves ions such as ammonium and acetate. */
  const inputAtomCount = molecule.getAllAtoms();
  if (!Number.isInteger(inputAtomCount) || inputAtomCount <= 0) {
    throw codedError("smiles_err_atoms", "SMILES produced no atoms");
  }
  if (inputAtomCount > maxAtoms) {
    throw codedError("smiles_err_too_many", `SMILES produced ${inputAtomCount} atoms`);
  }
  const fragments = new Array(molecule.getAllAtoms()).fill(-1);
  if (molecule.getFragmentNumbers(fragments, false, true) !== 1) {
    throw codedError(
      "smiles_err_fragments",
      "disconnected SMILES fragments have no defined relative placement",
    );
  }
  for (let atom = 0; atom < inputAtomCount; atom += 1) {
    const atomicNumber = molecule.getAtomicNo(atom);
    if (!Number.isInteger(atomicNumber) || atomicNumber < 1 || atomicNumber >= GFN_XTB_SYMBOLS.length) {
      throw codedError(
        "smiles_err_element",
        `GFN-xTB does not support atomic number ${atomicNumber}`,
      );
    }
    if (molecule.getAtomRadical(atom) !== 0) {
      throw codedError(
        "smiles_err_radical",
        "radical SMILES require an explicit unpaired-electron count",
      );
    }
  }

  let conformer;
  try {
    conformer = new OCL.ConformerGenerator(seed).getOneConformerAsMolecule(molecule);
  } catch (cause) {
    throw codedError("smiles_err_conformer", "OpenChemLib conformer generation failed", cause);
  }
  if (!conformer) {
    throw codedError("smiles_err_conformer", "OpenChemLib returned no 3D conformer");
  }

  const atomCount = conformer.getAllAtoms();
  if (!Number.isInteger(atomCount) || atomCount <= 0) {
    throw codedError("smiles_err_atoms", "SMILES produced no atoms");
  }
  if (atomCount > maxAtoms) {
    throw codedError("smiles_err_too_many", `SMILES produced ${atomCount} atoms`);
  }

  let formalCharge = 0;
  for (let atom = 0; atom < atomCount; atom += 1) {
    const atomicNumber = conformer.getAtomicNo(atom);
    if (!Number.isInteger(atomicNumber) || atomicNumber < 1 || atomicNumber >= GFN_XTB_SYMBOLS.length) {
      throw codedError(
        "smiles_err_element",
        `GFN-xTB does not support atomic number ${atomicNumber}`,
      );
    }
    formalCharge += conformer.getAtomCharge(atom);
    if (conformer.getAtomRadical(atom) !== 0) {
      throw codedError(
        "smiles_err_radical",
        "radical SMILES require an explicit unpaired-electron count",
      );
    }
  }
  if (!Number.isSafeInteger(formalCharge)) {
    throw codedError("smiles_err_parse", "SMILES produced an invalid formal charge");
  }

  let mmffStatus;
  try {
    const forceField = new OCL.ForceFieldMMFF94(
      conformer,
      OCL.ForceFieldMMFF94.MMFF94 || "MMFF94",
    );
    mmffStatus = forceField.minimise({ maxIts: 1000, gradTol: 1e-4, funcTol: 1e-6 });
  } catch (cause) {
    throw codedError("smiles_err_mmff", "MMFF94 pre-optimization failed", cause);
  }
  if (mmffStatus !== 0) {
    throw codedError("smiles_err_mmff", `MMFF94 returned status ${mmffStatus}`);
  }

  const lines = [];
  for (let atom = 0; atom < atomCount; atom += 1) {
    const atomicNumber = conformer.getAtomicNo(atom);
    const coordinates = [
      conformer.getAtomX(atom),
      conformer.getAtomY(atom),
      conformer.getAtomZ(atom),
    ];
    if (!coordinates.every(Number.isFinite)) {
      throw codedError("smiles_err_coords", `atom ${atom + 1} has non-finite coordinates`);
    }
    lines.push(
      `${GFN_XTB_SYMBOLS[atomicNumber]} ${coordinates.map((value) => value.toFixed(10)).join(" ")}`,
    );
  }

  return {
    xyz: lines.join("\n"),
    atomCount,
    formalCharge,
    seed,
    mmffStatus,
  };
}
