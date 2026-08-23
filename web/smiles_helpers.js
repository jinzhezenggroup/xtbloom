/* OpenChemLib browser integration kept DOM-free so conversion and CDN
 * failover semantics are testable under Node. Both providers serve the exact
 * reviewed npm release bytes; do not replace these URLs with latest/+esm. */

export const OPEN_CHEMLIB_VERSION = "9.21.0";
export const CDN_REGION_MAINLAND_CHINA = "mainland-china";
export const CDN_REGION_GLOBAL = "global";
export const OPEN_CHEMLIB_CDN_URLS = Object.freeze({
  jsdelivr: Object.freeze({
    module: "https://cdn.jsdelivr.net/npm/openchemlib@9.21.0/dist/openchemlib.js",
    resources: "https://cdn.jsdelivr.net/npm/openchemlib@9.21.0/dist/resources.json",
  }),
  jsdmirror: Object.freeze({
    module: "https://cdn.jsdmirror.com/npm/openchemlib@9.21.0/dist/openchemlib.js",
    resources: "https://cdn.jsdmirror.com/npm/openchemlib@9.21.0/dist/resources.json",
  }),
});
export const OPEN_CHEMLIB_MODULE_SHA256 =
  "5978967b12e938208e8d36222370f88fd615a2b5ec83f02e435caab26f3f4cb3";
export const OPEN_CHEMLIB_MODULE_BYTES = 1097449;
export const OPEN_CHEMLIB_RESOURCES_SHA256 =
  "d2741130d5a5546aeebebc43eb3dac937881b04755fefe5925e4b228a56bee14";
export const OPEN_CHEMLIB_RESOURCES_BYTES = 1351963;
export const MAX_SMILES_LENGTH = 2048;
export const MAX_WEB_ATOMS = 512;

/* The page bootstrap resolves the visitor region before creating the Worker.
 * Treat an absent or unknown value as global so overseas users keep jsDelivr
 * as the normal primary provider. */
export function cdnProviderOrder(region) {
  return region === CDN_REGION_MAINLAND_CHINA
    ? ["jsdmirror", "jsdelivr"]
    : ["jsdelivr", "jsdmirror"];
}

export function normalizeCdnProviderOrder(providers, region = CDN_REGION_GLOBAL) {
  const preferred = Array.isArray(providers) ? providers : [];
  return Array.from(new Set([
    ...preferred.filter((provider) => provider in OPEN_CHEMLIB_CDN_URLS),
    ...cdnProviderOrder(region),
  ]));
}

export function openChemLibUrlsForProviders(providers, region = CDN_REGION_GLOBAL) {
  const orderedProviders = normalizeCdnProviderOrder(providers, region);
  return {
    providers: orderedProviders,
    modules: orderedProviders.map((provider) => OPEN_CHEMLIB_CDN_URLS[provider].module),
    resources: orderedProviders.map((provider) => OPEN_CHEMLIB_CDN_URLS[provider].resources),
  };
}

async function sha256Hex(bytes, cryptoImpl) {
  if (!cryptoImpl?.subtle?.digest) throw new Error("SHA-256 verification is unavailable");
  const digest = new Uint8Array(await cryptoImpl.subtle.digest("SHA-256", bytes));
  return Array.from(digest, (value) => value.toString(16).padStart(2, "0")).join("");
}

async function fetchPinnedBytes(url, expectedBytes, expectedSha256, {
  fetchImpl,
  cryptoImpl,
  signal,
}) {
  const response = await fetchImpl(url, { cache: "default", signal });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength !== expectedBytes) {
    throw new Error(`${url}: expected ${expectedBytes} bytes, received ${bytes.byteLength}`);
  }
  if (await sha256Hex(bytes, cryptoImpl) !== expectedSha256) {
    throw new Error(`${url}: SHA-256 mismatch`);
  }
  return bytes;
}

/* Treat the module and resource registry as one provider generation. This
 * prevents a partial or modified mirror response from being mixed with the
 * other provider while still allowing the complete pinned pair to fail over. */
export async function loadOpenChemLibRuntime(providers, {
  region = CDN_REGION_GLOBAL,
  importModule = (url) => import(url),
  fetchImpl = globalThis.fetch,
  cryptoImpl = globalThis.crypto,
  attemptTimeoutMs = 20000,
  fetchPinnedBytesImpl = fetchPinnedBytes,
  createModuleUrl = (bytes) => {
    const url = URL.createObjectURL(new Blob([bytes], { type: "text/javascript" }));
    return { url, revoke: () => URL.revokeObjectURL(url) };
  },
} = {}) {
  if (typeof fetchImpl !== "function") throw new TypeError("fetch is unavailable");
  const urls = openChemLibUrlsForProviders(providers, region);
  const errors = [];
  for (let index = 0; index < urls.providers.length; index += 1) {
    const provider = urls.providers[index];
    const moduleUrl = urls.modules[index];
    const resourcesUrl = urls.resources[index];
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), attemptTimeoutMs);
    try {
      const [moduleBytes, resourcesBytes] = await Promise.all([
        fetchPinnedBytesImpl(
          moduleUrl,
          OPEN_CHEMLIB_MODULE_BYTES,
          OPEN_CHEMLIB_MODULE_SHA256,
          { fetchImpl, cryptoImpl, signal: controller.signal },
        ),
        fetchPinnedBytesImpl(
          resourcesUrl,
          OPEN_CHEMLIB_RESOURCES_BYTES,
          OPEN_CHEMLIB_RESOURCES_SHA256,
          { fetchImpl, cryptoImpl, signal: controller.signal },
        ),
      ]);
      const verifiedModule = createModuleUrl(moduleBytes);
      let OCL;
      try {
        OCL = await importModule(verifiedModule.url);
      } finally {
        verifiedModule.revoke();
      }
      if (String(OCL?.version) !== OPEN_CHEMLIB_VERSION) {
        throw new Error(`unexpected OpenChemLib version ${String(OCL?.version)}`);
      }
      if (typeof OCL.Resources?.register !== "function") {
        throw new Error("OpenChemLib resource registry API is unavailable");
      }
      const resources = JSON.parse(new TextDecoder().decode(resourcesBytes));
      OCL.Resources.register(resources);
      return { OCL, provider, moduleUrl, resourcesUrl };
    } catch (error) {
      errors.push(error instanceof Error ? error : new Error(String(error)));
    } finally {
      clearTimeout(timer);
      controller.abort();
    }
  }
  throw new AggregateError(errors, "OpenChemLib failed from every pinned provider");
}

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
