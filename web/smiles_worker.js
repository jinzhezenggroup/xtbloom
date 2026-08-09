/* Optional SMILES/conformer work lives in its own module Worker so downloading
 * OpenChemLib and running conformer/MMFF searches never blocks the UI or the
 * independent gpuxtb WASM worker. */

import {
  OPEN_CHEMLIB_MODULE_URL,
  OPEN_CHEMLIB_RESOURCES_URL,
  smilesToGeometry,
} from "./smiles_helpers.js";

let OCL = null;

async function initialize() {
  try {
    OCL = await import(OPEN_CHEMLIB_MODULE_URL);
    if (String(OCL.version) !== "9.21.0") {
      throw new Error(`unexpected OpenChemLib version ${String(OCL.version)}`);
    }
    await OCL.Resources.registerFromUrl(OPEN_CHEMLIB_RESOURCES_URL);
    self.postMessage({ type: "ready", version: OCL.version });
  } catch (error) {
    self.postMessage({
      type: "load-error",
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

self.onmessage = (event) => {
  const message = event.data;
  if (!message || message.type !== "generate") return;
  if (!OCL) {
    self.postMessage({
      type: "result",
      id: message.id,
      ok: false,
      errorCode: "smiles_err_library",
      error: "OpenChemLib is not ready",
    });
    return;
  }
  try {
    const result = smilesToGeometry(OCL, message.smiles);
    self.postMessage({ type: "result", id: message.id, ok: true, result });
  } catch (error) {
    self.postMessage({
      type: "result",
      id: message.id,
      ok: false,
      errorCode: error && error.code ? error.code : "smiles_err_unknown",
      error: error instanceof Error ? error.message : String(error),
    });
  }
};

void initialize();
