/* Optional SMILES/conformer work lives in its own module Worker so downloading
 * OpenChemLib and running conformer/MMFF searches never blocks the UI or the
 * independent xtbloom WASM worker. */

import {
  CDN_REGION_GLOBAL,
  CDN_REGION_MAINLAND_CHINA,
  loadOpenChemLibRuntime,
  smilesToGeometry,
} from "./smiles_helpers.js";

let OCL = null;
const requestedRegion = new URL(import.meta.url).searchParams.get("xtbloom_cdn_region");
const cdnRegion = requestedRegion === CDN_REGION_MAINLAND_CHINA
  ? CDN_REGION_MAINLAND_CHINA
  : CDN_REGION_GLOBAL;
const requestedProviders = new URL(import.meta.url).searchParams
  .get("xtbloom_cdn_providers")
  ?.split(",");

async function initialize() {
  try {
    const runtime = await loadOpenChemLibRuntime(requestedProviders, { region: cdnRegion });
    OCL = runtime.OCL;
    self.postMessage({
      type: "ready",
      version: OCL.version,
      moduleUrl: runtime.moduleUrl,
      resourcesUrl: runtime.resourcesUrl,
    });
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
