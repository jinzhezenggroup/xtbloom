/* Optional SMILES/conformer work lives in its own module Worker so downloading
 * OpenChemLib and running conformer/MMFF searches never blocks the UI or the
 * independent xtbloom WASM worker. */

let OCL = null;
let smilesToGeometryImpl = null;
const workerModuleUrl = new URL(import.meta.url);
const requestedRegion = workerModuleUrl.searchParams.get("xtbloom_cdn_region");
const requestedProviders = workerModuleUrl.searchParams
  .get("xtbloom_cdn_providers")
  ?.split(",");

/* A relative static import would discard the Worker's query string and could
 * therefore link a newly deployed Worker against a four-hour cached helper.
 * Propagate the verified manifest generation explicitly so both modules are
 * fetched and cached as one immutable generation. Per-attempt bootstrap
 * tokens are deliberately excluded so successful content remains reusable. */
const smilesHelpersUrl = new URL("./smiles_helpers.js", workerModuleUrl);
const contentVersion = workerModuleUrl.searchParams.get("xtbloom_version");
const hasValidContentVersion = /^[0-9a-f]{64}$/.test(contentVersion || "");
if (hasValidContentVersion) {
  smilesHelpersUrl.searchParams.set("xtbloom_version", contentVersion);
}

async function initialize() {
  try {
    if (!hasValidContentVersion) {
      throw new TypeError("SMILES Worker requires a 64-character SHA-256 content version");
    }
    const helpers = await import(smilesHelpersUrl.href);
    const loadOpenChemLibRuntime = helpers?.loadOpenChemLibRuntime;
    smilesToGeometryImpl = helpers?.smilesToGeometry;
    if (
      typeof loadOpenChemLibRuntime !== "function" ||
      typeof smilesToGeometryImpl !== "function"
    ) {
      throw new TypeError("invalid SMILES helper exports");
    }
    const cdnRegion = requestedRegion === helpers.CDN_REGION_MAINLAND_CHINA
      ? helpers.CDN_REGION_MAINLAND_CHINA
      : helpers.CDN_REGION_GLOBAL;
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
    const result = smilesToGeometryImpl(OCL, message.smiles);
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
