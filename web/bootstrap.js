/* Minimal unversioned entry point for the browser demo.
 *
 * Keep this file dependency-free: it runs pinned 3Dmol source ranking and its
 * verified loader alongside content-manifest revalidation, warms and verifies
 * the versioned application module graph, and only then imports app.js. That
 * prevents a deployment from linking a new app.js against stale helpers or
 * preset data before the normal retry UI can start.
 */

const BOOTSTRAP_MAX_ATTEMPTS = 3;
const BOOTSTRAP_ATTEMPT_TIMEOUT_MS = 60000;
const CDN_PROBE_BYTES = 65536;
const CDN_PROBE_TIMEOUT_MS = 2500;
const THREE_DMOL_ATTEMPT_TIMEOUT_MS = 30000;
const THREE_DMOL_BYTES = 537792;
const THREE_DMOL_SHA256 =
  "f7cc78921ae72e7623e89cdd111434f58c2efddd2ffda1cd212644b406fb8016";
const MAINLAND_TIME_ZONES = new Set([
  "Asia/Shanghai",
  "Asia/Urumqi",
  "Asia/Chongqing",
  "Asia/Chungking",
  "Asia/Harbin",
  "Asia/Kashgar",
  "PRC",
]);
export const THREE_DMOL_SOURCES = Object.freeze([
  Object.freeze({
    id: "jsdelivr",
    url: "https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js",
  }),
  Object.freeze({
    id: "jsdmirror",
    url: "https://cdn.jsdmirror.com/npm/3dmol@2.5.5/build/3Dmol-min.js",
  }),
  Object.freeze({ id: "local", url: "vendor/3Dmol-min.js" }),
]);
const bootstrapLoaderToken = new URL(import.meta.url).searchParams.get("xtbloom_bootstrap_loader");
let bootstrapGeneration = 0;
let bootstrapController = null;

export function cdnRegionForTimeZone(timeZone) {
  return MAINLAND_TIME_ZONES.has(String(timeZone || "")) ? "mainland-china" : "global";
}

function currentTimeZone(intlImpl = globalThis.Intl) {
  try {
    return intlImpl?.DateTimeFormat?.().resolvedOptions?.().timeZone || "";
  } catch {
    return "";
  }
}

function regionalSourceIds(region) {
  return region === "mainland-china"
    ? ["jsdmirror", "jsdelivr", "local"]
    : ["jsdelivr", "jsdmirror", "local"];
}

/* Probe only an initial byte window from the real pinned asset. Cancelling the
 * body after that window avoids downloading every full candidate merely to
 * choose one, while still measuring DNS, connection, TTFB, and early transfer. */
export async function probeSourceSpeed(source, {
  baseUrl = globalThis.document?.baseURI,
  fetchImpl = globalThis.fetch,
  now = () => globalThis.performance?.now?.() ?? Date.now(),
  probeBytes = CDN_PROBE_BYTES,
  timeoutMs = CDN_PROBE_TIMEOUT_MS,
  } = {}) {
  if (typeof fetchImpl !== "function") throw new TypeError("fetch is unavailable");
  const url = new URL(source.url, baseUrl).href;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const startedAt = now();
  let reader = null;
  try {
    const response = await fetchImpl(url, {
      cache: "no-store",
      headers: { Range: `bytes=0-${probeBytes - 1}` },
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    reader = response.body?.getReader?.();
    if (!reader) throw new Error(`${url}: streaming response is unavailable`);
    let receivedBytes = 0;
    while (receivedBytes < probeBytes) {
      const { done, value } = await reader.read();
      if (done) break;
      receivedBytes += value?.byteLength || 0;
    }
    if (receivedBytes === 0) throw new Error(`${url}: empty probe response`);
    return { id: source.id, url, elapsedMs: Math.max(0.001, now() - startedAt) };
  } finally {
    clearTimeout(timer);
    await reader?.cancel?.().catch(() => {});
    controller.abort();
  }
}

export async function rankCdnSources(sources, {
  region = cdnRegionForTimeZone(currentTimeZone()),
  probeImpl = probeSourceSpeed,
} = {}) {
  const fallbackIds = regionalSourceIds(region);
  const fallbackRank = new Map(fallbackIds.map((id, index) => [id, index]));
  const results = await Promise.all(sources.map(async (source) => {
    try {
      return { source, measurement: await probeImpl(source) };
    } catch (error) {
      return { source, measurement: null, error };
    }
  }));
  const measured = results
    .filter((result) => Number.isFinite(result.measurement?.elapsedMs))
    .sort((left, right) => left.measurement.elapsedMs - right.measurement.elapsedMs);
  const failed = results
    .filter((result) => !Number.isFinite(result.measurement?.elapsedMs))
    .sort((left, right) =>
      (fallbackRank.get(left.source.id) ?? 99) -
      (fallbackRank.get(right.source.id) ?? 99));
  const ranked = [];
  while (measured.length > 0) {
    const fastestTime = measured[0].measurement.elapsedMs;
    const tieWindow = Math.max(20, fastestTime * 0.15);
    let closeCount = 1;
    while (
      closeCount < measured.length &&
      measured[closeCount].measurement.elapsedMs - fastestTime <= tieWindow
    ) {
      closeCount += 1;
    }
    const closeGroup = measured.splice(0, closeCount);
    closeGroup.sort((left, right) =>
      (fallbackRank.get(left.source.id) ?? 99) -
      (fallbackRank.get(right.source.id) ?? 99));
    ranked.push(...closeGroup);
  }
  return [...ranked, ...failed];
}

async function fetchVerified3Dmol(url, {
  fetchImpl,
  cryptoImpl,
  timeoutMs = THREE_DMOL_ATTEMPT_TIMEOUT_MS,
}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetchImpl(url, { cache: "default", signal: controller.signal });
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    const bytes = await response.arrayBuffer();
    if (bytes.byteLength !== THREE_DMOL_BYTES) {
      throw new Error(`${url}: expected ${THREE_DMOL_BYTES} bytes, received ${bytes.byteLength}`);
    }
    if (await sha256Hex(bytes, cryptoImpl) !== THREE_DMOL_SHA256) {
      throw new Error(`${url}: SHA-256 mismatch`);
    }
    return bytes;
  } finally {
    clearTimeout(timer);
    controller.abort();
  }
}

async function executeClassicScript(bytes, {
  documentImpl,
  urlImpl = globalThis.URL,
}) {
  const blobUrl = urlImpl.createObjectURL(new Blob([bytes], { type: "text/javascript" }));
  try {
    await new Promise((resolve, reject) => {
      const script = documentImpl.createElement("script");
      script.async = true;
      script.src = blobUrl;
      script.onload = resolve;
      script.onerror = () => reject(new Error("3Dmol script execution failed"));
      documentImpl.head.appendChild(script);
    });
  } finally {
    urlImpl.revokeObjectURL(blobUrl);
  }
}

export async function loadThreeDmol(rankedSources, {
  baseUrl = globalThis.document?.baseURI,
  documentImpl = globalThis.document,
  fetchImpl = globalThis.fetch,
  cryptoImpl = globalThis.crypto,
  globalImpl = globalThis,
  fetchBytesImpl = fetchVerified3Dmol,
  executeScriptImpl = executeClassicScript,
} = {}) {
  if (globalImpl.$3Dmol) return { source: "existing" };
  const errors = [];
  for (const ranked of rankedSources) {
    const source = ranked.source || ranked;
    const url = new URL(source.url, baseUrl).href;
    try {
      const bytes = await fetchBytesImpl(url, { fetchImpl, cryptoImpl });
      await executeScriptImpl(bytes, { documentImpl });
      if (!globalImpl.$3Dmol) throw new Error(`${url}: 3Dmol global is unavailable`);
      return { source: source.id, url };
    } catch (error) {
      errors.push(error instanceof Error ? error : new Error(String(error)));
    }
  }
  throw new AggregateError(errors, "3Dmol failed from every verified source");
}

export async function initializeBrowserCdnRouting({
  intlImpl = globalThis.Intl,
  globalImpl = globalThis,
  sources = THREE_DMOL_SOURCES,
  rankImpl = rankCdnSources,
  loadThreeDmolImpl = loadThreeDmol,
} = {}) {
  const region = cdnRegionForTimeZone(currentTimeZone(intlImpl));
  let rankedSources;
  try {
    rankedSources = await rankImpl(sources, { region });
  } catch {
    const fallbackRank = new Map(regionalSourceIds(region).map((id, index) => [id, index]));
    rankedSources = sources
      .map((source) => ({ source, measurement: null }))
      .sort((left, right) =>
        (fallbackRank.get(left.source.id) ?? 99) -
        (fallbackRank.get(right.source.id) ?? 99));
  }
  const providers = rankedSources
    .map((ranked) => (ranked.source || ranked).id)
    .filter((id) => id === "jsdelivr" || id === "jsdmirror");
  globalImpl.__XTBLOOM_CDN_REGION = region;
  globalImpl.__XTBLOOM_CDN_PROVIDERS = providers;
  const ready = Promise.resolve()
    .then(() => loadThreeDmolImpl(rankedSources))
    .then((result) => ({ ok: true, ...result }))
    .catch((error) => ({ ok: false, error }));
  globalImpl.__XTBLOOM_3DMOL_READY = ready;
  return { region, providers, rankedSources, ready };
}

/* CDN probing is optional adapter work. Publish its promise immediately so
 * 3Dmol and the SMILES Worker can wait for the measured order while the core
 * application and WASM engine begin loading in parallel. */
export function startBrowserCdnAndApplication({
  globalImpl = globalThis,
  initializeRouting = initializeBrowserCdnRouting,
  startApplication = startBrowserApplication,
} = {}) {
  const routing = Promise.resolve()
    .then(() => initializeRouting({ globalImpl }))
    .catch((error) => {
      const region = "global";
      const providers = regionalSourceIds(region).filter((id) => id !== "local");
      const ready = Promise.resolve({ ok: false, error });
      globalImpl.__XTBLOOM_CDN_REGION = region;
      globalImpl.__XTBLOOM_CDN_PROVIDERS = providers;
      globalImpl.__XTBLOOM_3DMOL_READY = ready;
      return { region, providers, rankedSources: [], ready };
    });
  globalImpl.__XTBLOOM_CDN_ROUTING = routing;
  let applicationStart;
  try {
    applicationStart = Promise.resolve(startApplication());
  } catch (error) {
    applicationStart = Promise.reject(error);
  }
  globalImpl.__XTBLOOM_APPLICATION_START = applicationStart.catch((error) => {
    globalImpl.console?.error?.("xTBloom application startup failed", error);
    return { ok: false, error };
  });
  return routing;
}

function bootstrapError(message, extras = {}) {
  const error = new Error(message);
  Object.assign(error, extras);
  return error;
}

function isRetryableBootstrapError(error) {
  if (!error) return false;
  if (typeof error.retryable === "boolean") return error.retryable;
  const status = Number(error.status);
  if (status) return status === 408 || status === 425 || status === 429 || status >= 500;
  return error.name === "TypeError" || /network|load failed|timed? out/i.test(String(error.message));
}

function retryDelayMs(failedAttempt) {
  return Math.min(4000, 500 * (2 ** Math.max(0, failedAttempt - 1)));
}

function delayWithSignal(delayMs, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, delayMs);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function validateManifestAsset(asset, expectedId) {
  const id = String(asset?.id || "");
  const path = String(asset?.path || "");
  const bytes = Number(asset?.bytes);
  const sha256 = String(asset?.sha256 || "").toLowerCase();
  if (id !== expectedId) throw new TypeError(`engine manifest is missing ${expectedId}`);
  if (!/^[A-Za-z0-9_.-]+$/.test(path) || path === "." || path === "..") {
    throw new TypeError(`engine manifest has an unsafe path for ${expectedId}`);
  }
  if (!Number.isSafeInteger(bytes) || bytes <= 0) {
    throw new TypeError(`engine manifest has an invalid size for ${expectedId}`);
  }
  if (!/^[0-9a-f]{64}$/.test(sha256)) {
    throw new TypeError(`engine manifest has an invalid digest for ${expectedId}`);
  }
  return { id, path, bytes, sha256 };
}

export function validateBootstrapManifest(manifest) {
  if (!manifest || manifest.schema_version !== 1) {
    throw new TypeError("unsupported engine manifest schema");
  }
  const version = String(manifest.version || "").toLowerCase();
  if (!/^[0-9a-f]{64}$/.test(version) || !Array.isArray(manifest.assets)) {
    throw new TypeError("invalid engine manifest");
  }
  const byId = new Map(manifest.assets.map((asset) => [String(asset?.id || ""), asset]));
  return {
    manifest,
    version,
    app: validateManifestAsset(byId.get("app"), "app"),
    c60: validateManifestAsset(byId.get("c60"), "c60"),
    helpers: validateManifestAsset(byId.get("helpers"), "helpers"),
  };
}

function versionedAssetUrl(baseUrl, asset, version, bootstrapToken) {
  const url = new URL(asset.path, baseUrl);
  url.searchParams.set("xtbloom_version", version);
  if (bootstrapToken) url.searchParams.set("xtbloom_bootstrap", bootstrapToken);
  return url;
}

async function sha256Hex(bytes, cryptoImpl) {
  if (!cryptoImpl?.subtle?.digest) {
    throw bootstrapError("SHA-256 verification is unavailable", { retryable: false });
  }
  const digest = new Uint8Array(await cryptoImpl.subtle.digest("SHA-256", bytes));
  return Array.from(digest, (value) => value.toString(16).padStart(2, "0")).join("");
}

async function fetchVerifiedAsset(asset, url, { fetchImpl, cryptoImpl, signal, cache }) {
  let response;
  try {
    response = await fetchImpl(url, { signal, cache });
  } catch (cause) {
    throw bootstrapError(`Network error while loading ${asset.id}`, {
      cause,
      retryable: true,
    });
  }
  if (!response.ok) {
    const status = Number(response.status) || 0;
    throw bootstrapError(`HTTP ${status} while loading ${asset.id}`, {
      status,
      retryable: status === 408 || status === 425 || status === 429 || status >= 500,
    });
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength !== asset.bytes) {
    throw bootstrapError(`Incomplete response while loading ${asset.id}`, { retryable: true });
  }
  if (await sha256Hex(bytes, cryptoImpl) !== asset.sha256) {
    throw bootstrapError(`Digest mismatch while loading ${asset.id}`, { retryable: true });
  }
}

async function loadBootstrapAttempt({
  baseUrl,
  bootstrapToken,
  fetchImpl,
  cryptoImpl,
  signal,
  cache,
}) {
  const manifestUrl = new URL("engine-manifest.json", baseUrl);
  let response;
  try {
    response = await fetchImpl(manifestUrl, { signal, cache: cache === "reload" ? "reload" : "no-cache" });
  } catch (cause) {
    throw bootstrapError("Network error while checking the engine manifest", {
      cause,
      retryable: true,
    });
  }
  if (!response.ok) {
    const status = Number(response.status) || 0;
    throw bootstrapError(`HTTP ${status} while checking the engine manifest`, {
      status,
      retryable: status === 408 || status === 425 || status === 429 || status >= 500,
    });
  }

  let rawManifest;
  try {
    rawManifest = await response.json();
  } catch (cause) {
    throw bootstrapError("Invalid engine manifest response", { cause, retryable: true });
  }
  const validated = validateBootstrapManifest(rawManifest);
  const appUrl = versionedAssetUrl(baseUrl, validated.app, validated.version, bootstrapToken);
  const c60Url = versionedAssetUrl(baseUrl, validated.c60, validated.version, bootstrapToken);
  const helpersUrl = versionedAssetUrl(
    baseUrl,
    validated.helpers,
    validated.version,
    bootstrapToken,
  );
  await Promise.all([
    fetchVerifiedAsset(validated.app, appUrl, { fetchImpl, cryptoImpl, signal, cache }),
    fetchVerifiedAsset(validated.c60, c60Url, { fetchImpl, cryptoImpl, signal, cache }),
    fetchVerifiedAsset(validated.helpers, helpersUrl, { fetchImpl, cryptoImpl, signal, cache }),
  ]);
  return { manifest: rawManifest, appUrl, c60Url, helpersUrl };
}

function withAttemptTimeout(operation, timeoutMs, controller, onTimeout = () => {}) {
  let timer = null;
  let timedOut = false;
  let onAbort = null;
  const aborted = new Promise((_, reject) => {
    onAbort = () => {
      if (!timedOut) reject(new DOMException("Bootstrap attempt aborted", "AbortError"));
    };
    if (controller.signal.aborted) onAbort();
    else controller.signal.addEventListener("abort", onAbort, { once: true });
  });
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      timedOut = true;
      onTimeout();
      reject(bootstrapError("Engine bootstrap timed out", { retryable: true }));
      controller.abort();
    }, timeoutMs);
  });
  return Promise.race([operation, timeout, aborted]).finally(() => {
    clearTimeout(timer);
    controller.signal.removeEventListener("abort", onAbort);
  });
}

export async function prepareVersionedApplication({
  baseUrl = import.meta.url,
  bootstrapToken = "",
  forceReload = false,
  fetchImpl = globalThis.fetch,
  cryptoImpl = globalThis.crypto,
  signal,
  maxAttempts = BOOTSTRAP_MAX_ATTEMPTS,
  timeoutMs = BOOTSTRAP_ATTEMPT_TIMEOUT_MS,
  onRetry = () => {},
  delay = delayWithSignal,
  loadApplication = false,
  applicationTokenPrefix = "app",
  importImpl = (url) => import(url),
} = {}) {
  if (typeof fetchImpl !== "function") throw new TypeError("fetch is unavailable");
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const attemptController = new AbortController();
    const attemptToken = loadApplication
      ? `${applicationTokenPrefix}-${attempt}`
      : bootstrapToken;
    const invalidateApplicationAttempt = () => {
      if (loadApplication && globalThis.__XTBLOOM_APP_BOOT_TOKEN === attemptToken) {
        globalThis.__XTBLOOM_APP_BOOT_TOKEN = `expired:${attemptToken}`;
      }
    };
    const abortAttempt = () => {
      invalidateApplicationAttempt();
      attemptController.abort();
    };
    signal?.addEventListener("abort", abortAttempt, { once: true });
    if (signal?.aborted) abortAttempt();
    try {
      const cache = forceReload || attempt > 1 ? "reload" : "default";
      const operation = (async () => {
        const prepared = await loadBootstrapAttempt({
          baseUrl,
          bootstrapToken: attemptToken,
          fetchImpl,
          cryptoImpl,
          signal: attemptController.signal,
          cache,
        });
        if (loadApplication) {
          if (attemptController.signal.aborted) {
            throw new DOMException("Application import superseded", "AbortError");
          }
          globalThis.__XTBLOOM_BOOTSTRAP_MANIFEST = prepared.manifest;
          globalThis.__XTBLOOM_APP_BOOT_TOKEN = attemptToken;
          await importImpl(prepared.appUrl.href);
          if (
            attemptController.signal.aborted ||
            globalThis.__XTBLOOM_APP_BOOT_TOKEN !== attemptToken
          ) {
            throw new DOMException("Application import superseded", "AbortError");
          }
        }
        return prepared;
      })();
      return await withAttemptTimeout(
        operation,
        timeoutMs,
        attemptController,
        invalidateApplicationAttempt,
      );
    } catch (error) {
      invalidateApplicationAttempt();
      if (signal?.aborted || attempt >= maxAttempts || !isRetryableBootstrapError(error)) {
        throw error;
      }
      const waitMs = retryDelayMs(attempt);
      onRetry({ nextAttempt: attempt + 1, maxAttempts, waitMs });
      await delay(waitMs, signal);
    } finally {
      signal?.removeEventListener("abort", abortAttempt);
      attemptController.abort();
    }
  }
  throw new Error("bootstrap retry loop exhausted");
}

function bootstrapText(zh, en, navigatorImpl = globalThis.navigator) {
  return String(navigatorImpl?.language || "").toLowerCase().startsWith("zh") ? zh : en;
}

/* Browser globals and the verified-application preparer are injectable so the
 * page-owned recovery UI can be executed under Node; production callers use
 * the defaults and retain the same one-generation-at-a-time lifecycle. */
export async function startBrowserApplication({
  forceReload = false,
  documentImpl = globalThis.document,
  navigatorImpl = globalThis.navigator,
  prepareApplication = prepareVersionedApplication,
} = {}) {
  const generation = ++bootstrapGeneration;
  bootstrapController?.abort();
  const controller = new AbortController();
  bootstrapController = controller;

  const overlay = documentImpl.getElementById("overlay");
  const overlayText = documentImpl.getElementById("overlay-text");
  const loadBar = documentImpl.getElementById("load-bar-wrap");
  const errorBox = documentImpl.getElementById("error");
  const retryButton = documentImpl.getElementById("retry");
  overlay.hidden = false;
  overlayText.textContent = bootstrapText(
    "正在检查引擎更新…",
    "Checking engine updates…",
    navigatorImpl,
  );
  loadBar.hidden = true;
  errorBox.hidden = true;
  retryButton.hidden = true;

  try {
    await prepareApplication({
      forceReload,
      signal: controller.signal,
      loadApplication: true,
      applicationTokenPrefix: String(generation),
      onRetry: ({ nextAttempt, maxAttempts, waitMs }) => {
        if (generation !== bootstrapGeneration || controller.signal.aborted) return;
        overlayText.textContent = bootstrapText(
          `网络暂时不可用，${Math.ceil(waitMs / 1000)} 秒后重试（${nextAttempt}/${maxAttempts}）…`,
          `Network temporarily unavailable; retrying in ${Math.ceil(waitMs / 1000)} s (${nextAttempt}/${maxAttempts})…`,
          navigatorImpl,
        );
      },
    });
    if (generation !== bootstrapGeneration || controller.signal.aborted) return;

    retryButton.onclick = null;
  } catch (error) {
    if (generation !== bootstrapGeneration || controller.signal.aborted) return;
    overlay.hidden = true;
    errorBox.textContent = bootstrapText(
      `引擎启动文件加载失败：${String(error?.message || error)}`,
      `Failed to load the engine startup files: ${String(error?.message || error)}`,
      navigatorImpl,
    );
    errorBox.hidden = false;
    retryButton.hidden = false;
    retryButton.onclick = () => startBrowserApplication({
      forceReload: true,
      documentImpl,
      navigatorImpl,
      prepareApplication,
    });
  } finally {
    if (generation === bootstrapGeneration && bootstrapController === controller) {
      bootstrapController = null;
    }
  }
}

if (
  typeof document !== "undefined" &&
  (!bootstrapLoaderToken || globalThis.__XTBLOOM_BOOTSTRAP_LOADER_TOKEN === bootstrapLoaderToken)
) {
  void startBrowserCdnAndApplication({
    startApplication: () => {
      if (
        !bootstrapLoaderToken ||
        globalThis.__XTBLOOM_BOOTSTRAP_LOADER_TOKEN === bootstrapLoaderToken
      ) {
        return startBrowserApplication();
      }
      return undefined;
    },
  });
}
