/* Minimal unversioned entry point for the browser demo.
 *
 * Keep this file dependency-free: it revalidates the content manifest, warms
 * and verifies the versioned app/helper module graph, and only then imports
 * app.js. That prevents a deployment from linking a new app.js against a
 * stale app_helpers.js before the normal retry UI has had a chance to start.
 */

const BOOTSTRAP_MAX_ATTEMPTS = 3;
const BOOTSTRAP_ATTEMPT_TIMEOUT_MS = 60000;
const bootstrapLoaderToken = new URL(import.meta.url).searchParams.get("xtbloom_bootstrap_loader");
let bootstrapGeneration = 0;
let bootstrapController = null;

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
  const helpersUrl = versionedAssetUrl(
    baseUrl,
    validated.helpers,
    validated.version,
    bootstrapToken,
  );
  await Promise.all([
    fetchVerifiedAsset(validated.app, appUrl, { fetchImpl, cryptoImpl, signal, cache }),
    fetchVerifiedAsset(validated.helpers, helpersUrl, { fetchImpl, cryptoImpl, signal, cache }),
  ]);
  return { manifest: rawManifest, appUrl, helpersUrl };
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

function bootstrapText(zh, en) {
  return String(globalThis.navigator?.language || "").toLowerCase().startsWith("zh") ? zh : en;
}

export async function startBrowserApplication({ forceReload = false } = {}) {
  const generation = ++bootstrapGeneration;
  bootstrapController?.abort();
  const controller = new AbortController();
  bootstrapController = controller;

  const overlay = document.getElementById("overlay");
  const overlayText = document.getElementById("overlay-text");
  const loadBar = document.getElementById("load-bar-wrap");
  const errorBox = document.getElementById("error");
  const retryButton = document.getElementById("retry");
  overlay.hidden = false;
  overlayText.textContent = bootstrapText("正在检查引擎更新…", "Checking engine updates…");
  loadBar.hidden = true;
  errorBox.hidden = true;
  retryButton.hidden = true;

  try {
    await prepareVersionedApplication({
      forceReload,
      signal: controller.signal,
      loadApplication: true,
      applicationTokenPrefix: String(generation),
      onRetry: ({ nextAttempt, maxAttempts, waitMs }) => {
        if (generation !== bootstrapGeneration || controller.signal.aborted) return;
        overlayText.textContent = bootstrapText(
          `网络暂时不可用，${Math.ceil(waitMs / 1000)} 秒后重试（${nextAttempt}/${maxAttempts}）…`,
          `Network temporarily unavailable; retrying in ${Math.ceil(waitMs / 1000)} s (${nextAttempt}/${maxAttempts})…`,
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
    );
    errorBox.hidden = false;
    retryButton.hidden = false;
    retryButton.onclick = () => { void startBrowserApplication({ forceReload: true }); };
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
  void startBrowserApplication();
}
