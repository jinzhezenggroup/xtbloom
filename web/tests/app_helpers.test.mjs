import assert from "node:assert/strict";
import { createHash, webcrypto } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  cdnRegionForTimeZone,
  initializeBrowserCdnRouting,
  loadThreeDmol,
  prepareVersionedApplication,
  probeSourceSpeed,
  rankCdnSources,
  startBrowserCdnAndApplication,
  startBrowserApplication,
  THREE_DMOL_SOURCES,
  validateBootstrapManifest,
} from "../bootstrap.js";

import {
  BOHR_PER_ANGSTROM,
  ELEMENT_SYMBOLS,
  aggregateResourceProgress,
  angstromToBohr,
  canStartUrlSmiles,
  clampProgressPercent,
  comparableContentLength,
  copyFloat64FromMemory,
  createDebouncedPublisher,
  createRevisionOwner,
  createSmilesWorkerClient,
  delayWithSignal,
  fetchResourceBatch,
  downloadProgressPercent,
  initializeDownloadedEngineModule,
  initializeWorker,
  isRetryableLoadError,
  parseXyzCoordinates,
  postToReadyWorker,
  readSmilesQuery,
  runWithRetries,
  validateEngineManifest,
  withTimeout,
  xyzAtomsToText,
} from "../app_helpers.js";

class FakeWorker {
  constructor() {
    this.messages = [];
    this.terminated = false;
    this.onmessage = null;
    this.onerror = null;
  }

  postMessage(message, transfer) {
    this.messages.push({ message: structuredClone(message, { transfer }) });
  }

  emit(message) {
    this.onmessage({ data: message });
  }

  terminate() {
    this.terminated = true;
  }
}

function createManualTimers() {
  let sequence = 0;
  const callbacks = new Map();
  return {
    setTimer(callback, delayMs) {
      const id = ++sequence;
      callbacks.set(id, { callback, delayMs });
      return id;
    },
    clearTimer(id) { callbacks.delete(id); },
    delayOf: (id) => callbacks.get(id)?.delayMs,
    run(id) {
      const entry = callbacks.get(id);
      assert.equal(typeof entry?.callback, "function", `missing timer ${id}`);
      callbacks.delete(id);
      entry.callback();
    },
    ids: () => Array.from(callbacks.keys()),
    get size() { return callbacks.size; },
  };
}

function createBootstrapDocument() {
  const elements = new Map([
    "overlay",
    "overlay-text",
    "load-bar-wrap",
    "error",
    "retry",
  ].map((id) => [id, { hidden: false, textContent: "", onclick: null }]));
  return {
    documentImpl: {
      getElementById(id) {
        const element = elements.get(id);
        assert.ok(element, `missing bootstrap element ${id}`);
        return element;
      },
    },
    element: (id) => elements.get(id),
  };
}

test("CDN regional defaults identify mainland time zones without using language", () => {
  for (const timeZone of [
    "Asia/Shanghai",
    "Asia/Urumqi",
    "Asia/Chongqing",
    "Asia/Chungking",
    "Asia/Harbin",
    "Asia/Kashgar",
    "PRC",
  ]) {
    assert.equal(cdnRegionForTimeZone(timeZone), "mainland-china", timeZone);
  }
  for (const timeZone of [
    "UTC",
    "Asia/Hong_Kong",
    "Asia/Macau",
    "Asia/Taipei",
    "Asia/Singapore",
    "Etc/GMT-8",
    "",
  ]) {
    assert.equal(cdnRegionForTimeZone(timeZone), "global", timeZone);
  }
});

test("CDN probes choose the measured fastest source and use region only for close ties", async () => {
  const timings = new Map([
    ["jsdelivr", 40],
    ["jsdmirror", 100],
    ["local", 70],
  ]);
  const fastest = await rankCdnSources(THREE_DMOL_SOURCES, {
    region: "mainland-china",
    probeImpl: async (source) => ({ id: source.id, elapsedMs: timings.get(source.id) }),
  });
  assert.deepEqual(fastest.map((entry) => entry.source.id), [
    "jsdelivr",
    "local",
    "jsdmirror",
  ]);

  timings.set("jsdelivr", 100);
  timings.set("jsdmirror", 110);
  timings.set("local", 200);
  const mainlandTie = await rankCdnSources(THREE_DMOL_SOURCES, {
    region: "mainland-china",
    probeImpl: async (source) => ({ id: source.id, elapsedMs: timings.get(source.id) }),
  });
  assert.deepEqual(mainlandTie.map((entry) => entry.source.id), [
    "jsdmirror",
    "jsdelivr",
    "local",
  ]);

  const globalTie = await rankCdnSources(THREE_DMOL_SOURCES, {
    region: "global",
    probeImpl: async (source) => ({ id: source.id, elapsedMs: timings.get(source.id) }),
  });
  assert.deepEqual(globalTie.map((entry) => entry.source.id), [
    "jsdelivr",
    "jsdmirror",
    "local",
  ]);

  timings.set("jsdelivr", 35);
  timings.set("jsdmirror", 15);
  timings.set("local", 10);
  const chainedTie = await rankCdnSources(THREE_DMOL_SOURCES, {
    region: "global",
    probeImpl: async (source) => ({ id: source.id, elapsedMs: timings.get(source.id) }),
  });
  assert.deepEqual(chainedTie.map((entry) => entry.source.id), [
    "jsdmirror",
    "local",
    "jsdelivr",
  ]);
});

test("CDN ranking keeps failed probes behind measured sources in regional order", async () => {
  const ranked = await rankCdnSources(THREE_DMOL_SOURCES, {
    region: "mainland-china",
    probeImpl: async (source) => {
      if (source.id === "jsdelivr") return { id: source.id, elapsedMs: 30 };
      if (source.id === "jsdmirror") throw new Error("mirror probe failed");
      return { id: source.id, elapsedMs: Number.NaN };
    },
  });
  assert.deepEqual(ranked.map((entry) => entry.source.id), [
    "jsdelivr",
    "jsdmirror",
    "local",
  ]);
  assert.match(ranked[1].error.message, /mirror probe failed/);
  assert.equal(ranked[2].measurement.elapsedMs, Number.NaN);
});

test("CDN probes read and cancel only the bounded prefix", async () => {
  const chunks = [new Uint8Array(32768), new Uint8Array(32768)];
  const requests = [];
  let cancelled = false;
  let reads = 0;
  const result = await probeSourceSpeed(
    { id: "local", url: "vendor/3Dmol-min.js" },
    {
      baseUrl: "https://site.test/demo/",
      fetchImpl: async (url, options) => {
        requests.push({ url, options });
        return {
          ok: true,
          status: 206,
          body: {
            getReader: () => ({
              read: async () => ({ done: false, value: chunks[reads++] }),
              cancel: async () => { cancelled = true; },
            }),
          },
        };
      },
      now: (() => {
        const values = [100, 125];
        return () => values.shift();
      })(),
    },
  );
  assert.equal(result.elapsedMs, 25);
  assert.equal(reads, 2);
  assert.equal(cancelled, true);
  assert.equal(requests[0].url, "https://site.test/demo/vendor/3Dmol-min.js");
  assert.equal(requests[0].options.cache, "no-store");
  assert.equal(requests[0].options.headers.Range, "bytes=0-65535");
  assert.equal(requests[0].options.signal.aborted, true);
});

test("CDN probes reject unavailable, failed, non-streaming, and empty responses", async () => {
  const source = { id: "local", url: "vendor/3Dmol-min.js" };
  const options = { baseUrl: "https://site.test/demo/", timeoutMs: 100 };

  await assert.rejects(
    probeSourceSpeed(source, { ...options, fetchImpl: null }),
    /fetch is unavailable/,
  );
  await assert.rejects(
    probeSourceSpeed(source, {
      ...options,
      fetchImpl: async () => ({ ok: false, status: 503 }),
    }),
    /HTTP 503/,
  );
  await assert.rejects(
    probeSourceSpeed(source, {
      ...options,
      fetchImpl: async () => ({ ok: true, status: 200, body: null }),
    }),
    /streaming response is unavailable/,
  );
  await assert.rejects(
    probeSourceSpeed(source, {
      ...options,
      fetchImpl: async () => ({
        ok: true,
        status: 206,
        body: {
          getReader: () => ({
            read: async () => ({ done: true, value: undefined }),
            cancel: async () => {},
          }),
        },
      }),
    }),
    /empty probe response/,
  );
});

test("verified 3Dmol loading falls through ranked sources", async () => {
  const attempted = [];
  const globalImpl = {};
  const result = await loadThreeDmol([
    { source: { id: "jsdmirror", url: "https://mirror.test/3dmol.js" } },
    { source: { id: "jsdelivr", url: "https://jsdelivr.test/3dmol.js" } },
    { source: { id: "local", url: "vendor/3Dmol-min.js" } },
  ], {
    baseUrl: "https://site.test/",
    globalImpl,
    fetchBytesImpl: async (url) => {
      attempted.push(url);
      if (url.includes("mirror.test")) throw new Error("mirror unavailable");
      return new Uint8Array([1, 2, 3]).buffer;
    },
    executeScriptImpl: async () => { globalImpl.$3Dmol = { version: "2.5.5" }; },
  });
  assert.equal(result.source, "jsdelivr");
  assert.deepEqual(attempted, [
    "https://mirror.test/3dmol.js",
    "https://jsdelivr.test/3dmol.js",
  ]);
});

test("verified 3Dmol loading rejects a complete source with the wrong size", async () => {
  await assert.rejects(
    loadThreeDmol([
      { source: { id: "jsdelivr", url: "https://jsdelivr.test/3dmol.js" } },
    ], {
      baseUrl: "https://site.test/",
      globalImpl: {},
      fetchImpl: async () => ({
        ok: true,
        status: 200,
        arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer,
      }),
      cryptoImpl: webcrypto,
      executeScriptImpl: async () => {
        assert.fail("unverified 3Dmol bytes must not execute");
      },
    }),
    (error) => {
      assert.equal(error instanceof AggregateError, true);
      assert.match(error.errors[0].message, /expected 537792 bytes/);
      return true;
    },
  );
});

test("verified 3Dmol loading rejects a complete source with the wrong digest", async () => {
  await assert.rejects(
    loadThreeDmol([
      { source: { id: "jsdelivr", url: "https://jsdelivr.test/3dmol.js" } },
    ], {
      baseUrl: "https://site.test/",
      globalImpl: {},
      fetchImpl: async () => ({
        ok: true,
        status: 200,
        arrayBuffer: async () => new Uint8Array(537792).buffer,
      }),
      cryptoImpl: webcrypto,
      executeScriptImpl: async () => {
        assert.fail("digest-mismatched 3Dmol bytes must not execute");
      },
    }),
    (error) => {
      assert.equal(error instanceof AggregateError, true);
      assert.match(error.errors[0].message, /SHA-256 mismatch/);
      return true;
    },
  );
});

test("verified 3Dmol loading hashes and executes the pinned local bundle", async () => {
  const fileBytes = await readFile(new URL(
    "../node_modules/3dmol/build/3Dmol-min.js",
    import.meta.url,
  ));
  const pinnedBytes = Uint8Array.from(fileBytes).buffer;
  const globalImpl = {};
  let requestSignal = null;
  let appendedScript = null;
  const documentImpl = {
    createElement: (tagName) => {
      assert.equal(tagName, "script");
      return {};
    },
    head: {
      appendChild(script) {
        appendedScript = script;
        globalImpl.$3Dmol = { version: "2.5.5" };
        script.onload();
      },
    },
  };
  const result = await loadThreeDmol([
    { source: { id: "local", url: "vendor/3Dmol-min.js" } },
  ], {
    baseUrl: "https://site.test/demo/",
    documentImpl,
    globalImpl,
    fetchImpl: async (_url, options) => {
      requestSignal = options.signal;
      return {
        ok: true,
        status: 200,
        arrayBuffer: async () => pinnedBytes,
      };
    },
    cryptoImpl: webcrypto,
  });
  assert.equal(result.source, "local");
  assert.equal(result.url, "https://site.test/demo/vendor/3Dmol-min.js");
  assert.equal(appendedScript.async, true);
  assert.match(appendedScript.src, /^blob:/);
  assert.equal(requestSignal.aborted, true);
});

test("3Dmol loading preserves existing globals and reports script/global failures", async () => {
  assert.deepEqual(await loadThreeDmol([], { globalImpl: { $3Dmol: {} } }), {
    source: "existing",
  });

  await assert.rejects(
    loadThreeDmol([
      { source: { id: "local", url: "vendor/3Dmol-min.js" } },
    ], {
      baseUrl: "https://site.test/",
      globalImpl: {},
      fetchBytesImpl: async () => new Uint8Array([1]).buffer,
      documentImpl: {
        createElement: () => ({}),
        head: { appendChild: (script) => script.onerror() },
      },
    }),
    (error) => {
      assert.equal(error instanceof AggregateError, true);
      assert.match(error.errors[0].message, /script execution failed/);
      return true;
    },
  );

  await assert.rejects(
    loadThreeDmol([
      { source: { id: "jsdelivr", url: "https://cdn.test/3dmol.js" } },
    ], {
      globalImpl: {},
      fetchBytesImpl: async () => new Uint8Array([1]).buffer,
      executeScriptImpl: async () => {},
    }),
    (error) => {
      assert.match(error.errors[0].message, /3Dmol global is unavailable/);
      return true;
    },
  );
});

test("browser CDN routing shares measured provider order with optional workers", async () => {
  const globalImpl = {};
  const rankedSources = [
    { source: { id: "local", url: "vendor/3Dmol-min.js" } },
    { source: { id: "jsdmirror", url: "https://mirror.test/3dmol.js" } },
    { source: { id: "jsdelivr", url: "https://jsdelivr.test/3dmol.js" } },
  ];
  const routing = await initializeBrowserCdnRouting({
    intlImpl: {
      DateTimeFormat: () => ({ resolvedOptions: () => ({ timeZone: "Asia/Shanghai" }) }),
    },
    globalImpl,
    rankImpl: async () => rankedSources,
    loadThreeDmolImpl: async () => ({ source: "local" }),
  });
  assert.equal(routing.region, "mainland-china");
  assert.deepEqual(routing.providers, ["jsdmirror", "jsdelivr"]);
  assert.deepEqual(globalImpl.__XTBLOOM_CDN_PROVIDERS, ["jsdmirror", "jsdelivr"]);
  assert.deepEqual(await globalImpl.__XTBLOOM_3DMOL_READY, {
    ok: true,
    source: "local",
  });
});

test("browser CDN routing uses the regional order when ranking throws", async () => {
  const globalImpl = {};
  let attemptedOrder = null;
  const routing = await initializeBrowserCdnRouting({
    intlImpl: {
      DateTimeFormat: () => ({ resolvedOptions: () => ({ timeZone: "Asia/Shanghai" }) }),
    },
    globalImpl,
    rankImpl: async () => { throw new Error("probe setup failed"); },
    loadThreeDmolImpl: async (rankedSources) => {
      attemptedOrder = rankedSources.map((entry) => entry.source.id);
      return { source: "local" };
    },
  });
  assert.deepEqual(routing.providers, ["jsdmirror", "jsdelivr"]);
  assert.deepEqual(await routing.ready, { ok: true, source: "local" });
  assert.deepEqual(attemptedOrder, ["jsdmirror", "jsdelivr", "local"]);
});

test("browser CDN routing defaults globally when time-zone lookup or loading fails", async () => {
  const loadError = new Error("all 3Dmol sources failed");
  const globalImpl = {};
  const routing = await initializeBrowserCdnRouting({
    intlImpl: {
      DateTimeFormat: () => { throw new Error("time-zone database unavailable"); },
    },
    globalImpl,
    rankImpl: async (sources) => sources.map((source) => ({ source })),
    loadThreeDmolImpl: async () => { throw loadError; },
  });
  assert.equal(routing.region, "global");
  assert.deepEqual(routing.providers, ["jsdelivr", "jsdmirror"]);
  assert.deepEqual(await routing.ready, { ok: false, error: loadError });
});

test("browser application starts while CDN routing remains in flight", async () => {
  let resolveRouting;
  let applicationStarts = 0;
  const globalImpl = {};
  const routing = startBrowserCdnAndApplication({
    globalImpl,
    initializeRouting: () => new Promise((resolve) => { resolveRouting = resolve; }),
    startApplication: () => { applicationStarts += 1; },
  });
  assert.equal(applicationStarts, 1);
  assert.equal(globalImpl.__XTBLOOM_CDN_ROUTING, routing);
  await Promise.resolve();
  resolveRouting({ region: "global" });
  assert.deepEqual(await routing, { region: "global" });
});

test("browser application startup observes unexpected promise rejection", async () => {
  const startupError = new Error("application failed");
  const logged = [];
  const globalImpl = {
    console: { error: (...values) => logged.push(values) },
  };
  startBrowserCdnAndApplication({
    globalImpl,
    initializeRouting: async () => ({ region: "global" }),
    startApplication: () => Promise.reject(startupError),
  });
  assert.deepEqual(await globalImpl.__XTBLOOM_APPLICATION_START, {
    ok: false,
    error: startupError,
  });
  assert.deepEqual(logged, [["xTBloom application startup failed", startupError]]);
});

test("browser startup publishes global fallbacks after synchronous setup failures", async () => {
  const routingError = new Error("routing setup failed");
  const applicationError = new Error("application setup failed");
  const logged = [];
  const globalImpl = {
    console: { error: (...values) => logged.push(values) },
  };
  const routing = startBrowserCdnAndApplication({
    globalImpl,
    initializeRouting: () => { throw routingError; },
    startApplication: () => { throw applicationError; },
  });
  const fallback = await routing;
  assert.equal(fallback.region, "global");
  assert.deepEqual(fallback.providers, ["jsdelivr", "jsdmirror"]);
  assert.deepEqual(fallback.rankedSources, []);
  assert.deepEqual(await fallback.ready, { ok: false, error: routingError });
  assert.deepEqual(await globalImpl.__XTBLOOM_APPLICATION_START, {
    ok: false,
    error: applicationError,
  });
  assert.deepEqual(logged, [["xTBloom application startup failed", applicationError]]);
});

test("angstrom controls are converted to optimizer bohr", () => {
  assert.equal(angstromToBohr(1), BOHR_PER_ANGSTROM);
  assert.ok(Math.abs(angstromToBohr(0.4) - 0.7558904501831313) < 1e-15);
});

test("encoded response lengths are not compared with decoded fetch chunks", () => {
  const headers = new Headers({
    "content-encoding": "gzip",
    "content-length": "737252",
  });
  assert.equal(comparableContentLength(headers), 0);
  headers.set("content-encoding", "identity");
  assert.equal(comparableContentLength(headers), 737252);
});

test("download progress is monotonic, bounded, and completes explicitly", () => {
  assert.equal(downloadProgressPercent(50, 100), 50);
  assert.equal(downloadProgressPercent(1138031, 737252), 99);
  assert.equal(downloadProgressPercent(20, 100, 75), 75);
  assert.equal(downloadProgressPercent(0, 0, 42), 42);
  assert.equal(downloadProgressPercent(0, 0, 42, true), 100);
  assert.equal(clampProgressPercent(154.361), 100);
  assert.equal(clampProgressPercent(-4), 0);
  assert.equal(clampProgressPercent(Number.NaN), 0);
});

test("aggregate progress reports exact files and bytes when all lengths are comparable", () => {
  assert.deepEqual(aggregateResourceProgress([
    { loaded: 3, total: 3, complete: true },
    { loaded: 2, total: 4, complete: false },
  ]), {
    totalFiles: 2,
    completedFiles: 1,
    loadedBytes: 5,
    totalBytes: 7,
    percent: 5 / 7 * 100,
    barPercent: 5 / 7 * 100,
    complete: false,
  });
  assert.deepEqual(aggregateResourceProgress([
    { loaded: 3, total: 3, complete: true },
    { loaded: 4, total: 4, complete: true },
  ]), {
    totalFiles: 2,
    completedFiles: 2,
    loadedBytes: 7,
    totalBytes: 7,
    percent: 100,
    barPercent: 100,
    complete: true,
  });
});

test("aggregate progress does not invent a byte total for encoded assets", () => {
  assert.deepEqual(aggregateResourceProgress([
    { loaded: 30, total: 0, complete: false },
    { loaded: 10, total: 10, complete: true },
  ]), {
    totalFiles: 2,
    completedFiles: 1,
    loadedBytes: 40,
    totalBytes: null,
    percent: null,
    barPercent: 50,
    complete: false,
  });
});

test("engine manifests provide safe versioned paths and exact decoded sizes", () => {
  const digest = "a".repeat(64);
  assert.deepEqual(validateEngineManifest({
    schema_version: 1,
    version: "b".repeat(64),
    assets: [
      { id: "worker", path: "worker.js", bytes: 123, sha256: digest },
      { id: "wasm", path: "engine.wasm", bytes: 456, sha256: digest },
    ],
  }, ["worker", "wasm"]), {
    schemaVersion: 1,
    version: "b".repeat(64),
    assets: [
      { id: "worker", path: "worker.js", bytes: 123, sha256: digest },
      { id: "wasm", path: "engine.wasm", bytes: 456, sha256: digest },
    ],
  });
  assert.throws(() => validateEngineManifest({
    schema_version: 1,
    version: "b".repeat(64),
    assets: [{ id: "worker", path: "../worker.js", bytes: 1, sha256: digest }],
  }), /unsafe path/);
  assert.throws(() => validateEngineManifest({
    schema_version: 1,
    version: "b".repeat(64),
    assets: [{ id: "worker", path: "worker.js", bytes: 1, sha256: digest }],
  }, ["worker", "data"]), /missing data/);
});

test("bootstrap verifies the core graph and leaves versioned SMILES assets lazy", async () => {
  const app = new TextEncoder().encode("export const app = true;\n");
  const c60 = new TextEncoder().encode("export const C60_XYZ = 'C 0 0 0';\n");
  const helpers = new TextEncoder().encode("export const helper = true;\n");
  const smilesWorker = new TextEncoder().encode("export const worker = true;\n");
  const smilesHelpers = new TextEncoder().encode("export const smiles = true;\n");
  const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
  const assets = [
    { id: "app", path: "app.js", bytes: app.byteLength, sha256: digest(app) },
    { id: "c60", path: "c60_case.js", bytes: c60.byteLength, sha256: digest(c60) },
    {
      id: "helpers",
      path: "app_helpers.js",
      bytes: helpers.byteLength,
      sha256: digest(helpers),
    },
    {
      id: "smiles_worker",
      path: "smiles_worker.js",
      bytes: smilesWorker.byteLength,
      sha256: digest(smilesWorker),
    },
    {
      id: "smiles_helpers",
      path: "smiles_helpers.js",
      bytes: smilesHelpers.byteLength,
      sha256: digest(smilesHelpers),
    },
  ];
  const manifest = {
    schema_version: 1,
    version: "c".repeat(64),
    assets,
  };
  assert.equal(validateBootstrapManifest(manifest).version, "c".repeat(64));

  const requests = [];
  const result = await prepareVersionedApplication({
    baseUrl: "https://example.test/bootstrap.js",
    bootstrapToken: "manual-2",
    maxAttempts: 1,
    cryptoImpl: webcrypto,
    fetchImpl: async (url, options) => {
      const href = String(url);
      requests.push({ href, cache: options.cache });
      if (href.endsWith("engine-manifest.json")) {
        return new Response(JSON.stringify(manifest), { status: 200 });
      }
      if (href.includes("c60_case.js")) return new Response(c60, { status: 200 });
      if (href.includes("app_helpers.js")) return new Response(helpers, { status: 200 });
      if (href.includes("smiles_")) throw new Error("optional SMILES assets must remain lazy");
      if (href.includes("app.js")) return new Response(app, { status: 200 });
      return new Response(null, { status: 404 });
    },
  });
  assert.deepEqual(result.manifest, manifest);
  assert.equal(result.appUrl.searchParams.get("xtbloom_version"), "c".repeat(64));
  assert.equal(result.appUrl.searchParams.get("xtbloom_bootstrap"), "manual-2");
  assert.equal(result.c60Url.searchParams.get("xtbloom_version"), "c".repeat(64));
  assert.equal(result.c60Url.searchParams.get("xtbloom_bootstrap"), "manual-2");
  assert.equal(result.helpersUrl.searchParams.get("xtbloom_version"), "c".repeat(64));
  assert.deepEqual(
    requests.map((request) => request.cache),
    ["no-cache", "default", "default", "default"],
  );
});

test("bootstrap manifest validation rejects unsafe partial metadata", () => {
  const digest = "a".repeat(64);
  const valid = {
    schema_version: 1,
    version: "b".repeat(64),
    assets: [
      { id: "app", path: "app.js", bytes: 10, sha256: digest },
      { id: "c60", path: "c60_case.js", bytes: 15, sha256: digest },
      { id: "helpers", path: "app_helpers.js", bytes: 20, sha256: digest },
      { id: "smiles_worker", path: "smiles_worker.js", bytes: 25, sha256: digest },
      { id: "smiles_helpers", path: "smiles_helpers.js", bytes: 30, sha256: digest },
    ],
  };
  assert.throws(() => validateBootstrapManifest(null), /unsupported/);
  assert.throws(() => validateBootstrapManifest({ ...valid, version: "latest" }), /invalid/);
  assert.throws(() => validateBootstrapManifest({
    ...valid,
    assets: [
      { ...valid.assets[0], path: "../app.js" },
      valid.assets[1],
      valid.assets[2],
    ],
  }), /unsafe path/);
  assert.throws(() => validateBootstrapManifest({
    ...valid,
    assets: [
      { ...valid.assets[0], bytes: 0 },
      valid.assets[1],
      valid.assets[2],
    ],
  }), /invalid size/);
  assert.throws(() => validateBootstrapManifest({
    ...valid,
    assets: [
      { ...valid.assets[0], sha256: "unverified" },
      valid.assets[1],
      valid.assets[2],
    ],
  }), /invalid digest/);
  assert.throws(() => validateBootstrapManifest({
    ...valid,
    assets: [valid.assets[0], valid.assets[1]],
  }), /missing helpers/);
  assert.throws(() => validateBootstrapManifest({
    ...valid,
    assets: valid.assets.filter((asset) => asset.id !== "smiles_worker"),
  }), /missing smiles_worker/);
  assert.throws(() => validateBootstrapManifest({
    ...valid,
    assets: valid.assets.filter((asset) => asset.id !== "smiles_helpers"),
  }), /missing smiles_helpers/);
});

test("bootstrap UI exposes retry progress and clears stale recovery state", async () => {
  const page = createBootstrapDocument();
  let preparedOptions = null;
  await startBrowserApplication({
    documentImpl: page.documentImpl,
    navigatorImpl: { language: "en-US" },
    prepareApplication: async (options) => {
      preparedOptions = options;
      options.onRetry({ nextAttempt: 2, maxAttempts: 3, waitMs: 500 });
    },
  });
  assert.equal(preparedOptions.forceReload, false);
  assert.equal(preparedOptions.loadApplication, true);
  assert.match(preparedOptions.applicationTokenPrefix, /^\d+$/);
  assert.equal(page.element("overlay").hidden, false);
  assert.equal(page.element("load-bar-wrap").hidden, true);
  assert.equal(page.element("error").hidden, true);
  assert.equal(page.element("retry").hidden, true);
  assert.equal(page.element("retry").onclick, null);
  assert.match(page.element("overlay-text").textContent, /retrying in 1 s \(2\/3\)/);
});

test("bootstrap manual retry stays in-page and uses a forced reload", async () => {
  const page = createBootstrapDocument();
  const forceReloads = [];
  await startBrowserApplication({
    documentImpl: page.documentImpl,
    navigatorImpl: { language: "zh-CN" },
    prepareApplication: async (options) => {
      forceReloads.push(options.forceReload);
      if (!options.forceReload) throw new TypeError("Load failed");
    },
  });
  assert.equal(page.element("overlay").hidden, true);
  assert.equal(page.element("error").hidden, false);
  assert.match(page.element("error").textContent, /引擎启动文件加载失败：Load failed/);
  assert.equal(page.element("retry").hidden, false);
  assert.equal(typeof page.element("retry").onclick, "function");

  await page.element("retry").onclick();
  assert.deepEqual(forceReloads, [false, true]);
  assert.equal(page.element("overlay").hidden, false);
  assert.equal(page.element("error").hidden, true);
  assert.equal(page.element("retry").hidden, true);
  assert.equal(page.element("retry").onclick, null);
});

test("application import timeout retries with a distinct guarded module URL", async () => {
  const app = new TextEncoder().encode("export const app = true;\n");
  const c60 = new TextEncoder().encode("export const C60_XYZ = 'C 0 0 0';\n");
  const helpers = new TextEncoder().encode("export const helper = true;\n");
  const smilesWorker = new TextEncoder().encode("export const worker = true;\n");
  const smilesHelpers = new TextEncoder().encode("export const smiles = true;\n");
  const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
  const manifest = {
    schema_version: 1,
    version: "d".repeat(64),
    assets: [
      { id: "app", path: "app.js", bytes: app.byteLength, sha256: digest(app) },
      { id: "c60", path: "c60_case.js", bytes: c60.byteLength, sha256: digest(c60) },
      {
        id: "helpers",
        path: "app_helpers.js",
        bytes: helpers.byteLength,
        sha256: digest(helpers),
      },
      {
        id: "smiles_worker",
        path: "smiles_worker.js",
        bytes: smilesWorker.byteLength,
        sha256: digest(smilesWorker),
      },
      {
        id: "smiles_helpers",
        path: "smiles_helpers.js",
        bytes: smilesHelpers.byteLength,
        sha256: digest(smilesHelpers),
      },
    ],
  };
  const imports = [];
  const result = await prepareVersionedApplication({
    baseUrl: "https://example.test/bootstrap.js",
    maxAttempts: 2,
    // Leave ample time for response decoding and WebCrypto on loaded CI
    // runners; only the deliberately unresolved import should time out.
    timeoutMs: 100,
    delay: async () => {},
    loadApplication: true,
    applicationTokenPrefix: "generation",
    cryptoImpl: webcrypto,
    fetchImpl: async (url) => {
      const href = String(url);
      if (href.endsWith("engine-manifest.json")) {
        return new Response(JSON.stringify(manifest), { status: 200 });
      }
      if (href.includes("c60_case.js")) return new Response(c60, { status: 200 });
      if (href.includes("app_helpers.js")) return new Response(helpers, { status: 200 });
      if (href.includes("smiles_worker.js")) return new Response(smilesWorker, { status: 200 });
      if (href.includes("smiles_helpers.js")) return new Response(smilesHelpers, { status: 200 });
      if (href.includes("app.js")) return new Response(app, { status: 200 });
      return new Response(null, { status: 404 });
    },
    importImpl: async (url) => {
      imports.push(url);
      if (imports.length === 1) return new Promise(() => {});
      return {};
    },
  });
  assert.equal(result.appUrl.searchParams.get("xtbloom_bootstrap"), "generation-2");
  assert.deepEqual(
    imports.map((url) => new URL(url).searchParams.get("xtbloom_bootstrap")),
    ["generation-1", "generation-2"],
  );
  assert.equal(globalThis.__XTBLOOM_APP_BOOT_TOKEN, "generation-2");
  delete globalThis.__XTBLOOM_APP_BOOT_TOKEN;
  delete globalThis.__XTBLOOM_BOOTSTRAP_MANIFEST;
});

test("aborting a hanging application import invalidates its execution token", async () => {
  const app = new TextEncoder().encode("export const app = true;\n");
  const c60 = new TextEncoder().encode("export const C60_XYZ = 'C 0 0 0';\n");
  const helpers = new TextEncoder().encode("export const helper = true;\n");
  const smilesWorker = new TextEncoder().encode("export const worker = true;\n");
  const smilesHelpers = new TextEncoder().encode("export const smiles = true;\n");
  const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
  const manifest = {
    schema_version: 1,
    version: "e".repeat(64),
    assets: [
      { id: "app", path: "app.js", bytes: app.byteLength, sha256: digest(app) },
      { id: "c60", path: "c60_case.js", bytes: c60.byteLength, sha256: digest(c60) },
      {
        id: "helpers",
        path: "app_helpers.js",
        bytes: helpers.byteLength,
        sha256: digest(helpers),
      },
      {
        id: "smiles_worker",
        path: "smiles_worker.js",
        bytes: smilesWorker.byteLength,
        sha256: digest(smilesWorker),
      },
      {
        id: "smiles_helpers",
        path: "smiles_helpers.js",
        bytes: smilesHelpers.byteLength,
        sha256: digest(smilesHelpers),
      },
    ],
  };
  const controller = new AbortController();
  let markImportStarted;
  const importStarted = new Promise((resolve) => { markImportStarted = resolve; });
  const loading = prepareVersionedApplication({
    baseUrl: "https://example.test/bootstrap.js",
    maxAttempts: 1,
    timeoutMs: 1000,
    signal: controller.signal,
    loadApplication: true,
    applicationTokenPrefix: "aborted",
    cryptoImpl: webcrypto,
    fetchImpl: async (url) => {
      const href = String(url);
      if (href.endsWith("engine-manifest.json")) {
        return new Response(JSON.stringify(manifest), { status: 200 });
      }
      if (href.includes("c60_case.js")) return new Response(c60, { status: 200 });
      if (href.includes("app_helpers.js")) return new Response(helpers, { status: 200 });
      if (href.includes("smiles_worker.js")) return new Response(smilesWorker, { status: 200 });
      if (href.includes("smiles_helpers.js")) return new Response(smilesHelpers, { status: 200 });
      if (href.includes("app.js")) return new Response(app, { status: 200 });
      return new Response(null, { status: 404 });
    },
    importImpl: async () => {
      markImportStarted();
      return new Promise(() => {});
    },
  });
  await importStarted;
  assert.equal(globalThis.__XTBLOOM_APP_BOOT_TOKEN, "aborted-1");
  controller.abort();
  await assert.rejects(loading, (error) => error.name === "AbortError");
  assert.equal(globalThis.__XTBLOOM_APP_BOOT_TOKEN, "expired:aborted-1");
  delete globalThis.__XTBLOOM_APP_BOOT_TOKEN;
  delete globalThis.__XTBLOOM_BOOTSTRAP_MANIFEST;
});

test("downloaded engine initialization supplies wasm and data without file lookup", async () => {
  const emptyWasm = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]);
  const data = new Uint8Array([3, 4, 5]);
  const initialized = await initializeDownloadedEngineModule(
    async (options) => {
      let received = false;
      const exports = options.instantiateWasm({}, (instance, module) => {
        assert.ok(instance instanceof WebAssembly.Instance);
        assert.ok(module instanceof WebAssembly.Module);
        received = true;
      });
      assert.equal(received, true);
      assert.deepEqual(Object.keys(exports), []);
      return Array.from(new Uint8Array(options.getPreloadedPackage("engine.data", 3)));
    },
    emptyWasm,
    data,
  );
  assert.deepEqual(initialized, [3, 4, 5]);
});

test("resource batches stream all files under one exact progress ledger", async () => {
  const progress = [];
  const bodies = new Map([
    ["https://example.test/worker.js", new Uint8Array([1, 2, 3])],
    ["https://example.test/engine.wasm", new Uint8Array([4, 5, 6, 7])],
  ]);
  const results = await fetchResourceBatch([
    { id: "worker", url: "https://example.test/worker.js" },
    { id: "wasm", url: "https://example.test/engine.wasm" },
  ], {
    fetchImpl: async (url) => {
      const body = bodies.get(url);
      return new Response(body, {
        status: 200,
        headers: { "content-encoding": "identity", "content-length": String(body.byteLength) },
      });
    },
    onProgress: (state) => progress.push(state),
  });

  assert.deepEqual(Array.from(results.get("worker")), [1, 2, 3]);
  assert.deepEqual(Array.from(results.get("wasm")), [4, 5, 6, 7]);
  assert.deepEqual(progress.at(-1), {
    totalFiles: 2,
    completedFiles: 2,
    loadedBytes: 7,
    totalBytes: 7,
    percent: 100,
    barPercent: 100,
    complete: true,
  });
});

test("resource batches fall back to arrayBuffer when streaming is unavailable", async () => {
  const payload = new Uint8Array([8, 9]);
  const results = await fetchResourceBatch([{ id: "data", url: "data.test" }], {
    fetchImpl: async () => ({
      ok: true,
      status: 200,
      headers: new Headers(),
      body: null,
      arrayBuffer: async () => payload.buffer,
    }),
  });
  assert.deepEqual(Array.from(results.get("data")), [8, 9]);
});

test("manifest sizes keep decoded progress exact for compressed responses", async () => {
  const progress = [];
  const results = await fetchResourceBatch([
    { id: "module", url: "module.js", bytes: 3 },
  ], {
    fetchImpl: async () => new Response(new Uint8Array([1, 2, 3]), {
      status: 200,
      headers: { "content-encoding": "gzip", "content-length": "2" },
    }),
    onProgress: (state) => progress.push(state),
  });
  assert.deepEqual(Array.from(results.get("module")), [1, 2, 3]);
  assert.equal(progress.at(-1).totalBytes, 3);
  assert.equal(progress.at(-1).percent, 100);
});

test("resource batches reject a manifest digest mismatch as retryable", async () => {
  await assert.rejects(fetchResourceBatch([
    { id: "wasm", url: "engine.wasm", bytes: 3, sha256: "0".repeat(64) },
  ], {
    fetchImpl: async () => new Response(new Uint8Array([1, 2, 3]), { status: 200 }),
  }), (error) => {
    assert.equal(error.retryable, true);
    assert.match(error.message, /Digest mismatch/);
    return true;
  });
});

test("retry policy separates transient transport errors from deterministic failures", async () => {
  assert.equal(isRetryableLoadError(Object.assign(new Error("busy"), { status: 503 })), true);
  assert.equal(isRetryableLoadError(Object.assign(new Error("missing"), { status: 404 })), false);
  assert.equal(isRetryableLoadError(new TypeError("Load failed")), true);
  assert.equal(isRetryableLoadError(Object.assign(new Error("bad wasm"), { name: "CompileError" })), false);
  assert.equal(isRetryableLoadError(Object.assign(new Error("script error"), { phase: "worker-bootstrap" })), true);

  const attempts = [];
  const retries = [];
  const result = await runWithRetries(async (attempt) => {
    attempts.push(attempt);
    if (attempt < 3) throw new TypeError("Load failed");
    return "ready";
  }, {
    delay: async (ms) => retries.push(ms),
  });
  assert.equal(result, "ready");
  assert.deepEqual(attempts, [1, 2, 3]);
  assert.deepEqual(retries, [500, 1000]);
});

test("abortable retry delays resolve normally and reject both abort timings", async () => {
  await delayWithSignal(0);

  const alreadyAborted = new AbortController();
  alreadyAborted.abort();
  await assert.rejects(delayWithSignal(1000, alreadyAborted.signal), (error) => {
    assert.equal(error.name, "AbortError");
    return true;
  });

  const pendingAbort = new AbortController();
  const delayed = delayWithSignal(1000, pendingAbort.signal);
  pendingAbort.abort();
  await assert.rejects(delayed, (error) => {
    assert.equal(error.name, "AbortError");
    return true;
  });
});

test("deterministic loader failures are not retried", async () => {
  let attempts = 0;
  await assert.rejects(runWithRetries(async () => {
    attempts++;
    throw Object.assign(new Error("invalid module"), { name: "CompileError" });
  }, { delay: async () => {} }), /invalid module/);
  assert.equal(attempts, 1);
});

test("a transient 503 retries the complete resource batch", async () => {
  let fetches = 0;
  const result = await runWithRetries(() => fetchResourceBatch([
    { id: "wasm", url: "engine.wasm", bytes: 3 },
  ], {
    fetchImpl: async () => {
      fetches++;
      return fetches === 1
        ? new Response(null, { status: 503 })
        : new Response(new Uint8Array([1, 2, 3]), { status: 200 });
    },
  }), { delay: async () => {} });
  assert.equal(fetches, 2);
  assert.deepEqual(Array.from(result.get("wasm")), [1, 2, 3]);
});

test("SMILES query parsing is percent-decoded and length bounded", () => {
  assert.equal(readSmilesQuery("https://example.test/?smiles=CCO"), "CCO");
  assert.equal(readSmilesQuery("https://example.test/?smiles=%5BNH4%2B%5D"), "[NH4+]");
  assert.equal(readSmilesQuery("https://example.test/?other=CCO"), null);
  assert.throws(
    () => readSmilesQuery("https://example.test/?smiles=CCCC", 3),
    /exceeds 3 characters/,
  );
});

test("URL SMILES starts once only when both workers are ready and idle", () => {
  const ready = {
    smiles: "CCO",
    started: false,
    engineState: "ready",
    smilesState: "ready",
    engineBusy: false,
    smilesBusy: false,
  };
  assert.equal(canStartUrlSmiles(ready), true);
  for (const override of [
    { smiles: null },
    { started: true },
    { engineState: "loading" },
    { smilesState: "error" },
    { engineBusy: true },
    { smilesBusy: true },
  ]) {
    assert.equal(canStartUrlSmiles({ ...ready, ...override }), false);
  }
});

test("SMILES generation timeout restarts the worker and permits a one-click retry", async () => {
  const timers = createManualTimers();
  const workers = [];
  const states = [];
  const client = createSmilesWorkerClient({
    createWorker: () => {
      const worker = new FakeWorker();
      workers.push(worker);
      return worker;
    },
    onStateChange: (event) => states.push(event),
    setTimer: timers.setTimer,
    clearTimer: timers.clearTimer,
    loadTimeoutMs: 60,
    generationTimeoutMs: 120,
  });

  client.start();
  assert.equal(workers.length, 1);
  workers[0].emit({ type: "ready", version: "9.21.0" });
  const first = client.request("complex");
  assert.deepEqual(workers[0].messages[0].message, {
    type: "generate",
    id: 1,
    smiles: "complex",
  });
  const generationTimer = timers.ids()[0];
  assert.equal(timers.delayOf(generationTimer), 120);
  timers.run(generationTimer);
  await assert.rejects(first, (error) => error.code === "smiles_err_timeout");
  assert.equal(workers[0].terminated, true);
  assert.equal(workers.length, 2);
  assert.equal(client.getState(), "loading");
  assert.equal(states.at(-1).reason, "generation-timeout");
  assert.deepEqual(states.at(-1).recoveryStatus, {
    key: "smiles_err_timeout",
    tone: "err",
  });

  /* A queued event from the terminated instance must not publish readiness for
   * or otherwise alter the replacement Worker. */
  workers[0].emit({ type: "ready", version: "stale" });
  workers[0].onerror({ message: "stale failure" });
  assert.equal(client.getState(), "loading");
  assert.equal(workers[1].terminated, false);

  workers[1].emit({ type: "ready", version: "9.21.0" });
  assert.deepEqual(states.at(-1).recoveryStatus, {
    key: "smiles_err_timeout",
    tone: "err",
  });
  const retry = client.request("complex");
  workers[1].emit({ type: "result", id: 2, ok: true, result: { atomCount: 77 } });
  assert.deepEqual(await retry, { atomCount: 77 });
  assert.equal(client.getState(), "ready");
  client.dispose();
});

test("cancelling SMILES work terminates abandoned synchronous work", async () => {
  const timers = createManualTimers();
  const workers = [];
  const client = createSmilesWorkerClient({
    createWorker: () => {
      const worker = new FakeWorker();
      workers.push(worker);
      return worker;
    },
    setTimer: timers.setTimer,
    clearTimer: timers.clearTimer,
  });

  client.start();
  workers[0].emit({ type: "ready", version: "9.21.0" });
  const abandoned = client.request("old-smiles");
  const cancelled = client.cancel(new DOMException("superseded", "AbortError"));
  assert.equal(cancelled, true);
  await assert.rejects(abandoned, (error) => error.name === "AbortError");
  assert.equal(workers[0].terminated, true);
  assert.equal(workers.length, 2);

  workers[0].emit({
    type: "result",
    id: 1,
    ok: true,
    result: { atomCount: 1 },
  });
  assert.equal(client.getState(), "loading");
  workers[1].emit({ type: "ready", version: "9.21.0" });
  assert.equal(client.getState(), "ready");
  assert.equal(client.cancel(), false);

  const replacement = client.request("new-smiles");
  workers[1].emit({
    type: "result",
    id: 2,
    ok: true,
    result: { atomCount: 2 },
  });
  assert.deepEqual(await replacement, { atomCount: 2 });
  assert.equal(timers.size, 0, "cancelled work must not leave a timer ahead of the retry");
  client.dispose();
});

test("SMILES postMessage failure rejects locally and rebuilds the worker", async () => {
  const timers = createManualTimers();
  const workers = [];
  const client = createSmilesWorkerClient({
    createWorker: () => {
      const worker = new FakeWorker();
      workers.push(worker);
      return worker;
    },
    setTimer: timers.setTimer,
    clearTimer: timers.clearTimer,
  });

  client.start();
  workers[0].emit({ type: "ready", version: "9.21.0" });
  const cause = new DOMException("clone failed", "DataCloneError");
  workers[0].postMessage = () => { throw cause; };

  await assert.rejects(client.request("CCO"), (error) => {
    assert.equal(error.code, "smiles_err_library");
    assert.equal(error.cause, cause);
    return true;
  });
  assert.equal(workers[0].terminated, true);
  assert.equal(workers.length, 2);
  assert.equal(client.getState(), "loading");
  assert.equal(timers.size, 1, "only the replacement load timer should remain");

  workers[1].emit({ type: "ready", version: "9.21.0" });
  const retry = client.request("CCO");
  workers[1].emit({ type: "result", id: 2, ok: true, result: { atomCount: 9 } });
  assert.deepEqual(await retry, { atomCount: 9 });
  client.dispose();
});

test("worker initialization remains pending until ready", async () => {
  const worker = new FakeWorker();
  const wasmBinary = new Uint8Array([0, 1, 2]);
  const dataBinary = new Uint8Array([3, 4]);
  let settled = false;
  const initialized = initializeWorker(worker, {
    wasmBinary,
    dataBinary,
    moduleUrl: "https://example.test/xtbloom_web.js",
    helpersUrl: "https://example.test/app_helpers.js",
  }).then((message) => {
    settled = true;
    return message;
  });

  await Promise.resolve();
  assert.equal(settled, false);
  assert.equal(worker.messages[0].message.type, "init");
  assert.deepEqual(Array.from(worker.messages[0].message.wasmBinary), [0, 1, 2]);
  assert.deepEqual(Array.from(worker.messages[0].message.dataBinary), [3, 4]);
  assert.equal(worker.messages[0].message.moduleUrl, "https://example.test/xtbloom_web.js");
  assert.equal(wasmBinary.byteLength, 3);
  assert.equal(dataBinary.byteLength, 2);

  worker.emit({ type: "ready", version: "test" });
  assert.deepEqual(await initialized, { type: "ready", version: "test" });
});

test("worker initialization errors reject", async () => {
  const worker = new FakeWorker();
  const initialized = initializeWorker(worker, {
    wasmBinary: new Uint8Array([0]),
    dataBinary: new Uint8Array([1]),
    moduleUrl: "module.js",
    helpersUrl: "helpers.js",
  });
  worker.emit({
    type: "error",
    error: { name: "TypeError", message: "Load failed", phase: "module-import" },
  });
  await assert.rejects(initialized, (error) => {
    assert.equal(error.name, "TypeError");
    assert.equal(error.phase, "module-import");
    assert.match(error.message, /Load failed/);
    return true;
  });
});

test("worker postMessage failure leaves caller payloads reusable", async () => {
  const worker = new FakeWorker();
  worker.postMessage = () => { throw new DOMException("clone failed", "DataCloneError"); };
  const wasmBinary = new Uint8Array([1, 2, 3]);
  const dataBinary = new Uint8Array([4, 5]);
  await assert.rejects(initializeWorker(worker, {
    wasmBinary,
    dataBinary,
    moduleUrl: "module.js",
    helpersUrl: "helpers.js",
  }), /clone failed/);
  assert.deepEqual(Array.from(wasmBinary), [1, 2, 3]);
  assert.deepEqual(Array.from(dataBinary), [4, 5]);
});

test("calls require a live ready worker", () => {
  const worker = new FakeWorker();
  assert.throws(() => postToReadyWorker(null, "ready", { type: "call" }), /not ready/);
  assert.throws(() => postToReadyWorker(worker, "loading", { type: "call" }), /not ready/);

  postToReadyWorker(worker, "ready", { type: "call", id: 7 });
  assert.deepEqual(worker.messages[0].message, { type: "call", id: 7 });
});

test("Float64 callback slices accept wasm32 Numbers and wasm64 BigInts", () => {
  const memory = { buffer: new ArrayBuffer(64) };
  new Float64Array(memory.buffer).set([1.25, -2.5, 3.75], 2);
  assert.deepEqual(copyFloat64FromMemory(memory, 16, 3), [1.25, -2.5, 3.75]);
  assert.deepEqual(copyFloat64FromMemory(memory, 16n, 3), [1.25, -2.5, 3.75]);
  assert.throws(() => copyFloat64FromMemory(memory, 4, 1), /misaligned/);
  assert.throws(() => copyFloat64FromMemory(memory, 56, 2), /outside memory/);
  assert.throws(
    () => copyFloat64FromMemory(memory, BigInt(Number.MAX_SAFE_INTEGER) + 1n, 1),
    /safe integer range/,
  );
});

test("deployed page describes the universal browser artifact as wasm32", async () => {
  const [appSource, indexSource] = await Promise.all([
    readFile(new URL("../app.js", import.meta.url), "utf8"),
    readFile(new URL("../index.html", import.meta.url), "utf8"),
  ]);
  assert.doesNotMatch(appSource, /C\+\+17 WASM64/i);
  assert.doesNotMatch(indexSource, /C\+\+17 WASM64/i);
  assert.match(appSource, /wasm32/i);
  assert.match(indexSource, /wasm32/i);
});

test("page bootstraps pinned SMILES loading and applies URL-optimized geometry", async () => {
  const [appSource, indexSource] = await Promise.all([
    readFile(new URL("../app.js", import.meta.url), "utf8"),
    readFile(new URL("../index.html", import.meta.url), "utf8"),
  ]);
  assert.match(appSource, /startSmilesWorker\(\);/);
  assert.match(appSource, /readSmilesQuery\(window\.location\.href\)/);
  assert.match(appSource, /applyFinalGeometry:\s*true/);
  assert.match(appSource, /setCoordinateInput\(d\.geometry, \{ preserveOptimization: true \}\)/);
  assert.doesNotMatch(appSource, /smiles-alert/);
  assert.match(indexSource, /id="smiles"/);
  assert.match(indexSource, /id="smiles-generate"/);
  assert.doesNotMatch(indexSource, /id="smiles-alert"/);
});

test("engine bootstrap retries a coherent versioned generation without reloading the page", async () => {
  const [
    appSource,
    bootstrapSource,
    helperSource,
    workerSource,
    smilesHelperSource,
    smilesWorkerSource,
    manifestSource,
    indexSource,
  ] = await Promise.all([
    readFile(new URL("../app.js", import.meta.url), "utf8"),
    readFile(new URL("../bootstrap.js", import.meta.url), "utf8"),
    readFile(new URL("../app_helpers.js", import.meta.url), "utf8"),
    readFile(new URL("../worker.js", import.meta.url), "utf8"),
    readFile(new URL("../smiles_helpers.js", import.meta.url), "utf8"),
    readFile(new URL("../smiles_worker.js", import.meta.url), "utf8"),
    readFile(new URL("../write_engine_manifest.cmake", import.meta.url), "utf8"),
    readFile(new URL("../index.html", import.meta.url), "utf8"),
  ]);
  for (const asset of [
    "app.js",
    "c60_case.js",
    "worker.js",
    "app_helpers.js",
    "smiles_worker.js",
    "smiles_helpers.js",
    "xtbloom_web.js",
    "xtbloom_web.wasm",
    "xtbloom_web.side.wasm",
  ]) {
    assert.match(manifestSource, new RegExp(asset.replaceAll(".", "\\.")));
  }
  assert.doesNotMatch(manifestSource, /xtbloom_web\.data/);
  assert.match(appSource, /runWithRetries\(/);
  assert.match(appSource, /engineLoadGeneration/);
  assert.match(appSource, /engine-manifest\.json/);
  assert.match(appSource, /xtbloom_version/);
  assert.match(appSource, /c60CaseUrl\.searchParams\.set\("xtbloom_bootstrap"/);
  assert.match(appSource, /manifest\.version !== appContentVersion/);
  assert.match(appSource, /smilesWorkerModuleUrl\.searchParams\.set\("xtbloom_version"/);
  assert.match(appSource, /new URL\(smilesWorkerModuleUrl\.href\)/);
  assert.match(appSource, /Invalid engine manifest response/);
  assert.match(appSource, /currentLoadOrAbort\(generation, masterSignal, attemptController\.signal\)/);
  assert.match(appSource, /filter\(\(asset\) => engineIds\.has\(asset\.id\)\)/);
  assert.doesNotMatch(appSource, /window\.location\.reload\(/);
  assert.match(bootstrapSource, /fetchVerifiedAsset/);
  assert.match(bootstrapSource, /await importImpl\(prepared\.appUrl\.href\)/);
  assert.match(helperSource, /getPreloadedPackage/);
  assert.match(helperSource, /WebAssembly\.compile\(wasmBinary\)/);
  assert.match(helperSource, /instantiateWasm/);
  assert.match(workerSource, /msg\.dataBinary/);
  assert.match(workerSource, /await import\(msg\.moduleUrl\)/);
  assert.match(workerSource, /initializeDownloadedEngineModule/);
  assert.doesNotMatch(workerSource, /from "\.\/xtbloom_web\.js"/);
  assert.match(smilesHelperSource, /export async function loadOpenChemLibRuntime/);
  assert.match(smilesWorkerSource, /await import\(smilesHelpersUrl\.href\)/);
  assert.match(smilesWorkerSource, /contentVersion = workerModuleUrl\.searchParams\.get\("xtbloom_version"\)/);
  assert.match(smilesWorkerSource, /\^\[0-9a-f\]\{64\}\$/);
  assert.match(smilesWorkerSource, /requires a 64-character SHA-256 content version/);
  assert.doesNotMatch(smilesWorkerSource, /xtbloom_bootstrap/);
  assert.doesNotMatch(smilesWorkerSource, /from "\.\/smiles_helpers\.js"/);
  assert.match(indexSource, /<script type="module">/);
  assert.match(indexSource, /prefetchBootstrap/);
  assert.match(indexSource, /await import\(url\.href\)/);
  assert.doesNotMatch(indexSource, /src="bootstrap\.js"/);
  assert.doesNotMatch(indexSource, /type="module" src="app\.js"/);
  const inlineModule = indexSource.match(/<script type="module">([\s\S]*?)<\/script>/);
  assert.ok(inlineModule);
  assert.doesNotThrow(() => new Function(inlineModule[1]));

  const attemptStart = appSource.indexOf("const initialize = (async () => {");
  const manifestFetch = appSource.indexOf("const manifest = await loadEngineManifest", attemptStart);
  const timeoutWrap = appSource.indexOf("withTimeout(initialize", manifestFetch);
  assert.ok(attemptStart >= 0 && manifestFetch > attemptStart && timeoutWrap > manifestFetch);
});

test("result statistics distinguish SCC iterations from optimizer steps", async () => {
  const [appSource, indexSource] = await Promise.all([
    readFile(new URL("../app.js", import.meta.url), "utf8"),
    readFile(new URL("../index.html", import.meta.url), "utf8"),
  ]);
  assert.match(indexSource, /id="stat-iter-label"/);
  assert.match(appSource, /\$\("stat-iter-label"\)\.textContent = t\("stat_iter"\)/);
  assert.match(appSource, /\$\("stat-iter-label"\)\.textContent = t\("stat_opt_steps"\)/);
});

test("every literal app DOM lookup exists in the deployed HTML", async () => {
  const [appSource, indexSource] = await Promise.all([
    readFile(new URL("../app.js", import.meta.url), "utf8"),
    readFile(new URL("../index.html", import.meta.url), "utf8"),
  ]);
  const referenced = new Set(
    Array.from(appSource.matchAll(/\$\("([^"]+)"\)/g), (match) => match[1]),
  );
  const declared = new Set(
    Array.from(indexSource.matchAll(/\bid="([^"]+)"/g), (match) => match[1]),
  );
  assert.deepEqual([...referenced].filter((id) => !declared.has(id)).sort(), []);
});

test("timeout covers a worker that never becomes ready", async () => {
  const worker = new FakeWorker();
  const initialized = initializeWorker(worker, {
    wasmBinary: new Uint8Array([0]),
    dataBinary: new Uint8Array([1]),
    moduleUrl: "module.js",
    helpersUrl: "helpers.js",
  });
  await assert.rejects(
    withTimeout(initialized, 5, () => worker.terminate()),
    /TIME_OUT/,
  );
  assert.equal(worker.terminated, true);
});

test("element symbols cover the same period-1..103 table as the C adapter", () => {
  assert.equal(ELEMENT_SYMBOLS.length, 104);
  assert.equal(ELEMENT_SYMBOLS[0], "");
  assert.equal(ELEMENT_SYMBOLS[6], "C");
  assert.equal(ELEMENT_SYMBOLS[17], "Cl");
  assert.equal(ELEMENT_SYMBOLS[103], "Lr");
});

const WATER_XYZ =
  "O  0.00000000  0.00000000  0.00000000\n" +
  "H  0.00000000  0.00000000  0.95720000\n" +
  "H  0.00000000  0.75718000 -0.58552000";

test("valid XYZ preview parsing keeps canonical symbols and coordinates", () => {
  const parsed = parseXyzCoordinates(WATER_XYZ);
  assert.equal(parsed.ok, true);
  assert.equal(parsed.atomCount, 3);
  assert.deepEqual(parsed.atoms.map((atom) => atom.symbol), ["O", "H", "H"]);
  assert.deepEqual(parsed.atoms.map((atom) => atom.z), [0, 0.9572, -0.58552]);
  assert.equal(parsed.atoms[2].y, 0.75718);
  assert.equal(parsed.atoms[2].z, -0.58552);

  const text = xyzAtomsToText(parsed.atoms);
  assert.match(text, /^O 0 0 0\nH 0 0 0\.9572\nH 0 0\.75718 -0\.58552$/);
  assert.deepEqual(parseXyzCoordinates(text), parsed);
});

test("XYZ preview accepts atomic numbers and case-insensitive symbols like the engine", () => {
  const numeric = parseXyzCoordinates("8 0 0 0\n1 0 0 0.9572");
  assert.equal(numeric.ok, true);
  assert.deepEqual(numeric.atoms.map((atom) => atom.symbol), ["O", "H"]);
  const mixedCase = parseXyzCoordinates("o 0 0 0\ncl 0 0 1\nNA 0 0 2\nmg 0 0 3");
  assert.equal(mixedCase.ok, true);
  assert.deepEqual(mixedCase.atoms.map((atom) => atom.symbol), ["O", "Cl", "Na", "Mg"]);
});

test("XYZ preview accepts extra trailing tokens the engine parser ignores", () => {
  const parsed = parseXyzCoordinates("O 0 0 0 trailing\nH 0 0 0.9 x");
  assert.equal(parsed.ok, true);
  assert.equal(parsed.atomCount, 2);
});

test("XYZ preview canonicalizes whitespace the C parser rejects raw", () => {
  const parsed = parseXyzCoordinates("  O 0 0 0\n \t \n\tH 0 0 0.9");
  assert.equal(parsed.ok, true);
  assert.equal(xyzAtomsToText(parsed.atoms), "O 0 0 0\nH 0 0 0.9");
});

test("revision ownership supersedes stale callbacks without clearing newer busy work", () => {
  const owner = createRevisionOwner();
  const oldToken = owner.capture();
  assert.equal(owner.claim(oldToken), true);
  assert.equal(owner.claim(oldToken), false);
  assert.equal(owner.isBusy(), true);

  owner.advance(); /* Reset or an XYZ edit supersedes the old operation. */
  const newToken = owner.capture();
  assert.equal(owner.claim(newToken), true);
  assert.equal(owner.isCurrent(oldToken), false);
  assert.equal(owner.release(oldToken), false);
  assert.equal(owner.isBusy(), true);
  assert.equal(owner.release(newToken), true);
  assert.equal(owner.isBusy(), false);
});

test("debounced publication ignores cancelled and superseded callbacks", () => {
  const callbacks = new Map();
  const delays = [];
  let nextTimer = 0;
  const published = [];
  const publisher = createDebouncedPublisher((value) => published.push(value), {
    delayMs: 400,
    setTimer: (callback, delay) => {
      const id = ++nextTimer;
      callbacks.set(id, callback);
      delays.push(delay);
      return id;
    },
    /* Keep callbacks callable to model a timer already queued by the event loop. */
    clearTimer: () => {},
  });

  publisher.schedule("old-valid");
  const oldCallback = callbacks.get(1);
  publisher.cancel(); /* Invalid input retains the old viewer content. */
  oldCallback();
  assert.deepEqual(published, []);

  publisher.schedule("valid-a");
  const supersededCallback = callbacks.get(2);
  publisher.schedule("valid-b");
  supersededCallback();
  callbacks.get(3)();
  assert.deepEqual(published, ["valid-b"]);
  assert.deepEqual(delays, [400, 400, 400]);
});

test("XYZ preview rejects malformed lines, bad numbers, and incomplete rows", () => {
  for (const bad of [
    "O 0 0",
    "O  0  0  0\nH  0  0", /* missing z */
    "O a 0 0",
    "O 0 0 x",
    "O 1e999 0 0", /* overflows to Infinity */
    "O nan 0 0",
  ]) {
    assert.equal(parseXyzCoordinates(bad).ok, false, `should reject: ${bad}`);
    assert.equal(parseXyzCoordinates(bad).errorCode, "err_xyz_parse");
  }
});

test("XYZ preview rejects unknown or out-of-range element symbols", () => {
  for (const bad of ["Xx 0 0 0", "Hx 0 0 0", "104 0 0 0", "-1 0 0 0", "0 0 0 0"]) {
    const parsed = parseXyzCoordinates(bad);
    assert.equal(parsed.ok, false, `should reject: ${bad}`);
    assert.equal(parsed.errorCode, "err_xyz_element");
  }
});

test("XYZ preview enforces the 512-atom engine limit with the same boundary", () => {
  const block = "C 0 0 0\n";
  assert.equal(parseXyzCoordinates(block.repeat(512)).ok, true);
  const tooMany = parseXyzCoordinates(block.repeat(513));
  assert.equal(tooMany.ok, false);
  assert.equal(tooMany.errorCode, "err_xyz_too_many");
});

test("XYZ preview treats whitespace-only input as absent structure", () => {
  const parsed = parseXyzCoordinates("  \n\t\n");
  assert.equal(parsed.ok, false);
  assert.equal(parsed.errorCode, "no_xyz");
});

test("live preview is debounced and gates calculate on the current structure", async () => {
  const [appSource, indexSource] = await Promise.all([
    readFile(new URL("../app.js", import.meta.url), "utf8"),
    readFile(new URL("../index.html", import.meta.url), "utf8"),
  ]);
  /* Debounced auto-preview: editing XYZ updates the 3D view without compute. */
  assert.match(appSource, /PREVIEW_DEBOUNCE_MS = 400/);
  assert.match(appSource, /\$\("xyz"\)\.addEventListener\("input", schedulePreviewUpdate\)/);
  assert.match(
    appSource,
    /function schedulePreviewUpdate\(\) \{[\s\S]*?const parsed = refreshPreview\(\{ renderViewer: false \}\);[\s\S]*?if \(!parsed\.ok\)[\s\S]*?previewPublisher\.schedule/,
  );
  assert.match(appSource, /refreshPreview\(\)/);
  assert.match(
    appSource,
    /parsed\.errorCode === "no_xyz" \? "empty" : "error"/,
  );
  /* Malformed input keeps the last valid preview. */
  assert.match(appSource, /Malformed input never replaces the last valid preview/);
  assert.match(appSource, /updateMoleculeViewer\(previewState\.canonicalXyz\)/);
  assert.match(appSource, /if \(molViewer && !molUnavailable\) updateMoleculeViewer/);
  /* Calculate/optimize are gated on a valid structure. */
  assert.match(appSource, /!engineBusy && !smilesBusy && previewState\.status === "valid"/);
  assert.match(appSource, /parseXyzCoordinates\(\$\("xyz"\)\.value\)/);
  assert.equal(
    (appSource.match(/const xyz = xyzAtomsToText\(parsed\.atoms\);/g) || []).length,
    2,
  );
  assert.doesNotMatch(appSource, /const xyz = \$\("xyz"\)\.value/);
  /* SMILES import renders immediately through the preview path. */
  assert.match(appSource, /function applyGeneratedGeometry\(result\)/);
  /* Reset clears SMILES and restores the documented water template. */
  assert.match(appSource, /\$\("smiles"\)\.value = ""/);
  assert.match(appSource, /clearSmilesStatus\(\)/);
  assert.match(appSource, /setCoordinateInput\(PRESETS\.water\.xyz\)/);
  assert.match(indexSource, /id="xyz-hint"/);
});

test("stale calculations and SMILES workflows cannot overwrite newer input or Reset", async () => {
  const appSource = await readFile(new URL("../app.js", import.meta.url), "utf8");
  /* Both streamed optimization frames and the final result are revision-gated. */
  assert.match(
    appSource,
    /\(step\) => \{\s*if \(!coordinateRevisions\.isCurrent\(requestRevision\) \|\| !canPublish\(\)\) return;/,
  );
  assert.match(
    appSource,
    /const dt = performance\.now\(\) - t0;\s*if \(!coordinateRevisions\.isCurrent\(requestRevision\) \|\| !canPublish\(\)\) \{\s*throw supersededCoordinateError\(\);/,
  );
  /* Reset invalidates in-flight generation and a URL workflow waiting on either worker. */
  assert.match(
    appSource,
    /function invalidateSmilesWork\(\) \{[\s\S]*?smilesWorkflow\.advance\(\);[\s\S]*?urlSmiles = null;[\s\S]*?urlSmilesStarted = true;[\s\S]*?smilesClient\.cancel/,
  );
  assert.match(
    appSource,
    /\$\("reset"\)\.addEventListener\("click", \(\) => \{[\s\S]*?invalidateSmilesWork\(\);[\s\S]*?setCoordinateInput\(PRESETS\.water\.xyz\)/,
  );
  assert.match(
    appSource,
    /const result = await requestSmilesGeometry\(smiles\);\s*requireCurrentSmilesWorkflow\(workflowRevision\);/,
  );
  assert.match(
    appSource,
    /if \(\$\("smiles"\)\.value\.trim\(\) !== smiles\) throw supersededSmilesError\(\);/,
  );
  assert.match(
    appSource,
    /\$\("smiles"\)\.addEventListener\("input", \(\) => \{[\s\S]*?invalidateSmilesWork\(\);[\s\S]*?syncEngineControls\(\);/,
  );
  /* URL optimization must share the SMILES workflow's publication token, not
   * merely check it after runOptimize has already rendered final geometry. */
  assert.match(
    appSource,
    /runOptimize\(\{[\s\S]*?canPublish: \(\) => smilesWorkflow\.isCurrent\(workflowRevision\)/,
  );
  assert.match(
    appSource,
    /if \(!coordinateRevisions\.isCurrent\(requestRevision\) \|\| !canPublish\(\)\) return;/,
  );
  assert.match(
    appSource,
    /if \(!coordinateRevisions\.isCurrent\(requestRevision\) \|\| !canPublish\(\)\) \{\s*throw supersededCoordinateError\(\);/,
  );
  assert.match(appSource, /if \(hasCurrentResult\("optimize"\)\) renderOptimize/);
  assert.match(appSource, /if \(hasCurrentResult\("optimize"\) && d\.geometry\)/);
  assert.match(
    appSource,
    /if \(!smilesWorkflow\.isCurrent\(workflowRevision\) \|\| error\?\.name === "AbortError"\) return;/,
  );
});

test("UI copy describes the real-time preview and retains it on bad input", async () => {
  const [appSource, indexSource] = await Promise.all([
    readFile(new URL("../app.js", import.meta.url), "utf8"),
    readFile(new URL("../index.html", import.meta.url), "utf8"),
  ]);
  assert.match(appSource, /有效坐标实时预览/);
  assert.match(appSource, /Valid coordinates preview live as you type/);
  assert.match(appSource, /err_xyz_element: "含无法识别的元素符号/);
  assert.match(appSource, /err_xyz_element: "Unknown element symbol/);
  /* The static HTML hint must match the zh dictionary text. */
  assert.match(indexSource, /有效坐标实时预览/);
});
