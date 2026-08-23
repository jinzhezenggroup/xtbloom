import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";

const CHROMIUM_WIDTHS = [320, 360, 375, 390, 430, 768, 1024];
const WEBKIT_WIDTHS = [320, 390];
const VIEWPORT_HEIGHT = 900;
const CONTROL_SELECTOR = "button, input, textarea, select, a.btn, .badge";
const REGION_SELECTOR = [
  ".topbar",
  ".actions",
  ".roadmap-item",
  ".stats",
  ".output-tools",
  ".replay-controls",
  ".mol-head",
].join(", ");

function widthsFor(projectName) {
  return projectName === "webkit" ? WEBKIT_WIDTHS : CHROMIUM_WIDTHS;
}

async function setLanguage(page, language) {
  const expected = language === "zh" ? "zh-CN" : "en";
  if (await page.locator("html").getAttribute("lang") !== expected) {
    await page.locator("#lang-toggle").click();
  }
  await expect(page.locator("html")).toHaveAttribute("lang", expected);
}

async function layoutDiagnostics(page) {
  return page.evaluate(({ controlSelector, regionSelector }) => {
    const visible = (element) => {
      const style = getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      return !element.hidden && style.display !== "none" && style.visibility !== "hidden" &&
        rect.width > 0 && rect.height > 0;
    };
    const describe = (element) => {
      if (element.id) return `#${element.id}`;
      const classes = Array.from(element.classList).slice(0, 3).join(".");
      const text = String(element.textContent || element.getAttribute("aria-label") || "")
        .trim().replace(/\s+/g, " ").slice(0, 80);
      return `${element.tagName.toLowerCase()}${classes ? `.${classes}` : ""}${text ? `:${text}` : ""}`;
    };
    const viewportWidth = document.documentElement.clientWidth;
    const elements = Array.from(document.querySelectorAll(`${controlSelector}, ${regionSelector}`))
      .filter(visible)
      .map((element) => {
        const rect = element.getBoundingClientRect();
        return {
          selector: describe(element),
          left: rect.left,
          right: rect.right,
          width: rect.width,
          height: rect.height,
          display: getComputedStyle(element).display,
          overflowX: getComputedStyle(element).overflowX,
        };
      });
    const violations = elements.filter(
      (element) => element.left < -1 || element.right > viewportWidth + 1,
    );
    return {
      viewport: { width: viewportWidth, height: window.innerHeight },
      page: {
        clientWidth: viewportWidth,
        scrollWidth: document.documentElement.scrollWidth,
        bodyScrollWidth: document.body.scrollWidth,
      },
      elements,
      violations,
    };
  }, { controlSelector: CONTROL_SELECTOR, regionSelector: REGION_SELECTOR });
}

async function assertLayout(page, label) {
  const diagnostics = await layoutDiagnostics(page);
  expect(
    diagnostics.page.scrollWidth,
    `${label}: global horizontal overflow\n${JSON.stringify(diagnostics, null, 2)}`,
  ).toBeLessThanOrEqual(diagnostics.page.clientWidth + 1);
  expect(
    diagnostics.violations,
    `${label}: visible controls/regions outside viewport\n${JSON.stringify(diagnostics, null, 2)}`,
  ).toEqual([]);
}

async function assertTouchTargets(page, selectors, label) {
  for (const selector of selectors) {
    const matches = page.locator(selector);
    for (let index = 0; index < await matches.count(); index += 1) {
      const locator = matches.nth(index);
      if (!await locator.isVisible()) continue;
      const box = await locator.boundingBox();
      const target = `${selector}[${index}]`;
      expect(box, `${label}: ${target} has no box`).not.toBeNull();
      expect(box.height, `${label}: ${target} touch target is ${box.height}px`)
        .toBeGreaterThanOrEqual(44);
    }
  }
}

async function expectResultPanelVisible(page) {
  await expect.poll(async () => page.locator("#result-panel").evaluate((panel) => {
    const rect = panel.getBoundingClientRect();
    return rect.top >= -1 && rect.top < window.innerHeight - 44;
  })).toBe(true);
}

test.beforeEach(async ({ context, page }, testInfo) => {
  const localOrigin = new URL(String(testInfo.project.use.baseURL)).origin;
  await context.route("**/*", async (route) => {
    const url = new URL(route.request().url());
    if (url.origin === localOrigin) await route.continue();
    else await route.abort("blockedbyclient");
  });
  await page.addInitScript(() => {
    localStorage.setItem("xtbloom-lang", "zh");
    const original = Element.prototype.scrollIntoView;
    globalThis.__xtbloomScrollCalls = [];
    Element.prototype.scrollIntoView = function patchedScrollIntoView(options) {
      globalThis.__xtbloomScrollCalls.push({
        id: this.id || "",
        behavior: typeof options === "object" ? options.behavior || "auto" : "auto",
        block: typeof options === "object" ? options.block || "start" : "start",
      });
      return original.call(this, options);
    };
  });
});

test.afterEach(async ({ page }, testInfo) => {
  if (testInfo.status === testInfo.expectedStatus) return;
  const screenshot = testInfo.outputPath("full-page.png");
  await page.screenshot({ path: screenshot, fullPage: true }).catch(() => {});
  await testInfo.attach("full-page", { path: screenshot, contentType: "image/png" }).catch(() => {});
  await testInfo.attach("page.html", {
    body: Buffer.from(await page.content().catch(() => "")),
    contentType: "text/html",
  });
  await testInfo.attach("layout-diagnostics.json", {
    body: Buffer.from(JSON.stringify(await layoutDiagnostics(page).catch((error) => ({ error: String(error) })), null, 2)),
    contentType: "application/json",
  });
});

test("engine loading status is compact, accessible, and does not block editing", async ({ page }, testInfo) => {
  let releaseManifest;
  const manifestGate = new Promise((resolve) => { releaseManifest = resolve; });
  await page.route("**/engine-manifest.json*", async (route) => {
    await manifestGate;
    await route.continue();
  });

  try {
    await page.goto("/", { waitUntil: "domcontentloaded" });
    const overlay = page.locator("#overlay");
    await expect(overlay).toBeVisible();
    await expect(overlay).toHaveAttribute("role", "status");
    await expect(overlay).toHaveAttribute("aria-live", "polite");

    const loadingStyle = await overlay.evaluate((element) => {
      const style = getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      return {
        backdropFilter: style.backdropFilter,
        pointerEvents: style.pointerEvents,
        width: rect.width,
        height: rect.height,
        viewportWidth: window.innerWidth,
        viewportHeight: window.innerHeight,
        deferred: [".panel-output", ".roadmap", ".footer"].map((selector) => {
          const deferredStyle = getComputedStyle(document.querySelector(selector));
          return {
            selector,
            contentVisibility: deferredStyle.contentVisibility,
            containIntrinsicSize: deferredStyle.containIntrinsicSize,
          };
        }),
      };
    });
    expect(loadingStyle.backdropFilter).toBe("none");
    expect(loadingStyle.pointerEvents).toBe("none");
    expect(loadingStyle.width).toBeLessThan(loadingStyle.viewportWidth);
    expect(loadingStyle.height).toBeLessThan(loadingStyle.viewportHeight);
    if (testInfo.project.name === "chromium") {
      for (const deferred of loadingStyle.deferred) {
        expect(deferred.contentVisibility, deferred.selector).toBe("auto");
        expect(deferred.containIntrinsicSize, deferred.selector).toMatch(/auto\s+\d+px/);
      }
    }

    await page.locator("#xyz").fill("H 0 0 0");
    await expect(page.locator("#xyz")).toHaveValue("H 0 0 0");
  } finally {
    releaseManifest();
  }
});

test("verified regional 3Dmol starts directly without secondary probe traffic", async ({ context, page }, testInfo) => {
  test.skip(testInfo.project.name !== "chromium", "request-order evidence is owned by Chromium");
  await page.addInitScript(() => {
    const NativeDateTimeFormat = Intl.DateTimeFormat;
    Intl.DateTimeFormat = function DateTimeFormat(...args) {
      const formatter = NativeDateTimeFormat(...args);
      const nativeResolvedOptions = formatter.resolvedOptions.bind(formatter);
      formatter.resolvedOptions = () => ({ ...nativeResolvedOptions(), timeZone: "UTC" });
      return formatter;
    };
    Intl.DateTimeFormat.prototype = NativeDateTimeFormat.prototype;
  });
  const pinnedThreeDmol = await readFile(new URL(
    "../../node_modules/3dmol/build/3Dmol-min.js",
    import.meta.url,
  ));
  const requests = [];
  let releaseSecondary;
  const secondaryGate = new Promise((resolve) => { releaseSecondary = resolve; });
  context.on("request", (request) => {
    const url = new URL(request.url());
    if (url.pathname.endsWith("/3Dmol-min.js")) {
      requests.push({
        hostname: url.hostname,
        pathname: url.pathname,
        range: request.headers().range || "",
      });
    }
  });
  await context.route(
    "https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js",
    async (route) => route.fulfill({
      status: 200,
      contentType: "text/javascript",
      body: pinnedThreeDmol,
    }),
  );
  await context.route(
    "https://cdn.jsdmirror.com/npm/3dmol@2.5.5/build/3Dmol-min.js",
    async (route) => {
      await secondaryGate;
      await route.abort("timedout");
    },
  );

  try {
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await expect.poll(() => page.evaluate(() => Boolean(window.$3Dmol))).toBe(true);
    expect(requests).toEqual([{
      hostname: "cdn.jsdelivr.net",
      pathname: "/npm/3dmol@2.5.5/build/3Dmol-min.js",
      range: "",
    }]);
  } finally {
    releaseSecondary();
  }
});

test("delayed OpenChemLib remains background-only for engine and XYZ readiness", async ({ context, page }, testInfo) => {
  test.skip(testInfo.project.name !== "chromium", "readiness ordering is owned by Chromium");
  let releaseOpenChemLib;
  const openChemLibGate = new Promise((resolve) => { releaseOpenChemLib = resolve; });
  const openChemLibRequests = [];
  await context.route("**/openchemlib@9.21.0/dist/*", async (route) => {
    openChemLibRequests.push(route.request().url());
    await openChemLibGate;
    await route.abort("timedout");
  });

  try {
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await expect.poll(() => openChemLibRequests.length).toBeGreaterThanOrEqual(2);
    await expect(page.locator("#engine-badge")).toHaveClass(/ok/, { timeout: 180_000 });
    await expect.poll(() => page.evaluate(() => Boolean(window.$3Dmol))).toBe(true);
    await expect(page.locator("#run")).toBeEnabled();
    await page.locator("#xyz").fill("H 0 0 0");
    await expect(page.locator("#xyz")).toHaveValue("H 0 0 0");
  } finally {
    releaseOpenChemLib();
  }
});

test("SMILES Worker requires a hash version and ignores a stale helper", async ({ context, page }) => {
  const contentVersion = "d".repeat(64);
  const helperRequests = [];
  /* Isolate the Worker contract from the application, which starts its own
   * optional SMILES Worker asynchronously and would make request counts race. */
  await context.route("**/smiles-worker-contract.html", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "text/html",
      body: "<!doctype html><title>SMILES Worker contract</title>",
    });
  });
  await context.route("**/smiles_helpers.js*", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    helperRequests.push({
      url: url.href,
      resourceType: request.resourceType(),
      version: url.searchParams.get("xtbloom_version"),
    });
    if (!url.searchParams.has("xtbloom_version")) {
      await route.fulfill({
        status: 200,
        contentType: "application/javascript",
        body: [
          'export const OPEN_CHEMLIB_VERSION = "stale-cache-fixture";',
          'export const CDN_REGION_MAINLAND_CHINA = "mainland-china";',
          'export const CDN_REGION_GLOBAL = "global";',
          "export function smilesToGeometry() { return {}; }",
        ].join("\n"),
      });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/javascript",
      body: [
        'export const CDN_REGION_MAINLAND_CHINA = "mainland-china";',
        'export const CDN_REGION_GLOBAL = "global";',
        "export function smilesToGeometry() { return {}; }",
        "export async function loadOpenChemLibRuntime() {",
        "  return {",
        '    OCL: { version: "9.21.0" },',
        '    moduleUrl: "stub:openchemlib",',
        '    resourcesUrl: "stub:resources",',
        "  };",
        "}",
      ].join("\n"),
    });
  });

  await page.goto("/smiles-worker-contract.html", { waitUntil: "domcontentloaded" });
  const staleVersion = await page.evaluate(async () => {
    const stale = await import(new URL("smiles_helpers.js", window.location.href).href);
    return stale.OPEN_CHEMLIB_VERSION;
  });
  expect(staleVersion).toBe("stale-cache-fixture");

  const result = await page.evaluate(async (version) => {
    const workerUrl = new URL("smiles_worker.js", window.location.href);
    workerUrl.searchParams.set("xtbloom_version", version);
    workerUrl.searchParams.set("xtbloom_cdn_region", "global");
    const worker = new Worker(workerUrl, { type: "module" });
    return new Promise((resolve) => {
      const finish = (value) => {
        clearTimeout(timer);
        worker.terminate();
        resolve(value);
      };
      const timer = setTimeout(() => finish({ type: "timeout", error: "" }), 30000);
      worker.onmessage = (event) => {
        if (event.data?.type === "load-error" || event.data?.type === "ready") {
          finish(event.data);
        }
      };
      worker.onerror = (event) => {
        event.preventDefault();
        finish({ type: "worker-error", error: event.message });
      };
    });
  }, contentVersion);

  expect(result).toMatchObject({
    type: "ready",
    version: "9.21.0",
    moduleUrl: "stub:openchemlib",
    resourcesUrl: "stub:resources",
  });
  expect(helperRequests.filter((request) => request.version === null)).toHaveLength(1);
  expect(helperRequests.filter((request) => request.version === contentVersion)).toHaveLength(1);
  /* Chromium labels a module Worker's dynamic import as script, while WebKit
   * exposes the same successful module request to Playwright as xhr. */
  expect(["script", "xhr"]).toContain(
    helperRequests.find((request) => request.version === contentVersion)?.resourceType,
  );

  const unversionedResult = await page.evaluate(async () => {
    const worker = new Worker(new URL("smiles_worker.js", window.location.href), {
      type: "module",
    });
    return new Promise((resolve) => {
      const finish = (value) => {
        clearTimeout(timer);
        worker.terminate();
        resolve(value);
      };
      const timer = setTimeout(() => finish({ type: "timeout", error: "" }), 30000);
      worker.onmessage = (event) => {
        if (event.data?.type === "load-error") finish(event.data);
      };
      worker.onerror = (event) => {
        event.preventDefault();
        finish({ type: "worker-error", error: event.message });
      };
    });
  });
  expect(unversionedResult).toMatchObject({
    type: "load-error",
    error: "SMILES Worker requires a 64-character SHA-256 content version",
  });
  expect(helperRequests.filter((request) => request.version === null)).toHaveLength(1);

  const helperRequestsBeforeInvalidVersion = helperRequests.length;
  const invalidVersionResult = await page.evaluate(async () => {
    const workerUrl = new URL("smiles_worker.js", window.location.href);
    workerUrl.searchParams.set("xtbloom_version", "latest");
    const worker = new Worker(workerUrl, { type: "module" });
    return new Promise((resolve) => {
      const finish = (value) => {
        clearTimeout(timer);
        worker.terminate();
        resolve(value);
      };
      const timer = setTimeout(() => finish({ type: "timeout", error: "" }), 30000);
      worker.onmessage = (event) => {
        if (event.data?.type === "load-error") finish(event.data);
      };
      worker.onerror = (event) => {
        event.preventDefault();
        finish({ type: "worker-error", error: event.message });
      };
    });
  });
  expect(invalidVersionResult).toMatchObject({
    type: "load-error",
    error: "SMILES Worker requires a 64-character SHA-256 content version",
  });
  expect(helperRequests).toHaveLength(helperRequestsBeforeInvalidVersion);
});

test("mobile and desktop layouts survive both methods and completed states", async ({ page }, testInfo) => {
  const widths = widthsFor(testInfo.project.name);
  const siteOrigin = new URL(String(testInfo.project.use.baseURL)).origin;
  const enginePackageRequests = [];
  const sideModuleResponses = [];
  const threeDmolRequests = [];
  /* Observe the real application fetches without intercepting or replacing
   * them. Register before navigation because the engine resources begin
   * downloading as soon as the module graph finishes bootstrap. */
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.pathname.endsWith("/3Dmol-min.js")) {
      threeDmolRequests.push({
        origin: url.origin,
        range: request.headers().range || "",
      });
    }
    if (
      url.pathname.endsWith("/xtbloom_web.side.wasm") ||
      url.pathname.endsWith("/xtbloom_web.data")
    ) {
      enginePackageRequests.push({
        pathname: url.pathname,
        version: url.searchParams.get("xtbloom_version"),
      });
    }
  });
  page.on("response", (response) => {
    const url = new URL(response.url());
    if (url.pathname.endsWith("/xtbloom_web.side.wasm")) {
      sideModuleResponses.push({
        contentType: response.headers()["content-type"] || "",
        ok: response.ok(),
      });
    }
  });
  await page.setViewportSize({ width: widths[0], height: VIEWPORT_HEIGHT });
  await page.goto("/", { waitUntil: "domcontentloaded" });
  await expect(page.locator("#engine-badge")).toHaveClass(/ok/, { timeout: 180_000 });
  await expect.poll(() => sideModuleResponses.length).toBe(1);
  const sideModuleRequests = enginePackageRequests.filter(
    ({ pathname }) => pathname.endsWith("/xtbloom_web.side.wasm"),
  );
  const legacyDataRequests = enginePackageRequests.filter(
    ({ pathname }) => pathname.endsWith("/xtbloom_web.data"),
  );
  expect(sideModuleRequests).toHaveLength(1);
  expect(sideModuleRequests[0].version).toMatch(/^[0-9a-f]{64}$/);
  expect(legacyDataRequests).toHaveLength(0);
  expect(sideModuleResponses[0]).toMatchObject({ ok: true });
  expect(sideModuleResponses[0].contentType).toMatch(/^application\/wasm(?:\s*;|$)/i);
  /* Every non-local request is blocked by beforeEach. The failed preferred
   * full attempt unlocks one bounded local probe, followed by exactly one
   * verified full local transfer. */
  await expect.poll(() => page.evaluate(() => Boolean(window.$3Dmol))).toBe(true);
  expect(threeDmolRequests.filter(({ origin }) => origin === siteOrigin)).toEqual([
    { origin: siteOrigin, range: "bytes=0-65535" },
    { origin: siteOrigin, range: "" },
  ]);
  await expect(page.locator("#run")).toBeEnabled();
  await expect(page.locator("#method")).toHaveValue("2");
  await expect(page.locator("#method option")).toHaveCount(2);

  for (const width of widths) {
    await page.setViewportSize({ width, height: VIEWPORT_HEIGHT });
    await expect(page.locator("#advanced-settings")).toHaveJSProperty("open", width > 600);
    await expect(page.locator("#footer-legal")).toHaveJSProperty("open", width > 600);
    for (const language of ["zh", "en"]) {
      await setLanguage(page, language);
      await assertLayout(page, `${testInfo.project.name} initial ${width}px ${language}`);
      if (width <= 600) {
        await assertTouchTargets(
          page,
          ["#lang-toggle", "#method", "#run", "#opt-run", "#smiles-generate", ".roadmap-item a.btn"],
          `${testInfo.project.name} initial ${width}px ${language}`,
        );
      }
    }
  }

  await page.setViewportSize({ width: widths[0], height: VIEWPORT_HEIGHT });
  await setLanguage(page, "zh");
  const advanced = page.locator("#advanced-settings");
  const advancedSummary = advanced.locator("summary");
  await expect(advanced).not.toHaveAttribute("open", "");
  await advancedSummary.click();
  await page.locator("#maxiter").fill("1");
  await page.locator("#opt-maxiter").fill("2");
  await page.locator("#opt-gradtol").fill("1e-12");
  await advancedSummary.click();
  await advancedSummary.click();
  await expect(page.locator("#maxiter")).toHaveValue("1");
  await expect(page.locator("#opt-maxiter")).toHaveValue("2");
  await setLanguage(page, "en");
  await expect(advanced).toHaveAttribute("open", "");
  await advancedSummary.click();

  const footerLegal = page.locator("#footer-legal");
  await footerLegal.locator("summary").click();
  await expect(footerLegal.locator("a")).toHaveCount(8);
  await assertLayout(page, `${testInfo.project.name} expanded mobile footer`);
  await footerLegal.locator("summary").click();

  const scrollCallsBeforeEdit = await page.evaluate(() => globalThis.__xtbloomScrollCalls.length);
  await page.locator("#charge").fill("0");
  await expect.poll(() => page.evaluate(() => globalThis.__xtbloomScrollCalls.length)).toBe(scrollCallsBeforeEdit);

  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.locator("#run").click();
  await expect(page.locator("#error")).toBeVisible();
  await expect(page.locator("#error")).toContainText(/SCC|eigensolver|特征/);
  await expectResultPanelVisible(page);
  await expect.poll(() => page.evaluate(() => globalThis.__xtbloomScrollCalls.at(-1))).toMatchObject({
    id: "result-panel",
    behavior: "auto",
  });

  await advancedSummary.click();
  await page.locator("#maxiter").fill("250");
  await advancedSummary.click();
  await page.locator("#run").click();
  await expect(page.locator("#output-tools")).toBeVisible();
  await expect(page.locator("#stat-atoms")).toHaveText("3");
  await expect(page.locator("#result-method")).toHaveText("GFN2-xTB");
  const gfn2Energy = await page.locator("#energy").textContent();
  await expectResultPanelVisible(page);

  const scrollCallsBeforeMethodChange = await page.evaluate(() => globalThis.__xtbloomScrollCalls.length);
  await page.locator("#method").selectOption("1");
  await expect(page.locator("#output-tools")).toBeHidden();
  await expect(page.locator("#result-method")).toBeHidden();
  await expect(page.locator("#energy")).toHaveText("—");
  await expect(page.locator("#stat-atoms")).toHaveText("–");
  await expect(page.locator("#table-wrap")).toBeHidden();
  await expect.poll(() => page.evaluate(() => globalThis.__xtbloomScrollCalls.length)).toBe(scrollCallsBeforeMethodChange);
  await page.locator("#run").click();
  await expect(page.locator("#output-tools")).toBeVisible();
  await expect(page.locator("#result-method")).toHaveText("GFN1-xTB");
  await expect(page.locator("#energy")).not.toHaveText(gfn2Energy || "");

  await page.locator("#method").selectOption("2");
  if (testInfo.project.name === "chromium") {
    /* Chromium owns the complete matrix, including the callback-backed replay
     * controls. WebKit still covers a completed GFN2 calculation at phone
     * widths; the optimizer callback is an optional visualization path. */
    await page.locator("#opt-run").click();
    await expect(page.locator("#opt-apply")).toBeVisible({ timeout: 180_000 });
    await expect(page.locator("#replay")).toBeVisible();
  } else {
    await page.locator("#run").click();
    await expect(page.locator("#output-tools")).toBeVisible();
  }
  await expect(page.locator("#result-method")).toHaveText("GFN2-xTB");
  await expectResultPanelVisible(page);

  for (const width of widths) {
    await page.setViewportSize({ width, height: VIEWPORT_HEIGHT });
    for (const language of ["zh", "en"]) {
      await setLanguage(page, language);
      await assertLayout(page, `${testInfo.project.name} completed ${width}px ${language}`);
      if (width <= 600) {
        await assertTouchTargets(
          page,
          ["#lang-toggle", "#method", "#run", "#opt-run", "#copy-json", "#opt-apply", "#replay-play", ".roadmap-item a.btn"],
          `${testInfo.project.name} completed ${width}px ${language}`,
        );
      }
    }
  }

  /* A method change must erase every number and trajectory owned by the old
   * model, not merely hide the model badge and result actions. */
  if (testInfo.project.name === "chromium") {
    await expect(page.locator("#traj")).toBeVisible();
    await expect(page.locator("#replay")).toBeVisible();
  }
  await page.locator("#method").selectOption("1");
  await expect(page.locator("#energy")).toHaveText("—");
  await expect(page.locator("#energy-ev")).toHaveText("—");
  await expect(page.locator("#energy-kcal")).toHaveText("—");
  await expect(page.locator("#stat-atoms")).toHaveText("–");
  await expect(page.locator("#stat-iter")).toHaveText("–");
  await expect(page.locator("#stat-conv")).toHaveText("–");
  await expect(page.locator("#stat-ms")).toHaveText("–");
  await expect(page.locator("#table-wrap")).toBeHidden();
  await expect(page.locator("#traj")).toBeHidden();
  await expect(page.locator("#replay")).toBeHidden();
});

test("desktop disclosures are expanded by default", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "chromium", "Chromium owns the complete desktop matrix");
  await page.setViewportSize({ width: 1024, height: VIEWPORT_HEIGHT });
  await page.goto("/", { waitUntil: "domcontentloaded" });
  await expect(page.locator("#advanced-settings")).toHaveJSProperty("open", true);
  await expect(page.locator("#footer-legal")).toHaveJSProperty("open", true);
  await assertLayout(page, "chromium desktop default disclosures");
});
