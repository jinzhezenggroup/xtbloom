import { defineConfig } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const webRoot = path.dirname(fileURLToPath(import.meta.url));
const siteRoot = path.resolve(
  process.env.XTBLOOM_WEB_SITE || path.join(webRoot, "../build/wasm32-web/web/site"),
);
const port = Number(process.env.XTBLOOM_WEB_PORT || 4173);
const baseURL = `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: "./tests/browser",
  outputDir: "./test-results/browser",
  timeout: 180_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [
    ["line"],
    ["html", { outputFolder: "playwright-report", open: "never" }],
  ],
  use: {
    baseURL,
    viewport: { width: 320, height: 900 },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: `python3 -m http.server ${port} --bind 127.0.0.1 --directory ${JSON.stringify(siteRoot)}`,
    url: `${baseURL}/engine-manifest.json`,
    reuseExistingServer: false,
    timeout: 30_000,
  },
  projects: [
    { name: "chromium", use: { browserName: "chromium" } },
    { name: "webkit", use: { browserName: "webkit", hasTouch: true, isMobile: true } },
  ],
});
