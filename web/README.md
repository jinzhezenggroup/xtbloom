# Web demo implementation

The user-facing site is deployed at
[`https://xtbloom.jinzhezeng.group`](https://xtbloom.jinzhezeng.group).
Usage and capability boundaries are documented in the
[browser demo guide](../docs/user-guide/browser-demo.md). This file is for
maintainers of the Web/WASM implementation.

## Architecture

The deployment runs entirely in the browser:

- the xTBloom CPU backend is compiled as a single-threaded wasm32 module;
- `worker.js` owns the synchronous native calls so calculations do not block
  the UI thread;
- `bootstrap.js` probes a bounded prefix of the pinned 3Dmol.js asset from
  JSDMirror, jsDelivr, and the site origin, then loads the fastest verified
  source with ranked fallback. That optional routing runs in parallel with
  manifest revalidation and verification of the versioned app/helper module
  graph. The manifest gives the lazily loaded SMILES Worker/helper graph the
  same content version without making it gate the core app; the small inline
  loader in `index.html` also retries a transient failure fetching
  `bootstrap.js`;
- `app.js` downloads the five engine resources under one file/byte progress
  ledger, retries transient startup failures with generation-safe cleanup, and
  passes the wasm and Emscripten data bytes into the Worker;
- `c60_case.js` supplies the visible C60 preset and the independent native-CPU
  GFN2 checkpoints used by the browser scientific regression;
- `smiles_worker.js` independently loads one complete size- and SHA-256-
  verified OpenChemLib provider pair using the measured JSDMirror/jsDelivr
  order, generates explicit-hydrogen 3D conformers, and applies MMFF94
  pre-relaxation;
- `3dmol` renders the current geometry after the fastest of JSDMirror,
  jsDelivr, or the retained site-local bundle passes byte verification; and
- `app.js` provides a GFN1/GFN2 method selector, single-point calculation, and
  an adapter-local L-BFGS optimization loop. GFN2 is the UI default. It
  validates the coordinates box independently of the compute path: valid input
  is previewed live (debounced while typing), malformed input keeps the last
  valid preview and is flagged inline, and the calculate actions stay disabled
  until a valid structure is present.

The optional SMILES worker never gates ordinary XYZ calculations. The
optimizer repeatedly calls the same single-point adapter and is not part of the
stable C ABI. Successive optimizer evaluations warm-start SCC from the previous
fully converged electronic state (ABI-v2 `SCC_START_WARM`), while standalone
single-point calculations always start SCC fresh, so user calculations cannot
inherit electronic state from an unrelated request.

The build hashes the application module graph, optional SMILES Worker/helper
graph, and five engine files into `engine-manifest.json`. The browser
revalidates only that small manifest on refresh, verifies `app.js`,
`app_helpers.js`, and `c60_case.js` before linking them, while leaving the
optional SMILES pair lazy so it cannot delay the core app. Every manifest asset
is addressed with the shared SHA-256-derived content version, and the SMILES
Worker refuses to import a helper unless that 64-character hash is present.
It therefore imports the helper from the same cache generation instead of an
older stable URL. An unchanged version stays cacheable; a transient failure
reloads the complete resource set under that same content version, replacing
rather than abandoning reusable cache entries. Digest verification prevents the UI,
Worker glue, wasm, and preloaded data from being mixed across attempts; the
digest-derived shared cache key provides the corresponding generation boundary
for the lazy SMILES modules. Late messages from a failed Worker are ignored by
a monotonically increasing loader generation.

The CPU eigensolver still discovers the same LP64 LAPACKE/CBLAS symbols from a
preloaded side module named `libscipy_openblas.so`. For the Web build those
symbols are implemented using pinned Eigen 5.0.1 from the SHA-256-verified
official release archive; the filename and loader contract remain unchanged,
and the compatibility filename does not mean the browser module contains
OpenBLAS. Eigen does not become a native xTBloom dependency.

Emscripten internally emits that preload package as `xtbloom_web.data`. The
staged site publishes the exact package bytes as `xtbloom_web.side.wasm`, and the
manifest retains the logical asset ID `data`. The loader supplies the verified
bytes through `getPreloadedPackage`, while the WebAssembly suffix lets static
hosts apply their normal Wasm content type and transport compression.

The deployed build is wasm32 so it works without browser Memory64 support. CI
also builds wasm64 and compares its public results with wasm32 as a
pointer-width and numerical parity gate; wasm64 is not deployed.

## Production deployments

The `wasm-web-pages` workflow deploys the same validated wasm32 artifact to
both GitHub Pages and Cloudflare Pages after a push to `main`. Pull requests
build the wasm32/wasm64 sites and run the complete parity, scientific, and
legal checks, but neither production deployment job runs on a pull request.

Cloudflare uses a pre-existing **Direct Upload** Pages project. Create the
GitHub `cloudflare-pages` environment, restrict its deployment branches to
`main`, and configure:

- environment secret `CLOUDFLARE_API_TOKEN`: a custom Cloudflare token limited
  to `Account / Cloudflare Pages / Edit` for the target account;
- environment variable `CLOUDFLARE_ACCOUNT_ID`: the target account ID; and
- environment variable `CLOUDFLARE_PROJECT_NAME`: the exact Direct Upload
  project name.

Set the Cloudflare project's production branch to `main` when creating it. The
deployment job requires that the project and GitHub environment already exist;
it does not create them. No pull-request permission is granted because
Cloudflare deployment runs only on `main`.

Wrangler 4.123.0 and its complete transitive dependency graph are pinned by
`tools/cloudflare-pages/package-lock.json`. The credential-free validation job
runs `npm ci --omit=optional --ignore-scripts`, verifies the Wrangler version,
and archives that minimal Linux installation. The production job restores the
archive without checking out repository code or invoking npm; its
credential-bearing step only invokes the locked Wrangler installation.
Re-audit the lockfile and every explicit deployment package before upgrading
them.

Attaching a custom domain is a separate Cloudflare Pages configuration step.
Add the hostname through the project's **Custom domains** screen before
changing DNS; pointing a CNAME at Pages without first associating the hostname
can leave the site unavailable.

## Local build

Use the Emscripten version pinned in [`.github/workflows/pages.yml`](../.github/workflows/pages.yml):

```console
emcmake cmake -S . -B build/wasm32-web -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=OFF \
  -DXTBLOOM_BUILD_TESTS=OFF \
  -DXTBLOOM_BUILD_WEB_DEMO=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-m32 -fPIC" \
  -DCMAKE_CXX_FLAGS="-m32 -fPIC"
cmake --build build/wasm32-web --parallel
cmake --build build/wasm32-web --target xtbloom_web_linalg_test --parallel
```

CMake downloads the official Eigen 5.0.1 archive only for this Web-enabled
configuration and verifies its fixed SHA-256. For an offline build, download
that exact archive ahead of time, verify it, and pass its path explicitly:

```console
python3 tools/eigen_dependency.py verify-archive /path/eigen-5.0.1.tar.gz
emcmake cmake -S . -B build/wasm32-web -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=OFF \
  -DXTBLOOM_BUILD_TESTS=OFF \
  -DXTBLOOM_BUILD_WEB_DEMO=ON \
  -DXTBLOOM_WEB_EIGEN_ARCHIVE=/path/eigen-5.0.1.tar.gz \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-m32 -fPIC" \
  -DCMAKE_CXX_FLAGS="-m32 -fPIC"
```

The staged site is `build/wasm32-web/web/site`. Serve that directory over HTTP
rather than opening `index.html` directly, because module Workers and
WebAssembly loading require an origin.

## Validation

```console
python3 tools/eigen_dependency.py check
npm ci --prefix web --ignore-scripts --no-audit --no-fund
npm --prefix web test
node web/tests/3dmol_smoke.mjs
node web/tests/openchemlib_smoke.mjs
node web/tests/wasm_smoke.mjs build/wasm32-web/web/site
npm --prefix web run browser:install
XTBLOOM_WEB_SITE="$PWD/build/wasm32-web/web/site" \
  npm --prefix web run test:browser
```

Set `XTBLOOM_WEB_PORT` when port 4173 is unavailable. The Playwright test serves
the staged site itself and blocks every non-local request. It therefore proves
both that ordinary XYZ calculations do not depend on the optional OpenChemLib
CDN path and that 3Dmol falls through to the verified site-local bundle.

The provider target independently checks LAPACKE workspace and failure
behavior, Cholesky factorization and condition estimation, eigensolve, all
accepted GEMM transpose combinations, and the TRSM side/triangle/transpose/
diagonal matrix. The Web smoke suite includes the visible neutral-singlet C60
preset (60 atoms, 240 orbitals) and compares energy, charges, forces, SCC
status, iterations, total charge, and total force with native public-C-ABI
GFN2 checkpoints. It also runs GFN2 → GFN1 → GFN2 through one context, compares
the GFN1 H3+ energy and forces with the independent conformance golden, and
exercises a GFN1 optimization step.

CI additionally compiles wasm64, runs the same provider and C60 gates, checks C
ABI layout for both pointer widths, compares wasm32/wasm64 public results, and
audits the exact deployed legal payload. Chromium covers 320, 360, 375, 390,
430, 768, and 1024 CSS-pixel viewports; WebKit covers representative phone
widths. Both languages, initial/completed states, disclosures, touch targets,
and result-panel movement are deployment gates, with traces, screenshots, DOM,
and layout diagnostics retained on failure.

## Dependencies and provenance

`web/package.json` pins 3Dmol.js for the built site's local fallback and
Playwright 1.62.1 for developer/CI-only browser regression. The browser probes
the exact 3Dmol.js 2.5.5 asset from JSDMirror, jsDelivr, and that local copy,
uses the fastest measured source, and verifies the complete bundle before
execution. When probe results are close, recognized mainland-China time zones
prefer JSDMirror and other environments prefer jsDelivr. Playwright's browser
runtimes and its macOS-only optional `fsevents` dependency remain outside all
distributed artifacts. The optional SMILES worker reuses the measured CDN
order and loads an exact OpenChemLib 9.21.0 module/resource pair; provider URLs,
revisions, file sizes, and SHA-256 digests are recorded in
`web/openchemlib_manifest.json`. Eigen 5.0.1 is
obtained from tag revision `bc3b39870ecb690a623a3f49149a358b95c5781d`;
the official release archive has SHA-256
`e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec`.
`cmake/3rdparty/eigen_manifest.json` pins that archive and the nine exact legal
records retained under `LICENSES/eigen/`. Source checkouts and source
distributions do not carry the Eigen archive or header tree. The deployed Pages
site carries the compiled provider, upstream license records, and provenance
manifest, while native CMake installs and Python wheels exclude all Eigen
material. The site also carries the project license, third-party notices,
applicable dependency license texts, and parameter provenance.

Do not replace pinned URLs with floating versions or add a Web dependency
without updating the provenance, license payload, lockfile, and deployment
checks.

## README screenshot

`docs/assets/web-demo-ethanol.png` is a project-owned capture of the xTBloom
browser demo after the `?smiles=CCO` workflow completed successfully. It records
a real 9-atom ethanol optimization and WebGL molecular rendering; no synthetic
molecular image or benchmark timing was substituted. The PNG is stripped of
browser metadata. 3Dmol.js and OpenChemLib remain covered by the provenance and
license boundaries above; their JavaScript and resource payloads are not copied
into the screenshot asset.

Capture record: 2026-08-10, the `web/` assets versioned alongside this image,
route `?smiles=CCO`, 7 optimization steps, final energy `-11.3918507000 Eh`,
SHA-256
`90ceebb83e9c08ae2a3d6208b6c210a6ba9851e5a7d77bcbb2678a959ffc8f5f`.
