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
- `bootstrap.js` revalidates the manifest and verifies the versioned app/helper
  module graph before importing application code, so a deployment cannot link
  mismatched cached modules before recovery UI exists; the small inline loader
  in `index.html` also retries a transient failure fetching `bootstrap.js`;
- `app.js` downloads the five engine resources under one file/byte progress
  ledger, retries transient startup failures with generation-safe cleanup, and
  passes the wasm and Emscripten data bytes into the Worker;
- `smiles_worker.js` independently loads the pinned OpenChemLib release,
  generates explicit-hydrogen 3D conformers, and applies MMFF94
  pre-relaxation;
- `3dmol` renders the current geometry; and
- `app.js` provides single-point calculation and an adapter-local L-BFGS
  optimization loop.

The optional SMILES worker never gates ordinary XYZ calculations. The
optimizer repeatedly calls the same single-point adapter and is not part of the
stable C ABI.

The build hashes `app.js` and the five engine files into
`engine-manifest.json`. The browser revalidates only that small manifest on
refresh, then addresses every application/engine asset with the shared content
version. An unchanged version stays cacheable; a transient failure reloads the
complete resource set under that same content
version, replacing rather than abandoning the reusable cache entries. Digest
verification prevents the Worker glue, wasm, and preloaded data from being
mixed across attempts. Late messages from a failed Worker are ignored by a
monotonically increasing loader generation.

The deployed build is wasm32 so it works without browser Memory64 support. CI
also builds wasm64 and compares its public results with wasm32 as a
pointer-width and numerical parity gate; wasm64 is not deployed.

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
```

The staged site is `build/wasm32-web/web/site`. Serve that directory over HTTP
rather than opening `index.html` directly, because module Workers and
WebAssembly loading require an origin.

## Validation

```console
bun install --frozen-lockfile --cwd web
bun test --cwd web
bun web/tests/openchemlib_smoke.mjs
node web/tests/wasm_smoke.mjs build/wasm32-web/web/site
```

CI additionally compiles wasm64, checks C ABI layout for both pointer widths,
compares wasm32/wasm64 scientific results, and audits the exact deployed legal
payload.

## Dependencies and provenance

`web/package.json` pins 3Dmol.js for the built site. The optional SMILES worker
loads exact OpenChemLib 9.21.0 CDN URLs whose revisions, file sizes, and
SHA-256 digests are recorded in `web/openchemlib_manifest.json`. The deployed
site carries the project license, third-party notices, applicable license
texts, and parameter provenance.

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
