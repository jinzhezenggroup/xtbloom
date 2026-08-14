---
name: xtbloom-audit-dependency
description: Audit new or changed xTBloom dependencies, copied sources, generated datasets, parameter tables, CI actions, build downloads, native linking, install exports, sdists, and wheels for provenance, licensing, ABI, and payload correctness. Use for dependency or packaging changes and whenever external bytes enter source, build, install, or distribution artifacts.
---

# Audit a xTBloom Dependency

Trace external material from its origin through every distributed artifact. Read `AGENTS.md`, `THIRD_PARTY_NOTICES.md`, `CUDA_MKL_LINKING_EXCEPTION`, and the relevant generator or packaging documentation first.

## Inventory External Material

Use the diff and build configuration to list every new or changed external byte, including indirect cases:

- linked or dynamically loaded native libraries;
- Python runtime, test, build, and optional dependencies;
- copied or generated source, parameter tables, datasets, patches, licenses, and notices;
- CMake downloads, vendored tools, CI actions, cache installers, and container images;
- install-tree, sdist, and wheel payload additions.

Classify each item as build-only, test-only, runtime-provided, dynamically loaded, vendored source, copied data, generated data, or distributed binary. Do not assume one classification covers every artifact.

## Establish Provenance

For every item, record the upstream project and URL, exact revision/version, retrieval or generation procedure, SHA-256 of source inputs and outputs, license and notice source, local destination, and redistribution boundary. Prefer immutable revisions and reviewed artifact digests.

For generated data, preserve a deterministic generator and manifest. Regenerate through the documented workflow; never modify the derived file or its hash directly.

For GitHub Actions and downloaded tools, pin the reviewed immutable revision or digest where practical and verify that runtime version checks reject unsupported old versions rather than merely finding a command with the same name.

## Review Legal Boundaries

Update, as applicable:

- `THIRD_PARTY_NOTICES.md`;
- verbatim texts under `LICENSES/`;
- source/data provenance manifests and notices;
- `CUDA_MKL_LINKING_EXCEPTION` only with an explicit owner/legal decision;
- CMake install rules, `pyproject.toml`, `uv.lock`, and package checks.

Do not infer copyright compatibility from `dlopen`, optional installation, or absence from a wheel. Dynamic loading can still combine works. Stop and request an owner/legal decision when the license grant or distribution permission is unclear.

## Audit Native and Package Boundaries

Inspect each applicable artifact:

1. Source tree: required notices, licenses, manifests, generator checks, and no untracked external payload.
2. Shared library: exported symbol allowlist, `DT_NEEDED`, RPATH/RUNPATH, SONAME expectations, and optional-provider diagnostics.
3. Static install: package export and external consumer without accidental private dependencies.
4. Shared install: external CPU/CUDA consumer and lazy runtime behavior.
5. Sdist: exact included source, license, notice, and generated-data payload.
6. Wheels: both supported architectures, size, licenses, bundled libraries, CUDA dependency policy, and installed inference.
7. Python resolution: canonical PyPI index and unchanged `uv lock --check` after the intended lock update.

Use structured tools from `tools/licensing/`, `tests/licensing/`, `tests/abi/`, `python/ci/`, and `tests/install_consumer/` instead of ad hoc filename checks.

## Run the Required Gates

At minimum, run the relevant subset and return the results to the caller's
validation ledger. Use the locked project sessions so the current checkout and
its reviewed dependencies are tested without requiring globally installed
Python tools:

```bash
UV_DEFAULT_INDEX=https://pypi.org/simple \
  uv run --isolated --locked --only-group nox nox -s canonical
UV_DEFAULT_INDEX=https://pypi.org/simple uv lock --check
uv build --sdist --out-dir build/dist-license
uv run --no-sync python tools/licensing/check_licenses.py --source-root . \
  build/dist-license/*.tar.gz
```

Add installed-prefix licensing, shared/static consumers, symbol checks, CUDA dependency checks, and actual wheel inspection whenever those boundaries changed. A source-tree pass does not prove the sdist, wheel, or install tree.

## Report the Decision

Record the inventory, provenance pins, license basis, artifacts inspected, exact checks and counts, and unresolved legal or packaging questions. Do not state that vendor CUDA/MKL code is GPL-covered; preserve the repository's precise additional-permission and vendor-terms language.
