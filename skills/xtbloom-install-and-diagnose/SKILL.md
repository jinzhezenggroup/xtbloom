---
name: xtbloom-install-and-diagnose
description: Install and diagnose xTBloom from PyPI, a source checkout, or an installed native distribution, locate its native library, and verify whether the requested CPU, CUDA, or automatic backend can create a real runtime context. Use when setting up xTBloom, choosing CPU versus CUDA artifacts or build options, investigating import or shared-library failures, checking LP64 linear algebra or NVIDIA runtime prerequisites, or proving which backend `auto` actually selected.
---

# Install and Diagnose xTBloom

Establish the intended deployment and backend before changing an environment. Distinguish package import, native-library discovery, context creation, and successful inference; each proves a different boundary.

## Choose the Installation Path

Read [references/installation.md](references/installation.md) before installing or rebuilding.

1. Determine whether the user needs the published PyPI package, an existing Python environment, a source checkout, or a native CMake install.
2. Record the platform, Python version, desired backend, and whether CPU fallback is acceptable.
3. Use `backend="cpu"` or `backend="cuda"` when the named backend is required. Use `"auto"` only when CUDA preference with CPU fallback is intended.
4. Prefer the reviewed PyPI wheel for ordinary Python use. Use a source build only for development, an unsupported target, or an explicitly requested native configuration.
5. For a one-off agent task, prefer an isolated `uv run` invocation over changing the user's environment. Reinstall a local source package after changing native build environment variables because a cached wheel may otherwise retain the previous configuration.

Do not install `scipy-openblas32` as an xTBloom runtime dependency. Published
desktop wheels use it only as a reviewed build input and contain their private
provider; source builds need a compatible system provider.

## Diagnose in Layers

Read [references/diagnosis.md](references/diagnosis.md) before interpreting an error.

For a fresh probe of the current PyPI release, run the bundled read-only
inventory through its PEP 723 metadata so xTBloom need not be preinstalled:

```bash
UV_DEFAULT_INDEX=https://pypi.org/simple \
  uv run --script scripts/diagnose_xtbloom.py
```

The default mode imports the package and resolves the candidate native library without creating a backend context. Probe exactly the backend the user needs:

```bash
UV_DEFAULT_INDEX=https://pypi.org/simple \
  uv run --script scripts/diagnose_xtbloom.py --backend cpu
UV_DEFAULT_INDEX=https://pypi.org/simple \
  uv run --with 'xtbloom[cuda12]>=0.1.1' \
  --script scripts/diagnose_xtbloom.py --backend cuda
UV_DEFAULT_INDEX=https://pypi.org/simple \
  uv run --script scripts/diagnose_xtbloom.py --backend auto
```

The `--script` form creates an isolated environment and therefore proves the
declared PyPI artifact, not an already configured application environment. To
diagnose an active virtual environment, invoke the helper with that environment's
Python, for example `uv run --active --no-project python
scripts/diagnose_xtbloom.py --backend cpu`. From an already synced xTBloom source
checkout, use `uv run --no-sync python
skills/xtbloom-install-and-diagnose/scripts/diagnose_xtbloom.py`. The helper emits
JSON and returns nonzero when package import, library discovery, or the requested
context probe fails. It does not write files, change environment variables, or
submit a scientific calculation. Import only the intended installation; Python
imports can execute package code.

Interpret the layers precisely:

- A distribution version proves only that package metadata is installed.
- A successful import does not prove that `libxtbloom` can be found or loaded.
- A located library path does not prove that CPU or CUDA providers are usable.
- A successful `auto` probe reports the resolved backend but does not prove CUDA specifically.
- A successful explicit `cuda` context proves runtime creation on a real NVIDIA device, but not numerical inference or scientific correctness.
- Use an explicit single-point calculation for an end-to-end inference smoke test after diagnosis.

## Preserve User-Facing Contracts

When installation succeeds, keep these boundaries visible in generated examples and explanations:

- Native positions use bohr, energy uses Hartree, forces use Hartree/bohr, and charge uses elementary-charge units.
- The high-level Python `electronic_temperature` argument is in kelvin. At finite temperature, the reported variational energy is the electronic Helmholtz free energy.
- A batch call can succeed while individual SCC or eigensolver results fail. Inspect per-system diagnostics; failed floating-point slices contain NaNs and successful peers remain valid.
- Only GFN2-xTB is implemented. Do not claim GFN1-xTB, ROCm, lattice/PBC inputs, solvation, native geometry optimization, molecular dynamics, or Hessians.
- CUDA compilation, a visible GPU, or a loader stub is not by itself successful CUDA execution.

## Report the Narrow Result

State the exact executable, installation source, package and native versions, native-library candidate, requested backend, resolved backend, and full diagnostic on failure. Separate unavailable prerequisites from unsupported xTBloom features, and never report `auto` fallback as a CUDA pass.
