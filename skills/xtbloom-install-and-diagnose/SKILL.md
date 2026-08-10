---
name: xtbloom-install-and-diagnose
description: Install and diagnose xTBloom from a source checkout or an installed Python distribution, locate its native library, and verify whether the requested CPU, CUDA, or automatic backend can create a real runtime context. Use when setting up xTBloom, choosing CPU versus CUDA build options, investigating import or shared-library failures, checking LP64 linear algebra or NVIDIA runtime prerequisites, or proving which backend `auto` actually selected.
---

# Install and Diagnose xTBloom

Establish the intended deployment and backend before changing an environment. Distinguish package import, native-library discovery, context creation, and successful inference; each proves a different boundary.

## Choose the Installation Path

Read [references/installation.md](references/installation.md) before installing or rebuilding.

1. Determine whether the user has a source checkout, an installed wheel, or a native CMake install.
2. Record the platform, Python version, desired backend, and whether CPU fallback is acceptable.
3. Use `backend="cpu"` or `backend="cuda"` when the named backend is required. Use `"auto"` only when CUDA preference with CPU fallback is intended.
4. Do not invent a PyPI install command. xTBloom is not yet published on PyPI; follow the source-checkout workflow unless the user supplies a built wheel or another reviewed artifact.
5. Reinstall the local package after changing native build environment variables because an existing cached wheel may otherwise retain the previous configuration.

Do not install `scipy-openblas32` as an xTBloom runtime dependency. Official Linux wheels use it only as a reviewed build input and contain their private provider; source builds need a compatible system provider.

## Diagnose in Layers

Read [references/diagnosis.md](references/diagnosis.md) before interpreting an error.

Run the bundled read-only inventory first:

```bash
python3 scripts/diagnose_xtbloom.py
```

The default mode imports the package and resolves the candidate native library without creating a backend context. Probe exactly the backend the user needs:

```bash
python3 scripts/diagnose_xtbloom.py --backend cpu
python3 scripts/diagnose_xtbloom.py --backend cuda
python3 scripts/diagnose_xtbloom.py --backend auto
```

Run the script with the same Python executable and environment that will run the application. It emits JSON and returns nonzero when package import, library discovery, or the requested context probe fails. It does not install packages, write files, change environment variables, or submit a scientific calculation. Import only the intended xTBloom installation; Python imports can execute package code.

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
