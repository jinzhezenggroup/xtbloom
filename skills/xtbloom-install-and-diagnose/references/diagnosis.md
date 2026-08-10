# xTBloom Runtime Diagnosis

Diagnose the earliest failing boundary and preserve its exact message. Do not change several providers, environment variables, or build options at once.

## Diagnostic Layers

| Layer | What success proves | What it does not prove |
| --- | --- | --- |
| Distribution metadata | A Python distribution named `xtbloom` is installed | Package import or native compatibility |
| `import xtbloom` | Python modules and metadata import | Native-library discovery or a backend |
| `library_path()` | A candidate `libxtbloom` was found | The library and providers load correctly |
| Native version query | The shared library loaded and exported the expected symbol | CPU/CUDA context creation |
| Explicit context probe | The requested backend initialized | Scientific inference or conformance |
| Explicit calculation | The selected backend executed that workload | Broad scientific or performance validation |

Run `scripts/diagnose_xtbloom.py` with the same Python executable and environment as the application. Its JSON keeps these layers separate.

## Import and Library Discovery Failures

- Confirm `sys.executable` belongs to the intended environment.
- Confirm the distribution and imported package paths point to the same installation.
- Inspect whether `XTBLOOM_LIBRARY` is set. A stale override takes precedence over a bundled wheel library.
- If no library candidate exists after a source sync, rebuild with `--no-editable --reinstall-package xtbloom`; editable source layouts do not satisfy bundled-library discovery.
- Preserve loader diagnostics. Do not add broad global library paths before identifying the missing image and intended provider.

## CPU Context Failures

Source builds require one monolithic shared LP64 LAPACKE+CBLAS provider. Common invalid substitutions include:

- ILP64 interfaces;
- separate `liblapack` and `libblas` images;
- static archives; and
- a library without the provider-local thread-control contract.

Pass an absolute compatible provider through `CMAKE_ARGS=-DXTBLOOM_CPU_LINALG_LIBRARY=...` and reinstall the package. Official Linux wheels should use their bundled private provider; do not add `scipy-openblas32` as a runtime dependency.

## CUDA Context Failures

Check each boundary independently:

1. The package was compiled with CUDA explicitly or through a confirmed `AUTO` configuration.
2. The system has a real NVIDIA GPU and compatible driver.
3. The runtime can locate cudart, cuBLAS, and cuSOLVER for the build CUDA major.
4. The selected device is usable by the current process and scheduler allocation.
5. An explicit `backend="cuda"` context succeeds.

The `cuda12` extra installs supported NVIDIA host-library packages but cannot install the driver, GPU, toolkit compiler, or scheduler allocation. Hosted compile-only checks and loader stubs are not runtime evidence.

If `backend="auto"` succeeds, read the reported resolved backend. A CPU resolution is a valid fallback only when fallback was requested; it is not a CUDA pass.

## End-to-End Smoke Test

After context diagnosis, run a small explicit-backend single-point calculation. Use bohr input and report Hartree energy and Hartree/bohr forces. The high-level Python `electronic_temperature` value is kelvin; at finite temperature the returned variational energy is the Helmholtz free energy.

For batches, a successful call can contain failed systems. Inspect `failed_indices`, `per_system_status`, `scc_converged`, and `scc_iterations`; failed requested floating-point slices are NaNs while successful peers remain valid.

Do not use a smoke test to claim support for GFN1-xTB, ROCm, PBC/lattice inputs, solvation, native optimization, molecular dynamics, or Hessians.
