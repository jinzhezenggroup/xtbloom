# gpuxtb documentation

gpuxtb exposes GFN2-xTB single-point inference through one stable C ABI and
Python interfaces built on that ABI. Start with the guide for your role.

## Browser demo

The experimental browser build is available at
<https://jinzhezeng.group/gpuxtb/>. It runs the CPU backend entirely on the
client as wasm32 without requiring Memory64, targeting modern iOS Safari,
Safari, Chrome, and Firefox with WebAssembly and module Worker support. It
includes an adapter-local L-BFGS geometry optimizer for interactive
demonstrations. The page also loads the pinned OpenChemLib 9.21.0 module and
resource registry from jsDelivr in the background. Its SMILES workflow adds
explicit hydrogens, generates a seeded 3D conformer, applies an MMFF94
pre-relaxation, and populates the existing XYZ and formal-charge inputs without
blocking ordinary XYZ use if the optional CDN dependency is unavailable.

The query parameter `?smiles=CCO` preloads the SMILES and, after both workers
are ready, automatically runs the gpuxtb geometry optimizer and writes the
final angstrom coordinates back to the XYZ editor. URL syntax requires a
literal `+` in a charged SMILES to be encoded as `%2B`, for example
`?smiles=%5BNH4%2B%5D`. The SMILES/conformer workflow and optimizer are browser
adapter features; neither is exported by the stable C ABI or supported as a
native-library capability. A separate wasm64 CI build verifies that
pointer-width changes do not alter ABI-local behavior or numerical results.
OpenChemLib provenance and exact CDN digests are recorded in
[`web/openchemlib_manifest.json`](../web/openchemlib_manifest.json).

## User guide

The [user guide](user-guide/index.md) covers supported capabilities,
installation, units, backend selection, and runtime behavior.

- [Python API](user-guide/python.md): single systems, ragged batches, spin,
  explicit point charges, periodic response, ASE, and dpdata.
- [C and C++ API](user-guide/c-api.md): native installation, a complete C
  example, descriptor ownership, CUDA memory, and error handling.
- [QM/MM usage](user-guide/qmmm.md): what gpuxtb computes and what the caller
  must provide.

## Theory guide

The [theory guide](theory/index.md) explains the numerical meaning behind the
public results rather than reproducing every implementation equation.

- [GFN2-xTB model and SCC](theory/gfn2.md)
- [Explicit point charges and periodic response](theory/qmmm.md)

## Developer guide

The [developer guide](developer-guide/index.md) documents contracts that must
remain true when changing the implementation.

- [Architecture](developer-guide/architecture.md)
- [Validation and scientific evidence](developer-guide/validation.md)
- [Packaging, dependencies, and licensing](developer-guide/packaging.md)

Repository contributors and coding agents must also follow
[`AGENTS.md`](../AGENTS.md). The installed public contract is
[`include/gpuxtb/gpuxtb.h`](../include/gpuxtb/gpuxtb.h).

## Current support

| Capability | Status |
| --- | --- |
| GFN2-xTB restricted and unrestricted energy/forces/charges | CPU and CUDA |
| Ragged batches and peer-local SCC/eigensolver failures | Supported |
| Host input/output descriptors | CPU and CUDA |
| CUDA-device and mixed descriptors | Low-level C ABI |
| Explicit point charges in SCC and point-charge forces | Supported |
| Caller-supplied periodic `b + A q` response | Supported; no lattice descriptor |
| Uniform external electric field (ABI-v3 interaction) | CPU; CUDA reserved |
| Per-system molecular dipole moments | CPU; CUDA reserved |
| ASE and dpdata | Supported Python integrations |
| External interaction slot (solvation, field gradient, ...) | ABI-v3 slot reserved; not implemented |
| Browser adapter single points, SMILES-to-3D, and demo geometry optimization | Experimental wasm32 CPU site |
| Native GFN1-xTB, ROCm, solvation, optimization, MD, Hessians | Not implemented |

Documentation describes the current repository state. Reserved ABI values and
planned extensions are not reported as supported features.
