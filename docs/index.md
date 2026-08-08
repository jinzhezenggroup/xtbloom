# gpuxtb documentation

gpuxtb exposes GFN2-xTB single-point inference through one stable C ABI and
Python interfaces built on that ABI. Start with the guide for your role.

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
| ASE and dpdata | Supported Python integrations |
| External interaction slot (electric field, solvation, field gradient, ...) | ABI-v3 slot reserved; not implemented |
| GFN1-xTB, ROCm, solvation, optimization, MD, Hessians | Not implemented |

Documentation describes the current repository state. Reserved ABI values and
planned extensions are not reported as supported features.
