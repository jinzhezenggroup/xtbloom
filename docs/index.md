# xTBloom documentation

xTBloom provides native, batched GFN1/GFN2-xTB energies, analytic forces, and
charges through one stable C ABI and Python interfaces built on that ABI.

[Try the browser demo](https://xtbloom.jinzhezeng.group) ·
[Install xTBloom](user-guide/index.md#installation) ·
[Python API](user-guide/python.md) ·
[C/C++ API](user-guide/c-api.md)

## See it run

[![xTBloom browser demo running an ethanol calculation](assets/web-demo-ethanol.png)](https://xtbloom.jinzhezeng.group/?smiles=CCO)

The browser demo runs entirely on the client: enter SMILES or XYZ coordinates,
inspect the 3D structure, and calculate GFN1-xTB or GFN2-xTB results without
uploading the molecule. GFN2-xTB remains the default. Its SMILES-to-3D and
geometry-optimization workflow belongs to the demo adapter, not the native
single-point API.

[Open the demo](https://xtbloom.jinzhezeng.group) ·
[Browser usage and limitations](user-guide/browser-demo.md)

## Start here

- **Using Python:** start with the
  [Python installation guide](user-guide/python.md#installation), then continue
  there for single systems, native ragged batches, spin, point charges, Array
  API/DLPack, ASE, dpdata, and
  [ASE molecular dynamics](user-guide/ase-md.md).
- **Analyzing vibrations:** use the Python
  [vibrational analysis guide](user-guide/vibrations.md) for numerical
  Hessians, rigid-mode projection, frequencies, and normal modes.
- **Using C or C++:** [C ABI guide](user-guide/c-api.md) for installation, a
  complete example, descriptor ownership, CUDA memory, and error handling.
- **Embedding QM/MM:** [QM/MM guide](user-guide/qmmm.md) for explicit point
  charges and caller-supplied charge response.
- **Working with AI agents:** [user skills](user-guide/agent-skills.md) for
  installation, Python, ASE/dpdata, zero-copy ML, C/C++, and QM/MM workflows.
- **Evaluating performance:** [performance summary](user-guide/performance.md)
  for published results and [benchmark harnesses](../benchmarks/README.md) for
  reproducible methodology.

## Understand the model

The [theory guide](theory/index.md) explains the numerical meaning of public
results and external interactions:

- [GFN2-xTB model and SCC](theory/gfn2.md)
- [GFN1-xTB model and publication contract](theory/gfn1.md)
- [Explicit point charges and periodic response](theory/qmmm.md)

## Develop xTBloom

The [developer guide](developer-guide/index.md) records implementation
contracts and validation requirements:

- [Architecture](developer-guide/architecture.md)
- [Validation and scientific evidence](developer-guide/validation.md)
- [Packaging, dependencies, and licensing](developer-guide/packaging.md)
- [Web demo implementation](../web/README.md)

Repository contributors and coding agents must also follow
[`AGENTS.md`](../AGENTS.md). The installed public contract is
[`include/xtbloom/xtbloom.h`](../include/xtbloom/xtbloom.h).

## Capability boundary

xTBloom implements restricted and unrestricted GFN1-xTB and GFN2-xTB on CPU
and CUDA. Both models publish native ragged batches, analytic forces,
charges, explicit point charges, caller-supplied periodic charge response, the
high-level Python calculators, Array API/DLPack, ASE, and dpdata. Uniform
electric fields, molecular dipoles, and PyTorch autograd are GFN2-only. The
single-threaded CPU/WebAssembly browser demo exposes both GFN1 and GFN2, with
GFN2 selected by default. The low-level CUDA ABI accepts host, device, and
mixed descriptors, including independently placed interaction descriptor and
payload buffers.

The ABI-v4 native-cell descriptors validate molecular `NONE` and fully
periodic `XYZ` inputs, but valid `XYZ` compute requests return
`NOT_IMPLEMENTED` transactionally. GFN1 field/dipole properties, ROCm, native
drivers for geometry optimization and molecular dynamics, solvation,
native/analytic Hessians, and periodic GFN1/GFN2 execution are not implemented.
ASE-driven molecular dynamics, the Python numerical Hessian and vibrational
analysis, and the browser/dpdata optimizers are higher-level adapters built on
repeated native calculations.
