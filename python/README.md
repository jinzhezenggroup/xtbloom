# xTBloom for Python

[![PyPI version](https://img.shields.io/pypi/v/xtbloom.svg)](https://pypi.org/project/xtbloom/)

xTBloom provides batched GFN1/GFN2-xTB energies, analytic forces, and charges
through a NumPy-friendly interface backed by the same stable C ABI used by
native C and C++ applications.

GFN1-xTB and GFN2-xTB support CPU and CUDA through `Calculator`,
`BatchCalculator`, ASE, and dpdata. Both models support native ragged batches,
explicit point charges with force output, and caller-supplied periodic charge
response. The packed Array API/DLPack surface also supports both models;
PyTorch autograd remains explicitly GFN2-only.

## Installation

Install xTBloom from PyPI. Python 3.10 or newer is required:

```console
pip install xtbloom
```

Linux x86_64 and aarch64 wheels include the CUDA backend. Add the supported
CUDA 12 user-space libraries when the environment does not already provide
them:

```console
pip install "xtbloom[cuda12]"
```

Optional integrations can be combined with either backend. For example, add
ASE and dpdata to the CUDA environment with:

```console
pip install "xtbloom[cuda12,ase,dpdata]"
```

Published Linux, macOS, and Windows wheels include a private LP64 OpenBLAS
provider for CPU inference; `scipy-openblas32` is used only while building the
wheels and is not installed as a runtime dependency. CUDA execution additionally
needs a real NVIDIA GPU and compatible driver. The `cuda12` extra supplies the
supported `nvidia-*` user-space packages but cannot install the driver.

## Build from source

Use a source build only when developing xTBloom or when a published wheel does
not cover the target. From a complete source checkout, sync the locked,
non-editable package into uv's project environment:

```console
uv sync --locked --no-editable --no-default-groups --reinstall-package xtbloom
```

CUDA build selection defaults to `AUTO`: an available `nvcc` enables CUDA;
otherwise the source build is CPU-only. Add `--extra cuda12` when the supported
CUDA 12 host libraries are not supplied by the system. Run commands with
`uv run --no-sync` or activate `.venv` directly.

Ordinary source builds do not bundle OpenBLAS. They auto-discover a compatible
system monolithic LP64 LAPACKE+CBLAS runtime; if none is discoverable, add
`CMAKE_ARGS="-DXTBLOOM_CPU_LINALG_LIBRARY=/absolute/path/to/provider.so"` to the
sync command. Keep `--reinstall-package xtbloom` when changing this path or
explicitly overriding the `XTBLOOM_ENABLE_CUDA=AUTO` default, because uv's local
wheel cache does not key native builds by those environment variables.

A normal branch checkout must include complete Git tag history; an exact-tag
Python build is the documented shallow-checkout exception. Source builds need
C/C++ compilers with C11/C++17 support, and repository test configurations
require Python 3.11 or newer. CMake, GCC/Clang, NVCC/CUDA Toolkit, Ninja/uv,
BLAS, platform, driver, and wheel/source-build boundaries are listed in the
authoritative
[prerequisites matrix](https://github.com/jinzhezenggroup/xtbloom/blob/main/docs/user-guide/index.md#prerequisites).

Source-build and package-boundary details are in the
[developer guide](https://github.com/jinzhezenggroup/xtbloom/blob/main/docs/developer-guide/packaging.md).

## Single-point calculation

The high-level API uses atomic units: positions are in bohr, energies in
Hartree, forces in Hartree/bohr, and charges in elementary-charge units.
`electronic_temperature` is the exception: Python accepts kelvin.

```python
import numpy as np
from xtbloom import BatchCalculator, Calculator, Structure

numbers = np.array([8, 1, 1])
positions = np.array(
    [
        [0.0000000000, 0.0000000000, -0.7357858611],
        [1.4418315287, 0.0000000000, 0.3678929305],
        [-1.4418315287, 0.0000000000, 0.3678929305],
    ]
)

backend = "cuda"  # Use "cpu" to require CPU execution instead.
with Calculator("GFN2-xTB", numbers, positions, backend=backend) as calc:
    result = calc.singlepoint()

print(result["energy"])
print(result["forces"])
print(result["charges"])
```

`result["gradient"]` is the negative of `result["forces"]`. At finite
electronic temperature, the reported variational energy is the electronic
Helmholtz free energy.

`Calculator.hessian()` evaluates one dense numerical QM-coordinate energy
Hessian as central differences of analytic forces. `BatchCalculator.hessian()`
returns one matrix per structure and interleaves their displacement tasks in
native ragged force calls under one fixed thread/device budget:

```python
with Calculator("GFN2-xTB", numbers, positions, backend="cuda") as calc:
    hessian = calc.hessian(step=0.005, symmetrize=True)

structures = [Structure(numbers, positions), Structure(numbers, positions * 1.01)]
with BatchCalculator(structures, backend="cuda", cpu_threads=16) as calc:
    hessians = calc.hessian(step=0.005, symmetrize=True)
```

Each result is a NumPy `float64` array with shape `(3 * natoms, 3 * natoms)` and
units Hartree/bohr²; the batch method returns an input-ordered list for ragged
atom counts. By default, the methods automatically chunk the displaced
geometries; a positive `auto_batch_size` sets the same atom-count limit accepted
by `BatchCalculator.compute()`, while `False` or `None` submits all
displacements at once. The raw finite-difference matrices are returned by
default so antisymmetric numerical error remains visible, while
`symmetrize=True` applies `0.5 * (H + H.T)` to each matrix.

Only QM coordinates are displaced. Point-charge coordinates and values,
electric fields, and caller-supplied charge-response `b/A` operators remain
fixed, so no QM–point-charge or point-charge–point-charge blocks are included
and derivatives of `b/A` remain caller-owned. This explicit numerical method
does not change the narrower PyTorch autograd contract described below.

Set `backend="cpu"` or `backend="cuda"` to require one backend. The CUDA
quickstart above deliberately uses `"cuda"` so an unavailable GPU fails clearly
instead of running on CPU. `"auto"` prefers CUDA but falls back to CPU.
The same AUTO policy applies to GFN1-xTB and GFN2-xTB. A build without CUDA may
return `BACKEND_UNAVAILABLE` when creating an explicitly requested CUDA
context; a nonnegative `device_id` can be used with AUTO or CUDA.
Compatible calls can opt into electronic warm starts; the default is an
independent fresh SCC solve.

## Native ragged batches

`BatchCalculator` packs differently sized `Structure` objects into one native
request. Per-system SCC or eigensolver failures remain local: successful peers
are preserved, and failed floating-point slices contain NaNs plus diagnostics.

```python
import numpy as np
from xtbloom import BatchCalculator, Structure

structures = [
    Structure([1, 1], np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]])),
    Structure(
        [8, 1, 1],
        np.array(
            [
                [0.0000, 0.0000, -0.7358],
                [1.4418, 0.0000, 0.3679],
                [-1.4418, 0.0000, 0.3679],
            ]
        ),
    ),
]

with BatchCalculator(structures, backend="cuda") as calc:  # Use "cpu" for CPU-only builds.
    batch = calc.compute()

print(batch.energies)
print(batch[1].forces)
print(batch.failed_indices)
```

`compute(auto_batch_size=True)` can split very large workloads into
conservative CUDA chunks while preserving input order.

## Advanced array and CUDA paths

`ArrayBatch` accepts `method="GFN1-xTB"`/`"GFN1"` and
`method="GFN2-xTB"`/`"GFN2"`, with GFN2-xTB retained as the default. It accepts
packed ragged descriptors from eager NumPy, CuPy, JAX, or PyTorch arrays through
`__dlpack__` and `__dlpack_device__`. Host arrays map to host descriptors; CUDA
arrays can remain device-resident. By default, results return as host NumPy
arrays.

Use an `out=` mapping for caller-owned NumPy, CuPy, or PyTorch output buffers,
or `result_memory="cuda"` for one xTBloom-owned packed device arena exported as
DLPack producers. Exact dtype, shape, layout, lifetime, stream, and ownership
rules are documented in the
[Python API guide](https://github.com/jinzhezenggroup/xtbloom/blob/main/docs/user-guide/python.md#array-api-and-dlpack-input-arrays).

`xtbloom_torch` is currently GFN2-xTB-only. `xtbloom_torch(positions, atomic_numbers, atom_offsets, molecular_charges,
unpaired_electrons, ...)` runs xTBloom inference on PyTorch tensors (host or
CUDA) and is the only autograd entry point in the Python API. It supports
exactly the positions gradient `dE/dR = -F`; autograd on any other input, or a
gradient flowing through the `forces` output (the Hessian), raises
`XTBloomNotSupportedError`. Higher-order differentiation is likewise rejected
explicitly rather than returning a partial or zero Hessian. The native data
plane is a compiled extension written against the LibTorch Stable ABI
(torch >= 2.10), so a single binary works across torch releases; its stable
headers are vendored in `cmake/3rdparty/torch-stable` and it links a
build-time-only stub, so building xTBloom never downloads or requires torch
(torch is still required at runtime to call `xtbloom_torch`). PyTorch is
imported only when the op is called. CPU execution is synchronous; CUDA follows
`torch.cuda.current_stream()` and returns the ordinary `(energies, forces)`
pair. See
`docs/user-guide/python.md` for the full contract.

## Charge, spin, and embedding

Use either `multiplicity` or `uhf = multiplicity - 1` for open-shell
calculations. Open-shell Python calculations default to two unrestricted spin
channels; `spin_channels=1` requests the restricted open-shell form.

`PointCharge` inputs participate in every SCC iteration, and xTBloom can
return forces on both QM atoms and point charges. `ChargeResponse(shifts=b,
matrix=A)` supplies a caller-owned `b + A q` operator on the atomic-charge
channel. Returned forces hold those external fields fixed; callers own their
coordinate derivatives and classical MM-MM terms.

See the
[QM/MM guide](https://github.com/jinzhezenggroup/xtbloom/blob/main/docs/user-guide/qmmm.md)
for the complete contract.

## ASE and dpdata

ASE exposes xTBloom through its usual eV and angstrom conventions:

```python
from ase.build import molecule
from xtbloom.ase import XTBloom

atoms = molecule("H2O")
atoms.calc = XTBloom(method="GFN2-xTB")
energy_ev = atoms.get_potential_energy()
forces_ev_per_angstrom = atoms.get_forces()
```

dpdata can label systems through the xTBloom driver:

```python
import dpdata

system = dpdata.System("geometry.xyz", fmt="xyz")
labeled = system.predict(driver="xtbloom", charge=0, multiplicity=1)
```

The dpdata integration also provides a batch-native minimizer built from
repeated xTBloom single-point calls. This is a higher-level adapter, not native
geometry optimization in the C ABI.

## Scope

GFN1 electric fields/dipoles, ROCm, lattice/PBC inputs, solvation, native
geometry optimization, molecular dynamics, native/analytic
Hessians, vibrational analysis, and
higher-order autograd are not implemented. A numerical QM Cartesian Hessian is
available through Python `Calculator.hessian()` and `BatchCalculator.hessian()`.
The high-level `Calculator` and `BatchCalculator` APIs use host NumPy arrays;
direct device and mixed descriptors are exposed through the model-aware
`ArrayBatch` surface and the low-level C ABI. PyTorch autograd remains GFN2-only.

## More documentation

- [PyPI project](https://pypi.org/project/xtbloom/)
- [Documentation home](https://github.com/jinzhezenggroup/xtbloom/blob/main/docs/index.md)
- [Python API guide](https://github.com/jinzhezenggroup/xtbloom/blob/main/docs/user-guide/python.md)
- [Units and result meaning](https://github.com/jinzhezenggroup/xtbloom/blob/main/docs/user-guide/index.md#units-and-result-meaning)
- [QM/MM theory](https://github.com/jinzhezenggroup/xtbloom/blob/main/docs/theory/qmmm.md)
- [Browser demo](https://xtbloom.jinzhezeng.group)
- [Source repository](https://github.com/jinzhezenggroup/xtbloom)
- [Issue tracker](https://github.com/jinzhezenggroup/xtbloom/issues)
- [License and notices](https://github.com/jinzhezenggroup/xtbloom/blob/main/THIRD_PARTY_NOTICES.md)
