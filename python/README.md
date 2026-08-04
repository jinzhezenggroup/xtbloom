# gpuxtb Python package

A Python binding around the gpuxtb public C ABI. The package builds the C++
library through CMake with **scikit-build-core** and wraps it with `ctypes`, so
there is no separate Python extension to compile. It follows the API shape of
`tblite`/`xtb-python` and additionally exposes the native ragged-batch model of
the C API.

## Install

```console
pip install .
```

This builds `libgpuxtb` from the repository CMake project and bundles it inside
the wheel under `gpuxtb/lib`. Requires Python >= 3.10, a C++17 compiler, a
working BLAS/MKL runtime (the CPU eigensolver dlopens `libmkl_rt`), and `numpy`.

Optional extras:

```console
pip install ".[ase]"        # ASE calculator
pip install ".[dpdata]"     # dpdata driver plugin
pip install ".[cuda]"       # cuda-python for future device-memory work
pip install ".[test]"       # pytest suite dependencies
```

### CUDA wheels

`GPUXTB_ENABLE_CUDA` defaults to `AUTO`: the CUDA backend is compiled in
whenever a CUDA compiler is present at build time, and a CPU-only wheel is
produced otherwise. The library can be built entirely from PyPI-distributed
CUDA packages (no preinstalled system CUDA toolkit needed):

```console
pip install cuda-toolkit nvidia-cublas-cu12 nvidia-cusolver-cu12 nvidia-cuda-runtime-cu12
GPUXTB_ENABLE_CUDA=ON pip install .
```

If the resulting wheel is CUDA-enabled it does **not** bundle the CUDA runtime
libraries; at runtime it needs the system CUDA driver plus the PyPI CUDA
packages above.

CI builds wheels with cibuildwheel (`.github/workflows/wheels.yml`):

* **Linux** wheels build in the PyPA CUDA manylinux images
  (`quay.io/manylinux_cuda/manylinux_2_28_*_cuda12_9`), so the CUDA backend is
  compiled in by default (cuSOLVER, which those images do not ship, is pulled
  from PyPI into the container before the build). The full test suite runs
  against the built wheel on a host with an MKL runtime and the PyPI
  `nvidia-*` runtime packages.
* **macOS** wheels build CPU-only (no CUDA toolkit); they are covered by the
  cibuildwheel import smoke test. Note their CPU inference currently needs an
  MKL runtime exposing the `libmkl_rt.so` names the C++ eigensolver dlopens,
  which the bundled library does not provide on macOS.
* **Windows** wheels are a follow-up (the CPU eigensolver uses `dlopen`).

The public Python interface always uses host buffers on both backends; CUDA
device-resident memory is a future extension. The current CUDA backend is
restricted closed-shell only (see the C API documentation).

## Usage

Atomic units throughout: positions in bohr, energies in Hartree, forces in
Hartree/bohr, charges in elementary-charge units.

### Single molecule (tblite-like)

```python
import numpy as np
from gpuxtb import Calculator

calc = Calculator(
    "GFN2-xTB",
    numbers=np.array([8, 1, 1]),
    positions=np.array([
        [+0.00000000000000, +0.00000000000000, -0.73578586109551],
        [+1.44183152868459, +0.00000000000000, +0.36789293054775],
        [-1.44183152868459, +0.00000000000000, +0.36789293054775],
    ]),
)
result = calc.singlepoint()
print(result["energy"], result["forces"], result["charges"])
```

Charge and spin multiplicity are supported directly:

```python
calc = Calculator("GFN2-xTB", numbers, positions, charge=-1, multiplicity=2)
```

* `charge` maps to the C ABI `molecular_charges`.
* `multiplicity` (or `uhf = multiplicity - 1`) maps to `unpaired_electrons`.
* `spin_channels` selects restricted (1) or unrestricted (2) orbitals; the
  CPU backend defaults to unrestricted for open-shell systems.

### Batched (native)

```python
from gpuxtb import Structure, BatchCalculator

structures = [Structure(numbers, positions, charge=0.0) for positions in many_positions]
result = BatchCalculator(structures).compute()
print(result.energies)   # per-system
print(result[0].forces)  # per-system via Result
```

### ASE

```python
from ase.build import molecule
from gpuxtb.ase import GPUxtb

atoms = molecule("H2O")
atoms.calc = GPUxtb(method="GFN2-xTB")
atoms.get_potential_energy()  # eV
atoms.get_forces()            # eV/Angstrom
atoms.get_charges()           # e
```

### dpdata

```python
import dpdata

system = dpdata.System("some_geometry.xyz", fmt="xyz")
labeled = system.predict(driver="gpuxtb")          # whole system in one batch
# or with explicit charge / multiplicity:
labeled = system.predict(driver="gpuxtb", charge=0, multiplicity=2)
```

Importing the package registers the driver under the key `"gpuxtb"` through
the `dpdata.plugins` entry point. Energies come back in eV and forces in
eV/Angstrom, matching dpdata conventions. Periodic systems are not supported
(the public C ABI has no lattice input).

## Tests

```console
pip install ".[test]"
python -m pytest python/tests
```

The suite validates the bindings against the committed conformance goldens in
`data/conformance` (neutral, charged, open-shell, and QM/MM point-charge cases).
On a machine without MKL the tests should be run with the MKL runtime on
`LD_LIBRARY_PATH`, e.g. `LD_LIBRARY_PATH=/path/to/mkl python -m pytest python/tests`.