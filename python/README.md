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
pip install ".[cuda]"       # CUDA runtime libs (nvidia-*) for CUDA-enabled wheels
pip install ".[test]"       # pytest suite dependencies
```

### CUDA wheels

`GPUXTB_ENABLE_CUDA` defaults to `AUTO`: the CUDA backend is compiled in
whenever a CUDA compiler is present at build time, and a CPU-only wheel is
produced otherwise. The library can be built entirely from PyPI-distributed
CUDA packages (no preinstalled system CUDA toolkit needed):

```console
pip install ".[cuda]"       # nvidia-* runtime libraries
GPUXTB_ENABLE_CUDA=ON pip install .
```

A CUDA-enabled wheel does **not** bundle the CUDA runtime libraries; at runtime
it needs the system CUDA driver plus the PyPI CUDA packages, which the
``[cuda]`` extra installs: ``pip install "gpuxtb[cuda]"``.

### Runtime libraries are loaded automatically

You never need to set ``LD_LIBRARY_PATH``. On import,
:func:`gpuxtb.library` locates and preloads the runtime libraries the native
library depends on:

* the **MKL** runtime ``libmkl_rt`` (dlopen'ed by the CPU eigensolver) — it is
  pulled in automatically on Linux x86_64 via a platform-tagged dependency, and
  discovered from any installed ``mkl`` package / system path elsewhere; and
* the **CUDA** runtime libraries (cuBLAS, cuSOLVER, ...) — resolved from the
  installed ``nvidia-*`` PyPI packages or a CUDA toolkit.

Only the CUDA *driver* interface (``libcuda``) is never shipped or preloaded by
the package; it always comes from the system NVIDIA kernel driver.

CI builds wheels with cibuildwheel (`.github/workflows/wheels.yml`) and tests
them through cibuildwheel's own test feature:

* **Linux x86_64 and aarch64** wheels build in the PyPA CUDA manylinux images
  (`quay.io/manylinux_cuda/manylinux_2_28_*_cuda12_9`) with the CUDA backend
  compiled in (aarch64 natively on a GitHub ARM runner, no QEMU). cuSOLVER,
  which those images do not ship, is pulled from PyPI into the container before
  the build. x86_64 additionally runs the full conformance suite as the
  cibuildwheel test (deps from the package ``[test]`` extra through
  ``test-extras``, MKL included).
* **macOS** (x86_64 and arm64) wheels build CPU-only and get an import smoke
  test.
* **Windows** wheels build CPU-only; the C++ CPU eigensolver's MKL runtime
  loader is not ported to Windows yet, so they also get an import smoke test.

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