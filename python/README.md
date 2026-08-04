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
pip install ".[test]"       # pytest suite dependencies
```

### CUDA wheels

`GPUXTB_ENABLE_CUDA` defaults to `AUTO`: the CUDA backend is compiled in when a
CUDA toolkit with `nvcc` is present at build time, and a CPU-only wheel is
produced otherwise. To force a CUDA build, point CMake at the toolkit explicitly
when it is not already on `PATH`:

```console
PATH=/path/to/cuda/bin:$PATH \
CUDACXX=/path/to/cuda/bin/nvcc \
GPUXTB_ENABLE_CUDA=ON pip install .
```

A CUDA-enabled wheel does **not** bundle the CUDA runtime libraries; at runtime
it needs the system CUDA driver plus the PyPI CUDA packages. Published Linux
wheels declare those ``nvidia-*`` packages as normal dependencies, so an
ordinary ``pip install gpuxtb`` installs a loadable runtime automatically.
The published CUDA 12.9 wheels contain SASS for sm_80, sm_89, sm_90, and
sm_120; source builds can override this with ``GPUXTB_CUDA_ARCHITECTURES``.

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
* **macOS and Windows** wheels are not published yet. Their CPU eigensolver
  runtime is not functional, and gpuxtb does not ship import-only artifacts
  that cannot perform inference.

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
  CPU backend defaults to unrestricted for open-shell systems. While the public
  CUDA path remains restricted-only, ``backend="auto"`` routes open-shell or
  unrestricted structures to CPU instead of failing only on GPU-equipped hosts.

### Batched (native)

```python
from gpuxtb import Structure, BatchCalculator

structures = [Structure(numbers, positions, charge=0.0) for positions in many_positions]
result = BatchCalculator(structures).compute()
print(result.energies)   # per-system
print(result[0].forces)  # per-system via Result
```

Batch failures are peer-local: failed slices contain NaNs and are listed by
``result.failed_indices``, while successful peer results remain accessible.
Call ``result.raise_for_status()`` (or ``compute(raise_on_failure=True)``) when
strict all-or-nothing exception behavior is desired.

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
