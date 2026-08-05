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
working LP64 LAPACKE+CBLAS runtime, and `numpy`. Linux installs use the
BSD-licensed `scipy-openblas32` runtime by default.
If `GPUXTB_LIBRARY` overrides the bundled native library, keep it version-matched
with the Python package; older libraries may not implement newer optional ABI
suffixes such as unrestricted `spin_channels`.

Optional extras:

```console
pip install ".[ase]"        # ASE calculator
pip install ".[dpdata]"     # dpdata driver plugin
pip install ".[cuda12]"     # CUDA 12 host runtime/math providers
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
it needs the system CUDA driver plus compatible system libraries or the PyPI
CUDA packages. Install ``gpuxtb[cuda12]`` to obtain those optional
``nvidia-*`` providers. An ordinary ``pip install gpuxtb`` remains a complete
CPU installation without the proprietary CUDA stack.
The published CUDA 12.9 wheels contain SASS for sm_80, sm_89, sm_90, and
sm_120; source builds can override this with ``GPUXTB_CUDA_ARCHITECTURES``.

### Runtime libraries are loaded automatically

You never need to set ``LD_LIBRARY_PATH``. On import,
:func:`gpuxtb.library` locates and preloads the runtime libraries the native
library depends on:

* the **BLAS** runtime for the CPU eigensolver — the LP64 OpenBLAS wheel
  `scipy-openblas32` on linux x86_64 and aarch64. On other platforms,
  compatible system OpenBLAS or MKL runtimes are discovered when available;
  and
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
  cibuildwheel test (deps from the package ``[test,cuda12]`` extras through
  ``test-extras``).
* **macOS and Windows** wheels are not published yet. Their CPU eigensolver
  runtime is not functional, and gpuxtb does not ship import-only artifacts
  that cannot perform inference.

The public Python interface always uses host buffers on both backends; CUDA
device-resident memory is a future extension. CUDA supports the same restricted
and unrestricted GFN2 spin descriptors as the CPU backend.

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
* `spin_channels` selects restricted (1) or unrestricted (2) orbitals. The
  high-level Python interface defaults open-shell systems to unrestricted and
  explicitly submits that choice to either backend; a missing C ABI suffix
  itself remains the restricted compatibility default.

### Batched (native)

```python
from gpuxtb import Structure, BatchCalculator

structures = [Structure(numbers, positions, charge=0.0) for positions in many_positions]
result = BatchCalculator(structures).compute()
print(result.energies)   # per-system
print(result[0].forces)  # per-system via Result
```

Large ragged batches can be split into several synchronous C calls while
preserving system order and peer-local diagnostics:

```python
# Query current CUDA free memory and choose a conservative grouping target.
result = BatchCalculator(structures, backend="cuda").compute(auto_batch_size=True)

# Or set an explicit target maximum total atom count per call.
result = BatchCalculator(structures).compute(auto_batch_size=20_000)
```

The integer is a grouping target rather than a promise that an individual
system can be subdivided: a system larger than the target is attempted alone.
Automatic CUDA sizing re-queries current free memory for each call, retains a
fixed reserve, and retries native allocation failures by splitting only
multi-system chunks. Other native failures, and allocation failure for one
indivisible system, are returned unchanged. When CUDA memory cannot be queried,
a conservative fixed target is used. ``None`` (the default) or ``False`` keeps
the historical single-call behavior.

Batch failures are peer-local: failed slices contain NaNs and are listed by
``result.failed_indices``, while successful peer results remain accessible.
Call ``result.raise_for_status()`` (or ``compute(raise_on_failure=True)``) when
strict all-or-nothing exception behavior is desired.

### Periodic charge response (b + A q)

`ChargeResponse` exposes the periodic QM/MM coupling of the C ABI: per-atom SCC
potential shifts ``b`` and a symmetric charge-response matrix ``A``, giving the
shift ``b + A q`` and the variational energy ``q^T b + 0.5 q^T A q`` on the
atomic-charge channel. Derivatives of ``b``/``A`` with respect to coordinates
are outside gpuxtb and are not included in forces.

```python
import numpy as np
from gpuxtb import Calculator, ChargeResponse

response = ChargeResponse(
    shifts=np.array([0.003, -0.002]),
    matrix=np.array([[0.02, 0.001], [0.001, 0.018]]),
)
calc = Calculator("GFN2-xTB", numbers=[1, 1], positions=positions, charge_response=response)
result = calc.singlepoint()
```

The same object is accepted by ``Structure`` inside ``BatchCalculator``; systems
without a charge response are treated with zero ``b``/``A``.

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
On Linux, installing the package supplies `scipy-openblas32`; no loader-path
setup is required. Native builds may instead select an absolute compatible
runtime with `GPUXTB_CPU_LINALG_LIBRARY`.
