# xTBloom for Python

xTBloom provides batched GFN2-xTB energies, analytic forces, and atomic charges
through a NumPy-friendly interface backed by the same stable C ABI used by
native C and C++ applications.

It supports restricted and unrestricted GFN2-xTB, native ragged batches,
explicit point charges with force output, caller-supplied periodic charge
response, CPU and CUDA backends, ASE, dpdata, and eager Array API/DLPack arrays.

## Installation

xTBloom is not yet published on PyPI. From a source checkout, build and install
the CPU package with:

```console
XTBLOOM_ENABLE_CUDA=OFF python -m pip install .
```

Install optional integrations from the checkout:

```console
XTBLOOM_ENABLE_CUDA=OFF python -m pip install ".[ase,dpdata]"
XTBLOOM_ENABLE_CUDA=ON python -m pip install ".[cuda12]"
```

Python 3.10 or newer is required. Linux wheels include a private LP64 OpenBLAS
provider for CPU inference; `scipy-openblas32` is used only while building the
wheel and is not installed as a runtime dependency. A CUDA-enabled wheel
additionally needs an NVIDIA driver and compatible CUDA 12 host libraries; the
`cuda12` extra supplies the supported `nvidia-*` packages. CUDA libraries are
not bundled inside the xTBloom wheel.

Source-build and package-boundary details are in the
[developer guide](https://github.com/jinzhezenggroup/xtbloom/blob/main/docs/developer-guide/packaging.md).

## Single-point calculation

The high-level API uses atomic units: positions are in bohr, energies in
Hartree, forces in Hartree/bohr, and charges in elementary-charge units.
`electronic_temperature` is the exception: Python accepts kelvin.

```python
import numpy as np
from xtbloom import Calculator

numbers = np.array([8, 1, 1])
positions = np.array(
    [
        [0.0000000000, 0.0000000000, -0.7357858611],
        [1.4418315287, 0.0000000000, 0.3678929305],
        [-1.4418315287, 0.0000000000, 0.3678929305],
    ]
)

with Calculator("GFN2-xTB", numbers, positions, backend="auto") as calc:
    result = calc.singlepoint()

print(result["energy"])
print(result["forces"])
print(result["charges"])
```

`result["gradient"]` is the negative of `result["forces"]`. At finite
electronic temperature, the reported variational energy is the electronic
Helmholtz free energy.

Set `backend="cpu"` or `backend="cuda"` to require one backend. `"auto"`
prefers an available CUDA backend and otherwise selects CPU. Compatible calls
can opt into electronic warm starts; the default is an independent fresh SCC
solve.

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

with BatchCalculator(structures, backend="auto") as calc:
    batch = calc.compute()

print(batch.energies)
print(batch[1].forces)
print(batch.failed_indices)
```

`compute(auto_batch_size=True)` can split very large workloads into
conservative CUDA chunks while preserving input order.

## Advanced array and CUDA paths

`ArrayBatch` accepts packed ragged descriptors from eager NumPy, CuPy, JAX, or
PyTorch arrays through `__dlpack__` and `__dlpack_device__`. Host arrays map
to host descriptors; CUDA arrays can remain device-resident. By default,
results return as host NumPy arrays.

Use an `out=` mapping for caller-owned NumPy, CuPy, or PyTorch output buffers,
or `result_memory="cuda"` for one xTBloom-owned packed device arena exported as
DLPack producers. Exact dtype, shape, layout, lifetime, stream, and ownership
rules are documented in the
[Python API guide](https://github.com/jinzhezenggroup/xtbloom/blob/main/docs/user-guide/python.md#array-api-and-dlpack-input-arrays).

`xtbloom_torch(...)` is the optional PyTorch autograd entry point. It supports
the positions gradient `dE/dR = -F`. Gradients with respect to other inputs,
force-output differentiation, Hessians, and higher-order differentiation are
rejected explicitly.

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

GFN1-xTB, ROCm, lattice/PBC inputs, solvation, native geometry optimization,
molecular dynamics, Hessians, and higher-order autograd are not implemented.
The high-level `Calculator` and `BatchCalculator` APIs use host NumPy arrays;
direct device and mixed descriptors are exposed through `ArrayBatch` and the
low-level C ABI.

## More documentation

- [Documentation home](https://github.com/jinzhezenggroup/xtbloom/blob/main/docs/index.md)
- [Python API guide](https://github.com/jinzhezenggroup/xtbloom/blob/main/docs/user-guide/python.md)
- [Units and result meaning](https://github.com/jinzhezenggroup/xtbloom/blob/main/docs/user-guide/index.md#units-and-result-meaning)
- [QM/MM theory](https://github.com/jinzhezenggroup/xtbloom/blob/main/docs/theory/qmmm.md)
- [Browser demo](https://xtbloom.jinzhezeng.group)
- [Source repository](https://github.com/jinzhezenggroup/xtbloom)
- [Issue tracker](https://github.com/jinzhezenggroup/xtbloom/issues)
- [License and notices](https://github.com/jinzhezenggroup/xtbloom/blob/main/THIRD_PARTY_NOTICES.md)
