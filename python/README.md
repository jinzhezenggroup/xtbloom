# gpuxtb for Python

gpuxtb provides batched GFN2-xTB energies, analytic forces, and atomic charges
through a NumPy-friendly Python interface. The package calls the same native C
ABI used by C and C++ applications and can select a CPU or CUDA backend without
changing the calculation API.

Current support includes restricted and unrestricted GFN2-xTB, ragged batches,
explicit point charges with force output, periodic caller-supplied charge
response, ASE, and dpdata. GFN1-xTB, ROCm, lattice/PBC inputs, solvation,
optimization, and Hessians are not implemented.

## Installation

gpuxtb is being prepared for its first PyPI release. For a published release,
install the CPU package with:

```console
python -m pip install gpuxtb
```

Optional extras install integrations or CUDA 12 host libraries:

```console
python -m pip install "gpuxtb[ase]"
python -m pip install "gpuxtb[dpdata]"
python -m pip install "gpuxtb[cuda12]"
```

Python 3.10 or newer is required. Linux wheels use the separately installed
`scipy-openblas32` package as the LP64 LAPACKE+CBLAS runtime for CPU inference.
A CUDA-enabled wheel additionally needs an NVIDIA driver and compatible CUDA
12 host libraries; the `cuda12` extra supplies the supported `nvidia-*`
packages. CUDA libraries are not bundled inside the gpuxtb wheel.

The planned PyPI artifacts are Linux x86_64 and aarch64 wheels. macOS and
Windows wheels are not supported yet. Source-build instructions are kept in the
[developer guide](https://github.com/njzjz/gpuxtb/blob/main/docs/developer-guide/packaging.md).

## Single-point calculation

The high-level API uses atomic units: positions are in bohr, energies in
Hartree, forces in Hartree/bohr, and charges in elementary-charge units.
`electronic_temperature` is the one exception: Python accepts kelvin and
converts it to the C ABI's `k_B T` energy scale.

```python
import numpy as np
from gpuxtb import Calculator

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
electronic temperature, the reported energy is the variational electronic
Helmholtz free energy used by xTB and tblite.

Set `backend="cpu"` or `backend="cuda"` to require a backend. `"auto"`
prefers an available CUDA backend and otherwise selects CPU. `cpu_threads`
controls molecule-level CPU parallelism and defaults to one in the Python API.
The high-level Python interface uses host NumPy arrays by default; through the
`memory_space` option below the same calculators can place their C-ABI
descriptors in CUDA device memory while still returning host arrays.

## Device-resident CUDA memory

The public C ABI accepts CUDA-device buffers, and the Python interface can
place its input/output descriptors there through the CUDA runtime (libcudart)
instead of staging them through the host. Select a placement per calculator:

```python
import numpy as np
from gpuxtb import Calculator

calc = Calculator(
    "GFN2-xTB",
    numbers=np.array([8, 1, 1]),
    positions=np.array(
        [
            [0.0000000000, 0.0000000000, -0.7357858611],
            [1.4418315287, 0.0000000000, 0.3678929305],
            [-1.4418315287, 0.0000000000, 0.3678929305],
        ]
    ),
    backend="cuda",
    memory_space="device",  # or "mixed"
)
result = calc.singlepoint()  # energy/forces/charges are ordinary numpy arrays
```

- `memory_space="host"` (default) keeps every descriptor in CPU memory.
- `memory_space="device"` places every descriptor in CUDA device memory.
- `memory_space="mixed"` keeps small scalar/offset descriptors on the host
  while large numerical inputs (`positions`, charges, point charges,
  charge-response matrix) and outputs (`forces`, `point_charge_forces`,
  `scc_converged`) stay on the device.

Device modes require a CUDA backend context and a loadable CUDA runtime
(libcudart, provided by the `cuda12` extra or a compatible system toolkit);
the same placement is available on `BatchCalculator` and works with
`auto_batch_size` slicing, point charges, and charge-response descriptors.
Device buffers are uploaded before the call and downloaded back into host
numpy arrays after it, so the returned `Result`/`BatchResult` objects are
identical in shape and semantics to host-mode results.

## Charge and spin

Use either `multiplicity` or `uhf = multiplicity - 1`. Open-shell Python
calculations default to two unrestricted spin channels; `spin_channels=1`
requests the restricted open-shell form explicitly.

```python
with Calculator(
    "GFN2-xTB",
    numbers=[7, 1, 1],
    positions=np.array(
        [
            [0.0, 0.0, 0.0],
            [1.8, 0.0, 0.0],
            [-0.6, 1.7, 0.0],
        ]
    ),
    charge=0,
    multiplicity=2,
) as calc:
    radical = calc.singlepoint()
```

## Native ragged batches

`BatchCalculator` packs differently sized structures into one native request.
Per-system SCC or eigensolver failures remain local: successful peers are
preserved, and failed floating-point slices contain NaNs.

```python
import numpy as np
from gpuxtb import BatchCalculator, Structure

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
batch.raise_for_status()
```

For large workloads, `compute(auto_batch_size=True)` chooses conservative CUDA
chunks from current free memory. An integer such as
`compute(auto_batch_size=20_000)` instead limits the target total atom count per
native call while preserving input order.

## Explicit point charges

Point charges participate in every SCC iteration. Their positions are in bohr,
charges in elementary-charge units, and positive screening parameters
`gammas` in Hartree. gpuxtb returns analytic forces on both QM atoms and point
charges, but does not calculate point-charge/point-charge interactions.

```python
from gpuxtb import Calculator, PointCharge

embedding = PointCharge(
    positions=np.array([[4.0, 0.0, 0.0]]),
    charges=np.array([0.5]),
    gammas=np.array([0.405771]),
)

with Calculator(
    "GFN2-xTB",
    numbers=[1, 1],
    positions=np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]]),
    point_charges=embedding,
) as calc:
    embedded = calc.singlepoint()

print(embedded.point_charge_forces)
```

`ChargeResponse(shifts=b, matrix=A)` additionally supplies a periodic
`b + A q` operator on the atomic-charge channel. gpuxtb treats `b` and `A` as
caller-owned fixed fields; returned forces exclude their coordinate
derivatives. The caller must add those derivatives and classical MM-MM terms.

## ASE

ASE converts gpuxtb's atomic units to its usual eV and Angstrom conventions.

```python
from ase.build import molecule
from gpuxtb.ase import GPUxtb

atoms = molecule("H2O")
atoms.calc = GPUxtb(method="GFN2-xTB")
energy_ev = atoms.get_potential_energy()
forces_ev_per_angstrom = atoms.get_forces()
charges_e = atoms.get_charges()
```

## dpdata

```python
import dpdata

system = dpdata.System("geometry.xyz", fmt="xyz")
labeled = system.predict(driver="gpuxtb", charge=0, multiplicity=1)
```

dpdata receives energies in eV and forces in eV/Angstrom. Periodic systems are
rejected because gpuxtb does not expose a lattice/PBC descriptor.

## More documentation

- [Python user guide](https://github.com/njzjz/gpuxtb/blob/main/docs/user-guide/python.md)
- [Units and model semantics](https://github.com/njzjz/gpuxtb/blob/main/docs/user-guide/index.md#units-and-result-meaning)
- [QM/MM theory](https://github.com/njzjz/gpuxtb/blob/main/docs/theory/qmmm.md)
- [Source repository](https://github.com/njzjz/gpuxtb)
- [Issue tracker](https://github.com/njzjz/gpuxtb/issues)
- [License and third-party notices](https://github.com/njzjz/gpuxtb/blob/main/THIRD_PARTY_NOTICES.md)
