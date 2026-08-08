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
The high-level Python interface uses host NumPy arrays for both backends; direct
CUDA-device and mixed descriptors are available only through the low-level C
ABI.

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

## Array API and DLPack input arrays

`ArrayBatch` is the packed, zero-copy entry point: instead of building
`Structure` objects, pass the flat ragged-batch descriptor arrays directly.
Every array may come from any library that implements the Array API
`__dlpack__`/`__dlpack_device__` producer protocols — NumPy, CuPy, JAX eager
arrays, or PyTorch tensors — without gpuxtb importing any of those libraries.

```python
import numpy as np
from gpuxtb import ArrayBatch

atom_offsets = np.array([0, 2, 5], dtype=np.int64)
atomic_numbers = np.array([8, 1, 1, 1, 1], dtype=np.int32)
positions = np.array([
    [0.0000, 0.0000, -0.7358],
    [1.4418, 0.0000, 0.3679],
    [-1.4418, 0.0000, 0.3679],
    [-0.7, 0.0, 0.0],
    [0.7, 0.0, 0.0],
])
molecular_charges = np.array([0.0, 0.0])
unpaired_electrons = np.array([0, 0], dtype=np.int32)

with ArrayBatch(
    atom_offsets,
    atomic_numbers,
    positions,
    molecular_charges,
    unpaired_electrons,
    backend="cuda",
) as batch:
    result = batch.compute()
print(result.energies, result.forces, result.charges)
```

Host arrays become `GPUXTB_MEMORY_HOST` descriptors; CUDA device arrays (a
PyTorch tensor or CuPy array on `cuda`) become `GPUXTB_MEMORY_CUDA_DEVICE`
descriptors and are executed by the CUDA backend with no host round trip.
CUDA-managed memory, ROCm, and lazy/tracer objects (`jit`/`grad`/`vmap`
inputs, `torch.compile` graphs) are rejected with a precise error — pass a
concrete eager array instead.

- `copy=False` (default) requires the exact dtype, shape, and a compact
  C-contiguous layout; anything else raises rather than silently copying. Set
  `copy=True` to ask the producer for a compact copy. Copying never coerces
  dtype; descriptors must still match the C ABI's exact scalar types.
- The optional `point_charge_*` and `atomic_potential_shifts` /
  `charge_response_offsets` / `charge_response_matrix` groups mirror the
  `PointCharge`/`ChargeResponse` descriptors and must each be supplied
  all-or-nothing.
- `stream` selects the native `CUstream` for the context (the default `None`
  means the CUDA legacy default stream; DLPack producers receive stream value
  `1` in that case). `stream` is not meaningful for the CPU backend.

### Output policy

Results are ordinary host NumPy arrays by default, matching the rest of the
Python API. Pass an `out=` mapping to have gpuxtb write directly into your own
writable NumPy, CuPy, or PyTorch buffers (no copy); JAX arrays are never
mutated and are rejected as outputs.

```python
out_forces = torch.empty((5, 3), dtype=torch.float64, device="cuda")
result = batch.compute(out={"forces": out_forces})
assert result.forces is out_forces
```

`out=` keys: `energies`, `forces`, `charges` (alias `atomic_charges`),
`point_charge_forces`, `scc_iterations`, `scc_converged`, and
`per_system_status`.

`compute_arrays(...)` is a convenience alias that builds a temporary
`ArrayBatch` and computes in one call.

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
from gpuxtb.dpdata import GPUxtbDriver

system = dpdata.System("geometry.xyz", fmt="xyz")
labeled = system.predict(driver="gpuxtb", charge=0, multiplicity=1)
```

Geometries can be minimized with the batch-native minimizer, which relaxes
every frame in lockstep and evaluates energies and forces for all active
frames in one gpuxtb ragged-batch call per step:

```python
labeled = system.minimize(
    minimizer="gpuxtb",
    driver=GPUxtbDriver(backend="cpu"),
    fmax=5e-3,  # eV/Angstrom
    max_steps=1000,
)
```

Unlike the reference ``ase`` minimizer (one frame per optimizer step), a batch
of molecules is relaxed with full gpuxtb batch throughput; converged frames are
frozen and dropped from the batch as it shrinks. dpdata receives energies in eV
and forces in eV/Angstrom. Periodic systems are rejected because gpuxtb does
not expose a lattice/PBC descriptor.

## More documentation

- [Python user guide](https://github.com/njzjz/gpuxtb/blob/main/docs/user-guide/python.md)
- [Units and model semantics](https://github.com/njzjz/gpuxtb/blob/main/docs/user-guide/index.md#units-and-result-meaning)
- [QM/MM theory](https://github.com/njzjz/gpuxtb/blob/main/docs/theory/qmmm.md)
- [Source repository](https://github.com/njzjz/gpuxtb)
- [Issue tracker](https://github.com/njzjz/gpuxtb/issues)
- [License and third-party notices](https://github.com/njzjz/gpuxtb/blob/main/THIRD_PARTY_NOTICES.md)
