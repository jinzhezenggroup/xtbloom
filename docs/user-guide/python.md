# Python API

The Python package wraps the public gpuxtb C ABI with `ctypes`. There is no
separate CPython extension API, and every high-level calculation owns or shares
a native gpuxtb context.

For the concise package page and install commands, see
[`python/README.md`](../../python/README.md).

## Single systems and context reuse

`Calculator` holds the atomic species, compute settings, and native context.
Use it as a context manager so native resources are released deterministically.
Updating only positions reuses the context and immutable topology setup.

```python
import numpy as np
from gpuxtb import Calculator

numbers = np.array([1, 1])
positions = np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]])

with Calculator(
    "GFN2-xTB",
    numbers,
    positions,
    backend="cpu",
    cpu_threads=1,
    electronic_temperature=300.0,
) as calc:
    first = calc.singlepoint()
    calc.update(positions=positions * 1.01)
    second = calc.singlepoint()
```

The high-level calculator deliberately performs fresh SCC initialization for
each calculation. Strict electronic `WARM` checkpoints are an advanced C ABI
feature because callers must preserve an exact topology and compute-policy
identity.

`Calculator.set()` updates `max_scc_iterations`, `charge_tolerance`,
`energy_tolerance`, or `electronic_temperature` in kelvin.

## Results

`Result` supports both attributes and tblite-like string keys:

```python
energy = first.energy
forces = first["forces"]
gradient = first["gradient"]  # exactly -forces
charges = first.get("charges")
iterations = first.scc_iterations
```

Single-system `Calculator.singlepoint()` raises when its one system does not
converge or its eigensolver fails. A batch preserves successful peers by
default.

## Ragged batches

Each `Structure` can have a different atom count, charge, spin state, and
embedding data. `BatchCalculator` flattens them into offsets and arrays for one
native call.

```python
import numpy as np
from gpuxtb import BatchCalculator, Structure

systems = [
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

with BatchCalculator(systems, backend="auto") as calc:
    result = calc.compute()

print(result.energies)
print(result.failed_indices)
result.raise_for_status()
```

`auto_batch_size=True` partitions a large CUDA workload using current free
memory and retries recoverable multi-system native allocation failures at a
smaller chunk size. An integer sets a target maximum total atom count per
chunk. Systems are indivisible, and chunking preserves input order and
peer-local diagnostics.

## Open-shell calculations

`multiplicity` and `uhf` describe the same state, with
`uhf = multiplicity - 1`. Supplying inconsistent values is an error.
Open-shell systems default to two unrestricted spin channels:

```python
radical = Structure(
    [7, 1, 1],
    np.array(
        [
            [0.0, 0.0, 0.0],
            [1.8, 0.0, 0.0],
            [-0.6, 1.7, 0.0],
        ]
    ),
    charge=0,
    multiplicity=2,
)
```

Set `spin_channels=1` only when the restricted open-shell formulation is
intended. Both restricted and unrestricted GFN2-xTB are implemented on CPU and
CUDA.

## Point charges and periodic response

`PointCharge` positions use bohr, charge values use elementary-charge units,
and `gammas` are positive screening parameters in Hartree.

```python
from gpuxtb import PointCharge

embedding = PointCharge(
    positions=np.array([[4.0, 0.0, 0.0]]),
    charges=np.array([0.5]),
    gammas=np.array([0.405771]),
)
embedded_system = Structure([1, 1], systems[0].positions, point_charges=embedding)
```

When point charges are present, results include `point_charge_forces`. gpuxtb
does not include point-charge/point-charge energy or force.

`ChargeResponse` adds a per-atom shift `b` and symmetric response matrix `A`:

```python
from gpuxtb import ChargeResponse

response = ChargeResponse(
    shifts=np.array([0.003, -0.002]),
    matrix=np.array([[0.020, 0.001], [0.001, 0.018]]),
)
periodic_embedding = Structure(
    [1, 1],
    np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]]),
    charge_response=response,
)
```

The caller owns coordinate derivatives of `b` and `A`. See the
[QM/MM user guide](qmmm.md) and [theory page](../theory/qmmm.md).

## ASE and dpdata

Install the corresponding extra and use `gpuxtb.ase.GPUxtb` as an ASE
calculator or `driver="gpuxtb"` with dpdata. These integrations convert native
atomic units to eV and Angstrom conventions. dpdata periodic systems are
rejected because the native ABI has no lattice descriptor.

## Native-library discovery

Published wheels place `libgpuxtb` inside the Python package. On supported
Linux platforms, the package discovers the separately installed OpenBLAS and
optional `nvidia-*` CUDA providers without requiring `LD_LIBRARY_PATH`.

`GPUXTB_LIBRARY` can override the bundled native library for development or a
managed deployment. The override must be ABI-compatible with the Python
package, and its own loader search paths remain the deployer's responsibility.
