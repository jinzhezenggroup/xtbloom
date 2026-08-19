# Python API

[![PyPI version](https://img.shields.io/pypi/v/xtbloom.svg)](https://pypi.org/project/xtbloom/)

The Python package wraps the public xTBloom C ABI with `ctypes`. There is no
separate CPython extension API, and every high-level calculation owns or shares
a native xTBloom context.

## Installation

Install the published package from PyPI:

```console
pip install xtbloom
```

For supported Linux CUDA 12 environments or optional adapters, install the
matching extras:

```console
pip install "xtbloom[cuda12]"
pip install "xtbloom[ase,dpdata]"
```

Python 3.10 or newer is required. For platform details, native toolchain and
CUDA requirements, and the secondary source-build path, see the
[installation guide](index.md#installation). The concise PyPI-facing package
page is [`python/README.md`](../../python/README.md).

## Single systems and context reuse

`Calculator` holds the atomic species, compute settings, and native context.
Use it as a context manager so native resources are released deterministically.
Updating only positions reuses the context and immutable topology setup.

`Calculator` and `BatchCalculator` accept `"GFN1-xTB"`/`"GFN1"` and
`"GFN2-xTB"`/`"GFN2"`. Both models run on CPU or CUDA through the same native
backend policy. `backend="auto"` prefers CUDA when available and otherwise
falls back to CPU; `backend="cuda"` never substitutes one model for the other.
A build without CUDA rejects an explicit CUDA context with
`BACKEND_UNAVAILABLE`. A nonnegative `device_id` can be used with AUTO or an
explicit CUDA request.

```python
import numpy as np
from xtbloom import BatchCalculator, Calculator, Structure

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

The high-level calculator defaults to fresh SCC initialization for each
calculation (`warm_start=False`), keeping every call reproducible and
independent. With `warm_start=True`, a call reuses the previous fully
converged compatible electronic state retained on the same context as the SCC
initial guess (the strict native `WARM` checkpoint); the first call on a
context and any topology or compute-policy identity change transparently fall
back to a fresh solve. Geometry is not part of the native identity, so a
dynamics or optimization loop that reuses one `Calculator` reconverges from
each previous step's state.

`Calculator.set()` also accepts `scc_mixer="modified_broyden"`,
`scc_mixer_history` in `1..64`, `scc_mixer_damping` in `(0, 1]`, and
`determinism="default"` or `"reproducible"`, in addition to
`max_scc_iterations`, `charge_tolerance`, `energy_tolerance`, and
`electronic_temperature` in kelvin. Reproducible mode is an exact-repeat
contract only for an unchanged build/backend/provider-or-toolkit/device,
descriptor/options, launch geometry, and FRESH/WARM sequence; it does not
promise cross-backend or cross-machine bitwise identity.

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

## Numerical Cartesian Hessians

`Calculator.hessian()` returns one dense QM-coordinate energy Hessian, while
`BatchCalculator.hessian()` returns one matrix per input structure. Both use a
central difference of analytic forces,

\[
H_{:,j} = -\frac{F(R + h e_j) - F(R - h e_j)}{2h}.
\]

```python
systems = [Structure(numbers, positions), Structure(numbers, positions * 1.01)]
with Calculator("GFN2-xTB", numbers, positions, backend="cuda") as calc:
    raw = calc.hessian(step=0.005)
    symmetric = calc.hessian(step=0.005, symmetrize=True)

with BatchCalculator(systems, backend="cuda", cpu_threads=16) as calc:
    hessians = calc.hessian(step=0.005)
```

The single-system output is a C-contiguous NumPy `float64` matrix with shape
`(3 * natoms, 3 * natoms)` and units Hartree/bohr². The batch output is an
input-ordered list of such matrices, so ragged atom counts need no padding.
Batch size does not alter the calculator's `cpu_threads` budget. The default
step is `0.005` bohr. The default `symmetrize=False` preserves the raw
antisymmetric residual as a finite-difference/SCC convergence diagnostic;
`symmetrize=True` applies `0.5 * (H + H.T)` independently to every matrix.

One dense Hessian requires `6 * natoms` independent force calculations. For a
batch, displacement tasks from different Hessians are interleaved in the same
native ragged force calls rather than evaluating one complete Hessian at a
time.
`auto_batch_size=True` is the default: it chooses a conservative atom limit,
creates only one displacement chunk at a time, and retries recoverable native
allocation failures at smaller automatic chunks. A positive integer sets an
explicit maximum atom count per native call; `False` or `None` submits every
displacement at once. For CUDA, these high-level descriptors are host NumPy
inputs and the Hessian returns to host as NumPy.

The method displaces only QM atoms. Explicit point-charge coordinates,
point-charge values/gammas, the uniform electric field, and caller-owned
charge-response `b/A` operators stay fixed. The returned matrix is therefore
only the QM–QM block at that external environment: it excludes QM–point-charge
and point-charge–point-charge blocks, and still excludes `db/dR` and `dA/dR`.
Any displacement SCC/eigensolver failure aborts the call with its batch member,
atom, axis, sign, status, and iteration count. A temporary fresh-SCC context
leaves every calculator geometry and any original warm checkpoint unchanged.

These are explicit numerical Python methods, not analytic coupled-response
Hessians or native C ABI outputs. xTBloom does not yet perform mass weighting,
translation/rotation projection, normal-mode analysis, or thermochemistry.
They do not change the compiled autograd operator, so PyTorch higher-order
autograd remains unsupported.

## Ragged batches

Each `Structure` can have a different atom count, charge, spin state, and
embedding data. `BatchCalculator` flattens them into offsets and arrays for one
native call.

```python
import numpy as np
from xtbloom import BatchCalculator, Structure

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

with BatchCalculator(systems, backend="cuda") as calc:  # Use "cpu" for CPU-only builds.
    result = calc.compute()

print(result.energies)
print(result.failed_indices)
result.raise_for_status()
```

`auto_batch_size=True` partitions a large CUDA workload using current free
memory and retries recoverable multi-system native allocation failures at a
smaller chunk size. An integer sets a target maximum total atom count per
chunk. Systems are indivisible, and chunking preserves input order and
peer-local diagnostics. Automatic slicing cannot be combined with
`warm_start=True`: a native context retains one whole-batch checkpoint, not an
independent checkpoint for each chunk.

## Array API and DLPack input arrays

`ArrayBatch` takes the flat ragged-batch descriptor arrays directly and is
the zero-copy entry point. Each array can come from any library implementing
the Array API `__dlpack__`/`__dlpack_device__` producer protocols (NumPy,
CuPy, JAX eager arrays, PyTorch tensors); xTBloom imports none of those
libraries, consumed buffers are caller-owned, and interface devices are
accepted without a host round trip on the CUDA backend.

`ArrayBatch` and `compute_arrays` accept `"GFN1-xTB"`/`"GFN1"` and
`"GFN2-xTB"`/`"GFN2"` through `method=`, with GFN2-xTB retained as the default.
Both models use the same CPU/CUDA backend policy and packed descriptor contract.

```python
from xtbloom import ArrayBatch

batch = ArrayBatch(
    atom_offsets=np.array([0, 2, 5], dtype=np.int64),
    atomic_numbers=np.array([8, 1, 1, 1, 1], dtype=np.int32),
    positions=positions_np,          # (natoms, 3) float64
    molecular_charges=np.array([0.0, 0.0]),
    unpaired_electrons=np.array([0, 0], dtype=np.int32),
    method="GFN1-xTB",
    backend="cuda",
)
result = batch.compute()
```

Host arrays become `XTBLOOM_MEMORY_HOST` descriptors; CUDA device arrays
become `XTBLOOM_MEMORY_CUDA_DEVICE` descriptors that skip host staging. The
optional `point_charge_*` group and the `atomic_potential_shifts` /
`charge_response_offsets` / `charge_response_matrix` response group mirror
`PointCharge`/`ChargeResponse` and must be supplied all-or-nothing.

Zero-copy is the default contract: `copy=False` requires the exact dtype,
shape, and a compact C-contiguous layout, and anything else raises instead of
silently copying. `copy=True` asks the producer for a suitable copy. CUDA
device arrays are only accepted on the CUDA backend and must belong to the
context's resolved device; CUDA-managed memory, ROCm, and lazy/tracer
objects are rejected with precise diagnostics. `stream` selects the native
`CUstream` used by the context and passed to CUDA producers (DLPack stream
value `1` for the legacy default stream).

Copies may pack a non-contiguous layout but do not coerce scalar types; every
descriptor must still use the exact dtype required by the C ABI.

`ArrayBatch.compute()` supports an explicit output policy: results are
ordinary host NumPy arrays by default, or `out=` may name writable
NumPy/CuPy/PyTorch buffers into which xTBloom writes directly. JAX arrays are
never mutated in place.

### xTBloom-owned device results through DLPack

With `result_memory="cuda"`, `ArrayBatch.compute()` (and the
`compute_arrays()` convenience alias) allocates one xTBloom-owned CUDA device
arena on the context device for the outputs *not* supplied through `out=` and
returns each slice as a `xtbloom.DLPackResultBuffer` DLPack producer. Such a
producer hands finished device bytes to importing frameworks without a host
round trip:

```python
import torch
from xtbloom import ArrayBatch

result = ArrayBatch(..., backend="cuda").compute(result_memory="cuda")
energies = torch.from_dlpack(result.energies)   # CUDA tensor, zero-copy
forces = torch.from_dlpack(result.forces)
```

- `cupy.from_dlpack`, `torch.from_dlpack`, and `jax.dlpack.from_dlpack` all
  consume the same producer object; nothing is copied for the device case.
- A supplied `out=` buffer always wins over the arena, so caller-owned and
  xTBloom-owned outputs can be mixed in one call.
- Every xTBloom-owned arena is ref-counted natively and survives the
  `ArrayBatch`/`Context` that filled it. Each exported capsule retains the
  arena independently; `DLPackResultBuffer.close()`/`delete()` (and the
  context manager) release only that producer's reference. A repeated
  `__dlpack__` call creates a fresh single-use capsule, so importing the same
  output more than once is safe.
- The managed-tensor deleter is a native xTBloom function, so an importing
  framework may release the tensor from its own code long after the Python
  producer is gone; the arena is freed exactly once, when the last reference
  (producer, producer slices, or exported capsules) drops.
- Device-resident `per_system_status`/`scc_converged` diagnostics cannot be
  inspected by `failed_indices`; that helper is a host-NumPy feature and
  raises a precise error when the arrays are xTBloom-owned device buffers.

`result_memory` accepts only `"host"` (the historical default: fresh host
NumPy arrays) and `"cuda"` (requires the resolved CUDA backend). The arena is
laid out at 64-byte alignment so every exported slice satisfies the alignment
requirements of common DLPack consumers (JAX, CuPy, PyTorch).

For repeated inference on the same topology, the caller-owned `out=` path
remains the preferred steady-state zero-copy route: it allocates no xTBloom
device memory per call and accepts preallocated caller buffers that the caller
can reuse. The xTBloom-owned `result_memory="cuda"` path is intended for
callers that want finished device results without managing output buffers
themselves. The allocation-cost question has its own
[benchmark protocol](../../benchmarks/dlpack-result-memory.md) and
[archived evidence](../../benchmarks/evidence/issue-214/2026-08-07-rtx5090/README.md);
API semantics do not depend on one device's measured timing.

## PyTorch autograd op (positions gradient only)

`xtbloom.xtbloom_torch` is a GFN2-xTB-only adapter with no model selector. It
runs packed xtbloom inference on PyTorch tensors (host CPU
or CUDA device) and returns `(energies, forces)` as float64 tensors, with
zero-copy tensor data plane. It is the only autograd entry point in the Python
API, and its gradient contract is deliberately narrow: it supports exactly
`dE/dR = -F` with respect to `positions`, which the native library evaluates
analytically.

```python
import torch
from xtbloom import xtbloom_torch

positions = torch.tensor(system.positions, dtype=torch.float64, requires_grad=True)
energies, forces = xtbloom_torch(
    positions,
    atomic_numbers,   # (natoms,) int32 torch or numpy
    atom_offsets,     # (nsystems + 1,) int64
    molecular_charges,  # (nsystems,) float64
    unpaired_electrons, # (nsystems,) int32
    backend="cuda",
)
loss = energies.sum()
loss.backward()      # positions.grad == -forces
```

- Backpropagation through `forces` (the force Hessian `dF/dR`) raises
  `XTBloomNotSupportedError`.
- Higher-order differentiation requests such as `create_graph=True` or
  `torch.autograd.functional.hessian` raise `XTBloomNotSupportedError`; xTBloom
  never substitutes a partial or zero Hessian for the unavailable `dF/dR`.
- Requesting autograd on any other input — `atomic_numbers`, `atom_offsets`,
  `molecular_charges`, `unpaired_electrons`, `spin_channels` — raises
  `XTBloomNotSupportedError` eagerly at forward time.
- PyTorch is imported lazily only when the op is called; `import xtbloom` never
  imports torch or the compiled torch extension.
- Non-contiguous or strided inputs are packed into a compact copy by the op;
  scalar types must still match the C ABI exactly.
- CUDA calls always follow `torch.cuda.current_stream()`. Select a custom
  stream with the ordinary PyTorch context manager:

  ```python
  with torch.cuda.stream(stream):
      energies, forces = xtbloom_torch(...)
  ```

The op itself is a compiled extension, `libxtbloom_torch_ext`, written against
the LibTorch Stable ABI (torch >= 2.10). It binds tensor data pointers directly
to the public xtbloom C ABI. CPU calls complete synchronously; CUDA calls return
ordinary `(energies, forces)` tensors ordered on `torch.cuda.current_stream()`,
just like other CUDA-enabled PyTorch operations. Stream management and native
execution lifetime are implementation details rather than Python API choices.

Because only ABI-stable symbols are used, a single binary works across torch
releases without per-version rebuilds. The extension is optional: when the
wheels were built without it (or Torch < 2.10 is installed), calling
`xtbloom_torch` raises a clear error instead of silently degrading. Building the
extension never downloads or requires torch: its stable headers are vendored
in `cmake/3rdparty/torch-stable` and it links a build-time-only platform stub.
The shipped plugin retains only the official runtime edge
(`DT_NEEDED libtorch_cpu.so` on Linux, `@rpath/libtorch_cpu.dylib` on macOS, or
`torch_cpu.dll` on Windows) and binds to the torch the end user already
imported. Torch is still required at *runtime* to call `xtbloom_torch`; the rest
of xtbloom builds and runs without torch. See
`cmake/3rdparty/torch-stable/README.md` for provenance, supported wheel cohorts,
and regeneration.

`xtbloom_torch` is eager-only: it drives the native library through a custom op,
which `torch.compile` cannot trace.  The op is therefore marked opaque to the
compiler, so wrapping it in `torch.compile` inserts a clean graph
break and runs it eagerly: correct results and no trace-time error, but no
compilation speedup for the xtbloom call itself (the surrounding graph is still
compiled).

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
intended. Both restricted and unrestricted GFN1-xTB and GFN2-xTB are
implemented on CPU and CUDA.

## Point charges and periodic response

`PointCharge` positions use bohr, charge values use elementary-charge units,
and `gammas` are positive screening parameters in Hartree.

```python
from xtbloom import PointCharge

embedding = PointCharge(
    positions=np.array([[4.0, 0.0, 0.0]]),
    charges=np.array([0.5]),
    gammas=np.array([0.405771]),
)
embedded_system = Structure([1, 1], systems[0].positions, point_charges=embedding)
```

When point charges are present, results include `point_charge_forces`. xTBloom
does not include point-charge/point-charge energy or force.
GFN1 and GFN2 use distinct model-specific screening equations; supplied GFN1
gamma values are harmonic hardnesses, not GFN2 shell-hardness parameters.

`ChargeResponse` adds a per-atom shift `b` and symmetric response matrix `A`:

```python
from xtbloom import ChargeResponse

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

Install `xtbloom[ase]`, `xtbloom[dpdata]`, or both with
`pip install "xtbloom[ase,dpdata]"`, then use `xtbloom.ase.XTBloom` as an ASE
calculator or `driver="xtbloom"` with dpdata. These integrations convert native
atomic units to eV and Angstrom conventions. dpdata periodic systems are
rejected because the adapters do not yet expose the ABI-v4 lattice descriptor
and native periodic GFN1/GFN2 execution is not implemented. The ASE calculator
enables warm start by default (`warm_start=True`), so an ASE dynamics run
automatically seeds each step's SCC from the previous converged state and
falls back to a fresh solve whenever the request's identity changes; pass
`warm_start=False` for bit-reproducible independent steps.

ASE and dpdata accept both model names and use the same CPU/CUDA AUTO policy as
the high-level calculators. GFN1 electric-field/dipole requests are rejected
rather than evaluated as GFN2.

For geometry relaxation, `xtbloom` also registers a batch minimizer under the
`"xtbloom"` key. It moves every frame of a dpdata system in lockstep and
evaluates energies and forces for all active frames in a single ragged-batch
`xtbloom_compute` call per step, instead of the one-frame-at-a-time loop of the
reference `ase` minimizer; converged frames are frozen and dropped from the
batch as it shrinks:

```python
import dpdata
from xtbloom.dpdata import XTBloomDriver

system = dpdata.System("geometry.xyz", fmt="xyz")
labeled = system.minimize(
    minimizer="xtbloom",
    driver=XTBloomDriver(backend="cuda"),
    fmax=5e-3,  # eV/Angstrom
    max_steps=1000,
)
```

## Native-library discovery

Published wheels place `libxtbloom` inside the Python package. On supported
Linux platforms, CPU inference uses the wheel's private, auditwheel-vendored
OpenBLAS provider. The upstream `scipy-openblas32` package is not a runtime
dependency and should not be installed for xTBloom. Optional `nvidia-*` CUDA
providers are still discovered separately without requiring `LD_LIBRARY_PATH`.

`XTBLOOM_LIBRARY` can override the bundled native library for development or a
managed deployment. The override must be ABI-compatible with the Python
package, and its own loader search paths remain the deployer's responsibility.
