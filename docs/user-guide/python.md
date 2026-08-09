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

The high-level calculator defaults to fresh SCC initialization for each
calculation (`warm_start=False`), keeping every call reproducible and
independent. With `warm_start=True`, a call reuses the previous fully
converged compatible electronic state retained on the same context as the SCC
initial guess (the strict native `WARM` checkpoint); the first call on a
context and any topology or compute-policy identity change transparently fall
back to a fresh solve. Geometry is not part of the native identity, so a
dynamics or optimization loop that reuses one `Calculator` reconverges from
each previous step's state.

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
peer-local diagnostics. Automatic slicing cannot be combined with
`warm_start=True`: a native context retains one whole-batch checkpoint, not an
independent checkpoint for each chunk.

## Array API and DLPack input arrays

`ArrayBatch` takes the flat ragged-batch descriptor arrays directly and is
the zero-copy entry point. Each array can come from any library implementing
the Array API `__dlpack__`/`__dlpack_device__` producer protocols (NumPy,
CuPy, JAX eager arrays, PyTorch tensors); gpuxtb imports none of those
libraries, consumed buffers are caller-owned, and interface devices are
accepted without a host round trip on the CUDA backend.

```python
from gpuxtb import ArrayBatch

batch = ArrayBatch(
    atom_offsets=np.array([0, 2, 5], dtype=np.int64),
    atomic_numbers=np.array([8, 1, 1, 1, 1], dtype=np.int32),
    positions=positions_np,          # (natoms, 3) float64
    molecular_charges=np.array([0.0, 0.0]),
    unpaired_electrons=np.array([0, 0], dtype=np.int32),
    backend="cuda",
)
result = batch.compute()
```

Host arrays become `GPUXTB_MEMORY_HOST` descriptors; CUDA device arrays
become `GPUXTB_MEMORY_CUDA_DEVICE` descriptors that skip host staging. The
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
NumPy/CuPy/PyTorch buffers into which gpuxtb writes directly. JAX arrays are
never mutated in place.

### gpuxtb-owned device results through DLPack

With `result_memory="cuda"`, `ArrayBatch.compute()` (and the
`compute_arrays()` convenience alias) allocates one gpuxtb-owned CUDA device
arena on the context device for the outputs *not* supplied through `out=` and
returns each slice as a `gpuxtb.DLPackResultBuffer` DLPack producer. Such a
producer hands finished device bytes to importing frameworks without a host
round trip:

```python
import torch
from gpuxtb import ArrayBatch

result = ArrayBatch(..., backend="cuda").compute(result_memory="cuda")
energies = torch.from_dlpack(result.energies)   # CUDA tensor, zero-copy
forces = torch.from_dlpack(result.forces)
```

- `cupy.from_dlpack`, `torch.from_dlpack`, and `jax.dlpack.from_dlpack` all
  consume the same producer object; nothing is copied for the device case.
- A supplied `out=` buffer always wins over the arena, so caller-owned and
  gpuxtb-owned outputs can be mixed in one call.
- Every gpuxtb-owned arena is ref-counted natively and survives the
  `ArrayBatch`/`Context` that filled it. Each exported capsule retains the
  arena independently; `DLPackResultBuffer.close()`/`delete()` (and the
  context manager) release only that producer's reference. A repeated
  `__dlpack__` call creates a fresh single-use capsule, so importing the same
  output more than once is safe.
- The managed-tensor deleter is a native gpuxtb function, so an importing
  framework may release the tensor from its own code long after the Python
  producer is gone; the arena is freed exactly once, when the last reference
  (producer, producer slices, or exported capsules) drops.
- Device-resident `per_system_status`/`scc_converged` diagnostics cannot be
  inspected by `failed_indices`; that helper is a host-NumPy feature and
  raises a precise error when the arrays are gpuxtb-owned device buffers.

`result_memory` accepts only `"host"` (the historical default: fresh host
NumPy arrays) and `"cuda"` (requires the resolved CUDA backend). The arena is
laid out at 64-byte alignment so every exported slice satisfies the alignment
requirements of common DLPack consumers (JAX, CuPy, PyTorch).

For repeated inference on the same topology, the caller-owned `out=` path
remains the preferred steady-state zero-copy route: it allocates no gpuxtb
device memory per call and accepts preallocated caller buffers that the caller
can reuse. The gpuxtb-owned `result_memory="cuda"` path is intended for
callers that want finished device results without managing output buffers
themselves. On the archived RTX 5090 small-molecule workload it passed the
  explicit maximum 5% mean-overhead gate (`7.530 ms` arena versus `7.834 ms`
`out=` across 300 counterbalanced pairs); see
`benchmarks/evidence/issue-214/2026-08-07-rtx5090/` for the raw profiler and
latency evidence.

## PyTorch autograd op (positions gradient only)

`gpuxtb.gpuxtb_torch` runs packed gpuxtb inference on PyTorch tensors (host CPU
or CUDA device) and returns `(energies, forces)` as float64 tensors, with
zero-copy tensor data plane. It is the only autograd entry point in the Python
API, and its gradient contract is deliberately narrow: it supports exactly
`dE/dR = -F` with respect to `positions`, which the native library evaluates
analytically.

```python
import torch
from gpuxtb import gpuxtb_torch

positions = torch.tensor(system.positions, dtype=torch.float64, requires_grad=True)
energies, forces = gpuxtb_torch(
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
  `GPUxtbNotSupportedError`.
- Higher-order differentiation requests such as `create_graph=True` or
  `torch.autograd.functional.hessian` raise `GPUxtbNotSupportedError`; gpuxtb
  never substitutes a partial or zero Hessian for the unavailable `dF/dR`.
- Requesting autograd on any other input — `atomic_numbers`, `atom_offsets`,
  `molecular_charges`, `unpaired_electrons`, `spin_channels` — raises
  `GPUxtbNotSupportedError` eagerly at forward time.
- PyTorch is imported lazily only when the op is called; `import gpuxtb` never
  imports torch or the compiled torch extension.
- Non-contiguous or strided inputs are packed into a compact copy by the op;
  scalar types must still match the C ABI exactly.
- CUDA calls always follow `torch.cuda.current_stream()`. Select a custom
  stream with the ordinary PyTorch context manager:

  ```python
  with torch.cuda.stream(stream):
      energies, forces = gpuxtb_torch(...)
  ```

The op itself is a compiled extension, `libgpuxtb_torch_ext`, written against
the LibTorch Stable ABI (torch >= 2.10). It binds tensor data pointers directly
to the public gpuxtb C ABI and runs one synchronous `gpuxtb_compute` per call,
so energies, forces, and failure semantics match the rest of the package
exactly. Because only ABI-stable symbols are used, a single binary works across
torch releases without per-version rebuilds.  The extension is optional: when
the wheels were built without it (or Torch < 2.10 is installed), calling
`gpuxtb_torch` raises a clear error instead of silently degrading. Building the
extension never downloads or requires torch: its stable headers are vendored in
`cmake/3rdparty/torch-stable` and it links a build-time-only stub
`libtorch_cpu.so`, so the shipped binary simply carries `DT_NEEDED
libtorch_cpu.so` and binds to the torch the end user already imported. Torch is
still required at *runtime* to call `gpuxtb_torch`; the rest of gpuxtb builds
and runs without torch. See `cmake/3rdparty/torch-stable/README.md` for
provenance and regeneration.

`gpuxtb_torch` is eager-only: it drives the native library through a custom op,
which `torch.compile` cannot trace.  The op is therefore marked opaque to the
compiler, so wrapping it in `torch.compile` inserts a clean graph
break and runs it eagerly: correct results and no trace-time error, but no
compilation speedup for the gpuxtb call itself (the surrounding graph is still
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
rejected because the native ABI has no lattice descriptor. The ASE calculator
enables warm start by default (`warm_start=True`), so an ASE dynamics run
automatically seeds each step's SCC from the previous converged state and
falls back to a fresh solve whenever the request's identity changes; pass
`warm_start=False` for bit-reproducible independent steps.

For geometry relaxation, `gpuxtb` also registers a batch minimizer under the
`"gpuxtb"` key. It moves every frame of a dpdata system in lockstep and
evaluates energies and forces for all active frames in a single ragged-batch
`gpuxtb_compute` call per step, instead of the one-frame-at-a-time loop of the
reference `ase` minimizer; converged frames are frozen and dropped from the
batch as it shrinks:

```python
import dpdata
from gpuxtb.dpdata import GPUxtbDriver

system = dpdata.System("geometry.xyz", fmt="xyz")
labeled = system.minimize(
    minimizer="gpuxtb",
    driver=GPUxtbDriver(backend="cuda"),
    fmax=5e-3,  # eV/Angstrom
    max_steps=1000,
)
```

## Native-library discovery

Published wheels place `libgpuxtb` inside the Python package. On supported
Linux platforms, the package discovers the separately installed OpenBLAS and
optional `nvidia-*` CUDA providers without requiring `LD_LIBRARY_PATH`.

`GPUXTB_LIBRARY` can override the bundled native library for development or a
managed deployment. The override must be ABI-compatible with the Python
package, and its own loader search paths remain the deployer's responsibility.
