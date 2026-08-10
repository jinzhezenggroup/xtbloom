# Array, DLPack, and Autograd Contract

## Entry Points and Units

`xtbloom.ArrayBatch` is the reusable packed-array interface. `xtbloom.compute_arrays` is its one-shot convenience form. Both consume eager Array API/DLPack producers without importing their framework.

The Python array interfaces use:

- positions in bohr;
- energies in Hartree;
- forces in Hartree/bohr;
- charges in elementary-charge units;
- `electronic_temperature` in kelvin.

## Required Input Arrays

All dtypes are exact; no scalar conversion is performed.

| Name | Shape | Dtype |
| --- | --- | --- |
| `atom_offsets` | `(nsystems + 1,)` | `int64` |
| `atomic_numbers` | `(natoms,)` | `int32` |
| `positions` | `(natoms, 3)` | `float64` |
| `molecular_charges` | `(nsystems,)` | `float64` |
| `unpaired_electrons` | `(nsystems,)` | `int32` |
| `spin_channels` | `(nsystems,)` | `int32` |

If `spin_channels` is omitted, `ArrayBatch` creates a host `int32` array filled with restricted value `1`.

Optional point-charge inputs are all-or-none:

| Name | Shape | Dtype |
| --- | --- | --- |
| `point_charge_offsets` | `(nsystems + 1,)` | `int64` |
| `point_charge_positions` | `(npoints, 3)` | `float64` |
| `point_charge_values` | `(npoints,)` | `float64` |
| `point_charge_gammas` | `(npoints,)` | `float64` |

Optional charge-response inputs are all-or-none:

| Name | Shape | Dtype |
| --- | --- | --- |
| `atomic_potential_shifts` | `(natoms,)` | `float64` |
| `charge_response_offsets` | `(nsystems + 1,)` | `int64` |
| `charge_response_matrix` | `(total_response_elements,)` | `float64` |

Each per-system response matrix block is packed row-major into the flat matrix array.

## Output Arrays

| Public output name | Shape | Dtype |
| --- | --- | --- |
| `energies` | `(nsystems,)` | `float64` |
| `forces` | `(natoms, 3)` | `float64` |
| `charges` or `atomic_charges` | `(natoms,)` | `float64` |
| `point_charge_forces` | `(npoints, 3)` | `float64` |
| `scc_iterations` | `(nsystems,)` | `int32` |
| `scc_converged` | `(nsystems,)` | `uint8` |
| `per_system_status` | `(nsystems,)` | `int32` |

Diagnostics are required for every nonempty batch internally, even when the caller is interested only in energies or forces.

## Layout and Copy Policy

Every array must implement both `__dlpack__` and `__dlpack_device__`.

With `copy=False`, the default, an input must already have the exact shape, dtype, lane count, byte extent, alignment, and compact C-contiguous layout. Validation raises instead of silently copying.

With `copy=True`, the producer may create a compact copy for a non-contiguous input. Dtype is still exact and is never coerced. Report this path as a packing copy, not zero-copy.

`out=` buffers always bind directly with `copy=False`, regardless of the input copy policy. They must be writable, exact-shaped, exact-typed, and compact. A temporary copied output would violate the advertised alias, so it is never allowed.

`xtbloom_torch` differs slightly: it packs non-contiguous tensor inputs into compact tensors before the native call. That operation remains correct but is not end-to-end zero-copy for those inputs.

## Supported Devices

Supported DLPack device kinds are:

- CPU memory;
- CUDA-pinned host memory, treated as a host descriptor;
- CUDA device memory.

CUDA device arrays require the resolved CUDA backend and must belong to the context's exact device. xTBloom does not perform implicit cross-device copies. Host and CUDA descriptors may be mixed intentionally on a CUDA context.

ROCm, CUDA-managed memory, and other device kinds are rejected. Lazy or tracer values from JIT, grad, vmap, or compilation tracing are rejected because xTBloom requires an eager concrete buffer.

## Stream Semantics

`ArrayBatch(stream=...)` accepts a raw native `CUstream` handle. When `stream=None`, the native context uses the CUDA legacy default stream and CUDA DLPack producers receive protocol stream value `1`. A custom stream handle is forwarded according to the DLPack producer contract.

`ArrayBatch.compute()` uses the synchronous public compute boundary. Input producers and bound output buffers must remain alive through its return.

`xtbloom_torch` does not expose a raw stream argument. CUDA execution follows `torch.cuda.current_stream()` and returns tensors ordered on that stream, like an ordinary CUDA-enabled PyTorch operation. Select another stream with `torch.cuda.stream(stream)`.

## Output Allocation Policies

The default `result_memory="host"` allocates fresh host NumPy storage for every requested output not present in `out=`.

`out=` accepts a mapping from output name to a writable NumPy, CuPy, or PyTorch array. Caller-supplied buffers always win over `result_memory`, so numerical outputs can be on CUDA while diagnostics remain in preallocated host NumPy arrays.

JAX arrays are immutable and cannot be `out=` targets. Consume a DLPack result into JAX or construct a new JAX array instead.

`result_memory="cuda"` requires a resolved CUDA backend. xTBloom allocates one aligned device arena for outputs not supplied through `out=` and exposes each slice as a `DLPackResultBuffer`. Importers such as PyTorch, CuPy, and JAX can consume those producers without a host round trip.

If `per_system_status` and `scc_converged` are device-resident, `ArrayBatchResult.failed_indices` is unavailable because that helper requires host NumPy diagnostics. Keep those two outputs on host through `out=` when the helper is needed.

For repeated fixed-shape inference, preallocated `out=` is the preferred steady-state route. It avoids xTBloom allocating a device result arena on each call. `result_memory="cuda"` is convenient when the caller does not want to manage result buffers, but it performs a result-arena allocation per call.

## Ownership and Lifetime

Inputs and caller-owned outputs remain owned by the caller. xTBloom borrows their bytes and does not retain them after synchronous `ArrayBatch.compute()` returns.

An xTBloom-owned result arena is native and reference-counted. Each `DLPackResultBuffer` slice retains it, and each exported DLPack capsule takes an independent reference. Consequences:

- an importing framework tensor can outlive the `ArrayBatch` and its native context;
- repeated `__dlpack__` calls produce fresh single-use capsules;
- closing one producer prevents future exports from that producer but does not invalidate tensors already imported from earlier capsules;
- the native arena is freed exactly once after the last producer or imported capsule releases its reference.

Use context managers or explicit `close()` calls in long-running processes. Never close a producer before all desired exports have been created.

## Failure Semantics

Call-level validation happens before caller-output publication. A validation failure must not be treated as a partially valid result.

SCC or eigensolver failure is per-system data. Successful peers remain valid, while failed floating-point slices contain NaNs and diagnostics identify the failed systems. Inspect `per_system_status` and `scc_converged`; do not use finiteness of a batch aggregate as the only success criterion.

## PyTorch Autograd Boundary

`xtbloom.xtbloom_torch` accepts the same required packed inputs and returns:

- `energies`: `(nsystems,)`, `float64`, Hartree;
- `forces`: `(natoms, 3)`, `float64`, Hartree/bohr.

`positions` must be a `float64` PyTorch tensor of shape `(natoms, 3)` and is the only differentiable input. For a loss depending on energies, backward uses the analytic relation `dE/dR = -F`, weighted by each system's upstream energy gradient.

The operation rejects:

- `requires_grad=True` on atomic numbers, offsets, molecular charges, unpaired electrons, or spin channels;
- gradients flowing through the returned forces, because the force Hessian `dF/dR` is unavailable;
- higher-order differentiation, Hessians, and `create_graph=True`;
- attempts to substitute a zero or partial higher derivative.

The operation is eager-only. Inside `torch.compile` it creates a graph break and executes correctly outside the compiled graph; the xTBloom call itself receives no compilation speedup.
