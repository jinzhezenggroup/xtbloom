# xTBloom C ABI contract

## Public boundary

- Include `<xtbloom/xtbloom.h>` from an installed package.
- Link the exported CMake target `xtbloom::xtbloom`.
- The public boundary is C-compatible even though the implementation is C++17.
- GFN1-xTB is implemented on CPU; GFN2-xTB is implemented on CPU and CUDA.
  Treat ROCm as reserved, and never silently route GFN1-xTB to CUDA.
- Prefer the symbols and size macros in the installed header over copied
  numeric values. Additive suffixes may appear without changing an older
  compatible prefix.

## Structure initialization

Every extensible structure starts with `struct_size` and `api_version`. Always
call its matching initializer with the caller's complete size before assigning
fields, for example:

```c
xtbloom_batch_t batch;
if (xtbloom_batch_init(&batch, sizeof(batch)) != XTBLOOM_STATUS_SUCCESS) {
  /* Read xtbloom_get_last_error() before another failing API call. */
}
```

Use the initializer for context options, batch, compute options, result, and
any request/workspace structure used by the application. Leave reserved fields
at the initialized zero value. Do not zero a new structure and assume that its
semantic defaults match the library.

## Units and scalar types

All public real values are IEEE binary64. Use:

| Quantity | C type | Unit |
| --- | --- | --- |
| Positions | `double` | bohr |
| Energies | `double` | Hartree |
| Forces | `double` | Hartree/bohr |
| Molecular, atomic, and point charges | `double` | elementary charge |
| Electronic temperature | `double` | `k_B T` in Hartree |
| Atomic numbers, unpaired electrons, statuses | fixed public 32-bit types | exact integer/tag |
| Ragged offsets and totals | `int64_t` | element counts |

Use `XTBLOOM_KELVIN_TO_HARTREE` when converting a Kelvin temperature at the C
boundary. At finite temperature the reported variational energy is the
electronic Helmholtz free energy, and forces are its negative coordinate
derivative under the documented external-operator convention.

## Ragged batch inputs

For `batch_size = B` and `total_atoms = N`:

- `atom_offsets`: `B + 1` `int64_t` entries; starts at zero, is strictly
  increasing, and ends at `N`.
- `atomic_numbers`: `N` `int32_t` entries.
- `positions`: `3N` `double` entries.
- `molecular_charges`: `B` `double` entries.
- `unpaired_electrons`: `B` `int32_t` entries.
- optional `spin_channels`: `B` `int32_t` entries, each one for restricted or
  two for unrestricted; absence preserves the restricted default.

Optional point-charge, charge-response, and interaction fields have additional
all-or-nothing and shape rules. Use the dedicated QM/MM skill for those fields.

## Borrowed buffer descriptors

`xtbloom_const_buffer_t` and `xtbloom_buffer_t` are byte-sized views. Set:

- `data` to the first byte;
- `size_bytes` to the available byte extent, not an element count;
- `memory_space` to `XTBLOOM_MEMORY_HOST` or, on a CUDA context, optionally
  `XTBLOOM_MEMORY_CUDA_DEVICE`;
- `reserved` to zero.

The library never takes ownership. Inputs may alias other read-only inputs,
but an active output must not overlap any input or another active output. A CPU
context accepts host buffers only. A CUDA context accepts host, device, and
mixed descriptors, but every device pointer must belong to the resolved CUDA
device.

For synchronous `xtbloom_compute`, keep all views valid until the function
returns. For an accepted `xtbloom_plan_compute_enqueue`, descriptor images and
host inputs are copied or consumed before enqueue returns, but CUDA-device
inputs and all outputs remain borrowed until the request reaches `COMPLETE`.

## Compute options and required outputs

Set `model = XTBLOOM_MODEL_GFN2_XTB` explicitly. Select at least one known
compute flag and keep reserved flag bits zero. The initializer supplies valid
positive SCC defaults; override them only with finite, valid values.

For every nonempty batch, bind these diagnostic outputs regardless of selected
properties:

- `scc_iterations`: `B` `int32_t` values;
- `scc_converged`: `B` `uint8_t` values;
- `per_system_status`: `B` `xtbloom_status_t` values.

Bind floating-point outputs for every requested property with these logical
extents:

- energies: `B` doubles;
- forces: `3N` doubles;
- atomic charges: `N` doubles;
- point-charge forces: three doubles per point charge;
- dipoles, when supported and requested: three doubles per system.

An unrequested output may be an empty descriptor. Do not bind bytes to
ABI-reserved outputs whose shape is not published.

## Failure and publication semantics

Always distinguish two levels:

1. The API return value reports request validation or call-level execution.
2. After an API-level `SUCCESS`, each `per_system_status` reports `SUCCESS`,
   `SCC_NOT_CONVERGED`, or `EIGENSOLVER_FAILED`.

Peer-local SCC/eigensolver failure does not fail the whole call. Requested
floating-point slices for a failed system are filled completely with quiet
NaNs; successful peers remain valid. `scc_converged` is one exactly for a
successful system.

Complete request validation happens before execution or output publication.
A failure before caller-output commit leaves result flags and buffers
unchanged. Once CUDA output commit begins, a later catastrophic failure may
have modified output and returns `XTBLOOM_STATUS_INTERNAL_ERROR`; read the
thread-local diagnostic immediately.

## Backend and stream semantics

- `XTBLOOM_BACKEND_CPU` requires CPU execution.
- `XTBLOOM_BACKEND_CUDA` requires CUDA execution and fails if unavailable.
- `XTBLOOM_BACKEND_AUTO` may choose CUDA or fall back to CPU; inspect
  `xtbloom_context_get_backend()` before making device-specific assumptions.
- A CUDA context may receive a native `cudaStream_t` cast to `void *`.
- Synchronous compute waits for its work and output publication before return.
- Active CUDA stream capture is rejected.
- The library attempts to restore the caller's current CUDA device on every
  exit; restoration failure is an internal error and may leave device
  selection changed.

Request objects can be created for CPU and CUDA contexts, but the enqueue
capabilities are narrower:

- `xtbloom_compute_enqueue` is a reserved context-convenience entry point. It
  returns `XTBLOOM_STATUS_NOT_SUPPORTED` on CPU and
  `XTBLOOM_STATUS_NOT_IMPLEMENTED` on CUDA in the current API.
- `xtbloom_plan_compute_enqueue` is the connected fixed-topology CUDA path. It
  returns `XTBLOOM_STATUS_NOT_SUPPORTED` on CPU and currently accepts `FRESH`
  only; strict `WARM` is not supported asynchronously.

Reusing a pending request is invalid. Query/wait API status is separate from
the submitted compute status stored in request info.

## Context, plan, and warm-state reuse

Keep a context alive to reuse CPU workers and backend workspaces. A fixed-
topology plan can additionally reuse validated topology and queried workspace.
Destroy requests and plans before destroying their context.

`XTBLOOM_SCC_START_WARM` strictly consumes the latest fully converged compatible
checkpoint on the same context. Compatibility includes topology and compute
policy: requested properties; charge, spin, and unpaired-electron state;
point-charge/response structure; SCC tolerances and iteration limit; and
electronic temperature. Geometry may change and is reconverged from the saved
electronic state.

A first-call warm request, changed topology/policy, or missing checkpoint is an
invalid argument before output modification. Warm never falls back to fresh.
An accepted fresh attempt consumes an older compatible checkpoint before it
runs; if that fresh attempt later fails, a stale checkpoint cannot be reused.
Do not require bitwise-identical warm results: reconvergence may differ in the
last ulp while respecting the SCC tolerance.
