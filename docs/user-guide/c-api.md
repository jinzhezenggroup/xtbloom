# C and C++ API

xTBloom is implemented in C++17, but its only public native interface is the C
header `xtbloom/xtbloom.h`. The header is C11-compatible and has `extern "C"`
guards, so C and C++ consumers share the same ABI and semantics.

## Install and link

```console
cmake -S . -B build/release -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --parallel
cmake --install build/release --prefix "$PWD/build/install"
```

A consumer CMake project needs only the exported target:

```cmake
cmake_minimum_required(VERSION 3.24)
project(xtbloom_example LANGUAGES C CXX)

find_package(xtbloom CONFIG REQUIRED)
add_executable(xtbloom_example main.c)
target_link_libraries(xtbloom_example PRIVATE xtbloom::xtbloom)
target_compile_features(xtbloom_example PRIVATE c_std_11)
```

Configure it with the install prefix:

```console
cmake -S example -B example/build -G Ninja \
  -DCMAKE_PREFIX_PATH="$PWD/build/install"
cmake --build example/build
example/build/xtbloom_example
```

## Complete energy and force example

This example submits one H2 molecule using host buffers. The same descriptors
are accepted by a CUDA context, which stages host data internally.

```c
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <xtbloom/xtbloom.h>

static xtbloom_const_buffer_t input_buffer(const void *data, size_t size) {
  xtbloom_const_buffer_t buffer = {data, size, XTBLOOM_MEMORY_HOST, 0};
  return buffer;
}

static xtbloom_buffer_t output_buffer(void *data, size_t size) {
  xtbloom_buffer_t buffer = {data, size, XTBLOOM_MEMORY_HOST, 0};
  return buffer;
}

int main(void) {
  const int64_t atom_offsets[] = {0, 2};
  const int32_t atomic_numbers[] = {1, 1};
  const double positions[] = {-0.70, 0.0, 0.0, 0.70, 0.0, 0.0};
  const double molecular_charges[] = {0.0};
  const int32_t unpaired_electrons[] = {0};

  xtbloom_context_options_t context_options;
  xtbloom_batch_t batch;
  xtbloom_compute_options_t compute_options;
  xtbloom_batch_result_t result;
  if (xtbloom_context_options_init(&context_options, sizeof(context_options)) !=
          XTBLOOM_STATUS_SUCCESS ||
      xtbloom_batch_init(&batch, sizeof(batch)) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom_compute_options_init(&compute_options, sizeof(compute_options)) !=
          XTBLOOM_STATUS_SUCCESS ||
      xtbloom_batch_result_init(&result, sizeof(result)) != XTBLOOM_STATUS_SUCCESS) {
    fprintf(stderr, "descriptor initialization failed: %s\n",
            xtbloom_get_last_error());
    return 1;
  }

  context_options.backend = XTBLOOM_BACKEND_CPU;

  batch.batch_size = 1;
  batch.total_atoms = 2;
  batch.atom_offsets = input_buffer(atom_offsets, sizeof(atom_offsets));
  batch.atomic_numbers = input_buffer(atomic_numbers, sizeof(atomic_numbers));
  batch.positions = input_buffer(positions, sizeof(positions));
  batch.molecular_charges =
      input_buffer(molecular_charges, sizeof(molecular_charges));
  batch.unpaired_electrons =
      input_buffer(unpaired_electrons, sizeof(unpaired_electrons));

  compute_options.flags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES |
                          XTBLOOM_COMPUTE_ATOMIC_CHARGES;

  double energy = NAN;
  double forces[6];
  double charges[2];
  int32_t iterations = -1;
  uint8_t converged = 0;
  xtbloom_status_t system_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  result.energies = output_buffer(&energy, sizeof(energy));
  result.forces = output_buffer(forces, sizeof(forces));
  result.atomic_charges = output_buffer(charges, sizeof(charges));
  result.scc_iterations = output_buffer(&iterations, sizeof(iterations));
  result.scc_converged = output_buffer(&converged, sizeof(converged));
  result.per_system_status =
      output_buffer(&system_status, sizeof(system_status));

  xtbloom_context_t *context = NULL;
  xtbloom_status_t status = xtbloom_context_create(&context_options, &context);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    fprintf(stderr, "context creation failed: %s\n", xtbloom_get_last_error());
    return 2;
  }

  status = xtbloom_compute(context, &batch, &compute_options, &result);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    fprintf(stderr, "compute failed: %s\n", xtbloom_get_last_error());
    xtbloom_context_destroy(context);
    return 3;
  }
  xtbloom_context_destroy(context);
  if (system_status != XTBLOOM_STATUS_SUCCESS) {
    fprintf(stderr, "system failed: %s\n", xtbloom_status_string(system_status));
    return 4;
  }

  printf("energy = %.16g Hartree, SCC iterations = %d\n", energy, iterations);
  printf("force on atom 0 = (%.16g, %.16g, %.16g) Hartree/bohr\n",
         forces[0], forces[1], forces[2]);
  return converged == 1 && isfinite(energy) ? 0 : 5;
}
```

## Descriptor rules

Every extensible structure begins with `struct_size` and `api_version`. Always
call the matching initializer with the caller's `sizeof(struct)` before setting
fields. This lets a newer library recognize a shorter compatible structure and
prevents accidental reads beyond the supplied ABI prefix.

All buffer views are caller-owned borrowed memory. xTBloom never takes
ownership, reallocates a caller buffer, or retains request pointers after the
synchronous call returns. Buffer `size_bytes`, memory-space tags, offsets, and
all extents are validated before execution.

Required batch topology uses flat arrays:

- `atom_offsets` has `batch_size + 1` signed 64-bit values;
- `atomic_numbers` has `total_atoms` signed 32-bit values;
- `positions` has `3 * total_atoms` binary64 values;
- molecular charges and unpaired-electron counts have one value per system;
- optional point-charge and charge-response arrays have their own 64-bit
  offsets and total extents.

See the public header for every optional field and its exact element type.

## External interaction attachments (ABI-v3)

The ABI-v3 `xtbloom_batch_t` suffix carries a generic, versioned attachment
slot so future external interactions do not renumber or regrow the batch
layout for each new feature. When present:

- `total_interactions` counts `xtbloom_interaction_t` entries in
  `interaction_descriptors`;
- each descriptor attaches one caller-owned payload block inside the
  `interaction_payload` byte buffer to one batch item (`system_index`);
- `interaction_descriptors` and `interaction_payload` may independently use
  host or CUDA-device memory, like every other buffer.

Every payload block starts with an `int32_t block_version` so the byte layout
of one tag can evolve independently; the header must fit in the payload view
and be aligned to an `int32_t`. The released electric-field block
(`XTBLOOM_INTERACTION_ELECTRIC_FIELD`, block version 1) is 32 bytes:
`int32_t version`, a zero `int32_t reserved`, and three finite binary64 field
components in Hartree per elementary charge per bohr. Its payload offset is
8-byte aligned.

The tag set is reserved for the xtb/tblite/dxtb interaction family: uniform
electric field and field gradient, multipole point charges, atomic-potential
grids, ALPB/GBSA/GB/GBE/ddX solvation, D3/D4 dispersion variants, and
halogen-bond corrections. The **CPU backend executes the uniform electric
field** (`XTBLOOM_INTERACTION_ELECTRIC_FIELD`): every other reserved tag is
refused with `XTBLOOM_STATUS_NOT_IMPLEMENTED`, and the CUDA backend currently
refuses all interaction execution with `XTBLOOM_STATUS_NOT_IMPLEMENTED`, both
before any caller output is touched, so a reserved interaction can never
silently contribute to a result. Unknown or `XTBLOOM_INTERACTION_NONE` tags,
duplicate `(system_index, type)` attachments, descriptor flag bits, and payload
blocks that are undersized, oversized, misaligned, or outside the payload view
are `XTBLOOM_STATUS_INVALID_ARGUMENT`.

On the CPU backend the uniform electric field contributes a per-atom scalar
potential `vat_i = -E . r_i` and a per-atom dipolar potential `vdp = -E` to the
charge channel of the SCC Hamiltonian on every iteration (matching the pinned
tblite `field.f90` potential), an energy term `-sum_i q_i (E . r_i)
- sum_i E . d_i` in the SCC trace, and the explicit Hellmann-Feynman force
`+q_i E` on atom `i`. The stationary response of the converged charges and
atomic dipoles is already carried by the field potentials. The pinned tblite
0.7.0 analytic gradient applies `+E` per atom and is nonvariational for partial
charges, so xTBloom validates field forces against central differences of its
reported energy instead of treating that gradient as an oracle. Because the
field participates in every SCC iteration it is part of
the strict warm-start identity: a `WARM` call whose field differs from the
latest fully converged compatible call is rejected like any other changed
compute policy.

The ABI-v2 `xtbloom_batch_result_t` suffix adds the dipole outlet:
`dipole_moments` holds `batch_size * 3` binary64 values in atomic units. It is
reported when `XTBLOOM_COMPUTE_DIPOLE_MOMENTS` is set in `compute_options.flags`.
The CPU backend publishes the molecular dipole `sum_i (r_i * q_i + d_i)` over
the converged charge-channel SCC multipoles and sets `XTBLOOM_RESULT_DIPOLE_MOMENTS`;
the CUDA backend currently returns `XTBLOOM_STATUS_NOT_IMPLEMENTED` for this
output until its publication lands. `quadrupole_moments`, `wiberg_orders`, and
`spin_populations` are ABI-reserved outlets whose shape contract is not
published: supplying bytes there is refused with
`XTBLOOM_STATUS_NOT_SUPPORTED`.

Compute-flag bits 16-31 are reserved and must be zero on input. Result flags
are outputs; xTBloom keeps their reserved bits 16-31 zero on successful return.

## Outputs and failures

The three diagnostic arrays `scc_iterations`, `scc_converged`, and
`per_system_status` are required for every nonempty request. Floating-point
output buffers are required only for properties selected in
`compute_options.flags`.

A successful function return means the batch diagnostics were published; it
does not mean every system converged. Inspect each `per_system_status` entry.
Failed system slices contain quiet NaNs, while successful peer slices remain
valid.

Request validation completes before execution or caller-output publication.
Failures before publication leave output buffers and result flags unchanged.
After CUDA publication begins, a catastrophic device/runtime failure may have
modified outputs and returns `XTBLOOM_STATUS_INTERNAL_ERROR` with a thread-local
diagnostic from `xtbloom_get_last_error()`.

## CUDA descriptors and streams

Set `context_options.backend = XTBLOOM_BACKEND_CUDA` and optionally choose
`device_id` or a native `cudaStream_t` cast to `void *`. Each buffer independently
uses `XTBLOOM_MEMORY_HOST` or `XTBLOOM_MEMORY_CUDA_DEVICE`, so mixed requests are
valid. Pointer ownership and the selected device are validated.

`xtbloom_compute` is synchronous with respect to the caller even when a custom
stream is supplied. Active CUDA stream capture is rejected. xTBloom attempts to
restore the caller's current device on every exit.

## xTBloom-owned result arenas and DLPack export

`xtbloom_result_owner_t` is an additive, ref-counted result-allocation owner:
one contiguous host or CUDA-device arena that xTBloom itself allocates, fills
through a normal compute call, and can hand to an importing framework with the
DLPack producer protocol without copying data.

- `xtbloom_result_owner_create` allocates one arena with an initial reference
  (options: memory space, device id, byte size). `xtbloom_result_owner_buffer`
  exposes the arena as a caller-owned `xtbloom_buffer_t` so output slices can
  be bound and computed into.
- `xtbloom_result_owner_retain` / `xtbloom_result_owner_release` manage the
  reference count; the allocation is freed exactly once when the last
  reference drops. `release(NULL)` is a no-op and every release must
  correspond to exactly one prior create or retain.
- `xtbloom_result_owner_export_dltensor` wraps one compact C-contiguous arena
  slice as a heap-allocated `DLManagedTensor` (legacy, `version == 0`) or
  `DLManagedTensorVersioned` (DLPack 1.0, `version != 0`). The managed tensor
  carries a native exact-once deleter, so importing frameworks can release it
  from their own code after the producing context and Python wrapper are gone;
  no hidden host polling or extra device-wide synchronization is required
  because public compute is synchronous. On any failure `*out_managed` is set
  to NULL and no arena reference is taken.
- The Python package mirrors these functions in `xtbloom.library` and exposes
  the producer through `ArrayBatch.compute(result_memory="cuda")`; see the
  Python guide for the user-facing contract.

## Reuse and warm starts

Keep a context alive across calls to retain worker pools, topology plans, and
backend workspaces. ABI-v2 `XTBLOOM_SCC_START_WARM` additionally consumes the
checkpoint from the most recent fully converged compatible batch. It is strict:
a first call, topology/policy change, or missing compatible checkpoint is an
invalid argument and never falls back to `FRESH`.

For otherwise identical inputs at the same geometry, a `WARM` call restarts SCC
from the checkpoint electronic state and reconverges, so its reported energy
agrees with the energy associated with the consumed checkpoint within the SCC
energy tolerance rather than bit-for-bit: the exact reconverged point can
differ in the final ulp across hosts and BLAS kernels. A compatible checkpoint
can come from either `FRESH` or `WARM`. Each fully converged batch call
publishes the checkpoint consumed by the next compatible `WARM` request, so
consecutive `WARM` calls are stateful and do not carry a bitwise-identical
result guarantee.

Geometry may change while topology and compute policy remain identical. CPU
restarts its mixing window from the converged electronic state; CUDA preserves
only epoch-compatible mixer history. See
[architecture](../developer-guide/architecture.md) for the full identity and
cache contract.
