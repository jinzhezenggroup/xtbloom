# C and C++ API

gpuxtb is implemented in C++17, but its only public native interface is the C
header `gpuxtb/gpuxtb.h`. The header is C11-compatible and has `extern "C"`
guards, so C and C++ consumers share the same ABI and semantics.

## Install and link

```console
cmake -S . -B build/release -G Ninja \
  -DGPUXTB_ENABLE_CUDA=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --parallel
cmake --install build/release --prefix "$PWD/build/install"
```

A consumer CMake project needs only the exported target:

```cmake
cmake_minimum_required(VERSION 3.24)
project(gpuxtb_example LANGUAGES C CXX)

find_package(gpuxtb CONFIG REQUIRED)
add_executable(gpuxtb_example main.c)
target_link_libraries(gpuxtb_example PRIVATE gpuxtb::gpuxtb)
target_compile_features(gpuxtb_example PRIVATE c_std_11)
```

Configure it with the install prefix:

```console
cmake -S example -B example/build -G Ninja \
  -DCMAKE_PREFIX_PATH="$PWD/build/install"
cmake --build example/build
example/build/gpuxtb_example
```

## Complete energy and force example

This example submits one H2 molecule using host buffers. The same descriptors
are accepted by a CUDA context, which stages host data internally.

```c
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <gpuxtb/gpuxtb.h>

static gpuxtb_const_buffer_t input_buffer(const void *data, size_t size) {
  gpuxtb_const_buffer_t buffer = {data, size, GPUXTB_MEMORY_HOST, 0};
  return buffer;
}

static gpuxtb_buffer_t output_buffer(void *data, size_t size) {
  gpuxtb_buffer_t buffer = {data, size, GPUXTB_MEMORY_HOST, 0};
  return buffer;
}

int main(void) {
  const int64_t atom_offsets[] = {0, 2};
  const int32_t atomic_numbers[] = {1, 1};
  const double positions[] = {-0.70, 0.0, 0.0, 0.70, 0.0, 0.0};
  const double molecular_charges[] = {0.0};
  const int32_t unpaired_electrons[] = {0};

  gpuxtb_context_options_t context_options;
  gpuxtb_batch_t batch;
  gpuxtb_compute_options_t compute_options;
  gpuxtb_batch_result_t result;
  if (gpuxtb_context_options_init(&context_options, sizeof(context_options)) !=
          GPUXTB_STATUS_SUCCESS ||
      gpuxtb_batch_init(&batch, sizeof(batch)) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb_compute_options_init(&compute_options, sizeof(compute_options)) !=
          GPUXTB_STATUS_SUCCESS ||
      gpuxtb_batch_result_init(&result, sizeof(result)) != GPUXTB_STATUS_SUCCESS) {
    fprintf(stderr, "descriptor initialization failed: %s\n",
            gpuxtb_get_last_error());
    return 1;
  }

  context_options.backend = GPUXTB_BACKEND_CPU;

  batch.batch_size = 1;
  batch.total_atoms = 2;
  batch.atom_offsets = input_buffer(atom_offsets, sizeof(atom_offsets));
  batch.atomic_numbers = input_buffer(atomic_numbers, sizeof(atomic_numbers));
  batch.positions = input_buffer(positions, sizeof(positions));
  batch.molecular_charges =
      input_buffer(molecular_charges, sizeof(molecular_charges));
  batch.unpaired_electrons =
      input_buffer(unpaired_electrons, sizeof(unpaired_electrons));

  compute_options.flags = GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES |
                          GPUXTB_COMPUTE_ATOMIC_CHARGES;

  double energy = NAN;
  double forces[6];
  double charges[2];
  int32_t iterations = -1;
  uint8_t converged = 0;
  gpuxtb_status_t system_status = GPUXTB_STATUS_INTERNAL_ERROR;
  result.energies = output_buffer(&energy, sizeof(energy));
  result.forces = output_buffer(forces, sizeof(forces));
  result.atomic_charges = output_buffer(charges, sizeof(charges));
  result.scc_iterations = output_buffer(&iterations, sizeof(iterations));
  result.scc_converged = output_buffer(&converged, sizeof(converged));
  result.per_system_status =
      output_buffer(&system_status, sizeof(system_status));

  gpuxtb_context_t *context = NULL;
  gpuxtb_status_t status = gpuxtb_context_create(&context_options, &context);
  if (status != GPUXTB_STATUS_SUCCESS) {
    fprintf(stderr, "context creation failed: %s\n", gpuxtb_get_last_error());
    return 2;
  }

  status = gpuxtb_compute(context, &batch, &compute_options, &result);
  if (status != GPUXTB_STATUS_SUCCESS) {
    fprintf(stderr, "compute failed: %s\n", gpuxtb_get_last_error());
    gpuxtb_context_destroy(context);
    return 3;
  }
  gpuxtb_context_destroy(context);
  if (system_status != GPUXTB_STATUS_SUCCESS) {
    fprintf(stderr, "system failed: %s\n", gpuxtb_status_string(system_status));
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

All buffer views are caller-owned borrowed memory. gpuxtb never takes
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
modified outputs and returns `GPUXTB_STATUS_INTERNAL_ERROR` with a thread-local
diagnostic from `gpuxtb_get_last_error()`.

## CUDA descriptors and streams

Set `context_options.backend = GPUXTB_BACKEND_CUDA` and optionally choose
`device_id` or a native `cudaStream_t` cast to `void *`. Each buffer independently
uses `GPUXTB_MEMORY_HOST` or `GPUXTB_MEMORY_CUDA_DEVICE`, so mixed requests are
valid. Pointer ownership and the selected device are validated.

`gpuxtb_compute` is synchronous with respect to the caller even when a custom
stream is supplied. Active CUDA stream capture is rejected. gpuxtb attempts to
restore the caller's current device on every exit.

## gpuxtb-owned result arenas and DLPack export

`gpuxtb_result_owner_t` is an additive, ref-counted result-allocation owner:
one contiguous host or CUDA-device arena that gpuxtb itself allocates, fills
through a normal compute call, and can hand to an importing framework with the
DLPack producer protocol without copying data.

- `gpuxtb_result_owner_create` allocates one arena with an initial reference
  (options: memory space, device id, byte size). `gpuxtb_result_owner_buffer`
  exposes the arena as a caller-owned `gpuxtb_buffer_t` so output slices can
  be bound and computed into.
- `gpuxtb_result_owner_retain` / `gpuxtb_result_owner_release` manage the
  reference count; the allocation is freed exactly once when the last
  reference drops. `release(NULL)` is a no-op and every release must
  correspond to exactly one prior create or retain.
- `gpuxtb_result_owner_export_dltensor` wraps one compact C-contiguous arena
  slice as a heap-allocated `DLManagedTensor` (legacy, `version == 0`) or
  `DLManagedTensorVersioned` (DLPack 1.0, `version != 0`). The managed tensor
  carries a native exact-once deleter, so importing frameworks can release it
  from their own code after the producing context and Python wrapper are gone;
  no hidden host polling or extra device-wide synchronization is required
  because public compute is synchronous. On any failure `*out_managed` is set
  to NULL and no arena reference is taken.
- The Python package mirrors these functions in `gpuxtb.library` and exposes
  the producer through `ArrayBatch.compute(result_memory="cuda")`; see the
  Python guide for the user-facing contract.

## Reuse and warm starts

Keep a context alive across calls to retain worker pools, topology plans, and
backend workspaces. ABI-v2 `GPUXTB_SCC_START_WARM` additionally consumes the
checkpoint from the most recent fully converged compatible batch. It is strict:
a first call, topology/policy change, or missing compatible checkpoint is an
invalid argument and never falls back to `FRESH`.

Geometry may change while topology and compute policy remain identical. CPU
restarts its mixing window from the converged electronic state; CUDA preserves
only epoch-compatible mixer history. See
[architecture](../developer-guide/architecture.md) for the full identity and
cache contract.
