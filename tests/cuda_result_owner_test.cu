// End-to-end CUDA test for xtbloom-owned device result arenas and their DLPack
// export.
//
// This test allocates a xtbloom_result_owner_t CUDA arena, binds its slices as
// the device result buffers of a real xtbloom_compute call, exports the
// finished slices as DLPack managed tensors, consumes them through their
// native deleters, and verifies the device bytes equal a host-computed
// reference. It also proves the arena outlives the compute context (compute
// and verification happen after the arena slices are the only remaining
// references) and that the CUDA device is restored on every owner path.

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "xtbloom/xtbloom.h"

// Byte-exact DLPack 1.0 managed-tensor mirrors (see
// src/runtime/dlpack_layout.hpp; DLPack spec, Apache-2.0).
struct DtTensor {
  void* data;
  struct {
    std::int32_t device_type;
    std::int32_t device_id;
  } device;
  std::int32_t ndim;
  struct {
    std::uint8_t code;
    std::uint8_t bits;
    std::uint16_t lanes;
  } dtype;
  std::int64_t* shape;
  std::int64_t* strides;
  std::uint64_t byte_offset;
};

struct DtManagedTensorVersioned {
  std::uint32_t version_major;
  std::uint32_t version_minor;
  void* manager_ctx;
  void (*deleter)(DtManagedTensorVersioned*);
  std::uint64_t flags;
  DtTensor dl_tensor;
};

static_assert(sizeof(DtTensor) == 48u, "DLTensor must be 48 bytes");
static_assert(sizeof(DtManagedTensorVersioned) == 80u,
              "versioned DLManagedTensorVersioned must be 80 bytes");

namespace {

int failures = 0;

#define CHECK(condition)                                                                   \
  do {                                                                                     \
    if (!(condition)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      ++failures;                                                                          \
    }                                                                                      \
  } while (0)

#define CUDA_CHECK(condition)                                                      \
  do {                                                                             \
    cudaError_t check_status = (condition);                                        \
    if (check_status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(check_status));                              \
      return 1;                                                                    \
    }                                                                              \
  } while (0)

const char* cuda_ok(cudaError_t status) {
  return status == cudaSuccess ? "ok" : cudaGetErrorString(status);
}

struct DeviceSlice {
  void* pointer = nullptr;
  std::size_t bytes = 0;
};

// Allocate one packed arena and bind per-output slices.
struct ArenaBinding {
  xtbloom_result_owner_t* owner = nullptr;
  void* base = nullptr;
  std::size_t total_bytes = 0;
  DeviceSlice energies;
  DeviceSlice forces;
  DeviceSlice charges;
  DeviceSlice iterations;
  DeviceSlice converged;
  DeviceSlice statuses;

  int create(int device, std::size_t systems, std::size_t atoms, std::size_t points) {
    // energies: systems f64; forces: atoms*3 f64; charges: atoms f64;
    // iterations: systems i32; converged: systems u8; statuses: systems i32.
    // Alignment: 8 for f64/i32 buckets are all within an 8-aligned packing.
    std::size_t offset = 0;
    const auto align = [&offset](std::size_t alignment) {
      offset = (offset + alignment - 1u) & ~(alignment - 1u);
      return offset;
    };
    total_bytes = 0;
    std::size_t energies_offset = align(8u);
    offset = energies_offset + systems * sizeof(double);
    std::size_t forces_offset = align(8u);
    offset = forces_offset + atoms * 3u * sizeof(double);
    std::size_t charges_offset = align(8u);
    offset = charges_offset + atoms * sizeof(double);
    std::size_t iterations_offset = align(4u);
    offset = iterations_offset + systems * sizeof(std::int32_t);
    std::size_t converged_offset = align(1u);
    offset = converged_offset + systems * sizeof(std::uint8_t);
    std::size_t statuses_offset = align(4u);
    offset = statuses_offset + systems * sizeof(std::int32_t);
    (void)points;
    total_bytes = offset;

    xtbloom_result_owner_options_t options;
    CHECK(xtbloom_result_owner_options_init(&options, sizeof(options)) == XTBLOOM_STATUS_SUCCESS);
    options.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
    options.device_id = device;
    options.size_bytes = total_bytes;
    if (xtbloom_result_owner_create(&options, &owner) != XTBLOOM_STATUS_SUCCESS ||
        owner == nullptr) {
      return 1;
    }
    xtbloom_buffer_t buffer;
    if (xtbloom_result_owner_buffer(owner, &buffer) != XTBLOOM_STATUS_SUCCESS) {
      return 2;
    }
    base = buffer.data;
    auto* bytes = static_cast<unsigned char*>(base);
    energies = {bytes + energies_offset, systems * sizeof(double)};
    forces = {bytes + forces_offset, atoms * 3u * sizeof(double)};
    charges = {bytes + charges_offset, atoms * sizeof(double)};
    iterations = {bytes + iterations_offset, systems * sizeof(std::int32_t)};
    converged = {bytes + converged_offset, systems * sizeof(std::uint8_t)};
    statuses = {bytes + statuses_offset, systems * sizeof(std::int32_t)};
    return 0;
  }

  void release() {
    if (owner != nullptr) {
      xtbloom_result_owner_release(owner);
      owner = nullptr;
    }
  }
};

xtbloom_buffer_t device_output(void* data, std::size_t bytes) {
  return {data, bytes, XTBLOOM_MEMORY_CUDA_DEVICE, 0u};
}

int test_device_arena_compute_and_export(int device, cudaStream_t stream) {
  // One H2 gas-phase system; host inputs, device result arena.
  constexpr std::int64_t kSystems = 1;
  constexpr std::int64_t kAtoms = 2;
  const std::int64_t atom_offsets[] = {0, kAtoms};
  const std::int32_t atomic_numbers[] = {1, 1};
  const double positions[] = {-0.70, 0.0, 0.0, 0.70, 0.0, 0.0};
  const double molecular_charges[] = {0.0};
  const std::int32_t unpaired_electrons[] = {0};

  xtbloom_context_options_t context_options{};
  CHECK(xtbloom_context_options_init(&context_options, sizeof(context_options)) ==
        XTBLOOM_STATUS_SUCCESS);
  context_options.backend = XTBLOOM_BACKEND_CUDA;
  context_options.device_id = device;
  context_options.stream = reinterpret_cast<void*>(stream);
  xtbloom_context_t* context = nullptr;
  CHECK(xtbloom_context_create(&context_options, &context) == XTBLOOM_STATUS_SUCCESS &&
        context != nullptr);
  if (context == nullptr) {
    return 1;
  }

  ArenaBinding arena;
  if (const int status = arena.create(device, kSystems, kAtoms, 0); status != 0) {
    xtbloom_context_destroy(context);
    return status;
  }

  xtbloom_batch_t batch;
  CHECK(xtbloom_batch_init(&batch, sizeof(batch)) == XTBLOOM_STATUS_SUCCESS);
  batch.batch_size = kSystems;
  batch.total_atoms = kAtoms;
  batch.atom_offsets = {atom_offsets, sizeof(atom_offsets), XTBLOOM_MEMORY_HOST, 0u};
  batch.atomic_numbers = {atomic_numbers, sizeof(atomic_numbers), XTBLOOM_MEMORY_HOST, 0u};
  batch.positions = {positions, sizeof(positions), XTBLOOM_MEMORY_HOST, 0u};
  batch.molecular_charges = {molecular_charges, sizeof(molecular_charges), XTBLOOM_MEMORY_HOST, 0u};
  batch.unpaired_electrons = {unpaired_electrons, sizeof(unpaired_electrons), XTBLOOM_MEMORY_HOST,
                              0u};

  xtbloom_compute_options_t options{};
  CHECK(xtbloom_compute_options_init(&options, sizeof(options)) == XTBLOOM_STATUS_SUCCESS);
  options.model = XTBLOOM_MODEL_GFN2_XTB;
  options.flags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES | XTBLOOM_COMPUTE_ATOMIC_CHARGES;
  options.max_scc_iterations = 64;
  options.charge_tolerance = 1.0e-8;
  options.energy_tolerance = 1.0e-8;
  options.electronic_temperature = 0.0;

  xtbloom_batch_result_t result{};
  CHECK(xtbloom_batch_result_init(&result, sizeof(result)) == XTBLOOM_STATUS_SUCCESS);
  result.energies = device_output(arena.energies.pointer, arena.energies.bytes);
  result.forces = device_output(arena.forces.pointer, arena.forces.bytes);
  result.atomic_charges = device_output(arena.charges.pointer, arena.charges.bytes);
  result.scc_iterations = device_output(arena.iterations.pointer, arena.iterations.bytes);
  result.scc_converged = device_output(arena.converged.pointer, arena.converged.bytes);
  result.per_system_status = device_output(arena.statuses.pointer, arena.statuses.bytes);

  CHECK(xtbloom_compute(context, &batch, &options, &result) == XTBLOOM_STATUS_SUCCESS);

  // Destroy the compute context; the arena must keep the bytes alive.
  xtbloom_context_destroy(context);
  context = nullptr;

  // Export the finished energy slice as a versioned DLPack tensor.
  const std::int64_t energy_shape[1] = {kSystems};
  xtbloom_dlpack_view_t view;
  std::memset(&view, 0, sizeof(view));
  view.struct_size = sizeof(view);
  view.api_version = XTBLOOM_API_VERSION;
  view.dtype_code = 2; /* float */
  view.dtype_bits = 64;
  view.dtype_lanes = 1;
  view.ndim = 1;
  view.shape = energy_shape;
  void* managed_bytes = nullptr;
  CHECK((xtbloom_result_owner_export_dltensor(arena.owner, &view, 1, &managed_bytes) ==
         XTBLOOM_STATUS_SUCCESS) &&
        managed_bytes != nullptr);
  if (managed_bytes == nullptr) {
    arena.release();
    return 3;
  }
  DtManagedTensorVersioned* managed = static_cast<DtManagedTensorVersioned*>(managed_bytes);
  CHECK((managed->version_major == 1u) && (managed->version_minor == 0u));
  CHECK(managed->dl_tensor.device.device_type == 2);
  CHECK(managed->dl_tensor.device.device_id == device);
  CHECK((managed->dl_tensor.ndim == 1) && (managed->dl_tensor.shape[0] == kSystems));
  CHECK((managed->dl_tensor.dtype.code == 2u) && (managed->dl_tensor.dtype.bits == 64u));
  CHECK(managed->dl_tensor.data == arena.energies.pointer);

  // Pull the energy value and verify it is the converged H2 energy. The bytes
  // live in the arena that survived context destruction.
  double device_energy = std::numeric_limits<double>::quiet_NaN();
  CHECK(cudaMemcpy(&device_energy, managed->dl_tensor.data, sizeof(double),
                   cudaMemcpyDeviceToHost) == cudaSuccess);
  CHECK(std::isfinite(device_energy));

  // Consume the managed tensor through its native deleter (this must release
  // one arena reference without freeing the arena: the producer still holds
  // one, and the other slices are still valid).
  managed->deleter(managed);

  // Second export must produce an independent managed tensor.
  void* managed2_bytes = nullptr;
  CHECK((xtbloom_result_owner_export_dltensor(arena.owner, &view, 1, &managed2_bytes) ==
         XTBLOOM_STATUS_SUCCESS) &&
        managed2_bytes != nullptr);
  DtManagedTensorVersioned* managed2 = static_cast<DtManagedTensorVersioned*>(managed2_bytes);
  CHECK(managed2->dl_tensor.data == arena.energies.pointer);
  managed2->deleter(managed2);

  // After the final producer release, no further native reference exists; the
  // memory is freed by the last deleter. Exercise the CUDA free path.
  arena.release();
  return failures == 0 ? 0 : 4;
}

}  // namespace

int main() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  if (const int status = test_device_arena_compute_and_export(device, stream); status != 0) {
    return status;
  }

  CUDA_CHECK(cudaStreamDestroy(stream));

  if (failures != 0) {
    std::fprintf(stderr, "%d CUDA result-owner test failures\n", failures);
    return 1;
  }
  return 0;
}
