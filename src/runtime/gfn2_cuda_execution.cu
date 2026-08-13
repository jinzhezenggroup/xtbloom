// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_d4.cuh"
#include "backends/cuda/gfn2_energy_force_execution.cuh"
#include "backends/cuda/gfn2_external_point_charges.cuh"
#include "backends/cuda/gfn2_geometry.cuh"
#include "backends/cuda/gfn2_inference_publication.cuh"
#include "backends/cuda/gfn2_preprocessing.cuh"
#include "backends/cuda/gfn2_public_result_bridge.cuh"
#include "backends/cuda/gfn2_scc_iteration_arena.cuh"
#include "backends/cuda/gfn2_scc_iteration_initialize.cuh"
#include "backends/cuda/gfn2_scc_iteration_reports.cuh"
#include "backends/cuda/gfn2_scc_loop.cuh"
#include "backends/cuda/gfn2_scc_setup_eigensolver.cuh"
#include "backends/cuda/gfn2_scc_setup_inputs.cuh"
#include "backends/cuda/gfn2_scc_setup_topology.hpp"
#include "backends/cuda/gfn2_terminal_classical_energy.cuh"
#include "data/parameters/d4.hpp"
#include "model/gfn2/aes2.hpp"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/coordination.hpp"
#include "model/gfn2/d4.hpp"
#include "model/gfn2/eigensolver.hpp"
#include "model/gfn2/es2.hpp"
#include "model/gfn2/es3.hpp"
#include "model/gfn2/external_point_charges.hpp"
#include "model/gfn2/h0.hpp"
#include "model/gfn2/integrals.hpp"
#include "model/gfn2/lattice.hpp"
#include "model/gfn2/mulliken.hpp"
#include "model/gfn2/periodic_embedding.hpp"
#include "model/gfn2/repulsion.hpp"
#include "model/gfn2/scc_driver.hpp"
#include "model/gfn2/scc_mixer.hpp"
#include "model/gfn2/wavefunction.hpp"
#include "runtime/backend.hpp"
#include "runtime/cuda_descriptor_validation.hpp"
#include "runtime/gfn2_cuda_execution.hpp"
#include "runtime/gfn2_cuda_topology_staging.hpp"
#include "runtime/nvidia_host_api.h"
#include "runtime/request.hpp"

namespace xtbloom::detail {
namespace {

using namespace xtbloom::detail::cuda;
using namespace xtbloom::detail::gfn2;

constexpr std::int32_t kDefaultMixerHistory = 8;
constexpr double kDefaultMixerDamping = 0.4;
constexpr std::uint64_t kInitialGeometryGeneration = 1u;
constexpr std::uint64_t kInitialStateGeneration = 1u;
constexpr std::size_t kArenaAlignment = 256u;
/* One committed physical superset serves every pair consumer.  D4 two-body
 * reaches 50 bohr; narrower roles re-evaluate positions with their own
 * inclusive 25/30-bohr predicates and never infer physical membership solely
 * from list presence. */
constexpr double kD4PairlistBuilderCutoffBohr = 50.0;

#if defined(XTBLOOM_CUDA_TEST_HOOKS)
std::atomic<std::uint32_t> g_execution_test_fault{
    static_cast<std::uint32_t>(Gfn2CudaExecutionTestFault::kNone)};
std::atomic<std::uint64_t> g_native_lattice_allocation_faults{0u};
std::atomic<std::uint64_t> g_native_lattice_completion_faults{0u};
std::atomic<std::uint64_t> g_native_lattice_teardown_faults{0u};
std::atomic<std::uint64_t> g_quarantined_native_lattice_arenas{0u};
std::atomic<std::uint64_t> g_quarantined_native_lattice_bytes{0u};

bool consume_execution_test_fault(Gfn2CudaExecutionTestFault fault) noexcept {
  std::uint32_t expected = static_cast<std::uint32_t>(fault);
  return g_execution_test_fault.compare_exchange_strong(
      expected, static_cast<std::uint32_t>(Gfn2CudaExecutionTestFault::kNone),
      std::memory_order_acq_rel, std::memory_order_relaxed);
}
#endif

Gfn2CudaSccStartMode public_scc_start_mode(const xtbloom_compute_options_t& options) noexcept {
  return options.struct_size >= XTBLOOM_COMPUTE_OPTIONS_V2_SIZE &&
                 options.scc_start_mode == XTBLOOM_SCC_START_WARM
             ? Gfn2CudaSccStartMode::kWarm
             : Gfn2CudaSccStartMode::kFresh;
}

/* ABI-v3 mixer and reproducibility controls form one complete suffix. A
 * caller that supplies only a prefix of the suffix receives the established
 * production defaults; no individual field becomes visible early. */
bool has_compute_options_v3(const xtbloom_compute_options_t& options) noexcept {
  return options.struct_size >= XTBLOOM_COMPUTE_OPTIONS_V3_SIZE;
}

xtbloom_scc_mixer_t public_scc_mixer(const xtbloom_compute_options_t& options) noexcept {
  return has_compute_options_v3(options) ? options.scc_mixer : XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN;
}

std::int32_t public_scc_mixer_history(const xtbloom_compute_options_t& options) noexcept {
  return has_compute_options_v3(options) ? options.scc_mixer_history : kDefaultMixerHistory;
}

double public_scc_mixer_damping(const xtbloom_compute_options_t& options) noexcept {
  return has_compute_options_v3(options) ? options.scc_mixer_damping : kDefaultMixerDamping;
}

xtbloom_determinism_t public_determinism(const xtbloom_compute_options_t& options) noexcept {
  return has_compute_options_v3(options) ? options.determinism : XTBLOOM_DETERMINISM_DEFAULT;
}

xtbloom_status_t validate_public_execution_policy(const xtbloom_compute_options_t& options,
                                                  std::string& error) {
  if (!has_compute_options_v3(options)) return XTBLOOM_STATUS_SUCCESS;
  if (options.scc_mixer != XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN) {
    error = "CUDA GFN2 setup supports only modified-Broyden SCC mixing";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (options.scc_mixer_history < 1 || options.scc_mixer_history > 64) {
    error = "CUDA GFN2 SCC mixer history must be between 1 and 64";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!std::isfinite(options.scc_mixer_damping) || options.scc_mixer_damping <= 0.0 ||
      options.scc_mixer_damping > 1.0) {
    error = "CUDA GFN2 SCC mixer damping must be finite and in (0, 1]";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (options.determinism != XTBLOOM_DETERMINISM_DEFAULT &&
      options.determinism != XTBLOOM_DETERMINISM_REPRODUCIBLE) {
    error = "CUDA GFN2 determinism policy is unknown";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (options.reserved_v3 != 0u) {
    error = "CUDA GFN2 compute-options reserved_v3 field must be zero";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

std::uintptr_t opaque_address(const void* pointer) noexcept {
  return reinterpret_cast<std::uintptr_t>(pointer);
}

template <typename T>
Gfn2SccSetupHostArray<T> setup_array(const std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), static_cast<std::int64_t>(values.size())};
}

template <typename T>
std::size_t vector_bytes(const std::vector<T>& values) noexcept {
  return values.capacity() * sizeof(T);
}

template <typename T>
Gfn2SccIterationHostArrayView<T> initialization_array(const T* values,
                                                      std::int64_t elements) noexcept {
  return {elements == 0 ? nullptr : values, elements};
}

bool checked_bytes(std::int64_t elements, std::size_t element_size, std::size_t& bytes) noexcept {
  if (elements < 0) {
    return false;
  }
  const auto count = static_cast<std::uint64_t>(elements);
  if (count > std::numeric_limits<std::size_t>::max() / element_size) {
    return false;
  }
  bytes = static_cast<std::size_t>(count) * element_size;
  return true;
}

bool checked_elements(std::int64_t elements, std::int64_t factor, std::int64_t& product) noexcept {
  if (elements < 0 || factor < 0 ||
      (factor != 0 && elements > std::numeric_limits<std::int64_t>::max() / factor)) {
    return false;
  }
  product = elements * factor;
  return true;
}

bool checked_triangle(std::int64_t atoms, std::int64_t& pairs) noexcept {
  if (atoms < 0) {
    return false;
  }
  if (atoms <= 1) {
    pairs = 0;
    return true;
  }
  const std::int64_t lower = atoms - 1;
  if ((atoms & 1) == 0) {
    return checked_elements(atoms / 2, lower, pairs);
  }
  return checked_elements(atoms, lower / 2, pairs);
}

bool native_lattice_active(const xtbloom_batch_t& batch) noexcept {
  return batch.struct_size >= XTBLOOM_BATCH_V4_SIZE &&
         (batch.cell_matrices.data != nullptr || batch.cell_matrices.size_bytes != 0u ||
          batch.periodic_axes.data != nullptr || batch.periodic_axes.size_bytes != 0u);
}

/*
 * CUDA backend objects are linked directly into white-box test executables,
 * while validation.cpp remains hidden inside libxtbloom. Keep this staged HOST
 * content check local instead of creating a link-time dependency on internal
 * shared-library symbols. The predicate itself is shared with the CPU ABI and
 * lattice builder through model/gfn2/lattice.cpp.
 */
xtbloom_status_t validate_host_native_lattice_request(const xtbloom_batch_t& batch,
                                                      std::string& error) {
  if (!native_lattice_active(batch)) return XTBLOOM_STATUS_SUCCESS;

  std::int64_t cell_elements = 0;
  std::size_t cell_bytes = 0u;
  std::size_t axes_bytes = 0u;
  if (batch.batch_size <= 0 || !checked_elements(batch.batch_size, 9, cell_elements) ||
      !checked_bytes(cell_elements, sizeof(double), cell_bytes) ||
      !checked_bytes(batch.batch_size, sizeof(std::int32_t), axes_bytes)) {
    error = "CUDA native-cell extent overflows the host address space";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (batch.cell_matrices.memory_space != XTBLOOM_MEMORY_HOST ||
      batch.periodic_axes.memory_space != XTBLOOM_MEMORY_HOST ||
      batch.cell_matrices.data == nullptr || batch.periodic_axes.data == nullptr ||
      batch.cell_matrices.size_bytes < cell_bytes || batch.periodic_axes.size_bytes < axes_bytes) {
    error = "CUDA native-cell semantic validation requires complete HOST staging buffers";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  bool periodic = false;
  for (std::int64_t system = 0; system < batch.batch_size; ++system) {
    std::int32_t mask = 0;
    std::array<double, 9> cell{};
    std::memcpy(&mask,
                static_cast<const std::byte*>(batch.periodic_axes.data) +
                    static_cast<std::size_t>(system) * sizeof(mask),
                sizeof(mask));
    std::memcpy(cell.data(),
                static_cast<const std::byte*>(batch.cell_matrices.data) +
                    static_cast<std::size_t>(system) * sizeof(cell),
                sizeof(cell));

    if (mask == XTBLOOM_PERIODIC_AXES_NONE) {
      if (!std::all_of(cell.begin(), cell.end(), [](double value) { return value == 0.0; })) {
        error = "a nonperiodic batch item must use an all-zero cell matrix";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      continue;
    }
    if ((mask & ~XTBLOOM_PERIODIC_AXES_XYZ) != 0) {
      error = "periodic_axes contains unknown mask bits";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if (mask != XTBLOOM_PERIODIC_AXES_XYZ) {
      error = "one- and two-dimensional periodic axes are reserved but not supported";
      return XTBLOOM_STATUS_NOT_SUPPORTED;
    }
    if (!gfn2::valid_lattice_cell_3d(cell.data())) {
      error = "a periodic cell must be finite, right-handed, and nonsingular";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    periodic = true;
  }
  if (periodic) {
    error =
        "native lattice/PBC descriptors are valid but periodic GFN2 execution is not "
        "implemented yet";
    return XTBLOOM_STATUS_NOT_IMPLEMENTED;
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_cuda_execution_availability(const xtbloom_batch_t& batch,
                                                      const xtbloom_compute_options_t& options,
                                                      std::string& error) {
  if ((options.flags & XTBLOOM_COMPUTE_DIPOLE_MOMENTS) != 0u) {
    error =
        "dipole-moment output is implemented by the CPU backend but is not released on the "
        "CUDA backend yet";
    return XTBLOOM_STATUS_NOT_IMPLEMENTED;
  }
  if (batch.struct_size >= XTBLOOM_BATCH_V3_SIZE && batch.total_interactions != 0) {
    error =
        "interaction execution is implemented by the CPU backend but is not released on the "
        "CUDA backend yet";
    return XTBLOOM_STATUS_NOT_IMPLEMENTED;
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

template <typename T>
xtbloom_status_t copy_host_buffer(const char* name, const xtbloom_const_buffer_t& buffer,
                                  std::int64_t elements, std::vector<T>& output, std::string& error,
                                  bool allow_absent = false) {
  std::size_t required = 0u;
  if (!checked_bytes(elements, sizeof(T), required)) {
    error = std::string(name) + " extent overflows size_t";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (elements == 0 && buffer.data == nullptr && allow_absent) {
    output.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }
  if (buffer.memory_space != XTBLOOM_MEMORY_HOST || buffer.reserved != 0u ||
      (required != 0u && buffer.data == nullptr) || buffer.size_bytes < required) {
    error = std::string(name) + " is not a sufficiently large host buffer";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  output.resize(static_cast<std::size_t>(elements));
  if (required != 0u) {
    std::memcpy(output.data(), buffer.data, required);
  }
  return XTBLOOM_STATUS_SUCCESS;
}

std::uint64_t hash_mix(std::uint64_t value) noexcept {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31u);
}

void hash_append(std::uint64_t& hash, std::uint64_t value) noexcept {
  hash ^= hash_mix(value + 0x9e3779b97f4a7c15ULL + (hash << 6u) + (hash >> 2u));
}

std::uint64_t double_bits(double value) noexcept {
  std::uint64_t bits = 0u;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

template <typename T>
void hash_append_vector(std::uint64_t& hash, const std::vector<T>& values) noexcept {
  hash_append(hash, values.size());
  for (const T value : values) {
    if constexpr (std::is_same_v<T, double>) {
      hash_append(hash, double_bits(value));
    } else {
      hash_append(hash, static_cast<std::uint64_t>(value));
    }
  }
}

class DeviceArena {
 public:
  DeviceArena() = default;
  ~DeviceArena() { reset(); }
  DeviceArena(const DeviceArena&) = delete;
  DeviceArena& operator=(const DeviceArena&) = delete;
  DeviceArena(DeviceArena&&) = delete;
  DeviceArena& operator=(DeviceArena&&) = delete;

  cudaError_t allocate(std::size_t bytes) noexcept {
    reset();
    bytes_ = bytes;
    return bytes == 0u ? cudaSuccess : cudaMalloc(&pointer_, bytes);
  }

  void reset() noexcept {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
    pointer_ = nullptr;
    bytes_ = 0u;
  }

  void* get() const noexcept { return pointer_; }
  std::size_t bytes() const noexcept { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0u;
};

class PinnedArena {
 public:
  PinnedArena() = default;
  ~PinnedArena() { reset(); }
  PinnedArena(const PinnedArena&) = delete;
  PinnedArena& operator=(const PinnedArena&) = delete;
  PinnedArena(PinnedArena&&) = delete;
  PinnedArena& operator=(PinnedArena&&) = delete;

  cudaError_t allocate(std::size_t bytes, bool inject_allocation_failure = false) noexcept {
    if (bytes == 0u) {
      reset();
      return cudaSuccess;
    }

    /* Preserve the old image until a complete replacement exists. In
     * particular, a failed growth must not publish a nonzero capacity beside
     * a null pointer, because callers use capacity to decide whether the
     * retained image is reusable. */
    void* replacement = nullptr;
    const cudaError_t status =
        inject_allocation_failure ? cudaErrorMemoryAllocation : cudaMallocHost(&replacement, bytes);
    if (status != cudaSuccess) return status;

    void* const previous = pointer_;
    pointer_ = replacement;
    bytes_ = bytes;
    if (previous != nullptr) (void)cudaFreeHost(previous);
    return cudaSuccess;
  }

  void reset() noexcept {
    if (pointer_ != nullptr) {
      (void)cudaFreeHost(pointer_);
    }
    pointer_ = nullptr;
    bytes_ = 0u;
  }

  void* get() const noexcept { return pointer_; }
  std::size_t bytes() const noexcept { return bytes_; }

  /* Relinquish an image that cannot be proved idle. CUDA may still reference
   * it, so the caller deliberately leaves it allocated instead of risking a
   * use-after-free in a noexcept teardown path. */
  void* release_without_free() noexcept {
    void* const released = pointer_;
    pointer_ = nullptr;
    bytes_ = 0u;
    return released;
  }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0u;
};

class CudaEvent {
 public:
  CudaEvent() = default;
  ~CudaEvent() { reset(); }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  CudaEvent(CudaEvent&&) = delete;
  CudaEvent& operator=(CudaEvent&&) = delete;

  cudaError_t create(unsigned int flags) noexcept {
    reset();
    return cudaEventCreateWithFlags(&event_, flags);
  }

  void reset() noexcept {
    if (event_ != nullptr) {
      (void)cudaEventDestroy(event_);
    }
    event_ = nullptr;
  }

  cudaEvent_t get() const noexcept { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

class CudaStream {
 public:
  CudaStream() = default;
  ~CudaStream() { reset(); }
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  CudaStream(CudaStream&&) = delete;
  CudaStream& operator=(CudaStream&&) = delete;

  cudaError_t create(unsigned int flags) noexcept {
    reset();
    return cudaStreamCreateWithFlags(&stream_, flags);
  }

  void reset() noexcept {
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
    }
    stream_ = nullptr;
  }

  cudaStream_t get() const noexcept { return stream_; }
  bool valid() const noexcept { return stream_ != nullptr; }

 private:
  cudaStream_t stream_ = nullptr;
};

bool align_up(std::size_t value, std::size_t alignment, std::size_t& aligned) noexcept {
  if (alignment == 0u || (alignment & (alignment - 1u)) != 0u) {
    return false;
  }
  const std::size_t mask = alignment - 1u;
  if (value > std::numeric_limits<std::size_t>::max() - mask) {
    return false;
  }
  aligned = (value + mask) & ~mask;
  return true;
}

class ArenaLayout {
 public:
  template <typename T>
  std::size_t append(std::int64_t elements) {
    std::size_t bytes = 0u;
    if (!checked_bytes(elements, sizeof(T), bytes)) {
      valid_ = false;
      return 0u;
    }
    const std::size_t alignment = std::max<std::size_t>(alignof(T), 8u);
    std::size_t offset = 0u;
    if (!align_up(bytes_, alignment, offset)) {
      valid_ = false;
      return 0u;
    }
    if (offset > std::numeric_limits<std::size_t>::max() - bytes) {
      valid_ = false;
      return 0u;
    }
    bytes_ = offset + bytes;
    return offset;
  }

  bool valid() const noexcept {
    std::size_t ignored = 0u;
    return valid_ && align_up(bytes_, kArenaAlignment, ignored);
  }
  std::size_t bytes() const noexcept {
    std::size_t aligned = 0u;
    return valid_ && align_up(bytes_, kArenaAlignment, aligned) ? aligned : 0u;
  }

 private:
  std::size_t bytes_ = 0u;
  bool valid_ = true;
};

template <typename T>
T* arena_pointer(void* arena, std::size_t offset) noexcept {
  return reinterpret_cast<T*>(static_cast<std::byte*>(arena) + offset);
}

template <typename T>
T* arena_pointer_if(void* arena, std::size_t offset, std::int64_t elements) noexcept {
  return elements == 0 ? nullptr : arena_pointer<T>(arena, offset);
}

/* Stable numerical leaves and transaction metadata passed by value to the
 * runtime-owned refresh kernels. CUDA types stay confined to this translation
 * unit; the public internal boundary above remains usable by a future HIP
 * implementation. */
struct NumericalRefreshDeviceSources {
  const double* positions = nullptr;
  const double* point_positions = nullptr;
  const double* point_values = nullptr;
  const double* point_gammas = nullptr;
  const double* periodic_shifts = nullptr;
  const double* periodic_response = nullptr;
};

/* One accepted asynchronous FRESH request consumes every prior WARM token
 * even when its stream-ordered descriptor validation later rejects numerical
 * execution. The request condition is set after that consumption so only
 * committed numerical/SCC/inference state is protected by the conditional
 * body. */
__global__ void set_request_execution_condition_kernel(const std::uint32_t* request_error,
                                                       cudaGraphConditionalHandle condition) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    cudaGraphSetConditional(condition, *request_error == 0u ? 1u : 0u);
  }
}

struct NumericalRefreshDeviceBinding {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_matrices = 0;
  std::int64_t total_point_charges = 0;
  std::int64_t total_response_elements = 0;
  std::int64_t geometry_pair_elements = 0;
  std::int64_t es2_elements = 0;
  std::int64_t aes2_elements = 0;
  std::uint8_t d4_enabled = 0u;
  std::uint8_t point_enabled = 0u;
  std::uint8_t periodic_enabled = 0u;
  std::uint64_t plan_token = 0u;

  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* shell_offsets = nullptr;
  const std::int64_t* matrix_offsets = nullptr;
  const std::int64_t* geometry_pair_offsets = nullptr;
  const std::int64_t* es2_offsets = nullptr;
  const std::int64_t* point_offsets = nullptr;
  const std::int64_t* response_offsets = nullptr;
  const std::uint8_t* requested = nullptr;
  const std::uint8_t* preprocessing_published = nullptr;
  std::uint8_t* eligible = nullptr;
  /* Canonical eigensolver/SCC activity leaf snapshotted by overlap refactor. */
  std::uint8_t* factor_active = nullptr;
  std::uint64_t* committed_generations = nullptr;
  /* The immediately preceding successfully committed numerical epoch. Warm
   * reset uses this edge instead of rewriting checkpoint generations, so two
   * refreshes without an intervening inference cannot chain a stale state. */
  std::uint64_t* refresh_predecessor_generations = nullptr;
  /* Bound after the inference arena is built. A failed refresh must revoke
   * both the predecessor edge and its consumable warm checkpoint. */
  std::uint64_t* warm_checkpoint_generations = nullptr;

  double* candidate_positions = nullptr;
  double* candidate_point_positions = nullptr;
  double* candidate_point_values = nullptr;
  double* candidate_point_gammas = nullptr;
  double* candidate_periodic_shifts = nullptr;
  double* candidate_periodic_response = nullptr;

  double* committed_positions = nullptr;
  double* committed_point_positions = nullptr;
  double* committed_point_values = nullptr;
  double* committed_point_gammas = nullptr;
  double* committed_periodic_shifts = nullptr;
  double* committed_periodic_response = nullptr;

  const double* candidate_geometry_pairs = nullptr;
  const double* candidate_coordination = nullptr;
  const double* candidate_overlap = nullptr;
  const double* candidate_dipole = nullptr;
  const double* candidate_quadrupole = nullptr;
  const double* candidate_h0 = nullptr;
  const double* candidate_es2 = nullptr;
  const double* candidate_aes2 = nullptr;

  double* public_geometry_pairs = nullptr;
  double* public_coordination = nullptr;
  double* public_overlap = nullptr;
  double* public_dipole = nullptr;
  double* public_quadrupole = nullptr;
  double* public_h0 = nullptr;
  double* public_es2 = nullptr;
  double* public_aes2 = nullptr;

  const double* candidate_d4_coordination = nullptr;
  double* public_d4_coordination = nullptr;
  const double* candidate_point_shell = nullptr;
  double* public_point_shell = nullptr;

  const std::uint32_t* preprocessing_plan_error = nullptr;
  const std::uint32_t* d4_system_errors = nullptr;
  const std::uint32_t* d4_device_error = nullptr;
  const std::uint32_t* point_system_errors = nullptr;
  const std::uint32_t* point_plan_error = nullptr;
  std::uint32_t* periodic_system_errors = nullptr;
  std::uint32_t* periodic_plan_error = nullptr;
  const std::uint64_t* factor_generations = nullptr;
  const std::uint32_t* factor_statuses = nullptr;
  const std::uint64_t* geometry_epoch = nullptr;
};

static_assert(std::is_trivially_copyable_v<NumericalRefreshDeviceSources>);
static_assert(std::is_trivially_copyable_v<NumericalRefreshDeviceBinding>);

__global__ void stage_gfn2_numerical_inputs_kernel(NumericalRefreshDeviceBinding binding,
                                                   NumericalRefreshDeviceSources sources) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= binding.batch_size) return;
  const bool use_source = binding.requested[system] == 1u;

  const std::int64_t atom_begin = binding.atom_offsets[system];
  const std::int64_t atom_end = binding.atom_offsets[system + 1];
  for (std::int64_t index = atom_begin * 3 + threadIdx.x; index < atom_end * 3;
       index += blockDim.x) {
    binding.candidate_positions[index] =
        use_source ? sources.positions[index] : binding.committed_positions[index];
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    if (binding.periodic_enabled != 0u) {
      binding.candidate_periodic_shifts[atom] =
          use_source ? sources.periodic_shifts[atom] : binding.committed_periodic_shifts[atom];
    }
  }

  if (binding.point_enabled != 0u) {
    const std::int64_t point_begin = binding.point_offsets[system];
    const std::int64_t point_end = binding.point_offsets[system + 1];
    for (std::int64_t index = point_begin * 3 + threadIdx.x; index < point_end * 3;
         index += blockDim.x) {
      binding.candidate_point_positions[index] =
          use_source ? sources.point_positions[index] : binding.committed_point_positions[index];
    }
    for (std::int64_t point = point_begin + threadIdx.x; point < point_end; point += blockDim.x) {
      binding.candidate_point_values[point] =
          use_source ? sources.point_values[point] : binding.committed_point_values[point];
      binding.candidate_point_gammas[point] =
          use_source ? sources.point_gammas[point] : binding.committed_point_gammas[point];
    }
  }

  if (binding.periodic_enabled != 0u) {
    const std::int64_t response_begin = binding.response_offsets[system];
    const std::int64_t response_end = binding.response_offsets[system + 1];
    for (std::int64_t index = response_begin + threadIdx.x; index < response_end;
         index += blockDim.x) {
      binding.candidate_periodic_response[index] = use_source
                                                       ? sources.periodic_response[index]
                                                       : binding.committed_periodic_response[index];
    }
  }
}

__global__ void validate_gfn2_periodic_refresh_kernel(NumericalRefreshDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= binding.batch_size || binding.periodic_enabled == 0u ||
      binding.preprocessing_published[system] == 0u ||
      atomicAdd(binding.periodic_plan_error, 0u) != 0u) {
    return;
  }
  const std::int64_t atom_begin = binding.atom_offsets[system];
  const std::int64_t atom_end = binding.atom_offsets[system + 1];
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    if (!isfinite(binding.candidate_periodic_shifts[atom])) {
      atomicCAS(binding.periodic_system_errors + system, 0u, 1u);
    }
  }
  __syncthreads();
  if (atomicAdd(binding.periodic_system_errors + system, 0u) != 0u) return;

  const std::int64_t count = atom_end - atom_begin;
  const std::int64_t response_begin = binding.response_offsets[system];
  const std::int64_t response_end = binding.response_offsets[system + 1];
  if (count < 0 || count > 3037000499LL || response_begin < 0 || response_end < response_begin ||
      response_end - response_begin != count * count) {
    if (threadIdx.x == 0) atomicCAS(binding.periodic_plan_error, 0u, 1u);
    return;
  }
  for (std::int64_t local = threadIdx.x; local < count * count; local += blockDim.x) {
    const double value = binding.candidate_periodic_response[response_begin + local];
    if (!isfinite(value)) {
      atomicCAS(binding.periodic_system_errors + system, 0u, 2u);
      continue;
    }
    const std::int64_t row = local / count;
    const std::int64_t column = local - row * count;
    const double transpose =
        binding.candidate_periodic_response[response_begin + column * count + row];
    if (value != transpose) {
      atomicCAS(binding.periodic_system_errors + system, 0u, 3u);
    }
  }
}

__global__ void initialize_gfn2_refresh_sequence_kernel(std::uint32_t* sequence) {
  if (threadIdx.x == 0) *sequence = 1u;
}

/* Publish the common pre-factor gate into the eigensolver owner's canonical
 * activity mask. The overlap owner then commits factors only for these peers. */
__global__ void gate_gfn2_numerical_refresh_kernel(NumericalRefreshDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= binding.batch_size) return;
  const bool plan_healthy =
      atomicAdd(const_cast<std::uint32_t*>(binding.preprocessing_plan_error), 0u) == 0u &&
      (binding.d4_enabled == 0u ||
       atomicAdd(const_cast<std::uint32_t*>(binding.d4_device_error), 0u) == 0u) &&
      (binding.point_enabled == 0u ||
       atomicAdd(const_cast<std::uint32_t*>(binding.point_plan_error), 0u) == 0u) &&
      (binding.periodic_enabled == 0u || atomicAdd(binding.periodic_plan_error, 0u) == 0u);
  const bool peer_healthy =
      binding.requested[system] == 1u && binding.preprocessing_published[system] == 1u &&
      (binding.d4_enabled == 0u || binding.d4_system_errors[system] == 0u) &&
      (binding.point_enabled == 0u || binding.point_system_errors[system] == 0u) &&
      (binding.periodic_enabled == 0u || binding.periodic_system_errors[system] == 0u);
  if (threadIdx.x == 0) {
    const std::uint8_t eligible = plan_healthy && peer_healthy ? 1u : 0u;
    binding.eligible[system] = eligible;
    binding.factor_active[system] = eligible;
  }
}

__global__ void commit_gfn2_numerical_refresh_kernel(NumericalRefreshDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= binding.batch_size) return;
  const std::uint64_t generation = *binding.geometry_epoch;
  const std::uint64_t previous_generation = binding.committed_generations[system];
  const bool publish = binding.eligible[system] == 1u && binding.factor_statuses[system] == 0u &&
                       binding.factor_generations[system] == generation;
  if (threadIdx.x == 0) {
    binding.eligible[system] = publish ? 1u : 0u;
    binding.refresh_predecessor_generations[system] = publish ? previous_generation : 0u;
    if (!publish) binding.warm_checkpoint_generations[system] = 0u;
  }
  if (!publish) return;

  const std::int64_t atom_begin = binding.atom_offsets[system];
  const std::int64_t atom_end = binding.atom_offsets[system + 1];
  for (std::int64_t index = atom_begin * 3 + threadIdx.x; index < atom_end * 3;
       index += blockDim.x) {
    binding.committed_positions[index] = binding.candidate_positions[index];
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    binding.public_coordination[atom] = binding.candidate_coordination[atom];
    if (binding.d4_enabled != 0u) {
      binding.public_d4_coordination[atom] = binding.candidate_d4_coordination[atom];
    }
    if (binding.periodic_enabled != 0u) {
      binding.committed_periodic_shifts[atom] = binding.candidate_periodic_shifts[atom];
    }
  }

  const std::int64_t matrix_begin = binding.matrix_offsets[system];
  const std::int64_t matrix_end = binding.matrix_offsets[system + 1];
  for (std::int64_t index = matrix_begin + threadIdx.x; index < matrix_end; index += blockDim.x) {
    binding.public_overlap[index] = binding.candidate_overlap[index];
    binding.public_h0[index] = binding.candidate_h0[index];
  }
  /* Integral multipoles are global component-major arrays.  A peer owns its
   * matrix interval in every component plane; treating 3*M or 6*M as one
   * contiguous per-peer interval would publish bytes belonging to adjacent
   * peers and violate transactional rollback. */
  for (std::int64_t matrix = matrix_begin + threadIdx.x; matrix < matrix_end;
       matrix += blockDim.x) {
    for (std::int64_t component = 0; component < 3; ++component) {
      const std::int64_t index = component * binding.total_matrices + matrix;
      binding.public_dipole[index] = binding.candidate_dipole[index];
    }
    for (std::int64_t component = 0; component < 6; ++component) {
      const std::int64_t index = component * binding.total_matrices + matrix;
      binding.public_quadrupole[index] = binding.candidate_quadrupole[index];
    }
  }

  const std::int64_t pair_begin = binding.geometry_pair_offsets[system];
  const std::int64_t pair_end = binding.geometry_pair_offsets[system + 1];
  for (std::int64_t index = pair_begin * kGfn2GeometryPairDataElements + threadIdx.x;
       index < pair_end * kGfn2GeometryPairDataElements; index += blockDim.x) {
    binding.public_geometry_pairs[index] = binding.candidate_geometry_pairs[index];
  }
  for (std::int64_t index = pair_begin * kGfn2AES2PairDataElements + threadIdx.x;
       index < pair_end * kGfn2AES2PairDataElements; index += blockDim.x) {
    binding.public_aes2[index] = binding.candidate_aes2[index];
  }
  const std::int64_t es2_begin = binding.es2_offsets[system];
  const std::int64_t es2_end = binding.es2_offsets[system + 1];
  for (std::int64_t index = es2_begin + threadIdx.x; index < es2_end; index += blockDim.x) {
    binding.public_es2[index] = binding.candidate_es2[index];
  }

  if (binding.point_enabled != 0u) {
    const std::int64_t point_begin = binding.point_offsets[system];
    const std::int64_t point_end = binding.point_offsets[system + 1];
    for (std::int64_t index = point_begin * 3 + threadIdx.x; index < point_end * 3;
         index += blockDim.x) {
      binding.committed_point_positions[index] = binding.candidate_point_positions[index];
    }
    for (std::int64_t point = point_begin + threadIdx.x; point < point_end; point += blockDim.x) {
      binding.committed_point_values[point] = binding.candidate_point_values[point];
      binding.committed_point_gammas[point] = binding.candidate_point_gammas[point];
    }
    const std::int64_t shell_begin = binding.shell_offsets[system];
    const std::int64_t shell_end = binding.shell_offsets[system + 1];
    for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
      binding.public_point_shell[shell] = binding.candidate_point_shell[shell];
    }
  }
  if (binding.periodic_enabled != 0u) {
    const std::int64_t response_begin = binding.response_offsets[system];
    const std::int64_t response_end = binding.response_offsets[system + 1];
    for (std::int64_t index = response_begin + threadIdx.x; index < response_end;
         index += blockDim.x) {
      binding.committed_periodic_response[index] = binding.candidate_periodic_response[index];
    }
  }
  if (threadIdx.x == 0) binding.committed_generations[system] = generation;
}

/*
 * Warm execution reuses the device wavefunction and published multipoles.
 * Same-epoch reuse also retains modified-Broyden history, while predecessor-
 * epoch migration starts a new mixer history window for the refreshed
 * operator. The driver-visible terminal trace belongs to one inference
 * attempt and must always be restarted or the bounded loop would treat the
 * prior converged state as inactive.
 */
struct WarmSccResetDeviceBinding {
  std::int64_t batch_size = 0;
  std::uint64_t plan_token = 0u;
  Gfn2GeometryEpochDevice geometry_epoch{};
  const std::uint8_t* eligible = nullptr;
  const std::uint64_t* committed_generations = nullptr;
  const std::uint64_t* refresh_predecessor_generations = nullptr;
  std::uint64_t* warm_checkpoint_generations = nullptr;
  Gfn2SccMixerDeviceState mixer{};
  Gfn2SccDeviceState scc{};
};

static_assert(std::is_trivially_copyable_v<WarmSccResetDeviceBinding>);

__global__ void reset_gfn2_warm_scc_trace_kernel(WarmSccResetDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= binding.batch_size) return;
  const std::uint64_t epoch = *binding.geometry_epoch.value;
  const std::uint64_t checkpoint = atomicExch(
      reinterpret_cast<unsigned long long*>(binding.warm_checkpoint_generations + system), 0ULL);
  const std::uint64_t predecessor = binding.refresh_predecessor_generations[system];
  const bool compatible = epoch != 0u && checkpoint != 0u && binding.eligible[system] == 1u &&
                          binding.committed_generations[system] == epoch &&
                          (checkpoint == epoch || checkpoint == predecessor);
  const bool migrated = compatible && checkpoint != epoch && checkpoint == predecessor;

  /* A geometry-epoch migration keeps the converged multipoles/wavefunction
   * but starts a new Broyden history window. The old finite-difference basis
   * belongs to the predecessor operator and can be substantially worse than
   * simple damping after even a small coordinate change. Setting iteration
   * zero makes subsequent slots overwrite old history before it can be read. */
  if (migrated) {
    binding.mixer.residual_rms[system] = 0.0;
    binding.mixer.residual_maximum[system] = 0.0;
    binding.mixer.iterations[system] = 0u;
    binding.mixer.system_statuses[system] = XTBLOOM_STATUS_SUCCESS;
    binding.mixer.residual_converged[system] = 0u;
  }

  /* iteration==0 deliberately seeds the first warm energy delta from zero,
   * matching fresh driver accounting while retaining the expensive electronic
   * checkpoint and, for same-epoch reuse, the mixer history. */
  binding.scc.free_energies[system] = 0.0;
  binding.scc.previous_free_energies[system] = 0.0;
  binding.scc.free_energy_changes[system] = 0.0;
  binding.scc.residual_rms[system] = 0.0;
  binding.scc.iterations[system] = 0u;
  binding.scc.converged[system] = 0u;
  binding.scc.system_statuses[system] = compatible || binding.eligible[system] == 0u
                                            ? XTBLOOM_STATUS_SUCCESS
                                            : XTBLOOM_STATUS_INTERNAL_ERROR;
}

struct WarmCheckpointPublicationDeviceBinding {
  std::int64_t batch_size = 0;
  Gfn2GeometryEpochDevice geometry_epoch{};
  const std::uint8_t* eligible = nullptr;
  const std::uint64_t* committed_generations = nullptr;
  const std::uint32_t* publication_plan_error = nullptr;
  const xtbloom_status_t* result_statuses = nullptr;
  const std::uint8_t* result_converged = nullptr;
  std::uint64_t* warm_checkpoint_generations = nullptr;
  std::uint32_t* batch_ready = nullptr;
};

static_assert(std::is_trivially_copyable_v<WarmCheckpointPublicationDeviceBinding>);

__global__ void publish_gfn2_warm_checkpoint_generation_kernel(
    WarmCheckpointPublicationDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= binding.batch_size) return;
  const std::uint64_t epoch = *binding.geometry_epoch.value;
  const bool publish =
      atomicAdd(const_cast<std::uint32_t*>(binding.publication_plan_error), 0u) == 0u &&
      epoch != 0u && binding.eligible[system] == 1u &&
      binding.committed_generations[system] == epoch &&
      binding.result_statuses[system] == XTBLOOM_STATUS_SUCCESS &&
      binding.result_converged[system] == 1u;
  binding.warm_checkpoint_generations[system] = publish ? epoch : 0u;
  if (!publish) atomicExch(binding.batch_ready, 0u);
}

/*
 * Terminal analytic forces consume physical topology-major arrays, whereas
 * the converged SCC state is system-major and spin-major. This stable binding
 * projects the committed state once after the SCC loop so every downstream
 * force primitive can retain its established physical-offset contract.
 */
struct StationaryForceProjectionDeviceBinding {
  std::uint8_t enabled = 0u;
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_matrix_elements = 0;
  std::int64_t total_spin_atoms = 0;
  std::int64_t total_spin_shells = 0;
  std::int64_t total_spin_matrix_elements = 0;

  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* shell_offsets = nullptr;
  const std::int64_t* matrix_offsets = nullptr;
  const std::int64_t* atom_shell_offsets = nullptr;
  const std::int32_t* spin_channels = nullptr;
  const std::int64_t* spin_atom_offsets = nullptr;
  const std::int64_t* spin_shell_offsets = nullptr;
  const std::int64_t* spin_matrix_offsets = nullptr;
  const std::int64_t* spin_coupling_offsets = nullptr;
  const double* spin_coupling_matrices = nullptr;

  const double* packed_density = nullptr;
  const double* packed_energy_weighted_density = nullptr;
  const double* packed_shell_charges = nullptr;
  const double* packed_atomic_charges = nullptr;
  const double* packed_atomic_dipoles = nullptr;
  const double* packed_atomic_quadrupoles = nullptr;

  double* total_density = nullptr;
  double* total_energy_weighted_density = nullptr;
  double* spin_density = nullptr;
  double* shell_charges = nullptr;
  double* atomic_charges = nullptr;
  double* atomic_dipoles = nullptr;
  double* atomic_quadrupoles = nullptr;
  double* spin_shell_potentials = nullptr;
};

static_assert(std::is_trivially_copyable_v<StationaryForceProjectionDeviceBinding>);

__global__ void project_gfn2_stationary_force_state_kernel(
    StationaryForceProjectionDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= binding.batch_size) return;

  const std::int32_t channels = binding.spin_channels[system];
  const std::int64_t atom_begin = binding.atom_offsets[system];
  const std::int64_t atom_end = binding.atom_offsets[system + 1];
  const std::int64_t shell_begin = binding.shell_offsets[system];
  const std::int64_t shell_end = binding.shell_offsets[system + 1];
  const std::int64_t matrix_begin = binding.matrix_offsets[system];
  const std::int64_t matrix_end = binding.matrix_offsets[system + 1];
  const std::int64_t spin_atom_begin = binding.spin_atom_offsets[system];
  const std::int64_t spin_shell_begin = binding.spin_shell_offsets[system];
  const std::int64_t spin_matrix_begin = binding.spin_matrix_offsets[system];
  const std::int64_t physical_shells = shell_end - shell_begin;
  const std::int64_t physical_matrices = matrix_end - matrix_begin;

  for (std::int64_t local = threadIdx.x; local < physical_matrices; local += blockDim.x) {
    const double alpha_density = binding.packed_density[spin_matrix_begin + local];
    const double alpha_weighted = binding.packed_energy_weighted_density[spin_matrix_begin + local];
    if (channels == 1) {
      binding.total_density[matrix_begin + local] = alpha_density;
      binding.total_energy_weighted_density[matrix_begin + local] = alpha_weighted;
      binding.spin_density[matrix_begin + local] = 0.0;
    } else {
      const double beta_density =
          binding.packed_density[spin_matrix_begin + physical_matrices + local];
      const double beta_weighted =
          binding.packed_energy_weighted_density[spin_matrix_begin + physical_matrices + local];
      binding.total_density[matrix_begin + local] = alpha_density + beta_density;
      binding.total_energy_weighted_density[matrix_begin + local] = alpha_weighted + beta_weighted;
      binding.spin_density[matrix_begin + local] = alpha_density - beta_density;
    }
  }

  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    const std::int64_t local = shell - shell_begin;
    binding.shell_charges[shell] = binding.packed_shell_charges[spin_shell_begin + local];
    if (channels == 1) binding.spin_shell_potentials[shell] = 0.0;
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const std::int64_t local_atom = atom - atom_begin;
    const std::int64_t spin_atom = spin_atom_begin + local_atom;
    binding.atomic_charges[atom] = binding.packed_atomic_charges[spin_atom];
    for (std::int64_t component = 0; component < 3; ++component) {
      binding.atomic_dipoles[atom * 3 + component] =
          binding.packed_atomic_dipoles[spin_atom * 3 + component];
    }
    for (std::int64_t component = 0; component < 6; ++component) {
      binding.atomic_quadrupoles[atom * 6 + component] =
          binding.packed_atomic_quadrupoles[spin_atom * 6 + component];
    }
  }

  if (channels == 2) {
    /* v_mag = W*m uses the committed raw magnetization shell population.
     * Coupling rows are atom-local and retain the CPU column accumulation
     * order, avoiding a different roundoff path in force parity checks. */
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
      const std::int64_t atom_shell_begin = binding.atom_shell_offsets[atom];
      const std::int64_t atom_shell_end = binding.atom_shell_offsets[atom + 1];
      const std::int64_t atom_shells = atom_shell_end - atom_shell_begin;
      const std::int64_t coupling_begin = binding.spin_coupling_offsets[atom];
      for (std::int64_t shell = atom_shell_begin + threadIdx.x; shell < atom_shell_end;
           shell += blockDim.x) {
        const std::int64_t row = shell - atom_shell_begin;
        double potential = 0.0;
        for (std::int64_t column = 0; column < atom_shells; ++column) {
          const std::int64_t magnetization_shell =
              spin_shell_begin + physical_shells + atom_shell_begin - shell_begin + column;
          potential += binding.spin_coupling_matrices[coupling_begin + row * atom_shells + column] *
                       binding.packed_shell_charges[magnetization_shell];
        }
        binding.spin_shell_potentials[shell] = potential;
      }
    }
  }
}

cudaError_t project_gfn2_stationary_force_state_cuda(
    const StationaryForceProjectionDeviceBinding& binding, cudaStream_t stream) noexcept {
  if (binding.enabled == 0u) return cudaSuccess;
  if (binding.enabled != 1u || binding.batch_size <= 0 || binding.total_atoms <= 0 ||
      binding.total_shells <= 0 || binding.total_matrix_elements <= 0 ||
      binding.total_spin_atoms < binding.total_atoms ||
      binding.total_spin_shells < binding.total_shells ||
      binding.total_spin_matrix_elements < binding.total_matrix_elements ||
      binding.atom_offsets == nullptr || binding.shell_offsets == nullptr ||
      binding.matrix_offsets == nullptr || binding.atom_shell_offsets == nullptr ||
      binding.spin_channels == nullptr || binding.spin_atom_offsets == nullptr ||
      binding.spin_shell_offsets == nullptr || binding.spin_matrix_offsets == nullptr ||
      binding.spin_coupling_offsets == nullptr || binding.spin_coupling_matrices == nullptr ||
      binding.packed_density == nullptr || binding.packed_energy_weighted_density == nullptr ||
      binding.packed_shell_charges == nullptr || binding.packed_atomic_charges == nullptr ||
      binding.packed_atomic_dipoles == nullptr || binding.packed_atomic_quadrupoles == nullptr ||
      binding.total_density == nullptr || binding.total_energy_weighted_density == nullptr ||
      binding.spin_density == nullptr || binding.shell_charges == nullptr ||
      binding.atomic_charges == nullptr || binding.atomic_dipoles == nullptr ||
      binding.atomic_quadrupoles == nullptr || binding.spin_shell_potentials == nullptr ||
      binding.batch_size > static_cast<std::int64_t>(std::numeric_limits<unsigned int>::max())) {
    return cudaErrorInvalidValue;
  }
  project_gfn2_stationary_force_state_kernel<<<static_cast<unsigned int>(binding.batch_size), 256,
                                               0, stream>>>(binding);
  return cudaPeekAtLastError();
}

struct TopologyKey {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int32_t> spin_channels;
  std::vector<std::int64_t> point_offsets;
  std::vector<std::int64_t> response_offsets;
  std::uint32_t flags = 0u;
  std::int32_t maximum_iterations = 0;
  double charge_tolerance = 0.0;
  double energy_tolerance = 0.0;
  double electronic_temperature = 0.0;
  xtbloom_scc_mixer_t scc_mixer = XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN;
  std::int32_t scc_mixer_history = kDefaultMixerHistory;
  double scc_mixer_damping = kDefaultMixerDamping;
  xtbloom_determinism_t determinism = XTBLOOM_DETERMINISM_DEFAULT;
  bool periodic_enabled = false;

  std::uint64_t fingerprint() const noexcept {
    std::uint64_t hash = 0x4750555854424b59ULL;
    hash_append_vector(hash, atom_offsets);
    hash_append_vector(hash, atomic_numbers);
    hash_append_vector(hash, molecular_charges);
    hash_append_vector(hash, unpaired_electrons);
    hash_append_vector(hash, spin_channels);
    hash_append_vector(hash, point_offsets);
    hash_append_vector(hash, response_offsets);
    hash_append(hash, flags);
    hash_append(hash, static_cast<std::uint32_t>(maximum_iterations));
    hash_append(hash, double_bits(charge_tolerance));
    hash_append(hash, double_bits(energy_tolerance));
    hash_append(hash, double_bits(electronic_temperature));
    hash_append(hash, static_cast<std::uint32_t>(scc_mixer));
    hash_append(hash, static_cast<std::uint32_t>(scc_mixer_history));
    hash_append(hash, double_bits(scc_mixer_damping));
    hash_append(hash, static_cast<std::uint32_t>(determinism));
    hash_append(hash, periodic_enabled ? 1u : 0u);
    return hash == 0u ? 1u : hash;
  }
};

/*
 * One stream-ordered comparison of the immutable fixed-plan descriptors.
 * HOST fields are compared by the submitting thread before launch; only
 * CUDA_DEVICE fields are populated here. The canonical device arrays are
 * plan-owned copies, so the kernel never waits for or dereferences host
 * storage and naturally follows preceding Torch work on the owner stream.
 */
struct FixedTopologyComparisonDeviceBinding {
  const std::int64_t* atom_offsets = nullptr;
  const std::int32_t* atomic_numbers = nullptr;
  const double* molecular_charges = nullptr;
  const std::int32_t* unpaired_electrons = nullptr;
  const std::int32_t* spin_channels = nullptr;
  const std::int64_t* point_offsets = nullptr;
  const std::int64_t* response_offsets = nullptr;
  const double* cell_matrices = nullptr;
  const std::int32_t* periodic_axes = nullptr;

  const std::int64_t* expected_atom_offsets = nullptr;
  const std::int32_t* expected_atomic_numbers = nullptr;
  const double* expected_molecular_charges = nullptr;
  const std::int32_t* expected_unpaired_electrons = nullptr;
  const std::int32_t* expected_spin_channels = nullptr;
  const std::int64_t* expected_point_offsets = nullptr;
  const std::int64_t* expected_response_offsets = nullptr;

  std::int64_t atom_offset_elements = 0;
  std::int64_t atomic_number_elements = 0;
  std::int64_t batch_elements = 0;
  std::int64_t point_offset_elements = 0;
  std::int64_t response_offset_elements = 0;
  std::int64_t lattice_systems = 0;
  std::uint32_t* request_error = nullptr;
};

enum RequestValidationCode : std::uint32_t {
  kRequestValidationSuccess = 0u,
  kRequestNotImplemented = 1u,
  kRequestNotSupported = 2u,
  kRequestInvalidArgument = 3u,
  kRequestTopologyMismatch = 4u,
};

template <typename T>
__device__ bool fixed_topology_value_differs(const T* actual, const T* expected, std::int64_t index,
                                             std::int64_t elements) noexcept {
  return actual != nullptr && index < elements && actual[index] != expected[index];
}

__global__ void compare_fixed_topology_kernel(FixedTopologyComparisonDeviceBinding binding) {
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  std::int64_t maximum = binding.atom_offset_elements;
  if (binding.atomic_number_elements > maximum) maximum = binding.atomic_number_elements;
  if (binding.batch_elements > maximum) maximum = binding.batch_elements;
  if (binding.point_offset_elements > maximum) maximum = binding.point_offset_elements;
  if (binding.response_offset_elements > maximum) maximum = binding.response_offset_elements;
  for (std::int64_t element = index; element < maximum; element += stride) {
    const bool mismatch =
        fixed_topology_value_differs(binding.atom_offsets, binding.expected_atom_offsets, element,
                                     binding.atom_offset_elements) ||
        fixed_topology_value_differs(binding.atomic_numbers, binding.expected_atomic_numbers,
                                     element, binding.atomic_number_elements) ||
        fixed_topology_value_differs(binding.molecular_charges, binding.expected_molecular_charges,
                                     element, binding.batch_elements) ||
        fixed_topology_value_differs(binding.unpaired_electrons,
                                     binding.expected_unpaired_electrons, element,
                                     binding.batch_elements) ||
        fixed_topology_value_differs(binding.spin_channels, binding.expected_spin_channels, element,
                                     binding.batch_elements) ||
        fixed_topology_value_differs(binding.point_offsets, binding.expected_point_offsets, element,
                                     binding.point_offset_elements) ||
        fixed_topology_value_differs(binding.response_offsets, binding.expected_response_offsets,
                                     element, binding.response_offset_elements);
    if (mismatch) atomicMax(binding.request_error, kRequestTopologyMismatch);
  }
}

__global__ void validate_lattice_request_kernel(FixedTopologyComparisonDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= binding.lattice_systems || binding.cell_matrices == nullptr ||
      binding.periodic_axes == nullptr) {
    return;
  }
  const std::int32_t mask = binding.periodic_axes[system];
  const double* cell = binding.cell_matrices + system * 9;
  if (mask == XTBLOOM_PERIODIC_AXES_NONE) {
    for (int element = 0; element < 9; ++element) {
      if (cell[element] != 0.0) {
        atomicMax(binding.request_error, kRequestInvalidArgument);
        return;
      }
    }
    return;
  }
  if ((mask & ~XTBLOOM_PERIODIC_AXES_XYZ) != 0) {
    atomicMax(binding.request_error, kRequestInvalidArgument);
    return;
  }
  if (mask != XTBLOOM_PERIODIC_AXES_XYZ) {
    atomicMax(binding.request_error, kRequestNotSupported);
    return;
  }
  if (!gfn2::valid_lattice_cell_3d_binary64(cell)) {
    atomicMax(binding.request_error, kRequestInvalidArgument);
    return;
  }
  atomicMax(binding.request_error, kRequestNotImplemented);
}

__global__ void mark_request_validation_kernel(std::uint32_t* request_error, std::uint32_t code) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    atomicMax(request_error, code);
  }
}

cudaError_t compare_fixed_topology_async(const FixedTopologyComparisonDeviceBinding& binding,
                                         cudaStream_t stream) noexcept {
  if (binding.request_error == nullptr || binding.atom_offset_elements <= 0 ||
      binding.atomic_number_elements <= 0 || binding.batch_elements <= 0 ||
      binding.point_offset_elements < 0 || binding.response_offset_elements < 0 ||
      binding.expected_atom_offsets == nullptr || binding.expected_atomic_numbers == nullptr ||
      binding.expected_molecular_charges == nullptr ||
      binding.expected_unpaired_electrons == nullptr || binding.expected_spin_channels == nullptr ||
      (binding.point_offset_elements != 0 && binding.expected_point_offsets == nullptr) ||
      (binding.response_offset_elements != 0 && binding.expected_response_offsets == nullptr)) {
    return cudaErrorInvalidValue;
  }
  std::int64_t maximum = binding.atom_offset_elements;
  maximum = std::max(maximum, binding.atomic_number_elements);
  maximum = std::max(maximum, binding.batch_elements);
  maximum = std::max(maximum, binding.point_offset_elements);
  maximum = std::max(maximum, binding.response_offset_elements);
  constexpr int kThreads = 256;
  const std::int64_t required_blocks = (maximum + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(std::min<std::int64_t>(required_blocks, 256));
  compare_fixed_topology_kernel<<<blocks, kThreads, 0, stream>>>(binding);
  return cudaPeekAtLastError();
}

cudaError_t validate_lattice_request_async(const FixedTopologyComparisonDeviceBinding& binding,
                                           cudaStream_t stream) noexcept {
  if (binding.request_error == nullptr || binding.lattice_systems < 0 ||
      (binding.lattice_systems != 0 &&
       (binding.cell_matrices == nullptr || binding.periodic_axes == nullptr))) {
    return cudaErrorInvalidValue;
  }
  if (binding.lattice_systems == 0) {
    return cudaSuccess;
  }
  constexpr int kThreads = 128;
  const int blocks = static_cast<int>((binding.lattice_systems + kThreads - 1) / kThreads);
  validate_lattice_request_kernel<<<blocks, kThreads, 0, stream>>>(binding);
  return cudaPeekAtLastError();
}

cudaError_t mark_request_validation_async(std::uint32_t* request_error, std::uint32_t code,
                                          cudaStream_t stream) noexcept {
  if (request_error == nullptr || code == kRequestValidationSuccess ||
      code > kRequestInvalidArgument) {
    return cudaErrorInvalidValue;
  }
  mark_request_validation_kernel<<<1, 1, 0, stream>>>(request_error, code);
  return cudaPeekAtLastError();
}

enum class TopologyMatch { kMatch, kMismatch, kInvalid };

bool valid_host_extent(const xtbloom_const_buffer_t& buffer, std::size_t required,
                       bool allow_absent = false) noexcept {
  if (required == 0u && buffer.data == nullptr && allow_absent) return true;
  return buffer.memory_space == XTBLOOM_MEMORY_HOST && buffer.reserved == 0u &&
         (required == 0u || buffer.data != nullptr) && buffer.size_bytes >= required;
}

template <typename T>
bool buffer_equals(const xtbloom_const_buffer_t& buffer, const std::vector<T>& expected) noexcept {
  const std::size_t bytes = expected.size() * sizeof(T);
  return valid_host_extent(buffer, bytes, expected.empty()) &&
         (bytes == 0u || std::memcmp(buffer.data, expected.data(), bytes) == 0);
}

bool double_buffer_equals(const xtbloom_const_buffer_t& buffer,
                          const std::vector<double>& expected) noexcept {
  const std::size_t bytes = expected.size() * sizeof(double);
  if (!valid_host_extent(buffer, bytes, expected.empty())) return false;
  const auto* source = static_cast<const std::byte*>(buffer.data);
  for (std::size_t index = 0; index < expected.size(); ++index) {
    double value = 0.0;
    std::memcpy(&value, source + index * sizeof(double), sizeof(double));
    if (value != expected[index]) return false;
  }
  return true;
}

template <typename T>
bool host_topology_buffer_equals(const xtbloom_const_buffer_t& buffer,
                                 const std::vector<T>& expected) noexcept {
  return buffer_equals(buffer, expected);
}

bool host_topology_buffer_equals(const xtbloom_const_buffer_t& buffer,
                                 const std::vector<double>& expected) noexcept {
  return double_buffer_equals(buffer, expected);
}

template <typename T>
bool validate_or_bind_fixed_topology_field(const char* name, const xtbloom_const_buffer_t& buffer,
                                           const std::vector<T>& expected, const T* expected_device,
                                           const T*& device_source, std::string& error) {
  device_source = nullptr;
  if (buffer.memory_space == XTBLOOM_MEMORY_HOST) {
    if (!host_topology_buffer_equals(buffer, expected)) {
      error = std::string(name) + " does not match the fixed CUDA plan topology";
      return false;
    }
    return true;
  }
  if (buffer.memory_space != XTBLOOM_MEMORY_CUDA_DEVICE ||
      (!expected.empty() && (buffer.data == nullptr || expected_device == nullptr))) {
    error = std::string(name) + " has no valid fixed CUDA plan comparison binding";
    return false;
  }
  device_source = static_cast<const T*>(buffer.data);
  return true;
}

bool spin_channel_buffer_equals(const xtbloom_batch_t& batch,
                                const std::vector<std::int32_t>& expected) noexcept {
  /* ABI-v1 and an empty ABI-v2 suffix both mean one restricted channel. */
  const bool supplied =
      batch.struct_size >= XTBLOOM_BATCH_V2_SIZE &&
      (batch.spin_channels.data != nullptr || batch.spin_channels.size_bytes != 0u);
  if (!supplied) {
    return std::all_of(expected.begin(), expected.end(),
                       [](std::int32_t channels) { return channels == 1; });
  }
  return buffer_equals(batch.spin_channels, expected);
}

bool finite_double_buffer(const xtbloom_const_buffer_t& buffer, std::int64_t elements,
                          bool positive = false, bool allow_absent = false) noexcept {
  std::size_t bytes = 0u;
  if (!checked_bytes(elements, sizeof(double), bytes) ||
      !valid_host_extent(buffer, bytes, allow_absent)) {
    return false;
  }
  const auto* source = static_cast<const std::byte*>(buffer.data);
  for (std::int64_t index = 0; index < elements; ++index) {
    double value = 0.0;
    std::memcpy(&value, source + static_cast<std::size_t>(index) * sizeof(double), sizeof(double));
    if (!std::isfinite(value) || (positive && value <= 0.0)) return false;
  }
  return true;
}

/* Allocation-free fixed-topology probe used before constructing any temporary
 * vectors. The public API performs structural validation before reaching this
 * owner; the checks repeated here keep the internal white-box entry point
 * fail-closed and ensure a reused call never hides malformed numerical views. */
TopologyMatch match_existing_topology(const xtbloom_batch_t& batch,
                                      const xtbloom_compute_options_t& options,
                                      const TopologyKey& key, std::string& error,
                                      bool validate_host_numerical = true) {
  const std::int64_t expected_batch = static_cast<std::int64_t>(key.molecular_charges.size());
  const std::int64_t expected_atoms = static_cast<std::int64_t>(key.atomic_numbers.size());
  const std::int64_t expected_points = key.point_offsets.empty() ? 0 : key.point_offsets.back();
  if (batch.batch_size != expected_batch || batch.total_atoms != expected_atoms ||
      batch.total_point_charges != expected_points) {
    return TopologyMatch::kMismatch;
  }
  const bool periodic_enabled =
      batch.atomic_potential_shifts.data != nullptr || batch.total_charge_response_elements != 0 ||
      batch.charge_response_offsets.data != nullptr || batch.charge_response_matrix.data != nullptr;
  if (options.model != XTBLOOM_MODEL_GFN2_XTB || options.flags != key.flags ||
      options.max_scc_iterations != key.maximum_iterations ||
      options.charge_tolerance != key.charge_tolerance ||
      options.energy_tolerance != key.energy_tolerance ||
      options.electronic_temperature != key.electronic_temperature ||
      public_scc_mixer(options) != key.scc_mixer ||
      public_scc_mixer_history(options) != key.scc_mixer_history ||
      public_scc_mixer_damping(options) != key.scc_mixer_damping ||
      public_determinism(options) != key.determinism || periodic_enabled != key.periodic_enabled) {
    return TopologyMatch::kMismatch;
  }

  if (!buffer_equals(batch.atom_offsets, key.atom_offsets) ||
      !buffer_equals(batch.atomic_numbers, key.atomic_numbers) ||
      !double_buffer_equals(batch.molecular_charges, key.molecular_charges) ||
      !buffer_equals(batch.unpaired_electrons, key.unpaired_electrons) ||
      !spin_channel_buffer_equals(batch, key.spin_channels)) {
    /* Distinguish a short/wrong-space descriptor from a legitimate topology
     * change so invalid reuse attempts cannot enter candidate construction. */
    std::size_t atom_offset_bytes = 0u;
    std::size_t atomic_number_bytes = 0u;
    std::size_t batch_double_bytes = 0u;
    std::size_t batch_integer_bytes = 0u;
    if (!checked_bytes(expected_batch + 1, sizeof(std::int64_t), atom_offset_bytes) ||
        !checked_bytes(expected_atoms, sizeof(std::int32_t), atomic_number_bytes) ||
        !checked_bytes(expected_batch, sizeof(double), batch_double_bytes) ||
        !checked_bytes(expected_batch, sizeof(std::int32_t), batch_integer_bytes) ||
        !valid_host_extent(batch.atom_offsets, atom_offset_bytes) ||
        !valid_host_extent(batch.atomic_numbers, atomic_number_bytes) ||
        !valid_host_extent(batch.molecular_charges, batch_double_bytes) ||
        !valid_host_extent(batch.unpaired_electrons, batch_integer_bytes)) {
      error = "fixed-topology reuse received a malformed host topology descriptor";
      return TopologyMatch::kInvalid;
    }
    const bool spin_supplied =
        batch.struct_size >= XTBLOOM_BATCH_V2_SIZE &&
        (batch.spin_channels.data != nullptr || batch.spin_channels.size_bytes != 0u);
    if (spin_supplied && !valid_host_extent(batch.spin_channels, batch_integer_bytes)) {
      error = "fixed-topology reuse received malformed host spin_channels";
      return TopologyMatch::kInvalid;
    }
    return TopologyMatch::kMismatch;
  }

  if (batch.point_charge_offsets.data != nullptr) {
    if (!buffer_equals(batch.point_charge_offsets, key.point_offsets)) {
      std::size_t bytes = 0u;
      if (!checked_bytes(expected_batch + 1, sizeof(std::int64_t), bytes) ||
          !valid_host_extent(batch.point_charge_offsets, bytes)) {
        error = "fixed-topology reuse received malformed point-charge offsets";
        return TopologyMatch::kInvalid;
      }
      return TopologyMatch::kMismatch;
    }
  } else if (expected_points != 0) {
    error = "fixed-topology reuse omitted required point-charge offsets";
    return TopologyMatch::kInvalid;
  }

  /* Response metadata is an all-or-nothing numerical view. Potential shifts
   * may enable the periodic component without a response matrix, but once any
   * response descriptor field is active it must retain the topology-derived
   * dense extent and canonical offsets. This mirrors public validation and
   * keeps the internal white-box entry point fail-closed as well. */
  const bool response_enabled =
      batch.total_charge_response_elements != 0 || batch.charge_response_offsets.data != nullptr ||
      batch.charge_response_offsets.size_bytes != 0u ||
      batch.charge_response_matrix.data != nullptr || batch.charge_response_matrix.size_bytes != 0u;
  if (response_enabled) {
    const std::int64_t expected_response_elements = key.response_offsets.back();
    if (batch.total_charge_response_elements != expected_response_elements ||
        batch.charge_response_offsets.data == nullptr ||
        batch.charge_response_matrix.data == nullptr ||
        !buffer_equals(batch.charge_response_offsets, key.response_offsets) ||
        (validate_host_numerical &&
         !finite_double_buffer(batch.charge_response_matrix, expected_response_elements))) {
      error = "fixed-topology reuse received incomplete or malformed dense charge-response data";
      return TopologyMatch::kInvalid;
    }
  }

  if (validate_host_numerical &&
      (!finite_double_buffer(batch.positions, expected_atoms * 3) ||
       !finite_double_buffer(batch.point_charge_positions, expected_points * 3, false, true) ||
       !finite_double_buffer(batch.point_charge_values, expected_points, false, true) ||
       !finite_double_buffer(batch.point_charge_gammas, expected_points, true, true) ||
       (batch.atomic_potential_shifts.data != nullptr &&
        !finite_double_buffer(batch.atomic_potential_shifts, expected_atoms)))) {
    error = "fixed-topology reuse received a malformed or nonfinite numerical host buffer";
    return TopologyMatch::kInvalid;
  }
  error.clear();
  return TopologyMatch::kMatch;
}

bool context_enqueue_host_topology_probe_available(const xtbloom_batch_t& batch) noexcept {
  const auto host_or_absent = [](const xtbloom_const_buffer_t& buffer) {
    return buffer.data == nullptr || buffer.memory_space == XTBLOOM_MEMORY_HOST;
  };
  return host_or_absent(batch.atom_offsets) && host_or_absent(batch.atomic_numbers) &&
         host_or_absent(batch.molecular_charges) && host_or_absent(batch.unpaired_electrons) &&
         (batch.struct_size < XTBLOOM_BATCH_V2_SIZE || host_or_absent(batch.spin_channels)) &&
         host_or_absent(batch.point_charge_offsets) &&
         host_or_absent(batch.charge_response_offsets);
}

bool context_enqueue_shape_policy_matches(const xtbloom_batch_t& batch,
                                          const xtbloom_compute_options_t& options,
                                          const TopologyKey& key) noexcept {
  const std::int64_t expected_batch = static_cast<std::int64_t>(key.molecular_charges.size());
  const std::int64_t expected_atoms = static_cast<std::int64_t>(key.atomic_numbers.size());
  const std::int64_t expected_points = key.point_offsets.empty() ? 0 : key.point_offsets.back();
  const std::int64_t expected_response =
      key.response_offsets.empty() ? 0 : key.response_offsets.back();
  const bool response_active =
      batch.total_charge_response_elements != 0 || batch.charge_response_offsets.data != nullptr ||
      batch.charge_response_offsets.size_bytes != 0u ||
      batch.charge_response_matrix.data != nullptr || batch.charge_response_matrix.size_bytes != 0u;
  const bool periodic_enabled = batch.atomic_potential_shifts.data != nullptr ||
                                batch.atomic_potential_shifts.size_bytes != 0u || response_active;
  return batch.batch_size == expected_batch && batch.total_atoms == expected_atoms &&
         batch.total_point_charges == expected_points &&
         (!response_active || batch.total_charge_response_elements == expected_response) &&
         options.model == XTBLOOM_MODEL_GFN2_XTB && options.flags == key.flags &&
         options.max_scc_iterations == key.maximum_iterations &&
         options.charge_tolerance == key.charge_tolerance &&
         options.energy_tolerance == key.energy_tolerance &&
         options.electronic_temperature == key.electronic_temperature &&
         public_scc_mixer(options) == key.scc_mixer &&
         public_scc_mixer_history(options) == key.scc_mixer_history &&
         public_scc_mixer_damping(options) == key.scc_mixer_damping &&
         public_determinism(options) == key.determinism && periodic_enabled == key.periodic_enabled;
}

xtbloom_status_t validate_offsets(const char* name, const std::vector<std::int64_t>& offsets,
                                  std::int64_t batch_size, std::int64_t total,
                                  bool require_nonempty, std::string& error) {
  if (offsets.size() != static_cast<std::size_t>(batch_size + 1) || offsets.front() != 0 ||
      offsets.back() != total) {
    error = std::string(name) + " does not delimit the declared ragged batch";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t system = 0; system < batch_size; ++system) {
    const std::int64_t begin = offsets[static_cast<std::size_t>(system)];
    const std::int64_t end = offsets[static_cast<std::size_t>(system + 1)];
    if (end < begin || (require_nonempty && end == begin)) {
      error = std::string(name) + " is not monotone or contains an empty molecule";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t make_topology_key(const xtbloom_batch_t& batch,
                                   const xtbloom_compute_options_t& options, TopologyKey& key,
                                   std::vector<double>& positions,
                                   std::vector<double>& point_positions,
                                   std::vector<double>& point_values,
                                   std::vector<double>& point_gammas,
                                   std::vector<double>& periodic_shifts,
                                   std::vector<double>& periodic_response, std::string& error) {
  if (batch.batch_size <= 0 || batch.total_atoms <= 0 || batch.total_point_charges < 0 ||
      batch.total_charge_response_elements < 0 || options.model != XTBLOOM_MODEL_GFN2_XTB ||
      options.max_scc_iterations <= 0 || !std::isfinite(options.charge_tolerance) ||
      options.charge_tolerance <= 0.0 || !std::isfinite(options.energy_tolerance) ||
      options.energy_tolerance <= 0.0 || !std::isfinite(options.electronic_temperature) ||
      options.electronic_temperature < 0.0) {
    error = "invalid GFN2 CUDA setup dimensions or compute policy";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  xtbloom_status_t status = validate_public_execution_policy(options, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = copy_host_buffer("atom_offsets", batch.atom_offsets, batch.batch_size + 1,
                            key.atom_offsets, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = validate_offsets("atom_offsets", key.atom_offsets, batch.batch_size, batch.total_atoms,
                            true, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = copy_host_buffer("atomic_numbers", batch.atomic_numbers, batch.total_atoms,
                            key.atomic_numbers, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = copy_host_buffer("positions", batch.positions, batch.total_atoms * 3, positions, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = copy_host_buffer("molecular_charges", batch.molecular_charges, batch.batch_size,
                            key.molecular_charges, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = copy_host_buffer("unpaired_electrons", batch.unpaired_electrons, batch.batch_size,
                            key.unpaired_electrons, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  const bool spin_channels_present =
      batch.struct_size >= XTBLOOM_BATCH_V2_SIZE &&
      (batch.spin_channels.data != nullptr || batch.spin_channels.size_bytes != 0u);
  if (spin_channels_present) {
    status = copy_host_buffer("spin_channels", batch.spin_channels, batch.batch_size,
                              key.spin_channels, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  } else {
    key.spin_channels.assign(static_cast<std::size_t>(batch.batch_size), 1);
  }

  for (std::int64_t atom = 0; atom < batch.total_atoms; ++atom) {
    const std::int32_t atomic_number = key.atomic_numbers[static_cast<std::size_t>(atom)];
    if (atomic_number <= 0 || atomic_number > 118) {
      error = "atomic_numbers contains an unsupported element";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::int64_t system = 0; system < batch.batch_size; ++system) {
    if (!std::isfinite(key.molecular_charges[static_cast<std::size_t>(system)])) {
      error = "molecular_charges contains a nonfinite value";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    const std::int32_t channels = key.spin_channels[static_cast<std::size_t>(system)];
    if (channels != 1 && channels != 2) {
      error = "spin_channels values must be one or two";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (const double coordinate : positions) {
    if (!std::isfinite(coordinate)) {
      error = "positions contains a nonfinite value";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  if (batch.total_point_charges != 0 || batch.point_charge_offsets.data != nullptr) {
    status = copy_host_buffer("point_charge_offsets", batch.point_charge_offsets,
                              batch.batch_size + 1, key.point_offsets, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_offsets("point_charge_offsets", key.point_offsets, batch.batch_size,
                              batch.total_point_charges, false, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  } else {
    key.point_offsets.assign(static_cast<std::size_t>(batch.batch_size + 1), 0);
  }
  status = copy_host_buffer("point_charge_positions", batch.point_charge_positions,
                            batch.total_point_charges * 3, point_positions, error, true);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = copy_host_buffer("point_charge_values", batch.point_charge_values,
                            batch.total_point_charges, point_values, error, true);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = copy_host_buffer("point_charge_gammas", batch.point_charge_gammas,
                            batch.total_point_charges, point_gammas, error, true);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  for (const double value : point_positions) {
    if (!std::isfinite(value)) {
      error = "point_charge_positions contains a nonfinite value";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::size_t point = 0; point < point_values.size(); ++point) {
    if (!std::isfinite(point_values[point]) || !std::isfinite(point_gammas[point]) ||
        point_gammas[point] <= 0.0) {
      error = "point charge values must be finite and gammas must be finite and positive";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  key.periodic_enabled =
      batch.atomic_potential_shifts.data != nullptr || batch.total_charge_response_elements != 0 ||
      batch.charge_response_offsets.data != nullptr || batch.charge_response_matrix.data != nullptr;
  if (batch.atomic_potential_shifts.data != nullptr) {
    status = copy_host_buffer("atomic_potential_shifts", batch.atomic_potential_shifts,
                              batch.total_atoms, periodic_shifts, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  } else {
    periodic_shifts.assign(static_cast<std::size_t>(batch.total_atoms), 0.0);
  }

  std::vector<std::int64_t> expected_response_offsets(
      static_cast<std::size_t>(batch.batch_size + 1), 0);
  for (std::int64_t system = 0; system < batch.batch_size; ++system) {
    const std::int64_t atoms = key.atom_offsets[static_cast<std::size_t>(system + 1)] -
                               key.atom_offsets[static_cast<std::size_t>(system)];
    if (atoms > 0 && atoms > std::numeric_limits<std::int64_t>::max() / atoms) {
      error = "periodic response extent overflows int64_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    const std::int64_t elements = atoms * atoms;
    const std::int64_t previous = expected_response_offsets[static_cast<std::size_t>(system)];
    if (elements > std::numeric_limits<std::int64_t>::max() - previous) {
      error = "periodic response prefix sum overflows int64_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    expected_response_offsets[static_cast<std::size_t>(system + 1)] = previous + elements;
  }
  if (batch.total_charge_response_elements != 0 || batch.charge_response_offsets.data != nullptr ||
      batch.charge_response_matrix.data != nullptr) {
    status = copy_host_buffer("charge_response_offsets", batch.charge_response_offsets,
                              batch.batch_size + 1, key.response_offsets, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    if (key.response_offsets != expected_response_offsets ||
        batch.total_charge_response_elements != expected_response_offsets.back()) {
      error = "charge_response_offsets does not match the dense per-system atom layout";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    status = copy_host_buffer("charge_response_matrix", batch.charge_response_matrix,
                              batch.total_charge_response_elements, periodic_response, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  } else {
    key.response_offsets = expected_response_offsets;
    periodic_response.assign(static_cast<std::size_t>(expected_response_offsets.back()), 0.0);
  }
  for (const double value : periodic_shifts) {
    if (!std::isfinite(value)) {
      error = "atomic_potential_shifts contains a nonfinite value";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (const double value : periodic_response) {
    if (!std::isfinite(value)) {
      error = "charge_response_matrix contains a nonfinite value";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  key.flags = options.flags;
  key.maximum_iterations = options.max_scc_iterations;
  key.charge_tolerance = options.charge_tolerance;
  key.energy_tolerance = options.energy_tolerance;
  key.electronic_temperature = options.electronic_temperature;
  key.scc_mixer = public_scc_mixer(options);
  key.scc_mixer_history = public_scc_mixer_history(options);
  key.scc_mixer_damping = public_scc_mixer_damping(options);
  key.determinism = public_determinism(options);
  return XTBLOOM_STATUS_SUCCESS;
}

/*
 * Build topology-derived host plans without downloading caller numerical
 * buffers.  The deterministic seed is used only while constructing stable
 * owners, descriptors, arenas, and provider workspaces.  A public CUDA
 * transaction must refresh the resulting Prepared candidate from the real
 * caller buffers before it may replace the committed runtime.
 */
xtbloom_status_t make_topology_only_seed(
    const Gfn2CudaTopologyHostSnapshot& snapshot, const xtbloom_compute_options_t& options,
    TopologyKey& key, std::vector<double>& positions, std::vector<double>& point_positions,
    std::vector<double>& point_values, std::vector<double>& point_gammas,
    std::vector<double>& periodic_shifts, std::vector<double>& periodic_response,
    std::string& error) {
  if (snapshot.batch_size <= 0 || snapshot.total_atoms < snapshot.batch_size ||
      snapshot.total_point_charges < 0 ||
      snapshot.atom_offsets.size() != static_cast<std::size_t>(snapshot.batch_size + 1) ||
      snapshot.atomic_numbers.size() != static_cast<std::size_t>(snapshot.total_atoms) ||
      snapshot.molecular_charges.size() != static_cast<std::size_t>(snapshot.batch_size) ||
      snapshot.unpaired_electrons.size() != static_cast<std::size_t>(snapshot.batch_size) ||
      snapshot.spin_channels.size() != static_cast<std::size_t>(snapshot.batch_size) ||
      snapshot.point_charge_offsets.size() != static_cast<std::size_t>(snapshot.batch_size + 1) ||
      snapshot.charge_response_offsets.size() !=
          static_cast<std::size_t>(snapshot.batch_size + 1)) {
    error = "CUDA topology staging returned an inconsistent host snapshot";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  xtbloom_status_t status = validate_public_execution_policy(options, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  key.atom_offsets = snapshot.atom_offsets;
  key.atomic_numbers = snapshot.atomic_numbers;
  key.molecular_charges = snapshot.molecular_charges;
  key.unpaired_electrons = snapshot.unpaired_electrons;
  key.spin_channels = snapshot.spin_channels;
  key.point_offsets = snapshot.point_charge_offsets;
  key.response_offsets = snapshot.charge_response_offsets;
  key.flags = options.flags;
  key.maximum_iterations = options.max_scc_iterations;
  key.charge_tolerance = options.charge_tolerance;
  key.energy_tolerance = options.energy_tolerance;
  key.electronic_temperature = options.electronic_temperature;
  key.scc_mixer = public_scc_mixer(options);
  key.scc_mixer_history = public_scc_mixer_history(options);
  key.scc_mixer_damping = public_scc_mixer_damping(options);
  key.determinism = public_determinism(options);
  key.periodic_enabled = snapshot.periodic_enabled;

  std::int64_t coordinate_elements = 0;
  std::int64_t point_coordinate_elements = 0;
  if (!checked_elements(snapshot.total_atoms, 3, coordinate_elements) ||
      !checked_elements(snapshot.total_point_charges, 3, point_coordinate_elements) ||
      snapshot.total_charge_response_elements < 0) {
    error = "CUDA topology-only seed extent overflows int64_t";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
  try {
    positions.assign(static_cast<std::size_t>(coordinate_elements), 0.0);
    for (std::int64_t system = 0; system < snapshot.batch_size; ++system) {
      const std::int64_t begin = snapshot.atom_offsets[static_cast<std::size_t>(system)];
      const std::int64_t end = snapshot.atom_offsets[static_cast<std::size_t>(system + 1)];
      for (std::int64_t atom = begin; atom < end; ++atom) {
        const std::int64_t local = atom - begin;
        positions[static_cast<std::size_t>(3 * atom)] = 8.0 * static_cast<double>(local);
      }
    }
    point_positions.assign(static_cast<std::size_t>(point_coordinate_elements), 0.0);
    point_values.assign(static_cast<std::size_t>(snapshot.total_point_charges), 0.0);
    point_gammas.assign(static_cast<std::size_t>(snapshot.total_point_charges), 1.0);
    periodic_shifts.assign(static_cast<std::size_t>(snapshot.total_atoms), 0.0);
    periodic_response.assign(static_cast<std::size_t>(snapshot.total_charge_response_elements),
                             0.0);
  } catch (const std::bad_alloc&) {
    error = "failed to allocate CUDA topology-only numerical seed";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

bool topology_snapshot_matches(const Gfn2CudaTopologyHostSnapshot& snapshot,
                               const xtbloom_compute_options_t& options,
                               const TopologyKey& key) noexcept {
  return snapshot.atom_offsets == key.atom_offsets &&
         snapshot.atomic_numbers == key.atomic_numbers &&
         snapshot.molecular_charges == key.molecular_charges &&
         snapshot.unpaired_electrons == key.unpaired_electrons &&
         snapshot.spin_channels == key.spin_channels &&
         snapshot.point_charge_offsets == key.point_offsets &&
         snapshot.charge_response_offsets == key.response_offsets &&
         snapshot.periodic_enabled == key.periodic_enabled && options.flags == key.flags &&
         options.max_scc_iterations == key.maximum_iterations &&
         options.charge_tolerance == key.charge_tolerance &&
         options.energy_tolerance == key.energy_tolerance &&
         options.electronic_temperature == key.electronic_temperature &&
         public_scc_mixer(options) == key.scc_mixer &&
         public_scc_mixer_history(options) == key.scc_mixer_history &&
         public_scc_mixer_damping(options) == key.scc_mixer_damping &&
         public_determinism(options) == key.determinism;
}

struct HostPlans {
  TopologyKey key;
  std::uint64_t fingerprint = 0u;
  std::uint64_t plan_token = 0u;
  std::uint64_t geometry_generation = kInitialGeometryGeneration;

  BasisPlan basis;
  IntegralPlan integrals;
  CoordinationPlan coordination;
  RepulsionPlan repulsion;
  H0Plan h0;
  WavefunctionLayout wavefunction_layout;
  ES2Plan es2;
  ES3Plan es3;
  AES2Plan aes2;
  MullikenPlan mulliken;
  EigensolverPlan eigensolver;
  SccMixerPlan mixer;
  D4Plan d4;
  ExternalPointChargePlan external;
  PeriodicEmbeddingPlan periodic;
  SccDriverPlan driver;
  bool d4_enabled = false;
  bool periodic_enabled = false;

  std::vector<double> positions;
  std::vector<double> point_positions;
  std::vector<double> point_values;
  std::vector<double> point_gammas;
  std::vector<double> periodic_shifts;
  std::vector<double> periodic_response;

  std::vector<double> coordination_numbers;
  std::vector<double> overlap;
  std::vector<double> dipole_integrals;
  std::vector<double> quadrupole_integrals;
  std::vector<double> core_hamiltonian;
  std::vector<double> integral_workspace;

  std::vector<double> geometry_pair_data;
  std::vector<std::uint64_t> geometry_generations;

  std::vector<double> es2_matrix;
  std::vector<double> es2_matrix_scratch;
  std::vector<double> es2_shell_scratch;
  std::vector<double> es2_batch_scratch;
  std::vector<double> es2_gradient_scratch;
  ES2Workspace es2_workspace{};
  ES2GeometryCache es2_cache{};

  std::vector<double> aes2_pairs;
  std::vector<double> aes2_pair_scratch;
  std::vector<double> aes2_potential_scratch;
  std::vector<double> aes2_batch_scratch;
  std::vector<double> aes2_gradient_scratch;
  std::vector<double> aes2_coordination_scratch;
  AES2Workspace aes2_workspace{};
  AES2GeometryCache aes2_cache{};

  std::vector<Gfn2D4DeviceElementData> d4_elements;
  std::vector<Gfn2D4DeviceReferenceData> d4_references;
  std::vector<double> d4_coordination;

  std::vector<double> explicit_point_shell_potential;
  PinnedArena wavefunction_storage;
  WavefunctionView wavefunction{};

  [[nodiscard]] std::size_t retained_host_bytes() const noexcept {
    const std::size_t direct_plan_vectors =
        vector_bytes(key.atom_offsets) + vector_bytes(key.atomic_numbers) +
        vector_bytes(key.molecular_charges) + vector_bytes(key.unpaired_electrons) +
        vector_bytes(key.spin_channels) + vector_bytes(key.point_offsets) +
        vector_bytes(key.response_offsets) + vector_bytes(basis.atom_offsets) +
        vector_bytes(basis.batch_shell_offsets) + vector_bytes(basis.batch_orbital_offsets) +
        vector_bytes(basis.batch_cartesian_orbital_offsets) +
        vector_bytes(basis.batch_primitive_offsets) + vector_bytes(basis.atom_shell_offsets) +
        vector_bytes(basis.atom_orbital_offsets) +
        vector_bytes(basis.atom_cartesian_orbital_offsets) +
        vector_bytes(basis.atom_primitive_offsets) + vector_bytes(basis.shell_orbital_offsets) +
        vector_bytes(basis.shell_cartesian_orbital_offsets) +
        vector_bytes(basis.shell_primitive_offsets) + vector_bytes(basis.shell_to_atom) +
        vector_bytes(basis.principal_quantum_numbers) + vector_bytes(basis.angular_momenta) +
        vector_bytes(basis.slater_exponents) + vector_bytes(basis.primitive_exponents) +
        vector_bytes(basis.primitive_coefficients) + vector_bytes(integrals.matrix_offsets) +
        vector_bytes(coordination.atom_offsets) + vector_bytes(coordination.covalent_radius) +
        vector_bytes(repulsion.atom_offsets) + vector_bytes(repulsion.sqrt_alpha) +
        vector_bytes(repulsion.effective_charge) + vector_bytes(repulsion.light_element) +
        vector_bytes(h0.atom_offsets) + vector_bytes(h0.batch_shell_offsets) +
        vector_bytes(h0.batch_orbital_offsets) + vector_bytes(h0.matrix_offsets) +
        vector_bytes(h0.shell_pair_offsets) + vector_bytes(h0.atomic_radii) +
        vector_bytes(h0.shell_levels) + vector_bytes(h0.shell_coordination_scale) +
        vector_bytes(h0.shell_polynomial) + vector_bytes(h0.shell_pair_scale) +
        vector_bytes(es3.batch_shell_offsets) + vector_bytes(es3.shell_gamma3) +
        vector_bytes(external.atom_offsets) + vector_bytes(external.batch_shell_offsets) +
        vector_bytes(external.point_charge_offsets) + vector_bytes(external.shell_to_atom) +
        vector_bytes(external.shell_hardness) + vector_bytes(wavefunction_layout.atom_offsets) +
        vector_bytes(wavefunction_layout.batch_shell_offsets) +
        vector_bytes(wavefunction_layout.batch_orbital_offsets) +
        vector_bytes(wavefunction_layout.atomic_numbers) +
        vector_bytes(wavefunction_layout.molecular_charges) +
        vector_bytes(wavefunction_layout.unpaired_electrons) +
        vector_bytes(wavefunction_layout.spin_channels) +
        vector_bytes(wavefunction_layout.reference_atom_occupations) +
        vector_bytes(wavefunction_layout.reference_shell_occupations) +
        vector_bytes(wavefunction_layout.electron_counts) +
        vector_bytes(wavefunction_layout.alpha_electron_counts) +
        vector_bytes(wavefunction_layout.beta_electron_counts) +
        vector_bytes(wavefunction_layout.coefficients.system_offsets) +
        vector_bytes(wavefunction_layout.eigenvalues.system_offsets) +
        vector_bytes(wavefunction_layout.occupations.system_offsets) +
        vector_bytes(wavefunction_layout.density.system_offsets) +
        vector_bytes(wavefunction_layout.qsh.system_offsets) +
        vector_bytes(wavefunction_layout.qat.system_offsets) +
        vector_bytes(wavefunction_layout.dipole.system_offsets) +
        vector_bytes(wavefunction_layout.quadrupole.system_offsets) +
        vector_bytes(wavefunction_layout.energy_weighted_density.system_offsets);
    const std::size_t model_plan_storage =
        es2.resident_bytes() + aes2.resident_bytes() + mulliken.resident_bytes() +
        eigensolver.resident_bytes() + mixer.resident_bytes() + d4.resident_bytes() +
        periodic.resident_bytes() + driver.resident_bytes();
    const std::size_t numerical_vectors =
        vector_bytes(positions) + vector_bytes(point_positions) + vector_bytes(point_values) +
        vector_bytes(point_gammas) + vector_bytes(periodic_shifts) +
        vector_bytes(periodic_response) + vector_bytes(coordination_numbers) +
        vector_bytes(overlap) + vector_bytes(dipole_integrals) +
        vector_bytes(quadrupole_integrals) + vector_bytes(core_hamiltonian) +
        vector_bytes(integral_workspace) + vector_bytes(geometry_pair_data) +
        vector_bytes(geometry_generations) + vector_bytes(es2_matrix) +
        vector_bytes(es2_matrix_scratch) + vector_bytes(es2_shell_scratch) +
        vector_bytes(es2_batch_scratch) + vector_bytes(es2_gradient_scratch) +
        vector_bytes(aes2_pairs) + vector_bytes(aes2_pair_scratch) +
        vector_bytes(aes2_potential_scratch) + vector_bytes(aes2_batch_scratch) +
        vector_bytes(aes2_gradient_scratch) + vector_bytes(aes2_coordination_scratch) +
        vector_bytes(d4_elements) + vector_bytes(d4_references) + vector_bytes(d4_coordination) +
        vector_bytes(explicit_point_shell_potential);
    return direct_plan_vectors + model_plan_storage + numerical_vectors +
           wavefunction_storage.bytes();
  }

  xtbloom_status_t build(TopologyKey&& new_key, std::vector<double>&& new_positions,
                         std::vector<double>&& new_point_positions,
                         std::vector<double>&& new_point_values,
                         std::vector<double>&& new_point_gammas,
                         std::vector<double>&& new_periodic_shifts,
                         std::vector<double>&& new_periodic_response, std::uint64_t token,
                         std::string& error) {
    key = std::move(new_key);
    positions = std::move(new_positions);
    point_positions = std::move(new_point_positions);
    point_values = std::move(new_point_values);
    point_gammas = std::move(new_point_gammas);
    periodic_shifts = std::move(new_periodic_shifts);
    periodic_response = std::move(new_periodic_response);
    fingerprint = key.fingerprint();
    plan_token = token;

    const std::int64_t batch = static_cast<std::int64_t>(key.molecular_charges.size());
    const std::int64_t atoms = static_cast<std::int64_t>(key.atomic_numbers.size());
    const std::int64_t points = static_cast<std::int64_t>(point_values.size());
    xtbloom_status_t status = make_basis_plan(batch, atoms, key.atom_offsets.data(),
                                              key.atomic_numbers.data(), basis, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_integral_plan(basis, integrals, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_coordination_plan(batch, atoms, key.atom_offsets.data(),
                                    key.atomic_numbers.data(), coordination, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_repulsion_plan(batch, atoms, key.atom_offsets.data(), key.atomic_numbers.data(),
                                 repulsion, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_h0_plan(basis, integrals, key.atomic_numbers.data(), h0, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_wavefunction_layout(basis, key.atomic_numbers.data(),
                                      key.molecular_charges.data(), key.unpaired_electrons.data(),
                                      key.spin_channels.data(), wavefunction_layout, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_es2_plan(basis, key.atomic_numbers.data(), es2, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_es3_plan(basis, key.atomic_numbers.data(), es3, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_aes2_plan(basis, key.atomic_numbers.data(), aes2, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_mulliken_plan(basis, integrals, wavefunction_layout, mulliken, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_eigensolver_plan(wavefunction_layout, eigensolver, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_scc_mixer_plan(wavefunction_layout, key.scc_mixer_history, key.scc_mixer_damping,
                                 key.charge_tolerance, key.charge_tolerance, mixer, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    d4_enabled = false;
    for (std::int64_t system = 0; system < batch; ++system) {
      d4_enabled = d4_enabled || key.atom_offsets[static_cast<std::size_t>(system + 1)] -
                                         key.atom_offsets[static_cast<std::size_t>(system)] >
                                     1;
    }
    if (d4_enabled) {
      status =
          make_d4_plan(batch, atoms, key.atom_offsets.data(), key.atomic_numbers.data(), d4, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
    }
    status = make_external_point_charge_plan(basis, key.atomic_numbers.data(), points,
                                             points == 0 ? nullptr : key.point_offsets.data(),
                                             external, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    periodic_enabled = key.periodic_enabled;
    if (periodic_enabled) {
      status = make_periodic_embedding_plan(batch, atoms, key.atom_offsets.data(), periodic, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
    }
    status =
        make_scc_driver_plan(wavefunction_layout, mulliken, es2, es3, aes2, eigensolver, mixer,
                             d4_enabled ? &d4 : nullptr, periodic_enabled ? &periodic : nullptr,
                             static_cast<std::uint64_t>(key.maximum_iterations),
                             key.electronic_temperature, key.energy_tolerance, driver, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    const std::size_t atom_count = static_cast<std::size_t>(atoms);
    const std::size_t shells = static_cast<std::size_t>(basis.total_shells);
    const std::size_t matrices = static_cast<std::size_t>(integrals.total_matrix_elements);
    const std::size_t integral_doubles =
        (integrals.workspace_size_bytes + sizeof(double) - 1u) / sizeof(double);
    coordination_numbers.resize(atom_count);
    overlap.resize(matrices);
    dipole_integrals.resize(3u * matrices);
    quadrupole_integrals.resize(6u * matrices);
    core_hamiltonian.resize(matrices);
    integral_workspace.resize(std::max<std::size_t>(integral_doubles, 1u));

    status = evaluate_coordination_cpu(coordination, positions.data(), coordination_numbers.data(),
                                       error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = evaluate_overlap_cpu(basis, integrals, positions.data(), overlap.data(),
                                  integral_workspace.data(),
                                  integral_workspace.size() * sizeof(double), error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = evaluate_multipole_cpu(basis, integrals, positions.data(), dipole_integrals.data(),
                                    quadrupole_integrals.data(), integral_workspace.data(),
                                    integral_workspace.size() * sizeof(double), error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = evaluate_h0_cpu(basis, integrals, h0, positions.data(), coordination_numbers.data(),
                             overlap.data(), core_hamiltonian.data(), error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    geometry_pair_data.assign(static_cast<std::size_t>(aes2.total_pairs()) *
                                  static_cast<std::size_t>(kGfn2GeometryPairDataElements),
                              0.0);
    geometry_generations.assign(static_cast<std::size_t>(batch), geometry_generation);

    es2_matrix.resize(static_cast<std::size_t>(es2.total_matrix_elements()));
    es2_matrix_scratch.resize(es2_matrix.size());
    es2_shell_scratch.resize(shells);
    es2_batch_scratch.resize(static_cast<std::size_t>(batch));
    es2_gradient_scratch.resize(3u * atom_count);
    es2_workspace = {es2_matrix_scratch.data(),   es2.total_matrix_elements(),
                     es2_shell_scratch.data(),    es2.total_shells(),
                     es2_batch_scratch.data(),    batch,
                     es2_gradient_scratch.data(), atoms * 3};
    status =
        update_es2_geometry_cache_cpu(es2, positions.data(), geometry_generation, es2_matrix.data(),
                                      es2_matrix.size(), es2_workspace, es2_cache, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    aes2_pairs.resize(static_cast<std::size_t>(aes2.pair_data_elements()));
    aes2_pair_scratch.resize(aes2_pairs.size());
    aes2_potential_scratch.resize(static_cast<std::size_t>(aes2.potential_scratch_elements()));
    aes2_batch_scratch.resize(static_cast<std::size_t>(batch));
    aes2_gradient_scratch.resize(3u * atom_count);
    aes2_coordination_scratch.resize(atom_count);
    aes2_workspace = {aes2_pair_scratch.data(),         aes2.pair_data_elements(),
                      aes2_potential_scratch.data(),    aes2.potential_scratch_elements(),
                      aes2_batch_scratch.data(),        batch,
                      aes2_gradient_scratch.data(),     atoms * 3,
                      aes2_coordination_scratch.data(), atoms};
    status = update_aes2_geometry_cache_cpu(aes2, positions.data(), coordination_numbers.data(),
                                            geometry_generation, aes2_pairs.data(),
                                            aes2_pairs.size(), aes2_workspace, aes2_cache, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    if (d4_enabled) {
      /* The setup owner needs only a stable initial CN image. The first CUDA
       * preprocessing transaction replaces these zeros before any D4
       * consumer is eligible; dense CPU D4 pair evaluation is intentionally
       * not part of cache construction anymore. */
      d4_coordination.assign(atom_count, 0.0);
      d4_elements.reserve(parameters::d4::kElements.size());
      for (const auto& element : parameters::d4::kElements) {
        d4_elements.push_back({element.reference_offset, element.reference_count,
                               element.covalent_radius, element.electronegativity,
                               element.effective_charge, element.hardness, element.r4r2});
      }
      d4_references.reserve(parameters::d4::kReferences.size());
      for (const auto& reference : parameters::d4::kReferences) {
        d4_references.push_back(
            {reference.coordination_number, reference.charge, reference.gaussian_count});
      }
    }

    explicit_point_shell_potential.resize(shells);
    status = evaluate_external_point_charge_potential_cpu(
        external, positions.data(), point_positions.empty() ? nullptr : point_positions.data(),
        point_values.empty() ? nullptr : point_values.data(),
        point_gammas.empty() ? nullptr : point_gammas.data(), explicit_point_shell_potential.data(),
        error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    if (wavefunction_storage.allocate(wavefunction_layout.workspace_size_bytes) != cudaSuccess) {
      error = "failed to allocate pinned host wavefunction initialization storage";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    status = bind_wavefunction_view(wavefunction_layout, wavefunction_storage.get(),
                                    wavefunction_storage.bytes(), wavefunction, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = initialize_sad_multipole_state(wavefunction_layout, wavefunction, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    return XTBLOOM_STATUS_SUCCESS;
  }

  Gfn2SccSetupInputSources input_sources() const noexcept {
    Gfn2SccSetupInputSources sources{};
    sources.basis = &basis;
    sources.integrals = &integrals;
    sources.h0_plan = &h0;
    sources.wavefunction = &wavefunction_layout;
    sources.es2 = &es2;
    sources.es3 = &es3;
    sources.aes2 = &aes2;
    sources.mulliken = &mulliken;
    sources.mixer = &mixer;
    sources.driver = &driver;
    sources.eigensolver_options.deterministic_debug =
        key.determinism == XTBLOOM_DETERMINISM_REPRODUCIBLE;
    sources.geometry_generation = geometry_generation;
    sources.atomic_numbers = setup_array(key.atomic_numbers);
    sources.positions = setup_array(positions);
    sources.covalent_radii = setup_array(coordination.covalent_radius);
    sources.h0 = setup_array(core_hamiltonian);
    sources.overlap = setup_array(overlap);
    sources.dipole_integrals = setup_array(dipole_integrals);
    sources.quadrupole_integrals = setup_array(quadrupole_integrals);
    sources.geometry_cache.pair_data = setup_array(geometry_pair_data);
    sources.geometry_cache.coordination_numbers = setup_array(coordination_numbers);
    sources.geometry_cache.system_generations = setup_array(geometry_generations);
    sources.es2_cache.coulomb_matrix = setup_array(es2_matrix);
    sources.aes2_cache.pair_data = setup_array(aes2_pairs);
    if (d4_enabled) {
      sources.d4.plan = &d4;
      sources.d4.elements = setup_array(d4_elements);
      sources.d4.references = setup_array(d4_references);
      sources.d4.reference_c6 = {parameters::d4::kReferenceC6.data(),
                                 static_cast<std::int64_t>(parameters::d4::kReferenceC6.size())};
      sources.d4.coordination_numbers = setup_array(d4_coordination);
    }
    if (!point_values.empty()) {
      sources.point_charges.plan = &external;
      sources.point_charges.positions = setup_array(point_positions);
      sources.point_charges.charges = setup_array(point_values);
      sources.point_charges.hardnesses = setup_array(point_gammas);
      sources.point_charges.shell_potential_cache = setup_array(explicit_point_shell_potential);
    }
    if (periodic_enabled) {
      sources.periodic.plan = &periodic;
      sources.periodic.shifts = setup_array(periodic_shifts);
      sources.periodic.response_matrices = setup_array(periodic_response);
    }
    return sources;
  }

  Gfn2SccIterationHostInitialization initialization() const noexcept {
    Gfn2SccIterationHostInitialization host{};
    host.mode = Gfn2SccIterationInitializationMode::kFresh;
    host.plan_token = plan_token;
    host.initialization_generation = kInitialStateGeneration;
    host.topology = {initialization_array(key.atom_offsets.data(),
                                          static_cast<std::int64_t>(key.atom_offsets.size())),
                     initialization_array(
                         wavefunction_layout.batch_shell_offsets.data(),
                         static_cast<std::int64_t>(wavefunction_layout.batch_shell_offsets.size())),
                     plan_token};
    host.wavefunction.plan_token = plan_token;
    host.wavefunction.population = {
        initialization_array(wavefunction.qsh, wavefunction_layout.qsh.element_count),
        initialization_array(wavefunction.qat, wavefunction_layout.qat.element_count),
        initialization_array(wavefunction.dipole, wavefunction_layout.dipole.element_count),
        initialization_array(wavefunction.quadrupole, wavefunction_layout.quadrupole.element_count),
        plan_token};
    return host;
  }
};

std::string cuda_error_message(const char* operation, cudaError_t status) {
  std::ostringstream message;
  message << operation << " failed: " << cudaGetErrorName(status) << " ("
          << cudaGetErrorString(status) << ')';
  return message.str();
}

std::string setup_error_message(const char* operation, xtbloom_status_t status,
                                std::uint32_t error_code, std::uint32_t field, std::int64_t index) {
  std::ostringstream message;
  message << operation << " failed: status=" << status << " error=" << error_code
          << " field=" << field << " index=" << index;
  return message.str();
}

struct NumericalRefreshState {
  Gfn2PreprocessingDeviceBinding preprocessing{};
  NumericalRefreshDeviceBinding device{};
  /* Refresh reads preprocessing candidate positions; SCC/post/force/terminal
   * consumers read d4_cache with the exact committed-position identity. */
  Gfn2D4PairListDeviceCache d4_refresh_cache{};
  Gfn2D4PairListDeviceCache d4_cache{};
  Gfn2D4DeviceWorkspace d4_workspace{};
  Gfn2ExternalPointChargeDeviceBatch point_batch{};
  Gfn2ExternalPointChargeDeviceCache point_candidate{};
  Gfn2ExternalPointChargeDeviceWorkspace point_workspace{};
  Gfn2SccIterationDeviceActivity point_activity{};

  double* host_positions = nullptr;
  double* host_point_positions = nullptr;
  double* host_point_values = nullptr;
  double* host_point_gammas = nullptr;
  double* host_periodic_shifts = nullptr;
  double* host_periodic_response = nullptr;
  std::uint8_t* host_requested = nullptr;

  /*
   * Host submissions first copy synchronously into this packed, pinned image.
   * The fixed device staging leaves above are then populated asynchronously,
   * so no queued CUDA work retains a caller-owned host pointer after return.
   */
  double* owned_host_positions = nullptr;
  double* owned_host_point_positions = nullptr;
  double* owned_host_point_values = nullptr;
  double* owned_host_point_gammas = nullptr;
  double* owned_host_periodic_shifts = nullptr;
  double* owned_host_periodic_response = nullptr;
  std::uint8_t* owned_host_requested = nullptr;

  bool host_staging_poisoned = false;

  bool ready = false;
};

/*
 * The host-owned pinned snapshot is a single-flight resource.  A host function
 * on a private completion stream releases it after that stream has waited on
 * the owner-stream H2D event.  The submission thread therefore needs only one
 * acquire load; it never queries CUDA progress or waits for either stream.
 *
 * This state is deliberately separate from NumericalRefreshState because the
 * latter is reset as an aggregate while constructing a fixed-topology owner.
 */
struct NumericalHostUploadCompletion {
  std::atomic<bool> pending{false};
};
static_assert(std::atomic<bool>::is_always_lock_free,
              "CUDA host-upload completion must stay allocation-free and nonblocking");

void CUDART_CB release_numerical_host_upload(void* user_data) noexcept {
  auto* const completion = static_cast<NumericalHostUploadCompletion*>(user_data);
  completion->pending.store(false, std::memory_order_release);
}

/*
 * Stable terminal descriptors and their context-owned result arena.  These
 * views are constructed once for a fixed topology and copied by value into
 * allocation-free stream submissions.  The result buffers are deliberately
 * internal: #125 later bridges them to host or device C-API destinations.
 */
struct InferenceState {
  Gfn2GeometryEpochConsumerDevice epoch_consumer{};

  Gfn2TerminalClassicalEnergyDevicePlan terminal_plan{};
  Gfn2TerminalClassicalEnergyDeviceActivity terminal_activity{};
  Gfn2TerminalClassicalEnergyDeviceResults terminal_results{};
  Gfn2TerminalClassicalEnergyDeviceWorkspace terminal_workspace{};
  Gfn2TerminalClassicalEnergyDeviceDiagnostics terminal_diagnostics{};

  Gfn2InferencePublicationDevicePlan publication_plan{};
  Gfn2InferencePublicationDeviceInput publication_input{};
  Gfn2InferencePublicationDeviceResults publication_results{};
  Gfn2InferencePublicationDeviceWorkspace publication_workspace{};
  Gfn2InferencePublicationDeviceDiagnostics publication_diagnostics{};

  /* Per-peer generation of the last successfully published SCC checkpoint. */
  std::uint64_t* warm_checkpoint_generations = nullptr;
  /* Device aggregate: nonzero only when every peer published a checkpoint. */
  std::uint32_t* warm_checkpoint_batch_ready = nullptr;
  bool ready = false;
  bool warm_checkpoint_ready = false;
};

/*
 * Fixed-topology public result staging. Every possible HOST route has a stable
 * pinned slice, while one device control record seals aggregate publication
 * before any CUDA caller destination is touched. Per-call route descriptors
 * borrow these addresses but never outlive the synchronous cache transaction.
 */
struct PublicResultState {
  double* energies = nullptr;
  double* qm_forces = nullptr;
  double* atomic_charges = nullptr;
  double* point_forces = nullptr;
  std::int32_t* iterations = nullptr;
  std::uint8_t* converged = nullptr;
  xtbloom_status_t* system_statuses = nullptr;
  Gfn2PublicResultBridgeControl* host_control = nullptr;
  /* Pinned mirror copied under the existing public completion event. */
  std::uint32_t* warm_checkpoint_ready = nullptr;
  /* The request flag and canonical topology arrays share the plan-owned
   * result arena. They are reset/compared on the owner stream before
   * inference, then consumed by the public-result preflight gate. */
  std::uint32_t* request_topology_error = nullptr;
  const std::int64_t* expected_atom_offsets = nullptr;
  const std::int32_t* expected_atomic_numbers = nullptr;
  const double* expected_molecular_charges = nullptr;
  const std::int32_t* expected_unpaired_electrons = nullptr;
  const std::int32_t* expected_spin_channels = nullptr;
  const std::int64_t* expected_point_offsets = nullptr;
  const std::int64_t* expected_response_offsets = nullptr;
  double* staged_lattice_cells = nullptr;
  std::int32_t* staged_periodic_axes = nullptr;
  double* host_lattice_cells = nullptr;
  std::int32_t* host_periodic_axes = nullptr;
  Gfn2PublicResultBridgeDeviceStaging device_staging{};
  Gfn2PublicResultBridgeDeviceDiagnostics diagnostics{};
  std::uint32_t pending_result_flags = 0u;
  bool ready = false;
};

/*
 * The public bridge is intentionally represented as explicit host phases.
 * CUDA submission, completion observation, aggregate acceptance, and host
 * publication have different lifetime and failure boundaries; keeping those
 * boundaries visible lets the synchronous owner preserve today's ordering
 * while a later request owner can drive the same transaction incrementally.
 */
enum class PublicResultTransactionPhase : std::uint32_t {
  kEmpty = 0u,
  kPrepareSubmitted = 1u,
  kPrepareCompletionRecorded = 2u,
  kPrepareCompleted = 3u,
  kPrepareAccepted = 4u,
  kCommitSubmitted = 5u,
  kCommitCompletionRecorded = 6u,
  kCommitCompleted = 7u,
  kPublished = 8u,
};

/* Descriptor image retained across one public result transaction. All
 * referenced storage is owned by Prepared or borrowed from the caller until
 * synchronous return / asynchronous request completion. */
struct PublicResultTransaction {
  Gfn2PublicResultBridgeDevicePlan plan{};
  Gfn2PublicResultBridgeDeviceDestinations destinations{};
  PublicResultTransactionPhase phase = PublicResultTransactionPhase::kEmpty;
};

class RequestExecutionGraphOwner {
 public:
  RequestExecutionGraphOwner() = default;
  ~RequestExecutionGraphOwner() { reset(); }
  RequestExecutionGraphOwner(const RequestExecutionGraphOwner&) = delete;
  RequestExecutionGraphOwner& operator=(const RequestExecutionGraphOwner&) = delete;

  void reset() noexcept {
    if (executable_ != nullptr) (void)cudaGraphExecDestroy(executable_);
    if (graph_ != nullptr) (void)cudaGraphDestroy(graph_);
    executable_ = nullptr;
    graph_ = nullptr;
    condition_ = 0u;
  }

  [[nodiscard]] bool ready() const noexcept { return executable_ != nullptr; }
  [[nodiscard]] cudaGraph_t graph() const noexcept { return graph_; }
  [[nodiscard]] cudaGraphConditionalHandle condition() const noexcept { return condition_; }

  cudaError_t create() noexcept {
    reset();
    cudaError_t status = cudaGraphCreate(&graph_, 0u);
    if (status == cudaSuccess) {
      status =
          cudaGraphConditionalHandleCreate(&condition_, graph_, 0u, cudaGraphCondAssignDefault);
    }
    if (status != cudaSuccess) reset();
    return status;
  }

  cudaError_t add_if_node(cudaGraphNode_t dependency, cudaGraph_t& body) noexcept {
    body = nullptr;
    if (graph_ == nullptr || condition_ == 0u || dependency == nullptr) {
      return cudaErrorInvalidValue;
    }
    cudaGraphNodeParams parameters{};
    parameters.type = cudaGraphNodeTypeConditional;
    parameters.conditional.handle = condition_;
    parameters.conditional.type = cudaGraphCondTypeIf;
    parameters.conditional.size = 1u;
    cudaGraphNode_t node = nullptr;
    const cudaError_t status = cudaGraphAddNode(&node, graph_, &dependency, 1u, &parameters);
    if (status == cudaSuccess) body = parameters.conditional.phGraph_out[0];
    return status;
  }

  cudaError_t instantiate() noexcept {
    if (graph_ == nullptr || executable_ != nullptr) return cudaErrorInvalidValue;
    return cudaGraphInstantiate(&executable_, graph_, 0u);
  }

  cudaError_t launch(cudaStream_t stream) const noexcept {
    return executable_ == nullptr ? cudaErrorInvalidResourceHandle
                                  : cudaGraphLaunch(executable_, stream);
  }

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
  cudaGraphConditionalHandle condition_ = 0u;
};

}  // namespace

struct Gfn2CudaExecutionCache::Impl {
  struct EnergyForceBindings {
    Gfn2EnergyForceExecutionDevicePlan plan{};
    Gfn2EnergyForceExecutionDeviceInput input{};
    Gfn2EnergyForceExecutionDeviceResults results{};
    Gfn2EnergyForceExecutionDeviceIntermediates intermediates{};
    Gfn2EnergyForceExecutionDeviceWorkspace workspace{};
    Gfn2EnergyForceExecutionDeviceDiagnostics diagnostics{};
    StationaryForceProjectionDeviceBinding stationary_projection{};
  };

  struct Prepared {
    explicit Prepared(cudaStream_t owner_stream) noexcept : stream(owner_stream) {}
    ~Prepared() {
      /* Setup owners retain pinned provider images and the SCC initializer
       * retains its immutable device checkpoint. Numerical host-completion
       * functions also retain the address of this Prepared object's atomic
       * state. A failed candidate and a replaced cache therefore wait for the
       * owner and private completion streams before releasing any source image,
       * callback state, or arena referenced by queued work. CUDA does not
       * invoke a host function after a context failure; in that case pending
       * remains conservatively set and both failed synchronization attempts
       * still precede destruction. */
      if (submitted) {
        /* cudaStreamSynchronize(nullptr) waits for the selected legacy default
         * stream only; setup teardown never escalates to a device-wide fence. */
        (void)cudaStreamSynchronize(stream);
      }
      if (numerical_host_completion_stream.valid()) {
        (void)cudaStreamSynchronize(numerical_host_completion_stream.get());
      }
    }

    cudaStream_t stream = nullptr;
    bool submitted = false;
    HostPlans host;
    Gfn2SccSetupTopology topology_owner;
    Gfn2SccSetupInputs inputs_owner;
    Gfn2SccSetupEigensolver eigensolver_owner;
    Gfn2SccIterationInitializer initializer;
    DeviceArena topology_arena;
    DeviceArena input_arena;
    DeviceArena iteration_arena;
    DeviceArena eigensolver_setup_arena;
    PinnedArena provider_host_workspace;
    PinnedArena numerical_host_staging_arena;
    CudaStream numerical_host_completion_stream;
    CudaEvent numerical_host_upload_complete;
    CudaEvent numerical_host_release_complete;
    NumericalHostUploadCompletion numerical_host_upload_completion;
    DeviceArena force_immutable_arena;
    DeviceArena force_execution_arena;
    DeviceArena numerical_refresh_arena;
    DeviceArena inference_arena;
    DeviceArena public_result_device_arena;
    PinnedArena public_result_host_arena;
    PinnedArena candidate_validation_arena;
    CudaEvent public_result_completion_event;
    Gfn2RaggedTopologyView device_topology{};
    Gfn2WavefunctionLayoutView device_wavefunction{};
    Gfn2SccIterationDevicePlan plan_seed{};
    Gfn2SccIterationDeviceInput input_seed{};
    Gfn2SccIterationArenaRequirements iteration_requirements{};
    Gfn2SccIterationDeviceState state_seed{};
    Gfn2SccIterationDeviceWorkspace workspace_seed{};
    Gfn2SccIterationReportStorage report_storage{};
    Gfn2SccSetupEigensolverBinding eigensolver_binding{};
    Gfn2SccIterationInitializationReady ready{};
    Gfn2SccIterationBinding scc_binding{};
    /* Built once after setup validation and replayed by fresh/warm inference.
     * The owner retains a bounded fallback when conditional capture is not
     * supported by the selected CUDA/provider stack. */
    Gfn2SccLoopCudaGraphOwner scc_loop;
    /* Declared after scc_loop so the outer graph is destroyed first. Its SCC
     * child retains the scc_loop root graph by reference for the Prepared
     * lifetime. */
    RequestExecutionGraphOwner request_execution_graph;
    const std::int32_t* atomic_numbers = nullptr;
    EnergyForceBindings energy_force{};
    NumericalRefreshState numerical{};
    InferenceState inference{};
    PublicResultState public_result{};
    bool energy_force_smoke_ready = false;
  };

  struct ActiveRequest {
    std::uint64_t id = 0u;
    xtbloom_compute_options_t options{};
    xtbloom_batch_result_t result{};
    PublicResultTransaction transaction{};
    bool completion_ready = false;
    xtbloom_status_t completion_status = XTBLOOM_STATUS_SUCCESS;
    std::uint32_t result_flags = 0u;
    std::string completion_error;
    xtbloom_status_t deferred_status = XTBLOOM_STATUS_SUCCESS;
    std::string deferred_error;
    bool host_upload_release_ordered = false;
    /* A post-Graph host-side submission failure is already an accepted CUDA
     * request because the Graph may have consumed persistent state. If the
     * first exact-stream settlement attempt itself fails, keep the request
     * PENDING and retain this cache as its completion owner until a later
     * wait/destroy proves both the owner and private callback streams idle. */
    bool settlement_only_pending = false;
    /* Context enqueue builds a new Prepared owner transactionally and does not
     * publish it into the cache until caller-output commit is accepted. If a
     * post-launch failure cannot be settled immediately, retain that candidate
     * here so queued GPU work cannot outlive its arenas. The matching topology
     * staging candidate remains unpublished and is aborted only after both
     * execution streams are known idle. */
    std::unique_ptr<Prepared> pending_prepared;
    bool pending_topology_candidate = false;
  };

  Impl(std::int32_t selected_device, void* selected_stream) noexcept
      : device_id(selected_device),
        stream(reinterpret_cast<cudaStream_t>(selected_stream)),
        topology_staging(selected_device, selected_stream) {}

  ~Impl() {
    std::lock_guard<std::mutex> lock(mutex);
    /* CUDA current-device state is thread-local. Context teardown can run on a
     * different thread or after the caller selected another device. Select the
     * owner only for teardown and restore the caller's selection afterwards;
     * destructors cannot report a restoration failure, so both operations are
     * necessarily best effort. */
    int caller_device = -1;
    const bool caller_device_known = cudaGetDevice(&caller_device) == cudaSuccess;
    bool owner_device_selected = caller_device_known && caller_device == device_id;
    bool restore_caller_device = false;
    if (!owner_device_selected && cudaSetDevice(device_id) == cudaSuccess) {
      owner_device_selected = true;
      restore_caller_device = caller_device_known && caller_device != device_id;
    }

    /* One owner-stream fence settles both handle-owned work and the native
     * lattice D2H. A failed or wrong-device fence cannot authorize freeing the
     * pinned destination: detach it below so member destruction cannot issue
     * cudaFreeHost while DMA may still reference the image. */
    cudaError_t owner_stream_status = cudaSuccess;
    bool owner_stream_settlement_attempted = false;
    if (owner_device_selected && (handles_created || native_lattice_staging_pending)) {
      owner_stream_settlement_attempted = true;
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
      if (native_lattice_staging_pending &&
          consume_execution_test_fault(
              Gfn2CudaExecutionTestFault::kNativeLatticeTeardownSettlement)) {
        g_native_lattice_teardown_faults.fetch_add(1u, std::memory_order_relaxed);
        owner_stream_status = cudaErrorUnknown;
      } else
#endif
      {
        owner_stream_status = cudaStreamSynchronize(stream);
      }
    }
    if (native_lattice_staging_pending) {
      const bool settled = owner_stream_settlement_attempted && owner_stream_status == cudaSuccess;
      if (settled) {
        native_lattice_staging_pending = false;
      } else {
        const std::size_t quarantined_bytes = native_lattice_host_arena.bytes();
        void* const quarantined = native_lattice_host_arena.release_without_free();
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
        if (quarantined != nullptr) {
          g_quarantined_native_lattice_arenas.fetch_add(1u, std::memory_order_relaxed);
          g_quarantined_native_lattice_bytes.fetch_add(
              static_cast<std::uint64_t>(quarantined_bytes), std::memory_order_relaxed);
        }
#else
        (void)quarantined;
        (void)quarantined_bytes;
#endif
      }
    }
    if (prepared != nullptr) {
      prepared.reset();
    }
    if (active_request.pending_prepared != nullptr) {
      active_request.pending_prepared.reset();
    }
    if (active_request.pending_topology_candidate) {
      topology_staging.abort_candidate();
      active_request.pending_topology_candidate = false;
    }
    if (blas != nullptr) (void)cublasDestroy(blas);
    if (solver_jacobi != nullptr) (void)cusolverDnDestroySyevjInfo(solver_jacobi);
    if (solver_parameters != nullptr) (void)cusolverDnDestroyParams(solver_parameters);
    if (solver != nullptr) (void)cusolverDnDestroy(solver);
    if (restore_caller_device) (void)cudaSetDevice(caller_device);
  }

  /* Exception-only safety net for asynchronous admission while mutex is held.
   * Normal status failures use the diagnostic-preserving settlement helpers;
   * this no-throw path exists so an allocation or string exception can never
   * strand queued work after the public Request rolls back its reservation. */
  void settle_active_request_exception_noexcept_locked() noexcept {
    if (active_request.id == 0u) return;

    int caller_device = -1;
    const bool caller_device_known = cudaGetDevice(&caller_device) == cudaSuccess;
    const bool owner_selected = (caller_device_known && caller_device == device_id) ||
                                cudaSetDevice(device_id) == cudaSuccess;
    bool settled = false;
    Prepared* const active_prepared = active_request.pending_prepared != nullptr
                                          ? active_request.pending_prepared.get()
                                          : prepared.get();
    if (owner_selected && active_prepared != nullptr) {
      auto& current = *active_prepared;
      cudaError_t owner_status = cudaSuccess;
      if (current.submitted) {
        owner_status = cudaStreamSynchronize(stream);
        if (owner_status == cudaSuccess) current.submitted = false;
      }
      cudaError_t host_status = cudaSuccess;
      if (current.numerical_host_completion_stream.valid()) {
        host_status = cudaStreamSynchronize(current.numerical_host_completion_stream.get());
        if (host_status == cudaSuccess) {
          current.numerical_host_upload_completion.pending.store(false, std::memory_order_release);
        }
      }
      settled = owner_status == cudaSuccess && host_status == cudaSuccess;
    } else if (owner_selected) {
      settled = true;
    }

    if (caller_device_known && caller_device != device_id) {
      (void)cudaSetDevice(caller_device);
    }
    if (settled) {
      if (active_request.pending_topology_candidate) {
        topology_staging.abort_candidate();
      }
      active_request = {};
    } else {
      request_poisoned = true;
      if (active_prepared != nullptr) active_prepared->numerical.host_staging_poisoned = true;
    }
  }

  xtbloom_status_t ensure_handles(std::string& error) {
    /* cudaSetDevice is intentionally repeated on every prepare. CUDA current
     * device selection is thread-local and is not preserved by this context. */
    cudaError_t cuda_status = cudaSetDevice(device_id);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("cudaSetDevice", cuda_status);
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }
    if (!ensure_cuda_gfn2_parameters(device_id, error)) {
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }
    if (handles_created) return XTBLOOM_STATUS_SUCCESS;

    /* Publish context handles only after the entire construction succeeds.
     * This keeps retries leak-free after a partial provider failure. */
    cusolverDnHandle_t candidate_solver = nullptr;
    cusolverDnParams_t candidate_parameters = nullptr;
    syevjInfo_t candidate_jacobi = nullptr;
    cublasHandle_t candidate_blas = nullptr;
    const auto destroy_candidate = [&]() noexcept {
      if (candidate_blas != nullptr) (void)cublasDestroy(candidate_blas);
      if (candidate_jacobi != nullptr) (void)cusolverDnDestroySyevjInfo(candidate_jacobi);
      if (candidate_parameters != nullptr) (void)cusolverDnDestroyParams(candidate_parameters);
      if (candidate_solver != nullptr) (void)cusolverDnDestroy(candidate_solver);
    };

    cusolverStatus_t solver_status = cusolverDnCreate(&candidate_solver);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
      error = "cusolverDnCreate failed";
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }
    solver_status = cusolverDnCreateParams(&candidate_parameters);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
      destroy_candidate();
      error = "cusolverDnCreateParams failed";
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }
    solver_status = cusolverDnCreateSyevjInfo(&candidate_jacobi);
    if (solver_status == CUSOLVER_STATUS_SUCCESS) {
      solver_status =
          cusolverDnXsyevjSetTolerance(candidate_jacobi, std::numeric_limits<double>::epsilon());
    }
    if (solver_status == CUSOLVER_STATUS_SUCCESS) {
      solver_status = cusolverDnXsyevjSetMaxSweeps(candidate_jacobi, 100);
    }
    if (solver_status == CUSOLVER_STATUS_SUCCESS) {
      solver_status = cusolverDnXsyevjSetSortEig(candidate_jacobi, 1);
    }
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
      destroy_candidate();
      error = "failed to configure the CUDA small-matrix Jacobi eigensolver";
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }
    cublasStatus_t blas_status = cublasCreate(&candidate_blas);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
      destroy_candidate();
      error = "cublasCreate failed";
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }
    solver_status = cusolverDnSetStream(candidate_solver, stream);
    blas_status = cublasSetStream(candidate_blas, stream);
    if (solver_status != CUSOLVER_STATUS_SUCCESS || blas_status != CUBLAS_STATUS_SUCCESS) {
      destroy_candidate();
      error = "failed to bind CUDA linear-algebra handles to the context stream";
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }
    solver = candidate_solver;
    solver_parameters = candidate_parameters;
    solver_jacobi = candidate_jacobi;
    blas = candidate_blas;
    handles_created = true;
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t build_numerical_refresh_binding(Prepared& candidate, std::string& error) {
    const std::int64_t batch = candidate.host.basis.batch_size;
    const std::int64_t atoms = candidate.host.basis.total_atoms;
    const std::int64_t shells = candidate.host.basis.total_shells;
    const std::int64_t orbitals = candidate.host.basis.total_orbitals;
    const std::int64_t primitives = candidate.host.basis.total_primitives;
    const std::int64_t matrices = candidate.host.integrals.total_matrix_elements;
    const std::int64_t points = candidate.host.external.total_point_charges;
    const std::int64_t response =
        static_cast<std::int64_t>(candidate.host.periodic_response.size());
    const std::int64_t geometry_pairs = candidate.plan_seed.geometry_batch.total_pairs;
    const std::int64_t es2_elements = candidate.plan_seed.es2_batch.total_matrix_elements;
    const std::int64_t aes2_pairs = candidate.plan_seed.aes2_batch.total_pairs;
    const std::uint64_t token = candidate.host.plan_token;
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    std::int64_t geometry_pair_elements = 0;
    std::int64_t aes2_elements = 0;
    std::int64_t dipole_elements = 0;
    std::int64_t quadrupole_elements = 0;
    /* Fixed-topology capacities for the optional sparse pair-list gate.  The
     * cell arrays use a per-system trailing slot so adjacent systems never
     * race on the exclusive-end sentinel. */
    std::int64_t maximum_system_atoms = 0;
    for (std::int64_t system = 0; system < batch; ++system) {
      maximum_system_atoms =
          std::max(maximum_system_atoms,
                   candidate.host.basis.atom_offsets[static_cast<std::size_t>(system + 1)] -
                       candidate.host.basis.atom_offsets[static_cast<std::size_t>(system)]);
    }
    /* Per-system sparse/dense dispatch decisions, uploaded once at setup. Each
     * peer independently crosses the measured 40-atom crossover, so a
     * heterogeneous batch no longer applies one strategy to every member. D4
     * still forces the leaf to exist for small dense peers; dense mode then
     * publishes the full triangle into the same committed fixed-capacity
     * transaction used by sparse peers. */
    std::vector<std::int32_t> pairlist_system_modes(static_cast<std::size_t>(batch));
    for (std::int64_t system = 0; system < batch; ++system) {
      const std::int64_t atoms_per_system =
          candidate.host.basis.atom_offsets[static_cast<std::size_t>(system + 1)] -
          candidate.host.basis.atom_offsets[static_cast<std::size_t>(system)];
      pairlist_system_modes[static_cast<std::size_t>(system)] = static_cast<std::int32_t>(
          xtbloom::detail::cuda::gfn2_pairlist_use_sparse_for(atoms_per_system)
              ? xtbloom::detail::cuda::Gfn2PairListMode::kSparse
              : xtbloom::detail::cuda::Gfn2PairListMode::kDense);
    }
    std::int64_t scaled_cells = 0;
    if (!checked_elements(maximum_system_atoms, 8, scaled_cells)) {
      error = "numerical refresh sparse cell capacity overflows int64_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    std::int64_t sparse_cells_per_system = std::max<std::int64_t>(16, scaled_cells);
    std::int64_t sparse_neighbors_per_atom = maximum_system_atoms;
    std::int64_t sparse_pairs_per_system = 0;
    if (!checked_triangle(maximum_system_atoms, sparse_pairs_per_system)) {
      error = "numerical refresh sparse pair capacity overflows int64_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    /* The committed pair-list schema requires a positive fixed slot capacity
     * even when every system is a singleton. Reserve one inert pair slot per
     * peer; builders still publish pair_count=0, so no synthetic pair becomes
     * visible to D4 or coordination consumers. */
    sparse_pairs_per_system = std::max<std::int64_t>(1, sparse_pairs_per_system);
    std::int64_t sparse_cell_capacity = 0;
    std::int64_t sparse_neighbor_capacity = 0;
    std::int64_t sparse_pair_capacity = 0;
    if (!checked_elements(batch, sparse_cells_per_system, sparse_cell_capacity) ||
        !checked_elements(atoms, sparse_neighbors_per_atom, sparse_neighbor_capacity) ||
        !checked_elements(batch, sparse_pairs_per_system, sparse_pair_capacity) ||
        sparse_cell_capacity > std::numeric_limits<std::int64_t>::max() - batch) {
      error = "numerical refresh sparse pair-list capacity overflows int64_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    if (!checked_elements(atoms, 3, coordinates) ||
        !checked_elements(points, 3, point_coordinates) ||
        !checked_elements(geometry_pairs, kGfn2GeometryPairDataElements, geometry_pair_elements) ||
        !checked_elements(aes2_pairs, kGfn2AES2PairDataElements, aes2_elements) ||
        !checked_elements(matrices, 3, dipole_elements) ||
        !checked_elements(matrices, 6, quadrupole_elements)) {
      error = "numerical refresh element count overflows int64_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }

    struct Offsets {
      std::size_t shell_pair_offsets = 0u;
      std::size_t shell_primitive_offsets = 0u;
      std::size_t angular_momenta = 0u;
      std::size_t primitive_exponents = 0u;
      std::size_t primitive_coefficients = 0u;
      std::size_t h0_atomic_radii = 0u;
      std::size_t h0_shell_levels = 0u;
      std::size_t h0_coordination_scale = 0u;
      std::size_t h0_polynomial = 0u;
      std::size_t h0_pair_scale = 0u;

      std::size_t committed_positions = 0u;
      std::size_t committed_point_positions = 0u;
      std::size_t committed_point_values = 0u;
      std::size_t committed_point_gammas = 0u;
      std::size_t committed_periodic_shifts = 0u;
      std::size_t committed_periodic_response = 0u;
      std::size_t candidate_positions = 0u;
      std::size_t candidate_point_positions = 0u;
      std::size_t candidate_point_values = 0u;
      std::size_t candidate_point_gammas = 0u;
      std::size_t candidate_periodic_shifts = 0u;
      std::size_t candidate_periodic_response = 0u;
      std::size_t host_positions = 0u;
      std::size_t host_point_positions = 0u;
      std::size_t host_point_values = 0u;
      std::size_t host_point_gammas = 0u;
      std::size_t host_periodic_shifts = 0u;
      std::size_t host_periodic_response = 0u;
      std::size_t requested = 0u;
      std::size_t host_requested = 0u;
      std::size_t eligible = 0u;
      std::size_t committed_generations = 0u;
      std::size_t refresh_predecessor_generations = 0u;
      std::size_t geometry_epoch = 0u;

      std::size_t output_geometry_pairs = 0u;
      std::size_t output_coordination = 0u;
      std::size_t output_geometry_generations = 0u;
      std::size_t output_overlap = 0u;
      std::size_t output_dipole = 0u;
      std::size_t output_quadrupole = 0u;
      std::size_t output_h0 = 0u;
      std::size_t output_es2 = 0u;
      std::size_t output_aes2 = 0u;
      std::size_t output_operator_generations = 0u;
      std::size_t output_published = 0u;

      std::size_t positions_scratch = 0u;
      std::size_t geometry_candidate_pairs = 0u;
      std::size_t geometry_candidate_coordination = 0u;
      std::size_t geometry_candidate_generations = 0u;
      std::size_t geometry_pair_scratch = 0u;
      std::size_t geometry_coordination_scratch = 0u;
      std::size_t geometry_sequence = 0u;
      std::size_t overlap_candidate = 0u;
      std::size_t dipole_candidate = 0u;
      std::size_t quadrupole_candidate = 0u;
      std::size_t h0_candidate = 0u;
      std::size_t overlap_scratch = 0u;
      std::size_t dipole_scratch = 0u;
      std::size_t quadrupole_scratch = 0u;
      std::size_t h0_scratch = 0u;
      std::size_t integral_sequence = 0u;
      std::size_t es2_candidate = 0u;
      std::size_t es2_scratch = 0u;
      std::size_t aes2_candidate = 0u;
      std::size_t aes2_scratch = 0u;
      std::size_t geometry_system_errors = 0u;
      std::size_t geometry_device_error = 0u;
      std::size_t integral_system_errors = 0u;
      std::size_t integral_device_error = 0u;
      std::size_t es2_device_error = 0u;
      std::size_t aes2_system_errors = 0u;
      std::size_t aes2_device_error = 0u;
      std::size_t preprocessing_stages = 0u;
      std::size_t preprocessing_plan_error = 0u;
      std::size_t sparse_system_errors = 0u;
      std::size_t sparse_device_error = 0u;
      std::size_t sparse_coordination = 0u;
      std::size_t pairlist_pairs = 0u;
      std::size_t pairlist_offsets = 0u;
      std::size_t pairlist_pair_counts = 0u;
      std::size_t pairlist_neighbor_offsets = 0u;
      std::size_t pairlist_neighbor_counts = 0u;
      std::size_t pairlist_neighbors = 0u;
      std::size_t pairlist_generations = 0u;
      std::size_t pairlist_system_modes = 0u;
      std::size_t pairlist_meta = 0u;
      std::size_t pairlist_atom_cells = 0u;
      std::size_t pairlist_cell_counts = 0u;
      std::size_t pairlist_cell_offsets = 0u;
      std::size_t pairlist_cell_fill = 0u;
      std::size_t pairlist_cell_atoms = 0u;
      std::size_t pairlist_neighbor_cursor = 0u;
      std::size_t pairlist_neighbor_scratch = 0u;
      std::size_t pairlist_pair_cursor = 0u;
      std::size_t pairlist_sequence = 0u;

      /* Committed public pair-list view (preprocessing step 4).  Distinct
       * storage from the candidate so a failed peer can never expose a partial
       * candidate slice and consumers read only eligible, current peers. */
      std::size_t committed_pairs = 0u;
      std::size_t committed_pair_offsets = 0u;
      std::size_t committed_pair_counts = 0u;
      std::size_t committed_neighbor_offsets = 0u;
      std::size_t committed_neighbor_counts = 0u;
      std::size_t committed_neighbors = 0u;
      std::size_t committed_pair_generations = 0u;
      std::size_t committed_eligible_mask = 0u;

      std::size_t d4_coordination_scratch = 0u;
      std::size_t d4_system_errors = 0u;
      std::size_t d4_device_error = 0u;
      std::size_t point_candidate_shell = 0u;
      std::size_t point_shell_scratch = 0u;
      std::size_t point_system_errors = 0u;
      std::size_t point_plan_error = 0u;
      std::size_t point_sequence = 0u;
      std::size_t periodic_system_errors = 0u;
      std::size_t periodic_plan_error = 0u;
    } offset;

    struct HostStagingOffsets {
      std::size_t positions = 0u;
      std::size_t point_positions = 0u;
      std::size_t point_values = 0u;
      std::size_t point_gammas = 0u;
      std::size_t periodic_shifts = 0u;
      std::size_t periodic_response = 0u;
      std::size_t requested = 0u;
    } host_offset;

    ArenaLayout layout;
    offset.shell_pair_offsets = layout.append<std::int64_t>(
        static_cast<std::int64_t>(candidate.host.h0.shell_pair_offsets.size()));
    offset.shell_primitive_offsets = layout.append<std::int64_t>(
        static_cast<std::int64_t>(candidate.host.basis.shell_primitive_offsets.size()));
    offset.angular_momenta = layout.append<std::uint8_t>(
        static_cast<std::int64_t>(candidate.host.basis.angular_momenta.size()));
    offset.primitive_exponents = layout.append<double>(primitives);
    offset.primitive_coefficients = layout.append<double>(primitives);
    offset.h0_atomic_radii = layout.append<double>(atoms);
    offset.h0_shell_levels = layout.append<double>(shells);
    offset.h0_coordination_scale = layout.append<double>(shells);
    offset.h0_polynomial = layout.append<double>(shells);
    offset.h0_pair_scale = layout.append<double>(candidate.host.h0.shell_pair_offsets.back());

    const auto append_numerical =
        [&](std::size_t& positions_offset, std::size_t& point_positions_offset,
            std::size_t& point_values_offset, std::size_t& point_gammas_offset,
            std::size_t& shifts_offset, std::size_t& response_offset) {
          positions_offset = layout.append<double>(coordinates);
          point_positions_offset = layout.append<double>(point_coordinates);
          point_values_offset = layout.append<double>(points);
          point_gammas_offset = layout.append<double>(points);
          shifts_offset = layout.append<double>(candidate.host.periodic_enabled ? atoms : 0);
          response_offset = layout.append<double>(candidate.host.periodic_enabled ? response : 0);
        };
    append_numerical(offset.committed_positions, offset.committed_point_positions,
                     offset.committed_point_values, offset.committed_point_gammas,
                     offset.committed_periodic_shifts, offset.committed_periodic_response);
    append_numerical(offset.candidate_positions, offset.candidate_point_positions,
                     offset.candidate_point_values, offset.candidate_point_gammas,
                     offset.candidate_periodic_shifts, offset.candidate_periodic_response);
    append_numerical(offset.host_positions, offset.host_point_positions, offset.host_point_values,
                     offset.host_point_gammas, offset.host_periodic_shifts,
                     offset.host_periodic_response);
    offset.requested = layout.append<std::uint8_t>(batch);
    offset.host_requested = layout.append<std::uint8_t>(batch);
    offset.eligible = layout.append<std::uint8_t>(batch);
    offset.committed_generations = layout.append<std::uint64_t>(batch);
    offset.refresh_predecessor_generations = layout.append<std::uint64_t>(batch);
    offset.geometry_epoch = layout.append<std::uint64_t>(1);

    offset.output_geometry_pairs = layout.append<double>(geometry_pair_elements);
    offset.output_coordination = layout.append<double>(atoms);
    offset.output_geometry_generations = layout.append<std::uint64_t>(batch);
    offset.output_overlap = layout.append<double>(matrices);
    offset.output_dipole = layout.append<double>(dipole_elements);
    offset.output_quadrupole = layout.append<double>(quadrupole_elements);
    offset.output_h0 = layout.append<double>(matrices);
    offset.output_es2 = layout.append<double>(es2_elements);
    offset.output_aes2 = layout.append<double>(aes2_elements);
    offset.output_operator_generations = layout.append<std::uint64_t>(batch);
    offset.output_published = layout.append<std::uint8_t>(batch);

    offset.positions_scratch = layout.append<double>(coordinates);
    offset.geometry_candidate_pairs = layout.append<double>(geometry_pair_elements);
    offset.geometry_candidate_coordination = layout.append<double>(atoms);
    offset.geometry_candidate_generations = layout.append<std::uint64_t>(batch);
    offset.geometry_pair_scratch = layout.append<double>(geometry_pair_elements);
    offset.geometry_coordination_scratch = layout.append<double>(atoms);
    offset.geometry_sequence = layout.append<std::uint32_t>(1);
    offset.overlap_candidate = layout.append<double>(matrices);
    offset.dipole_candidate = layout.append<double>(dipole_elements);
    offset.quadrupole_candidate = layout.append<double>(quadrupole_elements);
    offset.h0_candidate = layout.append<double>(matrices);
    offset.overlap_scratch = layout.append<double>(matrices);
    offset.dipole_scratch = layout.append<double>(dipole_elements);
    offset.quadrupole_scratch = layout.append<double>(quadrupole_elements);
    offset.h0_scratch = layout.append<double>(matrices);
    offset.integral_sequence = layout.append<std::uint32_t>(1);
    offset.es2_candidate = layout.append<double>(es2_elements);
    offset.es2_scratch = layout.append<double>(es2_elements);
    offset.aes2_candidate = layout.append<double>(aes2_elements);
    offset.aes2_scratch = layout.append<double>(aes2_elements);
    offset.geometry_system_errors = layout.append<std::uint32_t>(batch);
    offset.geometry_device_error = layout.append<std::uint32_t>(1);
    offset.integral_system_errors = layout.append<std::uint32_t>(batch);
    offset.integral_device_error = layout.append<std::uint32_t>(1);
    offset.es2_device_error = layout.append<std::uint32_t>(1);
    offset.aes2_system_errors = layout.append<std::uint32_t>(batch);
    offset.aes2_device_error = layout.append<std::uint32_t>(1);
    offset.preprocessing_stages = layout.append<std::uint32_t>(batch);
    offset.preprocessing_plan_error = layout.append<std::uint32_t>(1);

    /* Optional sparse pair-list gate.  The segments are always appended with
     * conservative fixed-topology capacities so a later dispatch toggle does
     * not require an arena rebuild; unused segments cost bytes only. */
    offset.sparse_system_errors = layout.append<std::uint32_t>(batch);
    offset.sparse_device_error = layout.append<std::uint32_t>(1);
    offset.sparse_coordination = layout.append<double>(atoms);
    offset.pairlist_pairs = layout.append<xtbloom::detail::Gfn2AtomPair>(sparse_pair_capacity);
    offset.pairlist_offsets = layout.append<std::int64_t>(batch + 1);
    offset.pairlist_pair_counts = layout.append<std::int64_t>(batch);
    offset.pairlist_neighbor_offsets = layout.append<std::int64_t>(atoms + 1);
    offset.pairlist_neighbor_counts = layout.append<std::int64_t>(atoms);
    offset.pairlist_neighbors = layout.append<std::int64_t>(sparse_neighbor_capacity);
    offset.pairlist_generations = layout.append<std::uint64_t>(batch);
    offset.pairlist_system_modes = layout.append<std::int32_t>(batch);
    offset.pairlist_meta = layout.append<xtbloom::detail::cuda::Gfn2PairListSystemMeta>(batch);
    offset.pairlist_atom_cells = layout.append<std::int64_t>(atoms);
    const std::int64_t sparse_cell_storage = sparse_cell_capacity + batch;
    offset.pairlist_cell_counts = layout.append<std::int64_t>(sparse_cell_storage);
    offset.pairlist_cell_offsets = layout.append<std::int64_t>(sparse_cell_storage);
    offset.pairlist_cell_fill = layout.append<std::int64_t>(sparse_cell_storage);
    offset.pairlist_cell_atoms = layout.append<std::int64_t>(atoms);
    offset.pairlist_neighbor_cursor = layout.append<std::int64_t>(atoms);
    offset.pairlist_neighbor_scratch = layout.append<std::int64_t>(sparse_neighbor_capacity);
    offset.pairlist_pair_cursor = layout.append<std::int64_t>(batch);
    offset.pairlist_sequence = layout.append<std::uint32_t>(1);

    /* Committed output pair-list storage: distinct fixed-capacity slices so a
     * failed peer's slice is never exposed and later peers never shift. */
    offset.committed_pairs = layout.append<xtbloom::detail::Gfn2AtomPair>(sparse_pair_capacity);
    offset.committed_pair_offsets = layout.append<std::int64_t>(batch + 1);
    offset.committed_pair_counts = layout.append<std::int64_t>(batch);
    offset.committed_neighbor_offsets = layout.append<std::int64_t>(atoms + 1);
    offset.committed_neighbor_counts = layout.append<std::int64_t>(atoms);
    offset.committed_neighbors = layout.append<std::int64_t>(sparse_neighbor_capacity);
    offset.committed_pair_generations = layout.append<std::uint64_t>(batch);
    offset.committed_eligible_mask = layout.append<std::uint8_t>(batch);

    if (candidate.host.d4_enabled) {
      offset.d4_coordination_scratch = layout.append<double>(atoms);
      offset.d4_system_errors = layout.append<std::uint32_t>(batch);
      offset.d4_device_error = layout.append<std::uint32_t>(1);
    }
    if (points != 0) {
      offset.point_candidate_shell = layout.append<double>(shells);
      offset.point_shell_scratch = layout.append<double>(shells);
      offset.point_system_errors = layout.append<std::uint32_t>(batch);
      offset.point_plan_error = layout.append<std::uint32_t>(1);
      offset.point_sequence = layout.append<std::uint32_t>(1);
    }
    if (candidate.host.periodic_enabled) {
      offset.periodic_system_errors = layout.append<std::uint32_t>(batch);
      offset.periodic_plan_error = layout.append<std::uint32_t>(1);
    }
    if (!layout.valid()) {
      error = "numerical refresh arena layout overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }

    ArenaLayout host_layout;
    host_offset.positions = host_layout.append<double>(coordinates);
    host_offset.point_positions = host_layout.append<double>(point_coordinates);
    host_offset.point_values = host_layout.append<double>(points);
    host_offset.point_gammas = host_layout.append<double>(points);
    host_offset.periodic_shifts =
        host_layout.append<double>(candidate.host.periodic_enabled ? atoms : 0);
    host_offset.periodic_response =
        host_layout.append<double>(candidate.host.periodic_enabled ? response : 0);
    host_offset.requested = host_layout.append<std::uint8_t>(batch);
    if (!host_layout.valid()) {
      error = "numerical host-staging arena layout overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }

    cudaError_t cuda_status = candidate.numerical_refresh_arena.allocate(layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical refresh arena allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = candidate.numerical_host_staging_arena.allocate(host_layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical pinned host-staging allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = candidate.numerical_host_completion_stream.create(cudaStreamNonBlocking);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical host-completion stream creation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = candidate.numerical_host_upload_complete.create(cudaEventDisableTiming);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical host-upload event creation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = candidate.numerical_host_release_complete.create(cudaEventDisableTiming);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical host-release event creation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    void* const arena = candidate.numerical_refresh_arena.get();
    cuda_status = cudaMemsetAsync(arena, 0, candidate.numerical_refresh_arena.bytes(), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical refresh arena initialization", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    candidate.submitted = true;

    const auto upload = [&](std::size_t destination, const void* source, std::size_t bytes) {
      if (cuda_status == cudaSuccess && bytes != 0u) {
        cuda_status = cudaMemcpyAsync(static_cast<std::byte*>(arena) + destination, source, bytes,
                                      cudaMemcpyHostToDevice, stream);
      }
    };
    const auto upload_vector = [&](std::size_t destination, const auto& values) {
      using Value = typename std::decay_t<decltype(values)>::value_type;
      upload(destination, values.data(), values.size() * sizeof(Value));
    };
    upload_vector(offset.shell_pair_offsets, candidate.host.h0.shell_pair_offsets);
    upload_vector(offset.shell_primitive_offsets, candidate.host.basis.shell_primitive_offsets);
    upload_vector(offset.angular_momenta, candidate.host.basis.angular_momenta);
    upload_vector(offset.primitive_exponents, candidate.host.basis.primitive_exponents);
    upload_vector(offset.primitive_coefficients, candidate.host.basis.primitive_coefficients);
    upload_vector(offset.h0_atomic_radii, candidate.host.h0.atomic_radii);
    upload_vector(offset.h0_shell_levels, candidate.host.h0.shell_levels);
    upload_vector(offset.h0_coordination_scale, candidate.host.h0.shell_coordination_scale);
    upload_vector(offset.h0_polynomial, candidate.host.h0.shell_polynomial);
    upload_vector(offset.h0_pair_scale, candidate.host.h0.shell_pair_scale);
    upload_vector(offset.pairlist_system_modes, pairlist_system_modes);
    upload_vector(offset.committed_positions, candidate.host.positions);
    upload_vector(offset.committed_point_positions, candidate.host.point_positions);
    upload_vector(offset.committed_point_values, candidate.host.point_values);
    upload_vector(offset.committed_point_gammas, candidate.host.point_gammas);
    upload_vector(offset.committed_periodic_shifts, candidate.host.periodic_shifts);
    upload_vector(offset.committed_periodic_response, candidate.host.periodic_response);
    /* Refresh-owned committed caches start unpublished. prepare_host performs
     * the first real numerical transaction from the caller's geometry, while
     * topology-only plans defer that transaction to their first compute. This
     * prevents epoch 1 from advertising zero-initialized pair-list/D4 leaves. */
    std::vector<std::uint64_t> initial_generations(static_cast<std::size_t>(batch), 0u);
    upload_vector(offset.committed_generations, initial_generations);
    constexpr std::uint64_t kUnpublishedGeometryEpoch = 0u;
    upload(offset.geometry_epoch, &kUnpublishedGeometryEpoch, sizeof(kUnpublishedGeometryEpoch));
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemsetAsync(arena_pointer<std::uint8_t>(arena, offset.requested), 1,
                                    static_cast<std::size_t>(batch), stream);
    }
    if (cuda_status == cudaSuccess) {
      /* The dedicated eligibility vector begins unpublished and is set only
       * by the final refresh gate. It then survives SCC activity derivation so
       * terminal publication can distinguish a refresh-rejected peer after the
       * SCC ledger becomes inactive through convergence or failure. */
      cuda_status = cudaMemsetAsync(arena_pointer<std::uint8_t>(arena, offset.eligible), 0,
                                    static_cast<std::size_t>(batch), stream);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemsetAsync(
          arena_pointer<std::uint64_t>(arena, offset.refresh_predecessor_generations), 0,
          static_cast<std::size_t>(batch) * sizeof(std::uint64_t), stream);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical refresh setup upload", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    auto& numerical = candidate.numerical;
    numerical = {};
    numerical.host_positions = arena_pointer<double>(arena, offset.host_positions);
    numerical.host_point_positions =
        arena_pointer_if<double>(arena, offset.host_point_positions, point_coordinates);
    numerical.host_point_values = arena_pointer_if<double>(arena, offset.host_point_values, points);
    numerical.host_point_gammas = arena_pointer_if<double>(arena, offset.host_point_gammas, points);
    numerical.host_periodic_shifts = arena_pointer_if<double>(
        arena, offset.host_periodic_shifts, candidate.host.periodic_enabled ? atoms : 0);
    numerical.host_periodic_response = arena_pointer_if<double>(
        arena, offset.host_periodic_response, candidate.host.periodic_enabled ? response : 0);
    numerical.host_requested = arena_pointer<std::uint8_t>(arena, offset.host_requested);
    void* const host_arena = candidate.numerical_host_staging_arena.get();
    numerical.owned_host_positions = arena_pointer<double>(host_arena, host_offset.positions);
    numerical.owned_host_point_positions =
        arena_pointer_if<double>(host_arena, host_offset.point_positions, point_coordinates);
    numerical.owned_host_point_values =
        arena_pointer_if<double>(host_arena, host_offset.point_values, points);
    numerical.owned_host_point_gammas =
        arena_pointer_if<double>(host_arena, host_offset.point_gammas, points);
    numerical.owned_host_periodic_shifts = arena_pointer_if<double>(
        host_arena, host_offset.periodic_shifts, candidate.host.periodic_enabled ? atoms : 0);
    numerical.owned_host_periodic_response = arena_pointer_if<double>(
        host_arena, host_offset.periodic_response, candidate.host.periodic_enabled ? response : 0);
    numerical.owned_host_requested = arena_pointer<std::uint8_t>(host_arena, host_offset.requested);
    auto& binding = numerical.preprocessing;
    binding = {};
    binding.plan.abi_version = kGfn2PreprocessingAbiVersion;
    binding.plan.reserved = 0u;
    binding.plan.plan_token = token;
    binding.plan.geometry = candidate.plan_seed.geometry_batch;
    std::int64_t maximum_system_shells = 0;
    for (std::int64_t system = 0; system < batch; ++system) {
      maximum_system_shells =
          std::max(maximum_system_shells,
                   candidate.host.basis.batch_shell_offsets[static_cast<std::size_t>(system + 1)] -
                       candidate.host.basis.batch_shell_offsets[static_cast<std::size_t>(system)]);
    }
    binding.plan.integrals = {
        batch,
        atoms,
        shells,
        orbitals,
        primitives,
        matrices,
        candidate.host.h0.shell_pair_offsets.back(),
        maximum_system_shells,
        candidate.host.integrals.integral_cutoff,
        token,
        batch + 1,
        batch + 1,
        batch + 1,
        batch + 1,
        static_cast<std::int64_t>(candidate.host.h0.shell_pair_offsets.size()),
        atoms + 1,
        shells + 1,
        static_cast<std::int64_t>(candidate.host.basis.shell_primitive_offsets.size()),
        shells,
        static_cast<std::int64_t>(candidate.host.basis.angular_momenta.size()),
        primitives,
        primitives,
        candidate.device_topology.atom_offsets,
        candidate.device_topology.batch_shell_offsets,
        candidate.device_topology.batch_orbital_offsets,
        candidate.device_topology.matrix_offsets,
        arena_pointer<std::int64_t>(arena, offset.shell_pair_offsets),
        candidate.device_topology.atom_shell_offsets,
        candidate.device_topology.shell_orbital_offsets,
        arena_pointer<std::int64_t>(arena, offset.shell_primitive_offsets),
        candidate.device_topology.shell_to_atom,
        arena_pointer<std::uint8_t>(arena, offset.angular_momenta),
        arena_pointer<double>(arena, offset.primitive_exponents),
        arena_pointer<double>(arena, offset.primitive_coefficients)};
    binding.plan.h0 = {atoms,
                       shells,
                       shells,
                       shells,
                       candidate.host.h0.shell_pair_offsets.back(),
                       token,
                       arena_pointer<double>(arena, offset.h0_atomic_radii),
                       arena_pointer<double>(arena, offset.h0_shell_levels),
                       arena_pointer<double>(arena, offset.h0_coordination_scale),
                       arena_pointer<double>(arena, offset.h0_polynomial),
                       arena_pointer<double>(arena, offset.h0_pair_scale)};
    binding.plan.es2 = candidate.plan_seed.es2_batch;
    binding.plan.aes2 = candidate.plan_seed.aes2_batch;
    binding.input = {arena_pointer<double>(arena, offset.candidate_positions), coordinates, token};
    binding.activity = {arena_pointer<std::uint8_t>(arena, offset.requested), batch,
                        arena_pointer<std::uint8_t>(arena, offset.output_published), batch, token};
    binding.output.geometry = {
        arena_pointer_if<double>(arena, offset.output_geometry_pairs, geometry_pair_elements),
        geometry_pair_elements,
        arena_pointer<double>(arena, offset.output_coordination),
        atoms,
        arena_pointer<std::uint64_t>(arena, offset.output_geometry_generations),
        batch,
        token};
    binding.output.overlap = arena_pointer<double>(arena, offset.output_overlap);
    binding.output.overlap_elements = matrices;
    binding.output.dipole_integrals = arena_pointer<double>(arena, offset.output_dipole);
    binding.output.dipole_elements = dipole_elements;
    binding.output.quadrupole_integrals = arena_pointer<double>(arena, offset.output_quadrupole);
    binding.output.quadrupole_elements = quadrupole_elements;
    binding.output.h0 = arena_pointer<double>(arena, offset.output_h0);
    binding.output.h0_elements = matrices;
    binding.output.es2 = {arena_pointer<double>(arena, offset.output_es2), es2_elements, 0u, token};
    binding.output.aes2 = {arena_pointer_if<double>(arena, offset.output_aes2, aes2_elements),
                           aes2_elements, 0u, token};
    binding.output.operator_generations =
        arena_pointer<std::uint64_t>(arena, offset.output_operator_generations);
    binding.output.generation_elements = batch;
    binding.output.plan_token = token;
    const bool pairlist_enabled =
        candidate.host.d4_enabled ||
        xtbloom::detail::cuda::gfn2_pairlist_use_sparse_for(maximum_system_atoms);
    const double pairlist_builder_cutoff = candidate.host.d4_enabled
                                               ? kD4PairlistBuilderCutoffBohr
                                               : xtbloom::detail::cuda::kDefaultPairlistCutoffBohr;
    if (pairlist_enabled) {
      /* Capacities were provisioned once from fixed topology above. D4 makes
       * the leaf mandatory and widens only the physical builder superset to
       * 50 bohr. This source GFN2 coordination view remains canonical at 25
       * bohr; the D4 binding later projects its 30/50/25-bohr role views. */
      binding.plan.pairlist = {
          static_cast<std::int64_t>(batch),
          static_cast<std::int64_t>(atoms),
          static_cast<std::int64_t>(batch + 1),
          pairlist_builder_cutoff,
          sparse_cells_per_system,
          sparse_neighbors_per_atom,
          sparse_pairs_per_system,
          xtbloom::detail::cuda::Gfn2PairListMode::kSparse,
          token,
          candidate.device_topology.atom_offsets,
          xtbloom::detail::cuda::kGfn2PairListAllowDenseFallback,
          arena_pointer<std::int32_t>(arena, offset.pairlist_system_modes),
          batch,
      };
    }
    binding.diagnostics = {
        arena_pointer<std::uint32_t>(arena, offset.geometry_system_errors),
        batch,
        arena_pointer<std::uint32_t>(arena, offset.geometry_device_error),
        arena_pointer<std::uint32_t>(arena, offset.integral_system_errors),
        batch,
        arena_pointer<std::uint32_t>(arena, offset.integral_device_error),
        arena_pointer<std::uint32_t>(arena, offset.es2_device_error),
        arena_pointer<std::uint32_t>(arena, offset.aes2_system_errors),
        batch,
        arena_pointer<std::uint32_t>(arena, offset.aes2_device_error),
        arena_pointer<std::uint32_t>(arena, offset.preprocessing_stages),
        batch,
        arena_pointer<std::uint32_t>(arena, offset.preprocessing_plan_error),
        pairlist_enabled ? arena_pointer<std::uint32_t>(arena, offset.sparse_system_errors)
                         : nullptr,
        pairlist_enabled ? batch : 0,
        pairlist_enabled ? arena_pointer<std::uint32_t>(arena, offset.sparse_device_error)
                         : nullptr,
        token};
    binding.workspace.positions_scratch = arena_pointer<double>(arena, offset.positions_scratch);
    binding.workspace.position_elements = coordinates;
    binding.workspace.geometry_candidate = {
        arena_pointer_if<double>(arena, offset.geometry_candidate_pairs, geometry_pair_elements),
        geometry_pair_elements,
        arena_pointer<double>(arena, offset.geometry_candidate_coordination),
        atoms,
        arena_pointer<std::uint64_t>(arena, offset.geometry_candidate_generations),
        batch,
        token};
    binding.workspace.geometry = {
        arena_pointer_if<double>(arena, offset.geometry_pair_scratch, geometry_pair_elements),
        geometry_pair_elements,
        arena_pointer<double>(arena, offset.geometry_coordination_scratch),
        atoms,
        nullptr,
        0,
        arena_pointer<std::uint32_t>(arena, offset.geometry_sequence),
        1,
        token};
    binding.workspace.sparse_coordination =
        pairlist_enabled ? arena_pointer<double>(arena, offset.sparse_coordination) : nullptr;
    binding.workspace.sparse_coordination_elements = pairlist_enabled ? atoms : 0;
    binding.workspace.pairlist_candidate = {
        pairlist_enabled
            ? arena_pointer<xtbloom::detail::Gfn2AtomPair>(arena, offset.pairlist_pairs)
            : nullptr,
        pairlist_enabled ? sparse_pair_capacity : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_offsets) : nullptr,
        pairlist_enabled ? batch + 1 : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_pair_counts)
                         : nullptr,
        pairlist_enabled ? batch : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_neighbor_offsets)
                         : nullptr,
        pairlist_enabled ? atoms + 1 : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_neighbor_counts)
                         : nullptr,
        pairlist_enabled ? atoms : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_neighbors) : nullptr,
        pairlist_enabled ? sparse_neighbor_capacity : 0,
        pairlist_enabled ? arena_pointer<std::uint64_t>(arena, offset.pairlist_generations)
                         : nullptr,
        pairlist_enabled ? batch : 0,
        pairlist_enabled ? token : 0u};
    binding.workspace.pairlist = {
        pairlist_enabled ? arena_pointer<xtbloom::detail::cuda::Gfn2PairListSystemMeta>(
                               arena, offset.pairlist_meta)
                         : nullptr,
        pairlist_enabled ? batch : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_atom_cells) : nullptr,
        pairlist_enabled ? atoms : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_cell_counts)
                         : nullptr,
        pairlist_enabled ? sparse_cell_capacity + batch : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_cell_offsets)
                         : nullptr,
        pairlist_enabled ? sparse_cell_capacity + batch : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_cell_fill) : nullptr,
        pairlist_enabled ? sparse_cell_capacity + batch : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_cell_atoms) : nullptr,
        pairlist_enabled ? atoms : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_neighbor_cursor)
                         : nullptr,
        pairlist_enabled ? atoms : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_neighbor_scratch)
                         : nullptr,
        pairlist_enabled ? sparse_neighbor_capacity : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_pair_cursor)
                         : nullptr,
        pairlist_enabled ? batch : 0,
        pairlist_enabled ? arena_pointer<std::uint32_t>(arena, offset.pairlist_sequence) : nullptr,
        pairlist_enabled ? 1 : 0,
        pairlist_enabled ? token : 0u};
    /* Committed pair-list consumer view published through the final per-system
     * gate (preprocessing step 4).  Uses dedicated output storage so a failed
     * peer never exposes a partial candidate slice; only eligible peers with a
     * current committed generation are consumable.  A disabled leaf keeps the
     * canonical empty viewed with a zero token. */
    binding.output.pairlist =
        pairlist_enabled
            ? xtbloom::detail::
                  Gfn2PairListConsumerView{xtbloom::detail::Gfn2PlanMemorySpace::kCudaDevice,
                                           xtbloom::detail::Gfn2PairListState::kCommitted,
                                           xtbloom::detail::Gfn2PairListRole::kCoordination,
                                           xtbloom::detail::Gfn2PairMapKind::kExplicit,
                                           token,
                                           xtbloom::detail::cuda::kDefaultPairlistCutoffBohr,
                                           pairlist_builder_cutoff,
                                           static_cast<std::int64_t>(batch),
                                           static_cast<std::int64_t>(atoms),
                                           sparse_pairs_per_system,
                                           sparse_neighbors_per_atom,
                                           batch + 1,
                                           atoms + 1,
                                           sparse_pair_capacity,
                                           sparse_neighbor_capacity,
                                           arena_pointer<std::int64_t>(
                                               arena, offset.committed_pair_offsets),
                                           arena_pointer<xtbloom::detail::Gfn2AtomPair>(
                                               arena, offset.committed_pairs),
                                           static_cast<std::int64_t>(batch),
                                           static_cast<std::int64_t>(atoms),
                                           arena_pointer<std::int64_t>(
                                               arena, offset.committed_pair_counts),
                                           arena_pointer<std::int64_t>(
                                               arena, offset.committed_neighbor_counts),
                                           arena_pointer<std::int64_t>(
                                               arena, offset.committed_neighbor_offsets),
                                           arena_pointer<std::int64_t>(arena,
                                                                       offset.committed_neighbors),
                                           static_cast<std::int64_t>(batch),
                                           static_cast<std::int64_t>(batch),
                                           0,
                                           arena_pointer<std::uint64_t>(
                                               arena, offset.committed_pair_generations),
                                           arena_pointer<std::uint8_t>(
                                               arena, offset.committed_eligible_mask),
                                           nullptr}
            : xtbloom::detail::Gfn2PairListConsumerView{};
    binding.workspace.overlap_candidate = arena_pointer<double>(arena, offset.overlap_candidate);
    binding.workspace.overlap_elements = matrices;
    binding.workspace.dipole_candidate = arena_pointer<double>(arena, offset.dipole_candidate);
    binding.workspace.dipole_elements = dipole_elements;
    binding.workspace.quadrupole_candidate =
        arena_pointer<double>(arena, offset.quadrupole_candidate);
    binding.workspace.quadrupole_elements = quadrupole_elements;
    binding.workspace.h0_candidate = arena_pointer<double>(arena, offset.h0_candidate);
    binding.workspace.h0_elements = matrices;
    binding.workspace.integrals = {arena_pointer<double>(arena, offset.overlap_scratch),
                                   matrices,
                                   arena_pointer<double>(arena, offset.dipole_scratch),
                                   dipole_elements,
                                   arena_pointer<double>(arena, offset.quadrupole_scratch),
                                   quadrupole_elements,
                                   arena_pointer<double>(arena, offset.h0_scratch),
                                   matrices,
                                   arena_pointer<std::uint32_t>(arena, offset.integral_sequence),
                                   1,
                                   token};
    binding.workspace.es2_candidate = {arena_pointer<double>(arena, offset.es2_candidate),
                                       es2_elements, 0u, token};
    binding.workspace.es2 = {arena_pointer<double>(arena, offset.es2_scratch),
                             es2_elements,
                             nullptr,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             0};
    binding.workspace.aes2_candidate = {
        arena_pointer_if<double>(arena, offset.aes2_candidate, aes2_elements), aes2_elements, 0u,
        token};
    binding.workspace.aes2 = {arena_pointer_if<double>(arena, offset.aes2_scratch, aes2_elements),
                              aes2_elements,
                              nullptr,
                              0,
                              nullptr,
                              0,
                              nullptr,
                              0,
                              nullptr,
                              0,
                              nullptr,
                              0};
    binding.workspace.plan_token = token;
    binding.geometry_epoch = {arena_pointer<std::uint64_t>(arena, offset.geometry_epoch), 1, token};
    binding.plan_token = token;

    const auto seal = seal_gfn2_preprocessing_binding_cuda(binding);
    if (!seal.success()) {
      error = "CUDA runtime preprocessing binding rejected its fixed arena projection";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    auto& device = numerical.device;
    device = {};
    device.batch_size = batch;
    device.total_atoms = atoms;
    device.total_shells = shells;
    device.total_matrices = matrices;
    device.total_point_charges = points;
    device.total_response_elements = response;
    device.geometry_pair_elements = geometry_pair_elements;
    device.es2_elements = es2_elements;
    device.aes2_elements = aes2_elements;
    device.d4_enabled = candidate.host.d4_enabled ? 1u : 0u;
    device.point_enabled = points != 0 ? 1u : 0u;
    device.periodic_enabled = candidate.host.periodic_enabled ? 1u : 0u;
    device.plan_token = token;
    device.atom_offsets = candidate.device_topology.atom_offsets;
    device.shell_offsets = candidate.device_topology.batch_shell_offsets;
    device.matrix_offsets = candidate.device_topology.matrix_offsets;
    device.geometry_pair_offsets = candidate.plan_seed.geometry_batch.pair_offsets;
    device.es2_offsets = candidate.plan_seed.es2_batch.matrix_offsets;
    device.point_offsets = candidate.plan_seed.explicit_point_charge_batch.point_charge_offsets;
    device.response_offsets = candidate.plan_seed.periodic_batch.matrix_offsets;
    device.requested = binding.activity.requested_mask;
    device.preprocessing_published = binding.activity.published_mask;
    device.eligible = arena_pointer<std::uint8_t>(arena, offset.eligible);
    device.factor_active = candidate.workspace_seed.ledger.active_mask;
    device.committed_generations =
        arena_pointer<std::uint64_t>(arena, offset.committed_generations);
    device.refresh_predecessor_generations =
        arena_pointer<std::uint64_t>(arena, offset.refresh_predecessor_generations);
    device.candidate_positions = arena_pointer<double>(arena, offset.candidate_positions);
    device.candidate_point_positions =
        arena_pointer_if<double>(arena, offset.candidate_point_positions, point_coordinates);
    device.candidate_point_values =
        arena_pointer_if<double>(arena, offset.candidate_point_values, points);
    device.candidate_point_gammas =
        arena_pointer_if<double>(arena, offset.candidate_point_gammas, points);
    device.candidate_periodic_shifts = arena_pointer_if<double>(
        arena, offset.candidate_periodic_shifts, candidate.host.periodic_enabled ? atoms : 0);
    device.candidate_periodic_response = arena_pointer_if<double>(
        arena, offset.candidate_periodic_response, candidate.host.periodic_enabled ? response : 0);
    device.committed_positions = arena_pointer<double>(arena, offset.committed_positions);
    device.committed_point_positions =
        arena_pointer_if<double>(arena, offset.committed_point_positions, point_coordinates);
    device.committed_point_values =
        arena_pointer_if<double>(arena, offset.committed_point_values, points);
    device.committed_point_gammas =
        arena_pointer_if<double>(arena, offset.committed_point_gammas, points);
    device.committed_periodic_shifts = arena_pointer_if<double>(
        arena, offset.committed_periodic_shifts, candidate.host.periodic_enabled ? atoms : 0);
    device.committed_periodic_response = arena_pointer_if<double>(
        arena, offset.committed_periodic_response, candidate.host.periodic_enabled ? response : 0);
    device.candidate_geometry_pairs = binding.output.geometry.pair_data;
    device.candidate_coordination = binding.output.geometry.coordination_numbers;
    device.candidate_overlap = binding.output.overlap;
    device.candidate_dipole = binding.output.dipole_integrals;
    device.candidate_quadrupole = binding.output.quadrupole_integrals;
    device.candidate_h0 = binding.output.h0;
    device.candidate_es2 = binding.output.es2.coulomb_matrix;
    device.candidate_aes2 = binding.output.aes2.pair_data;
    device.public_geometry_pairs =
        const_cast<double*>(candidate.plan_seed.geometry_cache.pair_data);
    device.public_coordination =
        const_cast<double*>(candidate.plan_seed.geometry_cache.coordination_numbers);
    device.public_overlap = const_cast<double*>(candidate.input_seed.hamiltonian.overlap);
    device.public_dipole = const_cast<double*>(candidate.input_seed.hamiltonian.dipole_integrals);
    device.public_quadrupole =
        const_cast<double*>(candidate.input_seed.hamiltonian.quadrupole_integrals);
    device.public_h0 = const_cast<double*>(candidate.input_seed.hamiltonian.h0);
    device.public_es2 = candidate.plan_seed.es2_cache.coulomb_matrix;
    device.public_aes2 = candidate.plan_seed.aes2_cache.pair_data;
    device.preprocessing_plan_error = binding.diagnostics.plan_error;
    device.geometry_epoch = binding.geometry_epoch.value;

    if (candidate.host.d4_enabled) {
      numerical.d4_workspace.coordination_scratch =
          arena_pointer<double>(arena, offset.d4_coordination_scratch);
      numerical.d4_workspace.coordination_scratch_elements = atoms;
      numerical.d4_workspace.system_errors =
          arena_pointer<std::uint32_t>(arena, offset.d4_system_errors);
      numerical.d4_workspace.system_error_elements = batch;
      device.candidate_d4_coordination = numerical.d4_workspace.coordination_scratch;
      device.public_d4_coordination = candidate.plan_seed.d4_pairlist_cache.coordination_numbers;
      device.d4_system_errors = numerical.d4_workspace.system_errors;
      device.d4_device_error = arena_pointer<std::uint32_t>(arena, offset.d4_device_error);

      Gfn2PairListConsumerView d4_coordination_pairs{};
      Gfn2PairListConsumerView d4_two_body_pairs{};
      Gfn2PairListConsumerView d4_atm_pairs{};
      const auto d4_coordination_projection = project_gfn2_pair_list_role_binding(
          candidate.device_topology, binding.output.pairlist, Gfn2PairListRole::kD4Coordination,
          Gfn2PlanMemorySpace::kCudaDevice, d4_coordination_pairs);
      const auto d4_two_body_projection = project_gfn2_pair_list_role_binding(
          candidate.device_topology, binding.output.pairlist, Gfn2PairListRole::kD4TwoBody,
          Gfn2PlanMemorySpace::kCudaDevice, d4_two_body_pairs);
      const auto d4_atm_projection = project_gfn2_pair_list_role_binding(
          candidate.device_topology, binding.output.pairlist, Gfn2PairListRole::kD4Atm,
          Gfn2PlanMemorySpace::kCudaDevice, d4_atm_pairs);
      if (d4_coordination_projection.error != Gfn2PlanSchemaError::kSuccess ||
          d4_two_body_projection.error != Gfn2PlanSchemaError::kSuccess ||
          d4_atm_projection.error != Gfn2PlanSchemaError::kSuccess) {
        error = "CUDA runtime rejected a D4 role projection of the committed pair-list superset";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }

      /* The role views borrow one committed structural transaction. D4 CN is
       * built into scratch after that transaction commits, then the terminal
       * numerical gate alone copies it to the public outlet and advances the
       * common generation/eligibility authority. This keeps Graph replay on a
       * device epoch and prevents a D4-local failure from publishing stale CN. */
      numerical.d4_refresh_cache = {binding.workspace.positions_scratch,
                                    coordinates,
                                    device.public_d4_coordination,
                                    atoms,
                                    device.committed_generations,
                                    batch,
                                    device.eligible,
                                    batch,
                                    d4_coordination_pairs,
                                    d4_two_body_pairs,
                                    d4_atm_pairs,
                                    token};
      numerical.d4_cache = {device.committed_positions,
                            coordinates,
                            device.public_d4_coordination,
                            atoms,
                            device.committed_generations,
                            batch,
                            device.eligible,
                            batch,
                            d4_coordination_pairs,
                            d4_two_body_pairs,
                            d4_atm_pairs,
                            token};
      candidate.plan_seed.d4_pairlist_cache = numerical.d4_cache;
    }
    if (points != 0) {
      numerical.point_batch = candidate.plan_seed.explicit_point_charge_batch;
      numerical.point_batch.qm_positions = device.candidate_positions;
      numerical.point_batch.point_positions = device.candidate_point_positions;
      numerical.point_batch.point_charges = device.candidate_point_values;
      numerical.point_batch.point_hardnesses = device.candidate_point_gammas;
      numerical.point_candidate = {arena_pointer<double>(arena, offset.point_candidate_shell),
                                   shells, candidate.host.geometry_generation, token};
      numerical.point_workspace = {arena_pointer<double>(arena, offset.point_shell_scratch),
                                   shells};
      numerical.point_activity = {binding.activity.published_mask,
                                  arena_pointer<std::uint32_t>(arena, offset.point_sequence), batch,
                                  1, token};
      device.candidate_point_shell = numerical.point_candidate.shell_potentials;
      device.public_point_shell = candidate.plan_seed.explicit_point_charge_cache.shell_potentials;
      device.point_system_errors = arena_pointer<std::uint32_t>(arena, offset.point_system_errors);
      device.point_plan_error = arena_pointer<std::uint32_t>(arena, offset.point_plan_error);
    }
    if (candidate.host.periodic_enabled) {
      device.periodic_system_errors =
          arena_pointer<std::uint32_t>(arena, offset.periodic_system_errors);
      device.periodic_plan_error = arena_pointer<std::uint32_t>(arena, offset.periodic_plan_error);
    }

    /* Canonical runtime numerical pointers replace setup-only copies. */
    candidate.plan_seed.geometry_cache.geometry_generations = device.committed_generations;
    candidate.plan_seed.geometry_cache.generation_elements = batch;
    if (points != 0) {
      candidate.plan_seed.explicit_point_charge_batch.qm_positions = device.committed_positions;
      candidate.plan_seed.explicit_point_charge_batch.point_positions =
          device.committed_point_positions;
      candidate.plan_seed.explicit_point_charge_batch.point_charges = device.committed_point_values;
      candidate.plan_seed.explicit_point_charge_batch.point_hardnesses =
          device.committed_point_gammas;
    }
    if (candidate.host.periodic_enabled) {
      candidate.plan_seed.periodic_batch.shifts = device.committed_periodic_shifts;
      candidate.plan_seed.periodic_batch.response_matrices = device.committed_periodic_response;
    }

    std::vector<Gfn2SccCacheProvenanceBinding> provenance;
    provenance.reserve(
        static_cast<std::size_t>(candidate.plan_seed.provenance.cache_binding_count));
    const auto append_provenance = [&](Gfn2SccStageId stage) {
      Gfn2SccCacheProvenanceBinding record{};
      record.provenance = {Gfn2PlanMemorySpace::kCudaDevice,
                           Gfn2GenerationScope::kPerSystem,
                           token,
                           0u,
                           batch,
                           batch,
                           device.committed_generations};
      record.owner_stage = stage;
      provenance.push_back(record);
    };
    append_provenance(Gfn2SccStageId::kGeometry);
    append_provenance(Gfn2SccStageId::kES2Potential);
    append_provenance(Gfn2SccStageId::kAES2Potential);
    if (candidate.host.d4_enabled) append_provenance(Gfn2SccStageId::kD4Potential);
    if (points != 0) append_provenance(Gfn2SccStageId::kExplicitPointChargePotential);
    if (candidate.host.periodic_enabled) append_provenance(Gfn2SccStageId::kPeriodicPotential);
    if (static_cast<std::int64_t>(provenance.size()) !=
        candidate.plan_seed.provenance.cache_binding_count) {
      error = "runtime refresh provenance count disagrees with the setup owner";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    cuda_status = cudaMemcpyAsync(
        const_cast<Gfn2SccCacheProvenanceBinding*>(candidate.plan_seed.provenance.cache_bindings),
        provenance.data(), provenance.size() * sizeof(provenance.front()), cudaMemcpyHostToDevice,
        stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA runtime provenance publication", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    numerical.ready = true;
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t build_energy_force_bindings(Prepared& candidate, std::string& error) {
    const std::int64_t batch = candidate.host.basis.batch_size;
    const std::int64_t atoms = candidate.host.basis.total_atoms;
    const std::int64_t shells = candidate.host.basis.total_shells;
    const std::int64_t orbitals = candidate.host.basis.total_orbitals;
    const std::int64_t primitives = candidate.host.basis.total_primitives;
    const std::int64_t matrices = candidate.host.integrals.total_matrix_elements;
    const std::int64_t points = candidate.host.external.total_point_charges;
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    std::int64_t dipole_matrix_elements = 0;
    std::int64_t quadrupole_matrix_elements = 0;
    std::int64_t quadrupole_elements = 0;
    std::int64_t d4_weight_elements = 0;
    if (!checked_elements(atoms, 3, coordinates) ||
        !checked_elements(points, 3, point_coordinates) ||
        !checked_elements(matrices, 3, dipole_matrix_elements) ||
        !checked_elements(matrices, 6, quadrupole_matrix_elements) ||
        !checked_elements(atoms, 6, quadrupole_elements) ||
        !checked_elements(atoms, kGfn2D4MaximumReferences, d4_weight_elements)) {
      error = "force descriptor element count overflows int64_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    const bool force_mode =
        (candidate.host.key.flags &
         (static_cast<std::uint32_t>(XTBLOOM_COMPUTE_FORCES) |
          static_cast<std::uint32_t>(XTBLOOM_COMPUTE_POINT_CHARGE_FORCES))) != 0u;
    const bool explicit_points = points != 0;
    const bool d4_enabled = candidate.host.d4_enabled;
    const bool periodic_enabled = candidate.host.periodic_enabled;
    const std::uint64_t token = candidate.host.plan_token;

    struct ImmutableOffsets {
      std::size_t atomic_numbers = 0u;
      std::size_t positions = 0u;
      std::size_t point_offsets = 0u;
      std::size_t shell_pair_offsets = 0u;
      std::size_t shell_primitive_offsets = 0u;
      std::size_t angular_momenta = 0u;
      std::size_t primitive_exponents = 0u;
      std::size_t primitive_coefficients = 0u;
      std::size_t h0_atomic_radii = 0u;
      std::size_t h0_shell_levels = 0u;
      std::size_t h0_coordination_scale = 0u;
      std::size_t h0_polynomial = 0u;
      std::size_t h0_pair_scale = 0u;
    } immutable;

    ArenaLayout immutable_layout;
    if (force_mode) {
      immutable.atomic_numbers = immutable_layout.append<std::int32_t>(atoms);
      immutable.positions = immutable_layout.append<double>(explicit_points ? 0 : coordinates);
      /* A zero-point plan still needs a valid all-zero ragged offset leaf for
       * force composition. Real point-charge plans reuse the SCC input owner. */
      immutable.point_offsets =
          immutable_layout.append<std::int64_t>(explicit_points ? 0 : batch + 1);
      immutable.shell_pair_offsets = immutable_layout.append<std::int64_t>(
          static_cast<std::int64_t>(candidate.host.h0.shell_pair_offsets.size()));
      immutable.shell_primitive_offsets = immutable_layout.append<std::int64_t>(
          static_cast<std::int64_t>(candidate.host.basis.shell_primitive_offsets.size()));
      immutable.angular_momenta = immutable_layout.append<std::uint8_t>(
          static_cast<std::int64_t>(candidate.host.basis.angular_momenta.size()));
      immutable.primitive_exponents = immutable_layout.append<double>(primitives);
      immutable.primitive_coefficients = immutable_layout.append<double>(primitives);
      immutable.h0_atomic_radii = immutable_layout.append<double>(atoms);
      immutable.h0_shell_levels = immutable_layout.append<double>(shells);
      immutable.h0_coordination_scale = immutable_layout.append<double>(shells);
      immutable.h0_polynomial = immutable_layout.append<double>(shells);
      immutable.h0_pair_scale =
          immutable_layout.append<double>(candidate.host.h0.shell_pair_offsets.back());
    }
    if (!immutable_layout.valid()) {
      error = "force immutable arena layout overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cudaError_t cuda_status = candidate.force_immutable_arena.allocate(immutable_layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("force immutable arena allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    const auto upload = [&](std::size_t offset, const void* source, std::size_t bytes) {
      if (bytes == 0u) return cudaSuccess;
      return cudaMemcpyAsync(
          static_cast<std::byte*>(candidate.force_immutable_arena.get()) + offset, source, bytes,
          cudaMemcpyHostToDevice, stream);
    };
    if (force_mode) {
      const auto upload_immutable = [&](std::size_t offset, const auto& source) {
        if (cuda_status != cudaSuccess) return;
        cuda_status = upload(offset, source.data(), source.size() * sizeof(source[0]));
      };
      upload_immutable(immutable.atomic_numbers, candidate.host.key.atomic_numbers);
      if (!explicit_points) {
        upload_immutable(immutable.positions, candidate.host.positions);
        upload_immutable(immutable.point_offsets, candidate.host.external.point_charge_offsets);
      }
      upload_immutable(immutable.shell_pair_offsets, candidate.host.h0.shell_pair_offsets);
      upload_immutable(immutable.shell_primitive_offsets,
                       candidate.host.basis.shell_primitive_offsets);
      upload_immutable(immutable.angular_momenta, candidate.host.basis.angular_momenta);
      upload_immutable(immutable.primitive_exponents, candidate.host.basis.primitive_exponents);
      upload_immutable(immutable.primitive_coefficients,
                       candidate.host.basis.primitive_coefficients);
      upload_immutable(immutable.h0_atomic_radii, candidate.host.h0.atomic_radii);
      upload_immutable(immutable.h0_shell_levels, candidate.host.h0.shell_levels);
      upload_immutable(immutable.h0_coordination_scale, candidate.host.h0.shell_coordination_scale);
      upload_immutable(immutable.h0_polynomial, candidate.host.h0.shell_polynomial);
      upload_immutable(immutable.h0_pair_scale, candidate.host.h0.shell_pair_scale);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("force immutable upload", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    candidate.submitted = candidate.submitted || force_mode;

    enum SequenceIndex : std::int64_t {
      kTotalSequence,
      kPostSequence,
      kPostCompositionSequence,
      kPostBridgeSequence,
      kPostPeriodicSequence,
      kH0Sequence,
      kHamiltonianSequence,
      kIntegralSequence,
      kCoordinationSequence,
      kClassicalSequence,
      kClassicalPrimitiveSequence,
      kExternalSequence,
      kCompositionSequence,
      kSequenceCount,
    };
    enum MaskIndex : std::int64_t {
      kRequestedMask,
      kEnergySuccessMask,
      kPostSuccessMask,
      kElectronicSuccessMask,
      kCoordinationSuccessMask,
      kClassicalSuccessMask,
      kExternalSuccessMask,
      kH0SuccessMask,
      kHamiltonianSuccessMask,
      kClassicalSelectedMask,
      kPostActiveMask,
      kMaskCount,
    };
    enum SystemErrorIndex : std::int64_t {
      kExecutionSystemError,
      kTotalSystemError,
      kPostStageSystemError,
      kPostSystemError,
      kH0SystemError,
      kHamiltonianSystemError,
      kIntegralSystemError,
      kCoordinationSystemError,
      kClassicalSystemError,
      kExternalSystemError,
      kCompositionSystemError,
      kClassicalPrimitiveSystemError,
      kSystemErrorCount,
    };
    enum DeviceErrorIndex : std::int64_t {
      kExecutionDeviceError,
      kTotalDeviceError,
      kPostStageDeviceError,
      kPostDeviceError,
      kH0DeviceError,
      kHamiltonianDeviceError,
      kIntegralDeviceError,
      kCoordinationDeviceError,
      kClassicalDeviceError,
      kExternalDeviceError,
      kCompositionPlanError,
      kClassicalPrimitiveDeviceError,
      kPostAes2PeerError,
      kDeviceErrorCount,
    };

    struct ExecutionOffsets {
      std::size_t repulsion = 0u;
      std::size_t d4_atm = 0u;
      std::size_t staged_energy = 0u;
      std::size_t public_energy = 0u;
      std::size_t sequences = 0u;
      std::size_t plan_failure = 0u;
      std::size_t masks = 0u;
      std::size_t system_errors = 0u;
      std::size_t device_errors = 0u;

      std::size_t public_qm_force = 0u;
      std::size_t public_point_force = 0u;
      std::size_t staged_qm_force = 0u;
      std::size_t staged_point_force = 0u;
      std::size_t stationary_density = 0u;
      std::size_t stationary_weighted_density = 0u;
      std::size_t stationary_spin_density = 0u;
      std::size_t stationary_shell_charges = 0u;
      std::size_t stationary_atomic_charges = 0u;
      std::size_t stationary_atomic_dipoles = 0u;
      std::size_t stationary_atomic_quadrupoles = 0u;
      std::size_t stationary_spin_shell_potential = 0u;
      std::size_t post_complete_shell = 0u;
      std::size_t post_complete_atom = 0u;
      std::size_t post_complete_dipole = 0u;
      std::size_t post_complete_quadrupole = 0u;
      std::size_t post_shell_scalar = 0u;
      std::size_t post_es2_shell = 0u;
      std::size_t post_es3_shell = 0u;
      std::size_t post_aes2_atom = 0u;
      std::size_t post_aes2_dipole = 0u;
      std::size_t post_aes2_quadrupole = 0u;
      std::size_t post_d4_atom = 0u;
      std::size_t post_periodic_atom = 0u;
      std::size_t post_staged_shell = 0u;
      std::size_t post_staged_atom = 0u;
      std::size_t post_staged_dipole = 0u;
      std::size_t post_staged_quadrupole = 0u;
      std::size_t post_staged_shell_scalar = 0u;
      std::size_t overlap_adjoint = 0u;
      std::size_t coordination_adjoint = 0u;
      std::size_t dipole_adjoint = 0u;
      std::size_t quadrupole_adjoint = 0u;
      std::size_t electronic_gradient = 0u;
      std::size_t classical_force = 0u;
      std::size_t explicit_qm_force = 0u;
      std::size_t explicit_point_force = 0u;

      std::size_t post_es2_shell_scratch = 0u;
      std::size_t post_aes2_potential_scratch = 0u;
      std::size_t post_d4_weights = 0u;
      std::size_t post_d4_weight_cn = 0u;
      std::size_t post_d4_weight_charge = 0u;
      std::size_t post_d4_atom_scratch = 0u;
      std::size_t post_d4_coordination = 0u;
      std::size_t post_d4_batch = 0u;
      std::size_t post_d4_gradient = 0u;
      std::size_t post_periodic_potential = 0u;
      std::size_t post_periodic_raw_response = 0u;
      std::size_t post_composition_shell = 0u;
      std::size_t post_composition_atom = 0u;
      std::size_t post_composition_dipole = 0u;
      std::size_t post_composition_quadrupole = 0u;
      std::size_t post_bridge_shell = 0u;
      std::size_t h0_overlap_scratch = 0u;
      std::size_t h0_coordination_scratch = 0u;
      std::size_t h0_gradient_scratch = 0u;
      std::size_t hamiltonian_overlap_scratch = 0u;
      std::size_t hamiltonian_dipole_scratch = 0u;
      std::size_t hamiltonian_quadrupole_scratch = 0u;
      std::size_t integral_gradient_scratch = 0u;
      std::size_t coordination_gradient_scratch = 0u;
      std::size_t classical_gradient_scratch = 0u;
      std::size_t classical_force_scratch = 0u;
      std::size_t sparse_gradient_scratch = 0u;
      std::size_t sparse_sequence = 0u;
      std::size_t classical_coordination_adjoint = 0u;
      std::size_t classical_aes2_gradient = 0u;
      std::size_t classical_aes2_coordination = 0u;
      std::size_t classical_d4_weights = 0u;
      std::size_t classical_d4_weight_cn = 0u;
      std::size_t classical_d4_weight_charge = 0u;
      std::size_t classical_d4_atom_scratch = 0u;
      std::size_t classical_d4_coordination = 0u;
      std::size_t classical_d4_batch = 0u;
      std::size_t classical_d4_gradient = 0u;
      std::size_t classical_geometry_gradient = 0u;
      std::size_t external_qm_scratch = 0u;
      std::size_t external_point_scratch = 0u;
      std::size_t composition_qm_scratch = 0u;
      std::size_t composition_point_scratch = 0u;
    } execution;

    ArenaLayout execution_layout;
    execution.repulsion = execution_layout.append<double>(batch);
    execution.d4_atm = execution_layout.append<double>(d4_enabled ? batch : 0);
    execution.staged_energy = execution_layout.append<double>(batch);
    execution.public_energy = execution_layout.append<double>(batch);
    execution.sequences = execution_layout.append<std::uint32_t>(
        force_mode ? static_cast<std::int64_t>(kSequenceCount) : 1);
    execution.plan_failure = execution_layout.append<std::uint32_t>(1);
    execution.masks = execution_layout.append<std::uint8_t>(force_mode ? kMaskCount * batch : 0);
    execution.system_errors =
        execution_layout.append<std::uint32_t>(force_mode ? kSystemErrorCount * batch : 2 * batch);
    execution.device_errors = execution_layout.append<std::uint32_t>(
        force_mode ? static_cast<std::int64_t>(kDeviceErrorCount) : 2);
    if (force_mode) {
      execution.public_qm_force = execution_layout.append<double>(coordinates);
      execution.public_point_force = execution_layout.append<double>(point_coordinates);
      execution.staged_qm_force = execution_layout.append<double>(coordinates);
      execution.staged_point_force = execution_layout.append<double>(point_coordinates);
      execution.stationary_density = execution_layout.append<double>(matrices);
      execution.stationary_weighted_density = execution_layout.append<double>(matrices);
      execution.stationary_spin_density = execution_layout.append<double>(matrices);
      execution.stationary_shell_charges = execution_layout.append<double>(shells);
      execution.stationary_atomic_charges = execution_layout.append<double>(atoms);
      execution.stationary_atomic_dipoles = execution_layout.append<double>(coordinates);
      execution.stationary_atomic_quadrupoles =
          execution_layout.append<double>(quadrupole_elements);
      execution.stationary_spin_shell_potential = execution_layout.append<double>(shells);
      execution.post_complete_shell = execution_layout.append<double>(shells);
      execution.post_complete_atom = execution_layout.append<double>(atoms);
      execution.post_complete_dipole = execution_layout.append<double>(coordinates);
      execution.post_complete_quadrupole = execution_layout.append<double>(quadrupole_elements);
      execution.post_shell_scalar = execution_layout.append<double>(shells);
      execution.post_es2_shell = execution_layout.append<double>(shells);
      execution.post_es3_shell = execution_layout.append<double>(shells);
      execution.post_aes2_atom = execution_layout.append<double>(atoms);
      execution.post_aes2_dipole = execution_layout.append<double>(coordinates);
      execution.post_aes2_quadrupole = execution_layout.append<double>(quadrupole_elements);
      execution.post_d4_atom = execution_layout.append<double>(d4_enabled ? atoms : 0);
      execution.post_periodic_atom = execution_layout.append<double>(periodic_enabled ? atoms : 0);
      execution.post_staged_shell = execution_layout.append<double>(shells);
      execution.post_staged_atom = execution_layout.append<double>(atoms);
      execution.post_staged_dipole = execution_layout.append<double>(coordinates);
      execution.post_staged_quadrupole = execution_layout.append<double>(quadrupole_elements);
      execution.post_staged_shell_scalar = execution_layout.append<double>(shells);
      execution.overlap_adjoint = execution_layout.append<double>(matrices);
      execution.coordination_adjoint = execution_layout.append<double>(atoms);
      execution.dipole_adjoint = execution_layout.append<double>(dipole_matrix_elements);
      execution.quadrupole_adjoint = execution_layout.append<double>(quadrupole_matrix_elements);
      execution.electronic_gradient = execution_layout.append<double>(coordinates);
      execution.classical_force = execution_layout.append<double>(coordinates);
      execution.explicit_qm_force =
          execution_layout.append<double>(explicit_points ? coordinates : 0);
      execution.explicit_point_force =
          execution_layout.append<double>(explicit_points ? point_coordinates : 0);

      execution.post_es2_shell_scratch = execution_layout.append<double>(shells);
      execution.post_aes2_potential_scratch =
          execution_layout.append<double>(candidate.host.aes2.potential_scratch_elements());
      execution.post_d4_weights =
          execution_layout.append<double>(d4_enabled ? d4_weight_elements : 0);
      execution.post_d4_weight_cn =
          execution_layout.append<double>(d4_enabled ? d4_weight_elements : 0);
      execution.post_d4_weight_charge =
          execution_layout.append<double>(d4_enabled ? d4_weight_elements : 0);
      execution.post_d4_atom_scratch = execution_layout.append<double>(d4_enabled ? atoms : 0);
      execution.post_d4_coordination = execution_layout.append<double>(d4_enabled ? atoms : 0);
      execution.post_d4_batch = execution_layout.append<double>(d4_enabled ? batch : 0);
      execution.post_d4_gradient = execution_layout.append<double>(d4_enabled ? coordinates : 0);
      execution.post_periodic_potential =
          execution_layout.append<double>(periodic_enabled ? atoms : 0);
      execution.post_periodic_raw_response =
          execution_layout.append<double>(periodic_enabled ? atoms : 0);
      execution.post_composition_shell = execution_layout.append<double>(shells);
      execution.post_composition_atom = execution_layout.append<double>(atoms);
      execution.post_composition_dipole = execution_layout.append<double>(coordinates);
      execution.post_composition_quadrupole = execution_layout.append<double>(quadrupole_elements);
      execution.post_bridge_shell = execution_layout.append<double>(shells);
      execution.h0_overlap_scratch = execution_layout.append<double>(matrices);
      execution.h0_coordination_scratch = execution_layout.append<double>(atoms);
      execution.h0_gradient_scratch = execution_layout.append<double>(coordinates);
      execution.hamiltonian_overlap_scratch = execution_layout.append<double>(matrices);
      execution.hamiltonian_dipole_scratch =
          execution_layout.append<double>(dipole_matrix_elements);
      execution.hamiltonian_quadrupole_scratch =
          execution_layout.append<double>(quadrupole_matrix_elements);
      execution.integral_gradient_scratch = execution_layout.append<double>(coordinates);
      execution.coordination_gradient_scratch = execution_layout.append<double>(coordinates);
      execution.classical_gradient_scratch = execution_layout.append<double>(coordinates);
      execution.classical_force_scratch = execution_layout.append<double>(coordinates);
      execution.sparse_gradient_scratch = execution_layout.append<double>(coordinates);
      execution.sparse_sequence = execution_layout.append<std::uint32_t>(1);
      execution.classical_coordination_adjoint = execution_layout.append<double>(atoms);
      execution.classical_aes2_gradient = execution_layout.append<double>(coordinates);
      execution.classical_aes2_coordination = execution_layout.append<double>(atoms);
      execution.classical_d4_weights =
          execution_layout.append<double>(d4_enabled ? d4_weight_elements : 0);
      execution.classical_d4_weight_cn =
          execution_layout.append<double>(d4_enabled ? d4_weight_elements : 0);
      execution.classical_d4_weight_charge =
          execution_layout.append<double>(d4_enabled ? d4_weight_elements : 0);
      execution.classical_d4_atom_scratch = execution_layout.append<double>(d4_enabled ? atoms : 0);
      execution.classical_d4_coordination = execution_layout.append<double>(d4_enabled ? atoms : 0);
      execution.classical_d4_batch = execution_layout.append<double>(d4_enabled ? batch : 0);
      execution.classical_d4_gradient =
          execution_layout.append<double>(d4_enabled ? coordinates : 0);
      execution.classical_geometry_gradient = execution_layout.append<double>(coordinates);
      execution.external_qm_scratch =
          execution_layout.append<double>(explicit_points ? coordinates : 0);
      execution.external_point_scratch =
          execution_layout.append<double>(explicit_points ? point_coordinates : 0);
      execution.composition_qm_scratch = execution_layout.append<double>(coordinates);
      execution.composition_point_scratch = execution_layout.append<double>(point_coordinates);
    }
    if (!execution_layout.valid()) {
      error = "force execution arena layout overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = candidate.force_execution_arena.allocate(execution_layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("force execution arena allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = cudaMemsetAsync(candidate.force_execution_arena.get(), 0,
                                  candidate.force_execution_arena.bytes(), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("force execution arena initialization", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    void* const execution_arena = candidate.force_execution_arena.get();
    void* const immutable_arena = candidate.force_immutable_arena.get();
    candidate.atomic_numbers =
        force_mode ? arena_pointer<std::int32_t>(immutable_arena, immutable.atomic_numbers)
                   : nullptr;
    auto* const repulsion = arena_pointer<double>(execution_arena, execution.repulsion);
    auto* const d4_atm =
        arena_pointer_if<double>(execution_arena, execution.d4_atm, d4_enabled ? batch : 0);
    auto* const staged_energy = arena_pointer<double>(execution_arena, execution.staged_energy);
    auto* const public_energy = arena_pointer<double>(execution_arena, execution.public_energy);
    auto* const sequence_pool = arena_pointer<std::uint32_t>(execution_arena, execution.sequences);
    auto* const plan_failure =
        arena_pointer<std::uint32_t>(execution_arena, execution.plan_failure);
    auto* const system_error_pool =
        arena_pointer<std::uint32_t>(execution_arena, execution.system_errors);
    auto* const device_error_pool =
        arena_pointer<std::uint32_t>(execution_arena, execution.device_errors);
    const auto system_errors = [&](SystemErrorIndex index) {
      return system_error_pool + static_cast<std::int64_t>(index) * batch;
    };
    const auto device_error = [&](DeviceErrorIndex index) {
      return device_error_pool + static_cast<std::int64_t>(index);
    };

    auto& binding = candidate.energy_force;
    binding = {};
    binding.plan.compute_forces = force_mode ? 1u : 0u;
    binding.plan.scc_potential_components = candidate.plan_seed.enabled_components;
    binding.plan.scc_energy_components = candidate.plan_seed.enabled_components;
    binding.plan.plan_token = token;
    const std::uint32_t total_components =
        d4_enabled ? static_cast<std::uint32_t>(Gfn2TotalEnergyComponent::kD4Atm) : 0u;
    binding.plan.total_energy_batch = {batch, total_components, token};

    binding.input.total_energy = {candidate.state_seed.scc.free_energies,
                                  batch,
                                  repulsion,
                                  batch,
                                  d4_atm,
                                  d4_enabled ? batch : 0,
                                  token};
    binding.input.scc_state = {candidate.state_seed.scc.system_statuses,
                               candidate.state_seed.scc.converged, batch, token};
    binding.input.plan_token = token;
    binding.results.energy = {public_energy, batch, token};
    binding.results.plan_token = token;
    binding.intermediates.energy = {staged_energy, batch, token};
    binding.intermediates.plan_token = token;
    binding.workspace.total_energy = {sequence_pool + kTotalSequence, 1, token};
    binding.workspace.plan_failure = plan_failure;
    binding.workspace.plan_failure_elements = 1;
    binding.workspace.plan_token = token;
    binding.diagnostics.execution_system_errors = system_errors(kExecutionSystemError);
    binding.diagnostics.execution_device_error = device_error(kExecutionDeviceError);
    binding.diagnostics.total_energy_system_errors = system_errors(kTotalSystemError);
    binding.diagnostics.total_energy_device_error = device_error(kTotalDeviceError);
    binding.diagnostics.batch_elements = batch;
    binding.diagnostics.plan_token = token;

    if (force_mode) {
      auto* const mask_pool = arena_pointer<std::uint8_t>(execution_arena, execution.masks);
      const auto mask = [&](MaskIndex index) {
        return mask_pool + static_cast<std::int64_t>(index) * batch;
      };
      cuda_status = cudaMemsetAsync(mask(kRequestedMask), 1,
                                    static_cast<std::size_t>(batch) * sizeof(std::uint8_t), stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("force request-mask initialization", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      candidate.submitted = true;

      const auto* const atomic_numbers = candidate.atomic_numbers;
      /* All force descriptors consume the same runtime-owned committed
       * coordinates, including zero-point batches. This is the address updated
       * transactionally by #122 after overlap factorization succeeds. */
      const auto* const positions = candidate.numerical.device.committed_positions;

      std::int64_t maximum_system_shells = 0;
      for (std::int64_t system = 0; system < batch; ++system) {
        maximum_system_shells = std::max(
            maximum_system_shells,
            candidate.host.basis.batch_shell_offsets[static_cast<std::size_t>(system + 1)] -
                candidate.host.basis.batch_shell_offsets[static_cast<std::size_t>(system)]);
      }
      binding.plan.integral_batch = {
          batch,
          atoms,
          shells,
          orbitals,
          primitives,
          matrices,
          candidate.host.h0.shell_pair_offsets.back(),
          maximum_system_shells,
          candidate.host.integrals.integral_cutoff,
          token,
          batch + 1,
          batch + 1,
          batch + 1,
          batch + 1,
          static_cast<std::int64_t>(candidate.host.h0.shell_pair_offsets.size()),
          atoms + 1,
          shells + 1,
          static_cast<std::int64_t>(candidate.host.basis.shell_primitive_offsets.size()),
          shells,
          static_cast<std::int64_t>(candidate.host.basis.angular_momenta.size()),
          primitives,
          primitives,
          candidate.device_topology.atom_offsets,
          candidate.device_topology.batch_shell_offsets,
          candidate.device_topology.batch_orbital_offsets,
          candidate.device_topology.matrix_offsets,
          arena_pointer<std::int64_t>(immutable_arena, immutable.shell_pair_offsets),
          candidate.device_topology.atom_shell_offsets,
          candidate.device_topology.shell_orbital_offsets,
          arena_pointer<std::int64_t>(immutable_arena, immutable.shell_primitive_offsets),
          candidate.device_topology.shell_to_atom,
          arena_pointer<std::uint8_t>(immutable_arena, immutable.angular_momenta),
          arena_pointer<double>(immutable_arena, immutable.primitive_exponents),
          arena_pointer<double>(immutable_arena, immutable.primitive_coefficients)};
      binding.plan.h0_plan = {
          atoms,
          shells,
          shells,
          shells,
          candidate.host.h0.shell_pair_offsets.back(),
          token,
          arena_pointer<double>(immutable_arena, immutable.h0_atomic_radii),
          arena_pointer<double>(immutable_arena, immutable.h0_shell_levels),
          arena_pointer<double>(immutable_arena, immutable.h0_coordination_scale),
          arena_pointer<double>(immutable_arena, immutable.h0_polynomial),
          arena_pointer<double>(immutable_arena, immutable.h0_pair_scale)};
      binding.plan.hamiltonian_batch = candidate.plan_seed.hamiltonian_batch;
      binding.plan.coordination_batch = candidate.plan_seed.geometry_batch;
      binding.plan.coordination_cache = candidate.plan_seed.geometry_cache;
      binding.plan.geometry_generation = candidate.host.geometry_generation;

      Gfn2ExternalPointChargeDeviceBatch external_batch{};
      if (explicit_points) {
        external_batch = candidate.plan_seed.explicit_point_charge_batch;
      } else {
        const auto* const point_offsets =
            arena_pointer<std::int64_t>(immutable_arena, immutable.point_offsets);
        external_batch = {batch,
                          atoms,
                          shells,
                          0,
                          candidate.device_topology.atom_offsets,
                          candidate.device_topology.batch_shell_offsets,
                          point_offsets,
                          candidate.device_topology.shell_to_atom,
                          nullptr,
                          positions,
                          nullptr,
                          nullptr,
                          nullptr,
                          token};
      }
      binding.plan.external_point_charge_batch = external_batch;
      const Gfn2PeriodicEmbeddingDeviceBatch periodic_batch = candidate.plan_seed.periodic_batch;

      auto& post_plan = binding.plan.post_scc_potential_plan;
      post_plan.enabled_components = candidate.plan_seed.enabled_components;
      post_plan.geometry_generation = candidate.host.geometry_generation;
      post_plan.plan_token = token;
      post_plan.potential_batch = candidate.plan_seed.potential_batch;
      post_plan.scalar_bridge_batch = candidate.plan_seed.scalar_bridge_batch;
      post_plan.es2_batch = candidate.plan_seed.es2_batch;
      post_plan.es2_cache = candidate.plan_seed.es2_cache;
      post_plan.es3_batch = candidate.plan_seed.es3_batch;
      post_plan.aes2_batch = candidate.plan_seed.aes2_batch;
      post_plan.aes2_cache = candidate.plan_seed.aes2_cache;
      post_plan.d4_batch = candidate.plan_seed.d4_batch;
      post_plan.d4_parameters = candidate.plan_seed.d4_parameters;
      post_plan.d4_cache = candidate.numerical.d4_cache;
      post_plan.external_point_charge_batch = external_batch;
      post_plan.external_point_charge_cache = candidate.plan_seed.explicit_point_charge_cache;
      post_plan.periodic_batch = periodic_batch;

      std::uint32_t classical_components =
          static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kRepulsion) |
          static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kES2) |
          static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kAES2);
      if (d4_enabled) {
        classical_components |=
            static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4TwoBody) |
            static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4ATM);
      }
      Gfn2D4DeviceBatch classical_d4_batch = candidate.plan_seed.d4_batch;
      if (d4_enabled) {
        /* The classical-force validator requires the repulsion and nested D4
         * views to share one canonical atomic-number leaf. The SCC setup owner
         * stores an equivalent copy in its input arena, but retaining both
         * addresses here would make the composed force plan cross-projection. */
        classical_d4_batch.atomic_numbers = atomic_numbers;
      }
      binding.plan.classical_plan = {batch,
                                     atoms,
                                     shells,
                                     classical_components,
                                     candidate.host.geometry_generation,
                                     token,
                                     candidate.device_topology.atom_offsets,
                                     atomic_numbers,
                                     candidate.plan_seed.geometry_batch,
                                     candidate.plan_seed.geometry_cache,
                                     candidate.plan_seed.es2_batch,
                                     candidate.plan_seed.es2_cache,
                                     candidate.plan_seed.aes2_batch,
                                     candidate.plan_seed.aes2_cache,
                                     classical_d4_batch,
                                     candidate.plan_seed.d4_parameters,
                                     candidate.numerical.d4_cache};
      std::uint32_t composition_components =
          static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kElectronicGradient) |
          static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kClassicalForce);
      if (explicit_points) {
        composition_components |=
            static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kExplicitPointChargeForce);
      }
      binding.plan.force_composition_batch = {batch,
                                              atoms,
                                              points,
                                              batch + 1,
                                              batch + 1,
                                              candidate.device_topology.atom_offsets,
                                              external_batch.point_charge_offsets,
                                              composition_components,
                                              token};

      /* Step 5 sparse CN VJP leaf: borrow the committed pair-list consumer view
       * and the pair-list dispatch batch from the preprocessing binding.  The
       * final per-system gate already published committed generations and
       * eligibility, so the H0 CN VJP can parity-gate a sparse result against
       * the dense reference.  A disabled leaf keeps zero tokens and disables
       * the differential path. */
      binding.plan.pairlist_committed = candidate.numerical.preprocessing.output.pairlist;
      binding.plan.pairlist_batch = candidate.numerical.preprocessing.plan.pairlist;

      /* Project the committed spin-major state into stable physical arrays.
       * Every terminal force descriptor below consumes these buffers, never
       * SCC iteration scratch or the alpha channel by itself. */
      auto* const stationary_density =
          arena_pointer<double>(execution_arena, execution.stationary_density);
      auto* const stationary_weighted_density =
          arena_pointer<double>(execution_arena, execution.stationary_weighted_density);
      auto* const stationary_spin_density =
          arena_pointer<double>(execution_arena, execution.stationary_spin_density);
      auto* const stationary_shell_charges =
          arena_pointer<double>(execution_arena, execution.stationary_shell_charges);
      auto* const stationary_atomic_charges =
          arena_pointer<double>(execution_arena, execution.stationary_atomic_charges);
      auto* const stationary_atomic_dipoles =
          arena_pointer<double>(execution_arena, execution.stationary_atomic_dipoles);
      auto* const stationary_atomic_quadrupoles =
          arena_pointer<double>(execution_arena, execution.stationary_atomic_quadrupoles);
      auto* const stationary_spin_shell_potential =
          arena_pointer<double>(execution_arena, execution.stationary_spin_shell_potential);
      const Gfn2SccPotentialDeviceTopologyMultipoles physical{stationary_shell_charges,
                                                              shells,
                                                              stationary_atomic_charges,
                                                              atoms,
                                                              stationary_atomic_dipoles,
                                                              coordinates,
                                                              stationary_atomic_quadrupoles,
                                                              quadrupole_elements,
                                                              token};

      auto& projection = binding.stationary_projection;
      projection = {};
      projection.enabled = 1u;
      projection.batch_size = batch;
      projection.total_atoms = atoms;
      projection.total_shells = shells;
      projection.total_matrix_elements = matrices;
      projection.total_spin_atoms = candidate.device_wavefunction.total_spin_atoms;
      projection.total_spin_shells = candidate.device_wavefunction.total_spin_shells;
      projection.total_spin_matrix_elements =
          candidate.device_wavefunction.total_spin_matrix_elements;
      projection.atom_offsets = candidate.device_topology.atom_offsets;
      projection.shell_offsets = candidate.device_topology.batch_shell_offsets;
      projection.matrix_offsets = candidate.device_topology.matrix_offsets;
      projection.atom_shell_offsets = candidate.device_topology.atom_shell_offsets;
      projection.spin_channels = candidate.device_wavefunction.spin_channels;
      projection.spin_atom_offsets = candidate.device_wavefunction.spin_atom_offsets;
      projection.spin_shell_offsets = candidate.device_wavefunction.spin_shell_offsets;
      projection.spin_matrix_offsets = candidate.device_wavefunction.spin_matrix_offsets;
      projection.spin_coupling_offsets = candidate.plan_seed.spin_batch.coupling_offsets;
      projection.spin_coupling_matrices = candidate.plan_seed.spin_batch.coupling_matrices;
      projection.packed_density = candidate.state_seed.density.density;
      projection.packed_energy_weighted_density =
          candidate.state_seed.density.energy_weighted_density;
      projection.packed_shell_charges = candidate.state_seed.raw_population.qsh;
      projection.packed_atomic_charges = candidate.state_seed.raw_population.qat;
      projection.packed_atomic_dipoles = candidate.state_seed.raw_population.dipole;
      projection.packed_atomic_quadrupoles = candidate.state_seed.raw_population.quadrupole;
      projection.total_density = stationary_density;
      projection.total_energy_weighted_density = stationary_weighted_density;
      projection.spin_density = stationary_spin_density;
      projection.shell_charges = stationary_shell_charges;
      projection.atomic_charges = stationary_atomic_charges;
      projection.atomic_dipoles = stationary_atomic_dipoles;
      projection.atomic_quadrupoles = stationary_atomic_quadrupoles;
      projection.spin_shell_potentials = stationary_spin_shell_potential;

      binding.input.force_activity = {mask(kRequestedMask),
                                      candidate.state_seed.scc.system_statuses, batch, token};
      binding.input.post_scc_potential = {
          {mask(kRequestedMask), candidate.state_seed.scc.system_statuses, batch, token},
          physical.shell_charges,
          physical.shell_elements,
          physical.atomic_charges,
          physical.atom_elements,
          physical.atomic_dipoles,
          physical.dipole_elements,
          physical.atomic_quadrupoles,
          physical.quadrupole_elements,
          token};
      binding.input.h0 = {positions,
                          coordinates,
                          candidate.plan_seed.geometry_cache.coordination_numbers,
                          atoms,
                          candidate.input_seed.hamiltonian.overlap,
                          matrices,
                          stationary_density,
                          matrices,
                          stationary_weighted_density,
                          matrices,
                          token};

      auto* const post_complete_shell =
          arena_pointer<double>(execution_arena, execution.post_complete_shell);
      auto* const post_complete_atom =
          arena_pointer<double>(execution_arena, execution.post_complete_atom);
      auto* const post_complete_dipole =
          arena_pointer<double>(execution_arena, execution.post_complete_dipole);
      auto* const post_complete_quadrupole =
          arena_pointer<double>(execution_arena, execution.post_complete_quadrupole);
      auto* const post_shell_scalar =
          arena_pointer<double>(execution_arena, execution.post_shell_scalar);
      binding.input.hamiltonian = {stationary_density,
                                   matrices,
                                   post_shell_scalar,
                                   shells,
                                   post_complete_dipole,
                                   coordinates,
                                   post_complete_quadrupole,
                                   quadrupole_elements,
                                   token};
      binding.input.hamiltonian.spin_density = stationary_spin_density;
      binding.input.hamiltonian.spin_density_elements = matrices;
      binding.input.hamiltonian.spin_shell_scalar_potentials = stationary_spin_shell_potential;
      binding.input.hamiltonian.spin_shell_scalar_elements = shells;
      binding.input.classical = {positions,
                                 coordinates,
                                 candidate.plan_seed.geometry_cache.coordination_numbers,
                                 atoms,
                                 physical.shell_charges,
                                 physical.shell_elements,
                                 physical.atomic_charges,
                                 physical.atom_elements,
                                 physical.atomic_dipoles,
                                 physical.dipole_elements,
                                 physical.atomic_quadrupoles,
                                 physical.quadrupole_elements,
                                 token};
      binding.input.external_shell_charges = physical.shell_charges;
      binding.input.external_shell_elements = explicit_points ? shells : 0;

      auto* const public_qm_force =
          arena_pointer<double>(execution_arena, execution.public_qm_force);
      auto* const public_point_force = arena_pointer_if<double>(
          execution_arena, execution.public_point_force, point_coordinates);
      auto* const staged_qm_force =
          arena_pointer<double>(execution_arena, execution.staged_qm_force);
      auto* const staged_point_force = arena_pointer_if<double>(
          execution_arena, execution.staged_point_force, point_coordinates);
      binding.results.forces = {public_qm_force, coordinates, public_point_force, point_coordinates,
                                token};

      binding.intermediates.post_scc_potential = {
          {post_complete_shell, shells, post_complete_atom, atoms, post_complete_dipole,
           coordinates, post_complete_quadrupole, quadrupole_elements, token},
          post_shell_scalar,
          shells,
          token};
      auto& post_intermediates = binding.intermediates.post_scc_potential_intermediates;
      post_intermediates.es2_shell =
          arena_pointer<double>(execution_arena, execution.post_es2_shell);
      post_intermediates.es2_shell_elements = shells;
      post_intermediates.es3_shell =
          arena_pointer<double>(execution_arena, execution.post_es3_shell);
      post_intermediates.es3_shell_elements = shells;
      post_intermediates.aes2_atomic =
          arena_pointer<double>(execution_arena, execution.post_aes2_atom);
      post_intermediates.aes2_atomic_elements = atoms;
      post_intermediates.aes2_dipole =
          arena_pointer<double>(execution_arena, execution.post_aes2_dipole);
      post_intermediates.aes2_dipole_elements = coordinates;
      post_intermediates.aes2_quadrupole =
          arena_pointer<double>(execution_arena, execution.post_aes2_quadrupole);
      post_intermediates.aes2_quadrupole_elements = quadrupole_elements;
      post_intermediates.d4_atomic =
          arena_pointer_if<double>(execution_arena, execution.post_d4_atom, d4_enabled ? atoms : 0);
      post_intermediates.d4_atomic_elements = d4_enabled ? atoms : 0;
      post_intermediates.periodic_atomic = arena_pointer_if<double>(
          execution_arena, execution.post_periodic_atom, periodic_enabled ? atoms : 0);
      post_intermediates.periodic_atomic_elements = periodic_enabled ? atoms : 0;
      post_intermediates.complete = {
          arena_pointer<double>(execution_arena, execution.post_staged_shell),
          shells,
          arena_pointer<double>(execution_arena, execution.post_staged_atom),
          atoms,
          arena_pointer<double>(execution_arena, execution.post_staged_dipole),
          coordinates,
          arena_pointer<double>(execution_arena, execution.post_staged_quadrupole),
          quadrupole_elements,
          token};
      post_intermediates.shell_scalar =
          arena_pointer<double>(execution_arena, execution.post_staged_shell_scalar);
      post_intermediates.shell_scalar_elements = shells;
      post_intermediates.plan_token = token;

      auto* const overlap_adjoint =
          arena_pointer<double>(execution_arena, execution.overlap_adjoint);
      auto* const electronic_gradient =
          arena_pointer<double>(execution_arena, execution.electronic_gradient);
      binding.intermediates.h0 = {
          overlap_adjoint,
          matrices,
          arena_pointer<double>(execution_arena, execution.coordination_adjoint),
          atoms,
          electronic_gradient,
          coordinates,
          token};
      binding.intermediates.hamiltonian = {
          overlap_adjoint,
          matrices,
          arena_pointer<double>(execution_arena, execution.dipole_adjoint),
          dipole_matrix_elements,
          arena_pointer<double>(execution_arena, execution.quadrupole_adjoint),
          quadrupole_matrix_elements,
          token};
      binding.intermediates.classical = {
          arena_pointer<double>(execution_arena, execution.classical_force), coordinates, token};
      binding.intermediates.explicit_qm_forces = arena_pointer_if<double>(
          execution_arena, execution.explicit_qm_force, explicit_points ? coordinates : 0);
      binding.intermediates.explicit_qm_force_elements = explicit_points ? coordinates : 0;
      binding.intermediates.explicit_point_forces = arena_pointer_if<double>(
          execution_arena, execution.explicit_point_force, explicit_points ? point_coordinates : 0);
      binding.intermediates.explicit_point_force_elements = explicit_points ? point_coordinates : 0;
      binding.intermediates.forces = {staged_qm_force, coordinates, staged_point_force,
                                      point_coordinates, token};

      auto& post_workspace = binding.workspace.post_scc_potential;
      post_workspace.es2.shell_scratch =
          arena_pointer<double>(execution_arena, execution.post_es2_shell_scratch);
      post_workspace.es2.shell_elements = shells;
      post_workspace.aes2.potential_scratch =
          arena_pointer<double>(execution_arena, execution.post_aes2_potential_scratch);
      post_workspace.aes2.potential_elements = candidate.host.aes2.potential_scratch_elements();
      post_workspace.aes2.scc_peer_error_scratch = device_error(kPostAes2PeerError);
      post_workspace.aes2.scc_peer_error_elements = 1;
      if (d4_enabled) {
        post_workspace.d4 = {
            arena_pointer<double>(execution_arena, execution.post_d4_weights),
            arena_pointer<double>(execution_arena, execution.post_d4_weight_cn),
            arena_pointer<double>(execution_arena, execution.post_d4_weight_charge),
            d4_weight_elements,
            arena_pointer<double>(execution_arena, execution.post_d4_atom_scratch),
            arena_pointer<double>(execution_arena, execution.post_d4_coordination),
            atoms,
            arena_pointer<double>(execution_arena, execution.post_d4_batch),
            batch,
            arena_pointer<double>(execution_arena, execution.post_d4_gradient),
            coordinates,
            system_errors(kPostStageSystemError),
            batch};
      }
      if (periodic_enabled) {
        post_workspace.periodic = {
            arena_pointer<double>(execution_arena, execution.post_periodic_potential),
            arena_pointer<double>(execution_arena, execution.post_periodic_raw_response),
            sequence_pool + kPostPeriodicSequence,
            atoms,
            1,
            token};
      }
      post_workspace.composition = {
          arena_pointer<double>(execution_arena, execution.post_composition_shell),
          shells,
          arena_pointer<double>(execution_arena, execution.post_composition_atom),
          atoms,
          arena_pointer<double>(execution_arena, execution.post_composition_dipole),
          coordinates,
          arena_pointer<double>(execution_arena, execution.post_composition_quadrupole),
          quadrupole_elements,
          sequence_pool + kPostCompositionSequence,
          1,
          token};
      post_workspace.scalar_bridge = {
          arena_pointer<double>(execution_arena, execution.post_bridge_shell), shells,
          sequence_pool + kPostBridgeSequence, 1, token};
      post_workspace.active_mask = mask(kPostActiveMask);
      post_workspace.active_elements = batch;
      post_workspace.sequence_active = sequence_pool + kPostSequence;
      post_workspace.sequence_elements = 1;
      post_workspace.stage_system_errors = system_errors(kPostStageSystemError);
      post_workspace.stage_system_error_elements = batch;
      post_workspace.stage_device_error = device_error(kPostStageDeviceError);
      post_workspace.stage_device_error_elements = 1;
      post_workspace.plan_token = token;

      binding.workspace.h0 = {
          arena_pointer<double>(execution_arena, execution.h0_overlap_scratch),
          matrices,
          arena_pointer<double>(execution_arena, execution.h0_coordination_scratch),
          atoms,
          arena_pointer<double>(execution_arena, execution.h0_gradient_scratch),
          coordinates,
          sequence_pool + kH0Sequence,
          1,
          token};
      binding.workspace.hamiltonian = {
          arena_pointer<double>(execution_arena, execution.hamiltonian_overlap_scratch),
          matrices,
          arena_pointer<double>(execution_arena, execution.hamiltonian_dipole_scratch),
          dipole_matrix_elements,
          arena_pointer<double>(execution_arena, execution.hamiltonian_quadrupole_scratch),
          quadrupole_matrix_elements,
          sequence_pool + kHamiltonianSequence,
          1,
          token};
      binding.workspace.integral = {
          arena_pointer<double>(execution_arena, execution.integral_gradient_scratch), coordinates,
          sequence_pool + kIntegralSequence, 1, token};
      binding.workspace.electronic = {mask(kH0SuccessMask), mask(kHamiltonianSuccessMask), batch,
                                      token};
      binding.workspace.coordination = {
          nullptr,
          0,
          nullptr,
          0,
          arena_pointer<double>(execution_arena, execution.coordination_gradient_scratch),
          coordinates,
          sequence_pool + kCoordinationSequence,
          1,
          token};

      Gfn2AES2DeviceWorkspace classical_aes2{};
      /* Only the AES2 reverse pass is used here. Assign by name because the
       * update/potential scratch fields precede the gradient fields. */
      classical_aes2.gradient_scratch =
          arena_pointer<double>(execution_arena, execution.classical_aes2_gradient);
      classical_aes2.gradient_elements = coordinates;
      classical_aes2.coordination_scratch =
          arena_pointer<double>(execution_arena, execution.classical_aes2_coordination);
      classical_aes2.coordination_elements = atoms;
      Gfn2D4DeviceWorkspace classical_d4{};
      if (d4_enabled) {
        classical_d4 = {
            arena_pointer<double>(execution_arena, execution.classical_d4_weights),
            arena_pointer<double>(execution_arena, execution.classical_d4_weight_cn),
            arena_pointer<double>(execution_arena, execution.classical_d4_weight_charge),
            d4_weight_elements,
            arena_pointer<double>(execution_arena, execution.classical_d4_atom_scratch),
            arena_pointer<double>(execution_arena, execution.classical_d4_coordination),
            atoms,
            arena_pointer<double>(execution_arena, execution.classical_d4_batch),
            batch,
            arena_pointer<double>(execution_arena, execution.classical_d4_gradient),
            coordinates,
            system_errors(kClassicalPrimitiveSystemError),
            batch};
      }
      const Gfn2GeometryDeviceWorkspace classical_geometry{
          nullptr,
          0,
          nullptr,
          0,
          arena_pointer<double>(execution_arena, execution.classical_geometry_gradient),
          coordinates,
          sequence_pool + kClassicalPrimitiveSequence,
          1,
          token};
      binding.workspace.classical = {
          arena_pointer<double>(execution_arena, execution.classical_gradient_scratch),
          coordinates,
          arena_pointer<double>(execution_arena, execution.classical_force_scratch),
          coordinates,
          arena_pointer<double>(execution_arena, execution.classical_coordination_adjoint),
          atoms,
          mask(kClassicalSelectedMask),
          batch,
          system_errors(kClassicalPrimitiveSystemError),
          batch,
          device_error(kClassicalPrimitiveDeviceError),
          1,
          sequence_pool + kClassicalSequence,
          1,
          classical_aes2,
          classical_d4,
          classical_geometry,
          token};
      binding.workspace.external_point_charge = {
          arena_pointer_if<double>(execution_arena, execution.external_qm_scratch,
                                   explicit_points ? coordinates : 0),
          explicit_points ? coordinates : 0,
          arena_pointer_if<double>(execution_arena, execution.external_point_scratch,
                                   explicit_points ? point_coordinates : 0),
          explicit_points ? point_coordinates : 0,
          sequence_pool + kExternalSequence,
          1,
          token};
      binding.workspace.force_composition = {
          arena_pointer<double>(execution_arena, execution.composition_qm_scratch),
          coordinates,
          arena_pointer_if<double>(execution_arena, execution.composition_point_scratch,
                                   point_coordinates),
          point_coordinates,
          sequence_pool + kCompositionSequence,
          1,
          token};
      binding.workspace.energy_success_mask = mask(kEnergySuccessMask);
      binding.workspace.post_scc_success_mask = mask(kPostSuccessMask);
      binding.workspace.electronic_success_mask = mask(kElectronicSuccessMask);
      binding.workspace.coordination_success_mask = mask(kCoordinationSuccessMask);
      binding.workspace.classical_success_mask = mask(kClassicalSuccessMask);
      binding.workspace.external_success_mask = mask(kExternalSuccessMask);
      binding.workspace.mask_elements = batch;
      /* Step 5 sparse CN VJP differential scratch and sequence. */
      binding.workspace.sparse_gradient_scratch =
          arena_pointer<double>(execution_arena, execution.sparse_gradient_scratch);
      binding.workspace.sparse_gradient_elements = coordinates;
      binding.workspace.sparse_sequence_active =
          arena_pointer<std::uint32_t>(execution_arena, execution.sparse_sequence);
      binding.workspace.sparse_sequence_elements = 1;

      binding.diagnostics.post_scc_potential = {system_errors(kPostSystemError),
                                                device_error(kPostDeviceError), batch, token};
      binding.diagnostics.electronic = {system_errors(kH0SystemError),
                                        device_error(kH0DeviceError),
                                        system_errors(kHamiltonianSystemError),
                                        device_error(kHamiltonianDeviceError),
                                        system_errors(kIntegralSystemError),
                                        device_error(kIntegralDeviceError),
                                        batch,
                                        token};
      binding.diagnostics.coordination_system_errors = system_errors(kCoordinationSystemError);
      binding.diagnostics.coordination_device_error = device_error(kCoordinationDeviceError);
      binding.diagnostics.classical_system_errors = system_errors(kClassicalSystemError);
      binding.diagnostics.classical_device_error = device_error(kClassicalDeviceError);
      binding.diagnostics.external_system_errors = system_errors(kExternalSystemError);
      binding.diagnostics.external_device_error = device_error(kExternalDeviceError);
      binding.diagnostics.force_composition_system_errors = system_errors(kCompositionSystemError);
      binding.diagnostics.force_composition_plan_error = device_error(kCompositionPlanError);

      /* The setup owner uploads host-generated geometry values, but the force
       * reverse passes compare their compact caches against CUDA arithmetic.
       * Build the geometry and CN-dependent AES2 caches once on the setup
       * stream; the numerical refresh transaction reuses these fixed public
       * addresses after geometry changes. */
      cuda_status = reset_gfn2_geometry_device_errors_cuda(
          batch, binding.diagnostics.coordination_system_errors,
          binding.diagnostics.coordination_device_error, stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA initial geometry diagnostic reset", cuda_status);
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      cuda_status = update_gfn2_geometry_cache_cuda(
          binding.plan.coordination_batch, positions, candidate.host.geometry_generation,
          binding.plan.coordination_cache, candidate.workspace_seed.geometry_workspace,
          binding.diagnostics.coordination_system_errors,
          binding.diagnostics.coordination_device_error, stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA initial geometry-cache construction", cuda_status);
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      cuda_status =
          reset_gfn2_aes2_device_errors_cuda(batch, binding.diagnostics.coordination_system_errors,
                                             binding.diagnostics.coordination_device_error, stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA initial AES2 diagnostic reset", cuda_status);
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      cuda_status = update_gfn2_aes2_geometry_cache_cuda(
          candidate.plan_seed.aes2_batch, positions,
          binding.plan.coordination_cache.coordination_numbers, candidate.plan_seed.aes2_cache,
          candidate.workspace_seed.aes2_workspace, binding.diagnostics.coordination_system_errors,
          binding.diagnostics.coordination_device_error, stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA initial AES2-cache construction", cuda_status);
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }

    /* Exercise the published descriptors, not merely their host validation.
     * All-bits-one is a NaN for the CUDA-supported IEEE-754 double format, so
     * a finite download proves that terminal publication actually ran. */
    std::size_t output_bytes = 0u;
    if (!checked_bytes(batch, sizeof(double), output_bytes)) {
      error = "energy smoke output extent overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = cudaMemsetAsync(binding.results.energy.total_energy, 0xff, output_bytes, stream);
    if (cuda_status == cudaSuccess && force_mode) {
      if (!checked_bytes(binding.results.forces.qm_force_elements, sizeof(double), output_bytes)) {
        error = "QM-force smoke output extent overflows size_t";
        return XTBLOOM_STATUS_ALLOCATION_FAILED;
      }
      cuda_status = cudaMemsetAsync(binding.results.forces.qm_forces, 0xff, output_bytes, stream);
      if (cuda_status == cudaSuccess && binding.results.forces.point_force_elements != 0) {
        if (!checked_bytes(binding.results.forces.point_force_elements, sizeof(double),
                           output_bytes)) {
          error = "point-force smoke output extent overflows size_t";
          return XTBLOOM_STATUS_ALLOCATION_FAILED;
        }
        cuda_status =
            cudaMemsetAsync(binding.results.forces.point_forces, 0xff, output_bytes, stream);
      }
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemsetAsync(candidate.state_seed.scc.free_energies, 0,
                                    static_cast<std::size_t>(batch) * sizeof(double), stream);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status =
          cudaMemsetAsync(candidate.state_seed.scc.system_statuses, 0,
                          static_cast<std::size_t>(batch) * sizeof(xtbloom_status_t), stream);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemsetAsync(candidate.state_seed.scc.converged, 1,
                                    static_cast<std::size_t>(batch) * sizeof(std::uint8_t), stream);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA energy/force smoke initialization", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    cuda_status = project_gfn2_stationary_force_state_cuda(binding.stationary_projection, stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA stationary force projection smoke", cuda_status);
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    /* The binding is constructed before the first device numerical refresh, so
     * neither the general committed pair list nor the D4 committed pair-list/CN
     * bundle is eligible yet. Run the validation smoke on a plan copy with those
     * refresh-owned consumers disabled. The dense coordination VJP and non-D4
     * leaves still exercise the complete publication chain; the production plan
     * retains every D4 component and exercises it after the first real refresh
     * has atomically published the pair lists, CN, generation, and eligibility. */
    Gfn2EnergyForceExecutionDevicePlan validation_plan = binding.plan;
    validation_plan.pairlist_committed.plan_token = 0u;
    validation_plan.pairlist_batch.plan_token = 0u;
    constexpr std::uint32_t kD4SccPotential =
        static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kD4TwoBody);
    constexpr std::uint32_t kD4SccEnergy =
        static_cast<std::uint32_t>(Gfn2SccClassicalEnergyComponent::kD4TwoBody);
    constexpr std::uint32_t kD4AtmEnergy =
        static_cast<std::uint32_t>(Gfn2TotalEnergyComponent::kD4Atm);
    constexpr std::uint32_t kD4ClassicalForces =
        static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4TwoBody) |
        static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4ATM);
    validation_plan.scc_potential_components &= ~kD4SccPotential;
    validation_plan.scc_energy_components &= ~kD4SccEnergy;
    validation_plan.post_scc_potential_plan.enabled_components &= ~kD4SccPotential;
    validation_plan.total_energy_batch.enabled_components &= ~kD4AtmEnergy;
    validation_plan.classical_plan.enabled_components &= ~kD4ClassicalForces;
    Gfn2EnergyForceExecutionDeviceInput validation_input = binding.input;
    validation_input.total_energy.d4_atm = nullptr;
    validation_input.total_energy.d4_atm_elements = 0;
    cuda_status = execute_gfn2_energy_force_cuda(validation_plan, validation_input, binding.results,
                                                 binding.intermediates, binding.workspace,
                                                 binding.diagnostics, stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA energy/force composed smoke", cuda_status);
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    /* The smoke borrows the initialized SCC storage only long enough to drive
     * the terminal chain. Restore the immutable device checkpoint so the
     * published runtime still starts at iteration zero. */
    const auto restore_diagnostic = candidate.initializer.upload_async(
        candidate.iteration_arena.get(), candidate.iteration_arena.bytes(), candidate.ready,
        stream);
    if (!restore_diagnostic.success()) {
      error = setup_error_message("CUDA SCC fresh-state restoration", restore_diagnostic.status,
                                  static_cast<std::uint32_t>(restore_diagnostic.error),
                                  static_cast<std::uint32_t>(restore_diagnostic.field),
                                  restore_diagnostic.index);
      return restore_diagnostic.status;
    }
    candidate.submitted = true;
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t build_inference_bindings(Prepared& candidate, std::string& error) {
    const std::int64_t batch = candidate.host.basis.batch_size;
    const std::int64_t atoms = candidate.host.basis.total_atoms;
    const std::int64_t points = candidate.host.external.total_point_charges;
    const std::uint64_t token = candidate.host.plan_token;
    const std::uint32_t requested = candidate.host.key.flags;
    const bool d4_enabled = candidate.host.d4_enabled;
    const bool energy_requested =
        (requested & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ENERGY)) != 0u;
    const bool force_requested =
        (requested & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_FORCES)) != 0u;
    const bool charges_requested =
        (requested & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ATOMIC_CHARGES)) != 0u;
    const bool point_forces_requested =
        (requested & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_POINT_CHARGE_FORCES)) != 0u;
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    std::int64_t d4_weight_elements = 0;
    if (requested == 0u || !checked_elements(atoms, 3, coordinates) ||
        !checked_elements(points, 3, point_coordinates) ||
        !checked_elements(atoms, kGfn2D4MaximumReferences, d4_weight_elements)) {
      error = "inference result extent or requested-property mask is invalid";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    struct Offsets {
      std::size_t atomic_numbers = 0u;
      std::size_t terminal_repulsion_candidate = 0u;
      std::size_t terminal_d4_candidate = 0u;
      std::size_t terminal_d4_weights = 0u;
      std::size_t terminal_d4_weight_cn = 0u;
      std::size_t terminal_d4_weight_charge = 0u;
      std::size_t terminal_d4_atom_scratch = 0u;
      std::size_t terminal_d4_coordination_adjoints = 0u;
      std::size_t terminal_d4_batch_scratch = 0u;
      std::size_t terminal_d4_gradient_scratch = 0u;
      std::size_t terminal_epoch_snapshot = 0u;
      std::size_t terminal_repulsion_error = 0u;
      std::size_t terminal_d4_system_errors = 0u;
      std::size_t terminal_d4_device_error = 0u;
      std::size_t terminal_system_errors = 0u;
      std::size_t terminal_plan_error = 0u;

      std::size_t result_energies = 0u;
      std::size_t result_qm_forces = 0u;
      std::size_t result_atomic_charges = 0u;
      std::size_t result_point_forces = 0u;
      std::size_t result_iterations = 0u;
      std::size_t result_converged = 0u;
      std::size_t result_system_statuses = 0u;
      std::size_t publication_epoch_snapshot = 0u;
      std::size_t publication_system_errors = 0u;
      std::size_t publication_plan_error = 0u;
      std::size_t warm_checkpoint_generations = 0u;
      std::size_t warm_checkpoint_batch_ready = 0u;
    } offset;

    ArenaLayout layout;
    offset.atomic_numbers =
        layout.append<std::int32_t>(candidate.atomic_numbers == nullptr ? atoms : 0);
    offset.terminal_repulsion_candidate = layout.append<double>(batch);
    offset.terminal_d4_candidate = layout.append<double>(d4_enabled ? batch : 0);
    offset.terminal_d4_weights = layout.append<double>(d4_enabled ? d4_weight_elements : 0);
    offset.terminal_d4_weight_cn = layout.append<double>(d4_enabled ? d4_weight_elements : 0);
    offset.terminal_d4_weight_charge = layout.append<double>(d4_enabled ? d4_weight_elements : 0);
    offset.terminal_d4_atom_scratch = layout.append<double>(d4_enabled ? atoms : 0);
    offset.terminal_d4_coordination_adjoints = layout.append<double>(d4_enabled ? atoms : 0);
    offset.terminal_d4_batch_scratch = layout.append<double>(d4_enabled ? batch : 0);
    offset.terminal_d4_gradient_scratch = layout.append<double>(d4_enabled ? coordinates : 0);
    offset.terminal_epoch_snapshot = layout.append<std::uint64_t>(1);
    offset.terminal_repulsion_error = layout.append<std::uint32_t>(1);
    offset.terminal_d4_system_errors = layout.append<std::uint32_t>(d4_enabled ? batch : 0);
    offset.terminal_d4_device_error = layout.append<std::uint32_t>(d4_enabled ? 1 : 0);
    offset.terminal_system_errors = layout.append<std::uint32_t>(batch);
    offset.terminal_plan_error = layout.append<std::uint32_t>(1);

    offset.result_energies = layout.append<double>(energy_requested ? batch : 0);
    offset.result_qm_forces = layout.append<double>(force_requested ? coordinates : 0);
    offset.result_atomic_charges = layout.append<double>(charges_requested ? atoms : 0);
    offset.result_point_forces =
        layout.append<double>(point_forces_requested ? point_coordinates : 0);
    offset.result_iterations = layout.append<std::int32_t>(batch);
    offset.result_converged = layout.append<std::uint8_t>(batch);
    offset.result_system_statuses = layout.append<xtbloom_status_t>(batch);
    offset.publication_epoch_snapshot = layout.append<std::uint64_t>(1);
    offset.publication_system_errors = layout.append<std::uint32_t>(batch);
    offset.publication_plan_error = layout.append<std::uint32_t>(1);
    offset.warm_checkpoint_generations = layout.append<std::uint64_t>(batch);
    offset.warm_checkpoint_batch_ready = layout.append<std::uint32_t>(1);
    if (!layout.valid()) {
      error = "inference arena layout overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }

    cudaError_t cuda_status = candidate.inference_arena.allocate(layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA inference arena allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    void* const arena = candidate.inference_arena.get();
    cuda_status = cudaMemsetAsync(arena, 0, candidate.inference_arena.bytes(), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA inference arena initialization", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const std::int32_t* terminal_atomic_numbers = candidate.atomic_numbers;
    if (terminal_atomic_numbers == nullptr) {
      auto* const destination = arena_pointer<std::int32_t>(arena, offset.atomic_numbers);
      cuda_status = cudaMemcpyAsync(destination, candidate.host.key.atomic_numbers.data(),
                                    static_cast<std::size_t>(atoms) * sizeof(std::int32_t),
                                    cudaMemcpyHostToDevice, stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA terminal atomic-number upload", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      terminal_atomic_numbers = destination;
    }
    candidate.submitted = true;

    auto& inference = candidate.inference;
    inference = {};
    inference.epoch_consumer = {
        candidate.numerical.preprocessing.geometry_epoch,
        candidate.numerical.device.committed_generations,
        candidate.numerical.device.eligible,
        batch,
        token,
    };

    const Gfn2RepulsionDeviceBatch repulsion{
        batch,
        atoms,
        candidate.device_topology.atom_offsets,
        terminal_atomic_numbers,
        candidate.numerical.device.committed_positions,
    };
    Gfn2D4DeviceBatch terminal_d4 = candidate.plan_seed.d4_batch;
    if (d4_enabled) terminal_d4.atomic_numbers = repulsion.atomic_numbers;
    inference.terminal_plan = {
        kGfn2TerminalClassicalEnergyAbiVersion,
        d4_enabled ? kGfn2TerminalClassicalEnergyAllComponents : 0u,
        token,
        repulsion,
        d4_enabled ? terminal_d4 : Gfn2D4DeviceBatch{},
        d4_enabled ? candidate.plan_seed.d4_parameters : Gfn2D4DeviceParameters{},
        d4_enabled ? candidate.numerical.d4_cache : Gfn2D4PairListDeviceCache{},
        candidate.numerical.preprocessing.geometry_epoch,
        candidate.numerical.device.committed_generations,
        batch,
    };
    inference.terminal_activity = {candidate.numerical.device.eligible, batch, token};
    inference.terminal_results = {
        const_cast<double*>(candidate.energy_force.input.total_energy.repulsion),
        batch,
        d4_enabled ? const_cast<double*>(candidate.energy_force.input.total_energy.d4_atm)
                   : nullptr,
        d4_enabled ? batch : 0,
        token,
    };
    inference.terminal_workspace.repulsion_candidate =
        arena_pointer<double>(arena, offset.terminal_repulsion_candidate);
    inference.terminal_workspace.repulsion_elements = batch;
    inference.terminal_workspace.d4_atm_candidate =
        arena_pointer_if<double>(arena, offset.terminal_d4_candidate, d4_enabled ? batch : 0);
    inference.terminal_workspace.d4_atm_elements = d4_enabled ? batch : 0;
    if (d4_enabled) {
      inference.terminal_workspace.d4 = {
          arena_pointer<double>(arena, offset.terminal_d4_weights),
          arena_pointer<double>(arena, offset.terminal_d4_weight_cn),
          arena_pointer<double>(arena, offset.terminal_d4_weight_charge),
          d4_weight_elements,
          arena_pointer<double>(arena, offset.terminal_d4_atom_scratch),
          arena_pointer<double>(arena, offset.terminal_d4_coordination_adjoints),
          atoms,
          arena_pointer<double>(arena, offset.terminal_d4_batch_scratch),
          batch,
          arena_pointer<double>(arena, offset.terminal_d4_gradient_scratch),
          coordinates,
          arena_pointer<std::uint32_t>(arena, offset.terminal_d4_system_errors),
          batch,
      };
    }
    inference.terminal_workspace.epoch_snapshot =
        arena_pointer<std::uint64_t>(arena, offset.terminal_epoch_snapshot);
    inference.terminal_workspace.epoch_snapshot_elements = 1;
    inference.terminal_workspace.plan_token = token;
    inference.terminal_diagnostics = {
        arena_pointer<std::uint32_t>(arena, offset.terminal_repulsion_error),
        arena_pointer_if<std::uint32_t>(arena, offset.terminal_d4_system_errors,
                                        d4_enabled ? batch : 0),
        d4_enabled ? batch : 0,
        arena_pointer_if<std::uint32_t>(arena, offset.terminal_d4_device_error, d4_enabled ? 1 : 0),
        arena_pointer<std::uint32_t>(arena, offset.terminal_system_errors),
        batch,
        arena_pointer<std::uint32_t>(arena, offset.terminal_plan_error),
        1,
        token,
    };

    const auto input_if_requested = [](const double* pointer, bool enabled) {
      return enabled ? pointer : nullptr;
    };
    inference.publication_plan = {
        kGfn2InferencePublicationAbiVersion,
        requested,
        token,
        static_cast<std::uint64_t>(candidate.host.key.maximum_iterations),
        batch,
        atoms,
        points,
        candidate.device_topology.atom_offsets,
        points == 0 ? nullptr
                    : candidate.plan_seed.explicit_point_charge_batch.point_charge_offsets,
        candidate.numerical.preprocessing.geometry_epoch,
        candidate.numerical.device.committed_generations,
        batch,
    };
    inference.publication_input = {
        candidate.numerical.device.eligible,
        batch,
        candidate.state_seed.scc.iterations,
        candidate.state_seed.scc.converged,
        candidate.state_seed.scc.system_statuses,
        batch,
        input_if_requested(candidate.energy_force.results.energy.total_energy, energy_requested),
        energy_requested ? batch : 0,
        input_if_requested(candidate.energy_force.results.forces.qm_forces, force_requested),
        force_requested ? coordinates : 0,
        input_if_requested(candidate.energy_force.stationary_projection.enabled == 1u
                               ? candidate.energy_force.stationary_projection.atomic_charges
                               : candidate.workspace_seed.physical_topology.atomic_charges,
                           charges_requested),
        charges_requested ? atoms : 0,
        input_if_requested(candidate.energy_force.results.forces.point_forces,
                           point_forces_requested),
        point_forces_requested ? point_coordinates : 0,
        inference.terminal_diagnostics.system_errors,
        batch,
        inference.terminal_diagnostics.plan_error,
        candidate.energy_force.diagnostics.execution_system_errors,
        batch,
        candidate.energy_force.workspace.plan_failure,
        token,
    };
    inference.publication_results = {
        arena_pointer_if<double>(arena, offset.result_energies, energy_requested ? batch : 0),
        energy_requested ? batch : 0,
        arena_pointer_if<double>(arena, offset.result_qm_forces, force_requested ? coordinates : 0),
        force_requested ? coordinates : 0,
        arena_pointer_if<double>(arena, offset.result_atomic_charges,
                                 charges_requested ? atoms : 0),
        charges_requested ? atoms : 0,
        arena_pointer_if<double>(arena, offset.result_point_forces,
                                 point_forces_requested ? point_coordinates : 0),
        point_forces_requested ? point_coordinates : 0,
        arena_pointer<std::int32_t>(arena, offset.result_iterations),
        arena_pointer<std::uint8_t>(arena, offset.result_converged),
        arena_pointer<xtbloom_status_t>(arena, offset.result_system_statuses),
        batch,
        token,
    };
    inference.publication_workspace = {
        arena_pointer<std::uint64_t>(arena, offset.publication_epoch_snapshot), 1, token};
    inference.publication_diagnostics = {
        arena_pointer<std::uint32_t>(arena, offset.publication_system_errors),
        batch,
        arena_pointer<std::uint32_t>(arena, offset.publication_plan_error),
        1,
        token,
    };
    inference.warm_checkpoint_generations =
        arena_pointer<std::uint64_t>(arena, offset.warm_checkpoint_generations);
    inference.warm_checkpoint_batch_ready =
        arena_pointer<std::uint32_t>(arena, offset.warm_checkpoint_batch_ready);
    inference.ready = true;
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t build_public_result_state(Prepared& candidate, std::string& error) {
    const std::int64_t batch = candidate.host.basis.batch_size;
    const std::int64_t atoms = candidate.host.basis.total_atoms;
    const std::int64_t points = candidate.host.external.total_point_charges;
    const std::uint32_t requested = candidate.host.key.flags;
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    std::int64_t lattice_elements = 0;
    if (!checked_elements(atoms, 3, coordinates) ||
        !checked_elements(points, 3, point_coordinates) ||
        !checked_elements(batch, 9, lattice_elements)) {
      error = "public CUDA result staging extent overflows int64_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    const bool energy_requested =
        (requested & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ENERGY)) != 0u;
    const bool force_requested =
        (requested & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_FORCES)) != 0u;
    const bool charges_requested =
        (requested & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ATOMIC_CHARGES)) != 0u;
    const bool point_forces_requested =
        (requested & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_POINT_CHARGE_FORCES)) != 0u;

    struct HostOffsets {
      std::size_t energies = 0u;
      std::size_t qm_forces = 0u;
      std::size_t atomic_charges = 0u;
      std::size_t point_forces = 0u;
      std::size_t iterations = 0u;
      std::size_t converged = 0u;
      std::size_t system_statuses = 0u;
      std::size_t control = 0u;
      std::size_t warm_checkpoint_ready = 0u;
      std::size_t lattice_cells = 0u;
      std::size_t periodic_axes = 0u;
    } host_offset;
    struct DeviceOffsets {
      std::size_t energies = 0u;
      std::size_t qm_forces = 0u;
      std::size_t atomic_charges = 0u;
      std::size_t point_forces = 0u;
      std::size_t iterations = 0u;
      std::size_t converged = 0u;
      std::size_t system_statuses = 0u;
      std::size_t control = 0u;
      std::size_t request_topology_error = 0u;
      std::size_t expected_atom_offsets = 0u;
      std::size_t expected_atomic_numbers = 0u;
      std::size_t expected_molecular_charges = 0u;
      std::size_t expected_unpaired_electrons = 0u;
      std::size_t expected_spin_channels = 0u;
      std::size_t expected_point_offsets = 0u;
      std::size_t expected_response_offsets = 0u;
      std::size_t lattice_cells = 0u;
      std::size_t periodic_axes = 0u;
    } device_offset;
    ArenaLayout host_layout;
    host_offset.energies = host_layout.append<double>(energy_requested ? batch : 0);
    host_offset.qm_forces = host_layout.append<double>(force_requested ? coordinates : 0);
    host_offset.atomic_charges = host_layout.append<double>(charges_requested ? atoms : 0);
    host_offset.point_forces =
        host_layout.append<double>(point_forces_requested ? point_coordinates : 0);
    host_offset.iterations = host_layout.append<std::int32_t>(batch);
    host_offset.converged = host_layout.append<std::uint8_t>(batch);
    host_offset.system_statuses = host_layout.append<xtbloom_status_t>(batch);
    host_offset.control = host_layout.append<Gfn2PublicResultBridgeControl>(1);
    host_offset.warm_checkpoint_ready = host_layout.append<std::uint32_t>(1);
    host_offset.lattice_cells = host_layout.append<double>(lattice_elements);
    host_offset.periodic_axes = host_layout.append<std::int32_t>(batch);

    ArenaLayout device_layout;
    device_offset.energies = device_layout.append<double>(energy_requested ? batch : 0);
    device_offset.qm_forces = device_layout.append<double>(force_requested ? coordinates : 0);
    device_offset.atomic_charges = device_layout.append<double>(charges_requested ? atoms : 0);
    device_offset.point_forces =
        device_layout.append<double>(point_forces_requested ? point_coordinates : 0);
    device_offset.iterations = device_layout.append<std::int32_t>(batch);
    device_offset.converged = device_layout.append<std::uint8_t>(batch);
    device_offset.system_statuses = device_layout.append<xtbloom_status_t>(batch);
    device_offset.control = device_layout.append<Gfn2PublicResultBridgeControl>(1);
    device_offset.request_topology_error = device_layout.append<std::uint32_t>(1);
    device_offset.expected_atom_offsets = device_layout.append<std::int64_t>(batch + 1);
    device_offset.expected_atomic_numbers = device_layout.append<std::int32_t>(atoms);
    device_offset.expected_molecular_charges = device_layout.append<double>(batch);
    device_offset.expected_unpaired_electrons = device_layout.append<std::int32_t>(batch);
    device_offset.expected_spin_channels = device_layout.append<std::int32_t>(batch);
    device_offset.expected_point_offsets = device_layout.append<std::int64_t>(batch + 1);
    device_offset.expected_response_offsets = device_layout.append<std::int64_t>(batch + 1);
    device_offset.lattice_cells = device_layout.append<double>(lattice_elements);
    device_offset.periodic_axes = device_layout.append<std::int32_t>(batch);
    if (!host_layout.valid() || !device_layout.valid()) {
      error = "public CUDA result staging layout overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cudaError_t cuda_status = candidate.public_result_host_arena.allocate(host_layout.bytes());
    if (cuda_status == cudaSuccess) {
      cuda_status = candidate.public_result_device_arena.allocate(device_layout.bytes());
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = candidate.public_result_completion_event.create(cudaEventDisableTiming);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA public result staging allocation", cuda_status);
      return cuda_status == cudaErrorMemoryAllocation ? XTBLOOM_STATUS_ALLOCATION_FAILED
                                                      : XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    cuda_status = cudaMemsetAsync(candidate.public_result_device_arena.get(), 0,
                                  candidate.public_result_device_arena.bytes(), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA public result control initialization", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    candidate.submitted = true;

    void* const host_arena = candidate.public_result_host_arena.get();
    auto& state = candidate.public_result;
    state = {};
    state.energies =
        arena_pointer_if<double>(host_arena, host_offset.energies, energy_requested ? batch : 0);
    state.qm_forces = arena_pointer_if<double>(host_arena, host_offset.qm_forces,
                                               force_requested ? coordinates : 0);
    state.atomic_charges = arena_pointer_if<double>(host_arena, host_offset.atomic_charges,
                                                    charges_requested ? atoms : 0);
    state.point_forces = arena_pointer_if<double>(host_arena, host_offset.point_forces,
                                                  point_forces_requested ? point_coordinates : 0);
    state.iterations = arena_pointer<std::int32_t>(host_arena, host_offset.iterations);
    state.converged = arena_pointer<std::uint8_t>(host_arena, host_offset.converged);
    state.system_statuses =
        arena_pointer<xtbloom_status_t>(host_arena, host_offset.system_statuses);
    state.host_control =
        arena_pointer<Gfn2PublicResultBridgeControl>(host_arena, host_offset.control);
    state.warm_checkpoint_ready =
        arena_pointer<std::uint32_t>(host_arena, host_offset.warm_checkpoint_ready);
    void* const device_arena = candidate.public_result_device_arena.get();
    state.request_topology_error =
        arena_pointer<std::uint32_t>(device_arena, device_offset.request_topology_error);
    state.expected_atom_offsets =
        arena_pointer<std::int64_t>(device_arena, device_offset.expected_atom_offsets);
    state.expected_atomic_numbers =
        arena_pointer<std::int32_t>(device_arena, device_offset.expected_atomic_numbers);
    state.expected_molecular_charges =
        arena_pointer<double>(device_arena, device_offset.expected_molecular_charges);
    state.expected_unpaired_electrons =
        arena_pointer<std::int32_t>(device_arena, device_offset.expected_unpaired_electrons);
    state.expected_spin_channels =
        arena_pointer<std::int32_t>(device_arena, device_offset.expected_spin_channels);
    state.expected_point_offsets =
        arena_pointer<std::int64_t>(device_arena, device_offset.expected_point_offsets);
    state.expected_response_offsets =
        arena_pointer<std::int64_t>(device_arena, device_offset.expected_response_offsets);
    state.staged_lattice_cells = arena_pointer<double>(device_arena, device_offset.lattice_cells);
    state.staged_periodic_axes =
        arena_pointer<std::int32_t>(device_arena, device_offset.periodic_axes);
    state.host_lattice_cells = arena_pointer<double>(host_arena, host_offset.lattice_cells);
    state.host_periodic_axes = arena_pointer<std::int32_t>(host_arena, host_offset.periodic_axes);
    state.device_staging = {
        arena_pointer_if<double>(device_arena, device_offset.energies,
                                 energy_requested ? batch : 0),
        energy_requested ? batch : 0,
        arena_pointer_if<double>(device_arena, device_offset.qm_forces,
                                 force_requested ? coordinates : 0),
        force_requested ? coordinates : 0,
        arena_pointer_if<double>(device_arena, device_offset.atomic_charges,
                                 charges_requested ? atoms : 0),
        charges_requested ? atoms : 0,
        arena_pointer_if<double>(device_arena, device_offset.point_forces,
                                 point_forces_requested ? point_coordinates : 0),
        point_forces_requested ? point_coordinates : 0,
        arena_pointer<std::int32_t>(device_arena, device_offset.iterations),
        arena_pointer<std::uint8_t>(device_arena, device_offset.converged),
        arena_pointer<xtbloom_status_t>(device_arena, device_offset.system_statuses),
        batch,
        candidate.host.plan_token,
    };
    state.diagnostics = {
        arena_pointer<Gfn2PublicResultBridgeControl>(device_arena, device_offset.control),
        1,
        candidate.host.plan_token,
    };
    const auto upload = [&](std::size_t offset, const auto& values) {
      if (cuda_status != cudaSuccess || values.empty()) return;
      cuda_status =
          cudaMemcpyAsync(static_cast<std::byte*>(device_arena) + offset, values.data(),
                          values.size() * sizeof(values[0]), cudaMemcpyHostToDevice, stream);
    };
    upload(device_offset.expected_atom_offsets, candidate.host.key.atom_offsets);
    upload(device_offset.expected_atomic_numbers, candidate.host.key.atomic_numbers);
    upload(device_offset.expected_molecular_charges, candidate.host.key.molecular_charges);
    upload(device_offset.expected_unpaired_electrons, candidate.host.key.unpaired_electrons);
    upload(device_offset.expected_spin_channels, candidate.host.key.spin_channels);
    upload(device_offset.expected_point_offsets, candidate.host.key.point_offsets);
    upload(device_offset.expected_response_offsets, candidate.host.key.response_offsets);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA fixed-topology comparison upload", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    state.ready = true;
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t validate_candidate_setup(Prepared& candidate, std::string& error) {
    const auto& binding = candidate.energy_force;
    const std::int64_t batch = candidate.host.basis.batch_size;
    const std::int64_t setup_systems = candidate.eigensolver_binding.setup_system_error_elements;
    const std::int64_t factor_statuses = candidate.eigensolver_binding.cache.status_elements;
    const std::int64_t qm_force_elements =
        binding.plan.compute_forces == 1u ? binding.results.forces.qm_force_elements : 0;
    const std::int64_t point_force_elements =
        binding.plan.compute_forces == 1u ? binding.results.forces.point_force_elements : 0;
    struct Offsets {
      std::size_t setup_device_error = 0u;
      std::size_t setup_system_errors = 0u;
      std::size_t factor_statuses = 0u;
      std::size_t execution_system_errors = 0u;
      std::size_t execution_device_error = 0u;
      std::size_t plan_failure = 0u;
      std::size_t energies = 0u;
      std::size_t qm_forces = 0u;
      std::size_t point_forces = 0u;
      std::size_t converged = 0u;
    } offset;
    ArenaLayout layout;
    offset.setup_device_error = layout.append<std::uint32_t>(1);
    offset.setup_system_errors = layout.append<std::uint32_t>(setup_systems);
    offset.factor_statuses = layout.append<std::uint32_t>(factor_statuses);
    offset.execution_system_errors = layout.append<std::uint32_t>(batch);
    offset.execution_device_error = layout.append<std::uint32_t>(1);
    offset.plan_failure = layout.append<std::uint32_t>(1);
    offset.energies = layout.append<double>(batch);
    offset.qm_forces = layout.append<double>(qm_force_elements);
    offset.point_forces = layout.append<double>(point_force_elements);
    offset.converged = layout.append<std::uint8_t>(batch);
    if (!layout.valid()) {
      error = "CUDA candidate validation staging layout overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cudaError_t cuda_status = candidate.candidate_validation_arena.allocate(layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA candidate validation staging allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    void* const arena = candidate.candidate_validation_arena.get();
    auto* const setup_device_error = arena_pointer<std::uint32_t>(arena, offset.setup_device_error);
    auto* const setup_system_errors =
        arena_pointer_if<std::uint32_t>(arena, offset.setup_system_errors, setup_systems);
    auto* const factor_status_values =
        arena_pointer_if<std::uint32_t>(arena, offset.factor_statuses, factor_statuses);
    auto* const execution_system_errors =
        arena_pointer<std::uint32_t>(arena, offset.execution_system_errors);
    auto* const execution_device_error =
        arena_pointer<std::uint32_t>(arena, offset.execution_device_error);
    auto* const plan_failure = arena_pointer<std::uint32_t>(arena, offset.plan_failure);
    auto* const energies = arena_pointer<double>(arena, offset.energies);
    auto* const qm_forces = arena_pointer_if<double>(arena, offset.qm_forces, qm_force_elements);
    auto* const point_forces =
        arena_pointer_if<double>(arena, offset.point_forces, point_force_elements);
    auto* const converged = arena_pointer<std::uint8_t>(arena, offset.converged);
    const auto enqueue = [&](void* destination, const void* source, std::int64_t elements,
                             std::size_t element_size) {
      return elements == 0 ? cudaSuccess
                           : cudaMemcpyAsync(destination, source,
                                             static_cast<std::size_t>(elements) * element_size,
                                             cudaMemcpyDeviceToHost, stream);
    };
    cuda_status = enqueue(setup_device_error, candidate.eigensolver_binding.setup_device_error, 1,
                          sizeof(std::uint32_t));
    if (cuda_status == cudaSuccess) {
      cuda_status = enqueue(setup_system_errors, candidate.eigensolver_binding.setup_system_errors,
                            setup_systems, sizeof(std::uint32_t));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status =
          enqueue(factor_status_values, candidate.eigensolver_binding.cache.factor_statuses,
                  factor_statuses, sizeof(std::uint32_t));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = enqueue(execution_system_errors, binding.diagnostics.execution_system_errors,
                            batch, sizeof(std::uint32_t));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = enqueue(execution_device_error, binding.diagnostics.execution_device_error, 1,
                            sizeof(std::uint32_t));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = enqueue(plan_failure, binding.workspace.plan_failure, 1, sizeof(std::uint32_t));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = enqueue(energies, binding.results.energy.total_energy, batch, sizeof(double));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status =
          enqueue(qm_forces, binding.results.forces.qm_forces, qm_force_elements, sizeof(double));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = enqueue(point_forces, binding.results.forces.point_forces, point_force_elements,
                            sizeof(double));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status =
          enqueue(converged, candidate.state_seed.scc.converged, batch, sizeof(std::uint8_t));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaEventRecord(candidate.public_result_completion_event.get(), stream);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaEventSynchronize(candidate.public_result_completion_event.get());
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA candidate validation completion", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    candidate.submitted = false;

    const auto any_nonzero = [](const std::uint32_t* values, std::int64_t elements) {
      return elements != 0 && std::any_of(values, values + elements,
                                          [](std::uint32_t value) { return value != 0u; });
    };
    if (*setup_device_error != 0u || any_nonzero(setup_system_errors, setup_systems) ||
        any_nonzero(factor_status_values, factor_statuses)) {
      error = "CUDA eigensolver setup reported an asynchronous factorization failure";
      return XTBLOOM_STATUS_EIGENSOLVER_FAILED;
    }

    const auto first_system_error =
        std::find_if(execution_system_errors, execution_system_errors + batch,
                     [](std::uint32_t value) { return value != 0u; });
    if (*execution_device_error != 0u || *plan_failure != 0u ||
        first_system_error != execution_system_errors + batch) {
      std::ostringstream message;
      message << "CUDA energy/force smoke reported device_error=" << *execution_device_error
              << " plan_failure=" << *plan_failure;
      if (first_system_error != execution_system_errors + batch) {
        message << " system=" << std::distance(execution_system_errors, first_system_error)
                << " system_error=" << *first_system_error;
      }
      error = message.str();
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const auto all_finite = [](const double* values, std::int64_t elements) {
      return elements == 0 || std::all_of(values, values + elements,
                                          [](double value) { return std::isfinite(value); });
    };
    if (!all_finite(energies, batch) || !all_finite(qm_forces, qm_force_elements) ||
        !all_finite(point_forces, point_force_elements)) {
      error = "CUDA energy/force smoke did not publish every requested output";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    if (std::any_of(converged, converged + batch, [](std::uint8_t value) { return value != 0u; })) {
      error = "CUDA energy/force smoke failed to restore the fresh SCC state";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    candidate.energy_force_smoke_ready = true;
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t build_candidate(TopologyKey&& key, std::vector<double>&& positions,
                                   std::vector<double>&& point_positions,
                                   std::vector<double>&& point_values,
                                   std::vector<double>&& point_gammas,
                                   std::vector<double>&& periodic_shifts,
                                   std::vector<double>&& periodic_response,
                                   std::unique_ptr<Prepared>& output, std::string& error) {
    auto candidate = std::make_unique<Prepared>(stream);
    const std::uint64_t fingerprint = key.fingerprint();
    std::uint64_t token = hash_mix(fingerprint ^ next_plan_token++ ^ 0x112112112ULL);
    if (token == 0u) token = next_plan_token++;
    xtbloom_status_t status = candidate->host.build(
        std::move(key), std::move(positions), std::move(point_positions), std::move(point_values),
        std::move(point_gammas), std::move(periodic_shifts), std::move(periodic_response), token,
        error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    auto topology_diagnostic = Gfn2SccSetupTopology::create(
        candidate->host.basis, candidate->host.integrals, candidate->host.wavefunction_layout,
        token, candidate->topology_owner);
    if (!topology_diagnostic.success()) {
      error = setup_error_message("CUDA topology owner construction", topology_diagnostic.status,
                                  static_cast<std::uint32_t>(topology_diagnostic.error),
                                  static_cast<std::uint32_t>(topology_diagnostic.field),
                                  topology_diagnostic.index);
      return topology_diagnostic.status;
    }
    cudaError_t cuda_status = candidate->topology_arena.allocate(
        candidate->topology_owner.requirements().immutable_device_bytes);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA topology arena allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    topology_diagnostic = candidate->topology_owner.bind_device_arena_and_upload_async(
        candidate->topology_arena.get(), candidate->topology_arena.bytes(),
        candidate->device_topology, candidate->device_wavefunction, stream);
    if (!topology_diagnostic.success()) {
      error = setup_error_message("CUDA topology upload", topology_diagnostic.status,
                                  static_cast<std::uint32_t>(topology_diagnostic.error),
                                  static_cast<std::uint32_t>(topology_diagnostic.field),
                                  topology_diagnostic.index);
      return topology_diagnostic.status;
    }
    candidate->submitted = true;

    auto input_diagnostic = Gfn2SccSetupInputs::create(candidate->host.input_sources(),
                                                       candidate->topology_owner.host_topology(),
                                                       token, candidate->inputs_owner);
    if (!input_diagnostic.success()) {
      error = setup_error_message("CUDA input owner construction", input_diagnostic.status,
                                  static_cast<std::uint32_t>(input_diagnostic.error),
                                  static_cast<std::uint32_t>(input_diagnostic.field),
                                  input_diagnostic.index);
      return input_diagnostic.status;
    }
    cuda_status =
        candidate->input_arena.allocate(candidate->inputs_owner.requirements().device_bytes);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA input arena allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    input_diagnostic = candidate->inputs_owner.bind_device_arena_and_upload_async(
        candidate->device_topology, candidate->device_wavefunction, candidate->input_arena.get(),
        candidate->input_arena.bytes(), candidate->plan_seed, candidate->input_seed, stream);
    if (!input_diagnostic.success()) {
      error = setup_error_message("CUDA input upload", input_diagnostic.status,
                                  static_cast<std::uint32_t>(input_diagnostic.error),
                                  static_cast<std::uint32_t>(input_diagnostic.field),
                                  input_diagnostic.index);
      return input_diagnostic.status;
    }

    Gfn2EigensolverOptions eigensolver_options = candidate->plan_seed.eigensolver_options;
    eigensolver_options.jacobi = solver_jacobi;
    auto eigensolver_diagnostic = Gfn2SccSetupEigensolver::create(
        candidate->topology_owner, candidate->host.overlap.data(),
        static_cast<std::int64_t>(candidate->host.overlap.size()),
        candidate->host.geometry_generation, token, solver, solver_parameters, blas,
        eigensolver_options, candidate->eigensolver_owner);
    if (!eigensolver_diagnostic.success()) {
      error = setup_error_message(
          "CUDA eigensolver owner construction", eigensolver_diagnostic.status,
          static_cast<std::uint32_t>(eigensolver_diagnostic.error),
          static_cast<std::uint32_t>(eigensolver_diagnostic.field), eigensolver_diagnostic.index);
      return eigensolver_diagnostic.status;
    }
    const auto& eigensolver_requirements = candidate->eigensolver_owner.requirements();
    const auto arena_diagnostic = query_gfn2_scc_iteration_arena_requirements_cuda(
        candidate->plan_seed, eigensolver_requirements.provider, candidate->iteration_requirements);
    if (!arena_diagnostic.success()) {
      error = "CUDA SCC iteration arena query rejected the composed production plan";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    cuda_status =
        candidate->iteration_arena.allocate(candidate->iteration_requirements.total_bytes);
    if (cuda_status == cudaSuccess) {
      cuda_status = candidate->provider_host_workspace.allocate(
          eigensolver_requirements.provider.solver_host_workspace_bytes);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA SCC iteration/provider workspace allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    const auto bind_diagnostic = bind_gfn2_scc_iteration_arena_cuda(
        candidate->plan_seed, eigensolver_requirements.provider, candidate->iteration_requirements,
        candidate->iteration_arena.get(), candidate->iteration_arena.bytes(),
        candidate->provider_host_workspace.get(), candidate->provider_host_workspace.bytes(),
        candidate->state_seed, candidate->workspace_seed, candidate->report_storage);
    if (!bind_diagnostic.success()) {
      error = "CUDA SCC iteration arena binding rejected its own sealed requirements";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    status = build_numerical_refresh_binding(*candidate, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    cuda_status =
        candidate->eigensolver_setup_arena.allocate(eigensolver_requirements.setup_device_bytes);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA eigensolver setup arena allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    eigensolver_diagnostic = candidate->eigensolver_owner.bind_and_factor_overlap_async(
        candidate->device_topology, candidate->plan_seed, candidate->iteration_requirements,
        candidate->iteration_arena.get(), candidate->iteration_arena.bytes(),
        candidate->workspace_seed, candidate->provider_host_workspace.get(),
        candidate->provider_host_workspace.bytes(), candidate->eigensolver_setup_arena.get(),
        candidate->eigensolver_setup_arena.bytes(), candidate->eigensolver_binding, stream);
    if (!eigensolver_diagnostic.success()) {
      error = setup_error_message(
          "CUDA overlap factorization submission", eigensolver_diagnostic.status,
          static_cast<std::uint32_t>(eigensolver_diagnostic.error),
          static_cast<std::uint32_t>(eigensolver_diagnostic.field), eigensolver_diagnostic.index);
      return eigensolver_diagnostic.status;
    }
    candidate->plan_seed.eigensolver_batch = candidate->eigensolver_binding.batch;
    candidate->plan_seed.eigensolver_provider = candidate->eigensolver_binding.provider;
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
    if (consume_execution_test_fault(
            Gfn2CudaExecutionTestFault::kSccProviderUncapturedFallback)) {
      candidate->plan_seed.eigensolver_provider.capture_mode =
          Gfn2SccIterationProviderCaptureMode::kUncapturedSegmentRequired;
    }
#endif
    candidate->plan_seed.overlap_cache = candidate->eigensolver_binding.cache;
    candidate->plan_seed.eigensolver_options = candidate->eigensolver_binding.options;

    Gfn2SccIterationHostInitialization initialization = candidate->host.initialization();
    /* Mixed-spin fresh checkpoints must carry the same setup-sealed host
     * layout that produced the device plan; physical offsets alone cannot
     * reconstruct per-system spin-major extents. */
    initialization.wavefunction_layout = candidate->topology_owner.host_wavefunction_layout();
    auto initialization_diagnostic = Gfn2SccIterationInitializer::create(
        candidate->plan_seed, candidate->iteration_requirements, candidate->iteration_arena.get(),
        candidate->iteration_arena.bytes(), candidate->state_seed, candidate->workspace_seed,
        candidate->report_storage, initialization, candidate->initializer);
    if (!initialization_diagnostic.success()) {
      error =
          setup_error_message("CUDA SCC initializer construction", initialization_diagnostic.status,
                              static_cast<std::uint32_t>(initialization_diagnostic.error),
                              static_cast<std::uint32_t>(initialization_diagnostic.field),
                              initialization_diagnostic.index);
      return initialization_diagnostic.status;
    }
    initialization_diagnostic = candidate->initializer.upload_async(
        candidate->iteration_arena.get(), candidate->iteration_arena.bytes(), candidate->ready,
        stream);
    if (!initialization_diagnostic.success()) {
      error = setup_error_message("CUDA SCC initializer upload", initialization_diagnostic.status,
                                  static_cast<std::uint32_t>(initialization_diagnostic.error),
                                  static_cast<std::uint32_t>(initialization_diagnostic.field),
                                  initialization_diagnostic.index);
      return initialization_diagnostic.status;
    }
    const auto report_diagnostic = build_gfn2_scc_iteration_report_binding_cuda(
        candidate->report_storage, candidate->plan_seed, candidate->input_seed,
        candidate->state_seed, candidate->workspace_seed, candidate->scc_binding);
    if (report_diagnostic.error != Gfn2SccIterationBindingError::kSuccess) {
      error = "CUDA SCC report factory rejected the composed runtime binding";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    status = build_energy_force_bindings(*candidate, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = build_inference_bindings(*candidate, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    /* Numerical refresh is built first because inference consumes its epoch
     * descriptors. Close the reverse failure-invalidation edge only after the
     * inference arena owns the stable checkpoint array. */
    candidate->numerical.device.warm_checkpoint_generations =
        candidate->inference.warm_checkpoint_generations;
    status = build_public_result_state(*candidate, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    status = validate_candidate_setup(*candidate, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    const Gfn2SccLoopGraphBuildResult loop_graph =
        candidate->scc_loop.build(candidate->scc_binding, candidate->inference.epoch_consumer);
    if (!loop_graph.success()) {
      std::ostringstream message;
      message << "CUDA SCC conditional Graph setup rejected the production binding: status="
              << static_cast<std::uint32_t>(loop_graph.status)
              << " iteration_status=" << static_cast<std::uint32_t>(loop_graph.iteration.status)
              << " stage=" << static_cast<std::uint32_t>(loop_graph.iteration.stage)
              << " cuda=" << static_cast<int>(loop_graph.cuda_status);
      error = message.str();
      return loop_graph.iteration.status == Gfn2SccIterationLaunchStatus::kInvalidBinding
                 ? XTBLOOM_STATUS_INVALID_ARGUMENT
                 : XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    /* The setup owner factors its deterministic topology-only seed at
     * generation 1 so setup and graph validation can exercise a usable
     * overlap cache.  The externally visible numerical runtime, however,
     * starts unpublished at epoch 0.  Invalidate only the seed provenance
     * before publishing the candidate so the first real epoch-1 refresh must
     * refactor the caller geometry instead of mistaking the seed factor for a
     * cache hit. */
    cuda_status = cudaMemsetAsync(
        candidate->eigensolver_binding.cache.geometry_generations, 0,
        static_cast<std::size_t>(candidate->host.basis.batch_size) * sizeof(std::uint64_t), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA topology-only overlap cache invalidation", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    candidate->submitted = true;
    status = build_request_execution_graph(*candidate, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    output = std::move(candidate);
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t validate_public_request_pointers(const xtbloom_batch_t& batch,
                                                    const xtbloom_compute_options_t& options,
                                                    const xtbloom_batch_result_t* result,
                                                    std::string& error) {
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    std::int64_t cell_elements = 0;
    std::int64_t dipole_elements = 0;
    const bool lattice_active =
        batch.struct_size >= XTBLOOM_BATCH_V4_SIZE &&
        (batch.cell_matrices.data != nullptr || batch.cell_matrices.size_bytes != 0u ||
         batch.periodic_axes.data != nullptr || batch.periodic_axes.size_bytes != 0u);
    const bool interaction_suffix_present = batch.struct_size >= XTBLOOM_BATCH_V3_SIZE;
    const bool interaction_descriptors_active =
        interaction_suffix_present &&
        (batch.interaction_descriptors.data != nullptr ||
         batch.interaction_descriptors.size_bytes != 0u);
    const bool interaction_payload_active =
        interaction_suffix_present &&
        (batch.interaction_payload.data != nullptr || batch.interaction_payload.size_bytes != 0u);
    if (!checked_elements(batch.total_atoms, 3, coordinates) ||
        !checked_elements(batch.total_point_charges, 3, point_coordinates) ||
        !checked_elements(batch.batch_size, 3, dipole_elements) ||
        (lattice_active && !checked_elements(batch.batch_size, 9, cell_elements))) {
      error = "CUDA public descriptor coordinate extent overflows int64_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    const bool point_topology_active = batch.total_point_charges != 0 ||
                                       batch.point_charge_offsets.data != nullptr ||
                                       batch.point_charge_offsets.size_bytes != 0u;
    const bool shift_active = batch.atomic_potential_shifts.data != nullptr ||
                              batch.atomic_potential_shifts.size_bytes != 0u;
    const bool response_active = batch.total_charge_response_elements != 0 ||
                                 batch.charge_response_offsets.data != nullptr ||
                                 batch.charge_response_offsets.size_bytes != 0u ||
                                 batch.charge_response_matrix.data != nullptr ||
                                 batch.charge_response_matrix.size_bytes != 0u;
    CudaValidatedConstBuffer validated_const{};
    CudaValidatedBuffer validated_output{};
    const auto validate_const = [&](const char* name, const xtbloom_const_buffer_t& buffer,
                                    std::int64_t elements, std::size_t element_size,
                                    std::size_t alignment) -> xtbloom_status_t {
      std::size_t bytes = 0u;
      if (!checked_bytes(elements, element_size, bytes)) {
        error = std::string(name) + " extent overflows size_t";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      return validate_cuda_const_buffer(device_id, name, buffer, bytes, alignment,
                                        CudaManagedMemoryPolicy::kReject, validated_const, error);
    };
    const auto validate_output = [&](const char* name, const xtbloom_buffer_t& buffer,
                                     std::int64_t elements, std::size_t element_size,
                                     std::size_t alignment) -> xtbloom_status_t {
      std::size_t bytes = 0u;
      if (!checked_bytes(elements, element_size, bytes)) {
        error = std::string(name) + " extent overflows size_t";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      return validate_cuda_buffer(device_id, name, buffer, bytes, alignment,
                                  CudaManagedMemoryPolicy::kReject, validated_output, error);
    };

    xtbloom_status_t status =
        validate_const("atom_offsets", batch.atom_offsets, batch.batch_size + 1,
                       sizeof(std::int64_t), alignof(std::int64_t));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("atomic_numbers", batch.atomic_numbers, batch.total_atoms,
                            sizeof(std::int32_t), alignof(std::int32_t));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status =
        validate_const("positions", batch.positions, coordinates, sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("molecular_charges", batch.molecular_charges, batch.batch_size,
                            sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("unpaired_electrons", batch.unpaired_electrons, batch.batch_size,
                            sizeof(std::int32_t), alignof(std::int32_t));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    const bool spin_present =
        batch.struct_size >= XTBLOOM_BATCH_V2_SIZE &&
        (batch.spin_channels.data != nullptr || batch.spin_channels.size_bytes != 0u);
    const xtbloom_const_buffer_t absent_spin{};
    const xtbloom_const_buffer_t& spin_buffer = spin_present ? batch.spin_channels : absent_spin;
    status = validate_const("spin_channels", spin_buffer, spin_present ? batch.batch_size : 0,
                            sizeof(std::int32_t), alignof(std::int32_t));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("point_charge_offsets", batch.point_charge_offsets,
                            point_topology_active ? batch.batch_size + 1 : 0, sizeof(std::int64_t),
                            alignof(std::int64_t));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("point_charge_positions", batch.point_charge_positions,
                            point_coordinates, sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("point_charge_values", batch.point_charge_values,
                            batch.total_point_charges, sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("point_charge_gammas", batch.point_charge_gammas,
                            batch.total_point_charges, sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("atomic_potential_shifts", batch.atomic_potential_shifts,
                            shift_active ? batch.total_atoms : 0, sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("charge_response_offsets", batch.charge_response_offsets,
                            response_active ? batch.batch_size + 1 : 0, sizeof(std::int64_t),
                            alignof(std::int64_t));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("charge_response_matrix", batch.charge_response_matrix,
                            response_active ? batch.total_charge_response_elements : 0,
                            sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    const xtbloom_const_buffer_t absent_interaction{};
    const xtbloom_const_buffer_t& interaction_descriptor_buffer =
        interaction_descriptors_active ? batch.interaction_descriptors : absent_interaction;
    const xtbloom_const_buffer_t& interaction_payload_buffer =
        interaction_payload_active ? batch.interaction_payload : absent_interaction;
    status = validate_const("interaction_descriptors", interaction_descriptor_buffer,
                            interaction_descriptors_active ? batch.total_interactions : 0,
                            sizeof(xtbloom_interaction_t), alignof(std::byte));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_cuda_const_buffer(
        device_id, "interaction_payload", interaction_payload_buffer,
        interaction_payload_active ? interaction_payload_buffer.size_bytes : 0u,
        alignof(std::byte),
        CudaManagedMemoryPolicy::kReject, validated_const, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    const xtbloom_const_buffer_t absent_lattice{};
    const xtbloom_const_buffer_t& cell_buffer =
        lattice_active ? batch.cell_matrices : absent_lattice;
    const xtbloom_const_buffer_t& axes_buffer =
        lattice_active ? batch.periodic_axes : absent_lattice;
    status = validate_const("cell_matrices", cell_buffer, lattice_active ? cell_elements : 0,
                            sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("periodic_axes", axes_buffer, lattice_active ? batch.batch_size : 0,
                            sizeof(std::int32_t), alignof(std::int32_t));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    if (result == nullptr) return XTBLOOM_STATUS_SUCCESS;

    const bool energy_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ENERGY)) != 0u;
    const bool force_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_FORCES)) != 0u;
    const bool charges_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ATOMIC_CHARGES)) != 0u;
    const bool point_forces_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_POINT_CHARGE_FORCES)) != 0u;
    const bool dipoles_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_DIPOLE_MOMENTS)) != 0u;
    status = validate_output("energies", result->energies, energy_requested ? batch.batch_size : 0,
                             sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_output("forces", result->forces, force_requested ? coordinates : 0,
                             sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status =
        validate_output("atomic_charges", result->atomic_charges,
                        charges_requested ? batch.total_atoms : 0, sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_output("point_charge_forces", result->point_charge_forces,
                             point_forces_requested ? point_coordinates : 0, sizeof(double),
                             alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    const xtbloom_buffer_t absent_output{};
    const xtbloom_buffer_t& dipole_buffer =
        result->struct_size >= XTBLOOM_BATCH_RESULT_V2_SIZE ? result->dipole_moments : absent_output;
    status = validate_output("dipole_moments", dipole_buffer,
                             dipoles_requested ? dipole_elements : 0, sizeof(double),
                             alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_output("scc_iterations", result->scc_iterations, batch.batch_size,
                             sizeof(std::int32_t), alignof(std::int32_t));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_output("scc_converged", result->scc_converged, batch.batch_size,
                             sizeof(std::uint8_t), alignof(std::uint8_t));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    return validate_output("per_system_status", result->per_system_status, batch.batch_size,
                           sizeof(xtbloom_status_t), alignof(xtbloom_status_t));
  }

  xtbloom_status_t validate_native_lattice_request_sync(const xtbloom_batch_t& batch,
                                                        std::string& error) {
    if (native_lattice_staging_poisoned) {
      error = "CUDA native-cell staging is poisoned by an earlier stream-settlement failure";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    if (!native_lattice_active(batch)) {
      return XTBLOOM_STATUS_SUCCESS;
    }
    if (native_lattice_staging_pending) {
      const cudaError_t settle_status = cudaStreamSynchronize(stream);
      if (settle_status != cudaSuccess) {
        native_lattice_staging_poisoned = true;
        error = cuda_error_message("CUDA native-cell staging settlement", settle_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      native_lattice_staging_pending = false;
    }

    std::int64_t cell_elements = 0;
    if (!checked_elements(batch.batch_size, 9, cell_elements)) {
      error = "CUDA native-cell extent overflows int64_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    std::size_t cell_bytes = 0u;
    std::size_t axes_bytes = 0u;
    if (!checked_bytes(cell_elements, sizeof(double), cell_bytes) ||
        !checked_bytes(batch.batch_size, sizeof(std::int32_t), axes_bytes)) {
      error = "CUDA native-cell extent overflows size_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    const bool cells_host = batch.cell_matrices.memory_space == XTBLOOM_MEMORY_HOST;
    const bool axes_host = batch.periodic_axes.memory_space == XTBLOOM_MEMORY_HOST;
    if (cells_host && axes_host) {
      return validate_host_native_lattice_request(batch, error);
    }

    if (native_lattice_event.get() == nullptr) {
      const cudaError_t event_status = native_lattice_event.create(cudaEventDisableTiming);
      if (event_status != cudaSuccess) {
        error = cuda_error_message("CUDA native-cell validation event creation", event_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
    }

    ArenaLayout layout;
    const std::size_t cell_offset = layout.append<double>(cell_elements);
    const std::size_t axes_offset = layout.append<std::int32_t>(batch.batch_size);
    if (!layout.valid()) {
      error = "CUDA native-cell validation staging layout overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    if (native_lattice_host_arena.bytes() < layout.bytes()) {
      bool inject_allocation_failure = false;
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
      inject_allocation_failure =
          consume_execution_test_fault(Gfn2CudaExecutionTestFault::kNativeLatticePinnedAllocation);
      if (inject_allocation_failure) {
        g_native_lattice_allocation_faults.fetch_add(1u, std::memory_order_relaxed);
      }
#endif
      const cudaError_t allocation_status =
          native_lattice_host_arena.allocate(layout.bytes(), inject_allocation_failure);
      if (allocation_status != cudaSuccess) {
        error =
            cuda_error_message("CUDA native-cell validation staging allocation", allocation_status);
        return XTBLOOM_STATUS_ALLOCATION_FAILED;
      }
    }
    void* const arena = native_lattice_host_arena.get();
    auto* const staged_cells = arena_pointer<double>(arena, cell_offset);
    auto* const staged_axes = arena_pointer<std::int32_t>(arena, axes_offset);
    if (cells_host) {
      std::memcpy(staged_cells, batch.cell_matrices.data, cell_bytes);
    } else {
      const cudaError_t copy_status = cudaMemcpyAsync(staged_cells, batch.cell_matrices.data,
                                                      cell_bytes, cudaMemcpyDeviceToHost, stream);
      if (copy_status != cudaSuccess) {
        error = cuda_error_message("CUDA cell-matrix validation copy", copy_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      native_lattice_staging_pending = true;
    }
    if (axes_host) {
      std::memcpy(staged_axes, batch.periodic_axes.data, axes_bytes);
    } else {
      const cudaError_t copy_status = cudaMemcpyAsync(staged_axes, batch.periodic_axes.data,
                                                      axes_bytes, cudaMemcpyDeviceToHost, stream);
      if (copy_status != cudaSuccess) {
        error = cuda_error_message("CUDA periodic-axis validation copy", copy_status);
        if (native_lattice_staging_pending) {
          const cudaError_t settle_status = cudaStreamSynchronize(stream);
          if (settle_status == cudaSuccess) {
            native_lattice_staging_pending = false;
          } else {
            native_lattice_staging_poisoned = true;
            error += "; additionally, " +
                     cuda_error_message("CUDA native-cell staging settlement", settle_status);
          }
        }
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      native_lattice_staging_pending = true;
    }
    const cudaError_t record_status = cudaEventRecord(native_lattice_event.get(), stream);
    if (record_status != cudaSuccess) {
      error = cuda_error_message("CUDA native-cell validation event", record_status);
      if (native_lattice_staging_pending) {
        const cudaError_t settle_status = cudaStreamSynchronize(stream);
        if (settle_status == cudaSuccess) {
          native_lattice_staging_pending = false;
        } else {
          native_lattice_staging_poisoned = true;
          error += "; additionally, " +
                   cuda_error_message("CUDA native-cell staging settlement", settle_status);
        }
      }
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    cudaError_t wait_status = cudaSuccess;
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
    if (consume_execution_test_fault(Gfn2CudaExecutionTestFault::kNativeLatticeCompletionWait)) {
      g_native_lattice_completion_faults.fetch_add(1u, std::memory_order_relaxed);
      wait_status = cudaErrorUnknown;
    } else
#endif
    {
      wait_status = cudaEventSynchronize(native_lattice_event.get());
    }
    if (wait_status != cudaSuccess) {
      native_lattice_staging_poisoned = true;
      error = cuda_error_message("CUDA native-cell validation completion", wait_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    native_lattice_staging_pending = false;

    xtbloom_batch_t staged_batch{};
    std::memcpy(&staged_batch, &batch,
                std::min<std::size_t>(batch.struct_size, sizeof(staged_batch)));
    staged_batch.cell_matrices = {staged_cells, cell_bytes, XTBLOOM_MEMORY_HOST, 0u};
    staged_batch.periodic_axes = {staged_axes, axes_bytes, XTBLOOM_MEMORY_HOST, 0u};
    return validate_host_native_lattice_request(staged_batch, error);
  }

  xtbloom_status_t reset_request_topology_error_locked(Prepared& current, std::string& error) {
    if (!current.public_result.ready || current.public_result.request_topology_error == nullptr) {
      error = "CUDA fixed-topology request gate is not initialized";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const cudaError_t cuda_status = cudaMemsetAsync(current.public_result.request_topology_error, 0,
                                                    sizeof(std::uint32_t), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA fixed-topology request gate reset", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    current.submitted = true;
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t enqueue_fixed_topology_validation_locked(
      Prepared& current, const xtbloom_batch_t& batch, const xtbloom_compute_options_t& options,
      std::string& error) {
    const TopologyKey& key = current.host.key;
    const std::int64_t expected_batch = static_cast<std::int64_t>(key.molecular_charges.size());
    const std::int64_t expected_atoms = static_cast<std::int64_t>(key.atomic_numbers.size());
    const std::int64_t expected_points = key.point_offsets.empty() ? 0 : key.point_offsets.back();
    const std::int64_t expected_response =
        key.response_offsets.empty() ? 0 : key.response_offsets.back();
    const bool periodic_enabled = batch.atomic_potential_shifts.data != nullptr ||
                                  batch.atomic_potential_shifts.size_bytes != 0u ||
                                  batch.total_charge_response_elements != 0 ||
                                  batch.charge_response_offsets.data != nullptr ||
                                  batch.charge_response_offsets.size_bytes != 0u ||
                                  batch.charge_response_matrix.data != nullptr ||
                                  batch.charge_response_matrix.size_bytes != 0u;
    if (batch.batch_size != expected_batch || batch.total_atoms != expected_atoms ||
        batch.total_point_charges != expected_points || options.model != XTBLOOM_MODEL_GFN2_XTB ||
        options.flags != key.flags || options.max_scc_iterations != key.maximum_iterations ||
        options.charge_tolerance != key.charge_tolerance ||
        options.energy_tolerance != key.energy_tolerance ||
        options.electronic_temperature != key.electronic_temperature ||
        public_scc_mixer(options) != key.scc_mixer ||
        public_scc_mixer_history(options) != key.scc_mixer_history ||
        public_scc_mixer_damping(options) != key.scc_mixer_damping ||
        public_determinism(options) != key.determinism ||
        periodic_enabled != key.periodic_enabled) {
      error = "the batch or compute policy does not match the fixed CUDA plan topology";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    const bool point_offsets_active = expected_points != 0 ||
                                      batch.point_charge_offsets.data != nullptr ||
                                      batch.point_charge_offsets.size_bytes != 0u;
    const bool response_active = batch.total_charge_response_elements != 0 ||
                                 batch.charge_response_offsets.data != nullptr ||
                                 batch.charge_response_offsets.size_bytes != 0u ||
                                 batch.charge_response_matrix.data != nullptr ||
                                 batch.charge_response_matrix.size_bytes != 0u;
    if (response_active && batch.total_charge_response_elements != expected_response) {
      error = "total_charge_response_elements does not match the fixed CUDA plan topology";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    const auto& state = current.public_result;
    FixedTopologyComparisonDeviceBinding binding{};
    binding.expected_atom_offsets = state.expected_atom_offsets;
    binding.expected_atomic_numbers = state.expected_atomic_numbers;
    binding.expected_molecular_charges = state.expected_molecular_charges;
    binding.expected_unpaired_electrons = state.expected_unpaired_electrons;
    binding.expected_spin_channels = state.expected_spin_channels;
    binding.expected_point_offsets = state.expected_point_offsets;
    binding.expected_response_offsets = state.expected_response_offsets;
    binding.atom_offset_elements = expected_batch + 1;
    binding.atomic_number_elements = expected_atoms;
    binding.batch_elements = expected_batch;
    binding.point_offset_elements = point_offsets_active ? expected_batch + 1 : 0;
    binding.response_offset_elements = response_active ? expected_batch + 1 : 0;
    binding.request_error = state.request_topology_error;

    if (!validate_or_bind_fixed_topology_field("atom_offsets", batch.atom_offsets, key.atom_offsets,
                                               state.expected_atom_offsets, binding.atom_offsets,
                                               error) ||
        !validate_or_bind_fixed_topology_field("atomic_numbers", batch.atomic_numbers,
                                               key.atomic_numbers, state.expected_atomic_numbers,
                                               binding.atomic_numbers, error) ||
        !validate_or_bind_fixed_topology_field(
            "molecular_charges", batch.molecular_charges, key.molecular_charges,
            state.expected_molecular_charges, binding.molecular_charges, error) ||
        !validate_or_bind_fixed_topology_field(
            "unpaired_electrons", batch.unpaired_electrons, key.unpaired_electrons,
            state.expected_unpaired_electrons, binding.unpaired_electrons, error)) {
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    const bool spin_supplied =
        batch.struct_size >= XTBLOOM_BATCH_V2_SIZE &&
        (batch.spin_channels.data != nullptr || batch.spin_channels.size_bytes != 0u);
    if (spin_supplied) {
      if (!validate_or_bind_fixed_topology_field("spin_channels", batch.spin_channels,
                                                 key.spin_channels, state.expected_spin_channels,
                                                 binding.spin_channels, error)) {
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    } else if (!std::all_of(key.spin_channels.begin(), key.spin_channels.end(),
                            [](std::int32_t channels) { return channels == 1; })) {
      error = "spin_channels does not match the fixed CUDA plan topology";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if (point_offsets_active &&
        !validate_or_bind_fixed_topology_field("point_charge_offsets", batch.point_charge_offsets,
                                               key.point_offsets, state.expected_point_offsets,
                                               binding.point_offsets, error)) {
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if (response_active &&
        !validate_or_bind_fixed_topology_field(
            "charge_response_offsets", batch.charge_response_offsets, key.response_offsets,
            state.expected_response_offsets, binding.response_offsets, error)) {
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    xtbloom_status_t status = reset_request_topology_error_locked(current, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    const cudaError_t cuda_status = compare_fixed_topology_async(binding, stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA fixed-topology comparison submission", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? XTBLOOM_STATUS_INVALID_ARGUMENT
                                                  : XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    current.submitted = true;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t enqueue_native_lattice_validation_locked(Prepared& current,
                                                            const xtbloom_batch_t& batch,
                                                            std::string& error) {
    if (batch.struct_size < XTBLOOM_BATCH_V4_SIZE ||
        (batch.cell_matrices.data == nullptr && batch.cell_matrices.size_bytes == 0u &&
         batch.periodic_axes.data == nullptr && batch.periodic_axes.size_bytes == 0u)) {
      return XTBLOOM_STATUS_SUCCESS;
    }
    const auto& state = current.public_result;
    if (state.staged_lattice_cells == nullptr || state.staged_periodic_axes == nullptr ||
        state.host_lattice_cells == nullptr || state.host_periodic_axes == nullptr) {
      error = "CUDA fixed-plan lattice validation storage is not initialized";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    const auto stage = [&](const xtbloom_const_buffer_t& source, void* host_stage,
                           void* device_stage, std::size_t bytes) -> cudaError_t {
      if (source.memory_space == XTBLOOM_MEMORY_HOST) {
        std::memcpy(host_stage, source.data, bytes);
        return cudaMemcpyAsync(device_stage, host_stage, bytes, cudaMemcpyHostToDevice, stream);
      }
      return cudaMemcpyAsync(device_stage, source.data, bytes, cudaMemcpyDeviceToDevice, stream);
    };
    const std::size_t cell_bytes = static_cast<std::size_t>(batch.batch_size) * 9u * sizeof(double);
    const std::size_t axes_bytes =
        static_cast<std::size_t>(batch.batch_size) * sizeof(std::int32_t);
    cudaError_t cuda_status = stage(batch.cell_matrices, state.host_lattice_cells,
                                    state.staged_lattice_cells, cell_bytes);
    if (cuda_status == cudaSuccess) {
      cuda_status = stage(batch.periodic_axes, state.host_periodic_axes, state.staged_periodic_axes,
                          axes_bytes);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA fixed-plan lattice staging", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    FixedTopologyComparisonDeviceBinding binding{};
    binding.cell_matrices = state.staged_lattice_cells;
    binding.periodic_axes = state.staged_periodic_axes;
    binding.lattice_systems = batch.batch_size;
    binding.request_error = state.request_topology_error;
    cuda_status = validate_lattice_request_async(binding, stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA fixed-plan lattice validation submission", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? XTBLOOM_STATUS_INVALID_ARGUMENT
                                                  : XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    current.submitted = true;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t stage_numerical_ingress_locked(Prepared& current,
                                                  const Gfn2CudaNumericalInputView& input,
                                                  bool& host_upload_enqueued,
                                                  std::string& error) {
    host_upload_enqueued = false;
    if (!current.numerical.ready) {
      error = "CUDA GFN2 numerical refresh requires a prepared fixed topology";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    auto& numerical = current.numerical;
    auto& device = numerical.device;
    if (device.refresh_predecessor_generations == nullptr ||
        device.warm_checkpoint_generations == nullptr) {
      error = "CUDA GFN2 numerical refresh has an incomplete warm-checkpoint binding";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    auto& preprocessing = numerical.preprocessing;
    cudaError_t cuda_status = cudaSetDevice(device_id);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("cudaSetDevice for numerical refresh", cuda_status);
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }

    /* Validate the complete view before enqueueing any transfer. A synchronous
     * descriptor rejection must not leave an earlier asynchronous host read in
     * flight after the caller is told that the refresh did not start. */
    const auto validate_source = [&](const char* name, const xtbloom_const_buffer_t& buffer,
                                     std::int64_t elements, const double* device_stage,
                                     const double* owned_host_stage,
                                     bool allow_absent_zero = false) -> xtbloom_status_t {
      std::size_t bytes = 0u;
      if (!checked_bytes(elements, sizeof(double), bytes)) {
        error = std::string(name) + " extent overflows size_t";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      const bool supported_space = buffer.memory_space == XTBLOOM_MEMORY_HOST ||
                                   buffer.memory_space == XTBLOOM_MEMORY_CUDA_DEVICE;
      const bool absent = buffer.data == nullptr && buffer.size_bytes == 0u;
      if (elements == 0) {
        if (buffer.reserved != 0u || !supported_space) {
          error = std::string(name) + " has malformed empty-buffer metadata";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        return XTBLOOM_STATUS_SUCCESS;
      }
      if (allow_absent_zero && absent) {
        if (buffer.reserved != 0u || !supported_space || device_stage == nullptr) {
          error = std::string(name) + " has malformed absent-buffer metadata";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        return XTBLOOM_STATUS_SUCCESS;
      }
      if (buffer.reserved != 0u || buffer.data == nullptr || buffer.size_bytes < bytes ||
          !supported_space) {
        error = std::string(name) + " is not a sufficiently large host/CUDA buffer";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      if (buffer.memory_space == XTBLOOM_MEMORY_HOST &&
          (device_stage == nullptr || owned_host_stage == nullptr)) {
        error = std::string(name) + " has no runtime host-staging projection";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      return XTBLOOM_STATUS_SUCCESS;
    };

    xtbloom_status_t status =
        validate_source("positions", input.positions, device.total_atoms * 3,
                        numerical.host_positions, numerical.owned_host_positions);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_source("point_charge_positions", input.point_charge_positions,
                             device.total_point_charges * 3, numerical.host_point_positions,
                             numerical.owned_host_point_positions);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_source("point_charge_values", input.point_charge_values,
                             device.total_point_charges, numerical.host_point_values,
                             numerical.owned_host_point_values);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_source("point_charge_gammas", input.point_charge_gammas,
                             device.total_point_charges, numerical.host_point_gammas,
                             numerical.owned_host_point_gammas);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status =
        validate_source("atomic_potential_shifts", input.atomic_potential_shifts,
                        device.periodic_enabled != 0u ? device.total_atoms : 0,
                        numerical.host_periodic_shifts, numerical.owned_host_periodic_shifts, true);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_source("charge_response_matrix", input.charge_response_matrix,
                             device.periodic_enabled != 0u ? device.total_response_elements : 0,
                             numerical.host_periodic_response,
                             numerical.owned_host_periodic_response, true);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    std::size_t mask_bytes = 0u;
    if (!checked_bytes(device.batch_size, sizeof(std::uint8_t), mask_bytes)) {
      error = "numerical refresh activity extent overflows size_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    auto* const requested = const_cast<std::uint8_t*>(preprocessing.activity.requested_mask);
    const bool absent_mask =
        input.requested_mask.data == nullptr && input.requested_mask.size_bytes == 0u;
    const bool valid_mask_space = input.requested_mask.memory_space == XTBLOOM_MEMORY_HOST ||
                                  input.requested_mask.memory_space == XTBLOOM_MEMORY_CUDA_DEVICE;
    if ((absent_mask && (input.requested_mask.reserved != 0u || !valid_mask_space)) ||
        (!absent_mask &&
         (input.requested_mask.reserved != 0u || input.requested_mask.data == nullptr ||
          input.requested_mask.size_bytes < mask_bytes || !valid_mask_space))) {
      error = "requested_mask is not a sufficiently large host/CUDA uint8 buffer";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    const auto is_host_source = [](const xtbloom_const_buffer_t& buffer,
                                   std::int64_t elements) noexcept {
      return elements != 0 && buffer.data != nullptr && buffer.memory_space == XTBLOOM_MEMORY_HOST;
    };
    const bool uses_host_staging =
        is_host_source(input.positions, device.total_atoms * 3) ||
        is_host_source(input.point_charge_positions, device.total_point_charges * 3) ||
        is_host_source(input.point_charge_values, device.total_point_charges) ||
        is_host_source(input.point_charge_gammas, device.total_point_charges) ||
        is_host_source(input.atomic_potential_shifts,
                       device.periodic_enabled != 0u ? device.total_atoms : 0) ||
        is_host_source(input.charge_response_matrix,
                       device.periodic_enabled != 0u ? device.total_response_elements : 0) ||
        (!absent_mask && input.requested_mask.memory_space == XTBLOOM_MEMORY_HOST);

    if (uses_host_staging) {
      cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
      cuda_status = cudaStreamIsCapturing(stream, &capture_status);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA numerical host-upload capture query", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      if (capture_status != cudaStreamCaptureStatusNone) {
        error = "CUDA Graph numerical refresh requires CUDA-device input buffers";
        return XTBLOOM_STATUS_NOT_SUPPORTED;
      }
      if (numerical.host_staging_poisoned) {
        error = "CUDA numerical host staging is unavailable after an earlier completion failure";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      if (current.numerical_host_upload_completion.pending.load(std::memory_order_acquire)) {
        error = "a previous CUDA numerical host upload is still in flight";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }

    const auto copy_to_owned_host = [&](const char* name, const xtbloom_const_buffer_t& buffer,
                                        std::int64_t elements,
                                        double* destination) -> xtbloom_status_t {
      if (elements == 0 || buffer.data == nullptr || buffer.memory_space != XTBLOOM_MEMORY_HOST) {
        return XTBLOOM_STATUS_SUCCESS;
      }
      std::size_t bytes = 0u;
      if (!checked_bytes(elements, sizeof(double), bytes)) {
        error = std::string(name) + " extent changed after validation";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      std::memcpy(destination, buffer.data, bytes);
      return XTBLOOM_STATUS_SUCCESS;
    };
    status = copy_to_owned_host("positions", input.positions, device.total_atoms * 3,
                                numerical.owned_host_positions);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status =
        copy_to_owned_host("point_charge_positions", input.point_charge_positions,
                           device.total_point_charges * 3, numerical.owned_host_point_positions);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = copy_to_owned_host("point_charge_values", input.point_charge_values,
                                device.total_point_charges, numerical.owned_host_point_values);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = copy_to_owned_host("point_charge_gammas", input.point_charge_gammas,
                                device.total_point_charges, numerical.owned_host_point_gammas);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = copy_to_owned_host("atomic_potential_shifts", input.atomic_potential_shifts,
                                device.periodic_enabled != 0u ? device.total_atoms : 0,
                                numerical.owned_host_periodic_shifts);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = copy_to_owned_host("charge_response_matrix", input.charge_response_matrix,
                                device.periodic_enabled != 0u ? device.total_response_elements : 0,
                                numerical.owned_host_periodic_response);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    if (!absent_mask && input.requested_mask.memory_space == XTBLOOM_MEMORY_HOST) {
      std::memcpy(numerical.owned_host_requested, input.requested_mask.data, mask_bytes);
    }

    const auto seal_host_uploads = [&]() -> cudaError_t {
      if (!host_upload_enqueued) return cudaSuccess;
      auto& completion = current.numerical_host_upload_completion;
      /* Publish busy before recording completion: sufficiently short H2Ds may
       * let the private-stream callback run before this function returns. */
      completion.pending.store(true, std::memory_order_release);
      cudaError_t completion_status =
          cudaEventRecord(current.numerical_host_upload_complete.get(), stream);
      if (completion_status == cudaSuccess) {
        completion_status = cudaStreamWaitEvent(current.numerical_host_completion_stream.get(),
                                                current.numerical_host_upload_complete.get(), 0u);
      }
      if (completion_status == cudaSuccess) {
        completion_status = cudaLaunchHostFunc(current.numerical_host_completion_stream.get(),
                                               release_numerical_host_upload, &completion);
      }
      if (completion_status == cudaSuccess) {
        /* Request completion can wait on this event so the public event also
         * proves that the private callback no longer retains Prepared-owned
         * state. Ordinary query/wait then needs no second host stream fence. */
        completion_status = cudaEventRecord(current.numerical_host_release_complete.get(),
                                            current.numerical_host_completion_stream.get());
      }
      if (completion_status == cudaSuccess) {
        current.submitted = true;
      } else {
        /* Earlier H2Ds may already reference the pinned image.  Leave pending
         * set and poison host staging rather than making an unsafe retry. */
        numerical.host_staging_poisoned = true;
      }
      return completion_status;
    };

    const auto resolve_validated = [&](const char* name, const xtbloom_const_buffer_t& buffer,
                                       std::int64_t elements, double* device_stage,
                                       const double* owned_host_stage, const double*& source,
                                       bool allow_absent_zero = false) -> xtbloom_status_t {
      if (elements == 0) {
        source = nullptr;
        return XTBLOOM_STATUS_SUCCESS;
      }
      std::size_t bytes = 0u;
      if (!checked_bytes(elements, sizeof(double), bytes)) {
        error = std::string(name) + " extent changed after validation";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      if (allow_absent_zero && buffer.data == nullptr && buffer.size_bytes == 0u) {
        cuda_status = cudaMemsetAsync(device_stage, 0, bytes, stream);
        if (cuda_status != cudaSuccess) {
          error = cuda_error_message(name, cuda_status);
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        current.submitted = true;
        source = device_stage;
        return XTBLOOM_STATUS_SUCCESS;
      }
      if (buffer.memory_space == XTBLOOM_MEMORY_HOST) {
        cuda_status =
            cudaMemcpyAsync(device_stage, owned_host_stage, bytes, cudaMemcpyHostToDevice, stream);
        if (cuda_status != cudaSuccess) {
          const cudaError_t completion_status = seal_host_uploads();
          error = cuda_error_message(name, cuda_status);
          if (completion_status != cudaSuccess) {
            error += "; host-upload completion enqueue also failed";
          }
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        host_upload_enqueued = true;
        current.submitted = true;
        source = device_stage;
      } else {
        source = static_cast<const double*>(buffer.data);
      }
      return XTBLOOM_STATUS_SUCCESS;
    };

    NumericalRefreshDeviceSources sources{};
    status = resolve_validated("positions", input.positions, device.total_atoms * 3,
                               numerical.host_positions, numerical.owned_host_positions,
                               sources.positions);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = resolve_validated("point_charge_positions", input.point_charge_positions,
                               device.total_point_charges * 3, numerical.host_point_positions,
                               numerical.owned_host_point_positions, sources.point_positions);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = resolve_validated("point_charge_values", input.point_charge_values,
                               device.total_point_charges, numerical.host_point_values,
                               numerical.owned_host_point_values, sources.point_values);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = resolve_validated("point_charge_gammas", input.point_charge_gammas,
                               device.total_point_charges, numerical.host_point_gammas,
                               numerical.owned_host_point_gammas, sources.point_gammas);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = resolve_validated("atomic_potential_shifts", input.atomic_potential_shifts,
                               device.periodic_enabled != 0u ? device.total_atoms : 0,
                               numerical.host_periodic_shifts, numerical.owned_host_periodic_shifts,
                               sources.periodic_shifts, true);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status =
        resolve_validated("charge_response_matrix", input.charge_response_matrix,
                          device.periodic_enabled != 0u ? device.total_response_elements : 0,
                          numerical.host_periodic_response, numerical.owned_host_periodic_response,
                          sources.periodic_response, true);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    if (absent_mask) {
      cuda_status = cudaMemsetAsync(requested, 1, mask_bytes, stream);
    } else if (input.requested_mask.memory_space == XTBLOOM_MEMORY_HOST) {
      cuda_status = cudaMemcpyAsync(numerical.host_requested, numerical.owned_host_requested,
                                    mask_bytes, cudaMemcpyHostToDevice, stream);
      if (cuda_status == cudaSuccess) {
        host_upload_enqueued = true;
        current.submitted = true;
        cuda_status = cudaMemcpyAsync(requested, numerical.host_requested, mask_bytes,
                                      cudaMemcpyDeviceToDevice, stream);
      }
    } else {
      cuda_status = cudaMemcpyAsync(requested, input.requested_mask.data, mask_bytes,
                                    cudaMemcpyDeviceToDevice, stream);
    }
    if (cuda_status != cudaSuccess) {
      const cudaError_t completion_status = seal_host_uploads();
      error = cuda_error_message("CUDA numerical refresh activity staging", cuda_status);
      if (completion_status != cudaSuccess) {
        error += "; host-upload completion enqueue also failed";
      }
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    current.submitted = true;
    cuda_status = seal_host_uploads();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical host-upload completion enqueue", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    stage_gfn2_numerical_inputs_kernel<<<static_cast<unsigned int>(device.batch_size), 256, 0,
                                         stream>>>(device, sources);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical input staging", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    current.submitted = true;

    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t execute_numerical_body_locked(Prepared& current, cudaStream_t execution_stream,
                                                 std::string& error) {
    if (!current.numerical.ready) {
      error = "CUDA GFN2 numerical refresh requires a prepared fixed topology";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    auto& numerical = current.numerical;
    auto& device = numerical.device;
    auto& preprocessing = numerical.preprocessing;
    cudaError_t cuda_status = cudaSetDevice(device_id);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("cudaSetDevice for numerical refresh", cuda_status);
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }

    const auto preprocessing_launch =
        compose_gfn2_preprocessing_epoch_cuda(preprocessing, execution_stream);
    if (!preprocessing_launch.success()) {
      error = "CUDA numerical preprocessing composer rejected the runtime binding";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    if (device.d4_enabled != 0u) {
      cuda_status = reset_gfn2_d4_device_errors_cuda(
          device.batch_size, const_cast<std::uint32_t*>(device.d4_system_errors),
          const_cast<std::uint32_t*>(device.d4_device_error), execution_stream);
      if (cuda_status == cudaSuccess) {
        /* compose_gfn2_preprocessing_epoch_cuda commits the physical pair
         * superset earlier on this stream. The D4 refresh therefore observes
         * current role views without host polling, and its epoch overload is
         * safe under CUDA Graph replay. */
        cuda_status = update_gfn2_d4_pairlist_cache_cuda(
            current.plan_seed.d4_batch, current.plan_seed.d4_parameters,
            preprocessing.geometry_epoch, numerical.d4_refresh_cache, numerical.d4_workspace,
            const_cast<std::uint32_t*>(device.d4_device_error), execution_stream);
      }
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA D4 numerical refresh", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
    }
    if (device.point_enabled != 0u) {
      cuda_status = reset_gfn2_external_point_charge_scc_errors_cuda(
          device.batch_size, const_cast<std::uint32_t*>(device.point_system_errors),
          const_cast<std::uint32_t*>(device.point_plan_error), execution_stream);
      if (cuda_status == cudaSuccess) {
        initialize_gfn2_refresh_sequence_kernel<<<1, 1, 0, execution_stream>>>(
            const_cast<std::uint32_t*>(numerical.point_activity.sequence_active));
        cuda_status = cudaPeekAtLastError();
      }
      if (cuda_status == cudaSuccess) {
        cuda_status = update_gfn2_external_point_charge_scc_potential_cache_cuda(
            numerical.point_batch, numerical.point_activity, numerical.point_candidate,
            kInitialGeometryGeneration, numerical.point_workspace,
            const_cast<std::uint32_t*>(device.point_system_errors),
            const_cast<std::uint32_t*>(device.point_plan_error), execution_stream);
      }
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA explicit-point-charge numerical refresh", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
    }
    if (device.periodic_enabled != 0u) {
      cuda_status = cudaMemsetAsync(
          device.periodic_system_errors, 0,
          static_cast<std::size_t>(device.batch_size) * sizeof(std::uint32_t), execution_stream);
      if (cuda_status == cudaSuccess) {
        cuda_status =
            cudaMemsetAsync(device.periodic_plan_error, 0, sizeof(std::uint32_t), execution_stream);
      }
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA periodic refresh diagnostic reset", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      validate_gfn2_periodic_refresh_kernel<<<static_cast<unsigned int>(device.batch_size), 256, 0,
                                              execution_stream>>>(device);
      cuda_status = cudaPeekAtLastError();
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA periodic numerical validation", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
    }

    gate_gfn2_numerical_refresh_kernel<<<static_cast<unsigned int>(device.batch_size), 1, 0,
                                         execution_stream>>>(device);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical pre-factor publication gate", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const auto factor = current.eigensolver_owner.refactor_overlap_from_device_epoch_async(
        current.eigensolver_setup_arena.get(), current.eigensolver_setup_arena.bytes(),
        current.eigensolver_binding, preprocessing.output.overlap,
        preprocessing.output.overlap_elements, preprocessing.geometry_epoch, execution_stream);
    if (!factor.success()) {
      error = setup_error_message("CUDA numerical overlap refactor", factor.status,
                                  static_cast<std::uint32_t>(factor.error),
                                  static_cast<std::uint32_t>(factor.field), factor.index);
      return factor.status;
    }
    device.factor_generations = current.eigensolver_binding.cache.geometry_generations;
    device.factor_statuses = current.eigensolver_binding.cache.factor_statuses;
    commit_gfn2_numerical_refresh_kernel<<<static_cast<unsigned int>(device.batch_size), 256, 0,
                                           execution_stream>>>(device);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical final publication", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    current.submitted = true;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t enqueue_inference_tail_locked(Prepared& current, bool capture_bounded_scc,
                                                 cudaStream_t execution_stream,
                                                 std::string& error) {
    auto& inference = current.inference;
    cudaError_t cuda_status = cudaSetDevice(device_id);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("cudaSetDevice for CUDA GFN2 inference", cuda_status);
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }

    /* The synchronous path launches the production device-resident early-stop
     * Graph. The asynchronous request Graph instead captures the bounded
     * iteration DAG at setup. Nesting a device-launched/tail-launched SCC
     * Graph inside a conditional Graph did not provide a reliable completion
     * boundary on CUDA 12.9/Blackwell; the bounded body keeps one prebuilt
     * request-Graph launch per call and introduces no steady-state allocation,
     * host polling, transfer, or synchronization. */
    const Gfn2SccLoopLaunchResult loop =
        capture_bounded_scc
            ? launch_gfn2_restricted_scc_loop_cuda(current.scc_binding,
                                                   inference.epoch_consumer, execution_stream)
            : current.scc_loop.launch(execution_stream);
    if (!loop.success()) {
      std::ostringstream message;
      message << "CUDA SCC loop submission failed: mode="
              << static_cast<std::uint32_t>(loop.execution_mode)
              << " status=" << static_cast<std::uint32_t>(loop.iteration.status)
              << " stage=" << static_cast<std::uint32_t>(loop.iteration.stage)
              << " cuda=" << static_cast<int>(loop.iteration.cuda_status);
      error = message.str();
      if (loop.iteration.status == Gfn2SccIterationLaunchStatus::kInvalidBinding) {
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      if (loop.iteration.status == Gfn2SccIterationLaunchStatus::kCublasError ||
          loop.iteration.status == Gfn2SccIterationLaunchStatus::kCusolverError) {
        return XTBLOOM_STATUS_EIGENSOLVER_FAILED;
      }
      if (loop.iteration.status == Gfn2SccIterationLaunchStatus::kProviderCaptureUnsupported) {
        return XTBLOOM_STATUS_NOT_SUPPORTED;
      }
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    cuda_status = evaluate_gfn2_terminal_classical_energy_cuda(
        inference.terminal_plan, inference.terminal_activity, inference.terminal_results,
        inference.terminal_workspace, inference.terminal_diagnostics, execution_stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA terminal classical energy", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? XTBLOOM_STATUS_INVALID_ARGUMENT
                                                  : XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    cuda_status = project_gfn2_stationary_force_state_cuda(
        current.energy_force.stationary_projection, execution_stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA stationary force projection", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? XTBLOOM_STATUS_INVALID_ARGUMENT
                                                  : XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    cuda_status = execute_gfn2_energy_force_cuda(
        current.energy_force.plan, current.energy_force.input, current.energy_force.results,
        current.energy_force.intermediates, current.energy_force.workspace,
        current.energy_force.diagnostics, inference.epoch_consumer, execution_stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA terminal energy/force execution", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? XTBLOOM_STATUS_INVALID_ARGUMENT
                                                  : XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    cuda_status = publish_gfn2_inference_results_cuda(
        inference.publication_plan, inference.publication_input, inference.publication_results,
        inference.publication_workspace, inference.publication_diagnostics, execution_stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA internal inference publication", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? XTBLOOM_STATUS_INVALID_ARGUMENT
                                                  : XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    cuda_status = cudaMemsetAsync(inference.warm_checkpoint_batch_ready, 0xff,
                                  sizeof(std::uint32_t), execution_stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA warm-checkpoint aggregate initialization", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    const WarmCheckpointPublicationDeviceBinding checkpoint{
        current.host.basis.batch_size,
        inference.epoch_consumer.epoch,
        inference.epoch_consumer.eligible_mask,
        inference.epoch_consumer.committed_generations,
        inference.publication_diagnostics.plan_error,
        inference.publication_results.system_statuses,
        inference.publication_results.converged,
        inference.warm_checkpoint_generations,
        inference.warm_checkpoint_batch_ready,
    };
    constexpr int kThreads = 256;
    const auto blocks = static_cast<unsigned int>(
        (static_cast<std::uint64_t>(checkpoint.batch_size) + kThreads - 1u) / kThreads);
    publish_gfn2_warm_checkpoint_generation_kernel<<<blocks, kThreads, 0, execution_stream>>>(
        checkpoint);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA warm-checkpoint publication", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    current.submitted = true;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t execute_inference_body_locked(Prepared& current, std::string& error) {
    return enqueue_inference_tail_locked(current, false, stream, error);
  }

  xtbloom_status_t build_request_execution_graph(Prepared& current, std::string& error) {
    if (!current.public_result.ready || current.public_result.request_topology_error == nullptr ||
        current.inference.warm_checkpoint_generations == nullptr ||
        current.inference.warm_checkpoint_batch_ready == nullptr) {
      error = "CUDA asynchronous request graph has an incomplete fixed binding";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    /* A bounded SCC fallback remains a valid synchronous runtime. Async
     * enqueue reports that narrower capability gap without preventing plan
     * creation or synchronous plan execution on older CUDA/provider stacks. */
    if (!current.scc_loop.conditional_graph_ready()) {
      error.clear();
      return XTBLOOM_STATUS_SUCCESS;
    }

    auto& owner = current.request_execution_graph;
    cudaError_t cuda_status = owner.create();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA asynchronous request graph creation", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    cudaStream_t capture_stream = nullptr;
    cuda_status = cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking);
    if (cuda_status != cudaSuccess) {
      owner.reset();
      error = cuda_error_message("CUDA asynchronous request capture stream creation", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    bool capture_active = false;
    const auto finish_capture = [&]() noexcept {
      if (capture_active) {
        cudaGraph_t ended = nullptr;
        (void)cudaStreamEndCapture(capture_stream, &ended);
      }
      (void)cudaStreamDestroy(capture_stream);
      owner.reset();
    };

    cuda_status = cudaStreamBeginCaptureToGraph(capture_stream, owner.graph(), nullptr, nullptr, 0u,
                                                cudaStreamCaptureModeThreadLocal);
    if (cuda_status != cudaSuccess) {
      finish_capture();
      error = cuda_error_message("CUDA asynchronous request root capture", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    capture_active = true;

    /* Accepted FRESH submission consumes the old checkpoint irrespective of
     * the IF body, matching the public asynchronous contract. */
    cuda_status = cudaMemsetAsync(
        current.inference.warm_checkpoint_generations, 0,
        static_cast<std::size_t>(current.host.basis.batch_size) * sizeof(std::uint64_t),
        capture_stream);
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemsetAsync(current.inference.warm_checkpoint_batch_ready, 0,
                                    sizeof(std::uint32_t), capture_stream);
    }
    if (cuda_status == cudaSuccess) {
      set_request_execution_condition_kernel<<<1, 1, 0, capture_stream>>>(
          current.public_result.request_topology_error, owner.condition());
      cuda_status = cudaPeekAtLastError();
    }
    if (cuda_status != cudaSuccess) {
      finish_capture();
      error = cuda_error_message("CUDA asynchronous request admission capture", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cudaGraph_t captured_graph = nullptr;
    const cudaGraphNode_t* dependencies = nullptr;
    std::size_t dependency_count = 0u;
    cuda_status = cudaStreamGetCaptureInfo(capture_stream, &capture_status, nullptr,
                                           &captured_graph, &dependencies, &dependency_count);
    if (cuda_status != cudaSuccess || capture_status == cudaStreamCaptureStatusNone ||
        captured_graph != owner.graph() || dependencies == nullptr || dependency_count != 1u) {
      finish_capture();
      error = "CUDA asynchronous request capture has no admission dependency frontier";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    const cudaGraphNode_t admission = dependencies[0];
    cudaGraph_t body = nullptr;
    cuda_status = owner.add_if_node(admission, body);
    cudaGraph_t ended_graph = nullptr;
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaStreamEndCapture(capture_stream, &ended_graph);
      capture_active = false;
    }
    if (cuda_status != cudaSuccess || ended_graph != owner.graph() || body == nullptr) {
      finish_capture();
      error = cuda_error_message(
          "CUDA asynchronous request conditional graph",
          cuda_status == cudaSuccess ? cudaErrorStreamCaptureInvalidated : cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    /* Populate the IF body after root capture. The body owns a stable copy of
     * numerical ingress, fresh-state restore, the SCC root child, and terminal
     * internal publication. */
    cuda_status = cudaStreamBeginCaptureToGraph(capture_stream, body, nullptr, nullptr, 0u,
                                                cudaStreamCaptureModeThreadLocal);
    if (cuda_status == cudaSuccess) {
      capture_active = true;
      std::string capture_error;
      xtbloom_status_t status =
          execute_numerical_body_locked(current, capture_stream, capture_error);
      if (status == XTBLOOM_STATUS_SUCCESS) {
        const auto diagnostic = current.initializer.upload_async(current.iteration_arena.get(),
                                                                 current.iteration_arena.bytes(),
                                                                 current.ready, capture_stream);
        if (!diagnostic.success()) {
          status = diagnostic.status;
          capture_error =
              setup_error_message("CUDA request-graph SCC fresh-state restore", diagnostic.status,
                                  static_cast<std::uint32_t>(diagnostic.error),
                                  static_cast<std::uint32_t>(diagnostic.field), diagnostic.index);
        }
      }
      if (status == XTBLOOM_STATUS_SUCCESS) {
        status = enqueue_inference_tail_locked(current, true, capture_stream, capture_error);
      }
      if (status != XTBLOOM_STATUS_SUCCESS) {
        finish_capture();
        error = capture_error.empty() ? "CUDA asynchronous request body capture failed"
                                      : std::move(capture_error);
        return status;
      }
      cuda_status = cudaStreamEndCapture(capture_stream, &ended_graph);
      capture_active = false;
    }
    if (cuda_status != cudaSuccess || ended_graph != body) {
      finish_capture();
      error = cuda_error_message(
          "CUDA asynchronous request body capture",
          cuda_status == cudaSuccess ? cudaErrorStreamCaptureInvalidated : cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    cuda_status = owner.instantiate();
    (void)cudaStreamDestroy(capture_stream);
    if (cuda_status != cudaSuccess) {
      owner.reset();
      error = cuda_error_message("CUDA asynchronous request graph instantiation", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t launch_request_execution_graph_locked(Prepared& current, std::string& error) {
    if (!current.request_execution_graph.ready()) {
      error = "CUDA asynchronous request execution graph is not initialized";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const cudaError_t cuda_status = current.request_execution_graph.launch(stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA asynchronous request graph launch", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    current.submitted = true;
    current.inference.warm_checkpoint_ready = false;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t refresh_numerical_locked(Prepared& current,
                                            const Gfn2CudaNumericalInputView& input,
                                            std::string& error) {
    bool host_upload_enqueued = false;
    xtbloom_status_t status =
        stage_numerical_ingress_locked(current, input, host_upload_enqueued, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = execute_numerical_body_locked(current, stream, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    if (host_upload_enqueued) {
      /* Make owner-stream completion prove that the private callback has
       * released the pinned snapshot. This preserves allocation-free,
       * no-polling steady state while preventing a completed refresh from
       * spuriously rejecting the next host-backed refresh. */
      const cudaError_t release_status =
          cudaStreamWaitEvent(stream, current.numerical_host_release_complete.get(), 0u);
      if (release_status != cudaSuccess) {
        current.numerical.host_staging_poisoned = true;
        error = cuda_error_message("CUDA numerical host-release tail ordering", release_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      current.submitted = true;
    }
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t execute_inference_locked(Prepared& current, Gfn2CudaSccStartMode mode,
                                            std::string& error) {
    if (!current.inference.ready || !current.numerical.ready) {
      error = "CUDA GFN2 inference requires a prepared numerical/runtime binding";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if (current.numerical.device.refresh_predecessor_generations == nullptr ||
        current.inference.warm_checkpoint_generations == nullptr ||
        current.inference.warm_checkpoint_batch_ready == nullptr) {
      error = "CUDA GFN2 inference has an incomplete warm-checkpoint binding";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    if (mode != Gfn2CudaSccStartMode::kFresh && mode != Gfn2CudaSccStartMode::kWarm) {
      error = "CUDA GFN2 inference received an unknown SCC start mode";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    auto& inference = current.inference;
    if (mode == Gfn2CudaSccStartMode::kWarm && !inference.warm_checkpoint_ready) {
      error = "CUDA GFN2 warm inference requires a previously submitted checkpoint";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    cudaError_t cuda_status = cudaSetDevice(device_id);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("cudaSetDevice for CUDA GFN2 inference", cuda_status);
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }

    if (mode == Gfn2CudaSccStartMode::kFresh) {
      /* A fresh attempt consumes every old checkpoint before any state image
       * is restored. If a later enqueue fails, no stale warm token can survive
       * and masquerade as the failed attempt's checkpoint. */
      cuda_status = cudaMemsetAsync(
          inference.warm_checkpoint_generations, 0,
          static_cast<std::size_t>(current.host.basis.batch_size) * sizeof(std::uint64_t), stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA fresh warm-checkpoint invalidation", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      current.submitted = true;
      inference.warm_checkpoint_ready = false;
      const auto diagnostic = current.initializer.upload_async(
          current.iteration_arena.get(), current.iteration_arena.bytes(), current.ready, stream);
      if (!diagnostic.success()) {
        error = setup_error_message("CUDA SCC fresh-state restore", diagnostic.status,
                                    static_cast<std::uint32_t>(diagnostic.error),
                                    static_cast<std::uint32_t>(diagnostic.field), diagnostic.index);
        return diagnostic.status;
      }
      current.submitted = true;
    } else {
      const WarmSccResetDeviceBinding warm{
          current.host.basis.batch_size,
          current.host.plan_token,
          inference.epoch_consumer.epoch,
          inference.epoch_consumer.eligible_mask,
          inference.epoch_consumer.committed_generations,
          current.numerical.device.refresh_predecessor_generations,
          inference.warm_checkpoint_generations,
          current.state_seed.mixer,
          current.state_seed.scc,
      };
      constexpr int kThreads = 256;
      const auto blocks = static_cast<unsigned int>(
          (static_cast<std::uint64_t>(warm.batch_size) + kThreads - 1u) / kThreads);
      reset_gfn2_warm_scc_trace_kernel<<<blocks, kThreads, 0, stream>>>(warm);
      cuda_status = cudaPeekAtLastError();
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA warm SCC trace reset", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      current.submitted = true;
      inference.warm_checkpoint_ready = false;
    }

    const xtbloom_status_t body_status = execute_inference_body_locked(current, error);
    if (body_status != XTBLOOM_STATUS_SUCCESS) return body_status;
    inference.warm_checkpoint_ready = true;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  /*
   * A synchronous public call must not return while either the owner stream
   * can still read caller CUDA pointers or the private host-upload callback
   * can still access Prepared-owned lease state.  Failure paths use the same
   * disable-timing event as successful publication and fall back to an exact
   * owner-stream fence if recording that event itself fails.
   */
  xtbloom_status_t settle_public_submissions_locked(Prepared& current,
                                                    xtbloom_status_t primary_status,
                                                    std::string& error) {
    cudaError_t owner_status = cudaSuccess;
    if (current.submitted) {
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
      if (consume_execution_test_fault(Gfn2CudaExecutionTestFault::kRequestSettlement)) {
        owner_status = cudaErrorUnknown;
      } else
#endif
      {
      owner_status = cudaEventRecord(current.public_result_completion_event.get(), stream);
      if (owner_status == cudaSuccess) {
        owner_status = cudaEventSynchronize(current.public_result_completion_event.get());
      }
      if (owner_status != cudaSuccess) {
        const cudaError_t fallback_status = cudaStreamSynchronize(stream);
        if (fallback_status == cudaSuccess) owner_status = cudaSuccess;
      }
      if (owner_status == cudaSuccess) current.submitted = false;
      }
    }

    cudaError_t host_status = cudaSuccess;
    if (current.numerical_host_completion_stream.valid()) {
      host_status = cudaStreamSynchronize(current.numerical_host_completion_stream.get());
      if (host_status == cudaSuccess) {
        /* The stream fence is after the release callback, so this store cannot
         * race a previous lease generation or clear a future one. */
        current.numerical_host_upload_completion.pending.store(false, std::memory_order_release);
      } else {
        current.numerical.host_staging_poisoned = true;
      }
    }
    if (owner_status == cudaSuccess && host_status == cudaSuccess) return primary_status;

    std::ostringstream message;
    if (!error.empty()) message << error << "; additionally, ";
    message << "CUDA public transaction settlement failed";
    if (owner_status != cudaSuccess) {
      message << " owner_stream=" << cudaGetErrorString(owner_status);
    }
    if (host_status != cudaSuccess) {
      message << " host_completion_stream=" << cudaGetErrorString(host_status);
    }
    error = message.str();
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  xtbloom_status_t enqueue_public_result_prepare_locked(
      Prepared& current, const xtbloom_batch_t& batch, const xtbloom_compute_options_t& options,
      xtbloom_batch_result_t& result, PublicResultTransaction& transaction, std::string& error) {
    transaction = {};
    if (!current.public_result.ready || !current.inference.ready ||
        current.public_result_completion_event.get() == nullptr) {
      error = "CUDA public result bridge requires a complete prepared runtime";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const std::int64_t batch_size = current.host.basis.batch_size;
    const std::int64_t atoms = current.host.basis.total_atoms;
    const std::int64_t points = current.host.external.total_point_charges;
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    if (!checked_elements(atoms, 3, coordinates) ||
        !checked_elements(points, 3, point_coordinates)) {
      error = "CUDA public result extent overflows int64_t";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const std::uint64_t token = current.host.plan_token;
    const bool external_operator = batch.atomic_potential_shifts.data != nullptr ||
                                   batch.atomic_potential_shifts.size_bytes != 0u ||
                                   batch.charge_response_matrix.data != nullptr ||
                                   batch.charge_response_matrix.size_bytes != 0u;
    const std::uint32_t result_flags =
        external_operator ? static_cast<std::uint32_t>(
                                XTBLOOM_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES)
                          : 0u;
    const Gfn2PublicResultBridgeDevicePlan plan{
        kGfn2PublicResultBridgeAbiVersion,
        options.flags,
        result_flags,
        0u,
        token,
        batch_size,
        atoms,
        points,
    };
    const auto& inference = current.inference;
    const Gfn2PublicResultBridgeDeviceInput input{
        inference.publication_results.energies,
        inference.publication_results.energy_elements,
        inference.publication_results.qm_forces,
        inference.publication_results.qm_force_elements,
        inference.publication_results.atomic_charges,
        inference.publication_results.atomic_charge_elements,
        inference.publication_results.point_forces,
        inference.publication_results.point_force_elements,
        inference.publication_results.iterations,
        inference.publication_results.converged,
        inference.publication_results.system_statuses,
        inference.publication_results.batch_elements,
        inference.publication_diagnostics.plan_error,
        current.public_result.request_topology_error,
        inference.publication_workspace.epoch_snapshot,
        current.numerical.preprocessing.geometry_epoch.value,
        token,
    };

    Gfn2PublicResultBridgeDeviceDestinations destinations{};
    Gfn2PublicResultBridgeHostStaging staging{};
    const auto bind = [](const xtbloom_buffer_t& output, std::int64_t elements, void* host_stage,
                         Gfn2PublicResultBridgeDestination& destination,
                         Gfn2PublicResultBridgeHostBuffer& host) {
      if (elements == 0) {
        destination = {};
        host = {};
        return;
      }
      const bool host_route = output.memory_space == XTBLOOM_MEMORY_HOST;
      destination.route =
          host_route ? Gfn2PublicResultRoute::kHost : Gfn2PublicResultRoute::kCudaDevice;
      destination.device_data = host_route ? nullptr : output.data;
      destination.elements = elements;
      host.data = host_route ? host_stage : nullptr;
      host.elements = host_route ? elements : 0;
    };
    const bool energy_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ENERGY)) != 0u;
    const bool force_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_FORCES)) != 0u;
    const bool charges_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ATOMIC_CHARGES)) != 0u;
    const bool point_forces_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_POINT_CHARGE_FORCES)) != 0u;
    auto& public_state = current.public_result;
    bind(result.energies, energy_requested ? batch_size : 0, public_state.energies,
         destinations.energies, staging.energies);
    bind(result.forces, force_requested ? coordinates : 0, public_state.qm_forces,
         destinations.qm_forces, staging.qm_forces);
    bind(result.atomic_charges, charges_requested ? atoms : 0, public_state.atomic_charges,
         destinations.atomic_charges, staging.atomic_charges);
    bind(result.point_charge_forces, point_forces_requested ? point_coordinates : 0,
         public_state.point_forces, destinations.point_forces, staging.point_forces);
    bind(result.scc_iterations, batch_size, public_state.iterations, destinations.iterations,
         staging.iterations);
    bind(result.scc_converged, batch_size, public_state.converged, destinations.converged,
         staging.converged);
    bind(result.per_system_status, batch_size, public_state.system_statuses,
         destinations.system_statuses, staging.system_statuses);
    destinations.plan_token = token;
    staging.control = public_state.host_control;
    staging.control_elements = 1;
    staging.pending_result_flags = &public_state.pending_result_flags;
    staging.plan_token = token;

    cudaError_t cuda_status =
        prepare_gfn2_public_results_cuda(plan, input, public_state.device_staging, destinations,
                                         staging, public_state.diagnostics, stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA public result bridge submission", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? XTBLOOM_STATUS_INVALID_ARGUMENT
                                                  : XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    cuda_status =
        cudaMemcpyAsync(public_state.warm_checkpoint_ready, inference.warm_checkpoint_batch_ready,
                        sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA warm-checkpoint readiness download", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    transaction.plan = plan;
    transaction.destinations = destinations;
    transaction.phase = PublicResultTransactionPhase::kPrepareSubmitted;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  /*
   * Append one completion marker after a submitted public bridge phase. The
   * commit phase retains the historical stream-fence fallback when recording
   * the event itself fails; prepare keeps its existing strict event behavior.
   */
  xtbloom_status_t record_public_result_completion_locked(
      Prepared& current, PublicResultTransaction& transaction,
      PublicResultTransactionPhase submitted_phase, PublicResultTransactionPhase recorded_phase,
      PublicResultTransactionPhase completed_phase, bool fallback_to_stream,
      const char* failure_context, std::string& error) {
    if (transaction.phase != submitted_phase ||
        current.public_result_completion_event.get() == nullptr) {
      error = "CUDA public result completion marker has an invalid transaction phase";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    const cudaError_t record_status =
        cudaEventRecord(current.public_result_completion_event.get(), stream);
    if (record_status == cudaSuccess) {
      transaction.phase = recorded_phase;
      return XTBLOOM_STATUS_SUCCESS;
    }
    if (fallback_to_stream && cudaStreamSynchronize(stream) == cudaSuccess) {
      current.submitted = false;
      transaction.phase = completed_phase;
      return XTBLOOM_STATUS_SUCCESS;
    }

    error = cuda_error_message(failure_context, record_status);
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  /* Completion observation never publishes host bytes or result.flags. */
  xtbloom_status_t wait_public_result_completion_locked(
      Prepared& current, PublicResultTransaction& transaction,
      PublicResultTransactionPhase recorded_phase, PublicResultTransactionPhase completed_phase,
      const char* failure_context, std::string& error) {
    if (transaction.phase == completed_phase) return XTBLOOM_STATUS_SUCCESS;
    if (transaction.phase != recorded_phase ||
        current.public_result_completion_event.get() == nullptr) {
      error = "CUDA public result completion wait has an invalid transaction phase";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    const cudaError_t cuda_status =
        cudaEventSynchronize(current.public_result_completion_event.get());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message(failure_context, cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    current.submitted = false;
    transaction.phase = completed_phase;
    return XTBLOOM_STATUS_SUCCESS;
  }

  /* Read the pinned aggregate only after prepare completion is observed. */
  xtbloom_status_t accept_public_result_prepare_locked(Prepared& current,
                                                       PublicResultTransaction& transaction,
                                                       std::string& error) {
    if (transaction.phase != PublicResultTransactionPhase::kPrepareCompleted) {
      error = "CUDA public result prepare acceptance precedes completion";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const auto& public_state = current.public_result;
    const auto aggregate =
        static_cast<Gfn2PublicResultBridgeError>(public_state.host_control->aggregate_error);
    if (aggregate != Gfn2PublicResultBridgeError::kSuccess) {
      if (aggregate == Gfn2PublicResultBridgeError::kRequestTopologyMismatch) {
        error = "the batch topology does not match the fixed CUDA plan topology";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestInvalidArgument) {
        error = "CUDA stream-ordered request validation rejected an invalid descriptor";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestNotSupported) {
        error = "CUDA stream-ordered request validation rejected unsupported periodic axes";
        return XTBLOOM_STATUS_NOT_SUPPORTED;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestNotImplemented) {
        error = "the CUDA request uses a validated feature that is not implemented yet";
        return XTBLOOM_STATUS_NOT_IMPLEMENTED;
      }
      std::ostringstream message;
      message << "CUDA public result bridge rejected inference: aggregate_error="
              << static_cast<std::uint32_t>(aggregate) << " publication_plan_error="
              << public_state.host_control->internal_publication_plan_error
              << " publication_epoch=" << public_state.host_control->publication_epoch_snapshot
              << " current_epoch=" << public_state.host_control->current_geometry_epoch;
      error = message.str();
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    transaction.phase = PublicResultTransactionPhase::kPrepareAccepted;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  /* The asynchronous path submits the device-gated commit before returning,
   * so its host acceptance occurs only after that commit event completes. The
   * device kernel itself reads the same aggregate control and is a no-op on a
   * rejected prepare image. */
  xtbloom_status_t accept_public_result_after_commit_locked(Prepared& current,
                                                            PublicResultTransaction& transaction,
                                                            std::string& error) {
    if (transaction.phase != PublicResultTransactionPhase::kCommitCompleted) {
      error = "CUDA public result acceptance precedes asynchronous commit completion";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const auto& public_state = current.public_result;
    const auto aggregate =
        static_cast<Gfn2PublicResultBridgeError>(public_state.host_control->aggregate_error);
    if (aggregate != Gfn2PublicResultBridgeError::kSuccess) {
      if (aggregate == Gfn2PublicResultBridgeError::kRequestTopologyMismatch) {
        error = "the batch topology does not match the fixed CUDA plan topology";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestInvalidArgument) {
        error = "CUDA stream-ordered request validation rejected an invalid descriptor";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestNotSupported) {
        error = "CUDA stream-ordered request validation rejected unsupported periodic axes";
        return XTBLOOM_STATUS_NOT_SUPPORTED;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestNotImplemented) {
        error = "the CUDA request uses a validated feature that is not implemented yet";
        return XTBLOOM_STATUS_NOT_IMPLEMENTED;
      }
      std::ostringstream message;
      message << "CUDA public result bridge rejected inference: aggregate_error="
              << static_cast<std::uint32_t>(aggregate) << " publication_plan_error="
              << public_state.host_control->internal_publication_plan_error
              << " publication_epoch=" << public_state.host_control->publication_epoch_snapshot
              << " current_epoch=" << public_state.host_control->current_geometry_epoch;
      error = message.str();
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t enqueue_public_result_commit_locked(Prepared& current,
                                                       PublicResultTransaction& transaction,
                                                       bool allow_device_gated_prepare,
                                                       std::string& error) {
    if (transaction.phase != PublicResultTransactionPhase::kPrepareAccepted &&
        !(allow_device_gated_prepare &&
          transaction.phase == PublicResultTransactionPhase::kPrepareSubmitted)) {
      error = "CUDA public result commit has no accepted prepared image";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const cudaError_t cuda_status = commit_gfn2_public_results_cuda(
        transaction.plan, current.public_result.device_staging, transaction.destinations,
        current.public_result.diagnostics, stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA caller-device result commit submission", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? XTBLOOM_STATUS_INVALID_ARGUMENT
                                                  : XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    /* From this point onward caller CUDA bytes may be modified. No later
     * failure is recoverable by pretending that the transaction rolled back. */
    current.submitted = true;
    transaction.phase = PublicResultTransactionPhase::kCommitSubmitted;
    return XTBLOOM_STATUS_SUCCESS;
  }

  /* Host result publication is a separate, non-CUDA phase after commit wait. */
  xtbloom_status_t publish_public_results_locked(
      Prepared& current, const xtbloom_compute_options_t& options, xtbloom_batch_result_t& result,
      PublicResultTransaction& transaction, bool commit_host_outputs, bool publish_descriptor_flags,
      std::uint32_t& completed_result_flags, std::string& error) {
    if (transaction.phase != PublicResultTransactionPhase::kCommitCompleted) {
      error = "CUDA public result host publication precedes commit completion";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    if (!commit_host_outputs) return XTBLOOM_STATUS_INTERNAL_ERROR;

    const std::int64_t batch_size = current.host.basis.batch_size;
    const std::int64_t atoms = current.host.basis.total_atoms;
    const std::int64_t points = current.host.external.total_point_charges;
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    if (!checked_elements(atoms, 3, coordinates) ||
        !checked_elements(points, 3, point_coordinates)) {
      error = "CUDA public result extent changed after commit acceptance";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const bool energy_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ENERGY)) != 0u;
    const bool force_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_FORCES)) != 0u;
    const bool charges_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ATOMIC_CHARGES)) != 0u;
    const bool point_forces_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_POINT_CHARGE_FORCES)) != 0u;
    auto& public_state = current.public_result;
    const auto commit_host = [](const xtbloom_buffer_t& output, const void* staging_data,
                                std::int64_t elements, std::size_t element_size) {
      if (elements != 0 && output.memory_space == XTBLOOM_MEMORY_HOST) {
        std::memcpy(output.data, staging_data, static_cast<std::size_t>(elements) * element_size);
      }
    };
    commit_host(result.energies, public_state.energies, energy_requested ? batch_size : 0,
                sizeof(double));
    commit_host(result.forces, public_state.qm_forces, force_requested ? coordinates : 0,
                sizeof(double));
    commit_host(result.atomic_charges, public_state.atomic_charges, charges_requested ? atoms : 0,
                sizeof(double));
    commit_host(result.point_charge_forces, public_state.point_forces,
                point_forces_requested ? point_coordinates : 0, sizeof(double));
    commit_host(result.scc_iterations, public_state.iterations, batch_size, sizeof(std::int32_t));
    commit_host(result.scc_converged, public_state.converged, batch_size, sizeof(std::uint8_t));
    commit_host(result.per_system_status, public_state.system_statuses, batch_size,
                sizeof(xtbloom_status_t));
    completed_result_flags = public_state.pending_result_flags;
    if (publish_descriptor_flags) result.flags = completed_result_flags;
    current.inference.warm_checkpoint_ready = *public_state.warm_checkpoint_ready != 0u;
    transaction.phase = PublicResultTransactionPhase::kPublished;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  /* Complete the one request-owned publication transaction in a single place.
   * In particular, a deferred stream-ordered failure must not leave the host
   * checkpoint-ready bit resurrected by an otherwise successful D2H result
   * bridge. The device token was already consumed by FRESH/WARM admission, so
   * keeping the host bit false makes every later strict WARM reject safely. */
  void finalize_active_request_after_commit_locked(Prepared& current) {
    auto& active = active_request;
    if (active.deferred_status != XTBLOOM_STATUS_SUCCESS) {
      active.completion_status = active.deferred_status;
      active.result_flags = 0u;
      /* The device-gated commit is already ordered, so caller CUDA buffers may
       * have changed. Host publication remains controllable: close readiness
       * before diagnostic composition and do not copy staged host bytes for a
       * request whose accepted transaction already has a deferred failure. */
      current.inference.warm_checkpoint_ready = false;
      if (!active.completion_error.empty()) active.completion_error += "; additionally, ";
      active.completion_error += active.deferred_error;
      active.completion_ready = true;
      return;
    }

    xtbloom_status_t status = accept_public_result_after_commit_locked(current, active.transaction,
                                                                       active.completion_error);
    if (status == XTBLOOM_STATUS_SUCCESS) {
      status =
          publish_public_results_locked(current, active.options, active.result, active.transaction,
                                        true, false, active.result_flags, active.completion_error);
    }
    active.completion_status = status;
    if (status != XTBLOOM_STATUS_SUCCESS) {
      active.result_flags = 0u;
      current.inference.warm_checkpoint_ready = false;
    }
    active.completion_ready = true;
  }

  Gfn2CudaExecutionIdentity snapshot() const noexcept {
    Gfn2CudaExecutionIdentity identity{};
    if (prepared == nullptr) return identity;
    const Prepared& current = *prepared;
    identity.topology_fingerprint = current.host.fingerprint;
    identity.plan_token = current.host.plan_token;
    identity.iteration_layout_fingerprint = current.iteration_requirements.layout_fingerprint;
    identity.enabled_component_mask = current.plan_seed.enabled_components;
    identity.scc_binding_ready = current.scc_binding.plan.plan_token == current.host.plan_token;
    identity.energy_force_binding_ready =
        current.energy_force.plan.plan_token == current.host.plan_token &&
        current.energy_force.input.plan_token == current.host.plan_token &&
        current.energy_force.results.plan_token == current.host.plan_token &&
        current.energy_force.intermediates.plan_token == current.host.plan_token &&
        current.energy_force.workspace.plan_token == current.host.plan_token &&
        current.energy_force.diagnostics.plan_token == current.host.plan_token;
    identity.force_mode_ready = current.energy_force.plan.compute_forces == 1u;
    identity.energy_force_smoke_ready = current.energy_force_smoke_ready ? 1u : 0u;
    identity.numerical_refresh_ready = current.numerical.ready ? 1u : 0u;
    identity.inference_ready =
        current.inference.ready &&
                current.inference.terminal_plan.plan_token == current.host.plan_token &&
                current.inference.publication_plan.plan_token == current.host.plan_token &&
                current.inference.publication_results.plan_token == current.host.plan_token
            ? 1u
            : 0u;
    identity.warm_checkpoint_ready = current.inference.warm_checkpoint_ready ? 1u : 0u;
    identity.scc_conditional_graph_ready = current.scc_loop.conditional_graph_ready() ? 1u : 0u;
    identity.scc_loop_fallback_reason =
        static_cast<std::uint32_t>(current.scc_loop.fallback_reason());
    identity.batch_size = current.host.basis.batch_size;
    identity.total_atoms = current.host.basis.total_atoms;
    identity.total_shells = current.host.basis.total_shells;
    identity.total_orbitals = current.host.basis.total_orbitals;
    identity.total_point_charges = current.host.external.total_point_charges;
    identity.solver_handle = opaque_address(solver);
    identity.solver_parameters = opaque_address(solver_parameters);
    identity.blas_handle = opaque_address(blas);
    identity.topology_owner = opaque_address(&current.topology_owner);
    identity.inputs_owner = opaque_address(&current.inputs_owner);
    identity.eigensolver_owner = opaque_address(&current.eigensolver_owner);
    identity.initializer_owner = opaque_address(&current.initializer);
    identity.scc_binding = opaque_address(&current.scc_binding);
    identity.scc_loop_owner = opaque_address(&current.scc_loop);
    identity.scc_loop_active_count =
        opaque_address(current.scc_loop.canonical_active_count_device());
    identity.scc_loop_numerical_body_count =
        opaque_address(current.scc_loop.numerical_body_count_device());
    identity.scc_loop_device_launch_error =
        opaque_address(current.scc_loop.device_launch_error_device());
    identity.energy_force_descriptors = opaque_address(&current.energy_force);
    identity.request_submissions = request_submissions;
    identity.request_active = active_request.id == 0u ? 0u : 1u;
    identity.topology_arena = opaque_address(current.topology_arena.get());
    identity.input_arena = opaque_address(current.input_arena.get());
    identity.iteration_arena = opaque_address(current.iteration_arena.get());
    identity.eigensolver_setup_arena = opaque_address(current.eigensolver_setup_arena.get());
    identity.provider_host_workspace = opaque_address(current.provider_host_workspace.get());
    identity.force_immutable_arena = opaque_address(current.force_immutable_arena.get());
    identity.force_execution_arena = opaque_address(current.force_execution_arena.get());
    identity.numerical_refresh_arena = opaque_address(current.numerical_refresh_arena.get());
    identity.numerical_refresh_binding = opaque_address(&current.numerical.preprocessing);
    identity.numerical_epoch = opaque_address(current.numerical.preprocessing.geometry_epoch.value);
    identity.committed_generations = opaque_address(current.numerical.device.committed_generations);
    identity.numerical_eligible_mask = opaque_address(current.numerical.device.eligible);
    identity.overlap_factor_generations =
        opaque_address(current.eigensolver_binding.cache.geometry_generations);
    identity.overlap_factor_statuses =
        opaque_address(current.eigensolver_binding.cache.factor_statuses);
    const auto opaque_buffer = [](const double* address, std::int64_t elements) noexcept {
      return Gfn2CudaOpaqueBufferIdentity{opaque_address(address), elements};
    };
    const auto& numerical = current.numerical.device;
    identity.committed_positions =
        opaque_buffer(numerical.committed_positions, numerical.total_atoms * 3);
    identity.committed_geometry_pairs =
        opaque_buffer(numerical.public_geometry_pairs, numerical.geometry_pair_elements);
    identity.committed_coordination_numbers =
        opaque_buffer(numerical.public_coordination, numerical.total_atoms);
    identity.committed_overlap = opaque_buffer(numerical.public_overlap, numerical.total_matrices);
    identity.committed_dipole_integrals =
        opaque_buffer(numerical.public_dipole, numerical.total_matrices * 3);
    identity.committed_quadrupole_integrals =
        opaque_buffer(numerical.public_quadrupole, numerical.total_matrices * 6);
    identity.committed_h0 = opaque_buffer(numerical.public_h0, numerical.total_matrices);
    identity.committed_es2 = opaque_buffer(numerical.public_es2, numerical.es2_elements);
    identity.committed_aes2 = opaque_buffer(numerical.public_aes2, numerical.aes2_elements);
    /* Production D4 consumes the committed physical pair-list superset and no
     * longer retains a dense five-double pair-value cache. Keep the diagnostic
     * leaf canonically empty so memory evidence can detect a regression. */
    identity.committed_d4_pairs = {};
    identity.committed_d4_coordination_numbers = opaque_buffer(
        numerical.public_d4_coordination, numerical.d4_enabled != 0u ? numerical.total_atoms : 0);
    identity.committed_point_charge_positions =
        opaque_buffer(numerical.committed_point_positions,
                      numerical.point_enabled != 0u ? numerical.total_point_charges * 3 : 0);
    identity.committed_point_charge_values =
        opaque_buffer(numerical.committed_point_values,
                      numerical.point_enabled != 0u ? numerical.total_point_charges : 0);
    identity.committed_point_charge_gammas =
        opaque_buffer(numerical.committed_point_gammas,
                      numerical.point_enabled != 0u ? numerical.total_point_charges : 0);
    identity.committed_point_charge_shell_potential = opaque_buffer(
        numerical.public_point_shell, numerical.point_enabled != 0u ? numerical.total_shells : 0);
    identity.committed_periodic_shifts =
        opaque_buffer(numerical.committed_periodic_shifts,
                      numerical.periodic_enabled != 0u ? numerical.total_atoms : 0);
    identity.committed_periodic_response =
        opaque_buffer(numerical.committed_periodic_response,
                      numerical.periodic_enabled != 0u ? numerical.total_response_elements : 0);
    identity.committed_generation_elements = numerical.batch_size;
    identity.numerical_eligible_elements = numerical.batch_size;
    identity.overlap_factor_generation_elements = numerical.batch_size;
    identity.overlap_factor_status_elements = numerical.batch_size;
    identity.inference_arena = opaque_address(current.inference_arena.get());
    identity.inference_epoch_consumer = opaque_address(&current.inference.epoch_consumer);
    identity.inference_results = opaque_address(&current.inference.publication_results);
    identity.inference_energies = opaque_address(current.inference.publication_results.energies);
    identity.inference_qm_forces = opaque_address(current.inference.publication_results.qm_forces);
    identity.inference_atomic_charges =
        opaque_address(current.inference.publication_results.atomic_charges);
    identity.inference_point_forces =
        opaque_address(current.inference.publication_results.point_forces);
    identity.inference_iterations =
        opaque_address(current.inference.publication_results.iterations);
    identity.inference_converged = opaque_address(current.inference.publication_results.converged);
    identity.inference_system_statuses =
        opaque_address(current.inference.publication_results.system_statuses);
    identity.inference_publication_epoch_snapshot =
        opaque_address(current.inference.publication_workspace.epoch_snapshot);
    identity.inference_publication_system_errors =
        opaque_address(current.inference.publication_diagnostics.system_errors);
    identity.inference_publication_plan_error =
        opaque_address(current.inference.publication_diagnostics.plan_error);
    identity.warm_checkpoint_generations =
        opaque_address(current.inference.warm_checkpoint_generations);
    identity.scc_state_image = opaque_address(current.iteration_arena.get());
    identity.scc_state_image_bytes = current.iteration_arena.bytes();
    identity.scc_workspace_image = opaque_address(current.eigensolver_setup_arena.get());
    identity.scc_workspace_image_bytes = current.eigensolver_setup_arena.bytes();
    identity.topology_arena_bytes = current.topology_arena.bytes();
    identity.input_arena_bytes = current.input_arena.bytes();
    identity.iteration_arena_bytes = current.iteration_arena.bytes();
    identity.eigensolver_setup_arena_bytes = current.eigensolver_setup_arena.bytes();
    identity.provider_host_workspace_bytes = current.provider_host_workspace.bytes();
    identity.force_immutable_arena_bytes = current.force_immutable_arena.bytes();
    identity.force_execution_arena_bytes = current.force_execution_arena.bytes();
    identity.numerical_refresh_arena_bytes = current.numerical_refresh_arena.bytes();
    identity.inference_arena_bytes = current.inference_arena.bytes();
    identity.numerical_host_staging_arena_bytes = current.numerical_host_staging_arena.bytes();
    identity.public_result_device_arena_bytes = current.public_result_device_arena.bytes();
    identity.public_result_host_arena_bytes = current.public_result_host_arena.bytes();
    identity.candidate_validation_arena_bytes = current.candidate_validation_arena.bytes();
    identity.native_lattice_host_staging = opaque_address(native_lattice_host_arena.get());
    identity.native_lattice_host_staging_bytes = native_lattice_host_arena.bytes();
    const Gfn2CudaTopologyStagingWorkspaceBytes topology_workspace =
        topology_staging.workspace_bytes();
    identity.topology_staging_host_bytes = topology_workspace.host_bytes;
    identity.topology_staging_device_bytes = topology_workspace.device_bytes;
    /* Cache/prepared records contain the inline host plans, arena owners, and
     * descriptor metadata. Their nested heap payloads are counted below. */
    identity.runtime_owner_host_bytes = sizeof(Impl) + sizeof(current);
    identity.host_plans_bytes = current.host.retained_host_bytes();
    identity.topology_setup_host_bytes = current.topology_owner.retained_host_bytes();
    identity.inputs_setup_host_bytes = current.inputs_owner.retained_host_bytes();
    identity.eigensolver_setup_host_bytes = current.eigensolver_owner.retained_host_bytes();
    identity.initializer_host_bytes = current.initializer.retained_host_bytes();
    identity.initializer_device_checkpoint_bytes = current.initializer.image_bytes();
    identity.scc_loop_device_control_bytes = current.scc_loop.retained_device_bytes();
    identity.retained_host_workspace_bytes =
        identity.provider_host_workspace_bytes + identity.numerical_host_staging_arena_bytes +
        identity.public_result_host_arena_bytes + identity.candidate_validation_arena_bytes +
        identity.native_lattice_host_staging_bytes + identity.topology_staging_host_bytes +
        identity.runtime_owner_host_bytes + identity.host_plans_bytes +
        identity.topology_setup_host_bytes + identity.inputs_setup_host_bytes +
        identity.eigensolver_setup_host_bytes + identity.initializer_host_bytes;
    identity.retained_device_workspace_bytes =
        identity.topology_arena_bytes + identity.input_arena_bytes +
        identity.iteration_arena_bytes + identity.eigensolver_setup_arena_bytes +
        identity.force_immutable_arena_bytes + identity.force_execution_arena_bytes +
        identity.numerical_refresh_arena_bytes + identity.inference_arena_bytes +
        identity.public_result_device_arena_bytes + identity.topology_staging_device_bytes +
        identity.initializer_device_checkpoint_bytes + identity.scc_loop_device_control_bytes;
    return identity;
  }

  std::int32_t device_id = -1;
  cudaStream_t stream = nullptr;
  Gfn2CudaTopologyStaging topology_staging;
  cusolverDnHandle_t solver = nullptr;
  cusolverDnParams_t solver_parameters = nullptr;
  syevjInfo_t solver_jacobi = nullptr;
  cublasHandle_t blas = nullptr;
  bool handles_created = false;
  std::uint64_t next_plan_token = 1u;
  std::uint64_t next_request_id = 1u;
  std::uint64_t request_submissions = 0u;
  bool request_poisoned = false;
  PinnedArena native_lattice_host_arena;
  CudaEvent native_lattice_event;
  bool native_lattice_staging_pending = false;
  bool native_lattice_staging_poisoned = false;
  ActiveRequest active_request;
  std::unique_ptr<Prepared> prepared;
  mutable std::mutex mutex;
};

Gfn2CudaExecutionCache::Gfn2CudaExecutionCache(std::int32_t device_id, void* stream)
    : impl_(std::make_unique<Impl>(device_id, stream)) {}

Gfn2CudaExecutionCache::~Gfn2CudaExecutionCache() = default;

xtbloom_status_t execute_restricted_gfn2_cuda_impl(Gfn2CudaExecutionCache& cache,
                                                   const xtbloom_batch_t& batch,
                                                   const xtbloom_compute_options_t& options,
                                                   xtbloom_batch_result_t& result,
                                                   bool require_prepared_topology,
                                                   std::string& error) {
  if (cache.impl_ == nullptr) {
    error = "CUDA GFN2 execution cache has no implementation";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  auto& implementation = *cache.impl_;
  std::lock_guard<std::mutex> lock(implementation.mutex);
  if (implementation.request_poisoned) {
    error = "CUDA execution cache is poisoned by a failed request teardown";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  if (implementation.active_request.id != 0u) {
    error = "CUDA execution cache already has an active asynchronous request";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  ScopedCudaDevice device(implementation.device_id, error);
  if (!device.ok()) return device.status();
  const auto finish = [&](xtbloom_status_t status) -> xtbloom_status_t {
    std::string restore_error;
    const xtbloom_status_t restore_status = device.restore(restore_error);
    if (restore_status == XTBLOOM_STATUS_SUCCESS) return status;
    if (status != XTBLOOM_STATUS_SUCCESS && !error.empty()) {
      error += "; additionally, " + restore_error;
    } else {
      error = std::move(restore_error);
    }
    return restore_status;
  };

  /* Keep every transaction-owned CUDA object inside this scope.  It ends
   * before finish() restores the caller's current device, so failed candidate
   * teardown and host-callback settlement always run on the context device. */
  const xtbloom_status_t transaction_status = [&]() -> xtbloom_status_t {
    xtbloom_status_t status =
        validate_cuda_stream_owner(implementation.device_id, implementation.stream, true, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = implementation.validate_public_request_pointers(batch, options, &result, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = implementation.validate_native_lattice_request_sync(batch, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_cuda_execution_availability(batch, options, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = implementation.ensure_handles(error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    const Gfn2CudaTopologyStagingDiagnostic staged =
        implementation.topology_staging.stage_and_validate(batch, error);
    if (!staged.success()) return staged.status;
    bool topology_candidate_pending =
        staged.disposition == Gfn2CudaTopologyStageDisposition::kCandidate;
    const auto abort_topology_candidate = [&]() noexcept {
      if (topology_candidate_pending) {
        implementation.topology_staging.abort_candidate();
        topology_candidate_pending = false;
      }
    };
    if (require_prepared_topology && topology_candidate_pending) {
      abort_topology_candidate();
      error = "the batch topology does not match the fixed CUDA plan topology";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    const Gfn2CudaTopologyHostSnapshot* topology =
        topology_candidate_pending ? implementation.topology_staging.candidate_snapshot()
                                   : implementation.topology_staging.committed_snapshot();
    if (topology == nullptr) {
      abort_topology_candidate();
      error = "CUDA topology staging did not expose the selected canonical snapshot";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    Gfn2CudaExecutionCache::Impl::Prepared* working = implementation.prepared.get();
    const bool reuse_runtime =
        working != nullptr && topology_snapshot_matches(*topology, options, working->host.key);
    const Gfn2CudaSccStartMode start_mode = public_scc_start_mode(options);
    if (start_mode == Gfn2CudaSccStartMode::kWarm) {
      if (!reuse_runtime) {
        abort_topology_candidate();
        error =
            "CUDA strict WARM SCC start requires the existing compatible fixed-topology runtime";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      if (!working->inference.warm_checkpoint_ready) {
        abort_topology_candidate();
        error = "CUDA strict WARM SCC start requires a preceding successful public checkpoint";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
    std::unique_ptr<Gfn2CudaExecutionCache::Impl::Prepared> candidate;
    if (!reuse_runtime) {
      TopologyKey key;
      std::vector<double> seed_positions;
      std::vector<double> seed_point_positions;
      std::vector<double> seed_point_values;
      std::vector<double> seed_point_gammas;
      std::vector<double> seed_periodic_shifts;
      std::vector<double> seed_periodic_response;
      status = make_topology_only_seed(*topology, options, key, seed_positions,
                                       seed_point_positions, seed_point_values, seed_point_gammas,
                                       seed_periodic_shifts, seed_periodic_response, error);
      if (status == XTBLOOM_STATUS_SUCCESS) {
        try {
          status = implementation.build_candidate(
              std::move(key), std::move(seed_positions), std::move(seed_point_positions),
              std::move(seed_point_values), std::move(seed_point_gammas),
              std::move(seed_periodic_shifts), std::move(seed_periodic_response), candidate, error);
        } catch (const std::bad_alloc&) {
          error = "failed to allocate a CUDA GFN2 runtime candidate";
          status = XTBLOOM_STATUS_ALLOCATION_FAILED;
        } catch (const std::exception& exception) {
          error = exception.what();
          status = XTBLOOM_STATUS_INTERNAL_ERROR;
        } catch (...) {
          error = "unknown exception while constructing a CUDA GFN2 runtime candidate";
          status = XTBLOOM_STATUS_INTERNAL_ERROR;
        }
      }
      if (status != XTBLOOM_STATUS_SUCCESS) {
        abort_topology_candidate();
        return status;
      }
      working = candidate.get();
    }

    const auto fail_working_transaction = [&](xtbloom_status_t failure) {
      const xtbloom_status_t settled =
          implementation.settle_public_submissions_locked(*working, failure, error);
      abort_topology_candidate();
      return settled;
    };

    status = implementation.reset_request_topology_error_locked(*working, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return fail_working_transaction(status);

    Gfn2CudaNumericalInputView numerical{};
    numerical.positions = batch.positions;
    numerical.point_charge_positions = batch.point_charge_positions;
    numerical.point_charge_values = batch.point_charge_values;
    numerical.point_charge_gammas = batch.point_charge_gammas;
    numerical.atomic_potential_shifts = batch.atomic_potential_shifts;
    numerical.charge_response_matrix = batch.charge_response_matrix;
    status = implementation.refresh_numerical_locked(*working, numerical, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return fail_working_transaction(status);
    status = implementation.execute_inference_locked(*working, start_mode, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return fail_working_transaction(status);
    /* Public synchronous readiness is finalized only after the completion
     * event and aggregate bridge diagnostics are known to have succeeded. */
    working->inference.warm_checkpoint_ready = false;

    PublicResultTransaction public_result_transaction;
    status = implementation.enqueue_public_result_prepare_locked(*working, batch, options, result,
                                                                 public_result_transaction, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return fail_working_transaction(status);
    status = implementation.record_public_result_completion_locked(
        *working, public_result_transaction, PublicResultTransactionPhase::kPrepareSubmitted,
        PublicResultTransactionPhase::kPrepareCompletionRecorded,
        PublicResultTransactionPhase::kPrepareCompleted, false, "CUDA public inference completion",
        error);
    if (status != XTBLOOM_STATUS_SUCCESS) return fail_working_transaction(status);
    status = implementation.wait_public_result_completion_locked(
        *working, public_result_transaction,
        PublicResultTransactionPhase::kPrepareCompletionRecorded,
        PublicResultTransactionPhase::kPrepareCompleted, "CUDA public inference completion", error);
    if (status != XTBLOOM_STATUS_SUCCESS) return fail_working_transaction(status);
    status = implementation.accept_public_result_prepare_locked(*working, public_result_transaction,
                                                                error);
    if (status != XTBLOOM_STATUS_SUCCESS) return fail_working_transaction(status);

    /* The completed prepare is accepted before the private host-upload stream
     * is settled and before any caller CUDA output is touched. */
    status = implementation.settle_public_submissions_locked(*working, status, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      abort_topology_candidate();
      return status;
    }
    if (topology_candidate_pending) {
      const Gfn2CudaTopologyStagingDiagnostic prepared_commit =
          implementation.topology_staging.prepare_candidate_commit(error);
      if (!prepared_commit.success()) return fail_working_transaction(prepared_commit.status);
      if (!implementation.topology_staging.candidate_publishable()) {
        error = "prepared CUDA topology candidate is not publishable";
        return fail_working_transaction(XTBLOOM_STATUS_INTERNAL_ERROR);
      }
    }

    status = implementation.enqueue_public_result_commit_locked(*working, public_result_transaction,
                                                                false, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return fail_working_transaction(status);

    /* A successful launch is the caller-device commit point. Ownership moves
     * must now follow even if stream completion later reports a hard fault;
     * claiming rollback could not restore caller CUDA bytes. */
    if (candidate != nullptr) implementation.prepared = std::move(candidate);
    bool ownership_published = true;
    if (topology_candidate_pending) {
      ownership_published = implementation.topology_staging.publish_candidate();
      topology_candidate_pending = false;
      if (!ownership_published) {
        error = "CUDA topology publication invariant failed after caller-device commit acceptance";
      }
    }
    constexpr const char* kCommitCompletionFailure =
        "CUDA caller-device result commit failed after its kernel was accepted; caller CUDA "
        "outputs may have been modified";
    status = implementation.record_public_result_completion_locked(
        *working, public_result_transaction, PublicResultTransactionPhase::kCommitSubmitted,
        PublicResultTransactionPhase::kCommitCompletionRecorded,
        PublicResultTransactionPhase::kCommitCompleted, true, kCommitCompletionFailure, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = implementation.wait_public_result_completion_locked(
        *working, public_result_transaction,
        PublicResultTransactionPhase::kCommitCompletionRecorded,
        PublicResultTransactionPhase::kCommitCompleted, kCommitCompletionFailure, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    std::uint32_t completed_result_flags = 0u;
    status = implementation.publish_public_results_locked(
        *working, options, result, public_result_transaction, ownership_published, true,
        completed_result_flags, error);
    return status;
  }();
  return finish(transaction_status);
}

xtbloom_status_t execute_restricted_gfn2_cuda(Gfn2CudaExecutionCache& cache,
                                              const xtbloom_batch_t& batch,
                                              const xtbloom_compute_options_t& options,
                                              xtbloom_batch_result_t& result, std::string& error) {
  return execute_restricted_gfn2_cuda_impl(cache, batch, options, result, false, error);
}

xtbloom_status_t execute_restricted_gfn2_cuda_plan(Gfn2CudaExecutionCache& cache,
                                                   const xtbloom_batch_t& batch,
                                                   const xtbloom_compute_options_t& options,
                                                   xtbloom_batch_result_t& result,
                                                   std::string& error) {
  return execute_restricted_gfn2_cuda_impl(cache, batch, options, result, true, error);
}

xtbloom_status_t enqueue_restricted_gfn2_cuda_impl(
    const std::shared_ptr<Gfn2CudaExecutionCache>& cache, const xtbloom_batch_t& batch,
    const xtbloom_compute_options_t& options, const xtbloom_batch_result_t& result,
    bool require_prepared_topology, RequestSubmission& submission, std::string& error) {
  submission = {};
  if (cache == nullptr || cache->impl_ == nullptr) {
    error = "CUDA GFN2 execution cache has no implementation";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  auto& implementation = *cache->impl_;
  xtbloom_status_t final_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  {
    std::lock_guard<std::mutex> lock(implementation.mutex);
    if (implementation.request_poisoned) {
      error = "CUDA execution cache is poisoned by a failed request teardown";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    if (implementation.active_request.id != 0u) {
      error = "CUDA execution cache already has an active asynchronous request";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    std::uint64_t request_id = implementation.next_request_id++;
    if (request_id == 0u) request_id = implementation.next_request_id++;
    implementation.active_request = {};
    implementation.active_request.id = request_id;
    struct ActiveRequestExceptionGuard {
      Gfn2CudaExecutionCache::Impl& implementation;
      bool armed = true;
      ~ActiveRequestExceptionGuard() {
        if (armed) implementation.settle_active_request_exception_noexcept_locked();
      }
      void dismiss() noexcept { armed = false; }
    } exception_guard{implementation};
    /* Valid short-prefix callers own no bytes beyond struct_size. Snapshot
     * only the validated prefix so asynchronous host publication never reads
     * optional suffix storage after enqueue returns. */
    std::memcpy(&implementation.active_request.options, &options,
                std::min<std::size_t>(options.struct_size, sizeof(xtbloom_compute_options_t)));
    std::memcpy(&implementation.active_request.result, &result,
                std::min<std::size_t>(result.struct_size, sizeof(xtbloom_batch_result_t)));
    ScopedCudaDevice device(implementation.device_id, error);
    if (!device.ok()) {
      implementation.active_request = {};
      return device.status();
    }

    const xtbloom_status_t transaction_status = [&]() -> xtbloom_status_t {
      xtbloom_status_t status =
          validate_cuda_stream_owner(implementation.device_id, implementation.stream, true, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
      status = implementation.validate_public_request_pointers(batch, options, &result, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
      status = implementation.ensure_handles(error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
      const Gfn2CudaSccStartMode start_mode = public_scc_start_mode(options);
      std::unique_ptr<Gfn2CudaExecutionCache::Impl::Prepared> candidate;
      bool topology_candidate_pending = false;
      const auto abort_topology_candidate = [&]() noexcept {
        if (topology_candidate_pending) {
          implementation.topology_staging.abort_candidate();
          topology_candidate_pending = false;
        }
      };
      Gfn2CudaExecutionCache::Impl::Prepared* working = implementation.prepared.get();
      if (start_mode == Gfn2CudaSccStartMode::kWarm) {
        if (working == nullptr) {
          error = "CUDA strict WARM SCC start requires an existing compatible prepared runtime";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        if (!working->inference.warm_checkpoint_ready) {
          error = "CUDA strict WARM SCC start requires a preceding successful public checkpoint";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
      }
      if (require_prepared_topology) {
        if (working == nullptr) {
          error = "CUDA plan request has no prepared fixed-topology runtime";
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        status = implementation.enqueue_fixed_topology_validation_locked(*working, batch, options,
                                                                         error);
        if (status != XTBLOOM_STATUS_SUCCESS) {
          return implementation.settle_public_submissions_locked(*working, status, error);
        }
      } else if (working != nullptr && context_enqueue_host_topology_probe_available(batch)) {
        /* Host topology can prove reuse or a legitimate topology replacement
         * without waiting on the owner stream, even when numerical leaves are
         * device-resident. Device topology still uses stream-ordered compare
         * or the bounded staging/setup admission below. */
        const TopologyMatch match =
            match_existing_topology(batch, options, working->host.key, error, false);
        if (match == TopologyMatch::kInvalid) return XTBLOOM_STATUS_INVALID_ARGUMENT;
        if (match == TopologyMatch::kMatch) {
          status = implementation.enqueue_fixed_topology_validation_locked(*working, batch, options,
                                                                           error);
          if (status != XTBLOOM_STATUS_SUCCESS) {
            return implementation.settle_public_submissions_locked(*working, status, error);
          }
        } else {
          working = nullptr;
        }
      } else if (working != nullptr &&
                 context_enqueue_shape_policy_matches(batch, options, working->host.key)) {
        /* Device-resident topology bytes cannot be dereferenced at admission.
         * Reuse the prepared shape and compare the immutable key in stream
         * order. A mutation completes this request with INVALID_ARGUMENT; a
         * caller that intentionally changes device topology can use a changed
         * shape/policy or the synchronous convenience path to rebuild it. */
        status = implementation.enqueue_fixed_topology_validation_locked(*working, batch, options,
                                                                         error);
        if (status != XTBLOOM_STATUS_SUCCESS) {
          return implementation.settle_public_submissions_locked(*working, status, error);
        }
      } else {
        working = nullptr;
      }
      if (start_mode == Gfn2CudaSccStartMode::kWarm && working == nullptr) {
        error = "CUDA strict WARM SCC start requires the existing compatible topology and policy";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      if (!require_prepared_topology && working == nullptr) {
        const Gfn2CudaTopologyStagingDiagnostic staged =
            implementation.topology_staging.stage_and_validate(batch, error);
        if (!staged.success()) return staged.status;
        topology_candidate_pending =
            staged.disposition == Gfn2CudaTopologyStageDisposition::kCandidate;
        const Gfn2CudaTopologyHostSnapshot* topology =
            topology_candidate_pending ? implementation.topology_staging.candidate_snapshot()
                                       : implementation.topology_staging.committed_snapshot();
        if (topology == nullptr) {
          abort_topology_candidate();
          error = "CUDA context enqueue did not expose a canonical topology snapshot";
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }

        /* Reaching this block means every reuse path above rejected and cleared
         * `working`, so the staged topology always needs a new candidate. */
        TopologyKey key;
        std::vector<double> seed_positions;
        std::vector<double> seed_point_positions;
        std::vector<double> seed_point_values;
        std::vector<double> seed_point_gammas;
        std::vector<double> seed_periodic_shifts;
        std::vector<double> seed_periodic_response;
        status = make_topology_only_seed(*topology, options, key, seed_positions,
                                         seed_point_positions, seed_point_values, seed_point_gammas,
                                         seed_periodic_shifts, seed_periodic_response, error);
        if (status == XTBLOOM_STATUS_SUCCESS) {
          try {
            status = implementation.build_candidate(
                std::move(key), std::move(seed_positions), std::move(seed_point_positions),
                std::move(seed_point_values), std::move(seed_point_gammas),
                std::move(seed_periodic_shifts), std::move(seed_periodic_response), candidate,
                error);
          } catch (const std::bad_alloc&) {
            error = "failed to allocate a CUDA context-enqueue runtime candidate";
            status = XTBLOOM_STATUS_ALLOCATION_FAILED;
          } catch (const std::exception& exception) {
            error = exception.what();
            status = XTBLOOM_STATUS_INTERNAL_ERROR;
          } catch (...) {
            error = "unknown exception while constructing a CUDA context-enqueue runtime";
            status = XTBLOOM_STATUS_INTERNAL_ERROR;
          }
        }
        if (status != XTBLOOM_STATUS_SUCCESS) {
          abort_topology_candidate();
          return status;
        }
        working = candidate.get();
        if (topology_candidate_pending) {
          /* Seal the canonical topology before numerical inference is queued.
           * The bounded wait here belongs to topology/setup admission; placing
           * it after inference would accidentally turn a new-topology enqueue
           * into a synchronous compute. Ownership is still published only
           * after the caller-output commit has been accepted. */
          const Gfn2CudaTopologyStagingDiagnostic prepared_commit =
              implementation.topology_staging.prepare_candidate_commit(error);
          if (!prepared_commit.success()) {
            abort_topology_candidate();
            return prepared_commit.status;
          }
          if (!implementation.topology_staging.candidate_publishable()) {
            abort_topology_candidate();
            error = "prepared CUDA context-enqueue topology candidate is not publishable";
            return XTBLOOM_STATUS_INTERNAL_ERROR;
          }
        }
      }
      if (!working->scc_loop.conditional_graph_ready()) {
        abort_topology_candidate();
        error =
            "asynchronous CUDA execution is unavailable because the selected SCC "
            "provider requires the bounded uncaptured fallback";
        return XTBLOOM_STATUS_NOT_SUPPORTED;
      }

      const auto fail_submitted = [&](xtbloom_status_t failure) {
        const xtbloom_status_t settled =
            implementation.settle_public_submissions_locked(*working, failure, error);
        abort_topology_candidate();
        return settled;
      };
      /* Fixed-topology comparison above owns the one request-error reset. The
       * lattice and availability gates append priority codes with atomicMax so
       * a topology mismatch remains the most specific eventual diagnostic. */
      status = implementation.enqueue_native_lattice_validation_locked(*working, batch, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return fail_submitted(status);
      const xtbloom_status_t availability =
          validate_cuda_execution_availability(batch, options, error);
      if (availability != XTBLOOM_STATUS_SUCCESS) {
        const std::uint32_t validation_code =
            availability == XTBLOOM_STATUS_NOT_IMPLEMENTED
                ? kRequestNotImplemented
                : (availability == XTBLOOM_STATUS_NOT_SUPPORTED ? kRequestNotSupported
                                                                : kRequestInvalidArgument);
        const cudaError_t mark_status = mark_request_validation_async(
            working->public_result.request_topology_error, validation_code, implementation.stream);
        if (mark_status != cudaSuccess) {
          error = cuda_error_message("CUDA request availability gate submission", mark_status);
          return fail_submitted(XTBLOOM_STATUS_INTERNAL_ERROR);
        }
        working->submitted = true;
      }
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
      if (consume_execution_test_fault(Gfn2CudaExecutionTestFault::kUnknownRequestValidationCode)) {
        constexpr std::uint32_t kUnknownRequestCode = 99u;
        const cudaError_t mark_status = cudaMemcpyAsync(
            working->public_result.request_topology_error, &kUnknownRequestCode,
            sizeof(kUnknownRequestCode), cudaMemcpyHostToDevice, implementation.stream);
        if (mark_status != cudaSuccess) {
          error = cuda_error_message("CUDA unknown request-code test injection", mark_status);
          return fail_submitted(XTBLOOM_STATUS_INTERNAL_ERROR);
        }
        working->submitted = true;
      }
#endif
      Gfn2CudaNumericalInputView numerical{};
      numerical.positions = batch.positions;
      numerical.point_charge_positions = batch.point_charge_positions;
      numerical.point_charge_values = batch.point_charge_values;
      numerical.point_charge_gammas = batch.point_charge_gammas;
      numerical.atomic_potential_shifts = batch.atomic_potential_shifts;
      numerical.charge_response_matrix = batch.charge_response_matrix;
      status = implementation.refresh_numerical_locked(*working, numerical, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return fail_submitted(status);
      status = implementation.execute_inference_locked(*working, start_mode, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return fail_submitted(status);
      working->inference.warm_checkpoint_ready = false;

      auto& active = implementation.active_request;
      const auto accept_post_launch_failure = [&](xtbloom_status_t failure) {
        const xtbloom_status_t settled =
            implementation.settle_public_submissions_locked(*working, failure, error);
        const bool owner_idle = !working->submitted;
        const bool host_idle =
            !working->numerical_host_upload_completion.pending.load(std::memory_order_acquire);
        if (owner_idle && host_idle) {
          active.completion_ready = true;
          active.completion_status = settled;
          active.result_flags = 0u;
          active.completion_error = error;
          abort_topology_candidate();
        } else {
          /* The original failure remains the eventual compute status. The
           * settlement diagnostic describes why this enqueue must retain a
           * PENDING completion owner instead of publishing an unsafe inline
           * COMPLETE state. */
          active.settlement_only_pending = true;
          active.deferred_status = failure;
          active.deferred_error = error;
          if (candidate != nullptr) {
            active.pending_prepared = std::move(candidate);
            working = active.pending_prepared.get();
          }
          if (topology_candidate_pending) {
            active.pending_topology_candidate = true;
            topology_candidate_pending = false;
          }
        }
        return XTBLOOM_STATUS_SUCCESS;
      };
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
      const bool prepare_and_settlement_failure = consume_execution_test_fault(
          Gfn2CudaExecutionTestFault::kRequestPrepareSubmissionAndSettlement);
      if (prepare_and_settlement_failure) {
        g_execution_test_fault.store(
            static_cast<std::uint32_t>(Gfn2CudaExecutionTestFault::kRequestSettlement),
            std::memory_order_release);
      }
      if (prepare_and_settlement_failure || consume_execution_test_fault(
                                                Gfn2CudaExecutionTestFault::kRequestPrepareSubmission)) {
        error = "injected CUDA asynchronous result-prepare submission failure";
        return accept_post_launch_failure(XTBLOOM_STATUS_INTERNAL_ERROR);
      }
#endif
      status = implementation.enqueue_public_result_prepare_locked(
          *working, batch, options, active.result, active.transaction, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return accept_post_launch_failure(status);
      /* Commit is device-gated by the control record produced by prepare. It
       * is submitted now so downstream work on the same CUDA stream observes
       * real outputs without requiring a host query to advance execution. */
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
      const bool commit_and_settlement_failure = consume_execution_test_fault(
          Gfn2CudaExecutionTestFault::kRequestCommitSubmissionAndSettlement);
      if (commit_and_settlement_failure) {
        g_execution_test_fault.store(
            static_cast<std::uint32_t>(Gfn2CudaExecutionTestFault::kRequestSettlement),
            std::memory_order_release);
      }
      if (commit_and_settlement_failure || consume_execution_test_fault(
                                               Gfn2CudaExecutionTestFault::kRequestCommitSubmission)) {
        error = "injected CUDA asynchronous result-commit submission failure";
        return accept_post_launch_failure(XTBLOOM_STATUS_INTERNAL_ERROR);
      }
#endif
      status = implementation.enqueue_public_result_commit_locked(*working, active.transaction,
                                                                  true, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return accept_post_launch_failure(status);
      if (candidate != nullptr) {
        implementation.prepared = std::move(candidate);
        working = implementation.prepared.get();
      }
      if (topology_candidate_pending) {
        if (!implementation.topology_staging.publish_candidate()) {
          /* Prepared ownership and caller-output commit are already accepted,
           * so rollback is impossible. Poison the cache rather than exposing
           * a new runtime key beside the old committed topology snapshot. */
          implementation.request_poisoned = true;
          working->numerical.host_staging_poisoned = true;
          active.deferred_status = XTBLOOM_STATUS_INTERNAL_ERROR;
          active.deferred_error =
              "CUDA context-enqueue topology publication invariant failed after caller-output "
              "commit acceptance";
        }
        topology_candidate_pending = false;
      }
      ++implementation.request_submissions;
      if (working->numerical_host_upload_completion.pending.load(std::memory_order_acquire)) {
        const cudaError_t release_status = cudaStreamWaitEvent(
            implementation.stream, working->numerical_host_release_complete.get(), 0u);
        if (release_status != cudaSuccess) {
          /* The commit is already accepted. Preserve a PENDING request and
           * let its exceptional settlement fence the two exact streams. */
          active.deferred_status = XTBLOOM_STATUS_INTERNAL_ERROR;
          active.deferred_error = cuda_error_message(
              "CUDA numerical host-release ordering for request completion", release_status);
          return XTBLOOM_STATUS_SUCCESS;
        }
        active.host_upload_release_ordered = true;
      }
      status = implementation.record_public_result_completion_locked(
          *working, active.transaction, PublicResultTransactionPhase::kCommitSubmitted,
          PublicResultTransactionPhase::kCommitCompletionRecorded,
          PublicResultTransactionPhase::kCommitCompleted, true,
          "CUDA asynchronous caller-output completion", error);
      if (status != XTBLOOM_STATUS_SUCCESS) {
        /* The commit was accepted, but neither event recording nor the
         * helper's exact-stream fallback settled it. Keep the request PENDING:
         * wait/destroy must synchronize the owner stream because the event
         * object may still name an older completed phase. */
        active.deferred_status = XTBLOOM_STATUS_INTERNAL_ERROR;
        active.deferred_error = error;
        return XTBLOOM_STATUS_SUCCESS;
      }
      if (active.transaction.phase == PublicResultTransactionPhase::kCommitCompleted) {
        const bool host_release_pending =
            working->numerical_host_upload_completion.pending.load(std::memory_order_acquire);
        if (host_release_pending && active.host_upload_release_ordered) {
          active.deferred_status = XTBLOOM_STATUS_INTERNAL_ERROR;
          active.deferred_error =
              "CUDA numerical host release remained pending after "
              "owner-stream completion fallback";
          return XTBLOOM_STATUS_SUCCESS;
        }
        cudaError_t host_status = cudaSuccess;
        if (host_release_pending) {
          host_status = cudaStreamSynchronize(working->numerical_host_completion_stream.get());
        }
        if (host_status != cudaSuccess) {
          active.deferred_status = XTBLOOM_STATUS_INTERNAL_ERROR;
          active.deferred_error = cuda_error_message(
              "CUDA numerical host snapshot settlement after event-record fallback", host_status);
          return XTBLOOM_STATUS_SUCCESS;
        }
        working->numerical_host_upload_completion.pending.store(false, std::memory_order_release);
        implementation.finalize_active_request_after_commit_locked(*working);
      }
      return XTBLOOM_STATUS_SUCCESS;
    }();

    std::string restore_error;
    const xtbloom_status_t restore_status = device.restore(restore_error);
    final_status = transaction_status;
    if (restore_status != XTBLOOM_STATUS_SUCCESS) {
      if (transaction_status == XTBLOOM_STATUS_SUCCESS) {
        auto& active = implementation.active_request;
        active.deferred_status = XTBLOOM_STATUS_INTERNAL_ERROR;
        /* Device restoration is part of the accepted request contract. Close
         * readiness before any potentially-throwing diagnostic allocation. */
        auto* const active_prepared =
            implementation.active_request.pending_prepared != nullptr
                ? implementation.active_request.pending_prepared.get()
                : implementation.prepared.get();
        if (active_prepared != nullptr) {
          active_prepared->inference.warm_checkpoint_ready = false;
        }
        if (!active.deferred_error.empty()) active.deferred_error += "; additionally, ";
        active.deferred_error += restore_error;
        if (active.completion_ready) {
          active.completion_status = XTBLOOM_STATUS_INTERNAL_ERROR;
          active.result_flags = 0u;
          if (!active.completion_error.empty()) active.completion_error += "; additionally, ";
          active.completion_error += active.deferred_error;
        }
        final_status = XTBLOOM_STATUS_SUCCESS;
      } else {
        final_status = restore_status;
        if (!error.empty() && error != restore_error) {
          error += "; additionally, " + restore_error;
        } else {
          error = std::move(restore_error);
        }
      }
    }
    if (final_status != XTBLOOM_STATUS_SUCCESS) {
      implementation.active_request = {};
    } else if (implementation.active_request.completion_ready) {
      submission.completed_inline = true;
      submission.completion_status = implementation.active_request.completion_status;
      submission.result_flags = implementation.active_request.result_flags;
      submission.completion_error = implementation.active_request.completion_error;
      implementation.active_request = {};
    }
    exception_guard.dismiss();
  }

  if (final_status != XTBLOOM_STATUS_SUCCESS) return final_status;
  if (!submission.completed_inline) {
    /* The plan already owns this shared_ptr/control block. Casting and copying
     * it into the request is allocation-free in steady state. */
    submission.pending = std::static_pointer_cast<RequestCompletion>(cache);
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t enqueue_restricted_gfn2_cuda_plan(
    const std::shared_ptr<Gfn2CudaExecutionCache>& cache, const xtbloom_batch_t& batch,
    const xtbloom_compute_options_t& options, const xtbloom_batch_result_t& result,
    RequestSubmission& submission, std::string& error) {
  return enqueue_restricted_gfn2_cuda_impl(cache, batch, options, result, true, submission, error);
}

xtbloom_status_t enqueue_restricted_gfn2_cuda(const std::shared_ptr<Gfn2CudaExecutionCache>& cache,
                                              const xtbloom_batch_t& batch,
                                              const xtbloom_compute_options_t& options,
                                              const xtbloom_batch_result_t& result,
                                              RequestSubmission& submission, std::string& error) {
  return enqueue_restricted_gfn2_cuda_impl(cache, batch, options, result, false, submission, error);
}

xtbloom_status_t Gfn2CudaExecutionCache::probe(bool wait,
                                               RequestCompletionResult& result) noexcept {
  result = {};
  if (impl_ == nullptr) {
    result.completion_error = "CUDA GFN2 execution cache has no implementation";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  try {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& active = impl_->active_request;
    Gfn2CudaExecutionCache::Impl::Prepared* current_owner =
        active.pending_prepared != nullptr ? active.pending_prepared.get() : impl_->prepared.get();
    if (active.id == 0u || current_owner == nullptr) {
      result.completion_error = "CUDA request completion does not own the active cache submission";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    if (active.completion_ready) {
      result.complete = true;
      result.completion_status = active.completion_status;
      result.result_flags = active.result_flags;
      result.completion_error = active.completion_error;
      return XTBLOOM_STATUS_SUCCESS;
    }

    std::string device_error;
    ScopedCudaDevice device(impl_->device_id, device_error);
    if (!device.ok()) {
      result.completion_error = std::move(device_error);
      return device.status();
    }
    auto& current = *current_owner;
    auto& transaction = active.transaction;

    if (active.settlement_only_pending) {
      bool incomplete = false;
      const auto observe_stream = [&](cudaStream_t stream, const char* operation,
                                      bool& submitted) -> xtbloom_status_t {
        if (!submitted) return XTBLOOM_STATUS_SUCCESS;
        const cudaError_t cuda_status = wait ? cudaStreamSynchronize(stream) : cudaStreamQuery(stream);
        if (!wait && cuda_status == cudaErrorNotReady) {
          incomplete = true;
          return XTBLOOM_STATUS_SUCCESS;
        }
        if (cuda_status != cudaSuccess) {
          result.completion_error = cuda_error_message(operation, cuda_status);
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        submitted = false;
        return XTBLOOM_STATUS_SUCCESS;
      };

      xtbloom_status_t status = observe_stream(impl_->stream, "CUDA post-launch request settlement",
                                               current.submitted);
      bool host_pending =
          current.numerical_host_upload_completion.pending.load(std::memory_order_acquire);
      if (status == XTBLOOM_STATUS_SUCCESS && !incomplete && host_pending) {
        status = observe_stream(current.numerical_host_completion_stream.get(),
                                "CUDA post-launch host snapshot settlement", host_pending);
        if (status == XTBLOOM_STATUS_SUCCESS && !host_pending) {
          current.numerical_host_upload_completion.pending.store(false,
                                                                 std::memory_order_release);
        }
      }
      if (status != XTBLOOM_STATUS_SUCCESS) {
        std::string ignored;
        (void)device.restore(ignored);
        return status;
      }
      if (!incomplete && !current.submitted &&
          !current.numerical_host_upload_completion.pending.load(std::memory_order_acquire)) {
        active.completion_ready = true;
        active.completion_status = active.deferred_status;
        active.result_flags = 0u;
        active.completion_error = active.deferred_error;
        active.settlement_only_pending = false;
        if (active.pending_topology_candidate) {
          impl_->topology_staging.abort_candidate();
          active.pending_topology_candidate = false;
        }
        active.pending_prepared.reset();
      }

      std::string restore_error;
      const xtbloom_status_t restore_status = device.restore(restore_error);
      if (restore_status != XTBLOOM_STATUS_SUCCESS) {
        result.completion_error = std::move(restore_error);
        return restore_status;
      }
      if (active.completion_ready) {
        result.complete = true;
        result.completion_status = active.completion_status;
        result.result_flags = active.result_flags;
        result.completion_error = active.completion_error;
      }
      return XTBLOOM_STATUS_SUCCESS;
    }

    bool incomplete = false;
    const auto observe_event = [&]() -> xtbloom_status_t {
      if (transaction.phase == PublicResultTransactionPhase::kCommitCompleted) {
        return XTBLOOM_STATUS_SUCCESS;
      }
      if (transaction.phase == PublicResultTransactionPhase::kCommitSubmitted) {
        const cudaError_t stream_status =
            wait ? cudaStreamSynchronize(impl_->stream) : cudaStreamQuery(impl_->stream);
        if (!wait && stream_status == cudaErrorNotReady) {
          incomplete = true;
          return XTBLOOM_STATUS_SUCCESS;
        }
        if (stream_status != cudaSuccess) {
          result.completion_error = cuda_error_message(
              "CUDA asynchronous caller-output settlement after event-record failure",
              stream_status);
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        current.submitted = false;
        transaction.phase = PublicResultTransactionPhase::kCommitCompleted;
        return XTBLOOM_STATUS_SUCCESS;
      }
      if (transaction.phase != PublicResultTransactionPhase::kCommitCompletionRecorded ||
          current.public_result_completion_event.get() == nullptr) {
        result.completion_error = "CUDA request has an invalid completion-event phase";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      const cudaError_t cuda_status =
          wait ? cudaEventSynchronize(current.public_result_completion_event.get())
               : cudaEventQuery(current.public_result_completion_event.get());
      if (!wait && cuda_status == cudaErrorNotReady) {
        incomplete = true;
        return XTBLOOM_STATUS_SUCCESS;
      }
      if (cuda_status != cudaSuccess) {
        result.completion_error =
            cuda_error_message("CUDA asynchronous caller-output completion", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      current.submitted = false;
      transaction.phase = PublicResultTransactionPhase::kCommitCompleted;
      return XTBLOOM_STATUS_SUCCESS;
    };

    const auto observe_private_stream = [&]() -> xtbloom_status_t {
      if (!current.numerical_host_completion_stream.valid()) return XTBLOOM_STATUS_SUCCESS;
      if (!current.numerical_host_upload_completion.pending.load(std::memory_order_acquire)) {
        return XTBLOOM_STATUS_SUCCESS;
      }
      if (active.host_upload_release_ordered) {
        result.completion_error =
            "CUDA numerical host release remained pending after request completion";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      const cudaError_t cuda_status =
          wait ? cudaStreamSynchronize(current.numerical_host_completion_stream.get())
               : cudaStreamQuery(current.numerical_host_completion_stream.get());
      if (!wait && cuda_status == cudaErrorNotReady) {
        incomplete = true;
        return XTBLOOM_STATUS_SUCCESS;
      }
      if (cuda_status != cudaSuccess) {
        result.completion_error =
            cuda_error_message("CUDA numerical host snapshot completion", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      current.numerical_host_upload_completion.pending.store(false, std::memory_order_release);
      return XTBLOOM_STATUS_SUCCESS;
    };

    xtbloom_status_t status = observe_event();
    if (status != XTBLOOM_STATUS_SUCCESS) {
      std::string ignored;
      (void)device.restore(ignored);
      return status;
    }
    if (!incomplete) {
      status = observe_private_stream();
      if (status != XTBLOOM_STATUS_SUCCESS) {
        std::string ignored;
        (void)device.restore(ignored);
        return status;
      }
    }
    if (!incomplete) {
      impl_->finalize_active_request_after_commit_locked(current);
    }

    std::string restore_error;
    const xtbloom_status_t restore_status = device.restore(restore_error);
    if (restore_status != XTBLOOM_STATUS_SUCCESS) {
      /* Restoring the query thread's device is part of accessing the request,
       * not the submitted computation. Preserve an already finalized compute
       * snapshot so a later query can retrieve it, but report this call's
       * access failure through the API return channel. */
      result.completion_error = std::move(restore_error);
      return restore_status;
    }
    if (active.completion_ready) {
      result.complete = true;
      result.completion_status = active.completion_status;
      result.result_flags = active.result_flags;
      result.completion_error = active.completion_error;
    }
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    result.completion_error = "failed to allocate while probing CUDA request completion";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  } catch (const std::exception& exception) {
    result.completion_error = exception.what();
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  } catch (...) {
    result.completion_error = "unknown exception while probing CUDA request completion";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
}

void Gfn2CudaExecutionCache::settle_noexcept() noexcept {
  if (impl_ == nullptr) return;
  try {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->active_request.id == 0u) return;
    if (impl_->active_request.completion_ready) {
      if (impl_->active_request.pending_topology_candidate) {
        impl_->topology_staging.abort_candidate();
      }
      impl_->active_request = {};
      return;
    }
    bool settled = false;
    Impl::Prepared* const current_owner = impl_->active_request.pending_prepared != nullptr
                                              ? impl_->active_request.pending_prepared.get()
                                              : impl_->prepared.get();
    if (current_owner != nullptr) {
      auto& current = *current_owner;
      std::string ignored;
      ScopedCudaDevice device(impl_->device_id, ignored);
      if (device.ok()) {
        cudaError_t owner_status = cudaSuccess;
        if (current.submitted) {
          owner_status = cudaErrorInvalidValue;
          if (impl_->active_request.transaction.phase ==
                  PublicResultTransactionPhase::kCommitCompletionRecorded &&
              current.public_result_completion_event.get() != nullptr) {
            owner_status = cudaEventSynchronize(current.public_result_completion_event.get());
            if (owner_status != cudaSuccess) owner_status = cudaStreamSynchronize(impl_->stream);
          } else {
            /* A failed record leaves the reusable event pointing at an older
             * phase; only the exact owner stream can settle this commit. */
            owner_status = cudaStreamSynchronize(impl_->stream);
          }
          if (owner_status == cudaSuccess) current.submitted = false;
        }
        cudaError_t host_status = cudaSuccess;
        if (current.numerical_host_completion_stream.valid() &&
            current.numerical_host_upload_completion.pending.load(std::memory_order_acquire)) {
          host_status = cudaStreamSynchronize(current.numerical_host_completion_stream.get());
          if (host_status == cudaSuccess) {
            current.numerical_host_upload_completion.pending.store(false,
                                                                   std::memory_order_release);
          }
        }
        settled = owner_status == cudaSuccess && host_status == cudaSuccess;
        (void)device.restore(ignored);
      }
    }
    if (settled) {
      if (impl_->active_request.pending_topology_candidate) {
        impl_->topology_staging.abort_candidate();
      }
      impl_->active_request = {};
    } else {
      impl_->request_poisoned = true;
      if (current_owner != nullptr) current_owner->numerical.host_staging_poisoned = true;
    }
  } catch (...) {
    /* Request destruction cannot report. Prepared teardown retains a second
     * exact-stream safety net if CUDA or allocation state is already broken. */
  }
}

xtbloom_status_t Gfn2CudaExecutionCache::prepare_host(const xtbloom_batch_t& batch,
                                                      const xtbloom_compute_options_t& options,
                                                      bool& reused, std::string& error) {
  reused = false;
  if (impl_ == nullptr) {
    error = "CUDA GFN2 execution cache has no implementation";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  xtbloom_status_t status = impl_->ensure_handles(error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  if (impl_->prepared != nullptr) {
    const TopologyMatch match =
        match_existing_topology(batch, options, impl_->prepared->host.key, error);
    if (match == TopologyMatch::kMatch) {
      Gfn2CudaNumericalInputView numerical{};
      numerical.positions = batch.positions;
      numerical.point_charge_positions = batch.point_charge_positions;
      numerical.point_charge_values = batch.point_charge_values;
      numerical.point_charge_gammas = batch.point_charge_gammas;
      numerical.atomic_potential_shifts = batch.atomic_potential_shifts;
      numerical.charge_response_matrix = batch.charge_response_matrix;
      status = impl_->refresh_numerical_locked(*impl_->prepared, numerical, error);
      reused = status == XTBLOOM_STATUS_SUCCESS;
      return status;
    }
    if (match == TopologyMatch::kInvalid) {
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  TopologyKey key;
  std::vector<double> positions;
  std::vector<double> point_positions;
  std::vector<double> point_values;
  std::vector<double> point_gammas;
  std::vector<double> periodic_shifts;
  std::vector<double> periodic_response;
  std::unique_ptr<Impl::Prepared> candidate;
  try {
    /* Topology staging allocates host vectors too. Keep it in the same
     * exception-to-status boundary as candidate construction so no allocation
     * failure can escape toward the C ABI. */
    status = make_topology_key(batch, options, key, positions, point_positions, point_values,
                               point_gammas, periodic_shifts, periodic_response, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = impl_->build_candidate(std::move(key), std::move(positions),
                                    std::move(point_positions), std::move(point_values),
                                    std::move(point_gammas), std::move(periodic_shifts),
                                    std::move(periodic_response), candidate, error);
  } catch (const std::bad_alloc&) {
    error = "failed to allocate host metadata for the CUDA GFN2 runtime candidate";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  } catch (const std::exception& exception) {
    error = exception.what();
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  } catch (...) {
    error = "unknown exception while constructing CUDA GFN2 runtime candidate";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  /* A direct host preparation promises an immediately executable numerical
   * runtime. Publish the initial geometry through the same transaction used
   * by reuse and public compute so pair-list/D4 leaves, overlap factors, the
   * common generation, and eligibility all become epoch 1 together. */
  Gfn2CudaNumericalInputView numerical{};
  numerical.positions = batch.positions;
  numerical.point_charge_positions = batch.point_charge_positions;
  numerical.point_charge_values = batch.point_charge_values;
  numerical.point_charge_gammas = batch.point_charge_gammas;
  numerical.atomic_potential_shifts = batch.atomic_potential_shifts;
  numerical.charge_response_matrix = batch.charge_response_matrix;
  status = impl_->refresh_numerical_locked(*candidate, numerical, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  impl_->prepared = std::move(candidate);
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t Gfn2CudaExecutionCache::prepare_topology_only(
    const xtbloom_batch_t& batch, const xtbloom_compute_options_t& options, std::string& error) {
  if (impl_ == nullptr) {
    error = "CUDA GFN2 execution cache has no implementation";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  ScopedCudaDevice device(impl_->device_id, error);
  if (!device.ok()) return device.status();
  const auto finish = [&](xtbloom_status_t status) {
    std::string restore_error;
    const xtbloom_status_t restore_status = device.restore(restore_error);
    if (restore_status == XTBLOOM_STATUS_SUCCESS) return status;
    if (status != XTBLOOM_STATUS_SUCCESS && !error.empty()) {
      error += "; additionally, " + restore_error;
    } else {
      error = std::move(restore_error);
    }
    return restore_status;
  };

  const xtbloom_status_t setup_status = [&]() -> xtbloom_status_t {
    xtbloom_status_t status =
        validate_cuda_stream_owner(impl_->device_id, impl_->stream, true, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = impl_->validate_public_request_pointers(batch, options, nullptr, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = impl_->validate_native_lattice_request_sync(batch, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_cuda_execution_availability(batch, options, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = impl_->ensure_handles(error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    const Gfn2CudaTopologyStagingDiagnostic staged =
        impl_->topology_staging.stage_and_validate(batch, error);
    if (!staged.success()) return staged.status;
    bool candidate_pending = staged.disposition == Gfn2CudaTopologyStageDisposition::kCandidate;
    const auto abort_candidate = [&]() noexcept {
      if (candidate_pending) {
        impl_->topology_staging.abort_candidate();
        candidate_pending = false;
      }
    };
    const Gfn2CudaTopologyHostSnapshot* topology =
        candidate_pending ? impl_->topology_staging.candidate_snapshot()
                          : impl_->topology_staging.committed_snapshot();
    if (topology == nullptr) {
      abort_candidate();
      error = "CUDA plan setup did not expose a canonical topology snapshot";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    if (impl_->prepared != nullptr &&
        topology_snapshot_matches(*topology, options, impl_->prepared->host.key)) {
      if (candidate_pending) {
        const Gfn2CudaTopologyStagingDiagnostic committed =
            impl_->topology_staging.commit_candidate(error);
        candidate_pending = false;
        if (!committed.success()) return committed.status;
      }
      error.clear();
      return XTBLOOM_STATUS_SUCCESS;
    }

    TopologyKey key;
    std::vector<double> positions;
    std::vector<double> point_positions;
    std::vector<double> point_values;
    std::vector<double> point_gammas;
    std::vector<double> periodic_shifts;
    std::vector<double> periodic_response;
    std::unique_ptr<Impl::Prepared> candidate;
    try {
      status =
          make_topology_only_seed(*topology, options, key, positions, point_positions, point_values,
                                  point_gammas, periodic_shifts, periodic_response, error);
      if (status == XTBLOOM_STATUS_SUCCESS) {
        status = impl_->build_candidate(std::move(key), std::move(positions),
                                        std::move(point_positions), std::move(point_values),
                                        std::move(point_gammas), std::move(periodic_shifts),
                                        std::move(periodic_response), candidate, error);
      }
    } catch (const std::bad_alloc&) {
      error = "failed to allocate host metadata for the CUDA GFN2 runtime candidate";
      status = XTBLOOM_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception& exception) {
      error = exception.what();
      status = XTBLOOM_STATUS_INTERNAL_ERROR;
    } catch (...) {
      error = "unknown exception while constructing CUDA GFN2 runtime candidate";
      status = XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    if (status != XTBLOOM_STATUS_SUCCESS) {
      abort_candidate();
      return status;
    }
    if (candidate_pending) {
      const Gfn2CudaTopologyStagingDiagnostic committed =
          impl_->topology_staging.commit_candidate(error);
      candidate_pending = false;
      if (!committed.success()) return committed.status;
    }
    impl_->prepared = std::move(candidate);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }();
  return finish(setup_status);
}

xtbloom_status_t Gfn2CudaExecutionCache::refresh_numerical_async(
    const Gfn2CudaNumericalInputView& input, std::string& error) {
  if (impl_ == nullptr) {
    error = "CUDA GFN2 execution cache has no implementation";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  xtbloom_status_t status = impl_->ensure_handles(error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (impl_->prepared == nullptr) {
    error = "CUDA GFN2 numerical refresh requires a prepared fixed topology";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return impl_->refresh_numerical_locked(*impl_->prepared, input, error);
}

xtbloom_status_t Gfn2CudaExecutionCache::execute_inference_async(Gfn2CudaSccStartMode mode,
                                                                 std::string& error) {
  if (impl_ == nullptr) {
    error = "CUDA GFN2 execution cache has no implementation";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  xtbloom_status_t status = impl_->ensure_handles(error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (impl_->prepared == nullptr) {
    error = "CUDA GFN2 inference requires a prepared numerical/runtime binding";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return impl_->execute_inference_locked(*impl_->prepared, mode, error);
}

bool Gfn2CudaExecutionCache::valid() const noexcept {
  if (impl_ == nullptr) return false;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->prepared != nullptr;
}

Gfn2CudaExecutionIdentity Gfn2CudaExecutionCache::identity() const noexcept {
  if (impl_ == nullptr) return {};
  std::lock_guard<std::mutex> lock(impl_->mutex);
  Gfn2CudaExecutionIdentity identity = impl_->snapshot();
  identity.request_completion_owner = opaque_address(this);
  return identity;
}

#if defined(XTBLOOM_CUDA_TEST_HOOKS)
xtbloom_status_t Gfn2CudaExecutionCache::validate_native_lattice_test_only(
    const xtbloom_batch_t& batch, std::string& error) {
  if (impl_ == nullptr) {
    error = "CUDA GFN2 execution cache has no implementation";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  ScopedCudaDevice device(impl_->device_id, error);
  if (!device.ok()) return device.status();
  const xtbloom_status_t status = impl_->validate_native_lattice_request_sync(batch, error);
  std::string restore_error;
  const xtbloom_status_t restore_status = device.restore(restore_error);
  if (restore_status == XTBLOOM_STATUS_SUCCESS) return status;
  if (status != XTBLOOM_STATUS_SUCCESS && !error.empty()) {
    error += "; additionally, " + restore_error;
  } else {
    error = std::move(restore_error);
  }
  return restore_status;
}

Gfn2CudaNativeLatticeTestIdentity Gfn2CudaExecutionCache::native_lattice_test_identity()
    const noexcept {
  if (impl_ == nullptr) return {};
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return {opaque_address(impl_->native_lattice_host_arena.get()),
          impl_->native_lattice_host_arena.bytes(), impl_->native_lattice_staging_pending,
          impl_->native_lattice_staging_poisoned};
}

void reset_gfn2_cuda_execution_test_state() noexcept {
  g_execution_test_fault.store(static_cast<std::uint32_t>(Gfn2CudaExecutionTestFault::kNone),
                               std::memory_order_release);
  g_native_lattice_allocation_faults.store(0u, std::memory_order_relaxed);
  g_native_lattice_completion_faults.store(0u, std::memory_order_relaxed);
  g_native_lattice_teardown_faults.store(0u, std::memory_order_relaxed);
  g_quarantined_native_lattice_arenas.store(0u, std::memory_order_relaxed);
  g_quarantined_native_lattice_bytes.store(0u, std::memory_order_relaxed);
}

void arm_gfn2_cuda_execution_test_fault(Gfn2CudaExecutionTestFault fault) noexcept {
  g_execution_test_fault.store(static_cast<std::uint32_t>(fault), std::memory_order_release);
}

Gfn2CudaExecutionTestStats gfn2_cuda_execution_test_stats() noexcept {
  Gfn2CudaExecutionTestStats stats{};
  stats.native_lattice_allocation_faults =
      g_native_lattice_allocation_faults.load(std::memory_order_relaxed);
  stats.native_lattice_completion_faults =
      g_native_lattice_completion_faults.load(std::memory_order_relaxed);
  stats.native_lattice_teardown_faults =
      g_native_lattice_teardown_faults.load(std::memory_order_relaxed);
  stats.quarantined_native_lattice_arenas =
      g_quarantined_native_lattice_arenas.load(std::memory_order_relaxed);
  stats.quarantined_native_lattice_bytes =
      g_quarantined_native_lattice_bytes.load(std::memory_order_relaxed);
  return stats;
}
#endif

}  // namespace xtbloom::detail
