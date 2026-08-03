#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <algorithm>
#include <array>
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
#include "backends/cuda/gfn2_preprocessing.cuh"
#include "backends/cuda/gfn2_scc_iteration_arena.cuh"
#include "backends/cuda/gfn2_scc_iteration_initialize.cuh"
#include "backends/cuda/gfn2_scc_iteration_reports.cuh"
#include "backends/cuda/gfn2_scc_setup_eigensolver.cuh"
#include "backends/cuda/gfn2_scc_setup_inputs.cuh"
#include "backends/cuda/gfn2_scc_setup_topology.hpp"
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
#include "model/gfn2/mulliken.hpp"
#include "model/gfn2/periodic_embedding.hpp"
#include "model/gfn2/repulsion.hpp"
#include "model/gfn2/scc_driver.hpp"
#include "model/gfn2/scc_mixer.hpp"
#include "model/gfn2/wavefunction.hpp"
#include "runtime/backend.hpp"
#include "runtime/gfn2_cuda_execution.hpp"

namespace gpuxtb::detail {
namespace {

using namespace gpuxtb::detail::cuda;
using namespace gpuxtb::detail::gfn2;

constexpr std::int64_t kMixerHistory = 8;
constexpr double kMixerDamping = 0.4;
constexpr std::uint64_t kInitialGeometryGeneration = 1u;
constexpr std::uint64_t kInitialStateGeneration = 1u;
constexpr std::size_t kArenaAlignment = 256u;

std::uintptr_t opaque_address(const void* pointer) noexcept {
  return reinterpret_cast<std::uintptr_t>(pointer);
}

template <typename T>
Gfn2SccSetupHostArray<T> setup_array(const std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), static_cast<std::int64_t>(values.size())};
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

template <typename T>
gpuxtb_status_t copy_host_buffer(const char* name, const gpuxtb_const_buffer_t& buffer,
                                 std::int64_t elements, std::vector<T>& output, std::string& error,
                                 bool allow_absent = false) {
  std::size_t required = 0u;
  if (!checked_bytes(elements, sizeof(T), required)) {
    error = std::string(name) + " extent overflows size_t";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (elements == 0 && buffer.data == nullptr && allow_absent) {
    output.clear();
    return GPUXTB_STATUS_SUCCESS;
  }
  if (buffer.memory_space != GPUXTB_MEMORY_HOST || buffer.reserved != 0u ||
      (required != 0u && buffer.data == nullptr) || buffer.size_bytes < required) {
    error = std::string(name) + " is not a sufficiently large host buffer";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  output.resize(static_cast<std::size_t>(elements));
  if (required != 0u) {
    std::memcpy(output.data(), buffer.data, required);
  }
  return GPUXTB_STATUS_SUCCESS;
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

  cudaError_t allocate(std::size_t bytes) noexcept {
    reset();
    bytes_ = bytes;
    return bytes == 0u ? cudaSuccess : cudaMallocHost(&pointer_, bytes);
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

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0u;
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
  std::int64_t d4_pair_elements = 0;
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
  const std::int64_t* d4_pair_offsets = nullptr;

  const std::uint8_t* requested = nullptr;
  const std::uint8_t* preprocessing_published = nullptr;
  std::uint8_t* eligible = nullptr;
  std::uint64_t* committed_generations = nullptr;

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

  const double* candidate_d4_pairs = nullptr;
  const double* candidate_d4_coordination = nullptr;
  double* public_d4_pairs = nullptr;
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
  if (threadIdx.x == 0) binding.eligible[system] = plan_healthy && peer_healthy ? 1u : 0u;
}

__global__ void commit_gfn2_numerical_refresh_kernel(NumericalRefreshDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= binding.batch_size) return;
  const std::uint64_t generation = *binding.geometry_epoch;
  const bool publish = binding.eligible[system] == 1u && binding.factor_statuses[system] == 0u &&
                       binding.factor_generations[system] == generation;
  if (threadIdx.x == 0) binding.eligible[system] = publish ? 1u : 0u;
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
  for (std::int64_t index = matrix_begin * 3 + threadIdx.x; index < matrix_end * 3;
       index += blockDim.x) {
    binding.public_dipole[index] = binding.candidate_dipole[index];
  }
  for (std::int64_t index = matrix_begin * 6 + threadIdx.x; index < matrix_end * 6;
       index += blockDim.x) {
    binding.public_quadrupole[index] = binding.candidate_quadrupole[index];
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

  if (binding.d4_enabled != 0u) {
    const std::int64_t d4_begin = binding.d4_pair_offsets[system];
    const std::int64_t d4_end = binding.d4_pair_offsets[system + 1];
    for (std::int64_t index = d4_begin * kGfn2D4PairDataElements + threadIdx.x;
         index < d4_end * kGfn2D4PairDataElements; index += blockDim.x) {
      binding.public_d4_pairs[index] = binding.candidate_d4_pairs[index];
    }
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

struct TopologyKey {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int64_t> point_offsets;
  std::vector<std::int64_t> response_offsets;
  std::uint32_t flags = 0u;
  std::int32_t maximum_iterations = 0;
  double charge_tolerance = 0.0;
  double energy_tolerance = 0.0;
  double electronic_temperature = 0.0;
  bool periodic_enabled = false;

  std::uint64_t fingerprint() const noexcept {
    std::uint64_t hash = 0x4750555854424b59ULL;
    const auto append_vector = [&hash](const auto& values) {
      hash_append(hash, values.size());
      for (const auto value : values) {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, double>) {
          hash_append(hash, double_bits(value));
        } else {
          hash_append(hash, static_cast<std::uint64_t>(value));
        }
      }
    };
    append_vector(atom_offsets);
    append_vector(atomic_numbers);
    append_vector(molecular_charges);
    append_vector(unpaired_electrons);
    append_vector(point_offsets);
    append_vector(response_offsets);
    hash_append(hash, flags);
    hash_append(hash, static_cast<std::uint32_t>(maximum_iterations));
    hash_append(hash, double_bits(charge_tolerance));
    hash_append(hash, double_bits(energy_tolerance));
    hash_append(hash, double_bits(electronic_temperature));
    hash_append(hash, periodic_enabled ? 1u : 0u);
    return hash == 0u ? 1u : hash;
  }
};

enum class TopologyMatch { kMatch, kMismatch, kInvalid };

bool valid_host_extent(const gpuxtb_const_buffer_t& buffer, std::size_t required,
                       bool allow_absent = false) noexcept {
  if (required == 0u && buffer.data == nullptr && allow_absent) return true;
  return buffer.memory_space == GPUXTB_MEMORY_HOST && buffer.reserved == 0u &&
         (required == 0u || buffer.data != nullptr) && buffer.size_bytes >= required;
}

template <typename T>
bool buffer_equals(const gpuxtb_const_buffer_t& buffer, const std::vector<T>& expected) noexcept {
  const std::size_t bytes = expected.size() * sizeof(T);
  return valid_host_extent(buffer, bytes, expected.empty()) &&
         (bytes == 0u || std::memcmp(buffer.data, expected.data(), bytes) == 0);
}

bool double_buffer_equals(const gpuxtb_const_buffer_t& buffer,
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

bool finite_double_buffer(const gpuxtb_const_buffer_t& buffer, std::int64_t elements,
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
TopologyMatch match_existing_topology(const gpuxtb_batch_t& batch,
                                      const gpuxtb_compute_options_t& options,
                                      const TopologyKey& key, std::string& error) {
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
  if (options.model != GPUXTB_MODEL_GFN2_XTB || options.flags != key.flags ||
      options.max_scc_iterations != key.maximum_iterations ||
      options.charge_tolerance != key.charge_tolerance ||
      options.energy_tolerance != key.energy_tolerance ||
      options.electronic_temperature != key.electronic_temperature ||
      periodic_enabled != key.periodic_enabled) {
    return TopologyMatch::kMismatch;
  }

  if (!buffer_equals(batch.atom_offsets, key.atom_offsets) ||
      !buffer_equals(batch.atomic_numbers, key.atomic_numbers) ||
      !double_buffer_equals(batch.molecular_charges, key.molecular_charges) ||
      !buffer_equals(batch.unpaired_electrons, key.unpaired_electrons)) {
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
        !finite_double_buffer(batch.charge_response_matrix, expected_response_elements)) {
      error = "fixed-topology reuse received incomplete or malformed dense charge-response data";
      return TopologyMatch::kInvalid;
    }
  }

  if (!finite_double_buffer(batch.positions, expected_atoms * 3) ||
      !finite_double_buffer(batch.point_charge_positions, expected_points * 3, false, true) ||
      !finite_double_buffer(batch.point_charge_values, expected_points, false, true) ||
      !finite_double_buffer(batch.point_charge_gammas, expected_points, true, true) ||
      (batch.atomic_potential_shifts.data != nullptr &&
       !finite_double_buffer(batch.atomic_potential_shifts, expected_atoms))) {
    error = "fixed-topology reuse received a malformed or nonfinite numerical host buffer";
    return TopologyMatch::kInvalid;
  }
  error.clear();
  return TopologyMatch::kMatch;
}

gpuxtb_status_t validate_offsets(const char* name, const std::vector<std::int64_t>& offsets,
                                 std::int64_t batch_size, std::int64_t total, bool require_nonempty,
                                 std::string& error) {
  if (offsets.size() != static_cast<std::size_t>(batch_size + 1) || offsets.front() != 0 ||
      offsets.back() != total) {
    error = std::string(name) + " does not delimit the declared ragged batch";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t system = 0; system < batch_size; ++system) {
    const std::int64_t begin = offsets[static_cast<std::size_t>(system)];
    const std::int64_t end = offsets[static_cast<std::size_t>(system + 1)];
    if (end < begin || (require_nonempty && end == begin)) {
      error = std::string(name) + " is not monotone or contains an empty molecule";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t make_topology_key(const gpuxtb_batch_t& batch,
                                  const gpuxtb_compute_options_t& options, TopologyKey& key,
                                  std::vector<double>& positions,
                                  std::vector<double>& point_positions,
                                  std::vector<double>& point_values,
                                  std::vector<double>& point_gammas,
                                  std::vector<double>& periodic_shifts,
                                  std::vector<double>& periodic_response, std::string& error) {
  if (batch.batch_size <= 0 || batch.total_atoms <= 0 || batch.total_point_charges < 0 ||
      batch.total_charge_response_elements < 0 || options.model != GPUXTB_MODEL_GFN2_XTB ||
      options.max_scc_iterations <= 0 || !std::isfinite(options.charge_tolerance) ||
      options.charge_tolerance <= 0.0 || !std::isfinite(options.energy_tolerance) ||
      options.energy_tolerance <= 0.0 || !std::isfinite(options.electronic_temperature) ||
      options.electronic_temperature < 0.0) {
    error = "invalid restricted GFN2 CUDA setup dimensions or compute policy";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  gpuxtb_status_t status = copy_host_buffer("atom_offsets", batch.atom_offsets,
                                            batch.batch_size + 1, key.atom_offsets, error);
  if (status != GPUXTB_STATUS_SUCCESS) return status;
  status = validate_offsets("atom_offsets", key.atom_offsets, batch.batch_size, batch.total_atoms,
                            true, error);
  if (status != GPUXTB_STATUS_SUCCESS) return status;
  status = copy_host_buffer("atomic_numbers", batch.atomic_numbers, batch.total_atoms,
                            key.atomic_numbers, error);
  if (status != GPUXTB_STATUS_SUCCESS) return status;
  status = copy_host_buffer("positions", batch.positions, batch.total_atoms * 3, positions, error);
  if (status != GPUXTB_STATUS_SUCCESS) return status;
  status = copy_host_buffer("molecular_charges", batch.molecular_charges, batch.batch_size,
                            key.molecular_charges, error);
  if (status != GPUXTB_STATUS_SUCCESS) return status;
  status = copy_host_buffer("unpaired_electrons", batch.unpaired_electrons, batch.batch_size,
                            key.unpaired_electrons, error);
  if (status != GPUXTB_STATUS_SUCCESS) return status;

  for (std::int64_t atom = 0; atom < batch.total_atoms; ++atom) {
    const std::int32_t atomic_number = key.atomic_numbers[static_cast<std::size_t>(atom)];
    if (atomic_number <= 0 || atomic_number > 118) {
      error = "atomic_numbers contains an unsupported element";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::int64_t system = 0; system < batch.batch_size; ++system) {
    if (!std::isfinite(key.molecular_charges[static_cast<std::size_t>(system)])) {
      error = "molecular_charges contains a nonfinite value";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    if (key.unpaired_electrons[static_cast<std::size_t>(system)] != 0) {
      error = "restricted CUDA GFN2 currently requires zero unpaired electrons";
      return GPUXTB_STATUS_NOT_SUPPORTED;
    }
  }
  for (const double coordinate : positions) {
    if (!std::isfinite(coordinate)) {
      error = "positions contains a nonfinite value";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  if (batch.total_point_charges != 0 || batch.point_charge_offsets.data != nullptr) {
    status = copy_host_buffer("point_charge_offsets", batch.point_charge_offsets,
                              batch.batch_size + 1, key.point_offsets, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = validate_offsets("point_charge_offsets", key.point_offsets, batch.batch_size,
                              batch.total_point_charges, false, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
  } else {
    key.point_offsets.assign(static_cast<std::size_t>(batch.batch_size + 1), 0);
  }
  status = copy_host_buffer("point_charge_positions", batch.point_charge_positions,
                            batch.total_point_charges * 3, point_positions, error, true);
  if (status != GPUXTB_STATUS_SUCCESS) return status;
  status = copy_host_buffer("point_charge_values", batch.point_charge_values,
                            batch.total_point_charges, point_values, error, true);
  if (status != GPUXTB_STATUS_SUCCESS) return status;
  status = copy_host_buffer("point_charge_gammas", batch.point_charge_gammas,
                            batch.total_point_charges, point_gammas, error, true);
  if (status != GPUXTB_STATUS_SUCCESS) return status;
  for (const double value : point_positions) {
    if (!std::isfinite(value)) {
      error = "point_charge_positions contains a nonfinite value";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::size_t point = 0; point < point_values.size(); ++point) {
    if (!std::isfinite(point_values[point]) || !std::isfinite(point_gammas[point]) ||
        point_gammas[point] <= 0.0) {
      error = "point charge values must be finite and gammas must be finite and positive";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  key.periodic_enabled =
      batch.atomic_potential_shifts.data != nullptr || batch.total_charge_response_elements != 0 ||
      batch.charge_response_offsets.data != nullptr || batch.charge_response_matrix.data != nullptr;
  if (batch.atomic_potential_shifts.data != nullptr) {
    status = copy_host_buffer("atomic_potential_shifts", batch.atomic_potential_shifts,
                              batch.total_atoms, periodic_shifts, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
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
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    const std::int64_t elements = atoms * atoms;
    const std::int64_t previous = expected_response_offsets[static_cast<std::size_t>(system)];
    if (elements > std::numeric_limits<std::int64_t>::max() - previous) {
      error = "periodic response prefix sum overflows int64_t";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    expected_response_offsets[static_cast<std::size_t>(system + 1)] = previous + elements;
  }
  if (batch.total_charge_response_elements != 0 || batch.charge_response_offsets.data != nullptr ||
      batch.charge_response_matrix.data != nullptr) {
    status = copy_host_buffer("charge_response_offsets", batch.charge_response_offsets,
                              batch.batch_size + 1, key.response_offsets, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    if (key.response_offsets != expected_response_offsets ||
        batch.total_charge_response_elements != expected_response_offsets.back()) {
      error = "charge_response_offsets does not match the dense per-system atom layout";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    status = copy_host_buffer("charge_response_matrix", batch.charge_response_matrix,
                              batch.total_charge_response_elements, periodic_response, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
  } else {
    key.response_offsets = expected_response_offsets;
    periodic_response.assign(static_cast<std::size_t>(expected_response_offsets.back()), 0.0);
  }
  for (const double value : periodic_shifts) {
    if (!std::isfinite(value)) {
      error = "atomic_potential_shifts contains a nonfinite value";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  for (const double value : periodic_response) {
    if (!std::isfinite(value)) {
      error = "charge_response_matrix contains a nonfinite value";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  key.flags = options.flags;
  key.maximum_iterations = options.max_scc_iterations;
  key.charge_tolerance = options.charge_tolerance;
  key.energy_tolerance = options.energy_tolerance;
  key.electronic_temperature = options.electronic_temperature;
  return GPUXTB_STATUS_SUCCESS;
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

  PinnedArena d4_workspace_storage;
  D4Workspace d4_workspace{};
  std::vector<double> d4_pair_data;
  std::vector<double> d4_coordination;
  D4GeometryCache d4_cache{};
  std::vector<Gfn2D4DeviceElementData> d4_elements;
  std::vector<Gfn2D4DeviceReferenceData> d4_references;

  std::vector<double> explicit_point_shell_potential;
  PinnedArena wavefunction_storage;
  WavefunctionView wavefunction{};

  gpuxtb_status_t build(TopologyKey&& new_key, std::vector<double>&& new_positions,
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
    std::vector<std::int32_t> spin_channels(static_cast<std::size_t>(batch), 1);

    gpuxtb_status_t status = make_basis_plan(batch, atoms, key.atom_offsets.data(),
                                             key.atomic_numbers.data(), basis, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = make_integral_plan(basis, integrals, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = make_coordination_plan(batch, atoms, key.atom_offsets.data(),
                                    key.atomic_numbers.data(), coordination, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = make_repulsion_plan(batch, atoms, key.atom_offsets.data(), key.atomic_numbers.data(),
                                 repulsion, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = make_h0_plan(basis, integrals, key.atomic_numbers.data(), h0, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = make_wavefunction_layout(basis, key.atomic_numbers.data(),
                                      key.molecular_charges.data(), key.unpaired_electrons.data(),
                                      spin_channels.data(), wavefunction_layout, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = make_es2_plan(basis, key.atomic_numbers.data(), es2, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = make_es3_plan(basis, key.atomic_numbers.data(), es3, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = make_aes2_plan(basis, key.atomic_numbers.data(), aes2, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = make_mulliken_plan(basis, integrals, wavefunction_layout, mulliken, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = make_eigensolver_plan(wavefunction_layout, eigensolver, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = make_scc_mixer_plan(wavefunction_layout, kMixerHistory, kMixerDamping,
                                 key.charge_tolerance, key.charge_tolerance, mixer, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;

    d4_enabled = false;
    for (std::int64_t system = 0; system < batch; ++system) {
      d4_enabled = d4_enabled || key.atom_offsets[static_cast<std::size_t>(system + 1)] -
                                         key.atom_offsets[static_cast<std::size_t>(system)] >
                                     1;
    }
    if (d4_enabled) {
      status =
          make_d4_plan(batch, atoms, key.atom_offsets.data(), key.atomic_numbers.data(), d4, error);
      if (status != GPUXTB_STATUS_SUCCESS) return status;
    }
    status = make_external_point_charge_plan(basis, key.atomic_numbers.data(), points,
                                             points == 0 ? nullptr : key.point_offsets.data(),
                                             external, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    periodic_enabled = key.periodic_enabled;
    if (periodic_enabled) {
      status = make_periodic_embedding_plan(batch, atoms, key.atom_offsets.data(), periodic, error);
      if (status != GPUXTB_STATUS_SUCCESS) return status;
    }
    status =
        make_scc_driver_plan(wavefunction_layout, mulliken, es2, es3, aes2, eigensolver, mixer,
                             d4_enabled ? &d4 : nullptr, periodic_enabled ? &periodic : nullptr,
                             static_cast<std::uint64_t>(key.maximum_iterations),
                             key.electronic_temperature, key.energy_tolerance, driver, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;

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
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = evaluate_overlap_cpu(basis, integrals, positions.data(), overlap.data(),
                                  integral_workspace.data(),
                                  integral_workspace.size() * sizeof(double), error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = evaluate_multipole_cpu(basis, integrals, positions.data(), dipole_integrals.data(),
                                    quadrupole_integrals.data(), integral_workspace.data(),
                                    integral_workspace.size() * sizeof(double), error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = evaluate_h0_cpu(basis, integrals, h0, positions.data(), coordination_numbers.data(),
                             overlap.data(), core_hamiltonian.data(), error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;

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
    if (status != GPUXTB_STATUS_SUCCESS) return status;

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
    if (status != GPUXTB_STATUS_SUCCESS) return status;

    if (d4_enabled) {
      if (d4_workspace_storage.allocate(d4.workspace_size_bytes()) != cudaSuccess) {
        error = "failed to allocate pinned host D4 setup workspace";
        return GPUXTB_STATUS_ALLOCATION_FAILED;
      }
      status = bind_d4_workspace(d4, d4_workspace_storage.get(), d4_workspace_storage.bytes(),
                                 d4_workspace, error);
      if (status != GPUXTB_STATUS_SUCCESS) return status;
      d4_pair_data.resize(static_cast<std::size_t>(d4.total_pairs()) * kD4PairDataElements);
      d4_coordination.resize(atom_count);
      status = update_d4_geometry_cache_cpu(
          d4, positions.data(), geometry_generation, d4_pair_data.data(), d4_pair_data.size(),
          d4_coordination.data(), d4_coordination.size(), d4_workspace, d4_cache, error);
      if (status != GPUXTB_STATUS_SUCCESS) return status;

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
    if (status != GPUXTB_STATUS_SUCCESS) return status;

    if (wavefunction_storage.allocate(wavefunction_layout.workspace_size_bytes) != cudaSuccess) {
      error = "failed to allocate pinned host wavefunction initialization storage";
      return GPUXTB_STATUS_ALLOCATION_FAILED;
    }
    status = bind_wavefunction_view(wavefunction_layout, wavefunction_storage.get(),
                                    wavefunction_storage.bytes(), wavefunction, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = initialize_sad_multipole_state(wavefunction_layout, wavefunction, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    return GPUXTB_STATUS_SUCCESS;
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
      sources.d4.pair_data = setup_array(d4_pair_data);
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

std::string setup_error_message(const char* operation, gpuxtb_status_t status,
                                std::uint32_t error_code, std::uint32_t field, std::int64_t index) {
  std::ostringstream message;
  message << operation << " failed: status=" << status << " error=" << error_code
          << " field=" << field << " index=" << index;
  return message.str();
}

struct NumericalRefreshState {
  Gfn2PreprocessingDeviceBinding preprocessing{};
  NumericalRefreshDeviceBinding device{};
  Gfn2D4DeviceCache d4_candidate{};
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

  bool ready = false;
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
  };

  struct Prepared {
    explicit Prepared(cudaStream_t owner_stream) noexcept : stream(owner_stream) {}
    ~Prepared() {
      /* Setup owners retain pinned upload images. A failed candidate and a
       * replaced cache both wait for only their own stream before releasing
       * those images or any arena referenced by queued setup work. */
      if (submitted) {
        /* cudaStreamSynchronize(nullptr) waits for the selected legacy default
         * stream only; setup teardown never escalates to a device-wide fence. */
        (void)cudaStreamSynchronize(stream);
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
    DeviceArena force_immutable_arena;
    DeviceArena force_execution_arena;
    DeviceArena numerical_refresh_arena;
    Gfn2RaggedTopologyView device_topology{};
    Gfn2SccIterationDevicePlan plan_seed{};
    Gfn2SccIterationDeviceInput input_seed{};
    Gfn2SccIterationArenaRequirements iteration_requirements{};
    Gfn2SccIterationDeviceState state_seed{};
    Gfn2SccIterationDeviceWorkspace workspace_seed{};
    Gfn2SccIterationReportStorage report_storage{};
    Gfn2SccSetupEigensolverBinding eigensolver_binding{};
    Gfn2SccIterationInitializationReady ready{};
    Gfn2SccIterationBinding scc_binding{};
    EnergyForceBindings energy_force{};
    NumericalRefreshState numerical{};
    bool energy_force_smoke_ready = false;
  };

  Impl(std::int32_t selected_device, void* selected_stream) noexcept
      : device_id(selected_device), stream(reinterpret_cast<cudaStream_t>(selected_stream)) {}

  ~Impl() {
    std::lock_guard<std::mutex> lock(mutex);
    /* CUDA current-device state is thread-local. Context teardown can run on a
     * different thread or after the caller selected another device, so restore
     * the owner device before releasing device allocations and handles. */
    (void)cudaSetDevice(device_id);
    if (prepared != nullptr) {
      prepared.reset();
    }
    if (handles_created) (void)cudaStreamSynchronize(stream);
    if (blas != nullptr) (void)cublasDestroy(blas);
    if (solver_parameters != nullptr) (void)cusolverDnDestroyParams(solver_parameters);
    if (solver != nullptr) (void)cusolverDnDestroy(solver);
  }

  gpuxtb_status_t ensure_handles(std::string& error) {
    /* cudaSetDevice is intentionally repeated on every prepare. CUDA current
     * device selection is thread-local and is not preserved by this context. */
    cudaError_t cuda_status = cudaSetDevice(device_id);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("cudaSetDevice", cuda_status);
      return GPUXTB_STATUS_BACKEND_UNAVAILABLE;
    }
    if (!ensure_cuda_gfn2_parameters(device_id, error)) {
      return GPUXTB_STATUS_BACKEND_UNAVAILABLE;
    }
    if (handles_created) return GPUXTB_STATUS_SUCCESS;

    /* Publish context handles only after the entire construction succeeds.
     * This keeps retries leak-free after a partial provider failure. */
    cusolverDnHandle_t candidate_solver = nullptr;
    cusolverDnParams_t candidate_parameters = nullptr;
    cublasHandle_t candidate_blas = nullptr;
    const auto destroy_candidate = [&]() noexcept {
      if (candidate_blas != nullptr) (void)cublasDestroy(candidate_blas);
      if (candidate_parameters != nullptr) (void)cusolverDnDestroyParams(candidate_parameters);
      if (candidate_solver != nullptr) (void)cusolverDnDestroy(candidate_solver);
    };

    cusolverStatus_t solver_status = cusolverDnCreate(&candidate_solver);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
      error = "cusolverDnCreate failed";
      return GPUXTB_STATUS_BACKEND_UNAVAILABLE;
    }
    solver_status = cusolverDnCreateParams(&candidate_parameters);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
      destroy_candidate();
      error = "cusolverDnCreateParams failed";
      return GPUXTB_STATUS_BACKEND_UNAVAILABLE;
    }
    cublasStatus_t blas_status = cublasCreate(&candidate_blas);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
      destroy_candidate();
      error = "cublasCreate failed";
      return GPUXTB_STATUS_BACKEND_UNAVAILABLE;
    }
    solver_status = cusolverDnSetStream(candidate_solver, stream);
    blas_status = cublasSetStream(candidate_blas, stream);
    if (solver_status != CUSOLVER_STATUS_SUCCESS || blas_status != CUBLAS_STATUS_SUCCESS) {
      destroy_candidate();
      error = "failed to bind CUDA linear-algebra handles to the context stream";
      return GPUXTB_STATUS_BACKEND_UNAVAILABLE;
    }
    solver = candidate_solver;
    solver_parameters = candidate_parameters;
    blas = candidate_blas;
    handles_created = true;
    return GPUXTB_STATUS_SUCCESS;
  }

  gpuxtb_status_t build_numerical_refresh_binding(Prepared& candidate, std::string& error) {
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
    const std::int64_t d4_pairs = candidate.plan_seed.d4_batch.total_pairs;
    const std::uint64_t token = candidate.host.plan_token;
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    std::int64_t geometry_pair_elements = 0;
    std::int64_t aes2_elements = 0;
    std::int64_t d4_pair_elements = 0;
    std::int64_t dipole_elements = 0;
    std::int64_t quadrupole_elements = 0;
    if (!checked_elements(atoms, 3, coordinates) ||
        !checked_elements(points, 3, point_coordinates) ||
        !checked_elements(geometry_pairs, kGfn2GeometryPairDataElements, geometry_pair_elements) ||
        !checked_elements(aes2_pairs, kGfn2AES2PairDataElements, aes2_elements) ||
        !checked_elements(d4_pairs, kGfn2D4PairDataElements, d4_pair_elements) ||
        !checked_elements(matrices, 3, dipole_elements) ||
        !checked_elements(matrices, 6, quadrupole_elements)) {
      error = "numerical refresh element count overflows int64_t";
      return GPUXTB_STATUS_ALLOCATION_FAILED;
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
      std::size_t committed_generations = 0u;
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

      std::size_t d4_candidate_pairs = 0u;
      std::size_t d4_candidate_coordination = 0u;
      std::size_t d4_pair_scratch = 0u;
      std::size_t d4_coordination_scratch = 0u;
      std::size_t d4_generations = 0u;
      std::size_t d4_sequence = 0u;
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
    offset.committed_generations = layout.append<std::uint64_t>(batch);
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

    if (candidate.host.d4_enabled) {
      offset.d4_candidate_pairs = layout.append<double>(d4_pair_elements);
      offset.d4_candidate_coordination = layout.append<double>(atoms);
      offset.d4_pair_scratch = layout.append<double>(d4_pair_elements);
      offset.d4_coordination_scratch = layout.append<double>(atoms);
      offset.d4_generations = layout.append<std::uint64_t>(batch);
      offset.d4_sequence = layout.append<std::uint32_t>(1);
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
      return GPUXTB_STATUS_ALLOCATION_FAILED;
    }
    cudaError_t cuda_status = candidate.numerical_refresh_arena.allocate(layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical refresh arena allocation", cuda_status);
      return GPUXTB_STATUS_ALLOCATION_FAILED;
    }
    void* const arena = candidate.numerical_refresh_arena.get();
    cuda_status = cudaMemsetAsync(arena, 0, candidate.numerical_refresh_arena.bytes(), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical refresh arena initialization", cuda_status);
      return GPUXTB_STATUS_INTERNAL_ERROR;
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
    upload_vector(offset.committed_positions, candidate.host.positions);
    upload_vector(offset.committed_point_positions, candidate.host.point_positions);
    upload_vector(offset.committed_point_values, candidate.host.point_values);
    upload_vector(offset.committed_point_gammas, candidate.host.point_gammas);
    upload_vector(offset.committed_periodic_shifts, candidate.host.periodic_shifts);
    upload_vector(offset.committed_periodic_response, candidate.host.periodic_response);
    std::vector<std::uint64_t> initial_generations(static_cast<std::size_t>(batch),
                                                   candidate.host.geometry_generation);
    upload_vector(offset.committed_generations, initial_generations);
    upload(offset.geometry_epoch, &candidate.host.geometry_generation,
           sizeof(candidate.host.geometry_generation));
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemsetAsync(arena_pointer<std::uint8_t>(arena, offset.requested), 1,
                                    static_cast<std::size_t>(batch), stream);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical refresh setup upload", cuda_status);
      return GPUXTB_STATUS_INTERNAL_ERROR;
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
    binding.diagnostics = {arena_pointer<std::uint32_t>(arena, offset.geometry_system_errors),
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
      return GPUXTB_STATUS_INVALID_ARGUMENT;
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
    device.d4_pair_elements = d4_pair_elements;
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
    device.d4_pair_offsets = candidate.plan_seed.d4_batch.pair_offsets;
    device.requested = binding.activity.requested_mask;
    device.preprocessing_published = binding.activity.published_mask;
    device.eligible = candidate.workspace_seed.ledger.active_mask;
    device.committed_generations =
        arena_pointer<std::uint64_t>(arena, offset.committed_generations);
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
      numerical.d4_candidate = {
          arena_pointer_if<double>(arena, offset.d4_candidate_pairs, d4_pair_elements),
          d4_pair_elements,
          arena_pointer<double>(arena, offset.d4_candidate_coordination),
          atoms,
          candidate.host.geometry_generation,
          token};
      numerical.d4_workspace.pair_scratch =
          arena_pointer_if<double>(arena, offset.d4_pair_scratch, d4_pair_elements);
      numerical.d4_workspace.pair_scratch_elements = d4_pair_elements;
      numerical.d4_workspace.coordination_scratch =
          arena_pointer<double>(arena, offset.d4_coordination_scratch);
      numerical.d4_workspace.coordination_scratch_elements = atoms;
      numerical.d4_workspace.geometry_generations =
          arena_pointer<std::uint64_t>(arena, offset.d4_generations);
      numerical.d4_workspace.geometry_generation_elements = batch;
      numerical.d4_workspace.geometry_sequence_active =
          arena_pointer<std::uint32_t>(arena, offset.d4_sequence);
      numerical.d4_workspace.geometry_sequence_elements = 1;
      numerical.d4_workspace.system_errors =
          arena_pointer<std::uint32_t>(arena, offset.d4_system_errors);
      numerical.d4_workspace.system_error_elements = batch;
      device.candidate_d4_pairs = numerical.d4_candidate.pair_data;
      device.candidate_d4_coordination = numerical.d4_candidate.coordination_numbers;
      device.public_d4_pairs = const_cast<double*>(candidate.plan_seed.d4_cache.pair_data);
      device.public_d4_coordination =
          const_cast<double*>(candidate.plan_seed.d4_cache.coordination_numbers);
      device.d4_system_errors = numerical.d4_workspace.system_errors;
      device.d4_device_error = arena_pointer<std::uint32_t>(arena, offset.d4_device_error);
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
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }
    cuda_status = cudaMemcpyAsync(
        const_cast<Gfn2SccCacheProvenanceBinding*>(candidate.plan_seed.provenance.cache_bindings),
        provenance.data(), provenance.size() * sizeof(provenance.front()), cudaMemcpyHostToDevice,
        stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA runtime provenance publication", cuda_status);
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }
    numerical.ready = true;
    return GPUXTB_STATUS_SUCCESS;
  }

  gpuxtb_status_t build_energy_force_bindings(Prepared& candidate, std::string& error) {
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
      return GPUXTB_STATUS_ALLOCATION_FAILED;
    }
    const bool force_mode =
        (candidate.host.key.flags &
         (static_cast<std::uint32_t>(GPUXTB_COMPUTE_FORCES) |
          static_cast<std::uint32_t>(GPUXTB_COMPUTE_POINT_CHARGE_FORCES))) != 0u;
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
      return GPUXTB_STATUS_ALLOCATION_FAILED;
    }
    cudaError_t cuda_status = candidate.force_immutable_arena.allocate(immutable_layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("force immutable arena allocation", cuda_status);
      return GPUXTB_STATUS_ALLOCATION_FAILED;
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
      return GPUXTB_STATUS_INTERNAL_ERROR;
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
      return GPUXTB_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = candidate.force_execution_arena.allocate(execution_layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("force execution arena allocation", cuda_status);
      return GPUXTB_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = cudaMemsetAsync(candidate.force_execution_arena.get(), 0,
                                  candidate.force_execution_arena.bytes(), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("force execution arena initialization", cuda_status);
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }

    void* const execution_arena = candidate.force_execution_arena.get();
    void* const immutable_arena = candidate.force_immutable_arena.get();
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
        return GPUXTB_STATUS_INTERNAL_ERROR;
      }
      candidate.submitted = true;

      const auto* const atomic_numbers =
          arena_pointer<std::int32_t>(immutable_arena, immutable.atomic_numbers);
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
      post_plan.d4_cache = candidate.plan_seed.d4_cache;
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
                                     candidate.plan_seed.d4_cache};
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

      const auto& raw = candidate.state_seed.raw_population;
      binding.input.force_activity = {mask(kRequestedMask),
                                      candidate.state_seed.scc.system_statuses, batch, token};
      binding.input.post_scc_potential = {
          {mask(kRequestedMask), candidate.state_seed.scc.system_statuses, batch, token},
          raw.qsh,
          raw.qsh_elements,
          raw.qat,
          raw.qat_elements,
          raw.dipole,
          raw.dipole_elements,
          raw.quadrupole,
          raw.quadrupole_elements,
          token};
      binding.input.h0 = {positions,
                          coordinates,
                          candidate.plan_seed.geometry_cache.coordination_numbers,
                          atoms,
                          candidate.input_seed.hamiltonian.overlap,
                          matrices,
                          candidate.state_seed.density.density,
                          matrices,
                          candidate.state_seed.density.energy_weighted_density,
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
      binding.input.hamiltonian = {candidate.state_seed.density.density,
                                   matrices,
                                   post_shell_scalar,
                                   shells,
                                   post_complete_dipole,
                                   coordinates,
                                   post_complete_quadrupole,
                                   quadrupole_elements,
                                   token};
      binding.input.classical = {positions,
                                 coordinates,
                                 candidate.plan_seed.geometry_cache.coordination_numbers,
                                 atoms,
                                 raw.qsh,
                                 raw.qsh_elements,
                                 raw.qat,
                                 raw.qat_elements,
                                 raw.dipole,
                                 raw.dipole_elements,
                                 raw.quadrupole,
                                 raw.quadrupole_elements,
                                 token};
      binding.input.external_shell_charges = raw.qsh;
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
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      cuda_status = update_gfn2_geometry_cache_cuda(
          binding.plan.coordination_batch, positions, candidate.host.geometry_generation,
          binding.plan.coordination_cache, candidate.workspace_seed.geometry_workspace,
          binding.diagnostics.coordination_system_errors,
          binding.diagnostics.coordination_device_error, stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA initial geometry-cache construction", cuda_status);
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      cuda_status =
          reset_gfn2_aes2_device_errors_cuda(batch, binding.diagnostics.coordination_system_errors,
                                             binding.diagnostics.coordination_device_error, stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA initial AES2 diagnostic reset", cuda_status);
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      cuda_status = update_gfn2_aes2_geometry_cache_cuda(
          candidate.plan_seed.aes2_batch, positions,
          binding.plan.coordination_cache.coordination_numbers, candidate.plan_seed.aes2_cache,
          candidate.workspace_seed.aes2_workspace, binding.diagnostics.coordination_system_errors,
          binding.diagnostics.coordination_device_error, stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA initial AES2-cache construction", cuda_status);
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
    }

    /* Exercise the published descriptors, not merely their host validation.
     * All-bits-one is a NaN for the CUDA-supported IEEE-754 double format, so
     * a finite download proves that terminal publication actually ran. */
    std::size_t output_bytes = 0u;
    if (!checked_bytes(batch, sizeof(double), output_bytes)) {
      error = "energy smoke output extent overflows size_t";
      return GPUXTB_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = cudaMemsetAsync(binding.results.energy.total_energy, 0xff, output_bytes, stream);
    if (cuda_status == cudaSuccess && force_mode) {
      if (!checked_bytes(binding.results.forces.qm_force_elements, sizeof(double), output_bytes)) {
        error = "QM-force smoke output extent overflows size_t";
        return GPUXTB_STATUS_ALLOCATION_FAILED;
      }
      cuda_status = cudaMemsetAsync(binding.results.forces.qm_forces, 0xff, output_bytes, stream);
      if (cuda_status == cudaSuccess && binding.results.forces.point_force_elements != 0) {
        if (!checked_bytes(binding.results.forces.point_force_elements, sizeof(double),
                           output_bytes)) {
          error = "point-force smoke output extent overflows size_t";
          return GPUXTB_STATUS_ALLOCATION_FAILED;
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
                          static_cast<std::size_t>(batch) * sizeof(gpuxtb_status_t), stream);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemsetAsync(candidate.state_seed.scc.converged, 1,
                                    static_cast<std::size_t>(batch) * sizeof(std::uint8_t), stream);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA energy/force smoke initialization", cuda_status);
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }
    cuda_status = execute_gfn2_energy_force_cuda(binding.plan, binding.input, binding.results,
                                                 binding.intermediates, binding.workspace,
                                                 binding.diagnostics, stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA energy/force composed smoke", cuda_status);
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }

    /* The smoke borrows the initialized SCC storage only long enough to drive
     * the terminal chain. Re-upload the immutable fresh image so the published
     * runtime still starts at iteration zero. */
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
    return GPUXTB_STATUS_SUCCESS;
  }

  gpuxtb_status_t validate_energy_force_smoke(Prepared& candidate, std::string& error) {
    const auto& binding = candidate.energy_force;
    const std::int64_t batch = candidate.host.basis.batch_size;
    std::vector<std::uint32_t> system_errors(static_cast<std::size_t>(batch));
    std::uint32_t device_error = 0u;
    std::uint32_t plan_failure = 0u;
    std::vector<double> energies(static_cast<std::size_t>(batch));
    std::vector<double> qm_forces;
    std::vector<double> point_forces;
    if (binding.plan.compute_forces == 1u) {
      qm_forces.resize(static_cast<std::size_t>(binding.results.forces.qm_force_elements));
      point_forces.resize(static_cast<std::size_t>(binding.results.forces.point_force_elements));
    }

    cudaError_t cuda_status =
        cudaMemcpy(system_errors.data(), binding.diagnostics.execution_system_errors,
                   system_errors.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemcpy(&device_error, binding.diagnostics.execution_device_error,
                               sizeof(device_error), cudaMemcpyDeviceToHost);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemcpy(&plan_failure, binding.workspace.plan_failure, sizeof(plan_failure),
                               cudaMemcpyDeviceToHost);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemcpy(energies.data(), binding.results.energy.total_energy,
                               energies.size() * sizeof(double), cudaMemcpyDeviceToHost);
    }
    if (cuda_status == cudaSuccess && !qm_forces.empty()) {
      cuda_status = cudaMemcpy(qm_forces.data(), binding.results.forces.qm_forces,
                               qm_forces.size() * sizeof(double), cudaMemcpyDeviceToHost);
    }
    if (cuda_status == cudaSuccess && !point_forces.empty()) {
      cuda_status = cudaMemcpy(point_forces.data(), binding.results.forces.point_forces,
                               point_forces.size() * sizeof(double), cudaMemcpyDeviceToHost);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA energy/force smoke download", cuda_status);
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }

    const auto first_system_error = std::find_if(system_errors.begin(), system_errors.end(),
                                                 [](std::uint32_t value) { return value != 0u; });
    if (device_error != 0u || plan_failure != 0u || first_system_error != system_errors.end()) {
      std::ostringstream message;
      message << "CUDA energy/force smoke reported device_error=" << device_error
              << " plan_failure=" << plan_failure;
      if (first_system_error != system_errors.end()) {
        message << " system=" << std::distance(system_errors.begin(), first_system_error)
                << " system_error=" << *first_system_error;
      }
      error = message.str();
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }
    const auto all_finite = [](const std::vector<double>& values) {
      return std::all_of(values.begin(), values.end(),
                         [](double value) { return std::isfinite(value); });
    };
    if (!all_finite(energies) || !all_finite(qm_forces) || !all_finite(point_forces)) {
      error = "CUDA energy/force smoke did not publish every requested output";
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }

    /* upload_async above restores the complete fresh checkpoint. Verify the
     * public convergence byte as the minimal externally meaningful witness. */
    std::vector<std::uint8_t> converged(static_cast<std::size_t>(batch), 1u);
    cuda_status = cudaMemcpy(converged.data(), candidate.state_seed.scc.converged,
                             converged.size() * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA restored SCC-state download", cuda_status);
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }
    if (std::any_of(converged.begin(), converged.end(),
                    [](std::uint8_t value) { return value != 0u; })) {
      error = "CUDA energy/force smoke failed to restore the fresh SCC state";
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }

    candidate.energy_force_smoke_ready = true;
    return GPUXTB_STATUS_SUCCESS;
  }

  gpuxtb_status_t build_candidate(TopologyKey&& key, std::vector<double>&& positions,
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
    gpuxtb_status_t status = candidate->host.build(
        std::move(key), std::move(positions), std::move(point_positions), std::move(point_values),
        std::move(point_gammas), std::move(periodic_shifts), std::move(periodic_response), token,
        error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;

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
      return GPUXTB_STATUS_ALLOCATION_FAILED;
    }
    topology_diagnostic = candidate->topology_owner.bind_device_arena_and_upload_async(
        candidate->topology_arena.get(), candidate->topology_arena.bytes(),
        candidate->device_topology, stream);
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
      return GPUXTB_STATUS_ALLOCATION_FAILED;
    }
    input_diagnostic = candidate->inputs_owner.bind_device_arena_and_upload_async(
        candidate->device_topology, candidate->input_arena.get(), candidate->input_arena.bytes(),
        candidate->plan_seed, candidate->input_seed, stream);
    if (!input_diagnostic.success()) {
      error = setup_error_message("CUDA input upload", input_diagnostic.status,
                                  static_cast<std::uint32_t>(input_diagnostic.error),
                                  static_cast<std::uint32_t>(input_diagnostic.field),
                                  input_diagnostic.index);
      return input_diagnostic.status;
    }

    auto eigensolver_diagnostic = Gfn2SccSetupEigensolver::create(
        candidate->topology_owner, candidate->host.overlap.data(),
        static_cast<std::int64_t>(candidate->host.overlap.size()),
        candidate->host.geometry_generation, token, solver, solver_parameters, blas,
        candidate->plan_seed.eigensolver_options, candidate->eigensolver_owner);
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
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    cuda_status =
        candidate->iteration_arena.allocate(candidate->iteration_requirements.total_bytes);
    if (cuda_status == cudaSuccess) {
      cuda_status = candidate->provider_host_workspace.allocate(
          eigensolver_requirements.provider.solver_host_workspace_bytes);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA SCC iteration/provider workspace allocation", cuda_status);
      return GPUXTB_STATUS_ALLOCATION_FAILED;
    }
    const auto bind_diagnostic = bind_gfn2_scc_iteration_arena_cuda(
        candidate->plan_seed, eigensolver_requirements.provider, candidate->iteration_requirements,
        candidate->iteration_arena.get(), candidate->iteration_arena.bytes(),
        candidate->provider_host_workspace.get(), candidate->provider_host_workspace.bytes(),
        candidate->state_seed, candidate->workspace_seed, candidate->report_storage);
    if (!bind_diagnostic.success()) {
      error = "CUDA SCC iteration arena binding rejected its own sealed requirements";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    status = build_numerical_refresh_binding(*candidate, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    cuda_status =
        candidate->eigensolver_setup_arena.allocate(eigensolver_requirements.setup_device_bytes);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA eigensolver setup arena allocation", cuda_status);
      return GPUXTB_STATUS_ALLOCATION_FAILED;
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
    candidate->plan_seed.overlap_cache = candidate->eigensolver_binding.cache;
    candidate->plan_seed.eigensolver_options = candidate->eigensolver_binding.options;

    auto initialization_diagnostic = Gfn2SccIterationInitializer::create(
        candidate->plan_seed, candidate->iteration_requirements, candidate->iteration_arena.get(),
        candidate->iteration_arena.bytes(), candidate->state_seed, candidate->workspace_seed,
        candidate->report_storage, candidate->host.initialization(), candidate->initializer);
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
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }

    status = build_energy_force_bindings(*candidate, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;

    cuda_status = cudaStreamSynchronize(stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA runtime candidate setup synchronization", cuda_status);
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }
    std::uint32_t setup_device_error = 0u;
    std::vector<std::uint32_t> setup_system_errors(
        static_cast<std::size_t>(candidate->eigensolver_binding.setup_system_error_elements));
    std::vector<std::uint32_t> factor_statuses(
        static_cast<std::size_t>(candidate->eigensolver_binding.cache.status_elements));
    cuda_status = cudaMemcpy(&setup_device_error, candidate->eigensolver_binding.setup_device_error,
                             sizeof(setup_device_error), cudaMemcpyDeviceToHost);
    if (cuda_status == cudaSuccess && !setup_system_errors.empty()) {
      cuda_status =
          cudaMemcpy(setup_system_errors.data(), candidate->eigensolver_binding.setup_system_errors,
                     setup_system_errors.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    }
    if (cuda_status == cudaSuccess && !factor_statuses.empty()) {
      cuda_status =
          cudaMemcpy(factor_statuses.data(), candidate->eigensolver_binding.cache.factor_statuses,
                     factor_statuses.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA eigensolver setup diagnostics download", cuda_status);
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }
    if (setup_device_error != 0u ||
        std::any_of(setup_system_errors.begin(), setup_system_errors.end(),
                    [](std::uint32_t value) { return value != 0u; }) ||
        std::any_of(factor_statuses.begin(), factor_statuses.end(),
                    [](std::uint32_t value) { return value != 0u; })) {
      error = "CUDA eigensolver setup reported an asynchronous factorization failure";
      return GPUXTB_STATUS_EIGENSOLVER_FAILED;
    }
    status = validate_energy_force_smoke(*candidate, error);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    candidate->submitted = false;
    output = std::move(candidate);
    return GPUXTB_STATUS_SUCCESS;
  }

  gpuxtb_status_t refresh_numerical_locked(const Gfn2CudaNumericalInputView& input,
                                           std::string& error) {
    if (prepared == nullptr || !prepared->numerical.ready) {
      error = "CUDA GFN2 numerical refresh requires a prepared fixed topology";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    Prepared& current = *prepared;
    auto& numerical = current.numerical;
    auto& device = numerical.device;
    auto& preprocessing = numerical.preprocessing;
    cudaError_t cuda_status = cudaSetDevice(device_id);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("cudaSetDevice for numerical refresh", cuda_status);
      return GPUXTB_STATUS_BACKEND_UNAVAILABLE;
    }

    /* Validate the complete view before enqueueing any transfer. A synchronous
     * descriptor rejection must not leave an earlier asynchronous host read in
     * flight after the caller is told that the refresh did not start. */
    const auto validate_source = [&](const char* name, const gpuxtb_const_buffer_t& buffer,
                                     std::int64_t elements,
                                     const double* host_stage) -> gpuxtb_status_t {
      std::size_t bytes = 0u;
      if (!checked_bytes(elements, sizeof(double), bytes)) {
        error = std::string(name) + " extent overflows size_t";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      const bool supported_space = buffer.memory_space == GPUXTB_MEMORY_HOST ||
                                   buffer.memory_space == GPUXTB_MEMORY_CUDA_DEVICE;
      if (elements == 0) {
        if (buffer.reserved != 0u || !supported_space) {
          error = std::string(name) + " has malformed empty-buffer metadata";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        return GPUXTB_STATUS_SUCCESS;
      }
      if (buffer.reserved != 0u || buffer.data == nullptr || buffer.size_bytes < bytes ||
          !supported_space) {
        error = std::string(name) + " is not a sufficiently large host/CUDA buffer";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      if (buffer.memory_space == GPUXTB_MEMORY_HOST && host_stage == nullptr) {
        error = std::string(name) + " has no runtime host-staging projection";
        return GPUXTB_STATUS_INTERNAL_ERROR;
      }
      return GPUXTB_STATUS_SUCCESS;
    };

    const auto resolve_validated = [&](const char* name, const gpuxtb_const_buffer_t& buffer,
                                       std::int64_t elements, double* host_stage,
                                       const double*& source) -> gpuxtb_status_t {
      if (elements == 0) {
        source = nullptr;
        return GPUXTB_STATUS_SUCCESS;
      }
      std::size_t bytes = 0u;
      if (!checked_bytes(elements, sizeof(double), bytes)) {
        error = std::string(name) + " extent changed after validation";
        return GPUXTB_STATUS_INTERNAL_ERROR;
      }
      if (buffer.memory_space == GPUXTB_MEMORY_HOST) {
        cuda_status =
            cudaMemcpyAsync(host_stage, buffer.data, bytes, cudaMemcpyHostToDevice, stream);
        if (cuda_status != cudaSuccess) {
          error = cuda_error_message(name, cuda_status);
          return GPUXTB_STATUS_INTERNAL_ERROR;
        }
        source = host_stage;
      } else {
        source = static_cast<const double*>(buffer.data);
      }
      return GPUXTB_STATUS_SUCCESS;
    };

    gpuxtb_status_t status = validate_source("positions", input.positions, device.total_atoms * 3,
                                             numerical.host_positions);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = validate_source("point_charge_positions", input.point_charge_positions,
                             device.total_point_charges * 3, numerical.host_point_positions);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = validate_source("point_charge_values", input.point_charge_values,
                             device.total_point_charges, numerical.host_point_values);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = validate_source("point_charge_gammas", input.point_charge_gammas,
                             device.total_point_charges, numerical.host_point_gammas);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = validate_source("atomic_potential_shifts", input.atomic_potential_shifts,
                             device.periodic_enabled != 0u ? device.total_atoms : 0,
                             numerical.host_periodic_shifts);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = validate_source("charge_response_matrix", input.charge_response_matrix,
                             device.periodic_enabled != 0u ? device.total_response_elements : 0,
                             numerical.host_periodic_response);
    if (status != GPUXTB_STATUS_SUCCESS) return status;

    std::size_t mask_bytes = 0u;
    if (!checked_bytes(device.batch_size, sizeof(std::uint8_t), mask_bytes)) {
      error = "numerical refresh activity extent overflows size_t";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    auto* const requested = const_cast<std::uint8_t*>(preprocessing.activity.requested_mask);
    const bool absent_mask =
        input.requested_mask.data == nullptr && input.requested_mask.size_bytes == 0u;
    const bool valid_mask_space = input.requested_mask.memory_space == GPUXTB_MEMORY_HOST ||
                                  input.requested_mask.memory_space == GPUXTB_MEMORY_CUDA_DEVICE;
    if ((absent_mask && (input.requested_mask.reserved != 0u || !valid_mask_space)) ||
        (!absent_mask &&
         (input.requested_mask.reserved != 0u || input.requested_mask.data == nullptr ||
          input.requested_mask.size_bytes < mask_bytes || !valid_mask_space))) {
      error = "requested_mask is not a sufficiently large host/CUDA uint8 buffer";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }

    NumericalRefreshDeviceSources sources{};
    status = resolve_validated("positions", input.positions, device.total_atoms * 3,
                               numerical.host_positions, sources.positions);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = resolve_validated("point_charge_positions", input.point_charge_positions,
                               device.total_point_charges * 3, numerical.host_point_positions,
                               sources.point_positions);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = resolve_validated("point_charge_values", input.point_charge_values,
                               device.total_point_charges, numerical.host_point_values,
                               sources.point_values);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = resolve_validated("point_charge_gammas", input.point_charge_gammas,
                               device.total_point_charges, numerical.host_point_gammas,
                               sources.point_gammas);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = resolve_validated("atomic_potential_shifts", input.atomic_potential_shifts,
                               device.periodic_enabled != 0u ? device.total_atoms : 0,
                               numerical.host_periodic_shifts, sources.periodic_shifts);
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = resolve_validated("charge_response_matrix", input.charge_response_matrix,
                               device.periodic_enabled != 0u ? device.total_response_elements : 0,
                               numerical.host_periodic_response, sources.periodic_response);
    if (status != GPUXTB_STATUS_SUCCESS) return status;

    if (absent_mask) {
      cuda_status = cudaMemsetAsync(requested, 1, mask_bytes, stream);
    } else {
      const cudaMemcpyKind kind = input.requested_mask.memory_space == GPUXTB_MEMORY_HOST
                                      ? cudaMemcpyHostToDevice
                                      : cudaMemcpyDeviceToDevice;
      cuda_status = cudaMemcpyAsync(requested, input.requested_mask.data, mask_bytes, kind, stream);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical refresh activity staging", cuda_status);
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }

    stage_gfn2_numerical_inputs_kernel<<<static_cast<unsigned int>(device.batch_size), 256, 0,
                                         stream>>>(device, sources);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical input staging", cuda_status);
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }

    const auto preprocessing_launch = compose_gfn2_preprocessing_epoch_cuda(preprocessing, stream);
    if (!preprocessing_launch.success()) {
      error = "CUDA numerical preprocessing composer rejected the runtime binding";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }

    if (device.d4_enabled != 0u) {
      cuda_status = reset_gfn2_d4_device_errors_cuda(
          device.batch_size, const_cast<std::uint32_t*>(device.d4_system_errors),
          const_cast<std::uint32_t*>(device.d4_device_error), stream);
      if (cuda_status == cudaSuccess) {
        cuda_status = update_gfn2_d4_geometry_cache_cuda(
            current.plan_seed.d4_batch, current.plan_seed.d4_parameters, device.candidate_positions,
            numerical.d4_candidate, numerical.d4_workspace,
            const_cast<std::uint32_t*>(device.d4_device_error), stream);
      }
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA D4 numerical refresh", cuda_status);
        return GPUXTB_STATUS_INTERNAL_ERROR;
      }
    }
    if (device.point_enabled != 0u) {
      cuda_status = reset_gfn2_external_point_charge_scc_errors_cuda(
          device.batch_size, const_cast<std::uint32_t*>(device.point_system_errors),
          const_cast<std::uint32_t*>(device.point_plan_error), stream);
      if (cuda_status == cudaSuccess) {
        initialize_gfn2_refresh_sequence_kernel<<<1, 1, 0, stream>>>(
            const_cast<std::uint32_t*>(numerical.point_activity.sequence_active));
        cuda_status = cudaPeekAtLastError();
      }
      if (cuda_status == cudaSuccess) {
        cuda_status = update_gfn2_external_point_charge_scc_potential_cache_cuda(
            numerical.point_batch, numerical.point_activity, numerical.point_candidate,
            kInitialGeometryGeneration, numerical.point_workspace,
            const_cast<std::uint32_t*>(device.point_system_errors),
            const_cast<std::uint32_t*>(device.point_plan_error), stream);
      }
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA explicit-point-charge numerical refresh", cuda_status);
        return GPUXTB_STATUS_INTERNAL_ERROR;
      }
    }
    if (device.periodic_enabled != 0u) {
      cuda_status = cudaMemsetAsync(
          device.periodic_system_errors, 0,
          static_cast<std::size_t>(device.batch_size) * sizeof(std::uint32_t), stream);
      if (cuda_status == cudaSuccess) {
        cuda_status = cudaMemsetAsync(device.periodic_plan_error, 0, sizeof(std::uint32_t), stream);
      }
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA periodic refresh diagnostic reset", cuda_status);
        return GPUXTB_STATUS_INTERNAL_ERROR;
      }
      validate_gfn2_periodic_refresh_kernel<<<static_cast<unsigned int>(device.batch_size), 256, 0,
                                              stream>>>(device);
      cuda_status = cudaPeekAtLastError();
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA periodic numerical validation", cuda_status);
        return GPUXTB_STATUS_INTERNAL_ERROR;
      }
    }

    gate_gfn2_numerical_refresh_kernel<<<static_cast<unsigned int>(device.batch_size), 1, 0,
                                         stream>>>(device);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical pre-factor publication gate", cuda_status);
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }
    const auto factor = current.eigensolver_owner.refactor_overlap_from_device_epoch_async(
        current.eigensolver_setup_arena.get(), current.eigensolver_setup_arena.bytes(),
        current.eigensolver_binding, preprocessing.output.overlap,
        preprocessing.output.overlap_elements, preprocessing.geometry_epoch, stream);
    if (!factor.success()) {
      error = setup_error_message("CUDA numerical overlap refactor", factor.status,
                                  static_cast<std::uint32_t>(factor.error),
                                  static_cast<std::uint32_t>(factor.field), factor.index);
      return factor.status;
    }
    device.factor_generations = current.eigensolver_binding.cache.geometry_generations;
    device.factor_statuses = current.eigensolver_binding.cache.factor_statuses;
    commit_gfn2_numerical_refresh_kernel<<<static_cast<unsigned int>(device.batch_size), 256, 0,
                                           stream>>>(device);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical final publication", cuda_status);
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }
    current.submitted = true;
    error.clear();
    return GPUXTB_STATUS_SUCCESS;
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
    identity.energy_force_descriptors = opaque_address(&current.energy_force);
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
    identity.topology_arena_bytes = current.topology_arena.bytes();
    identity.input_arena_bytes = current.input_arena.bytes();
    identity.iteration_arena_bytes = current.iteration_arena.bytes();
    identity.eigensolver_setup_arena_bytes = current.eigensolver_setup_arena.bytes();
    identity.provider_host_workspace_bytes = current.provider_host_workspace.bytes();
    identity.force_immutable_arena_bytes = current.force_immutable_arena.bytes();
    identity.force_execution_arena_bytes = current.force_execution_arena.bytes();
    identity.numerical_refresh_arena_bytes = current.numerical_refresh_arena.bytes();
    return identity;
  }

  std::int32_t device_id = -1;
  cudaStream_t stream = nullptr;
  cusolverDnHandle_t solver = nullptr;
  cusolverDnParams_t solver_parameters = nullptr;
  cublasHandle_t blas = nullptr;
  bool handles_created = false;
  std::uint64_t next_plan_token = 1u;
  std::unique_ptr<Prepared> prepared;
  mutable std::mutex mutex;
};

Gfn2CudaExecutionCache::Gfn2CudaExecutionCache(std::int32_t device_id, void* stream)
    : impl_(std::make_unique<Impl>(device_id, stream)) {}

Gfn2CudaExecutionCache::~Gfn2CudaExecutionCache() = default;

gpuxtb_status_t Gfn2CudaExecutionCache::prepare_host(const gpuxtb_batch_t& batch,
                                                     const gpuxtb_compute_options_t& options,
                                                     bool& reused, std::string& error) {
  reused = false;
  if (impl_ == nullptr) {
    error = "CUDA GFN2 execution cache has no implementation";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  gpuxtb_status_t status = impl_->ensure_handles(error);
  if (status != GPUXTB_STATUS_SUCCESS) return status;

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
      status = impl_->refresh_numerical_locked(numerical, error);
      reused = status == GPUXTB_STATUS_SUCCESS;
      return status;
    }
    if (match == TopologyMatch::kInvalid) {
      return GPUXTB_STATUS_INVALID_ARGUMENT;
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
    if (status != GPUXTB_STATUS_SUCCESS) return status;
    status = impl_->build_candidate(std::move(key), std::move(positions),
                                    std::move(point_positions), std::move(point_values),
                                    std::move(point_gammas), std::move(periodic_shifts),
                                    std::move(periodic_response), candidate, error);
  } catch (const std::bad_alloc&) {
    error = "failed to allocate host metadata for the CUDA GFN2 runtime candidate";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  } catch (const std::exception& exception) {
    error = exception.what();
    return GPUXTB_STATUS_INTERNAL_ERROR;
  } catch (...) {
    error = "unknown exception while constructing CUDA GFN2 runtime candidate";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  if (status != GPUXTB_STATUS_SUCCESS) return status;

  impl_->prepared = std::move(candidate);
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t Gfn2CudaExecutionCache::refresh_numerical_async(
    const Gfn2CudaNumericalInputView& input, std::string& error) {
  if (impl_ == nullptr) {
    error = "CUDA GFN2 execution cache has no implementation";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  gpuxtb_status_t status = impl_->ensure_handles(error);
  return status == GPUXTB_STATUS_SUCCESS ? impl_->refresh_numerical_locked(input, error) : status;
}

bool Gfn2CudaExecutionCache::valid() const noexcept {
  if (impl_ == nullptr) return false;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->prepared != nullptr;
}

Gfn2CudaExecutionIdentity Gfn2CudaExecutionCache::identity() const noexcept {
  if (impl_ == nullptr) return {};
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->snapshot();
}

}  // namespace gpuxtb::detail
