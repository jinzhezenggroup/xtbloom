#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_scc_energy.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;

static_assert((kThreadsPerBlock & (kThreadsPerBlock - 1)) == 0,
              "SCC energy reduction requires a power-of-two block size");

__device__ void record_error(std::uint32_t* device_error, Gfn2SccEnergyDeviceError error) {
  atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2SccEnergyDeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    std::uint32_t* device_error, Gfn2SccEnergyDeviceError error) {
  const std::uint32_t success = static_cast<std::uint32_t>(Gfn2SccEnergyDeviceError::kSuccess);
  if (atomicCAS(system_errors + system, success, static_cast<std::uint32_t>(error)) == success) {
    record_error(device_error, error);
  }
}

__global__ void reset_errors_kernel(std::int64_t batch_size, std::uint32_t* system_errors,
                                    std::uint32_t* device_error) {
  for (std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       system < batch_size; system += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    system_errors[system] = static_cast<std::uint32_t>(Gfn2SccEnergyDeviceError::kSuccess);
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *device_error = static_cast<std::uint32_t>(Gfn2SccEnergyDeviceError::kSuccess);
  }
}

__global__ void topology_preflight_kernel(Gfn2SccEnergyDeviceBatch batch,
                                          std::uint32_t* device_error) {
  if (atomicAdd(device_error, 0u) !=
      static_cast<std::uint32_t>(Gfn2SccEnergyDeviceError::kSuccess)) {
    return;
  }
  if (threadIdx.x == 0 && (batch.matrix_offsets[0] != 0 ||
                           batch.matrix_offsets[batch.batch_size] != batch.total_matrix_elements)) {
    record_error(device_error, Gfn2SccEnergyDeviceError::kInvalidOffsets);
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    const std::int64_t begin = batch.matrix_offsets[system];
    const std::int64_t end = batch.matrix_offsets[system + 1];
    if (begin < 0 || begin > end || end > batch.total_matrix_elements) {
      record_error(device_error, Gfn2SccEnergyDeviceError::kInvalidOffsets);
    }
  }
}

/* Snapshot topology/upstream validity before per-system arithmetic can fail. */
__global__ void capture_sequence_kernel(const std::uint32_t* device_error,
                                        std::uint32_t* sequence_active) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *sequence_active = atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                               static_cast<std::uint32_t>(Gfn2SccEnergyDeviceError::kSuccess)
                           ? 1u
                           : 0u;
  }
}

__global__ void reduce_electronic_energy_kernel(
    Gfn2SccEnergyDeviceBatch batch, const double* density, const double* h0,
    const double* entropies, double electronic_temperature, const std::uint8_t* active_systems,
    Gfn2SccEnergyDeviceWorkspace workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int active;
  __shared__ int valid;
  __shared__ double partial[kThreadsPerBlock];

  if (threadIdx.x == 0) {
    active = 0;
    valid = 1;
    if (atomicAdd(workspace.sequence_active, 0u) == 1u &&
        atomicAdd(system_errors + system, 0u) ==
            static_cast<std::uint32_t>(Gfn2SccEnergyDeviceError::kSuccess)) {
      const std::uint8_t state = active_systems == nullptr ? 1u : active_systems[system];
      if (state > 1u) {
        record_system_error(system_errors, system, device_error,
                            Gfn2SccEnergyDeviceError::kInvalidActiveState);
        valid = 0;
      } else {
        active = state == 1u ? 1 : 0;
      }
    }
  }
  __syncthreads();
  if (active == 0) {
    return;
  }

  const std::int64_t begin = batch.matrix_offsets[system];
  const std::int64_t end = batch.matrix_offsets[system + 1];
  double local = 0.0;
  for (std::int64_t element = begin + threadIdx.x; element < end; element += blockDim.x) {
    const double density_value = density[element];
    const double h0_value = h0[element];
    if (!isfinite(density_value)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccEnergyDeviceError::kNonfiniteDensity);
      atomicExch(&valid, 0);
      continue;
    }
    if (!isfinite(h0_value)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccEnergyDeviceError::kNonfiniteH0);
      atomicExch(&valid, 0);
      continue;
    }
    const double updated = fma(h0_value, density_value, local);
    if (!isfinite(updated)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccEnergyDeviceError::kNonfiniteCoreArithmetic);
      atomicExch(&valid, 0);
      continue;
    }
    local = updated;
  }
  partial[threadIdx.x] = local;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      const double updated = partial[threadIdx.x] + partial[threadIdx.x + offset];
      if (!isfinite(updated)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2SccEnergyDeviceError::kNonfiniteCoreArithmetic);
        atomicExch(&valid, 0);
      } else {
        partial[threadIdx.x] = updated;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && valid != 0) {
    const double entropy = entropies[system];
    if (!isfinite(entropy)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccEnergyDeviceError::kNonfiniteEntropy);
    } else {
      const double free_energy = fma(-electronic_temperature, entropy, partial[0]);
      if (!isfinite(free_energy)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2SccEnergyDeviceError::kNonfiniteFreeEnergy);
      } else {
        workspace.core_energy_scratch[system] = partial[0];
        workspace.electronic_free_energy_scratch[system] = free_energy;
      }
    }
  }
}

__global__ void publish_electronic_energy_kernel(Gfn2SccEnergyDeviceBatch batch,
                                                 const std::uint8_t* active_systems,
                                                 double* core_energies,
                                                 double* electronic_free_energies,
                                                 Gfn2SccEnergyDeviceWorkspace workspace,
                                                 const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= batch.batch_size || atomicAdd(workspace.sequence_active, 0u) != 1u ||
      atomicAdd(const_cast<std::uint32_t*>(system_errors + system), 0u) !=
          static_cast<std::uint32_t>(Gfn2SccEnergyDeviceError::kSuccess) ||
      (active_systems != nullptr && active_systems[system] != 1u)) {
    return;
  }
  core_energies[system] = workspace.core_energy_scratch[system];
  electronic_free_energies[system] = workspace.electronic_free_energy_scratch[system];
}

bool is_aligned(const void* pointer, std::size_t alignment) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

bool required_pointer(const void* pointer, std::int64_t elements) noexcept {
  return elements == 0 || pointer != nullptr;
}

bool aligned_or_empty(const void* pointer, std::int64_t elements, std::size_t alignment) noexcept {
  return elements == 0 || is_aligned(pointer, alignment);
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool make_range(const void* pointer, std::int64_t elements, std::size_t element_size,
                AddressRange* range) noexcept {
  if (elements < 0 || element_size == 0u ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(elements) * element_size;
  if (bytes == 0u) {
    *range = {};
    return true;
  }
  if (pointer == nullptr) {
    return false;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  *range = {begin, begin + bytes};
  return true;
}

bool ranges_overlap(const AddressRange& lhs, const AddressRange& rhs) noexcept {
  return lhs.begin != lhs.end && rhs.begin != rhs.end && lhs.begin < rhs.end && rhs.begin < lhs.end;
}

bool validate_launch(const Gfn2SccEnergyDeviceBatch& batch, const double* density, const double* h0,
                     const double* entropies, double electronic_temperature,
                     const std::uint8_t* active_systems, double* core_energies,
                     double* electronic_free_energies,
                     const Gfn2SccEnergyDeviceWorkspace& workspace, std::uint32_t* system_errors,
                     std::uint32_t* device_error) noexcept {
  if (batch.batch_size <= 0 ||
      batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      batch.total_matrix_elements < 0 || batch.matrix_offset_count != batch.batch_size + 1 ||
      batch.plan_token == 0u || workspace.plan_token != batch.plan_token ||
      workspace.batch_elements < batch.batch_size || workspace.sequence_elements < 1 ||
      !std::isfinite(electronic_temperature) || electronic_temperature < 0.0 ||
      !is_aligned(batch.matrix_offsets, alignof(std::int64_t)) ||
      !required_pointer(density, batch.total_matrix_elements) ||
      !required_pointer(h0, batch.total_matrix_elements) || entropies == nullptr ||
      core_energies == nullptr || electronic_free_energies == nullptr ||
      workspace.core_energy_scratch == nullptr ||
      workspace.electronic_free_energy_scratch == nullptr || workspace.sequence_active == nullptr ||
      system_errors == nullptr || device_error == nullptr ||
      !aligned_or_empty(density, batch.total_matrix_elements, alignof(double)) ||
      !aligned_or_empty(h0, batch.total_matrix_elements, alignof(double)) ||
      !is_aligned(entropies, alignof(double)) || !is_aligned(core_energies, alignof(double)) ||
      !is_aligned(electronic_free_energies, alignof(double)) ||
      !is_aligned(workspace.core_energy_scratch, alignof(double)) ||
      !is_aligned(workspace.electronic_free_energy_scratch, alignof(double)) ||
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return false;
  }

  AddressRange reads[5];
  AddressRange writes[7];
  if (!make_range(batch.matrix_offsets, batch.matrix_offset_count, sizeof(*batch.matrix_offsets),
                  &reads[0]) ||
      !make_range(density, batch.total_matrix_elements, sizeof(*density), &reads[1]) ||
      !make_range(h0, batch.total_matrix_elements, sizeof(*h0), &reads[2]) ||
      !make_range(entropies, batch.batch_size, sizeof(*entropies), &reads[3]) ||
      !make_range(active_systems, active_systems == nullptr ? 0 : batch.batch_size,
                  sizeof(*active_systems), &reads[4]) ||
      !make_range(core_energies, batch.batch_size, sizeof(*core_energies), &writes[0]) ||
      !make_range(electronic_free_energies, batch.batch_size, sizeof(*electronic_free_energies),
                  &writes[1]) ||
      !make_range(workspace.core_energy_scratch, batch.batch_size,
                  sizeof(*workspace.core_energy_scratch), &writes[2]) ||
      !make_range(workspace.electronic_free_energy_scratch, batch.batch_size,
                  sizeof(*workspace.electronic_free_energy_scratch), &writes[3]) ||
      !make_range(workspace.sequence_active, 1, sizeof(*workspace.sequence_active), &writes[4]) ||
      !make_range(system_errors, batch.batch_size, sizeof(*system_errors), &writes[5]) ||
      !make_range(device_error, 1, sizeof(*device_error), &writes[6])) {
    return false;
  }
  for (std::size_t lhs = 0; lhs < 7; ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < 7; ++rhs) {
      if (ranges_overlap(writes[lhs], writes[rhs])) {
        return false;
      }
    }
    for (const AddressRange& read : reads) {
      if (ranges_overlap(writes[lhs], read)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

cudaError_t reset_gfn2_scc_energy_device_errors_cuda(std::int64_t batch_size,
                                                     std::uint32_t* system_errors,
                                                     std::uint32_t* device_error,
                                                     cudaStream_t stream) noexcept {
  if (batch_size <= 0 || system_errors == nullptr || device_error == nullptr ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  AddressRange system_range;
  AddressRange device_range;
  if (!make_range(system_errors, batch_size, sizeof(*system_errors), &system_range) ||
      !make_range(device_error, 1, sizeof(*device_error), &device_range) ||
      ranges_overlap(system_range, device_range)) {
    return cudaErrorInvalidValue;
  }
  const std::int64_t needed = 1 + (batch_size - 1) / kThreadsPerBlock;
  const int blocks = static_cast<int>(needed > 1024 ? 1024 : needed);
  reset_errors_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(batch_size, system_errors,
                                                               device_error);
  return cudaGetLastError();
}

cudaError_t evaluate_gfn2_scc_electronic_energy_cuda(
    const Gfn2SccEnergyDeviceBatch& batch, const double* density, const double* h0,
    const double* entropies, double electronic_temperature, const std::uint8_t* active_systems,
    double* core_energies, double* electronic_free_energies,
    const Gfn2SccEnergyDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (!validate_launch(batch, density, h0, entropies, electronic_temperature, active_systems,
                       core_energies, electronic_free_energies, workspace, system_errors,
                       device_error)) {
    return cudaErrorInvalidValue;
  }

  topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, device_error);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace.sequence_active);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  reduce_electronic_energy_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock,
                                    0, stream>>>(batch, density, h0, entropies,
                                                 electronic_temperature, active_systems, workspace,
                                                 system_errors, device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  const unsigned int blocks =
      static_cast<unsigned int>((batch.batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock);
  publish_electronic_energy_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, active_systems, core_energies, electronic_free_energies, workspace, system_errors);
  return cudaGetLastError();
}

}  // namespace gpuxtb::detail::cuda
