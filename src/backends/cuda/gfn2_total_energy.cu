#include <array>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_total_energy.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;

constexpr __host__ __device__ bool component_enabled(std::uint32_t mask,
                                                     Gfn2TotalEnergyComponent component) noexcept {
  return (mask & static_cast<std::uint32_t>(component)) != 0u;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool is_aligned(const void* pointer, std::size_t alignment) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

bool make_range(const void* pointer, std::int64_t elements, std::size_t element_size,
                AddressRange* range) noexcept {
  if (range == nullptr || elements < 0 || element_size == 0u ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(elements) * element_size;
  if (bytes == 0u) {
    if (pointer != nullptr) {
      return false;
    }
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

bool valid_input(const double* pointer, std::int64_t elements, std::int64_t batch_size,
                 bool enabled) noexcept {
  if (!enabled) {
    return pointer == nullptr && elements == 0;
  }
  return elements == batch_size && is_aligned(pointer, alignof(double));
}

bool validate_launch(const Gfn2TotalEnergyDeviceBatch& batch,
                     const Gfn2TotalEnergyDeviceInput& input,
                     const Gfn2TotalEnergyDeviceSccState& scc_state,
                     const Gfn2TotalEnergyDeviceResults& results,
                     const Gfn2TotalEnergyDeviceWorkspace& workspace, std::uint32_t* system_errors,
                     std::uint32_t* device_error) noexcept {
  const bool d4_atm = component_enabled(batch.enabled_components, Gfn2TotalEnergyComponent::kD4Atm);
  if (batch.batch_size <= 0 ||
      batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      (batch.enabled_components & ~kGfn2TotalEnergyAllComponents) != 0u || batch.plan_token == 0u ||
      input.plan_token != batch.plan_token || scc_state.plan_token != batch.plan_token ||
      results.plan_token != batch.plan_token || workspace.plan_token != batch.plan_token ||
      !valid_input(input.scc_free_energy, input.scc_free_energy_elements, batch.batch_size, true) ||
      !valid_input(input.repulsion, input.repulsion_elements, batch.batch_size, true) ||
      !valid_input(input.d4_atm, input.d4_atm_elements, batch.batch_size, d4_atm) ||
      scc_state.elements != batch.batch_size ||
      !is_aligned(scc_state.system_statuses, alignof(xtbloom_status_t)) ||
      !is_aligned(scc_state.converged, alignof(std::uint8_t)) ||
      results.elements != batch.batch_size || !is_aligned(results.total_energy, alignof(double)) ||
      workspace.elements != 1 || !is_aligned(workspace.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return false;
  }

  std::array<AddressRange, 5> reads{};
  std::array<AddressRange, 4> writes{};
  if (!make_range(input.scc_free_energy, input.scc_free_energy_elements, sizeof(double),
                  &reads[0]) ||
      !make_range(input.repulsion, input.repulsion_elements, sizeof(double), &reads[1]) ||
      !make_range(input.d4_atm, input.d4_atm_elements, sizeof(double), &reads[2]) ||
      !make_range(scc_state.system_statuses, scc_state.elements, sizeof(xtbloom_status_t),
                  &reads[3]) ||
      !make_range(scc_state.converged, scc_state.elements, sizeof(std::uint8_t), &reads[4]) ||
      !make_range(results.total_energy, results.elements, sizeof(double), &writes[0]) ||
      !make_range(workspace.sequence_active, workspace.elements, sizeof(std::uint32_t),
                  &writes[1]) ||
      !make_range(system_errors, batch.batch_size, sizeof(std::uint32_t), &writes[2]) ||
      !make_range(device_error, 1, sizeof(std::uint32_t), &writes[3])) {
    return false;
  }
  for (std::size_t lhs = 0; lhs < writes.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < writes.size(); ++rhs) {
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

__device__ bool sequence_is_active(const Gfn2TotalEnergyDeviceWorkspace& workspace) {
  return atomicAdd(workspace.sequence_active, 0u) == 1u;
}

__device__ bool system_is_valid(const std::uint32_t* system_errors, std::int64_t system) {
  return atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) ==
         static_cast<std::uint32_t>(Gfn2TotalEnergyDeviceError::kSuccess);
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    std::uint32_t* device_error, Gfn2TotalEnergyDeviceError error) {
  constexpr std::uint32_t success =
      static_cast<std::uint32_t>(Gfn2TotalEnergyDeviceError::kSuccess);
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  if (atomicCAS(system_errors + system, success, code) == success) {
    atomicCAS(device_error, success, code);
  }
}

__global__ void capture_sequence_kernel(const std::uint32_t* device_error,
                                        Gfn2TotalEnergyDeviceWorkspace workspace) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *workspace.sequence_active =
        atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                static_cast<std::uint32_t>(Gfn2TotalEnergyDeviceError::kSuccess)
            ? 1u
            : 0u;
  }
}

/* One lane owns the exact serial CPU addition chain for one batch member. */
__global__ void compose_total_energy_kernel(Gfn2TotalEnergyDeviceBatch batch,
                                            Gfn2TotalEnergyDeviceInput input,
                                            Gfn2TotalEnergyDeviceSccState scc_state,
                                            Gfn2TotalEnergyDeviceResults results,
                                            Gfn2TotalEnergyDeviceWorkspace workspace,
                                            std::uint32_t* system_errors,
                                            std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= batch.batch_size || !sequence_is_active(workspace) ||
      !system_is_valid(system_errors, system)) {
    return;
  }

  const std::uint8_t converged = scc_state.converged[system];
  if (converged == 0u) {
    return;
  }
  if (converged != 1u) {
    record_system_error(system_errors, system, device_error,
                        Gfn2TotalEnergyDeviceError::kInvalidConvergenceFlag);
    return;
  }
  if (scc_state.system_statuses[system] != XTBLOOM_STATUS_SUCCESS) {
    record_system_error(system_errors, system, device_error,
                        Gfn2TotalEnergyDeviceError::kInconsistentSccStatus);
    return;
  }

  const double scc = input.scc_free_energy[system];
  const double repulsion = input.repulsion[system];
  if (!isfinite(scc)) {
    record_system_error(system_errors, system, device_error,
                        Gfn2TotalEnergyDeviceError::kNonfiniteSccFreeEnergy);
    return;
  }
  if (!isfinite(repulsion)) {
    record_system_error(system_errors, system, device_error,
                        Gfn2TotalEnergyDeviceError::kNonfiniteRepulsion);
    return;
  }

  double total = scc + repulsion;
  if (!isfinite(total)) {
    record_system_error(system_errors, system, device_error,
                        Gfn2TotalEnergyDeviceError::kNonfiniteSccRepulsionSum);
    return;
  }
  if (component_enabled(batch.enabled_components, Gfn2TotalEnergyComponent::kD4Atm)) {
    const double d4_atm = input.d4_atm[system];
    if (!isfinite(d4_atm)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2TotalEnergyDeviceError::kNonfiniteD4Atm);
      return;
    }
    total += d4_atm;
    if (!isfinite(total)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2TotalEnergyDeviceError::kNonfiniteTotalArithmetic);
      return;
    }
  }
  results.total_energy[system] = total;
}

}  // namespace

cudaError_t reset_gfn2_total_energy_device_errors_cuda(std::int64_t batch_size,
                                                       std::uint32_t* system_errors,
                                                       std::uint32_t* device_error,
                                                       cudaStream_t stream) noexcept {
  if (batch_size <= 0 ||
      static_cast<std::uint64_t>(batch_size) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() /
                                     sizeof(*system_errors)) ||
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
  cudaError_t status = cudaMemsetAsync(
      system_errors, 0, static_cast<std::size_t>(batch_size) * sizeof(*system_errors), stream);
  if (status != cudaSuccess) {
    return status;
  }
  return cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream);
}

cudaError_t compose_gfn2_total_energy_cuda(
    const Gfn2TotalEnergyDeviceBatch& batch, const Gfn2TotalEnergyDeviceInput& input,
    const Gfn2TotalEnergyDeviceSccState& scc_state, const Gfn2TotalEnergyDeviceResults& results,
    const Gfn2TotalEnergyDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (!validate_launch(batch, input, scc_state, results, workspace, system_errors, device_error)) {
    return cudaErrorInvalidValue;
  }

  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  const unsigned int blocks =
      static_cast<unsigned int>((batch.batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock);
  compose_total_energy_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, input, scc_state, results, workspace, system_errors, device_error);
  return cudaGetLastError();
}

}  // namespace xtbloom::detail::cuda
