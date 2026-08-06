#include <cuda_runtime.h>
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_terminal_classical_energy.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;

using PlanError = Gfn2TerminalClassicalEnergyPlanError;
using SystemError = Gfn2TerminalClassicalEnergySystemError;

bool component_enabled(const Gfn2TerminalClassicalEnergyDevicePlan& plan,
                       Gfn2TerminalClassicalEnergyComponent component) noexcept {
  return (plan.enabled_components & static_cast<std::uint32_t>(component)) != 0u;
}

bool aligned(const void* pointer, std::size_t alignment) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

template <typename T>
bool canonical_pointer(const T* pointer, std::int64_t elements) noexcept {
  if (elements < 0) return false;
  if (elements == 0) return pointer == nullptr;
  return aligned(pointer, alignof(T));
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

template <typename T>
bool make_range(const T* pointer, std::int64_t elements, AddressRange& range) noexcept {
  if (elements < 0 ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(T))) {
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(T);
  if (bytes == 0u) {
    range = {};
    return pointer == nullptr;
  }
  if (pointer == nullptr) return false;
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) return false;
  range = {begin, begin + bytes};
  return true;
}

bool overlaps(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin < second.end && second.begin < first.end;
}

template <std::size_t ReadCount, std::size_t WriteCount>
bool writes_are_disjoint(const std::array<AddressRange, ReadCount>& reads,
                         const std::array<AddressRange, WriteCount>& writes) noexcept {
  for (std::size_t write = 0u; write < WriteCount; ++write) {
    for (const AddressRange& read : reads) {
      if (overlaps(writes[write], read)) return false;
    }
    for (std::size_t peer = write + 1u; peer < WriteCount; ++peer) {
      if (overlaps(writes[write], writes[peer])) return false;
    }
  }
  return true;
}

bool valid_common(const Gfn2TerminalClassicalEnergyDevicePlan& plan,
                  const Gfn2TerminalClassicalEnergyDeviceActivity& activity,
                  const Gfn2TerminalClassicalEnergyDeviceResults& results,
                  const Gfn2TerminalClassicalEnergyDeviceWorkspace& workspace,
                  const Gfn2TerminalClassicalEnergyDeviceDiagnostics& diagnostics) noexcept {
  const std::int64_t batch = plan.repulsion.batch_size;
  const bool d4 = component_enabled(plan, Gfn2TerminalClassicalEnergyComponent::kD4Atm);
  if (plan.abi_version != kGfn2TerminalClassicalEnergyAbiVersion || plan.plan_token == 0u ||
      (plan.enabled_components & ~kGfn2TerminalClassicalEnergyAllComponents) != 0u || batch <= 0 ||
      batch > std::numeric_limits<int>::max() || plan.repulsion.total_atoms <= 0 ||
      plan.repulsion.total_atoms > std::numeric_limits<std::int64_t>::max() / 3 ||
      plan.repulsion.atom_offsets == nullptr || plan.repulsion.atomic_numbers == nullptr ||
      plan.repulsion.positions == nullptr || plan.geometry_epoch.value_elements != 1 ||
      plan.geometry_epoch.plan_token != plan.plan_token ||
      !aligned(plan.geometry_epoch.value, alignof(std::uint64_t)) ||
      plan.generation_elements != batch ||
      !aligned(plan.committed_generations, alignof(std::uint64_t)) ||
      activity.plan_token != plan.plan_token || activity.batch_elements != batch ||
      !aligned(activity.requested_mask, alignof(std::uint8_t)) ||
      results.plan_token != plan.plan_token || results.repulsion_elements != batch ||
      !aligned(results.repulsion, alignof(double)) || workspace.plan_token != plan.plan_token ||
      workspace.repulsion_elements != batch ||
      !aligned(workspace.repulsion_candidate, alignof(double)) ||
      workspace.epoch_snapshot_elements != 1 ||
      !aligned(workspace.epoch_snapshot, alignof(std::uint64_t)) ||
      diagnostics.plan_token != plan.plan_token || diagnostics.system_error_elements != batch ||
      !aligned(diagnostics.system_errors, alignof(std::uint32_t)) ||
      diagnostics.plan_error_elements != 1 ||
      !aligned(diagnostics.plan_error, alignof(std::uint32_t)) ||
      !aligned(diagnostics.repulsion_device_error, alignof(std::uint32_t))) {
    return false;
  }
  if (d4) {
    if (plan.d4_batch.batch_size != batch ||
        plan.d4_batch.total_atoms != plan.repulsion.total_atoms ||
        plan.d4_batch.plan_token != plan.plan_token ||
        plan.d4_cache.plan_token != plan.plan_token ||
        plan.d4_batch.atom_offsets != plan.repulsion.atom_offsets ||
        plan.d4_batch.atomic_numbers != plan.repulsion.atomic_numbers ||
        results.d4_atm_elements != batch || !aligned(results.d4_atm, alignof(double)) ||
        workspace.d4_atm_elements != batch ||
        !aligned(workspace.d4_atm_candidate, alignof(double)) ||
        diagnostics.d4_system_error_elements != batch ||
        !aligned(diagnostics.d4_system_errors, alignof(std::uint32_t)) ||
        !aligned(diagnostics.d4_device_error, alignof(std::uint32_t)) ||
        workspace.d4.system_errors != diagnostics.d4_system_errors ||
        workspace.d4.system_error_elements < batch) {
      return false;
    }
  } else if (results.d4_atm != nullptr || results.d4_atm_elements != 0 ||
             workspace.d4_atm_candidate != nullptr || workspace.d4_atm_elements != 0 ||
             diagnostics.d4_system_errors != nullptr || diagnostics.d4_system_error_elements != 0 ||
             diagnostics.d4_device_error != nullptr) {
    return false;
  }

  std::array<AddressRange, 6> reads{};
  std::array<AddressRange, 9> writes{};
  if (!make_range(plan.repulsion.atom_offsets, batch + 1, reads[0]) ||
      !make_range(plan.repulsion.atomic_numbers, plan.repulsion.total_atoms, reads[1]) ||
      !make_range(plan.repulsion.positions, plan.repulsion.total_atoms * 3, reads[2]) ||
      !make_range(plan.geometry_epoch.value, 1, reads[3]) ||
      !make_range(plan.committed_generations, batch, reads[4]) ||
      !make_range(activity.requested_mask, batch, reads[5]) ||
      !make_range(results.repulsion, batch, writes[0]) ||
      !make_range(results.d4_atm, d4 ? batch : 0, writes[1]) ||
      !make_range(workspace.repulsion_candidate, batch, writes[2]) ||
      !make_range(workspace.d4_atm_candidate, d4 ? batch : 0, writes[3]) ||
      !make_range(workspace.epoch_snapshot, 1, writes[4]) ||
      !make_range(diagnostics.repulsion_device_error, 1, writes[5]) ||
      !make_range(diagnostics.d4_system_errors, d4 ? batch : 0, writes[6]) ||
      !make_range(diagnostics.d4_device_error, d4 ? 1 : 0, writes[7]) ||
      !make_range(diagnostics.system_errors, batch, writes[8]) ||
      !writes_are_disjoint(reads, writes)) {
    return false;
  }
  AddressRange plan_error{};
  if (!make_range(diagnostics.plan_error, 1, plan_error)) return false;
  for (const AddressRange& read : reads) {
    if (overlaps(plan_error, read)) return false;
  }
  for (const AddressRange& write : writes) {
    if (overlaps(plan_error, write)) return false;
  }
  return true;
}

__device__ void record_plan_error(std::uint32_t* output, PlanError error) {
  atomicCAS(output, static_cast<std::uint32_t>(PlanError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ void record_system_error(std::uint32_t* output, std::int64_t system, SystemError error) {
  atomicCAS(output + system, static_cast<std::uint32_t>(SystemError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__global__ void capture_epoch_and_validate_activity_kernel(
    Gfn2TerminalClassicalEnergyDevicePlan plan, Gfn2TerminalClassicalEnergyDeviceActivity activity,
    Gfn2TerminalClassicalEnergyDeviceWorkspace workspace,
    Gfn2TerminalClassicalEnergyDeviceDiagnostics diagnostics) {
  __shared__ unsigned long long epoch;
  if (threadIdx.x == 0) {
    epoch = atomicAdd(reinterpret_cast<unsigned long long*>(plan.geometry_epoch.value), 0ULL);
    *workspace.epoch_snapshot = static_cast<std::uint64_t>(epoch);
    if (epoch == 0ULL) record_plan_error(diagnostics.plan_error, PlanError::kInvalidEpoch);
  }
  __syncthreads();
  for (std::int64_t system = threadIdx.x; system < plan.repulsion.batch_size;
       system += blockDim.x) {
    if (activity.requested_mask[system] > 1u) {
      record_plan_error(diagnostics.plan_error, PlanError::kInvalidRequestedMask);
    }
  }
}

__global__ void publish_terminal_classical_energy_kernel(
    Gfn2TerminalClassicalEnergyDevicePlan plan, Gfn2TerminalClassicalEnergyDeviceActivity activity,
    Gfn2TerminalClassicalEnergyDeviceResults results,
    Gfn2TerminalClassicalEnergyDeviceWorkspace workspace,
    Gfn2TerminalClassicalEnergyDeviceDiagnostics diagnostics) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool d4 = (plan.enabled_components &
                   static_cast<std::uint32_t>(Gfn2TerminalClassicalEnergyComponent::kD4Atm)) != 0u;

  /*
   * Repulsion exposes one device-wide diagnostic rather than peer codes. The
   * runtime refresh gate has already removed peers with invalid coordinates,
   * atomic data, or geometry provenance before this terminal stage. A
   * remaining repulsion diagnostic therefore denotes a violated execution
   * contract and must suppress publication for the complete batch.
   */
  if (atomicAdd(diagnostics.repulsion_device_error, 0u) != 0u) {
    record_plan_error(diagnostics.plan_error, PlanError::kRepulsionFailure);
  }
  if (d4 && atomicAdd(diagnostics.d4_device_error, 0u) != 0u) {
    record_plan_error(diagnostics.plan_error, PlanError::kD4AtmPlanFailure);
  }
  __threadfence();
  if (system >= plan.repulsion.batch_size || activity.requested_mask[system] != 1u ||
      atomicAdd(diagnostics.plan_error, 0u) != 0u) {
    return;
  }

  const std::uint64_t epoch =
      atomicAdd(reinterpret_cast<unsigned long long*>(workspace.epoch_snapshot), 0ULL);
  if (plan.committed_generations[system] != epoch) {
    record_system_error(diagnostics.system_errors, system, SystemError::kStaleGeneration);
    return;
  }
  if (d4 && diagnostics.d4_system_errors[system] != 0u) {
    record_system_error(diagnostics.system_errors, system, SystemError::kD4AtmFailure);
    return;
  }
  const double repulsion = workspace.repulsion_candidate[system];
  if (!isfinite(repulsion)) {
    record_system_error(diagnostics.system_errors, system, SystemError::kNonfiniteRepulsion);
    return;
  }
  double atm = 0.0;
  if (d4) {
    atm = workspace.d4_atm_candidate[system];
    if (!isfinite(atm)) {
      record_system_error(diagnostics.system_errors, system, SystemError::kNonfiniteD4Atm);
      return;
    }
  }

  /* No operation below this point can fail: publish one complete tuple. */
  results.repulsion[system] = repulsion;
  if (d4) results.d4_atm[system] = atm;
}

cudaError_t check_launch() noexcept { return cudaPeekAtLastError(); }

}  // namespace

cudaError_t evaluate_gfn2_terminal_classical_energy_cuda(
    const Gfn2TerminalClassicalEnergyDevicePlan& plan,
    const Gfn2TerminalClassicalEnergyDeviceActivity& activity,
    const Gfn2TerminalClassicalEnergyDeviceResults& results,
    const Gfn2TerminalClassicalEnergyDeviceWorkspace& workspace,
    const Gfn2TerminalClassicalEnergyDeviceDiagnostics& diagnostics, cudaStream_t stream) noexcept {
  if (!valid_common(plan, activity, results, workspace, diagnostics)) {
    return cudaErrorInvalidValue;
  }
  const std::int64_t batch = plan.repulsion.batch_size;
  const bool d4 = component_enabled(plan, Gfn2TerminalClassicalEnergyComponent::kD4Atm);

  cudaError_t status =
      cudaMemsetAsync(diagnostics.system_errors, 0,
                      static_cast<std::size_t>(batch) * sizeof(std::uint32_t), stream);
  if (status == cudaSuccess) {
    status = cudaMemsetAsync(diagnostics.plan_error, 0, sizeof(std::uint32_t), stream);
  }
  if (status == cudaSuccess) {
    status = cudaMemsetAsync(workspace.repulsion_candidate, 0,
                             static_cast<std::size_t>(batch) * sizeof(double), stream);
  }
  if (status == cudaSuccess) {
    status = cudaMemsetAsync(diagnostics.repulsion_device_error, 0, sizeof(std::uint32_t), stream);
  }
  if (status != cudaSuccess) return status;

  if (d4) {
    status = cudaMemsetAsync(workspace.d4_atm_candidate, 0,
                             static_cast<std::size_t>(batch) * sizeof(double), stream);
    if (status == cudaSuccess) {
      status = reset_gfn2_d4_device_errors_cuda(batch, diagnostics.d4_system_errors,
                                                diagnostics.d4_device_error, stream);
    }
    if (status != cudaSuccess) return status;
  }

  capture_epoch_and_validate_activity_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
      plan, activity, workspace, diagnostics);
  status = check_launch();
  if (status != cudaSuccess) return status;

  status = add_gfn2_repulsion_cuda(plan.repulsion, workspace.repulsion_candidate, nullptr,
                                   diagnostics.repulsion_device_error, stream);
  if (status != cudaSuccess) return status;
  if (d4) {
    status = evaluate_gfn2_d4_atm_cuda(plan.d4_batch, plan.d4_parameters, plan.d4_cache,
                                       workspace.d4_atm_candidate, workspace.d4,
                                       diagnostics.d4_device_error, stream);
    if (status != cudaSuccess) return status;
  }

  const unsigned int blocks = static_cast<unsigned int>(
      (static_cast<std::uint64_t>(batch) + kThreadsPerBlock - 1u) / kThreadsPerBlock);
  publish_terminal_classical_energy_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      plan, activity, results, workspace, diagnostics);
  return check_launch();
}

}  // namespace gpuxtb::detail::cuda
