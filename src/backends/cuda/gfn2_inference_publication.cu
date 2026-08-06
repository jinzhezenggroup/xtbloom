#include <cuda_runtime.h>
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_inference_publication.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
/* Keep device code independent of host-only libstdc++ numeric_limits methods. */
constexpr std::int32_t kInt32Maximum = 2147483647;
static_assert(kInt32Maximum == std::numeric_limits<std::int32_t>::max());
constexpr std::uint32_t kKnownProperties =
    static_cast<std::uint32_t>(GPUXTB_COMPUTE_ENERGY) |
    static_cast<std::uint32_t>(GPUXTB_COMPUTE_FORCES) |
    static_cast<std::uint32_t>(GPUXTB_COMPUTE_ATOMIC_CHARGES) |
    static_cast<std::uint32_t>(GPUXTB_COMPUTE_POINT_CHARGE_FORCES);

using PlanError = Gfn2InferencePublicationPlanError;
using SystemError = Gfn2InferencePublicationSystemError;

bool property_requested(const Gfn2InferencePublicationDevicePlan& plan,
                        gpuxtb_compute_flag_t property) noexcept {
  return (plan.requested_properties & static_cast<std::uint32_t>(property)) != 0u;
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

bool checked_multiply(std::int64_t first, std::int64_t second, std::int64_t& product) noexcept {
  if (first < 0 || second < 0 ||
      (first != 0 && second > std::numeric_limits<std::int64_t>::max() / first)) {
    return false;
  }
  product = first * second;
  return true;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool overlaps(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin < second.end && second.begin < first.end;
}

template <std::size_t Capacity>
class RangeList {
 public:
  template <typename T>
  bool add(const T* pointer, std::int64_t elements) noexcept {
    if (size_ == Capacity || elements < 0 ||
        static_cast<std::uint64_t>(elements) >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(T))) {
      return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(T);
    if (bytes == 0u) {
      ranges_[size_++] = {};
      return pointer == nullptr;
    }
    if (pointer == nullptr) return false;
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) return false;
    ranges_[size_++] = {begin, begin + bytes};
    return true;
  }

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] const AddressRange& operator[](std::size_t index) const noexcept {
    return ranges_[index];
  }

 private:
  std::array<AddressRange, Capacity> ranges_{};
  std::size_t size_ = 0u;
};

template <std::size_t ReadCapacity, std::size_t WriteCapacity>
bool writes_are_disjoint(const RangeList<ReadCapacity>& reads,
                         const RangeList<WriteCapacity>& writes) noexcept {
  for (std::size_t write = 0u; write < writes.size(); ++write) {
    for (std::size_t read = 0u; read < reads.size(); ++read) {
      if (overlaps(writes[write], reads[read])) return false;
    }
    for (std::size_t peer = write + 1u; peer < writes.size(); ++peer) {
      if (overlaps(writes[write], writes[peer])) return false;
    }
  }
  return true;
}

bool valid_binding(const Gfn2InferencePublicationDevicePlan& plan,
                   const Gfn2InferencePublicationDeviceInput& input,
                   const Gfn2InferencePublicationDeviceResults& results,
                   const Gfn2InferencePublicationDeviceWorkspace& workspace,
                   const Gfn2InferencePublicationDeviceDiagnostics& diagnostics) noexcept {
  std::int64_t atom_coordinates = 0;
  std::int64_t point_coordinates = 0;
  if (!checked_multiply(plan.total_atoms, 3, atom_coordinates) ||
      !checked_multiply(plan.total_point_charges, 3, point_coordinates)) {
    return false;
  }
  const bool energy = property_requested(plan, GPUXTB_COMPUTE_ENERGY);
  const bool forces = property_requested(plan, GPUXTB_COMPUTE_FORCES);
  const bool charges = property_requested(plan, GPUXTB_COMPUTE_ATOMIC_CHARGES);
  const bool point_forces = property_requested(plan, GPUXTB_COMPUTE_POINT_CHARGE_FORCES);
  if (plan.abi_version != kGfn2InferencePublicationAbiVersion || plan.plan_token == 0u ||
      plan.requested_properties == 0u || (plan.requested_properties & ~kKnownProperties) != 0u ||
      plan.maximum_iterations == 0u ||
      plan.maximum_iterations >
          static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
      plan.batch_size <= 0 || plan.batch_size > std::numeric_limits<int>::max() ||
      plan.total_atoms < plan.batch_size || plan.total_point_charges < 0 ||
      !aligned(plan.atom_offsets, alignof(std::int64_t)) ||
      (plan.total_point_charges != 0 &&
       !aligned(plan.point_charge_offsets, alignof(std::int64_t))) ||
      plan.geometry_epoch.value_elements != 1 ||
      plan.geometry_epoch.plan_token != plan.plan_token ||
      !aligned(plan.geometry_epoch.value, alignof(std::uint64_t)) ||
      plan.generation_elements != plan.batch_size ||
      !aligned(plan.committed_generations, alignof(std::uint64_t)) ||
      input.plan_token != plan.plan_token || input.eligible_elements != plan.batch_size ||
      !aligned(input.eligible_mask, alignof(std::uint8_t)) ||
      input.scc_elements != plan.batch_size || !aligned(input.iterations, alignof(std::uint64_t)) ||
      !aligned(input.converged, alignof(std::uint8_t)) ||
      !aligned(input.system_statuses, alignof(gpuxtb_status_t)) ||
      !canonical_pointer(input.energies, energy ? plan.batch_size : 0) ||
      input.energy_elements != (energy ? plan.batch_size : 0) ||
      !canonical_pointer(input.qm_forces, forces ? atom_coordinates : 0) ||
      input.qm_force_elements != (forces ? atom_coordinates : 0) ||
      !canonical_pointer(input.atomic_charges, charges ? plan.total_atoms : 0) ||
      input.atomic_charge_elements != (charges ? plan.total_atoms : 0) ||
      !canonical_pointer(input.point_forces, point_forces ? point_coordinates : 0) ||
      input.point_force_elements != (point_forces ? point_coordinates : 0) ||
      input.terminal_system_error_elements != plan.batch_size ||
      !aligned(input.terminal_system_errors, alignof(std::uint32_t)) ||
      !aligned(input.terminal_plan_error, alignof(std::uint32_t)) ||
      input.execution_system_error_elements != plan.batch_size ||
      !aligned(input.execution_system_errors, alignof(std::uint32_t)) ||
      !aligned(input.execution_plan_error, alignof(std::uint32_t)) ||
      results.plan_token != plan.plan_token || results.batch_elements != plan.batch_size ||
      !canonical_pointer(results.energies, energy ? plan.batch_size : 0) ||
      results.energy_elements != (energy ? plan.batch_size : 0) ||
      !canonical_pointer(results.qm_forces, forces ? atom_coordinates : 0) ||
      results.qm_force_elements != (forces ? atom_coordinates : 0) ||
      !canonical_pointer(results.atomic_charges, charges ? plan.total_atoms : 0) ||
      results.atomic_charge_elements != (charges ? plan.total_atoms : 0) ||
      !canonical_pointer(results.point_forces, point_forces ? point_coordinates : 0) ||
      results.point_force_elements != (point_forces ? point_coordinates : 0) ||
      !aligned(results.iterations, alignof(std::int32_t)) ||
      !aligned(results.converged, alignof(std::uint8_t)) ||
      !aligned(results.system_statuses, alignof(gpuxtb_status_t)) ||
      workspace.plan_token != plan.plan_token || workspace.epoch_snapshot_elements != 1 ||
      !aligned(workspace.epoch_snapshot, alignof(std::uint64_t)) ||
      diagnostics.plan_token != plan.plan_token ||
      diagnostics.system_error_elements != plan.batch_size ||
      !aligned(diagnostics.system_errors, alignof(std::uint32_t)) ||
      diagnostics.plan_error_elements != 1 ||
      !aligned(diagnostics.plan_error, alignof(std::uint32_t))) {
    return false;
  }

  RangeList<24> reads;
  RangeList<12> writes;
  const bool ranges_valid =
      reads.add(plan.atom_offsets, plan.batch_size + 1) &&
      reads.add(plan.point_charge_offsets,
                plan.total_point_charges == 0 ? 0 : plan.batch_size + 1) &&
      reads.add(plan.geometry_epoch.value, 1) &&
      reads.add(plan.committed_generations, plan.batch_size) &&
      reads.add(input.eligible_mask, plan.batch_size) &&
      reads.add(input.iterations, plan.batch_size) && reads.add(input.converged, plan.batch_size) &&
      reads.add(input.system_statuses, plan.batch_size) &&
      reads.add(input.energies, energy ? plan.batch_size : 0) &&
      reads.add(input.qm_forces, forces ? atom_coordinates : 0) &&
      reads.add(input.atomic_charges, charges ? plan.total_atoms : 0) &&
      reads.add(input.point_forces, point_forces ? point_coordinates : 0) &&
      reads.add(input.terminal_system_errors, plan.batch_size) &&
      reads.add(input.terminal_plan_error, 1) &&
      reads.add(input.execution_system_errors, plan.batch_size) &&
      reads.add(input.execution_plan_error, 1) &&
      writes.add(results.energies, energy ? plan.batch_size : 0) &&
      writes.add(results.qm_forces, forces ? atom_coordinates : 0) &&
      writes.add(results.atomic_charges, charges ? plan.total_atoms : 0) &&
      writes.add(results.point_forces, point_forces ? point_coordinates : 0) &&
      writes.add(results.iterations, plan.batch_size) &&
      writes.add(results.converged, plan.batch_size) &&
      writes.add(results.system_statuses, plan.batch_size) &&
      writes.add(workspace.epoch_snapshot, 1) &&
      writes.add(diagnostics.system_errors, plan.batch_size) &&
      writes.add(diagnostics.plan_error, 1);
  return ranges_valid && writes_are_disjoint(reads, writes);
}

__device__ void record_plan_error(std::uint32_t* output, PlanError error) {
  atomicCAS(output, static_cast<std::uint32_t>(PlanError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ bool known_terminal_status(gpuxtb_status_t status) {
  return status == GPUXTB_STATUS_SUCCESS || status == GPUXTB_STATUS_INTERNAL_ERROR ||
         status == GPUXTB_STATUS_SCC_NOT_CONVERGED || status == GPUXTB_STATUS_EIGENSOLVER_FAILED;
}

__device__ double quiet_nan() {
  return __longlong_as_double(static_cast<long long>(0x7ff8000000000000ULL));
}

__global__ void publication_plan_preflight_kernel(
    Gfn2InferencePublicationDevicePlan plan, Gfn2InferencePublicationDeviceInput input,
    Gfn2InferencePublicationDeviceWorkspace workspace,
    Gfn2InferencePublicationDeviceDiagnostics diagnostics) {
  __shared__ int offsets_valid;
  if (threadIdx.x == 0) {
    offsets_valid = 1;
    const std::uint64_t epoch =
        atomicAdd(reinterpret_cast<unsigned long long*>(plan.geometry_epoch.value), 0ULL);
    *workspace.epoch_snapshot = epoch;
    if (epoch == 0u) record_plan_error(diagnostics.plan_error, PlanError::kInvalidEpoch);
    if (atomicAdd(const_cast<std::uint32_t*>(input.terminal_plan_error), 0u) != 0u) {
      record_plan_error(diagnostics.plan_error, PlanError::kTerminalClassicalPlanFailure);
    }
    if (atomicAdd(const_cast<std::uint32_t*>(input.execution_plan_error), 0u) != 0u) {
      record_plan_error(diagnostics.plan_error, PlanError::kEnergyForcePlanFailure);
    }
    if (plan.atom_offsets[0] != 0 || plan.atom_offsets[plan.batch_size] != plan.total_atoms) {
      offsets_valid = 0;
    }
    if (plan.total_point_charges != 0 &&
        (plan.point_charge_offsets[0] != 0 ||
         plan.point_charge_offsets[plan.batch_size] != plan.total_point_charges)) {
      offsets_valid = 0;
    }
  }
  __syncthreads();
  for (std::int64_t system = threadIdx.x; system < plan.batch_size; system += blockDim.x) {
    const std::int64_t atom_begin = plan.atom_offsets[system];
    const std::int64_t atom_end = plan.atom_offsets[system + 1];
    if (atom_begin < 0 || atom_begin >= atom_end || atom_end > plan.total_atoms) {
      atomicExch(&offsets_valid, 0);
    }
    if (plan.total_point_charges != 0) {
      const std::int64_t point_begin = plan.point_charge_offsets[system];
      const std::int64_t point_end = plan.point_charge_offsets[system + 1];
      if (point_begin < 0 || point_begin > point_end || point_end > plan.total_point_charges) {
        atomicExch(&offsets_valid, 0);
      }
    }
    if (input.eligible_mask[system] > 1u) {
      record_plan_error(diagnostics.plan_error, PlanError::kInvalidEligibilityMask);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && offsets_valid == 0) {
    record_plan_error(diagnostics.plan_error, PlanError::kInvalidOffsets);
  }
}

__global__ void publish_inference_results_kernel(
    Gfn2InferencePublicationDevicePlan plan, Gfn2InferencePublicationDeviceInput input,
    Gfn2InferencePublicationDeviceResults results,
    Gfn2InferencePublicationDeviceWorkspace workspace,
    Gfn2InferencePublicationDeviceDiagnostics diagnostics) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= plan.batch_size || atomicAdd(diagnostics.plan_error, 0u) != 0u) return;

  __shared__ int finite_results;
  __shared__ gpuxtb_status_t final_status;
  __shared__ std::uint32_t final_error;
  __shared__ std::int32_t public_iterations;
  if (threadIdx.x == 0) {
    finite_results = 1;
    final_status = input.system_statuses[system];
    final_error = static_cast<std::uint32_t>(SystemError::kSuccess);
    const std::uint64_t iterations = input.iterations[system];
    public_iterations = iterations > static_cast<std::uint64_t>(kInt32Maximum)
                            ? kInt32Maximum
                            : static_cast<std::int32_t>(iterations);
    const std::uint64_t epoch =
        atomicAdd(reinterpret_cast<unsigned long long*>(workspace.epoch_snapshot), 0ULL);
    if (input.eligible_mask[system] != 1u) {
      /* The member never entered SCC for this inference generation. */
      public_iterations = 0;
      final_status = GPUXTB_STATUS_INTERNAL_ERROR;
      final_error = static_cast<std::uint32_t>(SystemError::kIneligibleNumericalRefresh);
    } else if (plan.committed_generations[system] != epoch) {
      /* Do not leak a previous generation's terminal attempt count. */
      public_iterations = 0;
      final_status = GPUXTB_STATUS_INTERNAL_ERROR;
      final_error = static_cast<std::uint32_t>(SystemError::kStaleGeneration);
    } else if (iterations > plan.maximum_iterations ||
               iterations > static_cast<std::uint64_t>(kInt32Maximum)) {
      final_status = GPUXTB_STATUS_INTERNAL_ERROR;
      final_error = static_cast<std::uint32_t>(SystemError::kIterationOverflow);
    } else if (!known_terminal_status(final_status) || input.converged[system] > 1u ||
               (input.converged[system] == 1u && final_status != GPUXTB_STATUS_SUCCESS) ||
               (final_status == GPUXTB_STATUS_SUCCESS && input.converged[system] != 1u) ||
               (final_status == GPUXTB_STATUS_SCC_NOT_CONVERGED &&
                (input.converged[system] != 0u || iterations < plan.maximum_iterations))) {
      final_status = GPUXTB_STATUS_INTERNAL_ERROR;
      final_error = static_cast<std::uint32_t>(SystemError::kInvalidSccState);
    } else if (final_status == GPUXTB_STATUS_SUCCESS &&
               input.terminal_system_errors[system] != 0u) {
      final_status = GPUXTB_STATUS_INTERNAL_ERROR;
      final_error = static_cast<std::uint32_t>(SystemError::kTerminalClassicalFailure);
    } else if (final_status == GPUXTB_STATUS_SUCCESS &&
               input.execution_system_errors[system] != 0u) {
      final_status = GPUXTB_STATUS_INTERNAL_ERROR;
      final_error = static_cast<std::uint32_t>(SystemError::kEnergyForceFailure);
    }
  }
  __syncthreads();

  const bool success = final_status == GPUXTB_STATUS_SUCCESS;
  const bool energy = (plan.requested_properties & GPUXTB_COMPUTE_ENERGY) != 0u;
  const bool forces = (plan.requested_properties & GPUXTB_COMPUTE_FORCES) != 0u;
  const bool charges = (plan.requested_properties & GPUXTB_COMPUTE_ATOMIC_CHARGES) != 0u;
  const bool point_forces = (plan.requested_properties & GPUXTB_COMPUTE_POINT_CHARGE_FORCES) != 0u;
  const std::int64_t atom_begin = plan.atom_offsets[system];
  const std::int64_t atom_end = plan.atom_offsets[system + 1];
  const std::int64_t point_begin =
      plan.total_point_charges == 0 ? 0 : plan.point_charge_offsets[system];
  const std::int64_t point_end =
      plan.total_point_charges == 0 ? 0 : plan.point_charge_offsets[system + 1];

  if (success) {
    if (energy && threadIdx.x == 0 && !isfinite(input.energies[system])) {
      atomicExch(&finite_results, 0);
    }
    if (forces) {
      for (std::int64_t element = atom_begin * 3 + threadIdx.x; element < atom_end * 3;
           element += blockDim.x) {
        if (!isfinite(input.qm_forces[element])) atomicExch(&finite_results, 0);
      }
    }
    if (charges) {
      for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
        if (!isfinite(input.atomic_charges[atom])) atomicExch(&finite_results, 0);
      }
    }
    if (point_forces) {
      for (std::int64_t element = point_begin * 3 + threadIdx.x; element < point_end * 3;
           element += blockDim.x) {
        if (!isfinite(input.point_forces[element])) atomicExch(&finite_results, 0);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (final_status == GPUXTB_STATUS_SUCCESS && finite_results == 0) {
      final_status = GPUXTB_STATUS_INTERNAL_ERROR;
      final_error = static_cast<std::uint32_t>(SystemError::kNonfiniteResult);
    }
  }
  __syncthreads();

  const bool publish_success = final_status == GPUXTB_STATUS_SUCCESS;
  const double failed_value = quiet_nan();
  if (energy && threadIdx.x == 0) {
    results.energies[system] = publish_success ? input.energies[system] : failed_value;
  }
  if (forces) {
    for (std::int64_t element = atom_begin * 3 + threadIdx.x; element < atom_end * 3;
         element += blockDim.x) {
      results.qm_forces[element] = publish_success ? input.qm_forces[element] : failed_value;
    }
  }
  if (charges) {
    for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
      results.atomic_charges[atom] = publish_success ? input.atomic_charges[atom] : failed_value;
    }
  }
  if (point_forces) {
    for (std::int64_t element = point_begin * 3 + threadIdx.x; element < point_end * 3;
         element += blockDim.x) {
      results.point_forces[element] = publish_success ? input.point_forces[element] : failed_value;
    }
  }
  if (threadIdx.x == 0) {
    results.iterations[system] = public_iterations;
    results.converged[system] = publish_success ? 1u : 0u;
    results.system_statuses[system] = final_status;
    diagnostics.system_errors[system] = final_error;
  }
}

cudaError_t check_launch() noexcept { return cudaPeekAtLastError(); }

}  // namespace

cudaError_t publish_gfn2_inference_results_cuda(
    const Gfn2InferencePublicationDevicePlan& plan,
    const Gfn2InferencePublicationDeviceInput& input,
    const Gfn2InferencePublicationDeviceResults& results,
    const Gfn2InferencePublicationDeviceWorkspace& workspace,
    const Gfn2InferencePublicationDeviceDiagnostics& diagnostics, cudaStream_t stream) noexcept {
  if (!valid_binding(plan, input, results, workspace, diagnostics)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status =
      cudaMemsetAsync(diagnostics.system_errors, 0,
                      static_cast<std::size_t>(plan.batch_size) * sizeof(std::uint32_t), stream);
  if (status == cudaSuccess) {
    status = cudaMemsetAsync(diagnostics.plan_error, 0, sizeof(std::uint32_t), stream);
  }
  if (status != cudaSuccess) return status;

  publication_plan_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(plan, input, workspace,
                                                                        diagnostics);
  status = check_launch();
  if (status != cudaSuccess) return status;
  publish_inference_results_kernel<<<static_cast<unsigned int>(plan.batch_size), kThreadsPerBlock,
                                     0, stream>>>(plan, input, results, workspace, diagnostics);
  return check_launch();
}

}  // namespace gpuxtb::detail::cuda
