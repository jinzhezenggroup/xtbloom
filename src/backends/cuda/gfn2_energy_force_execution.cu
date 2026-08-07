#include <cuda_runtime.h>
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_energy_force_execution.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 128;
constexpr std::uint32_t kInactivePrimitiveError = std::numeric_limits<std::uint32_t>::max();

__device__ void record_execution_error(std::uint32_t* system_errors, std::int64_t system,
                                       std::uint32_t* device_error,
                                       Gfn2EnergyForceExecutionDeviceError error) {
  const std::uint32_t value = static_cast<std::uint32_t>(error);
  atomicCAS(system_errors + system, 0u, value);
  atomicCAS(device_error, 0u, value);
}

__global__ void preflight_geometry_transaction_kernel(std::int64_t batch_size,
                                                      Gfn2GeometryEpochConsumerDevice geometry,
                                                      std::uint32_t* plan_failure,
                                                      std::uint32_t* execution_device_error) {
  __shared__ int invalid;
  if (threadIdx.x == 0) {
    invalid = *geometry.epoch.value == 0u ? 1 : 0;
  }
  __syncthreads();
  for (std::int64_t system = threadIdx.x; system < batch_size; system += blockDim.x) {
    if (geometry.eligible_mask[system] > 1u) {
      atomicExch(&invalid, 1);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && invalid != 0) {
    const std::uint32_t value = static_cast<std::uint32_t>(
        Gfn2EnergyForceExecutionDeviceError::kInvalidGeometryTransaction);
    atomicCAS(plan_failure, 0u, value);
    atomicCAS(execution_device_error, 0u, value);
  }
}

__device__ bool geometry_transaction_allows(std::int64_t system,
                                            Gfn2GeometryEpochConsumerDevice geometry,
                                            int dynamic_epoch, std::uint32_t* execution_errors,
                                            std::uint32_t* execution_device_error) {
  if (dynamic_epoch == 0) {
    return true;
  }
  const std::uint8_t eligible = geometry.eligible_mask[system];
  if (eligible != 1u) {
    record_execution_error(execution_errors, system, execution_device_error,
                           Gfn2EnergyForceExecutionDeviceError::kIneligibleGeometry);
    return false;
  }
  if (geometry.committed_generations[system] != *geometry.epoch.value) {
    record_execution_error(execution_errors, system, execution_device_error,
                           Gfn2EnergyForceExecutionDeviceError::kStaleGeometry);
    return false;
  }
  return true;
}

__global__ void gate_after_energy_kernel(std::int64_t batch_size, Gfn2ForceDeviceActivity activity,
                                         Gfn2TotalEnergyDeviceSccState scc_state,
                                         Gfn2GeometryEpochConsumerDevice geometry,
                                         int dynamic_epoch, const std::uint32_t* energy_errors,
                                         const std::uint32_t* plan_failure,
                                         std::uint8_t* success_mask,
                                         std::uint32_t* execution_errors,
                                         std::uint32_t* execution_device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= batch_size) {
    return;
  }
  success_mask[system] = 0u;
  if (atomicAdd(const_cast<std::uint32_t*>(plan_failure), 0u) != 0u) {
    return;
  }
  if (!geometry_transaction_allows(system, geometry, dynamic_epoch, execution_errors,
                                   execution_device_error)) {
    return;
  }
  if (energy_errors[system] != 0u) {
    record_execution_error(execution_errors, system, execution_device_error,
                           Gfn2EnergyForceExecutionDeviceError::kTotalEnergyFailure);
    return;
  }
  const std::uint8_t requested = activity.requested_mask[system];
  if (requested > 1u) {
    record_execution_error(execution_errors, system, execution_device_error,
                           Gfn2EnergyForceExecutionDeviceError::kInvalidForceRequest);
    return;
  }
  if (requested == 0u) {
    return;
  }
  const std::uint8_t converged = scc_state.converged[system];
  if (converged == 1u && activity.system_statuses[system] == GPUXTB_STATUS_SUCCESS) {
    success_mask[system] = 1u;
  }
}

__global__ void publish_energy_only_kernel(
    std::int64_t batch_size, Gfn2TotalEnergyDeviceSccState scc_state,
    Gfn2GeometryEpochConsumerDevice geometry, int dynamic_epoch, const std::uint32_t* energy_errors,
    const double* staged_energy, double* public_energy, const std::uint32_t* plan_failure,
    std::uint32_t* execution_errors, std::uint32_t* execution_device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= batch_size || threadIdx.x != 0) {
    return;
  }
  if (atomicAdd(const_cast<std::uint32_t*>(plan_failure), 0u) != 0u) {
    return;
  }
  if (!geometry_transaction_allows(system, geometry, dynamic_epoch, execution_errors,
                                   execution_device_error)) {
    return;
  }
  if (energy_errors[system] != 0u) {
    record_execution_error(execution_errors, system, execution_device_error,
                           Gfn2EnergyForceExecutionDeviceError::kTotalEnergyFailure);
    return;
  }
  if (scc_state.converged[system] == 1u &&
      scc_state.system_statuses[system] == GPUXTB_STATUS_SUCCESS) {
    public_energy[system] = staged_energy[system];
  }
}

__global__ void zero_force_intermediates_kernel(
    Gfn2IntegralDeviceBatch integral_batch, Gfn2ExternalPointChargeDeviceBatch external_batch,
    Gfn2ForceCompositionDeviceBatch composition_batch, const std::uint8_t* selected_mask,
    Gfn2EnergyForceExecutionDeviceIntermediates intermediates) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (selected_mask[system] != 1u) {
    return;
  }
  const std::int64_t atom_begin = integral_batch.atom_offsets[system];
  const std::int64_t atom_end = integral_batch.atom_offsets[system + 1];
  const std::int64_t matrix_begin = integral_batch.matrix_offsets[system];
  const std::int64_t matrix_end = integral_batch.matrix_offsets[system + 1];
  const std::int64_t matrix_elements = integral_batch.total_matrix_elements;
  for (std::int64_t matrix = matrix_begin + threadIdx.x; matrix < matrix_end;
       matrix += blockDim.x) {
    intermediates.h0.overlap_adjoint[matrix] = 0.0;
    for (int component = 0; component < 3; ++component) {
      intermediates.hamiltonian.dipole_adjoint[component * matrix_elements + matrix] = 0.0;
    }
    for (int component = 0; component < 6; ++component) {
      intermediates.hamiltonian.quadrupole_adjoint[component * matrix_elements + matrix] = 0.0;
    }
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    intermediates.h0.coordination_adjoint[atom] = 0.0;
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      const std::int64_t element = atom * 3 + coordinate;
      intermediates.h0.gradients[element] = 0.0;
      intermediates.classical.forces[element] = 0.0;
      if ((composition_batch.enabled_components &
           static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kExplicitPointChargeForce)) !=
              0u &&
          external_batch.total_point_charges != 0) {
        intermediates.explicit_qm_forces[element] = 0.0;
      }
    }
  }
  if ((composition_batch.enabled_components &
       static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kExplicitPointChargeForce)) !=
          0u &&
      external_batch.total_point_charges != 0 && intermediates.explicit_point_forces != nullptr) {
    const std::int64_t point_begin = external_batch.point_charge_offsets[system];
    const std::int64_t point_end = external_batch.point_charge_offsets[system + 1];
    for (std::int64_t point = point_begin + threadIdx.x; point < point_end; point += blockDim.x) {
      for (int coordinate = 0; coordinate < 3; ++coordinate) {
        intermediates.explicit_point_forces[point * 3 + coordinate] = 0.0;
      }
    }
  }
}

__global__ void merge_electronic_errors_kernel(
    std::int64_t batch_size, const std::uint8_t* incoming_mask, const std::uint32_t* h0_errors,
    const std::uint32_t* hamiltonian_errors, const std::uint32_t* integral_errors,
    const std::uint32_t* plan_failure, std::uint8_t* outgoing_mask, std::uint32_t* execution_errors,
    std::uint32_t* execution_device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= batch_size) {
    return;
  }
  outgoing_mask[system] = 0u;
  if (incoming_mask[system] != 1u ||
      atomicAdd(const_cast<std::uint32_t*>(plan_failure), 0u) != 0u) {
    return;
  }
  if (h0_errors[system] != 0u || hamiltonian_errors[system] != 0u ||
      integral_errors[system] != 0u) {
    record_execution_error(execution_errors, system, execution_device_error,
                           Gfn2EnergyForceExecutionDeviceError::kElectronicGradientFailure);
    return;
  }
  outgoing_mask[system] = 1u;
}

__global__ void prepare_coordination_errors_kernel(std::int64_t batch_size,
                                                   const std::uint8_t* selected_mask,
                                                   std::uint32_t* coordination_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system < batch_size) {
    coordination_errors[system] = selected_mask[system] == 1u ? 0u : kInactivePrimitiveError;
  }
}

__global__ void merge_stage_errors_kernel(
    std::int64_t batch_size, const std::uint8_t* incoming_mask, const std::uint32_t* stage_errors,
    const std::uint32_t* plan_failure, std::uint8_t* outgoing_mask,
    Gfn2EnergyForceExecutionDeviceError mapped_error, std::uint32_t* execution_errors,
    std::uint32_t* execution_device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= batch_size) {
    return;
  }
  outgoing_mask[system] = 0u;
  if (incoming_mask[system] != 1u ||
      atomicAdd(const_cast<std::uint32_t*>(plan_failure), 0u) != 0u) {
    return;
  }
  if (stage_errors[system] != 0u) {
    record_execution_error(execution_errors, system, execution_device_error, mapped_error);
    return;
  }
  outgoing_mask[system] = 1u;
}

__global__ void copy_mask_kernel(std::int64_t batch_size, const std::uint8_t* input,
                                 std::uint8_t* output) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system < batch_size) {
    output[system] = input[system];
  }
}

/*
 * Step 5 sparse/dense CN VJP parity gate.  Every requested, healthy peer whose
 * H0 coordination VJP produced a bitwise-identical dense and sparse per-atom
 * gradient slice is accepted; the sparse result (in sparse_gradients) is then
 * published into the production gradients.  Any bitwise disagreement records a
 * coordination error for that peer so a sparse/dense regression can never
 * silently publish different forces.  The gate is one block per system and
 * follows the same per-peer transaction discipline as the other H0 stages.
 */
__global__ void gate_cn_vjp_parity_kernel(
    std::int64_t batch_size, const std::int64_t* atom_offsets,
    const std::uint8_t* incoming_mask, const std::uint32_t* plan_failure,
    const double* dense_gradients, const double* sparse_gradients, double* production_gradients,
    std::uint32_t* coordination_errors, std::uint32_t* execution_device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t atom_begin = atom_offsets[system];
  const std::int64_t atom_end = atom_offsets[system + 1];
  __shared__ unsigned int mismatched;
  if (threadIdx.x == 0) {
    mismatched = 0u;
  }
  __syncthreads();
  if (incoming_mask[system] == 1u &&
      atomicAdd(const_cast<std::uint32_t*>(plan_failure), 0u) == 0u) {
    for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
      for (int axis = 0; axis < 3; ++axis) {
        const std::int64_t index = atom * 3 + axis;
        const std::uint64_t dense_bits =
            *reinterpret_cast<const std::uint64_t*>(dense_gradients + index);
        const std::uint64_t sparse_bits =
            *reinterpret_cast<const std::uint64_t*>(sparse_gradients + index);
        if (dense_bits != sparse_bits) {
          atomicExch(&mismatched, 1u);
        }
      }
    }
  }
  __syncthreads();
  if (mismatched != 0u && threadIdx.x == 0) {
    atomicCAS(coordination_errors + system,
              static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess),
              static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSparseCoordinationMismatch));
    const std::uint32_t value = static_cast<std::uint32_t>(
        Gfn2EnergyForceExecutionDeviceError::kCoordinationGradientFailure);
    atomicCAS(execution_device_error, 0u, value);
    return;
  }
  if (mismatched == 0u && incoming_mask[system] == 1u &&
      atomicAdd(const_cast<std::uint32_t*>(plan_failure), 0u) == 0u) {
    for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
      for (int axis = 0; axis < 3; ++axis) {
        const std::int64_t index = atom * 3 + axis;
        production_gradients[index] = sparse_gradients[index];
      }
    }
  }
}

/* Seed the sparse CN VJP scratch with the current dense-coordination gradient
 * seed (the electronic gradient contributions before the CN VJP runs), so the
 * sparse consumer VJP accumulates onto exactly the same seed the dense VJP
 * uses. */
__global__ void seed_sparse_gradient_kernel(std::int64_t total_orbitals,
                                            const double* source, double* destination) {
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < total_orbitals) {
    destination[index] = source[index];
  }
}

/* Arm the one-element sparse VJP sequence. */
__global__ void arm_sparse_sequence_kernel(std::uint32_t* sequence_active) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *sequence_active = 1u;
  }
}

__global__ void merge_final_errors_kernel(std::int64_t batch_size,
                                          const std::uint8_t* incoming_mask,
                                          const std::uint32_t* composition_errors,
                                          const std::uint32_t* plan_failure,
                                          std::uint8_t* outgoing_mask,
                                          std::uint32_t* execution_errors,
                                          std::uint32_t* execution_device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= batch_size) {
    return;
  }
  outgoing_mask[system] = 0u;
  if (incoming_mask[system] != 1u ||
      atomicAdd(const_cast<std::uint32_t*>(plan_failure), 0u) != 0u) {
    return;
  }
  if (composition_errors[system] != 0u) {
    record_execution_error(execution_errors, system, execution_device_error,
                           Gfn2EnergyForceExecutionDeviceError::kForceCompositionFailure);
    return;
  }
  outgoing_mask[system] = 1u;
}

__global__ void record_sequence_plan_failure_kernel(
    const std::uint32_t* sequence_active, Gfn2EnergyForceExecutionDeviceError mapped_error,
    std::uint32_t* plan_failure, std::uint32_t* execution_device_error) {
  if (blockIdx.x == 0 && threadIdx.x == 0 &&
      atomicAdd(const_cast<std::uint32_t*>(sequence_active), 0u) == 0u) {
    const std::uint32_t value = static_cast<std::uint32_t>(mapped_error);
    atomicCAS(plan_failure, 0u, value);
    atomicCAS(execution_device_error, 0u, value);
  }
}

__global__ void record_composition_plan_failure_kernel(const std::uint32_t* composition_plan_error,
                                                       std::uint32_t* plan_failure,
                                                       std::uint32_t* execution_device_error) {
  if (blockIdx.x == 0 && threadIdx.x == 0 &&
      atomicAdd(const_cast<std::uint32_t*>(composition_plan_error), 0u) != 0u) {
    const std::uint32_t value =
        static_cast<std::uint32_t>(Gfn2EnergyForceExecutionDeviceError::kForceCompositionFailure);
    atomicCAS(plan_failure, 0u, value);
    atomicCAS(execution_device_error, 0u, value);
  }
}

__global__ void publish_execution_results_kernel(
    Gfn2IntegralDeviceBatch integral_batch, Gfn2ExternalPointChargeDeviceBatch external_batch,
    Gfn2ForceDeviceActivity activity, Gfn2TotalEnergyDeviceSccState scc_state,
    Gfn2GeometryEpochConsumerDevice geometry, int dynamic_epoch,
    const std::uint8_t* final_success_mask, const std::uint32_t* energy_errors,
    const std::uint32_t* execution_errors, const std::uint32_t* plan_failure,
    Gfn2EnergyForceExecutionDeviceIntermediates intermediates,
    Gfn2EnergyForceExecutionDeviceResults results) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (atomicAdd(const_cast<std::uint32_t*>(plan_failure), 0u) != 0u ||
      energy_errors[system] != 0u || scc_state.converged[system] != 1u ||
      scc_state.system_statuses[system] != GPUXTB_STATUS_SUCCESS) {
    return;
  }
  if (dynamic_epoch != 0 && (geometry.eligible_mask[system] != 1u ||
                             geometry.committed_generations[system] != *geometry.epoch.value)) {
    return;
  }
  const std::uint8_t requested = activity.requested_mask[system];
  if (requested == 0u) {
    if (threadIdx.x == 0) {
      results.energy.total_energy[system] = intermediates.energy.total_energy[system];
    }
    return;
  }
  if (requested != 1u || final_success_mask[system] != 1u || execution_errors[system] != 0u) {
    return;
  }
  if (threadIdx.x == 0) {
    results.energy.total_energy[system] = intermediates.energy.total_energy[system];
  }
  const std::int64_t atom_begin = integral_batch.atom_offsets[system];
  const std::int64_t atom_end = integral_batch.atom_offsets[system + 1];
  if (results.forces.qm_forces != nullptr) {
    for (std::int64_t coordinate = atom_begin * 3 + threadIdx.x; coordinate < atom_end * 3;
         coordinate += blockDim.x) {
      results.forces.qm_forces[coordinate] = intermediates.forces.qm_forces[coordinate];
    }
  }
  if (results.forces.point_forces != nullptr) {
    const std::int64_t point_begin = external_batch.point_charge_offsets[system];
    const std::int64_t point_end = external_batch.point_charge_offsets[system + 1];
    for (std::int64_t coordinate = point_begin * 3 + threadIdx.x; coordinate < point_end * 3;
         coordinate += blockDim.x) {
      results.forces.point_forces[coordinate] = intermediates.forces.point_forces[coordinate];
    }
  }
}

bool force_component_enabled(const Gfn2ForceCompositionDeviceBatch& batch,
                             Gfn2ForceCompositionComponent component) noexcept {
  return (batch.enabled_components & static_cast<std::uint32_t>(component)) != 0u;
}

bool mask_component_enabled(std::uint32_t mask, Gfn2SccPotentialComponent component) noexcept {
  return (mask & static_cast<std::uint32_t>(component)) != 0u;
}

bool total_energy_component_enabled(const Gfn2TotalEnergyDeviceBatch& batch,
                                    Gfn2TotalEnergyComponent component) noexcept {
  return (batch.enabled_components & static_cast<std::uint32_t>(component)) != 0u;
}

bool validate_restricted_physics_masks(const Gfn2EnergyForceExecutionDevicePlan& plan) noexcept {
  constexpr std::uint32_t kRequiredSccComponents =
      static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES2) |
      static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES3) |
      static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kAES2);
  constexpr std::uint32_t kRequiredClassicalComponents =
      static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kRepulsion) |
      static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kES2) |
      static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kAES2);
  constexpr std::uint32_t kRequiredCompositionComponents =
      static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kElectronicGradient) |
      static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kClassicalForce);

  const std::uint32_t potential_mask = plan.scc_potential_components;
  const std::uint32_t energy_mask = plan.scc_energy_components;
  if ((potential_mask & ~kGfn2SccPotentialAllComponents) != 0u ||
      (energy_mask & ~kGfn2SccClassicalAllComponents) != 0u || potential_mask != energy_mask ||
      plan.post_scc_potential_plan.enabled_components != potential_mask ||
      (potential_mask & kRequiredSccComponents) != kRequiredSccComponents) {
    return false;
  }

  const bool d4_two_body =
      mask_component_enabled(potential_mask, Gfn2SccPotentialComponent::kD4TwoBody);
  const bool d4_atm =
      total_energy_component_enabled(plan.total_energy_batch, Gfn2TotalEnergyComponent::kD4Atm);
  const bool explicit_points = plan.external_point_charge_batch.total_point_charges > 0;
  const bool scc_explicit_points =
      mask_component_enabled(potential_mask, Gfn2SccPotentialComponent::kExplicitPointCharge);

  std::uint32_t required_classical = kRequiredClassicalComponents;
  if (d4_two_body) {
    required_classical |= static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4TwoBody);
  }
  if (d4_atm) {
    required_classical |= static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4ATM);
  }

  std::uint32_t required_composition = kRequiredCompositionComponents;
  if (explicit_points) {
    required_composition |=
        static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kExplicitPointChargeForce);
  }
  return explicit_points == scc_explicit_points &&
         plan.classical_plan.enabled_components == required_classical &&
         plan.force_composition_batch.enabled_components == required_composition;
}

cudaError_t reset_execution_errors(std::int64_t batch_size,
                                   const Gfn2EnergyForceExecutionDeviceDiagnostics& diagnostics,
                                   cudaStream_t stream) noexcept {
  if (batch_size <= 0 || diagnostics.execution_system_errors == nullptr ||
      diagnostics.execution_device_error == nullptr) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(
      diagnostics.execution_system_errors, 0,
      static_cast<std::size_t>(batch_size) * sizeof(*diagnostics.execution_system_errors), stream);
  return status == cudaSuccess
             ? cudaMemsetAsync(diagnostics.execution_device_error, 0,
                               sizeof(*diagnostics.execution_device_error), stream)
             : status;
}

cudaError_t validate_force_binding(
    const Gfn2EnergyForceExecutionDevicePlan& plan,
    const Gfn2EnergyForceExecutionDeviceInput& input,
    const Gfn2EnergyForceExecutionDeviceResults& results,
    const Gfn2EnergyForceExecutionDeviceIntermediates& intermediates,
    const Gfn2EnergyForceExecutionDeviceWorkspace& workspace,
    const Gfn2EnergyForceExecutionDeviceDiagnostics& diagnostics) noexcept {
  const std::uint64_t token = plan.plan_token;
  const std::int64_t batch_size = plan.total_energy_batch.batch_size;
  const bool has_spin = input.hamiltonian.spin_density != nullptr;
  const bool explicit_pc = force_component_enabled(
      plan.force_composition_batch, Gfn2ForceCompositionComponent::kExplicitPointChargeForce);
  const bool point_output = results.forces.point_forces != nullptr;
  if (!validate_restricted_physics_masks(plan) || input.plan_token != token ||
      results.plan_token != token || intermediates.plan_token != token ||
      workspace.plan_token != token || diagnostics.plan_token != token ||
      plan.integral_batch.plan_token != token || plan.post_scc_potential_plan.plan_token != token ||
      plan.h0_plan.plan_token != token || plan.hamiltonian_batch.plan_token != token ||
      plan.coordination_batch.plan_token != token || plan.coordination_cache.plan_token != token ||
      plan.classical_plan.plan_token != token || plan.force_composition_batch.plan_token != token ||
      input.force_activity.plan_token != token || input.post_scc_potential.plan_token != token ||
      input.post_scc_potential.activity.requested_mask != input.force_activity.requested_mask ||
      input.post_scc_potential.activity.system_statuses != input.force_activity.system_statuses ||
      input.post_scc_potential.activity.batch_elements != batch_size ||
      input.post_scc_potential.activity.plan_token != token ||
      input.force_activity.batch_elements != batch_size ||
      input.force_activity.system_statuses != input.scc_state.system_statuses ||
      workspace.mask_elements < batch_size || diagnostics.batch_elements < batch_size ||
      workspace.energy_success_mask == nullptr || workspace.post_scc_success_mask == nullptr ||
      workspace.electronic_success_mask == nullptr ||
      workspace.coordination_success_mask == nullptr ||
      workspace.classical_success_mask == nullptr || workspace.external_success_mask == nullptr ||
      workspace.plan_failure == nullptr || workspace.plan_failure_elements < 1 ||
      intermediates.post_scc_potential.plan_token != token ||
      intermediates.post_scc_potential_intermediates.plan_token != token ||
      workspace.post_scc_potential.plan_token != token ||
      diagnostics.post_scc_potential.plan_token != token ||
      intermediates.h0.overlap_adjoint != intermediates.hamiltonian.overlap_adjoint ||
      input.h0.density != input.hamiltonian.density ||
      input.h0.density_elements != input.hamiltonian.density_elements ||
      has_spin != (input.hamiltonian.spin_shell_scalar_potentials != nullptr) ||
      (!has_spin && (input.hamiltonian.spin_density_elements != 0 ||
                     input.hamiltonian.spin_shell_scalar_elements != 0)) ||
      (has_spin &&
       (input.hamiltonian.spin_density_elements < plan.integral_batch.total_matrix_elements ||
        input.hamiltonian.spin_shell_scalar_elements < plan.integral_batch.total_shells)) ||
      intermediates.h0.gradients == nullptr || intermediates.h0.coordination_adjoint == nullptr ||
      intermediates.classical.forces == nullptr || results.forces.plan_token != token ||
      intermediates.forces.plan_token != token || results.forces.qm_forces == nullptr ||
      results.forces.qm_force_elements != plan.integral_batch.total_atoms * 3 ||
      intermediates.forces.qm_forces == nullptr ||
      intermediates.forces.qm_force_elements != plan.integral_batch.total_atoms * 3 ||
      (point_output && results.forces.point_force_elements !=
                           plan.external_point_charge_batch.total_point_charges * 3) ||
      (!point_output && results.forces.point_force_elements != 0) ||
      (point_output != (intermediates.forces.point_forces != nullptr)) ||
      intermediates.forces.point_force_elements != results.forces.point_force_elements ||
      plan.coordination_batch.batch_size != batch_size ||
      plan.integral_batch.batch_size != batch_size ||
      plan.hamiltonian_batch.batch_size != batch_size ||
      plan.classical_plan.batch_size != batch_size ||
      plan.force_composition_batch.batch_size != batch_size ||
      plan.force_composition_batch.total_atoms != plan.integral_batch.total_atoms ||
      plan.force_composition_batch.total_point_charges !=
          plan.external_point_charge_batch.total_point_charges ||
      plan.force_composition_batch.atom_offsets != plan.integral_batch.atom_offsets ||
      plan.coordination_batch.atom_offsets != plan.integral_batch.atom_offsets ||
      plan.classical_plan.atom_offsets != plan.integral_batch.atom_offsets ||
      plan.post_scc_potential_plan.potential_batch.batch_size != batch_size ||
      plan.post_scc_potential_plan.potential_batch.total_atoms != plan.integral_batch.total_atoms ||
      plan.post_scc_potential_plan.potential_batch.total_shells !=
          plan.integral_batch.total_shells ||
      plan.post_scc_potential_plan.potential_batch.atom_offsets !=
          plan.integral_batch.atom_offsets ||
      input.h0.coordination_numbers != plan.coordination_cache.coordination_numbers ||
      input.hamiltonian.shell_scalar_potentials != intermediates.post_scc_potential.shell_scalar ||
      input.hamiltonian.atomic_dipole_potentials !=
          intermediates.post_scc_potential.complete.dipole ||
      input.hamiltonian.atomic_quadrupole_potentials !=
          intermediates.post_scc_potential.complete.quadrupole ||
      input.classical.shell_charges != input.post_scc_potential.raw_shell_charges ||
      input.classical.atomic_charges != input.post_scc_potential.raw_atomic_charges ||
      input.classical.atomic_dipoles != input.post_scc_potential.raw_atomic_dipoles ||
      input.classical.atomic_quadrupoles != input.post_scc_potential.raw_atomic_quadrupoles ||
      input.external_shell_charges != input.post_scc_potential.raw_shell_charges ||
      input.h0.positions != input.classical.positions || plan.geometry_generation == 0u ||
      diagnostics.total_energy_system_errors == nullptr ||
      diagnostics.total_energy_device_error == nullptr ||
      diagnostics.coordination_system_errors == nullptr ||
      diagnostics.coordination_device_error == nullptr ||
      diagnostics.classical_system_errors == nullptr ||
      diagnostics.classical_device_error == nullptr ||
      diagnostics.force_composition_system_errors == nullptr ||
      diagnostics.force_composition_plan_error == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (explicit_pc &&
      (plan.external_point_charge_batch.plan_token != token ||
       plan.external_point_charge_batch.batch_size != batch_size ||
       plan.external_point_charge_batch.atom_offsets != plan.integral_batch.atom_offsets ||
       plan.force_composition_batch.point_charge_offsets !=
           plan.external_point_charge_batch.point_charge_offsets ||
       input.external_shell_charges == nullptr ||
       input.external_shell_elements != plan.external_point_charge_batch.total_shells ||
       intermediates.explicit_qm_forces == nullptr ||
       intermediates.explicit_qm_force_elements != plan.integral_batch.total_atoms * 3 ||
       (point_output && (intermediates.explicit_point_forces == nullptr ||
                         intermediates.explicit_point_force_elements !=
                             plan.external_point_charge_batch.total_point_charges * 3)) ||
       (!point_output && (intermediates.explicit_point_forces != nullptr ||
                          intermediates.explicit_point_force_elements != 0)) ||
       (plan.external_point_charge_batch.total_point_charges != 0 &&
        (diagnostics.external_system_errors == nullptr ||
         diagnostics.external_device_error == nullptr)))) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t check_launch() noexcept { return cudaPeekAtLastError(); }

}  // namespace

static cudaError_t execute_energy_force_impl(
    const Gfn2EnergyForceExecutionDevicePlan& plan,
    const Gfn2EnergyForceExecutionDeviceInput& input,
    const Gfn2EnergyForceExecutionDeviceResults& results,
    const Gfn2EnergyForceExecutionDeviceIntermediates& intermediates,
    const Gfn2EnergyForceExecutionDeviceWorkspace& workspace,
    const Gfn2EnergyForceExecutionDeviceDiagnostics& diagnostics,
    const Gfn2GeometryEpochConsumerDevice* geometry, cudaStream_t stream) noexcept {
  const std::int64_t batch_size = plan.total_energy_batch.batch_size;
  if (plan.compute_forces > 1u || plan.plan_token == 0u || batch_size <= 0 ||
      plan.total_energy_batch.plan_token != plan.plan_token ||
      input.total_energy.plan_token != plan.plan_token ||
      input.scc_state.plan_token != plan.plan_token || input.plan_token != plan.plan_token ||
      results.plan_token != plan.plan_token || intermediates.plan_token != plan.plan_token ||
      workspace.plan_token != plan.plan_token || results.energy.plan_token != plan.plan_token ||
      intermediates.energy.plan_token != plan.plan_token ||
      results.energy.total_energy == nullptr || results.energy.elements != batch_size ||
      intermediates.energy.total_energy == nullptr || intermediates.energy.elements != batch_size ||
      workspace.total_energy.plan_token != plan.plan_token ||
      workspace.total_energy.sequence_active == nullptr || workspace.total_energy.elements < 1 ||
      workspace.plan_failure == nullptr || workspace.plan_failure_elements < 1 ||
      diagnostics.plan_token != plan.plan_token || diagnostics.batch_elements < batch_size ||
      diagnostics.total_energy_system_errors == nullptr ||
      diagnostics.total_energy_device_error == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (geometry != nullptr &&
      (geometry->plan_token != plan.plan_token || geometry->epoch.plan_token != plan.plan_token ||
       geometry->epoch.value_elements != 1 || geometry->batch_elements != batch_size ||
       geometry->epoch.value == nullptr || geometry->committed_generations == nullptr ||
       geometry->eligible_mask == nullptr ||
       reinterpret_cast<std::uintptr_t>(geometry->epoch.value) % alignof(std::uint64_t) != 0u ||
       reinterpret_cast<std::uintptr_t>(geometry->committed_generations) % alignof(std::uint64_t) !=
           0u ||
       reinterpret_cast<std::uintptr_t>(geometry->eligible_mask) % alignof(std::uint8_t) != 0u)) {
    return cudaErrorInvalidValue;
  }
  if (plan.compute_forces == 1u) {
    const cudaError_t binding_status =
        validate_force_binding(plan, input, results, intermediates, workspace, diagnostics);
    if (binding_status != cudaSuccess) {
      return binding_status;
    }
  }

  cudaError_t status = reset_execution_errors(batch_size, diagnostics, stream);
  if (status == cudaSuccess) {
    status = cudaMemsetAsync(workspace.plan_failure, 0, sizeof(*workspace.plan_failure), stream);
  }
  if (status != cudaSuccess) {
    return status;
  }
  const Gfn2GeometryEpochConsumerDevice consumer =
      geometry == nullptr ? Gfn2GeometryEpochConsumerDevice{} : *geometry;
  if (geometry != nullptr) {
    preflight_geometry_transaction_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
        batch_size, consumer, workspace.plan_failure, diagnostics.execution_device_error);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
  }
  status =
      reset_gfn2_total_energy_device_errors_cuda(batch_size, diagnostics.total_energy_system_errors,
                                                 diagnostics.total_energy_device_error, stream);
  if (status == cudaSuccess) {
    status = compose_gfn2_total_energy_cuda(
        plan.total_energy_batch, input.total_energy, input.scc_state, intermediates.energy,
        workspace.total_energy, diagnostics.total_energy_system_errors,
        diagnostics.total_energy_device_error, stream);
  }
  if (status != cudaSuccess) {
    return status;
  }
  record_sequence_plan_failure_kernel<<<1, 1, 0, stream>>>(
      workspace.total_energy.sequence_active,
      Gfn2EnergyForceExecutionDeviceError::kTotalEnergyFailure, workspace.plan_failure,
      diagnostics.execution_device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  if (plan.compute_forces == 0u) {
    publish_energy_only_kernel<<<static_cast<unsigned int>(batch_size), 1, 0, stream>>>(
        batch_size, input.scc_state, consumer, geometry == nullptr ? 0 : 1,
        diagnostics.total_energy_system_errors, intermediates.energy.total_energy,
        results.energy.total_energy, workspace.plan_failure, diagnostics.execution_system_errors,
        diagnostics.execution_device_error);
    return check_launch();
  }

  const int blocks = static_cast<int>((batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock);
  gate_after_energy_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch_size, input.force_activity, input.scc_state, consumer, geometry == nullptr ? 0 : 1,
      diagnostics.total_energy_system_errors, workspace.plan_failure, workspace.energy_success_mask,
      diagnostics.execution_system_errors, diagnostics.execution_device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  zero_force_intermediates_kernel<<<static_cast<unsigned int>(batch_size), kThreadsPerBlock, 0,
                                    stream>>>(plan.integral_batch, plan.external_point_charge_batch,
                                              plan.force_composition_batch,
                                              workspace.energy_success_mask, intermediates);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }

  Gfn2PostSccPotentialDeviceInput post_scc_input = input.post_scc_potential;
  post_scc_input.activity = {workspace.energy_success_mask, input.force_activity.system_statuses,
                             batch_size, plan.plan_token};
  status = geometry == nullptr
               ? refresh_gfn2_post_scc_potentials_cuda(
                     plan.post_scc_potential_plan, post_scc_input, intermediates.post_scc_potential,
                     intermediates.post_scc_potential_intermediates, workspace.post_scc_potential,
                     diagnostics.post_scc_potential, stream)
               : refresh_gfn2_post_scc_potentials_cuda(
                     plan.post_scc_potential_plan, post_scc_input, intermediates.post_scc_potential,
                     intermediates.post_scc_potential_intermediates, workspace.post_scc_potential,
                     diagnostics.post_scc_potential, *geometry, stream);
  if (status != cudaSuccess) {
    return status;
  }
  record_sequence_plan_failure_kernel<<<1, 1, 0, stream>>>(
      workspace.post_scc_potential.sequence_active,
      Gfn2EnergyForceExecutionDeviceError::kPostSccPotentialFailure, workspace.plan_failure,
      diagnostics.execution_device_error);
  merge_stage_errors_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch_size, workspace.energy_success_mask, diagnostics.post_scc_potential.system_errors,
      workspace.plan_failure, workspace.post_scc_success_mask,
      Gfn2EnergyForceExecutionDeviceError::kPostSccPotentialFailure,
      diagnostics.execution_system_errors, diagnostics.execution_device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }

  const Gfn2ForceDeviceActivity electronic_activity{workspace.post_scc_success_mask,
                                                    input.force_activity.system_statuses,
                                                    batch_size, plan.plan_token};
  status = compose_gfn2_electronic_gradient_cuda(
      Gfn2ElectronicGradientRequest{1u, plan.plan_token}, plan.integral_batch, plan.h0_plan,
      plan.hamiltonian_batch, electronic_activity, input.h0, intermediates.h0, workspace.h0,
      input.hamiltonian, intermediates.hamiltonian, workspace.hamiltonian, workspace.integral,
      workspace.electronic, diagnostics.electronic, stream);
  if (status != cudaSuccess) {
    return status;
  }
  record_sequence_plan_failure_kernel<<<1, 1, 0, stream>>>(
      workspace.h0.sequence_active, Gfn2EnergyForceExecutionDeviceError::kElectronicGradientFailure,
      workspace.plan_failure, diagnostics.execution_device_error);
  record_sequence_plan_failure_kernel<<<1, 1, 0, stream>>>(
      workspace.hamiltonian.sequence_active,
      Gfn2EnergyForceExecutionDeviceError::kElectronicGradientFailure, workspace.plan_failure,
      diagnostics.execution_device_error);
  record_sequence_plan_failure_kernel<<<1, 1, 0, stream>>>(
      workspace.integral.sequence_active,
      Gfn2EnergyForceExecutionDeviceError::kElectronicGradientFailure, workspace.plan_failure,
      diagnostics.execution_device_error);
  merge_electronic_errors_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch_size, workspace.post_scc_success_mask, diagnostics.electronic.h0_system_errors,
      diagnostics.electronic.hamiltonian_system_errors,
      diagnostics.electronic.integral_system_errors, workspace.plan_failure,
      workspace.electronic_success_mask, diagnostics.execution_system_errors,
      diagnostics.execution_device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }

  status =
      reset_gfn2_geometry_device_errors_cuda(batch_size, diagnostics.coordination_system_errors,
                                             diagnostics.coordination_device_error, stream);
  if (status != cudaSuccess) {
    return status;
  }
  prepare_coordination_errors_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch_size, workspace.electronic_success_mask, diagnostics.coordination_system_errors);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  /* Step 5: when the committed sparse pair list is present, run the H0
   * coordination VJP on the sparse consumer path as well and parity-gate it
   * against the dense reference.  The dense VJP remains the production
   * reference; the sparse path must match bitwise, so a regression can never
   * silently change forces.  The gate is enabled only on the scalar-generation
   * path where the expected committed generation is a concrete host value;
   * the device-epoch path keeps the dense reference until a device-side epoch
   * version of the consumer VJP is wired. */
  const bool sparse_vjp_enabled =
      geometry == nullptr && plan.pairlist_committed.plan_token != 0u &&
      plan.pairlist_batch.plan_token != 0u;
  const std::int64_t gradient_elements = plan.integral_batch.total_atoms * 3;
  if (sparse_vjp_enabled) {
    arm_sparse_sequence_kernel<<<1, 1, 0, stream>>>(workspace.sparse_sequence_active);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
    seed_sparse_gradient_kernel<<<static_cast<unsigned int>(
                                      (gradient_elements + kThreadsPerBlock - 1) /
                                      kThreadsPerBlock),
                                  kThreadsPerBlock, 0, stream>>>(
        gradient_elements, intermediates.h0.gradients, workspace.sparse_gradient_scratch);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
    status = add_gfn2_pairlist_consumer_coordination_vjp_cuda(
        plan.pairlist_batch, plan.pairlist_committed, input.h0.positions,
        plan.coordination_batch.covalent_radii, plan.geometry_generation,
        intermediates.h0.coordination_adjoint, workspace.sparse_gradient_scratch,
        workspace.coordination.gradient_scratch, gradient_elements,
        workspace.sparse_sequence_active, diagnostics.coordination_system_errors,
        diagnostics.coordination_device_error, stream);
    if (status != cudaSuccess) {
      return status;
    }
  }
  status = geometry == nullptr
               ? add_gfn2_coordination_vjp_cuda(
                     plan.coordination_batch, plan.coordination_cache, plan.geometry_generation,
                     intermediates.h0.coordination_adjoint, intermediates.h0.gradients,
                     workspace.coordination, diagnostics.coordination_system_errors,
                     diagnostics.coordination_device_error, stream)
               : add_gfn2_coordination_vjp_cuda(
                     plan.coordination_batch, plan.coordination_cache, geometry->epoch,
                     intermediates.h0.coordination_adjoint, intermediates.h0.gradients,
                     workspace.coordination, diagnostics.coordination_system_errors,
                     diagnostics.coordination_device_error, stream);
  if (status != cudaSuccess) {
    return status;
  }
  if (sparse_vjp_enabled) {
    gate_cn_vjp_parity_kernel<<<static_cast<unsigned int>(batch_size), kThreadsPerBlock, 0,
                                stream>>>(
        batch_size, plan.integral_batch.atom_offsets, workspace.electronic_success_mask,
        workspace.plan_failure, intermediates.h0.gradients, workspace.sparse_gradient_scratch,
        intermediates.h0.gradients, diagnostics.coordination_system_errors,
        diagnostics.execution_device_error);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
  }
  record_sequence_plan_failure_kernel<<<1, 1, 0, stream>>>(
      workspace.coordination.sequence_active,
      Gfn2EnergyForceExecutionDeviceError::kCoordinationGradientFailure, workspace.plan_failure,
      diagnostics.execution_device_error);
  merge_stage_errors_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch_size, workspace.electronic_success_mask, diagnostics.coordination_system_errors,
      workspace.plan_failure, workspace.coordination_success_mask,
      Gfn2EnergyForceExecutionDeviceError::kCoordinationGradientFailure,
      diagnostics.execution_system_errors, diagnostics.execution_device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }

  status = reset_gfn2_classical_force_device_errors_cuda(
      batch_size, diagnostics.classical_system_errors, diagnostics.classical_device_error, stream);
  if (status != cudaSuccess) {
    return status;
  }
  const Gfn2ForceDeviceActivity classical_activity{workspace.coordination_success_mask,
                                                   input.force_activity.system_statuses, batch_size,
                                                   plan.plan_token};
  status =
      geometry == nullptr
          ? add_gfn2_classical_forces_cuda(plan.classical_plan, classical_activity, input.classical,
                                           intermediates.classical, workspace.classical,
                                           diagnostics.classical_system_errors,
                                           diagnostics.classical_device_error, stream)
          : add_gfn2_classical_forces_cuda(plan.classical_plan, classical_activity, input.classical,
                                           intermediates.classical, workspace.classical,
                                           geometry->epoch, diagnostics.classical_system_errors,
                                           diagnostics.classical_device_error, stream);
  if (status != cudaSuccess) {
    return status;
  }
  record_sequence_plan_failure_kernel<<<1, 1, 0, stream>>>(
      workspace.classical.sequence_active,
      Gfn2EnergyForceExecutionDeviceError::kClassicalForceFailure, workspace.plan_failure,
      diagnostics.execution_device_error);
  merge_stage_errors_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch_size, workspace.coordination_success_mask, diagnostics.classical_system_errors,
      workspace.plan_failure, workspace.classical_success_mask,
      Gfn2EnergyForceExecutionDeviceError::kClassicalForceFailure,
      diagnostics.execution_system_errors, diagnostics.execution_device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }

  const bool explicit_pc = force_component_enabled(
      plan.force_composition_batch, Gfn2ForceCompositionComponent::kExplicitPointChargeForce);
  if (explicit_pc && plan.external_point_charge_batch.total_point_charges != 0) {
    status = reset_gfn2_external_point_charge_force_errors_cuda(
        batch_size, diagnostics.external_system_errors, diagnostics.external_device_error, stream);
    if (status != cudaSuccess) {
      return status;
    }
    const Gfn2ForceDeviceActivity external_activity{workspace.classical_success_mask,
                                                    input.force_activity.system_statuses,
                                                    batch_size, plan.plan_token};
    status = add_gfn2_external_point_charge_gated_forces_cuda(
        plan.external_point_charge_batch, external_activity, input.external_shell_charges,
        intermediates.explicit_qm_forces, intermediates.explicit_point_forces,
        workspace.external_point_charge, diagnostics.external_system_errors,
        diagnostics.external_device_error, stream);
    if (status != cudaSuccess) {
      return status;
    }
    record_sequence_plan_failure_kernel<<<1, 1, 0, stream>>>(
        workspace.external_point_charge.sequence_active,
        Gfn2EnergyForceExecutionDeviceError::kExternalPointChargeForceFailure,
        workspace.plan_failure, diagnostics.execution_device_error);
    merge_stage_errors_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        batch_size, workspace.classical_success_mask, diagnostics.external_system_errors,
        workspace.plan_failure, workspace.external_success_mask,
        Gfn2EnergyForceExecutionDeviceError::kExternalPointChargeForceFailure,
        diagnostics.execution_system_errors, diagnostics.execution_device_error);
  } else {
    copy_mask_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        batch_size, workspace.classical_success_mask, workspace.external_success_mask);
  }
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }

  status = reset_gfn2_force_composition_device_errors_cuda(
      batch_size, diagnostics.force_composition_system_errors,
      diagnostics.force_composition_plan_error, stream);
  if (status != cudaSuccess) {
    return status;
  }
  const Gfn2ForceDeviceActivity composition_activity{workspace.external_success_mask,
                                                     input.force_activity.system_statuses,
                                                     batch_size, plan.plan_token};
  const Gfn2ForceCompositionDeviceInput composition_input{
      intermediates.h0.gradients,
      intermediates.h0.gradient_elements,
      intermediates.classical.forces,
      intermediates.classical.force_elements,
      explicit_pc && plan.external_point_charge_batch.total_point_charges != 0
          ? intermediates.explicit_qm_forces
          : nullptr,
      explicit_pc && plan.external_point_charge_batch.total_point_charges != 0
          ? intermediates.explicit_qm_force_elements
          : 0,
      explicit_pc && plan.external_point_charge_batch.total_point_charges != 0
          ? intermediates.explicit_point_forces
          : nullptr,
      explicit_pc && plan.external_point_charge_batch.total_point_charges != 0
          ? intermediates.explicit_point_force_elements
          : 0,
      plan.plan_token};
  status = compose_gfn2_forces_cuda(
      plan.force_composition_batch, composition_activity, composition_input, intermediates.forces,
      workspace.force_composition, diagnostics.force_composition_system_errors,
      diagnostics.force_composition_plan_error, stream);
  if (status != cudaSuccess) {
    return status;
  }
  record_composition_plan_failure_kernel<<<1, 1, 0, stream>>>(
      diagnostics.force_composition_plan_error, workspace.plan_failure,
      diagnostics.execution_device_error);
  merge_final_errors_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch_size, workspace.external_success_mask, diagnostics.force_composition_system_errors,
      workspace.plan_failure, workspace.energy_success_mask, diagnostics.execution_system_errors,
      diagnostics.execution_device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_execution_results_kernel<<<static_cast<unsigned int>(batch_size), kThreadsPerBlock, 0,
                                     stream>>>(
      plan.integral_batch, plan.external_point_charge_batch, input.force_activity, input.scc_state,
      consumer, geometry == nullptr ? 0 : 1, workspace.energy_success_mask,
      diagnostics.total_energy_system_errors, diagnostics.execution_system_errors,
      workspace.plan_failure, intermediates, results);
  return check_launch();
}

cudaError_t execute_gfn2_energy_force_cuda(
    const Gfn2EnergyForceExecutionDevicePlan& plan,
    const Gfn2EnergyForceExecutionDeviceInput& input,
    const Gfn2EnergyForceExecutionDeviceResults& results,
    const Gfn2EnergyForceExecutionDeviceIntermediates& intermediates,
    const Gfn2EnergyForceExecutionDeviceWorkspace& workspace,
    const Gfn2EnergyForceExecutionDeviceDiagnostics& diagnostics, cudaStream_t stream) noexcept {
  return execute_energy_force_impl(plan, input, results, intermediates, workspace, diagnostics,
                                   nullptr, stream);
}

cudaError_t execute_gfn2_energy_force_cuda(
    const Gfn2EnergyForceExecutionDevicePlan& plan,
    const Gfn2EnergyForceExecutionDeviceInput& input,
    const Gfn2EnergyForceExecutionDeviceResults& results,
    const Gfn2EnergyForceExecutionDeviceIntermediates& intermediates,
    const Gfn2EnergyForceExecutionDeviceWorkspace& workspace,
    const Gfn2EnergyForceExecutionDeviceDiagnostics& diagnostics,
    const Gfn2GeometryEpochConsumerDevice& geometry, cudaStream_t stream) noexcept {
  return execute_energy_force_impl(plan, input, results, intermediates, workspace, diagnostics,
                                   &geometry, stream);
}

}  // namespace gpuxtb::detail::cuda
