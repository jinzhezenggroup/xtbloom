#include <cuda_runtime.h>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cstdint>

#include "backends/cuda/gfn2_electronic_gradient.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 128;

__global__ void propagate_success_mask_kernel(const Gfn2ForceDeviceActivity activity,
                                              const std::uint32_t* upstream_errors,
                                              std::uint8_t* success_mask) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= activity.batch_elements) {
    return;
  }
  const std::uint8_t requested = activity.requested_mask[system];
  success_mask[system] = requested == 1u &&
                                 activity.system_statuses[system] == XTBLOOM_STATUS_SUCCESS &&
                                 upstream_errors[system] == 0u
                             ? 1u
                             : 0u;
}

/*
 * The CPU periodic force composer accumulates the stationary Hamiltonian
 * adjoint into a zeroed overlap buffer and subtracts W only afterwards.  The
 * molecular CUDA path historically seeded H0 with -W, which is algebraically
 * equivalent but not bitwise equivalent for large cancelling values.  Native
 * periodic requests clear that seed before the Hamiltonian reverse so the
 * device follows the CPU operation order exactly.
 */
__global__ void clear_native_periodic_overlap_seed_kernel(const Gfn2IntegralDeviceBatch batch,
                                                          const std::uint8_t* active_mask,
                                                          double* overlap_adjoint) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (active_mask[system] != 1u) {
    return;
  }
  const std::int64_t begin = batch.matrix_offsets[system];
  const std::int64_t end = batch.matrix_offsets[system + 1];
  for (std::int64_t matrix = begin + threadIdx.x; matrix < end; matrix += blockDim.x) {
    overlap_adjoint[matrix] = 0.0;
  }
}

/* Apply the deferred -W step after the stationary Hamiltonian adjoint. */
__global__ void subtract_native_periodic_weighted_density_kernel(
    const Gfn2IntegralDeviceBatch batch, const std::uint8_t* active_mask,
    const double* energy_weighted_density, double* overlap_adjoint, std::uint32_t* system_errors,
    std::uint32_t* device_error, std::uint32_t* sequence_active) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (active_mask[system] != 1u) {
    return;
  }
  const std::int64_t begin = batch.matrix_offsets[system];
  const std::int64_t end = batch.matrix_offsets[system + 1];
  for (std::int64_t matrix = begin + threadIdx.x; matrix < end; matrix += blockDim.x) {
    const double updated = overlap_adjoint[matrix] - energy_weighted_density[matrix];
    if (!isfinite(updated)) {
      constexpr std::uint32_t kNonfiniteArithmetic =
          static_cast<std::uint32_t>(Gfn2HamiltonianForceDeviceError::kNonfiniteArithmetic);
      atomicCAS(system_errors + system, 0u, kNonfiniteArithmetic);
      atomicCAS(device_error, 0u, kNonfiniteArithmetic);
      atomicExch(sequence_active, 0u);
      continue;
    }
    overlap_adjoint[matrix] = updated;
  }
}

cudaError_t validate_active_composition(
    const Gfn2ElectronicGradientRequest& request, const Gfn2IntegralDeviceBatch& integral_batch,
    const Gfn2HamiltonianDeviceBatch& hamiltonian_batch, const Gfn2ForceDeviceActivity& activity,
    const Gfn2H0ForceDeviceInput& h0_input, const Gfn2H0ForceDeviceOutput& h0_output,
    const Gfn2HamiltonianForceDeviceInput& hamiltonian_input,
    const Gfn2HamiltonianForceDeviceOutput& hamiltonian_output,
    const Gfn2ElectronicGradientDeviceWorkspace& workspace,
    const Gfn2ElectronicGradientDeviceDiagnostics& diagnostics) noexcept {
  const std::uint64_t token = integral_batch.plan_token;
  if (request.plan_token != token || hamiltonian_batch.plan_token != token ||
      hamiltonian_batch.model != integral_batch.model || activity.plan_token != token ||
      h0_input.plan_token != token || h0_output.plan_token != token ||
      hamiltonian_input.plan_token != token || hamiltonian_output.plan_token != token ||
      workspace.plan_token != token || diagnostics.plan_token != token ||
      integral_batch.batch_size != hamiltonian_batch.batch_size ||
      integral_batch.total_atoms != hamiltonian_batch.total_atoms ||
      integral_batch.total_shells != hamiltonian_batch.total_shells ||
      integral_batch.total_orbitals != hamiltonian_batch.total_orbitals ||
      integral_batch.total_matrix_elements != hamiltonian_batch.total_matrix_elements ||
      integral_batch.linear_tiles_per_system <= 0 ||
      integral_batch.linear_tiles_per_system > kGfn2IntegralLinearBlockBudget ||
      activity.batch_elements != integral_batch.batch_size ||
      workspace.batch_elements < integral_batch.batch_size ||
      diagnostics.batch_elements < integral_batch.batch_size ||
      activity.requested_mask == nullptr || activity.system_statuses == nullptr ||
      workspace.h0_success_mask == nullptr || workspace.hamiltonian_success_mask == nullptr ||
      workspace.h0_success_mask == workspace.hamiltonian_success_mask ||
      diagnostics.h0_system_errors == nullptr || diagnostics.h0_device_error == nullptr ||
      diagnostics.hamiltonian_system_errors == nullptr ||
      diagnostics.hamiltonian_device_error == nullptr ||
      diagnostics.integral_system_errors == nullptr ||
      diagnostics.integral_device_error == nullptr ||
      h0_output.overlap_adjoint != hamiltonian_output.overlap_adjoint ||
      h0_input.density != hamiltonian_input.density) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t check_launch() noexcept { return cudaPeekAtLastError(); }

}  // namespace

cudaError_t compose_gfn2_electronic_gradient_cuda(
    const Gfn2ElectronicGradientRequest& request, const Gfn2IntegralDeviceBatch& integral_batch,
    const Gfn2H0DevicePlan& h0_plan, const Gfn2HamiltonianDeviceBatch& hamiltonian_batch,
    const Gfn2ForceDeviceActivity& activity, const Gfn2H0ForceDeviceInput& h0_input,
    const Gfn2H0ForceDeviceOutput& h0_output, const Gfn2H0ForceDeviceWorkspace& h0_workspace,
    const Gfn2HamiltonianForceDeviceInput& hamiltonian_input,
    const Gfn2HamiltonianForceDeviceOutput& hamiltonian_output,
    const Gfn2HamiltonianForceDeviceWorkspace& hamiltonian_workspace,
    const Gfn2IntegralForceDeviceWorkspace& integral_workspace,
    const Gfn2ElectronicGradientDeviceWorkspace& workspace,
    const Gfn2ElectronicGradientDeviceDiagnostics& diagnostics, cudaStream_t stream) noexcept {
  if (request.gradients_requested == 0u) {
    return cudaSuccess;
  }
  if (request.gradients_requested != 1u) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = validate_active_composition(request, integral_batch, hamiltonian_batch,
                                                   activity, h0_input, h0_output, hamiltonian_input,
                                                   hamiltonian_output, workspace, diagnostics);
  if (status != cudaSuccess) {
    return status;
  }

  status = reset_gfn2_h0_force_device_errors_cuda(
      integral_batch.batch_size, diagnostics.h0_system_errors, diagnostics.h0_device_error, stream);
  if (status == cudaSuccess) {
    status = reset_gfn2_hamiltonian_force_device_errors_cuda(
        integral_batch.batch_size, diagnostics.hamiltonian_system_errors,
        diagnostics.hamiltonian_device_error, stream);
  }
  if (status == cudaSuccess) {
    status = reset_gfn2_integral_force_device_errors_cuda(
        integral_batch.batch_size, diagnostics.integral_system_errors,
        diagnostics.integral_device_error, stream);
  }
  if (status != cudaSuccess) {
    return status;
  }

  status = add_gfn2_h0_pulay_gradient_cuda(integral_batch, h0_plan, activity, h0_input, h0_output,
                                           h0_workspace, diagnostics.h0_system_errors,
                                           diagnostics.h0_device_error, stream);
  if (status != cudaSuccess) {
    return status;
  }
  const int blocks =
      static_cast<int>((integral_batch.batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock);
  propagate_success_mask_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      activity, diagnostics.h0_system_errors, workspace.h0_success_mask);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }

  const bool native_periodic = h0_input.native_periodic.plan_token != 0u;
  if (native_periodic) {
    /* H0's generic seed includes -W for the molecular route.  Native
     * periodic H0 intentionally defers that subtraction until after the
     * stationary Hamiltonian reverse, matching the CPU accumulation order. */
    clear_native_periodic_overlap_seed_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        integral_batch, workspace.h0_success_mask, h0_output.overlap_adjoint);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
  }

  const Gfn2ForceDeviceActivity hamiltonian_activity{workspace.h0_success_mask,
                                                     activity.system_statuses,
                                                     activity.batch_elements, activity.plan_token};
  status = add_gfn2_hamiltonian_integral_adjoints_cuda(
      hamiltonian_batch, hamiltonian_activity, hamiltonian_input, hamiltonian_output,
      hamiltonian_workspace, diagnostics.hamiltonian_system_errors,
      diagnostics.hamiltonian_device_error, stream);
  if (status != cudaSuccess) {
    return status;
  }
  propagate_success_mask_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      hamiltonian_activity, diagnostics.hamiltonian_system_errors,
      workspace.hamiltonian_success_mask);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }

  if (native_periodic) {
    subtract_native_periodic_weighted_density_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        integral_batch, workspace.hamiltonian_success_mask, h0_input.energy_weighted_density,
        hamiltonian_output.overlap_adjoint, diagnostics.hamiltonian_system_errors,
        diagnostics.hamiltonian_device_error, hamiltonian_workspace.sequence_active);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
    /* A non-finite deferred subtraction is a peer-local Hamiltonian error;
     * refresh the mask before allowing the integral reverse to consume it. */
    propagate_success_mask_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        hamiltonian_activity, diagnostics.hamiltonian_system_errors,
        workspace.hamiltonian_success_mask);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
  }

  const Gfn2ForceDeviceActivity integral_activity{workspace.hamiltonian_success_mask,
                                                  activity.system_statuses, activity.batch_elements,
                                                  activity.plan_token};
  Gfn2IntegralForceDeviceInput integral_input{h0_input.positions,
                                              h0_input.position_elements,
                                              hamiltonian_output.overlap_adjoint,
                                              hamiltonian_output.overlap_adjoint_elements,
                                              hamiltonian_output.dipole_adjoint,
                                              hamiltonian_output.dipole_adjoint_elements,
                                              hamiltonian_output.quadrupole_adjoint,
                                              hamiltonian_output.quadrupole_adjoint_elements,
                                              integral_batch.plan_token};
  integral_input.native_periodic = h0_input.native_periodic;
  integral_input.density = h0_input.density;
  integral_input.density_elements = h0_input.density_elements;
  integral_input.coordination_numbers = h0_input.coordination_numbers;
  integral_input.coordination_elements = h0_input.coordination_elements;
  integral_input.h0_plan = h0_plan;
  const Gfn2IntegralForceDeviceOutput integral_output{
      h0_output.gradients, h0_output.gradient_elements, integral_batch.plan_token};
  status = add_gfn2_integral_gradient_cuda(
      integral_batch, integral_activity, integral_input, integral_output, integral_workspace,
      diagnostics.integral_system_errors, diagnostics.integral_device_error, stream);
  return status;
}

}  // namespace xtbloom::detail::cuda
