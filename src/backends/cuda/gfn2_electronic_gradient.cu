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

  const Gfn2ForceDeviceActivity integral_activity{workspace.hamiltonian_success_mask,
                                                  activity.system_statuses, activity.batch_elements,
                                                  activity.plan_token};
  const Gfn2IntegralForceDeviceInput integral_input{h0_input.positions,
                                                    h0_input.position_elements,
                                                    hamiltonian_output.overlap_adjoint,
                                                    hamiltonian_output.overlap_adjoint_elements,
                                                    hamiltonian_output.dipole_adjoint,
                                                    hamiltonian_output.dipole_adjoint_elements,
                                                    hamiltonian_output.quadrupole_adjoint,
                                                    hamiltonian_output.quadrupole_adjoint_elements,
                                                    integral_batch.plan_token};
  const Gfn2IntegralForceDeviceOutput integral_output{
      h0_output.gradients, h0_output.gradient_elements, integral_batch.plan_token};
  return add_gfn2_integral_gradient_cuda(
      integral_batch, integral_activity, integral_input, integral_output, integral_workspace,
      diagnostics.integral_system_errors, diagnostics.integral_device_error, stream);
}

}  // namespace xtbloom::detail::cuda
