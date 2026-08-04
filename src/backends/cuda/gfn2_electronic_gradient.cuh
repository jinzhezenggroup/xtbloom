#ifndef GPUXTB_BACKENDS_CUDA_GFN2_ELECTRONIC_GRADIENT_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_ELECTRONIC_GRADIENT_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_h0_force.cuh"
#include "backends/cuda/gfn2_hamiltonian_force.cuh"
#include "backends/cuda/gfn2_integral_force.cuh"

namespace gpuxtb::detail::cuda {

/* A zero request is the energy-only fast path and binds no force storage. */
struct Gfn2ElectronicGradientRequest {
  std::uint8_t gradients_requested = 0u;
  std::uint64_t plan_token = 0u;
};

/* Per-stage diagnostics remain separate so one failed peer cannot stop healthy peers downstream. */
struct Gfn2ElectronicGradientDeviceDiagnostics {
  std::uint32_t* h0_system_errors = nullptr;
  std::uint32_t* h0_device_error = nullptr;
  std::uint32_t* hamiltonian_system_errors = nullptr;
  std::uint32_t* hamiltonian_device_error = nullptr;
  std::uint32_t* integral_system_errors = nullptr;
  std::uint32_t* integral_device_error = nullptr;
  std::int64_t batch_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Unpublished masks propagate successful members between stationary reverse stages. */
struct Gfn2ElectronicGradientDeviceWorkspace {
  std::uint8_t* h0_success_mask = nullptr;
  std::uint8_t* hamiltonian_success_mask = nullptr;
  std::int64_t batch_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2ElectronicGradientRequest>);
static_assert(std::is_standard_layout_v<Gfn2ElectronicGradientRequest>);
static_assert(std::is_trivially_copyable_v<Gfn2ElectronicGradientDeviceDiagnostics>);
static_assert(std::is_standard_layout_v<Gfn2ElectronicGradientDeviceDiagnostics>);
static_assert(std::is_trivially_copyable_v<Gfn2ElectronicGradientDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2ElectronicGradientDeviceWorkspace>);

/*
 * Compose the stationary electronic derivative without differentiating SCC:
 *
 *   H0/Pulay seeds -> SCC-Hamiltonian S/D/Q adjoints -> integral dE/dR.
 *
 * h0_output.overlap_adjoint and hamiltonian_output.overlap_adjoint must name
 * the same accumulator. Unrestricted callers pass physical total P/W to H0
 * and Hamiltonian charge response, plus physical P_alpha-P_beta and v_mag to
 * the Hamiltonian overlap response. The final atom-major value remains dE/dR;
 * the public force composer applies the minus sign. Coordination adjoints
 * remain exposed for the separate coordination-number reverse pass.
 *
 * A request with gradients_requested == 0 returns success before inspecting
 * any batch, input, output, diagnostic, or workspace descriptor. This makes
 * an energy-only plan independent of all force-only allocations. The active
 * path allocates and synchronizes nothing, uses only stream-ordered launches,
 * and is CUDA Graph capture compatible.
 */
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
    const Gfn2ElectronicGradientDeviceDiagnostics& diagnostics,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_ELECTRONIC_GRADIENT_CUH
