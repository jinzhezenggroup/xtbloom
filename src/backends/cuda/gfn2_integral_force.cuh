#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_INTEGRAL_FORCE_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN2_INTEGRAL_FORCE_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_force_common.cuh"
#include "backends/cuda/gfn2_integrals.cuh"

namespace xtbloom::detail::cuda {

/* Complete directed S/D/Q adjoints produced by the stationary force stages. */
struct Gfn2IntegralForceDeviceInput {
  const double* positions = nullptr;
  std::int64_t position_elements = 0;
  const double* overlap_adjoint = nullptr;
  std::int64_t overlap_adjoint_elements = 0;
  const double* dipole_adjoint = nullptr;
  std::int64_t dipole_adjoint_elements = 0;
  const double* quadrupole_adjoint = nullptr;
  std::int64_t quadrupole_adjoint_elements = 0;
  std::uint64_t plan_token = 0u;
};

struct Gfn2IntegralForceDeviceOutput {
  /* Accumulated dE/dR in atom-major xyz order; public force = -gradient. */
  double* gradients = nullptr;
  std::int64_t gradient_elements = 0;
  std::uint64_t plan_token = 0u;
};

struct Gfn2IntegralForceDeviceWorkspace {
  double* gradient_scratch = nullptr;
  std::int64_t gradient_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2IntegralForceDeviceInput>);
static_assert(std::is_standard_layout_v<Gfn2IntegralForceDeviceInput>);
static_assert(std::is_trivially_copyable_v<Gfn2IntegralForceDeviceOutput>);
static_assert(std::is_standard_layout_v<Gfn2IntegralForceDeviceOutput>);
static_assert(std::is_trivially_copyable_v<Gfn2IntegralForceDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2IntegralForceDeviceWorkspace>);

cudaError_t reset_gfn2_integral_force_device_errors_cuda(std::int64_t batch_size,
                                                         std::uint32_t* system_errors,
                                                         std::uint32_t* device_error,
                                                         cudaStream_t stream = nullptr) noexcept;

/*
 * Apply the analytic reverse pass of the overlap and ket-origin multipole
 * integral evaluator without materializing an integral-derivative tensor.
 * Reverse D/Q origin translations are differentiated explicitly, matching
 * add_overlap_gradient_cpu + add_multipole_gradient_cpu.
 *
 * Each eligible system publishes its complete accumulated gradient only after
 * all shell pairs succeed. The launcher allocates and synchronizes nothing,
 * enqueues only on stream, and is CUDA Graph capture compatible.
 */
cudaError_t add_gfn2_integral_gradient_cuda(
    const Gfn2IntegralDeviceBatch& batch, const Gfn2ForceDeviceActivity& activity,
    const Gfn2IntegralForceDeviceInput& input, const Gfn2IntegralForceDeviceOutput& output,
    const Gfn2IntegralForceDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_INTEGRAL_FORCE_CUH
