#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_H0_FORCE_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN2_H0_FORCE_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_force_common.cuh"
#include "backends/cuda/gfn2_integrals.cuh"

namespace xtbloom::detail::cuda {

/* Per-system failure codes for the stationary H0/Pulay seed contraction. */
enum class Gfn2H0ForceDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidActiveMask = 1u,
  kInvalidOffsets = 2u,
  kInvalidShellMetadata = 3u,
  kInvalidH0Parameter = 4u,
  kNonfinitePosition = 5u,
  kCoordinateDifferenceOverflow = 6u,
  kCoincidentAtoms = 7u,
  kNonfiniteInput = 8u,
  kNonfiniteOutputSeed = 9u,
  kNonfiniteArithmetic = 10u,
};

/*
 * Restricted stationary electronic inputs. All matrices use the complete
 * row-major ragged packing described by Gfn2IntegralDeviceBatch.
 */
struct Gfn2H0ForceDeviceInput {
  const double* positions = nullptr;
  std::int64_t position_elements = 0;
  const double* coordination_numbers = nullptr;
  std::int64_t coordination_elements = 0;
  const double* overlap = nullptr;
  std::int64_t overlap_elements = 0;
  const double* density = nullptr;
  std::int64_t density_elements = 0;
  const double* energy_weighted_density = nullptr;
  std::int64_t energy_weighted_density_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Caller-owned accumulators. overlap_adjoint and coordination_adjoint are
 * intermediate dE/dS and dE/dCN values for the downstream overlap and
 * coordination VJPs. gradients stores dE/dR in Hartree/bohr, not forces;
 * the public force convention is force = -gradient.
 */
struct Gfn2H0ForceDeviceOutput {
  double* overlap_adjoint = nullptr;
  std::int64_t overlap_adjoint_elements = 0;
  double* coordination_adjoint = nullptr;
  std::int64_t coordination_adjoint_elements = 0;
  double* gradients = nullptr;
  std::int64_t gradient_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Unpublished transactional storage; counts are elements rather than bytes. */
struct Gfn2H0ForceDeviceWorkspace {
  double* overlap_adjoint_scratch = nullptr;
  std::int64_t overlap_adjoint_elements = 0;
  double* coordination_adjoint_scratch = nullptr;
  std::int64_t coordination_adjoint_elements = 0;
  double* gradient_scratch = nullptr;
  std::int64_t gradient_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2H0ForceDeviceInput>);
static_assert(std::is_standard_layout_v<Gfn2H0ForceDeviceInput>);
static_assert(std::is_trivially_copyable_v<Gfn2H0ForceDeviceOutput>);
static_assert(std::is_standard_layout_v<Gfn2H0ForceDeviceOutput>);
static_assert(std::is_trivially_copyable_v<Gfn2H0ForceDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2H0ForceDeviceWorkspace>);

/* Clear peer diagnostics and the sticky sequence-wide first-error scalar. */
cudaError_t reset_gfn2_h0_force_device_errors_cuda(std::int64_t batch_size,
                                                   std::uint32_t* system_errors,
                                                   std::uint32_t* device_error,
                                                   cudaStream_t stream = nullptr) noexcept;

/*
 * Accumulate the stationary restricted core-Hamiltonian/Pulay seed:
 *
 *   dE/dS  += P : (dH0/dS) - W,
 *   dE/dCN += P : (dH0/dCN),
 *   dE/dR  += P : (direct dH0/dR at fixed S and CN).
 *
 * The overlap and coordination adjoints are intentionally left explicit for
 * the analytic integral/CN reverse passes. Publication is atomic per system.
 * The launcher allocates and synchronizes nothing, enqueues only on stream,
 * and is compatible with CUDA Graph capture and replay.
 */
cudaError_t add_gfn2_h0_pulay_gradient_cuda(
    const Gfn2IntegralDeviceBatch& batch, const Gfn2H0DevicePlan& h0_plan,
    const Gfn2ForceDeviceActivity& activity, const Gfn2H0ForceDeviceInput& input,
    const Gfn2H0ForceDeviceOutput& output, const Gfn2H0ForceDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_H0_FORCE_CUH
