#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_INTEGRALS_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN2_INTEGRALS_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/common/xtb_model.hpp"

namespace xtbloom::detail::cuda {

inline constexpr std::int64_t kGfn2IntegralDipoleComponents = 3;
inline constexpr std::int64_t kGfn2IntegralQuadrupoleComponents = 6;
/* Keep linear staging grids bounded independently of ragged batch size. */
inline constexpr std::int64_t kGfn2IntegralLinearBlockBudget = 512;

/* Compact task identity within one fixed topology. The precomputed shell
 * endpoints remove repeated row-major division in the generic kernels, while
 * local_pair preserves the dense H0 scale index and canonical ordering. */
struct Gfn2IntegralShellPairTask {
  std::uint32_t system = 0u;
  std::uint32_t local_pair = 0u;
  std::uint32_t bra_shell = 0u;
  std::uint32_t ket_shell = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2IntegralShellPairTask>);
static_assert(std::is_standard_layout_v<Gfn2IntegralShellPairTask>);
static_assert(sizeof(Gfn2IntegralShellPairTask) == 16u);

struct Gfn2IntegralDeviceTaskDomains {
  std::int64_t forward_generic_task_count = 0;
  std::int64_t forward_ss_task_count = 0;
  std::int64_t h0_generic_task_count = 0;
  std::int64_t h0_ss_task_count = 0;
  std::int64_t force_generic_task_count = 0;
  std::int64_t force_ss_task_count = 0;
  const Gfn2IntegralShellPairTask* forward_generic_tasks = nullptr;
  const Gfn2IntegralShellPairTask* forward_ss_tasks = nullptr;
  const Gfn2IntegralShellPairTask* h0_generic_tasks = nullptr;
  const Gfn2IntegralShellPairTask* h0_ss_tasks = nullptr;
  const Gfn2IntegralShellPairTask* force_generic_tasks = nullptr;
  const Gfn2IntegralShellPairTask* force_ss_tasks = nullptr;
};

static_assert(std::is_trivially_copyable_v<Gfn2IntegralDeviceTaskDomains>);
static_assert(std::is_standard_layout_v<Gfn2IntegralDeviceTaskDomains>);

/* Host-visible shape for topology-fixed matrix staging and publication nodes. */
struct Gfn2IntegralLinearLaunchShape {
  std::uint32_t systems = 0u;
  std::uint32_t tiles = 0u;
};

[[nodiscard]] bool make_gfn2_integral_linear_launch_shape(
    std::int64_t batch_size, std::int64_t tiles_per_system,
    Gfn2IntegralLinearLaunchShape& shape) noexcept;

/* First asynchronous semantic or arithmetic failure in an integral sequence. */
enum class Gfn2IntegralDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kInvalidShellMetadata = 2u,
  kInvalidPrimitiveData = 3u,
  kNonfinitePosition = 4u,
  kCoordinateDifferenceOverflow = 5u,
  kNonfiniteIntegralArithmetic = 6u,
  kInvalidH0Parameter = 7u,
  kInvalidCoordination = 8u,
  kNonfiniteOverlap = 9u,
  kNonfiniteH0Arithmetic = 10u,
  kInvalidActiveMask = 11u,
  kNonfiniteAdjoint = 12u,
  kNonfiniteGradientSeed = 13u,
  kNonfiniteGradientArithmetic = 14u,
};

/*
 * Flat device counterpart of BasisPlan and IntegralPlan. The caller uploads
 * every array explicitly and supplies dense shell-pair offsets: system s owns
 * [shell_pair_offsets[s], shell_pair_offsets[s+1]), whose length is nshell^2.
 * maximum_system_shells is the exact maximum nshell over the batch and bounds
 * the allocation-free shell-pair launch grid. linear_tiles_per_system is
 * selected once from the same host topology as density contraction and keeps
 * direct and Graph matrix scans independent of runtime activity. All offsets
 * are zero-based half-open.
 *
 * The basis is limited to the complete GFN2 s/p/d path (l <= 2). Primitive
 * coefficients already include the Cartesian normalization from BasisPlan.
 */
struct Gfn2IntegralDeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t total_primitives = 0;
  std::int64_t total_matrix_elements = 0;
  std::int64_t total_shell_pair_elements = 0;
  std::int64_t maximum_system_shells = 0;
  std::int64_t linear_tiles_per_system = 0;
  double integral_cutoff = 0.0;
  std::uint64_t plan_token = 0;

  std::int64_t atom_offset_count = 0;
  std::int64_t batch_shell_offset_count = 0;
  std::int64_t batch_orbital_offset_count = 0;
  std::int64_t matrix_offset_count = 0;
  std::int64_t shell_pair_offset_count = 0;
  std::int64_t atom_shell_offset_count = 0;
  std::int64_t shell_orbital_offset_count = 0;
  std::int64_t shell_primitive_offset_count = 0;
  std::int64_t shell_to_atom_count = 0;
  std::int64_t angular_momentum_count = 0;
  std::int64_t primitive_exponent_count = 0;
  std::int64_t primitive_coefficient_count = 0;

  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* batch_shell_offsets = nullptr;
  const std::int64_t* batch_orbital_offsets = nullptr;
  const std::int64_t* matrix_offsets = nullptr;
  const std::int64_t* shell_pair_offsets = nullptr;
  const std::int64_t* atom_shell_offsets = nullptr;
  const std::int64_t* shell_orbital_offsets = nullptr;
  const std::int64_t* shell_primitive_offsets = nullptr;
  const std::int64_t* shell_to_atom = nullptr;
  const std::uint8_t* angular_momenta = nullptr;
  const double* primitive_exponents = nullptr;
  const double* primitive_coefficients = nullptr;

  /* The overlap primitive is shared by GFN1 and GFN2. Only GFN2 constructs
   * dipole/quadrupole Hamiltonian operators, so the selected model controls
   * whether those component planes are required, evaluated, and published. */
  XtbModelFlavor model = XtbModelFlavor::kGfn2;

  /* Plan-owned canonical domains. The legacy square-grid path remains a
   * same-binary diagnostic fallback when use_compact_tasks is zero. */
  std::uint32_t use_compact_tasks = 0u;
  std::uint32_t reserved = 0u;
  std::int64_t forward_generic_task_count = 0;
  std::int64_t forward_ss_task_count = 0;
  std::int64_t h0_generic_task_count = 0;
  std::int64_t h0_ss_task_count = 0;
  std::int64_t force_generic_task_count = 0;
  std::int64_t force_ss_task_count = 0;
  const Gfn2IntegralShellPairTask* forward_generic_tasks = nullptr;
  const Gfn2IntegralShellPairTask* forward_ss_tasks = nullptr;
  const Gfn2IntegralShellPairTask* h0_generic_tasks = nullptr;
  const Gfn2IntegralShellPairTask* h0_ss_tasks = nullptr;
  const Gfn2IntegralShellPairTask* force_generic_tasks = nullptr;
  const Gfn2IntegralShellPairTask* force_ss_tasks = nullptr;
};

/* Device-resident immutable arrays uploaded from H0Plan accessors. */
struct Gfn2H0DevicePlan {
  std::int64_t atomic_radius_count = 0;
  std::int64_t shell_level_count = 0;
  std::int64_t shell_coordination_scale_count = 0;
  std::int64_t shell_polynomial_count = 0;
  std::int64_t shell_pair_scale_count = 0;
  std::uint64_t plan_token = 0;

  const double* atomic_radii = nullptr;
  const double* shell_levels = nullptr;
  const double* shell_coordination_scale = nullptr;
  const double* shell_polynomial = nullptr;
  const double* shell_pair_scale = nullptr;
};

/*
 * Caller-owned unpublished storage. Component scratch uses the public
 * component-major packed-matrix layout. sequence_active is one uint32_t used
 * to preserve a sticky pre-existing device_error without host synchronization.
 */
struct Gfn2IntegralDeviceWorkspace {
  double* overlap_scratch = nullptr;
  std::int64_t overlap_elements = 0;
  double* dipole_scratch = nullptr;
  std::int64_t dipole_elements = 0;
  double* quadrupole_scratch = nullptr;
  std::int64_t quadrupole_elements = 0;
  double* h0_scratch = nullptr;
  std::int64_t h0_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_elements = 0;
  std::uint64_t plan_token = 0;
};

static_assert(std::is_trivially_copyable_v<Gfn2IntegralDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2IntegralDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2H0DevicePlan>);
static_assert(std::is_standard_layout_v<Gfn2H0DevicePlan>);
static_assert(std::is_trivially_copyable_v<Gfn2IntegralDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2IntegralDeviceWorkspace>);

/* Clear all per-system errors and the first-error diagnostic asynchronously. */
cudaError_t reset_gfn2_integral_device_errors_cuda(std::int64_t batch_size,
                                                   std::uint32_t* system_errors,
                                                   std::uint32_t* device_error,
                                                   cudaStream_t stream = nullptr) noexcept;

/*
 * Overwrite overlap, dipole, and traceless-quadrupole matrices for the full
 * ragged batch. Multipole operators use the ket AO's atom as origin, matching
 * tblite/xtb. Publication is atomic per system: invalid members retain every
 * prior public matrix element while valid peers commit normally.
 */
cudaError_t evaluate_gfn2_integrals_cuda(const Gfn2IntegralDeviceBatch& batch,
                                         const double* positions, double* overlap, double* dipole,
                                         double* quadrupole,
                                         const Gfn2IntegralDeviceWorkspace& workspace,
                                         std::uint32_t* system_errors, std::uint32_t* device_error,
                                         cudaStream_t stream = nullptr) noexcept;

/*
 * Assemble the complete coordination-dependent GFN2 zero-order Hamiltonian
 * from an existing packed overlap matrix. H0 is staged and published with the
 * same per-system failure atomicity as the integral evaluator.
 */
cudaError_t evaluate_gfn2_h0_cuda(const Gfn2IntegralDeviceBatch& batch,
                                  const Gfn2H0DevicePlan& plan, const double* positions,
                                  const double* coordination_numbers, const double* overlap,
                                  double* hamiltonian, const Gfn2IntegralDeviceWorkspace& workspace,
                                  std::uint32_t* system_errors, std::uint32_t* device_error,
                                  cudaStream_t stream = nullptr) noexcept;

/*
 * Launchers allocate nothing, enqueue only on stream, never synchronize, and
 * are CUDA Graph capture compatible. Call reset before each independent
 * sequence. A pre-existing device_error makes the whole call a no-op; errors
 * discovered by this call are sticky per system and preserve healthy peers.
 */

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_INTEGRALS_CUH
