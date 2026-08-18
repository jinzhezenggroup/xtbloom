#ifndef XTBLOOM_BACKENDS_CUDA_GFN1_CLASSICAL_CORRECTIONS_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN1_CLASSICAL_CORRECTIONS_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/common/xtb_model.hpp"

namespace xtbloom::detail::cuda {

inline constexpr std::int64_t kGfn1D3MaximumReferences = 7;
inline constexpr std::int64_t kGfn1D3ReferencePairStride =
    kGfn1D3MaximumReferences * kGfn1D3MaximumReferences;

/* Peer-local failures reported by the fused GFN1 D3/halogen correction. */
enum class Gfn1ClassicalCorrectionDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidTopology = 1u,
  kNonfiniteInput = 2u,
  kNonfiniteArithmetic = 3u,
  kInvalidActiveMask = 4u,
};

/* Immutable CUDA projection of the charge-independent GFN1 D3(BJ) and
 * halogen-bond corrections. All parameter arrays are expanded by the host
 * model plan so device code never indexes a different model's parameter table. */
struct Gfn1ClassicalCorrectionDevicePlan {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_pairs = 0;
  std::uint64_t plan_token = 0u;

  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* pair_offsets = nullptr;
  const double* covalent_radii = nullptr;

  const std::uint8_t* reference_counts = nullptr;
  const double* reference_cn = nullptr;  // [nat, 7]
  const double* reference_c6 = nullptr;  // [npair, 49]
  const double* pair_rrij = nullptr;
  const double* pair_damping_radii = nullptr;

  const double* halogen_scaled_radii = nullptr;
  const double* halogen_bond_strength = nullptr;
  const std::uint8_t* halogen_donor = nullptr;
  const std::uint8_t* halogen_acceptor = nullptr;

  /* Exact extents make the nested descriptor independently auditable before
   * any CUDA launch. The correction is GFN1-only and must project the same
   * canonical topology/model identity as its outer runtime plan. */
  XtbModelFlavor model = XtbModelFlavor::kGfn1;
  std::int64_t atom_offset_elements = 0;
  std::int64_t pair_offset_elements = 0;
  std::int64_t covalent_radius_elements = 0;
  std::int64_t reference_count_elements = 0;
  std::int64_t reference_cn_elements = 0;
  std::int64_t reference_c6_elements = 0;
  std::int64_t pair_rrij_elements = 0;
  std::int64_t pair_damping_radius_elements = 0;
  std::int64_t halogen_scaled_radius_elements = 0;
  std::int64_t halogen_bond_strength_elements = 0;
  std::int64_t halogen_donor_elements = 0;
  std::int64_t halogen_acceptor_elements = 0;
};

struct Gfn1ClassicalCorrectionDeviceWorkspace {
  double* weights = nullptr;                // [nat, 7]
  double* weight_cn_derivatives = nullptr;  // [nat, 7]
  double* coordination_adjoints = nullptr;  // [nat]
  std::int64_t* axis_neighbors = nullptr;   // [nat]
  double* batch_scratch = nullptr;          // [batch], unpublished energy candidates
  double* gradient_scratch = nullptr;       // [3*nat], unpublished gradient candidates
  std::uint64_t plan_token = 0u;
  std::int64_t weight_elements = 0;
  std::int64_t weight_cn_derivative_elements = 0;
  std::int64_t coordination_adjoint_elements = 0;
  std::int64_t axis_neighbor_elements = 0;
  std::int64_t batch_scratch_elements = 0;
  std::int64_t gradient_scratch_elements = 0;
};

static_assert(std::is_trivially_copyable_v<Gfn1ClassicalCorrectionDevicePlan>);
static_assert(std::is_standard_layout_v<Gfn1ClassicalCorrectionDevicePlan>);
static_assert(std::is_trivially_copyable_v<Gfn1ClassicalCorrectionDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn1ClassicalCorrectionDeviceWorkspace>);

/* Host-side descriptor/range validation shared by the standalone primitive
 * and composed terminal/force paths. It dereferences no device storage. */
bool validate_gfn1_classical_correction_binding(
    const Gfn1ClassicalCorrectionDevicePlan& plan, const double* positions,
    const double* coordination_numbers, const std::uint8_t* active_mask, double* energies,
    double* gradients, const Gfn1ClassicalCorrectionDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* plan_error) noexcept;

/* Add GFN1 D3(BJ)+halogen corrections to finite caller-owned accumulators.
 * Either energies or gradients may be null, but not both. Each active system
 * publishes both candidates only after every correction and final finiteness
 * check succeeds, so a peer-local failure leaves its accumulators unchanged.
 * The kernel is one serial worker per ragged member to retain the CPU
 * reference's deterministic pair/triple accumulation order. No allocation,
 * transfer, synchronization, or host polling occurs and the launch is CUDA
 * Graph compatible. */
cudaError_t add_gfn1_classical_corrections_cuda(
    const Gfn1ClassicalCorrectionDevicePlan& plan, const double* positions,
    const double* coordination_numbers, const std::uint8_t* active_mask, double* energies,
    double* gradients, const Gfn1ClassicalCorrectionDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* plan_error,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif
