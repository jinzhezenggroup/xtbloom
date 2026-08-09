#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_FORCE_COMPOSITION_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN2_FORCE_COMPOSITION_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_force_common.cuh"

namespace xtbloom::detail::cuda {

/* Stationary gradient/force fields accepted by the final force composer. */
enum class Gfn2ForceCompositionComponent : std::uint32_t {
  kElectronicGradient = 1u << 0u,
  kClassicalForce = 1u << 1u,
  kExplicitPointChargeForce = 1u << 2u,
};

inline constexpr std::uint32_t kGfn2ForceCompositionAllComponents =
    static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kElectronicGradient) |
    static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kClassicalForce) |
    static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kExplicitPointChargeForce);

/* First plan or peer-local semantic/arithmetic failure. */
enum class Gfn2ForceCompositionDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kInvalidRequestedMask = 2u,
  kNonfiniteElectronicGradient = 3u,
  kNonfiniteClassicalForce = 4u,
  kNonfiniteExplicitQmForce = 5u,
  kNonfiniteExplicitPointForce = 6u,
  kNonfiniteForceArithmetic = 7u,
};

/* Immutable ragged output topology and component contract. */
struct Gfn2ForceCompositionDeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_point_charges = 0;
  std::int64_t atom_offset_count = 0;
  std::int64_t point_charge_offset_count = 0;
  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* point_charge_offsets = nullptr;
  std::uint32_t enabled_components = 0u;
  std::uint64_t plan_token = 0u;
};

/*
 * electronic_gradients uses dE/dR. classical_forces and explicit point-charge
 * arrays already use the public force convention and are added after the
 * electronic gradient is negated. Disabled fields use canonical null/zero views.
 */
struct Gfn2ForceCompositionDeviceInput {
  const double* electronic_gradients = nullptr;
  std::int64_t electronic_gradient_elements = 0;
  const double* classical_forces = nullptr;
  std::int64_t classical_force_elements = 0;
  const double* explicit_qm_forces = nullptr;
  std::int64_t explicit_qm_force_elements = 0;
  const double* explicit_point_forces = nullptr;
  std::int64_t explicit_point_force_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Either output may be omitted; point output is empty when the batch has no points. */
struct Gfn2ForceCompositionDeviceOutput {
  double* qm_forces = nullptr;
  std::int64_t qm_force_elements = 0;
  double* point_forces = nullptr;
  std::int64_t point_force_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Caller-owned unpublished force slices plus one plan-preflight snapshot. */
struct Gfn2ForceCompositionDeviceWorkspace {
  double* qm_force_scratch = nullptr;
  std::int64_t qm_force_elements = 0;
  double* point_force_scratch = nullptr;
  std::int64_t point_force_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2ForceCompositionDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2ForceCompositionDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2ForceCompositionDeviceInput>);
static_assert(std::is_standard_layout_v<Gfn2ForceCompositionDeviceInput>);
static_assert(std::is_trivially_copyable_v<Gfn2ForceCompositionDeviceOutput>);
static_assert(std::is_standard_layout_v<Gfn2ForceCompositionDeviceOutput>);
static_assert(std::is_trivially_copyable_v<Gfn2ForceCompositionDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2ForceCompositionDeviceWorkspace>);

/* Clear peer diagnostics and the plan-only sticky scalar asynchronously. */
cudaError_t reset_gfn2_force_composition_device_errors_cuda(std::int64_t batch_size,
                                                            std::uint32_t* system_errors,
                                                            std::uint32_t* plan_error,
                                                            cudaStream_t stream = nullptr) noexcept;

/*
 * Publish final forces for requested terminal-success members:
 *
 *   F_QM = -g_electronic + F_classical + F_explicit-PC-on-QM,
 *   F_PC = F_explicit-PC-on-points.
 *
 * An unrequested or failed member is rejected before any component value is
 * read. Each eligible member is fully preflighted into scratch before either
 * public slice is overwritten, so numerical failure cannot expose a partial
 * force. The launcher allocates, transfers, and synchronizes nothing; it is
 * custom-stream safe and CUDA Graph capturable.
 */
cudaError_t compose_gfn2_forces_cuda(const Gfn2ForceCompositionDeviceBatch& batch,
                                     const Gfn2ForceDeviceActivity& activity,
                                     const Gfn2ForceCompositionDeviceInput& input,
                                     const Gfn2ForceCompositionDeviceOutput& output,
                                     const Gfn2ForceCompositionDeviceWorkspace& workspace,
                                     std::uint32_t* system_errors, std::uint32_t* plan_error,
                                     cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_FORCE_COMPOSITION_CUH
