#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_INFERENCE_PUBLICATION_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN2_INFERENCE_PUBLICATION_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_device_admission.cuh"
#include "backends/cuda/gfn2_geometry.cuh"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::cuda {

inline constexpr std::uint32_t kGfn2InferencePublicationAbiVersion = 2u;

/* Plan-wide structural/upstream failures. Any nonzero value publishes nothing. */
enum class Gfn2InferencePublicationPlanError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kInvalidEligibilityMask = 2u,
  kInvalidEpoch = 3u,
  kTerminalClassicalPlanFailure = 4u,
  kEnergyForcePlanFailure = 5u,
};

/* Stable per-system reason accompanying the public xtbloom status projection. */
enum class Gfn2InferencePublicationSystemError : std::uint32_t {
  kSuccess = 0u,
  kIneligibleNumericalRefresh = 1u,
  kStaleGeneration = 2u,
  kTerminalClassicalFailure = 3u,
  kEnergyForceFailure = 4u,
  kInvalidSccState = 5u,
  kIterationOverflow = 6u,
  kNonfiniteResult = 7u,
};

/*
 * Immutable ragged output shape and requested-property contract. The compute
 * flag values are reused intentionally so #114 can bind the public C request
 * without translating another bit domain.
 */
struct Gfn2InferencePublicationDevicePlan {
  std::uint32_t abi_version = kGfn2InferencePublicationAbiVersion;
  std::uint32_t requested_properties = 0u;
  std::uint64_t plan_token = 0u;
  std::uint64_t maximum_iterations = 0u;

  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_point_charges = 0;
  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* point_charge_offsets = nullptr;

  /* Replay-safe current-geometry authority. */
  Gfn2GeometryEpochDevice geometry_epoch{};
  const std::uint64_t* committed_generations = nullptr;
  std::int64_t generation_elements = 0;
};

/*
 * Complete terminal state and execution staging. terminal_* refers to the
 * repulsion/D4-ATM composer; execution_* refers to the energy/force composer.
 * A nonzero plan scalar is whole-batch, while system arrays remain peer-local.
 */
struct Gfn2InferencePublicationDeviceInput {
  Gfn2DeviceAdmission admission{};
  const std::uint8_t* eligible_mask = nullptr;
  std::int64_t eligible_elements = 0;

  const std::uint64_t* iterations = nullptr;
  const std::uint8_t* converged = nullptr;
  const xtbloom_status_t* system_statuses = nullptr;
  std::int64_t scc_elements = 0;

  const double* energies = nullptr;
  std::int64_t energy_elements = 0;
  const double* qm_forces = nullptr;
  std::int64_t qm_force_elements = 0;
  const double* atomic_charges = nullptr;
  std::int64_t atomic_charge_elements = 0;
  const double* point_forces = nullptr;
  std::int64_t point_force_elements = 0;

  const std::uint32_t* terminal_system_errors = nullptr;
  std::int64_t terminal_system_error_elements = 0;
  const std::uint32_t* terminal_plan_error = nullptr;
  const std::uint32_t* execution_system_errors = nullptr;
  std::int64_t execution_system_error_elements = 0;
  const std::uint32_t* execution_plan_error = nullptr;
  std::uint64_t plan_token = 0u;

  /*
   * Molecular-dipole inputs are stationary, charge-channel multipoles in the
   * same global atom order as plan.atom_offsets. They are required only when
   * XTBLOOM_COMPUTE_DIPOLE_MOMENTS is requested. positions and
   * atomic_dipoles contain three doubles per atom; atomic_charges is shared
   * with the optional public atomic-charge output above and must therefore be
   * bound when either property is requested.
   */
  const double* positions = nullptr;
  std::int64_t position_elements = 0;
  const double* atomic_dipoles = nullptr;
  std::int64_t atomic_dipole_elements = 0;
};

/* Stable internal results later bridged to host or CUDA C-API buffers by #114. */
struct Gfn2InferencePublicationDeviceResults {
  double* energies = nullptr;
  std::int64_t energy_elements = 0;
  double* qm_forces = nullptr;
  std::int64_t qm_force_elements = 0;
  double* atomic_charges = nullptr;
  std::int64_t atomic_charge_elements = 0;
  double* point_forces = nullptr;
  std::int64_t point_force_elements = 0;

  std::int32_t* iterations = nullptr;
  std::uint8_t* converged = nullptr;
  xtbloom_status_t* system_statuses = nullptr;
  std::int64_t batch_elements = 0;
  std::uint64_t plan_token = 0u;

  /* Three molecular Cartesian components per system, in atomic units. */
  double* dipole_moments = nullptr;
  std::int64_t dipole_moment_elements = 0;
};

struct Gfn2InferencePublicationDeviceWorkspace {
  std::uint64_t* epoch_snapshot = nullptr;
  std::int64_t epoch_snapshot_elements = 0;
  std::uint64_t plan_token = 0u;
};

struct Gfn2InferencePublicationDeviceDiagnostics {
  std::uint32_t* system_errors = nullptr;
  std::int64_t system_error_elements = 0;
  std::uint32_t* plan_error = nullptr;
  std::int64_t plan_error_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2InferencePublicationDevicePlan>);
static_assert(std::is_standard_layout_v<Gfn2InferencePublicationDevicePlan>);
static_assert(std::is_trivially_copyable_v<Gfn2InferencePublicationDeviceInput>);
static_assert(std::is_standard_layout_v<Gfn2InferencePublicationDeviceInput>);
static_assert(std::is_trivially_copyable_v<Gfn2InferencePublicationDeviceResults>);
static_assert(std::is_standard_layout_v<Gfn2InferencePublicationDeviceResults>);
static_assert(std::is_trivially_copyable_v<Gfn2InferencePublicationDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2InferencePublicationDeviceWorkspace>);
static_assert(std::is_trivially_copyable_v<Gfn2InferencePublicationDeviceDiagnostics>);
static_assert(std::is_standard_layout_v<Gfn2InferencePublicationDeviceDiagnostics>);

/*
 * Publish one complete internal inference result on stream.
 *
 * A separate plan-preflight kernel runs before any result write, providing
 * whole-batch failure atomicity for malformed topology, epoch, or upstream
 * plan diagnostics. Each peer is then fully preflighted before publication.
 * Successful peers copy every requested finite property. Failed peers publish
 * converged=0, their terminal status, and quiet NaNs across every requested
 * floating-point slice, never stale bytes from a prior inference. A refresh-
 * ineligible or stale-generation peer publishes iterations=0 because it did
 * not enter SCC for the current inference; later SCC/terminal failures retain
 * the current generation's real attempt count. No allocation, transfer,
 * polling, or synchronization is performed.
 */
cudaError_t publish_gfn2_inference_results_cuda(
    const Gfn2InferencePublicationDevicePlan& plan,
    const Gfn2InferencePublicationDeviceInput& input,
    const Gfn2InferencePublicationDeviceResults& results,
    const Gfn2InferencePublicationDeviceWorkspace& workspace,
    const Gfn2InferencePublicationDeviceDiagnostics& diagnostics,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_INFERENCE_PUBLICATION_CUH
