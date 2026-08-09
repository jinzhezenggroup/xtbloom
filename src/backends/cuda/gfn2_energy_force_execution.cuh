#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_ENERGY_FORCE_EXECUTION_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN2_ENERGY_FORCE_EXECUTION_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_classical_force.cuh"
#include "backends/cuda/gfn2_electronic_gradient.cuh"
#include "backends/cuda/gfn2_external_point_charges.cuh"
#include "backends/cuda/gfn2_force_composition.cuh"
#include "backends/cuda/gfn2_pairlist.cuh"
#include "backends/cuda/gfn2_post_scc_potential.cuh"
#include "backends/cuda/gfn2_scc_classical_energy.cuh"
#include "backends/cuda/gfn2_scc_potential.cuh"
#include "backends/cuda/gfn2_total_energy.cuh"

namespace xtbloom::detail::cuda {

/* Stable execution-level mapping of failures from the composed device stages. */
enum class Gfn2EnergyForceExecutionDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidForceRequest = 1u,
  kTotalEnergyFailure = 2u,
  kPostSccPotentialFailure = 3u,
  kElectronicGradientFailure = 4u,
  kCoordinationGradientFailure = 5u,
  kClassicalForceFailure = 6u,
  kExternalPointChargeForceFailure = 7u,
  kForceCompositionFailure = 8u,
  kInvalidGeometryTransaction = 9u,
  kIneligibleGeometry = 10u,
  kStaleGeometry = 11u,
};

/*
 * Immutable stationary-GFN2 execution binding. compute_forces distinguishes
 * an energy-only plan from an energy+force plan at setup time. Energy-only
 * execution ignores every force-specific nested descriptor.
 */
struct Gfn2EnergyForceExecutionDevicePlan {
  std::uint8_t compute_forces = 0u;
  /*
   * Exact SCC component contracts used to construct the converged
   * Hamiltonian and free energy. Restricted execution requires the two masks
   * to agree; post-SCC refresh and force selection are derived from them.
   */
  std::uint32_t scc_potential_components = 0u;
  std::uint32_t scc_energy_components = 0u;
  std::uint64_t plan_token = 0u;

  Gfn2TotalEnergyDeviceBatch total_energy_batch{};
  Gfn2PostSccPotentialDevicePlan post_scc_potential_plan{};
  Gfn2IntegralDeviceBatch integral_batch{};
  Gfn2H0DevicePlan h0_plan{};
  Gfn2HamiltonianDeviceBatch hamiltonian_batch{};
  Gfn2GeometryDeviceBatch coordination_batch{};
  Gfn2GeometryDeviceCache coordination_cache{};
  std::uint64_t geometry_generation = 0u;
  Gfn2ClassicalForceDevicePlan classical_plan{};
  Gfn2ExternalPointChargeDeviceBatch external_point_charge_batch{};
  Gfn2ForceCompositionDeviceBatch force_composition_batch{};
  /* Step 5: optional sparse CN VJP leaf.  The committed pair-list consumer
   * view is the projection published by the preprocessing final gate; when it
   * is present (nonzero token) the H0 coordination VJP is computed with the
   * sparse path and checked for dense parity.  pairlist_batch carries the
   * per-system dispatch used by the builder. */
  Gfn2PairListConsumerView pairlist_committed{};
  Gfn2PairListDeviceBatch pairlist_batch{};
};

/* Converged SCC state plus all stationary post-SCC force inputs. */
struct Gfn2EnergyForceExecutionDeviceInput {
  Gfn2TotalEnergyDeviceInput total_energy{};
  Gfn2TotalEnergyDeviceSccState scc_state{};
  Gfn2ForceDeviceActivity force_activity{};
  Gfn2PostSccPotentialDeviceInput post_scc_potential{};
  Gfn2H0ForceDeviceInput h0{};
  Gfn2HamiltonianForceDeviceInput hamiltonian{};
  Gfn2ClassicalForceDeviceInput classical{};
  const double* external_shell_charges = nullptr;
  std::int64_t external_shell_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Only these buffers are externally published by the execution. */
struct Gfn2EnergyForceExecutionDeviceResults {
  Gfn2TotalEnergyDeviceResults energy{};
  Gfn2ForceCompositionDeviceOutput forces{};
  std::uint64_t plan_token = 0u;
};

/*
 * Force component storage owned by the execution plan. The composer zeros
 * eligible slices before use, so stale graph-replay contents are never seeds.
 */
struct Gfn2EnergyForceExecutionDeviceIntermediates {
  Gfn2TotalEnergyDeviceResults energy{};
  Gfn2PostSccPotentialDeviceResults post_scc_potential{};
  Gfn2PostSccPotentialDeviceIntermediates post_scc_potential_intermediates{};
  Gfn2H0ForceDeviceOutput h0{};
  Gfn2HamiltonianForceDeviceOutput hamiltonian{};
  Gfn2ClassicalForceDeviceOutput classical{};
  double* explicit_qm_forces = nullptr;
  std::int64_t explicit_qm_force_elements = 0;
  double* explicit_point_forces = nullptr;
  std::int64_t explicit_point_force_elements = 0;
  Gfn2ForceCompositionDeviceOutput forces{};
  std::uint64_t plan_token = 0u;
};

/* Caller-owned, allocation-free storage for every nested stage. */
struct Gfn2EnergyForceExecutionDeviceWorkspace {
  Gfn2TotalEnergyDeviceWorkspace total_energy{};
  Gfn2PostSccPotentialDeviceWorkspace post_scc_potential{};
  Gfn2H0ForceDeviceWorkspace h0{};
  Gfn2HamiltonianForceDeviceWorkspace hamiltonian{};
  Gfn2IntegralForceDeviceWorkspace integral{};
  Gfn2ElectronicGradientDeviceWorkspace electronic{};
  Gfn2GeometryDeviceWorkspace coordination{};
  Gfn2ClassicalForceDeviceWorkspace classical{};
  Gfn2ExternalPointChargeForceDeviceWorkspace external_point_charge{};
  Gfn2ForceCompositionDeviceWorkspace force_composition{};
  /* Step 5 sparse CN VJP differential scratch.  When plan.pairlist_committed
   * is enabled, the H0 coordination VJP runs through both the dense geometry
   * cache (production reference) and the committed sparse pair list; the two
   * per-atom gradient results are compared bitwise and any peer whose slices
   * disagree fails closed.  sparse_gradient_scratch is the sparse result
   * buffer and sparse_sequence_active is the one-element VJP sequence watched
   * by the paired kernels. */
  double* sparse_gradient_scratch = nullptr;
  std::int64_t sparse_gradient_elements = 0;
  std::uint32_t* sparse_sequence_active = nullptr;
  std::int64_t sparse_sequence_elements = 0;

  std::uint8_t* energy_success_mask = nullptr;
  std::uint8_t* post_scc_success_mask = nullptr;
  std::uint8_t* electronic_success_mask = nullptr;
  std::uint8_t* coordination_success_mask = nullptr;
  std::uint8_t* classical_success_mask = nullptr;
  std::uint8_t* external_success_mask = nullptr;
  std::int64_t mask_elements = 0;
  /*
   * Whole-execution transaction latch. Zero means open; the first mapped
   * plan-wide asynchronous failure closes publication for the complete batch.
   * This storage is required in both energy-only and force modes.
   */
  std::uint32_t* plan_failure = nullptr;
  std::int64_t plan_failure_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Per-stage details plus one stable execution-level error projection. */
struct Gfn2EnergyForceExecutionDeviceDiagnostics {
  std::uint32_t* execution_system_errors = nullptr;
  std::uint32_t* execution_device_error = nullptr;
  std::uint32_t* total_energy_system_errors = nullptr;
  std::uint32_t* total_energy_device_error = nullptr;
  Gfn2PostSccPotentialDeviceDiagnostics post_scc_potential{};
  Gfn2ElectronicGradientDeviceDiagnostics electronic{};
  std::uint32_t* coordination_system_errors = nullptr;
  std::uint32_t* coordination_device_error = nullptr;
  std::uint32_t* classical_system_errors = nullptr;
  std::uint32_t* classical_device_error = nullptr;
  std::uint32_t* external_system_errors = nullptr;
  std::uint32_t* external_device_error = nullptr;
  std::uint32_t* force_composition_system_errors = nullptr;
  std::uint32_t* force_composition_plan_error = nullptr;
  std::int64_t batch_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2EnergyForceExecutionDevicePlan>);
static_assert(std::is_standard_layout_v<Gfn2EnergyForceExecutionDevicePlan>);
static_assert(std::is_trivially_copyable_v<Gfn2EnergyForceExecutionDeviceInput>);
static_assert(std::is_standard_layout_v<Gfn2EnergyForceExecutionDeviceInput>);
static_assert(std::is_trivially_copyable_v<Gfn2EnergyForceExecutionDeviceResults>);
static_assert(std::is_standard_layout_v<Gfn2EnergyForceExecutionDeviceResults>);
static_assert(std::is_trivially_copyable_v<Gfn2EnergyForceExecutionDeviceIntermediates>);
static_assert(std::is_standard_layout_v<Gfn2EnergyForceExecutionDeviceIntermediates>);
static_assert(std::is_trivially_copyable_v<Gfn2EnergyForceExecutionDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2EnergyForceExecutionDeviceWorkspace>);
static_assert(std::is_trivially_copyable_v<Gfn2EnergyForceExecutionDeviceDiagnostics>);
static_assert(std::is_standard_layout_v<Gfn2EnergyForceExecutionDeviceDiagnostics>);

/*
 * Execute post-SCC GFN2 energy and optional analytic forces.
 *
 * The sequence consumes converged SCC outputs and never launches another SCC
 * iteration. Every stage is enqueued on stream and stage success is propagated
 * through device masks; no host polling or synchronization is performed.
 * Healthy ragged peers continue after another peer fails.
 *
 * Force execution is
 *
 *   electronic H0/S/D/Q reverse
 *     -> H0 coordination-number reverse through the existing geometry cache
 *     -> classical force reverse
 *     -> optional explicit point-charge forces
 *     -> final force convention/composition.
 *
 * Energy and final QM/point forces are first written to execution-owned
 * staging. One terminal publication kernel copies all requested public values
 * only after the entire member-local chain succeeds; a failed member retains
 * every public byte. Plan-wide asynchronous failures suppress publication for
 * the complete batch.
 *
 * Energy-only plans inspect and bind no force-specific descriptor, workspace,
 * intermediate, or diagnostic storage. Both modes are allocation-free,
 * custom-stream safe, and CUDA Graph capture/replay compatible.
 */
cudaError_t execute_gfn2_energy_force_cuda(
    const Gfn2EnergyForceExecutionDevicePlan& plan,
    const Gfn2EnergyForceExecutionDeviceInput& input,
    const Gfn2EnergyForceExecutionDeviceResults& results,
    const Gfn2EnergyForceExecutionDeviceIntermediates& intermediates,
    const Gfn2EnergyForceExecutionDeviceWorkspace& workspace,
    const Gfn2EnergyForceExecutionDeviceDiagnostics& diagnostics,
    cudaStream_t stream = nullptr) noexcept;

/* Replay-safe execution consuming the runtime-owned geometry transaction. */
cudaError_t execute_gfn2_energy_force_cuda(
    const Gfn2EnergyForceExecutionDevicePlan& plan,
    const Gfn2EnergyForceExecutionDeviceInput& input,
    const Gfn2EnergyForceExecutionDeviceResults& results,
    const Gfn2EnergyForceExecutionDeviceIntermediates& intermediates,
    const Gfn2EnergyForceExecutionDeviceWorkspace& workspace,
    const Gfn2EnergyForceExecutionDeviceDiagnostics& diagnostics,
    const Gfn2GeometryEpochConsumerDevice& geometry, cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_ENERGY_FORCE_EXECUTION_CUH
