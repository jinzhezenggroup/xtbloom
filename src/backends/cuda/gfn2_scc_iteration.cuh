#ifndef GPUXTB_BACKENDS_CUDA_GFN2_SCC_ITERATION_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_SCC_ITERATION_CUH

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "backends/common/gfn2_plan_schema.hpp"
#include "backends/cuda/gfn2_aes2.cuh"
#include "backends/cuda/gfn2_d4.cuh"
#include "backends/cuda/gfn2_density.cuh"
#include "backends/cuda/gfn2_eigensolver.cuh"
#include "backends/cuda/gfn2_es2.cuh"
#include "backends/cuda/gfn2_es3.cuh"
#include "backends/cuda/gfn2_external_point_charges.cuh"
#include "backends/cuda/gfn2_geometry.cuh"
#include "backends/cuda/gfn2_hamiltonian.cuh"
#include "backends/cuda/gfn2_mulliken.cuh"
#include "backends/cuda/gfn2_occupations.cuh"
#include "backends/cuda/gfn2_periodic_embedding.cuh"
#include "backends/cuda/gfn2_scc.cuh"
#include "backends/cuda/gfn2_scc_bridge.cuh"
#include "backends/cuda/gfn2_scc_classical_energy.cuh"
#include "backends/cuda/gfn2_scc_energy.cuh"
#include "backends/cuda/gfn2_scc_free_energy.cuh"
#include "backends/cuda/gfn2_scc_iteration_control.cuh"
#include "backends/cuda/gfn2_scc_mixer.cuh"
#include "backends/cuda/gfn2_scc_potential.cuh"
#include "backends/cuda/gfn2_scc_publication.cuh"

namespace gpuxtb::detail::cuda {

inline constexpr std::uint32_t kGfn2SccIterationAbiVersion = 1u;
inline constexpr std::int64_t kGfn2SccIterationBaseStageReportCount = 19;
inline constexpr std::int64_t kGfn2SccIterationMaximumStageReportCount = 24;
inline constexpr std::int64_t kGfn2SccIterationStageReportCapacity = 40;

/*
 * CUDA-provider capture policy fixed during setup. kUncapturedSegmentRequired
 * is not a numerical failure: it tells the #89 composer to return a distinct
 * synchronous result if the caller asks to capture the provider segment.
 */
enum class Gfn2SccIterationProviderCaptureMode : std::uint32_t {
  kUnspecified = 0u,
  kGraphSupported = 1u,
  kUncapturedSegmentRequired = 2u,
};

/* Setup-time binding failures. No device work is enqueued for these errors. */
enum class Gfn2SccIterationBindingError : std::uint32_t {
  kSuccess = 0u,
  kInvalidAbiVersion = 1u,
  kInvalidPlanToken = 2u,
  kCrossPlan = 3u,
  kInvalidCount = 4u,
  kInsufficientCapacity = 5u,
  kNullPointer = 6u,
  kMisalignedPointer = 7u,
  kAddressOverflow = 8u,
  kForbiddenAlias = 9u,
  kInvalidTopology = 10u,
  kInvalidStageReport = 11u,
  kInvalidBucket = 12u,
  kInvalidProvider = 13u,
  kInvalidZeroCopyView = 14u,
};

/* Coarse owner of the first rejected field. index narrows array/stage entries. */
enum class Gfn2SccIterationBindingField : std::uint32_t {
  kNone = 0u,
  kPlan = 1u,
  kTopology = 2u,
  kActivity = 3u,
  kGeometry = 4u,
  kPotential = 5u,
  kES2 = 6u,
  kES3 = 7u,
  kAES2 = 8u,
  kD4 = 9u,
  kExplicitPointCharge = 10u,
  kPeriodicEmbedding = 11u,
  kScalarBridge = 12u,
  kHamiltonian = 13u,
  kEigensolver = 14u,
  kOccupations = 15u,
  kDensity = 16u,
  kMulliken = 17u,
  kElectronicEnergy = 18u,
  kClassicalEnergy = 19u,
  kFreeEnergy = 20u,
  kMixer = 21u,
  kStatePublication = 22u,
  kStageReports = 23u,
  kWorkspace = 24u,
};

struct Gfn2SccIterationBindingDiagnostic {
  Gfn2SccIterationBindingError error = Gfn2SccIterationBindingError::kSuccess;
  Gfn2SccIterationBindingField field = Gfn2SccIterationBindingField::kNone;
  std::int64_t index = -1;
};

/*
 * Complete synchronous status envelope reserved for the #89 launch entry.
 * Provider errors are intentionally not collapsed into cudaError_t: callers
 * can distinguish CUDA launch failure, cuBLAS failure, cuSOLVER failure, and a
 * setup-declared Graph capture boundary without polling device memory.
 */
enum class Gfn2SccIterationLaunchStatus : std::uint32_t {
  kSuccess = 0u,
  kInvalidBinding = 1u,
  kCudaError = 2u,
  kCublasError = 3u,
  kCusolverError = 4u,
  kProviderCaptureUnsupported = 5u,
};

struct Gfn2SccIterationLaunchResult {
  Gfn2SccIterationLaunchStatus status = Gfn2SccIterationLaunchStatus::kSuccess;
  Gfn2SccStageId stage = Gfn2SccStageId::kNone;
  Gfn2SccIterationBindingDiagnostic binding{};
  cudaError_t cuda_status = cudaSuccess;
  cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

  [[nodiscard]] bool success() const noexcept {
    return status == Gfn2SccIterationLaunchStatus::kSuccess;
  }
};

/*
 * CUDA-specific provider leaf. All other SCC iteration descriptors are POD
 * pointer/count views and can be mirrored unchanged by a HIP implementation.
 * A ROCm backend replaces only this leaf with rocBLAS/rocSOLVER handles and
 * provider workspace while preserving the surrounding plan/input/state ABI.
 * buckets is a setup-owned host array; every other pointer is caller-owned.
 */
struct Gfn2SccIterationCudaEigensolverProvider {
  const Gfn2EigensolverBucket* buckets = nullptr;
  std::int64_t bucket_count = 0;
  cusolverDnHandle_t solver = nullptr;
  cusolverDnParams_t parameters = nullptr;
  cublasHandle_t blas = nullptr;

  void* device_workspace = nullptr;
  std::size_t device_workspace_bytes = 0u;
  void* host_workspace = nullptr;
  std::size_t host_workspace_bytes = 0u;
  Gfn2EigensolverWorkspaceRequirements requirements{};
  Gfn2SccIterationProviderCaptureMode capture_mode =
      Gfn2SccIterationProviderCaptureMode::kUnspecified;
  std::uint32_t reserved = 0u;
  std::uint64_t plan_token = 0u;
};

/* Mutable component outputs, kept separate from their const composer views. */
struct Gfn2SccIterationDeviceComponentStorage {
  double* es2_shell_potential = nullptr;
  std::int64_t es2_shell_elements = 0;
  double* es3_shell_potential = nullptr;
  std::int64_t es3_shell_elements = 0;
  double* aes2_atomic_potential = nullptr;
  std::int64_t aes2_atomic_elements = 0;
  double* aes2_dipole_potential = nullptr;
  std::int64_t aes2_dipole_elements = 0;
  double* aes2_quadrupole_potential = nullptr;
  std::int64_t aes2_quadrupole_elements = 0;
  double* d4_atomic_potential = nullptr;
  std::int64_t d4_atomic_elements = 0;
  double* periodic_atomic_potential = nullptr;
  std::int64_t periodic_atomic_elements = 0;

  double* es2_energy = nullptr;
  std::int64_t es2_energy_elements = 0;
  double* es3_energy = nullptr;
  std::int64_t es3_energy_elements = 0;
  double* aes2_energy = nullptr;
  std::int64_t aes2_energy_elements = 0;
  double* d4_two_body_energy = nullptr;
  std::int64_t d4_two_body_energy_elements = 0;
  double* explicit_point_charge_energy = nullptr;
  std::int64_t explicit_point_charge_energy_elements = 0;
  double* periodic_embedding_energy = nullptr;
  std::int64_t periodic_embedding_energy_elements = 0;
  double* core_energy = nullptr;
  std::int64_t core_energy_elements = 0;
  double* electronic_free_energy = nullptr;
  std::int64_t electronic_free_energy_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Fixed zero-copy arguments for the primitive whose API uses raw pointers. */
struct Gfn2SccIterationDeviceElectronicEnergyInput {
  const double* density = nullptr;
  std::int64_t density_elements = 0;
  const double* h0 = nullptr;
  std::int64_t h0_elements = 0;
  const double* entropies = nullptr;
  std::int64_t entropy_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Canonical-activity scalar adapter boundary. This deliberately does not bind
 * the legacy bridge's downstream_active output: #89 consumes the one canonical
 * activity/sequence ledger and uses only shell_scalar plus unpublished scratch.
 */
struct Gfn2SccIterationDeviceScalarBridge {
  Gfn2SccBridgeDevicePotentialFields fields{};
  double* shell_scalar = nullptr;
  std::int64_t shell_elements = 0;
  Gfn2SccBridgeDeviceWorkspace workspace{};
  std::uint64_t plan_token = 0u;
};

/*
 * Immutable topology, policies, primitive plans/caches, and provider binding.
 * The complete object is copied once at setup and remains byte-identical for
 * repeated launches and CUDA Graph replay. Numerical arrays may change value;
 * their addresses, capacities, plan identity, and layout do not.
 */
struct Gfn2SccIterationDevicePlan {
  std::uint32_t abi_version = kGfn2SccIterationAbiVersion;
  std::uint32_t enabled_components = 0u;
  std::uint64_t plan_token = 0u;
  std::uint64_t geometry_generation = 0u;

  Gfn2RaggedTopologyView topology{};
  Gfn2SccIterationDevicePolicy activity_policy{};
  Gfn2SccDevicePolicy state_policy{};
  Gfn2SccMixerDevicePolicy mixer_policy{};
  Gfn2SccIterationDeviceProvenance provenance{};

  Gfn2GeometryDeviceBatch geometry_batch{};
  Gfn2GeometryDeviceCache geometry_cache{};
  Gfn2SccDeviceBatch scc_batch{};
  Gfn2SccPotentialDeviceBatch potential_batch{};
  Gfn2ES2DeviceBatch es2_batch{};
  Gfn2ES2DeviceCache es2_cache{};
  Gfn2ES3DeviceBatch es3_batch{};
  Gfn2AES2DeviceBatch aes2_batch{};
  Gfn2AES2DeviceCache aes2_cache{};
  Gfn2D4DeviceBatch d4_batch{};
  Gfn2D4DeviceParameters d4_parameters{};
  Gfn2D4DeviceCache d4_cache{};
  Gfn2ExternalPointChargeDeviceBatch explicit_point_charge_batch{};
  Gfn2ExternalPointChargeDeviceCache explicit_point_charge_cache{};
  Gfn2PeriodicEmbeddingDeviceBatch periodic_batch{};
  Gfn2SccBridgeDeviceBatch scalar_bridge_batch{};
  Gfn2HamiltonianDeviceBatch hamiltonian_batch{};
  Gfn2EigensolverDeviceBatch eigensolver_batch{};
  Gfn2EigensolverOverlapCache overlap_cache{};
  Gfn2EigensolverOptions eigensolver_options{};
  Gfn2SccIterationCudaEigensolverProvider eigensolver_provider{};
  Gfn2OccupationsDeviceBatch occupations_batch{};
  Gfn2DensityDeviceBatch density_batch{};
  Gfn2MullikenDeviceBatch mulliken_batch{};
  Gfn2SccEnergyDeviceBatch electronic_energy_batch{};
  Gfn2SccClassicalEnergyDeviceBatch classical_energy_batch{};
  Gfn2SccFreeEnergyDeviceBatch free_energy_batch{};
  Gfn2SccPublicationDevicePlan publication_plan{};

  /* Immutable stage-to-diagnostic mapping; storage remains caller-owned. */
  Gfn2SccStageDeviceReport reports[kGfn2SccIterationStageReportCapacity]{};
  std::int64_t report_count = 0;
};

/*
 * Numerical input projections. Downstream pointers intentionally repeat the
 * exact upstream state/workspace addresses; setup validation proves these
 * zero-copy edges once so the hot composer never rebuilds a descriptor.
 */
struct Gfn2SccIterationDeviceInput {
  Gfn2SccIterationDeviceStateInput activity_state{};
  Gfn2SccPotentialDeviceMixedFields mixed_fields{};
  Gfn2HamiltonianDeviceInput hamiltonian{};
  const double* eigensolver_hamiltonians = nullptr;
  std::int64_t eigensolver_hamiltonian_elements = 0;
  const double* occupation_eigenvalues = nullptr;
  std::int64_t occupation_eigenvalue_elements = 0;
  Gfn2DensityDeviceInput density{};
  Gfn2MullikenDeviceInput mulliken{};
  Gfn2SccIterationDeviceElectronicEnergyInput electronic_energy{};
  Gfn2SccClassicalEnergyDeviceInput classical_energy{};
  Gfn2SccFreeEnergyDeviceInput free_energy{};
  Gfn2SccDeviceConstMultipoles raw_multipoles{};
  const double* complete_free_energies = nullptr;
  std::int64_t complete_free_energy_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Persistent public wavefunction, mixer history, energy trace, and status. */
struct Gfn2SccIterationDeviceState {
  Gfn2EigensolverDeviceResults eigenpairs{};
  Gfn2OccupationsDeviceResults occupations{};
  Gfn2DensityDeviceResults density{};
  Gfn2MullikenDevicePopulation raw_population{};
  Gfn2SccClassicalEnergyDeviceDiagnostics classical_energy{};
  Gfn2SccFreeEnergyDeviceDiagnostics free_energy{};
  Gfn2SccMixerDeviceState mixer{};
  Gfn2SccDeviceMultipoles published{};
  Gfn2SccDeviceState scc{};
  /* Exact zero-copy projection consumed by the transactional #99 commit. */
  Gfn2SccPublicationDevicePublicState publication{};
  std::uint64_t plan_token = 0u;
};

/*
 * Complete caller-owned scratch graph. Primitive launchers receive the plan's
 * already-bound diagnostic pointers directly; no per-call report or pointer
 * arithmetic is needed.
 */
struct Gfn2SccIterationDeviceWorkspace {
  Gfn2SccIterationDeviceLedger ledger{};
  Gfn2SccIterationDeviceActivity activity{};
  Gfn2SccPotentialDeviceActivity potential_activity{};
  Gfn2HamiltonianDeviceActivity hamiltonian_activity{};
  Gfn2MullikenDeviceActivity mulliken_activity{};
  Gfn2SccClassicalEnergyDeviceActivity classical_energy_activity{};
  Gfn2SccFreeEnergyDeviceActivity free_energy_activity{};

  Gfn2SccPotentialDeviceTopologyMultipoles mixed_topology{};
  Gfn2SccIterationDeviceComponentStorage components{};
  Gfn2SccPotentialDeviceComponents potential_components{};
  Gfn2SccPotentialDeviceResults complete_potentials{};
  Gfn2SccIterationDeviceScalarBridge scalar_bridge{};
  Gfn2HamiltonianDeviceOutput hamiltonian{};

  /* Unpublished wavefunction/energy/history transaction for active members. */
  Gfn2EigensolverDeviceResults staged_eigenpairs{};
  Gfn2OccupationsDeviceResults staged_occupations{};
  Gfn2DensityDeviceResults staged_density{};
  Gfn2MullikenDevicePopulation staged_raw_population{};
  Gfn2SccClassicalEnergyDeviceDiagnostics staged_classical_energy{};
  Gfn2SccFreeEnergyDeviceDiagnostics staged_free_energy{};
  Gfn2SccMixerDeviceState staged_mixer{};
  Gfn2SccDeviceMultipoles next_mixed{};
  /* Exact zero-copy projection of the complete unpublished transaction. */
  Gfn2SccPublicationDeviceStagedState staged_publication{};

  Gfn2GeometryDeviceWorkspace geometry_workspace{};
  Gfn2ES2DeviceWorkspace es2_workspace{};
  Gfn2AES2DeviceWorkspace aes2_workspace{};
  Gfn2D4DeviceWorkspace d4_workspace{};
  Gfn2PeriodicEmbeddingDeviceWorkspace periodic_workspace{};
  Gfn2SccPotentialDeviceWorkspace potential_workspace{};
  Gfn2HamiltonianDeviceWorkspace hamiltonian_workspace{};
  Gfn2EigensolverDeviceWorkspace eigensolver_workspace{};
  Gfn2OccupationsDeviceWorkspace occupations_workspace{};
  Gfn2DensityDeviceWorkspace density_workspace{};
  Gfn2MullikenDeviceWorkspace mulliken_workspace{};
  Gfn2SccEnergyDeviceWorkspace electronic_energy_workspace{};
  Gfn2SccClassicalEnergyDeviceWorkspace classical_energy_workspace{};
  Gfn2SccFreeEnergyDeviceWorkspace free_energy_workspace{};
  Gfn2SccMixerDeviceWorkspace mixer_workspace{};
  /* The mixer error scalar is tracing-only; #87 consumes staged statuses. */
  std::uint32_t* mixer_device_error = nullptr;
  std::int64_t mixer_device_error_elements = 0;
  Gfn2SccPublicationDeviceWorkspace publication_workspace{};

  std::uint64_t plan_token = 0u;
};

struct Gfn2SccIterationBinding {
  Gfn2SccIterationDevicePlan plan{};
  Gfn2SccIterationDeviceInput input{};
  Gfn2SccIterationDeviceState state{};
  Gfn2SccIterationDeviceWorkspace workspace{};
};

static_assert(std::is_trivially_copyable_v<Gfn2SccIterationBindingDiagnostic>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationBindingDiagnostic>);
static_assert(std::is_trivially_copyable_v<Gfn2SccIterationLaunchResult>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationLaunchResult>);
static_assert(std::is_trivially_copyable_v<Gfn2SccIterationCudaEigensolverProvider>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationCudaEigensolverProvider>);
static_assert(std::is_trivially_copyable_v<Gfn2SccIterationDeviceComponentStorage>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationDeviceComponentStorage>);
static_assert(std::is_trivially_copyable_v<Gfn2SccIterationDeviceElectronicEnergyInput>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationDeviceElectronicEnergyInput>);
static_assert(std::is_trivially_copyable_v<Gfn2SccIterationDeviceScalarBridge>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationDeviceScalarBridge>);
static_assert(std::is_trivially_copyable_v<Gfn2SccIterationDevicePlan>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationDevicePlan>);
static_assert(std::is_trivially_copyable_v<Gfn2SccIterationDeviceInput>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationDeviceInput>);
static_assert(std::is_trivially_copyable_v<Gfn2SccIterationDeviceState>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationDeviceState>);
static_assert(std::is_trivially_copyable_v<Gfn2SccIterationDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationDeviceWorkspace>);
static_assert(std::is_trivially_copyable_v<Gfn2SccIterationBinding>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationBinding>);

/*
 * Inspect only host descriptor values and bucket records. Device arrays are
 * never dereferenced, and no CUDA API, allocation, transfer, synchronization,
 * or stream operation is performed. The result is therefore synchronous and
 * safe to call before capture.
 */
[[nodiscard]] Gfn2SccIterationBindingDiagnostic validate_gfn2_scc_iteration_binding_cuda(
    const Gfn2SccIterationDevicePlan& plan, const Gfn2SccIterationDeviceInput& input,
    const Gfn2SccIterationDeviceState& state,
    const Gfn2SccIterationDeviceWorkspace& workspace) noexcept;

/* Fail-closed setup builder. binding is zeroed unless every check succeeds. */
[[nodiscard]] Gfn2SccIterationBindingDiagnostic bind_gfn2_scc_iteration_cuda(
    const Gfn2SccIterationDevicePlan& plan, const Gfn2SccIterationDeviceInput& input,
    const Gfn2SccIterationDeviceState& state, const Gfn2SccIterationDeviceWorkspace& workspace,
    Gfn2SccIterationBinding& binding) noexcept;

/*
 * Advance one restricted GFN2 SCC iteration entirely on the caller stream.
 * binding must have been produced by bind_gfn2_scc_iteration_cuda and remain
 * immutable between launches; numerical buffers referenced by it may change.
 * The hot path allocates, transfers, rebuilds descriptors, polls, and
 * synchronizes nothing, and is suitable for CUDA Graph capture subject to the
 * setup-declared eigensolver provider capture mode.
 */
[[nodiscard]] Gfn2SccIterationLaunchResult launch_gfn2_restricted_scc_iteration_cuda(
    const Gfn2SccIterationBinding& binding, cudaStream_t stream = nullptr) noexcept;

/* Replay-safe iteration consuming numerical caches published for the device epoch. */
[[nodiscard]] Gfn2SccIterationLaunchResult launch_gfn2_restricted_scc_iteration_cuda(
    const Gfn2SccIterationBinding& binding,
    const Gfn2GeometryEpochConsumerDevice& geometry,
    cudaStream_t stream = nullptr) noexcept;

/*
 * Root phase of one SCC iteration. This derives the canonical active mask from
 * the published state and current geometry epoch, but launches no numerical
 * work. Keeping this phase separate lets a conditional Graph decide whether
 * the reusable numerical body must execute at all.
 */
[[nodiscard]] Gfn2SccIterationLaunchResult launch_gfn2_restricted_scc_activity_cuda(
    const Gfn2SccIterationBinding& binding, cudaStream_t stream = nullptr) noexcept;

[[nodiscard]] Gfn2SccIterationLaunchResult launch_gfn2_restricted_scc_activity_cuda(
    const Gfn2SccIterationBinding& binding,
    const Gfn2GeometryEpochConsumerDevice& geometry,
    cudaStream_t stream = nullptr) noexcept;

/*
 * Numerical body of one SCC iteration. The caller must have enqueued the root
 * activity phase first on the same ordered stream. The body consumes the
 * canonical ledger, publishes at most one transition, and never re-derives
 * activity itself. It is capture-safe when the setup-declared provider mode is
 * kGraphSupported.
 */
[[nodiscard]] Gfn2SccIterationLaunchResult launch_gfn2_restricted_scc_numerical_body_cuda(
    const Gfn2SccIterationBinding& binding, cudaStream_t stream = nullptr) noexcept;

[[nodiscard]] Gfn2SccIterationLaunchResult launch_gfn2_restricted_scc_numerical_body_cuda(
    const Gfn2SccIterationBinding& binding,
    const Gfn2GeometryEpochConsumerDevice& geometry,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_SCC_ITERATION_CUH
