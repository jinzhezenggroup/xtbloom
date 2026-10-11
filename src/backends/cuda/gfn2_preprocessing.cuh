#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_PREPROCESSING_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN2_PREPROCESSING_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_aes2.cuh"
#include "backends/cuda/gfn2_device_admission.cuh"
#include "backends/cuda/gfn2_es2.cuh"
#include "backends/cuda/gfn2_geometry.cuh"
#include "backends/cuda/gfn2_integrals.cuh"
#include "backends/cuda/gfn2_native_periodic_integrals.cuh"
#include "backends/cuda/gfn2_native_periodic_short_range.cuh"
#include "backends/cuda/gfn2_pairlist.cuh"

namespace xtbloom::detail::cuda {

inline constexpr std::uint32_t kGfn2PreprocessingAbiVersion = 1u;

/* Synchronous descriptor/provenance failures detected before any launch. */
enum class Gfn2PreprocessingBindingError : std::uint32_t {
  kSuccess = 0u,
  kInvalidAbi = 1u,
  kInvalidPlanToken = 2u,
  kCrossPlan = 3u,
  kInvalidExtent = 4u,
  kInvalidPointer = 5u,
  kInvalidAlias = 6u,
  kInvalidActivity = 7u,
  kInvalidDiagnostics = 8u,
  kInvalidWorkspace = 9u,
  kUnsealedBinding = 10u,
  kStaleSeal = 11u,
  kInvalidGeneration = 12u,
  kInvalidEpoch = 13u,
};

enum class Gfn2PreprocessingBindingField : std::uint32_t {
  kNone = 0u,
  kBinding = 1u,
  kPlan = 2u,
  kGeometry = 3u,
  kIntegrals = 4u,
  kH0 = 5u,
  kEs2 = 6u,
  kAes2 = 7u,
  kPositions = 8u,
  kOutput = 9u,
  kActivity = 10u,
  kDiagnostics = 11u,
  kWorkspace = 12u,
  kSeal = 13u,
  kGeneration = 14u,
  kEpoch = 15u,
  kPairlist = 16u,
};

struct Gfn2PreprocessingBindingDiagnostic {
  Gfn2PreprocessingBindingError error = Gfn2PreprocessingBindingError::kSuccess;
  Gfn2PreprocessingBindingField field = Gfn2PreprocessingBindingField::kNone;
  std::int64_t index = -1;

  [[nodiscard]] bool success() const noexcept {
    return error == Gfn2PreprocessingBindingError::kSuccess;
  }
};

/* First plan-wide asynchronous failure. Public numerical caches are untouched. */
enum class Gfn2PreprocessingDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidActivity = 1u,
  kInvalidComposerOffsets = 2u,
  kGeometryPlanFailure = 3u,
  kIntegralPlanFailure = 4u,
  kEs2PlanFailure = 5u,
  kAes2PlanFailure = 6u,
  kGeometryEpochOverflow = 7u,
  /* The sparse pair-list builder failed or its coordination numbers disagreed
   * bitwise with the dense geometry cache for a healthy peer.  Either event
   * fails that peer closed so a sparse/dense regression can never silently
   * publish different physics. */
  kSparsePairlistFailure = 8u,
  kSparseCoordinationMismatch = 9u,
};

/* Per-system summary; primitive-domain codes remain available in diagnostics. */
enum class Gfn2PreprocessingSystemStage : std::uint32_t {
  kSuccess = 0u,
  kGeometry = 1u,
  kIntegralsOrH0 = 2u,
  kAes2 = 3u,
};

/*
 * Immutable CUDA plan leaves. Canonical topology projections deliberately
 * share atom/shell/pair offset pointers across primitive descriptors; the
 * validator rejects equal-sized descriptors copied from another plan.
 */
struct Gfn2PreprocessingDevicePlan {
  std::uint32_t abi_version = kGfn2PreprocessingAbiVersion;
  std::uint32_t reserved = 0u;
  Gfn2GeometryDeviceBatch geometry{};
  Gfn2IntegralDeviceBatch integrals{};
  Gfn2H0DevicePlan h0{};
  Gfn2ES2DeviceBatch es2{};
  Gfn2AES2DeviceBatch aes2{};
  /* Native XYZ periodic S/D/Q integrals and image-dependent H0. */
  Gfn2NativePeriodicIntegralDeviceBatch native_integrals{};
  /* Native XYZ periodic short-range terms replace the molecular geometry CN
   * for lattice requests.  The descriptor is immutable; positions are
   * rebound to the refresh candidate at composition time. */
  Gfn2NativePeriodicShortRangeDeviceBatch native_short_range{};
  /* Optional sparse pair-list view selected by the host dispatch policy.  When
   * batch_size is zero the sparse CN consistency check is disabled and all
   * pairlist pointers must be null.  When enabled it shares the canonical
   * atom_offsets/covalent_radii pointers with geometry and the sparse bucketed
   * builder produces a coordination_numbers check that must match the dense
   * geometry cache bitwise for every healthy peer. */
  Gfn2PairListDeviceBatch pairlist{};
  std::uint64_t plan_token = 0u;
};

struct Gfn2PreprocessingDeviceInput {
  const double* positions = nullptr;
  std::int64_t position_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * requested_mask is immutable caller intent. published_mask is overwritten on
 * stream completion: one means every public operator/cache slice for that
 * system was committed for the requested generation. Inactive members retain
 * all public bytes and report a zero stage code.
 */
struct Gfn2PreprocessingDeviceActivity {
  const std::uint8_t* requested_mask = nullptr;
  std::int64_t requested_elements = 0;
  std::uint8_t* published_mask = nullptr;
  std::int64_t published_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Complete public geometry/operator/cache state consumed by SCC and forces. */
struct Gfn2PreprocessingDeviceOutput {
  Gfn2GeometryDeviceCache geometry{};
  double* overlap = nullptr;
  std::int64_t overlap_elements = 0;
  double* dipole_integrals = nullptr;
  std::int64_t dipole_elements = 0;
  double* quadrupole_integrals = nullptr;
  std::int64_t quadrupole_elements = 0;
  double* h0 = nullptr;
  std::int64_t h0_elements = 0;
  /* In the legacy scalar API these embedded generations name the latest
   * attempted refresh and may advance before asynchronous diagnostics are
   * known. The device-epoch API deliberately leaves them unchanged because a
   * host descriptor cannot change during Graph replay. All consumers of that
   * path must use operator_generations and published_mask as the commit record. */
  Gfn2ES2DeviceCache es2{};
  Gfn2AES2DeviceCache aes2{};
  /* Committed sparse pair-list consumer view (ABI step 4).  Storage is laid
   * out with fixed per-system capacity (system s owns a slice starting at
   * s*max_pairs_per_system; per-atom neighbors likewise) so a failed peer
   * never shifts a later peer's slice.  The final per-system publication gate
   * copies candidate pairs/neighbors, writes explicit counts, sets the
   * committed generation, and flips eligibility so consumers accept only
   * complete, current per-peer slices.  A peer-local failed or inactive
   * refresh clears current eligibility but preserves the last committed
   * payload; a plan-wide failure preserves every committed byte. Setup
   * initializes never-published metadata to zero. */
  Gfn2PairListConsumerView pairlist{};
  /* Per-system generation for the complete S/D/Q/H0/ES2/AES2 transaction. */
  std::uint64_t* operator_generations = nullptr;
  std::int64_t generation_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Primitive diagnostics are intentionally kept in their native domains. The
 * two composer summaries are the only diagnostics a pipeline controller needs
 * to inspect; they are still device-resident and require no hot-path polling.
 */
struct Gfn2PreprocessingDeviceDiagnostics {
  std::uint32_t* geometry_system_errors = nullptr;
  std::int64_t geometry_system_elements = 0;
  std::uint32_t* geometry_device_error = nullptr;
  std::uint32_t* integral_system_errors = nullptr;
  std::int64_t integral_system_elements = 0;
  std::uint32_t* integral_device_error = nullptr;
  std::uint32_t* es2_device_error = nullptr;
  std::uint32_t* aes2_system_errors = nullptr;
  std::int64_t aes2_system_elements = 0;
  std::uint32_t* aes2_device_error = nullptr;
  std::uint32_t* system_stages = nullptr;
  std::int64_t system_stage_elements = 0;
  std::uint32_t* plan_error = nullptr;
  /* Sparse pair-list domain.  The bucketed builder and coordination evaluator
   * report here, then the consistency gate folds any mismatch into the
   * geometry system errors so the existing publication gate rejects the peer.
   * Both buffers must be present when plan.pairlist is enabled. */
  std::uint32_t* sparse_system_errors = nullptr;
  std::int64_t sparse_system_elements = 0;
  std::uint32_t* sparse_device_error = nullptr;
  std::uint64_t plan_token = 0u;
};

/*
 * Every numerical primitive publishes into candidate storage below. The final
 * composer kernel copies a complete peer slice to Gfn2PreprocessingDeviceOutput
 * only after all dependent stages succeed. positions_scratch first masks
 * inactive peers and later replaces failed peers before the batch-transactional
 * ES2 primitive, without rewriting its numerical kernel.
 */
struct Gfn2PreprocessingDeviceWorkspace {
  double* positions_scratch = nullptr;
  std::int64_t position_elements = 0;

  Gfn2GeometryDeviceCache geometry_candidate{};
  Gfn2GeometryDeviceWorkspace geometry{};

  double* overlap_candidate = nullptr;
  std::int64_t overlap_elements = 0;
  double* dipole_candidate = nullptr;
  std::int64_t dipole_elements = 0;
  double* quadrupole_candidate = nullptr;
  std::int64_t quadrupole_elements = 0;
  double* h0_candidate = nullptr;
  std::int64_t h0_elements = 0;
  Gfn2IntegralDeviceWorkspace integrals{};

  Gfn2ES2DeviceCache es2_candidate{};
  Gfn2ES2DeviceWorkspace es2{};
  Gfn2AES2DeviceCache aes2_candidate{};
  Gfn2AES2DeviceWorkspace aes2{};
  /* Reuses the SCC arena's fixed native workspace.  The preprocessing
   * composer writes CN and repulsion candidates before H0/integral consumers
   * run, so no separate native-periodic allocation is needed here. */
  Gfn2NativePeriodicShortRangeDeviceWorkspace native_short_range{};
  /* Sparse pair-list candidate state; only valid when the plan leaf is enabled
   * and its host scheduler has provisioned these buffers.  The sparse
   * coordination check writes here before the dense/parity gate publishes. */
  Gfn2PairListDeviceCache pairlist_candidate{};
  Gfn2PairListDeviceWorkspace pairlist{};
  double* sparse_coordination = nullptr;
  std::int64_t sparse_coordination_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * A sealed binding is stable across repeated calls and CUDA Graph replay.
 * geometry_generation inside the ES2/AES2 public and candidate descriptors is
 * dynamic attempted-refresh metadata and is therefore excluded from the seal;
 * device-resident per-peer generations remain authoritative for publication.
 * All pointers, extents, topology leaves, and tokens are covered.
 */
struct Gfn2PreprocessingDeviceBinding {
  Gfn2PreprocessingDevicePlan plan{};
  Gfn2PreprocessingDeviceInput input{};
  Gfn2PreprocessingDeviceActivity activity{};
  Gfn2PreprocessingDeviceOutput output{};
  Gfn2PreprocessingDeviceDiagnostics diagnostics{};
  Gfn2PreprocessingDeviceWorkspace workspace{};
  /* Optional for the scalar API and required by the replay-safe epoch API.
   * The descriptor is covered by binding_seal, including its stable address. */
  Gfn2GeometryEpochDevice geometry_epoch{};
  Gfn2DeviceAdmission admission{};
  std::uint64_t binding_seal = 0u;
  std::uint64_t plan_token = 0u;
};

struct Gfn2PreprocessingLaunchDiagnostic {
  Gfn2PreprocessingBindingDiagnostic binding{};
  cudaError_t cuda_status = cudaSuccess;

  [[nodiscard]] bool success() const noexcept {
    return binding.success() && cuda_status == cudaSuccess;
  }
};

static_assert(std::is_trivially_copyable_v<Gfn2PreprocessingDevicePlan>);
static_assert(std::is_standard_layout_v<Gfn2PreprocessingDevicePlan>);
static_assert(std::is_trivially_copyable_v<Gfn2PreprocessingDeviceInput>);
static_assert(std::is_standard_layout_v<Gfn2PreprocessingDeviceInput>);
static_assert(std::is_trivially_copyable_v<Gfn2PreprocessingDeviceActivity>);
static_assert(std::is_standard_layout_v<Gfn2PreprocessingDeviceActivity>);
static_assert(std::is_trivially_copyable_v<Gfn2PreprocessingDeviceOutput>);
static_assert(std::is_standard_layout_v<Gfn2PreprocessingDeviceOutput>);
static_assert(std::is_trivially_copyable_v<Gfn2PreprocessingDeviceDiagnostics>);
static_assert(std::is_standard_layout_v<Gfn2PreprocessingDeviceDiagnostics>);
static_assert(std::is_trivially_copyable_v<Gfn2PreprocessingDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2PreprocessingDeviceWorkspace>);
static_assert(std::is_trivially_copyable_v<Gfn2PreprocessingDeviceBinding>);
static_assert(std::is_standard_layout_v<Gfn2PreprocessingDeviceBinding>);
static_assert(std::is_trivially_copyable_v<Gfn2PreprocessingLaunchDiagnostic>);
static_assert(std::is_standard_layout_v<Gfn2PreprocessingLaunchDiagnostic>);

/* Validate a binding without touching CUDA state. A valid unsealed binding is
 * reported as kUnsealedBinding; seal_gfn2_preprocessing_binding_cuda performs
 * the same validation and publishes the owner-controlled descriptor seal. */
[[nodiscard]] Gfn2PreprocessingBindingDiagnostic validate_gfn2_preprocessing_binding_cuda(
    const Gfn2PreprocessingDeviceBinding& binding) noexcept;

[[nodiscard]] Gfn2PreprocessingBindingDiagnostic seal_gfn2_preprocessing_binding_cuda(
    Gfn2PreprocessingDeviceBinding& binding) noexcept;

/*
 * Enqueue positions -> geometry/CN -> S/D/Q -> H0 -> ES2/AES2 on stream.
 * The call allocates, transfers, synchronizes, and polls nowhere. Pointer
 * provenance must be established by the owner before sealing. On every
 * synchronous rejection no numerical public cache is touched; asynchronous
 * plan failures reach the final publication gate and likewise commit nothing.
 */
[[nodiscard]] Gfn2PreprocessingLaunchDiagnostic compose_gfn2_preprocessing_cuda(
    Gfn2PreprocessingDeviceBinding& binding, std::uint64_t geometry_generation,
    cudaStream_t stream = nullptr) noexcept;

/*
 * Replay-safe variant. Its first queued kernel advances binding.geometry_epoch
 * once, and every publication kernel reads that device value. The binding,
 * arenas, and Graph executable remain unchanged across inference calls.
 * Healthy peers publish the new epoch; failed/inactive peers retain their
 * prior cache generations. Epoch overflow fails the complete plan closed.
 */
[[nodiscard]] Gfn2PreprocessingLaunchDiagnostic compose_gfn2_preprocessing_epoch_cuda(
    Gfn2PreprocessingDeviceBinding& binding, cudaStream_t stream = nullptr) noexcept;

/*
 * Evaluate the complete native-periodic one-electron transaction.  Unlike the
 * molecular evaluator this sums the setup-owned Gaussian image list, applies
 * the image H0 spatial factor to every translated shell pair, and publishes
 * only complete peer slices.  The caller normally reaches this through
 * compose_gfn2_preprocessing_*; the entry point is public to the CUDA test
 * harness so the periodic leaf can be validated independently of SCC.
 */
[[nodiscard]] cudaError_t evaluate_gfn2_native_periodic_integrals_h0_cuda(
    const Gfn2IntegralDeviceBatch& batch, const Gfn2H0DevicePlan& h0,
    const Gfn2NativePeriodicIntegralDeviceBatch& periodic, const double* wrapped_positions,
    const double* coordination_numbers, double* overlap, double* dipole, double* quadrupole,
    double* hamiltonian, const Gfn2IntegralDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

/*
 * Run only the sparse/dense CN consistency gate over an already-composed
 * binding.  Compares the dense geometry-candidate coordination numbers with
 * the caller-populated workspace.sparse_coordination and, on any bitwise
 * disagreement, records kSparseCoordinationMismatch in the peer's geometry
 * error slot so the existing publication gate rejects it.  This entry lets a
 * host test corrupt the sparse output after evaluate but before the gate, to
 * prove the fail-closed path deterministically.
 */
[[nodiscard]] Gfn2PreprocessingLaunchDiagnostic gate_gfn2_sparse_coordination_cuda(
    Gfn2PreprocessingDeviceBinding& binding, cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_PREPROCESSING_CUH
