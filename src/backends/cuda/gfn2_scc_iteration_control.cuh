#ifndef GPUXTB_BACKENDS_CUDA_GFN2_SCC_ITERATION_CONTROL_CUH
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_BACKENDS_CUDA_GFN2_SCC_ITERATION_CONTROL_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/common/gfn2_plan_schema.hpp"
#include "backends/cuda/gfn2_geometry.cuh"
#include "backends/cuda/gfn2_scc.cuh"

namespace gpuxtb::detail::cuda {

/*
 * Stable internal stage identities for the SCC iteration ledger. Primitive
 * error numbers are meaningful only together with one of these identities.
 */
enum class Gfn2SccStageId : std::uint32_t {
  kNone = 0u,
  kActivity = 1u,
  kWarmStartProvenance = 2u,
  kGeometry = 3u,
  kMixedGather = 4u,
  kES2Potential = 5u,
  kES3Potential = 6u,
  kAES2Potential = 7u,
  kD4Potential = 8u,
  kExplicitPointChargePotential = 9u,
  kPeriodicPotential = 10u,
  kPotentialCompose = 11u,
  kScalarBridge = 12u,
  kHamiltonian = 13u,
  kOverlapFactor = 14u,
  kEigensolver = 15u,
  kOccupations = 16u,
  kDensity = 17u,
  kMulliken = 18u,
  kClassicalEnergy = 19u,
  kElectronicEnergy = 20u,
  kFreeEnergy = 21u,
  kMixer = 22u,
  kStatePublication = 23u,
  // Raw classical-energy evaluation is a distinct DAG phase from potential
  // construction even when both phases reuse one primitive's diagnostics.
  // Append new identities so every pre-existing stage value remains stable.
  kES2RawEnergy = 24u,
  kES3RawEnergy = 25u,
  kAES2RawEnergy = 26u,
  kD4RawEnergy = 27u,
  kExplicitPointChargeRawEnergy = 28u,
  kPeriodicRawEnergy = 29u,
  // Spin stages are appended so all restricted-stage identities above remain
  // stable for persisted diagnostics and existing Graph/report consumers.
  kSpinPotential = 30u,
  kSpinRawEnergy = 31u,
};

/* True for every enumerator, including the non-executable kNone sentinel. */
[[nodiscard]] inline __host__ __device__ constexpr bool gfn2_scc_stage_id_in_domain(
    Gfn2SccStageId stage) noexcept {
  return static_cast<std::uint32_t>(stage) <=
         static_cast<std::uint32_t>(Gfn2SccStageId::kSpinRawEnergy);
}

/* Stage reports and cache owners must name one of the executable stages. */
[[nodiscard]] inline __host__ __device__ constexpr bool gfn2_scc_stage_id_is_valid(
    Gfn2SccStageId stage) noexcept {
  return stage != Gfn2SccStageId::kNone && gfn2_scc_stage_id_in_domain(stage);
}

/* Object representation of a stage-local per-system diagnostic array. */
enum class Gfn2SccStageCodeFormat : std::uint32_t {
  kUint32Error = 0u,
  kGpuxtbStatus = 1u,
};

/*
 * Interpretation of the primitive's sticky device-wide first-error scalar.
 * kMixedFirstError preserves the legacy contract: peer-mask membership makes
 * the scalar peer-local when it can be matched to an active system, while all
 * other nonzero values are plan failures. kPlanOnly prevents plan diagnostics
 * from being mistaken for peer errors when the two domains reuse an integer.
 * Per-system codes are classified by peer_error_mask under either role.
 */
enum class Gfn2SccStageDeviceCodeRole : std::uint32_t {
  kMixedFirstError = 0u,
  kPlanOnly = 1u,
};

[[nodiscard]] inline __host__ __device__ constexpr bool gfn2_scc_stage_device_code_role_is_valid(
    Gfn2SccStageDeviceCodeRole role) noexcept {
  return role == Gfn2SccStageDeviceCodeRole::kMixedFirstError ||
         role == Gfn2SccStageDeviceCodeRole::kPlanOnly;
}

/* Controller-owned codes stored under kActivity or kWarmStartProvenance. */
enum class Gfn2SccIterationControlCode : std::uint32_t {
  kSuccess = 0u,
  kInvalidState = 1u,
  kInvalidProvenance = 2u,
  kCrossPlan = 3u,
  kStaleGeneration = 4u,
  /* cudaGraphLaunch() failed in the device-resident SCC loop controller. */
  kDeviceGraphLaunchFailed = 5u,
};

/*
 * Fallback raw codes used only when a stage's own first-error scalar cannot
 * identify the plan failure precisely. They deliberately sit outside the
 * 1..63 peer-mask domain.
 */
inline constexpr std::uint32_t kGfn2SccStageMalformedReportCode = 0xfffffffdu;
inline constexpr std::uint32_t kGfn2SccStageUnlocalizedPeerCode = 0xfffffffeu;
inline constexpr std::uint32_t kGfn2SccStageSequenceClosedCode = 0xffffffffu;

[[nodiscard]] inline __host__ __device__ constexpr std::uint64_t gfn2_scc_stage_failure_record(
    Gfn2SccStageId stage, std::uint32_t raw_code) noexcept {
  return (static_cast<std::uint64_t>(stage) << 32u) | static_cast<std::uint64_t>(raw_code);
}

[[nodiscard]] inline __host__ __device__ constexpr Gfn2SccStageId gfn2_scc_failure_stage(
    std::uint64_t record) noexcept {
  return static_cast<Gfn2SccStageId>(record >> 32u);
}

[[nodiscard]] inline __host__ __device__ constexpr std::uint32_t gfn2_scc_failure_code(
    std::uint64_t record) noexcept {
  return static_cast<std::uint32_t>(record);
}

/* Exact CPU-driver activity policy for one reusable ragged batch. */
struct Gfn2SccIterationDevicePolicy {
  std::int64_t batch_size = 0;
  std::uint64_t maximum_iterations = 0u;
  std::uint64_t plan_token = 0u;
};

/* Read-only projection of the driver state fields used by the CPU predicate. */
struct Gfn2SccIterationDeviceStateInput {
  const std::uint64_t* iterations = nullptr;
  const gpuxtb_status_t* system_statuses = nullptr;
  const std::uint8_t* converged = nullptr;
  std::int64_t batch_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * One common cache-provenance view and the consumer stage that owns it. The
 * array of these records is uploaded once during setup and remains immutable.
 */
struct Gfn2SccCacheProvenanceBinding {
  Gfn2GeometryCacheProvenanceView provenance{};
  Gfn2SccStageId owner_stage = Gfn2SccStageId::kNone;
  std::uint32_t reserved = 0u;
};

/* Geometry/cache and optional warm-start provenance for activity derivation. */
struct Gfn2SccIterationDeviceProvenance {
  const Gfn2SccCacheProvenanceBinding* cache_bindings = nullptr;
  std::int64_t cache_binding_count = 0;
  std::uint64_t expected_geometry_generation = 0u;

  const std::uint64_t* warm_start_generations = nullptr;
  std::int64_t warm_start_elements = 0;
  std::uint64_t expected_warm_start_generation = 0u;

  std::uint64_t plan_token = 0u;
};

/*
 * Caller-owned canonical control ledger for one SCC iteration. The two
 * failure records store (stage_id << 32) | stage_local_code. sequence_active
 * is written to one by derive and may only transition to zero until the next
 * derive call.
 */
struct Gfn2SccIterationDeviceLedger {
  std::uint8_t* active_mask = nullptr;
  gpuxtb_status_t* pending_statuses = nullptr;
  std::uint64_t* system_failure_records = nullptr;
  std::uint64_t* plan_failure_record = nullptr;
  std::uint32_t* sequence_active = nullptr;

  std::int64_t batch_elements = 0;
  std::int64_t scalar_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Minimal read-only projection consumed by SCC numerical stages. Consumers
 * must test sequence_active and the member's active byte before reading any
 * numerical input. In particular, #91 uses this instead of mixer residual
 * diagnostics to decide whether a Broyden transition is requested.
 */
struct Gfn2SccIterationDeviceActivity {
  const std::uint8_t* active_mask = nullptr;
  const std::uint32_t* sequence_active = nullptr;
  std::int64_t batch_elements = 0;
  std::int64_t sequence_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * One stage's raw diagnostics. peer_error_mask classifies only codes 1..63;
 * bit N corresponds to raw code N. A classified peer summary must also be
 * localizable through system_codes, otherwise normalization fails closed as a
 * plan error. device_code_role determines whether peer_error_mask may classify
 * the device scalar or that scalar is plan-only. A non-peer device_error is
 * the stage's sticky first plan code and is preserved ahead of indexed system
 * plan codes. A closed stage_sequence_active still promotes a peer device_error
 * to a plan failure, preventing a later unclassified failure from being hidden
 * by an earlier peer error.
 */
struct Gfn2SccStageDeviceReport {
  Gfn2SccStageId stage = Gfn2SccStageId::kNone;
  Gfn2SccStageCodeFormat system_code_format = Gfn2SccStageCodeFormat::kUint32Error;

  const void* system_codes = nullptr;
  std::int64_t system_code_elements = 0;
  const std::uint32_t* device_error = nullptr;
  std::int64_t device_error_elements = 0;
  const std::uint32_t* stage_sequence_active = nullptr;
  std::int64_t stage_sequence_elements = 0;

  std::uint64_t peer_error_mask = 0u;
  gpuxtb_status_t peer_failure_status = GPUXTB_STATUS_INTERNAL_ERROR;
  std::uint64_t plan_token = 0u;

  // Kept at the tail so legacy aggregate initializers retain the mixed-first-
  // error behavior while callers migrate raw-energy reports to kPlanOnly.
  Gfn2SccStageDeviceCodeRole device_code_role = Gfn2SccStageDeviceCodeRole::kMixedFirstError;
};

static_assert(std::is_trivially_copyable_v<Gfn2SccIterationDevicePolicy>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationDevicePolicy>);
static_assert(std::is_trivially_copyable_v<Gfn2SccIterationDeviceStateInput>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationDeviceStateInput>);
static_assert(std::is_trivially_copyable_v<Gfn2SccCacheProvenanceBinding>);
static_assert(std::is_standard_layout_v<Gfn2SccCacheProvenanceBinding>);
static_assert(std::is_trivially_copyable_v<Gfn2SccIterationDeviceProvenance>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationDeviceProvenance>);
static_assert(std::is_trivially_copyable_v<Gfn2SccIterationDeviceLedger>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationDeviceLedger>);
static_assert(std::is_trivially_copyable_v<Gfn2SccIterationDeviceActivity>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationDeviceActivity>);
static_assert(std::is_trivially_copyable_v<Gfn2SccStageDeviceReport>);
static_assert(std::is_standard_layout_v<Gfn2SccStageDeviceReport>);

/*
 * Reset the canonical ledger and derive the exact CPU active predicate from
 * Gfn2SccDeviceState: SUCCESS, not converged, and iterations below the limit.
 * Active members additionally undergo cache and optional warm-start generation
 * validation; inactive generation slots are not read. The launcher performs
 * no allocation, transfer, host polling, or synchronization and is Graph safe.
 */
cudaError_t derive_gfn2_scc_iteration_activity_cuda(
    const Gfn2SccIterationDevicePolicy& policy, const Gfn2SccIterationDeviceStateInput& state,
    const Gfn2SccIterationDeviceProvenance& provenance, const Gfn2SccIterationDeviceLedger& ledger,
    cudaStream_t stream = nullptr) noexcept;

/*
 * Replay-safe activity derivation.  Active SCC members must additionally be
 * eligible in the numerical-refresh transaction and have every cache bound to
 * the current device epoch.  The scalar overload above remains the setup-time
 * and legacy path.
 */
cudaError_t derive_gfn2_scc_iteration_activity_cuda(
    const Gfn2SccIterationDevicePolicy& policy, const Gfn2SccIterationDeviceStateInput& state,
    const Gfn2SccIterationDeviceProvenance& provenance,
    const Gfn2GeometryEpochConsumerDevice& geometry, const Gfn2SccIterationDeviceLedger& ledger,
    cudaStream_t stream = nullptr) noexcept;

/*
 * Fold one completed stage into the canonical ledger. Plan classification has
 * precedence over peer publication. A plan failure closes the complete
 * remaining sequence; peer failures disable only their active members. Raw
 * stage diagnostics remain untouched for tracing.
 */
cudaError_t normalize_gfn2_scc_stage_cuda(const Gfn2SccStageDeviceReport& report,
                                          const Gfn2SccIterationDeviceLedger& ledger,
                                          cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_SCC_ITERATION_CONTROL_CUH
