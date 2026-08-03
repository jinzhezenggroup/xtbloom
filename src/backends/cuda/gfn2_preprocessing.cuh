#ifndef GPUXTB_BACKENDS_CUDA_GFN2_PREPROCESSING_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_PREPROCESSING_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_aes2.cuh"
#include "backends/cuda/gfn2_es2.cuh"
#include "backends/cuda/gfn2_geometry.cuh"
#include "backends/cuda/gfn2_integrals.cuh"

namespace gpuxtb::detail::cuda {

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
  /* The embedded scalar generations name the latest attempted refresh. They
   * may advance before asynchronous plan diagnostics are known. Consumers must
   * use operator_generations and published_mask as the per-peer commit record. */
  Gfn2ES2DeviceCache es2{};
  Gfn2AES2DeviceCache aes2{};
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

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_PREPROCESSING_CUH
