#ifndef GPUXTB_BACKENDS_CUDA_GFN2_EIGENSOLVER_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_EIGENSOLVER_CUH

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace gpuxtb::detail::cuda {

/* Per-system numerical and semantic diagnostics produced asynchronously. */
enum class Gfn2EigensolverDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidActiveMask = 1u,
  kInvalidOffsets = 2u,
  kInvalidBucketMap = 3u,
  kNonfiniteOverlap = 4u,
  kNonsymmetricOverlap = 5u,
  kOverlapConditionEstimateFailed = 6u,
  kOverlapNotPositiveDefinite = 7u,
  kOverlapIllConditioned = 8u,
  kStaleOverlapCache = 9u,
  kNonfiniteHamiltonian = 10u,
  kNonsymmetricHamiltonian = 11u,
  kEigensolverFailed = 12u,
  kNonfiniteEigenpair = 13u,
};

/* Synchronous launcher/provider diagnostics; device failures remain per system. */
enum class Gfn2EigensolverLaunchStatus : std::uint32_t {
  kSuccess = 0u,
  kInvalidArgument = 1u,
  kCudaError = 2u,
  kCublasError = 3u,
  kCusolverError = 4u,
};

struct Gfn2EigensolverLaunchResult {
  Gfn2EigensolverLaunchStatus status = Gfn2EigensolverLaunchStatus::kSuccess;
  cudaError_t cuda_status = cudaSuccess;
  cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

  [[nodiscard]] bool success() const noexcept {
    return status == Gfn2EigensolverLaunchStatus::kSuccess;
  }
};

/*
 * One homogeneous AO-size bucket. Buckets and their scalar fields live on the
 * host and are immutable after setup. system_index_offset addresses the device
 * bucket_systems array. Matrix/orbital scratch offsets address bucket-packed
 * caller storage; each bucket is tightly strided by n*n and n respectively.
 */
struct Gfn2EigensolverBucket {
  std::int32_t orbital_count = 0;
  std::int32_t system_count = 0;
  std::int64_t system_index_offset = 0;
  std::int64_t matrix_scratch_offset = 0;
  std::int64_t orbital_scratch_offset = 0;
};

/*
 * Ragged public layout plus the device permutation into homogeneous buckets.
 * Public H, S, and C matrices are system-major dense row-major. C is AO by MO.
 * bucket_systems contains every system exactly once in a setup-validated plan.
 */
struct Gfn2EigensolverDeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t total_matrix_elements = 0;
  std::int64_t orbital_offset_count = 0;
  std::int64_t matrix_offset_count = 0;
  std::int64_t bucket_system_count = 0;
  std::int64_t active_elements = 0;
  std::uint64_t plan_token = 0u;

  const std::int64_t* orbital_offsets = nullptr;
  const std::int64_t* matrix_offsets = nullptr;
  const std::int32_t* bucket_systems = nullptr;
  const std::uint8_t* active = nullptr;
};

/* Geometry-persistent, bucket-packed lower Cholesky factors and metadata. */
struct Gfn2EigensolverOverlapCache {
  double* cholesky_factors = nullptr;
  std::int64_t factor_elements = 0;
  std::uint64_t* geometry_generations = nullptr;
  std::int64_t generation_elements = 0;
  std::uint32_t* factor_statuses = nullptr;
  std::int64_t status_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Transactional ragged outputs. Failed and inactive members remain untouched. */
struct Gfn2EigensolverDeviceResults {
  double* eigenvalues = nullptr;
  std::int64_t eigenvalue_elements = 0;
  double* coefficients = nullptr;
  std::int64_t coefficient_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Caller-owned unpublished storage. matrix_scratch_a and matrix_scratch_b each
 * contain total_matrix_elements bucket-packed doubles. eigenvalue_scratch is
 * similarly bucket-packed. Pointer arrays are device arrays consumed by
 * potrfBatched/trsmBatched. sequence_active captures the sticky device-error
 * state at call entry so a pre-existing global failure makes the whole launch
 * fail closed without turning a later per-system numerical error into a batch
 * failure. The generic cuSOLVER workspace may include both a device allocation
 * and pinned/pageable host storage, allocated during setup.
 */
struct Gfn2EigensolverDeviceWorkspace {
  double* matrix_scratch_a = nullptr;
  std::int64_t matrix_a_elements = 0;
  double* matrix_scratch_b = nullptr;
  std::int64_t matrix_b_elements = 0;
  double* eigenvalue_scratch = nullptr;
  std::int64_t eigenvalue_elements = 0;
  double** factor_pointers = nullptr;
  std::int64_t factor_pointer_elements = 0;
  double** matrix_pointers = nullptr;
  std::int64_t matrix_pointer_elements = 0;
  int* info_a = nullptr;
  std::int64_t info_a_elements = 0;
  int* info_b = nullptr;
  std::int64_t info_b_elements = 0;
  std::uint8_t* eligible = nullptr;
  std::int64_t eligible_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_active_elements = 0;
  void* solver_device_workspace = nullptr;
  std::size_t solver_device_workspace_bytes = 0u;
  void* solver_host_workspace = nullptr;
  std::size_t solver_host_workspace_bytes = 0u;
  std::uint64_t plan_token = 0u;
};

struct Gfn2EigensolverOptions {
  /* Exact eigenvalue ratio threshold for the symmetric positive overlap. */
  double minimum_overlap_rcond = 1.0e-12;
  /* Matches the CPU symmetric-input acceptance envelope by default. */
  double symmetry_tolerance = 64.0 * 2.220446049250313080847263336181640625e-16;
  /* Requests pedantic cuBLAS math; cuSOLVER remains fixed-algorithm per toolkit. */
  bool deterministic_debug = false;
};

/* Controls whether a failed factorization invalidates cache metadata. Initial
 * setup records the failure so an uninitialized cache cannot be consumed;
 * geometry refresh keeps the last valid generation/status while still
 * reporting the new failure through system_errors. */
enum class Gfn2EigensolverFactorCachePolicy : std::uint32_t {
  kPublishFailure = 0u,
  kPreservePriorOnFailure = 1u,
};

struct Gfn2EigensolverWorkspaceRequirements {
  std::size_t solver_device_workspace_bytes = 0u;
  std::size_t solver_host_workspace_bytes = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2EigensolverBucket>);
static_assert(std::is_standard_layout_v<Gfn2EigensolverBucket>);
static_assert(std::is_trivially_copyable_v<Gfn2EigensolverDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2EigensolverDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2EigensolverOverlapCache>);
static_assert(std::is_standard_layout_v<Gfn2EigensolverOverlapCache>);
static_assert(std::is_trivially_copyable_v<Gfn2EigensolverDeviceResults>);
static_assert(std::is_standard_layout_v<Gfn2EigensolverDeviceResults>);
static_assert(std::is_trivially_copyable_v<Gfn2EigensolverDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2EigensolverDeviceWorkspace>);

/*
 * Setup-time workspace query for the largest bucket requirement. Query every
 * bucket and retain the componentwise maximum. No numerical work is enqueued.
 */
Gfn2EigensolverLaunchResult query_gfn2_eigensolver_bucket_workspace_cuda(
    cusolverDnHandle_t solver, cusolverDnParams_t parameters, const Gfn2EigensolverBucket& bucket,
    const double* device_matrix, const double* device_eigenvalues,
    Gfn2EigensolverWorkspaceRequirements& requirements) noexcept;

/* Clear per-system and sticky diagnostics asynchronously. */
cudaError_t reset_gfn2_eigensolver_device_errors_cuda(std::int64_t batch_size,
                                                      std::uint32_t* system_errors,
                                                      std::uint32_t* device_error,
                                                      cudaStream_t stream = nullptr) noexcept;

/*
 * Update reusable overlap factors for geometry_generation. The overlap is
 * checked for finiteness, symmetry, positive definiteness, and the configured
 * exact 2-norm reciprocal condition threshold. Healthy members commit their
 * lower factor and generation independently. kPublishFailure records the
 * attempted generation/error for failed members, while
 * kPreservePriorOnFailure retains their previous valid cache metadata.
 */
Gfn2EigensolverLaunchResult factor_gfn2_overlap_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const double* overlap, std::uint64_t geometry_generation,
    const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver, cusolverDnParams_t parameters,
    const Gfn2EigensolverDeviceWorkspace& workspace, const Gfn2EigensolverOverlapCache& cache,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream = nullptr,
    Gfn2EigensolverFactorCachePolicy cache_policy =
        Gfn2EigensolverFactorCachePolicy::kPublishFailure) noexcept;

/*
 * Solve H C = S C eps for every active member whose cache generation matches.
 * Each bucket performs L^-1 H L^-T, a symmetric eigensolve, and L^-T Q. No
 * inverse is formed. Results publish per system only after provider info and
 * all eigenpair values are valid.
 *
 * The numerical path allocates nothing, transfers nothing, never synchronizes,
 * and uses only stream. Handles/parameters/workspaces are caller-owned and may
 * not be shared concurrently. CUDA Graph capture support depends on the linked
 * cuSOLVER provider; a provider rejection is returned synchronously and leaves
 * the setup/query layer usable for an explicitly uncaptured solver segment.
 */
Gfn2EigensolverLaunchResult solve_gfn2_eigensystems_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const Gfn2EigensolverOverlapCache& cache,
    std::uint64_t geometry_generation, const double* hamiltonians,
    const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver, cusolverDnParams_t parameters,
    cublasHandle_t blas, const Gfn2EigensolverDeviceWorkspace& workspace,
    const Gfn2EigensolverDeviceResults& results, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_EIGENSOLVER_CUH
