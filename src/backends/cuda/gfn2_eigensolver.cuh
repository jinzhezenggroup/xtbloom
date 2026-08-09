#ifndef GPUXTB_BACKENDS_CUDA_GFN2_EIGENSOLVER_CUH
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_BACKENDS_CUDA_GFN2_EIGENSOLVER_CUH

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "backends/common/gfn2_plan_schema.hpp"
#include "backends/cuda/gfn2_geometry.cuh"
#include "runtime/nvidia_host_api.h"

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
  kInvalidSpinLayout = 14u,
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

  /*
   * Optional spin solve projection. Legacy restricted buckets leave these
   * fields zero. solve_count is sum(nspin) over the bucket's physical systems;
   * solve_index_offset addresses pointer/info/eligibility arrays, while the
   * spin scratch offsets address nspin-expanded matrices and eigenvalues.
   * Canonical work order is bucket system order, spin 0 then spin 1.
   */
  std::int32_t solve_count = 0;
  std::int64_t solve_index_offset = 0;
  std::int64_t spin_matrix_scratch_offset = 0;
  std::int64_t spin_orbital_scratch_offset = 0;
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
 * Device-visible per-bucket compaction telemetry. The three counts are
 * published by the same ordered Graph launch that performs the solve, so a
 * profiler or asynchronous diagnostic copy observes one coherent iteration.
 * Exact-capacity submission keeps submitted_eigensolver_count equal to
 * active_count; completed_count can be smaller when the provider reports a
 * failed eigenpair before the back transformation.
 */
struct Gfn2EigensolverBucketActivity {
  std::uint32_t active_count = 0u;
  std::uint32_t submitted_eigensolver_count = 0u;
  std::uint32_t submitted_backtransform_count = 0u;
  std::uint32_t completed_count = 0u;
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

  /*
   * Reusable active-set storage for exact-capacity Graph submission. Entries
   * are bucket-local but addressed through each bucket's system_index_offset.
   * compact_systems preserves canonical bucket order. compact_source_slots
   * records the packed matrix slot retained after an eigensolver failure so
   * the final TRSM and scatter never consume a failed peer. No field is
   * allocated or rebuilt by the hot path.
   */
  std::int32_t* compact_systems = nullptr;
  std::int64_t compact_system_elements = 0;
  std::int32_t* compact_source_slots = nullptr;
  std::int64_t compact_source_slot_elements = 0;
  Gfn2EigensolverBucketActivity* bucket_activity = nullptr;
  std::int64_t bucket_activity_elements = 0;
};

/* Internal-only provider selection. Production uses kAuto; focused tests may
 * pin one provider to compare dispatch candidates on identical inputs. */
enum class Gfn2EigensolverStrategy : std::uint32_t {
  kAuto = 0u,
  kBatchedDivideAndConquer = 1u,
  kBatchedJacobi = 2u,
  /* Low-level symmetric reduction plus a device-resident tridiagonal solve.
   * This avoids the CUDA 12.9 vector-capture cliff in large singleton buckets. */
  kTridiagonalBisection = 3u,
};

struct Gfn2EigensolverOptions {
  /* Exact eigenvalue ratio threshold for the symmetric positive overlap. */
  double minimum_overlap_rcond = 1.0e-12;
  /* Matches the CPU symmetric-input acceptance envelope by default. */
  double symmetry_tolerance = 64.0 * 2.220446049250313080847263336181640625e-16;
  /* Requests pedantic cuBLAS math; cuSOLVER remains fixed-algorithm per toolkit. */
  bool deterministic_debug = false;
  Gfn2EigensolverStrategy strategy = Gfn2EigensolverStrategy::kAuto;
  /* Borrowed setup-owned Jacobi configuration. A null handle keeps legacy
   * low-level callers on XsyevBatched even when they use kAuto. */
  syevjInfo_t jacobi = nullptr;
};

/* The production crossover is intentionally narrow. On RTX 5090 / CUDA 12.9,
 * Jacobi reduces Graph-replayed provider latency through 16 AOs, while public
 * FRESH latency does not improve below 12 AOs and the provider loses by 24. */
inline constexpr std::int32_t kGfn2JacobiMinimumOrbitals = 12;
inline constexpr std::int32_t kGfn2JacobiMaximumOrbitals = 16;
/* Tests and the dispatch benchmark may force the provider through its
 * documented small-matrix limit to keep the rejected crossover measurable. */
inline constexpr std::int32_t kGfn2JacobiProviderMaximumOrbitals = 32;

/* CUDA 12.9 changes the vector-mode XsyevBatched implementation at 513 AOs;
 * the larger implementation is not stream-capturable. The custom path is
 * bounded to the measured issue-264 regime so unrelated batch dispatch and
 * very-large-system policy remain explicit. */
inline constexpr std::int32_t kGfn2TridiagonalMinimumOrbitals = 513;
inline constexpr std::int32_t kGfn2TridiagonalMaximumOrbitals = 1024;

[[nodiscard]] inline bool gfn2_eigensolver_uses_jacobi(const Gfn2EigensolverOptions& options,
                                                       std::int32_t orbital_count) noexcept {
  return options.strategy == Gfn2EigensolverStrategy::kBatchedJacobi ||
         (options.strategy == Gfn2EigensolverStrategy::kAuto && options.jacobi != nullptr &&
          orbital_count >= kGfn2JacobiMinimumOrbitals &&
          orbital_count <= kGfn2JacobiMaximumOrbitals);
}

[[nodiscard]] inline bool gfn2_eigensolver_uses_tridiagonal(
    const Gfn2EigensolverOptions& options, const Gfn2EigensolverBucket& policy_bucket) noexcept {
  if (policy_bucket.orbital_count <= 0 ||
      policy_bucket.orbital_count > kGfn2TridiagonalMaximumOrbitals) {
    return false;
  }
  return options.strategy == Gfn2EigensolverStrategy::kTridiagonalBisection ||
         (options.strategy == Gfn2EigensolverStrategy::kAuto && policy_bucket.system_count == 1 &&
          policy_bucket.orbital_count >= kGfn2TridiagonalMinimumOrbitals);
}

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
static_assert(std::is_trivially_copyable_v<Gfn2EigensolverBucketActivity>);
static_assert(std::is_standard_layout_v<Gfn2EigensolverBucketActivity>);
static_assert(std::is_trivially_copyable_v<Gfn2EigensolverDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2EigensolverDeviceWorkspace>);

/*
 * Reusable exact-capacity solve graph. Build is a setup-time operation and may
 * create CUDA Graph objects and a temporary capture stream. launch() only
 * enqueues the already-instantiated graph: it allocates, transfers, polls, and
 * synchronizes nothing. All descriptors, provider handles, pointer targets,
 * and workspaces supplied at build time must remain alive and at stable
 * addresses until the owner is destroyed.
 *
 * The graph uses one device-selected SWITCH per numerical submission. Body n
 * calls cuBLAS/cuSOLVER with batchCount exactly n; body zero launches no
 * provider arithmetic. Bucket order and the within-bucket canonical order are
 * deterministic for every launch.
 */
class Gfn2EigensolverCompactedSolveGraph {
 public:
  /* Public only so the translation-unit build helper can construct a graph
   * transaction without exposing CUDA Graph handles in this header. */
  struct Impl;

  Gfn2EigensolverCompactedSolveGraph() noexcept;
  ~Gfn2EigensolverCompactedSolveGraph();
  Gfn2EigensolverCompactedSolveGraph(Gfn2EigensolverCompactedSolveGraph&&) noexcept;
  Gfn2EigensolverCompactedSolveGraph& operator=(Gfn2EigensolverCompactedSolveGraph&&) noexcept;
  Gfn2EigensolverCompactedSolveGraph(const Gfn2EigensolverCompactedSolveGraph&) = delete;
  Gfn2EigensolverCompactedSolveGraph& operator=(const Gfn2EigensolverCompactedSolveGraph&) = delete;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] Gfn2EigensolverLaunchResult launch(cudaStream_t stream = nullptr) const noexcept;

 private:
  std::unique_ptr<Impl> impl_;

  friend Gfn2EigensolverLaunchResult build_gfn2_compacted_eigensolver_graph_cuda(
      const Gfn2EigensolverDeviceBatch&, const Gfn2EigensolverBucket*, std::int64_t,
      const Gfn2EigensolverOverlapCache&, std::uint64_t, const double*,
      const Gfn2EigensolverOptions&, cusolverDnHandle_t, cusolverDnParams_t, cublasHandle_t,
      const Gfn2EigensolverDeviceWorkspace&, const Gfn2EigensolverDeviceResults&, std::uint32_t*,
      std::uint32_t*, Gfn2EigensolverCompactedSolveGraph&) noexcept;
  friend Gfn2EigensolverLaunchResult build_gfn2_compacted_eigensolver_graph_cuda(
      const Gfn2EigensolverDeviceBatch&, const Gfn2EigensolverBucket*, std::int64_t,
      const Gfn2EigensolverOverlapCache&, const Gfn2GeometryEpochDevice&, const double*,
      const Gfn2EigensolverOptions&, cusolverDnHandle_t, cusolverDnParams_t, cublasHandle_t,
      const Gfn2EigensolverDeviceWorkspace&, const Gfn2EigensolverDeviceResults&, std::uint32_t*,
      std::uint32_t*, Gfn2EigensolverCompactedSolveGraph&) noexcept;
};

/* Build against a fixed host generation or a replay-advanced device epoch. */
Gfn2EigensolverLaunchResult build_gfn2_compacted_eigensolver_graph_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const Gfn2EigensolverOverlapCache& cache,
    std::uint64_t geometry_generation, const double* hamiltonians,
    const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver, cusolverDnParams_t parameters,
    cublasHandle_t blas, const Gfn2EigensolverDeviceWorkspace& workspace,
    const Gfn2EigensolverDeviceResults& results, std::uint32_t* system_errors,
    std::uint32_t* device_error, Gfn2EigensolverCompactedSolveGraph& output) noexcept;

Gfn2EigensolverLaunchResult build_gfn2_compacted_eigensolver_graph_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const Gfn2EigensolverOverlapCache& cache,
    const Gfn2GeometryEpochDevice& geometry_epoch, const double* hamiltonians,
    const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver, cusolverDnParams_t parameters,
    cublasHandle_t blas, const Gfn2EigensolverDeviceWorkspace& workspace,
    const Gfn2EigensolverDeviceResults& results, std::uint32_t* system_errors,
    std::uint32_t* device_error, Gfn2EigensolverCompactedSolveGraph& output) noexcept;

/*
 * Setup-time workspace query for the largest bucket requirement. Query every
 * bucket and retain the componentwise maximum. No numerical work is enqueued.
 */
Gfn2EigensolverLaunchResult query_gfn2_eigensolver_bucket_workspace_cuda(
    cusolverDnHandle_t solver, cusolverDnParams_t parameters, const Gfn2EigensolverBucket& bucket,
    const double* device_matrix, const double* device_eigenvalues,
    Gfn2EigensolverWorkspaceRequirements& requirements) noexcept;

/* Query the provider workspace for every spin-expanded batch count. */
Gfn2EigensolverLaunchResult query_gfn2_spin_eigensolver_bucket_workspace_cuda(
    cusolverDnHandle_t solver, cusolverDnParams_t parameters, const Gfn2EigensolverBucket& bucket,
    const double* device_matrix, const double* device_eigenvalues,
    Gfn2EigensolverWorkspaceRequirements& requirements) noexcept;

/* Add the Jacobi workspace for every reachable exact-capacity body through
 * the provider limit. Production calls this only for auto-selected buckets;
 * focused tests may query the wider forced-provider range. */
Gfn2EigensolverLaunchResult query_gfn2_jacobi_bucket_workspace_cuda(
    cusolverDnHandle_t solver, syevjInfo_t jacobi, const Gfn2EigensolverBucket& bucket,
    const double* device_matrix, const double* device_eigenvalues,
    Gfn2EigensolverWorkspaceRequirements& requirements) noexcept;

/* Add setup-owned workspace for the Graph-capturable large-singleton provider.
 * The same arena is reused sequentially for restricted and spin-expanded
 * solves, so the requirement depends on AO size rather than solve count. */
Gfn2EigensolverLaunchResult query_gfn2_tridiagonal_bucket_workspace_cuda(
    cusolverDnHandle_t solver, const Gfn2EigensolverBucket& bucket, const double* device_matrix,
    const double* device_eigenvalues, Gfn2EigensolverWorkspaceRequirements& requirements) noexcept;

/*
 * Production exact-capacity dispatch primitives. The production device-tail
 * loop cannot nest the #131 conditional (SWITCH) compaction graph, because a
 * graph containing conditional nodes cannot be launched from device code. It
 * instead builds one device-launchable executable per (bucket, capacity) and a
 * device-resident table of executable handles that a device dispatcher selects
 * from without any host decision. These entry points expose the per-capacity
 * body captures and the no-conditional compaction kernels the loop needs.
 *
 * All helpers are capture-safe: they launch ordinary kernels (and, for the
 * eigensolver/backtransform bodies, cuBLAS/cuSOLVER calls) that record into an
 * active stream capture. No allocation, transfer, polling, or synchronization
 * is performed.
 */

/* Compact the eligible systems of one bucket into compact_systems and publish
 * bucket_activity.active_count, with no conditional handle. */
Gfn2EigensolverLaunchResult compact_gfn2_solve_bucket_counts_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket& bucket,
    std::int64_t bucket_index, const Gfn2EigensolverDeviceWorkspace& workspace,
    const std::uint32_t* system_errors, cudaStream_t stream = nullptr) noexcept;

/* Run prepare_solve_bucket + compact_solve_bucket_counts for every bucket. */
Gfn2EigensolverLaunchResult prepare_and_compact_gfn2_solve_buckets_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const Gfn2EigensolverOverlapCache& cache,
    std::uint64_t geometry_generation, const double* hamiltonians,
    const Gfn2EigensolverOptions& options, const Gfn2EigensolverDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

/* Replay-safe epoch variant consumed by the production dispatch chain. */
Gfn2EigensolverLaunchResult prepare_and_compact_gfn2_solve_buckets_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const Gfn2EigensolverOverlapCache& cache,
    const Gfn2GeometryEpochDevice& geometry_epoch, const double* hamiltonians,
    const Gfn2EigensolverOptions& options, const Gfn2EigensolverDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

/* Compact successful eigenpairs after the eigensolver body; no conditional. */
Gfn2EigensolverLaunchResult compact_gfn2_successful_eigenpair_counts_cuda(
    const Gfn2EigensolverBucket& bucket, std::int64_t bucket_index,
    const Gfn2EigensolverDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

/*
 * Capture the exact-capacity eigensolver body (gather, TRSM x2, sym, provider)
 * into `body` for exactly `capacity` compacted systems of one bucket. The
 * caller owns `body` and must instantiate it. Provider handle streams are set
 * to capture_stream for the capture and left to the caller afterward.
 */
Gfn2EigensolverLaunchResult capture_gfn2_eigensolver_capacity_cuda(
    cudaGraph_t body, cudaStream_t capture_stream, std::uint32_t capacity,
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket& bucket,
    std::int64_t bucket_index, const Gfn2EigensolverOverlapCache& cache, const double* hamiltonians,
    cusolverDnHandle_t solver, cusolverDnParams_t parameters, cublasHandle_t blas,
    const Gfn2EigensolverDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, const Gfn2EigensolverOptions& options) noexcept;

/*
 * Enqueue the exact-capacity eigensolver body kernels into an active capture
 * on `stream`. The caller owns the surrounding capture so it can append the
 * chain dispatch kernel after the provider body in one graph. capacity==0
 * enqueues no provider work and reports success.
 */
Gfn2EigensolverLaunchResult enqueue_gfn2_eigensolver_capacity_cuda(
    cudaStream_t capture_stream, std::uint32_t capacity, const Gfn2EigensolverDeviceBatch& batch,
    const Gfn2EigensolverBucket& bucket, std::int64_t bucket_index,
    const Gfn2EigensolverOverlapCache& cache, const double* hamiltonians, cusolverDnHandle_t solver,
    cusolverDnParams_t parameters, cublasHandle_t blas,
    const Gfn2EigensolverDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, const Gfn2EigensolverOptions& options) noexcept;

/*
 * Capture the exact-capacity backtransform body (mark, TRSM-T, scatter) into
 * `body` for exactly `capacity` compacted systems of one bucket.
 */
Gfn2EigensolverLaunchResult capture_gfn2_backtransform_capacity_cuda(
    cudaGraph_t body, cudaStream_t capture_stream, std::uint32_t capacity,
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket& bucket,
    std::int64_t bucket_index, cublasHandle_t blas, const Gfn2EigensolverDeviceWorkspace& workspace,
    const Gfn2EigensolverDeviceResults& results, std::uint32_t* system_errors,
    std::uint32_t* device_error, bool deterministic_debug) noexcept;

/* Enqueue the backtransform body kernels into an active capture on `stream`. */
Gfn2EigensolverLaunchResult enqueue_gfn2_backtransform_capacity_cuda(
    cudaStream_t capture_stream, std::uint32_t capacity, const Gfn2EigensolverDeviceBatch& batch,
    const Gfn2EigensolverBucket& bucket, std::int64_t bucket_index, cublasHandle_t blas,
    const Gfn2EigensolverDeviceWorkspace& workspace, const Gfn2EigensolverDeviceResults& results,
    std::uint32_t* system_errors, std::uint32_t* device_error, bool deterministic_debug) noexcept;

/* Clear per-system and sticky diagnostics asynchronously. */
cudaError_t reset_gfn2_eigensolver_device_errors_cuda(std::int64_t batch_size,
                                                      std::uint32_t* system_errors,
                                                      std::uint32_t* device_error,
                                                      cudaStream_t stream = nullptr) noexcept;

/*
 * Public open of one eigensolver launch sequence: capture the sticky device
 * state into workspace.sequence_active, clear info_a, and validate the device
 * bucket permutation. The monolithic solver calls this at the head of every
 * solve; the production dispatch chain must reproduce it before compaction so
 * prepare_solve_bucket sees an open sequence.
 */
Gfn2EigensolverLaunchResult prepare_gfn2_eigensolver_launch_sequence_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverDeviceWorkspace& workspace,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

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
 * Device-epoch overload for reusable CUDA Graph execution. The epoch pointer
 * is captured by address and read only when each healthy peer publishes its
 * factor, so replay observes the value advanced by the preprocessing head.
 * The descriptor must share batch.plan_token and remain stable for all queued
 * work. A zero device value fails peers closed instead of publishing a cache.
 */
Gfn2EigensolverLaunchResult factor_gfn2_overlap_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const double* overlap, const Gfn2GeometryEpochDevice& geometry_epoch,
    const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver, cusolverDnParams_t parameters,
    const Gfn2EigensolverDeviceWorkspace& workspace, const Gfn2EigensolverOverlapCache& cache,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream = nullptr,
    Gfn2EigensolverFactorCachePolicy cache_policy =
        Gfn2EigensolverFactorCachePolicy::kPreservePriorOnFailure) noexcept;

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

/*
 * Solve the spin-expanded Hamiltonian projection while reusing one physical
 * overlap factor per system. Restricted systems submit exactly one provider
 * solve; unrestricted systems submit alpha then beta. Both spin outputs are
 * validated before either is published, making the physical system the
 * transaction boundary. This leaf entry point is intentionally independent
 * of SCC setup/arena ownership so those layers can adopt it in a later issue.
 */
Gfn2EigensolverLaunchResult solve_gfn2_spin_eigensystems_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2EigensolverBucket* buckets, std::int64_t bucket_count,
    const Gfn2EigensolverOverlapCache& cache, std::uint64_t geometry_generation,
    const double* hamiltonians, const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver,
    cusolverDnParams_t parameters, cublasHandle_t blas,
    const Gfn2EigensolverDeviceWorkspace& workspace, const Gfn2EigensolverDeviceResults& results,
    std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

Gfn2EigensolverLaunchResult solve_gfn2_spin_eigensystems_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2EigensolverBucket* buckets, std::int64_t bucket_count,
    const Gfn2EigensolverOverlapCache& cache, const Gfn2GeometryEpochDevice& geometry_epoch,
    const double* hamiltonians, const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver,
    cusolverDnParams_t parameters, cublasHandle_t blas,
    const Gfn2EigensolverDeviceWorkspace& workspace, const Gfn2EigensolverDeviceResults& results,
    std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

/* Replay-safe solve against overlap factors published for the current epoch. */
Gfn2EigensolverLaunchResult solve_gfn2_eigensystems_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const Gfn2EigensolverOverlapCache& cache,
    const Gfn2GeometryEpochDevice& geometry_epoch, const double* hamiltonians,
    const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver, cusolverDnParams_t parameters,
    cublasHandle_t blas, const Gfn2EigensolverDeviceWorkspace& workspace,
    const Gfn2EigensolverDeviceResults& results, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_EIGENSOLVER_CUH
