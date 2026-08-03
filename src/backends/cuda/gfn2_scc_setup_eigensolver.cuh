#ifndef GPUXTB_BACKENDS_CUDA_GFN2_SCC_SETUP_EIGENSOLVER_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_SCC_SETUP_EIGENSOLVER_CUH

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "backends/cuda/gfn2_scc_iteration_arena.cuh"
#include "backends/cuda/gfn2_scc_setup_topology.hpp"
#include "gpuxtb/gpuxtb.h"

namespace gpuxtb::detail::cuda {

inline constexpr std::size_t kGfn2SccSetupEigensolverArenaAlignment = 256u;

enum class Gfn2SccSetupEigensolverError : std::uint32_t {
  kSuccess = 0u,
  kInvalidTopology = 1u,
  kCrossPlan = 2u,
  kInvalidGeneration = 3u,
  kInvalidOverlap = 4u,
  kInvalidOptions = 5u,
  kInvalidProvider = 6u,
  kCountOverflow = 7u,
  kAllocationFailed = 8u,
  kWorkspaceQueryFailed = 9u,
  kNullArena = 10u,
  kMisalignedArena = 11u,
  kInsufficientArena = 12u,
  kInvalidArenaMemory = 13u,
  kInvalidIterationProvenance = 14u,
  kInvalidIterationWorkspace = 15u,
  kInvalidHostWorkspace = 16u,
  kCudaError = 17u,
  kProviderLaunchFailed = 18u,
};

enum class Gfn2SccSetupEigensolverField : std::uint32_t {
  kNone = 0u,
  kTopology = 1u,
  kPlanToken = 2u,
  kGeometryGeneration = 3u,
  kOverlap = 4u,
  kOptions = 5u,
  kHandles = 6u,
  kWorkspaceQuery = 7u,
  kSetupArena = 8u,
  kIterationArena = 9u,
  kIterationWorkspace = 10u,
  kProviderHostWorkspace = 11u,
  kOverlapFactorization = 12u,
};

struct Gfn2SccSetupEigensolverDiagnostic {
  gpuxtb_status_t status = GPUXTB_STATUS_SUCCESS;
  Gfn2SccSetupEigensolverError error = Gfn2SccSetupEigensolverError::kSuccess;
  Gfn2SccSetupEigensolverField field = Gfn2SccSetupEigensolverField::kNone;
  std::int64_t index = -1;
  std::size_t required_bytes = 0u;
  std::size_t provided_bytes = 0u;
  cudaError_t cuda_status = cudaSuccess;
  cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

  [[nodiscard]] bool success() const noexcept {
    return status == GPUXTB_STATUS_SUCCESS && error == Gfn2SccSetupEigensolverError::kSuccess;
  }
};

/*
 * Exact persistent setup arena. overlap_input remains available for an
 * explicitly requested refactor; active/setup diagnostics support asynchronous
 * completion inspection; cache is the only part consumed by hot SCC launches.
 * The #101 iteration arena remains the sole owner of all eigensolver scratch
 * and the generic cuSOLVER device workspace.
 */
struct Gfn2SccSetupEigensolverRequirements {
  std::size_t alignment = kGfn2SccSetupEigensolverArenaAlignment;
  std::size_t overlap_input_offset = 0u;
  std::size_t active_offset = 0u;
  std::size_t system_error_offset = 0u;
  std::size_t device_error_offset = 0u;
  std::size_t cache_factor_offset = 0u;
  std::size_t cache_generation_offset = 0u;
  std::size_t cache_status_offset = 0u;
  std::size_t setup_device_bytes = 0u;
  std::size_t provider_host_workspace_bytes = 0u;
  Gfn2EigensolverWorkspaceRequirements provider{};
  std::uint64_t plan_token = 0u;
  std::uint64_t geometry_generation = 0u;
  std::uint64_t layout_fingerprint = 0u;
};

/*
 * Stable leaves published after every setup operation has been accepted by
 * CUDA and the borrowed provider. setup_system_errors/device_error and cache
 * statuses become host-observable only after caller stream completion.
 */
struct Gfn2SccSetupEigensolverBinding {
  Gfn2EigensolverDeviceBatch batch{};
  Gfn2SccIterationCudaEigensolverProvider provider{};
  Gfn2EigensolverOverlapCache cache{};
  Gfn2EigensolverDeviceWorkspace workspace{};
  Gfn2EigensolverOptions options{};
  const double* overlap_input = nullptr;
  std::int64_t overlap_elements = 0;
  std::uint32_t* setup_system_errors = nullptr;
  std::int64_t setup_system_error_elements = 0;
  std::uint32_t* setup_device_error = nullptr;
  /* Host-only provenance. These values are never dereferenced by device code
   * and let owner methods reject a binding copied from another owner/arena. */
  const void* owner_identity = nullptr;
  void* setup_device_arena = nullptr;
  std::size_t setup_device_arena_bytes = 0u;
  /* Owner-keyed seal over every bound pointer/count. It detects a descriptor
   * that was copied and then edited before a refresh call. */
  std::uint64_t provenance_seal = 0u;
  std::uint64_t geometry_generation = 0u;
  std::uint64_t iteration_layout_fingerprint = 0u;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2SccSetupEigensolverDiagnostic>);
static_assert(std::is_standard_layout_v<Gfn2SccSetupEigensolverDiagnostic>);
static_assert(std::is_trivially_copyable_v<Gfn2SccSetupEigensolverRequirements>);
static_assert(std::is_standard_layout_v<Gfn2SccSetupEigensolverRequirements>);
static_assert(std::is_trivially_copyable_v<Gfn2SccSetupEigensolverBinding>);
static_assert(std::is_standard_layout_v<Gfn2SccSetupEigensolverBinding>);

/*
 * Move-only setup owner. cuBLAS/cuSOLVER handles are borrowed and must outlive
 * this owner plus all queued factor/solve work. The owner copies stable buckets
 * from topology and owns a pinned immutable overlap upload image. Construction
 * may perform temporary setup allocations solely to query every bucket's exact
 * cuSOLVER workspace; it enqueues no numerical work.
 */
class Gfn2SccSetupEigensolver {
 public:
  Gfn2SccSetupEigensolver() noexcept;
  ~Gfn2SccSetupEigensolver();
  Gfn2SccSetupEigensolver(Gfn2SccSetupEigensolver&&) noexcept;
  Gfn2SccSetupEigensolver& operator=(Gfn2SccSetupEigensolver&&) noexcept;
  Gfn2SccSetupEigensolver(const Gfn2SccSetupEigensolver&) = delete;
  Gfn2SccSetupEigensolver& operator=(const Gfn2SccSetupEigensolver&) = delete;

  [[nodiscard]] static Gfn2SccSetupEigensolverDiagnostic create(
      const Gfn2SccSetupTopology& topology, const double* host_overlap,
      std::int64_t host_overlap_elements, std::uint64_t geometry_generation,
      std::uint64_t plan_token, cusolverDnHandle_t solver, cusolverDnParams_t parameters,
      cublasHandle_t blas, const Gfn2EigensolverOptions& options,
      Gfn2SccSetupEigensolver& output) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] const Gfn2SccSetupEigensolverRequirements& requirements() const noexcept;

  /*
   * Bind this owner's persistent setup arena, validate that iteration_workspace
   * is a #101 projection from iteration_arena/iteration_requirements (including
   * its canonical activity ledger, plan token, layout fingerprint, segment
   * bounds, provider offset, alignment, and disjoint ranges), upload the overlap
   * on stream, and enqueue reusable Cholesky factor construction. The published
   * hot-path batch always aliases iteration_workspace.ledger.active_mask; a
   * private all-active setup batch is used only while factoring the overlap.
   *
   * provider_host_workspace is caller-owned pinned host memory and is the same
   * pointer published into workspace/provider. No allocation, descriptor
   * rebuilding, transfer outside stream, polling, or synchronization occurs.
   * This owner, both arenas, topology device storage, handles, and host workspace
   * must outlive queued work. Cross-stream consumers need explicit event order.
   * binding is unchanged on every synchronous setup/provider rejection.
   */
  [[nodiscard]] Gfn2SccSetupEigensolverDiagnostic bind_and_factor_overlap_async(
      const Gfn2RaggedTopologyView& device_topology,
      const Gfn2SccIterationDevicePlan& iteration_plan,
      const Gfn2SccIterationArenaRequirements& iteration_requirements, void* iteration_arena,
      std::size_t iteration_arena_bytes, const Gfn2SccIterationDeviceWorkspace& iteration_workspace,
      void* provider_host_workspace, std::size_t provider_host_workspace_bytes,
      void* setup_device_arena, std::size_t setup_device_arena_bytes,
      Gfn2SccSetupEigensolverBinding& binding, cudaStream_t stream = nullptr) const noexcept;

  /*
   * Re-factor the bound overlap cache from the matrix evaluated for the
   * current device geometry. The binding and setup arena must be the exact
   * objects previously published by this owner; validating that provenance
   * before enqueueing work prevents a forged descriptor from mutating an
   * unrelated arena.
   *
   * device_overlap may exact-alias the owner's setup-time overlap slot or may
   * reside in another CUDA/managed allocation on the selected device. It may
   * not alias cache, diagnostic, activity, topology, or eigensolver workspace
   * storage. Healthy systems publish their new factors and generation
   * independently, while a numerical failure retains that member's previous
   * factor bytes and publishes its failure metadata for geometry_generation.
   *
   * The call allocates, transfers, polls, and synchronizes nowhere. It is
   * suitable for CUDA Graph capture when the linked provider supports the
   * low-level factorization sequence.
   */
  [[nodiscard]] Gfn2SccSetupEigensolverDiagnostic refactor_overlap_from_device_async(
      void* setup_device_arena, std::size_t setup_device_arena_bytes,
      Gfn2SccSetupEigensolverBinding& binding, const double* device_overlap,
      std::int64_t device_overlap_elements, std::uint64_t geometry_generation,
      cudaStream_t stream = nullptr) const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_SCC_SETUP_EIGENSOLVER_CUH
