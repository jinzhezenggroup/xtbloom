#ifndef GPUXTB_BACKENDS_CUDA_GFN2_SCC_LOOP_CUH
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_BACKENDS_CUDA_GFN2_SCC_LOOP_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_scc_iteration.cuh"

namespace gpuxtb::detail::cuda {

inline constexpr std::uint32_t kGfn2SccLoopAbiVersion = 3u;

enum class Gfn2SccLoopExecutionMode : std::uint32_t {
  kBoundedFallback = 0u,
  kDeviceTailGraph = 1u,
  /* Device-dispatched exact-capacity chain. Each bucket's eigensolver and
   * backtransform runs at the exact compacted active count selected by a
   * device-resident executable table, with no host decision per iteration. */
  kDeviceDispatchChain = 2u,
  /* Compatibility name retained for runtime identity consumers. */
  kConditionalGraph = kDeviceTailGraph,
};

/*
 * Device-resident dispatch state shared by every exact-capacity chain kernel.
 * The executable table is filled once after all chain executables exist and is
 * read only at replay, so no per-iteration host transfer, allocation, polling,
 * or synchronization is introduced.
 */
struct Gfn2SccDispatchChainDevice {
  const cudaGraphExec_t* table = nullptr;
  std::int64_t table_slots = 0;
  std::int64_t pre_slot = -1;
  std::int64_t post_slot = -1;
  std::int32_t bucket_count = 0;
};

static_assert(std::is_trivially_copyable_v<Gfn2SccDispatchChainDevice>);
static_assert(std::is_standard_layout_v<Gfn2SccDispatchChainDevice>);

/*
 * Synchronous submission result for a bounded SCC loop. submitted_iterations
 * counts complete one-iteration DAGs enqueued before the first launch failure.
 * Numerical convergence and peer failures remain device-resident in the
 * binding state and never require host polling by this launcher.
 */
struct Gfn2SccLoopLaunchResult {
  std::uint32_t abi_version = kGfn2SccLoopAbiVersion;
  Gfn2SccLoopExecutionMode execution_mode = Gfn2SccLoopExecutionMode::kBoundedFallback;
  std::uint64_t submitted_iterations = 0u;
  std::uint64_t submitted_graphs = 0u;
  Gfn2SccIterationLaunchResult iteration{};

  [[nodiscard]] bool success() const noexcept {
    return abi_version == kGfn2SccLoopAbiVersion &&
           iteration.status == Gfn2SccIterationLaunchStatus::kSuccess;
  }
};

static_assert(std::is_trivially_copyable_v<Gfn2SccLoopLaunchResult>);
static_assert(std::is_standard_layout_v<Gfn2SccLoopLaunchResult>);

/* Setup-time reason why the bounded production path was retained. */
enum class Gfn2SccLoopGraphFallbackReason : std::uint32_t {
  kNone = 0u,
  kConditionalNodesUnavailable = 1u,
  kProviderCaptureUnsupported = 2u,
  kControlAllocationFailed = 3u,
  kRootCaptureFailed = 4u,
  kNumericalBodyCaptureFailed = 5u,
  kInstantiationFailed = 6u,
  kDeviceGraphLaunchUnavailable = 7u,
  kDeviceGraphNodeUnsupported = 8u,
  kDeviceGraphInstantiationFailed = 9u,
  kDeviceGraphUploadFailed = 10u,
  /* Exact-capacity dispatch chain rejected this binding. */
  kDispatchUnsupportedLayout = 11u,
  kDispatchCapacityOverflow = 12u,
  kDispatchBuildFailed = 13u,
  kDispatchTableAllocationFailed = 14u,
};

enum class Gfn2SccLoopGraphBuildStatus : std::uint32_t {
  kDeviceTailGraphReady = 0u,
  /* Compatibility name retained for existing callers and tests. */
  kConditionalGraphReady = kDeviceTailGraphReady,
  kBoundedFallbackReady = 1u,
  kInvalidBinding = 2u,
  /* Device-dispatched exact-capacity chain is ready and used by launch(). */
  kDeviceDispatchChainReady = 3u,
};

struct Gfn2SccLoopGraphBuildResult {
  Gfn2SccLoopGraphBuildStatus status = Gfn2SccLoopGraphBuildStatus::kInvalidBinding;
  Gfn2SccLoopGraphFallbackReason fallback_reason = Gfn2SccLoopGraphFallbackReason::kNone;
  cudaError_t cuda_status = cudaSuccess;
  Gfn2SccIterationLaunchResult iteration{};

  [[nodiscard]] bool success() const noexcept {
    return status == Gfn2SccLoopGraphBuildStatus::kConditionalGraphReady ||
           status == Gfn2SccLoopGraphBuildStatus::kBoundedFallbackReady ||
           status == Gfn2SccLoopGraphBuildStatus::kDeviceDispatchChainReady;
  }

  [[nodiscard]] bool device_tail_graph_ready() const noexcept {
    return status == Gfn2SccLoopGraphBuildStatus::kDeviceTailGraphReady;
  }

  [[nodiscard]] bool device_dispatch_chain_ready() const noexcept {
    return status == Gfn2SccLoopGraphBuildStatus::kDeviceDispatchChainReady;
  }

  [[nodiscard]] bool conditional_graph_ready() const noexcept {
    return device_tail_graph_ready() || device_dispatch_chain_ready();
  }
};

static_assert(std::is_trivially_copyable_v<Gfn2SccLoopGraphBuildResult>);
static_assert(std::is_standard_layout_v<Gfn2SccLoopGraphBuildResult>);

/*
 * Which Graph family an owner should build. kAuto is the production default:
 * the exact-capacity device-dispatch chain is preferred and the monolithic
 * full-capacity device-tail graph remains the restricted fallback. The
 * explicit choices exist so benchmarks and tests can measure or force one
 * family deterministically from the same binary.
 */
enum class Gfn2SccLoopGraphPreference : std::uint32_t {
  kAuto = 0u,
  kDeviceTailGraph = 1u,
  kDeviceDispatchChain = 2u,
};

/*
 * Fixed-context owner for one reusable device-resident SCC loop. build() is a
 * setup-only operation and may allocate Graph/control resources. The ordinary
 * root Graph derives activity and launches a pre-uploaded device Graph exactly
 * once when work exists. The device Graph tail-relaunches itself until the
 * canonical active count reaches zero. launch() therefore performs one root
 * submission, or uses the sealed bounded fallback when the provider/runtime
 * cannot build the device-launchable numerical body. Neither hot path polls
 * device state, transfers per-iteration host data, allocates, or synchronizes.
 *
 * The owner is single-flight with its binding and provider handles. Callers
 * must order every earlier launch before reset/destruction or rebuilding.
 */
class Gfn2SccLoopCudaGraphOwner {
 public:
  /* Opaque implementation record; exposed only so the CUDA translation unit
   * can build setup helpers without placing device-Graph handles here. */
  struct State;

  Gfn2SccLoopCudaGraphOwner() noexcept = default;
  ~Gfn2SccLoopCudaGraphOwner();
  Gfn2SccLoopCudaGraphOwner(const Gfn2SccLoopCudaGraphOwner&) = delete;
  Gfn2SccLoopCudaGraphOwner& operator=(const Gfn2SccLoopCudaGraphOwner&) = delete;
  Gfn2SccLoopCudaGraphOwner(Gfn2SccLoopCudaGraphOwner&&) = delete;
  Gfn2SccLoopCudaGraphOwner& operator=(Gfn2SccLoopCudaGraphOwner&&) = delete;

  [[nodiscard]] Gfn2SccLoopGraphBuildResult build(const Gfn2SccIterationBinding& binding) noexcept;

  [[nodiscard]] Gfn2SccLoopGraphBuildResult build(
      const Gfn2SccIterationBinding& binding,
      const Gfn2GeometryEpochConsumerDevice& geometry) noexcept;

  [[nodiscard]] Gfn2SccLoopGraphBuildResult build(const Gfn2SccIterationBinding& binding,
                                                  Gfn2SccLoopGraphPreference preference) noexcept;

  [[nodiscard]] Gfn2SccLoopGraphBuildResult build(const Gfn2SccIterationBinding& binding,
                                                  const Gfn2GeometryEpochConsumerDevice& geometry,
                                                  Gfn2SccLoopGraphPreference preference) noexcept;

  [[nodiscard]] Gfn2SccLoopLaunchResult launch(cudaStream_t stream = nullptr) const noexcept;

  void reset() noexcept;
  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] bool device_tail_graph_ready() const noexcept;
  [[nodiscard]] bool device_dispatch_chain_ready() const noexcept;
  [[nodiscard]] bool conditional_graph_ready() const noexcept;
  [[nodiscard]] Gfn2SccLoopGraphFallbackReason fallback_reason() const noexcept;

  /* Setup-owned device counters for testing/profiling after caller ordering. */
  [[nodiscard]] const std::uint32_t* canonical_active_count_device() const noexcept;
  [[nodiscard]] const std::uint64_t* numerical_body_count_device() const noexcept;
  [[nodiscard]] const std::uint32_t* device_launch_error_device() const noexcept;
  /* Explicit device control storage retained by the graph owner. */
  [[nodiscard]] std::size_t retained_device_bytes() const noexcept;

  /* Number of separate executable graphs in the device dispatch chain family:
   * exact-capacity eigensolver/backtransform bodies plus the pre/post chain
   * heads. Returns zero when the owner did not build a dispatch chain. */
  [[nodiscard]] std::size_t dispatch_chain_executable_count() const noexcept;

 private:
  [[nodiscard]] Gfn2SccLoopGraphBuildResult build_impl(
      const Gfn2SccIterationBinding& binding, const Gfn2GeometryEpochConsumerDevice* geometry,
      Gfn2SccLoopGraphPreference preference) noexcept;

  State* state_ = nullptr;
};

/*
 * Enqueue the complete configured restricted SCC iteration bound on the caller
 * stream. Each iteration derives activity from the state published by its
 * predecessor, so converged, failed, and max-iteration peers cannot modify
 * public numerical state while healthy peers continue. Most numerical kernels
 * skip inactive peers. The current dense eigensolver provider still executes
 * its fixed bucket calls with identity placeholders for inactive members;
 * #80 active-set compaction is responsible for removing that residual work.
 * The fixed bound removes host status polling and performs no allocation,
 * transfer, or synchronization.
 *
 * Extra submissions after every peer becomes terminal intentionally remain in
 * the stream as publication-gated iterations. Active-set compaction and Graph
 * segmentation may later reduce that overhead without changing this
 * correctness contract.
 *
 * The binding, its provider handles, and its mutable arena are single-flight:
 * callers must not overlap launches for one binding on different streams.
 * Reusing a binding on another stream is valid only after event or stream
 * ordering proves all earlier setup and launches complete. Every setup owner,
 * provider handle, host-provider workspace, and device arena referenced by the
 * binding must remain alive until the final queued stream operation completes.
 */
[[nodiscard]] Gfn2SccLoopLaunchResult launch_gfn2_restricted_scc_loop_cuda(
    const Gfn2SccIterationBinding& binding, cudaStream_t stream = nullptr) noexcept;

/* Replay-safe bounded loop using one immutable device-epoch consumer view. */
[[nodiscard]] Gfn2SccLoopLaunchResult launch_gfn2_restricted_scc_loop_cuda(
    const Gfn2SccIterationBinding& binding, const Gfn2GeometryEpochConsumerDevice& geometry,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_SCC_LOOP_CUH
