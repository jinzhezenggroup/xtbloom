#ifndef GPUXTB_BACKENDS_CUDA_GFN2_SCC_LOOP_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_SCC_LOOP_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_scc_iteration.cuh"

namespace gpuxtb::detail::cuda {

inline constexpr std::uint32_t kGfn2SccLoopAbiVersion = 1u;

/*
 * Synchronous submission result for a bounded SCC loop. submitted_iterations
 * counts complete one-iteration DAGs enqueued before the first launch failure.
 * Numerical convergence and peer failures remain device-resident in the
 * binding state and never require host polling by this launcher.
 */
struct Gfn2SccLoopLaunchResult {
  std::uint32_t abi_version = kGfn2SccLoopAbiVersion;
  std::uint32_t reserved = 0u;
  std::uint64_t submitted_iterations = 0u;
  Gfn2SccIterationLaunchResult iteration{};

  [[nodiscard]] bool success() const noexcept {
    return abi_version == kGfn2SccLoopAbiVersion &&
           iteration.status == Gfn2SccIterationLaunchStatus::kSuccess;
  }
};

static_assert(std::is_trivially_copyable_v<Gfn2SccLoopLaunchResult>);
static_assert(std::is_standard_layout_v<Gfn2SccLoopLaunchResult>);

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

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_SCC_LOOP_CUH
