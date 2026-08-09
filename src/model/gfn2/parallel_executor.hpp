#ifndef XTBLOOM_MODEL_GFN2_PARALLEL_EXECUTOR_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_PARALLEL_EXECUTOR_HPP

#include <cstddef>

namespace xtbloom::detail::gfn2 {

/*
 * Optional chunked executor used to parallelize intra-system phases when the
 * runtime has idle worker capacity (a single-system batch whose outer worker
 * loop is otherwise serial).
 *
 * A null `dispatch_chunks` means serial execution; callers must keep the
 * existing serial code path bit-identical when the executor is disabled so
 * cpu_threads=1, tests, and the SCC trace capture never observe a change.
 * When enabled, `dispatch_chunks` runs `body` exactly once per index in
 * [0, chunk_count), each on one worker or the calling thread, in arbitrary
 * order. `body` must therefore be safe for concurrent execution, must not
 * perform dynamic allocation, and must write only output elements uniquely
 * owned by its chunk so the computed values are identical to the serial path.
 *
 * The model layer never creates the executor; the CPU runtime supplies it by
 * wrapping its context-owned worker pool only for batch==1, when the pool is
 * otherwise idle and reentrant use is safe.
 */
struct SccParallelExecutor {
  void* pool_context = nullptr;
  std::size_t worker_count = 1;
  void (*dispatch_chunks)(void* pool_context, std::size_t chunk_count,
                          void (*body)(void* body_context, std::size_t chunk) noexcept,
                          void* body_context) = nullptr;
};

inline bool scc_parallel_enabled(const SccParallelExecutor& executor) noexcept {
  return executor.dispatch_chunks != nullptr;
}

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_PARALLEL_EXECUTOR_HPP
