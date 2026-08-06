#ifndef GPUXTB_RUNTIME_GFN2_CPU_EXECUTION_HPP
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_RUNTIME_GFN2_CPU_EXECUTION_HPP

#include <cstdint>
#include <memory>
#include <string>

#include "gpuxtb/gpuxtb.h"

namespace gpuxtb::detail {

/*
 * Context-owned cache for restricted host GFN2 execution.
 *
 * The implementation keeps immutable per-system topology plans and their
 * caller-owned numerical workspaces alive across repeated public C API calls.
 * Geometry-dependent caches and SCC state are refreshed for every inference.
 * A fully converged inference also retains a per-system electronic checkpoint
 * so a strict WARM start can seed the next SCC run from the converged state
 * instead of resuming from the SAD guess.
 */
class Gfn2CpuExecutionCache {
 public:
  /*
   * cpu_threads is the context-wide outer batch parallelism requested by the
   * public API. Zero selects an affinity-aware automatic value; one keeps the
   * execution path serial. The implementation owns persistent workers so a
   * steady-state compute call never creates or destroys threads.
   */
  explicit Gfn2CpuExecutionCache(std::int32_t cpu_threads);
  ~Gfn2CpuExecutionCache();

  Gfn2CpuExecutionCache(const Gfn2CpuExecutionCache&) = delete;
  Gfn2CpuExecutionCache& operator=(const Gfn2CpuExecutionCache&) = delete;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend gpuxtb_status_t execute_restricted_gfn2_cpu(Gfn2CpuExecutionCache& cache,
                                                     const gpuxtb_batch_t& batch,
                                                     const gpuxtb_compute_options_t& options,
                                                     gpuxtb_batch_result_t& result,
                                                     std::string& error);
};

/*
 * Execute one already descriptor-validated host request.
 *
 * Inputs are copied before numerical work so under-aligned C buffers remain
 * well-defined and caller mutations cannot race an in-flight synchronous
 * call. Requested outputs and result flags are committed only after every
 * batch member reaches either a successful or documented terminal state.
 */
gpuxtb_status_t execute_restricted_gfn2_cpu(Gfn2CpuExecutionCache& cache,
                                            const gpuxtb_batch_t& batch,
                                            const gpuxtb_compute_options_t& options,
                                            gpuxtb_batch_result_t& result, std::string& error);

}  // namespace gpuxtb::detail

#endif  // GPUXTB_RUNTIME_GFN2_CPU_EXECUTION_HPP
