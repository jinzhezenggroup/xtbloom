#ifndef GPUXTB_RUNTIME_GFN2_CPU_EXECUTION_HPP
#define GPUXTB_RUNTIME_GFN2_CPU_EXECUTION_HPP

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
 */
class Gfn2CpuExecutionCache {
 public:
  Gfn2CpuExecutionCache();
  ~Gfn2CpuExecutionCache();

  Gfn2CpuExecutionCache(const Gfn2CpuExecutionCache&) = delete;
  Gfn2CpuExecutionCache& operator=(const Gfn2CpuExecutionCache&) = delete;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend gpuxtb_status_t execute_restricted_gfn2_cpu(
      Gfn2CpuExecutionCache& cache, const gpuxtb_batch_t& batch,
      const gpuxtb_compute_options_t& options, gpuxtb_batch_result_t& result,
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
                                             gpuxtb_batch_result_t& result,
                                             std::string& error);

}  // namespace gpuxtb::detail

#endif  // GPUXTB_RUNTIME_GFN2_CPU_EXECUTION_HPP
