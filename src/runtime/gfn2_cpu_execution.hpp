#ifndef XTBLOOM_RUNTIME_GFN2_CPU_EXECUTION_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_RUNTIME_GFN2_CPU_EXECUTION_HPP

#include <cstdint>
#include <memory>
#include <string>

#include "xtbloom/xtbloom.h"

namespace xtbloom::detail {

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

  friend xtbloom_status_t execute_restricted_gfn2_cpu(Gfn2CpuExecutionCache& cache,
                                                      const xtbloom_batch_t& batch,
                                                      const xtbloom_compute_options_t& options,
                                                      xtbloom_batch_result_t& result,
                                                      std::string& error);
  friend xtbloom_status_t prepare_restricted_gfn2_cpu(Gfn2CpuExecutionCache& cache,
                                                      const xtbloom_batch_t& batch,
                                                      const xtbloom_compute_options_t& options,
                                                      bool& reused, std::string& error);
  friend std::size_t persistent_workspace_bytes_restricted_gfn2_cpu(
      Gfn2CpuExecutionCache& cache) noexcept;
};

/*
 * Execute one already descriptor-validated host request.
 *
 * Inputs are copied before numerical work so under-aligned C buffers remain
 * well-defined and caller mutations cannot race an in-flight synchronous
 * call. Requested outputs and result flags are committed only after every
 * batch member reaches either a successful or documented terminal state.
 */
xtbloom_status_t execute_restricted_gfn2_cpu(Gfn2CpuExecutionCache& cache,
                                             const xtbloom_batch_t& batch,
                                             const xtbloom_compute_options_t& options,
                                             xtbloom_batch_result_t& result, std::string& error);

/*
 * Allocation-permitted fixed-topology setup for a public plan.
 *
 * Stages and validates the request and builds (or reuses) the per-system
 * SystemExecution objects for the requested identity, leaving the cache warm
 * so the following xtbloom_plan_compute runs allocation-free. `reused` is true
 * when the cache already held an identical identity and no system was rebuilt.
 */
xtbloom_status_t prepare_restricted_gfn2_cpu(Gfn2CpuExecutionCache& cache,
                                             const xtbloom_batch_t& batch,
                                             const xtbloom_compute_options_t& options, bool& reused,
                                             std::string& error);

/*
 * Topology- and spin-dependent persistent host reservation (per-system
 * storage plus the copied input request), independent of the requested
 * property flags. Returned value is used by fixed-topology plan queries so
 * their result stays correct even after another topology replaced the shared
 * cache's prepared systems.
 */
std::size_t persistent_workspace_bytes_restricted_gfn2_cpu(Gfn2CpuExecutionCache& cache) noexcept;

}  // namespace xtbloom::detail

#endif  // XTBLOOM_RUNTIME_GFN2_CPU_EXECUTION_HPP
