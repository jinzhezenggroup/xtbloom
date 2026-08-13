#ifndef XTBLOOM_RUNTIME_GFN1_CPU_EXECUTION_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_RUNTIME_GFN1_CPU_EXECUTION_HPP

#include <cstddef>
#include <memory>
#include <string>

#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {
class CpuLinearAlgebraBackend;
}

namespace xtbloom::detail {

/*
 * Context- or plan-owned CPU execution cache for the complete GFN1 model.
 *
 * The type stays private to libxtbloom, but the reserved public GFN1 tag now
 * selects it on CPU. It owns model-specific topology, SCC/WARM state, and
 * publication staging so no request can fall through to GFN2 equations.
 */
class Gfn1CpuExecutionCache {
 public:
  /* Direct internal tests default to the historical serial executor. Public
   * contexts and plans always pass their resolved cpu_threads request. */
  explicit Gfn1CpuExecutionCache(std::int32_t cpu_threads = 1);
  ~Gfn1CpuExecutionCache();

  Gfn1CpuExecutionCache(const Gfn1CpuExecutionCache&) = delete;
  Gfn1CpuExecutionCache& operator=(const Gfn1CpuExecutionCache&) = delete;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend xtbloom_status_t prepare_gfn1_cpu(Gfn1CpuExecutionCache&, const xtbloom_batch_t&,
                                           const xtbloom_compute_options_t&, bool&, std::string&);
  friend xtbloom_status_t execute_gfn1_cpu(Gfn1CpuExecutionCache&, const xtbloom_batch_t&,
                                           const xtbloom_compute_options_t&,
                                           xtbloom_batch_result_t&, std::string&);
  friend std::size_t persistent_workspace_bytes_gfn1_cpu(Gfn1CpuExecutionCache&) noexcept;
  friend xtbloom_status_t set_gfn1_cpu_linear_algebra_backend_for_testing(
      Gfn1CpuExecutionCache&, const gfn2::CpuLinearAlgebraBackend&, std::string&);
};

/*
 * The model executor consumes already structurally validated HOST
 * descriptors. It still copies all input bytes, validates numerical values,
 * rejects unreleased GFN1 interactions/outputs, and publishes caller outputs
 * only after the complete batch reaches documented terminal states.
 */
xtbloom_status_t prepare_gfn1_cpu(Gfn1CpuExecutionCache& cache, const xtbloom_batch_t& batch,
                                  const xtbloom_compute_options_t& options, bool& reused,
                                  std::string& error);
xtbloom_status_t execute_gfn1_cpu(Gfn1CpuExecutionCache& cache, const xtbloom_batch_t& batch,
                                  const xtbloom_compute_options_t& options,
                                  xtbloom_batch_result_t& result, std::string& error);
std::size_t persistent_workspace_bytes_gfn1_cpu(Gfn1CpuExecutionCache& cache) noexcept;

/*
 * Install one verified internal-test LP64 backend before preparing or
 * executing the cache. This hidden dependency-injection seam lets executor
 * tests force precise LAPACK failures and observe provider cleanup without
 * changing production backend discovery or the public C ABI.
 */
xtbloom_status_t set_gfn1_cpu_linear_algebra_backend_for_testing(
    Gfn1CpuExecutionCache& cache, const gfn2::CpuLinearAlgebraBackend& backend, std::string& error);

}  // namespace xtbloom::detail

#endif  // XTBLOOM_RUNTIME_GFN1_CPU_EXECUTION_HPP
