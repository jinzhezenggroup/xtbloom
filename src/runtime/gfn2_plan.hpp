#ifndef XTBLOOM_RUNTIME_GFN2_PLAN_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_RUNTIME_GFN2_PLAN_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "xtbloom/xtbloom.h"

namespace xtbloom::detail {

class Gfn2CpuExecutionCache;
class Gfn2CudaExecutionCache;
struct RequestSubmission;
struct Context;

/*
 * Opaque fixed-topology GFN2 plan.
 *
 * A plan is created from one validated batch descriptor and compute policy. It
 * binds the immutable ragged topology (atom offsets, element numbers,
 * molecular charges, unpaired electrons, spin channels, point-charge and
 * response structure) and owns a prepared cache for the selected backend.
 * Geometry (positions and point-charge positions/values) is intentionally
 * excluded: it may change on every xtbloom_plan_compute call while the plan is
 * reused.
 *
 * Creating a plan validates the topology and pre-warms the backend runtime so
 * repeated xtbloom_plan_compute calls for the same fixed topology perform zero
 * steady-state allocations. The plan must be destroyed before its context.
 * A plan is bound to the context that created it; using a plan with a batch
 * whose immutable topology differs, or with a different context, fails with
 * XTBLOOM_STATUS_INVALID_ARGUMENT before any caller output is modified.
 */
class Gfn2Plan {
 public:
  Gfn2Plan();
  ~Gfn2Plan();

  Gfn2Plan(const Gfn2Plan&) = delete;
  Gfn2Plan& operator=(const Gfn2Plan&) = delete;

  xtbloom_status_t create(Context& context, const xtbloom_batch_t& batch,
                          const xtbloom_compute_options_t& options, std::string& error);
  xtbloom_status_t query_workspace(std::uint32_t compute_flags, xtbloom_workspace_query_t& query,
                                   std::string& error);
  xtbloom_status_t compute(const xtbloom_batch_t& batch, const xtbloom_compute_options_t& options,
                           xtbloom_batch_result_t& result, std::string& error);
  xtbloom_status_t enqueue(const xtbloom_batch_t& batch, const xtbloom_compute_options_t& options,
                           const xtbloom_batch_result_t& result, RequestSubmission& submission,
                           std::string& error);
  void destroy() noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] Context* context() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xtbloom::detail

#endif  // XTBLOOM_RUNTIME_GFN2_PLAN_HPP
