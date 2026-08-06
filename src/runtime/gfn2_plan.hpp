#ifndef GPUXTB_RUNTIME_GFN2_PLAN_HPP
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_RUNTIME_GFN2_PLAN_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gpuxtb/gpuxtb.h"

namespace gpuxtb::detail {

class Gfn2CpuExecutionCache;
class Gfn2CudaExecutionCache;
struct Context;

/*
 * Opaque fixed-topology GFN2 plan.
 *
 * A plan is created from one validated batch descriptor and binds the
 * immutable ragged topology (atom offsets, element numbers, molecular
 * charges, unpaired electrons, spin channels, point-charge and response
 * structure) to the context backend. Geometry (positions and point-charge
 * positions/values) is intentionally excluded: it may change on every
 * gpuxtb_plan_compute call while the plan is reused.
 *
 * Creating a plan validates the topology and pre-warms the backend runtime so
 * repeated gpuxtb_plan_compute calls for the same fixed topology perform zero
 * steady-state allocations. The plan must be destroyed before its context.
 * A plan is bound to the context that created it; using a plan with a batch
 * whose immutable topology differs, or with a different context, fails with
 * GPUXTB_STATUS_INVALID_ARGUMENT before any caller output is modified.
 */
class Gfn2Plan {
 public:
  Gfn2Plan();
  ~Gfn2Plan();

  Gfn2Plan(const Gfn2Plan&) = delete;
  Gfn2Plan& operator=(const Gfn2Plan&) = delete;

  gpuxtb_status_t create(Context& context, const gpuxtb_batch_t& batch, std::string& error);
  gpuxtb_status_t query_workspace(std::uint32_t compute_flags, gpuxtb_workspace_query_t& query,
                                  std::string& error);
  gpuxtb_status_t compute(const gpuxtb_batch_t& batch, const gpuxtb_compute_options_t& options,
                          gpuxtb_batch_result_t& result, std::string& error);
  void destroy() noexcept;

  [[nodiscard]] bool valid() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpuxtb::detail

#endif  // GPUXTB_RUNTIME_GFN2_PLAN_HPP
