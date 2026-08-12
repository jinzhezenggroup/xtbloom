#ifndef XTBLOOM_RUNTIME_MODEL_PLAN_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_RUNTIME_MODEL_PLAN_HPP

#include <cstdint>
#include <memory>
#include <string>

#include "runtime/model_registry.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail {

class Gfn2Plan;
struct Context;
struct RequestSubmission;

/*
 * Model-dispatched public fixed-topology plan.
 *
 * The wrapper resolves the model route before constructing any model-specific
 * cache. Every later operation switches on that retained route, so enabling a
 * registry entry cannot silently reinterpret GFN1 topology as a GFN2 plan.
 */
class ModelPlan {
 public:
  ModelPlan();
  ~ModelPlan();

  ModelPlan(const ModelPlan&) = delete;
  ModelPlan& operator=(const ModelPlan&) = delete;

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
  ModelBackendRoute route_ = ModelBackendRoute::kUnavailable;
  Context* context_ = nullptr;
  std::unique_ptr<Gfn2Plan> gfn2_;
};

}  // namespace xtbloom::detail

#endif  // XTBLOOM_RUNTIME_MODEL_PLAN_HPP
