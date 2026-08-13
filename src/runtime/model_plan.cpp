// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "runtime/model_plan.hpp"

#include <new>

#include "runtime/backend.hpp"
#include "runtime/gfn2_plan.hpp"
#include "runtime/validation.hpp"

namespace xtbloom::detail {
namespace {

xtbloom_status_t missing_executor(ModelBackendRoute route, std::string& error) {
  error = route == ModelBackendRoute::kGfn1
              ? "the registered GFN1 plan route has no linked executor"
              : "the registered model plan route is unavailable";
  return XTBLOOM_STATUS_INTERNAL_ERROR;
}

}  // namespace

ModelPlan::ModelPlan() = default;
ModelPlan::~ModelPlan() = default;

xtbloom_status_t ModelPlan::create(Context& context, const xtbloom_batch_t& batch,
                                   const xtbloom_compute_options_t& options, std::string& error) {
  if (context_ != nullptr) {
    error = "plan has already been created";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  /* Prove the complete public ABI prefix before reading options.model.  This
   * wrapper is the first model-neutral plan boundary, so dispatch must retain
   * the historical short-structure ordering enforced by Gfn2Plan::create. */
  DescriptorValidationResult validation =
      validate_plan_descriptor_structure_for_dispatch(context.backend, &batch, &options);
  if (!validation.ok()) {
    error = std::move(validation.error);
    return validation.status;
  }
  ModelBackendRoute selected = ModelBackendRoute::kUnavailable;
  const xtbloom_status_t route_status =
      validate_model_dispatch(options.model, context.backend, error, &selected);
  if (route_status != XTBLOOM_STATUS_SUCCESS) {
    return route_status;
  }

  switch (selected) {
    case ModelBackendRoute::kGfn2: {
      /* Preserve GFN2's established NOT_IMPLEMENTED ordering before allocating
       * any model-specific plan state. Gfn2Plan::create validates again because
       * it is also an independently usable internal boundary. */
      DescriptorValidationResult availability =
          validate_compute_execution_availability(context.backend, batch, options);
      if (!availability.ok()) {
        error = std::move(availability.error);
        return availability.status;
      }
      auto plan = std::unique_ptr<Gfn2Plan>(new (std::nothrow) Gfn2Plan{});
      if (plan == nullptr) {
        error = "failed to allocate a GFN2 plan implementation";
        return XTBLOOM_STATUS_ALLOCATION_FAILED;
      }
      const xtbloom_status_t status = plan->create(context, batch, options, error);
      if (status != XTBLOOM_STATUS_SUCCESS) {
        plan->destroy();
        return status;
      }
      route_ = selected;
      context_ = &context;
      gfn2_ = std::move(plan);
      return XTBLOOM_STATUS_SUCCESS;
    }
    case ModelBackendRoute::kGfn1:
    case ModelBackendRoute::kUnavailable:
      return missing_executor(selected, error);
  }
  return missing_executor(selected, error);
}

xtbloom_status_t ModelPlan::query_workspace(std::uint32_t compute_flags,
                                            xtbloom_workspace_query_t& query, std::string& error) {
  if (!valid()) {
    error = "plan is not created or has been destroyed";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  switch (route_) {
    case ModelBackendRoute::kGfn2:
      return gfn2_->query_workspace(compute_flags, query, error);
    case ModelBackendRoute::kGfn1:
    case ModelBackendRoute::kUnavailable:
      return missing_executor(route_, error);
  }
  return missing_executor(route_, error);
}

xtbloom_status_t ModelPlan::compute(const xtbloom_batch_t& batch,
                                    const xtbloom_compute_options_t& options,
                                    xtbloom_batch_result_t& result, std::string& error) {
  if (!valid()) {
    error = "plan is not created or has been destroyed";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  switch (route_) {
    case ModelBackendRoute::kGfn2:
      return gfn2_->compute(batch, options, result, error);
    case ModelBackendRoute::kGfn1:
    case ModelBackendRoute::kUnavailable:
      return missing_executor(route_, error);
  }
  return missing_executor(route_, error);
}

xtbloom_status_t ModelPlan::enqueue(const xtbloom_batch_t& batch,
                                    const xtbloom_compute_options_t& options,
                                    const xtbloom_batch_result_t& result,
                                    RequestSubmission& submission, std::string& error) {
  if (!valid()) {
    error = "plan is not created or has been destroyed";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  switch (route_) {
    case ModelBackendRoute::kGfn2:
      return gfn2_->enqueue(batch, options, result, submission, error);
    case ModelBackendRoute::kGfn1:
    case ModelBackendRoute::kUnavailable:
      return missing_executor(route_, error);
  }
  return missing_executor(route_, error);
}

void ModelPlan::destroy() noexcept {
  if (gfn2_ != nullptr) {
    gfn2_->destroy();
    gfn2_.reset();
  }
  route_ = ModelBackendRoute::kUnavailable;
  context_ = nullptr;
}

bool ModelPlan::valid() const noexcept {
  return context_ != nullptr && route_ != ModelBackendRoute::kUnavailable &&
         (route_ != ModelBackendRoute::kGfn2 || (gfn2_ != nullptr && gfn2_->valid()));
}

Context* ModelPlan::context() const noexcept { return context_; }

}  // namespace xtbloom::detail
