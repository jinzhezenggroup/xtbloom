#include "backends/cuda/gfn2_scc_loop.cuh"

namespace gpuxtb::detail::cuda {

Gfn2SccLoopLaunchResult launch_gfn2_restricted_scc_loop_cuda(const Gfn2SccIterationBinding& binding,
                                                             cudaStream_t stream) noexcept {
  Gfn2SccLoopLaunchResult result{};
  const std::uint64_t submission_bound = binding.plan.activity_policy.maximum_iterations;
  const auto reject = [&](Gfn2SccIterationBindingError error) {
    result.iteration.status = Gfn2SccIterationLaunchStatus::kInvalidBinding;
    result.iteration.binding.error = error;
    result.iteration.binding.field = Gfn2SccIterationBindingField::kPlan;
    return result;
  };
  if (binding.plan.abi_version != kGfn2SccIterationAbiVersion) {
    return reject(Gfn2SccIterationBindingError::kInvalidAbiVersion);
  }
  if (binding.plan.plan_token == 0u) {
    return reject(Gfn2SccIterationBindingError::kInvalidPlanToken);
  }
  if (submission_bound == 0u || binding.plan.state_policy.maximum_iterations != submission_bound ||
      binding.plan.publication_plan.maximum_iterations != submission_bound) {
    return reject(Gfn2SccIterationBindingError::kInvalidCount);
  }

  for (std::uint64_t iteration = 0u; iteration < submission_bound; ++iteration) {
    result.iteration = launch_gfn2_restricted_scc_iteration_cuda(binding, stream);
    if (!result.iteration.success()) {
      return result;
    }
    ++result.submitted_iterations;
  }
  return result;
}

}  // namespace gpuxtb::detail::cuda
