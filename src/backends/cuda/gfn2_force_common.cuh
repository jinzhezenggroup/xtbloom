#ifndef GPUXTB_BACKENDS_CUDA_GFN2_FORCE_COMMON_CUH
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_BACKENDS_CUDA_GFN2_FORCE_COMMON_CUH

#include <cstdint>
#include <type_traits>

#include "gpuxtb/gpuxtb.h"

namespace gpuxtb::detail::cuda {

/*
 * Common post-SCC force gate. A member is evaluated only when it is requested
 * and its terminal SCC status is SUCCESS. Unrequested and failed members must
 * be rejected before any member-local numerical input is read, which lets one
 * ragged force launch safely follow a partially failed SCC batch.
 *
 * This is deliberately distinct from Gfn2SccIterationDeviceActivity:
 * converged members are inactive for another SCC iteration but are exactly the
 * members for which stationary analytic forces may be evaluated.
 */
struct Gfn2ForceDeviceActivity {
  const std::uint8_t* requested_mask = nullptr;
  const gpuxtb_status_t* system_statuses = nullptr;
  std::int64_t batch_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2ForceDeviceActivity>);
static_assert(std::is_standard_layout_v<Gfn2ForceDeviceActivity>);

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_FORCE_COMMON_CUH
