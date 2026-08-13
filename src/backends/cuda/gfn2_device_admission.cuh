#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_DEVICE_ADMISSION_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN2_DEVICE_ADMISSION_CUH

#include <cstdint>
#include <type_traits>

namespace xtbloom::detail::cuda {

/* Stream-ordered public request error codes. The public result bridge always
 * reads this scalar, so rejected requests need not execute any numerical or
 * publication mutation merely to report their precise status. */
inline constexpr std::uint32_t kGfn2RequestErrorNone = 0u;
inline constexpr std::uint32_t kGfn2RequestErrorInvalid = 1u;
inline constexpr std::uint32_t kGfn2RequestErrorNotImplemented = 2u;
inline constexpr std::uint32_t kGfn2RequestErrorWarmIncompatible = 3u;

/* Plan-owned stream-ordered request gate used by persistent mutation roots. */
struct Gfn2DeviceAdmission {
  const std::uint32_t* error = nullptr;
  std::int64_t error_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2DeviceAdmission>);
static_assert(std::is_standard_layout_v<Gfn2DeviceAdmission>);

[[nodiscard]] inline __host__ __device__ bool gfn2_request_admitted(
    const Gfn2DeviceAdmission& admission) noexcept {
  if (admission.error == nullptr) return true;
  return *admission.error == kGfn2RequestErrorNone;
}

[[nodiscard]] inline __host__ __device__ bool gfn2_request_mutation_allowed(
    const Gfn2DeviceAdmission& admission) noexcept {
  return admission.error == nullptr || *admission.error == kGfn2RequestErrorNone;
}

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_DEVICE_ADMISSION_CUH
