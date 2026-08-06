#ifndef GPUXTB_BACKENDS_CUDA_GFN2_PARAMETERS_CUH
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_BACKENDS_CUDA_GFN2_PARAMETERS_CUH

#include <cstddef>

#include "data/parameters/gfn2.hpp"

namespace gpuxtb::detail::cuda {

inline constexpr std::size_t kPairOverrideStorage =
    parameters::gfn2::kPairScaleOverrides.size() == 0u
        ? 1u
        : parameters::gfn2::kPairScaleOverrides.size();

/* Device declarations shared by native GFN2 CUDA kernel translation units. */
extern __device__ __constant__ parameters::gfn2::GlobalParameters g_gfn2_global;
extern __device__ __constant__
    parameters::gfn2::ElementParameters g_gfn2_elements[parameters::gfn2::kElementCount];
extern __device__ __constant__
    parameters::gfn2::ShellParameters g_gfn2_shells[parameters::gfn2::kShellCount];
extern __device__ __constant__
    parameters::gfn2::PairScaleOverride g_gfn2_pair_overrides[kPairOverrideStorage];

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_PARAMETERS_CUH
