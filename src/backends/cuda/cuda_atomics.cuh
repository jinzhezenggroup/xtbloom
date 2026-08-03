#ifndef GPUXTB_BACKENDS_CUDA_CUDA_ATOMICS_CUH
#define GPUXTB_BACKENDS_CUDA_CUDA_ATOMICS_CUH

#include <cuda_runtime.h>

namespace gpuxtb::detail::cuda {

/*
 * FP64 atomic addition is native from sm_60 onward. CUDA can still compile
 * gpuxtb for older targets selected by a toolchain or downstream packager, so
 * retain the standard integer-CAS fallback there. Comparing integer payloads
 * instead of doubles also guarantees that a NaN already stored at address
 * cannot make the retry loop spin forever.
 */
__device__ inline double atomic_add_fp64(double* address, double value) noexcept {
#if !defined(GPUXTB_FORCE_FP64_ATOMIC_CAS) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
  return atomicAdd(address, value);
#else
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *bits;
  unsigned long long assumed = 0u;
  do {
    assumed = old;
    old = atomicCAS(bits, assumed, __double_as_longlong(value + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif
}

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_CUDA_ATOMICS_CUH
