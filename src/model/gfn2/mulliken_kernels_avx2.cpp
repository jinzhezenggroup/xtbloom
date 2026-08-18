#include "model/gfn2/mulliken_kernels.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn2/mulliken_kernels_impl.hpp"

#if defined(_MSC_VER)
#define XTBLOOM_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define XTBLOOM_NOINLINE __attribute__((noinline))
#else
#define XTBLOOM_NOINLINE
#endif

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#if defined(__has_attribute)
#if __has_attribute(target)
#define XTBLOOM_AVX512_TARGET __attribute__((target("avx512f,fma")))
#else
#define XTBLOOM_AVX512_TARGET
#endif
#if __has_attribute(flatten)
#define XTBLOOM_FLATTEN __attribute__((flatten))
#else
#define XTBLOOM_FLATTEN
#endif
#else
#define XTBLOOM_AVX512_TARGET __attribute__((target("avx512f,fma")))
#define XTBLOOM_FLATTEN __attribute__((flatten))
#endif
#else
#define XTBLOOM_AVX512_TARGET
#define XTBLOOM_FLATTEN
#endif

namespace xtbloom::detail::gfn2 {

XTBLOOM_NOINLINE void mulliken_population_chunk_avx2_fma(void* opaque, std::size_t chunk) noexcept {
  kernel_implementation::population_chunk(opaque, chunk);
}

XTBLOOM_NOINLINE void mulliken_hamiltonian_chunk_avx2_fma(void* opaque,
                                                          std::size_t chunk) noexcept {
  kernel_implementation::hamiltonian_chunk(opaque, chunk);
}

/* Keep the AVX-512 experiment in the already isolated x86 ISA translation
 * unit so baseline objects remain load-safe and no new public build option is
 * introduced. `flatten` encourages the canonical internal implementation to
 * be cloned into the higher-target wrapper rather than leaving this as a thin
 * AVX2 call-through. Performance evidence must still inspect the emitted
 * object before treating this as a wider-vector optimization. */
XTBLOOM_AVX512_TARGET XTBLOOM_FLATTEN XTBLOOM_NOINLINE void
mulliken_population_chunk_avx512_fma(void* opaque, std::size_t chunk) noexcept {
  kernel_implementation::population_chunk(opaque, chunk);
}

XTBLOOM_AVX512_TARGET XTBLOOM_FLATTEN XTBLOOM_NOINLINE void
mulliken_hamiltonian_chunk_avx512_fma(void* opaque, std::size_t chunk) noexcept {
  kernel_implementation::hamiltonian_chunk(opaque, chunk);
}

const MullikenKernelTable& mulliken_avx2_fma_kernels() noexcept {
  static constexpr MullikenKernelTable kernels{
      &mulliken_population_chunk_avx2_fma, &mulliken_hamiltonian_chunk_avx2_fma, CpuIsa::kAvx2Fma};
  return kernels;
}

const MullikenKernelTable& mulliken_avx512_fma_kernels() noexcept {
  static constexpr MullikenKernelTable kernels{&mulliken_population_chunk_avx512_fma,
                                                &mulliken_hamiltonian_chunk_avx512_fma,
                                                CpuIsa::kAvx512Fma};
  return kernels;
}

}  // namespace xtbloom::detail::gfn2

#undef XTBLOOM_FLATTEN
#undef XTBLOOM_AVX512_TARGET
#undef XTBLOOM_NOINLINE
