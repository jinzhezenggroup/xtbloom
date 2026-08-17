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

namespace xtbloom::detail::gfn2 {

XTBLOOM_NOINLINE void mulliken_population_chunk_baseline(void* opaque,
                                                         std::size_t chunk) noexcept {
  kernel_implementation::population_chunk(opaque, chunk);
}

XTBLOOM_NOINLINE void mulliken_hamiltonian_chunk_baseline(void* opaque,
                                                          std::size_t chunk) noexcept {
  kernel_implementation::hamiltonian_chunk(opaque, chunk);
}

const MullikenKernelTable& mulliken_baseline_kernels() noexcept {
  static constexpr MullikenKernelTable kernels{&mulliken_population_chunk_baseline,
                                                &mulliken_hamiltonian_chunk_baseline,
                                                CpuIsa::kBaseline};
  return kernels;
}

}  // namespace xtbloom::detail::gfn2

#undef XTBLOOM_NOINLINE
