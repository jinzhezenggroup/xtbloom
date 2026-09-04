#include "model/gfn2/mulliken_kernels.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

namespace xtbloom::detail::gfn2 {

#if !defined(XTBLOOM_HAS_AVX2_FMA_KERNELS)
const MullikenKernelTable& mulliken_avx2_fma_kernels() noexcept {
  /* Keep baseline-only builds linkable for generic diagnostics and tests that
   * mention the AVX2 table behind cpu_avx2_fma_kernels_built(). Selection can
   * never reach this fallback because the feature resolver rejects AVX2 first. */
  return mulliken_baseline_kernels();
}

const MullikenKernelTable& mulliken_avx512_fma_kernels() noexcept {
  /* The AVX-512 experiment is emitted from the AVX2/FMA object. A build that
   * omitted that object cannot advertise or select the higher ISA either. */
  return mulliken_baseline_kernels();
}
#endif

const MullikenKernelTable& mulliken_kernels_for_cpu_isa(CpuIsa isa) noexcept {
#if defined(XTBLOOM_HAS_AVX2_FMA_KERNELS)
  if (isa == CpuIsa::kAvx512Fma) {
    return mulliken_avx512_fma_kernels();
  }
  if (isa == CpuIsa::kAvx2Fma) {
    return mulliken_avx2_fma_kernels();
  }
#else
  (void)isa;
#endif
  return mulliken_baseline_kernels();
}

}  // namespace xtbloom::detail::gfn2
