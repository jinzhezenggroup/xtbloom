#ifndef XTBLOOM_CPU_DISPATCH_FEATURES_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_CPU_DISPATCH_FEATURES_HPP

#include <cstdint>
#include <string>

#include "xtbloom/xtbloom.h"

namespace xtbloom::detail {

/* CPU ISA selection is internal implementation state, frozen when a CPU
 * context is created. It is deliberately absent from the stable public ABI. */
enum class CpuIsa : std::uint8_t { kBaseline, kAvx2Fma, kAvx512Fma };

/* Keep the individual architectural and OS-state conditions visible so the
 * selector can be exhaustively tested without executing an AVX instruction. */
struct CpuFeatureSnapshot {
  bool avx = false;
  bool avx2 = false;
  bool fma = false;
  bool avx512f = false;
  bool os_xsave = false;
  bool xmm_ymm_state = false;
  bool zmm_state = false;

  [[nodiscard]] bool supports_avx2_fma() const noexcept {
    return avx && avx2 && fma && os_xsave && xmm_ymm_state;
  }

  [[nodiscard]] bool supports_avx512_fma() const noexcept {
    return supports_avx2_fma() && avx512f && zmm_state;
  }
};

[[nodiscard]] const char* cpu_isa_name(CpuIsa isa) noexcept;
[[nodiscard]] bool cpu_avx2_fma_kernels_built() noexcept;
[[nodiscard]] bool cpu_avx512_fma_kernels_built() noexcept;
[[nodiscard]] CpuFeatureSnapshot detect_cpu_features() noexcept;

/* Resolve an explicit testable request. A null request has the same meaning as
 * `auto`; all non-null values are matched exactly and without whitespace or
 * case normalization so evidence cannot silently describe a different mode. */
xtbloom_status_t resolve_cpu_isa_request(const char* request, bool avx2_kernels_built,
                                         bool avx512_kernels_built,
                                         const CpuFeatureSnapshot& features, CpuIsa& selected,
                                         std::string& error);

/* Read XTBLOOM_CPU_ISA once for one CPU context and freeze the resulting ISA.
 * CUDA callers do not invoke this function and therefore ignore the variable. */
xtbloom_status_t resolve_cpu_isa_from_environment(CpuIsa& selected, std::string& error);

}  // namespace xtbloom::detail

#endif  // XTBLOOM_CPU_DISPATCH_FEATURES_HPP
