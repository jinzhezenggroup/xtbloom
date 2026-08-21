#include "cpu_dispatch/features.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

namespace xtbloom::detail {
namespace {

constexpr const char* kCpuIsaEnvironment = "XTBLOOM_CPU_ISA";

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#if defined(__has_builtin)
#define XTBLOOM_HAS_CPU_BUILTINS \
  (__has_builtin(__builtin_cpu_init) && __has_builtin(__builtin_cpu_supports))
#else
#define XTBLOOM_HAS_CPU_BUILTINS 1
#endif
#if defined(__has_attribute)
#define XTBLOOM_HAS_AVX512_TARGET_ATTRIBUTE __has_attribute(target)
#else
#define XTBLOOM_HAS_AVX512_TARGET_ATTRIBUTE 1
#endif
#else
#define XTBLOOM_HAS_CPU_BUILTINS 0
#define XTBLOOM_HAS_AVX512_TARGET_ATTRIBUTE 0
#endif

}  // namespace

const char* cpu_isa_name(CpuIsa isa) noexcept {
  switch (isa) {
    case CpuIsa::kAvx512Fma:
      return "avx512";
    case CpuIsa::kAvx2Fma:
      return "avx2";
    case CpuIsa::kBaseline:
    default:
      return "baseline";
  }
}

bool cpu_avx2_fma_kernels_built() noexcept {
#if defined(XTBLOOM_HAS_AVX2_FMA_KERNELS)
  return true;
#else
  return false;
#endif
}

bool cpu_avx512_fma_kernels_built() noexcept {
#if defined(XTBLOOM_HAS_AVX2_FMA_KERNELS) && XTBLOOM_HAS_AVX512_TARGET_ATTRIBUTE
  /* The experimental AVX-512 leaf is emitted from the isolated AVX2/FMA
   * translation unit with a per-function target attribute. MSVC deliberately
   * reports false until it has a separately compiled /arch:AVX512 object. */
  return true;
#else
  return false;
#endif
}

CpuFeatureSnapshot detect_cpu_features() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  int registers[4] = {0, 0, 0, 0};
  __cpuid(registers, 0);
  const int maximum_leaf = registers[0];
  if (maximum_leaf < 1) {
    return {};
  }
  __cpuidex(registers, 1, 0);
  CpuFeatureSnapshot features;
  const unsigned int ecx = static_cast<unsigned int>(registers[2]);
  features.fma = (ecx & (1u << 12u)) != 0u;
  const bool xsave = (ecx & (1u << 26u)) != 0u;
  features.os_xsave = xsave && (ecx & (1u << 27u)) != 0u;
  features.avx = (ecx & (1u << 28u)) != 0u;
  if (features.os_xsave) {
    /* XGETBV is safe after the OSXSAVE gate and is not itself an AVX
     * instruction. AVX2 needs XMM/YMM; AVX-512 additionally needs opmask,
     * ZMM_Hi256, and Hi16_ZMM state (XCR0 bits 5, 6, and 7). */
    const unsigned __int64 xcr0 = _xgetbv(0);
    features.xmm_ymm_state = (xcr0 & 0x6u) == 0x6u;
    features.zmm_state = (xcr0 & 0xe6u) == 0xe6u;
  }
  if (maximum_leaf >= 7) {
    __cpuidex(registers, 7, 0);
    const unsigned int ebx = static_cast<unsigned int>(registers[1]);
    features.avx2 = (ebx & (1u << 5u)) != 0u;
    features.avx512f = (ebx & (1u << 16u)) != 0u;
  }
  return features;
#elif XTBLOOM_HAS_CPU_BUILTINS
  /* GCC and Clang's x86 CPU builtins include the operating-system extended
   * state gate. A reported AVX/AVX-512 capability therefore also proves the
   * corresponding XCR0 state without executing AVX in this translation unit. */
  __builtin_cpu_init();
  CpuFeatureSnapshot features;
  features.avx = __builtin_cpu_supports("avx");
  features.avx2 = __builtin_cpu_supports("avx2");
  features.fma = __builtin_cpu_supports("fma");
  features.avx512f = __builtin_cpu_supports("avx512f");
  features.os_xsave = features.avx;
  features.xmm_ymm_state = features.avx;
  features.zmm_state = features.avx512f;
  return features;
#else
  return {};
#endif
}

xtbloom_status_t resolve_cpu_isa_request(const char* request, bool avx2_kernels_built,
                                         bool avx512_kernels_built,
                                         const CpuFeatureSnapshot& features, CpuIsa& selected,
                                         std::string& error) {
  const char* mode = request == nullptr ? "auto" : request;
  if (std::strcmp(mode, "baseline") == 0) {
    selected = CpuIsa::kBaseline;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }
  if (std::strcmp(mode, "auto") == 0) {
    if (avx512_kernels_built && features.supports_avx512_fma()) {
      selected = CpuIsa::kAvx512Fma;
    } else if (avx2_kernels_built && features.supports_avx2_fma()) {
      selected = CpuIsa::kAvx2Fma;
    } else {
      selected = CpuIsa::kBaseline;
    }
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }
  if (std::strcmp(mode, "avx512") == 0) {
    if (!avx512_kernels_built) {
      error =
          "XTBLOOM_CPU_ISA=avx512 was requested, but this xTBloom build has no AVX-512/FMA "
          "kernels";
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }
    if (!features.supports_avx512_fma()) {
      error =
          "XTBLOOM_CPU_ISA=avx512 was requested, but the CPU or operating system does not "
          "support AVX-512F with FMA and enabled XMM/YMM/ZMM/opmask state";
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }
    selected = CpuIsa::kAvx512Fma;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }
  if (std::strcmp(mode, "avx2") != 0) {
    error = "XTBLOOM_CPU_ISA must be exactly one of auto, baseline, avx2, or avx512 when it is set";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!avx2_kernels_built) {
    error = "XTBLOOM_CPU_ISA=avx2 was requested, but this xTBloom build has no AVX2/FMA kernels";
    return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
  }
  if (!features.supports_avx2_fma()) {
    error =
        "XTBLOOM_CPU_ISA=avx2 was requested, but the CPU or operating system does not support "
        "AVX2 with FMA and enabled XMM/YMM state";
    return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
  }
  selected = CpuIsa::kAvx2Fma;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t resolve_cpu_isa_from_environment(CpuIsa& selected, std::string& error) {
  return resolve_cpu_isa_request(std::getenv(kCpuIsaEnvironment), cpu_avx2_fma_kernels_built(),
                                 cpu_avx512_fma_kernels_built(), detect_cpu_features(), selected,
                                 error);
}

}  // namespace xtbloom::detail

#undef XTBLOOM_HAS_AVX512_TARGET_ATTRIBUTE
#undef XTBLOOM_HAS_CPU_BUILTINS
