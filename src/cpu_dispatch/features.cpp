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

#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
#if defined(__has_builtin)
#define XTBLOOM_HAS_CPU_BUILTINS \
  (__has_builtin(__builtin_cpu_init) && __has_builtin(__builtin_cpu_supports))
#else
#define XTBLOOM_HAS_CPU_BUILTINS 1
#endif
#else
#define XTBLOOM_HAS_CPU_BUILTINS 0
#endif

}  // namespace

const char* cpu_isa_name(CpuIsa isa) noexcept {
  return isa == CpuIsa::kAvx2Fma ? "avx2" : "baseline";
}

bool cpu_avx2_fma_kernels_built() noexcept {
#if defined(XTBLOOM_HAS_AVX2_FMA_KERNELS)
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
     * instruction. XMM and YMM state must both be enabled by the OS. */
    features.xmm_ymm_state = (_xgetbv(0) & 0x6u) == 0x6u;
  }
  if (maximum_leaf >= 7) {
    __cpuidex(registers, 7, 0);
    features.avx2 =
        (static_cast<unsigned int>(registers[1]) & (1u << 5u)) != 0u;
  }
  return features;
#elif XTBLOOM_HAS_CPU_BUILTINS
  /* GCC and Clang's x86 CPU builtins include the operating-system AVX state
   * gate. A reported AVX capability therefore also proves OSXSAVE plus the
   * required XMM/YMM XCR0 state without executing AVX in this translation unit. */
  __builtin_cpu_init();
  CpuFeatureSnapshot features;
  features.avx = __builtin_cpu_supports("avx");
  features.avx2 = __builtin_cpu_supports("avx2");
  features.fma = __builtin_cpu_supports("fma");
  features.os_xsave = features.avx;
  features.xmm_ymm_state = features.avx;
  return features;
#else
  return {};
#endif
}

xtbloom_status_t resolve_cpu_isa_request(const char* request, bool avx2_kernels_built,
                                         const CpuFeatureSnapshot& features, CpuIsa& selected,
                                         std::string& error) {
  const char* mode = request == nullptr ? "auto" : request;
  if (std::strcmp(mode, "baseline") == 0) {
    selected = CpuIsa::kBaseline;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }
  if (std::strcmp(mode, "auto") == 0) {
    selected = avx2_kernels_built && features.supports_avx2_fma() ? CpuIsa::kAvx2Fma
                                                                  : CpuIsa::kBaseline;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }
  if (std::strcmp(mode, "avx2") != 0) {
    error =
        "XTBLOOM_CPU_ISA must be exactly one of auto, baseline, or avx2 when it is set";
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
                                 detect_cpu_features(), selected, error);
}

}  // namespace xtbloom::detail

#undef XTBLOOM_HAS_CPU_BUILTINS
