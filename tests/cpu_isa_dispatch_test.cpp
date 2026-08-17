#include <array>
#include <cstdio>
#include <string>

#include "cpu_dispatch/features.hpp"
#include "model/gfn2/mulliken_kernels.hpp"

#define CHECK(condition)                                                                   \
  do {                                                                                     \
    if (!(condition)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      return __LINE__;                                                                     \
    }                                                                                      \
  } while (false)

namespace {

using xtbloom::detail::cpu_avx2_fma_kernels_built;
using xtbloom::detail::CpuFeatureSnapshot;
using xtbloom::detail::CpuIsa;
using xtbloom::detail::detect_cpu_features;
using xtbloom::detail::resolve_cpu_isa_request;
using xtbloom::detail::gfn2::mulliken_avx2_fma_kernels;
using xtbloom::detail::gfn2::mulliken_baseline_kernels;
using xtbloom::detail::gfn2::mulliken_kernels_for_cpu_isa;

constexpr CpuFeatureSnapshot kCapable{true, true, true, true, true};

int test_exact_override_parsing_and_selection() {
  CpuIsa selected = CpuIsa::kBaseline;
  std::string error;
  CHECK(resolve_cpu_isa_request(nullptr, true, kCapable, selected, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(selected == CpuIsa::kAvx2Fma);
  CHECK(error.empty());

  CHECK(resolve_cpu_isa_request("auto", false, kCapable, selected, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(selected == CpuIsa::kBaseline);
  CHECK(resolve_cpu_isa_request("baseline", true, {}, selected, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(selected == CpuIsa::kBaseline);
  CHECK(resolve_cpu_isa_request("avx2", true, kCapable, selected, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(selected == CpuIsa::kAvx2Fma);

  for (const char* invalid :
       std::array<const char*, 6>{{"", "AVX2", " avx2", "avx2 ", "native", "sse2"}}) {
    error.clear();
    CHECK(resolve_cpu_isa_request(invalid, true, kCapable, selected, error) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(error.find("XTBLOOM_CPU_ISA") != std::string::npos);
    CHECK(error.find("auto") != std::string::npos);
    CHECK(error.find("baseline") != std::string::npos);
    CHECK(error.find("avx2") != std::string::npos);
  }
  return 0;
}

int test_every_capability_gate() {
  CpuIsa selected = CpuIsa::kBaseline;
  std::string error;
  CHECK(resolve_cpu_isa_request("avx2", false, kCapable, selected, error) ==
        XTBLOOM_STATUS_BACKEND_UNAVAILABLE);
  CHECK(error.find("no AVX2/FMA kernels") != std::string::npos);

  const std::array<CpuFeatureSnapshot, 5> missing{{
      {false, true, true, true, true},
      {true, false, true, true, true},
      {true, true, false, true, true},
      {true, true, true, false, true},
      {true, true, true, true, false},
  }};
  for (const CpuFeatureSnapshot& features : missing) {
    error.clear();
    CHECK(resolve_cpu_isa_request("auto", true, features, selected, error) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(selected == CpuIsa::kBaseline);
    CHECK(resolve_cpu_isa_request("avx2", true, features, selected, error) ==
          XTBLOOM_STATUS_BACKEND_UNAVAILABLE);
    CHECK(error.find("CPU or operating system") != std::string::npos);
  }
  return 0;
}

int test_kernel_identity_and_real_host_selection() {
  const auto& baseline = mulliken_baseline_kernels();
  CHECK(baseline.population != nullptr);
  CHECK(baseline.hamiltonian != nullptr);
  CHECK(baseline.isa == CpuIsa::kBaseline);
  CHECK(mulliken_kernels_for_cpu_isa(CpuIsa::kBaseline).population == baseline.population);

  if (cpu_avx2_fma_kernels_built()) {
    const auto& avx2 = mulliken_avx2_fma_kernels();
    CHECK(avx2.population != nullptr);
    CHECK(avx2.hamiltonian != nullptr);
    CHECK(avx2.isa == CpuIsa::kAvx2Fma);
    CHECK(avx2.population != baseline.population);
    CHECK(avx2.hamiltonian != baseline.hamiltonian);
    CHECK(mulliken_kernels_for_cpu_isa(CpuIsa::kAvx2Fma).population == avx2.population);
  }

  CpuIsa selected = CpuIsa::kBaseline;
  std::string error;
  const CpuFeatureSnapshot actual = detect_cpu_features();
  CHECK(resolve_cpu_isa_request("auto", cpu_avx2_fma_kernels_built(), actual, selected, error) ==
        XTBLOOM_STATUS_SUCCESS);
  const CpuIsa expected = cpu_avx2_fma_kernels_built() && actual.supports_avx2_fma()
                              ? CpuIsa::kAvx2Fma
                              : CpuIsa::kBaseline;
  CHECK(selected == expected);
  CHECK(mulliken_kernels_for_cpu_isa(selected).isa == expected);
  return 0;
}

}  // namespace

int main() {
  if (const int line = test_exact_override_parsing_and_selection(); line != 0) {
    return line;
  }
  if (const int line = test_every_capability_gate(); line != 0) {
    return line;
  }
  return test_kernel_identity_and_real_host_selection();
}
