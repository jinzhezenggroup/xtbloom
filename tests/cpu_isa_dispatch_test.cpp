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
using xtbloom::detail::cpu_avx512_fma_kernels_built;
using xtbloom::detail::cpu_isa_name;
using xtbloom::detail::CpuFeatureSnapshot;
using xtbloom::detail::CpuIsa;
using xtbloom::detail::detect_cpu_features;
using xtbloom::detail::resolve_cpu_isa_request;
using xtbloom::detail::gfn2::mulliken_avx2_fma_kernels;
using xtbloom::detail::gfn2::mulliken_avx512_fma_kernels;
using xtbloom::detail::gfn2::mulliken_baseline_kernels;
using xtbloom::detail::gfn2::mulliken_kernels_for_cpu_isa;
using xtbloom::detail::gfn2::MullikenKernelTable;

constexpr CpuFeatureSnapshot kAvx2Capable{true, true, true, false, true, true, false};
constexpr CpuFeatureSnapshot kAvx512Capable{true, true, true, true, true, true, true};

int test_exact_override_parsing_and_selection() {
  CpuIsa selected = CpuIsa::kBaseline;
  std::string error;
  CHECK(std::string(cpu_isa_name(CpuIsa::kBaseline)) == "baseline");
  CHECK(std::string(cpu_isa_name(CpuIsa::kAvx2Fma)) == "avx2");
  CHECK(std::string(cpu_isa_name(CpuIsa::kAvx512Fma)) == "avx512");

  CHECK(resolve_cpu_isa_request(nullptr, true, true, kAvx512Capable, selected, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(selected == CpuIsa::kAvx512Fma);
  CHECK(error.empty());

  CHECK(resolve_cpu_isa_request("auto", true, false, kAvx512Capable, selected, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(selected == CpuIsa::kAvx2Fma);
  CHECK(resolve_cpu_isa_request("auto", false, false, kAvx512Capable, selected, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(selected == CpuIsa::kBaseline);
  CHECK(resolve_cpu_isa_request("baseline", true, true, {}, selected, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(selected == CpuIsa::kBaseline);
  CHECK(resolve_cpu_isa_request("avx2", true, true, kAvx2Capable, selected, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(selected == CpuIsa::kAvx2Fma);
  CHECK(resolve_cpu_isa_request("avx512", true, true, kAvx512Capable, selected, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(selected == CpuIsa::kAvx512Fma);

  for (const char* invalid : std::array<const char*, 8>{
           {"", "AVX2", "AVX512", " avx2", "avx512 ", "native", "sse2", "avx-512"}}) {
    error.clear();
    CHECK(resolve_cpu_isa_request(invalid, true, true, kAvx512Capable, selected, error) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(error.find("XTBLOOM_CPU_ISA") != std::string::npos);
    CHECK(error.find("auto") != std::string::npos);
    CHECK(error.find("baseline") != std::string::npos);
    CHECK(error.find("avx2") != std::string::npos);
    CHECK(error.find("avx512") != std::string::npos);
  }
  return 0;
}

int test_avx2_capability_gates() {
  CpuIsa selected = CpuIsa::kBaseline;
  std::string error;
  CHECK(resolve_cpu_isa_request("avx2", false, false, kAvx2Capable, selected, error) ==
        XTBLOOM_STATUS_BACKEND_UNAVAILABLE);
  CHECK(error.find("no AVX2/FMA kernels") != std::string::npos);

  const std::array<CpuFeatureSnapshot, 5> missing{{
      {false, true, true, false, true, true, false},
      {true, false, true, false, true, true, false},
      {true, true, false, false, true, true, false},
      {true, true, true, false, false, true, false},
      {true, true, true, false, true, false, false},
  }};
  for (const CpuFeatureSnapshot& features : missing) {
    error.clear();
    CHECK(resolve_cpu_isa_request("auto", true, true, features, selected, error) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(selected == CpuIsa::kBaseline);
    CHECK(resolve_cpu_isa_request("avx2", true, true, features, selected, error) ==
          XTBLOOM_STATUS_BACKEND_UNAVAILABLE);
    CHECK(error.find("CPU or operating system") != std::string::npos);
  }
  return 0;
}

int test_avx512_capability_gates_and_fallback() {
  CpuIsa selected = CpuIsa::kBaseline;
  std::string error;
  CHECK(resolve_cpu_isa_request("avx512", true, false, kAvx512Capable, selected, error) ==
        XTBLOOM_STATUS_BACKEND_UNAVAILABLE);
  CHECK(error.find("no AVX-512/FMA kernels") != std::string::npos);

  /* Losing an AVX-512-only condition must retain the proven AVX2 fallback. */
  const std::array<CpuFeatureSnapshot, 2> missing_avx512_only{{
      {true, true, true, false, true, true, true},
      {true, true, true, true, true, true, false},
  }};
  for (const CpuFeatureSnapshot& features : missing_avx512_only) {
    error.clear();
    CHECK(resolve_cpu_isa_request("auto", true, true, features, selected, error) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(selected == CpuIsa::kAvx2Fma);
    CHECK(resolve_cpu_isa_request("avx512", true, true, features, selected, error) ==
          XTBLOOM_STATUS_BACKEND_UNAVAILABLE);
    CHECK(error.find("CPU or operating system") != std::string::npos);
  }

  /* Shared AVX/FMA/XMM/YMM gates disable both higher paths. */
  const std::array<CpuFeatureSnapshot, 5> missing_shared{{
      {false, true, true, true, true, true, true},
      {true, false, true, true, true, true, true},
      {true, true, false, true, true, true, true},
      {true, true, true, true, false, true, true},
      {true, true, true, true, true, false, true},
  }};
  for (const CpuFeatureSnapshot& features : missing_shared) {
    error.clear();
    CHECK(resolve_cpu_isa_request("auto", true, true, features, selected, error) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(selected == CpuIsa::kBaseline);
    CHECK(resolve_cpu_isa_request("avx512", true, true, features, selected, error) ==
          XTBLOOM_STATUS_BACKEND_UNAVAILABLE);
  }
  return 0;
}

int test_kernel_identity_and_real_host_selection() {
  const auto& baseline = mulliken_baseline_kernels();
  const CpuFeatureSnapshot actual = detect_cpu_features();
  CHECK(baseline.population != nullptr);
  CHECK(baseline.hamiltonian != nullptr);
  CHECK(baseline.isa == CpuIsa::kBaseline);
  CHECK(mulliken_kernels_for_cpu_isa(CpuIsa::kBaseline).population == baseline.population);

  /* Never enter an ISA-compiled translation unit on an incapable host merely
   * to inspect its table. Synthetic snapshots cover unavailable branches. */
  const MullikenKernelTable* avx2_table = nullptr;
  if (cpu_avx2_fma_kernels_built() && actual.supports_avx2_fma()) {
    const auto& avx2 = mulliken_avx2_fma_kernels();
    avx2_table = &avx2;
    CHECK(avx2.population != nullptr);
    CHECK(avx2.hamiltonian != nullptr);
    CHECK(avx2.isa == CpuIsa::kAvx2Fma);
    CHECK(avx2.population != baseline.population);
    CHECK(avx2.hamiltonian != baseline.hamiltonian);
    CHECK(mulliken_kernels_for_cpu_isa(CpuIsa::kAvx2Fma).population == avx2.population);
  }

  if (cpu_avx512_fma_kernels_built() && actual.supports_avx512_fma()) {
    const auto& avx512 = mulliken_avx512_fma_kernels();
    CHECK(avx512.population != nullptr);
    CHECK(avx512.hamiltonian != nullptr);
    CHECK(avx512.isa == CpuIsa::kAvx512Fma);
    CHECK(avx512.population != baseline.population);
    CHECK(avx512.hamiltonian != baseline.hamiltonian);
    if (avx2_table != nullptr) {
      CHECK(avx512.population != avx2_table->population);
      CHECK(avx512.hamiltonian != avx2_table->hamiltonian);
    }
    CHECK(mulliken_kernels_for_cpu_isa(CpuIsa::kAvx512Fma).population == avx512.population);
  }

  CpuIsa selected = CpuIsa::kBaseline;
  std::string error;
  CHECK(resolve_cpu_isa_request("auto", cpu_avx2_fma_kernels_built(),
                                cpu_avx512_fma_kernels_built(), actual, selected,
                                error) == XTBLOOM_STATUS_SUCCESS);
  const CpuIsa expected =
      cpu_avx512_fma_kernels_built() && actual.supports_avx512_fma() ? CpuIsa::kAvx512Fma
      : cpu_avx2_fma_kernels_built() && actual.supports_avx2_fma()   ? CpuIsa::kAvx2Fma
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
  if (const int line = test_avx2_capability_gates(); line != 0) {
    return line;
  }
  if (const int line = test_avx512_capability_gates_and_fallback(); line != 0) {
    return line;
  }
  return test_kernel_identity_and_real_host_selection();
}
