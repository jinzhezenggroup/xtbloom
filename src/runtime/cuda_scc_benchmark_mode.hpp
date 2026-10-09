#ifndef XTBLOOM_RUNTIME_CUDA_SCC_BENCHMARK_MODE_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_RUNTIME_CUDA_SCC_BENCHMARK_MODE_HPP

#include <cstdint>
#include <string_view>

namespace xtbloom::detail {

enum class CudaSccBenchmarkMode : std::uint32_t {
  kAuto = 0u,
  kDeviceTailGraph = 1u,
  kDeviceDispatchChain = 2u,
};

struct CudaSccBenchmarkSelection {
  CudaSccBenchmarkMode mode = CudaSccBenchmarkMode::kAuto;
  bool valid = true;
};

/* Return a value-only selection: cache owners must snapshot this once rather
 * than retaining getenv storage or rereading a mutable environment at replay.
 * An unset variable preserves production selection; malformed explicit input
 * must not silently select a convenient benchmark baseline. */
[[nodiscard]] inline CudaSccBenchmarkSelection parse_cuda_scc_benchmark_mode(
    const char* mode) noexcept {
  if (mode == nullptr) return {};
  const std::string_view value(mode);
  if (value == "auto") return {};
  if (value == "tail") return {CudaSccBenchmarkMode::kDeviceTailGraph, true};
  if (value == "chain") return {CudaSccBenchmarkMode::kDeviceDispatchChain, true};
  return {CudaSccBenchmarkMode::kAuto, false};
}

}  // namespace xtbloom::detail

#endif
