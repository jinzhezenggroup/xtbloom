#include "runtime/cuda_scc_benchmark_mode.hpp"

#include <array>

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

int main() {
  using xtbloom::detail::CudaSccBenchmarkMode;
  using xtbloom::detail::parse_cuda_scc_benchmark_mode;

  const auto unset = parse_cuda_scc_benchmark_mode(nullptr);
  CHECK(unset.valid);
  CHECK(unset.mode == CudaSccBenchmarkMode::kAuto);
  const auto automatic = parse_cuda_scc_benchmark_mode("auto");
  CHECK(automatic.valid);
  CHECK(automatic.mode == CudaSccBenchmarkMode::kAuto);
  const auto tail = parse_cuda_scc_benchmark_mode("tail");
  CHECK(tail.valid);
  CHECK(tail.mode == CudaSccBenchmarkMode::kDeviceTailGraph);
  const auto chain = parse_cuda_scc_benchmark_mode("chain");
  CHECK(chain.valid);
  CHECK(chain.mode == CudaSccBenchmarkMode::kDeviceDispatchChain);

  constexpr std::array<const char*, 9> invalid_modes{
      "", "AUTO", "Chain", " tail", "chain ", "bounded", "0", "tail\n", "chain-tail"};
  for (const char* mode : invalid_modes) {
    const auto selection = parse_cuda_scc_benchmark_mode(mode);
    CHECK(!selection.valid);
    CHECK(selection.mode == CudaSccBenchmarkMode::kAuto);
  }

  /* Mutating the source buffer cannot change a captured owner's selection. */
  char environment_storage[] = "chain";
  const auto captured = parse_cuda_scc_benchmark_mode(environment_storage);
  environment_storage[0] = 't';
  CHECK(captured.valid);
  CHECK(captured.mode == CudaSccBenchmarkMode::kDeviceDispatchChain);
  CHECK(!parse_cuda_scc_benchmark_mode(environment_storage).valid);
  return 0;
}
