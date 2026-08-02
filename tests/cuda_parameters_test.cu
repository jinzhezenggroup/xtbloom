#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

#include "runtime/backend.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

int main() {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    /* CUDA-enabled packages must remain testable on CPU-only CI runners. */
    return 0;
  }

  int device = -1;
  CHECK(cudaGetDevice(&device) == cudaSuccess);
  std::string error;
  CHECK(gpuxtb::detail::ensure_cuda_gfn2_parameters(device, error));
  const std::uint64_t first_count = gpuxtb::detail::cuda_gfn2_parameter_upload_count(device);
  CHECK(first_count == 1u);
  CHECK(gpuxtb::detail::ensure_cuda_gfn2_parameters(device, error));
  CHECK(gpuxtb::detail::cuda_gfn2_parameter_upload_count(device) == first_count);

  CHECK(gpuxtb::detail::cuda_gfn2_parameters_match_host(device, error));
  return 0;
}
