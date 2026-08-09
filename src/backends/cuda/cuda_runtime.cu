#include <cuda_runtime_api.h>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "runtime/backend.hpp"

namespace xtbloom::detail {

bool resolve_cuda_device(std::int32_t requested_device, std::int32_t& resolved_device,
                         std::string& error) {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess) {
    error = std::string("CUDA runtime initialization failed: ") + cudaGetErrorString(count_status);
    return false;
  }
  if (device_count == 0) {
    error = "no CUDA devices are visible";
    return false;
  }
  if (requested_device >= device_count) {
    error = "the requested CUDA device index is not visible";
    return false;
  }
  if (requested_device >= 0) {
    resolved_device = requested_device;
    return true;
  }

  int current_device = -1;
  const cudaError_t device_status = cudaGetDevice(&current_device);
  if (device_status != cudaSuccess) {
    error = std::string("failed to query the current CUDA device: ") +
            cudaGetErrorString(device_status);
    return false;
  }
  resolved_device = current_device;
  return true;
}

}  // namespace xtbloom::detail
