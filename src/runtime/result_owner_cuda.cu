// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cstring>
#include <string>

// The CUDA host-API shims provide cudaMalloc/cudaFree through the dlopen
// cohort, so this translation unit never links the CUDA runtime statically.
#include <cuda_runtime_api.h>

#include "runtime/result_owner.hpp"

namespace gpuxtb::detail {
namespace {

const char* cuda_error_name(cudaError_t status) {
  switch (status) {
    case cudaSuccess:
      return "cudaSuccess";
    case cudaErrorMemoryAllocation:
      return "cudaErrorMemoryAllocation";
    case cudaErrorInvalidValue:
      return "cudaErrorInvalidValue";
    case cudaErrorInvalidDevice:
      return "cudaErrorInvalidDevice";
    case cudaErrorInsufficientDriver:
      return "cudaErrorInsufficientDriver";
    case cudaErrorNoDevice:
      return "cudaErrorNoDevice";
    case cudaErrorDevicesUnavailable:
      return "cudaErrorDevicesUnavailable";
    default:
      return cudaGetErrorString(status);
  }
}

/* Restore the caller's previous device; reports failure but always returns. */
void restore_device(int previous_device, int selected_device) noexcept {
  if (previous_device == selected_device || previous_device < 0) {
    return;
  }
  (void)cudaSetDevice(previous_device);
}

}  // namespace

gpuxtb_status_t allocate_cuda_result_arena(std::int32_t device_id, std::size_t size_bytes,
                                           void** data, std::string& error) noexcept {
  if (data == nullptr) {
    error = "CUDA arena output pointer is NULL";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  *data = nullptr;
  if (device_id < 0) {
    error = "CUDA result arena requires a nonnegative device id";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (size_bytes == 0u) {
    error = "CUDA arena byte size must be nonzero";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  int previous_device = -1;
  if (cudaGetDevice(&previous_device) != cudaSuccess) {
    error = "failed to query the current CUDA device before allocating a result arena";
    return GPUXTB_STATUS_BACKEND_UNAVAILABLE;
  }
  const bool changed_device = previous_device != device_id;
  if (changed_device && cudaSetDevice(device_id) != cudaSuccess) {
    error = "failed to select CUDA device " + std::to_string(device_id) +
            " before allocating a result arena";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  cudaError_t status = cudaMalloc(data, size_bytes);
  const gpuxtb_status_t result =
      status == cudaSuccess
          ? GPUXTB_STATUS_SUCCESS
          : (status == cudaErrorMemoryAllocation ? GPUXTB_STATUS_ALLOCATION_FAILED
                                                 : GPUXTB_STATUS_BACKEND_UNAVAILABLE);
  if (status != cudaSuccess) {
    error = std::string("cudaMalloc failed: ") + cuda_error_name(status);
  }

  restore_device(previous_device, device_id);
  if (result != GPUXTB_STATUS_SUCCESS) {
    *data = nullptr;
  }
  return result;
}

void free_cuda_result_arena(std::int32_t device_id, void* data) noexcept {
  if (data == nullptr) {
    return;
  }
  int previous_device = -1;
  const bool have_previous = cudaGetDevice(&previous_device) == cudaSuccess;
  const bool changed_device = have_previous && previous_device != device_id;
  if (changed_device) {
    (void)cudaSetDevice(device_id);
  }
  (void)cudaFree(data);
  if (changed_device) {
    (void)cudaSetDevice(previous_device);
  }
}

}  // namespace gpuxtb::detail