// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "runtime/result_owner.hpp"

#include <new>
#include <string>

namespace gpuxtb::detail {
namespace {

constexpr std::size_t kHostArenaAlignment = 64u;

}  // namespace

gpuxtb_status_t allocate_host_result_arena(std::size_t size_bytes, void** data,
                                           std::string& error) noexcept {
  if (data == nullptr) {
    error = "host arena output pointer is NULL";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  *data = nullptr;
  if (size_bytes == 0u) {
    error = "host arena byte size must be nonzero";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  void* pointer = ::operator new(size_bytes, std::align_val_t(kHostArenaAlignment), std::nothrow);
  if (pointer == nullptr) {
    error = "failed to allocate a " + std::to_string(size_bytes) + "-byte host result arena";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  }
  *data = pointer;
  return GPUXTB_STATUS_SUCCESS;
}

void free_host_result_arena(void* data) noexcept {
  if (data != nullptr) {
    ::operator delete(data, std::align_val_t(kHostArenaAlignment));
  }
}

std::size_t dlpack_dtype_size(std::int32_t code, std::int32_t bits) noexcept {
  switch (bits) {
    case 8:
      if (code == kDlpackInt || code == kDlpackUInt) return 1u;
      break;
    case 16:
      if (code == kDlpackInt) return 2u;
      break;
    case 32:
      if (code == kDlpackInt || code == kDlpackFloat) return 4u;
      break;
    case 64:
      if (code == kDlpackInt || code == kDlpackFloat) return 8u;
      break;
  }
  return 0u;
}

std::int32_t dlpack_device_type(gpuxtb_memory_space_t memory_space) noexcept {
  switch (memory_space) {
    case GPUXTB_MEMORY_HOST:
      return kDlpackCpu;
    case GPUXTB_MEMORY_CUDA_DEVICE:
      return kDlpackCuda;
    default:
      return -1;
  }
}

void ResultOwner::destroy() noexcept {
  void* pointer = data_;
  if (memory_space_ == GPUXTB_MEMORY_HOST) {
    free_host_result_arena(pointer);
#if defined(GPUXTB_HAS_CUDA)
  } else if (memory_space_ == GPUXTB_MEMORY_CUDA_DEVICE) {
    free_cuda_result_arena(device_id_, pointer);
#endif
  } else {
    /* Reserved ROCm arenas are never allocatable; this path is defensive. */
    (void)pointer;
  }
  delete this;
}

}  // namespace gpuxtb::detail
