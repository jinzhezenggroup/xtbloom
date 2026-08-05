#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "runtime/cuda_descriptor_validation.hpp"

namespace gpuxtb::detail {
namespace {

struct PointerFacts {
  cudaMemoryType type = cudaMemoryTypeUnregistered;
  std::int32_t allocation_device = -1;
};

const char* display_name(const char* name) noexcept {
  return name == nullptr || name[0] == '\0' ? "buffer" : name;
}

std::string cuda_error_message(const char* operation, cudaError_t status) {
  return std::string(operation) + " failed: " + cudaGetErrorString(status);
}

std::string cuda_driver_error_message(const char* operation, CUresult status) {
  const char* detail = nullptr;
  if (cuGetErrorString(status, &detail) != CUDA_SUCCESS || detail == nullptr) {
    return std::string(operation) + " failed with CUDA driver status " +
           std::to_string(static_cast<int>(status));
  }
  return std::string(operation) + " failed: " + detail;
}

/* Failed metadata queries are validation diagnostics, not deferred launch
 * failures. Consume their last-error entry so a handled bad descriptor cannot
 * poison a later cudaPeekAtLastError() in the execution path. */
void consume_validation_error(cudaError_t status) noexcept {
  if (status != cudaSuccess) (void)cudaGetLastError();
}

gpuxtb_status_t invalid(const char* name, const char* detail, std::string& error) {
  error = std::string(display_name(name)) + detail;
  return GPUXTB_STATUS_INVALID_ARGUMENT;
}

gpuxtb_status_t finish_with_restore(ScopedCudaDevice& device, gpuxtb_status_t status,
                                    std::string& error) {
  std::string restore_error;
  const gpuxtb_status_t restore_status = device.restore(restore_error);
  if (restore_status == GPUXTB_STATUS_SUCCESS) return status;

  if (status != GPUXTB_STATUS_SUCCESS && !error.empty()) {
    error += "; additionally, " + restore_error;
  } else {
    error = std::move(restore_error);
  }
  return restore_status;
}

gpuxtb_status_t validate_basic_descriptor(const char* name, const void* data,
                                          std::size_t size_bytes,
                                          gpuxtb_memory_space_t memory_space,
                                          std::uint32_t reserved, std::size_t logical_bytes,
                                          std::size_t alignment, std::string& error) {
  if (reserved != 0u) return invalid(name, ".reserved must be zero", error);
  if (memory_space == GPUXTB_MEMORY_ROCM_DEVICE) {
    error = std::string(display_name(name)) + " uses reserved ROCm device memory";
    return GPUXTB_STATUS_NOT_SUPPORTED;
  }
  if (memory_space != GPUXTB_MEMORY_HOST && memory_space != GPUXTB_MEMORY_CUDA_DEVICE) {
    return invalid(name, " has an unknown memory_space value", error);
  }
  if (data == nullptr && size_bytes != 0u) {
    return invalid(name, " has nonzero size_bytes but a NULL data pointer", error);
  }
  if (logical_bytes == 0u) return GPUXTB_STATUS_SUCCESS;
  if (data == nullptr) return invalid(name, " is required but its data pointer is NULL", error);
  if (size_bytes < logical_bytes) {
    return invalid(name, " is smaller than the required logical byte range", error);
  }
  if (alignment == 0u || (alignment & (alignment - 1u)) != 0u) {
    return invalid(name, " was validated with a non-power-of-two alignment", error);
  }

  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(data);
  if ((address & (alignment - 1u)) != 0u) {
    return invalid(name, " does not satisfy the required alignment", error);
  }
  /* size_bytes describes the caller's complete view, not merely the prefix
   * consumed by the current operation.  Reject an impossible half-open range
   * before asking CUDA about allocation ownership. */
  if (size_bytes > std::numeric_limits<std::uintptr_t>::max() - address) {
    return invalid(name, " declared address range overflows uintptr_t", error);
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_device_allocation_range(const char* name, const void* data,
                                                 std::size_t declared_bytes, std::string& error) {
  CUdeviceptr allocation_base = 0u;
  std::size_t allocation_bytes = 0u;
  const CUresult range_status = cuMemGetAddressRange(&allocation_base, &allocation_bytes,
                                                     reinterpret_cast<CUdeviceptr>(data));
  if (range_status != CUDA_SUCCESS) {
    error = cuda_driver_error_message("cuMemGetAddressRange", range_status);
    if (range_status == CUDA_ERROR_INVALID_VALUE || range_status == CUDA_ERROR_NOT_FOUND) {
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    if (range_status == CUDA_ERROR_DEINITIALIZED || range_status == CUDA_ERROR_NOT_INITIALIZED ||
        range_status == CUDA_ERROR_INVALID_CONTEXT) {
      return GPUXTB_STATUS_BACKEND_UNAVAILABLE;
    }
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }

  /* cudaMemGetAddressRange accepts an interior device pointer and returns the
   * containing allocation.  Use subtraction rather than constructing an end
   * address so the validation itself cannot wrap uintptr_t. */
  if (allocation_base == 0u || allocation_bytes == 0u) {
    return invalid(name, " has invalid CUDA allocation-range metadata", error);
  }
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(data);
  const std::uintptr_t base = static_cast<std::uintptr_t>(allocation_base);
  if (address < base) {
    return invalid(name, " lies before its reported CUDA allocation", error);
  }
  const std::uintptr_t offset = address - base;
  if (offset > allocation_bytes || declared_bytes > allocation_bytes - offset) {
    return invalid(name, " declared byte range extends past its CUDA allocation", error);
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_pointer_facts(std::int32_t device_id, const char* name, const void* data,
                                       gpuxtb_memory_space_t memory_space,
                                       CudaManagedMemoryPolicy managed_policy, bool writable,
                                       PointerFacts& facts, std::string& error) {
  if (managed_policy != CudaManagedMemoryPolicy::kReject &&
      managed_policy != CudaManagedMemoryPolicy::kAllowOnAllocationDevice) {
    return invalid(name, " has an unknown managed-memory policy", error);
  }

  cudaPointerAttributes attributes{};
  const cudaError_t attribute_status = cudaPointerGetAttributes(&attributes, data);
  if (attribute_status == cudaErrorInvalidValue) {
    /* Older runtimes report an ordinary host pointer as invalid instead of
     * returning cudaMemoryTypeUnregistered. Both forms have the same meaning. */
    consume_validation_error(attribute_status);
    facts = {};
  } else if (attribute_status != cudaSuccess) {
    consume_validation_error(attribute_status);
    error = cuda_error_message("cudaPointerGetAttributes", attribute_status);
    return attribute_status == cudaErrorInvalidDevice ? GPUXTB_STATUS_BACKEND_UNAVAILABLE
                                                      : GPUXTB_STATUS_INTERNAL_ERROR;
  } else {
    facts.type = attributes.type;
    facts.allocation_device = attributes.device;
  }

  if (memory_space == GPUXTB_MEMORY_HOST) {
    if (facts.type == cudaMemoryTypeDevice || facts.type == cudaMemoryTypeManaged) {
      return invalid(name, " is CUDA-accessible memory mislabeled as HOST", error);
    }
    if (facts.type != cudaMemoryTypeUnregistered && facts.type != cudaMemoryTypeHost) {
      return invalid(name, " has an unsupported CUDA pointer type for HOST memory", error);
    }
    if (facts.type == cudaMemoryTypeHost && attributes.hostPointer == nullptr) {
      return invalid(name, " is registered host memory without a host-accessible alias", error);
    }
    if (writable && facts.type == cudaMemoryTypeHost) {
      unsigned int flags = 0u;
      const cudaError_t flag_status = cudaHostGetFlags(&flags, const_cast<void*>(data));
      if (flag_status != cudaSuccess) {
        consume_validation_error(flag_status);
        error = cuda_error_message("cudaHostGetFlags", flag_status);
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      if ((flags & cudaHostRegisterReadOnly) != 0u) {
        return invalid(name, " is a read-only CUDA host registration used as writable output",
                       error);
      }
    }
    return GPUXTB_STATUS_SUCCESS;
  }

  if (facts.type == cudaMemoryTypeManaged) {
    if (managed_policy != CudaManagedMemoryPolicy::kAllowOnAllocationDevice) {
      return invalid(name, " is managed memory but this bridge rejects managed storage", error);
    }
    if (facts.allocation_device != device_id) {
      return invalid(name, " is managed memory allocated against a different CUDA device", error);
    }
    if (attributes.devicePointer == nullptr) {
      return invalid(name, " is managed memory without a device-accessible alias", error);
    }
    return GPUXTB_STATUS_SUCCESS;
  }
  if (facts.type != cudaMemoryTypeDevice) {
    return invalid(name, " is host memory mislabeled as CUDA_DEVICE", error);
  }
  if (facts.allocation_device != device_id) {
    return invalid(name, " belongs to a different CUDA device", error);
  }
  if (attributes.devicePointer == nullptr) {
    return invalid(name, " has no device-accessible alias on the context device", error);
  }
  return GPUXTB_STATUS_SUCCESS;
}

}  // namespace

ScopedCudaDevice::ScopedCudaDevice(std::int32_t device_id, std::string& error)
    : selected_device_(device_id) {
  error.clear();
  if (device_id < 0) {
    error = "CUDA device id must be nonnegative";
    status_ = GPUXTB_STATUS_INVALID_ARGUMENT;
    return;
  }

  cudaError_t cuda_status = cudaGetDevice(&previous_device_);
  if (cuda_status != cudaSuccess) {
    consume_validation_error(cuda_status);
    error = cuda_error_message("cudaGetDevice", cuda_status);
    status_ = GPUXTB_STATUS_BACKEND_UNAVAILABLE;
    return;
  }
  if (previous_device_ == selected_device_) {
    status_ = GPUXTB_STATUS_SUCCESS;
    return;
  }

  cuda_status = cudaSetDevice(selected_device_);
  if (cuda_status != cudaSuccess) {
    consume_validation_error(cuda_status);
    error = cuda_error_message("cudaSetDevice", cuda_status);
    status_ = cuda_status == cudaErrorInvalidDevice ? GPUXTB_STATUS_INVALID_ARGUMENT
                                                    : GPUXTB_STATUS_BACKEND_UNAVAILABLE;
    return;
  }
  restore_pending_ = true;
  status_ = GPUXTB_STATUS_SUCCESS;
}

ScopedCudaDevice::~ScopedCudaDevice() {
  if (restore_pending_) {
    const cudaError_t cuda_status = cudaSetDevice(previous_device_);
    consume_validation_error(cuda_status);
  }
}

gpuxtb_status_t ScopedCudaDevice::restore(std::string& error) {
  if (!restore_pending_) return GPUXTB_STATUS_SUCCESS;
  const cudaError_t cuda_status = cudaSetDevice(previous_device_);
  if (cuda_status != cudaSuccess) {
    consume_validation_error(cuda_status);
    error = cuda_error_message("cudaSetDevice while restoring the caller device", cuda_status);
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  restore_pending_ = false;
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_cuda_stream_owner(std::int32_t device_id, cudaStream_t stream,
                                           bool reject_capture, std::string& error) {
  error.clear();
  ScopedCudaDevice device(device_id, error);
  if (!device.ok()) return device.status();

  if (reject_capture) {
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const cudaError_t capture_query_status = cudaStreamIsCapturing(stream, &capture_status);
    if (capture_query_status != cudaSuccess) {
      consume_validation_error(capture_query_status);
      error = cuda_error_message("cudaStreamIsCapturing", capture_query_status);
      return finish_with_restore(device, GPUXTB_STATUS_INVALID_ARGUMENT, error);
    }
    if (capture_status != cudaStreamCaptureStatusNone) {
      error = "CUDA stream capture is incompatible with this synchronous execution boundary";
      return finish_with_restore(device, GPUXTB_STATUS_NOT_SUPPORTED, error);
    }
  }

  int stream_device = -1;
  const cudaError_t cuda_status = cudaStreamGetDevice(stream, &stream_device);
  if (cuda_status != cudaSuccess) {
    consume_validation_error(cuda_status);
    error = cuda_error_message("cudaStreamGetDevice", cuda_status);
    const gpuxtb_status_t status =
        cuda_status == cudaErrorInvalidValue || cuda_status == cudaErrorInvalidResourceHandle
            ? GPUXTB_STATUS_INVALID_ARGUMENT
            : GPUXTB_STATUS_BACKEND_UNAVAILABLE;
    return finish_with_restore(device, status, error);
  }
  if (stream_device != device_id) {
    error = "CUDA stream belongs to device " + std::to_string(stream_device) +
            ", not context device " + std::to_string(device_id);
    return finish_with_restore(device, GPUXTB_STATUS_INVALID_ARGUMENT, error);
  }

  return finish_with_restore(device, GPUXTB_STATUS_SUCCESS, error);
}

gpuxtb_status_t validate_cuda_const_buffer(std::int32_t device_id, const char* name,
                                           const gpuxtb_const_buffer_t& buffer,
                                           std::size_t logical_bytes, std::size_t alignment,
                                           CudaManagedMemoryPolicy managed_policy,
                                           CudaValidatedConstBuffer& validated,
                                           std::string& error) {
  validated = {};
  error.clear();
  gpuxtb_status_t status =
      validate_basic_descriptor(name, buffer.data, buffer.size_bytes, buffer.memory_space,
                                buffer.reserved, logical_bytes, alignment, error);
  if (status != GPUXTB_STATUS_SUCCESS || logical_bytes == 0u) return status;

  ScopedCudaDevice device(device_id, error);
  if (!device.ok()) return device.status();
  PointerFacts facts{};
  status = validate_pointer_facts(device_id, name, buffer.data, buffer.memory_space, managed_policy,
                                  false, facts, error);
  if (status == GPUXTB_STATUS_SUCCESS && buffer.memory_space == GPUXTB_MEMORY_CUDA_DEVICE) {
    status = validate_device_allocation_range(name, buffer.data, buffer.size_bytes, error);
  }
  status = finish_with_restore(device, status, error);
  if (status != GPUXTB_STATUS_SUCCESS) return status;

  validated = {buffer.data, logical_bytes, buffer.memory_space, facts.type,
               facts.allocation_device};
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_cuda_buffer(std::int32_t device_id, const char* name,
                                     const gpuxtb_buffer_t& buffer, std::size_t logical_bytes,
                                     std::size_t alignment, CudaManagedMemoryPolicy managed_policy,
                                     CudaValidatedBuffer& validated, std::string& error) {
  validated = {};
  error.clear();
  gpuxtb_status_t status =
      validate_basic_descriptor(name, buffer.data, buffer.size_bytes, buffer.memory_space,
                                buffer.reserved, logical_bytes, alignment, error);
  if (status != GPUXTB_STATUS_SUCCESS || logical_bytes == 0u) return status;

  ScopedCudaDevice device(device_id, error);
  if (!device.ok()) return device.status();
  PointerFacts facts{};
  status = validate_pointer_facts(device_id, name, buffer.data, buffer.memory_space, managed_policy,
                                  true, facts, error);
  if (status == GPUXTB_STATUS_SUCCESS && buffer.memory_space == GPUXTB_MEMORY_CUDA_DEVICE) {
    status = validate_device_allocation_range(name, buffer.data, buffer.size_bytes, error);
  }
  status = finish_with_restore(device, status, error);
  if (status != GPUXTB_STATUS_SUCCESS) return status;

  validated = {buffer.data, logical_bytes, buffer.memory_space, facts.type,
               facts.allocation_device};
  return GPUXTB_STATUS_SUCCESS;
}

}  // namespace gpuxtb::detail
