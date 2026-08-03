#ifndef GPUXTB_RUNTIME_CUDA_DESCRIPTOR_VALIDATION_HPP
#define GPUXTB_RUNTIME_CUDA_DESCRIPTOR_VALIDATION_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "gpuxtb/gpuxtb.h"

namespace gpuxtb::detail {

/*
 * Managed storage is rejected by default at the public CUDA boundary.  An
 * individual bridge may opt in only when its access and synchronization model
 * is valid for managed memory.  Even then, the allocation must belong to the
 * context device; peer-access capability never relaxes descriptor ownership.
 */
enum class CudaManagedMemoryPolicy : std::uint8_t {
  kReject = 0u,
  kAllowOnAllocationDevice = 1u,
};

/*
 * Result of validating a read-only public buffer.  This is a borrowed view:
 * validation neither takes ownership nor extends the caller allocation's
 * lifetime.  pointer_type and allocation_device are diagnostic facts captured
 * from CUDA before any caller storage is accessed.
 */
struct CudaValidatedConstBuffer {
  const void* data = nullptr;
  std::size_t logical_bytes = 0u;
  gpuxtb_memory_space_t memory_space = GPUXTB_MEMORY_HOST;
  cudaMemoryType pointer_type = cudaMemoryTypeUnregistered;
  std::int32_t allocation_device = -1;
};

/* Mutable counterpart used for result buffers and writable staging views. */
struct CudaValidatedBuffer {
  void* data = nullptr;
  std::size_t logical_bytes = 0u;
  gpuxtb_memory_space_t memory_space = GPUXTB_MEMORY_HOST;
  cudaMemoryType pointer_type = cudaMemoryTypeUnregistered;
  std::int32_t allocation_device = -1;
};

/*
 * RAII guard for CUDA's thread-local current-device state.
 *
 * Construction records the current device and selects device_id.  The
 * destructor makes a best-effort restoration, while restore() is the required
 * path at a status-returning boundary because it can report a restoration
 * failure.  Successful restore leaves the supplied error text unchanged so it
 * can safely be used while unwinding an earlier validation failure.
 */
class ScopedCudaDevice {
 public:
  explicit ScopedCudaDevice(std::int32_t device_id, std::string& error);
  ~ScopedCudaDevice();

  ScopedCudaDevice(const ScopedCudaDevice&) = delete;
  ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

  [[nodiscard]] bool ok() const noexcept { return status_ == GPUXTB_STATUS_SUCCESS; }
  [[nodiscard]] gpuxtb_status_t status() const noexcept { return status_; }
  [[nodiscard]] std::int32_t previous_device() const noexcept { return previous_device_; }
  [[nodiscard]] std::int32_t selected_device() const noexcept { return selected_device_; }

  [[nodiscard]] gpuxtb_status_t restore(std::string& error);

 private:
  std::int32_t previous_device_ = -1;
  std::int32_t selected_device_ = -1;
  gpuxtb_status_t status_ = GPUXTB_STATUS_BACKEND_UNAVAILABLE;
  bool restore_pending_ = false;
};

/*
 * Validate that stream belongs to device_id.  The default, legacy, and
 * per-thread streams are interpreted after selecting device_id.  When
 * reject_capture is true, an actively captured stream is rejected before a
 * synchronous bridge can enqueue work or touch caller output.
 *
 * The calling thread's current device is preserved on every reported path.
 */
[[nodiscard]] gpuxtb_status_t validate_cuda_stream_owner(std::int32_t device_id,
                                                         cudaStream_t stream, bool reject_capture,
                                                         std::string& error);

/*
 * Validate a complete logical byte range without dereferencing it.
 *
 * Both functions check descriptor metadata, capacity, declared-range address
 * overflow, natural alignment, memory-space truthfulness, CUDA pointer type,
 * allocation device, and the explicit managed-memory policy.  HOST accepts
 * ordinary and CUDA-registered host storage but rejects device/managed
 * pointers mislabeled as host.  CUDA_DEVICE additionally requires the entire
 * declared size_bytes range, including one beginning at an interior pointer,
 * to fit in the allocation reported by cuMemGetAddressRange.  It accepts
 * only a device allocation on device_id, or managed storage when explicitly
 * enabled on that same allocation device.
 *
 * validate_cuda_buffer additionally rejects CUDA host registrations marked
 * read-only, preserving the writable contract of public result descriptors.
 * On failure validated is reset, CUDA runtime validation errors are consumed
 * from the per-thread last-error slot, and no caller bytes are read or written.
 */
[[nodiscard]] gpuxtb_status_t validate_cuda_const_buffer(
    std::int32_t device_id, const char* name, const gpuxtb_const_buffer_t& buffer,
    std::size_t logical_bytes, std::size_t alignment, CudaManagedMemoryPolicy managed_policy,
    CudaValidatedConstBuffer& validated, std::string& error);

[[nodiscard]] gpuxtb_status_t validate_cuda_buffer(std::int32_t device_id, const char* name,
                                                   const gpuxtb_buffer_t& buffer,
                                                   std::size_t logical_bytes, std::size_t alignment,
                                                   CudaManagedMemoryPolicy managed_policy,
                                                   CudaValidatedBuffer& validated,
                                                   std::string& error);

}  // namespace gpuxtb::detail

#endif  // GPUXTB_RUNTIME_CUDA_DESCRIPTOR_VALIDATION_HPP
