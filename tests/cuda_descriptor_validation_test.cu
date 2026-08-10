#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

#include "runtime/cuda_descriptor_validation.hpp"

namespace {

using xtbloom::detail::CudaManagedMemoryPolicy;
using xtbloom::detail::CudaValidatedBuffer;
using xtbloom::detail::CudaValidatedConstBuffer;
using xtbloom::detail::ScopedCudaDevice;
using xtbloom::detail::validate_cuda_buffer;
using xtbloom::detail::validate_cuda_const_buffer;
using xtbloom::detail::validate_cuda_stream_owner;

#define CHECK(condition)                                                                       \
  do {                                                                                         \
    if (!(condition)) {                                                                        \
      std::fprintf(stderr, "CUDA descriptor validation check failed at %s:%d: %s\n", __FILE__, \
                   __LINE__, #condition);                                                      \
      return __LINE__;                                                                         \
    }                                                                                          \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

int current_device() {
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) return -1;
  return device;
}

xtbloom_const_buffer_t const_view(const void* data, std::size_t bytes,
                                  xtbloom_memory_space_t memory_space) {
  return {data, bytes, memory_space, 0u};
}

xtbloom_buffer_t mutable_view(void* data, std::size_t bytes, xtbloom_memory_space_t memory_space) {
  return {data, bytes, memory_space, 0u};
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status == cudaErrorNoDevice || count_status == cudaErrorInsufficientDriver) {
    std::fprintf(stderr, "CUDA descriptor validation test skipped: %s\n",
                 cudaGetErrorString(count_status));
    (void)cudaGetLastError();
    return 0;
  }
  CUDA_CHECK(count_status);
  CHECK(device_count > 0);

  const int context_device = current_device();
  CHECK(context_device >= 0);
  std::string error;
  CudaValidatedConstBuffer const_validated{};
  CudaValidatedBuffer mutable_validated{};

  alignas(double) double ordinary_host[4]{1.0, 2.0, 3.0, 4.0};
  auto host_input = const_view(ordinary_host, sizeof(ordinary_host), XTBLOOM_MEMORY_HOST);
  auto host_output = mutable_view(ordinary_host, sizeof(ordinary_host), XTBLOOM_MEMORY_HOST);
  CHECK(validate_cuda_const_buffer(
            context_device, "ordinary_host", host_input, sizeof(ordinary_host), alignof(double),
            CudaManagedMemoryPolicy::kReject, const_validated, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(error.empty());
  CHECK(const_validated.data == ordinary_host);
  CHECK(const_validated.logical_bytes == sizeof(ordinary_host));
  CHECK(const_validated.pointer_type == cudaMemoryTypeUnregistered);
  CHECK(current_device() == context_device);
  CHECK(validate_cuda_buffer(context_device, "ordinary_host_output", host_output,
                             sizeof(ordinary_host), alignof(double),
                             CudaManagedMemoryPolicy::kReject, mutable_validated,
                             error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(mutable_validated.data == ordinary_host);
  CHECK(current_device() == context_device);

  void* pinned_host = nullptr;
  CUDA_CHECK(cudaMallocHost(&pinned_host, 128u));
  auto registered_input = const_view(pinned_host, 128u, XTBLOOM_MEMORY_HOST);
  auto registered_output = mutable_view(pinned_host, 128u, XTBLOOM_MEMORY_HOST);
  CHECK(validate_cuda_const_buffer(context_device, "registered_host", registered_input, 128u,
                                   alignof(double), CudaManagedMemoryPolicy::kReject,
                                   const_validated, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(const_validated.pointer_type == cudaMemoryTypeHost);
  CHECK(validate_cuda_buffer(context_device, "registered_host_output", registered_output, 128u,
                             alignof(double), CudaManagedMemoryPolicy::kReject, mutable_validated,
                             error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(mutable_validated.pointer_type == cudaMemoryTypeHost);
  auto registered_as_device = const_view(pinned_host, 128u, XTBLOOM_MEMORY_CUDA_DEVICE);
  CHECK(validate_cuda_const_buffer(context_device, "registered_as_device", registered_as_device,
                                   128u, alignof(double), CudaManagedMemoryPolicy::kReject,
                                   const_validated, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(current_device() == context_device);

  double* device_data = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_data), 4u * sizeof(double)));
  auto device_input = const_view(device_data, 4u * sizeof(double), XTBLOOM_MEMORY_CUDA_DEVICE);
  auto device_output = mutable_view(device_data, 4u * sizeof(double), XTBLOOM_MEMORY_CUDA_DEVICE);
  CHECK(validate_cuda_const_buffer(
            context_device, "device_input", device_input, 4u * sizeof(double), alignof(double),
            CudaManagedMemoryPolicy::kReject, const_validated, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(const_validated.pointer_type == cudaMemoryTypeDevice);
  CHECK(const_validated.allocation_device == context_device);
  CHECK(validate_cuda_buffer(context_device, "device_output", device_output, 4u * sizeof(double),
                             alignof(double), CudaManagedMemoryPolicy::kReject, mutable_validated,
                             error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(mutable_validated.pointer_type == cudaMemoryTypeDevice);
  CHECK(current_device() == context_device);

  /* CUDA descriptors may start inside an allocation, but their full declared
   * capacity must remain inside that allocation even when the logical prefix
   * needed by this call would fit. */
  auto interior_device =
      const_view(device_data + 1u, 3u * sizeof(double), XTBLOOM_MEMORY_CUDA_DEVICE);
  CHECK(validate_cuda_const_buffer(context_device, "interior_device", interior_device,
                                   2u * sizeof(double), alignof(double),
                                   CudaManagedMemoryPolicy::kReject, const_validated,
                                   error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(const_validated.data == device_data + 1u);
  auto interior_device_overrun =
      const_view(device_data + 1u, 4u * sizeof(double), XTBLOOM_MEMORY_CUDA_DEVICE);
  CHECK(validate_cuda_const_buffer(context_device, "interior_device_overrun",
                                   interior_device_overrun, sizeof(double), alignof(double),
                                   CudaManagedMemoryPolicy::kReject, const_validated,
                                   error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(error.find("extends past") != std::string::npos);
  CHECK(const_validated.data == nullptr);
  auto device_output_overrun =
      mutable_view(device_data, 4u * sizeof(double) + 1u, XTBLOOM_MEMORY_CUDA_DEVICE);
  CHECK(validate_cuda_buffer(context_device, "device_output_overrun", device_output_overrun,
                             sizeof(double), alignof(double), CudaManagedMemoryPolicy::kReject,
                             mutable_validated, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(error.find("extends past") != std::string::npos);
  CHECK(mutable_validated.data == nullptr);

  const std::uintptr_t device_address = reinterpret_cast<std::uintptr_t>(device_data);
  const std::size_t overflowing_device_bytes =
      std::numeric_limits<std::uintptr_t>::max() - device_address + 1u;
  auto overflowing_device =
      const_view(device_data, overflowing_device_bytes, XTBLOOM_MEMORY_CUDA_DEVICE);
  CHECK(validate_cuda_const_buffer(context_device, "overflowing_device", overflowing_device,
                                   sizeof(double), alignof(double),
                                   CudaManagedMemoryPolicy::kReject, const_validated,
                                   error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(error.find("overflows uintptr_t") != std::string::npos);
  CHECK(cudaPeekAtLastError() == cudaSuccess);
  CHECK(current_device() == context_device);

  /* A truthful CUDA allocation and an ordinary host pointer must both fail
   * closed when their public memory-space tags are swapped. */
  auto device_as_host = const_view(device_data, 4u * sizeof(double), XTBLOOM_MEMORY_HOST);
  CHECK(validate_cuda_const_buffer(context_device, "device_as_host", device_as_host,
                                   4u * sizeof(double), alignof(double),
                                   CudaManagedMemoryPolicy::kReject, const_validated,
                                   error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(!error.empty());
  CHECK(const_validated.data == nullptr);
  CHECK(current_device() == context_device);
  auto host_as_device =
      const_view(ordinary_host, sizeof(ordinary_host), XTBLOOM_MEMORY_CUDA_DEVICE);
  CHECK(validate_cuda_const_buffer(context_device, "host_as_device", host_as_device,
                                   sizeof(ordinary_host), alignof(double),
                                   CudaManagedMemoryPolicy::kReject, const_validated,
                                   error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(cudaPeekAtLastError() == cudaSuccess);
  CHECK(current_device() == context_device);

  double* managed_data = nullptr;
  CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void**>(&managed_data), 4u * sizeof(double)));
  auto managed_input = const_view(managed_data, 4u * sizeof(double), XTBLOOM_MEMORY_CUDA_DEVICE);
  CHECK(validate_cuda_const_buffer(context_device, "managed_rejected", managed_input,
                                   4u * sizeof(double), alignof(double),
                                   CudaManagedMemoryPolicy::kReject, const_validated,
                                   error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(validate_cuda_const_buffer(context_device, "managed_allowed", managed_input,
                                   4u * sizeof(double), alignof(double),
                                   CudaManagedMemoryPolicy::kAllowOnAllocationDevice,
                                   const_validated, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(const_validated.pointer_type == cudaMemoryTypeManaged);
  CHECK(const_validated.allocation_device == context_device);
  auto managed_output = mutable_view(managed_data, 4u * sizeof(double), XTBLOOM_MEMORY_CUDA_DEVICE);
  CHECK(validate_cuda_buffer(context_device, "managed_output", managed_output, 4u * sizeof(double),
                             alignof(double), CudaManagedMemoryPolicy::kAllowOnAllocationDevice,
                             mutable_validated, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(mutable_validated.pointer_type == cudaMemoryTypeManaged);
  auto interior_managed =
      const_view(managed_data + 2u, 2u * sizeof(double), XTBLOOM_MEMORY_CUDA_DEVICE);
  CHECK(validate_cuda_const_buffer(context_device, "interior_managed", interior_managed,
                                   sizeof(double), alignof(double),
                                   CudaManagedMemoryPolicy::kAllowOnAllocationDevice,
                                   const_validated, error) == XTBLOOM_STATUS_SUCCESS);
  auto interior_managed_overrun =
      const_view(managed_data + 2u, 3u * sizeof(double), XTBLOOM_MEMORY_CUDA_DEVICE);
  CHECK(validate_cuda_const_buffer(context_device, "interior_managed_overrun",
                                   interior_managed_overrun, sizeof(double), alignof(double),
                                   CudaManagedMemoryPolicy::kAllowOnAllocationDevice,
                                   const_validated, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(error.find("extends past") != std::string::npos);
  CHECK(const_validated.data == nullptr);
  auto managed_as_host = const_view(managed_data, 4u * sizeof(double), XTBLOOM_MEMORY_HOST);
  CHECK(validate_cuda_const_buffer(context_device, "managed_as_host", managed_as_host,
                                   4u * sizeof(double), alignof(double),
                                   CudaManagedMemoryPolicy::kAllowOnAllocationDevice,
                                   const_validated, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(current_device() == context_device);

  alignas(double) std::byte misaligned_storage[2u * sizeof(double)]{};
  auto misaligned_host = const_view(misaligned_storage + 1u, sizeof(double), XTBLOOM_MEMORY_HOST);
  CHECK(validate_cuda_const_buffer(context_device, "misaligned_host", misaligned_host,
                                   sizeof(double), alignof(double),
                                   CudaManagedMemoryPolicy::kReject, const_validated,
                                   error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  auto misaligned_device = const_view(reinterpret_cast<const std::byte*>(device_data) + 1u,
                                      sizeof(double), XTBLOOM_MEMORY_CUDA_DEVICE);
  CHECK(validate_cuda_const_buffer(context_device, "misaligned_device", misaligned_device,
                                   sizeof(double), alignof(double),
                                   CudaManagedMemoryPolicy::kReject, const_validated,
                                   error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  auto short_host = const_view(ordinary_host, sizeof(double) - 1u, XTBLOOM_MEMORY_HOST);
  CHECK(validate_cuda_const_buffer(context_device, "short_host", short_host, sizeof(double),
                                   alignof(double), CudaManagedMemoryPolicy::kReject,
                                   const_validated, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(validate_cuda_const_buffer(context_device, "bad_alignment", host_input, sizeof(double), 3u,
                                   CudaManagedMemoryPolicy::kReject, const_validated,
                                   error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  const auto* overflowing_pointer = reinterpret_cast<const void*>(
      std::numeric_limits<std::uintptr_t>::max() - static_cast<std::uintptr_t>(3u));
  auto overflowing_host = const_view(overflowing_pointer, 8u, XTBLOOM_MEMORY_HOST);
  CHECK(validate_cuda_const_buffer(context_device, "overflowing_host", overflowing_host, 8u, 1u,
                                   CudaManagedMemoryPolicy::kReject, const_validated,
                                   error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(current_device() == context_device);

  auto null_nonempty = const_view(nullptr, sizeof(double), XTBLOOM_MEMORY_HOST);
  CHECK(validate_cuda_const_buffer(context_device, "null_nonempty", null_nonempty, sizeof(double),
                                   alignof(double), CudaManagedMemoryPolicy::kReject,
                                   const_validated, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  auto empty = const_view(nullptr, 0u, XTBLOOM_MEMORY_HOST);
  CHECK(validate_cuda_const_buffer(context_device, "empty", empty, 0u, alignof(double),
                                   CudaManagedMemoryPolicy::kReject, const_validated,
                                   error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(const_validated.data == nullptr);
  CHECK(error.empty());
  auto reserved = host_input;
  reserved.reserved = 1u;
  CHECK(validate_cuda_const_buffer(context_device, "reserved", reserved, sizeof(double),
                                   alignof(double), CudaManagedMemoryPolicy::kReject,
                                   const_validated, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  auto rocm = host_input;
  rocm.memory_space = XTBLOOM_MEMORY_ROCM_DEVICE;
  CHECK(validate_cuda_const_buffer(context_device, "rocm", rocm, sizeof(double), alignof(double),
                                   CudaManagedMemoryPolicy::kReject, const_validated,
                                   error) == XTBLOOM_STATUS_NOT_SUPPORTED);
  auto unknown_space = host_input;
  unknown_space.memory_space = 97;
  CHECK(validate_cuda_const_buffer(context_device, "unknown_space", unknown_space, sizeof(double),
                                   alignof(double), CudaManagedMemoryPolicy::kReject,
                                   const_validated, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(current_device() == context_device);

  /* A stale device address forces cudaPointerGetAttributes down its failure
   * path. The validator must consume that diagnostic before returning. */
  double* stale_device = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&stale_device), sizeof(double)));
  CUDA_CHECK(cudaFree(stale_device));
  auto stale = const_view(stale_device, sizeof(double), XTBLOOM_MEMORY_CUDA_DEVICE);
  CHECK(validate_cuda_const_buffer(context_device, "stale_device", stale, sizeof(double),
                                   alignof(double), CudaManagedMemoryPolicy::kReject,
                                   const_validated, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(cudaPeekAtLastError() == cudaSuccess);
  CHECK(current_device() == context_device);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CHECK(validate_cuda_stream_owner(context_device, nullptr, true, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(validate_cuda_stream_owner(context_device, stream, true, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(error.empty());
  CHECK(current_device() == context_device);

  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CHECK(validate_cuda_stream_owner(context_device, stream, true, error) ==
        XTBLOOM_STATUS_NOT_SUPPORTED);
  CHECK(!error.empty());
  CHECK(current_device() == context_device);
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  if (graph != nullptr) CUDA_CHECK(cudaGraphDestroy(graph));

  /* Invalid device selection is rejected without changing the caller's
   * thread-local current device. */
  CHECK(validate_cuda_stream_owner(device_count, stream, true, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(cudaPeekAtLastError() == cudaSuccess);
  CHECK(current_device() == context_device);
  {
    ScopedCudaDevice guard(device_count, error);
    CHECK(!guard.ok());
    CHECK(guard.status() == XTBLOOM_STATUS_INVALID_ARGUMENT);
  }
  CHECK(current_device() == context_device);

  if (device_count >= 2) {
    const int other_device = context_device == 0 ? 1 : 0;
    cudaStream_t other_stream = nullptr;
    double* other_data = nullptr;
    double* other_managed = nullptr;
    CUDA_CHECK(cudaSetDevice(other_device));
    CUDA_CHECK(cudaStreamCreateWithFlags(&other_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&other_data), 4u * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void**>(&other_managed), 4u * sizeof(double)));

    /* Enter every validator with another current device. Success and each
     * ownership failure must restore it before returning. */
    CHECK(validate_cuda_const_buffer(context_device, "target_from_other_current", device_input,
                                     4u * sizeof(double), alignof(double),
                                     CudaManagedMemoryPolicy::kReject, const_validated,
                                     error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(current_device() == other_device);
    auto wrong_device = const_view(other_data, 4u * sizeof(double), XTBLOOM_MEMORY_CUDA_DEVICE);
    CHECK(validate_cuda_const_buffer(context_device, "wrong_device", wrong_device,
                                     4u * sizeof(double), alignof(double),
                                     CudaManagedMemoryPolicy::kReject, const_validated,
                                     error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(current_device() == other_device);
    auto wrong_managed = const_view(other_managed, 4u * sizeof(double), XTBLOOM_MEMORY_CUDA_DEVICE);
    CHECK(validate_cuda_const_buffer(context_device, "wrong_managed", wrong_managed,
                                     4u * sizeof(double), alignof(double),
                                     CudaManagedMemoryPolicy::kAllowOnAllocationDevice,
                                     const_validated, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(current_device() == other_device);
    CHECK(validate_cuda_stream_owner(context_device, other_stream, true, error) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(current_device() == other_device);
    CHECK(validate_cuda_stream_owner(context_device, nullptr, true, error) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(current_device() == other_device);

    {
      ScopedCudaDevice guard(context_device, error);
      CHECK(guard.ok());
      CHECK(current_device() == context_device);
      CHECK(guard.restore(error) == XTBLOOM_STATUS_SUCCESS);
      CHECK(current_device() == other_device);
    }
    {
      ScopedCudaDevice guard(context_device, error);
      CHECK(guard.ok());
      CHECK(current_device() == context_device);
    }
    CHECK(current_device() == other_device);

    CUDA_CHECK(cudaFree(other_managed));
    CUDA_CHECK(cudaFree(other_data));
    CUDA_CHECK(cudaStreamDestroy(other_stream));
    CUDA_CHECK(cudaSetDevice(context_device));
  }

  /* Exercise registered read-only output rejection when the visible device
   * and host OS permit this optional CUDA registration mode. */
  int readonly_supported = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&readonly_supported, cudaDevAttrHostRegisterReadOnlySupported,
                                    context_device));
  if (readonly_supported != 0) {
    constexpr std::size_t kPageBytes = 4096u;
    void* readonly_host = std::aligned_alloc(kPageBytes, kPageBytes);
    CHECK(readonly_host != nullptr);
    const cudaError_t register_status =
        cudaHostRegister(readonly_host, kPageBytes, cudaHostRegisterReadOnly);
    if (register_status == cudaSuccess) {
      auto readonly_input = const_view(readonly_host, kPageBytes, XTBLOOM_MEMORY_HOST);
      auto readonly_output = mutable_view(readonly_host, kPageBytes, XTBLOOM_MEMORY_HOST);
      CHECK(validate_cuda_const_buffer(context_device, "readonly_input", readonly_input,
                                       sizeof(double), alignof(double),
                                       CudaManagedMemoryPolicy::kReject, const_validated,
                                       error) == XTBLOOM_STATUS_SUCCESS);
      CHECK(validate_cuda_buffer(context_device, "readonly_output", readonly_output, sizeof(double),
                                 alignof(double), CudaManagedMemoryPolicy::kReject,
                                 mutable_validated, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
      CHECK(cudaPeekAtLastError() == cudaSuccess);
      CUDA_CHECK(cudaHostUnregister(readonly_host));
    } else {
      (void)cudaGetLastError();
    }
    std::free(readonly_host);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(managed_data));
  CUDA_CHECK(cudaFree(device_data));
  CUDA_CHECK(cudaFreeHost(pinned_host));
  CHECK(current_device() == context_device);
  return 0;
}
