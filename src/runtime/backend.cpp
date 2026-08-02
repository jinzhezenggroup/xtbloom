#include "runtime/backend.hpp"

#include <new>
#include <utility>

namespace gpuxtb::detail {
namespace {

bool resolve_cuda(std::int32_t requested_device, std::int32_t& resolved_device,
                  std::string& error) {
#if defined(GPUXTB_HAS_CUDA)
  return resolve_cuda_device(requested_device, resolved_device, error);
#else
  (void)requested_device;
  resolved_device = -1;
  error = "the gpuxtb library was built without CUDA support";
  return false;
#endif
}

}  // namespace

gpuxtb_status_t create_context(const gpuxtb_context_options_t& options, Context*& context,
                               std::string& error) {
  context = nullptr;

  if (options.device_id < -1) {
    error = "device_id must be -1 (automatic) or a non-negative device index";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (options.cpu_threads < 0) {
    error = "cpu_threads must be zero (automatic) or positive";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  gpuxtb_backend_t selected = options.backend;
  std::int32_t resolved_device = -1;
  if (selected == GPUXTB_BACKEND_AUTO) {
    std::string cuda_error;
    if (resolve_cuda(options.device_id, resolved_device, cuda_error)) {
      selected = GPUXTB_BACKEND_CUDA;
    } else if (options.device_id >= 0 || options.stream != nullptr) {
      error = std::move(cuda_error);
      return GPUXTB_STATUS_BACKEND_UNAVAILABLE;
    } else {
      selected = GPUXTB_BACKEND_CPU;
    }
  }

  if (selected == GPUXTB_BACKEND_CUDA && !resolve_cuda(options.device_id, resolved_device, error)) {
    return GPUXTB_STATUS_BACKEND_UNAVAILABLE;
  }
#if defined(GPUXTB_HAS_CUDA)
  if (selected == GPUXTB_BACKEND_CUDA && !ensure_cuda_gfn2_parameters(resolved_device, error)) {
    return GPUXTB_STATUS_BACKEND_UNAVAILABLE;
  }
#endif
  if (selected == GPUXTB_BACKEND_ROCM) {
    error = "the ROCm backend is reserved but not implemented";
    return GPUXTB_STATUS_NOT_SUPPORTED;
  }
  if (selected != GPUXTB_BACKEND_CPU && selected != GPUXTB_BACKEND_CUDA) {
    error = "unknown backend value";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (selected == GPUXTB_BACKEND_CPU && options.stream != nullptr) {
    error = "a native GPU stream cannot be attached to the CPU backend";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  Context* created = new (std::nothrow) Context{};
  if (created == nullptr) {
    error = "failed to allocate a gpuxtb context";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  }

  created->backend = selected;
  created->device_id = resolved_device;
  created->cpu_threads = options.cpu_threads;
  created->stream = options.stream;
  context = created;
  return GPUXTB_STATUS_SUCCESS;
}

}  // namespace gpuxtb::detail
