#include "runtime/backend.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <memory>
#include <new>
#include <utility>

#include "runtime/gfn2_cpu_execution.hpp"
#if defined(XTBLOOM_HAS_CUDA)
#include "runtime/gfn2_cuda_execution.hpp"
#endif

namespace xtbloom::detail {
namespace {

bool resolve_cuda(std::int32_t requested_device, std::int32_t& resolved_device,
                  std::string& error) {
#if defined(XTBLOOM_HAS_CUDA)
  return resolve_cuda_device(requested_device, resolved_device, error);
#else
  (void)requested_device;
  resolved_device = -1;
  error = "the xtbloom library was built without CUDA support";
  return false;
#endif
}

}  // namespace

xtbloom_status_t create_context(const xtbloom_context_options_t& options, Context*& context,
                                std::string& error) {
  context = nullptr;

  if (options.device_id < -1) {
    error = "device_id must be -1 (automatic) or a non-negative device index";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (options.cpu_threads < 0) {
    error = "cpu_threads must be zero (automatic) or positive";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  xtbloom_backend_t selected = options.backend;
  std::int32_t resolved_device = -1;
  if (selected == XTBLOOM_BACKEND_AUTO) {
    std::string cuda_error;
    if (resolve_cuda(options.device_id, resolved_device, cuda_error)) {
      selected = XTBLOOM_BACKEND_CUDA;
    } else if (options.device_id >= 0 || options.stream != nullptr) {
      error = std::move(cuda_error);
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    } else {
      selected = XTBLOOM_BACKEND_CPU;
    }
  }

  if (selected == XTBLOOM_BACKEND_CUDA &&
      !resolve_cuda(options.device_id, resolved_device, error)) {
    return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
  }
#if defined(XTBLOOM_HAS_CUDA)
  if (selected == XTBLOOM_BACKEND_CUDA && !ensure_cuda_gfn2_parameters(resolved_device, error)) {
    return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
  }
#endif
  if (selected == XTBLOOM_BACKEND_ROCM) {
    error = "the ROCm backend is reserved but not implemented";
    return XTBLOOM_STATUS_NOT_SUPPORTED;
  }
  if (selected != XTBLOOM_BACKEND_CPU && selected != XTBLOOM_BACKEND_CUDA) {
    error = "unknown backend value";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (selected == XTBLOOM_BACKEND_CPU && options.stream != nullptr) {
    error = "a native GPU stream cannot be attached to the CPU backend";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  Context* created = new (std::nothrow) Context{};
  if (created == nullptr) {
    error = "failed to allocate a xtbloom context";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }

  created->backend = selected;
  created->device_id = resolved_device;
  created->cpu_threads = options.cpu_threads;
  created->stream = options.stream;
  if (selected == XTBLOOM_BACKEND_CPU) {
    try {
      /* Eager construction removes the first-compute shared_ptr data race. */
      created->gfn2_cpu_execution_cache =
          std::make_shared<Gfn2CpuExecutionCache>(options.cpu_threads);
    } catch (const std::bad_alloc&) {
      delete created;
      error = "failed to allocate CPU GFN2 execution cache";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
  }
#if defined(XTBLOOM_HAS_CUDA)
  if (selected == XTBLOOM_BACKEND_CUDA) {
    try {
      /* The public CUDA transaction owns one context-scoped fixed-topology
       * runtime so repeated calls can reuse stable plans and device arenas. */
      created->gfn2_cuda_execution_cache =
          std::make_shared<Gfn2CudaExecutionCache>(resolved_device, options.stream);
    } catch (const std::bad_alloc&) {
      delete created;
      error = "failed to allocate CUDA GFN2 execution cache";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
  }
#endif
  context = created;
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail
