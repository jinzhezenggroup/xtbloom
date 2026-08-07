#include <algorithm>
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>

#include "gpuxtb/gpuxtb.h"
#include "runtime/backend.hpp"
#include "runtime/gfn2_cpu_execution.hpp"
#include "runtime/gfn2_plan.hpp"
#if defined(GPUXTB_HAS_CUDA)
#include "runtime/gfn2_cuda_execution.hpp"
#endif
#include "runtime/result_owner.hpp"
#include "runtime/validation.hpp"

struct gpuxtb_context {
  gpuxtb::detail::Context* implementation;
};

struct gpuxtb_plan {
  gpuxtb::detail::Gfn2Plan* implementation;
};

struct gpuxtb_result_owner {
  gpuxtb::detail::ResultOwner* implementation;
};

namespace {

thread_local std::string last_error;

gpuxtb_status_t fail(gpuxtb_status_t status, std::string message) {
  last_error = std::move(message);
  return status;
}

template <typename T>
bool valid_header(const T* value, std::size_t minimum_size) {
  return value != nullptr && value->struct_size >= minimum_size &&
         value->api_version == GPUXTB_API_VERSION;
}

template <typename Enum>
std::uint32_t raw_enum(const Enum& value) {
  static_assert(sizeof(Enum) == sizeof(std::uint32_t));
  std::uint32_t raw = 0;
  std::memcpy(&raw, &value, sizeof(raw));
  return raw;
}

template <typename T>
gpuxtb_status_t initialize_structure(T* value, std::size_t caller_size, std::size_t minimum_size,
                                     const char* name) {
  if (value == nullptr) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, std::string(name) + " is NULL");
  }
  if (caller_size < minimum_size || caller_size > std::numeric_limits<std::uint32_t>::max()) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT,
                std::string(name) + " size is smaller than ABI v1 or exceeds uint32_t");
  }

  /* Never write beyond the layout known by this library version. */
  std::memset(value, 0, std::min(caller_size, sizeof(T)));
  value->struct_size = static_cast<std::uint32_t>(caller_size);
  value->api_version = GPUXTB_API_VERSION;
  last_error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

const char* gpuxtb_version_string(void) { return "0.1.0"; }

const char* gpuxtb_status_string(gpuxtb_status_t status) {
  switch (status) {
    case GPUXTB_STATUS_SUCCESS:
      return "success";
    case GPUXTB_STATUS_INVALID_ARGUMENT:
      return "invalid argument";
    case GPUXTB_STATUS_BACKEND_UNAVAILABLE:
      return "backend unavailable";
    case GPUXTB_STATUS_NOT_SUPPORTED:
      return "not supported";
    case GPUXTB_STATUS_ALLOCATION_FAILED:
      return "allocation failed";
    case GPUXTB_STATUS_NOT_IMPLEMENTED:
      return "not implemented";
    case GPUXTB_STATUS_INTERNAL_ERROR:
      return "internal error";
    case GPUXTB_STATUS_SCC_NOT_CONVERGED:
      return "SCC not converged";
    case GPUXTB_STATUS_EIGENSOLVER_FAILED:
      return "eigensolver failed";
  }
  return "unknown status";
}

const char* gpuxtb_get_last_error(void) { return last_error.c_str(); }

gpuxtb_status_t gpuxtb_context_options_init(gpuxtb_context_options_t* options, size_t struct_size) {
  const gpuxtb_status_t status =
      initialize_structure(options, struct_size, GPUXTB_CONTEXT_OPTIONS_V1_SIZE, "context options");
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  options->backend = GPUXTB_BACKEND_AUTO;
  options->device_id = -1;
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t gpuxtb_batch_init(gpuxtb_batch_t* batch, size_t struct_size) {
  return initialize_structure(batch, struct_size, GPUXTB_BATCH_V1_SIZE, "batch");
}

gpuxtb_status_t gpuxtb_compute_options_init(gpuxtb_compute_options_t* options, size_t struct_size) {
  const gpuxtb_status_t status =
      initialize_structure(options, struct_size, GPUXTB_COMPUTE_OPTIONS_V1_SIZE, "compute options");
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  options->model = GPUXTB_MODEL_GFN2_XTB;
  options->flags = GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES;
  options->max_scc_iterations = 250;
  options->charge_tolerance = 1.0e-6;
  options->energy_tolerance = 1.0e-8;
  options->electronic_temperature = GPUXTB_DEFAULT_ELECTRONIC_TEMPERATURE;
  if (struct_size >= GPUXTB_COMPUTE_OPTIONS_V2_SIZE) {
    options->scc_start_mode = GPUXTB_SCC_START_FRESH;
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t gpuxtb_batch_result_init(gpuxtb_batch_result_t* result, size_t struct_size) {
  return initialize_structure(result, struct_size, GPUXTB_BATCH_RESULT_V1_SIZE, "batch result");
}

gpuxtb_status_t gpuxtb_workspace_query_init(gpuxtb_workspace_query_t* query, size_t struct_size) {
  return initialize_structure(query, struct_size, GPUXTB_WORKSPACE_QUERY_V1_SIZE,
                              "workspace query");
}

gpuxtb_status_t gpuxtb_context_create(const gpuxtb_context_options_t* options,
                                      gpuxtb_context_t** context) {
  if (context == nullptr) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "context output pointer is NULL");
  }
  *context = nullptr;
  if (!valid_header(options, GPUXTB_CONTEXT_OPTIONS_V1_SIZE)) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT,
                "context options are NULL, too small, or use an unsupported API version");
  }
  if (options->reserved != 0) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "context options reserved field must be zero");
  }
  const std::uint32_t backend = raw_enum(options->backend);
  if (backend > GPUXTB_BACKEND_ROCM) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "context options contain an unknown backend");
  }

  try {
    gpuxtb::detail::Context* implementation = nullptr;
    std::string error;
    const gpuxtb_status_t status = gpuxtb::detail::create_context(*options, implementation, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return fail(status, std::move(error));
    }

    gpuxtb_context_t* wrapper = new (std::nothrow) gpuxtb_context_t{implementation};
    if (wrapper == nullptr) {
      delete implementation;
      return fail(GPUXTB_STATUS_ALLOCATION_FAILED, "failed to allocate a context handle");
    }
    *context = wrapper;
    last_error.clear();
    return GPUXTB_STATUS_SUCCESS;
  } catch (const std::exception& exception) {
    return fail(GPUXTB_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(GPUXTB_STATUS_INTERNAL_ERROR, "unknown exception while creating context");
  }
}

void gpuxtb_context_destroy(gpuxtb_context_t* context) {
  if (context == nullptr) {
    return;
  }
  delete context->implementation;
  delete context;
}

gpuxtb_backend_t gpuxtb_context_get_backend(const gpuxtb_context_t* context) {
  if (context == nullptr || context->implementation == nullptr) {
    last_error = "context is NULL";
    return GPUXTB_BACKEND_AUTO;
  }
  last_error.clear();
  return context->implementation->backend;
}

int32_t gpuxtb_context_get_device_id(const gpuxtb_context_t* context) {
  if (context == nullptr || context->implementation == nullptr) {
    last_error = "context is NULL";
    return -1;
  }
  last_error.clear();
  return context->implementation->device_id;
}

gpuxtb_status_t gpuxtb_compute(gpuxtb_context_t* context, const gpuxtb_batch_t* batch,
                               const gpuxtb_compute_options_t* options,
                               gpuxtb_batch_result_t* result) {
  if (context == nullptr || context->implementation == nullptr) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "context is NULL");
  }
  try {
    const bool cuda_backend = context->implementation->backend == GPUXTB_BACKEND_CUDA;
    gpuxtb::detail::DescriptorValidationResult validation =
        cuda_backend ? gpuxtb::detail::validate_compute_descriptor_structure(
                           context->implementation->backend, batch, options, result)
                     : gpuxtb::detail::validate_compute_descriptors(
                           context->implementation->backend, batch, options, result);
    if (!validation.ok()) {
      return fail(validation.status, std::move(validation.error));
    }

    /* CUDA completes pointer-attribute and topology semantic validation under
     * the cache transaction before accessing caller storage. CPU retains the
     * historical complete host validation sequence here. */
    (void)validation.pending_offset_checks;
  } catch (const std::bad_alloc&) {
    return fail(GPUXTB_STATUS_ALLOCATION_FAILED,
                "failed to allocate temporary storage while validating a compute request");
  } catch (const std::exception& exception) {
    return fail(GPUXTB_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(GPUXTB_STATUS_INTERNAL_ERROR,
                "unknown exception while validating a compute request");
  }

  if (options->model == GPUXTB_MODEL_GFN1_XTB) {
    return fail(GPUXTB_STATUS_NOT_SUPPORTED,
                "GFN1-xTB is reserved by the ABI but is not implemented yet");
  }

  if (context->implementation->backend == GPUXTB_BACKEND_CPU) {
    try {
      const std::shared_ptr<gpuxtb::detail::Gfn2CpuExecutionCache>& cache =
          context->implementation->gfn2_cpu_execution_cache;
      if (cache == nullptr) {
        return fail(GPUXTB_STATUS_INTERNAL_ERROR,
                    "CPU context does not own a GFN2 execution cache");
      }
      std::string error;
      const gpuxtb_status_t status =
          gpuxtb::detail::execute_restricted_gfn2_cpu(*cache, *batch, *options, *result, error);
      if (status != GPUXTB_STATUS_SUCCESS) {
        return fail(status, std::move(error));
      }
      last_error.clear();
      return GPUXTB_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
      return fail(GPUXTB_STATUS_ALLOCATION_FAILED, "failed to allocate CPU GFN2 execution state");
    } catch (const std::exception& exception) {
      return fail(GPUXTB_STATUS_INTERNAL_ERROR, exception.what());
    } catch (...) {
      return fail(GPUXTB_STATUS_INTERNAL_ERROR,
                  "unknown exception while executing CPU GFN2 inference");
    }
  }

#if defined(GPUXTB_HAS_CUDA)
  try {
    const std::shared_ptr<gpuxtb::detail::Gfn2CudaExecutionCache>& cache =
        context->implementation->gfn2_cuda_execution_cache;
    if (cache == nullptr) {
      return fail(GPUXTB_STATUS_INTERNAL_ERROR, "CUDA context does not own a GFN2 execution cache");
    }
    std::string error;
    const gpuxtb_status_t status =
        gpuxtb::detail::execute_restricted_gfn2_cuda(*cache, *batch, *options, *result, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return fail(status, std::move(error));
    }
    last_error.clear();
    return GPUXTB_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return fail(GPUXTB_STATUS_ALLOCATION_FAILED, "failed to allocate CUDA GFN2 execution state");
  } catch (const std::exception& exception) {
    return fail(GPUXTB_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(GPUXTB_STATUS_INTERNAL_ERROR,
                "unknown exception while executing CUDA GFN2 inference");
  }
#else
  return fail(GPUXTB_STATUS_BACKEND_UNAVAILABLE,
              "the gpuxtb library was built without CUDA support");
#endif
}

gpuxtb_status_t gpuxtb_plan_create(gpuxtb_context_t* context, const gpuxtb_batch_t* batch,
                                   const gpuxtb_compute_options_t* options, gpuxtb_plan_t** plan) {
  if (plan == nullptr) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "plan output pointer is NULL");
  }
  *plan = nullptr;
  if (context == nullptr || context->implementation == nullptr) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "context is NULL");
  }
  if (batch == nullptr || options == nullptr) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "batch or compute options is NULL");
  }
  try {
    std::unique_ptr<gpuxtb::detail::Gfn2Plan> implementation(new (std::nothrow)
                                                                 gpuxtb::detail::Gfn2Plan{});
    if (implementation == nullptr) {
      return fail(GPUXTB_STATUS_ALLOCATION_FAILED, "failed to allocate a plan implementation");
    }
    std::string error;
    const gpuxtb_status_t status =
        implementation->create(*context->implementation, *batch, *options, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      implementation->destroy();
      return fail(status, std::move(error));
    }

    gpuxtb_plan_t* wrapper = new (std::nothrow) gpuxtb_plan_t{implementation.get()};
    if (wrapper == nullptr) {
      implementation->destroy();
      return fail(GPUXTB_STATUS_ALLOCATION_FAILED, "failed to allocate a plan handle");
    }
    implementation.release();
    *plan = wrapper;
    last_error.clear();
    return GPUXTB_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return fail(GPUXTB_STATUS_ALLOCATION_FAILED, "failed to allocate a plan");
  } catch (const std::exception& exception) {
    return fail(GPUXTB_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(GPUXTB_STATUS_INTERNAL_ERROR, "unknown exception while creating a plan");
  }
}

void gpuxtb_plan_destroy(gpuxtb_plan_t* plan) {
  if (plan == nullptr) {
    return;
  }
  if (plan->implementation != nullptr) {
    plan->implementation->destroy();
    delete plan->implementation;
  }
  delete plan;
}

gpuxtb_status_t gpuxtb_plan_query_workspace(const gpuxtb_plan_t* plan,
                                            gpuxtb_workspace_query_t* query) {
  if (plan == nullptr || plan->implementation == nullptr) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "plan is NULL");
  }
  if (!valid_header(query, GPUXTB_WORKSPACE_QUERY_V1_SIZE)) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT,
                "workspace query is NULL, too small, or uses an unsupported API version");
  }
  try {
    std::string error;
    const gpuxtb_status_t status = plan->implementation->query_workspace(
        static_cast<std::uint32_t>(query->compute_flags), *query, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return fail(status, std::move(error));
    }
    last_error.clear();
    return GPUXTB_STATUS_SUCCESS;
  } catch (const std::exception& exception) {
    return fail(GPUXTB_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(GPUXTB_STATUS_INTERNAL_ERROR, "unknown exception while querying plan workspace");
  }
}

gpuxtb_status_t gpuxtb_plan_compute(gpuxtb_plan_t* plan, const gpuxtb_batch_t* batch,
                                    const gpuxtb_compute_options_t* options,
                                    gpuxtb_batch_result_t* result) {
  if (plan == nullptr || plan->implementation == nullptr) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "plan is NULL");
  }
  if (batch == nullptr || options == nullptr || result == nullptr) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "batch, compute options, or batch result is NULL");
  }
  try {
    std::string error;
    const gpuxtb_status_t status = plan->implementation->compute(*batch, *options, *result, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return fail(status, std::move(error));
    }
    last_error.clear();
    return GPUXTB_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return fail(GPUXTB_STATUS_ALLOCATION_FAILED,
                "failed to allocate temporary storage while executing a plan compute request");
  } catch (const std::exception& exception) {
    return fail(GPUXTB_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(GPUXTB_STATUS_INTERNAL_ERROR,
                "unknown exception while executing a plan compute request");
  }
}

namespace {

/*
 * The exported managed tensor is one allocation: the managed-tensor struct
 * followed by the copied shape storage. manager_ctx holds the arena owner
 * (retained once per export). The deleter is a plain native function that
 * importing frameworks may call from any thread after the Python wrapper is
 * gone, so it never touches Python state.
 */
void* dlpack_export_block(std::size_t managed_size, std::size_t ndim) {
  const std::size_t shape_bytes = ndim * sizeof(std::int64_t);
  if (shape_bytes > std::numeric_limits<std::size_t>::max() - managed_size) {
    return nullptr;
  }
  return ::operator new(managed_size + shape_bytes, std::nothrow);
}

/*
 * Both deleters receive the managed tensor whose manager_ctx is the public
 * wrapper handle. Releasing the wrapper may be the final reference, in which
 * case the wrapper must be freed here because the Python producer has already
 * closed: importing frameworks can call this deleter from any thread and long
 * after the producer is gone.
 */
gpuxtb_result_owner_t* wrapper_from_manager_ctx(const void* manager_ctx) noexcept {
  return static_cast<gpuxtb_result_owner_t*>(const_cast<void*>(manager_ctx));
}

void finish_dlpack_deletion(gpuxtb_result_owner_t* wrapper, void* block) noexcept {
  gpuxtb::detail::ResultOwner* implementation = wrapper->implementation;
  const bool final = implementation->release();
  ::operator delete(block);
  if (final) {
    delete wrapper;
  }
}

void legacy_dlpack_deleter(gpuxtb::detail::DlpackManagedTensor* self) {
  if (self == nullptr) {
    return;
  }
  finish_dlpack_deletion(wrapper_from_manager_ctx(self->manager_ctx), self);
}

void versioned_dlpack_deleter(gpuxtb::detail::DlpackManagedTensorVersioned* self) {
  if (self == nullptr) {
    return;
  }
  finish_dlpack_deletion(wrapper_from_manager_ctx(self->manager_ctx), self);
}

gpuxtb_status_t populate_dlpack_view(gpuxtb_result_owner_t* wrapper,
                                     gpuxtb::detail::ResultOwner* owner,
                                     const gpuxtb_dlpack_view_t* view, bool versioned,
                                     void** out_managed) {
  if (view == nullptr || out_managed == nullptr) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "DLPack export view or output pointer is NULL");
  }
  if (!valid_header(view, GPUXTB_DLPACK_VIEW_V1_SIZE)) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT,
                "DLPack view is NULL, too small, or uses an unsupported API version");
  }
  if (view->reserved != 0) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "DLPack view reserved field must be zero");
  }
  if (view->ndim < 0 || view->ndim > GPUXTB_DLPACK_MAX_NDIM) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT,
                "DLPack view ndim must lie between 0 and GPUXTB_DLPACK_MAX_NDIM");
  }
  if (view->dtype_lanes != 1) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT,
                "DLPack view lanes must be 1 (scalar descriptors only)");
  }
  const std::size_t dtype_size =
      gpuxtb::detail::dlpack_dtype_size(view->dtype_code, view->dtype_bits);
  if (dtype_size == 0u) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT,
                "DLPack view uses an unsupported dtype code/bits combination");
  }
  if (view->ndim > 0 && view->shape == nullptr) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT,
                "DLPack view with ndim > 0 requires a shape pointer");
  }

  /* Validate extents before touching the arena or the output pointer. */
  std::uint64_t element_count = 1u;
  const std::int64_t* shape = view->shape;
  for (std::int32_t index = 0; index < view->ndim; ++index) {
    if (shape[index] < 0) {
      return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "DLPack view has a negative shape extent");
    }
    if (shape[index] == 0) {
      /* Once one extent is zero the tensor has no elements; keep validating
       * the remaining extents without dividing by a zero element count. */
      element_count = 0u;
      continue;
    }
    if (element_count == 0u) {
      continue;
    }
    if (static_cast<std::uint64_t>(shape[index]) >
        std::numeric_limits<std::uint64_t>::max() / element_count) {
      return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "DLPack view shape overflows element count");
    }
    element_count *= static_cast<std::uint64_t>(shape[index]);
  }
  std::uint64_t payload_bytes = 0u;
  if (element_count != 0u) {
    if (element_count > std::numeric_limits<std::uint64_t>::max() / dtype_size) {
      return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "DLPack view payload size overflows uint64_t");
    }
    payload_bytes = element_count * dtype_size;
  }
  const std::uint64_t arena_size = static_cast<std::uint64_t>(owner->size_bytes());
  if (view->byte_offset > arena_size) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "DLPack view starts past the arena end");
  }
  if (payload_bytes > arena_size - view->byte_offset) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "DLPack view payload extends past the arena end");
  }
  if (payload_bytes != 0u) {
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(owner->data());
    const std::uintptr_t address = base + static_cast<std::uintptr_t>(view->byte_offset);
    if (address % dtype_size != 0u) {
      return fail(GPUXTB_STATUS_INVALID_ARGUMENT,
                  "DLPack view offset does not preserve the dtype alignment");
    }
  }

  const std::size_t managed_size = versioned ? sizeof(gpuxtb::detail::DlpackManagedTensorVersioned)
                                             : sizeof(gpuxtb::detail::DlpackManagedTensor);
  void* block = dlpack_export_block(managed_size, static_cast<std::size_t>(view->ndim));
  if (block == nullptr) {
    return fail(GPUXTB_STATUS_ALLOCATION_FAILED,
                "failed to allocate a DLPack managed tensor export");
  }

  /* Transfer definitely: from here on the deleter owns the block and the
   * retained arena reference; a failure must not leave either behind. */
  std::int64_t* shape_storage =
      reinterpret_cast<std::int64_t*>(static_cast<unsigned char*>(block) + managed_size);
  void* data = payload_bytes == 0u ? nullptr
                                   : static_cast<unsigned char*>(owner->data()) + view->byte_offset;
  const std::int32_t device_id =
      owner->memory_space() == GPUXTB_MEMORY_CUDA_DEVICE ? owner->device_id() : 0;
  const std::int32_t device_type = gpuxtb::detail::dlpack_device_type(owner->memory_space());

  if (versioned) {
    gpuxtb::detail::DlpackManagedTensorVersioned* managed =
        static_cast<gpuxtb::detail::DlpackManagedTensorVersioned*>(block);
    managed->version_major = gpuxtb::detail::kDlpackVersionMajor;
    managed->version_minor = gpuxtb::detail::kDlpackVersionMinor;
    managed->manager_ctx = wrapper;
    managed->deleter = &versioned_dlpack_deleter;
    managed->flags = 0u;
    managed->dl_tensor.data = data;
    managed->dl_tensor.device.device_type = device_type;
    managed->dl_tensor.device.device_id = device_id;
    managed->dl_tensor.ndim = view->ndim;
    managed->dl_tensor.dtype.code = static_cast<std::uint8_t>(view->dtype_code);
    managed->dl_tensor.dtype.bits = static_cast<std::uint8_t>(view->dtype_bits);
    managed->dl_tensor.dtype.lanes = static_cast<std::uint16_t>(view->dtype_lanes);
    managed->dl_tensor.shape = shape_storage;
    managed->dl_tensor.strides = nullptr; /* compact row-major by convention */
    managed->dl_tensor.byte_offset = 0u;
    for (std::int32_t index = 0; index < view->ndim; ++index) {
      shape_storage[index] = shape[index];
    }
  } else {
    gpuxtb::detail::DlpackManagedTensor* managed =
        static_cast<gpuxtb::detail::DlpackManagedTensor*>(block);
    managed->manager_ctx = wrapper;
    managed->deleter = &legacy_dlpack_deleter;
    managed->dl_tensor.data = data;
    managed->dl_tensor.device.device_type = device_type;
    managed->dl_tensor.device.device_id = device_id;
    managed->dl_tensor.ndim = view->ndim;
    managed->dl_tensor.dtype.code = static_cast<std::uint8_t>(view->dtype_code);
    managed->dl_tensor.dtype.bits = static_cast<std::uint8_t>(view->dtype_bits);
    managed->dl_tensor.dtype.lanes = static_cast<std::uint16_t>(view->dtype_lanes);
    managed->dl_tensor.shape = shape_storage;
    managed->dl_tensor.strides = nullptr;
    managed->dl_tensor.byte_offset = 0u;
    for (std::int32_t index = 0; index < view->ndim; ++index) {
      shape_storage[index] = shape[index];
    }
  }

  owner->retain();
  *out_managed = block;
  return GPUXTB_STATUS_SUCCESS;
}

}  // namespace

gpuxtb_status_t gpuxtb_result_owner_options_init(gpuxtb_result_owner_options_t* options,
                                                 size_t struct_size) {
  const gpuxtb_status_t status = initialize_structure(
      options, struct_size, GPUXTB_RESULT_OWNER_OPTIONS_V1_SIZE, "result owner options");
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  options->memory_space = GPUXTB_MEMORY_HOST;
  options->device_id = -1;
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t gpuxtb_result_owner_create(const gpuxtb_result_owner_options_t* options,
                                           gpuxtb_result_owner_t** owner) {
  if (owner == nullptr) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "result owner output pointer is NULL");
  }
  *owner = nullptr;
  if (!valid_header(options, GPUXTB_RESULT_OWNER_OPTIONS_V1_SIZE)) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT,
                "result owner options are NULL, too small, or use an unsupported API version");
  }
  if (options->reserved != 0) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "result owner options reserved field must be zero");
  }
  if (options->memory_space != GPUXTB_MEMORY_HOST &&
      options->memory_space != GPUXTB_MEMORY_CUDA_DEVICE) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "result owner uses an unsupported memory space");
  }
  if ((options->memory_space == GPUXTB_MEMORY_HOST && options->device_id != -1) ||
      (options->memory_space == GPUXTB_MEMORY_CUDA_DEVICE && options->device_id < 0)) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT,
                "result owner device_id is inconsistent with its memory space");
  }
  if (options->size_bytes == 0u) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "result owner arena size must be nonzero");
  }
  if (options->size_bytes > std::numeric_limits<std::size_t>::max()) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "result owner arena size overflows size_t");
  }

  try {
    void* data = nullptr;
    std::string error;
    gpuxtb_status_t status = GPUXTB_STATUS_INVALID_ARGUMENT;
    if (options->memory_space == GPUXTB_MEMORY_HOST) {
      status = gpuxtb::detail::allocate_host_result_arena(
          static_cast<std::size_t>(options->size_bytes), &data, error);
    } else {
#if defined(GPUXTB_HAS_CUDA)
      status = gpuxtb::detail::allocate_cuda_result_arena(
          options->device_id, static_cast<std::size_t>(options->size_bytes), &data, error);
#else
      return fail(GPUXTB_STATUS_BACKEND_UNAVAILABLE,
                  "the gpuxtb library was built without CUDA support");
#endif
    }
    if (status != GPUXTB_STATUS_SUCCESS) {
      return fail(status, std::move(error));
    }

    gpuxtb::detail::ResultOwner* implementation = new (std::nothrow)
        gpuxtb::detail::ResultOwner(options->memory_space, options->device_id,
                                    static_cast<std::size_t>(options->size_bytes), data);
    if (implementation == nullptr) {
      if (options->memory_space == GPUXTB_MEMORY_HOST) {
        gpuxtb::detail::free_host_result_arena(data);
      } else {
#if defined(GPUXTB_HAS_CUDA)
        gpuxtb::detail::free_cuda_result_arena(options->device_id, data);
#endif
      }
      return fail(GPUXTB_STATUS_ALLOCATION_FAILED, "failed to allocate a result owner handle");
    }
    gpuxtb_result_owner_t* wrapper = new (std::nothrow) gpuxtb_result_owner_t{implementation};
    if (wrapper == nullptr) {
      implementation->release();
      return fail(GPUXTB_STATUS_ALLOCATION_FAILED, "failed to allocate a result owner wrapper");
    }
    *owner = wrapper;
    last_error.clear();
    return GPUXTB_STATUS_SUCCESS;
  } catch (const std::exception& exception) {
    return fail(GPUXTB_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(GPUXTB_STATUS_INTERNAL_ERROR, "unknown exception while creating a result owner");
  }
}

gpuxtb_status_t gpuxtb_result_owner_buffer(const gpuxtb_result_owner_t* owner,
                                           gpuxtb_buffer_t* buffer) {
  if (owner == nullptr || owner->implementation == nullptr) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "result owner is NULL");
  }
  if (buffer == nullptr) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "result owner buffer output is NULL");
  }
  *buffer = {owner->implementation->data(), owner->implementation->size_bytes(),
             owner->implementation->memory_space(), 0u};
  last_error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

void gpuxtb_result_owner_retain(gpuxtb_result_owner_t* owner) {
  if (owner == nullptr || owner->implementation == nullptr) {
    last_error = "result owner is NULL";
    return;
  }
  owner->implementation->retain();
  last_error.clear();
}

void gpuxtb_result_owner_release(gpuxtb_result_owner_t* owner) {
  if (owner == nullptr || owner->implementation == nullptr) {
    /* The public release contract makes NULL a harmless no-op.  Preserve any
     * prior diagnostic so callers can still inspect the failing operation. */
    return;
  }
  gpuxtb::detail::ResultOwner* implementation = owner->implementation;
  const bool final = implementation->release();
  last_error.clear();
  if (final) {
    /* ResultOwner::release() already destroyed the implementation. */
    delete owner;
  }
}

gpuxtb_status_t gpuxtb_result_owner_export_dltensor(const gpuxtb_result_owner_t* owner,
                                                    const gpuxtb_dlpack_view_t* view, int version,
                                                    void** out_managed) {
  /* On any failure *out_managed is set to NULL and no arena reference is
   * taken, so callers can treat a non-success status uniformly. */
  if (out_managed != nullptr) {
    *out_managed = nullptr;
  }
  if (owner == nullptr || owner->implementation == nullptr) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT, "result owner is NULL");
  }
  if (version != 0 && version != 1) {
    return fail(GPUXTB_STATUS_INVALID_ARGUMENT,
                "DLPack export version must be 0 (legacy) or 1 (versioned)");
  }
  try {
    return populate_dlpack_view(const_cast<gpuxtb_result_owner_t*>(owner), owner->implementation,
                                view, version != 0, out_managed);
  } catch (const std::bad_alloc&) {
    return fail(GPUXTB_STATUS_ALLOCATION_FAILED, "failed to allocate a DLPack managed tensor");
  } catch (const std::exception& exception) {
    return fail(GPUXTB_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(GPUXTB_STATUS_INTERNAL_ERROR,
                "unknown exception while exporting a DLPack managed tensor");
  }
}

}  // extern "C"
