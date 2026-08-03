#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <string>
#include <utility>

#include "gpuxtb/gpuxtb.h"
#include "runtime/backend.hpp"
#include "runtime/gfn2_cpu_execution.hpp"
#include "runtime/validation.hpp"

struct gpuxtb_context {
  gpuxtb::detail::Context* implementation;
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
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t gpuxtb_batch_result_init(gpuxtb_batch_result_t* result, size_t struct_size) {
  return initialize_structure(result, struct_size, GPUXTB_BATCH_RESULT_V1_SIZE, "batch result");
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

    /*
     * HOST tags at a CUDA boundary are not proof that the pointer is CPU
     * accessible: a caller may have mislabeled device memory. Until the CUDA
     * topology bridge verifies pointer attributes and stages its six metadata
     * classes, CUDA stops after the no-dereference structural layer. The
     * placeholder below therefore returns NOT_IMPLEMENTED without launching
     * kernels or modifying caller-owned output. CPU retains complete host
     * topology semantic validation through validate_compute_descriptors().
     */
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
      const gpuxtb_status_t status = gpuxtb::detail::execute_restricted_gfn2_cpu(
          *cache, *batch, *options, *result, error);
      if (status != GPUXTB_STATUS_SUCCESS) {
        return fail(status, std::move(error));
      }
      last_error.clear();
      return GPUXTB_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
      return fail(GPUXTB_STATUS_ALLOCATION_FAILED,
                  "failed to allocate CPU GFN2 execution state");
    } catch (const std::exception& exception) {
      return fail(GPUXTB_STATUS_INTERNAL_ERROR, exception.what());
    } catch (...) {
      return fail(GPUXTB_STATUS_INTERNAL_ERROR,
                  "unknown exception while executing CPU GFN2 inference");
    }
  }

  return fail(GPUXTB_STATUS_NOT_IMPLEMENTED,
              "CUDA GFN2 public inference is not implemented yet");
}

}  // extern "C"
