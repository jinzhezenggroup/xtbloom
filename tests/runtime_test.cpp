#include <cstdint>
#include <cstring>

#include "gpuxtb/gpuxtb.h"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

int main() {
  gpuxtb_context_options_t options;
  CHECK(gpuxtb_context_options_init(&options, sizeof(options)) == GPUXTB_STATUS_SUCCESS);
  options.backend = GPUXTB_BACKEND_CPU;

  gpuxtb_context_t* context = nullptr;
  CHECK(gpuxtb_context_create(&options, &context) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb_context_get_device_id(context) == -1);

  gpuxtb_batch_t batch;
  gpuxtb_compute_options_t compute_options;
  gpuxtb_batch_result_t result;
  CHECK(gpuxtb_batch_init(&batch, sizeof(batch)) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb_compute_options_init(&compute_options, sizeof(compute_options)) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb_batch_result_init(&result, sizeof(result)) == GPUXTB_STATUS_SUCCESS);

  const gpuxtb_status_t compute_status = gpuxtb_compute(context, &batch, &compute_options, &result);
  CHECK(compute_status == GPUXTB_STATUS_NOT_IMPLEMENTED);
  CHECK(std::strstr(gpuxtb_get_last_error(), "not been implemented") != nullptr);

  gpuxtb_context_destroy(context);

  gpuxtb_context_t* invalid_context = nullptr;
  options.device_id = -2;
  CHECK(gpuxtb_context_create(&options, &invalid_context) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(invalid_context == nullptr);

  struct ExtendedOptions {
    gpuxtb_context_options_t options;
    std::uint64_t canary;
  } extended{};
  extended.canary = UINT64_C(0x5a5a5a5aa5a5a5a5);
  CHECK(gpuxtb_context_options_init(&extended.options, sizeof(extended)) == GPUXTB_STATUS_SUCCESS);
  CHECK(extended.options.struct_size == sizeof(extended));
  CHECK(extended.canary == UINT64_C(0x5a5a5a5aa5a5a5a5));

  CHECK(gpuxtb_context_options_init(&options, GPUXTB_CONTEXT_OPTIONS_V1_SIZE - 1) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);

#if defined(GPUXTB_TEST_HAS_CUDA)
  CHECK(gpuxtb_context_options_init(&options, sizeof(options)) == GPUXTB_STATUS_SUCCESS);
  options.backend = GPUXTB_BACKEND_CUDA;
  context = nullptr;
  const gpuxtb_status_t cuda_status = gpuxtb_context_create(&options, &context);
  if (cuda_status == GPUXTB_STATUS_SUCCESS) {
    CHECK(gpuxtb_context_get_backend(context) == GPUXTB_BACKEND_CUDA);
    CHECK(gpuxtb_context_get_device_id(context) >= 0);
    gpuxtb_context_destroy(context);
  } else {
    /* CUDA-enabled builds also run on hosts where the runtime exposes no device. */
    CHECK(cuda_status == GPUXTB_STATUS_BACKEND_UNAVAILABLE);
    CHECK(context == nullptr);
  }

  CHECK(gpuxtb_context_options_init(&options, sizeof(options)) == GPUXTB_STATUS_SUCCESS);
  options.backend = GPUXTB_BACKEND_AUTO;
  options.device_id = INT32_MAX;
  context = nullptr;
  CHECK(gpuxtb_context_create(&options, &context) == GPUXTB_STATUS_BACKEND_UNAVAILABLE);
  CHECK(context == nullptr);
#endif
  return 0;
}
