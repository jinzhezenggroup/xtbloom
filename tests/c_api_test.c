#include <stdio.h>
#include <string.h>

#include "gpuxtb/gpuxtb.h"

int main(void) {
  if (sizeof(gpuxtb_status_t) != sizeof(int32_t) || sizeof(gpuxtb_backend_t) != sizeof(int32_t) ||
      sizeof(gpuxtb_memory_space_t) != sizeof(int32_t) ||
      sizeof(gpuxtb_model_t) != sizeof(int32_t) ||
      sizeof(gpuxtb_compute_flag_t) != sizeof(int32_t) ||
      sizeof(gpuxtb_result_flag_t) != sizeof(int32_t)) {
    fprintf(stderr, "public ABI tags and flags must all be 32-bit\n");
    return 1;
  }

  gpuxtb_context_options_t options;
  if (gpuxtb_context_options_init(&options, sizeof(options)) != GPUXTB_STATUS_SUCCESS) {
    fprintf(stderr, "options initialization failed: %s\n", gpuxtb_get_last_error());
    return 2;
  }
  options.backend = GPUXTB_BACKEND_CPU;

  gpuxtb_context_t* context = NULL;
  gpuxtb_status_t status = gpuxtb_context_create(&options, &context);
  if (status != GPUXTB_STATUS_SUCCESS || context == NULL) {
    fprintf(stderr, "context creation failed: %s\n", gpuxtb_get_last_error());
    return 3;
  }
  if (gpuxtb_context_get_backend(context) != GPUXTB_BACKEND_CPU) {
    fprintf(stderr, "explicit CPU backend was not selected\n");
    return 4;
  }
  if (strcmp(gpuxtb_version_string(), "0.1.0") != 0) {
    fprintf(stderr, "unexpected version string\n");
    return 5;
  }

  gpuxtb_context_destroy(context);
  return 0;
}
