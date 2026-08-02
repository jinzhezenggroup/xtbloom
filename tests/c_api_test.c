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

  gpuxtb_compute_options_t compute_options;
  if (gpuxtb_compute_options_init(&compute_options, sizeof(compute_options)) !=
          GPUXTB_STATUS_SUCCESS ||
      compute_options.electronic_temperature != GPUXTB_DEFAULT_ELECTRONIC_TEMPERATURE) {
    fprintf(stderr, "compute options must default to the 300 K atomic-unit energy scale\n");
    return 2;
  }

  gpuxtb_context_options_t options;
  if (gpuxtb_context_options_init(&options, sizeof(options)) != GPUXTB_STATUS_SUCCESS) {
    fprintf(stderr, "options initialization failed: %s\n", gpuxtb_get_last_error());
    return 3;
  }
  options.backend = GPUXTB_BACKEND_CPU;

  gpuxtb_context_t* context = NULL;
  gpuxtb_status_t status = gpuxtb_context_create(&options, &context);
  if (status != GPUXTB_STATUS_SUCCESS || context == NULL) {
    fprintf(stderr, "context creation failed: %s\n", gpuxtb_get_last_error());
    return 4;
  }
  if (gpuxtb_context_get_backend(context) != GPUXTB_BACKEND_CPU) {
    fprintf(stderr, "explicit CPU backend was not selected\n");
    return 5;
  }
  if (strcmp(gpuxtb_version_string(), "0.1.0") != 0) {
    fprintf(stderr, "unexpected version string\n");
    return 6;
  }
  if (strcmp(gpuxtb_status_string(GPUXTB_STATUS_SCC_NOT_CONVERGED), "SCC not converged") != 0 ||
      strcmp(gpuxtb_status_string(GPUXTB_STATUS_EIGENSOLVER_FAILED), "eigensolver failed") != 0) {
    fprintf(stderr, "per-system failure status strings are not stable\n");
    return 7;
  }

  gpuxtb_context_destroy(context);
  return 0;
}
