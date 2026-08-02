#include <stdio.h>
#include <string.h>

#include "gpuxtb/gpuxtb.h"

int main(void) {
  gpuxtb_context_options_t options;
  if (gpuxtb_context_options_init(&options, sizeof(options)) != GPUXTB_STATUS_SUCCESS) {
    fprintf(stderr, "options initialization failed: %s\n", gpuxtb_get_last_error());
    return 1;
  }
  options.backend = GPUXTB_BACKEND_CPU;

  gpuxtb_context_t* context = NULL;
  gpuxtb_status_t status = gpuxtb_context_create(&options, &context);
  if (status != GPUXTB_STATUS_SUCCESS || context == NULL) {
    fprintf(stderr, "context creation failed: %s\n", gpuxtb_get_last_error());
    return 2;
  }
  if (gpuxtb_context_get_backend(context) != GPUXTB_BACKEND_CPU) {
    fprintf(stderr, "explicit CPU backend was not selected\n");
    return 3;
  }
  if (strcmp(gpuxtb_version_string(), "0.1.0") != 0) {
    fprintf(stderr, "unexpected version string\n");
    return 4;
  }

  gpuxtb_context_destroy(context);
  return 0;
}
