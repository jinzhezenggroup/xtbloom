#include <stdio.h>

#include "gpuxtb/gpuxtb.h"

int main(void) {
  gpuxtb_context_options_t options;
  if (gpuxtb_context_options_init(&options, sizeof(options)) != GPUXTB_STATUS_SUCCESS) {
    fprintf(stderr, "%s\n", gpuxtb_get_last_error());
    return 1;
  }
  options.backend = GPUXTB_BACKEND_CPU;

  gpuxtb_context_t* context = NULL;
  if (gpuxtb_context_create(&options, &context) != GPUXTB_STATUS_SUCCESS) {
    fprintf(stderr, "%s\n", gpuxtb_get_last_error());
    return 2;
  }
  gpuxtb_context_destroy(context);
  return 0;
}
