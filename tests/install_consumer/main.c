#include <stdio.h>

#include "gpuxtb/gpuxtb.h"

_Static_assert(GPUXTB_COMPUTE_OPTIONS_V1_SIZE == 48,
               "installed ABI-v1 compute-options prefix must remain 48 bytes");
_Static_assert(GPUXTB_COMPUTE_OPTIONS_V2_SIZE == 56,
               "installed ABI-v2 compute-options prefix must remain 56 bytes");
_Static_assert(offsetof(gpuxtb_compute_options_t, scc_start_mode) == 48,
               "installed ABI-v2 start mode offset must remain stable");
_Static_assert(sizeof(gpuxtb_compute_options_t) == GPUXTB_COMPUTE_OPTIONS_V2_SIZE,
               "installed compute-options layout must include the ABI-v2 suffix");

int main(void) {
  gpuxtb_compute_options_t compute_options;
  if (gpuxtb_compute_options_init(&compute_options, sizeof(compute_options)) !=
          GPUXTB_STATUS_SUCCESS ||
      compute_options.scc_start_mode != GPUXTB_SCC_START_FRESH ||
      compute_options.reserved_v2 != 0) {
    fprintf(stderr, "installed compute-options ABI-v2 defaults are incorrect\n");
    return 1;
  }

  gpuxtb_context_options_t options;
  if (gpuxtb_context_options_init(&options, sizeof(options)) != GPUXTB_STATUS_SUCCESS) {
    fprintf(stderr, "%s\n", gpuxtb_get_last_error());
    return 2;
  }
  options.backend = GPUXTB_BACKEND_CPU;

  gpuxtb_context_t* context = NULL;
  if (gpuxtb_context_create(&options, &context) != GPUXTB_STATUS_SUCCESS) {
    fprintf(stderr, "%s\n", gpuxtb_get_last_error());
    return 3;
  }
  gpuxtb_context_destroy(context);
  return 0;
}
