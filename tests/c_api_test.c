#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpuxtb/gpuxtb.h"

_Static_assert(GPUXTB_COMPUTE_OPTIONS_V1_SIZE == 48,
               "compute-options ABI-v1 prefix must remain 48 bytes");
_Static_assert(GPUXTB_COMPUTE_OPTIONS_V2_SIZE == 56,
               "compute-options ABI-v2 prefix must remain 56 bytes");
_Static_assert(offsetof(gpuxtb_compute_options_t, scc_start_mode) == 48,
               "compute-options ABI-v2 mode must begin after the 48-byte prefix");
_Static_assert(offsetof(gpuxtb_compute_options_t, reserved_v2) == 52,
               "compute-options ABI-v2 reserved field must follow the mode");
_Static_assert(sizeof(gpuxtb_compute_options_t) == GPUXTB_COMPUTE_OPTIONS_V2_SIZE,
               "compute-options public layout must end at the ABI-v2 suffix");

static int check_short_compute_options_init(size_t caller_size) {
  enum { CANARY_BYTES = 16 };
  unsigned char* storage = (unsigned char*)malloc(caller_size + CANARY_BYTES);
  if (storage == NULL) {
    return 0;
  }
  memset(storage, 0xa5, caller_size + CANARY_BYTES);

  gpuxtb_compute_options_t* options = (gpuxtb_compute_options_t*)storage;
  const gpuxtb_status_t status = gpuxtb_compute_options_init(options, caller_size);
  const int prefix_ok = status == GPUXTB_STATUS_SUCCESS && options->struct_size == caller_size &&
                        options->api_version == GPUXTB_API_VERSION &&
                        options->model == GPUXTB_MODEL_GFN2_XTB &&
                        options->max_scc_iterations == 250 &&
                        options->electronic_temperature == GPUXTB_DEFAULT_ELECTRONIC_TEMPERATURE;

  int short_suffix_ok = 1;
  for (size_t index = GPUXTB_COMPUTE_OPTIONS_V1_SIZE; index < caller_size; ++index) {
    if (storage[index] != 0) {
      short_suffix_ok = 0;
      break;
    }
  }
  int canary_ok = 1;
  for (size_t index = caller_size; index < caller_size + CANARY_BYTES; ++index) {
    if (storage[index] != 0xa5) {
      canary_ok = 0;
      break;
    }
  }
  free(storage);
  return prefix_ok && short_suffix_ok && canary_ok;
}

/* Exercise the reusable workspace-query descriptor on a CPU-backed plan without
 * requiring a real inference (the value contract is validated in the native
 * plan tests; here we only prove the descriptor initializes and rejects short
 * structs). */
static int check_workspace_query_init(void) {
  gpuxtb_workspace_query_t query;
  if (gpuxtb_workspace_query_init(&query, sizeof(query)) != GPUXTB_STATUS_SUCCESS) {
    return 0;
  }
  if (query.struct_size != GPUXTB_WORKSPACE_QUERY_V1_SIZE ||
      query.api_version != GPUXTB_API_VERSION || query.compute_flags != 0 ||
      query.host_required_bytes != 0u || query.host_required_alignment != 0u ||
      query.device_required_bytes != 0u || query.device_required_alignment != 0u ||
      query.reserved != 0u || query.reserved_v2 != 0u) {
    return 0;
  }
  if (gpuxtb_workspace_query_init(&query, GPUXTB_WORKSPACE_QUERY_V1_SIZE - 1) !=
      GPUXTB_STATUS_INVALID_ARGUMENT) {
    return 0;
  }
  return 1;
}

int main(void) {
  if (sizeof(gpuxtb_status_t) != sizeof(int32_t) || sizeof(gpuxtb_backend_t) != sizeof(int32_t) ||
      sizeof(gpuxtb_memory_space_t) != sizeof(int32_t) ||
      sizeof(gpuxtb_model_t) != sizeof(int32_t) ||
      sizeof(gpuxtb_scc_start_mode_t) != sizeof(int32_t) ||
      sizeof(gpuxtb_compute_flag_t) != sizeof(int32_t) ||
      sizeof(gpuxtb_result_flag_t) != sizeof(int32_t)) {
    fprintf(stderr, "public ABI tags and flags must all be 32-bit\n");
    return 1;
  }

  gpuxtb_compute_options_t compute_options;
  if (gpuxtb_compute_options_init(&compute_options, sizeof(compute_options)) !=
          GPUXTB_STATUS_SUCCESS ||
      compute_options.struct_size != GPUXTB_COMPUTE_OPTIONS_V2_SIZE ||
      compute_options.electronic_temperature != GPUXTB_DEFAULT_ELECTRONIC_TEMPERATURE ||
      compute_options.scc_start_mode != GPUXTB_SCC_START_FRESH ||
      compute_options.reserved_v2 != 0) {
    fprintf(stderr, "ABI-v2 compute-options defaults are incorrect\n");
    return 2;
  }
  if (!check_short_compute_options_init(GPUXTB_COMPUTE_OPTIONS_V1_SIZE) ||
      !check_short_compute_options_init(GPUXTB_COMPUTE_OPTIONS_V2_SIZE - 1)) {
    fprintf(stderr, "short compute-options initialization wrote beyond the caller allocation\n");
    return 3;
  }
  if (!check_workspace_query_init()) {
    fprintf(stderr, "workspace-query descriptor initialization is incorrect\n");
    return 4;
  }

  gpuxtb_context_options_t options;
  if (gpuxtb_context_options_init(&options, sizeof(options)) != GPUXTB_STATUS_SUCCESS) {
    fprintf(stderr, "options initialization failed: %s\n", gpuxtb_get_last_error());
    return 4;
  }
  options.backend = GPUXTB_BACKEND_CPU;

  gpuxtb_context_t* context = NULL;
  gpuxtb_status_t status = gpuxtb_context_create(&options, &context);
  if (status != GPUXTB_STATUS_SUCCESS || context == NULL) {
    fprintf(stderr, "context creation failed: %s\n", gpuxtb_get_last_error());
    return 5;
  }
  if (gpuxtb_context_get_backend(context) != GPUXTB_BACKEND_CPU) {
    fprintf(stderr, "explicit CPU backend was not selected\n");
    return 6;
  }
  if (strcmp(gpuxtb_version_string(), "0.1.0") != 0) {
    fprintf(stderr, "unexpected version string\n");
    return 7;
  }
  if (strcmp(gpuxtb_status_string(GPUXTB_STATUS_SCC_NOT_CONVERGED), "SCC not converged") != 0 ||
      strcmp(gpuxtb_status_string(GPUXTB_STATUS_EIGENSOLVER_FAILED), "eigensolver failed") != 0) {
    fprintf(stderr, "per-system failure status strings are not stable\n");
    return 8;
  }

  gpuxtb_context_destroy(context);
  return 0;
}
