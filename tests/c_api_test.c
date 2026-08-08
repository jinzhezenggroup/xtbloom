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
_Static_assert(sizeof(gpuxtb_result_owner_options_t) >= GPUXTB_RESULT_OWNER_OPTIONS_V1_SIZE,
               "result-owner options prefix must fit the public layout");
_Static_assert(offsetof(gpuxtb_result_owner_options_t, memory_space) == 8,
               "result-owner memory-space offset must remain stable");
_Static_assert(offsetof(gpuxtb_result_owner_options_t, size_bytes) == 16,
               "result-owner size offset must remain stable");
_Static_assert(offsetof(gpuxtb_result_owner_options_t, reserved) == 24,
               "result-owner reserved offset must remain stable");
_Static_assert(sizeof(gpuxtb_dlpack_view_t) == GPUXTB_DLPACK_VIEW_V1_SIZE,
               "DLPack view public layout must end at the ABI-v1 suffix");
_Static_assert(offsetof(gpuxtb_dlpack_view_t, byte_offset) == 8,
               "DLPack view byte-offset offset must remain stable");
_Static_assert(offsetof(gpuxtb_dlpack_view_t, shape) == 40,
               "DLPack view shape offset must remain stable");
_Static_assert(GPUXTB_BATCH_V1_SIZE == 328, "batch ABI-v1 prefix must remain 328 bytes");
_Static_assert(GPUXTB_BATCH_V2_SIZE == 352, "batch ABI-v2 prefix must remain 352 bytes");
_Static_assert(GPUXTB_BATCH_V3_SIZE == 408, "batch ABI-v3 image must remain 408 bytes");
_Static_assert(offsetof(gpuxtb_batch_t, total_interactions) == 352,
               "batch ABI-v3 interaction count must begin after the 352-byte prefix");
_Static_assert(offsetof(gpuxtb_batch_t, interaction_payload) == 384,
               "batch ABI-v3 payload must begin after the descriptor buffers");
_Static_assert(sizeof(gpuxtb_interaction_t) == GPUXTB_INTERACTION_V1_SIZE,
               "interaction descriptor public layout must end at its ABI-v1 suffix");
_Static_assert(offsetof(gpuxtb_interaction_t, type) == 0,
               "interaction type offset must remain stable");
_Static_assert(offsetof(gpuxtb_interaction_t, system_index) == 8,
               "interaction system-index offset must remain stable");
_Static_assert(offsetof(gpuxtb_interaction_t, payload_size) == 24,
               "interaction payload-size offset must remain stable");
_Static_assert(GPUXTB_BATCH_RESULT_V1_SIZE == 184,
               "batch-result ABI-v1 prefix must remain 184 bytes");
_Static_assert(GPUXTB_BATCH_RESULT_V2_SIZE == 280,
               "batch-result ABI-v2 image must remain 280 bytes");
_Static_assert(offsetof(gpuxtb_batch_result_t, dipole_moments) == 184,
               "batch-result ABI-v2 suffix must begin after the 184-byte prefix");

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

/* Exercise the ABI-v3 batch and ABI-v2 result initializers: full-size images
 * must zero the interaction suffix and the new result outlets, while a short
 * ABI-v1 prefix must still initialize without touching caller bytes it does
 * not own. */
static int check_batch_result_suffix_init(void) {
  gpuxtb_batch_t batch;
  if (gpuxtb_batch_init(&batch, sizeof(batch)) != GPUXTB_STATUS_SUCCESS) {
    return 0;
  }
  if (batch.struct_size != GPUXTB_BATCH_V3_SIZE || batch.api_version != GPUXTB_API_VERSION ||
      batch.total_interactions != 0 || batch.interaction_descriptors.data != NULL ||
      batch.interaction_payload.data != NULL) {
    return 0;
  }
  gpuxtb_batch_t short_batch;
  if (gpuxtb_batch_init(&short_batch, GPUXTB_BATCH_V1_SIZE) != GPUXTB_STATUS_SUCCESS ||
      short_batch.struct_size != GPUXTB_BATCH_V1_SIZE) {
    return 0;
  }

  gpuxtb_batch_result_t result;
  if (gpuxtb_batch_result_init(&result, sizeof(result)) != GPUXTB_STATUS_SUCCESS) {
    return 0;
  }
  if (result.struct_size != GPUXTB_BATCH_RESULT_V2_SIZE ||
      result.api_version != GPUXTB_API_VERSION || result.dipole_moments.data != NULL ||
      result.quadrupole_moments.data != NULL || result.wiberg_orders.data != NULL ||
      result.spin_populations.data != NULL) {
    return 0;
  }
  gpuxtb_batch_result_t short_result;
  if (gpuxtb_batch_result_init(&short_result, GPUXTB_BATCH_RESULT_V1_SIZE) !=
          GPUXTB_STATUS_SUCCESS ||
      short_result.struct_size != GPUXTB_BATCH_RESULT_V1_SIZE) {
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
  if (!check_batch_result_suffix_init()) {
    fprintf(stderr, "batch/result suffix initialization is incorrect\n");
    return 5;
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
