#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpuxtb/gpuxtb.h"

#if UINTPTR_MAX == UINT64_MAX
#define EXPECTED_CONTEXT_OPTIONS_SIZE 32
#define EXPECTED_BUFFER_SIZE 24
#define EXPECTED_BUFFER_SIZE_OFFSET 8
#define EXPECTED_BUFFER_MEMORY_OFFSET 16
#define EXPECTED_BATCH_V1_SIZE 328
#define EXPECTED_BATCH_V2_SIZE 352
#define EXPECTED_BATCH_V3_SIZE 408
#define EXPECTED_BATCH_TOTAL_INTERACTIONS_OFFSET 352
#define EXPECTED_BATCH_DESCRIPTORS_OFFSET 360
#define EXPECTED_BATCH_PAYLOAD_OFFSET 384
#define EXPECTED_RESULT_V1_SIZE 184
#define EXPECTED_RESULT_V2_SIZE 280
#define EXPECTED_RESULT_DIPOLE_OFFSET 184
#define EXPECTED_RESULT_QUADRUPOLE_OFFSET 208
#define EXPECTED_RESULT_WIBERG_OFFSET 232
#define EXPECTED_RESULT_SPIN_OFFSET 256
#define EXPECTED_DLPACK_SHAPE_OFFSET 40
#elif UINTPTR_MAX == UINT32_MAX
#define EXPECTED_CONTEXT_OPTIONS_SIZE 28
#define EXPECTED_BUFFER_SIZE 16
#define EXPECTED_BUFFER_SIZE_OFFSET 4
#define EXPECTED_BUFFER_MEMORY_OFFSET 8
#define EXPECTED_BATCH_V1_SIZE 232
#define EXPECTED_BATCH_V2_SIZE 248
#define EXPECTED_BATCH_V3_SIZE 288
#define EXPECTED_BATCH_TOTAL_INTERACTIONS_OFFSET 248
#define EXPECTED_BATCH_DESCRIPTORS_OFFSET 256
#define EXPECTED_BATCH_PAYLOAD_OFFSET 272
#define EXPECTED_RESULT_V1_SIZE 128
#define EXPECTED_RESULT_V2_SIZE 192
#define EXPECTED_RESULT_DIPOLE_OFFSET 128
#define EXPECTED_RESULT_QUADRUPOLE_OFFSET 144
#define EXPECTED_RESULT_WIBERG_OFFSET 160
#define EXPECTED_RESULT_SPIN_OFFSET 176
#define EXPECTED_DLPACK_SHAPE_OFFSET 36
#else
#error "c_api_test requires a 32-bit or 64-bit pointer ABI"
#endif

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
_Static_assert(offsetof(gpuxtb_context_options_t, stream) == 24,
               "context stream offset must remain stable");
_Static_assert(sizeof(gpuxtb_context_options_t) == EXPECTED_CONTEXT_OPTIONS_SIZE,
               "context layout must match the target pointer width");
_Static_assert(sizeof(gpuxtb_const_buffer_t) == EXPECTED_BUFFER_SIZE,
               "const-buffer layout must match the target pointer width");
_Static_assert(sizeof(gpuxtb_buffer_t) == EXPECTED_BUFFER_SIZE,
               "buffer layout must match the target pointer width");
_Static_assert(offsetof(gpuxtb_const_buffer_t, size_bytes) == EXPECTED_BUFFER_SIZE_OFFSET,
               "const-buffer size offset must match the target pointer width");
_Static_assert(offsetof(gpuxtb_const_buffer_t, memory_space) == EXPECTED_BUFFER_MEMORY_OFFSET,
               "const-buffer memory offset must match the target pointer width");
_Static_assert(offsetof(gpuxtb_dlpack_view_t, shape) == EXPECTED_DLPACK_SHAPE_OFFSET,
               "DLPack view shape offset must match the target pointer width");
_Static_assert(GPUXTB_BATCH_V1_SIZE == EXPECTED_BATCH_V1_SIZE,
               "batch ABI-v1 prefix must match the target pointer width");
_Static_assert(GPUXTB_BATCH_V2_SIZE == EXPECTED_BATCH_V2_SIZE,
               "batch ABI-v2 prefix must match the target pointer width");
_Static_assert(GPUXTB_BATCH_V3_SIZE == EXPECTED_BATCH_V3_SIZE,
               "batch ABI-v3 image must match the target pointer width");
_Static_assert(offsetof(gpuxtb_batch_t, total_interactions) ==
                   EXPECTED_BATCH_TOTAL_INTERACTIONS_OFFSET,
               "batch ABI-v3 interaction count must match the target pointer width");
_Static_assert(offsetof(gpuxtb_batch_t, interaction_descriptors) ==
                   EXPECTED_BATCH_DESCRIPTORS_OFFSET,
               "batch ABI-v3 descriptors must follow the interaction count");
_Static_assert(offsetof(gpuxtb_batch_t, interaction_payload) == EXPECTED_BATCH_PAYLOAD_OFFSET,
               "batch ABI-v3 payload must begin after the descriptor buffers");
_Static_assert(sizeof(gpuxtb_interaction_t) == GPUXTB_INTERACTION_V1_SIZE,
               "interaction descriptor public layout must end at its ABI-v1 suffix");
_Static_assert(offsetof(gpuxtb_interaction_t, type) == 0,
               "interaction type offset must remain stable");
_Static_assert(offsetof(gpuxtb_interaction_t, flags) == 4,
               "interaction flags offset must remain stable");
_Static_assert(offsetof(gpuxtb_interaction_t, system_index) == 8,
               "interaction system-index offset must remain stable");
_Static_assert(offsetof(gpuxtb_interaction_t, payload_offset) == 16,
               "interaction payload-offset offset must remain stable");
_Static_assert(offsetof(gpuxtb_interaction_t, payload_size) == 24,
               "interaction payload-size offset must remain stable");
_Static_assert(GPUXTB_BATCH_RESULT_V1_SIZE == EXPECTED_RESULT_V1_SIZE,
               "batch-result ABI-v1 prefix must match the target pointer width");
_Static_assert(GPUXTB_BATCH_RESULT_V2_SIZE == EXPECTED_RESULT_V2_SIZE,
               "batch-result ABI-v2 image must match the target pointer width");
_Static_assert(offsetof(gpuxtb_batch_result_t, dipole_moments) == EXPECTED_RESULT_DIPOLE_OFFSET,
               "batch-result ABI-v2 suffix must match the target pointer width");
_Static_assert(offsetof(gpuxtb_batch_result_t, quadrupole_moments) ==
                   EXPECTED_RESULT_QUADRUPOLE_OFFSET,
               "batch-result quadrupole outlet offset must remain stable");
_Static_assert(offsetof(gpuxtb_batch_result_t, wiberg_orders) == EXPECTED_RESULT_WIBERG_OFFSET,
               "batch-result Wiberg outlet offset must remain stable");
_Static_assert(offsetof(gpuxtb_batch_result_t, spin_populations) == EXPECTED_RESULT_SPIN_OFFSET,
               "batch-result spin outlet offset must remain stable");
_Static_assert(GPUXTB_RESULT_DIPOLE_MOMENTS == (1 << 4),
               "dipole publication result flag must remain at bit 4");
_Static_assert(sizeof(gpuxtb_request_state_t) == sizeof(int32_t),
               "request state tag must remain 32-bit");
_Static_assert(GPUXTB_REQUEST_INFO_V1_SIZE == 24, "request-info ABI-v1 image must remain 24 bytes");
_Static_assert(sizeof(gpuxtb_request_info_t) == GPUXTB_REQUEST_INFO_V1_SIZE,
               "request-info public layout must end at ABI v1");
_Static_assert(offsetof(gpuxtb_request_info_t, state) == 8,
               "request-info state offset must remain stable");
_Static_assert(offsetof(gpuxtb_request_info_t, completion_status) == 12,
               "request-info completion status offset must remain stable");
_Static_assert(offsetof(gpuxtb_request_info_t, result_flags) == 16,
               "request-info result flags offset must remain stable");

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

static int check_short_batch_init(size_t caller_size) {
  enum { CANARY_BYTES = 16 };
  unsigned char* storage = (unsigned char*)malloc(caller_size + CANARY_BYTES);
  if (storage == NULL) {
    return 0;
  }
  memset(storage, 0xa5, caller_size + CANARY_BYTES);

  gpuxtb_batch_t* batch = (gpuxtb_batch_t*)storage;
  const gpuxtb_status_t status = gpuxtb_batch_init(batch, caller_size);
  int ok = status == GPUXTB_STATUS_SUCCESS && batch->struct_size == caller_size &&
           batch->api_version == GPUXTB_API_VERSION;
  for (size_t index = GPUXTB_BATCH_V1_SIZE; ok && index < caller_size; ++index) {
    ok = storage[index] == 0;
  }
  for (size_t index = caller_size; ok && index < caller_size + CANARY_BYTES; ++index) {
    ok = storage[index] == 0xa5;
  }
  free(storage);
  return ok;
}

static int check_short_batch_result_init(size_t caller_size) {
  enum { CANARY_BYTES = 16 };
  unsigned char* storage = (unsigned char*)malloc(caller_size + CANARY_BYTES);
  if (storage == NULL) {
    return 0;
  }
  memset(storage, 0xa5, caller_size + CANARY_BYTES);

  gpuxtb_batch_result_t* result = (gpuxtb_batch_result_t*)storage;
  const gpuxtb_status_t status = gpuxtb_batch_result_init(result, caller_size);
  int ok = status == GPUXTB_STATUS_SUCCESS && result->struct_size == caller_size &&
           result->api_version == GPUXTB_API_VERSION;
  for (size_t index = GPUXTB_BATCH_RESULT_V1_SIZE; ok && index < caller_size; ++index) {
    ok = storage[index] == 0;
  }
  for (size_t index = caller_size; ok && index < caller_size + CANARY_BYTES; ++index) {
    ok = storage[index] == 0xa5;
  }
  free(storage);
  return ok;
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
  if (!check_short_batch_init(GPUXTB_BATCH_V1_SIZE) ||
      !check_short_batch_init(GPUXTB_BATCH_V2_SIZE) ||
      !check_short_batch_init(GPUXTB_BATCH_V3_SIZE - 1)) {
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
  if (!check_short_batch_result_init(GPUXTB_BATCH_RESULT_V1_SIZE) ||
      !check_short_batch_result_init(GPUXTB_BATCH_RESULT_V2_SIZE - 1)) {
    return 0;
  }
  return 1;
}

static int check_request_info_init(void) {
  struct extended_request_info {
    gpuxtb_request_info_t info;
    unsigned char future[16];
  } extended;
  memset(&extended, 0xa5, sizeof(extended));
  if (gpuxtb_request_info_init(&extended.info, sizeof(extended)) != GPUXTB_STATUS_SUCCESS) {
    return 0;
  }
  if (extended.info.struct_size != sizeof(extended) ||
      extended.info.api_version != GPUXTB_API_VERSION ||
      extended.info.state != GPUXTB_REQUEST_IDLE ||
      extended.info.completion_status != GPUXTB_STATUS_SUCCESS ||
      extended.info.result_flags != 0u || extended.info.reserved != 0u) {
    return 0;
  }
  for (size_t index = 0; index < sizeof(extended.future); ++index) {
    if (extended.future[index] != 0xa5) {
      return 0;
    }
  }

  gpuxtb_request_info_t short_info;
  memset(&short_info, 0xa5, sizeof(short_info));
  if (gpuxtb_request_info_init(&short_info, GPUXTB_REQUEST_INFO_V1_SIZE - 1) !=
          GPUXTB_STATUS_INVALID_ARGUMENT ||
      short_info.struct_size != 0xa5a5a5a5u ||
      gpuxtb_request_info_init(NULL, sizeof(short_info)) != GPUXTB_STATUS_INVALID_ARGUMENT) {
    return 0;
  }
  return 1;
}

static int check_cpu_request_shell(gpuxtb_context_t* context) {
  gpuxtb_request_t* request = NULL;
  if (gpuxtb_request_create(context, &request) != GPUXTB_STATUS_SUCCESS || request == NULL) {
    return 0;
  }

  gpuxtb_request_info_t info;
  if (gpuxtb_request_info_init(&info, sizeof(info)) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb_request_query(request, &info) != GPUXTB_STATUS_SUCCESS ||
      info.state != GPUXTB_REQUEST_IDLE || info.completion_status != GPUXTB_STATUS_SUCCESS ||
      info.result_flags != 0u || strcmp(gpuxtb_request_get_error(request), "") != 0) {
    gpuxtb_request_destroy(request);
    return 0;
  }

  /* CPU capability probing happens before descriptor validation. Even hostile
   * NULL descriptors therefore leave both caller outputs and request state
   * untouched, allowing a synchronous fallback without state repair. */
  gpuxtb_batch_result_t result;
  memset(&result, 0, sizeof(result));
  result.flags = 0x5a5a5a5au;
  double energy = 123.5;
  result.energies.data = &energy;
  result.energies.size_bytes = sizeof(energy);
  result.energies.memory_space = GPUXTB_MEMORY_HOST;
  if (gpuxtb_compute_enqueue(context, NULL, NULL, &result, request) !=
          GPUXTB_STATUS_NOT_SUPPORTED ||
      result.flags != 0x5a5a5a5au || energy != 123.5) {
    gpuxtb_request_destroy(request);
    return 0;
  }
  /* Repeated unsupported probes exercise reuse without creating a hidden
   * PENDING transition or retaining the first call's descriptors. */
  if (gpuxtb_compute_enqueue(context, NULL, NULL, NULL, request) != GPUXTB_STATUS_NOT_SUPPORTED) {
    gpuxtb_request_destroy(request);
    return 0;
  }
  if (gpuxtb_request_wait(request, &info) != GPUXTB_STATUS_SUCCESS ||
      info.state != GPUXTB_REQUEST_IDLE || info.completion_status != GPUXTB_STATUS_SUCCESS ||
      info.result_flags != 0u || strcmp(gpuxtb_request_get_error(request), "") != 0) {
    gpuxtb_request_destroy(request);
    return 0;
  }

  gpuxtb_request_info_t invalid_info = info;
  invalid_info.api_version = GPUXTB_API_VERSION + 1u;
  invalid_info.state = (gpuxtb_request_state_t)77;
  if (gpuxtb_request_query(request, &invalid_info) != GPUXTB_STATUS_INVALID_ARGUMENT ||
      invalid_info.state != (gpuxtb_request_state_t)77) {
    gpuxtb_request_destroy(request);
    return 0;
  }
  invalid_info = info;
  invalid_info.reserved = 1u;
  invalid_info.result_flags = 0x11223344u;
  if (gpuxtb_request_wait(request, &invalid_info) != GPUXTB_STATUS_INVALID_ARGUMENT ||
      invalid_info.result_flags != 0x11223344u) {
    gpuxtb_request_destroy(request);
    return 0;
  }

  gpuxtb_request_destroy(request);
  gpuxtb_request_destroy(NULL);
  return 1;
}

int main(void) {
  if (sizeof(gpuxtb_status_t) != sizeof(int32_t) || sizeof(gpuxtb_backend_t) != sizeof(int32_t) ||
      sizeof(gpuxtb_memory_space_t) != sizeof(int32_t) ||
      sizeof(gpuxtb_model_t) != sizeof(int32_t) ||
      sizeof(gpuxtb_scc_start_mode_t) != sizeof(int32_t) ||
      sizeof(gpuxtb_compute_flag_t) != sizeof(int32_t) ||
      sizeof(gpuxtb_result_flag_t) != sizeof(int32_t) ||
      sizeof(gpuxtb_request_state_t) != sizeof(int32_t) ||
      sizeof(gpuxtb_interaction_type_t) != sizeof(int32_t)) {
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
  if (!check_request_info_init()) {
    fprintf(stderr, "request-info descriptor initialization is incorrect\n");
    return 6;
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
  if (!check_cpu_request_shell(context)) {
    fprintf(stderr, "CPU request ABI shell behavior is incorrect: %s\n", gpuxtb_get_last_error());
    return 9;
  }

  gpuxtb_context_destroy(context);
  return 0;
}
