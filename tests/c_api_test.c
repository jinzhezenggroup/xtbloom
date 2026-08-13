#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "xtbloom/xtbloom.h"

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
#define EXPECTED_BATCH_V4_SIZE 456
#define EXPECTED_BATCH_CELL_MATRICES_OFFSET 408
#define EXPECTED_BATCH_PERIODIC_AXES_OFFSET 432
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
#define EXPECTED_BATCH_V4_SIZE 320
#define EXPECTED_BATCH_CELL_MATRICES_OFFSET 288
#define EXPECTED_BATCH_PERIODIC_AXES_OFFSET 304
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

_Static_assert(XTBLOOM_COMPUTE_OPTIONS_V1_SIZE == 48,
               "compute-options ABI-v1 prefix must remain 48 bytes");
_Static_assert(XTBLOOM_COMPUTE_OPTIONS_V2_SIZE == 56,
               "compute-options ABI-v2 prefix must remain 56 bytes");
_Static_assert(XTBLOOM_COMPUTE_OPTIONS_V3_SIZE == 80,
               "compute-options ABI-v3 image must remain 80 bytes");
_Static_assert(offsetof(xtbloom_compute_options_t, scc_start_mode) == 48,
               "compute-options ABI-v2 mode must begin after the 48-byte prefix");
_Static_assert(offsetof(xtbloom_compute_options_t, reserved_v2) == 52,
               "compute-options ABI-v2 reserved field must follow the mode");
_Static_assert(offsetof(xtbloom_compute_options_t, scc_mixer) == 56,
               "compute-options ABI-v3 mixer must begin after the ABI-v2 suffix");
_Static_assert(offsetof(xtbloom_compute_options_t, scc_mixer_history) == 60,
               "compute-options ABI-v3 history offset must remain stable");
_Static_assert(offsetof(xtbloom_compute_options_t, scc_mixer_damping) == 64,
               "compute-options ABI-v3 damping offset must remain stable");
_Static_assert(offsetof(xtbloom_compute_options_t, determinism) == 72,
               "compute-options ABI-v3 determinism offset must remain stable");
_Static_assert(offsetof(xtbloom_compute_options_t, reserved_v3) == 76,
               "compute-options ABI-v3 reserved offset must remain stable");
_Static_assert(sizeof(xtbloom_compute_options_t) == XTBLOOM_COMPUTE_OPTIONS_V3_SIZE,
               "compute-options public layout must end at the ABI-v3 suffix");
_Static_assert(sizeof(xtbloom_result_owner_options_t) >= XTBLOOM_RESULT_OWNER_OPTIONS_V1_SIZE,
               "result-owner options prefix must fit the public layout");
_Static_assert(offsetof(xtbloom_result_owner_options_t, memory_space) == 8,
               "result-owner memory-space offset must remain stable");
_Static_assert(offsetof(xtbloom_result_owner_options_t, size_bytes) == 16,
               "result-owner size offset must remain stable");
_Static_assert(offsetof(xtbloom_result_owner_options_t, reserved) == 24,
               "result-owner reserved offset must remain stable");
_Static_assert(sizeof(xtbloom_dlpack_view_t) == XTBLOOM_DLPACK_VIEW_V1_SIZE,
               "DLPack view public layout must end at the ABI-v1 suffix");
_Static_assert(offsetof(xtbloom_dlpack_view_t, byte_offset) == 8,
               "DLPack view byte-offset offset must remain stable");
_Static_assert(offsetof(xtbloom_context_options_t, stream) == 24,
               "context stream offset must remain stable");
_Static_assert(sizeof(xtbloom_context_options_t) == EXPECTED_CONTEXT_OPTIONS_SIZE,
               "context layout must match the target pointer width");
_Static_assert(sizeof(xtbloom_const_buffer_t) == EXPECTED_BUFFER_SIZE,
               "const-buffer layout must match the target pointer width");
_Static_assert(sizeof(xtbloom_buffer_t) == EXPECTED_BUFFER_SIZE,
               "buffer layout must match the target pointer width");
_Static_assert(offsetof(xtbloom_const_buffer_t, size_bytes) == EXPECTED_BUFFER_SIZE_OFFSET,
               "const-buffer size offset must match the target pointer width");
_Static_assert(offsetof(xtbloom_const_buffer_t, memory_space) == EXPECTED_BUFFER_MEMORY_OFFSET,
               "const-buffer memory offset must match the target pointer width");
_Static_assert(offsetof(xtbloom_dlpack_view_t, shape) == EXPECTED_DLPACK_SHAPE_OFFSET,
               "DLPack view shape offset must match the target pointer width");
_Static_assert(XTBLOOM_BATCH_V1_SIZE == EXPECTED_BATCH_V1_SIZE,
               "batch ABI-v1 prefix must match the target pointer width");
_Static_assert(XTBLOOM_BATCH_V2_SIZE == EXPECTED_BATCH_V2_SIZE,
               "batch ABI-v2 prefix must match the target pointer width");
_Static_assert(XTBLOOM_BATCH_V3_SIZE == EXPECTED_BATCH_V3_SIZE,
               "batch ABI-v3 image must match the target pointer width");
_Static_assert(offsetof(xtbloom_batch_t, total_interactions) ==
                   EXPECTED_BATCH_TOTAL_INTERACTIONS_OFFSET,
               "batch ABI-v3 interaction count must match the target pointer width");
_Static_assert(offsetof(xtbloom_batch_t, interaction_descriptors) ==
                   EXPECTED_BATCH_DESCRIPTORS_OFFSET,
               "batch ABI-v3 descriptors must follow the interaction count");
_Static_assert(offsetof(xtbloom_batch_t, interaction_payload) == EXPECTED_BATCH_PAYLOAD_OFFSET,
               "batch ABI-v3 payload must begin after the descriptor buffers");
_Static_assert(XTBLOOM_BATCH_V4_SIZE == EXPECTED_BATCH_V4_SIZE,
               "batch ABI-v4 image must match the target pointer width");
_Static_assert(offsetof(xtbloom_batch_t, cell_matrices) ==
                   EXPECTED_BATCH_CELL_MATRICES_OFFSET,
               "batch ABI-v4 cell matrices must follow the v3 image");
_Static_assert(offsetof(xtbloom_batch_t, periodic_axes) ==
                   EXPECTED_BATCH_PERIODIC_AXES_OFFSET,
               "batch ABI-v4 periodic axes must follow the cell matrices");
_Static_assert(sizeof(xtbloom_periodic_axes_t) == sizeof(int32_t),
               "periodic-axis mask must remain fixed-width");
_Static_assert(sizeof(xtbloom_interaction_t) == XTBLOOM_INTERACTION_V1_SIZE,
               "interaction descriptor public layout must end at its ABI-v1 suffix");
_Static_assert(offsetof(xtbloom_interaction_t, type) == 0,
               "interaction type offset must remain stable");
_Static_assert(offsetof(xtbloom_interaction_t, flags) == 4,
               "interaction flags offset must remain stable");
_Static_assert(offsetof(xtbloom_interaction_t, system_index) == 8,
               "interaction system-index offset must remain stable");
_Static_assert(offsetof(xtbloom_interaction_t, payload_offset) == 16,
               "interaction payload-offset offset must remain stable");
_Static_assert(offsetof(xtbloom_interaction_t, payload_size) == 24,
               "interaction payload-size offset must remain stable");
_Static_assert(XTBLOOM_BATCH_RESULT_V1_SIZE == EXPECTED_RESULT_V1_SIZE,
               "batch-result ABI-v1 prefix must match the target pointer width");
_Static_assert(XTBLOOM_BATCH_RESULT_V2_SIZE == EXPECTED_RESULT_V2_SIZE,
               "batch-result ABI-v2 image must match the target pointer width");
_Static_assert(offsetof(xtbloom_batch_result_t, dipole_moments) == EXPECTED_RESULT_DIPOLE_OFFSET,
               "batch-result ABI-v2 suffix must match the target pointer width");
_Static_assert(offsetof(xtbloom_batch_result_t, quadrupole_moments) ==
                   EXPECTED_RESULT_QUADRUPOLE_OFFSET,
               "batch-result quadrupole outlet offset must remain stable");
_Static_assert(offsetof(xtbloom_batch_result_t, wiberg_orders) == EXPECTED_RESULT_WIBERG_OFFSET,
               "batch-result Wiberg outlet offset must remain stable");
_Static_assert(offsetof(xtbloom_batch_result_t, spin_populations) == EXPECTED_RESULT_SPIN_OFFSET,
               "batch-result spin outlet offset must remain stable");
_Static_assert(XTBLOOM_RESULT_DIPOLE_MOMENTS == (1 << 4),
               "dipole publication result flag must remain at bit 4");
_Static_assert(sizeof(xtbloom_request_state_t) == sizeof(int32_t),
               "request state tag must remain 32-bit");
_Static_assert(XTBLOOM_REQUEST_INFO_V1_SIZE == 24,
               "request-info ABI-v1 image must remain 24 bytes");
_Static_assert(sizeof(xtbloom_request_info_t) == XTBLOOM_REQUEST_INFO_V1_SIZE,
               "request-info public layout must end at ABI v1");
_Static_assert(offsetof(xtbloom_request_info_t, state) == 8,
               "request-info state offset must remain stable");
_Static_assert(offsetof(xtbloom_request_info_t, completion_status) == 12,
               "request-info completion status offset must remain stable");
_Static_assert(offsetof(xtbloom_request_info_t, result_flags) == 16,
               "request-info result flags offset must remain stable");

static int check_compute_options_init(size_t caller_size) {
  enum { CANARY_BYTES = 16 };
  unsigned char* storage = (unsigned char*)malloc(caller_size + CANARY_BYTES);
  if (storage == NULL) {
    return 0;
  }
  memset(storage, 0xa5, caller_size + CANARY_BYTES);

  xtbloom_compute_options_t* options = (xtbloom_compute_options_t*)storage;
  const xtbloom_status_t status = xtbloom_compute_options_init(options, caller_size);
  const int prefix_ok = status == XTBLOOM_STATUS_SUCCESS && options->struct_size == caller_size &&
                        options->api_version == XTBLOOM_API_VERSION &&
                        options->model == XTBLOOM_MODEL_GFN2_XTB &&
                        options->max_scc_iterations == 250 &&
                        options->electronic_temperature == XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE;

  int suffix_ok = 1;
  if (caller_size >= XTBLOOM_COMPUTE_OPTIONS_V2_SIZE) {
    suffix_ok = options->scc_start_mode == XTBLOOM_SCC_START_FRESH && options->reserved_v2 == 0u;
  }
  if (caller_size >= XTBLOOM_COMPUTE_OPTIONS_V3_SIZE) {
    suffix_ok = suffix_ok && options->scc_mixer == XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN &&
                options->scc_mixer_history == 8 && options->scc_mixer_damping == 0.4 &&
                options->determinism == XTBLOOM_DETERMINISM_DEFAULT && options->reserved_v3 == 0u;
    for (size_t index = XTBLOOM_COMPUTE_OPTIONS_V3_SIZE; index < caller_size; ++index) {
      if (storage[index] != 0xa5) suffix_ok = 0;
    }
  } else {
    const size_t initialized_prefix = caller_size >= XTBLOOM_COMPUTE_OPTIONS_V2_SIZE
                                          ? XTBLOOM_COMPUTE_OPTIONS_V2_SIZE
                                          : XTBLOOM_COMPUTE_OPTIONS_V1_SIZE;
    for (size_t index = initialized_prefix; index < caller_size; ++index) {
      if (storage[index] != 0) suffix_ok = 0;
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
  return prefix_ok && suffix_ok && canary_ok;
}

static int check_short_batch_init(size_t caller_size) {
  enum { CANARY_BYTES = 16 };
  unsigned char* storage = (unsigned char*)malloc(caller_size + CANARY_BYTES);
  if (storage == NULL) {
    return 0;
  }
  memset(storage, 0xa5, caller_size + CANARY_BYTES);

  xtbloom_batch_t* batch = (xtbloom_batch_t*)storage;
  const xtbloom_status_t status = xtbloom_batch_init(batch, caller_size);
  int ok = status == XTBLOOM_STATUS_SUCCESS && batch->struct_size == caller_size &&
           batch->api_version == XTBLOOM_API_VERSION;
  for (size_t index = XTBLOOM_BATCH_V1_SIZE; ok && index < caller_size; ++index) {
    ok = storage[index] == 0;
  }
  for (size_t index = caller_size; ok && index < caller_size + CANARY_BYTES; ++index) {
    ok = storage[index] == 0xa5;
  }
  free(storage);
  return ok;
}

static int check_future_batch_init(void) {
  struct future_batch {
    xtbloom_batch_t batch;
    unsigned char suffix[16];
  } value;
  memset(&value, 0xa5, sizeof(value));
  if (xtbloom_batch_init(&value.batch, sizeof(value)) != XTBLOOM_STATUS_SUCCESS ||
      value.batch.struct_size != sizeof(value) ||
      value.batch.api_version != XTBLOOM_API_VERSION) {
    return 0;
  }
  for (size_t index = 0; index < sizeof(value.suffix); ++index) {
    if (value.suffix[index] != 0xa5) {
      return 0;
    }
  }
  return 1;
}

static int check_short_batch_result_init(size_t caller_size) {
  enum { CANARY_BYTES = 16 };
  unsigned char* storage = (unsigned char*)malloc(caller_size + CANARY_BYTES);
  if (storage == NULL) {
    return 0;
  }
  memset(storage, 0xa5, caller_size + CANARY_BYTES);

  xtbloom_batch_result_t* result = (xtbloom_batch_result_t*)storage;
  const xtbloom_status_t status = xtbloom_batch_result_init(result, caller_size);
  int ok = status == XTBLOOM_STATUS_SUCCESS && result->struct_size == caller_size &&
           result->api_version == XTBLOOM_API_VERSION;
  for (size_t index = XTBLOOM_BATCH_RESULT_V1_SIZE; ok && index < caller_size; ++index) {
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
  xtbloom_workspace_query_t query;
  if (xtbloom_workspace_query_init(&query, sizeof(query)) != XTBLOOM_STATUS_SUCCESS) {
    return 0;
  }
  if (query.struct_size != XTBLOOM_WORKSPACE_QUERY_V1_SIZE ||
      query.api_version != XTBLOOM_API_VERSION || query.compute_flags != 0 ||
      query.host_required_bytes != 0u || query.host_required_alignment != 0u ||
      query.device_required_bytes != 0u || query.device_required_alignment != 0u ||
      query.reserved != 0u || query.reserved_v2 != 0u) {
    return 0;
  }
  if (xtbloom_workspace_query_init(&query, XTBLOOM_WORKSPACE_QUERY_V1_SIZE - 1) !=
      XTBLOOM_STATUS_INVALID_ARGUMENT) {
    return 0;
  }
  return 1;
}

/* Exercise the ABI-v4 batch and ABI-v2 result initializers: full-size images
 * must zero the interaction/lattice suffixes and the result outlets, while a short
 * ABI-v1 prefix must still initialize without touching caller bytes it does
 * not own. */
static int check_batch_result_suffix_init(void) {
  xtbloom_batch_t batch;
  if (xtbloom_batch_init(&batch, sizeof(batch)) != XTBLOOM_STATUS_SUCCESS) {
    return 0;
  }
  if (batch.struct_size != XTBLOOM_BATCH_V4_SIZE || batch.api_version != XTBLOOM_API_VERSION ||
      batch.total_interactions != 0 || batch.interaction_descriptors.data != NULL ||
      batch.interaction_payload.data != NULL || batch.cell_matrices.data != NULL ||
      batch.periodic_axes.data != NULL) {
    return 0;
  }
  if (!check_short_batch_init(XTBLOOM_BATCH_V1_SIZE) ||
      !check_short_batch_init(XTBLOOM_BATCH_V2_SIZE) ||
      !check_short_batch_init(XTBLOOM_BATCH_V3_SIZE) ||
      !check_short_batch_init(XTBLOOM_BATCH_V4_SIZE - 1) || !check_future_batch_init()) {
    return 0;
  }

  xtbloom_batch_result_t result;
  if (xtbloom_batch_result_init(&result, sizeof(result)) != XTBLOOM_STATUS_SUCCESS) {
    return 0;
  }
  if (result.struct_size != XTBLOOM_BATCH_RESULT_V2_SIZE ||
      result.api_version != XTBLOOM_API_VERSION || result.dipole_moments.data != NULL ||
      result.quadrupole_moments.data != NULL || result.wiberg_orders.data != NULL ||
      result.spin_populations.data != NULL) {
    return 0;
  }
  if (!check_short_batch_result_init(XTBLOOM_BATCH_RESULT_V1_SIZE) ||
      !check_short_batch_result_init(XTBLOOM_BATCH_RESULT_V2_SIZE - 1)) {
    return 0;
  }
  return 1;
}

static xtbloom_const_buffer_t input_view(const void* data, size_t size_bytes) {
  xtbloom_const_buffer_t buffer = {data, size_bytes, XTBLOOM_MEMORY_HOST, 0};
  return buffer;
}

static xtbloom_buffer_t output_view(void* data, size_t size_bytes) {
  xtbloom_buffer_t buffer = {data, size_bytes, XTBLOOM_MEMORY_HOST, 0};
  return buffer;
}

/* Prove that a valid native 3D request is rejected before caller publication.
 * No eigensolver provider is needed because the availability gate precedes
 * model setup and execution. */
static int check_periodic_refusal_is_atomic(xtbloom_context_t* context) {
  const int64_t atom_offsets[2] = {0, 1};
  const int32_t atomic_numbers[1] = {1};
  const double positions[3] = {0.0, 0.0, 0.0};
  const double molecular_charges[1] = {0.0};
  const int32_t unpaired_electrons[1] = {0};
  const double cell[9] = {4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0};
  const xtbloom_periodic_axes_t axes[1] = {XTBLOOM_PERIODIC_AXES_XYZ};

  double energy = 123.0;
  double forces[3] = {11.0, 12.0, 13.0};
  int32_t iterations = 17;
  uint8_t converged = 9;
  xtbloom_status_t system_status = XTBLOOM_STATUS_INTERNAL_ERROR;

  xtbloom_batch_t batch;
  xtbloom_compute_options_t compute;
  xtbloom_batch_result_t result;
  if (xtbloom_batch_init(&batch, sizeof(batch)) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom_compute_options_init(&compute, sizeof(compute)) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom_batch_result_init(&result, sizeof(result)) != XTBLOOM_STATUS_SUCCESS) {
    return 0;
  }
  batch.batch_size = 1;
  batch.total_atoms = 1;
  batch.atom_offsets = input_view(atom_offsets, sizeof(atom_offsets));
  batch.atomic_numbers = input_view(atomic_numbers, sizeof(atomic_numbers));
  batch.positions = input_view(positions, sizeof(positions));
  batch.molecular_charges = input_view(molecular_charges, sizeof(molecular_charges));
  batch.unpaired_electrons = input_view(unpaired_electrons, sizeof(unpaired_electrons));
  batch.cell_matrices = input_view(cell, sizeof(cell));
  batch.periodic_axes = input_view(axes, sizeof(axes));
  result.flags = 0x5a5a5a5au;
  result.energies = output_view(&energy, sizeof(energy));
  result.forces = output_view(forces, sizeof(forces));
  result.scc_iterations = output_view(&iterations, sizeof(iterations));
  result.scc_converged = output_view(&converged, sizeof(converged));
  result.per_system_status = output_view(&system_status, sizeof(system_status));

  const xtbloom_status_t status = xtbloom_compute(context, &batch, &compute, &result);
  return status == XTBLOOM_STATUS_NOT_IMPLEMENTED && result.flags == 0x5a5a5a5au &&
         energy == 123.0 && forces[0] == 11.0 && forces[1] == 12.0 && forces[2] == 13.0 &&
         iterations == 17 && converged == 9 && system_status == XTBLOOM_STATUS_INTERNAL_ERROR;
}

static int check_request_info_init(void) {
  struct extended_request_info {
    xtbloom_request_info_t info;
    unsigned char future[16];
  } extended;
  memset(&extended, 0xa5, sizeof(extended));
  if (xtbloom_request_info_init(&extended.info, sizeof(extended)) != XTBLOOM_STATUS_SUCCESS) {
    return 0;
  }
  if (extended.info.struct_size != sizeof(extended) ||
      extended.info.api_version != XTBLOOM_API_VERSION ||
      extended.info.state != XTBLOOM_REQUEST_IDLE ||
      extended.info.completion_status != XTBLOOM_STATUS_SUCCESS ||
      extended.info.result_flags != 0u || extended.info.reserved != 0u) {
    return 0;
  }
  for (size_t index = 0; index < sizeof(extended.future); ++index) {
    if (extended.future[index] != 0xa5) {
      return 0;
    }
  }

  xtbloom_request_info_t short_info;
  xtbloom_request_info_t short_before;
  memset(&short_info, 0xa5, sizeof(short_info));
  memcpy(&short_before, &short_info, sizeof(short_info));
  if (xtbloom_request_info_init(&short_info, XTBLOOM_REQUEST_INFO_V1_SIZE - 1) !=
          XTBLOOM_STATUS_INVALID_ARGUMENT ||
      memcmp(&short_info, &short_before, sizeof(short_info)) != 0 ||
      xtbloom_request_info_init(NULL, sizeof(short_info)) != XTBLOOM_STATUS_INVALID_ARGUMENT) {
    return 0;
  }
  return 1;
}

static int check_cpu_request_shell(xtbloom_context_t* context) {
  xtbloom_request_t* request = NULL;
  if (xtbloom_request_create(context, &request) != XTBLOOM_STATUS_SUCCESS || request == NULL) {
    return 0;
  }

  xtbloom_request_info_t info;
  if (xtbloom_request_info_init(&info, sizeof(info)) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom_request_query(request, &info) != XTBLOOM_STATUS_SUCCESS ||
      info.state != XTBLOOM_REQUEST_IDLE || info.completion_status != XTBLOOM_STATUS_SUCCESS ||
      info.result_flags != 0u || strcmp(xtbloom_request_get_error(request), "") != 0) {
    xtbloom_request_destroy(request);
    return 0;
  }

  struct query_future_info {
    xtbloom_request_info_t info;
    unsigned char future[16];
  } future_info;
  memset(&future_info, 0xa5, sizeof(future_info));
  if (xtbloom_request_info_init(&future_info.info, sizeof(future_info)) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom_request_query(request, &future_info.info) != XTBLOOM_STATUS_SUCCESS) {
    xtbloom_request_destroy(request);
    return 0;
  }
  for (size_t index = 0; index < sizeof(future_info.future); ++index) {
    if (future_info.future[index] != 0xa5) {
      xtbloom_request_destroy(request);
      return 0;
    }
  }

  xtbloom_request_info_t one_byte_short = info;
  one_byte_short.struct_size = XTBLOOM_REQUEST_INFO_V1_SIZE - 1u;
  xtbloom_request_info_t one_byte_short_before = one_byte_short;
  if (xtbloom_request_query(request, &one_byte_short) != XTBLOOM_STATUS_INVALID_ARGUMENT ||
      memcmp(&one_byte_short, &one_byte_short_before, sizeof(one_byte_short)) != 0) {
    xtbloom_request_destroy(request);
    return 0;
  }

  /* CPU capability probing happens before descriptor validation. Even hostile
   * NULL descriptors therefore leave both caller outputs and request state
   * untouched, allowing a synchronous fallback without state repair. */
  xtbloom_batch_result_t result;
  memset(&result, 0, sizeof(result));
  result.flags = 0x5a5a5a5au;
  double energy = 123.5;
  result.energies.data = &energy;
  result.energies.size_bytes = sizeof(energy);
  result.energies.memory_space = XTBLOOM_MEMORY_HOST;
  if (xtbloom_compute_enqueue(context, NULL, NULL, &result, request) !=
          XTBLOOM_STATUS_NOT_SUPPORTED ||
      result.flags != 0x5a5a5a5au || energy != 123.5) {
    xtbloom_request_destroy(request);
    return 0;
  }
  /* Repeated unsupported probes exercise reuse without creating a hidden
   * PENDING transition or retaining the first call's descriptors. */
  if (xtbloom_compute_enqueue(context, NULL, NULL, NULL, request) != XTBLOOM_STATUS_NOT_SUPPORTED) {
    xtbloom_request_destroy(request);
    return 0;
  }
  if (xtbloom_request_wait(request, &info) != XTBLOOM_STATUS_SUCCESS ||
      info.state != XTBLOOM_REQUEST_IDLE || info.completion_status != XTBLOOM_STATUS_SUCCESS ||
      info.result_flags != 0u || strcmp(xtbloom_request_get_error(request), "") != 0) {
    xtbloom_request_destroy(request);
    return 0;
  }

  xtbloom_request_info_t invalid_info = info;
  invalid_info.api_version = XTBLOOM_API_VERSION + 1u;
  invalid_info.state = (xtbloom_request_state_t)77;
  if (xtbloom_request_query(request, &invalid_info) != XTBLOOM_STATUS_INVALID_ARGUMENT ||
      invalid_info.state != (xtbloom_request_state_t)77) {
    xtbloom_request_destroy(request);
    return 0;
  }
  invalid_info = info;
  invalid_info.reserved = 1u;
  invalid_info.result_flags = 0x11223344u;
  if (xtbloom_request_wait(request, &invalid_info) != XTBLOOM_STATUS_INVALID_ARGUMENT ||
      invalid_info.result_flags != 0x11223344u) {
    xtbloom_request_destroy(request);
    return 0;
  }

  xtbloom_request_destroy(request);
  xtbloom_request_destroy(NULL);
  return 1;
}

int main(void) {
  if (sizeof(xtbloom_status_t) != sizeof(int32_t) || sizeof(xtbloom_backend_t) != sizeof(int32_t) ||
      sizeof(xtbloom_memory_space_t) != sizeof(int32_t) ||
      sizeof(xtbloom_model_t) != sizeof(int32_t) ||
      sizeof(xtbloom_scc_start_mode_t) != sizeof(int32_t) ||
      sizeof(xtbloom_scc_mixer_t) != sizeof(int32_t) ||
      sizeof(xtbloom_determinism_t) != sizeof(int32_t) ||
      sizeof(xtbloom_compute_flag_t) != sizeof(int32_t) ||
      sizeof(xtbloom_result_flag_t) != sizeof(int32_t) ||
      sizeof(xtbloom_request_state_t) != sizeof(int32_t) ||
      sizeof(xtbloom_interaction_type_t) != sizeof(int32_t) ||
      sizeof(xtbloom_periodic_axes_t) != sizeof(int32_t)) {
    fprintf(stderr, "public ABI tags and flags must all be 32-bit\n");
    return 1;
  }

  xtbloom_compute_options_t compute_options;
  if (xtbloom_compute_options_init(&compute_options, sizeof(compute_options)) !=
          XTBLOOM_STATUS_SUCCESS ||
      compute_options.struct_size != XTBLOOM_COMPUTE_OPTIONS_V3_SIZE ||
      compute_options.electronic_temperature != XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE ||
      compute_options.scc_start_mode != XTBLOOM_SCC_START_FRESH ||
      compute_options.reserved_v2 != 0 ||
      compute_options.scc_mixer != XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN ||
      compute_options.scc_mixer_history != 8 || compute_options.scc_mixer_damping != 0.4 ||
      compute_options.determinism != XTBLOOM_DETERMINISM_DEFAULT ||
      compute_options.reserved_v3 != 0) {
    fprintf(stderr, "ABI-v3 compute-options defaults are incorrect\n");
    return 2;
  }
  const size_t compute_option_sizes[] = {XTBLOOM_COMPUTE_OPTIONS_V1_SIZE,
                                         XTBLOOM_COMPUTE_OPTIONS_V2_SIZE - 1,
                                         XTBLOOM_COMPUTE_OPTIONS_V2_SIZE,
                                         XTBLOOM_COMPUTE_OPTIONS_V2_SIZE + 1,
                                         63u,
                                         64u,
                                         71u,
                                         72u,
                                         XTBLOOM_COMPUTE_OPTIONS_V3_SIZE - 1,
                                         XTBLOOM_COMPUTE_OPTIONS_V3_SIZE,
                                         XTBLOOM_COMPUTE_OPTIONS_V3_SIZE + 16};
  for (size_t index = 0; index < sizeof(compute_option_sizes) / sizeof(compute_option_sizes[0]);
       ++index) {
    if (!check_compute_options_init(compute_option_sizes[index])) {
      fprintf(stderr, "compute-options initialization violated a suffix boundary\n");
      return 3;
    }
  }
  if (xtbloom_compute_options_init(&compute_options, XTBLOOM_COMPUTE_OPTIONS_V1_SIZE - 1) !=
      XTBLOOM_STATUS_INVALID_ARGUMENT) {
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

  xtbloom_context_options_t options;
  if (xtbloom_context_options_init(&options, sizeof(options)) != XTBLOOM_STATUS_SUCCESS) {
    fprintf(stderr, "options initialization failed: %s\n", xtbloom_get_last_error());
    return 4;
  }
  options.backend = XTBLOOM_BACKEND_CPU;

  xtbloom_context_t* context = NULL;
  xtbloom_status_t status = xtbloom_context_create(&options, &context);
  if (status != XTBLOOM_STATUS_SUCCESS || context == NULL) {
    fprintf(stderr, "context creation failed: %s\n", xtbloom_get_last_error());
    return 5;
  }
  if (xtbloom_context_get_backend(context) != XTBLOOM_BACKEND_CPU) {
    fprintf(stderr, "explicit CPU backend was not selected\n");
    return 6;
  }
  if (strcmp(xtbloom_version_string(), XTBLOOM_VERSION_STRING) != 0) {
    fprintf(stderr, "unexpected version string\n");
    return 7;
  }
  if (strcmp(xtbloom_status_string(XTBLOOM_STATUS_SCC_NOT_CONVERGED), "SCC not converged") != 0 ||
      strcmp(xtbloom_status_string(XTBLOOM_STATUS_EIGENSOLVER_FAILED), "eigensolver failed") != 0) {
    fprintf(stderr, "per-system failure status strings are not stable\n");
    return 8;
  }
  if (!check_cpu_request_shell(context)) {
    fprintf(stderr, "CPU request ABI shell behavior is incorrect: %s\n", xtbloom_get_last_error());
    return 9;
  }
  if (!check_periodic_refusal_is_atomic(context)) {
    fprintf(stderr, "native periodic refusal modified caller output: %s\n",
            xtbloom_get_last_error());
    return 10;
  }

  xtbloom_context_destroy(context);
  return 0;
}
