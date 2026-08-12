#include <math.h>
#include <stdio.h>
#include <string.h>

#include "xtbloom/xtbloom.h"

/* Minimal DLPack 1.0 layout mirror used only to consume the managed tensor
 * exported by the installed-library ABI.  The native producer owns this
 * object until the importing consumer invokes its deleter. */
typedef struct consumer_dl_tensor {
  void* data;
  struct {
    int32_t device_type;
    int32_t device_id;
  } device;
  int32_t ndim;
  struct {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
  } dtype;
  int64_t* shape;
  int64_t* strides;
  uint64_t byte_offset;
} consumer_dl_tensor_t;

typedef struct consumer_dl_managed_tensor_versioned consumer_dl_managed_tensor_versioned_t;
typedef void (*consumer_dlpack_deleter_t)(consumer_dl_managed_tensor_versioned_t*);
struct consumer_dl_managed_tensor_versioned {
  uint32_t version_major;
  uint32_t version_minor;
  void* manager_ctx;
  consumer_dlpack_deleter_t deleter;
  uint64_t flags;
  consumer_dl_tensor_t dl_tensor;
};

_Static_assert(sizeof(consumer_dl_tensor_t) == 48,
               "installed DLPack tensor mirror must remain 48 bytes");
_Static_assert(sizeof(consumer_dl_managed_tensor_versioned_t) == 80,
               "installed DLPack managed-tensor mirror must remain 80 bytes");
_Static_assert(XTBLOOM_RESULT_OWNER_OPTIONS_V1_SIZE <= sizeof(xtbloom_result_owner_options_t),
               "result-owner options prefix must fit the public layout");
_Static_assert(XTBLOOM_DLPACK_VIEW_V1_SIZE == sizeof(xtbloom_dlpack_view_t),
               "DLPack view v1 prefix must cover the complete public layout");

_Static_assert(XTBLOOM_COMPUTE_OPTIONS_V1_SIZE == 48,
               "installed ABI-v1 compute-options prefix must remain 48 bytes");
_Static_assert(XTBLOOM_COMPUTE_OPTIONS_V2_SIZE == 56,
               "installed ABI-v2 compute-options prefix must remain 56 bytes");
_Static_assert(offsetof(xtbloom_compute_options_t, scc_start_mode) == 48,
               "installed ABI-v2 start mode offset must remain stable");
_Static_assert(sizeof(xtbloom_compute_options_t) == XTBLOOM_COMPUTE_OPTIONS_V2_SIZE,
               "installed compute-options layout must include the ABI-v2 suffix");

_Static_assert(XTBLOOM_BATCH_V1_SIZE == 328, "installed ABI-v1 batch prefix must remain 328 bytes");
_Static_assert(XTBLOOM_BATCH_V2_SIZE == 352, "installed ABI-v2 batch prefix must remain 352 bytes");
_Static_assert(XTBLOOM_BATCH_V3_SIZE == 408, "installed ABI-v3 batch image must remain 408 bytes");
_Static_assert(sizeof(xtbloom_batch_t) == XTBLOOM_BATCH_V3_SIZE,
               "installed batch layout must include the ABI-v3 interaction suffix");
_Static_assert(XTBLOOM_BATCH_RESULT_V1_SIZE == 184,
               "installed ABI-v1 batch-result prefix must remain 184 bytes");
_Static_assert(XTBLOOM_BATCH_RESULT_V2_SIZE == 280,
               "installed ABI-v2 batch-result image must remain 280 bytes");
_Static_assert(sizeof(xtbloom_batch_result_t) == XTBLOOM_BATCH_RESULT_V2_SIZE,
               "installed batch-result layout must include the ABI-v2 suffix");
_Static_assert(sizeof(xtbloom_interaction_t) == XTBLOOM_INTERACTION_V1_SIZE,
               "installed interaction descriptor image must remain 32 bytes");
_Static_assert(offsetof(xtbloom_batch_t, interaction_descriptors) == 360,
               "installed interaction descriptor outlet offset must remain stable");
_Static_assert(offsetof(xtbloom_interaction_t, payload_offset) == 16,
               "installed interaction payload offset must remain stable");
_Static_assert(offsetof(xtbloom_batch_result_t, spin_populations) == 256,
               "installed reserved result outlet offsets must remain stable");
_Static_assert(XTBLOOM_RESULT_DIPOLE_MOMENTS == (1 << 4),
               "installed dipole publication result flag must remain at bit 4");
_Static_assert(sizeof(xtbloom_request_state_t) == sizeof(int32_t),
               "installed request state tag must remain 32-bit");
_Static_assert(XTBLOOM_REQUEST_INFO_V1_SIZE == 24,
               "installed request-info ABI-v1 image must remain 24 bytes");
_Static_assert(sizeof(xtbloom_request_info_t) == XTBLOOM_REQUEST_INFO_V1_SIZE,
               "installed request-info layout must end at ABI v1");
_Static_assert(offsetof(xtbloom_request_info_t, completion_status) == 12,
               "installed request completion status offset must remain stable");

/* Prove that an external C consumer can own and inspect the additive request
 * ABI without any CUDA headers. CPU enqueue is an explicit capability probe:
 * it returns NOT_SUPPORTED before touching descriptors or request state. */
static int run_installed_request_shell(xtbloom_context_t* context) {
  xtbloom_request_info_t info;
  if (xtbloom_request_info_init(&info, sizeof(info)) != XTBLOOM_STATUS_SUCCESS) {
    return 30;
  }
  xtbloom_request_t* request = NULL;
  if (xtbloom_request_create(context, &request) != XTBLOOM_STATUS_SUCCESS || request == NULL) {
    return 31;
  }
  if (xtbloom_request_query(request, &info) != XTBLOOM_STATUS_SUCCESS ||
      info.state != XTBLOOM_REQUEST_IDLE || info.completion_status != XTBLOOM_STATUS_SUCCESS ||
      info.result_flags != 0u || strcmp(xtbloom_request_get_error(request), "") != 0) {
    xtbloom_request_destroy(request);
    return 32;
  }
  if (xtbloom_context_get_backend(context) == XTBLOOM_BACKEND_CPU &&
      xtbloom_compute_enqueue(context, NULL, NULL, NULL, request) != XTBLOOM_STATUS_NOT_SUPPORTED) {
    xtbloom_request_destroy(request);
    return 33;
  }
  if (xtbloom_request_wait(request, &info) != XTBLOOM_STATUS_SUCCESS ||
      info.state != XTBLOOM_REQUEST_IDLE) {
    xtbloom_request_destroy(request);
    return 34;
  }
  xtbloom_request_destroy(request);
  return 0;
}

typedef enum consumer_mode {
  CONSUMER_MODE_SMOKE,
  CONSUMER_MODE_CPU,
  CONSUMER_MODE_CUDA
} consumer_mode_t;

/* Exercise the installed xtbloom-owned result arena and DLPack export ABI on a
 * host arena (works on every backend without a CUDA compiler in this
 * consumer, and proves the additive #214 symbols are exported and usable). */
static const int64_t result_shape[] = {4};
static int run_installed_result_owner(void) {
  xtbloom_result_owner_options_t options;
  if (xtbloom_result_owner_options_init(&options, sizeof(options)) != XTBLOOM_STATUS_SUCCESS) {
    fprintf(stderr, "installed result-owner options init failed: %s\n", xtbloom_get_last_error());
    return 20;
  }
  options.memory_space = XTBLOOM_MEMORY_HOST;
  options.device_id = -1;
  options.size_bytes = 128;

  xtbloom_result_owner_t* owner = NULL;
  if (xtbloom_result_owner_create(&options, &owner) != XTBLOOM_STATUS_SUCCESS || owner == NULL) {
    fprintf(stderr, "installed result-owner create failed: %s\n", xtbloom_get_last_error());
    return 21;
  }

  xtbloom_buffer_t arena;
  if (xtbloom_result_owner_buffer(owner, &arena) != XTBLOOM_STATUS_SUCCESS || arena.data == NULL ||
      arena.size_bytes != 128u || arena.memory_space != XTBLOOM_MEMORY_HOST) {
    fprintf(stderr, "installed result-owner buffer view is inconsistent\n");
    xtbloom_result_owner_release(owner);
    return 22;
  }

  xtbloom_dlpack_view_t view;
  memset(&view, 0, sizeof(view));
  view.struct_size = sizeof(view);
  view.api_version = XTBLOOM_API_VERSION;
  view.byte_offset = 0;
  view.dtype_code = 2;
  view.dtype_bits = 64;
  view.dtype_lanes = 1;
  view.ndim = 1;
  view.shape = result_shape;
  void* managed = NULL;
  if (xtbloom_result_owner_export_dltensor(owner, &view, 1, &managed) != XTBLOOM_STATUS_SUCCESS ||
      managed == NULL) {
    fprintf(stderr, "installed result-owner DLPack export failed: %s\n", xtbloom_get_last_error());
    xtbloom_result_owner_release(owner);
    return 23;
  }

  /* Release the producer reference, then consume the single-use DLPack
   * transfer.  The managed-tensor deleter drops the export reference and
   * frees both the copied descriptor and the arena exactly once. */
  xtbloom_result_owner_release(owner);
  consumer_dl_managed_tensor_versioned_t* tensor = (consumer_dl_managed_tensor_versioned_t*)managed;
  if (tensor->version_major != 1u || tensor->version_minor != 0u || tensor->deleter == NULL ||
      tensor->dl_tensor.device.device_type != 1 || tensor->dl_tensor.device.device_id != 0 ||
      tensor->dl_tensor.ndim != 1 || tensor->dl_tensor.shape[0] != result_shape[0]) {
    fprintf(stderr, "installed DLPack managed-tensor fields are inconsistent\n");
    if (tensor->deleter != NULL) {
      tensor->deleter(tensor);
    }
    return 24;
  }
  tensor->deleter(tensor);
  return 0;
}

static xtbloom_const_buffer_t input_buffer(const void* data, size_t size_bytes) {
  xtbloom_const_buffer_t buffer = {data, size_bytes, XTBLOOM_MEMORY_HOST, 0};
  return buffer;
}

static xtbloom_buffer_t output_buffer(void* data, size_t size_bytes) {
  xtbloom_buffer_t buffer = {data, size_bytes, XTBLOOM_MEMORY_HOST, 0};
  return buffer;
}

/* Exercise actual inference through the installed C ABI without requiring a
 * CUDA compiler in the external consumer: the CUDA backend stages these host
 * descriptors and publishes the host result through its public transaction. */
static int run_installed_inference(xtbloom_context_t* context, const char* mode_name) {
  const int64_t atom_offsets[] = {0, 2};
  const int32_t atomic_numbers[] = {1, 1};
  double positions[] = {-0.70, 0.0, 0.0, 0.70, 0.0, 0.0};
  const double molecular_charges[] = {0.0};
  const int32_t unpaired_electrons[] = {0};

  xtbloom_batch_t batch;
  xtbloom_compute_options_t options;
  xtbloom_batch_result_t result;
  if (xtbloom_batch_init(&batch, sizeof(batch)) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom_compute_options_init(&options, sizeof(options)) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom_batch_result_init(&result, sizeof(result)) != XTBLOOM_STATUS_SUCCESS) {
    fprintf(stderr, "installed %s inference descriptor initialization failed: %s\n", mode_name,
            xtbloom_get_last_error());
    return 10;
  }

  batch.batch_size = 1;
  batch.total_atoms = 2;
  batch.atom_offsets = input_buffer(atom_offsets, sizeof(atom_offsets));
  batch.atomic_numbers = input_buffer(atomic_numbers, sizeof(atomic_numbers));
  batch.positions = input_buffer(positions, sizeof(positions));
  batch.molecular_charges = input_buffer(molecular_charges, sizeof(molecular_charges));
  batch.unpaired_electrons = input_buffer(unpaired_electrons, sizeof(unpaired_electrons));

  options.flags = XTBLOOM_COMPUTE_ENERGY;

  double energy = NAN;
  int32_t iterations = -1;
  uint8_t converged = 0;
  xtbloom_status_t system_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  result.energies = output_buffer(&energy, sizeof(energy));
  result.scc_iterations = output_buffer(&iterations, sizeof(iterations));
  result.scc_converged = output_buffer(&converged, sizeof(converged));
  result.per_system_status = output_buffer(&system_status, sizeof(system_status));

  const xtbloom_status_t status = xtbloom_compute(context, &batch, &options, &result);
  if (status != XTBLOOM_STATUS_SUCCESS || system_status != XTBLOOM_STATUS_SUCCESS ||
      converged != 1 || iterations <= 0 || !isfinite(energy)) {
    fprintf(stderr,
            "installed %s inference failed: call=%d system=%d converged=%u iterations=%d "
            "energy=%.17g error=%s\n",
            mode_name, (int)status, (int)system_status, (unsigned int)converged, (int)iterations,
            energy, xtbloom_get_last_error());
    return 11;
  }

  if (xtbloom_context_get_backend(context) == XTBLOOM_BACKEND_CUDA) {
    const uint32_t result_flags_canary = UINT32_C(0xa55a39c6);
    xtbloom_request_t* request = NULL;
    xtbloom_request_info_t info;
    energy = NAN;
    iterations = -1;
    converged = 0;
    system_status = XTBLOOM_STATUS_INTERNAL_ERROR;
    result.flags = result_flags_canary;
    if (xtbloom_request_info_init(&info, sizeof(info)) != XTBLOOM_STATUS_SUCCESS ||
        xtbloom_request_create(context, &request) != XTBLOOM_STATUS_SUCCESS || request == NULL ||
        xtbloom_compute_enqueue(context, &batch, &options, &result, request) !=
            XTBLOOM_STATUS_SUCCESS ||
        xtbloom_request_wait(request, &info) != XTBLOOM_STATUS_SUCCESS ||
        info.state != XTBLOOM_REQUEST_COMPLETE ||
        info.completion_status != XTBLOOM_STATUS_SUCCESS || info.result_flags != 0u ||
        result.flags != result_flags_canary || system_status != XTBLOOM_STATUS_SUCCESS ||
        converged != 1 || iterations <= 0 || !isfinite(energy)) {
      fprintf(stderr, "installed %s context enqueue failed: call_error=%s request_error=%s\n",
              mode_name, xtbloom_get_last_error(),
              request == NULL ? "request is NULL" : xtbloom_request_get_error(request));
      xtbloom_request_destroy(request);
      return 18;
    }

    /* A successful async FRESH publishes the strict checkpoint consumed by
     * the next changed-geometry WARM request. Keep the C-only installed
     * consumer independent of CUDA headers while exercising the public ABI. */
    positions[0] -= 0.01;
    positions[3] += 0.0075;
    energy = NAN;
    iterations = -1;
    converged = 0;
    system_status = XTBLOOM_STATUS_INTERNAL_ERROR;
    options.scc_start_mode = XTBLOOM_SCC_START_WARM;
    if (xtbloom_compute_enqueue(context, &batch, &options, &result, request) !=
            XTBLOOM_STATUS_SUCCESS ||
        xtbloom_request_wait(request, &info) != XTBLOOM_STATUS_SUCCESS ||
        info.state != XTBLOOM_REQUEST_COMPLETE ||
        info.completion_status != XTBLOOM_STATUS_SUCCESS || info.result_flags != 0u ||
        result.flags != result_flags_canary || system_status != XTBLOOM_STATUS_SUCCESS ||
        converged != 1 || iterations <= 0 || !isfinite(energy)) {
      fprintf(stderr, "installed %s context async WARM failed: call_error=%s\n", mode_name,
              xtbloom_get_last_error());
      fprintf(stderr, "installed %s context async WARM request_error=%s\n", mode_name,
              xtbloom_request_get_error(request));
      xtbloom_request_destroy(request);
      return 25;
    }
    xtbloom_request_destroy(request);
    /* Plan creation below owns an independent cache. Reset the per-call
     * policy so its first compute establishes that cache with FRESH. */
    options.scc_start_mode = XTBLOOM_SCC_START_FRESH;
  }

  /* Exercise the CPU-released ABI-v3 electric-field attachment and ABI-v2
   * dipole-moment outlet end to end. CUDA deliberately returns NOT_IMPLEMENTED
   * for both until #237 P3, so its installed consumer retains the field-free
   * inference and plan probes below. The released field block is 32 bytes:
   * int32 block_version=1, int32 reserved=0, three doubles in atomic units. */
  if (xtbloom_context_get_backend(context) == XTBLOOM_BACKEND_CPU) {
    const uint32_t field_flags =
        XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES | XTBLOOM_COMPUTE_DIPOLE_MOMENTS;
    uint8_t payload[32];
    memset(payload, 0, sizeof(payload));
    {
      const int32_t version = 1;
      const double efield[3] = {0.001, 0.002, -0.0015};
      memcpy(payload, &version, sizeof(version));
      memcpy(payload + 8, efield, sizeof(efield));
    }
    xtbloom_interaction_t interaction;
    memset(&interaction, 0, sizeof(interaction));
    interaction.type = XTBLOOM_INTERACTION_ELECTRIC_FIELD;
    interaction.system_index = 0;
    interaction.payload_offset = 0;
    interaction.payload_size = sizeof(payload);
    double field_forces[6] = {NAN, NAN, NAN, NAN, NAN, NAN};
    double dipole[3] = {NAN, NAN, NAN};
    xtbloom_batch_t field_batch = batch;
    xtbloom_compute_options_t field_options = options;
    xtbloom_batch_result_t field_result = result;
    field_options.flags = field_flags;
    field_batch.total_interactions = 1;
    field_batch.interaction_descriptors = input_buffer(&interaction, sizeof(interaction));
    field_batch.interaction_payload = input_buffer(payload, sizeof(payload));
    field_result.forces = output_buffer(field_forces, sizeof(field_forces));
    field_result.dipole_moments = output_buffer(dipole, sizeof(dipole));
    if (xtbloom_compute(context, &field_batch, &field_options, &field_result) !=
        XTBLOOM_STATUS_SUCCESS) {
      fprintf(stderr, "installed %s electric-field inference failed: %s\n", mode_name,
              xtbloom_get_last_error());
      return 11;
    }
    if (!(field_result.flags & XTBLOOM_RESULT_DIPOLE_MOMENTS)) {
      fprintf(stderr, "installed %s dipole publication flag not set\n", mode_name);
      return 11;
    }
    for (int component = 0; component < 6; ++component) {
      if (!isfinite(field_forces[component])) {
        fprintf(stderr, "installed %s electric-field forces are not finite\n", mode_name);
        return 11;
      }
    }
    for (int component = 0; component < 3; ++component) {
      if (!isfinite(dipole[component])) {
        fprintf(stderr, "installed %s dipole moments are not finite\n", mode_name);
        return 11;
      }
    }
  }

  /* Exercise the installed fixed-topology plan and workspace-query ABI. */
  const xtbloom_backend_t backend = xtbloom_context_get_backend(context);
  xtbloom_plan_t* plan = NULL;
  if (xtbloom_plan_create(context, &batch, &options, &plan) != XTBLOOM_STATUS_SUCCESS ||
      plan == NULL) {
    fprintf(stderr, "installed %s plan creation failed: %s\n", mode_name, xtbloom_get_last_error());
    return 12;
  }
  if (backend == XTBLOOM_BACKEND_CPU) {
    xtbloom_request_t* request = NULL;
    if (xtbloom_request_create(context, &request) != XTBLOOM_STATUS_SUCCESS || request == NULL ||
        xtbloom_plan_compute_enqueue(plan, NULL, NULL, NULL, request) !=
            XTBLOOM_STATUS_NOT_SUPPORTED) {
      fprintf(stderr, "installed %s plan request probe failed: %s\n", mode_name,
              xtbloom_get_last_error());
      xtbloom_request_destroy(request);
      xtbloom_plan_destroy(plan);
      return 17;
    }
    xtbloom_request_destroy(request);
  }
  xtbloom_workspace_query_t workspace;
  if (xtbloom_workspace_query_init(&workspace, sizeof(workspace)) != XTBLOOM_STATUS_SUCCESS) {
    fprintf(stderr, "installed %s workspace-query init failed: %s\n", mode_name,
            xtbloom_get_last_error());
    xtbloom_plan_destroy(plan);
    return 13;
  }
  workspace.compute_flags = XTBLOOM_COMPUTE_ENERGY;
  if (xtbloom_plan_query_workspace(plan, &workspace) != XTBLOOM_STATUS_SUCCESS) {
    fprintf(stderr, "installed %s workspace query failed: %s\n", mode_name,
            xtbloom_get_last_error());
    xtbloom_plan_destroy(plan);
    return 14;
  }
  const int device_workspace_invalid =
      backend == XTBLOOM_BACKEND_CUDA
          ? workspace.device_required_bytes == 0u || workspace.device_required_alignment == 0u
          : workspace.device_required_bytes != 0u || workspace.device_required_alignment != 1u;
  if (workspace.host_required_bytes == 0u || workspace.host_required_alignment == 0u ||
      device_workspace_invalid) {
    fprintf(stderr, "installed %s workspace query returned inconsistent sizes\n", mode_name);
    xtbloom_plan_destroy(plan);
    return 15;
  }
  /* Repeated changed-geometry plan calls must succeed like the convenience path. */
  energy = NAN;
  iterations = -1;
  converged = 0;
  system_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  if (xtbloom_plan_compute(plan, &batch, &options, &result) != XTBLOOM_STATUS_SUCCESS ||
      system_status != XTBLOOM_STATUS_SUCCESS || converged != 1 || iterations <= 0 ||
      !isfinite(energy)) {
    fprintf(stderr, "installed %s plan inference failed: %s\n", mode_name,
            xtbloom_get_last_error());
    xtbloom_plan_destroy(plan);
    return 16;
  }
  xtbloom_plan_destroy(plan);
  return 0;
}

static int parse_mode(int argc, char** argv, consumer_mode_t* mode) {
  if (argc == 1 || (argc == 2 && strcmp(argv[1], "smoke") == 0)) {
    *mode = CONSUMER_MODE_SMOKE;
    return 1;
  }
  if (argc == 2 && strcmp(argv[1], "cpu") == 0) {
    *mode = CONSUMER_MODE_CPU;
    return 1;
  }
  if (argc == 2 && strcmp(argv[1], "cuda") == 0) {
    *mode = CONSUMER_MODE_CUDA;
    return 1;
  }
  fprintf(stderr, "usage: %s [smoke|cpu|cuda]\n", argv[0]);
  return 0;
}

int main(int argc, char** argv) {
  if (strcmp(xtbloom_version_string(), XTBLOOM_VERSION_STRING) != 0) {
    fprintf(stderr, "installed product-version header and C API disagree\n");
    return 2;
  }

  consumer_mode_t mode;
  if (!parse_mode(argc, argv, &mode)) {
    return 1;
  }

  xtbloom_compute_options_t compute_options;
  if (xtbloom_compute_options_init(&compute_options, sizeof(compute_options)) !=
          XTBLOOM_STATUS_SUCCESS ||
      compute_options.scc_start_mode != XTBLOOM_SCC_START_FRESH ||
      compute_options.reserved_v2 != 0) {
    fprintf(stderr, "installed compute-options ABI-v2 defaults are incorrect\n");
    return 1;
  }

  xtbloom_context_options_t options;
  if (xtbloom_context_options_init(&options, sizeof(options)) != XTBLOOM_STATUS_SUCCESS) {
    fprintf(stderr, "%s\n", xtbloom_get_last_error());
    return 2;
  }
  options.backend = mode == CONSUMER_MODE_CUDA ? XTBLOOM_BACKEND_CUDA : XTBLOOM_BACKEND_CPU;

  xtbloom_context_t* context = NULL;
  if (xtbloom_context_create(&options, &context) != XTBLOOM_STATUS_SUCCESS) {
    fprintf(stderr, "%s\n", xtbloom_get_last_error());
    return 3;
  }
  if (xtbloom_context_get_backend(context) != options.backend) {
    fprintf(stderr, "installed consumer selected an unexpected backend\n");
    xtbloom_context_destroy(context);
    return 4;
  }

  const int request_status = run_installed_request_shell(context);
  if (request_status != 0) {
    fprintf(stderr, "installed request ABI shell failed (%d): %s\n", request_status,
            xtbloom_get_last_error());
    xtbloom_context_destroy(context);
    return request_status;
  }

  int inference_status = 0;
  if (mode != CONSUMER_MODE_SMOKE) {
    inference_status =
        run_installed_inference(context, mode == CONSUMER_MODE_CUDA ? "CUDA" : "CPU");
  }
  xtbloom_context_destroy(context);
  if (inference_status != 0) {
    return inference_status;
  }
  return run_installed_result_owner();
}
