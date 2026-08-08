#include <math.h>
#include <stdio.h>
#include <string.h>

#include "gpuxtb/gpuxtb.h"

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
_Static_assert(GPUXTB_RESULT_OWNER_OPTIONS_V1_SIZE <= sizeof(gpuxtb_result_owner_options_t),
               "result-owner options prefix must fit the public layout");
_Static_assert(GPUXTB_DLPACK_VIEW_V1_SIZE == sizeof(gpuxtb_dlpack_view_t),
               "DLPack view v1 prefix must cover the complete public layout");

_Static_assert(GPUXTB_COMPUTE_OPTIONS_V1_SIZE == 48,
               "installed ABI-v1 compute-options prefix must remain 48 bytes");
_Static_assert(GPUXTB_COMPUTE_OPTIONS_V2_SIZE == 56,
               "installed ABI-v2 compute-options prefix must remain 56 bytes");
_Static_assert(offsetof(gpuxtb_compute_options_t, scc_start_mode) == 48,
               "installed ABI-v2 start mode offset must remain stable");
_Static_assert(sizeof(gpuxtb_compute_options_t) == GPUXTB_COMPUTE_OPTIONS_V2_SIZE,
               "installed compute-options layout must include the ABI-v2 suffix");

_Static_assert(GPUXTB_BATCH_V1_SIZE == 328, "installed ABI-v1 batch prefix must remain 328 bytes");
_Static_assert(GPUXTB_BATCH_V2_SIZE == 352, "installed ABI-v2 batch prefix must remain 352 bytes");
_Static_assert(GPUXTB_BATCH_V3_SIZE == 408, "installed ABI-v3 batch image must remain 408 bytes");
_Static_assert(sizeof(gpuxtb_batch_t) == GPUXTB_BATCH_V3_SIZE,
               "installed batch layout must include the ABI-v3 interaction suffix");
_Static_assert(GPUXTB_BATCH_RESULT_V1_SIZE == 184,
               "installed ABI-v1 batch-result prefix must remain 184 bytes");
_Static_assert(GPUXTB_BATCH_RESULT_V2_SIZE == 280,
               "installed ABI-v2 batch-result image must remain 280 bytes");
_Static_assert(sizeof(gpuxtb_batch_result_t) == GPUXTB_BATCH_RESULT_V2_SIZE,
               "installed batch-result layout must include the ABI-v2 suffix");
_Static_assert(sizeof(gpuxtb_interaction_t) == GPUXTB_INTERACTION_V1_SIZE,
               "installed interaction descriptor image must remain 32 bytes");
_Static_assert(offsetof(gpuxtb_batch_t, interaction_descriptors) == 360,
               "installed interaction descriptor outlet offset must remain stable");
_Static_assert(offsetof(gpuxtb_interaction_t, payload_offset) == 16,
               "installed interaction payload offset must remain stable");
_Static_assert(offsetof(gpuxtb_batch_result_t, spin_populations) == 256,
               "installed reserved result outlet offsets must remain stable");
_Static_assert(GPUXTB_RESULT_DIPOLE_MOMENTS == (1 << 4),
               "installed dipole publication result flag must remain at bit 4");

typedef enum consumer_mode {
  CONSUMER_MODE_SMOKE,
  CONSUMER_MODE_CPU,
  CONSUMER_MODE_CUDA
} consumer_mode_t;

/* Exercise the installed gpuxtb-owned result arena and DLPack export ABI on a
 * host arena (works on every backend without a CUDA compiler in this
 * consumer, and proves the additive #214 symbols are exported and usable). */
static const int64_t result_shape[] = {4};
static int run_installed_result_owner(void) {
  gpuxtb_result_owner_options_t options;
  if (gpuxtb_result_owner_options_init(&options, sizeof(options)) != GPUXTB_STATUS_SUCCESS) {
    fprintf(stderr, "installed result-owner options init failed: %s\n", gpuxtb_get_last_error());
    return 20;
  }
  options.memory_space = GPUXTB_MEMORY_HOST;
  options.device_id = -1;
  options.size_bytes = 128;

  gpuxtb_result_owner_t* owner = NULL;
  if (gpuxtb_result_owner_create(&options, &owner) != GPUXTB_STATUS_SUCCESS || owner == NULL) {
    fprintf(stderr, "installed result-owner create failed: %s\n", gpuxtb_get_last_error());
    return 21;
  }

  gpuxtb_buffer_t arena;
  if (gpuxtb_result_owner_buffer(owner, &arena) != GPUXTB_STATUS_SUCCESS || arena.data == NULL ||
      arena.size_bytes != 128u || arena.memory_space != GPUXTB_MEMORY_HOST) {
    fprintf(stderr, "installed result-owner buffer view is inconsistent\n");
    gpuxtb_result_owner_release(owner);
    return 22;
  }

  gpuxtb_dlpack_view_t view;
  memset(&view, 0, sizeof(view));
  view.struct_size = sizeof(view);
  view.api_version = GPUXTB_API_VERSION;
  view.byte_offset = 0;
  view.dtype_code = 2;
  view.dtype_bits = 64;
  view.dtype_lanes = 1;
  view.ndim = 1;
  view.shape = result_shape;
  void* managed = NULL;
  if (gpuxtb_result_owner_export_dltensor(owner, &view, 1, &managed) != GPUXTB_STATUS_SUCCESS ||
      managed == NULL) {
    fprintf(stderr, "installed result-owner DLPack export failed: %s\n", gpuxtb_get_last_error());
    gpuxtb_result_owner_release(owner);
    return 23;
  }

  /* Release the producer reference, then consume the single-use DLPack
   * transfer.  The managed-tensor deleter drops the export reference and
   * frees both the copied descriptor and the arena exactly once. */
  gpuxtb_result_owner_release(owner);
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

static gpuxtb_const_buffer_t input_buffer(const void* data, size_t size_bytes) {
  gpuxtb_const_buffer_t buffer = {data, size_bytes, GPUXTB_MEMORY_HOST, 0};
  return buffer;
}

static gpuxtb_buffer_t output_buffer(void* data, size_t size_bytes) {
  gpuxtb_buffer_t buffer = {data, size_bytes, GPUXTB_MEMORY_HOST, 0};
  return buffer;
}

/* Exercise actual inference through the installed C ABI without requiring a
 * CUDA compiler in the external consumer: the CUDA backend stages these host
 * descriptors and publishes the host result through its public transaction. */
static int run_installed_inference(gpuxtb_context_t* context, const char* mode_name) {
  const int64_t atom_offsets[] = {0, 2};
  const int32_t atomic_numbers[] = {1, 1};
  const double positions[] = {-0.70, 0.0, 0.0, 0.70, 0.0, 0.0};
  const double molecular_charges[] = {0.0};
  const int32_t unpaired_electrons[] = {0};

  gpuxtb_batch_t batch;
  gpuxtb_compute_options_t options;
  gpuxtb_batch_result_t result;
  if (gpuxtb_batch_init(&batch, sizeof(batch)) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb_compute_options_init(&options, sizeof(options)) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb_batch_result_init(&result, sizeof(result)) != GPUXTB_STATUS_SUCCESS) {
    fprintf(stderr, "installed %s inference descriptor initialization failed: %s\n", mode_name,
            gpuxtb_get_last_error());
    return 10;
  }

  batch.batch_size = 1;
  batch.total_atoms = 2;
  batch.atom_offsets = input_buffer(atom_offsets, sizeof(atom_offsets));
  batch.atomic_numbers = input_buffer(atomic_numbers, sizeof(atomic_numbers));
  batch.positions = input_buffer(positions, sizeof(positions));
  batch.molecular_charges = input_buffer(molecular_charges, sizeof(molecular_charges));
  batch.unpaired_electrons = input_buffer(unpaired_electrons, sizeof(unpaired_electrons));

  options.flags = GPUXTB_COMPUTE_ENERGY;

  double energy = NAN;
  int32_t iterations = -1;
  uint8_t converged = 0;
  gpuxtb_status_t system_status = GPUXTB_STATUS_INTERNAL_ERROR;
  result.energies = output_buffer(&energy, sizeof(energy));
  result.scc_iterations = output_buffer(&iterations, sizeof(iterations));
  result.scc_converged = output_buffer(&converged, sizeof(converged));
  result.per_system_status = output_buffer(&system_status, sizeof(system_status));

  const gpuxtb_status_t status = gpuxtb_compute(context, &batch, &options, &result);
  if (status != GPUXTB_STATUS_SUCCESS || system_status != GPUXTB_STATUS_SUCCESS || converged != 1 ||
      iterations <= 0 || !isfinite(energy)) {
    fprintf(stderr,
            "installed %s inference failed: call=%d system=%d converged=%u iterations=%d "
            "energy=%.17g error=%s\n",
            mode_name, (int)status, (int)system_status, (unsigned int)converged, (int)iterations,
            energy, gpuxtb_get_last_error());
    return 11;
  }

  /* Exercise the installed fixed-topology plan and workspace-query ABI. */
  gpuxtb_plan_t* plan = NULL;
  if (gpuxtb_plan_create(context, &batch, &options, &plan) != GPUXTB_STATUS_SUCCESS ||
      plan == NULL) {
    fprintf(stderr, "installed %s plan creation failed: %s\n", mode_name, gpuxtb_get_last_error());
    return 12;
  }
  gpuxtb_workspace_query_t workspace;
  if (gpuxtb_workspace_query_init(&workspace, sizeof(workspace)) != GPUXTB_STATUS_SUCCESS) {
    fprintf(stderr, "installed %s workspace-query init failed: %s\n", mode_name,
            gpuxtb_get_last_error());
    gpuxtb_plan_destroy(plan);
    return 13;
  }
  workspace.compute_flags = GPUXTB_COMPUTE_ENERGY;
  if (gpuxtb_plan_query_workspace(plan, &workspace) != GPUXTB_STATUS_SUCCESS) {
    fprintf(stderr, "installed %s workspace query failed: %s\n", mode_name,
            gpuxtb_get_last_error());
    gpuxtb_plan_destroy(plan);
    return 14;
  }
  const gpuxtb_backend_t backend = gpuxtb_context_get_backend(context);
  const int device_workspace_invalid =
      backend == GPUXTB_BACKEND_CUDA
          ? workspace.device_required_bytes == 0u || workspace.device_required_alignment == 0u
          : workspace.device_required_bytes != 0u || workspace.device_required_alignment != 1u;
  if (workspace.host_required_bytes == 0u || workspace.host_required_alignment == 0u ||
      device_workspace_invalid) {
    fprintf(stderr, "installed %s workspace query returned inconsistent sizes\n", mode_name);
    gpuxtb_plan_destroy(plan);
    return 15;
  }
  /* Repeated changed-geometry plan calls must succeed like the convenience path. */
  energy = NAN;
  iterations = -1;
  converged = 0;
  system_status = GPUXTB_STATUS_INTERNAL_ERROR;
  if (gpuxtb_plan_compute(plan, &batch, &options, &result) != GPUXTB_STATUS_SUCCESS ||
      system_status != GPUXTB_STATUS_SUCCESS || converged != 1 || iterations <= 0 ||
      !isfinite(energy)) {
    fprintf(stderr, "installed %s plan inference failed: %s\n", mode_name, gpuxtb_get_last_error());
    gpuxtb_plan_destroy(plan);
    return 16;
  }
  gpuxtb_plan_destroy(plan);
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
  consumer_mode_t mode;
  if (!parse_mode(argc, argv, &mode)) {
    return 1;
  }

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
  options.backend = mode == CONSUMER_MODE_CUDA ? GPUXTB_BACKEND_CUDA : GPUXTB_BACKEND_CPU;

  gpuxtb_context_t* context = NULL;
  if (gpuxtb_context_create(&options, &context) != GPUXTB_STATUS_SUCCESS) {
    fprintf(stderr, "%s\n", gpuxtb_get_last_error());
    return 3;
  }
  if (gpuxtb_context_get_backend(context) != options.backend) {
    fprintf(stderr, "installed consumer selected an unexpected backend\n");
    gpuxtb_context_destroy(context);
    return 4;
  }

  int inference_status = 0;
  if (mode != CONSUMER_MODE_SMOKE) {
    inference_status =
        run_installed_inference(context, mode == CONSUMER_MODE_CUDA ? "CUDA" : "CPU");
  }
  gpuxtb_context_destroy(context);
  if (inference_status != 0) {
    return inference_status;
  }
  return run_installed_result_owner();
}
