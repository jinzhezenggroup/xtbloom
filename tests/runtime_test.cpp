#include <cstdint>
#include <cstring>

#include "gpuxtb/gpuxtb.h"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

int main() {
  CHECK(std::strcmp(gpuxtb_status_string(GPUXTB_STATUS_SCC_NOT_CONVERGED), "SCC not converged") ==
        0);
  CHECK(std::strcmp(gpuxtb_status_string(GPUXTB_STATUS_EIGENSOLVER_FAILED), "eigensolver failed") ==
        0);

  gpuxtb_context_options_t options;
  CHECK(gpuxtb_context_options_init(&options, sizeof(options)) == GPUXTB_STATUS_SUCCESS);
  options.backend = GPUXTB_BACKEND_CPU;

  gpuxtb_context_t* context = nullptr;
  CHECK(gpuxtb_context_create(&options, &context) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb_context_get_device_id(context) == -1);

  gpuxtb_batch_t batch;
  gpuxtb_compute_options_t compute_options;
  gpuxtb_batch_result_t result;
  CHECK(gpuxtb_batch_init(&batch, sizeof(batch)) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb_compute_options_init(&compute_options, sizeof(compute_options)) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(compute_options.electronic_temperature == GPUXTB_DEFAULT_ELECTRONIC_TEMPERATURE);
  CHECK(gpuxtb_batch_result_init(&result, sizeof(result)) == GPUXTB_STATUS_SUCCESS);

  /* Descriptor errors take precedence over the unfinished physics backend. */
  const gpuxtb_status_t compute_status = gpuxtb_compute(context, &batch, &compute_options, &result);
  CHECK(compute_status == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::strstr(gpuxtb_get_last_error(), "batch_size") != nullptr);

  const std::int64_t atom_offsets[] = {0, 1};
  const std::int32_t atomic_numbers[] = {1};
  const double positions[] = {0.0, 0.0, 0.0};
  const double molecular_charges[] = {0.0};
  const std::int32_t unpaired_electrons[] = {0};
  double energies[1] = {};
  double forces[3] = {};
  std::int32_t scc_iterations[1] = {};
  std::uint8_t scc_converged[1] = {};
  std::int32_t per_system_status[1] = {};

  batch.batch_size = 1;
  batch.total_atoms = 1;
  batch.atom_offsets = {atom_offsets, sizeof(atom_offsets), GPUXTB_MEMORY_HOST, 0};
  batch.atomic_numbers = {atomic_numbers, sizeof(atomic_numbers), GPUXTB_MEMORY_HOST, 0};
  batch.positions = {positions, sizeof(positions), GPUXTB_MEMORY_HOST, 0};
  batch.molecular_charges = {molecular_charges, sizeof(molecular_charges), GPUXTB_MEMORY_HOST, 0};
  batch.unpaired_electrons = {unpaired_electrons, sizeof(unpaired_electrons), GPUXTB_MEMORY_HOST,
                              0};
  result.energies = {energies, sizeof(energies), GPUXTB_MEMORY_HOST, 0};
  result.forces = {forces, sizeof(forces), GPUXTB_MEMORY_HOST, 0};
  result.scc_iterations = {scc_iterations, sizeof(scc_iterations), GPUXTB_MEMORY_HOST, 0};
  result.scc_converged = {scc_converged, sizeof(scc_converged), GPUXTB_MEMORY_HOST, 0};
  result.per_system_status = {per_system_status, sizeof(per_system_status), GPUXTB_MEMORY_HOST, 0};

  const gpuxtb_status_t valid_compute_status =
      gpuxtb_compute(context, &batch, &compute_options, &result);
  CHECK(valid_compute_status == GPUXTB_STATUS_NOT_IMPLEMENTED);
  CHECK(std::strstr(gpuxtb_get_last_error(), "not been implemented") != nullptr);

  compute_options.model = GPUXTB_MODEL_GFN1_XTB;
  CHECK(gpuxtb_compute(context, &batch, &compute_options, &result) == GPUXTB_STATUS_NOT_SUPPORTED);
  CHECK(std::strstr(gpuxtb_get_last_error(), "GFN1-xTB") != nullptr);

  gpuxtb_context_destroy(context);

  gpuxtb_context_t* invalid_context = nullptr;
  options.device_id = -2;
  CHECK(gpuxtb_context_create(&options, &invalid_context) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(invalid_context == nullptr);

  options.device_id = -1;
  options.backend = static_cast<gpuxtb_backend_t>(99);
  CHECK(gpuxtb_context_create(&options, &invalid_context) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(invalid_context == nullptr);

  options.backend = GPUXTB_BACKEND_CPU;
  options.reserved = 1;
  CHECK(gpuxtb_context_create(&options, &invalid_context) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(invalid_context == nullptr);

  struct ExtendedOptions {
    gpuxtb_context_options_t options;
    std::uint64_t canary;
  } extended{};
  extended.canary = UINT64_C(0x5a5a5a5aa5a5a5a5);
  CHECK(gpuxtb_context_options_init(&extended.options, sizeof(extended)) == GPUXTB_STATUS_SUCCESS);
  CHECK(extended.options.struct_size == sizeof(extended));
  CHECK(extended.canary == UINT64_C(0x5a5a5a5aa5a5a5a5));

  CHECK(gpuxtb_context_options_init(&options, GPUXTB_CONTEXT_OPTIONS_V1_SIZE - 1) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);

#if defined(GPUXTB_TEST_HAS_CUDA)
  CHECK(gpuxtb_context_options_init(&options, sizeof(options)) == GPUXTB_STATUS_SUCCESS);
  options.backend = GPUXTB_BACKEND_CUDA;
  context = nullptr;
  const gpuxtb_status_t cuda_status = gpuxtb_context_create(&options, &context);
  if (cuda_status == GPUXTB_STATUS_SUCCESS) {
    CHECK(gpuxtb_context_get_backend(context) == GPUXTB_BACKEND_CUDA);
    CHECK(gpuxtb_context_get_device_id(context) >= 0);
    gpuxtb_context_destroy(context);
  } else {
    /* CUDA-enabled builds also run on hosts where the runtime exposes no device. */
    CHECK(cuda_status == GPUXTB_STATUS_BACKEND_UNAVAILABLE);
    CHECK(context == nullptr);
  }

  CHECK(gpuxtb_context_options_init(&options, sizeof(options)) == GPUXTB_STATUS_SUCCESS);
  options.backend = GPUXTB_BACKEND_AUTO;
  options.device_id = INT32_MAX;
  context = nullptr;
  CHECK(gpuxtb_context_create(&options, &context) == GPUXTB_STATUS_BACKEND_UNAVAILABLE);
  CHECK(context == nullptr);
#endif
  return 0;
}
