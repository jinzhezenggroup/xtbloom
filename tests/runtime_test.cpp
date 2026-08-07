#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>

#include "gpuxtb/gpuxtb.h"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

struct ContextDeleter {
  void operator()(gpuxtb_context_t* context) const noexcept { gpuxtb_context_destroy(context); }
};

using ContextHandle = std::unique_ptr<gpuxtb_context_t, ContextDeleter>;

ContextHandle create_context(const gpuxtb_context_options_t& options, gpuxtb_status_t& status) {
  gpuxtb_context_t* raw_context = nullptr;
  status = gpuxtb_context_create(&options, &raw_context);
  return ContextHandle(raw_context);
}

}  // namespace

int main() {
  CHECK(std::strcmp(gpuxtb_status_string(GPUXTB_STATUS_SCC_NOT_CONVERGED), "SCC not converged") ==
        0);
  CHECK(std::strcmp(gpuxtb_status_string(GPUXTB_STATUS_EIGENSOLVER_FAILED), "eigensolver failed") ==
        0);

  gpuxtb_context_options_t options;
  CHECK(gpuxtb_context_options_init(&options, sizeof(options)) == GPUXTB_STATUS_SUCCESS);
  options.backend = GPUXTB_BACKEND_CPU;

  gpuxtb_status_t context_status = GPUXTB_STATUS_INTERNAL_ERROR;
  ContextHandle context = create_context(options, context_status);
  CHECK(context_status == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb_context_get_device_id(context.get()) == -1);

  gpuxtb_batch_t batch;
  gpuxtb_compute_options_t compute_options;
  gpuxtb_batch_result_t result;
  CHECK(gpuxtb_batch_init(&batch, sizeof(batch)) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb_compute_options_init(&compute_options, sizeof(compute_options)) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(compute_options.electronic_temperature == GPUXTB_DEFAULT_ELECTRONIC_TEMPERATURE);
  CHECK(compute_options.struct_size == GPUXTB_COMPUTE_OPTIONS_V2_SIZE);
  CHECK(compute_options.scc_start_mode == GPUXTB_SCC_START_FRESH);
  CHECK(compute_options.reserved_v2 == 0u);
  CHECK(gpuxtb_batch_result_init(&result, sizeof(result)) == GPUXTB_STATUS_SUCCESS);

  /* Descriptor errors are reported before entering numerical execution. */
  const gpuxtb_status_t compute_status =
      gpuxtb_compute(context.get(), &batch, &compute_options, &result);
  CHECK(compute_status == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::strstr(gpuxtb_get_last_error(), "batch_size") != nullptr);

  const std::int64_t atom_offsets[] = {0, 1};
  /* Closed-shell helium exercises the real restricted CPU inference path. */
  const std::int32_t atomic_numbers[] = {2};
  const double positions[] = {0.0, 0.0, 0.0};
  const double molecular_charges[] = {0.0};
  const std::int32_t unpaired_electrons[] = {0};
  double energies[1] = {};
  double forces[3] = {};
  double atomic_charges[1] = {};
  double point_charge_forces[3] = {};
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
  result.atomic_charges = {atomic_charges, sizeof(atomic_charges), GPUXTB_MEMORY_HOST, 0};
  result.point_charge_forces = {point_charge_forces, sizeof(point_charge_forces),
                                GPUXTB_MEMORY_HOST, 0};
  result.scc_iterations = {scc_iterations, sizeof(scc_iterations), GPUXTB_MEMORY_HOST, 0};
  result.scc_converged = {scc_converged, sizeof(scc_converged), GPUXTB_MEMORY_HOST, 0};
  result.per_system_status = {per_system_status, sizeof(per_system_status), GPUXTB_MEMORY_HOST, 0};

  const gpuxtb_status_t valid_compute_status =
      gpuxtb_compute(context.get(), &batch, &compute_options, &result);
  if (valid_compute_status == GPUXTB_STATUS_BACKEND_UNAVAILABLE) {
    /* CPU-only CI configurations need not provide a production LP64 BLAS
     * runtime, but the diagnostic must identify that missing contract. */
    CHECK(std::strstr(gpuxtb_get_last_error(), "LP64") != nullptr);
  } else {
    CHECK(valid_compute_status == GPUXTB_STATUS_SUCCESS);
    CHECK(per_system_status[0] == GPUXTB_STATUS_SUCCESS);
    CHECK(scc_converged[0] == 1u);
    CHECK(scc_iterations[0] > 0);
    CHECK(std::isfinite(energies[0]));
    CHECK(std::isfinite(forces[0]));
    CHECK(std::isfinite(forces[1]));
    CHECK(std::isfinite(forces[2]));
  }

  /* ABI-v1 callers do not expose the suffix and therefore retain strict FRESH
   * behavior even if adjacent bytes contain invalid V2 values. */
  compute_options.struct_size = GPUXTB_COMPUTE_OPTIONS_V1_SIZE;
  compute_options.scc_start_mode = 0;
  compute_options.reserved_v2 = UINT32_MAX;
  const gpuxtb_status_t v1_compute_status =
      gpuxtb_compute(context.get(), &batch, &compute_options, &result);
  if (v1_compute_status == GPUXTB_STATUS_BACKEND_UNAVAILABLE) {
    CHECK(std::strstr(gpuxtb_get_last_error(), "LP64") != nullptr);
  } else {
    CHECK(v1_compute_status == GPUXTB_STATUS_SUCCESS);
  }

  compute_options.struct_size = GPUXTB_COMPUTE_OPTIONS_V2_SIZE;
  compute_options.scc_start_mode = GPUXTB_SCC_START_WARM;
  compute_options.reserved_v2 = 0u;
  energies[0] = 123.25;
  forces[0] = -4.0;
  forces[1] = -5.0;
  forces[2] = -6.0;
  atomic_charges[0] = 71.25;
  point_charge_forces[0] = 81.0;
  point_charge_forces[1] = 82.0;
  point_charge_forces[2] = 83.0;
  scc_iterations[0] = 91;
  scc_converged[0] = 1u;
  per_system_status[0] = GPUXTB_STATUS_INTERNAL_ERROR;
  result.flags = UINT32_C(0xa5a55a5a);
  const gpuxtb_status_t warm_compute_status =
      gpuxtb_compute(context.get(), &batch, &compute_options, &result);
  if (valid_compute_status == GPUXTB_STATUS_BACKEND_UNAVAILABLE) {
    /* No LP64 BLAS runtime means the preceding FRESH call never converged, so
     * the strict WARM identity precondition (a fully converged compatible
     * predecessor) is not met and the request is rejected before execution. */
    CHECK(warm_compute_status == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(std::strstr(gpuxtb_get_last_error(), "WARM") != nullptr);
    CHECK(energies[0] == 123.25);
    CHECK(forces[0] == -4.0 && forces[1] == -5.0 && forces[2] == -6.0);
    CHECK(atomic_charges[0] == 71.25);
    CHECK(point_charge_forces[0] == 81.0 && point_charge_forces[1] == 82.0 &&
          point_charge_forces[2] == 83.0);
    CHECK(scc_iterations[0] == 91);
    CHECK(scc_converged[0] == 1u);
    CHECK(per_system_status[0] == GPUXTB_STATUS_INTERNAL_ERROR);
    CHECK(result.flags == UINT32_C(0xa5a55a5a));
  } else {
    /* The FRESH call converged, so WARM consumes that converged electronic
     * checkpoint and reconverges (fewer iterations) with unchanged physics. */
    CHECK(warm_compute_status == GPUXTB_STATUS_SUCCESS);
    CHECK(per_system_status[0] == GPUXTB_STATUS_SUCCESS);
    CHECK(scc_converged[0] == 1u);
    CHECK(scc_iterations[0] >= 1);
    CHECK(std::isfinite(energies[0]));
    CHECK(std::isfinite(forces[0]) && std::isfinite(forces[1]) && std::isfinite(forces[2]));
    CHECK(std::isfinite(atomic_charges[0]));
  }

  compute_options.scc_start_mode = GPUXTB_SCC_START_FRESH;

  compute_options.model = GPUXTB_MODEL_GFN1_XTB;
  CHECK(gpuxtb_compute(context.get(), &batch, &compute_options, &result) ==
        GPUXTB_STATUS_NOT_SUPPORTED);
  CHECK(std::strstr(gpuxtb_get_last_error(), "GFN1-xTB") != nullptr);

  context.reset();

  options.cpu_threads = -1;
  gpuxtb_status_t invalid_context_status = GPUXTB_STATUS_INTERNAL_ERROR;
  ContextHandle invalid_context = create_context(options, invalid_context_status);
  CHECK(invalid_context_status == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(invalid_context == nullptr);
  CHECK(std::strstr(gpuxtb_get_last_error(), "cpu_threads") != nullptr);

  options.cpu_threads = 0;
  options.device_id = -2;
  invalid_context = create_context(options, invalid_context_status);
  CHECK(invalid_context_status == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(invalid_context == nullptr);

  options.device_id = -1;
  options.backend = static_cast<gpuxtb_backend_t>(99);
  invalid_context = create_context(options, invalid_context_status);
  CHECK(invalid_context_status == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(invalid_context == nullptr);

  options.backend = GPUXTB_BACKEND_CPU;
  options.reserved = 1;
  invalid_context = create_context(options, invalid_context_status);
  CHECK(invalid_context_status == GPUXTB_STATUS_INVALID_ARGUMENT);
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
  gpuxtb_status_t cuda_status = GPUXTB_STATUS_INTERNAL_ERROR;
  ContextHandle cuda_context = create_context(options, cuda_status);
  if (cuda_status == GPUXTB_STATUS_SUCCESS) {
    CHECK(gpuxtb_context_get_backend(cuda_context.get()) == GPUXTB_BACKEND_CUDA);
    CHECK(gpuxtb_context_get_device_id(cuda_context.get()) >= 0);

    /* A host allocation mislabeled as CUDA device memory must be rejected by
     * pointer preflight before topology staging or output publication. */
    compute_options.model = GPUXTB_MODEL_GFN2_XTB;
    energies[0] = 123.25;
    forces[0] = -4.0;
    forces[1] = -5.0;
    forces[2] = -6.0;
    scc_iterations[0] = 91;
    scc_converged[0] = 1u;
    per_system_status[0] = GPUXTB_STATUS_INTERNAL_ERROR;
    atomic_charges[0] = 71.25;
    point_charge_forces[0] = 81.0;
    point_charge_forces[1] = 82.0;
    point_charge_forces[2] = 83.0;
    result.flags = UINT32_C(0xa5a55a5a);
    gpuxtb_batch_t opaque_batch = batch;
    opaque_batch.atom_offsets.memory_space = GPUXTB_MEMORY_CUDA_DEVICE;
    CHECK(gpuxtb_compute(cuda_context.get(), &opaque_batch, &compute_options, &result) ==
          GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(std::strstr(gpuxtb_get_last_error(), "atom_offsets") != nullptr);
    CHECK(energies[0] == 123.25);
    CHECK(forces[0] == -4.0 && forces[1] == -5.0 && forces[2] == -6.0);
    CHECK(scc_iterations[0] == 91);
    CHECK(scc_converged[0] == 1u);
    CHECK(per_system_status[0] == GPUXTB_STATUS_INTERNAL_ERROR);
    CHECK(atomic_charges[0] == 71.25);
    CHECK(point_charge_forces[0] == 81.0 && point_charge_forces[1] == 82.0 &&
          point_charge_forces[2] == 83.0);
    CHECK(result.flags == UINT32_C(0xa5a55a5a));

    /* Structural failures remain deterministic without consulting storage. */
    opaque_batch.atom_offsets.memory_space = GPUXTB_MEMORY_HOST;
    opaque_batch.atom_offsets.size_bytes = sizeof(std::int64_t);
    CHECK(gpuxtb_compute(cuda_context.get(), &opaque_batch, &compute_options, &result) ==
          GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(std::strstr(gpuxtb_get_last_error(), "atom_offsets") != nullptr);
    CHECK(energies[0] == 123.25);
    CHECK(forces[0] == -4.0 && forces[1] == -5.0 && forces[2] == -6.0);
    CHECK(scc_iterations[0] == 91);
    CHECK(scc_converged[0] == 1u);
    CHECK(per_system_status[0] == GPUXTB_STATUS_INTERNAL_ERROR);
    CHECK(atomic_charges[0] == 71.25);
    CHECK(point_charge_forces[0] == 81.0 && point_charge_forces[1] == 82.0 &&
          point_charge_forces[2] == 83.0);
    CHECK(result.flags == UINT32_C(0xa5a55a5a));
  } else {
    /* CUDA-enabled builds also run on hosts where the runtime exposes no device. */
    CHECK(cuda_status == GPUXTB_STATUS_BACKEND_UNAVAILABLE);
    CHECK(cuda_context == nullptr);
  }

  CHECK(gpuxtb_context_options_init(&options, sizeof(options)) == GPUXTB_STATUS_SUCCESS);
  options.backend = GPUXTB_BACKEND_AUTO;
  options.device_id = INT32_MAX;
  gpuxtb_status_t automatic_status = GPUXTB_STATUS_INTERNAL_ERROR;
  ContextHandle automatic_context = create_context(options, automatic_status);
  CHECK(automatic_status == GPUXTB_STATUS_BACKEND_UNAVAILABLE);
  CHECK(automatic_context == nullptr);
#endif
  return 0;
}
