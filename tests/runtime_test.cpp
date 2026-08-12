#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>

#include "xtbloom/xtbloom.h"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

struct ContextDeleter {
  void operator()(xtbloom_context_t* context) const noexcept { xtbloom_context_destroy(context); }
};

using ContextHandle = std::unique_ptr<xtbloom_context_t, ContextDeleter>;

ContextHandle create_context(const xtbloom_context_options_t& options, xtbloom_status_t& status) {
  xtbloom_context_t* raw_context = nullptr;
  status = xtbloom_context_create(&options, &raw_context);
  return ContextHandle(raw_context);
}

}  // namespace

int main() {
  CHECK(std::strcmp(xtbloom_status_string(XTBLOOM_STATUS_SCC_NOT_CONVERGED), "SCC not converged") ==
        0);
  CHECK(std::strcmp(xtbloom_status_string(XTBLOOM_STATUS_EIGENSOLVER_FAILED),
                    "eigensolver failed") == 0);

  xtbloom_context_options_t options;
  CHECK(xtbloom_context_options_init(&options, sizeof(options)) == XTBLOOM_STATUS_SUCCESS);
  options.backend = XTBLOOM_BACKEND_CPU;

  xtbloom_status_t context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle context = create_context(options, context_status);
  CHECK(context_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom_context_get_device_id(context.get()) == -1);

  xtbloom_batch_t batch;
  xtbloom_compute_options_t compute_options;
  xtbloom_batch_result_t result;
  CHECK(xtbloom_batch_init(&batch, sizeof(batch)) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom_compute_options_init(&compute_options, sizeof(compute_options)) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(compute_options.electronic_temperature == XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE);
  CHECK(compute_options.struct_size == XTBLOOM_COMPUTE_OPTIONS_V3_SIZE);
  CHECK(compute_options.scc_start_mode == XTBLOOM_SCC_START_FRESH);
  CHECK(compute_options.reserved_v2 == 0u);
  CHECK(compute_options.scc_mixer == XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN);
  CHECK(compute_options.scc_mixer_history == 8);
  CHECK(compute_options.scc_mixer_damping == 0.4);
  CHECK(compute_options.determinism == XTBLOOM_DETERMINISM_DEFAULT);
  CHECK(compute_options.reserved_v3 == 0u);
  CHECK(xtbloom_batch_result_init(&result, sizeof(result)) == XTBLOOM_STATUS_SUCCESS);

  /* Descriptor errors are reported before entering numerical execution. */
  const xtbloom_status_t compute_status =
      xtbloom_compute(context.get(), &batch, &compute_options, &result);
  CHECK(compute_status == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::strstr(xtbloom_get_last_error(), "batch_size") != nullptr);

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
  batch.atom_offsets = {atom_offsets, sizeof(atom_offsets), XTBLOOM_MEMORY_HOST, 0};
  batch.atomic_numbers = {atomic_numbers, sizeof(atomic_numbers), XTBLOOM_MEMORY_HOST, 0};
  batch.positions = {positions, sizeof(positions), XTBLOOM_MEMORY_HOST, 0};
  batch.molecular_charges = {molecular_charges, sizeof(molecular_charges), XTBLOOM_MEMORY_HOST, 0};
  batch.unpaired_electrons = {unpaired_electrons, sizeof(unpaired_electrons), XTBLOOM_MEMORY_HOST,
                              0};
  result.energies = {energies, sizeof(energies), XTBLOOM_MEMORY_HOST, 0};
  result.forces = {forces, sizeof(forces), XTBLOOM_MEMORY_HOST, 0};
  result.atomic_charges = {atomic_charges, sizeof(atomic_charges), XTBLOOM_MEMORY_HOST, 0};
  result.point_charge_forces = {point_charge_forces, sizeof(point_charge_forces),
                                XTBLOOM_MEMORY_HOST, 0};
  result.scc_iterations = {scc_iterations, sizeof(scc_iterations), XTBLOOM_MEMORY_HOST, 0};
  result.scc_converged = {scc_converged, sizeof(scc_converged), XTBLOOM_MEMORY_HOST, 0};
  result.per_system_status = {per_system_status, sizeof(per_system_status), XTBLOOM_MEMORY_HOST, 0};

  const xtbloom_status_t valid_compute_status =
      xtbloom_compute(context.get(), &batch, &compute_options, &result);
  if (valid_compute_status == XTBLOOM_STATUS_BACKEND_UNAVAILABLE) {
    /* CPU-only CI configurations need not provide a production LP64 BLAS
     * runtime, but the diagnostic must identify that missing contract. */
    CHECK(std::strstr(xtbloom_get_last_error(), "LP64") != nullptr);
  } else {
    CHECK(valid_compute_status == XTBLOOM_STATUS_SUCCESS);
    CHECK(per_system_status[0] == XTBLOOM_STATUS_SUCCESS);
    CHECK(scc_converged[0] == 1u);
    CHECK(scc_iterations[0] > 0);
    CHECK(std::isfinite(energies[0]));
    CHECK(std::isfinite(forces[0]));
    CHECK(std::isfinite(forces[1]));
    CHECK(std::isfinite(forces[2]));
  }

  /* The always-registered runtime smoke also carries the complete V3 policy
   * through fixed-plan normalization. Provider-free builds may stop at the
   * same explicit LP64 availability boundary as convenience compute. */
  xtbloom_plan_t* raw_plan = reinterpret_cast<xtbloom_plan_t*>(UINTPTR_MAX);
  const xtbloom_status_t plan_status =
      xtbloom_plan_create(context.get(), &batch, &compute_options, &raw_plan);
  if (valid_compute_status == XTBLOOM_STATUS_BACKEND_UNAVAILABLE) {
    CHECK(plan_status == XTBLOOM_STATUS_BACKEND_UNAVAILABLE);
    CHECK(raw_plan == nullptr);
    CHECK(std::strstr(xtbloom_get_last_error(), "LP64") != nullptr);
  } else {
    CHECK(plan_status == XTBLOOM_STATUS_SUCCESS);
    CHECK(raw_plan != nullptr);
    CHECK(xtbloom_plan_compute(raw_plan, &batch, &compute_options, &result) ==
          XTBLOOM_STATUS_SUCCESS);
    xtbloom_plan_destroy(raw_plan);
  }

  /* ABI-v1 callers do not expose the suffix and therefore retain strict FRESH
   * behavior even if adjacent bytes contain invalid V2 values. */
  compute_options.struct_size = XTBLOOM_COMPUTE_OPTIONS_V1_SIZE;
  compute_options.scc_start_mode = 0;
  compute_options.reserved_v2 = UINT32_MAX;
  const xtbloom_status_t v1_compute_status =
      xtbloom_compute(context.get(), &batch, &compute_options, &result);
  if (v1_compute_status == XTBLOOM_STATUS_BACKEND_UNAVAILABLE) {
    CHECK(std::strstr(xtbloom_get_last_error(), "LP64") != nullptr);
  } else {
    CHECK(v1_compute_status == XTBLOOM_STATUS_SUCCESS);
  }

  compute_options.struct_size = XTBLOOM_COMPUTE_OPTIONS_V2_SIZE;
  compute_options.scc_start_mode = XTBLOOM_SCC_START_WARM;
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
  per_system_status[0] = XTBLOOM_STATUS_INTERNAL_ERROR;
  result.flags = UINT32_C(0xa5a55a5a);
  const xtbloom_status_t warm_compute_status =
      xtbloom_compute(context.get(), &batch, &compute_options, &result);
  if (valid_compute_status == XTBLOOM_STATUS_BACKEND_UNAVAILABLE) {
    /* No LP64 BLAS runtime means the preceding FRESH call never converged, so
     * the strict WARM identity precondition (a fully converged compatible
     * predecessor) is not met and the request is rejected before execution. */
    CHECK(warm_compute_status == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(std::strstr(xtbloom_get_last_error(), "WARM") != nullptr);
    CHECK(energies[0] == 123.25);
    CHECK(forces[0] == -4.0 && forces[1] == -5.0 && forces[2] == -6.0);
    CHECK(atomic_charges[0] == 71.25);
    CHECK(point_charge_forces[0] == 81.0 && point_charge_forces[1] == 82.0 &&
          point_charge_forces[2] == 83.0);
    CHECK(scc_iterations[0] == 91);
    CHECK(scc_converged[0] == 1u);
    CHECK(per_system_status[0] == XTBLOOM_STATUS_INTERNAL_ERROR);
    CHECK(result.flags == UINT32_C(0xa5a55a5a));
  } else {
    /* The FRESH call converged, so WARM consumes that converged electronic
     * checkpoint and reconverges (fewer iterations) with unchanged physics. */
    CHECK(warm_compute_status == XTBLOOM_STATUS_SUCCESS);
    CHECK(per_system_status[0] == XTBLOOM_STATUS_SUCCESS);
    CHECK(scc_converged[0] == 1u);
    CHECK(scc_iterations[0] >= 1);
    CHECK(std::isfinite(energies[0]));
    CHECK(std::isfinite(forces[0]) && std::isfinite(forces[1]) && std::isfinite(forces[2]));
    CHECK(std::isfinite(atomic_charges[0]));
  }

  compute_options.scc_start_mode = XTBLOOM_SCC_START_FRESH;

  compute_options.model = XTBLOOM_MODEL_GFN1_XTB;
  CHECK(xtbloom_compute(context.get(), &batch, &compute_options, &result) ==
        XTBLOOM_STATUS_NOT_SUPPORTED);
  CHECK(std::strstr(xtbloom_get_last_error(), "GFN1-xTB") != nullptr);

  context.reset();

  options.cpu_threads = -1;
  xtbloom_status_t invalid_context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle invalid_context = create_context(options, invalid_context_status);
  CHECK(invalid_context_status == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(invalid_context == nullptr);
  CHECK(std::strstr(xtbloom_get_last_error(), "cpu_threads") != nullptr);

  options.cpu_threads = 0;
  options.device_id = -2;
  invalid_context = create_context(options, invalid_context_status);
  CHECK(invalid_context_status == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(invalid_context == nullptr);

  options.device_id = -1;
  options.backend = static_cast<xtbloom_backend_t>(99);
  invalid_context = create_context(options, invalid_context_status);
  CHECK(invalid_context_status == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(invalid_context == nullptr);

  options.backend = XTBLOOM_BACKEND_CPU;
  options.reserved = 1;
  invalid_context = create_context(options, invalid_context_status);
  CHECK(invalid_context_status == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(invalid_context == nullptr);

  struct ExtendedOptions {
    xtbloom_context_options_t options;
    std::uint64_t canary;
  } extended{};
  extended.canary = UINT64_C(0x5a5a5a5aa5a5a5a5);
  CHECK(xtbloom_context_options_init(&extended.options, sizeof(extended)) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(extended.options.struct_size == sizeof(extended));
  CHECK(extended.canary == UINT64_C(0x5a5a5a5aa5a5a5a5));

  CHECK(xtbloom_context_options_init(&options, XTBLOOM_CONTEXT_OPTIONS_V1_SIZE - 1) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);

#if defined(XTBLOOM_TEST_HAS_CUDA)
  CHECK(xtbloom_context_options_init(&options, sizeof(options)) == XTBLOOM_STATUS_SUCCESS);
  options.backend = XTBLOOM_BACKEND_CUDA;
  xtbloom_status_t cuda_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle cuda_context = create_context(options, cuda_status);
  if (cuda_status == XTBLOOM_STATUS_SUCCESS) {
    CHECK(xtbloom_context_get_backend(cuda_context.get()) == XTBLOOM_BACKEND_CUDA);
    CHECK(xtbloom_context_get_device_id(cuda_context.get()) >= 0);

    /* A host allocation mislabeled as CUDA device memory must be rejected by
     * pointer preflight before topology staging or output publication. */
    compute_options.model = XTBLOOM_MODEL_GFN2_XTB;
    energies[0] = 123.25;
    forces[0] = -4.0;
    forces[1] = -5.0;
    forces[2] = -6.0;
    scc_iterations[0] = 91;
    scc_converged[0] = 1u;
    per_system_status[0] = XTBLOOM_STATUS_INTERNAL_ERROR;
    atomic_charges[0] = 71.25;
    point_charge_forces[0] = 81.0;
    point_charge_forces[1] = 82.0;
    point_charge_forces[2] = 83.0;
    result.flags = UINT32_C(0xa5a55a5a);
    xtbloom_batch_t opaque_batch = batch;
    opaque_batch.atom_offsets.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
    CHECK(xtbloom_compute(cuda_context.get(), &opaque_batch, &compute_options, &result) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(std::strstr(xtbloom_get_last_error(), "atom_offsets") != nullptr);
    CHECK(energies[0] == 123.25);
    CHECK(forces[0] == -4.0 && forces[1] == -5.0 && forces[2] == -6.0);
    CHECK(scc_iterations[0] == 91);
    CHECK(scc_converged[0] == 1u);
    CHECK(per_system_status[0] == XTBLOOM_STATUS_INTERNAL_ERROR);
    CHECK(atomic_charges[0] == 71.25);
    CHECK(point_charge_forces[0] == 81.0 && point_charge_forces[1] == 82.0 &&
          point_charge_forces[2] == 83.0);
    CHECK(result.flags == UINT32_C(0xa5a55a5a));

    /* Structural failures remain deterministic without consulting storage. */
    opaque_batch.atom_offsets.memory_space = XTBLOOM_MEMORY_HOST;
    opaque_batch.atom_offsets.size_bytes = sizeof(std::int64_t);
    CHECK(xtbloom_compute(cuda_context.get(), &opaque_batch, &compute_options, &result) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(std::strstr(xtbloom_get_last_error(), "atom_offsets") != nullptr);
    CHECK(energies[0] == 123.25);
    CHECK(forces[0] == -4.0 && forces[1] == -5.0 && forces[2] == -6.0);
    CHECK(scc_iterations[0] == 91);
    CHECK(scc_converged[0] == 1u);
    CHECK(per_system_status[0] == XTBLOOM_STATUS_INTERNAL_ERROR);
    CHECK(atomic_charges[0] == 71.25);
    CHECK(point_charge_forces[0] == 81.0 && point_charge_forces[1] == 82.0 &&
          point_charge_forces[2] == 83.0);
    CHECK(result.flags == UINT32_C(0xa5a55a5a));
  } else {
    /* CUDA-enabled builds also run on hosts where the runtime exposes no device. */
    CHECK(cuda_status == XTBLOOM_STATUS_BACKEND_UNAVAILABLE);
    CHECK(cuda_context == nullptr);
  }

  CHECK(xtbloom_context_options_init(&options, sizeof(options)) == XTBLOOM_STATUS_SUCCESS);
  options.backend = XTBLOOM_BACKEND_AUTO;
  options.device_id = INT32_MAX;
  xtbloom_status_t automatic_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle automatic_context = create_context(options, automatic_status);
  CHECK(automatic_status == XTBLOOM_STATUS_BACKEND_UNAVAILABLE);
  CHECK(automatic_context == nullptr);
#endif
  return 0;
}
