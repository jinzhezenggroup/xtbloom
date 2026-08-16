#include <cmath>
#include <cstddef>
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

struct RequestDeleter {
  void operator()(xtbloom_request_t* request) const noexcept { xtbloom_request_destroy(request); }
};

using ContextHandle = std::unique_ptr<xtbloom_context_t, ContextDeleter>;
using RequestHandle = std::unique_ptr<xtbloom_request_t, RequestDeleter>;

ContextHandle create_context(const xtbloom_context_options_t& options, xtbloom_status_t& status) {
  xtbloom_context_t* raw_context = nullptr;
  status = xtbloom_context_create(&options, &raw_context);
  return ContextHandle(raw_context);
}

struct DlpackTensorMirror {
  void* data;
  struct {
    std::int32_t device_type;
    std::int32_t device_id;
  } device;
  std::int32_t ndim;
  struct {
    std::uint8_t code;
    std::uint8_t bits;
    std::uint16_t lanes;
  } dtype;
  std::int64_t* shape;
  std::int64_t* strides;
  std::uint64_t byte_offset;
};

struct DlpackManagedTensorMirror {
  DlpackTensorMirror dl_tensor;
  void* manager_ctx;
  void (*deleter)(DlpackManagedTensorMirror*);
};

#if UINTPTR_MAX == UINT64_MAX
static_assert(sizeof(DlpackTensorMirror) == 48u, "DLPack tensor layout must remain stable");
static_assert(sizeof(DlpackManagedTensorMirror) == 64u,
              "legacy DLPack managed-tensor layout must remain stable");
#endif

int check_status_and_initializer_rejections() {
  struct StatusCase {
    xtbloom_status_t status;
    const char* text;
  };
  const StatusCase cases[] = {
      {XTBLOOM_STATUS_SUCCESS, "success"},
      {XTBLOOM_STATUS_INVALID_ARGUMENT, "invalid argument"},
      {XTBLOOM_STATUS_BACKEND_UNAVAILABLE, "backend unavailable"},
      {XTBLOOM_STATUS_NOT_SUPPORTED, "not supported"},
      {XTBLOOM_STATUS_ALLOCATION_FAILED, "allocation failed"},
      {XTBLOOM_STATUS_NOT_IMPLEMENTED, "not implemented"},
      {XTBLOOM_STATUS_INTERNAL_ERROR, "internal error"},
      {XTBLOOM_STATUS_SCC_NOT_CONVERGED, "SCC not converged"},
      {XTBLOOM_STATUS_EIGENSOLVER_FAILED, "eigensolver failed"},
  };
  for (const StatusCase& status_case : cases) {
    CHECK(std::strcmp(xtbloom_status_string(status_case.status), status_case.text) == 0);
  }
  CHECK(std::strcmp(xtbloom_status_string(static_cast<xtbloom_status_t>(INT32_MAX)),
                    "unknown status") == 0);

  CHECK(xtbloom_context_options_init(nullptr, sizeof(xtbloom_context_options_t)) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom_batch_init(nullptr, sizeof(xtbloom_batch_t)) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom_compute_options_init(nullptr, sizeof(xtbloom_compute_options_t)) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom_batch_result_init(nullptr, sizeof(xtbloom_batch_result_t)) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom_workspace_query_init(nullptr, sizeof(xtbloom_workspace_query_t)) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom_request_info_init(nullptr, sizeof(xtbloom_request_info_t)) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom_result_owner_options_init(nullptr, sizeof(xtbloom_result_owner_options_t)) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);

  if (sizeof(std::size_t) > sizeof(std::uint32_t)) {
    xtbloom_context_options_t oversized;
    const std::size_t oversized_size = static_cast<std::size_t>(UINT32_MAX) + 1u;
    CHECK(xtbloom_context_options_init(&oversized, oversized_size) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
  }
  return 0;
}

int check_context_creation_rejections(const xtbloom_context_options_t& valid_options) {
  CHECK(xtbloom_context_create(&valid_options, nullptr) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  xtbloom_context_t* context = reinterpret_cast<xtbloom_context_t*>(UINTPTR_MAX);
  CHECK(xtbloom_context_create(nullptr, &context) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(context == nullptr);

  xtbloom_context_options_t invalid = valid_options;
  invalid.struct_size = XTBLOOM_CONTEXT_OPTIONS_V1_SIZE - 1u;
  context = reinterpret_cast<xtbloom_context_t*>(UINTPTR_MAX);
  CHECK(xtbloom_context_create(&invalid, &context) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(context == nullptr);

  invalid = valid_options;
  invalid.api_version = XTBLOOM_API_VERSION + 1u;
  context = reinterpret_cast<xtbloom_context_t*>(UINTPTR_MAX);
  CHECK(xtbloom_context_create(&invalid, &context) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(context == nullptr);

  CHECK(xtbloom_context_get_backend(nullptr) == XTBLOOM_BACKEND_AUTO);
  CHECK(std::strstr(xtbloom_get_last_error(), "context is NULL") != nullptr);
  CHECK(xtbloom_context_get_device_id(nullptr) == -1);
  CHECK(std::strstr(xtbloom_get_last_error(), "context is NULL") != nullptr);
  xtbloom_context_destroy(nullptr);
  return 0;
}

bool owner_create_returns(const xtbloom_result_owner_options_t* options,
                          xtbloom_status_t expected) {
  xtbloom_result_owner_t* owner = reinterpret_cast<xtbloom_result_owner_t*>(UINTPTR_MAX);
  const xtbloom_status_t status = xtbloom_result_owner_create(options, &owner);
  return status == expected && owner == nullptr;
}

bool export_is_rejected(const xtbloom_result_owner_t* owner,
                        const xtbloom_dlpack_view_t* view, int version = 0) {
  void* managed = reinterpret_cast<void*>(UINTPTR_MAX);
  const xtbloom_status_t status =
      xtbloom_result_owner_export_dltensor(owner, view, version, &managed);
  return status == XTBLOOM_STATUS_INVALID_ARGUMENT && managed == nullptr;
}

int check_result_owner_rejections() {
  xtbloom_result_owner_options_t options;
  CHECK(xtbloom_result_owner_options_init(&options, sizeof(options)) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom_result_owner_create(&options, nullptr) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(owner_create_returns(nullptr, XTBLOOM_STATUS_INVALID_ARGUMENT));

  xtbloom_result_owner_options_t invalid = options;
  invalid.struct_size = XTBLOOM_RESULT_OWNER_OPTIONS_V1_SIZE - 1u;
  CHECK(owner_create_returns(&invalid, XTBLOOM_STATUS_INVALID_ARGUMENT));
  invalid = options;
  invalid.api_version = XTBLOOM_API_VERSION + 1u;
  CHECK(owner_create_returns(&invalid, XTBLOOM_STATUS_INVALID_ARGUMENT));
  invalid = options;
  invalid.memory_space = static_cast<xtbloom_memory_space_t>(INT32_MAX);
  CHECK(owner_create_returns(&invalid, XTBLOOM_STATUS_INVALID_ARGUMENT));
  invalid = options;
  invalid.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
  invalid.device_id = -1;
  CHECK(owner_create_returns(&invalid, XTBLOOM_STATUS_INVALID_ARGUMENT));

#if !defined(XTBLOOM_TEST_HAS_CUDA)
  invalid = options;
  invalid.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
  invalid.device_id = 0;
  invalid.size_bytes = 64u;
  CHECK(owner_create_returns(&invalid, XTBLOOM_STATUS_BACKEND_UNAVAILABLE));
#endif

  options.size_bytes = 128u;
  xtbloom_result_owner_t* owner = nullptr;
  CHECK(xtbloom_result_owner_create(&options, &owner) == XTBLOOM_STATUS_SUCCESS);
  CHECK(owner != nullptr);
  CHECK(xtbloom_result_owner_buffer(owner, nullptr) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  xtbloom_result_owner_retain(nullptr);
  CHECK(std::strstr(xtbloom_get_last_error(), "result owner is NULL") != nullptr);
  xtbloom_result_owner_release(nullptr);
  CHECK(std::strstr(xtbloom_get_last_error(), "result owner is NULL") != nullptr);

  const std::int64_t one[1] = {1};
  xtbloom_dlpack_view_t view{};
  view.struct_size = sizeof(view);
  view.api_version = XTBLOOM_API_VERSION;
  view.dtype_code = 2;
  view.dtype_bits = 64;
  view.dtype_lanes = 1;
  view.ndim = 1;
  view.shape = one;

  CHECK(export_is_rejected(owner, nullptr));
  CHECK(xtbloom_result_owner_export_dltensor(owner, &view, 0, nullptr) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);

  xtbloom_dlpack_view_t invalid_view = view;
  invalid_view.struct_size = XTBLOOM_DLPACK_VIEW_V1_SIZE - 1u;
  CHECK(export_is_rejected(owner, &invalid_view));
  invalid_view = view;
  invalid_view.api_version = XTBLOOM_API_VERSION + 1u;
  CHECK(export_is_rejected(owner, &invalid_view));
  invalid_view = view;
  invalid_view.reserved = 1u;
  CHECK(export_is_rejected(owner, &invalid_view));
  invalid_view = view;
  invalid_view.ndim = -1;
  CHECK(export_is_rejected(owner, &invalid_view));
  invalid_view = view;
  invalid_view.ndim = XTBLOOM_DLPACK_MAX_NDIM + 1;
  CHECK(export_is_rejected(owner, &invalid_view));
  invalid_view = view;
  invalid_view.dtype_lanes = 2;
  CHECK(export_is_rejected(owner, &invalid_view));
  invalid_view = view;
  invalid_view.shape = nullptr;
  CHECK(export_is_rejected(owner, &invalid_view));

  const std::int64_t element_overflow_shape[2] = {INT64_MAX, 3};
  invalid_view = view;
  invalid_view.ndim = 2;
  invalid_view.shape = element_overflow_shape;
  CHECK(export_is_rejected(owner, &invalid_view));
  const std::int64_t payload_overflow_shape[1] = {INT64_MAX};
  invalid_view = view;
  invalid_view.shape = payload_overflow_shape;
  CHECK(export_is_rejected(owner, &invalid_view));
  invalid_view = view;
  invalid_view.byte_offset = 129u;
  CHECK(export_is_rejected(owner, &invalid_view));

  xtbloom_dlpack_view_t scalar = view;
  scalar.ndim = 0;
  scalar.shape = nullptr;
  void* managed = nullptr;
  CHECK(xtbloom_result_owner_export_dltensor(owner, &scalar, 0, &managed) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(managed != nullptr);
  DlpackManagedTensorMirror* legacy = static_cast<DlpackManagedTensorMirror*>(managed);
  CHECK(legacy->dl_tensor.ndim == 0 && legacy->deleter != nullptr);
  legacy->deleter(nullptr);
  legacy->deleter(legacy);

  xtbloom_result_owner_release(owner);
  return 0;
}

}  // namespace

int main() {
  CHECK(check_status_and_initializer_rejections() == 0);

  xtbloom_context_options_t options;
  CHECK(xtbloom_context_options_init(&options, sizeof(options)) == XTBLOOM_STATUS_SUCCESS);
  options.backend = XTBLOOM_BACKEND_CPU;
  CHECK(check_context_creation_rejections(options) == 0);
  CHECK(check_result_owner_rejections() == 0);

  xtbloom_status_t context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle context = create_context(options, context_status);
  CHECK(context_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom_context_get_backend(context.get()) == XTBLOOM_BACKEND_CPU);
  CHECK(xtbloom_context_get_device_id(context.get()) == -1);

  xtbloom_request_t* raw_request = reinterpret_cast<xtbloom_request_t*>(UINTPTR_MAX);
  CHECK(xtbloom_request_create(nullptr, &raw_request) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(raw_request == nullptr);
  CHECK(xtbloom_request_create(context.get(), nullptr) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom_request_create(context.get(), &raw_request) == XTBLOOM_STATUS_SUCCESS);
  CHECK(raw_request != nullptr);
  RequestHandle request(raw_request);

  xtbloom_request_info_t request_info;
  CHECK(xtbloom_request_info_init(&request_info, sizeof(request_info)) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom_request_query(nullptr, &request_info) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom_request_wait(nullptr, &request_info) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom_request_get_error(nullptr) == nullptr);
  CHECK(xtbloom_request_query(request.get(), nullptr) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom_request_wait(request.get(), nullptr) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom_request_query(request.get(), &request_info) == XTBLOOM_STATUS_SUCCESS);
  CHECK(request_info.state == XTBLOOM_REQUEST_IDLE);
  CHECK(xtbloom_request_wait(request.get(), &request_info) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::strcmp(xtbloom_request_get_error(request.get()), "") == 0);

  xtbloom_status_t other_context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle other_context = create_context(options, other_context_status);
  CHECK(other_context_status == XTBLOOM_STATUS_SUCCESS);
  xtbloom_request_t* raw_other_request = nullptr;
  CHECK(xtbloom_request_create(other_context.get(), &raw_other_request) == XTBLOOM_STATUS_SUCCESS);
  RequestHandle other_request(raw_other_request);
  CHECK(xtbloom_compute_enqueue(nullptr, nullptr, nullptr, nullptr, request.get()) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom_compute_enqueue(context.get(), nullptr, nullptr, nullptr, nullptr) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom_compute_enqueue(other_context.get(), nullptr, nullptr, nullptr, request.get()) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom_compute_enqueue(context.get(), nullptr, nullptr, nullptr, request.get()) ==
        XTBLOOM_STATUS_NOT_SUPPORTED);

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

  CHECK(xtbloom_compute(nullptr, &batch, &compute_options, &result) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom_compute(context.get(), nullptr, &compute_options, &result) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom_compute(context.get(), &batch, nullptr, &result) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom_compute(context.get(), &batch, &compute_options, nullptr) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);

  xtbloom_plan_t* rejected_plan = reinterpret_cast<xtbloom_plan_t*>(UINTPTR_MAX);
  CHECK(xtbloom_plan_create(nullptr, &batch, &compute_options, &rejected_plan) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(rejected_plan == nullptr);
  rejected_plan = reinterpret_cast<xtbloom_plan_t*>(UINTPTR_MAX);
  CHECK(xtbloom_plan_create(context.get(), nullptr, &compute_options, &rejected_plan) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(rejected_plan == nullptr);
  rejected_plan = reinterpret_cast<xtbloom_plan_t*>(UINTPTR_MAX);
  CHECK(xtbloom_plan_create(context.get(), &batch, nullptr, &rejected_plan) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(rejected_plan == nullptr);
  CHECK(xtbloom_plan_create(context.get(), &batch, &compute_options, nullptr) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom_plan_query_workspace(nullptr, nullptr) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom_plan_compute(nullptr, &batch, &compute_options, &result) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom_plan_compute_enqueue(nullptr, nullptr, nullptr, nullptr, request.get()) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  xtbloom_plan_destroy(nullptr);

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

  xtbloom_compute_options_t unknown_model_options = compute_options;
  unknown_model_options.model = static_cast<xtbloom_model_t>(INT32_MAX);
  result.flags = UINT32_C(0xa5a55a5a);
  CHECK(xtbloom_compute(context.get(), &batch, &unknown_model_options, &result) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(result.flags == UINT32_C(0xa5a55a5a));
  result.flags = 0u;

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

    xtbloom_workspace_query_t workspace;
    CHECK(xtbloom_workspace_query_init(&workspace, sizeof(workspace)) == XTBLOOM_STATUS_SUCCESS);
    workspace.compute_flags = compute_options.flags;
    CHECK(xtbloom_plan_query_workspace(raw_plan, &workspace) == XTBLOOM_STATUS_SUCCESS);
    xtbloom_workspace_query_t short_workspace = workspace;
    short_workspace.struct_size = XTBLOOM_WORKSPACE_QUERY_V1_SIZE - 1u;
    CHECK(xtbloom_plan_query_workspace(raw_plan, &short_workspace) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(xtbloom_plan_query_workspace(raw_plan, nullptr) == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(xtbloom_plan_compute(raw_plan, nullptr, &compute_options, &result) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(xtbloom_plan_compute(raw_plan, &batch, nullptr, &result) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(xtbloom_plan_compute(raw_plan, &batch, &compute_options, nullptr) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(xtbloom_plan_compute_enqueue(raw_plan, nullptr, nullptr, nullptr, nullptr) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(xtbloom_plan_compute_enqueue(raw_plan, nullptr, nullptr, nullptr,
                                       other_request.get()) == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(xtbloom_plan_compute_enqueue(raw_plan, nullptr, nullptr, nullptr, request.get()) ==
          XTBLOOM_STATUS_NOT_SUPPORTED);
    CHECK(xtbloom_plan_compute(raw_plan, &batch, &compute_options, &result) ==
          XTBLOOM_STATUS_SUCCESS);
    xtbloom_plan_destroy(raw_plan);
  }

  other_request.reset();
  other_context.reset();

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

  /* The reserved GFN1 tag selects its own CPU executor rather than falling
   * through to GFN2. This always-registered smoke intentionally checks only
   * public convergence/publication; independent goldens live in the dedicated
   * GFN1 conformance gate. */
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
  compute_options.model = XTBLOOM_MODEL_GFN1_XTB;
  const xtbloom_status_t gfn1_status =
      xtbloom_compute(context.get(), &batch, &compute_options, &result);
  if (gfn1_status == XTBLOOM_STATUS_BACKEND_UNAVAILABLE) {
    CHECK(std::strstr(xtbloom_get_last_error(), "LP64") != nullptr);
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
    CHECK(gfn1_status == XTBLOOM_STATUS_SUCCESS);
    CHECK(per_system_status[0] == XTBLOOM_STATUS_SUCCESS);
    CHECK(scc_converged[0] == 1u);
    CHECK(scc_iterations[0] > 0);
    CHECK(std::isfinite(energies[0]));
    CHECK(std::isfinite(forces[0]) && std::isfinite(forces[1]) && std::isfinite(forces[2]));
    CHECK(std::isfinite(atomic_charges[0]));
  }

  request.reset();
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
