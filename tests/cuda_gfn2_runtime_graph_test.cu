#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_inference_publication.cuh"
#include "backends/cuda/gfn2_scc_loop.cuh"
#include "runtime/gfn2_cuda_execution.hpp"
#include "tests/support/gfn2_scc_test_case.hpp"
#include "xtbloom/xtbloom.h"
#include "xtbloom/xtbloom_external_energy.h"

#define CHECK(condition)                                                                 \
  do {                                                                                   \
    if (!(condition)) {                                                                  \
      std::fprintf(stderr, "CUDA runtime-graph check failed at line %d: %s\n", __LINE__, \
                   #condition);                                                          \
      return EXIT_FAILURE;                                                               \
    }                                                                                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using xtbloom::detail::Gfn2CudaExecutionCache;
using xtbloom::detail::Gfn2CudaExecutionIdentity;
using xtbloom::detail::Gfn2CudaNumericalInputView;
using xtbloom::detail::Gfn2CudaSccStartMode;
using xtbloom::detail::cuda::Gfn2InferencePublicationPlanError;
using xtbloom::detail::cuda::Gfn2InferencePublicationSystemError;
using xtbloom::test::gfn2::HostSccCase;
using xtbloom::test::gfn2::HostSccCaseOptions;
using xtbloom::test::gfn2::SmallSystemKind;

class EnvironmentGuard {
 public:
  explicit EnvironmentGuard(const char* name) : name_(name) {
    if (const char* original = std::getenv(name_); original != nullptr) {
      present_ = true;
      original_ = original;
    }
  }
  EnvironmentGuard(const EnvironmentGuard&) = delete;
  EnvironmentGuard& operator=(const EnvironmentGuard&) = delete;
  ~EnvironmentGuard() { (void)set(present_ ? original_.c_str() : nullptr); }

  bool set(const char* value) noexcept {
#if defined(_WIN32)
    return _putenv_s(name_, value == nullptr ? "" : value) == 0;
#else
    return value == nullptr ? unsetenv(name_) == 0 : setenv(name_, value, 1) == 0;
#endif
  }

 private:
  const char* name_;
  bool present_ = false;
  std::string original_;
};

struct ContextDeleter {
  void operator()(xtbloom_context_t* context) const noexcept { xtbloom_context_destroy(context); }
};

struct RequestDeleter {
  void operator()(xtbloom_request_t* request) const noexcept { xtbloom_request_destroy(request); }
};

using ContextHandle = std::unique_ptr<xtbloom_context_t, ContextDeleter>;
using RequestHandle = std::unique_ptr<xtbloom_request_t, RequestDeleter>;

ContextHandle create_cuda_context(std::int32_t device_id, xtbloom_status_t& status) {
  xtbloom_context_options_t options{};
  status = xtbloom_context_options_init(&options, sizeof(options));
  if (status != XTBLOOM_STATUS_SUCCESS) return {};
  options.backend = XTBLOOM_BACKEND_CUDA;
  options.device_id = device_id;
  xtbloom_context_t* raw_context = nullptr;
  status = xtbloom_context_create(&options, &raw_context);
  return ContextHandle(raw_context);
}

template <typename T>
xtbloom_buffer_t host_output_buffer(std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), XTBLOOM_MEMORY_HOST,
          0u};
}

struct PublicBatch;

struct PublicCanaryResult {
  static constexpr std::uint32_t kDescriptorCanary = UINT32_C(0x5a17c0de);
  static constexpr double kDoubleCanary = -9137.25;
  static constexpr std::int32_t kIntegerCanary = INT32_C(0x13572468);
  static constexpr std::uint8_t kByteCanary = UINT8_C(0xa5);

  std::vector<double> energies;
  std::vector<double> forces;
  std::vector<double> charges;
  std::vector<double> strain;
  std::vector<std::int32_t> iterations;
  std::vector<std::uint8_t> converged;
  std::vector<xtbloom_status_t> statuses;
  xtbloom_batch_result_t descriptor{};
  xtbloom_batch_result_t descriptor_before{};

  bool bind(const PublicBatch& batch, std::uint32_t flags);
  void capture_descriptor() noexcept;
  bool unchanged() const noexcept;
  bool successful() const noexcept;
};

xtbloom_compute_options_t options_with_start_mode(xtbloom_compute_options_t options,
                                                  xtbloom_scc_start_mode_t mode) noexcept {
  options.struct_size = XTBLOOM_COMPUTE_OPTIONS_V2_SIZE;
  options.scc_start_mode = mode;
  options.reserved_v2 = 0u;
  return options;
}

#ifdef XTBLOOM_CUDA_SCC_BENCHMARK_OVERRIDES
bool last_error_mentions_benchmark() {
  const char* error = xtbloom_get_last_error();
  return error != nullptr &&
         (std::strstr(error, "benchmark") != nullptr || std::strstr(error, "BENCHMARK") != nullptr);
}

bool last_error_mentions_mode_variable() {
  const char* error = xtbloom_get_last_error();
  return error != nullptr && std::strstr(error, "XTBLOOM_CUDA_SCC_BENCHMARK_MODE") != nullptr;
}

bool request_is_idle(xtbloom_request_t* request) {
  xtbloom_request_info_t info{};
  return xtbloom_request_info_init(&info, sizeof(info)) == XTBLOOM_STATUS_SUCCESS &&
         xtbloom_request_query(request, &info) == XTBLOOM_STATUS_SUCCESS &&
         info.state == XTBLOOM_REQUEST_IDLE;
}

xtbloom_status_t counting_external_energy_callback(
    void* opaque, std::int64_t, xtbloom_external_energy_phase_t, std::int32_t, std::int64_t,
    std::int64_t, const std::int32_t*, std::int64_t, const double*, std::int64_t,
    const std::int64_t*, std::int64_t, std::int64_t, std::int64_t, const std::int64_t*,
    std::int64_t, const std::int64_t*, std::int64_t, const std::uint8_t*, std::int64_t,
    const std::uint8_t*, std::int64_t, std::int64_t, std::int64_t, double, std::int32_t,
    const double*, std::int64_t, const double*, std::int64_t, const double*, std::int64_t,
    std::int64_t, std::int64_t, const std::int64_t*, std::int64_t, const std::int64_t*,
    std::int64_t, const std::uint8_t*, std::int64_t, const double*, std::int64_t, const double*,
    std::int64_t, double*, std::int64_t, double*, std::int64_t, double*) {
  static_cast<std::atomic<std::size_t>*>(opaque)->fetch_add(1u, std::memory_order_relaxed);
  return XTBLOOM_STATUS_SUCCESS;
}
#endif

template <typename T>
xtbloom_const_buffer_t host_buffer(const std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), XTBLOOM_MEMORY_HOST,
          0u};
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (data_ != nullptr) (void)cudaFree(data_);
  }

  cudaError_t assign(const std::vector<T>& values, cudaStream_t stream) {
    if (values.size() != elements_) {
      if (data_ != nullptr) {
        const cudaError_t free_status = cudaFree(data_);
        if (free_status != cudaSuccess) return free_status;
        data_ = nullptr;
      }
      elements_ = values.size();
      if (elements_ != 0u) {
        const cudaError_t allocation_status =
            cudaMalloc(reinterpret_cast<void**>(&data_), elements_ * sizeof(T));
        if (allocation_status != cudaSuccess) return allocation_status;
      }
    }
    return elements_ == 0u ? cudaSuccess
                           : cudaMemcpyAsync(data_, values.data(), elements_ * sizeof(T),
                                             cudaMemcpyHostToDevice, stream);
  }

  [[nodiscard]] xtbloom_const_buffer_t view() const noexcept {
    return {data_, elements_ * sizeof(T), XTBLOOM_MEMORY_CUDA_DEVICE, 0u};
  }

 private:
  T* data_ = nullptr;
  std::size_t elements_ = 0u;
};

class GraphOwner {
 public:
  GraphOwner() = default;
  GraphOwner(const GraphOwner&) = delete;
  GraphOwner& operator=(const GraphOwner&) = delete;
  ~GraphOwner() {
    if (executable_ != nullptr) (void)cudaGraphExecDestroy(executable_);
    if (graph_ != nullptr) (void)cudaGraphDestroy(graph_);
  }

  cudaGraph_t* graph_address() noexcept { return &graph_; }
  cudaGraph_t graph() const noexcept { return graph_; }
  cudaGraphExec_t* executable_address() noexcept { return &executable_; }
  cudaGraphExec_t executable() const noexcept { return executable_; }

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
};

struct PublicBatch {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int64_t> point_offsets;
  std::vector<double> point_positions;
  std::vector<double> point_values;
  std::vector<double> point_gammas;
  std::vector<double> periodic_shifts;
  std::vector<std::int64_t> response_offsets;
  std::vector<double> response_matrix;
  std::vector<xtbloom_interaction_t> interactions;
  std::vector<std::uint8_t> interaction_payload;
  xtbloom_batch_t descriptor{};

  void bind() noexcept {
    descriptor = {};
    descriptor.struct_size = interactions.empty() ? XTBLOOM_BATCH_V1_SIZE : XTBLOOM_BATCH_V3_SIZE;
    descriptor.api_version = XTBLOOM_API_VERSION;
    descriptor.batch_size = static_cast<std::int64_t>(molecular_charges.size());
    descriptor.total_atoms = static_cast<std::int64_t>(atomic_numbers.size());
    descriptor.total_point_charges = static_cast<std::int64_t>(point_values.size());
    descriptor.total_charge_response_elements = static_cast<std::int64_t>(response_matrix.size());
    descriptor.atom_offsets = host_buffer(atom_offsets);
    descriptor.atomic_numbers = host_buffer(atomic_numbers);
    descriptor.positions = host_buffer(positions);
    descriptor.molecular_charges = host_buffer(molecular_charges);
    descriptor.unpaired_electrons = host_buffer(unpaired_electrons);
    descriptor.point_charge_offsets = host_buffer(point_offsets);
    descriptor.point_charge_positions = host_buffer(point_positions);
    descriptor.point_charge_values = host_buffer(point_values);
    descriptor.point_charge_gammas = host_buffer(point_gammas);
    descriptor.atomic_potential_shifts = host_buffer(periodic_shifts);
    descriptor.charge_response_offsets = host_buffer(response_offsets);
    descriptor.charge_response_matrix = host_buffer(response_matrix);
    if (!interactions.empty()) {
      descriptor.total_interactions = static_cast<std::int64_t>(interactions.size());
      descriptor.interaction_descriptors = host_buffer(interactions);
      descriptor.interaction_payload = host_buffer(interaction_payload);
    }
  }

  void set_electric_fields(const std::vector<std::array<double, 3>>& fields) {
    interactions.assign(fields.size(), {});
    interaction_payload.assign(fields.size() * 32u, 0u);
    for (std::size_t system = 0; system < fields.size(); ++system) {
      const std::int32_t version = 1;
      const std::size_t offset = 32u * system;
      std::memcpy(interaction_payload.data() + offset, &version, sizeof(version));
      std::memcpy(interaction_payload.data() + offset + 8u, fields[system].data(),
                  sizeof(fields[system]));
      interactions[system].type = XTBLOOM_INTERACTION_ELECTRIC_FIELD;
      interactions[system].system_index = static_cast<std::int64_t>(system);
      interactions[system].payload_offset = offset;
      interactions[system].payload_size = 32u;
    }
    bind();
  }

  static PublicBatch from_host(const HostSccCase& host) {
    PublicBatch batch;
    batch.atom_offsets = host.atom_offsets();
    batch.atomic_numbers = host.atomic_numbers();
    batch.positions = host.positions();
    batch.molecular_charges = host.molecular_charges();
    batch.unpaired_electrons = host.unpaired_electrons();
    batch.point_offsets = host.point_charge_offsets();
    batch.point_positions = host.point_charge_positions();
    batch.point_values = host.point_charge_charges();
    batch.point_gammas = host.point_charge_hardnesses();
    batch.periodic_shifts = host.periodic_shifts();
    batch.response_matrix = host.periodic_response_matrices();
    batch.response_offsets.assign(static_cast<std::size_t>(host.batch_size() + 1), 0);
    for (std::int64_t system = 0; system < host.batch_size(); ++system) {
      const std::int64_t atoms = host.atom_offsets()[static_cast<std::size_t>(system + 1)] -
                                 host.atom_offsets()[static_cast<std::size_t>(system)];
      batch.response_offsets[static_cast<std::size_t>(system + 1)] =
          batch.response_offsets[static_cast<std::size_t>(system)] + atoms * atoms;
    }
    batch.bind();
    return batch;
  }
};

bool PublicCanaryResult::bind(const PublicBatch& batch, std::uint32_t flags) {
  const std::size_t systems = static_cast<std::size_t>(batch.descriptor.batch_size);
  const std::size_t atoms = static_cast<std::size_t>(batch.descriptor.total_atoms);
  energies.assign(systems, kDoubleCanary);
  forces.assign(3u * atoms, kDoubleCanary);
  charges.assign(atoms, kDoubleCanary);
  strain.assign((flags & XTBLOOM_COMPUTE_STRAIN_DERIVATIVES) != 0u ? 9u * systems : 0u,
                kDoubleCanary);
  iterations.assign(systems, kIntegerCanary);
  converged.assign(systems, kByteCanary);
  statuses.assign(systems, XTBLOOM_STATUS_EIGENSOLVER_FAILED);

  if (xtbloom_batch_result_init(&descriptor, sizeof(descriptor)) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  descriptor.flags = kDescriptorCanary;
  descriptor.energies = host_output_buffer(energies);
  descriptor.forces = host_output_buffer(forces);
  descriptor.atomic_charges = host_output_buffer(charges);
  descriptor.scc_iterations = host_output_buffer(iterations);
  descriptor.scc_converged = host_output_buffer(converged);
  descriptor.per_system_status = host_output_buffer(statuses);
  if (!strain.empty()) descriptor.strain_derivatives = host_output_buffer(strain);
  capture_descriptor();
  return true;
}

void PublicCanaryResult::capture_descriptor() noexcept {
  std::memcpy(&descriptor_before, &descriptor, sizeof(descriptor));
}

bool PublicCanaryResult::unchanged() const noexcept {
  const auto all_equal = [](const auto& values, const auto& expected) {
    return std::all_of(values.begin(), values.end(),
                       [&](const auto& value) { return value == expected; });
  };
  return std::memcmp(&descriptor, &descriptor_before, sizeof(descriptor)) == 0 &&
         all_equal(energies, kDoubleCanary) && all_equal(forces, kDoubleCanary) &&
         all_equal(charges, kDoubleCanary) && all_equal(strain, kDoubleCanary) &&
         all_equal(iterations, kIntegerCanary) && all_equal(converged, kByteCanary) &&
         all_equal(statuses, XTBLOOM_STATUS_EIGENSOLVER_FAILED);
}

bool PublicCanaryResult::successful() const noexcept {
  return std::all_of(statuses.begin(), statuses.end(),
                     [](xtbloom_status_t status) { return status == XTBLOOM_STATUS_SUCCESS; }) &&
         std::all_of(converged.begin(), converged.end(),
                     [](std::uint8_t value) { return value == 1u; }) &&
         std::all_of(energies.begin(), energies.end(),
                     [](double value) { return std::isfinite(value); }) &&
         std::all_of(forces.begin(), forces.end(),
                     [](double value) { return std::isfinite(value); }) &&
         std::all_of(charges.begin(), charges.end(),
                     [](double value) { return std::isfinite(value); });
}

int run_public_sync(xtbloom_context_t* context, const PublicBatch& batch,
                    const xtbloom_compute_options_t& base_options,
                    xtbloom_scc_start_mode_t start_mode) {
  const xtbloom_compute_options_t options = options_with_start_mode(base_options, start_mode);
  PublicCanaryResult result;
  CHECK(result.bind(batch, options.flags));
  CHECK(xtbloom_compute(context, &batch.descriptor, &options, &result.descriptor) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(result.successful());
  return 0;
}

#ifdef XTBLOOM_CUDA_SCC_BENCHMARK_OVERRIDES
int run_public_enqueue_rejection(xtbloom_context_t* context, const PublicBatch& batch,
                                 const xtbloom_compute_options_t& base_options,
                                 xtbloom_status_t expected_status) {
  xtbloom_request_t* raw_request = nullptr;
  CHECK(xtbloom_request_create(context, &raw_request) == XTBLOOM_STATUS_SUCCESS);
  RequestHandle request(raw_request);
  const xtbloom_compute_options_t options =
      options_with_start_mode(base_options, XTBLOOM_SCC_START_FRESH);
  PublicCanaryResult result;
  CHECK(result.bind(batch, options.flags));
  CHECK(request_is_idle(request.get()));
  for (int attempt = 0; attempt < 2; ++attempt) {
    const xtbloom_status_t status = xtbloom_compute_enqueue(context, &batch.descriptor, &options,
                                                            &result.descriptor, request.get());
    CHECK(status == expected_status);
    CHECK(expected_status == XTBLOOM_STATUS_INVALID_ARGUMENT ? last_error_mentions_mode_variable()
                                                             : last_error_mentions_benchmark());
    CHECK(result.unchanged());
    CHECK(request_is_idle(request.get()));
  }
  return 0;
}

int run_public_native_strain_rejection(xtbloom_context_t* context,
                                       const PublicBatch& molecular_batch,
                                       const xtbloom_compute_options_t& base_options,
                                       bool malformed_mode = false) {
  PublicBatch native_batch = molecular_batch;
  native_batch.bind();
  const std::size_t systems = static_cast<std::size_t>(native_batch.descriptor.batch_size);
  std::vector<double> cells(9u * systems, 0.0);
  std::vector<std::int32_t> axes(systems, XTBLOOM_PERIODIC_AXES_XYZ);
  for (std::size_t system = 0u; system < systems; ++system) {
    cells[9u * system] = 8.0;
    cells[9u * system + 4u] = 8.0;
    cells[9u * system + 8u] = 8.0;
  }
  native_batch.descriptor.struct_size = XTBLOOM_BATCH_V4_SIZE;
  native_batch.descriptor.cell_matrices = host_buffer(cells);
  native_batch.descriptor.periodic_axes = host_buffer(axes);

  xtbloom_compute_options_t options = base_options;
  options.flags |= XTBLOOM_COMPUTE_STRAIN_DERIVATIVES;
  options = options_with_start_mode(options, XTBLOOM_SCC_START_FRESH);
  PublicCanaryResult result;
  CHECK(result.bind(native_batch, options.flags));
  const xtbloom_status_t status =
      xtbloom_compute(context, &native_batch.descriptor, &options, &result.descriptor);
  CHECK(status == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(last_error_mentions_benchmark());
  if (malformed_mode) CHECK(last_error_mentions_mode_variable());
  CHECK(result.unchanged());
  return 0;
}

int run_public_forced_mode(const char* mode, bool family_available, bool exercise_scope,
                           std::int32_t device_id, const PublicBatch& batch,
                           const xtbloom_compute_options_t& options,
                           EnvironmentGuard& environment_guard) {
  CHECK(environment_guard.set(mode));
  xtbloom_status_t context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle context = create_cuda_context(device_id, context_status);
  CHECK(context_status == XTBLOOM_STATUS_SUCCESS && context != nullptr);
  /* The context owns a frozen selection; later changes must not alter routing. */
  CHECK(environment_guard.set("malformed"));

  if (family_available) {
    CHECK(run_public_sync(context.get(), batch, options, XTBLOOM_SCC_START_FRESH) == 0);
  }
  CHECK(run_public_enqueue_rejection(context.get(), batch, options, XTBLOOM_STATUS_NOT_SUPPORTED) ==
        0);
  if (family_available) {
    CHECK(run_public_sync(context.get(), batch, options, XTBLOOM_SCC_START_WARM) == 0);
    CHECK(run_public_sync(context.get(), batch, options, XTBLOOM_SCC_START_FRESH) == 0);
    CHECK(run_public_sync(context.get(), batch, options, XTBLOOM_SCC_START_WARM) == 0);
  }

  if (exercise_scope) {
    PublicBatch point_charge_batch = batch;
    point_charge_batch.point_offsets = {0, 1, 1, 1, 1};
    point_charge_batch.point_positions = {20.0, 0.0, 0.0};
    point_charge_batch.point_values = {0.1};
    point_charge_batch.point_gammas = {1.0};
    point_charge_batch.bind();
    const auto fresh_options = options_with_start_mode(options, XTBLOOM_SCC_START_FRESH);
    PublicCanaryResult scope_result;
    CHECK(scope_result.bind(point_charge_batch, fresh_options.flags));
    CHECK(xtbloom_compute(context.get(), &point_charge_batch.descriptor, &fresh_options,
                          &scope_result.descriptor) == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(last_error_mentions_benchmark());
    CHECK(scope_result.unchanged());
    if (family_available) {
      CHECK(run_public_sync(context.get(), batch, options, XTBLOOM_SCC_START_WARM) == 0);
    }
  }

  CHECK(run_public_native_strain_rejection(context.get(), batch, options) == 0);
  if (family_available) {
    CHECK(run_public_sync(context.get(), batch, options, XTBLOOM_SCC_START_WARM) == 0);
  }
  return 0;
}

int run_public_invalid_mode(std::int32_t device_id, const PublicBatch& batch,
                            const xtbloom_compute_options_t& options,
                            EnvironmentGuard& environment_guard) {
  CHECK(environment_guard.set("malformed"));
  xtbloom_status_t context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle context = create_cuda_context(device_id, context_status);
  CHECK(context_status == XTBLOOM_STATUS_SUCCESS && context != nullptr);
  CHECK(environment_guard.set("tail"));
  CHECK(run_public_enqueue_rejection(context.get(), batch, options,
                                     XTBLOOM_STATUS_INVALID_ARGUMENT) == 0);
  CHECK(run_public_native_strain_rejection(context.get(), batch, options, true) == 0);
  return 0;
}

int run_public_callback_bridge_rejection(const char* mode, std::int32_t device_id,
                                         const PublicBatch& batch,
                                         const xtbloom_compute_options_t& base_options,
                                         EnvironmentGuard& environment_guard) {
  CHECK(environment_guard.set(mode));
  xtbloom_status_t context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle context = create_cuda_context(device_id, context_status);
  CHECK(context_status == XTBLOOM_STATUS_SUCCESS && context != nullptr);
  CHECK(environment_guard.set(std::strcmp(mode, "malformed") == 0 ? "tail" : "malformed"));
  std::atomic<std::size_t> callback_calls{0u};
  CHECK(xtbloom_context_set_external_energy_callback(context.get(),
                                                     &counting_external_energy_callback,
                                                     &callback_calls) == XTBLOOM_STATUS_SUCCESS);

  const auto options = options_with_start_mode(base_options, XTBLOOM_SCC_START_FRESH);
  PublicCanaryResult result;
  CHECK(result.bind(batch, options.flags));
  const xtbloom_status_t status =
      xtbloom_compute(context.get(), &batch.descriptor, &options, &result.descriptor);
  CHECK(status == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(last_error_mentions_benchmark());
  if (std::strcmp(mode, "malformed") == 0) CHECK(last_error_mentions_mode_variable());
  CHECK(callback_calls.load(std::memory_order_relaxed) == 0u);
  CHECK(result.unchanged());
  return 0;
}
#endif

int run_public_benchmark_override_regressions(std::int32_t device_id, const PublicBatch& batch,
                                              const xtbloom_compute_options_t& base_options,
                                              bool chain_available) {
  constexpr const char* kVariable = "XTBLOOM_CUDA_SCC_BENCHMARK_MODE";
  /* Four H2 systems keep the public energy/force/charge payload at 4+24+8 values. */
  CHECK(batch.descriptor.batch_size == 4);
  CHECK(batch.descriptor.total_atoms == 8);
  EnvironmentGuard environment_guard(kVariable);
#ifdef XTBLOOM_CUDA_SCC_BENCHMARK_OVERRIDES
  CHECK(run_public_forced_mode("tail", true, true, device_id, batch, base_options,
                               environment_guard) == 0);
  CHECK(run_public_forced_mode("chain", chain_available, false, device_id, batch, base_options,
                               environment_guard) == 0);
  CHECK(run_public_invalid_mode(device_id, batch, base_options, environment_guard) == 0);
  CHECK(run_public_callback_bridge_rejection("tail", device_id, batch, base_options,
                                             environment_guard) == 0);
  CHECK(run_public_callback_bridge_rejection("chain", device_id, batch, base_options,
                                             environment_guard) == 0);
  CHECK(run_public_callback_bridge_rejection("malformed", device_id, batch, base_options,
                                             environment_guard) == 0);
#else
  (void)chain_available;
  CHECK(environment_guard.set("malformed"));
  xtbloom_status_t context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle context = create_cuda_context(device_id, context_status);
  CHECK(context_status == XTBLOOM_STATUS_SUCCESS && context != nullptr);
  CHECK(environment_guard.set("tail"));
  CHECK(run_public_sync(context.get(), batch, base_options, XTBLOOM_SCC_START_FRESH) == 0);
  CHECK(run_public_sync(context.get(), batch, base_options, XTBLOOM_SCC_START_WARM) == 0);
#endif
  return 0;
}

struct DeviceInputs {
  DeviceBuffer<double> positions;
  DeviceBuffer<double> point_positions;
  DeviceBuffer<double> point_values;
  DeviceBuffer<double> point_gammas;
  DeviceBuffer<double> periodic_shifts;
  DeviceBuffer<double> response_matrix;
  DeviceBuffer<xtbloom_interaction_t> interactions;
  DeviceBuffer<std::uint8_t> interaction_payload;
  DeviceBuffer<std::uint8_t> requested;
  Gfn2CudaNumericalInputView view{};

  int initialize(const PublicBatch& batch, const std::vector<std::uint8_t>& mask,
                 cudaStream_t stream) {
    CUDA_CHECK(positions.assign(batch.positions, stream));
    CUDA_CHECK(point_positions.assign(batch.point_positions, stream));
    CUDA_CHECK(point_values.assign(batch.point_values, stream));
    CUDA_CHECK(point_gammas.assign(batch.point_gammas, stream));
    CUDA_CHECK(periodic_shifts.assign(batch.periodic_shifts, stream));
    CUDA_CHECK(response_matrix.assign(batch.response_matrix, stream));
    CUDA_CHECK(interactions.assign(batch.interactions, stream));
    CUDA_CHECK(interaction_payload.assign(batch.interaction_payload, stream));
    CUDA_CHECK(requested.assign(mask, stream));
    view.positions = positions.view();
    view.point_charge_positions = point_positions.view();
    view.point_charge_values = point_values.view();
    view.point_charge_gammas = point_gammas.view();
    view.atomic_potential_shifts = periodic_shifts.view();
    view.charge_response_matrix = response_matrix.view();
    view.total_interactions = static_cast<std::int64_t>(batch.interactions.size());
    view.interaction_descriptors = interactions.view();
    view.interaction_payload = interaction_payload.view();
    view.requested_mask = requested.view();
    return 0;
  }
};

struct Snapshot {
  std::uint64_t epoch = 0u;
  std::vector<std::uint64_t> committed;
  std::vector<std::uint8_t> eligible;
  std::vector<std::uint64_t> scc_state_iterations;
  std::vector<std::uint8_t> scc_state_converged;
  std::vector<xtbloom_status_t> scc_state_statuses;
  std::vector<double> energies;
  std::vector<double> qm_forces;
  std::vector<double> atomic_charges;
  std::vector<double> point_forces;
  std::vector<double> dipole_moments;
  std::vector<std::int32_t> iterations;
  std::vector<std::uint8_t> converged;
  std::vector<xtbloom_status_t> statuses;
  std::uint64_t publication_epoch = 0u;
  std::vector<std::uint32_t> publication_errors;
  std::uint32_t publication_plan_error = 0u;
  std::vector<std::uint64_t> warm_generations;
};

int download(const Gfn2CudaExecutionIdentity& identity, cudaStream_t stream, Snapshot& snapshot) {
  const std::size_t batch = static_cast<std::size_t>(identity.batch_size);
  const std::size_t atoms = static_cast<std::size_t>(identity.total_atoms);
  const std::size_t points = static_cast<std::size_t>(identity.total_point_charges);
  snapshot.committed.resize(batch);
  snapshot.eligible.resize(batch);
  snapshot.scc_state_iterations.resize(batch);
  snapshot.scc_state_converged.resize(batch);
  snapshot.scc_state_statuses.resize(batch);
  snapshot.energies.resize(batch);
  snapshot.qm_forces.resize(atoms * 3u);
  snapshot.atomic_charges.resize(atoms);
  snapshot.point_forces.resize(points * 3u);
  snapshot.dipole_moments.resize(batch * 3u);
  snapshot.iterations.resize(batch);
  snapshot.converged.resize(batch);
  snapshot.statuses.resize(batch);
  snapshot.publication_errors.resize(batch);
  snapshot.warm_generations.resize(batch);

  CUDA_CHECK(cudaMemcpyAsync(&snapshot.epoch,
                             reinterpret_cast<const void*>(identity.numerical_epoch),
                             sizeof(snapshot.epoch), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(snapshot.committed.data(),
                             reinterpret_cast<const void*>(identity.committed_generations),
                             batch * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(snapshot.eligible.data(),
                             reinterpret_cast<const void*>(identity.numerical_eligible_mask),
                             batch * sizeof(std::uint8_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(snapshot.scc_state_iterations.data(),
                             reinterpret_cast<const void*>(identity.scc_state_iterations),
                             batch * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(snapshot.scc_state_converged.data(),
                             reinterpret_cast<const void*>(identity.scc_state_converged),
                             batch * sizeof(std::uint8_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(snapshot.scc_state_statuses.data(),
                             reinterpret_cast<const void*>(identity.scc_state_system_statuses),
                             batch * sizeof(xtbloom_status_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(snapshot.energies.data(),
                             reinterpret_cast<const void*>(identity.inference_energies),
                             batch * sizeof(double), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(
      snapshot.qm_forces.data(), reinterpret_cast<const void*>(identity.inference_qm_forces),
      snapshot.qm_forces.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(snapshot.atomic_charges.data(),
                             reinterpret_cast<const void*>(identity.inference_atomic_charges),
                             snapshot.atomic_charges.size() * sizeof(double),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(
      snapshot.point_forces.data(), reinterpret_cast<const void*>(identity.inference_point_forces),
      snapshot.point_forces.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
  const auto* publication =
      reinterpret_cast<const xtbloom::detail::cuda::Gfn2InferencePublicationDeviceResults*>(
          identity.inference_results);
  CHECK(publication != nullptr);
  CHECK(publication->dipole_moment_elements == static_cast<std::int64_t>(3u * batch));
  CUDA_CHECK(cudaMemcpyAsync(snapshot.dipole_moments.data(), publication->dipole_moments,
                             snapshot.dipole_moments.size() * sizeof(double),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(snapshot.iterations.data(),
                             reinterpret_cast<const void*>(identity.inference_iterations),
                             batch * sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(snapshot.converged.data(),
                             reinterpret_cast<const void*>(identity.inference_converged),
                             batch * sizeof(std::uint8_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(snapshot.statuses.data(),
                             reinterpret_cast<const void*>(identity.inference_system_statuses),
                             batch * sizeof(xtbloom_status_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(
      cudaMemcpyAsync(&snapshot.publication_epoch,
                      reinterpret_cast<const void*>(identity.inference_publication_epoch_snapshot),
                      sizeof(snapshot.publication_epoch), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(
      cudaMemcpyAsync(snapshot.publication_errors.data(),
                      reinterpret_cast<const void*>(identity.inference_publication_system_errors),
                      batch * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(
      cudaMemcpyAsync(&snapshot.publication_plan_error,
                      reinterpret_cast<const void*>(identity.inference_publication_plan_error),
                      sizeof(snapshot.publication_plan_error), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(snapshot.warm_generations.data(),
                             reinterpret_cast<const void*>(identity.warm_checkpoint_generations),
                             batch * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return 0;
}

/* Graph replay is only useful when every topology-scoped allocation remains
 * fixed. Include all addresses consumed or published by the complete runtime
 * pipeline so this test catches a hidden rebuild between submissions. */
bool stable_graph_addresses(const Gfn2CudaExecutionIdentity& expected,
                            const Gfn2CudaExecutionIdentity& actual) noexcept {
  return expected.topology_fingerprint == actual.topology_fingerprint &&
         expected.plan_token == actual.plan_token &&
         expected.topology_owner == actual.topology_owner &&
         expected.inputs_owner == actual.inputs_owner &&
         expected.eigensolver_owner == actual.eigensolver_owner &&
         expected.initializer_owner == actual.initializer_owner &&
         expected.scc_binding == actual.scc_binding &&
         expected.scc_state_iterations == actual.scc_state_iterations &&
         expected.scc_state_converged == actual.scc_state_converged &&
         expected.scc_state_system_statuses == actual.scc_state_system_statuses &&
         expected.scc_loop_owner == actual.scc_loop_owner &&
         expected.scc_loop_active_count == actual.scc_loop_active_count &&
         expected.scc_loop_numerical_body_count == actual.scc_loop_numerical_body_count &&
         expected.energy_force_descriptors == actual.energy_force_descriptors &&
         expected.topology_arena == actual.topology_arena &&
         expected.input_arena == actual.input_arena &&
         expected.iteration_arena == actual.iteration_arena &&
         expected.eigensolver_setup_arena == actual.eigensolver_setup_arena &&
         expected.force_immutable_arena == actual.force_immutable_arena &&
         expected.force_execution_arena == actual.force_execution_arena &&
         expected.numerical_refresh_arena == actual.numerical_refresh_arena &&
         expected.interaction_device_staging_arena == actual.interaction_device_staging_arena &&
         expected.interaction_host_staging_arena == actual.interaction_host_staging_arena &&
         expected.numerical_refresh_binding == actual.numerical_refresh_binding &&
         expected.numerical_epoch == actual.numerical_epoch &&
         expected.committed_generations == actual.committed_generations &&
         expected.numerical_eligible_mask == actual.numerical_eligible_mask &&
         expected.overlap_factor_generations == actual.overlap_factor_generations &&
         expected.inference_arena == actual.inference_arena &&
         expected.inference_epoch_consumer == actual.inference_epoch_consumer &&
         expected.inference_results == actual.inference_results &&
         expected.inference_energies == actual.inference_energies &&
         expected.inference_qm_forces == actual.inference_qm_forces &&
         expected.inference_atomic_charges == actual.inference_atomic_charges &&
         expected.inference_point_forces == actual.inference_point_forces &&
         expected.inference_iterations == actual.inference_iterations &&
         expected.inference_converged == actual.inference_converged &&
         expected.inference_system_statuses == actual.inference_system_statuses &&
         expected.inference_publication_epoch_snapshot ==
             actual.inference_publication_epoch_snapshot &&
         expected.inference_publication_system_errors ==
             actual.inference_publication_system_errors &&
         expected.inference_publication_plan_error == actual.inference_publication_plan_error &&
         expected.warm_checkpoint_generations == actual.warm_checkpoint_generations;
}

bool finite_triplet(const std::vector<double>& values, std::size_t offset) noexcept {
  return std::isfinite(values[offset]) && std::isfinite(values[offset + 1u]) &&
         std::isfinite(values[offset + 2u]);
}

bool same_binary64_image(const std::vector<double>& first,
                         const std::vector<double>& second) noexcept {
  return first.size() == second.size() &&
         (first.empty() ||
          std::memcmp(first.data(), second.data(), first.size() * sizeof(double)) == 0);
}

int check_successful_peer(const PublicBatch& batch, const Snapshot& snapshot, std::size_t system,
                          std::uint64_t generation) {
  CHECK(snapshot.committed[system] == generation);
  CHECK(snapshot.eligible[system] == 1u);
  CHECK(snapshot.statuses[system] == XTBLOOM_STATUS_SUCCESS);
  CHECK(snapshot.converged[system] == 1u);
  CHECK(snapshot.iterations[system] > 0);
  CHECK(std::isfinite(snapshot.energies[system]));
  CHECK(snapshot.publication_errors[system] ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationSystemError::kSuccess));
  CHECK(snapshot.warm_generations[system] == generation);
  CHECK(finite_triplet(snapshot.dipole_moments, 3u * system));
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1u];
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    CHECK(std::isfinite(snapshot.atomic_charges[static_cast<std::size_t>(atom)]));
    CHECK(finite_triplet(snapshot.qm_forces, static_cast<std::size_t>(atom * 3)));
  }
  const std::int64_t point_begin = batch.point_offsets[system];
  const std::int64_t point_end = batch.point_offsets[system + 1u];
  for (std::int64_t point = point_begin; point < point_end; ++point) {
    CHECK(finite_triplet(snapshot.point_forces, static_cast<std::size_t>(point * 3)));
  }
  return 0;
}

int check_ineligible_peer(const PublicBatch& batch, const Snapshot& snapshot, std::size_t system,
                          std::uint64_t retained_generation) {
  CHECK(snapshot.committed[system] == retained_generation);
  CHECK(snapshot.eligible[system] == 0u);
  CHECK(snapshot.statuses[system] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(snapshot.converged[system] == 0u);
  CHECK(snapshot.iterations[system] == 0);
  CHECK(std::isnan(snapshot.energies[system]));
  CHECK(
      snapshot.publication_errors[system] ==
      static_cast<std::uint32_t>(Gfn2InferencePublicationSystemError::kIneligibleNumericalRefresh));
  CHECK(snapshot.warm_generations[system] == 0u);
  CHECK(std::isnan(snapshot.dipole_moments[3u * system]));
  CHECK(std::isnan(snapshot.dipole_moments[3u * system + 1u]));
  CHECK(std::isnan(snapshot.dipole_moments[3u * system + 2u]));
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1u];
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    CHECK(std::isnan(snapshot.atomic_charges[static_cast<std::size_t>(atom)]));
    const std::size_t xyz = static_cast<std::size_t>(atom * 3);
    CHECK(std::isnan(snapshot.qm_forces[xyz]));
    CHECK(std::isnan(snapshot.qm_forces[xyz + 1u]));
    CHECK(std::isnan(snapshot.qm_forces[xyz + 2u]));
  }
  const std::int64_t point_begin = batch.point_offsets[system];
  const std::int64_t point_end = batch.point_offsets[system + 1u];
  for (std::int64_t point = point_begin; point < point_end; ++point) {
    const std::size_t xyz = static_cast<std::size_t>(point * 3);
    CHECK(std::isnan(snapshot.point_forces[xyz]));
    CHECK(std::isnan(snapshot.point_forces[xyz + 1u]));
    CHECK(std::isnan(snapshot.point_forces[xyz + 2u]));
  }
  return 0;
}

xtbloom_compute_options_t compute_options() noexcept {
  xtbloom_compute_options_t options{};
  options.struct_size = XTBLOOM_COMPUTE_OPTIONS_V1_SIZE;
  options.api_version = XTBLOOM_API_VERSION;
  options.model = XTBLOOM_MODEL_GFN2_XTB;
  options.flags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES | XTBLOOM_COMPUTE_ATOMIC_CHARGES |
                  XTBLOOM_COMPUTE_POINT_CHARGE_FORCES | XTBLOOM_COMPUTE_DIPOLE_MOMENTS;
  options.max_scc_iterations = 32;
  options.charge_tolerance = 1.0e-10;
  options.energy_tolerance = 1.0e-8;
  options.electronic_temperature = 0.0;
  return options;
}

int run_complete_graph_case(cudaStream_t stream, std::int32_t device_id, const char* stream_name) {
  constexpr std::size_t kBatch = 4u;
  HostSccCaseOptions host_options{};
  host_options.systems.assign(kBatch, SmallSystemKind::kH2);
  host_options.maximum_iterations = 32u;
  host_options.mixer_history = 8;
  host_options.enable_d4 = true;
  host_options.enable_explicit_point_charges = true;
  host_options.enable_periodic_embedding = true;

  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(host_options, host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicBatch batch = PublicBatch::from_host(host);
  std::vector<std::array<double, 3>> fields(kBatch);
  for (std::size_t system = 0; system < kBatch; ++system) {
    fields[system] = {0.001 * static_cast<double>(system + 1u), -0.0007, 0.0004};
  }
  batch.set_electric_fields(fields);
  std::vector<std::uint8_t> requested(kBatch, 1u);
  DeviceInputs inputs;
  CHECK(inputs.initialize(batch, requested, stream) == 0);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  xtbloom_compute_options_t options = compute_options();
  bool reused = true;
  const xtbloom_status_t prepare_status =
      cache.prepare_host(batch.descriptor, options, reused, error);
  if (prepare_status != XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s stream runtime setup failed: status=%d error=%s\n", stream_name,
                 prepare_status, error.c_str());
  }
  CHECK(prepare_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  const Gfn2CudaExecutionIdentity stable = cache.identity();
  CHECK(stable.inference_ready == 1u);
  CHECK(stable.force_mode_ready == 1u);
  CHECK(stable.numerical_refresh_ready == 1u);
  CHECK(stable.scc_conditional_graph_ready == 1u);
  CHECK(stable.scc_loop_fallback_reason == 0u);
  CHECK(stable.scc_loop_owner != 0u);
  CHECK(stable.scc_loop_active_count != 0u);
  CHECK(stable.scc_loop_numerical_body_count != 0u);

  /* Capture the public runtime transaction as one graph. The fresh
   * device-checkpoint restore, capture-compatible bounded SCC fallback,
   * terminal energy/force, and internal publication must all become
   * replayable nodes. Normal uncaptured inference uses the internal
   * conditional WHILE Graph verified by the runtime parity matrix. */
  GraphOwner graph;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  const xtbloom_status_t refresh_status = cache.refresh_numerical_async(inputs.view, error);
  const xtbloom_status_t inference_status =
      refresh_status == XTBLOOM_STATUS_SUCCESS
          ? cache.execute_inference_async(Gfn2CudaSccStartMode::kFresh, error)
          : refresh_status;
  const cudaError_t end_capture = cudaStreamEndCapture(stream, graph.graph_address());
  if (refresh_status != XTBLOOM_STATUS_SUCCESS || inference_status != XTBLOOM_STATUS_SUCCESS ||
      end_capture != cudaSuccess) {
    std::fprintf(stderr,
                 "%s stream full-pipeline capture failed: refresh=%d inference=%d end=%s "
                 "error=%s\n",
                 stream_name, refresh_status, inference_status, cudaGetErrorString(end_capture),
                 error.c_str());
  }
  CHECK(refresh_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(inference_status == XTBLOOM_STATUS_SUCCESS);
  CUDA_CHECK(end_capture);
  CHECK(graph.graph() != nullptr);
  CUDA_CHECK(cudaGraphInstantiate(graph.executable_address(), graph.graph(), nullptr, nullptr, 0));
  CHECK(graph.executable() != nullptr);
  CHECK(stable_graph_addresses(stable, cache.identity()));

  CUDA_CHECK(cudaGraphLaunch(graph.executable(), stream));
  Snapshot first;
  CHECK(download(cache.identity(), stream, first) == 0);
  CHECK(first.epoch == 2u);
  CHECK(first.publication_epoch == 2u);
  CHECK(first.publication_plan_error ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationPlanError::kSuccess));
  for (std::size_t system = 0; system < kBatch; ++system) {
    CHECK(check_successful_peer(batch, first, system, 2u) == 0);
  }
  CHECK(stable_graph_addresses(stable, cache.identity()));

  /* Graph memcpy nodes retain the device source addresses, not the values.
   * Updating one caller-owned device input must therefore change the next
   * energy while every runtime allocation and descriptor remains stable. */
  batch.positions[0] += 0.137;
  batch.point_values[0] += 0.083;
  CUDA_CHECK(inputs.positions.assign(batch.positions, stream));
  CUDA_CHECK(inputs.point_values.assign(batch.point_values, stream));
  CUDA_CHECK(cudaGraphLaunch(graph.executable(), stream));
  Snapshot changed;
  CHECK(download(cache.identity(), stream, changed) == 0);
  CHECK(changed.epoch == 3u);
  CHECK(changed.publication_epoch == 3u);
  CHECK(changed.publication_plan_error ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationPlanError::kSuccess));
  for (std::size_t system = 0; system < kBatch; ++system) {
    CHECK(check_successful_peer(batch, changed, system, 3u) == 0);
  }
  CHECK(changed.energies[0] != first.energies[0]);
  CHECK(stable_graph_addresses(stable, cache.identity()));

  /* The captured descriptor and payload memcpy nodes retain caller device
   * addresses. Change only the field bytes, then inject a bad block version;
   * admission must leave the last published result intact. The uncaptured
   * prepare below clears the persistent request gate, after which the same
   * executable must recover without rebuilding any graph-owned storage. */
  fields[0][0] += 0.0025;
  batch.set_electric_fields(fields);
  CUDA_CHECK(inputs.interaction_payload.assign(batch.interaction_payload, stream));
  CUDA_CHECK(cudaGraphLaunch(graph.executable(), stream));
  Snapshot changed_field;
  CHECK(download(cache.identity(), stream, changed_field) == 0);
  CHECK(changed_field.epoch == 4u);
  CHECK(changed_field.energies[0] != changed.energies[0]);
  CHECK(stable_graph_addresses(stable, cache.identity()));

  const std::int32_t invalid_version = 2;
  std::memcpy(batch.interaction_payload.data(), &invalid_version, sizeof(invalid_version));
  CUDA_CHECK(inputs.interaction_payload.assign(batch.interaction_payload, stream));
  CUDA_CHECK(cudaGraphLaunch(graph.executable(), stream));
  Snapshot rejected;
  CHECK(download(cache.identity(), stream, rejected) == 0);
  CHECK(rejected.epoch == changed_field.epoch);
  CHECK(rejected.publication_epoch == changed_field.publication_epoch);
  CHECK(rejected.energies == changed_field.energies);
  CHECK(rejected.qm_forces == changed_field.qm_forces);
  CHECK(rejected.atomic_charges == changed_field.atomic_charges);
  CHECK(rejected.point_forces == changed_field.point_forces);
  CHECK(rejected.dipole_moments == changed_field.dipole_moments);
  CHECK(stable_graph_addresses(stable, cache.identity()));

  const std::int32_t valid_version = 1;
  std::memcpy(batch.interaction_payload.data(), &valid_version, sizeof(valid_version));
  CUDA_CHECK(inputs.interaction_payload.assign(batch.interaction_payload, stream));
  reused = false;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(reused);
  CHECK(stable_graph_addresses(stable, cache.identity()));
  CUDA_CHECK(cudaGraphLaunch(graph.executable(), stream));
  Snapshot field_recovered;
  CHECK(download(cache.identity(), stream, field_recovered) == 0);
  CHECK(field_recovered.epoch == 6u);
  for (std::size_t system = 0; system < kBatch; ++system) {
    CHECK(check_successful_peer(batch, field_recovered, system, 6u) == 0);
  }
  CHECK(stable_graph_addresses(stable, cache.identity()));

  /* One inactive peer exercises dynamic failure isolation inside the same
   * executable. Its prior generation is retained while the other peers run
   * SCC and publish energy, charges, QM force, and point-charge force. */
  requested[1] = 0u;
  batch.positions[0] -= 0.041;
  CUDA_CHECK(inputs.positions.assign(batch.positions, stream));
  CUDA_CHECK(inputs.requested.assign(requested, stream));
  CUDA_CHECK(cudaGraphLaunch(graph.executable(), stream));
  Snapshot mixed;
  CHECK(download(cache.identity(), stream, mixed) == 0);
  CHECK(mixed.epoch == 7u);
  CHECK(mixed.publication_epoch == 7u);
  CHECK(mixed.publication_plan_error ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationPlanError::kSuccess));
  CHECK(check_ineligible_peer(batch, mixed, 1u, 6u) == 0);
  for (const std::size_t system : {0u, 2u, 3u}) {
    CHECK(check_successful_peer(batch, mixed, system, 7u) == 0);
  }
  CHECK(stable_graph_addresses(stable, cache.identity()));

  /* Re-enable the failed member to prove that replay derives activity from
   * current inputs instead of baking the capture-time all-active mask. */
  requested[1] = 1u;
  CUDA_CHECK(inputs.requested.assign(requested, stream));
  CUDA_CHECK(cudaGraphLaunch(graph.executable(), stream));
  Snapshot recovered;
  CHECK(download(cache.identity(), stream, recovered) == 0);
  CHECK(recovered.epoch == 8u);
  CHECK(recovered.publication_epoch == 8u);
  for (std::size_t system = 0; system < kBatch; ++system) {
    CHECK(check_successful_peer(batch, recovered, system, 8u) == 0);
  }
  CHECK(stable_graph_addresses(stable, cache.identity()));
  return 0;
}

int run_maximum_iteration_termination(cudaStream_t stream, std::int32_t device_id) {
  HostSccCaseOptions host_options{};
  host_options.systems = {SmallSystemKind::kH2};
  host_options.maximum_iterations = 1u;
  host_options.mixer_history = 8;
  host_options.enable_d4 = true;
  host_options.enable_explicit_point_charges = true;
  host_options.enable_periodic_embedding = true;

  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(host_options, host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicBatch batch = PublicBatch::from_host(host);
  std::vector<std::array<double, 3>> fields{{{0.001, -0.0007, 0.0004}}};
  batch.set_electric_fields(fields);
  std::vector<std::uint8_t> requested(1u, 1u);
  DeviceInputs inputs;
  CHECK(inputs.initialize(batch, requested, stream) == 0);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  xtbloom_compute_options_t options = compute_options();
  options.max_scc_iterations = 1;
  bool reused = true;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  const Gfn2CudaExecutionIdentity stable = cache.identity();

  /* Capture forces the SCC owner through its bounded fallback: every one-
   * iteration DAG is present in the outer Graph even when request admission
   * later fails. Reaching the configured SCC bound is a device-published peer
   * result, not a host submission failure. */
  GraphOwner graph;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  const xtbloom_status_t refresh_status = cache.refresh_numerical_async(inputs.view, error);
  const xtbloom_status_t inference_status =
      refresh_status == XTBLOOM_STATUS_SUCCESS
          ? cache.execute_inference_async(Gfn2CudaSccStartMode::kFresh, error)
          : refresh_status;
  CUDA_CHECK(cudaStreamEndCapture(stream, graph.graph_address()));
  CHECK(refresh_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(inference_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(graph.graph() != nullptr);
  CUDA_CHECK(cudaGraphInstantiate(graph.executable_address(), graph.graph(), nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphLaunch(graph.executable(), stream));
  Snapshot snapshot;
  CHECK(download(cache.identity(), stream, snapshot) == 0);
  CHECK(snapshot.epoch == 2u);
  CHECK(snapshot.committed[0] == 2u);
  CHECK(snapshot.eligible[0] == 1u);
  CHECK(snapshot.statuses[0] == XTBLOOM_STATUS_SCC_NOT_CONVERGED);
  CHECK(snapshot.converged[0] == 0u);
  CHECK(snapshot.iterations[0] == 1);
  CHECK(snapshot.scc_state_statuses[0] == XTBLOOM_STATUS_SCC_NOT_CONVERGED);
  CHECK(snapshot.scc_state_converged[0] == 0u);
  CHECK(snapshot.scc_state_iterations[0] == 1u);
  CHECK(snapshot.publication_epoch == 2u);
  CHECK(snapshot.publication_plan_error ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationPlanError::kSuccess));
  CHECK(snapshot.publication_errors[0] ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationSystemError::kSuccess));
  CHECK(snapshot.warm_generations[0] == 0u);
  CHECK(std::isnan(snapshot.energies[0]));
  for (const double value : snapshot.qm_forces) CHECK(std::isnan(value));
  for (const double value : snapshot.atomic_charges) CHECK(std::isnan(value));
  for (const double value : snapshot.point_forces) CHECK(std::isnan(value));
  CHECK(stable_graph_addresses(stable, cache.identity()));

  /* The last bounded iteration derived this peer as active before publishing
   * max-iteration nonconvergence, so its activity ledger is deliberately the
   * dangerous stale predecessor for the next request. A malformed device
   * field must close that ledger before any already-captured SCC body mutates
   * canonical state, a checkpoint, or a published result. */
  const std::int32_t invalid_version = 2;
  std::memcpy(batch.interaction_payload.data(), &invalid_version, sizeof(invalid_version));
  CUDA_CHECK(inputs.interaction_payload.assign(batch.interaction_payload, stream));
  CUDA_CHECK(cudaGraphLaunch(graph.executable(), stream));
  Snapshot rejected;
  CHECK(download(cache.identity(), stream, rejected) == 0);
  CHECK(rejected.epoch == snapshot.epoch);
  CHECK(rejected.committed == snapshot.committed);
  CHECK(rejected.eligible == snapshot.eligible);
  CHECK(rejected.scc_state_iterations == snapshot.scc_state_iterations);
  CHECK(rejected.scc_state_converged == snapshot.scc_state_converged);
  CHECK(rejected.scc_state_statuses == snapshot.scc_state_statuses);
  CHECK(same_binary64_image(rejected.energies, snapshot.energies));
  CHECK(same_binary64_image(rejected.qm_forces, snapshot.qm_forces));
  CHECK(same_binary64_image(rejected.atomic_charges, snapshot.atomic_charges));
  CHECK(same_binary64_image(rejected.point_forces, snapshot.point_forces));
  CHECK(same_binary64_image(rejected.dipole_moments, snapshot.dipole_moments));
  CHECK(rejected.iterations == snapshot.iterations);
  CHECK(rejected.converged == snapshot.converged);
  CHECK(rejected.statuses == snapshot.statuses);
  CHECK(rejected.publication_epoch == snapshot.publication_epoch);
  CHECK(rejected.publication_errors == snapshot.publication_errors);
  CHECK(rejected.publication_plan_error == snapshot.publication_plan_error);
  CHECK(rejected.warm_generations == snapshot.warm_generations);
  CHECK(stable_graph_addresses(stable, cache.identity()));

  /* Nonconvergence minted no device checkpoint, and malformed admission did
   * not invent one. Public strict-WARM rejection after this result is covered
   * by test_public_warm_start_state_machine; this lower-level asynchronous
   * cache deliberately does not poll device completion to revise its host
   * readiness hint. */
  CHECK(rejected.warm_generations == std::vector<std::uint64_t>{0u});
  return 0;
}

int run_legacy_default_submission(std::int32_t device_id) {
  /* CUDA defines the legacy-null stream as non-capturable (attempting it
   * returns cudaErrorStreamCaptureUnsupported). Exercise that stream through
   * the identical non-Graph runtime path without turning an expected CUDA API
   * error into a compute-sanitizer failure. */
  HostSccCaseOptions host_options{};
  host_options.systems = {SmallSystemKind::kH2};
  host_options.maximum_iterations = 32u;
  host_options.mixer_history = 8;
  host_options.enable_d4 = true;
  host_options.enable_explicit_point_charges = true;
  host_options.enable_periodic_embedding = true;
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(host_options, host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicBatch batch = PublicBatch::from_host(host);
  std::vector<std::uint8_t> requested(1u, 1u);
  DeviceInputs inputs;
  CHECK(inputs.initialize(batch, requested, nullptr) == 0);
  CUDA_CHECK(cudaStreamSynchronize(nullptr));

  Gfn2CudaExecutionCache cache(device_id, nullptr);
  xtbloom_compute_options_t options = compute_options();
  bool reused = true;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  const Gfn2CudaExecutionIdentity stable = cache.identity();
  CHECK(cache.refresh_numerical_async(inputs.view, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(cache.execute_inference_async(Gfn2CudaSccStartMode::kFresh, error) ==
        XTBLOOM_STATUS_SUCCESS);
  Snapshot snapshot;
  CHECK(download(cache.identity(), nullptr, snapshot) == 0);
  CHECK(snapshot.epoch == 2u);
  CHECK(snapshot.publication_epoch == 2u);
  CHECK(snapshot.publication_plan_error ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationPlanError::kSuccess));
  CHECK(check_successful_peer(batch, snapshot, 0u, 2u) == 0);
  CHECK(stable_graph_addresses(stable, cache.identity()));
  return 0;
}

/* Exercise the owner snapshot independently of later environment mutations.
 * This is a Graph/cache regression, not an independent numerical oracle. */
int run_benchmark_owner_selection(std::int32_t device_id) {
  constexpr const char* kVariable = "XTBLOOM_CUDA_SCC_BENCHMARK_MODE";
  EnvironmentGuard environment_guard(kVariable);

  HostSccCaseOptions host_options{};
  host_options.systems.assign(4u, SmallSystemKind::kH2);
  host_options.maximum_iterations = 32u;
  host_options.mixer_history = 8;
  host_options.enable_d4 = true;
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(host_options, host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicBatch batch = PublicBatch::from_host(host);
  batch.response_offsets.clear();
  batch.bind();
  auto options = compute_options();
  options.flags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES | XTBLOOM_COMPUTE_ATOMIC_CHARGES;

  CHECK(environment_guard.set("tail"));
  Gfn2CudaExecutionCache tail_cache(device_id, nullptr);
  CHECK(environment_guard.set("chain"));
  Gfn2CudaExecutionCache chain_cache(device_id, nullptr);
  CHECK(environment_guard.set("malformed"));
  Gfn2CudaExecutionCache invalid_cache(device_id, nullptr);
  bool reused = true;
  CHECK(tail_cache.prepare_host(batch.descriptor, options, reused, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  const auto tail_identity = tail_cache.identity();
  const auto chain_status = chain_cache.prepare_host(batch.descriptor, options, reused, error);
  bool chain_available = true;
#ifdef XTBLOOM_CUDA_SCC_BENCHMARK_OVERRIDES
  if (chain_status == XTBLOOM_STATUS_NOT_IMPLEMENTED) {
    chain_available = false;
    std::printf("benchmark dispatch-chain owner: SKIP (%s)\n", error.c_str());
  } else {
    CHECK(chain_status == XTBLOOM_STATUS_SUCCESS);
    CHECK(!reused);
  }
#else
  CHECK(chain_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
#endif
  const auto chain_identity = chain_cache.identity();
  CHECK(tail_identity.scc_loop_owner != 0u);
  if (chain_available) CHECK(chain_identity.scc_loop_owner != 0u);
#ifdef XTBLOOM_CUDA_SCC_BENCHMARK_OVERRIDES
  const auto* tail_owner =
      reinterpret_cast<const xtbloom::detail::cuda::Gfn2SccLoopCudaGraphOwner*>(
          tail_identity.scc_loop_owner);
  CHECK(tail_owner->device_tail_graph_ready());
  CHECK(!tail_owner->device_dispatch_chain_ready());
  if (chain_available) {
    const auto* chain_owner =
        reinterpret_cast<const xtbloom::detail::cuda::Gfn2SccLoopCudaGraphOwner*>(
            chain_identity.scc_loop_owner);
    CHECK(chain_owner->device_dispatch_chain_ready());
  }
  CHECK(invalid_cache.prepare_host(batch.descriptor, options, reused, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(!invalid_cache.valid());
  CHECK(error.find(kVariable) != std::string::npos);

  std::vector<std::int32_t> spin_channels(4u, 1);
  spin_channels[1] = 2;
  batch.descriptor.struct_size = XTBLOOM_BATCH_V2_SIZE;
  batch.descriptor.spin_channels = host_buffer(spin_channels);
  CHECK(tail_cache.prepare_host(batch.descriptor, options, reused, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(error.find("restricted GFN2") != std::string::npos);
  CHECK(stable_graph_addresses(tail_identity, tail_cache.identity()));
  batch.bind();

  auto gfn1_options = options;
  gfn1_options.model = XTBLOOM_MODEL_GFN1_XTB;
  CHECK(tail_cache.prepare_host(batch.descriptor, gfn1_options, reused, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(stable_graph_addresses(tail_identity, tail_cache.identity()));
#else
  CHECK(invalid_cache.prepare_host(batch.descriptor, options, reused, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  CHECK(invalid_cache.prepare_host(batch.descriptor, options, reused, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(reused);
#endif
  CHECK(tail_cache.prepare_host(batch.descriptor, options, reused, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(reused);
  CHECK(stable_graph_addresses(tail_identity, tail_cache.identity()));
  if (chain_available) {
    CHECK(chain_cache.prepare_host(batch.descriptor, options, reused, error) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(reused);
    CHECK(stable_graph_addresses(chain_identity, chain_cache.identity()));
  }
  CHECK(run_public_benchmark_override_regressions(device_id, batch, options, chain_available) == 0);
  std::puts("benchmark owner selection: PASS");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  /* A source line divisible by 256 becomes process success on POSIX. Keep a
   * host-only exact-exit probe so nested CHECK failures cannot become green. */
  if (argc == 2 && std::strcmp(argv[1], "--check-failure-exit-only") == 0) {
    CHECK(false);
  }
  const bool benchmark_modes_only =
      argc == 2 && std::strcmp(argv[1], "--benchmark-modes-only") == 0;
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    /* CUDA-enabled packages must remain testable on CPU-only CI runners. */
    std::puts("cuda_gfn2_runtime_graph_test: SKIP (no CUDA device)");
    return benchmark_modes_only ? 77 : 0;
  }

  std::int32_t device_id = -1;
  CUDA_CHECK(cudaGetDevice(&device_id));

  if (benchmark_modes_only) return run_benchmark_owner_selection(device_id);

  cudaStream_t custom_stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&custom_stream, cudaStreamNonBlocking));
  int status = run_complete_graph_case(custom_stream, device_id, "custom");
  if (status == 0) status = run_maximum_iteration_termination(custom_stream, device_id);
  CUDA_CHECK(cudaStreamSynchronize(custom_stream));
  CUDA_CHECK(cudaStreamDestroy(custom_stream));

  /* The CUDA legacy-null stream is explicitly non-capturable. The per-thread
   * default stream is CUDA's graph-capable default-stream mode and exercises
   * the runtime without a caller-created stream object. */
  if (status == 0) {
    status = run_complete_graph_case(cudaStreamPerThread, device_id, "per-thread default");
  }
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
  if (status == 0) status = run_legacy_default_submission(device_id);
  CUDA_CHECK(cudaStreamSynchronize(nullptr));
  if (status == 0) std::puts("cuda_gfn2_runtime_graph_test: PASS");
  return status;
}
