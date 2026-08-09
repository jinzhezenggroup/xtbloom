#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_inference_publication.cuh"
#include "runtime/gfn2_cuda_execution.hpp"
#include "tests/support/gfn2_scc_test_case.hpp"
#include "xtbloom/xtbloom.h"

#define CHECK(condition)                                                                 \
  do {                                                                                   \
    if (!(condition)) {                                                                  \
      std::fprintf(stderr, "CUDA runtime-graph check failed at line %d: %s\n", __LINE__, \
                   #condition);                                                          \
      return __LINE__;                                                                   \
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
  xtbloom_batch_t descriptor{};

  void bind() noexcept {
    descriptor = {};
    descriptor.struct_size = XTBLOOM_BATCH_V1_SIZE;
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

struct DeviceInputs {
  DeviceBuffer<double> positions;
  DeviceBuffer<double> point_positions;
  DeviceBuffer<double> point_values;
  DeviceBuffer<double> point_gammas;
  DeviceBuffer<double> periodic_shifts;
  DeviceBuffer<double> response_matrix;
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
    CUDA_CHECK(requested.assign(mask, stream));
    view.positions = positions.view();
    view.point_charge_positions = point_positions.view();
    view.point_charge_values = point_values.view();
    view.point_charge_gammas = point_gammas.view();
    view.atomic_potential_shifts = periodic_shifts.view();
    view.charge_response_matrix = response_matrix.view();
    view.requested_mask = requested.view();
    return 0;
  }
};

struct Snapshot {
  std::uint64_t epoch = 0u;
  std::vector<std::uint64_t> committed;
  std::vector<std::uint8_t> eligible;
  std::vector<double> energies;
  std::vector<double> qm_forces;
  std::vector<double> atomic_charges;
  std::vector<double> point_forces;
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
  snapshot.energies.resize(batch);
  snapshot.qm_forces.resize(atoms * 3u);
  snapshot.atomic_charges.resize(atoms);
  snapshot.point_forces.resize(points * 3u);
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
                  XTBLOOM_COMPUTE_POINT_CHARGE_FORCES;
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
  CHECK(mixed.epoch == 4u);
  CHECK(mixed.publication_epoch == 4u);
  CHECK(mixed.publication_plan_error ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationPlanError::kSuccess));
  CHECK(check_ineligible_peer(batch, mixed, 1u, 3u) == 0);
  for (const std::size_t system : {0u, 2u, 3u}) {
    CHECK(check_successful_peer(batch, mixed, system, 4u) == 0);
  }
  CHECK(stable_graph_addresses(stable, cache.identity()));

  /* Re-enable the failed member to prove that replay derives activity from
   * current inputs instead of baking the capture-time all-active mask. */
  requested[1] = 1u;
  CUDA_CHECK(inputs.requested.assign(requested, stream));
  CUDA_CHECK(cudaGraphLaunch(graph.executable(), stream));
  Snapshot recovered;
  CHECK(download(cache.identity(), stream, recovered) == 0);
  CHECK(recovered.epoch == 5u);
  CHECK(recovered.publication_epoch == 5u);
  for (std::size_t system = 0; system < kBatch; ++system) {
    CHECK(check_successful_peer(batch, recovered, system, 5u) == 0);
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

  /* Reaching the configured SCC bound is a device-published peer result, not
   * a host submission failure. Terminal publication must preserve its
   * provenance while withholding every requested floating-point result and
   * refusing to mint a warm checkpoint for the unconverged state. */
  CHECK(cache.refresh_numerical_async(inputs.view, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(cache.execute_inference_async(Gfn2CudaSccStartMode::kFresh, error) ==
        XTBLOOM_STATUS_SUCCESS);
  Snapshot snapshot;
  CHECK(download(cache.identity(), stream, snapshot) == 0);
  CHECK(snapshot.epoch == 2u);
  CHECK(snapshot.committed[0] == 2u);
  CHECK(snapshot.eligible[0] == 1u);
  CHECK(snapshot.statuses[0] == XTBLOOM_STATUS_SCC_NOT_CONVERGED);
  CHECK(snapshot.converged[0] == 0u);
  CHECK(snapshot.iterations[0] == 1);
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

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    /* CUDA-enabled packages must remain testable on CPU-only CI runners. */
    std::puts("cuda_gfn2_runtime_graph_test: SKIP (no CUDA device)");
    return 0;
  }

  std::int32_t device_id = -1;
  CUDA_CHECK(cudaGetDevice(&device_id));

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
