#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_inference_publication.cuh"
#include "backends/cuda/gfn2_scc_potential.cuh"
#include "runtime/backend.hpp"
#include "runtime/gfn2_cuda_execution.hpp"
#include "tests/support/gfn2_scc_test_case.hpp"
#include "xtbloom/xtbloom.h"

#define CHECK(condition)                                                                 \
  do {                                                                                   \
    if (!(condition)) {                                                                  \
      std::fprintf(stderr, "CUDA runtime-owner check failed at line %d: %s\n", __LINE__, \
                   #condition);                                                          \
      return __LINE__;                                                                   \
    }                                                                                    \
  } while (false)

#define CHECK_STATUS(expression, expected, error)                                           \
  do {                                                                                      \
    const xtbloom_status_t observed_status = (expression);                                  \
    if (observed_status != (expected)) {                                                    \
      std::fprintf(stderr,                                                                  \
                   "CUDA runtime-owner status check failed at line %d: status=%d "          \
                   "expected=%d error=%s\n",                                                \
                   __LINE__, static_cast<int>(observed_status), static_cast<int>(expected), \
                   (error).c_str());                                                        \
      return __LINE__;                                                                      \
    }                                                                                       \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using xtbloom::detail::arm_gfn2_cuda_execution_test_fault;
using xtbloom::detail::Context;
using xtbloom::detail::enqueue_restricted_gfn2_cuda_plan;
using xtbloom::detail::execute_restricted_gfn2_cuda;
using xtbloom::detail::execute_restricted_gfn2_cuda_plan;
using xtbloom::detail::gfn2_cuda_execution_test_stats;
using xtbloom::detail::Gfn2CudaExecutionCache;
using xtbloom::detail::Gfn2CudaExecutionIdentity;
using xtbloom::detail::Gfn2CudaExecutionTestFault;
using xtbloom::detail::Gfn2CudaExecutionTestStats;
using xtbloom::detail::Gfn2CudaNativeLatticeTestIdentity;
using xtbloom::detail::Gfn2CudaNumericalInputView;
using xtbloom::detail::Gfn2CudaSccStartMode;
using xtbloom::detail::RequestCompletionResult;
using xtbloom::detail::RequestSubmission;
using xtbloom::detail::reset_gfn2_cuda_execution_test_state;
using xtbloom::detail::cuda::Gfn2InferencePublicationDeviceResults;
using xtbloom::detail::cuda::Gfn2InferencePublicationPlanError;
using xtbloom::detail::cuda::Gfn2InferencePublicationSystemError;
using xtbloom::detail::cuda::Gfn2SccPotentialComponent;
using xtbloom::test::gfn2::HostSccCase;
using xtbloom::test::gfn2::HostSccCaseOptions;
using xtbloom::test::gfn2::SmallSystemKind;

template <typename T>
xtbloom_const_buffer_t host_buffer(const std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), XTBLOOM_MEMORY_HOST,
          0u};
}

template <typename T>
xtbloom_buffer_t mutable_host_buffer(std::vector<T>& values) noexcept {
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

  cudaError_t upload(const std::vector<T>& values, cudaStream_t stream) {
    if (values.size() != count_) {
      if (data_ != nullptr) {
        (void)cudaFree(data_);
        data_ = nullptr;
      }
      count_ = values.size();
      cudaError_t status = count_ == 0u
                               ? cudaSuccess
                               : cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T));
      if (status != cudaSuccess) return status;
    }
    return count_ == 0u ? cudaSuccess
                        : cudaMemcpyAsync(data_, values.data(), count_ * sizeof(T),
                                          cudaMemcpyHostToDevice, stream);
  }

  xtbloom_const_buffer_t view() const noexcept {
    return {data_, count_ * sizeof(T), XTBLOOM_MEMORY_CUDA_DEVICE, 0u};
  }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

template <typename T>
class PinnedHostBuffer {
 public:
  PinnedHostBuffer() = default;
  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
  ~PinnedHostBuffer() { reset(); }

  cudaError_t assign(const std::vector<T>& values) {
    reset();
    count_ = values.size();
    if (count_ == 0u) return cudaSuccess;
    cudaError_t status = cudaMallocHost(reinterpret_cast<void**>(&data_), count_ * sizeof(T));
    if (status == cudaSuccess) {
      std::memcpy(data_, values.data(), count_ * sizeof(T));
    }
    return status;
  }

  void reset() noexcept {
    if (data_ != nullptr) (void)cudaFreeHost(data_);
    data_ = nullptr;
    count_ = 0u;
  }

  T* data() noexcept { return data_; }
  std::size_t size() const noexcept { return count_; }

  xtbloom_const_buffer_t view() const noexcept {
    return {data_, count_ * sizeof(T), XTBLOOM_MEMORY_HOST, 0u};
  }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

__global__ void hold_owner_stream_kernel(unsigned long long clock_cycles) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const unsigned long long start = clock64();
  while (clock64() - start < clock_cycles) {
    __nanosleep(1000u);
  }
}

struct RefreshSnapshot {
  std::uint64_t epoch = 0u;
  std::vector<std::uint64_t> committed;
  std::vector<std::uint64_t> factors;
  std::vector<std::uint32_t> factor_statuses;
  std::vector<std::uint8_t> eligible;
};

int download_refresh_snapshot(const Gfn2CudaExecutionIdentity& identity, cudaStream_t stream,
                              RefreshSnapshot& snapshot) {
  const std::size_t batch = static_cast<std::size_t>(identity.batch_size);
  snapshot.committed.resize(batch);
  snapshot.factors.resize(batch);
  snapshot.factor_statuses.resize(batch);
  snapshot.eligible.resize(batch);
  CUDA_CHECK(cudaMemcpyAsync(&snapshot.epoch,
                             reinterpret_cast<const void*>(identity.numerical_epoch),
                             sizeof(snapshot.epoch), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(snapshot.committed.data(),
                             reinterpret_cast<const void*>(identity.committed_generations),
                             batch * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(snapshot.factors.data(),
                             reinterpret_cast<const void*>(identity.overlap_factor_generations),
                             batch * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(snapshot.factor_statuses.data(),
                             reinterpret_cast<const void*>(identity.overlap_factor_statuses),
                             batch * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(snapshot.eligible.data(),
                             reinterpret_cast<const void*>(identity.numerical_eligible_mask),
                             batch * sizeof(std::uint8_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return 0;
}

struct InferenceSnapshot {
  std::vector<double> energies;
  std::vector<double> qm_forces;
  std::vector<double> atomic_charges;
  std::vector<double> point_forces;
  std::vector<std::int32_t> iterations;
  std::vector<std::uint8_t> converged;
  std::vector<xtbloom_status_t> statuses;
  std::uint64_t publication_epoch = 0u;
  std::vector<std::uint32_t> publication_system_errors;
  std::uint32_t publication_plan_error = 0u;
  std::vector<std::uint64_t> warm_generations;
};

struct RequestStateSnapshot {
  RefreshSnapshot refresh;
  InferenceSnapshot inference;
  std::vector<std::byte> scc_state;
  std::vector<std::byte> scc_workspace;
  std::uint64_t numerical_body_count = 0u;
};

int download_inference_snapshot(const Gfn2CudaExecutionIdentity& identity, cudaStream_t stream,
                                bool force_mode, InferenceSnapshot& snapshot) {
  const std::size_t batch = static_cast<std::size_t>(identity.batch_size);
  const std::size_t coordinates = static_cast<std::size_t>(identity.total_atoms * 3);
  const std::size_t point_coordinates = static_cast<std::size_t>(identity.total_point_charges * 3);
  snapshot.energies.resize(batch);
  snapshot.qm_forces.resize(force_mode ? coordinates : 0u);
  snapshot.atomic_charges.resize(force_mode ? static_cast<std::size_t>(identity.total_atoms) : 0u);
  snapshot.point_forces.resize(force_mode ? point_coordinates : 0u);
  snapshot.iterations.resize(batch);
  snapshot.converged.resize(batch);
  snapshot.statuses.resize(batch);
  snapshot.publication_system_errors.resize(batch);
  snapshot.warm_generations.resize(batch);
  CUDA_CHECK(cudaMemcpyAsync(snapshot.energies.data(),
                             reinterpret_cast<const void*>(identity.inference_energies),
                             batch * sizeof(double), cudaMemcpyDeviceToHost, stream));
  if (!snapshot.qm_forces.empty()) {
    CUDA_CHECK(cudaMemcpyAsync(
        snapshot.qm_forces.data(), reinterpret_cast<const void*>(identity.inference_qm_forces),
        snapshot.qm_forces.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
  }
  if (!snapshot.atomic_charges.empty()) {
    CUDA_CHECK(cudaMemcpyAsync(snapshot.atomic_charges.data(),
                               reinterpret_cast<const void*>(identity.inference_atomic_charges),
                               snapshot.atomic_charges.size() * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
  }
  if (!snapshot.point_forces.empty()) {
    CUDA_CHECK(cudaMemcpyAsync(snapshot.point_forces.data(),
                               reinterpret_cast<const void*>(identity.inference_point_forces),
                               snapshot.point_forces.size() * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
  }
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
      cudaMemcpyAsync(snapshot.publication_system_errors.data(),
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

int download_request_state_snapshot(const Gfn2CudaExecutionIdentity& identity, cudaStream_t stream,
                                    bool force_mode, RequestStateSnapshot& snapshot) {
  CHECK(download_refresh_snapshot(identity, stream, snapshot.refresh) == 0);
  CHECK(download_inference_snapshot(identity, stream, force_mode, snapshot.inference) == 0);
  snapshot.scc_state.resize(identity.scc_state_image_bytes);
  snapshot.scc_workspace.resize(identity.scc_workspace_image_bytes);
  if (!snapshot.scc_state.empty()) {
    CUDA_CHECK(cudaMemcpyAsync(snapshot.scc_state.data(),
                               reinterpret_cast<const void*>(identity.scc_state_image),
                               snapshot.scc_state.size(), cudaMemcpyDeviceToHost, stream));
  }
  if (!snapshot.scc_workspace.empty()) {
    CUDA_CHECK(cudaMemcpyAsync(snapshot.scc_workspace.data(),
                               reinterpret_cast<const void*>(identity.scc_workspace_image),
                               snapshot.scc_workspace.size(), cudaMemcpyDeviceToHost, stream));
  }
  if (identity.scc_loop_numerical_body_count != 0u) {
    CUDA_CHECK(
        cudaMemcpyAsync(&snapshot.numerical_body_count,
                        reinterpret_cast<const void*>(identity.scc_loop_numerical_body_count),
                        sizeof(snapshot.numerical_body_count), cudaMemcpyDeviceToHost, stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return 0;
}

bool same_refresh_snapshot(const RefreshSnapshot& first, const RefreshSnapshot& second) {
  return first.epoch == second.epoch && first.committed == second.committed &&
         first.factors == second.factors && first.factor_statuses == second.factor_statuses &&
         first.eligible == second.eligible;
}

bool same_inference_without_warm(const InferenceSnapshot& first, const InferenceSnapshot& second) {
  return first.energies == second.energies && first.qm_forces == second.qm_forces &&
         first.atomic_charges == second.atomic_charges &&
         first.point_forces == second.point_forces && first.iterations == second.iterations &&
         first.converged == second.converged && first.statuses == second.statuses &&
         first.publication_epoch == second.publication_epoch &&
         first.publication_system_errors == second.publication_system_errors &&
         first.publication_plan_error == second.publication_plan_error;
}

struct PublicHostBatch {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int32_t> spin_channels;
  std::vector<std::int64_t> point_offsets;
  std::vector<double> point_positions;
  std::vector<double> point_values;
  std::vector<double> point_gammas;
  std::vector<double> periodic_shifts;
  std::vector<std::int64_t> response_offsets;
  std::vector<double> response_matrix;
  std::vector<double> cell_matrices;
  std::vector<std::int32_t> periodic_axes;
  xtbloom_batch_t descriptor{};

  void bind() noexcept {
    descriptor = {};
    descriptor.struct_size =
        !cell_matrices.empty()
            ? XTBLOOM_BATCH_V4_SIZE
            : (spin_channels.empty() ? XTBLOOM_BATCH_V1_SIZE : XTBLOOM_BATCH_V2_SIZE);
    descriptor.api_version = XTBLOOM_API_VERSION;
    descriptor.batch_size = static_cast<std::int64_t>(molecular_charges.size());
    descriptor.total_atoms = static_cast<std::int64_t>(atomic_numbers.size());
    descriptor.total_point_charges = static_cast<std::int64_t>(point_values.size());
    descriptor.total_charge_response_elements =
        response_matrix.empty() ? 0 : static_cast<std::int64_t>(response_matrix.size());
    descriptor.atom_offsets = host_buffer(atom_offsets);
    descriptor.atomic_numbers = host_buffer(atomic_numbers);
    descriptor.positions = host_buffer(positions);
    descriptor.molecular_charges = host_buffer(molecular_charges);
    descriptor.unpaired_electrons = host_buffer(unpaired_electrons);
    if (!spin_channels.empty()) descriptor.spin_channels = host_buffer(spin_channels);
    descriptor.point_charge_offsets = host_buffer(point_offsets);
    descriptor.point_charge_positions = host_buffer(point_positions);
    descriptor.point_charge_values = host_buffer(point_values);
    descriptor.point_charge_gammas = host_buffer(point_gammas);
    descriptor.atomic_potential_shifts = host_buffer(periodic_shifts);
    descriptor.charge_response_offsets = host_buffer(response_offsets);
    descriptor.charge_response_matrix = host_buffer(response_matrix);
    if (!cell_matrices.empty()) {
      descriptor.cell_matrices = host_buffer(cell_matrices);
      descriptor.periodic_axes = host_buffer(periodic_axes);
    }
  }

  static PublicHostBatch from_host(const HostSccCase& host, bool periodic_enabled) {
    PublicHostBatch batch;
    batch.atom_offsets = host.atom_offsets();
    batch.atomic_numbers = host.atomic_numbers();
    batch.positions = host.positions();
    batch.molecular_charges = host.molecular_charges();
    batch.unpaired_electrons = host.unpaired_electrons();
    batch.spin_channels = host.spin_channels();
    batch.point_offsets = host.point_charge_offsets();
    batch.point_positions = host.point_charge_positions();
    batch.point_values = host.point_charge_charges();
    batch.point_gammas = host.point_charge_hardnesses();
    if (periodic_enabled) {
      batch.periodic_shifts = host.periodic_shifts();
      batch.response_matrix = host.periodic_response_matrices();
      batch.response_offsets.assign(static_cast<std::size_t>(host.batch_size() + 1), 0);
      for (std::int64_t system = 0; system < host.batch_size(); ++system) {
        const std::int64_t atoms = host.atom_offsets()[static_cast<std::size_t>(system + 1)] -
                                   host.atom_offsets()[static_cast<std::size_t>(system)];
        batch.response_offsets[static_cast<std::size_t>(system + 1)] =
            batch.response_offsets[static_cast<std::size_t>(system)] + atoms * atoms;
      }
    }
    batch.bind();
    return batch;
  }
};

HostSccCaseOptions case_options(std::int64_t batch_size, bool all_optional) {
  HostSccCaseOptions options{};
  constexpr std::array<SmallSystemKind, 4> systems{SmallSystemKind::kH2, SmallSystemKind::kHe,
                                                   SmallSystemKind::kLiH, SmallSystemKind::kCH2};
  options.systems.clear();
  options.systems.reserve(static_cast<std::size_t>(batch_size));
  for (std::int64_t system = 0; system < batch_size; ++system) {
    options.systems.push_back(systems[static_cast<std::size_t>(system) % systems.size()]);
  }
  options.maximum_iterations = 8u;
  options.mixer_history = 8;
  options.enable_d4 = all_optional;
  options.enable_explicit_point_charges = all_optional;
  options.enable_periodic_embedding = all_optional;
  return options;
}

HostSccCaseOptions homogeneous_case_options(std::int64_t batch_size, SmallSystemKind system,
                                            bool enable_d4, bool enable_points,
                                            bool enable_periodic) {
  HostSccCaseOptions options{};
  options.systems.assign(static_cast<std::size_t>(batch_size), system);
  options.maximum_iterations = 8u;
  options.mixer_history = 8;
  options.enable_d4 = enable_d4;
  options.enable_explicit_point_charges = enable_points;
  options.enable_periodic_embedding = enable_periodic;
  return options;
}

xtbloom_compute_options_t compute_options(bool force_mode = true) noexcept {
  xtbloom_compute_options_t options{};
  options.struct_size = XTBLOOM_COMPUTE_OPTIONS_V1_SIZE;
  options.api_version = XTBLOOM_API_VERSION;
  options.model = XTBLOOM_MODEL_GFN2_XTB;
  options.flags = XTBLOOM_COMPUTE_ENERGY;
  if (force_mode) {
    options.flags |= XTBLOOM_COMPUTE_FORCES | XTBLOOM_COMPUTE_ATOMIC_CHARGES |
                     XTBLOOM_COMPUTE_POINT_CHARGE_FORCES;
  }
  options.max_scc_iterations = 8;
  options.charge_tolerance = 1.0e-10;
  options.energy_tolerance = 1.0e-8;
  options.electronic_temperature = 0.0;
  return options;
}

bool same_identity(const Gfn2CudaExecutionIdentity& first,
                   const Gfn2CudaExecutionIdentity& second) noexcept {
  return first.topology_fingerprint == second.topology_fingerprint &&
         first.plan_token == second.plan_token &&
         first.iteration_layout_fingerprint == second.iteration_layout_fingerprint &&
         first.enabled_component_mask == second.enabled_component_mask &&
         first.scc_binding_ready == second.scc_binding_ready &&
         first.energy_force_binding_ready == second.energy_force_binding_ready &&
         first.force_mode_ready == second.force_mode_ready &&
         first.energy_force_smoke_ready == second.energy_force_smoke_ready &&
         first.numerical_refresh_ready == second.numerical_refresh_ready &&
         first.inference_ready == second.inference_ready && first.batch_size == second.batch_size &&
         first.total_atoms == second.total_atoms && first.total_shells == second.total_shells &&
         first.total_orbitals == second.total_orbitals &&
         first.total_point_charges == second.total_point_charges &&
         first.solver_handle == second.solver_handle &&
         first.solver_parameters == second.solver_parameters &&
         first.blas_handle == second.blas_handle && first.topology_owner == second.topology_owner &&
         first.inputs_owner == second.inputs_owner &&
         first.eigensolver_owner == second.eigensolver_owner &&
         first.initializer_owner == second.initializer_owner &&
         first.scc_binding == second.scc_binding &&
         first.scc_loop_active_count == second.scc_loop_active_count &&
         first.scc_loop_numerical_body_count == second.scc_loop_numerical_body_count &&
         first.scc_loop_device_launch_error == second.scc_loop_device_launch_error &&
         first.energy_force_descriptors == second.energy_force_descriptors &&
         first.topology_arena == second.topology_arena && first.input_arena == second.input_arena &&
         first.iteration_arena == second.iteration_arena &&
         first.eigensolver_setup_arena == second.eigensolver_setup_arena &&
         first.provider_host_workspace == second.provider_host_workspace &&
         first.native_lattice_host_staging == second.native_lattice_host_staging &&
         first.force_immutable_arena == second.force_immutable_arena &&
         first.force_execution_arena == second.force_execution_arena &&
         first.numerical_refresh_arena == second.numerical_refresh_arena &&
         first.numerical_refresh_binding == second.numerical_refresh_binding &&
         first.numerical_epoch == second.numerical_epoch &&
         first.committed_generations == second.committed_generations &&
         first.numerical_eligible_mask == second.numerical_eligible_mask &&
         first.overlap_factor_generations == second.overlap_factor_generations &&
         first.overlap_factor_statuses == second.overlap_factor_statuses &&
         first.inference_arena == second.inference_arena &&
         first.inference_epoch_consumer == second.inference_epoch_consumer &&
         first.inference_results == second.inference_results &&
         first.inference_energies == second.inference_energies &&
         first.inference_qm_forces == second.inference_qm_forces &&
         first.inference_atomic_charges == second.inference_atomic_charges &&
         first.inference_point_forces == second.inference_point_forces &&
         first.inference_iterations == second.inference_iterations &&
         first.inference_converged == second.inference_converged &&
         first.inference_system_statuses == second.inference_system_statuses &&
         first.inference_publication_epoch_snapshot ==
             second.inference_publication_epoch_snapshot &&
         first.inference_publication_system_errors == second.inference_publication_system_errors &&
         first.inference_publication_plan_error == second.inference_publication_plan_error &&
         first.warm_checkpoint_generations == second.warm_checkpoint_generations &&
         first.scc_state_image == second.scc_state_image &&
         first.scc_state_image_bytes == second.scc_state_image_bytes &&
         first.scc_workspace_image == second.scc_workspace_image &&
         first.scc_workspace_image_bytes == second.scc_workspace_image_bytes &&
         first.topology_arena_bytes == second.topology_arena_bytes &&
         first.input_arena_bytes == second.input_arena_bytes &&
         first.iteration_arena_bytes == second.iteration_arena_bytes &&
         first.eigensolver_setup_arena_bytes == second.eigensolver_setup_arena_bytes &&
         first.provider_host_workspace_bytes == second.provider_host_workspace_bytes &&
         first.force_immutable_arena_bytes == second.force_immutable_arena_bytes &&
         first.force_execution_arena_bytes == second.force_execution_arena_bytes &&
         first.numerical_refresh_arena_bytes == second.numerical_refresh_arena_bytes &&
         first.inference_arena_bytes == second.inference_arena_bytes &&
         first.numerical_host_staging_arena_bytes == second.numerical_host_staging_arena_bytes &&
         first.public_result_device_arena_bytes == second.public_result_device_arena_bytes &&
         first.public_result_host_arena_bytes == second.public_result_host_arena_bytes &&
         first.candidate_validation_arena_bytes == second.candidate_validation_arena_bytes &&
         first.native_lattice_host_staging_bytes == second.native_lattice_host_staging_bytes &&
         first.topology_staging_host_bytes == second.topology_staging_host_bytes &&
         first.topology_staging_device_bytes == second.topology_staging_device_bytes &&
         first.runtime_owner_host_bytes == second.runtime_owner_host_bytes &&
         first.host_plans_bytes == second.host_plans_bytes &&
         first.topology_setup_host_bytes == second.topology_setup_host_bytes &&
         first.inputs_setup_host_bytes == second.inputs_setup_host_bytes &&
         first.eigensolver_setup_host_bytes == second.eigensolver_setup_host_bytes &&
         first.initializer_host_bytes == second.initializer_host_bytes &&
         first.initializer_device_checkpoint_bytes == second.initializer_device_checkpoint_bytes &&
         first.scc_loop_device_control_bytes == second.scc_loop_device_control_bytes &&
         first.retained_host_workspace_bytes == second.retained_host_workspace_bytes &&
         first.retained_device_workspace_bytes == second.retained_device_workspace_bytes;
}

int validate_identity(const Gfn2CudaExecutionIdentity& identity, std::int64_t batch_size,
                      bool expect_d4, bool expect_points, bool expect_periodic,
                      bool force_mode = true) {
  CHECK(identity.topology_fingerprint != 0u);
  CHECK(identity.plan_token != 0u);
  CHECK(identity.iteration_layout_fingerprint != 0u);
  CHECK(identity.batch_size == batch_size);
  CHECK(identity.total_atoms > 0);
  CHECK(identity.total_shells > 0);
  CHECK(identity.total_orbitals > 0);
  CHECK(identity.scc_binding_ready == 1u);
  CHECK(identity.energy_force_binding_ready == 1u);
  CHECK(identity.force_mode_ready == (force_mode ? 1u : 0u));
  CHECK(identity.energy_force_smoke_ready == 1u);
  CHECK(identity.numerical_refresh_ready == 1u);
  CHECK(identity.inference_ready == 1u);
  CHECK(identity.warm_checkpoint_ready == 0u);
  CHECK(identity.solver_handle != 0u);
  CHECK(identity.solver_parameters != 0u);
  CHECK(identity.blas_handle != 0u);
  CHECK(identity.topology_owner != 0u);
  CHECK(identity.inputs_owner != 0u);
  CHECK(identity.eigensolver_owner != 0u);
  CHECK(identity.initializer_owner != 0u);
  CHECK(identity.scc_binding != 0u);
  CHECK(identity.energy_force_descriptors != 0u);
  CHECK(identity.topology_arena % 256u == 0u);
  CHECK(identity.input_arena % 256u == 0u);
  CHECK(identity.iteration_arena % 256u == 0u);
  CHECK(identity.eigensolver_setup_arena % 256u == 0u);
  CHECK(identity.force_immutable_arena % 256u == 0u);
  CHECK(identity.force_execution_arena % 256u == 0u);
  CHECK(identity.numerical_refresh_arena % 256u == 0u);
  CHECK(identity.inference_arena % 256u == 0u);
  CHECK(identity.topology_arena_bytes > 0u);
  CHECK(identity.input_arena_bytes > 0u);
  CHECK(identity.iteration_arena_bytes > 0u);
  CHECK(identity.eigensolver_setup_arena_bytes > 0u);
  CHECK(identity.force_immutable_arena_bytes > 0u || !force_mode);
  CHECK((identity.force_immutable_arena == 0u) == !force_mode);
  CHECK(identity.force_execution_arena_bytes > 0u);
  CHECK(identity.numerical_refresh_arena_bytes > 0u);
  CHECK(identity.inference_arena_bytes > 0u);
  CHECK(identity.numerical_host_staging_arena_bytes > 0u);
  CHECK(identity.public_result_device_arena_bytes > 0u);
  CHECK(identity.public_result_host_arena_bytes > 0u);
  CHECK(identity.candidate_validation_arena_bytes > 0u);
  CHECK((identity.native_lattice_host_staging == 0u) ==
        (identity.native_lattice_host_staging_bytes == 0u));
  CHECK(identity.topology_staging_host_bytes > 0u);
  CHECK(identity.topology_staging_device_bytes > 0u);
  CHECK(identity.runtime_owner_host_bytes > 0u);
  CHECK(identity.host_plans_bytes > 0u);
  CHECK(identity.topology_setup_host_bytes > 0u);
  CHECK(identity.inputs_setup_host_bytes > 0u);
  CHECK(identity.eigensolver_setup_host_bytes > 0u);
  CHECK(identity.initializer_host_bytes > 0u);
  CHECK(identity.initializer_device_checkpoint_bytes > 0u);
  CHECK(identity.retained_host_workspace_bytes ==
        identity.provider_host_workspace_bytes + identity.numerical_host_staging_arena_bytes +
            identity.public_result_host_arena_bytes + identity.candidate_validation_arena_bytes +
            identity.native_lattice_host_staging_bytes + identity.topology_staging_host_bytes +
            identity.runtime_owner_host_bytes + identity.host_plans_bytes +
            identity.topology_setup_host_bytes + identity.inputs_setup_host_bytes +
            identity.eigensolver_setup_host_bytes + identity.initializer_host_bytes);
  CHECK(identity.retained_device_workspace_bytes ==
        identity.topology_arena_bytes + identity.input_arena_bytes +
            identity.iteration_arena_bytes + identity.eigensolver_setup_arena_bytes +
            identity.force_immutable_arena_bytes + identity.force_execution_arena_bytes +
            identity.numerical_refresh_arena_bytes + identity.inference_arena_bytes +
            identity.public_result_device_arena_bytes + identity.topology_staging_device_bytes +
            identity.initializer_device_checkpoint_bytes + identity.scc_loop_device_control_bytes);
  CHECK(identity.numerical_refresh_binding != 0u);
  CHECK(identity.numerical_epoch != 0u);
  CHECK(identity.committed_generations != 0u);
  CHECK(identity.numerical_eligible_mask != 0u);
  CHECK(identity.overlap_factor_generations != 0u);
  CHECK(identity.overlap_factor_statuses != 0u);
  CHECK(identity.inference_epoch_consumer != 0u);
  CHECK(identity.inference_results != 0u);
  CHECK(identity.inference_energies != 0u);
  CHECK((identity.inference_qm_forces != 0u) == force_mode);
  CHECK((identity.inference_atomic_charges != 0u) == force_mode);
  CHECK((identity.inference_point_forces != 0u) == (force_mode && expect_points));
  CHECK(identity.inference_iterations != 0u);
  CHECK(identity.inference_converged != 0u);
  CHECK(identity.inference_system_statuses != 0u);
  CHECK(identity.inference_publication_epoch_snapshot != 0u);
  CHECK(identity.inference_publication_system_errors != 0u);
  CHECK(identity.inference_publication_plan_error != 0u);
  CHECK(identity.warm_checkpoint_generations != 0u);

  constexpr std::uint32_t required_base =
      static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES2) |
      static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES3) |
      static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kAES2);
  constexpr std::uint32_t d4_component =
      static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kD4TwoBody);
  constexpr std::uint32_t point_component =
      static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kExplicitPointCharge);
  constexpr std::uint32_t periodic_component =
      static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kPeriodicEmbedding);
  CHECK((identity.enabled_component_mask & required_base) == required_base);
  CHECK(((identity.enabled_component_mask & d4_component) != 0u) == expect_d4);
  CHECK(((identity.enabled_component_mask & point_component) != 0u) == expect_points);
  CHECK(((identity.enabled_component_mask & periodic_component) != 0u) == expect_periodic);
  if (expect_points) {
    CHECK(identity.total_point_charges == batch_size);
  } else {
    CHECK(identity.total_point_charges == 0);
  }

  if (identity.provider_host_workspace_bytes != 0u) {
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(
        &attributes, reinterpret_cast<const void*>(identity.provider_host_workspace)));
#if CUDART_VERSION >= 10000
    CHECK(attributes.type == cudaMemoryTypeHost);
#else
    CHECK(attributes.memoryType == cudaMemoryTypeHost);
#endif
  }
  return 0;
}

int test_ragged_runtime_shapes(cudaStream_t stream, std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  xtbloom_compute_options_t options = compute_options();
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    HostSccCase host;
    std::string error;
    CHECK(HostSccCase::create(case_options(batch_size, true), host, error) ==
          XTBLOOM_STATUS_SUCCESS);
    PublicHostBatch batch = PublicHostBatch::from_host(host, true);
    bool reused = true;
    const xtbloom_status_t status = cache.prepare_host(batch.descriptor, options, reused, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      std::fprintf(stderr, "batch %lld runtime setup failed: status=%d error=%s\n",
                   static_cast<long long>(batch_size), status, error.c_str());
    }
    CHECK(status == XTBLOOM_STATUS_SUCCESS);
    CHECK(!reused);
    CHECK(cache.valid());
    const Gfn2CudaExecutionIdentity initial = cache.identity();
    CHECK(validate_identity(initial, batch_size, true, true, true) == 0);
    batch.positions[0] += 0.01;
    batch.bind();
    CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(reused);
    CHECK(same_identity(initial, cache.identity()));
    RefreshSnapshot snapshot;
    CHECK(download_refresh_snapshot(cache.identity(), stream, snapshot) == 0);
    CHECK(snapshot.epoch == 2u);
    for (std::size_t system = 0; system < snapshot.committed.size(); ++system) {
      CHECK(snapshot.committed[system] == 2u);
      CHECK(snapshot.factors[system] == 2u);
      CHECK(snapshot.factor_statuses[system] == 0u);
      CHECK(snapshot.eligible[system] == 1u);
    }
  }
  return 0;
}

int test_topology_only_seed_factor_is_unpublished(cudaStream_t stream, std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(homogeneous_case_options(1, SmallSystemKind::kH2, true, false, false),
                            host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, false);
  const xtbloom_compute_options_t options = compute_options(false);

  CHECK(cache.prepare_topology_only(batch.descriptor, options, error) == XTBLOOM_STATUS_SUCCESS);
  RefreshSnapshot unpublished;
  CHECK(download_refresh_snapshot(cache.identity(), stream, unpublished) == 0);
  CHECK(unpublished.epoch == 0u);
  CHECK(unpublished.committed[0] == 0u);
  CHECK(unpublished.factors[0] == 0u);
  CHECK(unpublished.eligible[0] == 0u);

  Gfn2CudaNumericalInputView numerical{};
  numerical.positions = batch.descriptor.positions;
  CHECK(cache.refresh_numerical_async(numerical, error) == XTBLOOM_STATUS_SUCCESS);
  RefreshSnapshot published;
  CHECK(download_refresh_snapshot(cache.identity(), stream, published) == 0);
  CHECK(published.epoch == 1u);
  CHECK(published.committed[0] == 1u);
  CHECK(published.factors[0] == 1u);
  CHECK(published.factor_statuses[0] == 0u);
  CHECK(published.eligible[0] == 1u);
  return 0;
}

int test_reuse_and_transactions(cudaStream_t stream, std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(case_options(8, true), host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, true);
  xtbloom_compute_options_t options = compute_options();
  bool reused = true;
  const xtbloom_status_t initial_status =
      cache.prepare_host(batch.descriptor, options, reused, error);
  if (initial_status != XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "reuse runtime setup failed: status=%d error=%s\n", initial_status,
                 error.c_str());
  }
  CHECK(initial_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  const Gfn2CudaExecutionIdentity initial = cache.identity();
  CHECK(validate_identity(initial, 8, true, true, true) == 0);

  /* Coordinates and all numerical QM/MM fields are intentionally excluded
   * from the topology key and refresh through stable runtime-owned staging. */
  batch.positions[0] += 0.037;
  batch.point_values[0] -= 0.05;
  batch.periodic_shifts[0] += 0.02;
  batch.bind();
  CHECK_STATUS(cache.prepare_host(batch.descriptor, options, reused, error),
               XTBLOOM_STATUS_SUCCESS, error);
  CHECK(reused);
  CHECK(same_identity(initial, cache.identity()));

  /* Descriptor and semantic failures must not evict the working cache. */
  const std::size_t good_atomic_number_bytes = batch.descriptor.atomic_numbers.size_bytes;
  batch.descriptor.atomic_numbers.size_bytes = good_atomic_number_bytes - 1u;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(!reused);
  CHECK(same_identity(initial, cache.identity()));
  batch.descriptor.atomic_numbers.size_bytes = good_atomic_number_bytes;

  /* A non-null zero-length response matrix must not keep an existing periodic
   * topology alive while omitting its dense count and offsets. This malformed
   * internal descriptor is rejected even when public validation is bypassed. */
  batch.descriptor.atomic_potential_shifts = {};
  batch.descriptor.total_charge_response_elements = 0;
  batch.descriptor.charge_response_offsets = {};
  batch.descriptor.charge_response_matrix = {batch.response_matrix.data(), 0u, XTBLOOM_MEMORY_HOST,
                                             0u};
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(!reused);
  CHECK(same_identity(initial, cache.identity()));
  batch.bind();

  /* Force vector::resize to reject the staging extent before it can read the
   * deliberately tiny source. The call must translate the C++ exception into
   * a status and preserve the published cache rather than letting it escape. */
  const std::size_t vector_max = std::vector<std::int64_t>{}.max_size();
  CHECK(vector_max < static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max() - 1));
  batch.descriptor.batch_size = static_cast<std::int64_t>(vector_max + 1u);
  batch.descriptor.atom_offsets.size_bytes = std::numeric_limits<std::size_t>::max();
  const xtbloom_status_t oversized_status =
      cache.prepare_host(batch.descriptor, options, reused, error);
  CHECK(oversized_status == XTBLOOM_STATUS_ALLOCATION_FAILED ||
        oversized_status == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(!reused);
  CHECK(!error.empty());
  CHECK(same_identity(initial, cache.identity()));
  batch.bind();

  batch.spin_channels[3] = 3;
  batch.bind();
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(same_identity(initial, cache.identity()));
  batch.spin_channels[3] = 1;
  batch.bind();

  /* A compute-policy change is topology scoped. Handles remain context scoped
   * while owners, tokens, arenas, and the layout transaction are replaced. */
  ++options.max_scc_iterations;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  const Gfn2CudaExecutionIdentity replaced = cache.identity();
  CHECK(replaced.plan_token != initial.plan_token);
  CHECK(replaced.topology_fingerprint != initial.topology_fingerprint);
  CHECK(replaced.solver_handle == initial.solver_handle);
  CHECK(replaced.solver_parameters == initial.solver_parameters);
  CHECK(replaced.blas_handle == initial.blas_handle);
  CHECK(replaced.topology_owner != initial.topology_owner);
  CHECK(replaced.iteration_arena != initial.iteration_arena);

  /* A candidate that passes descriptor copying but cannot define a physical
   * wavefunction must fail before publication and retain the replacement. */
  batch.molecular_charges[0] = 1000.0;
  batch.bind();
  const xtbloom_status_t invalid_physics =
      cache.prepare_host(batch.descriptor, options, reused, error);
  CHECK(invalid_physics != XTBLOOM_STATUS_SUCCESS);
  CHECK(same_identity(replaced, cache.identity()));
  batch.molecular_charges[0] = host.molecular_charges()[0];
  batch.bind();

  /* ABI-v3 policies are setup/arena/Graph identity, not merely public tags.
   * Direct runtime entry points fail closed on hostile complete suffixes, and
   * every valid policy change replaces the topology-scoped owners atomically. */
  xtbloom_compute_options_t hostile = options;
  hostile.struct_size = XTBLOOM_COMPUTE_OPTIONS_V3_SIZE;
  hostile.scc_mixer = XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN;
  hostile.scc_mixer_history = 8;
  hostile.scc_mixer_damping = 0.4;
  hostile.determinism = XTBLOOM_DETERMINISM_DEFAULT;
  hostile.scc_mixer_history = 65;
  CHECK(cache.prepare_host(batch.descriptor, hostile, reused, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(same_identity(replaced, cache.identity()));

  xtbloom_compute_options_t short_history = hostile;
  short_history.scc_mixer_history = 4;
  CHECK(cache.prepare_host(batch.descriptor, short_history, reused, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  const Gfn2CudaExecutionIdentity history_replaced = cache.identity();
  CHECK(history_replaced.plan_token != replaced.plan_token);
  CHECK(history_replaced.topology_fingerprint != replaced.topology_fingerprint);
  CHECK(history_replaced.iteration_layout_fingerprint != replaced.iteration_layout_fingerprint);
  CHECK(history_replaced.iteration_arena != replaced.iteration_arena);
  CHECK(history_replaced.scc_loop_owner != replaced.scc_loop_owner);
  CHECK(history_replaced.solver_handle == replaced.solver_handle);
  CHECK(history_replaced.blas_handle == replaced.blas_handle);
  CHECK(cache.prepare_host(batch.descriptor, short_history, reused, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(reused);
  CHECK(same_identity(history_replaced, cache.identity()));

  xtbloom_compute_options_t changed_damping = short_history;
  changed_damping.scc_mixer_damping = 0.2;
  CHECK(cache.prepare_host(batch.descriptor, changed_damping, reused, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  const Gfn2CudaExecutionIdentity damping_replaced = cache.identity();
  CHECK(damping_replaced.plan_token != history_replaced.plan_token);
  CHECK(damping_replaced.topology_fingerprint != history_replaced.topology_fingerprint);
  CHECK(damping_replaced.scc_loop_owner != history_replaced.scc_loop_owner);

  xtbloom_compute_options_t reproducible = changed_damping;
  reproducible.determinism = XTBLOOM_DETERMINISM_REPRODUCIBLE;
  CHECK(cache.prepare_host(batch.descriptor, reproducible, reused, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  const Gfn2CudaExecutionIdentity deterministic_replaced = cache.identity();
  CHECK(deterministic_replaced.plan_token != damping_replaced.plan_token);
  CHECK(deterministic_replaced.topology_fingerprint != damping_replaced.topology_fingerprint);
  CHECK(deterministic_replaced.eigensolver_owner != damping_replaced.eigensolver_owner);
  CHECK(deterministic_replaced.scc_loop_owner != damping_replaced.scc_loop_owner);
  return 0;
}

int test_native_lattice_staging_reuse(cudaStream_t stream, std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(case_options(4, false), host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, false);
  batch.cell_matrices.assign(4u * 9u, 0.0);
  batch.periodic_axes.assign(4u, XTBLOOM_PERIODIC_AXES_NONE);
  batch.bind();

  DeviceBuffer<double> device_cells;
  DeviceBuffer<std::int32_t> device_axes;
  CHECK(device_cells.upload(batch.cell_matrices, stream) == cudaSuccess);
  CHECK(device_axes.upload(batch.periodic_axes, stream) == cudaSuccess);
  batch.descriptor.cell_matrices = device_cells.view();
  batch.descriptor.periodic_axes = device_axes.view();

  CHECK(cache.validate_native_lattice_test_only(batch.descriptor, error) == XTBLOOM_STATUS_SUCCESS);
  const Gfn2CudaNativeLatticeTestIdentity initial = cache.native_lattice_test_identity();
  CHECK(initial.host_staging != 0u);
  CHECK(initial.host_staging_bytes >= batch.cell_matrices.size() * sizeof(double) +
                                          batch.periodic_axes.size() * sizeof(std::int32_t));
  CHECK(!initial.pending);
  CHECK(!initial.poisoned);

  /* Revalidation reuses the exact pinned image and prepared runtime. A fixed
   * V4 NONE descriptor must not allocate a new host arena per plan call. */
  for (int repeat = 0; repeat < 3; ++repeat) {
    CHECK(cache.validate_native_lattice_test_only(batch.descriptor, error) ==
          XTBLOOM_STATUS_SUCCESS);
    const Gfn2CudaNativeLatticeTestIdentity repeated = cache.native_lattice_test_identity();
    CHECK(repeated.host_staging == initial.host_staging);
    CHECK(repeated.host_staging_bytes == initial.host_staging_bytes);
    CHECK(!repeated.pending);
    CHECK(!repeated.poisoned);
  }
  return 0;
}

int test_native_lattice_allocation_failure_recovery(cudaStream_t stream, std::int32_t device_id) {
  reset_gfn2_cuda_execution_test_state();
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  std::string error;
  HostSccCase small_host;
  CHECK(HostSccCase::create(case_options(2, false), small_host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch small = PublicHostBatch::from_host(small_host, false);
  small.cell_matrices.assign(2u * 9u, 0.0);
  small.periodic_axes.assign(2u, XTBLOOM_PERIODIC_AXES_NONE);
  small.bind();
  DeviceBuffer<double> small_cells;
  DeviceBuffer<std::int32_t> small_axes;
  CUDA_CHECK(small_cells.upload(small.cell_matrices, stream));
  CUDA_CHECK(small_axes.upload(small.periodic_axes, stream));
  small.descriptor.cell_matrices = small_cells.view();
  small.descriptor.periodic_axes = small_axes.view();

  /* An initial allocation failure must leave an empty arena retryable. The
   * second call exercises the exact capacity gate that would otherwise trust
   * stale nonzero metadata beside a null pointer. */
  arm_gfn2_cuda_execution_test_fault(Gfn2CudaExecutionTestFault::kNativeLatticePinnedAllocation);
  CHECK_STATUS(cache.validate_native_lattice_test_only(small.descriptor, error),
               XTBLOOM_STATUS_ALLOCATION_FAILED, error);
  Gfn2CudaNativeLatticeTestIdentity arena = cache.native_lattice_test_identity();
  CHECK(arena.host_staging == 0u);
  CHECK(arena.host_staging_bytes == 0u);
  CHECK(!arena.pending);
  CHECK(!arena.poisoned);
  Gfn2CudaExecutionTestStats stats = gfn2_cuda_execution_test_stats();
  CHECK(stats.native_lattice_allocation_faults == 1u);
  CHECK_STATUS(cache.validate_native_lattice_test_only(small.descriptor, error),
               XTBLOOM_STATUS_SUCCESS, error);
  const Gfn2CudaNativeLatticeTestIdentity initial = cache.native_lattice_test_identity();
  CHECK(initial.host_staging != 0u);
  CHECK(initial.host_staging_bytes > 0u);

  HostSccCase large_host;
  CHECK(HostSccCase::create(case_options(8, false), large_host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch large = PublicHostBatch::from_host(large_host, false);
  large.cell_matrices.assign(8u * 9u, 0.0);
  large.periodic_axes.assign(8u, XTBLOOM_PERIODIC_AXES_NONE);
  large.bind();
  DeviceBuffer<double> large_cells;
  DeviceBuffer<std::int32_t> large_axes;
  CUDA_CHECK(large_cells.upload(large.cell_matrices, stream));
  CUDA_CHECK(large_axes.upload(large.periodic_axes, stream));
  large.descriptor.cell_matrices = large_cells.view();
  large.descriptor.periodic_axes = large_axes.view();

  /* Failed growth preserves both the committed runtime and the complete old
   * arena identity/capacity. Successful retry publishes a larger image only
   * after cudaMallocHost has returned it. */
  arm_gfn2_cuda_execution_test_fault(Gfn2CudaExecutionTestFault::kNativeLatticePinnedAllocation);
  CHECK_STATUS(cache.validate_native_lattice_test_only(large.descriptor, error),
               XTBLOOM_STATUS_ALLOCATION_FAILED, error);
  arena = cache.native_lattice_test_identity();
  CHECK(arena.host_staging == initial.host_staging);
  CHECK(arena.host_staging_bytes == initial.host_staging_bytes);
  CHECK(!arena.pending);
  CHECK(!arena.poisoned);
  stats = gfn2_cuda_execution_test_stats();
  CHECK(stats.native_lattice_allocation_faults == 2u);

  CHECK_STATUS(cache.validate_native_lattice_test_only(large.descriptor, error),
               XTBLOOM_STATUS_SUCCESS, error);
  const Gfn2CudaNativeLatticeTestIdentity grown = cache.native_lattice_test_identity();
  CHECK(grown.host_staging != initial.host_staging);
  CHECK(grown.host_staging_bytes > initial.host_staging_bytes);
  CHECK_STATUS(cache.validate_native_lattice_test_only(large.descriptor, error),
               XTBLOOM_STATUS_SUCCESS, error);
  arena = cache.native_lattice_test_identity();
  CHECK(arena.host_staging == grown.host_staging);
  CHECK(arena.host_staging_bytes == grown.host_staging_bytes);
  reset_gfn2_cuda_execution_test_state();
  return 0;
}

int test_native_lattice_teardown_quarantine(cudaStream_t stream, std::int32_t device_id) {
  reset_gfn2_cuda_execution_test_state();
  std::string error;
  HostSccCase host;
  CHECK(HostSccCase::create(case_options(4, false), host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, false);
  batch.cell_matrices.assign(4u * 9u, 0.0);
  batch.periodic_axes.assign(4u, XTBLOOM_PERIODIC_AXES_NONE);
  batch.bind();
  DeviceBuffer<double> device_cells;
  DeviceBuffer<std::int32_t> device_axes;
  CUDA_CHECK(device_cells.upload(batch.cell_matrices, stream));
  CUDA_CHECK(device_axes.upload(batch.periodic_axes, stream));
  batch.descriptor.cell_matrices = device_cells.view();
  batch.descriptor.periodic_axes = device_axes.view();

  auto cache = std::make_unique<Gfn2CudaExecutionCache>(device_id, reinterpret_cast<void*>(stream));
  CHECK(cache->validate_native_lattice_test_only(batch.descriptor, error) ==
        XTBLOOM_STATUS_SUCCESS);
  const Gfn2CudaNativeLatticeTestIdentity retained = cache->native_lattice_test_identity();
  CHECK(retained.host_staging != 0u);
  CHECK(retained.host_staging_bytes > 0u);

  /* Report completion failure only after both real D2H operations are queued.
   * Pending ownership must survive the failed call so teardown can quarantine
   * the pinned destination when its owner-stream settlement is also failed. */
  arm_gfn2_cuda_execution_test_fault(Gfn2CudaExecutionTestFault::kNativeLatticeCompletionWait);
  CHECK(cache->validate_native_lattice_test_only(batch.descriptor, error) ==
        XTBLOOM_STATUS_INTERNAL_ERROR);
  const Gfn2CudaNativeLatticeTestIdentity failed = cache->native_lattice_test_identity();
  CHECK(failed.host_staging == retained.host_staging);
  CHECK(failed.host_staging_bytes == retained.host_staging_bytes);
  CHECK(failed.pending);
  CHECK(failed.poisoned);
  Gfn2CudaExecutionTestStats stats = gfn2_cuda_execution_test_stats();
  CHECK(stats.native_lattice_completion_faults == 1u);
  CHECK(stats.quarantined_native_lattice_arenas == 0u);

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  const int caller_device = device_count > 1 ? (device_id == 0 ? 1 : 0) : device_id;
  CUDA_CHECK(cudaSetDevice(caller_device));
  arm_gfn2_cuda_execution_test_fault(Gfn2CudaExecutionTestFault::kNativeLatticeTeardownSettlement);
  cache.reset();
  int restored_device = -1;
  CUDA_CHECK(cudaGetDevice(&restored_device));
  CHECK(restored_device == caller_device);

  stats = gfn2_cuda_execution_test_stats();
  CHECK(stats.native_lattice_teardown_faults == 1u);
  CHECK(stats.quarantined_native_lattice_arenas == 1u);
  CHECK(stats.quarantined_native_lattice_bytes == retained.host_staging_bytes);

  /* The test owns the stream and can establish the completion fence that the
   * injected destructor path deliberately could not trust. Restore the owner
   * before synchronizing and before device-buffer destruction. */
  CUDA_CHECK(cudaSetDevice(device_id));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  reset_gfn2_cuda_execution_test_state();
  return 0;
}

int test_device_refresh_and_peer_rollback(cudaStream_t stream, std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(case_options(8, true), host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, true);
  xtbloom_compute_options_t options = compute_options();
  bool reused = true;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  const Gfn2CudaExecutionIdentity initial = cache.identity();

  DeviceBuffer<double> positions;
  DeviceBuffer<double> point_positions;
  DeviceBuffer<double> point_values;
  DeviceBuffer<double> point_gammas;
  DeviceBuffer<double> periodic_shifts;
  DeviceBuffer<double> periodic_response;
  DeviceBuffer<std::uint8_t> requested;
  batch.positions[0] += 0.021;
  std::vector<std::uint8_t> activity(8u, 1u);
  CUDA_CHECK(positions.upload(batch.positions, stream));
  CUDA_CHECK(point_positions.upload(batch.point_positions, stream));
  CUDA_CHECK(point_values.upload(batch.point_values, stream));
  CUDA_CHECK(point_gammas.upload(batch.point_gammas, stream));
  CUDA_CHECK(periodic_shifts.upload(batch.periodic_shifts, stream));
  CUDA_CHECK(periodic_response.upload(batch.response_matrix, stream));
  CUDA_CHECK(requested.upload(activity, stream));

  Gfn2CudaNumericalInputView numerical{};
  numerical.positions = positions.view();
  numerical.point_charge_positions = point_positions.view();
  numerical.point_charge_values = point_values.view();
  numerical.point_charge_gammas = point_gammas.view();
  numerical.atomic_potential_shifts = periodic_shifts.view();
  numerical.charge_response_matrix = periodic_response.view();
  numerical.requested_mask = requested.view();
  CHECK(cache.refresh_numerical_async(numerical, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(same_identity(initial, cache.identity()));
  RefreshSnapshot first;
  CHECK(download_refresh_snapshot(cache.identity(), stream, first) == 0);
  CHECK(first.epoch == 2u);
  for (std::size_t system = 0; system < first.committed.size(); ++system) {
    CHECK(first.committed[system] == 2u);
    CHECK(first.factors[system] == 2u);
    CHECK(first.factor_statuses[system] == 0u);
    CHECK(first.eligible[system] == 1u);
  }

  /* System 1 is explicitly inactive and system 2 has a nonfinite coordinate.
   * Both retain generation-2 operators and overlap factors while every other
   * peer advances through the same device-only transaction. */
  activity[1] = 0u;
  const std::int64_t failed_atom = batch.atom_offsets[2];
  batch.positions[static_cast<std::size_t>(failed_atom * 3)] =
      std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(positions.upload(batch.positions, stream));
  CUDA_CHECK(requested.upload(activity, stream));
  CHECK(cache.refresh_numerical_async(numerical, error) == XTBLOOM_STATUS_SUCCESS);
  RefreshSnapshot second;
  CHECK(download_refresh_snapshot(cache.identity(), stream, second) == 0);
  CHECK(second.epoch == 3u);
  for (std::size_t system = 0; system < second.committed.size(); ++system) {
    const bool retained = system == 1u || system == 2u;
    CHECK(second.committed[system] == (retained ? 2u : 3u));
    CHECK(second.factors[system] == (retained ? 2u : 3u));
    CHECK(second.factor_statuses[system] == 0u);
    CHECK(second.eligible[system] == (retained ? 0u : 1u));
  }

  /* Fail one peer in the explicit point-charge stage and another in periodic
   * validation after preprocessing has succeeded. The complete transaction,
   * including overlap factors, must still roll those peers back. */
  activity.assign(8u, 1u);
  batch.positions[static_cast<std::size_t>(failed_atom * 3)] =
      host.positions()[static_cast<std::size_t>(failed_atom * 3)];
  const std::int64_t bad_point = batch.point_offsets[3];
  batch.point_gammas[static_cast<std::size_t>(bad_point)] = -1.0;
  const std::int64_t periodic_system = 4;
  const std::int64_t periodic_atoms =
      batch.atom_offsets[static_cast<std::size_t>(periodic_system + 1)] -
      batch.atom_offsets[static_cast<std::size_t>(periodic_system)];
  CHECK(periodic_atoms > 1);
  const std::int64_t response_begin =
      batch.response_offsets[static_cast<std::size_t>(periodic_system)];
  batch.response_matrix[static_cast<std::size_t>(response_begin + 1)] += 0.125;
  CUDA_CHECK(positions.upload(batch.positions, stream));
  CUDA_CHECK(point_gammas.upload(batch.point_gammas, stream));
  CUDA_CHECK(periodic_response.upload(batch.response_matrix, stream));
  CUDA_CHECK(requested.upload(activity, stream));
  CHECK(cache.refresh_numerical_async(numerical, error) == XTBLOOM_STATUS_SUCCESS);
  RefreshSnapshot third;
  CHECK(download_refresh_snapshot(cache.identity(), stream, third) == 0);
  CHECK(third.epoch == 4u);
  for (std::size_t system = 0; system < third.committed.size(); ++system) {
    const bool retained = system == 3u || system == 4u;
    CHECK(third.committed[system] == (retained ? 3u : 4u));
    CHECK(third.factors[system] == (retained ? 3u : 4u));
    CHECK(third.factor_statuses[system] == 0u);
    CHECK(third.eligible[system] == (retained ? 0u : 1u));
  }

  CHECK(cache.execute_inference_async(Gfn2CudaSccStartMode::kFresh, error) ==
        XTBLOOM_STATUS_SUCCESS);
  InferenceSnapshot mixed;
  CHECK(download_inference_snapshot(cache.identity(), stream, true, mixed) == 0);
  CHECK(mixed.publication_epoch == 4u);
  CHECK(mixed.publication_plan_error ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationPlanError::kSuccess));
  for (const std::size_t failed_system : {3u, 4u}) {
    CHECK(mixed.statuses[failed_system] == XTBLOOM_STATUS_INTERNAL_ERROR);
    CHECK(mixed.converged[failed_system] == 0u);
    CHECK(mixed.iterations[failed_system] == 0);
    CHECK(std::isnan(mixed.energies[failed_system]));
    CHECK(mixed.warm_generations[failed_system] == 0u);
    CHECK(mixed.publication_system_errors[failed_system] ==
          static_cast<std::uint32_t>(
              Gfn2InferencePublicationSystemError::kIneligibleNumericalRefresh));
    const std::int64_t atom_begin = batch.atom_offsets[failed_system];
    const std::int64_t atom_end = batch.atom_offsets[failed_system + 1u];
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
      CHECK(std::isnan(mixed.atomic_charges[static_cast<std::size_t>(atom)]));
      for (std::int64_t component = 0; component < 3; ++component) {
        CHECK(std::isnan(mixed.qm_forces[static_cast<std::size_t>(atom * 3 + component)]));
      }
    }
    const std::int64_t point_begin = batch.point_offsets[failed_system];
    const std::int64_t point_end = batch.point_offsets[failed_system + 1u];
    for (std::int64_t point = point_begin; point < point_end; ++point) {
      for (std::int64_t component = 0; component < 3; ++component) {
        CHECK(std::isnan(mixed.point_forces[static_cast<std::size_t>(point * 3 + component)]));
      }
    }
  }
  for (std::size_t system = 0; system < mixed.statuses.size(); ++system) {
    if (system == 3u || system == 4u) continue;
    CHECK(mixed.publication_system_errors[system] ==
          static_cast<std::uint32_t>(Gfn2InferencePublicationSystemError::kSuccess));
  }
  Gfn2CudaNumericalInputView malformed = numerical;
  malformed.positions.size_bytes -= sizeof(double);
  CHECK(cache.refresh_numerical_async(malformed, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  RefreshSnapshot after_rejection;
  CHECK(download_refresh_snapshot(cache.identity(), stream, after_rejection) == 0);
  CHECK(after_rejection.epoch == third.epoch);
  CHECK(after_rejection.committed == third.committed);
  CHECK(after_rejection.factors == third.factors);
  return 0;
}

/*
 * Production path with the sparse pair-list CN and force gates active.  A single
 * 62-atom C20H42 chain crosses the 40-atom dense-fallback crossover, so the
 * runtime provisioning enables both the bucketed CN consistency gate and the
 * sparse coordination VJP parity gate.  Force-mode setup must succeed before
 * the first committed pair list exists, then a fresh numerical refresh and
 * energy/force inference must complete with no peer failure.
 */
int test_large_system_sparse_gate(cudaStream_t stream, std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  HostSccCaseOptions case_options =
      homogeneous_case_options(1, SmallSystemKind::kC20H42, false, false, false);
  case_options.maximum_iterations = 100u;
  case_options.electronic_temperature = 300.0 * 3.166811563e-06;
  CHECK(HostSccCase::create(case_options, host, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(host.total_atoms() == 62u);
  PublicHostBatch batch = PublicHostBatch::from_host(host, false);
  xtbloom_compute_options_t options = compute_options();
  options.max_scc_iterations = 200;
  options.electronic_temperature = 300.0 * 3.166811563e-06;
  bool reused = true;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  const Gfn2CudaExecutionIdentity initial = cache.identity();
  CHECK(initial.force_mode_ready == 1u);
  CHECK(initial.energy_force_smoke_ready == 1u);
  CHECK(initial.scc_conditional_graph_ready == 1u);

  PinnedHostBuffer<double> caller_positions;
  PinnedHostBuffer<std::uint8_t> caller_requested;
  CHECK(caller_positions.assign(batch.positions) == cudaSuccess);
  CHECK(caller_requested.assign(std::vector<std::uint8_t>{1u}) == cudaSuccess);

  Gfn2CudaNumericalInputView numerical{};
  numerical.positions = caller_positions.view();
  numerical.requested_mask = caller_requested.view();
  CHECK(cache.refresh_numerical_async(numerical, error) == XTBLOOM_STATUS_SUCCESS);
  RefreshSnapshot refreshed;
  CHECK(download_refresh_snapshot(cache.identity(), stream, refreshed) == 0);
  CHECK(refreshed.epoch == 2u);
  CHECK(std::all_of(refreshed.committed.begin(), refreshed.committed.end(),
                    [](std::uint64_t value) { return value == 2u; }));
  CHECK(std::all_of(refreshed.factor_statuses.begin(), refreshed.factor_statuses.end(),
                    [](std::uint32_t value) { return value == 0u; }));

  CHECK(cache.execute_inference_async(Gfn2CudaSccStartMode::kFresh, error) ==
        XTBLOOM_STATUS_SUCCESS);
  InferenceSnapshot result;
  CHECK(download_inference_snapshot(cache.identity(), stream, true, result) == 0);
  CHECK(result.publication_epoch == 2u);
  CHECK(result.publication_plan_error ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationPlanError::kSuccess));
  /* The CN gate is applied during numerical refresh, while the sparse VJP is
   * parity-gated during terminal force execution.  Either disagreement fails
   * the peer closed instead of publishing finite results. */
  CHECK(result.statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::isfinite(result.energies[0]));
  CHECK(result.qm_forces.size() == static_cast<std::size_t>(3 * host.total_atoms()));
  CHECK(std::all_of(result.qm_forces.begin(), result.qm_forces.end(),
                    [](double value) { return std::isfinite(value); }));

  /* Reuse the fixed-topology runtime at a new geometry epoch.  This exercises
   * the stable committed pair-list addresses and sparse force consumer on the
   * replay path rather than proving only the first execution. */
  caller_positions.data()[0] += 0.01;
  CHECK(cache.refresh_numerical_async(numerical, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(same_identity(initial, cache.identity()));
  RefreshSnapshot replay_refresh;
  CHECK(download_refresh_snapshot(cache.identity(), stream, replay_refresh) == 0);
  CHECK(replay_refresh.epoch == 3u);
  CHECK(replay_refresh.committed[0] == 3u);
  CHECK(replay_refresh.eligible[0] == 1u);
  CHECK(cache.execute_inference_async(Gfn2CudaSccStartMode::kFresh, error) ==
        XTBLOOM_STATUS_SUCCESS);
  InferenceSnapshot replay_result;
  CHECK(download_inference_snapshot(cache.identity(), stream, true, replay_result) == 0);
  CHECK(replay_result.publication_epoch == 3u);
  CHECK(replay_result.publication_plan_error ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationPlanError::kSuccess));
  CHECK(replay_result.statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::isfinite(replay_result.energies[0]));
  CHECK(std::all_of(replay_result.qm_forces.begin(), replay_result.qm_forces.end(),
                    [](double value) { return std::isfinite(value); }));
  return 0;
}

int test_host_refresh_snapshot_lifetime(cudaStream_t stream, std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(homogeneous_case_options(1, SmallSystemKind::kH2, false, false, false),
                            host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, false);
  xtbloom_compute_options_t options = compute_options(false);
  bool reused = true;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  const Gfn2CudaExecutionIdentity initial = cache.identity();

  std::vector<double> submitted_positions = batch.positions;
  submitted_positions[3] += 0.031;
  PinnedHostBuffer<double> caller_positions;
  PinnedHostBuffer<std::uint8_t> caller_requested;
  CHECK(caller_positions.assign(submitted_positions) == cudaSuccess);
  CHECK(caller_requested.assign(std::vector<std::uint8_t>{1u}) == cudaSuccess);

  Gfn2CudaNumericalInputView numerical{};
  numerical.positions = caller_positions.view();
  numerical.requested_mask = caller_requested.view();

  /* Keep the owner's stream busy so the H2D cannot possibly consume the
   * caller's pinned allocation before refresh_numerical_async returns. */
  int clock_rate_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device_id));
  CHECK(clock_rate_khz > 0);
  const unsigned long long delay_cycles = static_cast<unsigned long long>(clock_rate_khz) * 250ULL;
  hold_owner_stream_kernel<<<1, 1, 0, stream>>>(delay_cycles);
  CUDA_CHECK(cudaPeekAtLastError());
  CHECK(cache.refresh_numerical_async(numerical, error) == XTBLOOM_STATUS_SUCCESS);

  /* A second host call cannot overwrite a busy single-flight staging image.
   * Sanitizer modes may serialize the delay and run the stream completion
   * function before control returns; accepting that second snapshot is then
   * the correct state. */
  const xtbloom_status_t second_submit = cache.refresh_numerical_async(numerical, error);
  CHECK(second_submit == XTBLOOM_STATUS_INVALID_ARGUMENT ||
        second_submit == XTBLOOM_STATUS_SUCCESS);
  if (second_submit == XTBLOOM_STATUS_INVALID_ARGUMENT) {
    CHECK(error.find("still in flight") != std::string::npos);
  }
  const std::uint64_t first_expected_epoch = second_submit == XTBLOOM_STATUS_SUCCESS ? 3u : 2u;

  /* Every successful call already owns its call-time bytes. Poisoning and
   * releasing the caller allocation immediately cannot affect queued H2D. */
  std::fill(caller_positions.data(), caller_positions.data() + caller_positions.size(),
            std::numeric_limits<double>::quiet_NaN());
  caller_requested.data()[0] = 0u;
  caller_positions.reset();
  caller_requested.reset();

  RefreshSnapshot first;
  CHECK(download_refresh_snapshot(cache.identity(), stream, first) == 0);
  CHECK(first.epoch == first_expected_epoch);
  CHECK(first.committed[0] == first_expected_epoch);
  CHECK(first.factors[0] == first_expected_epoch);
  CHECK(first.factor_statuses[0] == 0u);
  CHECK(first.eligible[0] == 1u);
  CHECK(same_identity(initial, cache.identity()));

  /* Once the stream completion runs, the same fixed staging allocation is
   * reusable without querying an event or synchronizing in the submit path. */
  submitted_positions[3] -= 0.012;
  CHECK(caller_positions.assign(submitted_positions) == cudaSuccess);
  CHECK(caller_requested.assign(std::vector<std::uint8_t>{1u}) == cudaSuccess);
  numerical.positions = caller_positions.view();
  numerical.requested_mask = caller_requested.view();
  CHECK(cache.refresh_numerical_async(numerical, error) == XTBLOOM_STATUS_SUCCESS);
  caller_positions.reset();
  caller_requested.reset();
  RefreshSnapshot second;
  CHECK(download_refresh_snapshot(cache.identity(), stream, second) == 0);
  CHECK(second.epoch == first_expected_epoch + 1u);
  CHECK(second.committed[0] == first_expected_epoch + 1u);
  CHECK(second.factors[0] == first_expected_epoch + 1u);
  CHECK(second.factor_statuses[0] == 0u);
  CHECK(second.eligible[0] == 1u);
  CHECK(same_identity(initial, cache.identity()));
  return 0;
}

int test_refresh_cuda_graph_replay(cudaStream_t stream, std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(case_options(8, true), host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, true);
  xtbloom_compute_options_t options = compute_options();
  bool reused = true;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);

  DeviceBuffer<double> positions;
  DeviceBuffer<double> point_positions;
  DeviceBuffer<double> point_values;
  DeviceBuffer<double> point_gammas;
  DeviceBuffer<double> periodic_shifts;
  DeviceBuffer<double> periodic_response;
  DeviceBuffer<std::uint8_t> requested;
  std::vector<std::uint8_t> activity(8u, 1u);
  CUDA_CHECK(positions.upload(batch.positions, stream));
  CUDA_CHECK(point_positions.upload(batch.point_positions, stream));
  CUDA_CHECK(point_values.upload(batch.point_values, stream));
  CUDA_CHECK(point_gammas.upload(batch.point_gammas, stream));
  CUDA_CHECK(periodic_shifts.upload(batch.periodic_shifts, stream));
  CUDA_CHECK(periodic_response.upload(batch.response_matrix, stream));
  CUDA_CHECK(requested.upload(activity, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  Gfn2CudaNumericalInputView numerical{};
  numerical.positions = positions.view();
  numerical.point_charge_positions = point_positions.view();
  numerical.point_charge_values = point_values.view();
  numerical.point_charge_gammas = point_gammas.view();
  numerical.atomic_potential_shifts = periodic_shifts.view();
  numerical.charge_response_matrix = periodic_response.view();
  numerical.requested_mask = requested.view();

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CHECK(cache.refresh_numerical_async(numerical, error) == XTBLOOM_STATUS_SUCCESS);
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CHECK(graph != nullptr);
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CHECK(executable != nullptr);

  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  RefreshSnapshot first;
  CHECK(download_refresh_snapshot(cache.identity(), stream, first) == 0);
  CHECK(first.epoch == 2u);
  for (std::size_t system = 0; system < first.committed.size(); ++system) {
    CHECK(first.committed[system] == 2u);
    CHECK(first.factors[system] == 2u);
    CHECK(first.eligible[system] == 1u);
  }

  batch.positions[0] += 0.043;
  CUDA_CHECK(positions.upload(batch.positions, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  RefreshSnapshot second;
  CHECK(download_refresh_snapshot(cache.identity(), stream, second) == 0);
  CHECK(second.epoch == 3u);
  for (std::size_t system = 0; system < second.committed.size(); ++system) {
    CHECK(second.committed[system] == 3u);
    CHECK(second.factors[system] == 3u);
    CHECK(second.eligible[system] == 1u);
  }

  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  return 0;
}

int test_host_refresh_rejected_during_cuda_graph_capture(cudaStream_t stream,
                                                         std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(case_options(8, true), host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, true);
  xtbloom_compute_options_t options = compute_options();
  bool reused = true;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);

  Gfn2CudaNumericalInputView numerical{};
  numerical.positions = batch.descriptor.positions;
  numerical.point_charge_positions = batch.descriptor.point_charge_positions;
  numerical.point_charge_values = batch.descriptor.point_charge_values;
  numerical.point_charge_gammas = batch.descriptor.point_charge_gammas;
  numerical.atomic_potential_shifts = batch.descriptor.atomic_potential_shifts;
  numerical.charge_response_matrix = batch.descriptor.charge_response_matrix;

  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CHECK(cache.refresh_numerical_async(numerical, error) == XTBLOOM_STATUS_NOT_SUPPORTED);
  CHECK(error.find("requires CUDA-device input buffers") != std::string::npos);
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CHECK(graph != nullptr);

  /* The rejected capture must not reserve or poison the fixed host staging
   * image.  It is immediately usable by the ordinary asynchronous path. */
  batch.positions[0] += 0.027;
  batch.bind();
  numerical.positions = batch.descriptor.positions;
  numerical.point_charge_positions = batch.descriptor.point_charge_positions;
  numerical.point_charge_values = batch.descriptor.point_charge_values;
  numerical.point_charge_gammas = batch.descriptor.point_charge_gammas;
  numerical.atomic_potential_shifts = batch.descriptor.atomic_potential_shifts;
  numerical.charge_response_matrix = batch.descriptor.charge_response_matrix;
  CHECK(cache.refresh_numerical_async(numerical, error) == XTBLOOM_STATUS_SUCCESS);
  RefreshSnapshot refreshed;
  CHECK(download_refresh_snapshot(cache.identity(), stream, refreshed) == 0);
  CHECK(refreshed.epoch == 2u);
  for (std::size_t system = 0; system < refreshed.committed.size(); ++system) {
    CHECK(refreshed.committed[system] == 2u);
    CHECK(refreshed.factors[system] == 2u);
    CHECK(refreshed.eligible[system] == 1u);
  }

  CUDA_CHECK(cudaGraphDestroy(graph));
  return 0;
}

int test_periodic_refresh_uses_zero_for_absent_optional_leaf(cudaStream_t stream,
                                                             std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(case_options(4, true), host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, true);
  xtbloom_compute_options_t options = compute_options();
  bool reused = true;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);

  Gfn2CudaNumericalInputView numerical{};
  numerical.positions = batch.descriptor.positions;
  numerical.point_charge_positions = batch.descriptor.point_charge_positions;
  numerical.point_charge_values = batch.descriptor.point_charge_values;
  numerical.point_charge_gammas = batch.descriptor.point_charge_gammas;
  numerical.charge_response_matrix = batch.descriptor.charge_response_matrix;

  /* A periodic response matrix without an explicit shift means b=0. The
   * runtime must source that zero from owned device staging instead of
   * requiring the absent public descriptor to name storage. */
  CHECK(cache.refresh_numerical_async(numerical, error) == XTBLOOM_STATUS_SUCCESS);
  RefreshSnapshot response_only;
  CHECK(download_refresh_snapshot(cache.identity(), stream, response_only) == 0);
  std::vector<double> committed_shifts(batch.periodic_shifts.size(), 1.0);
  CUDA_CHECK(cudaMemcpyAsync(
      committed_shifts.data(),
      reinterpret_cast<const void*>(cache.identity().committed_periodic_shifts.address),
      committed_shifts.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(std::all_of(committed_shifts.begin(), committed_shifts.end(),
                    [](double value) { return value == 0.0; }));

  numerical.atomic_potential_shifts = batch.descriptor.atomic_potential_shifts;
  numerical.charge_response_matrix = {};

  /* Conversely, a shift-only periodic request means A=0 and must not depend
   * on caller-owned response storage. */
  CHECK(cache.refresh_numerical_async(numerical, error) == XTBLOOM_STATUS_SUCCESS);
  RefreshSnapshot shift_only;
  CHECK(download_refresh_snapshot(cache.identity(), stream, shift_only) == 0);
  std::vector<double> committed_response(batch.response_matrix.size(), 1.0);
  CUDA_CHECK(cudaMemcpyAsync(
      committed_response.data(),
      reinterpret_cast<const void*>(cache.identity().committed_periodic_response.address),
      committed_response.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(std::all_of(committed_response.begin(), committed_response.end(),
                    [](double value) { return value == 0.0; }));
  return 0;
}

int test_base_configuration(cudaStream_t stream, std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(case_options(4, false), host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, false);
  xtbloom_compute_options_t options = compute_options();
  bool reused = true;
  const xtbloom_status_t status = cache.prepare_host(batch.descriptor, options, reused, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "base runtime setup failed: status=%d error=%s\n", status, error.c_str());
  }
  CHECK(status == XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  const Gfn2CudaExecutionIdentity identity = cache.identity();
  CHECK(validate_identity(identity, 4, true, false, false) == 0);
  CHECK(identity.total_point_charges == 0);
  CHECK((identity.enabled_component_mask &
         static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kExplicitPointCharge)) == 0u);
  CHECK((identity.enabled_component_mask &
         static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kPeriodicEmbedding)) == 0u);
  return 0;
}

int test_energy_only_configuration(cudaStream_t stream, std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(case_options(8, true), host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, true);
  xtbloom_compute_options_t options = compute_options(false);
  bool reused = true;
  const xtbloom_status_t status = cache.prepare_host(batch.descriptor, options, reused, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "energy-only runtime setup failed: status=%d error=%s\n", status,
                 error.c_str());
  }
  CHECK(status == XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  CHECK(validate_identity(cache.identity(), 8, true, true, true, false) == 0);
  return 0;
}

int test_independent_optional_configurations(cudaStream_t stream, std::int32_t device_id) {
  struct Configuration {
    const char* name;
    std::int64_t batch_size;
    SmallSystemKind system;
    bool fixture_d4;
    bool expect_runtime_d4;
    bool points;
    bool periodic;
  };
  /* The public runtime derives D4 applicability from topology: an all-singleton
   * batch has exactly zero two-body/ATM contribution and keeps the D4 plan
   * canonical-empty even when the richer internal fixture owns D4 sources. */
  constexpr std::array<Configuration, 6> configurations{{
      {"single-atom base", 4, SmallSystemKind::kHe, false, false, false, false},
      {"D4 singleton elision", 1, SmallSystemKind::kHe, true, false, false, false},
      {"D4 all-singleton ragged elision", 8, SmallSystemKind::kHe, true, false, false, false},
      {"D4 only", 4, SmallSystemKind::kH2, true, true, false, false},
      {"point charge only", 4, SmallSystemKind::kHe, false, false, true, false},
      {"periodic only", 4, SmallSystemKind::kHe, false, false, false, true},
  }};

  for (const Configuration& configuration : configurations) {
    Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
    HostSccCase host;
    std::string error;
    CHECK(
        HostSccCase::create(homogeneous_case_options(configuration.batch_size, configuration.system,
                                                     configuration.fixture_d4, configuration.points,
                                                     configuration.periodic),
                            host, error) == XTBLOOM_STATUS_SUCCESS);
    PublicHostBatch batch = PublicHostBatch::from_host(host, configuration.periodic);
    xtbloom_compute_options_t options = compute_options();
    bool reused = true;
    const xtbloom_status_t status = cache.prepare_host(batch.descriptor, options, reused, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      std::fprintf(stderr, "%s runtime setup failed: status=%d error=%s\n", configuration.name,
                   status, error.c_str());
    }
    CHECK(status == XTBLOOM_STATUS_SUCCESS);
    CHECK(!reused);
    CHECK(validate_identity(cache.identity(), configuration.batch_size,
                            configuration.expect_runtime_d4, configuration.points,
                            configuration.periodic) == 0);
  }
  return 0;
}

int test_context_owned_runtime(cudaStream_t stream, std::int32_t device_id) {
  xtbloom_context_options_t context_options{};
  context_options.struct_size = XTBLOOM_CONTEXT_OPTIONS_V1_SIZE;
  context_options.api_version = XTBLOOM_API_VERSION;
  context_options.backend = XTBLOOM_BACKEND_CUDA;
  context_options.device_id = device_id;
  context_options.stream = reinterpret_cast<void*>(stream);

  Context* raw_context = nullptr;
  std::string error;
  CHECK(xtbloom::detail::create_context(context_options, raw_context, error) ==
        XTBLOOM_STATUS_SUCCESS);
  std::unique_ptr<Context> context(raw_context);
  CHECK(context != nullptr);
  CHECK(context->gfn2_cuda_execution_cache != nullptr);
  CHECK(!context->gfn2_cuda_execution_cache->valid());

  HostSccCase host;
  CHECK(HostSccCase::create(homogeneous_case_options(1, SmallSystemKind::kHe, false, false, false),
                            host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, false);
  xtbloom_compute_options_t options = compute_options();
  bool reused = true;
  CHECK(context->gfn2_cuda_execution_cache->prepare_host(batch.descriptor, options, reused,
                                                         error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  CHECK(context->gfn2_cuda_execution_cache->valid());
  CHECK(validate_identity(context->gfn2_cuda_execution_cache->identity(), 1, false, false, false) ==
        0);
  return 0;
}

int test_default_stream_refresh(std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, nullptr);
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(case_options(8, true), host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, true);
  xtbloom_compute_options_t options = compute_options();
  bool reused = true;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  batch.positions[0] += 0.018;
  batch.bind();
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(reused);
  RefreshSnapshot snapshot;
  CHECK(download_refresh_snapshot(cache.identity(), nullptr, snapshot) == 0);
  CHECK(snapshot.epoch == 2u);
  for (std::size_t system = 0; system < snapshot.committed.size(); ++system) {
    CHECK(snapshot.committed[system] == 2u);
    CHECK(snapshot.factors[system] == 2u);
    CHECK(snapshot.eligible[system] == 1u);
  }
  return 0;
}

int test_fresh_warm_inference_and_post_scc_refresh(cudaStream_t stream, std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  /* Keep D4 enabled so changed-geometry WARM/FRESH execution proves that the
   * final committed-position/CN provenance survives the complete SCC and
   * post-SCC consumer chain. */
  CHECK(HostSccCase::create(homogeneous_case_options(1, SmallSystemKind::kH2, true, false, false),
                            host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, false);
  xtbloom_compute_options_t options = compute_options(false);
  options.max_scc_iterations = 32;
  bool reused = true;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  const Gfn2CudaExecutionIdentity initial = cache.identity();
  CHECK(validate_identity(initial, 1, true, false, false, false) == 0);
  RefreshSnapshot initial_refresh;
  CHECK(download_refresh_snapshot(cache.identity(), stream, initial_refresh) == 0);
  CHECK(initial_refresh.epoch == 1u);
  CHECK(initial_refresh.committed[0] == 1u);
  CHECK(initial_refresh.eligible[0] == 1u);
  CHECK(initial_refresh.factors[0] == 1u);
  CHECK(initial_refresh.factor_statuses[0] == 0u);

  /* Warm is a real checkpoint mode, not an alias for the setup SAD image. */
  CHECK(cache.execute_inference_async(Gfn2CudaSccStartMode::kWarm, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(cache.execute_inference_async(Gfn2CudaSccStartMode::kFresh, error) ==
        XTBLOOM_STATUS_SUCCESS);
  InferenceSnapshot fresh;
  CHECK(download_inference_snapshot(cache.identity(), stream, false, fresh) == 0);
  CHECK(fresh.statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(fresh.converged[0] == 1u);
  CHECK(fresh.iterations[0] > 0 && fresh.iterations[0] <= options.max_scc_iterations);
  CHECK(std::isfinite(fresh.energies[0]));
  CHECK(fresh.publication_epoch == 1u);
  CHECK(fresh.publication_plan_error ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationPlanError::kSuccess));
  CHECK(fresh.publication_system_errors[0] ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationSystemError::kSuccess));
  CHECK(fresh.warm_generations[0] == 1u);
  CHECK(cache.identity().warm_checkpoint_ready == 1u);
  CHECK(same_identity(initial, cache.identity()));

  CHECK(cache.execute_inference_async(Gfn2CudaSccStartMode::kWarm, error) ==
        XTBLOOM_STATUS_SUCCESS);
  InferenceSnapshot warm;
  CHECK(download_inference_snapshot(cache.identity(), stream, false, warm) == 0);
  CHECK(warm.statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(warm.converged[0] == 1u);
  CHECK(warm.iterations[0] > 0 && warm.iterations[0] <= options.max_scc_iterations);
  CHECK(std::isfinite(warm.energies[0]));
  CHECK(warm.publication_epoch == 1u);
  CHECK(warm.publication_plan_error ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationPlanError::kSuccess));
  CHECK(warm.publication_system_errors[0] ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationSystemError::kSuccess));
  CHECK(warm.warm_generations[0] == 1u);
  CHECK(same_identity(initial, cache.identity()));

  /* One successful numerical refresh may warm-start from the immediately
   * preceding committed electronic state. */
  batch.positions[0] += 0.019;
  batch.bind();
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(reused);
  RefreshSnapshot epoch_two;
  CHECK(download_refresh_snapshot(cache.identity(), stream, epoch_two) == 0);
  CHECK(epoch_two.epoch == 2u);
  CHECK(epoch_two.committed[0] == 2u);
  CHECK(epoch_two.factors[0] == 2u);
  CHECK(epoch_two.eligible[0] == 1u);
  CHECK(cache.execute_inference_async(Gfn2CudaSccStartMode::kWarm, error) ==
        XTBLOOM_STATUS_SUCCESS);
  InferenceSnapshot refreshed_warm;
  CHECK(download_inference_snapshot(cache.identity(), stream, false, refreshed_warm) == 0);
  CHECK(refreshed_warm.statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(refreshed_warm.converged[0] == 1u);
  CHECK(refreshed_warm.iterations[0] > 0);
  CHECK(std::isfinite(refreshed_warm.energies[0]));
  CHECK(refreshed_warm.publication_epoch == 2u);
  CHECK(refreshed_warm.publication_plan_error ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationPlanError::kSuccess));
  CHECK(refreshed_warm.publication_system_errors[0] ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationSystemError::kSuccess));
  CHECK(refreshed_warm.warm_generations[0] == 2u);

  /* The predecessor edge is deliberately only one committed refresh deep.
   * Two refreshes without an intervening inference must not chain the epoch-2
   * electronic checkpoint into epoch 4. */
  batch.positions[0] -= 0.007;
  batch.bind();
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(reused);
  RefreshSnapshot epoch_three;
  CHECK(download_refresh_snapshot(cache.identity(), stream, epoch_three) == 0);
  CHECK(epoch_three.epoch == 3u);
  CHECK(epoch_three.committed[0] == 3u);
  batch.positions[0] += 0.011;
  batch.bind();
  CHECK_STATUS(cache.prepare_host(batch.descriptor, options, reused, error),
               XTBLOOM_STATUS_SUCCESS, error);
  CHECK(reused);
  RefreshSnapshot epoch_four;
  CHECK(download_refresh_snapshot(cache.identity(), stream, epoch_four) == 0);
  CHECK(epoch_four.epoch == 4u);
  CHECK(epoch_four.committed[0] == 4u);
  CHECK(cache.execute_inference_async(Gfn2CudaSccStartMode::kWarm, error) ==
        XTBLOOM_STATUS_SUCCESS);
  InferenceSnapshot stale_warm;
  CHECK(download_inference_snapshot(cache.identity(), stream, false, stale_warm) == 0);
  CHECK(stale_warm.statuses[0] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(stale_warm.converged[0] == 0u);
  CHECK(stale_warm.iterations[0] == 0);
  CHECK(std::isnan(stale_warm.energies[0]));
  CHECK(stale_warm.publication_epoch == 4u);
  CHECK(stale_warm.publication_plan_error ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationPlanError::kSuccess));
  CHECK(stale_warm.publication_system_errors[0] ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationSystemError::kSuccess));
  CHECK(stale_warm.warm_generations[0] == 0u);

  CHECK(cache.execute_inference_async(Gfn2CudaSccStartMode::kFresh, error) ==
        XTBLOOM_STATUS_SUCCESS);
  InferenceSnapshot refreshed_fresh;
  CHECK(download_inference_snapshot(cache.identity(), stream, false, refreshed_fresh) == 0);
  CHECK(refreshed_fresh.statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(refreshed_fresh.converged[0] == 1u);
  CHECK(std::isfinite(refreshed_fresh.energies[0]));
  CHECK(refreshed_fresh.publication_epoch == 4u);
  CHECK(refreshed_fresh.publication_plan_error ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationPlanError::kSuccess));
  CHECK(refreshed_fresh.publication_system_errors[0] ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationSystemError::kSuccess));
  CHECK(refreshed_fresh.warm_generations[0] == 4u);

  /* Regression: SCC convergence leaves its activity ledger terminal. A later
   * refresh must repopulate the factorization mask from refresh eligibility. */
  batch.positions[0] -= 0.005;
  batch.bind();
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(reused);
  RefreshSnapshot epoch_five;
  CHECK(download_refresh_snapshot(cache.identity(), stream, epoch_five) == 0);
  CHECK(epoch_five.epoch == 5u);
  CHECK(epoch_five.committed[0] == 5u);
  CHECK(epoch_five.factors[0] == 5u);
  CHECK(epoch_five.factor_statuses[0] == 0u);
  CHECK(epoch_five.eligible[0] == 1u);
  CHECK(same_identity(initial, cache.identity()));
  return 0;
}

int test_failed_inference_consumes_warm_checkpoint(cudaStream_t stream, std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(homogeneous_case_options(1, SmallSystemKind::kHe, false, false, false),
                            host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, false);
  xtbloom_compute_options_t options = compute_options(false);
  options.max_scc_iterations = 32;
  bool reused = true;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(cache.execute_inference_async(Gfn2CudaSccStartMode::kFresh, error) ==
        XTBLOOM_STATUS_SUCCESS);
  InferenceSnapshot first;
  CHECK(download_inference_snapshot(cache.identity(), stream, false, first) == 0);
  CHECK(first.warm_generations[0] == 1u);

  /* Inject a host-side descriptor mismatch that is discovered only after the
   * warm reset has entered the owner stream. The consumed token must stay zero
   * even though publication cannot be submitted and therefore cannot reissue
   * a checkpoint. */
  const Gfn2CudaExecutionIdentity identity = cache.identity();
  auto* const publication_results =
      reinterpret_cast<Gfn2InferencePublicationDeviceResults*>(identity.inference_results);
  CHECK(publication_results != nullptr);
  const std::uint64_t saved_token = publication_results->plan_token;
  publication_results->plan_token = 0u;
  CHECK(cache.execute_inference_async(Gfn2CudaSccStartMode::kWarm, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(cache.identity().warm_checkpoint_ready == 0u);
  publication_results->plan_token = saved_token;

  InferenceSnapshot failed;
  CHECK(download_inference_snapshot(cache.identity(), stream, false, failed) == 0);
  CHECK(failed.warm_generations[0] == 0u);
  CHECK(failed.energies == first.energies);
  CHECK(failed.iterations == first.iterations);
  CHECK(failed.converged == first.converged);
  CHECK(failed.statuses == first.statuses);

  CHECK(cache.execute_inference_async(Gfn2CudaSccStartMode::kFresh, error) ==
        XTBLOOM_STATUS_SUCCESS);
  InferenceSnapshot recovered;
  CHECK(download_inference_snapshot(cache.identity(), stream, false, recovered) == 0);
  CHECK(recovered.statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(recovered.warm_generations[0] == 1u);
  return 0;
}

int test_rejected_async_request_preserves_internal_state(cudaStream_t stream,
                                                         std::int32_t device_id) {
  auto cache = std::make_shared<Gfn2CudaExecutionCache>(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(homogeneous_case_options(1, SmallSystemKind::kHe, false, false, false),
                            host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, false);
  batch.cell_matrices.assign(9u, 0.0);
  batch.periodic_axes.assign(1u, XTBLOOM_PERIODIC_AXES_NONE);
  batch.bind();
  xtbloom_compute_options_t options = compute_options(false);
  options.max_scc_iterations = 32;
  CHECK(cache->prepare_topology_only(batch.descriptor, options, error) == XTBLOOM_STATUS_SUCCESS);

  auto submit = [&](xtbloom_status_t expected, bool expect_inline = false,
                    const char* expected_error = nullptr) {
    constexpr double kEnergyCanary = 8125.75;
    constexpr std::int32_t kIterationCanary = INT32_C(0x13572468);
    constexpr std::uint8_t kConvergedCanary = UINT8_C(0xa5);
    constexpr xtbloom_status_t kStatusCanary = INT32_C(0x24681357);
    std::vector<double> energies(1u, kEnergyCanary);
    std::vector<std::int32_t> iterations(1u, kIterationCanary);
    std::vector<std::uint8_t> converged(1u, kConvergedCanary);
    std::vector<xtbloom_status_t> statuses(1u, kStatusCanary);
    xtbloom_batch_result_t result{};
    CHECK(xtbloom_batch_result_init(&result, sizeof(result)) == XTBLOOM_STATUS_SUCCESS);
    result.flags = UINT32_C(0xc35aa53c);
    result.energies = mutable_host_buffer(energies);
    result.scc_iterations = mutable_host_buffer(iterations);
    result.scc_converged = mutable_host_buffer(converged);
    result.per_system_status = mutable_host_buffer(statuses);
    RequestSubmission submission;
    CHECK(enqueue_restricted_gfn2_cuda_plan(cache, batch.descriptor, options, result, submission,
                                            error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(submission.completed_inline == expect_inline);
    xtbloom_status_t completion_status = submission.completion_status;
    std::uint32_t completion_flags = submission.result_flags;
    std::string completion_error = submission.completion_error;
    if (!submission.completed_inline) {
      CHECK(submission.pending != nullptr);
      RequestCompletionResult completion;
      CHECK(submission.pending->probe(true, completion) == XTBLOOM_STATUS_SUCCESS);
      CHECK(completion.complete);
      completion_status = completion.completion_status;
      completion_flags = completion.result_flags;
      completion_error = completion.completion_error;
      submission.pending->settle_noexcept();
    } else {
      CHECK(submission.pending == nullptr);
    }
    CHECK(completion_status == expected);
    CHECK(completion_flags == 0u);
    if (expected_error != nullptr) {
      CHECK(completion_error.find(expected_error) != std::string::npos);
    }
    CHECK(result.flags == UINT32_C(0xc35aa53c));
    if (expected == XTBLOOM_STATUS_SUCCESS) {
      CHECK(std::isfinite(energies[0]));
      CHECK(iterations[0] > 0 && iterations[0] <= options.max_scc_iterations);
      CHECK(converged[0] == 1u);
      CHECK(statuses[0] == XTBLOOM_STATUS_SUCCESS);
    } else {
      CHECK(energies[0] == kEnergyCanary);
      CHECK(iterations[0] == kIterationCanary);
      CHECK(converged[0] == kConvergedCanary);
      CHECK(statuses[0] == kStatusCanary);
    }
    return 0;
  };

  CHECK(submit(XTBLOOM_STATUS_SUCCESS) == 0);
  const Gfn2CudaExecutionIdentity identity = cache->identity();
  RequestStateSnapshot baseline;
  CHECK(download_request_state_snapshot(identity, stream, false, baseline) == 0);
  CHECK(baseline.inference.warm_generations[0] == baseline.refresh.epoch);

  const auto verify_post_launch_failure = [&](Gfn2CudaExecutionTestFault fault,
                                              bool expect_inline = true) {
    arm_gfn2_cuda_execution_test_fault(fault);
    CHECK(submit(XTBLOOM_STATUS_INTERNAL_ERROR, expect_inline) == 0);
    CHECK(cache->identity().warm_checkpoint_ready == 0u);
    CHECK(cache->execute_inference_async(Gfn2CudaSccStartMode::kWarm, error) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(cache->execute_inference_async(Gfn2CudaSccStartMode::kFresh, error) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(download_request_state_snapshot(cache->identity(), stream, false, baseline) == 0);
    CHECK(baseline.inference.warm_generations[0] == baseline.refresh.epoch);
    return 0;
  };
  CHECK(verify_post_launch_failure(Gfn2CudaExecutionTestFault::kRequestPrepareSubmission) == 0);
  CHECK(verify_post_launch_failure(Gfn2CudaExecutionTestFault::kRequestCommitSubmission) == 0);
  CHECK(verify_post_launch_failure(
            Gfn2CudaExecutionTestFault::kRequestPrepareSubmissionAndSettlement, false) == 0);
  CHECK(verify_post_launch_failure(
            Gfn2CudaExecutionTestFault::kRequestCommitSubmissionAndSettlement, false) == 0);

  const std::array<double, 9> valid_cell{4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0};
  const auto verify_rejected = [&](xtbloom_status_t expected) {
    CHECK(submit(expected) == 0);
    const Gfn2CudaExecutionIdentity after_identity = cache->identity();
    RequestStateSnapshot after;
    CHECK(download_request_state_snapshot(after_identity, stream, false, after) == 0);
    CHECK(same_refresh_snapshot(after.refresh, baseline.refresh));
    CHECK(same_inference_without_warm(after.inference, baseline.inference));
    CHECK(after.inference.warm_generations == std::vector<std::uint64_t>{0u});
    CHECK(after.scc_state == baseline.scc_state);
    CHECK(after.scc_workspace == baseline.scc_workspace);
    CHECK(after.numerical_body_count == baseline.numerical_body_count);
    CHECK(after_identity.warm_checkpoint_ready == 0u);
    CHECK(cache->execute_inference_async(Gfn2CudaSccStartMode::kWarm, error) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(cache->execute_inference_async(Gfn2CudaSccStartMode::kFresh, error) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(download_request_state_snapshot(cache->identity(), stream, false, baseline) == 0);
    CHECK(baseline.inference.warm_generations[0] == baseline.refresh.epoch);
    return 0;
  };

  std::copy(valid_cell.begin(), valid_cell.end(), batch.cell_matrices.begin());
  batch.periodic_axes[0] = XTBLOOM_PERIODIC_AXES_XYZ;
  batch.bind();
  CHECK(verify_rejected(XTBLOOM_STATUS_NOT_IMPLEMENTED) == 0);

  batch.periodic_axes[0] = XTBLOOM_PERIODIC_AXIS_X;
  batch.bind();
  CHECK(verify_rejected(XTBLOOM_STATUS_NOT_SUPPORTED) == 0);

  batch.periodic_axes[0] = 8;
  batch.bind();
  CHECK(verify_rejected(XTBLOOM_STATUS_INVALID_ARGUMENT) == 0);

  /* A device-resident immutable topology mismatch has higher priority than
   * every lattice availability/semantic reason, matching the synchronous
   * HOST comparison. Exercise all lattice pointer placements so atomicMax
   * cannot accidentally make the observable status depend on which leaf was
   * staged from HOST versus CUDA_DEVICE. */
  std::vector<double> mismatched_charges = batch.molecular_charges;
  mismatched_charges[0] += 0.25;
  const std::vector<double> priority_cells(valid_cell.begin(), valid_cell.end());
  DeviceBuffer<double> device_mismatched_charges;
  DeviceBuffer<double> device_cells;
  DeviceBuffer<std::int32_t> device_axes;
  CHECK(device_mismatched_charges.upload(mismatched_charges, stream) == cudaSuccess);
  CHECK(device_cells.upload(priority_cells, stream) == cudaSuccess);
  const std::array<std::int32_t, 3> lattice_reasons{
      XTBLOOM_PERIODIC_AXES_XYZ,
      XTBLOOM_PERIODIC_AXIS_X,
      8,
  };
  bool priority_baseline_captured = false;
  RequestStateSnapshot priority_baseline;
  for (std::int32_t lattice_reason : lattice_reasons) {
    const std::vector<std::int32_t> axes{lattice_reason};
    CHECK(device_axes.upload(axes, stream) == cudaSuccess);
    for (int layout = 0; layout < 4; ++layout) {
      batch.bind();
      batch.descriptor.molecular_charges = device_mismatched_charges.view();
      batch.descriptor.cell_matrices =
          layout == 0 || layout == 3 ? host_buffer(priority_cells) : device_cells.view();
      batch.descriptor.periodic_axes =
          layout == 0 || layout == 2 ? host_buffer(axes) : device_axes.view();
      CHECK(submit(XTBLOOM_STATUS_INVALID_ARGUMENT, false, "topology") == 0);
      RequestStateSnapshot after;
      CHECK(download_request_state_snapshot(cache->identity(), stream, false, after) == 0);
      if (!priority_baseline_captured) {
        priority_baseline = after;
        priority_baseline_captured = true;
      } else {
        CHECK(same_refresh_snapshot(after.refresh, priority_baseline.refresh));
        CHECK(same_inference_without_warm(after.inference, priority_baseline.inference));
        CHECK(after.scc_state == priority_baseline.scc_state);
        CHECK(after.scc_workspace == priority_baseline.scc_workspace);
        CHECK(after.numerical_body_count == priority_baseline.numerical_body_count);
      }
      CHECK(after.inference.warm_generations == std::vector<std::uint64_t>{0u});
    }
  }
  CHECK(cache->execute_inference_async(Gfn2CudaSccStartMode::kWarm, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(cache->execute_inference_async(Gfn2CudaSccStartMode::kFresh, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(download_request_state_snapshot(cache->identity(), stream, false, baseline) == 0);
  CHECK(baseline.inference.warm_generations[0] == baseline.refresh.epoch);

  batch.molecular_charges = mismatched_charges;
  batch.molecular_charges[0] -= 0.25;
  batch.cell_matrices.assign(9u, 0.0);
  batch.periodic_axes.assign(1u, XTBLOOM_PERIODIC_AXES_NONE);
  std::fill(batch.cell_matrices.begin(), batch.cell_matrices.end(), 0.0);
  batch.bind();
  arm_gfn2_cuda_execution_test_fault(Gfn2CudaExecutionTestFault::kUnknownRequestValidationCode);
  CHECK(verify_rejected(XTBLOOM_STATUS_INTERNAL_ERROR) == 0);
  return 0;
}

int test_bounded_fallback_preserves_synchronous_plan(cudaStream_t stream,
                                                     std::int32_t device_id) {
  auto cache = std::make_shared<Gfn2CudaExecutionCache>(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(homogeneous_case_options(1, SmallSystemKind::kHe, false, false, false),
                            host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, false);
  xtbloom_compute_options_t options = compute_options(false);
  options.max_scc_iterations = 32;

  arm_gfn2_cuda_execution_test_fault(
      Gfn2CudaExecutionTestFault::kSccProviderUncapturedFallback);
  CHECK(cache->prepare_topology_only(batch.descriptor, options, error) == XTBLOOM_STATUS_SUCCESS);
  const Gfn2CudaExecutionIdentity identity = cache->identity();
  CHECK(identity.scc_conditional_graph_ready == 0u);
  CHECK(identity.scc_loop_fallback_reason != 0u);

  constexpr double kEnergyCanary = 8125.75;
  constexpr std::int32_t kIterationCanary = INT32_C(0x13572468);
  constexpr std::uint8_t kConvergedCanary = UINT8_C(0xa5);
  constexpr xtbloom_status_t kStatusCanary = INT32_C(0x24681357);
  std::vector<double> energies(1u, kEnergyCanary);
  std::vector<std::int32_t> iterations(1u, kIterationCanary);
  std::vector<std::uint8_t> converged(1u, kConvergedCanary);
  std::vector<xtbloom_status_t> statuses(1u, kStatusCanary);
  xtbloom_batch_result_t result{};
  CHECK(xtbloom_batch_result_init(&result, sizeof(result)) == XTBLOOM_STATUS_SUCCESS);
  result.flags = UINT32_C(0xc35aa53c);
  result.energies = mutable_host_buffer(energies);
  result.scc_iterations = mutable_host_buffer(iterations);
  result.scc_converged = mutable_host_buffer(converged);
  result.per_system_status = mutable_host_buffer(statuses);

  CHECK(execute_restricted_gfn2_cuda_plan(*cache, batch.descriptor, options, result, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(std::isfinite(energies[0]));
  CHECK(iterations[0] > 0 && iterations[0] <= options.max_scc_iterations);
  CHECK(converged[0] == 1u);
  CHECK(statuses[0] == XTBLOOM_STATUS_SUCCESS);

  energies[0] = kEnergyCanary;
  iterations[0] = kIterationCanary;
  converged[0] = kConvergedCanary;
  statuses[0] = kStatusCanary;
  result.flags = UINT32_C(0xc35aa53c);
  RequestSubmission submission;
  CHECK(enqueue_restricted_gfn2_cuda_plan(cache, batch.descriptor, options, result, submission,
                                          error) == XTBLOOM_STATUS_NOT_SUPPORTED);
  CHECK(submission.pending == nullptr);
  CHECK(!submission.completed_inline);
  CHECK(result.flags == UINT32_C(0xc35aa53c));
  CHECK(energies[0] == kEnergyCanary);
  CHECK(iterations[0] == kIterationCanary);
  CHECK(converged[0] == kConvergedCanary);
  CHECK(statuses[0] == kStatusCanary);
  return 0;
}

int test_failed_refresh_revokes_warm_checkpoint(cudaStream_t stream, std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(homogeneous_case_options(1, SmallSystemKind::kHe, false, false, false),
                            host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, false);
  xtbloom_compute_options_t options = compute_options(false);
  options.max_scc_iterations = 32;
  bool reused = true;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(cache.execute_inference_async(Gfn2CudaSccStartMode::kFresh, error) ==
        XTBLOOM_STATUS_SUCCESS);
  InferenceSnapshot checkpoint;
  CHECK(download_inference_snapshot(cache.identity(), stream, false, checkpoint) == 0);
  CHECK(checkpoint.warm_generations[0] == 1u);

  /* A peer-local preprocessing failure rolls back its numerical publication.
   * It must also revoke the otherwise consumable electronic checkpoint. */
  batch.positions[0] = std::numeric_limits<double>::quiet_NaN();
  batch.bind();
  Gfn2CudaNumericalInputView numerical{};
  numerical.positions = batch.descriptor.positions;
  numerical.point_charge_positions = batch.descriptor.point_charge_positions;
  numerical.point_charge_values = batch.descriptor.point_charge_values;
  numerical.point_charge_gammas = batch.descriptor.point_charge_gammas;
  numerical.atomic_potential_shifts = batch.descriptor.atomic_potential_shifts;
  numerical.charge_response_matrix = batch.descriptor.charge_response_matrix;
  numerical.requested_mask = {nullptr, 0u, XTBLOOM_MEMORY_HOST, 0u};
  CHECK(cache.refresh_numerical_async(numerical, error) == XTBLOOM_STATUS_SUCCESS);
  RefreshSnapshot failed_refresh;
  CHECK(download_refresh_snapshot(cache.identity(), stream, failed_refresh) == 0);
  CHECK(failed_refresh.epoch == 2u);
  CHECK(failed_refresh.committed[0] == 1u);
  CHECK(failed_refresh.eligible[0] == 0u);
  InferenceSnapshot revoked;
  CHECK(download_inference_snapshot(cache.identity(), stream, false, revoked) == 0);
  CHECK(revoked.warm_generations[0] == 0u);
  return 0;
}

int test_publication_plan_failure_provenance(cudaStream_t stream, std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(homogeneous_case_options(1, SmallSystemKind::kH2, false, false, false),
                            host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, false);
  xtbloom_compute_options_t options = compute_options(false);
  options.max_scc_iterations = 32;
  bool reused = true;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(cache.execute_inference_async(Gfn2CudaSccStartMode::kFresh, error) ==
        XTBLOOM_STATUS_SUCCESS);
  InferenceSnapshot epoch_one;
  CHECK(download_inference_snapshot(cache.identity(), stream, false, epoch_one) == 0);
  CHECK(epoch_one.publication_epoch == 1u);
  CHECK(epoch_one.publication_plan_error ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationPlanError::kSuccess));

  batch.positions[3] += 0.047;
  batch.bind();
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(reused);
  RefreshSnapshot refreshed;
  CHECK(download_refresh_snapshot(cache.identity(), stream, refreshed) == 0);
  CHECK(refreshed.epoch == 2u);
  CHECK(refreshed.committed[0] == 2u);

  /* A malformed device eligibility byte fails the upstream terminal plan and
   * is propagated as a whole-plan publication failure. Result bytes stay at
   * epoch one, while epoch/error provenance records that epoch two did not
   * publish. */
  const std::uint8_t invalid_eligibility = 2u;
  CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(cache.identity().numerical_eligible_mask),
                             &invalid_eligibility, sizeof(invalid_eligibility),
                             cudaMemcpyHostToDevice, stream));
  CHECK(cache.execute_inference_async(Gfn2CudaSccStartMode::kFresh, error) ==
        XTBLOOM_STATUS_SUCCESS);
  InferenceSnapshot failed;
  CHECK(download_inference_snapshot(cache.identity(), stream, false, failed) == 0);
  CHECK(failed.publication_epoch == 2u);
  CHECK(
      failed.publication_plan_error ==
      static_cast<std::uint32_t>(Gfn2InferencePublicationPlanError::kTerminalClassicalPlanFailure));
  CHECK(failed.publication_system_errors[0] ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationSystemError::kSuccess));
  CHECK(failed.energies == epoch_one.energies);
  CHECK(failed.iterations == epoch_one.iterations);
  CHECK(failed.converged == epoch_one.converged);
  CHECK(failed.statuses == epoch_one.statuses);
  CHECK(failed.warm_generations[0] == 0u);

  const std::uint8_t valid_eligibility = 1u;
  CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(cache.identity().numerical_eligible_mask),
                             &valid_eligibility, sizeof(valid_eligibility), cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return 0;
}

int test_synchronous_public_result_transaction(cudaStream_t stream, std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(homogeneous_case_options(1, SmallSystemKind::kHe, false, false, false),
                            host, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, false);
  xtbloom_compute_options_t options = compute_options(false);
  options.max_scc_iterations = 32;

  constexpr double kEnergyCanary = 8125.75;
  constexpr std::int32_t kIterationCanary = INT32_C(0x13572468);
  constexpr std::uint8_t kConvergedCanary = UINT8_C(0xa5);
  constexpr xtbloom_status_t kStatusCanary = INT32_C(0x24681357);
  constexpr std::uint32_t kFlagsCanary = UINT32_C(0xc35aa53c);
  std::vector<double> energies(1, kEnergyCanary);
  std::vector<std::int32_t> iterations(1, kIterationCanary);
  std::vector<std::uint8_t> converged(1, kConvergedCanary);
  std::vector<xtbloom_status_t> statuses(1, kStatusCanary);
  xtbloom_batch_result_t result{};
  result.struct_size = XTBLOOM_BATCH_RESULT_V1_SIZE;
  result.api_version = XTBLOOM_API_VERSION;
  result.flags = kFlagsCanary;
  result.energies = mutable_host_buffer(energies);
  result.scc_iterations = mutable_host_buffer(iterations);
  result.scc_converged = mutable_host_buffer(converged);
  result.per_system_status = mutable_host_buffer(statuses);

  CHECK(execute_restricted_gfn2_cuda(cache, batch.descriptor, options, result, error) ==
        XTBLOOM_STATUS_SUCCESS);
  /* The synchronous owner must observe commit completion before publishing
   * any pinned result image or result.flags to the caller. */
  CHECK(result.flags == 0u);
  CHECK(std::isfinite(energies[0]));
  CHECK(iterations[0] > 0);
  CHECK(converged[0] == 1u);
  CHECK(statuses[0] == XTBLOOM_STATUS_SUCCESS);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::int32_t device_id = -1;
  CUDA_CHECK(cudaGetDevice(&device_id));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  const bool native_lattice_lifecycle_only =
      argc == 2 && std::strcmp(argv[1], "--native-lattice-lifecycle") == 0;
  if (native_lattice_lifecycle_only) {
    int status = test_native_lattice_allocation_failure_recovery(stream, device_id);
    if (status == 0) status = test_native_lattice_staging_reuse(stream, device_id);
    if (status == 0) status = test_native_lattice_teardown_quarantine(stream, device_id);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return status;
  }

  const bool refresh_chain_only =
      argc == 2 && std::strcmp(argv[1], "--refresh-chain") == 0;
  if (refresh_chain_only) {
    const int status = test_fresh_warm_inference_and_post_scc_refresh(stream, device_id);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return status;
  }

  const bool request_state_only =
      argc == 2 && std::strcmp(argv[1], "--request-state") == 0;
  if (request_state_only) {
    const int status = test_rejected_async_request_preserves_internal_state(stream, device_id);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return status;
  }

  const bool bounded_fallback_only =
      argc == 2 && std::strcmp(argv[1], "--bounded-fallback") == 0;
  if (bounded_fallback_only) {
    const int status = test_bounded_fallback_preserves_synchronous_plan(stream, device_id);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return status;
  }

  int status = test_context_owned_runtime(stream, device_id);
  if (status == 0) status = test_base_configuration(stream, device_id);
  if (status == 0) status = test_energy_only_configuration(stream, device_id);
  if (status == 0) status = test_independent_optional_configurations(stream, device_id);
  if (status == 0) status = test_reuse_and_transactions(stream, device_id);
  if (status == 0) status = test_native_lattice_allocation_failure_recovery(stream, device_id);
  if (status == 0) status = test_native_lattice_staging_reuse(stream, device_id);
  if (status == 0) status = test_native_lattice_teardown_quarantine(stream, device_id);
  if (status == 0) status = test_device_refresh_and_peer_rollback(stream, device_id);
  if (status == 0) status = test_large_system_sparse_gate(stream, device_id);
  if (status == 0) status = test_host_refresh_snapshot_lifetime(stream, device_id);
  if (status == 0) status = test_refresh_cuda_graph_replay(stream, device_id);
  if (status == 0) status = test_host_refresh_rejected_during_cuda_graph_capture(stream, device_id);
  if (status == 0) {
    status = test_periodic_refresh_uses_zero_for_absent_optional_leaf(stream, device_id);
  }
  if (status == 0) status = test_ragged_runtime_shapes(stream, device_id);
  if (status == 0) status = test_topology_only_seed_factor_is_unpublished(stream, device_id);
  if (status == 0) status = test_fresh_warm_inference_and_post_scc_refresh(stream, device_id);
  if (status == 0) status = test_failed_refresh_revokes_warm_checkpoint(stream, device_id);
  if (status == 0) status = test_failed_inference_consumes_warm_checkpoint(stream, device_id);
  if (status == 0) status = test_rejected_async_request_preserves_internal_state(stream, device_id);
  if (status == 0) status = test_bounded_fallback_preserves_synchronous_plan(stream, device_id);
  if (status == 0) status = test_publication_plan_failure_provenance(stream, device_id);
  if (status == 0) status = test_synchronous_public_result_transaction(stream, device_id);
  if (status == 0) status = test_default_stream_refresh(device_id);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return status;
}
