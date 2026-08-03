#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_scc_potential.cuh"
#include "gpuxtb/gpuxtb.h"
#include "runtime/backend.hpp"
#include "runtime/gfn2_cuda_execution.hpp"
#include "tests/support/gfn2_scc_test_case.hpp"

#define CHECK(condition)                                                                 \
  do {                                                                                   \
    if (!(condition)) {                                                                  \
      std::fprintf(stderr, "CUDA runtime-owner check failed at line %d: %s\n", __LINE__, \
                   #condition);                                                          \
      return __LINE__;                                                                   \
    }                                                                                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using gpuxtb::detail::Context;
using gpuxtb::detail::Gfn2CudaExecutionCache;
using gpuxtb::detail::Gfn2CudaExecutionIdentity;
using gpuxtb::detail::Gfn2CudaNumericalInputView;
using gpuxtb::detail::cuda::Gfn2SccPotentialComponent;
using gpuxtb::test::gfn2::HostSccCase;
using gpuxtb::test::gfn2::HostSccCaseOptions;
using gpuxtb::test::gfn2::SmallSystemKind;

template <typename T>
gpuxtb_const_buffer_t host_buffer(const std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), GPUXTB_MEMORY_HOST,
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

  gpuxtb_const_buffer_t view() const noexcept {
    return {data_, count_ * sizeof(T), GPUXTB_MEMORY_CUDA_DEVICE, 0u};
  }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

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

struct PublicHostBatch {
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
  gpuxtb_batch_t descriptor{};

  void bind() noexcept {
    descriptor = {};
    descriptor.struct_size = GPUXTB_BATCH_V1_SIZE;
    descriptor.api_version = GPUXTB_API_VERSION;
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
    descriptor.point_charge_offsets = host_buffer(point_offsets);
    descriptor.point_charge_positions = host_buffer(point_positions);
    descriptor.point_charge_values = host_buffer(point_values);
    descriptor.point_charge_gammas = host_buffer(point_gammas);
    descriptor.atomic_potential_shifts = host_buffer(periodic_shifts);
    descriptor.charge_response_offsets = host_buffer(response_offsets);
    descriptor.charge_response_matrix = host_buffer(response_matrix);
  }

  static PublicHostBatch from_host(const HostSccCase& host, bool periodic_enabled) {
    PublicHostBatch batch;
    batch.atom_offsets = host.atom_offsets();
    batch.atomic_numbers = host.atomic_numbers();
    batch.positions = host.positions();
    batch.molecular_charges = host.molecular_charges();
    batch.unpaired_electrons = host.unpaired_electrons();
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

gpuxtb_compute_options_t compute_options(bool force_mode = true) noexcept {
  gpuxtb_compute_options_t options{};
  options.struct_size = GPUXTB_COMPUTE_OPTIONS_V1_SIZE;
  options.api_version = GPUXTB_API_VERSION;
  options.model = GPUXTB_MODEL_GFN2_XTB;
  options.flags = GPUXTB_COMPUTE_ENERGY;
  if (force_mode) {
    options.flags |=
        GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES | GPUXTB_COMPUTE_POINT_CHARGE_FORCES;
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
         first.batch_size == second.batch_size && first.total_atoms == second.total_atoms &&
         first.total_shells == second.total_shells &&
         first.total_orbitals == second.total_orbitals &&
         first.total_point_charges == second.total_point_charges &&
         first.solver_handle == second.solver_handle &&
         first.solver_parameters == second.solver_parameters &&
         first.blas_handle == second.blas_handle && first.topology_owner == second.topology_owner &&
         first.inputs_owner == second.inputs_owner &&
         first.eigensolver_owner == second.eigensolver_owner &&
         first.initializer_owner == second.initializer_owner &&
         first.scc_binding == second.scc_binding &&
         first.energy_force_descriptors == second.energy_force_descriptors &&
         first.topology_arena == second.topology_arena && first.input_arena == second.input_arena &&
         first.iteration_arena == second.iteration_arena &&
         first.eigensolver_setup_arena == second.eigensolver_setup_arena &&
         first.provider_host_workspace == second.provider_host_workspace &&
         first.force_immutable_arena == second.force_immutable_arena &&
         first.force_execution_arena == second.force_execution_arena &&
         first.numerical_refresh_arena == second.numerical_refresh_arena &&
         first.numerical_refresh_binding == second.numerical_refresh_binding &&
         first.numerical_epoch == second.numerical_epoch &&
         first.committed_generations == second.committed_generations &&
         first.numerical_eligible_mask == second.numerical_eligible_mask &&
         first.overlap_factor_generations == second.overlap_factor_generations &&
         first.overlap_factor_statuses == second.overlap_factor_statuses &&
         first.topology_arena_bytes == second.topology_arena_bytes &&
         first.input_arena_bytes == second.input_arena_bytes &&
         first.iteration_arena_bytes == second.iteration_arena_bytes &&
         first.eigensolver_setup_arena_bytes == second.eigensolver_setup_arena_bytes &&
         first.provider_host_workspace_bytes == second.provider_host_workspace_bytes &&
         first.force_immutable_arena_bytes == second.force_immutable_arena_bytes &&
         first.force_execution_arena_bytes == second.force_execution_arena_bytes &&
         first.numerical_refresh_arena_bytes == second.numerical_refresh_arena_bytes;
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
  CHECK(identity.topology_arena_bytes > 0u);
  CHECK(identity.input_arena_bytes > 0u);
  CHECK(identity.iteration_arena_bytes > 0u);
  CHECK(identity.eigensolver_setup_arena_bytes > 0u);
  CHECK(identity.force_immutable_arena_bytes > 0u || !force_mode);
  CHECK((identity.force_immutable_arena == 0u) == !force_mode);
  CHECK(identity.force_execution_arena_bytes > 0u);
  CHECK(identity.numerical_refresh_arena_bytes > 0u);
  CHECK(identity.numerical_refresh_binding != 0u);
  CHECK(identity.numerical_epoch != 0u);
  CHECK(identity.committed_generations != 0u);
  CHECK(identity.numerical_eligible_mask != 0u);
  CHECK(identity.overlap_factor_generations != 0u);
  CHECK(identity.overlap_factor_statuses != 0u);

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
  gpuxtb_compute_options_t options = compute_options();
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    HostSccCase host;
    std::string error;
    CHECK(HostSccCase::create(case_options(batch_size, true), host, error) ==
          GPUXTB_STATUS_SUCCESS);
    PublicHostBatch batch = PublicHostBatch::from_host(host, true);
    bool reused = true;
    const gpuxtb_status_t status = cache.prepare_host(batch.descriptor, options, reused, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      std::fprintf(stderr, "batch %lld runtime setup failed: status=%d error=%s\n",
                   static_cast<long long>(batch_size), status, error.c_str());
    }
    CHECK(status == GPUXTB_STATUS_SUCCESS);
    CHECK(!reused);
    CHECK(cache.valid());
    const Gfn2CudaExecutionIdentity initial = cache.identity();
    CHECK(validate_identity(initial, batch_size, true, true, true) == 0);
    batch.positions[0] += 0.01;
    batch.bind();
    CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == GPUXTB_STATUS_SUCCESS);
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

int test_reuse_and_transactions(cudaStream_t stream, std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(case_options(8, true), host, error) == GPUXTB_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, true);
  gpuxtb_compute_options_t options = compute_options();
  bool reused = true;
  const gpuxtb_status_t initial_status =
      cache.prepare_host(batch.descriptor, options, reused, error);
  if (initial_status != GPUXTB_STATUS_SUCCESS) {
    std::fprintf(stderr, "reuse runtime setup failed: status=%d error=%s\n", initial_status,
                 error.c_str());
  }
  CHECK(initial_status == GPUXTB_STATUS_SUCCESS);
  CHECK(!reused);
  const Gfn2CudaExecutionIdentity initial = cache.identity();
  CHECK(validate_identity(initial, 8, true, true, true) == 0);

  /* Coordinates and all numerical QM/MM fields are intentionally excluded
   * from the topology key and refresh through stable runtime-owned staging. */
  batch.positions[0] += 0.037;
  batch.point_values[0] -= 0.05;
  batch.periodic_shifts[0] += 0.02;
  batch.bind();
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(reused);
  CHECK(same_identity(initial, cache.identity()));

  /* Descriptor and semantic failures must not evict the working cache. */
  const std::size_t good_atomic_number_bytes = batch.descriptor.atomic_numbers.size_bytes;
  batch.descriptor.atomic_numbers.size_bytes = good_atomic_number_bytes - 1u;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(!reused);
  CHECK(same_identity(initial, cache.identity()));
  batch.descriptor.atomic_numbers.size_bytes = good_atomic_number_bytes;

  /* A non-null zero-length response matrix must not keep an existing periodic
   * topology alive while omitting its dense count and offsets. This malformed
   * internal descriptor is rejected even when public validation is bypassed. */
  batch.descriptor.atomic_potential_shifts = {};
  batch.descriptor.total_charge_response_elements = 0;
  batch.descriptor.charge_response_offsets = {};
  batch.descriptor.charge_response_matrix = {batch.response_matrix.data(), 0u, GPUXTB_MEMORY_HOST,
                                             0u};
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
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
  const gpuxtb_status_t oversized_status =
      cache.prepare_host(batch.descriptor, options, reused, error);
  CHECK(oversized_status == GPUXTB_STATUS_ALLOCATION_FAILED ||
        oversized_status == GPUXTB_STATUS_INTERNAL_ERROR);
  CHECK(!reused);
  CHECK(!error.empty());
  CHECK(same_identity(initial, cache.identity()));
  batch.bind();

  batch.unpaired_electrons[3] = 1;
  batch.bind();
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) ==
        GPUXTB_STATUS_NOT_SUPPORTED);
  CHECK(same_identity(initial, cache.identity()));
  batch.unpaired_electrons[3] = 0;
  batch.bind();

  /* A compute-policy change is topology scoped. Handles remain context scoped
   * while owners, tokens, arenas, and the layout transaction are replaced. */
  ++options.max_scc_iterations;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == GPUXTB_STATUS_SUCCESS);
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
  const gpuxtb_status_t invalid_physics =
      cache.prepare_host(batch.descriptor, options, reused, error);
  CHECK(invalid_physics != GPUXTB_STATUS_SUCCESS);
  CHECK(same_identity(replaced, cache.identity()));
  return 0;
}

int test_device_refresh_and_peer_rollback(cudaStream_t stream, std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(case_options(8, true), host, error) == GPUXTB_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, true);
  gpuxtb_compute_options_t options = compute_options();
  bool reused = true;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == GPUXTB_STATUS_SUCCESS);
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
  CHECK(cache.refresh_numerical_async(numerical, error) == GPUXTB_STATUS_SUCCESS);
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
  CHECK(cache.refresh_numerical_async(numerical, error) == GPUXTB_STATUS_SUCCESS);
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
  CHECK(cache.refresh_numerical_async(numerical, error) == GPUXTB_STATUS_SUCCESS);
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
  Gfn2CudaNumericalInputView malformed = numerical;
  malformed.positions.size_bytes -= sizeof(double);
  CHECK(cache.refresh_numerical_async(malformed, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  RefreshSnapshot after_rejection;
  CHECK(download_refresh_snapshot(cache.identity(), stream, after_rejection) == 0);
  CHECK(after_rejection.epoch == third.epoch);
  CHECK(after_rejection.committed == third.committed);
  CHECK(after_rejection.factors == third.factors);
  return 0;
}

int test_refresh_cuda_graph_replay(cudaStream_t stream, std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(case_options(8, true), host, error) == GPUXTB_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, true);
  gpuxtb_compute_options_t options = compute_options();
  bool reused = true;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == GPUXTB_STATUS_SUCCESS);
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
  CHECK(cache.refresh_numerical_async(numerical, error) == GPUXTB_STATUS_SUCCESS);
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

int test_base_configuration(cudaStream_t stream, std::int32_t device_id) {
  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  HostSccCase host;
  std::string error;
  CHECK(HostSccCase::create(case_options(4, false), host, error) == GPUXTB_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, false);
  gpuxtb_compute_options_t options = compute_options();
  bool reused = true;
  const gpuxtb_status_t status = cache.prepare_host(batch.descriptor, options, reused, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    std::fprintf(stderr, "base runtime setup failed: status=%d error=%s\n", status, error.c_str());
  }
  CHECK(status == GPUXTB_STATUS_SUCCESS);
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
  CHECK(HostSccCase::create(case_options(8, true), host, error) == GPUXTB_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, true);
  gpuxtb_compute_options_t options = compute_options(false);
  bool reused = true;
  const gpuxtb_status_t status = cache.prepare_host(batch.descriptor, options, reused, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    std::fprintf(stderr, "energy-only runtime setup failed: status=%d error=%s\n", status,
                 error.c_str());
  }
  CHECK(status == GPUXTB_STATUS_SUCCESS);
  CHECK(!reused);
  CHECK(validate_identity(cache.identity(), 8, true, true, true, false) == 0);
  return 0;
}

int test_independent_optional_configurations(cudaStream_t stream, std::int32_t device_id) {
  struct Configuration {
    const char* name;
    SmallSystemKind system;
    bool d4;
    bool points;
    bool periodic;
  };
  constexpr std::array<Configuration, 4> configurations{{
      {"single-atom base", SmallSystemKind::kHe, false, false, false},
      {"D4 only", SmallSystemKind::kH2, true, false, false},
      {"point charge only", SmallSystemKind::kHe, false, true, false},
      {"periodic only", SmallSystemKind::kHe, false, false, true},
  }};

  for (const Configuration& configuration : configurations) {
    Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
    HostSccCase host;
    std::string error;
    CHECK(
        HostSccCase::create(homogeneous_case_options(4, configuration.system, configuration.d4,
                                                     configuration.points, configuration.periodic),
                            host, error) == GPUXTB_STATUS_SUCCESS);
    PublicHostBatch batch = PublicHostBatch::from_host(host, configuration.periodic);
    gpuxtb_compute_options_t options = compute_options();
    bool reused = true;
    const gpuxtb_status_t status = cache.prepare_host(batch.descriptor, options, reused, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      std::fprintf(stderr, "%s runtime setup failed: status=%d error=%s\n", configuration.name,
                   status, error.c_str());
    }
    CHECK(status == GPUXTB_STATUS_SUCCESS);
    CHECK(!reused);
    CHECK(validate_identity(cache.identity(), 4, configuration.d4, configuration.points,
                            configuration.periodic) == 0);
  }
  return 0;
}

int test_context_owned_runtime(cudaStream_t stream, std::int32_t device_id) {
  gpuxtb_context_options_t context_options{};
  context_options.struct_size = GPUXTB_CONTEXT_OPTIONS_V1_SIZE;
  context_options.api_version = GPUXTB_API_VERSION;
  context_options.backend = GPUXTB_BACKEND_CUDA;
  context_options.device_id = device_id;
  context_options.stream = reinterpret_cast<void*>(stream);

  Context* raw_context = nullptr;
  std::string error;
  CHECK(gpuxtb::detail::create_context(context_options, raw_context, error) ==
        GPUXTB_STATUS_SUCCESS);
  std::unique_ptr<Context> context(raw_context);
  CHECK(context != nullptr);
  CHECK(context->gfn2_cuda_execution_cache != nullptr);
  CHECK(!context->gfn2_cuda_execution_cache->valid());

  HostSccCase host;
  CHECK(HostSccCase::create(homogeneous_case_options(1, SmallSystemKind::kHe, false, false, false),
                            host, error) == GPUXTB_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, false);
  gpuxtb_compute_options_t options = compute_options();
  bool reused = true;
  CHECK(context->gfn2_cuda_execution_cache->prepare_host(batch.descriptor, options, reused,
                                                         error) == GPUXTB_STATUS_SUCCESS);
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
  CHECK(HostSccCase::create(case_options(8, true), host, error) == GPUXTB_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_host(host, true);
  gpuxtb_compute_options_t options = compute_options();
  bool reused = true;
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(!reused);
  batch.positions[0] += 0.018;
  batch.bind();
  CHECK(cache.prepare_host(batch.descriptor, options, reused, error) == GPUXTB_STATUS_SUCCESS);
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

}  // namespace

int main() {
  std::int32_t device_id = -1;
  CUDA_CHECK(cudaGetDevice(&device_id));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  int status = test_context_owned_runtime(stream, device_id);
  if (status == 0) status = test_base_configuration(stream, device_id);
  if (status == 0) status = test_energy_only_configuration(stream, device_id);
  if (status == 0) status = test_independent_optional_configurations(stream, device_id);
  if (status == 0) status = test_reuse_and_transactions(stream, device_id);
  if (status == 0) status = test_device_refresh_and_peer_rollback(stream, device_id);
  if (status == 0) status = test_refresh_cuda_graph_replay(stream, device_id);
  if (status == 0) status = test_ragged_runtime_shapes(stream, device_id);
  if (status == 0) status = test_default_stream_refresh(device_id);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return status;
}
