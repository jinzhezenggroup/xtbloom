#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_geometry.cuh"
#include "runtime/gfn2_cuda_execution.hpp"
#include "tests/support/gfn2_scc_test_case.hpp"
#include "xtbloom/xtbloom.h"

namespace {

using xtbloom::detail::Gfn2CudaExecutionCache;
using xtbloom::detail::Gfn2CudaExecutionIdentity;
using xtbloom::detail::Gfn2CudaNumericalInputView;
using xtbloom::detail::Gfn2CudaOpaqueBufferIdentity;
using xtbloom::test::gfn2::HostSccCase;
using xtbloom::test::gfn2::HostSccCaseOptions;
using xtbloom::test::gfn2::SmallSystemKind;

constexpr std::array<std::int64_t, 4> kBatchSizes{1, 8, 32, 128};
constexpr double kCacheTolerance = 1.0e-12;

const char* g_case = "uninitialized";

#define CHECK(condition)                                                                    \
  do {                                                                                      \
    if (!(condition)) {                                                                     \
      std::fprintf(stderr, "runtime cache diagnostics failed in %s at %s:%d: %s\n", g_case, \
                   __FILE__, __LINE__, #condition);                                         \
      return __LINE__;                                                                      \
    }                                                                                       \
  } while (false)

#define CUDA_CHECK(expression)                                                                    \
  do {                                                                                            \
    const cudaError_t status_ = (expression);                                                     \
    if (status_ != cudaSuccess) {                                                                 \
      std::fprintf(stderr, "runtime cache diagnostics CUDA failure in %s at %s:%d: %s\n", g_case, \
                   __FILE__, __LINE__, cudaGetErrorString(status_));                              \
      return __LINE__;                                                                            \
    }                                                                                             \
  } while (false)

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

  cudaError_t upload(const std::vector<T>& values, cudaStream_t stream) {
    if (values.size() != elements_) {
      if (data_ != nullptr) {
        (void)cudaFree(data_);
        data_ = nullptr;
      }
      elements_ = values.size();
      const cudaError_t status =
          elements_ == 0u ? cudaSuccess
                          : cudaMalloc(reinterpret_cast<void**>(&data_), elements_ * sizeof(T));
      if (status != cudaSuccess) return status;
    }
    return elements_ == 0u ? cudaSuccess
                           : cudaMemcpyAsync(data_, values.data(), elements_ * sizeof(T),
                                             cudaMemcpyHostToDevice, stream);
  }

  xtbloom_const_buffer_t view() const noexcept {
    return {data_, elements_ * sizeof(T), XTBLOOM_MEMORY_CUDA_DEVICE, 0u};
  }

 private:
  T* data_ = nullptr;
  std::size_t elements_ = 0u;
};

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
  std::vector<double> periodic_response;
  xtbloom_batch_t descriptor{};

  static PublicHostBatch from_fixture(const HostSccCase& fixture, bool periodic) {
    PublicHostBatch batch;
    batch.atom_offsets = fixture.atom_offsets();
    batch.atomic_numbers = fixture.atomic_numbers();
    batch.positions = fixture.positions();
    batch.molecular_charges = fixture.molecular_charges();
    batch.unpaired_electrons = fixture.unpaired_electrons();
    batch.point_offsets = fixture.point_charge_offsets();
    batch.point_positions = fixture.point_charge_positions();
    batch.point_values = fixture.point_charge_charges();
    batch.point_gammas = fixture.point_charge_hardnesses();
    if (periodic) {
      batch.periodic_shifts = fixture.periodic_shifts();
      batch.periodic_response = fixture.periodic_response_matrices();
      batch.response_offsets = fixture.periodic_plan()->matrix_offsets();
    }
    batch.bind();
    return batch;
  }

  void bind() noexcept {
    (void)xtbloom_batch_init(&descriptor, sizeof(descriptor));
    descriptor.batch_size = static_cast<std::int64_t>(molecular_charges.size());
    descriptor.total_atoms = static_cast<std::int64_t>(atomic_numbers.size());
    descriptor.total_point_charges = static_cast<std::int64_t>(point_values.size());
    descriptor.total_charge_response_elements = static_cast<std::int64_t>(periodic_response.size());
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
    descriptor.charge_response_matrix = host_buffer(periodic_response);
  }

  /* Distort every H2 bond and independently change every optional numerical
   * source.  Topology-defining offsets, counts, and atomic identities remain
   * fixed, so every call must use the sealed refresh transaction. */
  void perturb(int phase) noexcept {
    const std::int64_t batch_size = static_cast<std::int64_t>(atom_offsets.size() - 1u);
    for (std::int64_t system = 0; system < batch_size; ++system) {
      const std::int64_t atom_begin = atom_offsets[static_cast<std::size_t>(system)];
      const std::int64_t atom_end = atom_offsets[static_cast<std::size_t>(system + 1)];
      const double step = 0.004 + 0.0004 * static_cast<double>((system + phase) % 5);
      positions[static_cast<std::size_t>(3 * atom_begin)] -= step;
      positions[static_cast<std::size_t>(3 * atom_begin + 2)] -= 0.2 * step;
      positions[static_cast<std::size_t>(3 * (atom_end - 1))] += 0.7 * step;
      positions[static_cast<std::size_t>(3 * (atom_end - 1) + 1)] += 0.3 * step;
      positions[static_cast<std::size_t>(3 * (atom_end - 1) + 2)] += 0.4 * step;

      if (!point_values.empty()) {
        const std::int64_t point_begin = point_offsets[static_cast<std::size_t>(system)];
        const std::int64_t point_end = point_offsets[static_cast<std::size_t>(system + 1)];
        for (std::int64_t point = point_begin; point < point_end; ++point) {
          point_positions[static_cast<std::size_t>(3 * point)] += 0.45 * step;
          point_positions[static_cast<std::size_t>(3 * point + 1)] -= 0.25 * step;
          point_positions[static_cast<std::size_t>(3 * point + 2)] += 0.15 * step;
          point_values[static_cast<std::size_t>(point)] +=
              (system % 2 == 0 ? 1.0 : -1.0) * 0.05 * step;
          point_gammas[static_cast<std::size_t>(point)] += 0.08 * step;
        }
      }

      if (!periodic_shifts.empty()) {
        for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
          periodic_shifts[static_cast<std::size_t>(atom)] +=
              (atom == atom_begin ? 1.0 : -0.6) * 0.2 * step;
        }
        const std::int64_t response_begin = response_offsets[static_cast<std::size_t>(system)];
        const std::int64_t atoms = atom_end - atom_begin;
        for (std::int64_t row = 0; row < atoms; ++row) {
          for (std::int64_t column = row; column < atoms; ++column) {
            const double delta = (row == column ? 0.09 : 0.04) * step;
            periodic_response[static_cast<std::size_t>(response_begin + row * atoms + column)] +=
                delta;
            if (row != column) {
              periodic_response[static_cast<std::size_t>(response_begin + column * atoms + row)] +=
                  delta;
            }
          }
        }
      }
    }
    bind();
  }
};

Gfn2CudaNumericalInputView host_view(const PublicHostBatch& batch,
                                     const std::vector<std::uint8_t>& requested) noexcept {
  Gfn2CudaNumericalInputView view{};
  view.positions = host_buffer(batch.positions);
  view.point_charge_positions = host_buffer(batch.point_positions);
  view.point_charge_values = host_buffer(batch.point_values);
  view.point_charge_gammas = host_buffer(batch.point_gammas);
  view.atomic_potential_shifts = host_buffer(batch.periodic_shifts);
  view.charge_response_matrix = host_buffer(batch.periodic_response);
  view.requested_mask = host_buffer(requested);
  return view;
}

struct DeviceNumericalInput {
  DeviceBuffer<double> positions;
  DeviceBuffer<double> point_positions;
  DeviceBuffer<double> point_values;
  DeviceBuffer<double> point_gammas;
  DeviceBuffer<double> periodic_shifts;
  DeviceBuffer<double> periodic_response;
  DeviceBuffer<std::uint8_t> requested;

  cudaError_t upload(const PublicHostBatch& batch, const std::vector<std::uint8_t>& mask,
                     cudaStream_t stream) {
    cudaError_t status = positions.upload(batch.positions, stream);
    if (status == cudaSuccess) status = point_positions.upload(batch.point_positions, stream);
    if (status == cudaSuccess) status = point_values.upload(batch.point_values, stream);
    if (status == cudaSuccess) status = point_gammas.upload(batch.point_gammas, stream);
    if (status == cudaSuccess) status = periodic_shifts.upload(batch.periodic_shifts, stream);
    if (status == cudaSuccess) status = periodic_response.upload(batch.periodic_response, stream);
    if (status == cudaSuccess) status = requested.upload(mask, stream);
    return status;
  }

  Gfn2CudaNumericalInputView view() const noexcept {
    Gfn2CudaNumericalInputView result{};
    result.positions = positions.view();
    result.point_charge_positions = point_positions.view();
    result.point_charge_values = point_values.view();
    result.point_charge_gammas = point_gammas.view();
    result.atomic_potential_shifts = periodic_shifts.view();
    result.charge_response_matrix = periodic_response.view();
    result.requested_mask = requested.view();
    return result;
  }
};

struct CacheSnapshot {
  std::uint64_t epoch = 0u;
  std::vector<std::uint64_t> committed_generations;
  std::vector<std::uint64_t> factor_generations;
  std::vector<std::uint32_t> factor_statuses;
  std::vector<std::uint8_t> eligible;

  std::vector<double> positions;
  std::vector<double> geometry_pairs;
  std::vector<double> coordination;
  std::vector<double> overlap;
  std::vector<double> dipole;
  std::vector<double> quadrupole;
  std::vector<double> h0;
  std::vector<double> es2;
  std::vector<double> aes2;
  std::vector<double> d4_pairs;
  std::vector<double> d4_coordination;
  std::vector<double> point_positions;
  std::vector<double> point_values;
  std::vector<double> point_gammas;
  std::vector<double> point_shell_potential;
  std::vector<double> periodic_shifts;
  std::vector<double> periodic_response;
};

cudaError_t download_leaf(const Gfn2CudaOpaqueBufferIdentity& identity, std::vector<double>& output,
                          cudaStream_t stream) {
  if (identity.elements < 0) return cudaErrorInvalidValue;
  output.resize(static_cast<std::size_t>(identity.elements));
  if (output.empty()) return identity.address == 0u ? cudaSuccess : cudaErrorInvalidValue;
  if (identity.address == 0u) return cudaErrorInvalidDevicePointer;
  return cudaMemcpyAsync(output.data(), reinterpret_cast<const void*>(identity.address),
                         output.size() * sizeof(double), cudaMemcpyDeviceToHost, stream);
}

int download_snapshot(const Gfn2CudaExecutionIdentity& identity, cudaStream_t stream,
                      CacheSnapshot& snapshot) {
  const std::size_t batch = static_cast<std::size_t>(identity.batch_size);
  CHECK(identity.committed_generation_elements == identity.batch_size);
  CHECK(identity.numerical_eligible_elements == identity.batch_size);
  CHECK(identity.overlap_factor_generation_elements == identity.batch_size);
  CHECK(identity.overlap_factor_status_elements == identity.batch_size);
  snapshot.committed_generations.resize(batch);
  snapshot.factor_generations.resize(batch);
  snapshot.factor_statuses.resize(batch);
  snapshot.eligible.resize(batch);
  CUDA_CHECK(cudaMemcpyAsync(&snapshot.epoch,
                             reinterpret_cast<const void*>(identity.numerical_epoch),
                             sizeof(snapshot.epoch), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(snapshot.committed_generations.data(),
                             reinterpret_cast<const void*>(identity.committed_generations),
                             batch * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(snapshot.factor_generations.data(),
                             reinterpret_cast<const void*>(identity.overlap_factor_generations),
                             batch * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(snapshot.factor_statuses.data(),
                             reinterpret_cast<const void*>(identity.overlap_factor_statuses),
                             batch * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(snapshot.eligible.data(),
                             reinterpret_cast<const void*>(identity.numerical_eligible_mask),
                             batch * sizeof(std::uint8_t), cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(download_leaf(identity.committed_positions, snapshot.positions, stream));
  CUDA_CHECK(download_leaf(identity.committed_geometry_pairs, snapshot.geometry_pairs, stream));
  CUDA_CHECK(download_leaf(identity.committed_coordination_numbers, snapshot.coordination, stream));
  CUDA_CHECK(download_leaf(identity.committed_overlap, snapshot.overlap, stream));
  CUDA_CHECK(download_leaf(identity.committed_dipole_integrals, snapshot.dipole, stream));
  CUDA_CHECK(download_leaf(identity.committed_quadrupole_integrals, snapshot.quadrupole, stream));
  CUDA_CHECK(download_leaf(identity.committed_h0, snapshot.h0, stream));
  CUDA_CHECK(download_leaf(identity.committed_es2, snapshot.es2, stream));
  CUDA_CHECK(download_leaf(identity.committed_aes2, snapshot.aes2, stream));
  CUDA_CHECK(download_leaf(identity.committed_d4_pairs, snapshot.d4_pairs, stream));
  CUDA_CHECK(
      download_leaf(identity.committed_d4_coordination_numbers, snapshot.d4_coordination, stream));
  CUDA_CHECK(
      download_leaf(identity.committed_point_charge_positions, snapshot.point_positions, stream));
  CUDA_CHECK(download_leaf(identity.committed_point_charge_values, snapshot.point_values, stream));
  CUDA_CHECK(download_leaf(identity.committed_point_charge_gammas, snapshot.point_gammas, stream));
  CUDA_CHECK(download_leaf(identity.committed_point_charge_shell_potential,
                           snapshot.point_shell_potential, stream));
  CUDA_CHECK(download_leaf(identity.committed_periodic_shifts, snapshot.periodic_shifts, stream));
  CUDA_CHECK(
      download_leaf(identity.committed_periodic_response, snapshot.periodic_response, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return 0;
}

bool same_buffer_identity(const Gfn2CudaOpaqueBufferIdentity& first,
                          const Gfn2CudaOpaqueBufferIdentity& second) noexcept {
  return first.address == second.address && first.elements == second.elements;
}

bool stable_cache_identities(const Gfn2CudaExecutionIdentity& first,
                             const Gfn2CudaExecutionIdentity& second) noexcept {
  return first.topology_fingerprint == second.topology_fingerprint &&
         first.plan_token == second.plan_token && first.topology_owner == second.topology_owner &&
         first.inputs_owner == second.inputs_owner &&
         first.eigensolver_owner == second.eigensolver_owner &&
         first.numerical_refresh_arena == second.numerical_refresh_arena &&
         first.numerical_refresh_arena_bytes == second.numerical_refresh_arena_bytes &&
         first.numerical_refresh_binding == second.numerical_refresh_binding &&
         first.numerical_epoch == second.numerical_epoch &&
         first.committed_generations == second.committed_generations &&
         first.numerical_eligible_mask == second.numerical_eligible_mask &&
         first.overlap_factor_generations == second.overlap_factor_generations &&
         first.overlap_factor_statuses == second.overlap_factor_statuses &&
         first.committed_generation_elements == second.committed_generation_elements &&
         first.numerical_eligible_elements == second.numerical_eligible_elements &&
         first.overlap_factor_generation_elements == second.overlap_factor_generation_elements &&
         first.overlap_factor_status_elements == second.overlap_factor_status_elements &&
         same_buffer_identity(first.committed_positions, second.committed_positions) &&
         same_buffer_identity(first.committed_geometry_pairs, second.committed_geometry_pairs) &&
         same_buffer_identity(first.committed_coordination_numbers,
                              second.committed_coordination_numbers) &&
         same_buffer_identity(first.committed_overlap, second.committed_overlap) &&
         same_buffer_identity(first.committed_dipole_integrals,
                              second.committed_dipole_integrals) &&
         same_buffer_identity(first.committed_quadrupole_integrals,
                              second.committed_quadrupole_integrals) &&
         same_buffer_identity(first.committed_h0, second.committed_h0) &&
         same_buffer_identity(first.committed_es2, second.committed_es2) &&
         same_buffer_identity(first.committed_aes2, second.committed_aes2) &&
         same_buffer_identity(first.committed_d4_pairs, second.committed_d4_pairs) &&
         same_buffer_identity(first.committed_d4_coordination_numbers,
                              second.committed_d4_coordination_numbers) &&
         same_buffer_identity(first.committed_point_charge_positions,
                              second.committed_point_charge_positions) &&
         same_buffer_identity(first.committed_point_charge_values,
                              second.committed_point_charge_values) &&
         same_buffer_identity(first.committed_point_charge_gammas,
                              second.committed_point_charge_gammas) &&
         same_buffer_identity(first.committed_point_charge_shell_potential,
                              second.committed_point_charge_shell_potential) &&
         same_buffer_identity(first.committed_periodic_shifts, second.committed_periodic_shifts) &&
         same_buffer_identity(first.committed_periodic_response,
                              second.committed_periodic_response);
}

bool close(double first, double second) noexcept {
  const double scale = std::max({1.0, std::abs(first), std::abs(second)});
  return std::abs(first - second) <= kCacheTolerance * scale;
}

bool vectors_close(const std::vector<double>& first, const std::vector<double>& second) noexcept {
  return first.size() == second.size() &&
         std::equal(first.begin(), first.end(), second.begin(),
                    [](double lhs, double rhs) { return close(lhs, rhs); });
}

bool same_snapshot_values(const CacheSnapshot& first, const CacheSnapshot& second) noexcept {
  return first.epoch == second.epoch &&
         first.committed_generations == second.committed_generations &&
         first.factor_generations == second.factor_generations &&
         first.factor_statuses == second.factor_statuses && first.eligible == second.eligible &&
         vectors_close(first.positions, second.positions) &&
         vectors_close(first.geometry_pairs, second.geometry_pairs) &&
         vectors_close(first.coordination, second.coordination) &&
         vectors_close(first.overlap, second.overlap) &&
         vectors_close(first.dipole, second.dipole) &&
         vectors_close(first.quadrupole, second.quadrupole) && vectors_close(first.h0, second.h0) &&
         vectors_close(first.es2, second.es2) && vectors_close(first.aes2, second.aes2) &&
         vectors_close(first.d4_pairs, second.d4_pairs) &&
         vectors_close(first.d4_coordination, second.d4_coordination) &&
         vectors_close(first.point_positions, second.point_positions) &&
         vectors_close(first.point_values, second.point_values) &&
         vectors_close(first.point_gammas, second.point_gammas) &&
         vectors_close(first.point_shell_potential, second.point_shell_potential) &&
         vectors_close(first.periodic_shifts, second.periodic_shifts) &&
         vectors_close(first.periodic_response, second.periodic_response);
}

int validate_element_counts(const Gfn2CudaExecutionIdentity& identity, const HostSccCase& fixture,
                            bool d4, bool points, bool periodic) {
  const std::int64_t atoms = fixture.total_atoms();
  const std::int64_t matrices = fixture.mulliken_plan().matrix_elements();
  CHECK(identity.committed_positions.elements == atoms * 3);
  CHECK(identity.committed_geometry_pairs.elements ==
        fixture.aes2_plan().total_pairs() * xtbloom::detail::cuda::kGfn2GeometryPairDataElements);
  CHECK(identity.committed_coordination_numbers.elements == atoms);
  CHECK(identity.committed_overlap.elements == matrices);
  CHECK(identity.committed_dipole_integrals.elements == matrices * 3);
  CHECK(identity.committed_quadrupole_integrals.elements == matrices * 6);
  CHECK(identity.committed_h0.elements == matrices);
  CHECK(identity.committed_es2.elements == fixture.es2_cache().matrix_elements);
  CHECK(identity.committed_aes2.elements == fixture.aes2_cache().pair_data_elements);
  CHECK(identity.committed_d4_pairs.elements == (d4 ? fixture.d4_cache()->pair_data_elements : 0));
  CHECK(identity.committed_d4_coordination_numbers.elements == (d4 ? atoms : 0));
  CHECK(identity.committed_point_charge_positions.elements ==
        (points ? static_cast<std::int64_t>(fixture.point_charge_positions().size()) : 0));
  CHECK(identity.committed_point_charge_values.elements ==
        (points ? static_cast<std::int64_t>(fixture.point_charge_charges().size()) : 0));
  CHECK(identity.committed_point_charge_gammas.elements ==
        (points ? static_cast<std::int64_t>(fixture.point_charge_hardnesses().size()) : 0));
  CHECK(identity.committed_point_charge_shell_potential.elements ==
        (points ? identity.total_shells : 0));
  CHECK(identity.committed_periodic_shifts.elements == (periodic ? atoms : 0));
  CHECK(identity.committed_periodic_response.elements ==
        (periodic ? fixture.periodic_plan()->total_matrix_elements() : 0));
  return 0;
}

bool exact_range(const std::vector<double>& first, const std::vector<double>& second,
                 std::int64_t begin, std::int64_t end) noexcept {
  if (begin < 0 || end < begin || static_cast<std::size_t>(end) > first.size() ||
      first.size() != second.size()) {
    return false;
  }
  if (begin == end) return true;
  const std::size_t offset = static_cast<std::size_t>(begin);
  const std::size_t bytes = static_cast<std::size_t>(end - begin) * sizeof(double);
  return std::memcmp(first.data() + offset, second.data() + offset, bytes) == 0;
}

bool close_range(const std::vector<double>& first, const std::vector<double>& second,
                 std::int64_t begin, std::int64_t end) noexcept {
  if (begin < 0 || end < begin || static_cast<std::size_t>(end) > first.size() ||
      first.size() != second.size()) {
    return false;
  }
  for (std::int64_t index = begin; index < end; ++index) {
    if (!close(first[static_cast<std::size_t>(index)], second[static_cast<std::size_t>(index)])) {
      return false;
    }
  }
  return true;
}

bool changed_range(const std::vector<double>& first, const std::vector<double>& second,
                   std::int64_t begin, std::int64_t end) noexcept {
  if (begin < 0 || end <= begin || static_cast<std::size_t>(end) > first.size() ||
      first.size() != second.size()) {
    return false;
  }
  return !std::equal(first.begin() + begin, first.begin() + end, second.begin() + begin);
}

bool exact_components(const std::vector<double>& first, const std::vector<double>& second,
                      std::int64_t total_matrices, std::int64_t components, std::int64_t begin,
                      std::int64_t end) noexcept {
  for (std::int64_t component = 0; component < components; ++component) {
    if (!exact_range(first, second, component * total_matrices + begin,
                     component * total_matrices + end)) {
      return false;
    }
  }
  return true;
}

bool close_components(const std::vector<double>& first, const std::vector<double>& second,
                      std::int64_t total_matrices, std::int64_t components, std::int64_t begin,
                      std::int64_t end) noexcept {
  for (std::int64_t component = 0; component < components; ++component) {
    if (!close_range(first, second, component * total_matrices + begin,
                     component * total_matrices + end)) {
      return false;
    }
  }
  return true;
}

bool changed_components(const std::vector<double>& first, const std::vector<double>& second,
                        std::int64_t total_matrices, std::int64_t components, std::int64_t begin,
                        std::int64_t end) noexcept {
  for (std::int64_t component = 0; component < components; ++component) {
    if (changed_range(first, second, component * total_matrices + begin,
                      component * total_matrices + end)) {
      return true;
    }
  }
  return false;
}

std::int64_t pair_width(std::int64_t elements, std::int64_t total_pairs) noexcept {
  return total_pairs > 0 && elements % total_pairs == 0 ? elements / total_pairs : 0;
}

int verify_peer_unchanged(const CacheSnapshot& before, const CacheSnapshot& after,
                          const Gfn2CudaExecutionIdentity& identity, const HostSccCase& fixture,
                          std::int64_t system, bool d4, bool points, bool periodic) {
  const auto& atoms = fixture.atom_offsets();
  const auto& matrices = fixture.mulliken_plan().matrix_offsets();
  const auto& es2 = fixture.es2_plan().matrix_offsets();
  const auto& pairs = fixture.aes2_plan().pair_offsets();
  const auto& shells = fixture.basis_plan().batch_shell_offsets;
  const std::int64_t atom_begin = atoms[static_cast<std::size_t>(system)];
  const std::int64_t atom_end = atoms[static_cast<std::size_t>(system + 1)];
  const std::int64_t matrix_begin = matrices[static_cast<std::size_t>(system)];
  const std::int64_t matrix_end = matrices[static_cast<std::size_t>(system + 1)];
  const std::int64_t pair_begin = pairs[static_cast<std::size_t>(system)];
  const std::int64_t pair_end = pairs[static_cast<std::size_t>(system + 1)];
  const std::int64_t total_matrices = identity.committed_overlap.elements;
  CHECK(exact_range(before.positions, after.positions, atom_begin * 3, atom_end * 3));
  const std::int64_t geometry_width =
      pair_width(identity.committed_geometry_pairs.elements, fixture.aes2_plan().total_pairs());
  if (pair_end != pair_begin) {
    CHECK(geometry_width > 0);
    CHECK(exact_range(before.geometry_pairs, after.geometry_pairs, pair_begin * geometry_width,
                      pair_end * geometry_width));
  }
  CHECK(exact_range(before.coordination, after.coordination, atom_begin, atom_end));
  CHECK(exact_range(before.overlap, after.overlap, matrix_begin, matrix_end));
  CHECK(exact_components(before.dipole, after.dipole, total_matrices, 3, matrix_begin, matrix_end));
  CHECK(exact_components(before.quadrupole, after.quadrupole, total_matrices, 6, matrix_begin,
                         matrix_end));
  CHECK(exact_range(before.h0, after.h0, matrix_begin, matrix_end));
  CHECK(exact_range(before.es2, after.es2, es2[static_cast<std::size_t>(system)],
                    es2[static_cast<std::size_t>(system + 1)]));
  const std::int64_t aes2_width =
      pair_width(identity.committed_aes2.elements, fixture.aes2_plan().total_pairs());
  if (pair_end != pair_begin) {
    CHECK(aes2_width > 0);
    CHECK(exact_range(before.aes2, after.aes2, pair_begin * aes2_width, pair_end * aes2_width));
  }
  if (d4) {
    const auto& d4_offsets = fixture.d4_plan()->pair_offsets();
    const std::int64_t d4_width =
        pair_width(identity.committed_d4_pairs.elements, fixture.d4_plan()->total_pairs());
    CHECK(d4_width > 0);
    CHECK(exact_range(before.d4_pairs, after.d4_pairs,
                      d4_offsets[static_cast<std::size_t>(system)] * d4_width,
                      d4_offsets[static_cast<std::size_t>(system + 1)] * d4_width));
    CHECK(exact_range(before.d4_coordination, after.d4_coordination, atom_begin, atom_end));
  }
  if (points) {
    const auto& point_offsets = fixture.point_charge_offsets();
    const std::int64_t point_begin = point_offsets[static_cast<std::size_t>(system)];
    const std::int64_t point_end = point_offsets[static_cast<std::size_t>(system + 1)];
    CHECK(
        exact_range(before.point_positions, after.point_positions, point_begin * 3, point_end * 3));
    CHECK(exact_range(before.point_values, after.point_values, point_begin, point_end));
    CHECK(exact_range(before.point_gammas, after.point_gammas, point_begin, point_end));
    CHECK(exact_range(before.point_shell_potential, after.point_shell_potential,
                      shells[static_cast<std::size_t>(system)],
                      shells[static_cast<std::size_t>(system + 1)]));
  }
  if (periodic) {
    const auto& response = fixture.periodic_plan()->matrix_offsets();
    CHECK(exact_range(before.periodic_shifts, after.periodic_shifts, atom_begin, atom_end));
    CHECK(exact_range(before.periodic_response, after.periodic_response,
                      response[static_cast<std::size_t>(system)],
                      response[static_cast<std::size_t>(system + 1)]));
  }
  return 0;
}

/* Compare every scalar owned by one ragged peer against an independent cache
 * that refreshed the same finite input with every peer requested.  D/Q are
 * checked component plane by component plane, so a partial contiguous copy of
 * their global component-major layout cannot satisfy this oracle. */
int verify_peer_matches_reference(const CacheSnapshot& actual, const CacheSnapshot& reference,
                                  const Gfn2CudaExecutionIdentity& identity,
                                  const HostSccCase& fixture, std::int64_t system, bool d4,
                                  bool points, bool periodic) {
  const auto& atoms = fixture.atom_offsets();
  const auto& matrices = fixture.mulliken_plan().matrix_offsets();
  const auto& es2 = fixture.es2_plan().matrix_offsets();
  const auto& pairs = fixture.aes2_plan().pair_offsets();
  const auto& shells = fixture.basis_plan().batch_shell_offsets;
  const std::int64_t atom_begin = atoms[static_cast<std::size_t>(system)];
  const std::int64_t atom_end = atoms[static_cast<std::size_t>(system + 1)];
  const std::int64_t matrix_begin = matrices[static_cast<std::size_t>(system)];
  const std::int64_t matrix_end = matrices[static_cast<std::size_t>(system + 1)];
  const std::int64_t pair_begin = pairs[static_cast<std::size_t>(system)];
  const std::int64_t pair_end = pairs[static_cast<std::size_t>(system + 1)];
  const std::int64_t total_matrices = identity.committed_overlap.elements;
  CHECK(close_range(actual.positions, reference.positions, atom_begin * 3, atom_end * 3));
  const std::int64_t geometry_width =
      pair_width(identity.committed_geometry_pairs.elements, fixture.aes2_plan().total_pairs());
  if (pair_end != pair_begin) {
    CHECK(geometry_width > 0);
    CHECK(close_range(actual.geometry_pairs, reference.geometry_pairs, pair_begin * geometry_width,
                      pair_end * geometry_width));
  }
  CHECK(close_range(actual.coordination, reference.coordination, atom_begin, atom_end));
  CHECK(close_range(actual.overlap, reference.overlap, matrix_begin, matrix_end));
  CHECK(close_components(actual.dipole, reference.dipole, total_matrices, 3, matrix_begin,
                         matrix_end));
  CHECK(close_components(actual.quadrupole, reference.quadrupole, total_matrices, 6, matrix_begin,
                         matrix_end));
  CHECK(close_range(actual.h0, reference.h0, matrix_begin, matrix_end));
  CHECK(close_range(actual.es2, reference.es2, es2[static_cast<std::size_t>(system)],
                    es2[static_cast<std::size_t>(system + 1)]));
  const std::int64_t aes2_width =
      pair_width(identity.committed_aes2.elements, fixture.aes2_plan().total_pairs());
  if (pair_end != pair_begin) {
    CHECK(aes2_width > 0);
    CHECK(close_range(actual.aes2, reference.aes2, pair_begin * aes2_width, pair_end * aes2_width));
  }
  if (d4) {
    const auto& d4_offsets = fixture.d4_plan()->pair_offsets();
    const std::int64_t d4_width =
        pair_width(identity.committed_d4_pairs.elements, fixture.d4_plan()->total_pairs());
    CHECK(d4_width > 0);
    CHECK(close_range(actual.d4_pairs, reference.d4_pairs,
                      d4_offsets[static_cast<std::size_t>(system)] * d4_width,
                      d4_offsets[static_cast<std::size_t>(system + 1)] * d4_width));
    CHECK(close_range(actual.d4_coordination, reference.d4_coordination, atom_begin, atom_end));
  }
  if (points) {
    const auto& point_offsets = fixture.point_charge_offsets();
    const std::int64_t point_begin = point_offsets[static_cast<std::size_t>(system)];
    const std::int64_t point_end = point_offsets[static_cast<std::size_t>(system + 1)];
    CHECK(close_range(actual.point_positions, reference.point_positions, point_begin * 3,
                      point_end * 3));
    CHECK(close_range(actual.point_values, reference.point_values, point_begin, point_end));
    CHECK(close_range(actual.point_gammas, reference.point_gammas, point_begin, point_end));
    CHECK(close_range(actual.point_shell_potential, reference.point_shell_potential,
                      shells[static_cast<std::size_t>(system)],
                      shells[static_cast<std::size_t>(system + 1)]));
  }
  if (periodic) {
    const auto& response = fixture.periodic_plan()->matrix_offsets();
    CHECK(close_range(actual.periodic_shifts, reference.periodic_shifts, atom_begin, atom_end));
    CHECK(close_range(actual.periodic_response, reference.periodic_response,
                      response[static_cast<std::size_t>(system)],
                      response[static_cast<std::size_t>(system + 1)]));
  }
  return 0;
}

int verify_every_peer_leaf_changed(const CacheSnapshot& before, const CacheSnapshot& after,
                                   const Gfn2CudaExecutionIdentity& identity,
                                   const HostSccCase& fixture, std::int64_t system, bool d4,
                                   bool points, bool periodic) {
  const auto& atoms = fixture.atom_offsets();
  const auto& matrices = fixture.mulliken_plan().matrix_offsets();
  const auto& es2 = fixture.es2_plan().matrix_offsets();
  const auto& pairs = fixture.aes2_plan().pair_offsets();
  const auto& shells = fixture.basis_plan().batch_shell_offsets;
  const std::int64_t atom_begin = atoms[static_cast<std::size_t>(system)];
  const std::int64_t atom_end = atoms[static_cast<std::size_t>(system + 1)];
  const std::int64_t matrix_begin = matrices[static_cast<std::size_t>(system)];
  const std::int64_t matrix_end = matrices[static_cast<std::size_t>(system + 1)];
  const std::int64_t pair_begin = pairs[static_cast<std::size_t>(system)];
  const std::int64_t pair_end = pairs[static_cast<std::size_t>(system + 1)];
  const std::int64_t total_matrices = identity.committed_overlap.elements;
  CHECK(changed_range(before.positions, after.positions, atom_begin * 3, atom_end * 3));
  /* A monatomic base member has no relative geometry.  Its S/D/Q/H0,
   * electrostatic kernels, and zero-sized pair caches are translation
   * invariant, so positions are the only leaf whose mathematical value can
   * change.  Multiatom members below require every derived leaf to change. */
  if (pair_end == pair_begin) return 0;
  const std::int64_t geometry_width =
      pair_width(identity.committed_geometry_pairs.elements, fixture.aes2_plan().total_pairs());
  CHECK(geometry_width > 0);
  CHECK(changed_range(before.geometry_pairs, after.geometry_pairs, pair_begin * geometry_width,
                      pair_end * geometry_width));
  CHECK(changed_range(before.coordination, after.coordination, atom_begin, atom_end));
  CHECK(changed_range(before.overlap, after.overlap, matrix_begin, matrix_end));
  CHECK(
      changed_components(before.dipole, after.dipole, total_matrices, 3, matrix_begin, matrix_end));
  CHECK(changed_components(before.quadrupole, after.quadrupole, total_matrices, 6, matrix_begin,
                           matrix_end));
  CHECK(changed_range(before.h0, after.h0, matrix_begin, matrix_end));
  CHECK(changed_range(before.es2, after.es2, es2[static_cast<std::size_t>(system)],
                      es2[static_cast<std::size_t>(system + 1)]));
  const std::int64_t aes2_width =
      pair_width(identity.committed_aes2.elements, fixture.aes2_plan().total_pairs());
  CHECK(aes2_width > 0);
  CHECK(changed_range(before.aes2, after.aes2, pair_begin * aes2_width, pair_end * aes2_width));
  if (d4) {
    const auto& d4_offsets = fixture.d4_plan()->pair_offsets();
    const std::int64_t d4_width =
        pair_width(identity.committed_d4_pairs.elements, fixture.d4_plan()->total_pairs());
    CHECK(d4_width > 0);
    CHECK(changed_range(before.d4_pairs, after.d4_pairs,
                        d4_offsets[static_cast<std::size_t>(system)] * d4_width,
                        d4_offsets[static_cast<std::size_t>(system + 1)] * d4_width));
    CHECK(changed_range(before.d4_coordination, after.d4_coordination, atom_begin, atom_end));
  }
  if (points) {
    const auto& point_offsets = fixture.point_charge_offsets();
    const std::int64_t point_begin = point_offsets[static_cast<std::size_t>(system)];
    const std::int64_t point_end = point_offsets[static_cast<std::size_t>(system + 1)];
    CHECK(changed_range(before.point_positions, after.point_positions, point_begin * 3,
                        point_end * 3));
    CHECK(changed_range(before.point_values, after.point_values, point_begin, point_end));
    CHECK(changed_range(before.point_gammas, after.point_gammas, point_begin, point_end));
    CHECK(changed_range(before.point_shell_potential, after.point_shell_potential,
                        shells[static_cast<std::size_t>(system)],
                        shells[static_cast<std::size_t>(system + 1)]));
  }
  if (periodic) {
    const auto& response = fixture.periodic_plan()->matrix_offsets();
    CHECK(changed_range(before.periodic_shifts, after.periodic_shifts, atom_begin, atom_end));
    CHECK(changed_range(before.periodic_response, after.periodic_response,
                        response[static_cast<std::size_t>(system)],
                        response[static_cast<std::size_t>(system + 1)]));
  }
  return 0;
}

int verify_every_multipole_component_changed(const CacheSnapshot& before,
                                             const CacheSnapshot& after,
                                             const Gfn2CudaExecutionIdentity& identity,
                                             const HostSccCase& fixture, std::int64_t system) {
  const auto& pairs = fixture.aes2_plan().pair_offsets();
  if (pairs[static_cast<std::size_t>(system)] == pairs[static_cast<std::size_t>(system + 1)]) {
    return 0;
  }
  const auto& matrices = fixture.mulliken_plan().matrix_offsets();
  const std::int64_t begin = matrices[static_cast<std::size_t>(system)];
  const std::int64_t end = matrices[static_cast<std::size_t>(system + 1)];
  const std::int64_t total = identity.committed_overlap.elements;
  for (std::int64_t component = 0; component < 3; ++component) {
    CHECK(changed_range(before.dipole, after.dipole, component * total + begin,
                        component * total + end));
  }
  for (std::int64_t component = 0; component < 6; ++component) {
    CHECK(changed_range(before.quadrupole, after.quadrupole, component * total + begin,
                        component * total + end));
  }
  return 0;
}

xtbloom_compute_options_t compute_options() noexcept {
  xtbloom_compute_options_t options{};
  (void)xtbloom_compute_options_init(&options, sizeof(options));
  options.model = XTBLOOM_MODEL_GFN2_XTB;
  options.flags = XTBLOOM_COMPUTE_ENERGY;
  options.max_scc_iterations = 8;
  options.charge_tolerance = 1.0e-10;
  options.energy_tolerance = 1.0e-8;
  return options;
}

struct Scenario {
  const char* name;
  SmallSystemKind system;
  bool d4;
  bool points;
  bool periodic;
};

int run_case(const Scenario& scenario, std::int64_t batch_size, cudaStream_t stream,
             std::int32_t device_id) {
  char case_name[96]{};
  std::snprintf(case_name, sizeof(case_name), "%s/batch-%lld", scenario.name,
                static_cast<long long>(batch_size));
  g_case = case_name;

  HostSccCaseOptions fixture_options{};
  fixture_options.systems.assign(static_cast<std::size_t>(batch_size), scenario.system);
  fixture_options.enable_d4 = scenario.d4;
  fixture_options.enable_explicit_point_charges = scenario.points;
  fixture_options.enable_periodic_embedding = scenario.periodic;
  fixture_options.maximum_iterations = 8u;
  fixture_options.mixer_history = 4;
  HostSccCase fixture;
  std::string error;
  CHECK(HostSccCase::create(fixture_options, fixture, error) == XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch input = PublicHostBatch::from_fixture(fixture, scenario.periodic);

  Gfn2CudaExecutionCache host_cache(device_id, reinterpret_cast<void*>(stream));
  Gfn2CudaExecutionCache device_cache(device_id, reinterpret_cast<void*>(stream));
  Gfn2CudaExecutionCache reference_cache(device_id, reinterpret_cast<void*>(stream));
  const xtbloom_compute_options_t options = compute_options();
  bool reused = true;
  CHECK(host_cache.prepare_host(input.descriptor, options, reused, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  CHECK(device_cache.prepare_host(input.descriptor, options, reused, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  CHECK(reference_cache.prepare_host(input.descriptor, options, reused, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  const Gfn2CudaExecutionIdentity host_identity = host_cache.identity();
  const Gfn2CudaExecutionIdentity device_identity = device_cache.identity();
  const Gfn2CudaExecutionIdentity reference_identity = reference_cache.identity();
  CHECK(validate_element_counts(host_identity, fixture, scenario.d4, scenario.points,
                                scenario.periodic) == 0);
  CHECK(validate_element_counts(device_identity, fixture, scenario.d4, scenario.points,
                                scenario.periodic) == 0);
  CHECK(validate_element_counts(reference_identity, fixture, scenario.d4, scenario.points,
                                scenario.periodic) == 0);

  CacheSnapshot host_initial;
  CacheSnapshot device_initial;
  CacheSnapshot reference_initial;
  CHECK(download_snapshot(host_identity, stream, host_initial) == 0);
  CHECK(download_snapshot(device_identity, stream, device_initial) == 0);
  CHECK(download_snapshot(reference_identity, stream, reference_initial) == 0);
  CHECK(same_snapshot_values(host_initial, device_initial));
  CHECK(same_snapshot_values(host_initial, reference_initial));
  CHECK(host_initial.epoch == 1u);
  for (std::int64_t system = 0; system < batch_size; ++system) {
    const std::size_t index = static_cast<std::size_t>(system);
    CHECK(host_initial.committed_generations[index] == 1u);
    CHECK(host_initial.factor_generations[index] == 1u);
    CHECK(host_initial.factor_statuses[index] == 0u);
    CHECK(host_initial.eligible[index] == 1u);
  }

  DeviceNumericalInput device_input;
  const std::vector<std::uint8_t> all_requested(static_cast<std::size_t>(batch_size), 1u);
  std::vector<std::uint8_t> requested = all_requested;
  input.perturb(1);
  CUDA_CHECK(device_input.upload(input, requested, stream));
  CHECK(host_cache.refresh_numerical_async(host_view(input, requested), error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(device_cache.refresh_numerical_async(device_input.view(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(reference_cache.refresh_numerical_async(host_view(input, all_requested), error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(stable_cache_identities(host_identity, host_cache.identity()));
  CHECK(stable_cache_identities(device_identity, device_cache.identity()));
  CHECK(stable_cache_identities(reference_identity, reference_cache.identity()));

  CacheSnapshot host_changed;
  CacheSnapshot device_changed;
  CacheSnapshot reference_changed;
  CHECK(download_snapshot(host_cache.identity(), stream, host_changed) == 0);
  CHECK(download_snapshot(device_cache.identity(), stream, device_changed) == 0);
  CHECK(download_snapshot(reference_cache.identity(), stream, reference_changed) == 0);
  CHECK(same_snapshot_values(host_changed, device_changed));
  CHECK(same_snapshot_values(host_changed, reference_changed));
  CHECK(host_changed.epoch == 2u);
  for (std::int64_t system = 0; system < batch_size; ++system) {
    const std::size_t index = static_cast<std::size_t>(system);
    CHECK(host_changed.committed_generations[index] == 2u);
    CHECK(host_changed.factor_generations[index] == 2u);
    CHECK(host_changed.factor_statuses[index] == 0u);
    CHECK(host_changed.eligible[index] == 1u);
    CHECK(verify_every_peer_leaf_changed(host_initial, host_changed, host_identity, fixture, system,
                                         scenario.d4, scenario.points, scenario.periodic) == 0);
    CHECK(verify_every_multipole_component_changed(host_initial, host_changed, host_identity,
                                                   fixture, system) == 0);
  }

  /* The final member is deliberately unrequested after every source changes
   * again.  Its complete committed byte ranges and factor generation must be
   * preserved while every requested peer advances to the new global epoch. */
  input.perturb(2);
  const std::int64_t unrequested_system = batch_size - 1;
  requested[static_cast<std::size_t>(unrequested_system)] = 0u;
  CUDA_CHECK(device_input.upload(input, requested, stream));
  CHECK(host_cache.refresh_numerical_async(host_view(input, requested), error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(device_cache.refresh_numerical_async(device_input.view(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(reference_cache.refresh_numerical_async(host_view(input, all_requested), error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(stable_cache_identities(host_identity, host_cache.identity()));
  CHECK(stable_cache_identities(device_identity, device_cache.identity()));
  CHECK(stable_cache_identities(reference_identity, reference_cache.identity()));

  CacheSnapshot host_rollback;
  CacheSnapshot device_rollback;
  CacheSnapshot reference_phase2;
  CHECK(download_snapshot(host_cache.identity(), stream, host_rollback) == 0);
  CHECK(download_snapshot(device_cache.identity(), stream, device_rollback) == 0);
  CHECK(download_snapshot(reference_cache.identity(), stream, reference_phase2) == 0);
  CHECK(same_snapshot_values(host_rollback, device_rollback));
  CHECK(host_rollback.epoch == 3u);
  CHECK(reference_phase2.epoch == 3u);
  for (std::int64_t system = 0; system < batch_size; ++system) {
    const std::size_t index = static_cast<std::size_t>(system);
    CHECK(reference_phase2.committed_generations[index] == 3u);
    CHECK(reference_phase2.factor_generations[index] == 3u);
    CHECK(reference_phase2.factor_statuses[index] == 0u);
    CHECK(reference_phase2.eligible[index] == 1u);
    if (system == unrequested_system) {
      CHECK(host_rollback.committed_generations[index] ==
            host_changed.committed_generations[index]);
      CHECK(host_rollback.factor_generations[index] == host_changed.factor_generations[index]);
      CHECK(host_rollback.eligible[index] == 0u);
      CHECK(verify_peer_unchanged(host_changed, host_rollback, host_identity, fixture, system,
                                  scenario.d4, scenario.points, scenario.periodic) == 0);
    } else {
      CHECK(host_rollback.committed_generations[index] == 3u);
      CHECK(host_rollback.factor_generations[index] == 3u);
      CHECK(host_rollback.eligible[index] == 1u);
      CHECK(verify_every_peer_leaf_changed(host_changed, host_rollback, host_identity, fixture,
                                           system, scenario.d4, scenario.points,
                                           scenario.periodic) == 0);
      CHECK(verify_every_multipole_component_changed(reference_changed, reference_phase2,
                                                     reference_identity, fixture, system) == 0);
      CHECK(verify_peer_matches_reference(host_rollback, reference_phase2, host_identity, fixture,
                                          system, scenario.d4, scenario.points,
                                          scenario.periodic) == 0);
    }
    CHECK(host_rollback.factor_statuses[index] == 0u);
  }

  /* A nonfinite geometry is a peer-local preprocessing failure.  Exercise it
   * in the base matrix so #132 proves rollback for both an explicit omission
   * and a requested member that fails before overlap publication. */
  if (!scenario.d4 && !scenario.points && !scenario.periodic) {
    input.perturb(3);
    PublicHostBatch finite_phase3 = input;
    finite_phase3.bind();
    requested = all_requested;
    const std::int64_t failed_system = 0;
    const std::int64_t failed_atom = input.atom_offsets.front();
    input.positions[static_cast<std::size_t>(failed_atom * 3)] =
        std::numeric_limits<double>::quiet_NaN();
    input.bind();
    CUDA_CHECK(device_input.upload(input, requested, stream));
    CHECK(reference_cache.refresh_numerical_async(host_view(finite_phase3, all_requested), error) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(host_cache.refresh_numerical_async(host_view(input, requested), error) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(device_cache.refresh_numerical_async(device_input.view(), error) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(stable_cache_identities(host_identity, host_cache.identity()));
    CHECK(stable_cache_identities(device_identity, device_cache.identity()));
    CHECK(stable_cache_identities(reference_identity, reference_cache.identity()));
    CacheSnapshot host_failed;
    CacheSnapshot device_failed;
    CacheSnapshot reference_phase3;
    CHECK(download_snapshot(host_cache.identity(), stream, host_failed) == 0);
    CHECK(download_snapshot(device_cache.identity(), stream, device_failed) == 0);
    CHECK(download_snapshot(reference_cache.identity(), stream, reference_phase3) == 0);
    CHECK(same_snapshot_values(host_failed, device_failed));
    CHECK(host_failed.epoch == 4u);
    CHECK(reference_phase3.epoch == 4u);
    for (std::int64_t system = 0; system < batch_size; ++system) {
      const std::size_t index = static_cast<std::size_t>(system);
      CHECK(reference_phase3.committed_generations[index] == 4u);
      CHECK(reference_phase3.factor_generations[index] == 4u);
      CHECK(reference_phase3.factor_statuses[index] == 0u);
      CHECK(reference_phase3.eligible[index] == 1u);
      if (system == failed_system) {
        CHECK(host_failed.committed_generations[index] ==
              host_rollback.committed_generations[index]);
        CHECK(host_failed.factor_generations[index] == host_rollback.factor_generations[index]);
        CHECK(host_failed.eligible[index] == 0u);
        CHECK(verify_peer_unchanged(host_rollback, host_failed, host_identity, fixture, system,
                                    false, false, false) == 0);
      } else {
        CHECK(host_failed.committed_generations[index] == 4u);
        CHECK(host_failed.factor_generations[index] == 4u);
        CHECK(host_failed.eligible[index] == 1u);
        CHECK(verify_every_peer_leaf_changed(host_rollback, host_failed, host_identity, fixture,
                                             system, false, false, false) == 0);
        CHECK(verify_peer_matches_reference(host_failed, reference_phase3, host_identity, fixture,
                                            system, false, false, false) == 0);
      }
      CHECK(host_failed.factor_statuses[index] == 0u);
    }
  }
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status == cudaErrorNoDevice || count_status == cudaErrorInsufficientDriver) {
    std::fprintf(stderr, "SKIP: CUDA runtime cache diagnostics require a CUDA device\n");
    (void)cudaGetLastError();
    return 0;
  }
  CUDA_CHECK(count_status);
  if (device_count == 0) {
    std::fprintf(stderr, "SKIP: CUDA runtime cache diagnostics require a CUDA device\n");
    return 0;
  }
  std::int32_t device_id = -1;
  CUDA_CHECK(cudaGetDevice(&device_id));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  constexpr std::array<Scenario, 4> scenarios{{
      {"base", SmallSystemKind::kHe, false, false, false},
      {"d4", SmallSystemKind::kH2, true, false, false},
      {"qmmm", SmallSystemKind::kH2, true, true, false},
      {"periodic", SmallSystemKind::kH2, true, false, true},
  }};
  int status = 0;
  for (const Scenario& scenario : scenarios) {
    for (const std::int64_t batch_size : kBatchSizes) {
      status = run_case(scenario, batch_size, stream, device_id);
      if (status != 0) break;
    }
    if (status != 0) break;
  }

  g_case = "finalize";
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return status;
}
