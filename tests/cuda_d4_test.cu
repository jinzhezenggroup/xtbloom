#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_d4.cuh"
#include "data/parameters/d4.hpp"
#include "model/gfn2/d4.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using gpuxtb::detail::cuda::Gfn2D4DeviceBatch;
using gpuxtb::detail::cuda::Gfn2D4DeviceCache;
using gpuxtb::detail::cuda::Gfn2D4DeviceElementData;
using gpuxtb::detail::cuda::Gfn2D4DeviceError;
using gpuxtb::detail::cuda::Gfn2D4DeviceParameters;
using gpuxtb::detail::cuda::Gfn2D4DeviceReferenceData;
using gpuxtb::detail::cuda::Gfn2D4DeviceWorkspace;

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  cudaError_t allocate(std::size_t count) {
    count_ = count;
    return cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
  }

  cudaError_t copy_from(const T* source, std::size_t count, cudaStream_t stream = nullptr) {
    if (count > count_) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(data_, source, count * sizeof(T), cudaMemcpyHostToDevice, stream);
  }

  cudaError_t copy_to(T* destination, std::size_t count, cudaStream_t stream = nullptr) const {
    if (count > count_) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
  }

  T* get() { return data_; }
  const T* get() const { return data_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

bool near(double actual, double expected, double absolute_tolerance,
          double relative_tolerance = 0.0) {
  const double scale = std::max(std::abs(actual), std::abs(expected));
  return std::abs(actual - expected) <= absolute_tolerance + relative_tolerance * scale;
}

struct HostFixture {
  static constexpr std::array<std::int64_t, 3> atom_offsets{0, 3, 5};
  static constexpr std::array<std::int32_t, 5> atomic_numbers{8, 1, 1, 6, 8};
  static constexpr std::array<double, 15> positions{
      0.0, 0.0, 0.0, 1.43, 1.11, 0.0, -1.43, 1.11, 0.0, 0.0, 0.0, 0.0, 2.20, 0.0, 0.0,
  };
  static constexpr std::array<double, 5> charges{-0.42, 0.21, 0.21, 0.18, -0.18};

  gpuxtb::detail::gfn2::D4Plan plan;
  std::vector<std::byte> workspace_storage;
  gpuxtb::detail::gfn2::D4Workspace workspace;
  std::vector<double> pair_data;
  std::vector<double> coordination;
  gpuxtb::detail::gfn2::D4GeometryCache cache;
  std::array<double, 2> energies{};
  std::array<double, 5> potentials{};

  bool initialize() {
    std::string error;
    if (gpuxtb::detail::gfn2::make_d4_plan(2, 5, atom_offsets.data(), atomic_numbers.data(), plan,
                                           error) != GPUXTB_STATUS_SUCCESS) {
      return false;
    }
    workspace_storage.resize(plan.workspace_size_bytes() +
                             gpuxtb::detail::gfn2::kD4WorkspaceAlignment - 1u);
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(workspace_storage.data());
    const std::uintptr_t aligned = (address + gpuxtb::detail::gfn2::kD4WorkspaceAlignment - 1u) &
                                   ~(gpuxtb::detail::gfn2::kD4WorkspaceAlignment - 1u);
    if (gpuxtb::detail::gfn2::bind_d4_workspace(plan, reinterpret_cast<void*>(aligned),
                                                plan.workspace_size_bytes(), workspace,
                                                error) != GPUXTB_STATUS_SUCCESS) {
      return false;
    }
    pair_data.resize(static_cast<std::size_t>(plan.total_pairs()) *
                     gpuxtb::detail::gfn2::kD4PairDataElements);
    coordination.resize(static_cast<std::size_t>(plan.total_atoms()));
    if (gpuxtb::detail::gfn2::update_d4_geometry_cache_cpu(
            plan, positions.data(), 7u, pair_data.data(), pair_data.size(), coordination.data(),
            coordination.size(), workspace, cache, error) != GPUXTB_STATUS_SUCCESS) {
      return false;
    }
    return gpuxtb::detail::gfn2::evaluate_d4_two_body_cpu(
               plan, cache, charges.data(), energies.data(), potentials.data(), workspace, error) ==
           GPUXTB_STATUS_SUCCESS;
  }
};

struct DeviceFixture {
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> pair_offsets;
  DeviceBuffer<std::int32_t> atomic_numbers;
  DeviceBuffer<Gfn2D4DeviceElementData> elements;
  DeviceBuffer<Gfn2D4DeviceReferenceData> references;
  DeviceBuffer<double> reference_c6;
  DeviceBuffer<double> pair_data;
  DeviceBuffer<double> coordination;
  DeviceBuffer<double> charges;
  DeviceBuffer<double> energies;
  DeviceBuffer<double> potentials;
  DeviceBuffer<double> weights;
  DeviceBuffer<double> weight_charge_derivatives;
  DeviceBuffer<double> atom_scratch;
  DeviceBuffer<double> batch_scratch;
  DeviceBuffer<std::uint32_t> error;
  Gfn2D4DeviceBatch batch;
  Gfn2D4DeviceParameters parameters;
  Gfn2D4DeviceCache cache;
  Gfn2D4DeviceWorkspace workspace;

  bool initialize(const HostFixture& host, cudaStream_t stream) {
    return initialize(host.plan, HostFixture::atom_offsets, HostFixture::atomic_numbers,
                      host.pair_data, host.coordination, HostFixture::charges, stream);
  }

  template <typename Offsets, typename AtomicNumbers, typename Charges>
  bool initialize(const gpuxtb::detail::gfn2::D4Plan& host_plan, const Offsets& host_atom_offsets,
                  const AtomicNumbers& host_atomic_numbers,
                  const std::vector<double>& host_pair_data,
                  const std::vector<double>& host_coordination, const Charges& host_charges,
                  cudaStream_t stream) {
    std::vector<Gfn2D4DeviceElementData> host_elements;
    host_elements.reserve(gpuxtb::parameters::d4::kElements.size());
    for (const auto& element : gpuxtb::parameters::d4::kElements) {
      host_elements.push_back({element.reference_offset, element.reference_count,
                               element.covalent_radius, element.electronegativity,
                               element.effective_charge, element.hardness, element.r4r2});
    }
    std::vector<Gfn2D4DeviceReferenceData> host_references;
    host_references.reserve(gpuxtb::parameters::d4::kReferences.size());
    for (const auto& reference : gpuxtb::parameters::d4::kReferences) {
      host_references.push_back(
          {reference.coordination_number, reference.charge, reference.gaussian_count});
    }

    if (host_atom_offsets.size() < 2u) {
      return false;
    }
    const std::size_t batch_count = host_atom_offsets.size() - 1u;
    const std::size_t atom_count = host_atomic_numbers.size();
    const std::size_t weight_count = atom_count * gpuxtb::detail::cuda::kGfn2D4MaximumReferences;
    if (host_charges.size() != atom_count || host_coordination.size() != atom_count ||
        host_plan.batch_size() != static_cast<std::int64_t>(batch_count) ||
        host_plan.total_atoms() != static_cast<std::int64_t>(atom_count) ||
        host_pair_data.size() != static_cast<std::size_t>(host_plan.total_pairs()) *
                                     gpuxtb::detail::gfn2::kD4PairDataElements) {
      return false;
    }
    if (atom_offsets.allocate(host_atom_offsets.size()) != cudaSuccess ||
        pair_offsets.allocate(host_plan.pair_offsets().size()) != cudaSuccess ||
        atomic_numbers.allocate(atom_count) != cudaSuccess ||
        elements.allocate(host_elements.size()) != cudaSuccess ||
        references.allocate(host_references.size()) != cudaSuccess ||
        reference_c6.allocate(gpuxtb::parameters::d4::kReferenceC6.size()) != cudaSuccess ||
        pair_data.allocate(std::max<std::size_t>(host_pair_data.size(), 1u)) != cudaSuccess ||
        coordination.allocate(atom_count) != cudaSuccess ||
        charges.allocate(atom_count) != cudaSuccess ||
        energies.allocate(batch_count) != cudaSuccess ||
        potentials.allocate(atom_count) != cudaSuccess ||
        weights.allocate(weight_count) != cudaSuccess ||
        weight_charge_derivatives.allocate(weight_count) != cudaSuccess ||
        atom_scratch.allocate(atom_count) != cudaSuccess ||
        batch_scratch.allocate(batch_count) != cudaSuccess || error.allocate(1) != cudaSuccess) {
      return false;
    }
    if (atom_offsets.copy_from(host_atom_offsets.data(), host_atom_offsets.size(), stream) !=
            cudaSuccess ||
        pair_offsets.copy_from(host_plan.pair_offsets().data(), host_plan.pair_offsets().size(),
                               stream) != cudaSuccess ||
        atomic_numbers.copy_from(host_atomic_numbers.data(), atom_count, stream) != cudaSuccess ||
        elements.copy_from(host_elements.data(), host_elements.size(), stream) != cudaSuccess ||
        references.copy_from(host_references.data(), host_references.size(), stream) !=
            cudaSuccess ||
        reference_c6.copy_from(gpuxtb::parameters::d4::kReferenceC6.data(),
                               gpuxtb::parameters::d4::kReferenceC6.size(),
                               stream) != cudaSuccess ||
        (!host_pair_data.empty() &&
         pair_data.copy_from(host_pair_data.data(), host_pair_data.size(), stream) !=
             cudaSuccess) ||
        coordination.copy_from(host_coordination.data(), host_coordination.size(), stream) !=
            cudaSuccess ||
        charges.copy_from(host_charges.data(), host_charges.size(), stream) != cudaSuccess) {
      return false;
    }

    const std::uint64_t token =
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(host_plan.identity()));
    batch = {static_cast<std::int64_t>(batch_count),
             static_cast<std::int64_t>(atom_count),
             host_plan.total_pairs(),
             token,
             gpuxtb::detail::cuda::gfn2_d4_atomic_number_hash(
                 host_atomic_numbers.data(), static_cast<std::int64_t>(atom_count)),
             atom_offsets.get(),
             pair_offsets.get(),
             atomic_numbers.get()};
    parameters = {
        elements.get(),     static_cast<std::int64_t>(host_elements.size()),
        references.get(),   static_cast<std::int64_t>(host_references.size()),
        reference_c6.get(), static_cast<std::int64_t>(gpuxtb::parameters::d4::kReferenceC6.size())};
    cache = {pair_data.get(),
             static_cast<std::int64_t>(host_pair_data.size()),
             coordination.get(),
             static_cast<std::int64_t>(host_coordination.size()),
             7u,
             token};
    workspace = {weights.get(),
                 weight_charge_derivatives.get(),
                 static_cast<std::int64_t>(weight_count),
                 atom_scratch.get(),
                 static_cast<std::int64_t>(atom_count),
                 batch_scratch.get(),
                 static_cast<std::int64_t>(batch_count)};
    return cudaStreamSynchronize(stream) == cudaSuccess;
  }
};

int test_cpu_parity_and_ragged_batch() {
  HostFixture host;
  CHECK(host.initialize());
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CHECK(device.initialize(host, stream));

  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_d4_device_error_cuda(device.error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
      device.batch, device.parameters, device.cache, device.charges.get(), device.energies.get(),
      device.potentials.get(), device.workspace, device.error.get(), stream));
  std::array<double, 2> actual_energies{};
  std::array<double, 5> actual_potentials{};
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(device.energies.copy_to(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(device.potentials.copy_to(actual_potentials.data(), actual_potentials.size(), stream));
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
  for (std::size_t system = 0; system < actual_energies.size(); ++system) {
    CHECK(near(actual_energies[system], host.energies[system], 2.0e-12, 2.0e-13));
  }
  for (std::size_t atom = 0; atom < actual_potentials.size(); ++atom) {
    CHECK(near(actual_potentials[atom], host.potentials[atom], 2.0e-12, 2.0e-13));
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_all_supported_elements_cpu_parity() {
  constexpr std::size_t batch_count = gpuxtb::parameters::d4::kElementCount;
  constexpr std::size_t atom_count = batch_count * 2u;
  std::vector<std::int64_t> atom_offsets(batch_count + 1u);
  std::vector<std::int32_t> atomic_numbers(atom_count);
  std::vector<double> positions(atom_count * 3u, 0.0);
  std::vector<double> charges(atom_count);
  for (std::size_t system = 0; system < batch_count; ++system) {
    atom_offsets[system] = static_cast<std::int64_t>(system * 2u);
    const std::int32_t first = static_cast<std::int32_t>(system + 1u);
    const std::int32_t second = static_cast<std::int32_t>((system * 37u) % batch_count + 1u);
    atomic_numbers[system * 2u] = first;
    atomic_numbers[system * 2u + 1u] = second;
    positions[(system * 2u + 1u) * 3u] = 2.2 + 0.01 * static_cast<double>(system % 17u);
    charges[system * 2u] = -0.12 + 0.002 * static_cast<double>(system % 11u);
    charges[system * 2u + 1u] = -charges[system * 2u];
  }
  atom_offsets[batch_count] = static_cast<std::int64_t>(atom_count);

  gpuxtb::detail::gfn2::D4Plan plan;
  std::string error;
  CHECK(gpuxtb::detail::gfn2::make_d4_plan(
            static_cast<std::int64_t>(batch_count), static_cast<std::int64_t>(atom_count),
            atom_offsets.data(), atomic_numbers.data(), plan, error) == GPUXTB_STATUS_SUCCESS);
  std::vector<std::byte> workspace_storage(plan.workspace_size_bytes() +
                                           gpuxtb::detail::gfn2::kD4WorkspaceAlignment - 1u);
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(workspace_storage.data());
  const std::uintptr_t aligned = (address + gpuxtb::detail::gfn2::kD4WorkspaceAlignment - 1u) &
                                 ~(gpuxtb::detail::gfn2::kD4WorkspaceAlignment - 1u);
  gpuxtb::detail::gfn2::D4Workspace host_workspace;
  CHECK(gpuxtb::detail::gfn2::bind_d4_workspace(plan, reinterpret_cast<void*>(aligned),
                                                plan.workspace_size_bytes(), host_workspace,
                                                error) == GPUXTB_STATUS_SUCCESS);
  std::vector<double> pair_data(static_cast<std::size_t>(plan.total_pairs()) *
                                gpuxtb::detail::gfn2::kD4PairDataElements);
  std::vector<double> coordination(atom_count);
  gpuxtb::detail::gfn2::D4GeometryCache host_cache;
  CHECK(gpuxtb::detail::gfn2::update_d4_geometry_cache_cpu(
            plan, positions.data(), 9u, pair_data.data(), pair_data.size(), coordination.data(),
            coordination.size(), host_workspace, host_cache, error) == GPUXTB_STATUS_SUCCESS);
  std::vector<double> expected_energies(batch_count);
  std::vector<double> expected_potentials(atom_count);
  CHECK(gpuxtb::detail::gfn2::evaluate_d4_two_body_cpu(
            plan, host_cache, charges.data(), expected_energies.data(), expected_potentials.data(),
            host_workspace, error) == GPUXTB_STATUS_SUCCESS);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CHECK(device.initialize(plan, atom_offsets, atomic_numbers, pair_data, coordination, charges,
                          stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_d4_device_error_cuda(device.error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
      device.batch, device.parameters, device.cache, device.charges.get(), device.energies.get(),
      device.potentials.get(), device.workspace, device.error.get(), stream));
  std::vector<double> actual_energies(batch_count);
  std::vector<double> actual_potentials(atom_count);
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(device.energies.copy_to(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(device.potentials.copy_to(actual_potentials.data(), actual_potentials.size(), stream));
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
  for (std::size_t system = 0; system < batch_count; ++system) {
    CHECK(near(actual_energies[system], expected_energies[system], 2.0e-12, 2.0e-13));
  }
  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    CHECK(near(actual_potentials[atom], expected_potentials[atom], 2.0e-12, 2.0e-13));
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_empty_and_singleton_systems() {
  constexpr std::array<std::int64_t, 5> atom_offsets{0, 0, 1, 1, 2};
  constexpr std::array<std::int32_t, 2> atomic_numbers{1, 8};
  constexpr std::array<double, 2> charges{0.1, -0.1};
  const std::vector<double> pair_data;
  const std::vector<double> coordination(atomic_numbers.size(), 0.0);
  gpuxtb::detail::gfn2::D4Plan plan;
  std::string error;
  CHECK(gpuxtb::detail::gfn2::make_d4_plan(4, static_cast<std::int64_t>(atomic_numbers.size()),
                                           atom_offsets.data(), atomic_numbers.data(), plan,
                                           error) == GPUXTB_STATUS_SUCCESS);
  CHECK(plan.total_pairs() == 0);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CHECK(device.initialize(plan, atom_offsets, atomic_numbers, pair_data, coordination, charges,
                          stream));
  device.cache.pair_data = nullptr;
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_d4_device_error_cuda(device.error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
      device.batch, device.parameters, device.cache, device.charges.get(), device.energies.get(),
      device.potentials.get(), device.workspace, device.error.get(), stream));
  std::array<double, 4> energies{};
  std::array<double, 2> potentials{};
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(device.energies.copy_to(energies.data(), energies.size(), stream));
  CUDA_CHECK(device.potentials.copy_to(potentials.data(), potentials.size(), stream));
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
  CHECK((energies == std::array<double, 4>{}));
  CHECK((potentials == std::array<double, 2>{}));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_atomic_number_ordering_and_range_validation() {
  HostFixture host;
  CHECK(host.initialize());
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CHECK(device.initialize(host, stream));
  constexpr std::array<double, 2> energy_sentinel{123.0, -456.0};
  constexpr std::array<double, 5> potential_sentinel{1.0, 2.0, 3.0, 4.0, 5.0};

  std::array<std::int32_t, 5> reordered = HostFixture::atomic_numbers;
  std::swap(reordered[0], reordered[1]);
  CUDA_CHECK(device.atomic_numbers.copy_from(reordered.data(), reordered.size(), stream));
  CUDA_CHECK(device.energies.copy_from(energy_sentinel.data(), energy_sentinel.size(), stream));
  CUDA_CHECK(
      device.potentials.copy_from(potential_sentinel.data(), potential_sentinel.size(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_d4_device_error_cuda(device.error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
      device.batch, device.parameters, device.cache, device.charges.get(), device.energies.get(),
      device.potentials.get(), device.workspace, device.error.get(), stream));
  std::array<double, 2> actual_energies{};
  std::array<double, 5> actual_potentials{};
  std::uint32_t semantic_error = 0u;
  CUDA_CHECK(device.energies.copy_to(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(device.potentials.copy_to(actual_potentials.data(), actual_potentials.size(), stream));
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kInvalidAtomicNumber));
  CHECK(actual_energies == energy_sentinel);
  CHECK(actual_potentials == potential_sentinel);

  reordered = HostFixture::atomic_numbers;
  reordered[0] = 0;
  device.batch.atomic_number_hash =
      gpuxtb::detail::cuda::gfn2_d4_atomic_number_hash(reordered.data(), reordered.size());
  CUDA_CHECK(device.atomic_numbers.copy_from(reordered.data(), reordered.size(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_d4_device_error_cuda(device.error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
      device.batch, device.parameters, device.cache, device.charges.get(), device.energies.get(),
      device.potentials.get(), device.workspace, device.error.get(), stream));
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kInvalidAtomicNumber));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_semantic_error_atomicity_and_sticky_status() {
  HostFixture host;
  CHECK(host.initialize());
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CHECK(device.initialize(host, stream));
  constexpr std::array<double, 2> energy_sentinel{123.0, -456.0};
  constexpr std::array<double, 5> potential_sentinel{1.0, 2.0, 3.0, 4.0, 5.0};
  std::array<double, 5> invalid_charges = HostFixture::charges;
  invalid_charges[3] = std::numeric_limits<double>::quiet_NaN();

  CUDA_CHECK(device.charges.copy_from(invalid_charges.data(), invalid_charges.size(), stream));
  CUDA_CHECK(device.energies.copy_from(energy_sentinel.data(), energy_sentinel.size(), stream));
  CUDA_CHECK(
      device.potentials.copy_from(potential_sentinel.data(), potential_sentinel.size(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_d4_device_error_cuda(device.error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
      device.batch, device.parameters, device.cache, device.charges.get(), device.energies.get(),
      device.potentials.get(), device.workspace, device.error.get(), stream));
  std::array<double, 2> actual_energies{};
  std::array<double, 5> actual_potentials{};
  std::uint32_t semantic_error = 0u;
  CUDA_CHECK(device.energies.copy_to(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(device.potentials.copy_to(actual_potentials.data(), actual_potentials.size(), stream));
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kNonfiniteCharge));
  CHECK(actual_energies == energy_sentinel);
  CHECK(actual_potentials == potential_sentinel);

  /* A failed dependent sequence remains inert until the caller resets it. */
  CUDA_CHECK(
      device.charges.copy_from(HostFixture::charges.data(), HostFixture::charges.size(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
      device.batch, device.parameters, device.cache, device.charges.get(), device.energies.get(),
      device.potentials.get(), device.workspace, device.error.get(), stream));
  CUDA_CHECK(device.energies.copy_to(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(device.potentials.copy_to(actual_potentials.data(), actual_potentials.size(), stream));
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kNonfiniteCharge));
  CHECK(actual_energies == energy_sentinel);
  CHECK(actual_potentials == potential_sentinel);

  /* Invalid ragged topology is detected before any numerical kernel publishes. */
  constexpr std::array<std::int64_t, 3> invalid_pair_offsets{0, 2, 4};
  CUDA_CHECK(device.pair_offsets.copy_from(invalid_pair_offsets.data(), invalid_pair_offsets.size(),
                                           stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_d4_device_error_cuda(device.error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
      device.batch, device.parameters, device.cache, device.charges.get(), device.energies.get(),
      device.potentials.get(), device.workspace, device.error.get(), stream));
  CUDA_CHECK(device.energies.copy_to(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(device.potentials.copy_to(actual_potentials.data(), actual_potentials.size(), stream));
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kInvalidOffsets));
  CHECK(actual_energies == energy_sentinel);
  CHECK(actual_potentials == potential_sentinel);

  Gfn2D4DeviceWorkspace undersized_workspace = device.workspace;
  --undersized_workspace.weight_elements;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
            device.batch, device.parameters, device.cache, device.charges.get(),
            device.energies.get(), device.potentials.get(), undersized_workspace,
            device.error.get(), stream) == cudaErrorInvalidValue);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_range_alias_and_overflow_validation() {
  HostFixture host;
  CHECK(host.initialize());
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CHECK(device.initialize(host, stream));
  constexpr std::array<double, 2> energy_sentinel{123.0, -456.0};
  constexpr std::array<double, 5> potential_sentinel{1.0, 2.0, 3.0, 4.0, 5.0};
  CUDA_CHECK(device.energies.copy_from(energy_sentinel.data(), energy_sentinel.size(), stream));
  CUDA_CHECK(
      device.potentials.copy_from(potential_sentinel.data(), potential_sentinel.size(), stream));

  Gfn2D4DeviceWorkspace aliased_workspace = device.workspace;
  aliased_workspace.weight_charge_derivatives = aliased_workspace.weights;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
            device.batch, device.parameters, device.cache, device.charges.get(),
            device.energies.get(), device.potentials.get(), aliased_workspace, device.error.get(),
            stream) == cudaErrorInvalidValue);

  aliased_workspace = device.workspace;
  aliased_workspace.weights = const_cast<double*>(device.cache.coordination_numbers);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
            device.batch, device.parameters, device.cache, device.charges.get(),
            device.energies.get(), device.potentials.get(), aliased_workspace, device.error.get(),
            stream) == cudaErrorInvalidValue);

  aliased_workspace = device.workspace;
  aliased_workspace.atom_scratch = device.charges.get();
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
            device.batch, device.parameters, device.cache, device.charges.get(),
            device.energies.get(), device.potentials.get(), aliased_workspace, device.error.get(),
            stream) == cudaErrorInvalidValue);

  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
            device.batch, device.parameters, device.cache, device.charges.get(),
            device.workspace.batch_scratch, device.potentials.get(), device.workspace,
            device.error.get(), stream) == cudaErrorInvalidValue);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
            device.batch, device.parameters, device.cache, device.charges.get(),
            device.energies.get(), device.workspace.atom_scratch, device.workspace,
            device.error.get(), stream) == cudaErrorInvalidValue);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
            device.batch, device.parameters, device.cache, device.charges.get(),
            device.potentials.get(), device.potentials.get(), device.workspace, device.error.get(),
            stream) == cudaErrorInvalidValue);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
            device.batch, device.parameters, device.cache, device.charges.get(),
            device.charges.get(), device.potentials.get(), device.workspace, device.error.get(),
            stream) == cudaErrorInvalidValue);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
            device.batch, device.parameters, device.cache, device.charges.get(),
            device.energies.get(), device.charges.get(), device.workspace, device.error.get(),
            stream) == cudaErrorInvalidValue);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
            device.batch, device.parameters, device.cache, device.charges.get(),
            const_cast<double*>(device.cache.pair_data), device.potentials.get(), device.workspace,
            device.error.get(), stream) == cudaErrorInvalidValue);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
            device.batch, device.parameters, device.cache, device.charges.get(),
            device.energies.get(),
            reinterpret_cast<double*>(const_cast<std::int64_t*>(device.batch.atom_offsets)),
            device.workspace, device.error.get(), stream) == cudaErrorInvalidValue);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
            device.batch, device.parameters, device.cache, device.charges.get(),
            device.energies.get(), device.potentials.get(), device.workspace,
            reinterpret_cast<std::uint32_t*>(device.charges.get()),
            stream) == cudaErrorInvalidValue);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
            device.batch, device.parameters, device.cache, device.charges.get(),
            reinterpret_cast<double*>(device.error.get()), device.potentials.get(),
            device.workspace, device.error.get(), stream) == cudaErrorInvalidValue);

  Gfn2D4DeviceCache missing_pair_cache = device.cache;
  missing_pair_cache.pair_data = nullptr;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
            device.batch, device.parameters, missing_pair_cache, device.charges.get(),
            device.energies.get(), device.potentials.get(), device.workspace, device.error.get(),
            stream) == cudaErrorInvalidValue);

  Gfn2D4DeviceParameters overflowing_parameters = device.parameters;
  overflowing_parameters.reference_count = std::int64_t{1} << 31;
  overflowing_parameters.reference_c6_elements = std::int64_t{1} << 62;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
            device.batch, overflowing_parameters, device.cache, device.charges.get(),
            device.energies.get(), device.potentials.get(), device.workspace, device.error.get(),
            stream) == cudaErrorInvalidValue);

  Gfn2D4DeviceWorkspace overflowing_workspace = device.workspace;
  overflowing_workspace.weight_elements = std::numeric_limits<std::int64_t>::max();
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
            device.batch, device.parameters, device.cache, device.charges.get(),
            device.energies.get(), device.potentials.get(), overflowing_workspace,
            device.error.get(), stream) == cudaErrorInvalidValue);

  Gfn2D4DeviceParameters wrapping_parameters = device.parameters;
  constexpr std::uintptr_t maximum_address = std::numeric_limits<std::uintptr_t>::max();
  const std::uintptr_t aligned_maximum_element_address =
      maximum_address - maximum_address % alignof(Gfn2D4DeviceElementData);
  wrapping_parameters.elements =
      reinterpret_cast<const Gfn2D4DeviceElementData*>(aligned_maximum_element_address);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
            device.batch, wrapping_parameters, device.cache, device.charges.get(),
            device.energies.get(), device.potentials.get(), device.workspace, device.error.get(),
            stream) == cudaErrorInvalidValue);

  const std::uintptr_t aligned_maximum_error_address =
      maximum_address - maximum_address % alignof(std::uint32_t);
  CHECK(gpuxtb::detail::cuda::reset_gfn2_d4_device_error_cuda(
            reinterpret_cast<std::uint32_t*>(aligned_maximum_error_address), stream) ==
        cudaErrorInvalidValue);

  std::array<double, 2> actual_energies{};
  std::array<double, 5> actual_potentials{};
  CUDA_CHECK(device.energies.copy_to(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(device.potentials.copy_to(actual_potentials.data(), actual_potentials.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(actual_energies == energy_sentinel);
  CHECK(actual_potentials == potential_sentinel);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_graph_capture_and_replay() {
  HostFixture host;
  CHECK(host.initialize());
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CHECK(device.initialize(host, stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_d4_device_error_cuda(device.error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
      device.batch, device.parameters, device.cache, device.charges.get(), device.energies.get(),
      device.potentials.get(), device.workspace, device.error.get(), stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0u));

  constexpr std::array<double, 5> replay_charges{-0.37, 0.16, 0.21, 0.11, -0.11};
  std::array<double, 2> expected_energies{};
  std::array<double, 5> expected_potentials{};
  std::string error;
  CHECK(gpuxtb::detail::gfn2::evaluate_d4_two_body_cpu(
            host.plan, host.cache, replay_charges.data(), expected_energies.data(),
            expected_potentials.data(), host.workspace, error) == GPUXTB_STATUS_SUCCESS);
  CUDA_CHECK(device.charges.copy_from(replay_charges.data(), replay_charges.size(), stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));

  std::array<double, 2> actual_energies{};
  std::array<double, 5> actual_potentials{};
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(device.energies.copy_to(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(device.potentials.copy_to(actual_potentials.data(), actual_potentials.size(), stream));
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
  for (std::size_t system = 0; system < actual_energies.size(); ++system) {
    CHECK(near(actual_energies[system], expected_energies[system], 2.0e-12, 2.0e-13));
  }
  for (std::size_t atom = 0; atom < actual_potentials.size(); ++atom) {
    CHECK(near(actual_potentials[atom], expected_potentials[atom], 2.0e-12, 2.0e-13));
  }

  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_cpu_parity_and_ragged_batch(); status != 0) {
    std::cerr << "CUDA D4 CPU-parity test failed at line " << status << '\n';
    return status;
  }
  if (const int status = test_all_supported_elements_cpu_parity(); status != 0) {
    std::cerr << "CUDA D4 all-element parity test failed at line " << status << '\n';
    return status;
  }
  if (const int status = test_empty_and_singleton_systems(); status != 0) {
    std::cerr << "CUDA D4 empty/singleton batch test failed at line " << status << '\n';
    return status;
  }
  if (const int status = test_atomic_number_ordering_and_range_validation(); status != 0) {
    std::cerr << "CUDA D4 atomic-number validation test failed at line " << status << '\n';
    return status;
  }
  if (const int status = test_semantic_error_atomicity_and_sticky_status(); status != 0) {
    std::cerr << "CUDA D4 error-atomicity test failed at line " << status << '\n';
    return status;
  }
  if (const int status = test_range_alias_and_overflow_validation(); status != 0) {
    std::cerr << "CUDA D4 range-validation test failed at line " << status << '\n';
    return status;
  }
  if (const int status = test_graph_capture_and_replay(); status != 0) {
    std::cerr << "CUDA D4 graph-capture test failed at line " << status << '\n';
    return status;
  }
  return 0;
}
