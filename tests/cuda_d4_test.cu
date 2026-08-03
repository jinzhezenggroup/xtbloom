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

namespace gpuxtb::detail::cuda {
cudaError_t test_gfn2_d4_atm_reduction_cuda(const Gfn2D4DeviceBatch& batch,
                                            const double* finite_values, double* energies,
                                            const Gfn2D4DeviceWorkspace& workspace,
                                            std::uint32_t* device_error,
                                            cudaStream_t stream) noexcept;
cudaError_t test_gfn2_d4_atm_addition_cuda(const Gfn2D4DeviceBatch& batch,
                                           const double* finite_deltas, double* gradients,
                                           const Gfn2D4DeviceWorkspace& workspace,
                                           std::uint32_t* device_error,
                                           cudaStream_t stream) noexcept;
}  // namespace gpuxtb::detail::cuda

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
  DeviceBuffer<double> positions;
  DeviceBuffer<double> pair_data;
  DeviceBuffer<double> coordination;
  DeviceBuffer<double> charges;
  DeviceBuffer<double> energies;
  DeviceBuffer<double> potentials;
  DeviceBuffer<double> weights;
  DeviceBuffer<double> weight_cn_derivatives;
  DeviceBuffer<double> weight_charge_derivatives;
  DeviceBuffer<double> atom_scratch;
  DeviceBuffer<double> coordination_adjoints;
  DeviceBuffer<double> batch_scratch;
  DeviceBuffer<double> gradient_scratch;
  DeviceBuffer<double> gradients;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> error;
  DeviceBuffer<double> pair_scratch;
  DeviceBuffer<double> coordination_scratch;
  DeviceBuffer<std::uint64_t> geometry_generations;
  DeviceBuffer<std::uint32_t> geometry_sequence_active;
  Gfn2D4DeviceBatch batch;
  Gfn2D4DeviceParameters parameters;
  Gfn2D4DeviceCache cache;
  Gfn2D4DeviceWorkspace workspace;

  bool initialize(const HostFixture& host, cudaStream_t stream) {
    return initialize(host.plan, HostFixture::atom_offsets, HostFixture::atomic_numbers,
                      host.pair_data, host.coordination, HostFixture::charges, stream) &&
           positions.copy_from(HostFixture::positions.data(), HostFixture::positions.size(),
                               stream) == cudaSuccess;
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
        positions.allocate(atom_count * 3u) != cudaSuccess ||
        pair_data.allocate(std::max<std::size_t>(host_pair_data.size(), 1u)) != cudaSuccess ||
        coordination.allocate(atom_count) != cudaSuccess ||
        charges.allocate(atom_count) != cudaSuccess ||
        energies.allocate(batch_count) != cudaSuccess ||
        potentials.allocate(atom_count) != cudaSuccess ||
        weights.allocate(weight_count) != cudaSuccess ||
        weight_cn_derivatives.allocate(weight_count) != cudaSuccess ||
        weight_charge_derivatives.allocate(weight_count) != cudaSuccess ||
        atom_scratch.allocate(atom_count) != cudaSuccess ||
        coordination_adjoints.allocate(atom_count) != cudaSuccess ||
        batch_scratch.allocate(batch_count) != cudaSuccess ||
        gradient_scratch.allocate(atom_count * 3u) != cudaSuccess ||
        gradients.allocate(atom_count * 3u) != cudaSuccess ||
        system_errors.allocate(batch_count) != cudaSuccess || error.allocate(1) != cudaSuccess ||
        pair_scratch.allocate(std::max<std::size_t>(host_pair_data.size(), 1u)) != cudaSuccess ||
        coordination_scratch.allocate(atom_count) != cudaSuccess ||
        geometry_generations.allocate(batch_count) != cudaSuccess ||
        geometry_sequence_active.allocate(1) != cudaSuccess) {
      return false;
    }
    const std::vector<std::uint64_t> initial_generations(batch_count, 7u);
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
        charges.copy_from(host_charges.data(), host_charges.size(), stream) != cudaSuccess ||
        geometry_generations.copy_from(initial_generations.data(), initial_generations.size(),
                                       stream) != cudaSuccess) {
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
                 weight_cn_derivatives.get(),
                 weight_charge_derivatives.get(),
                 static_cast<std::int64_t>(weight_count),
                 atom_scratch.get(),
                 coordination_adjoints.get(),
                 static_cast<std::int64_t>(atom_count),
                 batch_scratch.get(),
                 static_cast<std::int64_t>(batch_count),
                 gradient_scratch.get(),
                 static_cast<std::int64_t>(atom_count * 3u),
                 system_errors.get(),
                 static_cast<std::int64_t>(batch_count),
                 pair_scratch.get(),
                 static_cast<std::int64_t>(host_pair_data.size()),
                 coordination_scratch.get(),
                 static_cast<std::int64_t>(atom_count),
                 geometry_generations.get(),
                 static_cast<std::int64_t>(batch_count),
                 geometry_sequence_active.get(),
                 1};
    return cudaStreamSynchronize(stream) == cudaSuccess;
  }

  cudaError_t reset(cudaStream_t stream) {
    return gpuxtb::detail::cuda::reset_gfn2_d4_device_errors_cuda(
        batch.batch_size, system_errors.get(), error.get(), stream);
  }
};

int run_geometry_refresh_batch_case(std::size_t batch_count) {
  std::vector<std::int64_t> atom_offsets(batch_count + 1u, 0);
  for (std::size_t system = 0; system < batch_count; ++system) {
    atom_offsets[system + 1u] = atom_offsets[system] + static_cast<std::int64_t>(2u + system % 4u);
  }
  const std::size_t atom_count = static_cast<std::size_t>(atom_offsets.back());
  constexpr std::array<std::int32_t, 5> element_pattern{8, 1, 6, 7, 16};
  constexpr std::array<std::array<double, 3>, 5> local_positions{{
      {{0.0, 0.0, 0.0}},
      {{1.43, 1.11, 0.0}},
      {{-1.26, 1.37, 0.42}},
      {{0.38, -0.71, 2.18}},
      {{-1.91, -0.53, 0.84}},
  }};
  std::vector<std::int32_t> atomic_numbers(atom_count);
  std::vector<double> positions(atom_count * 3u);
  std::vector<double> charges(atom_count, 0.0);
  for (std::size_t system = 0; system < batch_count; ++system) {
    const std::size_t begin = static_cast<std::size_t>(atom_offsets[system]);
    const std::size_t end = static_cast<std::size_t>(atom_offsets[system + 1u]);
    for (std::size_t atom = begin; atom < end; ++atom) {
      const std::size_t local = atom - begin;
      atomic_numbers[atom] = element_pattern[local];
      for (std::size_t axis = 0; axis < 3u; ++axis) {
        positions[atom * 3u + axis] =
            local_positions[local][axis] +
            2.0e-4 * static_cast<double>((system + 1u) * (local + axis + 1u));
      }
    }
  }

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
            plan, positions.data(), 7u, pair_data.data(), pair_data.size(), coordination.data(),
            coordination.size(), host_workspace, host_cache, error) == GPUXTB_STATUS_SUCCESS);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CHECK(device.initialize(plan, atom_offsets, atomic_numbers, pair_data, coordination, charges,
                          stream));

  auto refresh_and_compare = [&](std::uint64_t generation) -> int {
    CUDA_CHECK(device.positions.copy_from(positions.data(), positions.size(), stream));
    device.cache.geometry_generation = generation;
    CUDA_CHECK(device.reset(stream));
    CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_d4_geometry_cache_cuda(
        device.batch, device.parameters, device.positions.get(), device.cache, device.workspace,
        device.error.get(), stream));
    CHECK(gpuxtb::detail::gfn2::update_d4_geometry_cache_cpu(
              plan, positions.data(), generation, pair_data.data(), pair_data.size(),
              coordination.data(), coordination.size(), host_workspace, host_cache,
              error) == GPUXTB_STATUS_SUCCESS);

    std::vector<double> actual_pairs(pair_data.size());
    std::vector<double> actual_coordination(coordination.size());
    std::vector<std::uint64_t> actual_generations(batch_count, 0u);
    std::vector<std::uint32_t> system_errors(batch_count, 99u);
    std::uint32_t device_error = 99u;
    if (!actual_pairs.empty()) {
      CUDA_CHECK(device.pair_data.copy_to(actual_pairs.data(), actual_pairs.size(), stream));
    }
    CUDA_CHECK(device.coordination.copy_to(actual_coordination.data(), actual_coordination.size(),
                                           stream));
    CUDA_CHECK(device.geometry_generations.copy_to(actual_generations.data(),
                                                   actual_generations.size(), stream));
    CUDA_CHECK(device.system_errors.copy_to(system_errors.data(), system_errors.size(), stream));
    CUDA_CHECK(device.error.copy_to(&device_error, 1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(device_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
    CHECK(std::all_of(system_errors.begin(), system_errors.end(),
                      [](std::uint32_t value) { return value == 0u; }));
    CHECK(std::all_of(actual_generations.begin(), actual_generations.end(),
                      [generation](std::uint64_t value) { return value == generation; }));
    for (std::size_t element = 0; element < pair_data.size(); ++element) {
      CHECK(near(actual_pairs[element], pair_data[element], 3.0e-15, 4.0e-14));
    }
    for (std::size_t atom = 0; atom < coordination.size(); ++atom) {
      CHECK(near(actual_coordination[atom], coordination[atom], 3.0e-14, 4.0e-14));
    }
    return 0;
  };

  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    positions[atom * 3u] += 7.0e-4 * static_cast<double>(atom % 5u + 1u);
    positions[atom * 3u + 1u] -= 4.0e-4 * static_cast<double>(atom % 3u + 1u);
  }
  CHECK(refresh_and_compare(29u) == 0);
  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    positions[atom * 3u + 2u] += 9.0e-4 * static_cast<double>(atom % 7u + 1u);
  }
  CHECK(refresh_and_compare(30u) == 0);

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_geometry_refresh_cpu_parity_ragged_batches() {
  for (const std::size_t batch_count : std::array<std::size_t, 4>{1u, 8u, 32u, 128u}) {
    CHECK(run_geometry_refresh_batch_case(batch_count) == 0);
  }
  return 0;
}

int test_geometry_refresh_peer_and_plan_failure_atomicity() {
  HostFixture host;
  CHECK(host.initialize());
  const std::vector<double> baseline_pairs = host.pair_data;
  const std::vector<double> baseline_coordination = host.coordination;
  std::array<double, HostFixture::positions.size()> expected_positions = HostFixture::positions;
  expected_positions[12] += 0.31;
  std::string error;
  CHECK(gpuxtb::detail::gfn2::update_d4_geometry_cache_cpu(
            host.plan, expected_positions.data(), 31u, host.pair_data.data(), host.pair_data.size(),
            host.coordination.data(), host.coordination.size(), host.workspace, host.cache,
            error) == GPUXTB_STATUS_SUCCESS);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CHECK(device.initialize(host.plan, HostFixture::atom_offsets, HostFixture::atomic_numbers,
                          baseline_pairs, baseline_coordination, HostFixture::charges, stream));
  std::array<double, HostFixture::positions.size()> bad_positions = expected_positions;
  bad_positions[0] = std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(device.positions.copy_from(bad_positions.data(), bad_positions.size(), stream));
  device.cache.geometry_generation = 31u;
  CUDA_CHECK(device.reset(stream));
  CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_d4_geometry_cache_cuda(
      device.batch, device.parameters, device.positions.get(), device.cache, device.workspace,
      device.error.get(), stream));

  std::vector<double> actual_pairs(host.pair_data.size());
  std::vector<double> actual_coordination(host.coordination.size());
  std::array<std::uint64_t, 2> generations{};
  std::array<std::uint32_t, 2> system_errors{};
  std::uint32_t device_error = 99u;
  CUDA_CHECK(device.pair_data.copy_to(actual_pairs.data(), actual_pairs.size(), stream));
  CUDA_CHECK(
      device.coordination.copy_to(actual_coordination.data(), actual_coordination.size(), stream));
  CUDA_CHECK(device.geometry_generations.copy_to(generations.data(), generations.size(), stream));
  CUDA_CHECK(device.system_errors.copy_to(system_errors.data(), system_errors.size(), stream));
  CUDA_CHECK(device.error.copy_to(&device_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(system_errors[0] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kNonfinitePosition));
  CHECK(system_errors[1] == 0u);
  CHECK(device_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
  CHECK(generations[0] == 7u);
  CHECK(generations[1] == 31u);
  const std::size_t first_pair_elements = static_cast<std::size_t>(host.plan.pair_offsets()[1]) *
                                          gpuxtb::detail::gfn2::kD4PairDataElements;
  for (std::size_t element = 0; element < first_pair_elements; ++element) {
    CHECK(actual_pairs[element] == baseline_pairs[element]);
  }
  for (std::size_t element = first_pair_elements; element < actual_pairs.size(); ++element) {
    CHECK(near(actual_pairs[element], host.pair_data[element], 3.0e-15, 4.0e-14));
  }
  const std::size_t first_atoms = static_cast<std::size_t>(HostFixture::atom_offsets[1]);
  for (std::size_t atom = 0; atom < first_atoms; ++atom) {
    CHECK(actual_coordination[atom] == baseline_coordination[atom]);
  }
  for (std::size_t atom = first_atoms; atom < actual_coordination.size(); ++atom) {
    CHECK(near(actual_coordination[atom], host.coordination[atom], 3.0e-14, 4.0e-14));
  }

  /* Immutable topology corruption fails closed without publishing any peer. */
  const std::vector<double> before_plan_failure_pairs = actual_pairs;
  const std::vector<double> before_plan_failure_coordination = actual_coordination;
  constexpr std::array<std::int32_t, 5> invalid_atomic_numbers{0, 1, 1, 6, 8};
  CUDA_CHECK(device.atomic_numbers.copy_from(invalid_atomic_numbers.data(),
                                             invalid_atomic_numbers.size(), stream));
  CUDA_CHECK(
      device.positions.copy_from(expected_positions.data(), expected_positions.size(), stream));
  device.cache.geometry_generation = 32u;
  CUDA_CHECK(device.reset(stream));
  CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_d4_geometry_cache_cuda(
      device.batch, device.parameters, device.positions.get(), device.cache, device.workspace,
      device.error.get(), stream));
  CUDA_CHECK(device.pair_data.copy_to(actual_pairs.data(), actual_pairs.size(), stream));
  CUDA_CHECK(
      device.coordination.copy_to(actual_coordination.data(), actual_coordination.size(), stream));
  CUDA_CHECK(device.geometry_generations.copy_to(generations.data(), generations.size(), stream));
  CUDA_CHECK(device.system_errors.copy_to(system_errors.data(), system_errors.size(), stream));
  CUDA_CHECK(device.error.copy_to(&device_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(device_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kInvalidAtomicNumber));
  CHECK(system_errors[0] == 0u && system_errors[1] == 0u);
  CHECK(actual_pairs == before_plan_failure_pairs);
  CHECK(actual_coordination == before_plan_failure_coordination);
  CHECK(generations[0] == 7u && generations[1] == 31u);

  /* Host-visible aliases are rejected before any asynchronous work is queued. */
  Gfn2D4DeviceWorkspace aliased_workspace = device.workspace;
  aliased_workspace.pair_scratch = const_cast<double*>(device.cache.pair_data);
  CHECK(gpuxtb::detail::cuda::update_gfn2_d4_geometry_cache_cuda(
            device.batch, device.parameters, device.positions.get(), device.cache,
            aliased_workspace, device.error.get(), stream) == cudaErrorInvalidValue);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_geometry_refresh_graph_capture_and_replay() {
  HostFixture host;
  CHECK(host.initialize());
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CHECK(device.initialize(host, stream));
  device.cache.geometry_generation = 41u;

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CUDA_CHECK(device.reset(stream));
  CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_d4_geometry_cache_cuda(
      device.batch, device.parameters, device.positions.get(), device.cache, device.workspace,
      device.error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
      device.batch, device.parameters, device.cache, device.charges.get(), device.energies.get(),
      device.potentials.get(), device.workspace, device.error.get(), stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0u));

  auto replay_and_compare =
      [&](const std::array<double, HostFixture::positions.size()>& positions) -> int {
    std::string error;
    CHECK(gpuxtb::detail::gfn2::update_d4_geometry_cache_cpu(
              host.plan, positions.data(), 41u, host.pair_data.data(), host.pair_data.size(),
              host.coordination.data(), host.coordination.size(), host.workspace, host.cache,
              error) == GPUXTB_STATUS_SUCCESS);
    CHECK(gpuxtb::detail::gfn2::evaluate_d4_two_body_cpu(
              host.plan, host.cache, HostFixture::charges.data(), host.energies.data(),
              host.potentials.data(), host.workspace, error) == GPUXTB_STATUS_SUCCESS);
    CUDA_CHECK(device.positions.copy_from(positions.data(), positions.size(), stream));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));

    std::vector<double> actual_pairs(host.pair_data.size());
    std::vector<double> actual_coordination(host.coordination.size());
    std::array<double, 2> energies{};
    std::array<double, 5> potentials{};
    std::array<std::uint64_t, 2> generations{};
    std::uint32_t device_error = 99u;
    CUDA_CHECK(device.pair_data.copy_to(actual_pairs.data(), actual_pairs.size(), stream));
    CUDA_CHECK(device.coordination.copy_to(actual_coordination.data(), actual_coordination.size(),
                                           stream));
    CUDA_CHECK(device.energies.copy_to(energies.data(), energies.size(), stream));
    CUDA_CHECK(device.potentials.copy_to(potentials.data(), potentials.size(), stream));
    CUDA_CHECK(device.geometry_generations.copy_to(generations.data(), generations.size(), stream));
    CUDA_CHECK(device.error.copy_to(&device_error, 1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(device_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
    CHECK(generations[0] == 41u && generations[1] == 41u);
    for (std::size_t element = 0; element < actual_pairs.size(); ++element) {
      CHECK(near(actual_pairs[element], host.pair_data[element], 3.0e-15, 4.0e-14));
    }
    for (std::size_t atom = 0; atom < actual_coordination.size(); ++atom) {
      CHECK(near(actual_coordination[atom], host.coordination[atom], 3.0e-14, 4.0e-14));
      CHECK(near(potentials[atom], host.potentials[atom], 2.0e-12, 2.0e-13));
    }
    for (std::size_t system = 0; system < energies.size(); ++system) {
      CHECK(near(energies[system], host.energies[system], 2.0e-12, 2.0e-13));
    }
    return 0;
  };

  std::array<double, HostFixture::positions.size()> first_positions = HostFixture::positions;
  first_positions[3] += 0.08;
  first_positions[13] -= 0.06;
  CHECK(replay_and_compare(first_positions) == 0);
  std::array<double, HostFixture::positions.size()> second_positions = HostFixture::positions;
  second_positions[4] -= 0.11;
  second_positions[12] += 0.17;
  second_positions[14] += 0.09;
  CHECK(replay_and_compare(second_positions) == 0);

  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_cpu_parity_and_ragged_batch() {
  HostFixture host;
  CHECK(host.initialize());
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CHECK(device.initialize(host, stream));

  CUDA_CHECK(device.reset(stream));
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

int test_gradient_atm_and_finite_difference_parity() {
  HostFixture host;
  CHECK(host.initialize());
  std::array<double, 15> expected_two_body_gradient{};
  std::array<double, 15> expected_atm_gradient{};
  std::array<double, 2> expected_atm_energies{};
  std::string error;
  CHECK(gpuxtb::detail::gfn2::add_d4_two_body_gradient_cpu(
            host.plan, host.cache, HostFixture::charges.data(), expected_two_body_gradient.data(),
            host.workspace, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::evaluate_d4_atm_cpu(host.plan, host.cache,
                                                  expected_atm_energies.data(), host.workspace,
                                                  error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::add_d4_atm_gradient_cpu(host.plan, host.cache,
                                                      expected_atm_gradient.data(), host.workspace,
                                                      error) == GPUXTB_STATUS_SUCCESS);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CHECK(device.initialize(host, stream));
  std::array<double, 15> zeros{};

  CUDA_CHECK(device.gradients.copy_from(zeros.data(), zeros.size(), stream));
  CUDA_CHECK(device.reset(stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_d4_two_body_gradient_cuda(
      device.batch, device.parameters, device.cache, device.charges.get(), device.gradients.get(),
      device.workspace, device.error.get(), stream));
  std::array<double, 15> actual_two_body_gradient{};
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(device.gradients.copy_to(actual_two_body_gradient.data(),
                                      actual_two_body_gradient.size(), stream));
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
  for (std::size_t coordinate = 0; coordinate < zeros.size(); ++coordinate) {
    CHECK(near(actual_two_body_gradient[coordinate], expected_two_body_gradient[coordinate],
               5.0e-12, 5.0e-12));
  }

  CUDA_CHECK(device.reset(stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_atm_cuda(
      device.batch, device.parameters, device.cache, device.energies.get(), device.workspace,
      device.error.get(), stream));
  std::array<double, 2> actual_atm_energies{};
  CUDA_CHECK(
      device.energies.copy_to(actual_atm_energies.data(), actual_atm_energies.size(), stream));
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
  for (std::size_t system = 0; system < actual_atm_energies.size(); ++system) {
    CHECK(near(actual_atm_energies[system], expected_atm_energies[system], 5.0e-13, 5.0e-12));
  }

  CUDA_CHECK(device.gradients.copy_from(zeros.data(), zeros.size(), stream));
  CUDA_CHECK(device.reset(stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_d4_atm_gradient_cuda(
      device.batch, device.parameters, device.cache, device.gradients.get(), device.workspace,
      device.error.get(), stream));
  std::array<double, 15> actual_atm_gradient{};
  CUDA_CHECK(
      device.gradients.copy_to(actual_atm_gradient.data(), actual_atm_gradient.size(), stream));
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
  for (std::size_t coordinate = 0; coordinate < zeros.size(); ++coordinate) {
    CHECK(
        near(actual_atm_gradient[coordinate], expected_atm_gradient[coordinate], 2.0e-11, 2.0e-10));
  }

  constexpr double step = 1.0e-5;
  auto displaced_energy = [&](double displacement) -> double {
    std::array<double, 15> positions = HostFixture::positions;
    positions[3] += displacement;
    CHECK(gpuxtb::detail::gfn2::update_d4_geometry_cache_cpu(
              host.plan, positions.data(), 19u, host.pair_data.data(), host.pair_data.size(),
              host.coordination.data(), host.coordination.size(), host.workspace, host.cache,
              error) == GPUXTB_STATUS_SUCCESS);
    std::array<double, 2> two_body{};
    std::array<double, 5> potentials{};
    std::array<double, 2> atm{};
    CHECK(gpuxtb::detail::gfn2::evaluate_d4_two_body_cpu(
              host.plan, host.cache, HostFixture::charges.data(), two_body.data(),
              potentials.data(), host.workspace, error) == GPUXTB_STATUS_SUCCESS);
    CHECK(gpuxtb::detail::gfn2::evaluate_d4_atm_cpu(
              host.plan, host.cache, atm.data(), host.workspace, error) == GPUXTB_STATUS_SUCCESS);
    return two_body[0] + atm[0];
  };
  const double finite_difference =
      (displaced_energy(step) - displaced_energy(-step)) / (2.0 * step);
  CHECK(near(actual_two_body_gradient[3] + actual_atm_gradient[3], finite_difference, 2.0e-8,
             2.0e-6));

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
  CUDA_CHECK(device.reset(stream));
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

int test_complete_path_batch_sizes() {
  for (const std::size_t batch_count : std::array<std::size_t, 4>{1u, 8u, 32u, 128u}) {
    constexpr std::size_t atoms_per_system = 4u;
    const std::size_t atom_count = batch_count * atoms_per_system;
    std::vector<std::int64_t> atom_offsets(batch_count + 1u);
    std::vector<std::int32_t> atomic_numbers(atom_count);
    std::vector<double> positions(atom_count * 3u);
    std::vector<double> charges(atom_count);
    for (std::size_t system = 0; system < batch_count; ++system) {
      atom_offsets[system] = static_cast<std::int64_t>(system * atoms_per_system);
      const std::size_t atom = system * atoms_per_system;
      atomic_numbers[atom] = 8;
      atomic_numbers[atom + 1u] = 1;
      atomic_numbers[atom + 2u] = 1;
      atomic_numbers[atom + 3u] = 6;
      positions[(atom + 1u) * 3u] = 1.43 + 1.0e-3 * static_cast<double>(system % 7u);
      positions[(atom + 1u) * 3u + 1u] = 1.11;
      positions[(atom + 2u) * 3u] = -1.43;
      positions[(atom + 2u) * 3u + 1u] = 1.11 + 1.0e-3 * static_cast<double>(system % 5u);
      positions[(atom + 3u) * 3u] = 0.35;
      positions[(atom + 3u) * 3u + 1u] = -0.20;
      positions[(atom + 3u) * 3u + 2u] = 2.45 + 1.0e-3 * static_cast<double>(system % 3u);
      charges[atom] = -0.52;
      charges[atom + 1u] = 0.20;
      charges[atom + 2u] = 0.22;
      charges[atom + 3u] = 0.10;
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
              plan, positions.data(), 23u, pair_data.data(), pair_data.size(), coordination.data(),
              coordination.size(), host_workspace, host_cache, error) == GPUXTB_STATUS_SUCCESS);
    std::vector<double> expected_two_body(batch_count);
    std::vector<double> expected_potentials(atom_count);
    std::vector<double> expected_atm(batch_count);
    std::vector<double> expected_gradients(atom_count * 3u);
    CHECK(gpuxtb::detail::gfn2::evaluate_d4_two_body_cpu(
              plan, host_cache, charges.data(), expected_two_body.data(),
              expected_potentials.data(), host_workspace, error) == GPUXTB_STATUS_SUCCESS);
    CHECK(gpuxtb::detail::gfn2::evaluate_d4_atm_cpu(plan, host_cache, expected_atm.data(),
                                                    host_workspace,
                                                    error) == GPUXTB_STATUS_SUCCESS);
    CHECK(gpuxtb::detail::gfn2::add_d4_two_body_gradient_cpu(
              plan, host_cache, charges.data(), expected_gradients.data(), host_workspace, error) ==
          GPUXTB_STATUS_SUCCESS);
    CHECK(gpuxtb::detail::gfn2::add_d4_atm_gradient_cpu(plan, host_cache, expected_gradients.data(),
                                                        host_workspace,
                                                        error) == GPUXTB_STATUS_SUCCESS);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    DeviceFixture device;
    CHECK(device.initialize(plan, atom_offsets, atomic_numbers, pair_data, coordination, charges,
                            stream));
    CUDA_CHECK(device.reset(stream));
    CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
        device.batch, device.parameters, device.cache, device.charges.get(), device.energies.get(),
        device.potentials.get(), device.workspace, device.error.get(), stream));
    std::vector<double> actual_two_body(batch_count);
    std::vector<double> actual_potentials(atom_count);
    CUDA_CHECK(device.energies.copy_to(actual_two_body.data(), batch_count, stream));
    CUDA_CHECK(device.potentials.copy_to(actual_potentials.data(), atom_count, stream));
    CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_atm_cuda(
        device.batch, device.parameters, device.cache, device.energies.get(), device.workspace,
        device.error.get(), stream));
    std::vector<double> actual_atm(batch_count);
    CUDA_CHECK(device.energies.copy_to(actual_atm.data(), batch_count, stream));
    std::vector<double> zero_gradients(atom_count * 3u);
    CUDA_CHECK(device.gradients.copy_from(zero_gradients.data(), zero_gradients.size(), stream));
    CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_d4_two_body_gradient_cuda(
        device.batch, device.parameters, device.cache, device.charges.get(), device.gradients.get(),
        device.workspace, device.error.get(), stream));
    CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_d4_atm_gradient_cuda(
        device.batch, device.parameters, device.cache, device.gradients.get(), device.workspace,
        device.error.get(), stream));
    std::vector<double> actual_gradients(atom_count * 3u);
    std::uint32_t semantic_error = 99u;
    CUDA_CHECK(device.gradients.copy_to(actual_gradients.data(), actual_gradients.size(), stream));
    CUDA_CHECK(device.error.copy_to(&semantic_error, 1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
    for (std::size_t system = 0; system < batch_count; ++system) {
      CHECK(near(actual_two_body[system], expected_two_body[system], 2.0e-12, 2.0e-12));
      CHECK(near(actual_atm[system], expected_atm[system], 2.0e-12, 2.0e-11));
    }
    for (std::size_t atom = 0; atom < atom_count; ++atom) {
      CHECK(near(actual_potentials[atom], expected_potentials[atom], 2.0e-12, 2.0e-12));
    }
    for (std::size_t coordinate = 0; coordinate < actual_gradients.size(); ++coordinate) {
      CHECK(near(actual_gradients[coordinate], expected_gradients[coordinate], 3.0e-11, 3.0e-10));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
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
  CUDA_CHECK(device.reset(stream));
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
  CUDA_CHECK(device.reset(stream));
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
  CUDA_CHECK(device.reset(stream));
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
  std::array<std::uint32_t, 2> system_errors{};
  invalid_charges[3] = std::numeric_limits<double>::quiet_NaN();

  CUDA_CHECK(device.charges.copy_from(invalid_charges.data(), invalid_charges.size(), stream));
  CUDA_CHECK(device.energies.copy_from(energy_sentinel.data(), energy_sentinel.size(), stream));
  CUDA_CHECK(
      device.potentials.copy_from(potential_sentinel.data(), potential_sentinel.size(), stream));
  CUDA_CHECK(device.reset(stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
      device.batch, device.parameters, device.cache, device.charges.get(), device.energies.get(),
      device.potentials.get(), device.workspace, device.error.get(), stream));
  std::array<double, 2> actual_energies{};
  std::array<double, 5> actual_potentials{};
  std::uint32_t semantic_error = 0u;
  CUDA_CHECK(device.energies.copy_to(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(device.potentials.copy_to(actual_potentials.data(), actual_potentials.size(), stream));
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(device.system_errors.copy_to(system_errors.data(), system_errors.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
  CHECK(system_errors[0] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
  CHECK(system_errors[1] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kNonfiniteCharge));
  CHECK(near(actual_energies[0], host.energies[0], 2.0e-12, 2.0e-13));
  CHECK(actual_energies[1] == energy_sentinel[1]);
  for (std::size_t atom = 0; atom < 3u; ++atom) {
    CHECK(near(actual_potentials[atom], host.potentials[atom], 2.0e-12, 2.0e-13));
  }
  CHECK(actual_potentials[3] == potential_sentinel[3]);
  CHECK(actual_potentials[4] == potential_sentinel[4]);

  /* A failed dependent sequence remains inert until the caller resets it. */
  CUDA_CHECK(
      device.charges.copy_from(HostFixture::charges.data(), HostFixture::charges.size(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
      device.batch, device.parameters, device.cache, device.charges.get(), device.energies.get(),
      device.potentials.get(), device.workspace, device.error.get(), stream));
  CUDA_CHECK(device.energies.copy_to(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(device.potentials.copy_to(actual_potentials.data(), actual_potentials.size(), stream));
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(device.system_errors.copy_to(system_errors.data(), system_errors.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
  CHECK(system_errors[0] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
  CHECK(system_errors[1] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kNonfiniteCharge));
  CHECK(near(actual_energies[0], host.energies[0], 2.0e-12, 2.0e-13));
  CHECK(actual_energies[1] == energy_sentinel[1]);

  CUDA_CHECK(device.energies.copy_from(energy_sentinel.data(), energy_sentinel.size(), stream));
  CUDA_CHECK(
      device.potentials.copy_from(potential_sentinel.data(), potential_sentinel.size(), stream));

  constexpr std::array<double, 15> gradient_sentinel{1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0, 8.0,
                                                     9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
  std::array<double, 15> expected_two_body_gradient{};
  std::array<double, 15> actual_gradients{};
  std::string error;
  CHECK(gpuxtb::detail::gfn2::add_d4_two_body_gradient_cpu(
            host.plan, host.cache, HostFixture::charges.data(), expected_two_body_gradient.data(),
            host.workspace, error) == GPUXTB_STATUS_SUCCESS);
  CUDA_CHECK(device.charges.copy_from(invalid_charges.data(), invalid_charges.size(), stream));
  CUDA_CHECK(
      device.gradients.copy_from(gradient_sentinel.data(), gradient_sentinel.size(), stream));
  CUDA_CHECK(device.reset(stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_d4_two_body_gradient_cuda(
      device.batch, device.parameters, device.cache, device.charges.get(), device.gradients.get(),
      device.workspace, device.error.get(), stream));
  CUDA_CHECK(device.gradients.copy_to(actual_gradients.data(), actual_gradients.size(), stream));
  CUDA_CHECK(device.system_errors.copy_to(system_errors.data(), system_errors.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(system_errors[0] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
  CHECK(system_errors[1] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kNonfiniteCharge));
  for (std::size_t coordinate = 0; coordinate < 9u; ++coordinate) {
    CHECK(near(actual_gradients[coordinate],
               gradient_sentinel[coordinate] + expected_two_body_gradient[coordinate], 2.0e-11,
               2.0e-10));
  }
  for (std::size_t coordinate = 9u; coordinate < actual_gradients.size(); ++coordinate) {
    CHECK(actual_gradients[coordinate] == gradient_sentinel[coordinate]);
  }

  std::vector<double> invalid_pair_data = host.pair_data;
  invalid_pair_data[static_cast<std::size_t>(host.plan.pair_offsets()[1]) *
                    gpuxtb::detail::gfn2::kD4PairDataElements] =
      std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(
      device.pair_data.copy_from(invalid_pair_data.data(), invalid_pair_data.size(), stream));
  std::array<double, 2> expected_atm{};
  std::array<double, 15> expected_atm_gradient{};
  CHECK(gpuxtb::detail::gfn2::evaluate_d4_atm_cpu(host.plan, host.cache, expected_atm.data(),
                                                  host.workspace, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::add_d4_atm_gradient_cpu(host.plan, host.cache,
                                                      expected_atm_gradient.data(), host.workspace,
                                                      error) == GPUXTB_STATUS_SUCCESS);
  CUDA_CHECK(device.energies.copy_from(energy_sentinel.data(), energy_sentinel.size(), stream));
  CUDA_CHECK(
      device.gradients.copy_from(gradient_sentinel.data(), gradient_sentinel.size(), stream));
  CUDA_CHECK(device.reset(stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_atm_cuda(
      device.batch, device.parameters, device.cache, device.energies.get(), device.workspace,
      device.error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_d4_atm_gradient_cuda(
      device.batch, device.parameters, device.cache, device.gradients.get(), device.workspace,
      device.error.get(), stream));
  CUDA_CHECK(device.energies.copy_to(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(device.gradients.copy_to(actual_gradients.data(), actual_gradients.size(), stream));
  CUDA_CHECK(device.system_errors.copy_to(system_errors.data(), system_errors.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(system_errors[0] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
  CHECK(system_errors[1] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kNonfiniteArithmetic));
  CHECK(near(actual_energies[0], expected_atm[0], 2.0e-12, 2.0e-11));
  CHECK(actual_energies[1] == energy_sentinel[1]);
  for (std::size_t coordinate = 0; coordinate < 9u; ++coordinate) {
    CHECK(near(actual_gradients[coordinate],
               gradient_sentinel[coordinate] + expected_atm_gradient[coordinate], 3.0e-11,
               3.0e-10));
  }
  for (std::size_t coordinate = 9u; coordinate < actual_gradients.size(); ++coordinate) {
    CHECK(actual_gradients[coordinate] == gradient_sentinel[coordinate]);
  }
  CUDA_CHECK(device.pair_data.copy_from(host.pair_data.data(), host.pair_data.size(), stream));
  CUDA_CHECK(device.energies.copy_from(energy_sentinel.data(), energy_sentinel.size(), stream));
  CUDA_CHECK(
      device.potentials.copy_from(potential_sentinel.data(), potential_sentinel.size(), stream));

  /* Invalid ragged topology is detected before any numerical kernel publishes. */
  constexpr std::array<std::int64_t, 3> invalid_pair_offsets{0, 2, 4};
  CUDA_CHECK(device.pair_offsets.copy_from(invalid_pair_offsets.data(), invalid_pair_offsets.size(),
                                           stream));
  CUDA_CHECK(device.reset(stream));
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

  CUDA_CHECK(
      device.gradients.copy_from(gradient_sentinel.data(), gradient_sentinel.size(), stream));
  CUDA_CHECK(device.reset(stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_d4_two_body_gradient_cuda(
      device.batch, device.parameters, device.cache, device.charges.get(), device.gradients.get(),
      device.workspace, device.error.get(), stream));
  CUDA_CHECK(device.gradients.copy_to(actual_gradients.data(), actual_gradients.size(), stream));
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kInvalidOffsets));
  CHECK(actual_gradients == gradient_sentinel);

  CUDA_CHECK(device.energies.copy_from(energy_sentinel.data(), energy_sentinel.size(), stream));
  CUDA_CHECK(device.reset(stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_atm_cuda(
      device.batch, device.parameters, device.cache, device.energies.get(), device.workspace,
      device.error.get(), stream));
  CUDA_CHECK(device.energies.copy_to(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kInvalidOffsets));
  CHECK(actual_energies == energy_sentinel);

  CUDA_CHECK(device.reset(stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_d4_atm_gradient_cuda(
      device.batch, device.parameters, device.cache, device.gradients.get(), device.workspace,
      device.error.get(), stream));
  CUDA_CHECK(device.gradients.copy_to(actual_gradients.data(), actual_gradients.size(), stream));
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kInvalidOffsets));
  CHECK(actual_gradients == gradient_sentinel);

  /* A pre-set sticky failure makes every complete-D4 entry point output-inert. */
  CUDA_CHECK(device.pair_offsets.copy_from(host.plan.pair_offsets().data(),
                                           host.plan.pair_offsets().size(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_atm_cuda(
      device.batch, device.parameters, device.cache, device.energies.get(), device.workspace,
      device.error.get(), stream));
  CUDA_CHECK(device.energies.copy_to(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(actual_energies == energy_sentinel);

  Gfn2D4DeviceWorkspace undersized_workspace = device.workspace;
  --undersized_workspace.weight_elements;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
            device.batch, device.parameters, device.cache, device.charges.get(),
            device.energies.get(), device.potentials.get(), undersized_workspace,
            device.error.get(), stream) == cudaErrorInvalidValue);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_reduction_and_addition_overflow_atomicity() {
  HostFixture host;
  CHECK(host.initialize());
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  DeviceFixture reduction_device;
  CHECK(reduction_device.initialize(host, stream));
  std::vector<Gfn2D4DeviceElementData> synthetic_elements;
  synthetic_elements.reserve(gpuxtb::parameters::d4::kElements.size());
  for (const auto& element : gpuxtb::parameters::d4::kElements) {
    synthetic_elements.push_back({0u, 1u, element.covalent_radius, element.electronegativity, 1.0,
                                  element.hardness, element.r4r2});
  }
  std::vector<double> synthetic_c6(gpuxtb::parameters::d4::kReferenceC6.size(), 0.0);
  synthetic_c6[0] = 4.0e305;
  std::vector<double> synthetic_pair_data = host.pair_data;
  for (std::size_t pair = 0; pair < static_cast<std::size_t>(host.plan.total_pairs()); ++pair) {
    synthetic_pair_data[pair * gpuxtb::detail::gfn2::kD4PairDataElements + 3u] = 1.0;
  }
  const std::array<double, 5> negative_qmod_charges{-2.0, -2.0, -2.0, -2.0, -2.0};
  constexpr std::array<double, 2> energy_sentinel{123.0, -456.0};
  constexpr std::array<double, 5> potential_sentinel{1.0, 2.0, 3.0, 4.0, 5.0};
  CUDA_CHECK(reduction_device.elements.copy_from(synthetic_elements.data(),
                                                 synthetic_elements.size(), stream));
  CUDA_CHECK(
      reduction_device.reference_c6.copy_from(synthetic_c6.data(), synthetic_c6.size(), stream));
  CUDA_CHECK(reduction_device.pair_data.copy_from(synthetic_pair_data.data(),
                                                  synthetic_pair_data.size(), stream));
  CUDA_CHECK(reduction_device.charges.copy_from(negative_qmod_charges.data(),
                                                negative_qmod_charges.size(), stream));
  CUDA_CHECK(
      reduction_device.energies.copy_from(energy_sentinel.data(), energy_sentinel.size(), stream));
  CUDA_CHECK(reduction_device.potentials.copy_from(potential_sentinel.data(),
                                                   potential_sentinel.size(), stream));
  CUDA_CHECK(reduction_device.reset(stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
      reduction_device.batch, reduction_device.parameters, reduction_device.cache,
      reduction_device.charges.get(), reduction_device.energies.get(),
      reduction_device.potentials.get(), reduction_device.workspace, reduction_device.error.get(),
      stream));
  std::array<double, 2> actual_energies{};
  std::array<double, 5> actual_potentials{};
  std::array<std::uint32_t, 2> system_errors{};
  std::uint32_t sequence_error = 99u;
  CUDA_CHECK(
      reduction_device.energies.copy_to(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(reduction_device.potentials.copy_to(actual_potentials.data(), actual_potentials.size(),
                                                 stream));
  CUDA_CHECK(
      reduction_device.system_errors.copy_to(system_errors.data(), system_errors.size(), stream));
  CUDA_CHECK(reduction_device.error.copy_to(&sequence_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(sequence_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
  CHECK(system_errors[0] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kNonfiniteArithmetic));
  CHECK(system_errors[1] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
  CHECK(actual_energies[0] == energy_sentinel[0]);
  const double healthy_pair_energy = -std::exp(6.0) * synthetic_c6[0];
  CHECK(near(actual_energies[1], healthy_pair_energy, 0.0, 2.0e-15));
  CHECK(actual_potentials[0] == potential_sentinel[0]);
  CHECK(actual_potentials[1] == potential_sentinel[1]);
  CHECK(actual_potentials[2] == potential_sentinel[2]);
  CHECK(actual_potentials[3] == 0.0);
  CHECK(actual_potentials[4] == 0.0);

  const std::array<double, 5> reduction_values{9.0e307, 9.0e307, 0.0, 3.0, 4.0};
  CUDA_CHECK(
      reduction_device.charges.copy_from(reduction_values.data(), reduction_values.size(), stream));
  CUDA_CHECK(
      reduction_device.energies.copy_from(energy_sentinel.data(), energy_sentinel.size(), stream));
  CUDA_CHECK(reduction_device.reset(stream));
  CUDA_CHECK(gpuxtb::detail::cuda::test_gfn2_d4_atm_reduction_cuda(
      reduction_device.batch, reduction_device.charges.get(), reduction_device.energies.get(),
      reduction_device.workspace, reduction_device.error.get(), stream));
  CUDA_CHECK(
      reduction_device.energies.copy_to(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(
      reduction_device.system_errors.copy_to(system_errors.data(), system_errors.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(system_errors[0] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kNonfiniteArithmetic));
  CHECK(system_errors[1] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
  CHECK(actual_energies[0] == energy_sentinel[0]);
  CHECK(actual_energies[1] == 7.0);

  // Exercise the real two-body gradient path with a large but finite radial
  // derivative. Adding that derivative to DBL_MAX must reject the entire bad
  // system before any coordinate is published, while its healthy peer commits.
  synthetic_c6[0] = 1.0e298;
  for (std::size_t pair = 0; pair < static_cast<std::size_t>(host.plan.total_pairs()); ++pair) {
    synthetic_pair_data[pair * gpuxtb::detail::gfn2::kD4PairDataElements + 4u] = 1.0e6;
  }
  CUDA_CHECK(
      reduction_device.reference_c6.copy_from(synthetic_c6.data(), synthetic_c6.size(), stream));
  CUDA_CHECK(reduction_device.pair_data.copy_from(synthetic_pair_data.data(),
                                                  synthetic_pair_data.size(), stream));
  std::array<double, 15> zero_gradients{};
  CUDA_CHECK(
      reduction_device.gradients.copy_from(zero_gradients.data(), zero_gradients.size(), stream));
  CUDA_CHECK(reduction_device.reset(stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_d4_two_body_gradient_cuda(
      reduction_device.batch, reduction_device.parameters, reduction_device.cache,
      reduction_device.charges.get(), reduction_device.gradients.get(), reduction_device.workspace,
      reduction_device.error.get(), stream));
  std::array<double, 15> finite_two_body_delta{};
  CUDA_CHECK(reduction_device.gradients.copy_to(finite_two_body_delta.data(),
                                                finite_two_body_delta.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::size_t overflow_coordinate = 0u;
  while (overflow_coordinate < 9u &&
         std::abs(finite_two_body_delta[overflow_coordinate]) <= 1.0e292) {
    ++overflow_coordinate;
  }
  CHECK(overflow_coordinate < 9u);
  std::array<double, 15> two_body_seeds{};
  two_body_seeds[overflow_coordinate] =
      std::copysign(std::numeric_limits<double>::max(), finite_two_body_delta[overflow_coordinate]);
  CUDA_CHECK(
      reduction_device.gradients.copy_from(two_body_seeds.data(), two_body_seeds.size(), stream));
  CUDA_CHECK(reduction_device.reset(stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_d4_two_body_gradient_cuda(
      reduction_device.batch, reduction_device.parameters, reduction_device.cache,
      reduction_device.charges.get(), reduction_device.gradients.get(), reduction_device.workspace,
      reduction_device.error.get(), stream));
  std::array<double, 15> actual_two_body_gradient{};
  CUDA_CHECK(reduction_device.gradients.copy_to(actual_two_body_gradient.data(),
                                                actual_two_body_gradient.size(), stream));
  CUDA_CHECK(
      reduction_device.system_errors.copy_to(system_errors.data(), system_errors.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(system_errors[0] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kNonfiniteArithmetic));
  CHECK(system_errors[1] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
  for (std::size_t coordinate = 0; coordinate < 9u; ++coordinate) {
    CHECK(actual_two_body_gradient[coordinate] == two_body_seeds[coordinate]);
  }
  for (std::size_t coordinate = 9u; coordinate < actual_two_body_gradient.size(); ++coordinate) {
    CHECK(actual_two_body_gradient[coordinate] == finite_two_body_delta[coordinate]);
  }

  // The ATM hook only supplies finite deltas; publication still runs through
  // the production preflight and add kernels shared by both gradient paths.
  DeviceBuffer<double> atm_deltas;
  CUDA_CHECK(atm_deltas.allocate(15u));
  std::array<double, 15> finite_atm_deltas{};
  std::array<double, 15> atm_seeds{};
  finite_atm_deltas[0] = 9.0e307;
  finite_atm_deltas[9] = 2.0;
  atm_seeds[0] = 9.0e307;
  atm_seeds[10] = -3.0;
  CUDA_CHECK(atm_deltas.copy_from(finite_atm_deltas.data(), finite_atm_deltas.size(), stream));
  CUDA_CHECK(reduction_device.gradients.copy_from(atm_seeds.data(), atm_seeds.size(), stream));
  CUDA_CHECK(reduction_device.reset(stream));
  CUDA_CHECK(gpuxtb::detail::cuda::test_gfn2_d4_atm_addition_cuda(
      reduction_device.batch, atm_deltas.get(), reduction_device.gradients.get(),
      reduction_device.workspace, reduction_device.error.get(), stream));
  std::array<double, 15> actual_atm_gradient{};
  CUDA_CHECK(reduction_device.gradients.copy_to(actual_atm_gradient.data(),
                                                actual_atm_gradient.size(), stream));
  CUDA_CHECK(
      reduction_device.system_errors.copy_to(system_errors.data(), system_errors.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(system_errors[0] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kNonfiniteArithmetic));
  CHECK(system_errors[1] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess));
  for (std::size_t coordinate = 0; coordinate < 9u; ++coordinate) {
    CHECK(actual_atm_gradient[coordinate] == atm_seeds[coordinate]);
  }
  for (std::size_t coordinate = 9u; coordinate < actual_atm_gradient.size(); ++coordinate) {
    CHECK(actual_atm_gradient[coordinate] == atm_seeds[coordinate] + finite_atm_deltas[coordinate]);
  }

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
  aliased_workspace.weight_cn_derivatives = aliased_workspace.weights;
  CHECK(gpuxtb::detail::cuda::add_gfn2_d4_two_body_gradient_cuda(
            device.batch, device.parameters, device.cache, device.charges.get(),
            device.gradients.get(), aliased_workspace, device.error.get(),
            stream) == cudaErrorInvalidValue);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_atm_cuda(
            device.batch, device.parameters, device.cache, device.workspace.batch_scratch,
            device.workspace, device.error.get(), stream) == cudaErrorInvalidValue);
  CHECK(gpuxtb::detail::cuda::add_gfn2_d4_atm_gradient_cuda(
            device.batch, device.parameters, device.cache,
            const_cast<double*>(device.cache.coordination_numbers), device.workspace,
            device.error.get(), stream) == cudaErrorInvalidValue);

  Gfn2D4DeviceCache wrong_provenance = device.cache;
  wrong_provenance.plan_token ^= 0x1u;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_atm_cuda(
            device.batch, device.parameters, wrong_provenance, device.energies.get(),
            device.workspace, device.error.get(), stream) == cudaErrorInvalidValue);
  wrong_provenance = device.cache;
  wrong_provenance.geometry_generation = 0u;
  CHECK(gpuxtb::detail::cuda::add_gfn2_d4_two_body_gradient_cuda(
            device.batch, device.parameters, wrong_provenance, device.charges.get(),
            device.gradients.get(), device.workspace, device.error.get(),
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
  CHECK(gpuxtb::detail::cuda::reset_gfn2_d4_device_errors_cuda(
            device.batch.batch_size, device.system_errors.get(),
            reinterpret_cast<std::uint32_t*>(aligned_maximum_error_address),
            stream) == cudaErrorInvalidValue);

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
  const std::array<double, 15> zero_gradients{};
  CUDA_CHECK(device.gradients.copy_from(zero_gradients.data(), zero_gradients.size(), stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CUDA_CHECK(device.reset(stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_two_body_cuda(
      device.batch, device.parameters, device.cache, device.charges.get(), device.energies.get(),
      device.potentials.get(), device.workspace, device.error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_d4_two_body_gradient_cuda(
      device.batch, device.parameters, device.cache, device.charges.get(), device.gradients.get(),
      device.workspace, device.error.get(), stream));
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

  std::array<double, 2> expected_atm{};
  std::array<double, 15> expected_atm_gradient{};
  CHECK(gpuxtb::detail::gfn2::evaluate_d4_atm_cpu(host.plan, host.cache, expected_atm.data(),
                                                  host.workspace, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::add_d4_atm_gradient_cpu(host.plan, host.cache,
                                                      expected_atm_gradient.data(), host.workspace,
                                                      error) == GPUXTB_STATUS_SUCCESS);
  CUDA_CHECK(device.gradients.copy_from(zero_gradients.data(), zero_gradients.size(), stream));
  graph = nullptr;
  executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CUDA_CHECK(device.reset(stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_atm_cuda(
      device.batch, device.parameters, device.cache, device.energies.get(), device.workspace,
      device.error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_d4_atm_gradient_cuda(
      device.batch, device.parameters, device.cache, device.gradients.get(), device.workspace,
      device.error.get(), stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0u));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  std::array<double, 15> actual_atm_gradient{};
  CUDA_CHECK(device.energies.copy_to(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(
      device.gradients.copy_to(actual_atm_gradient.data(), actual_atm_gradient.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (std::size_t system = 0; system < actual_energies.size(); ++system) {
    CHECK(near(actual_energies[system], expected_atm[system], 2.0e-12, 2.0e-11));
  }
  for (std::size_t coordinate = 0; coordinate < actual_atm_gradient.size(); ++coordinate) {
    CHECK(near(actual_atm_gradient[coordinate], 2.0 * expected_atm_gradient[coordinate], 3.0e-11,
               3.0e-10));
  }
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_geometry_refresh_cpu_parity_ragged_batches(); status != 0) {
    std::cerr << "CUDA D4 geometry-refresh parity test failed at line " << status << '\n';
    return status;
  }
  if (const int status = test_geometry_refresh_peer_and_plan_failure_atomicity(); status != 0) {
    std::cerr << "CUDA D4 geometry-refresh atomicity test failed at line " << status << '\n';
    return status;
  }
  if (const int status = test_geometry_refresh_graph_capture_and_replay(); status != 0) {
    std::cerr << "CUDA D4 geometry-refresh Graph test failed at line " << status << '\n';
    return status;
  }
  if (const int status = test_cpu_parity_and_ragged_batch(); status != 0) {
    std::cerr << "CUDA D4 CPU-parity test failed at line " << status << '\n';
    return status;
  }
  if (const int status = test_gradient_atm_and_finite_difference_parity(); status != 0) {
    std::cerr << "CUDA D4 gradient/ATM parity test failed at line " << status << '\n';
    return status;
  }
  if (const int status = test_all_supported_elements_cpu_parity(); status != 0) {
    std::cerr << "CUDA D4 all-element parity test failed at line " << status << '\n';
    return status;
  }
  if (const int status = test_complete_path_batch_sizes(); status != 0) {
    std::cerr << "CUDA D4 complete batch path test failed at line " << status << '\n';
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
  if (const int status = test_reduction_and_addition_overflow_atomicity(); status != 0) {
    std::cerr << "CUDA D4 reduction/addition overflow test failed at line " << status << '\n';
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
