#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_aes2.cuh"
#include "model/gfn2/aes2.hpp"
#include "model/gfn2/basis.hpp"
#ifdef XTBLOOM_AES2_TERM_BENCHMARK_ONLY
#include "tests/support/cuda_term_benchmark.hpp"
#endif

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using xtbloom::detail::cuda::Gfn2AES2DeviceBatch;
using xtbloom::detail::cuda::Gfn2AES2DeviceCache;
using xtbloom::detail::cuda::Gfn2AES2DeviceError;
using xtbloom::detail::cuda::Gfn2AES2DeviceWorkspace;
using xtbloom::detail::gfn2::AES2GeometryCache;
using xtbloom::detail::gfn2::AES2Plan;
using xtbloom::detail::gfn2::AES2Workspace;
using xtbloom::detail::gfn2::BasisPlan;

constexpr std::uint64_t kGeneration = 91u;
constexpr std::uint64_t kPlanToken = 0x6a09e667f3bcc909ULL;

static_assert(std::is_trivially_copyable_v<Gfn2AES2DeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2AES2DeviceCache>);
static_assert(std::is_trivially_copyable_v<Gfn2AES2DeviceWorkspace>);

bool near(double actual, double expected, double tolerance = 2.0e-12) {
  return std::abs(actual - expected) <=
         tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0u)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0u);
    }
    return *this;
  }
  ~DeviceBuffer() { release(); }

  bool allocate(std::size_t count) {
    release();
    count_ = count;
    if (count == 0u) {
      return true;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)) != cudaSuccess) {
      data_ = nullptr;
      count_ = 0u;
      return false;
    }
    return true;
  }

  bool copy_from(const T* source, std::size_t count, cudaStream_t stream = nullptr) {
    if (count > count_ || (count != 0u && source == nullptr)) {
      return false;
    }
    return count == 0u || cudaMemcpyAsync(data_, source, count * sizeof(T), cudaMemcpyHostToDevice,
                                          stream) == cudaSuccess;
  }

  bool copy_to(T* target, std::size_t count, cudaStream_t stream = nullptr) const {
    if (count > count_ || (count != 0u && target == nullptr)) {
      return false;
    }
    return count == 0u || cudaMemcpyAsync(target, data_, count * sizeof(T), cudaMemcpyDeviceToHost,
                                          stream) == cudaSuccess;
  }

  bool zero(cudaStream_t stream = nullptr) {
    return count_ == 0u || cudaMemsetAsync(data_, 0, count_ * sizeof(T), stream) == cudaSuccess;
  }

  T* get() { return data_; }
  const T* get() const { return data_; }
  std::size_t size() const { return count_; }

 private:
  void release() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
    data_ = nullptr;
    count_ = 0u;
  }

  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

struct HostEvaluation {
  BasisPlan basis;
  AES2Plan plan;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> coordination;
  std::vector<double> charges;
  std::vector<double> dipoles;
  std::vector<double> quadrupoles;
  std::vector<double> pair_data;
  std::vector<double> charge_potentials;
  std::vector<double> dipole_potentials;
  std::vector<double> quadrupole_potentials;
  std::vector<double> energies;
  std::vector<double> gradients;
  std::vector<double> coordination_adjoints;
  std::vector<double> pair_scratch;
  std::vector<double> potential_scratch;
  std::vector<double> batch_scratch;
  std::vector<double> gradient_scratch;
  std::vector<double> coordination_scratch;
  AES2Workspace workspace;
  AES2GeometryCache cache;
};

bool make_host_evaluation(const std::vector<std::int64_t>& atom_offsets,
                          const std::vector<std::int32_t>& atomic_numbers,
                          const std::vector<double>& positions,
                          const std::vector<double>& coordination,
                          const std::vector<double>& charges, const std::vector<double>& dipoles,
                          const std::vector<double>& quadrupoles, HostEvaluation& host,
                          std::string& error) {
  const std::int64_t batch_size = static_cast<std::int64_t>(atom_offsets.size() - 1u);
  if (xtbloom::detail::gfn2::make_basis_plan(
          batch_size, static_cast<std::int64_t>(atomic_numbers.size()), atom_offsets.data(),
          atomic_numbers.data(), host.basis, error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_aes2_plan(host.basis, atomic_numbers.data(), host.plan, error) !=
          XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  const std::size_t atom_count = static_cast<std::size_t>(host.plan.total_atoms());
  if (positions.size() != atom_count * 3u || coordination.size() != atom_count ||
      charges.size() != atom_count || dipoles.size() != atom_count * 3u ||
      quadrupoles.size() != atom_count * 6u) {
    return false;
  }
  host.atomic_numbers = atomic_numbers;
  host.positions = positions;
  host.coordination = coordination;
  host.charges = charges;
  host.dipoles = dipoles;
  host.quadrupoles = quadrupoles;
  host.pair_data.resize(static_cast<std::size_t>(host.plan.pair_data_elements()));
  host.charge_potentials.resize(atom_count);
  host.dipole_potentials.resize(atom_count * 3u);
  host.quadrupole_potentials.resize(atom_count * 6u);
  host.energies.assign(static_cast<std::size_t>(batch_size), 0.0);
  host.gradients.assign(atom_count * 3u, 0.0);
  host.coordination_adjoints.assign(atom_count, 0.0);
  host.pair_scratch.resize(host.pair_data.size());
  host.potential_scratch.resize(static_cast<std::size_t>(host.plan.potential_scratch_elements()));
  host.batch_scratch.resize(static_cast<std::size_t>(batch_size));
  host.gradient_scratch.resize(atom_count * 3u);
  host.coordination_scratch.resize(atom_count);
  host.workspace = {host.pair_scratch.data(),         host.plan.pair_data_elements(),
                    host.potential_scratch.data(),    host.plan.potential_scratch_elements(),
                    host.batch_scratch.data(),        host.plan.batch_size(),
                    host.gradient_scratch.data(),     host.plan.gradient_scratch_elements(),
                    host.coordination_scratch.data(), host.plan.coordination_scratch_elements()};
  return true;
}

bool evaluate_cpu(HostEvaluation& host, std::uint64_t generation, std::string& error) {
  return xtbloom::detail::gfn2::update_aes2_geometry_cache_cpu(
             host.plan, host.positions.data(), host.coordination.data(), generation,
             host.pair_data.data(), host.pair_data.size(), host.workspace, host.cache,
             error) == XTBLOOM_STATUS_SUCCESS &&
         xtbloom::detail::gfn2::evaluate_aes2_potential_cpu(
             host.plan, host.cache, host.charges.data(), host.dipoles.data(),
             host.quadrupoles.data(), host.charge_potentials.data(), host.dipole_potentials.data(),
             host.quadrupole_potentials.data(), host.workspace, error) == XTBLOOM_STATUS_SUCCESS &&
         xtbloom::detail::gfn2::add_aes2_energy_cpu(host.plan, host.cache, host.charges.data(),
                                                    host.dipoles.data(), host.quadrupoles.data(),
                                                    host.energies.data(), host.workspace,
                                                    error) == XTBLOOM_STATUS_SUCCESS &&
         xtbloom::detail::gfn2::add_aes2_vjp_cpu(
             host.plan, host.cache, host.positions.data(), host.coordination.data(), generation,
             host.charges.data(), host.dipoles.data(), host.quadrupoles.data(),
             host.gradients.data(), host.coordination_adjoints.data(), host.workspace,
             error) == XTBLOOM_STATUS_SUCCESS;
}

HostEvaluation make_ragged_case() {
  HostEvaluation host;
  std::string error;
  const std::vector<std::int64_t> offsets{0, 3, 5};
  const std::vector<std::int32_t> atomic_numbers{8, 1, 1, 6, 8};
  const std::vector<double> positions{0.0,  0.0,  0.0,  1.43, 1.08, 0.05, -1.37, 1.12,
                                      -0.1, 0.25, -0.4, 0.1,  2.35, 0.2,  -0.3};
  const std::vector<double> coordination{1.82, 0.91, 0.94, 1.24, 1.37};
  const std::vector<double> charges{-0.42, 0.22, 0.20, 0.18, -0.18};
  const std::vector<double> dipoles{0.03,   -0.02, 0.01,  -0.015, 0.012, 0.004, 0.011, 0.008,
                                    -0.006, 0.024, -0.01, 0.007,  -0.02, 0.016, 0.005};
  std::vector<double> quadrupoles(30u);
  for (std::size_t index = 0; index < quadrupoles.size(); ++index) {
    quadrupoles[index] = 0.0025 * static_cast<double>(static_cast<int>(index % 9u) - 4);
  }
  if (!make_host_evaluation(offsets, atomic_numbers, positions, coordination, charges, dipoles,
                            quadrupoles, host, error)) {
    return {};
  }
  host.energies = {0.125, -0.25};
  for (std::size_t index = 0; index < host.gradients.size(); ++index) {
    host.gradients[index] = 0.001 * static_cast<double>(static_cast<int>(index % 5u) - 2);
  }
  for (std::size_t atom = 0; atom < host.coordination_adjoints.size(); ++atom) {
    host.coordination_adjoints[atom] = 0.002 * static_cast<double>(atom + 1u);
  }
  return host;
}

bool make_repeated_case(std::size_t batch_size, HostEvaluation& host, std::string& error) {
  std::vector<std::int64_t> offsets(batch_size + 1u);
  std::vector<std::int32_t> atomic_numbers(batch_size * 2u);
  std::vector<double> positions(batch_size * 6u);
  std::vector<double> coordination(batch_size * 2u);
  std::vector<double> charges(batch_size * 2u);
  std::vector<double> dipoles(batch_size * 6u);
  std::vector<double> quadrupoles(batch_size * 12u);
  for (std::size_t system = 0; system < batch_size; ++system) {
    offsets[system] = static_cast<std::int64_t>(system * 2u);
    const std::size_t first = system * 2u;
    const std::size_t second = first + 1u;
    atomic_numbers[first] = system % 2u == 0u ? 6 : 8;
    atomic_numbers[second] = 1;
    const double shift = 4.0 * static_cast<double>(system);
    positions[first * 3u] = shift;
    positions[first * 3u + 1u] = 0.1;
    positions[first * 3u + 2u] = -0.2;
    positions[second * 3u] = shift + 1.6 + 0.001 * static_cast<double>(system);
    positions[second * 3u + 1u] = 0.8;
    positions[second * 3u + 2u] = 0.3;
    coordination[first] = 1.1 + 0.002 * static_cast<double>(system);
    coordination[second] = 0.8 + 0.001 * static_cast<double>(system);
    charges[first] = -0.1 - 0.0002 * static_cast<double>(system);
    charges[second] = -charges[first];
    for (std::size_t component = 0; component < 6u; ++component) {
      dipoles[system * 6u + component] =
          0.004 * static_cast<double>(static_cast<int>((system + component) % 7u) - 3);
    }
    for (std::size_t component = 0; component < 12u; ++component) {
      quadrupoles[system * 12u + component] =
          0.001 * static_cast<double>(static_cast<int>((2u * system + component) % 11u) - 5);
    }
  }
  offsets[batch_size] = static_cast<std::int64_t>(batch_size * 2u);
  return make_host_evaluation(offsets, atomic_numbers, positions, coordination, charges, dipoles,
                              quadrupoles, host, error);
}

struct DeviceEvaluation {
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> pair_offsets;
  DeviceBuffer<double> dipole_kernel;
  DeviceBuffer<double> quadrupole_kernel;
  DeviceBuffer<double> multipole_radius;
  DeviceBuffer<double> multipole_valence_cn;
  DeviceBuffer<double> positions;
  DeviceBuffer<double> coordination;
  DeviceBuffer<double> charges;
  DeviceBuffer<double> dipoles;
  DeviceBuffer<double> quadrupoles;
  DeviceBuffer<double> pair_data;
  DeviceBuffer<double> charge_potentials;
  DeviceBuffer<double> dipole_potentials;
  DeviceBuffer<double> quadrupole_potentials;
  DeviceBuffer<double> energies;
  DeviceBuffer<double> gradients;
  DeviceBuffer<double> coordination_adjoints;
  DeviceBuffer<double> pair_scratch;
  DeviceBuffer<double> potential_scratch;
  DeviceBuffer<double> batch_scratch;
  DeviceBuffer<double> gradient_scratch;
  DeviceBuffer<double> coordination_scratch;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;
  Gfn2AES2DeviceBatch batch;
  Gfn2AES2DeviceCache cache;
  Gfn2AES2DeviceWorkspace workspace;

  bool initialize(const HostEvaluation& host, cudaStream_t stream) {
    const std::size_t atom_count = static_cast<std::size_t>(host.plan.total_atoms());
    const std::size_t batch_count = static_cast<std::size_t>(host.plan.batch_size());
    if (!atom_offsets.allocate(host.plan.atom_offsets().size()) ||
        !pair_offsets.allocate(host.plan.pair_offsets().size()) ||
        !dipole_kernel.allocate(atom_count) || !quadrupole_kernel.allocate(atom_count) ||
        !multipole_radius.allocate(atom_count) || !multipole_valence_cn.allocate(atom_count) ||
        !positions.allocate(atom_count * 3u) || !coordination.allocate(atom_count) ||
        !charges.allocate(atom_count) || !dipoles.allocate(atom_count * 3u) ||
        !quadrupoles.allocate(atom_count * 6u) || !pair_data.allocate(host.pair_data.size()) ||
        !charge_potentials.allocate(atom_count) || !dipole_potentials.allocate(atom_count * 3u) ||
        !quadrupole_potentials.allocate(atom_count * 6u) || !energies.allocate(batch_count) ||
        !gradients.allocate(atom_count * 3u) || !coordination_adjoints.allocate(atom_count) ||
        !pair_scratch.allocate(host.pair_data.size()) ||
        !potential_scratch.allocate(
            static_cast<std::size_t>(host.plan.potential_scratch_elements())) ||
        !batch_scratch.allocate(batch_count) || !gradient_scratch.allocate(atom_count * 3u) ||
        !coordination_scratch.allocate(atom_count) || !system_errors.allocate(batch_count) ||
        !device_error.allocate(1u)) {
      return false;
    }
    if (!atom_offsets.copy_from(host.plan.atom_offsets().data(), host.plan.atom_offsets().size(),
                                stream) ||
        !pair_offsets.copy_from(host.plan.pair_offsets().data(), host.plan.pair_offsets().size(),
                                stream) ||
        !dipole_kernel.copy_from(host.plan.dipole_kernel().data(), atom_count, stream) ||
        !quadrupole_kernel.copy_from(host.plan.quadrupole_kernel().data(), atom_count, stream) ||
        !multipole_radius.copy_from(host.plan.multipole_radius().data(), atom_count, stream) ||
        !multipole_valence_cn.copy_from(host.plan.multipole_valence_cn().data(), atom_count,
                                        stream) ||
        !positions.copy_from(host.positions.data(), host.positions.size(), stream) ||
        !coordination.copy_from(host.coordination.data(), host.coordination.size(), stream) ||
        !charges.copy_from(host.charges.data(), host.charges.size(), stream) ||
        !dipoles.copy_from(host.dipoles.data(), host.dipoles.size(), stream) ||
        !quadrupoles.copy_from(host.quadrupoles.data(), host.quadrupoles.size(), stream) ||
        !energies.copy_from(host.energies.data(), host.energies.size(), stream) ||
        !gradients.copy_from(host.gradients.data(), host.gradients.size(), stream) ||
        !coordination_adjoints.copy_from(host.coordination_adjoints.data(),
                                         host.coordination_adjoints.size(), stream)) {
      return false;
    }
    batch = {host.plan.batch_size(),
             host.plan.total_atoms(),
             host.plan.total_pairs(),
             kPlanToken,
             static_cast<std::int64_t>(host.plan.atom_offsets().size()),
             static_cast<std::int64_t>(host.plan.pair_offsets().size()),
             host.plan.total_atoms(),
             host.plan.total_atoms(),
             host.plan.total_atoms(),
             host.plan.total_atoms(),
             atom_offsets.get(),
             pair_offsets.get(),
             dipole_kernel.get(),
             quadrupole_kernel.get(),
             multipole_radius.get(),
             multipole_valence_cn.get()};
    cache = {pair_data.get(), host.plan.pair_data_elements(), kGeneration, kPlanToken};
    workspace = {pair_scratch.get(),         host.plan.pair_data_elements(),
                 potential_scratch.get(),    host.plan.potential_scratch_elements(),
                 batch_scratch.get(),        host.plan.batch_size(),
                 gradient_scratch.get(),     host.plan.gradient_scratch_elements(),
                 coordination_scratch.get(), host.plan.coordination_scratch_elements()};
    return true;
  }
};

bool enqueue_full(DeviceEvaluation& device, cudaStream_t stream) {
  return xtbloom::detail::cuda::reset_gfn2_aes2_device_errors_cuda(
             device.batch.batch_size, device.system_errors.get(), device.device_error.get(),
             stream) == cudaSuccess &&
         xtbloom::detail::cuda::update_gfn2_aes2_geometry_cache_cuda(
             device.batch, device.positions.get(), device.coordination.get(), device.cache,
             device.workspace, device.system_errors.get(), device.device_error.get(),
             stream) == cudaSuccess &&
         xtbloom::detail::cuda::evaluate_gfn2_aes2_potential_cuda(
             device.batch, device.cache, device.charges.get(), device.dipoles.get(),
             device.quadrupoles.get(), device.charge_potentials.get(),
             device.dipole_potentials.get(), device.quadrupole_potentials.get(), device.workspace,
             device.system_errors.get(), device.device_error.get(), stream) == cudaSuccess &&
         xtbloom::detail::cuda::add_gfn2_aes2_energy_cuda(
             device.batch, device.cache, device.charges.get(), device.dipoles.get(),
             device.quadrupoles.get(), device.energies.get(), device.workspace,
             device.system_errors.get(), device.device_error.get(), stream) == cudaSuccess &&
         xtbloom::detail::cuda::add_gfn2_aes2_vjp_cuda(
             device.batch, device.cache, device.positions.get(), device.coordination.get(),
             kGeneration, device.charges.get(), device.dipoles.get(), device.quadrupoles.get(),
             device.gradients.get(), device.coordination_adjoints.get(), device.workspace,
             device.system_errors.get(), device.device_error.get(), stream) == cudaSuccess;
}

bool upload_accumulation_seeds(DeviceEvaluation& device, const std::vector<double>& energies,
                               const std::vector<double>& gradients,
                               const std::vector<double>& coordination_adjoints,
                               cudaStream_t stream) {
  return device.energies.copy_from(energies.data(), energies.size(), stream) &&
         device.gradients.copy_from(gradients.data(), gradients.size(), stream) &&
         device.coordination_adjoints.copy_from(coordination_adjoints.data(),
                                                coordination_adjoints.size(), stream);
}

int compare_device(const HostEvaluation& expected, const DeviceEvaluation& device,
                   cudaStream_t stream, double tolerance = 2.0e-12) {
  std::vector<double> pair_data(expected.pair_data.size());
  std::vector<double> charge_potentials(expected.charge_potentials.size());
  std::vector<double> dipole_potentials(expected.dipole_potentials.size());
  std::vector<double> quadrupole_potentials(expected.quadrupole_potentials.size());
  std::vector<double> energies(expected.energies.size());
  std::vector<double> gradients(expected.gradients.size());
  std::vector<double> coordination_adjoints(expected.coordination_adjoints.size());
  std::vector<std::uint32_t> system_errors(expected.energies.size(), 99u);
  std::uint32_t device_error = 99u;
  CHECK(device.pair_data.copy_to(pair_data.data(), pair_data.size(), stream));
  CHECK(
      device.charge_potentials.copy_to(charge_potentials.data(), charge_potentials.size(), stream));
  CHECK(
      device.dipole_potentials.copy_to(dipole_potentials.data(), dipole_potentials.size(), stream));
  CHECK(device.quadrupole_potentials.copy_to(quadrupole_potentials.data(),
                                             quadrupole_potentials.size(), stream));
  CHECK(device.energies.copy_to(energies.data(), energies.size(), stream));
  CHECK(device.gradients.copy_to(gradients.data(), gradients.size(), stream));
  CHECK(device.coordination_adjoints.copy_to(coordination_adjoints.data(),
                                             coordination_adjoints.size(), stream));
  CHECK(device.system_errors.copy_to(system_errors.data(), system_errors.size(), stream));
  CHECK(device.device_error.copy_to(&device_error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(device_error == 0u);
  CHECK(std::all_of(system_errors.begin(), system_errors.end(),
                    [](std::uint32_t error) { return error == 0u; }));
  for (std::size_t index = 0; index < pair_data.size(); ++index) {
    CHECK(near(pair_data[index], expected.pair_data[index], 8.0e-13));
  }
  for (std::size_t index = 0; index < charge_potentials.size(); ++index) {
    CHECK(near(charge_potentials[index], expected.charge_potentials[index], tolerance));
  }
  for (std::size_t index = 0; index < dipole_potentials.size(); ++index) {
    CHECK(near(dipole_potentials[index], expected.dipole_potentials[index], tolerance));
  }
  for (std::size_t index = 0; index < quadrupole_potentials.size(); ++index) {
    CHECK(near(quadrupole_potentials[index], expected.quadrupole_potentials[index], tolerance));
  }
  for (std::size_t index = 0; index < energies.size(); ++index) {
    CHECK(near(energies[index], expected.energies[index], tolerance));
  }
  for (std::size_t index = 0; index < gradients.size(); ++index) {
    CHECK(near(gradients[index], expected.gradients[index], 8.0 * tolerance));
  }
  for (std::size_t index = 0; index < coordination_adjoints.size(); ++index) {
    CHECK(
        near(coordination_adjoints[index], expected.coordination_adjoints[index], 8.0 * tolerance));
  }
  return 0;
}

int test_ragged_cpu_parity_and_custom_stream() {
  HostEvaluation host = make_ragged_case();
  std::string error;
  CHECK(host.plan.sealed());
  const std::vector<double> energy_seeds = host.energies;
  const std::vector<double> gradient_seeds = host.gradients;
  const std::vector<double> coordination_seeds = host.coordination_adjoints;
  CHECK(evaluate_cpu(host, kGeneration, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceEvaluation device;
  CHECK(device.initialize(host, stream));
  CHECK(
      upload_accumulation_seeds(device, energy_seeds, gradient_seeds, coordination_seeds, stream));
  CHECK(enqueue_full(device, stream));
  if (const int line = compare_device(host, device, stream); line != 0) {
    return line;
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_batch_sizes() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (const std::size_t batch_size : {1u, 8u, 32u, 128u}) {
    HostEvaluation host;
    std::string error;
    CHECK(make_repeated_case(batch_size, host, error));
    const std::vector<double> energy_seeds = host.energies;
    const std::vector<double> gradient_seeds = host.gradients;
    const std::vector<double> coordination_seeds = host.coordination_adjoints;
    CHECK(evaluate_cpu(host, kGeneration, error));
    DeviceEvaluation device;
    CHECK(device.initialize(host, stream));
    CHECK(upload_accumulation_seeds(device, energy_seeds, gradient_seeds, coordination_seeds,
                                    stream));
    CHECK(enqueue_full(device, stream));
    if (const int line = compare_device(host, device, stream, 4.0e-12); line != 0) {
      return line;
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_geometry_and_compute_failure_isolation() {
  HostEvaluation baseline = make_ragged_case();
  std::string error;
  CHECK(baseline.plan.sealed());
  CHECK(evaluate_cpu(baseline, kGeneration, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  DeviceEvaluation geometry_device;
  CHECK(geometry_device.initialize(baseline, stream));
  std::vector<double> pair_sentinel(baseline.pair_data.size(), 73.0);
  std::vector<double> bad_coordination = baseline.coordination;
  bad_coordination[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK(geometry_device.pair_data.copy_from(pair_sentinel.data(), pair_sentinel.size(), stream));
  CHECK(geometry_device.coordination.copy_from(bad_coordination.data(), bad_coordination.size(),
                                               stream));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_aes2_device_errors_cuda(
      geometry_device.batch.batch_size, geometry_device.system_errors.get(),
      geometry_device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_aes2_geometry_cache_cuda(
      geometry_device.batch, geometry_device.positions.get(), geometry_device.coordination.get(),
      geometry_device.cache, geometry_device.workspace, geometry_device.system_errors.get(),
      geometry_device.device_error.get(), stream));
  std::vector<double> isolated_pair_data(baseline.pair_data.size());
  std::vector<std::uint32_t> geometry_errors(2u, 99u);
  std::uint32_t global_error = 99u;
  CHECK(geometry_device.pair_data.copy_to(isolated_pair_data.data(), isolated_pair_data.size(),
                                          stream));
  CHECK(geometry_device.system_errors.copy_to(geometry_errors.data(), geometry_errors.size(),
                                              stream));
  CHECK(geometry_device.device_error.copy_to(&global_error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(geometry_errors[0] ==
        static_cast<std::uint32_t>(Gfn2AES2DeviceError::kInvalidCoordination));
  CHECK(geometry_errors[1] == 0u);
  CHECK(global_error == geometry_errors[0]);
  const std::int64_t first_pair_end = baseline.plan.pair_offsets()[1];
  for (std::int64_t pair = 0; pair < first_pair_end; ++pair) {
    for (std::int64_t component = 0; component < 5; ++component) {
      CHECK(isolated_pair_data[static_cast<std::size_t>(pair * 5 + component)] == 73.0);
    }
  }
  for (std::int64_t pair = baseline.plan.pair_offsets()[1]; pair < baseline.plan.total_pairs();
       ++pair) {
    for (std::int64_t component = 0; component < 5; ++component) {
      const std::size_t index = static_cast<std::size_t>(pair * 5 + component);
      CHECK(near(isolated_pair_data[index], baseline.pair_data[index], 8.0e-13));
    }
  }

  HostEvaluation expected = make_ragged_case();
  CHECK(expected.plan.sealed());
  std::fill(expected.energies.begin(), expected.energies.end(), 19.0);
  std::fill(expected.gradients.begin(), expected.gradients.end(), 23.0);
  std::fill(expected.coordination_adjoints.begin(), expected.coordination_adjoints.end(), 29.0);
  CHECK(evaluate_cpu(expected, kGeneration, error));
  DeviceEvaluation compute_device;
  CHECK(compute_device.initialize(expected, stream));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_aes2_device_errors_cuda(
      compute_device.batch.batch_size, compute_device.system_errors.get(),
      compute_device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_aes2_geometry_cache_cuda(
      compute_device.batch, compute_device.positions.get(), compute_device.coordination.get(),
      compute_device.cache, compute_device.workspace, compute_device.system_errors.get(),
      compute_device.device_error.get(), stream));
  std::vector<double> bad_charges = expected.charges;
  bad_charges[0] = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> charge_sentinel(expected.charges.size(), 31.0);
  std::vector<double> dipole_sentinel(expected.dipoles.size(), 37.0);
  std::vector<double> quadrupole_sentinel(expected.quadrupoles.size(), 41.0);
  std::vector<double> energy_sentinel(expected.energies.size(), 19.0);
  std::vector<double> gradient_sentinel(expected.gradients.size(), 23.0);
  std::vector<double> coordination_sentinel(expected.coordination_adjoints.size(), 29.0);
  CHECK(compute_device.charges.copy_from(bad_charges.data(), bad_charges.size(), stream));
  CHECK(compute_device.charge_potentials.copy_from(charge_sentinel.data(), charge_sentinel.size(),
                                                   stream));
  CHECK(compute_device.dipole_potentials.copy_from(dipole_sentinel.data(), dipole_sentinel.size(),
                                                   stream));
  CHECK(compute_device.quadrupole_potentials.copy_from(quadrupole_sentinel.data(),
                                                       quadrupole_sentinel.size(), stream));
  CHECK(compute_device.energies.copy_from(energy_sentinel.data(), energy_sentinel.size(), stream));
  CHECK(compute_device.gradients.copy_from(gradient_sentinel.data(), gradient_sentinel.size(),
                                           stream));
  CHECK(compute_device.coordination_adjoints.copy_from(coordination_sentinel.data(),
                                                       coordination_sentinel.size(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_aes2_potential_cuda(
      compute_device.batch, compute_device.cache, compute_device.charges.get(),
      compute_device.dipoles.get(), compute_device.quadrupoles.get(),
      compute_device.charge_potentials.get(), compute_device.dipole_potentials.get(),
      compute_device.quadrupole_potentials.get(), compute_device.workspace,
      compute_device.system_errors.get(), compute_device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_aes2_energy_cuda(
      compute_device.batch, compute_device.cache, compute_device.charges.get(),
      compute_device.dipoles.get(), compute_device.quadrupoles.get(), compute_device.energies.get(),
      compute_device.workspace, compute_device.system_errors.get(),
      compute_device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_aes2_vjp_cuda(
      compute_device.batch, compute_device.cache, compute_device.positions.get(),
      compute_device.coordination.get(), kGeneration, compute_device.charges.get(),
      compute_device.dipoles.get(), compute_device.quadrupoles.get(),
      compute_device.gradients.get(), compute_device.coordination_adjoints.get(),
      compute_device.workspace, compute_device.system_errors.get(),
      compute_device.device_error.get(), stream));

  std::vector<double> actual_charge(charge_sentinel.size());
  std::vector<double> actual_dipole(dipole_sentinel.size());
  std::vector<double> actual_quadrupole(quadrupole_sentinel.size());
  std::vector<double> actual_energy(energy_sentinel.size());
  std::vector<double> actual_gradient(gradient_sentinel.size());
  std::vector<double> actual_coordination(coordination_sentinel.size());
  std::vector<std::uint32_t> compute_errors(2u, 99u);
  CHECK(
      compute_device.charge_potentials.copy_to(actual_charge.data(), actual_charge.size(), stream));
  CHECK(
      compute_device.dipole_potentials.copy_to(actual_dipole.data(), actual_dipole.size(), stream));
  CHECK(compute_device.quadrupole_potentials.copy_to(actual_quadrupole.data(),
                                                     actual_quadrupole.size(), stream));
  CHECK(compute_device.energies.copy_to(actual_energy.data(), actual_energy.size(), stream));
  CHECK(compute_device.gradients.copy_to(actual_gradient.data(), actual_gradient.size(), stream));
  CHECK(compute_device.coordination_adjoints.copy_to(actual_coordination.data(),
                                                     actual_coordination.size(), stream));
  CHECK(compute_device.system_errors.copy_to(compute_errors.data(), compute_errors.size(), stream));
  CHECK(compute_device.device_error.copy_to(&global_error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(compute_errors[0] == static_cast<std::uint32_t>(Gfn2AES2DeviceError::kNonfiniteMultipole));
  CHECK(compute_errors[1] == 0u);
  CHECK(global_error == compute_errors[0]);
  const std::int64_t failed_atoms = expected.plan.atom_offsets()[1];
  for (std::int64_t atom = 0; atom < failed_atoms; ++atom) {
    CHECK(actual_charge[static_cast<std::size_t>(atom)] == 31.0);
    CHECK(actual_coordination[static_cast<std::size_t>(atom)] == 29.0);
    for (std::int64_t component = 0; component < 3; ++component) {
      CHECK(actual_dipole[static_cast<std::size_t>(atom * 3 + component)] == 37.0);
      CHECK(actual_gradient[static_cast<std::size_t>(atom * 3 + component)] == 23.0);
    }
    for (std::int64_t component = 0; component < 6; ++component) {
      CHECK(actual_quadrupole[static_cast<std::size_t>(atom * 6 + component)] == 41.0);
    }
  }
  CHECK(actual_energy[0] == 19.0);
  for (std::int64_t atom = expected.plan.atom_offsets()[1]; atom < expected.plan.total_atoms();
       ++atom) {
    const std::size_t atom_index = static_cast<std::size_t>(atom);
    CHECK(near(actual_charge[atom_index], expected.charge_potentials[atom_index]));
    CHECK(
        near(actual_coordination[atom_index], expected.coordination_adjoints[atom_index], 1.0e-11));
    for (std::int64_t component = 0; component < 3; ++component) {
      const std::size_t index = static_cast<std::size_t>(atom * 3 + component);
      CHECK(near(actual_dipole[index], expected.dipole_potentials[index]));
      CHECK(near(actual_gradient[index], expected.gradients[index], 1.0e-11));
    }
    for (std::int64_t component = 0; component < 6; ++component) {
      const std::size_t index = static_cast<std::size_t>(atom * 6 + component);
      CHECK(near(actual_quadrupole[index], expected.quadrupole_potentials[index]));
    }
  }
  CHECK(near(actual_energy[1], expected.energies[1]));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int run_hostile_offset_case(bool corrupt_atom_offsets, cudaStream_t stream) {
  HostEvaluation host;
  std::string error;
  CHECK(make_repeated_case(1u, host, error));
  DeviceEvaluation device;
  CHECK(device.initialize(host, stream));

  std::vector<std::int64_t> atom_offsets = host.plan.atom_offsets();
  std::vector<std::int64_t> pair_offsets = host.plan.pair_offsets();
  std::vector<std::int64_t>& hostile_offsets = corrupt_atom_offsets ? atom_offsets : pair_offsets;
  hostile_offsets[0] = std::numeric_limits<std::int64_t>::min();
  hostile_offsets[1] = std::numeric_limits<std::int64_t>::max();
  CHECK(device.atom_offsets.copy_from(atom_offsets.data(), atom_offsets.size(), stream));
  CHECK(device.pair_offsets.copy_from(pair_offsets.data(), pair_offsets.size(), stream));

  /* Every published result must retain its seed when topology preflight fails. */
  std::vector<double> pair_sentinel(host.pair_data.size(), 101.0);
  std::vector<double> charge_sentinel(host.charges.size(), 103.0);
  std::vector<double> dipole_sentinel(host.dipoles.size(), 107.0);
  std::vector<double> quadrupole_sentinel(host.quadrupoles.size(), 109.0);
  std::vector<double> energy_sentinel(host.energies.size(), 113.0);
  std::vector<double> gradient_sentinel(host.gradients.size(), 127.0);
  std::vector<double> coordination_sentinel(host.coordination_adjoints.size(), 131.0);
  CHECK(device.pair_data.copy_from(pair_sentinel.data(), pair_sentinel.size(), stream));
  CHECK(device.charge_potentials.copy_from(charge_sentinel.data(), charge_sentinel.size(), stream));
  CHECK(device.dipole_potentials.copy_from(dipole_sentinel.data(), dipole_sentinel.size(), stream));
  CHECK(device.quadrupole_potentials.copy_from(quadrupole_sentinel.data(),
                                               quadrupole_sentinel.size(), stream));
  CHECK(device.energies.copy_from(energy_sentinel.data(), energy_sentinel.size(), stream));
  CHECK(device.gradients.copy_from(gradient_sentinel.data(), gradient_sentinel.size(), stream));
  CHECK(device.coordination_adjoints.copy_from(coordination_sentinel.data(),
                                               coordination_sentinel.size(), stream));

  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_aes2_device_errors_cuda(
      device.batch.batch_size, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_aes2_geometry_cache_cuda(
      device.batch, device.positions.get(), device.coordination.get(), device.cache,
      device.workspace, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_aes2_potential_cuda(
      device.batch, device.cache, device.charges.get(), device.dipoles.get(),
      device.quadrupoles.get(), device.charge_potentials.get(), device.dipole_potentials.get(),
      device.quadrupole_potentials.get(), device.workspace, device.system_errors.get(),
      device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_aes2_energy_cuda(
      device.batch, device.cache, device.charges.get(), device.dipoles.get(),
      device.quadrupoles.get(), device.energies.get(), device.workspace, device.system_errors.get(),
      device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_aes2_vjp_cuda(
      device.batch, device.cache, device.positions.get(), device.coordination.get(), kGeneration,
      device.charges.get(), device.dipoles.get(), device.quadrupoles.get(), device.gradients.get(),
      device.coordination_adjoints.get(), device.workspace, device.system_errors.get(),
      device.device_error.get(), stream));

  std::vector<double> actual_pair(pair_sentinel.size());
  std::vector<double> actual_charge(charge_sentinel.size());
  std::vector<double> actual_dipole(dipole_sentinel.size());
  std::vector<double> actual_quadrupole(quadrupole_sentinel.size());
  std::vector<double> actual_energy(energy_sentinel.size());
  std::vector<double> actual_gradient(gradient_sentinel.size());
  std::vector<double> actual_coordination(coordination_sentinel.size());
  std::uint32_t system_error = 99u;
  std::uint32_t device_error = 99u;
  CHECK(device.pair_data.copy_to(actual_pair.data(), actual_pair.size(), stream));
  CHECK(device.charge_potentials.copy_to(actual_charge.data(), actual_charge.size(), stream));
  CHECK(device.dipole_potentials.copy_to(actual_dipole.data(), actual_dipole.size(), stream));
  CHECK(device.quadrupole_potentials.copy_to(actual_quadrupole.data(), actual_quadrupole.size(),
                                             stream));
  CHECK(device.energies.copy_to(actual_energy.data(), actual_energy.size(), stream));
  CHECK(device.gradients.copy_to(actual_gradient.data(), actual_gradient.size(), stream));
  CHECK(device.coordination_adjoints.copy_to(actual_coordination.data(), actual_coordination.size(),
                                             stream));
  CHECK(device.system_errors.copy_to(&system_error, 1u, stream));
  CHECK(device.device_error.copy_to(&device_error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const auto unchanged = [](const std::vector<double>& actual,
                            const std::vector<double>& sentinel) { return actual == sentinel; };
  CHECK(system_error == static_cast<std::uint32_t>(Gfn2AES2DeviceError::kInvalidOffsets));
  CHECK(device_error == system_error);
  CHECK(unchanged(actual_pair, pair_sentinel));
  CHECK(unchanged(actual_charge, charge_sentinel));
  CHECK(unchanged(actual_dipole, dipole_sentinel));
  CHECK(unchanged(actual_quadrupole, quadrupole_sentinel));
  CHECK(unchanged(actual_energy, energy_sentinel));
  CHECK(unchanged(actual_gradient, gradient_sentinel));
  CHECK(unchanged(actual_coordination, coordination_sentinel));
  return 0;
}

int test_hostile_offsets_fail_closed() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  if (const int line = run_hostile_offset_case(true, stream); line != 0) {
    return line;
  }
  if (const int line = run_hostile_offset_case(false, stream); line != 0) {
    return line;
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_graph_capture_and_structural_guards() {
  HostEvaluation host = make_ragged_case();
  std::string error;
  CHECK(host.plan.sealed());
  std::fill(host.energies.begin(), host.energies.end(), 0.0);
  std::fill(host.gradients.begin(), host.gradients.end(), 0.0);
  std::fill(host.coordination_adjoints.begin(), host.coordination_adjoints.end(), 0.0);
  CHECK(evaluate_cpu(host, kGeneration, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceEvaluation device;
  CHECK(device.initialize(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  Gfn2AES2DeviceCache stale_cache = device.cache;
  stale_cache.plan_token ^= 1u;
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_aes2_potential_cuda(
            device.batch, stale_cache, device.charges.get(), device.dipoles.get(),
            device.quadrupoles.get(), device.charge_potentials.get(),
            device.dipole_potentials.get(), device.quadrupole_potentials.get(), device.workspace,
            device.system_errors.get(), device.device_error.get(),
            stream) == cudaErrorInvalidValue);
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_aes2_potential_cuda(
            device.batch, device.cache, device.charges.get(), device.dipoles.get(),
            device.quadrupoles.get(), device.charges.get(), device.dipole_potentials.get(),
            device.quadrupole_potentials.get(), device.workspace, device.system_errors.get(),
            device.device_error.get(), stream) == cudaErrorInvalidValue);
  CHECK(xtbloom::detail::cuda::add_gfn2_aes2_vjp_cuda(
            device.batch, device.cache, device.positions.get(), device.coordination.get(),
            kGeneration + 1u, device.charges.get(), device.dipoles.get(), device.quadrupoles.get(),
            device.gradients.get(), device.coordination_adjoints.get(), device.workspace,
            device.system_errors.get(), device.device_error.get(),
            stream) == cudaErrorInvalidValue);

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(
      cudaMemsetAsync(device.energies.get(), 0, host.energies.size() * sizeof(double), stream));
  CUDA_CHECK(
      cudaMemsetAsync(device.gradients.get(), 0, host.gradients.size() * sizeof(double), stream));
  CUDA_CHECK(cudaMemsetAsync(device.coordination_adjoints.get(), 0,
                             host.coordination_adjoints.size() * sizeof(double), stream));
  CHECK(enqueue_full(device, stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CHECK(graph != nullptr);
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0u));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  if (const int line = compare_device(host, device, stream); line != 0) {
    return line;
  }
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

#ifdef XTBLOOM_AES2_TERM_BENCHMARK_ONLY

using BenchmarkOptions = xtbloom::test::cuda_term_benchmark::Options;
using BenchmarkRow = xtbloom::test::cuda_term_benchmark::Row;
using BenchmarkSamples = xtbloom::test::cuda_term_benchmark::Samples;

bool make_aes2_benchmark_case(const BenchmarkOptions& options, HostEvaluation* host,
                              std::string* error) {
  if (host == nullptr || error == nullptr || options.batch_size <= 0 ||
      options.atoms_per_system < 2 ||
      options.batch_size > std::numeric_limits<std::int64_t>::max() / options.atoms_per_system ||
      options.batch_size * options.atoms_per_system >
          std::numeric_limits<std::int64_t>::max() / 6) {
    if (error != nullptr) *error = "AES2 benchmark requires at least two atoms per system";
    return false;
  }
  const std::int64_t total_atoms = options.batch_size * options.atoms_per_system;
  std::vector<std::int64_t> offsets(static_cast<std::size_t>(options.batch_size + 1));
  std::vector<std::int32_t> atomic_numbers(static_cast<std::size_t>(total_atoms));
  std::vector<double> positions(static_cast<std::size_t>(total_atoms * 3));
  std::vector<double> coordination(static_cast<std::size_t>(total_atoms));
  std::vector<double> charges(static_cast<std::size_t>(total_atoms));
  std::vector<double> dipoles(static_cast<std::size_t>(total_atoms * 3));
  std::vector<double> quadrupoles(static_cast<std::size_t>(total_atoms * 6));
  constexpr std::array<std::int32_t, 5> kElements{6, 1, 8, 7, 16};
  constexpr double kSpacing = 2.4;
  for (std::int64_t system = 0; system < options.batch_size; ++system) {
    offsets[static_cast<std::size_t>(system)] = system * options.atoms_per_system;
    const std::int64_t side = static_cast<std::int64_t>(
        std::ceil(std::cbrt(static_cast<double>(options.atoms_per_system))));
    for (std::int64_t local = 0; local < options.atoms_per_system; ++local) {
      const std::int64_t atom = system * options.atoms_per_system + local;
      atomic_numbers[static_cast<std::size_t>(atom)] =
          kElements[static_cast<std::size_t>((local + system) % kElements.size())];
      positions[static_cast<std::size_t>(atom * 3)] = kSpacing * static_cast<double>(local % side);
      positions[static_cast<std::size_t>(atom * 3 + 1)] =
          kSpacing * static_cast<double>((local / side) % side);
      positions[static_cast<std::size_t>(atom * 3 + 2)] =
          kSpacing * static_cast<double>(local / (side * side));
      coordination[static_cast<std::size_t>(atom)] = 0.8 + 0.03 * (local % 7);
      charges[static_cast<std::size_t>(atom)] =
          0.03 * static_cast<double>(static_cast<int>(local % 5) - 2);
      for (std::int64_t component = 0; component < 3; ++component) {
        dipoles[static_cast<std::size_t>(atom * 3 + component)] =
            0.002 * static_cast<double>(static_cast<int>((atom + component) % 7) - 3);
      }
      for (std::int64_t component = 0; component < 6; ++component) {
        quadrupoles[static_cast<std::size_t>(atom * 6 + component)] =
            0.0005 * static_cast<double>(static_cast<int>((atom + component) % 11) - 5);
      }
    }
  }
  offsets.back() = total_atoms;
  return make_host_evaluation(offsets, atomic_numbers, positions, coordination, charges, dipoles,
                              quadrupoles, *host, *error);
}

int benchmark_aes2_terms(int argc, char** argv) {
  BenchmarkOptions options;
  std::string error;
  if (!xtbloom::test::cuda_term_benchmark::parse_options(argc, argv, &options, &error)) {
    xtbloom::test::cuda_term_benchmark::print_usage(argv[0]);
    if (error != "help") std::cerr << error << '\n';
    return error == "help" ? 0 : 2;
  }
  HostEvaluation expected;
  if (!make_aes2_benchmark_case(options, &expected, &error)) {
    std::cerr << "failed to construct AES2 benchmark: " << error << '\n';
    return 1;
  }
  CHECK(evaluate_cpu(expected, kGeneration, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceEvaluation device;
  CHECK(device.initialize(expected, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const std::size_t batch_count = static_cast<std::size_t>(expected.plan.batch_size());
  const std::size_t atom_count = static_cast<std::size_t>(expected.plan.total_atoms());
  const auto reset_errors = [&]() {
    return xtbloom::detail::cuda::reset_gfn2_aes2_device_errors_cuda(
               device.batch.batch_size, device.system_errors.get(), device.device_error.get(),
               stream) == cudaSuccess;
  };
  const auto validate_errors = [&]() {
    std::vector<std::uint32_t> system_errors(batch_count, 99u);
    std::uint32_t device_error = 99u;
    if (!device.system_errors.copy_to(system_errors.data(), system_errors.size(), stream) ||
        !device.device_error.copy_to(&device_error, 1u, stream) ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return false;
    }
    return device_error == 0u && std::all_of(system_errors.begin(), system_errors.end(),
                                             [](std::uint32_t value) { return value == 0u; });
  };
  const auto near_vectors = [](const std::vector<double>& actual,
                               const std::vector<double>& reference, double tolerance) {
    if (actual.size() != reference.size()) return false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
      if (!near(actual[index], reference[index], tolerance)) return false;
    }
    return true;
  };
  const auto make_row = [&](const char* term, BenchmarkSamples timing) {
    return BenchmarkRow{term,
                        "compact_all_pairs",
                        options.batch_size,
                        options.atoms_per_system,
                        expected.plan.total_atoms(),
                        expected.plan.total_pairs(),
                        std::move(timing)};
  };
  std::vector<BenchmarkRow> rows;
  rows.reserve(5u);
  BenchmarkSamples timing;

  if (!xtbloom::test::cuda_term_benchmark::measure_term(
          options, "aes2_geometry", stream, reset_errors,
          [&]() {
            return xtbloom::detail::cuda::update_gfn2_aes2_geometry_cache_cuda(
                       device.batch, device.positions.get(), device.coordination.get(),
                       device.cache, device.workspace, device.system_errors.get(),
                       device.device_error.get(), stream) == cudaSuccess;
          },
          [&]() {
            std::vector<double> pair_data(expected.pair_data.size());
            return validate_errors() &&
                   device.pair_data.copy_to(pair_data.data(), pair_data.size(), stream) &&
                   cudaStreamSynchronize(stream) == cudaSuccess &&
                   near_vectors(pair_data, expected.pair_data, 8.0e-13);
          },
          &timing, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  rows.push_back(make_row("aes2_geometry", std::move(timing)));

  const auto prepare_potential = [&]() {
    return reset_errors() && device.charge_potentials.zero(stream) &&
           device.dipole_potentials.zero(stream) && device.quadrupole_potentials.zero(stream);
  };
  if (!xtbloom::test::cuda_term_benchmark::measure_term(
          options, "aes2_potential", stream, prepare_potential,
          [&]() {
            return xtbloom::detail::cuda::evaluate_gfn2_aes2_potential_cuda(
                       device.batch, device.cache, device.charges.get(), device.dipoles.get(),
                       device.quadrupoles.get(), device.charge_potentials.get(),
                       device.dipole_potentials.get(), device.quadrupole_potentials.get(),
                       device.workspace, device.system_errors.get(), device.device_error.get(),
                       stream) == cudaSuccess;
          },
          [&]() {
            std::vector<double> charge(atom_count);
            std::vector<double> dipole(atom_count * 3u);
            std::vector<double> quadrupole(atom_count * 6u);
            return validate_errors() &&
                   device.charge_potentials.copy_to(charge.data(), charge.size(), stream) &&
                   device.dipole_potentials.copy_to(dipole.data(), dipole.size(), stream) &&
                   device.quadrupole_potentials.copy_to(quadrupole.data(), quadrupole.size(),
                                                        stream) &&
                   cudaStreamSynchronize(stream) == cudaSuccess &&
                   near_vectors(charge, expected.charge_potentials, 2.0e-11) &&
                   near_vectors(dipole, expected.dipole_potentials, 2.0e-11) &&
                   near_vectors(quadrupole, expected.quadrupole_potentials, 2.0e-11);
          },
          &timing, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  rows.push_back(make_row("aes2_potential", std::move(timing)));

  const auto prepare_energy = [&]() { return reset_errors() && device.energies.zero(stream); };
  if (!xtbloom::test::cuda_term_benchmark::measure_term(
          options, "aes2_energy", stream, prepare_energy,
          [&]() {
            return xtbloom::detail::cuda::add_gfn2_aes2_energy_cuda(
                       device.batch, device.cache, device.charges.get(), device.dipoles.get(),
                       device.quadrupoles.get(), device.energies.get(), device.workspace,
                       device.system_errors.get(), device.device_error.get(),
                       stream) == cudaSuccess;
          },
          [&]() {
            std::vector<double> energies(batch_count);
            return validate_errors() &&
                   device.energies.copy_to(energies.data(), energies.size(), stream) &&
                   cudaStreamSynchronize(stream) == cudaSuccess &&
                   near_vectors(energies, expected.energies, 2.0e-11);
          },
          &timing, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  rows.push_back(make_row("aes2_energy", std::move(timing)));

  const auto prepare_vjp = [&]() {
    return reset_errors() && device.gradients.zero(stream) &&
           device.coordination_adjoints.zero(stream);
  };
  if (!xtbloom::test::cuda_term_benchmark::measure_term(
          options, "aes2_vjp", stream, prepare_vjp,
          [&]() {
            return xtbloom::detail::cuda::add_gfn2_aes2_vjp_cuda(
                       device.batch, device.cache, device.positions.get(),
                       device.coordination.get(), kGeneration, device.charges.get(),
                       device.dipoles.get(), device.quadrupoles.get(), device.gradients.get(),
                       device.coordination_adjoints.get(), device.workspace,
                       device.system_errors.get(), device.device_error.get(),
                       stream) == cudaSuccess;
          },
          [&]() {
            std::vector<double> gradients(atom_count * 3u);
            std::vector<double> coordination(atom_count);
            return validate_errors() &&
                   device.gradients.copy_to(gradients.data(), gradients.size(), stream) &&
                   device.coordination_adjoints.copy_to(coordination.data(), coordination.size(),
                                                        stream) &&
                   cudaStreamSynchronize(stream) == cudaSuccess &&
                   near_vectors(gradients, expected.gradients, 2.0e-10) &&
                   near_vectors(coordination, expected.coordination_adjoints, 2.0e-10);
          },
          &timing, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  rows.push_back(make_row("aes2_vjp", std::move(timing)));

  const auto prepare_full = [&]() {
    return reset_errors() && device.charge_potentials.zero(stream) &&
           device.dipole_potentials.zero(stream) && device.quadrupole_potentials.zero(stream) &&
           device.energies.zero(stream) && device.gradients.zero(stream) &&
           device.coordination_adjoints.zero(stream);
  };
  if (!xtbloom::test::cuda_term_benchmark::measure_term(
          options, "aes2_full", stream, prepare_full,
          [&]() {
            return xtbloom::detail::cuda::update_gfn2_aes2_geometry_cache_cuda(
                       device.batch, device.positions.get(), device.coordination.get(),
                       device.cache, device.workspace, device.system_errors.get(),
                       device.device_error.get(), stream) == cudaSuccess &&
                   xtbloom::detail::cuda::evaluate_gfn2_aes2_potential_cuda(
                       device.batch, device.cache, device.charges.get(), device.dipoles.get(),
                       device.quadrupoles.get(), device.charge_potentials.get(),
                       device.dipole_potentials.get(), device.quadrupole_potentials.get(),
                       device.workspace, device.system_errors.get(), device.device_error.get(),
                       stream) == cudaSuccess &&
                   xtbloom::detail::cuda::add_gfn2_aes2_energy_cuda(
                       device.batch, device.cache, device.charges.get(), device.dipoles.get(),
                       device.quadrupoles.get(), device.energies.get(), device.workspace,
                       device.system_errors.get(), device.device_error.get(),
                       stream) == cudaSuccess &&
                   xtbloom::detail::cuda::add_gfn2_aes2_vjp_cuda(
                       device.batch, device.cache, device.positions.get(),
                       device.coordination.get(), kGeneration, device.charges.get(),
                       device.dipoles.get(), device.quadrupoles.get(), device.gradients.get(),
                       device.coordination_adjoints.get(), device.workspace,
                       device.system_errors.get(), device.device_error.get(),
                       stream) == cudaSuccess;
          },
          [&]() {
            /* The individual term gates above prove every numerical output;
             * the full-chain row additionally proves the composed sequence
             * reaches a clean terminal status after repeated timing calls. */
            return validate_errors();
          },
          &timing, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  rows.push_back(make_row("aes2_full", std::move(timing)));
  if (!xtbloom::test::cuda_term_benchmark::write_results("xtbloom_cuda_aes2_term_benchmark",
                                                         options, argc, argv, rows, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

#endif  // XTBLOOM_AES2_TERM_BENCHMARK_ONLY

}  // namespace

#ifdef XTBLOOM_AES2_TERM_BENCHMARK_ONLY
int main(int argc, char** argv) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;
  return benchmark_aes2_terms(argc, argv);
}
#else
int main() {
  if (const int line = test_ragged_cpu_parity_and_custom_stream(); line != 0) {
    return line;
  }
  if (const int line = test_batch_sizes(); line != 0) {
    return line;
  }
  if (const int line = test_geometry_and_compute_failure_isolation(); line != 0) {
    return line;
  }
  if (const int line = test_hostile_offsets_fail_closed(); line != 0) {
    return line;
  }
  if (const int line = test_graph_capture_and_structural_guards(); line != 0) {
    return line;
  }
  return 0;
}
#endif  // XTBLOOM_AES2_TERM_BENCHMARK_ONLY
