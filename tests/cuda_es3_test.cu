#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_es3.cuh"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/es3.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using xtbloom::detail::cuda::Gfn2ES3DeviceBatch;
using xtbloom::detail::cuda::Gfn2ES3DeviceError;
using xtbloom::detail::gfn2::BasisPlan;
using xtbloom::detail::gfn2::ES3Plan;
using xtbloom::detail::gfn2::ES3View;

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
    if (count == 0u) {
      return cudaErrorInvalidValue;
    }
    count_ = count;
    return cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
  }

  cudaError_t copy_from(const T* source, std::size_t count, cudaStream_t stream = nullptr) {
    if (source == nullptr || count > count_) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(data_, source, count * sizeof(T), cudaMemcpyHostToDevice, stream);
  }

  cudaError_t copy_to(T* destination, std::size_t count, cudaStream_t stream = nullptr) const {
    if (destination == nullptr || count > count_) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
  }

  T* get() { return data_; }
  const T* get() const { return data_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

template <typename T>
cudaError_t allocate_and_copy(DeviceBuffer<T>& destination, const std::vector<T>& source,
                              cudaStream_t stream = nullptr) {
  cudaError_t status = destination.allocate(source.size());
  return status == cudaSuccess ? destination.copy_from(source.data(), source.size(), stream)
                               : status;
}

bool near(double actual, double expected, double absolute_tolerance,
          double relative_tolerance = 0.0) {
  const double scale = std::max(std::abs(actual), std::abs(expected));
  return std::abs(actual - expected) <= absolute_tolerance + relative_tolerance * scale;
}

bool make_plan(const std::vector<std::int64_t>& atom_offsets,
               const std::vector<std::int32_t>& atomic_numbers, BasisPlan& basis, ES3Plan& es3,
               std::string& error) {
  const std::int64_t batch_size = static_cast<std::int64_t>(atom_offsets.size() - 1u);
  const std::int64_t total_atoms = static_cast<std::int64_t>(atomic_numbers.size());
  return xtbloom::detail::gfn2::make_basis_plan(batch_size, total_atoms, atom_offsets.data(),
                                                atomic_numbers.data(), basis,
                                                error) == XTBLOOM_STATUS_SUCCESS &&
         xtbloom::detail::gfn2::make_es3_plan(basis, atomic_numbers.data(), es3, error) ==
             XTBLOOM_STATUS_SUCCESS;
}

struct DeviceFixture {
  DeviceBuffer<std::int64_t> offsets;
  DeviceBuffer<double> gamma3;
  DeviceBuffer<double> charges;
  DeviceBuffer<double> potentials;
  DeviceBuffer<double> energies;
  DeviceBuffer<std::uint32_t> error;

  cudaError_t allocate(const ES3Plan& plan, const std::vector<double>& host_charges,
                       const std::vector<double>& host_potentials,
                       const std::vector<double>& host_energies, cudaStream_t stream = nullptr) {
    cudaError_t status = allocate_and_copy(offsets, plan.batch_shell_offsets, stream);
    if (status == cudaSuccess) {
      status = allocate_and_copy(gamma3, plan.shell_gamma3, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(charges, host_charges, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(potentials, host_potentials, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(energies, host_energies, stream);
    }
    if (status == cudaSuccess) {
      status = error.allocate(1u);
    }
    return status;
  }

  Gfn2ES3DeviceBatch batch(const ES3Plan& plan) const {
    return Gfn2ES3DeviceBatch{
        plan.batch_size,
        plan.total_shells,
        static_cast<std::int64_t>(plan.batch_shell_offsets.size()),
        static_cast<std::int64_t>(plan.shell_gamma3.size()),
        offsets.get(),
        gamma3.get(),
    };
  }
};

int test_ragged_cpu_parity_custom_stream_and_graph() {
  /* Includes an empty member plus s, p, and d shells with both charge signs. */
  const std::vector<std::int64_t> atom_offsets{0, 2, 2, 5, 7};
  const std::vector<std::int32_t> atomic_numbers{1, 8, 6, 22, 7, 14, 79};
  BasisPlan basis;
  ES3Plan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, atomic_numbers, basis, plan, error));
  std::array<bool, 3> seen_angular_momentum{};
  bool seen_positive_gamma3 = false;
  bool seen_negative_gamma3 = false;
  for (std::uint8_t angular_momentum : basis.angular_momenta) {
    CHECK(angular_momentum <= 2u);
    seen_angular_momentum[angular_momentum] = true;
  }
  CHECK((seen_angular_momentum == std::array<bool, 3>{true, true, true}));
  for (double gamma3 : plan.shell_gamma3) {
    seen_positive_gamma3 = seen_positive_gamma3 || gamma3 > 0.0;
    seen_negative_gamma3 = seen_negative_gamma3 || gamma3 < 0.0;
  }
  CHECK(seen_positive_gamma3 && seen_negative_gamma3);

  std::vector<double> charges(static_cast<std::size_t>(plan.total_shells));
  bool seen_positive_charge = false;
  bool seen_negative_charge = false;
  for (std::size_t shell = 0; shell < charges.size(); ++shell) {
    const double magnitude = 0.031 + 0.017 * static_cast<double>(shell % 19u);
    charges[shell] = shell % 2u == 0u ? magnitude : -magnitude;
    seen_positive_charge = seen_positive_charge || charges[shell] > 0.0;
    seen_negative_charge = seen_negative_charge || charges[shell] < 0.0;
  }
  CHECK(seen_positive_charge && seen_negative_charge);

  std::vector<double> expected_potentials(charges.size(), -91.0);
  std::vector<double> energy_seeds{0.125, -0.25, 0.5, -1.0};
  std::vector<double> expected_energies = energy_seeds;
  const ES3View cpu_view = xtbloom::detail::gfn2::make_es3_view(plan);
  CHECK(xtbloom::detail::gfn2::evaluate_es3_potential_cpu(
            cpu_view, charges.data(), expected_potentials.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::add_es3_energy_cpu(
            cpu_view, charges.data(), expected_energies.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(expected_energies[1] == energy_seeds[1]);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture fixture;
  std::vector<double> actual_potentials(charges.size(), -91.0);
  std::vector<double> actual_energies = energy_seeds;
  CUDA_CHECK(fixture.allocate(plan, charges, actual_potentials, actual_energies, stream));
  const Gfn2ES3DeviceBatch batch = fixture.batch(plan);
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_es3_potential_cuda(
            batch, fixture.charges.get(), fixture.charges.get() + 1, fixture.error.get(), stream) ==
        cudaErrorInvalidValue);
  CHECK(xtbloom::detail::cuda::add_gfn2_es3_energy_cuda(
            batch, fixture.charges.get(), fixture.charges.get() + 1, fixture.error.get(), stream) ==
        cudaErrorInvalidValue);

  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_es3_device_error_cuda(fixture.error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_es3_potential_cuda(
      batch, fixture.charges.get(), fixture.potentials.get(), fixture.error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_es3_energy_cuda(
      batch, fixture.charges.get(), fixture.energies.get(), fixture.error.get(), stream));
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(
      fixture.potentials.copy_to(actual_potentials.data(), actual_potentials.size(), stream));
  CUDA_CHECK(fixture.energies.copy_to(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(fixture.error.copy_to(&semantic_error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2ES3DeviceError::kSuccess));
  for (std::size_t shell = 0; shell < actual_potentials.size(); ++shell) {
    CHECK(actual_potentials[shell] == expected_potentials[shell]);
  }
  for (std::size_t system = 0; system < actual_energies.size(); ++system) {
    CHECK(actual_energies[system] == expected_energies[system]);
  }

  /* Capture the complete reset -> potential -> energy sequence and replay it twice. */
  actual_energies = energy_seeds;
  std::fill(actual_potentials.begin(), actual_potentials.end(), 72.0);
  CUDA_CHECK(fixture.energies.copy_from(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(
      fixture.potentials.copy_from(actual_potentials.data(), actual_potentials.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_es3_device_error_cuda(fixture.error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_es3_potential_cuda(
      batch, fixture.charges.get(), fixture.potentials.get(), fixture.error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_es3_energy_cuda(
      batch, fixture.charges.get(), fixture.energies.get(), fixture.error.get(), stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0u));
  CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
  CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
  CUDA_CHECK(
      fixture.potentials.copy_to(actual_potentials.data(), actual_potentials.size(), stream));
  CUDA_CHECK(fixture.energies.copy_to(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(fixture.error.copy_to(&semantic_error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == 0u);
  for (std::size_t shell = 0; shell < actual_potentials.size(); ++shell) {
    CHECK(actual_potentials[shell] == expected_potentials[shell]);
  }
  for (std::size_t system = 0; system < actual_energies.size(); ++system) {
    const double contribution = expected_energies[system] - energy_seeds[system];
    CHECK(near(actual_energies[system], energy_seeds[system] + 2.0 * contribution, 2.0e-16));
  }
  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_extreme_cpu_cuda_parity() {
  const std::vector<std::int64_t> atom_offsets{0, 1};
  const std::vector<std::int32_t> atomic_numbers{1};
  BasisPlan basis;
  ES3Plan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, atomic_numbers, basis, plan, error));
  CHECK(plan.shell_gamma3.size() == 1u && plan.shell_gamma3[0] == 0.08);
  const ES3View cpu_view = xtbloom::detail::gfn2::make_es3_view(plan);

  std::vector<double> charges{2.0 * std::sqrt(std::numeric_limits<double>::max())};
  std::vector<double> expected_potential{-1.0};
  CHECK(std::isinf(charges[0] * charges[0]));
  CHECK(xtbloom::detail::gfn2::evaluate_es3_potential_cpu(
            cpu_view, charges.data(), expected_potential.data(), error) == XTBLOOM_STATUS_SUCCESS);

  std::vector<double> potential{-1.0};
  std::vector<double> energy{0.0};
  DeviceFixture fixture;
  CUDA_CHECK(fixture.allocate(plan, charges, potential, energy));
  const Gfn2ES3DeviceBatch batch = fixture.batch(plan);
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_es3_device_error_cuda(fixture.error.get()));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_es3_potential_cuda(
      batch, fixture.charges.get(), fixture.potentials.get(), fixture.error.get()));
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(fixture.potentials.copy_to(potential.data(), 1u));
  CUDA_CHECK(fixture.error.copy_to(&semantic_error, 1u));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error == 0u);
  CHECK(near(potential[0], expected_potential[0], 0.0, 2.0e-15));

  charges[0] = 1.5e103;
  std::vector<double> expected_energy{0.375};
  CHECK(std::isinf(charges[0] * charges[0] * charges[0]));
  CHECK(xtbloom::detail::gfn2::add_es3_energy_cpu(cpu_view, charges.data(), expected_energy.data(),
                                                  error) == XTBLOOM_STATUS_SUCCESS);
  energy[0] = 0.375;
  CUDA_CHECK(fixture.charges.copy_from(charges.data(), 1u));
  CUDA_CHECK(fixture.energies.copy_from(energy.data(), 1u));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_es3_device_error_cuda(fixture.error.get()));
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_es3_energy_cuda(
      batch, fixture.charges.get(), fixture.energies.get(), fixture.error.get()));
  CUDA_CHECK(fixture.energies.copy_to(energy.data(), 1u));
  CUDA_CHECK(fixture.error.copy_to(&semantic_error, 1u));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error == 0u);
  CHECK(near(energy[0], expected_energy[0], 0.0, 3.0e-15));

  /* q^2/q^3 may underflow before a huge finite Gamma3 restores normal range. */
  plan.shell_gamma3[0] = std::numeric_limits<double>::max();
  charges[0] = 1.0e-200;
  expected_potential[0] = -1.0;
  const ES3View restored_cpu_view = xtbloom::detail::gfn2::make_es3_view(plan);
  CHECK(xtbloom::detail::gfn2::evaluate_es3_potential_cpu(restored_cpu_view, charges.data(),
                                                          expected_potential.data(),
                                                          error) == XTBLOOM_STATUS_SUCCESS);
  potential[0] = -1.0;
  CUDA_CHECK(fixture.gamma3.copy_from(plan.shell_gamma3.data(), 1u));
  CUDA_CHECK(fixture.charges.copy_from(charges.data(), 1u));
  CUDA_CHECK(fixture.potentials.copy_from(potential.data(), 1u));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_es3_device_error_cuda(fixture.error.get()));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_es3_potential_cuda(
      batch, fixture.charges.get(), fixture.potentials.get(), fixture.error.get()));
  CUDA_CHECK(fixture.potentials.copy_to(potential.data(), 1u));
  CUDA_CHECK(fixture.error.copy_to(&semantic_error, 1u));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error == 0u);
  CHECK(near(potential[0], expected_potential[0], 0.0, 2.0e-15));
  CHECK(potential[0] > 0.0);

  charges[0] = 1.0e-110;
  expected_energy[0] = 0.0;
  CHECK(xtbloom::detail::gfn2::add_es3_energy_cpu(restored_cpu_view, charges.data(),
                                                  expected_energy.data(),
                                                  error) == XTBLOOM_STATUS_SUCCESS);
  energy[0] = 0.0;
  CUDA_CHECK(fixture.charges.copy_from(charges.data(), 1u));
  CUDA_CHECK(fixture.energies.copy_from(energy.data(), 1u));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_es3_device_error_cuda(fixture.error.get()));
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_es3_energy_cuda(
      batch, fixture.charges.get(), fixture.energies.get(), fixture.error.get()));
  CUDA_CHECK(fixture.energies.copy_to(energy.data(), 1u));
  CUDA_CHECK(fixture.error.copy_to(&semantic_error, 1u));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error == 0u);
  CHECK(near(energy[0], expected_energy[0], 0.0, 2.0e-15));
  CHECK(energy[0] > 0.0);

  /* Gamma3=0 is exact even when q^2/q^3 would overflow before multiplication. */
  plan.shell_gamma3[0] = 0.0;
  charges[0] = std::numeric_limits<double>::max();
  expected_potential[0] = 0.0;
  expected_energy[0] = 0.625;
  potential[0] = -1.0;
  energy[0] = 0.625;
  CUDA_CHECK(fixture.gamma3.copy_from(plan.shell_gamma3.data(), 1u));
  CUDA_CHECK(fixture.charges.copy_from(charges.data(), 1u));
  CUDA_CHECK(fixture.potentials.copy_from(potential.data(), 1u));
  CUDA_CHECK(fixture.energies.copy_from(energy.data(), 1u));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_es3_device_error_cuda(fixture.error.get()));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_es3_potential_cuda(
      batch, fixture.charges.get(), fixture.potentials.get(), fixture.error.get()));
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_es3_energy_cuda(
      batch, fixture.charges.get(), fixture.energies.get(), fixture.error.get()));
  CUDA_CHECK(fixture.potentials.copy_to(potential.data(), 1u));
  CUDA_CHECK(fixture.energies.copy_to(energy.data(), 1u));
  CUDA_CHECK(fixture.error.copy_to(&semantic_error, 1u));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error == 0u);
  CHECK(potential[0] == expected_potential[0]);
  CHECK(energy[0] == expected_energy[0]);

  /* A subnormal Gamma3 must not be rounded to zero before multiplying q^3. */
  plan.shell_gamma3[0] = std::numeric_limits<double>::denorm_min();
  charges[0] = 1.0e108;
  expected_energy[0] = -0.25;
  const ES3View subnormal_cpu_view = xtbloom::detail::gfn2::make_es3_view(plan);
  CHECK(xtbloom::detail::gfn2::add_es3_energy_cpu(subnormal_cpu_view, charges.data(),
                                                  expected_energy.data(),
                                                  error) == XTBLOOM_STATUS_SUCCESS);
  energy[0] = -0.25;
  CUDA_CHECK(fixture.gamma3.copy_from(plan.shell_gamma3.data(), 1u));
  CUDA_CHECK(fixture.charges.copy_from(charges.data(), 1u));
  CUDA_CHECK(fixture.energies.copy_from(energy.data(), 1u));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_es3_device_error_cuda(fixture.error.get()));
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_es3_energy_cuda(
      batch, fixture.charges.get(), fixture.energies.get(), fixture.error.get()));
  CUDA_CHECK(fixture.energies.copy_to(energy.data(), 1u));
  CUDA_CHECK(fixture.error.copy_to(&semantic_error, 1u));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error == 0u);
  CHECK(near(energy[0], expected_energy[0], 0.0, 3.0e-15));
  return 0;
}

int test_strided_potential_failure_atomicity() {
  /*
   * Put the failing shell in thread 0's second stride. This catches a kernel
   * that publishes its first stride before the complete system preflight has
   * established that every shell result is finite.
   */
  constexpr std::size_t shell_count = 257u;
  ES3Plan plan;
  plan.batch_size = 1;
  plan.total_shells = static_cast<std::int64_t>(shell_count);
  plan.batch_shell_offsets = {0, plan.total_shells};
  plan.shell_gamma3.assign(shell_count, 0.08);

  std::vector<double> charges(shell_count, 0.2);
  charges.back() = std::numeric_limits<double>::max();
  std::vector<double> potentials(shell_count);
  for (std::size_t shell = 0; shell < shell_count; ++shell) {
    potentials[shell] = 17.0 + static_cast<double>(shell);
  }
  const std::vector<double> seeds = potentials;
  std::vector<double> energies{0.0};

  DeviceFixture fixture;
  CUDA_CHECK(fixture.allocate(plan, charges, potentials, energies));
  const Gfn2ES3DeviceBatch batch = fixture.batch(plan);
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_es3_device_error_cuda(fixture.error.get()));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_es3_potential_cuda(
      batch, fixture.charges.get(), fixture.potentials.get(), fixture.error.get()));

  std::uint32_t semantic_error = 0u;
  CUDA_CHECK(fixture.potentials.copy_to(potentials.data(), potentials.size()));
  CUDA_CHECK(fixture.error.copy_to(&semantic_error, 1u));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error ==
        static_cast<std::uint32_t>(Gfn2ES3DeviceError::kNonfinitePotentialArithmetic));
  CHECK(potentials == seeds);
  return 0;
}

int test_alias_overflow_atomicity_and_sticky_error() {
  const std::vector<std::int64_t> atom_offsets{0, 1};
  const std::vector<std::int32_t> atomic_numbers{1};
  BasisPlan basis;
  ES3Plan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, atomic_numbers, basis, plan, error));
  std::vector<double> charges{0.2};
  std::vector<double> potentials{17.0};
  std::vector<double> energies{3.0};
  DeviceFixture fixture;
  CUDA_CHECK(fixture.allocate(plan, charges, potentials, energies));
  Gfn2ES3DeviceBatch batch = fixture.batch(plan);

  CHECK(xtbloom::detail::cuda::reset_gfn2_es3_device_error_cuda(nullptr) == cudaErrorInvalidValue);
  Gfn2ES3DeviceBatch invalid{};
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_es3_potential_cuda(invalid, nullptr, nullptr,
                                                                nullptr) == cudaErrorInvalidValue);
  CHECK(xtbloom::detail::cuda::add_gfn2_es3_energy_cuda(invalid, nullptr, nullptr, nullptr) ==
        cudaErrorInvalidValue);
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_es3_potential_cuda(
            batch, fixture.charges.get(), fixture.charges.get(), fixture.error.get()) ==
        cudaErrorInvalidValue);
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_es3_potential_cuda(
            batch, fixture.charges.get(), fixture.gamma3.get(), fixture.error.get()) ==
        cudaErrorInvalidValue);
  CHECK(xtbloom::detail::cuda::add_gfn2_es3_energy_cuda(
            batch, fixture.charges.get(), fixture.charges.get(), fixture.error.get()) ==
        cudaErrorInvalidValue);
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_es3_potential_cuda(
            batch, fixture.charges.get(), fixture.potentials.get(),
            reinterpret_cast<std::uint32_t*>(fixture.potentials.get())) == cudaErrorInvalidValue);
  invalid = batch;
  --invalid.shell_gamma3_count;
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_es3_potential_cuda(
            invalid, fixture.charges.get(), fixture.potentials.get(), fixture.error.get()) ==
        cudaErrorInvalidValue);
  invalid = batch;
  invalid.batch_size = static_cast<std::int64_t>(std::numeric_limits<int>::max()) + 1;
  invalid.batch_shell_offset_count = invalid.batch_size + 1;
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_es3_potential_cuda(
            invalid, fixture.charges.get(), fixture.potentials.get(), fixture.error.get()) ==
        cudaErrorInvalidConfiguration);

  auto run_potential_failure = [&](double gamma3, double charge,
                                   Gfn2ES3DeviceError expected_error) -> int {
    const double seed = 17.0;
    CUDA_CHECK(fixture.gamma3.copy_from(&gamma3, 1u));
    CUDA_CHECK(fixture.charges.copy_from(&charge, 1u));
    CUDA_CHECK(fixture.potentials.copy_from(&seed, 1u));
    CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_es3_device_error_cuda(fixture.error.get()));
    CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_es3_potential_cuda(
        batch, fixture.charges.get(), fixture.potentials.get(), fixture.error.get()));
    double actual = 0.0;
    std::uint32_t semantic_error = 0u;
    CUDA_CHECK(fixture.potentials.copy_to(&actual, 1u));
    CUDA_CHECK(fixture.error.copy_to(&semantic_error, 1u));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(semantic_error == static_cast<std::uint32_t>(expected_error));
    CHECK(actual == seed);
    return 0;
  };
  CHECK(run_potential_failure(std::numeric_limits<double>::quiet_NaN(), 0.2,
                              Gfn2ES3DeviceError::kNonfiniteGamma3) == 0);
  CHECK(run_potential_failure(0.08, std::numeric_limits<double>::infinity(),
                              Gfn2ES3DeviceError::kNonfiniteShellCharge) == 0);
  CHECK(run_potential_failure(0.08, std::numeric_limits<double>::max(),
                              Gfn2ES3DeviceError::kNonfinitePotentialArithmetic) == 0);

  /* A failed potential stage suppresses dependent energy without host polling. */
  const double gamma3 = 0.08;
  const double overflow_charge = std::numeric_limits<double>::max();
  double potential_seed = 17.0;
  double energy_seed = 3.0;
  CUDA_CHECK(fixture.gamma3.copy_from(&gamma3, 1u));
  CUDA_CHECK(fixture.charges.copy_from(&overflow_charge, 1u));
  CUDA_CHECK(fixture.potentials.copy_from(&potential_seed, 1u));
  CUDA_CHECK(fixture.energies.copy_from(&energy_seed, 1u));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_es3_device_error_cuda(fixture.error.get()));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_es3_potential_cuda(
      batch, fixture.charges.get(), fixture.potentials.get(), fixture.error.get()));
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_es3_energy_cuda(
      batch, fixture.charges.get(), fixture.energies.get(), fixture.error.get()));
  std::uint32_t semantic_error = 0u;
  CUDA_CHECK(fixture.energies.copy_to(&energy_seed, 1u));
  CUDA_CHECK(fixture.error.copy_to(&semantic_error, 1u));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error ==
        static_cast<std::uint32_t>(Gfn2ES3DeviceError::kNonfinitePotentialArithmetic));
  CHECK(energy_seed == 3.0);

  auto run_energy_failure = [&](double charge, double seed,
                                Gfn2ES3DeviceError expected_error) -> int {
    CUDA_CHECK(fixture.charges.copy_from(&charge, 1u));
    CUDA_CHECK(fixture.energies.copy_from(&seed, 1u));
    CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_es3_device_error_cuda(fixture.error.get()));
    CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_es3_energy_cuda(
        batch, fixture.charges.get(), fixture.energies.get(), fixture.error.get()));
    double actual = 0.0;
    std::uint32_t actual_error = 0u;
    CUDA_CHECK(fixture.energies.copy_to(&actual, 1u));
    CUDA_CHECK(fixture.error.copy_to(&actual_error, 1u));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(actual_error == static_cast<std::uint32_t>(expected_error));
    CHECK(actual == seed);
    return 0;
  };
  CHECK(run_energy_failure(0.2, std::numeric_limits<double>::infinity(),
                           Gfn2ES3DeviceError::kNonfiniteEnergySeed) == 0);
  CHECK(run_energy_failure(std::numeric_limits<double>::max(), 3.0,
                           Gfn2ES3DeviceError::kNonfiniteEnergyArithmetic) == 0);
  CHECK(run_energy_failure(1.5e103, std::numeric_limits<double>::max(),
                           Gfn2ES3DeviceError::kNonfiniteEnergyArithmetic) == 0);

  const std::vector<std::int64_t> invalid_offsets{1, 1};
  CUDA_CHECK(fixture.offsets.copy_from(invalid_offsets.data(), invalid_offsets.size()));
  potential_seed = 17.0;
  const double ordinary_charge = 0.2;
  CUDA_CHECK(fixture.charges.copy_from(&ordinary_charge, 1u));
  CUDA_CHECK(fixture.potentials.copy_from(&potential_seed, 1u));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_es3_device_error_cuda(fixture.error.get()));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_es3_potential_cuda(
      batch, fixture.charges.get(), fixture.potentials.get(), fixture.error.get()));
  CUDA_CHECK(fixture.potentials.copy_to(&potential_seed, 1u));
  CUDA_CHECK(fixture.error.copy_to(&semantic_error, 1u));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2ES3DeviceError::kInvalidOffsets));
  CHECK(potential_seed == 17.0);
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    if (std::getenv("XTBLOOM_TEST_REQUIRE_DEVICE") != nullptr) {
      std::cerr << "CUDA ES3 test requires a visible device: "
                << (count_status == cudaSuccess ? "none found" : cudaGetErrorString(count_status))
                << '\n';
      return 1;
    }
    std::cout << "CUDA ES3 test skipped: "
              << (count_status == cudaSuccess ? "no visible device"
                                              : cudaGetErrorString(count_status))
              << '\n';
    return 0;
  }
  if (const int status = test_ragged_cpu_parity_custom_stream_and_graph(); status != 0) {
    return status;
  }
  if (const int status = test_extreme_cpu_cuda_parity(); status != 0) {
    return status;
  }
  if (const int status = test_strided_potential_failure_atomicity(); status != 0) {
    return status;
  }
  return test_alias_overflow_atomicity_and_sticky_error();
}
