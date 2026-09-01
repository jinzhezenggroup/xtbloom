// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_native_periodic_ewald.cuh"
#include "backends/cuda/periodic_topology.cuh"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/es2.hpp"
#include "model/gfn2/periodic_ewald.hpp"
#include "model/gfn2/periodic_topology.hpp"

#define CHECK(condition)                                                                  \
  do {                                                                                    \
    if (!(condition)) {                                                                   \
      std::cerr << "CUDA native periodic Ewald check failed at line " << __LINE__ << ": " \
                << #condition << '\n';                                                    \
      return __LINE__;                                                                    \
    }                                                                                     \
  } while (false)

namespace {

using xtbloom::detail::cuda::Gfn2CudaPeriodicTopology;
using xtbloom::detail::cuda::Gfn2CudaPeriodicTopologyInput;
using xtbloom::detail::cuda::Gfn2CudaPeriodicTranslation;
using xtbloom::detail::cuda::Gfn2NativePeriodicEwaldDeviceBatch;
using xtbloom::detail::cuda::Gfn2NativePeriodicEwaldDeviceWorkspace;
using xtbloom::detail::gfn2::BasisPlan;
using xtbloom::detail::gfn2::ES2Plan;
using xtbloom::detail::gfn2::PeriodicEwaldPlan;
using xtbloom::detail::gfn2::PeriodicShortRangePlan;

constexpr std::uint64_t kPlanToken = 0x4557414c444e4154ULL;
constexpr double kSentinel = -913.25;

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (data_ != nullptr) (void)cudaFree(data_);
  }

  cudaError_t allocate(std::size_t count) {
    count_ = count;
    if (count == 0u) return cudaSuccess;
    return cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
  }

  cudaError_t upload(const std::vector<T>& values, cudaStream_t stream) {
    if (values.size() > count_ || values.empty())
      return values.empty() ? cudaSuccess : cudaErrorInvalidValue;
    return cudaMemcpyAsync(data_, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice,
                           stream);
  }

  cudaError_t download(std::vector<T>& values, cudaStream_t stream) const {
    if (values.size() > count_ || values.empty())
      return values.empty() ? cudaSuccess : cudaErrorInvalidValue;
    return cudaMemcpyAsync(values.data(), data_, values.size() * sizeof(T), cudaMemcpyDeviceToHost,
                           stream);
  }

  T* get() noexcept { return data_; }
  const T* get() const noexcept { return data_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

bool close(double actual, double expected, double tolerance = 2.0e-10) {
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::abs(actual - expected) <= tolerance * (1.0 + std::abs(expected));
}

std::vector<Gfn2CudaPeriodicTranslation> convert_translations(
    const std::vector<xtbloom::detail::gfn2::LatticeTranslation>& source) {
  std::vector<Gfn2CudaPeriodicTranslation> result(source.size());
  for (std::size_t index = 0u; index < source.size(); ++index) {
    for (int component = 0; component < 3; ++component) {
      result[index].index[component] = source[index].index[component];
      result[index].cartesian[component] = source[index].cartesian[component];
    }
  }
  return result;
}

int test_ewald_parity_and_peer_transaction(cudaStream_t stream) {
  constexpr std::array<std::int64_t, 3> atom_offsets{0, 3, 6};
  constexpr std::array<std::int32_t, 6> atomic_numbers{8, 1, 1, 8, 1, 1};
  constexpr std::array<double, 18> cells{
      11.7, 0.0, 0.0, 1.1, 12.9, 0.0, -0.7, 0.8, 14.3,
      11.7, 0.0, 0.0, 1.1, 12.9, 0.0, -0.7, 0.8, 14.3,
  };
  constexpr std::array<std::int32_t, 2> periodic_axes{XTBLOOM_PERIODIC_AXES_XYZ,
                                                      XTBLOOM_PERIODIC_AXES_XYZ};
  constexpr std::array<double, 18> positions{
      0.0, 0.0, 0.0, 1.42, 0.08, 1.08, -1.31, 0.17, 0.96,
      0.0, 0.0, 0.0, 1.42, 0.08, 1.08, -1.31, 0.17, 0.96,
  };
  constexpr std::array<double, 8> neutral_shell_charges{-0.12, -0.08, 0.1, 0.1,
                                                        -0.12, -0.08, 0.1, 0.1};
  constexpr std::array<double, 8> charged_shell_charges{-0.02, -0.08, 0.1, 0.1,
                                                        -0.02, -0.08, 0.1, 0.1};

  std::string error;
  BasisPlan basis;
  ES2Plan es2;
  PeriodicShortRangePlan topology_plan;
  PeriodicEwaldPlan ewald;
  CHECK(xtbloom::detail::gfn2::make_basis_plan(2, 6, atom_offsets.data(), atomic_numbers.data(),
                                               basis, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::make_es2_plan(basis, atomic_numbers.data(), es2, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::make_periodic_short_range_plan(2, 6, atom_offsets.data(),
                                                              cells.data(), topology_plan,
                                                              error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::make_periodic_ewald_plan(es2, topology_plan, ewald, error) ==
        XTBLOOM_STATUS_SUCCESS);

  const std::vector<double> neutral_charges(neutral_shell_charges.begin(),
                                            neutral_shell_charges.end());
  std::vector<double> cpu_matrix(static_cast<std::size_t>(ewald.total_matrix_elements()));
  std::vector<double> cpu_potential(static_cast<std::size_t>(ewald.total_shells()));
  std::vector<double> cpu_energy(static_cast<std::size_t>(ewald.batch_size()));
  std::vector<double> cpu_gradient(static_cast<std::size_t>(ewald.total_atoms()) * 3u);
  std::vector<double> cpu_strain(static_cast<std::size_t>(ewald.batch_size()) * 9u);
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_ewald_cpu(
            ewald, topology_plan, positions.data(), neutral_charges.data(), cpu_matrix.data(),
            cpu_potential.data(), cpu_energy.data(), cpu_gradient.data(), cpu_strain.data(),
            error) == XTBLOOM_STATUS_SUCCESS);

  Gfn2CudaPeriodicTopology cuda_topology;
  const Gfn2CudaPeriodicTopologyInput topology_input{
      2, 6, atom_offsets.data(), cells.data(), periodic_axes.data(), 25.0, kPlanToken, 1u};
  CHECK(Gfn2CudaPeriodicTopology::create(topology_input, stream, cuda_topology).success());

  const std::vector<Gfn2CudaPeriodicTranslation> direct_translations =
      convert_translations(ewald.direct_translations());
  const std::vector<Gfn2CudaPeriodicTranslation> reciprocal_translations =
      convert_translations(ewald.reciprocal_translations());
  DeviceBuffer<std::int64_t> batch_shell_offsets;
  DeviceBuffer<std::int64_t> atom_shell_offsets;
  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<double> shell_hardness;
  DeviceBuffer<double> alphas;
  DeviceBuffer<std::int64_t> direct_offsets;
  DeviceBuffer<Gfn2CudaPeriodicTranslation> direct;
  DeviceBuffer<std::int64_t> reciprocal_offsets;
  DeviceBuffer<Gfn2CudaPeriodicTranslation> reciprocal;
  DeviceBuffer<double> device_positions;
  DeviceBuffer<double> shell_charges;
  DeviceBuffer<double> wrapped_positions;
  DeviceBuffer<double> matrix_scratch;
  DeviceBuffer<double> potential_scratch;
  DeviceBuffer<double> energy_scratch;
  DeviceBuffer<double> gradient_scratch;
  DeviceBuffer<double> strain_scratch;
  DeviceBuffer<double> matrix;
  DeviceBuffer<double> potential;
  DeviceBuffer<double> energy;
  DeviceBuffer<double> gradient;
  DeviceBuffer<double> strain;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;

  CHECK(batch_shell_offsets.allocate(ewald.batch_shell_offsets().size()) == cudaSuccess);
  CHECK(atom_shell_offsets.allocate(ewald.atom_shell_offsets().size()) == cudaSuccess);
  CHECK(matrix_offsets.allocate(ewald.matrix_offsets().size()) == cudaSuccess);
  CHECK(shell_hardness.allocate(ewald.shell_hardness().size()) == cudaSuccess);
  CHECK(alphas.allocate(ewald.alphas().size()) == cudaSuccess);
  CHECK(direct_offsets.allocate(ewald.direct_translation_offsets().size()) == cudaSuccess);
  CHECK(direct.allocate(direct_translations.size()) == cudaSuccess);
  CHECK(reciprocal_offsets.allocate(ewald.reciprocal_translation_offsets().size()) == cudaSuccess);
  CHECK(reciprocal.allocate(reciprocal_translations.size()) == cudaSuccess);
  CHECK(device_positions.allocate(positions.size()) == cudaSuccess);
  CHECK(shell_charges.allocate(neutral_shell_charges.size()) == cudaSuccess);
  CHECK(wrapped_positions.allocate(positions.size()) == cudaSuccess);
  CHECK(matrix_scratch.allocate(cpu_matrix.size()) == cudaSuccess);
  CHECK(potential_scratch.allocate(cpu_potential.size()) == cudaSuccess);
  CHECK(energy_scratch.allocate(cpu_energy.size()) == cudaSuccess);
  CHECK(gradient_scratch.allocate(cpu_gradient.size()) == cudaSuccess);
  CHECK(strain_scratch.allocate(cpu_strain.size()) == cudaSuccess);
  CHECK(matrix.allocate(cpu_matrix.size()) == cudaSuccess);
  CHECK(potential.allocate(cpu_potential.size()) == cudaSuccess);
  CHECK(energy.allocate(cpu_energy.size()) == cudaSuccess);
  CHECK(gradient.allocate(cpu_gradient.size()) == cudaSuccess);
  CHECK(strain.allocate(cpu_strain.size()) == cudaSuccess);
  CHECK(system_errors.allocate(2u) == cudaSuccess);
  CHECK(device_error.allocate(1u) == cudaSuccess);

  CHECK(batch_shell_offsets.upload(ewald.batch_shell_offsets(), stream) == cudaSuccess);
  CHECK(atom_shell_offsets.upload(ewald.atom_shell_offsets(), stream) == cudaSuccess);
  CHECK(matrix_offsets.upload(ewald.matrix_offsets(), stream) == cudaSuccess);
  CHECK(shell_hardness.upload(ewald.shell_hardness(), stream) == cudaSuccess);
  CHECK(alphas.upload(ewald.alphas(), stream) == cudaSuccess);
  CHECK(direct_offsets.upload(ewald.direct_translation_offsets(), stream) == cudaSuccess);
  CHECK(direct.upload(direct_translations, stream) == cudaSuccess);
  CHECK(reciprocal_offsets.upload(ewald.reciprocal_translation_offsets(), stream) == cudaSuccess);
  CHECK(reciprocal.upload(reciprocal_translations, stream) == cudaSuccess);
  const std::vector<double> position_values(positions.begin(), positions.end());
  CHECK(device_positions.upload(position_values, stream) == cudaSuccess);
  CHECK(shell_charges.upload(neutral_charges, stream) == cudaSuccess);

  const auto topology_view = cuda_topology.device_view();
  const Gfn2NativePeriodicEwaldDeviceBatch batch{
      topology_view,
      batch_shell_offsets.get(),
      static_cast<std::int64_t>(ewald.batch_shell_offsets().size()),
      atom_shell_offsets.get(),
      static_cast<std::int64_t>(ewald.atom_shell_offsets().size()),
      matrix_offsets.get(),
      static_cast<std::int64_t>(ewald.matrix_offsets().size()),
      ewald.total_matrix_elements(),
      shell_hardness.get(),
      static_cast<std::int64_t>(ewald.shell_hardness().size()),
      alphas.get(),
      static_cast<std::int64_t>(ewald.alphas().size()),
      direct_offsets.get(),
      static_cast<std::int64_t>(ewald.direct_translation_offsets().size()),
      direct.get(),
      static_cast<std::int64_t>(direct_translations.size()),
      reciprocal_offsets.get(),
      static_cast<std::int64_t>(ewald.reciprocal_translation_offsets().size()),
      reciprocal.get(),
      static_cast<std::int64_t>(reciprocal_translations.size()),
      device_positions.get(),
      static_cast<std::int64_t>(positions.size()),
      shell_charges.get(),
      static_cast<std::int64_t>(neutral_shell_charges.size()),
  };
  const Gfn2NativePeriodicEwaldDeviceWorkspace workspace{
      wrapped_positions.get(),
      static_cast<std::int64_t>(positions.size()),
      matrix_scratch.get(),
      static_cast<std::int64_t>(cpu_matrix.size()),
      potential_scratch.get(),
      static_cast<std::int64_t>(cpu_potential.size()),
      energy_scratch.get(),
      static_cast<std::int64_t>(cpu_energy.size()),
      gradient_scratch.get(),
      static_cast<std::int64_t>(cpu_gradient.size()),
      strain_scratch.get(),
      static_cast<std::int64_t>(cpu_strain.size()),
      kPlanToken};

  const std::vector<double> matrix_seed(cpu_matrix.size(), kSentinel);
  const std::vector<double> potential_seed(cpu_potential.size(), kSentinel);
  const std::vector<double> energy_seed(cpu_energy.size(), kSentinel);
  const std::vector<double> gradient_seed(cpu_gradient.size(), kSentinel);
  const std::vector<double> strain_seed(cpu_strain.size(), kSentinel);
  CHECK(matrix.upload(matrix_seed, stream) == cudaSuccess);
  CHECK(potential.upload(potential_seed, stream) == cudaSuccess);
  CHECK(energy.upload(energy_seed, stream) == cudaSuccess);
  CHECK(gradient.upload(gradient_seed, stream) == cudaSuccess);
  CHECK(strain.upload(strain_seed, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);

  CHECK(xtbloom::detail::cuda::reset_gfn2_native_periodic_ewald_errors_cuda(
            2, system_errors.get(), device_error.get(), stream) == cudaSuccess);
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_native_periodic_ewald_cuda(
            batch, workspace, matrix.get(), potential.get(), energy.get(), gradient.get(),
            strain.get(), system_errors.get(), device_error.get(), stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);

  std::vector<double> gpu_matrix(cpu_matrix.size());
  std::vector<double> gpu_potential(cpu_potential.size());
  std::vector<double> gpu_energy(cpu_energy.size());
  std::vector<double> gpu_gradient(cpu_gradient.size());
  std::vector<double> gpu_strain(cpu_strain.size());
  std::vector<std::uint32_t> gpu_system_errors(2u);
  std::vector<std::uint32_t> gpu_device_error(1u);
  CHECK(matrix.download(gpu_matrix, stream) == cudaSuccess);
  CHECK(potential.download(gpu_potential, stream) == cudaSuccess);
  CHECK(energy.download(gpu_energy, stream) == cudaSuccess);
  CHECK(gradient.download(gpu_gradient, stream) == cudaSuccess);
  CHECK(strain.download(gpu_strain, stream) == cudaSuccess);
  CHECK(system_errors.download(gpu_system_errors, stream) == cudaSuccess);
  CHECK(device_error.download(gpu_device_error, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  for (std::size_t index = 0u; index < 16u; ++index) {
    if (!close(gpu_matrix[index], cpu_matrix[index])) {
      std::cerr << "matrix[" << index << "] gpu=" << gpu_matrix[index]
                << " cpu=" << cpu_matrix[index] << '\n';
      CHECK(false);
    }
  }
  for (std::size_t index = 0u; index < 4u; ++index)
    CHECK(close(gpu_potential[index], cpu_potential[index]));
  CHECK(close(gpu_energy[0], cpu_energy[0]));
  for (std::size_t index = 0u; index < 9u; ++index)
    CHECK(close(gpu_gradient[index], cpu_gradient[index]));
  for (std::size_t index = 0u; index < 9u; ++index)
    CHECK(close(gpu_strain[index], cpu_strain[index]));
  CHECK(gpu_system_errors[0] == 0u && gpu_system_errors[1] == 0u);
  CHECK(gpu_device_error[0] == 0u);

  /* Poison only peer one.  The first peer must still publish, while every
   * matrix/potential/energy/derivative slice of the failed peer stays seeded. */
  std::vector<double> poisoned_charges(neutral_charges);
  poisoned_charges[4] = std::numeric_limits<double>::quiet_NaN();
  CHECK(shell_charges.upload(poisoned_charges, stream) == cudaSuccess);
  CHECK(matrix.upload(matrix_seed, stream) == cudaSuccess);
  CHECK(potential.upload(potential_seed, stream) == cudaSuccess);
  CHECK(energy.upload(energy_seed, stream) == cudaSuccess);
  CHECK(gradient.upload(gradient_seed, stream) == cudaSuccess);
  CHECK(strain.upload(strain_seed, stream) == cudaSuccess);
  CHECK(xtbloom::detail::cuda::reset_gfn2_native_periodic_ewald_errors_cuda(
            2, system_errors.get(), device_error.get(), stream) == cudaSuccess);
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_native_periodic_ewald_cuda(
            batch, workspace, matrix.get(), potential.get(), energy.get(), gradient.get(),
            strain.get(), system_errors.get(), device_error.get(), stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  CHECK(matrix.download(gpu_matrix, stream) == cudaSuccess);
  CHECK(potential.download(gpu_potential, stream) == cudaSuccess);
  CHECK(energy.download(gpu_energy, stream) == cudaSuccess);
  CHECK(gradient.download(gpu_gradient, stream) == cudaSuccess);
  CHECK(strain.download(gpu_strain, stream) == cudaSuccess);
  CHECK(system_errors.download(gpu_system_errors, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  for (std::size_t index = 0u; index < 16u; ++index)
    CHECK(close(gpu_matrix[index], cpu_matrix[index]));
  CHECK(gpu_energy[0] != kSentinel);
  for (std::size_t index = 16u; index < gpu_matrix.size(); ++index)
    CHECK(gpu_matrix[index] == kSentinel);
  for (std::size_t index = 4u; index < gpu_potential.size(); ++index)
    CHECK(gpu_potential[index] == kSentinel);
  CHECK(gpu_energy[1] == kSentinel);
  for (std::size_t index = 9u; index < gpu_gradient.size(); ++index)
    CHECK(gpu_gradient[index] == kSentinel);
  for (std::size_t index = 9u; index < gpu_strain.size(); ++index)
    CHECK(gpu_strain[index] == kSentinel);
  CHECK(gpu_system_errors[0] == 0u && gpu_system_errors[1] == 3u);

  /* Charged cells exercise the uniform-background potential, energy, and
   * isotropic strain contribution independently of the neutral fixture. */
  const std::vector<double> charged_values(charged_shell_charges.begin(),
                                           charged_shell_charges.end());
  std::vector<double> charged_cpu_matrix(cpu_matrix.size());
  std::vector<double> charged_cpu_potential(cpu_potential.size());
  std::vector<double> charged_cpu_energy(cpu_energy.size());
  std::vector<double> charged_cpu_gradient(cpu_gradient.size());
  std::vector<double> charged_cpu_strain(cpu_strain.size());
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_ewald_cpu(
            ewald, topology_plan, positions.data(), charged_values.data(),
            charged_cpu_matrix.data(), charged_cpu_potential.data(), charged_cpu_energy.data(),
            charged_cpu_gradient.data(), charged_cpu_strain.data(),
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(shell_charges.upload(charged_values, stream) == cudaSuccess);
  CHECK(matrix.upload(matrix_seed, stream) == cudaSuccess);
  CHECK(potential.upload(potential_seed, stream) == cudaSuccess);
  CHECK(energy.upload(energy_seed, stream) == cudaSuccess);
  CHECK(gradient.upload(gradient_seed, stream) == cudaSuccess);
  CHECK(strain.upload(strain_seed, stream) == cudaSuccess);
  CHECK(xtbloom::detail::cuda::reset_gfn2_native_periodic_ewald_errors_cuda(
            2, system_errors.get(), device_error.get(), stream) == cudaSuccess);
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_native_periodic_ewald_cuda(
            batch, workspace, matrix.get(), potential.get(), energy.get(), gradient.get(),
            strain.get(), system_errors.get(), device_error.get(), stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  CHECK(matrix.download(gpu_matrix, stream) == cudaSuccess);
  CHECK(potential.download(gpu_potential, stream) == cudaSuccess);
  CHECK(energy.download(gpu_energy, stream) == cudaSuccess);
  CHECK(gradient.download(gpu_gradient, stream) == cudaSuccess);
  CHECK(strain.download(gpu_strain, stream) == cudaSuccess);
  CHECK(system_errors.download(gpu_system_errors, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  for (std::size_t index = 0u; index < 32u; ++index)
    CHECK(close(gpu_matrix[index], charged_cpu_matrix[index]));
  for (std::size_t index = 0u; index < 8u; ++index)
    CHECK(close(gpu_potential[index], charged_cpu_potential[index]));
  for (std::size_t index = 0u; index < 2u; ++index)
    CHECK(close(gpu_energy[index], charged_cpu_energy[index]));
  for (std::size_t index = 0u; index < 18u; ++index)
    CHECK(close(gpu_gradient[index], charged_cpu_gradient[index]));
  for (std::size_t index = 0u; index < 18u; ++index)
    CHECK(close(gpu_strain[index], charged_cpu_strain[index]));
  CHECK(gpu_system_errors[0] == 0u && gpu_system_errors[1] == 0u);

  Gfn2NativePeriodicEwaldDeviceBatch malformed = batch;
  malformed.position_elements -= 1;
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_native_periodic_ewald_cuda(
            malformed, workspace, matrix.get(), potential.get(), energy.get(), gradient.get(),
            strain.get(), system_errors.get(), device_error.get(),
            stream) == cudaErrorInvalidValue);
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    std::cout << "CUDA native periodic Ewald test skipped: "
              << (count_status == cudaSuccess ? "no visible device"
                                              : cudaGetErrorString(count_status))
              << '\n';
    return 0;
  }
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return 1;
  const int status = test_ewald_parity_and_peer_transaction(stream);
  (void)cudaStreamDestroy(stream);
  return status;
}
