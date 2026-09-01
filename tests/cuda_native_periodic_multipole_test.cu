// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_native_periodic_multipole.cuh"
#include "backends/cuda/periodic_topology.cuh"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/periodic_multipole.hpp"
#include "runtime/backend.hpp"

#define CHECK(condition)                                                                      \
  do {                                                                                        \
    if (!(condition)) {                                                                       \
      std::cerr << "CUDA native periodic multipole check failed at line " << __LINE__ << ": " \
                << #condition << '\n';                                                        \
      return __LINE__;                                                                        \
    }                                                                                         \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using xtbloom::detail::cuda::Gfn2CudaPeriodicTopology;
using xtbloom::detail::cuda::Gfn2CudaPeriodicTopologyInput;
using xtbloom::detail::cuda::Gfn2CudaPeriodicTranslation;
using xtbloom::detail::cuda::Gfn2NativePeriodicMultipoleDeviceBatch;
using xtbloom::detail::cuda::Gfn2NativePeriodicMultipoleDeviceWorkspace;
using xtbloom::detail::gfn2::AES2Plan;
using xtbloom::detail::gfn2::BasisPlan;
using xtbloom::detail::gfn2::PeriodicMultipolePlan;
using xtbloom::detail::gfn2::PeriodicShortRangePlan;

constexpr std::uint64_t kPlanToken = 0x4e41544956454d50ULL;
constexpr double kSentinel = -731.25;

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
    return cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
  }

  cudaError_t upload(const std::vector<T>& values, cudaStream_t stream) {
    if (values.size() > count_) return cudaErrorInvalidValue;
    return cudaMemcpyAsync(data_, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice,
                           stream);
  }

  cudaError_t download(std::vector<T>& values, cudaStream_t stream) const {
    if (values.size() > count_) return cudaErrorInvalidValue;
    return cudaMemcpyAsync(values.data(), data_, values.size() * sizeof(T), cudaMemcpyDeviceToHost,
                           stream);
  }

  T* get() noexcept { return data_; }
  const T* get() const noexcept { return data_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

bool close(double actual, double expected, double tolerance = 2.0e-9) {
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::abs(actual - expected) <= tolerance * (1.0 + std::abs(expected));
}

struct Values {
  std::vector<double> charge_dipole;
  std::vector<double> dipole_dipole;
  std::vector<double> charge_quadrupole;
  std::vector<double> charge_potential;
  std::vector<double> dipole_potential;
  std::vector<double> quadrupole_potential;
  std::vector<double> energy;
  std::vector<double> gradient;
  std::vector<double> strain;
  std::vector<double> coordination_adjoint;

  explicit Values(std::int64_t matrix_elements, std::int64_t atoms, std::int64_t systems)
      : charge_dipole(static_cast<std::size_t>(3 * matrix_elements)),
        dipole_dipole(static_cast<std::size_t>(9 * matrix_elements)),
        charge_quadrupole(static_cast<std::size_t>(6 * matrix_elements)),
        charge_potential(static_cast<std::size_t>(atoms)),
        dipole_potential(static_cast<std::size_t>(3 * atoms)),
        quadrupole_potential(static_cast<std::size_t>(6 * atoms)),
        energy(static_cast<std::size_t>(systems)),
        gradient(static_cast<std::size_t>(3 * atoms)),
        strain(static_cast<std::size_t>(9 * systems)),
        coordination_adjoint(static_cast<std::size_t>(atoms)) {}
};

int test_periodic_multipole_parity_and_peer_isolation(cudaStream_t stream, int device) {
  const std::vector<std::int64_t> atom_offsets{0, 3, 6};
  const std::vector<std::int32_t> atomic_numbers{1, 8, 1, 1, 8, 1};
  const std::vector<double> cells{
      11.7, 0.0, 0.0, 1.1, 12.9, 0.0, -0.7, 0.8, 14.3,
      11.7, 0.0, 0.0, 1.1, 12.9, 0.0, -0.7, 0.8, 14.3,
  };
  const std::vector<double> positions{
      12.10, 13.20, 14.30, -10.45, 0.20, 0.30, 0.10, 1.75, 0.30,
      0.40,  0.30,  0.50,  1.85,   0.30, 0.50, 0.40, 1.80, 0.50,
  };
  const std::vector<double> coordination{0.70, 1.15, 0.85, 0.70, 1.15, 0.85};
  const std::vector<double> charges{0.15, -0.20, 0.05, 0.15, -0.20, 0.05};
  const std::vector<double> dipoles{
      0.030, -0.020, 0.010, -0.015, 0.025, 0.005, 0.010, 0.004, -0.012,
      0.030, -0.020, 0.010, -0.015, 0.025, 0.005, 0.010, 0.004, -0.012,
  };
  const std::vector<double> quadrupoles{
      0.010,  0.002,  -0.006, 0.003, -0.001, 0.004,  -0.008, 0.001,  0.006,  -0.002, 0.003,  0.005,
      0.004,  -0.002, -0.003, 0.010, 0.002,  -0.006, 0.003,  -0.001, 0.004,  -0.008, 0.001,  0.006,
      -0.002, 0.003,  0.005,  0.004, -0.002, -0.003, 0.010,  0.002,  -0.006, 0.003,  -0.001, 0.004,
  };

  std::string error;
  BasisPlan basis;
  AES2Plan aes2;
  PeriodicShortRangePlan topology_plan;
  PeriodicMultipolePlan multipole;
  CHECK(xtbloom::detail::gfn2::make_basis_plan(2, 6, atom_offsets.data(), atomic_numbers.data(),
                                               basis, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::make_aes2_plan(basis, atomic_numbers.data(), aes2, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::make_periodic_short_range_plan(2, 6, atom_offsets.data(),
                                                              cells.data(), topology_plan,
                                                              error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::make_periodic_multipole_plan(aes2, topology_plan, multipole,
                                                            error) == XTBLOOM_STATUS_SUCCESS);

  const std::int64_t matrices = multipole.matrix_elements();
  Values cpu(matrices, multipole.total_atoms(), multipole.batch_size());
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_multipole_cpu(
            multipole, positions.data(), coordination.data(), charges.data(), dipoles.data(),
            quadrupoles.data(), cpu.charge_dipole.data(), cpu.dipole_dipole.data(),
            cpu.charge_quadrupole.data(), cpu.charge_potential.data(), cpu.dipole_potential.data(),
            cpu.quadrupole_potential.data(), cpu.energy.data(), cpu.gradient.data(),
            cpu.strain.data(), cpu.coordination_adjoint.data(), error) == XTBLOOM_STATUS_SUCCESS);

  const std::vector<std::int32_t> periodic_axes{XTBLOOM_PERIODIC_AXES_XYZ,
                                                XTBLOOM_PERIODIC_AXES_XYZ};
  Gfn2CudaPeriodicTopologyInput topology_input{
      2, 6, atom_offsets.data(), cells.data(), periodic_axes.data(), 25.0, kPlanToken, 1u};
  Gfn2CudaPeriodicTopology device_topology;
  CHECK(Gfn2CudaPeriodicTopology::create(topology_input, stream, device_topology).success());
  CHECK(xtbloom::detail::ensure_cuda_gfn2_parameters(device, error));

  std::vector<Gfn2CudaPeriodicTranslation> direct_translations;
  direct_translations.reserve(multipole.direct_translations().size());
  for (const auto& translation : multipole.direct_translations()) {
    Gfn2CudaPeriodicTranslation converted{};
    for (int component = 0; component < 3; ++component) {
      converted.index[component] = translation.index[component];
      converted.cartesian[component] = translation.cartesian[component];
    }
    direct_translations.push_back(converted);
  }
  std::vector<Gfn2CudaPeriodicTranslation> reciprocal_translations;
  reciprocal_translations.reserve(multipole.reciprocal_translations().size());
  for (const auto& translation : multipole.reciprocal_translations()) {
    Gfn2CudaPeriodicTranslation converted{};
    for (int component = 0; component < 3; ++component) {
      converted.index[component] = translation.index[component];
      converted.cartesian[component] = translation.cartesian[component];
    }
    reciprocal_translations.push_back(converted);
  }

  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<double> volumes, alphas;
  DeviceBuffer<std::int64_t> direct_offsets, reciprocal_offsets;
  DeviceBuffer<Gfn2CudaPeriodicTranslation> direct, reciprocal;
  DeviceBuffer<double> dipole_kernel, quadrupole_kernel, radii, valence;
  DeviceBuffer<double> device_positions, device_coordination, device_charges, device_dipoles,
      device_quadrupoles;
  DeviceBuffer<double> wrapped, sd_scratch, dd_scratch, sq_scratch, charge_scratch, dipole_scratch,
      quad_scratch, energy_scratch, gradient_scratch, strain_scratch, adjoint_scratch;
  DeviceBuffer<double> sd, dd, sq, charge_potential, dipole_potential, quadrupole_potential, energy,
      gradient, strain, adjoint;
  DeviceBuffer<std::uint32_t> system_errors, device_error;

  CHECK(matrix_offsets.allocate(3) == cudaSuccess);
  CHECK(volumes.allocate(2) == cudaSuccess);
  CHECK(alphas.allocate(2) == cudaSuccess);
  CHECK(direct_offsets.allocate(3) == cudaSuccess);
  CHECK(reciprocal_offsets.allocate(3) == cudaSuccess);
  CHECK(direct.allocate(direct_translations.size()) == cudaSuccess);
  CHECK(reciprocal.allocate(reciprocal_translations.size()) == cudaSuccess);
  CHECK(dipole_kernel.allocate(6) == cudaSuccess);
  CHECK(quadrupole_kernel.allocate(6) == cudaSuccess);
  CHECK(radii.allocate(6) == cudaSuccess);
  CHECK(valence.allocate(6) == cudaSuccess);
  CHECK(device_positions.allocate(18) == cudaSuccess);
  CHECK(device_coordination.allocate(6) == cudaSuccess);
  CHECK(device_charges.allocate(6) == cudaSuccess);
  CHECK(device_dipoles.allocate(18) == cudaSuccess);
  CHECK(device_quadrupoles.allocate(36) == cudaSuccess);
  CHECK(wrapped.allocate(18) == cudaSuccess);
  CHECK(sd_scratch.allocate(static_cast<std::size_t>(3 * matrices)) == cudaSuccess);
  CHECK(dd_scratch.allocate(static_cast<std::size_t>(9 * matrices)) == cudaSuccess);
  CHECK(sq_scratch.allocate(static_cast<std::size_t>(6 * matrices)) == cudaSuccess);
  CHECK(charge_scratch.allocate(6) == cudaSuccess);
  CHECK(dipole_scratch.allocate(18) == cudaSuccess);
  CHECK(quad_scratch.allocate(36) == cudaSuccess);
  CHECK(energy_scratch.allocate(2) == cudaSuccess);
  CHECK(gradient_scratch.allocate(18) == cudaSuccess);
  CHECK(strain_scratch.allocate(18) == cudaSuccess);
  CHECK(adjoint_scratch.allocate(6) == cudaSuccess);
  CHECK(sd.allocate(static_cast<std::size_t>(3 * matrices)) == cudaSuccess);
  CHECK(dd.allocate(static_cast<std::size_t>(9 * matrices)) == cudaSuccess);
  CHECK(sq.allocate(static_cast<std::size_t>(6 * matrices)) == cudaSuccess);
  CHECK(charge_potential.allocate(6) == cudaSuccess);
  CHECK(dipole_potential.allocate(18) == cudaSuccess);
  CHECK(quadrupole_potential.allocate(36) == cudaSuccess);
  CHECK(energy.allocate(2) == cudaSuccess);
  CHECK(gradient.allocate(18) == cudaSuccess);
  CHECK(strain.allocate(18) == cudaSuccess);
  CHECK(adjoint.allocate(6) == cudaSuccess);
  CHECK(system_errors.allocate(2) == cudaSuccess);
  CHECK(device_error.allocate(1) == cudaSuccess);

  std::vector<double> volume_values{multipole.lattice(0).volume, multipole.lattice(1).volume};
  std::vector<double> alpha_values{multipole.alpha(0), multipole.alpha(1)};
  CHECK(matrix_offsets.upload(multipole.matrix_offsets(), stream) == cudaSuccess);
  CHECK(volumes.upload(volume_values, stream) == cudaSuccess);
  CHECK(alphas.upload(alpha_values, stream) == cudaSuccess);
  CHECK(direct_offsets.upload(multipole.direct_translation_offsets(), stream) == cudaSuccess);
  CHECK(reciprocal_offsets.upload(multipole.reciprocal_translation_offsets(), stream) ==
        cudaSuccess);
  CHECK(direct.upload(direct_translations, stream) == cudaSuccess);
  CHECK(reciprocal.upload(reciprocal_translations, stream) == cudaSuccess);
  CHECK(dipole_kernel.upload(aes2.dipole_kernel(), stream) == cudaSuccess);
  CHECK(quadrupole_kernel.upload(aes2.quadrupole_kernel(), stream) == cudaSuccess);
  CHECK(radii.upload(aes2.multipole_radius(), stream) == cudaSuccess);
  CHECK(valence.upload(aes2.multipole_valence_cn(), stream) == cudaSuccess);
  CHECK(device_positions.upload(positions, stream) == cudaSuccess);
  std::vector<double> poisoned_coordination = coordination;
  poisoned_coordination[3] = std::numeric_limits<double>::quiet_NaN();
  CHECK(device_coordination.upload(poisoned_coordination, stream) == cudaSuccess);
  CHECK(device_charges.upload(charges, stream) == cudaSuccess);
  CHECK(device_dipoles.upload(dipoles, stream) == cudaSuccess);
  CHECK(device_quadrupoles.upload(quadrupoles, stream) == cudaSuccess);

  std::vector<double> seeds;
  auto seed = [&](DeviceBuffer<double>& buffer, std::size_t count) {
    seeds.assign(count, kSentinel);
    return buffer.upload(seeds, stream) == cudaSuccess;
  };
  CHECK(seed(sd, static_cast<std::size_t>(3 * matrices)));
  CHECK(seed(dd, static_cast<std::size_t>(9 * matrices)));
  CHECK(seed(sq, static_cast<std::size_t>(6 * matrices)));
  CHECK(seed(charge_potential, 6));
  CHECK(seed(dipole_potential, 18));
  CHECK(seed(quadrupole_potential, 36));
  CHECK(seed(energy, 2));
  CHECK(seed(gradient, 18));
  CHECK(seed(strain, 18));
  CHECK(seed(adjoint, 6));
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);

  const auto topology_view = device_topology.device_view();
  const Gfn2NativePeriodicMultipoleDeviceBatch batch{
      topology_view,
      matrix_offsets.get(),
      3,
      matrices,
      volumes.get(),
      2,
      alphas.get(),
      2,
      direct_offsets.get(),
      3,
      direct.get(),
      static_cast<std::int64_t>(direct_translations.size()),
      reciprocal_offsets.get(),
      3,
      reciprocal.get(),
      static_cast<std::int64_t>(reciprocal_translations.size()),
      dipole_kernel.get(),
      6,
      quadrupole_kernel.get(),
      6,
      radii.get(),
      6,
      valence.get(),
      6,
      device_positions.get(),
      18,
      device_coordination.get(),
      6,
      device_charges.get(),
      6,
      device_dipoles.get(),
      18,
      device_quadrupoles.get(),
      36,
  };
  const Gfn2NativePeriodicMultipoleDeviceWorkspace workspace{
      wrapped.get(),
      18,
      sd_scratch.get(),
      3 * matrices,
      dd_scratch.get(),
      9 * matrices,
      sq_scratch.get(),
      6 * matrices,
      charge_scratch.get(),
      6,
      dipole_scratch.get(),
      18,
      quad_scratch.get(),
      36,
      energy_scratch.get(),
      2,
      gradient_scratch.get(),
      18,
      strain_scratch.get(),
      18,
      adjoint_scratch.get(),
      6,
      kPlanToken,
  };
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_native_periodic_multipole_errors_cuda(
      2, system_errors.get(), device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_native_periodic_multipole_cuda(
      batch, workspace, sd.get(), dd.get(), sq.get(), charge_potential.get(),
      dipole_potential.get(), quadrupole_potential.get(), energy.get(), gradient.get(),
      strain.get(), adjoint.get(), system_errors.get(), device_error.get(), stream));
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);

  Values gpu(matrices, 6, 2);
  std::vector<std::uint32_t> gpu_system_errors(2), gpu_device_error(1);
  CHECK(sd.download(gpu.charge_dipole, stream) == cudaSuccess);
  CHECK(dd.download(gpu.dipole_dipole, stream) == cudaSuccess);
  CHECK(sq.download(gpu.charge_quadrupole, stream) == cudaSuccess);
  CHECK(charge_potential.download(gpu.charge_potential, stream) == cudaSuccess);
  CHECK(dipole_potential.download(gpu.dipole_potential, stream) == cudaSuccess);
  CHECK(quadrupole_potential.download(gpu.quadrupole_potential, stream) == cudaSuccess);
  CHECK(energy.download(gpu.energy, stream) == cudaSuccess);
  CHECK(gradient.download(gpu.gradient, stream) == cudaSuccess);
  CHECK(strain.download(gpu.strain, stream) == cudaSuccess);
  CHECK(adjoint.download(gpu.coordination_adjoint, stream) == cudaSuccess);
  CHECK(system_errors.download(gpu_system_errors, stream) == cudaSuccess);
  CHECK(device_error.download(gpu_device_error, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);

  const std::size_t first_matrix_count = 9u;
  for (std::size_t i = 0; i < 3u * first_matrix_count; ++i)
    CHECK(close(gpu.charge_dipole[i], cpu.charge_dipole[i]));
  for (std::size_t i = 0; i < 9u * first_matrix_count; ++i)
    CHECK(close(gpu.dipole_dipole[i], cpu.dipole_dipole[i]));
  for (std::size_t i = 0; i < 6u * first_matrix_count; ++i)
    CHECK(close(gpu.charge_quadrupole[i], cpu.charge_quadrupole[i]));
  for (std::size_t i = 0; i < 3u; ++i) {
    CHECK(close(gpu.charge_potential[i], cpu.charge_potential[i]));
    CHECK(close(gpu.energy[0], cpu.energy[0]));
    CHECK(close(gpu.coordination_adjoint[i], cpu.coordination_adjoint[i]));
  }
  for (std::size_t i = 0; i < 9u; ++i) {
    CHECK(close(gpu.dipole_potential[i], cpu.dipole_potential[i]));
    CHECK(close(gpu.strain[i], cpu.strain[i]));
  }
  for (std::size_t i = 0; i < 18u; ++i) {
    CHECK(close(gpu.quadrupole_potential[i], cpu.quadrupole_potential[i]));
  }
  for (std::size_t i = 0; i < 9u; ++i) {
    CHECK(close(gpu.gradient[i], cpu.gradient[i]));
  }
  CHECK(gpu_system_errors[0] == 0u);
  CHECK(gpu_system_errors[1] != 0u);
  CHECK(gpu_device_error[0] == 0u);

  for (std::size_t i = 3u * first_matrix_count; i < gpu.charge_dipole.size(); ++i)
    CHECK(gpu.charge_dipole[i] == kSentinel);
  for (std::size_t i = 9u * first_matrix_count; i < gpu.dipole_dipole.size(); ++i)
    CHECK(gpu.dipole_dipole[i] == kSentinel);
  for (std::size_t i = 6u * first_matrix_count; i < gpu.charge_quadrupole.size(); ++i)
    CHECK(gpu.charge_quadrupole[i] == kSentinel);
  for (std::size_t i = 3u; i < gpu.charge_potential.size(); ++i)
    CHECK(gpu.charge_potential[i] == kSentinel);
  for (std::size_t i = 9u; i < gpu.dipole_potential.size(); ++i)
    CHECK(gpu.dipole_potential[i] == kSentinel);
  for (std::size_t i = 18u; i < gpu.quadrupole_potential.size(); ++i)
    CHECK(gpu.quadrupole_potential[i] == kSentinel);
  CHECK(gpu.energy[1] == kSentinel);
  for (std::size_t i = 9u; i < gpu.gradient.size(); ++i) CHECK(gpu.gradient[i] == kSentinel);
  for (std::size_t i = 9u; i < gpu.strain.size(); ++i) CHECK(gpu.strain[i] == kSentinel);
  for (std::size_t i = 3u; i < gpu.coordination_adjoint.size(); ++i)
    CHECK(gpu.coordination_adjoint[i] == kSentinel);

  /* Structural alias rejection is synchronous and transactional. */
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_native_periodic_multipole_cuda(
            batch, workspace, reinterpret_cast<double*>(system_errors.get()), dd.get(), sq.get(),
            charge_potential.get(), dipole_potential.get(), quadrupole_potential.get(),
            energy.get(), gradient.get(), strain.get(), adjoint.get(), system_errors.get(),
            device_error.get(), stream) == cudaErrorInvalidValue);
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    std::cout << "CUDA native periodic multipole test skipped: "
              << (count_status == cudaSuccess ? "no visible device"
                                              : cudaGetErrorString(count_status))
              << '\n';
    return 0;
  }
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) return 1;
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return 1;
  const int status = test_periodic_multipole_parity_and_peer_isolation(stream, device);
  (void)cudaStreamDestroy(stream);
  return status;
}
