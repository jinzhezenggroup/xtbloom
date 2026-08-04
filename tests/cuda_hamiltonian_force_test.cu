#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_hamiltonian_force.cuh"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/integrals.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using gpuxtb::detail::cuda::Gfn2ForceDeviceActivity;
using gpuxtb::detail::cuda::Gfn2HamiltonianDeviceBatch;
using gpuxtb::detail::cuda::Gfn2HamiltonianForceDeviceInput;
using gpuxtb::detail::cuda::Gfn2HamiltonianForceDeviceOutput;
using gpuxtb::detail::cuda::Gfn2HamiltonianForceDeviceWorkspace;
using gpuxtb::detail::gfn2::BasisPlan;
using gpuxtb::detail::gfn2::IntegralPlan;

constexpr std::uint64_t kPlanToken = 0x643cec739a68e512ULL;

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
    count_ = std::max<std::size_t>(count, 1u);
    return cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T));
  }
  cudaError_t copy_from(const T* source, std::size_t count, cudaStream_t stream = nullptr) {
    return source == nullptr || count > count_
               ? cudaErrorInvalidValue
               : cudaMemcpyAsync(data_, source, count * sizeof(T), cudaMemcpyHostToDevice, stream);
  }
  cudaError_t copy_to(T* target, std::size_t count, cudaStream_t stream = nullptr) const {
    return target == nullptr || count > count_
               ? cudaErrorInvalidValue
               : cudaMemcpyAsync(target, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
  }
  T* get() { return data_; }
  const T* get() const { return data_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

template <typename T>
cudaError_t upload(DeviceBuffer<T>& target, const std::vector<T>& source, cudaStream_t stream) {
  cudaError_t status = target.allocate(source.size());
  return status == cudaSuccess ? target.copy_from(source.data(), source.size(), stream) : status;
}

bool near(double actual, double expected, double tolerance = 3.0e-13) {
  return std::abs(actual - expected) <=
         tolerance * (1.0 + std::max(std::abs(actual), std::abs(expected)));
}

struct HostCase {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t total_matrices = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> batch_shell_offsets;
  std::vector<std::int64_t> batch_orbital_offsets;
  std::vector<std::int64_t> matrix_offsets;
  std::vector<std::int64_t> atom_shell_offsets;
  std::vector<std::int64_t> shell_orbital_offsets;
  std::vector<std::int64_t> shell_to_atom;
  std::vector<std::int64_t> orbital_to_shell;
  std::vector<std::int64_t> orbital_to_atom;
  std::vector<double> density;
  std::vector<double> spin_density;
  std::vector<double> shell_scalar;
  std::vector<double> spin_shell_scalar;
  std::vector<double> dipole_potential;
  std::vector<double> quadrupole_potential;
  std::vector<double> overlap_seed;
  std::vector<double> dipole_seed;
  std::vector<double> quadrupole_seed;
};

bool make_case(HostCase& data, std::string& error) {
  data.atom_offsets = {0, 2, 5};
  const std::vector<std::int32_t> atomic_numbers{1, 1, 8, 1, 1};
  BasisPlan basis;
  IntegralPlan integrals;
  if (gpuxtb::detail::gfn2::make_basis_plan(2, 5, data.atom_offsets.data(), atomic_numbers.data(),
                                            basis, error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::make_integral_plan(basis, integrals, error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  data.batch_size = basis.batch_size;
  data.total_atoms = basis.total_atoms;
  data.total_shells = basis.total_shells;
  data.total_orbitals = basis.total_orbitals;
  data.total_matrices = integrals.total_matrix_elements;
  data.batch_shell_offsets = basis.batch_shell_offsets;
  data.batch_orbital_offsets = basis.batch_orbital_offsets;
  data.matrix_offsets = integrals.matrix_offsets;
  data.atom_shell_offsets = basis.atom_shell_offsets;
  data.shell_orbital_offsets = basis.shell_orbital_offsets;
  data.shell_to_atom = basis.shell_to_atom;
  data.orbital_to_shell.resize(static_cast<std::size_t>(basis.total_orbitals));
  data.orbital_to_atom.resize(static_cast<std::size_t>(basis.total_orbitals));
  for (std::int64_t shell = 0; shell < basis.total_shells; ++shell) {
    for (std::int64_t orbital = basis.shell_orbital_offsets[static_cast<std::size_t>(shell)];
         orbital < basis.shell_orbital_offsets[static_cast<std::size_t>(shell + 1)]; ++orbital) {
      data.orbital_to_shell[static_cast<std::size_t>(orbital)] = shell;
      data.orbital_to_atom[static_cast<std::size_t>(orbital)] =
          basis.shell_to_atom[static_cast<std::size_t>(shell)];
    }
  }
  const std::size_t matrices = static_cast<std::size_t>(data.total_matrices);
  data.density.resize(matrices);
  data.spin_density.resize(matrices);
  data.shell_scalar.resize(static_cast<std::size_t>(data.total_shells));
  data.spin_shell_scalar.resize(static_cast<std::size_t>(data.total_shells));
  data.dipole_potential.resize(static_cast<std::size_t>(3 * data.total_atoms));
  data.quadrupole_potential.resize(static_cast<std::size_t>(6 * data.total_atoms));
  data.overlap_seed.resize(matrices);
  data.dipole_seed.resize(3u * matrices);
  data.quadrupole_seed.resize(6u * matrices);
  for (std::size_t index = 0; index < matrices; ++index) {
    data.density[index] = -0.17 + 0.011 * static_cast<double>((index * 13u) % 37u);
    data.spin_density[index] = 0.12 - 0.009 * static_cast<double>((index * 5u) % 29u);
    data.overlap_seed[index] = 0.019 - 0.002 * static_cast<double>((index * 7u) % 11u);
  }
  for (std::size_t index = 0; index < data.shell_scalar.size(); ++index) {
    data.shell_scalar[index] = -0.23 + 0.037 * static_cast<double>(index % 13u);
    data.spin_shell_scalar[index] = 0.16 - 0.021 * static_cast<double>(index % 11u);
  }
  for (std::size_t index = 0; index < data.dipole_potential.size(); ++index) {
    data.dipole_potential[index] = 0.09 - 0.014 * static_cast<double>(index % 17u);
  }
  for (std::size_t index = 0; index < data.quadrupole_potential.size(); ++index) {
    data.quadrupole_potential[index] = -0.07 + 0.008 * static_cast<double>(index % 19u);
  }
  for (std::size_t index = 0; index < data.dipole_seed.size(); ++index) {
    data.dipole_seed[index] = -0.013 + 0.001 * static_cast<double>(index % 23u);
  }
  for (std::size_t index = 0; index < data.quadrupole_seed.size(); ++index) {
    data.quadrupole_seed[index] = 0.007 - 0.0004 * static_cast<double>(index % 29u);
  }
  return true;
}

struct Adjoints {
  std::vector<double> overlap;
  std::vector<double> dipole;
  std::vector<double> quadrupole;
};

Adjoints reference(const HostCase& data) {
  Adjoints result{data.overlap_seed, data.dipole_seed, data.quadrupole_seed};
  for (std::int64_t system = 0; system < data.batch_size; ++system) {
    const std::size_t s = static_cast<std::size_t>(system);
    const std::int64_t orbital_begin = data.batch_orbital_offsets[s];
    const std::int64_t orbitals = data.batch_orbital_offsets[s + 1u] - orbital_begin;
    const std::int64_t matrix_begin = data.matrix_offsets[s];
    for (std::int64_t local_row = 0; local_row < orbitals; ++local_row) {
      for (std::int64_t local_column = local_row; local_column < orbitals; ++local_column) {
        const std::int64_t row = orbital_begin + local_row;
        const std::int64_t column = orbital_begin + local_column;
        const std::int64_t row_shell = data.orbital_to_shell[static_cast<std::size_t>(row)];
        const std::int64_t column_shell = data.orbital_to_shell[static_cast<std::size_t>(column)];
        const std::int64_t row_atom = data.orbital_to_atom[static_cast<std::size_t>(row)];
        const std::int64_t column_atom = data.orbital_to_atom[static_cast<std::size_t>(column)];
        const std::int64_t forward = matrix_begin + local_row * orbitals + local_column;
        const std::int64_t reverse = matrix_begin + local_column * orbitals + local_row;
        const double pair_density =
            data.density[static_cast<std::size_t>(forward)] +
            (forward == reverse ? 0.0 : data.density[static_cast<std::size_t>(reverse)]);
        result.overlap[static_cast<std::size_t>(forward)] +=
            -0.5 * pair_density *
            (data.shell_scalar[static_cast<std::size_t>(row_shell)] +
             data.shell_scalar[static_cast<std::size_t>(column_shell)]);
        for (std::int64_t component = 0; component < 3; ++component) {
          result.dipole[static_cast<std::size_t>(component * data.total_matrices + forward)] +=
              -0.5 * pair_density *
              data.dipole_potential[static_cast<std::size_t>(3 * column_atom + component)];
          result.dipole[static_cast<std::size_t>(component * data.total_matrices + reverse)] +=
              -0.5 * pair_density *
              data.dipole_potential[static_cast<std::size_t>(3 * row_atom + component)];
        }
        for (std::int64_t component = 0; component < 6; ++component) {
          result.quadrupole[static_cast<std::size_t>(component * data.total_matrices + forward)] +=
              -0.5 * pair_density *
              data.quadrupole_potential[static_cast<std::size_t>(6 * column_atom + component)];
          result.quadrupole[static_cast<std::size_t>(component * data.total_matrices + reverse)] +=
              -0.5 * pair_density *
              data.quadrupole_potential[static_cast<std::size_t>(6 * row_atom + component)];
        }
      }
    }
  }
  return result;
}

Adjoints reference_with_spin(const HostCase& data) {
  Adjoints result = reference(data);
  for (std::int64_t system = 0; system < data.batch_size; ++system) {
    const std::size_t s = static_cast<std::size_t>(system);
    const std::int64_t orbital_begin = data.batch_orbital_offsets[s];
    const std::int64_t orbitals = data.batch_orbital_offsets[s + 1u] - orbital_begin;
    const std::int64_t matrix_begin = data.matrix_offsets[s];
    for (std::int64_t local_row = 0; local_row < orbitals; ++local_row) {
      for (std::int64_t local_column = local_row; local_column < orbitals; ++local_column) {
        const std::int64_t row = orbital_begin + local_row;
        const std::int64_t column = orbital_begin + local_column;
        const std::int64_t row_shell = data.orbital_to_shell[static_cast<std::size_t>(row)];
        const std::int64_t column_shell = data.orbital_to_shell[static_cast<std::size_t>(column)];
        const std::int64_t forward = matrix_begin + local_row * orbitals + local_column;
        const std::int64_t reverse = matrix_begin + local_column * orbitals + local_row;
        const double pair_spin_density =
            data.spin_density[static_cast<std::size_t>(forward)] +
            (forward == reverse ? 0.0 : data.spin_density[static_cast<std::size_t>(reverse)]);
        result.overlap[static_cast<std::size_t>(forward)] +=
            -0.5 * pair_spin_density *
            (data.spin_shell_scalar[static_cast<std::size_t>(row_shell)] +
             data.spin_shell_scalar[static_cast<std::size_t>(column_shell)]);
      }
    }
  }
  return result;
}

double stationary_shift_energy(const HostCase& data, const std::vector<double>& overlap,
                               const std::vector<double>& dipole,
                               const std::vector<double>& quadrupole) {
  double energy = 0.0;
  for (std::int64_t system = 0; system < data.batch_size; ++system) {
    const std::size_t s = static_cast<std::size_t>(system);
    const std::int64_t orbital_begin = data.batch_orbital_offsets[s];
    const std::int64_t orbitals = data.batch_orbital_offsets[s + 1u] - orbital_begin;
    const std::int64_t matrix_begin = data.matrix_offsets[s];
    for (std::int64_t row = 0; row < orbitals; ++row) {
      for (std::int64_t column = row; column < orbitals; ++column) {
        const std::int64_t row_global = orbital_begin + row;
        const std::int64_t column_global = orbital_begin + column;
        const std::int64_t row_shell = data.orbital_to_shell[static_cast<std::size_t>(row_global)];
        const std::int64_t column_shell =
            data.orbital_to_shell[static_cast<std::size_t>(column_global)];
        const std::int64_t row_atom = data.orbital_to_atom[static_cast<std::size_t>(row_global)];
        const std::int64_t column_atom =
            data.orbital_to_atom[static_cast<std::size_t>(column_global)];
        const std::int64_t forward = matrix_begin + row * orbitals + column;
        const std::int64_t reverse = matrix_begin + column * orbitals + row;
        double shift = -0.5 * overlap[static_cast<std::size_t>(forward)] *
                       (data.shell_scalar[static_cast<std::size_t>(row_shell)] +
                        data.shell_scalar[static_cast<std::size_t>(column_shell)]);
        for (std::int64_t component = 0; component < 3; ++component) {
          shift -=
              0.5 *
              (dipole[static_cast<std::size_t>(component * data.total_matrices + forward)] *
                   data.dipole_potential[static_cast<std::size_t>(3 * column_atom + component)] +
               dipole[static_cast<std::size_t>(component * data.total_matrices + reverse)] *
                   data.dipole_potential[static_cast<std::size_t>(3 * row_atom + component)]);
        }
        for (std::int64_t component = 0; component < 6; ++component) {
          shift -=
              0.5 *
              (quadrupole[static_cast<std::size_t>(component * data.total_matrices + forward)] *
                   data.quadrupole_potential[static_cast<std::size_t>(6 * column_atom +
                                                                      component)] +
               quadrupole[static_cast<std::size_t>(component * data.total_matrices + reverse)] *
                   data.quadrupole_potential[static_cast<std::size_t>(6 * row_atom + component)]);
        }
        const double pair_density =
            data.density[static_cast<std::size_t>(forward)] +
            (forward == reverse ? 0.0 : data.density[static_cast<std::size_t>(reverse)]);
        energy += pair_density * shift;
      }
    }
  }
  return energy;
}

struct DeviceFixture {
  DeviceBuffer<std::int64_t> atom_offsets, batch_shell_offsets, batch_orbital_offsets;
  DeviceBuffer<std::int64_t> matrix_offsets, atom_shell_offsets, shell_orbital_offsets;
  DeviceBuffer<std::int64_t> shell_to_atom, orbital_to_shell, orbital_to_atom;
  DeviceBuffer<double> density, spin_density, shell_scalar, spin_shell_scalar, dipole_potential,
      quadrupole_potential;
  DeviceBuffer<std::uint8_t> requested;
  DeviceBuffer<gpuxtb_status_t> statuses;
  DeviceBuffer<double> overlap, dipole, quadrupole, overlap_scratch, dipole_scratch,
      quadrupole_scratch;
  DeviceBuffer<std::uint32_t> sequence_active, system_errors, device_error;

  cudaError_t initialize(const HostCase& host, cudaStream_t stream) {
    cudaError_t status = upload(atom_offsets, host.atom_offsets, stream);
#define UPLOAD(field, source)               \
  if (status == cudaSuccess) {              \
    status = upload(field, source, stream); \
  }
    UPLOAD(batch_shell_offsets, host.batch_shell_offsets)
    UPLOAD(batch_orbital_offsets, host.batch_orbital_offsets)
    UPLOAD(matrix_offsets, host.matrix_offsets)
    UPLOAD(atom_shell_offsets, host.atom_shell_offsets)
    UPLOAD(shell_orbital_offsets, host.shell_orbital_offsets)
    UPLOAD(shell_to_atom, host.shell_to_atom)
    UPLOAD(orbital_to_shell, host.orbital_to_shell)
    UPLOAD(orbital_to_atom, host.orbital_to_atom)
    UPLOAD(density, host.density)
    UPLOAD(spin_density, host.spin_density)
    UPLOAD(shell_scalar, host.shell_scalar)
    UPLOAD(spin_shell_scalar, host.spin_shell_scalar)
    UPLOAD(dipole_potential, host.dipole_potential)
    UPLOAD(quadrupole_potential, host.quadrupole_potential)
#undef UPLOAD
    const std::size_t matrices = static_cast<std::size_t>(host.total_matrices);
#define ALLOCATE(field, count)        \
  if (status == cudaSuccess) {        \
    status = field.allocate((count)); \
  }
    ALLOCATE(requested, static_cast<std::size_t>(host.batch_size))
    ALLOCATE(statuses, static_cast<std::size_t>(host.batch_size))
    ALLOCATE(overlap, matrices)
    ALLOCATE(dipole, 3u * matrices)
    ALLOCATE(quadrupole, 6u * matrices)
    ALLOCATE(overlap_scratch, matrices)
    ALLOCATE(dipole_scratch, 3u * matrices)
    ALLOCATE(quadrupole_scratch, 6u * matrices)
    ALLOCATE(sequence_active, 1u)
    ALLOCATE(system_errors, static_cast<std::size_t>(host.batch_size))
    ALLOCATE(device_error, 1u)
#undef ALLOCATE
    return status;
  }

  Gfn2HamiltonianDeviceBatch batch(const HostCase& h) const {
    return {h.batch_size,
            h.total_atoms,
            h.total_shells,
            h.total_orbitals,
            h.total_matrices,
            kPlanToken,
            static_cast<std::int64_t>(h.atom_offsets.size()),
            static_cast<std::int64_t>(h.batch_shell_offsets.size()),
            static_cast<std::int64_t>(h.batch_orbital_offsets.size()),
            static_cast<std::int64_t>(h.matrix_offsets.size()),
            static_cast<std::int64_t>(h.atom_shell_offsets.size()),
            static_cast<std::int64_t>(h.shell_orbital_offsets.size()),
            static_cast<std::int64_t>(h.shell_to_atom.size()),
            static_cast<std::int64_t>(h.orbital_to_shell.size()),
            static_cast<std::int64_t>(h.orbital_to_atom.size()),
            atom_offsets.get(),
            batch_shell_offsets.get(),
            batch_orbital_offsets.get(),
            matrix_offsets.get(),
            atom_shell_offsets.get(),
            shell_orbital_offsets.get(),
            shell_to_atom.get(),
            orbital_to_shell.get(),
            orbital_to_atom.get()};
  }
  Gfn2ForceDeviceActivity activity(const HostCase& h) const {
    return {requested.get(), statuses.get(), h.batch_size, kPlanToken};
  }
  Gfn2HamiltonianForceDeviceInput input(const HostCase& h) const {
    return {density.get(),
            h.total_matrices,
            shell_scalar.get(),
            h.total_shells,
            dipole_potential.get(),
            3 * h.total_atoms,
            quadrupole_potential.get(),
            6 * h.total_atoms,
            kPlanToken};
  }
  Gfn2HamiltonianForceDeviceInput spin_input(const HostCase& h) const {
    Gfn2HamiltonianForceDeviceInput result = input(h);
    result.spin_density = spin_density.get();
    result.spin_density_elements = h.total_matrices;
    result.spin_shell_scalar_potentials = spin_shell_scalar.get();
    result.spin_shell_scalar_elements = h.total_shells;
    return result;
  }
  Gfn2HamiltonianForceDeviceOutput output(const HostCase& h) {
    return {overlap.get(),    h.total_matrices,     dipole.get(), 3 * h.total_matrices,
            quadrupole.get(), 6 * h.total_matrices, kPlanToken};
  }
  Gfn2HamiltonianForceDeviceWorkspace workspace(const HostCase& h) {
    return {overlap_scratch.get(),
            h.total_matrices,
            dipole_scratch.get(),
            3 * h.total_matrices,
            quadrupole_scratch.get(),
            6 * h.total_matrices,
            sequence_active.get(),
            1,
            kPlanToken};
  }
  cudaError_t seed(const HostCase& h, cudaStream_t stream) {
    cudaError_t status = overlap.copy_from(h.overlap_seed.data(), h.overlap_seed.size(), stream);
    if (status == cudaSuccess) {
      status = dipole.copy_from(h.dipole_seed.data(), h.dipole_seed.size(), stream);
    }
    return status == cudaSuccess
               ? quadrupole.copy_from(h.quadrupole_seed.data(), h.quadrupole_seed.size(), stream)
               : status;
  }
};

cudaError_t launch(DeviceFixture& device, const HostCase& host, cudaStream_t stream,
                   bool spin = false) {
  cudaError_t status = gpuxtb::detail::cuda::reset_gfn2_hamiltonian_force_device_errors_cuda(
      host.batch_size, device.system_errors.get(), device.device_error.get(), stream);
  return status == cudaSuccess ? gpuxtb::detail::cuda::add_gfn2_hamiltonian_integral_adjoints_cuda(
                                     device.batch(host), device.activity(host),
                                     spin ? device.spin_input(host) : device.input(host),
                                     device.output(host), device.workspace(host),
                                     device.system_errors.get(), device.device_error.get(), stream)
                               : status;
}

int compare_device(DeviceFixture& device, const HostCase& host, const Adjoints& expected,
                   cudaStream_t stream) {
  Adjoints actual;
  actual.overlap.resize(expected.overlap.size());
  actual.dipole.resize(expected.dipole.size());
  actual.quadrupole.resize(expected.quadrupole.size());
  std::vector<std::uint32_t> errors(static_cast<std::size_t>(host.batch_size));
  std::uint32_t diagnostic = 0u;
  CUDA_CHECK(device.overlap.copy_to(actual.overlap.data(), actual.overlap.size(), stream));
  CUDA_CHECK(device.dipole.copy_to(actual.dipole.data(), actual.dipole.size(), stream));
  CUDA_CHECK(device.quadrupole.copy_to(actual.quadrupole.data(), actual.quadrupole.size(), stream));
  CUDA_CHECK(device.system_errors.copy_to(errors.data(), errors.size(), stream));
  CUDA_CHECK(device.device_error.copy_to(&diagnostic, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(diagnostic == 0u);
  CHECK(std::all_of(errors.begin(), errors.end(), [](std::uint32_t value) { return value == 0u; }));
  for (std::size_t index = 0; index < actual.overlap.size(); ++index) {
    CHECK(near(actual.overlap[index], expected.overlap[index]));
  }
  for (std::size_t index = 0; index < actual.dipole.size(); ++index) {
    CHECK(near(actual.dipole[index], expected.dipole[index]));
  }
  for (std::size_t index = 0; index < actual.quadrupole.size(); ++index) {
    CHECK(near(actual.quadrupole[index], expected.quadrupole[index]));
  }
  return 0;
}

int test_reference_finite_difference_and_graph() {
  HostCase host;
  std::string error;
  CHECK(make_case(host, error));
  const Adjoints expected = reference(host);
  std::vector<double> overlap(host.overlap_seed.size(), 0.0);
  std::vector<double> dipole(host.dipole_seed.size(), 0.0);
  std::vector<double> quadrupole(host.quadrupole_seed.size(), 0.0);
  constexpr double step = 1.0e-6;
  overlap[1] += step;
  const double overlap_right = stationary_shift_energy(host, overlap, dipole, quadrupole);
  overlap[1] -= 2.0 * step;
  const double overlap_left = stationary_shift_energy(host, overlap, dipole, quadrupole);
  CHECK(near((overlap_right - overlap_left) / (2.0 * step),
             expected.overlap[1] - host.overlap_seed[1], 3.0e-10));
  overlap[1] += step;
  const std::size_t dipole_probe = static_cast<std::size_t>(host.total_matrices + 1);
  dipole[dipole_probe] += step;
  const double dipole_right = stationary_shift_energy(host, overlap, dipole, quadrupole);
  dipole[dipole_probe] -= 2.0 * step;
  const double dipole_left = stationary_shift_energy(host, overlap, dipole, quadrupole);
  CHECK(near((dipole_right - dipole_left) / (2.0 * step),
             expected.dipole[dipole_probe] - host.dipole_seed[dipole_probe], 3.0e-10));
  dipole[dipole_probe] += step;
  const std::size_t quadrupole_probe = static_cast<std::size_t>(4 * host.total_matrices);
  quadrupole[quadrupole_probe] += step;
  const double quadrupole_right = stationary_shift_energy(host, overlap, dipole, quadrupole);
  quadrupole[quadrupole_probe] -= 2.0 * step;
  const double quadrupole_left = stationary_shift_energy(host, overlap, dipole, quadrupole);
  CHECK(near((quadrupole_right - quadrupole_left) / (2.0 * step),
             expected.quadrupole[quadrupole_probe] - host.quadrupole_seed[quadrupole_probe],
             3.0e-10));

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  const std::vector<std::uint8_t> requested(2u, 1u);
  const std::vector<gpuxtb_status_t> statuses(2u, GPUXTB_STATUS_SUCCESS);
  CUDA_CHECK(device.requested.copy_from(requested.data(), requested.size(), stream));
  CUDA_CHECK(device.statuses.copy_from(statuses.data(), statuses.size(), stream));
  CUDA_CHECK(device.seed(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(launch(device, host, stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CHECK(compare_device(device, host, expected, stream) == 0);
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_status_gate_skips_poisoned_peer() {
  HostCase host;
  std::string error;
  CHECK(make_case(host, error));
  Adjoints expected = reference(host);
  const std::int64_t matrix_begin = host.matrix_offsets[1];
  const std::int64_t matrix_end = host.matrix_offsets[2];
  std::copy(host.overlap_seed.begin() + matrix_begin, host.overlap_seed.begin() + matrix_end,
            expected.overlap.begin() + matrix_begin);
  for (std::int64_t component = 0; component < 3; ++component) {
    std::copy(host.dipole_seed.begin() + component * host.total_matrices + matrix_begin,
              host.dipole_seed.begin() + component * host.total_matrices + matrix_end,
              expected.dipole.begin() + component * host.total_matrices + matrix_begin);
  }
  for (std::int64_t component = 0; component < 6; ++component) {
    std::copy(host.quadrupole_seed.begin() + component * host.total_matrices + matrix_begin,
              host.quadrupole_seed.begin() + component * host.total_matrices + matrix_end,
              expected.quadrupole.begin() + component * host.total_matrices + matrix_begin);
  }
  std::vector<double> poisoned = host.density;
  std::fill(poisoned.begin() + matrix_begin, poisoned.begin() + matrix_end,
            std::numeric_limits<double>::quiet_NaN());
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  const std::vector<std::uint8_t> requested{1u, 1u};
  const std::vector<gpuxtb_status_t> statuses{GPUXTB_STATUS_SUCCESS,
                                              GPUXTB_STATUS_SCC_NOT_CONVERGED};
  CUDA_CHECK(device.requested.copy_from(requested.data(), requested.size(), stream));
  CUDA_CHECK(device.statuses.copy_from(statuses.data(), statuses.size(), stream));
  CUDA_CHECK(device.density.copy_from(poisoned.data(), poisoned.size(), stream));
  CUDA_CHECK(device.seed(host, stream));
  CUDA_CHECK(launch(device, host, stream));
  CHECK(compare_device(device, host, expected, stream) == 0);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_unrestricted_overlap_response_and_binding_validation() {
  HostCase host;
  std::string error;
  CHECK(make_case(host, error));
  const Adjoints expected = reference_with_spin(host);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  const std::vector<std::uint8_t> requested(2u, 1u);
  const std::vector<gpuxtb_status_t> statuses(2u, GPUXTB_STATUS_SUCCESS);
  CUDA_CHECK(device.requested.copy_from(requested.data(), requested.size(), stream));
  CUDA_CHECK(device.statuses.copy_from(statuses.data(), statuses.size(), stream));
  CUDA_CHECK(device.seed(host, stream));
  CUDA_CHECK(launch(device, host, stream, true));
  CHECK(compare_device(device, host, expected, stream) == 0);

  Gfn2HamiltonianForceDeviceInput invalid = device.spin_input(host);
  invalid.spin_shell_scalar_potentials = nullptr;
  CHECK(gpuxtb::detail::cuda::add_gfn2_hamiltonian_integral_adjoints_cuda(
            device.batch(host), device.activity(host), invalid, device.output(host),
            device.workspace(host), device.system_errors.get(), device.device_error.get(),
            stream) == cudaErrorInvalidValue);
  invalid = device.spin_input(host);
  invalid.spin_density_elements = host.total_matrices - 1;
  CHECK(gpuxtb::detail::cuda::add_gfn2_hamiltonian_integral_adjoints_cuda(
            device.batch(host), device.activity(host), invalid, device.output(host),
            device.workspace(host), device.system_errors.get(), device.device_error.get(),
            stream) == cudaErrorInvalidValue);
  invalid = device.spin_input(host);
  invalid.spin_density = device.overlap.get();
  CHECK(gpuxtb::detail::cuda::add_gfn2_hamiltonian_integral_adjoints_cuda(
            device.batch(host), device.activity(host), invalid, device.output(host),
            device.workspace(host), device.system_errors.get(), device.device_error.get(),
            stream) == cudaErrorInvalidValue);

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_reference_finite_difference_and_graph(); status != 0) {
    return status;
  }
  if (const int status = test_status_gate_skips_poisoned_peer(); status != 0) {
    return status;
  }
  return test_unrestricted_overlap_response_and_binding_validation();
}
