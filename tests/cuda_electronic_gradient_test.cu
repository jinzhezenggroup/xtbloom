#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_electronic_gradient.cuh"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/h0.hpp"
#include "model/gfn2/integrals.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using xtbloom::detail::cuda::Gfn2ElectronicGradientDeviceDiagnostics;
using xtbloom::detail::cuda::Gfn2ElectronicGradientDeviceWorkspace;
using xtbloom::detail::cuda::Gfn2ElectronicGradientRequest;
using xtbloom::detail::cuda::Gfn2ForceDeviceActivity;
using xtbloom::detail::cuda::Gfn2H0DevicePlan;
using xtbloom::detail::cuda::Gfn2H0ForceDeviceInput;
using xtbloom::detail::cuda::Gfn2H0ForceDeviceOutput;
using xtbloom::detail::cuda::Gfn2H0ForceDeviceWorkspace;
using xtbloom::detail::cuda::Gfn2HamiltonianDeviceBatch;
using xtbloom::detail::cuda::Gfn2HamiltonianForceDeviceInput;
using xtbloom::detail::cuda::Gfn2HamiltonianForceDeviceOutput;
using xtbloom::detail::cuda::Gfn2HamiltonianForceDeviceWorkspace;
using xtbloom::detail::cuda::Gfn2IntegralDeviceBatch;
using xtbloom::detail::cuda::Gfn2IntegralForceDeviceWorkspace;
using xtbloom::detail::gfn2::BasisPlan;
using xtbloom::detail::gfn2::H0Plan;
using xtbloom::detail::gfn2::IntegralPlan;

constexpr std::uint64_t kPlanToken = 0x2786dcb5974f2013ULL;

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

  cudaError_t copy_from(const T* source, std::size_t count, cudaStream_t stream) {
    return source == nullptr || count > count_
               ? cudaErrorInvalidValue
               : cudaMemcpyAsync(data_, source, count * sizeof(T), cudaMemcpyHostToDevice, stream);
  }

  cudaError_t copy_to(T* target, std::size_t count, cudaStream_t stream) const {
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

bool near(double actual, double expected, double absolute = 4.0e-10, double relative = 4.0e-10) {
  return std::abs(actual - expected) <=
         absolute + relative * std::max(std::abs(actual), std::abs(expected));
}

struct Expected {
  std::vector<double> overlap;
  std::vector<double> coordination;
  std::vector<double> dipole;
  std::vector<double> quadrupole;
  std::vector<double> gradient;
};

struct HostCase {
  BasisPlan basis;
  IntegralPlan integrals;
  H0Plan h0;
  std::int64_t maximum_system_shells = 0;
  std::vector<std::int64_t> orbital_to_shell;
  std::vector<std::int64_t> orbital_to_atom;
  std::vector<double> positions;
  std::vector<double> coordination;
  std::vector<double> overlap;
  std::vector<double> density;
  std::vector<double> weighted_density;
  std::vector<double> shell_scalar;
  std::vector<double> dipole_potential;
  std::vector<double> quadrupole_potential;
  std::vector<double> overlap_seed;
  std::vector<double> coordination_seed;
  std::vector<double> dipole_seed;
  std::vector<double> quadrupole_seed;
  std::vector<double> gradient_seed;
  Expected expected;
};

void add_hamiltonian_adjoints(const HostCase& data, Expected& result) {
  const std::int64_t matrices = data.integrals.total_matrix_elements;
  for (std::int64_t system = 0; system < data.basis.batch_size; ++system) {
    const std::size_t s = static_cast<std::size_t>(system);
    const std::int64_t orbital_begin = data.basis.batch_orbital_offsets[s];
    const std::int64_t orbitals = data.basis.batch_orbital_offsets[s + 1u] - orbital_begin;
    const std::int64_t matrix_begin = data.integrals.matrix_offsets[s];
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
          result.dipole[static_cast<std::size_t>(component * matrices + forward)] +=
              -0.5 * pair_density *
              data.dipole_potential[static_cast<std::size_t>(3 * column_atom + component)];
          result.dipole[static_cast<std::size_t>(component * matrices + reverse)] +=
              -0.5 * pair_density *
              data.dipole_potential[static_cast<std::size_t>(3 * row_atom + component)];
        }
        for (std::int64_t component = 0; component < 6; ++component) {
          result.quadrupole[static_cast<std::size_t>(component * matrices + forward)] +=
              -0.5 * pair_density *
              data.quadrupole_potential[static_cast<std::size_t>(6 * column_atom + component)];
          result.quadrupole[static_cast<std::size_t>(component * matrices + reverse)] +=
              -0.5 * pair_density *
              data.quadrupole_potential[static_cast<std::size_t>(6 * row_atom + component)];
        }
      }
    }
  }
}

bool update_expected(HostCase& data, std::string& error) {
  data.expected = Expected{data.overlap_seed, data.coordination_seed, data.dipole_seed,
                           data.quadrupole_seed, data.gradient_seed};
  if (xtbloom::detail::gfn2::add_h0_vjp_cpu(
          data.basis, data.integrals, data.h0, data.positions.data(), data.coordination.data(),
          data.overlap.data(), data.density.data(), data.expected.overlap.data(),
          data.expected.coordination.data(), data.expected.gradient.data(),
          error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  for (std::size_t matrix = 0; matrix < data.expected.overlap.size(); ++matrix) {
    data.expected.overlap[matrix] -= data.weighted_density[matrix];
  }
  add_hamiltonian_adjoints(data, data.expected);

  std::vector<double> workspace((data.integrals.workspace_size_bytes + sizeof(double) - 1u) /
                                sizeof(double));
  return xtbloom::detail::gfn2::add_overlap_gradient_cpu(
             data.basis, data.integrals, data.positions.data(), data.expected.overlap.data(),
             data.expected.gradient.data(), workspace.data(), workspace.size() * sizeof(double),
             error) == XTBLOOM_STATUS_SUCCESS &&
         xtbloom::detail::gfn2::add_multipole_gradient_cpu(
             data.basis, data.integrals, data.positions.data(), data.expected.dipole.data(),
             data.expected.quadrupole.data(), data.expected.gradient.data(), workspace.data(),
             workspace.size() * sizeof(double), error) == XTBLOOM_STATUS_SUCCESS;
}

bool make_case(HostCase& data, std::string& error) {
  const std::vector<std::int64_t> atom_offsets{0, 2, 5};
  const std::vector<std::int32_t> atomic_numbers{1, 1, 8, 1, 1};
  data.positions = {0.00, 0.00, -0.71, 0.00,  0.00, 0.71, 4.10, -0.20,
                    0.13, 5.48, 0.37,  -0.22, 3.53, 1.19, 0.61};
  if (xtbloom::detail::gfn2::make_basis_plan(2, 5, atom_offsets.data(), atomic_numbers.data(),
                                             data.basis, error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_integral_plan(data.basis, data.integrals, error) !=
          XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_h0_plan(data.basis, data.integrals, atomic_numbers.data(),
                                          data.h0, error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  for (std::int64_t system = 0; system < data.basis.batch_size; ++system) {
    const std::size_t s = static_cast<std::size_t>(system);
    data.maximum_system_shells =
        std::max(data.maximum_system_shells,
                 data.basis.batch_shell_offsets[s + 1u] - data.basis.batch_shell_offsets[s]);
  }
  data.orbital_to_shell.resize(static_cast<std::size_t>(data.basis.total_orbitals));
  data.orbital_to_atom.resize(static_cast<std::size_t>(data.basis.total_orbitals));
  for (std::int64_t shell = 0; shell < data.basis.total_shells; ++shell) {
    for (std::int64_t orbital = data.basis.shell_orbital_offsets[static_cast<std::size_t>(shell)];
         orbital < data.basis.shell_orbital_offsets[static_cast<std::size_t>(shell + 1)];
         ++orbital) {
      data.orbital_to_shell[static_cast<std::size_t>(orbital)] = shell;
      data.orbital_to_atom[static_cast<std::size_t>(orbital)] =
          data.basis.shell_to_atom[static_cast<std::size_t>(shell)];
    }
  }

  const std::size_t atoms = static_cast<std::size_t>(data.basis.total_atoms);
  const std::size_t shells = static_cast<std::size_t>(data.basis.total_shells);
  const std::size_t matrices = static_cast<std::size_t>(data.integrals.total_matrix_elements);
  data.coordination.resize(atoms);
  data.overlap.resize(matrices);
  data.density.resize(matrices);
  data.weighted_density.resize(matrices);
  data.shell_scalar.resize(shells);
  data.dipole_potential.resize(3u * atoms);
  data.quadrupole_potential.resize(6u * atoms);
  data.overlap_seed.resize(matrices);
  data.coordination_seed.resize(atoms);
  data.dipole_seed.resize(3u * matrices);
  data.quadrupole_seed.resize(6u * matrices);
  data.gradient_seed.resize(3u * atoms);

  std::vector<double> workspace((data.integrals.workspace_size_bytes + sizeof(double) - 1u) /
                                sizeof(double));
  if (xtbloom::detail::gfn2::evaluate_overlap_cpu(
          data.basis, data.integrals, data.positions.data(), data.overlap.data(), workspace.data(),
          workspace.size() * sizeof(double), error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  for (std::size_t atom = 0; atom < atoms; ++atom) {
    data.coordination[atom] = 0.39 + 0.08 * static_cast<double>((atom * 7u) % 11u);
    data.coordination_seed[atom] = -0.012 + 0.003 * static_cast<double>(atom);
  }
  for (std::size_t shell = 0; shell < shells; ++shell) {
    data.shell_scalar[shell] = -0.21 + 0.031 * static_cast<double>(shell % 13u);
  }
  for (std::size_t index = 0; index < data.dipole_potential.size(); ++index) {
    data.dipole_potential[index] = 0.08 - 0.012 * static_cast<double>(index % 17u);
  }
  for (std::size_t index = 0; index < data.quadrupole_potential.size(); ++index) {
    data.quadrupole_potential[index] = -0.06 + 0.007 * static_cast<double>(index % 19u);
  }
  for (std::int64_t system = 0; system < data.basis.batch_size; ++system) {
    const std::size_t s = static_cast<std::size_t>(system);
    const std::int64_t begin = data.integrals.matrix_offsets[s];
    const std::int64_t orbitals =
        data.basis.batch_orbital_offsets[s + 1u] - data.basis.batch_orbital_offsets[s];
    for (std::int64_t row = 0; row < orbitals; ++row) {
      for (std::int64_t column = 0; column <= row; ++column) {
        const double density =
            0.10 + 0.006 * static_cast<double>((row * 13 + column * 5 + system) % 23);
        const double weighted =
            -0.17 + 0.005 * static_cast<double>((row * 7 + column * 11 + system) % 29);
        data.density[static_cast<std::size_t>(begin + row * orbitals + column)] = density;
        data.density[static_cast<std::size_t>(begin + column * orbitals + row)] = density;
        data.weighted_density[static_cast<std::size_t>(begin + row * orbitals + column)] = weighted;
        data.weighted_density[static_cast<std::size_t>(begin + column * orbitals + row)] = weighted;
      }
    }
  }
  for (std::size_t matrix = 0; matrix < matrices; ++matrix) {
    data.overlap_seed[matrix] = 0.013 - 0.001 * static_cast<double>(matrix % 17u);
  }
  for (std::size_t index = 0; index < data.dipole_seed.size(); ++index) {
    data.dipole_seed[index] = -0.009 + 0.0007 * static_cast<double>(index % 23u);
  }
  for (std::size_t index = 0; index < data.quadrupole_seed.size(); ++index) {
    data.quadrupole_seed[index] = 0.005 - 0.0003 * static_cast<double>(index % 31u);
  }
  for (std::size_t coordinate = 0; coordinate < data.gradient_seed.size(); ++coordinate) {
    data.gradient_seed[coordinate] = 0.004 * std::sin(0.23 * static_cast<double>(coordinate + 1u));
  }
  return update_expected(data, error);
}

struct DeviceFixture {
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> batch_shell_offsets;
  DeviceBuffer<std::int64_t> batch_orbital_offsets;
  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<std::int64_t> shell_pair_offsets;
  DeviceBuffer<std::int64_t> atom_shell_offsets;
  DeviceBuffer<std::int64_t> shell_orbital_offsets;
  DeviceBuffer<std::int64_t> shell_primitive_offsets;
  DeviceBuffer<std::int64_t> shell_to_atom;
  DeviceBuffer<std::int64_t> orbital_to_shell;
  DeviceBuffer<std::int64_t> orbital_to_atom;
  DeviceBuffer<std::uint8_t> angular_momenta;
  DeviceBuffer<double> primitive_exponents;
  DeviceBuffer<double> primitive_coefficients;
  DeviceBuffer<double> atomic_radii;
  DeviceBuffer<double> shell_levels;
  DeviceBuffer<double> shell_coordination_scale;
  DeviceBuffer<double> shell_polynomial;
  DeviceBuffer<double> shell_pair_scale;
  DeviceBuffer<double> positions;
  DeviceBuffer<double> coordination;
  DeviceBuffer<double> overlap;
  DeviceBuffer<double> density;
  DeviceBuffer<double> weighted_density;
  DeviceBuffer<double> shell_scalar;
  DeviceBuffer<double> dipole_potential;
  DeviceBuffer<double> quadrupole_potential;
  DeviceBuffer<double> overlap_adjoint;
  DeviceBuffer<double> coordination_adjoint;
  DeviceBuffer<double> dipole_adjoint;
  DeviceBuffer<double> quadrupole_adjoint;
  DeviceBuffer<double> gradient;
  DeviceBuffer<double> h0_overlap_scratch;
  DeviceBuffer<double> h0_coordination_scratch;
  DeviceBuffer<double> h0_gradient_scratch;
  DeviceBuffer<double> hamiltonian_overlap_scratch;
  DeviceBuffer<double> hamiltonian_dipole_scratch;
  DeviceBuffer<double> hamiltonian_quadrupole_scratch;
  DeviceBuffer<double> integral_gradient_scratch;
  DeviceBuffer<std::uint8_t> requested;
  DeviceBuffer<xtbloom_status_t> statuses;
  DeviceBuffer<std::uint8_t> h0_success_mask;
  DeviceBuffer<std::uint8_t> hamiltonian_success_mask;
  DeviceBuffer<std::uint32_t> h0_sequence;
  DeviceBuffer<std::uint32_t> hamiltonian_sequence;
  DeviceBuffer<std::uint32_t> integral_sequence;
  DeviceBuffer<std::uint32_t> h0_system_errors;
  DeviceBuffer<std::uint32_t> hamiltonian_system_errors;
  DeviceBuffer<std::uint32_t> integral_system_errors;
  DeviceBuffer<std::uint32_t> h0_device_error;
  DeviceBuffer<std::uint32_t> hamiltonian_device_error;
  DeviceBuffer<std::uint32_t> integral_device_error;

  cudaError_t initialize(const HostCase& host, cudaStream_t stream) {
    cudaError_t status = upload(atom_offsets, host.basis.atom_offsets, stream);
#define UPLOAD(field, source)               \
  if (status == cudaSuccess) {              \
    status = upload(field, source, stream); \
  }
    UPLOAD(batch_shell_offsets, host.basis.batch_shell_offsets)
    UPLOAD(batch_orbital_offsets, host.basis.batch_orbital_offsets)
    UPLOAD(matrix_offsets, host.integrals.matrix_offsets)
    UPLOAD(shell_pair_offsets, host.h0.shell_pair_offsets)
    UPLOAD(atom_shell_offsets, host.basis.atom_shell_offsets)
    UPLOAD(shell_orbital_offsets, host.basis.shell_orbital_offsets)
    UPLOAD(shell_primitive_offsets, host.basis.shell_primitive_offsets)
    UPLOAD(shell_to_atom, host.basis.shell_to_atom)
    UPLOAD(orbital_to_shell, host.orbital_to_shell)
    UPLOAD(orbital_to_atom, host.orbital_to_atom)
    UPLOAD(angular_momenta, host.basis.angular_momenta)
    UPLOAD(primitive_exponents, host.basis.primitive_exponents)
    UPLOAD(primitive_coefficients, host.basis.primitive_coefficients)
    UPLOAD(atomic_radii, host.h0.atomic_radii)
    UPLOAD(shell_levels, host.h0.shell_levels)
    UPLOAD(shell_coordination_scale, host.h0.shell_coordination_scale)
    UPLOAD(shell_polynomial, host.h0.shell_polynomial)
    UPLOAD(shell_pair_scale, host.h0.shell_pair_scale)
#undef UPLOAD
    const std::size_t atoms = static_cast<std::size_t>(host.basis.total_atoms);
    const std::size_t systems = static_cast<std::size_t>(host.basis.batch_size);
    const std::size_t matrices = static_cast<std::size_t>(host.integrals.total_matrix_elements);
#define ALLOCATE(field, count)        \
  if (status == cudaSuccess) {        \
    status = field.allocate((count)); \
  }
    ALLOCATE(positions, 3u * atoms)
    ALLOCATE(coordination, atoms)
    ALLOCATE(overlap, matrices)
    ALLOCATE(density, matrices)
    ALLOCATE(weighted_density, matrices)
    ALLOCATE(shell_scalar, static_cast<std::size_t>(host.basis.total_shells))
    ALLOCATE(dipole_potential, 3u * atoms)
    ALLOCATE(quadrupole_potential, 6u * atoms)
    ALLOCATE(overlap_adjoint, matrices)
    ALLOCATE(coordination_adjoint, atoms)
    ALLOCATE(dipole_adjoint, 3u * matrices)
    ALLOCATE(quadrupole_adjoint, 6u * matrices)
    ALLOCATE(gradient, 3u * atoms)
    ALLOCATE(h0_overlap_scratch, matrices)
    ALLOCATE(h0_coordination_scratch, atoms)
    ALLOCATE(h0_gradient_scratch, 3u * atoms)
    ALLOCATE(hamiltonian_overlap_scratch, matrices)
    ALLOCATE(hamiltonian_dipole_scratch, 3u * matrices)
    ALLOCATE(hamiltonian_quadrupole_scratch, 6u * matrices)
    ALLOCATE(integral_gradient_scratch, 3u * atoms)
    ALLOCATE(requested, systems)
    ALLOCATE(statuses, systems)
    ALLOCATE(h0_success_mask, systems)
    ALLOCATE(hamiltonian_success_mask, systems)
    ALLOCATE(h0_sequence, 1u)
    ALLOCATE(hamiltonian_sequence, 1u)
    ALLOCATE(integral_sequence, 1u)
    ALLOCATE(h0_system_errors, systems)
    ALLOCATE(hamiltonian_system_errors, systems)
    ALLOCATE(integral_system_errors, systems)
    ALLOCATE(h0_device_error, 1u)
    ALLOCATE(hamiltonian_device_error, 1u)
    ALLOCATE(integral_device_error, 1u)
#undef ALLOCATE
    if (status != cudaSuccess) {
      return status;
    }
    std::vector<std::uint8_t> all_requested(systems, 1u);
    std::vector<xtbloom_status_t> all_success(systems, XTBLOOM_STATUS_SUCCESS);
    status = requested.copy_from(all_requested.data(), systems, stream);
    if (status == cudaSuccess) {
      status = statuses.copy_from(all_success.data(), systems, stream);
    }
    return status == cudaSuccess ? upload_dynamic(host, stream) : status;
  }

  cudaError_t upload_dynamic(const HostCase& host, cudaStream_t stream) {
    cudaError_t status = positions.copy_from(host.positions.data(), host.positions.size(), stream);
#define COPY(field, source)                                         \
  if (status == cudaSuccess) {                                      \
    status = field.copy_from(source.data(), source.size(), stream); \
  }
    COPY(coordination, host.coordination)
    COPY(overlap, host.overlap)
    COPY(density, host.density)
    COPY(weighted_density, host.weighted_density)
    COPY(shell_scalar, host.shell_scalar)
    COPY(dipole_potential, host.dipole_potential)
    COPY(quadrupole_potential, host.quadrupole_potential)
    COPY(overlap_adjoint, host.overlap_seed)
    COPY(coordination_adjoint, host.coordination_seed)
    COPY(dipole_adjoint, host.dipole_seed)
    COPY(quadrupole_adjoint, host.quadrupole_seed)
    COPY(gradient, host.gradient_seed)
#undef COPY
    return status;
  }

  Gfn2IntegralDeviceBatch integral_batch(const HostCase& h) const {
    return {h.basis.batch_size,
            h.basis.total_atoms,
            h.basis.total_shells,
            h.basis.total_orbitals,
            h.basis.total_primitives,
            h.integrals.total_matrix_elements,
            h.h0.shell_pair_offsets.back(),
            h.maximum_system_shells,
            1,
            h.integrals.integral_cutoff,
            kPlanToken,
            static_cast<std::int64_t>(h.basis.atom_offsets.size()),
            static_cast<std::int64_t>(h.basis.batch_shell_offsets.size()),
            static_cast<std::int64_t>(h.basis.batch_orbital_offsets.size()),
            static_cast<std::int64_t>(h.integrals.matrix_offsets.size()),
            static_cast<std::int64_t>(h.h0.shell_pair_offsets.size()),
            static_cast<std::int64_t>(h.basis.atom_shell_offsets.size()),
            static_cast<std::int64_t>(h.basis.shell_orbital_offsets.size()),
            static_cast<std::int64_t>(h.basis.shell_primitive_offsets.size()),
            static_cast<std::int64_t>(h.basis.shell_to_atom.size()),
            static_cast<std::int64_t>(h.basis.angular_momenta.size()),
            static_cast<std::int64_t>(h.basis.primitive_exponents.size()),
            static_cast<std::int64_t>(h.basis.primitive_coefficients.size()),
            atom_offsets.get(),
            batch_shell_offsets.get(),
            batch_orbital_offsets.get(),
            matrix_offsets.get(),
            shell_pair_offsets.get(),
            atom_shell_offsets.get(),
            shell_orbital_offsets.get(),
            shell_primitive_offsets.get(),
            shell_to_atom.get(),
            angular_momenta.get(),
            primitive_exponents.get(),
            primitive_coefficients.get()};
  }

  Gfn2HamiltonianDeviceBatch hamiltonian_batch(const HostCase& h) const {
    return {h.basis.batch_size,
            h.basis.total_atoms,
            h.basis.total_shells,
            h.basis.total_orbitals,
            h.integrals.total_matrix_elements,
            kPlanToken,
            static_cast<std::int64_t>(h.basis.atom_offsets.size()),
            static_cast<std::int64_t>(h.basis.batch_shell_offsets.size()),
            static_cast<std::int64_t>(h.basis.batch_orbital_offsets.size()),
            static_cast<std::int64_t>(h.integrals.matrix_offsets.size()),
            static_cast<std::int64_t>(h.basis.atom_shell_offsets.size()),
            static_cast<std::int64_t>(h.basis.shell_orbital_offsets.size()),
            static_cast<std::int64_t>(h.basis.shell_to_atom.size()),
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
            orbital_to_atom.get(),
            xtbloom::detail::XtbModelFlavor::kGfn2,
            1};
  }

  Gfn2H0DevicePlan h0_plan(const HostCase& h) const {
    return {h.basis.total_atoms,
            h.basis.total_shells,
            h.basis.total_shells,
            h.basis.total_shells,
            h.h0.shell_pair_offsets.back(),
            kPlanToken,
            atomic_radii.get(),
            shell_levels.get(),
            shell_coordination_scale.get(),
            shell_polynomial.get(),
            shell_pair_scale.get()};
  }

  Gfn2ForceDeviceActivity activity(const HostCase& h) const {
    return {requested.get(), statuses.get(), h.basis.batch_size, kPlanToken};
  }

  Gfn2H0ForceDeviceInput h0_input(const HostCase& h) const {
    return {positions.get(),
            static_cast<std::int64_t>(h.positions.size()),
            coordination.get(),
            h.basis.total_atoms,
            overlap.get(),
            h.integrals.total_matrix_elements,
            density.get(),
            h.integrals.total_matrix_elements,
            weighted_density.get(),
            h.integrals.total_matrix_elements,
            kPlanToken};
  }

  Gfn2H0ForceDeviceOutput h0_output(const HostCase& h) {
    return {overlap_adjoint.get(),
            h.integrals.total_matrix_elements,
            coordination_adjoint.get(),
            h.basis.total_atoms,
            gradient.get(),
            3 * h.basis.total_atoms,
            kPlanToken};
  }

  Gfn2H0ForceDeviceWorkspace h0_workspace(const HostCase& h) {
    return {h0_overlap_scratch.get(),
            h.integrals.total_matrix_elements,
            h0_coordination_scratch.get(),
            h.basis.total_atoms,
            h0_gradient_scratch.get(),
            3 * h.basis.total_atoms,
            h0_sequence.get(),
            1,
            kPlanToken};
  }

  Gfn2HamiltonianForceDeviceInput hamiltonian_input(const HostCase& h) const {
    return {density.get(),
            h.integrals.total_matrix_elements,
            shell_scalar.get(),
            h.basis.total_shells,
            dipole_potential.get(),
            3 * h.basis.total_atoms,
            quadrupole_potential.get(),
            6 * h.basis.total_atoms,
            kPlanToken};
  }

  Gfn2HamiltonianForceDeviceOutput hamiltonian_output(const HostCase& h) {
    return {overlap_adjoint.get(),
            h.integrals.total_matrix_elements,
            dipole_adjoint.get(),
            3 * h.integrals.total_matrix_elements,
            quadrupole_adjoint.get(),
            6 * h.integrals.total_matrix_elements,
            kPlanToken};
  }

  Gfn2HamiltonianForceDeviceWorkspace hamiltonian_workspace(const HostCase& h) {
    return {hamiltonian_overlap_scratch.get(),
            h.integrals.total_matrix_elements,
            hamiltonian_dipole_scratch.get(),
            3 * h.integrals.total_matrix_elements,
            hamiltonian_quadrupole_scratch.get(),
            6 * h.integrals.total_matrix_elements,
            hamiltonian_sequence.get(),
            1,
            kPlanToken};
  }

  Gfn2IntegralForceDeviceWorkspace integral_workspace(const HostCase& h) {
    return {integral_gradient_scratch.get(), 3 * h.basis.total_atoms, integral_sequence.get(), 1,
            kPlanToken};
  }

  Gfn2ElectronicGradientDeviceWorkspace composer_workspace(const HostCase& h) {
    return {h0_success_mask.get(), hamiltonian_success_mask.get(), h.basis.batch_size, kPlanToken};
  }

  Gfn2ElectronicGradientDeviceDiagnostics diagnostics(const HostCase& h) {
    return {h0_system_errors.get(),
            h0_device_error.get(),
            hamiltonian_system_errors.get(),
            hamiltonian_device_error.get(),
            integral_system_errors.get(),
            integral_device_error.get(),
            h.basis.batch_size,
            kPlanToken};
  }
};

cudaError_t launch_with_integral_batch(DeviceFixture& device, const HostCase& host,
                                       const Gfn2IntegralDeviceBatch& integral_batch,
                                       cudaStream_t stream) {
  return xtbloom::detail::cuda::compose_gfn2_electronic_gradient_cuda(
      Gfn2ElectronicGradientRequest{1u, kPlanToken}, integral_batch, device.h0_plan(host),
      device.hamiltonian_batch(host), device.activity(host), device.h0_input(host),
      device.h0_output(host), device.h0_workspace(host), device.hamiltonian_input(host),
      device.hamiltonian_output(host), device.hamiltonian_workspace(host),
      device.integral_workspace(host), device.composer_workspace(host), device.diagnostics(host),
      stream);
}

cudaError_t launch(DeviceFixture& device, const HostCase& host, cudaStream_t stream) {
  return launch_with_integral_batch(device, host, device.integral_batch(host), stream);
}

int compare(DeviceFixture& device, const HostCase& host, cudaStream_t stream) {
  Expected actual;
  actual.overlap.resize(host.expected.overlap.size());
  actual.coordination.resize(host.expected.coordination.size());
  actual.dipole.resize(host.expected.dipole.size());
  actual.quadrupole.resize(host.expected.quadrupole.size());
  actual.gradient.resize(host.expected.gradient.size());
  CUDA_CHECK(device.overlap_adjoint.copy_to(actual.overlap.data(), actual.overlap.size(), stream));
  CUDA_CHECK(device.coordination_adjoint.copy_to(actual.coordination.data(),
                                                 actual.coordination.size(), stream));
  CUDA_CHECK(device.dipole_adjoint.copy_to(actual.dipole.data(), actual.dipole.size(), stream));
  CUDA_CHECK(device.quadrupole_adjoint.copy_to(actual.quadrupole.data(), actual.quadrupole.size(),
                                               stream));
  CUDA_CHECK(device.gradient.copy_to(actual.gradient.data(), actual.gradient.size(), stream));
  std::vector<std::uint32_t> h0_errors(static_cast<std::size_t>(host.basis.batch_size));
  std::vector<std::uint32_t> hamiltonian_errors(h0_errors.size());
  std::vector<std::uint32_t> integral_errors(h0_errors.size());
  CUDA_CHECK(device.h0_system_errors.copy_to(h0_errors.data(), h0_errors.size(), stream));
  CUDA_CHECK(device.hamiltonian_system_errors.copy_to(hamiltonian_errors.data(),
                                                      hamiltonian_errors.size(), stream));
  CUDA_CHECK(device.integral_system_errors.copy_to(integral_errors.data(), integral_errors.size(),
                                                   stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(std::all_of(h0_errors.begin(), h0_errors.end(),
                    [](std::uint32_t value) { return value == 0u; }));
  CHECK(std::all_of(hamiltonian_errors.begin(), hamiltonian_errors.end(),
                    [](std::uint32_t value) { return value == 0u; }));
  CHECK(std::all_of(integral_errors.begin(), integral_errors.end(),
                    [](std::uint32_t value) { return value == 0u; }));
  auto compare_vector = [](const std::vector<double>& values, const std::vector<double>& expected) {
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (!near(values[index], expected[index])) {
        return false;
      }
    }
    return true;
  };
  CHECK(compare_vector(actual.overlap, host.expected.overlap));
  CHECK(compare_vector(actual.coordination, host.expected.coordination));
  CHECK(compare_vector(actual.dipole, host.expected.dipole));
  CHECK(compare_vector(actual.quadrupole, host.expected.quadrupole));
  CHECK(compare_vector(actual.gradient, host.expected.gradient));
  return 0;
}

int test_composition_and_changed_input_graph() {
  HostCase host;
  std::string error;
  CHECK(make_case(host, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(launch(device, host, stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CHECK(compare(device, host, stream) == 0);

  host.positions[0] += 0.12;
  host.positions[4] -= 0.07;
  std::vector<double> cpu_workspace((host.integrals.workspace_size_bytes + sizeof(double) - 1u) /
                                    sizeof(double));
  CHECK(xtbloom::detail::gfn2::evaluate_overlap_cpu(
            host.basis, host.integrals, host.positions.data(), host.overlap.data(),
            cpu_workspace.data(), cpu_workspace.size() * sizeof(double),
            error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t coordinate = 0; coordinate < host.gradient_seed.size(); ++coordinate) {
    host.gradient_seed[coordinate] = -0.003 * std::cos(0.17 * static_cast<double>(coordinate + 2u));
  }
  CHECK(update_expected(host, error));
  CUDA_CHECK(device.upload_dynamic(host, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CHECK(compare(device, host, stream) == 0);

  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_stage_failure_propagates_without_peer_contamination() {
  HostCase host;
  std::string error;
  CHECK(make_case(host, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));

  constexpr std::size_t failed_system = 0u;
  HostCase poisoned = host;
  poisoned.coordination[0] = std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(device.upload_dynamic(poisoned, stream));
  CUDA_CHECK(launch(device, poisoned, stream));

  Expected expected = host.expected;
  const std::size_t atom_begin = static_cast<std::size_t>(host.basis.atom_offsets[failed_system]);
  const std::size_t atom_end =
      static_cast<std::size_t>(host.basis.atom_offsets[failed_system + 1u]);
  const std::size_t matrix_begin =
      static_cast<std::size_t>(host.integrals.matrix_offsets[failed_system]);
  const std::size_t matrix_end =
      static_cast<std::size_t>(host.integrals.matrix_offsets[failed_system + 1u]);
  std::copy(host.overlap_seed.begin() + static_cast<std::ptrdiff_t>(matrix_begin),
            host.overlap_seed.begin() + static_cast<std::ptrdiff_t>(matrix_end),
            expected.overlap.begin() + static_cast<std::ptrdiff_t>(matrix_begin));
  std::copy(host.coordination_seed.begin() + static_cast<std::ptrdiff_t>(atom_begin),
            host.coordination_seed.begin() + static_cast<std::ptrdiff_t>(atom_end),
            expected.coordination.begin() + static_cast<std::ptrdiff_t>(atom_begin));
  std::copy(host.gradient_seed.begin() + static_cast<std::ptrdiff_t>(3u * atom_begin),
            host.gradient_seed.begin() + static_cast<std::ptrdiff_t>(3u * atom_end),
            expected.gradient.begin() + static_cast<std::ptrdiff_t>(3u * atom_begin));
  const std::size_t matrices = host.overlap_seed.size();
  for (std::size_t component = 0; component < 3u; ++component) {
    const std::size_t begin = component * matrices + matrix_begin;
    const std::size_t end = component * matrices + matrix_end;
    std::copy(host.dipole_seed.begin() + static_cast<std::ptrdiff_t>(begin),
              host.dipole_seed.begin() + static_cast<std::ptrdiff_t>(end),
              expected.dipole.begin() + static_cast<std::ptrdiff_t>(begin));
  }
  for (std::size_t component = 0; component < 6u; ++component) {
    const std::size_t begin = component * matrices + matrix_begin;
    const std::size_t end = component * matrices + matrix_end;
    std::copy(host.quadrupole_seed.begin() + static_cast<std::ptrdiff_t>(begin),
              host.quadrupole_seed.begin() + static_cast<std::ptrdiff_t>(end),
              expected.quadrupole.begin() + static_cast<std::ptrdiff_t>(begin));
  }

  Expected actual;
  actual.overlap.resize(expected.overlap.size());
  actual.coordination.resize(expected.coordination.size());
  actual.dipole.resize(expected.dipole.size());
  actual.quadrupole.resize(expected.quadrupole.size());
  actual.gradient.resize(expected.gradient.size());
  CUDA_CHECK(device.overlap_adjoint.copy_to(actual.overlap.data(), actual.overlap.size(), stream));
  CUDA_CHECK(device.coordination_adjoint.copy_to(actual.coordination.data(),
                                                 actual.coordination.size(), stream));
  CUDA_CHECK(device.dipole_adjoint.copy_to(actual.dipole.data(), actual.dipole.size(), stream));
  CUDA_CHECK(device.quadrupole_adjoint.copy_to(actual.quadrupole.data(), actual.quadrupole.size(),
                                               stream));
  CUDA_CHECK(device.gradient.copy_to(actual.gradient.data(), actual.gradient.size(), stream));
  std::vector<std::uint32_t> h0_errors(2u);
  std::vector<std::uint32_t> hamiltonian_errors(2u);
  std::vector<std::uint32_t> integral_errors(2u);
  std::vector<std::uint8_t> h0_mask(2u);
  std::vector<std::uint8_t> hamiltonian_mask(2u);
  CUDA_CHECK(device.h0_system_errors.copy_to(h0_errors.data(), h0_errors.size(), stream));
  CUDA_CHECK(device.hamiltonian_system_errors.copy_to(hamiltonian_errors.data(),
                                                      hamiltonian_errors.size(), stream));
  CUDA_CHECK(device.integral_system_errors.copy_to(integral_errors.data(), integral_errors.size(),
                                                   stream));
  CUDA_CHECK(device.h0_success_mask.copy_to(h0_mask.data(), h0_mask.size(), stream));
  CUDA_CHECK(device.hamiltonian_success_mask.copy_to(hamiltonian_mask.data(),
                                                     hamiltonian_mask.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(h0_errors[failed_system] != 0u);
  CHECK(h0_errors[1] == 0u);
  CHECK(std::all_of(hamiltonian_errors.begin(), hamiltonian_errors.end(),
                    [](std::uint32_t value) { return value == 0u; }));
  CHECK(std::all_of(integral_errors.begin(), integral_errors.end(),
                    [](std::uint32_t value) { return value == 0u; }));
  CHECK(h0_mask[failed_system] == 0u);
  CHECK(hamiltonian_mask[failed_system] == 0u);
  CHECK(h0_mask[1] == 1u);
  CHECK(hamiltonian_mask[1] == 1u);
  auto equal_vectors = [](const std::vector<double>& values, const std::vector<double>& reference) {
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (!near(values[index], reference[index])) {
        return false;
      }
    }
    return true;
  };
  CHECK(equal_vectors(actual.overlap, expected.overlap));
  CHECK(equal_vectors(actual.coordination, expected.coordination));
  CHECK(equal_vectors(actual.dipole, expected.dipole));
  CHECK(equal_vectors(actual.quadrupole, expected.quadrupole));
  CHECK(equal_vectors(actual.gradient, expected.gradient));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_energy_only_requires_no_force_binding_or_launch() {
  const auto invalid_stream = reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(1u));
  CHECK(xtbloom::detail::cuda::compose_gfn2_electronic_gradient_cuda(
            Gfn2ElectronicGradientRequest{}, Gfn2IntegralDeviceBatch{}, Gfn2H0DevicePlan{},
            Gfn2HamiltonianDeviceBatch{}, Gfn2ForceDeviceActivity{}, Gfn2H0ForceDeviceInput{},
            Gfn2H0ForceDeviceOutput{}, Gfn2H0ForceDeviceWorkspace{},
            Gfn2HamiltonianForceDeviceInput{}, Gfn2HamiltonianForceDeviceOutput{},
            Gfn2HamiltonianForceDeviceWorkspace{}, Gfn2IntegralForceDeviceWorkspace{},
            Gfn2ElectronicGradientDeviceWorkspace{}, Gfn2ElectronicGradientDeviceDiagnostics{},
            invalid_stream) == cudaSuccess);
  return 0;
}

int test_invalid_linear_tiles_reject_before_error_reset() {
  HostCase host;
  std::string error;
  CHECK(make_case(host, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));

  constexpr std::uint32_t kSentinel = 0x6a17d4f2u;
  const std::vector<std::uint32_t> system_sentinel(static_cast<std::size_t>(host.basis.batch_size),
                                                   kSentinel);
  for (const std::int64_t invalid_tiles :
       std::array<std::int64_t, 2>{0, xtbloom::detail::cuda::kGfn2IntegralLinearBlockBudget + 1}) {
    CUDA_CHECK(
        device.h0_system_errors.copy_from(system_sentinel.data(), system_sentinel.size(), stream));
    CUDA_CHECK(device.h0_device_error.copy_from(&kSentinel, 1u, stream));
    Gfn2IntegralDeviceBatch invalid_batch = device.integral_batch(host);
    invalid_batch.linear_tiles_per_system = invalid_tiles;
    CHECK(launch_with_integral_batch(device, host, invalid_batch, stream) == cudaErrorInvalidValue);

    std::vector<std::uint32_t> actual_system_errors(system_sentinel.size());
    std::uint32_t actual_device_error = 0u;
    CUDA_CHECK(device.h0_system_errors.copy_to(actual_system_errors.data(),
                                               actual_system_errors.size(), stream));
    CUDA_CHECK(device.h0_device_error.copy_to(&actual_device_error, 1u, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(actual_system_errors == system_sentinel);
    CHECK(actual_device_error == kSentinel);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_composition_and_changed_input_graph(); status != 0) {
    return status;
  }
  if (const int status = test_stage_failure_propagates_without_peer_contamination(); status != 0) {
    return status;
  }
  if (const int status = test_energy_only_requires_no_force_binding_or_launch(); status != 0) {
    return status;
  }
  return test_invalid_linear_tiles_reject_before_error_reset();
}
