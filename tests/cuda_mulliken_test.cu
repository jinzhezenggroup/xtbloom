#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_mulliken.cuh"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/integrals.hpp"
#include "model/gfn2/mulliken.hpp"
#include "model/gfn2/wavefunction.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using gpuxtb::detail::Gfn2PlanMemorySpace;
using gpuxtb::detail::Gfn2WavefunctionLayoutView;
using gpuxtb::detail::cuda::Gfn2MullikenDeviceActivity;
using gpuxtb::detail::cuda::Gfn2MullikenDeviceBatch;
using gpuxtb::detail::cuda::Gfn2MullikenDeviceError;
using gpuxtb::detail::cuda::Gfn2MullikenDeviceInput;
using gpuxtb::detail::cuda::Gfn2MullikenDevicePopulation;
using gpuxtb::detail::cuda::Gfn2MullikenDeviceWorkspace;
using gpuxtb::detail::gfn2::BasisPlan;
using gpuxtb::detail::gfn2::IntegralPlan;
using gpuxtb::detail::gfn2::MullikenDensityView;
using gpuxtb::detail::gfn2::MullikenIntegralView;
using gpuxtb::detail::gfn2::MullikenPlan;
using gpuxtb::detail::gfn2::MullikenPopulationView;
using gpuxtb::detail::gfn2::MullikenWorkspace;
using gpuxtb::detail::gfn2::WavefunctionLayout;

constexpr std::uint64_t kPlanToken = 0x71c32d5b9e4a608fULL;
constexpr double kSentinel = -817.625;

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

  cudaError_t fill(const T& value, std::size_t count, cudaStream_t stream = nullptr) {
    if (count > count_) {
      return cudaErrorInvalidValue;
    }
    std::vector<T> host(count, value);
    return copy_from(host.data(), host.size(), stream);
  }

  T* get() { return data_; }
  const T* get() const { return data_; }
  std::size_t size() const { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

template <typename T>
cudaError_t upload(DeviceBuffer<T>& target, const std::vector<T>& source,
                   cudaStream_t stream = nullptr) {
  cudaError_t status = target.allocate(source.size());
  return status == cudaSuccess ? target.copy_from(source.data(), source.size(), stream) : status;
}

bool near(double actual, double expected, double absolute = 8.0e-12, double relative = 8.0e-12) {
  return std::abs(actual - expected) <=
         absolute + relative * std::max(std::abs(actual), std::abs(expected));
}

struct HostCase {
  std::int64_t batch_size = 0;
  std::int64_t maximum_system_atoms = 0;
  std::int64_t maximum_system_shells = 0;
  std::vector<std::int64_t> atom_offsets{0};
  std::vector<std::int64_t> batch_shell_offsets{0};
  std::vector<std::int64_t> batch_orbital_offsets{0};
  std::vector<std::int64_t> matrix_offsets{0};
  std::vector<std::int64_t> atom_shell_offsets{0};
  std::vector<std::int64_t> shell_orbital_offsets{0};
  std::vector<std::int64_t> shell_to_atom;
  std::vector<std::int64_t> spin_channel_offsets{0};
  std::vector<std::int64_t> spin_orbital_offsets{0};
  std::vector<std::int64_t> density_offsets{0};
  std::vector<std::int64_t> shell_population_offsets{0};
  std::vector<std::int64_t> atom_population_offsets{0};
  std::vector<std::int32_t> spin_channels;
  std::vector<double> reference;
  std::vector<double> density;
  std::vector<double> overlap;
  std::vector<double> dipole_integrals;
  std::vector<double> quadrupole_integrals;
  std::vector<std::uint8_t> active;
};

HostCase make_case(std::size_t batch_size, bool wide_shell = false) {
  HostCase data;
  data.batch_size = static_cast<std::int64_t>(batch_size);
  std::int64_t atoms_total = 0;
  std::int64_t shells_total = 0;
  std::int64_t orbitals_total = 0;
  std::int64_t matrices_total = 0;
  for (std::size_t system = 0; system < batch_size; ++system) {
    const std::int64_t atoms = wide_shell ? 1 : 1 + static_cast<std::int64_t>(system % 2u);
    const std::int64_t system_atom_begin = atoms_total;
    const std::int64_t system_shell_begin = shells_total;
    const std::int64_t system_orbital_begin = orbitals_total;
    for (std::int64_t local_atom = 0; local_atom < atoms; ++local_atom) {
      const std::int64_t shells =
          wide_shell ? 1 : 1 + static_cast<std::int64_t>((system + local_atom) % 2u);
      for (std::int64_t local_shell = 0; local_shell < shells; ++local_shell) {
        const std::int64_t orbitals =
            wide_shell ? 8
                       : 1 + static_cast<std::int64_t>((system + local_atom + local_shell) % 3u);
        data.shell_to_atom.push_back(atoms_total);
        data.reference.push_back(0.4 + 0.03 * static_cast<double>((shells_total * 7) % 17));
        orbitals_total += orbitals;
        ++shells_total;
        data.shell_orbital_offsets.push_back(orbitals_total);
      }
      ++atoms_total;
      data.atom_shell_offsets.push_back(shells_total);
    }
    const std::int64_t system_atoms = atoms_total - system_atom_begin;
    const std::int64_t system_shells = shells_total - system_shell_begin;
    const std::int64_t system_orbitals = orbitals_total - system_orbital_begin;
    matrices_total += system_orbitals * system_orbitals;
    data.maximum_system_atoms = std::max(data.maximum_system_atoms, system_atoms);
    data.maximum_system_shells = std::max(data.maximum_system_shells, system_shells);
    data.atom_offsets.push_back(atoms_total);
    data.batch_shell_offsets.push_back(shells_total);
    data.batch_orbital_offsets.push_back(orbitals_total);
    data.matrix_offsets.push_back(matrices_total);
    data.spin_channel_offsets.push_back(static_cast<std::int64_t>(system + 1u));
    data.spin_orbital_offsets.push_back(orbitals_total);
    data.density_offsets.push_back(matrices_total);
    data.shell_population_offsets.push_back(shells_total);
    data.atom_population_offsets.push_back(atoms_total);
    data.spin_channels.push_back(1);
  }
  data.active.assign(batch_size, 1u);
  data.density.resize(static_cast<std::size_t>(matrices_total));
  data.overlap.resize(static_cast<std::size_t>(matrices_total));
  data.dipole_integrals.resize(static_cast<std::size_t>(3 * matrices_total));
  data.quadrupole_integrals.resize(static_cast<std::size_t>(6 * matrices_total));
  for (std::int64_t matrix = 0; matrix < matrices_total; ++matrix) {
    data.density[static_cast<std::size_t>(matrix)] =
        0.02 * static_cast<double>(1 + (matrix * 13) % 29) - 0.21;
    data.overlap[static_cast<std::size_t>(matrix)] =
        0.015 * static_cast<double>(1 + (matrix * 11) % 31) - 0.18;
    for (int component = 0; component < 3; ++component) {
      data.dipole_integrals[static_cast<std::size_t>(component * matrices_total + matrix)] =
          0.007 * static_cast<double>(1 + ((component + 2) * matrix + component * 17) % 37) - 0.11;
    }
    for (int component = 0; component < 6; ++component) {
      data.quadrupole_integrals[static_cast<std::size_t>(component * matrices_total + matrix)] =
          0.004 * static_cast<double>(1 + ((component + 3) * matrix + component * 19) % 41) - 0.08;
    }
  }
  return data;
}

struct SpinOracleCase {
  HostCase host;
  MullikenPlan plan;
  std::vector<double> scratch;
};

/*
 * Build a real WavefunctionLayout/MullikenPlan batch so CUDA is compared with
 * the production CPU primitive rather than a second transcription of the spin
 * conversion. Even lanes are unrestricted oxygen atoms and odd lanes are
 * restricted H2, giving batch one a genuine two-channel case and larger
 * batches a ragged mixed-spin layout.
 */
bool make_spin_oracle_case(std::size_t batch_size, SpinOracleCase& result, std::string& error) {
  std::vector<std::int64_t> atom_offsets{0};
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> charges;
  std::vector<std::int32_t> unpaired;
  std::vector<std::int32_t> spin_channels;
  for (std::size_t system = 0; system < batch_size; ++system) {
    if (system % 2u == 0u) {
      atomic_numbers.push_back(8);
      unpaired.push_back(2);
      spin_channels.push_back(2);
    } else {
      atomic_numbers.push_back(1);
      atomic_numbers.push_back(1);
      unpaired.push_back(0);
      spin_channels.push_back(1);
    }
    charges.push_back(0.0);
    atom_offsets.push_back(static_cast<std::int64_t>(atomic_numbers.size()));
  }

  BasisPlan basis;
  IntegralPlan integral_plan;
  WavefunctionLayout wavefunction;
  if (gpuxtb::detail::gfn2::make_basis_plan(
          static_cast<std::int64_t>(batch_size), static_cast<std::int64_t>(atomic_numbers.size()),
          atom_offsets.data(), atomic_numbers.data(), basis, error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::make_integral_plan(basis, integral_plan, error) !=
          GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::make_wavefunction_layout(
          basis, atomic_numbers.data(), charges.data(), unpaired.data(), spin_channels.data(),
          wavefunction, error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::make_mulliken_plan(basis, integral_plan, wavefunction, result.plan,
                                               error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }

  HostCase& host = result.host;
  host.batch_size = static_cast<std::int64_t>(batch_size);
  host.atom_offsets = basis.atom_offsets;
  host.batch_shell_offsets = basis.batch_shell_offsets;
  host.batch_orbital_offsets = basis.batch_orbital_offsets;
  host.matrix_offsets = integral_plan.matrix_offsets;
  host.atom_shell_offsets = basis.atom_shell_offsets;
  host.shell_orbital_offsets = basis.shell_orbital_offsets;
  host.shell_to_atom = basis.shell_to_atom;
  host.reference = result.plan.reference_shell_occupations();
  host.spin_channels = spin_channels;
  host.spin_channel_offsets.assign(1u, 0);
  for (std::int32_t nspin : spin_channels) {
    host.spin_channel_offsets.push_back(host.spin_channel_offsets.back() + nspin);
  }
  host.density_offsets = wavefunction.density.system_offsets;
  host.spin_orbital_offsets = wavefunction.eigenvalues.system_offsets;
  host.shell_population_offsets = wavefunction.qsh.system_offsets;
  host.atom_population_offsets = wavefunction.qat.system_offsets;
  host.active.assign(batch_size, 1u);
  for (std::size_t system = 0; system < batch_size; ++system) {
    host.maximum_system_atoms = std::max(
        host.maximum_system_atoms, host.atom_offsets[system + 1u] - host.atom_offsets[system]);
    host.maximum_system_shells =
        std::max(host.maximum_system_shells,
                 host.batch_shell_offsets[system + 1u] - host.batch_shell_offsets[system]);
  }

  const std::int64_t matrices = integral_plan.total_matrix_elements;
  host.density.resize(static_cast<std::size_t>(wavefunction.density.element_count));
  host.overlap.resize(static_cast<std::size_t>(matrices));
  host.dipole_integrals.resize(static_cast<std::size_t>(3 * matrices));
  host.quadrupole_integrals.resize(static_cast<std::size_t>(6 * matrices));
  for (std::size_t index = 0; index < host.density.size(); ++index) {
    host.density[index] = 0.013 * static_cast<double>(1u + (index * 17u + 5u) % 43u) - 0.19;
  }
  for (std::int64_t matrix = 0; matrix < matrices; ++matrix) {
    host.overlap[static_cast<std::size_t>(matrix)] =
        0.009 * static_cast<double>(1 + (matrix * 11) % 47) - 0.16;
    for (int component = 0; component < 3; ++component) {
      host.dipole_integrals[static_cast<std::size_t>(component * matrices + matrix)] =
          0.006 * static_cast<double>(1 + ((component + 3) * matrix + component * 13) % 53) - 0.14;
    }
    for (int component = 0; component < 6; ++component) {
      host.quadrupole_integrals[static_cast<std::size_t>(component * matrices + matrix)] =
          0.003 * static_cast<double>(1 + ((component + 5) * matrix + component * 19) % 59) - 0.09;
    }
  }
  result.scratch.resize(static_cast<std::size_t>(result.plan.population_scratch_elements()));
  return true;
}

struct Population {
  std::vector<double> qsh;
  std::vector<double> qat;
  std::vector<double> dipole;
  std::vector<double> quadrupole;
};

Population sentinel_population(const HostCase& data) {
  Population result;
  result.qsh.assign(static_cast<std::size_t>(data.shell_population_offsets.back()), kSentinel);
  result.qat.assign(static_cast<std::size_t>(data.atom_population_offsets.back()), kSentinel);
  result.dipole.assign(3u * result.qat.size(), kSentinel);
  result.quadrupole.assign(6u * result.qat.size(), kSentinel);
  return result;
}

bool evaluate_cpu_oracle(SpinOracleCase& data, Population& result, std::string& error) {
  result = sentinel_population(data.host);
  const MullikenIntegralView integrals{data.host.overlap.data(), data.host.dipole_integrals.data(),
                                       data.host.quadrupole_integrals.data(),
                                       data.plan.matrix_elements(), data.plan.identity()};
  const MullikenDensityView density{data.host.density.data(), data.plan.density_elements(),
                                    data.plan.identity()};
  const MullikenPopulationView population{
      result.qsh.data(),        data.plan.shell_population_elements(),
      result.qat.data(),        data.plan.atom_population_elements(),
      result.dipole.data(),     data.plan.dipole_population_elements(),
      result.quadrupole.data(), data.plan.quadrupole_population_elements(),
      data.plan.identity()};
  const MullikenWorkspace workspace{data.scratch.data(),
                                    static_cast<std::int64_t>(data.scratch.size())};
  return gpuxtb::detail::gfn2::evaluate_mulliken_population_cpu(
             data.plan, integrals, density, population, workspace, error) == GPUXTB_STATUS_SUCCESS;
}

Population evaluate_cpu(const HostCase& data, const Population& initial) {
  Population result = initial;
  const std::int64_t matrices = data.matrix_offsets.back();
  for (std::int64_t system = 0; system < data.batch_size; ++system) {
    if (data.active[static_cast<std::size_t>(system)] == 0u) {
      continue;
    }
    const std::int64_t atom_begin = data.atom_offsets[static_cast<std::size_t>(system)];
    const std::int64_t atom_end = data.atom_offsets[static_cast<std::size_t>(system + 1)];
    const std::int64_t shell_begin = data.batch_shell_offsets[static_cast<std::size_t>(system)];
    const std::int64_t shell_end = data.batch_shell_offsets[static_cast<std::size_t>(system + 1)];
    const std::int64_t orbital_begin = data.batch_orbital_offsets[static_cast<std::size_t>(system)];
    const std::int64_t orbitals =
        data.batch_orbital_offsets[static_cast<std::size_t>(system + 1)] - orbital_begin;
    const std::int64_t matrix_begin = data.matrix_offsets[static_cast<std::size_t>(system)];
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      double charge = 0.0;
      for (std::int64_t ket = data.shell_orbital_offsets[static_cast<std::size_t>(shell)];
           ket < data.shell_orbital_offsets[static_cast<std::size_t>(shell + 1)]; ++ket) {
        for (std::int64_t bra = orbital_begin; bra < orbital_begin + orbitals; ++bra) {
          const std::int64_t index =
              matrix_begin + (bra - orbital_begin) * orbitals + ket - orbital_begin;
          charge = std::fma(-data.density[static_cast<std::size_t>(index)],
                            data.overlap[static_cast<std::size_t>(index)], charge);
        }
      }
      result.qsh[static_cast<std::size_t>(shell)] =
          charge + data.reference[static_cast<std::size_t>(shell)];
    }
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
      double atomic_charge = 0.0;
      for (std::int64_t shell = data.atom_shell_offsets[static_cast<std::size_t>(atom)];
           shell < data.atom_shell_offsets[static_cast<std::size_t>(atom + 1)]; ++shell) {
        atomic_charge += result.qsh[static_cast<std::size_t>(shell)];
      }
      result.qat[static_cast<std::size_t>(atom)] = atomic_charge;
      const std::int64_t first_shell = data.atom_shell_offsets[static_cast<std::size_t>(atom)];
      const std::int64_t last_shell = data.atom_shell_offsets[static_cast<std::size_t>(atom + 1)];
      const std::int64_t ket_begin =
          data.shell_orbital_offsets[static_cast<std::size_t>(first_shell)];
      const std::int64_t ket_end = data.shell_orbital_offsets[static_cast<std::size_t>(last_shell)];
      for (int component = 0; component < 9; ++component) {
        double value = 0.0;
        for (std::int64_t ket = ket_begin; ket < ket_end; ++ket) {
          for (std::int64_t bra = orbital_begin; bra < orbital_begin + orbitals; ++bra) {
            const std::int64_t index =
                matrix_begin + (bra - orbital_begin) * orbitals + ket - orbital_begin;
            const double integral =
                component < 3
                    ? data.dipole_integrals[static_cast<std::size_t>(component * matrices + index)]
                    : data.quadrupole_integrals[static_cast<std::size_t>(
                          (component - 3) * matrices + index)];
            value = std::fma(-data.density[static_cast<std::size_t>(index)], integral, value);
          }
        }
        if (component < 3) {
          result.dipole[static_cast<std::size_t>(atom * 3 + component)] = value;
        } else {
          result.quadrupole[static_cast<std::size_t>(atom * 6 + component - 3)] = value;
        }
      }
    }
  }
  return result;
}

bool equal_population(const Population& actual, const Population& expected) {
  auto equal = [](const std::vector<double>& first, const std::vector<double>& second) {
    if (first.size() != second.size()) {
      return false;
    }
    for (std::size_t index = 0; index < first.size(); ++index) {
      if (!near(first[index], second[index])) {
        return false;
      }
    }
    return true;
  };
  return equal(actual.qsh, expected.qsh) && equal(actual.qat, expected.qat) &&
         equal(actual.dipole, expected.dipole) && equal(actual.quadrupole, expected.quadrupole);
}

struct DeviceFixture {
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> batch_shell_offsets;
  DeviceBuffer<std::int64_t> batch_orbital_offsets;
  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<std::int64_t> atom_shell_offsets;
  DeviceBuffer<std::int64_t> shell_orbital_offsets;
  DeviceBuffer<std::int64_t> shell_to_atom;
  DeviceBuffer<std::int64_t> spin_channel_offsets;
  DeviceBuffer<std::int64_t> spin_orbital_offsets;
  DeviceBuffer<std::int64_t> density_offsets;
  DeviceBuffer<std::int64_t> shell_population_offsets;
  DeviceBuffer<std::int64_t> atom_population_offsets;
  DeviceBuffer<std::int32_t> spin_channels;
  DeviceBuffer<double> reference;
  DeviceBuffer<double> density;
  DeviceBuffer<double> overlap;
  DeviceBuffer<double> dipole_integrals;
  DeviceBuffer<double> quadrupole_integrals;
  DeviceBuffer<std::uint8_t> active;
  DeviceBuffer<double> qsh;
  DeviceBuffer<double> qat;
  DeviceBuffer<double> dipole;
  DeviceBuffer<double> quadrupole;
  DeviceBuffer<double> qsh_scratch;
  DeviceBuffer<double> qat_scratch;
  DeviceBuffer<double> dipole_scratch;
  DeviceBuffer<double> quadrupole_scratch;
  DeviceBuffer<std::uint32_t> sequence_active;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;

  cudaError_t initialize(const HostCase& host, cudaStream_t stream = nullptr) {
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
    UPLOAD(spin_channel_offsets, host.spin_channel_offsets)
    UPLOAD(spin_orbital_offsets, host.spin_orbital_offsets)
    UPLOAD(density_offsets, host.density_offsets)
    UPLOAD(shell_population_offsets, host.shell_population_offsets)
    UPLOAD(atom_population_offsets, host.atom_population_offsets)
    UPLOAD(spin_channels, host.spin_channels)
    UPLOAD(reference, host.reference)
    UPLOAD(density, host.density)
    UPLOAD(overlap, host.overlap)
    UPLOAD(dipole_integrals, host.dipole_integrals)
    UPLOAD(quadrupole_integrals, host.quadrupole_integrals)
    UPLOAD(active, host.active)
#undef UPLOAD
    const std::size_t shells = static_cast<std::size_t>(host.shell_population_offsets.back());
    const std::size_t atoms = static_cast<std::size_t>(host.atom_population_offsets.back());
#define ALLOCATE(field, count)      \
  if (status == cudaSuccess) {      \
    status = field.allocate(count); \
  }
    ALLOCATE(qsh, shells)
    ALLOCATE(qat, atoms)
    ALLOCATE(dipole, 3u * atoms)
    ALLOCATE(quadrupole, 6u * atoms)
    ALLOCATE(qsh_scratch, shells)
    ALLOCATE(qat_scratch, atoms)
    ALLOCATE(dipole_scratch, 3u * atoms)
    ALLOCATE(quadrupole_scratch, 6u * atoms)
    ALLOCATE(sequence_active, 1u)
    ALLOCATE(system_errors, static_cast<std::size_t>(host.batch_size))
    ALLOCATE(device_error, 1u)
#undef ALLOCATE
    if (status == cudaSuccess) {
      status = qsh.fill(kSentinel, shells, stream);
    }
    if (status == cudaSuccess) {
      status = qat.fill(kSentinel, atoms, stream);
    }
    if (status == cudaSuccess) {
      status = dipole.fill(kSentinel, 3u * atoms, stream);
    }
    if (status == cudaSuccess) {
      status = quadrupole.fill(kSentinel, 6u * atoms, stream);
    }
    return status;
  }

  Gfn2MullikenDeviceBatch batch(const HostCase& host) const {
    return {host.batch_size,
            host.atom_offsets.back(),
            static_cast<std::int64_t>(host.shell_to_atom.size()),
            host.batch_orbital_offsets.back(),
            host.matrix_offsets.back(),
            host.maximum_system_atoms,
            host.maximum_system_shells,
            kPlanToken,
            static_cast<std::int64_t>(host.atom_offsets.size()),
            static_cast<std::int64_t>(host.batch_shell_offsets.size()),
            static_cast<std::int64_t>(host.batch_orbital_offsets.size()),
            static_cast<std::int64_t>(host.matrix_offsets.size()),
            static_cast<std::int64_t>(host.atom_shell_offsets.size()),
            static_cast<std::int64_t>(host.shell_orbital_offsets.size()),
            static_cast<std::int64_t>(host.shell_to_atom.size()),
            static_cast<std::int64_t>(host.reference.size()),
            atom_offsets.get(),
            batch_shell_offsets.get(),
            batch_orbital_offsets.get(),
            matrix_offsets.get(),
            atom_shell_offsets.get(),
            shell_orbital_offsets.get(),
            shell_to_atom.get(),
            reference.get()};
  }

  Gfn2MullikenDeviceInput input(const HostCase& host) const {
    return {density.get(),
            static_cast<std::int64_t>(host.density.size()),
            overlap.get(),
            static_cast<std::int64_t>(host.overlap.size()),
            dipole_integrals.get(),
            static_cast<std::int64_t>(host.dipole_integrals.size()),
            quadrupole_integrals.get(),
            static_cast<std::int64_t>(host.quadrupole_integrals.size()),
            kPlanToken};
  }

  Gfn2WavefunctionLayoutView spin_layout(const HostCase& host) const {
    Gfn2WavefunctionLayoutView result;
    result.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
    result.plan_token = kPlanToken;
    result.batch_size = host.batch_size;
    result.total_spin_channels = host.spin_channel_offsets.back();
    result.total_spin_orbitals = host.spin_orbital_offsets.back();
    result.total_spin_matrix_elements = host.density_offsets.back();
    result.total_spin_shells = host.shell_population_offsets.back();
    result.total_spin_atoms = host.atom_population_offsets.back();
    result.spin_channel_count = static_cast<std::int64_t>(host.spin_channels.size());
    result.spin_channel_offset_count = static_cast<std::int64_t>(host.spin_channel_offsets.size());
    result.spin_orbital_offset_count = static_cast<std::int64_t>(host.spin_orbital_offsets.size());
    result.spin_matrix_offset_count = static_cast<std::int64_t>(host.density_offsets.size());
    result.spin_shell_offset_count =
        static_cast<std::int64_t>(host.shell_population_offsets.size());
    result.spin_atom_offset_count = static_cast<std::int64_t>(host.atom_population_offsets.size());
    result.spin_channels = spin_channels.get();
    result.spin_channel_offsets = spin_channel_offsets.get();
    result.spin_orbital_offsets = spin_orbital_offsets.get();
    result.spin_matrix_offsets = density_offsets.get();
    result.spin_shell_offsets = shell_population_offsets.get();
    result.spin_atom_offsets = atom_population_offsets.get();
    return result;
  }

  Gfn2MullikenDeviceActivity activity(const HostCase& host) const {
    return {active.get(), host.batch_size, kPlanToken};
  }

  Gfn2MullikenDevicePopulation population(const HostCase& host) {
    const std::int64_t atoms = host.atom_offsets.back();
    return {qsh.get(),        static_cast<std::int64_t>(host.shell_to_atom.size()),
            qat.get(),        atoms,
            dipole.get(),     3 * atoms,
            quadrupole.get(), 6 * atoms,
            kPlanToken};
  }

  Gfn2MullikenDeviceWorkspace workspace(const HostCase& host) {
    const std::int64_t atoms = host.atom_offsets.back();
    return {qsh_scratch.get(),
            static_cast<std::int64_t>(host.shell_to_atom.size()),
            qat_scratch.get(),
            atoms,
            dipole_scratch.get(),
            3 * atoms,
            quadrupole_scratch.get(),
            6 * atoms,
            sequence_active.get(),
            1,
            kPlanToken};
  }

  Gfn2MullikenDeviceInput spin_input(const HostCase& host) const {
    Gfn2MullikenDeviceInput result = input(host);
    result.density_elements = host.density_offsets.back();
    return result;
  }

  Gfn2MullikenDevicePopulation spin_population(const HostCase& host) {
    const std::int64_t shells = host.shell_population_offsets.back();
    const std::int64_t atoms = host.atom_population_offsets.back();
    return {qsh.get(), shells,           qat.get(), atoms,     dipole.get(),
            3 * atoms, quadrupole.get(), 6 * atoms, kPlanToken};
  }

  Gfn2MullikenDeviceWorkspace spin_workspace(const HostCase& host) {
    const std::int64_t shells = host.shell_population_offsets.back();
    const std::int64_t atoms = host.atom_population_offsets.back();
    return {qsh_scratch.get(),
            shells,
            qat_scratch.get(),
            atoms,
            dipole_scratch.get(),
            3 * atoms,
            quadrupole_scratch.get(),
            6 * atoms,
            sequence_active.get(),
            1,
            kPlanToken};
  }

  cudaError_t download(const HostCase& host, Population& output, std::vector<std::uint32_t>& errors,
                       std::uint32_t& diagnostic, cudaStream_t stream = nullptr) const {
    output = sentinel_population(host);
    errors.resize(static_cast<std::size_t>(host.batch_size));
    cudaError_t status = qsh.copy_to(output.qsh.data(), output.qsh.size(), stream);
#define DOWNLOAD(field, target)                                   \
  if (status == cudaSuccess) {                                    \
    status = field.copy_to(target.data(), target.size(), stream); \
  }
    DOWNLOAD(qat, output.qat)
    DOWNLOAD(dipole, output.dipole)
    DOWNLOAD(quadrupole, output.quadrupole)
    DOWNLOAD(system_errors, errors)
#undef DOWNLOAD
    if (status == cudaSuccess) {
      status = device_error.copy_to(&diagnostic, 1u, stream);
    }
    return status;
  }
};

cudaError_t launch(DeviceFixture& device, const HostCase& host, cudaStream_t stream = nullptr) {
  const Gfn2MullikenDeviceBatch batch = device.batch(host);
  const Gfn2MullikenDeviceInput input = device.input(host);
  const Gfn2MullikenDeviceActivity activity = device.activity(host);
  Gfn2MullikenDevicePopulation population = device.population(host);
  Gfn2MullikenDeviceWorkspace workspace = device.workspace(host);
  cudaError_t status = gpuxtb::detail::cuda::reset_gfn2_mulliken_device_errors_cuda(
      host.batch_size, device.system_errors.get(), device.device_error.get(), stream);
  return status == cudaSuccess ? gpuxtb::detail::cuda::evaluate_gfn2_mulliken_population_cuda(
                                     batch, input, activity, population, workspace,
                                     device.system_errors.get(), device.device_error.get(), stream)
                               : status;
}

cudaError_t launch_spin(DeviceFixture& device, const HostCase& host,
                        cudaStream_t stream = nullptr) {
  const Gfn2MullikenDeviceBatch batch = device.batch(host);
  const Gfn2WavefunctionLayoutView layout = device.spin_layout(host);
  const Gfn2MullikenDeviceInput input = device.spin_input(host);
  const Gfn2MullikenDeviceActivity activity = device.activity(host);
  Gfn2MullikenDevicePopulation population = device.spin_population(host);
  Gfn2MullikenDeviceWorkspace workspace = device.spin_workspace(host);
  cudaError_t status = gpuxtb::detail::cuda::reset_gfn2_mulliken_device_errors_cuda(
      host.batch_size, device.system_errors.get(), device.device_error.get(), stream);
  return status == cudaSuccess ? gpuxtb::detail::cuda::evaluate_gfn2_mulliken_population_spin_cuda(
                                     batch, layout, input, activity, population, workspace,
                                     device.system_errors.get(), device.device_error.get(), stream)
                               : status;
}

int run_parity(std::size_t batch_size, bool graph_capture) {
  HostCase host = make_case(batch_size);
  const Population expected = evaluate_cpu(host, sentinel_population(host));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (graph_capture) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    CUDA_CHECK(launch(device, host, stream));
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
  } else {
    CUDA_CHECK(launch(device, host, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  Population actual;
  std::vector<std::uint32_t> errors;
  std::uint32_t diagnostic = 99u;
  CUDA_CHECK(device.download(host, actual, errors, diagnostic, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(equal_population(actual, expected));
  CHECK(std::all_of(errors.begin(), errors.end(), [](std::uint32_t value) { return value == 0u; }));
  CHECK(diagnostic == 0u);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int run_spin_parity(std::size_t batch_size, bool graph_capture) {
  SpinOracleCase oracle;
  std::string error;
  CHECK(make_spin_oracle_case(batch_size, oracle, error));
  Population expected;
  CHECK(evaluate_cpu_oracle(oracle, expected, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(oracle.host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (graph_capture) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    CUDA_CHECK(launch_spin(device, oracle.host, stream));
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
  } else {
    CUDA_CHECK(launch_spin(device, oracle.host, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  Population actual;
  std::vector<std::uint32_t> errors;
  std::uint32_t diagnostic = 99u;
  CUDA_CHECK(device.download(oracle.host, actual, errors, diagnostic, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(equal_population(actual, expected));
  CHECK(std::all_of(errors.begin(), errors.end(), [](std::uint32_t value) { return value == 0u; }));
  CHECK(diagnostic == 0u);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

bool system_population_near(const HostCase& host, const Population& actual,
                            const Population& expected, std::size_t system) {
  const std::int64_t qsh_begin = host.shell_population_offsets[system];
  const std::int64_t qsh_end = host.shell_population_offsets[system + 1u];
  const std::int64_t qat_begin = host.atom_population_offsets[system];
  const std::int64_t qat_end = host.atom_population_offsets[system + 1u];
  for (std::int64_t index = qsh_begin; index < qsh_end; ++index) {
    if (!near(actual.qsh[static_cast<std::size_t>(index)],
              expected.qsh[static_cast<std::size_t>(index)])) {
      return false;
    }
  }
  for (std::int64_t index = qat_begin; index < qat_end; ++index) {
    if (!near(actual.qat[static_cast<std::size_t>(index)],
              expected.qat[static_cast<std::size_t>(index)])) {
      return false;
    }
    for (int component = 0; component < 3; ++component) {
      const std::size_t component_index = static_cast<std::size_t>(index * 3 + component);
      if (!near(actual.dipole[component_index], expected.dipole[component_index])) {
        return false;
      }
    }
    for (int component = 0; component < 6; ++component) {
      const std::size_t component_index = static_cast<std::size_t>(index * 6 + component);
      if (!near(actual.quadrupole[component_index], expected.quadrupole[component_index])) {
        return false;
      }
    }
  }
  return true;
}

bool slice_is_sentinel(const HostCase& host, const Population& value, std::size_t system) {
  const std::int64_t shell_begin = host.shell_population_offsets[system];
  const std::int64_t shell_end = host.shell_population_offsets[system + 1u];
  const std::int64_t atom_begin = host.atom_population_offsets[system];
  const std::int64_t atom_end = host.atom_population_offsets[system + 1u];
  for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
    if (value.qsh[static_cast<std::size_t>(shell)] != kSentinel) {
      return false;
    }
  }
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    if (value.qat[static_cast<std::size_t>(atom)] != kSentinel) {
      return false;
    }
    for (int component = 0; component < 3; ++component) {
      if (value.dipole[static_cast<std::size_t>(atom * 3 + component)] != kSentinel) {
        return false;
      }
    }
    for (int component = 0; component < 6; ++component) {
      if (value.quadrupole[static_cast<std::size_t>(atom * 6 + component)] != kSentinel) {
        return false;
      }
    }
  }
  return true;
}

int test_spin_swap_graph_replay() {
  SpinOracleCase oracle;
  std::string error;
  CHECK(make_spin_oracle_case(1u, oracle, error));
  Population expected_first;
  CHECK(evaluate_cpu_oracle(oracle, expected_first, error));

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(oracle.host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(launch_spin(device, oracle.host, stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  Population actual_first;
  std::vector<std::uint32_t> errors;
  std::uint32_t diagnostic = 99u;
  CUDA_CHECK(device.download(oracle.host, actual_first, errors, diagnostic, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(equal_population(actual_first, expected_first));
  CHECK(diagnostic == 0u);

  const std::int64_t matrix_elements =
      oracle.host.matrix_offsets[1] - oracle.host.matrix_offsets[0];
  std::swap_ranges(oracle.host.density.begin(), oracle.host.density.begin() + matrix_elements,
                   oracle.host.density.begin() + matrix_elements);
  Population expected_swapped;
  CHECK(evaluate_cpu_oracle(oracle, expected_swapped, error));
  CUDA_CHECK(
      device.density.copy_from(oracle.host.density.data(), oracle.host.density.size(), stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  Population actual_swapped;
  diagnostic = 99u;
  CUDA_CHECK(device.download(oracle.host, actual_swapped, errors, diagnostic, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(equal_population(actual_swapped, expected_swapped));
  CHECK(diagnostic == 0u);

  const std::int64_t shells = oracle.host.batch_shell_offsets[1];
  const std::int64_t atoms = oracle.host.atom_offsets[1];
  for (std::int64_t shell = 0; shell < shells; ++shell) {
    CHECK(near(actual_swapped.qsh[static_cast<std::size_t>(shell)],
               actual_first.qsh[static_cast<std::size_t>(shell)]));
    CHECK(near(actual_swapped.qsh[static_cast<std::size_t>(shells + shell)],
               -actual_first.qsh[static_cast<std::size_t>(shells + shell)]));
  }
  for (std::int64_t atom = 0; atom < atoms; ++atom) {
    CHECK(near(actual_swapped.qat[static_cast<std::size_t>(atom)],
               actual_first.qat[static_cast<std::size_t>(atom)]));
    CHECK(near(actual_swapped.qat[static_cast<std::size_t>(atoms + atom)],
               -actual_first.qat[static_cast<std::size_t>(atoms + atom)]));
    for (int component = 0; component < 3; ++component) {
      CHECK(near(actual_swapped.dipole[static_cast<std::size_t>(atom * 3 + component)],
                 actual_first.dipole[static_cast<std::size_t>(atom * 3 + component)]));
      CHECK(near(actual_swapped.dipole[static_cast<std::size_t>((atoms + atom) * 3 + component)],
                 -actual_first.dipole[static_cast<std::size_t>((atoms + atom) * 3 + component)]));
    }
    for (int component = 0; component < 6; ++component) {
      CHECK(near(actual_swapped.quadrupole[static_cast<std::size_t>(atom * 6 + component)],
                 actual_first.quadrupole[static_cast<std::size_t>(atom * 6 + component)]));
      CHECK(
          near(actual_swapped.quadrupole[static_cast<std::size_t>((atoms + atom) * 6 + component)],
               -actual_first.quadrupole[static_cast<std::size_t>((atoms + atom) * 6 + component)]));
    }
  }
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_equal_spin_densities_have_exact_zero_magnetization() {
  SpinOracleCase oracle;
  std::string error;
  CHECK(make_spin_oracle_case(8u, oracle, error));
  for (std::size_t system = 0; system < 8u; ++system) {
    if (oracle.host.spin_channels[system] != 2) {
      continue;
    }
    const std::int64_t density_begin = oracle.host.density_offsets[system];
    const std::int64_t matrix_elements =
        oracle.host.matrix_offsets[system + 1u] - oracle.host.matrix_offsets[system];
    std::copy_n(oracle.host.density.begin() + density_begin, matrix_elements,
                oracle.host.density.begin() + density_begin + matrix_elements);
  }
  Population expected;
  CHECK(evaluate_cpu_oracle(oracle, expected, error));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(oracle.host));
  CUDA_CHECK(launch_spin(device, oracle.host));
  CUDA_CHECK(cudaDeviceSynchronize());
  Population actual;
  std::vector<std::uint32_t> errors;
  std::uint32_t diagnostic = 99u;
  CUDA_CHECK(device.download(oracle.host, actual, errors, diagnostic));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(equal_population(actual, expected));
  CHECK(diagnostic == 0u);
  for (std::size_t system = 0; system < 8u; ++system) {
    if (oracle.host.spin_channels[system] != 2) {
      continue;
    }
    const std::int64_t shells =
        oracle.host.batch_shell_offsets[system + 1u] - oracle.host.batch_shell_offsets[system];
    const std::int64_t atoms =
        oracle.host.atom_offsets[system + 1u] - oracle.host.atom_offsets[system];
    const std::int64_t qsh_magnetization = oracle.host.shell_population_offsets[system] + shells;
    const std::int64_t qat_magnetization = oracle.host.atom_population_offsets[system] + atoms;
    for (std::int64_t local = 0; local < shells; ++local) {
      const double value = actual.qsh[static_cast<std::size_t>(qsh_magnetization + local)];
      CHECK(value == 0.0 && !std::signbit(value));
    }
    for (std::int64_t local = 0; local < atoms; ++local) {
      const std::int64_t atom = qat_magnetization + local;
      CHECK(actual.qat[static_cast<std::size_t>(atom)] == 0.0);
      for (int component = 0; component < 3; ++component) {
        const double value = actual.dipole[static_cast<std::size_t>(atom * 3 + component)];
        CHECK(value == 0.0 && !std::signbit(value));
      }
      for (int component = 0; component < 6; ++component) {
        const double value = actual.quadrupole[static_cast<std::size_t>(atom * 6 + component)];
        CHECK(value == 0.0 && !std::signbit(value));
      }
    }
  }
  return 0;
}

int test_spin_inactive_and_beta_failure_isolation() {
  SpinOracleCase oracle;
  std::string error;
  CHECK(make_spin_oracle_case(8u, oracle, error));
  Population expected;
  CHECK(evaluate_cpu_oracle(oracle, expected, error));
  constexpr std::size_t terminal = 2u;
  constexpr std::size_t bad_beta = 4u;
  oracle.host.active[terminal] = 0u;
  for (std::int64_t index = oracle.host.density_offsets[terminal];
       index < oracle.host.density_offsets[terminal + 1u]; ++index) {
    oracle.host.density[static_cast<std::size_t>(index)] = std::numeric_limits<double>::quiet_NaN();
  }
  const std::int64_t bad_matrix_elements =
      oracle.host.matrix_offsets[bad_beta + 1u] - oracle.host.matrix_offsets[bad_beta];
  oracle.host.density[static_cast<std::size_t>(oracle.host.density_offsets[bad_beta] +
                                               bad_matrix_elements)] =
      std::numeric_limits<double>::quiet_NaN();
  DeviceFixture device;
  CUDA_CHECK(device.initialize(oracle.host));
  std::vector<std::int32_t> hostile_spin = oracle.host.spin_channels;
  hostile_spin[terminal] = 3;
  CUDA_CHECK(device.spin_channels.copy_from(hostile_spin.data(), hostile_spin.size()));
  CUDA_CHECK(launch_spin(device, oracle.host));
  CUDA_CHECK(cudaDeviceSynchronize());
  Population actual;
  std::vector<std::uint32_t> errors;
  std::uint32_t diagnostic = 0u;
  CUDA_CHECK(device.download(oracle.host, actual, errors, diagnostic));
  CUDA_CHECK(cudaDeviceSynchronize());
  for (std::size_t system = 0; system < 8u; ++system) {
    if (system == terminal || system == bad_beta) {
      CHECK(slice_is_sentinel(oracle.host, actual, system));
    } else {
      CHECK(system_population_near(oracle.host, actual, expected, system));
      CHECK(errors[system] == 0u);
    }
  }
  CHECK(errors[terminal] == 0u);
  CHECK(errors[bad_beta] == static_cast<std::uint32_t>(Gfn2MullikenDeviceError::kNonfiniteDensity));
  CHECK(diagnostic == errors[bad_beta]);
  return 0;
}

int test_spin_layout_validation_and_offsets() {
  SpinOracleCase oracle;
  std::string error;
  CHECK(make_spin_oracle_case(1u, oracle, error));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(oracle.host));
  const Gfn2MullikenDeviceBatch batch = device.batch(oracle.host);
  const Gfn2MullikenDeviceInput input = device.spin_input(oracle.host);
  const Gfn2MullikenDeviceActivity activity = device.activity(oracle.host);
  Gfn2MullikenDevicePopulation population = device.spin_population(oracle.host);
  Gfn2MullikenDeviceWorkspace workspace = device.spin_workspace(oracle.host);
  Gfn2WavefunctionLayoutView layout = device.spin_layout(oracle.host);

  Gfn2WavefunctionLayoutView wrong_count = layout;
  --wrong_count.spin_matrix_offset_count;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_mulliken_population_spin_cuda(
            batch, wrong_count, input, activity, population, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  Gfn2WavefunctionLayoutView alias_layout = layout;
  alias_layout.spin_matrix_offsets = batch.matrix_offsets;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_mulliken_population_spin_cuda(
            batch, alias_layout, input, activity, population, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  Gfn2MullikenDevicePopulation alias_population = population;
  alias_population.qsh = workspace.qsh_scratch;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_mulliken_population_spin_cuda(
            batch, layout, input, activity, alias_population, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);

  const std::int32_t invalid_spin = 3;
  CUDA_CHECK(device.spin_channels.copy_from(&invalid_spin, 1u));
  CUDA_CHECK(launch_spin(device, oracle.host));
  CUDA_CHECK(cudaDeviceSynchronize());
  Population actual;
  std::vector<std::uint32_t> errors;
  std::uint32_t diagnostic = 0u;
  CUDA_CHECK(device.download(oracle.host, actual, errors, diagnostic));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(slice_is_sentinel(oracle.host, actual, 0u));
  CHECK(errors[0] == static_cast<std::uint32_t>(Gfn2MullikenDeviceError::kInvalidSpinChannels));

  DeviceFixture offset_device;
  CUDA_CHECK(offset_device.initialize(oracle.host));
  const std::int64_t hostile = std::numeric_limits<std::int64_t>::max();
  CUDA_CHECK(cudaMemcpy(offset_device.density_offsets.get() + 1, &hostile, sizeof(hostile),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(launch_spin(offset_device, oracle.host));
  CUDA_CHECK(cudaDeviceSynchronize());
  diagnostic = 0u;
  CUDA_CHECK(offset_device.download(oracle.host, actual, errors, diagnostic));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(slice_is_sentinel(oracle.host, actual, 0u));
  CHECK(errors[0] == static_cast<std::uint32_t>(Gfn2MullikenDeviceError::kInvalidOffsets));
  return 0;
}

int test_terminal_skip_and_failure_isolation() {
  HostCase host = make_case(8u);
  constexpr std::size_t terminal = 2u;
  host.active[terminal] = 0u;
  const std::int64_t terminal_matrix_begin = host.matrix_offsets[terminal];
  const std::int64_t terminal_matrix_end = host.matrix_offsets[terminal + 1u];
  for (std::int64_t index = terminal_matrix_begin; index < terminal_matrix_end; ++index) {
    host.density[static_cast<std::size_t>(index)] = std::numeric_limits<double>::quiet_NaN();
    host.overlap[static_cast<std::size_t>(index)] = std::numeric_limits<double>::infinity();
  }
  constexpr std::size_t bad_density = 4u;
  constexpr std::size_t bad_integral = 5u;
  host.density[static_cast<std::size_t>(host.matrix_offsets[bad_density])] =
      std::numeric_limits<double>::quiet_NaN();
  host.quadrupole_integrals[static_cast<std::size_t>(host.matrix_offsets[bad_integral])] =
      std::numeric_limits<double>::infinity();
  Population expected = evaluate_cpu(host, sentinel_population(host));

  DeviceFixture device;
  CUDA_CHECK(device.initialize(host));
  CUDA_CHECK(launch(device, host));
  CUDA_CHECK(cudaDeviceSynchronize());
  Population actual;
  std::vector<std::uint32_t> errors;
  std::uint32_t diagnostic = 0u;
  CUDA_CHECK(device.download(host, actual, errors, diagnostic));
  CUDA_CHECK(cudaDeviceSynchronize());
  for (std::size_t system = 0; system < 8u; ++system) {
    if (system == terminal || system == bad_density || system == bad_integral) {
      CHECK(slice_is_sentinel(host, actual, system));
    } else {
      const std::int64_t shell_begin = host.batch_shell_offsets[system];
      const std::int64_t shell_end = host.batch_shell_offsets[system + 1u];
      for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
        CHECK(near(actual.qsh[static_cast<std::size_t>(shell)],
                   expected.qsh[static_cast<std::size_t>(shell)]));
      }
    }
  }
  CHECK(errors[terminal] == 0u);
  CHECK(errors[bad_density] ==
        static_cast<std::uint32_t>(Gfn2MullikenDeviceError::kNonfiniteDensity));
  CHECK(errors[bad_integral] ==
        static_cast<std::uint32_t>(Gfn2MullikenDeviceError::kNonfiniteIntegral));
  CHECK(diagnostic != 0u);
  return 0;
}

int test_empty_active_set_ignores_hostile_topology() {
  HostCase host = make_case(8u);
  std::fill(host.active.begin(), host.active.end(), 0u);
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host));
  const std::int64_t hostile = std::numeric_limits<std::int64_t>::min();
  CUDA_CHECK(
      cudaMemcpy(device.atom_offsets.get(), &hostile, sizeof(hostile), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device.matrix_offsets.get() + 3, &hostile, sizeof(hostile),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device.shell_orbital_offsets.get() + 1, &hostile, sizeof(hostile),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(launch(device, host));
  CUDA_CHECK(cudaDeviceSynchronize());
  Population actual;
  std::vector<std::uint32_t> errors;
  std::uint32_t diagnostic = 99u;
  CUDA_CHECK(device.download(host, actual, errors, diagnostic));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(equal_population(actual, sentinel_population(host)));
  CHECK(std::all_of(errors.begin(), errors.end(), [](std::uint32_t value) { return value == 0u; }));
  CHECK(diagnostic == 0u);
  return 0;
}

int test_hostile_offsets_and_invalid_activity() {
  for (int target = 0; target < 7; ++target) {
    for (std::int64_t hostile :
         {std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::max()}) {
      HostCase host = make_case(1u);
      DeviceFixture device;
      CUDA_CHECK(device.initialize(host));
      std::int64_t* destination = nullptr;
      switch (target) {
        case 0:
          destination = device.atom_offsets.get() + 1;
          break;
        case 1:
          destination = device.batch_shell_offsets.get() + 1;
          break;
        case 2:
          destination = device.batch_orbital_offsets.get() + 1;
          break;
        case 3:
          destination = device.matrix_offsets.get() + 1;
          break;
        case 4:
          destination = device.atom_shell_offsets.get() + 1;
          break;
        case 5:
          destination = device.shell_orbital_offsets.get() + 1;
          break;
        default:
          destination = device.shell_to_atom.get();
          break;
      }
      CUDA_CHECK(cudaMemcpy(destination, &hostile, sizeof(hostile), cudaMemcpyHostToDevice));
      CUDA_CHECK(launch(device, host));
      CUDA_CHECK(cudaDeviceSynchronize());
      Population actual;
      std::vector<std::uint32_t> errors;
      std::uint32_t diagnostic = 0u;
      CUDA_CHECK(device.download(host, actual, errors, diagnostic));
      CUDA_CHECK(cudaDeviceSynchronize());
      CHECK(slice_is_sentinel(host, actual, 0u));
      CHECK(errors[0] != 0u);
      CHECK(diagnostic == errors[0]);
    }
  }
  HostCase host = make_case(1u);
  host.active[0] = 2u;
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host));
  CUDA_CHECK(launch(device, host));
  CUDA_CHECK(cudaDeviceSynchronize());
  Population actual;
  std::vector<std::uint32_t> errors;
  std::uint32_t diagnostic = 0u;
  CUDA_CHECK(device.download(host, actual, errors, diagnostic));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(slice_is_sentinel(host, actual, 0u));
  CHECK(errors[0] == static_cast<std::uint32_t>(Gfn2MullikenDeviceError::kInvalidActiveMask));
  return 0;
}

int test_reduction_overflow_is_transactional() {
  HostCase host = make_case(1u, true);
  std::fill(host.density.begin(), host.density.end(), -0.5 * std::numeric_limits<double>::max());
  std::fill(host.overlap.begin(), host.overlap.end(), 1.0);
  std::fill(host.dipole_integrals.begin(), host.dipole_integrals.end(), 1.0);
  std::fill(host.quadrupole_integrals.begin(), host.quadrupole_integrals.end(), 1.0);
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host));
  CUDA_CHECK(launch(device, host));
  CUDA_CHECK(cudaDeviceSynchronize());
  Population actual;
  std::vector<std::uint32_t> errors;
  std::uint32_t diagnostic = 0u;
  CUDA_CHECK(device.download(host, actual, errors, diagnostic));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(slice_is_sentinel(host, actual, 0u));
  CHECK(errors[0] == static_cast<std::uint32_t>(Gfn2MullikenDeviceError::kNonfiniteContraction));
  CHECK(diagnostic == errors[0]);
  return 0;
}

int test_sticky_error_alias_and_provenance_rejection() {
  HostCase host = make_case(1u);
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host));
  const std::uint32_t sticky = 77u;
  CUDA_CHECK(device.system_errors.fill(0u, 1u));
  CUDA_CHECK(device.device_error.copy_from(&sticky, 1u));
  const Gfn2MullikenDeviceBatch batch = device.batch(host);
  const Gfn2MullikenDeviceInput input = device.input(host);
  const Gfn2MullikenDeviceActivity activity = device.activity(host);
  Gfn2MullikenDevicePopulation population = device.population(host);
  Gfn2MullikenDeviceWorkspace workspace = device.workspace(host);
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_mulliken_population_cuda(
      batch, input, activity, population, workspace, device.system_errors.get(),
      device.device_error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  Population actual;
  std::vector<std::uint32_t> errors;
  std::uint32_t diagnostic = 0u;
  CUDA_CHECK(device.download(host, actual, errors, diagnostic));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(equal_population(actual, sentinel_population(host)));
  CHECK(diagnostic == sticky);

  Gfn2MullikenDeviceInput wrong_input = input;
  wrong_input.plan_token ^= 1u;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_mulliken_population_cuda(
            batch, wrong_input, activity, population, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  Gfn2MullikenDeviceActivity wrong_activity = activity;
  wrong_activity.plan_token ^= 1u;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_mulliken_population_cuda(
            batch, input, wrong_activity, population, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  Gfn2MullikenDevicePopulation wrong_population = population;
  wrong_population.plan_token ^= 1u;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_mulliken_population_cuda(
            batch, input, activity, wrong_population, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  Gfn2MullikenDeviceWorkspace wrong_workspace = workspace;
  wrong_workspace.plan_token ^= 1u;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_mulliken_population_cuda(
            batch, input, activity, population, wrong_workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  Gfn2MullikenDeviceBatch wrong_batch = batch;
  wrong_batch.plan_token = 0u;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_mulliken_population_cuda(
            wrong_batch, input, activity, population, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  Gfn2MullikenDevicePopulation alias_population = population;
  alias_population.qsh = workspace.qsh_scratch;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_mulliken_population_cuda(
            batch, input, activity, alias_population, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  Gfn2MullikenDeviceWorkspace alias_workspace = workspace;
  alias_workspace.qat_scratch = population.qat;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_mulliken_population_cuda(
            batch, input, activity, population, alias_workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  Gfn2MullikenDeviceBatch misaligned_batch = batch;
  misaligned_batch.matrix_offsets = reinterpret_cast<const std::int64_t*>(
      reinterpret_cast<const char*>(batch.matrix_offsets) + 1);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_mulliken_population_cuda(
            misaligned_batch, input, activity, population, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  return 0;
}

}  // namespace

int main() {
  for (std::size_t batch_size : {1u, 8u, 32u, 128u}) {
    if (const int status = run_parity(batch_size, false); status != 0) {
      std::fprintf(stderr, "CUDA Mulliken batch-%zu parity failed at line %d\n", batch_size,
                   status);
      return status;
    }
  }
  if (const int status = run_parity(8u, true); status != 0) {
    std::fprintf(stderr, "CUDA Mulliken Graph test failed at line %d\n", status);
    return status;
  }
  for (std::size_t batch_size : {1u, 8u, 32u, 128u}) {
    if (const int status = run_spin_parity(batch_size, false); status != 0) {
      std::fprintf(stderr, "CUDA spin Mulliken batch-%zu parity failed at line %d\n", batch_size,
                   status);
      return status;
    }
  }
  if (const int status = run_spin_parity(8u, true); status != 0) {
    std::fprintf(stderr, "CUDA spin Mulliken Graph test failed at line %d\n", status);
    return status;
  }
  if (const int status = test_spin_swap_graph_replay(); status != 0) {
    std::fprintf(stderr, "CUDA spin Mulliken swap replay failed at line %d\n", status);
    return status;
  }
  if (const int status = test_equal_spin_densities_have_exact_zero_magnetization(); status != 0) {
    std::fprintf(stderr, "CUDA spin Mulliken zero magnetization failed at line %d\n", status);
    return status;
  }
  if (const int status = test_spin_inactive_and_beta_failure_isolation(); status != 0) {
    std::fprintf(stderr, "CUDA spin Mulliken isolation failed at line %d\n", status);
    return status;
  }
  if (const int status = test_spin_layout_validation_and_offsets(); status != 0) {
    std::fprintf(stderr, "CUDA spin Mulliken validation failed at line %d\n", status);
    return status;
  }
  if (const int status = test_terminal_skip_and_failure_isolation(); status != 0) {
    std::fprintf(stderr, "CUDA Mulliken isolation test failed at line %d\n", status);
    return status;
  }
  if (const int status = test_empty_active_set_ignores_hostile_topology(); status != 0) {
    std::fprintf(stderr, "CUDA Mulliken terminal topology skip failed at line %d\n", status);
    return status;
  }
  if (const int status = test_hostile_offsets_and_invalid_activity(); status != 0) {
    std::fprintf(stderr, "CUDA Mulliken hostile metadata test failed at line %d\n", status);
    return status;
  }
  if (const int status = test_reduction_overflow_is_transactional(); status != 0) {
    std::fprintf(stderr, "CUDA Mulliken overflow test failed at line %d\n", status);
    return status;
  }
  if (const int status = test_sticky_error_alias_and_provenance_rejection(); status != 0) {
    std::fprintf(stderr, "CUDA Mulliken validation test failed at line %d\n", status);
    return status;
  }
  std::puts("CUDA Mulliken tests passed");
  return 0;
}
