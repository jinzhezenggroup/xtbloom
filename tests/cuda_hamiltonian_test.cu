#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_hamiltonian.cuh"
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
using gpuxtb::detail::cuda::assemble_gfn2_hamiltonian_cuda;
using gpuxtb::detail::cuda::assemble_gfn2_spin_hamiltonian_cuda;
using gpuxtb::detail::cuda::Gfn2HamiltonianDeviceActivity;
using gpuxtb::detail::cuda::Gfn2HamiltonianDeviceBatch;
using gpuxtb::detail::cuda::Gfn2HamiltonianDeviceError;
using gpuxtb::detail::cuda::Gfn2HamiltonianDeviceInput;
using gpuxtb::detail::cuda::Gfn2HamiltonianDeviceOutput;
using gpuxtb::detail::cuda::Gfn2HamiltonianDeviceWorkspace;
using gpuxtb::detail::cuda::reset_gfn2_hamiltonian_device_errors_cuda;
using gpuxtb::detail::gfn2::BasisPlan;
using gpuxtb::detail::gfn2::IntegralPlan;
using gpuxtb::detail::gfn2::MullikenHamiltonianView;
using gpuxtb::detail::gfn2::MullikenIntegralView;
using gpuxtb::detail::gfn2::MullikenPlan;
using gpuxtb::detail::gfn2::MullikenPotentialView;
using gpuxtb::detail::gfn2::MullikenWorkspace;
using gpuxtb::detail::gfn2::WavefunctionLayout;

constexpr std::uint64_t kPlanToken = 0x73a541c28de960bfULL;
constexpr double kSentinel = -973.375;

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

bool near(double actual, double expected, double absolute = 3.0e-13, double relative = 3.0e-13) {
  return std::abs(actual - expected) <=
         absolute + relative * std::max(std::abs(actual), std::abs(expected));
}

struct HostCase {
  std::int64_t batch_size = 0;
  std::vector<std::int64_t> atom_offsets{0};
  std::vector<std::int64_t> batch_shell_offsets{0};
  std::vector<std::int64_t> batch_orbital_offsets{0};
  std::vector<std::int64_t> matrix_offsets{0};
  std::vector<std::int64_t> atom_shell_offsets{0};
  std::vector<std::int64_t> shell_orbital_offsets{0};
  std::vector<std::int64_t> shell_to_atom;
  std::vector<std::int64_t> orbital_to_shell;
  std::vector<std::int64_t> orbital_to_atom;
  std::vector<double> h0;
  std::vector<double> overlap;
  std::vector<double> dipole_integrals;
  std::vector<double> quadrupole_integrals;
  std::vector<double> shell_scalar;
  std::vector<double> dipole_potential;
  std::vector<double> quadrupole_potential;
  std::vector<std::uint8_t> active;
};

struct HostSpinHamiltonianCase {
  std::vector<std::int32_t> channels;
  std::vector<std::int64_t> channel_offsets{0};
  std::vector<std::int64_t> orbital_offsets{0};
  std::vector<std::int64_t> matrix_offsets{0};
  std::vector<std::int64_t> shell_offsets{0};
  std::vector<std::int64_t> atom_offsets{0};
  std::vector<double> shell_scalar;
  std::vector<double> dipole_potential;
  std::vector<double> quadrupole_potential;
};

HostCase make_case(std::size_t batch_size) {
  HostCase data;
  data.batch_size = static_cast<std::int64_t>(batch_size);
  std::int64_t atoms_total = 0;
  std::int64_t shells_total = 0;
  std::int64_t orbitals_total = 0;
  std::int64_t matrices_total = 0;
  for (std::size_t system = 0; system < batch_size; ++system) {
    const std::int64_t atoms = 1 + static_cast<std::int64_t>(system % 2u);
    const std::int64_t orbital_begin = orbitals_total;
    for (std::int64_t local_atom = 0; local_atom < atoms; ++local_atom) {
      const std::int64_t shells = 1 + static_cast<std::int64_t>((system + local_atom) % 2u);
      for (std::int64_t local_shell = 0; local_shell < shells; ++local_shell) {
        const std::int64_t shell = shells_total;
        const std::int64_t orbitals =
            1 + static_cast<std::int64_t>((system + local_atom + local_shell) % 3u);
        data.shell_to_atom.push_back(atoms_total);
        data.shell_scalar.push_back(0.08 * static_cast<double>(1 + (shell * 7) % 13) - 0.31);
        for (std::int64_t orbital = 0; orbital < orbitals; ++orbital) {
          data.orbital_to_shell.push_back(shell);
          data.orbital_to_atom.push_back(atoms_total);
          ++orbitals_total;
        }
        ++shells_total;
        data.shell_orbital_offsets.push_back(orbitals_total);
      }
      ++atoms_total;
      data.atom_shell_offsets.push_back(shells_total);
    }
    const std::int64_t system_orbitals = orbitals_total - orbital_begin;
    matrices_total += system_orbitals * system_orbitals;
    data.atom_offsets.push_back(atoms_total);
    data.batch_shell_offsets.push_back(shells_total);
    data.batch_orbital_offsets.push_back(orbitals_total);
    data.matrix_offsets.push_back(matrices_total);
  }
  data.active.assign(batch_size, 1u);
  data.dipole_potential.resize(static_cast<std::size_t>(3 * atoms_total));
  data.quadrupole_potential.resize(static_cast<std::size_t>(6 * atoms_total));
  for (std::int64_t atom = 0; atom < atoms_total; ++atom) {
    for (int component = 0; component < 3; ++component) {
      data.dipole_potential[static_cast<std::size_t>(atom * 3 + component)] =
          0.035 * static_cast<double>(1 + (atom * 11 + component * 5) % 17) - 0.16;
    }
    for (int component = 0; component < 6; ++component) {
      data.quadrupole_potential[static_cast<std::size_t>(atom * 6 + component)] =
          0.019 * static_cast<double>(1 + (atom * 13 + component * 7) % 23) - 0.21;
    }
  }
  data.h0.resize(static_cast<std::size_t>(matrices_total));
  data.overlap.resize(static_cast<std::size_t>(matrices_total));
  data.dipole_integrals.resize(static_cast<std::size_t>(3 * matrices_total));
  data.quadrupole_integrals.resize(static_cast<std::size_t>(6 * matrices_total));
  for (std::int64_t matrix = 0; matrix < matrices_total; ++matrix) {
    /* Directed, deliberately nonsymmetric values catch accidental triangle mirroring. */
    data.h0[static_cast<std::size_t>(matrix)] =
        0.013 * static_cast<double>(1 + (matrix * 17) % 37) - 0.26;
    data.overlap[static_cast<std::size_t>(matrix)] =
        0.009 * static_cast<double>(1 + (matrix * 19) % 41) - 0.14;
    for (int component = 0; component < 3; ++component) {
      data.dipole_integrals[static_cast<std::size_t>(component * matrices_total + matrix)] =
          0.006 * static_cast<double>(1 + (matrix * (component + 3) + component * 29) % 43) - 0.12;
    }
    for (int component = 0; component < 6; ++component) {
      data.quadrupole_integrals[static_cast<std::size_t>(component * matrices_total + matrix)] =
          0.003 * static_cast<double>(1 + (matrix * (component + 5) + component * 31) % 47) - 0.07;
    }
  }
  return data;
}

std::vector<double> evaluate_cpu(const HostCase& data, double sentinel = kSentinel) {
  std::vector<double> result(static_cast<std::size_t>(data.matrix_offsets.back()), sentinel);
  const std::int64_t matrices = data.matrix_offsets.back();
  for (std::int64_t system = 0; system < data.batch_size; ++system) {
    if (data.active[static_cast<std::size_t>(system)] == 0u) {
      continue;
    }
    const std::int64_t orbital_begin = data.batch_orbital_offsets[static_cast<std::size_t>(system)];
    const std::int64_t orbitals =
        data.batch_orbital_offsets[static_cast<std::size_t>(system + 1)] - orbital_begin;
    const std::int64_t matrix_begin = data.matrix_offsets[static_cast<std::size_t>(system)];
    for (std::int64_t local_row = 0; local_row < orbitals; ++local_row) {
      const std::int64_t row = orbital_begin + local_row;
      for (std::int64_t local_column = local_row; local_column < orbitals; ++local_column) {
        const std::int64_t column = orbital_begin + local_column;
        const std::int64_t forward = matrix_begin + local_row * orbitals + local_column;
        const std::int64_t reverse = matrix_begin + local_column * orbitals + local_row;
        const std::int64_t row_shell = data.orbital_to_shell[static_cast<std::size_t>(row)];
        const std::int64_t column_shell = data.orbital_to_shell[static_cast<std::size_t>(column)];
        const std::int64_t row_atom = data.orbital_to_atom[static_cast<std::size_t>(row)];
        const std::int64_t column_atom = data.orbital_to_atom[static_cast<std::size_t>(column)];
        const double half_overlap = -0.5 * data.overlap[static_cast<std::size_t>(forward)];
        double shift = 0.0;
        shift =
            std::fma(half_overlap, data.shell_scalar[static_cast<std::size_t>(row_shell)], shift);
        shift = std::fma(half_overlap, data.shell_scalar[static_cast<std::size_t>(column_shell)],
                         shift);
        for (int component = 0; component < 3; ++component) {
          shift = std::fma(
              -0.5 *
                  data.dipole_integrals[static_cast<std::size_t>(component * matrices + forward)],
              data.dipole_potential[static_cast<std::size_t>(column_atom * 3 + component)], shift);
          shift = std::fma(
              -0.5 *
                  data.dipole_integrals[static_cast<std::size_t>(component * matrices + reverse)],
              data.dipole_potential[static_cast<std::size_t>(row_atom * 3 + component)], shift);
        }
        for (int component = 0; component < 6; ++component) {
          shift = std::fma(
              -0.5 * data.quadrupole_integrals[static_cast<std::size_t>(component * matrices +
                                                                        forward)],
              data.quadrupole_potential[static_cast<std::size_t>(column_atom * 6 + component)],
              shift);
          shift = std::fma(
              -0.5 * data.quadrupole_integrals[static_cast<std::size_t>(component * matrices +
                                                                        reverse)],
              data.quadrupole_potential[static_cast<std::size_t>(row_atom * 6 + component)], shift);
        }
        result[static_cast<std::size_t>(forward)] =
            data.h0[static_cast<std::size_t>(forward)] + shift;
        result[static_cast<std::size_t>(reverse)] =
            data.h0[static_cast<std::size_t>(reverse)] + shift;
      }
    }
  }
  return result;
}

std::vector<double> evaluate_spin_cpu(const HostCase& host, const HostSpinHamiltonianCase& spin) {
  HostCase alpha = host;
  HostCase beta = host;
  for (std::size_t system = 0; system < static_cast<std::size_t>(host.batch_size); ++system) {
    const std::int64_t shells =
        host.batch_shell_offsets[system + 1u] - host.batch_shell_offsets[system];
    const std::int64_t atoms = host.atom_offsets[system + 1u] - host.atom_offsets[system];
    for (std::int64_t local_shell = 0; local_shell < shells; ++local_shell) {
      const std::int64_t physical_shell = host.batch_shell_offsets[system] + local_shell;
      const double charge =
          spin.shell_scalar[static_cast<std::size_t>(spin.shell_offsets[system] + local_shell)];
      double alpha_value = charge;
      double beta_value = charge;
      if (spin.channels[system] == 2) {
        const double magnetization = spin.shell_scalar[static_cast<std::size_t>(
            spin.shell_offsets[system] + shells + local_shell)];
        alpha_value = 0.5 * charge + 0.5 * magnetization;
        beta_value = 0.5 * charge - 0.5 * magnetization;
      }
      alpha.shell_scalar[static_cast<std::size_t>(physical_shell)] = alpha_value;
      beta.shell_scalar[static_cast<std::size_t>(physical_shell)] = beta_value;
    }
    for (std::int64_t local_atom = 0; local_atom < atoms; ++local_atom) {
      const std::int64_t physical_atom = host.atom_offsets[system] + local_atom;
      const std::int64_t charge_atom = spin.atom_offsets[system] + local_atom;
      for (int component = 0; component < 3; ++component) {
        const double charge =
            spin.dipole_potential[static_cast<std::size_t>(3 * charge_atom + component)];
        double alpha_value = charge;
        double beta_value = charge;
        if (spin.channels[system] == 2) {
          const std::int64_t magnetization_atom = spin.atom_offsets[system] + atoms + local_atom;
          const double magnetization =
              spin.dipole_potential[static_cast<std::size_t>(3 * magnetization_atom + component)];
          alpha_value = 0.5 * charge + 0.5 * magnetization;
          beta_value = 0.5 * charge - 0.5 * magnetization;
        }
        alpha.dipole_potential[static_cast<std::size_t>(3 * physical_atom + component)] =
            alpha_value;
        beta.dipole_potential[static_cast<std::size_t>(3 * physical_atom + component)] = beta_value;
      }
      for (int component = 0; component < 6; ++component) {
        const double charge =
            spin.quadrupole_potential[static_cast<std::size_t>(6 * charge_atom + component)];
        double alpha_value = charge;
        double beta_value = charge;
        if (spin.channels[system] == 2) {
          const std::int64_t magnetization_atom = spin.atom_offsets[system] + atoms + local_atom;
          const double magnetization = spin.quadrupole_potential[static_cast<std::size_t>(
              6 * magnetization_atom + component)];
          alpha_value = 0.5 * charge + 0.5 * magnetization;
          beta_value = 0.5 * charge - 0.5 * magnetization;
        }
        alpha.quadrupole_potential[static_cast<std::size_t>(6 * physical_atom + component)] =
            alpha_value;
        beta.quadrupole_potential[static_cast<std::size_t>(6 * physical_atom + component)] =
            beta_value;
      }
    }
  }

  const std::vector<double> alpha_tmp = evaluate_cpu(alpha);
  const std::vector<double> beta_tmp = evaluate_cpu(beta);
  std::vector<double> result(static_cast<std::size_t>(spin.matrix_offsets.back()), kSentinel);
  for (std::size_t system = 0; system < static_cast<std::size_t>(host.batch_size); ++system) {
    if (host.active[system] == 0u) {
      continue;
    }
    const std::int64_t physical_begin = host.matrix_offsets[system];
    const std::int64_t elements = host.matrix_offsets[system + 1u] - physical_begin;
    for (int channel = 0; channel < spin.channels[system]; ++channel) {
      const std::vector<double>& temporary = channel == 0 ? alpha_tmp : beta_tmp;
      const std::int64_t output_begin = spin.matrix_offsets[system] + channel * elements;
      for (std::int64_t local = 0; local < elements; ++local) {
        double value = temporary[static_cast<std::size_t>(physical_begin + local)];
        if (spin.channels[system] == 2) {
          const double h0 = host.h0[static_cast<std::size_t>(physical_begin + local)];
          const double delta = value - h0;
          value = h0 + 2.0 * delta;
        }
        result[static_cast<std::size_t>(output_begin + local)] = value;
      }
    }
  }
  return result;
}

struct DeviceFixture {
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> batch_shell_offsets;
  DeviceBuffer<std::int64_t> batch_orbital_offsets;
  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<std::int64_t> atom_shell_offsets;
  DeviceBuffer<std::int64_t> shell_orbital_offsets;
  DeviceBuffer<std::int64_t> shell_to_atom;
  DeviceBuffer<std::int64_t> orbital_to_shell;
  DeviceBuffer<std::int64_t> orbital_to_atom;
  DeviceBuffer<double> h0;
  DeviceBuffer<double> overlap;
  DeviceBuffer<double> dipole_integrals;
  DeviceBuffer<double> quadrupole_integrals;
  DeviceBuffer<double> shell_scalar;
  DeviceBuffer<double> dipole_potential;
  DeviceBuffer<double> quadrupole_potential;
  DeviceBuffer<std::uint8_t> active;
  DeviceBuffer<double> output;
  DeviceBuffer<double> scratch;
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
    UPLOAD(orbital_to_shell, host.orbital_to_shell)
    UPLOAD(orbital_to_atom, host.orbital_to_atom)
    UPLOAD(h0, host.h0)
    UPLOAD(overlap, host.overlap)
    UPLOAD(dipole_integrals, host.dipole_integrals)
    UPLOAD(quadrupole_integrals, host.quadrupole_integrals)
    UPLOAD(shell_scalar, host.shell_scalar)
    UPLOAD(dipole_potential, host.dipole_potential)
    UPLOAD(quadrupole_potential, host.quadrupole_potential)
    UPLOAD(active, host.active)
#undef UPLOAD
    const std::size_t matrices = host.h0.size();
    if (status == cudaSuccess) {
      status = output.allocate(matrices);
    }
    if (status == cudaSuccess) {
      status = scratch.allocate(matrices);
    }
    if (status == cudaSuccess) {
      status = sequence_active.allocate(1u);
    }
    if (status == cudaSuccess) {
      status = system_errors.allocate(static_cast<std::size_t>(host.batch_size));
    }
    if (status == cudaSuccess) {
      status = device_error.allocate(1u);
    }
    return status == cudaSuccess ? output.fill(kSentinel, matrices, stream) : status;
  }

  Gfn2HamiltonianDeviceBatch batch(const HostCase& host) const {
    return {host.batch_size,
            host.atom_offsets.back(),
            static_cast<std::int64_t>(host.shell_to_atom.size()),
            static_cast<std::int64_t>(host.orbital_to_shell.size()),
            static_cast<std::int64_t>(host.h0.size()),
            kPlanToken,
            static_cast<std::int64_t>(host.atom_offsets.size()),
            static_cast<std::int64_t>(host.batch_shell_offsets.size()),
            static_cast<std::int64_t>(host.batch_orbital_offsets.size()),
            static_cast<std::int64_t>(host.matrix_offsets.size()),
            static_cast<std::int64_t>(host.atom_shell_offsets.size()),
            static_cast<std::int64_t>(host.shell_orbital_offsets.size()),
            static_cast<std::int64_t>(host.shell_to_atom.size()),
            static_cast<std::int64_t>(host.orbital_to_shell.size()),
            static_cast<std::int64_t>(host.orbital_to_atom.size()),
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

  Gfn2HamiltonianDeviceInput input(const HostCase& host) const {
    return {h0.get(),
            static_cast<std::int64_t>(host.h0.size()),
            overlap.get(),
            static_cast<std::int64_t>(host.overlap.size()),
            dipole_integrals.get(),
            static_cast<std::int64_t>(host.dipole_integrals.size()),
            quadrupole_integrals.get(),
            static_cast<std::int64_t>(host.quadrupole_integrals.size()),
            shell_scalar.get(),
            static_cast<std::int64_t>(host.shell_scalar.size()),
            dipole_potential.get(),
            static_cast<std::int64_t>(host.dipole_potential.size()),
            quadrupole_potential.get(),
            static_cast<std::int64_t>(host.quadrupole_potential.size()),
            kPlanToken};
  }

  Gfn2HamiltonianDeviceActivity activity(const HostCase& host) const {
    return {active.get(), host.batch_size, kPlanToken};
  }

  Gfn2HamiltonianDeviceOutput result(const HostCase& host) {
    return {output.get(), static_cast<std::int64_t>(host.h0.size()), kPlanToken};
  }

  Gfn2HamiltonianDeviceWorkspace workspace(const HostCase& host) {
    return {scratch.get(), static_cast<std::int64_t>(host.h0.size()), sequence_active.get(), 1,
            kPlanToken};
  }
};

HostSpinHamiltonianCase make_spin_hamiltonian_case(const HostCase& host) {
  HostSpinHamiltonianCase spin;
  for (std::size_t system = 0; system < static_cast<std::size_t>(host.batch_size); ++system) {
    const std::int32_t channels = system % 2u == 0u ? 1 : 2;
    const std::int64_t orbitals =
        host.batch_orbital_offsets[system + 1u] - host.batch_orbital_offsets[system];
    const std::int64_t matrices = orbitals * orbitals;
    const std::int64_t shells =
        host.batch_shell_offsets[system + 1u] - host.batch_shell_offsets[system];
    const std::int64_t atoms = host.atom_offsets[system + 1u] - host.atom_offsets[system];
    spin.channels.push_back(channels);
    spin.channel_offsets.push_back(spin.channel_offsets.back() + channels);
    spin.orbital_offsets.push_back(spin.orbital_offsets.back() + channels * orbitals);
    spin.matrix_offsets.push_back(spin.matrix_offsets.back() + channels * matrices);
    spin.shell_offsets.push_back(spin.shell_offsets.back() + channels * shells);
    spin.atom_offsets.push_back(spin.atom_offsets.back() + channels * atoms);
  }
  spin.shell_scalar.resize(static_cast<std::size_t>(spin.shell_offsets.back()));
  spin.dipole_potential.resize(static_cast<std::size_t>(3 * spin.atom_offsets.back()));
  spin.quadrupole_potential.resize(static_cast<std::size_t>(6 * spin.atom_offsets.back()));
  for (std::size_t system = 0; system < static_cast<std::size_t>(host.batch_size); ++system) {
    const std::int64_t shells =
        host.batch_shell_offsets[system + 1u] - host.batch_shell_offsets[system];
    const std::int64_t atoms = host.atom_offsets[system + 1u] - host.atom_offsets[system];
    for (std::int64_t local_shell = 0; local_shell < shells; ++local_shell) {
      const std::int64_t physical_shell = host.batch_shell_offsets[system] + local_shell;
      spin.shell_scalar[static_cast<std::size_t>(spin.shell_offsets[system] + local_shell)] =
          host.shell_scalar[static_cast<std::size_t>(physical_shell)];
      if (spin.channels[system] == 2) {
        spin.shell_scalar[static_cast<std::size_t>(spin.shell_offsets[system] + shells +
                                                   local_shell)] =
            0.043 * static_cast<double>(
                        1 + (system * 13u + static_cast<std::size_t>(local_shell) * 5u) % 29u) -
            0.24;
      }
    }
    for (std::int64_t local_atom = 0; local_atom < atoms; ++local_atom) {
      const std::int64_t physical_atom = host.atom_offsets[system] + local_atom;
      const std::int64_t charge_atom = spin.atom_offsets[system] + local_atom;
      for (int component = 0; component < 3; ++component) {
        spin.dipole_potential[static_cast<std::size_t>(3 * charge_atom + component)] =
            host.dipole_potential[static_cast<std::size_t>(3 * physical_atom + component)];
      }
      for (int component = 0; component < 6; ++component) {
        spin.quadrupole_potential[static_cast<std::size_t>(6 * charge_atom + component)] =
            host.quadrupole_potential[static_cast<std::size_t>(6 * physical_atom + component)];
      }
      if (spin.channels[system] == 2) {
        const std::int64_t magnetization_atom = spin.atom_offsets[system] + atoms + local_atom;
        for (int component = 0; component < 3; ++component) {
          spin.dipole_potential[static_cast<std::size_t>(3 * magnetization_atom + component)] =
              0.017 * static_cast<double>(1 + (system * 7u + local_atom * 3 + component) % 19) -
              0.12;
        }
        for (int component = 0; component < 6; ++component) {
          spin.quadrupole_potential[static_cast<std::size_t>(6 * magnetization_atom + component)] =
              0.009 * static_cast<double>(1 + (system * 11u + local_atom * 5 + component) % 23) -
              0.08;
        }
      }
    }
  }
  return spin;
}

struct SpinHamiltonianFixture {
  DeviceBuffer<std::int32_t> channels;
  DeviceBuffer<std::int64_t> channel_offsets;
  DeviceBuffer<std::int64_t> orbital_offsets;
  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<std::int64_t> shell_offsets;
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<double> shell_scalar;
  DeviceBuffer<double> dipole_potential;
  DeviceBuffer<double> quadrupole_potential;
  DeviceBuffer<double> output;
  DeviceBuffer<double> scratch;
  DeviceBuffer<std::uint32_t> sequence_active;

  cudaError_t initialize(const HostSpinHamiltonianCase& host, cudaStream_t stream = nullptr) {
    cudaError_t status = upload(channels, host.channels, stream);
#define UPLOAD_SPIN(field, source)          \
  if (status == cudaSuccess) {              \
    status = upload(field, source, stream); \
  }
    UPLOAD_SPIN(channel_offsets, host.channel_offsets)
    UPLOAD_SPIN(orbital_offsets, host.orbital_offsets)
    UPLOAD_SPIN(matrix_offsets, host.matrix_offsets)
    UPLOAD_SPIN(shell_offsets, host.shell_offsets)
    UPLOAD_SPIN(atom_offsets, host.atom_offsets)
    UPLOAD_SPIN(shell_scalar, host.shell_scalar)
    UPLOAD_SPIN(dipole_potential, host.dipole_potential)
    UPLOAD_SPIN(quadrupole_potential, host.quadrupole_potential)
#undef UPLOAD_SPIN
    if (status == cudaSuccess) {
      status = output.allocate(static_cast<std::size_t>(host.matrix_offsets.back()));
    }
    if (status == cudaSuccess) {
      status = scratch.allocate(static_cast<std::size_t>(host.matrix_offsets.back()));
    }
    if (status == cudaSuccess) {
      status = sequence_active.allocate(1u);
    }
    return status == cudaSuccess
               ? output.fill(kSentinel, static_cast<std::size_t>(host.matrix_offsets.back()),
                             stream)
               : status;
  }

  Gfn2WavefunctionLayoutView layout(const HostCase& physical,
                                    const HostSpinHamiltonianCase& host) const {
    return {Gfn2PlanMemorySpace::kCudaDevice,
            kPlanToken,
            0x51cc0deULL,
            physical.batch_size,
            host.channel_offsets.back(),
            host.orbital_offsets.back(),
            host.matrix_offsets.back(),
            host.shell_offsets.back(),
            host.atom_offsets.back(),
            static_cast<std::int64_t>(host.channels.size()),
            static_cast<std::int64_t>(host.channel_offsets.size()),
            static_cast<std::int64_t>(host.orbital_offsets.size()),
            static_cast<std::int64_t>(host.matrix_offsets.size()),
            static_cast<std::int64_t>(host.shell_offsets.size()),
            static_cast<std::int64_t>(host.atom_offsets.size()),
            channels.get(),
            channel_offsets.get(),
            orbital_offsets.get(),
            matrix_offsets.get(),
            shell_offsets.get(),
            atom_offsets.get()};
  }

  Gfn2HamiltonianDeviceInput input(const DeviceFixture& physical, const HostCase& physical_host,
                                   const HostSpinHamiltonianCase& host) const {
    return {physical.h0.get(),
            static_cast<std::int64_t>(physical_host.h0.size()),
            physical.overlap.get(),
            static_cast<std::int64_t>(physical_host.overlap.size()),
            physical.dipole_integrals.get(),
            static_cast<std::int64_t>(physical_host.dipole_integrals.size()),
            physical.quadrupole_integrals.get(),
            static_cast<std::int64_t>(physical_host.quadrupole_integrals.size()),
            shell_scalar.get(),
            static_cast<std::int64_t>(host.shell_scalar.size()),
            dipole_potential.get(),
            static_cast<std::int64_t>(host.dipole_potential.size()),
            quadrupole_potential.get(),
            static_cast<std::int64_t>(host.quadrupole_potential.size()),
            kPlanToken};
  }

  Gfn2HamiltonianDeviceOutput result(const HostSpinHamiltonianCase& host) {
    return {output.get(), host.matrix_offsets.back(), kPlanToken};
  }

  Gfn2HamiltonianDeviceWorkspace workspace(const HostSpinHamiltonianCase& host) {
    return {scratch.get(), host.matrix_offsets.back(), sequence_active.get(), 1, kPlanToken};
  }
};

cudaError_t launch_spin_hamiltonian(DeviceFixture& physical, SpinHamiltonianFixture& spin_device,
                                    const HostCase& host, const HostSpinHamiltonianCase& spin_host,
                                    cudaStream_t stream = nullptr) {
  cudaError_t status = reset_gfn2_hamiltonian_device_errors_cuda(
      host.batch_size, physical.system_errors.get(), physical.device_error.get(), stream);
  if (status == cudaSuccess) {
    status = assemble_gfn2_spin_hamiltonian_cuda(
        physical.batch(host), spin_device.layout(host, spin_host),
        spin_device.input(physical, host, spin_host), physical.activity(host),
        spin_device.result(spin_host), spin_device.workspace(spin_host),
        physical.system_errors.get(), physical.device_error.get(), stream);
  }
  return status;
}

cudaError_t launch(DeviceFixture& device, const HostCase& host, cudaStream_t stream = nullptr) {
  const auto batch = device.batch(host);
  const auto input = device.input(host);
  const auto activity = device.activity(host);
  const auto output = device.result(host);
  const auto workspace = device.workspace(host);
  cudaError_t status = reset_gfn2_hamiltonian_device_errors_cuda(
      host.batch_size, device.system_errors.get(), device.device_error.get(), stream);
  return status == cudaSuccess ? assemble_gfn2_hamiltonian_cuda(
                                     batch, input, activity, output, workspace,
                                     device.system_errors.get(), device.device_error.get(), stream)
                               : status;
}

/* Compare against the production CPU Mulliken implementation on an actual
 * GFN2 s/p/d basis.  The synthetic tests below deliberately use arbitrary
 * directed matrices; this gate independently protects the public topology,
 * component ordering, ket-origin convention, signs, and factors. */
int test_production_cpu_parity() {
  const std::vector<std::int64_t> atom_offsets{0, 3};
  /* H/C/Cl gives an even restricted electron count and exercises s/p/d. */
  const std::vector<std::int32_t> atomic_numbers{1, 6, 17};
  const std::vector<double> charges{0.0};
  const std::vector<std::int32_t> unpaired{0};
  const std::vector<std::int32_t> spin_channels{1};
  BasisPlan basis;
  IntegralPlan integral_plan;
  WavefunctionLayout wavefunction;
  MullikenPlan plan;
  std::string error;
  CHECK(gpuxtb::detail::gfn2::make_basis_plan(1, static_cast<std::int64_t>(atomic_numbers.size()),
                                              atom_offsets.data(), atomic_numbers.data(), basis,
                                              error) == GPUXTB_STATUS_SUCCESS);
  CHECK(basis.maximum_angular_momentum >= 2u);
  CHECK(gpuxtb::detail::gfn2::make_integral_plan(basis, integral_plan, error) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::make_wavefunction_layout(
            basis, atomic_numbers.data(), charges.data(), unpaired.data(), spin_channels.data(),
            wavefunction, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::make_mulliken_plan(basis, integral_plan, wavefunction, plan, error) ==
        GPUXTB_STATUS_SUCCESS);

  HostCase host;
  host.batch_size = 1;
  host.atom_offsets = basis.atom_offsets;
  host.batch_shell_offsets = basis.batch_shell_offsets;
  host.batch_orbital_offsets = basis.batch_orbital_offsets;
  host.matrix_offsets = integral_plan.matrix_offsets;
  host.atom_shell_offsets = basis.atom_shell_offsets;
  host.shell_orbital_offsets = basis.shell_orbital_offsets;
  host.shell_to_atom = plan.shell_to_atom();
  host.orbital_to_shell = plan.orbital_to_shell();
  host.orbital_to_atom = plan.orbital_to_atom();
  host.active = {1u};
  const std::int64_t matrices = plan.matrix_elements();
  host.h0.resize(static_cast<std::size_t>(matrices));
  host.overlap.resize(static_cast<std::size_t>(matrices));
  host.dipole_integrals.resize(static_cast<std::size_t>(3 * matrices));
  host.quadrupole_integrals.resize(static_cast<std::size_t>(6 * matrices));
  host.shell_scalar.resize(static_cast<std::size_t>(plan.total_shells()));
  host.dipole_potential.resize(static_cast<std::size_t>(3 * plan.total_atoms()));
  host.quadrupole_potential.resize(static_cast<std::size_t>(6 * plan.total_atoms()));
  for (std::int64_t matrix = 0; matrix < matrices; ++matrix) {
    host.h0[static_cast<std::size_t>(matrix)] =
        0.017 * static_cast<double>(1 + (matrix * 13) % 31) - 0.29;
    host.overlap[static_cast<std::size_t>(matrix)] =
        0.011 * static_cast<double>(1 + (matrix * 19) % 37) - 0.18;
    for (std::int64_t component = 0; component < 3; ++component) {
      host.dipole_integrals[static_cast<std::size_t>(component * matrices + matrix)] =
          0.007 * static_cast<double>(1 + (matrix * (component + 5) + component * 17) % 41) - 0.15;
    }
    for (std::int64_t component = 0; component < 6; ++component) {
      host.quadrupole_integrals[static_cast<std::size_t>(component * matrices + matrix)] =
          0.004 * static_cast<double>(1 + (matrix * (component + 7) + component * 23) % 43) - 0.09;
    }
  }
  for (std::int64_t shell = 0; shell < plan.total_shells(); ++shell) {
    host.shell_scalar[static_cast<std::size_t>(shell)] =
        0.031 * static_cast<double>(1 + (shell * 7) % 19) - 0.24;
  }
  for (std::int64_t atom = 0; atom < plan.total_atoms(); ++atom) {
    for (std::int64_t component = 0; component < 3; ++component) {
      host.dipole_potential[static_cast<std::size_t>(atom * 3 + component)] =
          0.023 * static_cast<double>(1 + (atom * 11 + component * 5) % 17) - 0.19;
    }
    for (std::int64_t component = 0; component < 6; ++component) {
      host.quadrupole_potential[static_cast<std::size_t>(atom * 6 + component)] =
          0.013 * static_cast<double>(1 + (atom * 13 + component * 7) % 29) - 0.16;
    }
  }

  std::vector<double> expected = host.h0;
  std::vector<double> atomic_scalar(static_cast<std::size_t>(plan.atom_population_elements()), 0.0);
  std::vector<double> scratch(static_cast<std::size_t>(plan.hamiltonian_scratch_elements()));
  const MullikenIntegralView integrals{host.overlap.data(), host.dipole_integrals.data(),
                                       host.quadrupole_integrals.data(), matrices, plan.identity()};
  const MullikenPotentialView potential{atomic_scalar.data(),
                                        plan.atom_population_elements(),
                                        host.shell_scalar.data(),
                                        plan.shell_population_elements(),
                                        host.dipole_potential.data(),
                                        plan.dipole_population_elements(),
                                        host.quadrupole_potential.data(),
                                        plan.quadrupole_population_elements(),
                                        plan.identity()};
  const MullikenHamiltonianView hamiltonian{expected.data(), plan.density_elements(),
                                            plan.identity()};
  const MullikenWorkspace workspace{scratch.data(), static_cast<std::int64_t>(scratch.size())};
  CHECK(gpuxtb::detail::gfn2::add_mulliken_hamiltonian_cpu(
            plan, integrals, potential, hamiltonian, workspace, error) == GPUXTB_STATUS_SUCCESS);

  DeviceFixture device;
  CUDA_CHECK(device.initialize(host));
  CUDA_CHECK(launch(device, host));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> actual(expected.size());
  CUDA_CHECK(device.output.copy_to(actual.data(), actual.size()));
  CUDA_CHECK(cudaDeviceSynchronize());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    CHECK(near(actual[index], expected[index]));
  }
  return 0;
}

int run_parity(std::size_t batch_size, bool graph) {
  HostCase host = make_case(batch_size);
  const std::vector<double> expected = evaluate_cpu(host);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (graph) {
    cudaGraph_t captured = nullptr;
    cudaGraphExec_t executable = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    CUDA_CHECK(launch(device, host, stream));
    CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
    CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    /* Replay the same executable after poisoning the public output.  The
     * captured reset and unpublished workspace must make every replay a
     * complete independent SCC-stage invocation. */
    CUDA_CHECK(device.output.fill(kSentinel, expected.size(), stream));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(captured));
  } else {
    CUDA_CHECK(launch(device, host, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  std::vector<double> actual(expected.size());
  std::vector<std::uint32_t> errors(static_cast<std::size_t>(host.batch_size));
  std::uint32_t diagnostic = 99u;
  CUDA_CHECK(device.output.copy_to(actual.data(), actual.size(), stream));
  CUDA_CHECK(device.system_errors.copy_to(errors.data(), errors.size(), stream));
  CUDA_CHECK(device.device_error.copy_to(&diagnostic, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (std::size_t index = 0; index < actual.size(); ++index) {
    CHECK(near(actual[index], expected[index]));
  }
  CHECK(std::all_of(errors.begin(), errors.end(), [](std::uint32_t error) { return error == 0u; }));
  CHECK(diagnostic == 0u);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

bool system_is_sentinel(const HostCase& host, const std::vector<double>& matrix,
                        std::size_t system) {
  for (std::int64_t element = host.matrix_offsets[system];
       element < host.matrix_offsets[system + 1u]; ++element) {
    if (matrix[static_cast<std::size_t>(element)] != kSentinel) {
      return false;
    }
  }
  return true;
}

int test_inactive_skip_and_peer_isolation() {
  HostCase host = make_case(8u);
  constexpr std::size_t inactive = 2u;
  constexpr std::size_t bad_h0 = 4u;
  constexpr std::size_t bad_integral = 5u;
  constexpr std::size_t bad_potential = 6u;
  host.active[inactive] = 0u;
  host.h0[static_cast<std::size_t>(host.matrix_offsets[inactive])] =
      std::numeric_limits<double>::quiet_NaN();
  host.shell_scalar[static_cast<std::size_t>(host.batch_shell_offsets[inactive])] =
      std::numeric_limits<double>::infinity();
  host.h0[static_cast<std::size_t>(host.matrix_offsets[bad_h0])] =
      std::numeric_limits<double>::quiet_NaN();
  host.dipole_integrals[static_cast<std::size_t>(host.matrix_offsets[bad_integral])] =
      std::numeric_limits<double>::infinity();
  host.shell_scalar[static_cast<std::size_t>(host.batch_shell_offsets[bad_potential])] =
      std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> expected = evaluate_cpu(host);
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host));
  CUDA_CHECK(launch(device, host));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> actual(host.h0.size());
  std::vector<std::uint32_t> errors(static_cast<std::size_t>(host.batch_size));
  CUDA_CHECK(device.output.copy_to(actual.data(), actual.size()));
  CUDA_CHECK(device.system_errors.copy_to(errors.data(), errors.size()));
  CUDA_CHECK(cudaDeviceSynchronize());
  for (std::size_t system = 0; system < 8u; ++system) {
    if (system == inactive || system == bad_h0 || system == bad_integral ||
        system == bad_potential) {
      CHECK(system_is_sentinel(host, actual, system));
    } else {
      for (std::int64_t element = host.matrix_offsets[system];
           element < host.matrix_offsets[system + 1u]; ++element) {
        CHECK(near(actual[static_cast<std::size_t>(element)],
                   expected[static_cast<std::size_t>(element)]));
      }
    }
  }
  CHECK(errors[inactive] == 0u);
  CHECK(errors[bad_h0] == static_cast<std::uint32_t>(Gfn2HamiltonianDeviceError::kNonfiniteH0));
  CHECK(errors[bad_integral] ==
        static_cast<std::uint32_t>(Gfn2HamiltonianDeviceError::kNonfiniteMultipoleIntegral));
  CHECK(errors[bad_potential] ==
        static_cast<std::uint32_t>(Gfn2HamiltonianDeviceError::kNonfinitePotential));
  return 0;
}

int test_all_inactive_hostile_topology() {
  HostCase host = make_case(8u);
  std::fill(host.active.begin(), host.active.end(), 0u);
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host));
  const std::int64_t hostile = std::numeric_limits<std::int64_t>::min();
  CUDA_CHECK(
      cudaMemcpy(device.atom_offsets.get(), &hostile, sizeof(hostile), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device.matrix_offsets.get() + 3, &hostile, sizeof(hostile),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(device.orbital_to_shell.get(), &hostile, sizeof(hostile), cudaMemcpyHostToDevice));
  CUDA_CHECK(launch(device, host));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> actual(host.h0.size());
  std::vector<std::uint32_t> errors(static_cast<std::size_t>(host.batch_size));
  std::uint32_t diagnostic = 99u;
  CUDA_CHECK(device.output.copy_to(actual.data(), actual.size()));
  CUDA_CHECK(device.system_errors.copy_to(errors.data(), errors.size()));
  CUDA_CHECK(device.device_error.copy_to(&diagnostic, 1u));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(std::all_of(actual.begin(), actual.end(), [](double value) { return value == kSentinel; }));
  CHECK(std::all_of(errors.begin(), errors.end(), [](std::uint32_t error) { return error == 0u; }));
  CHECK(diagnostic == 0u);
  return 0;
}

int test_hostile_offsets() {
  for (int target = 0; target < 9; ++target) {
    for (const std::int64_t hostile :
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
        case 6:
          destination = device.shell_to_atom.get();
          break;
        case 7:
          destination = device.orbital_to_shell.get();
          break;
        default:
          destination = device.orbital_to_atom.get();
          break;
      }
      CUDA_CHECK(cudaMemcpy(destination, &hostile, sizeof(hostile), cudaMemcpyHostToDevice));
      CUDA_CHECK(launch(device, host));
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<double> actual(host.h0.size());
      std::vector<std::uint32_t> errors(1u);
      CUDA_CHECK(device.output.copy_to(actual.data(), actual.size()));
      CUDA_CHECK(device.system_errors.copy_to(errors.data(), errors.size()));
      CUDA_CHECK(cudaDeviceSynchronize());
      CHECK(system_is_sentinel(host, actual, 0u));
      CHECK(errors[0] != 0u);
    }
  }
  return 0;
}

int test_arithmetic_overflow_is_transactional() {
  HostCase host = make_case(1u);
  std::fill(host.h0.begin(), host.h0.end(), 0.0);
  std::fill(host.overlap.begin(), host.overlap.end(), 0.0);
  std::fill(host.dipole_integrals.begin(), host.dipole_integrals.end(), 0.0);
  std::fill(host.quadrupole_integrals.begin(), host.quadrupole_integrals.end(), 0.0);
  std::fill(host.shell_scalar.begin(), host.shell_scalar.end(), 0.0);
  std::fill(host.dipole_potential.begin(), host.dipole_potential.end(), 0.0);
  std::fill(host.quadrupole_potential.begin(), host.quadrupole_potential.end(), 0.0);
  host.h0[0] = 0.75 * std::numeric_limits<double>::max();
  host.overlap[0] = -1.0;
  host.shell_scalar[0] = 0.5 * std::numeric_limits<double>::max();
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host));
  CUDA_CHECK(launch(device, host));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> actual(host.h0.size());
  std::vector<std::uint32_t> errors(1u);
  CUDA_CHECK(device.output.copy_to(actual.data(), actual.size()));
  CUDA_CHECK(device.system_errors.copy_to(errors.data(), errors.size()));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(system_is_sentinel(host, actual, 0u));
  CHECK(errors[0] ==
        static_cast<std::uint32_t>(Gfn2HamiltonianDeviceError::kNonfiniteAssemblyArithmetic));
  return 0;
}

int test_invalid_active_mask() {
  HostCase host = make_case(2u);
  host.active[0] = 2u;
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host));
  CUDA_CHECK(launch(device, host));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> actual(host.h0.size());
  std::vector<std::uint32_t> errors(2u);
  CUDA_CHECK(device.output.copy_to(actual.data(), actual.size()));
  CUDA_CHECK(device.system_errors.copy_to(errors.data(), errors.size()));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(system_is_sentinel(host, actual, 0u));
  CHECK(errors[0] == static_cast<std::uint32_t>(Gfn2HamiltonianDeviceError::kInvalidActiveMask));
  CHECK(errors[1] == 0u);
  CHECK(!system_is_sentinel(host, actual, 1u));
  return 0;
}

int test_sticky_alias_token_and_alignment() {
  HostCase host = make_case(1u);
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host));
  const std::uint32_t sticky = 79u;
  CUDA_CHECK(device.system_errors.fill(0u, 1u));
  CUDA_CHECK(device.device_error.copy_from(&sticky, 1u));
  auto batch = device.batch(host);
  auto input = device.input(host);
  auto activity = device.activity(host);
  auto output = device.result(host);
  auto workspace = device.workspace(host);
  CUDA_CHECK(assemble_gfn2_hamiltonian_cuda(batch, input, activity, output, workspace,
                                            device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> actual(host.h0.size());
  CUDA_CHECK(device.output.copy_to(actual.data(), actual.size()));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(system_is_sentinel(host, actual, 0u));

  auto wrong_input = input;
  wrong_input.plan_token ^= 1u;
  CHECK(assemble_gfn2_hamiltonian_cuda(batch, wrong_input, activity, output, workspace,
                                       device.system_errors.get(),
                                       device.device_error.get()) == cudaErrorInvalidValue);
  auto wrong_output = output;
  wrong_output.plan_token ^= 1u;
  CHECK(assemble_gfn2_hamiltonian_cuda(batch, input, activity, wrong_output, workspace,
                                       device.system_errors.get(),
                                       device.device_error.get()) == cudaErrorInvalidValue);
  auto wrong_activity = activity;
  wrong_activity.plan_token ^= 1u;
  CHECK(assemble_gfn2_hamiltonian_cuda(batch, input, wrong_activity, output, workspace,
                                       device.system_errors.get(),
                                       device.device_error.get()) == cudaErrorInvalidValue);
  auto wrong_workspace = workspace;
  wrong_workspace.plan_token ^= 1u;
  CHECK(assemble_gfn2_hamiltonian_cuda(batch, input, activity, output, wrong_workspace,
                                       device.system_errors.get(),
                                       device.device_error.get()) == cudaErrorInvalidValue);
  auto wrong_batch = batch;
  wrong_batch.plan_token = 0u;
  CHECK(assemble_gfn2_hamiltonian_cuda(wrong_batch, input, activity, output, workspace,
                                       device.system_errors.get(),
                                       device.device_error.get()) == cudaErrorInvalidValue);
  auto overflowing_shell_count = batch;
  overflowing_shell_count.total_shells = std::numeric_limits<std::int64_t>::max();
  CHECK(assemble_gfn2_hamiltonian_cuda(overflowing_shell_count, input, activity, output, workspace,
                                       device.system_errors.get(),
                                       device.device_error.get()) == cudaErrorInvalidValue);
  auto alias_output = output;
  alias_output.matrix = workspace.matrix_scratch;
  CHECK(assemble_gfn2_hamiltonian_cuda(batch, input, activity, alias_output, workspace,
                                       device.system_errors.get(),
                                       device.device_error.get()) == cudaErrorInvalidValue);
  auto alias_workspace = workspace;
  alias_workspace.matrix_scratch = output.matrix;
  CHECK(assemble_gfn2_hamiltonian_cuda(batch, input, activity, output, alias_workspace,
                                       device.system_errors.get(),
                                       device.device_error.get()) == cudaErrorInvalidValue);
  auto alias_input = input;
  alias_input.h0 = output.matrix;
  CHECK(assemble_gfn2_hamiltonian_cuda(batch, alias_input, activity, output, workspace,
                                       device.system_errors.get(),
                                       device.device_error.get()) == cudaErrorInvalidValue);
  alias_workspace = workspace;
  alias_workspace.matrix_scratch = const_cast<double*>(input.overlap);
  CHECK(assemble_gfn2_hamiltonian_cuda(batch, input, activity, output, alias_workspace,
                                       device.system_errors.get(),
                                       device.device_error.get()) == cudaErrorInvalidValue);
  auto misaligned = batch;
  misaligned.matrix_offsets = reinterpret_cast<const std::int64_t*>(
      reinterpret_cast<const char*>(batch.matrix_offsets) + 1);
  CHECK(assemble_gfn2_hamiltonian_cuda(misaligned, input, activity, output, workspace,
                                       device.system_errors.get(),
                                       device.device_error.get()) == cudaErrorInvalidValue);
  auto misaligned_input = input;
  misaligned_input.dipole_integrals =
      reinterpret_cast<const double*>(reinterpret_cast<const char*>(input.dipole_integrals) + 1);
  CHECK(assemble_gfn2_hamiltonian_cuda(batch, misaligned_input, activity, output, workspace,
                                       device.system_errors.get(),
                                       device.device_error.get()) == cudaErrorInvalidValue);
  CHECK(reset_gfn2_hamiltonian_device_errors_cuda(
            1, device.system_errors.get(), device.system_errors.get()) == cudaErrorInvalidValue);
  return 0;
}

int run_spin_hamiltonian_parity(std::size_t batch_size, bool graph) {
  HostCase host = make_case(batch_size);
  HostSpinHamiltonianCase spin_host = make_spin_hamiltonian_case(host);
  /* Model the vat contribution already collected by the mixed-spin potential
   * composer. This distinctive atom-owned scalar must reach both alpha and
   * beta Hamiltonians through the complete charge-channel shell field. */
  for (std::size_t system = 0; system < batch_size; ++system) {
    const std::int64_t shells =
        host.batch_shell_offsets[system + 1u] - host.batch_shell_offsets[system];
    for (std::int64_t local_shell = 0; local_shell < shells; ++local_shell) {
      const std::int64_t physical_shell = host.batch_shell_offsets[system] + local_shell;
      const std::int64_t atom = host.shell_to_atom[static_cast<std::size_t>(physical_shell)];
      spin_host
          .shell_scalar[static_cast<std::size_t>(spin_host.shell_offsets[system] + local_shell)] +=
          0.071 * static_cast<double>(1 + atom % 5);
    }
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture physical;
  SpinHamiltonianFixture spin_device;
  CUDA_CHECK(physical.initialize(host, stream));
  CUDA_CHECK(spin_device.initialize(spin_host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  /* The restricted slices of the new entry point must be bit-for-bit equal
   * to the legacy entry point when both consume the same complete scalar. */
  HostCase legacy_host = host;
  for (std::size_t system = 0; system < batch_size; system += 2u) {
    for (std::int64_t shell = host.batch_shell_offsets[system];
         shell < host.batch_shell_offsets[system + 1u]; ++shell) {
      legacy_host.shell_scalar[static_cast<std::size_t>(shell)] =
          spin_host.shell_scalar[static_cast<std::size_t>(spin_host.shell_offsets[system] + shell -
                                                          host.batch_shell_offsets[system])];
    }
  }
  CUDA_CHECK(physical.shell_scalar.copy_from(legacy_host.shell_scalar.data(),
                                             legacy_host.shell_scalar.size(), stream));
  CUDA_CHECK(launch(physical, legacy_host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<double> legacy(host.h0.size());
  CUDA_CHECK(physical.output.copy_to(legacy.data(), legacy.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  if (graph) {
    cudaGraph_t captured = nullptr;
    cudaGraphExec_t executable = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    CUDA_CHECK(launch_spin_hamiltonian(physical, spin_device, host, spin_host, stream));
    CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
    CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const std::size_t system = 1u;
    const std::int64_t shells =
        host.batch_shell_offsets[system + 1u] - host.batch_shell_offsets[system];
    spin_host.shell_scalar[static_cast<std::size_t>(spin_host.shell_offsets[system] + shells)] +=
        0.1875;
    CUDA_CHECK(spin_device.shell_scalar.copy_from(spin_host.shell_scalar.data(),
                                                  spin_host.shell_scalar.size(), stream));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(captured));
  } else {
    CUDA_CHECK(launch_spin_hamiltonian(physical, spin_device, host, spin_host, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  const std::vector<double> expected = evaluate_spin_cpu(host, spin_host);
  std::vector<double> actual(expected.size());
  std::vector<std::uint32_t> errors(batch_size);
  std::uint32_t diagnostic = 99u;
  CUDA_CHECK(spin_device.output.copy_to(actual.data(), actual.size(), stream));
  CUDA_CHECK(physical.system_errors.copy_to(errors.data(), errors.size(), stream));
  CUDA_CHECK(physical.device_error.copy_to(&diagnostic, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (std::size_t index = 0; index < actual.size(); ++index) {
    CHECK(near(actual[index], expected[index]));
  }
  for (std::size_t system = 0; system < batch_size; system += 2u) {
    const std::int64_t physical_begin = host.matrix_offsets[system];
    const std::int64_t elements = host.matrix_offsets[system + 1u] - physical_begin;
    const std::int64_t spin_begin = spin_host.matrix_offsets[system];
    for (std::int64_t local = 0; local < elements; ++local) {
      CHECK(actual[static_cast<std::size_t>(spin_begin + local)] ==
            legacy[static_cast<std::size_t>(physical_begin + local)]);
    }
  }
  CHECK(std::all_of(errors.begin(), errors.end(), [](std::uint32_t value) { return value == 0u; }));
  CHECK(diagnostic == 0u);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_spin_hamiltonian_failure_and_alias_contracts() {
  HostCase host = make_case(8u);
  HostSpinHamiltonianCase spin_host = make_spin_hamiltonian_case(host);
  constexpr std::size_t failed = 3u;
  const std::int64_t shells =
      host.batch_shell_offsets[failed + 1u] - host.batch_shell_offsets[failed];
  spin_host.shell_scalar[static_cast<std::size_t>(spin_host.shell_offsets[failed] + shells)] =
      std::numeric_limits<double>::quiet_NaN();
  DeviceFixture physical;
  SpinHamiltonianFixture spin_device;
  CUDA_CHECK(physical.initialize(host));
  CUDA_CHECK(spin_device.initialize(spin_host));
  CUDA_CHECK(launch_spin_hamiltonian(physical, spin_device, host, spin_host));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> actual(static_cast<std::size_t>(spin_host.matrix_offsets.back()));
  std::vector<std::uint32_t> errors(8u);
  CUDA_CHECK(spin_device.output.copy_to(actual.data(), actual.size()));
  CUDA_CHECK(physical.system_errors.copy_to(errors.data(), errors.size()));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(errors[failed] ==
        static_cast<std::uint32_t>(Gfn2HamiltonianDeviceError::kNonfinitePotential));
  for (std::int64_t element = spin_host.matrix_offsets[failed];
       element < spin_host.matrix_offsets[failed + 1u]; ++element) {
    CHECK(actual[static_cast<std::size_t>(element)] == kSentinel);
  }
  CHECK(actual[static_cast<std::size_t>(spin_host.matrix_offsets[1])] != kSentinel);

  auto layout = spin_device.layout(host, spin_host);
  auto input = spin_device.input(physical, host, spin_host);
  const auto activity = physical.activity(host);
  auto output = spin_device.result(spin_host);
  auto workspace = spin_device.workspace(spin_host);
  const auto batch = physical.batch(host);
  auto wrong_layout = layout;
  wrong_layout.memory_space = Gfn2PlanMemorySpace::kHost;
  CHECK(assemble_gfn2_spin_hamiltonian_cuda(batch, wrong_layout, input, activity, output, workspace,
                                            physical.system_errors.get(),
                                            physical.device_error.get()) == cudaErrorInvalidValue);
  auto alias_output = output;
  alias_output.matrix = const_cast<double*>(input.shell_scalar_potentials);
  CHECK(assemble_gfn2_spin_hamiltonian_cuda(batch, layout, input, activity, alias_output, workspace,
                                            physical.system_errors.get(),
                                            physical.device_error.get()) == cudaErrorInvalidValue);
  auto partial_workspace = workspace;
  partial_workspace.matrix_scratch = output.matrix + 1;
  CHECK(assemble_gfn2_spin_hamiltonian_cuda(batch, layout, input, activity, output,
                                            partial_workspace, physical.system_errors.get(),
                                            physical.device_error.get()) == cudaErrorInvalidValue);
  auto alias_layout = layout;
  alias_layout.spin_matrix_offsets = reinterpret_cast<const std::int64_t*>(output.matrix);
  CHECK(assemble_gfn2_spin_hamiltonian_cuda(batch, alias_layout, input, activity, output, workspace,
                                            physical.system_errors.get(),
                                            physical.device_error.get()) == cudaErrorInvalidValue);

  HostCase inactive_host = make_case(1u);
  inactive_host.active[0] = 0u;
  HostSpinHamiltonianCase inactive_spin = make_spin_hamiltonian_case(inactive_host);
  DeviceFixture inactive_physical;
  SpinHamiltonianFixture inactive_device;
  CUDA_CHECK(inactive_physical.initialize(inactive_host));
  CUDA_CHECK(inactive_device.initialize(inactive_spin));
  const std::int32_t hostile_channel = std::numeric_limits<std::int32_t>::min();
  const std::int64_t hostile_offset = std::numeric_limits<std::int64_t>::min();
  CUDA_CHECK(inactive_device.channels.copy_from(&hostile_channel, 1u));
  CUDA_CHECK(inactive_device.matrix_offsets.copy_from(&hostile_offset, 1u));
  CUDA_CHECK(
      launch_spin_hamiltonian(inactive_physical, inactive_device, inactive_host, inactive_spin));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> inactive_output(
      static_cast<std::size_t>(inactive_spin.matrix_offsets.back()));
  std::uint32_t inactive_error = 99u;
  CUDA_CHECK(inactive_device.output.copy_to(inactive_output.data(), inactive_output.size()));
  CUDA_CHECK(inactive_physical.system_errors.copy_to(&inactive_error, 1u));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(inactive_error == 0u);
  CHECK(std::all_of(inactive_output.begin(), inactive_output.end(),
                    [](double value) { return value == kSentinel; }));
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_production_cpu_parity(); status != 0) {
    std::fprintf(stderr, "CUDA Hamiltonian production CPU parity failed at line %d\n", status);
    return status;
  }
  for (const std::size_t batch_size : std::array<std::size_t, 4>{{1u, 8u, 32u, 128u}}) {
    if (const int status = run_parity(batch_size, false); status != 0) {
      std::fprintf(stderr, "CUDA Hamiltonian batch-%zu parity failed at line %d\n", batch_size,
                   status);
      return status;
    }
  }
  if (const int status = run_parity(8u, true); status != 0) {
    std::fprintf(stderr, "CUDA Hamiltonian Graph test failed at line %d\n", status);
    return status;
  }
  for (const std::size_t batch_size : std::array<std::size_t, 4>{{1u, 8u, 32u, 128u}}) {
    if (const int status = run_spin_hamiltonian_parity(batch_size, false); status != 0) {
      std::fprintf(stderr, "CUDA spin Hamiltonian batch-%zu parity failed at line %d\n", batch_size,
                   status);
      return status;
    }
  }
  if (const int status = run_spin_hamiltonian_parity(8u, true); status != 0) {
    std::fprintf(stderr, "CUDA spin Hamiltonian Graph test failed at line %d\n", status);
    return status;
  }
  const std::array<int (*)(), 7> tests{
      {test_inactive_skip_and_peer_isolation, test_all_inactive_hostile_topology,
       test_hostile_offsets, test_arithmetic_overflow_is_transactional, test_invalid_active_mask,
       test_sticky_alias_token_and_alignment, test_spin_hamiltonian_failure_and_alias_contracts}};
  for (const auto test : tests) {
    const int status = test();
    if (status != 0) {
      std::fprintf(stderr, "CUDA Hamiltonian regression failed at line %d\n", status);
      return status;
    }
  }
  std::puts("CUDA Hamiltonian tests passed");
  return 0;
}
