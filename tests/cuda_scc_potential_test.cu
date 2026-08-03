#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "backends/cuda/gfn2_scc_potential.cuh"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using gpuxtb::detail::cuda::compose_gfn2_scc_potentials_cuda;
using gpuxtb::detail::cuda::gather_gfn2_scc_mixed_multipoles_cuda;
using gpuxtb::detail::cuda::Gfn2SccPotentialComponent;
using gpuxtb::detail::cuda::Gfn2SccPotentialDeviceActivity;
using gpuxtb::detail::cuda::Gfn2SccPotentialDeviceBatch;
using gpuxtb::detail::cuda::Gfn2SccPotentialDeviceComponents;
using gpuxtb::detail::cuda::Gfn2SccPotentialDeviceError;
using gpuxtb::detail::cuda::Gfn2SccPotentialDeviceMixedFields;
using gpuxtb::detail::cuda::Gfn2SccPotentialDeviceResults;
using gpuxtb::detail::cuda::Gfn2SccPotentialDeviceTopologyMultipoles;
using gpuxtb::detail::cuda::Gfn2SccPotentialDeviceWorkspace;
using gpuxtb::detail::cuda::kGfn2SccPotentialAllComponents;
using gpuxtb::detail::cuda::reset_gfn2_scc_potential_device_errors_cuda;

constexpr std::uint64_t kPlanToken = 0x77c20a5e61d934bfULL;
constexpr double kSentinel = -770.625;

constexpr std::uint32_t bit(Gfn2SccPotentialComponent component) {
  return static_cast<std::uint32_t>(component);
}

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
    return count == 0u ? cudaSuccess
                       : cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
  }

  cudaError_t copy_from(const T* source, std::size_t count, cudaStream_t stream = nullptr) {
    if (count > count_ || (count != 0u && source == nullptr)) {
      return cudaErrorInvalidValue;
    }
    return count == 0u
               ? cudaSuccess
               : cudaMemcpyAsync(data_, source, count * sizeof(T), cudaMemcpyHostToDevice, stream);
  }

  cudaError_t copy_to(T* destination, std::size_t count, cudaStream_t stream = nullptr) const {
    if (count > count_ || (count != 0u && destination == nullptr)) {
      return cudaErrorInvalidValue;
    }
    return count == 0u ? cudaSuccess
                       : cudaMemcpyAsync(destination, data_, count * sizeof(T),
                                         cudaMemcpyDeviceToHost, stream);
  }

  cudaError_t fill(const T& value, std::size_t count, cudaStream_t stream = nullptr) {
    if (count > count_) {
      return cudaErrorInvalidValue;
    }
    std::vector<T> host(count, value);
    return copy_from(host.data(), count, stream);
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

bool near(double actual, double expected) {
  return std::abs(actual - expected) <=
         2.0e-14 + 2.0e-14 * std::max(std::abs(actual), std::abs(expected));
}

struct HostCase {
  std::int64_t batch_size = 0;
  std::vector<std::int64_t> atom_offsets{0};
  std::vector<std::int64_t> shell_offsets{0};
  std::vector<std::int64_t> qsh_offsets{0};
  std::vector<std::int64_t> qat_offsets{0};
  std::vector<std::int64_t> dipole_offsets{0};
  std::vector<std::int64_t> quadrupole_offsets{0};
  std::vector<std::int64_t> shell_to_atom;
  std::vector<std::uint8_t> active;
  std::vector<double> mixed_qsh;
  std::vector<double> mixed_dipole;
  std::vector<double> mixed_quadrupole;
  std::vector<double> es2;
  std::vector<double> es3;
  std::vector<double> pc;
  std::vector<double> aes2_atomic;
  std::vector<double> aes2_dipole;
  std::vector<double> aes2_quadrupole;
  std::vector<double> periodic;
  std::vector<double> d4;
};

HostCase make_case(std::size_t batch_size) {
  HostCase host;
  host.batch_size = static_cast<std::int64_t>(batch_size);
  std::int64_t atoms = 0;
  std::int64_t shells = 0;
  for (std::size_t system = 0; system < batch_size; ++system) {
    const bool empty = batch_size > 1u && system % 11u == 7u;
    const std::int64_t system_atoms = empty ? 0 : 1 + static_cast<std::int64_t>(system % 2u);
    for (std::int64_t local_atom = 0; local_atom < system_atoms; ++local_atom) {
      const std::int64_t atom_shells = 1 + static_cast<std::int64_t>((system + local_atom) % 2u);
      for (std::int64_t shell = 0; shell < atom_shells; ++shell) {
        host.shell_to_atom.push_back(atoms);
        ++shells;
      }
      ++atoms;
    }
    host.atom_offsets.push_back(atoms);
    host.shell_offsets.push_back(shells);
    host.qsh_offsets.push_back(shells);
    host.qat_offsets.push_back(atoms);
    host.dipole_offsets.push_back(3 * atoms);
    host.quadrupole_offsets.push_back(6 * atoms);
  }
  host.active.assign(batch_size, 1u);
  host.mixed_qsh.resize(static_cast<std::size_t>(shells));
  host.mixed_dipole.resize(static_cast<std::size_t>(3 * atoms));
  host.mixed_quadrupole.resize(static_cast<std::size_t>(6 * atoms));
  host.es2.resize(static_cast<std::size_t>(shells));
  host.es3.resize(static_cast<std::size_t>(shells));
  host.pc.resize(static_cast<std::size_t>(shells));
  host.aes2_atomic.resize(static_cast<std::size_t>(atoms));
  host.aes2_dipole.resize(static_cast<std::size_t>(3 * atoms));
  host.aes2_quadrupole.resize(static_cast<std::size_t>(6 * atoms));
  host.periodic.resize(static_cast<std::size_t>(atoms));
  host.d4.resize(static_cast<std::size_t>(atoms));
  for (std::int64_t shell = 0; shell < shells; ++shell) {
    host.mixed_qsh[static_cast<std::size_t>(shell)] =
        0.07 * static_cast<double>(1 + (shell * 7) % 17) - 0.41;
    host.es2[static_cast<std::size_t>(shell)] =
        0.013 * static_cast<double>(1 + (shell * 11) % 19) - 0.12;
    host.es3[static_cast<std::size_t>(shell)] =
        0.009 * static_cast<double>(1 + (shell * 13) % 23) - 0.08;
    host.pc[static_cast<std::size_t>(shell)] =
        0.005 * static_cast<double>(1 + (shell * 17) % 29) - 0.06;
  }
  for (std::int64_t atom = 0; atom < atoms; ++atom) {
    host.aes2_atomic[static_cast<std::size_t>(atom)] =
        0.021 * static_cast<double>(1 + (atom * 5) % 17) - 0.15;
    host.periodic[static_cast<std::size_t>(atom)] =
        0.016 * static_cast<double>(1 + (atom * 7) % 19) - 0.11;
    host.d4[static_cast<std::size_t>(atom)] =
        0.011 * static_cast<double>(1 + (atom * 11) % 23) - 0.09;
    for (int component = 0; component < 3; ++component) {
      const std::size_t index = static_cast<std::size_t>(atom * 3 + component);
      host.mixed_dipole[index] =
          0.031 * static_cast<double>(1 + (atom * 13 + component * 3) % 17) - 0.19;
      host.aes2_dipole[index] =
          0.017 * static_cast<double>(1 + (atom * 5 + component * 7) % 19) - 0.13;
    }
    for (int component = 0; component < 6; ++component) {
      const std::size_t index = static_cast<std::size_t>(atom * 6 + component);
      host.mixed_quadrupole[index] =
          0.023 * static_cast<double>(1 + (atom * 7 + component * 5) % 23) - 0.24;
      host.aes2_quadrupole[index] =
          0.012 * static_cast<double>(1 + (atom * 11 + component * 3) % 29) - 0.17;
    }
  }
  return host;
}

struct GatherResult {
  std::vector<double> shell;
  std::vector<double> atom;
  std::vector<double> dipole;
  std::vector<double> quadrupole;
};

GatherResult gather_reference(const HostCase& host) {
  GatherResult result;
  result.shell.assign(host.mixed_qsh.size(), kSentinel);
  result.atom.assign(host.aes2_atomic.size(), kSentinel);
  result.dipole.assign(host.mixed_dipole.size(), kSentinel);
  result.quadrupole.assign(host.mixed_quadrupole.size(), kSentinel);
  for (std::size_t system = 0; system < static_cast<std::size_t>(host.batch_size); ++system) {
    if (host.active[system] == 0u) {
      continue;
    }
    const std::int64_t atom_begin = host.atom_offsets[system];
    const std::int64_t atom_end = host.atom_offsets[system + 1u];
    const std::int64_t shell_begin = host.shell_offsets[system];
    const std::int64_t shell_end = host.shell_offsets[system + 1u];
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      result.shell[static_cast<std::size_t>(shell)] =
          host.mixed_qsh[static_cast<std::size_t>(host.qsh_offsets[system] + shell - shell_begin)];
    }
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
      double charge = 0.0;
      for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
        if (host.shell_to_atom[static_cast<std::size_t>(shell)] == atom) {
          charge += host.mixed_qsh[static_cast<std::size_t>(host.qsh_offsets[system] + shell -
                                                            shell_begin)];
        }
      }
      result.atom[static_cast<std::size_t>(atom)] = charge;
      for (int component = 0; component < 3; ++component) {
        result.dipole[static_cast<std::size_t>(atom * 3 + component)] =
            host.mixed_dipole[static_cast<std::size_t>(host.dipole_offsets[system] +
                                                       (atom - atom_begin) * 3 + component)];
      }
      for (int component = 0; component < 6; ++component) {
        result.quadrupole[static_cast<std::size_t>(atom * 6 + component)] =
            host.mixed_quadrupole[static_cast<std::size_t>(host.quadrupole_offsets[system] +
                                                           (atom - atom_begin) * 6 + component)];
      }
    }
  }
  return result;
}

struct ComposeResult {
  std::vector<double> shell;
  std::vector<double> atom;
  std::vector<double> dipole;
  std::vector<double> quadrupole;
};

ComposeResult compose_reference(const HostCase& host, std::uint32_t mask) {
  ComposeResult result;
  result.shell.assign(host.mixed_qsh.size(), kSentinel);
  result.atom.assign(host.aes2_atomic.size(), kSentinel);
  result.dipole.assign(host.mixed_dipole.size(), kSentinel);
  result.quadrupole.assign(host.mixed_quadrupole.size(), kSentinel);
  const bool es2 = (mask & bit(Gfn2SccPotentialComponent::kES2)) != 0u;
  const bool es3 = (mask & bit(Gfn2SccPotentialComponent::kES3)) != 0u;
  const bool pc = (mask & bit(Gfn2SccPotentialComponent::kExplicitPointCharge)) != 0u;
  const bool aes2 = (mask & bit(Gfn2SccPotentialComponent::kAES2)) != 0u;
  const bool periodic = (mask & bit(Gfn2SccPotentialComponent::kPeriodicEmbedding)) != 0u;
  const bool d4 = (mask & bit(Gfn2SccPotentialComponent::kD4TwoBody)) != 0u;
  for (std::size_t system = 0; system < static_cast<std::size_t>(host.batch_size); ++system) {
    if (host.active[system] == 0u) {
      continue;
    }
    for (std::int64_t shell = host.shell_offsets[system]; shell < host.shell_offsets[system + 1u];
         ++shell) {
      const double first = (es2 ? host.es2[static_cast<std::size_t>(shell)] : 0.0) +
                           (es3 ? host.es3[static_cast<std::size_t>(shell)] : 0.0);
      result.shell[static_cast<std::size_t>(host.qsh_offsets[system] + shell -
                                            host.shell_offsets[system])] =
          first + (pc ? host.pc[static_cast<std::size_t>(shell)] : 0.0);
    }
    for (std::int64_t atom = host.atom_offsets[system]; atom < host.atom_offsets[system + 1u];
         ++atom) {
      const std::int64_t local_atom = atom - host.atom_offsets[system];
      const double first = (aes2 ? host.aes2_atomic[static_cast<std::size_t>(atom)] : 0.0) +
                           (periodic ? host.periodic[static_cast<std::size_t>(atom)] : 0.0);
      result.atom[static_cast<std::size_t>(host.qat_offsets[system] + local_atom)] =
          first + (d4 ? host.d4[static_cast<std::size_t>(atom)] : 0.0);
      for (int component = 0; component < 3; ++component) {
        result.dipole[static_cast<std::size_t>(host.dipole_offsets[system] + local_atom * 3 +
                                               component)] =
            aes2 ? host.aes2_dipole[static_cast<std::size_t>(atom * 3 + component)] : 0.0;
      }
      for (int component = 0; component < 6; ++component) {
        result.quadrupole[static_cast<std::size_t>(host.quadrupole_offsets[system] +
                                                   local_atom * 6 + component)] =
            aes2 ? host.aes2_quadrupole[static_cast<std::size_t>(atom * 6 + component)] : 0.0;
      }
    }
  }
  return result;
}

struct DeviceFixture {
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> shell_offsets;
  DeviceBuffer<std::int64_t> qsh_offsets;
  DeviceBuffer<std::int64_t> qat_offsets;
  DeviceBuffer<std::int64_t> dipole_offsets;
  DeviceBuffer<std::int64_t> quadrupole_offsets;
  DeviceBuffer<std::int64_t> shell_to_atom;
  DeviceBuffer<std::uint8_t> active;
  DeviceBuffer<double> mixed_qsh;
  DeviceBuffer<double> mixed_dipole;
  DeviceBuffer<double> mixed_quadrupole;
  DeviceBuffer<double> topology_shell;
  DeviceBuffer<double> topology_atom;
  DeviceBuffer<double> topology_dipole;
  DeviceBuffer<double> topology_quadrupole;
  DeviceBuffer<double> es2;
  DeviceBuffer<double> es3;
  DeviceBuffer<double> pc;
  DeviceBuffer<double> aes2_atomic;
  DeviceBuffer<double> aes2_dipole;
  DeviceBuffer<double> aes2_quadrupole;
  DeviceBuffer<double> periodic;
  DeviceBuffer<double> d4;
  DeviceBuffer<double> result_shell;
  DeviceBuffer<double> result_atom;
  DeviceBuffer<double> result_dipole;
  DeviceBuffer<double> result_quadrupole;
  DeviceBuffer<double> scratch_shell;
  DeviceBuffer<double> scratch_atom;
  DeviceBuffer<double> scratch_dipole;
  DeviceBuffer<double> scratch_quadrupole;
  DeviceBuffer<std::uint32_t> sequence_active;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;

  cudaError_t initialize(const HostCase& host, cudaStream_t stream = nullptr) {
    cudaError_t status = upload(atom_offsets, host.atom_offsets, stream);
#define UPLOAD(field, source)               \
  if (status == cudaSuccess) {              \
    status = upload(field, source, stream); \
  }
    UPLOAD(shell_offsets, host.shell_offsets)
    UPLOAD(qsh_offsets, host.qsh_offsets)
    UPLOAD(qat_offsets, host.qat_offsets)
    UPLOAD(dipole_offsets, host.dipole_offsets)
    UPLOAD(quadrupole_offsets, host.quadrupole_offsets)
    UPLOAD(shell_to_atom, host.shell_to_atom)
    UPLOAD(active, host.active)
    UPLOAD(mixed_qsh, host.mixed_qsh)
    UPLOAD(mixed_dipole, host.mixed_dipole)
    UPLOAD(mixed_quadrupole, host.mixed_quadrupole)
    UPLOAD(es2, host.es2)
    UPLOAD(es3, host.es3)
    UPLOAD(pc, host.pc)
    UPLOAD(aes2_atomic, host.aes2_atomic)
    UPLOAD(aes2_dipole, host.aes2_dipole)
    UPLOAD(aes2_quadrupole, host.aes2_quadrupole)
    UPLOAD(periodic, host.periodic)
    UPLOAD(d4, host.d4)
#undef UPLOAD
    const std::size_t shells = host.mixed_qsh.size();
    const std::size_t atoms = host.aes2_atomic.size();
    const std::size_t dipoles = host.mixed_dipole.size();
    const std::size_t quadrupoles = host.mixed_quadrupole.size();
#define ALLOCATE(field, count)      \
  if (status == cudaSuccess) {      \
    status = field.allocate(count); \
  }
    ALLOCATE(topology_shell, shells)
    ALLOCATE(topology_atom, atoms)
    ALLOCATE(topology_dipole, dipoles)
    ALLOCATE(topology_quadrupole, quadrupoles)
    ALLOCATE(result_shell, shells)
    ALLOCATE(result_atom, atoms)
    ALLOCATE(result_dipole, dipoles)
    ALLOCATE(result_quadrupole, quadrupoles)
    ALLOCATE(scratch_shell, shells)
    ALLOCATE(scratch_atom, atoms)
    ALLOCATE(scratch_dipole, dipoles)
    ALLOCATE(scratch_quadrupole, quadrupoles)
    ALLOCATE(sequence_active, 1u)
    ALLOCATE(system_errors, static_cast<std::size_t>(host.batch_size))
    ALLOCATE(device_error, 1u)
#undef ALLOCATE
    return status == cudaSuccess ? reset_outputs(host, stream) : status;
  }

  cudaError_t reset_outputs(const HostCase& host, cudaStream_t stream = nullptr) {
    cudaError_t status = topology_shell.fill(kSentinel, host.mixed_qsh.size(), stream);
#define FILL(field, count)                                                   \
  if (status == cudaSuccess) {                                               \
    status = field.fill(kSentinel, static_cast<std::size_t>(count), stream); \
  }
    FILL(topology_atom, host.aes2_atomic.size())
    FILL(topology_dipole, host.mixed_dipole.size())
    FILL(topology_quadrupole, host.mixed_quadrupole.size())
    FILL(result_shell, host.mixed_qsh.size())
    FILL(result_atom, host.aes2_atomic.size())
    FILL(result_dipole, host.mixed_dipole.size())
    FILL(result_quadrupole, host.mixed_quadrupole.size())
#undef FILL
    return status;
  }

  Gfn2SccPotentialDeviceBatch batch(const HostCase& host) const {
    return {host.batch_size,
            host.atom_offsets.back(),
            host.shell_offsets.back(),
            kPlanToken,
            static_cast<std::int64_t>(host.atom_offsets.size()),
            static_cast<std::int64_t>(host.shell_offsets.size()),
            static_cast<std::int64_t>(host.qsh_offsets.size()),
            static_cast<std::int64_t>(host.qat_offsets.size()),
            static_cast<std::int64_t>(host.dipole_offsets.size()),
            static_cast<std::int64_t>(host.quadrupole_offsets.size()),
            static_cast<std::int64_t>(host.shell_to_atom.size()),
            atom_offsets.get(),
            shell_offsets.get(),
            qsh_offsets.get(),
            qat_offsets.get(),
            dipole_offsets.get(),
            quadrupole_offsets.get(),
            shell_to_atom.get()};
  }

  Gfn2SccPotentialDeviceActivity activity(const HostCase& host) const {
    return {active.get(), host.batch_size, kPlanToken};
  }

  Gfn2SccPotentialDeviceMixedFields mixed(const HostCase& host) const {
    return {mixed_qsh.get(),
            static_cast<std::int64_t>(host.mixed_qsh.size()),
            mixed_dipole.get(),
            static_cast<std::int64_t>(host.mixed_dipole.size()),
            mixed_quadrupole.get(),
            static_cast<std::int64_t>(host.mixed_quadrupole.size()),
            kPlanToken};
  }

  Gfn2SccPotentialDeviceTopologyMultipoles topology(const HostCase& host) {
    return {topology_shell.get(),
            static_cast<std::int64_t>(host.mixed_qsh.size()),
            topology_atom.get(),
            static_cast<std::int64_t>(host.aes2_atomic.size()),
            topology_dipole.get(),
            static_cast<std::int64_t>(host.mixed_dipole.size()),
            topology_quadrupole.get(),
            static_cast<std::int64_t>(host.mixed_quadrupole.size()),
            kPlanToken};
  }

  Gfn2SccPotentialDeviceComponents components(const HostCase& host, std::uint32_t mask) const {
    const auto enabled = [mask](Gfn2SccPotentialComponent component) {
      return (mask & bit(component)) != 0u;
    };
    return {
        mask,
        enabled(Gfn2SccPotentialComponent::kES2) ? es2.get() : nullptr,
        enabled(Gfn2SccPotentialComponent::kES2) ? static_cast<std::int64_t>(host.es2.size()) : 0,
        enabled(Gfn2SccPotentialComponent::kES3) ? es3.get() : nullptr,
        enabled(Gfn2SccPotentialComponent::kES3) ? static_cast<std::int64_t>(host.es3.size()) : 0,
        enabled(Gfn2SccPotentialComponent::kExplicitPointCharge) ? pc.get() : nullptr,
        enabled(Gfn2SccPotentialComponent::kExplicitPointCharge)
            ? static_cast<std::int64_t>(host.pc.size())
            : 0,
        enabled(Gfn2SccPotentialComponent::kAES2) ? aes2_atomic.get() : nullptr,
        enabled(Gfn2SccPotentialComponent::kAES2)
            ? static_cast<std::int64_t>(host.aes2_atomic.size())
            : 0,
        enabled(Gfn2SccPotentialComponent::kAES2) ? aes2_dipole.get() : nullptr,
        enabled(Gfn2SccPotentialComponent::kAES2)
            ? static_cast<std::int64_t>(host.aes2_dipole.size())
            : 0,
        enabled(Gfn2SccPotentialComponent::kAES2) ? aes2_quadrupole.get() : nullptr,
        enabled(Gfn2SccPotentialComponent::kAES2)
            ? static_cast<std::int64_t>(host.aes2_quadrupole.size())
            : 0,
        enabled(Gfn2SccPotentialComponent::kD4TwoBody) ? d4.get() : nullptr,
        enabled(Gfn2SccPotentialComponent::kD4TwoBody) ? static_cast<std::int64_t>(host.d4.size())
                                                       : 0,
        enabled(Gfn2SccPotentialComponent::kPeriodicEmbedding) ? periodic.get() : nullptr,
        enabled(Gfn2SccPotentialComponent::kPeriodicEmbedding)
            ? static_cast<std::int64_t>(host.periodic.size())
            : 0,
        kPlanToken};
  }

  Gfn2SccPotentialDeviceResults results(const HostCase& host) {
    return {result_shell.get(),
            static_cast<std::int64_t>(host.mixed_qsh.size()),
            result_atom.get(),
            static_cast<std::int64_t>(host.aes2_atomic.size()),
            result_dipole.get(),
            static_cast<std::int64_t>(host.mixed_dipole.size()),
            result_quadrupole.get(),
            static_cast<std::int64_t>(host.mixed_quadrupole.size()),
            kPlanToken};
  }

  Gfn2SccPotentialDeviceWorkspace workspace(const HostCase& host) {
    return {scratch_shell.get(),
            static_cast<std::int64_t>(host.mixed_qsh.size()),
            scratch_atom.get(),
            static_cast<std::int64_t>(host.aes2_atomic.size()),
            scratch_dipole.get(),
            static_cast<std::int64_t>(host.mixed_dipole.size()),
            scratch_quadrupole.get(),
            static_cast<std::int64_t>(host.mixed_quadrupole.size()),
            sequence_active.get(),
            1,
            kPlanToken};
  }
};

cudaError_t launch_sequence(DeviceFixture& device, const HostCase& host, std::uint32_t mask,
                            cudaStream_t stream = nullptr) {
  cudaError_t status = reset_gfn2_scc_potential_device_errors_cuda(
      host.batch_size, device.system_errors.get(), device.device_error.get(), stream);
  if (status == cudaSuccess) {
    status = gather_gfn2_scc_mixed_multipoles_cuda(
        device.batch(host), device.mixed(host), device.activity(host), device.topology(host),
        device.workspace(host), device.system_errors.get(), device.device_error.get(), stream);
  }
  if (status == cudaSuccess) {
    status = compose_gfn2_scc_potentials_cuda(device.batch(host), device.components(host, mask),
                                              device.activity(host), device.results(host),
                                              device.workspace(host), device.system_errors.get(),
                                              device.device_error.get(), stream);
  }
  return status;
}

template <typename Result>
bool equal_result(const Result& actual, const Result& expected) {
  const auto equal = [](const std::vector<double>& first, const std::vector<double>& second) {
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
  return equal(actual.shell, expected.shell) && equal(actual.atom, expected.atom) &&
         equal(actual.dipole, expected.dipole) && equal(actual.quadrupole, expected.quadrupole);
}

cudaError_t download_gather(const HostCase& host, const DeviceFixture& device, GatherResult& result,
                            cudaStream_t stream = nullptr) {
  result.shell.resize(host.mixed_qsh.size());
  result.atom.resize(host.aes2_atomic.size());
  result.dipole.resize(host.mixed_dipole.size());
  result.quadrupole.resize(host.mixed_quadrupole.size());
  cudaError_t status =
      device.topology_shell.copy_to(result.shell.data(), result.shell.size(), stream);
  if (status == cudaSuccess)
    status = device.topology_atom.copy_to(result.atom.data(), result.atom.size(), stream);
  if (status == cudaSuccess)
    status = device.topology_dipole.copy_to(result.dipole.data(), result.dipole.size(), stream);
  if (status == cudaSuccess)
    status = device.topology_quadrupole.copy_to(result.quadrupole.data(), result.quadrupole.size(),
                                                stream);
  return status;
}

cudaError_t download_compose(const HostCase& host, const DeviceFixture& device,
                             ComposeResult& result, cudaStream_t stream = nullptr) {
  result.shell.resize(host.mixed_qsh.size());
  result.atom.resize(host.aes2_atomic.size());
  result.dipole.resize(host.mixed_dipole.size());
  result.quadrupole.resize(host.mixed_quadrupole.size());
  cudaError_t status =
      device.result_shell.copy_to(result.shell.data(), result.shell.size(), stream);
  if (status == cudaSuccess)
    status = device.result_atom.copy_to(result.atom.data(), result.atom.size(), stream);
  if (status == cudaSuccess)
    status = device.result_dipole.copy_to(result.dipole.data(), result.dipole.size(), stream);
  if (status == cudaSuccess)
    status = device.result_quadrupole.copy_to(result.quadrupole.data(), result.quadrupole.size(),
                                              stream);
  return status;
}

int run_parity(std::size_t batch_size, bool graph) {
  HostCase host = make_case(batch_size);
  GatherResult expected_gather = gather_reference(host);
  ComposeResult expected_compose = compose_reference(host, kGfn2SccPotentialAllComponents);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (graph) {
    cudaGraph_t captured = nullptr;
    cudaGraphExec_t executable = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    CUDA_CHECK(launch_sequence(device, host, kGfn2SccPotentialAllComponents, stream));
    CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
    CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    GatherResult first_gather;
    ComposeResult first_compose;
    std::vector<std::uint32_t> first_errors(static_cast<std::size_t>(host.batch_size));
    std::uint32_t first_diagnostic = 99u;
    CUDA_CHECK(download_gather(host, device, first_gather, stream));
    CUDA_CHECK(download_compose(host, device, first_compose, stream));
    CUDA_CHECK(device.system_errors.copy_to(first_errors.data(), first_errors.size(), stream));
    CUDA_CHECK(device.device_error.copy_to(&first_diagnostic, 1u, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(equal_result(first_gather, expected_gather));
    CHECK(equal_result(first_compose, expected_compose));
    CHECK(std::all_of(first_errors.begin(), first_errors.end(),
                      [](std::uint32_t error) { return error == 0u; }));
    CHECK(first_diagnostic == 0u);

    /* Replay must reread caller-owned buffers rather than retain first-launch
     * numerical state in captured scratch/provider nodes. */
    CHECK(!host.mixed_qsh.empty() && !host.es2.empty());
    host.mixed_qsh[0] += 0.1875;
    host.es2[0] -= 0.09375;
    expected_gather = gather_reference(host);
    expected_compose = compose_reference(host, kGfn2SccPotentialAllComponents);
    CUDA_CHECK(device.mixed_qsh.copy_from(host.mixed_qsh.data(), host.mixed_qsh.size(), stream));
    CUDA_CHECK(device.es2.copy_from(host.es2.data(), host.es2.size(), stream));
    CUDA_CHECK(device.reset_outputs(host, stream));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(captured));
  } else {
    CUDA_CHECK(launch_sequence(device, host, kGfn2SccPotentialAllComponents, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  GatherResult gather;
  ComposeResult compose;
  std::vector<std::uint32_t> errors(static_cast<std::size_t>(host.batch_size));
  std::uint32_t diagnostic = 99u;
  CUDA_CHECK(download_gather(host, device, gather, stream));
  CUDA_CHECK(download_compose(host, device, compose, stream));
  CUDA_CHECK(device.system_errors.copy_to(errors.data(), errors.size(), stream));
  CUDA_CHECK(device.device_error.copy_to(&diagnostic, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(equal_result(gather, expected_gather));
  CHECK(equal_result(compose, expected_compose));
  CHECK(std::all_of(errors.begin(), errors.end(), [](std::uint32_t error) { return error == 0u; }));
  CHECK(diagnostic == 0u);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

bool system_is_sentinel(const HostCase& host, const ComposeResult& result, std::size_t system) {
  for (std::int64_t index = host.qsh_offsets[system]; index < host.qsh_offsets[system + 1u];
       ++index) {
    if (result.shell[static_cast<std::size_t>(index)] != kSentinel) return false;
  }
  for (std::int64_t index = host.qat_offsets[system]; index < host.qat_offsets[system + 1u];
       ++index) {
    if (result.atom[static_cast<std::size_t>(index)] != kSentinel) return false;
  }
  for (std::int64_t index = host.dipole_offsets[system]; index < host.dipole_offsets[system + 1u];
       ++index) {
    if (result.dipole[static_cast<std::size_t>(index)] != kSentinel) return false;
  }
  for (std::int64_t index = host.quadrupole_offsets[system];
       index < host.quadrupole_offsets[system + 1u]; ++index) {
    if (result.quadrupole[static_cast<std::size_t>(index)] != kSentinel) return false;
  }
  return true;
}

int test_disabled_null_zero() {
  HostCase host = make_case(8u);
  host.active[3] = 0u;
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host));
  CUDA_CHECK(reset_gfn2_scc_potential_device_errors_cuda(
      host.batch_size, device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(compose_gfn2_scc_potentials_cuda(
      device.batch(host), device.components(host, 0u), device.activity(host), device.results(host),
      device.workspace(host), device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  ComposeResult actual;
  CUDA_CHECK(download_compose(host, device, actual));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(equal_result(actual, compose_reference(host, 0u)));
  CHECK(system_is_sentinel(host, actual, 3u));
  return 0;
}

int test_one_hot_component_wiring() {
  constexpr std::array<Gfn2SccPotentialComponent, 6> components{{
      Gfn2SccPotentialComponent::kES2,
      Gfn2SccPotentialComponent::kES3,
      Gfn2SccPotentialComponent::kAES2,
      Gfn2SccPotentialComponent::kD4TwoBody,
      Gfn2SccPotentialComponent::kExplicitPointCharge,
      Gfn2SccPotentialComponent::kPeriodicEmbedding,
  }};
  for (const Gfn2SccPotentialComponent component : components) {
    HostCase host = make_case(8u);
    const std::uint32_t mask = bit(component);
    DeviceFixture device;
    CUDA_CHECK(device.initialize(host));
    CUDA_CHECK(reset_gfn2_scc_potential_device_errors_cuda(
        host.batch_size, device.system_errors.get(), device.device_error.get()));
    CUDA_CHECK(compose_gfn2_scc_potentials_cuda(device.batch(host), device.components(host, mask),
                                                device.activity(host), device.results(host),
                                                device.workspace(host), device.system_errors.get(),
                                                device.device_error.get()));
    CUDA_CHECK(cudaDeviceSynchronize());
    ComposeResult actual;
    CUDA_CHECK(download_compose(host, device, actual));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(equal_result(actual, compose_reference(host, mask)));
  }
  return 0;
}

int test_inactive_poison_and_bad_mapping_peer_isolation() {
  HostCase host = make_case(8u);
  constexpr std::size_t inactive = 2u;
  constexpr std::size_t bad_mapping = 4u;
  host.active[inactive] = 0u;
  if (host.qsh_offsets[inactive] != host.qsh_offsets[inactive + 1u]) {
    host.mixed_qsh[static_cast<std::size_t>(host.qsh_offsets[inactive])] =
        std::numeric_limits<double>::quiet_NaN();
    host.es2[static_cast<std::size_t>(host.shell_offsets[inactive])] =
        std::numeric_limits<double>::infinity();
  }
  const std::int64_t bad_shell = host.shell_offsets[bad_mapping];
  CHECK(bad_shell < host.shell_offsets[bad_mapping + 1u]);
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host));
  const std::int64_t hostile_atom = host.atom_offsets[bad_mapping + 1u];
  CUDA_CHECK(cudaMemcpy(device.shell_to_atom.get() + bad_shell, &hostile_atom, sizeof(hostile_atom),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(launch_sequence(device, host, kGfn2SccPotentialAllComponents));
  CUDA_CHECK(cudaDeviceSynchronize());
  GatherResult gather;
  ComposeResult compose;
  std::vector<std::uint32_t> errors(8u);
  CUDA_CHECK(download_gather(host, device, gather));
  CUDA_CHECK(download_compose(host, device, compose));
  CUDA_CHECK(device.system_errors.copy_to(errors.data(), errors.size()));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(errors[inactive] == 0u);
  CHECK(errors[bad_mapping] ==
        static_cast<std::uint32_t>(Gfn2SccPotentialDeviceError::kInvalidShellMetadata));
  CHECK(system_is_sentinel(host, compose, inactive));
  CHECK(system_is_sentinel(host, compose, bad_mapping));
  CHECK(!system_is_sentinel(host, compose, 0u));
  return 0;
}

int test_hostile_offsets_fail_closed() {
  for (int target = 0; target < 6; ++target) {
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
          destination = device.shell_offsets.get() + 1;
          break;
        case 2:
          destination = device.qsh_offsets.get() + 1;
          break;
        case 3:
          destination = device.qat_offsets.get() + 1;
          break;
        case 4:
          destination = device.dipole_offsets.get() + 1;
          break;
        default:
          destination = device.quadrupole_offsets.get() + 1;
          break;
      }
      CUDA_CHECK(cudaMemcpy(destination, &hostile, sizeof(hostile), cudaMemcpyHostToDevice));
      CUDA_CHECK(launch_sequence(device, host, kGfn2SccPotentialAllComponents));
      CUDA_CHECK(cudaDeviceSynchronize());
      ComposeResult actual;
      std::uint32_t diagnostic = 0u;
      CUDA_CHECK(download_compose(host, device, actual));
      CUDA_CHECK(device.device_error.copy_to(&diagnostic, 1u));
      CUDA_CHECK(cudaDeviceSynchronize());
      CHECK(system_is_sentinel(host, actual, 0u));
      CHECK(diagnostic == static_cast<std::uint32_t>(Gfn2SccPotentialDeviceError::kInvalidOffsets));
    }
  }
  return 0;
}

int run_compose_failure(HostCase host, std::uint32_t expected_error) {
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host));
  CUDA_CHECK(reset_gfn2_scc_potential_device_errors_cuda(
      host.batch_size, device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(compose_gfn2_scc_potentials_cuda(
      device.batch(host), device.components(host, kGfn2SccPotentialAllComponents),
      device.activity(host), device.results(host), device.workspace(host),
      device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  ComposeResult actual;
  std::vector<std::uint32_t> errors(static_cast<std::size_t>(host.batch_size));
  CUDA_CHECK(download_compose(host, device, actual));
  CUDA_CHECK(device.system_errors.copy_to(errors.data(), errors.size()));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(system_is_sentinel(host, actual, 0u));
  CHECK(errors[0] == expected_error);
  if (host.batch_size > 1) CHECK(!system_is_sentinel(host, actual, 1u));
  return 0;
}

int test_component_and_addition_overflow() {
  const double maximum = std::numeric_limits<double>::max();
  HostCase host = make_case(2u);
  host.es2[0] = 0.75 * maximum;
  host.es3[0] = 0.75 * maximum;
  CHECK(run_compose_failure(
            host, static_cast<std::uint32_t>(
                      Gfn2SccPotentialDeviceError::kNonfiniteShellPotentialArithmetic)) == 0);
  host = make_case(2u);
  host.es2[0] = 0.5 * maximum;
  host.es3[0] = 0.25 * maximum;
  host.pc[0] = 0.75 * maximum;
  CHECK(run_compose_failure(
            host, static_cast<std::uint32_t>(
                      Gfn2SccPotentialDeviceError::kNonfiniteShellPotentialArithmetic)) == 0);
  host = make_case(2u);
  host.aes2_atomic[0] = 0.75 * maximum;
  host.periodic[0] = 0.75 * maximum;
  CHECK(run_compose_failure(
            host, static_cast<std::uint32_t>(
                      Gfn2SccPotentialDeviceError::kNonfiniteAtomicPotentialArithmetic)) == 0);
  host = make_case(2u);
  host.aes2_atomic[0] = 0.5 * maximum;
  host.periodic[0] = 0.25 * maximum;
  host.d4[0] = 0.75 * maximum;
  CHECK(run_compose_failure(
            host, static_cast<std::uint32_t>(
                      Gfn2SccPotentialDeviceError::kNonfiniteAtomicPotentialArithmetic)) == 0);
  host = make_case(2u);
  host.aes2_dipole[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK(run_compose_failure(host, static_cast<std::uint32_t>(
                                      Gfn2SccPotentialDeviceError::kNonfiniteAES2Potential)) == 0);
  return 0;
}

int test_strict_cpu_association_order() {
  const double maximum = std::numeric_limits<double>::max();
  HostCase host = make_case(2u);
  /* CPU order is (ES2 + ES3) + PC. Reassociation to ES2 + (ES3 + PC)
   * would cancel to a finite value and incorrectly publish. */
  host.es2[0] = maximum;
  host.es3[0] = maximum;
  host.pc[0] = -maximum;
  CHECK(run_compose_failure(
            host, static_cast<std::uint32_t>(
                      Gfn2SccPotentialDeviceError::kNonfiniteShellPotentialArithmetic)) == 0);

  host = make_case(2u);
  /* CPU atomic order is (AES2 + periodic) + D4. */
  host.aes2_atomic[0] = maximum;
  host.periodic[0] = maximum;
  host.d4[0] = -maximum;
  CHECK(run_compose_failure(
            host, static_cast<std::uint32_t>(
                      Gfn2SccPotentialDeviceError::kNonfiniteAtomicPotentialArithmetic)) == 0);
  return 0;
}

int test_gather_reduction_overflow() {
  HostCase host;
  host.batch_size = 2;
  host.atom_offsets = {0, 1, 2};
  host.shell_offsets = {0, 2, 3};
  host.qsh_offsets = host.shell_offsets;
  host.qat_offsets = host.atom_offsets;
  host.dipole_offsets = {0, 3, 6};
  host.quadrupole_offsets = {0, 6, 12};
  host.shell_to_atom = {0, 0, 1};
  host.active = {1u, 1u};
  host.mixed_qsh = {0.75 * std::numeric_limits<double>::max(),
                    0.75 * std::numeric_limits<double>::max(), 0.2};
  host.mixed_dipole.assign(6u, 0.0);
  host.mixed_quadrupole.assign(12u, 0.0);
  host.es2.assign(3u, 0.0);
  host.es3.assign(3u, 0.0);
  host.pc.assign(3u, 0.0);
  host.aes2_atomic.assign(2u, 0.0);
  host.aes2_dipole.assign(6u, 0.0);
  host.aes2_quadrupole.assign(12u, 0.0);
  host.periodic.assign(2u, 0.0);
  host.d4.assign(2u, 0.0);
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host));
  CUDA_CHECK(reset_gfn2_scc_potential_device_errors_cuda(
      host.batch_size, device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(gather_gfn2_scc_mixed_multipoles_cuda(
      device.batch(host), device.mixed(host), device.activity(host), device.topology(host),
      device.workspace(host), device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::uint32_t> errors(2u);
  GatherResult actual;
  CUDA_CHECK(device.system_errors.copy_to(errors.data(), errors.size()));
  CUDA_CHECK(download_gather(host, device, actual));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(errors[0] ==
        static_cast<std::uint32_t>(Gfn2SccPotentialDeviceError::kNonfiniteAtomicChargeReduction));
  CHECK(actual.atom[0] == kSentinel);
  CHECK(near(actual.atom[1], 0.2));
  return 0;
}

int test_nonfinite_qsh_has_stable_shell_diagnostic() {
  HostCase host;
  host.batch_size = 2;
  host.atom_offsets = {0, 1, 2};
  host.shell_offsets = {0, 300, 301};
  host.qsh_offsets = host.shell_offsets;
  host.qat_offsets = host.atom_offsets;
  host.dipole_offsets = {0, 3, 6};
  host.quadrupole_offsets = {0, 6, 12};
  host.shell_to_atom.assign(300u, 0);
  host.shell_to_atom.push_back(1);
  host.active = {1u, 1u};
  host.mixed_qsh.assign(301u, 1.0e-4);
  host.mixed_qsh[299] = std::numeric_limits<double>::quiet_NaN();
  host.mixed_qsh[300] = 0.2;
  host.mixed_dipole.assign(6u, 0.0);
  host.mixed_quadrupole.assign(12u, 0.0);
  host.es2.assign(301u, 0.0);
  host.es3.assign(301u, 0.0);
  host.pc.assign(301u, 0.0);
  host.aes2_atomic.assign(2u, 0.0);
  host.aes2_dipole.assign(6u, 0.0);
  host.aes2_quadrupole.assign(12u, 0.0);
  host.periodic.assign(2u, 0.0);
  host.d4.assign(2u, 0.0);

  DeviceFixture device;
  CUDA_CHECK(device.initialize(host));
  CUDA_CHECK(reset_gfn2_scc_potential_device_errors_cuda(
      host.batch_size, device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(gather_gfn2_scc_mixed_multipoles_cuda(
      device.batch(host), device.mixed(host), device.activity(host), device.topology(host),
      device.workspace(host), device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::uint32_t> errors(2u);
  GatherResult actual;
  CUDA_CHECK(device.system_errors.copy_to(errors.data(), errors.size()));
  CUDA_CHECK(download_gather(host, device, actual));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(errors[0] ==
        static_cast<std::uint32_t>(Gfn2SccPotentialDeviceError::kNonfiniteMixedShellCharge));
  CHECK(errors[1] == 0u);
  CHECK(std::all_of(actual.shell.begin(), actual.shell.begin() + 300,
                    [](double value) { return value == kSentinel; }));
  CHECK(actual.atom[0] == kSentinel);
  CHECK(near(actual.shell[300], 0.2));
  CHECK(near(actual.atom[1], 0.2));
  return 0;
}

int test_sticky_plan_and_numerical_error() {
  const std::uint32_t plan_error =
      static_cast<std::uint32_t>(Gfn2SccPotentialDeviceError::kInvalidOffsets);
  const std::uint32_t numerical_error =
      static_cast<std::uint32_t>(Gfn2SccPotentialDeviceError::kNonfiniteES2Potential);

  /* numerical -> plan: retain the first numerical diagnostic, but let the
   * independent plan latch suppress every system in the later malformed call. */
  {
    HostCase host = make_case(2u);
    const double original_es2 = host.es2[0];
    host.es2[0] = std::numeric_limits<double>::quiet_NaN();
    DeviceFixture device;
    CUDA_CHECK(device.initialize(host));
    CUDA_CHECK(reset_gfn2_scc_potential_device_errors_cuda(
        host.batch_size, device.system_errors.get(), device.device_error.get()));
    CUDA_CHECK(compose_gfn2_scc_potentials_cuda(
        device.batch(host), device.components(host, kGfn2SccPotentialAllComponents),
        device.activity(host), device.results(host), device.workspace(host),
        device.system_errors.get(), device.device_error.get()));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::uint32_t> errors(2u);
    std::uint32_t diagnostic = 0u;
    ComposeResult actual;
    CUDA_CHECK(device.system_errors.copy_to(errors.data(), errors.size()));
    CUDA_CHECK(device.device_error.copy_to(&diagnostic, 1u));
    CUDA_CHECK(download_compose(host, device, actual));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(errors[0] == numerical_error && errors[1] == 0u);
    CHECK(diagnostic == numerical_error);
    CHECK(system_is_sentinel(host, actual, 0u));
    CHECK(!system_is_sentinel(host, actual, 1u));

    host.es2[0] = original_es2;
    CUDA_CHECK(device.es2.copy_from(host.es2.data(), host.es2.size()));
    CUDA_CHECK(device.reset_outputs(host));
    const std::int64_t hostile = std::numeric_limits<std::int64_t>::max();
    CUDA_CHECK(cudaMemcpy(device.atom_offsets.get() + host.batch_size, &hostile, sizeof(hostile),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(compose_gfn2_scc_potentials_cuda(
        device.batch(host), device.components(host, kGfn2SccPotentialAllComponents),
        device.activity(host), device.results(host), device.workspace(host),
        device.system_errors.get(), device.device_error.get()));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(device.device_error.copy_to(&diagnostic, 1u));
    CUDA_CHECK(download_compose(host, device, actual));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(diagnostic == numerical_error);
    CHECK(system_is_sentinel(host, actual, 0u) && system_is_sentinel(host, actual, 1u));
  }

  /* plan -> numerical: a sticky plan failure prevents later numerical inputs
   * from being inspected, so the plan diagnostic and zero system errors stay. */
  {
    HostCase host = make_case(2u);
    DeviceFixture device;
    CUDA_CHECK(device.initialize(host));
    CUDA_CHECK(reset_gfn2_scc_potential_device_errors_cuda(
        host.batch_size, device.system_errors.get(), device.device_error.get()));
    const std::int64_t hostile = std::numeric_limits<std::int64_t>::max();
    CUDA_CHECK(cudaMemcpy(device.atom_offsets.get() + host.batch_size, &hostile, sizeof(hostile),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(compose_gfn2_scc_potentials_cuda(
        device.batch(host), device.components(host, kGfn2SccPotentialAllComponents),
        device.activity(host), device.results(host), device.workspace(host),
        device.system_errors.get(), device.device_error.get()));
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::int64_t valid_end = host.atom_offsets.back();
    CUDA_CHECK(cudaMemcpy(device.atom_offsets.get() + host.batch_size, &valid_end,
                          sizeof(valid_end), cudaMemcpyHostToDevice));
    host.es2[0] = std::numeric_limits<double>::quiet_NaN();
    CUDA_CHECK(device.es2.copy_from(host.es2.data(), host.es2.size()));
    CUDA_CHECK(device.reset_outputs(host));
    CUDA_CHECK(compose_gfn2_scc_potentials_cuda(
        device.batch(host), device.components(host, kGfn2SccPotentialAllComponents),
        device.activity(host), device.results(host), device.workspace(host),
        device.system_errors.get(), device.device_error.get()));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::uint32_t> errors(2u);
    std::uint32_t diagnostic = 0u;
    ComposeResult actual;
    CUDA_CHECK(device.system_errors.copy_to(errors.data(), errors.size()));
    CUDA_CHECK(device.device_error.copy_to(&diagnostic, 1u));
    CUDA_CHECK(download_compose(host, device, actual));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(diagnostic == plan_error);
    CHECK(errors[0] == 0u && errors[1] == 0u);
    CHECK(system_is_sentinel(host, actual, 0u) && system_is_sentinel(host, actual, 1u));
  }
  return 0;
}

int test_invalid_active_mask_peer_isolation() {
  HostCase host = make_case(2u);
  host.active[0] = 2u;
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host));
  CUDA_CHECK(launch_sequence(device, host, kGfn2SccPotentialAllComponents));
  CUDA_CHECK(cudaDeviceSynchronize());
  ComposeResult actual;
  std::vector<std::uint32_t> errors(2u);
  CUDA_CHECK(download_compose(host, device, actual));
  CUDA_CHECK(device.system_errors.copy_to(errors.data(), errors.size()));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(errors[0] == static_cast<std::uint32_t>(Gfn2SccPotentialDeviceError::kInvalidActiveMask));
  CHECK(errors[1] == 0u);
  CHECK(system_is_sentinel(host, actual, 0u));
  CHECK(!system_is_sentinel(host, actual, 1u));
  return 0;
}

int test_alias_token_misalignment_and_reset_validation() {
  HostCase host = make_case(2u);
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host));
  auto batch = device.batch(host);
  auto mixed = device.mixed(host);
  auto activity = device.activity(host);
  auto topology = device.topology(host);
  auto components = device.components(host, kGfn2SccPotentialAllComponents);
  auto results = device.results(host);
  auto workspace = device.workspace(host);

  auto wrong_mixed = mixed;
  wrong_mixed.plan_token ^= 1u;
  CHECK(gather_gfn2_scc_mixed_multipoles_cuda(batch, wrong_mixed, activity, topology, workspace,
                                              device.system_errors.get(),
                                              device.device_error.get()) == cudaErrorInvalidValue);
  auto wrong_topology = topology;
  wrong_topology.plan_token ^= 1u;
  CHECK(gather_gfn2_scc_mixed_multipoles_cuda(batch, mixed, activity, wrong_topology, workspace,
                                              device.system_errors.get(),
                                              device.device_error.get()) == cudaErrorInvalidValue);
  auto wrong_components = components;
  wrong_components.plan_token ^= 1u;
  CHECK(compose_gfn2_scc_potentials_cuda(batch, wrong_components, activity, results, workspace,
                                         device.system_errors.get(),
                                         device.device_error.get()) == cudaErrorInvalidValue);
  auto wrong_results = results;
  wrong_results.plan_token ^= 1u;
  CHECK(compose_gfn2_scc_potentials_cuda(batch, components, activity, wrong_results, workspace,
                                         device.system_errors.get(),
                                         device.device_error.get()) == cudaErrorInvalidValue);
  auto wrong_workspace = workspace;
  wrong_workspace.plan_token ^= 1u;
  CHECK(compose_gfn2_scc_potentials_cuda(batch, components, activity, results, wrong_workspace,
                                         device.system_errors.get(),
                                         device.device_error.get()) == cudaErrorInvalidValue);
  auto wrong_batch = batch;
  wrong_batch.plan_token = 0u;
  CHECK(compose_gfn2_scc_potentials_cuda(wrong_batch, components, activity, results, workspace,
                                         device.system_errors.get(),
                                         device.device_error.get()) == cudaErrorInvalidValue);
  auto alias_topology = topology;
  alias_topology.shell_charges = workspace.shell_scratch;
  CHECK(gather_gfn2_scc_mixed_multipoles_cuda(batch, mixed, activity, alias_topology, workspace,
                                              device.system_errors.get(),
                                              device.device_error.get()) == cudaErrorInvalidValue);
  auto partial_alias_topology = topology;
  partial_alias_topology.shell_charges = workspace.shell_scratch + 1;
  CHECK(gather_gfn2_scc_mixed_multipoles_cuda(batch, mixed, activity, partial_alias_topology,
                                              workspace, device.system_errors.get(),
                                              device.device_error.get()) == cudaErrorInvalidValue);
  auto alias_results = results;
  alias_results.atomic = workspace.atom_scratch;
  CHECK(compose_gfn2_scc_potentials_cuda(batch, components, activity, alias_results, workspace,
                                         device.system_errors.get(),
                                         device.device_error.get()) == cudaErrorInvalidValue);
  auto input_alias_results = results;
  input_alias_results.shell = const_cast<double*>(components.es2_shell);
  CHECK(compose_gfn2_scc_potentials_cuda(batch, components, activity, input_alias_results,
                                         workspace, device.system_errors.get(),
                                         device.device_error.get()) == cudaErrorInvalidValue);
  auto misaligned = batch;
  misaligned.qsh_offsets =
      reinterpret_cast<const std::int64_t*>(reinterpret_cast<const char*>(batch.qsh_offsets) + 1);
  CHECK(gather_gfn2_scc_mixed_multipoles_cuda(misaligned, mixed, activity, topology, workspace,
                                              device.system_errors.get(),
                                              device.device_error.get()) == cudaErrorInvalidValue);
  auto invalid_disabled = device.components(host, 0u);
  invalid_disabled.es2_shell = device.es2.get();
  CHECK(compose_gfn2_scc_potentials_cuda(batch, invalid_disabled, activity, results, workspace,
                                         device.system_errors.get(),
                                         device.device_error.get()) == cudaErrorInvalidValue);
  auto wrong_activity = activity;
  wrong_activity.plan_token ^= 1u;
  CHECK(compose_gfn2_scc_potentials_cuda(batch, components, wrong_activity, results, workspace,
                                         device.system_errors.get(),
                                         device.device_error.get()) == cudaErrorInvalidValue);
  auto overflow_batch = batch;
  overflow_batch.total_atoms = std::numeric_limits<std::int64_t>::max() /
                                   gpuxtb::detail::cuda::kGfn2SccPotentialQuadrupoleComponents +
                               1;
  CHECK(compose_gfn2_scc_potentials_cuda(overflow_batch, components, activity, results, workspace,
                                         device.system_errors.get(),
                                         device.device_error.get()) == cudaErrorInvalidValue);
  CHECK(reset_gfn2_scc_potential_device_errors_cuda(
            2, device.system_errors.get(), device.system_errors.get()) == cudaErrorInvalidValue);
  CHECK(reset_gfn2_scc_potential_device_errors_cuda(2, device.system_errors.get(),
                                                    device.system_errors.get() + 1) ==
        cudaErrorInvalidValue);
  return 0;
}

}  // namespace

int main() {
  for (const std::size_t batch_size : {1u, 8u, 32u, 128u}) {
    if (const int status = run_parity(batch_size, false); status != 0) {
      std::fprintf(stderr, "CUDA SCC potential batch-%zu failed at line %d\n", batch_size, status);
      return status;
    }
  }
  if (const int status = run_parity(8u, true); status != 0) {
    std::fprintf(stderr, "CUDA SCC potential Graph failed at line %d\n", status);
    return status;
  }
  const std::array<int (*)(), 11> tests{{
      test_disabled_null_zero,
      test_one_hot_component_wiring,
      test_inactive_poison_and_bad_mapping_peer_isolation,
      test_hostile_offsets_fail_closed,
      test_component_and_addition_overflow,
      test_strict_cpu_association_order,
      test_gather_reduction_overflow,
      test_nonfinite_qsh_has_stable_shell_diagnostic,
      test_sticky_plan_and_numerical_error,
      test_invalid_active_mask_peer_isolation,
      test_alias_token_misalignment_and_reset_validation,
  }};
  for (const auto test : tests) {
    const int status = test();
    if (status != 0) {
      std::fprintf(stderr, "CUDA SCC potential regression failed at line %d\n", status);
      return status;
    }
  }
  std::puts("CUDA SCC potential tests passed");
  return 0;
}
