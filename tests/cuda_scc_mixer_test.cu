#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_scc_mixer.cuh"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/scc_mixer.hpp"
#include "model/gfn2/wavefunction.hpp"

#define CHECK(condition)                                                                          \
  do {                                                                                            \
    if (!(condition)) {                                                                           \
      std::fprintf(stderr, "CUDA SCC mixer test failure at line %d: %s\n", __LINE__, #condition); \
      return __LINE__;                                                                            \
    }                                                                                             \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using gpuxtb::detail::Gfn2PlanMemorySpace;
using gpuxtb::detail::Gfn2WavefunctionLayoutView;
using gpuxtb::detail::cuda::Gfn2SccDeviceBatch;
using gpuxtb::detail::cuda::Gfn2SccDeviceConstMultipoles;
using gpuxtb::detail::cuda::Gfn2SccDeviceMultipoles;
using gpuxtb::detail::cuda::Gfn2SccIterationDeviceActivity;
using gpuxtb::detail::cuda::Gfn2SccIterationDeviceLedger;
using gpuxtb::detail::cuda::Gfn2SccMixerDeviceError;
using gpuxtb::detail::cuda::Gfn2SccMixerDevicePolicy;
using gpuxtb::detail::cuda::Gfn2SccMixerDeviceState;
using gpuxtb::detail::cuda::Gfn2SccMixerDeviceWorkspace;
using gpuxtb::detail::cuda::Gfn2SccStageCodeFormat;
using gpuxtb::detail::cuda::Gfn2SccStageDeviceReport;
using gpuxtb::detail::cuda::Gfn2SccStageId;
using gpuxtb::detail::gfn2::BasisPlan;
using gpuxtb::detail::gfn2::SccMixerPlan;
using gpuxtb::detail::gfn2::SccMixerState;
using gpuxtb::detail::gfn2::SccMixerWorkspace;
using gpuxtb::detail::gfn2::WavefunctionLayout;
using gpuxtb::detail::gfn2::WavefunctionView;

constexpr std::uint64_t kPlanToken = 0x56b0d4e7c9012a3fULL;
constexpr std::int64_t kHistorySize = 4;
constexpr double kDamping = 0.4;
constexpr double kRmsTolerance = 1.0e-30;
constexpr double kMaximumTolerance = 1.0e-30;

class AlignedBuffer {
 public:
  explicit AlignedBuffer(std::size_t bytes) : bytes_(bytes) {
    data_ = std::aligned_alloc(gpuxtb::detail::gfn2::kSccMixerWorkspaceAlignment, bytes_);
    if (data_ != nullptr) {
      std::memset(data_, 0, bytes_);
    }
  }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
  ~AlignedBuffer() { std::free(data_); }
  void* data() const { return data_; }
  std::size_t size() const { return bytes_; }

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0u;
};

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
  cudaError_t fill_zero(cudaStream_t stream = nullptr) {
    return cudaMemsetAsync(data_, 0, count_ * sizeof(T), stream);
  }
  T* get() { return data_; }
  const T* get() const { return data_; }
  std::size_t size() const { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

struct CpuFixture {
  BasisPlan basis;
  WavefunctionLayout layout;
  SccMixerPlan plan;
  std::unique_ptr<AlignedBuffer> wavefunction_storage;
  std::unique_ptr<AlignedBuffer> state_storage;
  std::unique_ptr<AlignedBuffer> scratch_storage;
  WavefunctionView wavefunction;
  SccMixerState state;
  SccMixerWorkspace scratch;
};

bool make_cpu_fixture(std::size_t batch_size, CpuFixture& fixture, std::string& error,
                      double rms_tolerance = kRmsTolerance,
                      double maximum_tolerance = kMaximumTolerance) {
  std::vector<std::int64_t> atom_offsets(batch_size + 1u, 0);
  std::vector<std::int32_t> atomic_numbers;
  for (std::size_t system = 0u; system < batch_size; ++system) {
    const std::size_t atoms = 1u + system % 4u;
    for (std::size_t atom = 0u; atom < atoms; ++atom) {
      /* Even-Z atoms keep every neutral test system closed shell while shell
       * counts and atom counts remain ragged. */
      constexpr std::array<std::int32_t, 4> elements{{2, 6, 8, 10}};
      atomic_numbers.push_back(elements[(system + atom) % elements.size()]);
    }
    atom_offsets[system + 1u] = static_cast<std::int64_t>(atomic_numbers.size());
  }
  std::vector<double> charges(batch_size, 0.0);
  std::vector<std::int32_t> unpaired(batch_size, 0);
  std::vector<std::int32_t> spin_channels(batch_size, 1);
  if (gpuxtb::detail::gfn2::make_basis_plan(static_cast<std::int64_t>(batch_size),
                                            static_cast<std::int64_t>(atomic_numbers.size()),
                                            atom_offsets.data(), atomic_numbers.data(),
                                            fixture.basis, error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::make_wavefunction_layout(
          fixture.basis, atomic_numbers.data(), charges.data(), unpaired.data(),
          spin_channels.data(), fixture.layout, error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::make_scc_mixer_plan(fixture.layout, kHistorySize, kDamping,
                                                rms_tolerance, maximum_tolerance, fixture.plan,
                                                error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }

  fixture.wavefunction_storage =
      std::make_unique<AlignedBuffer>(fixture.layout.workspace_size_bytes);
  fixture.state_storage = std::make_unique<AlignedBuffer>(fixture.plan.state_size_bytes());
  fixture.scratch_storage = std::make_unique<AlignedBuffer>(fixture.plan.workspace_size_bytes());
  if (fixture.wavefunction_storage->data() == nullptr || fixture.state_storage->data() == nullptr ||
      fixture.scratch_storage->data() == nullptr) {
    error = "CPU fixture allocation failed";
    return false;
  }
  if (gpuxtb::detail::gfn2::bind_wavefunction_view(
          fixture.layout, fixture.wavefunction_storage->data(),
          fixture.wavefunction_storage->size(), fixture.wavefunction,
          error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::bind_scc_mixer_state(fixture.plan, fixture.state_storage->data(),
                                                 fixture.state_storage->size(), fixture.state,
                                                 error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::bind_scc_mixer_workspace(
          fixture.plan, fixture.scratch_storage->data(), fixture.scratch_storage->size(),
          fixture.scratch, error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }

  for (std::int64_t shell = 0; shell < fixture.layout.qsh.element_count; ++shell) {
    fixture.wavefunction.qsh[shell] = 0.002 * static_cast<double>(shell + 1);
  }
  for (std::int64_t component = 0; component < fixture.layout.dipole.element_count; ++component) {
    fixture.wavefunction.dipole[component] = -0.0015 * static_cast<double>(component % 23 + 1);
  }
  for (std::int64_t component = 0; component < fixture.layout.quadrupole.element_count;
       ++component) {
    fixture.wavefunction.quadrupole[component] = 0.0007 * static_cast<double>(component % 31 + 1);
  }
  return true;
}

std::int64_t vector_begin(const CpuFixture& fixture, std::size_t system) {
  return fixture.layout.batch_shell_offsets[system] + 9 * fixture.layout.atom_offsets[system];
}

std::int64_t dimension(const CpuFixture& fixture, std::size_t system) {
  return fixture.plan.vector_offsets()[system + 1u] - fixture.plan.vector_offsets()[system];
}

double get_field_component(const CpuFixture& fixture, std::size_t system, std::int64_t component) {
  const std::int64_t shell_begin = fixture.layout.batch_shell_offsets[system];
  const std::int64_t shells = fixture.layout.batch_shell_offsets[system + 1u] - shell_begin;
  if (component < shells) {
    return fixture.wavefunction.qsh[shell_begin + component];
  }
  component -= shells;
  const std::int64_t atom_begin = fixture.layout.atom_offsets[system];
  const std::int64_t atoms = fixture.layout.atom_offsets[system + 1u] - atom_begin;
  if (component < atoms * 3) {
    return fixture.wavefunction.dipole[atom_begin * 3 + component];
  }
  component -= atoms * 3;
  return fixture.wavefunction.quadrupole[atom_begin * 6 + component];
}

void set_field_component(CpuFixture& fixture, std::size_t system, std::int64_t component,
                         double value) {
  const std::int64_t shell_begin = fixture.layout.batch_shell_offsets[system];
  const std::int64_t shells = fixture.layout.batch_shell_offsets[system + 1u] - shell_begin;
  if (component < shells) {
    fixture.wavefunction.qsh[shell_begin + component] = value;
    return;
  }
  component -= shells;
  const std::int64_t atom_begin = fixture.layout.atom_offsets[system];
  const std::int64_t atoms = fixture.layout.atom_offsets[system + 1u] - atom_begin;
  if (component < atoms * 3) {
    fixture.wavefunction.dipole[atom_begin * 3 + component] = value;
    return;
  }
  component -= atoms * 3;
  fixture.wavefunction.quadrupole[atom_begin * 6 + component] = value;
}

double test_residual(std::size_t system, std::size_t iteration, std::int64_t component) {
  const double index = static_cast<double>(component + 1);
  const double alternating = component % 2 == 0 ? 1.0 : -1.0;
  const double phase = static_cast<double>(component % 5) - 2.0;
  return 0.00017 * static_cast<double>(system % 11u + 1u) * index +
         0.00031 * static_cast<double>(iteration + 1u) * alternating +
         0.000043 * static_cast<double>(iteration) * phase;
}

void install_raw_iteration(CpuFixture& fixture, std::size_t iteration) {
  const std::size_t batch = static_cast<std::size_t>(fixture.layout.batch_size);
  for (std::size_t system = 0u; system < batch; ++system) {
    const std::int64_t begin = vector_begin(fixture, system);
    const std::int64_t count = dimension(fixture, system);
    for (std::int64_t component = 0; component < count; ++component) {
      set_field_component(fixture, system, component,
                          fixture.state.current_inputs[begin + component] +
                              test_residual(system, iteration, component));
    }
  }
}

struct StateSnapshot {
  std::vector<double> current;
  std::vector<double> previous;
  std::vector<double> previous_residual;
  std::vector<double> df;
  std::vector<double> u;
  std::vector<double> omega;
  std::vector<double> rms;
  std::vector<double> maximum;
  std::vector<std::uint64_t> iterations;
  std::vector<std::uint64_t> restarts;
  std::vector<gpuxtb_status_t> statuses;
  std::vector<std::uint8_t> initialized;
  std::vector<std::uint8_t> residual_converged;
};

struct MultipoleSnapshot {
  std::vector<double> shell;
  std::vector<double> dipole;
  std::vector<double> quadrupole;
};

struct DeviceFixture {
  DeviceBuffer<std::int64_t> shell_offsets;
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int32_t> spin_channels;
  DeviceBuffer<std::int64_t> spin_channel_offsets;
  DeviceBuffer<std::int64_t> spin_orbital_offsets;
  DeviceBuffer<std::int64_t> spin_matrix_offsets;
  DeviceBuffer<std::int64_t> spin_shell_offsets;
  DeviceBuffer<std::int64_t> spin_atom_offsets;
  DeviceBuffer<double> raw_shell;
  DeviceBuffer<double> raw_dipole;
  DeviceBuffer<double> raw_quadrupole;
  DeviceBuffer<double> output_shell;
  DeviceBuffer<double> output_dipole;
  DeviceBuffer<double> output_quadrupole;
  DeviceBuffer<double> current;
  DeviceBuffer<double> previous;
  DeviceBuffer<double> previous_residual;
  DeviceBuffer<double> df;
  DeviceBuffer<double> u;
  DeviceBuffer<double> omega;
  DeviceBuffer<double> rms;
  DeviceBuffer<double> maximum;
  DeviceBuffer<std::uint64_t> iterations;
  DeviceBuffer<std::uint64_t> restarts;
  DeviceBuffer<gpuxtb_status_t> statuses;
  DeviceBuffer<std::uint8_t> initialized;
  DeviceBuffer<std::uint8_t> residual_converged;
  DeviceBuffer<std::uint8_t> active_mask;
  DeviceBuffer<std::uint32_t> canonical_sequence;
  DeviceBuffer<double> residual_scratch;
  DeviceBuffer<double> mixed_scratch;
  DeviceBuffer<double> df_scratch;
  DeviceBuffer<double> u_scratch;
  DeviceBuffer<double> beta;
  DeviceBuffer<double> coefficients;
  DeviceBuffer<std::uint32_t> sequence;
  DeviceBuffer<std::uint32_t> error;

  Gfn2SccDeviceBatch batch;
  Gfn2WavefunctionLayoutView layout;
  Gfn2SccMixerDevicePolicy policy;
  Gfn2SccDeviceConstMultipoles raw;
  Gfn2SccDeviceMultipoles output;
  Gfn2SccIterationDeviceActivity activity;
  Gfn2SccMixerDeviceState state;
  Gfn2SccMixerDeviceWorkspace workspace;

  cudaError_t setup(const CpuFixture& cpu, cudaStream_t stream = nullptr, bool mixed_spin = false) {
    const std::size_t batch_size = static_cast<std::size_t>(cpu.layout.batch_size);
    const std::size_t physical_shells = static_cast<std::size_t>(cpu.layout.total_shells);
    const std::size_t physical_atoms = static_cast<std::size_t>(cpu.layout.total_atoms);
    std::vector<std::int32_t> spin_channels_host(batch_size, 1);
    std::vector<std::int64_t> spin_channel_offsets_host(batch_size + 1u, 0);
    std::vector<std::int64_t> spin_orbital_offsets_host(batch_size + 1u, 0);
    std::vector<std::int64_t> spin_matrix_offsets_host(batch_size + 1u, 0);
    std::vector<std::int64_t> spin_shell_offsets_host(batch_size + 1u, 0);
    std::vector<std::int64_t> spin_atom_offsets_host(batch_size + 1u, 0);
    for (std::size_t system = 0u; system < batch_size; ++system) {
      const std::int32_t channels = mixed_spin && system % 2u == 0u ? 2 : 1;
      spin_channels_host[system] = channels;
      spin_channel_offsets_host[system + 1u] = spin_channel_offsets_host[system] + channels;
      spin_orbital_offsets_host[system + 1u] =
          spin_orbital_offsets_host[system] +
          channels * (cpu.layout.batch_orbital_offsets[system + 1u] -
                      cpu.layout.batch_orbital_offsets[system]);
      const std::int64_t orbitals =
          cpu.layout.batch_orbital_offsets[system + 1u] - cpu.layout.batch_orbital_offsets[system];
      spin_matrix_offsets_host[system + 1u] =
          spin_matrix_offsets_host[system] + channels * orbitals * orbitals;
      spin_shell_offsets_host[system + 1u] =
          spin_shell_offsets_host[system] +
          channels * (cpu.layout.batch_shell_offsets[system + 1u] -
                      cpu.layout.batch_shell_offsets[system]);
      spin_atom_offsets_host[system + 1u] =
          spin_atom_offsets_host[system] +
          channels * (cpu.layout.atom_offsets[system + 1u] - cpu.layout.atom_offsets[system]);
    }
    const std::size_t shells =
        mixed_spin ? static_cast<std::size_t>(spin_shell_offsets_host.back()) : physical_shells;
    const std::size_t atoms =
        mixed_spin ? static_cast<std::size_t>(spin_atom_offsets_host.back()) : physical_atoms;
    const std::size_t vectors = shells + atoms * 9u;
    const std::size_t history = vectors * static_cast<std::size_t>(kHistorySize);
    const std::size_t omega_count = batch_size * static_cast<std::size_t>(kHistorySize);
    const std::size_t beta_count = omega_count * static_cast<std::size_t>(kHistorySize);
    const std::array<cudaError_t, 31> allocations{{
        shell_offsets.allocate(batch_size + 1u),
        atom_offsets.allocate(batch_size + 1u),
        raw_shell.allocate(shells),
        raw_dipole.allocate(atoms * 3u),
        raw_quadrupole.allocate(atoms * 6u),
        output_shell.allocate(shells),
        output_dipole.allocate(atoms * 3u),
        output_quadrupole.allocate(atoms * 6u),
        current.allocate(vectors),
        previous.allocate(vectors),
        previous_residual.allocate(vectors),
        df.allocate(history),
        u.allocate(history),
        omega.allocate(omega_count),
        rms.allocate(batch_size),
        maximum.allocate(batch_size),
        iterations.allocate(batch_size),
        restarts.allocate(batch_size),
        statuses.allocate(batch_size),
        initialized.allocate(batch_size),
        residual_converged.allocate(batch_size),
        active_mask.allocate(batch_size),
        canonical_sequence.allocate(1u),
        residual_scratch.allocate(vectors),
        mixed_scratch.allocate(vectors),
        df_scratch.allocate(vectors),
        u_scratch.allocate(vectors),
        beta.allocate(beta_count),
        coefficients.allocate(omega_count),
        sequence.allocate(1u),
        error.allocate(1u),
    }};
    for (cudaError_t allocation : allocations) {
      if (allocation != cudaSuccess) {
        return allocation;
      }
    }
    if (mixed_spin) {
      const std::array<cudaError_t, 6> spin_allocations{{
          spin_channels.allocate(batch_size),
          spin_channel_offsets.allocate(batch_size + 1u),
          spin_orbital_offsets.allocate(batch_size + 1u),
          spin_matrix_offsets.allocate(batch_size + 1u),
          spin_shell_offsets.allocate(batch_size + 1u),
          spin_atom_offsets.allocate(batch_size + 1u),
      }};
      for (cudaError_t allocation : spin_allocations) {
        if (allocation != cudaSuccess) return allocation;
      }
    }

    /* The SCC batch always describes the physical topology.  Spin-expanded
     * multipole extents belong exclusively to the wavefunction layout; using
     * them here would make the physical offset endpoints fail validation. */
    batch = {static_cast<std::int64_t>(batch_size),
             static_cast<std::int64_t>(physical_shells),
             static_cast<std::int64_t>(physical_atoms),
             static_cast<std::int64_t>(batch_size + 1u),
             static_cast<std::int64_t>(batch_size + 1u),
             kPlanToken,
             shell_offsets.get(),
             atom_offsets.get()};
    policy = {cpu.plan.history_size(), cpu.plan.damping(), cpu.plan.rms_tolerance(),
              cpu.plan.maximum_tolerance(), kPlanToken};
    layout = {};
    if (mixed_spin) {
      layout.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
      layout.plan_token = kPlanToken;
      layout.batch_size = static_cast<std::int64_t>(batch_size);
      layout.total_spin_channels = spin_channel_offsets_host.back();
      layout.total_spin_orbitals = spin_orbital_offsets_host.back();
      layout.total_spin_matrix_elements = spin_matrix_offsets_host.back();
      layout.total_spin_shells = spin_shell_offsets_host.back();
      layout.total_spin_atoms = spin_atom_offsets_host.back();
      layout.spin_channel_count = static_cast<std::int64_t>(batch_size);
      layout.spin_channel_offset_count = static_cast<std::int64_t>(batch_size + 1u);
      layout.spin_orbital_offset_count = static_cast<std::int64_t>(batch_size + 1u);
      layout.spin_matrix_offset_count = static_cast<std::int64_t>(batch_size + 1u);
      layout.spin_shell_offset_count = static_cast<std::int64_t>(batch_size + 1u);
      layout.spin_atom_offset_count = static_cast<std::int64_t>(batch_size + 1u);
      layout.spin_channels = spin_channels.get();
      layout.spin_channel_offsets = spin_channel_offsets.get();
      layout.spin_orbital_offsets = spin_orbital_offsets.get();
      layout.spin_matrix_offsets = spin_matrix_offsets.get();
      layout.spin_shell_offsets = spin_shell_offsets.get();
      layout.spin_atom_offsets = spin_atom_offsets.get();
    }
    raw = {raw_shell.get(),
           static_cast<std::int64_t>(shells),
           raw_dipole.get(),
           static_cast<std::int64_t>(atoms * 3u),
           raw_quadrupole.get(),
           static_cast<std::int64_t>(atoms * 6u),
           kPlanToken};
    output = {output_shell.get(),
              static_cast<std::int64_t>(shells),
              output_dipole.get(),
              static_cast<std::int64_t>(atoms * 3u),
              output_quadrupole.get(),
              static_cast<std::int64_t>(atoms * 6u),
              kPlanToken};
    activity = {active_mask.get(), canonical_sequence.get(), static_cast<std::int64_t>(batch_size),
                1, kPlanToken};
    state = {current.get(),
             previous.get(),
             previous_residual.get(),
             df.get(),
             u.get(),
             omega.get(),
             rms.get(),
             maximum.get(),
             iterations.get(),
             restarts.get(),
             statuses.get(),
             initialized.get(),
             residual_converged.get(),
             static_cast<std::int64_t>(vectors),
             static_cast<std::int64_t>(history),
             static_cast<std::int64_t>(omega_count),
             static_cast<std::int64_t>(batch_size),
             kPlanToken};
    workspace = {residual_scratch.get(),
                 mixed_scratch.get(),
                 df_scratch.get(),
                 u_scratch.get(),
                 beta.get(),
                 coefficients.get(),
                 sequence.get(),
                 static_cast<std::int64_t>(vectors),
                 static_cast<std::int64_t>(beta_count),
                 static_cast<std::int64_t>(omega_count),
                 1,
                 kPlanToken};
    cudaError_t status =
        shell_offsets.copy_from(cpu.layout.batch_shell_offsets.data(), batch_size + 1u, stream);
    if (status == cudaSuccess) {
      status = atom_offsets.copy_from(cpu.layout.atom_offsets.data(), batch_size + 1u, stream);
    }
    if (status == cudaSuccess && mixed_spin) {
      status = spin_channels.copy_from(spin_channels_host.data(), batch_size, stream);
    }
    if (status == cudaSuccess && mixed_spin) {
      status =
          spin_channel_offsets.copy_from(spin_channel_offsets_host.data(), batch_size + 1u, stream);
    }
    if (status == cudaSuccess && mixed_spin) {
      status =
          spin_orbital_offsets.copy_from(spin_orbital_offsets_host.data(), batch_size + 1u, stream);
    }
    if (status == cudaSuccess && mixed_spin) {
      status =
          spin_matrix_offsets.copy_from(spin_matrix_offsets_host.data(), batch_size + 1u, stream);
    }
    if (status == cudaSuccess && mixed_spin) {
      status =
          spin_shell_offsets.copy_from(spin_shell_offsets_host.data(), batch_size + 1u, stream);
    }
    if (status == cudaSuccess && mixed_spin) {
      status = spin_atom_offsets.copy_from(spin_atom_offsets_host.data(), batch_size + 1u, stream);
    }
    if (status == cudaSuccess && !mixed_spin) {
      status = copy_raw(cpu, stream);
    }
    if (status == cudaSuccess && mixed_spin) {
      std::vector<double> initial_shell(shells, 0.0);
      std::vector<double> initial_dipole(atoms * 3u, 0.0);
      std::vector<double> initial_quadrupole(atoms * 6u, 0.0);
      for (std::size_t index = 0u; index < initial_shell.size(); ++index) {
        initial_shell[index] = 0.001 * static_cast<double>(index + 1u);
      }
      for (std::size_t index = 0u; index < initial_dipole.size(); ++index) {
        initial_dipole[index] = -0.0007 * static_cast<double>(index + 1u);
      }
      for (std::size_t index = 0u; index < initial_quadrupole.size(); ++index) {
        initial_quadrupole[index] = 0.0003 * static_cast<double>(index + 1u);
      }
      status = raw_shell.copy_from(initial_shell.data(), initial_shell.size(), stream);
      if (status == cudaSuccess) {
        status = raw_dipole.copy_from(initial_dipole.data(), initial_dipole.size(), stream);
      }
      if (status == cudaSuccess) {
        status =
            raw_quadrupole.copy_from(initial_quadrupole.data(), initial_quadrupole.size(), stream);
      }
    }
    if (status == cudaSuccess) {
      status = set_all_active(stream);
    }
    return status;
  }

  cudaError_t set_activity(const std::vector<std::uint8_t>& mask, std::uint32_t sequence_value,
                           cudaStream_t stream = nullptr) {
    if (mask.size() != active_mask.size()) {
      return cudaErrorInvalidValue;
    }
    cudaError_t status = active_mask.copy_from(mask.data(), mask.size(), stream);
    if (status == cudaSuccess) {
      status = canonical_sequence.copy_from(&sequence_value, 1u, stream);
    }
    return status;
  }

  cudaError_t set_all_active(cudaStream_t stream = nullptr) {
    return set_activity(std::vector<std::uint8_t>(active_mask.size(), 1u), 1u, stream);
  }

  cudaError_t copy_raw(const CpuFixture& cpu, cudaStream_t stream = nullptr) {
    cudaError_t status = raw_shell.copy_from(cpu.wavefunction.qsh, raw_shell.size(), stream);
    if (status == cudaSuccess) {
      status = raw_dipole.copy_from(cpu.wavefunction.dipole, raw_dipole.size(), stream);
    }
    if (status == cudaSuccess) {
      status = raw_quadrupole.copy_from(cpu.wavefunction.quadrupole, raw_quadrupole.size(), stream);
    }
    return status;
  }

  cudaError_t reset_error(cudaStream_t stream = nullptr) { return error.fill_zero(stream); }

  cudaError_t fill_output(double value, cudaStream_t stream = nullptr) {
    std::vector<double> shell(output_shell.size(), value);
    std::vector<double> dipole(output_dipole.size(), value);
    std::vector<double> quadrupole(output_quadrupole.size(), value);
    cudaError_t status = output_shell.copy_from(shell.data(), shell.size(), stream);
    if (status == cudaSuccess) {
      status = output_dipole.copy_from(dipole.data(), dipole.size(), stream);
    }
    if (status == cudaSuccess) {
      status = output_quadrupole.copy_from(quadrupole.data(), quadrupole.size(), stream);
    }
    return status;
  }

  cudaError_t snapshot_output(MultipoleSnapshot& snapshot, cudaStream_t stream = nullptr) const {
    snapshot.shell.resize(output_shell.size());
    snapshot.dipole.resize(output_dipole.size());
    snapshot.quadrupole.resize(output_quadrupole.size());
    cudaError_t status = output_shell.copy_to(snapshot.shell.data(), snapshot.shell.size(), stream);
    if (status == cudaSuccess) {
      status = output_dipole.copy_to(snapshot.dipole.data(), snapshot.dipole.size(), stream);
    }
    if (status == cudaSuccess) {
      status =
          output_quadrupole.copy_to(snapshot.quadrupole.data(), snapshot.quadrupole.size(), stream);
    }
    return status == cudaSuccess ? cudaStreamSynchronize(stream) : status;
  }

  cudaError_t snapshot_raw(MultipoleSnapshot& snapshot, cudaStream_t stream = nullptr) const {
    snapshot.shell.resize(raw_shell.size());
    snapshot.dipole.resize(raw_dipole.size());
    snapshot.quadrupole.resize(raw_quadrupole.size());
    cudaError_t status = raw_shell.copy_to(snapshot.shell.data(), snapshot.shell.size(), stream);
    if (status == cudaSuccess) {
      status = raw_dipole.copy_to(snapshot.dipole.data(), snapshot.dipole.size(), stream);
    }
    if (status == cudaSuccess) {
      status =
          raw_quadrupole.copy_to(snapshot.quadrupole.data(), snapshot.quadrupole.size(), stream);
    }
    return status == cudaSuccess ? cudaStreamSynchronize(stream) : status;
  }

  cudaError_t install_raw_iteration(const CpuFixture& cpu, std::size_t iteration,
                                    cudaStream_t stream = nullptr) {
    StateSnapshot snapshot;
    cudaError_t status = this->snapshot(snapshot, stream);
    if (status != cudaSuccess) {
      return status;
    }
    std::vector<double> shell(raw_shell.size());
    std::vector<double> dipole(raw_dipole.size());
    std::vector<double> quadrupole(raw_quadrupole.size());
    const std::size_t batch_size = static_cast<std::size_t>(cpu.layout.batch_size);
    for (std::size_t system = 0u; system < batch_size; ++system) {
      const std::int64_t packed_begin = vector_begin(cpu, system);
      const std::int64_t shell_begin = cpu.layout.batch_shell_offsets[system];
      const std::int64_t shells = cpu.layout.batch_shell_offsets[system + 1u] - shell_begin;
      const std::int64_t atom_begin = cpu.layout.atom_offsets[system];
      const std::int64_t atoms = cpu.layout.atom_offsets[system + 1u] - atom_begin;
      const std::int64_t count = shells + atoms * 9;
      for (std::int64_t component = 0; component < count; ++component) {
        const double value = snapshot.current[static_cast<std::size_t>(packed_begin + component)] +
                             test_residual(system, iteration, component);
        if (component < shells) {
          shell[static_cast<std::size_t>(shell_begin + component)] = value;
        } else if (component < shells + atoms * 3) {
          dipole[static_cast<std::size_t>(atom_begin * 3 + component - shells)] = value;
        } else {
          quadrupole[static_cast<std::size_t>(atom_begin * 6 + component - shells - atoms * 3)] =
              value;
        }
      }
    }
    status = raw_shell.copy_from(shell.data(), shell.size(), stream);
    if (status == cudaSuccess) {
      status = raw_dipole.copy_from(dipole.data(), dipole.size(), stream);
    }
    if (status == cudaSuccess) {
      status = raw_quadrupole.copy_from(quadrupole.data(), quadrupole.size(), stream);
    }
    return status;
  }

  cudaError_t snapshot(StateSnapshot& snapshot, cudaStream_t stream = nullptr) const {
    snapshot.current.resize(current.size());
    snapshot.previous.resize(previous.size());
    snapshot.previous_residual.resize(previous_residual.size());
    snapshot.df.resize(df.size());
    snapshot.u.resize(u.size());
    snapshot.omega.resize(omega.size());
    snapshot.rms.resize(rms.size());
    snapshot.maximum.resize(maximum.size());
    snapshot.iterations.resize(iterations.size());
    snapshot.restarts.resize(restarts.size());
    snapshot.statuses.resize(statuses.size());
    snapshot.initialized.resize(initialized.size());
    snapshot.residual_converged.resize(residual_converged.size());
    const std::array<cudaError_t, 13> copies{{
        current.copy_to(snapshot.current.data(), snapshot.current.size(), stream),
        previous.copy_to(snapshot.previous.data(), snapshot.previous.size(), stream),
        previous_residual.copy_to(snapshot.previous_residual.data(),
                                  snapshot.previous_residual.size(), stream),
        df.copy_to(snapshot.df.data(), snapshot.df.size(), stream),
        u.copy_to(snapshot.u.data(), snapshot.u.size(), stream),
        omega.copy_to(snapshot.omega.data(), snapshot.omega.size(), stream),
        rms.copy_to(snapshot.rms.data(), snapshot.rms.size(), stream),
        maximum.copy_to(snapshot.maximum.data(), snapshot.maximum.size(), stream),
        iterations.copy_to(snapshot.iterations.data(), snapshot.iterations.size(), stream),
        restarts.copy_to(snapshot.restarts.data(), snapshot.restarts.size(), stream),
        statuses.copy_to(snapshot.statuses.data(), snapshot.statuses.size(), stream),
        initialized.copy_to(snapshot.initialized.data(), snapshot.initialized.size(), stream),
        residual_converged.copy_to(snapshot.residual_converged.data(),
                                   snapshot.residual_converged.size(), stream),
    }};
    for (cudaError_t copy : copies) {
      if (copy != cudaSuccess) {
        return copy;
      }
    }
    return cudaStreamSynchronize(stream);
  }
};

bool near(double first, double second, double absolute_tolerance = 2.0e-7,
          double relative_tolerance = 2.0e-7) {
  return std::abs(first - second) <=
         absolute_tolerance + relative_tolerance * std::max(std::abs(first), std::abs(second));
}

bool near_arrays(const double* first, const double* second, std::size_t count,
                 double absolute_tolerance = 2.0e-7, double relative_tolerance = 2.0e-7) {
  for (std::size_t index = 0u; index < count; ++index) {
    if (!near(first[index], second[index], absolute_tolerance, relative_tolerance)) {
      std::fprintf(stderr, "array mismatch at %zu: %.17g versus %.17g (abs %.3g, rel %.3g)\n",
                   index, first[index], second[index], absolute_tolerance, relative_tolerance);
      return false;
    }
  }
  return true;
}

int initialize_device(DeviceFixture& device, cudaStream_t stream = nullptr) {
  CUDA_CHECK(device.reset_error(stream));
  const cudaError_t status = device.layout.spin_channels == nullptr
                                 ? gpuxtb::detail::cuda::initialize_gfn2_scc_mixer_cuda(
                                       device.batch, device.policy, device.raw, device.state,
                                       device.workspace, device.error.get(), stream)
                                 : gpuxtb::detail::cuda::initialize_gfn2_scc_mixer_cuda(
                                       device.batch, device.layout, device.policy, device.raw,
                                       device.state, device.workspace, device.error.get(), stream);
  CUDA_CHECK(status);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::uint32_t error = 99u;
  CUDA_CHECK(device.error.copy_to(&error, 1u));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(error == static_cast<std::uint32_t>(Gfn2SccMixerDeviceError::kSuccess));
  return 0;
}

int compare_cpu_and_device(const CpuFixture& cpu, DeviceFixture& device,
                           cudaStream_t stream = nullptr) {
  std::vector<double> output_shell(device.output_shell.size());
  std::vector<double> output_dipole(device.output_dipole.size());
  std::vector<double> output_quadrupole(device.output_quadrupole.size());
  StateSnapshot gpu;
  CUDA_CHECK(device.output_shell.copy_to(output_shell.data(), output_shell.size(), stream));
  CUDA_CHECK(device.output_dipole.copy_to(output_dipole.data(), output_dipole.size(), stream));
  CUDA_CHECK(
      device.output_quadrupole.copy_to(output_quadrupole.data(), output_quadrupole.size(), stream));
  CUDA_CHECK(device.snapshot(gpu, stream));
  CHECK(near_arrays(output_shell.data(), cpu.wavefunction.qsh, output_shell.size()));
  CHECK(near_arrays(output_dipole.data(), cpu.wavefunction.dipole, output_dipole.size()));
  CHECK(
      near_arrays(output_quadrupole.data(), cpu.wavefunction.quadrupole, output_quadrupole.size()));
  const std::size_t vectors = static_cast<std::size_t>(cpu.plan.total_vector_elements());
  const std::size_t batch = static_cast<std::size_t>(cpu.plan.batch_size());
  const std::size_t history = vectors * static_cast<std::size_t>(cpu.plan.history_size());
  CHECK(near_arrays(gpu.current.data(), cpu.state.current_inputs, vectors));
  CHECK(near_arrays(gpu.previous.data(), cpu.state.previous_inputs, vectors));
  CHECK(near_arrays(gpu.previous_residual.data(), cpu.state.previous_residuals, vectors));
  /* Parallel CUDA norm reductions and device FMA contraction perturb the
   * ill-conditioned Broyden history. Across batch 1/8/32/128 and six ring
   * updates, the measured output envelope is 1.21e-7 absolute, while the
   * intentionally large normalized u vectors reach 1.58e-5 absolute. Keep a
   * separate evidence-based bound for that internal vector. */
  CHECK(near_arrays(gpu.df.data(), cpu.state.df_history, history));
  CHECK(near_arrays(gpu.u.data(), cpu.state.u_history, history, 2.0e-5, 2.0e-7));
  CHECK(near_arrays(gpu.omega.data(), cpu.state.omega,
                    batch * static_cast<std::size_t>(cpu.plan.history_size())));
  CHECK(near_arrays(gpu.rms.data(), cpu.state.residual_rms, batch, 2.0e-12, 1.0e-11));
  CHECK(near_arrays(gpu.maximum.data(), cpu.state.residual_maximum, batch, 2.0e-12, 1.0e-11));
  CHECK(std::equal(gpu.iterations.begin(), gpu.iterations.end(), cpu.state.iterations));
  CHECK(std::equal(gpu.restarts.begin(), gpu.restarts.end(), cpu.state.restart_counts));
  CHECK(std::equal(gpu.statuses.begin(), gpu.statuses.end(), cpu.state.system_statuses));
  CHECK(std::equal(gpu.initialized.begin(), gpu.initialized.end(), cpu.state.initialized));
  CHECK(std::equal(gpu.residual_converged.begin(), gpu.residual_converged.end(),
                   cpu.state.converged));
  return 0;
}

int test_cpu_parity_for_batch(std::size_t batch_size) {
  CpuFixture cpu;
  std::string error;
  CHECK(make_cpu_fixture(batch_size, cpu, error));
  CHECK(gpuxtb::detail::gfn2::initialize_scc_mixer_state_cpu(cpu.plan, cpu.wavefunction, cpu.state,
                                                             error) == GPUXTB_STATUS_SUCCESS);
  DeviceFixture device;
  CUDA_CHECK(device.setup(cpu));
  CHECK(initialize_device(device) == 0);

  StateSnapshot initialized;
  CUDA_CHECK(device.snapshot(initialized));
  CHECK(near_arrays(initialized.current.data(), cpu.state.current_inputs,
                    initialized.current.size(), 0.0, 0.0));
  CHECK(std::all_of(initialized.initialized.begin(), initialized.initialized.end(),
                    [](std::uint8_t value) { return value == 1u; }));
  for (std::size_t iteration = 0u; iteration < 6u; ++iteration) {
    install_raw_iteration(cpu, iteration);
    CUDA_CHECK(device.install_raw_iteration(cpu, iteration));
    CHECK(gpuxtb::detail::gfn2::mix_scc_broyden_batch_cpu(
              cpu.plan, cpu.wavefunction, cpu.state, cpu.scratch, error) == GPUXTB_STATUS_SUCCESS);
    CUDA_CHECK(device.reset_error());
    CUDA_CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
        device.batch, device.policy, device.activity, device.raw, device.output, device.state,
        device.workspace, device.error.get()));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::uint32_t semantic_error = 99u;
    CUDA_CHECK(device.error.copy_to(&semantic_error, 1u));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2SccMixerDeviceError::kSuccess));
    CHECK(compare_cpu_and_device(cpu, device) == 0);
  }
  return 0;
}

int test_nextafter_convergence_boundaries_match_cpu() {
  CpuFixture probe;
  std::string error;
  CHECK(make_cpu_fixture(1u, probe, error));
  CHECK(gpuxtb::detail::gfn2::initialize_scc_mixer_state_cpu(
            probe.plan, probe.wavefunction, probe.state, error) == GPUXTB_STATUS_SUCCESS);
  install_raw_iteration(probe, 0u);
  const std::int64_t count = dimension(probe, 0u);
  const std::int64_t begin = vector_begin(probe, 0u);
  double residual_square = 0.0;
  double residual_maximum = 0.0;
  for (std::int64_t component = 0; component < count; ++component) {
    const double residual =
        get_field_component(probe, 0u, component) - probe.state.current_inputs[begin + component];
    residual_square += residual * residual;
    residual_maximum = std::max(residual_maximum, std::abs(residual));
  }
  const double residual_rms = std::sqrt(residual_square) / std::sqrt(static_cast<double>(count));
  const double rms_above = std::nextafter(residual_rms, std::numeric_limits<double>::infinity());
  const double rms_below = std::nextafter(residual_rms, 0.0);
  const double maximum_above =
      std::nextafter(residual_maximum, std::numeric_limits<double>::infinity());
  const double maximum_below = std::nextafter(residual_maximum, 0.0);

  struct BoundaryCase {
    double rms_tolerance;
    double maximum_tolerance;
    bool converged;
  };
  const std::array<BoundaryCase, 6> cases{{
      {rms_above, maximum_above, true},
      {residual_rms, maximum_above, false},
      {rms_below, maximum_above, false},
      {rms_above, residual_maximum, false},
      {rms_above, maximum_below, false},
      {rms_below, maximum_below, false},
  }};

  for (const BoundaryCase& boundary : cases) {
    CpuFixture cpu;
    CHECK(make_cpu_fixture(1u, cpu, error, boundary.rms_tolerance, boundary.maximum_tolerance));
    CHECK(gpuxtb::detail::gfn2::initialize_scc_mixer_state_cpu(
              cpu.plan, cpu.wavefunction, cpu.state, error) == GPUXTB_STATUS_SUCCESS);
    DeviceFixture device;
    CUDA_CHECK(device.setup(cpu));
    CHECK(initialize_device(device) == 0);
    install_raw_iteration(cpu, 0u);
    CUDA_CHECK(device.install_raw_iteration(cpu, 0u));
    CHECK(gpuxtb::detail::gfn2::mix_scc_broyden_batch_cpu(
              cpu.plan, cpu.wavefunction, cpu.state, cpu.scratch, error) == GPUXTB_STATUS_SUCCESS);
    CUDA_CHECK(device.reset_error());
    CUDA_CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
        device.batch, device.policy, device.activity, device.raw, device.output, device.state,
        device.workspace, device.error.get()));
    CUDA_CHECK(cudaDeviceSynchronize());
    StateSnapshot gpu;
    CUDA_CHECK(device.snapshot(gpu));
    CHECK(cpu.state.residual_rms[0] == gpu.rms[0]);
    CHECK(cpu.state.residual_maximum[0] == gpu.maximum[0]);
    CHECK((cpu.state.converged[0] == 1u) == boundary.converged);
    CHECK(gpu.residual_converged[0] == cpu.state.converged[0]);
  }
  return 0;
}

bool equal_target_slice(const StateSnapshot& before, const StateSnapshot& after,
                        const CpuFixture& cpu, std::size_t system) {
  const std::size_t begin = static_cast<std::size_t>(vector_begin(cpu, system));
  const std::size_t count = static_cast<std::size_t>(dimension(cpu, system));
  const std::size_t history_begin = begin * static_cast<std::size_t>(kHistorySize);
  const std::size_t history_count = count * static_cast<std::size_t>(kHistorySize);
  const std::size_t omega_begin = system * static_cast<std::size_t>(kHistorySize);
  return std::equal(before.current.begin() + begin, before.current.begin() + begin + count,
                    after.current.begin() + begin) &&
         std::equal(before.previous.begin() + begin, before.previous.begin() + begin + count,
                    after.previous.begin() + begin) &&
         std::equal(before.previous_residual.begin() + begin,
                    before.previous_residual.begin() + begin + count,
                    after.previous_residual.begin() + begin) &&
         std::equal(before.df.begin() + history_begin,
                    before.df.begin() + history_begin + history_count,
                    after.df.begin() + history_begin) &&
         std::equal(before.u.begin() + history_begin,
                    before.u.begin() + history_begin + history_count,
                    after.u.begin() + history_begin) &&
         std::equal(before.omega.begin() + omega_begin,
                    before.omega.begin() + omega_begin + kHistorySize,
                    after.omega.begin() + omega_begin) &&
         before.rms[system] == after.rms[system] &&
         before.maximum[system] == after.maximum[system] &&
         before.iterations[system] == after.iterations[system] &&
         before.restarts[system] == after.restarts[system] &&
         before.initialized[system] == after.initialized[system] &&
         before.residual_converged[system] == after.residual_converged[system];
}

bool equal_complete_system(const StateSnapshot& before, const StateSnapshot& after,
                           const CpuFixture& cpu, std::size_t system) {
  return equal_target_slice(before, after, cpu, system) &&
         before.statuses[system] == after.statuses[system];
}

bool equal_snapshots(const StateSnapshot& first, const StateSnapshot& second) {
  return first.current == second.current && first.previous == second.previous &&
         first.previous_residual == second.previous_residual && first.df == second.df &&
         first.u == second.u && first.omega == second.omega && first.rms == second.rms &&
         first.maximum == second.maximum && first.iterations == second.iterations &&
         first.restarts == second.restarts && first.statuses == second.statuses &&
         first.initialized == second.initialized &&
         first.residual_converged == second.residual_converged;
}

bool equal_multipoles(const MultipoleSnapshot& first, const MultipoleSnapshot& second) {
  return first.shell == second.shell && first.dipole == second.dipole &&
         first.quadrupole == second.quadrupole;
}

bool target_multipoles_have_value(const MultipoleSnapshot& multipoles, const CpuFixture& cpu,
                                  std::size_t system, double expected) {
  const std::int64_t shell_begin = cpu.layout.batch_shell_offsets[system];
  const std::int64_t shell_end = cpu.layout.batch_shell_offsets[system + 1u];
  const std::int64_t atom_begin = cpu.layout.atom_offsets[system];
  const std::int64_t atom_end = cpu.layout.atom_offsets[system + 1u];
  return std::all_of(multipoles.shell.begin() + shell_begin, multipoles.shell.begin() + shell_end,
                     [=](double value) { return value == expected; }) &&
         std::all_of(multipoles.dipole.begin() + atom_begin * 3,
                     multipoles.dipole.begin() + atom_end * 3,
                     [=](double value) { return value == expected; }) &&
         std::all_of(multipoles.quadrupole.begin() + atom_begin * 6,
                     multipoles.quadrupole.begin() + atom_end * 6,
                     [=](double value) { return value == expected; });
}

int test_nonfinite_initialize_is_whole_call_atomic() {
  CpuFixture cpu;
  std::string error;
  CHECK(make_cpu_fixture(8u, cpu, error));
  DeviceFixture device;
  CUDA_CHECK(device.setup(cpu));
  CHECK(initialize_device(device) == 0);
  StateSnapshot before;
  CUDA_CHECK(device.snapshot(before));

  std::vector<double> shell(device.raw_shell.size());
  std::copy_n(cpu.wavefunction.qsh, shell.size(), shell.begin());
  shell[static_cast<std::size_t>(cpu.layout.batch_shell_offsets[4])] =
      std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(device.raw_shell.copy_from(shell.data(), shell.size()));
  CUDA_CHECK(device.reset_error());
  CUDA_CHECK(gpuxtb::detail::cuda::initialize_gfn2_scc_mixer_cuda(
      device.batch, device.policy, device.raw, device.state, device.workspace, device.error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  StateSnapshot after;
  CUDA_CHECK(device.snapshot(after));
  std::uint32_t semantic_error = 0u;
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1u));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error ==
        static_cast<std::uint32_t>(Gfn2SccMixerDeviceError::kNonfiniteInitialMultipole));
  CHECK(equal_snapshots(before, after));
  return 0;
}

int test_topology_and_canonical_sequence_gates() {
  CpuFixture cpu;
  std::string error;
  CHECK(make_cpu_fixture(8u, cpu, error));
  DeviceFixture device;
  CUDA_CHECK(device.setup(cpu));
  CHECK(initialize_device(device) == 0);
  CUDA_CHECK(device.install_raw_iteration(cpu, 0u));

  /* The future composer normalizes system_statuses in gpuxtb_status_t space
   * and this sequence latch separately; the mixer enum scalar is tracing-only
   * because its numeric domain must never be compared with status codes. */
  constexpr double first_sentinel = -901.25;
  CUDA_CHECK(device.fill_output(first_sentinel));
  StateSnapshot before_invalid_offsets;
  MultipoleSnapshot output_before_invalid_offsets;
  CUDA_CHECK(device.snapshot(before_invalid_offsets));
  CUDA_CHECK(device.snapshot_output(output_before_invalid_offsets));
  std::vector<std::int64_t> invalid_offsets = cpu.layout.batch_shell_offsets;
  invalid_offsets[4] = invalid_offsets[3] - 1;
  CUDA_CHECK(device.shell_offsets.copy_from(invalid_offsets.data(), invalid_offsets.size()));
  CUDA_CHECK(device.reset_error());
  CUDA_CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
      device.batch, device.policy, device.activity, device.raw, device.output, device.state,
      device.workspace, device.error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  StateSnapshot after_invalid_offsets;
  MultipoleSnapshot output_after_invalid_offsets;
  CUDA_CHECK(device.snapshot(after_invalid_offsets));
  CUDA_CHECK(device.snapshot_output(output_after_invalid_offsets));
  std::uint32_t semantic_error = 0u;
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1u));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2SccMixerDeviceError::kInvalidOffsets));
  CHECK(equal_snapshots(before_invalid_offsets, after_invalid_offsets));
  CHECK(equal_multipoles(output_before_invalid_offsets, output_after_invalid_offsets));

  CUDA_CHECK(device.shell_offsets.copy_from(cpu.layout.batch_shell_offsets.data(),
                                            cpu.layout.batch_shell_offsets.size()));
  constexpr double second_sentinel = -902.5;
  CUDA_CHECK(device.fill_output(second_sentinel));
  StateSnapshot before_sticky;
  MultipoleSnapshot output_before_sticky;
  CUDA_CHECK(device.snapshot(before_sticky));
  CUDA_CHECK(device.snapshot_output(output_before_sticky));
  const std::uint32_t sticky =
      static_cast<std::uint32_t>(Gfn2SccMixerDeviceError::kNonfiniteRawMultipole);
  CUDA_CHECK(device.set_activity(std::vector<std::uint8_t>(8u, 1u), 0u));
  CUDA_CHECK(device.error.copy_from(&sticky, 1u));
  CUDA_CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
      device.batch, device.policy, device.activity, device.raw, device.output, device.state,
      device.workspace, device.error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  StateSnapshot after_sticky;
  MultipoleSnapshot output_after_sticky;
  CUDA_CHECK(device.snapshot(after_sticky));
  CUDA_CHECK(device.snapshot_output(output_after_sticky));
  semantic_error = 0u;
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1u));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error == sticky);
  CHECK(equal_snapshots(before_sticky, after_sticky));
  CHECK(equal_multipoles(output_before_sticky, output_after_sticky));

  /* A mixer diagnostic is sticky for tracing but is not an execution gate.
   * Reopening only the canonical sequence must advance every requested member. */
  CUDA_CHECK(device.set_all_active());
  StateSnapshot before_reopened;
  CUDA_CHECK(device.snapshot(before_reopened));
  CUDA_CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
      device.batch, device.policy, device.activity, device.raw, device.output, device.state,
      device.workspace, device.error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  StateSnapshot after_reopened;
  CUDA_CHECK(device.snapshot(after_reopened));
  CHECK(after_reopened.iterations[0] == before_reopened.iterations[0] + 1u);
  semantic_error = 0u;
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1u));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error == sticky);
  return 0;
}

int test_active_topology_projection_and_normalization() {
  for (bool shift_first_begin : {true, false}) {
    CpuFixture cpu;
    std::string error;
    CHECK(make_cpu_fixture(8u, cpu, error));
    DeviceFixture device;
    CUDA_CHECK(device.setup(cpu));
    CHECK(initialize_device(device) == 0);
    CUDA_CHECK(device.install_raw_iteration(cpu, 0u));

    constexpr std::size_t stale = 3u;
    const gpuxtb_status_t stale_status = GPUXTB_STATUS_SCC_NOT_CONVERGED;
    CUDA_CHECK(cudaMemcpy(device.statuses.get() + stale, &stale_status, sizeof(stale_status),
                          cudaMemcpyHostToDevice));
    constexpr double sentinel = -906.25;
    CUDA_CHECK(device.fill_output(sentinel));
    StateSnapshot before;
    MultipoleSnapshot output_before;
    CUDA_CHECK(device.snapshot(before));
    CUDA_CHECK(device.snapshot_output(output_before));

    std::vector<std::int64_t> shell_offsets = cpu.layout.batch_shell_offsets;
    if (shift_first_begin) {
      /* Keep both leading active slices locally valid and nonoverlapping while
       * shifting system zero away from the required packed origin. */
      CHECK(shell_offsets[2] > 2);
      shell_offsets[0] = 1;
      shell_offsets[1] = 2;
    } else {
      /* Keep the last slice nonempty while shifting its active endpoint away
       * from the complete packed shell extent. */
      CHECK(shell_offsets[7] < shell_offsets[8] - 1);
      --shell_offsets[8];
    }
    CUDA_CHECK(device.shell_offsets.copy_from(shell_offsets.data(), shell_offsets.size()));
    CUDA_CHECK(device.reset_error());
    CUDA_CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
        device.batch, device.policy, device.activity, device.raw, device.output, device.state,
        device.workspace, device.error.get()));
    CUDA_CHECK(cudaDeviceSynchronize());

    StateSnapshot after;
    MultipoleSnapshot output_after;
    CUDA_CHECK(device.snapshot(after));
    CUDA_CHECK(device.snapshot_output(output_after));
    for (std::size_t system = 0u; system < 8u; ++system) {
      CHECK(equal_target_slice(before, after, cpu, system));
      CHECK(after.statuses[system] == GPUXTB_STATUS_SUCCESS);
    }
    CHECK(equal_multipoles(output_before, output_after));
    std::uint32_t mixer_error = 0u;
    std::uint32_t stage_sequence = 1u;
    CUDA_CHECK(device.error.copy_to(&mixer_error, 1u));
    CUDA_CHECK(device.sequence.copy_to(&stage_sequence, 1u));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(mixer_error == static_cast<std::uint32_t>(Gfn2SccMixerDeviceError::kInvalidOffsets));
    CHECK(stage_sequence == 0u);

    DeviceBuffer<gpuxtb_status_t> pending_statuses;
    DeviceBuffer<std::uint64_t> system_failure_records;
    DeviceBuffer<std::uint64_t> plan_failure_record;
    CUDA_CHECK(pending_statuses.allocate(8u));
    CUDA_CHECK(system_failure_records.allocate(8u));
    CUDA_CHECK(plan_failure_record.allocate(1u));
    CUDA_CHECK(pending_statuses.fill_zero());
    CUDA_CHECK(system_failure_records.fill_zero());
    CUDA_CHECK(plan_failure_record.fill_zero());
    Gfn2SccIterationDeviceLedger ledger = {device.active_mask.get(),
                                           pending_statuses.get(),
                                           system_failure_records.get(),
                                           plan_failure_record.get(),
                                           device.canonical_sequence.get(),
                                           8,
                                           1,
                                           kPlanToken};
    Gfn2SccStageDeviceReport report;
    report.stage = Gfn2SccStageId::kMixer;
    report.system_code_format = Gfn2SccStageCodeFormat::kGpuxtbStatus;
    report.system_codes = device.statuses.get();
    report.system_code_elements = 8;
    report.device_error = nullptr;
    report.device_error_elements = 0;
    report.stage_sequence_active = device.sequence.get();
    report.stage_sequence_elements = 1;
    report.peer_error_mask = std::uint64_t{1}
                             << static_cast<std::uint32_t>(GPUXTB_STATUS_INTERNAL_ERROR);
    report.peer_failure_status = GPUXTB_STATUS_INTERNAL_ERROR;
    report.plan_token = kPlanToken;
    CUDA_CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(report, ledger));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::uint64_t plan_failure = 0u;
    std::vector<std::uint64_t> member_failures(8u, 1u);
    CUDA_CHECK(plan_failure_record.copy_to(&plan_failure, 1u));
    CUDA_CHECK(system_failure_records.copy_to(member_failures.data(), member_failures.size()));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(gpuxtb::detail::cuda::gfn2_scc_failure_stage(plan_failure) == Gfn2SccStageId::kMixer);
    CHECK(gpuxtb::detail::cuda::gfn2_scc_failure_code(plan_failure) ==
          gpuxtb::detail::cuda::kGfn2SccStageSequenceClosedCode);
    CHECK(std::all_of(member_failures.begin(), member_failures.end(),
                      [](std::uint64_t value) { return value == 0u; }));
  }

  CpuFixture cpu;
  std::string error;
  CHECK(make_cpu_fixture(8u, cpu, error));
  DeviceFixture device;
  CUDA_CHECK(device.setup(cpu));
  CHECK(initialize_device(device) == 0);
  CUDA_CHECK(device.install_raw_iteration(cpu, 0u));
  std::vector<std::uint8_t> active(8u, 0u);
  active[0] = 1u;
  active[2] = 1u;
  CUDA_CHECK(device.set_activity(active, 1u));
  std::vector<std::int64_t> overlapping_offsets = cpu.layout.batch_shell_offsets;
  overlapping_offsets[2] = overlapping_offsets[1] - 1;
  CUDA_CHECK(
      device.shell_offsets.copy_from(overlapping_offsets.data(), overlapping_offsets.size()));
  constexpr double sentinel = -907.5;
  CUDA_CHECK(device.fill_output(sentinel));
  StateSnapshot before;
  MultipoleSnapshot output_before;
  CUDA_CHECK(device.snapshot(before));
  CUDA_CHECK(device.snapshot_output(output_before));
  CUDA_CHECK(device.reset_error());
  CUDA_CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
      device.batch, device.policy, device.activity, device.raw, device.output, device.state,
      device.workspace, device.error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  StateSnapshot after;
  MultipoleSnapshot output_after;
  CUDA_CHECK(device.snapshot(after));
  CUDA_CHECK(device.snapshot_output(output_after));
  CHECK(equal_snapshots(before, after));
  CHECK(equal_multipoles(output_before, output_after));
  std::uint32_t mixer_error = 0u;
  std::uint32_t stage_sequence = 1u;
  CUDA_CHECK(device.error.copy_to(&mixer_error, 1u));
  CUDA_CHECK(device.sequence.copy_to(&stage_sequence, 1u));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(mixer_error == static_cast<std::uint32_t>(Gfn2SccMixerDeviceError::kInvalidOffsets));
  CHECK(stage_sequence == 0u);
  return 0;
}

int test_iteration_overflow_isolated_from_healthy_peers() {
  CpuFixture cpu;
  std::string error;
  CHECK(make_cpu_fixture(8u, cpu, error));
  DeviceFixture device;
  CUDA_CHECK(device.setup(cpu));
  CHECK(initialize_device(device) == 0);
  CUDA_CHECK(device.install_raw_iteration(cpu, 0u));
  constexpr std::size_t failed = 3u;
  const std::uint64_t maximum_iteration = std::numeric_limits<std::uint64_t>::max();
  CUDA_CHECK(cudaMemcpy(device.iterations.get() + failed, &maximum_iteration,
                        sizeof(maximum_iteration), cudaMemcpyHostToDevice));
  constexpr double sentinel = -903.75;
  CUDA_CHECK(device.fill_output(sentinel));
  StateSnapshot before;
  CUDA_CHECK(device.snapshot(before));
  CUDA_CHECK(device.reset_error());
  CUDA_CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
      device.batch, device.policy, device.activity, device.raw, device.output, device.state,
      device.workspace, device.error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  StateSnapshot after;
  MultipoleSnapshot output;
  CUDA_CHECK(device.snapshot(after));
  CUDA_CHECK(device.snapshot_output(output));
  std::uint32_t semantic_error = 0u;
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1u));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2SccMixerDeviceError::kIterationOverflow));
  CHECK(equal_target_slice(before, after, cpu, failed));
  CHECK(after.statuses[failed] == GPUXTB_STATUS_INTERNAL_ERROR);
  CHECK(target_multipoles_have_value(output, cpu, failed, sentinel));
  CHECK(after.iterations[0] == before.iterations[0] + 1u);
  CHECK(after.statuses[0] == GPUXTB_STATUS_SUCCESS);
  return 0;
}

int test_corrupted_active_history_isolated_from_healthy_peers() {
  for (bool corrupt_omega : {false, true}) {
    CpuFixture cpu;
    std::string error;
    CHECK(make_cpu_fixture(8u, cpu, error));
    DeviceFixture device;
    CUDA_CHECK(device.setup(cpu));
    CHECK(initialize_device(device) == 0);
    for (std::size_t iteration = 0u; iteration < 2u; ++iteration) {
      CUDA_CHECK(device.install_raw_iteration(cpu, iteration));
      CUDA_CHECK(device.reset_error());
      CUDA_CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
          device.batch, device.policy, device.activity, device.raw, device.output, device.state,
          device.workspace, device.error.get()));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    constexpr std::size_t failed = 3u;
    const double corrupt_value = std::numeric_limits<double>::max();
    if (corrupt_omega) {
      CUDA_CHECK(cudaMemcpy(device.omega.get() + failed * static_cast<std::size_t>(kHistorySize),
                            &corrupt_value, sizeof(corrupt_value), cudaMemcpyHostToDevice));
    } else {
      const std::size_t history_begin = static_cast<std::size_t>(vector_begin(cpu, failed)) *
                                        static_cast<std::size_t>(kHistorySize);
      CUDA_CHECK(cudaMemcpy(device.df.get() + history_begin, &corrupt_value, sizeof(corrupt_value),
                            cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(device.install_raw_iteration(cpu, 2u));
    constexpr double sentinel = -905.0;
    CUDA_CHECK(device.fill_output(sentinel));
    StateSnapshot before;
    CUDA_CHECK(device.snapshot(before));
    CUDA_CHECK(device.reset_error());
    CUDA_CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
        device.batch, device.policy, device.activity, device.raw, device.output, device.state,
        device.workspace, device.error.get()));
    CUDA_CHECK(cudaDeviceSynchronize());
    StateSnapshot after;
    MultipoleSnapshot output;
    CUDA_CHECK(device.snapshot(after));
    CUDA_CHECK(device.snapshot_output(output));
    std::uint32_t semantic_error = 0u;
    CUDA_CHECK(device.error.copy_to(&semantic_error, 1u));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(semantic_error != static_cast<std::uint32_t>(Gfn2SccMixerDeviceError::kSuccess));
    CHECK(equal_target_slice(before, after, cpu, failed));
    CHECK(after.statuses[failed] == GPUXTB_STATUS_INTERNAL_ERROR);
    CHECK(target_multipoles_have_value(output, cpu, failed, sentinel));
    CHECK(after.iterations[0] == before.iterations[0] + 1u);
    CHECK(after.statuses[0] == GPUXTB_STATUS_SUCCESS);
  }
  return 0;
}

int test_peer_failure_and_canonical_inactive_members() {
  CpuFixture cpu;
  std::string error;
  CHECK(make_cpu_fixture(8u, cpu, error));
  CHECK(gpuxtb::detail::gfn2::initialize_scc_mixer_state_cpu(cpu.plan, cpu.wavefunction, cpu.state,
                                                             error) == GPUXTB_STATUS_SUCCESS);
  DeviceFixture device;
  CUDA_CHECK(device.setup(cpu));
  CHECK(initialize_device(device) == 0);
  install_raw_iteration(cpu, 0u);
  CUDA_CHECK(device.copy_raw(cpu));
  CUDA_CHECK(device.reset_error());
  CUDA_CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
      device.batch, device.policy, device.activity, device.raw, device.output, device.state,
      device.workspace, device.error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());

  StateSnapshot before;
  CUDA_CHECK(device.snapshot(before));
  constexpr std::size_t failed = 3u;
  constexpr std::size_t stale_mixer_status = 4u;
  constexpr std::size_t terminal = 5u;
  constexpr std::size_t residual_pass = 6u;
  constexpr std::size_t maximum_iteration = 7u;
  std::vector<double> raw_shell(device.raw_shell.size());
  std::vector<double> raw_dipole(device.raw_dipole.size());
  std::vector<double> raw_quadrupole(device.raw_quadrupole.size());
  for (std::size_t index = 0u; index < raw_shell.size(); ++index) {
    raw_shell[index] = 0.01 + 0.0001 * static_cast<double>(index);
  }
  for (std::size_t index = 0u; index < raw_dipole.size(); ++index) {
    raw_dipole[index] = -0.02 + 0.0002 * static_cast<double>(index);
  }
  for (std::size_t index = 0u; index < raw_quadrupole.size(); ++index) {
    raw_quadrupole[index] = 0.03 - 0.00007 * static_cast<double>(index);
  }
  raw_shell[static_cast<std::size_t>(cpu.layout.batch_shell_offsets[failed])] =
      std::numeric_limits<double>::quiet_NaN();
  raw_shell[static_cast<std::size_t>(cpu.layout.batch_shell_offsets[terminal])] =
      std::numeric_limits<double>::quiet_NaN();
  raw_shell[static_cast<std::size_t>(cpu.layout.batch_shell_offsets[residual_pass])] =
      std::numeric_limits<double>::quiet_NaN();
  raw_shell[static_cast<std::size_t>(cpu.layout.batch_shell_offsets[maximum_iteration])] =
      std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(device.raw_shell.copy_from(raw_shell.data(), raw_shell.size()));
  CUDA_CHECK(device.raw_dipole.copy_from(raw_dipole.data(), raw_dipole.size()));
  CUDA_CHECK(device.raw_quadrupole.copy_from(raw_quadrupole.data(), raw_quadrupole.size()));
  const gpuxtb_status_t terminal_status = GPUXTB_STATUS_SCC_NOT_CONVERGED;
  const std::uint8_t poisoned_residual_diagnostic = 2u;
  const std::uint8_t residual_pass_flag = 1u;
  const std::uint64_t exhausted_iterations = std::numeric_limits<std::uint64_t>::max();
  CUDA_CHECK(cudaMemcpy(device.statuses.get() + terminal, &terminal_status, sizeof(terminal_status),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device.statuses.get() + stale_mixer_status, &terminal_status,
                        sizeof(terminal_status), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device.residual_converged.get() + stale_mixer_status,
                        &poisoned_residual_diagnostic, sizeof(poisoned_residual_diagnostic),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device.residual_converged.get() + residual_pass, &residual_pass_flag,
                        sizeof(residual_pass_flag), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device.iterations.get() + maximum_iteration, &exhausted_iterations,
                        sizeof(exhausted_iterations), cudaMemcpyHostToDevice));
  before.statuses[terminal] = terminal_status;
  before.statuses[stale_mixer_status] = terminal_status;
  before.residual_converged[stale_mixer_status] = poisoned_residual_diagnostic;
  before.residual_converged[residual_pass] = residual_pass_flag;
  before.iterations[maximum_iteration] = exhausted_iterations;
  std::vector<std::uint8_t> active(8u, 1u);
  active[terminal] = 0u;
  active[residual_pass] = 0u;
  active[maximum_iteration] = 0u;
  CUDA_CHECK(device.set_activity(active, 1u));
  std::vector<std::int64_t> inactive_poisoned_offsets = cpu.layout.batch_shell_offsets;
  inactive_poisoned_offsets[residual_pass] = -17;
  CUDA_CHECK(device.shell_offsets.copy_from(inactive_poisoned_offsets.data(),
                                            inactive_poisoned_offsets.size()));
  constexpr double sentinel = -1234.5;
  std::vector<double> sent_shell(device.output_shell.size(), sentinel);
  std::vector<double> sent_dipole(device.output_dipole.size(), sentinel);
  std::vector<double> sent_quadrupole(device.output_quadrupole.size(), sentinel);
  CUDA_CHECK(device.output_shell.copy_from(sent_shell.data(), sent_shell.size()));
  CUDA_CHECK(device.output_dipole.copy_from(sent_dipole.data(), sent_dipole.size()));
  CUDA_CHECK(device.output_quadrupole.copy_from(sent_quadrupole.data(), sent_quadrupole.size()));

  CUDA_CHECK(device.reset_error());
  CUDA_CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
      device.batch, device.policy, device.activity, device.raw, device.output, device.state,
      device.workspace, device.error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::uint32_t semantic_error = 0u;
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1u));
  StateSnapshot after;
  CUDA_CHECK(device.snapshot(after));
  CHECK(semantic_error ==
        static_cast<std::uint32_t>(Gfn2SccMixerDeviceError::kNonfiniteRawMultipole));
  CHECK(equal_target_slice(before, after, cpu, failed));
  CHECK(after.statuses[failed] == GPUXTB_STATUS_INTERNAL_ERROR);
  CHECK(equal_target_slice(before, after, cpu, terminal));
  CHECK(after.statuses[terminal] == terminal_status);
  CHECK(equal_complete_system(before, after, cpu, residual_pass));
  CHECK(equal_complete_system(before, after, cpu, maximum_iteration));
  CHECK(after.iterations[stale_mixer_status] == before.iterations[stale_mixer_status] + 1u);
  CHECK(after.statuses[stale_mixer_status] == GPUXTB_STATUS_SUCCESS);
  CHECK(after.residual_converged[stale_mixer_status] == 0u);
  CHECK(after.iterations[0] == before.iterations[0] + 1u);
  std::vector<double> output_shell(device.output_shell.size());
  std::vector<double> output_dipole(device.output_dipole.size());
  std::vector<double> output_quadrupole(device.output_quadrupole.size());
  CUDA_CHECK(device.output_shell.copy_to(output_shell.data(), output_shell.size()));
  CUDA_CHECK(device.output_dipole.copy_to(output_dipole.data(), output_dipole.size()));
  CUDA_CHECK(device.output_quadrupole.copy_to(output_quadrupole.data(), output_quadrupole.size()));
  CUDA_CHECK(cudaDeviceSynchronize());
  for (std::size_t skipped : {failed, terminal, residual_pass, maximum_iteration}) {
    const std::int64_t shell_begin = cpu.layout.batch_shell_offsets[skipped];
    const std::int64_t shell_end = cpu.layout.batch_shell_offsets[skipped + 1u];
    const std::int64_t atom_begin = cpu.layout.atom_offsets[skipped];
    const std::int64_t atom_end = cpu.layout.atom_offsets[skipped + 1u];
    CHECK(std::all_of(output_shell.begin() + shell_begin, output_shell.begin() + shell_end,
                      [=](double value) { return value == sentinel; }));
    CHECK(std::all_of(output_dipole.begin() + atom_begin * 3, output_dipole.begin() + atom_end * 3,
                      [=](double value) { return value == sentinel; }));
    CHECK(std::all_of(output_quadrupole.begin() + atom_begin * 6,
                      output_quadrupole.begin() + atom_end * 6,
                      [=](double value) { return value == sentinel; }));
  }
  return 0;
}

int test_restart_atomicity_and_private_mixed_semantics() {
  CpuFixture cpu;
  std::string error;
  CHECK(make_cpu_fixture(8u, cpu, error));
  CHECK(gpuxtb::detail::gfn2::initialize_scc_mixer_state_cpu(cpu.plan, cpu.wavefunction, cpu.state,
                                                             error) == GPUXTB_STATUS_SUCCESS);
  DeviceFixture device;
  CUDA_CHECK(device.setup(cpu));
  CHECK(initialize_device(device) == 0);
  install_raw_iteration(cpu, 0u);
  CUDA_CHECK(device.copy_raw(cpu));
  CUDA_CHECK(device.reset_error());
  CUDA_CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
      device.batch, device.policy, device.activity, device.raw, device.output, device.state,
      device.workspace, device.error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());

  constexpr std::size_t target = 2u;
  std::vector<double> public_shell(device.raw_shell.size(), 0.125);
  std::vector<double> public_dipole(device.raw_dipole.size(), -0.25);
  std::vector<double> public_quadrupole(device.raw_quadrupole.size(), 0.375);
  CUDA_CHECK(device.raw_shell.copy_from(public_shell.data(), public_shell.size()));
  CUDA_CHECK(device.raw_dipole.copy_from(public_dipole.data(), public_dipole.size()));
  CUDA_CHECK(device.raw_quadrupole.copy_from(public_quadrupole.data(), public_quadrupole.size()));
  StateSnapshot before;
  CUDA_CHECK(device.snapshot(before));
  CUDA_CHECK(device.reset_error());
  CUDA_CHECK(gpuxtb::detail::cuda::restart_gfn2_scc_mixer_system_cuda(
      device.batch, device.policy, static_cast<std::int64_t>(target), device.raw, device.state,
      device.workspace, device.error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  StateSnapshot restarted;
  CUDA_CHECK(device.snapshot(restarted));
  const std::size_t begin = static_cast<std::size_t>(vector_begin(cpu, target));
  const std::size_t count = static_cast<std::size_t>(dimension(cpu, target));
  CHECK(restarted.iterations[target] == 0u);
  CHECK(restarted.restarts[target] == before.restarts[target] + 1u);
  CHECK(restarted.statuses[target] == GPUXTB_STATUS_SUCCESS);
  CHECK(restarted.residual_converged[target] == 0u);
  CHECK(std::all_of(restarted.previous.begin() + begin, restarted.previous.begin() + begin + count,
                    [](double value) { return value == 0.0; }));
  CHECK(std::all_of(restarted.previous_residual.begin() + begin,
                    restarted.previous_residual.begin() + begin + count,
                    [](double value) { return value == 0.0; }));
  CHECK(restarted.current[begin] == 0.125);
  CHECK(restarted
            .current[begin + static_cast<std::size_t>(cpu.layout.batch_shell_offsets[target + 1u] -
                                                      cpu.layout.batch_shell_offsets[target])] ==
        -0.25);
  for (std::size_t peer = 0u; peer < static_cast<std::size_t>(cpu.layout.batch_size); ++peer) {
    if (peer != target) {
      CHECK(equal_complete_system(before, restarted, cpu, peer));
    }
  }

  StateSnapshot before_failed_restart = restarted;
  public_shell[static_cast<std::size_t>(cpu.layout.batch_shell_offsets[target])] =
      std::numeric_limits<double>::infinity();
  CUDA_CHECK(device.raw_shell.copy_from(public_shell.data(), public_shell.size()));
  CUDA_CHECK(device.reset_error());
  CUDA_CHECK(gpuxtb::detail::cuda::restart_gfn2_scc_mixer_system_cuda(
      device.batch, device.policy, static_cast<std::int64_t>(target), device.raw, device.state,
      device.workspace, device.error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  StateSnapshot after_failed_restart;
  CUDA_CHECK(device.snapshot(after_failed_restart));
  for (std::size_t system = 0u; system < static_cast<std::size_t>(cpu.layout.batch_size);
       ++system) {
    CHECK(equal_complete_system(before_failed_restart, after_failed_restart, cpu, system));
  }

  public_shell[static_cast<std::size_t>(cpu.layout.batch_shell_offsets[target])] = 0.125;
  CUDA_CHECK(device.raw_shell.copy_from(public_shell.data(), public_shell.size()));
  const std::uint64_t maximum_restart = std::numeric_limits<std::uint64_t>::max();
  CUDA_CHECK(cudaMemcpy(device.restarts.get() + target, &maximum_restart, sizeof(maximum_restart),
                        cudaMemcpyHostToDevice));
  StateSnapshot before_overflow_restart;
  CUDA_CHECK(device.snapshot(before_overflow_restart));
  CUDA_CHECK(device.reset_error());
  CUDA_CHECK(gpuxtb::detail::cuda::restart_gfn2_scc_mixer_system_cuda(
      device.batch, device.policy, static_cast<std::int64_t>(target), device.raw, device.state,
      device.workspace, device.error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  StateSnapshot after_overflow_restart;
  CUDA_CHECK(device.snapshot(after_overflow_restart));
  std::uint32_t restart_error = 0u;
  CUDA_CHECK(device.error.copy_to(&restart_error, 1u));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(restart_error == static_cast<std::uint32_t>(Gfn2SccMixerDeviceError::kRestartOverflow));
  for (std::size_t system = 0u; system < static_cast<std::size_t>(cpu.layout.batch_size);
       ++system) {
    CHECK(equal_complete_system(before_overflow_restart, after_overflow_restart, cpu, system));
  }

  /* Passing residual thresholds is diagnostic only. The composer may still
   * reject its energy criterion, so canonical activity must request and run
   * the following Broyden transition. */
  Gfn2SccMixerDevicePolicy loose = device.policy;
  loose.rms_tolerance = 1.0e10;
  loose.maximum_tolerance = 1.0e10;
  public_shell.assign(public_shell.size(), 0.5);
  public_dipole.assign(public_dipole.size(), -0.4);
  public_quadrupole.assign(public_quadrupole.size(), 0.3);
  CUDA_CHECK(device.raw_shell.copy_from(public_shell.data(), public_shell.size()));
  CUDA_CHECK(device.raw_dipole.copy_from(public_dipole.data(), public_dipole.size()));
  CUDA_CHECK(device.raw_quadrupole.copy_from(public_quadrupole.data(), public_quadrupole.size()));
  CUDA_CHECK(device.reset_error());
  CUDA_CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
      device.batch, loose, device.activity, device.raw, device.output, device.state,
      device.workspace, device.error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  StateSnapshot mixed;
  CUDA_CHECK(device.snapshot(mixed));
  std::vector<double> output_shell(device.output_shell.size());
  CUDA_CHECK(device.output_shell.copy_to(output_shell.data(), output_shell.size()));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(mixed.residual_converged[target] == 1u);
  const std::size_t target_shell = static_cast<std::size_t>(cpu.layout.batch_shell_offsets[target]);
  CHECK(output_shell[target_shell] == mixed.current[begin]);
  CHECK(output_shell[target_shell] != public_shell[target_shell]);
  CUDA_CHECK(device.reset_error());
  CUDA_CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
      device.batch, loose, device.activity, device.raw, device.output, device.state,
      device.workspace, device.error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  StateSnapshot after_energy_fail;
  CUDA_CHECK(device.snapshot(after_energy_fail));
  CHECK(after_energy_fail.iterations[target] == mixed.iterations[target] + 1u);
  CHECK(after_energy_fail.previous[begin] == mixed.current[begin]);
  return 0;
}

int test_alias_matches_distinct_ragged_batch() {
  CpuFixture cpu;
  std::string error;
  CHECK(make_cpu_fixture(8u, cpu, error));
  DeviceFixture distinct;
  DeviceFixture aliased;
  CUDA_CHECK(distinct.setup(cpu));
  CUDA_CHECK(aliased.setup(cpu));
  CHECK(initialize_device(distinct) == 0);
  CHECK(initialize_device(aliased) == 0);
  CUDA_CHECK(distinct.install_raw_iteration(cpu, 0u));
  CUDA_CHECK(aliased.install_raw_iteration(cpu, 0u));

  Gfn2SccDeviceMultipoles in_place = {aliased.raw_shell.get(),
                                      aliased.raw.shell_elements,
                                      aliased.raw_dipole.get(),
                                      aliased.raw.dipole_elements,
                                      aliased.raw_quadrupole.get(),
                                      aliased.raw.quadrupole_elements,
                                      kPlanToken};
  CUDA_CHECK(distinct.reset_error());
  CUDA_CHECK(aliased.reset_error());
  CUDA_CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
      distinct.batch, distinct.policy, distinct.activity, distinct.raw, distinct.output,
      distinct.state, distinct.workspace, distinct.error.get()));
  CUDA_CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
      aliased.batch, aliased.policy, aliased.activity, aliased.raw, in_place, aliased.state,
      aliased.workspace, aliased.error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());

  StateSnapshot distinct_state;
  StateSnapshot aliased_state;
  MultipoleSnapshot distinct_output;
  MultipoleSnapshot aliased_output;
  CUDA_CHECK(distinct.snapshot(distinct_state));
  CUDA_CHECK(aliased.snapshot(aliased_state));
  CUDA_CHECK(distinct.snapshot_output(distinct_output));
  CUDA_CHECK(aliased.snapshot_raw(aliased_output));
  CHECK(equal_snapshots(distinct_state, aliased_state));
  CHECK(equal_multipoles(distinct_output, aliased_output));
  return 0;
}

int test_custom_stream_and_graph_replay() {
  CpuFixture cpu;
  std::string error;
  CHECK(make_cpu_fixture(8u, cpu, error));
  CHECK(gpuxtb::detail::gfn2::initialize_scc_mixer_state_cpu(cpu.plan, cpu.wavefunction, cpu.state,
                                                             error) == GPUXTB_STATUS_SUCCESS);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.setup(cpu, stream));
  CHECK(initialize_device(device, stream) == 0);
  install_raw_iteration(cpu, 0u);
  CUDA_CHECK(device.copy_raw(cpu, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CUDA_CHECK(device.reset_error(stream));
  CUDA_CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
      device.batch, device.policy, device.activity, device.raw, device.output, device.state,
      device.workspace, device.error.get(), stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  StateSnapshot after_first_replay;
  CUDA_CHECK(device.snapshot(after_first_replay, stream));
  CUDA_CHECK(device.install_raw_iteration(cpu, 1u, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  StateSnapshot after;
  CUDA_CHECK(device.snapshot(after, stream));
  CHECK(std::all_of(after.iterations.begin(), after.iterations.end(),
                    [](std::uint64_t value) { return value == 2u; }));
  CHECK(after.current != after_first_replay.current);
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(device.error.copy_to(&semantic_error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2SccMixerDeviceError::kSuccess));
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_mixed_spin_full_vector_batches() {
  for (const std::size_t batch_size : {1u, 8u, 32u, 128u}) {
    CpuFixture cpu;
    std::string error;
    CHECK(make_cpu_fixture(batch_size, cpu, error));
    DeviceFixture device;
    CUDA_CHECK(device.setup(cpu, nullptr, true));

    MultipoleSnapshot initial;
    CUDA_CHECK(device.snapshot_raw(initial));
    CHECK(initialize_device(device) == 0);

    MultipoleSnapshot raw = initial;
    for (std::size_t index = 0u; index < raw.shell.size(); ++index) {
      raw.shell[index] += 0.00011 * static_cast<double>(index + 1u);
    }
    for (std::size_t index = 0u; index < raw.dipole.size(); ++index) {
      raw.dipole[index] -= 0.00007 * static_cast<double>(index + 1u);
    }
    for (std::size_t index = 0u; index < raw.quadrupole.size(); ++index) {
      raw.quadrupole[index] += 0.00003 * static_cast<double>(index + 1u);
    }
    CUDA_CHECK(device.raw_shell.copy_from(raw.shell.data(), raw.shell.size()));
    CUDA_CHECK(device.raw_dipole.copy_from(raw.dipole.data(), raw.dipole.size()));
    CUDA_CHECK(device.raw_quadrupole.copy_from(raw.quadrupole.data(), raw.quadrupole.size()));
    CUDA_CHECK(device.fill_output(-911.0));
    CUDA_CHECK(device.reset_error());
    CUDA_CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
        device.batch, device.layout, device.policy, device.activity, device.raw, device.output,
        device.state, device.workspace, device.error.get()));

    MultipoleSnapshot output;
    StateSnapshot state;
    CUDA_CHECK(device.snapshot_output(output));
    CUDA_CHECK(device.snapshot(state));
    for (std::size_t index = 0u; index < output.shell.size(); ++index) {
      CHECK(near(
          output.shell[index],
          initial.shell[index] + device.policy.damping * (raw.shell[index] - initial.shell[index]),
          1.0e-15, 1.0e-15));
    }
    for (std::size_t index = 0u; index < output.dipole.size(); ++index) {
      CHECK(near(output.dipole[index],
                 initial.dipole[index] +
                     device.policy.damping * (raw.dipole[index] - initial.dipole[index]),
                 1.0e-15, 1.0e-15));
    }
    for (std::size_t index = 0u; index < output.quadrupole.size(); ++index) {
      CHECK(near(output.quadrupole[index],
                 initial.quadrupole[index] +
                     device.policy.damping * (raw.quadrupole[index] - initial.quadrupole[index]),
                 1.0e-15, 1.0e-15));
    }
    CHECK(std::all_of(state.iterations.begin(), state.iterations.end(),
                      [](std::uint64_t value) { return value == 1u; }));
    CHECK(std::all_of(state.statuses.begin(), state.statuses.end(),
                      [](gpuxtb_status_t value) { return value == GPUXTB_STATUS_SUCCESS; }));
  }
  return 0;
}

int test_host_validation() {
  CpuFixture cpu;
  std::string error;
  CHECK(make_cpu_fixture(1u, cpu, error));
  DeviceFixture device;
  CUDA_CHECK(device.setup(cpu));
  Gfn2SccMixerDevicePolicy invalid = device.policy;
  invalid.history_size = 0;
  CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
            device.batch, invalid, device.activity, device.raw, device.output, device.state,
            device.workspace, device.error.get()) == cudaErrorInvalidValue);
  Gfn2SccDeviceBatch wrong_token = device.batch;
  wrong_token.plan_token ^= 1u;
  CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
            wrong_token, device.policy, device.activity, device.raw, device.output, device.state,
            device.workspace, device.error.get()) == cudaErrorInvalidValue);
  Gfn2SccIterationDeviceActivity wrong_activity = device.activity;
  wrong_activity.plan_token ^= 1u;
  CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
            device.batch, device.policy, wrong_activity, device.raw, device.output, device.state,
            device.workspace, device.error.get()) == cudaErrorInvalidValue);
  wrong_activity = device.activity;
  wrong_activity.batch_elements = 0;
  CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
            device.batch, device.policy, wrong_activity, device.raw, device.output, device.state,
            device.workspace, device.error.get()) == cudaErrorInvalidValue);
  Gfn2SccDeviceMultipoles aliased = device.output;
  aliased.shell_charges = device.state.current_inputs;
  CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
            device.batch, device.policy, device.activity, device.raw, aliased, device.state,
            device.workspace, device.error.get()) == cudaErrorInvalidValue);
  Gfn2SccDeviceMultipoles in_place = {device.raw_shell.get(),
                                      device.raw.shell_elements,
                                      device.raw_dipole.get(),
                                      device.raw.dipole_elements,
                                      device.raw_quadrupole.get(),
                                      device.raw.quadrupole_elements,
                                      kPlanToken};
  CHECK(initialize_device(device) == 0);
  CUDA_CHECK(device.reset_error());
  CUDA_CHECK(gpuxtb::detail::cuda::mix_gfn2_scc_broyden_cuda(
      device.batch, device.policy, device.activity, device.raw, in_place, device.state,
      device.workspace, device.error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return 77;
  }
  for (std::size_t batch : {1u, 8u, 32u, 128u}) {
    const int status = test_cpu_parity_for_batch(batch);
    if (status != 0) {
      return status;
    }
  }
  if (const int status = test_nextafter_convergence_boundaries_match_cpu(); status != 0) {
    return status;
  }
  if (const int status = test_nonfinite_initialize_is_whole_call_atomic(); status != 0) {
    return status;
  }
  if (const int status = test_topology_and_canonical_sequence_gates(); status != 0) {
    return status;
  }
  if (const int status = test_active_topology_projection_and_normalization(); status != 0) {
    return status;
  }
  if (const int status = test_iteration_overflow_isolated_from_healthy_peers(); status != 0) {
    return status;
  }
  if (const int status = test_corrupted_active_history_isolated_from_healthy_peers(); status != 0) {
    return status;
  }
  if (const int status = test_peer_failure_and_canonical_inactive_members(); status != 0) {
    return status;
  }
  if (const int status = test_restart_atomicity_and_private_mixed_semantics(); status != 0) {
    return status;
  }
  if (const int status = test_alias_matches_distinct_ragged_batch(); status != 0) {
    return status;
  }
  if (const int status = test_custom_stream_and_graph_replay(); status != 0) {
    return status;
  }
  if (const int status = test_mixed_spin_full_vector_batches(); status != 0) {
    return status;
  }
  return test_host_validation();
}
