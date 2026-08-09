#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_spin.cuh"
#include "model/gfn2/spin.hpp"

namespace {

using xtbloom::detail::cuda::evaluate_gfn2_spin_polarization_cuda;
using xtbloom::detail::cuda::Gfn2SccIterationDeviceActivity;
using xtbloom::detail::cuda::Gfn2SpinDeviceBatch;
using xtbloom::detail::cuda::Gfn2SpinDeviceError;
using xtbloom::detail::cuda::Gfn2SpinDeviceInput;
using xtbloom::detail::cuda::Gfn2SpinDeviceOutput;
using xtbloom::detail::cuda::Gfn2SpinDeviceWorkspace;
using xtbloom::detail::cuda::reset_gfn2_spin_device_errors_cuda;
using xtbloom::detail::gfn2::BasisPlan;
using xtbloom::detail::gfn2::SpinPolarizationPlan;
using xtbloom::detail::gfn2::SpinPolarizationView;
using xtbloom::detail::gfn2::WavefunctionLayout;

#define CHECK(condition)                                                                \
  do {                                                                                  \
    if (!(condition)) {                                                                 \
      std::fprintf(stderr, "spin test check failed at %s:%d: %s\n", __FILE__, __LINE__, \
                   #condition);                                                         \
      return __LINE__;                                                                  \
    }                                                                                   \
  } while (false)

#define CUDA_CHECK(expression)                                                          \
  do {                                                                                  \
    const cudaError_t status_ = (expression);                                           \
    if (status_ != cudaSuccess) {                                                       \
      std::fprintf(stderr, "spin test CUDA failure at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(status_));                                        \
      return __LINE__;                                                                  \
    }                                                                                   \
  } while (false)

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t elements) { allocate_or_abort(elements); }
  ~DeviceBuffer() { (void)cudaFree(pointer_); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : pointer_(std::exchange(other.pointer_, nullptr)), elements_(other.elements_) {
    other.elements_ = 0u;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      (void)cudaFree(pointer_);
      pointer_ = std::exchange(other.pointer_, nullptr);
      elements_ = other.elements_;
      other.elements_ = 0u;
    }
    return *this;
  }

  cudaError_t upload(const std::vector<T>& host, cudaStream_t stream = nullptr) const {
    if (host.size() != elements_) {
      return cudaErrorInvalidValue;
    }
    return elements_ == 0u ? cudaSuccess
                           : cudaMemcpyAsync(pointer_, host.data(), elements_ * sizeof(T),
                                             cudaMemcpyHostToDevice, stream);
  }

  cudaError_t upload_one(const T& host, cudaStream_t stream = nullptr) const {
    if (elements_ != 1u) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(pointer_, &host, sizeof(T), cudaMemcpyHostToDevice, stream);
  }

  cudaError_t download(std::vector<T>& host, cudaStream_t stream = nullptr) const {
    host.resize(elements_);
    return elements_ == 0u ? cudaSuccess
                           : cudaMemcpyAsync(host.data(), pointer_, elements_ * sizeof(T),
                                             cudaMemcpyDeviceToHost, stream);
  }

  T* get() const noexcept { return pointer_; }
  std::size_t size() const noexcept { return elements_; }

 private:
  void allocate_or_abort(std::size_t elements) {
    elements_ = elements;
    const cudaError_t status =
        elements == 0u ? cudaSuccess
                       : cudaMalloc(reinterpret_cast<void**>(&pointer_), elements * sizeof(T));
    if (status != cudaSuccess) {
      std::fprintf(stderr, "spin test allocation failed: %s\n", cudaGetErrorString(status));
      std::abort();
    }
  }

  T* pointer_ = nullptr;
  std::size_t elements_ = 0u;
};

bool near(double actual, double expected, double tolerance = 3.0e-13) {
  return std::abs(actual - expected) <=
         tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

bool exact_positive_zero(double value) { return value == 0.0 && !std::signbit(value); }

struct HostCase {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> batch_shell_offsets;
  std::vector<std::int64_t> atom_shell_offsets;
  std::vector<std::int64_t> shell_population_offsets;
  std::vector<std::int32_t> spin_channels;
  std::vector<std::int64_t> coupling_offsets;
  std::vector<double> coupling_matrices;
  std::vector<double> populations;
  std::vector<std::uint8_t> active;
  std::vector<double> expected_energies;
  std::vector<double> expected_potentials;
  std::uint64_t plan_token = 0x5350494e475055ULL;

  std::int64_t batch_size() const { return static_cast<std::int64_t>(spin_channels.size()); }
  std::int64_t total_atoms() const { return atom_offsets.empty() ? 0 : atom_offsets.back(); }
  std::int64_t total_shells() const {
    return batch_shell_offsets.empty() ? 0 : batch_shell_offsets.back();
  }

  SpinPolarizationView cpu_view() const {
    return {batch_size(),
            total_atoms(),
            total_shells(),
            static_cast<std::int64_t>(populations.size()),
            static_cast<std::int64_t>(atom_offsets.size()),
            static_cast<std::int64_t>(batch_shell_offsets.size()),
            static_cast<std::int64_t>(atom_shell_offsets.size()),
            static_cast<std::int64_t>(shell_population_offsets.size()),
            static_cast<std::int64_t>(spin_channels.size()),
            static_cast<std::int64_t>(coupling_offsets.size()),
            static_cast<std::int64_t>(coupling_matrices.size()),
            atom_offsets.data(),
            batch_shell_offsets.data(),
            atom_shell_offsets.data(),
            shell_population_offsets.data(),
            spin_channels.data(),
            coupling_offsets.data(),
            coupling_matrices.data()};
  }

  bool refresh_cpu_reference() {
    expected_energies.assign(static_cast<std::size_t>(batch_size()), 0.0);
    expected_potentials.assign(populations.size(), 0.0);
    std::string error;
    const xtbloom_status_t status = xtbloom::detail::gfn2::evaluate_spin_polarization_cpu(
        cpu_view(), populations.data(), expected_energies.data(), expected_potentials.data(),
        error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      std::fprintf(stderr, "CPU spin reference failed: status=%d error=%s\n", status,
                   error.c_str());
      return false;
    }
    return true;
  }
};

HostCase make_case(std::int64_t batch_size, bool all_restricted = false) {
  HostCase host;
  host.atom_offsets.push_back(0);
  host.batch_shell_offsets.push_back(0);
  host.atom_shell_offsets.push_back(0);
  host.shell_population_offsets.push_back(0);
  host.coupling_offsets.push_back(0);

  std::int64_t atoms = 0;
  std::int64_t shells = 0;
  std::int64_t populations = 0;
  std::int64_t couplings = 0;
  for (std::int64_t system = 0; system < batch_size; ++system) {
    const std::int64_t system_atoms = 1 + system % 2;
    const std::int32_t channels = all_restricted ? 1 : (system % 3 == 1 ? 1 : 2);
    const std::int64_t system_shell_begin = shells;
    for (std::int64_t local_atom = 0; local_atom < system_atoms; ++local_atom) {
      const std::int64_t atom_shells = 1 + (system + local_atom) % 3;
      const std::int64_t matrix_begin = couplings;
      for (std::int64_t row = 0; row < atom_shells; ++row) {
        for (std::int64_t column = 0; column < atom_shells; ++column) {
          const double value =
              row == column ? -0.026 - 0.001 * static_cast<double>((system + local_atom + row) % 7)
                            : -0.0025 * static_cast<double>(1 + row + column);
          host.coupling_matrices.push_back(value);
        }
      }
      couplings = matrix_begin + atom_shells * atom_shells;
      host.coupling_offsets.push_back(couplings);
      shells += atom_shells;
      host.atom_shell_offsets.push_back(shells);
      ++atoms;
    }
    host.atom_offsets.push_back(atoms);
    host.batch_shell_offsets.push_back(shells);
    host.spin_channels.push_back(channels);
    populations += channels * (shells - system_shell_begin);
    host.shell_population_offsets.push_back(populations);
  }

  host.populations.resize(static_cast<std::size_t>(populations));
  host.active.assign(static_cast<std::size_t>(batch_size), 1u);
  for (std::int64_t system = 0; system < batch_size; ++system) {
    const std::int64_t shell_count =
        host.batch_shell_offsets[static_cast<std::size_t>(system) + 1u] -
        host.batch_shell_offsets[static_cast<std::size_t>(system)];
    const std::int64_t population_begin =
        host.shell_population_offsets[static_cast<std::size_t>(system)];
    for (std::int64_t shell = 0; shell < shell_count; ++shell) {
      host.populations[static_cast<std::size_t>(population_begin + shell)] =
          0.13 + 0.017 * static_cast<double>((system + 3 * shell) % 11);
      if (host.spin_channels[static_cast<std::size_t>(system)] == 2) {
        host.populations[static_cast<std::size_t>(population_begin + shell_count + shell)] =
            -0.41 + 0.073 * static_cast<double>((2 * system + shell) % 9);
      }
    }
  }
  if (!host.refresh_cpu_reference()) {
    std::abort();
  }
  return host;
}

HostCase make_chromium_literal_case() {
  HostCase host;
  host.atom_offsets = {0, 1};
  host.batch_shell_offsets = {0, 3};
  host.atom_shell_offsets = {0, 3};
  host.shell_population_offsets = {0, 6};
  host.spin_channels = {2};
  host.coupling_offsets = {0, 9};
  /*
   * Chromium's GFN2 basis stores shells in d,s,p order.  Pinning the literal
   * W matrix here catches an otherwise numerically plausible s,p,d reorder.
   */
  host.coupling_matrices = {
      -0.015775, -0.003725, -0.001463, -0.003725, -0.014475,
      -0.011612, -0.001463, -0.011612, -0.016000,
  };
  host.populations = {0.0, 0.0, 0.0, 0.2, -0.3, 0.4};
  host.active = {1u};
  if (!host.refresh_cpu_reference()) {
    std::abort();
  }
  return host;
}

struct DeviceCase {
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> batch_shell_offsets;
  DeviceBuffer<std::int64_t> atom_shell_offsets;
  DeviceBuffer<std::int64_t> shell_population_offsets;
  DeviceBuffer<std::int32_t> spin_channels;
  DeviceBuffer<std::int64_t> coupling_offsets;
  DeviceBuffer<double> coupling_matrices;
  DeviceBuffer<double> populations;
  DeviceBuffer<std::uint8_t> active;
  DeviceBuffer<std::uint32_t> canonical_sequence;
  DeviceBuffer<double> energies;
  DeviceBuffer<double> potentials;
  DeviceBuffer<double> energy_scratch;
  DeviceBuffer<double> potential_scratch;
  DeviceBuffer<std::uint32_t> workspace_sequence;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;

  Gfn2SpinDeviceBatch batch{};
  Gfn2SpinDeviceInput input{};
  Gfn2SpinDeviceOutput output{};
  Gfn2SpinDeviceWorkspace workspace{};

  explicit DeviceCase(const HostCase& host, cudaStream_t stream = nullptr)
      : atom_offsets(host.atom_offsets.size()),
        batch_shell_offsets(host.batch_shell_offsets.size()),
        atom_shell_offsets(host.atom_shell_offsets.size()),
        shell_population_offsets(host.shell_population_offsets.size()),
        spin_channels(host.spin_channels.size()),
        coupling_offsets(host.coupling_offsets.size()),
        coupling_matrices(host.coupling_matrices.size()),
        populations(host.populations.size()),
        active(host.active.size()),
        canonical_sequence(1u),
        energies(host.active.size()),
        potentials(host.populations.size()),
        energy_scratch(host.active.size()),
        potential_scratch(host.populations.size()),
        workspace_sequence(1u),
        system_errors(host.active.size()),
        device_error(1u) {
    upload_or_abort(atom_offsets.upload(host.atom_offsets, stream));
    upload_or_abort(batch_shell_offsets.upload(host.batch_shell_offsets, stream));
    upload_or_abort(atom_shell_offsets.upload(host.atom_shell_offsets, stream));
    upload_or_abort(shell_population_offsets.upload(host.shell_population_offsets, stream));
    upload_or_abort(spin_channels.upload(host.spin_channels, stream));
    upload_or_abort(coupling_offsets.upload(host.coupling_offsets, stream));
    upload_or_abort(coupling_matrices.upload(host.coupling_matrices, stream));
    upload_or_abort(populations.upload(host.populations, stream));
    upload_or_abort(active.upload(host.active, stream));
    const std::uint32_t sequence = 1u;
    upload_or_abort(canonical_sequence.upload_one(sequence, stream));

    batch.batch_size = host.batch_size();
    batch.total_atoms = host.total_atoms();
    batch.total_shells = host.total_shells();
    batch.shell_population_elements = static_cast<std::int64_t>(host.populations.size());
    batch.atom_offset_count = static_cast<std::int64_t>(host.atom_offsets.size());
    batch.batch_shell_offset_count = static_cast<std::int64_t>(host.batch_shell_offsets.size());
    batch.atom_shell_offset_count = static_cast<std::int64_t>(host.atom_shell_offsets.size());
    batch.shell_population_offset_count =
        static_cast<std::int64_t>(host.shell_population_offsets.size());
    batch.spin_channel_count = static_cast<std::int64_t>(host.spin_channels.size());
    batch.coupling_offset_count = static_cast<std::int64_t>(host.coupling_offsets.size());
    batch.coupling_matrix_count = static_cast<std::int64_t>(host.coupling_matrices.size());
    batch.plan_token = host.plan_token;
    batch.atom_offsets = atom_offsets.get();
    batch.batch_shell_offsets = batch_shell_offsets.get();
    batch.atom_shell_offsets = atom_shell_offsets.get();
    batch.shell_population_offsets = shell_population_offsets.get();
    batch.spin_channels = spin_channels.get();
    batch.coupling_offsets = coupling_offsets.get();
    batch.coupling_matrices = coupling_matrices.get();

    input = {populations.get(), static_cast<std::int64_t>(host.populations.size()),
             host.plan_token};
    output = {energies.get(), static_cast<std::int64_t>(host.active.size()), potentials.get(),
              static_cast<std::int64_t>(host.populations.size()), host.plan_token};
    workspace = {energy_scratch.get(),     static_cast<std::int64_t>(host.active.size()),
                 potential_scratch.get(),  static_cast<std::int64_t>(host.populations.size()),
                 workspace_sequence.get(), 1,
                 host.plan_token};
  }

  Gfn2SccIterationDeviceActivity activity_view(std::uint64_t token = 0u) const {
    return {active.get(), canonical_sequence.get(), batch.batch_size, 1,
            token == 0u ? batch.plan_token : token};
  }

  void seed_outputs_or_abort(std::vector<double>& energy_seed, std::vector<double>& potential_seed,
                             cudaStream_t stream = nullptr) const {
    energy_seed.resize(static_cast<std::size_t>(batch.batch_size));
    potential_seed.resize(static_cast<std::size_t>(batch.shell_population_elements));
    for (std::size_t index = 0; index < energy_seed.size(); ++index) {
      energy_seed[index] = 113.0 + static_cast<double>(index);
    }
    for (std::size_t index = 0; index < potential_seed.size(); ++index) {
      potential_seed[index] = 227.0 + 0.25 * static_cast<double>(index);
    }
    upload_or_abort(energies.upload(energy_seed, stream));
    upload_or_abort(potentials.upload(potential_seed, stream));
  }

  cudaError_t launch(const Gfn2SccIterationDeviceActivity& activity,
                     cudaStream_t stream = nullptr) const {
    cudaError_t status = reset_gfn2_spin_device_errors_cuda(batch.batch_size, system_errors.get(),
                                                            device_error.get(), stream);
    if (status != cudaSuccess) {
      return status;
    }
    return evaluate_gfn2_spin_polarization_cuda(batch, input, activity, output, workspace,
                                                system_errors.get(), device_error.get(), stream);
  }

 private:
  static void upload_or_abort(cudaError_t status) {
    if (status != cudaSuccess) {
      std::fprintf(stderr, "spin test fixture upload failed: %s\n", cudaGetErrorString(status));
      std::abort();
    }
  }
};

int download_results(const DeviceCase& device, std::vector<double>& energies,
                     std::vector<double>& potentials, std::vector<std::uint32_t>& system_errors,
                     std::vector<std::uint32_t>& device_error, cudaStream_t stream = nullptr) {
  CUDA_CHECK(device.energies.download(energies, stream));
  CUDA_CHECK(device.potentials.download(potentials, stream));
  CUDA_CHECK(device.system_errors.download(system_errors, stream));
  CUDA_CHECK(device.device_error.download(device_error, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return 0;
}

int check_healthy_parity(const HostCase& host, const std::vector<double>& energies,
                         const std::vector<double>& potentials,
                         const std::vector<std::uint32_t>& system_errors) {
  CHECK(energies.size() == static_cast<std::size_t>(host.batch_size()));
  CHECK(potentials.size() == host.populations.size());
  CHECK(system_errors.size() == static_cast<std::size_t>(host.batch_size()));
  for (std::int64_t system = 0; system < host.batch_size(); ++system) {
    CHECK(system_errors[static_cast<std::size_t>(system)] == 0u);
    CHECK(near(energies[static_cast<std::size_t>(system)],
               host.expected_energies[static_cast<std::size_t>(system)]));
    const std::int64_t begin = host.shell_population_offsets[static_cast<std::size_t>(system)];
    const std::int64_t end = host.shell_population_offsets[static_cast<std::size_t>(system) + 1u];
    for (std::int64_t element = begin; element < end; ++element) {
      CHECK(near(potentials[static_cast<std::size_t>(element)],
                 host.expected_potentials[static_cast<std::size_t>(element)]));
    }
    if (host.spin_channels[static_cast<std::size_t>(system)] == 1) {
      CHECK(exact_positive_zero(energies[static_cast<std::size_t>(system)]));
      for (std::int64_t element = begin; element < end; ++element) {
        CHECK(exact_positive_zero(potentials[static_cast<std::size_t>(element)]));
      }
    }
  }
  return 0;
}

int check_peer_failure_result(const HostCase& host, const DeviceCase& device,
                              std::int64_t failed_system, Gfn2SpinDeviceError expected_error,
                              const std::vector<double>& energy_seed,
                              const std::vector<double>& potential_seed) {
  std::vector<double> energies;
  std::vector<double> potentials;
  std::vector<std::uint32_t> system_errors;
  std::vector<std::uint32_t> device_error;
  CHECK(download_results(device, energies, potentials, system_errors, device_error) == 0);
  const std::uint32_t expected_code = static_cast<std::uint32_t>(expected_error);
  CHECK(device_error == std::vector<std::uint32_t>({expected_code}));
  CHECK(system_errors[static_cast<std::size_t>(failed_system)] == expected_code);
  CHECK(energies[static_cast<std::size_t>(failed_system)] ==
        energy_seed[static_cast<std::size_t>(failed_system)]);
  const std::int64_t failed_begin =
      host.shell_population_offsets[static_cast<std::size_t>(failed_system)];
  const std::int64_t failed_end =
      host.shell_population_offsets[static_cast<std::size_t>(failed_system) + 1u];
  for (std::int64_t element = failed_begin; element < failed_end; ++element) {
    CHECK(potentials[static_cast<std::size_t>(element)] ==
          potential_seed[static_cast<std::size_t>(element)]);
  }
  for (std::int64_t system = 0; system < host.batch_size(); ++system) {
    if (system == failed_system) {
      continue;
    }
    CHECK(system_errors[static_cast<std::size_t>(system)] == 0u);
    CHECK(near(energies[static_cast<std::size_t>(system)],
               host.expected_energies[static_cast<std::size_t>(system)]));
    const std::int64_t begin = host.shell_population_offsets[static_cast<std::size_t>(system)];
    const std::int64_t end = host.shell_population_offsets[static_cast<std::size_t>(system) + 1u];
    for (std::int64_t element = begin; element < end; ++element) {
      CHECK(near(potentials[static_cast<std::size_t>(element)],
                 host.expected_potentials[static_cast<std::size_t>(element)]));
    }
  }
  return 0;
}

int test_cpu_parity_mixed_ragged_batches_and_custom_stream() {
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    HostCase host = make_case(batch_size);
    DeviceCase device(host);
    std::vector<double> energy_seed;
    std::vector<double> potential_seed;
    device.seed_outputs_or_abort(energy_seed, potential_seed);
    CHECK(device.launch(device.activity_view()) == cudaSuccess);

    std::vector<double> energies;
    std::vector<double> potentials;
    std::vector<std::uint32_t> system_errors;
    std::vector<std::uint32_t> device_error;
    CHECK(download_results(device, energies, potentials, system_errors, device_error) == 0);
    CHECK(device_error == std::vector<std::uint32_t>({0u}));
    CHECK(check_healthy_parity(host, energies, potentials, system_errors) == 0);
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  HostCase host = make_case(32);
  DeviceCase device(host, stream);
  std::vector<double> energy_seed;
  std::vector<double> potential_seed;
  device.seed_outputs_or_abort(energy_seed, potential_seed, stream);
  CHECK(device.launch(device.activity_view(), stream) == cudaSuccess);
  std::vector<double> energies;
  std::vector<double> potentials;
  std::vector<std::uint32_t> system_errors;
  std::vector<std::uint32_t> device_error;
  CHECK(download_results(device, energies, potentials, system_errors, device_error, stream) == 0);
  CHECK(device_error == std::vector<std::uint32_t>({0u}));
  CHECK(check_healthy_parity(host, energies, potentials, system_errors) == 0);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_restricted_exact_zero_without_reading_poison() {
  HostCase host = make_case(1, true);
  std::fill(host.populations.begin(), host.populations.end(),
            std::numeric_limits<double>::quiet_NaN());
  std::fill(host.coupling_matrices.begin(), host.coupling_matrices.end(),
            std::numeric_limits<double>::quiet_NaN());
  DeviceCase device(host);
  std::vector<double> energy_seed;
  std::vector<double> potential_seed;
  device.seed_outputs_or_abort(energy_seed, potential_seed);
  CHECK(device.launch(device.activity_view()) == cudaSuccess);

  std::vector<double> energies;
  std::vector<double> potentials;
  std::vector<std::uint32_t> system_errors;
  std::vector<std::uint32_t> device_error;
  CHECK(download_results(device, energies, potentials, system_errors, device_error) == 0);
  CHECK(device_error == std::vector<std::uint32_t>({0u}));
  CHECK(system_errors == std::vector<std::uint32_t>({0u}));
  CHECK(energies.size() == 1u && exact_positive_zero(energies[0]));
  CHECK(std::all_of(potentials.begin(), potentials.end(), exact_positive_zero));
  return 0;
}

int test_chromium_d_s_p_shell_order_literal() {
  const std::vector<std::int64_t> atom_offsets{0, 1};
  const std::vector<std::int32_t> atomic_numbers{24};
  const std::vector<double> charges{0.0};
  const std::vector<std::int32_t> unpaired{0};
  const std::vector<std::int32_t> spin_channels{2};
  BasisPlan basis;
  WavefunctionLayout wavefunction;
  SpinPolarizationPlan spin;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_basis_plan(1, 1, atom_offsets.data(), atomic_numbers.data(),
                                               basis, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::make_wavefunction_layout(
            basis, atomic_numbers.data(), charges.data(), unpaired.data(), spin_channels.data(),
            wavefunction, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::make_spin_polarization_plan(basis, wavefunction, spin, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(basis.angular_momenta == std::vector<std::uint8_t>({2u, 0u, 1u}));

  HostCase host = make_chromium_literal_case();
  CHECK(spin.coupling_matrices == host.coupling_matrices);
  DeviceCase device(host);
  std::vector<double> energy_seed;
  std::vector<double> potential_seed;
  device.seed_outputs_or_abort(energy_seed, potential_seed);
  CHECK(device.launch(device.activity_view()) == cudaSuccess);

  std::vector<double> energies;
  std::vector<double> potentials;
  std::vector<std::uint32_t> system_errors;
  std::vector<std::uint32_t> device_error;
  CHECK(download_results(device, energies, potentials, system_errors, device_error) == 0);
  CHECK(system_errors == std::vector<std::uint32_t>({0u}));
  CHECK(device_error == std::vector<std::uint32_t>({0u}));

  const std::vector<double> magnetization{0.2, -0.3, 0.4};
  std::vector<double> literal_potential(3u, 0.0);
  double literal_energy = 0.0;
  for (std::size_t row = 0; row < 3u; ++row) {
    for (std::size_t column = 0; column < 3u; ++column) {
      literal_potential[row] = std::fma(host.coupling_matrices[row * 3u + column],
                                        magnetization[column], literal_potential[row]);
    }
    literal_energy = std::fma(0.5 * magnetization[row], literal_potential[row], literal_energy);
  }
  CHECK(near(energies[0], literal_energy, 2.0e-15));
  for (std::size_t shell = 0; shell < 3u; ++shell) {
    CHECK(exact_positive_zero(potentials[shell]));
    CHECK(near(potentials[3u + shell], literal_potential[shell], 2.0e-15));
  }
  return 0;
}

int test_cuda_energy_derivative_matches_potential() {
  HostCase host = make_case(1);
  CHECK(host.spin_channels == std::vector<std::int32_t>({2}));
  DeviceCase device(host);
  std::vector<double> energy_seed;
  std::vector<double> potential_seed;
  device.seed_outputs_or_abort(energy_seed, potential_seed);
  CHECK(device.launch(device.activity_view()) == cudaSuccess);
  std::vector<double> baseline_energy;
  std::vector<double> baseline_potential;
  std::vector<std::uint32_t> system_errors;
  std::vector<std::uint32_t> device_error;
  CHECK(download_results(device, baseline_energy, baseline_potential, system_errors,
                         device_error) == 0);
  CHECK(system_errors == std::vector<std::uint32_t>({0u}));

  constexpr double step = 1.0e-5;
  const std::int64_t system_shells = host.total_shells();
  const std::int64_t magnetization_begin = system_shells;
  for (std::int64_t local = 0; local < system_shells; ++local) {
    const std::size_t index = static_cast<std::size_t>(magnetization_begin + local);
    host.populations[index] += step;
    CUDA_CHECK(device.populations.upload(host.populations));
    CHECK(device.launch(device.activity_view()) == cudaSuccess);
    std::vector<double> right_energy;
    std::vector<double> ignored_potential;
    CHECK(download_results(device, right_energy, ignored_potential, system_errors, device_error) ==
          0);

    host.populations[index] -= 2.0 * step;
    CUDA_CHECK(device.populations.upload(host.populations));
    CHECK(device.launch(device.activity_view()) == cudaSuccess);
    std::vector<double> left_energy;
    CHECK(download_results(device, left_energy, ignored_potential, system_errors, device_error) ==
          0);

    host.populations[index] += step;
    const double derivative = (right_energy[0] - left_energy[0]) / (2.0 * step);
    CHECK(near(derivative, baseline_potential[index], 4.0e-11));
  }
  return 0;
}

int test_inactive_poison_is_unread_and_byte_stable() {
  HostCase host = make_case(8);
  constexpr std::int64_t inactive_system = 2;
  CHECK(host.spin_channels[static_cast<std::size_t>(inactive_system)] == 2);
  host.active[static_cast<std::size_t>(inactive_system)] = 0u;
  const std::int64_t population_begin =
      host.shell_population_offsets[static_cast<std::size_t>(inactive_system)];
  const std::int64_t population_end =
      host.shell_population_offsets[static_cast<std::size_t>(inactive_system) + 1u];
  for (std::int64_t element = population_begin; element < population_end; ++element) {
    host.populations[static_cast<std::size_t>(element)] = std::numeric_limits<double>::quiet_NaN();
  }
  const std::int64_t atom_begin = host.atom_offsets[static_cast<std::size_t>(inactive_system)];
  const std::int64_t atom_end = host.atom_offsets[static_cast<std::size_t>(inactive_system) + 1u];
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    const std::int64_t coupling_begin = host.coupling_offsets[static_cast<std::size_t>(atom)];
    const std::int64_t coupling_end = host.coupling_offsets[static_cast<std::size_t>(atom) + 1u];
    for (std::int64_t element = coupling_begin; element < coupling_end; ++element) {
      host.coupling_matrices[static_cast<std::size_t>(element)] =
          std::numeric_limits<double>::quiet_NaN();
    }
  }

  DeviceCase device(host);
  std::vector<double> energy_seed;
  std::vector<double> potential_seed;
  device.seed_outputs_or_abort(energy_seed, potential_seed);
  CHECK(device.launch(device.activity_view()) == cudaSuccess);
  std::vector<double> energies;
  std::vector<double> potentials;
  std::vector<std::uint32_t> system_errors;
  std::vector<std::uint32_t> device_error;
  CHECK(download_results(device, energies, potentials, system_errors, device_error) == 0);
  CHECK(device_error == std::vector<std::uint32_t>({0u}));
  CHECK(std::all_of(system_errors.begin(), system_errors.end(),
                    [](std::uint32_t value) { return value == 0u; }));
  CHECK(energies[static_cast<std::size_t>(inactive_system)] ==
        energy_seed[static_cast<std::size_t>(inactive_system)]);
  for (std::int64_t element = population_begin; element < population_end; ++element) {
    CHECK(potentials[static_cast<std::size_t>(element)] ==
          potential_seed[static_cast<std::size_t>(element)]);
  }
  for (std::int64_t system = 0; system < host.batch_size(); ++system) {
    if (system == inactive_system) {
      continue;
    }
    CHECK(near(energies[static_cast<std::size_t>(system)],
               host.expected_energies[static_cast<std::size_t>(system)]));
    const std::int64_t begin = host.shell_population_offsets[static_cast<std::size_t>(system)];
    const std::int64_t end = host.shell_population_offsets[static_cast<std::size_t>(system) + 1u];
    for (std::int64_t element = begin; element < end; ++element) {
      CHECK(near(potentials[static_cast<std::size_t>(element)],
                 host.expected_potentials[static_cast<std::size_t>(element)]));
    }
  }
  return 0;
}

int test_inactive_topology_and_spin_metadata_are_unread() {
  HostCase host = make_case(8);
  constexpr std::int64_t inactive_system = 7;
  host.active[static_cast<std::size_t>(inactive_system)] = 0u;
  DeviceCase device(host);

  /*
   * Poison only topology entries owned exclusively by the final inactive
   * system.  The shared start boundaries remain valid for its healthy
   * predecessor, so any diagnostic here proves the inactive peer was read.
   */
  std::vector<std::int64_t> atom_offsets = host.atom_offsets;
  std::vector<std::int64_t> batch_shell_offsets = host.batch_shell_offsets;
  std::vector<std::int64_t> atom_shell_offsets = host.atom_shell_offsets;
  std::vector<std::int64_t> population_offsets = host.shell_population_offsets;
  std::vector<std::int32_t> spin_channels = host.spin_channels;
  std::vector<std::int64_t> coupling_offsets = host.coupling_offsets;
  atom_offsets.back() = -101;
  batch_shell_offsets.back() = -103;
  population_offsets.back() = -107;
  spin_channels.back() = 17;
  const std::int64_t atom_begin = host.atom_offsets[static_cast<std::size_t>(inactive_system)];
  const std::int64_t atom_end = host.atom_offsets[static_cast<std::size_t>(inactive_system) + 1u];
  for (std::int64_t offset = atom_begin + 1; offset <= atom_end; ++offset) {
    atom_shell_offsets[static_cast<std::size_t>(offset)] = -109 - offset;
    coupling_offsets[static_cast<std::size_t>(offset)] = -127 - offset;
  }
  CUDA_CHECK(device.atom_offsets.upload(atom_offsets));
  CUDA_CHECK(device.batch_shell_offsets.upload(batch_shell_offsets));
  CUDA_CHECK(device.atom_shell_offsets.upload(atom_shell_offsets));
  CUDA_CHECK(device.shell_population_offsets.upload(population_offsets));
  CUDA_CHECK(device.spin_channels.upload(spin_channels));
  CUDA_CHECK(device.coupling_offsets.upload(coupling_offsets));

  std::vector<double> poisoned_populations = host.populations;
  const std::int64_t population_begin =
      host.shell_population_offsets[static_cast<std::size_t>(inactive_system)];
  const std::int64_t population_end =
      host.shell_population_offsets[static_cast<std::size_t>(inactive_system) + 1u];
  for (std::int64_t element = population_begin; element < population_end; ++element) {
    poisoned_populations[static_cast<std::size_t>(element)] =
        std::numeric_limits<double>::quiet_NaN();
  }
  CUDA_CHECK(device.populations.upload(poisoned_populations));

  std::vector<double> energy_seed;
  std::vector<double> potential_seed;
  device.seed_outputs_or_abort(energy_seed, potential_seed);
  CHECK(device.launch(device.activity_view()) == cudaSuccess);
  std::vector<double> energies;
  std::vector<double> potentials;
  std::vector<std::uint32_t> system_errors;
  std::vector<std::uint32_t> device_error;
  CHECK(download_results(device, energies, potentials, system_errors, device_error) == 0);
  CHECK(device_error == std::vector<std::uint32_t>({0u}));
  CHECK(std::all_of(system_errors.begin(), system_errors.end(),
                    [](std::uint32_t value) { return value == 0u; }));
  CHECK(energies[static_cast<std::size_t>(inactive_system)] ==
        energy_seed[static_cast<std::size_t>(inactive_system)]);
  for (std::int64_t element = population_begin; element < population_end; ++element) {
    CHECK(potentials[static_cast<std::size_t>(element)] ==
          potential_seed[static_cast<std::size_t>(element)]);
  }
  for (std::int64_t system = 0; system < inactive_system; ++system) {
    CHECK(near(energies[static_cast<std::size_t>(system)],
               host.expected_energies[static_cast<std::size_t>(system)]));
    const std::int64_t begin = host.shell_population_offsets[static_cast<std::size_t>(system)];
    const std::int64_t end = host.shell_population_offsets[static_cast<std::size_t>(system) + 1u];
    for (std::int64_t element = begin; element < end; ++element) {
      CHECK(near(potentials[static_cast<std::size_t>(element)],
                 host.expected_potentials[static_cast<std::size_t>(element)]));
    }
  }
  return 0;
}

int test_peer_nan_failure_isolation() {
  HostCase host = make_case(8);
  constexpr std::int64_t failed_system = 2;
  CHECK(host.spin_channels[static_cast<std::size_t>(failed_system)] == 2);
  const std::int64_t shell_count =
      host.batch_shell_offsets[static_cast<std::size_t>(failed_system) + 1u] -
      host.batch_shell_offsets[static_cast<std::size_t>(failed_system)];
  const std::int64_t population_begin =
      host.shell_population_offsets[static_cast<std::size_t>(failed_system)];
  host.populations[static_cast<std::size_t>(population_begin + shell_count)] =
      std::numeric_limits<double>::quiet_NaN();

  DeviceCase device(host);
  std::vector<double> energy_seed;
  std::vector<double> potential_seed;
  device.seed_outputs_or_abort(energy_seed, potential_seed);
  CHECK(device.launch(device.activity_view()) == cudaSuccess);
  std::vector<double> energies;
  std::vector<double> potentials;
  std::vector<std::uint32_t> system_errors;
  std::vector<std::uint32_t> device_error;
  CHECK(download_results(device, energies, potentials, system_errors, device_error) == 0);
  const std::uint32_t expected_error =
      static_cast<std::uint32_t>(Gfn2SpinDeviceError::kNonfinitePopulation);
  CHECK(device_error == std::vector<std::uint32_t>({expected_error}));
  CHECK(system_errors[static_cast<std::size_t>(failed_system)] == expected_error);
  CHECK(energies[static_cast<std::size_t>(failed_system)] ==
        energy_seed[static_cast<std::size_t>(failed_system)]);
  const std::int64_t population_end =
      host.shell_population_offsets[static_cast<std::size_t>(failed_system) + 1u];
  for (std::int64_t element = population_begin; element < population_end; ++element) {
    CHECK(potentials[static_cast<std::size_t>(element)] ==
          potential_seed[static_cast<std::size_t>(element)]);
  }
  for (std::int64_t system = 0; system < host.batch_size(); ++system) {
    if (system == failed_system) {
      continue;
    }
    CHECK(system_errors[static_cast<std::size_t>(system)] == 0u);
    CHECK(near(energies[static_cast<std::size_t>(system)],
               host.expected_energies[static_cast<std::size_t>(system)]));
    const std::int64_t begin = host.shell_population_offsets[static_cast<std::size_t>(system)];
    const std::int64_t end = host.shell_population_offsets[static_cast<std::size_t>(system) + 1u];
    for (std::int64_t element = begin; element < end; ++element) {
      CHECK(near(potentials[static_cast<std::size_t>(element)],
                 host.expected_potentials[static_cast<std::size_t>(element)]));
    }
  }
  return 0;
}

int test_device_error_classification_and_sequence_gate() {
  constexpr std::int64_t middle_system = 2;
  constexpr std::int64_t final_system = 7;
  constexpr std::int64_t coupling_system = 6;

  {
    HostCase host = make_case(8);
    DeviceCase device(host);
    std::vector<double> energy_seed;
    std::vector<double> potential_seed;
    device.seed_outputs_or_abort(energy_seed, potential_seed);
    std::vector<std::uint8_t> invalid_active = host.active;
    invalid_active[static_cast<std::size_t>(middle_system)] = 9u;
    CUDA_CHECK(device.active.upload(invalid_active));
    CHECK(device.launch(device.activity_view()) == cudaSuccess);
    CHECK(check_peer_failure_result(host, device, middle_system,
                                    Gfn2SpinDeviceError::kInvalidActiveMask, energy_seed,
                                    potential_seed) == 0);
  }

  {
    HostCase host = make_case(8);
    DeviceCase device(host);
    std::vector<double> energy_seed;
    std::vector<double> potential_seed;
    device.seed_outputs_or_abort(energy_seed, potential_seed);
    std::vector<std::int32_t> invalid_spin_channels = host.spin_channels;
    invalid_spin_channels[static_cast<std::size_t>(middle_system)] = 3;
    CUDA_CHECK(device.spin_channels.upload(invalid_spin_channels));
    CHECK(device.launch(device.activity_view()) == cudaSuccess);
    CHECK(check_peer_failure_result(host, device, middle_system,
                                    Gfn2SpinDeviceError::kInvalidSpinChannels, energy_seed,
                                    potential_seed) == 0);
  }

  {
    HostCase host = make_case(8);
    DeviceCase device(host);
    std::vector<double> energy_seed;
    std::vector<double> potential_seed;
    device.seed_outputs_or_abort(energy_seed, potential_seed);
    std::vector<std::int64_t> invalid_offsets = host.shell_population_offsets;
    invalid_offsets.back() = static_cast<std::int64_t>(host.populations.size()) + 1;
    CUDA_CHECK(device.shell_population_offsets.upload(invalid_offsets));
    CHECK(device.launch(device.activity_view()) == cudaSuccess);
    CHECK(check_peer_failure_result(host, device, final_system,
                                    Gfn2SpinDeviceError::kInvalidOffsets, energy_seed,
                                    potential_seed) == 0);
  }

  {
    HostCase host = make_case(8);
    DeviceCase device(host);
    std::vector<double> energy_seed;
    std::vector<double> potential_seed;
    device.seed_outputs_or_abort(energy_seed, potential_seed);
    std::vector<std::int64_t> invalid_coupling = host.coupling_offsets;
    /* System 7 is restricted and must not inspect W topology.  Corrupt the
     * final atom of the preceding unrestricted peer so this case exercises
     * coupling validation without weakening the restricted no-read contract. */
    const std::int64_t atom = host.atom_offsets[static_cast<std::size_t>(coupling_system) + 1u] - 1;
    ++invalid_coupling[static_cast<std::size_t>(atom) + 1u];
    CUDA_CHECK(device.coupling_offsets.upload(invalid_coupling));
    CHECK(device.launch(device.activity_view()) == cudaSuccess);
    CHECK(check_peer_failure_result(host, device, coupling_system,
                                    Gfn2SpinDeviceError::kInvalidCoupling, energy_seed,
                                    potential_seed) == 0);
  }

  {
    HostCase host = make_case(8);
    DeviceCase device(host);
    std::vector<double> energy_seed;
    std::vector<double> potential_seed;
    device.seed_outputs_or_abort(energy_seed, potential_seed);
    const std::uint32_t closed = 0u;
    CUDA_CHECK(device.canonical_sequence.upload_one(closed));
    CHECK(device.launch(device.activity_view()) == cudaSuccess);

    std::vector<double> energies;
    std::vector<double> potentials;
    std::vector<std::uint32_t> system_errors;
    std::vector<std::uint32_t> device_error;
    CHECK(download_results(device, energies, potentials, system_errors, device_error) == 0);
    CHECK(energies == energy_seed);
    CHECK(potentials == potential_seed);
    CHECK(device_error == std::vector<std::uint32_t>({0u}));
    CHECK(std::all_of(system_errors.begin(), system_errors.end(),
                      [](std::uint32_t value) { return value == 0u; }));
    std::vector<std::uint32_t> workspace_sequence;
    CUDA_CHECK(device.workspace_sequence.download(workspace_sequence));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(workspace_sequence == std::vector<std::uint32_t>({0u}));
  }
  return 0;
}

int test_alias_and_invalid_plan_rejection_are_atomic() {
  HostCase host = make_case(8);
  DeviceCase device(host);
  std::vector<double> energy_seed;
  std::vector<double> potential_seed;
  device.seed_outputs_or_abort(energy_seed, potential_seed);

  Gfn2SpinDeviceOutput aliased = device.output;
  aliased.spin_energies = device.populations.get();
  CHECK(evaluate_gfn2_spin_polarization_cuda(device.batch, device.input, device.activity_view(),
                                             aliased, device.workspace, device.system_errors.get(),
                                             device.device_error.get()) == cudaErrorInvalidValue);

  aliased = device.output;
  aliased.spin_energies = device.potentials.get();
  CHECK(evaluate_gfn2_spin_polarization_cuda(device.batch, device.input, device.activity_view(),
                                             aliased, device.workspace, device.system_errors.get(),
                                             device.device_error.get()) == cudaErrorInvalidValue);

  Gfn2SpinDeviceWorkspace aliased_workspace = device.workspace;
  aliased_workspace.energy_scratch = device.energies.get();
  CHECK(evaluate_gfn2_spin_polarization_cuda(
            device.batch, device.input, device.activity_view(), device.output, aliased_workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);

  aliased_workspace = device.workspace;
  aliased_workspace.sequence_active =
      const_cast<std::uint32_t*>(device.activity_view().sequence_active);
  CHECK(evaluate_gfn2_spin_polarization_cuda(
            device.batch, device.input, device.activity_view(), device.output, aliased_workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);

  CHECK(evaluate_gfn2_spin_polarization_cuda(
            device.batch, device.input, device.activity_view(), device.output, device.workspace,
            device.system_errors.get(), device.system_errors.get()) == cudaErrorInvalidValue);

  Gfn2SpinDeviceBatch malformed = device.batch;
  --malformed.shell_population_offset_count;
  CHECK(evaluate_gfn2_spin_polarization_cuda(
            malformed, device.input, device.activity_view(), device.output, device.workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);

  Gfn2SpinDeviceInput wrong_token = device.input;
  ++wrong_token.plan_token;
  CHECK(evaluate_gfn2_spin_polarization_cuda(
            device.batch, wrong_token, device.activity_view(), device.output, device.workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);

  Gfn2SpinDeviceBatch oversized_batch = device.batch;
  oversized_batch.batch_size =
      static_cast<std::int64_t>(std::numeric_limits<int>::max()) + std::int64_t{1};
  CHECK(evaluate_gfn2_spin_polarization_cuda(
            oversized_batch, device.input, device.activity_view(), device.output, device.workspace,
            device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidConfiguration);

  Gfn2SpinDeviceBatch overflowing_range_batch = device.batch;
  Gfn2SpinDeviceInput overflowing_range_input = device.input;
  Gfn2SpinDeviceOutput overflowing_range_output = device.output;
  Gfn2SpinDeviceWorkspace overflowing_range_workspace = device.workspace;
  overflowing_range_batch.shell_population_elements = std::numeric_limits<std::int64_t>::max();
  overflowing_range_input.shell_population_elements =
      overflowing_range_batch.shell_population_elements;
  overflowing_range_output.shell_potential_elements =
      overflowing_range_batch.shell_population_elements;
  overflowing_range_workspace.potential_elements =
      overflowing_range_batch.shell_population_elements;
  CHECK(evaluate_gfn2_spin_polarization_cuda(
            overflowing_range_batch, overflowing_range_input, device.activity_view(),
            overflowing_range_output, overflowing_range_workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);

  std::vector<double> energies;
  std::vector<double> potentials;
  CUDA_CHECK(device.energies.download(energies));
  CUDA_CHECK(device.potentials.download(potentials));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(energies == energy_seed);
  CHECK(potentials == potential_seed);
  return 0;
}

int test_changed_input_cuda_graph_replay() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  HostCase host = make_case(8);
  DeviceCase device(host, stream);
  std::vector<double> energy_seed;
  std::vector<double> potential_seed;
  device.seed_outputs_or_abort(energy_seed, potential_seed, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CHECK(device.launch(device.activity_view(), stream) == cudaSuccess);
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CHECK(graph != nullptr);
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));

  std::vector<double> first_energies;
  std::vector<double> first_potentials;
  std::vector<std::uint32_t> system_errors;
  std::vector<std::uint32_t> device_error;
  CHECK(download_results(device, first_energies, first_potentials, system_errors, device_error,
                         stream) == 0);
  CHECK(check_healthy_parity(host, first_energies, first_potentials, system_errors) == 0);

  constexpr std::int64_t changed_system = 2;
  const std::int64_t shell_count =
      host.batch_shell_offsets[static_cast<std::size_t>(changed_system) + 1u] -
      host.batch_shell_offsets[static_cast<std::size_t>(changed_system)];
  const std::int64_t population_begin =
      host.shell_population_offsets[static_cast<std::size_t>(changed_system)];
  host.populations[static_cast<std::size_t>(population_begin + shell_count)] += 0.19;
  CHECK(host.refresh_cpu_reference());
  CUDA_CHECK(device.populations.upload(host.populations, stream));
  /* The captured reset must dominate stale diagnostics from a prior caller. */
  const std::vector<std::uint32_t> poisoned_system_errors(
      static_cast<std::size_t>(host.batch_size()), 0xdeadbeefu);
  const std::uint32_t poisoned_device_error = 0xcafebabeu;
  CUDA_CHECK(device.system_errors.upload(poisoned_system_errors, stream));
  CUDA_CHECK(device.device_error.upload_one(poisoned_device_error, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));

  std::vector<double> second_energies;
  std::vector<double> second_potentials;
  CHECK(download_results(device, second_energies, second_potentials, system_errors, device_error,
                         stream) == 0);
  CHECK(device_error == std::vector<std::uint32_t>({0u}));
  CHECK(check_healthy_parity(host, second_energies, second_potentials, system_errors) == 0);
  CHECK(!near(first_energies[static_cast<std::size_t>(changed_system)],
              second_energies[static_cast<std::size_t>(changed_system)], 1.0e-15));

  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status == cudaErrorNoDevice || count_status == cudaErrorInsufficientDriver ||
      device_count == 0) {
    (void)cudaGetLastError();
    return 0;
  }
  CUDA_CHECK(count_status);
  CUDA_CHECK(cudaSetDevice(0));

  if (const int line = test_cpu_parity_mixed_ragged_batches_and_custom_stream(); line != 0) {
    return line;
  }
  if (const int line = test_restricted_exact_zero_without_reading_poison(); line != 0) {
    return line;
  }
  if (const int line = test_chromium_d_s_p_shell_order_literal(); line != 0) {
    return line;
  }
  if (const int line = test_cuda_energy_derivative_matches_potential(); line != 0) {
    return line;
  }
  if (const int line = test_inactive_poison_is_unread_and_byte_stable(); line != 0) {
    return line;
  }
  if (const int line = test_inactive_topology_and_spin_metadata_are_unread(); line != 0) {
    return line;
  }
  if (const int line = test_peer_nan_failure_isolation(); line != 0) {
    return line;
  }
  if (const int line = test_device_error_classification_and_sequence_gate(); line != 0) {
    return line;
  }
  if (const int line = test_alias_and_invalid_plan_rejection_are_atomic(); line != 0) {
    return line;
  }
  return test_changed_input_cuda_graph_replay();
}
