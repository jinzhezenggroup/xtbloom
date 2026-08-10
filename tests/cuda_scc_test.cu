#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "backends/cuda/gfn2_scc.cuh"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using xtbloom::detail::cuda::Gfn2SccDeviceBatch;
using xtbloom::detail::cuda::Gfn2SccDeviceConstMultipoles;
using xtbloom::detail::cuda::Gfn2SccDeviceError;
using xtbloom::detail::cuda::Gfn2SccDeviceMultipoles;
using xtbloom::detail::cuda::Gfn2SccDevicePolicy;
using xtbloom::detail::cuda::Gfn2SccDeviceState;
using xtbloom::detail::cuda::Gfn2SccDeviceWorkspace;

constexpr std::uint64_t kPlanToken = 0x9ac53e7b41d26f08ULL;

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
  std::size_t size() const { return count_; }

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

struct HostCase {
  std::vector<std::int64_t> shell_offsets;
  std::vector<std::int64_t> atom_offsets;
  std::vector<double> current_shell;
  std::vector<double> current_dipole;
  std::vector<double> current_quadrupole;
  std::vector<double> mixed_shell;
  std::vector<double> mixed_dipole;
  std::vector<double> mixed_quadrupole;
  std::vector<double> raw_shell;
  std::vector<double> raw_dipole;
  std::vector<double> raw_quadrupole;
  std::vector<double> published_shell;
  std::vector<double> published_dipole;
  std::vector<double> published_quadrupole;
  std::vector<double> complete_free_energies;
  std::vector<double> free_energies;
  std::vector<double> previous_free_energies;
  std::vector<double> free_energy_changes;
  std::vector<double> residual_rms;
  std::vector<std::uint64_t> iterations;
  std::vector<xtbloom_status_t> statuses;
  std::vector<std::uint8_t> converged;

  std::size_t batch_size() const { return shell_offsets.size() - 1u; }
  std::size_t total_shells() const { return current_shell.size(); }
  std::size_t total_atoms() const { return current_dipole.size() / 3u; }
};

HostCase make_case(std::size_t batch_size) {
  HostCase data;
  data.shell_offsets.resize(batch_size + 1u);
  data.atom_offsets.resize(batch_size + 1u);
  std::int64_t shells = 0;
  std::int64_t atoms = 0;
  for (std::size_t system = 0u; system < batch_size; ++system) {
    data.shell_offsets[system] = shells;
    data.atom_offsets[system] = atoms;
    const std::int64_t system_atoms = 1 + static_cast<std::int64_t>(system % 4u);
    const std::int64_t system_shells = system_atoms + 1 + static_cast<std::int64_t>(system % 2u);
    shells += system_shells;
    atoms += system_atoms;
  }
  data.shell_offsets[batch_size] = shells;
  data.atom_offsets[batch_size] = atoms;

  const auto fill_components = [](std::vector<double>& current, std::vector<double>& mixed,
                                  std::vector<double>& raw, std::vector<double>& published,
                                  std::size_t count, double scale) {
    current.resize(count);
    mixed.resize(count);
    raw.resize(count);
    published.assign(count, -71.0);
    for (std::size_t element = 0u; element < count; ++element) {
      current[element] = scale * (static_cast<double>(element % 17u) - 8.0);
    }
  };
  fill_components(data.current_shell, data.mixed_shell, data.raw_shell, data.published_shell,
                  static_cast<std::size_t>(shells), 0.013);
  fill_components(data.current_dipole, data.mixed_dipole, data.raw_dipole, data.published_dipole,
                  static_cast<std::size_t>(atoms) * 3u, -0.007);
  fill_components(data.current_quadrupole, data.mixed_quadrupole, data.raw_quadrupole,
                  data.published_quadrupole, static_cast<std::size_t>(atoms) * 6u, 0.003);

  for (std::size_t system = 0u; system < batch_size; ++system) {
    const double residual = system % 4u == 0u   ? 1.0e-9
                            : system % 4u == 1u ? 2.0e-5
                            : system % 4u == 2u ? -7.0e-5
                                                : 4.0e-4;
    const auto set_system = [&](std::vector<double>& mixed, std::vector<double>& raw,
                                const std::vector<double>& current, std::int64_t begin,
                                std::int64_t end) {
      for (std::int64_t element = begin; element < end; ++element) {
        const std::size_t index = static_cast<std::size_t>(element);
        raw[index] = current[index] + residual;
        mixed[index] = current[index] + 0.35 * residual;
      }
    };
    set_system(data.mixed_shell, data.raw_shell, data.current_shell, data.shell_offsets[system],
               data.shell_offsets[system + 1u]);
    set_system(data.mixed_dipole, data.raw_dipole, data.current_dipole,
               data.atom_offsets[system] * 3, data.atom_offsets[system + 1u] * 3);
    set_system(data.mixed_quadrupole, data.raw_quadrupole, data.current_quadrupole,
               data.atom_offsets[system] * 6, data.atom_offsets[system + 1u] * 6);
  }

  data.complete_free_energies.resize(batch_size);
  data.free_energies.resize(batch_size);
  data.previous_free_energies.assign(batch_size, 92.0);
  data.free_energy_changes.assign(batch_size, 93.0);
  data.residual_rms.assign(batch_size, 94.0);
  data.iterations.resize(batch_size);
  data.statuses.assign(batch_size, XTBLOOM_STATUS_SUCCESS);
  data.converged.assign(batch_size, 0u);
  for (std::size_t system = 0u; system < batch_size; ++system) {
    data.iterations[system] = static_cast<std::uint64_t>(system % 2u);
    data.free_energies[system] = -0.6 - 0.002 * static_cast<double>(system);
    const double old_energy = data.iterations[system] == 0u ? 0.0 : data.free_energies[system];
    const double delta = system % 4u == 0u ? 1.0e-9 : 3.0e-5;
    data.complete_free_energies[system] = old_energy + delta;
  }
  return data;
}

double residual_square(const HostCase& data, std::size_t system) {
  double sum = 0.0;
  const auto accumulate = [&](const std::vector<double>& current, const std::vector<double>& raw,
                              std::int64_t begin, std::int64_t end) {
    for (std::int64_t element = begin; element < end; ++element) {
      const double residual =
          raw[static_cast<std::size_t>(element)] - current[static_cast<std::size_t>(element)];
      sum += residual * residual;
    }
  };
  accumulate(data.current_shell, data.raw_shell, data.shell_offsets[system],
             data.shell_offsets[system + 1u]);
  accumulate(data.current_dipole, data.raw_dipole, data.atom_offsets[system] * 3,
             data.atom_offsets[system + 1u] * 3);
  accumulate(data.current_quadrupole, data.raw_quadrupole, data.atom_offsets[system] * 6,
             data.atom_offsets[system + 1u] * 6);
  return sum;
}

void simulate_successful_step(HostCase& data, const Gfn2SccDevicePolicy& policy) {
  for (std::size_t system = 0u; system < data.batch_size(); ++system) {
    if (data.statuses[system] != XTBLOOM_STATUS_SUCCESS || data.converged[system] == 1u) {
      continue;
    }
    if (data.iterations[system] >= policy.maximum_iterations) {
      continue;
    }
    const std::int64_t shells = data.shell_offsets[system + 1u] - data.shell_offsets[system];
    const std::int64_t atoms = data.atom_offsets[system + 1u] - data.atom_offsets[system];
    const std::int64_t dimension = shells + 9 * atoms;
    const double rms =
        std::sqrt(residual_square(data, system)) / std::sqrt(static_cast<double>(dimension));
    const double old_energy = data.iterations[system] == 0u ? 0.0 : data.free_energies[system];
    const double delta = data.complete_free_energies[system] - old_energy;
    const bool converged =
        rms < policy.residual_rms_tolerance && std::abs(delta) < policy.energy_tolerance;
    const auto publish = [&](std::vector<double>& current, const std::vector<double>& mixed,
                             const std::vector<double>& raw, std::vector<double>& published,
                             std::int64_t begin, std::int64_t end) {
      for (std::int64_t element = begin; element < end; ++element) {
        const std::size_t index = static_cast<std::size_t>(element);
        current[index] = mixed[index];
        published[index] = converged ? raw[index] : mixed[index];
      }
    };
    publish(data.current_shell, data.mixed_shell, data.raw_shell, data.published_shell,
            data.shell_offsets[system], data.shell_offsets[system + 1u]);
    publish(data.current_dipole, data.mixed_dipole, data.raw_dipole, data.published_dipole,
            data.atom_offsets[system] * 3, data.atom_offsets[system + 1u] * 3);
    publish(data.current_quadrupole, data.mixed_quadrupole, data.raw_quadrupole,
            data.published_quadrupole, data.atom_offsets[system] * 6,
            data.atom_offsets[system + 1u] * 6);
    data.previous_free_energies[system] = old_energy;
    data.free_energies[system] = data.complete_free_energies[system];
    data.free_energy_changes[system] = delta;
    data.residual_rms[system] = rms;
    ++data.iterations[system];
    data.converged[system] = converged ? 1u : 0u;
    data.statuses[system] = !converged && data.iterations[system] >= policy.maximum_iterations
                                ? XTBLOOM_STATUS_SCC_NOT_CONVERGED
                                : XTBLOOM_STATUS_SUCCESS;
  }
}

void simulate_numeric_failure(HostCase& data, std::size_t system) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  data.free_energies[system] = nan;
  data.previous_free_energies[system] = nan;
  data.free_energy_changes[system] = nan;
  data.residual_rms[system] = nan;
  ++data.iterations[system];
  data.statuses[system] = XTBLOOM_STATUS_INTERNAL_ERROR;
}

struct DeviceFixture {
  DeviceBuffer<std::int64_t> shell_offsets;
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<double> current_shell;
  DeviceBuffer<double> current_dipole;
  DeviceBuffer<double> current_quadrupole;
  DeviceBuffer<double> mixed_shell;
  DeviceBuffer<double> mixed_dipole;
  DeviceBuffer<double> mixed_quadrupole;
  DeviceBuffer<double> raw_shell;
  DeviceBuffer<double> raw_dipole;
  DeviceBuffer<double> raw_quadrupole;
  DeviceBuffer<double> published_shell;
  DeviceBuffer<double> published_dipole;
  DeviceBuffer<double> published_quadrupole;
  DeviceBuffer<double> complete_free_energies;
  DeviceBuffer<double> free_energies;
  DeviceBuffer<double> previous_free_energies;
  DeviceBuffer<double> free_energy_changes;
  DeviceBuffer<double> residual_rms;
  DeviceBuffer<std::uint64_t> iterations;
  DeviceBuffer<xtbloom_status_t> statuses;
  DeviceBuffer<std::uint8_t> converged;
  DeviceBuffer<std::uint32_t> sequence_active;
  DeviceBuffer<std::uint32_t> error;
  std::int64_t batch_count = 0;
  std::int64_t shell_count = 0;
  std::int64_t atom_count = 0;

  cudaError_t initialize(const HostCase& host, cudaStream_t stream = nullptr) {
    batch_count = static_cast<std::int64_t>(host.batch_size());
    shell_count = static_cast<std::int64_t>(host.total_shells());
    atom_count = static_cast<std::int64_t>(host.total_atoms());
    cudaError_t status = allocate_and_copy(shell_offsets, host.shell_offsets, stream);
#define COPY_FIELD(name)                                 \
  if (status == cudaSuccess) {                           \
    status = allocate_and_copy(name, host.name, stream); \
  }
    COPY_FIELD(atom_offsets)
    COPY_FIELD(current_shell)
    COPY_FIELD(current_dipole)
    COPY_FIELD(current_quadrupole)
    COPY_FIELD(mixed_shell)
    COPY_FIELD(mixed_dipole)
    COPY_FIELD(mixed_quadrupole)
    COPY_FIELD(raw_shell)
    COPY_FIELD(raw_dipole)
    COPY_FIELD(raw_quadrupole)
    COPY_FIELD(published_shell)
    COPY_FIELD(published_dipole)
    COPY_FIELD(published_quadrupole)
    COPY_FIELD(complete_free_energies)
    COPY_FIELD(free_energies)
    COPY_FIELD(previous_free_energies)
    COPY_FIELD(free_energy_changes)
    COPY_FIELD(residual_rms)
    COPY_FIELD(iterations)
    COPY_FIELD(statuses)
    COPY_FIELD(converged)
#undef COPY_FIELD
    if (status == cudaSuccess) {
      status = sequence_active.allocate(1u);
    }
    if (status == cudaSuccess) {
      status = error.allocate(1u);
    }
    return status;
  }

  Gfn2SccDeviceBatch batch(std::uint64_t token = kPlanToken) const {
    return {batch_count,     shell_count, atom_count,          batch_count + 1,
            batch_count + 1, token,       shell_offsets.get(), atom_offsets.get()};
  }

  Gfn2SccDeviceConstMultipoles mixed(std::uint64_t token = kPlanToken) const {
    return {
        mixed_shell.get(), shell_count, mixed_dipole.get(), atom_count * 3, mixed_quadrupole.get(),
        atom_count * 6,    token};
  }

  Gfn2SccDeviceConstMultipoles raw(std::uint64_t token = kPlanToken) const {
    return {raw_shell.get(), shell_count, raw_dipole.get(), atom_count * 3, raw_quadrupole.get(),
            atom_count * 6,  token};
  }

  Gfn2SccDeviceMultipoles published(bool in_place_mixed = false, std::uint64_t token = kPlanToken) {
    return {in_place_mixed ? mixed_shell.get() : published_shell.get(),
            shell_count,
            in_place_mixed ? mixed_dipole.get() : published_dipole.get(),
            atom_count * 3,
            in_place_mixed ? mixed_quadrupole.get() : published_quadrupole.get(),
            atom_count * 6,
            token};
  }

  Gfn2SccDeviceState state(std::uint64_t token = kPlanToken) {
    const Gfn2SccDeviceMultipoles current{current_shell.get(),
                                          shell_count,
                                          current_dipole.get(),
                                          atom_count * 3,
                                          current_quadrupole.get(),
                                          atom_count * 6,
                                          token};
    return {current,
            free_energies.get(),
            previous_free_energies.get(),
            free_energy_changes.get(),
            residual_rms.get(),
            iterations.get(),
            statuses.get(),
            converged.get(),
            batch_count,
            token};
  }

  Gfn2SccDeviceWorkspace workspace(std::uint64_t token = kPlanToken) {
    return {sequence_active.get(), 1, token};
  }

  cudaError_t copy_results(HostCase& host, std::uint32_t* semantic_error,
                           cudaStream_t stream = nullptr, bool in_place_mixed = false) const {
    cudaError_t status =
        current_shell.copy_to(host.current_shell.data(), host.current_shell.size(), stream);
#define COPY_RESULT(device_name, host_name)                                             \
  if (status == cudaSuccess) {                                                          \
    status = device_name.copy_to(host.host_name.data(), host.host_name.size(), stream); \
  }
    COPY_RESULT(current_dipole, current_dipole)
    COPY_RESULT(current_quadrupole, current_quadrupole)
    if (in_place_mixed) {
      COPY_RESULT(mixed_shell, published_shell)
      COPY_RESULT(mixed_dipole, published_dipole)
      COPY_RESULT(mixed_quadrupole, published_quadrupole)
    } else {
      COPY_RESULT(published_shell, published_shell)
      COPY_RESULT(published_dipole, published_dipole)
      COPY_RESULT(published_quadrupole, published_quadrupole)
    }
    COPY_RESULT(free_energies, free_energies)
    COPY_RESULT(previous_free_energies, previous_free_energies)
    COPY_RESULT(free_energy_changes, free_energy_changes)
    COPY_RESULT(residual_rms, residual_rms)
    COPY_RESULT(iterations, iterations)
    COPY_RESULT(statuses, statuses)
    COPY_RESULT(converged, converged)
#undef COPY_RESULT
    if (status == cudaSuccess) {
      status = error.copy_to(semantic_error, 1u, stream);
    }
    return status;
  }
};

bool near(double actual, double expected, double tolerance = 2.0e-15) {
  return std::abs(actual - expected) <=
         tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

bool same_results(const HostCase& actual, const HostCase& expected) {
  const auto exact_double = [](const std::vector<double>& first,
                               const std::vector<double>& second) {
    if (first.size() != second.size()) {
      return false;
    }
    for (std::size_t index = 0u; index < first.size(); ++index) {
      if (first[index] != second[index] &&
          !(std::isnan(first[index]) && std::isnan(second[index]))) {
        return false;
      }
    }
    return true;
  };
  if (!exact_double(actual.current_shell, expected.current_shell) ||
      !exact_double(actual.current_dipole, expected.current_dipole) ||
      !exact_double(actual.current_quadrupole, expected.current_quadrupole) ||
      !exact_double(actual.published_shell, expected.published_shell) ||
      !exact_double(actual.published_dipole, expected.published_dipole) ||
      !exact_double(actual.published_quadrupole, expected.published_quadrupole) ||
      !exact_double(actual.free_energies, expected.free_energies) ||
      !exact_double(actual.previous_free_energies, expected.previous_free_energies) ||
      !exact_double(actual.free_energy_changes, expected.free_energy_changes) ||
      actual.iterations != expected.iterations || actual.statuses != expected.statuses ||
      actual.converged != expected.converged ||
      actual.residual_rms.size() != expected.residual_rms.size()) {
    return false;
  }
  for (std::size_t system = 0u; system < actual.residual_rms.size(); ++system) {
    if (!(std::isnan(actual.residual_rms[system]) && std::isnan(expected.residual_rms[system])) &&
        !near(actual.residual_rms[system], expected.residual_rms[system], 8.0e-15)) {
      return false;
    }
  }
  return true;
}

int run_and_compare(std::size_t batch_size, cudaStream_t stream) {
  const Gfn2SccDevicePolicy policy{8u, 1.0e-6, 1.0e-7, kPlanToken};
  HostCase input = make_case(batch_size);
  HostCase expected = input;
  simulate_successful_step(expected, policy);
  DeviceFixture fixture;
  CUDA_CHECK(fixture.initialize(input, stream));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_scc_device_error_cuda(fixture.error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_scc_state_cuda(
      fixture.batch(), policy, fixture.mixed(), fixture.raw(), fixture.complete_free_energies.get(),
      fixture.published(), fixture.state(), fixture.workspace(), fixture.error.get(), stream));
  HostCase actual = input;
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(fixture.copy_results(actual, &semantic_error, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2SccDeviceError::kSuccess));
  CHECK(same_results(actual, expected));
  return 0;
}

int test_batch_sizes_and_custom_stream() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (const std::size_t batch_size : {1u, 8u, 32u, 128u}) {
    const int status = run_and_compare(batch_size, stream);
    CHECK(status == 0);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_strict_boundaries_first_iteration_and_maximum() {
  HostCase input = make_case(2u);
  std::fill(input.current_shell.begin(), input.current_shell.end(), 0.0);
  std::fill(input.current_dipole.begin(), input.current_dipole.end(), 0.0);
  std::fill(input.current_quadrupole.begin(), input.current_quadrupole.end(), 0.0);
  std::fill(input.raw_shell.begin(), input.raw_shell.end(), 0.0);
  std::fill(input.raw_dipole.begin(), input.raw_dipole.end(), 0.0);
  std::fill(input.raw_quadrupole.begin(), input.raw_quadrupole.end(), 0.0);
  std::fill(input.mixed_shell.begin(), input.mixed_shell.end(), 0.01);
  std::fill(input.mixed_dipole.begin(), input.mixed_dipole.end(), 0.01);
  std::fill(input.mixed_quadrupole.begin(), input.mixed_quadrupole.end(), 0.01);
  input.raw_shell[0] = 3.0e-5;
  const std::int64_t dimension0 = input.shell_offsets[1] - input.shell_offsets[0] +
                                  9 * (input.atom_offsets[1] - input.atom_offsets[0]);
  const double residual_boundary = std::sqrt(input.raw_shell[0] * input.raw_shell[0]) /
                                   std::sqrt(static_cast<double>(dimension0));
  input.iterations = {0u, 0u};
  input.free_energies = {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN()};
  input.complete_free_energies = {0.01, 0.125};
  input.previous_free_energies = {17.0, 18.0};
  input.free_energy_changes = {19.0, 20.0};
  input.residual_rms = {21.0, 22.0};
  input.statuses = {XTBLOOM_STATUS_SUCCESS, XTBLOOM_STATUS_SUCCESS};
  input.converged = {0u, 0u};
  const Gfn2SccDevicePolicy policy{1u, residual_boundary, 0.125, kPlanToken};
  HostCase expected = input;
  simulate_successful_step(expected, policy);
  CHECK(expected.converged[0] == 0u && expected.converged[1] == 0u);
  CHECK(expected.previous_free_energies[0] == 0.0 && expected.previous_free_energies[1] == 0.0);
  CHECK(expected.statuses[0] == XTBLOOM_STATUS_SCC_NOT_CONVERGED &&
        expected.statuses[1] == XTBLOOM_STATUS_SCC_NOT_CONVERGED);

  DeviceFixture fixture;
  CUDA_CHECK(fixture.initialize(input));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_scc_device_error_cuda(fixture.error.get()));
  CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_scc_state_cuda(
      fixture.batch(), policy, fixture.mixed(), fixture.raw(), fixture.complete_free_energies.get(),
      fixture.published(), fixture.state(), fixture.workspace(), fixture.error.get()));
  HostCase actual = input;
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(fixture.copy_results(actual, &semantic_error));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error == 0u);
  CHECK(same_results(actual, expected));
  return 0;
}

int test_failure_isolation_terminal_skip_and_sticky_error() {
  const Gfn2SccDevicePolicy policy{8u, 1.0e-6, 1.0e-7, kPlanToken};
  HostCase input = make_case(4u);
  const std::size_t failed_shell = static_cast<std::size_t>(input.shell_offsets[1]);
  const double saved_failed_raw = input.raw_shell[failed_shell];
  input.raw_shell[failed_shell] = std::numeric_limits<double>::quiet_NaN();
  input.converged[2] = 1u;
  input.raw_shell[static_cast<std::size_t>(input.shell_offsets[2])] =
      std::numeric_limits<double>::quiet_NaN();
  input.statuses[3] = XTBLOOM_STATUS_SCC_NOT_CONVERGED;
  input.raw_shell[static_cast<std::size_t>(input.shell_offsets[3])] =
      std::numeric_limits<double>::quiet_NaN();

  HostCase expected = input;
  simulate_numeric_failure(expected, 1u);
  simulate_successful_step(expected, policy);

  DeviceFixture fixture;
  CUDA_CHECK(fixture.initialize(input));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_scc_device_error_cuda(fixture.error.get()));
  CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_scc_state_cuda(
      fixture.batch(), policy, fixture.mixed(), fixture.raw(), fixture.complete_free_energies.get(),
      fixture.published(), fixture.state(), fixture.workspace(), fixture.error.get()));
  HostCase actual = input;
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(fixture.copy_results(actual, &semantic_error));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2SccDeviceError::kNonfiniteRawMultipole));
  CHECK(same_results(actual, expected));

  /* Without reset the sequence is a complete no-op even after inputs recover. */
  HostCase sticky_seed = actual;
  std::vector<double> restored_raw_shell = input.raw_shell;
  restored_raw_shell[failed_shell] = saved_failed_raw;
  CUDA_CHECK(fixture.raw_shell.copy_from(restored_raw_shell.data(), restored_raw_shell.size()));
  CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_scc_state_cuda(
      fixture.batch(), policy, fixture.mixed(), fixture.raw(), fixture.complete_free_energies.get(),
      fixture.published(), fixture.state(), fixture.workspace(), fixture.error.get()));
  HostCase sticky_actual = actual;
  CUDA_CHECK(fixture.copy_results(sticky_actual, &semantic_error));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(same_results(sticky_actual, sticky_seed));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2SccDeviceError::kNonfiniteRawMultipole));
  return 0;
}

int test_each_numeric_failure_matches_cpu_attempt_accounting() {
  enum class FailureKind { kCurrent, kMixed, kRaw, kFreeEnergy, kResidual };
  struct FailureCase {
    FailureKind kind;
    Gfn2SccDeviceError error;
  };
  const std::vector<FailureCase> failures{
      {FailureKind::kCurrent, Gfn2SccDeviceError::kNonfiniteCurrentMultipole},
      {FailureKind::kMixed, Gfn2SccDeviceError::kNonfiniteMixedMultipole},
      {FailureKind::kRaw, Gfn2SccDeviceError::kNonfiniteRawMultipole},
      {FailureKind::kFreeEnergy, Gfn2SccDeviceError::kNonfiniteFreeEnergy},
      {FailureKind::kResidual, Gfn2SccDeviceError::kNonfiniteResidual},
  };
  const Gfn2SccDevicePolicy policy{8u, 1.0e-6, 1.0e-7, kPlanToken};

  for (const FailureCase& failure : failures) {
    HostCase input = make_case(2u);
    const std::size_t shell = static_cast<std::size_t>(input.shell_offsets[0]);
    switch (failure.kind) {
      case FailureKind::kCurrent:
        input.current_shell[shell] = std::numeric_limits<double>::quiet_NaN();
        break;
      case FailureKind::kMixed:
        input.mixed_shell[shell] = std::numeric_limits<double>::infinity();
        break;
      case FailureKind::kRaw:
        input.raw_shell[shell] = -std::numeric_limits<double>::infinity();
        break;
      case FailureKind::kFreeEnergy:
        input.complete_free_energies[0] = std::numeric_limits<double>::quiet_NaN();
        break;
      case FailureKind::kResidual:
        input.current_shell[shell] = std::numeric_limits<double>::max();
        input.raw_shell[shell] = -std::numeric_limits<double>::max();
        input.mixed_shell[shell] = 0.0;
        break;
    }

    HostCase expected = input;
    const std::uint64_t old_iteration = expected.iterations[0];
    const std::uint8_t old_converged = expected.converged[0];
    const std::vector<double> old_current_shell = expected.current_shell;
    const std::vector<double> old_published_shell = expected.published_shell;
    simulate_numeric_failure(expected, 0u);
    simulate_successful_step(expected, policy);
    CHECK(expected.iterations[0] == old_iteration + 1u);
    CHECK(expected.converged[0] == old_converged);
    const auto unchanged = [](double current, double previous) {
      return current == previous || (std::isnan(current) && std::isnan(previous));
    };
    /* The failed member publishes no multipoles. The healthy peer is expected
     * to commit in the same launch, so compare only the failed system slice. */
    for (std::int64_t element = expected.shell_offsets[0]; element < expected.shell_offsets[1];
         ++element) {
      const std::size_t index = static_cast<std::size_t>(element);
      CHECK(unchanged(expected.current_shell[index], old_current_shell[index]));
      CHECK(unchanged(expected.published_shell[index], old_published_shell[index]));
    }
    CHECK(std::isnan(expected.free_energies[0]));
    CHECK(std::isnan(expected.previous_free_energies[0]));
    CHECK(std::isnan(expected.free_energy_changes[0]));
    CHECK(std::isnan(expected.residual_rms[0]));

    DeviceFixture fixture;
    CUDA_CHECK(fixture.initialize(input));
    CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_scc_device_error_cuda(fixture.error.get()));
    CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_scc_state_cuda(
        fixture.batch(), policy, fixture.mixed(), fixture.raw(),
        fixture.complete_free_energies.get(), fixture.published(), fixture.state(),
        fixture.workspace(), fixture.error.get()));
    HostCase actual = input;
    std::uint32_t semantic_error = 99u;
    CUDA_CHECK(fixture.copy_results(actual, &semantic_error));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(semantic_error == static_cast<std::uint32_t>(failure.error));
    CHECK(same_results(actual, expected));
  }
  return 0;
}

int test_maximum_iteration_entry_matches_cpu_skip() {
  const Gfn2SccDevicePolicy policy{4u, 1.0e-6, 1.0e-7, kPlanToken};
  HostCase input = make_case(2u);
  input.iterations[0] = policy.maximum_iterations;
  input.statuses[0] = XTBLOOM_STATUS_SUCCESS;
  input.converged[0] = 0u;
  input.raw_shell[static_cast<std::size_t>(input.shell_offsets[0])] =
      std::numeric_limits<double>::quiet_NaN();
  HostCase expected = input;
  simulate_successful_step(expected, policy);
  CHECK(expected.statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(expected.iterations[0] == policy.maximum_iterations);

  DeviceFixture fixture;
  CUDA_CHECK(fixture.initialize(input));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_scc_device_error_cuda(fixture.error.get()));
  CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_scc_state_cuda(
      fixture.batch(), policy, fixture.mixed(), fixture.raw(), fixture.complete_free_energies.get(),
      fixture.published(), fixture.state(), fixture.workspace(), fixture.error.get()));
  HostCase actual = input;
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(fixture.copy_results(actual, &semantic_error));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error == 0u);
  CHECK(same_results(actual, expected));
  return 0;
}

int test_unknown_status_failure_isolation() {
  const Gfn2SccDevicePolicy policy{8u, 1.0e-6, 1.0e-7, kPlanToken};
  HostCase input = make_case(2u);
  input.statuses[0] = 12345;
  HostCase expected = input;
  expected.statuses[0] = XTBLOOM_STATUS_INTERNAL_ERROR;
  simulate_successful_step(expected, policy);

  DeviceFixture fixture;
  CUDA_CHECK(fixture.initialize(input));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_scc_device_error_cuda(fixture.error.get()));
  CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_scc_state_cuda(
      fixture.batch(), policy, fixture.mixed(), fixture.raw(), fixture.complete_free_energies.get(),
      fixture.published(), fixture.state(), fixture.workspace(), fixture.error.get()));
  HostCase actual = input;
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(fixture.copy_results(actual, &semantic_error));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2SccDeviceError::kInvalidState));
  CHECK(same_results(actual, expected));
  return 0;
}

int test_invalid_device_offsets_are_whole_call_atomic() {
  const Gfn2SccDevicePolicy policy{8u, 1.0e-6, 1.0e-7, kPlanToken};
  HostCase input = make_case(3u);
  DeviceFixture fixture;
  CUDA_CHECK(fixture.initialize(input));
  std::vector<std::int64_t> bad_offsets = input.shell_offsets;
  bad_offsets[1] = static_cast<std::int64_t>(input.total_shells()) + 1;
  CUDA_CHECK(fixture.shell_offsets.copy_from(bad_offsets.data(), bad_offsets.size()));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_scc_device_error_cuda(fixture.error.get()));
  CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_scc_state_cuda(
      fixture.batch(), policy, fixture.mixed(), fixture.raw(), fixture.complete_free_energies.get(),
      fixture.published(), fixture.state(), fixture.workspace(), fixture.error.get()));
  HostCase actual = input;
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(fixture.copy_results(actual, &semantic_error));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2SccDeviceError::kInvalidOffsets));
  CHECK(same_results(actual, input));
  return 0;
}

int test_host_validation_provenance_and_aliases() {
  const Gfn2SccDevicePolicy policy{8u, 1.0e-6, 1.0e-7, kPlanToken};
  HostCase input = make_case(2u);
  DeviceFixture fixture;
  CUDA_CHECK(fixture.initialize(input));

  /*
   * Pointer provenance is validated once by the setup layer, not by every hot
   * launch. Exercise one pointer from each descriptor category here so the
   * fixture proves the native kernel contract uses device-resident topology,
   * inputs, outputs, persistent state, scratch, and sticky error storage.
   */
  const std::vector<const void*> device_ranges{
      fixture.shell_offsets.get(), fixture.mixed_shell.get(),
      fixture.raw_shell.get(),     fixture.published_shell.get(),
      fixture.current_shell.get(), fixture.complete_free_energies.get(),
      fixture.statuses.get(),      fixture.sequence_active.get(),
      fixture.error.get(),
  };
  for (const void* pointer : device_ranges) {
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
    CHECK(attributes.type == cudaMemoryTypeDevice);
  }

  CHECK(xtbloom::detail::cuda::update_gfn2_scc_state_cuda(
            fixture.batch(kPlanToken + 1u), policy, fixture.mixed(), fixture.raw(),
            fixture.complete_free_energies.get(), fixture.published(), fixture.state(),
            fixture.workspace(), fixture.error.get()) == cudaErrorInvalidValue);
  CHECK(xtbloom::detail::cuda::update_gfn2_scc_state_cuda(
            fixture.batch(), policy, fixture.mixed(kPlanToken + 1u), fixture.raw(),
            fixture.complete_free_energies.get(), fixture.published(), fixture.state(),
            fixture.workspace(), fixture.error.get()) == cudaErrorInvalidValue);
  CHECK(xtbloom::detail::cuda::update_gfn2_scc_state_cuda(
            fixture.batch(), policy, fixture.mixed(), fixture.raw(),
            fixture.complete_free_energies.get(), fixture.published(),
            fixture.state(kPlanToken + 1u), fixture.workspace(),
            fixture.error.get()) == cudaErrorInvalidValue);

  Gfn2SccDeviceMultipoles partial_alias = fixture.published();
  partial_alias.shell_charges = fixture.mixed_shell.get() + 1;
  CHECK(xtbloom::detail::cuda::update_gfn2_scc_state_cuda(
            fixture.batch(), policy, fixture.mixed(), fixture.raw(),
            fixture.complete_free_energies.get(), partial_alias, fixture.state(),
            fixture.workspace(), fixture.error.get()) == cudaErrorInvalidValue);

  /* Exact corresponding next-mixed aliases are the supported in-place form. */
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_scc_device_error_cuda(fixture.error.get()));
  CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_scc_state_cuda(
      fixture.batch(), policy, fixture.mixed(), fixture.raw(), fixture.complete_free_energies.get(),
      fixture.published(true), fixture.state(), fixture.workspace(), fixture.error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

int test_independent_streams() {
  cudaStream_t first_stream = nullptr;
  cudaStream_t second_stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&first_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&second_stream, cudaStreamNonBlocking));
  const Gfn2SccDevicePolicy policy{8u, 1.0e-6, 1.0e-7, kPlanToken};
  HostCase first_input = make_case(8u);
  HostCase second_input = make_case(32u);
  HostCase first_expected = first_input;
  HostCase second_expected = second_input;
  simulate_successful_step(first_expected, policy);
  simulate_successful_step(second_expected, policy);
  DeviceFixture first;
  DeviceFixture second;
  CUDA_CHECK(first.initialize(first_input, first_stream));
  CUDA_CHECK(second.initialize(second_input, second_stream));
  CUDA_CHECK(
      xtbloom::detail::cuda::reset_gfn2_scc_device_error_cuda(first.error.get(), first_stream));
  CUDA_CHECK(
      xtbloom::detail::cuda::reset_gfn2_scc_device_error_cuda(second.error.get(), second_stream));
  CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_scc_state_cuda(
      first.batch(), policy, first.mixed(), first.raw(), first.complete_free_energies.get(),
      first.published(), first.state(), first.workspace(), first.error.get(), first_stream));
  CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_scc_state_cuda(
      second.batch(), policy, second.mixed(), second.raw(), second.complete_free_energies.get(),
      second.published(), second.state(), second.workspace(), second.error.get(), second_stream));
  HostCase first_actual = first_input;
  HostCase second_actual = second_input;
  std::uint32_t first_error = 99u;
  std::uint32_t second_error = 99u;
  CUDA_CHECK(first.copy_results(first_actual, &first_error, first_stream));
  CUDA_CHECK(second.copy_results(second_actual, &second_error, second_stream));
  CUDA_CHECK(cudaStreamSynchronize(first_stream));
  CUDA_CHECK(cudaStreamSynchronize(second_stream));
  CHECK(first_error == 0u && second_error == 0u);
  CHECK(same_results(first_actual, first_expected));
  CHECK(same_results(second_actual, second_expected));
  CUDA_CHECK(cudaStreamDestroy(first_stream));
  CUDA_CHECK(cudaStreamDestroy(second_stream));
  return 0;
}

int test_cuda_graph_double_replay() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const Gfn2SccDevicePolicy policy{10u, 1.0e-12, 1.0e-12, kPlanToken};
  HostCase input = make_case(8u);
  for (std::size_t system = 0u; system < input.batch_size(); ++system) {
    input.iterations[system] = 0u;
    input.free_energies[system] = -3.0;
    input.complete_free_energies[system] = 0.01 + 0.001 * static_cast<double>(system);
  }
  HostCase expected = input;
  simulate_successful_step(expected, policy);
  simulate_successful_step(expected, policy);

  DeviceFixture fixture;
  CUDA_CHECK(fixture.initialize(input, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_scc_device_error_cuda(fixture.error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_scc_state_cuda(
      fixture.batch(), policy, fixture.mixed(), fixture.raw(), fixture.complete_free_energies.get(),
      fixture.published(true), fixture.state(), fixture.workspace(), fixture.error.get(), stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0u));
  CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
  CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
  HostCase actual = input;
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(fixture.copy_results(actual, &semantic_error, stream, true));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == 0u);
  CHECK(same_results(actual, expected));
  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_batch_sizes_and_custom_stream(); status != 0) {
    return status;
  }
  if (const int status = test_strict_boundaries_first_iteration_and_maximum(); status != 0) {
    return status;
  }
  if (const int status = test_failure_isolation_terminal_skip_and_sticky_error(); status != 0) {
    return status;
  }
  if (const int status = test_each_numeric_failure_matches_cpu_attempt_accounting(); status != 0) {
    return status;
  }
  if (const int status = test_maximum_iteration_entry_matches_cpu_skip(); status != 0) {
    return status;
  }
  if (const int status = test_unknown_status_failure_isolation(); status != 0) {
    return status;
  }
  if (const int status = test_invalid_device_offsets_are_whole_call_atomic(); status != 0) {
    return status;
  }
  if (const int status = test_host_validation_provenance_and_aliases(); status != 0) {
    return status;
  }
  if (const int status = test_independent_streams(); status != 0) {
    return status;
  }
  return test_cuda_graph_double_replay();
}
