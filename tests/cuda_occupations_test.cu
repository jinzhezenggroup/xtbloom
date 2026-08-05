#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_occupations.cuh"
#include "gpuxtb/gpuxtb.h"
#include "model/gfn2/eigensolver.hpp"

#define CHECK(condition)                                                                           \
  do {                                                                                             \
    if (!(condition)) {                                                                            \
      std::fprintf(stderr, "CUDA occupations test failed at line %d: %s\n", __LINE__, #condition); \
      return __LINE__;                                                                             \
    }                                                                                              \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using gpuxtb::detail::cuda::Gfn2OccupationsDeviceBatch;
using gpuxtb::detail::cuda::Gfn2OccupationsDeviceError;
using gpuxtb::detail::cuda::Gfn2OccupationsDeviceResults;
using gpuxtb::detail::cuda::Gfn2OccupationsDeviceWorkspace;

constexpr std::uint64_t kPlanToken = 0x3c6ef372fe94f82bULL;
constexpr double kSentinel = -771.25;

static_assert(std::is_trivially_copyable_v<Gfn2OccupationsDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2OccupationsDeviceResults>);
static_assert(std::is_trivially_copyable_v<Gfn2OccupationsDeviceWorkspace>);

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <=
         tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

bool within_ulps(double actual, double expected, int ulps) {
  if (!std::isfinite(actual) || !std::isfinite(expected) || ulps < 0) {
    return false;
  }
  double lower = expected;
  double upper = expected;
  for (int step = 0; step < ulps; ++step) {
    lower = std::nextafter(lower, -std::numeric_limits<double>::infinity());
    upper = std::nextafter(upper, std::numeric_limits<double>::infinity());
  }
  return actual >= lower && actual <= upper;
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0u)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0u);
    }
    return *this;
  }
  ~DeviceBuffer() { release(); }

  cudaError_t allocate(std::size_t count) {
    release();
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

  T* get() { return data_; }
  const T* get() const { return data_; }
  std::size_t size() const { return count_; }

 private:
  void release() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
    data_ = nullptr;
    count_ = 0u;
  }

  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

template <typename T>
cudaError_t allocate_and_copy(DeviceBuffer<T>& device, const std::vector<T>& host,
                              cudaStream_t stream) {
  const cudaError_t status = device.allocate(host.size());
  return status == cudaSuccess ? device.copy_from(host.data(), host.size(), stream) : status;
}

struct HostCase {
  std::vector<std::int64_t> offsets;
  std::vector<std::int32_t> spin_channels;
  std::vector<std::int64_t> spin_channel_offsets;
  std::vector<std::int64_t> spin_orbital_offsets;
  std::vector<double> eigenvalues;
  std::vector<double> electron_counts;
  std::vector<double> temperatures;
  std::vector<std::uint8_t> active;
  std::vector<double> expected_occupations;
  std::vector<double> expected_chemical_potentials;
  std::vector<double> expected_electron_sums;
  std::vector<double> expected_entropies;

  std::size_t batch_size() const { return offsets.size() - 1u; }
  std::size_t total_orbitals() const { return static_cast<std::size_t>(offsets.back()); }
  bool spin_layout() const { return !spin_channels.empty(); }
};

bool build_cpu_reference(HostCase& host, std::string& error) {
  host.expected_occupations.assign(2u * host.total_orbitals(), kSentinel);
  host.expected_chemical_potentials.assign(2u * host.batch_size(), kSentinel);
  host.expected_electron_sums.assign(2u * host.batch_size(), kSentinel);
  host.expected_entropies.assign(host.batch_size(), kSentinel);
  for (std::size_t system = 0u; system < host.batch_size(); ++system) {
    if (host.active[system] == 0u) {
      continue;
    }
    const std::int64_t begin = host.offsets[system];
    const std::int64_t end = host.offsets[system + 1u];
    const std::int64_t count = end - begin;
    const std::int64_t spectrum_begin =
        host.spin_layout() ? host.spin_orbital_offsets[system] : begin;
    const std::uint8_t channels = host.spin_layout() ? host.spin_channels[system] : 1u;
    double total_entropy = 0.0;
    for (int spin = 0; spin < 2; ++spin) {
      double mu = 0.0;
      double entropy = 0.0;
      double* const output = host.expected_occupations.data() + 2 * begin + spin * count;
      const std::int64_t spin_spectrum_begin =
          spectrum_begin + (channels == 2u ? static_cast<std::int64_t>(spin) * count : 0);
      if (gpuxtb::detail::gfn2::fill_occupations_cpu(
              count, host.eigenvalues.data() + spin_spectrum_begin,
              host.electron_counts[2u * system + spin], host.temperatures[system], output, mu,
              entropy, error) != GPUXTB_STATUS_SUCCESS) {
        return false;
      }
      host.expected_chemical_potentials[2u * system + spin] = mu;
      host.expected_electron_sums[2u * system + spin] =
          std::accumulate(output, output + count, 0.0);
      total_entropy += entropy;
    }
    host.expected_entropies[system] = total_entropy;
  }
  return true;
}

HostCase make_regular_case(std::size_t batch_size) {
  HostCase host;
  host.offsets.assign(batch_size + 1u, 0);
  std::int64_t total = 0;
  for (std::size_t system = 0u; system < batch_size; ++system) {
    host.offsets[system] = total;
    total += batch_size == 1u ? 6 : 2 + static_cast<std::int64_t>(system % 7u);
  }
  host.offsets[batch_size] = total;
  host.eigenvalues.resize(static_cast<std::size_t>(total));
  host.electron_counts.resize(2u * batch_size);
  host.temperatures.resize(batch_size);
  host.active.assign(batch_size, 1u);
  for (std::size_t system = 0u; system < batch_size; ++system) {
    const std::int64_t begin = host.offsets[system];
    const std::int64_t end = host.offsets[system + 1u];
    const std::int64_t count = end - begin;
    for (std::int64_t orbital = 0; orbital < count; ++orbital) {
      double value =
          -0.7 + 0.23 * static_cast<double>(orbital) + 0.003 * static_cast<double>(system);
      if (system % 5u == 1u) {
        value = 0.125 + 0.003 * static_cast<double>(system);
      } else if (system % 5u == 2u && orbital > 0) {
        value = host.eigenvalues[static_cast<std::size_t>(begin)] +
                1.0e-13 * static_cast<double>(orbital);
      } else if (system % 5u == 3u) {
        value += 100.0;
      }
      host.eigenvalues[static_cast<std::size_t>(begin + orbital)] = value;
    }
    host.temperatures[system] =
        system % 3u == 0u ? 0.0 : (system % 3u == 1u ? 9.50043469e-4 : 0.02);
    const double capacity = static_cast<double>(count);
    host.electron_counts[2u * system] =
        system % 7u == 0u
            ? 0.0
            : (system % 7u == 1u ? capacity : std::min(capacity, 0.4 + (system % 6u)));
    host.electron_counts[2u * system + 1u] =
        system % 6u == 0u ? std::min(capacity, 0.5)
                          : std::min(capacity, 0.2 + static_cast<double>(system % 5u));
    if (batch_size > 1u && system % 13u == 12u) {
      host.active[system] = 0u;
      host.eigenvalues[static_cast<std::size_t>(begin)] = std::numeric_limits<double>::quiet_NaN();
      host.electron_counts[2u * system] = std::numeric_limits<double>::infinity();
      host.temperatures[system] = -1.0;
    }
  }
  return host;
}

struct DeviceFixture {
  DeviceBuffer<std::int64_t> offsets;
  DeviceBuffer<std::int32_t> spin_channels;
  DeviceBuffer<std::int64_t> spin_channel_offsets;
  DeviceBuffer<std::int64_t> spin_orbital_offsets;
  DeviceBuffer<double> eigenvalues;
  DeviceBuffer<double> electron_counts;
  DeviceBuffer<double> temperatures;
  DeviceBuffer<std::uint8_t> active;
  DeviceBuffer<double> occupations;
  DeviceBuffer<double> chemical_potentials;
  DeviceBuffer<double> electron_sums;
  DeviceBuffer<double> entropies;
  DeviceBuffer<double> occupation_scratch;
  DeviceBuffer<double> chemical_potential_scratch;
  DeviceBuffer<double> electron_sum_scratch;
  DeviceBuffer<double> entropy_scratch;
  DeviceBuffer<std::uint32_t> sequence_active;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;

  cudaError_t initialize(const HostCase& host, cudaStream_t stream) {
    cudaError_t status = allocate_and_copy(offsets, host.offsets, stream);
    if (status == cudaSuccess) {
      status = allocate_and_copy(eigenvalues, host.eigenvalues, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(electron_counts, host.electron_counts, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(temperatures, host.temperatures, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(active, host.active, stream);
    }
    if (status == cudaSuccess && host.spin_layout()) {
      status = allocate_and_copy(spin_channels, host.spin_channels, stream);
    }
    if (status == cudaSuccess && host.spin_layout()) {
      status = allocate_and_copy(spin_channel_offsets, host.spin_channel_offsets, stream);
    }
    if (status == cudaSuccess && host.spin_layout()) {
      status = allocate_and_copy(spin_orbital_offsets, host.spin_orbital_offsets, stream);
    }
    const std::size_t occupation_count = 2u * host.total_orbitals();
    const std::size_t spin_count = 2u * host.batch_size();
    if (status == cudaSuccess) {
      status = occupations.allocate(occupation_count);
    }
    if (status == cudaSuccess) {
      status = chemical_potentials.allocate(spin_count);
    }
    if (status == cudaSuccess) {
      status = electron_sums.allocate(spin_count);
    }
    if (status == cudaSuccess) {
      status = entropies.allocate(host.batch_size());
    }
    if (status == cudaSuccess) {
      status = occupation_scratch.allocate(occupation_count);
    }
    if (status == cudaSuccess) {
      status = chemical_potential_scratch.allocate(spin_count);
    }
    if (status == cudaSuccess) {
      status = electron_sum_scratch.allocate(spin_count);
    }
    if (status == cudaSuccess) {
      status = entropy_scratch.allocate(host.batch_size());
    }
    if (status == cudaSuccess) {
      status = sequence_active.allocate(1u);
    }
    if (status == cudaSuccess) {
      status = system_errors.allocate(host.batch_size());
    }
    if (status == cudaSuccess) {
      status = device_error.allocate(1u);
    }
    return status == cudaSuccess ? reset_outputs(host, stream) : status;
  }

  cudaError_t reset_outputs(const HostCase& host, cudaStream_t stream) {
    const std::vector<double> occupation_seed(2u * host.total_orbitals(), kSentinel);
    const std::vector<double> spin_seed(2u * host.batch_size(), kSentinel);
    const std::vector<double> entropy_seed(host.batch_size(), kSentinel);
    cudaError_t status =
        occupations.copy_from(occupation_seed.data(), occupation_seed.size(), stream);
    if (status == cudaSuccess) {
      status = chemical_potentials.copy_from(spin_seed.data(), spin_seed.size(), stream);
    }
    if (status == cudaSuccess) {
      status = electron_sums.copy_from(spin_seed.data(), spin_seed.size(), stream);
    }
    if (status == cudaSuccess) {
      status = entropies.copy_from(entropy_seed.data(), entropy_seed.size(), stream);
    }
    return status;
  }

  Gfn2OccupationsDeviceBatch batch(const HostCase& host) const {
    Gfn2OccupationsDeviceBatch result{static_cast<std::int64_t>(host.batch_size()),
                                      static_cast<std::int64_t>(host.total_orbitals()),
                                      static_cast<std::int64_t>(host.offsets.size()),
                                      static_cast<std::int64_t>(host.electron_counts.size()),
                                      static_cast<std::int64_t>(host.temperatures.size()),
                                      static_cast<std::int64_t>(host.active.size()),
                                      kPlanToken,
                                      offsets.get(),
                                      electron_counts.get(),
                                      temperatures.get(),
                                      active.get()};
    return result;
  }

  gpuxtb::detail::Gfn2WavefunctionLayoutView layout(const HostCase& host) const {
    gpuxtb::detail::Gfn2WavefunctionLayoutView result{};
    result.memory_space = gpuxtb::detail::Gfn2PlanMemorySpace::kCudaDevice;
    result.plan_token = kPlanToken;
    result.batch_size = static_cast<std::int64_t>(host.batch_size());
    result.total_spin_channels = host.spin_channel_offsets.back();
    result.total_spin_orbitals = host.spin_orbital_offsets.back();
    result.spin_channel_count = static_cast<std::int64_t>(host.spin_channels.size());
    result.spin_channel_offset_count = static_cast<std::int64_t>(host.spin_channel_offsets.size());
    result.spin_orbital_offset_count = static_cast<std::int64_t>(host.spin_orbital_offsets.size());
    result.spin_channels = spin_channels.get();
    result.spin_channel_offsets = spin_channel_offsets.get();
    result.spin_orbital_offsets = spin_orbital_offsets.get();
    return result;
  }

  Gfn2OccupationsDeviceResults results(const HostCase& host) {
    (void)host;
    return {occupations.get(),
            static_cast<std::int64_t>(occupations.size()),
            chemical_potentials.get(),
            static_cast<std::int64_t>(chemical_potentials.size()),
            electron_sums.get(),
            static_cast<std::int64_t>(electron_sums.size()),
            entropies.get(),
            static_cast<std::int64_t>(entropies.size()),
            kPlanToken};
  }

  Gfn2OccupationsDeviceWorkspace workspace() {
    return {occupation_scratch.get(),
            static_cast<std::int64_t>(occupation_scratch.size()),
            chemical_potential_scratch.get(),
            static_cast<std::int64_t>(chemical_potential_scratch.size()),
            electron_sum_scratch.get(),
            static_cast<std::int64_t>(electron_sum_scratch.size()),
            entropy_scratch.get(),
            static_cast<std::int64_t>(entropy_scratch.size()),
            sequence_active.get(),
            static_cast<std::int64_t>(sequence_active.size()),
            kPlanToken};
  }
};

struct Results {
  std::vector<double> occupations;
  std::vector<double> chemical_potentials;
  std::vector<double> electron_sums;
  std::vector<double> entropies;
  std::vector<std::uint32_t> system_errors;
  std::uint32_t device_error = 0u;
};

cudaError_t copy_results(const HostCase& host, const DeviceFixture& device, Results& results,
                         cudaStream_t stream) {
  results.occupations.resize(2u * host.total_orbitals());
  results.chemical_potentials.resize(2u * host.batch_size());
  results.electron_sums.resize(2u * host.batch_size());
  results.entropies.resize(host.batch_size());
  results.system_errors.resize(host.batch_size());
  cudaError_t status =
      device.occupations.copy_to(results.occupations.data(), results.occupations.size(), stream);
  if (status == cudaSuccess) {
    status = device.chemical_potentials.copy_to(results.chemical_potentials.data(),
                                                results.chemical_potentials.size(), stream);
  }
  if (status == cudaSuccess) {
    status = device.electron_sums.copy_to(results.electron_sums.data(),
                                          results.electron_sums.size(), stream);
  }
  if (status == cudaSuccess) {
    status = device.entropies.copy_to(results.entropies.data(), results.entropies.size(), stream);
  }
  if (status == cudaSuccess) {
    status = device.system_errors.copy_to(results.system_errors.data(),
                                          results.system_errors.size(), stream);
  }
  if (status == cudaSuccess) {
    status = device.device_error.copy_to(&results.device_error, 1u, stream);
  }
  return status;
}

int compare_success(const HostCase& host, const Results& actual) {
  if (actual.device_error != 0u) {
    std::fprintf(stderr, "device error %u; system errors:", actual.device_error);
    for (const std::uint32_t value : actual.system_errors) {
      std::fprintf(stderr, " %u", value);
    }
    std::fprintf(stderr, "\n");
  }
  CHECK(actual.device_error == 0u);
  CHECK(std::all_of(actual.system_errors.begin(), actual.system_errors.end(),
                    [](std::uint32_t value) { return value == 0u; }));
  for (std::size_t system = 0u; system < host.batch_size(); ++system) {
    const std::int64_t begin = host.offsets[system];
    const std::int64_t end = host.offsets[system + 1u];
    if (host.active[system] == 0u) {
      for (std::int64_t element = 2 * begin; element < 2 * end; ++element) {
        CHECK(actual.occupations[static_cast<std::size_t>(element)] == kSentinel);
      }
      CHECK(actual.chemical_potentials[2u * system] == kSentinel);
      CHECK(actual.chemical_potentials[2u * system + 1u] == kSentinel);
      CHECK(actual.electron_sums[2u * system] == kSentinel);
      CHECK(actual.electron_sums[2u * system + 1u] == kSentinel);
      CHECK(actual.entropies[system] == kSentinel);
      continue;
    }
    for (std::int64_t element = 2 * begin; element < 2 * end; ++element) {
      CHECK(near(actual.occupations[static_cast<std::size_t>(element)],
                 host.expected_occupations[static_cast<std::size_t>(element)], 4.0e-13));
    }
    for (int spin = 0; spin < 2; ++spin) {
      if (!near(actual.chemical_potentials[2u * system + spin],
                host.expected_chemical_potentials[2u * system + spin], 8.0e-12)) {
        std::fprintf(stderr, "mu mismatch system %zu spin %d actual %.17g expected %.17g\n", system,
                     spin, actual.chemical_potentials[2u * system + spin],
                     host.expected_chemical_potentials[2u * system + spin]);
      }
      CHECK(near(actual.chemical_potentials[2u * system + spin],
                 host.expected_chemical_potentials[2u * system + spin], 8.0e-12));
      CHECK(near(actual.electron_sums[2u * system + spin],
                 host.expected_electron_sums[2u * system + spin], 4.0e-13));
    }
    CHECK(near(actual.entropies[system], host.expected_entropies[system], 8.0e-12));
  }
  return 0;
}

int test_cpu_parity_ragged_batches_and_custom_stream() {
  for (const std::size_t batch_size : {1u, 8u, 32u, 128u}) {
    HostCase host = make_regular_case(batch_size);
    std::string error;
    CHECK(build_cpu_reference(host, error));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    DeviceFixture device;
    CUDA_CHECK(device.initialize(host, stream));
    CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_occupations_device_errors_cuda(
        static_cast<std::int64_t>(batch_size), device.system_errors.get(),
        device.device_error.get(), stream));
    CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_occupations_cuda(
        device.batch(host), device.eigenvalues.get(),
        static_cast<std::int64_t>(device.eigenvalues.size()), device.results(host),
        device.workspace(), device.system_errors.get(), device.device_error.get(), stream));
    Results actual;
    CUDA_CHECK(copy_results(host, device, actual, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int comparison = compare_success(host, actual);
    CHECK(comparison == 0);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
  return 0;
}

/* Degenerate finite-temperature representability policy parity (#31).
 *
 * Exactly degenerate blocks whose single-hole/electron surplus cannot be
 * expressed as equal binary64 occupations are relaxed to the nearest
 * representable symmetric state instead of failing, with an absolute error
 * bounded by count * 2 * eps_double. Both the CPU reference and the device
 * kernel must agree on the published (equal-member) occupations. */
HostCase make_representability_case() {
  HostCase host;
  const std::vector<std::vector<double>> levels{{
      {1.0, 1.0, 1.0},
      {1.0, 1.0, 1.0},
      {1.0, 1.0},
      {0.0, 1.0, 1.0, 2.0},
      {1.0, 1.0, 1.0, 1.0},
      {0.0, 1.0, 2.0},
      {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
      {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
       std::numeric_limits<double>::max()},
      {1.0, 1.0},
      {0.0, 1.0, 1.0, 1.0},
      {0.0, 1.0, 1.0, 1.0, 1.0, 1.0},
      {0.0, 1.0, 1.0},
      {0.0, 0.0, 1.0},
      {0.0, 0.0, 1.0},
  }};
  const std::vector<double> electron0{{
      std::nextafter(3.0, 0.0),
      std::nextafter(0.0, 1.0),
      std::nextafter(2.0, 0.0),
      1.3,
      std::nextafter(4.0, 0.0),
      std::nextafter(3.0, 0.0),
      std::nextafter(6.0, 0.0),
      std::nextafter(0.0, 1.0),
      9.0 * std::numeric_limits<double>::denorm_min(),
      std::nextafter(4.0, 0.0),
      std::nextafter(3.0, 0.0),
      std::nextafter(1.5, 0.0),
      std::nextafter(1.5, 3.0),
      std::numeric_limits<double>::denorm_min(),
  }};
  const std::size_t systems = levels.size();
  host.offsets.assign(systems + 1u, 0);
  host.electron_counts.resize(2u * systems);
  host.temperatures.resize(systems);
  host.active.assign(systems, 1u);
  std::int64_t total = 0;
  for (std::size_t system = 0u; system < systems; ++system) {
    host.offsets[system] = total;
    total += static_cast<std::int64_t>(levels[system].size());
  }
  host.offsets[systems] = total;
  host.eigenvalues.resize(static_cast<std::size_t>(total));
  for (std::size_t system = 0u; system < systems; ++system) {
    const std::int64_t begin = host.offsets[system];
    for (std::size_t orbital = 0u; orbital < levels[system].size(); ++orbital) {
      host.eigenvalues[static_cast<std::size_t>(begin + orbital)] = levels[system][orbital];
    }
    /* finite temperature everywhere so the Fermi (not Aufbau) path runs */
    host.temperatures[system] = system == 7u    ? std::numeric_limits<double>::max()
                                : system >= 10u ? 1.0e-7
                                                : GPUXTB_DEFAULT_ELECTRONIC_TEMPERATURE;
    host.electron_counts[2u * system] = electron0[system];
    host.electron_counts[2u * system + 1u] = electron0[system];
  }
  return host;
}

int test_degenerate_representability_policy_parity() {
  HostCase host = make_representability_case();
  std::string error;
  CHECK(build_cpu_reference(host, error));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, nullptr));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_occupations_device_errors_cuda(
      static_cast<std::int64_t>(host.batch_size()), device.system_errors.get(),
      device.device_error.get()));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_occupations_cuda(
      device.batch(host), device.eigenvalues.get(),
      static_cast<std::int64_t>(device.eigenvalues.size()), device.results(host),
      device.workspace(), device.system_errors.get(), device.device_error.get()));
  Results actual;
  CUDA_CHECK(copy_results(host, device, actual, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(actual.device_error == 0u);
  CHECK(std::all_of(actual.system_errors.begin(), actual.system_errors.end(),
                    [](std::uint32_t value) { return value == 0u; }));

  const double eps = std::numeric_limits<double>::epsilon();
  for (std::size_t system = 0u; system < host.batch_size(); ++system) {
    const std::int64_t begin = host.offsets[system];
    const std::int64_t count = host.offsets[system + 1u] - host.offsets[system];
    const double target = host.electron_counts[2u * system];
    /* Only a fully degenerate spectrum is governed by the representability
     * quantization bound; mixed spectra keep ordinary fractional accuracy. */
    bool fully_degenerate = true;
    for (std::int64_t orbital = 1u; orbital < count; ++orbital) {
      fully_degenerate =
          fully_degenerate && host.eigenvalues[static_cast<std::size_t>(begin + orbital)] ==
                                  host.eigenvalues[static_cast<std::size_t>(begin + orbital - 1u)];
    }
    for (int spin = 0; spin < 2; ++spin) {
      long double host_sum = 0.0L;
      long double device_sum = 0.0L;
      /* Degenerate members must share one value: for every orbital pair with
       * equal eigenvalues, the published (host and device) occupations agree. */
      bool block_uniform = true;
      bool occupations_match = true;
      for (std::int64_t orbital = 0; orbital < count; ++orbital) {
        const std::size_t element = 2u * static_cast<std::size_t>(begin) +
                                    static_cast<std::size_t>(spin) * count +
                                    static_cast<std::size_t>(orbital);
        host_sum += static_cast<long double>(host.expected_occupations[element]);
        device_sum += static_cast<long double>(actual.occupations[element]);
        occupations_match = occupations_match && near(actual.occupations[element],
                                                      host.expected_occupations[element], 4.0e-13);
        for (std::int64_t other = 0; other < orbital; ++other) {
          const double left = host.eigenvalues[static_cast<std::size_t>(begin + other)];
          const double right = host.eigenvalues[static_cast<std::size_t>(begin + orbital)];
          if (left == right) {
            const std::size_t other_element = 2u * static_cast<std::size_t>(begin) +
                                              static_cast<std::size_t>(spin) * count +
                                              static_cast<std::size_t>(other);
            block_uniform =
                block_uniform &&
                host.expected_occupations[element] == host.expected_occupations[other_element] &&
                actual.occupations[element] == actual.occupations[other_element];
          }
        }
      }
      CHECK(occupations_match);
      /* Exactly degenerate blocks publish equal, symmetric occupations. */
      CHECK(block_uniform);
      /* Fully degenerate relaxation systems must satisfy the documented
       * electron/hole quantization bound count * 2 * eps_double; mixed
       * spectra keep the ordinary fractional-filling accuracy. */
      if (fully_degenerate) {
        CHECK(std::abs(host_sum - static_cast<long double>(target)) <=
              2.0L * static_cast<long double>(eps) * static_cast<long double>(count));
        CHECK(std::abs(device_sum - static_cast<long double>(target)) <=
              2.0L * static_cast<long double>(eps) * static_cast<long double>(count));
      } else {
        CHECK(std::abs(host_sum - static_cast<long double>(target)) <=
              2.0e-13L * static_cast<long double>(std::max(1.0, target)));
        CHECK(std::abs(device_sum - static_cast<long double>(target)) <=
              2.0e-13L * static_cast<long double>(std::max(1.0, target)));
      }
      CHECK(near(actual.electron_sums[2u * system + spin],
                 host.expected_electron_sums[2u * system + spin], 4.0e-13));
      /* The analytic equal-level logit defines a canonical root when the
       * per-member subnormal target underflows. Ordinary targets retain the
       * established absolute parity tolerance. */
      const double device_mu = actual.chemical_potentials[2u * system + spin];
      const double host_mu = host.expected_chemical_potentials[2u * system + spin];
      const bool subnormal_target = target != 0.0 && std::fpclassify(target) == FP_SUBNORMAL;
      const bool canonical_subnormal_mu =
          fully_degenerate && subnormal_target && target / static_cast<double>(count) == 0.0;
      if (canonical_subnormal_mu) {
        CHECK(within_ulps(device_mu, host_mu, 8));
      } else if (subnormal_target) {
        /* A mixed-spectrum strict rescue need not be a direct Fermi vector,
         * so its root is noncanonical; #31 requires finite diagnostics while
         * exact broader chemical-potential parity remains part of #54. */
        CHECK(std::isfinite(device_mu) && std::isfinite(host_mu));
      } else {
        CHECK(near(device_mu, host_mu, 8.0e-12));
      }
    }
    CHECK(near(actual.entropies[system], host.expected_entropies[system], 8.0e-12));
  }

  /* The canonical three-orbital corner must choose the one-hole-ulp state on
   * both backends. The previous CUDA root selected the adjacent two-hole-ulp
   * state, which stayed inside the coarse error bound but was not nearest. */
  const double one_hole_state = 0x1.fffffffffffffp-1;
  CHECK(std::nextafter(1.0, 0.0) == one_hole_state);
  for (int spin = 0; spin < 2; ++spin) {
    const std::size_t system = 0u;
    const std::int64_t begin = host.offsets[system];
    for (std::int64_t orbital = 0; orbital < 3; ++orbital) {
      const std::size_t element = 2u * static_cast<std::size_t>(begin) +
                                  static_cast<std::size_t>(spin) * 3u +
                                  static_cast<std::size_t>(orbital);
      CHECK(host.expected_occupations[element] == one_hole_state);
      CHECK(actual.occupations[element] == one_hole_state);
    }
  }
  CHECK(std::abs(actual.entropies[0] - host.expected_entropies[0]) <= 2.0e-15);

  /* System 5 has no multi-orbital degeneracy block, so it must conserve the
   * near-capacity target at the strict publication tolerance rather than using
   * the relaxed block tolerance. */
  for (int spin = 0; spin < 2; ++spin) {
    const std::size_t system = 5u;
    const std::int64_t begin = host.offsets[system];
    long double holes = 0.0L;
    for (std::int64_t orbital = 0; orbital < 3; ++orbital) {
      const std::size_t element = 2u * static_cast<std::size_t>(begin) +
                                  static_cast<std::size_t>(spin) * 3u +
                                  static_cast<std::size_t>(orbital);
      holes += 1.0L - static_cast<long double>(actual.occupations[element]);
    }
    const long double target_holes =
        3.0L - static_cast<long double>(host.electron_counts[2u * system + spin]);
    CHECK(std::abs(holes - target_holes) <= 64.0L * static_cast<long double>(eps) * target_holes);
  }

  /* System 6 contains two exact-degeneracy blocks. The three frontier members
   * must select the three-hole-ulp state, which is closer to the target than
   * the adjacent two-hole-ulp state, identically on CPU and CUDA. */
  const double three_hole_state =
      std::nextafter(std::nextafter(std::nextafter(1.0, 0.0), 0.0), 0.0);
  CHECK(three_hole_state == 0x1.ffffffffffffdp-1);
  const double two_hole_state = std::nextafter(std::nextafter(1.0, 0.0), 0.0);
  for (int spin = 0; spin < 2; ++spin) {
    const std::size_t system = 6u;
    const std::int64_t begin = host.offsets[system];
    long double sum = 0.0L;
    for (std::int64_t orbital = 0; orbital < 6; ++orbital) {
      const std::size_t element = 2u * static_cast<std::size_t>(begin) +
                                  static_cast<std::size_t>(spin) * 6u +
                                  static_cast<std::size_t>(orbital);
      const double occupation = actual.occupations[element];
      CHECK(occupation == host.expected_occupations[element]);
      CHECK(occupation == (orbital < 3 ? 1.0 : three_hole_state));
      sum += static_cast<long double>(occupation);
    }
    const long double target = static_cast<long double>(host.electron_counts[2u * system + spin]);
    const long double selected_error = std::abs(sum - target);
    const long double adjacent_error =
        std::abs(3.0L + 3.0L * static_cast<long double>(two_hole_state) - target);
    CHECK(selected_error < adjacent_error);
    CHECK(selected_error <= 2.0L * static_cast<long double>(eps) * 3.0L);
  }
  CHECK(std::abs(actual.entropies[6] - host.expected_entropies[6]) <= 2.0e-15);
  CHECK(host.expected_chemical_potentials[14] == -std::numeric_limits<double>::max());
  CHECK(actual.chemical_potentials[14] == -std::numeric_limits<double>::max());
  CHECK(host.expected_chemical_potentials[15] == -std::numeric_limits<double>::max());
  CHECK(actual.chemical_potentials[15] == -std::numeric_limits<double>::max());

  /* System 8 is an exact half-subnormal tie. Both neighboring symmetric
   * states have the same long-double count error, so the canonical tuple
   * chooses the lower per-member occupation on CPU and CUDA. */
  const double denorm = std::numeric_limits<double>::denorm_min();
  const double lower_tie_occupation = 4.0 * denorm;
  const double upper_tie_occupation = 5.0 * denorm;
  const long double tie_target = 9.0L * static_cast<long double>(denorm);
  const long double lower_tie_error =
      std::abs(2.0L * static_cast<long double>(lower_tie_occupation) - tie_target);
  const long double upper_tie_error =
      std::abs(2.0L * static_cast<long double>(upper_tie_occupation) - tie_target);
  CHECK(lower_tie_error == upper_tie_error);
  for (int spin = 0; spin < 2; ++spin) {
    const std::size_t system = 8u;
    const std::int64_t begin = host.offsets[system];
    for (std::int64_t orbital = 0; orbital < 2; ++orbital) {
      const std::size_t element = 2u * static_cast<std::size_t>(begin) +
                                  static_cast<std::size_t>(spin) * 2u +
                                  static_cast<std::size_t>(orbital);
      CHECK(host.expected_occupations[element] == lower_tie_occupation);
      CHECK(actual.occupations[element] == lower_tie_occupation);
    }
    CHECK(within_ulps(actual.chemical_potentials[2u * system + spin],
                      host.expected_chemical_potentials[2u * system + spin], 8));
  }
  const double tie_hole = 1.0 - lower_tie_occupation;
  const double tie_contribution =
      -lower_tie_occupation * std::log(lower_tie_occupation) - tie_hole * std::log(tie_hole);
  const double expected_tie_entropy =
      (tie_contribution + tie_contribution) + (tie_contribution + tie_contribution);
  CHECK(actual.entropies[8] == expected_tie_entropy);
  CHECK(host.expected_entropies[8] == expected_tie_entropy);

  /* System 9 has a strict global solution even though the three-member upper
   * block also offers a relaxed state. All four orbitals must publish the
   * exact one-hole state, including the lower singleton. */
  for (int spin = 0; spin < 2; ++spin) {
    const std::size_t system = 9u;
    const std::int64_t begin = host.offsets[system];
    long double holes = 0.0L;
    for (std::int64_t orbital = 0; orbital < 4; ++orbital) {
      const std::size_t element = 2u * static_cast<std::size_t>(begin) +
                                  static_cast<std::size_t>(spin) * 4u +
                                  static_cast<std::size_t>(orbital);
      CHECK(host.expected_occupations[element] == one_hole_state);
      CHECK(actual.occupations[element] == one_hole_state);
      holes += 1.0L - static_cast<long double>(actual.occupations[element]);
    }
    CHECK(holes == 4.0L - static_cast<long double>(host.electron_counts[2u * system + spin]));
    CHECK(within_ulps(actual.chemical_potentials[2u * system + spin],
                      host.expected_chemical_potentials[2u * system + spin], 8));
  }
  const long double partial_occupation = static_cast<long double>(one_hole_state);
  const double expected_partial_entropy = static_cast<double>(
      -8.0L * (partial_occupation * std::log(partial_occupation) +
               (1.0L - partial_occupation) * std::log(1.0L - partial_occupation)));
  CHECK(within_ulps(actual.entropies[9], expected_partial_entropy, 8));
  CHECK(within_ulps(actual.entropies[9], host.expected_entropies[9], 8));

  /* System 10 exhausts binary64 chemical-potential spacing across a sharp
   * low-temperature degenerate frontier. The final adjacent root bracket
   * still contains the target, so CUDA must reach the same correction path as
   * the CPU instead of rejecting the system before publication. */
  for (int spin = 0; spin < 2; ++spin) {
    const std::size_t system = 10u;
    const std::int64_t begin = host.offsets[system];
    long double sum = 0.0L;
    double frontier_occupation = -1.0;
    for (std::int64_t orbital = 0; orbital < 6; ++orbital) {
      const std::size_t element = 2u * static_cast<std::size_t>(begin) +
                                  static_cast<std::size_t>(spin) * 6u +
                                  static_cast<std::size_t>(orbital);
      CHECK(std::isfinite(actual.occupations[element]));
      if (orbital == 1) {
        frontier_occupation = actual.occupations[element];
      } else if (orbital > 1) {
        CHECK(actual.occupations[element] == frontier_occupation);
      }
      sum += static_cast<long double>(actual.occupations[element]);
    }
    const long double target = static_cast<long double>(host.electron_counts[2u * system + spin]);
    CHECK(std::abs(sum - target) <= 64.0L * static_cast<long double>(eps) * std::max(1.0L, target));
  }

  /* Systems 11 and 12 are the exact electron/hole retry mirrors. Their
   * initial translated references sit on opposite singleton sides of the
   * fractional degenerate frontier, but the shared retry and publication
   * policy must produce bit-identical CPU/CUDA observables. */
  for (const std::size_t system : {11u, 12u}) {
    const std::int64_t begin = host.offsets[system];
    const bool solve_holes = system == 12u;
    for (int spin = 0; spin < 2; ++spin) {
      long double quantity = 0.0L;
      for (std::int64_t orbital = 0; orbital < 3; ++orbital) {
        const std::size_t element = 2u * static_cast<std::size_t>(begin) +
                                    static_cast<std::size_t>(spin) * 3u +
                                    static_cast<std::size_t>(orbital);
        CHECK(actual.occupations[element] == host.expected_occupations[element]);
        quantity += solve_holes ? 1.0L - static_cast<long double>(actual.occupations[element])
                                : static_cast<long double>(actual.occupations[element]);
      }
      const std::size_t first_frontier = solve_holes ? 0u : 1u;
      const std::size_t first_element = 2u * static_cast<std::size_t>(begin) +
                                        static_cast<std::size_t>(spin) * 3u + first_frontier;
      CHECK(actual.occupations[first_element] == actual.occupations[first_element + 1u]);
      const double requested = host.electron_counts[2u * system + spin];
      const long double target = solve_holes ? 3.0L - static_cast<long double>(requested)
                                             : static_cast<long double>(requested);
      CHECK(std::abs(quantity - target) <=
            64.0L * static_cast<long double>(eps) * static_cast<long double>(target));
      CHECK(within_ulps(actual.chemical_potentials[2u * system + spin],
                        host.expected_chemical_potentials[2u * system + spin], 8));
    }
    CHECK(actual.entropies[system] == host.expected_entropies[system]);
  }

  /* System 13 exhausts spacing at the causal lower pair, then conserves
   * strictly by publishing the denormal electron on the upper singleton. */
  for (int spin = 0; spin < 2; ++spin) {
    const std::size_t system = 13u;
    const std::int64_t begin = host.offsets[system];
    const std::size_t spin_begin =
        2u * static_cast<std::size_t>(begin) + static_cast<std::size_t>(spin) * 3u;
    CHECK(actual.occupations[spin_begin] == 0.0);
    CHECK(actual.occupations[spin_begin + 1u] == 0.0);
    CHECK(actual.occupations[spin_begin + 2u] == std::numeric_limits<double>::denorm_min());
    for (std::size_t orbital = 0u; orbital < 3u; ++orbital) {
      CHECK(actual.occupations[spin_begin + orbital] ==
            host.expected_occupations[spin_begin + orbital]);
    }
    CHECK(std::isfinite(actual.chemical_potentials[2u * system + spin]));
    CHECK(std::isfinite(host.expected_chemical_potentials[2u * system + spin]));
  }
  CHECK(actual.entropies[13] == host.expected_entropies[13]);
  return 0;
}

HostCase make_mixed_spin_case(std::size_t batch_size) {
  HostCase host;
  host.offsets.assign(batch_size + 1u, 0);
  host.spin_channels.resize(batch_size);
  host.spin_channel_offsets.assign(batch_size + 1u, 0);
  host.spin_orbital_offsets.assign(batch_size + 1u, 0);
  host.electron_counts.resize(2u * batch_size);
  host.temperatures.resize(batch_size);
  host.active.assign(batch_size, 1u);
  for (std::size_t system = 0; system < batch_size; ++system) {
    const std::int64_t count = batch_size == 1u ? 4 : 2 + static_cast<std::int64_t>(system % 4u);
    const std::int32_t channels = batch_size == 1u || system % 3u != 0u ? 2 : 1;
    host.offsets[system + 1u] = host.offsets[system] + count;
    host.spin_channels[system] = channels;
    host.spin_channel_offsets[system + 1u] = host.spin_channel_offsets[system] + channels;
    host.spin_orbital_offsets[system + 1u] =
        host.spin_orbital_offsets[system] + static_cast<std::int64_t>(channels) * count;
    host.temperatures[system] = system % 2u == 0u ? 0.0 : 0.01;
    host.electron_counts[2u * system] = std::min(1.25, static_cast<double>(count));
    host.electron_counts[2u * system + 1u] = std::min(0.75, static_cast<double>(count));
  }
  host.eigenvalues.resize(static_cast<std::size_t>(host.spin_orbital_offsets.back()));
  for (std::size_t system = 0; system < batch_size; ++system) {
    const std::int64_t count = host.offsets[system + 1u] - host.offsets[system];
    const std::int64_t spectrum_begin = host.spin_orbital_offsets[system];
    for (std::int32_t spin = 0; spin < host.spin_channels[system]; ++spin) {
      for (std::int64_t orbital = 0; orbital < count; ++orbital) {
        host.eigenvalues[static_cast<std::size_t>(
            spectrum_begin + static_cast<std::int64_t>(spin) * count + orbital)] =
            -0.8 + 0.21 * static_cast<double>(orbital) + 0.017 * static_cast<double>(spin) +
            0.001 * static_cast<double>(system);
      }
    }
  }
  return host;
}

int test_mixed_spin_spectra_batches_and_system_transaction() {
  for (const std::size_t batch_size : {1u, 8u, 32u, 128u}) {
    HostCase host = make_mixed_spin_case(batch_size);
    std::string error;
    CHECK(build_cpu_reference(host, error));
    DeviceFixture device;
    CUDA_CHECK(device.initialize(host, nullptr));
    CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_occupations_device_errors_cuda(
        static_cast<std::int64_t>(batch_size), device.system_errors.get(),
        device.device_error.get()));
    CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_occupations_cuda(
        device.batch(host), device.layout(host), device.eigenvalues.get(),
        static_cast<std::int64_t>(device.eigenvalues.size()), device.results(host),
        device.workspace(), device.system_errors.get(), device.device_error.get()));
    Results actual;
    CUDA_CHECK(copy_results(host, device, actual, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(compare_success(host, actual) == 0);
    CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_occupations_cuda(
              device.batch(host), device.eigenvalues.get(),
              static_cast<std::int64_t>(device.eigenvalues.size()), device.results(host),
              device.workspace(), device.system_errors.get(),
              device.device_error.get()) == cudaErrorInvalidValue);

    if (batch_size == 8u) {
      constexpr std::size_t failed_system = 1u;
      CHECK(host.spin_channels[failed_system] == 2u);
      std::vector<double> poisoned = host.eigenvalues;
      const std::int64_t count = host.offsets[failed_system + 1u] - host.offsets[failed_system];
      poisoned[static_cast<std::size_t>(host.spin_orbital_offsets[failed_system] + count)] =
          std::numeric_limits<double>::quiet_NaN();
      CUDA_CHECK(device.eigenvalues.copy_from(poisoned.data(), poisoned.size()));
      CUDA_CHECK(device.reset_outputs(host, nullptr));
      CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_occupations_device_errors_cuda(
          8, device.system_errors.get(), device.device_error.get()));
      CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_occupations_cuda(
          device.batch(host), device.layout(host), device.eigenvalues.get(),
          static_cast<std::int64_t>(device.eigenvalues.size()), device.results(host),
          device.workspace(), device.system_errors.get(), device.device_error.get()));
      CUDA_CHECK(copy_results(host, device, actual, nullptr));
      CUDA_CHECK(cudaDeviceSynchronize());
      CHECK(actual.system_errors[failed_system] ==
            static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kNonfiniteEigenvalue));
      for (std::int64_t element = 2 * host.offsets[failed_system];
           element < 2 * host.offsets[failed_system + 1u]; ++element) {
        CHECK(actual.occupations[static_cast<std::size_t>(element)] == kSentinel);
      }
      CHECK(actual.entropies[failed_system] == kSentinel);
      CHECK(actual.entropies[2] != kSentinel);

      /* Validate the full int32_t value before narrowing to the kernel's 1/2 channel type. */
      CUDA_CHECK(device.eigenvalues.copy_from(host.eigenvalues.data(), host.eigenvalues.size()));
      std::vector<std::int32_t> hostile_spin = host.spin_channels;
      hostile_spin[failed_system] = 257;
      CUDA_CHECK(device.spin_channels.copy_from(hostile_spin.data(), hostile_spin.size()));
      CUDA_CHECK(device.reset_outputs(host, nullptr));
      CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_occupations_device_errors_cuda(
          8, device.system_errors.get(), device.device_error.get()));
      CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_occupations_cuda(
          device.batch(host), device.layout(host), device.eigenvalues.get(),
          static_cast<std::int64_t>(device.eigenvalues.size()), device.results(host),
          device.workspace(), device.system_errors.get(), device.device_error.get()));
      CUDA_CHECK(copy_results(host, device, actual, nullptr));
      CUDA_CHECK(cudaDeviceSynchronize());
      CHECK(actual.system_errors[failed_system] ==
            static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kInvalidSpinLayout));
      CHECK(actual.entropies[failed_system] == kSentinel);
      CHECK(actual.entropies[2] != kSentinel);
    }
  }
  return 0;
}

HostCase make_difficult_case() {
  HostCase host;
  host.offsets = {0, 3, 5, 7, 9, 11, 13, 15, 18};
  host.eigenvalues = {-1.0,
                      0.0,
                      1.0,
                      0.0,
                      0.0,
                      -0.2,
                      0.3,
                      0.0,
                      1.0,
                      std::numeric_limits<double>::max() / 2.0,
                      std::numeric_limits<double>::max(),
                      100.0,
                      101.0,
                      -3.0,
                      7.0,
                      -1.0e-13,
                      0.0,
                      1.0e-13};
  host.electron_counts = {1.5,  0.5, 1.0,      1.3,      1.0e-16, 1.0e-20, std::nextafter(2.0, 0.0),
                          0.7,  1.0, 1.0e-300, 1.0e-300, 1.0e-16, 0.0,     2.0,
                          1.25, 1.75};
  host.temperatures = {0.0,
                       GPUXTB_DEFAULT_ELECTRONIC_TEMPERATURE,
                       0.01,
                       1.0e-7,
                       std::numeric_limits<double>::max(),
                       1.0e-7,
                       0.01,
                       1.0e-10};
  host.active.assign(8u, 1u);
  return host;
}

int test_boundary_degenerate_translated_and_extreme_spectra() {
  HostCase host = make_difficult_case();
  std::string error;
  CHECK(build_cpu_reference(host, error));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, nullptr));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_occupations_device_errors_cuda(
      8, device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_occupations_cuda(
      device.batch(host), device.eigenvalues.get(),
      static_cast<std::int64_t>(host.total_orbitals()), device.results(host), device.workspace(),
      device.system_errors.get(), device.device_error.get()));
  Results actual;
  CUDA_CHECK(copy_results(host, device, actual, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  const int comparison = compare_success(host, actual);
  CHECK(comparison == 0);
  CHECK(actual.occupations[6] > 0.0 && actual.occupations[6] < 1.0);
  CHECK(actual.occupations[22] > 0.0);
  return 0;
}

int test_peer_isolated_failures_inactive_mask_and_hostile_offsets() {
  HostCase host = make_regular_case(8u);
  std::string error;
  CHECK(build_cpu_reference(host, error));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, nullptr));

  std::vector<double> bad_eigenvalues = host.eigenvalues;
  bad_eigenvalues[static_cast<std::size_t>(host.offsets[2])] =
      std::numeric_limits<double>::quiet_NaN();
  bad_eigenvalues[static_cast<std::size_t>(host.offsets[4] + 1)] =
      bad_eigenvalues[static_cast<std::size_t>(host.offsets[4])] - 1.0;
  std::vector<double> bad_counts = host.electron_counts;
  bad_counts[2u * 6u] = 1000.0;
  std::vector<double> bad_temperatures = host.temperatures;
  bad_temperatures[5] = -1.0;
  CUDA_CHECK(device.eigenvalues.copy_from(bad_eigenvalues.data(), bad_eigenvalues.size()));
  CUDA_CHECK(device.electron_counts.copy_from(bad_counts.data(), bad_counts.size()));
  CUDA_CHECK(device.temperatures.copy_from(bad_temperatures.data(), bad_temperatures.size()));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_occupations_device_errors_cuda(
      8, device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_occupations_cuda(
      device.batch(host), device.eigenvalues.get(),
      static_cast<std::int64_t>(host.total_orbitals()), device.results(host), device.workspace(),
      device.system_errors.get(), device.device_error.get()));
  Results actual;
  CUDA_CHECK(copy_results(host, device, actual, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(actual.system_errors[2] ==
        static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kNonfiniteEigenvalue));
  CHECK(actual.system_errors[4] ==
        static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kUnsortedEigenvalues));
  CHECK(actual.system_errors[5] ==
        static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kInvalidTemperature));
  CHECK(actual.system_errors[6] ==
        static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kInvalidElectronCount));
  CHECK(actual.device_error ==
        static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kNonfiniteEigenvalue));
  for (const std::size_t system : {2u, 4u, 5u, 6u}) {
    const std::int64_t begin = host.offsets[system];
    const std::int64_t end = host.offsets[system + 1u];
    for (std::int64_t element = 2 * begin; element < 2 * end; ++element) {
      CHECK(actual.occupations[static_cast<std::size_t>(element)] == kSentinel);
    }
    CHECK(actual.chemical_potentials[2u * system] == kSentinel);
    CHECK(actual.chemical_potentials[2u * system + 1u] == kSentinel);
    CHECK(actual.electron_sums[2u * system] == kSentinel);
    CHECK(actual.electron_sums[2u * system + 1u] == kSentinel);
    CHECK(actual.entropies[system] == kSentinel);
  }
  CHECK(actual.system_errors[1] == 0u && actual.entropies[1] != kSentinel);

  /* Invalid values hidden behind an inactive member are never inspected. */
  host.active[2] = 0u;
  CUDA_CHECK(device.active.copy_from(host.active.data(), host.active.size()));
  CUDA_CHECK(device.eigenvalues.copy_from(bad_eigenvalues.data(), bad_eigenvalues.size()));
  CUDA_CHECK(
      device.electron_counts.copy_from(host.electron_counts.data(), host.electron_counts.size()));
  CUDA_CHECK(device.temperatures.copy_from(host.temperatures.data(), host.temperatures.size()));
  CUDA_CHECK(device.reset_outputs(host, nullptr));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_occupations_device_errors_cuda(
      8, device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_occupations_cuda(
      device.batch(host), device.eigenvalues.get(),
      static_cast<std::int64_t>(host.total_orbitals()), device.results(host), device.workspace(),
      device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(copy_results(host, device, actual, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(actual.system_errors[2] == 0u);
  CHECK(actual.entropies[2] == kSentinel);

  host.active[2] = 2u;
  CUDA_CHECK(device.active.copy_from(host.active.data(), host.active.size()));
  CUDA_CHECK(device.reset_outputs(host, nullptr));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_occupations_device_errors_cuda(
      8, device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_occupations_cuda(
      device.batch(host), device.eigenvalues.get(),
      static_cast<std::int64_t>(host.total_orbitals()), device.results(host), device.workspace(),
      device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(copy_results(host, device, actual, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(actual.system_errors[2] ==
        static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kInvalidActiveMask));
  CHECK(actual.device_error ==
        static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kInvalidActiveMask));
  CHECK(actual.entropies[2] == kSentinel);

  CUDA_CHECK(device.eigenvalues.copy_from(host.eigenvalues.data(), host.eigenvalues.size()));
  host.active[2] = 1u;
  CUDA_CHECK(device.active.copy_from(host.active.data(), host.active.size()));
  std::vector<std::int64_t> hostile = host.offsets;
  hostile[3] = std::numeric_limits<std::int64_t>::max();
  hostile[6] = std::numeric_limits<std::int64_t>::min();
  CUDA_CHECK(device.offsets.copy_from(hostile.data(), hostile.size()));
  CUDA_CHECK(device.reset_outputs(host, nullptr));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_occupations_device_errors_cuda(
      8, device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_occupations_cuda(
      device.batch(host), device.eigenvalues.get(),
      static_cast<std::int64_t>(host.total_orbitals()), device.results(host), device.workspace(),
      device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(copy_results(host, device, actual, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(actual.system_errors[2] ==
        static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kInvalidOffsets));
  CHECK(actual.system_errors[3] ==
        static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kInvalidOffsets));
  CHECK(actual.system_errors[5] ==
        static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kInvalidOffsets));
  CHECK(actual.system_errors[6] ==
        static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kInvalidOffsets));
  CHECK(actual.system_errors[1] == 0u && actual.system_errors[4] == 0u);
  CHECK(actual.device_error ==
        static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kInvalidOffsets));

  /* A pre-existing sticky diagnostic suppresses the complete sequence. */
  CUDA_CHECK(device.offsets.copy_from(host.offsets.data(), host.offsets.size()));
  CUDA_CHECK(device.reset_outputs(host, nullptr));
  CUDA_CHECK(cudaMemset(device.system_errors.get(), 0, 8u * sizeof(std::uint32_t)));
  const std::uint32_t sticky =
      static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kNonfiniteEntropy);
  CUDA_CHECK(
      cudaMemcpy(device.device_error.get(), &sticky, sizeof(sticky), cudaMemcpyHostToDevice));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_occupations_cuda(
      device.batch(host), device.eigenvalues.get(),
      static_cast<std::int64_t>(host.total_orbitals()), device.results(host), device.workspace(),
      device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(copy_results(host, device, actual, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(actual.device_error == sticky);
  CHECK(std::all_of(actual.occupations.begin(), actual.occupations.end(),
                    [](double value) { return value == kSentinel; }));
  CHECK(std::all_of(actual.entropies.begin(), actual.entropies.end(),
                    [](double value) { return value == kSentinel; }));
  return 0;
}

int test_cuda_graph_capture_and_replay() {
  HostCase host = make_regular_case(32u);
  std::string error;
  CHECK(build_cpu_reference(host, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_occupations_device_errors_cuda(
      32, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_occupations_cuda(
      device.batch(host), device.eigenvalues.get(),
      static_cast<std::int64_t>(host.total_orbitals()), device.results(host), device.workspace(),
      device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  for (int replay = 0; replay < 2; ++replay) {
    CUDA_CHECK(device.reset_outputs(host, stream));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    Results actual;
    CUDA_CHECK(copy_results(host, device, actual, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int comparison = compare_success(host, actual);
    CHECK(comparison == 0);
  }
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_host_argument_alias_and_alignment_validation() {
  HostCase host = make_regular_case(8u);
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, nullptr));
  Gfn2OccupationsDeviceBatch batch = device.batch(host);
  Gfn2OccupationsDeviceResults results = device.results(host);
  Gfn2OccupationsDeviceWorkspace workspace = device.workspace();
  batch.orbital_offsets = reinterpret_cast<const std::int64_t*>(
      reinterpret_cast<const unsigned char*>(device.offsets.get()) + 1u);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_occupations_cuda(
            batch, device.eigenvalues.get(), static_cast<std::int64_t>(host.total_orbitals()),
            results, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  batch = device.batch(host);
  results.occupations = workspace.occupation_scratch;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_occupations_cuda(
            batch, device.eigenvalues.get(), static_cast<std::int64_t>(host.total_orbitals()),
            results, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  results = device.results(host);
  workspace.entropy_scratch = results.entropies;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_occupations_cuda(
            batch, device.eigenvalues.get(), static_cast<std::int64_t>(host.total_orbitals()),
            results, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  workspace = device.workspace();
  results.electron_sums = reinterpret_cast<double*>(device.device_error.get());
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_occupations_cuda(
            batch, device.eigenvalues.get(), static_cast<std::int64_t>(host.total_orbitals()),
            results, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  results = device.results(host);
  batch.electron_count_elements = std::numeric_limits<std::int64_t>::max();
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_occupations_cuda(
            batch, device.eigenvalues.get(), static_cast<std::int64_t>(host.total_orbitals()),
            results, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  CHECK(gpuxtb::detail::cuda::reset_gfn2_occupations_device_errors_cuda(
            8, device.system_errors.get(), device.system_errors.get()) == cudaErrorInvalidValue);
  return 0;
}

}  // namespace

int main() {
  if (const int line = test_cpu_parity_ragged_batches_and_custom_stream(); line != 0) {
    return line;
  }
  if (const int line = test_degenerate_representability_policy_parity(); line != 0) {
    return line;
  }
  if (const int line = test_boundary_degenerate_translated_and_extreme_spectra(); line != 0) {
    return line;
  }
  if (const int line = test_mixed_spin_spectra_batches_and_system_transaction(); line != 0) {
    return line;
  }
  if (const int line = test_peer_isolated_failures_inactive_mask_and_hostile_offsets(); line != 0) {
    return line;
  }
  if (const int line = test_cuda_graph_capture_and_replay(); line != 0) {
    return line;
  }
  if (const int line = test_host_argument_alias_and_alignment_validation(); line != 0) {
    return line;
  }
  return 0;
}
