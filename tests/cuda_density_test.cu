#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_density.cuh"

#define CHECK(condition)                                                                       \
  do {                                                                                         \
    if (!(condition)) {                                                                        \
      std::fprintf(stderr, "CUDA density test failed at line %d: %s\n", __LINE__, #condition); \
      return __LINE__;                                                                         \
    }                                                                                          \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using gpuxtb::detail::cuda::Gfn2DensityDeviceBatch;
using gpuxtb::detail::cuda::Gfn2DensityDeviceError;
using gpuxtb::detail::cuda::Gfn2DensityDeviceInput;
using gpuxtb::detail::cuda::Gfn2DensityDeviceResults;
using gpuxtb::detail::cuda::Gfn2DensityDeviceWorkspace;

constexpr std::uint64_t kPlanToken = 0xa54ff53a5f1d36f1ULL;
constexpr double kSentinel = -663.75;

static_assert(std::is_trivially_copyable_v<Gfn2DensityDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2DensityDeviceInput>);
static_assert(std::is_trivially_copyable_v<Gfn2DensityDeviceResults>);
static_assert(std::is_trivially_copyable_v<Gfn2DensityDeviceWorkspace>);

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <=
         tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
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
  std::vector<std::int64_t> orbital_offsets;
  std::vector<std::int64_t> matrix_offsets;
  std::vector<double> coefficients;
  std::vector<double> eigenvalues;
  std::vector<double> occupations;
  std::vector<std::uint8_t> active;
  std::vector<double> expected_density;
  std::vector<double> expected_weighted_density;
  std::vector<double> expected_band;
  std::vector<double> expected_occupation_sum;
  std::vector<double> expected_density_trace;
  std::vector<double> expected_weighted_trace;

  std::size_t batch_size() const { return active.size(); }
  std::size_t total_orbitals() const { return eigenvalues.size(); }
  std::size_t total_matrix_elements() const { return coefficients.size(); }
};

void build_reference(HostCase& host) {
  host.expected_density.assign(host.total_matrix_elements(), kSentinel);
  host.expected_weighted_density.assign(host.total_matrix_elements(), kSentinel);
  host.expected_band.assign(host.batch_size(), kSentinel);
  host.expected_occupation_sum.assign(host.batch_size(), kSentinel);
  host.expected_density_trace.assign(host.batch_size(), kSentinel);
  host.expected_weighted_trace.assign(host.batch_size(), kSentinel);
  for (std::size_t system = 0u; system < host.batch_size(); ++system) {
    if (host.active[system] == 0u) {
      continue;
    }
    const std::int64_t orbital_begin = host.orbital_offsets[system];
    const std::int64_t count = host.orbital_offsets[system + 1u] - orbital_begin;
    const std::int64_t matrix_begin = host.matrix_offsets[system];
    double band = 0.0;
    double occupation_sum = 0.0;
    for (std::int64_t orbital = 0; orbital < count; ++orbital) {
      const double weight =
          host.occupations[static_cast<std::size_t>(2 * orbital_begin + orbital)] +
          host.occupations[static_cast<std::size_t>(2 * orbital_begin + count + orbital)];
      occupation_sum += weight;
      band += weight * host.eigenvalues[static_cast<std::size_t>(orbital_begin + orbital)];
    }
    double density_trace = 0.0;
    double weighted_trace = 0.0;
    for (std::int64_t row = 0; row < count; ++row) {
      for (std::int64_t column = 0; column < count; ++column) {
        double density = 0.0;
        double weighted = 0.0;
        for (std::int64_t orbital = 0; orbital < count; ++orbital) {
          const double weight =
              host.occupations[static_cast<std::size_t>(2 * orbital_begin + orbital)] +
              host.occupations[static_cast<std::size_t>(2 * orbital_begin + count + orbital)];
          const double first =
              host.coefficients[static_cast<std::size_t>(matrix_begin + row * count + orbital)];
          const double second =
              host.coefficients[static_cast<std::size_t>(matrix_begin + column * count + orbital)];
          density = std::fma(first * weight, second, density);
          weighted = std::fma(
              first * weight * host.eigenvalues[static_cast<std::size_t>(orbital_begin + orbital)],
              second, weighted);
        }
        const std::size_t index = static_cast<std::size_t>(matrix_begin + row * count + column);
        host.expected_density[index] = density;
        host.expected_weighted_density[index] = weighted;
        if (row == column) {
          density_trace += density;
          weighted_trace += weighted;
        }
      }
    }
    host.expected_band[system] = band;
    host.expected_occupation_sum[system] = occupation_sum;
    host.expected_density_trace[system] = density_trace;
    host.expected_weighted_trace[system] = weighted_trace;
  }
}

HostCase make_case(std::size_t batch_size) {
  HostCase host;
  host.orbital_offsets.assign(batch_size + 1u, 0);
  host.matrix_offsets.assign(batch_size + 1u, 0);
  std::int64_t orbitals = 0;
  std::int64_t matrices = 0;
  for (std::size_t system = 0u; system < batch_size; ++system) {
    host.orbital_offsets[system] = orbitals;
    host.matrix_offsets[system] = matrices;
    const std::int64_t count = batch_size == 1u ? 7 : 1 + static_cast<std::int64_t>(system % 9u);
    orbitals += count;
    matrices += count * count;
  }
  host.orbital_offsets[batch_size] = orbitals;
  host.matrix_offsets[batch_size] = matrices;
  host.coefficients.resize(static_cast<std::size_t>(matrices));
  host.eigenvalues.resize(static_cast<std::size_t>(orbitals));
  host.occupations.resize(static_cast<std::size_t>(2 * orbitals));
  host.active.assign(batch_size, 1u);
  for (std::size_t system = 0u; system < batch_size; ++system) {
    const std::int64_t orbital_begin = host.orbital_offsets[system];
    const std::int64_t count = host.orbital_offsets[system + 1u] - orbital_begin;
    const std::int64_t matrix_begin = host.matrix_offsets[system];
    for (std::int64_t row = 0; row < count; ++row) {
      for (std::int64_t orbital = 0; orbital < count; ++orbital) {
        host.coefficients[static_cast<std::size_t>(matrix_begin + row * count + orbital)] =
            (row == orbital ? 0.83 : 0.017 * static_cast<double>(1 + (row + 2 * orbital) % 7)) +
            0.0001 * static_cast<double>(system);
      }
    }
    for (std::int64_t orbital = 0; orbital < count; ++orbital) {
      host.eigenvalues[static_cast<std::size_t>(orbital_begin + orbital)] =
          -0.9 + 0.19 * static_cast<double>(orbital) + 0.002 * static_cast<double>(system);
      host.occupations[static_cast<std::size_t>(2 * orbital_begin + orbital)] =
          std::min(1.0, 0.13 * static_cast<double>(1 + (system + orbital) % 8));
      host.occupations[static_cast<std::size_t>(2 * orbital_begin + count + orbital)] =
          std::min(1.0, 0.09 * static_cast<double>(1 + (2 * system + orbital) % 10));
    }
    if (batch_size > 1u && system % 17u == 16u) {
      host.active[system] = 0u;
      host.coefficients[static_cast<std::size_t>(matrix_begin)] =
          std::numeric_limits<double>::quiet_NaN();
      host.eigenvalues[static_cast<std::size_t>(orbital_begin)] =
          std::numeric_limits<double>::infinity();
      host.occupations[static_cast<std::size_t>(2 * orbital_begin)] = 7.0;
    }
  }
  build_reference(host);
  return host;
}

struct DeviceFixture {
  DeviceBuffer<std::int64_t> orbital_offsets;
  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<double> coefficients;
  DeviceBuffer<double> eigenvalues;
  DeviceBuffer<double> occupations;
  DeviceBuffer<std::uint8_t> active;
  DeviceBuffer<double> density;
  DeviceBuffer<double> weighted_density;
  DeviceBuffer<double> band;
  DeviceBuffer<double> occupation_sum;
  DeviceBuffer<double> density_trace;
  DeviceBuffer<double> weighted_trace;
  DeviceBuffer<double> density_scratch;
  DeviceBuffer<double> weighted_density_scratch;
  DeviceBuffer<double> weights;
  DeviceBuffer<double> energy_weights;
  DeviceBuffer<double> band_scratch;
  DeviceBuffer<double> occupation_sum_scratch;
  DeviceBuffer<double> density_trace_scratch;
  DeviceBuffer<double> weighted_trace_scratch;
  DeviceBuffer<std::uint32_t> sequence_active;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;

  cudaError_t initialize(const HostCase& host, cudaStream_t stream) {
    cudaError_t status = allocate_and_copy(orbital_offsets, host.orbital_offsets, stream);
    if (status == cudaSuccess)
      status = allocate_and_copy(matrix_offsets, host.matrix_offsets, stream);
    if (status == cudaSuccess) status = allocate_and_copy(coefficients, host.coefficients, stream);
    if (status == cudaSuccess) status = allocate_and_copy(eigenvalues, host.eigenvalues, stream);
    if (status == cudaSuccess) status = allocate_and_copy(occupations, host.occupations, stream);
    if (status == cudaSuccess) status = allocate_and_copy(active, host.active, stream);
    const std::size_t batch = host.batch_size();
    const std::size_t matrices = host.total_matrix_elements();
    const std::size_t orbitals = host.total_orbitals();
    if (status == cudaSuccess) status = density.allocate(matrices);
    if (status == cudaSuccess) status = weighted_density.allocate(matrices);
    if (status == cudaSuccess) status = band.allocate(batch);
    if (status == cudaSuccess) status = occupation_sum.allocate(batch);
    if (status == cudaSuccess) status = density_trace.allocate(batch);
    if (status == cudaSuccess) status = weighted_trace.allocate(batch);
    if (status == cudaSuccess) status = density_scratch.allocate(matrices);
    if (status == cudaSuccess) status = weighted_density_scratch.allocate(matrices);
    if (status == cudaSuccess) status = weights.allocate(orbitals);
    if (status == cudaSuccess) status = energy_weights.allocate(orbitals);
    if (status == cudaSuccess) status = band_scratch.allocate(batch);
    if (status == cudaSuccess) status = occupation_sum_scratch.allocate(batch);
    if (status == cudaSuccess) status = density_trace_scratch.allocate(batch);
    if (status == cudaSuccess) status = weighted_trace_scratch.allocate(batch);
    if (status == cudaSuccess) status = sequence_active.allocate(1u);
    if (status == cudaSuccess) status = system_errors.allocate(batch);
    if (status == cudaSuccess) status = device_error.allocate(1u);
    return status == cudaSuccess ? reset_outputs(host, stream) : status;
  }

  cudaError_t reset_outputs(const HostCase& host, cudaStream_t stream) {
    const std::vector<double> matrix_seed(host.total_matrix_elements(), kSentinel);
    const std::vector<double> scalar_seed(host.batch_size(), kSentinel);
    cudaError_t status = density.copy_from(matrix_seed.data(), matrix_seed.size(), stream);
    if (status == cudaSuccess)
      status = weighted_density.copy_from(matrix_seed.data(), matrix_seed.size(), stream);
    if (status == cudaSuccess)
      status = band.copy_from(scalar_seed.data(), scalar_seed.size(), stream);
    if (status == cudaSuccess)
      status = occupation_sum.copy_from(scalar_seed.data(), scalar_seed.size(), stream);
    if (status == cudaSuccess)
      status = density_trace.copy_from(scalar_seed.data(), scalar_seed.size(), stream);
    if (status == cudaSuccess)
      status = weighted_trace.copy_from(scalar_seed.data(), scalar_seed.size(), stream);
    return status;
  }

  Gfn2DensityDeviceBatch batch(const HostCase& host) const {
    return {static_cast<std::int64_t>(host.batch_size()),
            static_cast<std::int64_t>(host.total_orbitals()),
            static_cast<std::int64_t>(host.total_matrix_elements()),
            static_cast<std::int64_t>(host.orbital_offsets.size()),
            static_cast<std::int64_t>(host.matrix_offsets.size()),
            kPlanToken,
            orbital_offsets.get(),
            matrix_offsets.get()};
  }
  Gfn2DensityDeviceInput input() const {
    return {coefficients.get(), static_cast<std::int64_t>(coefficients.size()),
            eigenvalues.get(),  static_cast<std::int64_t>(eigenvalues.size()),
            occupations.get(),  static_cast<std::int64_t>(occupations.size()),
            active.get(),       static_cast<std::int64_t>(active.size()),
            kPlanToken};
  }
  Gfn2DensityDeviceResults results() {
    return {density.get(),
            static_cast<std::int64_t>(density.size()),
            weighted_density.get(),
            static_cast<std::int64_t>(weighted_density.size()),
            band.get(),
            static_cast<std::int64_t>(band.size()),
            occupation_sum.get(),
            static_cast<std::int64_t>(occupation_sum.size()),
            density_trace.get(),
            static_cast<std::int64_t>(density_trace.size()),
            weighted_trace.get(),
            static_cast<std::int64_t>(weighted_trace.size()),
            kPlanToken};
  }
  Gfn2DensityDeviceWorkspace workspace() {
    return {density_scratch.get(),
            static_cast<std::int64_t>(density_scratch.size()),
            weighted_density_scratch.get(),
            static_cast<std::int64_t>(weighted_density_scratch.size()),
            weights.get(),
            static_cast<std::int64_t>(weights.size()),
            energy_weights.get(),
            static_cast<std::int64_t>(energy_weights.size()),
            band_scratch.get(),
            static_cast<std::int64_t>(band_scratch.size()),
            occupation_sum_scratch.get(),
            static_cast<std::int64_t>(occupation_sum_scratch.size()),
            density_trace_scratch.get(),
            static_cast<std::int64_t>(density_trace_scratch.size()),
            weighted_trace_scratch.get(),
            static_cast<std::int64_t>(weighted_trace_scratch.size()),
            sequence_active.get(),
            static_cast<std::int64_t>(sequence_active.size()),
            kPlanToken};
  }
};

struct Results {
  std::vector<double> density;
  std::vector<double> weighted_density;
  std::vector<double> band;
  std::vector<double> occupation_sum;
  std::vector<double> density_trace;
  std::vector<double> weighted_trace;
  std::vector<std::uint32_t> system_errors;
  std::uint32_t device_error = 0u;
};

cudaError_t copy_results(const HostCase& host, const DeviceFixture& device, Results& results,
                         cudaStream_t stream) {
  results.density.resize(host.total_matrix_elements());
  results.weighted_density.resize(host.total_matrix_elements());
  results.band.resize(host.batch_size());
  results.occupation_sum.resize(host.batch_size());
  results.density_trace.resize(host.batch_size());
  results.weighted_trace.resize(host.batch_size());
  results.system_errors.resize(host.batch_size());
  cudaError_t status =
      device.density.copy_to(results.density.data(), results.density.size(), stream);
  if (status == cudaSuccess)
    status = device.weighted_density.copy_to(results.weighted_density.data(),
                                             results.weighted_density.size(), stream);
  if (status == cudaSuccess)
    status = device.band.copy_to(results.band.data(), results.band.size(), stream);
  if (status == cudaSuccess)
    status = device.occupation_sum.copy_to(results.occupation_sum.data(),
                                           results.occupation_sum.size(), stream);
  if (status == cudaSuccess)
    status = device.density_trace.copy_to(results.density_trace.data(),
                                          results.density_trace.size(), stream);
  if (status == cudaSuccess)
    status = device.weighted_trace.copy_to(results.weighted_trace.data(),
                                           results.weighted_trace.size(), stream);
  if (status == cudaSuccess)
    status = device.system_errors.copy_to(results.system_errors.data(),
                                          results.system_errors.size(), stream);
  if (status == cudaSuccess)
    status = device.device_error.copy_to(&results.device_error, 1u, stream);
  return status;
}

int compare_success(const HostCase& host, const Results& actual) {
  CHECK(actual.device_error == 0u);
  CHECK(std::all_of(actual.system_errors.begin(), actual.system_errors.end(),
                    [](std::uint32_t error) { return error == 0u; }));
  for (std::size_t system = 0u; system < host.batch_size(); ++system) {
    const std::int64_t begin = host.matrix_offsets[system];
    const std::int64_t end = host.matrix_offsets[system + 1u];
    if (host.active[system] == 0u) {
      for (std::int64_t index = begin; index < end; ++index) {
        CHECK(actual.density[static_cast<std::size_t>(index)] == kSentinel);
        CHECK(actual.weighted_density[static_cast<std::size_t>(index)] == kSentinel);
      }
      CHECK(actual.band[system] == kSentinel && actual.occupation_sum[system] == kSentinel);
      CHECK(actual.density_trace[system] == kSentinel &&
            actual.weighted_trace[system] == kSentinel);
      continue;
    }
    for (std::int64_t index = begin; index < end; ++index) {
      CHECK(near(actual.density[static_cast<std::size_t>(index)],
                 host.expected_density[static_cast<std::size_t>(index)], 8.0e-13));
      CHECK(near(actual.weighted_density[static_cast<std::size_t>(index)],
                 host.expected_weighted_density[static_cast<std::size_t>(index)], 8.0e-13));
    }
    CHECK(near(actual.band[system], host.expected_band[system], 8.0e-13));
    CHECK(near(actual.occupation_sum[system], host.expected_occupation_sum[system], 8.0e-13));
    CHECK(near(actual.density_trace[system], host.expected_density_trace[system], 8.0e-13));
    CHECK(near(actual.weighted_trace[system], host.expected_weighted_trace[system], 8.0e-13));
  }
  return 0;
}

int run(DeviceFixture& device, const HostCase& host, cudaStream_t stream) {
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_density_device_errors_cuda(
      static_cast<std::int64_t>(host.batch_size()), device.system_errors.get(),
      device.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_density_cuda(
      device.batch(host), device.input(), device.results(), device.workspace(),
      device.system_errors.get(), device.device_error.get(), stream));
  return 0;
}

int test_ragged_cpu_parity_and_custom_stream() {
  for (const std::size_t batch_size : {1u, 8u, 32u, 128u}) {
    HostCase host = make_case(batch_size);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    DeviceFixture device;
    CUDA_CHECK(device.initialize(host, stream));
    CHECK(run(device, host, stream) == 0);
    Results actual;
    CUDA_CHECK(copy_results(host, device, actual, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(compare_success(host, actual) == 0);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
  return 0;
}

int test_identity_trace_and_alpha_beta_semantics() {
  HostCase host;
  host.orbital_offsets = {0, 3};
  host.matrix_offsets = {0, 9};
  host.coefficients = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  host.eigenvalues = {-0.7, 0.2, 1.3};
  host.occupations = {1.0, 0.4, 0.0, 0.8, 0.1, 0.25};
  host.active = {1u};
  build_reference(host);
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, nullptr));
  CHECK(run(device, host, nullptr) == 0);
  Results actual;
  CUDA_CHECK(copy_results(host, device, actual, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(compare_success(host, actual) == 0);
  CHECK(actual.density[0] == 1.8 && actual.density[4] == 0.5 && actual.density[8] == 0.25);
  CHECK(actual.density_trace[0] == actual.occupation_sum[0]);
  CHECK(actual.weighted_trace[0] == actual.band[0]);
  return 0;
}

HostCase extract_system(const HostCase& batch, std::size_t system) {
  HostCase single;
  const std::int64_t orbital_begin = batch.orbital_offsets[system];
  const std::int64_t orbital_end = batch.orbital_offsets[system + 1u];
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t matrix_end = batch.matrix_offsets[system + 1u];
  const std::int64_t count = orbital_end - orbital_begin;
  single.orbital_offsets = {0, count};
  single.matrix_offsets = {0, matrix_end - matrix_begin};
  single.coefficients.assign(batch.coefficients.begin() + matrix_begin,
                             batch.coefficients.begin() + matrix_end);
  single.eigenvalues.assign(batch.eigenvalues.begin() + orbital_begin,
                            batch.eigenvalues.begin() + orbital_end);
  single.occupations.resize(static_cast<std::size_t>(2 * count));
  std::copy_n(batch.occupations.begin() + 2 * orbital_begin, count, single.occupations.begin());
  std::copy_n(batch.occupations.begin() + 2 * orbital_begin + count, count,
              single.occupations.begin() + count);
  single.active = {1u};
  build_reference(single);
  return single;
}

int test_sequential_and_ragged_execution_are_identical() {
  HostCase batch = make_case(8u);
  DeviceFixture batch_device;
  CUDA_CHECK(batch_device.initialize(batch, nullptr));
  CHECK(run(batch_device, batch, nullptr) == 0);
  Results batch_results;
  CUDA_CHECK(copy_results(batch, batch_device, batch_results, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  for (const std::size_t system : {0u, 3u, 7u}) {
    HostCase single = extract_system(batch, system);
    DeviceFixture single_device;
    CUDA_CHECK(single_device.initialize(single, nullptr));
    CHECK(run(single_device, single, nullptr) == 0);
    Results single_results;
    CUDA_CHECK(copy_results(single, single_device, single_results, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::int64_t begin = batch.matrix_offsets[system];
    const std::int64_t end = batch.matrix_offsets[system + 1u];
    CHECK(std::equal(batch_results.density.begin() + begin, batch_results.density.begin() + end,
                     single_results.density.begin()));
    CHECK(std::equal(batch_results.weighted_density.begin() + begin,
                     batch_results.weighted_density.begin() + end,
                     single_results.weighted_density.begin()));
    CHECK(batch_results.band[system] == single_results.band[0]);
    CHECK(batch_results.occupation_sum[system] == single_results.occupation_sum[0]);
    CHECK(batch_results.density_trace[system] == single_results.density_trace[0]);
    CHECK(batch_results.weighted_trace[system] == single_results.weighted_trace[0]);
  }
  return 0;
}

int test_peer_isolated_input_arithmetic_trace_and_offset_failures() {
  HostCase host = make_case(8u);
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, nullptr));
  std::vector<double> coefficients = host.coefficients;
  std::vector<double> eigenvalues = host.eigenvalues;
  std::vector<double> occupations = host.occupations;
  coefficients[static_cast<std::size_t>(host.matrix_offsets[2])] =
      std::numeric_limits<double>::quiet_NaN();
  eigenvalues[static_cast<std::size_t>(host.orbital_offsets[3])] =
      std::numeric_limits<double>::infinity();
  occupations[static_cast<std::size_t>(2 * host.orbital_offsets[4])] = 1.5;
  coefficients[static_cast<std::size_t>(host.matrix_offsets[5])] =
      std::numeric_limits<double>::max();
  coefficients[static_cast<std::size_t>(host.matrix_offsets[6])] = 2.0;
  eigenvalues[static_cast<std::size_t>(host.orbital_offsets[6])] =
      std::numeric_limits<double>::max() / 2.0;
  const std::int64_t orbital_begin_7 = host.orbital_offsets[7];
  const std::int64_t orbital_count_7 = host.orbital_offsets[8] - orbital_begin_7;
  for (std::int64_t orbital = 0; orbital < orbital_count_7; ++orbital) {
    occupations[static_cast<std::size_t>(2 * orbital_begin_7 + orbital)] = 0.0;
    occupations[static_cast<std::size_t>(2 * orbital_begin_7 + orbital_count_7 + orbital)] = 0.0;
  }
  occupations[static_cast<std::size_t>(2 * orbital_begin_7)] = 1.0;
  occupations[static_cast<std::size_t>(2 * orbital_begin_7 + 1)] = 1.0;
  eigenvalues[static_cast<std::size_t>(orbital_begin_7)] = std::numeric_limits<double>::max();
  eigenvalues[static_cast<std::size_t>(orbital_begin_7 + 1)] = std::numeric_limits<double>::max();
  CUDA_CHECK(device.coefficients.copy_from(coefficients.data(), coefficients.size()));
  CUDA_CHECK(device.eigenvalues.copy_from(eigenvalues.data(), eigenvalues.size()));
  CUDA_CHECK(device.occupations.copy_from(occupations.data(), occupations.size()));
  CHECK(run(device, host, nullptr) == 0);
  Results actual;
  CUDA_CHECK(copy_results(host, device, actual, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(actual.system_errors[2] ==
        static_cast<std::uint32_t>(Gfn2DensityDeviceError::kNonfiniteCoefficient));
  CHECK(actual.system_errors[3] ==
        static_cast<std::uint32_t>(Gfn2DensityDeviceError::kNonfiniteEigenvalue));
  CHECK(actual.system_errors[4] ==
        static_cast<std::uint32_t>(Gfn2DensityDeviceError::kInvalidOccupation));
  CHECK(actual.system_errors[5] ==
        static_cast<std::uint32_t>(Gfn2DensityDeviceError::kNonfiniteDensityArithmetic));
  CHECK(actual.system_errors[6] ==
        static_cast<std::uint32_t>(Gfn2DensityDeviceError::kNonfiniteWeightedDensityArithmetic));
  CHECK(actual.system_errors[7] ==
        static_cast<std::uint32_t>(Gfn2DensityDeviceError::kNonfiniteBandEnergy));
  CHECK(actual.device_error ==
        static_cast<std::uint32_t>(Gfn2DensityDeviceError::kNonfiniteCoefficient));
  for (const std::size_t system : {2u, 3u, 4u, 5u, 6u, 7u}) {
    CHECK(actual.band[system] == kSentinel && actual.occupation_sum[system] == kSentinel);
    CHECK(actual.density_trace[system] == kSentinel && actual.weighted_trace[system] == kSentinel);
    for (std::int64_t index = host.matrix_offsets[system]; index < host.matrix_offsets[system + 1u];
         ++index) {
      CHECK(actual.density[static_cast<std::size_t>(index)] == kSentinel);
      CHECK(actual.weighted_density[static_cast<std::size_t>(index)] == kSentinel);
    }
  }
  CHECK(actual.system_errors[1] == 0u && actual.band[1] != kSentinel);

  /* Large finite diagonal densities whose trace overflows fail transactionally. */
  host = make_case(1u);
  const double large = std::sqrt(0.75 * std::numeric_limits<double>::max());
  std::fill(host.coefficients.begin(), host.coefficients.end(), 0.0);
  host.coefficients[0] = large;
  host.coefficients[8] = large;
  std::fill(host.eigenvalues.begin(), host.eigenvalues.end(), 0.0);
  std::fill(host.occupations.begin(), host.occupations.end(), 0.0);
  host.occupations[0] = 1.0;
  host.occupations[1] = 1.0;
  DeviceFixture trace_device;
  CUDA_CHECK(trace_device.initialize(host, nullptr));
  CHECK(run(trace_device, host, nullptr) == 0);
  CUDA_CHECK(copy_results(host, trace_device, actual, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(actual.system_errors[0] ==
        static_cast<std::uint32_t>(Gfn2DensityDeviceError::kNonfiniteTrace));
  CHECK(actual.band[0] == kSentinel && actual.occupation_sum[0] == kSentinel);
  CHECK(actual.density_trace[0] == kSentinel && actual.weighted_trace[0] == kSentinel);
  CHECK(std::all_of(actual.density.begin(), actual.density.end(),
                    [](double value) { return value == kSentinel; }));
  CHECK(std::all_of(actual.weighted_density.begin(), actual.weighted_density.end(),
                    [](double value) { return value == kSentinel; }));

  host = make_case(8u);
  DeviceFixture offset_device;
  CUDA_CHECK(offset_device.initialize(host, nullptr));
  std::vector<std::int64_t> orbital_offsets = host.orbital_offsets;
  std::vector<std::int64_t> matrix_offsets = host.matrix_offsets;
  orbital_offsets[3] = std::numeric_limits<std::int64_t>::max();
  matrix_offsets[6] = std::numeric_limits<std::int64_t>::min();
  CUDA_CHECK(
      offset_device.orbital_offsets.copy_from(orbital_offsets.data(), orbital_offsets.size()));
  CUDA_CHECK(offset_device.matrix_offsets.copy_from(matrix_offsets.data(), matrix_offsets.size()));
  CHECK(run(offset_device, host, nullptr) == 0);
  CUDA_CHECK(copy_results(host, offset_device, actual, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  for (const std::size_t system : {2u, 3u, 5u, 6u}) {
    CHECK(actual.system_errors[system] ==
          static_cast<std::uint32_t>(Gfn2DensityDeviceError::kInvalidOffsets));
  }
  CHECK(actual.system_errors[1] == 0u && actual.system_errors[4] == 0u);
  return 0;
}

int test_active_mask_sticky_error_and_graph_replay() {
  HostCase host = make_case(32u);
  host.active[4] = 0u;
  host.coefficients[static_cast<std::size_t>(host.matrix_offsets[4])] =
      std::numeric_limits<double>::quiet_NaN();
  build_reference(host);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<std::uint8_t> invalid_active = host.active;
  invalid_active[3] = 2u;
  CUDA_CHECK(device.active.copy_from(invalid_active.data(), invalid_active.size(), stream));
  CHECK(run(device, host, stream) == 0);
  Results invalid_results;
  CUDA_CHECK(copy_results(host, device, invalid_results, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(invalid_results.system_errors[3] ==
        static_cast<std::uint32_t>(Gfn2DensityDeviceError::kInvalidActiveMask));
  CHECK(invalid_results.band[3] == kSentinel);
  CUDA_CHECK(device.active.copy_from(host.active.data(), host.active.size(), stream));
  CUDA_CHECK(device.reset_outputs(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CHECK(run(device, host, stream) == 0);
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  for (int replay = 0; replay < 2; ++replay) {
    CUDA_CHECK(device.reset_outputs(host, stream));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    Results actual;
    CUDA_CHECK(copy_results(host, device, actual, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(compare_success(host, actual) == 0);
  }
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));

  CUDA_CHECK(device.reset_outputs(host, stream));
  CUDA_CHECK(cudaMemsetAsync(device.system_errors.get(), 0,
                             host.batch_size() * sizeof(std::uint32_t), stream));
  const std::uint32_t sticky = static_cast<std::uint32_t>(Gfn2DensityDeviceError::kNonfiniteTrace);
  CUDA_CHECK(cudaMemcpyAsync(device.device_error.get(), &sticky, sizeof(sticky),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_density_cuda(
      device.batch(host), device.input(), device.results(), device.workspace(),
      device.system_errors.get(), device.device_error.get(), stream));
  Results actual;
  CUDA_CHECK(copy_results(host, device, actual, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(actual.device_error == sticky);
  CHECK(std::all_of(actual.density.begin(), actual.density.end(),
                    [](double value) { return value == kSentinel; }));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_host_argument_alias_token_and_alignment_validation() {
  HostCase host = make_case(8u);
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, nullptr));
  Gfn2DensityDeviceBatch batch = device.batch(host);
  Gfn2DensityDeviceInput input = device.input();
  Gfn2DensityDeviceResults results = device.results();
  Gfn2DensityDeviceWorkspace workspace = device.workspace();
  batch.orbital_offsets = reinterpret_cast<const std::int64_t*>(
      reinterpret_cast<const unsigned char*>(device.orbital_offsets.get()) + 1u);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_density_cuda(
            batch, input, results, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  batch = device.batch(host);
  input.plan_token ^= 1u;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_density_cuda(
            batch, input, results, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  input = device.input();
  results.plan_token ^= 1u;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_density_cuda(
            batch, input, results, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  results = device.results();
  workspace.plan_token ^= 1u;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_density_cuda(
            batch, input, results, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  workspace = device.workspace();
  batch.plan_token = 0u;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_density_cuda(
            batch, input, results, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  batch = device.batch(host);
  batch.matrix_offsets = reinterpret_cast<const std::int64_t*>(
      reinterpret_cast<const unsigned char*>(device.matrix_offsets.get()) + 1u);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_density_cuda(
            batch, input, results, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  batch = device.batch(host);
  input.coefficients = reinterpret_cast<const double*>(
      reinterpret_cast<const unsigned char*>(device.coefficients.get()) + 1u);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_density_cuda(
            batch, input, results, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  input = device.input();
  results.density = workspace.density_scratch;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_density_cuda(
            batch, input, results, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  results = device.results();
  workspace.weights = const_cast<double*>(input.eigenvalues);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_density_cuda(
            batch, input, results, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  workspace = device.workspace();
  results.band_energies = reinterpret_cast<double*>(device.device_error.get());
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_restricted_density_cuda(
            batch, input, results, workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  CHECK(gpuxtb::detail::cuda::reset_gfn2_density_device_errors_cuda(
            8, device.system_errors.get(), device.system_errors.get()) == cudaErrorInvalidValue);
  return 0;
}

}  // namespace

int main() {
  if (const int line = test_ragged_cpu_parity_and_custom_stream(); line != 0) return line;
  if (const int line = test_identity_trace_and_alpha_beta_semantics(); line != 0) return line;
  if (const int line = test_sequential_and_ragged_execution_are_identical(); line != 0) return line;
  if (const int line = test_peer_isolated_input_arithmetic_trace_and_offset_failures(); line != 0)
    return line;
  if (const int line = test_active_mask_sticky_error_and_graph_replay(); line != 0) return line;
  if (const int line = test_host_argument_alias_token_and_alignment_validation(); line != 0)
    return line;
  return 0;
}
