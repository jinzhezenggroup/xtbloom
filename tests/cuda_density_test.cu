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

using gpuxtb::detail::Gfn2PlanMemorySpace;
using gpuxtb::detail::Gfn2WavefunctionLayoutView;
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

int test_rounded_weighted_contribution_overflow_is_transactional() {
  HostCase host = make_case(8u);
  constexpr std::size_t failed_system = 1u;
  const double maximum = std::numeric_limits<double>::max();
  const std::int64_t orbital_begin = host.orbital_offsets[failed_system];
  const std::int64_t count = host.orbital_offsets[failed_system + 1u] - orbital_begin;
  const std::int64_t matrix_begin = host.matrix_offsets[failed_system];
  CHECK(count == 2);

  std::fill(host.coefficients.begin() + matrix_begin,
            host.coefficients.begin() + host.matrix_offsets[failed_system + 1u], 0.0);
  /* The second standalone binary64 contribution overflows, but its exact FMA
   * result cancels against the first term and remains finite. */
  host.coefficients[static_cast<std::size_t>(matrix_begin)] = 1.0;
  host.coefficients[static_cast<std::size_t>(matrix_begin + 1)] = 1.25;
  for (std::int64_t local = 0; local < count; ++local) {
    host.occupations[static_cast<std::size_t>(2 * orbital_begin + local)] = 1.0;
    host.occupations[static_cast<std::size_t>(2 * orbital_begin + count + local)] = 0.0;
  }
  host.eigenvalues[static_cast<std::size_t>(orbital_begin)] = -0.75 * maximum;
  host.eigenvalues[static_cast<std::size_t>(orbital_begin + 1)] = 0.75 * maximum;

  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, nullptr));
  CHECK(run(device, host, nullptr) == 0);
  Results actual;
  CUDA_CHECK(copy_results(host, device, actual, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(actual.system_errors[failed_system] ==
        static_cast<std::uint32_t>(Gfn2DensityDeviceError::kNonfiniteWeightedDensityArithmetic));
  CHECK(actual.device_error ==
        static_cast<std::uint32_t>(Gfn2DensityDeviceError::kNonfiniteWeightedDensityArithmetic));
  for (std::int64_t index = matrix_begin; index < host.matrix_offsets[failed_system + 1u];
       ++index) {
    CHECK(actual.density[static_cast<std::size_t>(index)] == kSentinel);
    CHECK(actual.weighted_density[static_cast<std::size_t>(index)] == kSentinel);
  }
  CHECK(actual.band[failed_system] == kSentinel &&
        actual.occupation_sum[failed_system] == kSentinel);
  CHECK(actual.density_trace[failed_system] == kSentinel &&
        actual.weighted_trace[failed_system] == kSentinel);
  CHECK(actual.system_errors[0] == 0u && actual.band[0] != kSentinel);

  /* Weighted arithmetic fails at local 0; a later density overflow must not
   * replace that first diagnostic. */
  std::fill(host.coefficients.begin() + matrix_begin,
            host.coefficients.begin() + host.matrix_offsets[failed_system + 1u], 0.0);
  host.coefficients[static_cast<std::size_t>(matrix_begin)] = 2.0;
  host.coefficients[static_cast<std::size_t>(matrix_begin + 1)] = 2.0 * std::sqrt(maximum);
  host.eigenvalues[static_cast<std::size_t>(orbital_begin)] = 0.75 * maximum;
  host.eigenvalues[static_cast<std::size_t>(orbital_begin + 1)] = -0.75 * maximum;
  DeviceFixture precedence_device;
  CUDA_CHECK(precedence_device.initialize(host, nullptr));
  CHECK(run(precedence_device, host, nullptr) == 0);
  CUDA_CHECK(copy_results(host, precedence_device, actual, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(actual.system_errors[failed_system] ==
        static_cast<std::uint32_t>(Gfn2DensityDeviceError::kNonfiniteWeightedDensityArithmetic));
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

struct SpinHostCase {
  std::vector<std::int64_t> orbital_offsets;
  std::vector<std::int64_t> matrix_offsets;
  std::vector<std::int32_t> spin_channels;
  std::vector<std::int64_t> spin_orbital_offsets;
  std::vector<std::int64_t> spin_matrix_offsets;
  std::vector<std::int64_t> spin_channel_offsets;
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
  std::vector<double> expected_channel_band;
  std::vector<double> expected_channel_occupation_sum;
  std::vector<double> expected_channel_density_trace;
  std::vector<double> expected_channel_weighted_trace;

  std::size_t batch_size() const { return active.size(); }
  std::size_t total_orbitals() const { return static_cast<std::size_t>(orbital_offsets.back()); }
  std::size_t total_matrix_elements() const {
    return static_cast<std::size_t>(matrix_offsets.back());
  }
  std::size_t total_spin_orbitals() const { return eigenvalues.size(); }
  std::size_t total_spin_matrix_elements() const { return coefficients.size(); }
  std::size_t total_spin_channels() const {
    return static_cast<std::size_t>(spin_channel_offsets.back());
  }
};

void build_spin_reference(SpinHostCase& host) {
  host.expected_density.assign(host.total_spin_matrix_elements(), kSentinel);
  host.expected_weighted_density.assign(host.total_spin_matrix_elements(), kSentinel);
  host.expected_band.assign(host.batch_size(), kSentinel);
  host.expected_occupation_sum.assign(host.batch_size(), kSentinel);
  host.expected_density_trace.assign(host.batch_size(), kSentinel);
  host.expected_weighted_trace.assign(host.batch_size(), kSentinel);
  host.expected_channel_band.assign(host.total_spin_channels(), kSentinel);
  host.expected_channel_occupation_sum.assign(host.total_spin_channels(), kSentinel);
  host.expected_channel_density_trace.assign(host.total_spin_channels(), kSentinel);
  host.expected_channel_weighted_trace.assign(host.total_spin_channels(), kSentinel);
  for (std::size_t system = 0u; system < host.batch_size(); ++system) {
    if (host.active[system] == 0u) {
      continue;
    }
    const std::int64_t orbital_begin = host.orbital_offsets[system];
    const std::int64_t count = host.orbital_offsets[system + 1u] - orbital_begin;
    const std::int32_t channels = host.spin_channels[system];
    const std::int64_t spin_orbital_begin = host.spin_orbital_offsets[system];
    const std::int64_t spin_matrix_begin = host.spin_matrix_offsets[system];
    const std::int64_t channel_begin = host.spin_channel_offsets[system];
    for (std::int32_t channel = 0; channel < channels; ++channel) {
      double band = 0.0;
      double occupation_sum = 0.0;
      for (std::int64_t orbital = 0; orbital < count; ++orbital) {
        const double alpha =
            host.occupations[static_cast<std::size_t>(2 * orbital_begin + orbital)];
        const double beta =
            host.occupations[static_cast<std::size_t>(2 * orbital_begin + count + orbital)];
        const double occupation = channels == 1 ? alpha + beta : (channel == 0 ? alpha : beta);
        const double eigenvalue = host.eigenvalues[static_cast<std::size_t>(
            spin_orbital_begin + channel * count + orbital)];
        occupation_sum += occupation;
        band += occupation * eigenvalue;
      }
      double density_trace = 0.0;
      double weighted_trace = 0.0;
      const std::int64_t matrix_begin = spin_matrix_begin + channel * count * count;
      for (std::int64_t row = 0; row < count; ++row) {
        for (std::int64_t column = 0; column < count; ++column) {
          double density = 0.0;
          double weighted = 0.0;
          for (std::int64_t orbital = 0; orbital < count; ++orbital) {
            const double alpha =
                host.occupations[static_cast<std::size_t>(2 * orbital_begin + orbital)];
            const double beta =
                host.occupations[static_cast<std::size_t>(2 * orbital_begin + count + orbital)];
            const double occupation = channels == 1 ? alpha + beta : (channel == 0 ? alpha : beta);
            const double first =
                host.coefficients[static_cast<std::size_t>(matrix_begin + row * count + orbital)];
            const double second = host.coefficients[static_cast<std::size_t>(
                matrix_begin + column * count + orbital)];
            const double eigenvalue = host.eigenvalues[static_cast<std::size_t>(
                spin_orbital_begin + channel * count + orbital)];
            density = std::fma(first * occupation, second, density);
            weighted = std::fma(first * occupation * eigenvalue, second, weighted);
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
      const std::size_t diagnostic = static_cast<std::size_t>(channel_begin + channel);
      host.expected_channel_band[diagnostic] = band;
      host.expected_channel_occupation_sum[diagnostic] = occupation_sum;
      host.expected_channel_density_trace[diagnostic] = density_trace;
      host.expected_channel_weighted_trace[diagnostic] = weighted_trace;
    }
    const std::size_t alpha = static_cast<std::size_t>(channel_begin);
    double band = host.expected_channel_band[alpha];
    double occupation = host.expected_channel_occupation_sum[alpha];
    double density_trace = host.expected_channel_density_trace[alpha];
    double weighted_trace = host.expected_channel_weighted_trace[alpha];
    if (channels == 2) {
      band += host.expected_channel_band[alpha + 1u];
      occupation += host.expected_channel_occupation_sum[alpha + 1u];
      density_trace += host.expected_channel_density_trace[alpha + 1u];
      weighted_trace += host.expected_channel_weighted_trace[alpha + 1u];
    }
    host.expected_band[system] = band;
    host.expected_occupation_sum[system] = occupation;
    host.expected_density_trace[system] = density_trace;
    host.expected_weighted_trace[system] = weighted_trace;
  }
}

SpinHostCase make_spin_case(std::size_t batch_size) {
  SpinHostCase host;
  host.orbital_offsets.assign(batch_size + 1u, 0);
  host.matrix_offsets.assign(batch_size + 1u, 0);
  host.spin_orbital_offsets.assign(batch_size + 1u, 0);
  host.spin_matrix_offsets.assign(batch_size + 1u, 0);
  host.spin_channel_offsets.assign(batch_size + 1u, 0);
  host.spin_channels.resize(batch_size);
  host.active.assign(batch_size, 1u);
  std::int64_t orbitals = 0;
  std::int64_t matrices = 0;
  std::int64_t spin_orbitals = 0;
  std::int64_t spin_matrices = 0;
  std::int64_t spin_channels = 0;
  for (std::size_t system = 0u; system < batch_size; ++system) {
    const std::int64_t count = batch_size == 1u ? 6 : 1 + static_cast<std::int64_t>(system % 7u);
    const std::int32_t channels = batch_size == 1u || system % 2u == 1u ? 2 : 1;
    host.orbital_offsets[system] = orbitals;
    host.matrix_offsets[system] = matrices;
    host.spin_orbital_offsets[system] = spin_orbitals;
    host.spin_matrix_offsets[system] = spin_matrices;
    host.spin_channel_offsets[system] = spin_channels;
    host.spin_channels[system] = channels;
    orbitals += count;
    matrices += count * count;
    spin_orbitals += channels * count;
    spin_matrices += channels * count * count;
    spin_channels += channels;
  }
  host.orbital_offsets[batch_size] = orbitals;
  host.matrix_offsets[batch_size] = matrices;
  host.spin_orbital_offsets[batch_size] = spin_orbitals;
  host.spin_matrix_offsets[batch_size] = spin_matrices;
  host.spin_channel_offsets[batch_size] = spin_channels;
  host.coefficients.resize(static_cast<std::size_t>(spin_matrices));
  host.eigenvalues.resize(static_cast<std::size_t>(spin_orbitals));
  host.occupations.resize(static_cast<std::size_t>(2 * orbitals));
  for (std::size_t system = 0u; system < batch_size; ++system) {
    const std::int64_t orbital_begin = host.orbital_offsets[system];
    const std::int64_t count = host.orbital_offsets[system + 1u] - orbital_begin;
    const std::int64_t spin_orbital_begin = host.spin_orbital_offsets[system];
    const std::int64_t spin_matrix_begin = host.spin_matrix_offsets[system];
    for (std::int32_t channel = 0; channel < host.spin_channels[system]; ++channel) {
      for (std::int64_t row = 0; row < count; ++row) {
        for (std::int64_t orbital = 0; orbital < count; ++orbital) {
          host.coefficients[static_cast<std::size_t>(spin_matrix_begin + channel * count * count +
                                                     row * count + orbital)] =
              (row == orbital ? 0.79 + 0.035 * channel
                              : 0.012 * static_cast<double>(1 + (row + 3 * orbital) % 11)) +
              0.0002 * static_cast<double>(system);
        }
      }
      for (std::int64_t orbital = 0; orbital < count; ++orbital) {
        host.eigenvalues[static_cast<std::size_t>(spin_orbital_begin + channel * count + orbital)] =
            -0.82 + 0.16 * static_cast<double>(orbital) + 0.047 * channel +
            0.001 * static_cast<double>(system);
      }
    }
    for (std::int64_t orbital = 0; orbital < count; ++orbital) {
      host.occupations[static_cast<std::size_t>(2 * orbital_begin + orbital)] =
          std::min(1.0, 0.11 * static_cast<double>(1 + (system + orbital) % 9));
      host.occupations[static_cast<std::size_t>(2 * orbital_begin + count + orbital)] =
          std::min(1.0, 0.07 * static_cast<double>(1 + (2 * system + orbital) % 11));
    }
    if (batch_size > 16u && system == 16u) {
      /* Inactive numerical fields are deliberately poisoned in both channels. */
      host.active[system] = 0u;
      host.coefficients[static_cast<std::size_t>(spin_matrix_begin)] =
          std::numeric_limits<double>::quiet_NaN();
      host.eigenvalues[static_cast<std::size_t>(spin_orbital_begin)] =
          std::numeric_limits<double>::infinity();
      host.occupations[static_cast<std::size_t>(2 * orbital_begin)] = 9.0;
      if (host.spin_channels[system] == 2) {
        host.coefficients[static_cast<std::size_t>(spin_matrix_begin + count * count)] =
            std::numeric_limits<double>::quiet_NaN();
        host.eigenvalues[static_cast<std::size_t>(spin_orbital_begin + count)] =
            -std::numeric_limits<double>::infinity();
      }
    }
  }
  build_spin_reference(host);
  return host;
}

struct SpinDeviceFixture {
  DeviceBuffer<std::int64_t> orbital_offsets;
  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<std::int32_t> spin_channels;
  DeviceBuffer<std::int64_t> spin_orbital_offsets;
  DeviceBuffer<std::int64_t> spin_matrix_offsets;
  DeviceBuffer<std::int64_t> spin_channel_offsets;
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
  DeviceBuffer<double> channel_band;
  DeviceBuffer<double> channel_occupation_sum;
  DeviceBuffer<double> channel_density_trace;
  DeviceBuffer<double> channel_weighted_trace;
  DeviceBuffer<double> density_scratch;
  DeviceBuffer<double> weighted_density_scratch;
  DeviceBuffer<double> weights;
  DeviceBuffer<double> energy_weights;
  DeviceBuffer<double> band_scratch;
  DeviceBuffer<double> occupation_sum_scratch;
  DeviceBuffer<double> density_trace_scratch;
  DeviceBuffer<double> weighted_trace_scratch;
  DeviceBuffer<double> channel_band_scratch;
  DeviceBuffer<double> channel_occupation_sum_scratch;
  DeviceBuffer<double> channel_density_trace_scratch;
  DeviceBuffer<double> channel_weighted_trace_scratch;
  DeviceBuffer<std::uint32_t> sequence_active;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;

  cudaError_t initialize(const SpinHostCase& host, cudaStream_t stream) {
    cudaError_t status = allocate_and_copy(orbital_offsets, host.orbital_offsets, stream);
    if (status == cudaSuccess)
      status = allocate_and_copy(matrix_offsets, host.matrix_offsets, stream);
    if (status == cudaSuccess)
      status = allocate_and_copy(spin_channels, host.spin_channels, stream);
    if (status == cudaSuccess)
      status = allocate_and_copy(spin_orbital_offsets, host.spin_orbital_offsets, stream);
    if (status == cudaSuccess)
      status = allocate_and_copy(spin_matrix_offsets, host.spin_matrix_offsets, stream);
    if (status == cudaSuccess)
      status = allocate_and_copy(spin_channel_offsets, host.spin_channel_offsets, stream);
    if (status == cudaSuccess) status = allocate_and_copy(coefficients, host.coefficients, stream);
    if (status == cudaSuccess) status = allocate_and_copy(eigenvalues, host.eigenvalues, stream);
    if (status == cudaSuccess) status = allocate_and_copy(occupations, host.occupations, stream);
    if (status == cudaSuccess) status = allocate_and_copy(active, host.active, stream);
    const std::size_t batch = host.batch_size();
    const std::size_t matrices = host.total_spin_matrix_elements();
    const std::size_t orbitals = host.total_spin_orbitals();
    const std::size_t channels = host.total_spin_channels();
    for (DeviceBuffer<double>* buffer :
         {&density, &weighted_density, &density_scratch, &weighted_density_scratch}) {
      if (status == cudaSuccess) status = buffer->allocate(matrices);
    }
    for (DeviceBuffer<double>* buffer : {&weights, &energy_weights}) {
      if (status == cudaSuccess) status = buffer->allocate(orbitals);
    }
    for (DeviceBuffer<double>* buffer :
         {&band, &occupation_sum, &density_trace, &weighted_trace, &band_scratch,
          &occupation_sum_scratch, &density_trace_scratch, &weighted_trace_scratch}) {
      if (status == cudaSuccess) status = buffer->allocate(batch);
    }
    for (DeviceBuffer<double>* buffer :
         {&channel_band, &channel_occupation_sum, &channel_density_trace, &channel_weighted_trace,
          &channel_band_scratch, &channel_occupation_sum_scratch, &channel_density_trace_scratch,
          &channel_weighted_trace_scratch}) {
      if (status == cudaSuccess) status = buffer->allocate(channels);
    }
    if (status == cudaSuccess) status = sequence_active.allocate(1u);
    if (status == cudaSuccess) status = system_errors.allocate(batch);
    if (status == cudaSuccess) status = device_error.allocate(1u);
    return status == cudaSuccess ? reset_outputs(host, stream) : status;
  }

  cudaError_t reset_outputs(const SpinHostCase& host, cudaStream_t stream) {
    const std::vector<double> matrix_seed(host.total_spin_matrix_elements(), kSentinel);
    const std::vector<double> system_seed(host.batch_size(), kSentinel);
    const std::vector<double> channel_seed(host.total_spin_channels(), kSentinel);
    cudaError_t status = density.copy_from(matrix_seed.data(), matrix_seed.size(), stream);
    if (status == cudaSuccess)
      status = weighted_density.copy_from(matrix_seed.data(), matrix_seed.size(), stream);
    for (DeviceBuffer<double>* buffer : {&band, &occupation_sum, &density_trace, &weighted_trace}) {
      if (status == cudaSuccess)
        status = buffer->copy_from(system_seed.data(), system_seed.size(), stream);
    }
    for (DeviceBuffer<double>* buffer : {&channel_band, &channel_occupation_sum,
                                         &channel_density_trace, &channel_weighted_trace}) {
      if (status == cudaSuccess)
        status = buffer->copy_from(channel_seed.data(), channel_seed.size(), stream);
    }
    return status;
  }

  Gfn2DensityDeviceBatch batch(const SpinHostCase& host) const {
    Gfn2DensityDeviceBatch value{};
    value.batch_size = static_cast<std::int64_t>(host.batch_size());
    value.total_orbitals = static_cast<std::int64_t>(host.total_orbitals());
    value.total_matrix_elements = static_cast<std::int64_t>(host.total_matrix_elements());
    value.orbital_offset_count = static_cast<std::int64_t>(host.orbital_offsets.size());
    value.matrix_offset_count = static_cast<std::int64_t>(host.matrix_offsets.size());
    value.plan_token = kPlanToken;
    value.orbital_offsets = orbital_offsets.get();
    value.matrix_offsets = matrix_offsets.get();
    return value;
  }

  Gfn2WavefunctionLayoutView layout(const SpinHostCase& host) const {
    Gfn2WavefunctionLayoutView value{};
    value.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
    value.plan_token = kPlanToken;
    value.batch_size = static_cast<std::int64_t>(host.batch_size());
    value.total_spin_channels = static_cast<std::int64_t>(host.total_spin_channels());
    value.total_spin_orbitals = static_cast<std::int64_t>(host.total_spin_orbitals());
    value.total_spin_matrix_elements = static_cast<std::int64_t>(host.total_spin_matrix_elements());
    value.spin_channel_count = static_cast<std::int64_t>(host.spin_channels.size());
    value.spin_channel_offset_count = static_cast<std::int64_t>(host.spin_channel_offsets.size());
    value.spin_orbital_offset_count = static_cast<std::int64_t>(host.spin_orbital_offsets.size());
    value.spin_matrix_offset_count = static_cast<std::int64_t>(host.spin_matrix_offsets.size());
    value.spin_channels = spin_channels.get();
    value.spin_channel_offsets = spin_channel_offsets.get();
    value.spin_orbital_offsets = spin_orbital_offsets.get();
    value.spin_matrix_offsets = spin_matrix_offsets.get();
    return value;
  }

  Gfn2DensityDeviceInput input() const {
    return {coefficients.get(), static_cast<std::int64_t>(coefficients.size()),
            eigenvalues.get(),  static_cast<std::int64_t>(eigenvalues.size()),
            occupations.get(),  static_cast<std::int64_t>(occupations.size()),
            active.get(),       static_cast<std::int64_t>(active.size()),
            kPlanToken};
  }

  Gfn2DensityDeviceResults results() {
    Gfn2DensityDeviceResults value{};
    value.density = density.get();
    value.density_elements = static_cast<std::int64_t>(density.size());
    value.energy_weighted_density = weighted_density.get();
    value.weighted_density_elements = static_cast<std::int64_t>(weighted_density.size());
    value.band_energies = band.get();
    value.band_energy_elements = static_cast<std::int64_t>(band.size());
    value.occupation_sums = occupation_sum.get();
    value.occupation_sum_elements = static_cast<std::int64_t>(occupation_sum.size());
    value.density_traces = density_trace.get();
    value.density_trace_elements = static_cast<std::int64_t>(density_trace.size());
    value.weighted_density_traces = weighted_trace.get();
    value.weighted_density_trace_elements = static_cast<std::int64_t>(weighted_trace.size());
    value.plan_token = kPlanToken;
    value.channel_band_energies = channel_band.get();
    value.channel_band_energy_elements = static_cast<std::int64_t>(channel_band.size());
    value.channel_occupation_sums = channel_occupation_sum.get();
    value.channel_occupation_sum_elements =
        static_cast<std::int64_t>(channel_occupation_sum.size());
    value.channel_density_traces = channel_density_trace.get();
    value.channel_density_trace_elements = static_cast<std::int64_t>(channel_density_trace.size());
    value.channel_weighted_density_traces = channel_weighted_trace.get();
    value.channel_weighted_density_trace_elements =
        static_cast<std::int64_t>(channel_weighted_trace.size());
    return value;
  }

  Gfn2DensityDeviceWorkspace workspace() {
    Gfn2DensityDeviceWorkspace value{};
    value.density_scratch = density_scratch.get();
    value.density_elements = static_cast<std::int64_t>(density_scratch.size());
    value.weighted_density_scratch = weighted_density_scratch.get();
    value.weighted_density_elements = static_cast<std::int64_t>(weighted_density_scratch.size());
    value.weights = weights.get();
    value.weight_elements = static_cast<std::int64_t>(weights.size());
    value.energy_weights = energy_weights.get();
    value.energy_weight_elements = static_cast<std::int64_t>(energy_weights.size());
    value.band_energy_scratch = band_scratch.get();
    value.band_energy_elements = static_cast<std::int64_t>(band_scratch.size());
    value.occupation_sum_scratch = occupation_sum_scratch.get();
    value.occupation_sum_elements = static_cast<std::int64_t>(occupation_sum_scratch.size());
    value.density_trace_scratch = density_trace_scratch.get();
    value.density_trace_elements = static_cast<std::int64_t>(density_trace_scratch.size());
    value.weighted_density_trace_scratch = weighted_trace_scratch.get();
    value.weighted_density_trace_elements =
        static_cast<std::int64_t>(weighted_trace_scratch.size());
    value.sequence_active = sequence_active.get();
    value.sequence_active_elements = static_cast<std::int64_t>(sequence_active.size());
    value.plan_token = kPlanToken;
    value.channel_band_energy_scratch = channel_band_scratch.get();
    value.channel_band_energy_elements = static_cast<std::int64_t>(channel_band_scratch.size());
    value.channel_occupation_sum_scratch = channel_occupation_sum_scratch.get();
    value.channel_occupation_sum_elements =
        static_cast<std::int64_t>(channel_occupation_sum_scratch.size());
    value.channel_density_trace_scratch = channel_density_trace_scratch.get();
    value.channel_density_trace_elements =
        static_cast<std::int64_t>(channel_density_trace_scratch.size());
    value.channel_weighted_density_trace_scratch = channel_weighted_trace_scratch.get();
    value.channel_weighted_density_trace_elements =
        static_cast<std::int64_t>(channel_weighted_trace_scratch.size());
    return value;
  }
};

struct SpinResults {
  std::vector<double> density;
  std::vector<double> weighted_density;
  std::vector<double> band;
  std::vector<double> occupation_sum;
  std::vector<double> density_trace;
  std::vector<double> weighted_trace;
  std::vector<double> channel_band;
  std::vector<double> channel_occupation_sum;
  std::vector<double> channel_density_trace;
  std::vector<double> channel_weighted_trace;
  std::vector<std::uint32_t> system_errors;
  std::uint32_t device_error = 0u;
};

cudaError_t copy_spin_results(const SpinHostCase& host, const SpinDeviceFixture& device,
                              SpinResults& results, cudaStream_t stream) {
  results.density.resize(host.total_spin_matrix_elements());
  results.weighted_density.resize(host.total_spin_matrix_elements());
  results.band.resize(host.batch_size());
  results.occupation_sum.resize(host.batch_size());
  results.density_trace.resize(host.batch_size());
  results.weighted_trace.resize(host.batch_size());
  results.channel_band.resize(host.total_spin_channels());
  results.channel_occupation_sum.resize(host.total_spin_channels());
  results.channel_density_trace.resize(host.total_spin_channels());
  results.channel_weighted_trace.resize(host.total_spin_channels());
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
    status = device.channel_band.copy_to(results.channel_band.data(), results.channel_band.size(),
                                         stream);
  if (status == cudaSuccess)
    status = device.channel_occupation_sum.copy_to(results.channel_occupation_sum.data(),
                                                   results.channel_occupation_sum.size(), stream);
  if (status == cudaSuccess)
    status = device.channel_density_trace.copy_to(results.channel_density_trace.data(),
                                                  results.channel_density_trace.size(), stream);
  if (status == cudaSuccess)
    status = device.channel_weighted_trace.copy_to(results.channel_weighted_trace.data(),
                                                   results.channel_weighted_trace.size(), stream);
  if (status == cudaSuccess)
    status = device.system_errors.copy_to(results.system_errors.data(),
                                          results.system_errors.size(), stream);
  if (status == cudaSuccess)
    status = device.device_error.copy_to(&results.device_error, 1u, stream);
  return status;
}

int run_spin(SpinDeviceFixture& device, const SpinHostCase& host, cudaStream_t stream) {
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_density_device_errors_cuda(
      static_cast<std::int64_t>(host.batch_size()), device.system_errors.get(),
      device.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_spin_density_cuda(
      device.batch(host), device.layout(host), device.input(), device.results(), device.workspace(),
      device.system_errors.get(), device.device_error.get(), stream));
  return 0;
}

int compare_spin_success(const SpinHostCase& host, const SpinResults& actual) {
  CHECK(actual.device_error == 0u);
  CHECK(std::all_of(actual.system_errors.begin(), actual.system_errors.end(),
                    [](std::uint32_t error) { return error == 0u; }));
  for (std::size_t system = 0u; system < host.batch_size(); ++system) {
    const std::int64_t matrix_begin = host.spin_matrix_offsets[system];
    const std::int64_t matrix_end = host.spin_matrix_offsets[system + 1u];
    const std::int64_t channel_begin = host.spin_channel_offsets[system];
    const std::int64_t channel_end = host.spin_channel_offsets[system + 1u];
    if (host.active[system] == 0u) {
      for (std::int64_t index = matrix_begin; index < matrix_end; ++index) {
        CHECK(actual.density[static_cast<std::size_t>(index)] == kSentinel);
        CHECK(actual.weighted_density[static_cast<std::size_t>(index)] == kSentinel);
      }
      for (std::int64_t channel = channel_begin; channel < channel_end; ++channel) {
        CHECK(actual.channel_band[static_cast<std::size_t>(channel)] == kSentinel);
        CHECK(actual.channel_occupation_sum[static_cast<std::size_t>(channel)] == kSentinel);
        CHECK(actual.channel_density_trace[static_cast<std::size_t>(channel)] == kSentinel);
        CHECK(actual.channel_weighted_trace[static_cast<std::size_t>(channel)] == kSentinel);
      }
      CHECK(actual.band[system] == kSentinel && actual.occupation_sum[system] == kSentinel);
      CHECK(actual.density_trace[system] == kSentinel &&
            actual.weighted_trace[system] == kSentinel);
      continue;
    }
    for (std::int64_t index = matrix_begin; index < matrix_end; ++index) {
      CHECK(near(actual.density[static_cast<std::size_t>(index)],
                 host.expected_density[static_cast<std::size_t>(index)], 8.0e-13));
      CHECK(near(actual.weighted_density[static_cast<std::size_t>(index)],
                 host.expected_weighted_density[static_cast<std::size_t>(index)], 8.0e-13));
    }
    for (std::int64_t channel = channel_begin; channel < channel_end; ++channel) {
      const std::size_t index = static_cast<std::size_t>(channel);
      CHECK(near(actual.channel_band[index], host.expected_channel_band[index], 8.0e-13));
      CHECK(near(actual.channel_occupation_sum[index], host.expected_channel_occupation_sum[index],
                 8.0e-13));
      CHECK(near(actual.channel_density_trace[index], host.expected_channel_density_trace[index],
                 8.0e-13));
      CHECK(near(actual.channel_weighted_trace[index], host.expected_channel_weighted_trace[index],
                 8.0e-13));
    }
    CHECK(near(actual.band[system], host.expected_band[system], 8.0e-13));
    CHECK(near(actual.occupation_sum[system], host.expected_occupation_sum[system], 8.0e-13));
    CHECK(near(actual.density_trace[system], host.expected_density_trace[system], 8.0e-13));
    CHECK(near(actual.weighted_trace[system], host.expected_weighted_trace[system], 8.0e-13));
  }
  return 0;
}

int test_spin_entry_preserves_restricted_outputs_exactly() {
  HostCase restricted = make_case(8u);
  SpinHostCase spin;
  spin.orbital_offsets = restricted.orbital_offsets;
  spin.matrix_offsets = restricted.matrix_offsets;
  spin.spin_channels.assign(restricted.batch_size(), 1);
  spin.spin_orbital_offsets = restricted.orbital_offsets;
  spin.spin_matrix_offsets = restricted.matrix_offsets;
  spin.spin_channel_offsets.resize(restricted.batch_size() + 1u);
  std::iota(spin.spin_channel_offsets.begin(), spin.spin_channel_offsets.end(), 0);
  spin.coefficients = restricted.coefficients;
  spin.eigenvalues = restricted.eigenvalues;
  spin.occupations = restricted.occupations;
  spin.active = restricted.active;
  build_spin_reference(spin);

  DeviceFixture restricted_device;
  CUDA_CHECK(restricted_device.initialize(restricted, nullptr));
  CHECK(run(restricted_device, restricted, nullptr) == 0);
  Results restricted_results;
  CUDA_CHECK(copy_results(restricted, restricted_device, restricted_results, nullptr));
  SpinDeviceFixture spin_device;
  CUDA_CHECK(spin_device.initialize(spin, nullptr));
  CHECK(run_spin(spin_device, spin, nullptr) == 0);
  SpinResults spin_results;
  CUDA_CHECK(copy_spin_results(spin, spin_device, spin_results, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(restricted_results.density == spin_results.density);
  CHECK(restricted_results.weighted_density == spin_results.weighted_density);
  CHECK(restricted_results.band == spin_results.band);
  CHECK(restricted_results.occupation_sum == spin_results.occupation_sum);
  CHECK(restricted_results.density_trace == spin_results.density_trace);
  CHECK(restricted_results.weighted_trace == spin_results.weighted_trace);
  CHECK(spin_results.channel_band == spin_results.band);
  CHECK(spin_results.channel_occupation_sum == spin_results.occupation_sum);
  CHECK(spin_results.channel_density_trace == spin_results.density_trace);
  CHECK(spin_results.channel_weighted_trace == spin_results.weighted_trace);
  return 0;
}

int test_spin_mixed_ragged_custom_stream_and_inactive_poison() {
  for (const std::size_t batch_size : {1u, 8u, 32u, 128u}) {
    SpinHostCase host = make_spin_case(batch_size);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    SpinDeviceFixture device;
    CUDA_CHECK(device.initialize(host, stream));
    CHECK(run_spin(device, host, stream) == 0);
    SpinResults actual;
    CUDA_CHECK(copy_spin_results(host, device, actual, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(compare_spin_success(host, actual) == 0);
    if (batch_size == 1u) {
      const std::int64_t matrix_count = host.matrix_offsets[1];
      CHECK(!near(actual.density[0], actual.density[static_cast<std::size_t>(matrix_count)],
                  1.0e-15));
      CHECK(actual.channel_band[0] != actual.channel_band[1]);
      CHECK(actual.band[0] == actual.channel_band[0] + actual.channel_band[1]);
    }
    if (batch_size > 16u) {
      CHECK(actual.system_errors[16] == 0u);
      CHECK(actual.band[16] == kSentinel);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
  return 0;
}

int test_spin_beta_failure_is_transactional() {
  SpinHostCase host = make_spin_case(8u);
  constexpr std::size_t failed_system = 1u;
  CHECK(host.spin_channels[failed_system] == 2);
  SpinDeviceFixture device;
  CUDA_CHECK(device.initialize(host, nullptr));
  std::vector<double> eigenvalues = host.eigenvalues;
  const std::int64_t count =
      host.orbital_offsets[failed_system + 1u] - host.orbital_offsets[failed_system];
  eigenvalues[static_cast<std::size_t>(host.spin_orbital_offsets[failed_system] + count)] =
      std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(device.eigenvalues.copy_from(eigenvalues.data(), eigenvalues.size()));
  CHECK(run_spin(device, host, nullptr) == 0);
  SpinResults actual;
  CUDA_CHECK(copy_spin_results(host, device, actual, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(actual.system_errors[failed_system] ==
        static_cast<std::uint32_t>(Gfn2DensityDeviceError::kNonfiniteEigenvalue));
  CHECK(actual.device_error ==
        static_cast<std::uint32_t>(Gfn2DensityDeviceError::kNonfiniteEigenvalue));
  for (std::int64_t index = host.spin_matrix_offsets[failed_system];
       index < host.spin_matrix_offsets[failed_system + 1u]; ++index) {
    CHECK(actual.density[static_cast<std::size_t>(index)] == kSentinel);
    CHECK(actual.weighted_density[static_cast<std::size_t>(index)] == kSentinel);
  }
  for (std::int64_t channel = host.spin_channel_offsets[failed_system];
       channel < host.spin_channel_offsets[failed_system + 1u]; ++channel) {
    CHECK(actual.channel_band[static_cast<std::size_t>(channel)] == kSentinel);
    CHECK(actual.channel_occupation_sum[static_cast<std::size_t>(channel)] == kSentinel);
    CHECK(actual.channel_density_trace[static_cast<std::size_t>(channel)] == kSentinel);
    CHECK(actual.channel_weighted_trace[static_cast<std::size_t>(channel)] == kSentinel);
  }
  CHECK(actual.band[failed_system] == kSentinel &&
        actual.occupation_sum[failed_system] == kSentinel);
  CHECK(actual.system_errors[0] == 0u);
  CHECK(near(actual.band[0], host.expected_band[0], 8.0e-13));
  return 0;
}

int test_spin_rounded_weighted_contribution_overflow_is_transactional() {
  SpinHostCase host = make_spin_case(8u);
  constexpr std::size_t failed_system = 1u;
  const double maximum = std::numeric_limits<double>::max();
  const std::int64_t orbital_begin = host.orbital_offsets[failed_system];
  const std::int64_t count = host.orbital_offsets[failed_system + 1u] - orbital_begin;
  const std::int64_t spin_orbital_begin = host.spin_orbital_offsets[failed_system];
  const std::int64_t spin_matrix_begin = host.spin_matrix_offsets[failed_system];
  CHECK(count == 2 && host.spin_channels[failed_system] == 2);

  std::fill(host.coefficients.begin() + spin_matrix_begin,
            host.coefficients.begin() + host.spin_matrix_offsets[failed_system + 1u], 0.0);
  const std::int64_t beta_matrix = spin_matrix_begin + count * count;
  /* Exercise the same rounded-product contract specifically in beta. */
  host.coefficients[static_cast<std::size_t>(beta_matrix)] = 1.0;
  host.coefficients[static_cast<std::size_t>(beta_matrix + 1)] = 1.25;
  for (std::int64_t local = 0; local < count; ++local) {
    host.occupations[static_cast<std::size_t>(2 * orbital_begin + local)] = 0.0;
    host.occupations[static_cast<std::size_t>(2 * orbital_begin + count + local)] = 1.0;
    host.eigenvalues[static_cast<std::size_t>(spin_orbital_begin + local)] = 0.0;
  }
  host.eigenvalues[static_cast<std::size_t>(spin_orbital_begin + count)] = -0.75 * maximum;
  host.eigenvalues[static_cast<std::size_t>(spin_orbital_begin + count + 1)] = 0.75 * maximum;

  SpinDeviceFixture device;
  CUDA_CHECK(device.initialize(host, nullptr));
  CHECK(run_spin(device, host, nullptr) == 0);
  SpinResults actual;
  CUDA_CHECK(copy_spin_results(host, device, actual, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(actual.system_errors[failed_system] ==
        static_cast<std::uint32_t>(Gfn2DensityDeviceError::kNonfiniteWeightedDensityArithmetic));
  CHECK(actual.device_error ==
        static_cast<std::uint32_t>(Gfn2DensityDeviceError::kNonfiniteWeightedDensityArithmetic));
  for (std::int64_t index = spin_matrix_begin; index < host.spin_matrix_offsets[failed_system + 1u];
       ++index) {
    CHECK(actual.density[static_cast<std::size_t>(index)] == kSentinel);
    CHECK(actual.weighted_density[static_cast<std::size_t>(index)] == kSentinel);
  }
  for (std::int64_t channel = host.spin_channel_offsets[failed_system];
       channel < host.spin_channel_offsets[failed_system + 1u]; ++channel) {
    CHECK(actual.channel_band[static_cast<std::size_t>(channel)] == kSentinel);
    CHECK(actual.channel_occupation_sum[static_cast<std::size_t>(channel)] == kSentinel);
    CHECK(actual.channel_density_trace[static_cast<std::size_t>(channel)] == kSentinel);
    CHECK(actual.channel_weighted_trace[static_cast<std::size_t>(channel)] == kSentinel);
  }
  CHECK(actual.band[failed_system] == kSentinel &&
        actual.occupation_sum[failed_system] == kSentinel);
  CHECK(actual.system_errors[0] == 0u && actual.band[0] != kSentinel);

  /* Preserve the beta channel's weighted-first diagnostic when density
   * arithmetic fails at a later orbital. */
  std::fill(host.coefficients.begin() + spin_matrix_begin,
            host.coefficients.begin() + host.spin_matrix_offsets[failed_system + 1u], 0.0);
  host.coefficients[static_cast<std::size_t>(beta_matrix)] = 2.0;
  host.coefficients[static_cast<std::size_t>(beta_matrix + 1)] = 2.0 * std::sqrt(maximum);
  host.eigenvalues[static_cast<std::size_t>(spin_orbital_begin + count)] = 0.75 * maximum;
  host.eigenvalues[static_cast<std::size_t>(spin_orbital_begin + count + 1)] = -0.75 * maximum;
  SpinDeviceFixture precedence_device;
  CUDA_CHECK(precedence_device.initialize(host, nullptr));
  CHECK(run_spin(precedence_device, host, nullptr) == 0);
  CUDA_CHECK(copy_spin_results(host, precedence_device, actual, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(actual.system_errors[failed_system] ==
        static_cast<std::uint32_t>(Gfn2DensityDeviceError::kNonfiniteWeightedDensityArithmetic));
  return 0;
}

int test_spin_graph_replay_changed_beta_input() {
  SpinHostCase host = make_spin_case(32u);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  SpinDeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CHECK(run_spin(device, host, stream) == 0);
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUDA_CHECK(device.reset_outputs(host, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  SpinResults actual;
  CUDA_CHECK(copy_spin_results(host, device, actual, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(compare_spin_success(host, actual) == 0);

  constexpr std::size_t changed_system = 1u;
  const std::int64_t orbital_begin = host.orbital_offsets[changed_system];
  const std::int64_t count = host.orbital_offsets[changed_system + 1u] - orbital_begin;
  host.occupations[static_cast<std::size_t>(2 * orbital_begin + count)] *= 0.25;
  build_spin_reference(host);
  CUDA_CHECK(
      device.occupations.copy_from(host.occupations.data(), host.occupations.size(), stream));
  CUDA_CHECK(device.reset_outputs(host, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(copy_spin_results(host, device, actual, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(compare_spin_success(host, actual) == 0);
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_spin_entry_requires_explicit_metadata() {
  HostCase host = make_case(1u);
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, nullptr));
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_spin_density_cuda(
            device.batch(host), Gfn2WavefunctionLayoutView{}, device.input(), device.results(),
            device.workspace(), device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
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
  if (const int line = test_rounded_weighted_contribution_overflow_is_transactional(); line != 0)
    return line;
  if (const int line = test_active_mask_sticky_error_and_graph_replay(); line != 0) return line;
  if (const int line = test_spin_entry_preserves_restricted_outputs_exactly(); line != 0)
    return line;
  if (const int line = test_spin_mixed_ragged_custom_stream_and_inactive_poison(); line != 0)
    return line;
  if (const int line = test_spin_beta_failure_is_transactional(); line != 0) return line;
  if (const int line = test_spin_rounded_weighted_contribution_overflow_is_transactional();
      line != 0)
    return line;
  if (const int line = test_spin_graph_replay_changed_beta_input(); line != 0) return line;
  if (const int line = test_spin_entry_requires_explicit_metadata(); line != 0) return line;
  if (const int line = test_host_argument_alias_token_and_alignment_validation(); line != 0)
    return line;
  return 0;
}
