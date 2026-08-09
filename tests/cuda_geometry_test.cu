#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_geometry.cuh"
#include "model/gfn2/coordination.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using xtbloom::detail::cuda::Gfn2GeometryDeviceBatch;
using xtbloom::detail::cuda::Gfn2GeometryDeviceCache;
using xtbloom::detail::cuda::Gfn2GeometryDeviceError;
using xtbloom::detail::cuda::Gfn2GeometryDeviceWorkspace;
using xtbloom::detail::cuda::kGfn2GeometryPairDataElements;
using xtbloom::detail::gfn2::CoordinationPlan;

constexpr std::uint64_t kPlanToken = 0xbb67ae8584caa73bULL;
constexpr std::uint64_t kGeneration = 73u;
constexpr double kPairSentinel = -991.25;
constexpr double kCoordinationSentinel = -881.5;

static_assert(std::is_trivially_copyable_v<Gfn2GeometryDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2GeometryDeviceCache>);
static_assert(std::is_trivially_copyable_v<Gfn2GeometryDeviceWorkspace>);

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <=
         tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

std::int64_t triangle_count(std::int64_t value) {
  return (value & 1LL) == 0LL ? (value / 2LL) * (value - 1LL) : value * ((value - 1LL) / 2LL);
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
    if (count == 0u) {
      return cudaSuccess;
    }
    return cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
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
  cudaError_t status = device.allocate(host.size());
  return status == cudaSuccess ? device.copy_from(host.data(), host.size(), stream) : status;
}

struct HostCase {
  CoordinationPlan plan;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> pair_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> expected_coordination;
  std::vector<double> expected_pair_data;
  std::vector<double> adjoints;
  std::vector<double> gradient_seed;
  std::vector<double> expected_gradients;

  std::size_t batch_size() const { return atom_offsets.size() - 1u; }
  std::size_t total_atoms() const { return atomic_numbers.size(); }
  std::size_t total_pairs() const { return static_cast<std::size_t>(pair_offsets.back()); }
};

/* Independent pair-layout reference; CN and VJP parity use the CPU library. */
void build_expected_pairs(HostCase& host) {
  host.expected_pair_data.assign(
      host.total_pairs() * static_cast<std::size_t>(kGfn2GeometryPairDataElements), 0.0);
  for (std::size_t system = 0u; system < host.batch_size(); ++system) {
    const std::int64_t begin = host.atom_offsets[system];
    const std::int64_t end = host.atom_offsets[system + 1u];
    for (std::int64_t upper = begin + 1; upper < end; ++upper) {
      for (std::int64_t lower = begin; lower < upper; ++lower) {
        const std::int64_t pair =
            host.pair_offsets[system] + triangle_count(upper - begin) + (lower - begin);
        double* const data = host.expected_pair_data.data() +
                             static_cast<std::size_t>(pair * kGfn2GeometryPairDataElements);
        data[0] = host.positions[static_cast<std::size_t>(upper * 3)] -
                  host.positions[static_cast<std::size_t>(lower * 3)];
        data[1] = host.positions[static_cast<std::size_t>(upper * 3 + 1)] -
                  host.positions[static_cast<std::size_t>(lower * 3 + 1)];
        data[2] = host.positions[static_cast<std::size_t>(upper * 3 + 2)] -
                  host.positions[static_cast<std::size_t>(lower * 3 + 2)];
        data[3] = std::sqrt(data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
        data[4] = 1.0 / data[3];
      }
    }
  }
}

bool make_case(std::size_t batch_size, HostCase& host, std::string& error) {
  host.atom_offsets.assign(batch_size + 1u, 0);
  host.pair_offsets.assign(batch_size + 1u, 0);
  std::int64_t atoms = 0;
  std::int64_t pairs = 0;
  for (std::size_t system = 0u; system < batch_size; ++system) {
    host.atom_offsets[system] = atoms;
    host.pair_offsets[system] = pairs;
    const std::int64_t count = batch_size == 1u ? 4 : static_cast<std::int64_t>(system % 5u);
    atoms += count;
    pairs += triangle_count(count);
  }
  host.atom_offsets[batch_size] = atoms;
  host.pair_offsets[batch_size] = pairs;
  host.atomic_numbers.resize(static_cast<std::size_t>(atoms));
  host.positions.resize(static_cast<std::size_t>(atoms * 3));
  constexpr std::int32_t elements[] = {1, 6, 7, 8, 16, 17, 35, 53};
  for (std::size_t system = 0u; system < batch_size; ++system) {
    const std::int64_t begin = host.atom_offsets[system];
    const std::int64_t end = host.atom_offsets[system + 1u];
    for (std::int64_t atom = begin; atom < end; ++atom) {
      const std::int64_t local = atom - begin;
      host.atomic_numbers[static_cast<std::size_t>(atom)] =
          elements[(system + static_cast<std::size_t>(local)) %
                   (sizeof(elements) / sizeof(elements[0]))];
      host.positions[static_cast<std::size_t>(atom * 3)] =
          1.31 * static_cast<double>(local) + 0.007 * static_cast<double>(system);
      host.positions[static_cast<std::size_t>(atom * 3 + 1)] =
          0.23 * static_cast<double>(local * local + 1);
      host.positions[static_cast<std::size_t>(atom * 3 + 2)] =
          (local % 2 == 0 ? -0.17 : 0.19) * static_cast<double>(local + 1);
    }
    /* Exercise cached geometry outside the 25 bohr CN cutoff. */
    if (batch_size > 1u && system % 7u == 0u && end - begin >= 2) {
      host.positions[static_cast<std::size_t>((end - 1) * 3)] += 30.0;
    }
  }
  if (xtbloom::detail::gfn2::make_coordination_plan(
          static_cast<std::int64_t>(batch_size), atoms, host.atom_offsets.data(),
          host.atomic_numbers.data(), host.plan, error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  host.expected_coordination.resize(static_cast<std::size_t>(atoms));
  if (xtbloom::detail::gfn2::evaluate_coordination_cpu(host.plan, host.positions.data(),
                                                       host.expected_coordination.data(),
                                                       error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  host.adjoints.resize(static_cast<std::size_t>(atoms));
  host.gradient_seed.resize(static_cast<std::size_t>(atoms * 3));
  for (std::int64_t atom = 0; atom < atoms; ++atom) {
    host.adjoints[static_cast<std::size_t>(atom)] =
        0.013 * static_cast<double>(static_cast<int>(atom % 9) - 4);
    for (int axis = 0; axis < 3; ++axis) {
      host.gradient_seed[static_cast<std::size_t>(atom * 3 + axis)] =
          0.001 * static_cast<double>(static_cast<int>((atom * 3 + axis) % 7) - 3);
    }
  }
  host.expected_gradients = host.gradient_seed;
  if (xtbloom::detail::gfn2::add_coordination_gradient_cpu(
          host.plan, host.positions.data(), host.adjoints.data(), host.expected_gradients.data(),
          error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  build_expected_pairs(host);
  return true;
}

struct DeviceFixture {
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> pair_offsets;
  DeviceBuffer<double> radii;
  DeviceBuffer<double> positions;
  DeviceBuffer<double> pair_data;
  DeviceBuffer<double> coordination;
  DeviceBuffer<std::uint64_t> generations;
  DeviceBuffer<double> pair_scratch;
  DeviceBuffer<double> coordination_scratch;
  DeviceBuffer<double> gradient_scratch;
  DeviceBuffer<std::uint32_t> sequence_active;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;
  DeviceBuffer<double> adjoints;
  DeviceBuffer<double> gradients;

  cudaError_t initialize(const HostCase& host, cudaStream_t stream) {
    const std::size_t pair_elements =
        host.total_pairs() * static_cast<std::size_t>(kGfn2GeometryPairDataElements);
    const std::vector<double> pair_seed(pair_elements, kPairSentinel);
    const std::vector<double> coordination_seed(host.total_atoms(), kCoordinationSentinel);
    const std::vector<std::uint64_t> generation_seed(host.batch_size(), 19u);
    cudaError_t status = allocate_and_copy(atom_offsets, host.atom_offsets, stream);
    if (status == cudaSuccess) {
      status = allocate_and_copy(pair_offsets, host.pair_offsets, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(radii, host.plan.covalent_radius, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(positions, host.positions, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(pair_data, pair_seed, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(coordination, coordination_seed, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(generations, generation_seed, stream);
    }
    if (status == cudaSuccess) {
      status = pair_scratch.allocate(pair_elements);
    }
    if (status == cudaSuccess) {
      status = coordination_scratch.allocate(host.total_atoms());
    }
    if (status == cudaSuccess) {
      status = gradient_scratch.allocate(host.positions.size());
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
    if (status == cudaSuccess) {
      status = allocate_and_copy(adjoints, host.adjoints, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(gradients, host.gradient_seed, stream);
    }
    return status;
  }

  Gfn2GeometryDeviceBatch batch(const HostCase& host) const {
    return Gfn2GeometryDeviceBatch{
        static_cast<std::int64_t>(host.batch_size()),
        static_cast<std::int64_t>(host.total_atoms()),
        static_cast<std::int64_t>(host.total_pairs()),
        static_cast<std::int64_t>(host.atom_offsets.size()),
        static_cast<std::int64_t>(host.pair_offsets.size()),
        static_cast<std::int64_t>(host.plan.covalent_radius.size()),
        static_cast<std::int64_t>(host.positions.size()),
        kPlanToken,
        atom_offsets.get(),
        pair_offsets.get(),
        radii.get(),
    };
  }

  Gfn2GeometryDeviceCache cache(const HostCase& host) {
    return Gfn2GeometryDeviceCache{
        pair_data.get(),    static_cast<std::int64_t>(pair_data.size()),
        coordination.get(), static_cast<std::int64_t>(coordination.size()),
        generations.get(),  static_cast<std::int64_t>(host.batch_size()),
        kPlanToken,
    };
  }

  Gfn2GeometryDeviceWorkspace workspace() {
    return Gfn2GeometryDeviceWorkspace{
        pair_scratch.get(),
        static_cast<std::int64_t>(pair_scratch.size()),
        coordination_scratch.get(),
        static_cast<std::int64_t>(coordination_scratch.size()),
        gradient_scratch.get(),
        static_cast<std::int64_t>(gradient_scratch.size()),
        sequence_active.get(),
        1,
        kPlanToken,
    };
  }

  cudaError_t reset_outputs(const HostCase& host, cudaStream_t stream) {
    const std::vector<double> pair_seed(pair_data.size(), kPairSentinel);
    const std::vector<double> coordination_seed(host.total_atoms(), kCoordinationSentinel);
    const std::vector<std::uint64_t> generation_seed(host.batch_size(), 19u);
    cudaError_t status = pair_data.copy_from(pair_seed.data(), pair_seed.size(), stream);
    if (status == cudaSuccess) {
      status = coordination.copy_from(coordination_seed.data(), coordination_seed.size(), stream);
    }
    if (status == cudaSuccess) {
      status = generations.copy_from(generation_seed.data(), generation_seed.size(), stream);
    }
    return status == cudaSuccess
               ? gradients.copy_from(host.gradient_seed.data(), host.gradient_seed.size(), stream)
               : status;
  }
};

struct Results {
  std::vector<double> pair_data;
  std::vector<double> coordination;
  std::vector<std::uint64_t> generations;
  std::vector<double> gradients;
  std::vector<std::uint32_t> system_errors;
  std::uint32_t device_error = 99u;
};

cudaError_t copy_results(const HostCase& host, const DeviceFixture& device, Results& results,
                         cudaStream_t stream) {
  results.pair_data.resize(device.pair_data.size());
  results.coordination.resize(host.total_atoms());
  results.generations.resize(host.batch_size());
  results.gradients.resize(host.positions.size());
  results.system_errors.resize(host.batch_size());
  cudaError_t status =
      device.pair_data.copy_to(results.pair_data.data(), results.pair_data.size(), stream);
  if (status == cudaSuccess) {
    status = device.coordination.copy_to(results.coordination.data(), results.coordination.size(),
                                         stream);
  }
  if (status == cudaSuccess) {
    status =
        device.generations.copy_to(results.generations.data(), results.generations.size(), stream);
  }
  if (status == cudaSuccess) {
    status = device.gradients.copy_to(results.gradients.data(), results.gradients.size(), stream);
  }
  if (status == cudaSuccess) {
    status = device.system_errors.copy_to(results.system_errors.data(),
                                          results.system_errors.size(), stream);
  }
  return status == cudaSuccess ? device.device_error.copy_to(&results.device_error, 1u, stream)
                               : status;
}

int compare_success(const HostCase& host, const Results& results) {
  CHECK(results.device_error == static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess));
  CHECK(std::all_of(results.system_errors.begin(), results.system_errors.end(),
                    [](std::uint32_t value) { return value == 0u; }));
  CHECK(std::all_of(results.generations.begin(), results.generations.end(),
                    [](std::uint64_t value) { return value == kGeneration; }));
  for (std::size_t atom = 0u; atom < host.total_atoms(); ++atom) {
    CHECK(near(results.coordination[atom], host.expected_coordination[atom], 5.0e-13));
  }
  for (std::size_t pair = 0u; pair < host.total_pairs(); ++pair) {
    const std::size_t base = pair * static_cast<std::size_t>(kGfn2GeometryPairDataElements);
    CHECK(results.pair_data[base] == host.expected_pair_data[base]);
    CHECK(results.pair_data[base + 1u] == host.expected_pair_data[base + 1u]);
    CHECK(results.pair_data[base + 2u] == host.expected_pair_data[base + 2u]);
    CHECK(near(results.pair_data[base + 3u], host.expected_pair_data[base + 3u], 3.0e-15));
    CHECK(near(results.pair_data[base + 4u], host.expected_pair_data[base + 4u], 3.0e-15));
    CHECK(results.pair_data[base + 5u] >= 0.0);
    CHECK(std::isfinite(results.pair_data[base + 6u]));
  }
  for (std::size_t coordinate = 0u; coordinate < host.expected_gradients.size(); ++coordinate) {
    CHECK(near(results.gradients[coordinate], host.expected_gradients[coordinate], 8.0e-12));
  }
  return 0;
}

int test_cpu_parity_ragged_batches_and_custom_stream() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (const std::size_t batch_size : {1u, 8u, 32u, 128u}) {
    HostCase host;
    std::string error;
    CHECK(make_case(batch_size, host, error));
    DeviceFixture device;
    CUDA_CHECK(device.initialize(host, stream));
    CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_geometry_device_errors_cuda(
        static_cast<std::int64_t>(batch_size), device.system_errors.get(),
        device.device_error.get(), stream));
    CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_geometry_cache_cuda(
        device.batch(host), device.positions.get(), kGeneration, device.cache(host),
        device.workspace(), device.system_errors.get(), device.device_error.get(), stream));
    CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_coordination_vjp_cuda(
        device.batch(host), device.cache(host), kGeneration, device.adjoints.get(),
        device.gradients.get(), device.workspace(), device.system_errors.get(),
        device.device_error.get(), stream));
    Results results;
    CUDA_CHECK(copy_results(host, device, results, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int comparison = compare_success(host, results);
    CHECK(comparison == 0);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_numerical_failure_and_stale_vjp_are_peer_isolated() {
  HostCase host;
  std::string error;
  CHECK(make_case(8u, host, error));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, nullptr));

  /* Poison system 3 only; healthy cache slices and generations must commit. */
  constexpr std::size_t failed_system = 3u;
  const std::size_t failed_atom = static_cast<std::size_t>(host.atom_offsets[failed_system]);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(cudaMemcpy(device.positions.get() + failed_atom * 3u, &nan, sizeof(nan),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_geometry_device_errors_cuda(
      8, device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_geometry_cache_cuda(
      device.batch(host), device.positions.get(), kGeneration, device.cache(host),
      device.workspace(), device.system_errors.get(), device.device_error.get()));
  Results results;
  CUDA_CHECK(copy_results(host, device, results, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(results.device_error ==
        static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kNonfinitePosition));
  CHECK(results.system_errors[failed_system] ==
        static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kNonfinitePosition));
  for (std::size_t system = 0u; system < host.batch_size(); ++system) {
    CHECK(results.generations[system] == (system == failed_system ? 19u : kGeneration));
  }
  for (std::int64_t atom = host.atom_offsets[failed_system];
       atom < host.atom_offsets[failed_system + 1u]; ++atom) {
    CHECK(results.coordination[static_cast<std::size_t>(atom)] == kCoordinationSentinel);
  }

  /* Restore geometry, then make one generation stale for the VJP only. */
  CUDA_CHECK(device.positions.copy_from(host.positions.data(), host.positions.size()));
  CUDA_CHECK(device.reset_outputs(host, nullptr));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_geometry_device_errors_cuda(
      8, device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_geometry_cache_cuda(
      device.batch(host), device.positions.get(), kGeneration, device.cache(host),
      device.workspace(), device.system_errors.get(), device.device_error.get()));
  constexpr std::size_t stale_system = 4u;
  const std::uint64_t stale_generation = kGeneration - 1u;
  CUDA_CHECK(cudaMemcpy(device.generations.get() + stale_system, &stale_generation,
                        sizeof(stale_generation), cudaMemcpyHostToDevice));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_geometry_device_errors_cuda(
      8, device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_coordination_vjp_cuda(
      device.batch(host), device.cache(host), kGeneration, device.adjoints.get(),
      device.gradients.get(), device.workspace(), device.system_errors.get(),
      device.device_error.get()));
  CUDA_CHECK(copy_results(host, device, results, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(results.device_error ==
        static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kStaleGeometry));
  CHECK(results.system_errors[stale_system] ==
        static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kStaleGeometry));
  for (std::size_t system = 0u; system < host.batch_size(); ++system) {
    const std::int64_t begin = host.atom_offsets[system];
    const std::int64_t end = host.atom_offsets[system + 1u];
    for (std::int64_t atom = begin; atom < end; ++atom) {
      for (int axis = 0; axis < 3; ++axis) {
        const std::size_t coordinate = static_cast<std::size_t>(atom * 3 + axis);
        if (system == stale_system) {
          CHECK(results.gradients[coordinate] == host.gradient_seed[coordinate]);
        } else {
          CHECK(near(results.gradients[coordinate], host.expected_gradients[coordinate], 8.0e-12));
        }
      }
    }
  }
  return 0;
}

int test_extreme_device_offsets_and_sticky_error_fail_closed() {
  HostCase host;
  std::string error;
  CHECK(make_case(8u, host, error));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, nullptr));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_geometry_device_errors_cuda(
      8, device.system_errors.get(), device.device_error.get()));

  const std::int64_t atom_extreme = std::numeric_limits<std::int64_t>::min();
  const std::int64_t pair_extreme = std::numeric_limits<std::int64_t>::max();
  CUDA_CHECK(cudaMemcpy(device.atom_offsets.get() + 2, &atom_extreme, sizeof(atom_extreme),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device.pair_offsets.get() + 5, &pair_extreme, sizeof(pair_extreme),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_geometry_cache_cuda(
      device.batch(host), device.positions.get(), kGeneration, device.cache(host),
      device.workspace(), device.system_errors.get(), device.device_error.get()));
  Results results;
  CUDA_CHECK(copy_results(host, device, results, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(results.device_error ==
        static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kInvalidOffsets));
  CHECK(std::all_of(results.pair_data.begin(), results.pair_data.end(),
                    [](double value) { return value == kPairSentinel; }));
  CHECK(std::all_of(results.coordination.begin(), results.coordination.end(),
                    [](double value) { return value == kCoordinationSentinel; }));
  CHECK(std::all_of(results.generations.begin(), results.generations.end(),
                    [](std::uint64_t value) { return value == 19u; }));

  CUDA_CHECK(device.atom_offsets.copy_from(host.atom_offsets.data(), host.atom_offsets.size()));
  CUDA_CHECK(device.pair_offsets.copy_from(host.pair_offsets.data(), host.pair_offsets.size()));
  CUDA_CHECK(device.reset_outputs(host, nullptr));
  const std::uint32_t sticky =
      static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kNonfiniteVjpArithmetic);
  CUDA_CHECK(cudaMemset(device.system_errors.get(), 0, host.batch_size() * sizeof(std::uint32_t)));
  CUDA_CHECK(
      cudaMemcpy(device.device_error.get(), &sticky, sizeof(sticky), cudaMemcpyHostToDevice));
  CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_geometry_cache_cuda(
      device.batch(host), device.positions.get(), kGeneration, device.cache(host),
      device.workspace(), device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(copy_results(host, device, results, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(results.device_error == sticky);
  CHECK(std::all_of(results.pair_data.begin(), results.pair_data.end(),
                    [](double value) { return value == kPairSentinel; }));
  CHECK(std::all_of(results.coordination.begin(), results.coordination.end(),
                    [](double value) { return value == kCoordinationSentinel; }));
  CHECK(std::all_of(results.generations.begin(), results.generations.end(),
                    [](std::uint64_t value) { return value == 19u; }));
  return 0;
}

int test_cuda_graph_capture_and_replay() {
  HostCase host;
  std::string error;
  CHECK(make_case(32u, host, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_geometry_device_errors_cuda(
      32, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::update_gfn2_geometry_cache_cuda(
      device.batch(host), device.positions.get(), kGeneration, device.cache(host),
      device.workspace(), device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_coordination_vjp_cuda(
      device.batch(host), device.cache(host), kGeneration, device.adjoints.get(),
      device.gradients.get(), device.workspace(), device.system_errors.get(),
      device.device_error.get(), stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));

  for (int replay = 0; replay < 2; ++replay) {
    CUDA_CHECK(device.reset_outputs(host, stream));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    Results results;
    CUDA_CHECK(copy_results(host, device, results, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int comparison = compare_success(host, results);
    CHECK(comparison == 0);
  }
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_host_argument_and_alias_validation() {
  HostCase host;
  std::string error;
  CHECK(make_case(8u, host, error));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, nullptr));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_geometry_device_errors_cuda(
      8, device.system_errors.get(), device.device_error.get()));
  Gfn2GeometryDeviceBatch batch = device.batch(host);
  Gfn2GeometryDeviceCache cache = device.cache(host);
  Gfn2GeometryDeviceWorkspace workspace = device.workspace();
  batch.coordinate_elements -= 1;
  CHECK(xtbloom::detail::cuda::update_gfn2_geometry_cache_cuda(
            batch, device.positions.get(), kGeneration, cache, workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  batch = device.batch(host);
  workspace.plan_token ^= 1u;
  CHECK(xtbloom::detail::cuda::update_gfn2_geometry_cache_cuda(
            batch, device.positions.get(), kGeneration, cache, workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  workspace = device.workspace();
  workspace.pair_scratch = cache.pair_data;
  CHECK(xtbloom::detail::cuda::update_gfn2_geometry_cache_cuda(
            batch, device.positions.get(), kGeneration, cache, workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);

  /* Host validation must reject hostile device views before enqueueing work. */
  workspace = device.workspace();
  batch.atom_offsets = reinterpret_cast<const std::int64_t*>(
      reinterpret_cast<const unsigned char*>(device.atom_offsets.get()) + 1u);
  CHECK(xtbloom::detail::cuda::update_gfn2_geometry_cache_cuda(
            batch, device.positions.get(), kGeneration, cache, workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  batch = device.batch(host);
  batch.pair_offsets = reinterpret_cast<const std::int64_t*>(
      reinterpret_cast<const unsigned char*>(device.pair_offsets.get()) + 1u);
  CHECK(xtbloom::detail::cuda::update_gfn2_geometry_cache_cuda(
            batch, device.positions.get(), kGeneration, cache, workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);

  Results results;
  CUDA_CHECK(copy_results(host, device, results, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(std::all_of(results.pair_data.begin(), results.pair_data.end(),
                    [](double value) { return value == kPairSentinel; }));
  CHECK(std::all_of(results.coordination.begin(), results.coordination.end(),
                    [](double value) { return value == kCoordinationSentinel; }));
  CHECK(std::all_of(results.generations.begin(), results.generations.end(),
                    [](std::uint64_t value) { return value == 19u; }));

  CHECK(xtbloom::detail::cuda::reset_gfn2_geometry_device_errors_cuda(
            8, device.system_errors.get(), device.system_errors.get()) == cudaErrorInvalidValue);
  return 0;
}

}  // namespace

int main() {
  if (const int line = test_cpu_parity_ragged_batches_and_custom_stream(); line != 0) {
    return line;
  }
  if (const int line = test_numerical_failure_and_stale_vjp_are_peer_isolated(); line != 0) {
    return line;
  }
  if (const int line = test_extreme_device_offsets_and_sticky_error_fail_closed(); line != 0) {
    return line;
  }
  if (const int line = test_cuda_graph_capture_and_replay(); line != 0) {
    return line;
  }
  if (const int line = test_host_argument_and_alias_validation(); line != 0) {
    return line;
  }
  return 0;
}
