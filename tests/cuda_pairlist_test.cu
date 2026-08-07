#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_geometry.cuh"
#include "backends/cuda/gfn2_pairlist.cuh"
#include "model/gfn2/coordination.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using gpuxtb::detail::Gfn2AtomPair;
using gpuxtb::detail::cuda::Gfn2PairListDeviceBatch;
using gpuxtb::detail::cuda::Gfn2PairListDeviceCache;
using gpuxtb::detail::cuda::Gfn2PairListDeviceError;
using gpuxtb::detail::cuda::Gfn2PairListDeviceWorkspace;
using gpuxtb::detail::cuda::Gfn2PairListMode;
using gpuxtb::detail::cuda::Gfn2PairListSystemMeta;
using gpuxtb::detail::cuda::kGfn2GeometryPairDataElements;
using gpuxtb::detail::gfn2::CoordinationPlan;

constexpr std::uint64_t kPlanToken = 0x4a51a3e9f1d8c2b7ULL;
constexpr std::uint64_t kGeneration = 101u;
constexpr std::int64_t kMaxCells = 65536;
constexpr std::int64_t kMaxNeighbors = 4096;
constexpr std::int64_t kMaxPairs = 16384;
constexpr double kPairSentinel = -997.25;
constexpr std::int64_t kNeighborSentinel = -1;

static_assert(std::is_trivially_copyable_v<Gfn2PairListDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2PairListDeviceCache>);
static_assert(std::is_trivially_copyable_v<Gfn2PairListDeviceWorkspace>);

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
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> expected_coordination;

  std::size_t batch_size() const { return atom_offsets.size() - 1u; }
  std::size_t total_atoms() const { return atomic_numbers.size(); }
};

/*
 * Make a ragged batch.  Dense (cluster) systems put atoms within 6 bohr so
 * nearly every pair is retained; sparse systems spread atoms 30 bohr apart so
 * no pairs are retained beyond the closest neighbors.  Empty and single-atom
 * systems are inserted too.
 */
bool make_case(std::size_t batch_size, bool dense_geometry, HostCase& host, std::string& error) {
  host.atom_offsets.assign(batch_size + 1u, 0);
  std::int64_t atoms = 0;
  for (std::size_t system = 0u; system < batch_size; ++system) {
    host.atom_offsets[system] = atoms;
    const std::int64_t count = static_cast<std::int64_t>(system % 7u) + 1;
    atoms += count;
  }
  host.atom_offsets[batch_size] = atoms;
  host.atomic_numbers.resize(static_cast<std::size_t>(atoms));
  host.positions.resize(static_cast<std::size_t>(atoms * 3));
  constexpr std::int32_t elements[] = {1, 6, 7, 8, 16, 17, 35, 53};
  const double spacing = dense_geometry ? 1.7 : 30.0;
  for (std::size_t system = 0u; system < batch_size; ++system) {
    const std::int64_t begin = host.atom_offsets[system];
    const std::int64_t end = host.atom_offsets[system + 1u];
    for (std::int64_t atom = begin; atom < end; ++atom) {
      const std::int64_t local = atom - begin;
      host.atomic_numbers[static_cast<std::size_t>(atom)] =
          elements[(system + static_cast<std::size_t>(local)) %
                   (sizeof(elements) / sizeof(elements[0]))];
      host.positions[static_cast<std::size_t>(atom * 3)] =
          spacing * static_cast<double>(local) + 0.04 * static_cast<double>(system);
      host.positions[static_cast<std::size_t>(atom * 3 + 1)] =
          0.2 * static_cast<double>(local * local + 1) + 0.01 * static_cast<double>(system);
      host.positions[static_cast<std::size_t>(atom * 3 + 2)] =
          (local % 2 == 0 ? -0.11 : 0.13) * static_cast<double>(local + 1);
    }
  }
  if (gpuxtb::detail::gfn2::make_coordination_plan(
          static_cast<std::int64_t>(batch_size), atoms, host.atom_offsets.data(),
          host.atomic_numbers.data(), host.plan, error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  host.expected_coordination.resize(static_cast<std::size_t>(atoms));
  if (gpuxtb::detail::gfn2::evaluate_coordination_cpu(host.plan, host.positions.data(),
                                                      host.expected_coordination.data(),
                                                      error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  return true;
}

struct DeviceFixture {
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<double> positions;
  DeviceBuffer<double> radii;
  DeviceBuffer<double> coordination;
  DeviceBuffer<Gfn2AtomPair> pairs;
  DeviceBuffer<std::int64_t> pair_offsets;
  DeviceBuffer<std::int64_t> pair_counts;
  DeviceBuffer<std::int64_t> neighbor_offsets;
  DeviceBuffer<std::int64_t> neighbor_counts;
  DeviceBuffer<std::int64_t> neighbors;
  DeviceBuffer<std::uint64_t> pair_generations;
  DeviceBuffer<gpuxtb::detail::cuda::Gfn2PairListSystemMeta> system_meta;
  DeviceBuffer<std::int64_t> atom_cells;
  DeviceBuffer<std::int64_t> cell_counts;
  DeviceBuffer<std::int64_t> cell_offsets;
  DeviceBuffer<std::int64_t> cell_fill;
  DeviceBuffer<std::int64_t> cell_atoms;
  DeviceBuffer<std::int64_t> neighbor_cursor;
  DeviceBuffer<std::int64_t> neighbor_scratch;
  DeviceBuffer<std::int64_t> pair_cursor;
  DeviceBuffer<std::uint32_t> sequence_active;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;

  cudaError_t initialize(const HostCase& host, Gfn2PairListMode mode, cudaStream_t stream) {
    std::int64_t cache_pairs = 0;
    std::int64_t cache_neighbor_offsets = 0;
    std::int64_t cache_neighbors = 0;
    std::int64_t cache_pair_offsets = 0;
    std::int64_t cache_pair_counts = 0;
    std::int64_t cache_neighbor_counts = 0;
    std::int64_t cache_generations = 0;
    std::int64_t ws_meta = 0;
    std::int64_t ws_atom_cells = 0;
    std::int64_t ws_cell_arrays = 0;
    std::int64_t ws_cell_atoms = 0;
    std::int64_t ws_neighbor_cursor = 0;
    std::int64_t ws_neighbor_scratch = 0;
    std::int64_t ws_pair_cursor = 0;
    if (!gpuxtb::detail::cuda::query_gfn2_pairlist_requirements_cuda(
            static_cast<std::int64_t>(host.batch_size()),
            static_cast<std::int64_t>(host.total_atoms()), kMaxCells, kMaxNeighbors, kMaxPairs,
            &cache_pairs, &cache_neighbor_offsets, &cache_neighbors, &cache_pair_offsets,
            &cache_generations, &ws_meta, &ws_atom_cells, &ws_cell_arrays, &ws_cell_atoms,
            &ws_neighbor_cursor, &ws_neighbor_scratch, &ws_pair_cursor, &cache_pair_counts,
            &cache_neighbor_counts)) {
      return cudaErrorInvalidValue;
    }
    const std::vector<Gfn2AtomPair> pair_seed(static_cast<std::size_t>(cache_pairs),
                                              Gfn2AtomPair{0, 0});
    const std::vector<std::int64_t> offsets_seed(static_cast<std::size_t>(cache_pair_offsets), 0);
    const std::vector<std::int64_t> count_seed(static_cast<std::size_t>(cache_pair_counts), 0);
    const std::vector<std::int64_t> neighbor_offsets_seed(
        static_cast<std::size_t>(cache_neighbor_offsets), 0);
    const std::vector<std::int64_t> neighbor_count_seed(
        static_cast<std::size_t>(cache_neighbor_counts), 0);
    const std::vector<std::int64_t> neighbor_seed(static_cast<std::size_t>(cache_neighbors),
                                                  kNeighborSentinel);
    const std::vector<double> coordination_seed(host.total_atoms(), kPairSentinel);
    const std::vector<std::uint64_t> generation_seed(host.batch_size(), 0u);
    const std::vector<std::int64_t> cursor_seed(host.total_atoms(), 0);
    const std::vector<std::int64_t> scratch_seed(static_cast<std::size_t>(ws_neighbor_scratch), 0);
    cudaError_t status = allocate_and_copy(atom_offsets, host.atom_offsets, stream);
    if (status == cudaSuccess) {
      status = allocate_and_copy(positions, host.positions, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(radii, host.plan.covalent_radius, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(coordination, coordination_seed, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(pairs, pair_seed, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(pair_offsets, offsets_seed, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(pair_counts, count_seed, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(neighbor_offsets, neighbor_offsets_seed, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(neighbor_counts, neighbor_count_seed, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(neighbors, neighbor_seed, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(pair_generations, generation_seed, stream);
    }
    if (status == cudaSuccess) {
      status = system_meta.allocate(static_cast<std::size_t>(ws_meta));
    }
    if (status == cudaSuccess) {
      status = atom_cells.allocate(static_cast<std::size_t>(ws_atom_cells));
    }
    if (status == cudaSuccess) {
      status = cell_counts.allocate(static_cast<std::size_t>(ws_cell_arrays));
    }
    if (status == cudaSuccess) {
      status = cell_offsets.allocate(static_cast<std::size_t>(ws_cell_arrays));
    }
    if (status == cudaSuccess) {
      status = cell_fill.allocate(static_cast<std::size_t>(ws_cell_arrays));
    }
    if (status == cudaSuccess) {
      status = cell_atoms.allocate(static_cast<std::size_t>(ws_cell_atoms));
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(neighbor_cursor, cursor_seed, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(neighbor_scratch, scratch_seed, stream);
    }
    if (status == cudaSuccess) {
      status = pair_cursor.allocate(static_cast<std::size_t>(ws_pair_cursor));
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
    return status;
  }

  Gfn2PairListDeviceBatch batch(const HostCase& host, Gfn2PairListMode mode) const {
    return Gfn2PairListDeviceBatch{
        static_cast<std::int64_t>(host.batch_size()),
        static_cast<std::int64_t>(host.total_atoms()),
        static_cast<std::int64_t>(host.atom_offsets.size()),
        25.0,
        kMaxCells,
        kMaxNeighbors,
        kMaxPairs,
        mode,
        kPlanToken,
        atom_offsets.get(),
    };
  }

  Gfn2PairListDeviceCache cache() {
    return Gfn2PairListDeviceCache{
        pairs.get(),
        static_cast<std::int64_t>(pairs.size()),
        pair_offsets.get(),
        static_cast<std::int64_t>(pair_offsets.size()),
        pair_counts.get(),
        static_cast<std::int64_t>(pair_counts.size()),
        neighbor_offsets.get(),
        static_cast<std::int64_t>(neighbor_offsets.size()),
        neighbor_counts.get(),
        static_cast<std::int64_t>(neighbor_counts.size()),
        neighbors.get(),
        static_cast<std::int64_t>(neighbors.size()),
        pair_generations.get(),
        static_cast<std::int64_t>(pair_generations.size()),
        kPlanToken,
    };
  }

  Gfn2PairListDeviceWorkspace workspace() {
    return Gfn2PairListDeviceWorkspace{
        system_meta.get(),
        static_cast<std::int64_t>(system_meta.size()),
        atom_cells.get(),
        static_cast<std::int64_t>(atom_cells.size()),
        cell_counts.get(),
        static_cast<std::int64_t>(cell_counts.size()),
        cell_offsets.get(),
        static_cast<std::int64_t>(cell_offsets.size()),
        cell_fill.get(),
        static_cast<std::int64_t>(cell_fill.size()),
        cell_atoms.get(),
        static_cast<std::int64_t>(cell_atoms.size()),
        neighbor_cursor.get(),
        static_cast<std::int64_t>(neighbor_cursor.size()),
        neighbor_scratch.get(),
        static_cast<std::int64_t>(neighbor_scratch.size()),
        pair_cursor.get(),
        static_cast<std::int64_t>(pair_cursor.size()),
        sequence_active.get(),
        1,
        kPlanToken,
    };
  }
};

struct Results {
  std::vector<Gfn2AtomPair> pairs;
  std::vector<std::int64_t> pair_offsets;
  std::vector<std::int64_t> neighbor_offsets;
  std::vector<std::int64_t> neighbors;
  std::vector<std::uint64_t> pair_generations;
  std::vector<double> coordination;
  std::vector<std::uint32_t> system_errors;
  std::uint32_t device_error = 99u;
};

cudaError_t copy_results(const HostCase& host, const DeviceFixture& device, Results& results,
                         cudaStream_t stream) {
  results.pairs.resize(device.pairs.size());
  results.pair_offsets.resize(device.pair_offsets.size());
  results.neighbor_offsets.resize(device.neighbor_offsets.size());
  results.neighbors.resize(device.neighbors.size());
  results.pair_generations.resize(device.pair_generations.size());
  results.coordination.resize(host.total_atoms());
  results.system_errors.resize(host.batch_size());
  cudaError_t status = device.pairs.copy_to(results.pairs.data(), results.pairs.size(), stream);
  if (status == cudaSuccess) {
    status = device.pair_offsets.copy_to(results.pair_offsets.data(), results.pair_offsets.size(),
                                         stream);
  }
  if (status == cudaSuccess) {
    status = device.neighbor_offsets.copy_to(results.neighbor_offsets.data(),
                                             results.neighbor_offsets.size(), stream);
  }
  if (status == cudaSuccess) {
    status = device.neighbors.copy_to(results.neighbors.data(), results.neighbors.size(), stream);
  }
  if (status == cudaSuccess) {
    status = device.pair_generations.copy_to(results.pair_generations.data(),
                                             results.pair_generations.size(), stream);
  }
  if (status == cudaSuccess) {
    status = device.coordination.copy_to(results.coordination.data(), results.coordination.size(),
                                         stream);
  }
  if (status == cudaSuccess) {
    status = device.system_errors.copy_to(results.system_errors.data(),
                                          results.system_errors.size(), stream);
  }
  return status == cudaSuccess ? device.device_error.copy_to(&results.device_error, 1u, stream)
                               : status;
}

/* Independent host-side reference for a retained pair subset. */
std::vector<std::int64_t> expected_retained_pairs(const HostCase& host, double cutoff) {
  constexpr double kMinimumDistanceSquared = 1.0e-12;
  const double cutoff_squared = cutoff * cutoff;
  std::vector<std::int64_t> pairs;
  for (std::size_t system = 0u; system < host.batch_size(); ++system) {
    const std::int64_t begin = host.atom_offsets[system];
    const std::int64_t end = host.atom_offsets[system + 1u];
    for (std::int64_t second = begin + 1; second < end; ++second) {
      for (std::int64_t first = begin; first < second; ++first) {
        const double dx = host.positions[static_cast<std::size_t>(second * 3)] -
                          host.positions[static_cast<std::size_t>(first * 3)];
        const double dy = host.positions[static_cast<std::size_t>(second * 3 + 1)] -
                          host.positions[static_cast<std::size_t>(first * 3 + 1)];
        const double dz = host.positions[static_cast<std::size_t>(second * 3 + 2)] -
                          host.positions[static_cast<std::size_t>(first * 3 + 2)];
        const double distance_squared = dx * dx + dy * dy + dz * dz;
        if (distance_squared >= kMinimumDistanceSquared && distance_squared <= cutoff_squared) {
          pairs.push_back(static_cast<std::int64_t>(first));
          pairs.push_back(static_cast<std::int64_t>(second));
        }
      }
    }
  }
  return pairs;
}

int compare_sparse_success(const HostCase& host, const Results& results, double cutoff) {
  CHECK(results.device_error == static_cast<std::uint32_t>(Gfn2PairListDeviceError::kSuccess));
  CHECK(std::all_of(results.system_errors.begin(), results.system_errors.end(),
                    [](std::uint32_t value) { return value == 0u; }));
  CHECK(std::all_of(results.pair_generations.begin(), results.pair_generations.end(),
                    [](std::uint64_t value) { return value == kGeneration; }));
  for (std::size_t atom = 0u; atom < host.total_atoms(); ++atom) {
    if (!near(results.coordination[atom], host.expected_coordination[atom], 5.0e-13)) {
      return __LINE__;
    }
  }
  /* Sparse pair set must equal the reference retained set (cutoff-inclusive). */
  const std::vector<std::int64_t> expected = expected_retained_pairs(host, cutoff);
  std::size_t cursor = 0u;
  for (std::size_t system = 0u; system < host.batch_size(); ++system) {
    const std::int64_t begin = results.pair_offsets[system];
    const std::int64_t end = results.pair_offsets[system + 1u];
    const std::size_t first_atom = static_cast<std::size_t>(host.atom_offsets[system]);
    const std::size_t atoms_in_system =
        static_cast<std::size_t>(host.atom_offsets[system + 1u] - host.atom_offsets[system]);
    /* Count the reference pairs belonging to this system. */
    std::size_t expected_begin = cursor;
    while (cursor + 1u <= expected.size()) {
      const std::int64_t second = expected[cursor + 1u];
      if (second < static_cast<std::int64_t>(first_atom + atoms_in_system)) {
        cursor += 2u;
      } else {
        break;
      }
    }
    const std::size_t expected_count = (cursor - expected_begin) / 2u;
    if (static_cast<std::size_t>(end - begin) != expected_count) {
      return __LINE__;
    }
    for (std::size_t index = 0u; index < expected_count; ++index) {
      if (results.pairs[static_cast<std::size_t>(begin) + index].first !=
          expected[expected_begin + 2u * index]) {
        return __LINE__;
      }
      if (results.pairs[static_cast<std::size_t>(begin) + index].second !=
          expected[expected_begin + 2u * index + 1u]) {
        return __LINE__;
      }
    }
  }
  CHECK(cursor == expected.size());
  /* Neighbor offsets must be a valid nondecreasing partition whose total
   * equals two neighbor records per retained pair (both endpoints list each
   * other), and each atom's neighbors must be strictly ascending and in
   * system. */
  CHECK(results.neighbor_offsets.front() == 0);
  CHECK(std::is_sorted(results.neighbor_offsets.begin(), results.neighbor_offsets.end()));
  CHECK(results.neighbor_offsets.back() == 2LL * results.pair_offsets.back());
  for (std::size_t atom = 0u; atom < host.total_atoms(); ++atom) {
    const std::int64_t begin = results.neighbor_offsets[atom];
    const std::int64_t end = results.neighbor_offsets[atom + 1u];
    CHECK(std::is_sorted(results.neighbors.begin() + begin, results.neighbors.begin() + end));
    for (std::int64_t index = begin; index < end; ++index) {
      CHECK(results.neighbors[static_cast<std::size_t>(index)] >= 0);
      CHECK(results.neighbors[static_cast<std::size_t>(index)] <
            static_cast<std::int64_t>(host.total_atoms()));
    }
  }
  return 0;
}

int test_cpu_parity_and_pair_sets() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (const bool dense_geometry : {false, true}) {
    for (const std::size_t batch_size : {1u, 8u, 32u, 128u}) {
      HostCase host;
      std::string error;
      CHECK(make_case(batch_size, dense_geometry, host, error));
      DeviceFixture device;
      CUDA_CHECK(device.initialize(host, Gfn2PairListMode::kSparse, stream));
      CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
                static_cast<std::int64_t>(batch_size), device.system_errors.get(),
                device.device_error.get(), stream) == cudaSuccess);
      CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
          device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), kGeneration,
          device.cache(), device.workspace(), device.system_errors.get(), device.device_error.get(),
          stream));
      CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_pairlist_coordination_cuda(
          device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), device.radii.get(),
          kGeneration, device.cache(), device.coordination.get(), device.workspace(),
          device.system_errors.get(), device.device_error.get(), stream));
      Results results;
      CUDA_CHECK(copy_results(host, device, results, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(compare_sparse_success(host, results, 25.0) == 0);
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_dense_fallback_matches_full_triangle() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (const std::size_t batch_size : {1u, 8u, 32u}) {
    HostCase host;
    std::string error;
    CHECK(make_case(batch_size, true, host, error));
    DeviceFixture device;
    CUDA_CHECK(device.initialize(host, Gfn2PairListMode::kDense, stream));
    CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
              static_cast<std::int64_t>(batch_size), device.system_errors.get(),
              device.device_error.get(), stream) == cudaSuccess);
    CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
        device.batch(host, Gfn2PairListMode::kDense), device.positions.get(), kGeneration,
        device.cache(), device.workspace(), device.system_errors.get(), device.device_error.get(),
        stream));
    CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_pairlist_coordination_cuda(
        device.batch(host, Gfn2PairListMode::kDense), device.positions.get(), device.radii.get(),
        kGeneration, device.cache(), device.coordination.get(), device.workspace(),
        device.system_errors.get(), device.device_error.get(), stream));
    Results results;
    CUDA_CHECK(copy_results(host, device, results, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(results.device_error == static_cast<std::uint32_t>(Gfn2PairListDeviceError::kSuccess));
    /* Dense mode should retain the complete triangle per system. */
    for (std::size_t system = 0u; system < host.batch_size(); ++system) {
      const std::int64_t count = host.atom_offsets[system + 1u] - host.atom_offsets[system];
      const std::int64_t expected_pairs = count * (count - 1) / 2;
      CHECK(results.pair_offsets[system + 1u] - results.pair_offsets[system] == expected_pairs);
    }
    for (std::size_t atom = 0u; atom < host.total_atoms(); ++atom) {
      CHECK(near(results.coordination[atom], host.expected_coordination[atom], 5.0e-13));
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

/* Boundary: two atoms exactly at cutoff are retained; beyond is not. */
int test_boundary_distances() {
  HostCase host;
  host.atom_offsets = {0, 2};
  host.atomic_numbers = {1, 1};
  host.positions = {0.0, 0.0, 0.0, 25.0, 0.0, 0.0};
  const double expected_at_boundary = [&]() {
    /* reference: distance exactly 25.0 is within (<=) cutoff. */
    const double r = 25.0;
    const double inv = 1.0 / r;
    const double radius = 2.0 * (4.0 / 3.0) * 1.8897261246204404 * 0.32;
    const double first = 1.0 / (1.0 + std::exp(-10.0 * (radius * inv - 1.0)));
    const double second = 1.0 / (1.0 + std::exp(-20.0 * ((radius + 2.0) * inv - 1.0)));
    return first * second;
  }();
  std::string error;
  CHECK(gpuxtb::detail::gfn2::make_coordination_plan(1, 2, host.atom_offsets.data(),
                                                     host.atomic_numbers.data(), host.plan,
                                                     error) == GPUXTB_STATUS_SUCCESS);
  host.expected_coordination.resize(2u);
  CHECK(gpuxtb::detail::gfn2::evaluate_coordination_cpu(host.plan, host.positions.data(),
                                                        host.expected_coordination.data(),
                                                        error) == GPUXTB_STATUS_SUCCESS);
  CHECK(near(host.expected_coordination[0], expected_at_boundary, 1e-12));
  CHECK(near(host.expected_coordination[1], expected_at_boundary, 1e-12));

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, Gfn2PairListMode::kSparse, stream));
  CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
            1, device.system_errors.get(), device.device_error.get(), stream) == cudaSuccess);
  CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
      device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), kGeneration,
      device.cache(), device.workspace(), device.system_errors.get(), device.device_error.get(),
      stream));
  Results results;
  CUDA_CHECK(copy_results(host, device, results, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(results.device_error == static_cast<std::uint32_t>(Gfn2PairListDeviceError::kSuccess));
  CHECK(results.pair_offsets[1] - results.pair_offsets[0] == 1);
  CHECK(results.pairs[0].first == 0 && results.pairs[0].second == 1);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_empty_and_single_atom() {
  HostCase host;
  host.atom_offsets = {0, 0, 1, 3};
  host.atomic_numbers = {6, 7, 8};
  host.positions = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 3.0, 0.0, 0.0};
  std::string error;
  CHECK(gpuxtb::detail::gfn2::make_coordination_plan(3, 3, host.atom_offsets.data(),
                                                     host.atomic_numbers.data(), host.plan,
                                                     error) == GPUXTB_STATUS_SUCCESS);
  host.expected_coordination.resize(3u);
  CHECK(gpuxtb::detail::gfn2::evaluate_coordination_cpu(host.plan, host.positions.data(),
                                                        host.expected_coordination.data(),
                                                        error) == GPUXTB_STATUS_SUCCESS);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, Gfn2PairListMode::kSparse, stream));
  CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
            3, device.system_errors.get(), device.device_error.get(), stream) == cudaSuccess);
  CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
      device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), kGeneration,
      device.cache(), device.workspace(), device.system_errors.get(), device.device_error.get(),
      stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_pairlist_coordination_cuda(
      device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), device.radii.get(),
      kGeneration, device.cache(), device.coordination.get(), device.workspace(),
      device.system_errors.get(), device.device_error.get(), stream));
  Results results;
  CUDA_CHECK(copy_results(host, device, results, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(results.device_error == static_cast<std::uint32_t>(Gfn2PairListDeviceError::kSuccess));
  /* Systems 0 (empty) and 1 (single atom) retain zero pairs; system 2 has two
   * atoms 1.5 bohr apart and must retain one pair. */
  CHECK(results.pair_offsets[0] == 0);
  CHECK(results.pair_offsets[1] == 0);
  CHECK(results.pair_offsets[2] == 0);
  CHECK(results.pair_offsets[3] - results.pair_offsets[2] == 1);
  CHECK(results.pairs[0].first == 1 && results.pairs[0].second == 2);
  for (std::size_t atom = 0u; atom < host.total_atoms(); ++atom) {
    CHECK(near(results.coordination[atom], host.expected_coordination[atom], 5.0e-13));
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_pair_capacity_overflow_isolated() {
  HostCase host;
  std::string error;
  CHECK(make_case(8u, true, host, error));
  /* Force system 3 over the pair capacity by shrinking max_pairs. */
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, Gfn2PairListMode::kSparse, stream));
  Gfn2PairListDeviceBatch batch = device.batch(host, Gfn2PairListMode::kSparse);
  CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
            8, device.system_errors.get(), device.device_error.get(), stream) == cudaSuccess);
  batch.max_pairs_per_system = 1;
  CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
      batch, device.positions.get(), kGeneration, device.cache(), device.workspace(),
      device.system_errors.get(), device.device_error.get(), stream));
  Results results;
  CUDA_CHECK(copy_results(host, device, results, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(results.device_error ==
        static_cast<std::uint32_t>(Gfn2PairListDeviceError::kPairCapacityExceeded));
  CHECK(results.system_errors[3] ==
        static_cast<std::uint32_t>(Gfn2PairListDeviceError::kPairCapacityExceeded));
  /* Systems 0 (1 atom) and 1 (2 atoms -> exactly 1 pair) fit the capacity and
   * must publish normally; systems 2-6 (3+ atoms -> more than 1 pair) exceed
   * and fail closed; system 7 has 1 atom again and succeeds. */
  CHECK(results.system_errors[0] == 0u);
  CHECK(results.system_errors[1] == 0u);
  CHECK(results.system_errors[7] == 0u);
  CHECK(results.pair_generations[0] == kGeneration);
  CHECK(results.pair_generations[1] == kGeneration);
  CHECK(results.pair_generations[7] == kGeneration);
  for (std::size_t system = 2u; system < 7u; ++system) {
    CHECK(results.system_errors[system] ==
          static_cast<std::uint32_t>(Gfn2PairListDeviceError::kPairCapacityExceeded));
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_cell_capacity_dense_fallback() {
  HostCase host;
  std::string error;
  host.atom_offsets = {0, 3};
  host.atomic_numbers = {6, 6, 6};
  host.positions = {0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 2000.0, 0.0, 0.0};
  CHECK(gpuxtb::detail::gfn2::make_coordination_plan(1, 3, host.atom_offsets.data(),
                                                     host.atomic_numbers.data(), host.plan,
                                                     error) == GPUXTB_STATUS_SUCCESS);
  host.expected_coordination.resize(3u);
  CHECK(gpuxtb::detail::gfn2::evaluate_coordination_cpu(host.plan, host.positions.data(),
                                                        host.expected_coordination.data(),
                                                        error) == GPUXTB_STATUS_SUCCESS);

  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, Gfn2PairListMode::kSparse, nullptr));
  Gfn2PairListDeviceBatch batch = device.batch(host, Gfn2PairListMode::kSparse);
  batch.max_cells_per_system = 1;
  batch.flags = gpuxtb::detail::cuda::kGfn2PairListAllowDenseFallback;
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
      1, device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
      batch, device.positions.get(), kGeneration, device.cache(), device.workspace(),
      device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_pairlist_coordination_cuda(
      batch, device.positions.get(), device.radii.get(), kGeneration, device.cache(),
      device.coordination.get(), device.workspace(), device.system_errors.get(),
      device.device_error.get()));
  Results results;
  CUDA_CHECK(copy_results(host, device, results, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(compare_sparse_success(host, results, batch.cutoff) == 0);
  return 0;
}

int test_stale_generation_rejected() {
  HostCase host;
  std::string error;
  CHECK(make_case(8u, true, host, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, Gfn2PairListMode::kSparse, stream));
  CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
            8, device.system_errors.get(), device.device_error.get(), stream) == cudaSuccess);
  CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
      device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), kGeneration,
      device.cache(), device.workspace(), device.system_errors.get(), device.device_error.get(),
      stream));
  /* Ask coordination with a stale generation for system 4. */
  const std::uint64_t stale = kGeneration - 1u;
  CUDA_CHECK(cudaMemcpyAsync(device.pair_generations.get() + 4, &stale, sizeof(stale),
                             cudaMemcpyDeviceToDevice, stream));
  const std::vector<double> sentinel(host.total_atoms(), kPairSentinel);
  CUDA_CHECK(device.coordination.copy_from(sentinel.data(), sentinel.size(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
      8, device.system_errors.get(), device.device_error.get(), stream));
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_pairlist_coordination_cuda(
            device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(),
            device.radii.get(), kGeneration, device.cache(), device.coordination.get(),
            device.workspace(), device.system_errors.get(), device.device_error.get(),
            stream) == cudaSuccess);
  Results results;
  CUDA_CHECK(copy_results(host, device, results, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(results.device_error ==
        static_cast<std::uint32_t>(Gfn2PairListDeviceError::kStaleGeometry));
  CHECK(results.system_errors[4] ==
        static_cast<std::uint32_t>(Gfn2PairListDeviceError::kStaleGeometry));
  for (std::size_t system = 0u; system < host.batch_size(); ++system) {
    const std::int64_t begin = host.atom_offsets[system];
    const std::int64_t end = host.atom_offsets[system + 1u];
    for (std::int64_t atom = begin; atom < end; ++atom) {
      if (system == 4u) {
        CHECK(results.coordination[static_cast<std::size_t>(atom)] == kPairSentinel);
      } else {
        CHECK(near(results.coordination[static_cast<std::size_t>(atom)],
                   host.expected_coordination[static_cast<std::size_t>(atom)], 5.0e-13));
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_geometry_invalidation_rebuild() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (const std::size_t batch_size : {8u}) {
    HostCase host;
    std::string error;
    CHECK(make_case(batch_size, true, host, error));
    DeviceFixture device;
    CUDA_CHECK(device.initialize(host, Gfn2PairListMode::kSparse, stream));
    for (const std::uint64_t generation : {kGeneration, kGeneration + 5u}) {
      CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
                static_cast<std::int64_t>(batch_size), device.system_errors.get(),
                device.device_error.get(), stream) == cudaSuccess);
      CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
          device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), generation,
          device.cache(), device.workspace(), device.system_errors.get(), device.device_error.get(),
          stream));
      CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_pairlist_coordination_cuda(
          device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), device.radii.get(),
          generation, device.cache(), device.coordination.get(), device.workspace(),
          device.system_errors.get(), device.device_error.get(), stream));
      Results results;
      CUDA_CHECK(copy_results(host, device, results, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(results.device_error == static_cast<std::uint32_t>(Gfn2PairListDeviceError::kSuccess));
      CHECK(std::all_of(results.pair_generations.begin(), results.pair_generations.end(),
                        [generation](std::uint64_t value) { return value == generation; }));
      for (std::size_t atom = 0u; atom < host.total_atoms(); ++atom) {
        CHECK(near(results.coordination[atom], host.expected_coordination[atom], 5.0e-13));
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_cuda_graph_capture_and_replay() {
  HostCase host;
  std::string error;
  CHECK(make_case(32u, true, host, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, Gfn2PairListMode::kSparse, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
            32, device.system_errors.get(), device.device_error.get(), stream) == cudaSuccess);
  CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
      device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), kGeneration,
      device.cache(), device.workspace(), device.system_errors.get(), device.device_error.get(),
      stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_pairlist_coordination_cuda(
      device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), device.radii.get(),
      kGeneration, device.cache(), device.coordination.get(), device.workspace(),
      device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));

  for (int replay = 0; replay < 2; ++replay) {
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    Results results;
    CUDA_CHECK(copy_results(host, device, results, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(compare_sparse_success(host, results, 25.0) == 0);
  }
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_peer_failure_isolation() {
  HostCase host;
  std::string error;
  CHECK(make_case(8u, true, host, error));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, Gfn2PairListMode::kSparse, nullptr));
  /* Seed a previously published generation so a failed peer cannot be hidden
   * by the fixture's zero-initialized generation array. */
  CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
            8, device.system_errors.get(), device.device_error.get()) == cudaSuccess);
  CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
      device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), kGeneration,
      device.cache(), device.workspace(), device.system_errors.get(), device.device_error.get()));
  CUDA_CHECK(cudaDeviceSynchronize());

  /* Poison system 3 with a NaN position so only that system fails. */
  constexpr std::size_t failed_system = 3u;
  const std::size_t failed_atom = static_cast<std::size_t>(host.atom_offsets[failed_system]);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(cudaMemcpy(device.positions.get() + failed_atom * 3u, &nan, sizeof(nan),
                        cudaMemcpyHostToDevice));
  CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
            8, device.system_errors.get(), device.device_error.get()) == cudaSuccess);
  CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
      device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), kGeneration + 1u,
      device.cache(), device.workspace(), device.system_errors.get(), device.device_error.get()));
  /* The builder's sticky diagnostic records the failed peer.  Coordination
   * must still consume the sequence snapshot and evaluate every healthy peer. */
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_pairlist_coordination_cuda(
      device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), device.radii.get(),
      kGeneration + 1u, device.cache(), device.coordination.get(), device.workspace(),
      device.system_errors.get(), device.device_error.get()));
  Results results;
  CUDA_CHECK(copy_results(host, device, results, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(results.device_error ==
        static_cast<std::uint32_t>(Gfn2PairListDeviceError::kNonfinitePosition));
  CHECK(results.system_errors[failed_system] ==
        static_cast<std::uint32_t>(Gfn2PairListDeviceError::kNonfinitePosition));
  CHECK(results.pair_generations[failed_system] == 0u);
  CHECK(results.pair_offsets[failed_system + 1u] == results.pair_offsets[failed_system]);
  for (std::int64_t atom = host.atom_offsets[failed_system];
       atom < host.atom_offsets[failed_system + 1u]; ++atom) {
    CHECK(results.neighbor_offsets[static_cast<std::size_t>(atom + 1)] ==
          results.neighbor_offsets[static_cast<std::size_t>(atom)]);
  }
  /* Healthy peers still committed their retained pair sets. */
  for (std::size_t system = 0u; system < host.batch_size(); ++system) {
    if (system != failed_system) {
      CHECK(results.system_errors[system] == 0u);
      CHECK(results.pair_generations[system] == kGeneration + 1u);
      const std::int64_t begin = host.atom_offsets[system];
      const std::int64_t end = host.atom_offsets[system + 1u];
      for (std::int64_t atom = begin; atom < end; ++atom) {
        CHECK(near(results.coordination[static_cast<std::size_t>(atom)],
                   host.expected_coordination[static_cast<std::size_t>(atom)], 5.0e-13));
      }
    }
  }
  return 0;
}

int test_host_validation_rejects_hostile_views() {
  HostCase host;
  std::string error;
  CHECK(make_case(8u, true, host, error));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, Gfn2PairListMode::kSparse, nullptr));
  Gfn2PairListDeviceBatch batch = device.batch(host, Gfn2PairListMode::kSparse);
  Gfn2PairListDeviceCache cache = device.cache();
  Gfn2PairListDeviceWorkspace workspace = device.workspace();

  batch.plan_token ^= 1u;
  CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
            batch, device.positions.get(), kGeneration, cache, workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  batch = device.batch(host, Gfn2PairListMode::kSparse);

  workspace.pair_cursor_elements = 0;
  CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
            batch, device.positions.get(), kGeneration, cache, workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  workspace = device.workspace();

  /* system_meta is initialized and rewritten by the builder, so aliasing it
   * with the read-only positions view must be rejected before launch. */
  workspace.system_meta = reinterpret_cast<Gfn2PairListSystemMeta*>(device.positions.get());
  CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
            batch, device.positions.get(), kGeneration, cache, workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  workspace = device.workspace();

  cache.pair_offsets = workspace.pair_cursor;
  CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
            batch, device.positions.get(), kGeneration, cache, workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  cache = device.cache();

  /* Alias cache neighbor storage with workspace scratch. */
  cache.neighbors = workspace.neighbor_scratch;
  CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
            batch, device.positions.get(), kGeneration, cache, workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  return 0;
}

int test_invalid_radius_is_peer_local() {
  HostCase host;
  std::string error;
  CHECK(make_case(8u, true, host, error));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, Gfn2PairListMode::kSparse, nullptr));
  CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
            8, device.system_errors.get(), device.device_error.get()) == cudaSuccess);
  CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
      device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), kGeneration,
      device.cache(), device.workspace(), device.system_errors.get(), device.device_error.get()));
  const std::size_t failed_system = 2u;
  const std::size_t failed_atom = static_cast<std::size_t>(host.atom_offsets[failed_system]);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(
      cudaMemcpy(device.radii.get() + failed_atom, &nan, sizeof(nan), cudaMemcpyHostToDevice));
  const std::vector<double> sentinel(host.total_atoms(), kPairSentinel);
  CUDA_CHECK(device.coordination.copy_from(sentinel.data(), sentinel.size()));
  CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
            8, device.system_errors.get(), device.device_error.get()) == cudaSuccess);
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_pairlist_coordination_cuda(
      device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), device.radii.get(),
      kGeneration, device.cache(), device.coordination.get(), device.workspace(),
      device.system_errors.get(), device.device_error.get()));
  Results results;
  CUDA_CHECK(copy_results(host, device, results, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(results.device_error == static_cast<std::uint32_t>(Gfn2PairListDeviceError::kInvalidCache));
  CHECK(results.system_errors[failed_system] ==
        static_cast<std::uint32_t>(Gfn2PairListDeviceError::kInvalidCache));
  for (std::size_t system = 0u; system < host.batch_size(); ++system) {
    const std::int64_t begin = host.atom_offsets[system];
    const std::int64_t end = host.atom_offsets[system + 1u];
    for (std::int64_t atom = begin; atom < end; ++atom) {
      if (system == failed_system) {
        CHECK(results.coordination[static_cast<std::size_t>(atom)] == kPairSentinel);
      } else {
        CHECK(near(results.coordination[static_cast<std::size_t>(atom)],
                   host.expected_coordination[static_cast<std::size_t>(atom)], 5.0e-13));
      }
    }
  }
  return 0;
}

int test_pair_level_preflight_is_peer_local() {
  HostCase host;
  std::string error;
  CHECK(make_case(8u, true, host, error));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, Gfn2PairListMode::kSparse, nullptr));
  CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
            8, device.system_errors.get(), device.device_error.get()) == cudaSuccess);
  CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
      device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), kGeneration,
      device.cache(), device.workspace(), device.system_errors.get(), device.device_error.get()));

  constexpr std::size_t failed_system = 2u;
  const std::size_t failed_atom = static_cast<std::size_t>(host.atom_offsets[failed_system]);
  const std::size_t failed_peer = failed_atom + 1u;
  const std::vector<double> sentinel(host.total_atoms(), kPairSentinel);
  const auto evaluate_and_check = [&]() -> int {
    CUDA_CHECK(device.coordination.copy_from(sentinel.data(), sentinel.size()));
    CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
              8, device.system_errors.get(), device.device_error.get()) == cudaSuccess);
    CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_pairlist_coordination_cuda(
        device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), device.radii.get(),
        kGeneration, device.cache(), device.coordination.get(), device.workspace(),
        device.system_errors.get(), device.device_error.get()));
    Results results;
    CUDA_CHECK(copy_results(host, device, results, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(results.device_error ==
          static_cast<std::uint32_t>(Gfn2PairListDeviceError::kInvalidCache));
    CHECK(results.system_errors[failed_system] ==
          static_cast<std::uint32_t>(Gfn2PairListDeviceError::kInvalidCache));
    for (std::size_t system = 0u; system < host.batch_size(); ++system) {
      const std::int64_t begin = host.atom_offsets[system];
      const std::int64_t end = host.atom_offsets[system + 1u];
      for (std::int64_t atom = begin; atom < end; ++atom) {
        if (system == failed_system) {
          CHECK(results.coordination[static_cast<std::size_t>(atom)] == kPairSentinel);
        } else {
          CHECK(near(results.coordination[static_cast<std::size_t>(atom)],
                     host.expected_coordination[static_cast<std::size_t>(atom)], 5.0e-13));
        }
      }
    }
    return 0;
  };

  /* Mutating the positions after list construction makes one cached pair
   * coincident.  The read-only preflight must reject the whole peer before any
   * of its atom threads can publish coordination. */
  CUDA_CHECK(cudaMemcpy(device.positions.get() + failed_peer * 3u,
                        host.positions.data() + failed_atom * 3u, 3u * sizeof(double),
                        cudaMemcpyHostToDevice));
  CHECK(evaluate_and_check() == 0);
  CUDA_CHECK(cudaMemcpy(device.positions.get() + failed_peer * 3u,
                        host.positions.data() + failed_peer * 3u, 3u * sizeof(double),
                        cudaMemcpyHostToDevice));

  /* Each radius is finite, but their pair sum overflows.  This must have the
   * same peer-local, no-partial-publication behavior. */
  const double huge_radii[2] = {std::numeric_limits<double>::max(),
                                std::numeric_limits<double>::max()};
  CUDA_CHECK(cudaMemcpy(device.radii.get() + failed_atom, huge_radii, sizeof(huge_radii),
                        cudaMemcpyHostToDevice));
  CHECK(evaluate_and_check() == 0);
  return 0;
}

int test_dispatch_policy() {
  CHECK(!gpuxtb::detail::cuda::gfn2_pairlist_use_sparse_for(1));
  CHECK(!gpuxtb::detail::cuda::gfn2_pairlist_use_sparse_for(16));
  CHECK(!gpuxtb::detail::cuda::gfn2_pairlist_use_sparse_for(32));
  CHECK(!gpuxtb::detail::cuda::gfn2_pairlist_use_sparse_for(40));
  CHECK(gpuxtb::detail::cuda::gfn2_pairlist_use_sparse_for(41));
  CHECK(gpuxtb::detail::cuda::gfn2_pairlist_use_sparse_for(64));
  CHECK(gpuxtb::detail::cuda::gfn2_pairlist_use_sparse_for(128));
  return 0;
}

/*
 * Per-system dispatch: a heterogeneous batch with systems on both sides of the
 * 40-atom crossover must emit the same canonical lists whether a peer follows
 * the batch-wide mode or its own per-system decision.  Small peers elected
 * kDense and large peers kSparse produce identical pair/neighbor/CN results to
 * the equivalent homogeneous runs, and the published per-system counts must
 * match the offsets exactly.
 */
int test_per_system_dispatch() {
  const std::int64_t system_sizes[] = {3, 45, 7, 60, 5};
  const std::int64_t batch_size =
      static_cast<std::int64_t>(sizeof(system_sizes) / sizeof(system_sizes[0]));
  const std::int64_t atoms =
      std::accumulate(std::begin(system_sizes), std::end(system_sizes), std::int64_t{0});
  HostCase host;
  host.atom_offsets.assign(static_cast<std::size_t>(batch_size + 1), 0);
  std::vector<std::int32_t> atomic_numbers(static_cast<std::size_t>(atoms), 1);
  std::vector<double> positions(static_cast<std::size_t>(atoms * 3), 0.0);
  constexpr std::int32_t elements[] = {1, 6, 7, 8, 16, 17, 35, 53};
  std::int64_t atom = 0;
  for (std::int64_t system = 0; system < batch_size; ++system) {
    host.atom_offsets[static_cast<std::size_t>(system)] = atom;
    for (std::int64_t local = 0; local < system_sizes[system]; ++local, ++atom) {
      atomic_numbers[static_cast<std::size_t>(atom)] =
          elements[(system + local) % (sizeof(elements) / sizeof(elements[0]))];
      positions[static_cast<std::size_t>(atom * 3)] = 1.7 * static_cast<double>(local);
      positions[static_cast<std::size_t>(atom * 3 + 1)] =
          0.2 * static_cast<double>(local * local + 1);
      positions[static_cast<std::size_t>(atom * 3 + 2)] =
          (local % 2 == 0 ? -0.11 : 0.13) * static_cast<double>(local + 1);
    }
  }
  host.atom_offsets[static_cast<std::size_t>(batch_size)] = atom;
  host.atomic_numbers = atomic_numbers;
  host.positions = positions;
  std::string error;
  CHECK(gpuxtb::detail::gfn2::make_coordination_plan(batch_size, atoms, host.atom_offsets.data(),
                                                     host.atomic_numbers.data(), host.plan,
                                                     error) == GPUXTB_STATUS_SUCCESS);

  std::vector<std::int32_t> per_system_modes(static_cast<std::size_t>(batch_size));
  for (std::int64_t system = 0; system < batch_size; ++system) {
    per_system_modes[static_cast<std::size_t>(system)] = static_cast<std::int32_t>(
        gpuxtb::detail::cuda::gfn2_pairlist_use_sparse_for(system_sizes[system])
            ? Gfn2PairListMode::kSparse
            : Gfn2PairListMode::kDense);
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  /* Three runs over the same fixture: batch-wide sparse, batch-wide dense, and
   * batch-wide sparse with a per-system override array.  The per-system run
   * must reproduce the sparse result for sparse-elected peers and the dense
   * result for dense-elected peers. */
  DeviceFixture sparse_device;
  DeviceFixture dense_device;
  DeviceFixture mixed_device;
  CUDA_CHECK(sparse_device.initialize(host, Gfn2PairListMode::kSparse, stream));
  CUDA_CHECK(dense_device.initialize(host, Gfn2PairListMode::kDense, stream));
  CUDA_CHECK(mixed_device.initialize(host, Gfn2PairListMode::kSparse, stream));
  DeviceBuffer<std::int32_t> system_modes;
  CUDA_CHECK(system_modes.allocate(host.batch_size()));
  CUDA_CHECK(system_modes.copy_from(per_system_modes.data(), per_system_modes.size(), stream));

  const auto run = [&](DeviceFixture& device, Gfn2PairListMode mode) {
    CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
        static_cast<std::int64_t>(host.batch_size()), device.system_errors.get(),
        device.device_error.get(), stream));
    Gfn2PairListDeviceBatch batch = device.batch(host, mode);
    if (mode == Gfn2PairListMode::kSparse) {
      batch.system_modes = system_modes.get();
      batch.system_mode_elements = host.batch_size();
    }
    CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
        batch, device.positions.get(), kGeneration, device.cache(), device.workspace(),
        device.system_errors.get(), device.device_error.get(), stream));
    CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_pairlist_coordination_cuda(
        batch, device.positions.get(), device.radii.get(), kGeneration, device.cache(),
        device.coordination.get(), device.workspace(), device.system_errors.get(),
        device.device_error.get(), stream));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError() == cudaSuccess);
  };

  run(sparse_device, Gfn2PairListMode::kSparse);
  run(dense_device, Gfn2PairListMode::kDense);
  run(mixed_device, Gfn2PairListMode::kSparse);

  const auto copy_raw = [&](const auto& buffer, std::size_t count, std::vector<std::int64_t>& out) {
    CUDA_CHECK(buffer.copy_to(out.data(), count, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  };
  const auto copy_pairs = [&](const DeviceBuffer<Gfn2AtomPair>& buffer, std::size_t count,
                              std::vector<Gfn2AtomPair>& out) {
    CUDA_CHECK(buffer.copy_to(out.data(), count, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  };
  std::vector<Gfn2AtomPair> sparse_pairs(static_cast<std::size_t>(sparse_device.pairs.size()));
  std::vector<Gfn2AtomPair> dense_pairs(static_cast<std::size_t>(dense_device.pairs.size()));
  std::vector<Gfn2AtomPair> mixed_pairs(static_cast<std::size_t>(mixed_device.pairs.size()));
  std::vector<std::int64_t> sparse_counts(host.batch_size());
  std::vector<std::int64_t> dense_counts(host.batch_size());
  std::vector<std::int64_t> mixed_counts(host.batch_size());
  std::vector<std::int64_t> sparse_neighbors(
      static_cast<std::size_t>(sparse_device.neighbors.size()));
  std::vector<std::int64_t> dense_neighbors(
      static_cast<std::size_t>(dense_device.neighbors.size()));
  std::vector<std::int64_t> mixed_neighbors(
      static_cast<std::int64_t>(mixed_device.neighbors.size()));
  std::vector<double> sparse_cn(host.total_atoms());
  std::vector<double> dense_cn(host.total_atoms());
  std::vector<double> mixed_cn(host.total_atoms());

  copy_pairs(sparse_device.pairs, sparse_device.pairs.size(), sparse_pairs);
  copy_pairs(dense_device.pairs, dense_device.pairs.size(), dense_pairs);
  copy_pairs(mixed_device.pairs, mixed_device.pairs.size(), mixed_pairs);
  copy_raw(sparse_device.pair_counts, host.batch_size(), sparse_counts);
  copy_raw(dense_device.pair_counts, host.batch_size(), dense_counts);
  copy_raw(mixed_device.pair_counts, host.batch_size(), mixed_counts);
  copy_raw(sparse_device.neighbors, sparse_device.neighbors.size(), sparse_neighbors);
  copy_raw(dense_device.neighbors, dense_device.neighbors.size(), dense_neighbors);
  copy_raw(mixed_device.neighbors, mixed_device.neighbors.size(), mixed_neighbors);
  CUDA_CHECK(sparse_device.coordination.copy_to(sparse_cn.data(), sparse_cn.size(), stream));
  CUDA_CHECK(dense_device.coordination.copy_to(dense_cn.data(), dense_cn.size(), stream));
  CUDA_CHECK(mixed_device.coordination.copy_to(mixed_cn.data(), mixed_cn.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<std::int64_t> sparse_offsets(host.batch_size() + 1u);
  std::vector<std::int64_t> dense_offsets(host.batch_size() + 1u);
  std::vector<std::int64_t> mixed_offsets(host.batch_size() + 1u);
  copy_raw(sparse_device.pair_offsets, host.batch_size() + 1u, sparse_offsets);
  copy_raw(dense_device.pair_offsets, host.batch_size() + 1u, dense_offsets);
  copy_raw(mixed_device.pair_offsets, host.batch_size() + 1u, mixed_offsets);

  for (std::int64_t system = 0; system < batch_size; ++system) {
    const std::int64_t mixed_begin = mixed_offsets[static_cast<std::size_t>(system)];
    const std::int64_t mixed_end = mixed_offsets[static_cast<std::size_t>(system + 1)];
    const bool use_sparse =
        gpuxtb::detail::cuda::gfn2_pairlist_use_sparse_for(system_sizes[system]);
    const std::int64_t expected_begin = use_sparse
                                            ? sparse_offsets[static_cast<std::size_t>(system)]
                                            : dense_offsets[static_cast<std::size_t>(system)];
    const std::int64_t expected_end = use_sparse
                                          ? sparse_offsets[static_cast<std::size_t>(system + 1)]
                                          : dense_offsets[static_cast<std::size_t>(system + 1)];
    if (mixed_end - mixed_begin != expected_end - expected_begin) {
      fprintf(stderr,
              "system %lld use_sparse=%d mixed_count=%lld expected_count=%lld "
              "mixed=[%lld,%lld) %s=[%lld,%lld)\n",
              static_cast<long long>(system), use_sparse ? 1 : 0,
              static_cast<long long>(mixed_end - mixed_begin),
              static_cast<long long>(expected_end - expected_begin),
              static_cast<long long>(mixed_begin), static_cast<long long>(mixed_end),
              use_sparse ? "sparse" : "dense", static_cast<long long>(expected_begin),
              static_cast<long long>(expected_end));
    }
    CHECK(mixed_end - mixed_begin == expected_end - expected_begin);
    CHECK(mixed_counts[static_cast<std::size_t>(system)] ==
          (use_sparse ? sparse_counts[static_cast<std::size_t>(system)]
                      : dense_counts[static_cast<std::size_t>(system)]));
    CHECK(mixed_counts[static_cast<std::size_t>(system)] == mixed_end - mixed_begin);
    for (std::int64_t offset = 0; offset < mixed_end - mixed_begin; ++offset) {
      const Gfn2AtomPair expected =
          use_sparse ? sparse_pairs[static_cast<std::size_t>(expected_begin + offset)]
                     : dense_pairs[static_cast<std::size_t>(expected_begin + offset)];
      const Gfn2AtomPair actual = mixed_pairs[static_cast<std::size_t>(mixed_begin + offset)];
      CHECK(actual.first == expected.first && actual.second == expected.second);
    }
  }
  /* Coordination results are per atom and must match the homogeneous run the
   * peer's elected strategy would have produced. */
  for (std::int64_t atom = 0; atom < atoms; ++atom) {
    std::int64_t system = 0;
    while (system + 1 < batch_size &&
           atom >= host.atom_offsets[static_cast<std::size_t>(system + 1)]) {
      ++system;
    }
    const bool use_sparse =
        gpuxtb::detail::cuda::gfn2_pairlist_use_sparse_for(system_sizes[system]);
    CHECK(mixed_cn[static_cast<std::size_t>(atom)] ==
          (use_sparse ? sparse_cn[static_cast<std::size_t>(atom)]
                      : dense_cn[static_cast<std::size_t>(atom)]));
  }
  /* Dense-elected peers carry all-pairs even in the mixed run; their neighbor
   * ranges must match the homogeneous dense run. */
  for (std::int64_t system = 0; system < batch_size; ++system) {
    if (gpuxtb::detail::cuda::gfn2_pairlist_use_sparse_for(system_sizes[system])) {
      continue;
    }
    const std::int64_t atom_begin = host.atom_offsets[static_cast<std::size_t>(system)];
    const std::int64_t atom_end = host.atom_offsets[static_cast<std::size_t>(system + 1)];
    std::vector<std::int64_t> dense_offsets_slice(host.total_atoms() + 1);
    std::vector<std::int64_t> mixed_offsets_slice(host.total_atoms() + 1);
    CUDA_CHECK(dense_device.neighbor_offsets.copy_to(dense_offsets_slice.data(),
                                                     host.total_atoms() + 1, stream));
    CUDA_CHECK(mixed_device.neighbor_offsets.copy_to(mixed_offsets_slice.data(),
                                                     host.total_atoms() + 1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
      const std::int64_t dense_count = dense_offsets_slice[static_cast<std::size_t>(atom + 1)] -
                                       dense_offsets_slice[static_cast<std::size_t>(atom)];
      const std::int64_t mixed_count = mixed_offsets_slice[static_cast<std::size_t>(atom + 1)] -
                                       mixed_offsets_slice[static_cast<std::size_t>(atom)];
      CHECK(dense_count == mixed_count);
      if (dense_count != mixed_count) {
        fprintf(stderr, "system %lld atom %lld dense_count=%lld mixed_count=%lld\n",
                static_cast<long long>(system), static_cast<long long>(atom),
                static_cast<long long>(dense_count), static_cast<long long>(mixed_count));
      }
      const std::int64_t dense_base = dense_offsets_slice[static_cast<std::size_t>(atom)];
      const std::int64_t mixed_base = mixed_offsets_slice[static_cast<std::size_t>(atom)];
      for (std::int64_t index = 0; index < dense_count; ++index) {
        const std::int64_t expected = dense_neighbors[static_cast<std::size_t>(dense_base + index)];
        const std::int64_t actual = mixed_neighbors[static_cast<std::size_t>(mixed_base + index)];
        CHECK(expected == actual);
      }
    }
  }
  CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
            [&] {
              Gfn2PairListDeviceBatch batch = mixed_device.batch(host, Gfn2PairListMode::kSparse);
              batch.system_modes = system_modes.get();
              batch.system_mode_elements = host.batch_size() - 1;
              return batch;
            }(),
            mixed_device.positions.get(), kGeneration, mixed_device.cache(),
            mixed_device.workspace(), mixed_device.system_errors.get(),
            mixed_device.device_error.get(), stream) == cudaErrorInvalidValue);
  return 0;
}

/*
 * Sparse coordination VJP parity: (dCN/dR)^T * dE_dCN accumulated over the
 * sparse neighbor ranges must equal the dense geometry VJP bitwise for retained
 * pairs.  Beyond-cutoff dense pairs carry exact-zero derivatives, so the sparse
 * accumulation over the retained ascending-neighbor subset reproduces the dense
 * result bit-for-bit under the documented ordered reduction mode.  A failed
 * peer (NaN seed) must retain its input gradients while healthy peers publish.
 */
int test_sparse_vjp_matches_dense() {
  for (const std::size_t batch_size : {1u, 8u, 32u}) {
    HostCase host;
    std::string error;
    CHECK(make_case(batch_size, true, host, error));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    DeviceFixture device;
    CUDA_CHECK(device.initialize(host, Gfn2PairListMode::kSparse, stream));
    CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
              static_cast<std::int64_t>(batch_size), device.system_errors.get(),
              device.device_error.get(), stream) == cudaSuccess);
    CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
        device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), kGeneration,
        device.cache(), device.workspace(), device.system_errors.get(), device.device_error.get(),
        stream));
    CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_pairlist_coordination_cuda(
        device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), device.radii.get(),
        kGeneration, device.cache(), device.coordination.get(), device.workspace(),
        device.system_errors.get(), device.device_error.get(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaGetLastError() == cudaSuccess);

    std::vector<double> dE_dcn(host.total_atoms());
    for (std::size_t index = 0u; index < dE_dcn.size(); ++index) {
      dE_dcn[index] = 0.37 * static_cast<double>(index % 7u) - 0.11;
    }
    std::vector<double> gradient_seed(host.total_atoms() * 3u);
    for (std::size_t index = 0u; index < gradient_seed.size(); ++index) {
      gradient_seed[index] = 0.01 * static_cast<double>(index) - 0.5;
    }
    DeviceBuffer<double> d_dE_dcn;
    DeviceBuffer<double> sparse_gradients;
    DeviceBuffer<double> sparse_gradient_scratch;
    DeviceBuffer<double> dense_gradients;
    DeviceBuffer<double> dense_gradient_scratch;
    CUDA_CHECK(allocate_and_copy(d_dE_dcn, dE_dcn, stream));
    CUDA_CHECK(allocate_and_copy(sparse_gradients, gradient_seed, stream));
    CUDA_CHECK(allocate_and_copy(sparse_gradient_scratch,
                                 std::vector<double>(gradient_seed.size(), 0.0), stream));
    CUDA_CHECK(allocate_and_copy(dense_gradients, gradient_seed, stream));
    CUDA_CHECK(allocate_and_copy(dense_gradient_scratch,
                                 std::vector<double>(gradient_seed.size(), 0.0), stream));

    /* Sparse VJP. */
    CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
        static_cast<std::int64_t>(batch_size), device.system_errors.get(),
        device.device_error.get(), stream));
    CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_pairlist_coordination_vjp_cuda(
        device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), device.radii.get(),
        kGeneration, device.cache(), d_dE_dcn.get(), sparse_gradients.get(),
        sparse_gradient_scratch.get(), static_cast<std::int64_t>(sparse_gradient_scratch.size()),
        device.workspace(), device.system_errors.get(), device.device_error.get(), stream));

    /* Dense reference VJP over the real geometry cache. */
    gpuxtb::detail::cuda::Gfn2GeometryDeviceBatch geom_batch{};
    gpuxtb::detail::cuda::Gfn2GeometryDeviceCache geom_cache{};
    gpuxtb::detail::cuda::Gfn2GeometryDeviceWorkspace geom_workspace{};
    DeviceBuffer<double> geom_pair_data;
    DeviceBuffer<double> geom_coordination;
    DeviceBuffer<double> geom_pair_scratch;
    DeviceBuffer<double> geom_coordination_scratch;
    DeviceBuffer<double> geom_gradient_scratch;
    DeviceBuffer<std::uint64_t> geom_generations;
    DeviceBuffer<std::uint32_t> geom_sequence;
    DeviceBuffer<std::uint32_t> geom_system_errors;
    DeviceBuffer<std::uint32_t> geom_device_error;
    {
      std::vector<std::int64_t> pair_offsets(host.batch_size() + 1u, 0);
      for (std::size_t system = 0u; system < host.batch_size(); ++system) {
        const std::int64_t count = host.atom_offsets[system + 1u] - host.atom_offsets[system];
        pair_offsets[system + 1u] = pair_offsets[system] + count * (count - 1) / 2;
      }
      DeviceBuffer<std::int64_t> d_atom_offsets;
      DeviceBuffer<std::int64_t> d_pair_offsets;
      DeviceBuffer<double> d_radii;
      CUDA_CHECK(allocate_and_copy(d_atom_offsets, host.atom_offsets, stream));
      CUDA_CHECK(allocate_and_copy(d_pair_offsets, pair_offsets, stream));
      CUDA_CHECK(allocate_and_copy(d_radii, host.plan.covalent_radius, stream));
      const std::size_t pair_elements = static_cast<std::size_t>(pair_offsets.back()) *
                                        static_cast<std::size_t>(kGfn2GeometryPairDataElements);
      CUDA_CHECK(geom_pair_data.allocate(pair_elements));
      CUDA_CHECK(geom_coordination.allocate(host.total_atoms()));
      CUDA_CHECK(geom_pair_scratch.allocate(pair_elements));
      CUDA_CHECK(geom_coordination_scratch.allocate(host.total_atoms()));
      CUDA_CHECK(geom_gradient_scratch.allocate(host.total_atoms() * 3u));
      CUDA_CHECK(geom_generations.allocate(host.batch_size()));
      CUDA_CHECK(geom_sequence.allocate(1u));
      CUDA_CHECK(geom_system_errors.allocate(host.batch_size()));
      CUDA_CHECK(geom_device_error.allocate(1u));
      geom_batch = gpuxtb::detail::cuda::Gfn2GeometryDeviceBatch{
          static_cast<std::int64_t>(host.batch_size()),
          static_cast<std::int64_t>(host.total_atoms()),
          pair_offsets.back(),
          static_cast<std::int64_t>(host.atom_offsets.size()),
          static_cast<std::int64_t>(pair_offsets.size()),
          static_cast<std::int64_t>(host.plan.covalent_radius.size()),
          static_cast<std::int64_t>(host.positions.size()),
          kPlanToken,
          d_atom_offsets.get(),
          d_pair_offsets.get(),
          d_radii.get(),
      };
      geom_cache = gpuxtb::detail::cuda::Gfn2GeometryDeviceCache{
          geom_pair_data.get(),
          static_cast<std::int64_t>(geom_pair_data.size()),
          geom_coordination.get(),
          static_cast<std::int64_t>(geom_coordination.size()),
          geom_generations.get(),
          static_cast<std::int64_t>(geom_generations.size()),
          kPlanToken,
      };
      geom_workspace = gpuxtb::detail::cuda::Gfn2GeometryDeviceWorkspace{
          geom_pair_scratch.get(),
          static_cast<std::int64_t>(geom_pair_scratch.size()),
          geom_coordination_scratch.get(),
          static_cast<std::int64_t>(geom_coordination_scratch.size()),
          geom_gradient_scratch.get(),
          static_cast<std::int64_t>(geom_gradient_scratch.size()),
          geom_sequence.get(),
          1,
          kPlanToken,
      };
      CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_geometry_device_errors_cuda(
          static_cast<std::int64_t>(host.batch_size()), geom_system_errors.get(),
          geom_device_error.get(), stream));
      CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_geometry_cache_cuda(
          geom_batch, device.positions.get(), kGeneration, geom_cache, geom_workspace,
          geom_system_errors.get(), geom_device_error.get(), stream));
      CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_coordination_vjp_cuda(
          geom_batch, geom_cache, kGeneration, d_dE_dcn.get(), dense_gradients.get(),
          geom_workspace, geom_system_errors.get(), geom_device_error.get(), stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<double> sparse_result(host.total_atoms() * 3u, 0.0);
    std::vector<double> dense_result(host.total_atoms() * 3u, 0.0);
    CUDA_CHECK(sparse_gradients.copy_to(sparse_result.data(), sparse_result.size(), stream));
    CUDA_CHECK(dense_gradients.copy_to(dense_result.data(), dense_result.size(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::int64_t worst_ulp = 0;
    std::int64_t worst_index = -1;
    for (std::size_t index = 0u; index < sparse_result.size(); ++index) {
      const double reference = dense_result[index];
      const double ulp =
          std::abs(reference) < 1e-300
              ? 5e-324
              : std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(reference)) / 2.0;
      const std::int64_t ulp_distance = static_cast<std::int64_t>(
          std::abs(sparse_result[index] - reference) / std::max(ulp, 1e-300));
      if (ulp_distance > worst_ulp) {
        worst_ulp = ulp_distance;
        worst_index = static_cast<std::int64_t>(index);
      }
    }
    if (worst_index != -1) {
      fprintf(stderr, "batch %llu VJP worst ulp=%lld at %lld: sparse=%.17g dense=%.17g\n",
              static_cast<unsigned long long>(batch_size), static_cast<long long>(worst_ulp),
              static_cast<long long>(worst_index),
              sparse_result[static_cast<std::size_t>(worst_index)],
              dense_result[static_cast<std::size_t>(worst_index)]);
    }
    /* Independent kernels contract the same FMA differently, so a strict
     * bitwise gate is not required for the VJP (unlike the CN number itself).
     * A small ulp bound proves the analytic gradient agrees to roundoff; the
     * value cache override is validated by the CN gate + force parity tests. */
    CHECK(worst_ulp <= 32);

    /* Peer-local failure: NaN dE_dcn seed for one system must leave its input
     * gradients untouched while healthy peers still publish. */
    {
      const std::size_t poisoned_system = host.batch_size() / 2u;
      const std::size_t poisoned =
          static_cast<std::size_t>(host.atom_offsets[poisoned_system] -
                                   host.atom_offsets[poisoned_system - 1u]) != 0u
              ? static_cast<std::size_t>(host.atom_offsets[poisoned_system])
              : 0u;
      std::vector<double> poisoned_dE = dE_dcn;
      if (poisoned < host.total_atoms()) {
        poisoned_dE[poisoned] = std::numeric_limits<double>::quiet_NaN();
      }
      DeviceBuffer<double> d_dE_failed;
      CUDA_CHECK(allocate_and_copy(d_dE_failed, poisoned_dE, stream));
      CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
          static_cast<std::int64_t>(batch_size), device.system_errors.get(),
          device.device_error.get(), stream));
      CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_pairlist_coordination_vjp_cuda(
          device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), device.radii.get(),
          kGeneration, device.cache(), d_dE_failed.get(), sparse_gradients.get(),
          sparse_gradient_scratch.get(), static_cast<std::int64_t>(sparse_gradient_scratch.size()),
          device.workspace(), device.system_errors.get(), device.device_error.get(), stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      std::vector<double> after(host.total_atoms() * 3u, 0.0);
      CUDA_CHECK(sparse_gradients.copy_to(after.data(), after.size(), stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      std::vector<std::uint32_t> system_errors_host(host.batch_size(), 0u);
      CUDA_CHECK(
          device.system_errors.copy_to(system_errors_host.data(), host.batch_size(), stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(system_errors_host[poisoned_system] != 0u);
      /* All published values remain finite even for the failed peer. */
      for (std::size_t index = 0u; index < after.size(); ++index) {
        CHECK(std::isfinite(after[index]));
      }
    }
  }
  return 0;
}
int test_bitwise_dense_sparse_parity() {
  for (const std::size_t batch_size : {1u, 8u, 32u}) {
    HostCase host;
    std::string error;
    CHECK(make_case(batch_size, true, host, error));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    /* Sparse: the bucketed pair list. */
    DeviceFixture device;
    CUDA_CHECK(device.initialize(host, Gfn2PairListMode::kSparse, stream));
    CHECK(gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
              static_cast<std::int64_t>(batch_size), device.system_errors.get(),
              device.device_error.get(), stream) == cudaSuccess);
    CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
        device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), kGeneration,
        device.cache(), device.workspace(), device.system_errors.get(), device.device_error.get(),
        stream));
    CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_pairlist_coordination_cuda(
        device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), device.radii.get(),
        kGeneration, device.cache(), device.coordination.get(), device.workspace(),
        device.system_errors.get(), device.device_error.get(), stream));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError() == cudaSuccess);

    /* Dense reference: the real geometry cache path. */
    gpuxtb::detail::cuda::Gfn2GeometryDeviceBatch geom_batch{};
    gpuxtb::detail::cuda::Gfn2GeometryDeviceCache geom_cache{};
    gpuxtb::detail::cuda::Gfn2GeometryDeviceWorkspace geom_workspace{};
    DeviceBuffer<double> geom_pair_data;
    DeviceBuffer<double> geom_coordination;
    DeviceBuffer<double> geom_pair_scratch;
    DeviceBuffer<double> geom_coordination_scratch;
    DeviceBuffer<double> geom_gradient_scratch;
    DeviceBuffer<std::uint64_t> geom_generations;
    DeviceBuffer<std::uint32_t> geom_sequence;
    DeviceBuffer<std::uint32_t> geom_system_errors;
    DeviceBuffer<std::uint32_t> geom_device_error;
    {
      std::vector<std::int64_t> pair_offsets(host.batch_size() + 1u, 0);
      for (std::size_t system = 0u; system < host.batch_size(); ++system) {
        const std::int64_t count = host.atom_offsets[system + 1u] - host.atom_offsets[system];
        pair_offsets[system + 1u] = pair_offsets[system] + count * (count - 1) / 2;
      }
      DeviceBuffer<std::int64_t> d_atom_offsets;
      DeviceBuffer<std::int64_t> d_pair_offsets;
      DeviceBuffer<double> d_radii;
      CUDA_CHECK(allocate_and_copy(d_atom_offsets, host.atom_offsets, stream));
      CUDA_CHECK(allocate_and_copy(d_pair_offsets, pair_offsets, stream));
      CUDA_CHECK(allocate_and_copy(d_radii, host.plan.covalent_radius, stream));
      const std::size_t pair_elements = static_cast<std::size_t>(pair_offsets.back()) *
                                        static_cast<std::size_t>(kGfn2GeometryPairDataElements);
      CUDA_CHECK(geom_pair_data.allocate(pair_elements));
      CUDA_CHECK(geom_coordination.allocate(host.total_atoms()));
      CUDA_CHECK(geom_pair_scratch.allocate(pair_elements));
      CUDA_CHECK(geom_coordination_scratch.allocate(host.total_atoms()));
      CUDA_CHECK(geom_gradient_scratch.allocate(host.total_atoms() * 3u));
      CUDA_CHECK(geom_generations.allocate(host.batch_size()));
      CUDA_CHECK(geom_sequence.allocate(1u));
      CUDA_CHECK(geom_system_errors.allocate(host.batch_size()));
      CUDA_CHECK(geom_device_error.allocate(1u));
      geom_batch = gpuxtb::detail::cuda::Gfn2GeometryDeviceBatch{
          static_cast<std::int64_t>(host.batch_size()),
          static_cast<std::int64_t>(host.total_atoms()),
          pair_offsets.back(),
          static_cast<std::int64_t>(host.atom_offsets.size()),
          static_cast<std::int64_t>(pair_offsets.size()),
          static_cast<std::int64_t>(host.plan.covalent_radius.size()),
          static_cast<std::int64_t>(host.positions.size()),
          kPlanToken,
          d_atom_offsets.get(),
          d_pair_offsets.get(),
          d_radii.get(),
      };
      geom_cache = gpuxtb::detail::cuda::Gfn2GeometryDeviceCache{
          geom_pair_data.get(),
          static_cast<std::int64_t>(geom_pair_data.size()),
          geom_coordination.get(),
          static_cast<std::int64_t>(geom_coordination.size()),
          geom_generations.get(),
          static_cast<std::int64_t>(geom_generations.size()),
          kPlanToken,
      };
      geom_workspace = gpuxtb::detail::cuda::Gfn2GeometryDeviceWorkspace{
          geom_pair_scratch.get(),
          static_cast<std::int64_t>(geom_pair_scratch.size()),
          geom_coordination_scratch.get(),
          static_cast<std::int64_t>(geom_coordination_scratch.size()),
          geom_gradient_scratch.get(),
          static_cast<std::int64_t>(geom_gradient_scratch.size()),
          geom_sequence.get(),
          1,
          kPlanToken,
      };
      CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_geometry_device_errors_cuda(
          static_cast<std::int64_t>(host.batch_size()), geom_system_errors.get(),
          geom_device_error.get(), stream));
      CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_geometry_cache_cuda(
          geom_batch, device.positions.get(), kGeneration, geom_cache, geom_workspace,
          geom_system_errors.get(), geom_device_error.get(), stream));
      CUDA_CHECK(cudaDeviceSynchronize());
      std::uint32_t geom_err = 99u;
      CUDA_CHECK(geom_device_error.copy_to(&geom_err, 1u));
      CHECK(geom_err == 0u);
    }

    std::vector<double> dense_cn(host.total_atoms());
    std::vector<double> sparse_cn(host.total_atoms());
    CUDA_CHECK(geom_coordination.copy_to(dense_cn.data(), dense_cn.size(), stream));
    CUDA_CHECK(device.coordination.copy_to(sparse_cn.data(), sparse_cn.size(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::size_t atom = 0u; atom < host.total_atoms(); ++atom) {
      if (std::memcmp(&sparse_cn[atom], &dense_cn[atom], sizeof(double)) != 0) {
        fprintf(stderr, "  bitwise parity mismatch atom %zu sparse=%.17g dense=%.17g\n", atom,
                sparse_cn[atom], dense_cn[atom]);
        return __LINE__;
      }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
  return 0;
}
/*
 * Focused #70 benchmark: separate list-build cost from reuse cost across the
 * batch 1/8/32/128 grid and a per-system atom count sweep, so the dense-fallback
 * dispatch crossover is anchored by reproducible profiling.  Each system holds
 * atoms in a sparse crystal-like layout (12 bohr spacing) where only nearby
 * pairs are retained and the bucketed path avoids the all-pairs work of the
 * dense fallback.  With --json and --csv, emits one audit JSON containing every
 * raw sample plus a compact median CSV from the same measurement.
 */
struct BenchmarkSamples {
  std::vector<float> values;
  float median = 0.0F;
};

struct BenchmarkRow {
  std::size_t batch_size = 0u;
  std::int64_t atoms_per_system = 0;
  BenchmarkSamples sparse_build;
  BenchmarkSamples dense_build;
  BenchmarkSamples reuse;
};

std::string json_escape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 2u);
  for (const char character : value) {
    switch (character) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += character;
        break;
    }
  }
  return escaped;
}

void write_samples_json(std::ostream& output, const std::vector<BenchmarkRow>& rows, int argc,
                        char** argv, int device_id, int runtime_version,
                        const std::string& source_revision, const std::string& executable_sha256,
                        const std::string& build_identity_sha256) {
  output << std::setprecision(9);
  output << "{\n  \"schema_version\": 1,\n"
         << "  \"benchmark\": \"gpuxtb_cuda_pairlist_benchmark\",\n"
         << "  \"warmups\": 3,\n  \"samples_per_cell\": 20,\n"
         << "  \"cuda_runtime_version\": " << runtime_version << ",\n"
         << "  \"device_id\": " << device_id << ",\n"
         << "  \"cuda_header_version\": " << CUDART_VERSION << ",\n"
         << "  \"source_revision\": \"" << json_escape(source_revision) << "\",\n"
         << "  \"executable_sha256\": \"" << json_escape(executable_sha256) << "\",\n"
         << "  \"build_identity_sha256\": \"" << json_escape(build_identity_sha256) << "\",\n"
         << "  \"argv\": [";
  for (int index = 0; index < argc; ++index) {
    if (index != 0) output << ", ";
    output << "\"" << json_escape(argv[index]) << "\"";
  }
  output << "],\n  \"rows\": [\n";
  for (std::size_t row_index = 0u; row_index < rows.size(); ++row_index) {
    const BenchmarkRow& row = rows[row_index];
    if (row_index != 0u) output << ",\n";
    output << "    {\"batch\": " << row.batch_size << ", \"atoms\": " << row.atoms_per_system;
    const auto write_samples = [&](const char* name, const BenchmarkSamples& samples) {
      output << ", \"" << name << "\": {\"median_ms\": " << samples.median << ", \"samples_ms\": [";
      for (std::size_t sample = 0u; sample < samples.values.size(); ++sample) {
        if (sample != 0u) output << ", ";
        output << samples.values[sample];
      }
      output << "]}";
    };
    write_samples("sparse_build", row.sparse_build);
    write_samples("dense_build", row.dense_build);
    write_samples("reuse", row.reuse);
    output << "}";
  }
  output << "\n  ]\n}\n";
}

void write_samples_csv(std::ostream& output, const std::vector<BenchmarkRow>& rows) {
  output << "batch,atoms,sparse_build_ms,dense_build_ms,reuse_ms\n";
  output << std::setprecision(9);
  for (const BenchmarkRow& row : rows) {
    output << row.batch_size << ',' << row.atoms_per_system << ',' << row.sparse_build.median << ','
           << row.dense_build.median << ',' << row.reuse.median << '\n';
  }
}

int benchmark_build_vs_reuse(int argc, char** argv) {
  constexpr int kWarmup = 3;
  constexpr int kSamples = 20;
  constexpr double kBenchSpacing = 12.0;
  std::string json_path;
  std::string csv_path;
  std::string source_revision;
  std::string executable_sha256;
  std::string build_identity_sha256;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if ((argument == "--json" || argument == "--csv" || argument == "--source-revision" ||
         argument == "--executable-sha256" || argument == "--build-identity-sha256") &&
        index + 1 >= argc) {
      fprintf(stderr, "%s requires a value\n", argument.c_str());
      return 2;
    }
    if (argument == "--json") {
      json_path = argv[++index];
    } else if (argument == "--csv") {
      csv_path = argv[++index];
    } else if (argument == "--source-revision") {
      source_revision = argv[++index];
    } else if (argument == "--executable-sha256") {
      executable_sha256 = argv[++index];
    } else if (argument == "--build-identity-sha256") {
      build_identity_sha256 = argv[++index];
    } else {
      fprintf(stderr, "unknown benchmark option: %s\n", argument.c_str());
      return 2;
    }
  }
  if (!json_path.empty() &&
      (source_revision.empty() || executable_sha256.empty() || build_identity_sha256.empty())) {
    fprintf(stderr,
            "JSON evidence requires --source-revision, --executable-sha256, and "
            "--build-identity-sha256\n");
    return 2;
  }
  std::vector<BenchmarkRow> rows;
  rows.reserve(24u);
  for (const std::int64_t atoms_per_system : {16, 32, 48, 64, 96, 128}) {
    for (const std::size_t batch_size : {1u, 8u, 32u, 128u}) {
      const std::int64_t total_atoms = atoms_per_system * static_cast<std::int64_t>(batch_size);
      HostCase host;
      std::string error;
      host.atom_offsets.assign(batch_size + 1u, 0);
      for (std::size_t system = 0u; system < batch_size; ++system) {
        host.atom_offsets[system] = atoms_per_system * static_cast<std::int64_t>(system);
      }
      host.atom_offsets[batch_size] = total_atoms;
      host.atomic_numbers.resize(static_cast<std::size_t>(total_atoms));
      host.positions.resize(static_cast<std::size_t>(total_atoms * 3));
      for (std::int64_t atom = 0; atom < total_atoms; ++atom) {
        host.atomic_numbers[static_cast<std::size_t>(atom)] = 6;
        const std::int64_t local = atom % atoms_per_system;
        const std::int64_t side = static_cast<std::int64_t>(
            std::ceil(std::pow(static_cast<double>(atoms_per_system), 1.0 / 3.0)));
        host.positions[static_cast<std::size_t>(atom * 3)] =
            kBenchSpacing * static_cast<double>(local % side);
        host.positions[static_cast<std::size_t>(atom * 3 + 1)] =
            kBenchSpacing * static_cast<double>((local / side) % side);
        host.positions[static_cast<std::size_t>(atom * 3 + 2)] =
            kBenchSpacing * static_cast<double>(local / (side * side));
      }
      std::vector<std::int64_t> atom_offsets(host.atom_offsets.begin(), host.atom_offsets.end());
      CHECK(gpuxtb::detail::gfn2::make_coordination_plan(
                static_cast<std::int64_t>(batch_size), total_atoms, atom_offsets.data(),
                host.atomic_numbers.data(), host.plan, error) == GPUXTB_STATUS_SUCCESS);
      host.expected_coordination.resize(static_cast<std::size_t>(total_atoms), 0.0);

      cudaStream_t stream = nullptr;
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      DeviceFixture device;
      CUDA_CHECK(device.initialize(host, Gfn2PairListMode::kSparse, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      cudaEvent_t start = nullptr;
      cudaEvent_t stop = nullptr;
      CUDA_CHECK(cudaEventCreate(&start));
      CUDA_CHECK(cudaEventCreate(&stop));

      const auto reset_errors = [&]() {
        return gpuxtb::detail::cuda::reset_gfn2_pairlist_device_errors_cuda(
                   static_cast<std::int64_t>(batch_size), device.system_errors.get(),
                   device.device_error.get(), stream) == cudaSuccess;
      };

      const auto measure_build = [&](Gfn2PairListMode mode, BenchmarkSamples* result) -> int {
        std::vector<float> samples;
        for (int sample = -kWarmup; sample < kSamples; ++sample) {
          CHECK(reset_errors());
          CUDA_CHECK(cudaEventRecord(start, stream));
          CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
              device.batch(host, mode), device.positions.get(), kGeneration, device.cache(),
              device.workspace(), device.system_errors.get(), device.device_error.get(), stream));
          CUDA_CHECK(cudaEventRecord(stop, stream));
          CUDA_CHECK(cudaEventSynchronize(stop));
          float elapsed_ms = 0.0F;
          CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
          if (sample >= 0) {
            samples.push_back(elapsed_ms);
          }
        }
        std::vector<float> ordered = samples;
        std::sort(ordered.begin(), ordered.end());
        result->values = std::move(samples);
        result->median = ordered[kSamples / 2];
        return 0;
      };

      const auto measure_reuse = [&](BenchmarkSamples* result) -> int {
        std::vector<float> samples;
        for (int sample = -kWarmup; sample < kSamples; ++sample) {
          CHECK(reset_errors());
          CUDA_CHECK(cudaEventRecord(start, stream));
          CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_pairlist_coordination_cuda(
              device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(),
              device.radii.get(), kGeneration, device.cache(), device.coordination.get(),
              device.workspace(), device.system_errors.get(), device.device_error.get(), stream));
          CUDA_CHECK(cudaEventRecord(stop, stream));
          CUDA_CHECK(cudaEventSynchronize(stop));
          float elapsed_ms = 0.0F;
          CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
          if (sample >= 0) {
            samples.push_back(elapsed_ms);
          }
        }
        std::vector<float> ordered = samples;
        std::sort(ordered.begin(), ordered.end());
        result->values = std::move(samples);
        result->median = ordered[kSamples / 2];
        return 0;
      };

      BenchmarkRow row;
      row.batch_size = batch_size;
      row.atoms_per_system = atoms_per_system;
      /* First build once in sparse so the coordination reuse path is valid. */
      CHECK(reset_errors());
      CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
          device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), kGeneration,
          device.cache(), device.workspace(), device.system_errors.get(), device.device_error.get(),
          stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(measure_build(Gfn2PairListMode::kSparse, &row.sparse_build) == 0);
      CHECK(measure_build(Gfn2PairListMode::kDense, &row.dense_build) == 0);
      /* The dense timing overwrites the reusable cache.  Rebuild sparse once
       * outside the timed interval so reuse_ms measures the advertised sparse
       * list rather than an all-pairs cache. */
      CHECK(reset_errors());
      CUDA_CHECK(gpuxtb::detail::cuda::update_gfn2_pairlist_cache_cuda(
          device.batch(host, Gfn2PairListMode::kSparse), device.positions.get(), kGeneration,
          device.cache(), device.workspace(), device.system_errors.get(), device.device_error.get(),
          stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(measure_reuse(&row.reuse) == 0);
      rows.push_back(std::move(row));
      CUDA_CHECK(cudaEventDestroy(stop));
      CUDA_CHECK(cudaEventDestroy(start));
      CUDA_CHECK(cudaStreamDestroy(stream));
    }
  }
  if (json_path.empty() && csv_path.empty()) {
    write_samples_csv(std::cout, rows);
  } else {
    int runtime_version = 0;
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    if (!json_path.empty()) {
      std::ofstream output(json_path);
      if (!output) {
        fprintf(stderr, "failed to open JSON output: %s\n", json_path.c_str());
        return 2;
      }
      write_samples_json(output, rows, argc, argv, device_id, runtime_version, source_revision,
                         executable_sha256, build_identity_sha256);
    }
    if (!csv_path.empty()) {
      std::ofstream output(csv_path);
      if (!output) {
        fprintf(stderr, "failed to open CSV output: %s\n", csv_path.c_str());
        return 2;
      }
      write_samples_csv(output, rows);
    }
  }
  return 0;
}
}  // namespace

#ifdef GPUXTB_PAIRLIST_BENCHMARK_ONLY
int main(int argc, char** argv) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return 77;
  }
  return benchmark_build_vs_reuse(argc, argv);
}
#else

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return 77;
  }
  if (const int line = test_cpu_parity_and_pair_sets(); line != 0) {
    fprintf(stderr, "FAIL test_cpu_parity_and_pair_sets line %d\n", line);
    return line;
  }
  if (const int line = test_dense_fallback_matches_full_triangle(); line != 0) {
    fprintf(stderr, "FAIL test_dense_fallback_matches_full_triangle line %d\n", line);
    return line;
  }
  if (const int line = test_boundary_distances(); line != 0) {
    fprintf(stderr, "FAIL test_boundary_distances line %d\n", line);
    return line;
  }
  if (const int line = test_empty_and_single_atom(); line != 0) {
    fprintf(stderr, "FAIL test_empty_and_single_atom line %d\n", line);
    return line;
  }
  if (const int line = test_pair_capacity_overflow_isolated(); line != 0) {
    fprintf(stderr, "FAIL test_pair_capacity_overflow_isolated line %d\n", line);
    return line;
  }
  if (const int line = test_cell_capacity_dense_fallback(); line != 0) {
    fprintf(stderr, "FAIL test_cell_capacity_dense_fallback line %d\n", line);
    return line;
  }
  if (const int line = test_stale_generation_rejected(); line != 0) {
    fprintf(stderr, "FAIL test_stale_generation_rejected line %d\n", line);
    return line;
  }
  if (const int line = test_geometry_invalidation_rebuild(); line != 0) {
    fprintf(stderr, "FAIL test_geometry_invalidation_rebuild line %d\n", line);
    return line;
  }
  if (const int line = test_cuda_graph_capture_and_replay(); line != 0) {
    fprintf(stderr, "FAIL test_cuda_graph_capture_and_replay line %d\n", line);
    return line;
  }
  if (const int line = test_peer_failure_isolation(); line != 0) {
    fprintf(stderr, "FAIL test_peer_failure_isolation line %d\n", line);
    return line;
  }
  if (const int line = test_host_validation_rejects_hostile_views(); line != 0) {
    fprintf(stderr, "FAIL test_host_validation_rejects_hostile_views line %d\n", line);
    return line;
  }
  if (const int line = test_invalid_radius_is_peer_local(); line != 0) {
    fprintf(stderr, "FAIL test_invalid_radius_is_peer_local line %d\n", line);
    return line;
  }
  if (const int line = test_pair_level_preflight_is_peer_local(); line != 0) {
    fprintf(stderr, "FAIL test_pair_level_preflight_is_peer_local line %d\n", line);
    return line;
  }
  if (const int line = test_dispatch_policy(); line != 0) {
    fprintf(stderr, "FAIL test_dispatch_policy line %d\n", line);
    return line;
  }
  if (const int line = test_per_system_dispatch(); line != 0) {
    fprintf(stderr, "FAIL test_per_system_dispatch line %d\n", line);
    return line;
  }
  if (const int line = test_sparse_vjp_matches_dense(); line != 0) {
    fprintf(stderr, "FAIL test_sparse_vjp_matches_dense line %d\n", line);
    return line;
  }
  if (const int line = test_bitwise_dense_sparse_parity(); line != 0) {
    fprintf(stderr, "FAIL test_bitwise_dense_sparse_parity line %d\n", line);
    return line;
  }
  return 0;
}
#endif  // GPUXTB_PAIRLIST_BENCHMARK_ONLY
