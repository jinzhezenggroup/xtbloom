#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <system_error>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_eigensolver.cuh"

namespace {

using gpuxtb::detail::cuda::build_gfn2_compacted_eigensolver_graph_cuda;
using gpuxtb::detail::cuda::factor_gfn2_overlap_cuda;
using gpuxtb::detail::cuda::Gfn2EigensolverBucket;
using gpuxtb::detail::cuda::Gfn2EigensolverBucketActivity;
using gpuxtb::detail::cuda::Gfn2EigensolverCompactedSolveGraph;
using gpuxtb::detail::cuda::Gfn2EigensolverDeviceBatch;
using gpuxtb::detail::cuda::Gfn2EigensolverDeviceError;
using gpuxtb::detail::cuda::Gfn2EigensolverDeviceResults;
using gpuxtb::detail::cuda::Gfn2EigensolverDeviceWorkspace;
using gpuxtb::detail::cuda::Gfn2EigensolverLaunchStatus;
using gpuxtb::detail::cuda::Gfn2EigensolverOptions;
using gpuxtb::detail::cuda::Gfn2EigensolverOverlapCache;
using gpuxtb::detail::cuda::Gfn2EigensolverWorkspaceRequirements;
using gpuxtb::detail::cuda::Gfn2GeometryEpochDevice;
using gpuxtb::detail::cuda::query_gfn2_eigensolver_bucket_workspace_cuda;
using gpuxtb::detail::cuda::query_gfn2_spin_eigensolver_bucket_workspace_cuda;
using gpuxtb::detail::cuda::reset_gfn2_eigensolver_device_errors_cuda;
using gpuxtb::detail::cuda::solve_gfn2_eigensystems_cuda;
using gpuxtb::detail::cuda::solve_gfn2_spin_eigensystems_cuda;

constexpr double kSentinel = 91.25;

bool cuda_ok(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

bool solver_ok(cusolverStatus_t status, const char* operation) {
  if (status == CUSOLVER_STATUS_SUCCESS) {
    return true;
  }
  std::cerr << operation << ": cuSOLVER status " << static_cast<int>(status) << '\n';
  return false;
}

bool blas_ok(cublasStatus_t status, const char* operation) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return true;
  }
  std::cerr << operation << ": cuBLAS status " << static_cast<int>(status) << '\n';
  return false;
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { allocate(count); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept : pointer_(other.pointer_), count_(other.count_) {
    other.pointer_ = nullptr;
    other.count_ = 0u;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      pointer_ = other.pointer_;
      count_ = other.count_;
      other.pointer_ = nullptr;
      other.count_ = 0u;
    }
    return *this;
  }
  ~DeviceBuffer() { release(); }

  bool allocate(std::size_t count) {
    release();
    count_ = count;
    if (count == 0u) {
      return true;
    }
    const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T));
    if (status != cudaSuccess) {
      pointer_ = nullptr;
      count_ = 0u;
      std::cerr << "cudaMalloc: " << cudaGetErrorString(status) << '\n';
      return false;
    }
    return true;
  }

  bool upload(const std::vector<T>& host) {
    return host.size() <= count_ &&
           cuda_ok(
               cudaMemcpy(pointer_, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice),
               "cudaMemcpy host to device");
  }

  bool download(std::vector<T>& host) const {
    host.resize(count_);
    return cuda_ok(cudaMemcpy(host.data(), pointer_, count_ * sizeof(T), cudaMemcpyDeviceToHost),
                   "cudaMemcpy device to host");
  }

  T* get() const noexcept { return pointer_; }
  std::size_t size() const noexcept { return count_; }

 private:
  void release() noexcept {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
      pointer_ = nullptr;
    }
    count_ = 0u;
  }

  T* pointer_ = nullptr;
  std::size_t count_ = 0u;
};

class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  ~PinnedBuffer() {
    if (pointer_ != nullptr) {
      (void)cudaFreeHost(pointer_);
    }
  }

  bool allocate(std::size_t bytes) {
    bytes_ = bytes;
    if (bytes == 0u) {
      return true;
    }
    return cuda_ok(cudaMallocHost(&pointer_, bytes), "cudaMallocHost");
  }
  void* get() const noexcept { return pointer_; }
  std::size_t size() const noexcept { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0u;
};

struct ProviderHandles {
  cusolverDnHandle_t solver = nullptr;
  cusolverDnParams_t parameters = nullptr;
  cublasHandle_t blas = nullptr;
  cudaStream_t stream = nullptr;

  ProviderHandles() = default;
  ProviderHandles(const ProviderHandles&) = delete;
  ProviderHandles& operator=(const ProviderHandles&) = delete;
  ~ProviderHandles() {
    if (blas != nullptr) {
      (void)cublasDestroy(blas);
    }
    if (parameters != nullptr) {
      (void)cusolverDnDestroyParams(parameters);
    }
    if (solver != nullptr) {
      (void)cusolverDnDestroy(solver);
    }
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
    }
  }

  bool create() {
    return cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate") &&
           solver_ok(cusolverDnCreate(&solver), "cusolverDnCreate") &&
           solver_ok(cusolverDnCreateParams(&parameters), "cusolverDnCreateParams") &&
           blas_ok(cublasCreate(&blas), "cublasCreate");
  }
};

struct TestBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t total_matrices = 0;
  std::vector<std::int64_t> orbital_offsets;
  std::vector<std::int64_t> matrix_offsets;
  std::vector<std::int32_t> bucket_systems;
  std::vector<std::uint8_t> active;
  std::vector<Gfn2EigensolverBucket> buckets;
  std::vector<double> overlap;
  std::vector<double> hamiltonian;
};

/* Construct one requested AO size or an intentionally permuted n=2/n=3 pair. */
TestBatch make_batch(std::int64_t batch_size, bool mixed_buckets, std::int64_t ill_system = -1,
                     std::int32_t homogeneous_dimension = 3) {
  TestBatch data;
  data.batch_size = batch_size;
  data.orbital_offsets.assign(static_cast<std::size_t>(batch_size + 1), 0);
  data.matrix_offsets.assign(static_cast<std::size_t>(batch_size + 1), 0);
  data.active.assign(static_cast<std::size_t>(batch_size), 1u);
  std::vector<std::int32_t> dimensions(static_cast<std::size_t>(batch_size), homogeneous_dimension);
  if (mixed_buckets) {
    for (std::int64_t system = 0; system < batch_size; ++system) {
      dimensions[static_cast<std::size_t>(system)] = system % 2 == 0 ? 2 : 3;
    }
  }
  for (std::int64_t system = 0; system < batch_size; ++system) {
    const std::int64_t n = dimensions[static_cast<std::size_t>(system)];
    data.total_orbitals += n;
    data.total_matrices += n * n;
    data.orbital_offsets[static_cast<std::size_t>(system + 1)] = data.total_orbitals;
    data.matrix_offsets[static_cast<std::size_t>(system + 1)] = data.total_matrices;
  }
  data.overlap.resize(static_cast<std::size_t>(data.total_matrices), 0.0);
  data.hamiltonian.resize(static_cast<std::size_t>(data.total_matrices), 0.0);
  for (std::int64_t system = 0; system < batch_size; ++system) {
    const std::int64_t n = dimensions[static_cast<std::size_t>(system)];
    const std::int64_t begin = data.matrix_offsets[static_cast<std::size_t>(system)];
    for (std::int64_t row = 0; row < n; ++row) {
      for (std::int64_t column = 0; column < n; ++column) {
        const double distance = static_cast<double>(std::abs(row - column) + 1);
        double s = row == column
                       ? 1.05 + 0.05 * static_cast<double>(row + 1) / static_cast<double>(n)
                       : 0.01 / distance;
        if (system == ill_system) {
          s = row == column ? (row + 1 == n ? 1.0e-16 : 1.0) : 0.0;
        }
        const double h = row == column ? -0.8 + 0.035 * static_cast<double>(row) +
                                             0.001 * static_cast<double>(system)
                                       : 0.02 / distance;
        data.overlap[static_cast<std::size_t>(begin + row * n + column)] = s;
        data.hamiltonian[static_cast<std::size_t>(begin + row * n + column)] = h;
      }
    }
  }

  std::int64_t system_offset = 0;
  std::int64_t matrix_offset = 0;
  std::int64_t orbital_offset = 0;
  std::vector<std::int32_t> bucket_dimensions = dimensions;
  std::sort(bucket_dimensions.begin(), bucket_dimensions.end());
  bucket_dimensions.erase(std::unique(bucket_dimensions.begin(), bucket_dimensions.end()),
                          bucket_dimensions.end());
  for (const std::int32_t n : bucket_dimensions) {
    std::vector<std::int32_t> systems;
    for (std::int64_t system = 0; system < batch_size; ++system) {
      if (dimensions[static_cast<std::size_t>(system)] == n) {
        systems.push_back(static_cast<std::int32_t>(system));
      }
    }
    if (systems.empty()) {
      continue;
    }
    data.buckets.push_back({n, static_cast<std::int32_t>(systems.size()), system_offset,
                            matrix_offset, orbital_offset});
    data.bucket_systems.insert(data.bucket_systems.end(), systems.begin(), systems.end());
    system_offset += static_cast<std::int64_t>(systems.size());
    matrix_offset += static_cast<std::int64_t>(n) * n * static_cast<std::int64_t>(systems.size());
    orbital_offset += static_cast<std::int64_t>(n) * static_cast<std::int64_t>(systems.size());
  }
  return data;
}

bool near(double first, double second, double tolerance) {
  return std::abs(first - second) <= tolerance * std::max({1.0, std::abs(first), std::abs(second)});
}

bool validate_system(const TestBatch& batch, std::int64_t system,
                     const std::vector<double>& eigenvalues,
                     const std::vector<double>& coefficients) {
  const std::int64_t orbital_begin = batch.orbital_offsets[static_cast<std::size_t>(system)];
  const std::int64_t n =
      batch.orbital_offsets[static_cast<std::size_t>(system + 1)] - orbital_begin;
  const std::int64_t matrix_begin = batch.matrix_offsets[static_cast<std::size_t>(system)];
  for (std::int64_t orbital = 0; orbital < n; ++orbital) {
    for (std::int64_t row = 0; row < n; ++row) {
      double hc = 0.0;
      double sc = 0.0;
      for (std::int64_t column = 0; column < n; ++column) {
        const double coefficient =
            coefficients[static_cast<std::size_t>(matrix_begin + column * n + orbital)];
        hc += batch.hamiltonian[static_cast<std::size_t>(matrix_begin + row * n + column)] *
              coefficient;
        sc +=
            batch.overlap[static_cast<std::size_t>(matrix_begin + row * n + column)] * coefficient;
      }
      if (!near(hc, sc * eigenvalues[static_cast<std::size_t>(orbital_begin + orbital)], 3.0e-10)) {
        std::cerr << "generalized residual failed for system " << system << '\n';
        return false;
      }
    }
    for (std::int64_t other = 0; other < n; ++other) {
      double metric = 0.0;
      for (std::int64_t row = 0; row < n; ++row) {
        for (std::int64_t column = 0; column < n; ++column) {
          metric += coefficients[static_cast<std::size_t>(matrix_begin + row * n + orbital)] *
                    batch.overlap[static_cast<std::size_t>(matrix_begin + row * n + column)] *
                    coefficients[static_cast<std::size_t>(matrix_begin + column * n + other)];
        }
      }
      if (!near(metric, orbital == other ? 1.0 : 0.0, 3.0e-10)) {
        std::cerr << "S orthonormality failed for system " << system << '\n';
        return false;
      }
    }
  }
  return true;
}

bool system_outputs_equal(const TestBatch& batch, std::int64_t system,
                          const std::vector<double>& eigenvalues,
                          const std::vector<double>& coefficients, double expected) {
  const std::int64_t orbital_begin = batch.orbital_offsets[static_cast<std::size_t>(system)];
  const std::int64_t orbital_end = batch.orbital_offsets[static_cast<std::size_t>(system + 1)];
  const std::int64_t matrix_begin = batch.matrix_offsets[static_cast<std::size_t>(system)];
  const std::int64_t matrix_end = batch.matrix_offsets[static_cast<std::size_t>(system + 1)];
  return std::all_of(eigenvalues.begin() + orbital_begin, eigenvalues.begin() + orbital_end,
                     [expected](double value) { return value == expected; }) &&
         std::all_of(coefficients.begin() + matrix_begin, coefficients.begin() + matrix_end,
                     [expected](double value) { return value == expected; });
}

struct DeviceFixture {
  TestBatch host;
  ProviderHandles providers;
  DeviceBuffer<std::int64_t> orbital_offsets;
  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<std::int32_t> bucket_systems;
  DeviceBuffer<std::uint8_t> active;
  DeviceBuffer<double> overlap;
  DeviceBuffer<double> hamiltonian;
  DeviceBuffer<double> matrix_a;
  DeviceBuffer<double> matrix_b;
  DeviceBuffer<double> eigen_scratch;
  DeviceBuffer<double*> factor_pointers;
  DeviceBuffer<double*> matrix_pointers;
  DeviceBuffer<int> info_a;
  DeviceBuffer<int> info_b;
  DeviceBuffer<std::uint8_t> eligible;
  DeviceBuffer<std::uint32_t> sequence_active;
  DeviceBuffer<std::int32_t> compact_systems;
  DeviceBuffer<std::int32_t> compact_source_slots;
  DeviceBuffer<Gfn2EigensolverBucketActivity> bucket_activity;
  DeviceBuffer<double> cache_factors;
  DeviceBuffer<std::uint64_t> cache_generations;
  DeviceBuffer<std::uint32_t> cache_statuses;
  DeviceBuffer<double> eigenvalues;
  DeviceBuffer<double> coefficients;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;
  DeviceBuffer<std::byte> solver_device_workspace;
  PinnedBuffer solver_host_workspace;
  Gfn2EigensolverDeviceBatch batch;
  Gfn2EigensolverDeviceWorkspace workspace;
  Gfn2EigensolverOverlapCache cache;
  Gfn2EigensolverDeviceResults results;

  bool create(TestBatch input) {
    host = std::move(input);
    if (!providers.create() || !orbital_offsets.allocate(host.orbital_offsets.size()) ||
        !matrix_offsets.allocate(host.matrix_offsets.size()) ||
        !bucket_systems.allocate(host.bucket_systems.size()) ||
        !active.allocate(host.active.size()) || !overlap.allocate(host.overlap.size()) ||
        !hamiltonian.allocate(host.hamiltonian.size()) ||
        !matrix_a.allocate(static_cast<std::size_t>(host.total_matrices)) ||
        !matrix_b.allocate(static_cast<std::size_t>(host.total_matrices)) ||
        !eigen_scratch.allocate(static_cast<std::size_t>(host.total_orbitals)) ||
        !factor_pointers.allocate(static_cast<std::size_t>(host.batch_size)) ||
        !matrix_pointers.allocate(static_cast<std::size_t>(host.batch_size)) ||
        !info_a.allocate(static_cast<std::size_t>(host.batch_size)) ||
        !info_b.allocate(static_cast<std::size_t>(host.batch_size)) ||
        !eligible.allocate(static_cast<std::size_t>(host.batch_size)) ||
        !sequence_active.allocate(1u) ||
        !compact_systems.allocate(static_cast<std::size_t>(host.batch_size)) ||
        !compact_source_slots.allocate(static_cast<std::size_t>(host.batch_size)) ||
        !bucket_activity.allocate(host.buckets.size()) ||
        !cache_factors.allocate(static_cast<std::size_t>(host.total_matrices)) ||
        !cache_generations.allocate(static_cast<std::size_t>(host.batch_size)) ||
        !cache_statuses.allocate(static_cast<std::size_t>(host.batch_size)) ||
        !eigenvalues.allocate(static_cast<std::size_t>(host.total_orbitals)) ||
        !coefficients.allocate(static_cast<std::size_t>(host.total_matrices)) ||
        !system_errors.allocate(static_cast<std::size_t>(host.batch_size)) ||
        !device_error.allocate(1u) || !orbital_offsets.upload(host.orbital_offsets) ||
        !matrix_offsets.upload(host.matrix_offsets) ||
        !bucket_systems.upload(host.bucket_systems) || !active.upload(host.active) ||
        !overlap.upload(host.overlap) || !hamiltonian.upload(host.hamiltonian)) {
      return false;
    }
    batch = {host.batch_size,
             host.total_orbitals,
             host.total_matrices,
             static_cast<std::int64_t>(host.orbital_offsets.size()),
             static_cast<std::int64_t>(host.matrix_offsets.size()),
             static_cast<std::int64_t>(host.bucket_systems.size()),
             static_cast<std::int64_t>(host.active.size()),
             0xE19E72u,
             orbital_offsets.get(),
             matrix_offsets.get(),
             bucket_systems.get(),
             active.get()};

    Gfn2EigensolverWorkspaceRequirements requirements;
    for (const Gfn2EigensolverBucket& bucket : host.buckets) {
      const auto query = query_gfn2_eigensolver_bucket_workspace_cuda(
          providers.solver, providers.parameters, bucket, matrix_b.get(), eigen_scratch.get(),
          requirements);
      if (!query.success()) {
        std::cerr << "workspace query failed: " << static_cast<unsigned int>(query.status) << '\n';
        return false;
      }
    }
    const std::size_t device_elements =
        (requirements.solver_device_workspace_bytes + sizeof(std::byte) - 1u) / sizeof(std::byte);
    if (!solver_device_workspace.allocate(device_elements) ||
        !solver_host_workspace.allocate(requirements.solver_host_workspace_bytes)) {
      return false;
    }
    workspace = {matrix_a.get(),
                 static_cast<std::int64_t>(matrix_a.size()),
                 matrix_b.get(),
                 static_cast<std::int64_t>(matrix_b.size()),
                 eigen_scratch.get(),
                 static_cast<std::int64_t>(eigen_scratch.size()),
                 factor_pointers.get(),
                 static_cast<std::int64_t>(factor_pointers.size()),
                 matrix_pointers.get(),
                 static_cast<std::int64_t>(matrix_pointers.size()),
                 info_a.get(),
                 static_cast<std::int64_t>(info_a.size()),
                 info_b.get(),
                 static_cast<std::int64_t>(info_b.size()),
                 eligible.get(),
                 static_cast<std::int64_t>(eligible.size()),
                 sequence_active.get(),
                 static_cast<std::int64_t>(sequence_active.size()),
                 solver_device_workspace.get(),
                 requirements.solver_device_workspace_bytes,
                 solver_host_workspace.get(),
                 solver_host_workspace.size(),
                 batch.plan_token};
    workspace.compact_systems = compact_systems.get();
    workspace.compact_system_elements = static_cast<std::int64_t>(compact_systems.size());
    workspace.compact_source_slots = compact_source_slots.get();
    workspace.compact_source_slot_elements = static_cast<std::int64_t>(compact_source_slots.size());
    workspace.bucket_activity = bucket_activity.get();
    workspace.bucket_activity_elements = static_cast<std::int64_t>(bucket_activity.size());
    cache = {cache_factors.get(),     static_cast<std::int64_t>(cache_factors.size()),
             cache_generations.get(), static_cast<std::int64_t>(cache_generations.size()),
             cache_statuses.get(),    static_cast<std::int64_t>(cache_statuses.size()),
             batch.plan_token};
    results = {eigenvalues.get(), static_cast<std::int64_t>(eigenvalues.size()), coefficients.get(),
               static_cast<std::int64_t>(coefficients.size()), batch.plan_token};
    return cuda_ok(
               cudaMemsetAsync(cache_generations.get(), 0,
                               cache_generations.size() * sizeof(std::uint64_t), providers.stream),
               "clear cache generations") &&
           cuda_ok(cudaMemsetAsync(cache_statuses.get(), 0xFF,
                                   cache_statuses.size() * sizeof(std::uint32_t), providers.stream),
                   "clear cache statuses");
  }

  bool fill_outputs(double value) {
    std::vector<double> eigen(static_cast<std::size_t>(host.total_orbitals), value);
    std::vector<double> coeff(static_cast<std::size_t>(host.total_matrices), value);
    return eigenvalues.upload(eigen) && coefficients.upload(coeff);
  }
};

bool factor(DeviceFixture& fixture, std::uint64_t generation) {
  if (!cuda_ok(reset_gfn2_eigensolver_device_errors_cuda(
                   fixture.host.batch_size, fixture.system_errors.get(), fixture.device_error.get(),
                   fixture.providers.stream),
               "reset factor errors")) {
    return false;
  }
  const auto launch = factor_gfn2_overlap_cuda(
      fixture.batch, fixture.host.buckets.data(),
      static_cast<std::int64_t>(fixture.host.buckets.size()), fixture.overlap.get(), generation,
      Gfn2EigensolverOptions{}, fixture.providers.solver, fixture.providers.parameters,
      fixture.workspace, fixture.cache, fixture.system_errors.get(), fixture.device_error.get(),
      fixture.providers.stream);
  if (!launch.success()) {
    std::cerr << "factor launch failed: " << static_cast<unsigned int>(launch.status) << '\n';
    return false;
  }
  return cuda_ok(cudaStreamSynchronize(fixture.providers.stream), "factor synchronize");
}

bool solve(DeviceFixture& fixture, std::uint64_t generation, bool reset_errors = true) {
  if (reset_errors && !cuda_ok(reset_gfn2_eigensolver_device_errors_cuda(
                                   fixture.host.batch_size, fixture.system_errors.get(),
                                   fixture.device_error.get(), fixture.providers.stream),
                               "reset solve errors")) {
    return false;
  }
  const auto launch = solve_gfn2_eigensystems_cuda(
      fixture.batch, fixture.host.buckets.data(),
      static_cast<std::int64_t>(fixture.host.buckets.size()), fixture.cache, generation,
      fixture.hamiltonian.get(), Gfn2EigensolverOptions{}, fixture.providers.solver,
      fixture.providers.parameters, fixture.providers.blas, fixture.workspace, fixture.results,
      fixture.system_errors.get(), fixture.device_error.get(), fixture.providers.stream);
  if (!launch.success()) {
    std::cerr << "solve launch failed: " << static_cast<unsigned int>(launch.status) << '\n';
    return false;
  }
  return cuda_ok(cudaStreamSynchronize(fixture.providers.stream), "solve synchronize");
}

struct SpinSolveFixture {
  std::vector<std::int32_t> spin_channels;
  std::vector<std::int64_t> spin_channel_offsets;
  std::vector<std::int64_t> spin_orbital_offsets;
  std::vector<std::int64_t> spin_matrix_offsets;
  std::vector<Gfn2EigensolverBucket> buckets;
  std::vector<double> hamiltonian;
  DeviceBuffer<std::int32_t> device_spin_channels;
  DeviceBuffer<std::int64_t> device_spin_channel_offsets;
  DeviceBuffer<std::int64_t> device_spin_orbital_offsets;
  DeviceBuffer<std::int64_t> device_spin_matrix_offsets;
  DeviceBuffer<double> device_hamiltonian;
  DeviceBuffer<double> matrix_a;
  DeviceBuffer<double> matrix_b;
  DeviceBuffer<double> eigen_scratch;
  DeviceBuffer<double*> factor_pointers;
  DeviceBuffer<double*> matrix_pointers;
  DeviceBuffer<int> info_a;
  DeviceBuffer<int> info_b;
  DeviceBuffer<std::uint8_t> eligible;
  DeviceBuffer<std::uint32_t> sequence_active;
  DeviceBuffer<std::byte> solver_device_workspace;
  PinnedBuffer solver_host_workspace;
  DeviceBuffer<double> eigenvalues;
  DeviceBuffer<double> coefficients;
  gpuxtb::detail::Gfn2WavefunctionLayoutView layout{};
  Gfn2EigensolverDeviceWorkspace workspace{};
  Gfn2EigensolverDeviceResults results{};

  bool create(DeviceFixture& physical) {
    const TestBatch& host = physical.host;
    spin_channels.resize(static_cast<std::size_t>(host.batch_size));
    spin_channel_offsets.assign(static_cast<std::size_t>(host.batch_size + 1), 0);
    spin_orbital_offsets.assign(static_cast<std::size_t>(host.batch_size + 1), 0);
    spin_matrix_offsets.assign(static_cast<std::size_t>(host.batch_size + 1), 0);
    for (std::int64_t system = 0; system < host.batch_size; ++system) {
      const std::int32_t channels = host.batch_size == 1 || system % 3 != 0 ? 2 : 1;
      const std::int64_t orbitals = host.orbital_offsets[static_cast<std::size_t>(system + 1)] -
                                    host.orbital_offsets[static_cast<std::size_t>(system)];
      const std::int64_t matrices = host.matrix_offsets[static_cast<std::size_t>(system + 1)] -
                                    host.matrix_offsets[static_cast<std::size_t>(system)];
      spin_channels[static_cast<std::size_t>(system)] = channels;
      spin_channel_offsets[static_cast<std::size_t>(system + 1)] =
          spin_channel_offsets[static_cast<std::size_t>(system)] + channels;
      spin_orbital_offsets[static_cast<std::size_t>(system + 1)] =
          spin_orbital_offsets[static_cast<std::size_t>(system)] + channels * orbitals;
      spin_matrix_offsets[static_cast<std::size_t>(system + 1)] =
          spin_matrix_offsets[static_cast<std::size_t>(system)] + channels * matrices;
    }
    hamiltonian.assign(static_cast<std::size_t>(spin_matrix_offsets.back()), 0.0);
    for (std::int64_t system = 0; system < host.batch_size; ++system) {
      const std::int64_t n = host.orbital_offsets[static_cast<std::size_t>(system + 1)] -
                             host.orbital_offsets[static_cast<std::size_t>(system)];
      const std::int64_t matrix_stride = n * n;
      const std::int64_t source = host.matrix_offsets[static_cast<std::size_t>(system)];
      const std::int64_t destination = spin_matrix_offsets[static_cast<std::size_t>(system)];
      for (std::int32_t spin = 0; spin < spin_channels[static_cast<std::size_t>(system)]; ++spin) {
        for (std::int64_t index = 0; index < matrix_stride; ++index) {
          const std::int64_t row = index / n;
          const std::int64_t column = index - row * n;
          double value = host.hamiltonian[static_cast<std::size_t>(source + index)];
          if (row == column) {
            value += 0.025 * static_cast<double>(spin);
          }
          hamiltonian[static_cast<std::size_t>(destination + spin * matrix_stride + index)] = value;
        }
      }
    }

    buckets = host.buckets;
    std::int64_t solve_offset = 0;
    std::int64_t matrix_scratch_offset = 0;
    std::int64_t orbital_scratch_offset = 0;
    for (Gfn2EigensolverBucket& bucket : buckets) {
      std::int32_t solve_count = 0;
      for (std::int32_t local = 0; local < bucket.system_count; ++local) {
        const std::int32_t system = host.bucket_systems[static_cast<std::size_t>(
            bucket.system_index_offset + static_cast<std::int64_t>(local))];
        solve_count += spin_channels[static_cast<std::size_t>(system)];
      }
      bucket.solve_count = solve_count;
      bucket.solve_index_offset = solve_offset;
      bucket.spin_matrix_scratch_offset = matrix_scratch_offset;
      bucket.spin_orbital_scratch_offset = orbital_scratch_offset;
      solve_offset += solve_count;
      matrix_scratch_offset +=
          static_cast<std::int64_t>(solve_count) * bucket.orbital_count * bucket.orbital_count;
      orbital_scratch_offset += static_cast<std::int64_t>(solve_count) * bucket.orbital_count;
    }
    if (solve_offset != spin_channel_offsets.back() ||
        matrix_scratch_offset != spin_matrix_offsets.back() ||
        orbital_scratch_offset != spin_orbital_offsets.back() ||
        !device_spin_channels.allocate(spin_channels.size()) ||
        !device_spin_channel_offsets.allocate(spin_channel_offsets.size()) ||
        !device_spin_orbital_offsets.allocate(spin_orbital_offsets.size()) ||
        !device_spin_matrix_offsets.allocate(spin_matrix_offsets.size()) ||
        !device_hamiltonian.allocate(hamiltonian.size()) ||
        !matrix_a.allocate(static_cast<std::size_t>(matrix_scratch_offset)) ||
        !matrix_b.allocate(static_cast<std::size_t>(matrix_scratch_offset)) ||
        !eigen_scratch.allocate(static_cast<std::size_t>(orbital_scratch_offset)) ||
        !factor_pointers.allocate(static_cast<std::size_t>(solve_offset)) ||
        !matrix_pointers.allocate(static_cast<std::size_t>(solve_offset)) ||
        !info_a.allocate(static_cast<std::size_t>(solve_offset)) ||
        !info_b.allocate(static_cast<std::size_t>(solve_offset)) ||
        !eligible.allocate(static_cast<std::size_t>(solve_offset)) ||
        !sequence_active.allocate(1u) ||
        !eigenvalues.allocate(static_cast<std::size_t>(spin_orbital_offsets.back())) ||
        !coefficients.allocate(static_cast<std::size_t>(spin_matrix_offsets.back())) ||
        !device_spin_channels.upload(spin_channels) ||
        !device_spin_channel_offsets.upload(spin_channel_offsets) ||
        !device_spin_orbital_offsets.upload(spin_orbital_offsets) ||
        !device_spin_matrix_offsets.upload(spin_matrix_offsets) ||
        !device_hamiltonian.upload(hamiltonian)) {
      return false;
    }

    Gfn2EigensolverWorkspaceRequirements requirements{};
    for (const Gfn2EigensolverBucket& bucket : buckets) {
      const auto query = query_gfn2_spin_eigensolver_bucket_workspace_cuda(
          physical.providers.solver, physical.providers.parameters, bucket, matrix_b.get(),
          eigen_scratch.get(), requirements);
      if (!query.success()) {
        return false;
      }
    }
    if (!solver_device_workspace.allocate(requirements.solver_device_workspace_bytes) ||
        !solver_host_workspace.allocate(requirements.solver_host_workspace_bytes)) {
      return false;
    }
    layout.memory_space = gpuxtb::detail::Gfn2PlanMemorySpace::kCudaDevice;
    layout.plan_token = physical.batch.plan_token;
    layout.batch_size = host.batch_size;
    layout.total_spin_channels = spin_channel_offsets.back();
    layout.total_spin_orbitals = spin_orbital_offsets.back();
    layout.total_spin_matrix_elements = spin_matrix_offsets.back();
    layout.spin_channel_count = host.batch_size;
    layout.spin_channel_offset_count = host.batch_size + 1;
    layout.spin_orbital_offset_count = host.batch_size + 1;
    layout.spin_matrix_offset_count = host.batch_size + 1;
    layout.spin_channels = device_spin_channels.get();
    layout.spin_channel_offsets = device_spin_channel_offsets.get();
    layout.spin_orbital_offsets = device_spin_orbital_offsets.get();
    layout.spin_matrix_offsets = device_spin_matrix_offsets.get();
    workspace = {matrix_a.get(),
                 static_cast<std::int64_t>(matrix_a.size()),
                 matrix_b.get(),
                 static_cast<std::int64_t>(matrix_b.size()),
                 eigen_scratch.get(),
                 static_cast<std::int64_t>(eigen_scratch.size()),
                 factor_pointers.get(),
                 static_cast<std::int64_t>(factor_pointers.size()),
                 matrix_pointers.get(),
                 static_cast<std::int64_t>(matrix_pointers.size()),
                 info_a.get(),
                 static_cast<std::int64_t>(info_a.size()),
                 info_b.get(),
                 static_cast<std::int64_t>(info_b.size()),
                 eligible.get(),
                 static_cast<std::int64_t>(eligible.size()),
                 sequence_active.get(),
                 static_cast<std::int64_t>(sequence_active.size()),
                 solver_device_workspace.get(),
                 requirements.solver_device_workspace_bytes,
                 solver_host_workspace.get(),
                 solver_host_workspace.size(),
                 physical.batch.plan_token};
    results = {eigenvalues.get(), static_cast<std::int64_t>(eigenvalues.size()), coefficients.get(),
               static_cast<std::int64_t>(coefficients.size()), physical.batch.plan_token};
    return fill_outputs(kSentinel);
  }

  bool fill_outputs(double value) {
    return eigenvalues.upload(std::vector<double>(eigenvalues.size(), value)) &&
           coefficients.upload(std::vector<double>(coefficients.size(), value));
  }
};

bool validate_spin_system(const DeviceFixture& physical, const SpinSolveFixture& spin,
                          std::int64_t system, std::int32_t channel,
                          const std::vector<double>& eigenvalues,
                          const std::vector<double>& coefficients) {
  const TestBatch& batch = physical.host;
  const std::int64_t n = batch.orbital_offsets[static_cast<std::size_t>(system + 1)] -
                         batch.orbital_offsets[static_cast<std::size_t>(system)];
  const std::int64_t physical_matrix = batch.matrix_offsets[static_cast<std::size_t>(system)];
  const std::int64_t spin_matrix = spin.spin_matrix_offsets[static_cast<std::size_t>(system)] +
                                   static_cast<std::int64_t>(channel) * n * n;
  const std::int64_t spin_orbital = spin.spin_orbital_offsets[static_cast<std::size_t>(system)] +
                                    static_cast<std::int64_t>(channel) * n;
  for (std::int64_t orbital = 0; orbital < n; ++orbital) {
    for (std::int64_t row = 0; row < n; ++row) {
      double hc = 0.0;
      double sc = 0.0;
      for (std::int64_t column = 0; column < n; ++column) {
        const double coefficient =
            coefficients[static_cast<std::size_t>(spin_matrix + column * n + orbital)];
        hc += spin.hamiltonian[static_cast<std::size_t>(spin_matrix + row * n + column)] *
              coefficient;
        sc += batch.overlap[static_cast<std::size_t>(physical_matrix + row * n + column)] *
              coefficient;
      }
      if (!near(hc, sc * eigenvalues[static_cast<std::size_t>(spin_orbital + orbital)], 3.0e-10)) {
        return false;
      }
    }
  }
  return true;
}

bool solve_spin(DeviceFixture& physical, SpinSolveFixture& spin, std::uint64_t generation) {
  if (!cuda_ok(reset_gfn2_eigensolver_device_errors_cuda(
                   physical.host.batch_size, physical.system_errors.get(),
                   physical.device_error.get(), physical.providers.stream),
               "reset spin solve errors")) {
    return false;
  }
  const auto launch = solve_gfn2_spin_eigensystems_cuda(
      physical.batch, spin.layout, spin.buckets.data(),
      static_cast<std::int64_t>(spin.buckets.size()), physical.cache, generation,
      spin.device_hamiltonian.get(), Gfn2EigensolverOptions{}, physical.providers.solver,
      physical.providers.parameters, physical.providers.blas, spin.workspace, spin.results,
      physical.system_errors.get(), physical.device_error.get(), physical.providers.stream);
  return launch.success() &&
         cuda_ok(cudaStreamSynchronize(physical.providers.stream), "spin solve synchronize");
}

bool test_spin_eigensolver_mixed_batches_and_transaction() {
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    DeviceFixture physical;
    if (!physical.create(make_batch(batch_size, true)) || !factor(physical, 53u)) {
      return false;
    }
    SpinSolveFixture spin;
    if (!spin.create(physical) || !solve_spin(physical, spin, 53u)) {
      return false;
    }
    std::vector<std::uint32_t> errors;
    std::vector<double> eigenvalues;
    std::vector<double> coefficients;
    if (!physical.system_errors.download(errors) || !spin.eigenvalues.download(eigenvalues) ||
        !spin.coefficients.download(coefficients)) {
      return false;
    }
    for (std::int64_t system = 0; system < batch_size; ++system) {
      if (errors[static_cast<std::size_t>(system)] != 0u) {
        return false;
      }
      for (std::int32_t channel = 0; channel < spin.spin_channels[static_cast<std::size_t>(system)];
           ++channel) {
        if (!validate_spin_system(physical, spin, system, channel, eigenvalues, coefficients)) {
          return false;
        }
      }
    }

    if (batch_size == 8) {
      constexpr std::int64_t failed_system = 1;
      if (spin.spin_channels[failed_system] != 2 || !spin.fill_outputs(kSentinel)) {
        return false;
      }
      std::vector<double> poisoned = spin.hamiltonian;
      const std::int64_t n = physical.host.orbital_offsets[failed_system + 1] -
                             physical.host.orbital_offsets[failed_system];
      const std::int64_t beta = spin.spin_matrix_offsets[failed_system] + n * n;
      poisoned[static_cast<std::size_t>(beta)] = std::numeric_limits<double>::quiet_NaN();
      if (!spin.device_hamiltonian.upload(poisoned) || !solve_spin(physical, spin, 53u) ||
          !physical.system_errors.download(errors) || !spin.eigenvalues.download(eigenvalues) ||
          !spin.coefficients.download(coefficients)) {
        return false;
      }
      if (errors[failed_system] !=
          static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kNonfiniteHamiltonian)) {
        return false;
      }
      const std::int64_t orbital_begin = spin.spin_orbital_offsets[failed_system];
      const std::int64_t orbital_end = spin.spin_orbital_offsets[failed_system + 1];
      const std::int64_t matrix_begin = spin.spin_matrix_offsets[failed_system];
      const std::int64_t matrix_end = spin.spin_matrix_offsets[failed_system + 1];
      if (!std::all_of(eigenvalues.begin() + orbital_begin, eigenvalues.begin() + orbital_end,
                       [](double value) { return value == kSentinel; }) ||
          !std::all_of(coefficients.begin() + matrix_begin, coefficients.begin() + matrix_end,
                       [](double value) { return value == kSentinel; })) {
        return false;
      }
      if (errors[2] != 0u ||
          !validate_spin_system(physical, spin, 2, 0, eigenvalues, coefficients)) {
        return false;
      }
    }
  }
  return true;
}

bool launch_compacted(DeviceFixture& fixture, const Gfn2EigensolverCompactedSolveGraph& graph) {
  if (!cuda_ok(reset_gfn2_eigensolver_device_errors_cuda(
                   fixture.host.batch_size, fixture.system_errors.get(), fixture.device_error.get(),
                   fixture.providers.stream),
               "reset compacted solve errors")) {
    return false;
  }
  const auto launch = graph.launch(fixture.providers.stream);
  if (!launch.success()) {
    std::cerr << "compacted graph launch failed: " << static_cast<unsigned int>(launch.status)
              << " cuda=" << cudaGetErrorString(launch.cuda_status) << '\n';
    return false;
  }
  return cuda_ok(cudaStreamSynchronize(fixture.providers.stream), "compacted graph synchronize");
}

bool enqueue_compacted(DeviceFixture& fixture, const Gfn2EigensolverCompactedSolveGraph& graph) {
  const auto launch = graph.launch(fixture.providers.stream);
  if (launch.success()) {
    return true;
  }
  std::cerr << "compacted graph enqueue failed: " << static_cast<unsigned int>(launch.status)
            << " cuda=" << cudaGetErrorString(launch.cuda_status) << '\n';
  return false;
}

bool enqueue_uncompacted(DeviceFixture& fixture, std::uint64_t generation) {
  const auto launch = solve_gfn2_eigensystems_cuda(
      fixture.batch, fixture.host.buckets.data(),
      static_cast<std::int64_t>(fixture.host.buckets.size()), fixture.cache, generation,
      fixture.hamiltonian.get(), Gfn2EigensolverOptions{}, fixture.providers.solver,
      fixture.providers.parameters, fixture.providers.blas, fixture.workspace, fixture.results,
      fixture.system_errors.get(), fixture.device_error.get(), fixture.providers.stream);
  if (launch.success()) {
    return true;
  }
  std::cerr << "uncompacted solve enqueue failed: " << static_cast<unsigned int>(launch.status)
            << " cuda=" << cudaGetErrorString(launch.cuda_status) << '\n';
  return false;
}

bool validate_compacted_launch(DeviceFixture& fixture) {
  std::vector<std::uint32_t> errors;
  std::vector<std::uint32_t> device_error;
  std::vector<double> eigenvalues;
  std::vector<double> coefficients;
  std::vector<Gfn2EigensolverBucketActivity> activity;
  std::vector<std::int32_t> compact_systems;
  if (!fixture.system_errors.download(errors) || !fixture.device_error.download(device_error) ||
      device_error.size() != 1u || device_error[0] != 0u ||
      !fixture.eigenvalues.download(eigenvalues) || !fixture.coefficients.download(coefficients) ||
      !fixture.bucket_activity.download(activity) ||
      !fixture.compact_systems.download(compact_systems)) {
    return false;
  }
  for (std::int64_t system = 0; system < fixture.host.batch_size; ++system) {
    const bool active = fixture.host.active[static_cast<std::size_t>(system)] == 1u;
    if (active) {
      if (errors[static_cast<std::size_t>(system)] != 0u ||
          !validate_system(fixture.host, system, eigenvalues, coefficients)) {
        return false;
      }
    } else if (errors[static_cast<std::size_t>(system)] != 0u ||
               !system_outputs_equal(fixture.host, system, eigenvalues, coefficients, kSentinel)) {
      return false;
    }
  }

  for (std::size_t bucket_index = 0; bucket_index < fixture.host.buckets.size(); ++bucket_index) {
    const Gfn2EigensolverBucket& bucket = fixture.host.buckets[bucket_index];
    std::vector<std::int32_t> expected;
    for (std::int32_t local = 0; local < bucket.system_count; ++local) {
      const std::int64_t bucket_slot = bucket.system_index_offset + local;
      const std::int32_t system =
          fixture.host.bucket_systems[static_cast<std::size_t>(bucket_slot)];
      if (fixture.host.active[static_cast<std::size_t>(system)] == 1u) {
        expected.push_back(system);
      }
    }
    const auto& observed = activity[bucket_index];
    if (observed.active_count != expected.size() ||
        observed.submitted_eigensolver_count != expected.size() ||
        observed.completed_count != expected.size() ||
        observed.submitted_backtransform_count != expected.size()) {
      std::cerr << "compaction count mismatch in bucket " << bucket_index
                << ": expected=" << expected.size() << " active=" << observed.active_count
                << " eig=" << observed.submitted_eigensolver_count
                << " complete=" << observed.completed_count
                << " back=" << observed.submitted_backtransform_count << '\n';
      return false;
    }
    for (std::size_t local = 0; local < expected.size(); ++local) {
      const std::size_t slot = static_cast<std::size_t>(bucket.system_index_offset) + local;
      if (compact_systems[slot] != expected[local]) {
        std::cerr << "unstable compaction order in bucket " << bucket_index << '\n';
        return false;
      }
    }
  }
  return true;
}

bool validate_uncompacted_launch(DeviceFixture& fixture) {
  std::vector<std::uint32_t> errors;
  std::vector<std::uint32_t> device_error;
  std::vector<double> eigenvalues;
  std::vector<double> coefficients;
  if (!fixture.system_errors.download(errors) || !fixture.device_error.download(device_error) ||
      device_error.size() != 1u || device_error[0] != 0u ||
      !fixture.eigenvalues.download(eigenvalues) || !fixture.coefficients.download(coefficients)) {
    return false;
  }
  for (std::int64_t system = 0; system < fixture.host.batch_size; ++system) {
    const bool active = fixture.host.active[static_cast<std::size_t>(system)] == 1u;
    if (errors[static_cast<std::size_t>(system)] != 0u) {
      return false;
    }
    if (active) {
      if (!validate_system(fixture.host, system, eigenvalues, coefficients)) {
        return false;
      }
    } else if (!system_outputs_equal(fixture.host, system, eigenvalues, coefficients, kSentinel)) {
      return false;
    }
  }
  return true;
}

bool test_compacted_graph_batch(std::int64_t batch_size, bool mixed_buckets) {
  DeviceFixture fixture;
  if (!fixture.create(make_batch(batch_size, mixed_buckets)) || !factor(fixture, 31u)) {
    return false;
  }
  const std::vector<double> clean_hamiltonian = fixture.host.hamiltonian;
  Gfn2EigensolverCompactedSolveGraph graph;
  const auto build = build_gfn2_compacted_eigensolver_graph_cuda(
      fixture.batch, fixture.host.buckets.data(),
      static_cast<std::int64_t>(fixture.host.buckets.size()), fixture.cache, 31u,
      fixture.hamiltonian.get(), Gfn2EigensolverOptions{}, fixture.providers.solver,
      fixture.providers.parameters, fixture.providers.blas, fixture.workspace, fixture.results,
      fixture.system_errors.get(), fixture.device_error.get(), graph);
  if (!build.success() || !graph.valid()) {
    std::cerr << "compacted graph build failed: status=" << static_cast<unsigned int>(build.status)
              << " cuda=" << cudaGetErrorString(build.cuda_status)
              << " cublas=" << static_cast<unsigned int>(build.cublas_status)
              << " cusolver=" << static_cast<unsigned int>(build.cusolver_status) << '\n';
    return false;
  }

  for (std::int64_t system = 0; system < batch_size; ++system) {
    fixture.host.active[static_cast<std::size_t>(system)] = system % 3 == 1 ? 0u : 1u;
  }
  fixture.host.hamiltonian = clean_hamiltonian;
  for (std::int64_t system = 0; system < batch_size; ++system) {
    if (fixture.host.active[static_cast<std::size_t>(system)] == 0u) {
      const std::int64_t begin = fixture.host.matrix_offsets[static_cast<std::size_t>(system)];
      const std::int64_t end = fixture.host.matrix_offsets[static_cast<std::size_t>(system + 1)];
      std::fill(fixture.host.hamiltonian.begin() + begin, fixture.host.hamiltonian.begin() + end,
                std::numeric_limits<double>::quiet_NaN());
    }
  }
  if (!fixture.active.upload(fixture.host.active) ||
      !fixture.hamiltonian.upload(fixture.host.hamiltonian) || !fixture.fill_outputs(kSentinel) ||
      !launch_compacted(fixture, graph) || !validate_compacted_launch(fixture)) {
    return false;
  }

  /* Replay with a different active set and matrix bytes. The Graph, provider
   * descriptors, and compact storage remain unchanged. */
  fixture.host.hamiltonian = clean_hamiltonian;
  for (std::int64_t system = 0; system < batch_size; ++system) {
    fixture.host.active[static_cast<std::size_t>(system)] = system % 4 == 0 ? 1u : 0u;
    if (fixture.host.active[static_cast<std::size_t>(system)] == 0u) {
      const std::int64_t begin = fixture.host.matrix_offsets[static_cast<std::size_t>(system)];
      const std::int64_t end = fixture.host.matrix_offsets[static_cast<std::size_t>(system + 1)];
      std::fill(fixture.host.hamiltonian.begin() + begin, fixture.host.hamiltonian.begin() + end,
                std::numeric_limits<double>::quiet_NaN());
    }
  }
  if (!fixture.active.upload(fixture.host.active) ||
      !fixture.hamiltonian.upload(fixture.host.hamiltonian) || !fixture.fill_outputs(kSentinel) ||
      !launch_compacted(fixture, graph) || !validate_compacted_launch(fixture)) {
    return false;
  }

  /* Body zero must be a true no-op even when every numerical input is poison. */
  std::fill(fixture.host.active.begin(), fixture.host.active.end(), 0u);
  std::fill(fixture.host.hamiltonian.begin(), fixture.host.hamiltonian.end(),
            std::numeric_limits<double>::quiet_NaN());
  return fixture.active.upload(fixture.host.active) &&
         fixture.hamiltonian.upload(fixture.host.hamiltonian) && fixture.fill_outputs(kSentinel) &&
         launch_compacted(fixture, graph) && validate_compacted_launch(fixture);
}

bool test_compacted_graph_filters_failed_peer() {
  constexpr std::int64_t kFailedSystem = 3;
  DeviceFixture fixture;
  if (!fixture.create(make_batch(8, false)) || !factor(fixture, 37u)) {
    return false;
  }
  Gfn2EigensolverCompactedSolveGraph graph;
  const auto build = build_gfn2_compacted_eigensolver_graph_cuda(
      fixture.batch, fixture.host.buckets.data(),
      static_cast<std::int64_t>(fixture.host.buckets.size()), fixture.cache, 37u,
      fixture.hamiltonian.get(), Gfn2EigensolverOptions{}, fixture.providers.solver,
      fixture.providers.parameters, fixture.providers.blas, fixture.workspace, fixture.results,
      fixture.system_errors.get(), fixture.device_error.get(), graph);
  if (!build.success()) {
    return false;
  }
  const std::int64_t begin = fixture.host.matrix_offsets[kFailedSystem];
  const std::int64_t end = fixture.host.matrix_offsets[kFailedSystem + 1];
  std::fill(fixture.host.hamiltonian.begin() + begin, fixture.host.hamiltonian.begin() + end,
            std::numeric_limits<double>::quiet_NaN());
  if (!fixture.hamiltonian.upload(fixture.host.hamiltonian) || !fixture.fill_outputs(kSentinel) ||
      !launch_compacted(fixture, graph)) {
    return false;
  }
  std::vector<std::uint32_t> errors;
  std::vector<double> eigenvalues;
  std::vector<double> coefficients;
  std::vector<Gfn2EigensolverBucketActivity> activity;
  std::vector<std::int32_t> compact;
  if (!fixture.system_errors.download(errors) || !fixture.eigenvalues.download(eigenvalues) ||
      !fixture.coefficients.download(coefficients) || !fixture.bucket_activity.download(activity) ||
      !fixture.compact_systems.download(compact)) {
    return false;
  }
  if (errors[kFailedSystem] !=
          static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kNonfiniteHamiltonian) ||
      !system_outputs_equal(fixture.host, kFailedSystem, eigenvalues, coefficients, kSentinel) ||
      activity.size() != 1u || activity[0].active_count != 7u ||
      activity[0].submitted_eigensolver_count != 7u || activity[0].completed_count != 7u ||
      activity[0].submitted_backtransform_count != 7u) {
    return false;
  }
  std::size_t compact_slot = 0u;
  for (std::int64_t system = 0; system < fixture.host.batch_size; ++system) {
    if (system == kFailedSystem) {
      continue;
    }
    if (errors[static_cast<std::size_t>(system)] != 0u || compact[compact_slot++] != system ||
        !validate_system(fixture.host, system, eigenvalues, coefficients)) {
      return false;
    }
  }
  return true;
}

bool test_compacted_graph_device_epoch_and_transactional_rebuild() {
  DeviceFixture fixture;
  DeviceBuffer<std::uint64_t> geometry_epoch;
  if (!fixture.create(make_batch(8, true)) || !geometry_epoch.allocate(1u) ||
      !factor(fixture, 41u) || !geometry_epoch.upload(std::vector<std::uint64_t>{41u})) {
    return false;
  }

  const Gfn2GeometryEpochDevice epoch{geometry_epoch.get(), 1, fixture.batch.plan_token};
  cudaStream_t solver_stream_before = nullptr;
  cudaStream_t blas_stream_before = nullptr;
  cublasPointerMode_t pointer_mode_before = CUBLAS_POINTER_MODE_HOST;
  cublasMath_t math_mode_before = CUBLAS_DEFAULT_MATH;
  if (!solver_ok(cusolverDnGetStream(fixture.providers.solver, &solver_stream_before),
                 "get solver stream before compacted build") ||
      !blas_ok(cublasGetStream(fixture.providers.blas, &blas_stream_before),
               "get BLAS stream before compacted build") ||
      !blas_ok(cublasGetPointerMode(fixture.providers.blas, &pointer_mode_before),
               "get BLAS pointer mode before compacted build") ||
      !blas_ok(cublasGetMathMode(fixture.providers.blas, &math_mode_before),
               "get BLAS math mode before compacted build")) {
    return false;
  }

  Gfn2EigensolverCompactedSolveGraph graph;
  const auto build = build_gfn2_compacted_eigensolver_graph_cuda(
      fixture.batch, fixture.host.buckets.data(),
      static_cast<std::int64_t>(fixture.host.buckets.size()), fixture.cache, epoch,
      fixture.hamiltonian.get(), Gfn2EigensolverOptions{}, fixture.providers.solver,
      fixture.providers.parameters, fixture.providers.blas, fixture.workspace, fixture.results,
      fixture.system_errors.get(), fixture.device_error.get(), graph);
  cudaStream_t solver_stream_after = nullptr;
  cudaStream_t blas_stream_after = nullptr;
  cublasPointerMode_t pointer_mode_after = CUBLAS_POINTER_MODE_HOST;
  cublasMath_t math_mode_after = CUBLAS_DEFAULT_MATH;
  if (!build.success() || !graph.valid() ||
      !solver_ok(cusolverDnGetStream(fixture.providers.solver, &solver_stream_after),
                 "get solver stream after compacted build") ||
      !blas_ok(cublasGetStream(fixture.providers.blas, &blas_stream_after),
               "get BLAS stream after compacted build") ||
      !blas_ok(cublasGetPointerMode(fixture.providers.blas, &pointer_mode_after),
               "get BLAS pointer mode after compacted build") ||
      !blas_ok(cublasGetMathMode(fixture.providers.blas, &math_mode_after),
               "get BLAS math mode after compacted build") ||
      solver_stream_after != solver_stream_before || blas_stream_after != blas_stream_before ||
      pointer_mode_after != pointer_mode_before || math_mode_after != math_mode_before) {
    std::cerr << "compacted graph build did not preserve provider handle state\n";
    return false;
  }

  /* A rejected rebuild must not invalidate the previously usable graph. */
  Gfn2EigensolverDeviceWorkspace undersized = fixture.workspace;
  undersized.compact_system_elements = fixture.host.batch_size - 1;
  const auto rejected = build_gfn2_compacted_eigensolver_graph_cuda(
      fixture.batch, fixture.host.buckets.data(),
      static_cast<std::int64_t>(fixture.host.buckets.size()), fixture.cache, epoch,
      fixture.hamiltonian.get(), Gfn2EigensolverOptions{}, fixture.providers.solver,
      fixture.providers.parameters, fixture.providers.blas, undersized, fixture.results,
      fixture.system_errors.get(), fixture.device_error.get(), graph);
  Gfn2EigensolverCompactedSolveGraph empty_graph;
  if (rejected.status != Gfn2EigensolverLaunchStatus::kInvalidArgument || !graph.valid() ||
      empty_graph.launch(fixture.providers.stream).status !=
          Gfn2EigensolverLaunchStatus::kInvalidArgument ||
      !fixture.fill_outputs(kSentinel) || !launch_compacted(fixture, graph) ||
      !validate_compacted_launch(fixture)) {
    return false;
  }

  /* Replay reads the epoch by stable device address rather than freezing the
   * build-time scalar. Refactor the cache to a new generation and prove that
   * the unchanged graph accepts it while observing a changed active mask. */
  if (!factor(fixture, 42u) || !geometry_epoch.upload(std::vector<std::uint64_t>{42u})) {
    return false;
  }
  for (std::int64_t system = 0; system < fixture.host.batch_size; ++system) {
    fixture.host.active[static_cast<std::size_t>(system)] = system % 3 == 0 ? 0u : 1u;
  }
  return fixture.active.upload(fixture.host.active) && fixture.fill_outputs(kSentinel) &&
         launch_compacted(fixture, graph) && validate_compacted_launch(fixture);
}

bool test_batch(std::int64_t batch_size, bool mixed_buckets,
                std::int32_t homogeneous_dimension = 3) {
  DeviceFixture fixture;
  if (!fixture.create(make_batch(batch_size, mixed_buckets, -1, homogeneous_dimension)) ||
      !factor(fixture, 7u) || !fixture.fill_outputs(kSentinel) || !solve(fixture, 7u)) {
    return false;
  }
  std::vector<std::uint32_t> errors;
  std::vector<double> eigenvalues;
  std::vector<double> coefficients;
  if (!fixture.system_errors.download(errors) || !fixture.eigenvalues.download(eigenvalues) ||
      !fixture.coefficients.download(coefficients)) {
    return false;
  }
  for (std::int64_t system = 0; system < batch_size; ++system) {
    if (errors[static_cast<std::size_t>(system)] != 0u ||
        !validate_system(fixture.host, system, eigenvalues, coefficients)) {
      return false;
    }
  }
  return true;
}

bool test_cpu_literal_parity() {
  TestBatch batch = make_batch(1, false, -1, 2);
  batch.overlap = {1.2, 0.15, 0.15, 0.9};
  batch.hamiltonian = {-0.8, 0.13, 0.13, 0.25};
  DeviceFixture fixture;
  if (!fixture.create(std::move(batch)) || !factor(fixture, 9u) ||
      !fixture.fill_outputs(kSentinel) || !solve(fixture, 9u)) {
    return false;
  }
  std::vector<double> eigenvalues;
  std::vector<double> coefficients;
  return fixture.eigenvalues.download(eigenvalues) && fixture.coefficients.download(coefficients) &&
         near(eigenvalues[0], -0.7192210550444913, 4.0e-13) &&
         near(eigenvalues[1], 0.28517850185300192, 4.0e-13) &&
         validate_system(fixture.host, 0, eigenvalues, coefficients);
}

bool test_inactive_poison_is_skipped() {
  constexpr std::int64_t kInactive = 3;
  TestBatch batch = make_batch(8, true);
  batch.active[static_cast<std::size_t>(kInactive)] = 0u;
  const std::int64_t begin = batch.matrix_offsets[static_cast<std::size_t>(kInactive)];
  const std::int64_t end = batch.matrix_offsets[static_cast<std::size_t>(kInactive + 1)];
  std::fill(batch.overlap.begin() + begin, batch.overlap.begin() + end,
            std::numeric_limits<double>::quiet_NaN());
  std::fill(batch.hamiltonian.begin() + begin, batch.hamiltonian.begin() + end,
            std::numeric_limits<double>::quiet_NaN());
  DeviceFixture fixture;
  if (!fixture.create(std::move(batch)) || !factor(fixture, 13u) ||
      !fixture.fill_outputs(kSentinel) || !solve(fixture, 13u)) {
    return false;
  }
  std::vector<std::uint32_t> errors;
  std::vector<double> eigenvalues;
  std::vector<double> coefficients;
  if (!fixture.system_errors.download(errors) || !fixture.eigenvalues.download(eigenvalues) ||
      !fixture.coefficients.download(coefficients)) {
    return false;
  }
  for (std::int64_t system = 0; system < fixture.host.batch_size; ++system) {
    if (errors[static_cast<std::size_t>(system)] != 0u) {
      return false;
    }
    if (system == kInactive) {
      if (!system_outputs_equal(fixture.host, system, eigenvalues, coefficients, kSentinel)) {
        return false;
      }
    } else if (!validate_system(fixture.host, system, eigenvalues, coefficients)) {
      return false;
    }
  }
  return true;
}

bool test_overlap_and_hamiltonian_validation() {
  constexpr std::int64_t kNonfinite = 1;
  constexpr std::int64_t kNonsymmetric = 2;
  TestBatch overlap_batch = make_batch(8, false);
  const std::int64_t nonfinite_begin =
      overlap_batch.matrix_offsets[static_cast<std::size_t>(kNonfinite)];
  const std::int64_t nonsymmetric_begin =
      overlap_batch.matrix_offsets[static_cast<std::size_t>(kNonsymmetric)];
  overlap_batch.overlap[static_cast<std::size_t>(nonfinite_begin)] =
      std::numeric_limits<double>::quiet_NaN();
  overlap_batch.overlap[static_cast<std::size_t>(nonsymmetric_begin + 1)] += 1.0e-3;
  DeviceFixture overlap_fixture;
  if (!overlap_fixture.create(std::move(overlap_batch)) || !factor(overlap_fixture, 17u)) {
    return false;
  }
  std::vector<std::uint32_t> errors;
  if (!overlap_fixture.system_errors.download(errors)) {
    return false;
  }
  for (std::int64_t system = 0; system < overlap_fixture.host.batch_size; ++system) {
    const std::uint32_t expected =
        system == kNonfinite
            ? static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kNonfiniteOverlap)
        : system == kNonsymmetric
            ? static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kNonsymmetricOverlap)
            : 0u;
    if (errors[static_cast<std::size_t>(system)] != expected) {
      return false;
    }
  }

  TestBatch hamiltonian_batch = make_batch(8, false);
  const std::int64_t h_nonfinite_begin =
      hamiltonian_batch.matrix_offsets[static_cast<std::size_t>(kNonfinite)];
  const std::int64_t h_nonsymmetric_begin =
      hamiltonian_batch.matrix_offsets[static_cast<std::size_t>(kNonsymmetric)];
  hamiltonian_batch.hamiltonian[static_cast<std::size_t>(h_nonfinite_begin)] =
      std::numeric_limits<double>::infinity();
  hamiltonian_batch.hamiltonian[static_cast<std::size_t>(h_nonsymmetric_begin + 1)] += 1.0e-3;
  DeviceFixture hamiltonian_fixture;
  if (!hamiltonian_fixture.create(std::move(hamiltonian_batch)) ||
      !factor(hamiltonian_fixture, 18u) || !hamiltonian_fixture.fill_outputs(kSentinel) ||
      !solve(hamiltonian_fixture, 18u)) {
    return false;
  }
  std::vector<double> eigenvalues;
  std::vector<double> coefficients;
  if (!hamiltonian_fixture.system_errors.download(errors) ||
      !hamiltonian_fixture.eigenvalues.download(eigenvalues) ||
      !hamiltonian_fixture.coefficients.download(coefficients)) {
    return false;
  }
  for (std::int64_t system = 0; system < hamiltonian_fixture.host.batch_size; ++system) {
    const std::uint32_t expected =
        system == kNonfinite
            ? static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kNonfiniteHamiltonian)
        : system == kNonsymmetric
            ? static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kNonsymmetricHamiltonian)
            : 0u;
    if (errors[static_cast<std::size_t>(system)] != expected) {
      return false;
    }
    if (expected != 0u) {
      if (!system_outputs_equal(hamiltonian_fixture.host, system, eigenvalues, coefficients,
                                kSentinel)) {
        return false;
      }
    } else if (!validate_system(hamiltonian_fixture.host, system, eigenvalues, coefficients)) {
      return false;
    }
  }
  return true;
}

bool test_active_offset_and_singular_failures() {
  constexpr std::int64_t kFailedSystem = 4;
  TestBatch active_batch = make_batch(8, false);
  active_batch.active[static_cast<std::size_t>(kFailedSystem)] = 7u;
  DeviceFixture active_fixture;
  if (!active_fixture.create(std::move(active_batch)) || !factor(active_fixture, 31u)) {
    return false;
  }
  std::vector<std::uint32_t> errors;
  if (!active_fixture.system_errors.download(errors)) {
    return false;
  }
  for (std::int64_t system = 0; system < active_fixture.host.batch_size; ++system) {
    const std::uint32_t expected =
        system == kFailedSystem
            ? static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kInvalidActiveMask)
            : 0u;
    if (errors[static_cast<std::size_t>(system)] != expected) {
      return false;
    }
  }

  TestBatch offset_batch = make_batch(8, false);
  --offset_batch.orbital_offsets.back();
  --offset_batch.matrix_offsets.back();
  DeviceFixture offset_fixture;
  if (!offset_fixture.create(std::move(offset_batch)) || !factor(offset_fixture, 32u) ||
      !offset_fixture.system_errors.download(errors)) {
    return false;
  }
  for (std::int64_t system = 0; system < offset_fixture.host.batch_size; ++system) {
    const std::uint32_t expected =
        system + 1 == offset_fixture.host.batch_size
            ? static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kInvalidOffsets)
            : 0u;
    if (errors[static_cast<std::size_t>(system)] != expected) {
      return false;
    }
  }

  TestBatch singular_batch = make_batch(8, false);
  const std::int64_t begin = singular_batch.matrix_offsets[static_cast<std::size_t>(kFailedSystem)];
  const std::int64_t n =
      singular_batch.orbital_offsets[static_cast<std::size_t>(kFailedSystem + 1)] -
      singular_batch.orbital_offsets[static_cast<std::size_t>(kFailedSystem)];
  for (std::int64_t column = 0; column < n; ++column) {
    singular_batch.overlap[static_cast<std::size_t>(begin + (n - 1) * n + column)] = 0.0;
    singular_batch.overlap[static_cast<std::size_t>(begin + column * n + (n - 1))] = 0.0;
  }
  DeviceFixture singular_fixture;
  if (!singular_fixture.create(std::move(singular_batch)) || !factor(singular_fixture, 33u) ||
      !singular_fixture.system_errors.download(errors)) {
    return false;
  }
  for (std::int64_t system = 0; system < singular_fixture.host.batch_size; ++system) {
    const std::uint32_t expected =
        system == kFailedSystem
            ? static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kOverlapNotPositiveDefinite)
            : 0u;
    if (errors[static_cast<std::size_t>(system)] != expected) {
      return false;
    }
  }
  return true;
}

bool test_ill_conditioned_peer_isolation() {
  constexpr std::int64_t kFailedSystem = 3;
  DeviceFixture fixture;
  if (!fixture.create(make_batch(8, false, kFailedSystem)) || !factor(fixture, 11u)) {
    return false;
  }
  std::vector<std::uint32_t> factor_errors;
  if (!fixture.system_errors.download(factor_errors)) {
    return false;
  }
  for (std::int64_t system = 0; system < fixture.host.batch_size; ++system) {
    const std::uint32_t expected =
        system == kFailedSystem
            ? static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kOverlapIllConditioned)
            : 0u;
    if (factor_errors[static_cast<std::size_t>(system)] != expected) {
      std::cerr << "unexpected factor error for system " << system << '\n';
      return false;
    }
  }
  if (!fixture.fill_outputs(kSentinel) || !solve(fixture, 11u)) {
    return false;
  }
  std::vector<std::uint32_t> solve_errors;
  std::vector<double> eigenvalues;
  std::vector<double> coefficients;
  if (!fixture.system_errors.download(solve_errors) || !fixture.eigenvalues.download(eigenvalues) ||
      !fixture.coefficients.download(coefficients)) {
    return false;
  }
  for (std::int64_t system = 0; system < fixture.host.batch_size; ++system) {
    if (system == kFailedSystem) {
      const std::int64_t ob = fixture.host.orbital_offsets[static_cast<std::size_t>(system)];
      const std::int64_t oe = fixture.host.orbital_offsets[static_cast<std::size_t>(system + 1)];
      const std::int64_t mb = fixture.host.matrix_offsets[static_cast<std::size_t>(system)];
      const std::int64_t me = fixture.host.matrix_offsets[static_cast<std::size_t>(system + 1)];
      if (solve_errors[static_cast<std::size_t>(system)] !=
              static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kStaleOverlapCache) ||
          !std::all_of(eigenvalues.begin() + ob, eigenvalues.begin() + oe,
                       [](double value) { return value == kSentinel; }) ||
          !std::all_of(coefficients.begin() + mb, coefficients.begin() + me,
                       [](double value) { return value == kSentinel; })) {
        return false;
      }
    } else if (solve_errors[static_cast<std::size_t>(system)] != 0u ||
               !validate_system(fixture.host, system, eigenvalues, coefficients)) {
      return false;
    }
  }
  return true;
}

bool test_cache_generation_staleness() {
  DeviceFixture fixture;
  if (!fixture.create(make_batch(8, true)) || !factor(fixture, 19u) ||
      !fixture.fill_outputs(kSentinel) || !solve(fixture, 20u)) {
    return false;
  }
  std::vector<std::uint32_t> errors;
  std::vector<double> eigenvalues;
  std::vector<double> coefficients;
  if (!fixture.system_errors.download(errors) || !fixture.eigenvalues.download(eigenvalues) ||
      !fixture.coefficients.download(coefficients)) {
    return false;
  }
  return std::all_of(errors.begin(), errors.end(),
                     [](std::uint32_t error) {
                       return error == static_cast<std::uint32_t>(
                                           Gfn2EigensolverDeviceError::kStaleOverlapCache);
                     }) &&
         std::all_of(eigenvalues.begin(), eigenvalues.end(),
                     [](double value) { return value == kSentinel; }) &&
         std::all_of(coefficients.begin(), coefficients.end(),
                     [](double value) { return value == kSentinel; });
}

bool test_single_cache_member_peer_isolation() {
  constexpr std::int64_t kFailedSystem = 5;
  DeviceFixture fixture;
  if (!fixture.create(make_batch(8, true)) || !factor(fixture, 21u)) {
    return false;
  }
  std::vector<std::uint64_t> generations(static_cast<std::size_t>(fixture.host.batch_size), 21u);
  generations[static_cast<std::size_t>(kFailedSystem)] = 20u;
  if (!fixture.cache_generations.upload(generations) || !fixture.fill_outputs(kSentinel) ||
      !solve(fixture, 21u)) {
    return false;
  }
  std::vector<std::uint32_t> errors;
  std::vector<double> eigenvalues;
  std::vector<double> coefficients;
  if (!fixture.system_errors.download(errors) || !fixture.eigenvalues.download(eigenvalues) ||
      !fixture.coefficients.download(coefficients)) {
    return false;
  }
  for (std::int64_t system = 0; system < fixture.host.batch_size; ++system) {
    if (system == kFailedSystem) {
      if (errors[static_cast<std::size_t>(system)] !=
              static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kStaleOverlapCache) ||
          !system_outputs_equal(fixture.host, system, eigenvalues, coefficients, kSentinel)) {
        return false;
      }
    } else if (errors[static_cast<std::size_t>(system)] != 0u ||
               !validate_system(fixture.host, system, eigenvalues, coefficients)) {
      return false;
    }
  }
  return true;
}

bool test_sticky_error_and_invalid_bucket_map_fail_closed() {
  DeviceFixture sticky;
  if (!sticky.create(make_batch(8, true)) || !factor(sticky, 25u) ||
      !sticky.fill_outputs(kSentinel) ||
      !cuda_ok(reset_gfn2_eigensolver_device_errors_cuda(
                   sticky.host.batch_size, sticky.system_errors.get(), sticky.device_error.get(),
                   sticky.providers.stream),
               "reset sticky errors") ||
      !sticky.device_error.upload(
          {static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kInvalidBucketMap)}) ||
      !solve(sticky, 25u, false)) {
    return false;
  }
  std::vector<std::uint32_t> errors;
  std::vector<std::uint32_t> global_error;
  std::vector<double> eigenvalues;
  std::vector<double> coefficients;
  if (!sticky.system_errors.download(errors) || !sticky.device_error.download(global_error) ||
      !sticky.eigenvalues.download(eigenvalues) || !sticky.coefficients.download(coefficients) ||
      !std::all_of(errors.begin(), errors.end(), [](std::uint32_t value) { return value == 0u; }) ||
      global_error[0] !=
          static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kInvalidBucketMap) ||
      !std::all_of(eigenvalues.begin(), eigenvalues.end(),
                   [](double value) { return value == kSentinel; }) ||
      !std::all_of(coefficients.begin(), coefficients.end(),
                   [](double value) { return value == kSentinel; })) {
    return false;
  }

  DeviceFixture invalid_map;
  if (!invalid_map.create(make_batch(8, true)) || !factor(invalid_map, 26u) ||
      !invalid_map.fill_outputs(kSentinel)) {
    return false;
  }
  std::vector<std::int32_t> duplicate = invalid_map.host.bucket_systems;
  duplicate[1] = duplicate[0];
  if (!invalid_map.bucket_systems.upload(duplicate) || !solve(invalid_map, 26u) ||
      !invalid_map.system_errors.download(errors) ||
      !invalid_map.device_error.download(global_error) ||
      !invalid_map.eigenvalues.download(eigenvalues) ||
      !invalid_map.coefficients.download(coefficients)) {
    return false;
  }
  return std::all_of(errors.begin(), errors.end(),
                     [](std::uint32_t value) { return value == 0u; }) &&
         global_error[0] ==
             static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kInvalidBucketMap) &&
         std::all_of(eigenvalues.begin(), eigenvalues.end(),
                     [](double value) { return value == kSentinel; }) &&
         std::all_of(coefficients.begin(), coefficients.end(),
                     [](double value) { return value == kSentinel; });
}

bool test_host_validation_aliases_and_limits() {
  DeviceFixture fixture;
  if (!fixture.create(make_batch(8, true)) || !factor(fixture, 29u)) {
    return false;
  }
  const auto factor_launch = [&](const Gfn2EigensolverDeviceWorkspace& workspace,
                                 const Gfn2EigensolverOverlapCache& cache, const double* overlap) {
    return factor_gfn2_overlap_cuda(
        fixture.batch, fixture.host.buckets.data(),
        static_cast<std::int64_t>(fixture.host.buckets.size()), overlap, 30u,
        Gfn2EigensolverOptions{}, fixture.providers.solver, fixture.providers.parameters, workspace,
        cache, fixture.system_errors.get(), fixture.device_error.get(), fixture.providers.stream);
  };
  Gfn2EigensolverDeviceWorkspace aliased_workspace = fixture.workspace;
  aliased_workspace.matrix_scratch_b = aliased_workspace.matrix_scratch_a;
  if (factor_launch(aliased_workspace, fixture.cache, fixture.overlap.get()).success()) {
    return false;
  }
  Gfn2EigensolverOverlapCache aliased_cache = fixture.cache;
  aliased_cache.cholesky_factors = fixture.overlap.get();
  if (factor_launch(fixture.workspace, aliased_cache, fixture.overlap.get()).success()) {
    return false;
  }
  Gfn2EigensolverDeviceWorkspace wrong_token = fixture.workspace;
  ++wrong_token.plan_token;
  if (factor_launch(wrong_token, fixture.cache, fixture.overlap.get()).success()) {
    return false;
  }
  const auto* misaligned_overlap = reinterpret_cast<const double*>(
      reinterpret_cast<const std::byte*>(fixture.overlap.get()) + 1);
  if (factor_launch(fixture.workspace, fixture.cache, misaligned_overlap).success()) {
    return false;
  }

  Gfn2EigensolverDeviceResults aliased_results = fixture.results;
  aliased_results.coefficients = fixture.workspace.matrix_scratch_b;
  if (solve_gfn2_eigensystems_cuda(
          fixture.batch, fixture.host.buckets.data(),
          static_cast<std::int64_t>(fixture.host.buckets.size()), fixture.cache, 29u,
          fixture.hamiltonian.get(), Gfn2EigensolverOptions{}, fixture.providers.solver,
          fixture.providers.parameters, fixture.providers.blas, fixture.workspace, aliased_results,
          fixture.system_errors.get(), fixture.device_error.get(), fixture.providers.stream)
          .success()) {
    return false;
  }
  Gfn2EigensolverDeviceResults wrong_results = fixture.results;
  ++wrong_results.plan_token;
  if (solve_gfn2_eigensystems_cuda(
          fixture.batch, fixture.host.buckets.data(),
          static_cast<std::int64_t>(fixture.host.buckets.size()), fixture.cache, 29u,
          fixture.hamiltonian.get(), Gfn2EigensolverOptions{}, fixture.providers.solver,
          fixture.providers.parameters, fixture.providers.blas, fixture.workspace, wrong_results,
          fixture.system_errors.get(), fixture.device_error.get(), fixture.providers.stream)
          .success()) {
    return false;
  }
  if (reset_gfn2_eigensolver_device_errors_cuda(
          fixture.host.batch_size, fixture.system_errors.get(), fixture.system_errors.get() + 1,
          fixture.providers.stream) != cudaErrorInvalidValue) {
    return false;
  }
  const Gfn2EigensolverBucket huge{46341, 1, 0, 0, 0};
  Gfn2EigensolverWorkspaceRequirements requirements;
  return !query_gfn2_eigensolver_bucket_workspace_cuda(
              fixture.providers.solver, fixture.providers.parameters, huge,
              fixture.workspace.matrix_scratch_b, fixture.workspace.eigenvalue_scratch,
              requirements)
              .success();
}

/* Exercise capture without hiding a provider limitation behind host fallback. */
bool test_graph_capture() {
  DeviceFixture fixture;
  if (!fixture.create(make_batch(8, false)) || !fixture.fill_outputs(kSentinel)) {
    return false;
  }
  if (!cuda_ok(cudaStreamBeginCapture(fixture.providers.stream, cudaStreamCaptureModeThreadLocal),
               "cudaStreamBeginCapture")) {
    return false;
  }
  const cudaError_t factor_reset = reset_gfn2_eigensolver_device_errors_cuda(
      fixture.host.batch_size, fixture.system_errors.get(), fixture.device_error.get(),
      fixture.providers.stream);
  const auto factor_launch = factor_gfn2_overlap_cuda(
      fixture.batch, fixture.host.buckets.data(),
      static_cast<std::int64_t>(fixture.host.buckets.size()), fixture.overlap.get(), 23u,
      Gfn2EigensolverOptions{}, fixture.providers.solver, fixture.providers.parameters,
      fixture.workspace, fixture.cache, fixture.system_errors.get(), fixture.device_error.get(),
      fixture.providers.stream);
  const cudaError_t solve_reset = reset_gfn2_eigensolver_device_errors_cuda(
      fixture.host.batch_size, fixture.system_errors.get(), fixture.device_error.get(),
      fixture.providers.stream);
  const auto solve_launch = solve_gfn2_eigensystems_cuda(
      fixture.batch, fixture.host.buckets.data(),
      static_cast<std::int64_t>(fixture.host.buckets.size()), fixture.cache, 23u,
      fixture.hamiltonian.get(), Gfn2EigensolverOptions{}, fixture.providers.solver,
      fixture.providers.parameters, fixture.providers.blas, fixture.workspace, fixture.results,
      fixture.system_errors.get(), fixture.device_error.get(), fixture.providers.stream);
  cudaGraph_t graph = nullptr;
  const cudaError_t end_capture = cudaStreamEndCapture(fixture.providers.stream, &graph);
  if (factor_reset != cudaSuccess || !factor_launch.success() || solve_reset != cudaSuccess ||
      !solve_launch.success() || end_capture != cudaSuccess) {
    if (graph != nullptr) {
      (void)cudaGraphDestroy(graph);
    }
    std::cerr << "linked cuSOLVER provider rejected CUDA Graph capture: factor="
              << static_cast<unsigned int>(factor_launch.status)
              << " solve=" << static_cast<unsigned int>(solve_launch.status)
              << " end=" << cudaGetErrorString(end_capture) << '\n';
    return false;
  }
  cudaGraphExec_t executable = nullptr;
  const bool ok =
      cuda_ok(cudaGraphInstantiate(&executable, graph, 0), "cudaGraphInstantiate") &&
      cuda_ok(cudaGraphLaunch(executable, fixture.providers.stream), "cudaGraphLaunch") &&
      cuda_ok(cudaGraphLaunch(executable, fixture.providers.stream), "cudaGraphLaunch replay") &&
      cuda_ok(cudaStreamSynchronize(fixture.providers.stream), "graph synchronize");
  if (executable != nullptr) {
    (void)cudaGraphExecDestroy(executable);
  }
  (void)cudaGraphDestroy(graph);
  if (!ok) {
    return false;
  }
  std::vector<double> eigenvalues;
  std::vector<double> coefficients;
  if (!fixture.eigenvalues.download(eigenvalues) || !fixture.coefficients.download(coefficients)) {
    return false;
  }
  for (std::int64_t system = 0; system < fixture.host.batch_size; ++system) {
    if (!validate_system(fixture.host, system, eigenvalues, coefficients)) {
      return false;
    }
  }
  return true;
}

/* Benchmark-facing variant of make_batch() with caller-selected AO
 * dimensions. Homogeneous benchmarks use one n=32 bucket; heterogeneous
 * benchmarks interleave n=16 and n=40 so the two buckets are separable in
 * the compaction telemetry while remaining well conditioned for batched
 * symmetric eigensolves. Everything else (offsets, values, bucket order)
 * follows the validated make_batch() fabrication exactly. */
TestBatch make_benchmark_batch(std::int64_t batch_size, bool heterogeneous) {
  TestBatch data;
  data.batch_size = batch_size;
  data.orbital_offsets.assign(static_cast<std::size_t>(batch_size + 1), 0);
  data.matrix_offsets.assign(static_cast<std::size_t>(batch_size + 1), 0);
  data.active.assign(static_cast<std::size_t>(batch_size), 1u);
  std::vector<std::int32_t> dimensions(static_cast<std::size_t>(batch_size), 32);
  if (heterogeneous) {
    for (std::int64_t system = 0; system < batch_size; ++system) {
      dimensions[static_cast<std::size_t>(system)] = system % 2 == 0 ? 16 : 40;
    }
  }
  for (std::int64_t system = 0; system < batch_size; ++system) {
    const std::int64_t n = dimensions[static_cast<std::size_t>(system)];
    data.total_orbitals += n;
    data.total_matrices += n * n;
    data.orbital_offsets[static_cast<std::size_t>(system + 1)] = data.total_orbitals;
    data.matrix_offsets[static_cast<std::size_t>(system + 1)] = data.total_matrices;
  }
  data.overlap.resize(static_cast<std::size_t>(data.total_matrices), 0.0);
  data.hamiltonian.resize(static_cast<std::size_t>(data.total_matrices), 0.0);
  for (std::int64_t system = 0; system < batch_size; ++system) {
    const std::int64_t n = dimensions[static_cast<std::size_t>(system)];
    const std::int64_t begin = data.matrix_offsets[static_cast<std::size_t>(system)];
    for (std::int64_t row = 0; row < n; ++row) {
      for (std::int64_t column = 0; column < n; ++column) {
        const double distance = static_cast<double>(std::abs(row - column) + 1);
        const double s = row == column
                             ? 1.05 + 0.05 * static_cast<double>(row + 1) / static_cast<double>(n)
                             : 0.01 / distance;
        const double h = row == column ? -0.8 + 0.035 * static_cast<double>(row) +
                                             0.001 * static_cast<double>(system)
                                       : 0.02 / distance;
        data.overlap[static_cast<std::size_t>(begin + row * n + column)] = s;
        data.hamiltonian[static_cast<std::size_t>(begin + row * n + column)] = h;
      }
    }
  }
  std::int64_t system_offset = 0;
  std::int64_t matrix_offset = 0;
  std::int64_t orbital_offset = 0;
  std::vector<std::int32_t> bucket_dimensions = dimensions;
  std::sort(bucket_dimensions.begin(), bucket_dimensions.end());
  bucket_dimensions.erase(std::unique(bucket_dimensions.begin(), bucket_dimensions.end()),
                          bucket_dimensions.end());
  for (const std::int32_t n : bucket_dimensions) {
    std::vector<std::int32_t> systems;
    for (std::int64_t system = 0; system < batch_size; ++system) {
      if (dimensions[static_cast<std::size_t>(system)] == n) {
        systems.push_back(static_cast<std::int32_t>(system));
      }
    }
    if (systems.empty()) {
      continue;
    }
    data.buckets.push_back({n, static_cast<std::int32_t>(systems.size()), system_offset,
                            matrix_offset, orbital_offset});
    data.bucket_systems.insert(data.bucket_systems.end(), systems.begin(), systems.end());
    system_offset += static_cast<std::int64_t>(systems.size());
    matrix_offset += static_cast<std::int64_t>(n) * n * static_cast<std::int64_t>(systems.size());
    orbital_offset += static_cast<std::int64_t>(n) * static_cast<std::int64_t>(systems.size());
  }
  return data;
}

struct LatencyStats {
  double mean_us = 0.0;
  double min_us = 0.0;
  double p50_us = 0.0;
  double max_us = 0.0;
  std::vector<double> samples_us;
};

/* Repeated enqueue/record/synchronize timing for one launch path. Every API
 * result is checked so a failed enqueue or asynchronous CUDA error can never
 * become a plausible-looking performance sample. `prepare` is enqueued on
 * the stream before the timed interval so error resets do not count toward
 * the measured latency. */
template <typename Prepare, typename Enqueue>
bool measure_launch_latency(DeviceFixture& fixture, int warmup, int samples, Prepare&& prepare,
                            Enqueue&& enqueue, LatencyStats& stats) {
  if (warmup < 0 || samples <= 0) {
    return false;
  }
  std::vector<double> samples_us;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  bool ok = cuda_ok(cudaEventCreate(&start), "cudaEventCreate start") &&
            cuda_ok(cudaEventCreate(&stop), "cudaEventCreate stop");
  if (!ok) {
    if (start != nullptr) {
      (void)cudaEventDestroy(start);
    }
    return false;
  }
  for (int sample = -warmup; sample < samples; ++sample) {
    if (!prepare() ||
        !cuda_ok(cudaEventRecord(start, fixture.providers.stream), "cudaEventRecord start") ||
        !enqueue() ||
        !cuda_ok(cudaEventRecord(stop, fixture.providers.stream), "cudaEventRecord stop") ||
        !cuda_ok(cudaEventSynchronize(stop), "cudaEventSynchronize stop")) {
      ok = false;
      break;
    }
    float elapsed_ms = 0.0F;
    if (!cuda_ok(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime")) {
      ok = false;
      break;
    }
    if (sample >= 0) {
      samples_us.push_back(static_cast<double>(elapsed_ms) * 1000.0);
    }
  }
  ok = cuda_ok(cudaEventDestroy(stop), "cudaEventDestroy stop") &&
       cuda_ok(cudaEventDestroy(start), "cudaEventDestroy start") && ok;
  if (!ok || samples_us.size() != static_cast<std::size_t>(samples)) {
    return false;
  }
  stats.samples_us = samples_us;
  std::vector<double> sorted_samples = samples_us;
  std::sort(sorted_samples.begin(), sorted_samples.end());
  double total = 0.0;
  for (const double value : samples_us) {
    total += value;
    stats.min_us = stats.min_us == 0.0 ? value : std::min(stats.min_us, value);
    stats.max_us = std::max(stats.max_us, value);
  }
  stats.mean_us = total / static_cast<double>(samples_us.size());
  const std::size_t middle = sorted_samples.size() / 2u;
  stats.p50_us = sorted_samples.size() % 2u == 0u
                     ? 0.5 * (sorted_samples[middle - 1u] + sorted_samples[middle])
                     : sorted_samples[middle];
  return true;
}

void print_latency_stats(const char* key, const LatencyStats& stats) {
  std::printf(
      ",\"%s\":{\"mean_us\":%.6f,\"min_us\":%.6f,\"p50_us\":%.6f,\"max_us\":%.6f,"
      "\"samples_us\":[",
      key, stats.mean_us, stats.min_us, stats.p50_us, stats.max_us);
  for (std::size_t index = 0; index < stats.samples_us.size(); ++index) {
    if (index != 0u) {
      std::printf(",");
    }
    std::printf("%.6f", stats.samples_us[index]);
  }
  std::printf("]}");
}

int run_compaction_benchmark() {
  constexpr int kWarmup = 5;
  constexpr int kSamples = 50;
  std::printf(
      "{\"record_type\":\"protocol\",\"benchmark\":\"compaction\",\"warmups\":%d,"
      "\"samples\":%d,\"timing\":\"cudaEvent elapsed time on the benchmark stream\","
      "\"workloads\":\"homogeneous n=32; heterogeneous n=16/n=40 when batch >= 2\"}\n",
      kWarmup, kSamples);
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    for (const bool heterogeneous : {false, true}) {
      DeviceFixture fixture;
      if (!fixture.create(make_benchmark_batch(batch_size, heterogeneous)) ||
          !factor(fixture, 31u)) {
        return 1;
      }
      Gfn2EigensolverCompactedSolveGraph graph;
      const auto build = build_gfn2_compacted_eigensolver_graph_cuda(
          fixture.batch, fixture.host.buckets.data(),
          static_cast<std::int64_t>(fixture.host.buckets.size()), fixture.cache, 31u,
          fixture.hamiltonian.get(), Gfn2EigensolverOptions{}, fixture.providers.solver,
          fixture.providers.parameters, fixture.providers.blas, fixture.workspace, fixture.results,
          fixture.system_errors.get(), fixture.device_error.get(), graph);
      if (!build.success() || !graph.valid()) {
        std::cerr << "compacted graph build failed: status="
                  << static_cast<unsigned int>(build.status)
                  << " cuda=" << cudaGetErrorString(build.cuda_status) << '\n';
        return 1;
      }
      const std::vector<double> clean_hamiltonian = fixture.host.hamiltonian;
      const std::int64_t bucket_count = static_cast<std::int64_t>(fixture.host.buckets.size());

      /* Canonical deterministic active-count tier sweep. The largest tier is
       * always the full batch and the final tier is empty. */
      std::vector<std::int64_t> tiers;
      for (const std::int64_t fraction : {64, 48, 32, 24, 16, 8, 4, 1}) {
        const std::int64_t active =
            std::min<std::int64_t>(batch_size, (batch_size * fraction + 31) / 32);
        if (tiers.empty() || tiers.back() != active) {
          tiers.push_back(active);
        }
        if (active == 0) {
          break;
        }
      }
      tiers.push_back(0);
      for (const std::int64_t active_total : tiers) {
        for (std::int64_t system = 0; system < batch_size; ++system) {
          fixture.host.active[static_cast<std::size_t>(system)] = system < active_total ? 1u : 0u;
        }
        fixture.host.hamiltonian = clean_hamiltonian;
        for (std::int64_t system = 0; system < batch_size; ++system) {
          if (fixture.host.active[static_cast<std::size_t>(system)] == 0u) {
            const std::int64_t begin =
                fixture.host.matrix_offsets[static_cast<std::size_t>(system)];
            const std::int64_t end =
                fixture.host.matrix_offsets[static_cast<std::size_t>(system + 1)];
            std::fill(fixture.host.hamiltonian.begin() + begin,
                      fixture.host.hamiltonian.begin() + end,
                      std::numeric_limits<double>::quiet_NaN());
          }
        }
        if (!fixture.active.upload(fixture.host.active) ||
            !fixture.hamiltonian.upload(fixture.host.hamiltonian)) {
          return 1;
        }

        /* One correctness pass for each path, then repeated timed launches.
         * The first pass proves the compacted result equals the uncompacted
         * result for every active peer in this exact tier. */
        if (!fixture.fill_outputs(kSentinel)) {
          return 1;
        }
        if (!launch_compacted(fixture, graph) || !validate_compacted_launch(fixture)) {
          std::cerr << "compacted correctness failed at active=" << active_total << '\n';
          return 1;
        }
        std::vector<double> compacted_eigenvalues;
        std::vector<double> compacted_coefficients;
        if (!fixture.eigenvalues.download(compacted_eigenvalues) ||
            !fixture.coefficients.download(compacted_coefficients) ||
            !fixture.fill_outputs(kSentinel)) {
          return 1;
        }
        if (!solve(fixture, 31u)) {
          std::cerr << "uncompacted solve failed at active=" << active_total << '\n';
          return 1;
        }
        std::vector<double> uncompacted_eigenvalues;
        std::vector<double> uncompacted_coefficients;
        if (!fixture.eigenvalues.download(uncompacted_eigenvalues) ||
            !fixture.coefficients.download(uncompacted_coefficients)) {
          return 1;
        }
        for (std::int64_t system = 0; system < batch_size; ++system) {
          if (fixture.host.active[static_cast<std::size_t>(system)] == 0u) {
            continue;
          }
          const std::int64_t orbital_begin =
              fixture.host.orbital_offsets[static_cast<std::size_t>(system)];
          const std::int64_t orbital_end =
              fixture.host.orbital_offsets[static_cast<std::size_t>(system + 1)];
          const std::int64_t matrix_begin =
              fixture.host.matrix_offsets[static_cast<std::size_t>(system)];
          const std::int64_t matrix_end =
              fixture.host.matrix_offsets[static_cast<std::size_t>(system + 1)];
          for (std::int64_t index = orbital_begin; index < orbital_end; ++index) {
            if (!near(compacted_eigenvalues[static_cast<std::size_t>(index)],
                      uncompacted_eigenvalues[static_cast<std::size_t>(index)], 1.0e-12)) {
              std::cerr << "tier active=" << active_total << " system=" << system
                        << " eigen mismatch\n";
              return 1;
            }
          }
          for (std::int64_t index = matrix_begin; index < matrix_end; ++index) {
            if (!near(compacted_coefficients[static_cast<std::size_t>(index)],
                      uncompacted_coefficients[static_cast<std::size_t>(index)], 1.0e-10)) {
              std::cerr << "tier active=" << active_total << " system=" << system
                        << " coefficient mismatch\n";
              return 1;
            }
          }
          if (!validate_system(fixture.host, system, compacted_eigenvalues,
                               compacted_coefficients)) {
            return 1;
          }
        }

        /* Timed steady-state launches for this tier. Validate each path after
         * timing so the final sample cannot hide a per-system/device failure. */
        LatencyStats compacted;
        if (!fixture.fill_outputs(kSentinel) ||
            !measure_launch_latency(
                fixture, kWarmup, kSamples,
                [&]() {
                  return cuda_ok(reset_gfn2_eigensolver_device_errors_cuda(
                                     fixture.host.batch_size, fixture.system_errors.get(),
                                     fixture.device_error.get(), fixture.providers.stream),
                                 "reset compacted timing errors");
                },
                [&]() { return enqueue_compacted(fixture, graph); }, compacted) ||
            !validate_compacted_launch(fixture)) {
          std::cerr << "compacted timing failed at active=" << active_total << '\n';
          return 1;
        }

        std::vector<Gfn2EigensolverBucketActivity> activity;
        if (!fixture.bucket_activity.download(activity)) {
          return 1;
        }
        LatencyStats uncompacted;
        if (!fixture.fill_outputs(kSentinel) ||
            !measure_launch_latency(
                fixture, kWarmup, kSamples,
                [&]() {
                  return cuda_ok(reset_gfn2_eigensolver_device_errors_cuda(
                                     fixture.host.batch_size, fixture.system_errors.get(),
                                     fixture.device_error.get(), fixture.providers.stream),
                                 "reset uncompacted timing errors");
                },
                [&]() { return enqueue_uncompacted(fixture, 31u); }, uncompacted) ||
            !validate_uncompacted_launch(fixture)) {
          std::cerr << "uncompacted timing failed at active=" << active_total << '\n';
          return 1;
        }
        std::printf(
            "{\"record_type\":\"measurement\",\"batch\":%lld,\"workload\":\"%s\","
            "\"active_total\":%lld,"
            "\"tier_fraction\":%.6f,\"bucket_count\":%lld",
            static_cast<long long>(batch_size), heterogeneous ? "heterogeneous" : "homogeneous",
            static_cast<long long>(active_total),
            static_cast<double>(active_total) / static_cast<double>(batch_size),
            static_cast<long long>(bucket_count));
        for (std::size_t bucket_index = 0; bucket_index < activity.size(); ++bucket_index) {
          const auto& observed = activity[bucket_index];
          std::printf(
              ",\"b%zu\":{\"n\":%d,\"capacity\":%d,\"active\":%u,\"submitted_eig\":%u,"
              "\"submitted_back\":%u,\"completed\":%u}",
              bucket_index, fixture.host.buckets[bucket_index].orbital_count,
              fixture.host.buckets[bucket_index].system_count, observed.active_count,
              observed.submitted_eigensolver_count, observed.submitted_backtransform_count,
              observed.completed_count);
        }
        print_latency_stats("compacted", compacted);
        print_latency_stats("uncompacted", uncompacted);
        std::printf(",\"match\":true}\n");
        std::fflush(stdout);
      }
    }
  }
  return 0;
}

/* Long steady-state loop at one fixed active pattern for Nsight Compute
 * profiling. The profile keeps a fixed mid-convergence canonical tier and
 * replays the compacted graph kLoops times so a profiler can attribute
 * eigensolver work to the compacted submission counts. */
int run_compaction_profile(std::int64_t batch_size, bool heterogeneous, std::int64_t active_total,
                           int loops) {
  DeviceFixture fixture;
  if (!fixture.create(make_benchmark_batch(batch_size, heterogeneous)) || !factor(fixture, 31u)) {
    return 1;
  }
  Gfn2EigensolverCompactedSolveGraph graph;
  const auto build = build_gfn2_compacted_eigensolver_graph_cuda(
      fixture.batch, fixture.host.buckets.data(),
      static_cast<std::int64_t>(fixture.host.buckets.size()), fixture.cache, 31u,
      fixture.hamiltonian.get(), Gfn2EigensolverOptions{}, fixture.providers.solver,
      fixture.providers.parameters, fixture.providers.blas, fixture.workspace, fixture.results,
      fixture.system_errors.get(), fixture.device_error.get(), graph);
  if (!build.success() || !graph.valid()) {
    return 1;
  }
  const std::vector<double> clean_hamiltonian = fixture.host.hamiltonian;
  for (std::int64_t system = 0; system < batch_size; ++system) {
    fixture.host.active[static_cast<std::size_t>(system)] = system < active_total ? 1u : 0u;
  }
  fixture.host.hamiltonian = clean_hamiltonian;
  for (std::int64_t system = 0; system < batch_size; ++system) {
    if (fixture.host.active[static_cast<std::size_t>(system)] == 0u) {
      const std::int64_t begin = fixture.host.matrix_offsets[static_cast<std::size_t>(system)];
      const std::int64_t end = fixture.host.matrix_offsets[static_cast<std::size_t>(system + 1)];
      std::fill(fixture.host.hamiltonian.begin() + begin, fixture.host.hamiltonian.begin() + end,
                std::numeric_limits<double>::quiet_NaN());
    }
  }
  if (!fixture.active.upload(fixture.host.active) ||
      !fixture.hamiltonian.upload(fixture.host.hamiltonian) || !fixture.fill_outputs(kSentinel)) {
    return 1;
  }
  if (!cuda_ok(reset_gfn2_eigensolver_device_errors_cuda(
                   fixture.host.batch_size, fixture.system_errors.get(), fixture.device_error.get(),
                   fixture.providers.stream),
               "reset compaction profile errors")) {
    return 1;
  }
  for (int loop = 0; loop < loops; ++loop) {
    if (!enqueue_compacted(fixture, graph)) {
      return 1;
    }
  }
  if (!cuda_ok(cudaStreamSynchronize(fixture.providers.stream), "profile synchronize")) {
    return 1;
  }
  if (!validate_compacted_launch(fixture)) {
    std::cerr << "compaction profile correctness failed\n";
    return 1;
  }
  std::cout << "compaction profile passed: batch=" << batch_size
            << " heterogeneous=" << heterogeneous << " active=" << active_total
            << " loops=" << loops << '\n';
  return 0;
}

/* Sanitizer control for one exact provider capacity: every system is active,
 * and the ordinary solve is launched directly on the stream without a CUDA
 * Graph. Pair a compacted profile such as heterogeneous B=128, active=64
 * (32 systems per AO bucket) with this control at B=64 (also 32 per bucket).
 * This keeps AO matrices and provider batch counts identical while varying
 * only Graph replay and the compacted staging path. */
int run_direct_solve_control(std::int64_t batch_size, bool heterogeneous, int loops) {
  DeviceFixture fixture;
  if (!fixture.create(make_benchmark_batch(batch_size, heterogeneous)) || !factor(fixture, 31u)) {
    return 1;
  }
  if (!fixture.fill_outputs(kSentinel)) {
    return 1;
  }
  if (!cuda_ok(reset_gfn2_eigensolver_device_errors_cuda(
                   fixture.host.batch_size, fixture.system_errors.get(), fixture.device_error.get(),
                   fixture.providers.stream),
               "reset direct solve control errors")) {
    return 1;
  }
  for (int loop = 0; loop < loops; ++loop) {
    if (!enqueue_uncompacted(fixture, 31u)) {
      return 1;
    }
  }
  if (!cuda_ok(cudaStreamSynchronize(fixture.providers.stream), "direct solve synchronize")) {
    return 1;
  }
  if (!validate_uncompacted_launch(fixture)) {
    std::cerr << "direct solve control correctness failed\n";
    return 1;
  }
  std::cout << "direct exact-capacity solve control passed: batch=" << batch_size
            << " heterogeneous=" << heterogeneous << " loops=" << loops << '\n';
  return 0;
}

template <typename Integer>
bool parse_integer(const char* text, Integer& value) {
  const char* end = text + std::strlen(text);
  const auto parsed = std::from_chars(text, end, value);
  return parsed.ec == std::errc{} && parsed.ptr == end;
}

bool parse_profile_arguments(char** argv, std::int64_t& batch, bool& heterogeneous,
                             std::int64_t& active, int& loops) {
  int heterogeneous_value = 0;
  if (!parse_integer(argv[2], batch) || !parse_integer(argv[3], heterogeneous_value) ||
      !parse_integer(argv[4], active) || !parse_integer(argv[5], loops) || batch < 1 ||
      (heterogeneous_value != 0 && heterogeneous_value != 1) || active < 0 || active > batch ||
      loops < 1) {
    return false;
  }
  heterogeneous = heterogeneous_value == 1;
  return true;
}

bool parse_direct_control_arguments(char** argv, std::int64_t& batch, bool& heterogeneous,
                                    int& loops) {
  int heterogeneous_value = 0;
  if (!parse_integer(argv[2], batch) || !parse_integer(argv[3], heterogeneous_value) ||
      !parse_integer(argv[4], loops) || batch < 1 ||
      (heterogeneous_value != 0 && heterogeneous_value != 1) || loops < 1) {
    return false;
  }
  heterogeneous = heterogeneous_value == 1;
  return true;
}

}  // namespace

#ifdef GPUXTB_COMPACTION_BENCHMARK_ONLY
int main() {
  int device_count = 0;
  if (!cuda_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount") || device_count == 0) {
    return 1;
  }
  return run_compaction_benchmark();
}
#else
int main(int argc, char** argv) {
  int device_count = 0;
  if (!cuda_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount") || device_count == 0) {
    return 1;
  }
  if (argc >= 2 && std::strcmp(argv[1], "--compaction-benchmark") == 0) {
    if (argc != 2) {
      std::cerr << "usage: --compaction-benchmark\n";
      return 1;
    }
    return run_compaction_benchmark();
  }
  if (argc >= 2 && std::strcmp(argv[1], "--compaction-profile") == 0) {
    if (argc != 6) {
      std::cerr << "usage: --compaction-profile <batch> <heterogeneous 0/1> <active> <loops>\n";
      return 1;
    }
    std::int64_t batch = 0;
    bool heterogeneous = false;
    std::int64_t active = 0;
    int loops = 0;
    if (!parse_profile_arguments(argv, batch, heterogeneous, active, loops)) {
      return 1;
    }
    return run_compaction_profile(batch, heterogeneous, active, loops);
  }
  if (argc >= 2 && std::strcmp(argv[1], "--direct-solve") == 0) {
    if (argc != 5) {
      std::cerr << "usage: --direct-solve <batch> <heterogeneous 0/1> <loops>\n";
      return 1;
    }
    std::int64_t batch = 0;
    bool heterogeneous = false;
    int loops = 0;
    if (!parse_direct_control_arguments(argv, batch, heterogeneous, loops)) {
      return 1;
    }
    return run_direct_solve_control(batch, heterogeneous, loops);
  }
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    if (!test_batch(batch_size, false)) {
      std::cerr << "batch test failed for size " << batch_size << '\n';
      return 1;
    }
  }
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    if (!test_compacted_graph_batch(batch_size, false)) {
      std::cerr << "compacted graph batch test failed for size " << batch_size << '\n';
      return 1;
    }
  }
  if (!test_spin_eigensolver_mixed_batches_and_transaction()) {
    std::cerr << "spin eigensolver batch/transaction test failed\n";
    return 1;
  }
  if (!test_batch(8, true) || !test_batch(1, false, 8) || !test_batch(8, false, 16) ||
      !test_batch(4, false, 32) || !test_compacted_graph_batch(8, true) ||
      !test_compacted_graph_filters_failed_peer() ||
      !test_compacted_graph_device_epoch_and_transactional_rebuild() ||
      !test_cpu_literal_parity() || !test_inactive_poison_is_skipped() ||
      !test_overlap_and_hamiltonian_validation() || !test_active_offset_and_singular_failures() ||
      !test_ill_conditioned_peer_isolation() || !test_cache_generation_staleness() ||
      !test_single_cache_member_peer_isolation() ||
      !test_sticky_error_and_invalid_bucket_map_fail_closed() ||
      !test_host_validation_aliases_and_limits() || !test_graph_capture()) {
    return 1;
  }
  std::cout << "CUDA eigensolver tests passed\n";
  return 0;
}
#endif
