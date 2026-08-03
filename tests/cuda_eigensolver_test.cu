#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_eigensolver.cuh"

namespace {

using gpuxtb::detail::cuda::factor_gfn2_overlap_cuda;
using gpuxtb::detail::cuda::Gfn2EigensolverBucket;
using gpuxtb::detail::cuda::Gfn2EigensolverDeviceBatch;
using gpuxtb::detail::cuda::Gfn2EigensolverDeviceError;
using gpuxtb::detail::cuda::Gfn2EigensolverDeviceResults;
using gpuxtb::detail::cuda::Gfn2EigensolverDeviceWorkspace;
using gpuxtb::detail::cuda::Gfn2EigensolverOptions;
using gpuxtb::detail::cuda::Gfn2EigensolverOverlapCache;
using gpuxtb::detail::cuda::Gfn2EigensolverWorkspaceRequirements;
using gpuxtb::detail::cuda::query_gfn2_eigensolver_bucket_workspace_cuda;
using gpuxtb::detail::cuda::reset_gfn2_eigensolver_device_errors_cuda;
using gpuxtb::detail::cuda::solve_gfn2_eigensystems_cuda;

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

}  // namespace

int main() {
  int device_count = 0;
  if (!cuda_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount") || device_count == 0) {
    return 1;
  }
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    if (!test_batch(batch_size, false)) {
      std::cerr << "batch test failed for size " << batch_size << '\n';
      return 1;
    }
  }
  if (!test_batch(8, true) || !test_batch(1, false, 8) || !test_batch(8, false, 16) ||
      !test_batch(4, false, 32) || !test_cpu_literal_parity() ||
      !test_inactive_poison_is_skipped() || !test_overlap_and_hamiltonian_validation() ||
      !test_active_offset_and_singular_failures() || !test_ill_conditioned_peer_isolation() ||
      !test_cache_generation_staleness() || !test_single_cache_member_peer_isolation() ||
      !test_sticky_error_and_invalid_bucket_map_fail_closed() ||
      !test_host_validation_aliases_and_limits() || !test_graph_capture()) {
    return 1;
  }
  std::cout << "CUDA eigensolver tests passed\n";
  return 0;
}
