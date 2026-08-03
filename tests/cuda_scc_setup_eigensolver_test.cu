#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_scc_setup_eigensolver.cuh"
#include "tests/support/gfn2_scc_test_case.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)
#define CUSOLVER_CHECK(expression) CHECK((expression) == CUSOLVER_STATUS_SUCCESS)
#define CUBLAS_CHECK(expression) CHECK((expression) == CUBLAS_STATUS_SUCCESS)

namespace {

using gpuxtb::detail::Gfn2PlanMemorySpace;
using gpuxtb::detail::Gfn2RaggedTopologyView;
using gpuxtb::detail::cuda::Gfn2EigensolverBucket;
using gpuxtb::detail::cuda::Gfn2EigensolverDeviceError;
using gpuxtb::detail::cuda::Gfn2EigensolverDeviceWorkspace;
using gpuxtb::detail::cuda::Gfn2SccIterationArenaRequirements;
using gpuxtb::detail::cuda::Gfn2SccIterationDevicePlan;
using gpuxtb::detail::cuda::Gfn2SccIterationDeviceState;
using gpuxtb::detail::cuda::Gfn2SccIterationDeviceWorkspace;
using gpuxtb::detail::cuda::Gfn2SccIterationReportStorage;
using gpuxtb::detail::cuda::Gfn2SccSetupEigensolver;
using gpuxtb::detail::cuda::Gfn2SccSetupEigensolverBinding;
using gpuxtb::detail::cuda::Gfn2SccSetupEigensolverError;
using gpuxtb::detail::cuda::Gfn2SccSetupEigensolverField;
using gpuxtb::detail::cuda::Gfn2SccSetupTopology;
using gpuxtb::test::gfn2::HostSccCase;
using gpuxtb::test::gfn2::HostSccCaseOptions;
using gpuxtb::test::gfn2::SmallSystemKind;

constexpr std::uint64_t kPlanToken = 0x243f6a8885a308d3ULL;
constexpr std::uint64_t kGeneration = 37u;

class DeviceAllocation {
 public:
  DeviceAllocation() = default;
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  ~DeviceAllocation() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
  }

  bool allocate(std::size_t bytes) {
    return bytes != 0u && cudaMalloc(&pointer_, bytes) == cudaSuccess;
  }

  void* get() const noexcept { return pointer_; }

 private:
  void* pointer_ = nullptr;
};

class PinnedAllocation {
 public:
  PinnedAllocation() = default;
  PinnedAllocation(const PinnedAllocation&) = delete;
  PinnedAllocation& operator=(const PinnedAllocation&) = delete;
  ~PinnedAllocation() {
    if (pointer_ != nullptr) {
      (void)cudaFreeHost(pointer_);
    }
  }

  bool allocate(std::size_t bytes) {
    bytes_ = bytes;
    return bytes == 0u || cudaMallocHost(&pointer_, bytes) == cudaSuccess;
  }

  void* get() const noexcept { return pointer_; }
  std::size_t bytes() const noexcept { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0u;
};

class ProviderHandles {
 public:
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
    return cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess &&
           cusolverDnCreate(&solver) == CUSOLVER_STATUS_SUCCESS &&
           cusolverDnCreateParams(&parameters) == CUSOLVER_STATUS_SUCCESS &&
           cublasCreate(&blas) == CUBLAS_STATUS_SUCCESS;
  }

  cudaStream_t stream = nullptr;
  cusolverDnHandle_t solver = nullptr;
  cusolverDnParams_t parameters = nullptr;
  cublasHandle_t blas = nullptr;
};

struct IterationWorkspaceFixture {
  Gfn2SccIterationDevicePlan plan{};
  Gfn2SccIterationArenaRequirements requirements{};
  Gfn2SccIterationDeviceState state{};
  Gfn2SccIterationDeviceWorkspace workspace{};
  Gfn2SccIterationReportStorage reports{};
  DeviceAllocation arena;
  PinnedAllocation host_workspace;

  bool create(const HostSccCase& host,
              const gpuxtb::detail::cuda::Gfn2SccSetupEigensolverRequirements& setup) {
    plan.abi_version = gpuxtb::detail::cuda::kGfn2SccIterationAbiVersion;
    plan.enabled_components = gpuxtb::detail::cuda::kGfn2SccPotentialAllComponents;
    plan.plan_token = kPlanToken;
    plan.topology.plan_token = kPlanToken;
    plan.topology.batch_size = host.batch_size();
    plan.topology.total_atoms = host.total_atoms();
    plan.topology.total_shells = host.basis_plan().total_shells;
    plan.topology.total_orbitals = host.wavefunction_layout().total_orbitals;
    plan.topology.total_matrix_elements = host.integral_plan().total_matrix_elements;
    plan.mixer_policy.history_size = 3;
    plan.geometry_batch.total_pairs = 0;
    plan.es2_batch.total_matrix_elements = 0;
    plan.aes2_batch.total_pairs = 0;
    plan.d4_batch.total_pairs = 0;
    plan.eigensolver_provider.requirements = setup.provider;
    if (!gpuxtb::detail::cuda::query_gfn2_scc_iteration_arena_requirements_cuda(
             plan, setup.provider, requirements)
             .success()) {
      return false;
    }
    if (!arena.allocate(requirements.total_bytes) ||
        !host_workspace.allocate(setup.provider.solver_host_workspace_bytes)) {
      return false;
    }
    plan.eigensolver_provider.requirements = {};
    return gpuxtb::detail::cuda::bind_gfn2_scc_iteration_arena_cuda(
               plan, setup.provider, requirements, arena.get(), requirements.total_bytes,
               host_workspace.get(), host_workspace.bytes(), state, workspace, reports)
        .success();
  }
};

template <typename T>
bool download(const T* device, std::size_t count, std::vector<T>& host, cudaStream_t stream) {
  host.resize(count);
  return count == 0u || cudaMemcpyAsync(host.data(), device, count * sizeof(T),
                                        cudaMemcpyDeviceToHost, stream) == cudaSuccess;
}

bool near(double first, double second, double tolerance = 3.0e-10) {
  const double scale = std::max({1.0, std::abs(first), std::abs(second)});
  return std::abs(first - second) <= tolerance * scale;
}

bool factors_reconstruct_overlap(const HostSccCase& host, const Gfn2SccSetupTopology& topology,
                                 const std::vector<double>& factors) {
  const Gfn2RaggedTopologyView& topology_view = topology.host_topology();
  const auto& buckets = topology.eigensolver_buckets();
  for (const Gfn2EigensolverBucket& bucket : buckets) {
    const std::int64_t n = bucket.orbital_count;
    const std::int64_t stride = n * n;
    for (std::int64_t local = 0; local < bucket.system_count; ++local) {
      const std::int64_t slot = bucket.system_index_offset + local;
      const std::int64_t system = topology_view.bucket_systems[slot];
      const std::int64_t input_begin = topology_view.matrix_offsets[system];
      const std::int64_t factor_begin = bucket.matrix_scratch_offset + local * stride;
      for (std::int64_t row = 0; row < n; ++row) {
        for (std::int64_t column = 0; column < n; ++column) {
          double reconstructed = 0.0;
          for (std::int64_t inner = 0; inner <= std::min(row, column); ++inner) {
            reconstructed += factors[static_cast<std::size_t>(factor_begin + row + inner * n)] *
                             factors[static_cast<std::size_t>(factor_begin + column + inner * n)];
          }
          if (!near(reconstructed,
                    host.overlap()[static_cast<std::size_t>(input_begin + row * n + column)])) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

std::pair<std::int64_t, std::int64_t> provider_factor_range_for_system(
    const Gfn2SccSetupTopology& topology, std::int64_t requested_system) {
  const Gfn2RaggedTopologyView& view = topology.host_topology();
  for (const Gfn2EigensolverBucket& bucket : topology.eigensolver_buckets()) {
    const std::int64_t stride = static_cast<std::int64_t>(bucket.orbital_count) *
                                static_cast<std::int64_t>(bucket.orbital_count);
    for (std::int64_t local = 0; local < bucket.system_count; ++local) {
      if (view.bucket_systems[bucket.system_index_offset + local] == requested_system) {
        const std::int64_t begin = bucket.matrix_scratch_offset + local * stride;
        return {begin, begin + stride};
      }
    }
  }
  return {-1, -1};
}

struct ProductionFixture {
  HostSccCase host;
  Gfn2SccSetupTopology topology;
  ProviderHandles handles;
  Gfn2SccSetupEigensolver setup;
  DeviceAllocation topology_arena;
  Gfn2RaggedTopologyView device_topology{};
  IterationWorkspaceFixture iteration;
  DeviceAllocation setup_arena;
  Gfn2SccSetupEigensolverBinding binding{};

  bool create() {
    HostSccCaseOptions options;
    options.systems = {SmallSystemKind::kH2, SmallSystemKind::kHe, SmallSystemKind::kLiH,
                       SmallSystemKind::kCH2};
    options.geometry_generation = kGeneration;
    std::string error;
    if (HostSccCase::create(options, host, error) != GPUXTB_STATUS_SUCCESS || !handles.create()) {
      return false;
    }
    if (!Gfn2SccSetupTopology::create(host.basis_plan(), host.integral_plan(),
                                      host.wavefunction_layout(), kPlanToken, topology)
             .success()) {
      return false;
    }
    if (!topology_arena.allocate(topology.requirements().immutable_device_bytes) ||
        !topology
             .bind_device_arena_and_upload_async(topology_arena.get(),
                                                 topology.requirements().immutable_device_bytes,
                                                 device_topology, handles.stream)
             .success()) {
      return false;
    }
    if (!Gfn2SccSetupEigensolver::create(
             topology, host.overlap().data(), static_cast<std::int64_t>(host.overlap().size()),
             kGeneration, kPlanToken, handles.solver, handles.parameters, handles.blas,
             gpuxtb::detail::cuda::Gfn2EigensolverOptions{}, setup)
             .success()) {
      return false;
    }
    if (!iteration.create(host, setup.requirements()) ||
        !setup_arena.allocate(setup.requirements().setup_device_bytes)) {
      return false;
    }
    /* #107 setup must not consume or overwrite the hot SCC activity ledger.
     * All-zero canonical activity still factors every setup member through its
     * private all-active batch. */
    if (cudaMemsetAsync(iteration.workspace.ledger.active_mask, 0,
                        static_cast<std::size_t>(host.batch_size()),
                        handles.stream) != cudaSuccess) {
      return false;
    }
    return setup
        .bind_and_factor_overlap_async(
            device_topology, iteration.plan, iteration.requirements, iteration.arena.get(),
            iteration.requirements.total_bytes, iteration.workspace, iteration.host_workspace.get(),
            iteration.host_workspace.bytes(), setup_arena.get(),
            setup.requirements().setup_device_bytes, binding, handles.stream)
        .success();
  }
};

int test_four_system_provider_and_cache() {
  ProductionFixture fixture;
  CHECK(fixture.create());
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));

  CHECK(fixture.binding.plan_token == kPlanToken);
  CHECK(fixture.binding.geometry_generation == kGeneration);
  CHECK(fixture.binding.provider.buckets != fixture.topology.eigensolver_buckets().data());
  CHECK(fixture.binding.provider.bucket_count ==
        static_cast<std::int64_t>(fixture.topology.eigensolver_buckets().size()));
  CHECK(fixture.binding.provider.solver == fixture.handles.solver);
  CHECK(fixture.binding.provider.parameters == fixture.handles.parameters);
  CHECK(fixture.binding.provider.blas == fixture.handles.blas);
  CHECK(fixture.binding.provider.device_workspace ==
        fixture.iteration.workspace.eigensolver_workspace.solver_device_workspace);
  CHECK(fixture.binding.provider.host_workspace == fixture.iteration.host_workspace.get());
  CHECK(fixture.binding.workspace.matrix_scratch_a ==
        fixture.iteration.workspace.eigensolver_workspace.matrix_scratch_a);
  CHECK(fixture.binding.batch.active == fixture.iteration.workspace.ledger.active_mask);
  CHECK(fixture.binding.options.minimum_overlap_rcond == 1.0e-12);
  CHECK(fixture.binding.iteration_layout_fingerprint ==
        fixture.iteration.requirements.layout_fingerprint);

  std::vector<std::uint32_t> errors;
  std::vector<std::uint8_t> canonical_active;
  std::vector<std::uint32_t> statuses;
  std::vector<std::uint64_t> generations;
  std::vector<double> factors;
  CHECK(download(fixture.binding.setup_system_errors,
                 static_cast<std::size_t>(fixture.host.batch_size()), errors,
                 fixture.handles.stream));
  CHECK(download(fixture.binding.batch.active, static_cast<std::size_t>(fixture.host.batch_size()),
                 canonical_active, fixture.handles.stream));
  CHECK(download(fixture.binding.cache.factor_statuses,
                 static_cast<std::size_t>(fixture.host.batch_size()), statuses,
                 fixture.handles.stream));
  CHECK(download(fixture.binding.cache.geometry_generations,
                 static_cast<std::size_t>(fixture.host.batch_size()), generations,
                 fixture.handles.stream));
  CHECK(download(fixture.binding.cache.cholesky_factors,
                 static_cast<std::size_t>(fixture.host.integral_plan().total_matrix_elements),
                 factors, fixture.handles.stream));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));
  CHECK(std::all_of(errors.begin(), errors.end(), [](std::uint32_t value) { return value == 0u; }));
  CHECK(std::all_of(canonical_active.begin(), canonical_active.end(),
                    [](std::uint8_t value) { return value == 0u; }));
  CHECK(std::all_of(statuses.begin(), statuses.end(),
                    [](std::uint32_t value) { return value == 0u; }));
  CHECK(std::all_of(generations.begin(), generations.end(),
                    [](std::uint64_t value) { return value == kGeneration; }));
  CHECK(factors_reconstruct_overlap(fixture.host, fixture.topology, factors));

  const Gfn2SccSetupEigensolverBinding first = fixture.binding;
  Gfn2SccSetupEigensolverBinding second{};
  CHECK(fixture.setup
            .bind_and_factor_overlap_async(
                fixture.device_topology, fixture.iteration.plan, fixture.iteration.requirements,
                fixture.iteration.arena.get(), fixture.iteration.requirements.total_bytes,
                fixture.iteration.workspace, fixture.iteration.host_workspace.get(),
                fixture.iteration.host_workspace.bytes(), fixture.setup_arena.get(),
                fixture.setup.requirements().setup_device_bytes, second, fixture.handles.stream)
            .success());
  CHECK(first.cache.cholesky_factors == second.cache.cholesky_factors);
  CHECK(first.workspace.matrix_scratch_a == second.workspace.matrix_scratch_a);
  CHECK(first.provider.device_workspace == second.provider.device_workspace);
  CHECK(first.provider.host_workspace == second.provider.host_workspace);
  CHECK(first.overlap_input == second.overlap_input);
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));
  return 0;
}

int test_bad_overlap_preserves_factor_and_reports_failure() {
  ProductionFixture fixture;
  CHECK(fixture.create());
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));

  std::vector<double> before;
  CHECK(download(fixture.binding.cache.cholesky_factors,
                 static_cast<std::size_t>(fixture.host.integral_plan().total_matrix_elements),
                 before, fixture.handles.stream));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));

  std::vector<double> bad_overlap = fixture.host.overlap();
  bad_overlap[0] = std::numeric_limits<double>::quiet_NaN();
  Gfn2SccSetupEigensolver bad;
  CHECK(Gfn2SccSetupEigensolver::create(
            fixture.topology, bad_overlap.data(), static_cast<std::int64_t>(bad_overlap.size()),
            kGeneration + 1u, kPlanToken, fixture.handles.solver, fixture.handles.parameters,
            fixture.handles.blas, gpuxtb::detail::cuda::Gfn2EigensolverOptions{}, bad)
            .success());
  Gfn2SccSetupEigensolverBinding bad_binding{};
  CHECK(bad.bind_and_factor_overlap_async(
               fixture.device_topology, fixture.iteration.plan, fixture.iteration.requirements,
               fixture.iteration.arena.get(), fixture.iteration.requirements.total_bytes,
               fixture.iteration.workspace, fixture.iteration.host_workspace.get(),
               fixture.iteration.host_workspace.bytes(), fixture.setup_arena.get(),
               bad.requirements().setup_device_bytes, bad_binding, fixture.handles.stream)
            .success());
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));

  std::vector<double> after;
  std::vector<std::uint32_t> statuses;
  std::vector<std::uint32_t> errors;
  std::vector<std::uint64_t> generations;
  CHECK(download(bad_binding.cache.cholesky_factors, before.size(), after, fixture.handles.stream));
  CHECK(download(bad_binding.cache.factor_statuses,
                 static_cast<std::size_t>(fixture.host.batch_size()), statuses,
                 fixture.handles.stream));
  CHECK(download(bad_binding.setup_system_errors,
                 static_cast<std::size_t>(fixture.host.batch_size()), errors,
                 fixture.handles.stream));
  CHECK(download(bad_binding.cache.geometry_generations,
                 static_cast<std::size_t>(fixture.host.batch_size()), generations,
                 fixture.handles.stream));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));

  const auto [begin, end] = provider_factor_range_for_system(fixture.topology, 0);
  CHECK(begin >= 0 && end > begin);
  CHECK(std::equal(before.begin() + begin, before.begin() + end, after.begin() + begin));
  const std::uint32_t expected =
      static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kNonfiniteOverlap);
  CHECK(errors[0] == expected);
  CHECK(statuses[0] == expected);
  CHECK(generations[0] == kGeneration + 1u);
  CHECK(bad_binding.cache.cholesky_factors == fixture.binding.cache.cholesky_factors);
  return 0;
}

int test_fail_closed_plan_generation_and_provenance() {
  ProductionFixture fixture;
  CHECK(fixture.create());
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));

  Gfn2SccSetupEigensolver preserved = std::move(fixture.setup);
  const std::uint64_t preserved_fingerprint = preserved.requirements().layout_fingerprint;
  auto diagnostic = Gfn2SccSetupEigensolver::create(
      fixture.topology, fixture.host.overlap().data(),
      static_cast<std::int64_t>(fixture.host.overlap().size()), 0u, kPlanToken,
      fixture.handles.solver, fixture.handles.parameters, fixture.handles.blas,
      gpuxtb::detail::cuda::Gfn2EigensolverOptions{}, preserved);
  CHECK(diagnostic.error == Gfn2SccSetupEigensolverError::kInvalidGeneration);
  CHECK(diagnostic.field == Gfn2SccSetupEigensolverField::kGeometryGeneration);
  CHECK(preserved.valid());
  CHECK(preserved.requirements().layout_fingerprint == preserved_fingerprint);

  diagnostic = Gfn2SccSetupEigensolver::create(
      fixture.topology, fixture.host.overlap().data(),
      static_cast<std::int64_t>(fixture.host.overlap().size()), kGeneration, kPlanToken, nullptr,
      fixture.handles.parameters, fixture.handles.blas,
      gpuxtb::detail::cuda::Gfn2EigensolverOptions{}, preserved);
  CHECK(diagnostic.error == Gfn2SccSetupEigensolverError::kInvalidProvider);
  CHECK(preserved.valid());

  Gfn2SccSetupEigensolverBinding sentinel{};
  sentinel.plan_token = 0xdeadbeefULL;
  Gfn2RaggedTopologyView cross_plan = fixture.device_topology;
  cross_plan.plan_token = kPlanToken + 1u;
  diagnostic = preserved.bind_and_factor_overlap_async(
      cross_plan, fixture.iteration.plan, fixture.iteration.requirements,
      fixture.iteration.arena.get(), fixture.iteration.requirements.total_bytes,
      fixture.iteration.workspace, fixture.iteration.host_workspace.get(),
      fixture.iteration.host_workspace.bytes(), fixture.setup_arena.get(),
      preserved.requirements().setup_device_bytes, sentinel, fixture.handles.stream);
  CHECK(diagnostic.error == Gfn2SccSetupEigensolverError::kCrossPlan);
  CHECK(sentinel.plan_token == 0xdeadbeefULL);

  Gfn2SccIterationArenaRequirements stale = fixture.iteration.requirements;
  stale.layout_fingerprint = 0u;
  diagnostic = preserved.bind_and_factor_overlap_async(
      fixture.device_topology, fixture.iteration.plan, stale, fixture.iteration.arena.get(),
      stale.total_bytes, fixture.iteration.workspace, fixture.iteration.host_workspace.get(),
      fixture.iteration.host_workspace.bytes(), fixture.setup_arena.get(),
      preserved.requirements().setup_device_bytes, sentinel, fixture.handles.stream);
  CHECK(diagnostic.error == Gfn2SccSetupEigensolverError::kInvalidIterationProvenance);
  CHECK(sentinel.plan_token == 0xdeadbeefULL);

  Gfn2SccIterationDeviceWorkspace escaped = fixture.iteration.workspace;
  escaped.eigensolver_workspace.matrix_scratch_a = static_cast<double*>(fixture.setup_arena.get());
  diagnostic = preserved.bind_and_factor_overlap_async(
      fixture.device_topology, fixture.iteration.plan, fixture.iteration.requirements,
      fixture.iteration.arena.get(), fixture.iteration.requirements.total_bytes, escaped,
      fixture.iteration.host_workspace.get(), fixture.iteration.host_workspace.bytes(),
      fixture.setup_arena.get(), preserved.requirements().setup_device_bytes, sentinel,
      fixture.handles.stream);
  CHECK(diagnostic.error == Gfn2SccSetupEigensolverError::kInvalidIterationProvenance);
  CHECK(sentinel.plan_token == 0xdeadbeefULL);
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

  int status = test_four_system_provider_and_cache();
  if (status != 0) {
    return status;
  }
  status = test_bad_overlap_preserves_factor_and_reports_failure();
  if (status != 0) {
    return status;
  }
  return test_fail_closed_plan_generation_and_provenance();
}
