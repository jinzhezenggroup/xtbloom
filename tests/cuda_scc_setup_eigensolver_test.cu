#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_device_admission.cuh"
#include "backends/cuda/gfn2_scc_setup_eigensolver.cuh"
#include "tests/support/gfn2_scc_test_case.hpp"

#define CHECK(condition)                                                                   \
  do {                                                                                     \
    if (!(condition)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      return __LINE__;                                                                     \
    }                                                                                      \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)
#define CUSOLVER_CHECK(expression) CHECK((expression) == CUSOLVER_STATUS_SUCCESS)
#define CUBLAS_CHECK(expression) CHECK((expression) == CUBLAS_STATUS_SUCCESS)

namespace {

using xtbloom::detail::Gfn2PlanMemorySpace;
using xtbloom::detail::Gfn2RaggedTopologyView;
using xtbloom::detail::cuda::Gfn2EigensolverBucket;
using xtbloom::detail::cuda::Gfn2EigensolverDeviceError;
using xtbloom::detail::cuda::Gfn2EigensolverDeviceWorkspace;
using xtbloom::detail::cuda::Gfn2EigensolverOptions;
using xtbloom::detail::cuda::Gfn2EigensolverStrategy;
using xtbloom::detail::cuda::Gfn2GeometryEpochDevice;
using xtbloom::detail::cuda::Gfn2SccIterationArenaRequirements;
using xtbloom::detail::cuda::Gfn2SccIterationDevicePlan;
using xtbloom::detail::cuda::Gfn2SccIterationDeviceState;
using xtbloom::detail::cuda::Gfn2SccIterationDeviceWorkspace;
using xtbloom::detail::cuda::Gfn2SccIterationReportStorage;
using xtbloom::detail::cuda::Gfn2SccSetupEigensolver;
using xtbloom::detail::cuda::Gfn2SccSetupEigensolverBinding;
using xtbloom::detail::cuda::Gfn2SccSetupEigensolverError;
using xtbloom::detail::cuda::Gfn2SccSetupEigensolverField;
using xtbloom::detail::cuda::Gfn2SccSetupTopology;
using xtbloom::detail::cuda::kGfn2RequestErrorWarmIncompatible;
using xtbloom::test::gfn2::HostSccCase;
using xtbloom::test::gfn2::HostSccCaseOptions;
using xtbloom::test::gfn2::SmallSystemKind;

constexpr std::uint64_t kPlanToken = 0x243f6a8885a308d3ULL;
constexpr std::uint64_t kGeneration = 37u;

__global__ void advance_epoch_kernel(std::uint64_t* epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(epoch), 1ULL);
  }
}

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
    if (bytes == 0u || cudaMalloc(&pointer_, bytes) != cudaSuccess) {
      return false;
    }
    return true;
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

class GraphResources {
 public:
  GraphResources() = default;
  GraphResources(const GraphResources&) = delete;
  GraphResources& operator=(const GraphResources&) = delete;
  ~GraphResources() {
    if (executable != nullptr) {
      (void)cudaGraphExecDestroy(executable);
    }
    if (graph != nullptr) {
      (void)cudaGraphDestroy(graph);
    }
  }

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
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
              const xtbloom::detail::Gfn2WavefunctionLayoutView& wavefunction,
              const xtbloom::detail::cuda::Gfn2SccSetupEigensolverRequirements& setup,
              std::int64_t bucket_count) {
    plan.abi_version = xtbloom::detail::cuda::kGfn2SccIterationAbiVersion;
    plan.enabled_components = xtbloom::detail::cuda::kGfn2SccPotentialAllComponents;
    plan.plan_token = kPlanToken;
    plan.topology.plan_token = kPlanToken;
    plan.topology.batch_size = host.batch_size();
    plan.topology.bucket_count = bucket_count;
    plan.topology.total_atoms = host.total_atoms();
    plan.topology.total_shells = host.basis_plan().total_shells;
    plan.topology.total_orbitals = host.wavefunction_layout().total_orbitals;
    plan.topology.total_matrix_elements = host.integral_plan().total_matrix_elements;
    plan.wavefunction_layout = wavefunction;
    /* Arena sizing only consumes the immutable shape here; device upload and
     * numerical launch are owned by the setup-topology fixture below. */
    plan.wavefunction_layout.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
    plan.spin_batch.batch_size = host.batch_size();
    plan.spin_batch.total_atoms = host.total_atoms();
    plan.spin_batch.total_shells = host.basis_plan().total_shells;
    plan.spin_batch.shell_population_elements = wavefunction.total_spin_shells;
    plan.spin_batch.plan_token = kPlanToken;
    plan.mixer_policy.history_size = 3;
    plan.geometry_batch.total_pairs = 0;
    plan.es2_batch.total_matrix_elements = 0;
    plan.aes2_batch.total_pairs = 0;
    plan.d4_batch.total_pairs = 0;
    plan.eigensolver_provider.requirements = setup.provider;
    if (!xtbloom::detail::cuda::query_gfn2_scc_iteration_arena_requirements_cuda(
             plan, setup.provider, requirements)
             .success()) {
      return false;
    }
    if (!arena.allocate(requirements.total_bytes) ||
        !host_workspace.allocate(setup.provider.solver_host_workspace_bytes)) {
      return false;
    }
    plan.eigensolver_provider.requirements = {};
    return xtbloom::detail::cuda::bind_gfn2_scc_iteration_arena_cuda(
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

bool factors_reconstruct_overlap(const std::vector<double>& overlap,
                                 const Gfn2SccSetupTopology& topology,
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
                    overlap[static_cast<std::size_t>(input_begin + row * n + column)])) {
            std::fprintf(stderr,
                         "factor mismatch: system=%lld n=%lld row=%lld column=%lld "
                         "actual=%.17g expected=%.17g\n",
                         static_cast<long long>(system), static_cast<long long>(n),
                         static_cast<long long>(row), static_cast<long long>(column), reconstructed,
                         overlap[static_cast<std::size_t>(input_begin + row * n + column)]);
            return false;
          }
        }
      }
    }
  }
  return true;
}

bool factor_reconstructs_system(const std::vector<double>& overlap,
                                const Gfn2SccSetupTopology& topology,
                                const std::vector<double>& factors, std::int64_t requested_system) {
  const Gfn2RaggedTopologyView& view = topology.host_topology();
  for (const Gfn2EigensolverBucket& bucket : topology.eigensolver_buckets()) {
    const std::int64_t n = bucket.orbital_count;
    const std::int64_t stride = n * n;
    for (std::int64_t local = 0; local < bucket.system_count; ++local) {
      const std::int64_t slot = bucket.system_index_offset + local;
      const std::int64_t system = view.bucket_systems[slot];
      if (system != requested_system) {
        continue;
      }
      const std::int64_t input_begin = view.matrix_offsets[system];
      const std::int64_t factor_begin = bucket.matrix_scratch_offset + local * stride;
      for (std::int64_t row = 0; row < n; ++row) {
        for (std::int64_t column = 0; column < n; ++column) {
          double reconstructed = 0.0;
          for (std::int64_t inner = 0; inner <= std::min(row, column); ++inner) {
            reconstructed += factors[static_cast<std::size_t>(factor_begin + row + inner * n)] *
                             factors[static_cast<std::size_t>(factor_begin + column + inner * n)];
          }
          if (!near(reconstructed,
                    overlap[static_cast<std::size_t>(input_begin + row * n + column)])) {
            return false;
          }
        }
      }
      return true;
    }
  }
  return false;
}

std::vector<double> changed_overlap(const HostSccCase& host, const Gfn2SccSetupTopology& topology,
                                    double scale) {
  std::vector<double> result = host.overlap();
  const Gfn2RaggedTopologyView& view = topology.host_topology();
  for (std::int64_t system = 0; system < view.batch_size; ++system) {
    const std::int64_t orbitals =
        view.batch_orbital_offsets[system + 1] - view.batch_orbital_offsets[system];
    const std::int64_t matrix_begin = view.matrix_offsets[system];
    const double shift = scale * static_cast<double>(system + 1);
    for (std::int64_t orbital = 0; orbital < orbitals; ++orbital) {
      result[static_cast<std::size_t>(matrix_begin + orbital * orbitals + orbital)] += shift;
    }
  }
  return result;
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

  bool create(std::int64_t batch_size = 4,
              const Gfn2EigensolverOptions& eigensolver_options = Gfn2EigensolverOptions{}) {
    HostSccCaseOptions options;
    constexpr std::array<SmallSystemKind, 4> pattern{SmallSystemKind::kH2, SmallSystemKind::kHe,
                                                     SmallSystemKind::kLiH, SmallSystemKind::kCH2};
    /* HostSccCaseOptions carries a one-system convenience default. Replace it
     * instead of appending, otherwise a requested batch N silently becomes
     * N+1 and the last peer is left outside activity/generation assertions. */
    options.systems.clear();
    options.systems.reserve(static_cast<std::size_t>(batch_size));
    for (std::int64_t system = 0; system < batch_size; ++system) {
      options.systems.push_back(pattern[static_cast<std::size_t>(system) % pattern.size()]);
    }
    options.geometry_generation = kGeneration;
    std::string error;
    if (HostSccCase::create(options, host, error) != XTBLOOM_STATUS_SUCCESS || !handles.create()) {
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
             eigensolver_options, setup)
             .success()) {
      return false;
    }
    if (!iteration.create(host, topology.host_wavefunction_layout(), setup.requirements(),
                          static_cast<std::int64_t>(topology.eigensolver_buckets().size())) ||
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
    const auto diagnostic = setup.bind_and_factor_overlap_async(
        device_topology, iteration.plan, iteration.requirements, iteration.arena.get(),
        iteration.requirements.total_bytes, iteration.workspace, iteration.host_workspace.get(),
        iteration.host_workspace.bytes(), setup_arena.get(),
        setup.requirements().setup_device_bytes, binding, handles.stream);
    if (!diagnostic.success()) {
      std::fprintf(stderr, "setup eigensolver bind failed: error=%u field=%u index=%lld\n",
                   static_cast<unsigned>(diagnostic.error), static_cast<unsigned>(diagnostic.field),
                   static_cast<long long>(diagnostic.index));
    }
    return diagnostic.success();
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
  CHECK(fixture.binding.workspace.compact_systems ==
        fixture.iteration.workspace.eigensolver_workspace.compact_systems);
  CHECK(fixture.binding.workspace.compact_source_slots ==
        fixture.iteration.workspace.eigensolver_workspace.compact_source_slots);
  CHECK(fixture.binding.workspace.bucket_activity ==
        fixture.iteration.workspace.eigensolver_workspace.bucket_activity);
  CHECK(fixture.binding.workspace.compact_system_elements == fixture.host.batch_size());
  CHECK(fixture.binding.workspace.bucket_activity_elements ==
        fixture.binding.provider.bucket_count);
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
  CHECK(factors_reconstruct_overlap(fixture.host.overlap(), fixture.topology, factors));

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

int test_forced_tridiagonal_strategy_is_valid() {
  Gfn2EigensolverOptions options{};
  options.strategy = Gfn2EigensolverStrategy::kTridiagonalBisection;
  ProductionFixture fixture;
  CHECK(fixture.create(4, options));
  CHECK(fixture.setup.valid());
  return 0;
}

int test_device_overlap_refactor_batches_and_fail_closed_aliases() {
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    ProductionFixture fixture;
    CHECK(fixture.create(batch_size));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));

    const std::vector<double> changed = changed_overlap(fixture.host, fixture.topology, 0.004);
    DeviceAllocation device_overlap;
    CHECK(device_overlap.allocate(changed.size() * sizeof(double)));
    CUDA_CHECK(cudaMemcpyAsync(device_overlap.get(), changed.data(),
                               changed.size() * sizeof(double), cudaMemcpyHostToDevice,
                               fixture.handles.stream));

    /* A synchronous provenance/alias rejection must enqueue no diagnostic
     * reset and must not advance the host-side generation descriptor. */
    CUDA_CHECK(cudaMemsetAsync(fixture.binding.setup_system_errors, 0x5a,
                               static_cast<std::size_t>(batch_size) * sizeof(std::uint32_t),
                               fixture.handles.stream));
    CUDA_CHECK(cudaMemsetAsync(fixture.binding.setup_device_error, 0x5a, sizeof(std::uint32_t),
                               fixture.handles.stream));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));
    const std::uint64_t previous_generation = fixture.binding.geometry_generation;
    auto diagnostic = fixture.setup.refactor_overlap_from_device_async(
        fixture.setup_arena.get(), fixture.setup.requirements().setup_device_bytes, fixture.binding,
        fixture.binding.cache.cholesky_factors, fixture.binding.cache.factor_elements,
        kGeneration + 1u, fixture.handles.stream);
    CHECK(diagnostic.error == Gfn2SccSetupEigensolverError::kInvalidOverlap);
    CHECK(fixture.binding.geometry_generation == previous_generation);
    std::vector<std::uint32_t> untouched_errors;
    CHECK(download(fixture.binding.setup_system_errors, static_cast<std::size_t>(batch_size),
                   untouched_errors, fixture.handles.stream));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));
    CHECK(std::all_of(untouched_errors.begin(), untouched_errors.end(),
                      [](std::uint32_t value) { return value == UINT32_C(0x5a5a5a5a); }));

    Gfn2SccSetupEigensolverBinding forged = fixture.binding;
    forged.owner_identity = nullptr;
    diagnostic = fixture.setup.refactor_overlap_from_device_async(
        fixture.setup_arena.get(), fixture.setup.requirements().setup_device_bytes, forged,
        static_cast<const double*>(device_overlap.get()), static_cast<std::int64_t>(changed.size()),
        kGeneration + 1u, fixture.handles.stream);
    CHECK(diagnostic.error == Gfn2SccSetupEigensolverError::kInvalidIterationProvenance);

    forged = fixture.binding;
    forged.workspace.matrix_scratch_a = static_cast<double*>(device_overlap.get());
    diagnostic = fixture.setup.refactor_overlap_from_device_async(
        fixture.setup_arena.get(), fixture.setup.requirements().setup_device_bytes, forged,
        static_cast<const double*>(device_overlap.get()), static_cast<std::int64_t>(changed.size()),
        kGeneration + 1u, fixture.handles.stream);
    CHECK(diagnostic.error == Gfn2SccSetupEigensolverError::kInvalidIterationProvenance);

    diagnostic = fixture.setup.refactor_overlap_from_device_async(
        fixture.setup_arena.get(), fixture.setup.requirements().setup_device_bytes, fixture.binding,
        static_cast<const double*>(device_overlap.get()), static_cast<std::int64_t>(changed.size()),
        previous_generation, fixture.handles.stream);
    CHECK(diagnostic.error == Gfn2SccSetupEigensolverError::kInvalidGeneration);
    CHECK(fixture.binding.geometry_generation == previous_generation);

    diagnostic = fixture.setup.refactor_overlap_from_device_async(
        fixture.setup_arena.get(), fixture.setup.requirements().setup_device_bytes, fixture.binding,
        static_cast<const double*>(device_overlap.get()), static_cast<std::int64_t>(changed.size()),
        kGeneration + 1u, fixture.handles.stream);
    CHECK(diagnostic.success());
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));
    CHECK(fixture.binding.geometry_generation == kGeneration + 1u);

    std::vector<std::uint32_t> errors;
    std::vector<std::uint32_t> statuses;
    std::vector<std::uint64_t> generations;
    std::vector<double> factors;
    CHECK(download(fixture.binding.setup_system_errors, static_cast<std::size_t>(batch_size),
                   errors, fixture.handles.stream));
    CHECK(download(fixture.binding.cache.factor_statuses, static_cast<std::size_t>(batch_size),
                   statuses, fixture.handles.stream));
    CHECK(download(fixture.binding.cache.geometry_generations, static_cast<std::size_t>(batch_size),
                   generations, fixture.handles.stream));
    CHECK(download(fixture.binding.cache.cholesky_factors, changed.size(), factors,
                   fixture.handles.stream));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));
    CHECK(
        std::all_of(errors.begin(), errors.end(), [](std::uint32_t value) { return value == 0u; }));
    CHECK(std::all_of(statuses.begin(), statuses.end(),
                      [](std::uint32_t value) { return value == 0u; }));
    CHECK(std::all_of(generations.begin(), generations.end(),
                      [](std::uint64_t value) { return value == kGeneration + 1u; }));
    CHECK(factors_reconstruct_overlap(changed, fixture.topology, factors));

    /* Repeat with the same allocation and a new value. Batch one uses the
     * default stream while the remaining cases exercise the caller's custom
     * nonblocking stream. */
    const std::vector<double> repeated = changed_overlap(fixture.host, fixture.topology, 0.008);
    cudaStream_t refactor_stream = fixture.handles.stream;
    if (batch_size == 1) {
      CUDA_CHECK(cudaMemcpy(device_overlap.get(), repeated.data(), repeated.size() * sizeof(double),
                            cudaMemcpyHostToDevice));
      refactor_stream = nullptr;
    } else {
      CUDA_CHECK(cudaMemcpyAsync(device_overlap.get(), repeated.data(),
                                 repeated.size() * sizeof(double), cudaMemcpyHostToDevice,
                                 refactor_stream));
    }
    diagnostic = fixture.setup.refactor_overlap_from_device_async(
        fixture.setup_arena.get(), fixture.setup.requirements().setup_device_bytes, fixture.binding,
        static_cast<const double*>(device_overlap.get()),
        static_cast<std::int64_t>(repeated.size()), kGeneration + 2u, refactor_stream);
    CHECK(diagnostic.success());
    if (refactor_stream == nullptr) {
      CUDA_CHECK(cudaDeviceSynchronize());
    } else {
      CUDA_CHECK(cudaStreamSynchronize(refactor_stream));
    }
    CHECK(download(fixture.binding.cache.cholesky_factors, repeated.size(), factors,
                   fixture.handles.stream));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));
    CHECK(factors_reconstruct_overlap(repeated, fixture.topology, factors));
  }
  return 0;
}

int test_device_overlap_refactor_peer_failure() {
  constexpr std::int64_t kFailedSystem = 3;
  ProductionFixture fixture;
  CHECK(fixture.create(8));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));

  std::vector<double> before;
  CHECK(download(fixture.binding.cache.cholesky_factors,
                 static_cast<std::size_t>(fixture.host.integral_plan().total_matrix_elements),
                 before, fixture.handles.stream));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));

  std::vector<double> changed = changed_overlap(fixture.host, fixture.topology, 0.006);
  const std::int64_t failed_begin = fixture.topology.host_topology().matrix_offsets[kFailedSystem];
  changed[static_cast<std::size_t>(failed_begin)] = std::numeric_limits<double>::quiet_NaN();
  DeviceAllocation device_overlap;
  DeviceAllocation device_epoch;
  DeviceAllocation request_error;
  CHECK(device_overlap.allocate(changed.size() * sizeof(double)));
  CHECK(device_epoch.allocate(sizeof(std::uint64_t)));
  CHECK(request_error.allocate(sizeof(std::uint32_t)));
  const std::uint64_t attempted_epoch = kGeneration + 2u;
  const std::uint32_t admitted = 0u;
  std::vector<std::uint8_t> active(static_cast<std::size_t>(fixture.host.batch_size()), 1u);
  CUDA_CHECK(cudaMemcpyAsync(device_overlap.get(), changed.data(), changed.size() * sizeof(double),
                             cudaMemcpyHostToDevice, fixture.handles.stream));
  CUDA_CHECK(cudaMemcpyAsync(device_epoch.get(), &attempted_epoch, sizeof(attempted_epoch),
                             cudaMemcpyHostToDevice, fixture.handles.stream));
  CUDA_CHECK(cudaMemcpyAsync(request_error.get(), &admitted, sizeof(admitted),
                             cudaMemcpyHostToDevice, fixture.handles.stream));
  CUDA_CHECK(cudaMemcpyAsync(const_cast<std::uint8_t*>(fixture.binding.batch.active), active.data(),
                             active.size(), cudaMemcpyHostToDevice, fixture.handles.stream));
  const Gfn2GeometryEpochDevice epoch{static_cast<std::uint64_t*>(device_epoch.get()), 1,
                                      kPlanToken};
  CHECK(fixture.setup
            .refactor_overlap_from_device_epoch_async(
                fixture.setup_arena.get(), fixture.setup.requirements().setup_device_bytes,
                fixture.binding, static_cast<const double*>(device_overlap.get()),
                static_cast<std::int64_t>(changed.size()), epoch,
                static_cast<const std::uint32_t*>(request_error.get()), fixture.handles.stream)
            .success());
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));

  std::vector<double> after;
  std::vector<std::uint32_t> errors;
  std::vector<std::uint32_t> statuses;
  std::vector<std::uint64_t> generations;
  CHECK(download(fixture.binding.cache.cholesky_factors, before.size(), after,
                 fixture.handles.stream));
  CHECK(download(fixture.binding.setup_system_errors,
                 static_cast<std::size_t>(fixture.host.batch_size()), errors,
                 fixture.handles.stream));
  CHECK(download(fixture.binding.cache.factor_statuses,
                 static_cast<std::size_t>(fixture.host.batch_size()), statuses,
                 fixture.handles.stream));
  CHECK(download(fixture.binding.cache.geometry_generations,
                 static_cast<std::size_t>(fixture.host.batch_size()), generations,
                 fixture.handles.stream));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));

  const std::uint32_t expected =
      static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kNonfiniteOverlap);
  const auto [factor_begin, factor_end] =
      provider_factor_range_for_system(fixture.topology, kFailedSystem);
  CHECK(factor_begin >= 0 && factor_end > factor_begin);
  CHECK(std::equal(before.begin() + factor_begin, before.begin() + factor_end,
                   after.begin() + factor_begin));
  for (std::int64_t system = 0; system < fixture.host.batch_size(); ++system) {
    const std::size_t index = static_cast<std::size_t>(system);
    if (system == kFailedSystem) {
      CHECK(errors[index] == expected);
      CHECK(statuses[index] == 0u);
      CHECK(generations[index] == kGeneration);
    } else {
      CHECK(errors[index] == 0u);
      CHECK(statuses[index] == 0u);
      CHECK(generations[index] == kGeneration + 2u);
      CHECK(factor_reconstructs_system(changed, fixture.topology, after, system));
    }
  }

  /* Reusing the same device epoch is a stale replay, even if the overlap bytes
   * change. Every peer keeps its previously committed factor/generation. */
  CHECK(fixture.setup
            .refactor_overlap_from_device_epoch_async(
                fixture.setup_arena.get(), fixture.setup.requirements().setup_device_bytes,
                fixture.binding, static_cast<const double*>(device_overlap.get()),
                static_cast<std::int64_t>(changed.size()), epoch,
                static_cast<const std::uint32_t*>(request_error.get()), fixture.handles.stream)
            .success());
  std::vector<double> stale_factors;
  std::vector<std::uint32_t> stale_errors;
  std::vector<std::uint64_t> stale_generations;
  CHECK(download(fixture.binding.cache.cholesky_factors, after.size(), stale_factors,
                 fixture.handles.stream));
  CHECK(download(fixture.binding.setup_system_errors,
                 static_cast<std::size_t>(fixture.host.batch_size()), stale_errors,
                 fixture.handles.stream));
  CHECK(download(fixture.binding.cache.geometry_generations,
                 static_cast<std::size_t>(fixture.host.batch_size()), stale_generations,
                 fixture.handles.stream));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));
  CHECK(stale_factors == after);
  CHECK(stale_generations == generations);
  for (std::int64_t system = 0; system < fixture.host.batch_size(); ++system) {
    const auto expected_error = system == kFailedSystem
                                    ? Gfn2EigensolverDeviceError::kNonfiniteOverlap
                                    : Gfn2EigensolverDeviceError::kStaleOverlapCache;
    CHECK(stale_errors[static_cast<std::size_t>(system)] ==
          static_cast<std::uint32_t>(expected_error));
  }
  return 0;
}

int test_device_overlap_refactor_graph_replay() {
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    ProductionFixture fixture;
    CHECK(fixture.create(batch_size));
    CHECK(fixture.host.batch_size() == batch_size);
    CHECK(fixture.topology.host_topology().batch_size == batch_size);
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));
    const std::vector<double> first = changed_overlap(fixture.host, fixture.topology, 0.003);
    const std::vector<double> second = changed_overlap(fixture.host, fixture.topology, 0.009);
    DeviceAllocation device_overlap;
    DeviceAllocation device_epoch;
    DeviceAllocation request_error;
    CHECK(device_overlap.allocate(first.size() * sizeof(double)));
    CHECK(device_epoch.allocate(sizeof(std::uint64_t)));
    CHECK(request_error.allocate(sizeof(std::uint32_t)));
    const std::uint64_t initial_epoch = kGeneration;
    const std::uint32_t admitted = 0u;
    std::vector<std::uint8_t> active(static_cast<std::size_t>(batch_size), 1u);
    CUDA_CHECK(cudaMemcpyAsync(device_epoch.get(), &initial_epoch, sizeof(initial_epoch),
                               cudaMemcpyHostToDevice, fixture.handles.stream));
    CUDA_CHECK(cudaMemcpyAsync(request_error.get(), &admitted, sizeof(admitted),
                               cudaMemcpyHostToDevice, fixture.handles.stream));
    CUDA_CHECK(cudaMemcpyAsync(const_cast<std::uint8_t*>(fixture.binding.batch.active),
                               active.data(), active.size(), cudaMemcpyHostToDevice,
                               fixture.handles.stream));
    const Gfn2GeometryEpochDevice epoch{static_cast<std::uint64_t*>(device_epoch.get()), 1,
                                        kPlanToken};

    GraphResources graph;
    CUDA_CHECK(cudaStreamBeginCapture(fixture.handles.stream, cudaStreamCaptureModeThreadLocal));
    advance_epoch_kernel<<<1, 1, 0, fixture.handles.stream>>>(epoch.value);
    CUDA_CHECK(cudaPeekAtLastError());
    const auto diagnostic = fixture.setup.refactor_overlap_from_device_epoch_async(
        fixture.setup_arena.get(), fixture.setup.requirements().setup_device_bytes, fixture.binding,
        static_cast<const double*>(device_overlap.get()), static_cast<std::int64_t>(first.size()),
        epoch, static_cast<const std::uint32_t*>(request_error.get()), fixture.handles.stream);
    const cudaError_t end_capture = cudaStreamEndCapture(fixture.handles.stream, &graph.graph);
    CHECK(diagnostic.success());
    CUDA_CHECK(end_capture);
    CUDA_CHECK(cudaGraphInstantiate(&graph.executable, graph.graph, 0));
    CHECK(fixture.binding.geometry_epoch == epoch.value);

    CUDA_CHECK(cudaMemcpyAsync(device_overlap.get(), first.data(), first.size() * sizeof(double),
                               cudaMemcpyHostToDevice, fixture.handles.stream));
    CUDA_CHECK(cudaGraphLaunch(graph.executable, fixture.handles.stream));
    std::vector<double> factors;
    std::vector<double> uploaded_overlap;
    std::vector<std::uint8_t> actual_active;
    std::vector<std::uint64_t> generations;
    std::uint64_t actual_epoch = 0u;
    CHECK(download(fixture.binding.cache.cholesky_factors, first.size(), factors,
                   fixture.handles.stream));
    CHECK(download(static_cast<const double*>(device_overlap.get()), first.size(), uploaded_overlap,
                   fixture.handles.stream));
    CHECK(download(fixture.binding.batch.active, static_cast<std::size_t>(batch_size),
                   actual_active, fixture.handles.stream));
    CHECK(download(fixture.binding.cache.geometry_generations, static_cast<std::size_t>(batch_size),
                   generations, fixture.handles.stream));
    CUDA_CHECK(cudaMemcpyAsync(&actual_epoch, epoch.value, sizeof(actual_epoch),
                               cudaMemcpyDeviceToHost, fixture.handles.stream));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));
    CHECK(actual_epoch == kGeneration + 1u);
    CHECK(uploaded_overlap == first);
    CHECK(actual_active == active);
    CHECK(std::all_of(generations.begin(), generations.end(),
                      [](std::uint64_t value) { return value == kGeneration + 1u; }));
    CHECK(factors_reconstruct_overlap(first, fixture.topology, factors));

    CUDA_CHECK(cudaMemcpyAsync(device_overlap.get(), second.data(), second.size() * sizeof(double),
                               cudaMemcpyHostToDevice, fixture.handles.stream));
    CUDA_CHECK(cudaGraphLaunch(graph.executable, fixture.handles.stream));
    CHECK(download(fixture.binding.cache.cholesky_factors, second.size(), factors,
                   fixture.handles.stream));
    CHECK(download(fixture.binding.cache.geometry_generations, static_cast<std::size_t>(batch_size),
                   generations, fixture.handles.stream));
    CUDA_CHECK(cudaMemcpyAsync(&actual_epoch, epoch.value, sizeof(actual_epoch),
                               cudaMemcpyDeviceToHost, fixture.handles.stream));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));
    CHECK(actual_epoch == kGeneration + 2u);
    CHECK(std::all_of(generations.begin(), generations.end(),
                      [](std::uint64_t value) { return value == kGeneration + 2u; }));
    CHECK(factors_reconstruct_overlap(second, fixture.topology, factors));
  }
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
            fixture.handles.blas, xtbloom::detail::cuda::Gfn2EigensolverOptions{}, bad)
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
      xtbloom::detail::cuda::Gfn2EigensolverOptions{}, preserved);
  CHECK(diagnostic.error == Gfn2SccSetupEigensolverError::kInvalidGeneration);
  CHECK(diagnostic.field == Gfn2SccSetupEigensolverField::kGeometryGeneration);
  CHECK(preserved.valid());
  CHECK(preserved.requirements().layout_fingerprint == preserved_fingerprint);

  diagnostic = Gfn2SccSetupEigensolver::create(
      fixture.topology, fixture.host.overlap().data(),
      static_cast<std::int64_t>(fixture.host.overlap().size()), kGeneration, kPlanToken, nullptr,
      fixture.handles.parameters, fixture.handles.blas,
      xtbloom::detail::cuda::Gfn2EigensolverOptions{}, preserved);
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

int test_device_epoch_admission_validation_and_rejection() {
  ProductionFixture fixture;
  CHECK(fixture.create());
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));
  const std::vector<double> changed = changed_overlap(fixture.host, fixture.topology, 0.004);
  DeviceAllocation device_overlap;
  DeviceAllocation device_epoch;
  DeviceAllocation request_error;
  CHECK(device_overlap.allocate(changed.size() * sizeof(double)));
  CHECK(device_epoch.allocate(sizeof(std::uint64_t)));
  CHECK(request_error.allocate(sizeof(std::uint32_t)));
  const std::uint64_t epoch_value = kGeneration + 1u;
  const std::uint32_t rejected = kGfn2RequestErrorWarmIncompatible;
  CUDA_CHECK(cudaMemcpyAsync(device_overlap.get(), changed.data(), changed.size() * sizeof(double),
                             cudaMemcpyHostToDevice, fixture.handles.stream));
  CUDA_CHECK(cudaMemcpyAsync(device_epoch.get(), &epoch_value, sizeof(epoch_value),
                             cudaMemcpyHostToDevice, fixture.handles.stream));
  CUDA_CHECK(cudaMemcpyAsync(request_error.get(), &rejected, sizeof(rejected),
                             cudaMemcpyHostToDevice, fixture.handles.stream));
  std::vector<std::uint8_t> active(static_cast<std::size_t>(fixture.host.batch_size()), 1u);
  CUDA_CHECK(cudaMemcpyAsync(const_cast<std::uint8_t*>(fixture.binding.batch.active), active.data(),
                             active.size(), cudaMemcpyHostToDevice, fixture.handles.stream));
  const Gfn2GeometryEpochDevice epoch{static_cast<std::uint64_t*>(device_epoch.get()), 1,
                                      kPlanToken};
  std::vector<double> before;
  CHECK(download(fixture.binding.cache.cholesky_factors, changed.size(), before,
                 fixture.handles.stream));
  CHECK(fixture.setup
            .refactor_overlap_from_device_epoch_async(
                fixture.setup_arena.get(), fixture.setup.requirements().setup_device_bytes,
                fixture.binding, static_cast<const double*>(device_overlap.get()),
                static_cast<std::int64_t>(changed.size()), epoch,
                static_cast<const std::uint32_t*>(request_error.get()), fixture.handles.stream)
            .success());
  std::vector<double> after;
  std::vector<std::uint64_t> generations;
  CHECK(download(fixture.binding.cache.cholesky_factors, changed.size(), after,
                 fixture.handles.stream));
  CHECK(download(fixture.binding.cache.geometry_generations, active.size(), generations,
                 fixture.handles.stream));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream));
  CHECK(after == before);
  CHECK(std::all_of(generations.begin(), generations.end(),
                    [](std::uint64_t value) { return value == kGeneration; }));

  std::uint64_t host_epoch_value = epoch_value;
  Gfn2GeometryEpochDevice host_epoch{&host_epoch_value, 1, kPlanToken};
  auto diagnostic = fixture.setup.refactor_overlap_from_device_epoch_async(
      fixture.setup_arena.get(), fixture.setup.requirements().setup_device_bytes, fixture.binding,
      static_cast<const double*>(device_overlap.get()), static_cast<std::int64_t>(changed.size()),
      host_epoch, static_cast<const std::uint32_t*>(request_error.get()), fixture.handles.stream);
  CHECK(diagnostic.error == Gfn2SccSetupEigensolverError::kInvalidArenaMemory);
  CHECK(diagnostic.field == Gfn2SccSetupEigensolverField::kGeometryGeneration);

  diagnostic = fixture.setup.refactor_overlap_from_device_epoch_async(
      fixture.setup_arena.get(), fixture.setup.requirements().setup_device_bytes, fixture.binding,
      static_cast<const double*>(device_overlap.get()), static_cast<std::int64_t>(changed.size()),
      epoch,
      reinterpret_cast<const std::uint32_t*>(static_cast<const std::byte*>(request_error.get()) +
                                             1u),
      fixture.handles.stream);
  CHECK(diagnostic.error == Gfn2SccSetupEigensolverError::kInvalidArenaMemory);
  CHECK(diagnostic.field == Gfn2SccSetupEigensolverField::kAdmission);
  diagnostic = fixture.setup.refactor_overlap_from_device_epoch_async(
      fixture.setup_arena.get(), fixture.setup.requirements().setup_device_bytes, fixture.binding,
      static_cast<const double*>(device_overlap.get()), static_cast<std::int64_t>(changed.size()),
      epoch, reinterpret_cast<const std::uint32_t*>(fixture.setup_arena.get()),
      fixture.handles.stream);
  CHECK(diagnostic.error == Gfn2SccSetupEigensolverError::kInvalidOverlap);
  CHECK(diagnostic.field == Gfn2SccSetupEigensolverField::kAdmission);
  diagnostic = fixture.setup.refactor_overlap_from_device_epoch_async(
      fixture.setup_arena.get(), fixture.setup.requirements().setup_device_bytes, fixture.binding,
      static_cast<const double*>(device_overlap.get()), static_cast<std::int64_t>(changed.size()),
      epoch, reinterpret_cast<const std::uint32_t*>(device_epoch.get()), fixture.handles.stream);
  CHECK(diagnostic.error == Gfn2SccSetupEigensolverError::kInvalidOverlap);
  CHECK(diagnostic.field == Gfn2SccSetupEigensolverField::kAdmission);
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
  status = test_device_overlap_refactor_batches_and_fail_closed_aliases();
  if (status != 0) {
    return status;
  }
  status = test_device_overlap_refactor_peer_failure();
  if (status != 0) {
    return status;
  }
  status = test_device_overlap_refactor_graph_replay();
  if (status != 0) {
    return status;
  }
  status = test_bad_overlap_preserves_factor_and_reports_failure();
  if (status != 0) {
    return status;
  }
  status = test_forced_tridiagonal_strategy_is_valid();
  if (status != 0) {
    return status;
  }
  status = test_fail_closed_plan_generation_and_provenance();
  if (status != 0) {
    return status;
  }
  return test_device_epoch_admission_validation_and_rejection();
}
