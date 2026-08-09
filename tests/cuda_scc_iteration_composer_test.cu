#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_scc_iteration_arena.cuh"
#include "backends/cuda/gfn2_scc_iteration_initialize.cuh"
#include "backends/cuda/gfn2_scc_iteration_reports.cuh"
#include "backends/cuda/gfn2_scc_loop.cuh"
#include "backends/cuda/gfn2_scc_setup_eigensolver.cuh"
#include "backends/cuda/gfn2_scc_setup_inputs.cuh"
#include "backends/cuda/gfn2_scc_setup_topology.hpp"
#include "data/parameters/d4.hpp"
#include "model/gfn2/coordination.hpp"
#include "runtime/nvidia_host_api.h"
#include "tests/support/gfn2_scc_test_case.hpp"

/*
 * #95: CUDA SCC composer Graph replay and failure-DAG tests.
 *
 * This translation unit owns the reusable composer fixture for the restricted
 * GFN2 SCC iteration: immutable topology/input/eigensolver setup owners, the
 * single caller-owned iteration arena, the fresh initializer,
 * and the complete binding produced by the sealed setup path. Unlike the
 * historical partial fixture, the binding here is the real production binding
 * (eigensolver provider sealed as kGraphSupported), so it exercises
 * bind_gfn2_scc_iteration_arena_cuda + launch_gfn2_restricted_scc_iteration_cuda
 * directly.
 *
 * Coverage maps to the issue acceptance:
 *  - one-step CPU parity including raw-on-terminal versus next_mixed-on-nonterminal
 *    publication for batch 1/8/32/128 on a custom stream (plus a default-stream
 *    parity case at batch 8);
 *  - repeated launches with a reused binding (no descriptor rebuild);
 *  - changed-input CUDA Graph replay with unchanged descriptors and arenas;
 *  - peer-local numerical failure with healthy-peer full-DAG continuation;
 *  - plan/provenance failure suppressing downstream stages with public buffers
 *    left byte-stable;
 *  - inactive-peer poisoning of geometry, multipoles, mixer histories, energies,
 *    and publication buffers proving the composer neither reads nor writes them.
 */

#define CHECK(condition)                                                                    \
  do {                                                                                      \
    if (!(condition)) {                                                                     \
      std::fprintf(stderr, "composer check failed at line %d: %s\n", __LINE__, #condition); \
      return __LINE__;                                                                      \
    }                                                                                       \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using namespace gpuxtb::detail;
using namespace gpuxtb::detail::cuda;
using gpuxtb::test::gfn2::HostSccCase;
using gpuxtb::test::gfn2::HostSccCaseOptions;
using gpuxtb::test::gfn2::HostSccCheckpoint;
using gpuxtb::test::gfn2::SmallSystemKind;

constexpr std::uint64_t kPlanToken = 0x9500c0deULL;
constexpr std::uint64_t kGeometryGeneration = 105u;
constexpr std::uint64_t kInitializationGeneration = 1u;

struct CouplingSelection {
  bool d4 = false;
  bool point_charges = false;
  bool periodic = false;
};

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

  bool allocate(std::size_t bytes) noexcept {
    bytes_ = bytes;
    return bytes != 0u && cudaMalloc(&pointer_, bytes) == cudaSuccess;
  }

  void* get() const noexcept { return pointer_; }
  std::size_t bytes() const noexcept { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0u;
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

  bool allocate(std::size_t bytes) noexcept {
    bytes_ = bytes;
    return bytes == 0u || cudaMallocHost(&pointer_, bytes) == cudaSuccess;
  }

  void* get() const noexcept { return pointer_; }
  std::size_t bytes() const noexcept { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0u;
};

class GraphResources {
 public:
  GraphResources() = default;
  GraphResources(const GraphResources&) = delete;
  GraphResources& operator=(const GraphResources&) = delete;

  ~GraphResources() {
    /* Destroy the instantiated executable before the graph it was created
     * from, mirroring the lifetime order used by production graph owners. */
    if (executable_ != nullptr) {
      (void)cudaGraphExecDestroy(executable_);
    }
    if (graph_ != nullptr) {
      (void)cudaGraphDestroy(graph_);
    }
  }

  cudaGraph_t* graph_address() noexcept { return &graph_; }
  cudaGraphExec_t* executable_address() noexcept { return &executable_; }
  cudaGraph_t graph() const noexcept { return graph_; }
  cudaGraphExec_t executable() const noexcept { return executable_; }

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
};

class ProviderHandles {
 public:
  ProviderHandles() = default;
  ProviderHandles(const ProviderHandles&) = delete;
  ProviderHandles& operator=(const ProviderHandles&) = delete;

  ~ProviderHandles() {
    if (blas_ != nullptr) {
      (void)cublasDestroy(blas_);
    }
    if (parameters_ != nullptr) {
      (void)cusolverDnDestroyParams(parameters_);
    }
    if (solver_ != nullptr) {
      (void)cusolverDnDestroy(solver_);
    }
    if (stream_ != nullptr) {
      (void)cudaStreamSynchronize(stream_);
      (void)cudaStreamDestroy(stream_);
    }
  }

  bool create() noexcept {
    return cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) == cudaSuccess &&
           cusolverDnCreate(&solver_) == CUSOLVER_STATUS_SUCCESS &&
           cusolverDnCreateParams(&parameters_) == CUSOLVER_STATUS_SUCCESS &&
           cublasCreate(&blas_) == CUBLAS_STATUS_SUCCESS;
  }

  cudaStream_t stream() const noexcept { return stream_; }
  cusolverDnHandle_t solver() const noexcept { return solver_; }
  cusolverDnParams_t parameters() const noexcept { return parameters_; }
  cublasHandle_t blas() const noexcept { return blas_; }

 private:
  cudaStream_t stream_ = nullptr;
  cusolverDnHandle_t solver_ = nullptr;
  cusolverDnParams_t parameters_ = nullptr;
  cublasHandle_t blas_ = nullptr;
};

/* HostSccCase exposes the production component caches but not the common
 * geometry-pair cache. The SCC iteration only validates that common cache as
 * an energy-only input; a deterministic finite image supplies the production
 * owner until the host fixture grows a common geometry-cache accessor. */
template <typename T>
Gfn2SccSetupHostArray<T> setup_view(const std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), static_cast<std::int64_t>(values.size())};
}

template <typename T>
Gfn2SccIterationHostArrayView<T> initialization_view(const T* values,
                                                     std::int64_t elements) noexcept {
  return {elements == 0 ? nullptr : values, elements};
}

struct InputBacking {
  gpuxtb::detail::gfn2::CoordinationPlan coordination_plan;
  std::vector<double> geometry_pair_data;
  std::vector<std::uint64_t> geometry_generations;
  std::vector<Gfn2D4DeviceElementData> d4_elements;
  std::vector<Gfn2D4DeviceReferenceData> d4_references;

  bool prepare(const HostSccCase& host, std::string& error) {
    if (gpuxtb::detail::gfn2::make_coordination_plan(
            host.batch_size(), host.total_atoms(), host.atom_offsets().data(),
            host.atomic_numbers().data(), coordination_plan, error) != GPUXTB_STATUS_SUCCESS) {
      return false;
    }
    geometry_pair_data.resize(static_cast<std::size_t>(host.aes2_plan().total_pairs()) *
                              kGfn2GeometryPairDataElements);
    for (std::size_t index = 0; index < geometry_pair_data.size(); ++index) {
      geometry_pair_data[index] = 0.001 * static_cast<double>(index + 1u);
    }
    geometry_generations.assign(static_cast<std::size_t>(host.batch_size()),
                                host.options().geometry_generation);

    d4_elements.reserve(gpuxtb::parameters::d4::kElements.size());
    for (const auto& element : gpuxtb::parameters::d4::kElements) {
      d4_elements.push_back({element.reference_offset, element.reference_count,
                             element.covalent_radius, element.electronegativity,
                             element.effective_charge, element.hardness, element.r4r2});
    }
    d4_references.reserve(gpuxtb::parameters::d4::kReferences.size());
    for (const auto& reference : gpuxtb::parameters::d4::kReferences) {
      d4_references.push_back(
          {reference.coordination_number, reference.charge, reference.gaussian_count});
    }
    return true;
  }

  Gfn2SccSetupInputSources sources(const HostSccCase& host) const noexcept {
    Gfn2SccSetupInputSources result{};
    result.basis = &host.basis_plan();
    result.integrals = &host.integral_plan();
    result.h0_plan = &host.h0_plan();
    result.wavefunction = &host.wavefunction_layout();
    result.es2 = &host.es2_plan();
    result.es3 = &host.es3_plan();
    result.aes2 = &host.aes2_plan();
    result.mulliken = &host.mulliken_plan();
    result.mixer = &host.mixer_plan();
    result.driver = &host.driver_plan();
    result.geometry_generation = host.options().geometry_generation;
    result.atomic_numbers = setup_view(host.atomic_numbers());
    result.positions = setup_view(host.positions());
    result.covalent_radii = setup_view(coordination_plan.covalent_radius);
    result.h0 = setup_view(host.h0());
    result.overlap = setup_view(host.overlap());
    result.dipole_integrals = setup_view(host.dipole_integrals());
    result.quadrupole_integrals = setup_view(host.quadrupole_integrals());
    result.geometry_cache.pair_data = setup_view(geometry_pair_data);
    result.geometry_cache.coordination_numbers = setup_view(host.coordination_numbers());
    result.geometry_cache.system_generations = setup_view(geometry_generations);
    result.es2_cache.coulomb_matrix = {host.es2_cache().coulomb_matrix,
                                       host.es2_cache().matrix_elements};
    result.aes2_cache.pair_data = {host.aes2_cache().pair_data,
                                   host.aes2_cache().pair_data_elements};

    if (host.d4_plan() != nullptr) {
      result.d4.plan = host.d4_plan();
      result.d4.elements = setup_view(d4_elements);
      result.d4.references = setup_view(d4_references);
      result.d4.reference_c6 = {
          gpuxtb::parameters::d4::kReferenceC6.data(),
          static_cast<std::int64_t>(gpuxtb::parameters::d4::kReferenceC6.size())};
      result.d4.pair_data = {host.d4_cache()->pair_data, host.d4_cache()->pair_data_elements};
      result.d4.coordination_numbers = {host.d4_cache()->coordination_numbers,
                                        host.d4_cache()->coordination_elements};
    }
    if (host.point_charge_plan() != nullptr) {
      result.point_charges.plan = host.point_charge_plan();
      result.point_charges.positions = setup_view(host.point_charge_positions());
      result.point_charges.charges = setup_view(host.point_charge_charges());
      result.point_charges.hardnesses = setup_view(host.point_charge_hardnesses());
      result.point_charges.shell_potential_cache =
          setup_view(host.explicit_point_charge_shell_potential());
    }
    if (host.periodic_plan() != nullptr) {
      result.periodic.plan = host.periodic_plan();
      result.periodic.shifts = setup_view(host.periodic_shifts());
      result.periodic.response_matrices = setup_view(host.periodic_response_matrices());
    }
    return result;
  }
};

Gfn2SccIterationHostInitialization fresh_initialization(const HostSccCase& host) noexcept {
  const auto& layout = host.wavefunction_layout();
  const auto& wavefunction = host.wavefunction();
  Gfn2SccIterationHostInitialization result{};
  result.mode = Gfn2SccIterationInitializationMode::kFresh;
  result.plan_token = kPlanToken;
  result.initialization_generation = kInitializationGeneration;
  result.topology = {
      initialization_view(host.atom_offsets().data(),
                          static_cast<std::int64_t>(host.atom_offsets().size())),
      initialization_view(layout.batch_shell_offsets.data(),
                          static_cast<std::int64_t>(layout.batch_shell_offsets.size())),
      kPlanToken};
  result.wavefunction.plan_token = kPlanToken;
  result.wavefunction.population = {
      initialization_view(wavefunction.qsh, layout.qsh.element_count),
      initialization_view(wavefunction.qat, layout.qat.element_count),
      initialization_view(wavefunction.dipole, layout.dipole.element_count),
      initialization_view(wavefunction.quadrupole, layout.quadrupole.element_count), kPlanToken};
  return result;
}

/*
 * Reusable #95 composer fixture. All immutable setup owners, the single
 * iteration arena, caller-owned provider host workspace, and initialized
 * binding are built with the sealed production path; the eigensolver provider
 * is captured only when the linked provider supports it (kGraphSupported).
 */
struct ComposerFixture {
  HostSccCase host;
  InputBacking backing;
  ProviderHandles handles;
  Gfn2SccSetupTopology topology_owner;
  Gfn2SccSetupInputs inputs_owner;
  Gfn2SccSetupEigensolver eigensolver_owner;
  Gfn2SccIterationInitializer initializer;
  DeviceAllocation topology_arena;
  DeviceAllocation input_arena;
  DeviceAllocation iteration_arena;
  DeviceAllocation eigensolver_setup_arena;
  PinnedAllocation provider_host_workspace;
  Gfn2RaggedTopologyView device_topology{};
  Gfn2WavefunctionLayoutView device_wavefunction{};
  Gfn2SccIterationDevicePlan plan_seed{};
  Gfn2SccIterationDeviceInput input_seed{};
  Gfn2SccIterationArenaRequirements arena_requirements{};
  Gfn2SccIterationDeviceState state_seed{};
  Gfn2SccIterationDeviceWorkspace workspace_seed{};
  Gfn2SccIterationReportStorage report_storage{};
  Gfn2SccSetupEigensolverBinding eigensolver_binding{};
  Gfn2SccIterationInitializationReady ready{};
  Gfn2SccIterationBinding binding{};

  bool create(std::int64_t batch_size = 4, bool optional_components = false,
              const std::vector<SmallSystemKind>& systems = {}, double electronic_temperature = 0.0,
              std::uint64_t maximum_iterations = 8u, CouplingSelection coupling = {}) {
    if (batch_size <= 0) {
      return false;
    }
    constexpr std::array<SmallSystemKind, 4> kSystems{SmallSystemKind::kH2, SmallSystemKind::kHe,
                                                      SmallSystemKind::kLiH, SmallSystemKind::kCH2};
    HostSccCaseOptions options{};
    options.systems.clear();
    options.systems.reserve(static_cast<std::size_t>(batch_size));
    for (std::int64_t system = 0; system < batch_size; ++system) {
      if (!systems.empty()) {
        options.systems.push_back(systems[static_cast<std::size_t>(system) % systems.size()]);
      } else {
        options.systems.push_back(kSystems[static_cast<std::size_t>(system) % kSystems.size()]);
      }
    }
    options.geometry_generation = kGeometryGeneration;
    options.maximum_iterations = maximum_iterations;
    options.mixer_history = 3;
    options.electronic_temperature = electronic_temperature;
    options.enable_d4 = optional_components || coupling.d4;
    options.enable_explicit_point_charges = optional_components || coupling.point_charges;
    options.enable_periodic_embedding = optional_components || coupling.periodic;

    std::string error;
    if (HostSccCase::create(options, host, error) != GPUXTB_STATUS_SUCCESS) {
      std::fprintf(stderr, "HostSccCase::create failed: %s\n", error.c_str());
      return false;
    }
    if (!backing.prepare(host, error) || !handles.create()) {
      std::fprintf(stderr, "composer host/provider setup failed: %s\n", error.c_str());
      return false;
    }

    auto topology_diagnostic =
        Gfn2SccSetupTopology::create(host.basis_plan(), host.integral_plan(),
                                     host.wavefunction_layout(), kPlanToken, topology_owner);
    if (!topology_diagnostic.success() ||
        !topology_arena.allocate(topology_owner.requirements().immutable_device_bytes)) {
      std::fprintf(stderr, "composer topology create/allocation failed\n");
      return false;
    }
    topology_diagnostic = topology_owner.bind_device_arena_and_upload_async(
        topology_arena.get(), topology_arena.bytes(), device_topology, device_wavefunction,
        handles.stream());
    if (!topology_diagnostic.success()) {
      std::fprintf(stderr, "composer topology upload failed: error=%u field=%u\n",
                   static_cast<unsigned>(topology_diagnostic.error),
                   static_cast<unsigned>(topology_diagnostic.field));
      return false;
    }

    const Gfn2SccSetupInputSources sources = backing.sources(host);
    auto input_diagnostic = Gfn2SccSetupInputs::create(sources, topology_owner.host_topology(),
                                                       kPlanToken, inputs_owner);
    if (!input_diagnostic.success() ||
        !input_arena.allocate(inputs_owner.requirements().device_bytes)) {
      std::fprintf(stderr,
                   "composer immutable-input create/allocation failed: error=%u "
                   "field=%u index=%lld\n",
                   static_cast<unsigned>(input_diagnostic.error),
                   static_cast<unsigned>(input_diagnostic.field),
                   static_cast<long long>(input_diagnostic.index));
      return false;
    }
    input_diagnostic = inputs_owner.bind_device_arena_and_upload_async(
        device_topology, device_wavefunction, input_arena.get(), input_arena.bytes(), plan_seed,
        input_seed, handles.stream());
    if (!input_diagnostic.success()) {
      std::fprintf(stderr, "composer immutable-input upload failed: error=%u field=%u\n",
                   static_cast<unsigned>(input_diagnostic.error),
                   static_cast<unsigned>(input_diagnostic.field));
      return false;
    }

    auto eigensolver_diagnostic = Gfn2SccSetupEigensolver::create(
        topology_owner, host.overlap().data(), static_cast<std::int64_t>(host.overlap().size()),
        kGeometryGeneration, kPlanToken, handles.solver(), handles.parameters(), handles.blas(),
        plan_seed.eigensolver_options, eigensolver_owner);
    if (!eigensolver_diagnostic.success()) {
      std::fprintf(stderr, "composer eigensolver owner create failed: error=%u field=%u\n",
                   static_cast<unsigned>(eigensolver_diagnostic.error),
                   static_cast<unsigned>(eigensolver_diagnostic.field));
      return false;
    }

    const auto& eigensolver_requirements = eigensolver_owner.requirements();
    const auto arena_diagnostic = query_gfn2_scc_iteration_arena_requirements_cuda(
        plan_seed, eigensolver_requirements.provider, arena_requirements);
    if (!arena_diagnostic.success() || !iteration_arena.allocate(arena_requirements.total_bytes) ||
        !provider_host_workspace.allocate(
            eigensolver_requirements.provider.solver_host_workspace_bytes)) {
      std::fprintf(stderr, "composer iteration-arena query/allocation failed: error=%u\n",
                   static_cast<unsigned>(arena_diagnostic.error));
      return false;
    }

    auto bind_arena_diagnostic = bind_gfn2_scc_iteration_arena_cuda(
        plan_seed, eigensolver_requirements.provider, arena_requirements, iteration_arena.get(),
        iteration_arena.bytes(), provider_host_workspace.get(), provider_host_workspace.bytes(),
        state_seed, workspace_seed, report_storage);
    if (!bind_arena_diagnostic.success() ||
        !eigensolver_setup_arena.allocate(eigensolver_requirements.setup_device_bytes)) {
      std::fprintf(stderr, "composer iteration-arena bind failed: error=%u\n",
                   static_cast<unsigned>(bind_arena_diagnostic.error));
      return false;
    }

    eigensolver_diagnostic = eigensolver_owner.bind_and_factor_overlap_async(
        device_topology, plan_seed, arena_requirements, iteration_arena.get(),
        iteration_arena.bytes(), workspace_seed, provider_host_workspace.get(),
        provider_host_workspace.bytes(), eigensolver_setup_arena.get(),
        eigensolver_setup_arena.bytes(), eigensolver_binding, handles.stream());
    if (!eigensolver_diagnostic.success()) {
      std::fprintf(stderr,
                   "composer overlap factorization submission failed: error=%u field=%u "
                   "index=%lld\n",
                   static_cast<unsigned>(eigensolver_diagnostic.error),
                   static_cast<unsigned>(eigensolver_diagnostic.field),
                   static_cast<long long>(eigensolver_diagnostic.index));
      return false;
    }
    plan_seed.eigensolver_batch = eigensolver_binding.batch;
    plan_seed.eigensolver_provider = eigensolver_binding.provider;
    plan_seed.overlap_cache = eigensolver_binding.cache;
    plan_seed.eigensolver_options = eigensolver_binding.options;

    auto initialization_diagnostic = Gfn2SccIterationInitializer::create(
        plan_seed, arena_requirements, iteration_arena.get(), iteration_arena.bytes(), state_seed,
        workspace_seed, report_storage, fresh_initialization(host), initializer);
    if (!initialization_diagnostic.success()) {
      std::fprintf(stderr,
                   "composer initializer create failed: error=%u field=%u "
                   "index=%lld\n",
                   static_cast<unsigned>(initialization_diagnostic.error),
                   static_cast<unsigned>(initialization_diagnostic.field),
                   static_cast<long long>(initialization_diagnostic.index));
      return false;
    }
    initialization_diagnostic = initializer.upload_async(
        iteration_arena.get(), iteration_arena.bytes(), ready, handles.stream());
    if (!initialization_diagnostic.success()) {
      std::fprintf(stderr, "composer initializer upload failed: error=%u field=%u\n",
                   static_cast<unsigned>(initialization_diagnostic.error),
                   static_cast<unsigned>(initialization_diagnostic.field));
      return false;
    }

    const auto report_diagnostic = build_gfn2_scc_iteration_report_binding_cuda(
        report_storage, plan_seed, input_seed, state_seed, workspace_seed, binding);
    if (report_diagnostic.error != Gfn2SccIterationBindingError::kSuccess) {
      std::fprintf(stderr,
                   "composer report/binding build failed: error=%u field=%u "
                   "index=%lld\n",
                   static_cast<unsigned>(report_diagnostic.error),
                   static_cast<unsigned>(report_diagnostic.field),
                   static_cast<long long>(report_diagnostic.index));
      return false;
    }

    const auto validator_diagnostic = validate_gfn2_scc_iteration_binding_cuda(
        binding.plan, binding.input, state_seed, workspace_seed);
    if (validator_diagnostic.error != Gfn2SccIterationBindingError::kSuccess) {
      std::fprintf(stderr,
                   "composer binding validation failed: error=%u field=%u "
                   "index=%lld\n",
                   static_cast<unsigned>(validator_diagnostic.error),
                   static_cast<unsigned>(validator_diagnostic.field),
                   static_cast<long long>(validator_diagnostic.index));
      return false;
    }
    return true;
  }
};

template <typename T>
bool upload(const T* host, T* device, std::int64_t elements, cudaStream_t stream) {
  return elements >= 0 &&
         (elements == 0 ||
          cudaMemcpyAsync(device, host, static_cast<std::size_t>(elements) * sizeof(T),
                          cudaMemcpyHostToDevice, stream) == cudaSuccess);
}

template <typename T>
bool upload_value(const T& host, T* device, cudaStream_t stream) {
  return device != nullptr &&
         cudaMemcpyAsync(device, &host, sizeof(T), cudaMemcpyHostToDevice, stream) == cudaSuccess;
}

template <typename T>
bool download(const T* device, std::int64_t elements, std::vector<T>& host, cudaStream_t stream) {
  if (elements < 0) {
    return false;
  }
  host.resize(static_cast<std::size_t>(elements));
  return elements == 0 || cudaMemcpyAsync(host.data(), device, host.size() * sizeof(T),
                                          cudaMemcpyDeviceToHost, stream) == cudaSuccess;
}

template <typename T>
bool download_value(const T* device, T& host, cudaStream_t stream) {
  return device != nullptr &&
         cudaMemcpyAsync(&host, device, sizeof(T), cudaMemcpyDeviceToHost, stream) == cudaSuccess;
}

template <typename T>
bool upload_fill(T* device, std::int64_t elements, T value, cudaStream_t stream) {
  if (device == nullptr || elements < 0) {
    return false;
  }
  /* Fill through the caller's stream so the sentinel writes order with the
   * surrounding async transfers and launches on that stream. */
  std::vector<T> host(static_cast<std::size_t>(elements), value);
  return elements == 0 || cudaMemcpyAsync(device, host.data(), host.size() * sizeof(T),
                                          cudaMemcpyHostToDevice, stream) == cudaSuccess;
}

bool near(double first, double second, double tolerance) noexcept {
  const double scale = std::max({1.0, std::abs(first), std::abs(second)});
  return std::abs(first - second) <= tolerance * scale;
}

bool compare_doubles(const char* field, const std::vector<double>& actual, const double* expected,
                     std::int64_t elements, double tolerance) {
  if (expected == nullptr || elements < 0 || actual.size() != static_cast<std::size_t>(elements)) {
    std::fprintf(stderr, "%s has an invalid parity extent\n", field);
    return false;
  }
  for (std::int64_t index = 0; index < elements; ++index) {
    if (!near(actual[static_cast<std::size_t>(index)], expected[index], tolerance)) {
      std::fprintf(stderr, "%s mismatch at %lld: CUDA=%.17g CPU=%.17g delta=%.3e tolerance=%.3e\n",
                   field, static_cast<long long>(index), actual[static_cast<std::size_t>(index)],
                   expected[index], actual[static_cast<std::size_t>(index)] - expected[index],
                   tolerance);
      return false;
    }
  }
  return true;
}

template <typename T>
bool compare_exact_values(const char* field, const std::vector<T>& actual, const T* expected,
                          std::int64_t elements) {
  if (expected == nullptr || elements < 0 || actual.size() != static_cast<std::size_t>(elements)) {
    std::fprintf(stderr, "%s has an invalid parity extent\n", field);
    return false;
  }
  for (std::int64_t index = 0; index < elements; ++index) {
    if (actual[static_cast<std::size_t>(index)] != expected[index]) {
      std::fprintf(stderr, "%s mismatch at %lld\n", field, static_cast<long long>(index));
      return false;
    }
  }
  return true;
}

bool system_slice_is_byte_stable(const std::vector<double>& before,
                                 const std::vector<double>& after,
                                 const std::vector<std::int64_t>& offsets, std::int64_t system) {
  if (system < 0 || system + 1 >= static_cast<std::int64_t>(offsets.size()) ||
      before.size() != after.size()) {
    return false;
  }
  const std::int64_t begin = offsets[static_cast<std::size_t>(system)];
  const std::int64_t end = offsets[static_cast<std::size_t>(system + 1)];
  if (begin < 0 || begin > end || end > static_cast<std::int64_t>(before.size())) {
    return false;
  }
  return std::memcmp(before.data() + begin, after.data() + begin,
                     static_cast<std::size_t>(end - begin) * sizeof(double)) == 0;
}

bool compare_system_slice(const char* field, const std::vector<double>& actual,
                          const double* expected, const std::vector<std::int64_t>& offsets,
                          std::int64_t system, double tolerance) {
  if (expected == nullptr || system < 0 ||
      system + 1 >= static_cast<std::int64_t>(offsets.size())) {
    return false;
  }
  const std::int64_t begin = offsets[static_cast<std::size_t>(system)];
  const std::int64_t end = offsets[static_cast<std::size_t>(system + 1)];
  if (begin < 0 || begin > end || end > static_cast<std::int64_t>(actual.size())) {
    return false;
  }
  for (std::int64_t index = begin; index < end; ++index) {
    if (!near(actual[static_cast<std::size_t>(index)], expected[index], tolerance)) {
      std::fprintf(stderr,
                   "%s system=%lld index=%lld mismatch: CUDA=%.17g CPU=%.17g delta=%.3e "
                   "tolerance=%.3e\n",
                   field, static_cast<long long>(system), static_cast<long long>(index),
                   actual[static_cast<std::size_t>(index)], expected[index],
                   actual[static_cast<std::size_t>(index)] - expected[index], tolerance);
      return false;
    }
  }
  return true;
}

bool system_slice_is_finite(const std::vector<double>& values,
                            const std::vector<std::int64_t>& offsets, std::int64_t system) {
  if (system < 0 || system + 1 >= static_cast<std::int64_t>(offsets.size())) {
    return false;
  }
  const std::int64_t begin = offsets[static_cast<std::size_t>(system)];
  const std::int64_t end = offsets[static_cast<std::size_t>(system + 1)];
  if (begin < 0 || begin >= end || end > static_cast<std::int64_t>(values.size())) {
    return false;
  }
  return std::all_of(values.begin() + begin, values.begin() + end,
                     [](double value) { return std::isfinite(value); });
}

/* Compare the persistent iteration state and unpublished Mulliken result with
 * the independent CPU transition. This is intentionally shared by repeated
 * launches and Graph replays so neither path can pass by publishing stale but
 * finite data. */
int compare_cuda_cpu_checkpoint(const char* checkpoint, const ComposerFixture& fixture,
                                double tolerance, bool compare_private_history) {
  const auto& layout = fixture.host.wavefunction_layout();
  const auto& state = fixture.binding.state;
  const auto& workspace = fixture.binding.workspace;
  const std::int64_t batch = fixture.host.batch_size();
  const std::int64_t mixer_vector = fixture.host.mixer_plan().total_vector_elements();
  const std::int64_t mixer_history = mixer_vector * fixture.host.mixer_plan().history_size();
  const std::int64_t mixer_omega = batch * fixture.host.mixer_plan().history_size();
  std::vector<double> eigenvalues;
  std::vector<double> occupations;
  std::vector<double> density;
  std::vector<double> weighted_density;
  std::vector<double> public_qsh;
  std::vector<double> public_qat;
  std::vector<double> public_dipoles;
  std::vector<double> public_quadrupoles;
  std::vector<double> raw_qsh;
  std::vector<double> raw_qat;
  std::vector<double> raw_dipoles;
  std::vector<double> raw_quadrupoles;
  std::vector<double> free_energies;
  std::vector<double> mixer_inputs;
  std::vector<double> mixer_previous_inputs;
  std::vector<double> mixer_previous_residuals;
  std::vector<double> mixer_df_history;
  std::vector<double> mixer_u_history;
  std::vector<double> mixer_omega_values;
  std::vector<double> mixer_residual_rms;
  std::vector<double> mixer_residual_maximum;
  std::vector<std::uint64_t> mixer_iterations;
  std::vector<std::uint64_t> mixer_restart_counts;
  std::vector<gpuxtb_status_t> mixer_statuses;
  std::vector<std::uint8_t> mixer_initialized;
  std::vector<std::uint8_t> mixer_residual_converged;
  std::vector<double> previous_free_energies;
  std::vector<double> free_energy_changes;
  std::vector<double> scc_residual_rms;
  std::vector<std::uint64_t> iterations;
  std::vector<gpuxtb_status_t> statuses;
  std::vector<std::uint8_t> converged;

  CHECK(download(state.eigenpairs.eigenvalues, state.eigenpairs.eigenvalue_elements, eigenvalues,
                 fixture.handles.stream()));
  CHECK(download(state.occupations.occupations, state.occupations.occupation_elements, occupations,
                 fixture.handles.stream()));
  CHECK(download(state.density.density, state.density.density_elements, density,
                 fixture.handles.stream()));
  CHECK(download(state.density.energy_weighted_density, state.density.weighted_density_elements,
                 weighted_density, fixture.handles.stream()));
  CHECK(download(state.raw_population.qsh, state.raw_population.qsh_elements, public_qsh,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.qat, state.raw_population.qat_elements, public_qat,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.dipole, state.raw_population.dipole_elements, public_dipoles,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.quadrupole, state.raw_population.quadrupole_elements,
                 public_quadrupoles, fixture.handles.stream()));
  CHECK(download(workspace.staged_raw_population.qsh, workspace.staged_raw_population.qsh_elements,
                 raw_qsh, fixture.handles.stream()));
  CHECK(download(workspace.staged_raw_population.qat, workspace.staged_raw_population.qat_elements,
                 raw_qat, fixture.handles.stream()));
  CHECK(download(workspace.staged_raw_population.dipole,
                 workspace.staged_raw_population.dipole_elements, raw_dipoles,
                 fixture.handles.stream()));
  CHECK(download(workspace.staged_raw_population.quadrupole,
                 workspace.staged_raw_population.quadrupole_elements, raw_quadrupoles,
                 fixture.handles.stream()));
  CHECK(download(state.scc.free_energies, state.scc.batch_elements, free_energies,
                 fixture.handles.stream()));
  CHECK(download(state.mixer.current_inputs, state.mixer.total_vector_elements, mixer_inputs,
                 fixture.handles.stream()));
  CHECK(download(state.mixer.previous_inputs, mixer_vector, mixer_previous_inputs,
                 fixture.handles.stream()));
  CHECK(download(state.mixer.previous_residuals, mixer_vector, mixer_previous_residuals,
                 fixture.handles.stream()));
  if (compare_private_history) {
    CHECK(download(state.mixer.df_history, mixer_history, mixer_df_history,
                   fixture.handles.stream()));
    CHECK(
        download(state.mixer.u_history, mixer_history, mixer_u_history, fixture.handles.stream()));
    CHECK(download(state.mixer.omega, mixer_omega, mixer_omega_values, fixture.handles.stream()));
  }
  CHECK(download(state.mixer.residual_rms, batch, mixer_residual_rms, fixture.handles.stream()));
  CHECK(download(state.mixer.residual_maximum, batch, mixer_residual_maximum,
                 fixture.handles.stream()));
  CHECK(download(state.mixer.iterations, batch, mixer_iterations, fixture.handles.stream()));
  CHECK(
      download(state.mixer.restart_counts, batch, mixer_restart_counts, fixture.handles.stream()));
  CHECK(download(state.mixer.system_statuses, batch, mixer_statuses, fixture.handles.stream()));
  CHECK(download(state.mixer.initialized, batch, mixer_initialized, fixture.handles.stream()));
  CHECK(download(state.mixer.residual_converged, batch, mixer_residual_converged,
                 fixture.handles.stream()));
  CHECK(download(state.scc.previous_free_energies, batch, previous_free_energies,
                 fixture.handles.stream()));
  CHECK(download(state.scc.free_energy_changes, batch, free_energy_changes,
                 fixture.handles.stream()));
  CHECK(download(state.scc.residual_rms, batch, scc_residual_rms, fixture.handles.stream()));
  CHECK(download(state.scc.iterations, state.scc.batch_elements, iterations,
                 fixture.handles.stream()));
  CHECK(download(state.scc.system_statuses, state.scc.batch_elements, statuses,
                 fixture.handles.stream()));
  CHECK(
      download(state.scc.converged, state.scc.batch_elements, converged, fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  const auto field = [checkpoint](const char* name) {
    return std::string(checkpoint) + " " + name;
  };
  /* Individual eigenvector columns are intentionally excluded: a valid
   * eigensolver may rotate or sign-flip an exactly degenerate subspace. */
  CHECK(compare_doubles(field("eigenvalues").c_str(), eigenvalues,
                        fixture.host.wavefunction().eigenvalues, layout.eigenvalues.element_count,
                        tolerance));
  CHECK(compare_doubles(field("occupations").c_str(), occupations,
                        fixture.host.wavefunction().occupations, layout.occupations.element_count,
                        tolerance));
  CHECK(compare_doubles(field("density").c_str(), density, fixture.host.wavefunction().density,
                        layout.density.element_count, tolerance));
  CHECK(compare_doubles(field("weighted density").c_str(), weighted_density,
                        fixture.host.wavefunction().energy_weighted_density,
                        layout.energy_weighted_density.element_count, tolerance));
  CHECK(compare_doubles(field("public qsh").c_str(), public_qsh, fixture.host.wavefunction().qsh,
                        layout.qsh.element_count, tolerance));
  CHECK(compare_doubles(field("public qat").c_str(), public_qat, fixture.host.wavefunction().qat,
                        layout.qat.element_count, tolerance));
  CHECK(compare_doubles(field("public dipoles").c_str(), public_dipoles,
                        fixture.host.wavefunction().dipole, layout.dipole.element_count,
                        tolerance));
  CHECK(compare_doubles(field("public quadrupoles").c_str(), public_quadrupoles,
                        fixture.host.wavefunction().quadrupole, layout.quadrupole.element_count,
                        tolerance));
  CHECK(compare_doubles(field("raw qsh").c_str(), raw_qsh, fixture.host.driver_workspace().raw_qsh,
                        layout.qsh.element_count, tolerance));
  CHECK(compare_doubles(field("raw qat").c_str(), raw_qat, fixture.host.driver_workspace().raw_qat,
                        layout.qat.element_count, tolerance));
  CHECK(compare_doubles(field("raw dipoles").c_str(), raw_dipoles,
                        fixture.host.driver_workspace().raw_dipoles, layout.dipole.element_count,
                        tolerance));
  CHECK(compare_doubles(field("raw quadrupoles").c_str(), raw_quadrupoles,
                        fixture.host.driver_workspace().raw_quadrupoles,
                        layout.quadrupole.element_count, tolerance));
  CHECK(compare_doubles(field("free energies").c_str(), free_energies,
                        fixture.host.driver_state().free_energies, fixture.host.batch_size(),
                        tolerance));
  CHECK(compare_doubles(field("mixer current inputs").c_str(), mixer_inputs,
                        fixture.host.mixer_state().current_inputs,
                        fixture.host.mixer_plan().total_vector_elements(), tolerance));
  CHECK(compare_doubles(field("mixer previous inputs").c_str(), mixer_previous_inputs,
                        fixture.host.mixer_state().previous_inputs, mixer_vector, tolerance));
  CHECK(compare_doubles(field("mixer previous residuals").c_str(), mixer_previous_residuals,
                        fixture.host.mixer_state().previous_residuals, mixer_vector, tolerance));
  if (compare_private_history) {
    CHECK(compare_doubles(field("mixer df history").c_str(), mixer_df_history,
                          fixture.host.mixer_state().df_history, mixer_history, tolerance));
    CHECK(compare_doubles(field("mixer u history").c_str(), mixer_u_history,
                          fixture.host.mixer_state().u_history, mixer_history, tolerance));
    CHECK(compare_doubles(field("mixer omega").c_str(), mixer_omega_values,
                          fixture.host.mixer_state().omega, mixer_omega, tolerance));
  }
  CHECK(compare_doubles(field("mixer residual RMS").c_str(), mixer_residual_rms,
                        fixture.host.mixer_state().residual_rms, batch, tolerance));
  CHECK(compare_doubles(field("mixer residual maximum").c_str(), mixer_residual_maximum,
                        fixture.host.mixer_state().residual_maximum, batch, tolerance));
  CHECK(compare_exact_values(field("mixer iterations").c_str(), mixer_iterations,
                             fixture.host.mixer_state().iterations, batch));
  CHECK(compare_exact_values(field("mixer restart counts").c_str(), mixer_restart_counts,
                             fixture.host.mixer_state().restart_counts, batch));
  CHECK(compare_exact_values(field("mixer statuses").c_str(), mixer_statuses,
                             fixture.host.mixer_state().system_statuses, batch));
  CHECK(compare_exact_values(field("mixer initialized").c_str(), mixer_initialized,
                             fixture.host.mixer_state().initialized, batch));
  std::vector<std::uint8_t> expected_residual_converged(static_cast<std::size_t>(batch));
  for (std::int64_t system = 0; system < batch; ++system) {
    expected_residual_converged[static_cast<std::size_t>(system)] =
        fixture.host.mixer_state().residual_rms[system] <
                    fixture.host.mixer_plan().rms_tolerance() &&
                fixture.host.mixer_state().residual_maximum[system] <
                    fixture.host.mixer_plan().maximum_tolerance()
            ? 1u
            : 0u;
  }
  CHECK(compare_exact_values(field("mixer residual-converged").c_str(), mixer_residual_converged,
                             expected_residual_converged.data(), batch));
  CHECK(compare_doubles(field("SCC previous free energies").c_str(), previous_free_energies,
                        fixture.host.driver_state().previous_free_energies, batch, tolerance));
  CHECK(compare_doubles(field("SCC free-energy changes").c_str(), free_energy_changes,
                        fixture.host.driver_state().free_energy_changes, batch, tolerance));
  CHECK(compare_doubles(field("SCC residual RMS").c_str(), scc_residual_rms,
                        fixture.host.mixer_state().residual_rms, batch, tolerance));
  CHECK(compare_exact_values(field("iterations").c_str(), iterations,
                             fixture.host.driver_state().iterations, fixture.host.batch_size()));
  CHECK(compare_exact_values(field("statuses").c_str(), statuses,
                             fixture.host.driver_state().system_statuses,
                             fixture.host.batch_size()));
  CHECK(compare_exact_values(field("converged").c_str(), converged,
                             fixture.host.driver_state().converged, fixture.host.batch_size()));
  return 0;
}

std::uint64_t failure_record(Gfn2SccStageId stage, std::uint32_t code) {
  return gfn2_scc_stage_failure_record(stage, code);
}

/* One CPU driver call per binding launch, mirroring the CUDA loop bound. */
int run_host_fixed_scc_loop(HostSccCase& host) {
  std::string error;
  for (std::uint64_t iteration = 0u; iteration < host.options().maximum_iterations; ++iteration) {
    const gpuxtb_status_t status = host.run_one_iteration(error);
    if (status != GPUXTB_STATUS_SUCCESS && status != GPUXTB_STATUS_SCC_NOT_CONVERGED &&
        status != GPUXTB_STATUS_EIGENSOLVER_FAILED) {
      std::fprintf(stderr, "CPU composer reference failed at %llu: status=%d error=%s\n",
                   static_cast<unsigned long long>(iteration), status, error.c_str());
      return __LINE__;
    }
  }
  return 0;
}

/* Advance the CPU oracle until every peer is terminal (converged, failed, or
 * at the iteration bound), mirroring the device-tail graph termination the
 * terminal branch of the raw-publication policy test relies on. */
int run_host_until_globally_terminal(HostSccCase& host) {
  std::string error;
  for (;;) {
    bool any_active = false;
    const auto& state = host.driver_state();
    for (std::int64_t system = 0; system < host.batch_size(); ++system) {
      any_active = any_active || (state.system_statuses[system] == GPUXTB_STATUS_SUCCESS &&
                                  state.converged[system] == 0u &&
                                  state.iterations[system] < host.options().maximum_iterations);
    }
    if (!any_active) {
      return 0;
    }
    const gpuxtb_status_t status = host.run_one_iteration(error);
    if (status != GPUXTB_STATUS_SUCCESS && status != GPUXTB_STATUS_SCC_NOT_CONVERGED &&
        status != GPUXTB_STATUS_EIGENSOLVER_FAILED) {
      std::fprintf(stderr, "CPU composer terminal reference failed: status=%d error=%s\n", status,
                   error.c_str());
      return __LINE__;
    }
  }
}

std::vector<double> changed_core_hamiltonian(const HostSccCase& host) {
  std::vector<double> changed = host.h0();
  const auto& layout = host.wavefunction_layout();
  const auto& matrix_offsets = host.h0_plan().matrix_offsets;
  for (std::int64_t system = 0; system < layout.batch_size; ++system) {
    const std::int64_t n =
        layout.batch_orbital_offsets[system + 1] - layout.batch_orbital_offsets[system];
    /* H0 has one physical matrix per system. Keep every perturbation
     * symmetric so the changed input stays a valid one-electron Hamiltonian. */
    const std::int64_t matrix_begin = matrix_offsets[static_cast<std::size_t>(system)];
    for (std::int64_t row = 0; row < n; ++row) {
      for (std::int64_t column = row; column < n; ++column) {
        const double shift = 2.5e-3 * static_cast<double>(system + 1) +
                             5.0e-4 * static_cast<double>(row + column + 1);
        changed[static_cast<std::size_t>(matrix_begin + row * n + column)] += shift;
        if (row != column) {
          changed[static_cast<std::size_t>(matrix_begin + column * n + row)] += shift;
        }
      }
    }
  }
  return changed;
}

/* Snapshot every public numerical buffer the test isolates during poisoning. */
struct PublicSnapshot {
  std::vector<double> qsh;
  std::vector<double> qat;
  std::vector<double> published_shell_charges;
  std::vector<double> free_energies;
  std::vector<std::uint64_t> iterations;
  std::vector<gpuxtb_status_t> statuses;
  std::vector<std::uint8_t> converged;
};

int snapshot_public(const ComposerFixture& fixture, PublicSnapshot& snapshot) {
  const auto& layout = fixture.host.wavefunction_layout();
  const auto& state = fixture.binding.state;
  CHECK(download(state.raw_population.qsh, state.raw_population.qsh_elements, snapshot.qsh,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.qat, state.raw_population.qat_elements, snapshot.qat,
                 fixture.handles.stream()));
  CHECK(download(state.published.shell_charges, state.published.shell_elements,
                 snapshot.published_shell_charges, fixture.handles.stream()));
  CHECK(download(state.scc.free_energies, state.scc.batch_elements, snapshot.free_energies,
                 fixture.handles.stream()));
  CHECK(download(state.scc.iterations, state.scc.batch_elements, snapshot.iterations,
                 fixture.handles.stream()));
  CHECK(download(state.scc.system_statuses, state.scc.batch_elements, snapshot.statuses,
                 fixture.handles.stream()));
  CHECK(download(state.scc.converged, state.scc.batch_elements, snapshot.converged,
                 fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  (void)layout;
  return 0;
}

int test_composer_binding_and_repeat_launch() {
  ComposerFixture fixture;
  CHECK(fixture.create(4, true));
  const Gfn2SccIterationBindingDiagnostic validator =
      validate_gfn2_scc_iteration_binding_cuda(fixture.binding.plan, fixture.binding.input,
                                               fixture.binding.state, fixture.binding.workspace);
  CHECK(validator.error == Gfn2SccIterationBindingError::kSuccess);
  CHECK(fixture.binding.plan.plan_token == kPlanToken);
  CHECK(fixture.binding.workspace.eigensolver_workspace.plan_token == kPlanToken);
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  const Gfn2SccIterationLaunchResult first =
      launch_gfn2_restricted_scc_iteration_cuda(fixture.binding, fixture.handles.stream());
  if (!first.success()) {
    std::fprintf(stderr, "composer launch 1 failed: status=%u stage=%u binding_error=%u cuda=%d\n",
                 static_cast<unsigned>(first.status), static_cast<unsigned>(first.stage),
                 static_cast<unsigned>(first.binding.error), static_cast<int>(first.cuda_status));
  }
  CHECK(first.success());
  std::string error;
  CHECK(fixture.host.run_one_iteration(error) == GPUXTB_STATUS_SUCCESS);
  CHECK(compare_cuda_cpu_checkpoint("repeat launch 1", fixture, 3.0e-9, true) == 0);
  std::vector<std::uint64_t> first_cpu_iterations(
      fixture.host.driver_state().iterations,
      fixture.host.driver_state().iterations + fixture.host.batch_size());
  bool second_transition_expected = false;
  for (std::int64_t system = 0; system < fixture.host.batch_size(); ++system) {
    second_transition_expected =
        second_transition_expected ||
        (fixture.host.driver_state().system_statuses[system] == GPUXTB_STATUS_SUCCESS &&
         fixture.host.driver_state().converged[system] == 0u &&
         fixture.host.driver_state().iterations[system] <
             fixture.host.options().maximum_iterations);
  }
  CHECK(second_transition_expected);

  /* Repeated launch reuses the exact same binding and arenas: steady-state
   * contract consumed by Graph replay, no descriptor rebuild in the hot path.
   * The second CPU transition is compared in full so a no-op or stale
   * publication cannot satisfy this reuse check. */
  const Gfn2SccIterationLaunchResult second =
      launch_gfn2_restricted_scc_iteration_cuda(fixture.binding, fixture.handles.stream());
  CHECK(second.success());
  CHECK(fixture.host.run_one_iteration(error) == GPUXTB_STATUS_SUCCESS);
  CHECK(compare_cuda_cpu_checkpoint("repeat launch 2", fixture, 3.0e-9, true) == 0);
  bool second_transition_advanced = false;
  for (std::int64_t system = 0; system < fixture.host.batch_size(); ++system) {
    second_transition_advanced =
        second_transition_advanced || fixture.host.driver_state().iterations[system] >
                                          first_cpu_iterations[static_cast<std::size_t>(system)];
  }
  CHECK(second_transition_advanced);
  return 0;
}

int test_composer_one_step_cpu_parity(std::int64_t batch_size, bool optional_components) {
  ComposerFixture fixture;
  CHECK(fixture.create(batch_size, optional_components));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  const Gfn2SccIterationLaunchResult launch =
      launch_gfn2_restricted_scc_iteration_cuda(fixture.binding, fixture.handles.stream());
  if (!launch.success()) {
    std::fprintf(stderr,
                 "composer parity launch failed: status=%u stage=%u binding_error=%u cuda=%d\n",
                 static_cast<unsigned>(launch.status), static_cast<unsigned>(launch.stage),
                 static_cast<unsigned>(launch.binding.error), static_cast<int>(launch.cuda_status));
  }
  CHECK(launch.success());
  std::string error;
  const gpuxtb_status_t cpu_status = fixture.host.run_one_iteration(error);
  if (cpu_status != GPUXTB_STATUS_SUCCESS) {
    std::fprintf(stderr, "CPU composer iteration failed: status=%d error=%s\n", cpu_status,
                 error.c_str());
  }
  CHECK(cpu_status == GPUXTB_STATUS_SUCCESS);

  const auto& layout = fixture.host.wavefunction_layout();
  const auto& state = fixture.binding.state;
  const auto& workspace = fixture.binding.workspace;
  std::vector<double> public_qsh;
  std::vector<double> public_qat;
  std::vector<double> public_dipoles;
  std::vector<double> public_quadrupoles;
  std::vector<double> raw_qsh;
  std::vector<double> raw_qat;
  std::vector<double> raw_dipoles;
  std::vector<double> raw_quadrupoles;
  std::vector<double> free_energy;
  std::vector<double> current_inputs;
  std::vector<std::uint64_t> iterations;
  std::vector<gpuxtb_status_t> statuses;
  std::vector<std::uint8_t> converged;
  CHECK(download(state.raw_population.qsh, state.raw_population.qsh_elements, public_qsh,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.qat, state.raw_population.qat_elements, public_qat,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.dipole, state.raw_population.dipole_elements, public_dipoles,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.quadrupole, state.raw_population.quadrupole_elements,
                 public_quadrupoles, fixture.handles.stream()));
  CHECK(download(workspace.staged_raw_population.qsh, workspace.staged_raw_population.qsh_elements,
                 raw_qsh, fixture.handles.stream()));
  CHECK(download(workspace.staged_raw_population.qat, workspace.staged_raw_population.qat_elements,
                 raw_qat, fixture.handles.stream()));
  CHECK(download(workspace.staged_raw_population.dipole,
                 workspace.staged_raw_population.dipole_elements, raw_dipoles,
                 fixture.handles.stream()));
  CHECK(download(workspace.staged_raw_population.quadrupole,
                 workspace.staged_raw_population.quadrupole_elements, raw_quadrupoles,
                 fixture.handles.stream()));
  CHECK(download(state.free_energy.free_energy, state.free_energy.free_energy_elements, free_energy,
                 fixture.handles.stream()));
  CHECK(download(state.mixer.current_inputs, state.mixer.total_vector_elements, current_inputs,
                 fixture.handles.stream()));
  CHECK(download(state.scc.iterations, state.scc.batch_elements, iterations,
                 fixture.handles.stream()));
  CHECK(download(state.scc.system_statuses, state.scc.batch_elements, statuses,
                 fixture.handles.stream()));
  CHECK(
      download(state.scc.converged, state.scc.batch_elements, converged, fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  for (std::int64_t system = 0; system < fixture.host.batch_size(); ++system) {
    CHECK(iterations[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().iterations[system]);
    CHECK(statuses[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().system_statuses[system]);
    CHECK(converged[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().converged[system]);
    CHECK(std::isfinite(free_energy[static_cast<std::size_t>(system)]));
  }
  CHECK(compare_doubles("public qsh", public_qsh, fixture.host.wavefunction().qsh,
                        layout.qsh.element_count, 3.0e-9));
  CHECK(compare_doubles("public qat", public_qat, fixture.host.wavefunction().qat,
                        layout.qat.element_count, 3.0e-9));
  CHECK(compare_doubles("public dipoles", public_dipoles, fixture.host.wavefunction().dipole,
                        layout.dipole.element_count, 3.0e-9));
  CHECK(compare_doubles("public quadrupoles", public_quadrupoles,
                        fixture.host.wavefunction().quadrupole, layout.quadrupole.element_count,
                        3.0e-9));
  CHECK(compare_doubles("raw qsh", raw_qsh, fixture.host.driver_workspace().raw_qsh,
                        layout.qsh.element_count, 3.0e-9));
  CHECK(compare_doubles("raw qat", raw_qat, fixture.host.driver_workspace().raw_qat,
                        layout.qat.element_count, 3.0e-9));
  CHECK(compare_doubles("raw dipoles", raw_dipoles, fixture.host.driver_workspace().raw_dipoles,
                        layout.dipole.element_count, 3.0e-9));
  CHECK(compare_doubles("raw quadrupoles", raw_quadrupoles,
                        fixture.host.driver_workspace().raw_quadrupoles,
                        layout.quadrupole.element_count, 3.0e-9));
  CHECK(compare_doubles("free energy", free_energy, fixture.host.driver_state().free_energies,
                        fixture.host.batch_size(), 3.0e-9));
  CHECK(compare_doubles("mixer current inputs", current_inputs,
                        fixture.host.mixer_state().current_inputs,
                        fixture.host.mixer_plan().total_vector_elements(), 3.0e-9));
  return 0;
}

/*
 * raw-on-terminal versus next_mixed-on-nonterminal publication policy.
 *
 * After a single damping step no peer has converged, so the published
 * population must be the damped next-mixed state while the staged Mulliken
 * population holds the raw one-iteration multipoles. After a full loop every
 * peer either converged (terminal) or hit the bound; for converged peers the
 * published population must be exactly the final raw Mulliken population and
 * must agree with the CPU oracle's published raw slice.
 */
int test_composer_raw_publication_policy(bool terminal) {
  ComposerFixture fixture;
  CHECK(fixture.create(4, false));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  if (terminal) {
    /* The device-tail path requires a graph-capturable eigensolver provider;
     * the nonterminal branch below executes the plain sealed launch and must
     * stay meaningful even when the provider falls back to the uncaptured
     * segment contract. */
    CHECK(fixture.binding.plan.eigensolver_provider.capture_mode ==
          Gfn2SccIterationProviderCaptureMode::kGraphSupported);
    /* Terminal: run the device-tail graph owner until no peer is active, so
     * converged peers' raw Mulliken population is committed to the public
     * raw buffers by the state composer. Force the monolithic device-tail
     * family explicitly: the small-batch kAuto default prefers the
     * exact-capacity dispatch chain since #227, and this test measures the
     * terminal-publication policy on the device-tail path, not the kAuto
     * family choice itself. */
    Gfn2SccLoopCudaGraphOwner graph;
    const Gfn2SccLoopGraphBuildResult build =
        graph.build(fixture.binding, Gfn2SccLoopGraphPreference::kDeviceTailGraph);
    if (!build.success() || !build.device_tail_graph_ready() || !graph.ready()) {
      std::fprintf(stderr, "terminal composer graph build failed: status=%u fallback=%u\n",
                   static_cast<unsigned>(build.status),
                   static_cast<unsigned>(build.fallback_reason));
    }
    CHECK(build.success());
    CHECK(build.device_tail_graph_ready());
    CHECK(graph.ready());
    const Gfn2SccLoopLaunchResult launch = graph.launch(fixture.handles.stream());
    CHECK(launch.success());
    /* Prove the device-tail Graph, not the bounded fallback, actually ran:
     * the owner must report the device-tail mode, terminate the canonical
     * active count at zero, and report no device-side launch error, exactly
     * as the production device-tail test verifies. */
    CHECK(launch.execution_mode == Gfn2SccLoopExecutionMode::kDeviceTailGraph);
    std::uint32_t terminal_active_count = 1u;
    std::uint32_t device_launch_error = cudaErrorUnknown;
    CHECK(graph.canonical_active_count_device() != nullptr);
    CHECK(graph.device_launch_error_device() != nullptr);
    CHECK(download_value(graph.canonical_active_count_device(), terminal_active_count,
                         fixture.handles.stream()));
    CHECK(download_value(graph.device_launch_error_device(), device_launch_error,
                         fixture.handles.stream()));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
    CHECK(terminal_active_count == 0u);
    CHECK(device_launch_error == cudaSuccess);
    CHECK(run_host_until_globally_terminal(fixture.host) == 0);

    const auto& layout = fixture.host.wavefunction_layout();
    std::vector<double> public_qsh;
    std::vector<double> raw_qsh;
    std::vector<std::uint8_t> converged;
    std::vector<std::uint64_t> iterations;
    std::vector<gpuxtb_status_t> statuses;
    CHECK(download(fixture.binding.state.raw_population.qsh,
                   fixture.binding.state.raw_population.qsh_elements, public_qsh,
                   fixture.handles.stream()));
    CHECK(download(fixture.binding.workspace.staged_raw_population.qsh,
                   fixture.binding.workspace.staged_raw_population.qsh_elements, raw_qsh,
                   fixture.handles.stream()));
    CHECK(download(fixture.binding.state.scc.converged, fixture.host.batch_size(), converged,
                   fixture.handles.stream()));
    CHECK(download(fixture.binding.state.scc.iterations, fixture.host.batch_size(), iterations,
                   fixture.handles.stream()));
    CHECK(download(fixture.binding.state.scc.system_statuses, fixture.host.batch_size(), statuses,
                   fixture.handles.stream()));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

    /* Every peer that reached terminal convergence must have published its raw
     * Mulliken population; the device-tail owner guarantees at least one such
     * peer here. Public == staged raw is the raw-on-terminal atom of the
     * #89/#95 publication contract and must match the CPU oracle exactly. */
    std::size_t terminal_count = 0u;
    for (std::int64_t system = 0; system < fixture.host.batch_size(); ++system) {
      CHECK(iterations[static_cast<std::size_t>(system)] ==
            fixture.host.driver_state().iterations[system]);
      CHECK(statuses[static_cast<std::size_t>(system)] ==
            fixture.host.driver_state().system_statuses[system]);
      CHECK(converged[static_cast<std::size_t>(system)] ==
            fixture.host.driver_state().converged[system]);
      if (converged[static_cast<std::size_t>(system)] != 0u) {
        ++terminal_count;
        const std::int64_t begin = layout.qsh.system_offsets[system];
        const std::int64_t end = layout.qsh.system_offsets[system + 1];
        for (std::int64_t index = begin; index < end; ++index) {
          CHECK(public_qsh[static_cast<std::size_t>(index)] ==
                raw_qsh[static_cast<std::size_t>(index)]);
        }
      }
    }
    CHECK(terminal_count > 0u);
    CHECK(compare_doubles("terminal public qsh", public_qsh, fixture.host.wavefunction().qsh,
                          layout.qsh.element_count, 3.0e-9));
    return 0;
  }

  const Gfn2SccIterationLaunchResult launch =
      launch_gfn2_restricted_scc_iteration_cuda(fixture.binding, fixture.handles.stream());
  CHECK(launch.success());
  std::string error;
  CHECK(fixture.host.run_one_iteration(error) == GPUXTB_STATUS_SUCCESS);

  const auto& layout = fixture.host.wavefunction_layout();
  std::vector<double> public_qsh;
  std::vector<double> raw_qsh;
  std::vector<std::uint8_t> converged;
  CHECK(download(fixture.binding.state.raw_population.qsh,
                 fixture.binding.state.raw_population.qsh_elements, public_qsh,
                 fixture.handles.stream()));
  CHECK(download(fixture.binding.workspace.staged_raw_population.qsh,
                 fixture.binding.workspace.staged_raw_population.qsh_elements, raw_qsh,
                 fixture.handles.stream()));
  CHECK(download(fixture.binding.state.scc.converged, fixture.host.batch_size(), converged,
                 fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  /* No peer converges on the first damped step. */
  for (const std::uint8_t value : converged) {
    CHECK(value == 0u);
  }
  /* Public == damped next-mixed (CPU wavefunction), staged raw is the raw
   * Mulliken population, and the two genuinely differ somewhere after one
   * damped mixing step (the trivial H2 shell is zero and may coincide). */
  CHECK(compare_doubles("nonterminal public qsh", public_qsh, fixture.host.wavefunction().qsh,
                        layout.qsh.element_count, 3.0e-9));
  CHECK(compare_doubles("nonterminal raw qsh", raw_qsh, fixture.host.driver_workspace().raw_qsh,
                        layout.qsh.element_count, 3.0e-9));
  double max_public_raw_delta = 0.0;
  for (std::int64_t index = 0; index < layout.qsh.element_count; ++index) {
    max_public_raw_delta =
        std::max(max_public_raw_delta, std::abs(public_qsh[static_cast<std::size_t>(index)] -
                                                raw_qsh[static_cast<std::size_t>(index)]));
  }
  CHECK(max_public_raw_delta > 1.0e-9);
  return 0;
}

int test_composer_changed_input_graph_replay(std::int64_t batch_size, bool optional_components) {
  ComposerFixture fixture;
  CHECK(fixture.create(batch_size, optional_components));
  CHECK(fixture.binding.plan.eigensolver_provider.capture_mode ==
        Gfn2SccIterationProviderCaptureMode::kGraphSupported);
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  const HostSccCheckpoint initial = fixture.host.checkpoint();
  const void* const stable_iteration_arena = fixture.iteration_arena.get();
  const void* const stable_input_arena = fixture.input_arena.get();
  const void* const stable_setup_arena = fixture.eigensolver_setup_arena.get();
  const void* const stable_topology_arena = fixture.topology_arena.get();

  GraphResources graph;
  CUDA_CHECK(cudaStreamBeginCapture(fixture.handles.stream(), cudaStreamCaptureModeThreadLocal));
  const Gfn2SccLoopLaunchResult captured =
      launch_gfn2_restricted_scc_loop_cuda(fixture.binding, fixture.handles.stream());
  CHECK(captured.success());
  CHECK(captured.submitted_iterations == fixture.host.options().maximum_iterations);
  CUDA_CHECK(cudaStreamEndCapture(fixture.handles.stream(), graph.graph_address()));
  CHECK(graph.graph() != nullptr);
  CUDA_CHECK(cudaGraphInstantiate(graph.executable_address(), graph.graph(), nullptr, nullptr, 0u));

  CUDA_CHECK(cudaGraphLaunch(graph.executable(), fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  CHECK(run_host_fixed_scc_loop(fixture.host) == 0);
  CHECK(compare_cuda_cpu_checkpoint("Graph replay 1", fixture, 1.0e-8, false) == 0);

  std::vector<double> first_free_energies;
  CHECK(download(fixture.binding.state.scc.free_energies, fixture.host.batch_size(),
                 first_free_energies, fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  /* Replay the identical graph with changed numerical H0 inputs: restore the
   * CPU oracle and the device initialization checkpoint, then rewrite the
   * immutable H0 buffer in place. No descriptor, arena, or provider is
   * rebuilt between the two executions. */
  std::string error;
  CHECK(fixture.host.restore(initial, error) == GPUXTB_STATUS_SUCCESS);
  const std::vector<double> changed_h0 = changed_core_hamiltonian(fixture.host);
  fixture.host.h0() = changed_h0;
  CHECK(fixture.initializer
            .upload_async(fixture.iteration_arena.get(), fixture.iteration_arena.bytes(),
                          fixture.ready, fixture.handles.stream())
            .success());
  CUDA_CHECK(cudaMemcpyAsync(const_cast<double*>(fixture.binding.input.hamiltonian.h0),
                             changed_h0.data(), changed_h0.size() * sizeof(double),
                             cudaMemcpyHostToDevice, fixture.handles.stream()));
  CUDA_CHECK(cudaGraphLaunch(graph.executable(), fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  CHECK(run_host_fixed_scc_loop(fixture.host) == 0);
  CHECK(compare_cuda_cpu_checkpoint("Graph replay 2 changed H0", fixture, 1.0e-8, false) == 0);

  std::vector<double> changed_free_energies;
  CHECK(download(fixture.binding.state.scc.free_energies, fixture.host.batch_size(),
                 changed_free_energies, fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  CHECK(first_free_energies.size() == changed_free_energies.size());
  bool numerical_input_was_consumed = false;
  for (std::size_t index = 0; index < first_free_energies.size(); ++index) {
    numerical_input_was_consumed =
        numerical_input_was_consumed ||
        !near(first_free_energies[index], changed_free_energies[index], 1.0e-10);
  }
  CHECK(numerical_input_was_consumed);
  CHECK(fixture.iteration_arena.get() == stable_iteration_arena);
  CHECK(fixture.input_arena.get() == stable_input_arena);
  CHECK(fixture.eigensolver_setup_arena.get() == stable_setup_arena);
  CHECK(fixture.topology_arena.get() == stable_topology_arena);
  CHECK(fixture.binding.plan.eigensolver_provider.plan_token == kPlanToken);
  return 0;
}

/* Healthy peers advance through the full DAG when one peer fails numerically
 * inside the mixed-gather stage (first consumer of the mixed shell charges):
 * the failed peer is isolated while peers 0/2/3 continue through the complete
 * DAG one more iteration and publish fresh finite population. */
int test_composer_peer_numerical_failure_dag() {
  constexpr std::int64_t kTarget = 1;
  ComposerFixture fixture;
  CHECK(fixture.create(4, true));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  const auto& layout = fixture.host.wavefunction_layout();
  const std::int64_t target_shell_begin = layout.qsh.system_offsets[kTarget];
  const double nan = std::numeric_limits<double>::quiet_NaN();

  /* Snapshot the failed peer's public population before the launch: a peer
   * failure must never rewrite its own public multipoles. */
  std::vector<double> initial_qsh;
  std::vector<double> initial_qat;
  CHECK(download(fixture.binding.state.raw_population.qsh,
                 fixture.binding.state.raw_population.qsh_elements, initial_qsh,
                 fixture.handles.stream()));
  CHECK(download(fixture.binding.state.raw_population.qat,
                 fixture.binding.state.raw_population.qat_elements, initial_qat,
                 fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  CHECK(upload_value(nan,
                     fixture.binding.state.scc.current_inputs.shell_charges + target_shell_begin,
                     fixture.handles.stream()));

  const Gfn2SccIterationLaunchResult launch =
      launch_gfn2_restricted_scc_iteration_cuda(fixture.binding, fixture.handles.stream());
  CHECK(launch.success());

  std::vector<std::uint64_t> iterations;
  std::vector<gpuxtb_status_t> statuses;
  std::vector<std::uint64_t> failures;
  std::vector<double> public_qsh;
  std::vector<double> public_qat;
  std::uint64_t plan_failure = 1u;
  CHECK(download(fixture.binding.state.scc.iterations, fixture.host.batch_size(), iterations,
                 fixture.handles.stream()));
  CHECK(download(fixture.binding.state.scc.system_statuses, fixture.host.batch_size(), statuses,
                 fixture.handles.stream()));
  CHECK(download(fixture.binding.workspace.ledger.system_failure_records, fixture.host.batch_size(),
                 failures, fixture.handles.stream()));
  CHECK(download(fixture.binding.state.raw_population.qsh,
                 fixture.binding.state.raw_population.qsh_elements, public_qsh,
                 fixture.handles.stream()));
  CHECK(download(fixture.binding.state.raw_population.qat,
                 fixture.binding.state.raw_population.qat_elements, public_qat,
                 fixture.handles.stream()));
  CHECK(download_value(fixture.binding.workspace.ledger.plan_failure_record, plan_failure,
                       fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  CHECK(plan_failure == 0u);
  CHECK(iterations[static_cast<std::size_t>(kTarget)] == 0u);
  CHECK(statuses[static_cast<std::size_t>(kTarget)] == GPUXTB_STATUS_INTERNAL_ERROR);
  CHECK(failures[static_cast<std::size_t>(kTarget)] ==
        failure_record(
            Gfn2SccStageId::kMixedGather,
            static_cast<std::uint32_t>(Gfn2SccPotentialDeviceError::kNonfiniteMixedShellCharge)));
  /* A failed peer leaves its own public multipoles untouched, exactly as the
   * production failure-publication policy (#87) requires. */
  CHECK(system_slice_is_byte_stable(initial_qsh, public_qsh, layout.qsh.system_offsets, kTarget));
  CHECK(system_slice_is_byte_stable(initial_qat, public_qat, layout.qat.system_offsets, kTarget));
  /* Peers 0/2/3 continue through the complete DAG and publish new finite
   * population; their per-peer iteration/status/failure ledger matches the
   * expected peer-local isolation (no CPU reference values are compared here
   * because the injected NaN has no CPU equivalent). */
  for (std::int64_t system = 0; system < fixture.host.batch_size(); ++system) {
    if (system == kTarget) {
      continue;
    }
    CHECK(iterations[static_cast<std::size_t>(system)] == 1u);
    CHECK(statuses[static_cast<std::size_t>(system)] == GPUXTB_STATUS_SUCCESS);
    CHECK(failures[static_cast<std::size_t>(system)] == 0u);
    CHECK(system_slice_is_finite(public_qsh, layout.qsh.system_offsets, system));
    CHECK(system_slice_is_finite(public_qat, layout.qat.system_offsets, system));
  }
  return 0;
}

/* A per-system geometry-provenance failure (stale geometry generation) is
 * attributed to the owning peer at the kGeometry stage: it suppresses every
 * downstream stage of that peer's DAG, publishes a quiet NaN energy plus the
 * canonical stage record, and leaves the population/publication buffers
 * byte-stable. Healthy peers keep advancing through their complete DAG. */
int test_composer_plan_provenance_failure_dag() {
  constexpr std::int64_t kTarget = 0;
  ComposerFixture fixture;
  CHECK(fixture.create(4, true));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  PublicSnapshot before;
  CHECK(snapshot_public(fixture, before) == 0);

  const std::uint64_t stale_generation = kGeometryGeneration - 1u;
  CHECK(upload(&stale_generation, fixture.binding.plan.geometry_cache.geometry_generations, 1,
               fixture.handles.stream()));

  const Gfn2SccIterationLaunchResult launch =
      launch_gfn2_restricted_scc_iteration_cuda(fixture.binding, fixture.handles.stream());
  CHECK(launch.success());

  PublicSnapshot after;
  CHECK(snapshot_public(fixture, after) == 0);
  const auto& layout = fixture.host.wavefunction_layout();

  std::vector<std::uint64_t> failures;
  std::uint64_t plan_failure = 1u;
  CHECK(download(fixture.binding.workspace.ledger.system_failure_records, fixture.host.batch_size(),
                 failures, fixture.handles.stream()));
  CHECK(download_value(fixture.binding.workspace.ledger.plan_failure_record, plan_failure,
                       fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  CHECK(plan_failure == 0u);
  CHECK(after.iterations[static_cast<std::size_t>(kTarget)] == 0u);
  CHECK(after.statuses[static_cast<std::size_t>(kTarget)] == GPUXTB_STATUS_INTERNAL_ERROR);
  CHECK(failures[static_cast<std::size_t>(kTarget)] ==
        failure_record(Gfn2SccStageId::kGeometry,
                       static_cast<std::uint32_t>(Gfn2SccIterationControlCode::kStaleGeneration)));
  /* The peer failure publishes no partial numerical data: population and
   * published multipole slices are byte-stable, while the failed peer's
   * per-system energy slice is filled with a quiet NaN by the failure
   * publication policy (never a partial or stale finite value). */
  CHECK(system_slice_is_byte_stable(before.qsh, after.qsh, layout.qsh.system_offsets, kTarget));
  CHECK(system_slice_is_byte_stable(before.qat, after.qat, layout.qat.system_offsets, kTarget));
  CHECK(system_slice_is_byte_stable(before.published_shell_charges, after.published_shell_charges,
                                    layout.qsh.system_offsets, kTarget));
  CHECK(std::isnan(after.free_energies[static_cast<std::size_t>(kTarget)]));
  /* Healthy peers advance exactly one iteration. */
  for (std::int64_t system = 1; system < fixture.host.batch_size(); ++system) {
    CHECK(after.iterations[static_cast<std::size_t>(system)] ==
          before.iterations[static_cast<std::size_t>(system)] + 1u);
    CHECK(after.statuses[static_cast<std::size_t>(system)] == GPUXTB_STATUS_SUCCESS);
    CHECK(failures[static_cast<std::size_t>(system)] == 0u);
  }
  return 0;
}

/*
 * Poison one inactive peer's geometry coordination/generation, complete SCC
 * multipoles, mixer vectors/history, aggregate SCC energies, and multipole
 * publication buffers. Every poisoned slice must remain stable while active
 * peers match an unpoisoned CPU transition. The inactive peer is selected by a
 * terminal state (converged/iteration count), as a converged predecessor would
 * be in the production loop.
 */
int test_composer_inactive_dormant_poisoning() {
  constexpr std::int64_t kInactive = 1;
  constexpr double kSentinel = -777.125;
  ComposerFixture fixture;
  CHECK(fixture.create(4, true));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  const auto& layout = fixture.host.wavefunction_layout();
  const auto& state = fixture.binding.state;
  const auto& workspace = fixture.binding.workspace;

  std::vector<std::uint64_t> iterations(static_cast<std::size_t>(fixture.host.batch_size()), 0u);
  std::vector<gpuxtb_status_t> statuses(static_cast<std::size_t>(fixture.host.batch_size()),
                                        GPUXTB_STATUS_SUCCESS);
  std::vector<std::uint8_t> converged(static_cast<std::size_t>(fixture.host.batch_size()), 0u);
  converged[static_cast<std::size_t>(kInactive)] = 1u;
  iterations[static_cast<std::size_t>(kInactive)] = fixture.host.options().maximum_iterations;
  fixture.host.driver_state().converged[kInactive] = 1u;
  fixture.host.driver_state().iterations[kInactive] = fixture.host.options().maximum_iterations;
  fixture.host.driver_state().system_statuses[kInactive] = GPUXTB_STATUS_SUCCESS;
  CHECK(upload(iterations.data(), state.scc.iterations, fixture.host.batch_size(),
               fixture.handles.stream()));
  CHECK(upload(statuses.data(), state.scc.system_statuses, fixture.host.batch_size(),
               fixture.handles.stream()));
  CHECK(upload(converged.data(), state.scc.converged, fixture.host.batch_size(),
               fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  /* Poison the inactive peer's per-system slices. The inactive peer owns
   * shells [shell_begin, shell_end), atoms [atom_begin, atom_end), and mixer
   * vector [vector_begin, vector_end) in each history column. */
  const std::int64_t shell_begin = layout.qsh.system_offsets[kInactive];
  const std::int64_t shell_end = layout.qsh.system_offsets[kInactive + 1];
  const std::int64_t qat_begin = layout.qat.system_offsets[kInactive];
  const std::int64_t qat_end = layout.qat.system_offsets[kInactive + 1];
  const std::int64_t dipole_begin = layout.dipole.system_offsets[kInactive];
  const std::int64_t dipole_end = layout.dipole.system_offsets[kInactive + 1];
  const std::int64_t quadrupole_begin = layout.quadrupole.system_offsets[kInactive];
  const std::int64_t quadrupole_end = layout.quadrupole.system_offsets[kInactive + 1];
  const std::int64_t atom_begin = fixture.host.atom_offsets()[static_cast<std::size_t>(kInactive)];
  const std::int64_t atom_end =
      fixture.host.atom_offsets()[static_cast<std::size_t>(kInactive + 1)];
  const auto& vector_offsets = fixture.host.mixer_plan().vector_offsets();
  const std::int64_t vector_begin = vector_offsets[static_cast<std::size_t>(kInactive)];
  const std::int64_t vector_end = vector_offsets[static_cast<std::size_t>(kInactive + 1)];
  const std::int64_t mixer_vector = vector_end - vector_begin;
  const std::int64_t history_size = fixture.host.mixer_plan().history_size();

  /* Geometry: coordination numbers and per-system generations. */
  CHECK(upload_fill(
      const_cast<double*>(fixture.binding.plan.geometry_cache.coordination_numbers) + atom_begin,
      atom_end - atom_begin, kSentinel, fixture.handles.stream()));
  std::vector<std::uint64_t> sentinel_generation{std::numeric_limits<std::uint64_t>::max()};
  CHECK(upload(&sentinel_generation[0],
               fixture.binding.plan.geometry_cache.geometry_generations + kInactive, 1,
               fixture.handles.stream()));
  /* Complete mixed q/d/Q state. Atomic charges are derived from qsh and are
   * present only in the public and staged Mulliken populations below. */
  CHECK(upload_fill(const_cast<double*>(state.scc.current_inputs.shell_charges) + shell_begin,
                    shell_end - shell_begin, kSentinel, fixture.handles.stream()));
  CHECK(upload_fill(const_cast<double*>(state.scc.current_inputs.atomic_dipoles) + dipole_begin,
                    dipole_end - dipole_begin, kSentinel, fixture.handles.stream()));
  CHECK(upload_fill(
      const_cast<double*>(state.scc.current_inputs.atomic_quadrupoles) + quadrupole_begin,
      quadrupole_end - quadrupole_begin, kSentinel, fixture.handles.stream()));
  /* Mixer histories: df, u, and omega slices. */
  CHECK(upload_fill(state.mixer.current_inputs + vector_begin, mixer_vector, kSentinel,
                    fixture.handles.stream()));
  CHECK(upload_fill(state.mixer.previous_inputs + vector_begin, mixer_vector, kSentinel,
                    fixture.handles.stream()));
  CHECK(upload_fill(state.mixer.previous_residuals + vector_begin, mixer_vector, kSentinel,
                    fixture.handles.stream()));
  CHECK(upload_fill(state.mixer.df_history + vector_begin * history_size,
                    mixer_vector * history_size, kSentinel, fixture.handles.stream()));
  CHECK(upload_fill(state.mixer.u_history + vector_begin * history_size,
                    mixer_vector * history_size, kSentinel, fixture.handles.stream()));
  CHECK(upload_fill(state.mixer.omega + kInactive * history_size, history_size, kSentinel,
                    fixture.handles.stream()));
  /* Energies. */
  CHECK(upload_fill(state.free_energy.internal_energy + kInactive, 1, kSentinel,
                    fixture.handles.stream()));
  CHECK(upload_fill(state.free_energy.free_energy + kInactive, 1, kSentinel,
                    fixture.handles.stream()));
  CHECK(upload_fill(state.free_energy.entropy + kInactive, 1, kSentinel, fixture.handles.stream()));
  CHECK(upload_fill(state.scc.free_energies + kInactive, 1, kSentinel, fixture.handles.stream()));
  /* Complete multipole publication and unpublished raw Mulliken scratch.
   * Poisoning every qsh/qat/dipole/quadrupole slice makes omissions in either
   * read or write gating observable. */
  CHECK(upload_fill(state.raw_population.qsh + shell_begin, shell_end - shell_begin, kSentinel,
                    fixture.handles.stream()));
  CHECK(upload_fill(state.raw_population.qat + qat_begin, qat_end - qat_begin, kSentinel,
                    fixture.handles.stream()));
  CHECK(upload_fill(state.raw_population.dipole + dipole_begin, dipole_end - dipole_begin,
                    kSentinel, fixture.handles.stream()));
  CHECK(upload_fill(state.raw_population.quadrupole + quadrupole_begin,
                    quadrupole_end - quadrupole_begin, kSentinel, fixture.handles.stream()));
  CHECK(upload_fill(state.published.shell_charges + shell_begin, shell_end - shell_begin, kSentinel,
                    fixture.handles.stream()));
  CHECK(upload_fill(state.published.atomic_dipoles + dipole_begin, dipole_end - dipole_begin,
                    kSentinel, fixture.handles.stream()));
  CHECK(upload_fill(state.published.atomic_quadrupoles + quadrupole_begin,
                    quadrupole_end - quadrupole_begin, kSentinel, fixture.handles.stream()));
  CHECK(upload_fill(workspace.staged_raw_population.qsh + shell_begin, shell_end - shell_begin,
                    kSentinel, fixture.handles.stream()));
  CHECK(upload_fill(workspace.staged_raw_population.qat + qat_begin, qat_end - qat_begin, kSentinel,
                    fixture.handles.stream()));
  CHECK(upload_fill(workspace.staged_raw_population.dipole + dipole_begin,
                    dipole_end - dipole_begin, kSentinel, fixture.handles.stream()));
  CHECK(upload_fill(workspace.staged_raw_population.quadrupole + quadrupole_begin,
                    quadrupole_end - quadrupole_begin, kSentinel, fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  const Gfn2SccIterationLaunchResult launch =
      launch_gfn2_restricted_scc_iteration_cuda(fixture.binding, fixture.handles.stream());
  CHECK(launch.success());
  std::string error;
  CHECK(fixture.host.run_one_iteration(error) == GPUXTB_STATUS_SUCCESS);

  std::vector<double> coordination_numbers;
  std::vector<double> current_shell_charges;
  std::vector<double> current_dipoles;
  std::vector<double> current_quadrupoles;
  std::vector<double> mixer_current_inputs;
  std::vector<double> mixer_previous_inputs;
  std::vector<double> mixer_previous_residuals;
  std::vector<double> df_history;
  std::vector<double> u_history;
  std::vector<double> omega;
  std::vector<double> internal_energy;
  std::vector<double> free_energy;
  std::vector<double> entropy;
  std::vector<double> scc_free_energies;
  std::vector<double> published_qsh;
  std::vector<double> published_qat;
  std::vector<double> published_dipoles;
  std::vector<double> published_quadrupoles;
  std::vector<double> published_mixed_qsh;
  std::vector<double> published_mixed_dipoles;
  std::vector<double> published_mixed_quadrupoles;
  std::vector<double> staged_qsh;
  std::vector<double> staged_qat;
  std::vector<double> staged_dipoles;
  std::vector<double> staged_quadrupoles;
  std::vector<std::uint64_t> geometry_generations;
  std::vector<std::uint64_t> after_iterations;
  std::vector<gpuxtb_status_t> after_statuses;
  std::vector<std::uint8_t> after_converged;
  CHECK(download(fixture.binding.plan.geometry_cache.coordination_numbers,
                 fixture.host.total_atoms(), coordination_numbers, fixture.handles.stream()));
  CHECK(download(fixture.binding.plan.geometry_cache.geometry_generations,
                 fixture.host.batch_size(), geometry_generations, fixture.handles.stream()));
  CHECK(download(state.scc.current_inputs.shell_charges, layout.qsh.element_count,
                 current_shell_charges, fixture.handles.stream()));
  CHECK(download(state.scc.current_inputs.atomic_dipoles, layout.dipole.element_count,
                 current_dipoles, fixture.handles.stream()));
  CHECK(download(state.scc.current_inputs.atomic_quadrupoles, layout.quadrupole.element_count,
                 current_quadrupoles, fixture.handles.stream()));
  CHECK(download(state.mixer.current_inputs, state.mixer.total_vector_elements,
                 mixer_current_inputs, fixture.handles.stream()));
  CHECK(download(state.mixer.previous_inputs, state.mixer.total_vector_elements,
                 mixer_previous_inputs, fixture.handles.stream()));
  CHECK(download(state.mixer.previous_residuals, state.mixer.total_vector_elements,
                 mixer_previous_residuals, fixture.handles.stream()));
  CHECK(download(state.mixer.df_history, state.mixer.history_elements, df_history,
                 fixture.handles.stream()));
  CHECK(download(state.mixer.u_history, state.mixer.history_elements, u_history,
                 fixture.handles.stream()));
  CHECK(download(state.mixer.omega, state.mixer.omega_elements, omega, fixture.handles.stream()));
  CHECK(download(state.free_energy.internal_energy, fixture.host.batch_size(), internal_energy,
                 fixture.handles.stream()));
  CHECK(download(state.free_energy.free_energy, fixture.host.batch_size(), free_energy,
                 fixture.handles.stream()));
  CHECK(download(state.free_energy.entropy, fixture.host.batch_size(), entropy,
                 fixture.handles.stream()));
  CHECK(download(state.scc.free_energies, fixture.host.batch_size(), scc_free_energies,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.qsh, layout.qsh.element_count, published_qsh,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.qat, layout.qat.element_count, published_qat,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.dipole, layout.dipole.element_count, published_dipoles,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.quadrupole, layout.quadrupole.element_count,
                 published_quadrupoles, fixture.handles.stream()));
  CHECK(download(state.published.shell_charges, layout.qsh.element_count, published_mixed_qsh,
                 fixture.handles.stream()));
  CHECK(download(state.published.atomic_dipoles, layout.dipole.element_count,
                 published_mixed_dipoles, fixture.handles.stream()));
  CHECK(download(state.published.atomic_quadrupoles, layout.quadrupole.element_count,
                 published_mixed_quadrupoles, fixture.handles.stream()));
  CHECK(download(workspace.staged_raw_population.qsh, layout.qsh.element_count, staged_qsh,
                 fixture.handles.stream()));
  CHECK(download(workspace.staged_raw_population.qat, layout.qat.element_count, staged_qat,
                 fixture.handles.stream()));
  CHECK(download(workspace.staged_raw_population.dipole, layout.dipole.element_count,
                 staged_dipoles, fixture.handles.stream()));
  CHECK(download(workspace.staged_raw_population.quadrupole, layout.quadrupole.element_count,
                 staged_quadrupoles, fixture.handles.stream()));
  CHECK(download(state.scc.converged, fixture.host.batch_size(), after_converged,
                 fixture.handles.stream()));
  CHECK(download(state.scc.iterations, fixture.host.batch_size(), after_iterations,
                 fixture.handles.stream()));
  CHECK(download(state.scc.system_statuses, fixture.host.batch_size(), after_statuses,
                 fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  /* Every poisoned inactive slice remains byte-stable. */
  for (std::int64_t index = atom_begin; index < atom_end; ++index) {
    CHECK(coordination_numbers[static_cast<std::size_t>(index)] == kSentinel);
  }
  CHECK(geometry_generations[static_cast<std::size_t>(kInactive)] ==
        std::numeric_limits<std::uint64_t>::max());
  for (std::int64_t index = shell_begin; index < shell_end; ++index) {
    CHECK(current_shell_charges[static_cast<std::size_t>(index)] == kSentinel);
    CHECK(published_qsh[static_cast<std::size_t>(index)] == kSentinel);
    CHECK(published_mixed_qsh[static_cast<std::size_t>(index)] == kSentinel);
    CHECK(staged_qsh[static_cast<std::size_t>(index)] == kSentinel);
  }
  for (std::int64_t index = qat_begin; index < qat_end; ++index) {
    CHECK(published_qat[static_cast<std::size_t>(index)] == kSentinel);
    CHECK(staged_qat[static_cast<std::size_t>(index)] == kSentinel);
  }
  for (std::int64_t index = dipole_begin; index < dipole_end; ++index) {
    CHECK(current_dipoles[static_cast<std::size_t>(index)] == kSentinel);
    CHECK(published_dipoles[static_cast<std::size_t>(index)] == kSentinel);
    CHECK(published_mixed_dipoles[static_cast<std::size_t>(index)] == kSentinel);
    CHECK(staged_dipoles[static_cast<std::size_t>(index)] == kSentinel);
  }
  for (std::int64_t index = quadrupole_begin; index < quadrupole_end; ++index) {
    CHECK(current_quadrupoles[static_cast<std::size_t>(index)] == kSentinel);
    CHECK(published_quadrupoles[static_cast<std::size_t>(index)] == kSentinel);
    CHECK(published_mixed_quadrupoles[static_cast<std::size_t>(index)] == kSentinel);
    CHECK(staged_quadrupoles[static_cast<std::size_t>(index)] == kSentinel);
  }
  for (std::int64_t index = vector_begin; index < vector_end; ++index) {
    CHECK(mixer_current_inputs[static_cast<std::size_t>(index)] == kSentinel);
    CHECK(mixer_previous_inputs[static_cast<std::size_t>(index)] == kSentinel);
    CHECK(mixer_previous_residuals[static_cast<std::size_t>(index)] == kSentinel);
  }
  for (std::int64_t index = vector_begin * history_size; index < vector_end * history_size;
       ++index) {
    CHECK(df_history[static_cast<std::size_t>(index)] == kSentinel);
    CHECK(u_history[static_cast<std::size_t>(index)] == kSentinel);
  }
  for (std::int64_t index = kInactive * history_size; index < (kInactive + 1) * history_size;
       ++index) {
    CHECK(omega[static_cast<std::size_t>(index)] == kSentinel);
  }
  CHECK(internal_energy[static_cast<std::size_t>(kInactive)] == kSentinel);
  CHECK(free_energy[static_cast<std::size_t>(kInactive)] == kSentinel);
  CHECK(entropy[static_cast<std::size_t>(kInactive)] == kSentinel);
  CHECK(scc_free_energies[static_cast<std::size_t>(kInactive)] == kSentinel);
  CHECK(after_iterations[static_cast<std::size_t>(kInactive)] ==
        fixture.host.options().maximum_iterations);
  CHECK(after_statuses[static_cast<std::size_t>(kInactive)] == GPUXTB_STATUS_SUCCESS);
  CHECK(after_converged[static_cast<std::size_t>(kInactive)] == 1u);

  /* The unpoisoned CPU transition is the control for every active peer. This
   * proves inactive sentinel values were not cross-read into otherwise finite
   * active output. */
  for (std::int64_t system = 0; system < fixture.host.batch_size(); ++system) {
    if (system == kInactive) {
      continue;
    }
    CHECK(after_iterations[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().iterations[system]);
    CHECK(after_statuses[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().system_statuses[system]);
    CHECK(after_converged[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().converged[system]);
    CHECK(compare_system_slice("active public qsh", published_qsh, fixture.host.wavefunction().qsh,
                               layout.qsh.system_offsets, system, 3.0e-9));
    CHECK(compare_system_slice("active public qat", published_qat, fixture.host.wavefunction().qat,
                               layout.qat.system_offsets, system, 3.0e-9));
    CHECK(compare_system_slice("active public dipoles", published_dipoles,
                               fixture.host.wavefunction().dipole, layout.dipole.system_offsets,
                               system, 3.0e-9));
    CHECK(compare_system_slice("active public quadrupoles", published_quadrupoles,
                               fixture.host.wavefunction().quadrupole,
                               layout.quadrupole.system_offsets, system, 3.0e-9));
    CHECK(compare_system_slice("active raw qsh", staged_qsh,
                               fixture.host.driver_workspace().raw_qsh, layout.qsh.system_offsets,
                               system, 3.0e-9));
    CHECK(compare_system_slice("active raw qat", staged_qat,
                               fixture.host.driver_workspace().raw_qat, layout.qat.system_offsets,
                               system, 3.0e-9));
    CHECK(compare_system_slice("active raw dipoles", staged_dipoles,
                               fixture.host.driver_workspace().raw_dipoles,
                               layout.dipole.system_offsets, system, 3.0e-9));
    CHECK(compare_system_slice("active raw quadrupoles", staged_quadrupoles,
                               fixture.host.driver_workspace().raw_quadrupoles,
                               layout.quadrupole.system_offsets, system, 3.0e-9));
    CHECK(compare_system_slice("active mixer inputs", mixer_current_inputs,
                               fixture.host.mixer_state().current_inputs, vector_offsets, system,
                               3.0e-9));
    CHECK(near(scc_free_energies[static_cast<std::size_t>(system)],
               fixture.host.driver_state().free_energies[system], 3.0e-9));
  }
  return 0;
}

/* The same binding executes on the default stream and stays CPU-parity exact,
 * proving stream-identity independence for the composer. */
int test_composer_default_stream_parity() {
  ComposerFixture fixture;
  CHECK(fixture.create(8, false));
  /* All setup uploads ran on the custom nonblocking stream; establish stream
   * ordering before reusing the binding on the default stream. */
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  const Gfn2SccIterationLaunchResult launch =
      launch_gfn2_restricted_scc_iteration_cuda(fixture.binding, nullptr);
  CHECK(launch.success());
  CUDA_CHECK(cudaStreamSynchronize(nullptr));
  std::string error;
  CHECK(fixture.host.run_one_iteration(error) == GPUXTB_STATUS_SUCCESS);

  const auto& layout = fixture.host.wavefunction_layout();
  std::vector<double> public_qsh;
  std::vector<double> raw_qsh;
  std::vector<double> free_energy;
  std::vector<std::uint64_t> iterations;
  CHECK(download(fixture.binding.state.raw_population.qsh,
                 fixture.binding.state.raw_population.qsh_elements, public_qsh, nullptr));
  CHECK(download(fixture.binding.workspace.staged_raw_population.qsh,
                 fixture.binding.workspace.staged_raw_population.qsh_elements, raw_qsh, nullptr));
  CHECK(download(fixture.binding.state.free_energy.free_energy,
                 fixture.binding.state.free_energy.free_energy_elements, free_energy, nullptr));
  CHECK(download(fixture.binding.state.scc.iterations, fixture.host.batch_size(), iterations,
                 nullptr));
  CUDA_CHECK(cudaStreamSynchronize(nullptr));
  CHECK(compare_doubles("default-stream public qsh", public_qsh, fixture.host.wavefunction().qsh,
                        layout.qsh.element_count, 3.0e-9));
  CHECK(compare_doubles("default-stream raw qsh", raw_qsh, fixture.host.driver_workspace().raw_qsh,
                        layout.qsh.element_count, 3.0e-9));
  CHECK(compare_doubles("default-stream free energy", free_energy,
                        fixture.host.driver_state().free_energies, fixture.host.batch_size(),
                        3.0e-9));
  CHECK(compare_exact_values("default-stream iterations", iterations,
                             fixture.host.driver_state().iterations, fixture.host.batch_size()));
  return 0;
}

int run_all() {
  int status = test_composer_binding_and_repeat_launch();
  if (status != 0) {
    return status;
  }
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    status = test_composer_one_step_cpu_parity(batch_size, false);
    if (status != 0) {
      return status;
    }
    status = test_composer_one_step_cpu_parity(batch_size, true);
    if (status != 0) {
      return status;
    }
    status = test_composer_changed_input_graph_replay(batch_size, false);
    if (status != 0) {
      return status;
    }
  }
  status = test_composer_changed_input_graph_replay(8, true);
  if (status != 0) {
    return status;
  }
  status = test_composer_raw_publication_policy(false);
  if (status != 0) {
    return status;
  }
  status = test_composer_raw_publication_policy(true);
  if (status != 0) {
    return status;
  }
  status = test_composer_peer_numerical_failure_dag();
  if (status != 0) {
    return status;
  }
  status = test_composer_plan_provenance_failure_dag();
  if (status != 0) {
    return status;
  }
  status = test_composer_inactive_dormant_poisoning();
  if (status != 0) {
    return status;
  }
  status = test_composer_default_stream_parity();
  if (status != 0) {
    return status;
  }
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
  if (count_status != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceCount: %s\n", cudaGetErrorString(count_status));
    return 1;
  }
  CUDA_CHECK(cudaSetDevice(0));
  return run_all();
}
