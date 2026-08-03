#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
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
#include "tests/support/gfn2_scc_test_case.hpp"

#define CHECK(condition)                                                                          \
  do {                                                                                            \
    if (!(condition)) {                                                                           \
      std::fprintf(stderr, "production SCC check failed at line %d: %s\n", __LINE__, #condition); \
      return __LINE__;                                                                            \
    }                                                                                             \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using namespace gpuxtb::detail;
using namespace gpuxtb::detail::cuda;
using gpuxtb::test::gfn2::HostSccCase;
using gpuxtb::test::gfn2::HostSccCaseOptions;
using gpuxtb::test::gfn2::SmallSystemKind;

constexpr std::uint64_t kPlanToken = 0x105105105ULL;
constexpr std::uint64_t kGeometryGeneration = 105u;
constexpr std::uint64_t kInitializationGeneration = 1u;

template <typename T>
Gfn2SccSetupHostArray<T> setup_view(const std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), static_cast<std::int64_t>(values.size())};
}

template <typename T>
Gfn2SccIterationHostArrayView<T> initialization_view(const T* values,
                                                     std::int64_t elements) noexcept {
  return {elements == 0 ? nullptr : values, elements};
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
 * geometry-pair cache. The SCC iteration only validates that common cache in
 * this energy-only path; geometry VJPs consume it later. A deterministic
 * finite image therefore supplies the production owner until the host fixture
 * grows a common geometry-cache accessor. */
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

template <typename T>
bool download(const T* device, std::int64_t elements, std::vector<T>& host, cudaStream_t stream) {
  if (elements < 0) {
    return false;
  }
  host.resize(static_cast<std::size_t>(elements));
  return elements == 0 || cudaMemcpyAsync(host.data(), device, host.size() * sizeof(T),
                                          cudaMemcpyDeviceToHost, stream) == cudaSuccess;
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
    /* Terminal and failed peers intentionally publish the CPU driver's NaN
     * energy trace. Treat matching NaNs as parity, while all finite values
     * still use the regular scaled tolerance below. */
    if (std::isnan(actual[static_cast<std::size_t>(index)]) && std::isnan(expected[index])) {
      continue;
    }
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

struct ProductionFixture {
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
  Gfn2SccIterationDevicePlan plan_seed{};
  Gfn2SccIterationDeviceInput input_seed{};
  Gfn2SccIterationArenaRequirements arena_requirements{};
  Gfn2SccIterationDeviceState state_seed{};
  Gfn2SccIterationDeviceWorkspace workspace_seed{};
  Gfn2SccIterationReportStorage report_storage{};
  Gfn2SccSetupEigensolverBinding eigensolver_binding{};
  Gfn2SccIterationInitializationReady ready{};
  Gfn2SccIterationBinding binding{};

  bool create(bool optional_components, std::uint64_t maximum_iterations = 8u) {
    HostSccCaseOptions options{};
    options.systems = {SmallSystemKind::kH2, SmallSystemKind::kHe, SmallSystemKind::kLiH,
                       SmallSystemKind::kCH2};
    options.geometry_generation = kGeometryGeneration;
    options.maximum_iterations = maximum_iterations;
    options.mixer_history = 3;
    options.electronic_temperature = 0.0;
    options.enable_d4 = optional_components;
    options.enable_explicit_point_charges = optional_components;
    options.enable_periodic_embedding = optional_components;

    std::string error;
    if (HostSccCase::create(options, host, error) != GPUXTB_STATUS_SUCCESS) {
      std::fprintf(stderr, "HostSccCase::create failed: %s\n", error.c_str());
      return false;
    }
    if (!backing.prepare(host, error) || !handles.create()) {
      std::fprintf(stderr, "production SCC host/provider setup failed: %s\n", error.c_str());
      return false;
    }

    auto topology_diagnostic =
        Gfn2SccSetupTopology::create(host.basis_plan(), host.integral_plan(),
                                     host.wavefunction_layout(), kPlanToken, topology_owner);
    if (!topology_diagnostic.success() ||
        !topology_arena.allocate(topology_owner.requirements().immutable_device_bytes)) {
      std::fprintf(stderr, "production SCC topology create/allocation failed\n");
      return false;
    }
    topology_diagnostic = topology_owner.bind_device_arena_and_upload_async(
        topology_arena.get(), topology_arena.bytes(), device_topology, handles.stream());
    if (!topology_diagnostic.success()) {
      std::fprintf(stderr, "production SCC topology upload failed: error=%u field=%u\n",
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
                   "production SCC immutable-input create/allocation failed: error=%u "
                   "field=%u index=%lld\n",
                   static_cast<unsigned>(input_diagnostic.error),
                   static_cast<unsigned>(input_diagnostic.field),
                   static_cast<long long>(input_diagnostic.index));
      return false;
    }
    input_diagnostic = inputs_owner.bind_device_arena_and_upload_async(
        device_topology, input_arena.get(), input_arena.bytes(), plan_seed, input_seed,
        handles.stream());
    if (!input_diagnostic.success()) {
      std::fprintf(stderr, "production SCC immutable-input upload failed: error=%u field=%u\n",
                   static_cast<unsigned>(input_diagnostic.error),
                   static_cast<unsigned>(input_diagnostic.field));
      return false;
    }

    auto eigensolver_diagnostic = Gfn2SccSetupEigensolver::create(
        topology_owner, host.overlap().data(), static_cast<std::int64_t>(host.overlap().size()),
        kGeometryGeneration, kPlanToken, handles.solver(), handles.parameters(), handles.blas(),
        plan_seed.eigensolver_options, eigensolver_owner);
    if (!eigensolver_diagnostic.success()) {
      std::fprintf(stderr, "production SCC eigensolver owner create failed: error=%u field=%u\n",
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
      std::fprintf(stderr, "production SCC iteration-arena query/allocation failed: error=%u\n",
                   static_cast<unsigned>(arena_diagnostic.error));
      return false;
    }

    auto bind_arena_diagnostic = bind_gfn2_scc_iteration_arena_cuda(
        plan_seed, eigensolver_requirements.provider, arena_requirements, iteration_arena.get(),
        iteration_arena.bytes(), provider_host_workspace.get(), provider_host_workspace.bytes(),
        state_seed, workspace_seed, report_storage);
    if (!bind_arena_diagnostic.success() ||
        !eigensolver_setup_arena.allocate(eigensolver_requirements.setup_device_bytes)) {
      std::fprintf(stderr, "production SCC iteration-arena bind failed: error=%u\n",
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
                   "production SCC overlap factorization submission failed: error=%u field=%u "
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

    const Gfn2SccIterationHostInitialization host_initialization = fresh_initialization(host);
    auto initialization_diagnostic = Gfn2SccIterationInitializer::create(
        plan_seed, arena_requirements, iteration_arena.get(), iteration_arena.bytes(), state_seed,
        workspace_seed, report_storage, host_initialization, initializer);
    if (!initialization_diagnostic.success()) {
      std::fprintf(stderr,
                   "production SCC initializer create failed: error=%u field=%u "
                   "index=%lld\n",
                   static_cast<unsigned>(initialization_diagnostic.error),
                   static_cast<unsigned>(initialization_diagnostic.field),
                   static_cast<long long>(initialization_diagnostic.index));
      return false;
    }
    initialization_diagnostic = initializer.upload_async(
        iteration_arena.get(), iteration_arena.bytes(), ready, handles.stream());
    if (!initialization_diagnostic.success()) {
      std::fprintf(stderr, "production SCC initializer upload failed: error=%u field=%u\n",
                   static_cast<unsigned>(initialization_diagnostic.error),
                   static_cast<unsigned>(initialization_diagnostic.field));
      return false;
    }

    const auto report_diagnostic = build_gfn2_scc_iteration_report_binding_cuda(
        report_storage, plan_seed, input_seed, state_seed, workspace_seed, binding);
    if (report_diagnostic.error != Gfn2SccIterationBindingError::kSuccess) {
      std::fprintf(stderr,
                   "production SCC report/binding build failed: error=%u field=%u "
                   "index=%lld\n",
                   static_cast<unsigned>(report_diagnostic.error),
                   static_cast<unsigned>(report_diagnostic.field),
                   static_cast<long long>(report_diagnostic.index));
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
bool download_value(const T* device, T& host, cudaStream_t stream) {
  return cudaMemcpyAsync(&host, device, sizeof(T), cudaMemcpyDeviceToHost, stream) == cudaSuccess;
}

std::uint64_t failure_record(Gfn2SccStageId stage, std::uint32_t code) {
  return gfn2_scc_stage_failure_record(stage, code);
}

int test_terminal_peer_cpu_parity() {
  ProductionFixture fixture;
  CHECK(fixture.create(false, 1u));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  /*
   * Exercise all three terminal predicates next to one last-allowed active
   * peer. The CPU and CUDA drivers start from identical public state; terminal
   * members must remain byte-stable while the final peer publishes
   * SCC_NOT_CONVERGED after its sole allowed attempt.
   */
  auto& cpu_driver = fixture.host.driver_state();
  auto& cpu_mixer = fixture.host.mixer_state();
  cpu_driver.converged[0] = 1u;
  cpu_driver.system_statuses[1] = GPUXTB_STATUS_INTERNAL_ERROR;
  cpu_mixer.system_statuses[1] = GPUXTB_STATUS_INTERNAL_ERROR;
  cpu_driver.iterations[2] = 1u;
  cpu_driver.system_statuses[2] = GPUXTB_STATUS_SCC_NOT_CONVERGED;
  cpu_mixer.iterations[2] = 1u;
  cpu_mixer.system_statuses[2] = GPUXTB_STATUS_SCC_NOT_CONVERGED;

  const std::vector<std::uint64_t> iterations{0u, 0u, 1u, 0u};
  const std::vector<gpuxtb_status_t> statuses{GPUXTB_STATUS_SUCCESS, GPUXTB_STATUS_INTERNAL_ERROR,
                                              GPUXTB_STATUS_SCC_NOT_CONVERGED,
                                              GPUXTB_STATUS_SUCCESS};
  const std::vector<std::uint8_t> converged{1u, 0u, 0u, 0u};
  CHECK(
      upload(iterations.data(), fixture.binding.state.scc.iterations, 4, fixture.handles.stream()));
  CHECK(upload(statuses.data(), fixture.binding.state.scc.system_statuses, 4,
               fixture.handles.stream()));
  CHECK(upload(converged.data(), fixture.binding.state.scc.converged, 4, fixture.handles.stream()));
  CHECK(upload(iterations.data(), fixture.binding.state.mixer.iterations, 4,
               fixture.handles.stream()));
  CHECK(upload(statuses.data(), fixture.binding.state.mixer.system_statuses, 4,
               fixture.handles.stream()));

  const Gfn2SccIterationLaunchResult launch =
      launch_gfn2_restricted_scc_iteration_cuda(fixture.binding, fixture.handles.stream());
  CHECK(launch.success());

  std::string error;
  CHECK(fixture.host.run_one_iteration(error) == GPUXTB_STATUS_SCC_NOT_CONVERGED);

  const auto& layout = fixture.host.wavefunction_layout();
  std::vector<double> qsh;
  std::vector<double> density;
  std::vector<double> free_energies;
  std::vector<std::uint64_t> actual_iterations;
  std::vector<gpuxtb_status_t> actual_statuses;
  std::vector<std::uint8_t> actual_converged;
  CHECK(download(fixture.binding.state.raw_population.qsh,
                 fixture.binding.state.raw_population.qsh_elements, qsh, fixture.handles.stream()));
  CHECK(download(fixture.binding.state.density.density,
                 fixture.binding.state.density.density_elements, density,
                 fixture.handles.stream()));
  CHECK(download(fixture.binding.state.scc.free_energies, fixture.binding.state.scc.batch_elements,
                 free_energies, fixture.handles.stream()));
  CHECK(download(fixture.binding.state.scc.iterations, fixture.binding.state.scc.batch_elements,
                 actual_iterations, fixture.handles.stream()));
  CHECK(download(fixture.binding.state.scc.system_statuses,
                 fixture.binding.state.scc.batch_elements, actual_statuses,
                 fixture.handles.stream()));
  CHECK(download(fixture.binding.state.scc.converged, fixture.binding.state.scc.batch_elements,
                 actual_converged, fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  CHECK(compare_doubles("terminal-peer qsh", qsh, fixture.host.wavefunction().qsh,
                        layout.qsh.element_count, 3.0e-9));
  CHECK(compare_doubles("terminal-peer density", density, fixture.host.wavefunction().density,
                        layout.density.element_count, 3.0e-9));
  CHECK(compare_doubles("terminal-peer free energies", free_energies,
                        fixture.host.driver_state().free_energies, fixture.host.batch_size(),
                        3.0e-9));
  for (std::int64_t system = 0; system < fixture.host.batch_size(); ++system) {
    CHECK(actual_iterations[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().iterations[system]);
    CHECK(actual_statuses[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().system_statuses[system]);
    CHECK(actual_converged[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().converged[system]);
  }
  CHECK(actual_iterations[0] == 0u && actual_converged[0] == 1u);
  CHECK(actual_iterations[1] == 0u && actual_statuses[1] == GPUXTB_STATUS_INTERNAL_ERROR);
  CHECK(actual_iterations[2] == 1u && actual_statuses[2] == GPUXTB_STATUS_SCC_NOT_CONVERGED);
  CHECK(actual_iterations[3] == 1u && actual_statuses[3] == GPUXTB_STATUS_SCC_NOT_CONVERGED);
  return 0;
}

int test_mixed_warm_loop_state() {
  ProductionFixture fixture;
  CHECK(fixture.create(false, 8u));

  std::string error;
  for (int iteration = 0; iteration < 2; ++iteration) {
    CHECK(launch_gfn2_restricted_scc_iteration_cuda(fixture.binding, fixture.handles.stream())
              .success());
    CHECK(fixture.host.run_one_iteration(error) == GPUXTB_STATUS_SUCCESS);
  }
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  /* Resume from one consistent nonzero checkpoint while mixing every terminal
   * predicate into the same batch. Only system zero remains active. */
  auto& cpu_driver = fixture.host.driver_state();
  auto& cpu_mixer = fixture.host.mixer_state();
  cpu_driver.converged[1] = 1u;
  cpu_mixer.converged[1] = 1u;
  cpu_driver.system_statuses[2] = GPUXTB_STATUS_INTERNAL_ERROR;
  cpu_mixer.system_statuses[2] = GPUXTB_STATUS_INTERNAL_ERROR;
  cpu_driver.iterations[3] = 8u;
  cpu_driver.system_statuses[3] = GPUXTB_STATUS_SCC_NOT_CONVERGED;

  std::vector<std::uint64_t> iterations(cpu_driver.iterations, cpu_driver.iterations + 4);
  std::vector<gpuxtb_status_t> statuses(cpu_driver.system_statuses, cpu_driver.system_statuses + 4);
  std::vector<std::uint8_t> converged(cpu_driver.converged, cpu_driver.converged + 4);
  std::vector<std::uint64_t> mixer_iterations(cpu_mixer.iterations, cpu_mixer.iterations + 4);
  std::vector<gpuxtb_status_t> mixer_statuses(cpu_mixer.system_statuses,
                                              cpu_mixer.system_statuses + 4);
  std::vector<std::uint8_t> mixer_converged(cpu_mixer.converged, cpu_mixer.converged + 4);
  CHECK(
      upload(iterations.data(), fixture.binding.state.scc.iterations, 4, fixture.handles.stream()));
  CHECK(upload(statuses.data(), fixture.binding.state.scc.system_statuses, 4,
               fixture.handles.stream()));
  CHECK(upload(converged.data(), fixture.binding.state.scc.converged, 4, fixture.handles.stream()));
  CHECK(upload(mixer_iterations.data(), fixture.binding.state.mixer.iterations, 4,
               fixture.handles.stream()));
  CHECK(upload(mixer_statuses.data(), fixture.binding.state.mixer.system_statuses, 4,
               fixture.handles.stream()));
  CHECK(upload(mixer_converged.data(), fixture.binding.state.mixer.residual_converged, 4,
               fixture.handles.stream()));

  const Gfn2SccLoopLaunchResult loop =
      launch_gfn2_restricted_scc_loop_cuda(fixture.binding, fixture.handles.stream());
  CHECK(loop.success());
  CHECK(loop.submitted_iterations == 8u);
  for (int iteration = 0; iteration < 8; ++iteration) {
    const gpuxtb_status_t status = fixture.host.run_one_iteration(error);
    CHECK(status == GPUXTB_STATUS_SUCCESS || status == GPUXTB_STATUS_SCC_NOT_CONVERGED);
  }

  std::vector<std::uint64_t> actual_iterations;
  std::vector<gpuxtb_status_t> actual_statuses;
  std::vector<std::uint8_t> actual_converged;
  CHECK(download(fixture.binding.state.scc.iterations, 4, actual_iterations,
                 fixture.handles.stream()));
  CHECK(download(fixture.binding.state.scc.system_statuses, 4, actual_statuses,
                 fixture.handles.stream()));
  CHECK(
      download(fixture.binding.state.scc.converged, 4, actual_converged, fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  for (std::size_t system = 0; system < 4u; ++system) {
    CHECK(actual_iterations[system] == cpu_driver.iterations[system]);
    CHECK(actual_statuses[system] == cpu_driver.system_statuses[system]);
    CHECK(actual_converged[system] == cpu_driver.converged[system]);
  }
  CHECK(actual_iterations[1] == iterations[1] && actual_converged[1] == 1u);
  CHECK(actual_iterations[2] == iterations[2] &&
        actual_statuses[2] == GPUXTB_STATUS_INTERNAL_ERROR);
  CHECK(actual_iterations[3] == 8u && actual_statuses[3] == GPUXTB_STATUS_SCC_NOT_CONVERGED);
  CHECK(actual_iterations[0] > iterations[0]);
  return 0;
}

int test_stale_active_and_inactive_provenance() {
  {
    ProductionFixture fixture;
    CHECK(fixture.create(false));
    const std::uint64_t stale_generation = kGeometryGeneration - 1u;
    CHECK(upload(&stale_generation, fixture.binding.plan.geometry_cache.geometry_generations, 1,
                 fixture.handles.stream()));
    CHECK(launch_gfn2_restricted_scc_iteration_cuda(fixture.binding, fixture.handles.stream())
              .success());

    std::vector<std::uint64_t> iterations;
    std::vector<gpuxtb_status_t> statuses;
    std::vector<std::uint64_t> failures;
    std::uint64_t plan_failure = 1u;
    CHECK(download(fixture.binding.state.scc.iterations, 4, iterations, fixture.handles.stream()));
    CHECK(
        download(fixture.binding.state.scc.system_statuses, 4, statuses, fixture.handles.stream()));
    CHECK(download(fixture.binding.workspace.ledger.system_failure_records, 4, failures,
                   fixture.handles.stream()));
    CHECK(download_value(fixture.binding.workspace.ledger.plan_failure_record, plan_failure,
                         fixture.handles.stream()));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

    CHECK(plan_failure == 0u);
    CHECK(iterations[0] == 0u);
    CHECK(statuses[0] == GPUXTB_STATUS_INTERNAL_ERROR);
    CHECK(failures[0] == failure_record(Gfn2SccStageId::kGeometry,
                                        static_cast<std::uint32_t>(
                                            Gfn2SccIterationControlCode::kStaleGeneration)));
    for (std::size_t system = 1u; system < iterations.size(); ++system) {
      CHECK(iterations[system] == 1u);
      CHECK(statuses[system] == GPUXTB_STATUS_SUCCESS);
      CHECK(failures[system] == 0u);
    }
  }

  {
    ProductionFixture fixture;
    CHECK(fixture.create(false));
    const std::uint8_t converged = 1u;
    const std::uint64_t stale_generation = kGeometryGeneration - 1u;
    CHECK(upload(&converged, fixture.binding.state.scc.converged, 1, fixture.handles.stream()));
    CHECK(upload(&stale_generation, fixture.binding.plan.geometry_cache.geometry_generations, 1,
                 fixture.handles.stream()));
    CHECK(launch_gfn2_restricted_scc_iteration_cuda(fixture.binding, fixture.handles.stream())
              .success());

    std::vector<std::uint64_t> iterations;
    std::vector<gpuxtb_status_t> statuses;
    std::vector<std::uint64_t> failures;
    CHECK(download(fixture.binding.state.scc.iterations, 4, iterations, fixture.handles.stream()));
    CHECK(
        download(fixture.binding.state.scc.system_statuses, 4, statuses, fixture.handles.stream()));
    CHECK(download(fixture.binding.workspace.ledger.system_failure_records, 4, failures,
                   fixture.handles.stream()));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

    CHECK(iterations[0] == 0u);
    CHECK(statuses[0] == GPUXTB_STATUS_SUCCESS);
    CHECK(failures[0] == 0u);
    for (std::size_t system = 1u; system < iterations.size(); ++system) {
      CHECK(iterations[system] == 1u);
      CHECK(statuses[system] == GPUXTB_STATUS_SUCCESS);
      CHECK(failures[system] == 0u);
    }
  }
  return 0;
}

int test_early_middle_and_late_peer_failures() {
  struct Injection {
    Gfn2SccStageId expected_stage;
    std::uint32_t expected_code;
    bool counts_attempt;
    enum class Kind { kGeometry, kHamiltonian, kOccupations } kind;
  };
  const Injection injections[]{
      {Gfn2SccStageId::kGeometry,
       static_cast<std::uint32_t>(Gfn2SccIterationControlCode::kStaleGeneration), false,
       Injection::Kind::kGeometry},
      {Gfn2SccStageId::kHamiltonian,
       static_cast<std::uint32_t>(Gfn2HamiltonianDeviceError::kNonfiniteH0), false,
       Injection::Kind::kHamiltonian},
      {Gfn2SccStageId::kOccupations,
       static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kInvalidElectronCount), true,
       Injection::Kind::kOccupations},
  };

  for (const Injection& injection : injections) {
    ProductionFixture fixture;
    CHECK(fixture.create(false));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::uint64_t stale_generation = kGeometryGeneration - 1u;
    if (injection.kind == Injection::Kind::kGeometry) {
      CHECK(upload(&stale_generation, fixture.binding.plan.geometry_cache.geometry_generations, 1,
                   fixture.handles.stream()));
    } else if (injection.kind == Injection::Kind::kHamiltonian) {
      CHECK(upload(&nan, const_cast<double*>(fixture.binding.input.hamiltonian.h0), 1,
                   fixture.handles.stream()));
    } else {
      CHECK(upload(&nan,
                   const_cast<double*>(fixture.binding.plan.occupations_batch.electron_counts), 1,
                   fixture.handles.stream()));
    }

    CHECK(launch_gfn2_restricted_scc_iteration_cuda(fixture.binding, fixture.handles.stream())
              .success());
    std::vector<std::uint64_t> iterations;
    std::vector<gpuxtb_status_t> statuses;
    std::vector<std::uint64_t> failures;
    CHECK(download(fixture.binding.state.scc.iterations, 4, iterations, fixture.handles.stream()));
    CHECK(
        download(fixture.binding.state.scc.system_statuses, 4, statuses, fixture.handles.stream()));
    CHECK(download(fixture.binding.workspace.ledger.system_failure_records, 4, failures,
                   fixture.handles.stream()));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

    CHECK(iterations[0] == (injection.counts_attempt ? 1u : 0u));
    CHECK(statuses[0] == (injection.expected_stage == Gfn2SccStageId::kOccupations
                              ? GPUXTB_STATUS_EIGENSOLVER_FAILED
                              : GPUXTB_STATUS_INTERNAL_ERROR));
    CHECK(failures[0] == failure_record(injection.expected_stage, injection.expected_code));
    for (std::size_t system = 1u; system < iterations.size(); ++system) {
      CHECK(iterations[system] == 1u);
      CHECK(statuses[system] == GPUXTB_STATUS_SUCCESS);
      CHECK(failures[system] == 0u);
    }
  }
  return 0;
}

int test_graph_replay_peer_failure() {
  ProductionFixture fixture;
  CHECK(fixture.create(false));
  CHECK(fixture.binding.plan.eigensolver_provider.capture_mode ==
        Gfn2SccIterationProviderCaptureMode::kGraphSupported);
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(fixture.handles.stream(), cudaStreamCaptureModeThreadLocal));
  CHECK(launch_gfn2_restricted_scc_iteration_cuda(fixture.binding, fixture.handles.stream())
            .success());
  CUDA_CHECK(cudaStreamEndCapture(fixture.handles.stream(), &graph));
  CHECK(graph != nullptr);
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));

  const double nan = std::numeric_limits<double>::quiet_NaN();
  CHECK(upload(&nan, const_cast<double*>(fixture.binding.input.hamiltonian.h0), 1,
               fixture.handles.stream()));
  CUDA_CHECK(cudaGraphLaunch(executable, fixture.handles.stream()));

  std::vector<std::uint64_t> iterations;
  std::vector<gpuxtb_status_t> statuses;
  std::vector<std::uint64_t> failures;
  CHECK(download(fixture.binding.state.scc.iterations, 4, iterations, fixture.handles.stream()));
  CHECK(download(fixture.binding.state.scc.system_statuses, 4, statuses, fixture.handles.stream()));
  CHECK(download(fixture.binding.workspace.ledger.system_failure_records, 4, failures,
                 fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  CHECK(iterations[0] == 0u);
  CHECK(statuses[0] == GPUXTB_STATUS_INTERNAL_ERROR);
  CHECK(failures[0] ==
        failure_record(Gfn2SccStageId::kHamiltonian,
                       static_cast<std::uint32_t>(Gfn2HamiltonianDeviceError::kNonfiniteH0)));
  for (std::size_t system = 1u; system < iterations.size(); ++system) {
    CHECK(iterations[system] == 1u);
    CHECK(statuses[system] == GPUXTB_STATUS_SUCCESS);
    CHECK(failures[system] == 0u);
  }
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  return 0;
}

int test_batch_plan_failure_is_transactional() {
  ProductionFixture fixture;
  CHECK(fixture.create(false));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  std::vector<Gfn2SccCacheProvenanceBinding> bindings(
      static_cast<std::size_t>(fixture.binding.plan.provenance.cache_binding_count));
  CHECK(download(fixture.binding.plan.provenance.cache_bindings,
                 fixture.binding.plan.provenance.cache_binding_count, bindings,
                 fixture.handles.stream()));
  std::vector<double> density_before;
  std::vector<std::uint64_t> iterations_before;
  CHECK(download(fixture.binding.state.density.density,
                 fixture.binding.state.density.density_elements, density_before,
                 fixture.handles.stream()));
  CHECK(download(fixture.binding.state.scc.iterations, 4, iterations_before,
                 fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  auto batch_binding = std::find_if(
      bindings.begin(), bindings.end(), [](const Gfn2SccCacheProvenanceBinding& binding) {
        return binding.provenance.generation_scope == Gfn2GenerationScope::kBatch;
      });
  CHECK(batch_binding != bindings.end());
  const Gfn2SccStageId owner_stage = batch_binding->owner_stage;
  batch_binding->provenance.geometry_generation = kGeometryGeneration - 1u;
  CHECK(upload(
      bindings.data(),
      const_cast<Gfn2SccCacheProvenanceBinding*>(fixture.binding.plan.provenance.cache_bindings),
      static_cast<std::int64_t>(bindings.size()), fixture.handles.stream()));

  CHECK(launch_gfn2_restricted_scc_iteration_cuda(fixture.binding, fixture.handles.stream())
            .success());
  std::vector<double> density_after;
  std::vector<std::uint64_t> iterations_after;
  std::vector<std::uint64_t> failures;
  std::uint64_t plan_failure = 0u;
  std::uint32_t sequence_active = 1u;
  CHECK(download(fixture.binding.state.density.density,
                 fixture.binding.state.density.density_elements, density_after,
                 fixture.handles.stream()));
  CHECK(download(fixture.binding.state.scc.iterations, 4, iterations_after,
                 fixture.handles.stream()));
  CHECK(download(fixture.binding.workspace.ledger.system_failure_records, 4, failures,
                 fixture.handles.stream()));
  CHECK(download_value(fixture.binding.workspace.ledger.plan_failure_record, plan_failure,
                       fixture.handles.stream()));
  CHECK(download_value(fixture.binding.workspace.ledger.sequence_active, sequence_active,
                       fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  CHECK(density_after == density_before);
  CHECK(iterations_after == iterations_before);
  CHECK(std::all_of(failures.begin(), failures.end(),
                    [](std::uint64_t value) { return value == 0u; }));
  CHECK(sequence_active == 0u);
  CHECK(plan_failure ==
        failure_record(owner_stage,
                       static_cast<std::uint32_t>(Gfn2SccIterationControlCode::kStaleGeneration)));
  return 0;
}

int test_cross_plan_and_capture_fail_closed() {
  {
    ProductionFixture fixture;
    CHECK(fixture.create(false));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
    std::vector<std::uint64_t> before;
    CHECK(download(fixture.binding.state.scc.iterations, 4, before, fixture.handles.stream()));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

    Gfn2SccIterationBinding cross_plan = fixture.binding;
    ++cross_plan.input.plan_token;
    const Gfn2SccIterationLaunchResult result =
        launch_gfn2_restricted_scc_iteration_cuda(cross_plan, fixture.handles.stream());
    CHECK(result.status == Gfn2SccIterationLaunchStatus::kInvalidBinding);

    std::vector<std::uint64_t> after;
    CHECK(download(fixture.binding.state.scc.iterations, 4, after, fixture.handles.stream()));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
    CHECK(after == before);
  }

  {
    ProductionFixture fixture;
    CHECK(fixture.create(false));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
    Gfn2SccIterationDevicePlan capture_disabled_plan = fixture.binding.plan;
    capture_disabled_plan.eigensolver_provider.capture_mode =
        Gfn2SccIterationProviderCaptureMode::kUncapturedSegmentRequired;
    Gfn2SccIterationBinding capture_disabled{};
    CHECK(bind_gfn2_scc_iteration_cuda(capture_disabled_plan, fixture.binding.input,
                                       fixture.binding.state, fixture.binding.workspace,
                                       capture_disabled)
              .error == Gfn2SccIterationBindingError::kSuccess);

    Gfn2SccIterationDevicePlan malformed_plan = fixture.binding.plan;
    malformed_plan.eigensolver_provider.capture_mode =
        Gfn2SccIterationProviderCaptureMode::kUnspecified;
    Gfn2SccIterationBinding malformed{};
    const auto malformed_diagnostic =
        bind_gfn2_scc_iteration_cuda(malformed_plan, fixture.binding.input, fixture.binding.state,
                                     fixture.binding.workspace, malformed);
    CHECK(malformed_diagnostic.error == Gfn2SccIterationBindingError::kInvalidProvider);
    CHECK(malformed_diagnostic.field == Gfn2SccIterationBindingField::kEigensolver);

    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(fixture.handles.stream(), cudaStreamCaptureModeGlobal));
    const Gfn2SccLoopLaunchResult result =
        launch_gfn2_restricted_scc_loop_cuda(capture_disabled, fixture.handles.stream());
    CHECK(result.iteration.status == Gfn2SccIterationLaunchStatus::kProviderCaptureUnsupported);
    CHECK(result.iteration.stage == Gfn2SccStageId::kEigensolver);
    CHECK(result.submitted_iterations == 0u);
    CUDA_CHECK(cudaStreamEndCapture(fixture.handles.stream(), &graph));
    if (graph != nullptr) {
      CUDA_CHECK(cudaGraphDestroy(graph));
    }

    std::vector<std::uint64_t> iterations;
    CHECK(download(fixture.binding.state.scc.iterations, 4, iterations, fixture.handles.stream()));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
    CHECK(std::all_of(iterations.begin(), iterations.end(),
                      [](std::uint64_t value) { return value == 0u; }));
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
  CUDA_CHECK(count_status);
  CUDA_CHECK(cudaSetDevice(0));

  int status = test_terminal_peer_cpu_parity();
  if (status != 0) {
    return status;
  }
  status = test_mixed_warm_loop_state();
  if (status != 0) {
    return status;
  }
  status = test_stale_active_and_inactive_provenance();
  if (status != 0) {
    return status;
  }
  status = test_early_middle_and_late_peer_failures();
  if (status != 0) {
    return status;
  }
  status = test_graph_replay_peer_failure();
  if (status != 0) {
    return status;
  }
  status = test_batch_plan_failure_is_transactional();
  if (status != 0) {
    return status;
  }
  return test_cross_plan_and_capture_fail_closed();
}
