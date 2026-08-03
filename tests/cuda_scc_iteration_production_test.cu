#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <algorithm>
#include <array>
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

/* Establish setup-stream ordering for a distinct nonblocking execution
 * stream without synchronizing the host. The provider handles and mutable SCC
 * arena remain single-flight for the complete lifetime of this object. */
class OrderedExecutionStream {
 public:
  OrderedExecutionStream() = default;
  OrderedExecutionStream(const OrderedExecutionStream&) = delete;
  OrderedExecutionStream& operator=(const OrderedExecutionStream&) = delete;

  ~OrderedExecutionStream() {
    if (stream_ != nullptr) {
      (void)cudaStreamSynchronize(stream_);
    }
    if (event_ != nullptr) {
      (void)cudaEventDestroy(event_);
    }
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
    }
  }

  bool create_after(cudaStream_t setup_stream) noexcept {
    return cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) == cudaSuccess &&
           cudaEventCreateWithFlags(&event_, cudaEventDisableTiming) == cudaSuccess &&
           cudaEventRecord(event_, setup_stream) == cudaSuccess &&
           cudaStreamWaitEvent(stream_, event_, 0u) == cudaSuccess;
  }

  cudaStream_t stream() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
  cudaEvent_t event_ = nullptr;
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

bool compare_coefficients(const HostSccCase& host, std::vector<double> actual, double tolerance) {
  const auto& layout = host.wavefunction_layout();
  const double* expected = host.wavefunction().coefficients;
  for (std::int64_t system = 0; system < layout.batch_size; ++system) {
    const std::int64_t orbital_begin = layout.batch_orbital_offsets[system];
    const std::int64_t orbital_end = layout.batch_orbital_offsets[system + 1];
    const std::int64_t n = orbital_end - orbital_begin;
    const std::int64_t matrix_begin = layout.coefficients.system_offsets[system];
    for (std::int64_t orbital = 0; orbital < n; ++orbital) {
      double dot = 0.0;
      for (std::int64_t row = 0; row < n; ++row) {
        const std::int64_t index = matrix_begin + row * n + orbital;
        dot += actual[static_cast<std::size_t>(index)] * expected[index];
      }
      if (dot < 0.0) {
        for (std::int64_t row = 0; row < n; ++row) {
          const std::int64_t index = matrix_begin + row * n + orbital;
          actual[static_cast<std::size_t>(index)] = -actual[static_cast<std::size_t>(index)];
        }
      }
    }
  }
  return compare_doubles("coefficients", actual, expected, layout.coefficients.element_count,
                         tolerance);
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

  bool create(bool optional_components, std::int64_t batch_size = 4) {
    if (batch_size <= 0) {
      return false;
    }
    HostSccCaseOptions options{};
    constexpr std::array<SmallSystemKind, 4> kSystems{SmallSystemKind::kH2, SmallSystemKind::kHe,
                                                      SmallSystemKind::kLiH, SmallSystemKind::kCH2};
    options.systems.clear();
    options.systems.reserve(static_cast<std::size_t>(batch_size));
    for (std::int64_t system = 0; system < batch_size; ++system) {
      options.systems.push_back(kSystems[static_cast<std::size_t>(system) % kSystems.size()]);
    }
    options.geometry_generation = kGeometryGeneration;
    options.maximum_iterations = 8u;
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

int test_four_system_production_iteration_cpu_parity(bool optional_components) {
  ProductionFixture fixture;
  CHECK(fixture.create(optional_components));

  const Gfn2SccIterationLaunchResult launch =
      launch_gfn2_restricted_scc_iteration_cuda(fixture.binding, fixture.handles.stream());
  if (!launch.success()) {
    std::fprintf(stderr,
                 "production SCC launch failed: status=%u stage=%u binding_error=%u "
                 "binding_field=%u cuda=%d cublas=%d cusolver=%d\n",
                 static_cast<unsigned>(launch.status), static_cast<unsigned>(launch.stage),
                 static_cast<unsigned>(launch.binding.error),
                 static_cast<unsigned>(launch.binding.field), static_cast<int>(launch.cuda_status),
                 static_cast<int>(launch.cublas_status), static_cast<int>(launch.cusolver_status));
  }
  CHECK(launch.success());

  std::string error;
  const gpuxtb_status_t cpu_status = fixture.host.run_one_iteration(error);
  if (cpu_status != GPUXTB_STATUS_SUCCESS) {
    std::fprintf(stderr, "CPU production SCC iteration failed: status=%d error=%s\n", cpu_status,
                 error.c_str());
  }
  CHECK(cpu_status == GPUXTB_STATUS_SUCCESS);

  const auto& layout = fixture.host.wavefunction_layout();
  const auto& state = fixture.binding.state;
  const auto& workspace = fixture.binding.workspace;
  std::vector<double> eigenvalues;
  std::vector<double> coefficients;
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
  std::vector<double> free_energy;
  std::vector<double> current_inputs;
  std::vector<std::uint64_t> iterations;
  std::vector<gpuxtb_status_t> statuses;
  std::vector<std::uint8_t> converged;

  CHECK(download(state.eigenpairs.eigenvalues, state.eigenpairs.eigenvalue_elements, eigenvalues,
                 fixture.handles.stream()));
  CHECK(download(state.eigenpairs.coefficients, state.eigenpairs.coefficient_elements, coefficients,
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

  CHECK(compare_doubles("eigenvalues", eigenvalues, fixture.host.wavefunction().eigenvalues,
                        layout.eigenvalues.element_count, 3.0e-9));
  CHECK(compare_coefficients(fixture.host, coefficients, 3.0e-8));
  CHECK(compare_doubles("occupations", occupations, fixture.host.wavefunction().occupations,
                        layout.occupations.element_count, 3.0e-10));
  CHECK(compare_doubles("density", density, fixture.host.wavefunction().density,
                        layout.density.element_count, 3.0e-9));
  CHECK(compare_doubles("energy-weighted density", weighted_density,
                        fixture.host.wavefunction().energy_weighted_density,
                        layout.energy_weighted_density.element_count, 3.0e-9));
  CHECK(compare_doubles("public qsh", public_qsh, fixture.host.wavefunction().qsh,
                        layout.qsh.element_count, 3.0e-9));
  CHECK(compare_doubles("public qat", public_qat, fixture.host.wavefunction().qat,
                        layout.qat.element_count, 3.0e-9));
  CHECK(compare_doubles("public dipoles", public_dipoles, fixture.host.wavefunction().dipole,
                        layout.dipole.element_count, 3.0e-9));
  CHECK(compare_doubles("public quadrupoles", public_quadrupoles,
                        fixture.host.wavefunction().quadrupole, layout.quadrupole.element_count,
                        3.0e-9));

  const auto& cpu_workspace = fixture.host.driver_workspace();
  CHECK(
      compare_doubles("raw qsh", raw_qsh, cpu_workspace.raw_qsh, layout.qsh.element_count, 3.0e-9));
  CHECK(
      compare_doubles("raw qat", raw_qat, cpu_workspace.raw_qat, layout.qat.element_count, 3.0e-9));
  CHECK(compare_doubles("raw dipoles", raw_dipoles, cpu_workspace.raw_dipoles,
                        layout.dipole.element_count, 3.0e-9));
  CHECK(compare_doubles("raw quadrupoles", raw_quadrupoles, cpu_workspace.raw_quadrupoles,
                        layout.quadrupole.element_count, 3.0e-9));
  CHECK(compare_doubles("free energy", free_energy, fixture.host.driver_state().free_energies,
                        fixture.host.batch_size(), 3.0e-9));
  CHECK(compare_doubles("mixer current inputs", current_inputs,
                        fixture.host.mixer_state().current_inputs,
                        fixture.host.mixer_plan().total_vector_elements(), 3.0e-9));

  for (std::int64_t system = 0; system < fixture.host.batch_size(); ++system) {
    CHECK(iterations[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().iterations[system]);
    CHECK(statuses[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().system_statuses[system]);
    CHECK(converged[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().converged[system]);
  }

  /* Reuse the exact same production binding for a second iteration. This is
   * the steady-state contract consumed by Graph replay and proves that no
   * setup owner or descriptor must be rebuilt between SCC launches. */
  const Gfn2SccIterationLaunchResult repeat_launch =
      launch_gfn2_restricted_scc_iteration_cuda(fixture.binding, fixture.handles.stream());
  CHECK(repeat_launch.success());
  CHECK(fixture.host.run_one_iteration(error) == GPUXTB_STATUS_SUCCESS);

  CHECK(download(state.density.density, state.density.density_elements, density,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.qsh, state.raw_population.qsh_elements, public_qsh,
                 fixture.handles.stream()));
  CHECK(download(state.free_energy.free_energy, state.free_energy.free_energy_elements, free_energy,
                 fixture.handles.stream()));
  CHECK(download(state.mixer.current_inputs, state.mixer.total_vector_elements, current_inputs,
                 fixture.handles.stream()));
  CHECK(download(state.scc.iterations, state.scc.batch_elements, iterations,
                 fixture.handles.stream()));
  CHECK(download(state.scc.system_statuses, state.scc.batch_elements, statuses,
                 fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  CHECK(compare_doubles("repeat density", density, fixture.host.wavefunction().density,
                        layout.density.element_count, 3.0e-9));
  CHECK(compare_doubles("repeat public qsh", public_qsh, fixture.host.wavefunction().qsh,
                        layout.qsh.element_count, 3.0e-9));
  CHECK(compare_doubles("repeat free energy", free_energy,
                        fixture.host.driver_state().free_energies, fixture.host.batch_size(),
                        3.0e-9));
  CHECK(compare_doubles("repeat mixer current inputs", current_inputs,
                        fixture.host.mixer_state().current_inputs,
                        fixture.host.mixer_plan().total_vector_elements(), 3.0e-9));
  for (std::int64_t system = 0; system < fixture.host.batch_size(); ++system) {
    CHECK(iterations[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().iterations[system]);
    CHECK(statuses[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().system_statuses[system]);
  }
  return 0;
}

int test_production_loop_cpu_parity(std::int64_t batch_size, bool optional_components,
                                    bool use_default_stream, bool use_ordered_stream = false,
                                    std::uint64_t resumed_iterations = 0u) {
  ProductionFixture fixture;
  CHECK(fixture.create(optional_components, batch_size));
  CHECK(fixture.host.batch_size() == batch_size);
  CHECK(!(use_default_stream && use_ordered_stream));

  OrderedExecutionStream ordered_stream;
  cudaStream_t execution_stream = fixture.handles.stream();
  if (use_ordered_stream) {
    CHECK(ordered_stream.create_after(fixture.handles.stream()));
    execution_stream = ordered_stream.stream();
  } else if (use_default_stream) {
    /* Setup uploads are ordered on the owner's nonblocking stream. A public
     * caller using another stream must establish the same one-time dependency
     * before entering the allocation-free hot loop. */
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
    execution_stream = nullptr;
  }

  std::string error;
  for (std::uint64_t iteration = 0u; iteration < resumed_iterations; ++iteration) {
    const Gfn2SccIterationLaunchResult resumed_launch =
        launch_gfn2_restricted_scc_iteration_cuda(fixture.binding, execution_stream);
    CHECK(resumed_launch.success());
    const gpuxtb_status_t cpu_status = fixture.host.run_one_iteration(error);
    CHECK(cpu_status == GPUXTB_STATUS_SUCCESS || cpu_status == GPUXTB_STATUS_SCC_NOT_CONVERGED ||
          cpu_status == GPUXTB_STATUS_EIGENSOLVER_FAILED);
  }

  const Gfn2SccLoopLaunchResult launch =
      launch_gfn2_restricted_scc_loop_cuda(fixture.binding, execution_stream);
  if (!launch.success()) {
    std::fprintf(stderr,
                 "production SCC loop launch failed: submitted=%llu status=%u stage=%u "
                 "binding_error=%u binding_field=%u cuda=%d cublas=%d cusolver=%d\n",
                 static_cast<unsigned long long>(launch.submitted_iterations),
                 static_cast<unsigned>(launch.iteration.status),
                 static_cast<unsigned>(launch.iteration.stage),
                 static_cast<unsigned>(launch.iteration.binding.error),
                 static_cast<unsigned>(launch.iteration.binding.field),
                 static_cast<int>(launch.iteration.cuda_status),
                 static_cast<int>(launch.iteration.cublas_status),
                 static_cast<int>(launch.iteration.cusolver_status));
  }
  CHECK(launch.success());
  CHECK(launch.submitted_iterations == fixture.host.options().maximum_iterations);

  /* Observe the first completed loop before submitting a second one. This
   * distinguishes a genuinely terminal public-state replay from an
   * implementation where the second loop merely finishes work omitted by the
   * first submission. */
  CUDA_CHECK(cudaStreamSynchronize(execution_stream));
  std::vector<double> first_coefficients;
  std::vector<double> first_density;
  std::vector<double> first_qsh;
  std::vector<double> first_free_energies;
  std::vector<double> first_previous_free_energies;
  std::vector<double> first_free_energy_changes;
  std::vector<double> first_mixer_inputs;
  std::vector<std::uint64_t> first_iterations;
  std::vector<gpuxtb_status_t> first_statuses;
  std::vector<std::uint8_t> first_converged;
  const auto& first_state = fixture.binding.state;
  CHECK(download(first_state.eigenpairs.coefficients, first_state.eigenpairs.coefficient_elements,
                 first_coefficients, fixture.handles.stream()));
  CHECK(download(first_state.density.density, first_state.density.density_elements, first_density,
                 fixture.handles.stream()));
  CHECK(download(first_state.raw_population.qsh, first_state.raw_population.qsh_elements, first_qsh,
                 fixture.handles.stream()));
  CHECK(download(first_state.scc.free_energies, first_state.scc.batch_elements, first_free_energies,
                 fixture.handles.stream()));
  CHECK(download(first_state.scc.previous_free_energies, first_state.scc.batch_elements,
                 first_previous_free_energies, fixture.handles.stream()));
  CHECK(download(first_state.scc.free_energy_changes, first_state.scc.batch_elements,
                 first_free_energy_changes, fixture.handles.stream()));
  CHECK(download(first_state.mixer.current_inputs, first_state.mixer.total_vector_elements,
                 first_mixer_inputs, fixture.handles.stream()));
  CHECK(download(first_state.scc.iterations, first_state.scc.batch_elements, first_iterations,
                 fixture.handles.stream()));
  CHECK(download(first_state.scc.system_statuses, first_state.scc.batch_elements, first_statuses,
                 fixture.handles.stream()));
  CHECK(download(first_state.scc.converged, first_state.scc.batch_elements, first_converged,
                 fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  const Gfn2SccLoopLaunchResult repeat_launch =
      launch_gfn2_restricted_scc_loop_cuda(fixture.binding, execution_stream);
  CHECK(repeat_launch.success());
  CHECK(repeat_launch.submitted_iterations == fixture.host.options().maximum_iterations);
  if (execution_stream != fixture.handles.stream()) {
    /* Downloads below use the fixture stream, so complete the independent
     * execution-stream replay first. This synchronization is test observation,
     * not part of the production loop launcher. */
    CUDA_CHECK(cudaStreamSynchronize(execution_stream));
  }

  /* The CPU reference is advanced through the same fixed iteration bound.
   * Once all peers are terminal its active predicate makes later calls public-
   * state no-ops, matching the CUDA publication gate. The CUDA provider keeps
   * a fixed bucket schedule until #80 compacts inactive members. */
  for (std::uint64_t iteration = 0u; iteration < fixture.host.options().maximum_iterations;
       ++iteration) {
    const gpuxtb_status_t status = fixture.host.run_one_iteration(error);
    if (status != GPUXTB_STATUS_SUCCESS && status != GPUXTB_STATUS_SCC_NOT_CONVERGED &&
        status != GPUXTB_STATUS_EIGENSOLVER_FAILED) {
      std::fprintf(stderr, "CPU production SCC loop failed at %llu: status=%d error=%s\n",
                   static_cast<unsigned long long>(iteration), status, error.c_str());
      return __LINE__;
    }
  }
  const auto& layout = fixture.host.wavefunction_layout();
  const auto& state = fixture.binding.state;
  std::vector<double> eigenvalues;
  std::vector<double> coefficients;
  std::vector<double> occupations;
  std::vector<double> density;
  std::vector<double> weighted_density;
  std::vector<double> qsh;
  std::vector<double> qat;
  std::vector<double> dipoles;
  std::vector<double> quadrupoles;
  std::vector<double> free_energies;
  std::vector<double> previous_free_energies;
  std::vector<double> free_energy_changes;
  std::vector<double> residual_rms;
  std::vector<double> internal_energies;
  std::vector<double> entropies;
  std::vector<double> current_inputs;
  std::vector<std::uint64_t> iterations;
  std::vector<gpuxtb_status_t> statuses;
  std::vector<std::uint8_t> converged;

  CHECK(download(state.eigenpairs.eigenvalues, state.eigenpairs.eigenvalue_elements, eigenvalues,
                 fixture.handles.stream()));
  CHECK(download(state.eigenpairs.coefficients, state.eigenpairs.coefficient_elements, coefficients,
                 fixture.handles.stream()));
  CHECK(download(state.occupations.occupations, state.occupations.occupation_elements, occupations,
                 fixture.handles.stream()));
  CHECK(download(state.density.density, state.density.density_elements, density,
                 fixture.handles.stream()));
  CHECK(download(state.density.energy_weighted_density, state.density.weighted_density_elements,
                 weighted_density, fixture.handles.stream()));
  CHECK(download(state.raw_population.qsh, state.raw_population.qsh_elements, qsh,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.qat, state.raw_population.qat_elements, qat,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.dipole, state.raw_population.dipole_elements, dipoles,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.quadrupole, state.raw_population.quadrupole_elements,
                 quadrupoles, fixture.handles.stream()));
  CHECK(download(state.scc.free_energies, state.scc.batch_elements, free_energies,
                 fixture.handles.stream()));
  CHECK(download(state.scc.previous_free_energies, state.scc.batch_elements, previous_free_energies,
                 fixture.handles.stream()));
  CHECK(download(state.scc.free_energy_changes, state.scc.batch_elements, free_energy_changes,
                 fixture.handles.stream()));
  CHECK(download(state.scc.residual_rms, state.scc.batch_elements, residual_rms,
                 fixture.handles.stream()));
  CHECK(download(state.free_energy.internal_energy, state.free_energy.internal_energy_elements,
                 internal_energies, fixture.handles.stream()));
  CHECK(download(state.free_energy.entropy, state.free_energy.entropy_elements, entropies,
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

  CHECK(coefficients == first_coefficients);
  CHECK(density == first_density);
  CHECK(qsh == first_qsh);
  CHECK(free_energies == first_free_energies);
  CHECK(previous_free_energies == first_previous_free_energies);
  CHECK(free_energy_changes == first_free_energy_changes);
  CHECK(current_inputs == first_mixer_inputs);
  CHECK(iterations == first_iterations);
  CHECK(statuses == first_statuses);
  CHECK(converged == first_converged);

  constexpr double kLoopTolerance = 1.0e-8;
  for (std::int64_t system = 0; system < fixture.host.batch_size(); ++system) {
    if (iterations[static_cast<std::size_t>(system)] !=
            fixture.host.driver_state().iterations[system] ||
        statuses[static_cast<std::size_t>(system)] !=
            fixture.host.driver_state().system_statuses[system] ||
        converged[static_cast<std::size_t>(system)] !=
            fixture.host.driver_state().converged[system]) {
      std::fprintf(stderr,
                   "loop terminal mismatch system=%lld CUDA(iter=%llu status=%d conv=%u "
                   "free=%.17g residual=%.17g) CPU(iter=%llu status=%d conv=%u free=%.17g "
                   "residual=%.17g)\n",
                   static_cast<long long>(system),
                   static_cast<unsigned long long>(iterations[static_cast<std::size_t>(system)]),
                   statuses[static_cast<std::size_t>(system)],
                   static_cast<unsigned>(converged[static_cast<std::size_t>(system)]),
                   free_energies[static_cast<std::size_t>(system)],
                   residual_rms[static_cast<std::size_t>(system)],
                   static_cast<unsigned long long>(fixture.host.driver_state().iterations[system]),
                   fixture.host.driver_state().system_statuses[system],
                   static_cast<unsigned>(fixture.host.driver_state().converged[system]),
                   fixture.host.driver_state().free_energies[system],
                   fixture.host.mixer_state().residual_rms[system]);
    }
  }
  CHECK(compare_doubles("loop eigenvalues", eigenvalues, fixture.host.wavefunction().eigenvalues,
                        layout.eigenvalues.element_count, kLoopTolerance));
  /* The existing one-step gate compares individual coefficient columns after
   * sign alignment. Across a full trajectory, exactly or numerically
   * degenerate orbitals may undergo a valid provider-dependent subspace
   * rotation. Density and energy-weighted density below are the invariant
   * full-loop correctness observables. */
  CHECK(compare_doubles("loop occupations", occupations, fixture.host.wavefunction().occupations,
                        layout.occupations.element_count, kLoopTolerance));
  CHECK(compare_doubles("loop density", density, fixture.host.wavefunction().density,
                        layout.density.element_count, kLoopTolerance));
  CHECK(compare_doubles("loop weighted density", weighted_density,
                        fixture.host.wavefunction().energy_weighted_density,
                        layout.energy_weighted_density.element_count, kLoopTolerance));
  CHECK(compare_doubles("loop qsh", qsh, fixture.host.wavefunction().qsh, layout.qsh.element_count,
                        kLoopTolerance));
  CHECK(compare_doubles("loop qat", qat, fixture.host.wavefunction().qat, layout.qat.element_count,
                        kLoopTolerance));
  CHECK(compare_doubles("loop dipoles", dipoles, fixture.host.wavefunction().dipole,
                        layout.dipole.element_count, kLoopTolerance));
  CHECK(compare_doubles("loop quadrupoles", quadrupoles, fixture.host.wavefunction().quadrupole,
                        layout.quadrupole.element_count, kLoopTolerance));
  CHECK(compare_doubles("loop free energies", free_energies,
                        fixture.host.driver_state().free_energies, fixture.host.batch_size(),
                        kLoopTolerance));
  CHECK(compare_doubles("loop previous free energies", previous_free_energies,
                        fixture.host.driver_state().previous_free_energies,
                        fixture.host.batch_size(), kLoopTolerance));
  CHECK(compare_doubles("loop free-energy changes", free_energy_changes,
                        fixture.host.driver_state().free_energy_changes, fixture.host.batch_size(),
                        kLoopTolerance));
  CHECK(compare_doubles("loop residual RMS", residual_rms, fixture.host.mixer_state().residual_rms,
                        fixture.host.batch_size(), kLoopTolerance));
  CHECK(compare_doubles("loop internal energies", internal_energies,
                        fixture.host.driver_state().internal_energies, fixture.host.batch_size(),
                        kLoopTolerance));
  CHECK(compare_doubles("loop entropies", entropies, fixture.host.driver_state().entropies,
                        fixture.host.batch_size(), kLoopTolerance));
  CHECK(compare_doubles("loop mixer current inputs", current_inputs,
                        fixture.host.mixer_state().current_inputs,
                        fixture.host.mixer_plan().total_vector_elements(), kLoopTolerance));
  for (std::int64_t system = 0; system < fixture.host.batch_size(); ++system) {
    CHECK(iterations[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().iterations[system]);
    CHECK(statuses[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().system_statuses[system]);
    CHECK(converged[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().converged[system]);
  }
  if (batch_size == 128 && !optional_components && !use_default_stream) {
    const auto converged_count =
        std::count(converged.begin(), converged.end(), static_cast<std::uint8_t>(1u));
    const auto nonconverged_count =
        std::count(statuses.begin(), statuses.end(), GPUXTB_STATUS_SCC_NOT_CONVERGED);
    CHECK(converged_count > 0);
    CHECK(nonconverged_count > 0);
    CHECK(converged_count + nonconverged_count == batch_size);
  }
  return 0;
}

int test_loop_rejects_inconsistent_plan() {
  Gfn2SccIterationBinding binding{};
  binding.plan.plan_token = 7u;
  binding.plan.activity_policy.maximum_iterations = 1u;
  binding.plan.state_policy.maximum_iterations = 1u;
  binding.plan.publication_plan.maximum_iterations = 1u;

  binding.plan.abi_version = 0u;
  Gfn2SccLoopLaunchResult result = launch_gfn2_restricted_scc_loop_cuda(binding);
  CHECK(result.iteration.status == Gfn2SccIterationLaunchStatus::kInvalidBinding);
  CHECK(result.iteration.binding.error == Gfn2SccIterationBindingError::kInvalidAbiVersion);
  CHECK(result.submitted_iterations == 0u);

  binding.plan.abi_version = kGfn2SccIterationAbiVersion;
  binding.plan.plan_token = 0u;
  result = launch_gfn2_restricted_scc_loop_cuda(binding);
  CHECK(result.iteration.binding.error == Gfn2SccIterationBindingError::kInvalidPlanToken);
  CHECK(result.submitted_iterations == 0u);

  binding.plan.plan_token = 7u;
  binding.plan.state_policy.maximum_iterations = 2u;
  result = launch_gfn2_restricted_scc_loop_cuda(binding);
  CHECK(result.iteration.binding.error == Gfn2SccIterationBindingError::kInvalidCount);
  CHECK(result.submitted_iterations == 0u);
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
  int status = test_loop_rejects_inconsistent_plan();
  if (status != 0) {
    return status;
  }
  status = test_four_system_production_iteration_cpu_parity(false);
  if (status != 0) {
    return status;
  }
  status = test_four_system_production_iteration_cpu_parity(true);
  if (status != 0) {
    return status;
  }
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    status = test_production_loop_cpu_parity(batch_size, false, false);
    if (status != 0) {
      return status;
    }
  }
  for (bool use_default_stream : {false, true}) {
    status = test_production_loop_cpu_parity(8, true, use_default_stream);
    if (status != 0) {
      return status;
    }
  }
  status = test_production_loop_cpu_parity(8, false, true);
  if (status != 0) {
    return status;
  }
  status = test_production_loop_cpu_parity(8, true, false, false, 2u);
  if (status != 0) {
    return status;
  }
  status = test_production_loop_cpu_parity(8, true, false, true);
  if (status != 0) {
    return status;
  }
  return 0;
}
