#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "backends/cuda/gfn2_scc_iteration_initialize.cuh"

namespace {

using namespace gpuxtb::detail;
using namespace gpuxtb::detail::cuda;

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression)                        \
  do {                                                \
    if ((expression) != cudaSuccess) return __LINE__; \
  } while (false)

template <typename T>
Gfn2SccIterationHostArrayView<T> host_view(const std::vector<T>& values) {
  return {values.data(), static_cast<std::int64_t>(values.size())};
}

struct DeviceAllocation {
  void* pointer = nullptr;

  explicit DeviceAllocation(std::size_t bytes) {
    if (cudaMalloc(&pointer, bytes) != cudaSuccess) pointer = nullptr;
  }
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  ~DeviceAllocation() {
    if (pointer != nullptr) (void)cudaFree(pointer);
  }

  void* release() noexcept {
    void* result = pointer;
    pointer = nullptr;
    return result;
  }
};

struct Stream {
  cudaStream_t value = nullptr;
  Stream() { (void)cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking); }
  ~Stream() {
    if (value != nullptr) (void)cudaStreamDestroy(value);
  }
};

struct Graph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;

  Graph() = default;
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  ~Graph() {
    if (executable != nullptr) (void)cudaGraphExecDestroy(executable);
    if (graph != nullptr) (void)cudaGraphDestroy(graph);
  }
};

/* Keep a custom stream busy long enough to exercise initializer destruction
 * while a checkpoint restore is definitely still queued. */
__global__ void delay_device_cycles(unsigned long long cycles) {
  const unsigned long long begin = clock64();
  while (clock64() - begin < cycles) {
  }
}

constexpr std::uint64_t kToken = 0x106c0deULL;
constexpr std::int64_t kBatch = 2;
constexpr std::int64_t kAtoms = 3;
constexpr std::int64_t kShells = 4;
constexpr std::int64_t kOrbitals = 5;
constexpr std::int64_t kMatrices = 13;
constexpr std::int64_t kDipoles = 9;
constexpr std::int64_t kQuadrupoles = 18;
constexpr std::int64_t kMixerVector = 31;
constexpr std::int64_t kHistory = 2;

Gfn2SccIterationDevicePlan make_plan(std::uint32_t components) {
  Gfn2SccIterationDevicePlan plan{};
  plan.abi_version = kGfn2SccIterationAbiVersion;
  plan.enabled_components = components;
  plan.plan_token = kToken;
  plan.topology.plan_token = kToken;
  plan.topology.batch_size = kBatch;
  plan.topology.total_atoms = kAtoms;
  plan.topology.total_shells = kShells;
  plan.topology.total_orbitals = kOrbitals;
  plan.topology.total_matrix_elements = kMatrices;
  plan.mixer_policy.history_size = kHistory;
  plan.state_policy.maximum_iterations = 8u;
  plan.geometry_batch.total_pairs = 2;
  plan.es2_batch.total_matrix_elements = 10;
  plan.aes2_batch.total_pairs = 2;
  plan.d4_batch.total_pairs = 2;
  return plan;
}

struct BoundArena {
  Gfn2SccIterationDevicePlan plan{};
  Gfn2SccIterationArenaRequirements requirements{};
  DeviceAllocation allocation{1u};
  Gfn2SccIterationDeviceState state{};
  Gfn2SccIterationDeviceWorkspace workspace{};
  Gfn2SccIterationReportStorage reports{};
  bool valid = false;

  explicit BoundArena(std::uint32_t components)
      : plan(make_plan(components)), allocation(query_bytes(plan, requirements)) {
    if (allocation.pointer == nullptr || requirements.total_bytes == 0u) return;
    const auto diagnostic = bind_gfn2_scc_iteration_arena_cuda(
        plan, plan.eigensolver_provider.requirements, requirements, allocation.pointer,
        requirements.total_bytes, nullptr, 0u, state, workspace, reports);
    valid = diagnostic.success();
  }

  static std::size_t query_bytes(const Gfn2SccIterationDevicePlan& plan,
                                 Gfn2SccIterationArenaRequirements& requirements) {
    const auto diagnostic = query_gfn2_scc_iteration_arena_requirements_cuda(
        plan, plan.eigensolver_provider.requirements, requirements);
    return diagnostic.success() ? requirements.total_bytes : 1u;
  }
};

struct FreshData {
  std::vector<std::int64_t> atom_offsets{0, 2, 3};
  std::vector<std::int64_t> shell_offsets{0, 3, 4};
  std::vector<double> qsh{0.1, -0.2, 0.3, -0.4};
  std::vector<double> qat{0.5, -0.3, -0.4};
  std::vector<double> dipoles{0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09};
  std::vector<double> quadrupoles;

  FreshData() {
    quadrupoles.resize(kQuadrupoles);
    for (std::size_t index = 0; index < quadrupoles.size(); ++index) {
      quadrupoles[index] = 0.001 * static_cast<double>(index + 1u);
    }
  }

  Gfn2SccIterationHostInitialization view(std::uint64_t generation = 3u) const {
    Gfn2SccIterationHostInitialization host{};
    host.mode = Gfn2SccIterationInitializationMode::kFresh;
    host.plan_token = kToken;
    host.initialization_generation = generation;
    host.topology = {host_view(atom_offsets), host_view(shell_offsets), kToken};
    host.wavefunction.plan_token = kToken;
    host.wavefunction.population = {host_view(qsh), host_view(qat), host_view(dipoles),
                                    host_view(quadrupoles), kToken};
    return host;
  }
};

template <typename T>
std::vector<T> copy_from_device(const T* source, std::int64_t elements, cudaStream_t stream) {
  std::vector<T> result(static_cast<std::size_t>(elements));
  if (cudaMemcpyAsync(result.data(), source, result.size() * sizeof(T), cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    result.clear();
  }
  return result;
}

int test_fresh_initialization_and_stream_ready_publication() {
  const std::uint32_t mandatory = static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES2) |
                                  static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES3) |
                                  static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kAES2);
  BoundArena arena(mandatory);
  CHECK(arena.valid);
  FreshData data;
  auto host = data.view();

  CUDA_CHECK(cudaMemset(arena.allocation.pointer, 0xa5, arena.requirements.total_bytes));
  Gfn2SccIterationInitializer initializer;
  auto diagnostic = Gfn2SccIterationInitializer::create(
      arena.plan, arena.requirements, arena.allocation.pointer, arena.requirements.total_bytes,
      arena.state, arena.workspace, arena.reports, host, initializer);
  if (!diagnostic.success()) {
    std::fprintf(stderr, "fresh initializer failed: error=%u field=%u index=%lld\n",
                 static_cast<unsigned>(diagnostic.error), static_cast<unsigned>(diagnostic.field),
                 static_cast<long long>(diagnostic.index));
  }
  CHECK(diagnostic.success());
  CHECK(initializer.valid());
  CHECK(initializer.image_bytes() == arena.requirements.total_bytes);
  CHECK(initializer.plan_token() == kToken);
  CHECK(initializer.initialization_generation() == 3u);
  CHECK(initializer.device_checkpoint() != nullptr);
  CHECK(initializer.device_checkpoint() != arena.allocation.pointer);

  /* create() establishes a separate immutable device checkpoint and must not
   * touch the mutable destination arena. */
  std::array<std::byte, 32> sentinel{};
  CUDA_CHECK(cudaMemcpy(sentinel.data(), arena.allocation.pointer, sentinel.size(),
                        cudaMemcpyDeviceToHost));
  for (const std::byte value : sentinel) CHECK(value == std::byte{0xa5});

  Stream stream;
  CHECK(stream.value != nullptr);
  CUDA_CHECK(cudaMemsetAsync(arena.allocation.pointer, 0x3c, arena.requirements.total_bytes,
                             stream.value));
  Gfn2SccIterationInitializationReady ready{};
  diagnostic = initializer.upload_async(arena.allocation.pointer, arena.requirements.total_bytes,
                                        ready, stream.value);
  CHECK(diagnostic.success());
  CHECK(ready.ready_on_stream == 1u);
  CHECK(ready.plan_token == kToken);
  CHECK(ready.initialization_generation == 3u);
  CHECK(ready.device_arena == arena.allocation.pointer);

  const auto qsh = copy_from_device(arena.state.raw_population.qsh, kShells, stream.value);
  CHECK(qsh == data.qsh);
  const auto qat = copy_from_device(arena.state.raw_population.qat, kAtoms, stream.value);
  CHECK(qat == data.qat);
  const auto dipoles =
      copy_from_device(arena.state.scc.current_inputs.atomic_dipoles, kDipoles, stream.value);
  CHECK(dipoles == data.dipoles);
  const auto initialized = copy_from_device(arena.state.mixer.initialized, kBatch, stream.value);
  CHECK(initialized == std::vector<std::uint8_t>({1u, 1u}));

  const std::vector<double> expected_packed{
      data.qsh[0],          data.qsh[1],          data.qsh[2],          data.dipoles[0],
      data.dipoles[1],      data.dipoles[2],      data.dipoles[3],      data.dipoles[4],
      data.dipoles[5],      data.quadrupoles[0],  data.quadrupoles[1],  data.quadrupoles[2],
      data.quadrupoles[3],  data.quadrupoles[4],  data.quadrupoles[5],  data.quadrupoles[6],
      data.quadrupoles[7],  data.quadrupoles[8],  data.quadrupoles[9],  data.quadrupoles[10],
      data.quadrupoles[11], data.qsh[3],          data.dipoles[6],      data.dipoles[7],
      data.dipoles[8],      data.quadrupoles[12], data.quadrupoles[13], data.quadrupoles[14],
      data.quadrupoles[15], data.quadrupoles[16], data.quadrupoles[17]};
  const auto packed =
      copy_from_device(arena.state.mixer.current_inputs, kMixerVector, stream.value);
  CHECK(packed == expected_packed);

  const auto free_energy =
      copy_from_device(arena.state.free_energy.free_energy, kBatch, stream.value);
  CHECK(free_energy.size() == kBatch);
  CHECK(std::isnan(free_energy[0]) && std::isnan(free_energy[1]));
  const auto disabled_d4 =
      copy_from_device(arena.state.free_energy.d4_two_body, kBatch, stream.value);
  CHECK(disabled_d4 == std::vector<double>({0.0, 0.0}));
  const auto ledger =
      copy_from_device(arena.workspace.ledger.system_failure_records, kBatch, stream.value);
  CHECK(ledger == std::vector<std::uint64_t>({0u, 0u}));
  const auto report_errors = copy_from_device(arena.reports.system_errors,
                                              arena.reports.system_error_elements, stream.value);
  for (const std::uint32_t value : report_errors) CHECK(value == 0u);
  return 0;
}

int test_device_checkpoint_repeated_restore_and_graph_replay() {
  const std::uint32_t mandatory = static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES2) |
                                  static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES3) |
                                  static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kAES2);
  BoundArena arena(mandatory);
  CHECK(arena.valid);
  FreshData data;
  Gfn2SccIterationInitializer initializer;
  CHECK(
      Gfn2SccIterationInitializer::create(arena.plan, arena.requirements, arena.allocation.pointer,
                                          arena.requirements.total_bytes, arena.state,
                                          arena.workspace, arena.reports, data.view(), initializer)
          .success());

  const void* const checkpoint = initializer.device_checkpoint();
  CHECK(checkpoint != nullptr);
  CHECK(checkpoint != arena.allocation.pointer);
  cudaPointerAttributes checkpoint_attributes{};
  CUDA_CHECK(cudaPointerGetAttributes(&checkpoint_attributes, checkpoint));
  CHECK(checkpoint_attributes.type == cudaMemoryTypeDevice);

  Stream stream;
  CHECK(stream.value != nullptr);
  Gfn2SccIterationInitializationReady ready{};
  for (std::uint8_t fill : {std::uint8_t{0x19}, std::uint8_t{0xe3}}) {
    CUDA_CHECK(cudaMemsetAsync(arena.allocation.pointer, fill, arena.requirements.total_bytes,
                               stream.value));
    const auto diagnostic = initializer.upload_async(
        arena.allocation.pointer, arena.requirements.total_bytes, ready, stream.value);
    CHECK(diagnostic.success());
    CHECK(initializer.device_checkpoint() == checkpoint);
    CHECK(ready.device_arena == arena.allocation.pointer);
    CHECK(copy_from_device(arena.state.raw_population.qsh, kShells, stream.value) == data.qsh);
  }

  Graph captured;
  CUDA_CHECK(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
  const auto capture_diagnostic = initializer.upload_async(
      arena.allocation.pointer, arena.requirements.total_bytes, ready, stream.value);
  CHECK(capture_diagnostic.success());
  CUDA_CHECK(cudaStreamEndCapture(stream.value, &captured.graph));
  CHECK(captured.graph != nullptr);

  std::size_t node_count = 0u;
  CUDA_CHECK(cudaGraphGetNodes(captured.graph, nullptr, &node_count));
  CHECK(node_count == 1u);
  std::array<cudaGraphNode_t, 1> nodes{};
  CUDA_CHECK(cudaGraphGetNodes(captured.graph, nodes.data(), &node_count));
  cudaGraphNodeType node_type{};
  CUDA_CHECK(cudaGraphNodeGetType(nodes[0], &node_type));
  CHECK(node_type == cudaGraphNodeTypeMemcpy);
  cudaMemcpy3DParms copy_parameters{};
  CUDA_CHECK(cudaGraphMemcpyNodeGetParams(nodes[0], &copy_parameters));
  CHECK(copy_parameters.kind == cudaMemcpyDeviceToDevice);
  CHECK(copy_parameters.srcPtr.ptr == checkpoint);
  CHECK(copy_parameters.dstPtr.ptr == arena.allocation.pointer);
  CHECK(copy_parameters.extent.width == arena.requirements.total_bytes);
  CUDA_CHECK(cudaGraphInstantiate(&captured.executable, captured.graph, nullptr, nullptr, 0u));

  for (std::uint8_t fill : {std::uint8_t{0x47}, std::uint8_t{0xb2}, std::uint8_t{0x6d}}) {
    CUDA_CHECK(cudaMemsetAsync(arena.allocation.pointer, fill, arena.requirements.total_bytes,
                               stream.value));
    CUDA_CHECK(cudaGraphLaunch(captured.executable, stream.value));
    CHECK(copy_from_device(arena.state.raw_population.qsh, kShells, stream.value) == data.qsh);
    CHECK(initializer.device_checkpoint() == checkpoint);
  }
  return 0;
}

int test_queued_restore_is_drained_before_checkpoint_destruction() {
  const std::uint32_t mandatory = static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES2) |
                                  static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES3) |
                                  static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kAES2);
  BoundArena arena(mandatory);
  CHECK(arena.valid);
  FreshData data;
  Stream stream;
  CHECK(stream.value != nullptr);

  {
    Gfn2SccIterationInitializer initializer;
    CHECK(Gfn2SccIterationInitializer::create(
              arena.plan, arena.requirements, arena.allocation.pointer,
              arena.requirements.total_bytes, arena.state, arena.workspace, arena.reports,
              data.view(), initializer)
              .success());
    CUDA_CHECK(cudaMemsetAsync(arena.allocation.pointer, 0xa9, arena.requirements.total_bytes,
                               stream.value));
    delay_device_cycles<<<1, 1, 0, stream.value>>>(5'000'000ULL);
    CUDA_CHECK(cudaGetLastError());
    Gfn2SccIterationInitializationReady ready{};
    CHECK(initializer
              .upload_async(arena.allocation.pointer, arena.requirements.total_bytes, ready,
                            stream.value)
              .success());
    /* Deliberately leave the restore queued when the device checkpoint owner
     * is destroyed. Its completion event must drain source use before free. */
  }
  CHECK(copy_from_device(arena.state.raw_population.qsh, kShells, stream.value) == data.qsh);
  return 0;
}

struct WarmData {
  std::vector<std::int64_t> atom_offsets{0, 2, 3};
  std::vector<std::int64_t> shell_offsets{0, 3, 4};
  std::vector<double> eigenvalues;
  std::vector<double> coefficients;
  std::vector<double> occupations;
  std::vector<double> chemical_potentials;
  std::vector<double> electron_sums;
  std::vector<double> entropies{0.2, 0.3};
  std::vector<double> density;
  std::vector<double> weighted_density;
  std::vector<double> band_energies{1.1, 1.2};
  std::vector<double> occupation_sums{2.0, 3.0};
  std::vector<double> density_traces{2.0, 3.0};
  std::vector<double> weighted_density_traces{0.4, 0.5};
  std::vector<double> qsh{0.1, 0.2, 0.3, 9.9};
  std::vector<double> qat{0.3, 0.3, 9.9};
  std::vector<double> dipoles;
  std::vector<double> quadrupoles;
  std::vector<double> current_qsh{0.1, 0.2, 0.3, -0.6};
  std::vector<double> current_dipoles;
  std::vector<double> current_quadrupoles;

  std::vector<double> core{1.0, 2.0};
  std::vector<double> es2{0.1, 0.2};
  std::vector<double> es3{0.01, 0.02};
  std::vector<double> aes2{0.03, 0.04};
  std::vector<double> d4{0.05, 0.06};
  std::vector<double> point_charge{0.07, 0.08};
  std::vector<double> periodic{0.09, 0.10};
  std::vector<double> internal{1.35, 2.50};
  std::vector<double> free{1.34, 2.48};
  std::vector<double> classical_total{0.35, 0.50};

  std::vector<double> mixer_current;
  std::vector<double> previous_inputs;
  std::vector<double> previous_residuals;
  std::vector<double> df_history;
  std::vector<double> u_history;
  std::vector<double> omega;
  std::vector<double> residual_rms{0.2, 0.01};
  std::vector<double> residual_maximum{0.3, 0.02};
  std::vector<std::uint64_t> iterations{2u, 3u};
  std::vector<std::uint64_t> restart_counts{1u, 4u};
  std::vector<gpuxtb_status_t> mixer_statuses{GPUXTB_STATUS_SUCCESS, GPUXTB_STATUS_SUCCESS};
  std::vector<gpuxtb_status_t> scc_statuses{GPUXTB_STATUS_SUCCESS, GPUXTB_STATUS_SUCCESS};
  std::vector<std::uint8_t> initialized{1u, 1u};
  std::vector<std::uint8_t> residual_converged{0u, 1u};
  std::vector<double> previous_free{1.0, 2.0};
  std::vector<double> changes{0.34, 0.48};
  std::vector<std::uint8_t> converged{0u, 1u};

  static std::vector<double> sequence(std::size_t count, double start) {
    std::vector<double> result(count);
    for (std::size_t index = 0; index < count; ++index) {
      result[index] = start + 0.001 * static_cast<double>(index);
    }
    return result;
  }

  WarmData()
      : eigenvalues(sequence(kOrbitals, -1.0)),
        coefficients(sequence(kMatrices, 0.1)),
        occupations(sequence(2 * kOrbitals, 0.2)),
        chemical_potentials(sequence(2 * kBatch, -0.4)),
        electron_sums(sequence(2 * kBatch, 1.0)),
        density(sequence(kMatrices, 0.3)),
        weighted_density(sequence(kMatrices, 0.4)),
        dipoles(sequence(kDipoles, 0.01)),
        quadrupoles(sequence(kQuadrupoles, 0.02)),
        current_dipoles(dipoles),
        current_quadrupoles(quadrupoles),
        previous_inputs(sequence(kMixerVector, 0.5)),
        previous_residuals(sequence(kMixerVector, 0.6)),
        df_history(sequence(kMixerVector * kHistory, 0.7)),
        u_history(sequence(kMixerVector * kHistory, 0.8)),
        omega(sequence(kBatch * kHistory, 0.9)) {
    /* The first member exercises an energy decrease: the checkpoint stores
     * the signed driver trace, not a convergence-only absolute magnitude. */
    previous_free[0] = 2.0;
    changes[0] = free[0] - previous_free[0];
    changes[1] = free[1] - previous_free[1];
    /* Converged member publishes raw population while private current remains mixed. */
    dipoles[6] = 8.1;
    dipoles[7] = 8.2;
    dipoles[8] = 8.3;
    for (std::size_t index = 12u; index < quadrupoles.size(); ++index) {
      quadrupoles[index] = 8.0 + static_cast<double>(index);
    }
    mixer_current.reserve(kMixerVector);
    mixer_current.insert(mixer_current.end(), current_qsh.begin(), current_qsh.begin() + 3);
    mixer_current.insert(mixer_current.end(), current_dipoles.begin(), current_dipoles.begin() + 6);
    mixer_current.insert(mixer_current.end(), current_quadrupoles.begin(),
                         current_quadrupoles.begin() + 12);
    mixer_current.push_back(current_qsh[3]);
    mixer_current.insert(mixer_current.end(), current_dipoles.begin() + 6, current_dipoles.end());
    mixer_current.insert(mixer_current.end(), current_quadrupoles.begin() + 12,
                         current_quadrupoles.end());
  }

  Gfn2SccIterationHostInitialization view() const {
    Gfn2SccIterationHostInitialization host{};
    host.mode = Gfn2SccIterationInitializationMode::kWarm;
    host.plan_token = kToken;
    host.initialization_generation = 11u;
    host.topology = {host_view(atom_offsets), host_view(shell_offsets), kToken};
    host.wavefunction = {
        host_view(eigenvalues),
        host_view(coefficients),
        host_view(occupations),
        host_view(chemical_potentials),
        host_view(electron_sums),
        host_view(entropies),
        host_view(density),
        host_view(weighted_density),
        host_view(band_energies),
        host_view(occupation_sums),
        host_view(density_traces),
        host_view(weighted_density_traces),
        {host_view(qsh), host_view(qat), host_view(dipoles), host_view(quadrupoles), kToken},
        kToken};
    host.energy = {host_view(core),
                   host_view(es2),
                   host_view(es3),
                   host_view(aes2),
                   host_view(d4),
                   host_view(point_charge),
                   host_view(periodic),
                   host_view(entropies),
                   host_view(internal),
                   host_view(free),
                   host_view(classical_total),
                   kToken};
    host.mixer = {host_view(mixer_current),      host_view(previous_inputs),
                  host_view(previous_residuals), host_view(df_history),
                  host_view(u_history),          host_view(omega),
                  host_view(residual_rms),       host_view(residual_maximum),
                  host_view(iterations),         host_view(restart_counts),
                  host_view(mixer_statuses),     host_view(initialized),
                  host_view(residual_converged), kToken};
    host.scc = {host_view(current_qsh),
                host_view(current_dipoles),
                host_view(current_quadrupoles),
                host_view(free),
                host_view(previous_free),
                host_view(changes),
                host_view(residual_rms),
                host_view(iterations),
                host_view(scc_statuses),
                host_view(converged),
                kToken};
    return host;
  }
};

int test_warm_checkpoint_round_trip() {
  BoundArena arena(kGfn2SccPotentialAllComponents);
  CHECK(arena.valid);
  WarmData data;
  const auto host = data.view();
  Gfn2SccIterationInitializer initializer;
  auto diagnostic = Gfn2SccIterationInitializer::create(
      arena.plan, arena.requirements, arena.allocation.pointer, arena.requirements.total_bytes,
      arena.state, arena.workspace, arena.reports, host, initializer);
  if (!diagnostic.success()) {
    std::fprintf(stderr, "warm initializer failed: error=%u field=%u index=%lld\n",
                 static_cast<unsigned>(diagnostic.error), static_cast<unsigned>(diagnostic.field),
                 static_cast<long long>(diagnostic.index));
  }
  CHECK(diagnostic.success());
  Stream stream;
  Gfn2SccIterationInitializationReady ready{};
  diagnostic = initializer.upload_async(arena.allocation.pointer, arena.requirements.total_bytes,
                                        ready, stream.value);
  CHECK(diagnostic.success());
  CHECK(ready.mode == Gfn2SccIterationInitializationMode::kWarm);
  CHECK(ready.initialization_generation == 11u);

  CHECK(copy_from_device(arena.state.eigenpairs.coefficients, kMatrices, stream.value) ==
        data.coefficients);
  CHECK(copy_from_device(arena.state.raw_population.qsh, kShells, stream.value) == data.qsh);
  CHECK(copy_from_device(arena.state.mixer.current_inputs, kMixerVector, stream.value) ==
        data.mixer_current);
  CHECK(copy_from_device(arena.state.mixer.df_history, kMixerVector * kHistory, stream.value) ==
        data.df_history);
  CHECK(copy_from_device(arena.state.mixer.restart_counts, kBatch, stream.value) ==
        data.restart_counts);
  CHECK(copy_from_device(arena.state.scc.current_inputs.shell_charges, kShells, stream.value) ==
        data.current_qsh);
  CHECK(copy_from_device(arena.state.scc.free_energies, kBatch, stream.value) == data.free);
  CHECK(copy_from_device(arena.state.scc.previous_free_energies, kBatch, stream.value) ==
        data.previous_free);
  CHECK(data.changes[0] < 0.0);
  CHECK(copy_from_device(arena.state.scc.free_energy_changes, kBatch, stream.value) ==
        data.changes);
  CHECK(copy_from_device(arena.state.free_energy.periodic_embedding, kBatch, stream.value) ==
        data.periodic);
  CHECK(copy_from_device(arena.state.scc.converged, kBatch, stream.value) == data.converged);
  return 0;
}

int test_max_iteration_preserves_mixer_and_scc_status_roles() {
  BoundArena arena(kGfn2SccPotentialAllComponents);
  CHECK(arena.valid);
  WarmData data;
  data.iterations[0] = arena.plan.state_policy.maximum_iterations;
  data.mixer_statuses[0] = GPUXTB_STATUS_SUCCESS;
  data.scc_statuses[0] = GPUXTB_STATUS_SCC_NOT_CONVERGED;
  data.converged[0] = 0u;

  Gfn2SccIterationInitializer initializer;
  auto diagnostic = Gfn2SccIterationInitializer::create(
      arena.plan, arena.requirements, arena.allocation.pointer, arena.requirements.total_bytes,
      arena.state, arena.workspace, arena.reports, data.view(), initializer);
  CHECK(diagnostic.success());

  Stream stream;
  Gfn2SccIterationInitializationReady ready{};
  diagnostic = initializer.upload_async(arena.allocation.pointer, arena.requirements.total_bytes,
                                        ready, stream.value);
  CHECK(diagnostic.success());
  CHECK(copy_from_device(arena.state.mixer.system_statuses, kBatch, stream.value) ==
        data.mixer_statuses);
  CHECK(copy_from_device(arena.state.scc.system_statuses, kBatch, stream.value) ==
        data.scc_statuses);
  CHECK(copy_from_device(arena.state.scc.iterations, kBatch, stream.value) == data.iterations);

  /* Equal terminal statuses are not a publication-produced checkpoint: the
   * mixer reports its successful transition while the SCC driver reports the
   * maximum-iteration terminal state. */
  data.mixer_statuses[0] = GPUXTB_STATUS_SCC_NOT_CONVERGED;
  diagnostic = Gfn2SccIterationInitializer::create(
      arena.plan, arena.requirements, arena.allocation.pointer, arena.requirements.total_bytes,
      arena.state, arena.workspace, arena.reports, data.view(), initializer);
  CHECK(diagnostic.error == Gfn2SccIterationInitializationError::kInvalidWarmState);
  return 0;
}

int test_invalid_host_state_is_transactional() {
  BoundArena arena(kGfn2SccPotentialAllComponents);
  CHECK(arena.valid);
  FreshData fresh;
  Gfn2SccIterationInitializer initializer;
  auto host = fresh.view(5u);
  CHECK(Gfn2SccIterationInitializer::create(arena.plan, arena.requirements,
                                            arena.allocation.pointer,
                                            arena.requirements.total_bytes, arena.state,
                                            arena.workspace, arena.reports, host, initializer)
            .success());
  CHECK(initializer.initialization_generation() == 5u);

  host.wavefunction.population.shell_charges.elements -= 1;
  auto diagnostic = Gfn2SccIterationInitializer::create(
      arena.plan, arena.requirements, arena.allocation.pointer, arena.requirements.total_bytes,
      arena.state, arena.workspace, arena.reports, host, initializer);
  CHECK(diagnostic.error == Gfn2SccIterationInitializationError::kInvalidExtent);
  CHECK(initializer.initialization_generation() == 5u);

  host = fresh.view(6u);
  const double original = fresh.qsh[1];
  fresh.qsh[1] = std::numeric_limits<double>::infinity();
  host.wavefunction.population.shell_charges = host_view(fresh.qsh);
  diagnostic = Gfn2SccIterationInitializer::create(
      arena.plan, arena.requirements, arena.allocation.pointer, arena.requirements.total_bytes,
      arena.state, arena.workspace, arena.reports, host, initializer);
  CHECK(diagnostic.error == Gfn2SccIterationInitializationError::kNonfiniteValue);
  CHECK(initializer.initialization_generation() == 5u);
  fresh.qsh[1] = original;

  host = fresh.view(6u);
  host.plan_token ^= 1u;
  diagnostic = Gfn2SccIterationInitializer::create(
      arena.plan, arena.requirements, arena.allocation.pointer, arena.requirements.total_bytes,
      arena.state, arena.workspace, arena.reports, host, initializer);
  CHECK(diagnostic.error == Gfn2SccIterationInitializationError::kCrossPlan);
  CHECK(initializer.initialization_generation() == 5u);

  WarmData warm;
  auto warm_host = warm.view();
  warm.mixer_current[0] += 1.0;
  warm_host.mixer.current_inputs = host_view(warm.mixer_current);
  diagnostic = Gfn2SccIterationInitializer::create(
      arena.plan, arena.requirements, arena.allocation.pointer, arena.requirements.total_bytes,
      arena.state, arena.workspace, arena.reports, warm_host, initializer);
  CHECK(diagnostic.error == Gfn2SccIterationInitializationError::kInvalidWarmState);
  CHECK(initializer.initialization_generation() == 5u);
  return 0;
}

int test_failed_upload_clears_ready_and_submits_no_partial_copy() {
  const std::uint32_t mandatory = static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES2) |
                                  static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES3) |
                                  static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kAES2);
  BoundArena arena(mandatory);
  CHECK(arena.valid);
  FreshData data;
  Gfn2SccIterationInitializer initializer;
  CHECK(
      Gfn2SccIterationInitializer::create(arena.plan, arena.requirements, arena.allocation.pointer,
                                          arena.requirements.total_bytes, arena.state,
                                          arena.workspace, arena.reports, data.view(), initializer)
          .success());
  CUDA_CHECK(cudaMemset(arena.allocation.pointer, 0x7b, arena.requirements.total_bytes));

  Gfn2SccIterationInitializationReady ready{};
  ready.ready_on_stream = 1u;
  ready.plan_token = 99u;
  auto diagnostic = initializer.upload_async(
      static_cast<std::byte*>(arena.allocation.pointer) + arena.requirements.alignment,
      arena.requirements.total_bytes, ready, nullptr);
  CHECK(diagnostic.error == Gfn2SccIterationInitializationError::kInvalidArena);
  CHECK(ready.ready_on_stream == 0u);
  CHECK(ready.plan_token == 0u);
  std::array<std::byte, 64> sentinel{};
  CUDA_CHECK(cudaMemcpy(sentinel.data(), arena.allocation.pointer, sentinel.size(),
                        cudaMemcpyDeviceToHost));
  for (const std::byte value : sentinel) CHECK(value == std::byte{0x7b});

  void* released = arena.allocation.release();
  CUDA_CHECK(cudaFree(released));
  ready.ready_on_stream = 1u;
  diagnostic = initializer.upload_async(released, arena.requirements.total_bytes, ready, nullptr);
  CHECK(diagnostic.error == Gfn2SccIterationInitializationError::kInvalidArenaMemory);
  CHECK(ready.ready_on_stream == 0u);
  return 0;
}

}  // namespace

int main() {
  const std::array<int (*)(), 7> tests{{
      test_fresh_initialization_and_stream_ready_publication,
      test_device_checkpoint_repeated_restore_and_graph_replay,
      test_queued_restore_is_drained_before_checkpoint_destruction,
      test_warm_checkpoint_round_trip,
      test_max_iteration_preserves_mixer_and_scc_status_roles,
      test_invalid_host_state_is_transactional,
      test_failed_upload_clears_ready_and_submits_no_partial_copy,
  }};
  for (const auto test : tests) {
    if (const int line = test(); line != 0) {
      std::fprintf(stderr, "CUDA SCC iteration initialize test failed at line %d\n", line);
      return 1;
    }
  }
  return 0;
}
