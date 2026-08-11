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
#include <vector>

#include "backends/cuda/gfn2_inference_publication.cuh"
#include "backends/cuda/gfn2_terminal_classical_energy.cuh"
#include "data/parameters/d4.hpp"
#include "model/gfn2/d4.hpp"
#include "model/gfn2/repulsion.hpp"
#include "runtime/backend.hpp"

namespace {

using namespace xtbloom::detail;
using namespace xtbloom::detail::cuda;
using namespace xtbloom::detail::gfn2;

#define CHECK(condition)                                                                         \
  do {                                                                                           \
    if (!(condition)) {                                                                          \
      std::fprintf(stderr, "terminal execution check failed at %s:%d: %s\n", __FILE__, __LINE__, \
                   #condition);                                                                  \
      return __LINE__;                                                                           \
    }                                                                                            \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

constexpr double kSentinel = -7319.25;

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (data_ != nullptr) (void)cudaFree(data_);
  }

  cudaError_t allocate(std::size_t count) {
    count_ = count;
    return count == 0u ? cudaSuccess
                       : cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
  }

  cudaError_t upload(const T* source, std::size_t count, cudaStream_t stream = nullptr) {
    if (count > count_ || (count != 0u && source == nullptr)) return cudaErrorInvalidValue;
    return count == 0u
               ? cudaSuccess
               : cudaMemcpyAsync(data_, source, count * sizeof(T), cudaMemcpyHostToDevice, stream);
  }

  cudaError_t upload(const std::vector<T>& source, cudaStream_t stream = nullptr) {
    return upload(source.data(), source.size(), stream);
  }

  template <std::size_t Count>
  cudaError_t upload(const std::array<T, Count>& source, cudaStream_t stream = nullptr) {
    return upload(source.data(), source.size(), stream);
  }

  cudaError_t download(T* target, std::size_t count, cudaStream_t stream = nullptr) const {
    if (count > count_ || (count != 0u && target == nullptr)) return cudaErrorInvalidValue;
    return count == 0u
               ? cudaSuccess
               : cudaMemcpyAsync(target, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
  }

  cudaError_t fill(T value, cudaStream_t stream = nullptr) {
    std::vector<T> host(count_, value);
    return upload(host, stream);
  }

  T* get() noexcept { return data_; }
  const T* get() const noexcept { return data_; }
  std::size_t size() const noexcept { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

bool near(double actual, double expected, double tolerance = 8.0e-12) {
  return std::abs(actual - expected) <=
         tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

template <typename T>
bool byte_equal(const std::vector<T>& first, const std::vector<T>& second) {
  return first.size() == second.size() &&
         std::memcmp(first.data(), second.data(), first.size() * sizeof(T)) == 0;
}

struct TerminalHostData {
  static constexpr std::array<std::int64_t, 3> atom_offsets{0, 3, 4};
  /* The first peer is bent CH2 so the terminal path executes a real ATM
   * triple; the second singleton has zero published pairs despite nonzero
   * fixed backing capacity. */
  static constexpr std::array<std::int32_t, 4> atomic_numbers{6, 1, 1, 1};
  static constexpr std::array<double, 12> positions{
      0.0, 0.0, 0.0, 1.43, 1.11, 0.0, -1.43, 1.11, 0.0, 0.0, 0.0, 0.0,
  };

  RepulsionPlan repulsion;
  D4Plan d4;
  std::vector<std::byte> d4_workspace_storage;
  D4Workspace d4_workspace;
  std::vector<double> pair_data;
  std::vector<double> coordination;
  D4GeometryCache cache;
  std::array<double, 2> expected_repulsion{};
  std::array<double, 2> expected_atm{};

  bool initialize() {
    std::string error;
    const auto total_atoms = static_cast<std::int64_t>(atomic_numbers.size());
    if (make_repulsion_plan(2, total_atoms, atom_offsets.data(), atomic_numbers.data(), repulsion,
                            error) != XTBLOOM_STATUS_SUCCESS ||
        add_repulsion_cpu(repulsion, positions.data(), expected_repulsion.data(), nullptr, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        make_d4_plan(2, total_atoms, atom_offsets.data(), atomic_numbers.data(), d4, error) !=
            XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    d4_workspace_storage.resize(d4.workspace_size_bytes() + kD4WorkspaceAlignment - 1u);
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(d4_workspace_storage.data());
    const std::uintptr_t aligned =
        (address + kD4WorkspaceAlignment - 1u) & ~(kD4WorkspaceAlignment - 1u);
    if (bind_d4_workspace(d4, reinterpret_cast<void*>(aligned), d4.workspace_size_bytes(),
                          d4_workspace, error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    pair_data.resize(static_cast<std::size_t>(d4.total_pairs()) *
                     static_cast<std::size_t>(gfn2::kD4PairDataElements));
    coordination.resize(static_cast<std::size_t>(d4.total_atoms()));
    if (update_d4_geometry_cache_cpu(d4, positions.data(), 7u, pair_data.data(), pair_data.size(),
                                     coordination.data(), coordination.size(), d4_workspace, cache,
                                     error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    return evaluate_d4_atm_cpu(d4, cache, expected_atm.data(), d4_workspace, error) ==
           XTBLOOM_STATUS_SUCCESS;
  }
};

struct TerminalDeviceFixture {
  static constexpr std::int64_t kBatch = 2;
  static constexpr std::int64_t kAtoms = 4;

  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> pair_offsets;
  DeviceBuffer<std::int64_t> pairlist_offsets;
  DeviceBuffer<Gfn2AtomPair> pairs;
  DeviceBuffer<std::int64_t> pair_counts;
  DeviceBuffer<std::int64_t> neighbor_offsets;
  DeviceBuffer<std::int64_t> neighbor_counts;
  DeviceBuffer<std::int64_t> neighbors;
  DeviceBuffer<std::int32_t> atomic_numbers;
  DeviceBuffer<double> positions;
  DeviceBuffer<Gfn2D4DeviceElementData> elements;
  DeviceBuffer<Gfn2D4DeviceReferenceData> references;
  DeviceBuffer<double> reference_c6;
  DeviceBuffer<double> coordination;
  DeviceBuffer<double> weights;
  DeviceBuffer<double> weight_cn_derivatives;
  DeviceBuffer<double> weight_charge_derivatives;
  DeviceBuffer<double> atom_scratch;
  DeviceBuffer<double> coordination_adjoints;
  DeviceBuffer<double> batch_scratch;
  DeviceBuffer<double> gradient_scratch;
  DeviceBuffer<std::uint64_t> epoch;
  DeviceBuffer<std::uint64_t> committed_generations;
  DeviceBuffer<std::uint64_t> pair_generations;
  DeviceBuffer<std::uint8_t> pair_eligible;
  DeviceBuffer<std::uint8_t> coordination_eligible;
  DeviceBuffer<std::uint8_t> requested;
  DeviceBuffer<double> result_repulsion;
  DeviceBuffer<double> result_atm;
  DeviceBuffer<double> candidate_repulsion;
  DeviceBuffer<double> candidate_atm;
  DeviceBuffer<std::uint64_t> epoch_snapshot;
  DeviceBuffer<std::uint32_t> repulsion_device_error;
  DeviceBuffer<std::uint32_t> d4_system_errors;
  DeviceBuffer<std::uint32_t> d4_device_error;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> plan_error;

  Gfn2TerminalClassicalEnergyDevicePlan plan;
  Gfn2TerminalClassicalEnergyDeviceActivity activity;
  Gfn2TerminalClassicalEnergyDeviceResults results;
  Gfn2TerminalClassicalEnergyDeviceWorkspace workspace;
  Gfn2TerminalClassicalEnergyDeviceDiagnostics diagnostics;

  bool initialize(const TerminalHostData& host, cudaStream_t stream) {
    std::vector<Gfn2D4DeviceElementData> host_elements;
    host_elements.reserve(xtbloom::parameters::d4::kElements.size());
    for (const auto& element : xtbloom::parameters::d4::kElements) {
      host_elements.push_back({element.reference_offset, element.reference_count,
                               element.covalent_radius, element.electronegativity,
                               element.effective_charge, element.hardness, element.r4r2});
    }
    std::vector<Gfn2D4DeviceReferenceData> host_references;
    host_references.reserve(xtbloom::parameters::d4::kReferences.size());
    for (const auto& reference : xtbloom::parameters::d4::kReferences) {
      host_references.push_back(
          {reference.coordination_number, reference.charge, reference.gaussian_count});
    }
    const std::size_t weight_count = static_cast<std::size_t>(kAtoms * kGfn2D4MaximumReferences);
    const std::array<Gfn2AtomPair, 6> host_pairs{{{0, 1}, {0, 2}, {1, 2}, {0, 0}, {0, 0}, {0, 0}}};
    const std::array<std::int64_t, 2> host_pair_counts{3, 0};
    const std::array<std::int64_t, 3> host_pairlist_offsets{0, 3, 6};
    const std::array<std::int64_t, 5> host_neighbor_offsets{0, 2, 4, 6, 8};
    const std::array<std::int64_t, 4> host_neighbor_counts{2, 2, 2, 0};
    const std::array<std::int64_t, 8> host_neighbors{1, 2, 0, 2, 0, 1, 0, 0};
    const bool allocated =
        atom_offsets.allocate(TerminalHostData::atom_offsets.size()) == cudaSuccess &&
        pair_offsets.allocate(host.d4.pair_offsets().size()) == cudaSuccess &&
        pairlist_offsets.allocate(kBatch + 1) == cudaSuccess &&
        pairs.allocate(host_pairs.size()) == cudaSuccess &&
        pair_counts.allocate(host_pair_counts.size()) == cudaSuccess &&
        neighbor_offsets.allocate(host_neighbor_offsets.size()) == cudaSuccess &&
        neighbor_counts.allocate(host_neighbor_counts.size()) == cudaSuccess &&
        neighbors.allocate(host_neighbors.size()) == cudaSuccess &&
        atomic_numbers.allocate(TerminalHostData::atomic_numbers.size()) == cudaSuccess &&
        positions.allocate(TerminalHostData::positions.size()) == cudaSuccess &&
        elements.allocate(host_elements.size()) == cudaSuccess &&
        references.allocate(host_references.size()) == cudaSuccess &&
        reference_c6.allocate(xtbloom::parameters::d4::kReferenceC6.size()) == cudaSuccess &&
        coordination.allocate(host.coordination.size()) == cudaSuccess &&
        weights.allocate(weight_count) == cudaSuccess &&
        weight_cn_derivatives.allocate(weight_count) == cudaSuccess &&
        weight_charge_derivatives.allocate(weight_count) == cudaSuccess &&
        atom_scratch.allocate(kAtoms) == cudaSuccess &&
        coordination_adjoints.allocate(kAtoms) == cudaSuccess &&
        batch_scratch.allocate(kBatch) == cudaSuccess &&
        gradient_scratch.allocate(3 * kAtoms) == cudaSuccess && epoch.allocate(1) == cudaSuccess &&
        committed_generations.allocate(kBatch) == cudaSuccess &&
        pair_generations.allocate(kBatch) == cudaSuccess &&
        pair_eligible.allocate(kBatch) == cudaSuccess &&
        coordination_eligible.allocate(kBatch) == cudaSuccess &&
        requested.allocate(kBatch) == cudaSuccess &&
        result_repulsion.allocate(kBatch) == cudaSuccess &&
        result_atm.allocate(kBatch) == cudaSuccess &&
        candidate_repulsion.allocate(kBatch) == cudaSuccess &&
        candidate_atm.allocate(kBatch) == cudaSuccess &&
        epoch_snapshot.allocate(1) == cudaSuccess &&
        repulsion_device_error.allocate(1) == cudaSuccess &&
        d4_system_errors.allocate(kBatch) == cudaSuccess &&
        d4_device_error.allocate(1) == cudaSuccess &&
        system_errors.allocate(kBatch) == cudaSuccess && plan_error.allocate(1) == cudaSuccess;
    if (!allocated) return false;

    const std::array<std::uint64_t, 1> initial_epoch{7u};
    const std::array<std::uint64_t, 2> initial_generations{7u, 7u};
    const std::array<std::uint8_t, 2> initial_eligible{1u, 1u};
    const std::array<std::uint8_t, 2> initial_requested{1u, 1u};
    if (atom_offsets.upload(TerminalHostData::atom_offsets, stream) != cudaSuccess ||
        pair_offsets.upload(host.d4.pair_offsets().data(), host.d4.pair_offsets().size(), stream) !=
            cudaSuccess ||
        pairlist_offsets.upload(host_pairlist_offsets.data(), kBatch + 1, stream) != cudaSuccess ||
        pairs.upload(host_pairs, stream) != cudaSuccess ||
        pair_counts.upload(host_pair_counts, stream) != cudaSuccess ||
        neighbor_offsets.upload(host_neighbor_offsets, stream) != cudaSuccess ||
        neighbor_counts.upload(host_neighbor_counts, stream) != cudaSuccess ||
        neighbors.upload(host_neighbors, stream) != cudaSuccess ||
        atomic_numbers.upload(TerminalHostData::atomic_numbers, stream) != cudaSuccess ||
        positions.upload(TerminalHostData::positions, stream) != cudaSuccess ||
        elements.upload(host_elements, stream) != cudaSuccess ||
        references.upload(host_references, stream) != cudaSuccess ||
        reference_c6.upload(xtbloom::parameters::d4::kReferenceC6.data(),
                            xtbloom::parameters::d4::kReferenceC6.size(), stream) != cudaSuccess ||
        coordination.upload(host.coordination, stream) != cudaSuccess ||
        epoch.upload(initial_epoch, stream) != cudaSuccess ||
        committed_generations.upload(initial_generations, stream) != cudaSuccess ||
        pair_generations.upload(initial_generations, stream) != cudaSuccess ||
        pair_eligible.upload(initial_eligible, stream) != cudaSuccess ||
        coordination_eligible.upload(initial_eligible, stream) != cudaSuccess ||
        requested.upload(initial_requested, stream) != cudaSuccess) {
      return false;
    }

    const std::uint64_t token =
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(host.d4.identity()));
    plan.plan_token = token;
    plan.repulsion = {kBatch, kAtoms, atom_offsets.get(), atomic_numbers.get(), positions.get()};
    plan.d4_batch = {kBatch,
                     kAtoms,
                     host.d4.total_pairs(),
                     token,
                     gfn2_d4_atomic_number_hash(TerminalHostData::atomic_numbers.data(), kAtoms),
                     atom_offsets.get(),
                     pair_offsets.get(),
                     atomic_numbers.get()};
    plan.d4_parameters = {elements.get(),
                          static_cast<std::int64_t>(host_elements.size()),
                          references.get(),
                          static_cast<std::int64_t>(host_references.size()),
                          reference_c6.get(),
                          static_cast<std::int64_t>(xtbloom::parameters::d4::kReferenceC6.size())};
    plan.d4_cache.positions = positions.get();
    plan.d4_cache.position_elements = 3 * kAtoms;
    plan.d4_cache.coordination_numbers = coordination.get();
    plan.d4_cache.coordination_elements = kAtoms;
    plan.d4_cache.coordination_generations = committed_generations.get();
    plan.d4_cache.coordination_generation_elements = kBatch;
    plan.d4_cache.coordination_eligible_mask = coordination_eligible.get();
    plan.d4_cache.coordination_eligible_elements = kBatch;
    const auto pair_view = [&](Gfn2PairListRole role, double cutoff) {
      Gfn2PairListConsumerView view{};
      view.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
      view.state = Gfn2PairListState::kCommitted;
      view.role = role;
      view.pair_map_kind = Gfn2PairMapKind::kExplicit;
      view.plan_token = token;
      view.cutoff_bohr = cutoff;
      view.list_builder_cutoff_bohr = kGfn2D4TwoBodyCutoffBohr;
      view.batch_size = kBatch;
      view.total_atoms = kAtoms;
      view.max_pairs_per_system = 3;
      view.max_neighbors_per_atom = 2;
      view.pair_offset_count = kBatch + 1;
      view.neighbor_offset_count = kAtoms + 1;
      view.pair_count = static_cast<std::int64_t>(host_pairs.size());
      view.neighbor_count = static_cast<std::int64_t>(host_neighbors.size());
      view.pair_offsets = pairlist_offsets.get();
      view.pairs = pairs.get();
      view.pair_count_elements = kBatch;
      view.neighbor_count_elements = kAtoms;
      view.pair_counts = pair_counts.get();
      view.neighbor_counts = neighbor_counts.get();
      view.neighbor_offsets = neighbor_offsets.get();
      view.neighbors = neighbors.get();
      view.committed_generation_count = kBatch;
      view.eligible_mask_count = kBatch;
      view.committed_generations = pair_generations.get();
      view.eligible_mask = pair_eligible.get();
      return view;
    };
    plan.d4_cache.coordination_pairs =
        pair_view(Gfn2PairListRole::kD4Coordination, kGfn2D4CoordinationCutoffBohr);
    plan.d4_cache.two_body_pairs =
        pair_view(Gfn2PairListRole::kD4TwoBody, kGfn2D4TwoBodyCutoffBohr);
    plan.d4_cache.atm_pairs = pair_view(Gfn2PairListRole::kD4Atm, kGfn2D4AtmCutoffBohr);
    plan.d4_cache.plan_token = token;
    plan.geometry_epoch = {epoch.get(), 1, token};
    plan.committed_generations = committed_generations.get();
    plan.generation_elements = kBatch;
    activity = {requested.get(), kBatch, token};
    results = {result_repulsion.get(), kBatch, result_atm.get(), kBatch, token};
    workspace.repulsion_candidate = candidate_repulsion.get();
    workspace.repulsion_elements = kBatch;
    workspace.d4_atm_candidate = candidate_atm.get();
    workspace.d4_atm_elements = kBatch;
    workspace.d4 = {weights.get(),
                    weight_cn_derivatives.get(),
                    weight_charge_derivatives.get(),
                    static_cast<std::int64_t>(weight_count),
                    atom_scratch.get(),
                    coordination_adjoints.get(),
                    kAtoms,
                    batch_scratch.get(),
                    kBatch,
                    gradient_scratch.get(),
                    3 * kAtoms,
                    d4_system_errors.get(),
                    kBatch};
    workspace.epoch_snapshot = epoch_snapshot.get();
    workspace.epoch_snapshot_elements = 1;
    workspace.plan_token = token;
    diagnostics = {repulsion_device_error.get(),
                   d4_system_errors.get(),
                   kBatch,
                   d4_device_error.get(),
                   system_errors.get(),
                   kBatch,
                   plan_error.get(),
                   1,
                   token};
    return cudaStreamSynchronize(stream) == cudaSuccess;
  }

  void enable_d4(bool enabled) {
    plan.enabled_components =
        enabled ? static_cast<std::uint32_t>(Gfn2TerminalClassicalEnergyComponent::kD4Atm) : 0u;
    if (!enabled) {
      results.d4_atm = nullptr;
      results.d4_atm_elements = 0;
      workspace.d4_atm_candidate = nullptr;
      workspace.d4_atm_elements = 0;
      workspace.d4 = {};
      diagnostics.d4_system_errors = nullptr;
      diagnostics.d4_system_error_elements = 0;
      diagnostics.d4_device_error = nullptr;
    } else {
      results.d4_atm = result_atm.get();
      results.d4_atm_elements = kBatch;
      workspace.d4_atm_candidate = candidate_atm.get();
      workspace.d4_atm_elements = kBatch;
      const std::int64_t weight_count = kAtoms * kGfn2D4MaximumReferences;
      workspace.d4 = {weights.get(),
                      weight_cn_derivatives.get(),
                      weight_charge_derivatives.get(),
                      weight_count,
                      atom_scratch.get(),
                      coordination_adjoints.get(),
                      kAtoms,
                      batch_scratch.get(),
                      kBatch,
                      gradient_scratch.get(),
                      3 * kAtoms,
                      d4_system_errors.get(),
                      kBatch};
      diagnostics.d4_system_errors = d4_system_errors.get();
      diagnostics.d4_system_error_elements = kBatch;
      diagnostics.d4_device_error = d4_device_error.get();
    }
  }

  cudaError_t reset_results(cudaStream_t stream) {
    cudaError_t status = result_repulsion.fill(kSentinel, stream);
    if (status == cudaSuccess) status = result_atm.fill(kSentinel, stream);
    return status;
  }
};

int test_terminal_base_d4_and_rollbacks() {
  TerminalHostData host;
  CHECK(host.initialize());
  CHECK(std::isfinite(host.expected_atm[0]) && host.expected_atm[0] != 0.0);
  CHECK(host.expected_atm[1] == 0.0);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  TerminalDeviceFixture fixture;
  CHECK(fixture.initialize(host, stream));

  fixture.enable_d4(false);
  CUDA_CHECK(fixture.reset_results(stream));
  CUDA_CHECK(evaluate_gfn2_terminal_classical_energy_cuda(fixture.plan, fixture.activity,
                                                          fixture.results, fixture.workspace,
                                                          fixture.diagnostics, stream));
  std::array<double, 2> repulsion{};
  std::array<std::uint32_t, 2> base_system_errors{};
  std::uint32_t base_plan_error = 0u;
  std::uint32_t base_repulsion_error = 0u;
  CUDA_CHECK(fixture.result_repulsion.download(repulsion.data(), repulsion.size(), stream));
  CUDA_CHECK(
      fixture.system_errors.download(base_system_errors.data(), base_system_errors.size(), stream));
  CUDA_CHECK(fixture.plan_error.download(&base_plan_error, 1, stream));
  CUDA_CHECK(fixture.repulsion_device_error.download(&base_repulsion_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (!near(repulsion[0], host.expected_repulsion[0]) ||
      !near(repulsion[1], host.expected_repulsion[1])) {
    std::fprintf(stderr,
                 "base actual=[%.17g, %.17g] expected=[%.17g, %.17g] plan=%u rep=%u "
                 "system=[%u,%u]\n",
                 repulsion[0], repulsion[1], host.expected_repulsion[0], host.expected_repulsion[1],
                 base_plan_error, base_repulsion_error, base_system_errors[0],
                 base_system_errors[1]);
  }
  CHECK(base_plan_error == 0u && base_repulsion_error == 0u);
  CHECK(near(repulsion[0], host.expected_repulsion[0]));
  CHECK(near(repulsion[1], host.expected_repulsion[1]));

  fixture.enable_d4(true);
  CUDA_CHECK(fixture.reset_results(stream));
  CUDA_CHECK(evaluate_gfn2_terminal_classical_energy_cuda(fixture.plan, fixture.activity,
                                                          fixture.results, fixture.workspace,
                                                          fixture.diagnostics, stream));
  std::array<double, 2> atm{};
  CUDA_CHECK(fixture.result_repulsion.download(repulsion.data(), repulsion.size(), stream));
  CUDA_CHECK(fixture.result_atm.download(atm.data(), atm.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (std::size_t system = 0u; system < repulsion.size(); ++system) {
    CHECK(near(repulsion[system], host.expected_repulsion[system]));
    CHECK(near(atm[system], host.expected_atm[system], 2.0e-11));
  }

  /* A stale peer retains its complete old component tuple. */
  const std::array<std::uint64_t, 2> stale_generations{7u, 6u};
  CUDA_CHECK(fixture.committed_generations.upload(stale_generations, stream));
  CUDA_CHECK(fixture.reset_results(stream));
  CUDA_CHECK(evaluate_gfn2_terminal_classical_energy_cuda(fixture.plan, fixture.activity,
                                                          fixture.results, fixture.workspace,
                                                          fixture.diagnostics, stream));
  std::array<std::uint32_t, 2> system_errors{};
  CUDA_CHECK(fixture.result_repulsion.download(repulsion.data(), repulsion.size(), stream));
  CUDA_CHECK(fixture.result_atm.download(atm.data(), atm.size(), stream));
  CUDA_CHECK(fixture.system_errors.download(system_errors.data(), system_errors.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(near(repulsion[0], host.expected_repulsion[0]));
  CHECK(near(atm[0], host.expected_atm[0], 2.0e-11));
  CHECK(repulsion[1] == kSentinel && atm[1] == kSentinel);
  CHECK(system_errors[1] ==
        static_cast<std::uint32_t>(Gfn2TerminalClassicalEnergySystemError::kStaleGeneration));

  /* Pair-list provenance is independent of the terminal tuple generation.
   * A stale committed role therefore fails only D4 for that peer and rolls
   * back the complete terminal tuple through the existing publication gate. */
  const std::array<std::uint64_t, 2> fresh_generations{7u, 7u};
  const std::array<std::uint64_t, 2> stale_pair_generations{7u, 6u};
  CUDA_CHECK(fixture.committed_generations.upload(fresh_generations, stream));
  CUDA_CHECK(fixture.pair_generations.upload(stale_pair_generations, stream));
  CUDA_CHECK(fixture.reset_results(stream));
  CUDA_CHECK(evaluate_gfn2_terminal_classical_energy_cuda(fixture.plan, fixture.activity,
                                                          fixture.results, fixture.workspace,
                                                          fixture.diagnostics, stream));
  CUDA_CHECK(fixture.result_repulsion.download(repulsion.data(), repulsion.size(), stream));
  CUDA_CHECK(fixture.result_atm.download(atm.data(), atm.size(), stream));
  CUDA_CHECK(fixture.system_errors.download(system_errors.data(), system_errors.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(near(repulsion[0], host.expected_repulsion[0]));
  CHECK(near(atm[0], host.expected_atm[0], 2.0e-11));
  CHECK(repulsion[1] == kSentinel && atm[1] == kSentinel);
  CHECK(system_errors[1] ==
        static_cast<std::uint32_t>(Gfn2TerminalClassicalEnergySystemError::kD4AtmFailure));
  CUDA_CHECK(fixture.pair_generations.upload(fresh_generations, stream));

  /* A D4-local cache failure rolls back only the affected peer. */
  std::vector<double> invalid_coordination = host.coordination;
  invalid_coordination[3] = std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(fixture.committed_generations.upload(fresh_generations, stream));
  CUDA_CHECK(fixture.coordination.upload(invalid_coordination, stream));
  CUDA_CHECK(fixture.reset_results(stream));
  CUDA_CHECK(evaluate_gfn2_terminal_classical_energy_cuda(fixture.plan, fixture.activity,
                                                          fixture.results, fixture.workspace,
                                                          fixture.diagnostics, stream));
  CUDA_CHECK(fixture.result_repulsion.download(repulsion.data(), repulsion.size(), stream));
  CUDA_CHECK(fixture.result_atm.download(atm.data(), atm.size(), stream));
  CUDA_CHECK(fixture.system_errors.download(system_errors.data(), system_errors.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(near(repulsion[0], host.expected_repulsion[0]));
  CHECK(near(atm[0], host.expected_atm[0], 2.0e-11));
  CHECK(repulsion[1] == kSentinel && atm[1] == kSentinel);
  CHECK(system_errors[1] ==
        static_cast<std::uint32_t>(Gfn2TerminalClassicalEnergySystemError::kD4AtmFailure));
  CUDA_CHECK(fixture.coordination.upload(host.coordination, stream));

  /* A plan-wide mask error suppresses every public component write. */
  const std::array<std::uint8_t, 2> invalid_requested{1u, 2u};
  CUDA_CHECK(fixture.requested.upload(invalid_requested, stream));
  CUDA_CHECK(fixture.reset_results(stream));
  CUDA_CHECK(evaluate_gfn2_terminal_classical_energy_cuda(fixture.plan, fixture.activity,
                                                          fixture.results, fixture.workspace,
                                                          fixture.diagnostics, stream));
  std::uint32_t plan_error = 0u;
  CUDA_CHECK(fixture.result_repulsion.download(repulsion.data(), repulsion.size(), stream));
  CUDA_CHECK(fixture.result_atm.download(atm.data(), atm.size(), stream));
  CUDA_CHECK(fixture.plan_error.download(&plan_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(repulsion[0] == kSentinel && repulsion[1] == kSentinel);
  CHECK(atm[0] == kSentinel && atm[1] == kSentinel);
  CHECK(plan_error ==
        static_cast<std::uint32_t>(Gfn2TerminalClassicalEnergyPlanError::kInvalidRequestedMask));

  /* Role, provenance, and range corruption are synchronous plan errors: no
   * repulsion launch or diagnostic reset may precede their rejection. */
  const std::array<std::uint8_t, 2> valid_requested{1u, 1u};
  CUDA_CHECK(fixture.requested.upload(valid_requested, stream));
  auto invalid_plan = fixture.plan;
  invalid_plan.d4_cache.atm_pairs.role = Gfn2PairListRole::kD4TwoBody;
  CHECK(evaluate_gfn2_terminal_classical_energy_cuda(
            invalid_plan, fixture.activity, fixture.results, fixture.workspace, fixture.diagnostics,
            stream) == cudaErrorInvalidValue);
  invalid_plan = fixture.plan;
  invalid_plan.d4_cache.atm_pairs.plan_token ^= 1u;
  CHECK(evaluate_gfn2_terminal_classical_energy_cuda(
            invalid_plan, fixture.activity, fixture.results, fixture.workspace, fixture.diagnostics,
            stream) == cudaErrorInvalidValue);
  invalid_plan = fixture.plan;
  invalid_plan.d4_cache.coordination_generations = fixture.pair_generations.get();
  CHECK(evaluate_gfn2_terminal_classical_energy_cuda(
            invalid_plan, fixture.activity, fixture.results, fixture.workspace, fixture.diagnostics,
            stream) == cudaErrorInvalidValue);
  invalid_plan = fixture.plan;
  invalid_plan.d4_cache.atm_pairs.pairs =
      reinterpret_cast<const Gfn2AtomPair*>(fixture.result_repulsion.get());
  CHECK(evaluate_gfn2_terminal_classical_energy_cuda(
            invalid_plan, fixture.activity, fixture.results, fixture.workspace, fixture.diagnostics,
            stream) == cudaErrorInvalidValue);

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_terminal_graph_epoch_replay() {
  TerminalHostData host;
  CHECK(host.initialize());
  CHECK(std::isfinite(host.expected_atm[0]) && host.expected_atm[0] != 0.0);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  TerminalDeviceFixture fixture;
  CHECK(fixture.initialize(host, stream));
  fixture.enable_d4(true);
  const std::array<std::uint8_t, 2> requested{1u, 1u};
  CUDA_CHECK(fixture.requested.upload(requested, stream));
  CUDA_CHECK(fixture.reset_results(stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CUDA_CHECK(evaluate_gfn2_terminal_classical_energy_cuda(fixture.plan, fixture.activity,
                                                          fixture.results, fixture.workspace,
                                                          fixture.diagnostics, stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  std::array<double, 2> repulsion{};
  std::array<double, 2> atm{};
  CUDA_CHECK(fixture.result_repulsion.download(repulsion.data(), repulsion.size(), stream));
  CUDA_CHECK(fixture.result_atm.download(atm.data(), atm.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(near(repulsion[0], host.expected_repulsion[0]));
  CHECK(near(repulsion[1], host.expected_repulsion[1]));
  CHECK(near(atm[0], host.expected_atm[0], 2.0e-11));
  CHECK(near(atm[1], host.expected_atm[1], 2.0e-11));

  const std::array<std::uint64_t, 1> next_epoch{8u};
  const std::array<std::uint64_t, 2> mixed_generations{8u, 7u};
  CUDA_CHECK(fixture.epoch.upload(next_epoch, stream));
  CUDA_CHECK(fixture.committed_generations.upload(mixed_generations, stream));
  CUDA_CHECK(fixture.pair_generations.upload(mixed_generations, stream));
  CUDA_CHECK(fixture.reset_results(stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(fixture.result_repulsion.download(repulsion.data(), repulsion.size(), stream));
  CUDA_CHECK(fixture.result_atm.download(atm.data(), atm.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(near(repulsion[0], host.expected_repulsion[0]));
  CHECK(repulsion[1] == kSentinel);
  CHECK(near(atm[0], host.expected_atm[0], 2.0e-11));
  CHECK(atm[1] == kSentinel);

  const std::array<std::uint64_t, 2> next_generations{8u, 8u};
  CUDA_CHECK(fixture.committed_generations.upload(next_generations, stream));
  CUDA_CHECK(fixture.pair_generations.upload(next_generations, stream));
  CUDA_CHECK(fixture.reset_results(stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(fixture.result_repulsion.download(repulsion.data(), repulsion.size(), stream));
  CUDA_CHECK(fixture.result_atm.download(atm.data(), atm.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(near(repulsion[0], host.expected_repulsion[0]));
  CHECK(near(repulsion[1], host.expected_repulsion[1]));
  CHECK(near(atm[0], host.expected_atm[0], 2.0e-11));
  CHECK(near(atm[1], host.expected_atm[1], 2.0e-11));

  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

struct PublicationFixture {
  static constexpr std::int64_t kBatch = 6;
  static constexpr std::int64_t kAtoms = 7;
  static constexpr std::int64_t kPoints = 3;

  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> point_offsets;
  DeviceBuffer<std::uint64_t> epoch;
  DeviceBuffer<std::uint64_t> generations;
  DeviceBuffer<std::uint8_t> eligible;
  DeviceBuffer<std::uint64_t> iterations;
  DeviceBuffer<std::uint8_t> converged;
  DeviceBuffer<xtbloom_status_t> statuses;
  DeviceBuffer<double> energies;
  DeviceBuffer<double> qm_forces;
  DeviceBuffer<double> charges;
  DeviceBuffer<double> point_forces;
  DeviceBuffer<std::uint32_t> terminal_system_errors;
  DeviceBuffer<std::uint32_t> terminal_plan_error;
  DeviceBuffer<std::uint32_t> execution_system_errors;
  DeviceBuffer<std::uint32_t> execution_plan_error;
  DeviceBuffer<double> public_energies;
  DeviceBuffer<double> public_qm_forces;
  DeviceBuffer<double> public_charges;
  DeviceBuffer<double> public_point_forces;
  DeviceBuffer<std::int32_t> public_iterations;
  DeviceBuffer<std::uint8_t> public_converged;
  DeviceBuffer<xtbloom_status_t> public_statuses;
  DeviceBuffer<std::uint64_t> epoch_snapshot;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> plan_error;

  Gfn2InferencePublicationDevicePlan plan;
  Gfn2InferencePublicationDeviceInput input;
  Gfn2InferencePublicationDeviceResults results;
  Gfn2InferencePublicationDeviceWorkspace workspace;
  Gfn2InferencePublicationDeviceDiagnostics diagnostics;

  std::vector<std::int64_t> host_atom_offsets{0, 2, 3, 4, 5, 6, 7};
  std::vector<std::int64_t> host_point_offsets{0, 1, 1, 2, 2, 3, 3};
  std::vector<double> host_energies{1.0, 2.0, 3.0,
                                    4.0, 5.0, std::numeric_limits<double>::quiet_NaN()};
  std::vector<double> host_qm_forces;
  std::vector<double> host_charges;
  std::vector<double> host_point_forces;

  bool initialize(cudaStream_t stream) {
    host_qm_forces.resize(3u * static_cast<std::size_t>(kAtoms));
    host_charges.resize(static_cast<std::size_t>(kAtoms));
    host_point_forces.resize(3u * static_cast<std::size_t>(kPoints));
    for (std::size_t index = 0u; index < host_qm_forces.size(); ++index) {
      host_qm_forces[index] = 0.01 * static_cast<double>(index + 1u);
    }
    for (std::size_t index = 0u; index < host_charges.size(); ++index) {
      host_charges[index] = -0.2 + 0.03 * static_cast<double>(index);
    }
    for (std::size_t index = 0u; index < host_point_forces.size(); ++index) {
      host_point_forces[index] = -0.04 * static_cast<double>(index + 1u);
    }
    const bool allocated =
        atom_offsets.allocate(host_atom_offsets.size()) == cudaSuccess &&
        point_offsets.allocate(host_point_offsets.size()) == cudaSuccess &&
        epoch.allocate(1) == cudaSuccess && generations.allocate(kBatch) == cudaSuccess &&
        eligible.allocate(kBatch) == cudaSuccess && iterations.allocate(kBatch) == cudaSuccess &&
        converged.allocate(kBatch) == cudaSuccess && statuses.allocate(kBatch) == cudaSuccess &&
        energies.allocate(kBatch) == cudaSuccess && qm_forces.allocate(3 * kAtoms) == cudaSuccess &&
        charges.allocate(kAtoms) == cudaSuccess &&
        point_forces.allocate(3 * kPoints) == cudaSuccess &&
        terminal_system_errors.allocate(kBatch) == cudaSuccess &&
        terminal_plan_error.allocate(1) == cudaSuccess &&
        execution_system_errors.allocate(kBatch) == cudaSuccess &&
        execution_plan_error.allocate(1) == cudaSuccess &&
        public_energies.allocate(kBatch) == cudaSuccess &&
        public_qm_forces.allocate(3 * kAtoms) == cudaSuccess &&
        public_charges.allocate(kAtoms) == cudaSuccess &&
        public_point_forces.allocate(3 * kPoints) == cudaSuccess &&
        public_iterations.allocate(kBatch) == cudaSuccess &&
        public_converged.allocate(kBatch) == cudaSuccess &&
        public_statuses.allocate(kBatch) == cudaSuccess &&
        epoch_snapshot.allocate(1) == cudaSuccess &&
        system_errors.allocate(kBatch) == cudaSuccess && plan_error.allocate(1) == cudaSuccess;
    if (!allocated) return false;

    const std::array<std::uint64_t, 1> host_epoch{9u};
    const std::array<std::uint64_t, 6> host_generations{9u, 9u, 9u, 9u, 8u, 9u};
    const std::array<std::uint8_t, 6> host_eligible{1u, 1u, 1u, 0u, 1u, 1u};
    const std::array<std::uint64_t, 6> host_iterations{2u, 5u, 3u, 77u, 66u, 4u};
    const std::array<std::uint8_t, 6> host_converged{1u, 0u, 0u, 1u, 1u, 1u};
    const std::array<xtbloom_status_t, 6> host_statuses{
        XTBLOOM_STATUS_SUCCESS, XTBLOOM_STATUS_SCC_NOT_CONVERGED, XTBLOOM_STATUS_EIGENSOLVER_FAILED,
        XTBLOOM_STATUS_SUCCESS, XTBLOOM_STATUS_SUCCESS,           XTBLOOM_STATUS_SUCCESS};
    const std::array<std::uint32_t, 6> zero_system_errors{};
    const std::array<std::uint32_t, 1> zero_plan_error{};
    if (atom_offsets.upload(host_atom_offsets, stream) != cudaSuccess ||
        point_offsets.upload(host_point_offsets, stream) != cudaSuccess ||
        epoch.upload(host_epoch, stream) != cudaSuccess ||
        generations.upload(host_generations, stream) != cudaSuccess ||
        eligible.upload(host_eligible, stream) != cudaSuccess ||
        iterations.upload(host_iterations, stream) != cudaSuccess ||
        converged.upload(host_converged, stream) != cudaSuccess ||
        statuses.upload(host_statuses, stream) != cudaSuccess ||
        energies.upload(host_energies, stream) != cudaSuccess ||
        qm_forces.upload(host_qm_forces, stream) != cudaSuccess ||
        charges.upload(host_charges, stream) != cudaSuccess ||
        point_forces.upload(host_point_forces, stream) != cudaSuccess ||
        terminal_system_errors.upload(zero_system_errors, stream) != cudaSuccess ||
        terminal_plan_error.upload(zero_plan_error, stream) != cudaSuccess ||
        execution_system_errors.upload(zero_system_errors, stream) != cudaSuccess ||
        execution_plan_error.upload(zero_plan_error, stream) != cudaSuccess) {
      return false;
    }

    constexpr std::uint64_t token = 0x123125ULL;
    plan.requested_properties = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES |
                                XTBLOOM_COMPUTE_ATOMIC_CHARGES |
                                XTBLOOM_COMPUTE_POINT_CHARGE_FORCES;
    plan.plan_token = token;
    plan.maximum_iterations = 5u;
    plan.batch_size = kBatch;
    plan.total_atoms = kAtoms;
    plan.total_point_charges = kPoints;
    plan.atom_offsets = atom_offsets.get();
    plan.point_charge_offsets = point_offsets.get();
    plan.geometry_epoch = {epoch.get(), 1, token};
    plan.committed_generations = generations.get();
    plan.generation_elements = kBatch;
    input = {eligible.get(),
             kBatch,
             iterations.get(),
             converged.get(),
             statuses.get(),
             kBatch,
             energies.get(),
             kBatch,
             qm_forces.get(),
             3 * kAtoms,
             charges.get(),
             kAtoms,
             point_forces.get(),
             3 * kPoints,
             terminal_system_errors.get(),
             kBatch,
             terminal_plan_error.get(),
             execution_system_errors.get(),
             kBatch,
             execution_plan_error.get(),
             token};
    results = {public_energies.get(),
               kBatch,
               public_qm_forces.get(),
               3 * kAtoms,
               public_charges.get(),
               kAtoms,
               public_point_forces.get(),
               3 * kPoints,
               public_iterations.get(),
               public_converged.get(),
               public_statuses.get(),
               kBatch,
               token};
    workspace = {epoch_snapshot.get(), 1, token};
    diagnostics = {system_errors.get(), kBatch, plan_error.get(), 1, token};
    return cudaStreamSynchronize(stream) == cudaSuccess;
  }

  cudaError_t reset_results(cudaStream_t stream) {
    cudaError_t status = public_energies.fill(kSentinel, stream);
    if (status == cudaSuccess) status = public_qm_forces.fill(kSentinel, stream);
    if (status == cudaSuccess) status = public_charges.fill(kSentinel, stream);
    if (status == cudaSuccess) status = public_point_forces.fill(kSentinel, stream);
    if (status == cudaSuccess) status = public_iterations.fill(-73, stream);
    if (status == cudaSuccess) status = public_converged.fill(7u, stream);
    if (status == cudaSuccess)
      status = public_statuses.fill(XTBLOOM_STATUS_INVALID_ARGUMENT, stream);
    return status;
  }
};

bool floating_slice_is_nan(const std::vector<double>& values, std::int64_t begin, std::int64_t end,
                           std::int64_t width) {
  for (std::int64_t index = width * begin; index < width * end; ++index) {
    if (!std::isnan(values[static_cast<std::size_t>(index)])) return false;
  }
  return true;
}

int test_inference_publication_semantics() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  PublicationFixture fixture;
  CHECK(fixture.initialize(stream));
  CUDA_CHECK(fixture.reset_results(stream));
  CUDA_CHECK(publish_gfn2_inference_results_cuda(fixture.plan, fixture.input, fixture.results,
                                                 fixture.workspace, fixture.diagnostics, stream));

  std::vector<double> energies(static_cast<std::size_t>(fixture.kBatch));
  std::vector<double> forces(static_cast<std::size_t>(3 * fixture.kAtoms));
  std::vector<double> charges(static_cast<std::size_t>(fixture.kAtoms));
  std::vector<double> point_forces(static_cast<std::size_t>(3 * fixture.kPoints));
  std::vector<std::int32_t> iterations(static_cast<std::size_t>(fixture.kBatch));
  std::vector<std::uint8_t> converged(static_cast<std::size_t>(fixture.kBatch));
  std::vector<xtbloom_status_t> statuses(static_cast<std::size_t>(fixture.kBatch));
  std::vector<std::uint32_t> errors(static_cast<std::size_t>(fixture.kBatch));
  CUDA_CHECK(fixture.public_energies.download(energies.data(), energies.size(), stream));
  CUDA_CHECK(fixture.public_qm_forces.download(forces.data(), forces.size(), stream));
  CUDA_CHECK(fixture.public_charges.download(charges.data(), charges.size(), stream));
  CUDA_CHECK(
      fixture.public_point_forces.download(point_forces.data(), point_forces.size(), stream));
  CUDA_CHECK(fixture.public_iterations.download(iterations.data(), iterations.size(), stream));
  CUDA_CHECK(fixture.public_converged.download(converged.data(), converged.size(), stream));
  CUDA_CHECK(fixture.public_statuses.download(statuses.data(), statuses.size(), stream));
  CUDA_CHECK(fixture.system_errors.download(errors.data(), errors.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CHECK(energies[0] == fixture.host_energies[0]);
  CHECK(iterations[0] == 2 && converged[0] == 1u && statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(near(forces[0], fixture.host_qm_forces[0]));
  CHECK(near(charges[0], fixture.host_charges[0]));
  CHECK(near(point_forces[0], fixture.host_point_forces[0]));

  CHECK(std::isnan(energies[1]) && iterations[1] == 5 && converged[1] == 0u &&
        statuses[1] == XTBLOOM_STATUS_SCC_NOT_CONVERGED);
  CHECK(std::isnan(energies[2]) && iterations[2] == 3 && converged[2] == 0u &&
        statuses[2] == XTBLOOM_STATUS_EIGENSOLVER_FAILED);
  CHECK(std::isnan(energies[3]) && iterations[3] == 0 && converged[3] == 0u &&
        statuses[3] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(errors[3] == static_cast<std::uint32_t>(
                         Gfn2InferencePublicationSystemError::kIneligibleNumericalRefresh));
  CHECK(std::isnan(energies[4]) && iterations[4] == 0 && converged[4] == 0u &&
        statuses[4] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(errors[4] ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationSystemError::kStaleGeneration));
  CHECK(std::isnan(energies[5]) && iterations[5] == 4 && converged[5] == 0u &&
        statuses[5] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(errors[5] ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationSystemError::kNonfiniteResult));

  for (std::int64_t system = 1; system < fixture.kBatch; ++system) {
    CHECK(floating_slice_is_nan(forces, fixture.host_atom_offsets[system],
                                fixture.host_atom_offsets[system + 1], 3));
    CHECK(floating_slice_is_nan(charges, fixture.host_atom_offsets[system],
                                fixture.host_atom_offsets[system + 1], 1));
    CHECK(floating_slice_is_nan(point_forces, fixture.host_point_offsets[system],
                                fixture.host_point_offsets[system + 1], 3));
  }

  /* Malformed eligibility is plan-wide and leaves all public bytes untouched. */
  const std::array<std::uint8_t, 6> invalid_eligible{1u, 1u, 2u, 1u, 1u, 1u};
  CUDA_CHECK(fixture.eligible.upload(invalid_eligible, stream));
  CUDA_CHECK(fixture.reset_results(stream));
  std::vector<double> before_energies(static_cast<std::size_t>(fixture.kBatch), kSentinel);
  std::vector<double> before_forces(static_cast<std::size_t>(3 * fixture.kAtoms), kSentinel);
  std::vector<double> before_charges(static_cast<std::size_t>(fixture.kAtoms), kSentinel);
  std::vector<double> before_point_forces(static_cast<std::size_t>(3 * fixture.kPoints), kSentinel);
  std::vector<std::int32_t> before_iterations(static_cast<std::size_t>(fixture.kBatch), -73);
  std::vector<std::uint8_t> before_converged(static_cast<std::size_t>(fixture.kBatch), 7u);
  std::vector<xtbloom_status_t> before_statuses(static_cast<std::size_t>(fixture.kBatch),
                                                XTBLOOM_STATUS_INVALID_ARGUMENT);
  CUDA_CHECK(publish_gfn2_inference_results_cuda(fixture.plan, fixture.input, fixture.results,
                                                 fixture.workspace, fixture.diagnostics, stream));
  std::uint32_t plan_error = 0u;
  CUDA_CHECK(fixture.public_energies.download(energies.data(), energies.size(), stream));
  CUDA_CHECK(fixture.public_qm_forces.download(forces.data(), forces.size(), stream));
  CUDA_CHECK(fixture.public_charges.download(charges.data(), charges.size(), stream));
  CUDA_CHECK(
      fixture.public_point_forces.download(point_forces.data(), point_forces.size(), stream));
  CUDA_CHECK(fixture.public_iterations.download(iterations.data(), iterations.size(), stream));
  CUDA_CHECK(fixture.public_converged.download(converged.data(), converged.size(), stream));
  CUDA_CHECK(fixture.public_statuses.download(statuses.data(), statuses.size(), stream));
  CUDA_CHECK(fixture.plan_error.download(&plan_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(byte_equal(energies, before_energies));
  CHECK(byte_equal(forces, before_forces));
  CHECK(byte_equal(charges, before_charges));
  CHECK(byte_equal(point_forces, before_point_forces));
  CHECK(byte_equal(iterations, before_iterations));
  CHECK(byte_equal(converged, before_converged));
  CHECK(byte_equal(statuses, before_statuses));
  CHECK(plan_error ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationPlanError::kInvalidEligibilityMask));

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_inference_publication_graph_epoch_replay() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  PublicationFixture fixture;
  CHECK(fixture.initialize(stream));
  const std::array<std::uint8_t, 6> eligible{1u, 1u, 1u, 1u, 1u, 1u};
  const std::array<std::uint8_t, 6> converged{1u, 1u, 1u, 1u, 1u, 1u};
  const std::array<xtbloom_status_t, 6> statuses{XTBLOOM_STATUS_SUCCESS, XTBLOOM_STATUS_SUCCESS,
                                                 XTBLOOM_STATUS_SUCCESS, XTBLOOM_STATUS_SUCCESS,
                                                 XTBLOOM_STATUS_SUCCESS, XTBLOOM_STATUS_SUCCESS};
  const std::array<std::uint64_t, 6> iterations{1u, 2u, 3u, 4u, 5u, 1u};
  const std::array<std::uint64_t, 6> generations{9u, 9u, 9u, 9u, 9u, 9u};
  std::vector<double> finite_energies{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  CUDA_CHECK(fixture.eligible.upload(eligible, stream));
  CUDA_CHECK(fixture.converged.upload(converged, stream));
  CUDA_CHECK(fixture.statuses.upload(statuses, stream));
  CUDA_CHECK(fixture.iterations.upload(iterations, stream));
  CUDA_CHECK(fixture.generations.upload(generations, stream));
  CUDA_CHECK(fixture.energies.upload(finite_energies, stream));
  CUDA_CHECK(fixture.reset_results(stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CUDA_CHECK(publish_gfn2_inference_results_cuda(fixture.plan, fixture.input, fixture.results,
                                                 fixture.workspace, fixture.diagnostics, stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  std::vector<std::int32_t> public_iterations(static_cast<std::size_t>(fixture.kBatch));
  CUDA_CHECK(fixture.public_iterations.download(public_iterations.data(), public_iterations.size(),
                                                stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(public_iterations[4] == 5);

  const std::array<std::uint64_t, 1> next_epoch{10u};
  const std::array<std::uint64_t, 6> stale_generations{10u, 10u, 10u, 10u, 9u, 10u};
  const std::array<std::uint64_t, 6> stale_iterations{1u, 2u, 3u, 4u, 77u, 1u};
  CUDA_CHECK(fixture.epoch.upload(next_epoch, stream));
  CUDA_CHECK(fixture.generations.upload(stale_generations, stream));
  CUDA_CHECK(fixture.iterations.upload(stale_iterations, stream));
  CUDA_CHECK(fixture.reset_results(stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  std::vector<double> public_energies(static_cast<std::size_t>(fixture.kBatch));
  CUDA_CHECK(fixture.public_iterations.download(public_iterations.data(), public_iterations.size(),
                                                stream));
  CUDA_CHECK(
      fixture.public_energies.download(public_energies.data(), public_energies.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(public_iterations[4] == 0 && std::isnan(public_energies[4]));

  const std::array<std::uint64_t, 6> next_generations{10u, 10u, 10u, 10u, 10u, 10u};
  const std::array<std::uint64_t, 6> next_iterations{1u, 2u, 3u, 4u, 5u, 1u};
  finite_energies[4] = 12.5;
  CUDA_CHECK(fixture.generations.upload(next_generations, stream));
  CUDA_CHECK(fixture.iterations.upload(next_iterations, stream));
  CUDA_CHECK(fixture.energies.upload(finite_energies, stream));
  CUDA_CHECK(fixture.reset_results(stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(fixture.public_iterations.download(public_iterations.data(), public_iterations.size(),
                                                stream));
  CUDA_CHECK(
      fixture.public_energies.download(public_energies.data(), public_energies.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(public_iterations[4] == 5 && public_energies[4] == 12.5);

  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::fprintf(stderr, "CUDA device required\n");
    return 77;
  }
  int device = 0;
  std::string parameter_error;
  if (cudaGetDevice(&device) != cudaSuccess ||
      !xtbloom::detail::ensure_cuda_gfn2_parameters(device, parameter_error)) {
    std::fprintf(stderr, "GFN2 parameter upload failed: %s\n", parameter_error.c_str());
    return 78;
  }
  if (const int status = test_terminal_base_d4_and_rollbacks(); status != 0) return status;
  if (const int status = test_terminal_graph_epoch_replay(); status != 0) return status;
  if (const int status = test_inference_publication_semantics(); status != 0) return status;
  return test_inference_publication_graph_epoch_replay();
}
