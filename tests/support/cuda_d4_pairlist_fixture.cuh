#ifndef XTBLOOM_TESTS_SUPPORT_CUDA_D4_PAIRLIST_FIXTURE_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_TESTS_SUPPORT_CUDA_D4_PAIRLIST_FIXTURE_CUH

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "backends/common/gfn2_plan_schema.hpp"
#include "backends/cuda/gfn2_d4.cuh"

namespace xtbloom::test::cuda {

/*
 * Standalone SCC tests do not execute the production preprocessing owner, but
 * D4 consumers must still receive the same committed pair-list contract. This
 * fixture materializes one fixed-capacity 50-bohr superset from the host test
 * geometry, uploads it once, and projects the canonical 30/50/25-bohr roles.
 * Padding is never published: pair_counts/neighbor_counts name only the live
 * prefixes, including zero pairs for singleton systems.
 */
class D4CommittedPairListFixture {
 public:
  D4CommittedPairListFixture() = default;
  D4CommittedPairListFixture(const D4CommittedPairListFixture&) = delete;
  D4CommittedPairListFixture& operator=(const D4CommittedPairListFixture&) = delete;

  ~D4CommittedPairListFixture() { reset(); }

  bool bind(const std::vector<std::int64_t>& atom_offsets, const std::vector<double>& positions,
            const detail::Gfn2RaggedTopologyView& device_topology, const double* device_positions,
            double* device_coordination, std::uint64_t generation,
            detail::cuda::Gfn2D4PairListDeviceCache& cache, cudaStream_t stream) noexcept {
    reset();
    const std::int64_t batch = device_topology.batch_size;
    const std::int64_t atoms = device_topology.total_atoms;
    if (batch <= 0 || atoms <= 0 || generation == 0u || device_positions == nullptr ||
        device_coordination == nullptr ||
        atom_offsets.size() != static_cast<std::size_t>(batch + 1) ||
        positions.size() != static_cast<std::size_t>(atoms * 3) || atom_offsets.front() != 0 ||
        atom_offsets.back() != atoms) {
      return false;
    }

    std::int64_t maximum_atoms = 0;
    for (std::int64_t system = 0; system < batch; ++system) {
      const std::int64_t begin = atom_offsets[static_cast<std::size_t>(system)];
      const std::int64_t end = atom_offsets[static_cast<std::size_t>(system + 1)];
      if (begin < 0 || end < begin || end > atoms) return false;
      maximum_atoms = std::max(maximum_atoms, end - begin);
    }
    if (maximum_atoms > 3037000499LL) return false;
    max_pairs_per_system_ = std::max<std::int64_t>(1, maximum_atoms * (maximum_atoms - 1) / 2);
    max_neighbors_per_atom_ = std::max<std::int64_t>(1, maximum_atoms);
    if (batch > std::numeric_limits<std::int64_t>::max() / max_pairs_per_system_ ||
        atoms > std::numeric_limits<std::int64_t>::max() / max_neighbors_per_atom_) {
      return false;
    }

    std::vector<std::int64_t> pair_offsets(static_cast<std::size_t>(batch + 1));
    std::vector<detail::Gfn2AtomPair> pairs(
        static_cast<std::size_t>(batch * max_pairs_per_system_));
    std::vector<std::int64_t> pair_counts(static_cast<std::size_t>(batch), 0);
    std::vector<std::int64_t> neighbor_offsets(static_cast<std::size_t>(atoms + 1));
    std::vector<std::int64_t> neighbors(static_cast<std::size_t>(atoms * max_neighbors_per_atom_));
    std::vector<std::int64_t> neighbor_counts(static_cast<std::size_t>(atoms), 0);
    std::vector<std::vector<std::int64_t>> neighbor_lists(static_cast<std::size_t>(atoms));

    for (std::int64_t system = 0; system <= batch; ++system) {
      pair_offsets[static_cast<std::size_t>(system)] = system * max_pairs_per_system_;
    }
    for (std::int64_t atom = 0; atom <= atoms; ++atom) {
      neighbor_offsets[static_cast<std::size_t>(atom)] = atom * max_neighbors_per_atom_;
    }

    constexpr double kBuilderCutoffSquared = 50.0 * 50.0;
    for (std::int64_t system = 0; system < batch; ++system) {
      const std::int64_t atom_begin = atom_offsets[static_cast<std::size_t>(system)];
      const std::int64_t atom_end = atom_offsets[static_cast<std::size_t>(system + 1)];
      const std::int64_t pair_begin = pair_offsets[static_cast<std::size_t>(system)];
      std::int64_t count = 0;
      /* Match the production packed-pair order exactly: second ascends first,
       * then first ascends within that column. D4 reductions depend on this
       * deterministic order for CPU/CUDA comparison. */
      for (std::int64_t second = atom_begin + 1; second < atom_end; ++second) {
        for (std::int64_t first = atom_begin; first < second; ++first) {
          double distance_squared = 0.0;
          for (int axis = 0; axis < 3; ++axis) {
            const double delta = positions[static_cast<std::size_t>(second * 3 + axis)] -
                                 positions[static_cast<std::size_t>(first * 3 + axis)];
            distance_squared += delta * delta;
          }
          if (!std::isfinite(distance_squared) || distance_squared > kBuilderCutoffSquared) {
            continue;
          }
          if (count >= max_pairs_per_system_) return false;
          pairs[static_cast<std::size_t>(pair_begin + count)] = {first, second};
          ++count;
          neighbor_lists[static_cast<std::size_t>(first)].push_back(second);
          neighbor_lists[static_cast<std::size_t>(second)].push_back(first);
        }
      }
      pair_counts[static_cast<std::size_t>(system)] = count;
    }
    for (std::int64_t atom = 0; atom < atoms; ++atom) {
      auto& list = neighbor_lists[static_cast<std::size_t>(atom)];
      std::sort(list.begin(), list.end());
      if (list.size() > static_cast<std::size_t>(max_neighbors_per_atom_)) return false;
      const std::int64_t begin = neighbor_offsets[static_cast<std::size_t>(atom)];
      std::copy(list.begin(), list.end(), neighbors.begin() + begin);
      neighbor_counts[static_cast<std::size_t>(atom)] = static_cast<std::int64_t>(list.size());
    }

    const std::vector<std::uint64_t> generations(static_cast<std::size_t>(batch), generation);
    const std::vector<std::uint8_t> eligible(static_cast<std::size_t>(batch), 1u);
    if (!upload(pair_offsets_, pair_offsets, stream) || !upload(pairs_, pairs, stream) ||
        !upload(pair_counts_, pair_counts, stream) ||
        !upload(neighbor_offsets_, neighbor_offsets, stream) ||
        !upload(neighbors_, neighbors, stream) ||
        !upload(neighbor_counts_, neighbor_counts, stream) ||
        !upload(generations_, generations, stream) || !upload(eligible_, eligible, stream) ||
        !upload(coordination_generations_, generations, stream) ||
        !upload(coordination_eligible_, eligible, stream)) {
      reset();
      return false;
    }

    detail::Gfn2PairListConsumerView coordination{};
    coordination.memory_space = detail::Gfn2PlanMemorySpace::kCudaDevice;
    coordination.state = detail::Gfn2PairListState::kCommitted;
    coordination.role = detail::Gfn2PairListRole::kD4Coordination;
    coordination.pair_map_kind = detail::Gfn2PairMapKind::kExplicit;
    coordination.plan_token = device_topology.plan_token;
    coordination.cutoff_bohr = detail::cuda::kGfn2D4CoordinationCutoffBohr;
    coordination.list_builder_cutoff_bohr = 50.0;
    coordination.batch_size = batch;
    coordination.total_atoms = atoms;
    coordination.max_pairs_per_system = max_pairs_per_system_;
    coordination.max_neighbors_per_atom = max_neighbors_per_atom_;
    coordination.pair_offset_count = batch + 1;
    coordination.neighbor_offset_count = atoms + 1;
    coordination.pair_count = batch * max_pairs_per_system_;
    coordination.neighbor_count = atoms * max_neighbors_per_atom_;
    coordination.pair_offsets = pair_offsets_;
    coordination.pairs = pairs_;
    coordination.pair_count_elements = batch;
    coordination.neighbor_count_elements = atoms;
    coordination.pair_counts = pair_counts_;
    coordination.neighbor_counts = neighbor_counts_;
    coordination.neighbor_offsets = neighbor_offsets_;
    coordination.neighbors = neighbors_;
    coordination.committed_generation_count = batch;
    coordination.eligible_mask_count = batch;
    coordination.active_mask_count = 0;
    coordination.committed_generations = generations_;
    coordination.eligible_mask = eligible_;
    coordination.active_mask = nullptr;

    detail::Gfn2PairListConsumerView two_body{};
    detail::Gfn2PairListConsumerView atm{};
    if (detail::project_gfn2_pair_list_role_binding(
            device_topology, coordination, detail::Gfn2PairListRole::kD4TwoBody,
            detail::Gfn2PlanMemorySpace::kCudaDevice, two_body)
                .error != detail::Gfn2PlanSchemaError::kSuccess ||
        detail::project_gfn2_pair_list_role_binding(device_topology, coordination,
                                                    detail::Gfn2PairListRole::kD4Atm,
                                                    detail::Gfn2PlanMemorySpace::kCudaDevice, atm)
                .error != detail::Gfn2PlanSchemaError::kSuccess) {
      reset();
      return false;
    }
    cache = {};
    cache.positions = device_positions;
    cache.position_elements = atoms * 3;
    cache.coordination_numbers = device_coordination;
    cache.coordination_elements = atoms;
    cache.coordination_generations = coordination_generations_;
    cache.coordination_generation_elements = batch;
    cache.coordination_eligible_mask = coordination_eligible_;
    cache.coordination_eligible_elements = batch;
    cache.coordination_pairs = coordination;
    cache.two_body_pairs = two_body;
    cache.atm_pairs = atm;
    cache.plan_token = device_topology.plan_token;
    return true;
  }

 private:
  template <typename T>
  static bool upload(T*& destination, const std::vector<T>& source, cudaStream_t stream) noexcept {
    /* bind() guarantees positive batch/atom capacities, so an empty structural
     * array means the fixture invariant was violated and must not be hidden. */
    if (source.empty() || cudaMalloc(reinterpret_cast<void**>(&destination),
                                     source.size() * sizeof(T)) != cudaSuccess) {
      return false;
    }
    return cudaMemcpyAsync(destination, source.data(), source.size() * sizeof(T),
                           cudaMemcpyHostToDevice, stream) == cudaSuccess;
  }

  template <typename T>
  static void release(T*& pointer) noexcept {
    if (pointer != nullptr) (void)cudaFree(pointer);
    pointer = nullptr;
  }

  void reset() noexcept {
    release(pair_offsets_);
    release(pairs_);
    release(pair_counts_);
    release(neighbor_offsets_);
    release(neighbors_);
    release(neighbor_counts_);
    release(generations_);
    release(eligible_);
    release(coordination_generations_);
    release(coordination_eligible_);
    max_pairs_per_system_ = 0;
    max_neighbors_per_atom_ = 0;
  }

  std::int64_t* pair_offsets_ = nullptr;
  detail::Gfn2AtomPair* pairs_ = nullptr;
  std::int64_t* pair_counts_ = nullptr;
  std::int64_t* neighbor_offsets_ = nullptr;
  std::int64_t* neighbors_ = nullptr;
  std::int64_t* neighbor_counts_ = nullptr;
  std::uint64_t* generations_ = nullptr;
  std::uint8_t* eligible_ = nullptr;
  /* CN publication provenance is a distinct semantic array from pair-list
   * publication provenance; production validators reject aliasing them. */
  std::uint64_t* coordination_generations_ = nullptr;
  std::uint8_t* coordination_eligible_ = nullptr;
  std::int64_t max_pairs_per_system_ = 0;
  std::int64_t max_neighbors_per_atom_ = 0;
};

}  // namespace xtbloom::test::cuda

#endif  // XTBLOOM_TESTS_SUPPORT_CUDA_D4_PAIRLIST_FIXTURE_CUH
