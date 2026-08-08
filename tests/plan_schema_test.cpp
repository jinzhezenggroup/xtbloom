#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

#include "backends/common/gfn2_plan_schema.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

using gpuxtb::detail::bind_gfn2_topology_host;
using gpuxtb::detail::bind_gfn2_wavefunction_layout_host;
using gpuxtb::detail::gfn2_element_identity_fingerprint_host;
using gpuxtb::detail::Gfn2AOBucketProjectionView;
using gpuxtb::detail::Gfn2AOMatrixProjectionView;
using gpuxtb::detail::Gfn2AtomPair;
using gpuxtb::detail::Gfn2AtomProjectionView;
using gpuxtb::detail::Gfn2ElementIdentityProjectionView;
using gpuxtb::detail::Gfn2GenerationScope;
using gpuxtb::detail::Gfn2GeometryCacheProvenanceView;
using gpuxtb::detail::Gfn2PackedAllPairProjectionView;
using gpuxtb::detail::Gfn2PairListConsumerView;
using gpuxtb::detail::Gfn2PairListRole;
using gpuxtb::detail::Gfn2PairListState;
using gpuxtb::detail::Gfn2PairMapKind;
using gpuxtb::detail::Gfn2PlanMemorySpace;
using gpuxtb::detail::Gfn2PlanSchemaDiagnostic;
using gpuxtb::detail::Gfn2PlanSchemaError;
using gpuxtb::detail::Gfn2PlanSchemaField;
using gpuxtb::detail::Gfn2RaggedTopologyView;
using gpuxtb::detail::Gfn2ShellOwnershipProjectionView;
using gpuxtb::detail::Gfn2WavefunctionLayoutView;
using gpuxtb::detail::project_gfn2_ao_bucket_projection_host;
using gpuxtb::detail::project_gfn2_ao_matrix_projection_host;
using gpuxtb::detail::project_gfn2_atom_projection_host;
using gpuxtb::detail::project_gfn2_element_identity_projection_host;
using gpuxtb::detail::project_gfn2_packed_all_pair_projection_host;
using gpuxtb::detail::project_gfn2_shell_ownership_projection_host;
using gpuxtb::detail::validate_gfn2_ao_bucket_projection_binding;
using gpuxtb::detail::validate_gfn2_ao_matrix_projection_binding;
using gpuxtb::detail::validate_gfn2_atom_projection_binding;
using gpuxtb::detail::validate_gfn2_element_identity_projection_binding;
using gpuxtb::detail::validate_gfn2_geometry_provenance_host;
using gpuxtb::detail::validate_gfn2_packed_all_pair_projection_binding;
using gpuxtb::detail::validate_gfn2_pair_list_consumer_binding;
using gpuxtb::detail::validate_gfn2_pair_list_consumer_host;
using gpuxtb::detail::validate_gfn2_shell_ownership_projection_binding;
using gpuxtb::detail::validate_gfn2_topology_binding;
using gpuxtb::detail::validate_gfn2_topology_host;
using gpuxtb::detail::validate_gfn2_wavefunction_layout_binding;
using gpuxtb::detail::validate_gfn2_wavefunction_layout_host;

constexpr std::uint64_t kPlanToken = 0x6a09e667f3bcc909ULL;
constexpr std::uint64_t kGeneration = 41u;

std::int64_t triangle(std::int64_t value) {
  return (value & 1LL) == 0LL ? (value / 2LL) * (value - 1LL) : value * ((value - 1LL) / 2LL);
}

template <typename T>
const T* pointer_or_null(const std::vector<T>& values) {
  return values.empty() ? nullptr : values.data();
}

struct HostTopology {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> batch_shell_offsets;
  std::vector<std::int64_t> batch_orbital_offsets;
  std::vector<std::int64_t> matrix_offsets;
  std::vector<std::int64_t> atom_shell_offsets;
  std::vector<std::int64_t> shell_orbital_offsets;
  std::vector<std::int64_t> shell_to_atom;
  std::vector<std::int64_t> orbital_to_shell;
  std::vector<std::int64_t> orbital_to_atom;
  std::vector<std::int64_t> pair_offsets;
  std::vector<Gfn2AtomPair> atom_pairs;
  std::vector<std::int64_t> bucket_offsets;
  std::vector<std::int32_t> bucket_systems;
  std::vector<std::int32_t> bucket_orbital_counts;
  Gfn2RaggedTopologyView view{};

  void refresh_view(Gfn2PairMapKind pair_kind = Gfn2PairMapKind::kPackedLowerTriangle) {
    view.memory_space = Gfn2PlanMemorySpace::kHost;
    view.pair_map_kind = pair_kind;
    view.plan_token = kPlanToken;
    view.batch_size = static_cast<std::int64_t>(atom_offsets.size() - 1u);
    view.total_atoms = atom_offsets.back();
    view.total_shells = batch_shell_offsets.back();
    view.total_orbitals = batch_orbital_offsets.back();
    view.total_matrix_elements = matrix_offsets.back();
    view.total_pairs = pair_offsets.empty() ? 0 : pair_offsets.back();
    view.bucket_count = static_cast<std::int64_t>(bucket_orbital_counts.size());
    view.atom_offset_count = static_cast<std::int64_t>(atom_offsets.size());
    view.batch_shell_offset_count = static_cast<std::int64_t>(batch_shell_offsets.size());
    view.batch_orbital_offset_count = static_cast<std::int64_t>(batch_orbital_offsets.size());
    view.matrix_offset_count = static_cast<std::int64_t>(matrix_offsets.size());
    view.atom_shell_offset_count = static_cast<std::int64_t>(atom_shell_offsets.size());
    view.shell_orbital_offset_count = static_cast<std::int64_t>(shell_orbital_offsets.size());
    view.shell_to_atom_count = static_cast<std::int64_t>(shell_to_atom.size());
    view.orbital_to_shell_count = static_cast<std::int64_t>(orbital_to_shell.size());
    view.orbital_to_atom_count = static_cast<std::int64_t>(orbital_to_atom.size());
    view.pair_offset_count = static_cast<std::int64_t>(pair_offsets.size());
    view.atom_pair_count =
        pair_kind == Gfn2PairMapKind::kExplicit ? static_cast<std::int64_t>(atom_pairs.size()) : 0;
    view.bucket_offset_count = static_cast<std::int64_t>(bucket_offsets.size());
    view.bucket_system_count = static_cast<std::int64_t>(bucket_systems.size());
    view.bucket_orbital_count = static_cast<std::int64_t>(bucket_orbital_counts.size());
    view.atom_offsets = pointer_or_null(atom_offsets);
    view.batch_shell_offsets = pointer_or_null(batch_shell_offsets);
    view.batch_orbital_offsets = pointer_or_null(batch_orbital_offsets);
    view.matrix_offsets = pointer_or_null(matrix_offsets);
    view.atom_shell_offsets = pointer_or_null(atom_shell_offsets);
    view.shell_orbital_offsets = pointer_or_null(shell_orbital_offsets);
    view.shell_to_atom = pointer_or_null(shell_to_atom);
    view.orbital_to_shell = pointer_or_null(orbital_to_shell);
    view.orbital_to_atom = pointer_or_null(orbital_to_atom);
    view.pair_offsets = pointer_or_null(pair_offsets);
    view.atom_pairs =
        pair_kind == Gfn2PairMapKind::kExplicit ? pointer_or_null(atom_pairs) : nullptr;
    view.bucket_offsets = pointer_or_null(bucket_offsets);
    view.bucket_systems = pointer_or_null(bucket_systems);
    view.bucket_orbital_counts = pointer_or_null(bucket_orbital_counts);
  }
};

HostTopology make_topology(std::int64_t batch_size, bool explicit_pairs = false) {
  HostTopology host;
  host.atom_offsets.push_back(0);
  host.batch_shell_offsets.push_back(0);
  host.batch_orbital_offsets.push_back(0);
  host.matrix_offsets.push_back(0);
  host.pair_offsets.push_back(0);
  host.atom_shell_offsets.push_back(0);
  host.shell_orbital_offsets.push_back(0);
  std::vector<std::int32_t> system_orbitals(static_cast<std::size_t>(batch_size), 0);
  for (std::int64_t system = 0; system < batch_size; ++system) {
    const std::int64_t atom_count = batch_size == 1 ? 3 : system % 5;
    const std::int64_t atom_begin = host.atom_offsets.back();
    for (std::int64_t local_atom = 0; local_atom < atom_count; ++local_atom) {
      const std::int64_t atom = atom_begin + local_atom;
      const std::int64_t shell_count = 1 + ((system + local_atom) & 1LL);
      for (std::int64_t local_shell = 0; local_shell < shell_count; ++local_shell) {
        const std::int64_t shell = static_cast<std::int64_t>(host.shell_to_atom.size());
        host.shell_to_atom.push_back(atom);
        const std::int64_t orbital_count = 1 + 2 * ((system + local_atom + local_shell) % 3);
        for (std::int64_t local_orbital = 0; local_orbital < orbital_count; ++local_orbital) {
          static_cast<void>(local_orbital);
          host.orbital_to_shell.push_back(shell);
          host.orbital_to_atom.push_back(atom);
        }
        host.shell_orbital_offsets.push_back(
            static_cast<std::int64_t>(host.orbital_to_shell.size()));
      }
      host.atom_shell_offsets.push_back(static_cast<std::int64_t>(host.shell_to_atom.size()));
    }
    const std::int64_t atom_end = atom_begin + atom_count;
    host.atom_offsets.push_back(atom_end);
    host.batch_shell_offsets.push_back(static_cast<std::int64_t>(host.shell_to_atom.size()));
    host.batch_orbital_offsets.push_back(static_cast<std::int64_t>(host.orbital_to_shell.size()));
    const std::int64_t orbitals = host.batch_orbital_offsets.back() -
                                  host.batch_orbital_offsets[static_cast<std::size_t>(system)];
    system_orbitals[static_cast<std::size_t>(system)] = static_cast<std::int32_t>(orbitals);
    host.matrix_offsets.push_back(host.matrix_offsets.back() + orbitals * orbitals);
    host.pair_offsets.push_back(host.pair_offsets.back() + triangle(atom_count));
    if (explicit_pairs) {
      for (std::int64_t first = atom_begin; first < atom_end; ++first) {
        for (std::int64_t second = first + 1; second < atom_end; ++second) {
          host.atom_pairs.push_back({first, second});
        }
      }
    }
  }

  std::map<std::int32_t, std::vector<std::int32_t>> buckets;
  for (std::int64_t system = 0; system < batch_size; ++system) {
    buckets[system_orbitals[static_cast<std::size_t>(system)]].push_back(
        static_cast<std::int32_t>(system));
  }
  host.bucket_offsets.push_back(0);
  for (const auto& [orbitals, systems] : buckets) {
    host.bucket_orbital_counts.push_back(orbitals);
    host.bucket_systems.insert(host.bucket_systems.end(), systems.begin(), systems.end());
    host.bucket_offsets.push_back(static_cast<std::int64_t>(host.bucket_systems.size()));
  }
  host.refresh_view(explicit_pairs ? Gfn2PairMapKind::kExplicit
                                   : Gfn2PairMapKind::kPackedLowerTriangle);
  return host;
}

struct HostWavefunctionLayout {
  std::vector<std::int32_t> spin_channels;
  std::vector<std::int64_t> spin_channel_offsets;
  std::vector<std::int64_t> spin_orbital_offsets;
  std::vector<std::int64_t> spin_matrix_offsets;
  std::vector<std::int64_t> spin_shell_offsets;
  std::vector<std::int64_t> spin_atom_offsets;
  Gfn2WavefunctionLayoutView view{};

  void refresh_view(const HostTopology& topology) {
    view.memory_space = topology.view.memory_space;
    view.plan_token = topology.view.plan_token;
    view.batch_size = topology.view.batch_size;
    view.total_spin_channels = spin_channel_offsets.back();
    view.total_spin_orbitals = spin_orbital_offsets.back();
    view.total_spin_matrix_elements = spin_matrix_offsets.back();
    view.total_spin_shells = spin_shell_offsets.back();
    view.total_spin_atoms = spin_atom_offsets.back();
    view.spin_channel_count = static_cast<std::int64_t>(spin_channels.size());
    view.spin_channel_offset_count = static_cast<std::int64_t>(spin_channel_offsets.size());
    view.spin_orbital_offset_count = static_cast<std::int64_t>(spin_orbital_offsets.size());
    view.spin_matrix_offset_count = static_cast<std::int64_t>(spin_matrix_offsets.size());
    view.spin_shell_offset_count = static_cast<std::int64_t>(spin_shell_offsets.size());
    view.spin_atom_offset_count = static_cast<std::int64_t>(spin_atom_offsets.size());
    view.spin_channels = pointer_or_null(spin_channels);
    view.spin_channel_offsets = pointer_or_null(spin_channel_offsets);
    view.spin_orbital_offsets = pointer_or_null(spin_orbital_offsets);
    view.spin_matrix_offsets = pointer_or_null(spin_matrix_offsets);
    view.spin_shell_offsets = pointer_or_null(spin_shell_offsets);
    view.spin_atom_offsets = pointer_or_null(spin_atom_offsets);
    view.layout_fingerprint = gfn2_wavefunction_layout_fingerprint_host(view);
  }
};

HostWavefunctionLayout make_wavefunction_layout(const HostTopology& topology) {
  HostWavefunctionLayout layout;
  layout.spin_channel_offsets.push_back(0);
  layout.spin_orbital_offsets.push_back(0);
  layout.spin_matrix_offsets.push_back(0);
  layout.spin_shell_offsets.push_back(0);
  layout.spin_atom_offsets.push_back(0);
  for (std::int64_t system = 0; system < topology.view.batch_size; ++system) {
    const std::int32_t spin_channels = (system % 3) == 0 ? 2 : 1;
    layout.spin_channels.push_back(spin_channels);
    const std::int64_t atoms = topology.atom_offsets[static_cast<std::size_t>(system + 1)] -
                               topology.atom_offsets[static_cast<std::size_t>(system)];
    const std::int64_t shells = topology.batch_shell_offsets[static_cast<std::size_t>(system + 1)] -
                                topology.batch_shell_offsets[static_cast<std::size_t>(system)];
    const std::int64_t orbitals =
        topology.batch_orbital_offsets[static_cast<std::size_t>(system + 1)] -
        topology.batch_orbital_offsets[static_cast<std::size_t>(system)];
    const std::int64_t matrices = topology.matrix_offsets[static_cast<std::size_t>(system + 1)] -
                                  topology.matrix_offsets[static_cast<std::size_t>(system)];
    layout.spin_channel_offsets.push_back(layout.spin_channel_offsets.back() + spin_channels);
    layout.spin_orbital_offsets.push_back(layout.spin_orbital_offsets.back() +
                                          spin_channels * orbitals);
    layout.spin_matrix_offsets.push_back(layout.spin_matrix_offsets.back() +
                                         spin_channels * matrices);
    layout.spin_shell_offsets.push_back(layout.spin_shell_offsets.back() + spin_channels * shells);
    layout.spin_atom_offsets.push_back(layout.spin_atom_offsets.back() + spin_channels * atoms);
  }
  layout.refresh_view(topology);
  return layout;
}

int test_batches_and_layout() {
  static_assert(std::is_trivially_copyable_v<Gfn2RaggedTopologyView>);
  static_assert(std::is_standard_layout_v<Gfn2RaggedTopologyView>);
  static_assert(std::is_trivially_copyable_v<Gfn2GeometryCacheProvenanceView>);
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    HostTopology host = make_topology(batch_size);
    CHECK(validate_gfn2_topology_host(host.view).error == Gfn2PlanSchemaError::kSuccess);
    Gfn2RaggedTopologyView binding{};
    CHECK(bind_gfn2_topology_host(host.view, binding).error == Gfn2PlanSchemaError::kSuccess);
    CHECK(std::memcmp(&binding, &host.view, sizeof(binding)) == 0);

    std::array<std::byte, sizeof(Gfn2RaggedTopologyView)> bytes{};
    std::memcpy(bytes.data(), &binding, sizeof(binding));
    Gfn2RaggedTopologyView restored{};
    std::memcpy(&restored, bytes.data(), sizeof(restored));
    CHECK(std::memcmp(&restored, &binding, sizeof(binding)) == 0);

    host.view.memory_space = Gfn2PlanMemorySpace::kHipDevice;
    CHECK(validate_gfn2_topology_binding(host.view, Gfn2PlanMemorySpace::kHipDevice).error ==
          Gfn2PlanSchemaError::kSuccess);
  }
  HostTopology explicit_host = make_topology(8, true);
  CHECK(validate_gfn2_topology_host(explicit_host.view).error == Gfn2PlanSchemaError::kSuccess);
  HostTopology minimal_host = make_topology(8);
  minimal_host.pair_offsets.clear();
  minimal_host.bucket_offsets.clear();
  minimal_host.bucket_systems.clear();
  minimal_host.bucket_orbital_counts.clear();
  minimal_host.refresh_view(Gfn2PairMapKind::kNone);
  CHECK(validate_gfn2_topology_host(minimal_host.view).error == Gfn2PlanSchemaError::kSuccess);
  return 0;
}

int test_hostile_topologies() {
  HostTopology host = make_topology(8);
  Gfn2RaggedTopologyView candidate = host.view;
  Gfn2RaggedTopologyView binding = host.view;

  candidate.plan_token = 0u;
  CHECK(bind_gfn2_topology_host(candidate, binding).error ==
        Gfn2PlanSchemaError::kInvalidPlanToken);
  CHECK(binding.plan_token == 0u);

  candidate = host.view;
  candidate.batch_size = std::numeric_limits<std::int64_t>::max();
  CHECK(validate_gfn2_topology_binding(candidate, Gfn2PlanMemorySpace::kHost).error ==
        Gfn2PlanSchemaError::kCountOverflow);

  candidate = host.view;
  const std::uintptr_t hostile_address =
      std::numeric_limits<std::uintptr_t>::max() - (alignof(std::int64_t) - 1u);
  candidate.atom_offsets = reinterpret_cast<const std::int64_t*>(hostile_address);
  CHECK(validate_gfn2_topology_binding(candidate, Gfn2PlanMemorySpace::kHost).error ==
        Gfn2PlanSchemaError::kAddressOverflow);

  candidate = host.view;
  candidate.atom_offsets = reinterpret_cast<const std::int64_t*>(
      reinterpret_cast<const std::byte*>(host.atom_offsets.data()) + 1u);
  CHECK(validate_gfn2_topology_binding(candidate, Gfn2PlanMemorySpace::kHost).error ==
        Gfn2PlanSchemaError::kMisalignedPointer);

  candidate = host.view;
  candidate.batch_shell_offsets = host.atom_offsets.data();
  CHECK(validate_gfn2_topology_binding(candidate, Gfn2PlanMemorySpace::kHost).error ==
        Gfn2PlanSchemaError::kAliasedRange);

  host.atom_offsets[3] = host.atom_offsets[2] - 1;
  CHECK(validate_gfn2_topology_host(host.view).field == Gfn2PlanSchemaField::kAtomOffsets);
  host = make_topology(8);
  host.matrix_offsets[2] += 1;
  CHECK(validate_gfn2_topology_host(host.view).error == Gfn2PlanSchemaError::kInvalidMatrixExtent);
  host = make_topology(8);
  host.shell_to_atom[0] += 1;
  CHECK(validate_gfn2_topology_host(host.view).error == Gfn2PlanSchemaError::kInvalidShellMap);
  host = make_topology(8);
  host.orbital_to_shell[0] += 1;
  CHECK(validate_gfn2_topology_host(host.view).error == Gfn2PlanSchemaError::kInvalidOrbitalMap);
  host = make_topology(8);
  host.pair_offsets[2] += 1;
  CHECK(validate_gfn2_topology_host(host.view).error == Gfn2PlanSchemaError::kInvalidPairMap);
  host = make_topology(8);
  host.bucket_systems[1] = host.bucket_systems[0];
  CHECK(validate_gfn2_topology_host(host.view).error == Gfn2PlanSchemaError::kInvalidBucketMap);
  host = make_topology(8);
  host.bucket_orbital_counts[0] = -1;
  CHECK(validate_gfn2_topology_host(host.view).field == Gfn2PlanSchemaField::kBucketOrbitalCounts);

  host = make_topology(8, true);
  if (host.atom_pairs.size() > 1u) {
    host.atom_pairs[1] = host.atom_pairs[0];
    CHECK(validate_gfn2_topology_host(host.view).error == Gfn2PlanSchemaError::kInvalidPairMap);
  }
  return 0;
}

int test_provenance() {
  HostTopology host = make_topology(8);
  Gfn2GeometryCacheProvenanceView batch_cache{};
  batch_cache.memory_space = Gfn2PlanMemorySpace::kHost;
  batch_cache.generation_scope = Gfn2GenerationScope::kBatch;
  batch_cache.plan_token = kPlanToken;
  batch_cache.geometry_generation = kGeneration;
  batch_cache.batch_size = host.view.batch_size;
  CHECK(validate_gfn2_geometry_provenance_host(host.view, batch_cache, kGeneration).error ==
        Gfn2PlanSchemaError::kSuccess);
  CHECK(validate_gfn2_geometry_provenance_host(host.view, batch_cache, kGeneration + 1u).error ==
        Gfn2PlanSchemaError::kStaleGeometry);
  std::vector<std::uint8_t> batch_active(static_cast<std::size_t>(host.view.batch_size), 1u);
  batch_active[1] = 2u;
  CHECK(validate_gfn2_geometry_provenance_host(host.view, batch_cache, kGeneration,
                                               batch_active.data(), host.view.batch_size)
            .error == Gfn2PlanSchemaError::kInvalidActiveMask);
  batch_cache.plan_token += 1u;
  CHECK(validate_gfn2_geometry_provenance_host(host.view, batch_cache, kGeneration).error ==
        Gfn2PlanSchemaError::kCrossPlan);

  std::vector<std::uint64_t> generations(static_cast<std::size_t>(host.view.batch_size),
                                         kGeneration);
  std::vector<std::uint8_t> active(static_cast<std::size_t>(host.view.batch_size), 1u);
  Gfn2GeometryCacheProvenanceView system_cache{};
  system_cache.memory_space = Gfn2PlanMemorySpace::kHost;
  system_cache.generation_scope = Gfn2GenerationScope::kPerSystem;
  system_cache.plan_token = kPlanToken;
  system_cache.batch_size = host.view.batch_size;
  system_cache.system_generation_count = host.view.batch_size;
  system_cache.system_geometry_generations = generations.data();
  CHECK(validate_gfn2_geometry_provenance_host(host.view, system_cache, kGeneration, active.data(),
                                               host.view.batch_size)
            .error == Gfn2PlanSchemaError::kSuccess);
  generations[3] = kGeneration - 1u;
  CHECK(validate_gfn2_geometry_provenance_host(host.view, system_cache, kGeneration, active.data(),
                                               host.view.batch_size)
            .index == 3);
  active[3] = 0u;
  CHECK(validate_gfn2_geometry_provenance_host(host.view, system_cache, kGeneration, active.data(),
                                               host.view.batch_size)
            .error == Gfn2PlanSchemaError::kSuccess);
  active[2] = 2u;
  CHECK(validate_gfn2_geometry_provenance_host(host.view, system_cache, kGeneration, active.data(),
                                               host.view.batch_size)
            .error == Gfn2PlanSchemaError::kInvalidActiveMask);
  CHECK(validate_gfn2_geometry_provenance_host(
            host.view, system_cache, kGeneration,
            reinterpret_cast<const std::uint8_t*>(generations.data()), host.view.batch_size)
            .error == Gfn2PlanSchemaError::kAliasedRange);

  system_cache.system_geometry_generations =
      reinterpret_cast<const std::uint64_t*>(host.atom_offsets.data());
  CHECK(validate_gfn2_geometry_provenance_host(host.view, system_cache, kGeneration).error ==
        Gfn2PlanSchemaError::kAliasedRange);
  return 0;
}

int test_wavefunction_layout() {
  static_assert(std::is_trivially_copyable_v<Gfn2WavefunctionLayoutView>);
  static_assert(std::is_standard_layout_v<Gfn2WavefunctionLayoutView>);
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    HostTopology topology = make_topology(batch_size);
    HostWavefunctionLayout layout = make_wavefunction_layout(topology);
    CHECK(validate_gfn2_wavefunction_layout_host(topology.view, layout.view).error ==
          Gfn2PlanSchemaError::kSuccess);
    Gfn2WavefunctionLayoutView binding{};
    CHECK(bind_gfn2_wavefunction_layout_host(topology.view, layout.view, binding).error ==
          Gfn2PlanSchemaError::kSuccess);
    CHECK(std::memcmp(&binding, &layout.view, sizeof(binding)) == 0);

    topology.view.memory_space = Gfn2PlanMemorySpace::kHipDevice;
    layout.view.memory_space = Gfn2PlanMemorySpace::kHipDevice;
    CHECK(validate_gfn2_wavefunction_layout_binding(topology.view, layout.view,
                                                    Gfn2PlanMemorySpace::kHipDevice)
              .error == Gfn2PlanSchemaError::kSuccess);
  }

  HostTopology topology = make_topology(8);
  HostWavefunctionLayout layout = make_wavefunction_layout(topology);
  layout.view.layout_fingerprint ^= 1u;
  CHECK(validate_gfn2_wavefunction_layout_host(topology.view, layout.view).error ==
        Gfn2PlanSchemaError::kInvalidLayoutFingerprint);

  layout = make_wavefunction_layout(topology);
  layout.spin_channels[3] = 3;
  CHECK(validate_gfn2_wavefunction_layout_host(topology.view, layout.view).error ==
        Gfn2PlanSchemaError::kInvalidSpinChannels);

  layout = make_wavefunction_layout(topology);
  ++layout.spin_orbital_offsets[1];
  CHECK(validate_gfn2_wavefunction_layout_host(topology.view, layout.view).field ==
        Gfn2PlanSchemaField::kSpinOrbitalOffsets);

  layout = make_wavefunction_layout(topology);
  layout.view.plan_token += 1u;
  Gfn2WavefunctionLayoutView binding = layout.view;
  CHECK(bind_gfn2_wavefunction_layout_host(topology.view, layout.view, binding).error ==
        Gfn2PlanSchemaError::kCrossPlan);
  CHECK(binding.plan_token == 0u);

  layout = make_wavefunction_layout(topology);
  layout.view.spin_orbital_offsets = topology.atom_offsets.data();
  CHECK(validate_gfn2_wavefunction_layout_binding(topology.view, layout.view,
                                                  Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kAliasedRange);

  layout = make_wavefunction_layout(topology);
  layout.view.spin_matrix_offsets = nullptr;
  CHECK(validate_gfn2_wavefunction_layout_binding(topology.view, layout.view,
                                                  Gfn2PlanMemorySpace::kHost)
            .field == Gfn2PlanSchemaField::kSpinMatrixOffsets);

  Gfn2WavefunctionLayoutView overflowing{};
  overflowing.plan_token = kPlanToken;
  overflowing.batch_size = std::numeric_limits<std::int64_t>::max();
  CHECK(gfn2_wavefunction_layout_fingerprint_host(overflowing) == 0u);
  return 0;
}

struct HostPairListConsumer {
  std::vector<std::int64_t> pair_offsets;
  std::vector<Gfn2AtomPair> pairs;
  std::vector<std::int64_t> pair_counts;
  std::vector<std::int64_t> neighbor_offsets;
  std::vector<std::int64_t> neighbors;
  std::vector<std::int64_t> neighbor_counts;
  std::vector<std::uint64_t> committed_generations;
  std::vector<std::uint8_t> eligible_mask;
  Gfn2PairListConsumerView view{};

  void refresh_view(const HostTopology& topology, std::uint64_t generation) {
    committed_generations.assign(static_cast<std::size_t>(topology.view.batch_size), generation);
    eligible_mask.assign(static_cast<std::size_t>(topology.view.batch_size), 1u);
    view.memory_space = Gfn2PlanMemorySpace::kHost;
    view.state = Gfn2PairListState::kCommitted;
    view.role = Gfn2PairListRole::kCoordination;
    view.pair_map_kind = Gfn2PairMapKind::kExplicit;
    view.plan_token = topology.view.plan_token;
    view.cutoff_bohr = 25.0;
    view.list_builder_cutoff_bohr = 25.0;
    view.batch_size = topology.view.batch_size;
    view.total_atoms = topology.view.total_atoms;
    view.max_pairs_per_system = 10;
    view.max_neighbors_per_atom = 10;
    view.pair_offset_count = static_cast<std::int64_t>(pair_offsets.size());
    view.neighbor_offset_count = static_cast<std::int64_t>(neighbor_offsets.size());
    view.pair_count = static_cast<std::int64_t>(pairs.size());
    view.neighbor_count = static_cast<std::int64_t>(neighbors.size());
    view.pair_count_elements = static_cast<std::int64_t>(pair_counts.size());
    view.neighbor_count_elements = static_cast<std::int64_t>(neighbor_counts.size());
    view.committed_generation_count = static_cast<std::int64_t>(committed_generations.size());
    view.eligible_mask_count = static_cast<std::int64_t>(eligible_mask.size());
    view.active_mask_count = 0;
    view.pair_offsets = pointer_or_null(pair_offsets);
    view.pairs = pointer_or_null(pairs);
    view.pair_counts = pointer_or_null(pair_counts);
    view.neighbor_counts = pointer_or_null(neighbor_counts);
    view.neighbor_offsets = pointer_or_null(neighbor_offsets);
    view.neighbors = pointer_or_null(neighbors);
    view.committed_generations = committed_generations.data();
    view.eligible_mask = eligible_mask.data();
    view.active_mask = nullptr;
  }
};

HostPairListConsumer make_pair_list_consumer(const HostTopology& topology,
                                             std::uint64_t generation) {
  HostPairListConsumer consumer;
  consumer.pair_offsets.push_back(0);
  consumer.neighbor_offsets.push_back(0);
  for (std::int64_t system = 0; system < topology.view.batch_size; ++system) {
    const std::int64_t atom_begin = topology.view.atom_offsets[static_cast<std::size_t>(system)];
    const std::int64_t atom_end = topology.view.atom_offsets[static_cast<std::size_t>(system + 1)];
    const std::int64_t pair_begin = static_cast<std::int64_t>(consumer.pairs.size());
    for (std::int64_t second = atom_begin + 1; second < atom_end; ++second) {
      for (std::int64_t first = atom_begin; first < second; ++first) {
        consumer.pairs.push_back({first, second});
      }
    }
    consumer.pair_offsets.push_back(static_cast<std::int64_t>(consumer.pairs.size()));
    consumer.pair_counts.push_back(static_cast<std::int64_t>(consumer.pairs.size()) - pair_begin);
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
      const std::int64_t neighbor_begin = static_cast<std::int64_t>(consumer.neighbors.size());
      for (std::int64_t peer = atom_begin; peer < atom_end; ++peer) {
        if (peer != atom) {
          consumer.neighbors.push_back(peer);
        }
      }
      consumer.neighbor_offsets.push_back(static_cast<std::int64_t>(consumer.neighbors.size()));
      consumer.neighbor_counts.push_back(static_cast<std::int64_t>(consumer.neighbors.size()) -
                                         neighbor_begin);
    }
  }
  consumer.committed_generations.assign(static_cast<std::size_t>(topology.view.batch_size),
                                        generation);
  consumer.eligible_mask.assign(static_cast<std::size_t>(topology.view.batch_size), 1u);
  consumer.refresh_view(topology, generation);
  return consumer;
}

HostPairListConsumer make_fixed_stride_pair_list_consumer(const HostTopology& topology,
                                                          std::uint64_t generation) {
  const HostPairListConsumer compact = make_pair_list_consumer(topology, generation);
  HostPairListConsumer padded;
  padded.pair_offsets.resize(static_cast<std::size_t>(topology.view.batch_size + 1));
  padded.pair_counts = compact.pair_counts;
  padded.pairs.resize(static_cast<std::size_t>(topology.view.batch_size * 10));
  for (std::int64_t system = 0; system < topology.view.batch_size; ++system) {
    const std::int64_t source = compact.pair_offsets[static_cast<std::size_t>(system)];
    const std::int64_t count = compact.pair_counts[static_cast<std::size_t>(system)];
    const std::int64_t destination = system * 10;
    padded.pair_offsets[static_cast<std::size_t>(system)] = destination;
    std::copy_n(compact.pairs.begin() + source, count, padded.pairs.begin() + destination);
  }
  padded.pair_offsets.back() = topology.view.batch_size * 10;

  padded.neighbor_offsets.resize(static_cast<std::size_t>(topology.view.total_atoms + 1));
  padded.neighbor_counts = compact.neighbor_counts;
  padded.neighbors.resize(static_cast<std::size_t>(topology.view.total_atoms * 10));
  for (std::int64_t atom = 0; atom < topology.view.total_atoms; ++atom) {
    const std::int64_t source = compact.neighbor_offsets[static_cast<std::size_t>(atom)];
    const std::int64_t count = compact.neighbor_counts[static_cast<std::size_t>(atom)];
    const std::int64_t destination = atom * 10;
    padded.neighbor_offsets[static_cast<std::size_t>(atom)] = destination;
    std::copy_n(compact.neighbors.begin() + source, count, padded.neighbors.begin() + destination);
  }
  padded.neighbor_offsets.back() = topology.view.total_atoms * 10;
  padded.refresh_view(topology, generation);
  return padded;
}

int test_pair_list_consumer() {
  static_assert(std::is_trivially_copyable_v<Gfn2PairListConsumerView>);
  static_assert(std::is_standard_layout_v<Gfn2PairListConsumerView>);
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    HostTopology topology = make_topology(batch_size, /*explicit_pairs=*/true);
    HostPairListConsumer consumer = make_pair_list_consumer(topology, kGeneration);
    CHECK(validate_gfn2_pair_list_consumer_binding(topology.view, consumer.view,
                                                   Gfn2PlanMemorySpace::kHost)
              .error == Gfn2PlanSchemaError::kSuccess);
    CHECK(validate_gfn2_pair_list_consumer_host(topology.view, consumer.view, kGeneration).error ==
          Gfn2PlanSchemaError::kSuccess);
    HostPairListConsumer fixed_stride = make_fixed_stride_pair_list_consumer(topology, kGeneration);
    CHECK(validate_gfn2_pair_list_consumer_host(topology.view, fixed_stride.view, kGeneration)
              .error == Gfn2PlanSchemaError::kSuccess);

    consumer.view.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
    CHECK(validate_gfn2_pair_list_consumer_binding(topology.view, consumer.view,
                                                   Gfn2PlanMemorySpace::kHost)
              .error == Gfn2PlanSchemaError::kInvalidMemorySpace);
    consumer.view.memory_space = Gfn2PlanMemorySpace::kHost;

    consumer.view.plan_token += 1u;
    CHECK(validate_gfn2_pair_list_consumer_binding(topology.view, consumer.view,
                                                   Gfn2PlanMemorySpace::kHost)
              .error == Gfn2PlanSchemaError::kCrossPlan);
    consumer.view.plan_token -= 1u;

    consumer.view.list_builder_cutoff_bohr = 20.0;
    CHECK(validate_gfn2_pair_list_consumer_binding(topology.view, consumer.view,
                                                   Gfn2PlanMemorySpace::kHost)
              .error == Gfn2PlanSchemaError::kInsufficientPairListCutoff);
    consumer.view.list_builder_cutoff_bohr = 25.0;

    consumer.view.cutoff_bohr = std::nextafter(25.0, 0.0);
    CHECK(validate_gfn2_pair_list_consumer_binding(topology.view, consumer.view,
                                                   Gfn2PlanMemorySpace::kHost)
              .error == Gfn2PlanSchemaError::kInsufficientPairListCutoff);
    consumer.view.cutoff_bohr = 25.0;

    consumer.view.list_builder_cutoff_bohr = std::nextafter(25.0, 0.0);
    CHECK(validate_gfn2_pair_list_consumer_binding(topology.view, consumer.view,
                                                   Gfn2PlanMemorySpace::kHost)
              .error == Gfn2PlanSchemaError::kInsufficientPairListCutoff);
    consumer.view.list_builder_cutoff_bohr = 25.0;

    consumer.view.role = static_cast<Gfn2PairListRole>(99u);
    CHECK(validate_gfn2_pair_list_consumer_binding(topology.view, consumer.view,
                                                   Gfn2PlanMemorySpace::kHost)
              .error == Gfn2PlanSchemaError::kInvalidPairListRole);
    consumer.view.role = Gfn2PairListRole::kCoordination;

    consumer.view.state = Gfn2PairListState::kCandidate;
    CHECK(validate_gfn2_pair_list_consumer_binding(topology.view, consumer.view,
                                                   Gfn2PlanMemorySpace::kHost)
              .error == Gfn2PlanSchemaError::kInvalidPairListState);
    consumer.view.state = Gfn2PairListState::kCommitted;

    consumer.view.pair_map_kind = Gfn2PairMapKind::kPackedLowerTriangle;
    CHECK(validate_gfn2_pair_list_consumer_binding(topology.view, consumer.view,
                                                   Gfn2PlanMemorySpace::kHost)
              .error == Gfn2PlanSchemaError::kInvalidPairMap);
    consumer.view.pair_map_kind = Gfn2PairMapKind::kExplicit;

    consumer.view.max_pairs_per_system = 0;
    CHECK(validate_gfn2_pair_list_consumer_binding(topology.view, consumer.view,
                                                   Gfn2PlanMemorySpace::kHost)
              .error == Gfn2PlanSchemaError::kInvalidCount);
    consumer.view.max_pairs_per_system = 10;

    if (batch_size > 1) {
      consumer.view.max_pairs_per_system = std::numeric_limits<std::int64_t>::max();
      CHECK(validate_gfn2_pair_list_consumer_binding(topology.view, consumer.view,
                                                     Gfn2PlanMemorySpace::kHost)
                .error == Gfn2PlanSchemaError::kCountOverflow);
      consumer.view.max_pairs_per_system = 10;
    }

    /* Device-space structural validation (no dereference). */
    HostTopology device_topology = topology;
    device_topology.view.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
    Gfn2PairListConsumerView device_view = consumer.view;
    device_view.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
    CHECK(validate_gfn2_pair_list_consumer_binding(device_topology.view, device_view,
                                                   Gfn2PlanMemorySpace::kCudaDevice)
              .error == Gfn2PlanSchemaError::kSuccess);
  }

  HostTopology topology = make_topology(8, /*explicit_pairs=*/true);
  HostPairListConsumer consumer = make_pair_list_consumer(topology, kGeneration);

  struct RoleCutoffCase {
    Gfn2PairListRole role;
    double cutoff;
  };
  for (const RoleCutoffCase& role_case : {
           RoleCutoffCase{Gfn2PairListRole::kCoordination, 25.0},
           RoleCutoffCase{Gfn2PairListRole::kD4Coordination, 30.0},
           RoleCutoffCase{Gfn2PairListRole::kD4TwoBody, 50.0},
           RoleCutoffCase{Gfn2PairListRole::kD4Atm, 25.0},
       }) {
    consumer.view.role = role_case.role;
    consumer.view.cutoff_bohr = role_case.cutoff;
    consumer.view.list_builder_cutoff_bohr = role_case.cutoff;
    CHECK(validate_gfn2_pair_list_consumer_binding(topology.view, consumer.view,
                                                   Gfn2PlanMemorySpace::kHost)
              .error == Gfn2PlanSchemaError::kSuccess);
    consumer.view.list_builder_cutoff_bohr = std::nextafter(role_case.cutoff, 0.0);
    CHECK(validate_gfn2_pair_list_consumer_binding(topology.view, consumer.view,
                                                   Gfn2PlanMemorySpace::kHost)
              .error == Gfn2PlanSchemaError::kInsufficientPairListCutoff);
  }
  consumer = make_pair_list_consumer(topology, kGeneration);

  /* Host inspection: stale eligible peer. */
  consumer.committed_generations[3] = kGeneration - 1u;
  CHECK(validate_gfn2_pair_list_consumer_host(topology.view, consumer.view, kGeneration).index ==
        3);
  consumer.eligible_mask[3] = 0u;
  CHECK(validate_gfn2_pair_list_consumer_host(topology.view, consumer.view, kGeneration).error ==
        Gfn2PlanSchemaError::kSuccess);
  consumer.eligible_mask[3] = 1u;
  consumer.committed_generations[3] = kGeneration;

  /* Host inspection: inverted pair ordering is rejected. */
  if (!consumer.pairs.empty()) {
    std::swap(consumer.pairs.front().first, consumer.pairs.front().second);
    CHECK(validate_gfn2_pair_list_consumer_host(topology.view, consumer.view, kGeneration).error ==
          Gfn2PlanSchemaError::kInvalidPairMap);
    consumer = make_pair_list_consumer(topology, kGeneration);
  }

  /* Host inspection: the CUDA publisher's second-major pair order is strict. */
  if (consumer.pairs.size() >= 2u) {
    std::swap(consumer.pairs[0], consumer.pairs[1]);
    CHECK(validate_gfn2_pair_list_consumer_host(topology.view, consumer.view, kGeneration).error ==
          Gfn2PlanSchemaError::kInvalidPairMap);
    consumer = make_pair_list_consumer(topology, kGeneration);
  }

  /* Host inspection: neighbor ranges are strictly ascending and exclude self. */
  if (consumer.neighbors.size() >= 2u) {
    consumer.neighbors[1] = consumer.neighbors[0];
    CHECK(validate_gfn2_pair_list_consumer_host(topology.view, consumer.view, kGeneration).error ==
          Gfn2PlanSchemaError::kInvalidPairMap);
    consumer = make_pair_list_consumer(topology, kGeneration);
  }

  /* Counts are an independent commit record and must fit within each slot. */
  consumer.pair_counts[3] += 1;
  CHECK(validate_gfn2_pair_list_consumer_host(topology.view, consumer.view, kGeneration).error ==
        Gfn2PlanSchemaError::kInvalidOffsets);
  consumer = make_pair_list_consumer(topology, kGeneration);
  consumer.neighbor_counts[3] += 1;
  CHECK(validate_gfn2_pair_list_consumer_host(topology.view, consumer.view, kGeneration).error ==
        Gfn2PlanSchemaError::kInvalidOffsets);
  consumer = make_pair_list_consumer(topology, kGeneration);

  /* Host inspection: neighbor escapes its owning system. */
  {
    HostTopology single = make_topology(1, /*explicit_pairs=*/true);
    HostPairListConsumer single_consumer = make_pair_list_consumer(single, kGeneration);
    const std::int64_t atom_end = single.view.atom_offsets[1];
    bool neighbor_fixed = false;
    for (std::size_t index = 0; index < single_consumer.neighbors.size(); ++index) {
      const std::int64_t peer = single_consumer.neighbors[index];
      std::int64_t system_begin = 0;
      std::int64_t system_end = atom_end;
      if (peer == system_end - 1) {
        single_consumer.neighbors[index] = system_end;
        neighbor_fixed = true;
        break;
      }
      static_cast<void>(system_begin);
    }
    if (neighbor_fixed) {
      CHECK(validate_gfn2_pair_list_consumer_host(single.view, single_consumer.view, kGeneration)
                .error == Gfn2PlanSchemaError::kInvalidPairMap);
    }
  }

  /* Binding alias: pair_offsets aliasing a topology array. */
  consumer.view.pair_offsets = topology.view.atom_offsets;
  CHECK(validate_gfn2_pair_list_consumer_binding(topology.view, consumer.view,
                                                 Gfn2PlanMemorySpace::kHost)
                .field == Gfn2PlanSchemaField::kPairListPairs ||
        validate_gfn2_pair_list_consumer_binding(topology.view, consumer.view,
                                                 Gfn2PlanMemorySpace::kHost)
                .error == Gfn2PlanSchemaError::kAliasedRange);
  consumer.refresh_view(topology, kGeneration);

  /* Binding alias: eligible_mask aliasing a topology array. */
  consumer.view.eligible_mask = reinterpret_cast<const std::uint8_t*>(topology.view.atom_offsets);
  CHECK(validate_gfn2_pair_list_consumer_binding(topology.view, consumer.view,
                                                 Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kAliasedRange);
  consumer.refresh_view(topology, kGeneration);

  /* Binding: active mask length mismatch. */
  consumer.view.active_mask_count = topology.view.batch_size - 1;
  consumer.view.active_mask = reinterpret_cast<const std::uint8_t*>(consumer.pairs.data());
  CHECK(validate_gfn2_pair_list_consumer_binding(topology.view, consumer.view,
                                                 Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kInvalidActiveMask);
  consumer.refresh_view(topology, kGeneration);

  consumer.view.active_mask_count = topology.view.batch_size;
  consumer.view.active_mask = nullptr;
  CHECK(validate_gfn2_pair_list_consumer_binding(topology.view, consumer.view,
                                                 Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kInvalidActiveMask);

  /* All-empty batches still validate generations and mask bytes. */
  HostTopology empty = make_topology(1, /*explicit_pairs=*/true);
  empty.atom_offsets = {0, 0};
  empty.batch_shell_offsets = {0, 0};
  empty.batch_orbital_offsets = {0, 0};
  empty.matrix_offsets = {0, 0};
  empty.atom_shell_offsets = {0};
  empty.shell_orbital_offsets = {0};
  empty.shell_to_atom.clear();
  empty.orbital_to_shell.clear();
  empty.orbital_to_atom.clear();
  empty.pair_offsets = {0, 0};
  empty.atom_pairs.clear();
  empty.bucket_offsets = {0, 1};
  empty.bucket_systems = {0};
  empty.bucket_orbital_counts = {0};
  empty.refresh_view(Gfn2PairMapKind::kExplicit);
  CHECK(validate_gfn2_topology_host(empty.view).error == Gfn2PlanSchemaError::kSuccess);
  HostPairListConsumer empty_consumer = make_pair_list_consumer(empty, kGeneration);
  empty_consumer.eligible_mask[0] = 2u;
  CHECK(validate_gfn2_pair_list_consumer_host(empty.view, empty_consumer.view, kGeneration).error ==
        Gfn2PlanSchemaError::kInvalidActiveMask);
  return 0;
}

int test_projections() {
  static_assert(std::is_trivially_copyable_v<Gfn2AtomProjectionView>);
  static_assert(std::is_standard_layout_v<Gfn2AtomProjectionView>);
  static_assert(std::is_trivially_copyable_v<Gfn2ShellOwnershipProjectionView>);
  static_assert(std::is_standard_layout_v<Gfn2ShellOwnershipProjectionView>);
  static_assert(std::is_trivially_copyable_v<Gfn2AOMatrixProjectionView>);
  static_assert(std::is_standard_layout_v<Gfn2AOMatrixProjectionView>);
  static_assert(std::is_trivially_copyable_v<Gfn2PackedAllPairProjectionView>);
  static_assert(std::is_standard_layout_v<Gfn2PackedAllPairProjectionView>);
  static_assert(std::is_trivially_copyable_v<Gfn2AOBucketProjectionView>);
  static_assert(std::is_standard_layout_v<Gfn2AOBucketProjectionView>);
  static_assert(std::is_trivially_copyable_v<Gfn2ElementIdentityProjectionView>);
  static_assert(std::is_standard_layout_v<Gfn2ElementIdentityProjectionView>);

  Gfn2AtomProjectionView atom{};
  Gfn2ShellOwnershipProjectionView shell{};
  Gfn2AOMatrixProjectionView ao{};
  Gfn2PackedAllPairProjectionView pairs{};
  Gfn2AOBucketProjectionView buckets{};

  const HostTopology host = make_topology(8);
  const std::int64_t batch_offsets = host.view.batch_size + 1;
  CHECK(project_gfn2_atom_projection_host(host.view, atom).error == Gfn2PlanSchemaError::kSuccess);
  CHECK(atom.batch_size == host.view.batch_size);
  CHECK(atom.total_atoms == host.view.total_atoms);
  CHECK(atom.atom_offset_count == batch_offsets);
  CHECK(atom.atom_offsets == host.view.atom_offsets);
  CHECK(validate_gfn2_atom_projection_binding(host.view, atom, Gfn2PlanMemorySpace::kHost).error ==
        Gfn2PlanSchemaError::kSuccess);

  CHECK(project_gfn2_shell_ownership_projection_host(host.view, shell).error ==
        Gfn2PlanSchemaError::kSuccess);
  CHECK(shell.batch_shell_offset_count == batch_offsets);
  CHECK(shell.atom_shell_offset_count == host.view.total_atoms + 1);
  CHECK(shell.batch_shell_offsets == host.view.batch_shell_offsets);
  CHECK(shell.atom_shell_offsets == host.view.atom_shell_offsets);
  CHECK(shell.shell_to_atom == host.view.shell_to_atom);
  CHECK(
      validate_gfn2_shell_ownership_projection_binding(host.view, shell, Gfn2PlanMemorySpace::kHost)
          .error == Gfn2PlanSchemaError::kSuccess);

  CHECK(project_gfn2_ao_matrix_projection_host(host.view, ao).error ==
        Gfn2PlanSchemaError::kSuccess);
  CHECK(ao.batch_orbital_offset_count == batch_offsets);
  CHECK(ao.matrix_offset_count == batch_offsets);
  CHECK(ao.shell_orbital_offset_count == host.view.total_shells + 1);
  CHECK(ao.batch_orbital_offsets == host.view.batch_orbital_offsets);
  CHECK(ao.matrix_offsets == host.view.matrix_offsets);
  CHECK(ao.shell_orbital_offsets == host.view.shell_orbital_offsets);
  CHECK(ao.orbital_to_shell == host.view.orbital_to_shell);
  CHECK(ao.orbital_to_atom == host.view.orbital_to_atom);
  CHECK(
      validate_gfn2_ao_matrix_projection_binding(host.view, ao, Gfn2PlanMemorySpace::kHost).error ==
      Gfn2PlanSchemaError::kSuccess);

  CHECK(project_gfn2_packed_all_pair_projection_host(host.view, pairs).error ==
        Gfn2PlanSchemaError::kSuccess);
  CHECK(pairs.pair_offset_count == batch_offsets);
  CHECK(pairs.pair_offsets == host.view.pair_offsets);
  CHECK(
      validate_gfn2_packed_all_pair_projection_binding(host.view, pairs, Gfn2PlanMemorySpace::kHost)
          .error == Gfn2PlanSchemaError::kSuccess);

  CHECK(project_gfn2_ao_bucket_projection_host(host.view, buckets).error ==
        Gfn2PlanSchemaError::kSuccess);
  CHECK(buckets.bucket_count == host.view.bucket_count);
  CHECK(buckets.bucket_offsets == host.view.bucket_offsets);
  CHECK(buckets.bucket_systems == host.view.bucket_systems);
  CHECK(buckets.bucket_orbital_counts == host.view.bucket_orbital_counts);
  CHECK(validate_gfn2_ao_bucket_projection_binding(host.view, buckets, Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kSuccess);

  /* Element identity: order-sensitive fingerprint and count/token identity. */
  const std::vector<std::int32_t> atomic_numbers = {1, 6, 7, 8, 1, 6, 7, 8};
  Gfn2ElementIdentityProjectionView element{};
  CHECK(project_gfn2_element_identity_projection_host(
            atomic_numbers.data(), static_cast<std::int64_t>(atomic_numbers.size()), kPlanToken,
            element)
            .error == Gfn2PlanSchemaError::kSuccess);
  CHECK(element.element_fingerprint != 0u);
  CHECK(validate_gfn2_element_identity_projection_binding(element, Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kSuccess);
  Gfn2ElementIdentityProjectionView reordered = element;
  reordered.atomic_numbers = nullptr;
  reordered.total_atoms = 0;
  reordered.atomic_number_count = 0;
  CHECK(project_gfn2_element_identity_projection_host(
            atomic_numbers.data(), static_cast<std::int64_t>(atomic_numbers.size()), kPlanToken,
            reordered)
            .error == Gfn2PlanSchemaError::kSuccess);
  CHECK(reordered.element_fingerprint == element.element_fingerprint);
  CHECK(gfn2_element_identity_fingerprint_host(reordered) == element.element_fingerprint);

  /* Fingerprint is order-sensitive: swapping two elements changes it. */
  std::vector<std::int32_t> permuted = atomic_numbers;
  std::swap(permuted[0], permuted[1]);
  Gfn2ElementIdentityProjectionView permuted_element{};
  CHECK(project_gfn2_element_identity_projection_host(permuted.data(),
                                                      static_cast<std::int64_t>(permuted.size()),
                                                      kPlanToken, permuted_element)
            .error == Gfn2PlanSchemaError::kSuccess);
  CHECK(permuted_element.element_fingerprint != element.element_fingerprint);

  /* Cross-plan and memory-space rejection, and fail-closed clearing. */
  Gfn2AtomProjectionView wrong_plan = atom;
  wrong_plan.plan_token = kPlanToken + 1u;
  CHECK(validate_gfn2_atom_projection_binding(host.view, wrong_plan, Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kCrossPlan);
  Gfn2AtomProjectionView wrong_space = atom;
  wrong_space.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
  CHECK(validate_gfn2_atom_projection_binding(host.view, wrong_space, Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kInvalidMemorySpace);
  wrong_space = atom;
  CHECK(validate_gfn2_atom_projection_binding(host.view, wrong_space,
                                              Gfn2PlanMemorySpace::kCudaDevice)
            .error == Gfn2PlanSchemaError::kInvalidMemorySpace);
  Gfn2AtomProjectionView wrong_count = atom;
  CHECK(validate_gfn2_atom_projection_binding(host.view, wrong_count, Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kSuccess);
  wrong_count.atom_offset_count = batch_offsets + 1;
  CHECK(validate_gfn2_atom_projection_binding(host.view, wrong_count, Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kInvalidProjection);
  wrong_count = atom;
  CHECK(validate_gfn2_atom_projection_binding(host.view, wrong_count, Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kSuccess);
  wrong_count.atom_offsets = host.view.atom_shell_offsets;
  CHECK(validate_gfn2_atom_projection_binding(host.view, wrong_count, Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kInvalidProjection);

  Gfn2AOMatrixProjectionView wrong_domain = ao;
  CHECK(validate_gfn2_ao_matrix_projection_binding(host.view, wrong_domain,
                                                   Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kSuccess);
  wrong_domain.matrix_offsets = host.view.matrix_offsets;
  wrong_domain.matrix_offset_count = host.view.matrix_offset_count - 1;
  CHECK(validate_gfn2_ao_matrix_projection_binding(host.view, wrong_domain,
                                                   Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kInvalidProjection);

  Gfn2PackedAllPairProjectionView wrong_pairs = pairs;
  wrong_pairs.pair_offsets = host.view.atom_offsets;
  CHECK(validate_gfn2_packed_all_pair_projection_binding(host.view, wrong_pairs,
                                                         Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kInvalidProjection);

  Gfn2AOBucketProjectionView wrong_buckets = buckets;
  wrong_buckets.bucket_systems = host.view.bucket_systems;
  wrong_buckets.bucket_system_count = host.view.bucket_system_count + 1;
  CHECK(validate_gfn2_ao_bucket_projection_binding(host.view, wrong_buckets,
                                                   Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kInvalidProjection);

  /* Element identity: token, count, and fingerprint failures.  The plan token
   * is inside the order-sensitive seal, so a token change is provable as a
   * fingerprint mismatch without a second master token. */
  Gfn2ElementIdentityProjectionView bad_element = element;
  bad_element.plan_token = 0u;
  CHECK(validate_gfn2_element_identity_projection_binding(bad_element, Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kInvalidPlanToken);
  bad_element = element;
  bad_element.plan_token = kPlanToken + 1u;
  CHECK(validate_gfn2_element_identity_projection_binding(bad_element, Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kInvalidElementFingerprint);
  bad_element = element;
  bad_element.total_atoms = element.total_atoms - 1;
  CHECK(validate_gfn2_element_identity_projection_binding(bad_element, Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kElementCountMismatch);
  bad_element = element;
  bad_element.element_fingerprint = 0u;
  CHECK(validate_gfn2_element_identity_projection_binding(bad_element, Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kInvalidElementFingerprint);
  bad_element = element;
  bad_element.element_fingerprint ^= 1u;
  CHECK(validate_gfn2_element_identity_projection_binding(bad_element, Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kInvalidElementFingerprint);

  /* Empty element identity still fingerprints and validates. */
  Gfn2ElementIdentityProjectionView empty_element{};
  CHECK(
      project_gfn2_element_identity_projection_host(nullptr, 0, kPlanToken, empty_element).error ==
      Gfn2PlanSchemaError::kSuccess);
  CHECK(validate_gfn2_element_identity_projection_binding(empty_element, Gfn2PlanMemorySpace::kHost)
            .error == Gfn2PlanSchemaError::kSuccess);

  /* Projectors clear the output on a hostile master. */
  HostTopology hostile = make_topology(1);
  hostile.atom_offsets = {0, -1};
  hostile.refresh_view();
  Gfn2AtomProjectionView cleared{};
  cleared.plan_token = kPlanToken;
  CHECK(project_gfn2_atom_projection_host(hostile.view, cleared).error !=
        Gfn2PlanSchemaError::kSuccess);
  CHECK(cleared.plan_token == 0u && cleared.atom_offsets == nullptr);

  /* CUDA-space projections of the same plan are revalidated at the memory
   * space the binding expects; host projection of a device master fails. */
  Gfn2RaggedTopologyView device_topology = host.view;
  device_topology.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
  Gfn2AtomProjectionView device_atom{};
  CHECK(project_gfn2_atom_projection_host(device_topology, device_atom).error !=
        Gfn2PlanSchemaError::kSuccess);
  /* But structural binding accepts a hand-built device projection copy when
   * the memory space matches; the element seal travels unchanged. */
  Gfn2AtomProjectionView hand_device = atom;
  hand_device.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
  hand_device.plan_token = device_topology.plan_token;
  CHECK(validate_gfn2_atom_projection_binding(device_topology, hand_device,
                                              Gfn2PlanMemorySpace::kCudaDevice)
            .error == Gfn2PlanSchemaError::kSuccess);
  Gfn2ElementIdentityProjectionView device_element = element;
  device_element.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
  CHECK(validate_gfn2_element_identity_projection_binding(device_element,
                                                          Gfn2PlanMemorySpace::kCudaDevice)
            .error == Gfn2PlanSchemaError::kSuccess);
  return 0;
}

}  // namespace

int main() {
  int status = test_batches_and_layout();
  if (status == 0) {
    status = test_hostile_topologies();
  }
  if (status == 0) {
    status = test_provenance();
  }
  if (status == 0) {
    status = test_wavefunction_layout();
  }
  if (status == 0) {
    status = test_pair_list_consumer();
  }
  if (status == 0) {
    status = test_projections();
  }
  return status;
}
