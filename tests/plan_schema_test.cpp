#include <algorithm>
#include <array>
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
using gpuxtb::detail::Gfn2AtomPair;
using gpuxtb::detail::Gfn2GenerationScope;
using gpuxtb::detail::Gfn2GeometryCacheProvenanceView;
using gpuxtb::detail::Gfn2PairMapKind;
using gpuxtb::detail::Gfn2PlanMemorySpace;
using gpuxtb::detail::Gfn2PlanSchemaDiagnostic;
using gpuxtb::detail::Gfn2PlanSchemaError;
using gpuxtb::detail::Gfn2PlanSchemaField;
using gpuxtb::detail::Gfn2RaggedTopologyView;
using gpuxtb::detail::Gfn2WavefunctionLayoutView;
using gpuxtb::detail::validate_gfn2_geometry_provenance_host;
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
  return status;
}
