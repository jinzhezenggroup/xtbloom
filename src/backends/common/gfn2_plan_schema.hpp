#ifndef GPUXTB_BACKENDS_COMMON_GFN2_PLAN_SCHEMA_HPP
#define GPUXTB_BACKENDS_COMMON_GFN2_PLAN_SCHEMA_HPP

#include <cstdint>
#include <type_traits>

namespace gpuxtb::detail {

/*
 * The common schema deliberately contains no CUDA or HIP types.  A topology
 * has identical layout in host, CUDA-device, and future HIP-device memory;
 * only the setup-time validator used to inspect the pointed-to arrays changes.
 */
enum class Gfn2PlanMemorySpace : std::uint32_t {
  kHost = 1u,
  kCudaDevice = 2u,
  kHipDevice = 3u,
};

enum class Gfn2PairMapKind : std::uint32_t {
  kNone = 0u,
  kPackedLowerTriangle = 1u,
  kExplicit = 2u,
};

enum class Gfn2GenerationScope : std::uint32_t {
  kBatch = 1u,
  kPerSystem = 2u,
};

enum class Gfn2PlanSchemaError : std::uint32_t {
  kSuccess = 0u,
  kInvalidMemorySpace = 1u,
  kInvalidCount = 2u,
  kCountOverflow = 3u,
  kInvalidPlanToken = 4u,
  kNullPointer = 5u,
  kMisalignedPointer = 6u,
  kAddressOverflow = 7u,
  kAliasedRange = 8u,
  kInvalidOffsets = 9u,
  kInvalidMatrixExtent = 10u,
  kInvalidShellMap = 11u,
  kInvalidOrbitalMap = 12u,
  kInvalidPairMap = 13u,
  kInvalidBucketMap = 14u,
  kCrossPlan = 15u,
  kStaleGeometry = 16u,
  kInvalidActiveMask = 17u,
  kInvalidSpinChannels = 18u,
  kInvalidWavefunctionExtent = 19u,
};

/* Identifies the first field involved in a setup failure. */
enum class Gfn2PlanSchemaField : std::uint32_t {
  kNone = 0u,
  kTopology = 1u,
  kAtomOffsets = 2u,
  kBatchShellOffsets = 3u,
  kBatchOrbitalOffsets = 4u,
  kMatrixOffsets = 5u,
  kAtomShellOffsets = 6u,
  kShellOrbitalOffsets = 7u,
  kShellToAtom = 8u,
  kOrbitalToShell = 9u,
  kOrbitalToAtom = 10u,
  kPairOffsets = 11u,
  kAtomPairs = 12u,
  kBucketOffsets = 13u,
  kBucketSystems = 14u,
  kBucketOrbitalCounts = 15u,
  kGeometryProvenance = 16u,
  kSystemGeometryGenerations = 17u,
  kActiveMask = 18u,
  kSpinChannels = 19u,
  kSpinChannelOffsets = 20u,
  kSpinOrbitalOffsets = 21u,
  kSpinMatrixOffsets = 22u,
  kSpinShellOffsets = 23u,
  kSpinAtomOffsets = 24u,
};

struct Gfn2PlanSchemaDiagnostic {
  Gfn2PlanSchemaError error = Gfn2PlanSchemaError::kSuccess;
  Gfn2PlanSchemaField field = Gfn2PlanSchemaField::kNone;
  /* Global array index, system index, or -1 when no narrower location exists. */
  std::int64_t index = -1;
};

/* Explicit unordered atom pair.  Valid plans store first < second. */
struct Gfn2AtomPair {
  std::int64_t first = 0;
  std::int64_t second = 0;
};

/*
 * Immutable ragged GFN2 topology.  Every extent is in elements, never bytes.
 * Empty systems are valid, but nonempty atoms own at least one shell and every
 * shell owns at least one spherical AO.  Matrix slices are complete row-major
 * squares.  All global maps use the same batch-major numbering as the offsets.
 *
 * Pair maps and homogeneous AO-size buckets are optional.  Packed pairs need
 * only pair_offsets; explicit pairs are lexicographically sorted per system.
 * A bucket map is a permutation of all systems grouped by AO count.
 */
struct Gfn2RaggedTopologyView {
  Gfn2PlanMemorySpace memory_space = Gfn2PlanMemorySpace::kHost;
  Gfn2PairMapKind pair_map_kind = Gfn2PairMapKind::kNone;
  std::uint64_t plan_token = 0u;

  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t total_matrix_elements = 0;
  std::int64_t total_pairs = 0;
  std::int64_t bucket_count = 0;

  std::int64_t atom_offset_count = 0;
  std::int64_t batch_shell_offset_count = 0;
  std::int64_t batch_orbital_offset_count = 0;
  std::int64_t matrix_offset_count = 0;
  std::int64_t atom_shell_offset_count = 0;
  std::int64_t shell_orbital_offset_count = 0;
  std::int64_t shell_to_atom_count = 0;
  std::int64_t orbital_to_shell_count = 0;
  std::int64_t orbital_to_atom_count = 0;
  std::int64_t pair_offset_count = 0;
  std::int64_t atom_pair_count = 0;
  std::int64_t bucket_offset_count = 0;
  std::int64_t bucket_system_count = 0;
  std::int64_t bucket_orbital_count = 0;

  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* batch_shell_offsets = nullptr;
  const std::int64_t* batch_orbital_offsets = nullptr;
  const std::int64_t* matrix_offsets = nullptr;
  const std::int64_t* atom_shell_offsets = nullptr;
  const std::int64_t* shell_orbital_offsets = nullptr;
  const std::int64_t* shell_to_atom = nullptr;
  const std::int64_t* orbital_to_shell = nullptr;
  const std::int64_t* orbital_to_atom = nullptr;
  const std::int64_t* pair_offsets = nullptr;
  const Gfn2AtomPair* atom_pairs = nullptr;
  const std::int64_t* bucket_offsets = nullptr;
  const std::int32_t* bucket_systems = nullptr;
  const std::int32_t* bucket_orbital_counts = nullptr;
};

/*
 * Exact device-neutral projection of WavefunctionLayout's spin-dependent
 * system offsets.  The physical atom/shell/orbital/matrix partitions remain
 * authoritative in Gfn2RaggedTopologyView; this view adds only the extents
 * that grow with nspin.  Keeping the two views separate lets one overlap
 * factor and one integral matrix serve both unrestricted channels.
 *
 * spin_channel_offsets partitions scalar per-channel diagnostics and advances
 * by nspin.  The other offsets advance respectively by nspin*nao,
 * nspin*nao*nao, nspin*nsh, and nspin*nat.  Occupations intentionally do not
 * appear here because their immutable CPU/CUDA contract is always
 * 2*batch_orbital_offsets, even for restricted systems.
 */
struct Gfn2WavefunctionLayoutView {
  Gfn2PlanMemorySpace memory_space = Gfn2PlanMemorySpace::kHost;
  std::uint64_t plan_token = 0u;

  std::int64_t batch_size = 0;
  std::int64_t total_spin_channels = 0;
  std::int64_t total_spin_orbitals = 0;
  std::int64_t total_spin_matrix_elements = 0;
  std::int64_t total_spin_shells = 0;
  std::int64_t total_spin_atoms = 0;

  std::int64_t spin_channel_count = 0;
  std::int64_t spin_channel_offset_count = 0;
  std::int64_t spin_orbital_offset_count = 0;
  std::int64_t spin_matrix_offset_count = 0;
  std::int64_t spin_shell_offset_count = 0;
  std::int64_t spin_atom_offset_count = 0;

  const std::int32_t* spin_channels = nullptr;
  const std::int64_t* spin_channel_offsets = nullptr;
  const std::int64_t* spin_orbital_offsets = nullptr;
  const std::int64_t* spin_matrix_offsets = nullptr;
  const std::int64_t* spin_shell_offsets = nullptr;
  const std::int64_t* spin_atom_offsets = nullptr;
};

/*
 * Provenance for a geometry-derived cache.  Batch-scoped caches store one
 * scalar generation.  Transactional caches instead store one generation per
 * system; scalar geometry_generation must then remain zero to avoid two
 * competing authorities.  Generation zero is reserved for unpublished data.
 */
struct Gfn2GeometryCacheProvenanceView {
  Gfn2PlanMemorySpace memory_space = Gfn2PlanMemorySpace::kHost;
  Gfn2GenerationScope generation_scope = Gfn2GenerationScope::kBatch;
  std::uint64_t plan_token = 0u;
  std::uint64_t geometry_generation = 0u;
  std::int64_t batch_size = 0;
  std::int64_t system_generation_count = 0;
  const std::uint64_t* system_geometry_generations = nullptr;
};

static_assert(std::is_trivially_copyable_v<Gfn2PlanSchemaDiagnostic>);
static_assert(std::is_standard_layout_v<Gfn2PlanSchemaDiagnostic>);
static_assert(std::is_trivially_copyable_v<Gfn2AtomPair>);
static_assert(std::is_standard_layout_v<Gfn2AtomPair>);
static_assert(std::is_trivially_copyable_v<Gfn2RaggedTopologyView>);
static_assert(std::is_standard_layout_v<Gfn2RaggedTopologyView>);
static_assert(std::is_trivially_copyable_v<Gfn2WavefunctionLayoutView>);
static_assert(std::is_standard_layout_v<Gfn2WavefunctionLayoutView>);
static_assert(std::is_trivially_copyable_v<Gfn2GeometryCacheProvenanceView>);
static_assert(std::is_standard_layout_v<Gfn2GeometryCacheProvenanceView>);

/*
 * Validate counts, extents, alignment, address arithmetic, and aliases without
 * dereferencing any array.  This is the first stage for every address space.
 */
[[nodiscard]] Gfn2PlanSchemaDiagnostic validate_gfn2_topology_binding(
    const Gfn2RaggedTopologyView& topology, Gfn2PlanMemorySpace expected_memory_space) noexcept;

/* Full host inspection of every offset and map. */
[[nodiscard]] Gfn2PlanSchemaDiagnostic validate_gfn2_topology_host(
    const Gfn2RaggedTopologyView& topology) noexcept;

/*
 * Fail-closed host builder.  `binding` is cleared on failure and receives an
 * exact POD copy only after complete validation succeeds.
 */
[[nodiscard]] Gfn2PlanSchemaDiagnostic bind_gfn2_topology_host(
    const Gfn2RaggedTopologyView& candidate, Gfn2RaggedTopologyView& binding) noexcept;

/*
 * Validate the structural pointer/count contract without dereferencing device
 * metadata.  topology and layout must name the same plan, memory space, and
 * batch.  The layout arrays must be disjoint from each other and from every
 * immutable topology array.
 */
[[nodiscard]] Gfn2PlanSchemaDiagnostic validate_gfn2_wavefunction_layout_binding(
    const Gfn2RaggedTopologyView& topology, const Gfn2WavefunctionLayoutView& layout,
    Gfn2PlanMemorySpace expected_memory_space) noexcept;

/* Inspect every host spin value and prove its ragged extents against topology. */
[[nodiscard]] Gfn2PlanSchemaDiagnostic validate_gfn2_wavefunction_layout_host(
    const Gfn2RaggedTopologyView& topology, const Gfn2WavefunctionLayoutView& layout) noexcept;

/* Fail-closed host builder; binding is cleared before validation. */
[[nodiscard]] Gfn2PlanSchemaDiagnostic bind_gfn2_wavefunction_layout_host(
    const Gfn2RaggedTopologyView& topology, const Gfn2WavefunctionLayoutView& candidate,
    Gfn2WavefunctionLayoutView& binding) noexcept;

/* Structural validation that never dereferences the optional generation array. */
[[nodiscard]] Gfn2PlanSchemaDiagnostic validate_gfn2_geometry_provenance_binding(
    const Gfn2RaggedTopologyView& topology, const Gfn2GeometryCacheProvenanceView& provenance,
    Gfn2PlanMemorySpace expected_memory_space) noexcept;

/*
 * Validate plan identity and the requested nonzero geometry generation.  A
 * null active mask means every system is active; otherwise it must contain
 * exactly batch_size bytes whose values are zero or one.  Stale inactive
 * systems are accepted so peer-local transactional failures remain reusable.
 */
[[nodiscard]] Gfn2PlanSchemaDiagnostic validate_gfn2_geometry_provenance_host(
    const Gfn2RaggedTopologyView& topology, const Gfn2GeometryCacheProvenanceView& provenance,
    std::uint64_t expected_geometry_generation, const std::uint8_t* active_mask = nullptr,
    std::int64_t active_mask_elements = 0) noexcept;

}  // namespace gpuxtb::detail

#endif  // GPUXTB_BACKENDS_COMMON_GFN2_PLAN_SCHEMA_HPP
