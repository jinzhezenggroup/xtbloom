#ifndef GPUXTB_BACKENDS_COMMON_GFN2_PLAN_SCHEMA_HPP
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

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

/*
 * Physical role a sparse pair-list consumer computes.  The pair-map schema
 * deliberately separates the physical inclusion cutoff of each role from the
 * builder/dispatch strategy that produces the list (kSparse buckets versus the
 * kDense all-pairs fallback).  A 50-bohr D4 superset view may feed several D4
 * roles only when each consumer applies its own inclusive predicate.
 */
enum class Gfn2PairListRole : std::uint32_t {
  /* GFN2 coordination-number reduction over the 25-bohr reference cutoff. */
  kCoordination = 0u,
  /* D4 coordination number (erf form), 30 bohr physical cutoff. */
  kD4Coordination = 1u,
  /* D4 two-body dispersion, 50 bohr physical cutoff. */
  kD4TwoBody = 2u,
  /* D4 Axilrod--Teller--Muto, 25 bohr physical cutoff with ordered triples. */
  kD4Atm = 3u,
};

/*
 * Publication stage of a sparse list.  Candidate storage is unpublished
 * scratch owned by the numerical refresh transaction; the consumer validators
 * below accept only committed storage whose per-system generation and
 * eligibility match the current epoch.
 */
enum class Gfn2PairListState : std::uint32_t {
  kCandidate = 0u,
  kCommitted = 1u,
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
  kInvalidLayoutFingerprint = 20u,
  kInvalidPairListRole = 21u,
  kInvalidPairListState = 22u,
  kInsufficientPairListCutoff = 23u,
  kInvalidElementFingerprint = 24u,
  kInvalidProjection = 25u,
  kElementCountMismatch = 26u,
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
  kWavefunctionLayoutFingerprint = 25u,
  kPairListConsumer = 26u,
  kPairListOffsets = 27u,
  kPairListPairs = 28u,
  kPairListNeighborOffsets = 29u,
  kPairListNeighbors = 30u,
  kPairListGenerations = 31u,
  kPairListEligibleMask = 32u,
  kPairListActiveMask = 33u,
  kProjection = 34u,
  kElementIdentity = 35u,
  kElementFingerprint = 36u,
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
  /* Setup-only seal over the complete ordered spin packing. The scalar is
   * copied unchanged into device descriptors, so consumers can prove layout
   * identity without reading device metadata back to the host. */
  std::uint64_t layout_fingerprint = 0u;

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

/* Recompute the order-sensitive seal for a fully populated host descriptor.
 * The fingerprint intentionally excludes pointer values, memory_space, and
 * its own stored fingerprint so the same immutable arrays have one identity
 * in host, CUDA, and future HIP descriptors. Zero denotes an invalid input. */
[[nodiscard]] std::uint64_t gfn2_wavefunction_layout_fingerprint_host(
    const Gfn2WavefunctionLayoutView& layout) noexcept;

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

/*
 * Device-neutral projection of the sparse pair list that a consumer (CN, D4,
 * ...) reduces over.  It deliberately carries neither the builder strategy nor
 * any numerical value cache: pair indices, per-atom neighbor ranges, and
 * canonical ordering are the traversal contract: second endpoint ascending,
 * then first endpoint ascending, with first < second.  Each consumer re-derives
 * its own value cache from positions and its own inclusive cutoff predicate.
 * This keeps "pair-index traversal" and "numerical value cache" separate and
 * lets one 50-bohr superset list feed several D4 roles.
 *
 * Contract guarantees enforced by validation:
 *   - one plan token plus per-system committed generations; a consumer must
 *     never read a list from a different plan or geometry epoch;
 *   - eligible_mask (0/1 per system) marks peers whose committed list is
 *     usable for the current epoch; active_mask is optional caller intent,
 *     disjoint from every other array;
 *   - fixed-topology bucket/capacity metadata (max pairs per system, max
 *     neighbors per atom) is host-set at setup and never changes;
 *   - candidates and committed slices never alias topology; committed
 *     generations and eligible_mask never alias each other or the masks.
 *
 * All element counts are in elements, never bytes.
 */
struct Gfn2PairListConsumerView {
  Gfn2PlanMemorySpace memory_space = Gfn2PlanMemorySpace::kHost;
  Gfn2PairListState state = Gfn2PairListState::kCommitted;
  Gfn2PairListRole role = Gfn2PairListRole::kCoordination;
  Gfn2PairMapKind pair_map_kind = Gfn2PairMapKind::kExplicit;
  std::uint64_t plan_token = 0u;
  /* Physical inclusion cutoff in bohr for this role; validates both length and
   * that the list cutoff covers it (list_builder_cutoff_bohr >= cutoff_bohr).
   * Zero on a disabled leaf. */
  double cutoff_bohr = 0.0;
  double list_builder_cutoff_bohr = 0.0;

  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t max_pairs_per_system = 0;
  std::int64_t max_neighbors_per_atom = 0;

  /* Fixed-topology published arrays.  pair_offsets and neighbor_offsets name
   * the start of each system/atom slot; pair_counts and neighbor_counts name
   * the live prefix in that slot.  Compact CSR is valid, but publishers may
   * leave padding between slots so one failed peer never shifts another peer's
   * storage.  pair_count and neighbor_count are backing-array extents, not the
   * number of live entries. */
  std::int64_t pair_offset_count = 0;
  std::int64_t neighbor_offset_count = 0;
  std::int64_t pair_count = 0;
  std::int64_t neighbor_count = 0;
  const std::int64_t* pair_offsets = nullptr;
  const Gfn2AtomPair* pairs = nullptr;
  /* Explicit committed counts are validated independently from the compact
   * offsets, so a consumer does not trust neighboring prefix entries alone. */
  std::int64_t pair_count_elements = 0;
  std::int64_t neighbor_count_elements = 0;
  const std::int64_t* pair_counts = nullptr;
  const std::int64_t* neighbor_counts = nullptr;
  const std::int64_t* neighbor_offsets = nullptr;
  const std::int64_t* neighbors = nullptr;

  /* Per-system committed generations and eligibility, published host tensors. */
  std::int64_t committed_generation_count = 0;
  std::int64_t eligible_mask_count = 0;
  std::int64_t active_mask_count = 0;
  const std::uint64_t* committed_generations = nullptr;
  const std::uint8_t* eligible_mask = nullptr;
  const std::uint8_t* active_mask = nullptr;
};

/*
 * Device-neutral projections of the authoritative Gfn2RaggedTopologyView.
 *
 * A projection borrows a strict subset of the master topology's fields and
 * never allocates, transfers, or converts layout.  Projectors prove exact
 * pointer/count/token identity against the master view and fail closed, so a
 * consumer can never read a different plan's arrays even when the extents
 * happen to match.  The same struct layout is valid in host and CUDA-device
 * memory; only the setup-time validator used to inspect it differs.
 *
 * Physically different offset domains (shell Coulomb matrices, atom response
 * matrices, packed atom pairs, shell-pair launch grids) remain distinct and
 * are deliberately NOT projected here even when their element counts coincide.
 */
struct Gfn2AtomProjectionView {
  Gfn2PlanMemorySpace memory_space = Gfn2PlanMemorySpace::kHost;
  std::uint64_t plan_token = 0u;
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t atom_offset_count = 0;
  const std::int64_t* atom_offsets = nullptr;
};

struct Gfn2ShellOwnershipProjectionView {
  Gfn2PlanMemorySpace memory_space = Gfn2PlanMemorySpace::kHost;
  std::uint64_t plan_token = 0u;
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t batch_shell_offset_count = 0;
  std::int64_t atom_shell_offset_count = 0;
  std::int64_t shell_to_atom_count = 0;
  const std::int64_t* batch_shell_offsets = nullptr;
  const std::int64_t* atom_shell_offsets = nullptr;
  const std::int64_t* shell_to_atom = nullptr;
};

struct Gfn2AOMatrixProjectionView {
  Gfn2PlanMemorySpace memory_space = Gfn2PlanMemorySpace::kHost;
  std::uint64_t plan_token = 0u;
  std::int64_t batch_size = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t total_matrix_elements = 0;
  std::int64_t batch_orbital_offset_count = 0;
  std::int64_t matrix_offset_count = 0;
  std::int64_t shell_orbital_offset_count = 0;
  std::int64_t orbital_to_shell_count = 0;
  std::int64_t orbital_to_atom_count = 0;
  const std::int64_t* batch_orbital_offsets = nullptr;
  const std::int64_t* matrix_offsets = nullptr;
  const std::int64_t* shell_orbital_offsets = nullptr;
  const std::int64_t* orbital_to_shell = nullptr;
  const std::int64_t* orbital_to_atom = nullptr;
};

struct Gfn2PackedAllPairProjectionView {
  Gfn2PlanMemorySpace memory_space = Gfn2PlanMemorySpace::kHost;
  std::uint64_t plan_token = 0u;
  std::int64_t batch_size = 0;
  std::int64_t total_pairs = 0;
  std::int64_t pair_offset_count = 0;
  const std::int64_t* pair_offsets = nullptr;
};

struct Gfn2AOBucketProjectionView {
  Gfn2PlanMemorySpace memory_space = Gfn2PlanMemorySpace::kHost;
  std::uint64_t plan_token = 0u;
  std::int64_t batch_size = 0;
  std::int64_t bucket_count = 0;
  std::int64_t bucket_offset_count = 0;
  std::int64_t bucket_system_count = 0;
  std::int64_t bucket_orbital_count = 0;
  const std::int64_t* bucket_offsets = nullptr;
  const std::int32_t* bucket_systems = nullptr;
  const std::int32_t* bucket_orbital_counts = nullptr;
};

/*
 * Element identity is owned by setup, not by the topology: atomic numbers are
 * term-specific immutable input.  The projection therefore carries its own
 * order-sensitive fingerprint (a host setup seal like
 * gfn2_wavefunction_layout_fingerprint_host) plus the plan token, so a CUDA
 * consumer can prove it reads the same element ordering without pulling device
 * metadata back to the host.  Zero denotes an invalid fingerprint.
 */
struct Gfn2ElementIdentityProjectionView {
  Gfn2PlanMemorySpace memory_space = Gfn2PlanMemorySpace::kHost;
  std::uint64_t plan_token = 0u;
  std::int64_t total_atoms = 0;
  std::int64_t atomic_number_count = 0;
  std::uint64_t element_fingerprint = 0u;
  const std::int32_t* atomic_numbers = nullptr;
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
static_assert(std::is_trivially_copyable_v<Gfn2PairListConsumerView>);
static_assert(std::is_standard_layout_v<Gfn2PairListConsumerView>);
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

/*
 * Structural validation of a sparse pair-list consumer projection without
 * dereferencing any array.  Enforces one plan token, matching batch/atom
 * counts, a committed state, role/map-kind enumeration values, positive physical and
 * builder cutoffs with the builder cutoff covering the role cutoff, fixed
 * positive capacity metadata, exact offset counts vs batch/atoms, and full
 * pairwise address disjointness between every published array and immutable
 * topology.  The active mask is optional; when present it must be disjoint
 * from every other array.  host tuning constants are validated by the host
 * inspector, not here.
 */
[[nodiscard]] Gfn2PlanSchemaDiagnostic validate_gfn2_pair_list_consumer_binding(
    const Gfn2RaggedTopologyView& topology, const Gfn2PairListConsumerView& consumer,
    Gfn2PlanMemorySpace expected_memory_space) noexcept;

/* Full host inspection of every published pair/neighbor value, committed
 * generation, and eligible/active mask byte.  Offsets must form valid
 * partitions, pairs must store first < second within their system slice,
 * neighbors must stay inside their owning system, mask bytes must be
 * zero-or-one, and the eligible/active masks must not alias other
 * published arrays.  Batch-scope validation: committed_generations must be
 * the expected generation for every eligible system. */
[[nodiscard]] Gfn2PlanSchemaDiagnostic validate_gfn2_pair_list_consumer_host(
    const Gfn2RaggedTopologyView& topology, const Gfn2PairListConsumerView& consumer,
    std::uint64_t expected_geometry_generation) noexcept;

/*
 * Recompute the order-sensitive element-identity seal for a populated host
 * descriptor.  The fingerprint intentionally excludes pointer values,
 * memory_space, and its own stored fingerprint so the same immutable
 * atomic-number ordering has one identity in host, CUDA, and future HIP
 * descriptors.  Zero denotes an invalid input.
 */
[[nodiscard]] std::uint64_t gfn2_element_identity_fingerprint_host(
    const Gfn2ElementIdentityProjectionView& element) noexcept;

/*
 * Structural validation of a projection borrowed from the authoritative master
 * view.  Proves plan/memory-space/batch identity, exact required offset counts,
 * and exact pointer identity with the corresponding master arrays (fail
 * closed), without dereferencing any array in any address space.  A projection
 * must never name arrays owned by another plan, and hosts/CUDA copies of the
 * same projection must agree field-for-field.
 */
[[nodiscard]] Gfn2PlanSchemaDiagnostic validate_gfn2_atom_projection_binding(
    const Gfn2RaggedTopologyView& topology, const Gfn2AtomProjectionView& projection,
    Gfn2PlanMemorySpace expected_memory_space) noexcept;

[[nodiscard]] Gfn2PlanSchemaDiagnostic validate_gfn2_shell_ownership_projection_binding(
    const Gfn2RaggedTopologyView& topology, const Gfn2ShellOwnershipProjectionView& projection,
    Gfn2PlanMemorySpace expected_memory_space) noexcept;

[[nodiscard]] Gfn2PlanSchemaDiagnostic validate_gfn2_ao_matrix_projection_binding(
    const Gfn2RaggedTopologyView& topology, const Gfn2AOMatrixProjectionView& projection,
    Gfn2PlanMemorySpace expected_memory_space) noexcept;

[[nodiscard]] Gfn2PlanSchemaDiagnostic validate_gfn2_packed_all_pair_projection_binding(
    const Gfn2RaggedTopologyView& topology, const Gfn2PackedAllPairProjectionView& projection,
    Gfn2PlanMemorySpace expected_memory_space) noexcept;

[[nodiscard]] Gfn2PlanSchemaDiagnostic validate_gfn2_ao_bucket_projection_binding(
    const Gfn2RaggedTopologyView& topology, const Gfn2AOBucketProjectionView& projection,
    Gfn2PlanMemorySpace expected_memory_space) noexcept;

/*
 * Element identity is owned by setup (atomic numbers are term-specific
 * immutable input), so it validates against the authoritative host atomic
 * numbers and plan token rather than the master topology.  Proves
 * count/token identity and the exact order-sensitive fingerprint.
 */
[[nodiscard]] Gfn2PlanSchemaDiagnostic validate_gfn2_element_identity_projection_binding(
    const Gfn2ElementIdentityProjectionView& projection,
    Gfn2PlanMemorySpace expected_memory_space) noexcept;

/*
 * Fail-closed host builders for the atomic projections of a validated master
 * topology.  Each proves exact pointer identity with the master arrays and
 * clears `projection` unless complete validation succeeds.  They allocate
 * nothing and never convert layout.
 */
[[nodiscard]] Gfn2PlanSchemaDiagnostic project_gfn2_atom_projection_host(
    const Gfn2RaggedTopologyView& topology, Gfn2AtomProjectionView& projection) noexcept;

[[nodiscard]] Gfn2PlanSchemaDiagnostic project_gfn2_shell_ownership_projection_host(
    const Gfn2RaggedTopologyView& topology, Gfn2ShellOwnershipProjectionView& projection) noexcept;

[[nodiscard]] Gfn2PlanSchemaDiagnostic project_gfn2_ao_matrix_projection_host(
    const Gfn2RaggedTopologyView& topology, Gfn2AOMatrixProjectionView& projection) noexcept;

[[nodiscard]] Gfn2PlanSchemaDiagnostic project_gfn2_packed_all_pair_projection_host(
    const Gfn2RaggedTopologyView& topology, Gfn2PackedAllPairProjectionView& projection) noexcept;

[[nodiscard]] Gfn2PlanSchemaDiagnostic project_gfn2_ao_bucket_projection_host(
    const Gfn2RaggedTopologyView& topology, Gfn2AOBucketProjectionView& projection) noexcept;

/* Fail-closed builder for the setup-owned element identity projection. */
[[nodiscard]] Gfn2PlanSchemaDiagnostic project_gfn2_element_identity_projection_host(
    const std::int32_t* atomic_numbers, std::int64_t atomic_number_count, std::uint64_t plan_token,
    Gfn2ElementIdentityProjectionView& projection) noexcept;

}  // namespace gpuxtb::detail

#endif  // GPUXTB_BACKENDS_COMMON_GFN2_PLAN_SCHEMA_HPP
