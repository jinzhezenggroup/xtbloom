#ifndef GPUXTB_BACKENDS_CUDA_GFN2_PAIRLIST_CUH
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_BACKENDS_CUDA_GFN2_PAIRLIST_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/common/gfn2_plan_schema.hpp"

namespace gpuxtb::detail::cuda {

inline constexpr std::uint32_t kGfn2PairListAbiVersion = 1u;

/* Default sparse cutoff matching the GFN2 coordination/reference cutoffs. */
inline constexpr double kDefaultPairlistCutoffBohr = 25.0;

/*
 * Pair-list strategy for one fixed-topology batch.
 *
 * kSparse keeps only pairs whose squared distance is at most cutoff^2 and
 * builds them through per-system uniform buckets (cell list).  kDense is the
 * deterministic fallback: it retains every unordered in-system pair in the
 * canonical packed-triangle order, independent of any cutoff.  Both modes
 * publish the same layouts so a consumer can be written once; the dense mode
 * is selected for small systems from reproducible profiling, never guessed.
 */
enum class Gfn2PairListMode : std::uint32_t {
  kSparse = 0u,
  kDense = 1u,
};

/* First asynchronous semantic or numerical failure in a pair-list sequence. */
enum class Gfn2PairListDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kNonfinitePosition = 2u,
  kInvalidCutoff = 3u,
  kCellCapacityExceeded = 4u,
  kNeighborCapacityExceeded = 5u,
  kPairCapacityExceeded = 6u,
  kInvalidCache = 7u,
  kStaleGeometry = 8u,
  kCoincidentAtoms = 9u,
  kNonfiniteArithmetic = 10u,
};

/*
 * Per-system uniform bucket grid, resident in caller-owned workspace.
 * origin is the minimum corner; nx/ny/nz are the bucket counts; cells is the
 * product.  A system with at most one atom still owns one cell so the builder
 * has a well-defined stride.  cell_base offsets into the flat per-system cell
 * arrays (each system is provisioned max_cells_per_system cells).
 */
struct Gfn2PairListSystemMeta {
  double origin[3] = {0.0, 0.0, 0.0};
  std::int64_t cell_base = 0;
  std::int64_t nx = 1;
  std::int64_t ny = 1;
  std::int64_t nz = 1;
  std::int64_t cells = 1;
};

/*
 * Immutable, non-owning device topology.  atom_offsets use the standard ragged
 * batch partition.  cutoff is a pair inclusion radius in bohr (meaningful only
 * in kSparse mode but validated in both).  The three maxima are the documented
 * fixed-topology capacities; exceeding any of them fails that system closed
 * and never publishes a partial slice.
 */
struct Gfn2PairListDeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t atom_offset_elements = 0;
  double cutoff = 0.0;
  std::int64_t max_cells_per_system = 0;
  std::int64_t max_neighbors_per_atom = 0;
  std::int64_t max_pairs_per_system = 0;
  Gfn2PairListMode mode = Gfn2PairListMode::kSparse;
  std::uint64_t plan_token = 0u;
  const std::int64_t* atom_offsets = nullptr;
};

/*
 * Published per-system state.  pair_offsets partition the canonical
 * (first < second) pair list; neighbor_offsets partition the per-atom neighbor
 * ranges (each atom lists its own neighbors in ascending atom order, so a
 * consumer reproduces the dense packed-triangle reduction order for the
 * retained pairs).  pair_generations is per system.
 */
struct Gfn2PairListDeviceCache {
  Gfn2AtomPair* pairs = nullptr;
  std::int64_t pair_elements = 0;
  std::int64_t* pair_offsets = nullptr;
  std::int64_t pair_offset_elements = 0;
  std::int64_t* neighbor_offsets = nullptr;
  std::int64_t neighbor_offset_elements = 0;
  std::int64_t* neighbors = nullptr;
  std::int64_t neighbor_elements = 0;
  std::uint64_t* pair_generations = nullptr;
  std::int64_t generation_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Caller-owned unpublished storage; element counts are values, not bytes.
 * Cell arrays hold max_cells_per_system entries per system; neighbor scratch
 * holds max_neighbors_per_atom entries per atom; pair_cursor is one slot per
 * system for deterministic emit.
 */
struct Gfn2PairListDeviceWorkspace {
  Gfn2PairListSystemMeta* system_meta = nullptr;
  std::int64_t system_meta_elements = 0;
  std::int64_t* atom_cells = nullptr;
  std::int64_t atom_cell_elements = 0;
  std::int64_t* cell_counts = nullptr;
  std::int64_t cell_count_elements = 0;
  std::int64_t* cell_offsets = nullptr;
  std::int64_t cell_offset_elements = 0;
  std::int64_t* cell_fill = nullptr;
  std::int64_t cell_fill_elements = 0;
  std::int64_t* cell_atoms = nullptr;
  std::int64_t cell_atom_elements = 0;
  std::int64_t* neighbor_cursor = nullptr;
  std::int64_t neighbor_cursor_elements = 0;
  std::int64_t* neighbor_scratch = nullptr;
  std::int64_t neighbor_scratch_elements = 0;
  std::int64_t* pair_cursor = nullptr;
  std::int64_t pair_cursor_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2PairListSystemMeta>);
static_assert(std::is_standard_layout_v<Gfn2PairListSystemMeta>);
static_assert(std::is_trivially_copyable_v<Gfn2PairListDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2PairListDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2PairListDeviceCache>);
static_assert(std::is_standard_layout_v<Gfn2PairListDeviceCache>);
static_assert(std::is_trivially_copyable_v<Gfn2PairListDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2PairListDeviceWorkspace>);

/*
 * Host-side fixed-topology capacity query.  Given the batch/totals and the
 * documented per-system maxima it returns the exact element counts the
 * launchers require in cache and workspace, or false on overflow of int64.
 * This is the setup-time companion to the failure-isolated device overflow
 * detection: callers size once from fixed topology, then never reallocate.
 *
 * cache elements: pairs = batch*max_pairs; neighbor_offsets = total_atoms+1;
 * neighbor indices = total_atoms*max_neighbors_per_atom; offsets = batch+1;
 * generations = batch.
 * workspace elements: meta = batch; atom_cells/cell_atoms/neighbor_cursor =
 * total_atoms; each cell array (counts, offsets, fill) = batch*max_cells + 1
 * (offsets owns a trailing end slot); neighbor scratch =
 * total_atoms*max_neighbors_per_atom; pair_cursor = batch.
 */
[[nodiscard]] bool query_gfn2_pairlist_requirements_cuda(
    std::int64_t batch_size, std::int64_t total_atoms, std::int64_t max_cells_per_system,
    std::int64_t max_neighbors_per_atom, std::int64_t max_pairs_per_system,
    std::int64_t* cache_pairs, std::int64_t* cache_neighbor_offsets, std::int64_t* cache_neighbors,
    std::int64_t* cache_pair_offsets, std::int64_t* cache_generations, std::int64_t* ws_meta,
    std::int64_t* ws_atom_cells, std::int64_t* ws_cell_arrays, std::int64_t* ws_cell_atoms,
    std::int64_t* ws_neighbor_cursor, std::int64_t* ws_neighbor_scratch,
    std::int64_t* ws_pair_cursor) noexcept;

/* Clear per-system failures and the sequence-wide sticky first-error value. */
cudaError_t reset_gfn2_pairlist_device_errors_cuda(std::int64_t batch_size,
                                                   std::uint32_t* system_errors,
                                                   std::uint32_t* device_error,
                                                   cudaStream_t stream = nullptr) noexcept;

/*
 * Build the per-system bucketed neighbor/pair lists from atom-major xyz
 * positions in bohr and publish them transactionally.  Invalid device topology,
 * capacities, or a pre-existing device_error make the whole call a no-op;
 * numerical or capacity failure in one system does not prevent healthy peers
 * committing.  In kSparse mode only pairs with squared distance <= cutoff^2 are
 * retained; kDense retains the complete triangle.  The published pair list is
 * canonically ordered (first ascending, then second) and the per-atom neighbor
 * ranges are ascending, matching the dense packed-triangle reduction order so
 * downstream reductions reproduce dense results for the retained pairs.
 */
cudaError_t update_gfn2_pairlist_cache_cuda(const Gfn2PairListDeviceBatch& batch,
                                            const double* positions, std::uint64_t pair_generation,
                                            const Gfn2PairListDeviceCache& cache,
                                            const Gfn2PairListDeviceWorkspace& workspace,
                                            std::uint32_t* system_errors,
                                            std::uint32_t* device_error,
                                            cudaStream_t stream = nullptr) noexcept;

/*
 * Consume the published pair list to accumulate coordination numbers per atom
 * in the exact dense order (ascending neighbors), so sparse results equal the
 * dense geometry cache bitwise for the retained pairs.  radii are the
 * already-scaled GFN2 dexp CN radii.  The primitive validates the generation
 * pairing like the geometry VJP and never writes partial outputs for a failed
 * system.
 */
cudaError_t evaluate_gfn2_pairlist_coordination_cuda(
    const Gfn2PairListDeviceBatch& batch, const double* positions, const double* covalent_radii,
    std::uint64_t pair_generation, const Gfn2PairListDeviceCache& cache, double* coordination,
    const Gfn2PairListDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

/*
 * Deterministic host dispatch policy used to choose between the sparse
 * bucketed builder and the dense fallback.  The threshold is derived from
 * reproducible batch 1/8/32/128 profiling (issue #70); below it, the dense
 * path is selected because bucket overhead exceeds the all-pairs work it
 * saves.  Returns true when the sparse bucket build should be used for a
 * system of the given atom count.
 */
[[nodiscard]] bool gfn2_pairlist_use_sparse_for(std::int64_t atoms_per_system) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_PAIRLIST_CUH
