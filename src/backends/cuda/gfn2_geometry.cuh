#ifndef GPUXTB_BACKENDS_CUDA_GFN2_GEOMETRY_CUH
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_BACKENDS_CUDA_GFN2_GEOMETRY_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

namespace gpuxtb::detail::cuda {

/*
 * Per unordered atom pair, in lower-triangle order, the cache stores
 *
 *   [dx, dy, dz, distance, inverse_distance, cn_count, dcn_dr / distance].
 *
 * The displacement is R_higher - R_lower. Multiplying its last component by
 * the displacement therefore gives the Cartesian derivative of cn_count with
 * respect to R_higher. Consumers can also form a unit direction as dR/r.
 */
inline constexpr std::int64_t kGfn2GeometryPairDataElements = 7;

/*
 * Stable device-resident epoch shared by one fixed-topology execution DAG.
 * The execution owner initializes value once, then the preprocessing head
 * advances it exactly once per inference. Every downstream cache publisher
 * reads the same value on the caller stream, including during CUDA Graph
 * replay, so changed coordinates can never retain a capture-time generation.
 *
 * Concurrent inference through one descriptor is forbidden by the runtime's
 * single-flight contract. Cross-stream reuse requires explicit event ordering.
 */
struct Gfn2GeometryEpochDevice {
  std::uint64_t* value = nullptr;
  std::int64_t value_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Immutable consumer projection for one runtime-owned numerical transaction.
 *
 * epoch is the single device value advanced by preprocessing.  The two
 * peer-major arrays are published by the terminal numerical-refresh gate: a
 * consumer may read a member's numerical cache only when eligible_mask is one
 * and committed_generations equals the current epoch.  Keeping this provenance
 * device-resident avoids capture-time scalar generations during CUDA Graph
 * replay while preserving mixed-peer rollback.
 */
struct Gfn2GeometryEpochConsumerDevice {
  Gfn2GeometryEpochDevice epoch{};
  const std::uint64_t* committed_generations = nullptr;
  const std::uint8_t* eligible_mask = nullptr;
  std::int64_t batch_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* First asynchronous semantic or numerical failure in a geometry sequence. */
enum class Gfn2GeometryDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kInvalidCovalentRadius = 2u,
  kNonfinitePosition = 3u,
  kCoordinateDifferenceOverflow = 4u,
  kCoincidentAtoms = 5u,
  kNonfinitePairArithmetic = 6u,
  kNonfiniteCoordinationArithmetic = 7u,
  kInvalidCache = 8u,
  kStaleGeometry = 9u,
  kNonfiniteAdjoint = 10u,
  kNonfiniteGradientSeed = 11u,
  kNonfiniteVjpArithmetic = 12u,
};

/*
 * Immutable, non-owning device topology for a ragged batch. pair_offsets use
 * the same system partition as atom_offsets and each system contains exactly
 * n*(n-1)/2 pairs. covalent_radii are the already-scaled GFN2 dexp CN radii
 * uploaded from gfn2::CoordinationPlan.
 *
 * Counts are host-prevalidated capacities. Device preflight still validates
 * every offset value before any kernel subtracts or indexes through it.
 */
struct Gfn2GeometryDeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_pairs = 0;
  std::int64_t atom_offset_elements = 0;
  std::int64_t pair_offset_elements = 0;
  std::int64_t covalent_radius_elements = 0;
  std::int64_t coordinate_elements = 0;
  std::uint64_t plan_token = 0u;
  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* pair_offsets = nullptr;
  const double* covalent_radii = nullptr;
};

/*
 * Published geometry state. geometry_generations is per system, so a failed
 * member retains its previous pair/CN slice and provenance while healthy
 * peers commit the requested generation.
 */
struct Gfn2GeometryDeviceCache {
  double* pair_data = nullptr;
  std::int64_t pair_data_elements = 0;
  double* coordination_numbers = nullptr;
  std::int64_t coordination_elements = 0;
  std::uint64_t* geometry_generations = nullptr;
  std::int64_t generation_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Caller-owned unpublished storage; element counts are values, not bytes. */
struct Gfn2GeometryDeviceWorkspace {
  double* pair_scratch = nullptr;
  std::int64_t pair_elements = 0;
  double* coordination_scratch = nullptr;
  std::int64_t coordination_elements = 0;
  double* gradient_scratch = nullptr;
  std::int64_t gradient_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2GeometryDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2GeometryDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2GeometryEpochDevice>);
static_assert(std::is_standard_layout_v<Gfn2GeometryEpochDevice>);
static_assert(std::is_trivially_copyable_v<Gfn2GeometryEpochConsumerDevice>);
static_assert(std::is_standard_layout_v<Gfn2GeometryEpochConsumerDevice>);
static_assert(std::is_trivially_copyable_v<Gfn2GeometryDeviceCache>);
static_assert(std::is_standard_layout_v<Gfn2GeometryDeviceCache>);
static_assert(std::is_trivially_copyable_v<Gfn2GeometryDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2GeometryDeviceWorkspace>);

/* Clear per-system failures and the sequence-wide sticky first-error value. */
cudaError_t reset_gfn2_geometry_device_errors_cuda(std::int64_t batch_size,
                                                   std::uint32_t* system_errors,
                                                   std::uint32_t* device_error,
                                                   cudaStream_t stream = nullptr) noexcept;

/*
 * Build pair geometry and GFN2 coordination numbers from atom-major xyz
 * positions in bohr. Publication is transactional per system. Invalid device
 * topology or a pre-existing device_error makes the whole call a no-op;
 * numerical failure in one system does not prevent healthy peers committing.
 */
cudaError_t update_gfn2_geometry_cache_cuda(
    const Gfn2GeometryDeviceBatch& batch, const double* positions,
    std::uint64_t geometry_generation, const Gfn2GeometryDeviceCache& cache,
    const Gfn2GeometryDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

/*
 * Accumulate
 *
 *   gradients += (d coordination_numbers / d positions)^T * dE_dcn.
 *
 * The primitive consumes only the generation-bound pair cache; it does not
 * rescan positions or materialize a Jacobian. Gradients are dE/dR, not forces.
 * Failed systems retain their input gradients while healthy peers publish.
 */
cudaError_t add_gfn2_coordination_vjp_cuda(
    const Gfn2GeometryDeviceBatch& batch, const Gfn2GeometryDeviceCache& cache,
    std::uint64_t geometry_generation, const double* dE_dcn, double* gradients,
    const Gfn2GeometryDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

/* Replay-safe VJP consuming the current runtime-owned device epoch. */
cudaError_t add_gfn2_coordination_vjp_cuda(
    const Gfn2GeometryDeviceBatch& batch, const Gfn2GeometryDeviceCache& cache,
    const Gfn2GeometryEpochDevice& geometry_epoch, const double* dE_dcn, double* gradients,
    const Gfn2GeometryDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

/*
 * Launchers allocate nothing, transfer nothing, synchronize nowhere, enqueue
 * only on stream, and are CUDA Graph capture compatible. Setup must establish
 * pointer provenance; active writable ranges must be mutually disjoint from
 * immutable inputs and from one another.
 */

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_GEOMETRY_CUH
