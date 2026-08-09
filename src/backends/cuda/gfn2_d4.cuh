#ifndef GPUXTB_BACKENDS_CUDA_GFN2_D4_CUH
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_BACKENDS_CUDA_GFN2_D4_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_scc_iteration_control.cuh"

namespace gpuxtb::detail::cuda {

inline constexpr std::int64_t kGfn2D4MaximumReferences = 7;
inline constexpr std::int64_t kGfn2D4PairDataElements = 5;

/* Device copies of the pinned dftd4 parameter records. */
struct Gfn2D4DeviceElementData {
  std::uint16_t reference_offset;
  std::uint8_t reference_count;
  double covalent_radius;
  double electronegativity;
  double effective_charge;
  double hardness;
  double r4r2;
};

struct Gfn2D4DeviceReferenceData {
  double coordination_number;
  double charge;
  std::uint8_t gaussian_count;
};

/*
 * Non-owning device view of the immutable D4 tables. Large reference-C6 data
 * deliberately remains in global memory instead of consuming scarce constant
 * memory. Upload is a setup operation; repeated SCC evaluations allocate and
 * transfer nothing.
 */
struct Gfn2D4DeviceParameters {
  const Gfn2D4DeviceElementData* elements = nullptr;
  std::int64_t element_count = 0;
  const Gfn2D4DeviceReferenceData* references = nullptr;
  std::int64_t reference_count = 0;
  const double* reference_c6 = nullptr;
  std::int64_t reference_c6_elements = 0;
};

/*
 * CUDA counterpart of D4Plan's ragged topology. atomic_numbers is supplied by
 * the caller because D4Plan intentionally does not expose its private element
 * indices. It must preserve the exact ordering passed to make_d4_plan, and
 * atomic_number_hash must be computed from that same host array with the
 * helper below. All pointers address CUDA-accessible memory.
 */
struct Gfn2D4DeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_pairs = 0;
  std::uint64_t plan_token = 0;
  std::uint64_t atomic_number_hash = 0;
  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* pair_offsets = nullptr;
  const std::int32_t* atomic_numbers = nullptr;
  /*
   * Conservative host-side dispatch metadata. A zero value disables topology-
   * dependent optimizations; setup records the exact minimum over atom_offsets.
   * Device kernels still use the canonical offsets for all numerical indexing.
   */
  std::int64_t minimum_atoms_per_system = 0;
};

/*
 * Same packed pair/CN layout as gfn2::D4GeometryCache, but device-resident.
 * pair_data may be NULL exactly when pair_data_elements is zero.
 */
struct Gfn2D4DeviceCache {
  const double* pair_data = nullptr;
  std::int64_t pair_data_elements = 0;
  const double* coordination_numbers = nullptr;
  std::int64_t coordination_elements = 0;
  std::uint64_t geometry_generation = 0;
  std::uint64_t plan_token = 0;
};

/* Caller-owned device scratch. Numerical counts are doubles rather than bytes. */
struct Gfn2D4DeviceWorkspace {
  double* weights = nullptr;
  double* weight_cn_derivatives = nullptr;
  double* weight_charge_derivatives = nullptr;
  std::int64_t weight_elements = 0;
  double* atom_scratch = nullptr;
  double* coordination_adjoints = nullptr;
  std::int64_t atom_elements = 0;
  double* batch_scratch = nullptr;
  std::int64_t batch_elements = 0;
  double* gradient_scratch = nullptr;
  std::int64_t gradient_elements = 0;
  /* One sticky Gfn2D4DeviceError per ragged batch member. */
  std::uint32_t* system_errors = nullptr;
  std::int64_t system_error_elements = 0;

  /*
   * Unpublished changed-geometry state. These fields are used only by
   * update_gfn2_d4_geometry_cache_cuda; the energy/gradient launchers above
   * neither require nor inspect them. Keeping the storage caller-owned makes
   * repeated refreshes allocation-free and CUDA Graph capture safe.
   */
  double* pair_scratch = nullptr;
  std::int64_t pair_scratch_elements = 0;
  double* coordination_scratch = nullptr;
  std::int64_t coordination_scratch_elements = 0;

  /*
   * geometry_generations is published per system after its pair and CN slices
   * have committed. A failed peer therefore retains both its old numerical
   * cache and its old generation even though Gfn2D4DeviceCache keeps the
   * legacy scalar generation required by existing consumers.
   */
  std::uint64_t* geometry_generations = nullptr;
  std::int64_t geometry_generation_elements = 0;

  /* One internal stream-ordered gate snapshot; values are 0 or 1. */
  std::uint32_t* geometry_sequence_active = nullptr;
  std::int64_t geometry_sequence_elements = 0;
};

enum class Gfn2D4DeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kInvalidAtomicNumber = 2u,
  kInvalidParameterData = 3u,
  kInvalidCoordination = 4u,
  kNonfiniteCharge = 5u,
  kInvalidDamping = 6u,
  kNonfiniteArithmetic = 7u,
  kInvalidActivity = 8u,
  kStaleGeometry = 9u,
  kNonfinitePosition = 10u,
  kCoordinateDifferenceOverflow = 11u,
  kCoincidentAtoms = 12u,
  kNonfiniteGeometryArithmetic = 13u,
};

/* SplitMix64 finalizer used by the order-sensitive, parallelizable fingerprint. */
inline __host__ __device__ std::uint64_t gfn2_d4_hash_mix(std::uint64_t value) noexcept {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31u);
}

inline __host__ __device__ std::uint64_t gfn2_d4_atomic_number_hash_contribution(
    std::int32_t atomic_number, std::int64_t atom) noexcept {
  const std::uint64_t indexed =
      gfn2_d4_hash_mix(static_cast<std::uint64_t>(atom) + 0x9e3779b97f4a7c15ULL);
  return gfn2_d4_hash_mix(indexed ^ static_cast<std::uint32_t>(atomic_number));
}

/* Stable setup-time fingerprint used to reject a reordered element array. */
inline std::uint64_t gfn2_d4_atomic_number_hash(const std::int32_t* atomic_numbers,
                                                std::int64_t total_atoms) noexcept {
  if (atomic_numbers == nullptr || total_atoms < 0) {
    return 0u;
  }
  std::uint64_t hash =
      0x243f6a8885a308d3ULL ^ gfn2_d4_hash_mix(static_cast<std::uint64_t>(total_atoms));
  for (std::int64_t atom = 0; atom < total_atoms; ++atom) {
    hash ^= gfn2_d4_atomic_number_hash_contribution(atomic_numbers[atom], atom);
  }
  return hash;
}

static_assert(std::is_trivially_copyable_v<Gfn2D4DeviceElementData>);
static_assert(std::is_standard_layout_v<Gfn2D4DeviceElementData>);
static_assert(std::is_trivially_copyable_v<Gfn2D4DeviceReferenceData>);
static_assert(std::is_standard_layout_v<Gfn2D4DeviceReferenceData>);
static_assert(std::is_trivially_copyable_v<Gfn2D4DeviceParameters>);
static_assert(std::is_trivially_copyable_v<Gfn2D4DeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2D4DeviceCache>);
static_assert(std::is_trivially_copyable_v<Gfn2D4DeviceWorkspace>);

/* Clear per-system numerical status and the sequence-level topology status. */
cudaError_t reset_gfn2_d4_device_errors_cuda(std::int64_t batch_size, std::uint32_t* system_errors,
                                             std::uint32_t* device_error,
                                             cudaStream_t stream = nullptr) noexcept;

/*
 * Rebuild the D4 pair/CN cache directly from atom-major device positions in
 * bohr. cache.geometry_generation is the requested nonzero generation and
 * must already be bound to the generation argument used by downstream D4
 * consumers. The cache retains its legacy read-only pointer view; setup must
 * bind those pointers to writable CUDA allocations for this update call.
 *
 * Publication is transactional per ragged system. Pair data and coordination
 * numbers are first formed in workspace scratch, healthy peers then publish
 * their numerical slices, and only a later kernel publishes their entry in
 * workspace.geometry_generations. A peer-local numerical failure leaves all
 * three old slices unchanged; immutable topology/parameter failure closes the
 * whole sequence through device_error. The launcher allocates, transfers,
 * polls, and synchronizes nowhere and is safe on custom streams and in CUDA
 * Graph capture.
 */
cudaError_t update_gfn2_d4_geometry_cache_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    const double* positions, const Gfn2D4DeviceCache& cache, const Gfn2D4DeviceWorkspace& workspace,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

/*
 * Overwrite one two-body energy per system and dE_D4/dq per atom. Inputs,
 * outputs, topology, parameters, and scratch remain on device. The launch is
 * allocation-free, synchronization-free, custom-stream safe, and suitable for
 * CUDA Graph capture. Numerical failures are sticky per system: healthy peers
 * publish normally, while a failed member keeps its previous outputs.
 * Topology/provenance failures use device_error and disable the whole
 * sequence. Every writable range must be disjoint from every input and every
 * other writable range; read-only input ranges may alias one another.
 */
cudaError_t evaluate_gfn2_d4_two_body_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    const Gfn2D4DeviceCache& cache, const double* atomic_charges, double* energies,
    double* atomic_potentials, const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

/*
 * Evaluate only the self-consistent D4 charge derivative used by the SCC
 * Hamiltonian. The weights are prepared from mixed_atomic_charges; no energy
 * is formed or published. Inactive systems are skipped before cache or charge
 * values are read. A stale scalar cache generation is a plan failure only
 * when at least one system is active.
 *
 * Only workspace.weights, workspace.weight_charge_derivatives,
 * workspace.atom_scratch, and workspace.system_errors are required or
 * accessed. The other workspace fields may be null with zero extents.
 */
cudaError_t evaluate_gfn2_d4_scc_potential_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    const Gfn2D4DeviceCache& cache, std::uint64_t expected_geometry_generation,
    const double* mixed_atomic_charges, const Gfn2SccIterationDeviceActivity& activity,
    double* atomic_potentials, const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

/*
 * Evaluate only the pure self-consistent D4 two-body energy used by the final
 * SCC functional. The weights are prepared from raw_atomic_charges; charge
 * derivatives and atomic potentials are neither computed nor inspected.
 *
 * Only workspace.weights, workspace.batch_scratch, and
 * workspace.system_errors are required or accessed. The other workspace
 * fields may be null with zero extents.
 */
cudaError_t evaluate_gfn2_d4_scc_energy_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    const Gfn2D4DeviceCache& cache, std::uint64_t expected_geometry_generation,
    const double* raw_atomic_charges, const Gfn2SccIterationDeviceActivity& activity,
    double* energies, const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

/*
 * Accumulate the complete self-consistent two-body coordinate derivative at
 * fixed charges. The unpublished gradient delta and CN adjoints live in the
 * caller-owned workspace, so a failed member leaves its gradient slice
 * unchanged without suppressing healthy peers.
 */
cudaError_t add_gfn2_d4_two_body_gradient_cuda(const Gfn2D4DeviceBatch& batch,
                                               const Gfn2D4DeviceParameters& parameters,
                                               const Gfn2D4DeviceCache& cache,
                                               const double* atomic_charges, double* gradients,
                                               const Gfn2D4DeviceWorkspace& workspace,
                                               std::uint32_t* device_error,
                                               cudaStream_t stream = nullptr) noexcept;

/*
 * Overwrite one q=0 Axilrod--Teller--Muto energy per system. This is the GFN2
 * non-self-consistent three-body term and deliberately has no charge VJP.
 */
cudaError_t evaluate_gfn2_d4_atm_cuda(const Gfn2D4DeviceBatch& batch,
                                      const Gfn2D4DeviceParameters& parameters,
                                      const Gfn2D4DeviceCache& cache, double* energies,
                                      const Gfn2D4DeviceWorkspace& workspace,
                                      std::uint32_t* device_error,
                                      cudaStream_t stream = nullptr) noexcept;

/* Accumulate the analytic ATM coordinate derivative, including its CN VJP. */
cudaError_t add_gfn2_d4_atm_gradient_cuda(const Gfn2D4DeviceBatch& batch,
                                          const Gfn2D4DeviceParameters& parameters,
                                          const Gfn2D4DeviceCache& cache, double* gradients,
                                          const Gfn2D4DeviceWorkspace& workspace,
                                          std::uint32_t* device_error,
                                          cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_D4_CUH
