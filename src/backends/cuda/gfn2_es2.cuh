#ifndef GPUXTB_BACKENDS_CUDA_GFN2_ES2_CUH
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_BACKENDS_CUDA_GFN2_ES2_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_scc_iteration_control.cuh"

namespace gpuxtb::detail::cuda {

/* First semantic or arithmetic failure recorded by an ES2 device sequence. */
enum class Gfn2ES2DeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kInvalidShellMetadata = 2u,
  kNonfinitePosition = 3u,
  kCoordinateDifferenceOverflow = 4u,
  kNonfiniteHardnessArithmetic = 5u,
  kNonfiniteKernelArithmetic = 6u,
  kNonfiniteShellCharge = 7u,
  kInvalidCacheMatrix = 8u,
  kNonfinitePotentialArithmetic = 9u,
  kNonfiniteEnergySeed = 10u,
  kNonfiniteEnergyArithmetic = 11u,
  kNonfiniteGradientSeed = 12u,
  kNonfiniteGradientArithmetic = 13u,
};

/*
 * Flat, non-owning device topology uploaded explicitly from ES2Plan accessors.
 * Every pointer addresses CUDA-accessible caller storage. Counts are exact,
 * not capacities, and all offsets are zero-based half-open ragged partitions.
 *
 * plan_token is an opaque nonzero caller identity. Kernels never dereference
 * it; launchers use equality with Gfn2ES2DeviceCache::plan_token to prevent a
 * same-extent cache from being reused for another uploaded topology.
 */
struct Gfn2ES2DeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_matrix_elements = 0;
  std::uint64_t plan_token = 0;

  std::int64_t atom_offset_count = 0;
  std::int64_t batch_shell_offset_count = 0;
  std::int64_t atom_shell_offset_count = 0;
  std::int64_t matrix_offset_count = 0;
  std::int64_t shell_to_atom_count = 0;
  std::int64_t shell_hardness_count = 0;

  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* batch_shell_offsets = nullptr;
  const std::int64_t* atom_shell_offsets = nullptr;
  const std::int64_t* matrix_offsets = nullptr;
  const std::int64_t* shell_to_atom = nullptr;
  const double* shell_hardness = nullptr;
};

/*
 * Declarative cache binding. The caller chooses metadata before enqueueing an
 * update. It becomes usable only if the stream later reports device_error ==
 * kSuccess. A failed update leaves coulomb_matrix unchanged but does not roll
 * back this host descriptor. geometry_generation is compared by value for
 * coordinate VJPs; plan_token is compared by value for every cache consumer.
 */
struct Gfn2ES2DeviceCache {
  double* coulomb_matrix = nullptr;
  std::int64_t matrix_elements = 0;
  std::uint64_t geometry_generation = 0;
  std::uint64_t plan_token = 0;
};

/* Caller-owned unpublished device scratch; counts are doubles, not bytes. */
struct Gfn2ES2DeviceWorkspace {
  double* matrix_scratch = nullptr;
  std::int64_t matrix_elements = 0;
  double* shell_scratch = nullptr;
  std::int64_t shell_elements = 0;
  double* batch_scratch = nullptr;
  std::int64_t batch_elements = 0;
  double* gradient_scratch = nullptr;
  std::int64_t gradient_elements = 0;
};

static_assert(std::is_trivially_copyable_v<Gfn2ES2DeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2ES2DeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2ES2DeviceCache>);
static_assert(std::is_standard_layout_v<Gfn2ES2DeviceCache>);
static_assert(std::is_trivially_copyable_v<Gfn2ES2DeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2ES2DeviceWorkspace>);

/*
 * Preflight and overwrite Gamma in cache.coulomb_matrix. For different atoms,
 * Gamma_st = [R_AB^2 + gamma_st^-2]^-1/2 with arithmetic gamma averaging;
 * same-atom entries equal the arithmetic average. No public matrix element is
 * written unless every system passes topology, finite-input, and arithmetic
 * preflight.
 */
cudaError_t update_gfn2_es2_geometry_cache_cuda(const Gfn2ES2DeviceBatch& batch,
                                                const double* positions,
                                                const Gfn2ES2DeviceCache& cache,
                                                const Gfn2ES2DeviceWorkspace& workspace,
                                                std::uint32_t* device_error,
                                                cudaStream_t stream = nullptr) noexcept;

/* Preflight and overwrite v = Gamma*q for the complete ragged batch. */
cudaError_t evaluate_gfn2_es2_potential_cuda(const Gfn2ES2DeviceBatch& batch,
                                             const Gfn2ES2DeviceCache& cache,
                                             const double* shell_charges, double* shell_potentials,
                                             const Gfn2ES2DeviceWorkspace& workspace,
                                             std::uint32_t* device_error,
                                             cudaStream_t stream = nullptr) noexcept;

/* Preflight and accumulate E += q^T Gamma q / 2, one value per system. */
cudaError_t add_gfn2_es2_energy_cuda(const Gfn2ES2DeviceBatch& batch,
                                     const Gfn2ES2DeviceCache& cache, const double* shell_charges,
                                     double* energies, const Gfn2ES2DeviceWorkspace& workspace,
                                     std::uint32_t* device_error,
                                     cudaStream_t stream = nullptr) noexcept;

/*
 * Preflight and accumulate the fixed-q coordinate VJP. The supplied generation
 * must equal cache.geometry_generation. No gradient coordinate is published
 * unless the complete batch and every existing gradient seed pass preflight.
 */
cudaError_t add_gfn2_es2_gradient_cuda(const Gfn2ES2DeviceBatch& batch,
                                       const Gfn2ES2DeviceCache& cache, const double* positions,
                                       std::uint64_t geometry_generation,
                                       const double* shell_charges, double* gradients,
                                       const Gfn2ES2DeviceWorkspace& workspace,
                                       std::uint32_t* device_error,
                                       cudaStream_t stream = nullptr) noexcept;

/*
 * Reset once before a dependent ES2 sequence. Launchers never clear the error:
 * the first device semantic error is sticky, later stages become no-ops, and
 * scratch may be discarded. All functions are allocation-free, custom-stream
 * only, synchronization-free, and CUDA Graph capture compatible.
 */
cudaError_t reset_gfn2_es2_device_error_cuda(std::uint32_t* device_error,
                                             cudaStream_t stream = nullptr) noexcept;

/*
 * Reset the SCC-local peer diagnostics and plan-only latch.  The canonical
 * iteration activity is read-only; a later normalization stage folds these
 * diagnostics into that ledger without mixing ES2 codes with other domains.
 */
cudaError_t reset_gfn2_es2_scc_errors_cuda(std::int64_t batch_size, std::uint32_t* system_errors,
                                           std::uint32_t* plan_error,
                                           cudaStream_t stream = nullptr) noexcept;

/*
 * Evaluate the mixed-charge ES2 potential for active SCC members.  Inactive
 * members are rejected before any charge or Gamma-cache read and retain their
 * output bytes.  Numerical failures are isolated per system; malformed device
 * offset partitions set only plan_error and suppress publication for the
 * complete stage.
 */
cudaError_t evaluate_gfn2_es2_scc_potential_cuda(
    const Gfn2ES2DeviceBatch& batch, const Gfn2ES2DeviceCache& cache,
    std::uint64_t geometry_generation, const Gfn2SccIterationDeviceActivity& activity,
    const double* mixed_shell_charges, double* shell_potentials,
    const Gfn2ES2DeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* plan_error, cudaStream_t stream = nullptr) noexcept;

/*
 * Evaluate the pure raw-charge ES2 component and overwrite one value per
 * active system.  The destination is never read: each successful value starts
 * from exact zero and is published transactionally after the complete system
 * calculation succeeds.  The same charge-independent Gamma cache can be
 * reused by this raw phase and the mixed-potential phase above.
 */
cudaError_t evaluate_gfn2_es2_scc_energy_cuda(
    const Gfn2ES2DeviceBatch& batch, const Gfn2ES2DeviceCache& cache,
    std::uint64_t geometry_generation, const Gfn2SccIterationDeviceActivity& activity,
    const double* raw_shell_charges, double* component_energies,
    const Gfn2ES2DeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* plan_error, cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_ES2_CUH
