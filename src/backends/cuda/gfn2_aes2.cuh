#ifndef GPUXTB_BACKENDS_CUDA_GFN2_AES2_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_AES2_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

namespace gpuxtb::detail::cuda {

inline constexpr std::int64_t kGfn2AES2PairDataElements = 5;
inline constexpr std::int64_t kGfn2AES2PotentialElementsPerAtom = 10;

/* First semantic or arithmetic failure recorded for an AES2 batch member. */
enum class Gfn2AES2DeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kInvalidElementParameter = 2u,
  kNonfinitePosition = 3u,
  kInvalidCoordination = 4u,
  kCoordinateDifferenceOverflow = 5u,
  kInvalidGeometry = 6u,
  kNonfiniteKernelArithmetic = 7u,
  kNonfiniteMultipole = 8u,
  kInvalidCache = 9u,
  kNonfinitePotentialArithmetic = 10u,
  kNonfiniteEnergySeed = 11u,
  kNonfiniteEnergyArithmetic = 12u,
  kCacheMismatch = 13u,
  kNonfiniteGradientSeed = 14u,
  kNonfiniteVjpArithmetic = 15u,
};

/*
 * Flat device view uploaded from gfn2::AES2Plan accessors. Atom and pair
 * offsets are exact zero-based half-open ragged partitions. The four element
 * parameter arrays follow the global atom order and contain the GFN2 onsite
 * kernels and CN-dependent damping-radius parameters.
 *
 * plan_token is an opaque nonzero setup identity. Kernels never dereference
 * it; launchers compare it with the cache token so equal-sized storage from a
 * different plan cannot be consumed accidentally.
 */
struct Gfn2AES2DeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_pairs = 0;
  std::uint64_t plan_token = 0;

  std::int64_t atom_offset_count = 0;
  std::int64_t pair_offset_count = 0;
  std::int64_t dipole_kernel_count = 0;
  std::int64_t quadrupole_kernel_count = 0;
  std::int64_t multipole_radius_count = 0;
  std::int64_t multipole_valence_cn_count = 0;

  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* pair_offsets = nullptr;
  const double* dipole_kernel = nullptr;
  const double* quadrupole_kernel = nullptr;
  const double* multipole_radius = nullptr;
  const double* multipole_valence_cn = nullptr;
};

/*
 * Device-resident counterpart of gfn2::AES2GeometryCache. Each pair stores
 * [dx,dy,dz,R^-3*f3,R^-5*f5]. The caller binds geometry_generation before
 * enqueueing an update. Failed systems retain their previous pair slice and
 * remain disabled by their sticky system_errors entry.
 */
struct Gfn2AES2DeviceCache {
  double* pair_data = nullptr;
  std::int64_t pair_data_elements = 0;
  std::uint64_t geometry_generation = 0;
  std::uint64_t plan_token = 0;
};

/* Caller-owned unpublished device storage; all counts are doubles, not bytes. */
struct Gfn2AES2DeviceWorkspace {
  double* pair_scratch = nullptr;
  std::int64_t pair_elements = 0;
  double* potential_scratch = nullptr;
  std::int64_t potential_elements = 0;
  double* batch_scratch = nullptr;
  std::int64_t batch_elements = 0;
  double* gradient_scratch = nullptr;
  std::int64_t gradient_elements = 0;
  double* coordination_scratch = nullptr;
  std::int64_t coordination_elements = 0;
};

static_assert(std::is_trivially_copyable_v<Gfn2AES2DeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2AES2DeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2AES2DeviceCache>);
static_assert(std::is_standard_layout_v<Gfn2AES2DeviceCache>);
static_assert(std::is_trivially_copyable_v<Gfn2AES2DeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2AES2DeviceWorkspace>);

/*
 * Clear the caller-owned per-system errors and first-error diagnostic before
 * a dependent AES2 sequence. Later launchers never clear either buffer.
 */
cudaError_t reset_gfn2_aes2_device_errors_cuda(std::int64_t batch_size,
                                               std::uint32_t* system_errors,
                                               std::uint32_t* device_error,
                                               cudaStream_t stream = nullptr) noexcept;

/*
 * Update the compact pair cache from positions in bohr and GFN2 coordination
 * numbers. Publication is atomic per system: a failed member retains its old
 * cache slice while healthy peers commit normally.
 */
cudaError_t update_gfn2_aes2_geometry_cache_cuda(
    const Gfn2AES2DeviceBatch& batch, const double* positions, const double* coordination_numbers,
    const Gfn2AES2DeviceCache& cache, const Gfn2AES2DeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

/* Overwrite q/d/Q AES2 potentials using atom-major 1/3/6 packed layouts. */
cudaError_t evaluate_gfn2_aes2_potential_cuda(
    const Gfn2AES2DeviceBatch& batch, const Gfn2AES2DeviceCache& cache,
    const double* atomic_charges, const double* atomic_dipoles, const double* atomic_quadrupoles,
    double* charge_potentials, double* dipole_potentials, double* quadrupole_potentials,
    const Gfn2AES2DeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

/* Accumulate one complete AES2 energy into each existing per-system seed. */
cudaError_t add_gfn2_aes2_energy_cuda(const Gfn2AES2DeviceBatch& batch,
                                      const Gfn2AES2DeviceCache& cache,
                                      const double* atomic_charges, const double* atomic_dipoles,
                                      const double* atomic_quadrupoles, double* energies,
                                      const Gfn2AES2DeviceWorkspace& workspace,
                                      std::uint32_t* system_errors, std::uint32_t* device_error,
                                      cudaStream_t stream = nullptr) noexcept;

/*
 * Accumulate the fixed-multipole coordinate VJP and the damping-radius CN
 * adjoint. geometry_generation binds positions/CN to the active cache. The
 * outputs are derivatives dE/dR and dE/dCN, not forces.
 */
cudaError_t add_gfn2_aes2_vjp_cuda(const Gfn2AES2DeviceBatch& batch,
                                   const Gfn2AES2DeviceCache& cache, const double* positions,
                                   const double* coordination_numbers,
                                   std::uint64_t geometry_generation, const double* atomic_charges,
                                   const double* atomic_dipoles, const double* atomic_quadrupoles,
                                   double* gradients, double* coordination_adjoints,
                                   const Gfn2AES2DeviceWorkspace& workspace,
                                   std::uint32_t* system_errors, std::uint32_t* device_error,
                                   cudaStream_t stream = nullptr) noexcept;

/*
 * All launchers are allocation-free, enqueue exclusively on the supplied
 * stream, perform no synchronization, and support CUDA Graph capture. Pointer
 * provenance is a setup/binding contract; hot launchers deliberately avoid
 * cudaPointerGetAttributes. Writable ranges must be mutually disjoint.
 */

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_AES2_CUH
