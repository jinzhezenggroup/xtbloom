#ifndef GPUXTB_BACKENDS_CUDA_GFN2_MULLIKEN_CUH
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_BACKENDS_CUDA_GFN2_MULLIKEN_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/common/gfn2_plan_schema.hpp"

namespace gpuxtb::detail::cuda {

inline constexpr std::int64_t kGfn2MullikenDipoleComponents = 3;
inline constexpr std::int64_t kGfn2MullikenQuadrupoleComponents = 6;

/* First asynchronous semantic or arithmetic failure in a Mulliken sequence. */
enum class Gfn2MullikenDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kInvalidShellMetadata = 2u,
  kInvalidActiveMask = 3u,
  kNonfiniteDensity = 4u,
  kNonfiniteIntegral = 5u,
  kNonfiniteReferenceOccupation = 6u,
  kNonfiniteContraction = 7u,
  kNonfinitePopulationReduction = 8u,
  kInvalidSpinChannels = 9u,
  kNonfiniteSpinConversion = 10u,
};

/*
 * Restricted (one-spin-channel) ragged Mulliken topology. Matrix offsets own
 * system-major row-major square AO matrices. Multipole integrals use the
 * global component-major layout produced by evaluate_gfn2_integrals_cuda.
 * Atom shells and shell orbitals must be nonempty contiguous partitions.
 */
struct Gfn2MullikenDeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t total_matrix_elements = 0;
  std::int64_t maximum_system_atoms = 0;
  std::int64_t maximum_system_shells = 0;
  std::uint64_t plan_token = 0u;

  std::int64_t atom_offset_count = 0;
  std::int64_t batch_shell_offset_count = 0;
  std::int64_t batch_orbital_offset_count = 0;
  std::int64_t matrix_offset_count = 0;
  std::int64_t atom_shell_offset_count = 0;
  std::int64_t shell_orbital_offset_count = 0;
  std::int64_t shell_to_atom_count = 0;
  std::int64_t reference_occupation_count = 0;

  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* batch_shell_offsets = nullptr;
  const std::int64_t* batch_orbital_offsets = nullptr;
  const std::int64_t* matrix_offsets = nullptr;
  const std::int64_t* atom_shell_offsets = nullptr;
  const std::int64_t* shell_orbital_offsets = nullptr;
  const std::int64_t* shell_to_atom = nullptr;
  const double* reference_shell_occupations = nullptr;
};

/* All pointers are device/UVA addresses bound to the same immutable plan. */
struct Gfn2MullikenDeviceInput {
  const double* density = nullptr;
  std::int64_t density_elements = 0;
  const double* overlap = nullptr;
  std::int64_t overlap_elements = 0;
  const double* dipole_integrals = nullptr;
  std::int64_t dipole_integral_elements = 0;
  const double* quadrupole_integrals = nullptr;
  std::int64_t quadrupole_integral_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Restricted outputs: qsh[S], qat[A], dipole[3*A], quadrupole[6*A]. */
struct Gfn2MullikenDevicePopulation {
  double* qsh = nullptr;
  std::int64_t qsh_elements = 0;
  double* qat = nullptr;
  std::int64_t qat_elements = 0;
  double* dipole = nullptr;
  std::int64_t dipole_elements = 0;
  double* quadrupole = nullptr;
  std::int64_t quadrupole_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Zero means a terminal/inactive member. Such a member is skipped before its
 * topology or numerical ranges are inspected and its public output is kept
 * byte-for-byte. One means active; all other byte values are semantic errors.
 */
struct Gfn2MullikenDeviceActivity {
  const std::uint8_t* active_mask = nullptr;
  std::int64_t elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Caller-owned unpublished storage. Each extent is counted in elements, not
 * bytes. sequence_active is one uint32_t used to preserve sticky sequence
 * failure semantics without host synchronization.
 */
struct Gfn2MullikenDeviceWorkspace {
  double* qsh_scratch = nullptr;
  std::int64_t qsh_elements = 0;
  double* qat_scratch = nullptr;
  std::int64_t qat_elements = 0;
  double* dipole_scratch = nullptr;
  std::int64_t dipole_elements = 0;
  double* quadrupole_scratch = nullptr;
  std::int64_t quadrupole_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2MullikenDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2MullikenDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2MullikenDeviceInput>);
static_assert(std::is_standard_layout_v<Gfn2MullikenDeviceInput>);
static_assert(std::is_trivially_copyable_v<Gfn2MullikenDevicePopulation>);
static_assert(std::is_standard_layout_v<Gfn2MullikenDevicePopulation>);
static_assert(std::is_trivially_copyable_v<Gfn2MullikenDeviceActivity>);
static_assert(std::is_standard_layout_v<Gfn2MullikenDeviceActivity>);
static_assert(std::is_trivially_copyable_v<Gfn2MullikenDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2MullikenDeviceWorkspace>);

/* Clear all per-system errors and the sticky first-error diagnostic. */
cudaError_t reset_gfn2_mulliken_device_errors_cuda(std::int64_t batch_size,
                                                   std::uint32_t* system_errors,
                                                   std::uint32_t* device_error,
                                                   cudaStream_t stream = nullptr) noexcept;

/*
 * Evaluate restricted Mulliken populations for a ragged batch. Publication is
 * atomic per system: active healthy systems overwrite all four outputs, while
 * invalid and inactive peers retain every prior public element.
 *
 * The launcher allocates nothing, synchronizes nothing, enqueues exclusively
 * on stream, supports CUDA Graph capture, and leaves device_error sticky. The
 * advertised readable and writable ranges must be mutually non-aliasing as
 * documented by the descriptors. Pointer address-space provenance is checked
 * once by the higher-level binding; this reusable hot launcher intentionally
 * avoids cudaPointerGetAttributes and accepts device/UVA addresses only.
 */
cudaError_t evaluate_gfn2_mulliken_population_cuda(
    const Gfn2MullikenDeviceBatch& batch, const Gfn2MullikenDeviceInput& input,
    const Gfn2MullikenDeviceActivity& activity, const Gfn2MullikenDevicePopulation& population,
    const Gfn2MullikenDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

/*
 * Evaluate mixed restricted/unrestricted populations using the spin-aware
 * common WavefunctionLayout projection. Density is system-major alpha/beta matrix data.
 * Outputs are system-major charge/magnetization channels and therefore match
 * WavefunctionLayout qsh/qat/dipole/quadrupole packing exactly.
 *
 * The legacy restricted entry point remains available so existing consumers
 * retain their established launch geometry and arithmetic path. This entry
 * preserves the same per-system transactional publication, inactive-first
 * behavior, custom-stream execution, and CUDA Graph capture contract.
 */
cudaError_t evaluate_gfn2_mulliken_population_spin_cuda(
    const Gfn2MullikenDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2MullikenDeviceInput& input, const Gfn2MullikenDeviceActivity& activity,
    const Gfn2MullikenDevicePopulation& population, const Gfn2MullikenDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_MULLIKEN_CUH
