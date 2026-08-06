#ifndef GPUXTB_BACKENDS_CUDA_GFN2_SPIN_CUH
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_BACKENDS_CUDA_GFN2_SPIN_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_scc_iteration_control.cuh"

namespace gpuxtb::detail::cuda {

/*
 * First asynchronous peer-local failure in one spin-polarization stage.
 * Codes remain below 64 so the SCC report normalizer can classify them with
 * Gfn2SccStageDeviceReport::peer_error_mask once the stage is composed into
 * the unrestricted iteration DAG.
 */
enum class Gfn2SpinDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidActiveMask = 1u,
  kInvalidOffsets = 2u,
  kInvalidSpinChannels = 3u,
  kInvalidCoupling = 4u,
  kNonfinitePopulation = 5u,
  kNonfinitePotentialArithmetic = 6u,
  kNonfiniteEnergyArithmetic = 7u,
};

inline constexpr std::uint64_t kGfn2SpinDevicePeerErrorMask =
    (std::uint64_t{1} << static_cast<std::uint32_t>(Gfn2SpinDeviceError::kInvalidActiveMask)) |
    (std::uint64_t{1} << static_cast<std::uint32_t>(Gfn2SpinDeviceError::kInvalidOffsets)) |
    (std::uint64_t{1} << static_cast<std::uint32_t>(Gfn2SpinDeviceError::kInvalidSpinChannels)) |
    (std::uint64_t{1} << static_cast<std::uint32_t>(Gfn2SpinDeviceError::kInvalidCoupling)) |
    (std::uint64_t{1} << static_cast<std::uint32_t>(Gfn2SpinDeviceError::kNonfinitePopulation)) |
    (std::uint64_t{1} << static_cast<std::uint32_t>(
         Gfn2SpinDeviceError::kNonfinitePotentialArithmetic)) |
    (std::uint64_t{1} << static_cast<std::uint32_t>(
         Gfn2SpinDeviceError::kNonfiniteEnergyArithmetic));

/*
 * Immutable, device-resident projection of gfn2::SpinPolarizationView.
 *
 * System i owns atoms [atom_offsets[i], atom_offsets[i+1]), shells
 * [batch_shell_offsets[i], batch_shell_offsets[i+1]), and a packed qsh slice
 * [shell_population_offsets[i], shell_population_offsets[i+1]).  A restricted
 * slice has one charge channel; an unrestricted slice has charge followed by
 * magnetization, so its population extent is exactly 2*nsh_i.
 *
 * Atom A owns a dense row-major [nsh_A,nsh_A] W matrix in coupling_matrices.
 * GFN2 has at most s/p/d shells per atom, but device preflight still validates
 * every active atom partition before using it.  All pointer provenance is a
 * setup/binding responsibility; the hot launcher performs no CUDA pointer
 * queries and accepts device/UVA addresses only.
 */
struct Gfn2SpinDeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t shell_population_elements = 0;

  std::int64_t atom_offset_count = 0;
  std::int64_t batch_shell_offset_count = 0;
  std::int64_t atom_shell_offset_count = 0;
  std::int64_t shell_population_offset_count = 0;
  std::int64_t spin_channel_count = 0;
  std::int64_t coupling_offset_count = 0;
  std::int64_t coupling_matrix_count = 0;
  std::uint64_t plan_token = 0u;

  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* batch_shell_offsets = nullptr;
  const std::int64_t* atom_shell_offsets = nullptr;
  const std::int64_t* shell_population_offsets = nullptr;
  const std::int32_t* spin_channels = nullptr;
  const std::int64_t* coupling_offsets = nullptr;
  const double* coupling_matrices = nullptr;
};

/* Charge/magnetization shell populations in the batch's exact ragged packing. */
struct Gfn2SpinDeviceInput {
  const double* shell_populations = nullptr;
  std::int64_t shell_population_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Public stage results.  spin_energies has one value per system and
 * shell_potentials mirrors the input packing.  Restricted systems publish
 * exact +0.0 to both fields.  For unrestricted systems the charge channel is
 * exact +0.0 and only the magnetization channel contains dE_spin/dm.
 */
struct Gfn2SpinDeviceOutput {
  double* spin_energies = nullptr;
  std::int64_t spin_energy_elements = 0;
  double* shell_potentials = nullptr;
  std::int64_t shell_potential_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Caller-owned unpublished storage providing per-system transactional
 * publication.  sequence_active is the stage-local plan latch: the launcher
 * snapshots the canonical SCC sequence and incoming sticky device error before
 * peer-local validation can record a new error.
 */
struct Gfn2SpinDeviceWorkspace {
  double* energy_scratch = nullptr;
  std::int64_t energy_elements = 0;
  double* potential_scratch = nullptr;
  std::int64_t potential_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2SpinDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2SpinDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2SpinDeviceInput>);
static_assert(std::is_standard_layout_v<Gfn2SpinDeviceInput>);
static_assert(std::is_trivially_copyable_v<Gfn2SpinDeviceOutput>);
static_assert(std::is_standard_layout_v<Gfn2SpinDeviceOutput>);
static_assert(std::is_trivially_copyable_v<Gfn2SpinDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2SpinDeviceWorkspace>);

/* Clear peer diagnostics and the sticky first-error scalar asynchronously. */
cudaError_t reset_gfn2_spin_device_errors_cuda(std::int64_t batch_size,
                                               std::uint32_t* system_errors,
                                               std::uint32_t* device_error,
                                               cudaStream_t stream = nullptr) noexcept;

/*
 * Evaluate
 *
 *   E_spin = 1/2 sum_A m_A^T W_A m_A,   v_mag,A = W_A m_A.
 *
 * Only canonical active members are inspected.  Inactive slices may contain
 * poison and retain every output byte.  An active peer with invalid topology,
 * NaN/Inf input, or non-finite arithmetic records its own error and publishes
 * nothing; healthy peers commit independently.  Restricted peers publish
 * exact zeros without inspecting their numerical populations, W values, or
 * atom/shell/coupling topology.
 *
 * The launcher allocates nothing, transfers nothing, synchronizes nowhere,
 * enqueues exclusively on stream, and supports CUDA Graph capture/replay.
 * Read-only ranges may alias one another, but every writable range must be
 * disjoint from every readable range and from all other writable ranges.
 */
cudaError_t evaluate_gfn2_spin_polarization_cuda(
    const Gfn2SpinDeviceBatch& batch, const Gfn2SpinDeviceInput& input,
    const Gfn2SccIterationDeviceActivity& activity, const Gfn2SpinDeviceOutput& output,
    const Gfn2SpinDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_SPIN_CUH
