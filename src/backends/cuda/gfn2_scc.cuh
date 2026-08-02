#ifndef GPUXTB_BACKENDS_CUDA_GFN2_SCC_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_SCC_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "gpuxtb/gpuxtb.h"

namespace gpuxtb::detail::cuda {

/* First asynchronous semantic failure recorded by the SCC state sequence. */
enum class Gfn2SccDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kInvalidState = 2u,
  kNonfiniteCurrentMultipole = 3u,
  kNonfiniteMixedMultipole = 4u,
  kNonfiniteRawMultipole = 5u,
  kNonfiniteResidual = 6u,
  kNonfiniteFreeEnergy = 7u,
  kNonfiniteEnergyDelta = 8u,
};

/*
 * Ragged qsh/atom topology shared by current, next-mixed, raw, and published
 * multipoles. Dipoles and traceless quadrupoles use respectively 3 and 6
 * doubles per atom. A nonzero plan_token binds every descriptor below to the
 * same setup-time topology without exposing a host plan object to device code.
 */
struct Gfn2SccDeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_atoms = 0;
  std::int64_t shell_offset_count = 0;
  std::int64_t atom_offset_count = 0;
  std::uint64_t plan_token = 0u;
  const std::int64_t* shell_offsets = nullptr;
  const std::int64_t* atom_offsets = nullptr;
};

/* Read-only qsh/dipole/quadrupole buffers in their complete packed layouts. */
struct Gfn2SccDeviceConstMultipoles {
  const double* shell_charges = nullptr;
  std::int64_t shell_elements = 0;
  const double* atomic_dipoles = nullptr;
  std::int64_t dipole_elements = 0;
  const double* atomic_quadrupoles = nullptr;
  std::int64_t quadrupole_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Writable counterpart used for persistent current inputs and publication. */
struct Gfn2SccDeviceMultipoles {
  double* shell_charges = nullptr;
  std::int64_t shell_elements = 0;
  double* atomic_dipoles = nullptr;
  std::int64_t dipole_elements = 0;
  double* atomic_quadrupoles = nullptr;
  std::int64_t quadrupole_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Strict combined convergence policy, matching the production CPU driver. */
struct Gfn2SccDevicePolicy {
  std::uint64_t maximum_iterations = 0u;
  double residual_rms_tolerance = 0.0;
  double energy_tolerance = 0.0;
  std::uint64_t plan_token = 0u;
};

/*
 * Persistent device-resident SCC trace. current_inputs is the mixed vector
 * that produced the current raw result before a call, and becomes the supplied
 * next-mixed vector after every successful active-system transition.
 * iterations==0 makes the previous free-energy seed exactly zero.
 */
struct Gfn2SccDeviceState {
  Gfn2SccDeviceMultipoles current_inputs;
  double* free_energies = nullptr;
  double* previous_free_energies = nullptr;
  double* free_energy_changes = nullptr;
  double* residual_rms = nullptr;
  std::uint64_t* iterations = nullptr;
  gpuxtb_status_t* system_statuses = nullptr;
  std::uint8_t* converged = nullptr;
  std::int64_t batch_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * One caller-owned scalar snapshots whether the sticky sequence was clear
 * after topology preflight. Per-system numerical errors may then set the
 * sticky error without preventing independently scheduled healthy peers from
 * committing in the same launch.
 */
struct Gfn2SccDeviceWorkspace {
  std::uint32_t* sequence_active = nullptr;
  std::int64_t elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2SccDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2SccDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2SccDeviceConstMultipoles>);
static_assert(std::is_standard_layout_v<Gfn2SccDeviceConstMultipoles>);
static_assert(std::is_trivially_copyable_v<Gfn2SccDeviceMultipoles>);
static_assert(std::is_standard_layout_v<Gfn2SccDeviceMultipoles>);
static_assert(std::is_trivially_copyable_v<Gfn2SccDevicePolicy>);
static_assert(std::is_standard_layout_v<Gfn2SccDevicePolicy>);
static_assert(std::is_trivially_copyable_v<Gfn2SccDeviceState>);
static_assert(std::is_standard_layout_v<Gfn2SccDeviceState>);
static_assert(std::is_trivially_copyable_v<Gfn2SccDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2SccDeviceWorkspace>);

/* Clear the sticky semantic error before starting a dependent SCC sequence. */
cudaError_t reset_gfn2_scc_device_error_cuda(std::uint32_t* device_error,
                                             cudaStream_t stream = nullptr) noexcept;

/*
 * Advance every active SCC member by one state/convergence transition.
 *
 * residual_rms is computed from raw-current over the system-major
 * concatenation qsh,dipole,quadrupole. Convergence requires both RMS and the
 * absolute complete-free-energy change to be strictly below their tolerances.
 * Successful members increment iterations and publish raw multipoles when
 * converged, otherwise next_mixed. A last unconverged iteration publishes its
 * mixed result and records GPUXTB_STATUS_SCC_NOT_CONVERGED.
 *
 * A member with nonfinite numerical data records GPUXTB_STATUS_INTERNAL_ERROR,
 * counts the active attempt, and replaces free-energy/current-change/RMS trace
 * entries with NaN. Its public multipoles, private current inputs, and
 * convergence flag remain unchanged; healthy peers still commit. Existing
 * terminal or converged members are skipped without inspecting their numerical
 * buffers. For exact CPU active-predicate parity, a SUCCESS, unconverged member
 * already at maximum_iterations is also skipped byte-for-byte; well-formed
 * state reaches that condition only after the preceding active attempt has
 * already published GPUXTB_STATUS_SCC_NOT_CONVERGED.
 *
 * All storage remains on device. The launcher allocates nothing, performs no
 * synchronization or transfer, supports custom streams and CUDA Graph capture,
 * and leaves device_error sticky. Writable ranges must be pairwise disjoint
 * from inputs and one another, except that each published field may exactly
 * alias the corresponding next_mixed field for in-place publication.
 *
 * Address-space provenance is a setup/binding responsibility, as for the other
 * native CUDA kernels: every advertised range must be a device address (or a
 * UVA-managed address directly usable by kernels on the selected device).
 * The hot launcher deliberately does not call cudaPointerGetAttributes for
 * every field. Passing an ordinary host pointer violates this low-level API's
 * contract; higher-level C API bindings must validate memory-space tags once
 * when constructing these reusable descriptors.
 */
cudaError_t update_gfn2_scc_state_cuda(
    const Gfn2SccDeviceBatch& batch, const Gfn2SccDevicePolicy& policy,
    const Gfn2SccDeviceConstMultipoles& next_mixed, const Gfn2SccDeviceConstMultipoles& raw,
    const double* complete_free_energies, const Gfn2SccDeviceMultipoles& published,
    const Gfn2SccDeviceState& state, const Gfn2SccDeviceWorkspace& workspace,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_SCC_CUH
