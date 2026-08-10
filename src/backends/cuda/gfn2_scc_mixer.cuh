#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_SCC_MIXER_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN2_SCC_MIXER_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/common/gfn2_plan_schema.hpp"
#include "backends/cuda/gfn2_scc.cuh"
#include "backends/cuda/gfn2_scc_iteration_control.cuh"

namespace xtbloom::detail::cuda {

/* First asynchronous semantic failure recorded by a mixer sequence. */
enum class Gfn2SccMixerDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kInvalidState = 2u,
  kNonfiniteInitialMultipole = 3u,
  kNonfiniteRestartMultipole = 4u,
  kNonfiniteRawMultipole = 5u,
  kNonfiniteResidual = 6u,
  kIterationOverflow = 7u,
  kNonfiniteDifference = 8u,
  kNormalizationFailure = 9u,
  kNonfiniteUpdate = 10u,
  kNonfiniteWeight = 11u,
  kNonfiniteCoefficient = 12u,
  kNonfiniteHistory = 13u,
  kUnusableBroydenSystem = 14u,
  kNonfiniteMixedMultipole = 15u,
  kRestartOverflow = 16u,
};

/* Numerical policy for the tblite/xTB finite-memory modified-Broyden mixer. */
struct Gfn2SccMixerDevicePolicy {
  std::int64_t history_size = 0;
  double damping = 0.0;
  double rms_tolerance = 0.0;
  double maximum_tolerance = 0.0;
  std::uint64_t plan_token = 0u;
};

/*
 * Persistent device-resident mixer history.
 *
 * current_inputs, previous_inputs, and previous_residuals use a packed
 * system-major vector: qsh, then dipole, then quadrupole. System starts are
 * shell_offsets[system] + 9 * atom_offsets[system]. History is system-major,
 * then circular-slot-major, and therefore contains total_vector_elements *
 * history_size doubles in each of df_history and u_history.
 *
 * This state is intentionally separate from Gfn2SccDeviceState. The mixer
 * advances its private input before update_gfn2_scc_state_cuda compares the
 * raw result with the driver-visible input that produced it.
 */
struct Gfn2SccMixerDeviceState {
  double* current_inputs = nullptr;
  double* previous_inputs = nullptr;
  double* previous_residuals = nullptr;
  double* df_history = nullptr;
  double* u_history = nullptr;
  double* omega = nullptr;

  double* residual_rms = nullptr;
  double* residual_maximum = nullptr;
  std::uint64_t* iterations = nullptr;
  std::uint64_t* restart_counts = nullptr;
  xtbloom_status_t* system_statuses = nullptr;
  std::uint8_t* initialized = nullptr;
  /*
   * Residual-only diagnostic using the mixer's RMS and maximum thresholds.
   * This byte never authorizes or suppresses an SCC iteration. Terminal
   * convergence belongs exclusively to Gfn2SccDeviceState::converged.
   */
  std::uint8_t* residual_converged = nullptr;

  std::int64_t total_vector_elements = 0;
  std::int64_t history_elements = 0;
  std::int64_t omega_elements = 0;
  std::int64_t batch_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Caller-owned tentative storage for a parallel ragged-batch transition.
 * Four vector arrays have total_vector_elements entries. beta and
 * coefficients have respectively batch_size * history_size^2 and
 * batch_size * history_size entries. sequence_active is the stage-local plan
 * latch: mix refreshes it from canonical activity on every launch/Graph replay
 * and only plan failures may close it. Failed members may modify only their
 * private scratch slice.
 */
struct Gfn2SccMixerDeviceWorkspace {
  double* residual = nullptr;
  double* mixed = nullptr;
  double* delta_f = nullptr;
  double* new_u = nullptr;
  double* beta = nullptr;
  double* coefficients = nullptr;
  std::uint32_t* sequence_active = nullptr;

  std::int64_t vector_elements = 0;
  std::int64_t beta_elements = 0;
  std::int64_t coefficient_elements = 0;
  std::int64_t sequence_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2SccMixerDevicePolicy>);
static_assert(std::is_standard_layout_v<Gfn2SccMixerDevicePolicy>);
static_assert(std::is_trivially_copyable_v<Gfn2SccMixerDeviceState>);
static_assert(std::is_standard_layout_v<Gfn2SccMixerDeviceState>);
static_assert(std::is_trivially_copyable_v<Gfn2SccMixerDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2SccMixerDeviceWorkspace>);

/*
 * Capture finite initial multipoles and clear all history atomically.
 *
 * A nonfinite member prevents every member from being initialized, matching
 * initialize_scc_mixer_state_cpu. The launcher allocates nothing, transfers
 * nothing, synchronizes nothing, supports custom streams and CUDA Graph
 * capture, and leaves device_error sticky.
 */
cudaError_t initialize_gfn2_scc_mixer_cuda(const Gfn2SccDeviceBatch& batch,
                                           const Gfn2SccMixerDevicePolicy& policy,
                                           const Gfn2SccDeviceConstMultipoles& initial,
                                           const Gfn2SccMixerDeviceState& state,
                                           const Gfn2SccMixerDeviceWorkspace& workspace,
                                           std::uint32_t* device_error,
                                           cudaStream_t stream = nullptr) noexcept;

/*
 * Mixed one/two-channel initialization. layout expands each system vector to
 * include both charge and magnetization qsh/dipole/quadrupole channels while
 * preserving the established restricted entry point above.
 */
cudaError_t initialize_gfn2_scc_mixer_cuda(
    const Gfn2SccDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2SccMixerDevicePolicy& policy, const Gfn2SccDeviceConstMultipoles& initial,
    const Gfn2SccMixerDeviceState& state, const Gfn2SccMixerDeviceWorkspace& workspace,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

/*
 * Restart one initialized member from its supplied public multipoles.
 *
 * The target's raw values are validated before any persistent byte changes.
 * On success its history and diagnostics are cleared, its restart counter is
 * incremented, and it becomes active. Every peer remains byte-identical.
 */
cudaError_t restart_gfn2_scc_mixer_system_cuda(
    const Gfn2SccDeviceBatch& batch, const Gfn2SccMixerDevicePolicy& policy, std::int64_t system,
    const Gfn2SccDeviceConstMultipoles& current_public, const Gfn2SccMixerDeviceState& state,
    const Gfn2SccMixerDeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

/* Restart one mixed-spin member using the nspin-expanded layout. */
cudaError_t restart_gfn2_scc_mixer_system_cuda(
    const Gfn2SccDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2SccMixerDevicePolicy& policy, std::int64_t system,
    const Gfn2SccDeviceConstMultipoles& current_public, const Gfn2SccMixerDeviceState& state,
    const Gfn2SccMixerDeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

/*
 * Generate the next mixed qsh/dipole/quadrupole vectors entirely on device.
 *
 * The canonical iteration activity projection is the sole normal execution
 * authority: sequence_active is tested before active_mask, and both are
 * tested before any member offset, raw multipole, or mixer-history read.
 * residual_converged is diagnostic only and never suppresses a transition.
 * The private current input is always the generated mixed vector; raw/public
 * terminal publication remains the responsibility of the SCC state composer.
 * next_mixed may exactly alias raw field-by-field.
 *
 * Numerical failure leaves raw/next_mixed and all persistent numerical
 * history unchanged for that member; only system_statuses records
 * XTBLOOM_STATUS_INTERNAL_ERROR. Healthy peers still commit. The launcher has
 * no steady-state allocation, host/device transfer, or synchronization and is
 * safe for custom streams and CUDA Graph capture.
 *
 * The SCC composer must not feed device_error to #87 normalization because it
 * uses Gfn2SccMixerDeviceError codes while system_statuses uses
 * xtbloom_status_t codes. The canonical kMixer report therefore uses
 * system_statuses with kXTBloomStatus, a peer mask containing only
 * XTBLOOM_STATUS_INTERNAL_ERROR, a null device_error, and workspace's
 * sequence_active as the plan latch. device_error remains a tracing channel;
 * a closed plan latch is recorded through #87's SequenceClosed fallback.
 */
cudaError_t mix_gfn2_scc_broyden_cuda(
    const Gfn2SccDeviceBatch& batch, const Gfn2SccMixerDevicePolicy& policy,
    const Gfn2SccIterationDeviceActivity& activity, const Gfn2SccDeviceConstMultipoles& raw,
    const Gfn2SccDeviceMultipoles& next_mixed, const Gfn2SccMixerDeviceState& state,
    const Gfn2SccMixerDeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

/*
 * Mixed-spin Broyden transition. Charge and magnetization channels are one
 * complete vector, so every unrestricted multipole component participates in
 * the same finite-memory update and convergence diagnostics.
 */
cudaError_t mix_gfn2_scc_broyden_cuda(
    const Gfn2SccDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2SccMixerDevicePolicy& policy, const Gfn2SccIterationDeviceActivity& activity,
    const Gfn2SccDeviceConstMultipoles& raw, const Gfn2SccDeviceMultipoles& next_mixed,
    const Gfn2SccMixerDeviceState& state, const Gfn2SccMixerDeviceWorkspace& workspace,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_SCC_MIXER_CUH
