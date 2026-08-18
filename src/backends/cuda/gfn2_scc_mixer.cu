#include <array>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_scc_mixer.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr std::int64_t kDipoleComponents = 3;
constexpr std::int64_t kQuadrupoleComponents = 6;
constexpr std::int64_t kMultipoleComponentsPerAtom = kDipoleComponents + kQuadrupoleComponents;
constexpr double kOmegaZero = 0.01;
constexpr double kMinimumOmega = 1.0;
constexpr double kMaximumOmega = 100000.0;
constexpr double kOmegaFactor = 0.01;
constexpr double kDoubleEpsilon = 2.2204460492503131e-16;
constexpr std::uint64_t kMaximumUint64 = 0xffffffffffffffffULL;

static_assert((kThreadsPerBlock & (kThreadsPerBlock - 1)) == 0,
              "mixer reductions require a power-of-two block size");

__device__ void record_error(std::uint32_t* device_error, Gfn2SccMixerDeviceError error) {
  atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2SccMixerDeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ bool has_spin_layout(const Gfn2WavefunctionLayoutView& layout) {
  return layout.spin_channels != nullptr;
}

__device__ std::int64_t shell_begin(const Gfn2SccDeviceBatch& batch,
                                    const Gfn2WavefunctionLayoutView& layout, std::int64_t system) {
  return has_spin_layout(layout) ? layout.spin_shell_offsets[system] : batch.shell_offsets[system];
}

__device__ std::int64_t atom_begin(const Gfn2SccDeviceBatch& batch,
                                   const Gfn2WavefunctionLayoutView& layout, std::int64_t system) {
  return has_spin_layout(layout) ? layout.spin_atom_offsets[system] : batch.atom_offsets[system];
}

__device__ std::int64_t system_vector_begin(const Gfn2SccDeviceBatch& batch,
                                            const Gfn2WavefunctionLayoutView& layout,
                                            const Gfn2SccMixerDevicePolicy& policy,
                                            std::int64_t system) {
  return shell_begin(batch, layout, system) +
         static_cast<std::int64_t>(policy.atomic_multipole_components) *
             atom_begin(batch, layout, system);
}

__device__ std::int64_t system_dimension(const Gfn2SccDeviceBatch& batch,
                                         const Gfn2WavefunctionLayoutView& layout,
                                         const Gfn2SccMixerDevicePolicy& policy,
                                         std::int64_t system) {
  const std::int64_t shells =
      shell_begin(batch, layout, system + 1) - shell_begin(batch, layout, system);
  return shells + static_cast<std::int64_t>(policy.atomic_multipole_components) *
                      (atom_begin(batch, layout, system + 1) - atom_begin(batch, layout, system));
}

__device__ double load_component(const Gfn2SccDeviceBatch& batch,
                                 const Gfn2WavefunctionLayoutView& layout,
                                 const Gfn2SccMixerDevicePolicy& policy,
                                 const Gfn2SccDeviceConstMultipoles& multipoles,
                                 std::int64_t system, std::int64_t component) {
  const std::int64_t shell_offset = shell_begin(batch, layout, system);
  const std::int64_t shell_count = shell_begin(batch, layout, system + 1) - shell_offset;
  if (component < shell_count) {
    return multipoles.shell_charges[shell_offset + component];
  }
  if (policy.atomic_multipole_components == 0) return 0.0;
  component -= shell_count;
  const std::int64_t atom_offset = atom_begin(batch, layout, system);
  const std::int64_t atom_count = atom_begin(batch, layout, system + 1) - atom_offset;
  const std::int64_t dipole_count = atom_count * kDipoleComponents;
  if (component < dipole_count) {
    return multipoles.atomic_dipoles[atom_offset * kDipoleComponents + component];
  }
  component -= dipole_count;
  return multipoles.atomic_quadrupoles[atom_offset * kQuadrupoleComponents + component];
}

__device__ void store_component(const Gfn2SccDeviceBatch& batch,
                                const Gfn2WavefunctionLayoutView& layout,
                                const Gfn2SccMixerDevicePolicy& policy,
                                const Gfn2SccDeviceMultipoles& multipoles, std::int64_t system,
                                std::int64_t component, double value) {
  const std::int64_t shell_offset = shell_begin(batch, layout, system);
  const std::int64_t shell_count = shell_begin(batch, layout, system + 1) - shell_offset;
  if (component < shell_count) {
    multipoles.shell_charges[shell_offset + component] = value;
    return;
  }
  if (policy.atomic_multipole_components == 0) return;
  component -= shell_count;
  const std::int64_t atom_offset = atom_begin(batch, layout, system);
  const std::int64_t atom_count = atom_begin(batch, layout, system + 1) - atom_offset;
  const std::int64_t dipole_count = atom_count * kDipoleComponents;
  if (component < dipole_count) {
    multipoles.atomic_dipoles[atom_offset * kDipoleComponents + component] = value;
    return;
  }
  component -= dipole_count;
  multipoles.atomic_quadrupoles[atom_offset * kQuadrupoleComponents + component] = value;
}

__global__ void topology_preflight_kernel(Gfn2SccDeviceBatch batch,
                                          Gfn2WavefunctionLayoutView layout,
                                          std::uint32_t* device_error) {
  if (atomicAdd(device_error, 0u) !=
      static_cast<std::uint32_t>(Gfn2SccMixerDeviceError::kSuccess)) {
    return;
  }
  const bool spin_layout = has_spin_layout(layout);
  if (threadIdx.x == 0 &&
      (batch.shell_offsets[0] != 0 || batch.atom_offsets[0] != 0 ||
       batch.shell_offsets[batch.batch_size] != batch.total_shells ||
       batch.atom_offsets[batch.batch_size] != batch.total_atoms ||
       (spin_layout &&
        (layout.spin_shell_offsets[0] != 0 || layout.spin_atom_offsets[0] != 0 ||
         layout.spin_channel_offsets[0] != 0 ||
         layout.spin_shell_offsets[batch.batch_size] != layout.total_spin_shells ||
         layout.spin_atom_offsets[batch.batch_size] != layout.total_spin_atoms ||
         layout.spin_channel_offsets[batch.batch_size] != layout.total_spin_channels)))) {
    record_error(device_error, Gfn2SccMixerDeviceError::kInvalidOffsets);
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    const std::int64_t shell_begin = batch.shell_offsets[system];
    const std::int64_t shell_end = batch.shell_offsets[system + 1];
    const std::int64_t atom_begin = batch.atom_offsets[system];
    const std::int64_t atom_end = batch.atom_offsets[system + 1];
    bool invalid = shell_begin < 0 || shell_begin >= shell_end || shell_end > batch.total_shells ||
                   atom_begin < 0 || atom_begin >= atom_end || atom_end > batch.total_atoms;
    if (!invalid && spin_layout) {
      const std::int32_t channels = layout.spin_channels[system];
      const std::int64_t spin_channel_begin = layout.spin_channel_offsets[system];
      const std::int64_t spin_channel_end = layout.spin_channel_offsets[system + 1];
      const std::int64_t spin_shell_begin = layout.spin_shell_offsets[system];
      const std::int64_t spin_shell_end = layout.spin_shell_offsets[system + 1];
      const std::int64_t spin_atom_begin = layout.spin_atom_offsets[system];
      const std::int64_t spin_atom_end = layout.spin_atom_offsets[system + 1];
      invalid = (channels != 1 && channels != 2) || spin_channel_begin < 0 ||
                spin_channel_end - spin_channel_begin != channels ||
                spin_channel_end > layout.total_spin_channels || spin_shell_begin < 0 ||
                spin_shell_end - spin_shell_begin != channels * (shell_end - shell_begin) ||
                spin_shell_end > layout.total_spin_shells || spin_atom_begin < 0 ||
                spin_atom_end - spin_atom_begin != channels * (atom_end - atom_begin) ||
                spin_atom_end > layout.total_spin_atoms;
    }
    if (invalid) {
      record_error(device_error, Gfn2SccMixerDeviceError::kInvalidOffsets);
    }
  }
}

__global__ void initial_values_preflight_kernel(Gfn2SccDeviceBatch batch,
                                                Gfn2WavefunctionLayoutView layout,
                                                Gfn2SccMixerDevicePolicy policy,
                                                Gfn2SccDeviceConstMultipoles initial,
                                                std::uint32_t* device_error) {
  if (atomicAdd(device_error, 0u) !=
      static_cast<std::uint32_t>(Gfn2SccMixerDeviceError::kSuccess)) {
    return;
  }
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t dimension = system_dimension(batch, layout, policy, system);
  for (std::int64_t component = threadIdx.x; component < dimension; component += blockDim.x) {
    if (!isfinite(load_component(batch, layout, policy, initial, system, component))) {
      record_error(device_error, Gfn2SccMixerDeviceError::kNonfiniteInitialMultipole);
    }
  }
}

__global__ void capture_sequence_kernel(const std::uint32_t* device_error,
                                        std::uint32_t* sequence_active) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *sequence_active = atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                               static_cast<std::uint32_t>(Gfn2SccMixerDeviceError::kSuccess)
                           ? 1u
                           : 0u;
  }
}

/*
 * Start a reusable mixer stage from the canonical SCC sequence latch. Unlike
 * standalone initialize/restart, an old mixer diagnostic must not become an
 * independent execution authority. Each Graph replay therefore refreshes the
 * stage-local latch solely from the canonical ledger.
 */
__global__ void begin_mixer_stage_kernel(Gfn2SccIterationDeviceActivity activity,
                                         Gfn2SccMixerDeviceWorkspace workspace,
                                         std::uint32_t* device_error) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const std::uint32_t sequence =
        atomicAdd(const_cast<std::uint32_t*>(activity.sequence_active), 0u);
    if (sequence > 1u) {
      record_error(device_error, Gfn2SccMixerDeviceError::kInvalidState);
      *workspace.sequence_active = 0u;
      return;
    }
    *workspace.sequence_active = sequence;
  }
}

/*
 * Publish a clean per-member diagnostic generation before any preflight may
 * close the mixer stage. This separate launch is required: clearing inside a
 * multi-block topology kernel would race a peer block closing the stage latch,
 * potentially leaving stale non-peer statuses visible to #87 normalization.
 */
__global__ void prepare_active_mixer_statuses_kernel(Gfn2SccIterationDeviceActivity activity,
                                                     Gfn2SccMixerDeviceState state) {
  if (atomicAdd(const_cast<std::uint32_t*>(activity.sequence_active), 0u) != 1u) {
    return;
  }
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= activity.batch_elements) {
    return;
  }
  const std::uint8_t active = activity.active_mask[system];
  if (active == 1u) {
    state.system_statuses[system] = XTBLOOM_STATUS_SUCCESS;
  }
}

/*
 * Validate only members requested by the canonical ledger. Inactive members
 * may intentionally carry stale or poisoned generation-local data, including
 * offsets, and must remain unread. A malformed active slice is a plan failure
 * and closes this mixer stage before the numerical kernel can inspect it.
 */
__global__ void active_topology_preflight_kernel(Gfn2SccDeviceBatch batch,
                                                 Gfn2WavefunctionLayoutView layout,
                                                 Gfn2SccIterationDeviceActivity activity,
                                                 Gfn2SccMixerDeviceWorkspace workspace,
                                                 std::uint32_t* device_error) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  if (atomicAdd(const_cast<std::uint32_t*>(activity.sequence_active), 0u) != 1u ||
      atomicAdd(workspace.sequence_active, 0u) != 1u) {
    return;
  }
  bool have_previous_active = false;
  std::int64_t previous_shell_end = 0;
  std::int64_t previous_atom_end = 0;
  std::int64_t previous_spin_shell_end = 0;
  std::int64_t previous_spin_atom_end = 0;
  std::int64_t previous_spin_channel_end = 0;
  const bool spin_layout = has_spin_layout(layout);
  for (std::int64_t system = 0; system < batch.batch_size; ++system) {
    const std::uint8_t active = activity.active_mask[system];
    if (active > 1u) {
      record_error(device_error, Gfn2SccMixerDeviceError::kInvalidState);
      atomicExch(workspace.sequence_active, 0u);
      return;
    }
    if (active == 0u) {
      continue;
    }

    /* Read offsets only after this exact member passed the canonical gate. */
    const std::int64_t shell_begin = batch.shell_offsets[system];
    const std::int64_t shell_end = batch.shell_offsets[system + 1];
    const std::int64_t atom_begin = batch.atom_offsets[system];
    const std::int64_t atom_end = batch.atom_offsets[system + 1];
    bool invalid_local = shell_begin < 0 || shell_begin >= shell_end ||
                         shell_end > batch.total_shells || atom_begin < 0 ||
                         atom_begin >= atom_end || atom_end > batch.total_atoms;
    bool invalid_endpoint = (system == 0 && (shell_begin != 0 || atom_begin != 0)) ||
                            (system == batch.batch_size - 1 &&
                             (shell_end != batch.total_shells || atom_end != batch.total_atoms));
    bool overlaps_previous = have_previous_active &&
                             (shell_begin < previous_shell_end || atom_begin < previous_atom_end);
    std::int64_t spin_shell_end = 0;
    std::int64_t spin_atom_end = 0;
    std::int64_t spin_channel_end = 0;
    if (!invalid_local && spin_layout) {
      const std::int32_t channels = layout.spin_channels[system];
      const std::int64_t spin_shell_begin = layout.spin_shell_offsets[system];
      spin_shell_end = layout.spin_shell_offsets[system + 1];
      const std::int64_t spin_atom_begin = layout.spin_atom_offsets[system];
      spin_atom_end = layout.spin_atom_offsets[system + 1];
      const std::int64_t spin_channel_begin = layout.spin_channel_offsets[system];
      spin_channel_end = layout.spin_channel_offsets[system + 1];
      invalid_local = (channels != 1 && channels != 2) || spin_shell_begin < 0 ||
                      spin_shell_end - spin_shell_begin != channels * (shell_end - shell_begin) ||
                      spin_shell_end > layout.total_spin_shells || spin_atom_begin < 0 ||
                      spin_atom_end - spin_atom_begin != channels * (atom_end - atom_begin) ||
                      spin_atom_end > layout.total_spin_atoms || spin_channel_begin < 0 ||
                      spin_channel_end - spin_channel_begin != channels ||
                      spin_channel_end > layout.total_spin_channels;
      invalid_endpoint =
          invalid_endpoint ||
          (system == 0 &&
           (spin_shell_begin != 0 || spin_atom_begin != 0 || spin_channel_begin != 0)) ||
          (system == batch.batch_size - 1 && (spin_shell_end != layout.total_spin_shells ||
                                              spin_atom_end != layout.total_spin_atoms ||
                                              spin_channel_end != layout.total_spin_channels));
      overlaps_previous = overlaps_previous || (have_previous_active &&
                                                (spin_shell_begin < previous_spin_shell_end ||
                                                 spin_atom_begin < previous_spin_atom_end ||
                                                 spin_channel_begin < previous_spin_channel_end));
    }
    if (invalid_local || invalid_endpoint || overlaps_previous) {
      record_error(device_error, Gfn2SccMixerDeviceError::kInvalidOffsets);
      atomicExch(workspace.sequence_active, 0u);
      return;
    }
    have_previous_active = true;
    previous_shell_end = shell_end;
    previous_atom_end = atom_end;
    previous_spin_shell_end = spin_shell_end;
    previous_spin_atom_end = spin_atom_end;
    previous_spin_channel_end = spin_channel_end;
  }
}

__global__ void initialize_state_kernel(Gfn2SccDeviceBatch batch, Gfn2WavefunctionLayoutView layout,
                                        Gfn2SccMixerDevicePolicy policy,
                                        Gfn2SccDeviceConstMultipoles initial,
                                        Gfn2SccMixerDeviceState state,
                                        Gfn2SccMixerDeviceWorkspace workspace) {
  if (atomicAdd(workspace.sequence_active, 0u) != 1u) {
    return;
  }
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t vector_begin = system_vector_begin(batch, layout, policy, system);
  const std::int64_t dimension = system_dimension(batch, layout, policy, system);
  const std::int64_t history_begin = vector_begin * policy.history_size;
  const std::int64_t history_count = dimension * policy.history_size;
  for (std::int64_t component = threadIdx.x; component < dimension; component += blockDim.x) {
    state.current_inputs[vector_begin + component] =
        load_component(batch, layout, policy, initial, system, component);
    state.previous_inputs[vector_begin + component] = 0.0;
    state.previous_residuals[vector_begin + component] = 0.0;
  }
  for (std::int64_t element = threadIdx.x; element < history_count; element += blockDim.x) {
    state.df_history[history_begin + element] = 0.0;
    state.u_history[history_begin + element] = 0.0;
  }
  for (std::int64_t slot = threadIdx.x; slot < policy.history_size; slot += blockDim.x) {
    state.omega[system * policy.history_size + slot] = 0.0;
  }
  if (threadIdx.x == 0) {
    state.residual_rms[system] = 0.0;
    state.residual_maximum[system] = 0.0;
    state.iterations[system] = 0u;
    state.restart_counts[system] = 0u;
    state.system_statuses[system] = XTBLOOM_STATUS_SUCCESS;
    state.initialized[system] = 1u;
    state.residual_converged[system] = 0u;
  }
}

__global__ void restart_system_kernel(Gfn2SccDeviceBatch batch, Gfn2WavefunctionLayoutView layout,
                                      Gfn2SccMixerDevicePolicy policy, std::int64_t system,
                                      Gfn2SccDeviceConstMultipoles current_public,
                                      Gfn2SccMixerDeviceState state,
                                      Gfn2SccMixerDeviceWorkspace workspace,
                                      std::uint32_t* device_error) {
  __shared__ int valid;
  if (threadIdx.x == 0) {
    valid = atomicAdd(workspace.sequence_active, 0u) == 1u ? 1 : 0;
    if (valid != 0 && state.initialized[system] != 1u) {
      record_error(device_error, Gfn2SccMixerDeviceError::kInvalidState);
      valid = 0;
    } else if (valid != 0 && state.restart_counts[system] == kMaximumUint64) {
      record_error(device_error, Gfn2SccMixerDeviceError::kRestartOverflow);
      valid = 0;
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  const std::int64_t dimension = system_dimension(batch, layout, policy, system);
  for (std::int64_t component = threadIdx.x; component < dimension; component += blockDim.x) {
    if (!isfinite(load_component(batch, layout, policy, current_public, system, component))) {
      record_error(device_error, Gfn2SccMixerDeviceError::kNonfiniteRestartMultipole);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  const std::int64_t vector_begin = system_vector_begin(batch, layout, policy, system);
  const std::int64_t history_begin = vector_begin * policy.history_size;
  const std::int64_t history_count = dimension * policy.history_size;
  for (std::int64_t component = threadIdx.x; component < dimension; component += blockDim.x) {
    state.current_inputs[vector_begin + component] =
        load_component(batch, layout, policy, current_public, system, component);
    state.previous_inputs[vector_begin + component] = 0.0;
    state.previous_residuals[vector_begin + component] = 0.0;
  }
  for (std::int64_t element = threadIdx.x; element < history_count; element += blockDim.x) {
    state.df_history[history_begin + element] = 0.0;
    state.u_history[history_begin + element] = 0.0;
  }
  for (std::int64_t slot = threadIdx.x; slot < policy.history_size; slot += blockDim.x) {
    state.omega[system * policy.history_size + slot] = 0.0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    state.residual_rms[system] = 0.0;
    state.residual_maximum[system] = 0.0;
    state.iterations[system] = 0u;
    ++state.restart_counts[system];
    state.system_statuses[system] = XTBLOOM_STATUS_SUCCESS;
    state.residual_converged[system] = 0u;
  }
}

__device__ bool cholesky_solve(double* matrix, double* right_hand_side, std::int64_t dimension,
                               std::int64_t leading_dimension) {
  for (std::int64_t row = 0; row < dimension; ++row) {
    for (std::int64_t column = 0; column <= row; ++column) {
      double value = matrix[row * leading_dimension + column];
      for (std::int64_t inner = 0; inner < column; ++inner) {
        value -=
            matrix[row * leading_dimension + inner] * matrix[column * leading_dimension + inner];
      }
      if (!isfinite(value)) {
        return false;
      }
      if (row == column) {
        if (!(value > 0.0)) {
          return false;
        }
        matrix[row * leading_dimension + column] = sqrt(value);
      } else {
        value /= matrix[column * leading_dimension + column];
        if (!isfinite(value)) {
          return false;
        }
        matrix[row * leading_dimension + column] = value;
      }
    }
  }
  for (std::int64_t row = 0; row < dimension; ++row) {
    double value = right_hand_side[row];
    for (std::int64_t column = 0; column < row; ++column) {
      value -= matrix[row * leading_dimension + column] * right_hand_side[column];
    }
    value /= matrix[row * leading_dimension + row];
    if (!isfinite(value)) {
      return false;
    }
    right_hand_side[row] = value;
  }
  for (std::int64_t reverse = dimension; reverse > 0; --reverse) {
    const std::int64_t row = reverse - 1;
    double value = right_hand_side[row];
    for (std::int64_t column = row + 1; column < dimension; ++column) {
      value -= matrix[column * leading_dimension + row] * right_hand_side[column];
    }
    value /= matrix[row * leading_dimension + row];
    if (!isfinite(value)) {
      return false;
    }
    right_hand_side[row] = value;
  }
  return true;
}

__device__ void record_numeric_failure(Gfn2SccMixerDeviceState state, std::int64_t system) {
  state.system_statuses[system] = XTBLOOM_STATUS_INTERNAL_ERROR;
}

__global__ void mix_broyden_kernel(
    Gfn2SccDeviceBatch batch, Gfn2WavefunctionLayoutView layout, Gfn2SccMixerDevicePolicy policy,
    Gfn2SccIterationDeviceActivity activity, Gfn2SccDeviceConstMultipoles raw,
    Gfn2SccDeviceMultipoles next_mixed, Gfn2SccMixerDeviceState state,
    Gfn2SccMixerDeviceWorkspace workspace, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int active;
  __shared__ int valid;
  __shared__ double residual_norm;
  __shared__ double system_residual_rms;
  __shared__ double system_residual_maximum;
  __shared__ double inverse_difference_norm;
  __shared__ double new_omega;
  __shared__ std::uint64_t old_iteration;
  __shared__ std::int64_t history_count;
  __shared__ std::int64_t new_slot;

  if (threadIdx.x == 0) {
    active = 0;
    valid = 1;
    /* Canonical sequence and member activity precede every member-local read,
     * including offsets and private mixer diagnostics/history. */
    if (atomicAdd(const_cast<std::uint32_t*>(activity.sequence_active), 0u) != 1u) {
      valid = 0;
    } else {
      const std::uint8_t requested = activity.active_mask[system];
      if (requested > 1u) {
        record_error(device_error, Gfn2SccMixerDeviceError::kInvalidState);
        valid = 0;
      } else if (requested == 1u && atomicAdd(workspace.sequence_active, 0u) == 1u) {
        active = 1;
        const std::uint8_t initialized = state.initialized[system];
        if (initialized != 1u) {
          record_error(device_error, Gfn2SccMixerDeviceError::kInvalidState);
          record_numeric_failure(state, system);
          active = 0;
          valid = 0;
        }
      }
    }
  }
  __syncthreads();
  if (active == 0) {
    return;
  }

  const std::int64_t vector_begin = system_vector_begin(batch, layout, policy, system);
  const std::int64_t dimension = system_dimension(batch, layout, policy, system);
  const std::int64_t history_begin = vector_begin * policy.history_size;
  const std::int64_t beta_begin = system * policy.history_size * policy.history_size;
  const std::int64_t coefficient_begin = system * policy.history_size;

  for (std::int64_t component = threadIdx.x; component < dimension; component += blockDim.x) {
    const double raw_value = load_component(batch, layout, policy, raw, system, component);
    const double current_value = state.current_inputs[vector_begin + component];
    const double residual = raw_value - current_value;
    workspace.residual[vector_begin + component] = residual;
    if (!isfinite(raw_value)) {
      record_error(device_error, Gfn2SccMixerDeviceError::kNonfiniteRawMultipole);
      atomicExch(&valid, 0);
    } else if (!isfinite(current_value)) {
      record_error(device_error, Gfn2SccMixerDeviceError::kInvalidState);
      atomicExch(&valid, 0);
    } else if (!isfinite(residual)) {
      record_error(device_error, Gfn2SccMixerDeviceError::kNonfiniteResidual);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    if (threadIdx.x == 0) {
      record_numeric_failure(state, system);
    }
    return;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    /* CPU convergence uses a strict threshold, so even a last-bit change from
     * a tree reduction can flip a member. Preserve CPU's packed qsh/dipole/
     * quadrupole accumulation order after parallel validation. */
    bool serial_valid = true;
    double residual_square = 0.0;
    system_residual_maximum = 0.0;
    for (std::int64_t component = 0; component < dimension; ++component) {
      const double residual = workspace.residual[vector_begin + component];
      const double updated = __dadd_rn(residual_square, __dmul_rn(residual, residual));
      if (!isfinite(updated)) {
        record_error(device_error, Gfn2SccMixerDeviceError::kNonfiniteResidual);
        serial_valid = false;
        break;
      }
      residual_square = updated;
      system_residual_maximum = fmax(system_residual_maximum, fabs(residual));
    }
    if (serial_valid) {
      residual_norm = sqrt(residual_square);
      system_residual_rms = residual_norm / sqrt(static_cast<double>(dimension));
      old_iteration = state.iterations[system];
      if (!isfinite(residual_norm) || !isfinite(system_residual_rms) ||
          !isfinite(system_residual_maximum)) {
        record_error(device_error, Gfn2SccMixerDeviceError::kNonfiniteResidual);
        serial_valid = false;
      } else if (old_iteration == kMaximumUint64) {
        record_error(device_error, Gfn2SccMixerDeviceError::kIterationOverflow);
        serial_valid = false;
      }
    }
    if (!serial_valid) {
      valid = 0;
    }
  }
  __syncthreads();
  if (valid == 0) {
    if (threadIdx.x == 0) {
      record_numeric_failure(state, system);
    }
    return;
  }
  __syncthreads();

  if (old_iteration == 0u) {
    for (std::int64_t component = threadIdx.x; component < dimension; component += blockDim.x) {
      const double value = state.current_inputs[vector_begin + component] +
                           policy.damping * workspace.residual[vector_begin + component];
      workspace.mixed[vector_begin + component] = value;
      if (!isfinite(value)) {
        record_error(device_error, Gfn2SccMixerDeviceError::kNonfiniteMixedMultipole);
        atomicExch(&valid, 0);
      }
    }
  } else {
    for (std::int64_t component = threadIdx.x; component < dimension; component += blockDim.x) {
      const double value = workspace.residual[vector_begin + component] -
                           state.previous_residuals[vector_begin + component];
      workspace.delta_f[vector_begin + component] = value;
      if (!isfinite(value)) {
        record_error(device_error, Gfn2SccMixerDeviceError::kNonfiniteDifference);
        atomicExch(&valid, 0);
      }
    }
    __syncthreads();
    if (valid == 0) {
      if (threadIdx.x == 0) {
        record_numeric_failure(state, system);
      }
      return;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      /* Match the CPU normalization order for the same boundary stability as
       * the residual convergence metrics above. */
      bool serial_valid = true;
      double difference_square = 0.0;
      for (std::int64_t component = 0; component < dimension; ++component) {
        const double difference = workspace.delta_f[vector_begin + component];
        const double updated = __dadd_rn(difference_square, __dmul_rn(difference, difference));
        if (!isfinite(updated)) {
          record_error(device_error, Gfn2SccMixerDeviceError::kNonfiniteDifference);
          serial_valid = false;
          break;
        }
        difference_square = updated;
      }
      if (serial_valid) {
        const double norm = sqrt(difference_square);
        const double denominator = fmax(norm, kDoubleEpsilon);
        inverse_difference_norm = 1.0 / denominator;
        if (!isfinite(norm) || !isfinite(inverse_difference_norm)) {
          record_error(device_error, Gfn2SccMixerDeviceError::kNormalizationFailure);
          serial_valid = false;
        }
        new_omega = residual_norm > kOmegaFactor / kMaximumOmega
                        ? fmax(kMinimumOmega, kOmegaFactor / residual_norm)
                        : kMaximumOmega;
        if (!isfinite(new_omega)) {
          record_error(device_error, Gfn2SccMixerDeviceError::kNonfiniteWeight);
          serial_valid = false;
        }
        history_count = static_cast<std::int64_t>(
            min(static_cast<std::uint64_t>(policy.history_size), old_iteration));
        new_slot = static_cast<std::int64_t>((old_iteration - 1u) %
                                             static_cast<std::uint64_t>(policy.history_size));
      }
      if (!serial_valid) {
        valid = 0;
      }
    }
    __syncthreads();
    if (valid == 0) {
      if (threadIdx.x == 0) {
        record_numeric_failure(state, system);
      }
      return;
    }
    __syncthreads();

    for (std::int64_t component = threadIdx.x; component < dimension; component += blockDim.x) {
      const std::int64_t index = vector_begin + component;
      const double normalized = workspace.delta_f[index] * inverse_difference_norm;
      const double update =
          policy.damping * normalized +
          inverse_difference_norm * (state.current_inputs[index] - state.previous_inputs[index]);
      workspace.delta_f[index] = normalized;
      workspace.new_u[index] = update;
      if (!isfinite(normalized) || !isfinite(update)) {
        record_error(device_error, Gfn2SccMixerDeviceError::kNonfiniteUpdate);
        atomicExch(&valid, 0);
      }
    }
    __syncthreads();
    if (valid == 0) {
      if (threadIdx.x == 0) {
        record_numeric_failure(state, system);
      }
      return;
    }
    __syncthreads();

    const std::uint64_t first_iteration =
        old_iteration - static_cast<std::uint64_t>(history_count) + 1u;
    for (std::int64_t row = threadIdx.x; row < history_count; row += blockDim.x) {
      const std::uint64_t represented_row = first_iteration + static_cast<std::uint64_t>(row);
      const std::int64_t row_slot = static_cast<std::int64_t>(
          (represented_row - 1u) % static_cast<std::uint64_t>(policy.history_size));
      const double* const row_df = row_slot == new_slot
                                       ? workspace.delta_f + vector_begin
                                       : state.df_history + history_begin + row_slot * dimension;
      const double row_omega =
          row_slot == new_slot ? new_omega : state.omega[system * policy.history_size + row_slot];
      double coefficient_dot = 0.0;
      if (!isfinite(row_omega)) {
        record_error(device_error, Gfn2SccMixerDeviceError::kNonfiniteWeight);
        atomicExch(&valid, 0);
      }
      for (std::int64_t component = 0; component < dimension; ++component) {
        coefficient_dot += row_df[component] * workspace.residual[vector_begin + component];
        if (!isfinite(coefficient_dot)) {
          record_error(device_error, Gfn2SccMixerDeviceError::kNonfiniteCoefficient);
          atomicExch(&valid, 0);
          break;
        }
      }
      const double coefficient = row_omega * coefficient_dot;
      workspace.coefficients[coefficient_begin + row] = coefficient;
      if (!isfinite(coefficient)) {
        record_error(device_error, Gfn2SccMixerDeviceError::kNonfiniteCoefficient);
        atomicExch(&valid, 0);
      }
      for (std::int64_t column = 0; column < history_count; ++column) {
        const std::uint64_t represented_column =
            first_iteration + static_cast<std::uint64_t>(column);
        const std::int64_t column_slot = static_cast<std::int64_t>(
            (represented_column - 1u) % static_cast<std::uint64_t>(policy.history_size));
        const double* const column_df =
            column_slot == new_slot ? workspace.delta_f + vector_begin
                                    : state.df_history + history_begin + column_slot * dimension;
        const double column_omega = column_slot == new_slot
                                        ? new_omega
                                        : state.omega[system * policy.history_size + column_slot];
        double overlap = 0.0;
        if (!isfinite(column_omega)) {
          record_error(device_error, Gfn2SccMixerDeviceError::kNonfiniteWeight);
          atomicExch(&valid, 0);
        }
        for (std::int64_t component = 0; component < dimension; ++component) {
          overlap += row_df[component] * column_df[component];
          if (!isfinite(overlap)) {
            record_error(device_error, Gfn2SccMixerDeviceError::kNonfiniteHistory);
            atomicExch(&valid, 0);
            break;
          }
        }
        double value = row_omega * column_omega * overlap;
        if (row == column) {
          value += kOmegaZero * kOmegaZero;
        }
        workspace.beta[beta_begin + row * policy.history_size + column] = value;
        if (!isfinite(value)) {
          record_error(device_error, Gfn2SccMixerDeviceError::kNonfiniteHistory);
          atomicExch(&valid, 0);
        }
      }
    }
    __syncthreads();
    if (valid == 0) {
      if (threadIdx.x == 0) {
        record_numeric_failure(state, system);
      }
      return;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      if (!cholesky_solve(workspace.beta + beta_begin, workspace.coefficients + coefficient_begin,
                          history_count, policy.history_size)) {
        record_error(device_error, Gfn2SccMixerDeviceError::kUnusableBroydenSystem);
        valid = 0;
      }
    }
    __syncthreads();
    if (valid == 0) {
      if (threadIdx.x == 0) {
        record_numeric_failure(state, system);
      }
      return;
    }
    __syncthreads();

    for (std::int64_t component = threadIdx.x; component < dimension; component += blockDim.x) {
      const std::int64_t index = vector_begin + component;
      double value = state.current_inputs[index] + policy.damping * workspace.residual[index];
      for (std::int64_t history = 0; history < history_count; ++history) {
        const std::uint64_t represented = first_iteration + static_cast<std::uint64_t>(history);
        const std::int64_t slot = static_cast<std::int64_t>(
            (represented - 1u) % static_cast<std::uint64_t>(policy.history_size));
        const double omega =
            slot == new_slot ? new_omega : state.omega[system * policy.history_size + slot];
        const double* const u = slot == new_slot
                                    ? workspace.new_u + vector_begin
                                    : state.u_history + history_begin + slot * dimension;
        value -= omega * workspace.coefficients[coefficient_begin + history] * u[component];
      }
      workspace.mixed[index] = value;
      if (!isfinite(value)) {
        record_error(device_error, Gfn2SccMixerDeviceError::kNonfiniteMixedMultipole);
        atomicExch(&valid, 0);
      }
    }
  }
  __syncthreads();
  if (valid == 0) {
    if (threadIdx.x == 0) {
      record_numeric_failure(state, system);
    }
    return;
  }

  if (old_iteration != 0u) {
    for (std::int64_t component = threadIdx.x; component < dimension; component += blockDim.x) {
      state.df_history[history_begin + new_slot * dimension + component] =
          workspace.delta_f[vector_begin + component];
      state.u_history[history_begin + new_slot * dimension + component] =
          workspace.new_u[vector_begin + component];
    }
    if (threadIdx.x == 0) {
      state.omega[system * policy.history_size + new_slot] = new_omega;
    }
  }
  for (std::int64_t component = threadIdx.x; component < dimension; component += blockDim.x) {
    const std::int64_t index = vector_begin + component;
    const double mixed_value = workspace.mixed[index];
    state.previous_inputs[index] = state.current_inputs[index];
    state.previous_residuals[index] = workspace.residual[index];
    state.current_inputs[index] = mixed_value;
    store_component(batch, layout, policy, next_mixed, system, component, mixed_value);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    state.residual_rms[system] = system_residual_rms;
    state.residual_maximum[system] = system_residual_maximum;
    state.iterations[system] = old_iteration + 1u;
    state.system_statuses[system] = XTBLOOM_STATUS_SUCCESS;
    state.residual_converged[system] = system_residual_rms < policy.rms_tolerance &&
                                               system_residual_maximum < policy.maximum_tolerance
                                           ? 1u
                                           : 0u;
  }
}

bool checked_multiply(std::int64_t first, std::int64_t second, std::int64_t* product) noexcept {
  if (first < 0 || second < 0 ||
      (first != 0 && second > std::numeric_limits<std::int64_t>::max() / first)) {
    return false;
  }
  *product = first * second;
  return true;
}

bool checked_add(std::int64_t increment, std::int64_t* value) noexcept {
  if (increment < 0 || *value > std::numeric_limits<std::int64_t>::max() - increment) {
    return false;
  }
  *value += increment;
  return true;
}

bool is_aligned(const void* pointer, std::size_t alignment) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

bool is_empty_or_aligned(const void* pointer, std::int64_t elements,
                         std::size_t alignment) noexcept {
  return elements == 0 ? pointer == nullptr : is_aligned(pointer, alignment);
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool make_address_range(const void* pointer, std::int64_t elements, std::size_t element_size,
                        AddressRange* range) noexcept {
  if (elements <= 0 || element_size == 0u ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size) ||
      pointer == nullptr) {
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(elements) * element_size;
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  *range = {begin, begin + bytes};
  return true;
}

bool make_optional_address_range(const void* pointer, std::int64_t elements,
                                 std::size_t element_size, AddressRange* range) noexcept {
  if (elements == 0) {
    *range = {};
    return pointer == nullptr;
  }
  return make_address_range(pointer, elements, element_size, range);
}

bool ranges_overlap(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin < second.end && second.begin < first.end;
}

bool same_range(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin == second.begin && first.end == second.end;
}

template <std::size_t Count>
bool pairwise_disjoint(const std::array<AddressRange, Count>& ranges) noexcept {
  for (std::size_t first = 0u; first < Count; ++first) {
    for (std::size_t second = first + 1u; second < Count; ++second) {
      if (ranges_overlap(ranges[first], ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

struct ValidatedDimensions {
  std::int64_t shell_elements = 0;
  std::int64_t atom_elements = 0;
  std::int64_t dipole_elements = 0;
  std::int64_t quadrupole_elements = 0;
  std::int64_t vector_elements = 0;
  std::int64_t history_elements = 0;
  std::int64_t omega_elements = 0;
  std::int64_t beta_elements = 0;
  std::int64_t coefficient_elements = 0;
};

bool valid_layout_binding(const Gfn2SccDeviceBatch& batch,
                          const Gfn2WavefunctionLayoutView& layout) noexcept {
  if (layout.spin_channels == nullptr) {
    return layout.batch_size == 0 && layout.plan_token == 0u;
  }
  return layout.memory_space == Gfn2PlanMemorySpace::kCudaDevice &&
         layout.plan_token == batch.plan_token && layout.batch_size == batch.batch_size &&
         layout.total_spin_channels >= batch.batch_size &&
         layout.total_spin_channels <= 2 * batch.batch_size && layout.total_spin_orbitals > 0 &&
         layout.total_spin_matrix_elements > 0 && layout.total_spin_shells >= batch.total_shells &&
         layout.total_spin_shells - batch.total_shells <= batch.total_shells &&
         layout.total_spin_atoms >= batch.total_atoms &&
         layout.total_spin_atoms - batch.total_atoms <= batch.total_atoms &&
         layout.spin_channel_count == batch.batch_size &&
         layout.spin_channel_offset_count == batch.batch_size + 1 &&
         layout.spin_orbital_offset_count == batch.batch_size + 1 &&
         layout.spin_matrix_offset_count == batch.batch_size + 1 &&
         layout.spin_shell_offset_count == batch.batch_size + 1 &&
         layout.spin_atom_offset_count == batch.batch_size + 1 &&
         is_aligned(layout.spin_channels, alignof(std::int32_t)) &&
         is_aligned(layout.spin_channel_offsets, alignof(std::int64_t)) &&
         is_aligned(layout.spin_orbital_offsets, alignof(std::int64_t)) &&
         is_aligned(layout.spin_matrix_offsets, alignof(std::int64_t)) &&
         is_aligned(layout.spin_shell_offsets, alignof(std::int64_t)) &&
         is_aligned(layout.spin_atom_offsets, alignof(std::int64_t));
}

bool validate_dimensions(const Gfn2SccDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
                         const Gfn2SccMixerDevicePolicy& policy,
                         ValidatedDimensions* dimensions) noexcept {
  std::int64_t atomic_components = 0;
  std::int64_t history_squared = 0;
  dimensions->shell_elements =
      layout.spin_channels == nullptr ? batch.total_shells : layout.total_spin_shells;
  dimensions->atom_elements =
      layout.spin_channels == nullptr ? batch.total_atoms : layout.total_spin_atoms;
  return batch.batch_size > 0 && batch.batch_size <= std::numeric_limits<int>::max() &&
         batch.total_shells > 0 && batch.total_atoms > 0 &&
         batch.batch_size != std::numeric_limits<std::int64_t>::max() &&
         batch.shell_offset_count == batch.batch_size + 1 &&
         batch.atom_offset_count == batch.batch_size + 1 && batch.plan_token != 0u &&
         policy.plan_token == batch.plan_token && policy.history_size > 0 &&
         (policy.atomic_multipole_components == 0 ||
          policy.atomic_multipole_components == kMultipoleComponentsPerAtom) &&
         std::isfinite(policy.damping) && policy.damping > 0.0 && policy.damping <= 1.0 &&
         std::isfinite(policy.rms_tolerance) && policy.rms_tolerance > 0.0 &&
         std::isfinite(policy.maximum_tolerance) && policy.maximum_tolerance > 0.0 &&
         valid_layout_binding(batch, layout) &&
         checked_multiply(policy.atomic_multipole_components == 0 ? 0 : dimensions->atom_elements,
                          kDipoleComponents, &dimensions->dipole_elements) &&
         checked_multiply(policy.atomic_multipole_components == 0 ? 0 : dimensions->atom_elements,
                          kQuadrupoleComponents, &dimensions->quadrupole_elements) &&
         checked_multiply(dimensions->atom_elements,
                          static_cast<std::int64_t>(policy.atomic_multipole_components),
                          &atomic_components) &&
         checked_add(dimensions->shell_elements, &atomic_components) &&
         (dimensions->vector_elements = atomic_components, true) &&
         checked_multiply(dimensions->vector_elements, policy.history_size,
                          &dimensions->history_elements) &&
         checked_multiply(batch.batch_size, policy.history_size, &dimensions->omega_elements) &&
         checked_multiply(policy.history_size, policy.history_size, &history_squared) &&
         checked_multiply(batch.batch_size, history_squared, &dimensions->beta_elements) &&
         checked_multiply(batch.batch_size, policy.history_size, &dimensions->coefficient_elements);
}

bool valid_const_multipoles(const Gfn2SccDeviceConstMultipoles& view,
                            const Gfn2SccDeviceBatch& batch,
                            const ValidatedDimensions& dimensions) noexcept {
  return view.plan_token == batch.plan_token && view.shell_elements == dimensions.shell_elements &&
         view.dipole_elements == dimensions.dipole_elements &&
         view.quadrupole_elements == dimensions.quadrupole_elements &&
         is_aligned(view.shell_charges, alignof(double)) &&
         is_empty_or_aligned(view.atomic_dipoles, dimensions.dipole_elements, alignof(double)) &&
         is_empty_or_aligned(view.atomic_quadrupoles, dimensions.quadrupole_elements,
                             alignof(double));
}

bool valid_multipoles(const Gfn2SccDeviceMultipoles& view, const Gfn2SccDeviceBatch& batch,
                      const ValidatedDimensions& dimensions) noexcept {
  return view.plan_token == batch.plan_token && view.shell_elements == dimensions.shell_elements &&
         view.dipole_elements == dimensions.dipole_elements &&
         view.quadrupole_elements == dimensions.quadrupole_elements &&
         is_aligned(view.shell_charges, alignof(double)) &&
         is_empty_or_aligned(view.atomic_dipoles, dimensions.dipole_elements, alignof(double)) &&
         is_empty_or_aligned(view.atomic_quadrupoles, dimensions.quadrupole_elements,
                             alignof(double));
}

bool valid_state(const Gfn2SccMixerDeviceState& state, const Gfn2SccDeviceBatch& batch,
                 const ValidatedDimensions& dimensions) noexcept {
  return state.plan_token == batch.plan_token &&
         state.total_vector_elements == dimensions.vector_elements &&
         state.history_elements == dimensions.history_elements &&
         state.omega_elements == dimensions.omega_elements &&
         state.batch_elements == batch.batch_size &&
         is_aligned(state.current_inputs, alignof(double)) &&
         is_aligned(state.previous_inputs, alignof(double)) &&
         is_aligned(state.previous_residuals, alignof(double)) &&
         is_aligned(state.df_history, alignof(double)) &&
         is_aligned(state.u_history, alignof(double)) && is_aligned(state.omega, alignof(double)) &&
         is_aligned(state.residual_rms, alignof(double)) &&
         is_aligned(state.residual_maximum, alignof(double)) &&
         is_aligned(state.iterations, alignof(std::uint64_t)) &&
         is_aligned(state.restart_counts, alignof(std::uint64_t)) &&
         is_aligned(state.system_statuses, alignof(xtbloom_status_t)) &&
         is_aligned(state.initialized, alignof(std::uint8_t)) &&
         is_aligned(state.residual_converged, alignof(std::uint8_t));
}

bool valid_activity(const Gfn2SccIterationDeviceActivity& activity,
                    const Gfn2SccDeviceBatch& batch) noexcept {
  return activity.plan_token == batch.plan_token && activity.batch_elements == batch.batch_size &&
         activity.sequence_elements == 1 &&
         is_aligned(activity.active_mask, alignof(std::uint8_t)) &&
         is_aligned(activity.sequence_active, alignof(std::uint32_t));
}

bool valid_workspace(const Gfn2SccMixerDeviceWorkspace& workspace, const Gfn2SccDeviceBatch& batch,
                     const ValidatedDimensions& dimensions) noexcept {
  return workspace.plan_token == batch.plan_token &&
         workspace.vector_elements == dimensions.vector_elements &&
         workspace.beta_elements == dimensions.beta_elements &&
         workspace.coefficient_elements == dimensions.coefficient_elements &&
         workspace.sequence_elements == 1 && is_aligned(workspace.residual, alignof(double)) &&
         is_aligned(workspace.mixed, alignof(double)) &&
         is_aligned(workspace.delta_f, alignof(double)) &&
         is_aligned(workspace.new_u, alignof(double)) &&
         is_aligned(workspace.beta, alignof(double)) &&
         is_aligned(workspace.coefficients, alignof(double)) &&
         is_aligned(workspace.sequence_active, alignof(std::uint32_t));
}

bool validate_ranges(const Gfn2SccDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
                     const ValidatedDimensions& dimensions,
                     const Gfn2SccDeviceConstMultipoles& input,
                     const Gfn2SccDeviceMultipoles* output,
                     const Gfn2SccIterationDeviceActivity* activity,
                     const Gfn2SccMixerDeviceState& state,
                     const Gfn2SccMixerDeviceWorkspace& workspace,
                     std::uint32_t* device_error) noexcept {
  std::array<AddressRange, 13> reads{};
  std::array<AddressRange, 24> writes;
  if (!make_address_range(batch.shell_offsets, batch.shell_offset_count,
                          sizeof(*batch.shell_offsets), &reads[0]) ||
      !make_address_range(batch.atom_offsets, batch.atom_offset_count, sizeof(*batch.atom_offsets),
                          &reads[1]) ||
      !make_address_range(input.shell_charges, dimensions.shell_elements, sizeof(double),
                          &reads[8]) ||
      !make_optional_address_range(input.atomic_dipoles, dimensions.dipole_elements, sizeof(double),
                                   &reads[9]) ||
      !make_optional_address_range(input.atomic_quadrupoles, dimensions.quadrupole_elements,
                                   sizeof(double), &reads[10]) ||
      !make_address_range(state.current_inputs, dimensions.vector_elements, sizeof(double),
                          &writes[0]) ||
      !make_address_range(state.previous_inputs, dimensions.vector_elements, sizeof(double),
                          &writes[1]) ||
      !make_address_range(state.previous_residuals, dimensions.vector_elements, sizeof(double),
                          &writes[2]) ||
      !make_address_range(state.df_history, dimensions.history_elements, sizeof(double),
                          &writes[3]) ||
      !make_address_range(state.u_history, dimensions.history_elements, sizeof(double),
                          &writes[4]) ||
      !make_address_range(state.omega, dimensions.omega_elements, sizeof(double), &writes[5]) ||
      !make_address_range(state.residual_rms, batch.batch_size, sizeof(double), &writes[6]) ||
      !make_address_range(state.residual_maximum, batch.batch_size, sizeof(double), &writes[7]) ||
      !make_address_range(state.iterations, batch.batch_size, sizeof(std::uint64_t), &writes[8]) ||
      !make_address_range(state.restart_counts, batch.batch_size, sizeof(std::uint64_t),
                          &writes[9]) ||
      !make_address_range(state.system_statuses, batch.batch_size, sizeof(xtbloom_status_t),
                          &writes[10]) ||
      !make_address_range(state.initialized, batch.batch_size, sizeof(std::uint8_t), &writes[11]) ||
      !make_address_range(state.residual_converged, batch.batch_size, sizeof(std::uint8_t),
                          &writes[12]) ||
      !make_address_range(workspace.residual, dimensions.vector_elements, sizeof(double),
                          &writes[13]) ||
      !make_address_range(workspace.mixed, dimensions.vector_elements, sizeof(double),
                          &writes[14]) ||
      !make_address_range(workspace.delta_f, dimensions.vector_elements, sizeof(double),
                          &writes[15]) ||
      !make_address_range(workspace.new_u, dimensions.vector_elements, sizeof(double),
                          &writes[16]) ||
      !make_address_range(workspace.beta, dimensions.beta_elements, sizeof(double), &writes[17]) ||
      !make_address_range(workspace.coefficients, dimensions.coefficient_elements, sizeof(double),
                          &writes[18]) ||
      !make_address_range(workspace.sequence_active, 1, sizeof(std::uint32_t), &writes[19]) ||
      !make_address_range(device_error, 1, sizeof(std::uint32_t), &writes[23])) {
    return false;
  }
  if (layout.spin_channels != nullptr &&
      (!make_address_range(layout.spin_channels, layout.spin_channel_count,
                           sizeof(*layout.spin_channels), &reads[2]) ||
       !make_address_range(layout.spin_channel_offsets, layout.spin_channel_offset_count,
                           sizeof(*layout.spin_channel_offsets), &reads[3]) ||
       !make_address_range(layout.spin_orbital_offsets, layout.spin_orbital_offset_count,
                           sizeof(*layout.spin_orbital_offsets), &reads[4]) ||
       !make_address_range(layout.spin_matrix_offsets, layout.spin_matrix_offset_count,
                           sizeof(*layout.spin_matrix_offsets), &reads[5]) ||
       !make_address_range(layout.spin_shell_offsets, layout.spin_shell_offset_count,
                           sizeof(*layout.spin_shell_offsets), &reads[6]) ||
       !make_address_range(layout.spin_atom_offsets, layout.spin_atom_offset_count,
                           sizeof(*layout.spin_atom_offsets), &reads[7]))) {
    return false;
  }
  if (activity != nullptr &&
      (!make_address_range(activity->active_mask, batch.batch_size, sizeof(std::uint8_t),
                           &reads[11]) ||
       !make_address_range(activity->sequence_active, 1, sizeof(std::uint32_t), &reads[12]))) {
    return false;
  }
  if (!pairwise_disjoint(reads)) {
    return false;
  }
  if (output != nullptr) {
    if (!make_address_range(output->shell_charges, dimensions.shell_elements, sizeof(double),
                            &writes[20]) ||
        !make_optional_address_range(output->atomic_dipoles, dimensions.dipole_elements,
                                     sizeof(double), &writes[21]) ||
        !make_optional_address_range(output->atomic_quadrupoles, dimensions.quadrupole_elements,
                                     sizeof(double), &writes[22])) {
      return false;
    }
  } else {
    /* Non-output launchers still require pairwise-disjoint placeholders. */
    writes[20] = {};
    writes[21] = {};
    writes[22] = {};
  }

  for (std::size_t first = 0u; first < writes.size(); ++first) {
    if (writes[first].begin == writes[first].end) {
      continue;
    }
    for (std::size_t second = first + 1u; second < writes.size(); ++second) {
      if (writes[second].begin != writes[second].end &&
          ranges_overlap(writes[first], writes[second])) {
        return false;
      }
    }
  }
  for (std::size_t write = 0u; write < writes.size(); ++write) {
    if (writes[write].begin == writes[write].end) {
      continue;
    }
    for (std::size_t read = 0u; read < reads.size(); ++read) {
      const bool allowed_in_place = output != nullptr && write >= 20u && write <= 22u &&
                                    read == write - 12u && same_range(writes[write], reads[read]);
      if (ranges_overlap(writes[write], reads[read]) && !allowed_in_place) {
        return false;
      }
    }
  }
  return true;
}

cudaError_t validate_common(
    const Gfn2SccDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2SccMixerDevicePolicy& policy, const Gfn2SccDeviceConstMultipoles& input,
    const Gfn2SccDeviceMultipoles* output, const Gfn2SccIterationDeviceActivity* activity,
    const Gfn2SccMixerDeviceState& state, const Gfn2SccMixerDeviceWorkspace& workspace,
    std::uint32_t* device_error, ValidatedDimensions* dimensions) noexcept {
  if (!validate_dimensions(batch, layout, policy, dimensions) ||
      !is_aligned(batch.shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !valid_const_multipoles(input, batch, *dimensions) ||
      (output != nullptr && !valid_multipoles(*output, batch, *dimensions)) ||
      (activity != nullptr && !valid_activity(*activity, batch)) ||
      !valid_state(state, batch, *dimensions) || !valid_workspace(workspace, batch, *dimensions) ||
      !is_aligned(device_error, alignof(std::uint32_t)) ||
      !validate_ranges(batch, layout, *dimensions, input, output, activity, state, workspace,
                       device_error)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t launch_topology_and_capture(const Gfn2SccDeviceBatch& batch,
                                        const Gfn2WavefunctionLayoutView& layout,
                                        const Gfn2SccMixerDeviceWorkspace& workspace,
                                        std::uint32_t* device_error, cudaStream_t stream) noexcept {
  topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, layout, device_error);
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace.sequence_active);
  return cudaPeekAtLastError();
}

}  // namespace

cudaError_t initialize_gfn2_scc_mixer_cuda(const Gfn2SccDeviceBatch& batch,
                                           const Gfn2SccMixerDevicePolicy& policy,
                                           const Gfn2SccDeviceConstMultipoles& initial,
                                           const Gfn2SccMixerDeviceState& state,
                                           const Gfn2SccMixerDeviceWorkspace& workspace,
                                           std::uint32_t* device_error,
                                           cudaStream_t stream) noexcept {
  return initialize_gfn2_scc_mixer_cuda(batch, Gfn2WavefunctionLayoutView{}, policy, initial, state,
                                        workspace, device_error, stream);
}

cudaError_t initialize_gfn2_scc_mixer_cuda(
    const Gfn2SccDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2SccMixerDevicePolicy& policy, const Gfn2SccDeviceConstMultipoles& initial,
    const Gfn2SccMixerDeviceState& state, const Gfn2SccMixerDeviceWorkspace& workspace,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  ValidatedDimensions dimensions;
  cudaError_t status = validate_common(batch, layout, policy, initial, nullptr, nullptr, state,
                                       workspace, device_error, &dimensions);
  if (status != cudaSuccess) {
    return status;
  }
  topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, layout, device_error);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  initial_values_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock,
                                    0, stream>>>(batch, layout, policy, initial, device_error);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace.sequence_active);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  initialize_state_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                            stream>>>(batch, layout, policy, initial, state, workspace);
  return cudaPeekAtLastError();
}

cudaError_t restart_gfn2_scc_mixer_system_cuda(
    const Gfn2SccDeviceBatch& batch, const Gfn2SccMixerDevicePolicy& policy, std::int64_t system,
    const Gfn2SccDeviceConstMultipoles& current_public, const Gfn2SccMixerDeviceState& state,
    const Gfn2SccMixerDeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  return restart_gfn2_scc_mixer_system_cuda(batch, Gfn2WavefunctionLayoutView{}, policy, system,
                                            current_public, state, workspace, device_error, stream);
}

cudaError_t restart_gfn2_scc_mixer_system_cuda(
    const Gfn2SccDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2SccMixerDevicePolicy& policy, std::int64_t system,
    const Gfn2SccDeviceConstMultipoles& current_public, const Gfn2SccMixerDeviceState& state,
    const Gfn2SccMixerDeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  ValidatedDimensions dimensions;
  cudaError_t status = validate_common(batch, layout, policy, current_public, nullptr, nullptr,
                                       state, workspace, device_error, &dimensions);
  if (status != cudaSuccess || system < 0 || system >= batch.batch_size) {
    return cudaErrorInvalidValue;
  }
  status = launch_topology_and_capture(batch, layout, workspace, device_error, stream);
  if (status != cudaSuccess) {
    return status;
  }
  restart_system_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
      batch, layout, policy, system, current_public, state, workspace, device_error);
  return cudaPeekAtLastError();
}

cudaError_t mix_gfn2_scc_broyden_cuda(const Gfn2SccDeviceBatch& batch,
                                      const Gfn2SccMixerDevicePolicy& policy,
                                      const Gfn2SccIterationDeviceActivity& activity,
                                      const Gfn2SccDeviceConstMultipoles& raw,
                                      const Gfn2SccDeviceMultipoles& next_mixed,
                                      const Gfn2SccMixerDeviceState& state,
                                      const Gfn2SccMixerDeviceWorkspace& workspace,
                                      std::uint32_t* device_error, cudaStream_t stream) noexcept {
  return mix_gfn2_scc_broyden_cuda(batch, Gfn2WavefunctionLayoutView{}, policy, activity, raw,
                                   next_mixed, state, workspace, device_error, stream);
}

cudaError_t mix_gfn2_scc_broyden_cuda(
    const Gfn2SccDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2SccMixerDevicePolicy& policy, const Gfn2SccIterationDeviceActivity& activity,
    const Gfn2SccDeviceConstMultipoles& raw, const Gfn2SccDeviceMultipoles& next_mixed,
    const Gfn2SccMixerDeviceState& state, const Gfn2SccMixerDeviceWorkspace& workspace,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  ValidatedDimensions dimensions;
  cudaError_t status = validate_common(batch, layout, policy, raw, &next_mixed, &activity, state,
                                       workspace, device_error, &dimensions);
  if (status != cudaSuccess) {
    return status;
  }
  begin_mixer_stage_kernel<<<1, 1, 0, stream>>>(activity, workspace, device_error);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  const unsigned int status_blocks =
      static_cast<unsigned int>((batch.batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock);
  prepare_active_mixer_statuses_kernel<<<status_blocks, kThreadsPerBlock, 0, stream>>>(activity,
                                                                                       state);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  active_topology_preflight_kernel<<<1, 1, 0, stream>>>(batch, layout, activity, workspace,
                                                        device_error);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  mix_broyden_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0, stream>>>(
      batch, layout, policy, activity, raw, next_mixed, state, workspace, device_error);
  return cudaPeekAtLastError();
}

}  // namespace xtbloom::detail::cuda
