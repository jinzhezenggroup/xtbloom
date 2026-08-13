#include <array>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_scc_potential.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;

constexpr __host__ __device__ std::uint32_t component_bit(
    Gfn2SccPotentialComponent component) noexcept {
  return static_cast<std::uint32_t>(component);
}

constexpr __host__ __device__ bool component_enabled(std::uint32_t mask,
                                                     Gfn2SccPotentialComponent component) noexcept {
  return (mask & component_bit(component)) != 0u;
}

__device__ bool is_plan_error(std::uint32_t error) {
  /* Offset partitions describe the whole launch and are validated before any
   * system kernel may dereference them.  A bad shell-to-atom entry, however,
   * belongs to exactly one ragged system and must not suppress healthy peers
   * in this or a later SCC stage. */
  return error == static_cast<std::uint32_t>(Gfn2SccPotentialDeviceError::kInvalidOffsets);
}

__device__ void record_plan_error(std::uint32_t* device_error, std::uint32_t* sequence_active,
                                  Gfn2SccPotentialDeviceError error) {
  const std::uint32_t success = static_cast<std::uint32_t>(Gfn2SccPotentialDeviceError::kSuccess);
  /* A malformed plan must fail the complete sequence closed, but it must not
   * overwrite an earlier numerical diagnostic from another system/stage. */
  atomicCAS(device_error, success, static_cast<std::uint32_t>(error));
  atomicExch(sequence_active, 0u);
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    std::uint32_t* device_error,
                                    Gfn2SccPotentialDeviceError error) {
  const std::uint32_t success = static_cast<std::uint32_t>(Gfn2SccPotentialDeviceError::kSuccess);
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  if (atomicCAS(system_errors + system, success, code) == success) {
    atomicCAS(device_error, success, code);
  }
}

__device__ bool system_is_valid(const std::uint32_t* system_errors, std::int64_t system) {
  return atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) ==
         static_cast<std::uint32_t>(Gfn2SccPotentialDeviceError::kSuccess);
}

__global__ void topology_preflight_kernel(Gfn2SccPotentialDeviceBatch batch,
                                          std::uint32_t* sequence_active,
                                          std::uint32_t* device_error) {
  if (atomicAdd(sequence_active, 0u) != 1u) {
    return;
  }
  const std::int64_t dipole_total = batch.total_atoms * kGfn2SccPotentialDipoleComponents;
  const std::int64_t quadrupole_total = batch.total_atoms * kGfn2SccPotentialQuadrupoleComponents;
  if (threadIdx.x == 0 && (batch.atom_offsets[0] != 0 || batch.batch_shell_offsets[0] != 0 ||
                           batch.qsh_offsets[0] != 0 || batch.qat_offsets[0] != 0 ||
                           batch.dipole_offsets[0] != 0 || batch.quadrupole_offsets[0] != 0 ||
                           batch.atom_offsets[batch.batch_size] != batch.total_atoms ||
                           batch.batch_shell_offsets[batch.batch_size] != batch.total_shells ||
                           batch.qsh_offsets[batch.batch_size] != batch.total_shells ||
                           batch.qat_offsets[batch.batch_size] != batch.total_atoms ||
                           batch.dipole_offsets[batch.batch_size] != dipole_total ||
                           batch.quadrupole_offsets[batch.batch_size] != quadrupole_total)) {
    record_plan_error(device_error, sequence_active, Gfn2SccPotentialDeviceError::kInvalidOffsets);
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    const std::int64_t atom_begin = batch.atom_offsets[system];
    const std::int64_t atom_end = batch.atom_offsets[system + 1];
    const std::int64_t shell_begin = batch.batch_shell_offsets[system];
    const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
    const std::int64_t qsh_begin = batch.qsh_offsets[system];
    const std::int64_t qsh_end = batch.qsh_offsets[system + 1];
    const std::int64_t qat_begin = batch.qat_offsets[system];
    const std::int64_t qat_end = batch.qat_offsets[system + 1];
    const std::int64_t dipole_begin = batch.dipole_offsets[system];
    const std::int64_t dipole_end = batch.dipole_offsets[system + 1];
    const std::int64_t quadrupole_begin = batch.quadrupole_offsets[system];
    const std::int64_t quadrupole_end = batch.quadrupole_offsets[system + 1];
    if (atom_begin < 0 || atom_begin > atom_end || atom_end > batch.total_atoms ||
        shell_begin < 0 || shell_begin > shell_end || shell_end > batch.total_shells ||
        qsh_begin < 0 || qsh_begin > qsh_end || qsh_end > batch.total_shells || qat_begin < 0 ||
        qat_begin > qat_end || qat_end > batch.total_atoms || dipole_begin < 0 ||
        dipole_begin > dipole_end || dipole_end > dipole_total || quadrupole_begin < 0 ||
        quadrupole_begin > quadrupole_end || quadrupole_end > quadrupole_total ||
        qsh_end - qsh_begin != shell_end - shell_begin ||
        qat_end - qat_begin != atom_end - atom_begin ||
        dipole_end - dipole_begin != (atom_end - atom_begin) * kGfn2SccPotentialDipoleComponents ||
        quadrupole_end - quadrupole_begin !=
            (atom_end - atom_begin) * kGfn2SccPotentialQuadrupoleComponents) {
      record_plan_error(device_error, sequence_active,
                        Gfn2SccPotentialDeviceError::kInvalidOffsets);
    }
  }
}

__global__ void capture_sequence_kernel(const std::uint32_t* device_error,
                                        std::uint32_t* sequence_active) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const std::uint32_t error = atomicAdd(const_cast<std::uint32_t*>(device_error), 0u);
    *sequence_active = is_plan_error(error) ? 0u : 1u;
  }
}

__global__ void capture_canonical_sequence_kernel(Gfn2SccIterationDeviceActivity activity,
                                                  const std::uint32_t* device_error,
                                                  std::uint32_t* sequence_active) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const bool canonical_open =
        atomicAdd(const_cast<std::uint32_t*>(activity.sequence_active), 0u) == 1u;
    const std::uint32_t error = atomicAdd(const_cast<std::uint32_t*>(device_error), 0u);
    *sequence_active = canonical_open && !is_plan_error(error) ? 1u : 0u;
  }
}

__device__ bool sequence_is_active(const Gfn2SccPotentialDeviceWorkspace& workspace) {
  return atomicAdd(workspace.sequence_active, 0u) == 1u;
}

__device__ bool canonical_member_is_active(const Gfn2SccIterationDeviceActivity& activity,
                                           const Gfn2SccPotentialDeviceWorkspace& workspace,
                                           const std::uint32_t* system_errors,
                                           std::int64_t system) {
  /* The canonical sequence is authoritative and must precede every member or
   * numerical read.  workspace.sequence_active is only this stage's report
   * latch; it never derives a second activity policy. */
  return atomicAdd(const_cast<std::uint32_t*>(activity.sequence_active), 0u) == 1u &&
         sequence_is_active(workspace) && activity.active_mask[system] == 1u &&
         system_is_valid(system_errors, system);
}

/* Active-aware plan validation follows the same gate order as the physics
 * primitives: sequence, any-active, then active topology.  Inactive offsets
 * are generation-local poison and cannot be used to validate their peers. */
__global__ void canonical_topology_preflight_kernel(Gfn2SccPotentialDeviceBatch batch,
                                                    Gfn2SccIterationDeviceActivity activity,
                                                    Gfn2SccPotentialDeviceWorkspace workspace,
                                                    std::uint32_t* device_error) {
  __shared__ int sequence_open;
  __shared__ int any_active;
  if (threadIdx.x == 0) {
    sequence_open =
        atomicAdd(const_cast<std::uint32_t*>(activity.sequence_active), 0u) == 1u ? 1 : 0;
    any_active = 0;
    *workspace.sequence_active =
        sequence_open != 0 && !is_plan_error(atomicAdd(device_error, 0u)) ? 1u : 0u;
  }
  __syncthreads();
  if (sequence_open == 0 || !sequence_is_active(workspace)) {
    return;
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    if (activity.active_mask[system] == 1u) {
      atomicExch(&any_active, 1);
    }
  }
  __syncthreads();
  if (any_active == 0 || !sequence_is_active(workspace)) {
    return;
  }

  const std::int64_t dipole_total = batch.total_atoms * kGfn2SccPotentialDipoleComponents;
  const std::int64_t quadrupole_total = batch.total_atoms * kGfn2SccPotentialQuadrupoleComponents;
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    if (activity.active_mask[system] != 1u) {
      continue;
    }
    const std::int64_t atom_begin = batch.atom_offsets[system];
    const std::int64_t atom_end = batch.atom_offsets[system + 1];
    const std::int64_t shell_begin = batch.batch_shell_offsets[system];
    const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
    const std::int64_t qsh_begin = batch.qsh_offsets[system];
    const std::int64_t qsh_end = batch.qsh_offsets[system + 1];
    const std::int64_t qat_begin = batch.qat_offsets[system];
    const std::int64_t qat_end = batch.qat_offsets[system + 1];
    const std::int64_t dipole_begin = batch.dipole_offsets[system];
    const std::int64_t dipole_end = batch.dipole_offsets[system + 1];
    const std::int64_t quadrupole_begin = batch.quadrupole_offsets[system];
    const std::int64_t quadrupole_end = batch.quadrupole_offsets[system + 1];
    bool valid =
        atom_begin >= 0 && atom_begin <= atom_end && atom_end <= batch.total_atoms &&
        shell_begin >= 0 && shell_begin <= shell_end && shell_end <= batch.total_shells &&
        qsh_begin >= 0 && qsh_begin <= qsh_end && qsh_end <= batch.total_shells && qat_begin >= 0 &&
        qat_begin <= qat_end && qat_end <= batch.total_atoms && dipole_begin >= 0 &&
        dipole_begin <= dipole_end && dipole_end <= dipole_total && quadrupole_begin >= 0 &&
        quadrupole_begin <= quadrupole_end && quadrupole_end <= quadrupole_total &&
        qsh_end - qsh_begin == shell_end - shell_begin &&
        qat_end - qat_begin == atom_end - atom_begin &&
        dipole_end - dipole_begin == (atom_end - atom_begin) * kGfn2SccPotentialDipoleComponents &&
        quadrupole_end - quadrupole_begin ==
            (atom_end - atom_begin) * kGfn2SccPotentialQuadrupoleComponents;
    if (valid && system == 0) {
      valid = atom_begin == 0 && shell_begin == 0 && qsh_begin == 0 && qat_begin == 0 &&
              dipole_begin == 0 && quadrupole_begin == 0;
    }
    if (valid && system + 1 == batch.batch_size) {
      valid = atom_end == batch.total_atoms && shell_end == batch.total_shells &&
              qsh_end == batch.total_shells && qat_end == batch.total_atoms &&
              dipole_end == dipole_total && quadrupole_end == quadrupole_total;
    }
    if (!valid) {
      record_plan_error(device_error, workspace.sequence_active,
                        Gfn2SccPotentialDeviceError::kInvalidOffsets);
    }
  }
}

__global__ void spin_potential_topology_preflight_kernel(Gfn2SccPotentialDeviceBatch batch,
                                                         Gfn2WavefunctionLayoutView layout,
                                                         Gfn2SccPotentialDeviceActivity activity,
                                                         Gfn2SccPotentialDeviceWorkspace workspace,
                                                         std::uint32_t* system_errors,
                                                         std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!sequence_is_active(workspace) || !system_is_valid(system_errors, system)) {
    return;
  }
  if (threadIdx.x != 0) {
    return;
  }
  const std::uint8_t state = activity.active_mask == nullptr ? 1u : activity.active_mask[system];
  if (state == 0u) {
    return;
  }
  if (state != 1u) {
    record_system_error(system_errors, system, device_error,
                        Gfn2SccPotentialDeviceError::kInvalidActiveMask);
    return;
  }

  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t qsh_begin = batch.qsh_offsets[system];
  const std::int64_t qsh_end = batch.qsh_offsets[system + 1];
  const std::int64_t qat_begin = batch.qat_offsets[system];
  const std::int64_t qat_end = batch.qat_offsets[system + 1];
  const std::int64_t dipole_begin = batch.dipole_offsets[system];
  const std::int64_t dipole_end = batch.dipole_offsets[system + 1];
  const std::int64_t quadrupole_begin = batch.quadrupole_offsets[system];
  const std::int64_t quadrupole_end = batch.quadrupole_offsets[system + 1];
  const std::int32_t channels = layout.spin_channels[system];
  const std::int64_t channel_begin = layout.spin_channel_offsets[system];
  const std::int64_t channel_end = layout.spin_channel_offsets[system + 1];
  const std::int64_t spin_shell_begin = layout.spin_shell_offsets[system];
  const std::int64_t spin_shell_end = layout.spin_shell_offsets[system + 1];
  const std::int64_t spin_atom_begin = layout.spin_atom_offsets[system];
  const std::int64_t spin_atom_end = layout.spin_atom_offsets[system + 1];
  const std::int64_t atoms = atom_end - atom_begin;
  const std::int64_t shells = shell_end - shell_begin;
  const std::int64_t dipole_total = batch.total_atoms * kGfn2SccPotentialDipoleComponents;
  const std::int64_t quadrupole_total = batch.total_atoms * kGfn2SccPotentialQuadrupoleComponents;
  bool valid = atom_begin >= 0 && atom_begin <= atom_end && atom_end <= batch.total_atoms &&
               shell_begin >= 0 && shell_begin <= shell_end && shell_end <= batch.total_shells &&
               qsh_begin >= 0 && qsh_begin <= qsh_end && qsh_end <= batch.total_shells &&
               qat_begin >= 0 && qat_begin <= qat_end && qat_end <= batch.total_atoms &&
               dipole_begin >= 0 && dipole_begin <= dipole_end && dipole_end <= dipole_total &&
               quadrupole_begin >= 0 && quadrupole_begin <= quadrupole_end &&
               quadrupole_end <= quadrupole_total && qsh_end - qsh_begin == shells &&
               qat_end - qat_begin == atoms &&
               dipole_end - dipole_begin == atoms * kGfn2SccPotentialDipoleComponents &&
               quadrupole_end - quadrupole_begin == atoms * kGfn2SccPotentialQuadrupoleComponents &&
               (channels == 1 || channels == 2) && channel_begin >= 0 &&
               channel_begin <= channel_end && channel_end <= layout.total_spin_channels &&
               channel_end - channel_begin == channels && spin_shell_begin >= 0 &&
               spin_shell_begin <= spin_shell_end && spin_shell_end <= layout.total_spin_shells &&
               spin_shell_end - spin_shell_begin == static_cast<std::int64_t>(channels) * shells &&
               spin_atom_begin >= 0 && spin_atom_begin <= spin_atom_end &&
               spin_atom_end <= layout.total_spin_atoms &&
               spin_atom_end - spin_atom_begin == static_cast<std::int64_t>(channels) * atoms;
  if (valid && system == 0) {
    valid = atom_begin == 0 && shell_begin == 0 && qsh_begin == 0 && qat_begin == 0 &&
            dipole_begin == 0 && quadrupole_begin == 0 && channel_begin == 0 &&
            spin_shell_begin == 0 && spin_atom_begin == 0;
  }
  if (valid && system + 1 == batch.batch_size) {
    valid = atom_end == batch.total_atoms && shell_end == batch.total_shells &&
            qsh_end == batch.total_shells && qat_end == batch.total_atoms &&
            dipole_end == dipole_total && quadrupole_end == quadrupole_total &&
            channel_end == layout.total_spin_channels &&
            spin_shell_end == layout.total_spin_shells && spin_atom_end == layout.total_spin_atoms;
  }
  if (!valid) {
    record_system_error(system_errors, system, device_error,
                        Gfn2SccPotentialDeviceError::kInvalidSpinLayout);
    return;
  }
  for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
    const std::int64_t atom = batch.shell_to_atom[shell];
    if (atom < atom_begin || atom >= atom_end) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccPotentialDeviceError::kInvalidShellMetadata);
      return;
    }
  }
}

__global__ void reduce_mixed_atomic_charge_kernel(Gfn2SccPotentialDeviceBatch batch,
                                                  Gfn2SccPotentialDeviceMixedFields mixed,
                                                  Gfn2SccIterationDeviceActivity activity,
                                                  Gfn2SccPotentialDeviceWorkspace workspace,
                                                  std::uint32_t* system_errors,
                                                  std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int active;
  __shared__ int valid;
  if (threadIdx.x == 0) {
    active = canonical_member_is_active(activity, workspace, system_errors, system) ? 1 : 0;
    valid = 1;
  }
  __syncthreads();
  if (active == 0) {
    return;
  }

  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t qsh_begin = batch.qsh_offsets[system];
  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    const std::int64_t atom = batch.shell_to_atom[shell];
    if (atom < atom_begin || atom >= atom_end) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccPotentialDeviceError::kInvalidShellMetadata);
      atomicExch(&valid, 0);
      continue;
    }
    if (!isfinite(mixed.qsh[qsh_begin + shell - shell_begin])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccPotentialDeviceError::kNonfiniteMixedShellCharge);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    double charge = 0.0;
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      if (batch.shell_to_atom[shell] != atom) {
        continue;
      }
      const double updated = charge + mixed.qsh[qsh_begin + shell - shell_begin];
      if (!isfinite(updated)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2SccPotentialDeviceError::kNonfiniteAtomicChargeReduction);
        atomicExch(&valid, 0);
        break;
      }
      charge = updated;
    }
    workspace.atom_scratch[atom] = charge;
  }
}

__global__ void publish_mixed_atomic_charge_kernel(
    Gfn2SccPotentialDeviceBatch batch, Gfn2SccIterationDeviceActivity activity,
    Gfn2SccPotentialDeviceTopologyMultipoles topology, Gfn2SccPotentialDeviceWorkspace workspace,
    const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!canonical_member_is_active(activity, workspace, system_errors, system)) {
    return;
  }
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    topology.atomic_charges[atom] = workspace.atom_scratch[atom];
  }
}

__global__ void spin_reduction_preflight_kernel(Gfn2SccPotentialDeviceBatch batch,
                                                Gfn2WavefunctionLayoutView layout,
                                                Gfn2SccIterationDeviceActivity activity,
                                                Gfn2SccPotentialDeviceWorkspace workspace,
                                                std::uint32_t* system_errors,
                                                std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!canonical_member_is_active(activity, workspace, system_errors, system) || threadIdx.x != 0) {
    return;
  }

  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int32_t channels = layout.spin_channels[system];
  const std::int64_t channel_begin = layout.spin_channel_offsets[system];
  const std::int64_t channel_end = layout.spin_channel_offsets[system + 1];
  const std::int64_t spin_shell_begin = layout.spin_shell_offsets[system];
  const std::int64_t spin_shell_end = layout.spin_shell_offsets[system + 1];
  const std::int64_t spin_atom_begin = layout.spin_atom_offsets[system];
  const std::int64_t spin_atom_end = layout.spin_atom_offsets[system + 1];
  const std::int64_t atoms = atom_end - atom_begin;
  const std::int64_t shells = shell_end - shell_begin;
  bool valid = atom_begin >= 0 && atom_begin <= atom_end && atom_end <= batch.total_atoms &&
               shell_begin >= 0 && shell_begin <= shell_end && shell_end <= batch.total_shells &&
               (channels == 1 || channels == 2) && channel_begin >= 0 &&
               channel_begin <= channel_end && channel_end <= layout.total_spin_channels &&
               channel_end - channel_begin == channels && spin_shell_begin >= 0 &&
               spin_shell_begin <= spin_shell_end && spin_shell_end <= layout.total_spin_shells &&
               spin_shell_end - spin_shell_begin == static_cast<std::int64_t>(channels) * shells &&
               spin_atom_begin >= 0 && spin_atom_begin <= spin_atom_end &&
               spin_atom_end <= layout.total_spin_atoms &&
               spin_atom_end - spin_atom_begin == static_cast<std::int64_t>(channels) * atoms;
  if (valid && system == 0) {
    valid = atom_begin == 0 && shell_begin == 0 && channel_begin == 0 && spin_shell_begin == 0 &&
            spin_atom_begin == 0;
  }
  if (valid && system + 1 == batch.batch_size) {
    valid = atom_end == batch.total_atoms && shell_end == batch.total_shells &&
            channel_end == layout.total_spin_channels &&
            spin_shell_end == layout.total_spin_shells && spin_atom_end == layout.total_spin_atoms;
  }
  if (!valid) {
    record_system_error(system_errors, system, device_error,
                        Gfn2SccPotentialDeviceError::kInvalidSpinLayout);
    return;
  }
  for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
    const std::int64_t atom = batch.shell_to_atom[shell];
    if (atom < atom_begin || atom >= atom_end) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccPotentialDeviceError::kInvalidShellMetadata);
      return;
    }
  }
}

__global__ void reduce_spin_atomic_charges_kernel(Gfn2SccPotentialDeviceBatch batch,
                                                  Gfn2WavefunctionLayoutView layout,
                                                  Gfn2SccPotentialDeviceMixedFields mixed,
                                                  Gfn2SccIterationDeviceActivity activity,
                                                  Gfn2SccPotentialDeviceWorkspace workspace,
                                                  std::uint32_t* system_errors,
                                                  std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int active;
  __shared__ int valid;
  if (threadIdx.x == 0) {
    active = canonical_member_is_active(activity, workspace, system_errors, system) ? 1 : 0;
    valid = 1;
  }
  __syncthreads();
  if (active == 0) {
    return;
  }

  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t spin_shell_begin = layout.spin_shell_offsets[system];
  const std::int64_t spin_shell_end = layout.spin_shell_offsets[system + 1];
  const std::int64_t spin_atom_begin = layout.spin_atom_offsets[system];
  const std::int64_t spin_atom_end = layout.spin_atom_offsets[system + 1];
  const std::int64_t physical_atoms = atom_end - atom_begin;
  const std::int64_t physical_shells = shell_end - shell_begin;

  for (std::int64_t shell = spin_shell_begin + threadIdx.x; shell < spin_shell_end;
       shell += blockDim.x) {
    if (!isfinite(mixed.qsh[shell])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccPotentialDeviceError::kNonfiniteMixedShellCharge);
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t element = spin_atom_begin * kGfn2SccPotentialDipoleComponents + threadIdx.x;
       element < spin_atom_end * kGfn2SccPotentialDipoleComponents; element += blockDim.x) {
    if (!isfinite(mixed.dipoles[element])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccPotentialDeviceError::kNonfiniteMixedDipole);
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t element = spin_atom_begin * kGfn2SccPotentialQuadrupoleComponents + threadIdx.x;
       element < spin_atom_end * kGfn2SccPotentialQuadrupoleComponents; element += blockDim.x) {
    if (!isfinite(mixed.quadrupoles[element])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccPotentialDeviceError::kNonfiniteMixedQuadrupole);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  for (std::int64_t spin_atom = spin_atom_begin + threadIdx.x; spin_atom < spin_atom_end;
       spin_atom += blockDim.x) {
    const std::int64_t local = spin_atom - spin_atom_begin;
    const std::int64_t channel = local / physical_atoms;
    const std::int64_t physical_atom = atom_begin + local % physical_atoms;
    double charge = 0.0;
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      if (batch.shell_to_atom[shell] != physical_atom) {
        continue;
      }
      const std::int64_t spin_shell =
          spin_shell_begin + channel * physical_shells + (shell - shell_begin);
      const double updated = charge + mixed.qsh[spin_shell];
      if (!isfinite(updated)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2SccPotentialDeviceError::kNonfiniteAtomicChargeReduction);
        atomicExch(&valid, 0);
        break;
      }
      charge = updated;
    }
    workspace.atom_scratch[spin_atom] = charge;
  }
}

__global__ void publish_spin_multipole_projection_kernel(
    Gfn2SccPotentialDeviceBatch batch, Gfn2WavefunctionLayoutView layout,
    Gfn2SccPotentialDeviceMixedFields mixed, Gfn2SccIterationDeviceActivity activity,
    Gfn2SccPotentialDeviceTopologyMultipoles spin_topology,
    Gfn2SccPotentialDeviceTopologyMultipoles physical_topology,
    Gfn2SccPotentialDeviceWorkspace workspace, const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!canonical_member_is_active(activity, workspace, system_errors, system)) {
    return;
  }
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t spin_shell_begin = layout.spin_shell_offsets[system];
  const std::int64_t spin_atom_begin = layout.spin_atom_offsets[system];
  const std::int64_t spin_atom_end = layout.spin_atom_offsets[system + 1];

  for (std::int64_t spin_atom = spin_atom_begin + threadIdx.x; spin_atom < spin_atom_end;
       spin_atom += blockDim.x) {
    spin_topology.atomic_charges[spin_atom] = workspace.atom_scratch[spin_atom];
  }
  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    physical_topology.shell_charges[shell] = mixed.qsh[spin_shell_begin + shell - shell_begin];
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const std::int64_t spin_atom = spin_atom_begin + atom - atom_begin;
    physical_topology.atomic_charges[atom] = workspace.atom_scratch[spin_atom];
  }
  for (std::int64_t element = atom_begin * kGfn2SccPotentialDipoleComponents + threadIdx.x;
       element < atom_end * kGfn2SccPotentialDipoleComponents; element += blockDim.x) {
    const std::int64_t local = element - atom_begin * kGfn2SccPotentialDipoleComponents;
    physical_topology.atomic_dipoles[element] =
        mixed.dipoles[spin_atom_begin * kGfn2SccPotentialDipoleComponents + local];
  }
  for (std::int64_t element = atom_begin * kGfn2SccPotentialQuadrupoleComponents + threadIdx.x;
       element < atom_end * kGfn2SccPotentialQuadrupoleComponents; element += blockDim.x) {
    const std::int64_t local = element - atom_begin * kGfn2SccPotentialQuadrupoleComponents;
    physical_topology.atomic_quadrupoles[element] =
        mixed.quadrupoles[spin_atom_begin * kGfn2SccPotentialQuadrupoleComponents + local];
  }
}

__global__ void gather_mixed_kernel(Gfn2SccPotentialDeviceBatch batch,
                                    Gfn2SccPotentialDeviceMixedFields mixed,
                                    Gfn2SccPotentialDeviceActivity activity,
                                    Gfn2SccPotentialDeviceWorkspace workspace,
                                    std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int active;
  __shared__ int valid;
  if (threadIdx.x == 0) {
    active = 0;
    valid = 1;
    if (sequence_is_active(workspace) && system_is_valid(system_errors, system)) {
      const std::uint8_t state =
          activity.active_mask == nullptr ? 1u : activity.active_mask[system];
      if (state > 1u) {
        record_system_error(system_errors, system, device_error,
                            Gfn2SccPotentialDeviceError::kInvalidActiveMask);
        valid = 0;
      } else {
        active = state == 1u ? 1 : 0;
      }
    }
  }
  __syncthreads();
  if (active == 0) {
    return;
  }

  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t qsh_begin = batch.qsh_offsets[system];
  const std::int64_t dipole_begin = batch.dipole_offsets[system];
  const std::int64_t quadrupole_begin = batch.quadrupole_offsets[system];

  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    const std::int64_t atom = batch.shell_to_atom[shell];
    if (atom < atom_begin || atom >= atom_end) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccPotentialDeviceError::kInvalidShellMetadata);
      atomicExch(&valid, 0);
      continue;
    }
    const double value = mixed.qsh[qsh_begin + shell - shell_begin];
    if (!isfinite(value)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccPotentialDeviceError::kNonfiniteMixedShellCharge);
      atomicExch(&valid, 0);
      continue;
    }
    workspace.shell_scratch[shell] = value;
  }
  /* Finish validation of every shell before any atom reduction rereads qsh.
   * This makes a nonfinite qsh value deterministically a mixed-shell error;
   * the reduction phase below is then responsible only for finite addition
   * overflow. */
  __syncthreads();
  if (valid == 0) {
    return;
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const std::int64_t local_atom = atom - atom_begin;
    double charge = 0.0;
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      if (batch.shell_to_atom[shell] == atom) {
        const double value = mixed.qsh[qsh_begin + shell - shell_begin];
        const double updated = charge + value;
        if (!isfinite(updated)) {
          record_system_error(system_errors, system, device_error,
                              Gfn2SccPotentialDeviceError::kNonfiniteAtomicChargeReduction);
          atomicExch(&valid, 0);
          break;
        }
        charge = updated;
      }
    }
    workspace.atom_scratch[atom] = charge;
    for (std::int64_t component = 0; component < kGfn2SccPotentialDipoleComponents; ++component) {
      const double value =
          mixed.dipoles[dipole_begin + local_atom * kGfn2SccPotentialDipoleComponents + component];
      if (!isfinite(value)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2SccPotentialDeviceError::kNonfiniteMixedDipole);
        atomicExch(&valid, 0);
      } else {
        workspace.dipole_scratch[atom * kGfn2SccPotentialDipoleComponents + component] = value;
      }
    }
    for (std::int64_t component = 0; component < kGfn2SccPotentialQuadrupoleComponents;
         ++component) {
      const double value =
          mixed.quadrupoles[quadrupole_begin + local_atom * kGfn2SccPotentialQuadrupoleComponents +
                            component];
      if (!isfinite(value)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2SccPotentialDeviceError::kNonfiniteMixedQuadrupole);
        atomicExch(&valid, 0);
      } else {
        workspace.quadrupole_scratch[atom * kGfn2SccPotentialQuadrupoleComponents + component] =
            value;
      }
    }
  }
  __syncthreads();
  (void)valid;
}

__global__ void publish_mixed_kernel(Gfn2SccPotentialDeviceBatch batch,
                                     Gfn2SccPotentialDeviceActivity activity,
                                     Gfn2SccPotentialDeviceTopologyMultipoles topology,
                                     Gfn2SccPotentialDeviceWorkspace workspace,
                                     const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!sequence_is_active(workspace) || !system_is_valid(system_errors, system) ||
      (activity.active_mask != nullptr && activity.active_mask[system] != 1u)) {
    return;
  }
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    topology.shell_charges[shell] = workspace.shell_scratch[shell];
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    topology.atomic_charges[atom] = workspace.atom_scratch[atom];
    for (std::int64_t component = 0; component < kGfn2SccPotentialDipoleComponents; ++component) {
      topology.atomic_dipoles[atom * kGfn2SccPotentialDipoleComponents + component] =
          workspace.dipole_scratch[atom * kGfn2SccPotentialDipoleComponents + component];
    }
    for (std::int64_t component = 0; component < kGfn2SccPotentialQuadrupoleComponents;
         ++component) {
      topology.atomic_quadrupoles[atom * kGfn2SccPotentialQuadrupoleComponents + component] =
          workspace.quadrupole_scratch[atom * kGfn2SccPotentialQuadrupoleComponents + component];
    }
  }
}

__device__ double load_component(const double* values, std::int64_t index, bool enabled,
                                 Gfn2SccPotentialDeviceError error, int* valid,
                                 std::uint32_t* system_errors, std::int64_t system,
                                 std::uint32_t* device_error) {
  if (!enabled) {
    return 0.0;
  }
  const double value = values[index];
  if (!isfinite(value)) {
    record_system_error(system_errors, system, device_error, error);
    atomicExch(valid, 0);
    return 0.0;
  }
  return value;
}

__device__ int requested_member(const Gfn2SccPotentialDeviceActivity& activity,
                                const Gfn2SccPotentialDeviceWorkspace& workspace,
                                std::uint32_t* system_errors, std::int64_t system,
                                std::uint32_t* device_error) {
  if (!sequence_is_active(workspace) || !system_is_valid(system_errors, system)) {
    return 0;
  }
  const std::uint8_t state = activity.active_mask == nullptr ? 1u : activity.active_mask[system];
  if (state > 1u) {
    if (device_error != nullptr) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccPotentialDeviceError::kInvalidActiveMask);
    }
    return 0;
  }
  return state == 1u ? 1 : 0;
}

__device__ int requested_member(const Gfn2SccIterationDeviceActivity& activity,
                                const Gfn2SccPotentialDeviceWorkspace& workspace,
                                std::uint32_t* system_errors, std::int64_t system,
                                std::uint32_t* device_error) {
  if (atomicAdd(const_cast<std::uint32_t*>(activity.sequence_active), 0u) != 1u ||
      !sequence_is_active(workspace) || !system_is_valid(system_errors, system)) {
    return 0;
  }
  const std::uint8_t state = activity.active_mask[system];
  if (state > 1u) {
    if (device_error != nullptr) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccPotentialDeviceError::kInvalidActiveMask);
    }
    return 0;
  }
  return state == 1u ? 1 : 0;
}

template <typename Activity>
__global__ void compose_potential_kernel(Gfn2SccPotentialDeviceBatch batch,
                                         Gfn2SccPotentialDeviceComponents components,
                                         Activity activity,
                                         Gfn2SccPotentialDeviceWorkspace workspace,
                                         std::uint32_t* system_errors,
                                         std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int active;
  __shared__ int valid;
  if (threadIdx.x == 0) {
    active = 0;
    valid = 1;
    active = requested_member(activity, workspace, system_errors, system, device_error);
    valid = active;
  }
  __syncthreads();
  if (active == 0) {
    return;
  }

  const bool es2_enabled =
      component_enabled(components.enabled_components, Gfn2SccPotentialComponent::kES2);
  const bool es3_enabled =
      component_enabled(components.enabled_components, Gfn2SccPotentialComponent::kES3);
  const bool aes2_enabled =
      component_enabled(components.enabled_components, Gfn2SccPotentialComponent::kAES2);
  const bool d4_enabled =
      component_enabled(components.enabled_components, Gfn2SccPotentialComponent::kD4TwoBody);
  const bool pc_enabled = component_enabled(components.enabled_components,
                                            Gfn2SccPotentialComponent::kExplicitPointCharge);
  const bool periodic_enabled = component_enabled(components.enabled_components,
                                                  Gfn2SccPotentialComponent::kPeriodicEmbedding);
  const bool field_enabled = components.electric_field_atomic != nullptr;
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t qsh_begin = batch.qsh_offsets[system];
  const std::int64_t qat_begin = batch.qat_offsets[system];
  const std::int64_t dipole_begin = batch.dipole_offsets[system];
  const std::int64_t quadrupole_begin = batch.quadrupole_offsets[system];

  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    const double es2 = load_component(components.es2_shell, shell, es2_enabled,
                                      Gfn2SccPotentialDeviceError::kNonfiniteES2Potential, &valid,
                                      system_errors, system, device_error);
    const double es3 = load_component(components.es3_shell, shell, es3_enabled,
                                      Gfn2SccPotentialDeviceError::kNonfiniteES3Potential, &valid,
                                      system_errors, system, device_error);
    const double pc =
        load_component(components.explicit_point_charge_shell, shell, pc_enabled,
                       Gfn2SccPotentialDeviceError::kNonfiniteExplicitPointChargePotential, &valid,
                       system_errors, system, device_error);
    const double first = es2 + es3;
    const double total = first + pc;
    if (!isfinite(first) || !isfinite(total)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccPotentialDeviceError::kNonfiniteShellPotentialArithmetic);
      atomicExch(&valid, 0);
    } else {
      workspace.shell_scratch[qsh_begin + shell - shell_begin] = total;
    }
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const std::int64_t local_atom = atom - atom_begin;
    const double aes2_atomic = load_component(components.aes2_atomic, atom, aes2_enabled,
                                              Gfn2SccPotentialDeviceError::kNonfiniteAES2Potential,
                                              &valid, system_errors, system, device_error);
    const double periodic = load_component(components.periodic_atomic, atom, periodic_enabled,
                                           Gfn2SccPotentialDeviceError::kNonfinitePeriodicPotential,
                                           &valid, system_errors, system, device_error);
    const double d4 = load_component(components.d4_atomic, atom, d4_enabled,
                                     Gfn2SccPotentialDeviceError::kNonfiniteD4Potential, &valid,
                                     system_errors, system, device_error);
    const double field =
        load_component(components.electric_field_atomic, atom, field_enabled,
                       Gfn2SccPotentialDeviceError::kNonfiniteElectricFieldPotential, &valid,
                       system_errors, system, device_error);
    const double first = aes2_atomic + periodic;
    const double second = first + d4;
    const double total = second + field;
    if (!isfinite(first) || !isfinite(second) || !isfinite(total)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccPotentialDeviceError::kNonfiniteAtomicPotentialArithmetic);
      atomicExch(&valid, 0);
    } else {
      workspace.atom_scratch[qat_begin + local_atom] = total;
    }
    for (std::int64_t component = 0; component < kGfn2SccPotentialDipoleComponents; ++component) {
      const std::int64_t index = atom * kGfn2SccPotentialDipoleComponents + component;
      const double aes2 = load_component(components.aes2_dipole, index, aes2_enabled,
                                         Gfn2SccPotentialDeviceError::kNonfiniteAES2Potential,
                                         &valid, system_errors, system, device_error);
      const double field =
          load_component(components.electric_field_dipole, index, field_enabled,
                         Gfn2SccPotentialDeviceError::kNonfiniteElectricFieldPotential, &valid,
                         system_errors, system, device_error);
      const double total = aes2 + field;
      if (!isfinite(total)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2SccPotentialDeviceError::kNonfiniteAtomicPotentialArithmetic);
        atomicExch(&valid, 0);
      } else {
        workspace.dipole_scratch[dipole_begin + local_atom * kGfn2SccPotentialDipoleComponents +
                                 component] = total;
      }
    }
    for (std::int64_t component = 0; component < kGfn2SccPotentialQuadrupoleComponents;
         ++component) {
      workspace.quadrupole_scratch[quadrupole_begin +
                                   local_atom * kGfn2SccPotentialQuadrupoleComponents + component] =
          load_component(components.aes2_quadrupole,
                         atom * kGfn2SccPotentialQuadrupoleComponents + component, aes2_enabled,
                         Gfn2SccPotentialDeviceError::kNonfiniteAES2Potential, &valid,
                         system_errors, system, device_error);
    }
  }
  __syncthreads();
  (void)valid;
}

__global__ void compose_spin_potential_kernel(
    Gfn2SccPotentialDeviceBatch batch, Gfn2WavefunctionLayoutView layout,
    Gfn2SccPotentialDeviceComponents components, Gfn2SccPotentialDeviceSpinComponent spin,
    Gfn2SccPotentialDeviceActivity activity, Gfn2SccPotentialDeviceWorkspace workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int active;
  __shared__ int valid;
  if (threadIdx.x == 0) {
    active = requested_member(activity, workspace, system_errors, system, device_error);
    valid = active;
  }
  __syncthreads();
  if (active == 0) {
    return;
  }

  const bool es2_enabled =
      component_enabled(components.enabled_components, Gfn2SccPotentialComponent::kES2);
  const bool es3_enabled =
      component_enabled(components.enabled_components, Gfn2SccPotentialComponent::kES3);
  const bool aes2_enabled =
      component_enabled(components.enabled_components, Gfn2SccPotentialComponent::kAES2);
  const bool d4_enabled =
      component_enabled(components.enabled_components, Gfn2SccPotentialComponent::kD4TwoBody);
  const bool pc_enabled = component_enabled(components.enabled_components,
                                            Gfn2SccPotentialComponent::kExplicitPointCharge);
  const bool periodic_enabled = component_enabled(components.enabled_components,
                                                  Gfn2SccPotentialComponent::kPeriodicEmbedding);
  const bool field_enabled = components.electric_field_atomic != nullptr;
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t atoms = atom_end - atom_begin;
  const std::int64_t shells = shell_end - shell_begin;
  const std::int64_t spin_shell_begin = layout.spin_shell_offsets[system];
  const std::int64_t spin_atom_begin = layout.spin_atom_offsets[system];
  const std::int32_t channels = layout.spin_channels[system];

  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    const std::int64_t local_shell = shell - shell_begin;
    const std::int64_t atom = batch.shell_to_atom[shell];
    const double es2 = load_component(components.es2_shell, shell, es2_enabled,
                                      Gfn2SccPotentialDeviceError::kNonfiniteES2Potential, &valid,
                                      system_errors, system, device_error);
    const double es3 = load_component(components.es3_shell, shell, es3_enabled,
                                      Gfn2SccPotentialDeviceError::kNonfiniteES3Potential, &valid,
                                      system_errors, system, device_error);
    const double pc =
        load_component(components.explicit_point_charge_shell, shell, pc_enabled,
                       Gfn2SccPotentialDeviceError::kNonfiniteExplicitPointChargePotential, &valid,
                       system_errors, system, device_error);
    const double first = es2 + es3;
    const double shell_total = first + pc;
    if (!isfinite(first) || !isfinite(shell_total)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccPotentialDeviceError::kNonfiniteShellPotentialArithmetic);
      atomicExch(&valid, 0);
    }
    const double aes2_atomic = load_component(components.aes2_atomic, atom, aes2_enabled,
                                              Gfn2SccPotentialDeviceError::kNonfiniteAES2Potential,
                                              &valid, system_errors, system, device_error);
    const double periodic = load_component(components.periodic_atomic, atom, periodic_enabled,
                                           Gfn2SccPotentialDeviceError::kNonfinitePeriodicPotential,
                                           &valid, system_errors, system, device_error);
    const double d4 = load_component(components.d4_atomic, atom, d4_enabled,
                                     Gfn2SccPotentialDeviceError::kNonfiniteD4Potential, &valid,
                                     system_errors, system, device_error);
    const double field =
        load_component(components.electric_field_atomic, atom, field_enabled,
                       Gfn2SccPotentialDeviceError::kNonfiniteElectricFieldPotential, &valid,
                       system_errors, system, device_error);
    const double atomic_first = aes2_atomic + periodic;
    const double atomic_second = atomic_first + d4;
    const double atomic_total = atomic_second + field;
    if (!isfinite(atomic_first) || !isfinite(atomic_second) || !isfinite(atomic_total)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccPotentialDeviceError::kNonfiniteAtomicPotentialArithmetic);
      atomicExch(&valid, 0);
    }
    /* Match the legacy scalar bridge literally: first collect the raw shell
     * and atomic associations independently, then add vat to vsh once. */
    const double complete = shell_total + atomic_total;
    if (!isfinite(complete)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccPotentialDeviceError::kNonfiniteShellPotentialArithmetic);
      atomicExch(&valid, 0);
    } else {
      workspace.shell_scratch[spin_shell_begin + local_shell] = complete;
    }
    if (channels == 2) {
      const std::int64_t magnetization_index = spin_shell_begin + shells + local_shell;
      const double magnetization = spin.shell[magnetization_index];
      if (!isfinite(magnetization)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2SccPotentialDeviceError::kNonfiniteSpinPotential);
        atomicExch(&valid, 0);
      } else {
        workspace.shell_scratch[magnetization_index] = magnetization;
      }
    }
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const std::int64_t local_atom = atom - atom_begin;
    const std::int64_t charge_atom = spin_atom_begin + local_atom;
    const double aes2_atomic = load_component(components.aes2_atomic, atom, aes2_enabled,
                                              Gfn2SccPotentialDeviceError::kNonfiniteAES2Potential,
                                              &valid, system_errors, system, device_error);
    const double periodic = load_component(components.periodic_atomic, atom, periodic_enabled,
                                           Gfn2SccPotentialDeviceError::kNonfinitePeriodicPotential,
                                           &valid, system_errors, system, device_error);
    const double d4 = load_component(components.d4_atomic, atom, d4_enabled,
                                     Gfn2SccPotentialDeviceError::kNonfiniteD4Potential, &valid,
                                     system_errors, system, device_error);
    const double field =
        load_component(components.electric_field_atomic, atom, field_enabled,
                       Gfn2SccPotentialDeviceError::kNonfiniteElectricFieldPotential, &valid,
                       system_errors, system, device_error);
    const double first = aes2_atomic + periodic;
    const double second = first + d4;
    const double total = second + field;
    if (!isfinite(first) || !isfinite(second) || !isfinite(total)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccPotentialDeviceError::kNonfiniteAtomicPotentialArithmetic);
      atomicExch(&valid, 0);
    } else {
      workspace.atom_scratch[charge_atom] = total;
    }
    for (std::int64_t component = 0; component < kGfn2SccPotentialDipoleComponents; ++component) {
      const std::int64_t source = atom * kGfn2SccPotentialDipoleComponents + component;
      const double aes2 = load_component(components.aes2_dipole, source, aes2_enabled,
                                         Gfn2SccPotentialDeviceError::kNonfiniteAES2Potential,
                                         &valid, system_errors, system, device_error);
      const double field =
          load_component(components.electric_field_dipole, source, field_enabled,
                         Gfn2SccPotentialDeviceError::kNonfiniteElectricFieldPotential, &valid,
                         system_errors, system, device_error);
      const double total = aes2 + field;
      if (!isfinite(total)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2SccPotentialDeviceError::kNonfiniteAtomicPotentialArithmetic);
        atomicExch(&valid, 0);
      } else {
        workspace.dipole_scratch[charge_atom * kGfn2SccPotentialDipoleComponents + component] =
            total;
      }
    }
    for (std::int64_t component = 0; component < kGfn2SccPotentialQuadrupoleComponents;
         ++component) {
      workspace
          .quadrupole_scratch[charge_atom * kGfn2SccPotentialQuadrupoleComponents + component] =
          load_component(components.aes2_quadrupole,
                         atom * kGfn2SccPotentialQuadrupoleComponents + component, aes2_enabled,
                         Gfn2SccPotentialDeviceError::kNonfiniteAES2Potential, &valid,
                         system_errors, system, device_error);
    }
    if (channels == 2) {
      const std::int64_t magnetization_atom = spin_atom_begin + atoms + local_atom;
      workspace.atom_scratch[magnetization_atom] = 0.0;
      for (std::int64_t component = 0; component < kGfn2SccPotentialDipoleComponents; ++component) {
        workspace
            .dipole_scratch[magnetization_atom * kGfn2SccPotentialDipoleComponents + component] =
            0.0;
      }
      for (std::int64_t component = 0; component < kGfn2SccPotentialQuadrupoleComponents;
           ++component) {
        workspace.quadrupole_scratch[magnetization_atom * kGfn2SccPotentialQuadrupoleComponents +
                                     component] = 0.0;
      }
    }
  }
  __syncthreads();
  (void)valid;
}

template <typename Activity>
__global__ void publish_potential_kernel(Gfn2SccPotentialDeviceBatch batch, Activity activity,
                                         Gfn2SccPotentialDeviceResults results,
                                         Gfn2SccPotentialDeviceWorkspace workspace,
                                         const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (requested_member(activity, workspace, const_cast<std::uint32_t*>(system_errors), system,
                       nullptr) == 0) {
    return;
  }
  const std::int64_t shell_begin = batch.qsh_offsets[system];
  const std::int64_t shell_end = batch.qsh_offsets[system + 1];
  const std::int64_t atom_begin = batch.qat_offsets[system];
  const std::int64_t atom_end = batch.qat_offsets[system + 1];
  const std::int64_t dipole_begin = batch.dipole_offsets[system];
  const std::int64_t dipole_end = batch.dipole_offsets[system + 1];
  const std::int64_t quadrupole_begin = batch.quadrupole_offsets[system];
  const std::int64_t quadrupole_end = batch.quadrupole_offsets[system + 1];
  for (std::int64_t element = shell_begin + threadIdx.x; element < shell_end;
       element += blockDim.x) {
    results.shell[element] = workspace.shell_scratch[element];
  }
  for (std::int64_t element = atom_begin + threadIdx.x; element < atom_end; element += blockDim.x) {
    results.atomic[element] = workspace.atom_scratch[element];
  }
  for (std::int64_t element = dipole_begin + threadIdx.x; element < dipole_end;
       element += blockDim.x) {
    results.dipole[element] = workspace.dipole_scratch[element];
  }
  for (std::int64_t element = quadrupole_begin + threadIdx.x; element < quadrupole_end;
       element += blockDim.x) {
    results.quadrupole[element] = workspace.quadrupole_scratch[element];
  }
}

__global__ void publish_spin_potential_kernel(Gfn2SccPotentialDeviceBatch batch,
                                              Gfn2WavefunctionLayoutView layout,
                                              Gfn2SccPotentialDeviceActivity activity,
                                              Gfn2SccPotentialDeviceResults results,
                                              Gfn2SccPotentialDeviceWorkspace workspace,
                                              const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (requested_member(activity, workspace, const_cast<std::uint32_t*>(system_errors), system,
                       nullptr) == 0) {
    return;
  }
  const std::int64_t shell_begin = layout.spin_shell_offsets[system];
  const std::int64_t shell_end = layout.spin_shell_offsets[system + 1];
  const std::int64_t atom_begin = layout.spin_atom_offsets[system];
  const std::int64_t atom_end = layout.spin_atom_offsets[system + 1];
  for (std::int64_t element = shell_begin + threadIdx.x; element < shell_end;
       element += blockDim.x) {
    results.shell[element] = workspace.shell_scratch[element];
  }
  for (std::int64_t element = atom_begin + threadIdx.x; element < atom_end; element += blockDim.x) {
    results.atomic[element] = workspace.atom_scratch[element];
  }
  for (std::int64_t element = atom_begin * kGfn2SccPotentialDipoleComponents + threadIdx.x;
       element < atom_end * kGfn2SccPotentialDipoleComponents; element += blockDim.x) {
    results.dipole[element] = workspace.dipole_scratch[element];
  }
  for (std::int64_t element = atom_begin * kGfn2SccPotentialQuadrupoleComponents + threadIdx.x;
       element < atom_end * kGfn2SccPotentialQuadrupoleComponents; element += blockDim.x) {
    results.quadrupole[element] = workspace.quadrupole_scratch[element];
  }
}

bool is_aligned(const void* pointer, std::size_t alignment) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool make_range(const void* pointer, std::int64_t elements, std::size_t element_size,
                AddressRange* range) noexcept {
  if (elements < 0 || element_size == 0u ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(elements) * element_size;
  if (bytes == 0u) {
    *range = {};
    return pointer == nullptr;
  }
  if (pointer == nullptr) {
    return false;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  *range = {begin, begin + bytes};
  return true;
}

bool ranges_overlap(const AddressRange& lhs, const AddressRange& rhs) noexcept {
  return lhs.begin != lhs.end && rhs.begin != rhs.end && lhs.begin < rhs.end && rhs.begin < lhs.end;
}

bool checked_multiply(std::int64_t value, std::int64_t factor, std::int64_t* result) noexcept {
  if (value < 0 || factor < 0 ||
      (value != 0 && factor > std::numeric_limits<std::int64_t>::max() / value)) {
    return false;
  }
  *result = value * factor;
  return true;
}

bool validate_common(const Gfn2SccPotentialDeviceBatch& batch,
                     const Gfn2SccPotentialDeviceActivity& activity,
                     const Gfn2SccPotentialDeviceWorkspace& workspace, std::uint32_t* system_errors,
                     std::uint32_t* device_error) noexcept {
  std::int64_t dipole_elements = 0;
  std::int64_t quadrupole_elements = 0;
  return batch.batch_size > 0 && batch.batch_size <= std::numeric_limits<int>::max() &&
         batch.total_atoms >= 0 && batch.total_shells >= 0 && batch.plan_token != 0u &&
         checked_multiply(batch.total_atoms, kGfn2SccPotentialDipoleComponents, &dipole_elements) &&
         checked_multiply(batch.total_atoms, kGfn2SccPotentialQuadrupoleComponents,
                          &quadrupole_elements) &&
         batch.atom_offset_count == batch.batch_size + 1 &&
         batch.batch_shell_offset_count == batch.batch_size + 1 &&
         batch.qsh_offset_count == batch.batch_size + 1 &&
         batch.qat_offset_count == batch.batch_size + 1 &&
         batch.dipole_offset_count == batch.batch_size + 1 &&
         batch.quadrupole_offset_count == batch.batch_size + 1 &&
         batch.shell_to_atom_count == batch.total_shells &&
         is_aligned(batch.atom_offsets, alignof(std::int64_t)) &&
         is_aligned(batch.batch_shell_offsets, alignof(std::int64_t)) &&
         is_aligned(batch.qsh_offsets, alignof(std::int64_t)) &&
         is_aligned(batch.qat_offsets, alignof(std::int64_t)) &&
         is_aligned(batch.dipole_offsets, alignof(std::int64_t)) &&
         is_aligned(batch.quadrupole_offsets, alignof(std::int64_t)) &&
         (batch.total_shells == 0 || is_aligned(batch.shell_to_atom, alignof(std::int64_t))) &&
         activity.plan_token == batch.plan_token &&
         ((activity.active_mask == nullptr && activity.elements == 0) ||
          (activity.elements == batch.batch_size && activity.active_mask != nullptr)) &&
         workspace.plan_token == batch.plan_token &&
         workspace.shell_elements == batch.total_shells &&
         workspace.atom_elements == batch.total_atoms &&
         workspace.dipole_elements == dipole_elements &&
         workspace.quadrupole_elements == quadrupole_elements && workspace.sequence_elements == 1 &&
         (batch.total_shells == 0 || is_aligned(workspace.shell_scratch, alignof(double))) &&
         (batch.total_atoms == 0 || is_aligned(workspace.atom_scratch, alignof(double))) &&
         (dipole_elements == 0 || is_aligned(workspace.dipole_scratch, alignof(double))) &&
         (quadrupole_elements == 0 || is_aligned(workspace.quadrupole_scratch, alignof(double))) &&
         is_aligned(workspace.sequence_active, alignof(std::uint32_t)) &&
         is_aligned(system_errors, alignof(std::uint32_t)) &&
         is_aligned(device_error, alignof(std::uint32_t));
}

bool validate_canonical_activity(const Gfn2SccPotentialDeviceBatch& batch,
                                 const Gfn2SccIterationDeviceActivity& activity) noexcept {
  return activity.plan_token == batch.plan_token && activity.active_mask != nullptr &&
         activity.sequence_active != nullptr && activity.batch_elements == batch.batch_size &&
         activity.sequence_elements == 1 &&
         is_aligned(activity.active_mask, alignof(std::uint8_t)) &&
         is_aligned(activity.sequence_active, alignof(std::uint32_t));
}

template <std::size_t ReadCount, std::size_t WriteCount>
bool disjoint_bindings(const std::array<AddressRange, ReadCount>& reads,
                       const std::array<AddressRange, WriteCount>& writes) noexcept {
  for (std::size_t lhs = 0; lhs < writes.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < writes.size(); ++rhs) {
      if (ranges_overlap(writes[lhs], writes[rhs])) {
        return false;
      }
    }
    for (const AddressRange& read : reads) {
      if (ranges_overlap(writes[lhs], read)) {
        return false;
      }
    }
  }
  return true;
}

bool valid_component(const double* pointer, std::int64_t elements, std::int64_t required,
                     bool enabled) noexcept {
  return enabled ? elements == required && (required == 0 || is_aligned(pointer, alignof(double)))
                 : pointer == nullptr && elements == 0;
}

template <std::size_t Count>
bool pairwise_disjoint_bindings(const std::array<AddressRange, Count>& ranges) noexcept {
  for (std::size_t lhs = 0; lhs < ranges.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < ranges.size(); ++rhs) {
      if (ranges_overlap(ranges[lhs], ranges[rhs])) {
        return false;
      }
    }
  }
  return true;
}

bool validate_gather(const Gfn2SccPotentialDeviceBatch& batch,
                     const Gfn2SccPotentialDeviceMixedFields& mixed,
                     const Gfn2SccPotentialDeviceActivity& activity,
                     const Gfn2SccPotentialDeviceTopologyMultipoles& topology,
                     const Gfn2SccPotentialDeviceWorkspace& workspace, std::uint32_t* system_errors,
                     std::uint32_t* device_error) noexcept {
  std::int64_t dipole_elements = 0;
  std::int64_t quadrupole_elements = 0;
  if (!validate_common(batch, activity, workspace, system_errors, device_error) ||
      !checked_multiply(batch.total_atoms, kGfn2SccPotentialDipoleComponents, &dipole_elements) ||
      !checked_multiply(batch.total_atoms, kGfn2SccPotentialQuadrupoleComponents,
                        &quadrupole_elements) ||
      mixed.plan_token != batch.plan_token || mixed.qsh_elements != batch.total_shells ||
      mixed.dipole_elements != dipole_elements ||
      mixed.quadrupole_elements != quadrupole_elements || topology.plan_token != batch.plan_token ||
      topology.shell_elements != batch.total_shells ||
      topology.atom_elements != batch.total_atoms || topology.dipole_elements != dipole_elements ||
      topology.quadrupole_elements != quadrupole_elements ||
      (batch.total_shells != 0 && !is_aligned(mixed.qsh, alignof(double))) ||
      (dipole_elements != 0 && !is_aligned(mixed.dipoles, alignof(double))) ||
      (quadrupole_elements != 0 && !is_aligned(mixed.quadrupoles, alignof(double))) ||
      (batch.total_shells != 0 && !is_aligned(topology.shell_charges, alignof(double))) ||
      (batch.total_atoms != 0 && !is_aligned(topology.atomic_charges, alignof(double))) ||
      (dipole_elements != 0 && !is_aligned(topology.atomic_dipoles, alignof(double))) ||
      (quadrupole_elements != 0 && !is_aligned(topology.atomic_quadrupoles, alignof(double)))) {
    return false;
  }
  std::array<AddressRange, 11> reads{};
  std::array<AddressRange, 11> writes{};
  if (!make_range(batch.atom_offsets, batch.atom_offset_count, sizeof(std::int64_t), &reads[0]) ||
      !make_range(batch.batch_shell_offsets, batch.batch_shell_offset_count, sizeof(std::int64_t),
                  &reads[1]) ||
      !make_range(batch.qsh_offsets, batch.qsh_offset_count, sizeof(std::int64_t), &reads[2]) ||
      !make_range(batch.dipole_offsets, batch.dipole_offset_count, sizeof(std::int64_t),
                  &reads[3]) ||
      !make_range(batch.quadrupole_offsets, batch.quadrupole_offset_count, sizeof(std::int64_t),
                  &reads[4]) ||
      !make_range(batch.shell_to_atom, batch.shell_to_atom_count, sizeof(std::int64_t),
                  &reads[5]) ||
      !make_range(activity.active_mask, activity.elements, sizeof(std::uint8_t), &reads[6]) ||
      !make_range(mixed.qsh, mixed.qsh_elements, sizeof(double), &reads[7]) ||
      !make_range(mixed.dipoles, mixed.dipole_elements, sizeof(double), &reads[8]) ||
      !make_range(mixed.quadrupoles, mixed.quadrupole_elements, sizeof(double), &reads[9]) ||
      !make_range(batch.qat_offsets, batch.qat_offset_count, sizeof(std::int64_t), &reads[10]) ||
      !make_range(topology.shell_charges, topology.shell_elements, sizeof(double), &writes[0]) ||
      !make_range(topology.atomic_charges, topology.atom_elements, sizeof(double), &writes[1]) ||
      !make_range(topology.atomic_dipoles, topology.dipole_elements, sizeof(double), &writes[2]) ||
      !make_range(topology.atomic_quadrupoles, topology.quadrupole_elements, sizeof(double),
                  &writes[3]) ||
      !make_range(workspace.shell_scratch, workspace.shell_elements, sizeof(double), &writes[4]) ||
      !make_range(workspace.atom_scratch, workspace.atom_elements, sizeof(double), &writes[5]) ||
      !make_range(workspace.dipole_scratch, workspace.dipole_elements, sizeof(double),
                  &writes[6]) ||
      !make_range(workspace.quadrupole_scratch, workspace.quadrupole_elements, sizeof(double),
                  &writes[7]) ||
      !make_range(workspace.sequence_active, 1, sizeof(std::uint32_t), &writes[8]) ||
      !make_range(system_errors, batch.batch_size, sizeof(std::uint32_t), &writes[9]) ||
      !make_range(device_error, 1, sizeof(std::uint32_t), &writes[10])) {
    return false;
  }
  return disjoint_bindings(reads, writes);
}

bool validate_canonical_reduction(const Gfn2SccPotentialDeviceBatch& batch,
                                  const Gfn2SccPotentialDeviceMixedFields& mixed,
                                  const Gfn2SccIterationDeviceActivity& activity,
                                  const Gfn2SccPotentialDeviceTopologyMultipoles& topology,
                                  const Gfn2SccPotentialDeviceWorkspace& workspace,
                                  std::uint32_t* system_errors,
                                  std::uint32_t* device_error) noexcept {
  const Gfn2SccPotentialDeviceActivity compatibility_activity{
      activity.active_mask, activity.batch_elements, activity.plan_token};
  std::int64_t dipole_elements = 0;
  std::int64_t quadrupole_elements = 0;
  if (!validate_canonical_activity(batch, activity) ||
      !validate_common(batch, compatibility_activity, workspace, system_errors, device_error) ||
      !checked_multiply(batch.total_atoms, kGfn2SccPotentialDipoleComponents, &dipole_elements) ||
      !checked_multiply(batch.total_atoms, kGfn2SccPotentialQuadrupoleComponents,
                        &quadrupole_elements) ||
      mixed.plan_token != batch.plan_token || mixed.qsh_elements != batch.total_shells ||
      mixed.dipole_elements != dipole_elements ||
      mixed.quadrupole_elements != quadrupole_elements || topology.plan_token != batch.plan_token ||
      topology.shell_elements != batch.total_shells ||
      topology.atom_elements != batch.total_atoms || topology.dipole_elements != dipole_elements ||
      topology.quadrupole_elements != quadrupole_elements || topology.shell_charges != mixed.qsh ||
      topology.atomic_dipoles != mixed.dipoles ||
      topology.atomic_quadrupoles != mixed.quadrupoles ||
      (batch.total_shells != 0 && !is_aligned(mixed.qsh, alignof(double))) ||
      (dipole_elements != 0 && !is_aligned(mixed.dipoles, alignof(double))) ||
      (quadrupole_elements != 0 && !is_aligned(mixed.quadrupoles, alignof(double))) ||
      (batch.total_atoms != 0 && !is_aligned(topology.atomic_charges, alignof(double)))) {
    return false;
  }

  std::array<AddressRange, 10> reads{};
  std::array<AddressRange, 5> writes{};
  if (!make_range(batch.atom_offsets, batch.atom_offset_count, sizeof(std::int64_t), &reads[0]) ||
      !make_range(batch.batch_shell_offsets, batch.batch_shell_offset_count, sizeof(std::int64_t),
                  &reads[1]) ||
      !make_range(batch.qsh_offsets, batch.qsh_offset_count, sizeof(std::int64_t), &reads[2]) ||
      !make_range(batch.shell_to_atom, batch.shell_to_atom_count, sizeof(std::int64_t),
                  &reads[3]) ||
      !make_range(activity.active_mask, activity.batch_elements, sizeof(std::uint8_t), &reads[4]) ||
      !make_range(activity.sequence_active, 1, sizeof(std::uint32_t), &reads[5]) ||
      !make_range(mixed.qsh, mixed.qsh_elements, sizeof(double), &reads[6]) ||
      !make_range(mixed.dipoles, mixed.dipole_elements, sizeof(double), &reads[7]) ||
      !make_range(mixed.quadrupoles, mixed.quadrupole_elements, sizeof(double), &reads[8]) ||
      !make_range(batch.qat_offsets, batch.qat_offset_count, sizeof(std::int64_t), &reads[9]) ||
      !make_range(topology.atomic_charges, topology.atom_elements, sizeof(double), &writes[0]) ||
      !make_range(workspace.atom_scratch, workspace.atom_elements, sizeof(double), &writes[1]) ||
      !make_range(workspace.sequence_active, 1, sizeof(std::uint32_t), &writes[2]) ||
      !make_range(system_errors, batch.batch_size, sizeof(std::uint32_t), &writes[3]) ||
      !make_range(device_error, 1, sizeof(std::uint32_t), &writes[4])) {
    return false;
  }
  return disjoint_bindings(reads, writes);
}

bool validate_spin_reduction(const Gfn2SccPotentialDeviceBatch& batch,
                             const Gfn2WavefunctionLayoutView& layout,
                             const Gfn2SccPotentialDeviceMixedFields& mixed,
                             const Gfn2SccIterationDeviceActivity& activity,
                             const Gfn2SccPotentialDeviceTopologyMultipoles& spin_topology,
                             const Gfn2SccPotentialDeviceTopologyMultipoles& physical_topology,
                             const Gfn2SccPotentialDeviceWorkspace& workspace,
                             std::uint32_t* system_errors, std::uint32_t* device_error) noexcept {
  std::int64_t spin_dipoles = 0;
  std::int64_t spin_quadrupoles = 0;
  std::int64_t physical_dipoles = 0;
  std::int64_t physical_quadrupoles = 0;
  if (batch.batch_size <= 0 || batch.batch_size > std::numeric_limits<int>::max() ||
      batch.total_atoms <= 0 || batch.total_shells <= 0 || batch.plan_token == 0u ||
      batch.atom_offset_count != batch.batch_size + 1 ||
      batch.batch_shell_offset_count != batch.batch_size + 1 ||
      batch.shell_to_atom_count != batch.total_shells ||
      !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.batch_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_to_atom, alignof(std::int64_t)) ||
      layout.memory_space != Gfn2PlanMemorySpace::kCudaDevice ||
      layout.plan_token != batch.plan_token || layout.batch_size != batch.batch_size ||
      layout.total_spin_channels < batch.batch_size ||
      layout.total_spin_channels - batch.batch_size > batch.batch_size ||
      layout.total_spin_shells < batch.total_shells ||
      layout.total_spin_shells - batch.total_shells > batch.total_shells ||
      layout.total_spin_atoms < batch.total_atoms ||
      layout.total_spin_atoms - batch.total_atoms > batch.total_atoms ||
      layout.spin_channel_count != batch.batch_size ||
      layout.spin_channel_offset_count != batch.batch_size + 1 ||
      layout.spin_shell_offset_count != batch.batch_size + 1 ||
      layout.spin_atom_offset_count != batch.batch_size + 1 ||
      !is_aligned(layout.spin_channels, alignof(std::int32_t)) ||
      !is_aligned(layout.spin_channel_offsets, alignof(std::int64_t)) ||
      !is_aligned(layout.spin_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(layout.spin_atom_offsets, alignof(std::int64_t)) ||
      !validate_canonical_activity(batch, activity) ||
      !checked_multiply(layout.total_spin_atoms, kGfn2SccPotentialDipoleComponents,
                        &spin_dipoles) ||
      !checked_multiply(layout.total_spin_atoms, kGfn2SccPotentialQuadrupoleComponents,
                        &spin_quadrupoles) ||
      !checked_multiply(batch.total_atoms, kGfn2SccPotentialDipoleComponents, &physical_dipoles) ||
      !checked_multiply(batch.total_atoms, kGfn2SccPotentialQuadrupoleComponents,
                        &physical_quadrupoles) ||
      mixed.plan_token != batch.plan_token || mixed.qsh_elements != layout.total_spin_shells ||
      mixed.dipole_elements != spin_dipoles || mixed.quadrupole_elements != spin_quadrupoles ||
      !is_aligned(mixed.qsh, alignof(double)) || !is_aligned(mixed.dipoles, alignof(double)) ||
      !is_aligned(mixed.quadrupoles, alignof(double)) ||
      spin_topology.plan_token != batch.plan_token ||
      spin_topology.shell_elements != layout.total_spin_shells ||
      spin_topology.atom_elements != layout.total_spin_atoms ||
      spin_topology.dipole_elements != spin_dipoles ||
      spin_topology.quadrupole_elements != spin_quadrupoles ||
      spin_topology.shell_charges != mixed.qsh || spin_topology.atomic_dipoles != mixed.dipoles ||
      spin_topology.atomic_quadrupoles != mixed.quadrupoles ||
      !is_aligned(spin_topology.atomic_charges, alignof(double)) ||
      physical_topology.plan_token != batch.plan_token ||
      physical_topology.shell_elements != batch.total_shells ||
      physical_topology.atom_elements != batch.total_atoms ||
      physical_topology.dipole_elements != physical_dipoles ||
      physical_topology.quadrupole_elements != physical_quadrupoles ||
      !is_aligned(physical_topology.shell_charges, alignof(double)) ||
      !is_aligned(physical_topology.atomic_charges, alignof(double)) ||
      !is_aligned(physical_topology.atomic_dipoles, alignof(double)) ||
      !is_aligned(physical_topology.atomic_quadrupoles, alignof(double)) ||
      workspace.plan_token != batch.plan_token ||
      workspace.shell_elements != layout.total_spin_shells ||
      workspace.atom_elements != layout.total_spin_atoms ||
      workspace.dipole_elements != spin_dipoles ||
      workspace.quadrupole_elements != spin_quadrupoles || workspace.sequence_elements != 1 ||
      !is_aligned(workspace.atom_scratch, alignof(double)) ||
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return false;
  }

  std::array<AddressRange, 12> reads{};
  std::array<AddressRange, 9> writes{};
  if (!make_range(batch.atom_offsets, batch.atom_offset_count, sizeof(std::int64_t), &reads[0]) ||
      !make_range(batch.batch_shell_offsets, batch.batch_shell_offset_count, sizeof(std::int64_t),
                  &reads[1]) ||
      !make_range(batch.shell_to_atom, batch.shell_to_atom_count, sizeof(std::int64_t),
                  &reads[2]) ||
      !make_range(layout.spin_channels, layout.spin_channel_count, sizeof(std::int32_t),
                  &reads[3]) ||
      !make_range(layout.spin_channel_offsets, layout.spin_channel_offset_count,
                  sizeof(std::int64_t), &reads[4]) ||
      !make_range(layout.spin_shell_offsets, layout.spin_shell_offset_count, sizeof(std::int64_t),
                  &reads[5]) ||
      !make_range(layout.spin_atom_offsets, layout.spin_atom_offset_count, sizeof(std::int64_t),
                  &reads[6]) ||
      !make_range(activity.active_mask, activity.batch_elements, sizeof(std::uint8_t), &reads[7]) ||
      !make_range(activity.sequence_active, activity.sequence_elements, sizeof(std::uint32_t),
                  &reads[8]) ||
      !make_range(mixed.qsh, mixed.qsh_elements, sizeof(double), &reads[9]) ||
      !make_range(mixed.dipoles, mixed.dipole_elements, sizeof(double), &reads[10]) ||
      !make_range(mixed.quadrupoles, mixed.quadrupole_elements, sizeof(double), &reads[11]) ||
      !make_range(spin_topology.atomic_charges, spin_topology.atom_elements, sizeof(double),
                  &writes[0]) ||
      !make_range(physical_topology.shell_charges, physical_topology.shell_elements, sizeof(double),
                  &writes[1]) ||
      !make_range(physical_topology.atomic_charges, physical_topology.atom_elements, sizeof(double),
                  &writes[2]) ||
      !make_range(physical_topology.atomic_dipoles, physical_topology.dipole_elements,
                  sizeof(double), &writes[3]) ||
      !make_range(physical_topology.atomic_quadrupoles, physical_topology.quadrupole_elements,
                  sizeof(double), &writes[4]) ||
      !make_range(workspace.atom_scratch, workspace.atom_elements, sizeof(double), &writes[5]) ||
      !make_range(workspace.sequence_active, workspace.sequence_elements, sizeof(std::uint32_t),
                  &writes[6]) ||
      !make_range(system_errors, batch.batch_size, sizeof(std::uint32_t), &writes[7]) ||
      !make_range(device_error, 1, sizeof(std::uint32_t), &writes[8])) {
    return false;
  }
  return disjoint_bindings(reads, writes);
}

bool validate_compose(const Gfn2SccPotentialDeviceBatch& batch,
                      const Gfn2SccPotentialDeviceComponents& components,
                      const Gfn2SccPotentialDeviceActivity& activity,
                      const Gfn2SccPotentialDeviceResults& results,
                      const Gfn2SccPotentialDeviceWorkspace& workspace,
                      std::uint32_t* system_errors, std::uint32_t* device_error) noexcept {
  std::int64_t dipole_elements = 0;
  std::int64_t quadrupole_elements = 0;
  const std::uint32_t mask = components.enabled_components;
  const bool field_enabled = components.electric_field_atomic != nullptr ||
                             components.electric_field_atomic_elements != 0 ||
                             components.electric_field_dipole != nullptr ||
                             components.electric_field_dipole_elements != 0;
  if (!validate_common(batch, activity, workspace, system_errors, device_error) ||
      !checked_multiply(batch.total_atoms, kGfn2SccPotentialDipoleComponents, &dipole_elements) ||
      !checked_multiply(batch.total_atoms, kGfn2SccPotentialQuadrupoleComponents,
                        &quadrupole_elements) ||
      (mask & ~kGfn2SccPotentialAllComponents) != 0u || components.plan_token != batch.plan_token ||
      !valid_component(components.es2_shell, components.es2_shell_elements, batch.total_shells,
                       component_enabled(mask, Gfn2SccPotentialComponent::kES2)) ||
      !valid_component(components.es3_shell, components.es3_shell_elements, batch.total_shells,
                       component_enabled(mask, Gfn2SccPotentialComponent::kES3)) ||
      !valid_component(components.explicit_point_charge_shell,
                       components.explicit_point_charge_shell_elements, batch.total_shells,
                       component_enabled(mask, Gfn2SccPotentialComponent::kExplicitPointCharge)) ||
      !valid_component(components.aes2_atomic, components.aes2_atomic_elements, batch.total_atoms,
                       component_enabled(mask, Gfn2SccPotentialComponent::kAES2)) ||
      !valid_component(components.aes2_dipole, components.aes2_dipole_elements, dipole_elements,
                       component_enabled(mask, Gfn2SccPotentialComponent::kAES2)) ||
      !valid_component(components.aes2_quadrupole, components.aes2_quadrupole_elements,
                       quadrupole_elements,
                       component_enabled(mask, Gfn2SccPotentialComponent::kAES2)) ||
      !valid_component(components.d4_atomic, components.d4_atomic_elements, batch.total_atoms,
                       component_enabled(mask, Gfn2SccPotentialComponent::kD4TwoBody)) ||
      !valid_component(components.periodic_atomic, components.periodic_atomic_elements,
                       batch.total_atoms,
                       component_enabled(mask, Gfn2SccPotentialComponent::kPeriodicEmbedding)) ||
      !valid_component(components.electric_field_atomic, components.electric_field_atomic_elements,
                       batch.total_atoms, field_enabled) ||
      !valid_component(components.electric_field_dipole, components.electric_field_dipole_elements,
                       dipole_elements, field_enabled) ||
      results.plan_token != batch.plan_token || results.shell_elements != batch.total_shells ||
      results.atom_elements != batch.total_atoms || results.dipole_elements != dipole_elements ||
      results.quadrupole_elements != quadrupole_elements ||
      (batch.total_shells != 0 && !is_aligned(results.shell, alignof(double))) ||
      (batch.total_atoms != 0 && !is_aligned(results.atomic, alignof(double))) ||
      (dipole_elements != 0 && !is_aligned(results.dipole, alignof(double))) ||
      (quadrupole_elements != 0 && !is_aligned(results.quadrupole, alignof(double)))) {
    return false;
  }
  std::array<AddressRange, 18> reads{};
  std::array<AddressRange, 11> writes{};
  const std::array<const double*, 10> component_pointers{components.es2_shell,
                                                         components.es3_shell,
                                                         components.explicit_point_charge_shell,
                                                         components.aes2_atomic,
                                                         components.aes2_dipole,
                                                         components.aes2_quadrupole,
                                                         components.d4_atomic,
                                                         components.periodic_atomic,
                                                         components.electric_field_atomic,
                                                         components.electric_field_dipole};
  const std::array<std::int64_t, 10> component_elements{
      components.es2_shell_elements,
      components.es3_shell_elements,
      components.explicit_point_charge_shell_elements,
      components.aes2_atomic_elements,
      components.aes2_dipole_elements,
      components.aes2_quadrupole_elements,
      components.d4_atomic_elements,
      components.periodic_atomic_elements,
      components.electric_field_atomic_elements,
      components.electric_field_dipole_elements};
  if (!make_range(batch.atom_offsets, batch.atom_offset_count, sizeof(std::int64_t), &reads[0]) ||
      !make_range(batch.batch_shell_offsets, batch.batch_shell_offset_count, sizeof(std::int64_t),
                  &reads[1]) ||
      !make_range(batch.qsh_offsets, batch.qsh_offset_count, sizeof(std::int64_t), &reads[2]) ||
      !make_range(batch.qat_offsets, batch.qat_offset_count, sizeof(std::int64_t), &reads[3]) ||
      !make_range(batch.dipole_offsets, batch.dipole_offset_count, sizeof(std::int64_t),
                  &reads[4]) ||
      !make_range(batch.quadrupole_offsets, batch.quadrupole_offset_count, sizeof(std::int64_t),
                  &reads[5]) ||
      !make_range(activity.active_mask, activity.elements, sizeof(std::uint8_t), &reads[6])) {
    return false;
  }
  for (std::size_t component = 0; component < component_pointers.size(); ++component) {
    if (!make_range(component_pointers[component], component_elements[component], sizeof(double),
                    &reads[7 + component])) {
      return false;
    }
  }
  if (!make_range(batch.shell_to_atom, batch.shell_to_atom_count, sizeof(std::int64_t),
                  &reads[17]) ||
      !make_range(results.shell, results.shell_elements, sizeof(double), &writes[0]) ||
      !make_range(results.atomic, results.atom_elements, sizeof(double), &writes[1]) ||
      !make_range(results.dipole, results.dipole_elements, sizeof(double), &writes[2]) ||
      !make_range(results.quadrupole, results.quadrupole_elements, sizeof(double), &writes[3]) ||
      !make_range(workspace.shell_scratch, workspace.shell_elements, sizeof(double), &writes[4]) ||
      !make_range(workspace.atom_scratch, workspace.atom_elements, sizeof(double), &writes[5]) ||
      !make_range(workspace.dipole_scratch, workspace.dipole_elements, sizeof(double),
                  &writes[6]) ||
      !make_range(workspace.quadrupole_scratch, workspace.quadrupole_elements, sizeof(double),
                  &writes[7]) ||
      !make_range(workspace.sequence_active, 1, sizeof(std::uint32_t), &writes[8]) ||
      !make_range(system_errors, batch.batch_size, sizeof(std::uint32_t), &writes[9]) ||
      !make_range(device_error, 1, sizeof(std::uint32_t), &writes[10])) {
    return false;
  }
  return disjoint_bindings(reads, writes);
}

bool validate_spin_compose(const Gfn2SccPotentialDeviceBatch& batch,
                           const Gfn2WavefunctionLayoutView& layout,
                           const Gfn2SccPotentialDeviceComponents& components,
                           const Gfn2SccPotentialDeviceSpinComponent& spin,
                           const Gfn2SccPotentialDeviceActivity& activity,
                           const Gfn2SccPotentialDeviceResults& results,
                           const Gfn2SccPotentialDeviceWorkspace& workspace,
                           std::uint32_t* system_errors, std::uint32_t* device_error) noexcept {
  std::int64_t maximum_spin_channels = 0;
  std::int64_t maximum_spin_shells = 0;
  std::int64_t maximum_spin_atoms = 0;
  std::int64_t dipole_elements = 0;
  std::int64_t quadrupole_elements = 0;
  std::int64_t spin_dipole_elements = 0;
  std::int64_t spin_quadrupole_elements = 0;
  const std::uint32_t mask = components.enabled_components;
  const bool field_enabled = components.electric_field_atomic != nullptr ||
                             components.electric_field_atomic_elements != 0 ||
                             components.electric_field_dipole != nullptr ||
                             components.electric_field_dipole_elements != 0;
  if (batch.batch_size <= 0 || batch.batch_size > std::numeric_limits<int>::max() ||
      batch.total_atoms < 0 || batch.total_shells < 0 || batch.plan_token == 0u ||
      !checked_multiply(batch.batch_size, 2, &maximum_spin_channels) ||
      !checked_multiply(batch.total_shells, 2, &maximum_spin_shells) ||
      !checked_multiply(batch.total_atoms, 2, &maximum_spin_atoms) ||
      !checked_multiply(batch.total_atoms, kGfn2SccPotentialDipoleComponents, &dipole_elements) ||
      !checked_multiply(batch.total_atoms, kGfn2SccPotentialQuadrupoleComponents,
                        &quadrupole_elements) ||
      layout.memory_space != Gfn2PlanMemorySpace::kCudaDevice ||
      layout.plan_token != batch.plan_token || layout.batch_size != batch.batch_size ||
      layout.total_spin_channels < batch.batch_size ||
      layout.total_spin_channels > maximum_spin_channels ||
      layout.total_spin_shells < batch.total_shells ||
      layout.total_spin_shells > maximum_spin_shells ||
      layout.total_spin_atoms < batch.total_atoms || layout.total_spin_atoms > maximum_spin_atoms ||
      layout.spin_channel_count != batch.batch_size ||
      layout.spin_channel_offset_count != batch.batch_size + 1 ||
      layout.spin_shell_offset_count != batch.batch_size + 1 ||
      layout.spin_atom_offset_count != batch.batch_size + 1 ||
      !is_aligned(layout.spin_channels, alignof(std::int32_t)) ||
      !is_aligned(layout.spin_channel_offsets, alignof(std::int64_t)) ||
      !is_aligned(layout.spin_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(layout.spin_atom_offsets, alignof(std::int64_t)) ||
      batch.atom_offset_count != batch.batch_size + 1 ||
      batch.batch_shell_offset_count != batch.batch_size + 1 ||
      batch.qsh_offset_count != batch.batch_size + 1 ||
      batch.qat_offset_count != batch.batch_size + 1 ||
      batch.dipole_offset_count != batch.batch_size + 1 ||
      batch.quadrupole_offset_count != batch.batch_size + 1 ||
      batch.shell_to_atom_count != batch.total_shells ||
      !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.batch_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.qsh_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.qat_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.dipole_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.quadrupole_offsets, alignof(std::int64_t)) ||
      (batch.total_shells != 0 && !is_aligned(batch.shell_to_atom, alignof(std::int64_t))) ||
      activity.plan_token != batch.plan_token ||
      ((activity.active_mask == nullptr && activity.elements == 0) ||
       (activity.active_mask != nullptr && activity.elements == batch.batch_size)) == false ||
      (activity.active_mask != nullptr &&
       !is_aligned(activity.active_mask, alignof(std::uint8_t))) ||
      !checked_multiply(layout.total_spin_atoms, kGfn2SccPotentialDipoleComponents,
                        &spin_dipole_elements) ||
      !checked_multiply(layout.total_spin_atoms, kGfn2SccPotentialQuadrupoleComponents,
                        &spin_quadrupole_elements) ||
      (mask & ~kGfn2SccPotentialAllComponents) != 0u || components.plan_token != batch.plan_token ||
      !valid_component(components.es2_shell, components.es2_shell_elements, batch.total_shells,
                       component_enabled(mask, Gfn2SccPotentialComponent::kES2)) ||
      !valid_component(components.es3_shell, components.es3_shell_elements, batch.total_shells,
                       component_enabled(mask, Gfn2SccPotentialComponent::kES3)) ||
      !valid_component(components.explicit_point_charge_shell,
                       components.explicit_point_charge_shell_elements, batch.total_shells,
                       component_enabled(mask, Gfn2SccPotentialComponent::kExplicitPointCharge)) ||
      !valid_component(components.aes2_atomic, components.aes2_atomic_elements, batch.total_atoms,
                       component_enabled(mask, Gfn2SccPotentialComponent::kAES2)) ||
      !valid_component(components.aes2_dipole, components.aes2_dipole_elements, dipole_elements,
                       component_enabled(mask, Gfn2SccPotentialComponent::kAES2)) ||
      !valid_component(components.aes2_quadrupole, components.aes2_quadrupole_elements,
                       quadrupole_elements,
                       component_enabled(mask, Gfn2SccPotentialComponent::kAES2)) ||
      !valid_component(components.d4_atomic, components.d4_atomic_elements, batch.total_atoms,
                       component_enabled(mask, Gfn2SccPotentialComponent::kD4TwoBody)) ||
      !valid_component(components.periodic_atomic, components.periodic_atomic_elements,
                       batch.total_atoms,
                       component_enabled(mask, Gfn2SccPotentialComponent::kPeriodicEmbedding)) ||
      !valid_component(components.electric_field_atomic, components.electric_field_atomic_elements,
                       batch.total_atoms, field_enabled) ||
      !valid_component(components.electric_field_dipole, components.electric_field_dipole_elements,
                       dipole_elements, field_enabled) ||
      spin.plan_token != batch.plan_token || spin.shell_elements != layout.total_spin_shells ||
      (spin.shell_elements != 0 && !is_aligned(spin.shell, alignof(double))) ||
      results.plan_token != batch.plan_token ||
      results.shell_elements != layout.total_spin_shells ||
      results.atom_elements != layout.total_spin_atoms ||
      results.dipole_elements != spin_dipole_elements ||
      results.quadrupole_elements != spin_quadrupole_elements ||
      (results.shell_elements != 0 && !is_aligned(results.shell, alignof(double))) ||
      (results.atom_elements != 0 && !is_aligned(results.atomic, alignof(double))) ||
      (results.dipole_elements != 0 && !is_aligned(results.dipole, alignof(double))) ||
      (results.quadrupole_elements != 0 && !is_aligned(results.quadrupole, alignof(double))) ||
      workspace.plan_token != batch.plan_token ||
      workspace.shell_elements != layout.total_spin_shells ||
      workspace.atom_elements != layout.total_spin_atoms ||
      workspace.dipole_elements != spin_dipole_elements ||
      workspace.quadrupole_elements != spin_quadrupole_elements ||
      workspace.sequence_elements != 1 ||
      (workspace.shell_elements != 0 && !is_aligned(workspace.shell_scratch, alignof(double))) ||
      (workspace.atom_elements != 0 && !is_aligned(workspace.atom_scratch, alignof(double))) ||
      (workspace.dipole_elements != 0 && !is_aligned(workspace.dipole_scratch, alignof(double))) ||
      (workspace.quadrupole_elements != 0 &&
       !is_aligned(workspace.quadrupole_scratch, alignof(double))) ||
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return false;
  }

  std::array<AddressRange, 34> ranges{};
  const std::array<const double*, 10> component_pointers{components.es2_shell,
                                                         components.es3_shell,
                                                         components.explicit_point_charge_shell,
                                                         components.aes2_atomic,
                                                         components.aes2_dipole,
                                                         components.aes2_quadrupole,
                                                         components.d4_atomic,
                                                         components.periodic_atomic,
                                                         components.electric_field_atomic,
                                                         components.electric_field_dipole};
  const std::array<std::int64_t, 10> component_elements{
      components.es2_shell_elements,
      components.es3_shell_elements,
      components.explicit_point_charge_shell_elements,
      components.aes2_atomic_elements,
      components.aes2_dipole_elements,
      components.aes2_quadrupole_elements,
      components.d4_atomic_elements,
      components.periodic_atomic_elements,
      components.electric_field_atomic_elements,
      components.electric_field_dipole_elements};
  if (!make_range(batch.atom_offsets, batch.atom_offset_count, sizeof(std::int64_t), &ranges[0]) ||
      !make_range(batch.batch_shell_offsets, batch.batch_shell_offset_count, sizeof(std::int64_t),
                  &ranges[1]) ||
      !make_range(batch.qsh_offsets, batch.qsh_offset_count, sizeof(std::int64_t), &ranges[2]) ||
      !make_range(batch.qat_offsets, batch.qat_offset_count, sizeof(std::int64_t), &ranges[3]) ||
      !make_range(batch.dipole_offsets, batch.dipole_offset_count, sizeof(std::int64_t),
                  &ranges[4]) ||
      !make_range(batch.quadrupole_offsets, batch.quadrupole_offset_count, sizeof(std::int64_t),
                  &ranges[5]) ||
      !make_range(batch.shell_to_atom, batch.shell_to_atom_count, sizeof(std::int64_t),
                  &ranges[6]) ||
      !make_range(layout.spin_channels, layout.spin_channel_count, sizeof(std::int32_t),
                  &ranges[7]) ||
      !make_range(layout.spin_channel_offsets, layout.spin_channel_offset_count,
                  sizeof(std::int64_t), &ranges[8]) ||
      !make_range(layout.spin_shell_offsets, layout.spin_shell_offset_count, sizeof(std::int64_t),
                  &ranges[9]) ||
      !make_range(layout.spin_atom_offsets, layout.spin_atom_offset_count, sizeof(std::int64_t),
                  &ranges[10]) ||
      !make_range(activity.active_mask, activity.elements, sizeof(std::uint8_t), &ranges[11])) {
    return false;
  }
  for (std::size_t component = 0; component < component_pointers.size(); ++component) {
    if (!make_range(component_pointers[component], component_elements[component], sizeof(double),
                    &ranges[12 + component])) {
      return false;
    }
  }
  if (!make_range(spin.shell, spin.shell_elements, sizeof(double), &ranges[22]) ||
      !make_range(results.shell, results.shell_elements, sizeof(double), &ranges[23]) ||
      !make_range(results.atomic, results.atom_elements, sizeof(double), &ranges[24]) ||
      !make_range(results.dipole, results.dipole_elements, sizeof(double), &ranges[25]) ||
      !make_range(results.quadrupole, results.quadrupole_elements, sizeof(double), &ranges[26]) ||
      !make_range(workspace.shell_scratch, workspace.shell_elements, sizeof(double), &ranges[27]) ||
      !make_range(workspace.atom_scratch, workspace.atom_elements, sizeof(double), &ranges[28]) ||
      !make_range(workspace.dipole_scratch, workspace.dipole_elements, sizeof(double),
                  &ranges[29]) ||
      !make_range(workspace.quadrupole_scratch, workspace.quadrupole_elements, sizeof(double),
                  &ranges[30]) ||
      !make_range(workspace.sequence_active, 1, sizeof(std::uint32_t), &ranges[31]) ||
      !make_range(system_errors, batch.batch_size, sizeof(std::uint32_t), &ranges[32]) ||
      !make_range(device_error, 1, sizeof(std::uint32_t), &ranges[33])) {
    return false;
  }
  /* The physical topology projections intentionally borrow the same
   * read-only offset arrays: qsh_offsets aliases batch_shell_offsets and
   * qat_offsets aliases atom_offsets.  Read/read aliasing is therefore valid;
   * only writable outputs must remain mutually disjoint and isolated from all
   * read ranges. */
  for (std::size_t lhs = 22u; lhs < ranges.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1u; rhs < ranges.size(); ++rhs) {
      if (ranges_overlap(ranges[lhs], ranges[rhs])) {
        return false;
      }
    }
    for (std::size_t rhs = 0u; rhs < 22u; ++rhs) {
      if (ranges_overlap(ranges[lhs], ranges[rhs])) {
        return false;
      }
    }
  }
  return true;
}

bool validate_canonical_compose(const Gfn2SccPotentialDeviceBatch& batch,
                                const Gfn2SccPotentialDeviceComponents& components,
                                const Gfn2SccIterationDeviceActivity& activity,
                                const Gfn2SccPotentialDeviceResults& results,
                                const Gfn2SccPotentialDeviceWorkspace& workspace,
                                std::uint32_t* system_errors,
                                std::uint32_t* device_error) noexcept {
  if (!validate_canonical_activity(batch, activity)) {
    return false;
  }
  const Gfn2SccPotentialDeviceActivity compatibility_activity{
      activity.active_mask, activity.batch_elements, activity.plan_token};
  if (!validate_compose(batch, components, compatibility_activity, results, workspace,
                        system_errors, device_error)) {
    return false;
  }
  AddressRange sequence;
  std::array<AddressRange, 7> writes{};
  if (!make_range(activity.sequence_active, 1, sizeof(std::uint32_t), &sequence) ||
      !make_range(results.shell, results.shell_elements, sizeof(double), &writes[0]) ||
      !make_range(results.atomic, results.atom_elements, sizeof(double), &writes[1]) ||
      !make_range(results.dipole, results.dipole_elements, sizeof(double), &writes[2]) ||
      !make_range(results.quadrupole, results.quadrupole_elements, sizeof(double), &writes[3]) ||
      !make_range(workspace.sequence_active, 1, sizeof(std::uint32_t), &writes[4]) ||
      !make_range(system_errors, batch.batch_size, sizeof(std::uint32_t), &writes[5]) ||
      !make_range(device_error, 1, sizeof(std::uint32_t), &writes[6])) {
    return false;
  }
  for (const AddressRange& write : writes) {
    if (ranges_overlap(sequence, write)) {
      return false;
    }
  }
  return true;
}

cudaError_t launch_preflight(const Gfn2SccPotentialDeviceBatch& batch,
                             const Gfn2SccPotentialDeviceWorkspace& workspace,
                             std::uint32_t* device_error, cudaStream_t stream) noexcept {
  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace.sequence_active);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, workspace.sequence_active,
                                                                device_error);
  return cudaGetLastError();
}

cudaError_t launch_canonical_preflight(const Gfn2SccPotentialDeviceBatch& batch,
                                       const Gfn2SccIterationDeviceActivity& activity,
                                       const Gfn2SccPotentialDeviceWorkspace& workspace,
                                       std::uint32_t* device_error, cudaStream_t stream) noexcept {
  canonical_topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, activity,
                                                                          workspace, device_error);
  return cudaGetLastError();
}

}  // namespace

cudaError_t reset_gfn2_scc_potential_device_errors_cuda(std::int64_t batch_size,
                                                        std::uint32_t* system_errors,
                                                        std::uint32_t* device_error,
                                                        cudaStream_t stream) noexcept {
  if (batch_size <= 0 || !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  AddressRange systems;
  AddressRange device;
  if (!make_range(system_errors, batch_size, sizeof(std::uint32_t), &systems) ||
      !make_range(device_error, 1, sizeof(std::uint32_t), &device) ||
      ranges_overlap(systems, device)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(
      system_errors, 0, static_cast<std::size_t>(batch_size) * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess) {
    return status;
  }
  return cudaMemsetAsync(device_error, 0, sizeof(std::uint32_t), stream);
}

cudaError_t gather_gfn2_scc_mixed_multipoles_cuda(
    const Gfn2SccPotentialDeviceBatch& batch, const Gfn2SccPotentialDeviceMixedFields& mixed,
    const Gfn2SccPotentialDeviceActivity& activity,
    const Gfn2SccPotentialDeviceTopologyMultipoles& topology,
    const Gfn2SccPotentialDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (!validate_gather(batch, mixed, activity, topology, workspace, system_errors, device_error)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = launch_preflight(batch, workspace, device_error, stream);
  if (status != cudaSuccess) {
    return status;
  }
  gather_mixed_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0, stream>>>(
      batch, mixed, activity, workspace, system_errors, device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  publish_mixed_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                         stream>>>(batch, activity, topology, workspace, system_errors);
  return cudaGetLastError();
}

cudaError_t reduce_gfn2_scc_mixed_atomic_charges_cuda(
    const Gfn2SccPotentialDeviceBatch& batch, const Gfn2SccPotentialDeviceMixedFields& mixed,
    const Gfn2SccIterationDeviceActivity& activity,
    const Gfn2SccPotentialDeviceTopologyMultipoles& topology,
    const Gfn2SccPotentialDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (!validate_canonical_reduction(batch, mixed, activity, topology, workspace, system_errors,
                                    device_error)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = launch_canonical_preflight(batch, activity, workspace, device_error, stream);
  if (status != cudaSuccess) {
    return status;
  }
  reduce_mixed_atomic_charge_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock,
                                      0, stream>>>(batch, mixed, activity, workspace, system_errors,
                                                   device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  publish_mixed_atomic_charge_kernel<<<static_cast<unsigned int>(batch.batch_size),
                                       kThreadsPerBlock, 0, stream>>>(batch, activity, topology,
                                                                      workspace, system_errors);
  return cudaGetLastError();
}

cudaError_t reduce_gfn2_scc_spin_atomic_charges_cuda(
    const Gfn2SccPotentialDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2SccPotentialDeviceMixedFields& mixed, const Gfn2SccIterationDeviceActivity& activity,
    const Gfn2SccPotentialDeviceTopologyMultipoles& spin_topology,
    const Gfn2SccPotentialDeviceTopologyMultipoles& physical_topology,
    const Gfn2SccPotentialDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (!validate_spin_reduction(batch, layout, mixed, activity, spin_topology, physical_topology,
                               workspace, system_errors, device_error)) {
    return batch.batch_size > std::numeric_limits<int>::max() ? cudaErrorInvalidConfiguration
                                                              : cudaErrorInvalidValue;
  }
  capture_canonical_sequence_kernel<<<1, 1, 0, stream>>>(activity, device_error,
                                                         workspace.sequence_active);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  spin_reduction_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock,
                                    0, stream>>>(batch, layout, activity, workspace, system_errors,
                                                 device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  reduce_spin_atomic_charges_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock,
                                      0, stream>>>(batch, layout, mixed, activity, workspace,
                                                   system_errors, device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  publish_spin_multipole_projection_kernel<<<static_cast<unsigned int>(batch.batch_size),
                                             kThreadsPerBlock, 0, stream>>>(
      batch, layout, mixed, activity, spin_topology, physical_topology, workspace, system_errors);
  return cudaGetLastError();
}

cudaError_t compose_gfn2_scc_potentials_cuda(
    const Gfn2SccPotentialDeviceBatch& batch, const Gfn2SccPotentialDeviceComponents& components,
    const Gfn2SccPotentialDeviceActivity& activity, const Gfn2SccPotentialDeviceResults& results,
    const Gfn2SccPotentialDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (!validate_compose(batch, components, activity, results, workspace, system_errors,
                        device_error)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = launch_preflight(batch, workspace, device_error, stream);
  if (status != cudaSuccess) {
    return status;
  }
  compose_potential_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                             stream>>>(batch, components, activity, workspace, system_errors,
                                       device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  publish_potential_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                             stream>>>(batch, activity, results, workspace, system_errors);
  return cudaGetLastError();
}

cudaError_t compose_gfn2_scc_spin_potentials_cuda(
    const Gfn2SccPotentialDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2SccPotentialDeviceComponents& components,
    const Gfn2SccPotentialDeviceSpinComponent& spin, const Gfn2SccPotentialDeviceActivity& activity,
    const Gfn2SccPotentialDeviceResults& results, const Gfn2SccPotentialDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (!validate_spin_compose(batch, layout, components, spin, activity, results, workspace,
                             system_errors, device_error)) {
    return batch.batch_size > std::numeric_limits<int>::max() ? cudaErrorInvalidConfiguration
                                                              : cudaErrorInvalidValue;
  }

  /* This preflight is intentionally peer-local. Inactive systems may contain
   * generation-local poison in both topology and spin offsets and therefore
   * must not be inspected by a plan-wide compatibility preflight. */
  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace.sequence_active);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  spin_potential_topology_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size),
                                             kThreadsPerBlock, 0, stream>>>(
      batch, layout, activity, workspace, system_errors, device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  compose_spin_potential_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                                  stream>>>(batch, layout, components, spin, activity, workspace,
                                            system_errors, device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  publish_spin_potential_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                                  stream>>>(batch, layout, activity, results, workspace,
                                            system_errors);
  return cudaGetLastError();
}

cudaError_t compose_gfn2_scc_potentials_cuda(
    const Gfn2SccPotentialDeviceBatch& batch, const Gfn2SccPotentialDeviceComponents& components,
    const Gfn2SccIterationDeviceActivity& activity, const Gfn2SccPotentialDeviceResults& results,
    const Gfn2SccPotentialDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (!validate_canonical_compose(batch, components, activity, results, workspace, system_errors,
                                  device_error)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = launch_canonical_preflight(batch, activity, workspace, device_error, stream);
  if (status != cudaSuccess) {
    return status;
  }
  compose_potential_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                             stream>>>(batch, components, activity, workspace, system_errors,
                                       device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  publish_potential_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                             stream>>>(batch, activity, results, workspace, system_errors);
  return cudaGetLastError();
}

}  // namespace xtbloom::detail::cuda
