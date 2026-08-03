#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_scc_potential.cuh"

namespace gpuxtb::detail::cuda {
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

__device__ bool sequence_is_active(const Gfn2SccPotentialDeviceWorkspace& workspace) {
  return atomicAdd(workspace.sequence_active, 0u) == 1u;
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

__global__ void compose_potential_kernel(Gfn2SccPotentialDeviceBatch batch,
                                         Gfn2SccPotentialDeviceComponents components,
                                         Gfn2SccPotentialDeviceActivity activity,
                                         Gfn2SccPotentialDeviceWorkspace workspace,
                                         std::uint32_t* system_errors,
                                         std::uint32_t* device_error) {
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
    const double first = aes2_atomic + periodic;
    const double total = first + d4;
    if (!isfinite(first) || !isfinite(total)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccPotentialDeviceError::kNonfiniteAtomicPotentialArithmetic);
      atomicExch(&valid, 0);
    } else {
      workspace.atom_scratch[qat_begin + local_atom] = total;
    }
    for (std::int64_t component = 0; component < kGfn2SccPotentialDipoleComponents; ++component) {
      workspace.dipole_scratch[dipole_begin + local_atom * kGfn2SccPotentialDipoleComponents +
                               component] =
          load_component(components.aes2_dipole,
                         atom * kGfn2SccPotentialDipoleComponents + component, aes2_enabled,
                         Gfn2SccPotentialDeviceError::kNonfiniteAES2Potential, &valid,
                         system_errors, system, device_error);
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

__global__ void publish_potential_kernel(Gfn2SccPotentialDeviceBatch batch,
                                         Gfn2SccPotentialDeviceActivity activity,
                                         Gfn2SccPotentialDeviceResults results,
                                         Gfn2SccPotentialDeviceWorkspace workspace,
                                         const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!sequence_is_active(workspace) || !system_is_valid(system_errors, system) ||
      (activity.active_mask != nullptr && activity.active_mask[system] != 1u)) {
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

bool validate_compose(const Gfn2SccPotentialDeviceBatch& batch,
                      const Gfn2SccPotentialDeviceComponents& components,
                      const Gfn2SccPotentialDeviceActivity& activity,
                      const Gfn2SccPotentialDeviceResults& results,
                      const Gfn2SccPotentialDeviceWorkspace& workspace,
                      std::uint32_t* system_errors, std::uint32_t* device_error) noexcept {
  std::int64_t dipole_elements = 0;
  std::int64_t quadrupole_elements = 0;
  const std::uint32_t mask = components.enabled_components;
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
      results.plan_token != batch.plan_token || results.shell_elements != batch.total_shells ||
      results.atom_elements != batch.total_atoms || results.dipole_elements != dipole_elements ||
      results.quadrupole_elements != quadrupole_elements ||
      (batch.total_shells != 0 && !is_aligned(results.shell, alignof(double))) ||
      (batch.total_atoms != 0 && !is_aligned(results.atomic, alignof(double))) ||
      (dipole_elements != 0 && !is_aligned(results.dipole, alignof(double))) ||
      (quadrupole_elements != 0 && !is_aligned(results.quadrupole, alignof(double)))) {
    return false;
  }
  std::array<AddressRange, 16> reads{};
  std::array<AddressRange, 11> writes{};
  const std::array<const double*, 8> component_pointers{
      components.es2_shell,   components.es3_shell,      components.explicit_point_charge_shell,
      components.aes2_atomic, components.aes2_dipole,    components.aes2_quadrupole,
      components.d4_atomic,   components.periodic_atomic};
  const std::array<std::int64_t, 8> component_elements{
      components.es2_shell_elements,
      components.es3_shell_elements,
      components.explicit_point_charge_shell_elements,
      components.aes2_atomic_elements,
      components.aes2_dipole_elements,
      components.aes2_quadrupole_elements,
      components.d4_atomic_elements,
      components.periodic_atomic_elements};
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
                  &reads[15]) ||
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

}  // namespace gpuxtb::detail::cuda
