#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_hamiltonian_force.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr std::int64_t kMaximumInt64 = 9223372036854775807LL;

struct SystemRanges {
  std::int64_t atom_begin;
  std::int64_t atom_end;
  std::int64_t shell_begin;
  std::int64_t shell_end;
  std::int64_t orbital_begin;
  std::int64_t orbital_end;
  std::int64_t matrix_begin;
  std::int64_t matrix_end;
};

__device__ bool sequence_is_active(const Gfn2HamiltonianForceDeviceWorkspace& workspace) {
  return atomicAdd(workspace.sequence_active, 0u) == 1u;
}

__device__ bool system_is_valid(const std::uint32_t* system_errors, std::int64_t system) {
  return atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) ==
         static_cast<std::uint32_t>(Gfn2HamiltonianForceDeviceError::kSuccess);
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    std::uint32_t* device_error,
                                    Gfn2HamiltonianForceDeviceError error) {
  const std::uint32_t success =
      static_cast<std::uint32_t>(Gfn2HamiltonianForceDeviceError::kSuccess);
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  if (atomicCAS(system_errors + system, success, code) == success) {
    atomicCAS(device_error, success, code);
  }
}

__device__ bool valid_range(std::int64_t begin, std::int64_t end, std::int64_t total) {
  return begin >= 0 && begin <= end && end <= total;
}

__device__ bool checked_square(std::int64_t value, std::int64_t* square) {
  if (value < 0 || (value != 0 && value > kMaximumInt64 / value)) {
    return false;
  }
  *square = value * value;
  return true;
}

__global__ void capture_sequence_kernel(const std::uint32_t* device_error,
                                        Gfn2HamiltonianForceDeviceWorkspace workspace) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *workspace.sequence_active =
        atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                static_cast<std::uint32_t>(Gfn2HamiltonianForceDeviceError::kSuccess)
            ? 1u
            : 0u;
  }
}

__device__ bool load_force_gate(const Gfn2ForceDeviceActivity& activity, std::int64_t system,
                                int* selected, std::uint32_t* system_errors,
                                std::uint32_t* device_error) {
  if (threadIdx.x == 0) {
    *selected = 0;
    const std::uint8_t requested = activity.requested_mask[system];
    if (requested > 1u) {
      record_system_error(system_errors, system, device_error,
                          Gfn2HamiltonianForceDeviceError::kInvalidActiveMask);
    } else if (requested == 1u && activity.system_statuses[system] == GPUXTB_STATUS_SUCCESS) {
      *selected = 1;
    }
  }
  __syncthreads();
  return *selected != 0 && system_is_valid(system_errors, system);
}

__device__ bool load_ranges(const Gfn2HamiltonianDeviceBatch& batch, std::int64_t system,
                            SystemRanges* ranges) {
  ranges->atom_begin = batch.atom_offsets[system];
  ranges->atom_end = batch.atom_offsets[system + 1];
  ranges->shell_begin = batch.batch_shell_offsets[system];
  ranges->shell_end = batch.batch_shell_offsets[system + 1];
  ranges->orbital_begin = batch.batch_orbital_offsets[system];
  ranges->orbital_end = batch.batch_orbital_offsets[system + 1];
  ranges->matrix_begin = batch.matrix_offsets[system];
  ranges->matrix_end = batch.matrix_offsets[system + 1];
  if (!valid_range(ranges->atom_begin, ranges->atom_end, batch.total_atoms) ||
      !valid_range(ranges->shell_begin, ranges->shell_end, batch.total_shells) ||
      !valid_range(ranges->orbital_begin, ranges->orbital_end, batch.total_orbitals) ||
      !valid_range(ranges->matrix_begin, ranges->matrix_end, batch.total_matrix_elements)) {
    return false;
  }
  std::int64_t expected_matrix = 0;
  const std::int64_t orbitals = ranges->orbital_end - ranges->orbital_begin;
  return checked_square(orbitals, &expected_matrix) &&
         ranges->matrix_end - ranges->matrix_begin == expected_matrix &&
         ranges->atom_begin < ranges->atom_end && ranges->shell_begin < ranges->shell_end &&
         ranges->orbital_begin < ranges->orbital_end &&
         batch.atom_shell_offsets[ranges->atom_begin] == ranges->shell_begin &&
         batch.atom_shell_offsets[ranges->atom_end] == ranges->shell_end &&
         batch.shell_orbital_offsets[ranges->shell_begin] == ranges->orbital_begin &&
         batch.shell_orbital_offsets[ranges->shell_end] == ranges->orbital_end &&
         (system != 0 || (ranges->atom_begin == 0 && ranges->shell_begin == 0 &&
                          ranges->orbital_begin == 0 && ranges->matrix_begin == 0)) &&
         (system + 1 != batch.batch_size ||
          (ranges->atom_end == batch.total_atoms && ranges->shell_end == batch.total_shells &&
           ranges->orbital_end == batch.total_orbitals &&
           ranges->matrix_end == batch.total_matrix_elements));
}

__global__ void preflight_and_seed_kernel(Gfn2HamiltonianDeviceBatch batch,
                                          Gfn2ForceDeviceActivity activity,
                                          Gfn2HamiltonianForceDeviceInput input,
                                          Gfn2HamiltonianForceDeviceOutput output,
                                          Gfn2HamiltonianForceDeviceWorkspace workspace,
                                          std::uint32_t* system_errors,
                                          std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int selected;
  __shared__ int valid;
  if (!sequence_is_active(workspace) ||
      !load_force_gate(activity, system, &selected, system_errors, device_error)) {
    return;
  }
  if (threadIdx.x == 0) {
    valid = load_ranges(batch, system, &ranges) ? 1 : 0;
    if (valid == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2HamiltonianForceDeviceError::kInvalidOffsets);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    const std::int64_t begin = batch.atom_shell_offsets[atom];
    const std::int64_t end = batch.atom_shell_offsets[atom + 1];
    if (begin < ranges.shell_begin || begin >= end || end > ranges.shell_end) {
      record_system_error(system_errors, system, device_error,
                          Gfn2HamiltonianForceDeviceError::kInvalidOffsets);
      atomicExch(&valid, 0);
    }
    for (int component = 0; component < kGfn2HamiltonianDipoleComponents; ++component) {
      if (!isfinite(input.atomic_dipole_potentials[atom * kGfn2HamiltonianDipoleComponents +
                                                   component])) {
        record_system_error(system_errors, system, device_error,
                            Gfn2HamiltonianForceDeviceError::kNonfiniteInput);
        atomicExch(&valid, 0);
      }
    }
    for (int component = 0; component < kGfn2HamiltonianQuadrupoleComponents; ++component) {
      if (!isfinite(input.atomic_quadrupole_potentials[atom * kGfn2HamiltonianQuadrupoleComponents +
                                                       component])) {
        record_system_error(system_errors, system, device_error,
                            Gfn2HamiltonianForceDeviceError::kNonfiniteInput);
        atomicExch(&valid, 0);
      }
    }
  }
  for (std::int64_t shell = ranges.shell_begin + threadIdx.x; shell < ranges.shell_end;
       shell += blockDim.x) {
    const std::int64_t atom = batch.shell_to_atom[shell];
    const std::int64_t orbital_begin = batch.shell_orbital_offsets[shell];
    const std::int64_t orbital_end = batch.shell_orbital_offsets[shell + 1];
    if (atom < ranges.atom_begin || atom >= ranges.atom_end ||
        shell < batch.atom_shell_offsets[atom] || shell >= batch.atom_shell_offsets[atom + 1] ||
        orbital_begin < ranges.orbital_begin || orbital_begin >= orbital_end ||
        orbital_end > ranges.orbital_end) {
      record_system_error(system_errors, system, device_error,
                          Gfn2HamiltonianForceDeviceError::kInvalidOrbitalMetadata);
      atomicExch(&valid, 0);
    }
    if (!isfinite(input.shell_scalar_potentials[shell])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2HamiltonianForceDeviceError::kNonfiniteInput);
      atomicExch(&valid, 0);
    }
    if (input.spin_density != nullptr && !isfinite(input.spin_shell_scalar_potentials[shell])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2HamiltonianForceDeviceError::kNonfiniteInput);
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t orbital = ranges.orbital_begin + threadIdx.x; orbital < ranges.orbital_end;
       orbital += blockDim.x) {
    const std::int64_t shell = batch.orbital_to_shell[orbital];
    const std::int64_t atom = batch.orbital_to_atom[orbital];
    if (shell < ranges.shell_begin || shell >= ranges.shell_end || atom < ranges.atom_begin ||
        atom >= ranges.atom_end || orbital < batch.shell_orbital_offsets[shell] ||
        orbital >= batch.shell_orbital_offsets[shell + 1] || batch.shell_to_atom[shell] != atom) {
      record_system_error(system_errors, system, device_error,
                          Gfn2HamiltonianForceDeviceError::kInvalidOrbitalMetadata);
      atomicExch(&valid, 0);
    }
  }

  const std::int64_t total_matrix = batch.total_matrix_elements;
  for (std::int64_t matrix = ranges.matrix_begin + threadIdx.x; matrix < ranges.matrix_end;
       matrix += blockDim.x) {
    if (!isfinite(input.density[matrix])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2HamiltonianForceDeviceError::kNonfiniteInput);
      atomicExch(&valid, 0);
    }
    if (input.spin_density != nullptr && !isfinite(input.spin_density[matrix])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2HamiltonianForceDeviceError::kNonfiniteInput);
      atomicExch(&valid, 0);
    }
    const double overlap_seed = output.overlap_adjoint[matrix];
    if (!isfinite(overlap_seed)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2HamiltonianForceDeviceError::kNonfiniteOutputSeed);
      atomicExch(&valid, 0);
    } else {
      workspace.overlap_adjoint_scratch[matrix] = overlap_seed;
    }
    for (int component = 0; component < kGfn2HamiltonianDipoleComponents; ++component) {
      const std::int64_t index = component * total_matrix + matrix;
      const double seed = output.dipole_adjoint[index];
      if (!isfinite(seed)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2HamiltonianForceDeviceError::kNonfiniteOutputSeed);
        atomicExch(&valid, 0);
      } else {
        workspace.dipole_adjoint_scratch[index] = seed;
      }
    }
    for (int component = 0; component < kGfn2HamiltonianQuadrupoleComponents; ++component) {
      const std::int64_t index = component * total_matrix + matrix;
      const double seed = output.quadrupole_adjoint[index];
      if (!isfinite(seed)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2HamiltonianForceDeviceError::kNonfiniteOutputSeed);
        atomicExch(&valid, 0);
      } else {
        workspace.quadrupole_adjoint_scratch[index] = seed;
      }
    }
  }
}

__global__ void contract_kernel(Gfn2HamiltonianDeviceBatch batch, Gfn2ForceDeviceActivity activity,
                                Gfn2HamiltonianForceDeviceInput input,
                                Gfn2HamiltonianForceDeviceWorkspace workspace,
                                std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int selected;
  if (!sequence_is_active(workspace) ||
      !load_force_gate(activity, system, &selected, system_errors, device_error)) {
    return;
  }
  const std::int64_t orbital_begin = batch.batch_orbital_offsets[system];
  const std::int64_t orbitals = batch.batch_orbital_offsets[system + 1] - orbital_begin;
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t total_matrix = batch.total_matrix_elements;
  for (std::int64_t local = threadIdx.x; local < orbitals * orbitals; local += blockDim.x) {
    const std::int64_t local_row = local / orbitals;
    const std::int64_t local_column = local - local_row * orbitals;
    if (local_column < local_row) {
      continue;
    }
    const std::int64_t row = orbital_begin + local_row;
    const std::int64_t column = orbital_begin + local_column;
    const std::int64_t row_shell = batch.orbital_to_shell[row];
    const std::int64_t column_shell = batch.orbital_to_shell[column];
    const std::int64_t row_atom = batch.orbital_to_atom[row];
    const std::int64_t column_atom = batch.orbital_to_atom[column];
    const std::int64_t forward = matrix_begin + local_row * orbitals + local_column;
    const std::int64_t reverse = matrix_begin + local_column * orbitals + local_row;
    const double pair_density =
        input.density[forward] + (forward == reverse ? 0.0 : input.density[reverse]);
    const double scalar_factor = -0.5 * (input.shell_scalar_potentials[row_shell] +
                                         input.shell_scalar_potentials[column_shell]);
    const double overlap_contribution = pair_density * scalar_factor;
    const double charge_overlap_updated =
        workspace.overlap_adjoint_scratch[forward] + overlap_contribution;
    bool finite = isfinite(pair_density) && isfinite(scalar_factor) &&
                  isfinite(overlap_contribution) && isfinite(charge_overlap_updated);
    if (finite) {
      /* Match the CPU composer order: charge response is accumulated first,
       * followed by the independent magnetization overlap response. */
      workspace.overlap_adjoint_scratch[forward] = charge_overlap_updated;
      if (input.spin_density != nullptr) {
        const double pair_spin_density =
            input.spin_density[forward] + (forward == reverse ? 0.0 : input.spin_density[reverse]);
        const double spin_scalar_factor = -0.5 * (input.spin_shell_scalar_potentials[row_shell] +
                                                  input.spin_shell_scalar_potentials[column_shell]);
        const double spin_overlap_contribution = pair_spin_density * spin_scalar_factor;
        const double spin_overlap_updated = charge_overlap_updated + spin_overlap_contribution;
        finite = isfinite(pair_spin_density) && isfinite(spin_scalar_factor) &&
                 isfinite(spin_overlap_contribution) && isfinite(spin_overlap_updated);
        if (finite) {
          workspace.overlap_adjoint_scratch[forward] = spin_overlap_updated;
        }
      }
    }

    for (int component = 0; component < kGfn2HamiltonianDipoleComponents && finite; ++component) {
      const std::int64_t forward_index = component * total_matrix + forward;
      const std::int64_t reverse_index = component * total_matrix + reverse;
      const double forward_contribution =
          -0.5 * pair_density *
          input
              .atomic_dipole_potentials[column_atom * kGfn2HamiltonianDipoleComponents + component];
      const double reverse_contribution =
          -0.5 * pair_density *
          input.atomic_dipole_potentials[row_atom * kGfn2HamiltonianDipoleComponents + component];
      const double forward_updated =
          workspace.dipole_adjoint_scratch[forward_index] + forward_contribution;
      if (forward_index == reverse_index) {
        const double diagonal_updated = forward_updated + reverse_contribution;
        finite = isfinite(forward_contribution) && isfinite(reverse_contribution) &&
                 isfinite(forward_updated) && isfinite(diagonal_updated);
        if (finite) {
          workspace.dipole_adjoint_scratch[forward_index] = diagonal_updated;
        }
      } else {
        const double reverse_updated =
            workspace.dipole_adjoint_scratch[reverse_index] + reverse_contribution;
        finite = isfinite(forward_contribution) && isfinite(reverse_contribution) &&
                 isfinite(forward_updated) && isfinite(reverse_updated);
        if (finite) {
          workspace.dipole_adjoint_scratch[forward_index] = forward_updated;
          workspace.dipole_adjoint_scratch[reverse_index] = reverse_updated;
        }
      }
    }
    for (int component = 0; component < kGfn2HamiltonianQuadrupoleComponents && finite;
         ++component) {
      const std::int64_t forward_index = component * total_matrix + forward;
      const std::int64_t reverse_index = component * total_matrix + reverse;
      const double forward_contribution =
          -0.5 * pair_density *
          input.atomic_quadrupole_potentials[column_atom * kGfn2HamiltonianQuadrupoleComponents +
                                             component];
      const double reverse_contribution =
          -0.5 * pair_density *
          input.atomic_quadrupole_potentials[row_atom * kGfn2HamiltonianQuadrupoleComponents +
                                             component];
      const double forward_updated =
          workspace.quadrupole_adjoint_scratch[forward_index] + forward_contribution;
      if (forward_index == reverse_index) {
        const double diagonal_updated = forward_updated + reverse_contribution;
        finite = isfinite(forward_contribution) && isfinite(reverse_contribution) &&
                 isfinite(forward_updated) && isfinite(diagonal_updated);
        if (finite) {
          workspace.quadrupole_adjoint_scratch[forward_index] = diagonal_updated;
        }
      } else {
        const double reverse_updated =
            workspace.quadrupole_adjoint_scratch[reverse_index] + reverse_contribution;
        finite = isfinite(forward_contribution) && isfinite(reverse_contribution) &&
                 isfinite(forward_updated) && isfinite(reverse_updated);
        if (finite) {
          workspace.quadrupole_adjoint_scratch[forward_index] = forward_updated;
          workspace.quadrupole_adjoint_scratch[reverse_index] = reverse_updated;
        }
      }
    }
    if (!finite) {
      record_system_error(system_errors, system, device_error,
                          Gfn2HamiltonianForceDeviceError::kNonfiniteArithmetic);
    }
  }
}

__global__ void publish_kernel(Gfn2HamiltonianDeviceBatch batch, Gfn2ForceDeviceActivity activity,
                               Gfn2HamiltonianForceDeviceOutput output,
                               Gfn2HamiltonianForceDeviceWorkspace workspace,
                               const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!sequence_is_active(workspace) || activity.requested_mask[system] != 1u ||
      activity.system_statuses[system] != GPUXTB_STATUS_SUCCESS ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t begin = batch.matrix_offsets[system];
  const std::int64_t end = batch.matrix_offsets[system + 1];
  const std::int64_t total_matrix = batch.total_matrix_elements;
  for (std::int64_t matrix = begin + threadIdx.x; matrix < end; matrix += blockDim.x) {
    output.overlap_adjoint[matrix] = workspace.overlap_adjoint_scratch[matrix];
    for (int component = 0; component < kGfn2HamiltonianDipoleComponents; ++component) {
      const std::int64_t index = component * total_matrix + matrix;
      output.dipole_adjoint[index] = workspace.dipole_adjoint_scratch[index];
    }
    for (int component = 0; component < kGfn2HamiltonianQuadrupoleComponents; ++component) {
      const std::int64_t index = component * total_matrix + matrix;
      output.quadrupole_adjoint[index] = workspace.quadrupole_adjoint_scratch[index];
    }
  }
}

bool is_aligned(const void* pointer, std::size_t alignment) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

template <typename T>
bool required_pointer(const T* pointer, std::int64_t elements) noexcept {
  return elements == 0 || is_aligned(pointer, alignof(T));
}

struct MemoryRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool make_range(const void* pointer, std::int64_t elements, std::size_t element_size,
                MemoryRange* range) noexcept {
  if (elements < 0 || element_size == 0u ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(elements) * element_size;
  if (bytes == 0u) {
    *range = {};
    return true;
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

bool overlaps(const MemoryRange& first, const MemoryRange& second) noexcept {
  return first.begin < second.end && second.begin < first.end;
}

template <std::size_t ReadCount, std::size_t WriteCount>
bool writes_are_disjoint(const std::array<MemoryRange, ReadCount>& reads,
                         const std::array<MemoryRange, WriteCount>& writes) noexcept {
  for (std::size_t write = 0u; write < WriteCount; ++write) {
    for (const MemoryRange& read : reads) {
      if (overlaps(writes[write], read)) {
        return false;
      }
    }
    for (std::size_t other = write + 1u; other < WriteCount; ++other) {
      if (overlaps(writes[write], writes[other])) {
        return false;
      }
    }
  }
  return true;
}

cudaError_t validate_descriptors(const Gfn2HamiltonianDeviceBatch& batch,
                                 const Gfn2ForceDeviceActivity& activity,
                                 const Gfn2HamiltonianForceDeviceInput& input,
                                 const Gfn2HamiltonianForceDeviceOutput& output,
                                 const Gfn2HamiltonianForceDeviceWorkspace& workspace,
                                 std::uint32_t* system_errors,
                                 std::uint32_t* device_error) noexcept {
  const bool has_spin = input.spin_density != nullptr;
  if (batch.batch_size <= 0 || batch.batch_size > std::numeric_limits<int>::max() ||
      batch.total_atoms <= 0 || batch.total_shells <= 0 || batch.total_orbitals <= 0 ||
      batch.total_matrix_elements <= 0 ||
      batch.total_atoms > kMaximumInt64 / kGfn2HamiltonianQuadrupoleComponents ||
      batch.total_matrix_elements > kMaximumInt64 / kGfn2HamiltonianQuadrupoleComponents ||
      batch.plan_token == 0u || activity.plan_token != batch.plan_token ||
      input.plan_token != batch.plan_token || output.plan_token != batch.plan_token ||
      workspace.plan_token != batch.plan_token || activity.batch_elements != batch.batch_size ||
      batch.atom_offset_count != batch.batch_size + 1 ||
      batch.batch_shell_offset_count != batch.batch_size + 1 ||
      batch.batch_orbital_offset_count != batch.batch_size + 1 ||
      batch.matrix_offset_count != batch.batch_size + 1 ||
      batch.atom_shell_offset_count < batch.total_atoms + 1 ||
      batch.shell_orbital_offset_count < batch.total_shells + 1 ||
      batch.shell_to_atom_count < batch.total_shells ||
      batch.orbital_to_shell_count < batch.total_orbitals ||
      batch.orbital_to_atom_count < batch.total_orbitals ||
      input.density_elements < batch.total_matrix_elements ||
      input.shell_scalar_elements < batch.total_shells ||
      input.atomic_dipole_elements < batch.total_atoms * kGfn2HamiltonianDipoleComponents ||
      input.atomic_quadrupole_elements < batch.total_atoms * kGfn2HamiltonianQuadrupoleComponents ||
      has_spin != (input.spin_shell_scalar_potentials != nullptr) ||
      (!has_spin && (input.spin_density_elements != 0 || input.spin_shell_scalar_elements != 0)) ||
      (has_spin && (input.spin_density_elements < batch.total_matrix_elements ||
                    input.spin_shell_scalar_elements < batch.total_shells)) ||
      output.overlap_adjoint_elements < batch.total_matrix_elements ||
      output.dipole_adjoint_elements <
          batch.total_matrix_elements * kGfn2HamiltonianDipoleComponents ||
      output.quadrupole_adjoint_elements <
          batch.total_matrix_elements * kGfn2HamiltonianQuadrupoleComponents ||
      workspace.overlap_adjoint_elements < batch.total_matrix_elements ||
      workspace.dipole_adjoint_elements <
          batch.total_matrix_elements * kGfn2HamiltonianDipoleComponents ||
      workspace.quadrupole_adjoint_elements <
          batch.total_matrix_elements * kGfn2HamiltonianQuadrupoleComponents ||
      workspace.sequence_elements < 1 || !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.batch_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.batch_orbital_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.matrix_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.atom_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_orbital_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_to_atom, alignof(std::int64_t)) ||
      !is_aligned(batch.orbital_to_shell, alignof(std::int64_t)) ||
      !is_aligned(batch.orbital_to_atom, alignof(std::int64_t)) ||
      !is_aligned(activity.requested_mask, alignof(std::uint8_t)) ||
      !is_aligned(activity.system_statuses, alignof(gpuxtb_status_t)) ||
      !required_pointer(input.density, batch.total_matrix_elements) ||
      !required_pointer(input.shell_scalar_potentials, batch.total_shells) ||
      !required_pointer(input.atomic_dipole_potentials,
                        batch.total_atoms * kGfn2HamiltonianDipoleComponents) ||
      !required_pointer(input.atomic_quadrupole_potentials,
                        batch.total_atoms * kGfn2HamiltonianQuadrupoleComponents) ||
      (has_spin && (!is_aligned(input.spin_density, alignof(double)) ||
                    !is_aligned(input.spin_shell_scalar_potentials, alignof(double)))) ||
      !required_pointer(output.overlap_adjoint, batch.total_matrix_elements) ||
      !required_pointer(output.dipole_adjoint,
                        batch.total_matrix_elements * kGfn2HamiltonianDipoleComponents) ||
      !required_pointer(output.quadrupole_adjoint,
                        batch.total_matrix_elements * kGfn2HamiltonianQuadrupoleComponents) ||
      !required_pointer(workspace.overlap_adjoint_scratch, batch.total_matrix_elements) ||
      !required_pointer(workspace.dipole_adjoint_scratch,
                        batch.total_matrix_elements * kGfn2HamiltonianDipoleComponents) ||
      !required_pointer(workspace.quadrupole_adjoint_scratch,
                        batch.total_matrix_elements * kGfn2HamiltonianQuadrupoleComponents) ||
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return batch.batch_size > std::numeric_limits<int>::max() ? cudaErrorInvalidConfiguration
                                                              : cudaErrorInvalidValue;
  }

  std::array<MemoryRange, 17> reads;
  std::array<MemoryRange, 9> writes;
  if (!make_range(batch.atom_offsets, batch.atom_offset_count, sizeof(*batch.atom_offsets),
                  &reads[0]) ||
      !make_range(batch.batch_shell_offsets, batch.batch_shell_offset_count,
                  sizeof(*batch.batch_shell_offsets), &reads[1]) ||
      !make_range(batch.batch_orbital_offsets, batch.batch_orbital_offset_count,
                  sizeof(*batch.batch_orbital_offsets), &reads[2]) ||
      !make_range(batch.matrix_offsets, batch.matrix_offset_count, sizeof(*batch.matrix_offsets),
                  &reads[3]) ||
      !make_range(batch.atom_shell_offsets, batch.atom_shell_offset_count,
                  sizeof(*batch.atom_shell_offsets), &reads[4]) ||
      !make_range(batch.shell_orbital_offsets, batch.shell_orbital_offset_count,
                  sizeof(*batch.shell_orbital_offsets), &reads[5]) ||
      !make_range(batch.shell_to_atom, batch.total_shells, sizeof(*batch.shell_to_atom),
                  &reads[6]) ||
      !make_range(batch.orbital_to_shell, batch.total_orbitals, sizeof(*batch.orbital_to_shell),
                  &reads[7]) ||
      !make_range(batch.orbital_to_atom, batch.total_orbitals, sizeof(*batch.orbital_to_atom),
                  &reads[8]) ||
      !make_range(activity.requested_mask, batch.batch_size, sizeof(*activity.requested_mask),
                  &reads[9]) ||
      !make_range(activity.system_statuses, batch.batch_size, sizeof(*activity.system_statuses),
                  &reads[10]) ||
      !make_range(input.density, batch.total_matrix_elements, sizeof(*input.density), &reads[11]) ||
      !make_range(input.shell_scalar_potentials, batch.total_shells,
                  sizeof(*input.shell_scalar_potentials), &reads[12]) ||
      !make_range(input.atomic_dipole_potentials,
                  batch.total_atoms * kGfn2HamiltonianDipoleComponents,
                  sizeof(*input.atomic_dipole_potentials), &reads[13]) ||
      !make_range(input.atomic_quadrupole_potentials,
                  batch.total_atoms * kGfn2HamiltonianQuadrupoleComponents,
                  sizeof(*input.atomic_quadrupole_potentials), &reads[14]) ||
      !make_range(input.spin_density, has_spin ? batch.total_matrix_elements : 0,
                  sizeof(*input.spin_density), &reads[15]) ||
      !make_range(input.spin_shell_scalar_potentials, has_spin ? batch.total_shells : 0,
                  sizeof(*input.spin_shell_scalar_potentials), &reads[16]) ||
      !make_range(output.overlap_adjoint, batch.total_matrix_elements,
                  sizeof(*output.overlap_adjoint), &writes[0]) ||
      !make_range(output.dipole_adjoint,
                  batch.total_matrix_elements * kGfn2HamiltonianDipoleComponents,
                  sizeof(*output.dipole_adjoint), &writes[1]) ||
      !make_range(output.quadrupole_adjoint,
                  batch.total_matrix_elements * kGfn2HamiltonianQuadrupoleComponents,
                  sizeof(*output.quadrupole_adjoint), &writes[2]) ||
      !make_range(workspace.overlap_adjoint_scratch, batch.total_matrix_elements,
                  sizeof(*workspace.overlap_adjoint_scratch), &writes[3]) ||
      !make_range(workspace.dipole_adjoint_scratch,
                  batch.total_matrix_elements * kGfn2HamiltonianDipoleComponents,
                  sizeof(*workspace.dipole_adjoint_scratch), &writes[4]) ||
      !make_range(workspace.quadrupole_adjoint_scratch,
                  batch.total_matrix_elements * kGfn2HamiltonianQuadrupoleComponents,
                  sizeof(*workspace.quadrupole_adjoint_scratch), &writes[5]) ||
      !make_range(workspace.sequence_active, 1, sizeof(*workspace.sequence_active), &writes[6]) ||
      !make_range(system_errors, batch.batch_size, sizeof(*system_errors), &writes[7]) ||
      !make_range(device_error, 1, sizeof(*device_error), &writes[8]) ||
      !writes_are_disjoint(reads, writes)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t check_launch() noexcept { return cudaPeekAtLastError(); }

}  // namespace

cudaError_t reset_gfn2_hamiltonian_force_device_errors_cuda(std::int64_t batch_size,
                                                            std::uint32_t* system_errors,
                                                            std::uint32_t* device_error,
                                                            cudaStream_t stream) noexcept {
  if (batch_size <= 0 ||
      static_cast<std::uint64_t>(batch_size) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() /
                                     sizeof(*system_errors)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  MemoryRange systems;
  MemoryRange device;
  if (!make_range(system_errors, batch_size, sizeof(*system_errors), &systems) ||
      !make_range(device_error, 1, sizeof(*device_error), &device) || overlaps(systems, device)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(
      system_errors, 0, static_cast<std::size_t>(batch_size) * sizeof(*system_errors), stream);
  return status == cudaSuccess ? cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream)
                               : status;
}

cudaError_t add_gfn2_hamiltonian_integral_adjoints_cuda(
    const Gfn2HamiltonianDeviceBatch& batch, const Gfn2ForceDeviceActivity& activity,
    const Gfn2HamiltonianForceDeviceInput& input, const Gfn2HamiltonianForceDeviceOutput& output,
    const Gfn2HamiltonianForceDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  cudaError_t status =
      validate_descriptors(batch, activity, input, output, workspace, system_errors, device_error);
  if (status != cudaSuccess) {
    return status;
  }
  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  const unsigned int blocks = static_cast<unsigned int>(batch.batch_size);
  preflight_and_seed_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, activity, input, output, workspace, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  contract_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(batch, activity, input, workspace,
                                                           system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(batch, activity, output, workspace,
                                                          system_errors);
  return check_launch();
}

}  // namespace gpuxtb::detail::cuda
