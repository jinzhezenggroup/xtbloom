#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_hamiltonian.cuh"

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

__device__ bool sequence_is_active(const Gfn2HamiltonianDeviceWorkspace& workspace) {
  return atomicAdd(workspace.sequence_active, 0u) == 1u;
}

__device__ bool system_is_valid(const std::uint32_t* system_errors, std::int64_t system) {
  return atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) ==
         static_cast<std::uint32_t>(Gfn2HamiltonianDeviceError::kSuccess);
}

__device__ bool system_is_active(const Gfn2HamiltonianDeviceActivity& activity,
                                 std::int64_t system) {
  return activity.active_mask[system] == 1u;
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    std::uint32_t* device_error, Gfn2HamiltonianDeviceError error) {
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  if (atomicCAS(system_errors + system,
                static_cast<std::uint32_t>(Gfn2HamiltonianDeviceError::kSuccess),
                code) == static_cast<std::uint32_t>(Gfn2HamiltonianDeviceError::kSuccess)) {
    atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2HamiltonianDeviceError::kSuccess), code);
  }
}

__device__ bool valid_closed_range(std::int64_t begin, std::int64_t end, std::int64_t total) {
  return begin >= 0 && end >= 0 && begin <= end && end <= total;
}

__device__ bool checked_square(std::int64_t value, std::int64_t* square) {
  if (value < 0 || (value != 0 && value > kMaximumInt64 / value)) {
    return false;
  }
  *square = value * value;
  return true;
}

__device__ bool add_product(double left, double right, double* accumulator) {
  const double updated = fma(left, right, *accumulator);
  if (!isfinite(updated)) {
    return false;
  }
  *accumulator = updated;
  return true;
}

__global__ void capture_sequence_kernel(const std::uint32_t* device_error,
                                        Gfn2HamiltonianDeviceWorkspace workspace) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *workspace.sequence_active =
        atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                static_cast<std::uint32_t>(Gfn2HamiltonianDeviceError::kSuccess)
            ? 1u
            : 0u;
  }
}

__global__ void topology_preflight_kernel(Gfn2HamiltonianDeviceBatch batch,
                                          Gfn2HamiltonianDeviceActivity activity,
                                          Gfn2HamiltonianDeviceWorkspace workspace,
                                          std::uint32_t* system_errors,
                                          std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!sequence_is_active(workspace) || !system_is_valid(system_errors, system)) {
    return;
  }
  const std::uint8_t active = activity.active_mask[system];
  if (active == 0u) {
    return;
  }
  if (active != 1u) {
    if (threadIdx.x == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2HamiltonianDeviceError::kInvalidActiveMask);
    }
    return;
  }

  __shared__ SystemRanges ranges;
  __shared__ int valid;
  if (threadIdx.x == 0) {
    valid = 1;
    ranges.atom_begin = batch.atom_offsets[system];
    ranges.atom_end = batch.atom_offsets[system + 1];
    ranges.shell_begin = batch.batch_shell_offsets[system];
    ranges.shell_end = batch.batch_shell_offsets[system + 1];
    ranges.orbital_begin = batch.batch_orbital_offsets[system];
    ranges.orbital_end = batch.batch_orbital_offsets[system + 1];
    ranges.matrix_begin = batch.matrix_offsets[system];
    ranges.matrix_end = batch.matrix_offsets[system + 1];
    if (!valid_closed_range(ranges.atom_begin, ranges.atom_end, batch.total_atoms) ||
        !valid_closed_range(ranges.shell_begin, ranges.shell_end, batch.total_shells) ||
        !valid_closed_range(ranges.orbital_begin, ranges.orbital_end, batch.total_orbitals) ||
        !valid_closed_range(ranges.matrix_begin, ranges.matrix_end, batch.total_matrix_elements)) {
      valid = 0;
    }
    if (valid != 0) {
      const std::int64_t atoms = ranges.atom_end - ranges.atom_begin;
      const std::int64_t shells = ranges.shell_end - ranges.shell_begin;
      const std::int64_t orbitals = ranges.orbital_end - ranges.orbital_begin;
      std::int64_t expected_matrix = 0;
      valid = atoms > 0 && shells > 0 && orbitals > 0 &&
              checked_square(orbitals, &expected_matrix) &&
              ranges.matrix_end - ranges.matrix_begin == expected_matrix;
    }
    if (valid != 0) {
      valid = batch.atom_shell_offsets[ranges.atom_begin] == ranges.shell_begin &&
              batch.atom_shell_offsets[ranges.atom_end] == ranges.shell_end &&
              batch.shell_orbital_offsets[ranges.shell_begin] == ranges.orbital_begin &&
              batch.shell_orbital_offsets[ranges.shell_end] == ranges.orbital_end;
    }
    if (valid != 0 && system == 0) {
      valid = ranges.atom_begin == 0 && ranges.shell_begin == 0 && ranges.orbital_begin == 0 &&
              ranges.matrix_begin == 0;
    }
    if (valid != 0 && system + 1 == batch.batch_size) {
      valid = ranges.atom_end == batch.total_atoms && ranges.shell_end == batch.total_shells &&
              ranges.orbital_end == batch.total_orbitals &&
              ranges.matrix_end == batch.total_matrix_elements;
    }
    if (valid == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2HamiltonianDeviceError::kInvalidOffsets);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    const std::int64_t shell_begin = batch.atom_shell_offsets[atom];
    const std::int64_t shell_end = batch.atom_shell_offsets[atom + 1];
    if (!valid_closed_range(shell_begin, shell_end, batch.total_shells) ||
        shell_begin < ranges.shell_begin || shell_end > ranges.shell_end ||
        shell_begin == shell_end) {
      record_system_error(system_errors, system, device_error,
                          Gfn2HamiltonianDeviceError::kInvalidOffsets);
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t shell = ranges.shell_begin + threadIdx.x; shell < ranges.shell_end;
       shell += blockDim.x) {
    const std::int64_t atom = batch.shell_to_atom[shell];
    const std::int64_t orbital_begin = batch.shell_orbital_offsets[shell];
    const std::int64_t orbital_end = batch.shell_orbital_offsets[shell + 1];
    bool shell_valid = atom >= ranges.atom_begin && atom < ranges.atom_end;
    if (shell_valid) {
      shell_valid =
          shell >= batch.atom_shell_offsets[atom] && shell < batch.atom_shell_offsets[atom + 1];
    }
    shell_valid = shell_valid &&
                  valid_closed_range(orbital_begin, orbital_end, batch.total_orbitals) &&
                  orbital_begin >= ranges.orbital_begin && orbital_end <= ranges.orbital_end &&
                  orbital_begin < orbital_end;
    if (!shell_valid) {
      record_system_error(system_errors, system, device_error,
                          Gfn2HamiltonianDeviceError::kInvalidOrbitalMetadata);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  for (std::int64_t orbital = ranges.orbital_begin + threadIdx.x; orbital < ranges.orbital_end;
       orbital += blockDim.x) {
    const std::int64_t shell = batch.orbital_to_shell[orbital];
    const std::int64_t atom = batch.orbital_to_atom[orbital];
    const bool orbital_valid =
        shell >= ranges.shell_begin && shell < ranges.shell_end && atom >= ranges.atom_begin &&
        atom < ranges.atom_end && orbital >= batch.shell_orbital_offsets[shell] &&
        orbital < batch.shell_orbital_offsets[shell + 1] && batch.shell_to_atom[shell] == atom;
    if (!orbital_valid) {
      record_system_error(system_errors, system, device_error,
                          Gfn2HamiltonianDeviceError::kInvalidOrbitalMetadata);
      atomicExch(&valid, 0);
    }
  }
}

__global__ void assemble_hamiltonian_kernel(Gfn2HamiltonianDeviceBatch batch,
                                            Gfn2HamiltonianDeviceInput input,
                                            Gfn2HamiltonianDeviceActivity activity,
                                            Gfn2HamiltonianDeviceWorkspace workspace,
                                            std::uint32_t* system_errors,
                                            std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!sequence_is_active(workspace) || !system_is_active(activity, system) ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t orbital_begin = batch.batch_orbital_offsets[system];
  const std::int64_t orbitals = batch.batch_orbital_offsets[system + 1] - orbital_begin;
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t matrix_elements = orbitals * orbitals;

  for (std::int64_t local = threadIdx.x; local < matrix_elements; local += blockDim.x) {
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

    const double forward_h0 = input.h0[forward];
    const double reverse_h0 = input.h0[reverse];
    if (!isfinite(forward_h0) || !isfinite(reverse_h0)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2HamiltonianDeviceError::kNonfiniteH0);
      continue;
    }
    const double overlap = input.overlap[forward];
    const double reverse_overlap = input.overlap[reverse];
    if (!isfinite(overlap) || !isfinite(reverse_overlap)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2HamiltonianDeviceError::kNonfiniteOverlap);
      continue;
    }
    const double row_scalar = input.shell_scalar_potentials[row_shell];
    const double column_scalar = input.shell_scalar_potentials[column_shell];
    if (!isfinite(row_scalar) || !isfinite(column_scalar)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2HamiltonianDeviceError::kNonfinitePotential);
      continue;
    }

    double shift = 0.0;
    const double half_overlap = -0.5 * overlap;
    bool finite = add_product(half_overlap, row_scalar, &shift) &&
                  add_product(half_overlap, column_scalar, &shift);
    for (int component = 0; component < kGfn2HamiltonianDipoleComponents && finite; ++component) {
      const double row_potential =
          input.atomic_dipole_potentials[row_atom * kGfn2HamiltonianDipoleComponents + component];
      const double column_potential =
          input
              .atomic_dipole_potentials[column_atom * kGfn2HamiltonianDipoleComponents + component];
      const double forward_integral =
          -0.5 * input.dipole_integrals[component * batch.total_matrix_elements + forward];
      const double reverse_integral =
          -0.5 * input.dipole_integrals[component * batch.total_matrix_elements + reverse];
      if (!isfinite(forward_integral) || !isfinite(reverse_integral)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2HamiltonianDeviceError::kNonfiniteMultipoleIntegral);
        finite = false;
      } else if (!isfinite(row_potential) || !isfinite(column_potential)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2HamiltonianDeviceError::kNonfinitePotential);
        finite = false;
      } else {
        finite = add_product(forward_integral, column_potential, &shift) &&
                 add_product(reverse_integral, row_potential, &shift);
      }
    }
    for (int component = 0; component < kGfn2HamiltonianQuadrupoleComponents && finite;
         ++component) {
      const double row_potential =
          input.atomic_quadrupole_potentials[row_atom * kGfn2HamiltonianQuadrupoleComponents +
                                             component];
      const double column_potential =
          input.atomic_quadrupole_potentials[column_atom * kGfn2HamiltonianQuadrupoleComponents +
                                             component];
      const double forward_integral =
          -0.5 * input.quadrupole_integrals[component * batch.total_matrix_elements + forward];
      const double reverse_integral =
          -0.5 * input.quadrupole_integrals[component * batch.total_matrix_elements + reverse];
      if (!isfinite(forward_integral) || !isfinite(reverse_integral)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2HamiltonianDeviceError::kNonfiniteMultipoleIntegral);
        finite = false;
      } else if (!isfinite(row_potential) || !isfinite(column_potential)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2HamiltonianDeviceError::kNonfinitePotential);
        finite = false;
      } else {
        finite = add_product(forward_integral, column_potential, &shift) &&
                 add_product(reverse_integral, row_potential, &shift);
      }
    }
    if (!finite) {
      if (system_is_valid(system_errors, system)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2HamiltonianDeviceError::kNonfiniteAssemblyArithmetic);
      }
      continue;
    }
    const double forward_value = forward_h0 + shift;
    const double reverse_value = reverse_h0 + shift;
    if (!isfinite(forward_value) || !isfinite(reverse_value)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2HamiltonianDeviceError::kNonfiniteAssemblyArithmetic);
      continue;
    }
    workspace.matrix_scratch[forward] = forward_value;
    if (forward != reverse) {
      workspace.matrix_scratch[reverse] = reverse_value;
    }
  }
}

__global__ void publish_hamiltonian_kernel(Gfn2HamiltonianDeviceBatch batch,
                                           Gfn2HamiltonianDeviceActivity activity,
                                           Gfn2HamiltonianDeviceOutput output,
                                           Gfn2HamiltonianDeviceWorkspace workspace,
                                           const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!sequence_is_active(workspace) || !system_is_active(activity, system) ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t begin = batch.matrix_offsets[system];
  const std::int64_t end = batch.matrix_offsets[system + 1];
  for (std::int64_t element = begin + threadIdx.x; element < end; element += blockDim.x) {
    output.matrix[element] = workspace.matrix_scratch[element];
  }
}

bool is_aligned(const void* pointer, std::size_t alignment) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
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

template <std::size_t Count>
bool pairwise_disjoint(const std::array<MemoryRange, Count>& ranges) noexcept {
  for (std::size_t first = 0; first < Count; ++first) {
    for (std::size_t second = first + 1u; second < Count; ++second) {
      if (overlaps(ranges[first], ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

cudaError_t check_launch() noexcept { return cudaPeekAtLastError(); }

}  // namespace

cudaError_t reset_gfn2_hamiltonian_device_errors_cuda(std::int64_t batch_size,
                                                      std::uint32_t* system_errors,
                                                      std::uint32_t* device_error,
                                                      cudaStream_t stream) noexcept {
  if (batch_size <= 0 || !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t)) ||
      static_cast<std::uint64_t>(batch_size) >
          std::numeric_limits<std::size_t>::max() / sizeof(*system_errors)) {
    return cudaErrorInvalidValue;
  }
  MemoryRange systems;
  MemoryRange diagnostic;
  if (!make_range(system_errors, batch_size, sizeof(*system_errors), &systems) ||
      !make_range(device_error, 1, sizeof(*device_error), &diagnostic) ||
      overlaps(systems, diagnostic)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(
      system_errors, 0, static_cast<std::size_t>(batch_size) * sizeof(*system_errors), stream);
  return status == cudaSuccess ? cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream)
                               : status;
}

cudaError_t assemble_gfn2_hamiltonian_cuda(
    const Gfn2HamiltonianDeviceBatch& batch, const Gfn2HamiltonianDeviceInput& input,
    const Gfn2HamiltonianDeviceActivity& activity, const Gfn2HamiltonianDeviceOutput& output,
    const Gfn2HamiltonianDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (batch.batch_size <= 0 || batch.total_atoms <= 0 || batch.total_shells <= 0 ||
      batch.total_orbitals <= 0 || batch.total_matrix_elements <= 0 ||
      batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      batch.total_atoms > kMaximumInt64 / kGfn2HamiltonianQuadrupoleComponents ||
      batch.total_shells == kMaximumInt64 ||
      batch.total_matrix_elements > kMaximumInt64 / kGfn2HamiltonianQuadrupoleComponents ||
      batch.plan_token == 0u || input.plan_token != batch.plan_token ||
      activity.plan_token != batch.plan_token || output.plan_token != batch.plan_token ||
      workspace.plan_token != batch.plan_token || batch.atom_offset_count != batch.batch_size + 1 ||
      batch.batch_shell_offset_count != batch.batch_size + 1 ||
      batch.batch_orbital_offset_count != batch.batch_size + 1 ||
      batch.matrix_offset_count != batch.batch_size + 1 ||
      batch.atom_shell_offset_count != batch.total_atoms + 1 ||
      batch.shell_orbital_offset_count != batch.total_shells + 1 ||
      batch.shell_to_atom_count != batch.total_shells ||
      batch.orbital_to_shell_count != batch.total_orbitals ||
      batch.orbital_to_atom_count != batch.total_orbitals ||
      input.h0_elements != batch.total_matrix_elements ||
      input.overlap_elements != batch.total_matrix_elements ||
      input.dipole_integral_elements !=
          batch.total_matrix_elements * kGfn2HamiltonianDipoleComponents ||
      input.quadrupole_integral_elements !=
          batch.total_matrix_elements * kGfn2HamiltonianQuadrupoleComponents ||
      input.shell_scalar_elements != batch.total_shells ||
      input.atomic_dipole_elements != batch.total_atoms * kGfn2HamiltonianDipoleComponents ||
      input.atomic_quadrupole_elements !=
          batch.total_atoms * kGfn2HamiltonianQuadrupoleComponents ||
      activity.elements != batch.batch_size || output.elements != batch.total_matrix_elements ||
      workspace.matrix_elements < batch.total_matrix_elements || workspace.sequence_elements < 1 ||
      !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.batch_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.batch_orbital_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.matrix_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.atom_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_orbital_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_to_atom, alignof(std::int64_t)) ||
      !is_aligned(batch.orbital_to_shell, alignof(std::int64_t)) ||
      !is_aligned(batch.orbital_to_atom, alignof(std::int64_t)) ||
      !is_aligned(input.h0, alignof(double)) || !is_aligned(input.overlap, alignof(double)) ||
      !is_aligned(input.dipole_integrals, alignof(double)) ||
      !is_aligned(input.quadrupole_integrals, alignof(double)) ||
      !is_aligned(input.shell_scalar_potentials, alignof(double)) ||
      !is_aligned(input.atomic_dipole_potentials, alignof(double)) ||
      !is_aligned(input.atomic_quadrupole_potentials, alignof(double)) ||
      !is_aligned(activity.active_mask, alignof(std::uint8_t)) ||
      !is_aligned(output.matrix, alignof(double)) ||
      !is_aligned(workspace.matrix_scratch, alignof(double)) ||
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max())
               ? cudaErrorInvalidConfiguration
               : cudaErrorInvalidValue;
  }

  std::array<MemoryRange, 22> ranges;
  if (!make_range(batch.atom_offsets, batch.atom_offset_count, sizeof(*batch.atom_offsets),
                  &ranges[0]) ||
      !make_range(batch.batch_shell_offsets, batch.batch_shell_offset_count,
                  sizeof(*batch.batch_shell_offsets), &ranges[1]) ||
      !make_range(batch.batch_orbital_offsets, batch.batch_orbital_offset_count,
                  sizeof(*batch.batch_orbital_offsets), &ranges[2]) ||
      !make_range(batch.matrix_offsets, batch.matrix_offset_count, sizeof(*batch.matrix_offsets),
                  &ranges[3]) ||
      !make_range(batch.atom_shell_offsets, batch.atom_shell_offset_count,
                  sizeof(*batch.atom_shell_offsets), &ranges[4]) ||
      !make_range(batch.shell_orbital_offsets, batch.shell_orbital_offset_count,
                  sizeof(*batch.shell_orbital_offsets), &ranges[5]) ||
      !make_range(batch.shell_to_atom, batch.shell_to_atom_count, sizeof(*batch.shell_to_atom),
                  &ranges[6]) ||
      !make_range(batch.orbital_to_shell, batch.orbital_to_shell_count,
                  sizeof(*batch.orbital_to_shell), &ranges[7]) ||
      !make_range(batch.orbital_to_atom, batch.orbital_to_atom_count,
                  sizeof(*batch.orbital_to_atom), &ranges[8]) ||
      !make_range(input.h0, input.h0_elements, sizeof(*input.h0), &ranges[9]) ||
      !make_range(input.overlap, input.overlap_elements, sizeof(*input.overlap), &ranges[10]) ||
      !make_range(input.dipole_integrals, input.dipole_integral_elements,
                  sizeof(*input.dipole_integrals), &ranges[11]) ||
      !make_range(input.quadrupole_integrals, input.quadrupole_integral_elements,
                  sizeof(*input.quadrupole_integrals), &ranges[12]) ||
      !make_range(input.shell_scalar_potentials, input.shell_scalar_elements,
                  sizeof(*input.shell_scalar_potentials), &ranges[13]) ||
      !make_range(input.atomic_dipole_potentials, input.atomic_dipole_elements,
                  sizeof(*input.atomic_dipole_potentials), &ranges[14]) ||
      !make_range(input.atomic_quadrupole_potentials, input.atomic_quadrupole_elements,
                  sizeof(*input.atomic_quadrupole_potentials), &ranges[15]) ||
      !make_range(activity.active_mask, activity.elements, sizeof(*activity.active_mask),
                  &ranges[16]) ||
      !make_range(output.matrix, output.elements, sizeof(*output.matrix), &ranges[17]) ||
      !make_range(workspace.matrix_scratch, workspace.matrix_elements,
                  sizeof(*workspace.matrix_scratch), &ranges[18]) ||
      !make_range(workspace.sequence_active, 1, sizeof(*workspace.sequence_active), &ranges[19]) ||
      !make_range(system_errors, batch.batch_size, sizeof(*system_errors), &ranges[20]) ||
      !make_range(device_error, 1, sizeof(*device_error), &ranges[21])) {
    return cudaErrorInvalidValue;
  }
  if (!pairwise_disjoint(ranges)) {
    return cudaErrorInvalidValue;
  }

  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace);
  cudaError_t status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  topology_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                              stream>>>(batch, activity, workspace, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  assemble_hamiltonian_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                                stream>>>(batch, input, activity, workspace, system_errors,
                                          device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_hamiltonian_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                               stream>>>(batch, activity, output, workspace, system_errors);
  return check_launch();
}

}  // namespace gpuxtb::detail::cuda
