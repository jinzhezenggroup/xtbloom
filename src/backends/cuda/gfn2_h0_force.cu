#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/cuda_atomics.cuh"
#include "backends/cuda/gfn2_h0_force.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 128;
constexpr std::int64_t kMaximumInt64 = 9223372036854775807LL;
constexpr double kMinimumDistanceSquared = 1.0e-24;

struct SystemRanges {
  std::int64_t atom_begin;
  std::int64_t atom_end;
  std::int64_t shell_begin;
  std::int64_t shell_end;
  std::int64_t orbital_begin;
  std::int64_t orbital_end;
  std::int64_t matrix_begin;
  std::int64_t matrix_end;
  std::int64_t shell_pair_begin;
  std::int64_t shell_pair_end;
};

__device__ bool sequence_is_active(const Gfn2H0ForceDeviceWorkspace& workspace) {
  return atomicAdd(workspace.sequence_active, 0u) == 1u;
}

__device__ bool system_is_valid(const std::uint32_t* system_errors, std::int64_t system) {
  return atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) ==
         static_cast<std::uint32_t>(Gfn2H0ForceDeviceError::kSuccess);
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    std::uint32_t* device_error, Gfn2H0ForceDeviceError error) {
  const std::uint32_t success = static_cast<std::uint32_t>(Gfn2H0ForceDeviceError::kSuccess);
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

__device__ bool add_finite_atomic(double* target, double contribution) {
  if (!isfinite(contribution)) {
    return false;
  }
  const double previous = atomic_add_fp64(target, contribution);
  return isfinite(previous) && isfinite(previous + contribution);
}

__global__ void capture_sequence_kernel(const std::uint32_t* device_error,
                                        Gfn2H0ForceDeviceWorkspace workspace) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *workspace.sequence_active =
        atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                static_cast<std::uint32_t>(Gfn2H0ForceDeviceError::kSuccess)
            ? 1u
            : 0u;
  }
}

/*
 * Return true only for requested members whose terminal SCC status permits a
 * stationary force. The status is intentionally not read for mask==0.
 */
__device__ bool load_force_gate(const Gfn2ForceDeviceActivity& activity, std::int64_t system,
                                int* selected, std::uint32_t* system_errors,
                                std::uint32_t* device_error) {
  if (threadIdx.x == 0) {
    *selected = 0;
    const std::uint8_t requested = activity.requested_mask[system];
    if (requested > 1u) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kInvalidActiveMask);
    } else if (requested == 1u && activity.system_statuses[system] == GPUXTB_STATUS_SUCCESS) {
      *selected = 1;
    }
  }
  __syncthreads();
  return *selected != 0 && system_is_valid(system_errors, system);
}

__device__ bool load_ranges(const Gfn2IntegralDeviceBatch& batch, std::int64_t system,
                            SystemRanges* ranges) {
  ranges->atom_begin = batch.atom_offsets[system];
  ranges->atom_end = batch.atom_offsets[system + 1];
  ranges->shell_begin = batch.batch_shell_offsets[system];
  ranges->shell_end = batch.batch_shell_offsets[system + 1];
  ranges->orbital_begin = batch.batch_orbital_offsets[system];
  ranges->orbital_end = batch.batch_orbital_offsets[system + 1];
  ranges->matrix_begin = batch.matrix_offsets[system];
  ranges->matrix_end = batch.matrix_offsets[system + 1];
  ranges->shell_pair_begin = batch.shell_pair_offsets[system];
  ranges->shell_pair_end = batch.shell_pair_offsets[system + 1];
  if (!valid_range(ranges->atom_begin, ranges->atom_end, batch.total_atoms) ||
      !valid_range(ranges->shell_begin, ranges->shell_end, batch.total_shells) ||
      !valid_range(ranges->orbital_begin, ranges->orbital_end, batch.total_orbitals) ||
      !valid_range(ranges->matrix_begin, ranges->matrix_end, batch.total_matrix_elements) ||
      !valid_range(ranges->shell_pair_begin, ranges->shell_pair_end,
                   batch.total_shell_pair_elements)) {
    return false;
  }
  std::int64_t expected_matrix = 0;
  std::int64_t expected_pairs = 0;
  const std::int64_t orbitals = ranges->orbital_end - ranges->orbital_begin;
  const std::int64_t shells = ranges->shell_end - ranges->shell_begin;
  return checked_square(orbitals, &expected_matrix) && checked_square(shells, &expected_pairs) &&
         ranges->matrix_end - ranges->matrix_begin == expected_matrix &&
         ranges->shell_pair_end - ranges->shell_pair_begin == expected_pairs &&
         batch.atom_shell_offsets[ranges->atom_begin] == ranges->shell_begin &&
         batch.atom_shell_offsets[ranges->atom_end] == ranges->shell_end &&
         batch.shell_orbital_offsets[ranges->shell_begin] == ranges->orbital_begin &&
         batch.shell_orbital_offsets[ranges->shell_end] == ranges->orbital_end &&
         (system != 0 ||
          (ranges->atom_begin == 0 && ranges->shell_begin == 0 && ranges->orbital_begin == 0 &&
           ranges->matrix_begin == 0 && ranges->shell_pair_begin == 0)) &&
         (system + 1 != batch.batch_size ||
          (ranges->atom_end == batch.total_atoms && ranges->shell_end == batch.total_shells &&
           ranges->orbital_end == batch.total_orbitals &&
           ranges->matrix_end == batch.total_matrix_elements &&
           ranges->shell_pair_end == batch.total_shell_pair_elements));
}

__global__ void preflight_and_seed_kernel(Gfn2IntegralDeviceBatch batch, Gfn2H0DevicePlan h0_plan,
                                          Gfn2ForceDeviceActivity activity,
                                          Gfn2H0ForceDeviceInput input,
                                          Gfn2H0ForceDeviceOutput output,
                                          Gfn2H0ForceDeviceWorkspace workspace,
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
                          Gfn2H0ForceDeviceError::kInvalidOffsets);
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
    if (!valid_range(shell_begin, shell_end, batch.total_shells) ||
        shell_begin < ranges.shell_begin || shell_end > ranges.shell_end ||
        shell_begin == shell_end) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kInvalidOffsets);
      atomicExch(&valid, 0);
    }
    const double radius = h0_plan.atomic_radii[atom];
    const double coordination = input.coordination_numbers[atom];
    if (!(radius > 0.0) || !isfinite(radius)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kInvalidH0Parameter);
      atomicExch(&valid, 0);
    }
    if (!isfinite(coordination)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kNonfiniteInput);
      atomicExch(&valid, 0);
    }
    const std::int64_t coordinate = atom * 3;
    for (int axis = 0; axis < 3; ++axis) {
      if (!isfinite(input.positions[coordinate + axis])) {
        record_system_error(system_errors, system, device_error,
                            Gfn2H0ForceDeviceError::kNonfinitePosition);
        atomicExch(&valid, 0);
      }
      const double seed = output.gradients[coordinate + axis];
      if (!isfinite(seed)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2H0ForceDeviceError::kNonfiniteOutputSeed);
        atomicExch(&valid, 0);
      } else {
        workspace.gradient_scratch[coordinate + axis] = seed;
      }
    }
    const double cn_seed = output.coordination_adjoint[atom];
    if (!isfinite(cn_seed)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kNonfiniteOutputSeed);
      atomicExch(&valid, 0);
    } else {
      workspace.coordination_adjoint_scratch[atom] = cn_seed;
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
                          Gfn2H0ForceDeviceError::kInvalidShellMetadata);
      atomicExch(&valid, 0);
    }
    if (!isfinite(h0_plan.shell_levels[shell]) ||
        !isfinite(h0_plan.shell_coordination_scale[shell]) ||
        !isfinite(h0_plan.shell_polynomial[shell])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kInvalidH0Parameter);
      atomicExch(&valid, 0);
    }
  }

  for (std::int64_t pair = ranges.shell_pair_begin + threadIdx.x; pair < ranges.shell_pair_end;
       pair += blockDim.x) {
    const double scale = h0_plan.shell_pair_scale[pair];
    if (!(scale > 0.0) || !isfinite(scale)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kInvalidH0Parameter);
      atomicExch(&valid, 0);
    }
  }

  for (std::int64_t matrix = ranges.matrix_begin + threadIdx.x; matrix < ranges.matrix_end;
       matrix += blockDim.x) {
    const double overlap = input.overlap[matrix];
    const double density = input.density[matrix];
    const double weighted = input.energy_weighted_density[matrix];
    const double seed = output.overlap_adjoint[matrix];
    if (!isfinite(overlap) || !isfinite(density) || !isfinite(weighted)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kNonfiniteInput);
      atomicExch(&valid, 0);
    }
    const double pulay_seed = seed - weighted;
    if (!isfinite(seed)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kNonfiniteOutputSeed);
      atomicExch(&valid, 0);
    } else if (!isfinite(pulay_seed)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kNonfiniteArithmetic);
      atomicExch(&valid, 0);
    } else {
      workspace.overlap_adjoint_scratch[matrix] = pulay_seed;
    }
  }
}

__global__ void contract_h0_pulay_kernel(Gfn2IntegralDeviceBatch batch, Gfn2H0DevicePlan h0_plan,
                                         Gfn2ForceDeviceActivity activity,
                                         Gfn2H0ForceDeviceInput input,
                                         Gfn2H0ForceDeviceWorkspace workspace,
                                         std::uint32_t* system_errors,
                                         std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int selected;
  if (!sequence_is_active(workspace) ||
      !load_force_gate(activity, system, &selected, system_errors, device_error)) {
    return;
  }

  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t shell_count = shell_end - shell_begin;
  const std::int64_t orbital_begin = batch.batch_orbital_offsets[system];
  const std::int64_t orbital_count = batch.batch_orbital_offsets[system + 1] - orbital_begin;
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t pair_begin = batch.shell_pair_offsets[system];
  const std::int64_t pair_count = shell_count * shell_count;

  for (std::int64_t local_pair = threadIdx.x; local_pair < pair_count; local_pair += blockDim.x) {
    const std::int64_t first_shell = shell_begin + local_pair / shell_count;
    const std::int64_t second_shell = shell_begin + local_pair % shell_count;
    const std::int64_t first_atom = batch.shell_to_atom[first_shell];
    const std::int64_t second_atom = batch.shell_to_atom[second_shell];
    const double first_level =
        h0_plan.shell_levels[first_shell] -
        h0_plan.shell_coordination_scale[first_shell] * input.coordination_numbers[first_atom];
    const double second_level =
        h0_plan.shell_levels[second_shell] -
        h0_plan.shell_coordination_scale[second_shell] * input.coordination_numbers[second_atom];
    const double average_level = 0.5 * (first_level + second_level);
    double spatial_scale = 1.0;
    double spatial_scale_derivative = 0.0;
    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
    double distance = 0.0;
    bool finite = isfinite(first_level) && isfinite(second_level) && isfinite(average_level);

    if (finite && first_atom != second_atom) {
      dx = input.positions[first_atom * 3] - input.positions[second_atom * 3];
      dy = input.positions[first_atom * 3 + 1] - input.positions[second_atom * 3 + 1];
      dz = input.positions[first_atom * 3 + 2] - input.positions[second_atom * 3 + 2];
      if (!isfinite(dx) || !isfinite(dy) || !isfinite(dz)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2H0ForceDeviceError::kCoordinateDifferenceOverflow);
        finite = false;
      }
      const double distance_squared = dx * dx + dy * dy + dz * dz;
      if (finite && !isfinite(distance_squared)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2H0ForceDeviceError::kCoordinateDifferenceOverflow);
        finite = false;
      } else if (finite && distance_squared <= kMinimumDistanceSquared) {
        record_system_error(system_errors, system, device_error,
                            Gfn2H0ForceDeviceError::kCoincidentAtoms);
        finite = false;
      }
      if (finite) {
        distance = sqrt(distance_squared);
        const double radius_sum =
            h0_plan.atomic_radii[first_atom] + h0_plan.atomic_radii[second_atom];
        const double reduced_distance = sqrt(distance / radius_sum);
        const double first_polynomial =
            1.0 + h0_plan.shell_polynomial[first_shell] * reduced_distance;
        const double second_polynomial =
            1.0 + h0_plan.shell_polynomial[second_shell] * reduced_distance;
        const double pair_scale = h0_plan.shell_pair_scale[pair_begin + local_pair];
        spatial_scale = pair_scale * first_polynomial * second_polynomial;
        const double polynomial_derivative =
            (h0_plan.shell_polynomial[first_shell] * second_polynomial +
             h0_plan.shell_polynomial[second_shell] * first_polynomial) *
            reduced_distance / (2.0 * distance);
        spatial_scale_derivative = pair_scale * polynomial_derivative;
        finite = isfinite(radius_sum) && radius_sum > 0.0 && isfinite(distance) &&
                 isfinite(reduced_distance) && isfinite(first_polynomial) &&
                 isfinite(second_polynomial) && isfinite(spatial_scale) &&
                 isfinite(spatial_scale_derivative);
      }
    }

    const double factor = average_level * spatial_scale;
    if (!finite || !isfinite(factor)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kNonfiniteArithmetic);
      continue;
    }

    double block_weight = 0.0;
    const std::int64_t first_orbital_begin = batch.shell_orbital_offsets[first_shell];
    const std::int64_t first_orbital_end = batch.shell_orbital_offsets[first_shell + 1];
    const std::int64_t second_orbital_begin = batch.shell_orbital_offsets[second_shell];
    const std::int64_t second_orbital_end = batch.shell_orbital_offsets[second_shell + 1];
    for (std::int64_t first_orbital = first_orbital_begin;
         finite && first_orbital < first_orbital_end; ++first_orbital) {
      const std::int64_t row = first_orbital - orbital_begin;
      for (std::int64_t second_orbital = second_orbital_begin; second_orbital < second_orbital_end;
           ++second_orbital) {
        const std::int64_t column = second_orbital - orbital_begin;
        const std::int64_t matrix = matrix_begin + row * orbital_count + column;
        const double density = input.density[matrix];
        const double overlap = input.overlap[matrix];
        const double overlap_contribution = density * factor;
        const double overlap_updated =
            workspace.overlap_adjoint_scratch[matrix] + overlap_contribution;
        const double weight_contribution = density * overlap;
        const double weight_updated = block_weight + weight_contribution;
        if (!isfinite(overlap_contribution) || !isfinite(overlap_updated) ||
            !isfinite(weight_contribution) || !isfinite(weight_updated)) {
          record_system_error(system_errors, system, device_error,
                              Gfn2H0ForceDeviceError::kNonfiniteArithmetic);
          finite = false;
          break;
        }
        workspace.overlap_adjoint_scratch[matrix] = overlap_updated;
        block_weight = weight_updated;
      }
    }
    if (!finite) {
      continue;
    }

    const double level_weight = 0.5 * block_weight * spatial_scale;
    const double first_cn = -h0_plan.shell_coordination_scale[first_shell] * level_weight;
    const double second_cn = -h0_plan.shell_coordination_scale[second_shell] * level_weight;
    if (!isfinite(level_weight) ||
        !add_finite_atomic(workspace.coordination_adjoint_scratch + first_atom, first_cn) ||
        !add_finite_atomic(workspace.coordination_adjoint_scratch + second_atom, second_cn)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kNonfiniteArithmetic);
      continue;
    }

    if (first_atom != second_atom) {
      const double radial_derivative = block_weight * average_level * spatial_scale_derivative;
      const double coordinate_scale = radial_derivative / distance;
      const double contribution[3] = {coordinate_scale * dx, coordinate_scale * dy,
                                      coordinate_scale * dz};
      for (int axis = 0; axis < 3; ++axis) {
        if (!add_finite_atomic(workspace.gradient_scratch + first_atom * 3 + axis,
                               contribution[axis]) ||
            !add_finite_atomic(workspace.gradient_scratch + second_atom * 3 + axis,
                               -contribution[axis])) {
          record_system_error(system_errors, system, device_error,
                              Gfn2H0ForceDeviceError::kNonfiniteArithmetic);
        }
      }
    }
  }
}

__global__ void publish_kernel(Gfn2IntegralDeviceBatch batch, Gfn2ForceDeviceActivity activity,
                               Gfn2H0ForceDeviceOutput output, Gfn2H0ForceDeviceWorkspace workspace,
                               const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!sequence_is_active(workspace) || activity.requested_mask[system] != 1u ||
      activity.system_statuses[system] != GPUXTB_STATUS_SUCCESS ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t matrix_end = batch.matrix_offsets[system + 1];
  for (std::int64_t matrix = matrix_begin + threadIdx.x; matrix < matrix_end;
       matrix += blockDim.x) {
    output.overlap_adjoint[matrix] = workspace.overlap_adjoint_scratch[matrix];
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    output.coordination_adjoint[atom] = workspace.coordination_adjoint_scratch[atom];
    const std::int64_t coordinate = atom * 3;
    output.gradients[coordinate] = workspace.gradient_scratch[coordinate];
    output.gradients[coordinate + 1] = workspace.gradient_scratch[coordinate + 1];
    output.gradients[coordinate + 2] = workspace.gradient_scratch[coordinate + 2];
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

cudaError_t validate_descriptors(
    const Gfn2IntegralDeviceBatch& batch, const Gfn2H0DevicePlan& h0_plan,
    const Gfn2ForceDeviceActivity& activity, const Gfn2H0ForceDeviceInput& input,
    const Gfn2H0ForceDeviceOutput& output, const Gfn2H0ForceDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error) noexcept {
  if (batch.batch_size <= 0 || batch.batch_size > std::numeric_limits<int>::max() ||
      batch.total_atoms < 0 || batch.total_shells < 0 || batch.total_orbitals < 0 ||
      batch.total_matrix_elements < 0 || batch.total_shell_pair_elements < 0 ||
      batch.total_atoms > kMaximumInt64 / 3 || batch.plan_token == 0u ||
      batch.atom_offset_count != batch.batch_size + 1 ||
      batch.batch_shell_offset_count != batch.batch_size + 1 ||
      batch.batch_orbital_offset_count != batch.batch_size + 1 ||
      batch.matrix_offset_count != batch.batch_size + 1 ||
      batch.shell_pair_offset_count != batch.batch_size + 1 ||
      batch.atom_shell_offset_count < batch.total_atoms + 1 ||
      batch.shell_orbital_offset_count < batch.total_shells + 1 ||
      batch.shell_to_atom_count < batch.total_shells ||
      h0_plan.atomic_radius_count < batch.total_atoms ||
      h0_plan.shell_level_count < batch.total_shells ||
      h0_plan.shell_coordination_scale_count < batch.total_shells ||
      h0_plan.shell_polynomial_count < batch.total_shells ||
      h0_plan.shell_pair_scale_count < batch.total_shell_pair_elements ||
      h0_plan.plan_token != batch.plan_token || activity.plan_token != batch.plan_token ||
      input.plan_token != batch.plan_token || output.plan_token != batch.plan_token ||
      workspace.plan_token != batch.plan_token || activity.batch_elements != batch.batch_size ||
      input.position_elements < batch.total_atoms * 3 ||
      input.coordination_elements < batch.total_atoms ||
      input.overlap_elements < batch.total_matrix_elements ||
      input.density_elements < batch.total_matrix_elements ||
      input.energy_weighted_density_elements < batch.total_matrix_elements ||
      output.overlap_adjoint_elements < batch.total_matrix_elements ||
      output.coordination_adjoint_elements < batch.total_atoms ||
      output.gradient_elements < batch.total_atoms * 3 ||
      workspace.overlap_adjoint_elements < batch.total_matrix_elements ||
      workspace.coordination_adjoint_elements < batch.total_atoms ||
      workspace.gradient_elements < batch.total_atoms * 3 || workspace.sequence_elements < 1 ||
      !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.batch_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.batch_orbital_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.matrix_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_pair_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.atom_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_orbital_offsets, alignof(std::int64_t)) ||
      !required_pointer(batch.shell_to_atom, batch.total_shells) ||
      !required_pointer(h0_plan.atomic_radii, batch.total_atoms) ||
      !required_pointer(h0_plan.shell_levels, batch.total_shells) ||
      !required_pointer(h0_plan.shell_coordination_scale, batch.total_shells) ||
      !required_pointer(h0_plan.shell_polynomial, batch.total_shells) ||
      !required_pointer(h0_plan.shell_pair_scale, batch.total_shell_pair_elements) ||
      !is_aligned(activity.requested_mask, alignof(std::uint8_t)) ||
      !is_aligned(activity.system_statuses, alignof(gpuxtb_status_t)) ||
      !required_pointer(input.positions, batch.total_atoms * 3) ||
      !required_pointer(input.coordination_numbers, batch.total_atoms) ||
      !required_pointer(input.overlap, batch.total_matrix_elements) ||
      !required_pointer(input.density, batch.total_matrix_elements) ||
      !required_pointer(input.energy_weighted_density, batch.total_matrix_elements) ||
      !required_pointer(output.overlap_adjoint, batch.total_matrix_elements) ||
      !required_pointer(output.coordination_adjoint, batch.total_atoms) ||
      !required_pointer(output.gradients, batch.total_atoms * 3) ||
      !required_pointer(workspace.overlap_adjoint_scratch, batch.total_matrix_elements) ||
      !required_pointer(workspace.coordination_adjoint_scratch, batch.total_atoms) ||
      !required_pointer(workspace.gradient_scratch, batch.total_atoms * 3) ||
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return batch.batch_size > std::numeric_limits<int>::max() ? cudaErrorInvalidConfiguration
                                                              : cudaErrorInvalidValue;
  }

  std::array<MemoryRange, 20> reads;
  std::array<MemoryRange, 9> writes;
  if (!make_range(batch.atom_offsets, batch.atom_offset_count, sizeof(*batch.atom_offsets),
                  &reads[0]) ||
      !make_range(batch.batch_shell_offsets, batch.batch_shell_offset_count,
                  sizeof(*batch.batch_shell_offsets), &reads[1]) ||
      !make_range(batch.batch_orbital_offsets, batch.batch_orbital_offset_count,
                  sizeof(*batch.batch_orbital_offsets), &reads[2]) ||
      !make_range(batch.matrix_offsets, batch.matrix_offset_count, sizeof(*batch.matrix_offsets),
                  &reads[3]) ||
      !make_range(batch.shell_pair_offsets, batch.shell_pair_offset_count,
                  sizeof(*batch.shell_pair_offsets), &reads[4]) ||
      !make_range(batch.atom_shell_offsets, batch.atom_shell_offset_count,
                  sizeof(*batch.atom_shell_offsets), &reads[5]) ||
      !make_range(batch.shell_orbital_offsets, batch.shell_orbital_offset_count,
                  sizeof(*batch.shell_orbital_offsets), &reads[6]) ||
      !make_range(batch.shell_to_atom, batch.total_shells, sizeof(*batch.shell_to_atom),
                  &reads[7]) ||
      !make_range(h0_plan.atomic_radii, batch.total_atoms, sizeof(*h0_plan.atomic_radii),
                  &reads[8]) ||
      !make_range(h0_plan.shell_levels, batch.total_shells, sizeof(*h0_plan.shell_levels),
                  &reads[9]) ||
      !make_range(h0_plan.shell_coordination_scale, batch.total_shells,
                  sizeof(*h0_plan.shell_coordination_scale), &reads[10]) ||
      !make_range(h0_plan.shell_polynomial, batch.total_shells, sizeof(*h0_plan.shell_polynomial),
                  &reads[11]) ||
      !make_range(h0_plan.shell_pair_scale, batch.total_shell_pair_elements,
                  sizeof(*h0_plan.shell_pair_scale), &reads[12]) ||
      !make_range(activity.requested_mask, batch.batch_size, sizeof(*activity.requested_mask),
                  &reads[13]) ||
      !make_range(activity.system_statuses, batch.batch_size, sizeof(*activity.system_statuses),
                  &reads[14]) ||
      !make_range(input.positions, batch.total_atoms * 3, sizeof(*input.positions), &reads[15]) ||
      !make_range(input.coordination_numbers, batch.total_atoms,
                  sizeof(*input.coordination_numbers), &reads[16]) ||
      !make_range(input.overlap, batch.total_matrix_elements, sizeof(*input.overlap), &reads[17]) ||
      !make_range(input.density, batch.total_matrix_elements, sizeof(*input.density), &reads[18]) ||
      !make_range(input.energy_weighted_density, batch.total_matrix_elements,
                  sizeof(*input.energy_weighted_density), &reads[19]) ||
      !make_range(output.overlap_adjoint, batch.total_matrix_elements,
                  sizeof(*output.overlap_adjoint), &writes[0]) ||
      !make_range(output.coordination_adjoint, batch.total_atoms,
                  sizeof(*output.coordination_adjoint), &writes[1]) ||
      !make_range(output.gradients, batch.total_atoms * 3, sizeof(*output.gradients), &writes[2]) ||
      !make_range(workspace.overlap_adjoint_scratch, batch.total_matrix_elements,
                  sizeof(*workspace.overlap_adjoint_scratch), &writes[3]) ||
      !make_range(workspace.coordination_adjoint_scratch, batch.total_atoms,
                  sizeof(*workspace.coordination_adjoint_scratch), &writes[4]) ||
      !make_range(workspace.gradient_scratch, batch.total_atoms * 3,
                  sizeof(*workspace.gradient_scratch), &writes[5]) ||
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

cudaError_t reset_gfn2_h0_force_device_errors_cuda(std::int64_t batch_size,
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

cudaError_t add_gfn2_h0_pulay_gradient_cuda(
    const Gfn2IntegralDeviceBatch& batch, const Gfn2H0DevicePlan& h0_plan,
    const Gfn2ForceDeviceActivity& activity, const Gfn2H0ForceDeviceInput& input,
    const Gfn2H0ForceDeviceOutput& output, const Gfn2H0ForceDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  cudaError_t status = validate_descriptors(batch, h0_plan, activity, input, output, workspace,
                                            system_errors, device_error);
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
      batch, h0_plan, activity, input, output, workspace, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  contract_h0_pulay_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, h0_plan, activity, input, workspace, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(batch, activity, output, workspace,
                                                          system_errors);
  return check_launch();
}

}  // namespace gpuxtb::detail::cuda
