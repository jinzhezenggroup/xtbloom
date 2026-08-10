#include <cmath>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/cuda_atomics.cuh"
#include "backends/cuda/gfn2_es2.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr std::int64_t kInt64Maximum = 9223372036854775807LL;

__device__ void record_error(std::uint32_t* device_error, Gfn2ES2DeviceError error) {
  atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2ES2DeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ bool sequence_is_valid(const std::uint32_t* device_error) {
  return atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
         static_cast<std::uint32_t>(Gfn2ES2DeviceError::kSuccess);
}

struct SystemRanges {
  std::int64_t atom_begin;
  std::int64_t atom_end;
  std::int64_t shell_begin;
  std::int64_t shell_end;
  std::int64_t matrix_begin;
  std::int64_t matrix_end;
};

/* Validate every offset before it can participate in later pointer indexing. */
__device__ void load_and_validate_system(const Gfn2ES2DeviceBatch& batch, std::int64_t system,
                                         SystemRanges* ranges, int* valid,
                                         std::uint32_t* device_error) {
  if (threadIdx.x == 0) {
    if (!sequence_is_valid(device_error)) {
      *valid = 0;
    } else {
      ranges->atom_begin = batch.atom_offsets[system];
      ranges->atom_end = batch.atom_offsets[system + 1];
      ranges->shell_begin = batch.batch_shell_offsets[system];
      ranges->shell_end = batch.batch_shell_offsets[system + 1];
      ranges->matrix_begin = batch.matrix_offsets[system];
      ranges->matrix_end = batch.matrix_offsets[system + 1];
      const std::int64_t shells = ranges->shell_end - ranges->shell_begin;
      const bool square_representable =
          shells >= 0 && (shells == 0 || shells <= kInt64Maximum / shells);
      *valid =
          ranges->atom_begin >= 0 && ranges->atom_begin <= ranges->atom_end &&
          ranges->atom_end <= batch.total_atoms && ranges->shell_begin >= 0 &&
          ranges->shell_begin <= ranges->shell_end && ranges->shell_end <= batch.total_shells &&
          ranges->matrix_begin >= 0 && ranges->matrix_begin <= ranges->matrix_end &&
          ranges->matrix_end <= batch.total_matrix_elements && square_representable &&
          ranges->matrix_end - ranges->matrix_begin == shells * shells &&
          (system != 0 ||
           (ranges->atom_begin == 0 && ranges->shell_begin == 0 && ranges->matrix_begin == 0)) &&
          (system + 1 != batch.batch_size ||
           (ranges->atom_end == batch.total_atoms && ranges->shell_end == batch.total_shells &&
            ranges->matrix_end == batch.total_matrix_elements));
      if (*valid != 0) {
        *valid = batch.atom_shell_offsets[ranges->atom_begin] == ranges->shell_begin &&
                 batch.atom_shell_offsets[ranges->atom_end] == ranges->shell_end;
      }
      if (*valid == 0) {
        record_error(device_error, Gfn2ES2DeviceError::kInvalidOffsets);
      }
    }
  }
  __syncthreads();
  if (*valid == 0) {
    return;
  }

  for (std::int64_t atom = ranges->atom_begin + threadIdx.x; atom < ranges->atom_end;
       atom += blockDim.x) {
    const std::int64_t shell_begin = batch.atom_shell_offsets[atom];
    const std::int64_t shell_end = batch.atom_shell_offsets[atom + 1];
    if (shell_begin < ranges->shell_begin || shell_begin >= shell_end ||
        shell_end > ranges->shell_end) {
      record_error(device_error, Gfn2ES2DeviceError::kInvalidOffsets);
      atomicExch(valid, 0);
    }
  }
  for (std::int64_t shell = ranges->shell_begin + threadIdx.x; shell < ranges->shell_end;
       shell += blockDim.x) {
    const std::int64_t atom = batch.shell_to_atom[shell];
    const double hardness = batch.shell_hardness[shell];
    if (atom < ranges->atom_begin || atom >= ranges->atom_end ||
        shell < batch.atom_shell_offsets[atom] || shell >= batch.atom_shell_offsets[atom + 1] ||
        !(hardness > 0.0) || !isfinite(hardness)) {
      record_error(device_error, Gfn2ES2DeviceError::kInvalidShellMetadata);
      atomicExch(valid, 0);
    }
  }
  __syncthreads();
}

__device__ void validate_positions(const Gfn2ES2DeviceBatch& batch, const SystemRanges& ranges,
                                   const double* positions, int* valid,
                                   std::uint32_t* device_error) {
  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    if (!isfinite(positions[coordinate]) || !isfinite(positions[coordinate + 1]) ||
        !isfinite(positions[coordinate + 2])) {
      record_error(device_error, Gfn2ES2DeviceError::kNonfinitePosition);
      atomicExch(valid, 0);
    }
  }
  __syncthreads();
}

__device__ void validate_shell_charges(const SystemRanges& ranges, const double* shell_charges,
                                       int* valid, std::uint32_t* device_error) {
  for (std::int64_t shell = ranges.shell_begin + threadIdx.x; shell < ranges.shell_end;
       shell += blockDim.x) {
    if (!isfinite(shell_charges[shell])) {
      record_error(device_error, Gfn2ES2DeviceError::kNonfiniteShellCharge);
      atomicExch(valid, 0);
    }
  }
  __syncthreads();
}

__device__ bool arithmetic_hardness(double first, double second, double* average) {
  const double sum = first + second;
  /* Match CPU ES2 exactly, including its subnormal-preserving finite path. */
  *average = isfinite(sum) ? 0.5 * sum : 0.5 * first + 0.5 * second;
  const double inverse = 1.0 / *average;
  return *average > 0.0 && isfinite(*average) && isfinite(inverse);
}

__device__ bool softened_kernel(double dx, double dy, double dz, double first_hardness,
                                double second_hardness, double* kernel) {
  double average = 0.0;
  if (!arithmetic_hardness(first_hardness, second_hardness, &average)) {
    return false;
  }
  const double inverse_average = 1.0 / average;
  const double softened_distance = hypot(hypot(dx, dy), hypot(dz, inverse_average));
  *kernel = 1.0 / softened_distance;
  return softened_distance > 0.0 && isfinite(softened_distance) && *kernel > 0.0 &&
         isfinite(*kernel);
}

__global__ void geometry_preflight_kernel(Gfn2ES2DeviceBatch batch, const double* positions,
                                          double* matrix_scratch, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  load_and_validate_system(batch, system, &ranges, &valid, device_error);
  if (valid == 0) {
    return;
  }
  validate_positions(batch, ranges, positions, &valid, device_error);
  if (valid == 0) {
    return;
  }

  const std::int64_t shells = ranges.shell_end - ranges.shell_begin;
  for (std::int64_t packed = ranges.matrix_begin + threadIdx.x; packed < ranges.matrix_end;
       packed += blockDim.x) {
    const std::int64_t local = packed - ranges.matrix_begin;
    const std::int64_t row_shell = ranges.shell_begin + local / shells;
    const std::int64_t column_shell = ranges.shell_begin + local % shells;
    const std::int64_t row_atom = batch.shell_to_atom[row_shell];
    const std::int64_t column_atom = batch.shell_to_atom[column_shell];
    const double row_hardness = batch.shell_hardness[row_shell];
    const double column_hardness = batch.shell_hardness[column_shell];
    double kernel = 0.0;
    bool finite_result = true;
    if (row_atom == column_atom) {
      finite_result = arithmetic_hardness(row_hardness, column_hardness, &kernel);
      if (!finite_result) {
        record_error(device_error, Gfn2ES2DeviceError::kNonfiniteHardnessArithmetic);
      }
    } else {
      const std::int64_t row_coordinate = row_atom * 3;
      const std::int64_t column_coordinate = column_atom * 3;
      const double dx = positions[row_coordinate] - positions[column_coordinate];
      const double dy = positions[row_coordinate + 1] - positions[column_coordinate + 1];
      const double dz = positions[row_coordinate + 2] - positions[column_coordinate + 2];
      if (!isfinite(dx) || !isfinite(dy) || !isfinite(dz)) {
        record_error(device_error, Gfn2ES2DeviceError::kCoordinateDifferenceOverflow);
        finite_result = false;
      } else if (!softened_kernel(dx, dy, dz, row_hardness, column_hardness, &kernel)) {
        record_error(device_error, Gfn2ES2DeviceError::kNonfiniteKernelArithmetic);
        finite_result = false;
      }
    }
    if (finite_result) {
      matrix_scratch[packed] = kernel;
    }
  }
}

__global__ void potential_preflight_kernel(Gfn2ES2DeviceBatch batch, Gfn2ES2DeviceCache cache,
                                           const double* shell_charges, double* shell_scratch,
                                           std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  load_and_validate_system(batch, system, &ranges, &valid, device_error);
  if (valid == 0) {
    return;
  }
  const std::int64_t shells = ranges.shell_end - ranges.shell_begin;
  for (std::int64_t row_shell = ranges.shell_begin + threadIdx.x; row_shell < ranges.shell_end;
       row_shell += blockDim.x) {
    double potential = 0.0;
    bool finite_result = true;
    const std::int64_t local_row = row_shell - ranges.shell_begin;
    for (std::int64_t column_shell = ranges.shell_begin; column_shell < ranges.shell_end;
         ++column_shell) {
      const std::int64_t matrix_index =
          ranges.matrix_begin + local_row * shells + column_shell - ranges.shell_begin;
      const double kernel = cache.coulomb_matrix[matrix_index];
      const double charge = shell_charges[column_shell];
      if (!(kernel > 0.0) || !isfinite(kernel)) {
        record_error(device_error, Gfn2ES2DeviceError::kInvalidCacheMatrix);
        finite_result = false;
        break;
      }
      if (!isfinite(charge)) {
        record_error(device_error, Gfn2ES2DeviceError::kNonfiniteShellCharge);
        finite_result = false;
        break;
      }
      const double contribution = kernel * charge;
      const double updated = potential + contribution;
      if (!isfinite(contribution) || !isfinite(updated)) {
        record_error(device_error, Gfn2ES2DeviceError::kNonfinitePotentialArithmetic);
        finite_result = false;
        break;
      }
      potential = updated;
    }
    if (finite_result) {
      shell_scratch[row_shell] = potential;
    }
  }
}

__global__ void energy_preflight_kernel(Gfn2ES2DeviceBatch batch, Gfn2ES2DeviceCache cache,
                                        const double* shell_charges, const double* energies,
                                        double* batch_scratch, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  __shared__ double partial_energy[kThreadsPerBlock];
  load_and_validate_system(batch, system, &ranges, &valid, device_error);
  if (valid == 0) {
    return;
  }

  /* Each lane evaluates complete rows in CPU order before the block reduction. */
  double local_energy = 0.0;
  bool finite_result = true;
  const std::int64_t shells = ranges.shell_end - ranges.shell_begin;
  for (std::int64_t row_shell = ranges.shell_begin + threadIdx.x;
       row_shell < ranges.shell_end && finite_result; row_shell += blockDim.x) {
    const double row_charge = shell_charges[row_shell];
    if (!isfinite(row_charge)) {
      record_error(device_error, Gfn2ES2DeviceError::kNonfiniteShellCharge);
      finite_result = false;
      break;
    }
    double potential = 0.0;
    const std::int64_t local_row = row_shell - ranges.shell_begin;
    for (std::int64_t column_shell = ranges.shell_begin; column_shell < ranges.shell_end;
         ++column_shell) {
      const std::int64_t matrix_index =
          ranges.matrix_begin + local_row * shells + column_shell - ranges.shell_begin;
      const double kernel = cache.coulomb_matrix[matrix_index];
      const double column_charge = shell_charges[column_shell];
      if (!(kernel > 0.0) || !isfinite(kernel)) {
        record_error(device_error, Gfn2ES2DeviceError::kInvalidCacheMatrix);
        finite_result = false;
        break;
      }
      if (!isfinite(column_charge)) {
        record_error(device_error, Gfn2ES2DeviceError::kNonfiniteShellCharge);
        finite_result = false;
        break;
      }
      const double contribution = kernel * column_charge;
      const double updated = potential + contribution;
      if (!isfinite(contribution) || !isfinite(updated)) {
        record_error(device_error, Gfn2ES2DeviceError::kNonfiniteEnergyArithmetic);
        finite_result = false;
        break;
      }
      potential = updated;
    }
    if (!finite_result) {
      break;
    }
    const double contribution = 0.5 * row_charge * potential;
    const double updated = local_energy + contribution;
    if (!isfinite(contribution) || !isfinite(updated)) {
      record_error(device_error, Gfn2ES2DeviceError::kNonfiniteEnergyArithmetic);
      finite_result = false;
      break;
    }
    local_energy = updated;
  }

  partial_energy[threadIdx.x] = finite_result ? local_energy : 0.0;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      const double updated = partial_energy[threadIdx.x] + partial_energy[threadIdx.x + offset];
      if (!isfinite(updated)) {
        record_error(device_error, Gfn2ES2DeviceError::kNonfiniteEnergyArithmetic);
      }
      partial_energy[threadIdx.x] = updated;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && sequence_is_valid(device_error)) {
    const double seed = energies[system];
    if (!isfinite(seed)) {
      record_error(device_error, Gfn2ES2DeviceError::kNonfiniteEnergySeed);
      return;
    }
    const double updated = seed + partial_energy[0];
    if (!isfinite(updated)) {
      record_error(device_error, Gfn2ES2DeviceError::kNonfiniteEnergyArithmetic);
      return;
    }
    batch_scratch[system] = updated;
  }
}

__global__ void gradient_preflight_kernel(Gfn2ES2DeviceBatch batch, Gfn2ES2DeviceCache cache,
                                          const double* positions, const double* shell_charges,
                                          double* gradient_scratch, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  load_and_validate_system(batch, system, &ranges, &valid, device_error);
  if (valid == 0) {
    return;
  }
  validate_positions(batch, ranges, positions, &valid, device_error);
  if (valid == 0) {
    return;
  }
  validate_shell_charges(ranges, shell_charges, &valid, device_error);
  if (valid == 0) {
    return;
  }

  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    gradient_scratch[atom * 3] = 0.0;
    gradient_scratch[atom * 3 + 1] = 0.0;
    gradient_scratch[atom * 3 + 2] = 0.0;
  }
  __syncthreads();
  const std::int64_t shells = ranges.shell_end - ranges.shell_begin;
  for (std::int64_t first_atom = ranges.atom_begin + 1 + threadIdx.x; first_atom < ranges.atom_end;
       first_atom += blockDim.x) {
    const std::int64_t first_shell_begin = batch.atom_shell_offsets[first_atom];
    const std::int64_t first_shell_end = batch.atom_shell_offsets[first_atom + 1];
    for (std::int64_t second_atom = ranges.atom_begin; second_atom < first_atom; ++second_atom) {
      const std::int64_t first_coordinate = first_atom * 3;
      const std::int64_t second_coordinate = second_atom * 3;
      const double dx = positions[first_coordinate] - positions[second_coordinate];
      const double dy = positions[first_coordinate + 1] - positions[second_coordinate + 1];
      const double dz = positions[first_coordinate + 2] - positions[second_coordinate + 2];
      if (!isfinite(dx) || !isfinite(dy) || !isfinite(dz)) {
        record_error(device_error, Gfn2ES2DeviceError::kCoordinateDifferenceOverflow);
        return;
      }
      const std::int64_t second_shell_begin = batch.atom_shell_offsets[second_atom];
      const std::int64_t second_shell_end = batch.atom_shell_offsets[second_atom + 1];
      double weighted_derivative = 0.0;
      for (std::int64_t first_shell = first_shell_begin; first_shell < first_shell_end;
           ++first_shell) {
        const double first_charge = shell_charges[first_shell];
        for (std::int64_t second_shell = second_shell_begin; second_shell < second_shell_end;
             ++second_shell) {
          const double second_charge = shell_charges[second_shell];
          const std::int64_t matrix_index = ranges.matrix_begin +
                                            (first_shell - ranges.shell_begin) * shells +
                                            second_shell - ranges.shell_begin;
          const double kernel = cache.coulomb_matrix[matrix_index];
          if (!(kernel > 0.0) || !isfinite(kernel)) {
            record_error(device_error, Gfn2ES2DeviceError::kInvalidCacheMatrix);
            return;
          }
          double contribution = first_charge * kernel;
          contribution *= second_charge;
          contribution *= kernel;
          contribution *= kernel;
          const double updated = weighted_derivative + contribution;
          if (!isfinite(contribution) || !isfinite(updated)) {
            record_error(device_error, Gfn2ES2DeviceError::kNonfiniteGradientArithmetic);
            return;
          }
          weighted_derivative = updated;
        }
      }
      const double displacement[3]{dx, dy, dz};
      for (std::int64_t axis = 0; axis < 3; ++axis) {
        const double pair_contribution = -weighted_derivative * displacement[axis];
        const std::int64_t first_index = first_coordinate + axis;
        const std::int64_t second_index = second_coordinate + axis;
        if (!isfinite(pair_contribution)) {
          record_error(device_error, Gfn2ES2DeviceError::kNonfiniteGradientArithmetic);
          return;
        }
        const double old_first = atomic_add_fp64(gradient_scratch + first_index, pair_contribution);
        const double old_second =
            atomic_add_fp64(gradient_scratch + second_index, -pair_contribution);
        if (!isfinite(old_first) || !isfinite(old_first + pair_contribution) ||
            !isfinite(old_second) || !isfinite(old_second - pair_contribution)) {
          record_error(device_error, Gfn2ES2DeviceError::kNonfiniteGradientArithmetic);
          return;
        }
      }
    }
  }
}

__global__ void gradient_seed_preflight_kernel(std::int64_t coordinate_count,
                                               const double* gradient_scratch,
                                               const double* gradients,
                                               std::uint32_t* device_error) {
  if (!sequence_is_valid(device_error)) {
    return;
  }
  for (std::int64_t coordinate = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       coordinate < coordinate_count;
       coordinate += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const double seed = gradients[coordinate];
    const double updated = seed + gradient_scratch[coordinate];
    if (!isfinite(seed)) {
      record_error(device_error, Gfn2ES2DeviceError::kNonfiniteGradientSeed);
    } else if (!isfinite(gradient_scratch[coordinate]) || !isfinite(updated)) {
      record_error(device_error, Gfn2ES2DeviceError::kNonfiniteGradientArithmetic);
    }
  }
}

__global__ void copy_if_success_kernel(std::int64_t count, const double* source, double* target,
                                       const std::uint32_t* device_error) {
  if (!sequence_is_valid(device_error)) {
    return;
  }
  for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    target[index] = source[index];
  }
}

__global__ void add_if_success_kernel(std::int64_t count, const double* source, double* target,
                                      const std::uint32_t* device_error) {
  if (!sequence_is_valid(device_error)) {
    return;
  }
  for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    target[index] += source[index];
  }
}

__device__ void record_scc_plan_error(std::uint32_t* plan_error, Gfn2ES2DeviceError error) {
  atomicCAS(plan_error, static_cast<std::uint32_t>(Gfn2ES2DeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ void record_scc_system_error(std::uint32_t* system_errors, std::int64_t system,
                                        Gfn2ES2DeviceError error) {
  atomicCAS(system_errors + system, static_cast<std::uint32_t>(Gfn2ES2DeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ bool scc_member_is_active(const Gfn2SccIterationDeviceActivity& activity,
                                     const std::uint32_t* system_errors,
                                     const std::uint32_t* plan_error, std::int64_t system) {
  /* Canonical sequence and member activity always precede numerical reads. */
  if (atomicAdd(const_cast<std::uint32_t*>(activity.sequence_active), 0u) != 1u ||
      activity.active_mask[system] != 1u) {
    return false;
  }
  return atomicAdd(const_cast<std::uint32_t*>(plan_error), 0u) == 0u &&
         atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) == 0u;
}

/* Device offset contents are a plan-level contract and are checked once before physics reads. */
__global__ void es2_scc_plan_preflight_kernel(Gfn2ES2DeviceBatch batch, Gfn2ES2DeviceCache cache,
                                              std::uint64_t geometry_generation,
                                              Gfn2SccIterationDeviceActivity activity,
                                              std::uint32_t* plan_error) {
  __shared__ int sequence_active;
  __shared__ int any_active;
  if (threadIdx.x == 0) {
    sequence_active = atomicAdd(const_cast<std::uint32_t*>(activity.sequence_active), 0u) == 1u;
    any_active = 0;
  }
  __syncthreads();
  if (sequence_active == 0) {
    return;
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    if (activity.active_mask[system] == 1u) {
      atomicExch(&any_active, 1);
    }
  }
  __syncthreads();
  if (any_active == 0 || atomicAdd(plan_error, 0u) != 0u) {
    return;
  }
  if (threadIdx.x == 0 &&
      (geometry_generation == 0u || cache.geometry_generation != geometry_generation ||
       cache.plan_token != batch.plan_token)) {
    record_scc_plan_error(plan_error, Gfn2ES2DeviceError::kInvalidCacheMatrix);
  }
  __syncthreads();
  if (atomicAdd(plan_error, 0u) != 0u) {
    return;
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    /* Inactive topology is generation-local poison by contract. Only an active
     * member may authorize reads of its ragged boundaries and atom partition. */
    if (activity.active_mask[system] != 1u) {
      continue;
    }
    const std::int64_t atom_begin = batch.atom_offsets[system];
    const std::int64_t atom_end = batch.atom_offsets[system + 1];
    const std::int64_t shell_begin = batch.batch_shell_offsets[system];
    const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
    const std::int64_t matrix_begin = batch.matrix_offsets[system];
    const std::int64_t matrix_end = batch.matrix_offsets[system + 1];
    bool valid = atom_begin >= 0 && atom_begin <= atom_end && atom_end <= batch.total_atoms &&
                 shell_begin >= 0 && shell_begin <= shell_end && shell_end <= batch.total_shells &&
                 matrix_begin >= 0 && matrix_begin <= matrix_end &&
                 matrix_end <= batch.total_matrix_elements;
    if (valid) {
      const std::int64_t shells = shell_end - shell_begin;
      const bool square_representable = shells == 0 || shells <= kInt64Maximum / shells;
      valid = square_representable && matrix_end - matrix_begin == shells * shells;
    }
    if (valid && system == 0) {
      valid = atom_begin == 0 && shell_begin == 0 && matrix_begin == 0;
    }
    if (valid && system + 1 == batch.batch_size) {
      valid = atom_end == batch.total_atoms && shell_end == batch.total_shells &&
              matrix_end == batch.total_matrix_elements;
    }
    if (valid) {
      valid = batch.atom_shell_offsets[atom_begin] == shell_begin &&
              batch.atom_shell_offsets[atom_end] == shell_end;
    }
    if (valid) {
      for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
        const std::int64_t atom_shell_begin = batch.atom_shell_offsets[atom];
        const std::int64_t atom_shell_end = batch.atom_shell_offsets[atom + 1];
        if (atom_shell_begin < shell_begin || atom_shell_begin > atom_shell_end ||
            atom_shell_end > shell_end) {
          valid = false;
          break;
        }
      }
    }
    if (!valid) {
      record_scc_plan_error(plan_error, Gfn2ES2DeviceError::kInvalidOffsets);
    }
  }
}

__global__ void es2_scc_potential_preflight_kernel(
    Gfn2ES2DeviceBatch batch, Gfn2ES2DeviceCache cache, Gfn2SccIterationDeviceActivity activity,
    const double* shell_charges, double* shell_scratch, std::uint32_t* system_errors,
    const std::uint32_t* plan_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int active;
  if (threadIdx.x == 0) {
    active = scc_member_is_active(activity, system_errors, plan_error, system) ? 1 : 0;
  }
  __syncthreads();
  if (active == 0) {
    return;
  }

  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t shells = shell_end - shell_begin;
  for (std::int64_t row = shell_begin + threadIdx.x; row < shell_end; row += blockDim.x) {
    double potential = 0.0;
    const std::int64_t local_row = row - shell_begin;
    for (std::int64_t column = shell_begin; column < shell_end; ++column) {
      const double kernel =
          cache.coulomb_matrix[matrix_begin + local_row * shells + column - shell_begin];
      const double charge = shell_charges[column];
      if (!(kernel > 0.0) || !isfinite(kernel)) {
        record_scc_system_error(system_errors, system, Gfn2ES2DeviceError::kInvalidCacheMatrix);
        return;
      }
      if (!isfinite(charge)) {
        record_scc_system_error(system_errors, system, Gfn2ES2DeviceError::kNonfiniteShellCharge);
        return;
      }
      const double contribution = kernel * charge;
      const double updated = potential + contribution;
      if (!isfinite(contribution) || !isfinite(updated)) {
        record_scc_system_error(system_errors, system,
                                Gfn2ES2DeviceError::kNonfinitePotentialArithmetic);
        return;
      }
      potential = updated;
    }
    shell_scratch[row] = potential;
  }
}

__global__ void es2_scc_publish_potential_kernel(
    Gfn2ES2DeviceBatch batch, Gfn2SccIterationDeviceActivity activity, const double* shell_scratch,
    double* shell_potentials, const std::uint32_t* system_errors, const std::uint32_t* plan_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!scc_member_is_active(activity, system_errors, plan_error, system)) {
    return;
  }
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    shell_potentials[shell] = shell_scratch[shell];
  }
}

__global__ void es2_scc_energy_preflight_kernel(Gfn2ES2DeviceBatch batch, Gfn2ES2DeviceCache cache,
                                                Gfn2SccIterationDeviceActivity activity,
                                                const double* shell_charges, double* batch_scratch,
                                                std::uint32_t* system_errors,
                                                const std::uint32_t* plan_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!scc_member_is_active(activity, system_errors, plan_error, system)) {
    return;
  }
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t shells = shell_end - shell_begin;

  /* Each lane owns a strided row slice and keeps that row's column sweep as a
   * serial FMA chain (the same per-row order as the CPU oracle); only the
   * outer row combination becomes a block tree reduction. The former version
   * executed the whole O(shells^2) sweep on one thread per system, leaving
   * fewer than a warp of active lanes per block. The reordered outer sum
   * changes rounding far below every retained parity gate (2e-8 relative in
   * the production test, 5e-7/1e-6 in public conformance). */
  __shared__ int failure_code;
  __shared__ double partial[kThreadsPerBlock];
  if (threadIdx.x == 0) {
    failure_code = 0;
  }
  __syncthreads();

  double local_energy = 0.0;
  for (std::int64_t row = shell_begin + threadIdx.x; row < shell_end; row += blockDim.x) {
    const double row_charge = shell_charges[row];
    if (!isfinite(row_charge)) {
      atomicCAS(&failure_code, 0, static_cast<int>(Gfn2ES2DeviceError::kNonfiniteShellCharge));
      continue;
    }
    const std::int64_t row_index = (row - shell_begin) * shells;
    double potential = 0.0;
    for (std::int64_t column = shell_begin; column < shell_end; ++column) {
      const double kernel = cache.coulomb_matrix[matrix_begin + row_index + column - shell_begin];
      const double column_charge = shell_charges[column];
      if (!(kernel > 0.0) || !isfinite(kernel)) {
        atomicCAS(&failure_code, 0, static_cast<int>(Gfn2ES2DeviceError::kInvalidCacheMatrix));
        continue;
      }
      if (!isfinite(column_charge)) {
        atomicCAS(&failure_code, 0, static_cast<int>(Gfn2ES2DeviceError::kNonfiniteShellCharge));
        continue;
      }
      const double contribution = kernel * column_charge;
      const double updated = potential + contribution;
      if (!isfinite(contribution) || !isfinite(updated)) {
        atomicCAS(&failure_code, 0,
                  static_cast<int>(Gfn2ES2DeviceError::kNonfiniteEnergyArithmetic));
        continue;
      }
      potential = updated;
    }
    const double contribution = 0.5 * row_charge * potential;
    const double updated = local_energy + contribution;
    if (!isfinite(contribution) || !isfinite(updated)) {
      atomicCAS(&failure_code, 0, static_cast<int>(Gfn2ES2DeviceError::kNonfiniteEnergyArithmetic));
      continue;
    }
    local_energy = updated;
  }
  partial[threadIdx.x] = local_energy;
  __syncthreads();
  for (int offset = kThreadsPerBlock / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      const double updated = partial[threadIdx.x] + partial[threadIdx.x + offset];
      if (!isfinite(updated)) {
        atomicCAS(&failure_code, 0,
                  static_cast<int>(Gfn2ES2DeviceError::kNonfiniteEnergyArithmetic));
      } else {
        partial[threadIdx.x] = updated;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x != 0) {
    return;
  }
  if (failure_code != 0) {
    record_scc_system_error(system_errors, system, static_cast<Gfn2ES2DeviceError>(failure_code));
    return;
  }
  batch_scratch[system] = partial[0];
}

__global__ void es2_scc_publish_energy_kernel(Gfn2SccIterationDeviceActivity activity,
                                              const double* batch_scratch,
                                              double* component_energies,
                                              const std::uint32_t* system_errors,
                                              const std::uint32_t* plan_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (threadIdx.x == 0 && scc_member_is_active(activity, system_errors, plan_error, system)) {
    component_energies[system] = batch_scratch[system];
  }
}

bool count_bytes(std::int64_t count, std::size_t element_size, std::size_t* bytes) noexcept {
  if (count < 0 ||
      static_cast<std::uint64_t>(count) >
          static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
      static_cast<std::uint64_t>(count) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
    return false;
  }
  *bytes = static_cast<std::size_t>(count) * element_size;
  return true;
}

bool ranges_overlap(const void* first, std::size_t first_bytes, const void* second,
                    std::size_t second_bytes) noexcept {
  if (first_bytes == 0u || second_bytes == 0u) {
    return false;
  }
  const auto first_begin = reinterpret_cast<std::uintptr_t>(first);
  const auto second_begin = reinterpret_cast<std::uintptr_t>(second);
  if (first_begin > std::numeric_limits<std::uintptr_t>::max() - first_bytes ||
      second_begin > std::numeric_limits<std::uintptr_t>::max() - second_bytes) {
    return true;
  }
  const auto first_end = first_begin + first_bytes;
  const auto second_end = second_begin + second_bytes;
  return first_begin < second_end && second_begin < first_end;
}

bool is_aligned(const void* pointer, std::size_t alignment) noexcept {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

struct MemoryRange {
  const void* data;
  std::size_t bytes;
};

template <std::size_t N>
bool pairwise_disjoint(const MemoryRange (&ranges)[N]) noexcept {
  for (std::size_t first = 0; first < N; ++first) {
    for (std::size_t second = first + 1u; second < N; ++second) {
      if (ranges_overlap(ranges[first].data, ranges[first].bytes, ranges[second].data,
                         ranges[second].bytes)) {
        return false;
      }
    }
  }
  return true;
}

struct CommonBytes {
  std::size_t atom_offsets;
  std::size_t batch_shell_offsets;
  std::size_t atom_shell_offsets;
  std::size_t matrix_offsets;
  std::size_t shell_to_atom;
  std::size_t shell_hardness;
  std::size_t positions;
  std::size_t matrix;
  std::size_t shells;
  std::size_t batch;
  std::size_t gradients;
};

cudaError_t validate_common(const Gfn2ES2DeviceBatch& batch, std::uint32_t* device_error,
                            CommonBytes* bytes) noexcept {
  if (batch.batch_size <= 0 || batch.total_atoms <= 0 || batch.total_shells <= 0 ||
      batch.total_matrix_elements <= 0 || batch.plan_token == 0 ||
      batch.batch_size == std::numeric_limits<std::int64_t>::max() ||
      batch.total_atoms == std::numeric_limits<std::int64_t>::max() ||
      batch.atom_offset_count != batch.batch_size + 1 ||
      batch.batch_shell_offset_count != batch.batch_size + 1 ||
      batch.atom_shell_offset_count != batch.total_atoms + 1 ||
      batch.matrix_offset_count != batch.batch_size + 1 ||
      batch.shell_to_atom_count != batch.total_shells ||
      batch.shell_hardness_count != batch.total_shells || batch.atom_offsets == nullptr ||
      batch.batch_shell_offsets == nullptr || batch.atom_shell_offsets == nullptr ||
      batch.matrix_offsets == nullptr || batch.shell_to_atom == nullptr ||
      batch.shell_hardness == nullptr || device_error == nullptr ||
      !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.batch_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.atom_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.matrix_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_to_atom, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_hardness, alignof(double)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  if (static_cast<std::uint64_t>(batch.batch_size) >
      static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return cudaErrorInvalidConfiguration;
  }
  std::int64_t coordinate_count = 0;
  if (batch.total_atoms > std::numeric_limits<std::int64_t>::max() / 3) {
    return cudaErrorInvalidValue;
  }
  coordinate_count = batch.total_atoms * 3;
  if (!count_bytes(batch.atom_offset_count, sizeof(std::int64_t), &bytes->atom_offsets) ||
      !count_bytes(batch.batch_shell_offset_count, sizeof(std::int64_t),
                   &bytes->batch_shell_offsets) ||
      !count_bytes(batch.atom_shell_offset_count, sizeof(std::int64_t),
                   &bytes->atom_shell_offsets) ||
      !count_bytes(batch.matrix_offset_count, sizeof(std::int64_t), &bytes->matrix_offsets) ||
      !count_bytes(batch.shell_to_atom_count, sizeof(std::int64_t), &bytes->shell_to_atom) ||
      !count_bytes(batch.shell_hardness_count, sizeof(double), &bytes->shell_hardness) ||
      !count_bytes(coordinate_count, sizeof(double), &bytes->positions) ||
      !count_bytes(batch.total_matrix_elements, sizeof(double), &bytes->matrix) ||
      !count_bytes(batch.total_shells, sizeof(double), &bytes->shells) ||
      !count_bytes(batch.batch_size, sizeof(double), &bytes->batch) ||
      !count_bytes(coordinate_count, sizeof(double), &bytes->gradients)) {
    return cudaErrorInvalidValue;
  }
  const MemoryRange ranges[]{{batch.atom_offsets, bytes->atom_offsets},
                             {batch.batch_shell_offsets, bytes->batch_shell_offsets},
                             {batch.atom_shell_offsets, bytes->atom_shell_offsets},
                             {batch.matrix_offsets, bytes->matrix_offsets},
                             {batch.shell_to_atom, bytes->shell_to_atom},
                             {batch.shell_hardness, bytes->shell_hardness},
                             {device_error, sizeof(*device_error)}};
  return pairwise_disjoint(ranges) ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t validate_cache(const Gfn2ES2DeviceBatch& batch,
                           const Gfn2ES2DeviceCache& cache) noexcept {
  if (cache.coulomb_matrix == nullptr || cache.matrix_elements != batch.total_matrix_elements ||
      cache.plan_token != batch.plan_token || !is_aligned(cache.coulomb_matrix, alignof(double))) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t validate_scc_control(const Gfn2ES2DeviceBatch& batch, const Gfn2ES2DeviceCache& cache,
                                 std::uint64_t geometry_generation,
                                 const Gfn2SccIterationDeviceActivity& activity,
                                 std::uint32_t* system_errors, std::uint32_t* plan_error,
                                 CommonBytes* bytes, std::size_t* system_error_bytes) noexcept {
  cudaError_t status = validate_common(batch, plan_error, bytes);
  if (status != cudaSuccess) {
    return status;
  }
  if (cache.coulomb_matrix == nullptr || cache.matrix_elements != batch.total_matrix_elements ||
      !is_aligned(cache.coulomb_matrix, alignof(double)) || activity.active_mask == nullptr ||
      activity.sequence_active == nullptr || activity.batch_elements != batch.batch_size ||
      activity.sequence_elements != 1 || activity.plan_token != batch.plan_token ||
      system_errors == nullptr || !is_aligned(activity.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(plan_error, alignof(std::uint32_t)) ||
      !count_bytes(batch.batch_size, sizeof(std::uint32_t), system_error_bytes)) {
    return cudaErrorInvalidValue;
  }
  (void)geometry_generation;
  return cudaSuccess;
}

cudaError_t copy_grid(std::int64_t count, unsigned int* blocks) noexcept {
  if (count <= 0) {
    return cudaErrorInvalidValue;
  }
  const std::uint64_t block_count =
      (static_cast<std::uint64_t>(count) + kThreadsPerBlock - 1u) / kThreadsPerBlock;
  if (block_count == 0u || block_count > std::numeric_limits<unsigned int>::max()) {
    return cudaErrorInvalidConfiguration;
  }
  *blocks = static_cast<unsigned int>(block_count);
  return cudaSuccess;
}

cudaError_t check_launch() noexcept { return cudaGetLastError(); }

}  // namespace

cudaError_t reset_gfn2_es2_device_error_cuda(std::uint32_t* device_error,
                                             cudaStream_t stream) noexcept {
  if (device_error == nullptr || !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  return cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream);
}

cudaError_t reset_gfn2_es2_scc_errors_cuda(std::int64_t batch_size, std::uint32_t* system_errors,
                                           std::uint32_t* plan_error,
                                           cudaStream_t stream) noexcept {
  std::size_t system_error_bytes = 0;
  if (batch_size <= 0 || system_errors == nullptr || plan_error == nullptr ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(plan_error, alignof(std::uint32_t)) ||
      !count_bytes(batch_size, sizeof(std::uint32_t), &system_error_bytes) ||
      ranges_overlap(system_errors, system_error_bytes, plan_error, sizeof(*plan_error))) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(system_errors, 0, system_error_bytes, stream);
  return status == cudaSuccess ? cudaMemsetAsync(plan_error, 0, sizeof(*plan_error), stream)
                               : status;
}

cudaError_t update_gfn2_es2_geometry_cache_cuda(const Gfn2ES2DeviceBatch& batch,
                                                const double* positions,
                                                const Gfn2ES2DeviceCache& cache,
                                                const Gfn2ES2DeviceWorkspace& workspace,
                                                std::uint32_t* device_error,
                                                cudaStream_t stream) noexcept {
  CommonBytes bytes{};
  cudaError_t status = validate_common(batch, device_error, &bytes);
  if (status != cudaSuccess) {
    return status;
  }
  status = validate_cache(batch, cache);
  if (status != cudaSuccess || positions == nullptr || workspace.matrix_scratch == nullptr ||
      workspace.matrix_elements < batch.total_matrix_elements ||
      !is_aligned(positions, alignof(double)) ||
      !is_aligned(workspace.matrix_scratch, alignof(double))) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  const MemoryRange ranges[]{{batch.atom_offsets, bytes.atom_offsets},
                             {batch.batch_shell_offsets, bytes.batch_shell_offsets},
                             {batch.atom_shell_offsets, bytes.atom_shell_offsets},
                             {batch.matrix_offsets, bytes.matrix_offsets},
                             {batch.shell_to_atom, bytes.shell_to_atom},
                             {batch.shell_hardness, bytes.shell_hardness},
                             {device_error, sizeof(*device_error)},
                             {positions, bytes.positions},
                             {cache.coulomb_matrix, bytes.matrix},
                             {workspace.matrix_scratch, bytes.matrix},
                             {&batch, sizeof(batch)},
                             {&cache, sizeof(cache)},
                             {&workspace, sizeof(workspace)}};
  if (!pairwise_disjoint(ranges)) {
    return cudaErrorInvalidValue;
  }
  unsigned int blocks = 0;
  status = copy_grid(batch.total_matrix_elements, &blocks);
  if (status != cudaSuccess) {
    return status;
  }
  geometry_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                              stream>>>(batch, positions, workspace.matrix_scratch, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  copy_if_success_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch.total_matrix_elements, workspace.matrix_scratch, cache.coulomb_matrix, device_error);
  return check_launch();
}

cudaError_t evaluate_gfn2_es2_potential_cuda(const Gfn2ES2DeviceBatch& batch,
                                             const Gfn2ES2DeviceCache& cache,
                                             const double* shell_charges, double* shell_potentials,
                                             const Gfn2ES2DeviceWorkspace& workspace,
                                             std::uint32_t* device_error,
                                             cudaStream_t stream) noexcept {
  CommonBytes bytes{};
  cudaError_t status = validate_common(batch, device_error, &bytes);
  if (status != cudaSuccess) {
    return status;
  }
  status = validate_cache(batch, cache);
  if (status != cudaSuccess || shell_charges == nullptr || shell_potentials == nullptr ||
      workspace.shell_scratch == nullptr || workspace.shell_elements < batch.total_shells ||
      !is_aligned(shell_charges, alignof(double)) ||
      !is_aligned(shell_potentials, alignof(double)) ||
      !is_aligned(workspace.shell_scratch, alignof(double))) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  const MemoryRange ranges[]{{batch.atom_offsets, bytes.atom_offsets},
                             {batch.batch_shell_offsets, bytes.batch_shell_offsets},
                             {batch.atom_shell_offsets, bytes.atom_shell_offsets},
                             {batch.matrix_offsets, bytes.matrix_offsets},
                             {batch.shell_to_atom, bytes.shell_to_atom},
                             {batch.shell_hardness, bytes.shell_hardness},
                             {device_error, sizeof(*device_error)},
                             {cache.coulomb_matrix, bytes.matrix},
                             {shell_charges, bytes.shells},
                             {shell_potentials, bytes.shells},
                             {workspace.shell_scratch, bytes.shells},
                             {&batch, sizeof(batch)},
                             {&cache, sizeof(cache)},
                             {&workspace, sizeof(workspace)}};
  if (!pairwise_disjoint(ranges)) {
    return cudaErrorInvalidValue;
  }
  unsigned int blocks = 0;
  status = copy_grid(batch.total_shells, &blocks);
  if (status != cudaSuccess) {
    return status;
  }
  potential_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                               stream>>>(batch, cache, shell_charges, workspace.shell_scratch,
                                         device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  copy_if_success_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch.total_shells, workspace.shell_scratch, shell_potentials, device_error);
  return check_launch();
}

cudaError_t add_gfn2_es2_energy_cuda(const Gfn2ES2DeviceBatch& batch,
                                     const Gfn2ES2DeviceCache& cache, const double* shell_charges,
                                     double* energies, const Gfn2ES2DeviceWorkspace& workspace,
                                     std::uint32_t* device_error, cudaStream_t stream) noexcept {
  CommonBytes bytes{};
  cudaError_t status = validate_common(batch, device_error, &bytes);
  if (status != cudaSuccess) {
    return status;
  }
  status = validate_cache(batch, cache);
  if (status != cudaSuccess || shell_charges == nullptr || energies == nullptr ||
      workspace.batch_scratch == nullptr || workspace.batch_elements < batch.batch_size ||
      !is_aligned(shell_charges, alignof(double)) || !is_aligned(energies, alignof(double)) ||
      !is_aligned(workspace.batch_scratch, alignof(double))) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  const MemoryRange ranges[]{{batch.atom_offsets, bytes.atom_offsets},
                             {batch.batch_shell_offsets, bytes.batch_shell_offsets},
                             {batch.atom_shell_offsets, bytes.atom_shell_offsets},
                             {batch.matrix_offsets, bytes.matrix_offsets},
                             {batch.shell_to_atom, bytes.shell_to_atom},
                             {batch.shell_hardness, bytes.shell_hardness},
                             {device_error, sizeof(*device_error)},
                             {cache.coulomb_matrix, bytes.matrix},
                             {shell_charges, bytes.shells},
                             {energies, bytes.batch},
                             {workspace.batch_scratch, bytes.batch},
                             {&batch, sizeof(batch)},
                             {&cache, sizeof(cache)},
                             {&workspace, sizeof(workspace)}};
  if (!pairwise_disjoint(ranges)) {
    return cudaErrorInvalidValue;
  }
  unsigned int blocks = 0;
  status = copy_grid(batch.batch_size, &blocks);
  if (status != cudaSuccess) {
    return status;
  }
  energy_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                            stream>>>(batch, cache, shell_charges, energies,
                                      workspace.batch_scratch, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  copy_if_success_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch.batch_size, workspace.batch_scratch, energies, device_error);
  return check_launch();
}

cudaError_t add_gfn2_es2_gradient_cuda(const Gfn2ES2DeviceBatch& batch,
                                       const Gfn2ES2DeviceCache& cache, const double* positions,
                                       std::uint64_t geometry_generation,
                                       const double* shell_charges, double* gradients,
                                       const Gfn2ES2DeviceWorkspace& workspace,
                                       std::uint32_t* device_error, cudaStream_t stream) noexcept {
  CommonBytes bytes{};
  cudaError_t status = validate_common(batch, device_error, &bytes);
  if (status != cudaSuccess) {
    return status;
  }
  status = validate_cache(batch, cache);
  if (status != cudaSuccess || geometry_generation != cache.geometry_generation ||
      positions == nullptr || shell_charges == nullptr || gradients == nullptr ||
      workspace.gradient_scratch == nullptr ||
      workspace.gradient_elements < batch.total_atoms * 3 ||
      !is_aligned(positions, alignof(double)) || !is_aligned(shell_charges, alignof(double)) ||
      !is_aligned(gradients, alignof(double)) ||
      !is_aligned(workspace.gradient_scratch, alignof(double))) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  const MemoryRange ranges[]{{batch.atom_offsets, bytes.atom_offsets},
                             {batch.batch_shell_offsets, bytes.batch_shell_offsets},
                             {batch.atom_shell_offsets, bytes.atom_shell_offsets},
                             {batch.matrix_offsets, bytes.matrix_offsets},
                             {batch.shell_to_atom, bytes.shell_to_atom},
                             {batch.shell_hardness, bytes.shell_hardness},
                             {device_error, sizeof(*device_error)},
                             {cache.coulomb_matrix, bytes.matrix},
                             {positions, bytes.positions},
                             {shell_charges, bytes.shells},
                             {gradients, bytes.gradients},
                             {workspace.gradient_scratch, bytes.gradients},
                             {&batch, sizeof(batch)},
                             {&cache, sizeof(cache)},
                             {&workspace, sizeof(workspace)}};
  if (!pairwise_disjoint(ranges)) {
    return cudaErrorInvalidValue;
  }
  unsigned int blocks = 0;
  status = copy_grid(batch.total_atoms * 3, &blocks);
  if (status != cudaSuccess) {
    return status;
  }
  gradient_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                              stream>>>(batch, cache, positions, shell_charges,
                                        workspace.gradient_scratch, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  gradient_seed_preflight_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch.total_atoms * 3, workspace.gradient_scratch, gradients, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  add_if_success_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch.total_atoms * 3, workspace.gradient_scratch, gradients, device_error);
  return check_launch();
}

cudaError_t evaluate_gfn2_es2_scc_potential_cuda(
    const Gfn2ES2DeviceBatch& batch, const Gfn2ES2DeviceCache& cache,
    std::uint64_t geometry_generation, const Gfn2SccIterationDeviceActivity& activity,
    const double* mixed_shell_charges, double* shell_potentials,
    const Gfn2ES2DeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* plan_error, cudaStream_t stream) noexcept {
  CommonBytes bytes{};
  std::size_t system_error_bytes = 0;
  cudaError_t status = validate_scc_control(batch, cache, geometry_generation, activity,
                                            system_errors, plan_error, &bytes, &system_error_bytes);
  if (status != cudaSuccess || mixed_shell_charges == nullptr || shell_potentials == nullptr ||
      workspace.shell_scratch == nullptr || workspace.shell_elements < batch.total_shells ||
      !is_aligned(mixed_shell_charges, alignof(double)) ||
      !is_aligned(shell_potentials, alignof(double)) ||
      !is_aligned(workspace.shell_scratch, alignof(double))) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  const MemoryRange ranges[]{{batch.atom_offsets, bytes.atom_offsets},
                             {batch.batch_shell_offsets, bytes.batch_shell_offsets},
                             {batch.atom_shell_offsets, bytes.atom_shell_offsets},
                             {batch.matrix_offsets, bytes.matrix_offsets},
                             {batch.shell_to_atom, bytes.shell_to_atom},
                             {batch.shell_hardness, bytes.shell_hardness},
                             {cache.coulomb_matrix, bytes.matrix},
                             {activity.active_mask, static_cast<std::size_t>(batch.batch_size)},
                             {activity.sequence_active, sizeof(*activity.sequence_active)},
                             {system_errors, system_error_bytes},
                             {plan_error, sizeof(*plan_error)},
                             {mixed_shell_charges, bytes.shells},
                             {shell_potentials, bytes.shells},
                             {workspace.shell_scratch, bytes.shells}};
  if (!pairwise_disjoint(ranges)) {
    return cudaErrorInvalidValue;
  }
  es2_scc_plan_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
      batch, cache, geometry_generation, activity, plan_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  es2_scc_potential_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size),
                                       kThreadsPerBlock, 0, stream>>>(
      batch, cache, activity, mixed_shell_charges, workspace.shell_scratch, system_errors,
      plan_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  es2_scc_publish_potential_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock,
                                     0, stream>>>(batch, activity, workspace.shell_scratch,
                                                  shell_potentials, system_errors, plan_error);
  return check_launch();
}

cudaError_t evaluate_gfn2_es2_scc_energy_cuda(
    const Gfn2ES2DeviceBatch& batch, const Gfn2ES2DeviceCache& cache,
    std::uint64_t geometry_generation, const Gfn2SccIterationDeviceActivity& activity,
    const double* raw_shell_charges, double* component_energies,
    const Gfn2ES2DeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* plan_error, cudaStream_t stream) noexcept {
  CommonBytes bytes{};
  std::size_t system_error_bytes = 0;
  cudaError_t status = validate_scc_control(batch, cache, geometry_generation, activity,
                                            system_errors, plan_error, &bytes, &system_error_bytes);
  if (status != cudaSuccess || raw_shell_charges == nullptr || component_energies == nullptr ||
      workspace.batch_scratch == nullptr || workspace.batch_elements < batch.batch_size ||
      !is_aligned(raw_shell_charges, alignof(double)) ||
      !is_aligned(component_energies, alignof(double)) ||
      !is_aligned(workspace.batch_scratch, alignof(double))) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  const MemoryRange ranges[]{{batch.atom_offsets, bytes.atom_offsets},
                             {batch.batch_shell_offsets, bytes.batch_shell_offsets},
                             {batch.atom_shell_offsets, bytes.atom_shell_offsets},
                             {batch.matrix_offsets, bytes.matrix_offsets},
                             {batch.shell_to_atom, bytes.shell_to_atom},
                             {batch.shell_hardness, bytes.shell_hardness},
                             {cache.coulomb_matrix, bytes.matrix},
                             {activity.active_mask, static_cast<std::size_t>(batch.batch_size)},
                             {activity.sequence_active, sizeof(*activity.sequence_active)},
                             {system_errors, system_error_bytes},
                             {plan_error, sizeof(*plan_error)},
                             {raw_shell_charges, bytes.shells},
                             {component_energies, bytes.batch},
                             {workspace.batch_scratch, bytes.batch}};
  if (!pairwise_disjoint(ranges)) {
    return cudaErrorInvalidValue;
  }
  es2_scc_plan_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
      batch, cache, geometry_generation, activity, plan_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  es2_scc_energy_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock,
                                    0, stream>>>(batch, cache, activity, raw_shell_charges,
                                                 workspace.batch_scratch, system_errors,
                                                 plan_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  es2_scc_publish_energy_kernel<<<static_cast<unsigned int>(batch.batch_size), 1, 0, stream>>>(
      activity, workspace.batch_scratch, component_energies, system_errors, plan_error);
  return check_launch();
}

}  // namespace xtbloom::detail::cuda
