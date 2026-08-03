#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_external_point_charges.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;

__device__ void record_error(std::uint32_t* device_error,
                             Gfn2ExternalPointChargeDeviceError error) {
  atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2ExternalPointChargeDeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

/* Atomically accumulate without ever publishing a nonfinite output value. */
__device__ bool checked_atomic_add(double* address, double increment, std::uint32_t* device_error) {
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long observed = atomicCAS(bits, 0ULL, 0ULL);
  while (true) {
    const double current = __longlong_as_double(static_cast<long long>(observed));
    const double updated = current + increment;
    if (!isfinite(current) || !isfinite(updated)) {
      record_error(device_error, Gfn2ExternalPointChargeDeviceError::kNonfinitePairArithmetic);
      return false;
    }
    const unsigned long long desired =
        static_cast<unsigned long long>(__double_as_longlong(updated));
    const unsigned long long previous = atomicCAS(bits, observed, desired);
    if (previous == observed) {
      return true;
    }
    observed = previous;
  }
}

struct SystemRanges {
  std::int64_t atom_begin;
  std::int64_t atom_end;
  std::int64_t shell_begin;
  std::int64_t shell_end;
  std::int64_t point_begin;
  std::int64_t point_end;
};

/*
 * Validate the plan portion needed by all three stages before any result for
 * this system is written. The shared valid flag gives each system a local
 * failure boundary while device_error records the first batch-wide reason.
 */
__device__ void load_and_validate_ranges(const Gfn2ExternalPointChargeDeviceBatch& batch,
                                         std::int64_t system, SystemRanges* ranges, int* valid,
                                         std::uint32_t* device_error) {
  if (threadIdx.x == 0) {
    /* Preserve an upstream failure and avoid dereferencing any dependent input. */
    if (atomicAdd(device_error, 0u) !=
        static_cast<std::uint32_t>(Gfn2ExternalPointChargeDeviceError::kSuccess)) {
      *valid = 0;
    } else {
      ranges->atom_begin = batch.atom_offsets[system];
      ranges->atom_end = batch.atom_offsets[system + 1];
      ranges->shell_begin = batch.batch_shell_offsets[system];
      ranges->shell_end = batch.batch_shell_offsets[system + 1];
      ranges->point_begin = batch.point_charge_offsets[system];
      ranges->point_end = batch.point_charge_offsets[system + 1];
      *valid =
          ranges->atom_begin >= 0 && ranges->atom_begin <= ranges->atom_end &&
          ranges->atom_end <= batch.total_atoms && ranges->shell_begin >= 0 &&
          ranges->shell_begin <= ranges->shell_end && ranges->shell_end <= batch.total_shells &&
          ranges->point_begin >= 0 && ranges->point_begin <= ranges->point_end &&
          ranges->point_end <= batch.total_point_charges &&
          (system != 0 ||
           (ranges->atom_begin == 0 && ranges->shell_begin == 0 && ranges->point_begin == 0)) &&
          (system + 1 != batch.batch_size ||
           (ranges->atom_end == batch.total_atoms && ranges->shell_end == batch.total_shells &&
            ranges->point_end == batch.total_point_charges));
      if (*valid == 0) {
        record_error(device_error, Gfn2ExternalPointChargeDeviceError::kInvalidOffsets);
      }
    }
  }
  __syncthreads();
  if (*valid == 0) {
    return;
  }

  for (std::int64_t shell = ranges->shell_begin + threadIdx.x; shell < ranges->shell_end;
       shell += blockDim.x) {
    const std::int64_t atom = batch.shell_to_atom[shell];
    const double hardness = batch.shell_hardness[shell];
    if (atom < ranges->atom_begin || atom >= ranges->atom_end || !(hardness > 0.0) ||
        !isfinite(hardness)) {
      record_error(device_error, Gfn2ExternalPointChargeDeviceError::kInvalidShellMetadata);
      atomicExch(valid, 0);
    }
  }
  __syncthreads();
}

__device__ void validate_geometry(const Gfn2ExternalPointChargeDeviceBatch& batch,
                                  const SystemRanges& ranges, int* valid,
                                  std::uint32_t* device_error) {
  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    if (!isfinite(batch.qm_positions[coordinate]) ||
        !isfinite(batch.qm_positions[coordinate + 1]) ||
        !isfinite(batch.qm_positions[coordinate + 2])) {
      record_error(device_error, Gfn2ExternalPointChargeDeviceError::kNonfiniteQmPosition);
      atomicExch(valid, 0);
    }
  }
  for (std::int64_t point = ranges.point_begin + threadIdx.x; point < ranges.point_end;
       point += blockDim.x) {
    const std::int64_t coordinate = point * 3;
    const double hardness = batch.point_hardnesses[point];
    if (!isfinite(batch.point_positions[coordinate]) ||
        !isfinite(batch.point_positions[coordinate + 1]) ||
        !isfinite(batch.point_positions[coordinate + 2]) || !isfinite(batch.point_charges[point]) ||
        !(hardness > 0.0) || !isfinite(hardness)) {
      record_error(device_error, Gfn2ExternalPointChargeDeviceError::kInvalidPointChargeInput);
      atomicExch(valid, 0);
    }
  }
  __syncthreads();
}

__global__ void external_point_charge_potential_kernel(Gfn2ExternalPointChargeDeviceBatch batch,
                                                       double* shell_potentials,
                                                       std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  load_and_validate_ranges(batch, system, &ranges, &valid, device_error);
  if (valid == 0) {
    return;
  }
  validate_geometry(batch, ranges, &valid, device_error);
  if (valid == 0) {
    return;
  }

  for (std::int64_t shell = ranges.shell_begin + threadIdx.x; shell < ranges.shell_end;
       shell += blockDim.x) {
    const std::int64_t atom = batch.shell_to_atom[shell];
    const std::int64_t atom_coordinate = atom * 3;
    const double atom_x = batch.qm_positions[atom_coordinate];
    const double atom_y = batch.qm_positions[atom_coordinate + 1];
    const double atom_z = batch.qm_positions[atom_coordinate + 2];
    const double shell_hardness = batch.shell_hardness[shell];
    double potential = 0.0;
    bool finite_result = true;
    for (std::int64_t point = ranges.point_begin; point < ranges.point_end; ++point) {
      const std::int64_t point_coordinate = point * 3;
      const double dx = atom_x - batch.point_positions[point_coordinate];
      const double dy = atom_y - batch.point_positions[point_coordinate + 1];
      const double dz = atom_z - batch.point_positions[point_coordinate + 2];
      if (!isfinite(dx) || !isfinite(dy) || !isfinite(dz)) {
        finite_result = false;
        break;
      }
      const double inverse_average_hardness =
          2.0 / (shell_hardness + batch.point_hardnesses[point]);
      /* Match the nested-hypot CPU reference, including extreme hardnesses. */
      const double softened_distance = hypot(hypot(dx, dy), hypot(dz, inverse_average_hardness));
      const double contribution = batch.point_charges[point] / softened_distance;
      const double updated_potential = potential + contribution;
      if (!(softened_distance > 0.0) || !isfinite(softened_distance) || !isfinite(contribution) ||
          !isfinite(updated_potential)) {
        finite_result = false;
        break;
      }
      potential = updated_potential;
    }
    if (!finite_result) {
      record_error(device_error, Gfn2ExternalPointChargeDeviceError::kNonfinitePairArithmetic);
    } else {
      shell_potentials[shell] = potential;
    }
  }
}

__global__ void external_point_charge_energy_kernel(Gfn2ExternalPointChargeDeviceBatch batch,
                                                    const double* shell_charges,
                                                    const double* shell_potentials,
                                                    double* energies, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  __shared__ double energy_sums[kThreadsPerBlock];
  load_and_validate_ranges(batch, system, &ranges, &valid, device_error);
  if (valid == 0) {
    return;
  }

  double local_energy = 0.0;
  for (std::int64_t shell = ranges.shell_begin + threadIdx.x; shell < ranges.shell_end;
       shell += blockDim.x) {
    const double charge = shell_charges[shell];
    const double potential = shell_potentials[shell];
    if (!isfinite(charge) || !isfinite(potential)) {
      record_error(device_error, Gfn2ExternalPointChargeDeviceError::kNonfiniteShellValue);
      atomicExch(&valid, 0);
    } else {
      const double contribution = charge * potential;
      const double updated_energy = local_energy + contribution;
      if (!isfinite(contribution) || !isfinite(updated_energy)) {
        record_error(device_error, Gfn2ExternalPointChargeDeviceError::kNonfinitePairArithmetic);
        atomicExch(&valid, 0);
      } else {
        local_energy = updated_energy;
      }
    }
  }
  energy_sums[threadIdx.x] = local_energy;
  __syncthreads();
  if (valid == 0) {
    return;
  }

  for (int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      const double combined = energy_sums[threadIdx.x] + energy_sums[threadIdx.x + stride];
      if (!isfinite(combined)) {
        record_error(device_error, Gfn2ExternalPointChargeDeviceError::kNonfinitePairArithmetic);
        atomicExch(&valid, 0);
      } else {
        energy_sums[threadIdx.x] = combined;
      }
    }
    __syncthreads();
    if (valid == 0) {
      return;
    }
  }
  if (threadIdx.x == 0) {
    /* One block owns each system, so the final checked accumulation is race-free. */
    const double current_energy = energies[system];
    const double updated_energy = current_energy + energy_sums[0];
    if (!isfinite(current_energy) || !isfinite(updated_energy)) {
      record_error(device_error, Gfn2ExternalPointChargeDeviceError::kNonfinitePairArithmetic);
    } else {
      energies[system] = updated_energy;
    }
  }
}

__device__ void record_pc_scc_plan_error(std::uint32_t* plan_error,
                                         Gfn2ExternalPointChargeDeviceError error) {
  atomicCAS(plan_error, static_cast<std::uint32_t>(Gfn2ExternalPointChargeDeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ void record_pc_scc_system_error(std::uint32_t* system_errors, std::int64_t system,
                                           Gfn2ExternalPointChargeDeviceError error) {
  atomicCAS(system_errors + system,
            static_cast<std::uint32_t>(Gfn2ExternalPointChargeDeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ bool pc_scc_member_is_active(const Gfn2SccIterationDeviceActivity& activity,
                                        const std::uint32_t* system_errors,
                                        const std::uint32_t* plan_error, std::int64_t system) {
  if (atomicAdd(const_cast<std::uint32_t*>(activity.sequence_active), 0u) != 1u ||
      activity.active_mask[system] != 1u) {
    return false;
  }
  return atomicAdd(const_cast<std::uint32_t*>(plan_error), 0u) == 0u &&
         atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) == 0u;
}

__global__ void pc_scc_plan_preflight_kernel(Gfn2ExternalPointChargeDeviceBatch batch,
                                             Gfn2ExternalPointChargeDeviceCache cache,
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
      (batch.plan_token == 0u || cache.plan_token != batch.plan_token ||
       geometry_generation == 0u || cache.geometry_generation != geometry_generation)) {
    record_pc_scc_plan_error(plan_error, Gfn2ExternalPointChargeDeviceError::kCacheMismatch);
  }
  __syncthreads();
  if (atomicAdd(plan_error, 0u) != 0u) {
    return;
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    if (activity.active_mask[system] != 1u) {
      continue;
    }
    const std::int64_t atom_begin = batch.atom_offsets[system];
    const std::int64_t atom_end = batch.atom_offsets[system + 1];
    const std::int64_t shell_begin = batch.batch_shell_offsets[system];
    const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
    const std::int64_t point_begin = batch.point_charge_offsets[system];
    const std::int64_t point_end = batch.point_charge_offsets[system + 1];
    bool valid = atom_begin >= 0 && atom_begin <= atom_end && atom_end <= batch.total_atoms &&
                 shell_begin >= 0 && shell_begin <= shell_end && shell_end <= batch.total_shells &&
                 point_begin >= 0 && point_begin <= point_end &&
                 point_end <= batch.total_point_charges;
    if (valid && system == 0) {
      valid = atom_begin == 0 && shell_begin == 0 && point_begin == 0;
    }
    if (valid && system + 1 == batch.batch_size) {
      valid = atom_end == batch.total_atoms && shell_end == batch.total_shells &&
              point_end == batch.total_point_charges;
    }
    if (!valid) {
      record_pc_scc_plan_error(plan_error, Gfn2ExternalPointChargeDeviceError::kInvalidOffsets);
    }
  }
}

__global__ void pc_scc_potential_cache_kernel(Gfn2ExternalPointChargeDeviceBatch batch,
                                              Gfn2SccIterationDeviceActivity activity,
                                              double* shell_scratch, double* shell_cache,
                                              std::uint32_t* system_errors,
                                              const std::uint32_t* plan_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int active;
  __shared__ int valid;
  if (threadIdx.x == 0) {
    active = pc_scc_member_is_active(activity, system_errors, plan_error, system) ? 1 : 0;
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
  const std::int64_t point_begin = batch.point_charge_offsets[system];
  const std::int64_t point_end = batch.point_charge_offsets[system + 1];
  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    const std::int64_t atom = batch.shell_to_atom[shell];
    const double hardness = batch.shell_hardness[shell];
    if (atom < atom_begin || atom >= atom_end || !(hardness > 0.0) || !isfinite(hardness)) {
      record_pc_scc_system_error(system_errors, system,
                                 Gfn2ExternalPointChargeDeviceError::kInvalidShellMetadata);
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    if (!isfinite(batch.qm_positions[coordinate]) ||
        !isfinite(batch.qm_positions[coordinate + 1]) ||
        !isfinite(batch.qm_positions[coordinate + 2])) {
      record_pc_scc_system_error(system_errors, system,
                                 Gfn2ExternalPointChargeDeviceError::kNonfiniteQmPosition);
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t point = point_begin + threadIdx.x; point < point_end; point += blockDim.x) {
    const std::int64_t coordinate = point * 3;
    const double hardness = batch.point_hardnesses[point];
    if (!isfinite(batch.point_positions[coordinate]) ||
        !isfinite(batch.point_positions[coordinate + 1]) ||
        !isfinite(batch.point_positions[coordinate + 2]) || !isfinite(batch.point_charges[point]) ||
        !(hardness > 0.0) || !isfinite(hardness)) {
      record_pc_scc_system_error(system_errors, system,
                                 Gfn2ExternalPointChargeDeviceError::kInvalidPointChargeInput);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    const std::int64_t atom = batch.shell_to_atom[shell];
    const std::int64_t atom_coordinate = atom * 3;
    const double atom_x = batch.qm_positions[atom_coordinate];
    const double atom_y = batch.qm_positions[atom_coordinate + 1];
    const double atom_z = batch.qm_positions[atom_coordinate + 2];
    const double shell_hardness = batch.shell_hardness[shell];
    double potential = 0.0;
    for (std::int64_t point = point_begin; point < point_end; ++point) {
      const std::int64_t point_coordinate = point * 3;
      const double dx = atom_x - batch.point_positions[point_coordinate];
      const double dy = atom_y - batch.point_positions[point_coordinate + 1];
      const double dz = atom_z - batch.point_positions[point_coordinate + 2];
      const double inverse_average_hardness =
          2.0 / (shell_hardness + batch.point_hardnesses[point]);
      const double distance = hypot(hypot(dx, dy), hypot(dz, inverse_average_hardness));
      const double contribution = batch.point_charges[point] / distance;
      const double updated = potential + contribution;
      if (!isfinite(dx) || !isfinite(dy) || !isfinite(dz) || !(distance > 0.0) ||
          !isfinite(distance) || !isfinite(contribution) || !isfinite(updated)) {
        record_pc_scc_system_error(system_errors, system,
                                   Gfn2ExternalPointChargeDeviceError::kNonfinitePairArithmetic);
        atomicExch(&valid, 0);
        break;
      }
      potential = updated;
    }
    shell_scratch[shell] = potential;
  }
  __syncthreads();
  if (valid == 0 || atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) != 0u) {
    return;
  }
  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    shell_cache[shell] = shell_scratch[shell];
  }
}

__global__ void pc_scc_energy_kernel(Gfn2ExternalPointChargeDeviceBatch batch,
                                     Gfn2SccIterationDeviceActivity activity,
                                     const double* shell_charges, const double* shell_potentials,
                                     double* component_energies, std::uint32_t* system_errors,
                                     const std::uint32_t* plan_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (threadIdx.x != 0 || !pc_scc_member_is_active(activity, system_errors, plan_error, system)) {
    return;
  }
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  double energy = 0.0;
  for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
    const double charge = shell_charges[shell];
    const double potential = shell_potentials[shell];
    if (!isfinite(charge) || !isfinite(potential)) {
      record_pc_scc_system_error(system_errors, system,
                                 Gfn2ExternalPointChargeDeviceError::kNonfiniteShellValue);
      return;
    }
    const double updated = fma(charge, potential, energy);
    if (!isfinite(updated)) {
      record_pc_scc_system_error(system_errors, system,
                                 Gfn2ExternalPointChargeDeviceError::kNonfinitePairArithmetic);
      return;
    }
    energy = updated;
  }
  component_energies[system] = energy;
}

__global__ void external_point_charge_force_kernel(Gfn2ExternalPointChargeDeviceBatch batch,
                                                   const double* shell_charges, double* qm_forces,
                                                   double* point_forces,
                                                   std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  load_and_validate_ranges(batch, system, &ranges, &valid, device_error);
  if (valid == 0) {
    return;
  }
  validate_geometry(batch, ranges, &valid, device_error);
  if (valid == 0) {
    return;
  }

  for (std::int64_t shell = ranges.shell_begin + threadIdx.x; shell < ranges.shell_end;
       shell += blockDim.x) {
    if (!isfinite(shell_charges[shell])) {
      record_error(device_error, Gfn2ExternalPointChargeDeviceError::kNonfiniteShellValue);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  for (std::int64_t shell = ranges.shell_begin + threadIdx.x; shell < ranges.shell_end;
       shell += blockDim.x) {
    const std::int64_t atom = batch.shell_to_atom[shell];
    const std::int64_t atom_coordinate = atom * 3;
    const double atom_x = batch.qm_positions[atom_coordinate];
    const double atom_y = batch.qm_positions[atom_coordinate + 1];
    const double atom_z = batch.qm_positions[atom_coordinate + 2];
    const double shell_hardness = batch.shell_hardness[shell];
    const double shell_charge = shell_charges[shell];
    double atom_fx = 0.0;
    double atom_fy = 0.0;
    double atom_fz = 0.0;
    bool finite_result = true;

    for (std::int64_t point = ranges.point_begin; point < ranges.point_end; ++point) {
      const std::int64_t point_coordinate = point * 3;
      const double dx = atom_x - batch.point_positions[point_coordinate];
      const double dy = atom_y - batch.point_positions[point_coordinate + 1];
      const double dz = atom_z - batch.point_positions[point_coordinate + 2];
      if (!isfinite(dx) || !isfinite(dy) || !isfinite(dz)) {
        finite_result = false;
        break;
      }
      if (dx == 0.0 && dy == 0.0 && dz == 0.0) {
        continue;
      }
      const double inverse_average_hardness =
          2.0 / (shell_hardness + batch.point_hardnesses[point]);
      const double softened_distance = hypot(hypot(dx, dy), hypot(dz, inverse_average_hardness));
      const double inverse_distance = 1.0 / softened_distance;
      const double force_scale = shell_charge * batch.point_charges[point] * inverse_distance *
                                 inverse_distance * inverse_distance;
      const double fx = force_scale * dx;
      const double fy = force_scale * dy;
      const double fz = force_scale * dz;
      const double updated_fx = atom_fx + fx;
      const double updated_fy = atom_fy + fy;
      const double updated_fz = atom_fz + fz;
      if (!(softened_distance > 0.0) || !isfinite(softened_distance) ||
          !isfinite(inverse_distance) || !isfinite(force_scale) || !isfinite(fx) || !isfinite(fy) ||
          !isfinite(fz) || !isfinite(updated_fx) || !isfinite(updated_fy) ||
          !isfinite(updated_fz)) {
        finite_result = false;
        break;
      }
      atom_fx = updated_fx;
      atom_fy = updated_fy;
      atom_fz = updated_fz;
      if (point_forces != nullptr) {
        if (!checked_atomic_add(&point_forces[point_coordinate], -fx, device_error) ||
            !checked_atomic_add(&point_forces[point_coordinate + 1], -fy, device_error) ||
            !checked_atomic_add(&point_forces[point_coordinate + 2], -fz, device_error)) {
          finite_result = false;
          break;
        }
      }
    }

    if (!finite_result) {
      record_error(device_error, Gfn2ExternalPointChargeDeviceError::kNonfinitePairArithmetic);
    } else if (qm_forces != nullptr) {
      (void)(checked_atomic_add(&qm_forces[atom_coordinate], atom_fx, device_error) &&
             checked_atomic_add(&qm_forces[atom_coordinate + 1], atom_fy, device_error) &&
             checked_atomic_add(&qm_forces[atom_coordinate + 2], atom_fz, device_error));
    }
  }
}

cudaError_t validate_common_launcher_arguments(const Gfn2ExternalPointChargeDeviceBatch& batch,
                                               std::uint32_t* device_error) noexcept {
  if (batch.batch_size <= 0 || batch.total_atoms <= 0 || batch.total_shells <= 0 ||
      batch.total_point_charges < 0 || batch.atom_offsets == nullptr ||
      batch.batch_shell_offsets == nullptr || batch.point_charge_offsets == nullptr ||
      batch.shell_to_atom == nullptr || batch.shell_hardness == nullptr ||
      device_error == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (static_cast<std::uint64_t>(batch.batch_size) >
      static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return cudaErrorInvalidConfiguration;
  }
  if (batch.total_atoms > std::numeric_limits<std::int64_t>::max() / 3 ||
      batch.total_point_charges > std::numeric_limits<std::int64_t>::max() / 3) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

bool count_bytes(std::int64_t count, std::size_t element_size, std::size_t* bytes) noexcept {
  if (count < 0 ||
      static_cast<std::uint64_t>(count) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
    return false;
  }
  *bytes = static_cast<std::size_t>(count) * element_size;
  return true;
}

bool ranges_overlap(const void* first, std::size_t first_bytes, const void* second,
                    std::size_t second_bytes) noexcept {
  if (first == nullptr || second == nullptr || first_bytes == 0u || second_bytes == 0u) {
    return false;
  }
  const std::uintptr_t first_begin = reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t second_begin = reinterpret_cast<std::uintptr_t>(second);
  if (first_begin > std::numeric_limits<std::uintptr_t>::max() - first_bytes ||
      second_begin > std::numeric_limits<std::uintptr_t>::max() - second_bytes) {
    return true;
  }
  return first_begin < second_begin + second_bytes && second_begin < first_begin + first_bytes;
}

bool is_aligned(const void* pointer, std::size_t alignment) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

struct MemoryRange {
  const void* data = nullptr;
  std::size_t bytes = 0u;
};

template <std::size_t N, std::size_t M>
bool ranges_disjoint(const MemoryRange (&first)[N], const MemoryRange (&second)[M]) noexcept {
  for (const MemoryRange& left : first) {
    for (const MemoryRange& right : second) {
      if (ranges_overlap(left.data, left.bytes, right.data, right.bytes)) {
        return false;
      }
    }
  }
  return true;
}

template <std::size_t N>
bool pairwise_disjoint(const MemoryRange (&ranges)[N]) noexcept {
  for (std::size_t first = 0u; first < N; ++first) {
    for (std::size_t second = first + 1u; second < N; ++second) {
      if (ranges_overlap(ranges[first].data, ranges[first].bytes, ranges[second].data,
                         ranges[second].bytes)) {
        return false;
      }
    }
  }
  return true;
}

cudaError_t validate_scc_launcher_arguments(const Gfn2ExternalPointChargeDeviceBatch& batch,
                                            const Gfn2SccIterationDeviceActivity& activity,
                                            const Gfn2ExternalPointChargeDeviceCache& cache,
                                            std::uint32_t* system_errors,
                                            std::uint32_t* plan_error) noexcept {
  cudaError_t status = validate_common_launcher_arguments(batch, plan_error);
  std::size_t system_error_bytes = 0u;
  std::size_t shell_bytes = 0u;
  std::size_t offset_bytes = 0u;
  std::size_t shell_index_bytes = 0u;
  if (status != cudaSuccess || batch.plan_token == 0u || activity.active_mask == nullptr ||
      activity.sequence_active == nullptr || activity.batch_elements != batch.batch_size ||
      activity.sequence_elements != 1 || activity.plan_token != batch.plan_token ||
      cache.shell_potentials == nullptr || cache.shell_elements != batch.total_shells ||
      system_errors == nullptr || !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.batch_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.point_charge_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_to_atom, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_hardness, alignof(double)) ||
      !is_aligned(activity.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(cache.shell_potentials, alignof(double)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(plan_error, alignof(std::uint32_t)) ||
      !count_bytes(batch.batch_size, sizeof(std::uint32_t), &system_error_bytes) ||
      !count_bytes(batch.total_shells, sizeof(double), &shell_bytes) ||
      !count_bytes(batch.batch_size + 1, sizeof(std::int64_t), &offset_bytes) ||
      !count_bytes(batch.total_shells, sizeof(std::int64_t), &shell_index_bytes)) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  const MemoryRange controls[]{{activity.active_mask, static_cast<std::size_t>(batch.batch_size)},
                               {activity.sequence_active, sizeof(*activity.sequence_active)},
                               {system_errors, system_error_bytes},
                               {plan_error, sizeof(*plan_error)}};
  const MemoryRange common_reads[]{
      {batch.atom_offsets, offset_bytes},         {batch.batch_shell_offsets, offset_bytes},
      {batch.point_charge_offsets, offset_bytes}, {batch.shell_to_atom, shell_index_bytes},
      {batch.shell_hardness, shell_bytes},        {cache.shell_potentials, shell_bytes}};
  if (!pairwise_disjoint(controls) || !ranges_disjoint(controls, common_reads)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

}  // namespace

cudaError_t reset_gfn2_external_point_charge_device_error_cuda(std::uint32_t* device_error,
                                                               cudaStream_t stream) noexcept {
  if (device_error == nullptr) {
    return cudaErrorInvalidValue;
  }
  return cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream);
}

cudaError_t reset_gfn2_external_point_charge_scc_errors_cuda(std::int64_t batch_size,
                                                             std::uint32_t* system_errors,
                                                             std::uint32_t* plan_error,
                                                             cudaStream_t stream) noexcept {
  std::size_t system_error_bytes = 0u;
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

cudaError_t evaluate_gfn2_external_point_charge_potential_cuda(
    const Gfn2ExternalPointChargeDeviceBatch& batch, double* shell_potentials,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  cudaError_t status = validate_common_launcher_arguments(batch, device_error);
  if (status != cudaSuccess || batch.qm_positions == nullptr || shell_potentials == nullptr ||
      (batch.total_point_charges != 0 &&
       (batch.point_positions == nullptr || batch.point_charges == nullptr ||
        batch.point_hardnesses == nullptr))) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  external_point_charge_potential_kernel<<<static_cast<unsigned int>(batch.batch_size),
                                           kThreadsPerBlock, 0, stream>>>(batch, shell_potentials,
                                                                          device_error);
  return cudaGetLastError();
}

cudaError_t add_gfn2_external_point_charge_energy_cuda(
    const Gfn2ExternalPointChargeDeviceBatch& batch, const double* shell_charges,
    const double* shell_potentials, double* energies, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  cudaError_t status = validate_common_launcher_arguments(batch, device_error);
  if (status != cudaSuccess || shell_charges == nullptr || shell_potentials == nullptr ||
      energies == nullptr) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  external_point_charge_energy_kernel<<<static_cast<unsigned int>(batch.batch_size),
                                        kThreadsPerBlock, 0, stream>>>(
      batch, shell_charges, shell_potentials, energies, device_error);
  return cudaGetLastError();
}

cudaError_t add_gfn2_external_point_charge_forces_cuda(
    const Gfn2ExternalPointChargeDeviceBatch& batch, const double* shell_charges, double* qm_forces,
    double* point_forces, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  cudaError_t status = validate_common_launcher_arguments(batch, device_error);
  if (status != cudaSuccess || batch.qm_positions == nullptr || shell_charges == nullptr ||
      (qm_forces == nullptr && point_forces == nullptr) ||
      (batch.total_point_charges != 0 &&
       (batch.point_positions == nullptr || batch.point_charges == nullptr ||
        batch.point_hardnesses == nullptr))) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  external_point_charge_force_kernel<<<static_cast<unsigned int>(batch.batch_size),
                                       kThreadsPerBlock, 0, stream>>>(
      batch, shell_charges, qm_forces, point_forces, device_error);
  return cudaGetLastError();
}

cudaError_t update_gfn2_external_point_charge_scc_potential_cache_cuda(
    const Gfn2ExternalPointChargeDeviceBatch& batch, const Gfn2SccIterationDeviceActivity& activity,
    const Gfn2ExternalPointChargeDeviceCache& cache, std::uint64_t geometry_generation,
    const Gfn2ExternalPointChargeDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* plan_error, cudaStream_t stream) noexcept {
  cudaError_t status =
      validate_scc_launcher_arguments(batch, activity, cache, system_errors, plan_error);
  std::size_t shell_bytes = 0u;
  std::size_t atom_coordinate_bytes = 0u;
  std::size_t point_coordinate_bytes = 0u;
  std::size_t point_bytes = 0u;
  std::size_t system_error_bytes = 0u;
  std::size_t offset_bytes = 0u;
  std::size_t shell_index_bytes = 0u;
  if (status != cudaSuccess || batch.qm_positions == nullptr ||
      workspace.shell_scratch == nullptr || workspace.shell_elements < batch.total_shells ||
      (batch.total_point_charges != 0 &&
       (batch.point_positions == nullptr || batch.point_charges == nullptr ||
        batch.point_hardnesses == nullptr)) ||
      !is_aligned(batch.qm_positions, alignof(double)) ||
      (batch.total_point_charges != 0 && (!is_aligned(batch.point_positions, alignof(double)) ||
                                          !is_aligned(batch.point_charges, alignof(double)) ||
                                          !is_aligned(batch.point_hardnesses, alignof(double)))) ||
      !is_aligned(workspace.shell_scratch, alignof(double)) ||
      !count_bytes(batch.total_shells, sizeof(double), &shell_bytes) ||
      !count_bytes(batch.total_atoms * 3, sizeof(double), &atom_coordinate_bytes) ||
      !count_bytes(batch.total_point_charges * 3, sizeof(double), &point_coordinate_bytes) ||
      !count_bytes(batch.total_point_charges, sizeof(double), &point_bytes) ||
      !count_bytes(batch.batch_size, sizeof(std::uint32_t), &system_error_bytes) ||
      !count_bytes(batch.batch_size + 1, sizeof(std::int64_t), &offset_bytes) ||
      !count_bytes(batch.total_shells, sizeof(std::int64_t), &shell_index_bytes) ||
      ranges_overlap(workspace.shell_scratch, shell_bytes, cache.shell_potentials, shell_bytes) ||
      ranges_overlap(workspace.shell_scratch, shell_bytes, batch.qm_positions,
                     atom_coordinate_bytes) ||
      ranges_overlap(workspace.shell_scratch, shell_bytes, batch.point_positions,
                     point_coordinate_bytes) ||
      ranges_overlap(workspace.shell_scratch, shell_bytes, batch.point_charges, point_bytes) ||
      ranges_overlap(workspace.shell_scratch, shell_bytes, batch.point_hardnesses, point_bytes) ||
      ranges_overlap(workspace.shell_scratch, shell_bytes, activity.active_mask,
                     static_cast<std::size_t>(batch.batch_size)) ||
      ranges_overlap(workspace.shell_scratch, shell_bytes, activity.sequence_active,
                     sizeof(*activity.sequence_active)) ||
      ranges_overlap(workspace.shell_scratch, shell_bytes, system_errors, system_error_bytes) ||
      ranges_overlap(workspace.shell_scratch, shell_bytes, plan_error, sizeof(*plan_error)) ||
      ranges_overlap(workspace.shell_scratch, shell_bytes, batch.atom_offsets, offset_bytes) ||
      ranges_overlap(workspace.shell_scratch, shell_bytes, batch.batch_shell_offsets,
                     offset_bytes) ||
      ranges_overlap(workspace.shell_scratch, shell_bytes, batch.point_charge_offsets,
                     offset_bytes) ||
      ranges_overlap(workspace.shell_scratch, shell_bytes, batch.shell_to_atom,
                     shell_index_bytes) ||
      ranges_overlap(workspace.shell_scratch, shell_bytes, batch.shell_hardness, shell_bytes) ||
      ranges_overlap(cache.shell_potentials, shell_bytes, system_errors, system_error_bytes) ||
      ranges_overlap(cache.shell_potentials, shell_bytes, plan_error, sizeof(*plan_error)) ||
      ranges_overlap(cache.shell_potentials, shell_bytes, batch.atom_offsets, offset_bytes) ||
      ranges_overlap(cache.shell_potentials, shell_bytes, batch.batch_shell_offsets,
                     offset_bytes) ||
      ranges_overlap(cache.shell_potentials, shell_bytes, batch.point_charge_offsets,
                     offset_bytes) ||
      ranges_overlap(cache.shell_potentials, shell_bytes, batch.shell_to_atom, shell_index_bytes) ||
      ranges_overlap(cache.shell_potentials, shell_bytes, batch.shell_hardness, shell_bytes) ||
      ranges_overlap(cache.shell_potentials, shell_bytes, batch.qm_positions,
                     atom_coordinate_bytes) ||
      ranges_overlap(cache.shell_potentials, shell_bytes, batch.point_positions,
                     point_coordinate_bytes) ||
      ranges_overlap(cache.shell_potentials, shell_bytes, batch.point_charges, point_bytes) ||
      ranges_overlap(cache.shell_potentials, shell_bytes, batch.point_hardnesses, point_bytes)) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  const MemoryRange controls[]{{activity.active_mask, static_cast<std::size_t>(batch.batch_size)},
                               {activity.sequence_active, sizeof(*activity.sequence_active)},
                               {system_errors, system_error_bytes},
                               {plan_error, sizeof(*plan_error)}};
  const MemoryRange geometry_reads[]{{batch.qm_positions, atom_coordinate_bytes},
                                     {batch.point_positions, point_coordinate_bytes},
                                     {batch.point_charges, point_bytes},
                                     {batch.point_hardnesses, point_bytes}};
  if (!ranges_disjoint(controls, geometry_reads)) {
    return cudaErrorInvalidValue;
  }
  pc_scc_plan_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
      batch, cache, geometry_generation, activity, plan_error);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  pc_scc_potential_cache_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                                  stream>>>(batch, activity, workspace.shell_scratch,
                                            cache.shell_potentials, system_errors, plan_error);
  return cudaPeekAtLastError();
}

cudaError_t evaluate_gfn2_external_point_charge_scc_energy_cuda(
    const Gfn2ExternalPointChargeDeviceBatch& batch, const Gfn2SccIterationDeviceActivity& activity,
    const Gfn2ExternalPointChargeDeviceCache& cache, std::uint64_t geometry_generation,
    const double* raw_shell_charges, double* component_energies, std::uint32_t* system_errors,
    std::uint32_t* plan_error, cudaStream_t stream) noexcept {
  cudaError_t status =
      validate_scc_launcher_arguments(batch, activity, cache, system_errors, plan_error);
  std::size_t shell_bytes = 0u;
  std::size_t energy_bytes = 0u;
  std::size_t system_error_bytes = 0u;
  std::size_t offset_bytes = 0u;
  std::size_t shell_index_bytes = 0u;
  if (status != cudaSuccess || raw_shell_charges == nullptr || component_energies == nullptr ||
      !is_aligned(raw_shell_charges, alignof(double)) ||
      !is_aligned(component_energies, alignof(double)) ||
      !count_bytes(batch.total_shells, sizeof(double), &shell_bytes) ||
      !count_bytes(batch.batch_size, sizeof(double), &energy_bytes) ||
      !count_bytes(batch.batch_size, sizeof(std::uint32_t), &system_error_bytes) ||
      !count_bytes(batch.batch_size + 1, sizeof(std::int64_t), &offset_bytes) ||
      !count_bytes(batch.total_shells, sizeof(std::int64_t), &shell_index_bytes) ||
      ranges_overlap(raw_shell_charges, shell_bytes, cache.shell_potentials, shell_bytes) ||
      ranges_overlap(component_energies, energy_bytes, raw_shell_charges, shell_bytes) ||
      ranges_overlap(component_energies, energy_bytes, cache.shell_potentials, shell_bytes) ||
      ranges_overlap(component_energies, energy_bytes, activity.active_mask,
                     static_cast<std::size_t>(batch.batch_size)) ||
      ranges_overlap(component_energies, energy_bytes, activity.sequence_active,
                     sizeof(*activity.sequence_active)) ||
      ranges_overlap(component_energies, energy_bytes, batch.atom_offsets, offset_bytes) ||
      ranges_overlap(component_energies, energy_bytes, batch.batch_shell_offsets, offset_bytes) ||
      ranges_overlap(component_energies, energy_bytes, batch.point_charge_offsets, offset_bytes) ||
      ranges_overlap(component_energies, energy_bytes, batch.shell_to_atom, shell_index_bytes) ||
      ranges_overlap(component_energies, energy_bytes, batch.shell_hardness, shell_bytes) ||
      ranges_overlap(component_energies, energy_bytes, system_errors, system_error_bytes) ||
      ranges_overlap(component_energies, energy_bytes, plan_error, sizeof(*plan_error)) ||
      ranges_overlap(raw_shell_charges, shell_bytes, system_errors, system_error_bytes) ||
      ranges_overlap(raw_shell_charges, shell_bytes, plan_error, sizeof(*plan_error)) ||
      ranges_overlap(raw_shell_charges, shell_bytes, activity.active_mask,
                     static_cast<std::size_t>(batch.batch_size)) ||
      ranges_overlap(raw_shell_charges, shell_bytes, activity.sequence_active,
                     sizeof(*activity.sequence_active))) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  pc_scc_plan_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
      batch, cache, geometry_generation, activity, plan_error);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  pc_scc_energy_kernel<<<static_cast<unsigned int>(batch.batch_size), 1, 0, stream>>>(
      batch, activity, raw_shell_charges, cache.shell_potentials, component_energies, system_errors,
      plan_error);
  return cudaPeekAtLastError();
}

}  // namespace gpuxtb::detail::cuda
