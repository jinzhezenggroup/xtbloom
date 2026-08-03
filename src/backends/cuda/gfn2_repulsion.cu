#include <cmath>
#include <cstdint>
#include <limits>

#include "backends/cuda/cuda_atomics.cuh"
#include "backends/cuda/gfn2_parameters.cuh"
#include "backends/cuda/gfn2_repulsion.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr double kCutoffSquaredBohr = 25.0 * 25.0;
constexpr double kMinimumDistanceSquared = 1.0e-24;

__device__ void record_error(std::uint32_t* device_error, Gfn2RepulsionDeviceError error) {
  atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2RepulsionDeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__global__ void gfn2_repulsion_kernel(Gfn2RepulsionDeviceBatch batch, double* energies,
                                      double* forces, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ std::int64_t begin;
  __shared__ std::int64_t end;
  __shared__ int valid;
  __shared__ double energy_sums[kThreadsPerBlock];

  if (threadIdx.x == 0) {
    begin = batch.atom_offsets[system];
    end = batch.atom_offsets[system + 1];
    valid = begin >= 0 && begin <= end && end <= batch.total_atoms && (system != 0 || begin == 0) &&
            (system + 1 != batch.batch_size || end == batch.total_atoms);
    if (valid == 0) {
      record_error(device_error, Gfn2RepulsionDeviceError::kInvalidOffsets);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  /*
   * Validation is a separate phase so every atom is checked, including atoms
   * in empty/singleton systems and atoms whose pairs lie outside the cutoff.
   */
  for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
    const std::int32_t atomic_number = batch.atomic_numbers[atom];
    if (atomic_number < 1 ||
        atomic_number > static_cast<std::int32_t>(parameters::gfn2::kElementCount)) {
      record_error(device_error, Gfn2RepulsionDeviceError::kInvalidAtomicNumberOrParameter);
      atomicExch(&valid, 0);
      continue;
    }

    const parameters::gfn2::ElementParameters element = g_gfn2_elements[atomic_number - 1];
    if (element.atomic_number != atomic_number || !(element.arep > 0.0) || !(element.zeff > 0.0) ||
        !isfinite(element.arep) || !isfinite(element.zeff)) {
      record_error(device_error, Gfn2RepulsionDeviceError::kInvalidAtomicNumberOrParameter);
      atomicExch(&valid, 0);
    }

    const std::int64_t coordinate = atom * 3;
    if (!isfinite(batch.positions[coordinate]) || !isfinite(batch.positions[coordinate + 1]) ||
        !isfinite(batch.positions[coordinate + 2])) {
      record_error(device_error, Gfn2RepulsionDeviceError::kNonfinitePosition);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  double local_energy = 0.0;
  for (std::int64_t first = begin + 1 + threadIdx.x; first < end; first += blockDim.x) {
    const std::int32_t first_atomic_number = batch.atomic_numbers[first];
    const parameters::gfn2::ElementParameters first_element =
        g_gfn2_elements[first_atomic_number - 1];
    const double first_sqrt_alpha = sqrt(first_element.arep);
    const std::int64_t first_coordinate = first * 3;
    const double first_x = batch.positions[first_coordinate];
    const double first_y = batch.positions[first_coordinate + 1];
    const double first_z = batch.positions[first_coordinate + 2];
    double first_fx = 0.0;
    double first_fy = 0.0;
    double first_fz = 0.0;

    for (std::int64_t second = begin; second < first; ++second) {
      const std::int64_t second_coordinate = second * 3;
      const double dx = first_x - batch.positions[second_coordinate];
      const double dy = first_y - batch.positions[second_coordinate + 1];
      const double dz = first_z - batch.positions[second_coordinate + 2];
      const double distance_squared = dx * dx + dy * dy + dz * dz;
      if (distance_squared <= kMinimumDistanceSquared) {
        record_error(device_error, Gfn2RepulsionDeviceError::kCoincidentAtoms);
        continue;
      }
      if (distance_squared > kCutoffSquaredBohr) {
        continue;
      }

      const std::int32_t second_atomic_number = batch.atomic_numbers[second];
      const parameters::gfn2::ElementParameters second_element =
          g_gfn2_elements[second_atomic_number - 1];
      const double distance = sqrt(distance_squared);
      const bool light_pair = first_atomic_number <= 2 && second_atomic_number <= 2;
      const double exponent =
          light_pair ? g_gfn2_global.repulsion_klight : g_gfn2_global.repulsion_kexp;
      /* Keep this branch identical to the CPU reference's arithmetic. */
      const double distance_power = light_pair ? distance : distance * sqrt(distance);
      const double pair_alpha = first_sqrt_alpha * sqrt(second_element.arep);
      const double pair_charge = first_element.zeff * second_element.zeff;
      const double pair_energy = pair_charge * exp(-pair_alpha * distance_power) / distance;
      local_energy += pair_energy;

      if (forces != nullptr) {
        const double force_scale =
            (pair_alpha * exponent * distance_power + 1.0) * pair_energy / distance_squared;
        const double fx = force_scale * dx;
        const double fy = force_scale * dy;
        const double fz = force_scale * dz;
        first_fx += fx;
        first_fy += fy;
        first_fz += fz;
        atomic_add_fp64(&forces[second_coordinate], -fx);
        atomic_add_fp64(&forces[second_coordinate + 1], -fy);
        atomic_add_fp64(&forces[second_coordinate + 2], -fz);
      }
    }

    if (forces != nullptr) {
      atomic_add_fp64(&forces[first_coordinate], first_fx);
      atomic_add_fp64(&forces[first_coordinate + 1], first_fy);
      atomic_add_fp64(&forces[first_coordinate + 2], first_fz);
    }
  }

  energy_sums[threadIdx.x] = local_energy;
  __syncthreads();
  for (int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      energy_sums[threadIdx.x] += energy_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomic_add_fp64(&energies[system], energy_sums[0]);
  }
}

}  // namespace

cudaError_t add_gfn2_repulsion_cuda(const Gfn2RepulsionDeviceBatch& batch, double* energies,
                                    double* forces, std::uint32_t* device_error,
                                    cudaStream_t stream) noexcept {
  if (batch.batch_size <= 0 || batch.total_atoms <= 0 || batch.atom_offsets == nullptr ||
      batch.atomic_numbers == nullptr || batch.positions == nullptr || energies == nullptr ||
      device_error == nullptr) {
    return cudaErrorInvalidValue;
  }
  /* CUDA devices expose at most INT_MAX blocks in grid.x. */
  if (static_cast<std::uint64_t>(batch.batch_size) >
      static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return cudaErrorInvalidConfiguration;
  }
  if (batch.total_atoms > std::numeric_limits<std::int64_t>::max() / 3) {
    return cudaErrorInvalidValue;
  }

  const cudaError_t reset_status = cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream);
  if (reset_status != cudaSuccess) {
    return reset_status;
  }

  gfn2_repulsion_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                          stream>>>(batch, energies, forces, device_error);
  return cudaGetLastError();
}

}  // namespace gpuxtb::detail::cuda
