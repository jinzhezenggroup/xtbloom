#include <array>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_density.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr std::int64_t kMaximumInt64 = 9223372036854775807LL;

static_assert((kThreadsPerBlock & (kThreadsPerBlock - 1)) == 0,
              "density reductions require a power-of-two block size");

struct SystemRanges {
  std::int64_t orbital_begin;
  std::int64_t orbital_end;
  std::int64_t matrix_begin;
  std::int64_t matrix_end;
  std::int64_t orbital_count;
};

struct SpinSystemRanges {
  std::int64_t orbital_begin;
  std::int64_t orbital_end;
  std::int64_t matrix_begin;
  std::int64_t matrix_end;
  std::int64_t spin_orbital_begin;
  std::int64_t spin_orbital_end;
  std::int64_t spin_matrix_begin;
  std::int64_t spin_matrix_end;
  std::int64_t channel_begin;
  std::int64_t channel_end;
  std::int64_t orbital_count;
  std::int32_t spin_channels;
};

__device__ bool system_is_valid(const std::uint32_t* system_errors, std::int64_t system) {
  return atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) ==
         static_cast<std::uint32_t>(Gfn2DensityDeviceError::kSuccess);
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    std::uint32_t* device_error, Gfn2DensityDeviceError error) {
  const std::uint32_t success = static_cast<std::uint32_t>(Gfn2DensityDeviceError::kSuccess);
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  if (atomicCAS(system_errors + system, success, code) == success) {
    atomicCAS(device_error, success, code);
  }
}

__global__ void capture_sequence_kernel(const std::uint32_t* device_error,
                                        std::uint32_t* sequence_active) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *sequence_active = atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                               static_cast<std::uint32_t>(Gfn2DensityDeviceError::kSuccess)
                           ? 1u
                           : 0u;
  }
}

__device__ bool load_system(const Gfn2DensityDeviceBatch& batch,
                            const Gfn2DensityDeviceInput& input, std::int64_t system,
                            SystemRanges* ranges, int* active, int* valid,
                            std::uint32_t* system_errors, std::uint32_t* device_error) {
  if (threadIdx.x == 0) {
    *active = 0;
    *valid = 1;
    if (!system_is_valid(system_errors, system)) {
      *valid = 0;
    } else {
      const std::uint8_t state = input.active[system];
      if (state == 0u) {
        *valid = 0;
      } else if (state != 1u) {
        record_system_error(system_errors, system, device_error,
                            Gfn2DensityDeviceError::kInvalidActiveMask);
        *valid = 0;
      } else {
        *active = 1;
        ranges->orbital_begin = batch.orbital_offsets[system];
        ranges->orbital_end = batch.orbital_offsets[system + 1];
        ranges->matrix_begin = batch.matrix_offsets[system];
        ranges->matrix_end = batch.matrix_offsets[system + 1];
        const bool endpoints_valid =
            ranges->orbital_begin >= 0 && ranges->orbital_begin < ranges->orbital_end &&
            ranges->orbital_end <= batch.total_orbitals && ranges->matrix_begin >= 0 &&
            ranges->matrix_begin < ranges->matrix_end &&
            ranges->matrix_end <= batch.total_matrix_elements &&
            (system != 0 || (ranges->orbital_begin == 0 && ranges->matrix_begin == 0)) &&
            (system + 1 != batch.batch_size || (ranges->orbital_end == batch.total_orbitals &&
                                                ranges->matrix_end == batch.total_matrix_elements));
        if (endpoints_valid) {
          ranges->orbital_count = ranges->orbital_end - ranges->orbital_begin;
          const std::int64_t matrix_count = ranges->matrix_end - ranges->matrix_begin;
          *valid = ranges->orbital_count <= kMaximumInt64 / ranges->orbital_count &&
                           matrix_count == ranges->orbital_count * ranges->orbital_count
                       ? 1
                       : 0;
        } else {
          *valid = 0;
        }
        if (*valid == 0) {
          record_system_error(system_errors, system, device_error,
                              Gfn2DensityDeviceError::kInvalidOffsets);
        }
      }
    }
  }
  __syncthreads();
  return *active != 0 && *valid != 0;
}

__global__ void preflight_kernel(Gfn2DensityDeviceBatch batch, Gfn2DensityDeviceInput input,
                                 Gfn2DensityDeviceWorkspace workspace, std::uint32_t* system_errors,
                                 std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int active;
  __shared__ int valid;
  __shared__ double partial_band[kThreadsPerBlock];
  __shared__ double partial_occupation[kThreadsPerBlock];
  if (atomicAdd(workspace.sequence_active, 0u) == 0u ||
      !load_system(batch, input, system, &ranges, &active, &valid, system_errors, device_error)) {
    return;
  }

  for (std::int64_t matrix = ranges.matrix_begin + threadIdx.x; matrix < ranges.matrix_end;
       matrix += blockDim.x) {
    if (!isfinite(input.coefficients[matrix])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2DensityDeviceError::kNonfiniteCoefficient);
      atomicExch(&valid, 0);
    }
  }

  double local_band = 0.0;
  double local_occupation = 0.0;
  for (std::int64_t orbital = ranges.orbital_begin + threadIdx.x; orbital < ranges.orbital_end;
       orbital += blockDim.x) {
    const std::int64_t local = orbital - ranges.orbital_begin;
    const std::int64_t occupation_base = 2 * ranges.orbital_begin;
    const double alpha = input.occupations[occupation_base + local];
    const double beta = input.occupations[occupation_base + ranges.orbital_count + local];
    const double eigenvalue = input.eigenvalues[orbital];
    if (!(alpha >= 0.0 && alpha <= 1.0) || !isfinite(alpha) || !(beta >= 0.0 && beta <= 1.0) ||
        !isfinite(beta)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2DensityDeviceError::kInvalidOccupation);
      atomicExch(&valid, 0);
      continue;
    }
    if (!isfinite(eigenvalue)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2DensityDeviceError::kNonfiniteEigenvalue);
      atomicExch(&valid, 0);
      continue;
    }
    const double weight = alpha + beta;
    const double energy_weight = weight * eigenvalue;
    const double updated_occupation = local_occupation + weight;
    const double updated_band = local_band + energy_weight;
    if (!isfinite(weight) || !isfinite(energy_weight) || !isfinite(updated_occupation)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2DensityDeviceError::kNonfiniteWeightArithmetic);
      atomicExch(&valid, 0);
      continue;
    }
    if (!isfinite(updated_band)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2DensityDeviceError::kNonfiniteBandEnergy);
      atomicExch(&valid, 0);
      continue;
    }
    workspace.weights[orbital] = weight;
    workspace.energy_weights[orbital] = energy_weight;
    local_occupation = updated_occupation;
    local_band = updated_band;
  }
  partial_band[threadIdx.x] = local_band;
  partial_occupation[threadIdx.x] = local_occupation;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      const double band = partial_band[threadIdx.x] + partial_band[threadIdx.x + offset];
      const double occupation =
          partial_occupation[threadIdx.x] + partial_occupation[threadIdx.x + offset];
      if (!isfinite(occupation)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2DensityDeviceError::kNonfiniteWeightArithmetic);
        atomicExch(&valid, 0);
      } else {
        partial_occupation[threadIdx.x] = occupation;
      }
      if (!isfinite(band)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2DensityDeviceError::kNonfiniteBandEnergy);
        atomicExch(&valid, 0);
      } else {
        partial_band[threadIdx.x] = band;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && valid != 0) {
    workspace.band_energy_scratch[system] = partial_band[0];
    workspace.occupation_sum_scratch[system] = partial_occupation[0];
  }
}

__host__ __device__ std::int64_t triangle_inclusive(std::int64_t value) {
  return (value & 1LL) == 0LL ? (value / 2LL) * (value + 1LL) : value * ((value + 1LL) / 2LL);
}

struct MatrixPair {
  std::int64_t row;
  std::int64_t column;
};

__device__ MatrixPair matrix_pair(std::int64_t packed) {
  std::int64_t row =
      static_cast<std::int64_t>(0.5 * (sqrt(1.0 + 8.0 * static_cast<double>(packed)) - 1.0));
  while (row > 0 && triangle_inclusive(row) > packed) {
    --row;
  }
  while (triangle_inclusive(row + 1) <= packed) {
    ++row;
  }
  const std::int64_t previous = triangle_inclusive(row);
  return {row, packed - previous};
}

__global__ void contract_kernel(Gfn2DensityDeviceBatch batch, Gfn2DensityDeviceInput input,
                                Gfn2DensityDeviceWorkspace workspace, std::uint32_t* system_errors,
                                std::uint32_t* device_error, std::int64_t tiles_per_system) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t tile = static_cast<std::int64_t>(blockIdx.y);
  if (atomicAdd(workspace.sequence_active, 0u) == 0u || input.active[system] != 1u ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t orbital_begin = batch.orbital_offsets[system];
  const std::int64_t orbital_end = batch.orbital_offsets[system + 1];
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t count = orbital_end - orbital_begin;
  const std::int64_t pair_count = triangle_inclusive(count);
  const std::int64_t pair_stride = tiles_per_system * blockDim.x;
  /* Scheduling changes only pair ownership; one thread still executes the
   * complete ordered FP64 orbital loop for each matrix element. */
  for (std::int64_t pair = tile * blockDim.x + threadIdx.x; pair < pair_count;
       pair += pair_stride) {
    const MatrixPair indices = matrix_pair(pair);
    double density = 0.0;
    double weighted_density = 0.0;
    bool finite = true;
    for (std::int64_t local = 0; local < count; ++local) {
      const double first = input.coefficients[matrix_begin + indices.row * count + local];
      const double second = input.coefficients[matrix_begin + indices.column * count + local];
      const double density_left = first * workspace.weights[orbital_begin + local];
      const double density_contribution = density_left * second;
      const double density_updated = fma(density_left, second, density);
      if (!isfinite(density_left) || !isfinite(density_contribution) ||
          !isfinite(density_updated)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2DensityDeviceError::kNonfiniteDensityArithmetic);
        finite = false;
        break;
      }
      const double weighted_left = first * workspace.energy_weights[orbital_begin + local];
      const double weighted_contribution = weighted_left * second;
      const double weighted_updated = fma(weighted_left, second, weighted_density);
      if (!isfinite(weighted_left) || !isfinite(weighted_contribution) ||
          !isfinite(weighted_updated)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2DensityDeviceError::kNonfiniteWeightedDensityArithmetic);
        finite = false;
        break;
      }
      density = density_updated;
      weighted_density = weighted_updated;
    }
    if (finite) {
      const std::int64_t first = matrix_begin + indices.row * count + indices.column;
      const std::int64_t second = matrix_begin + indices.column * count + indices.row;
      workspace.density_scratch[first] = density;
      workspace.weighted_density_scratch[first] = weighted_density;
      workspace.density_scratch[second] = density;
      workspace.weighted_density_scratch[second] = weighted_density;
    }
  }
}

__global__ void trace_kernel(Gfn2DensityDeviceBatch batch, Gfn2DensityDeviceInput input,
                             Gfn2DensityDeviceWorkspace workspace, std::uint32_t* system_errors,
                             std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ double partial_density[kThreadsPerBlock];
  __shared__ double partial_weighted[kThreadsPerBlock];
  __shared__ int valid;
  if (threadIdx.x == 0) {
    valid = atomicAdd(workspace.sequence_active, 0u) == 1u && input.active[system] == 1u &&
                    system_is_valid(system_errors, system)
                ? 1
                : 0;
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  const std::int64_t orbital_begin = batch.orbital_offsets[system];
  const std::int64_t count = batch.orbital_offsets[system + 1] - orbital_begin;
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  double density_trace = 0.0;
  double weighted_trace = 0.0;
  for (std::int64_t diagonal = threadIdx.x; diagonal < count; diagonal += blockDim.x) {
    const std::int64_t index = matrix_begin + diagonal * count + diagonal;
    const double density = density_trace + workspace.density_scratch[index];
    const double weighted = weighted_trace + workspace.weighted_density_scratch[index];
    if (!isfinite(density) || !isfinite(weighted)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2DensityDeviceError::kNonfiniteTrace);
      atomicExch(&valid, 0);
    } else {
      density_trace = density;
      weighted_trace = weighted;
    }
  }
  partial_density[threadIdx.x] = density_trace;
  partial_weighted[threadIdx.x] = weighted_trace;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      const double density = partial_density[threadIdx.x] + partial_density[threadIdx.x + offset];
      const double weighted =
          partial_weighted[threadIdx.x] + partial_weighted[threadIdx.x + offset];
      if (!isfinite(density) || !isfinite(weighted)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2DensityDeviceError::kNonfiniteTrace);
        atomicExch(&valid, 0);
      } else {
        partial_density[threadIdx.x] = density;
        partial_weighted[threadIdx.x] = weighted;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && valid != 0) {
    workspace.density_trace_scratch[system] = partial_density[0];
    workspace.weighted_density_trace_scratch[system] = partial_weighted[0];
  }
}

__device__ bool load_spin_system(const Gfn2DensityDeviceBatch& batch,
                                 const Gfn2WavefunctionLayoutView& layout,
                                 const Gfn2DensityDeviceInput& input, std::int64_t system,
                                 SpinSystemRanges* ranges, int* active, int* valid,
                                 std::uint32_t* system_errors, std::uint32_t* device_error) {
  if (threadIdx.x == 0) {
    *active = 0;
    *valid = 1;
    if (!system_is_valid(system_errors, system)) {
      *valid = 0;
    } else {
      const std::uint8_t state = input.active[system];
      if (state == 0u) {
        *valid = 0;
      } else if (state != 1u) {
        record_system_error(system_errors, system, device_error,
                            Gfn2DensityDeviceError::kInvalidActiveMask);
        *valid = 0;
      } else {
        *active = 1;
        ranges->orbital_begin = batch.orbital_offsets[system];
        ranges->orbital_end = batch.orbital_offsets[system + 1];
        ranges->matrix_begin = batch.matrix_offsets[system];
        ranges->matrix_end = batch.matrix_offsets[system + 1];
        ranges->spin_orbital_begin = layout.spin_orbital_offsets[system];
        ranges->spin_orbital_end = layout.spin_orbital_offsets[system + 1];
        ranges->spin_matrix_begin = layout.spin_matrix_offsets[system];
        ranges->spin_matrix_end = layout.spin_matrix_offsets[system + 1];
        ranges->channel_begin = layout.spin_channel_offsets[system];
        ranges->channel_end = layout.spin_channel_offsets[system + 1];
        ranges->spin_channels = layout.spin_channels[system];
        if (ranges->spin_channels != 1 && ranges->spin_channels != 2) {
          record_system_error(system_errors, system, device_error,
                              Gfn2DensityDeviceError::kInvalidSpinChannels);
          *valid = 0;
        }
        const bool endpoints_valid =
            ranges->orbital_begin >= 0 && ranges->orbital_begin < ranges->orbital_end &&
            ranges->orbital_end <= batch.total_orbitals && ranges->matrix_begin >= 0 &&
            ranges->matrix_begin < ranges->matrix_end &&
            ranges->matrix_end <= batch.total_matrix_elements && ranges->spin_orbital_begin >= 0 &&
            ranges->spin_orbital_begin < ranges->spin_orbital_end &&
            ranges->spin_orbital_end <= layout.total_spin_orbitals &&
            ranges->spin_matrix_begin >= 0 && ranges->spin_matrix_begin < ranges->spin_matrix_end &&
            ranges->spin_matrix_end <= layout.total_spin_matrix_elements &&
            ranges->channel_begin >= 0 && ranges->channel_begin < ranges->channel_end &&
            ranges->channel_end <= layout.total_spin_channels &&
            (system != 0 || (ranges->orbital_begin == 0 && ranges->matrix_begin == 0 &&
                             ranges->spin_orbital_begin == 0 && ranges->spin_matrix_begin == 0 &&
                             ranges->channel_begin == 0)) &&
            (system + 1 != batch.batch_size ||
             (ranges->orbital_end == batch.total_orbitals &&
              ranges->matrix_end == batch.total_matrix_elements &&
              ranges->spin_orbital_end == layout.total_spin_orbitals &&
              ranges->spin_matrix_end == layout.total_spin_matrix_elements &&
              ranges->channel_end == layout.total_spin_channels));
        if (endpoints_valid && *valid != 0) {
          ranges->orbital_count = ranges->orbital_end - ranges->orbital_begin;
          const std::int64_t matrix_count = ranges->matrix_end - ranges->matrix_begin;
          const std::int64_t spin_orbitals = ranges->spin_orbital_end - ranges->spin_orbital_begin;
          const std::int64_t spin_matrices = ranges->spin_matrix_end - ranges->spin_matrix_begin;
          const std::int64_t channels = ranges->channel_end - ranges->channel_begin;
          const bool square_fits = ranges->orbital_count <= kMaximumInt64 / ranges->orbital_count;
          const bool spin_orbitals_fit =
              ranges->orbital_count <= kMaximumInt64 / ranges->spin_channels;
          const bool spin_matrices_fit = matrix_count <= kMaximumInt64 / ranges->spin_channels;
          *valid = square_fits && spin_orbitals_fit && spin_matrices_fit &&
                           matrix_count == ranges->orbital_count * ranges->orbital_count &&
                           spin_orbitals == ranges->orbital_count * ranges->spin_channels &&
                           spin_matrices == matrix_count * ranges->spin_channels &&
                           channels == ranges->spin_channels
                       ? 1
                       : 0;
        } else {
          *valid = 0;
        }
        if (*valid == 0 && ranges->spin_channels >= 1 && ranges->spin_channels <= 2) {
          record_system_error(system_errors, system, device_error,
                              Gfn2DensityDeviceError::kInvalidOffsets);
        }
      }
    }
  }
  __syncthreads();
  return *active != 0 && *valid != 0;
}

/*
 * Validate both channels before any publication and prepare their independent
 * orbital weights. The channel loop is deliberately ordered alpha then beta;
 * this same ordering is used later for the legacy per-system scalar sums.
 */
__global__ void spin_preflight_kernel(Gfn2DensityDeviceBatch batch,
                                      Gfn2WavefunctionLayoutView layout,
                                      Gfn2DensityDeviceInput input,
                                      Gfn2DensityDeviceWorkspace workspace,
                                      std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SpinSystemRanges ranges;
  __shared__ int active;
  __shared__ int valid;
  __shared__ double partial_band[kThreadsPerBlock];
  __shared__ double partial_occupation[kThreadsPerBlock];
  if (atomicAdd(workspace.sequence_active, 0u) == 0u ||
      !load_spin_system(batch, layout, input, system, &ranges, &active, &valid, system_errors,
                        device_error)) {
    return;
  }

  for (std::int64_t matrix = ranges.spin_matrix_begin + threadIdx.x;
       matrix < ranges.spin_matrix_end; matrix += blockDim.x) {
    if (!isfinite(input.coefficients[matrix])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2DensityDeviceError::kNonfiniteCoefficient);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();

  const std::int64_t occupation_base = 2 * ranges.orbital_begin;
  for (std::int32_t channel = 0; channel < ranges.spin_channels; ++channel) {
    double local_band = 0.0;
    double local_occupation = 0.0;
    for (std::int64_t local = threadIdx.x; local < ranges.orbital_count; local += blockDim.x) {
      const double alpha = input.occupations[occupation_base + local];
      const double beta = input.occupations[occupation_base + ranges.orbital_count + local];
      const double occupation =
          ranges.spin_channels == 1 ? alpha + beta : (channel == 0 ? alpha : beta);
      const std::int64_t spin_orbital =
          ranges.spin_orbital_begin + channel * ranges.orbital_count + local;
      const double eigenvalue = input.eigenvalues[spin_orbital];
      if (!(alpha >= 0.0 && alpha <= 1.0) || !isfinite(alpha) || !(beta >= 0.0 && beta <= 1.0) ||
          !isfinite(beta)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2DensityDeviceError::kInvalidOccupation);
        atomicExch(&valid, 0);
        continue;
      }
      if (!isfinite(eigenvalue)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2DensityDeviceError::kNonfiniteEigenvalue);
        atomicExch(&valid, 0);
        continue;
      }
      const double energy_weight = occupation * eigenvalue;
      const double updated_occupation = local_occupation + occupation;
      const double updated_band = local_band + energy_weight;
      if (!isfinite(occupation) || !isfinite(energy_weight) || !isfinite(updated_occupation)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2DensityDeviceError::kNonfiniteWeightArithmetic);
        atomicExch(&valid, 0);
        continue;
      }
      if (!isfinite(updated_band)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2DensityDeviceError::kNonfiniteBandEnergy);
        atomicExch(&valid, 0);
        continue;
      }
      workspace.weights[spin_orbital] = occupation;
      workspace.energy_weights[spin_orbital] = energy_weight;
      local_occupation = updated_occupation;
      local_band = updated_band;
    }
    partial_band[threadIdx.x] = local_band;
    partial_occupation[threadIdx.x] = local_occupation;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
      if (threadIdx.x < offset) {
        const double band = partial_band[threadIdx.x] + partial_band[threadIdx.x + offset];
        const double occupation =
            partial_occupation[threadIdx.x] + partial_occupation[threadIdx.x + offset];
        if (!isfinite(occupation)) {
          record_system_error(system_errors, system, device_error,
                              Gfn2DensityDeviceError::kNonfiniteWeightArithmetic);
          atomicExch(&valid, 0);
        } else {
          partial_occupation[threadIdx.x] = occupation;
        }
        if (!isfinite(band)) {
          record_system_error(system_errors, system, device_error,
                              Gfn2DensityDeviceError::kNonfiniteBandEnergy);
          atomicExch(&valid, 0);
        } else {
          partial_band[threadIdx.x] = band;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0 && valid != 0) {
      const std::int64_t diagnostic = ranges.channel_begin + channel;
      workspace.channel_band_energy_scratch[diagnostic] = partial_band[0];
      workspace.channel_occupation_sum_scratch[diagnostic] = partial_occupation[0];
    }
    __syncthreads();
  }
}

__global__ void spin_contract_kernel(Gfn2DensityDeviceBatch batch,
                                     Gfn2WavefunctionLayoutView layout,
                                     Gfn2DensityDeviceInput input,
                                     Gfn2DensityDeviceWorkspace workspace,
                                     std::uint32_t* system_errors, std::uint32_t* device_error,
                                     std::int64_t tiles_per_channel) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const std::int32_t channel = static_cast<std::int32_t>(blockIdx.y);
  const std::int64_t tile = static_cast<std::int64_t>(blockIdx.z);
  if (atomicAdd(workspace.sequence_active, 0u) == 0u || input.active[system] != 1u ||
      !system_is_valid(system_errors, system) || channel >= layout.spin_channels[system]) {
    return;
  }
  const std::int64_t count = batch.orbital_offsets[system + 1] - batch.orbital_offsets[system];
  const std::int64_t matrix_count = count * count;
  const std::int64_t matrix_begin = layout.spin_matrix_offsets[system] + channel * matrix_count;
  const std::int64_t orbital_begin = layout.spin_orbital_offsets[system] + channel * count;
  const std::int64_t pair_count = triangle_inclusive(count);
  const std::int64_t pair_stride = tiles_per_channel * blockDim.x;
  /* Match the restricted path's one-thread-per-pair arithmetic contract. */
  for (std::int64_t pair = tile * blockDim.x + threadIdx.x; pair < pair_count;
       pair += pair_stride) {
    const MatrixPair indices = matrix_pair(pair);
    double density = 0.0;
    double weighted_density = 0.0;
    bool finite = true;
    for (std::int64_t local = 0; local < count; ++local) {
      const double first = input.coefficients[matrix_begin + indices.row * count + local];
      const double second = input.coefficients[matrix_begin + indices.column * count + local];
      const double density_left = first * workspace.weights[orbital_begin + local];
      const double density_contribution = density_left * second;
      const double density_updated = fma(density_left, second, density);
      if (!isfinite(density_left) || !isfinite(density_contribution) ||
          !isfinite(density_updated)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2DensityDeviceError::kNonfiniteDensityArithmetic);
        finite = false;
        break;
      }
      const double weighted_left = first * workspace.energy_weights[orbital_begin + local];
      const double weighted_contribution = weighted_left * second;
      const double weighted_updated = fma(weighted_left, second, weighted_density);
      if (!isfinite(weighted_left) || !isfinite(weighted_contribution) ||
          !isfinite(weighted_updated)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2DensityDeviceError::kNonfiniteWeightedDensityArithmetic);
        finite = false;
        break;
      }
      density = density_updated;
      weighted_density = weighted_updated;
    }
    if (finite) {
      const std::int64_t first = matrix_begin + indices.row * count + indices.column;
      const std::int64_t second = matrix_begin + indices.column * count + indices.row;
      workspace.density_scratch[first] = density;
      workspace.weighted_density_scratch[first] = weighted_density;
      workspace.density_scratch[second] = density;
      workspace.weighted_density_scratch[second] = weighted_density;
    }
  }
}

__global__ void spin_trace_kernel(Gfn2DensityDeviceBatch batch, Gfn2WavefunctionLayoutView layout,
                                  Gfn2DensityDeviceInput input,
                                  Gfn2DensityDeviceWorkspace workspace,
                                  std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const std::int32_t channel = static_cast<std::int32_t>(blockIdx.y);
  __shared__ double partial_density[kThreadsPerBlock];
  __shared__ double partial_weighted[kThreadsPerBlock];
  __shared__ int valid;
  if (threadIdx.x == 0) {
    valid = atomicAdd(workspace.sequence_active, 0u) == 1u && input.active[system] == 1u &&
                    system_is_valid(system_errors, system) && channel < layout.spin_channels[system]
                ? 1
                : 0;
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  const std::int64_t count = batch.orbital_offsets[system + 1] - batch.orbital_offsets[system];
  const std::int64_t matrix_begin = layout.spin_matrix_offsets[system] + channel * count * count;
  double density_trace = 0.0;
  double weighted_trace = 0.0;
  for (std::int64_t diagonal = threadIdx.x; diagonal < count; diagonal += blockDim.x) {
    const std::int64_t index = matrix_begin + diagonal * count + diagonal;
    const double density = density_trace + workspace.density_scratch[index];
    const double weighted = weighted_trace + workspace.weighted_density_scratch[index];
    if (!isfinite(density) || !isfinite(weighted)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2DensityDeviceError::kNonfiniteTrace);
      atomicExch(&valid, 0);
    } else {
      density_trace = density;
      weighted_trace = weighted;
    }
  }
  partial_density[threadIdx.x] = density_trace;
  partial_weighted[threadIdx.x] = weighted_trace;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      const double density = partial_density[threadIdx.x] + partial_density[threadIdx.x + offset];
      const double weighted =
          partial_weighted[threadIdx.x] + partial_weighted[threadIdx.x + offset];
      if (!isfinite(density) || !isfinite(weighted)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2DensityDeviceError::kNonfiniteTrace);
        atomicExch(&valid, 0);
      } else {
        partial_density[threadIdx.x] = density;
        partial_weighted[threadIdx.x] = weighted;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && valid != 0) {
    const std::int64_t diagnostic = layout.spin_channel_offsets[system] + channel;
    workspace.channel_density_trace_scratch[diagnostic] = partial_density[0];
    workspace.channel_weighted_density_trace_scratch[diagnostic] = partial_weighted[0];
  }
}

__global__ void spin_sum_diagnostics_kernel(Gfn2DensityDeviceBatch batch,
                                            Gfn2WavefunctionLayoutView layout,
                                            Gfn2DensityDeviceInput input,
                                            Gfn2DensityDeviceWorkspace workspace,
                                            std::uint32_t* system_errors,
                                            std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (threadIdx.x != 0 || atomicAdd(workspace.sequence_active, 0u) == 0u ||
      input.active[system] != 1u || !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t channel = layout.spin_channel_offsets[system];
  double band = workspace.channel_band_energy_scratch[channel];
  double occupation = workspace.channel_occupation_sum_scratch[channel];
  double density_trace = workspace.channel_density_trace_scratch[channel];
  double weighted_trace = workspace.channel_weighted_density_trace_scratch[channel];
  if (layout.spin_channels[system] == 2) {
    band += workspace.channel_band_energy_scratch[channel + 1];
    occupation += workspace.channel_occupation_sum_scratch[channel + 1];
    density_trace += workspace.channel_density_trace_scratch[channel + 1];
    weighted_trace += workspace.channel_weighted_density_trace_scratch[channel + 1];
  }
  if (!isfinite(occupation)) {
    record_system_error(system_errors, system, device_error,
                        Gfn2DensityDeviceError::kNonfiniteWeightArithmetic);
    return;
  }
  if (!isfinite(band)) {
    record_system_error(system_errors, system, device_error,
                        Gfn2DensityDeviceError::kNonfiniteBandEnergy);
    return;
  }
  if (!isfinite(density_trace) || !isfinite(weighted_trace)) {
    record_system_error(system_errors, system, device_error,
                        Gfn2DensityDeviceError::kNonfiniteTrace);
    return;
  }
  workspace.band_energy_scratch[system] = band;
  workspace.occupation_sum_scratch[system] = occupation;
  workspace.density_trace_scratch[system] = density_trace;
  workspace.weighted_density_trace_scratch[system] = weighted_trace;
}

__global__ void canonicalize_device_error_kernel(std::int64_t batch_size,
                                                 const std::uint32_t* sequence_active,
                                                 const std::uint32_t* system_errors,
                                                 std::uint32_t* device_error) {
  if (blockIdx.x != 0 || threadIdx.x != 0 ||
      atomicAdd(const_cast<std::uint32_t*>(sequence_active), 0u) == 0u) {
    return;
  }
  for (std::int64_t system = 0; system < batch_size; ++system) {
    const std::uint32_t error = atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u);
    if (error != static_cast<std::uint32_t>(Gfn2DensityDeviceError::kSuccess)) {
      atomicExch(device_error, error);
      return;
    }
  }
  atomicExch(device_error, static_cast<std::uint32_t>(Gfn2DensityDeviceError::kSuccess));
}

__global__ void publish_kernel(Gfn2DensityDeviceBatch batch, Gfn2DensityDeviceInput input,
                               Gfn2DensityDeviceResults results,
                               Gfn2DensityDeviceWorkspace workspace,
                               const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (atomicAdd(workspace.sequence_active, 0u) == 0u || input.active[system] != 1u ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t begin = batch.matrix_offsets[system];
  const std::int64_t end = batch.matrix_offsets[system + 1];
  for (std::int64_t matrix = begin + threadIdx.x; matrix < end; matrix += blockDim.x) {
    results.density[matrix] = workspace.density_scratch[matrix];
    results.energy_weighted_density[matrix] = workspace.weighted_density_scratch[matrix];
  }
  if (threadIdx.x == 0) {
    results.band_energies[system] = workspace.band_energy_scratch[system];
    results.occupation_sums[system] = workspace.occupation_sum_scratch[system];
    results.density_traces[system] = workspace.density_trace_scratch[system];
    results.weighted_density_traces[system] = workspace.weighted_density_trace_scratch[system];
  }
}

__global__ void spin_publish_kernel(Gfn2DensityDeviceBatch batch, Gfn2WavefunctionLayoutView layout,
                                    Gfn2DensityDeviceInput input, Gfn2DensityDeviceResults results,
                                    Gfn2DensityDeviceWorkspace workspace,
                                    const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (atomicAdd(workspace.sequence_active, 0u) == 0u || input.active[system] != 1u ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t matrix_begin = layout.spin_matrix_offsets[system];
  const std::int64_t matrix_end = layout.spin_matrix_offsets[system + 1];
  for (std::int64_t matrix = matrix_begin + threadIdx.x; matrix < matrix_end;
       matrix += blockDim.x) {
    results.density[matrix] = workspace.density_scratch[matrix];
    results.energy_weighted_density[matrix] = workspace.weighted_density_scratch[matrix];
  }
  const std::int64_t channel_begin = layout.spin_channel_offsets[system];
  const std::int64_t channel_end = layout.spin_channel_offsets[system + 1];
  for (std::int64_t channel = channel_begin + threadIdx.x; channel < channel_end;
       channel += blockDim.x) {
    results.channel_band_energies[channel] = workspace.channel_band_energy_scratch[channel];
    results.channel_occupation_sums[channel] = workspace.channel_occupation_sum_scratch[channel];
    results.channel_density_traces[channel] = workspace.channel_density_trace_scratch[channel];
    results.channel_weighted_density_traces[channel] =
        workspace.channel_weighted_density_trace_scratch[channel];
  }
  if (threadIdx.x == 0) {
    results.band_energies[system] = workspace.band_energy_scratch[system];
    results.occupation_sums[system] = workspace.occupation_sum_scratch[system];
    results.density_traces[system] = workspace.density_trace_scratch[system];
    results.weighted_density_traces[system] = workspace.weighted_density_trace_scratch[system];
  }
}

bool checked_multiply(std::int64_t value, std::int64_t factor, std::int64_t* result) noexcept {
  if (value < 0 || factor < 0 ||
      (value != 0 && factor > std::numeric_limits<std::int64_t>::max() / value)) {
    return false;
  }
  *result = value * factor;
  return true;
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

bool ranges_overlap(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin < second.end && second.begin < first.end;
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

template <std::size_t FirstCount, std::size_t SecondCount>
bool disjoint_sets(const std::array<AddressRange, FirstCount>& first,
                   const std::array<AddressRange, SecondCount>& second) noexcept {
  for (const AddressRange& first_range : first) {
    for (const AddressRange& second_range : second) {
      if (ranges_overlap(first_range, second_range)) {
        return false;
      }
    }
  }
  return true;
}

bool validate_restricted_launch(const Gfn2DensityDeviceBatch& batch,
                                const Gfn2DensityDeviceInput& input,
                                const Gfn2DensityDeviceResults& results,
                                const Gfn2DensityDeviceWorkspace& workspace,
                                std::uint32_t* system_errors,
                                std::uint32_t* device_error) noexcept {
  std::int64_t two_orbitals = 0;
  if (batch.batch_size <= 0 || batch.batch_size > std::numeric_limits<int>::max() ||
      batch.total_orbitals <= 0 || batch.total_matrix_elements <= 0 || batch.plan_token == 0u ||
      batch.contraction_tiles_per_channel <= 0 ||
      batch.contraction_tiles_per_channel > kGfn2DensityContractBlockBudget ||
      !checked_multiply(batch.total_orbitals, 2, &two_orbitals) ||
      batch.orbital_offset_count != batch.batch_size + 1 ||
      batch.matrix_offset_count != batch.batch_size + 1 || input.plan_token != batch.plan_token ||
      input.coefficient_elements != batch.total_matrix_elements ||
      input.eigenvalue_elements != batch.total_orbitals ||
      input.occupation_elements != two_orbitals || input.active_elements != batch.batch_size ||
      results.plan_token != batch.plan_token ||
      results.density_elements != batch.total_matrix_elements ||
      results.weighted_density_elements != batch.total_matrix_elements ||
      results.band_energy_elements != batch.batch_size ||
      results.occupation_sum_elements != batch.batch_size ||
      results.density_trace_elements != batch.batch_size ||
      results.weighted_density_trace_elements != batch.batch_size ||
      workspace.plan_token != batch.plan_token ||
      workspace.density_elements != batch.total_matrix_elements ||
      workspace.weighted_density_elements != batch.total_matrix_elements ||
      workspace.weight_elements != batch.total_orbitals ||
      workspace.energy_weight_elements != batch.total_orbitals ||
      workspace.band_energy_elements != batch.batch_size ||
      workspace.occupation_sum_elements != batch.batch_size ||
      workspace.density_trace_elements != batch.batch_size ||
      workspace.weighted_density_trace_elements != batch.batch_size ||
      workspace.sequence_active_elements != 1 ||
      !is_aligned(batch.orbital_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.matrix_offsets, alignof(std::int64_t)) ||
      !is_aligned(input.coefficients, alignof(double)) ||
      !is_aligned(input.eigenvalues, alignof(double)) ||
      !is_aligned(input.occupations, alignof(double)) || input.active == nullptr ||
      !is_aligned(results.density, alignof(double)) ||
      !is_aligned(results.energy_weighted_density, alignof(double)) ||
      !is_aligned(results.band_energies, alignof(double)) ||
      !is_aligned(results.occupation_sums, alignof(double)) ||
      !is_aligned(results.density_traces, alignof(double)) ||
      !is_aligned(results.weighted_density_traces, alignof(double)) ||
      !is_aligned(workspace.density_scratch, alignof(double)) ||
      !is_aligned(workspace.weighted_density_scratch, alignof(double)) ||
      !is_aligned(workspace.weights, alignof(double)) ||
      !is_aligned(workspace.energy_weights, alignof(double)) ||
      !is_aligned(workspace.band_energy_scratch, alignof(double)) ||
      !is_aligned(workspace.occupation_sum_scratch, alignof(double)) ||
      !is_aligned(workspace.density_trace_scratch, alignof(double)) ||
      !is_aligned(workspace.weighted_density_trace_scratch, alignof(double)) ||
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return false;
  }

  std::array<AddressRange, 5> reads{};
  std::array<AddressRange, 17> writes{};
  if (!make_range(batch.orbital_offsets, batch.orbital_offset_count, sizeof(std::int64_t),
                  &reads[0]) ||
      !make_range(batch.matrix_offsets, batch.matrix_offset_count, sizeof(std::int64_t),
                  &reads[1]) ||
      !make_range(input.coefficients, input.coefficient_elements, sizeof(double), &reads[2]) ||
      !make_range(input.eigenvalues, input.eigenvalue_elements, sizeof(double), &reads[3]) ||
      !make_range(input.occupations, input.occupation_elements, sizeof(double), &reads[4]) ||
      !make_range(results.density, results.density_elements, sizeof(double), &writes[0]) ||
      !make_range(results.energy_weighted_density, results.weighted_density_elements,
                  sizeof(double), &writes[1]) ||
      !make_range(results.band_energies, results.band_energy_elements, sizeof(double),
                  &writes[2]) ||
      !make_range(results.occupation_sums, results.occupation_sum_elements, sizeof(double),
                  &writes[3]) ||
      !make_range(results.density_traces, results.density_trace_elements, sizeof(double),
                  &writes[4]) ||
      !make_range(results.weighted_density_traces, results.weighted_density_trace_elements,
                  sizeof(double), &writes[5]) ||
      !make_range(workspace.density_scratch, workspace.density_elements, sizeof(double),
                  &writes[6]) ||
      !make_range(workspace.weighted_density_scratch, workspace.weighted_density_elements,
                  sizeof(double), &writes[7]) ||
      !make_range(workspace.weights, workspace.weight_elements, sizeof(double), &writes[8]) ||
      !make_range(workspace.energy_weights, workspace.energy_weight_elements, sizeof(double),
                  &writes[9]) ||
      !make_range(workspace.band_energy_scratch, workspace.band_energy_elements, sizeof(double),
                  &writes[10]) ||
      !make_range(workspace.occupation_sum_scratch, workspace.occupation_sum_elements,
                  sizeof(double), &writes[11]) ||
      !make_range(workspace.density_trace_scratch, workspace.density_trace_elements, sizeof(double),
                  &writes[12]) ||
      !make_range(workspace.weighted_density_trace_scratch,
                  workspace.weighted_density_trace_elements, sizeof(double), &writes[13]) ||
      !make_range(workspace.sequence_active, 1, sizeof(std::uint32_t), &writes[14]) ||
      !make_range(system_errors, batch.batch_size, sizeof(std::uint32_t), &writes[15]) ||
      !make_range(device_error, 1, sizeof(std::uint32_t), &writes[16]) ||
      !pairwise_disjoint(reads) || !pairwise_disjoint(writes) || !disjoint_sets(reads, writes)) {
    return false;
  }
  AddressRange active_range;
  if (!make_range(input.active, input.active_elements, sizeof(std::uint8_t), &active_range)) {
    return false;
  }
  for (const AddressRange& read : reads) {
    if (ranges_overlap(active_range, read)) {
      return false;
    }
  }
  for (const AddressRange& write : writes) {
    if (ranges_overlap(active_range, write)) {
      return false;
    }
  }
  return true;
}

bool validate_spin_launch(const Gfn2DensityDeviceBatch& batch,
                          const Gfn2WavefunctionLayoutView& layout,
                          const Gfn2DensityDeviceInput& input,
                          const Gfn2DensityDeviceResults& results,
                          const Gfn2DensityDeviceWorkspace& workspace, std::uint32_t* system_errors,
                          std::uint32_t* device_error) noexcept {
  std::int64_t two_orbitals = 0;
  if (batch.batch_size <= 0 || batch.batch_size > std::numeric_limits<int>::max() ||
      batch.total_orbitals <= 0 || batch.total_matrix_elements <= 0 ||
      batch.contraction_tiles_per_channel <= 0 ||
      batch.contraction_tiles_per_channel > kGfn2DensityContractBlockBudget ||
      layout.memory_space != Gfn2PlanMemorySpace::kCudaDevice ||
      layout.plan_token != batch.plan_token || layout.batch_size != batch.batch_size ||
      layout.total_spin_orbitals <= 0 || layout.total_spin_matrix_elements <= 0 ||
      layout.total_spin_channels < batch.batch_size ||
      layout.total_spin_channels > 2 * batch.batch_size || batch.plan_token == 0u ||
      !checked_multiply(batch.total_orbitals, 2, &two_orbitals) ||
      batch.orbital_offset_count != batch.batch_size + 1 ||
      batch.matrix_offset_count != batch.batch_size + 1 ||
      layout.spin_channel_count != batch.batch_size ||
      layout.spin_orbital_offset_count != batch.batch_size + 1 ||
      layout.spin_matrix_offset_count != batch.batch_size + 1 ||
      layout.spin_channel_offset_count != batch.batch_size + 1 ||
      input.plan_token != batch.plan_token ||
      input.coefficient_elements != layout.total_spin_matrix_elements ||
      input.eigenvalue_elements != layout.total_spin_orbitals ||
      input.occupation_elements != two_orbitals || input.active_elements != batch.batch_size ||
      results.plan_token != batch.plan_token ||
      results.density_elements != layout.total_spin_matrix_elements ||
      results.weighted_density_elements != layout.total_spin_matrix_elements ||
      results.band_energy_elements != batch.batch_size ||
      results.occupation_sum_elements != batch.batch_size ||
      results.density_trace_elements != batch.batch_size ||
      results.weighted_density_trace_elements != batch.batch_size ||
      results.channel_band_energy_elements != layout.total_spin_channels ||
      results.channel_occupation_sum_elements != layout.total_spin_channels ||
      results.channel_density_trace_elements != layout.total_spin_channels ||
      results.channel_weighted_density_trace_elements != layout.total_spin_channels ||
      workspace.plan_token != batch.plan_token ||
      workspace.density_elements != layout.total_spin_matrix_elements ||
      workspace.weighted_density_elements != layout.total_spin_matrix_elements ||
      workspace.weight_elements != layout.total_spin_orbitals ||
      workspace.energy_weight_elements != layout.total_spin_orbitals ||
      workspace.band_energy_elements != batch.batch_size ||
      workspace.occupation_sum_elements != batch.batch_size ||
      workspace.density_trace_elements != batch.batch_size ||
      workspace.weighted_density_trace_elements != batch.batch_size ||
      workspace.channel_band_energy_elements != layout.total_spin_channels ||
      workspace.channel_occupation_sum_elements != layout.total_spin_channels ||
      workspace.channel_density_trace_elements != layout.total_spin_channels ||
      workspace.channel_weighted_density_trace_elements != layout.total_spin_channels ||
      workspace.sequence_active_elements != 1 ||
      !is_aligned(batch.orbital_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.matrix_offsets, alignof(std::int64_t)) ||
      !is_aligned(layout.spin_channels, alignof(std::int32_t)) ||
      !is_aligned(layout.spin_orbital_offsets, alignof(std::int64_t)) ||
      !is_aligned(layout.spin_matrix_offsets, alignof(std::int64_t)) ||
      !is_aligned(layout.spin_channel_offsets, alignof(std::int64_t)) ||
      !is_aligned(input.coefficients, alignof(double)) ||
      !is_aligned(input.eigenvalues, alignof(double)) ||
      !is_aligned(input.occupations, alignof(double)) || input.active == nullptr ||
      !is_aligned(results.density, alignof(double)) ||
      !is_aligned(results.energy_weighted_density, alignof(double)) ||
      !is_aligned(results.band_energies, alignof(double)) ||
      !is_aligned(results.occupation_sums, alignof(double)) ||
      !is_aligned(results.density_traces, alignof(double)) ||
      !is_aligned(results.weighted_density_traces, alignof(double)) ||
      !is_aligned(results.channel_band_energies, alignof(double)) ||
      !is_aligned(results.channel_occupation_sums, alignof(double)) ||
      !is_aligned(results.channel_density_traces, alignof(double)) ||
      !is_aligned(results.channel_weighted_density_traces, alignof(double)) ||
      !is_aligned(workspace.density_scratch, alignof(double)) ||
      !is_aligned(workspace.weighted_density_scratch, alignof(double)) ||
      !is_aligned(workspace.weights, alignof(double)) ||
      !is_aligned(workspace.energy_weights, alignof(double)) ||
      !is_aligned(workspace.band_energy_scratch, alignof(double)) ||
      !is_aligned(workspace.occupation_sum_scratch, alignof(double)) ||
      !is_aligned(workspace.density_trace_scratch, alignof(double)) ||
      !is_aligned(workspace.weighted_density_trace_scratch, alignof(double)) ||
      !is_aligned(workspace.channel_band_energy_scratch, alignof(double)) ||
      !is_aligned(workspace.channel_occupation_sum_scratch, alignof(double)) ||
      !is_aligned(workspace.channel_density_trace_scratch, alignof(double)) ||
      !is_aligned(workspace.channel_weighted_density_trace_scratch, alignof(double)) ||
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return false;
  }

  std::array<AddressRange, 9> reads{};
  std::array<AddressRange, 25> writes{};
  if (!make_range(batch.orbital_offsets, batch.orbital_offset_count, sizeof(std::int64_t),
                  &reads[0]) ||
      !make_range(batch.matrix_offsets, batch.matrix_offset_count, sizeof(std::int64_t),
                  &reads[1]) ||
      !make_range(layout.spin_channels, layout.spin_channel_count, sizeof(std::int32_t),
                  &reads[2]) ||
      !make_range(layout.spin_orbital_offsets, layout.spin_orbital_offset_count,
                  sizeof(std::int64_t), &reads[3]) ||
      !make_range(layout.spin_matrix_offsets, layout.spin_matrix_offset_count, sizeof(std::int64_t),
                  &reads[4]) ||
      !make_range(layout.spin_channel_offsets, layout.spin_channel_offset_count,
                  sizeof(std::int64_t), &reads[5]) ||
      !make_range(input.coefficients, input.coefficient_elements, sizeof(double), &reads[6]) ||
      !make_range(input.eigenvalues, input.eigenvalue_elements, sizeof(double), &reads[7]) ||
      !make_range(input.occupations, input.occupation_elements, sizeof(double), &reads[8]) ||
      !make_range(results.density, results.density_elements, sizeof(double), &writes[0]) ||
      !make_range(results.energy_weighted_density, results.weighted_density_elements,
                  sizeof(double), &writes[1]) ||
      !make_range(results.band_energies, results.band_energy_elements, sizeof(double),
                  &writes[2]) ||
      !make_range(results.occupation_sums, results.occupation_sum_elements, sizeof(double),
                  &writes[3]) ||
      !make_range(results.density_traces, results.density_trace_elements, sizeof(double),
                  &writes[4]) ||
      !make_range(results.weighted_density_traces, results.weighted_density_trace_elements,
                  sizeof(double), &writes[5]) ||
      !make_range(results.channel_band_energies, results.channel_band_energy_elements,
                  sizeof(double), &writes[6]) ||
      !make_range(results.channel_occupation_sums, results.channel_occupation_sum_elements,
                  sizeof(double), &writes[7]) ||
      !make_range(results.channel_density_traces, results.channel_density_trace_elements,
                  sizeof(double), &writes[8]) ||
      !make_range(results.channel_weighted_density_traces,
                  results.channel_weighted_density_trace_elements, sizeof(double), &writes[9]) ||
      !make_range(workspace.density_scratch, workspace.density_elements, sizeof(double),
                  &writes[10]) ||
      !make_range(workspace.weighted_density_scratch, workspace.weighted_density_elements,
                  sizeof(double), &writes[11]) ||
      !make_range(workspace.weights, workspace.weight_elements, sizeof(double), &writes[12]) ||
      !make_range(workspace.energy_weights, workspace.energy_weight_elements, sizeof(double),
                  &writes[13]) ||
      !make_range(workspace.band_energy_scratch, workspace.band_energy_elements, sizeof(double),
                  &writes[14]) ||
      !make_range(workspace.occupation_sum_scratch, workspace.occupation_sum_elements,
                  sizeof(double), &writes[15]) ||
      !make_range(workspace.density_trace_scratch, workspace.density_trace_elements, sizeof(double),
                  &writes[16]) ||
      !make_range(workspace.weighted_density_trace_scratch,
                  workspace.weighted_density_trace_elements, sizeof(double), &writes[17]) ||
      !make_range(workspace.channel_band_energy_scratch, workspace.channel_band_energy_elements,
                  sizeof(double), &writes[18]) ||
      !make_range(workspace.channel_occupation_sum_scratch,
                  workspace.channel_occupation_sum_elements, sizeof(double), &writes[19]) ||
      !make_range(workspace.channel_density_trace_scratch, workspace.channel_density_trace_elements,
                  sizeof(double), &writes[20]) ||
      !make_range(workspace.channel_weighted_density_trace_scratch,
                  workspace.channel_weighted_density_trace_elements, sizeof(double), &writes[21]) ||
      !make_range(workspace.sequence_active, 1, sizeof(std::uint32_t), &writes[22]) ||
      !make_range(system_errors, batch.batch_size, sizeof(std::uint32_t), &writes[23]) ||
      !make_range(device_error, 1, sizeof(std::uint32_t), &writes[24]) ||
      !pairwise_disjoint(reads) || !pairwise_disjoint(writes) || !disjoint_sets(reads, writes)) {
    return false;
  }
  AddressRange active_range;
  if (!make_range(input.active, input.active_elements, sizeof(std::uint8_t), &active_range)) {
    return false;
  }
  for (const AddressRange& read : reads) {
    if (ranges_overlap(active_range, read)) {
      return false;
    }
  }
  for (const AddressRange& write : writes) {
    if (ranges_overlap(active_range, write)) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool make_gfn2_density_contract_launch_shape(std::int64_t batch_size,
                                             std::int64_t total_spin_channels,
                                             std::int64_t tiles_per_channel,
                                             Gfn2DensityContractLaunchShape& shape) noexcept {
  if (batch_size <= 0 || batch_size > std::numeric_limits<int>::max() ||
      total_spin_channels < batch_size || total_spin_channels > 2 * batch_size ||
      tiles_per_channel <= 0 || tiles_per_channel > kGfn2DensityContractBlockBudget) {
    return false;
  }
  Gfn2DensityContractLaunchShape candidate{};
  candidate.systems = static_cast<std::uint32_t>(batch_size);
  candidate.channels = total_spin_channels == batch_size ? 1u : 2u;
  candidate.tiles = static_cast<std::uint32_t>(tiles_per_channel);
  shape = candidate;
  return true;
}

cudaError_t reset_gfn2_density_device_errors_cuda(std::int64_t batch_size,
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
  return status == cudaSuccess ? cudaMemsetAsync(device_error, 0, sizeof(std::uint32_t), stream)
                               : status;
}

cudaError_t evaluate_gfn2_restricted_density_cuda(
    const Gfn2DensityDeviceBatch& batch, const Gfn2DensityDeviceInput& input,
    const Gfn2DensityDeviceResults& results, const Gfn2DensityDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  const std::int64_t tiles_per_system = batch.contraction_tiles_per_channel;
  if (!validate_restricted_launch(batch, input, results, workspace, system_errors, device_error)) {
    return cudaErrorInvalidValue;
  }
  Gfn2DensityContractLaunchShape launch_shape{};
  if (!make_gfn2_density_contract_launch_shape(batch.batch_size, batch.batch_size, tiles_per_system,
                                               launch_shape)) {
    return cudaErrorInvalidValue;
  }
  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace.sequence_active);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0, stream>>>(
      batch, input, workspace, system_errors, device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  const dim3 contract_grid(launch_shape.systems, launch_shape.tiles, 1u);
  contract_kernel<<<contract_grid, kThreadsPerBlock, 0, stream>>>(
      batch, input, workspace, system_errors, device_error, tiles_per_system);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  trace_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0, stream>>>(
      batch, input, workspace, system_errors, device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  canonicalize_device_error_kernel<<<1, 1, 0, stream>>>(batch.batch_size, workspace.sequence_active,
                                                        system_errors, device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  publish_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0, stream>>>(
      batch, input, results, workspace, system_errors);
  return cudaGetLastError();
}

cudaError_t evaluate_gfn2_spin_density_cuda(
    const Gfn2DensityDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2DensityDeviceInput& input, const Gfn2DensityDeviceResults& results,
    const Gfn2DensityDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  const std::int64_t tiles_per_channel = batch.contraction_tiles_per_channel;
  if (!validate_spin_launch(batch, layout, input, results, workspace, system_errors,
                            device_error)) {
    return cudaErrorInvalidValue;
  }
  Gfn2DensityContractLaunchShape launch_shape{};
  if (!make_gfn2_density_contract_launch_shape(batch.batch_size, layout.total_spin_channels,
                                               tiles_per_channel, launch_shape)) {
    return cudaErrorInvalidValue;
  }
  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace.sequence_active);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  spin_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                          stream>>>(batch, layout, input, workspace, system_errors, device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  const dim3 contract_grid(launch_shape.systems, launch_shape.channels, launch_shape.tiles);
  spin_contract_kernel<<<contract_grid, kThreadsPerBlock, 0, stream>>>(
      batch, layout, input, workspace, system_errors, device_error, tiles_per_channel);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  const dim3 channel_grid(launch_shape.systems, launch_shape.channels, 1u);
  spin_trace_kernel<<<channel_grid, kThreadsPerBlock, 0, stream>>>(batch, layout, input, workspace,
                                                                   system_errors, device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  spin_sum_diagnostics_kernel<<<static_cast<unsigned int>(batch.batch_size), 1, 0, stream>>>(
      batch, layout, input, workspace, system_errors, device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  canonicalize_device_error_kernel<<<1, 1, 0, stream>>>(batch.batch_size, workspace.sequence_active,
                                                        system_errors, device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  spin_publish_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0, stream>>>(
      batch, layout, input, results, workspace, system_errors);
  return cudaGetLastError();
}

}  // namespace xtbloom::detail::cuda
