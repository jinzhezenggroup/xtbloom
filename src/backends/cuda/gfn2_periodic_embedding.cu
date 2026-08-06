#include <array>
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_periodic_embedding.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr std::int64_t kMaximumInt64 = 9223372036854775807LL;

__device__ void record_error(std::uint32_t* device_error, Gfn2PeriodicEmbeddingDeviceError error) {
  atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2PeriodicEmbeddingDeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    Gfn2PeriodicEmbeddingDeviceError error) {
  atomicCAS(system_errors + system,
            static_cast<std::uint32_t>(Gfn2PeriodicEmbeddingDeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ bool system_is_valid(const std::uint32_t* system_errors, std::int64_t system) {
  return atomicAdd(const_cast<std::uint32_t*>(system_errors + system), 0u) ==
         static_cast<std::uint32_t>(Gfn2PeriodicEmbeddingDeviceError::kSuccess);
}

__device__ bool scc_sequence_is_open(const Gfn2SccIterationDeviceActivity& activity) {
  return atomicAdd(const_cast<std::uint32_t*>(activity.sequence_active), 0u) == 1u;
}

/* Activity, scalar generation, and ragged topology are validated in that
 * order. An all-inactive sequence returns before loading even offset zero. */
__global__ void scc_preflight_kernel(Gfn2PeriodicEmbeddingDeviceBatch batch,
                                     std::uint64_t expected_geometry_generation,
                                     Gfn2SccIterationDeviceActivity activity,
                                     std::uint32_t* sequence_active, std::uint32_t* device_error) {
  if (blockIdx.x != 0) {
    return;
  }
  __shared__ int run;
  __shared__ int invalid_activity;
  __shared__ int any_active;
  if (threadIdx.x == 0) {
    const std::uint32_t sequence =
        atomicAdd(const_cast<std::uint32_t*>(activity.sequence_active), 0u);
    run = sequence == 1u ? 1 : 0;
    invalid_activity = sequence > 1u ? 1 : 0;
    any_active = 0;
    /* A canonically closed sequence performed no failing periodic stage. */
    *sequence_active = sequence <= 1u ? 1u : 0u;
  }
  __syncthreads();
  if (run == 0) {
    if (threadIdx.x == 0 && invalid_activity != 0) {
      record_error(device_error, Gfn2PeriodicEmbeddingDeviceError::kInvalidActivity);
    }
    return;
  }
  if (atomicAdd(device_error, 0u) !=
      static_cast<std::uint32_t>(Gfn2PeriodicEmbeddingDeviceError::kSuccess)) {
    if (threadIdx.x == 0) {
      *sequence_active = 0u;
    }
    return;
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    const std::uint8_t active = activity.active_mask[system];
    if (active > 1u) {
      atomicExch(&invalid_activity, 1);
    } else if (active == 1u) {
      atomicExch(&any_active, 1);
    }
  }
  __syncthreads();
  if (invalid_activity != 0) {
    if (threadIdx.x == 0) {
      record_error(device_error, Gfn2PeriodicEmbeddingDeviceError::kInvalidActivity);
      *sequence_active = 0u;
    }
    return;
  }
  if (any_active == 0) {
    return;
  }
  if (expected_geometry_generation == 0u || batch.geometry_generation == 0u ||
      batch.geometry_generation != expected_geometry_generation) {
    if (threadIdx.x == 0) {
      record_error(device_error, Gfn2PeriodicEmbeddingDeviceError::kStaleGeometry);
      *sequence_active = 0u;
    }
    return;
  }
  if (threadIdx.x == 0 && (batch.atom_offsets[0] != 0 || batch.matrix_offsets[0] != 0 ||
                           batch.atom_offsets[batch.batch_size] != batch.total_atoms ||
                           batch.matrix_offsets[batch.batch_size] != batch.total_matrix_elements)) {
    record_error(device_error, Gfn2PeriodicEmbeddingDeviceError::kInvalidOffsets);
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    if (activity.active_mask[system] != 1u) {
      continue;
    }
    const std::int64_t atom_begin = batch.atom_offsets[system];
    const std::int64_t atom_end = batch.atom_offsets[system + 1];
    const std::int64_t matrix_begin = batch.matrix_offsets[system];
    const std::int64_t matrix_end = batch.matrix_offsets[system + 1];
    if (atom_begin < 0 || atom_begin > atom_end || atom_end > batch.total_atoms ||
        matrix_begin < 0 || matrix_begin > matrix_end || matrix_end > batch.total_matrix_elements) {
      record_error(device_error, Gfn2PeriodicEmbeddingDeviceError::kInvalidOffsets);
      continue;
    }
    const std::int64_t atom_count = atom_end - atom_begin;
    const bool square_representable = atom_count == 0 || atom_count <= kMaximumInt64 / atom_count;
    if (!square_representable || matrix_end - matrix_begin != atom_count * atom_count) {
      record_error(device_error, Gfn2PeriodicEmbeddingDeviceError::kInvalidOffsets);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 &&
      atomicAdd(device_error, 0u) !=
          static_cast<std::uint32_t>(Gfn2PeriodicEmbeddingDeviceError::kSuccess)) {
    *sequence_active = 0u;
  }
}

/* Validate every ragged extent before a system kernel indexes device data. */
__global__ void topology_preflight_kernel(Gfn2PeriodicEmbeddingDeviceBatch batch,
                                          std::uint32_t* device_error) {
  if (atomicAdd(device_error, 0u) !=
      static_cast<std::uint32_t>(Gfn2PeriodicEmbeddingDeviceError::kSuccess)) {
    return;
  }
  if (threadIdx.x == 0 && (batch.atom_offsets[0] != 0 || batch.matrix_offsets[0] != 0 ||
                           batch.atom_offsets[batch.batch_size] != batch.total_atoms ||
                           batch.matrix_offsets[batch.batch_size] != batch.total_matrix_elements)) {
    record_error(device_error, Gfn2PeriodicEmbeddingDeviceError::kInvalidOffsets);
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    const std::int64_t atom_begin = batch.atom_offsets[system];
    const std::int64_t atom_end = batch.atom_offsets[system + 1];
    const std::int64_t matrix_begin = batch.matrix_offsets[system];
    const std::int64_t matrix_end = batch.matrix_offsets[system + 1];
    if (atom_begin < 0 || atom_begin > atom_end || atom_end > batch.total_atoms ||
        matrix_begin < 0 || matrix_begin > matrix_end || matrix_end > batch.total_matrix_elements) {
      record_error(device_error, Gfn2PeriodicEmbeddingDeviceError::kInvalidOffsets);
      continue;
    }
    const std::int64_t atom_count = atom_end - atom_begin;
    const bool square_representable = atom_count == 0 || atom_count <= kMaximumInt64 / atom_count;
    if (!square_representable || matrix_end - matrix_begin != atom_count * atom_count) {
      record_error(device_error, Gfn2PeriodicEmbeddingDeviceError::kInvalidOffsets);
    }
  }
}

/*
 * Preserve preflight/upstream validity before per-system errors can make the
 * shared device_error nonzero. All system blocks consult this snapshot.
 */
__global__ void capture_sequence_kernel(const std::uint32_t* device_error,
                                        std::uint32_t* sequence_active) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *sequence_active =
        atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                static_cast<std::uint32_t>(Gfn2PeriodicEmbeddingDeviceError::kSuccess)
            ? 1u
            : 0u;
  }
}

struct SystemRanges {
  std::int64_t atom_begin;
  std::int64_t atom_end;
  std::int64_t matrix_begin;
};

/* Return the upper-triangle representative used by the CPU implementation. */
__device__ double symmetric_element(const double* matrix, std::int64_t atom_count, std::int64_t row,
                                    std::int64_t column) {
  return column < row ? matrix[column * atom_count + row] : matrix[row * atom_count + column];
}

__global__ void periodic_embedding_kernel(Gfn2PeriodicEmbeddingDeviceBatch batch,
                                          const double* mixed_atomic_charges,
                                          const double* raw_atomic_charges,
                                          double* atomic_potentials, double* energies,
                                          gpuxtb_status_t* system_statuses,
                                          Gfn2PeriodicEmbeddingDeviceWorkspace workspace,
                                          std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int active;
  __shared__ int valid;
  __shared__ double system_energy;

  if (threadIdx.x == 0) {
    active = atomicAdd(workspace.sequence_active, 0u) == 1u ? 1 : 0;
    valid = 1;
    if (active != 0) {
      ranges.atom_begin = batch.atom_offsets[system];
      ranges.atom_end = batch.atom_offsets[system + 1];
      ranges.matrix_begin = batch.matrix_offsets[system];
    }
  }
  __syncthreads();
  if (active == 0) {
    return;
  }

  const std::int64_t atom_count = ranges.atom_end - ranges.atom_begin;
  if (atom_count == 0) {
    if (threadIdx.x == 0) {
      energies[system] = 0.0;
      system_statuses[system] = GPUXTB_STATUS_SUCCESS;
    }
    return;
  }

  const double* const shifts = batch.shifts + ranges.atom_begin;
  const double* const matrix = batch.response_matrices + ranges.matrix_begin;
  const double* const mixed = mixed_atomic_charges + ranges.atom_begin;
  const double* const raw = raw_atomic_charges + ranges.atom_begin;

  /* Validate vector inputs before any arithmetic uses a peer thread's value. */
  for (std::int64_t atom = threadIdx.x; atom < atom_count; atom += blockDim.x) {
    if (!isfinite(shifts[atom])) {
      record_error(device_error, Gfn2PeriodicEmbeddingDeviceError::kNonfiniteShift);
      atomicExch(&valid, 0);
    }
    if (!isfinite(mixed[atom])) {
      record_error(device_error, Gfn2PeriodicEmbeddingDeviceError::kNonfiniteMixedCharge);
      atomicExch(&valid, 0);
    }
    if (!isfinite(raw[atom])) {
      record_error(device_error, Gfn2PeriodicEmbeddingDeviceError::kNonfiniteRawCharge);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    if (threadIdx.x == 0) {
      system_statuses[system] = GPUXTB_STATUS_INTERNAL_ERROR;
    }
    return;
  }

  /*
   * One thread owns a complete matrix row. Both dot products traverse columns
   * in CPU order and use the upper-triangle representative, preserving the
   * CPU result even when accepted +0.0/-0.0 pairs have different sign bits.
   */
  for (std::int64_t row = threadIdx.x; row < atom_count; row += blockDim.x) {
    double mixed_response = 0.0;
    double raw_response = 0.0;
    bool row_valid = true;
    for (std::int64_t column = 0; column < atom_count; ++column) {
      const double upper = symmetric_element(matrix, atom_count, row, column);
      if (!isfinite(upper)) {
        record_error(device_error, Gfn2PeriodicEmbeddingDeviceError::kNonfiniteResponseMatrix);
        row_valid = false;
        break;
      }
      if (column != row) {
        const double lower =
            column < row ? matrix[row * atom_count + column] : matrix[column * atom_count + row];
        if (!isfinite(lower)) {
          record_error(device_error, Gfn2PeriodicEmbeddingDeviceError::kNonfiniteResponseMatrix);
          row_valid = false;
          break;
        }
        if (upper != lower) {
          record_error(device_error, Gfn2PeriodicEmbeddingDeviceError::kNonsymmetricResponseMatrix);
          row_valid = false;
          break;
        }
      }
      mixed_response = fma(upper, mixed[column], mixed_response);
      raw_response = fma(upper, raw[column], raw_response);
      if (!isfinite(mixed_response)) {
        record_error(device_error, Gfn2PeriodicEmbeddingDeviceError::kNonfinitePotentialArithmetic);
        row_valid = false;
        break;
      }
      if (!isfinite(raw_response)) {
        record_error(device_error, Gfn2PeriodicEmbeddingDeviceError::kNonfiniteEnergyArithmetic);
        row_valid = false;
        break;
      }
    }
    if (row_valid) {
      const double potential = shifts[row] + mixed_response;
      if (!isfinite(potential)) {
        record_error(device_error, Gfn2PeriodicEmbeddingDeviceError::kNonfinitePotentialArithmetic);
        row_valid = false;
      } else {
        const std::int64_t atom = ranges.atom_begin + row;
        workspace.potential_scratch[atom] = potential;
        workspace.raw_response_scratch[atom] = raw_response;
      }
    }
    if (!row_valid) {
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    if (threadIdx.x == 0) {
      system_statuses[system] = GPUXTB_STATUS_INTERNAL_ERROR;
    }
    return;
  }

  /* Serial checked accumulation matches the CPU atom order exactly. */
  if (threadIdx.x == 0) {
    double linear_energy = 0.0;
    double quadratic_energy = 0.0;
    for (std::int64_t row = 0; row < atom_count; ++row) {
      const std::int64_t atom = ranges.atom_begin + row;
      linear_energy = fma(raw[row], shifts[row], linear_energy);
      quadratic_energy = fma(raw[row], workspace.raw_response_scratch[atom], quadratic_energy);
      if (!isfinite(linear_energy) || !isfinite(quadratic_energy)) {
        record_error(device_error, Gfn2PeriodicEmbeddingDeviceError::kNonfiniteEnergyArithmetic);
        valid = 0;
        break;
      }
    }
    if (valid != 0) {
      system_energy = fma(0.5, quadratic_energy, linear_energy);
      if (!isfinite(system_energy)) {
        record_error(device_error, Gfn2PeriodicEmbeddingDeviceError::kNonfiniteEnergyArithmetic);
        valid = 0;
      }
    }
  }
  __syncthreads();
  if (valid == 0) {
    if (threadIdx.x == 0) {
      system_statuses[system] = GPUXTB_STATUS_INTERNAL_ERROR;
    }
    return;
  }

  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    atomic_potentials[atom] = workspace.potential_scratch[atom];
  }
  if (threadIdx.x == 0) {
    energies[system] = system_energy;
    system_statuses[system] = GPUXTB_STATUS_SUCCESS;
  }
}

__global__ void scc_potential_kernel(Gfn2PeriodicEmbeddingDeviceBatch batch,
                                     const double* mixed_atomic_charges,
                                     Gfn2SccIterationDeviceActivity activity,
                                     double* atomic_potentials,
                                     Gfn2PeriodicEmbeddingDeviceWorkspace workspace,
                                     std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int active;
  __shared__ int valid;
  if (threadIdx.x == 0) {
    active = 0;
    valid = 1;
    /* Canonical gate, member byte, then the stage-local plan latch. */
    if (scc_sequence_is_open(activity) && activity.active_mask[system] == 1u &&
        atomicAdd(workspace.sequence_active, 0u) == 1u && system_is_valid(system_errors, system)) {
      active = 1;
      ranges.atom_begin = batch.atom_offsets[system];
      ranges.atom_end = batch.atom_offsets[system + 1];
      ranges.matrix_begin = batch.matrix_offsets[system];
    }
  }
  __syncthreads();
  if (active == 0) {
    return;
  }
  const std::int64_t atom_count = ranges.atom_end - ranges.atom_begin;
  if (atom_count == 0) {
    return;
  }
  const double* const shifts = batch.shifts + ranges.atom_begin;
  const double* const matrix = batch.response_matrices + ranges.matrix_begin;
  const double* const mixed = mixed_atomic_charges + ranges.atom_begin;
  for (std::int64_t atom = threadIdx.x; atom < atom_count; atom += blockDim.x) {
    if (!isfinite(shifts[atom])) {
      record_system_error(system_errors, system, Gfn2PeriodicEmbeddingDeviceError::kNonfiniteShift);
      atomicExch(&valid, 0);
    }
    if (!isfinite(mixed[atom])) {
      record_system_error(system_errors, system,
                          Gfn2PeriodicEmbeddingDeviceError::kNonfiniteMixedCharge);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  for (std::int64_t row = threadIdx.x; row < atom_count; row += blockDim.x) {
    double response = 0.0;
    bool row_valid = true;
    for (std::int64_t column = 0; column < atom_count; ++column) {
      const double upper = symmetric_element(matrix, atom_count, row, column);
      if (!isfinite(upper)) {
        record_system_error(system_errors, system,
                            Gfn2PeriodicEmbeddingDeviceError::kNonfiniteResponseMatrix);
        row_valid = false;
        break;
      }
      if (column != row) {
        const double lower =
            column < row ? matrix[row * atom_count + column] : matrix[column * atom_count + row];
        if (!isfinite(lower)) {
          record_system_error(system_errors, system,
                              Gfn2PeriodicEmbeddingDeviceError::kNonfiniteResponseMatrix);
          row_valid = false;
          break;
        }
        if (upper != lower) {
          record_system_error(system_errors, system,
                              Gfn2PeriodicEmbeddingDeviceError::kNonsymmetricResponseMatrix);
          row_valid = false;
          break;
        }
      }
      response = fma(upper, mixed[column], response);
      if (!isfinite(response)) {
        record_system_error(system_errors, system,
                            Gfn2PeriodicEmbeddingDeviceError::kNonfinitePotentialArithmetic);
        row_valid = false;
        break;
      }
    }
    if (row_valid) {
      const double potential = shifts[row] + response;
      if (isfinite(potential)) {
        workspace.potential_scratch[ranges.atom_begin + row] = potential;
      } else {
        record_system_error(system_errors, system,
                            Gfn2PeriodicEmbeddingDeviceError::kNonfinitePotentialArithmetic);
        row_valid = false;
      }
    }
    if (!row_valid) {
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0 || !system_is_valid(system_errors, system)) {
    return;
  }
  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    atomic_potentials[atom] = workspace.potential_scratch[atom];
  }
}

__global__ void scc_energy_kernel(Gfn2PeriodicEmbeddingDeviceBatch batch,
                                  const double* raw_atomic_charges,
                                  Gfn2SccIterationDeviceActivity activity, double* energies,
                                  Gfn2PeriodicEmbeddingDeviceWorkspace workspace,
                                  std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int active;
  __shared__ int valid;
  __shared__ double system_energy;
  if (threadIdx.x == 0) {
    active = 0;
    valid = 1;
    if (scc_sequence_is_open(activity) && activity.active_mask[system] == 1u &&
        atomicAdd(workspace.sequence_active, 0u) == 1u && system_is_valid(system_errors, system)) {
      active = 1;
      ranges.atom_begin = batch.atom_offsets[system];
      ranges.atom_end = batch.atom_offsets[system + 1];
      ranges.matrix_begin = batch.matrix_offsets[system];
    }
  }
  __syncthreads();
  if (active == 0) {
    return;
  }
  const std::int64_t atom_count = ranges.atom_end - ranges.atom_begin;
  if (atom_count == 0) {
    if (threadIdx.x == 0) {
      energies[system] = 0.0;
    }
    return;
  }
  const double* const shifts = batch.shifts + ranges.atom_begin;
  const double* const matrix = batch.response_matrices + ranges.matrix_begin;
  const double* const raw = raw_atomic_charges + ranges.atom_begin;
  for (std::int64_t atom = threadIdx.x; atom < atom_count; atom += blockDim.x) {
    if (!isfinite(shifts[atom])) {
      record_system_error(system_errors, system, Gfn2PeriodicEmbeddingDeviceError::kNonfiniteShift);
      atomicExch(&valid, 0);
    }
    if (!isfinite(raw[atom])) {
      record_system_error(system_errors, system,
                          Gfn2PeriodicEmbeddingDeviceError::kNonfiniteRawCharge);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  for (std::int64_t row = threadIdx.x; row < atom_count; row += blockDim.x) {
    double response = 0.0;
    bool row_valid = true;
    for (std::int64_t column = 0; column < atom_count; ++column) {
      const double upper = symmetric_element(matrix, atom_count, row, column);
      if (!isfinite(upper)) {
        record_system_error(system_errors, system,
                            Gfn2PeriodicEmbeddingDeviceError::kNonfiniteResponseMatrix);
        row_valid = false;
        break;
      }
      if (column != row) {
        const double lower =
            column < row ? matrix[row * atom_count + column] : matrix[column * atom_count + row];
        if (!isfinite(lower)) {
          record_system_error(system_errors, system,
                              Gfn2PeriodicEmbeddingDeviceError::kNonfiniteResponseMatrix);
          row_valid = false;
          break;
        }
        if (upper != lower) {
          record_system_error(system_errors, system,
                              Gfn2PeriodicEmbeddingDeviceError::kNonsymmetricResponseMatrix);
          row_valid = false;
          break;
        }
      }
      response = fma(upper, raw[column], response);
      if (!isfinite(response)) {
        record_system_error(system_errors, system,
                            Gfn2PeriodicEmbeddingDeviceError::kNonfiniteEnergyArithmetic);
        row_valid = false;
        break;
      }
    }
    if (row_valid) {
      workspace.raw_response_scratch[ranges.atom_begin + row] = response;
    } else {
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  if (threadIdx.x == 0) {
    double linear_energy = 0.0;
    double quadratic_energy = 0.0;
    for (std::int64_t row = 0; row < atom_count; ++row) {
      linear_energy = fma(raw[row], shifts[row], linear_energy);
      quadratic_energy =
          fma(raw[row], workspace.raw_response_scratch[ranges.atom_begin + row], quadratic_energy);
      if (!isfinite(linear_energy) || !isfinite(quadratic_energy)) {
        record_system_error(system_errors, system,
                            Gfn2PeriodicEmbeddingDeviceError::kNonfiniteEnergyArithmetic);
        valid = 0;
        break;
      }
    }
    if (valid != 0) {
      system_energy = fma(0.5, quadratic_energy, linear_energy);
      if (!isfinite(system_energy)) {
        record_system_error(system_errors, system,
                            Gfn2PeriodicEmbeddingDeviceError::kNonfiniteEnergyArithmetic);
        valid = 0;
      }
    }
  }
  __syncthreads();
  if (valid != 0 && threadIdx.x == 0 && system_is_valid(system_errors, system)) {
    energies[system] = system_energy;
  }
}

bool is_aligned(const void* pointer, std::size_t alignment) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

template <typename T>
bool required_pointer(const T* pointer, std::int64_t elements) noexcept {
  return elements == 0 || is_aligned(pointer, alignof(T));
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool make_address_range(const void* pointer, std::int64_t elements, std::size_t element_size,
                        AddressRange& range) noexcept {
  if (elements < 0 || element_size == 0u ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(elements) * element_size;
  if (bytes == 0u) {
    range = {};
    return true;
  }
  if (pointer == nullptr) {
    return false;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  range = {begin, begin + bytes};
  return true;
}

bool ranges_overlap(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin < second.end && second.begin < first.end;
}

template <std::size_t ReadCount, std::size_t WriteCount>
bool writable_ranges_are_disjoint(const std::array<AddressRange, ReadCount>& reads,
                                  const std::array<AddressRange, WriteCount>& writes) noexcept {
  for (std::size_t write = 0u; write < WriteCount; ++write) {
    for (const AddressRange& read : reads) {
      if (ranges_overlap(writes[write], read)) {
        return false;
      }
    }
    for (std::size_t other = write + 1u; other < WriteCount; ++other) {
      if (ranges_overlap(writes[write], writes[other])) {
        return false;
      }
    }
  }
  return true;
}

cudaError_t validate_launcher_arguments(const Gfn2PeriodicEmbeddingDeviceBatch& batch,
                                        const double* mixed_atomic_charges,
                                        const double* raw_atomic_charges, double* atomic_potentials,
                                        double* energies, gpuxtb_status_t* system_statuses,
                                        const Gfn2PeriodicEmbeddingDeviceWorkspace& workspace,
                                        std::uint32_t* device_error) noexcept {
  if (batch.batch_size <= 0 || batch.total_atoms < 0 || batch.total_matrix_elements < 0 ||
      batch.batch_size == std::numeric_limits<std::int64_t>::max() ||
      batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      batch.atom_offset_count != batch.batch_size + 1 ||
      batch.matrix_offset_count != batch.batch_size + 1 ||
      batch.shift_elements < batch.total_atoms ||
      batch.response_elements < batch.total_matrix_elements || batch.plan_token == 0u ||
      batch.atom_offsets == nullptr || batch.matrix_offsets == nullptr ||
      !required_pointer(batch.shifts, batch.total_atoms) ||
      !required_pointer(batch.response_matrices, batch.total_matrix_elements) ||
      !required_pointer(mixed_atomic_charges, batch.total_atoms) ||
      !required_pointer(raw_atomic_charges, batch.total_atoms) ||
      !required_pointer(atomic_potentials, batch.total_atoms) || energies == nullptr ||
      system_statuses == nullptr || workspace.atom_elements < batch.total_atoms ||
      workspace.sequence_elements < 1 || workspace.plan_token != batch.plan_token ||
      !required_pointer(workspace.potential_scratch, batch.total_atoms) ||
      !required_pointer(workspace.raw_response_scratch, batch.total_atoms) ||
      !is_aligned(energies, alignof(double)) ||
      !is_aligned(system_statuses, alignof(gpuxtb_status_t)) ||
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max())
               ? cudaErrorInvalidConfiguration
               : cudaErrorInvalidValue;
  }

  std::array<AddressRange, 6> reads;
  std::array<AddressRange, 7> writes;
  if (!make_address_range(batch.atom_offsets, batch.atom_offset_count, sizeof(*batch.atom_offsets),
                          reads[0]) ||
      !make_address_range(batch.matrix_offsets, batch.matrix_offset_count,
                          sizeof(*batch.matrix_offsets), reads[1]) ||
      !make_address_range(batch.shifts, batch.total_atoms, sizeof(*batch.shifts), reads[2]) ||
      !make_address_range(batch.response_matrices, batch.total_matrix_elements,
                          sizeof(*batch.response_matrices), reads[3]) ||
      !make_address_range(mixed_atomic_charges, batch.total_atoms, sizeof(*mixed_atomic_charges),
                          reads[4]) ||
      !make_address_range(raw_atomic_charges, batch.total_atoms, sizeof(*raw_atomic_charges),
                          reads[5]) ||
      !make_address_range(atomic_potentials, batch.total_atoms, sizeof(*atomic_potentials),
                          writes[0]) ||
      !make_address_range(energies, batch.batch_size, sizeof(*energies), writes[1]) ||
      !make_address_range(system_statuses, batch.batch_size, sizeof(*system_statuses), writes[2]) ||
      !make_address_range(workspace.potential_scratch, batch.total_atoms,
                          sizeof(*workspace.potential_scratch), writes[3]) ||
      !make_address_range(workspace.raw_response_scratch, batch.total_atoms,
                          sizeof(*workspace.raw_response_scratch), writes[4]) ||
      !make_address_range(workspace.sequence_active, 1, sizeof(*workspace.sequence_active),
                          writes[5]) ||
      !make_address_range(device_error, 1, sizeof(*device_error), writes[6]) ||
      !writable_ranges_are_disjoint(reads, writes)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t validate_scc_launcher_arguments(
    const Gfn2PeriodicEmbeddingDeviceBatch& batch, const double* charges,
    const Gfn2SccIterationDeviceActivity& activity, double* output,
    const Gfn2PeriodicEmbeddingDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, bool potential_mode) noexcept {
  const bool mode_pointers_valid =
      potential_mode ? required_pointer(output, batch.total_atoms) &&
                           required_pointer(workspace.potential_scratch, batch.total_atoms)
                     : output != nullptr && is_aligned(output, alignof(double)) &&
                           required_pointer(workspace.raw_response_scratch, batch.total_atoms);
  if (batch.batch_size <= 0 || batch.total_atoms < 0 || batch.total_matrix_elements < 0 ||
      batch.batch_size == std::numeric_limits<std::int64_t>::max() ||
      batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      batch.atom_offset_count != batch.batch_size + 1 ||
      batch.matrix_offset_count != batch.batch_size + 1 ||
      batch.shift_elements < batch.total_atoms ||
      batch.response_elements < batch.total_matrix_elements || batch.plan_token == 0u ||
      batch.atom_offsets == nullptr || batch.matrix_offsets == nullptr ||
      !required_pointer(batch.shifts, batch.total_atoms) ||
      !required_pointer(batch.response_matrices, batch.total_matrix_elements) ||
      !required_pointer(charges, batch.total_atoms) || !mode_pointers_valid ||
      activity.batch_elements != batch.batch_size || activity.sequence_elements != 1 ||
      activity.plan_token != batch.plan_token || activity.active_mask == nullptr ||
      activity.sequence_active == nullptr || workspace.atom_elements < batch.total_atoms ||
      workspace.sequence_elements < 1 || workspace.plan_token != batch.plan_token ||
      system_errors == nullptr || device_error == nullptr ||
      !is_aligned(activity.active_mask, alignof(std::uint8_t)) ||
      !is_aligned(activity.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max())
               ? cudaErrorInvalidConfiguration
               : cudaErrorInvalidValue;
  }
  std::array<AddressRange, 7> reads;
  std::array<AddressRange, 5> writes;
  double* const scratch =
      potential_mode ? workspace.potential_scratch : workspace.raw_response_scratch;
  const std::int64_t output_elements = potential_mode ? batch.total_atoms : batch.batch_size;
  if (!make_address_range(batch.atom_offsets, batch.atom_offset_count, sizeof(*batch.atom_offsets),
                          reads[0]) ||
      !make_address_range(batch.matrix_offsets, batch.matrix_offset_count,
                          sizeof(*batch.matrix_offsets), reads[1]) ||
      !make_address_range(batch.shifts, batch.total_atoms, sizeof(*batch.shifts), reads[2]) ||
      !make_address_range(batch.response_matrices, batch.total_matrix_elements,
                          sizeof(*batch.response_matrices), reads[3]) ||
      !make_address_range(charges, batch.total_atoms, sizeof(*charges), reads[4]) ||
      !make_address_range(activity.active_mask, batch.batch_size, sizeof(*activity.active_mask),
                          reads[5]) ||
      !make_address_range(activity.sequence_active, 1, sizeof(*activity.sequence_active),
                          reads[6]) ||
      !make_address_range(output, output_elements, sizeof(*output), writes[0]) ||
      !make_address_range(scratch, batch.total_atoms, sizeof(*scratch), writes[1]) ||
      !make_address_range(workspace.sequence_active, 1, sizeof(*workspace.sequence_active),
                          writes[2]) ||
      !make_address_range(system_errors, batch.batch_size, sizeof(*system_errors), writes[3]) ||
      !make_address_range(device_error, 1, sizeof(*device_error), writes[4]) ||
      !writable_ranges_are_disjoint(reads, writes)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t check_launch() noexcept { return cudaPeekAtLastError(); }

}  // namespace

cudaError_t reset_gfn2_periodic_embedding_device_error_cuda(std::uint32_t* device_error,
                                                            cudaStream_t stream) noexcept {
  if (device_error == nullptr || !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  return cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream);
}

cudaError_t evaluate_gfn2_periodic_embedding_cuda(
    const Gfn2PeriodicEmbeddingDeviceBatch& batch, const double* mixed_atomic_charges,
    const double* raw_atomic_charges, double* atomic_potentials, double* energies,
    gpuxtb_status_t* system_statuses, const Gfn2PeriodicEmbeddingDeviceWorkspace& workspace,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  const cudaError_t validation = validate_launcher_arguments(
      batch, mixed_atomic_charges, raw_atomic_charges, atomic_potentials, energies, system_statuses,
      workspace, device_error);
  if (validation != cudaSuccess) {
    return validation;
  }

  topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, device_error);
  cudaError_t status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace.sequence_active);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  periodic_embedding_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                              stream>>>(batch, mixed_atomic_charges, raw_atomic_charges,
                                        atomic_potentials, energies, system_statuses, workspace,
                                        device_error);
  return check_launch();
}

cudaError_t reset_gfn2_periodic_embedding_scc_device_errors_cuda(std::int64_t batch_size,
                                                                 std::uint32_t* system_errors,
                                                                 std::uint32_t* device_error,
                                                                 cudaStream_t stream) noexcept {
  AddressRange system_range;
  AddressRange device_range;
  if (batch_size <= 0 || system_errors == nullptr || device_error == nullptr ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t)) ||
      !make_address_range(system_errors, batch_size, sizeof(*system_errors), system_range) ||
      !make_address_range(device_error, 1, sizeof(*device_error), device_range) ||
      ranges_overlap(system_range, device_range)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(
      system_errors, 0, static_cast<std::size_t>(batch_size) * sizeof(*system_errors), stream);
  if (status != cudaSuccess) {
    return status;
  }
  return cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream);
}

cudaError_t evaluate_gfn2_periodic_embedding_scc_potential_cuda(
    const Gfn2PeriodicEmbeddingDeviceBatch& batch, std::uint64_t expected_geometry_generation,
    const double* mixed_atomic_charges, const Gfn2SccIterationDeviceActivity& activity,
    double* atomic_potentials, const Gfn2PeriodicEmbeddingDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  const cudaError_t validation =
      validate_scc_launcher_arguments(batch, mixed_atomic_charges, activity, atomic_potentials,
                                      workspace, system_errors, device_error, true);
  if (validation != cudaSuccess) {
    return validation;
  }
  scc_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
      batch, expected_geometry_generation, activity, workspace.sequence_active, device_error);
  cudaError_t status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  scc_potential_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                         stream>>>(batch, mixed_atomic_charges, activity, atomic_potentials,
                                   workspace, system_errors);
  return check_launch();
}

cudaError_t evaluate_gfn2_periodic_embedding_scc_energy_cuda(
    const Gfn2PeriodicEmbeddingDeviceBatch& batch, std::uint64_t expected_geometry_generation,
    const double* raw_atomic_charges, const Gfn2SccIterationDeviceActivity& activity,
    double* energies, const Gfn2PeriodicEmbeddingDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  const cudaError_t validation = validate_scc_launcher_arguments(
      batch, raw_atomic_charges, activity, energies, workspace, system_errors, device_error, false);
  if (validation != cudaSuccess) {
    return validation;
  }
  scc_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
      batch, expected_geometry_generation, activity, workspace.sequence_active, device_error);
  cudaError_t status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  scc_energy_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0, stream>>>(
      batch, raw_atomic_charges, activity, energies, workspace, system_errors);
  return check_launch();
}

}  // namespace gpuxtb::detail::cuda
