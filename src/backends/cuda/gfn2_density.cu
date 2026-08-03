#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_density.cuh"

namespace gpuxtb::detail::cuda {
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
                                std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (atomicAdd(workspace.sequence_active, 0u) == 0u || input.active[system] != 1u ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t orbital_begin = batch.orbital_offsets[system];
  const std::int64_t orbital_end = batch.orbital_offsets[system + 1];
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t count = orbital_end - orbital_begin;
  const std::int64_t pair_count = triangle_inclusive(count);
  for (std::int64_t pair = threadIdx.x; pair < pair_count; pair += blockDim.x) {
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

bool validate_launch(const Gfn2DensityDeviceBatch& batch, const Gfn2DensityDeviceInput& input,
                     const Gfn2DensityDeviceResults& results,
                     const Gfn2DensityDeviceWorkspace& workspace, std::uint32_t* system_errors,
                     std::uint32_t* device_error) noexcept {
  std::int64_t two_orbitals = 0;
  if (batch.batch_size <= 0 || batch.batch_size > std::numeric_limits<int>::max() ||
      batch.total_orbitals <= 0 || batch.total_matrix_elements <= 0 || batch.plan_token == 0u ||
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

}  // namespace

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
  if (!validate_launch(batch, input, results, workspace, system_errors, device_error)) {
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
  contract_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0, stream>>>(
      batch, input, workspace, system_errors, device_error);
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

}  // namespace gpuxtb::detail::cuda
