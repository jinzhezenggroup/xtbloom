#include <array>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_geometry.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr std::int64_t kInt64Maximum = 9223372036854775807LL;
constexpr double kCutoffSquaredBohr = 25.0 * 25.0;
constexpr double kMinimumDistanceSquared = 1.0e-12;
constexpr double kFirstSteepness = 10.0;
constexpr double kSecondSteepness = 20.0;
constexpr double kSecondRadiusShiftBohr = 2.0;
constexpr double kGfn1Steepness = 16.0;

struct SystemRanges {
  std::int64_t atom_begin;
  std::int64_t atom_end;
  std::int64_t pair_begin;
  std::int64_t pair_end;
};

struct PairValues {
  double distance;
  double inverse_distance;
  double count;
  double derivative_over_distance;
};

__host__ __device__ std::int64_t triangle_count(std::int64_t value) {
  return (value & 1LL) == 0LL ? (value / 2LL) * (value - 1LL) : value * ((value - 1LL) / 2LL);
}

__device__ bool sequence_is_active(const std::uint32_t* sequence_active) {
  return atomicAdd(const_cast<std::uint32_t*>(sequence_active), 0u) == 1u;
}

__device__ bool system_is_valid(const std::uint32_t* system_errors, std::int64_t system) {
  return atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) ==
         static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess);
}

__device__ void record_error(std::uint32_t* device_error, Gfn2GeometryDeviceError error) {
  atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    std::uint32_t* device_error, Gfn2GeometryDeviceError error) {
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  if (atomicCAS(system_errors + system,
                static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess),
                code) == static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess)) {
    atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess), code);
  }
}

struct LogisticValue {
  double value;
  double derivative;
};

/* Stable logistic value and derivative. Computing both from the same
 * nonpositive exponential preserves the tiny GFN1 derivative after the value
 * itself has rounded to one, matching the CPU reference. */
__device__ LogisticValue logistic_with_derivative(double argument) {
  if (argument >= 0.0) {
    const double exponential = exp(-argument);
    const double denominator = 1.0 + exponential;
    return {1.0 / denominator, exponential / (denominator * denominator)};
  }
  const double exponential = exp(argument);
  const double denominator = 1.0 + exponential;
  return {exponential / denominator, exponential / (denominator * denominator)};
}

__device__ double logistic(double argument) { return logistic_with_derivative(argument).value; }

__device__ bool evaluate_pair(XtbModelFlavor model, double dx, double dy, double dz, double radius,
                              PairValues* values) {
  const double distance_squared = dx * dx + dy * dy + dz * dz;
  if (!isfinite(distance_squared)) {
    return false;
  }

  if (distance_squared < kMinimumDistanceSquared) {
    if (model != XtbModelFlavor::kGfn1) return false;
    /* GFN1 skips sub-threshold coincident pairs. Publish a canonical zero
     * cache entry so later CN/VJP consumers never observe stale arena data.
     * Equality remains active, matching the CPU contract. */
    *values = {};
    return true;
  }

  values->distance = sqrt(distance_squared);
  values->inverse_distance = 1.0 / values->distance;
  values->count = 0.0;
  values->derivative_over_distance = 0.0;
  if (!(values->distance > 0.0) || !isfinite(values->distance) ||
      !(values->inverse_distance > 0.0) || !isfinite(values->inverse_distance)) {
    return false;
  }
  if (distance_squared > kCutoffSquaredBohr) {
    return true;
  }

  const double inverse_distance_squared = values->inverse_distance * values->inverse_distance;
  double derivative = 0.0;
  if (model == XtbModelFlavor::kGfn1) {
    const LogisticValue count =
        logistic_with_derivative(kGfn1Steepness * (radius * values->inverse_distance - 1.0));
    values->count = count.value;
    derivative = -kGfn1Steepness * radius * inverse_distance_squared * count.derivative;
  } else if (model == XtbModelFlavor::kGfn2) {
    const double shifted_radius = radius + kSecondRadiusShiftBohr;
    const double first = logistic(kFirstSteepness * (radius * values->inverse_distance - 1.0));
    const double second =
        logistic(kSecondSteepness * (shifted_radius * values->inverse_distance - 1.0));
    values->count = first * second;
    derivative = -inverse_distance_squared *
                 (kFirstSteepness * radius * first * (1.0 - first) * second +
                  kSecondSteepness * shifted_radius * second * (1.0 - second) * first);
  } else {
    return false;
  }
  values->derivative_over_distance = derivative * values->inverse_distance;
  return values->count >= 0.0 && values->count <= 1.0 && isfinite(values->count) &&
         isfinite(values->derivative_over_distance);
}

/* Validate all offset endpoints before any later kernel subtracts them. */
__global__ void topology_preflight_kernel(Gfn2GeometryDeviceBatch batch,
                                          std::uint32_t* device_error) {
  if (atomicAdd(device_error, 0u) !=
      static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess)) {
    return;
  }
  if (threadIdx.x == 0 && (batch.atom_offsets[0] != 0 || batch.pair_offsets[0] != 0 ||
                           batch.atom_offsets[batch.batch_size] != batch.total_atoms ||
                           batch.pair_offsets[batch.batch_size] != batch.total_pairs)) {
    record_error(device_error, Gfn2GeometryDeviceError::kInvalidOffsets);
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    const std::int64_t atom_begin = batch.atom_offsets[system];
    const std::int64_t atom_end = batch.atom_offsets[system + 1];
    const std::int64_t pair_begin = batch.pair_offsets[system];
    const std::int64_t pair_end = batch.pair_offsets[system + 1];
    const bool endpoints_valid = atom_begin >= 0 && atom_begin <= atom_end &&
                                 atom_end <= batch.total_atoms && pair_begin >= 0 &&
                                 pair_begin <= pair_end && pair_end <= batch.total_pairs;
    if (!endpoints_valid) {
      record_error(device_error, Gfn2GeometryDeviceError::kInvalidOffsets);
      continue;
    }
    const std::int64_t atom_count = atom_end - atom_begin;
    const std::int64_t pair_count = pair_end - pair_begin;
    const bool triangle_representable =
        atom_count <= 1 || atom_count <= kInt64Maximum / (atom_count - 1);
    if (!triangle_representable || pair_count != triangle_count(atom_count)) {
      record_error(device_error, Gfn2GeometryDeviceError::kInvalidOffsets);
    }
  }
}

/* Snapshot topology/upstream validity before peer-local errors set device_error. */
__global__ void capture_sequence_kernel(const std::uint32_t* device_error,
                                        std::uint32_t* sequence_active) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *sequence_active = atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                               static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess)
                           ? 1u
                           : 0u;
  }
}

__device__ bool load_system(const Gfn2GeometryDeviceBatch& batch, std::int64_t system,
                            const std::uint32_t* sequence_active,
                            const std::uint32_t* system_errors, SystemRanges* ranges, int* valid) {
  if (threadIdx.x == 0) {
    *valid = sequence_is_active(sequence_active) && system_is_valid(system_errors, system) ? 1 : 0;
    if (*valid != 0) {
      ranges->atom_begin = batch.atom_offsets[system];
      ranges->atom_end = batch.atom_offsets[system + 1];
      ranges->pair_begin = batch.pair_offsets[system];
      ranges->pair_end = batch.pair_offsets[system + 1];
    }
  }
  __syncthreads();
  return *valid != 0;
}

__global__ void build_pair_cache_kernel(Gfn2GeometryDeviceBatch batch, const double* positions,
                                        double* pair_scratch, const std::uint32_t* sequence_active,
                                        std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  if (!load_system(batch, system, sequence_active, system_errors, &ranges, &valid)) {
    return;
  }
  /* All threads must finish load_system's shared valid read before it is reused. */
  __syncthreads();

  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    if (!isfinite(positions[coordinate]) || !isfinite(positions[coordinate + 1]) ||
        !isfinite(positions[coordinate + 2])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2GeometryDeviceError::kNonfinitePosition);
      atomicExch(&valid, 0);
    }
    const double radius = batch.covalent_radii[atom];
    if (!(radius > 0.0) || !isfinite(radius)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2GeometryDeviceError::kInvalidCovalentRadius);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  /* One thread owns an upper atom and writes all of its packed lower pairs. */
  for (std::int64_t upper = ranges.atom_begin + 1 + threadIdx.x; upper < ranges.atom_end;
       upper += blockDim.x) {
    const std::int64_t upper_coordinate = upper * 3;
    const std::int64_t local_upper = upper - ranges.atom_begin;
    for (std::int64_t lower = ranges.atom_begin; lower < upper; ++lower) {
      const std::int64_t lower_coordinate = lower * 3;
      const double dx = positions[upper_coordinate] - positions[lower_coordinate];
      const double dy = positions[upper_coordinate + 1] - positions[lower_coordinate + 1];
      const double dz = positions[upper_coordinate + 2] - positions[lower_coordinate + 2];
      if (!isfinite(dx) || !isfinite(dy) || !isfinite(dz)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2GeometryDeviceError::kCoordinateDifferenceOverflow);
        continue;
      }
      const double radius = batch.covalent_radii[upper] + batch.covalent_radii[lower];
      PairValues values{};
      if (!evaluate_pair(batch.model, dx, dy, dz, radius, &values)) {
        const double distance_squared = dx * dx + dy * dy + dz * dz;
        const Gfn2GeometryDeviceError error =
            isfinite(distance_squared) && distance_squared < kMinimumDistanceSquared
                ? Gfn2GeometryDeviceError::kCoincidentAtoms
                : Gfn2GeometryDeviceError::kNonfinitePairArithmetic;
        record_system_error(system_errors, system, device_error, error);
        continue;
      }
      const std::int64_t local_lower = lower - ranges.atom_begin;
      const std::int64_t pair = ranges.pair_begin + triangle_count(local_upper) + local_lower;
      double* const output = pair_scratch + pair * kGfn2GeometryPairDataElements;
      output[0] = dx;
      output[1] = dy;
      output[2] = dz;
      output[3] = values.distance;
      output[4] = values.inverse_distance;
      output[5] = values.count;
      output[6] = values.derivative_over_distance;
    }
  }
}

__global__ void build_coordination_kernel(Gfn2GeometryDeviceBatch batch, const double* pair_scratch,
                                          double* coordination_scratch,
                                          const std::uint32_t* sequence_active,
                                          std::uint32_t* system_errors,
                                          std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  if (!load_system(batch, system, sequence_active, system_errors, &ranges, &valid)) {
    return;
  }
  __syncthreads();

  /*
   * Each atom accumulates peers in ascending atom order. This is the same
   * contribution order seen by the CPU nested-pair loop and avoids atomics.
   */
  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    double coordination = 0.0;
    for (std::int64_t peer = ranges.atom_begin; peer < ranges.atom_end; ++peer) {
      if (peer == atom) {
        continue;
      }
      const std::int64_t upper = atom > peer ? atom : peer;
      const std::int64_t lower = atom > peer ? peer : atom;
      const std::int64_t pair = ranges.pair_begin + triangle_count(upper - ranges.atom_begin) +
                                (lower - ranges.atom_begin);
      coordination += pair_scratch[pair * kGfn2GeometryPairDataElements + 5];
      if (!isfinite(coordination)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2GeometryDeviceError::kNonfiniteCoordinationArithmetic);
        break;
      }
    }
    coordination_scratch[atom] = coordination;
  }
}

__global__ void publish_geometry_kernel(Gfn2GeometryDeviceBatch batch,
                                        std::uint64_t geometry_generation,
                                        Gfn2GeometryDeviceCache cache,
                                        Gfn2GeometryDeviceWorkspace workspace,
                                        const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!sequence_is_active(workspace.sequence_active) || !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  const std::int64_t pair_begin = batch.pair_offsets[system];
  const std::int64_t pair_end = batch.pair_offsets[system + 1];
  for (std::int64_t pair = pair_begin + threadIdx.x; pair < pair_end; pair += blockDim.x) {
    for (std::int64_t component = 0; component < kGfn2GeometryPairDataElements; ++component) {
      cache.pair_data[pair * kGfn2GeometryPairDataElements + component] =
          workspace.pair_scratch[pair * kGfn2GeometryPairDataElements + component];
    }
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    cache.coordination_numbers[atom] = workspace.coordination_scratch[atom];
  }
  if (threadIdx.x == 0) {
    cache.geometry_generations[system] = geometry_generation;
  }
}

__global__ void coordination_vjp_kernel(
    Gfn2GeometryDeviceBatch batch, Gfn2GeometryDeviceCache cache, std::uint64_t scalar_generation,
    const std::uint64_t* device_generation, const double* dE_dcn, const double* gradients,
    double* gradient_scratch, const std::uint32_t* sequence_active, std::uint32_t* system_errors,
    std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  if (!load_system(batch, system, sequence_active, system_errors, &ranges, &valid)) {
    return;
  }
  __syncthreads();
  const std::uint64_t geometry_generation =
      device_generation == nullptr ? scalar_generation
                                   : atomicAdd(reinterpret_cast<unsigned long long*>(
                                                   const_cast<std::uint64_t*>(device_generation)),
                                               0ULL);
  if (threadIdx.x == 0 && cache.geometry_generations[system] != geometry_generation) {
    record_system_error(system_errors, system, device_error,
                        Gfn2GeometryDeviceError::kStaleGeometry);
    valid = 0;
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    if (!isfinite(cache.coordination_numbers[atom]) || cache.coordination_numbers[atom] < 0.0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2GeometryDeviceError::kInvalidCache);
      atomicExch(&valid, 0);
    }
    if (!isfinite(dE_dcn[atom])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2GeometryDeviceError::kNonfiniteAdjoint);
      atomicExch(&valid, 0);
    }
    const std::int64_t coordinate = atom * 3;
    if (!isfinite(gradients[coordinate]) || !isfinite(gradients[coordinate + 1]) ||
        !isfinite(gradients[coordinate + 2])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2GeometryDeviceError::kNonfiniteGradientSeed);
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t upper = ranges.atom_begin + 1 + threadIdx.x; upper < ranges.atom_end;
       upper += blockDim.x) {
    const std::int64_t local_upper = upper - ranges.atom_begin;
    for (std::int64_t lower = ranges.atom_begin; lower < upper; ++lower) {
      const std::int64_t pair =
          ranges.pair_begin + triangle_count(local_upper) + (lower - ranges.atom_begin);
      const double* const data = cache.pair_data + pair * kGfn2GeometryPairDataElements;
      const double radius = batch.covalent_radii[upper] + batch.covalent_radii[lower];
      PairValues expected{};
      const bool expected_valid =
          evaluate_pair(batch.model, data[0], data[1], data[2], radius, &expected);
      if (!expected_valid || data[3] != expected.distance || data[4] != expected.inverse_distance ||
          data[5] != expected.count || data[6] != expected.derivative_over_distance) {
        record_system_error(system_errors, system, device_error,
                            Gfn2GeometryDeviceError::kInvalidCache);
        atomicExch(&valid, 0);
      }
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    double contribution[3] = {0.0, 0.0, 0.0};
    bool finite_result = true;
    for (std::int64_t peer = ranges.atom_begin; finite_result && peer < ranges.atom_end; ++peer) {
      if (peer == atom) {
        continue;
      }
      const bool target_is_upper = atom > peer;
      const std::int64_t upper = target_is_upper ? atom : peer;
      const std::int64_t lower = target_is_upper ? peer : atom;
      const std::int64_t pair = ranges.pair_begin + triangle_count(upper - ranges.atom_begin) +
                                (lower - ranges.atom_begin);
      const double* const data = cache.pair_data + pair * kGfn2GeometryPairDataElements;
      const double scale = (dE_dcn[upper] + dE_dcn[lower]) * data[6];
      const double sign = target_is_upper ? 1.0 : -1.0;
      for (int axis = 0; axis < 3; ++axis) {
        contribution[axis] += sign * scale * data[axis];
        finite_result = finite_result && isfinite(contribution[axis]);
      }
    }
    const std::int64_t coordinate = atom * 3;
    for (int axis = 0; axis < 3; ++axis) {
      const double updated = gradients[coordinate + axis] + contribution[axis];
      if (!finite_result || !isfinite(updated)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2GeometryDeviceError::kNonfiniteVjpArithmetic);
        finite_result = false;
      } else {
        gradient_scratch[coordinate + axis] = updated;
      }
    }
  }
}

__global__ void publish_vjp_kernel(Gfn2GeometryDeviceBatch batch, const double* gradient_scratch,
                                   double* gradients, const std::uint32_t* sequence_active,
                                   const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!sequence_is_active(sequence_active) || !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    gradients[coordinate] = gradient_scratch[coordinate];
    gradients[coordinate + 1] = gradient_scratch[coordinate + 1];
    gradients[coordinate + 2] = gradient_scratch[coordinate + 2];
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
                        AddressRange* range) noexcept {
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

struct RequiredElements {
  std::int64_t coordinates;
  std::int64_t pair_data;
};

cudaError_t validate_common(const Gfn2GeometryDeviceBatch& batch,
                            const Gfn2GeometryDeviceCache& cache,
                            const Gfn2GeometryDeviceWorkspace& workspace,
                            std::uint32_t* system_errors, std::uint32_t* device_error,
                            RequiredElements* required) noexcept {
  if (batch.batch_size <= 0 || batch.total_atoms < 0 || batch.total_pairs < 0 ||
      batch.batch_size == std::numeric_limits<std::int64_t>::max() ||
      batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      batch.total_atoms > std::numeric_limits<std::int64_t>::max() / 3 ||
      batch.total_pairs >
          std::numeric_limits<std::int64_t>::max() / kGfn2GeometryPairDataElements ||
      batch.atom_offset_elements != batch.batch_size + 1 ||
      batch.pair_offset_elements != batch.batch_size + 1 ||
      batch.covalent_radius_elements < batch.total_atoms || batch.plan_token == 0u ||
      !valid_xtb_model_flavor(batch.model) ||
      !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.pair_offsets, alignof(std::int64_t)) ||
      !required_pointer(batch.covalent_radii, batch.total_atoms) ||
      cache.plan_token != batch.plan_token || workspace.plan_token != batch.plan_token ||
      cache.generation_elements < batch.batch_size ||
      !is_aligned(cache.geometry_generations, alignof(std::uint64_t)) ||
      workspace.sequence_elements < 1 ||
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max())
               ? cudaErrorInvalidConfiguration
               : cudaErrorInvalidValue;
  }
  required->coordinates = batch.total_atoms * 3;
  required->pair_data = batch.total_pairs * kGfn2GeometryPairDataElements;
  if (batch.coordinate_elements < required->coordinates) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t validate_update(const Gfn2GeometryDeviceBatch& batch, const double* positions,
                            std::uint64_t geometry_generation, const Gfn2GeometryDeviceCache& cache,
                            const Gfn2GeometryDeviceWorkspace& workspace,
                            std::uint32_t* system_errors, std::uint32_t* device_error) noexcept {
  RequiredElements required{};
  cudaError_t status =
      validate_common(batch, cache, workspace, system_errors, device_error, &required);
  if (status != cudaSuccess) {
    return status;
  }
  if (geometry_generation == 0u || !required_pointer(positions, required.coordinates) ||
      cache.pair_data_elements < required.pair_data ||
      cache.coordination_elements < batch.total_atoms ||
      !required_pointer(cache.pair_data, required.pair_data) ||
      !required_pointer(cache.coordination_numbers, batch.total_atoms) ||
      workspace.pair_elements < required.pair_data ||
      workspace.coordination_elements < batch.total_atoms ||
      !required_pointer(workspace.pair_scratch, required.pair_data) ||
      !required_pointer(workspace.coordination_scratch, batch.total_atoms)) {
    return cudaErrorInvalidValue;
  }

  std::array<AddressRange, 4> reads;
  std::array<AddressRange, 8> writes;
  if (!make_address_range(batch.atom_offsets, batch.atom_offset_elements,
                          sizeof(*batch.atom_offsets), &reads[0]) ||
      !make_address_range(batch.pair_offsets, batch.pair_offset_elements,
                          sizeof(*batch.pair_offsets), &reads[1]) ||
      !make_address_range(batch.covalent_radii, batch.total_atoms, sizeof(*batch.covalent_radii),
                          &reads[2]) ||
      !make_address_range(positions, required.coordinates, sizeof(*positions), &reads[3]) ||
      !make_address_range(cache.pair_data, required.pair_data, sizeof(*cache.pair_data),
                          &writes[0]) ||
      !make_address_range(cache.coordination_numbers, batch.total_atoms,
                          sizeof(*cache.coordination_numbers), &writes[1]) ||
      !make_address_range(cache.geometry_generations, batch.batch_size,
                          sizeof(*cache.geometry_generations), &writes[2]) ||
      !make_address_range(workspace.pair_scratch, required.pair_data,
                          sizeof(*workspace.pair_scratch), &writes[3]) ||
      !make_address_range(workspace.coordination_scratch, batch.total_atoms,
                          sizeof(*workspace.coordination_scratch), &writes[4]) ||
      !make_address_range(workspace.sequence_active, 1, sizeof(*workspace.sequence_active),
                          &writes[5]) ||
      !make_address_range(system_errors, batch.batch_size, sizeof(*system_errors), &writes[6]) ||
      !make_address_range(device_error, 1, sizeof(*device_error), &writes[7]) ||
      !writable_ranges_are_disjoint(reads, writes)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t validate_vjp(const Gfn2GeometryDeviceBatch& batch, const Gfn2GeometryDeviceCache& cache,
                         std::uint64_t scalar_generation, const std::uint64_t* device_generation,
                         const double* dE_dcn, double* gradients,
                         const Gfn2GeometryDeviceWorkspace& workspace, std::uint32_t* system_errors,
                         std::uint32_t* device_error) noexcept {
  RequiredElements required{};
  cudaError_t status =
      validate_common(batch, cache, workspace, system_errors, device_error, &required);
  if (status != cudaSuccess) {
    return status;
  }
  if ((device_generation == nullptr
           ? scalar_generation == 0u
           : scalar_generation != 0u || !is_aligned(device_generation, alignof(std::uint64_t))) ||
      cache.pair_data_elements < required.pair_data ||
      cache.coordination_elements < batch.total_atoms ||
      !required_pointer(cache.pair_data, required.pair_data) ||
      !required_pointer(cache.coordination_numbers, batch.total_atoms) ||
      !required_pointer(dE_dcn, batch.total_atoms) ||
      !required_pointer(gradients, required.coordinates) ||
      workspace.gradient_elements < required.coordinates ||
      !required_pointer(workspace.gradient_scratch, required.coordinates)) {
    return cudaErrorInvalidValue;
  }

  std::array<AddressRange, 8> reads;
  std::array<AddressRange, 5> writes;
  if (!make_address_range(batch.atom_offsets, batch.atom_offset_elements,
                          sizeof(*batch.atom_offsets), &reads[0]) ||
      !make_address_range(batch.pair_offsets, batch.pair_offset_elements,
                          sizeof(*batch.pair_offsets), &reads[1]) ||
      !make_address_range(batch.covalent_radii, batch.total_atoms, sizeof(*batch.covalent_radii),
                          &reads[2]) ||
      !make_address_range(cache.pair_data, required.pair_data, sizeof(*cache.pair_data),
                          &reads[3]) ||
      !make_address_range(cache.coordination_numbers, batch.total_atoms,
                          sizeof(*cache.coordination_numbers), &reads[4]) ||
      !make_address_range(cache.geometry_generations, batch.batch_size,
                          sizeof(*cache.geometry_generations), &reads[5]) ||
      !make_address_range(dE_dcn, batch.total_atoms, sizeof(*dE_dcn), &reads[6]) ||
      !make_address_range(device_generation, device_generation == nullptr ? 0 : 1,
                          sizeof(std::uint64_t), &reads[7]) ||
      !make_address_range(gradients, required.coordinates, sizeof(*gradients), &writes[0]) ||
      !make_address_range(workspace.gradient_scratch, required.coordinates,
                          sizeof(*workspace.gradient_scratch), &writes[1]) ||
      !make_address_range(workspace.sequence_active, 1, sizeof(*workspace.sequence_active),
                          &writes[2]) ||
      !make_address_range(system_errors, batch.batch_size, sizeof(*system_errors), &writes[3]) ||
      !make_address_range(device_error, 1, sizeof(*device_error), &writes[4]) ||
      !writable_ranges_are_disjoint(reads, writes)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t check_launch() noexcept { return cudaPeekAtLastError(); }

}  // namespace

cudaError_t reset_gfn2_geometry_device_errors_cuda(std::int64_t batch_size,
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
  AddressRange system_range;
  AddressRange device_range;
  if (!make_address_range(system_errors, batch_size, sizeof(*system_errors), &system_range) ||
      !make_address_range(device_error, 1, sizeof(*device_error), &device_range) ||
      ranges_overlap(system_range, device_range)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(
      system_errors, 0, static_cast<std::size_t>(batch_size) * sizeof(*system_errors), stream);
  return status == cudaSuccess ? cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream)
                               : status;
}

cudaError_t update_gfn2_geometry_cache_cuda(
    const Gfn2GeometryDeviceBatch& batch, const double* positions,
    std::uint64_t geometry_generation, const Gfn2GeometryDeviceCache& cache,
    const Gfn2GeometryDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  cudaError_t status = validate_update(batch, positions, geometry_generation, cache, workspace,
                                       system_errors, device_error);
  if (status != cudaSuccess) {
    return status;
  }
  topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace.sequence_active);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  const unsigned int blocks = static_cast<unsigned int>(batch.batch_size);
  build_pair_cache_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, positions, workspace.pair_scratch, workspace.sequence_active, system_errors,
      device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  build_coordination_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, workspace.pair_scratch, workspace.coordination_scratch, workspace.sequence_active,
      system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_geometry_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(batch, geometry_generation,
                                                                   cache, workspace, system_errors);
  return check_launch();
}

static cudaError_t add_coordination_vjp_impl(
    const Gfn2GeometryDeviceBatch& batch, const Gfn2GeometryDeviceCache& cache,
    std::uint64_t scalar_generation, const Gfn2GeometryEpochDevice* geometry_epoch,
    const double* dE_dcn, double* gradients, const Gfn2GeometryDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (geometry_epoch != nullptr &&
      (geometry_epoch->value_elements != 1 || geometry_epoch->plan_token != batch.plan_token)) {
    return cudaErrorInvalidValue;
  }
  const std::uint64_t* const device_generation =
      geometry_epoch == nullptr ? nullptr : geometry_epoch->value;
  cudaError_t status = validate_vjp(batch, cache, scalar_generation, device_generation, dE_dcn,
                                    gradients, workspace, system_errors, device_error);
  if (status != cudaSuccess) {
    return status;
  }
  topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace.sequence_active);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  const unsigned int blocks = static_cast<unsigned int>(batch.batch_size);
  coordination_vjp_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, cache, scalar_generation, device_generation, dE_dcn, gradients,
      workspace.gradient_scratch, workspace.sequence_active, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_vjp_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, workspace.gradient_scratch, gradients, workspace.sequence_active, system_errors);
  return check_launch();
}

cudaError_t add_gfn2_coordination_vjp_cuda(
    const Gfn2GeometryDeviceBatch& batch, const Gfn2GeometryDeviceCache& cache,
    std::uint64_t geometry_generation, const double* dE_dcn, double* gradients,
    const Gfn2GeometryDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  return add_coordination_vjp_impl(batch, cache, geometry_generation, nullptr, dE_dcn, gradients,
                                   workspace, system_errors, device_error, stream);
}

cudaError_t add_gfn2_coordination_vjp_cuda(
    const Gfn2GeometryDeviceBatch& batch, const Gfn2GeometryDeviceCache& cache,
    const Gfn2GeometryEpochDevice& geometry_epoch, const double* dE_dcn, double* gradients,
    const Gfn2GeometryDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  return add_coordination_vjp_impl(batch, cache, 0u, &geometry_epoch, dE_dcn, gradients, workspace,
                                   system_errors, device_error, stream);
}

}  // namespace xtbloom::detail::cuda
