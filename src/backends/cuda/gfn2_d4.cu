#include <array>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/cuda_atomics.cuh"
#include "backends/cuda/gfn2_d4.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr double kChargeScalingHeight = 3.0;
constexpr double kChargeScalingSteepness = 2.0;
constexpr double kReferenceWeightFactor = 6.0;
constexpr double kMinimumWeightNorm = 1.4916681462400413e-154;
constexpr double kCoordinationCutoffSquared =
    kGfn2D4CoordinationCutoffBohr * kGfn2D4CoordinationCutoffBohr;
constexpr double kTwoBodyCutoffSquared = kGfn2D4TwoBodyCutoffBohr * kGfn2D4TwoBodyCutoffBohr;
constexpr double kAtmCutoffSquared = kGfn2D4AtmCutoffBohr * kGfn2D4AtmCutoffBohr;
constexpr double kMinimumDistanceSquared = 1.0e-12;
constexpr double kCoordinationSteepness = 7.5;
constexpr double kEnK4 = 4.10451;
constexpr double kEnK5 = 19.08857;
constexpr double kEnK6 = 2.0 * 11.28174 * 11.28174;
constexpr double kInverseSqrtPi = 0.5641895835477562869480794515607726;
constexpr double kDispersionS6 = 1.0;
constexpr double kDispersionS8 = 2.7;
constexpr double kDispersionA1 = 0.52;
constexpr double kDispersionA2 = 5.0;
constexpr double kDispersionS9 = 5.0;
constexpr double kAtmExponent = 16.0;
constexpr std::int64_t kMaximumInt64 = 9223372036854775807LL;

static_assert(kThreadsPerBlock > 0);
static_assert((kThreadsPerBlock & (kThreadsPerBlock - 1)) == 0,
              "the energy reduction requires a power-of-two block size");

__device__ void record_error(std::uint32_t* device_error, Gfn2D4DeviceError error) {
  atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ bool system_is_valid(const Gfn2D4DeviceWorkspace& workspace, std::int64_t system) {
  return atomicAdd(workspace.system_errors + system, 0u) ==
         static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess);
}

__device__ void record_system_error(const Gfn2D4DeviceWorkspace& workspace, std::int64_t system,
                                    Gfn2D4DeviceError error) {
  atomicCAS(workspace.system_errors + system,
            static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ bool sequence_is_valid(const std::uint32_t* device_error) {
  return atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
         static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess);
}

/* SCC consumers always inspect the canonical sequence gate before any member
 * byte or numerical buffer. The ledger is immutable for the duration of a
 * queued stage, so ordinary byte loads are sufficient after the stream-ordered
 * uint32 gate read. */
__device__ bool scc_sequence_is_open(const Gfn2SccIterationDeviceActivity& activity) {
  return atomicAdd(const_cast<std::uint32_t*>(activity.sequence_active), 0u) == 1u;
}

__device__ bool scc_system_is_active(const Gfn2SccIterationDeviceActivity& activity,
                                     std::int64_t system) {
  return scc_sequence_is_open(activity) && activity.active_mask[system] == 1u;
}

/*
 * Read the sticky error once per block. A per-thread atomic read creates severe
 * contention on a single address for large batches, while a non-atomic load
 * may race with record_error from a block that started earlier.
 */
__device__ bool block_sequence_is_valid(const std::uint32_t* device_error,
                                        std::uint32_t& shared_error) {
  if (threadIdx.x == 0) {
    shared_error = atomicAdd(const_cast<std::uint32_t*>(device_error), 0u);
  }
  __syncthreads();
  return shared_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess);
}

__device__ bool block_system_is_valid(const Gfn2D4DeviceWorkspace& workspace, std::int64_t system,
                                      const std::uint32_t* device_error,
                                      std::uint32_t& shared_error) {
  const bool sequence_valid = block_sequence_is_valid(device_error, shared_error);
  /* Finish every shared read before lane 0 reuses the slot for system status. */
  __syncthreads();
  if (!sequence_valid) {
    return false;
  }
  if (threadIdx.x == 0) {
    shared_error = atomicAdd(workspace.system_errors + system, 0u);
  }
  __syncthreads();
  return shared_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess);
}

__device__ std::int64_t system_for_atom(Gfn2D4DeviceBatch batch, std::int64_t atom) {
  std::int64_t lower = 0;
  std::int64_t upper = batch.batch_size;
  while (lower + 1 < upper) {
    const std::int64_t middle = lower + (upper - lower) / 2;
    if (batch.atom_offsets[middle] <= atom) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return lower;
}

__device__ std::int64_t system_for_pair(Gfn2D4DeviceBatch batch, std::int64_t pair) {
  std::int64_t lower = 0;
  std::int64_t upper = batch.batch_size;
  while (lower + 1 < upper) {
    const std::int64_t middle = lower + (upper - lower) / 2;
    if (batch.pair_offsets[middle] <= pair) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return lower;
}

struct D4GeometrySystemRanges {
  std::int64_t atom_begin;
  std::int64_t atom_end;
  std::int64_t pair_begin;
  std::int64_t pair_end;
};

__device__ bool geometry_sequence_is_active(const Gfn2D4DeviceWorkspace& workspace) {
  return atomicAdd(workspace.geometry_sequence_active, 0u) == 1u;
}

/*
 * Load one ragged member only after the immutable-sequence gate and the
 * peer-local sticky status both permit work. All callers invoke this helper
 * uniformly across a block because it contains a block synchronization.
 */
__device__ bool load_geometry_system(const Gfn2D4DeviceBatch& batch, std::int64_t system,
                                     const Gfn2D4DeviceWorkspace& workspace,
                                     D4GeometrySystemRanges* ranges, int* valid) {
  if (threadIdx.x == 0) {
    *valid = geometry_sequence_is_active(workspace) && system_is_valid(workspace, system) ? 1 : 0;
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

/* Match the CPU plan's checked n*(n-1)/2 construction without signed overflow. */
__device__ bool packed_pair_count(std::int64_t atoms, std::int64_t& pairs) {
  if (atoms < 0 || (atoms > 0 && atoms - 1 > kMaximumInt64 / atoms)) {
    return false;
  }
  pairs = atoms * (atoms - 1) / 2;
  return true;
}

__device__ double charge_scale(double a, double c, double qref, double qmod) {
  if (qmod < 0.0) {
    return exp(a);
  }
  return exp(a * (1.0 - exp(c * (1.0 - qref / qmod))));
}

__device__ double charge_scale_derivative(double a, double c, double qref, double qmod) {
  if (qmod < 0.0) {
    return 0.0;
  }
  const double inner = exp(c * (1.0 - qref / qmod));
  return -a * c * inner * charge_scale(a, c, qref, qmod) * qref / (qmod * qmod);
}

/* Scalar generations support ordinary launches; a stable device pointer is
 * used by the replay-safe overloads so CUDA Graph capture never freezes an
 * old host epoch. */
struct D4GenerationSource {
  std::uint64_t scalar = 0u;
  const std::uint64_t* device = nullptr;
};

__device__ std::uint64_t load_generation(D4GenerationSource source) {
  return source.device == nullptr ? source.scalar : *source.device;
}

struct D4PairGeometry {
  double vector[3];
  double distance_squared;
  double damping;
  double damping_derivative;
};

/* Re-evaluate the exact dense-cache pair arithmetic from positions.  The
 * caller chooses which role cutoff to consume; damping itself keeps the
 * inclusive 50-bohr predicate used by the historical cache. */
__device__ bool evaluate_d4_pair_geometry(Gfn2D4DeviceBatch batch,
                                          Gfn2D4DeviceParameters parameters,
                                          const double* positions, std::int64_t first,
                                          std::int64_t second, D4PairGeometry* values) {
  const std::int64_t first_coordinate = first * 3;
  const std::int64_t second_coordinate = second * 3;
  values->vector[0] = positions[first_coordinate] - positions[second_coordinate];
  values->vector[1] = positions[first_coordinate + 1] - positions[second_coordinate + 1];
  values->vector[2] = positions[first_coordinate + 2] - positions[second_coordinate + 2];
  if (!isfinite(values->vector[0]) || !isfinite(values->vector[1]) ||
      !isfinite(values->vector[2])) {
    return false;
  }
  values->distance_squared = values->vector[0] * values->vector[0] +
                             values->vector[1] * values->vector[1] +
                             values->vector[2] * values->vector[2];
  if (!isfinite(values->distance_squared) || values->distance_squared < kMinimumDistanceSquared) {
    return false;
  }
  values->damping = 0.0;
  values->damping_derivative = 0.0;
  if (values->distance_squared <= kTwoBodyCutoffSquared) {
    const Gfn2D4DeviceElementData first_element =
        parameters.elements[batch.atomic_numbers[first] - 1];
    const Gfn2D4DeviceElementData second_element =
        parameters.elements[batch.atomic_numbers[second] - 1];
    const double rrij = 3.0 * first_element.r4r2 * second_element.r4r2;
    const double r0 = kDispersionA1 * sqrt(rrij) + kDispersionA2;
    const double r2_squared = values->distance_squared * values->distance_squared;
    const double r2_cubed = r2_squared * values->distance_squared;
    const double r0_squared = r0 * r0;
    const double r0_fourth = r0_squared * r0_squared;
    const double r0_sixth = r0_fourth * r0_squared;
    const double t6 = 1.0 / (r2_cubed + r0_sixth);
    const double t8 = 1.0 / (r2_squared * r2_squared + r0_fourth * r0_fourth);
    values->damping = kDispersionS6 * t6 + kDispersionS8 * rrij * t8;
    values->damping_derivative = kDispersionS6 * (-6.0 * r2_squared * t6 * t6) +
                                 kDispersionS8 * rrij * (-8.0 * r2_cubed * t8 * t8);
  }
  return isfinite(values->damping) && isfinite(values->damping_derivative) &&
         values->damping >= 0.0;
}

__device__ bool d4_pairlist_role_active(const Gfn2PairListConsumerView& view, std::int64_t system) {
  return view.active_mask == nullptr || view.active_mask[system] == 1u;
}

__device__ bool binary_search_neighbor(const Gfn2PairListConsumerView& view, std::int64_t atom,
                                       std::int64_t peer) {
  std::int64_t lower = view.neighbor_offsets[atom];
  std::int64_t upper = lower + view.neighbor_counts[atom];
  while (lower < upper) {
    const std::int64_t middle = lower + (upper - lower) / 2;
    const std::int64_t value = view.neighbors[middle];
    if (value < peer) {
      lower = middle + 1;
    } else {
      upper = middle;
    }
  }
  return lower < view.neighbor_offsets[atom] + view.neighbor_counts[atom] &&
         view.neighbors[lower] == peer;
}

/* Validate one committed role directly on device.  Pair/neighbor corruption
 * is peer-local; immutable batch topology corruption closes the sequence in
 * topology_preflight_kernel before this launch. */
__global__ void d4_pairlist_role_preflight_kernel(
    Gfn2D4DeviceBatch batch, Gfn2D4PairListDeviceCache cache, Gfn2PairListConsumerView view,
    D4GenerationSource generation_source, bool require_d4_coordination,
    const std::uint8_t* external_active_mask, const std::uint32_t* external_sequence_active,
    Gfn2D4DeviceWorkspace workspace, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!sequence_is_valid(device_error)) {
    return;
  }
  if (external_sequence_active != nullptr) {
    const std::uint32_t sequence =
        atomicAdd(const_cast<std::uint32_t*>(external_sequence_active), 0u);
    if (sequence != 1u) {
      if (sequence > 1u && threadIdx.x == 0) {
        record_error(device_error, Gfn2D4DeviceError::kInvalidActivity);
      }
      return;
    }
  }
  if (external_active_mask != nullptr) {
    const std::uint8_t active = external_active_mask[system];
    if (active > 1u) {
      if (threadIdx.x == 0) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kInvalidActivity);
      }
      return;
    }
    if (active == 0u) {
      return;
    }
  }
  __shared__ int valid;
  __shared__ std::int64_t atom_begin;
  __shared__ std::int64_t atom_end;
  __shared__ std::int64_t pair_begin;
  __shared__ std::int64_t pair_count;
  __shared__ std::uint64_t expected_generation;
  __shared__ unsigned long long neighbor_total;
  if (threadIdx.x == 0) {
    neighbor_total = 0u;
    valid = system_is_valid(workspace, system) ? 1 : 0;
    if (valid != 0) {
      const std::uint8_t role_active = view.active_mask == nullptr ? 1u : view.active_mask[system];
      const std::uint8_t eligible = view.eligible_mask[system];
      expected_generation = load_generation(generation_source);
      if (role_active > 1u || eligible > 1u) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kInvalidActivity);
        valid = 0;
      } else if (role_active == 0u) {
        valid = 0;
      } else if (eligible != 1u || expected_generation == 0u ||
                 view.committed_generations[system] != expected_generation) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kStaleGeometry);
        valid = 0;
      } else if (require_d4_coordination) {
        const std::uint8_t coordination_eligible = cache.coordination_eligible_mask[system];
        if (coordination_eligible > 1u) {
          record_system_error(workspace, system, Gfn2D4DeviceError::kInvalidActivity);
          valid = 0;
        } else if (coordination_eligible != 1u ||
                   cache.coordination_generations[system] != expected_generation) {
          record_system_error(workspace, system, Gfn2D4DeviceError::kStaleCoordination);
          valid = 0;
        }
      }
      if (valid != 0) {
        atom_begin = batch.atom_offsets[system];
        atom_end = batch.atom_offsets[system + 1];
        pair_begin = view.pair_offsets[system];
        pair_count = view.pair_counts[system];
        const std::int64_t pair_slot_end = view.pair_offsets[system + 1];
        if (pair_begin < 0 || pair_count < 0 || pair_count > view.max_pairs_per_system ||
            pair_slot_end < pair_begin || pair_begin > view.pair_count - pair_count ||
            pair_begin + pair_count > pair_slot_end) {
          record_system_error(workspace, system, Gfn2D4DeviceError::kInvalidPairList);
          valid = 0;
        }
      }
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    if (!isfinite(cache.positions[coordinate]) || !isfinite(cache.positions[coordinate + 1]) ||
        !isfinite(cache.positions[coordinate + 2]) ||
        (require_d4_coordination && !isfinite(cache.coordination_numbers[atom]))) {
      record_system_error(workspace, system,
                          require_d4_coordination ? Gfn2D4DeviceError::kInvalidCoordination
                                                  : Gfn2D4DeviceError::kNonfinitePosition);
      atomicExch(&valid, 0);
      continue;
    }
    const std::int64_t neighbor_begin = view.neighbor_offsets[atom];
    const std::int64_t neighbor_count = view.neighbor_counts[atom];
    const std::int64_t neighbor_slot_end = view.neighbor_offsets[atom + 1];
    if (neighbor_begin < 0 || neighbor_count < 0 || neighbor_count > view.max_neighbors_per_atom ||
        neighbor_begin > view.neighbor_count - neighbor_count ||
        neighbor_slot_end < neighbor_begin || neighbor_begin + neighbor_count > neighbor_slot_end) {
      record_system_error(workspace, system, Gfn2D4DeviceError::kInvalidPairList);
      atomicExch(&valid, 0);
      continue;
    }
    std::int64_t previous = -1;
    for (std::int64_t index = neighbor_begin; index < neighbor_begin + neighbor_count; ++index) {
      const std::int64_t peer = view.neighbors[index];
      if (peer < atom_begin || peer >= atom_end || peer == atom || peer <= previous) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kInvalidPairList);
        atomicExch(&valid, 0);
        break;
      }
      previous = peer;
    }
    atomicAdd(&neighbor_total, static_cast<unsigned long long>(neighbor_count));
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  if (threadIdx.x == 0 && neighbor_total != 2u * static_cast<unsigned long long>(pair_count)) {
    record_system_error(workspace, system, Gfn2D4DeviceError::kInvalidPairList);
    valid = 0;
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  for (std::int64_t local = threadIdx.x; local < pair_count; local += blockDim.x) {
    const Gfn2AtomPair pair = view.pairs[pair_begin + local];
    if (pair.first < atom_begin || pair.first >= pair.second || pair.second >= atom_end ||
        !binary_search_neighbor(view, pair.first, pair.second) ||
        !binary_search_neighbor(view, pair.second, pair.first)) {
      record_system_error(workspace, system, Gfn2D4DeviceError::kInvalidPairList);
      atomicExch(&valid, 0);
      continue;
    }
    if (local > 0) {
      const Gfn2AtomPair previous = view.pairs[pair_begin + local - 1];
      if (pair.second < previous.second ||
          (pair.second == previous.second && pair.first <= previous.first)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kInvalidPairList);
        atomicExch(&valid, 0);
      }
    }
  }
}

__device__ double d4_coordination_count(Gfn2D4DeviceParameters parameters, Gfn2D4DeviceBatch batch,
                                        std::int64_t first, std::int64_t second,
                                        double distance_squared) {
  const Gfn2D4DeviceElementData first_element =
      parameters.elements[batch.atomic_numbers[first] - 1];
  const Gfn2D4DeviceElementData second_element =
      parameters.elements[batch.atomic_numbers[second] - 1];
  const double radius = first_element.covalent_radius + second_element.covalent_radius;
  const double en_delta = fabs(first_element.electronegativity - second_element.electronegativity);
  const double en_factor = kEnK4 * exp(-((en_delta + kEnK5) * (en_delta + kEnK5)) / kEnK6);
  const double distance = sqrt(distance_squared);
  const double exponent = kCoordinationSteepness * (distance - radius) / radius;
  return 0.5 * en_factor * (1.0 + erf(-exponent));
}

__global__ void build_d4_pairlist_coordination_kernel(Gfn2D4DeviceBatch batch,
                                                      Gfn2D4DeviceParameters parameters,
                                                      Gfn2D4PairListDeviceCache cache,
                                                      D4GenerationSource generation_source,
                                                      Gfn2D4DeviceWorkspace workspace,
                                                      std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!sequence_is_valid(device_error) || !system_is_valid(workspace, system) ||
      !d4_pairlist_role_active(cache.coordination_pairs, system)) {
    return;
  }
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    double coordination = 0.0;
    const std::int64_t begin = cache.coordination_pairs.neighbor_offsets[atom];
    const std::int64_t end = begin + cache.coordination_pairs.neighbor_counts[atom];
    for (std::int64_t index = begin; index < end; ++index) {
      const std::int64_t peer = cache.coordination_pairs.neighbors[index];
      const std::int64_t first = atom < peer ? atom : peer;
      const std::int64_t second = atom < peer ? peer : atom;
      D4PairGeometry pair{};
      if (!evaluate_d4_pair_geometry(batch, parameters, cache.positions, first, second, &pair)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteGeometryArithmetic);
        return;
      }
      if (pair.distance_squared <= kCoordinationCutoffSquared) {
        coordination +=
            d4_coordination_count(parameters, batch, first, second, pair.distance_squared);
      }
      if (!isfinite(coordination)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteGeometryArithmetic);
        return;
      }
    }
    workspace.coordination_scratch[atom] = coordination;
  }
  static_cast<void>(generation_source);
}

__global__ void topology_preflight_kernel(Gfn2D4DeviceBatch batch,
                                          Gfn2D4DeviceParameters parameters,
                                          std::uint32_t* device_error) {
  if (blockIdx.x != 0) {
    return;
  }
  __shared__ std::uint32_t shared_error;
  __shared__ std::uint64_t partial_hash[kThreadsPerBlock];
  if (!block_sequence_is_valid(device_error, shared_error)) {
    return;
  }
  if (threadIdx.x == 0 && (batch.atom_offsets[0] != 0 || batch.pair_offsets[0] != 0 ||
                           batch.atom_offsets[batch.batch_size] != batch.total_atoms ||
                           batch.pair_offsets[batch.batch_size] != batch.total_pairs)) {
    record_error(device_error, Gfn2D4DeviceError::kInvalidOffsets);
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    const std::int64_t atom_begin = batch.atom_offsets[system];
    const std::int64_t atom_end = batch.atom_offsets[system + 1];
    const std::int64_t pair_begin = batch.pair_offsets[system];
    const std::int64_t pair_end = batch.pair_offsets[system + 1];
    if (atom_begin < 0 || atom_begin > atom_end || atom_end > batch.total_atoms || pair_begin < 0 ||
        pair_begin > pair_end || pair_end > batch.total_pairs) {
      record_error(device_error, Gfn2D4DeviceError::kInvalidOffsets);
      continue;
    }
    const std::int64_t atoms = atom_end - atom_begin;
    std::int64_t expected_pairs = 0;
    if (!packed_pair_count(atoms, expected_pairs) || pair_end - pair_begin != expected_pairs) {
      record_error(device_error, Gfn2D4DeviceError::kInvalidOffsets);
    }
  }
  std::uint64_t local_hash = 0u;
  for (std::int64_t atom = threadIdx.x; atom < batch.total_atoms; atom += blockDim.x) {
    const std::int32_t atomic_number = batch.atomic_numbers[atom];
    local_hash ^= gfn2_d4_atomic_number_hash_contribution(atomic_number, atom);
    if (atomic_number <= 0 || atomic_number > parameters.element_count) {
      record_error(device_error, Gfn2D4DeviceError::kInvalidAtomicNumber);
      continue;
    }
    const Gfn2D4DeviceElementData element = parameters.elements[atomic_number - 1];
    if (element.reference_count == 0 || element.reference_count > kGfn2D4MaximumReferences ||
        static_cast<std::int64_t>(element.reference_offset) + element.reference_count >
            parameters.reference_count ||
        !(element.covalent_radius > 0.0) || !isfinite(element.covalent_radius) ||
        !isfinite(element.electronegativity) || !(element.r4r2 > 0.0) || !isfinite(element.r4r2) ||
        !(element.effective_charge > 0.0) || !isfinite(element.effective_charge) ||
        !(element.hardness > 0.0) || !isfinite(element.hardness)) {
      record_error(device_error, Gfn2D4DeviceError::kInvalidParameterData);
    }
  }
  partial_hash[threadIdx.x] = local_hash;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partial_hash[threadIdx.x] ^= partial_hash[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const std::uint64_t atomic_number_hash =
        partial_hash[0] ^ 0x243f6a8885a308d3ULL ^
        gfn2_d4_hash_mix(static_cast<std::uint64_t>(batch.total_atoms));
    if (atomic_number_hash != batch.atomic_number_hash) {
      record_error(device_error, Gfn2D4DeviceError::kInvalidAtomicNumber);
    }
  }
}

/* Snapshot plan validity before peer-local numerical work begins. */
__global__ void capture_geometry_sequence_kernel(const std::uint32_t* device_error,
                                                 std::uint32_t* sequence_active) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *sequence_active = atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                               static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess)
                           ? 1u
                           : 0u;
  }
}

/*
 * Form the exact five-value CPU D4 pair layout in unpublished storage. One
 * thread owns an upper atom and therefore all packed lower pairs below it;
 * no pair atomics or inter-block reductions are required.
 */
__global__ void build_d4_geometry_pairs_kernel(Gfn2D4DeviceBatch batch,
                                               Gfn2D4DeviceParameters parameters,
                                               const double* positions,
                                               Gfn2D4DeviceWorkspace workspace) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ D4GeometrySystemRanges ranges;
  __shared__ int valid;
  if (!load_geometry_system(batch, system, workspace, &ranges, &valid)) {
    return;
  }
  /* Finish every shared valid read before position validation may update it. */
  __syncthreads();

  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    if (!isfinite(positions[coordinate]) || !isfinite(positions[coordinate + 1]) ||
        !isfinite(positions[coordinate + 2])) {
      record_system_error(workspace, system, Gfn2D4DeviceError::kNonfinitePosition);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  for (std::int64_t second = ranges.atom_begin + 1 + threadIdx.x; second < ranges.atom_end;
       second += blockDim.x) {
    const std::int64_t local_second = second - ranges.atom_begin;
    const std::int64_t second_coordinate = second * 3;
    const Gfn2D4DeviceElementData second_element =
        parameters.elements[batch.atomic_numbers[second] - 1];
    for (std::int64_t first = ranges.atom_begin; first < second; ++first) {
      const std::int64_t first_coordinate = first * 3;
      const double dx = positions[first_coordinate] - positions[second_coordinate];
      const double dy = positions[first_coordinate + 1] - positions[second_coordinate + 1];
      const double dz = positions[first_coordinate + 2] - positions[second_coordinate + 2];
      if (!isfinite(dx) || !isfinite(dy) || !isfinite(dz)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kCoordinateDifferenceOverflow);
        continue;
      }
      const double distance_squared = dx * dx + dy * dy + dz * dz;
      if (!isfinite(distance_squared)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteGeometryArithmetic);
        continue;
      }
      if (distance_squared < kMinimumDistanceSquared) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kCoincidentAtoms);
        continue;
      }

      double damping = 0.0;
      double damping_derivative = 0.0;
      if (distance_squared <= kTwoBodyCutoffSquared) {
        const Gfn2D4DeviceElementData first_element =
            parameters.elements[batch.atomic_numbers[first] - 1];
        const double rrij = 3.0 * first_element.r4r2 * second_element.r4r2;
        const double r0 = kDispersionA1 * sqrt(rrij) + kDispersionA2;
        const double r2_squared = distance_squared * distance_squared;
        const double r2_cubed = r2_squared * distance_squared;
        const double r0_squared = r0 * r0;
        const double r0_fourth = r0_squared * r0_squared;
        const double r0_sixth = r0_fourth * r0_squared;
        const double t6 = 1.0 / (r2_cubed + r0_sixth);
        const double t8 = 1.0 / (r2_squared * r2_squared + r0_fourth * r0_fourth);
        damping = kDispersionS6 * t6 + kDispersionS8 * rrij * t8;
        damping_derivative = kDispersionS6 * (-6.0 * r2_squared * t6 * t6) +
                             kDispersionS8 * rrij * (-8.0 * r2_cubed * t8 * t8);
      }
      if (!isfinite(damping) || !isfinite(damping_derivative)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteGeometryArithmetic);
        continue;
      }

      const std::int64_t local_first = first - ranges.atom_begin;
      const std::int64_t pair =
          ranges.pair_begin + local_second * (local_second - 1) / 2 + local_first;
      double* const output = workspace.pair_scratch + pair * kGfn2D4PairDataElements;
      output[0] = dx;
      output[1] = dy;
      output[2] = dz;
      output[3] = damping;
      output[4] = damping_derivative;
    }
  }
}

/*
 * Accumulate each atom's peers in ascending atom order. That is exactly the
 * contribution order induced by update_d4_geometry_cache_cpu's nested pair
 * loop, while assigning one output atom to one thread avoids atomics.
 */
__global__ void build_d4_coordination_kernel(Gfn2D4DeviceBatch batch,
                                             Gfn2D4DeviceParameters parameters,
                                             Gfn2D4DeviceWorkspace workspace) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ D4GeometrySystemRanges ranges;
  __shared__ int valid;
  if (!load_geometry_system(batch, system, workspace, &ranges, &valid)) {
    return;
  }

  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    double coordination = 0.0;
    const Gfn2D4DeviceElementData atom_element =
        parameters.elements[batch.atomic_numbers[atom] - 1];
    for (std::int64_t peer = ranges.atom_begin; peer < ranges.atom_end; ++peer) {
      if (peer == atom) {
        continue;
      }
      const std::int64_t first = atom < peer ? atom : peer;
      const std::int64_t second = atom < peer ? peer : atom;
      const std::int64_t local_first = first - ranges.atom_begin;
      const std::int64_t local_second = second - ranges.atom_begin;
      const std::int64_t pair =
          ranges.pair_begin + local_second * (local_second - 1) / 2 + local_first;
      const double* const values = workspace.pair_scratch + pair * kGfn2D4PairDataElements;
      const double distance_squared =
          values[0] * values[0] + values[1] * values[1] + values[2] * values[2];
      if (distance_squared <= kCoordinationCutoffSquared) {
        const Gfn2D4DeviceElementData peer_element =
            parameters.elements[batch.atomic_numbers[peer] - 1];
        const double radius = atom_element.covalent_radius + peer_element.covalent_radius;
        const double en_delta =
            fabs(atom_element.electronegativity - peer_element.electronegativity);
        const double en_factor = kEnK4 * exp(-((en_delta + kEnK5) * (en_delta + kEnK5)) / kEnK6);
        const double distance = sqrt(distance_squared);
        const double exponent = kCoordinationSteepness * (distance - radius) / radius;
        coordination += 0.5 * en_factor * (1.0 + erf(-exponent));
      }
      if (!isfinite(coordination)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteGeometryArithmetic);
        break;
      }
    }
    workspace.coordination_scratch[atom] = coordination;
  }
}

__global__ void publish_d4_geometry_kernel(Gfn2D4DeviceBatch batch, Gfn2D4DeviceCache cache,
                                           Gfn2D4DeviceWorkspace workspace) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!geometry_sequence_is_active(workspace) || !system_is_valid(workspace, system)) {
    return;
  }
  double* const pair_data = const_cast<double*>(cache.pair_data);
  double* const coordination_numbers = const_cast<double*>(cache.coordination_numbers);
  const std::int64_t pair_begin = batch.pair_offsets[system];
  const std::int64_t pair_end = batch.pair_offsets[system + 1];
  for (std::int64_t pair = pair_begin + threadIdx.x; pair < pair_end; pair += blockDim.x) {
    const std::int64_t base = pair * kGfn2D4PairDataElements;
    for (std::int64_t component = 0; component < kGfn2D4PairDataElements; ++component) {
      pair_data[base + component] = workspace.pair_scratch[base + component];
    }
  }
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    coordination_numbers[atom] = workspace.coordination_scratch[atom];
  }
}

/* A separate launch makes generation publication strictly follow cache data. */
__global__ void publish_d4_geometry_generation_kernel(Gfn2D4DeviceBatch batch,
                                                      std::uint64_t geometry_generation,
                                                      Gfn2D4DeviceWorkspace workspace) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (threadIdx.x == 0 && geometry_sequence_is_active(workspace) &&
      system_is_valid(workspace, system)) {
    workspace.geometry_generations[system] = geometry_generation;
  }
}

/* Validate activity and scalar cache provenance without touching topology or
 * numerical data. In particular, an all-inactive call accepts a stale cache
 * generation and is safe even when every device-side geometry buffer is
 * deliberately poisoned. */
__global__ void scc_activity_preflight_kernel(Gfn2D4DeviceBatch batch, Gfn2D4DeviceCache cache,
                                              std::uint64_t expected_geometry_generation,
                                              Gfn2SccIterationDeviceActivity activity,
                                              std::uint32_t* device_error) {
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
  }
  __syncthreads();
  if (run == 0) {
    if (threadIdx.x == 0 && invalid_activity != 0) {
      record_error(device_error, Gfn2D4DeviceError::kInvalidActivity);
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
  if (threadIdx.x == 0) {
    if (invalid_activity != 0) {
      record_error(device_error, Gfn2D4DeviceError::kInvalidActivity);
    } else if (any_active != 0 &&
               (expected_geometry_generation == 0u || cache.geometry_generation == 0u ||
                cache.geometry_generation != expected_geometry_generation)) {
      record_error(device_error, Gfn2D4DeviceError::kStaleGeometry);
    }
  }
}

/* SCC topology validation is activity-gated. The all-inactive fast path exits
 * before even the first ragged offset is loaded. */
__global__ void scc_topology_preflight_kernel(Gfn2D4DeviceBatch batch,
                                              Gfn2D4DeviceParameters parameters,
                                              Gfn2SccIterationDeviceActivity activity,
                                              std::uint32_t* device_error) {
  if (blockIdx.x != 0) {
    return;
  }
  __shared__ int run;
  __shared__ int any_active;
  __shared__ std::uint64_t partial_hash[kThreadsPerBlock];
  if (threadIdx.x == 0) {
    run = scc_sequence_is_open(activity) && sequence_is_valid(device_error) ? 1 : 0;
    any_active = 0;
  }
  __syncthreads();
  if (run == 0) {
    return;
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    if (activity.active_mask[system] == 1u) {
      atomicExch(&any_active, 1);
    }
  }
  __syncthreads();
  if (any_active == 0) {
    return;
  }
  if (threadIdx.x == 0 && (batch.atom_offsets[0] != 0 || batch.pair_offsets[0] != 0 ||
                           batch.atom_offsets[batch.batch_size] != batch.total_atoms ||
                           batch.pair_offsets[batch.batch_size] != batch.total_pairs)) {
    record_error(device_error, Gfn2D4DeviceError::kInvalidOffsets);
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    if (activity.active_mask[system] != 1u) {
      continue;
    }
    const std::int64_t atom_begin = batch.atom_offsets[system];
    const std::int64_t atom_end = batch.atom_offsets[system + 1];
    const std::int64_t pair_begin = batch.pair_offsets[system];
    const std::int64_t pair_end = batch.pair_offsets[system + 1];
    if (atom_begin < 0 || atom_begin > atom_end || atom_end > batch.total_atoms || pair_begin < 0 ||
        pair_begin > pair_end || pair_end > batch.total_pairs) {
      record_error(device_error, Gfn2D4DeviceError::kInvalidOffsets);
      continue;
    }
    std::int64_t expected_pairs = 0;
    if (!packed_pair_count(atom_end - atom_begin, expected_pairs) ||
        pair_end - pair_begin != expected_pairs) {
      record_error(device_error, Gfn2D4DeviceError::kInvalidOffsets);
      continue;
    }
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
      const std::int32_t atomic_number = batch.atomic_numbers[atom];
      if (atomic_number <= 0 || atomic_number > parameters.element_count) {
        record_error(device_error, Gfn2D4DeviceError::kInvalidAtomicNumber);
        break;
      }
      const Gfn2D4DeviceElementData element = parameters.elements[atomic_number - 1];
      if (element.reference_count == 0 || element.reference_count > kGfn2D4MaximumReferences ||
          static_cast<std::int64_t>(element.reference_offset) + element.reference_count >
              parameters.reference_count ||
          !(element.covalent_radius > 0.0) || !isfinite(element.covalent_radius) ||
          !isfinite(element.electronegativity) || !(element.r4r2 > 0.0) ||
          !isfinite(element.r4r2) || !(element.effective_charge > 0.0) ||
          !isfinite(element.effective_charge) || !(element.hardness > 0.0) ||
          !isfinite(element.hardness)) {
        record_error(device_error, Gfn2D4DeviceError::kInvalidParameterData);
        break;
      }
    }
  }
  /* atomic_number_hash is the immutable batch/parameter binding authority.
   * Recompute it whenever the SCC stage has work, exactly as the standalone
   * D4 topology preflight does, so an in-range element reorder cannot silently
   * combine new element data with an old geometry cache. */
  std::uint64_t local_hash = 0u;
  for (std::int64_t atom = threadIdx.x; atom < batch.total_atoms; atom += blockDim.x) {
    local_hash ^= gfn2_d4_atomic_number_hash_contribution(batch.atomic_numbers[atom], atom);
  }
  partial_hash[threadIdx.x] = local_hash;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partial_hash[threadIdx.x] ^= partial_hash[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const std::uint64_t atomic_number_hash =
        partial_hash[0] ^ 0x243f6a8885a308d3ULL ^
        gfn2_d4_hash_mix(static_cast<std::uint64_t>(batch.total_atoms));
    if (atomic_number_hash != batch.atomic_number_hash) {
      record_error(device_error, Gfn2D4DeviceError::kInvalidAtomicNumber);
    }
  }
}

/* One block owns one active member, so inactive systems are rejected before
 * their offset, cache, or diagnostic slices are touched. */
__global__ void scc_cache_preflight_kernel(Gfn2D4DeviceBatch batch, Gfn2D4DeviceCache cache,
                                           Gfn2SccIterationDeviceActivity activity,
                                           Gfn2D4DeviceWorkspace workspace,
                                           std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!scc_system_is_active(activity, system)) {
    return;
  }
  __shared__ std::uint32_t shared_error;
  if (!block_sequence_is_valid(device_error, shared_error)) {
    return;
  }
  if (!system_is_valid(workspace, system)) {
    return;
  }
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    if (!isfinite(cache.coordination_numbers[atom])) {
      record_system_error(workspace, system, Gfn2D4DeviceError::kInvalidCoordination);
    }
  }
  const std::int64_t pair_begin = batch.pair_offsets[system];
  const std::int64_t pair_end = batch.pair_offsets[system + 1];
  for (std::int64_t pair = pair_begin + threadIdx.x; pair < pair_end; pair += blockDim.x) {
    const double* values = cache.pair_data + pair * kGfn2D4PairDataElements;
    const double distance_squared =
        values[0] * values[0] + values[1] * values[1] + values[2] * values[2];
    if (!isfinite(values[0]) || !isfinite(values[1]) || !isfinite(values[2]) ||
        !isfinite(values[3]) || !isfinite(values[4]) ||
        !(distance_squared >= kMinimumDistanceSquared) || values[3] < 0.0) {
      record_system_error(workspace, system,
                          values[3] < 0.0 ? Gfn2D4DeviceError::kInvalidDamping
                                          : Gfn2D4DeviceError::kNonfiniteArithmetic);
    }
  }
}

__global__ void cache_preflight_kernel(Gfn2D4DeviceBatch batch, Gfn2D4DeviceCache cache,
                                       Gfn2D4DeviceWorkspace workspace,
                                       std::uint32_t* device_error) {
  __shared__ std::uint32_t shared_error;
  if (!block_sequence_is_valid(device_error, shared_error)) {
    return;
  }
  for (std::int64_t atom = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       atom < batch.total_atoms; atom += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const std::int64_t system = system_for_atom(batch, atom);
    if (!system_is_valid(workspace, system)) {
      continue;
    }
    const double coordination = cache.coordination_numbers[atom];
    if (!isfinite(coordination)) {
      record_system_error(workspace, system, Gfn2D4DeviceError::kInvalidCoordination);
    }
  }
  for (std::int64_t pair = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       pair < batch.total_pairs; pair += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const std::int64_t system = system_for_pair(batch, pair);
    if (!system_is_valid(workspace, system)) {
      continue;
    }
    const double* values = cache.pair_data + pair * kGfn2D4PairDataElements;
    const double distance_squared =
        values[0] * values[0] + values[1] * values[1] + values[2] * values[2];
    if (!isfinite(values[0]) || !isfinite(values[1]) || !isfinite(values[2]) ||
        !isfinite(values[3]) || !isfinite(values[4]) ||
        !(distance_squared >= kMinimumDistanceSquared) || values[3] < 0.0) {
      record_system_error(workspace, system,
                          values[3] < 0.0 ? Gfn2D4DeviceError::kInvalidDamping
                                          : Gfn2D4DeviceError::kNonfiniteArithmetic);
    }
  }
}

__device__ void prepare_atom_weights(Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters,
                                     const double* coordination_numbers,
                                     const double* atomic_charges, bool zero_charges,
                                     bool write_cn_derivatives, bool write_charge_derivatives,
                                     std::int64_t system, std::int64_t atom,
                                     Gfn2D4DeviceWorkspace workspace, std::uint32_t* device_error) {
  if (!system_is_valid(workspace, system)) {
    return;
  }
  const double coordination = coordination_numbers[atom];
  const double charge = zero_charges ? 0.0 : atomic_charges[atom];
  if (!isfinite(coordination)) {
    record_system_error(workspace, system, Gfn2D4DeviceError::kInvalidCoordination);
    return;
  }
  if (!isfinite(charge)) {
    record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteCharge);
    return;
  }

  const std::int32_t atomic_number = batch.atomic_numbers[atom];
  if (atomic_number <= 0 || atomic_number > parameters.element_count) {
    record_error(device_error, Gfn2D4DeviceError::kInvalidAtomicNumber);
    return;
  }
  const Gfn2D4DeviceElementData element = parameters.elements[atomic_number - 1];
  if (element.reference_count == 0 || element.reference_count > kGfn2D4MaximumReferences ||
      static_cast<std::int64_t>(element.reference_offset) + element.reference_count >
          parameters.reference_count) {
    record_error(device_error, Gfn2D4DeviceError::kInvalidParameterData);
    return;
  }

  const std::int64_t output_offset = atom * kGfn2D4MaximumReferences;
  for (std::int64_t local = 0; local < kGfn2D4MaximumReferences; ++local) {
    workspace.weights[output_offset + local] = 0.0;
    if (write_cn_derivatives) {
      workspace.weight_cn_derivatives[output_offset + local] = 0.0;
    }
    if (write_charge_derivatives) {
      workspace.weight_charge_derivatives[output_offset + local] = 0.0;
    }
  }

  double normalization = 0.0;
  double normalization_derivative = 0.0;
  double maximum_reference_cn = -1.7976931348623157e308;
  for (std::int64_t local = 0; local < element.reference_count; ++local) {
    const Gfn2D4DeviceReferenceData reference =
        parameters.references[static_cast<std::int64_t>(element.reference_offset) + local];
    if (reference.gaussian_count == 0 || !isfinite(reference.coordination_number) ||
        !isfinite(reference.charge)) {
      record_error(device_error, Gfn2D4DeviceError::kInvalidParameterData);
      return;
    }
    maximum_reference_cn = fmax(maximum_reference_cn, reference.coordination_number);
    for (std::int64_t gaussian = 1; gaussian <= reference.gaussian_count; ++gaussian) {
      const double factor = static_cast<double>(gaussian) * kReferenceWeightFactor;
      const double delta = coordination - reference.coordination_number;
      const double value = exp(-factor * delta * delta);
      normalization += value;
      if (write_cn_derivatives) {
        normalization_derivative +=
            2.0 * factor * (reference.coordination_number - coordination) * value;
      }
    }
  }
  const double inverse_normalization =
      fabs(normalization) > kMinimumWeightNorm ? 1.0 / normalization : 0.0;
  const double qmod = charge + element.effective_charge;
  const double charge_steepness = element.hardness * kChargeScalingSteepness;

  for (std::int64_t local = 0; local < element.reference_count; ++local) {
    const Gfn2D4DeviceReferenceData reference =
        parameters.references[static_cast<std::int64_t>(element.reference_offset) + local];
    double numerator = 0.0;
    double numerator_derivative = 0.0;
    for (std::int64_t gaussian = 1; gaussian <= reference.gaussian_count; ++gaussian) {
      const double factor = static_cast<double>(gaussian) * kReferenceWeightFactor;
      const double delta = coordination - reference.coordination_number;
      const double value = exp(-factor * delta * delta);
      numerator += value;
      if (write_cn_derivatives) {
        numerator_derivative +=
            2.0 * factor * (reference.coordination_number - coordination) * value;
      }
    }
    double cn_weight = numerator * inverse_normalization;
    if (!isfinite(cn_weight) || inverse_normalization == 0.0) {
      cn_weight = fabs(maximum_reference_cn - reference.coordination_number) < 1.0e-12 ? 1.0 : 0.0;
    }
    double cn_derivative = 0.0;
    if (write_cn_derivatives) {
      cn_derivative =
          inverse_normalization *
          (numerator_derivative - numerator * normalization_derivative * inverse_normalization);
      if (!isfinite(cn_derivative) || inverse_normalization == 0.0) {
        cn_derivative = 0.0;
      }
    }
    const double qref = reference.charge + element.effective_charge;
    const double scaling = charge_scale(kChargeScalingHeight, charge_steepness, qref, qmod);
    const double derivative =
        write_charge_derivatives ? cn_weight * charge_scale_derivative(kChargeScalingHeight,
                                                                       charge_steepness, qref, qmod)
                                 : 0.0;
    const double weight = cn_weight * scaling;
    const double cn_scaled_derivative = cn_derivative * scaling;
    if (!isfinite(weight) || (write_charge_derivatives && !isfinite(derivative)) ||
        (write_cn_derivatives && !isfinite(cn_scaled_derivative))) {
      record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
      return;
    }
    workspace.weights[output_offset + local] = weight;
    if (write_cn_derivatives) {
      workspace.weight_cn_derivatives[output_offset + local] = cn_scaled_derivative;
    }
    if (write_charge_derivatives) {
      workspace.weight_charge_derivatives[output_offset + local] = derivative;
    }
  }
}

__global__ void prepare_weights_kernel(Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters,
                                       Gfn2D4DeviceCache cache, const double* atomic_charges,
                                       bool zero_charges, bool write_cn_derivatives,
                                       bool write_charge_derivatives,
                                       Gfn2D4DeviceWorkspace workspace,
                                       std::uint32_t* device_error) {
  __shared__ std::uint32_t shared_error;
  if (!block_sequence_is_valid(device_error, shared_error)) {
    return;
  }
  const std::int64_t atom = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (atom >= batch.total_atoms) {
    return;
  }
  prepare_atom_weights(batch, parameters, cache.coordination_numbers, atomic_charges, zero_charges,
                       write_cn_derivatives, write_charge_derivatives, system_for_atom(batch, atom),
                       atom, workspace, device_error);
}

/* A block owns one SCC system. The canonical sequence and member byte are
 * consumed before the first ragged offset, atom, cache value, or diagnostic
 * slice for that system. This is intentionally separate from the standalone
 * atom-grid kernel, whose system lookup necessarily reads the global offsets. */
__global__ void scc_prepare_weights_kernel(
    Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters, Gfn2D4DeviceCache cache,
    const double* atomic_charges, bool zero_charges, bool write_cn_derivatives,
    bool write_charge_derivatives, Gfn2SccIterationDeviceActivity activity,
    Gfn2D4DeviceWorkspace workspace, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!scc_system_is_active(activity, system)) {
    return;
  }
  __shared__ std::uint32_t shared_error;
  if (!block_sequence_is_valid(device_error, shared_error) || !system_is_valid(workspace, system)) {
    return;
  }
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    prepare_atom_weights(batch, parameters, cache.coordination_numbers, atomic_charges,
                         zero_charges, write_cn_derivatives, write_charge_derivatives, system, atom,
                         workspace, device_error);
  }
}

__global__ void pairlist_prepare_weights_kernel(
    Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters, const double* coordination_numbers,
    const std::uint8_t* role_active_mask, const double* atomic_charges, bool zero_charges,
    bool write_cn_derivatives, bool write_charge_derivatives,
    const std::uint8_t* external_active_mask, const std::uint32_t* external_sequence_active,
    double* weights, double* weight_cn_derivatives, double* weight_charge_derivatives,
    std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if ((external_sequence_active != nullptr &&
       atomicAdd(const_cast<std::uint32_t*>(external_sequence_active), 0u) != 1u) ||
      (external_active_mask != nullptr && external_active_mask[system] != 1u) ||
      !sequence_is_valid(device_error) ||
      atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) != 0u ||
      (role_active_mask != nullptr && role_active_mask[system] != 1u)) {
    return;
  }
  Gfn2D4DeviceWorkspace workspace{};
  workspace.weights = weights;
  workspace.weight_cn_derivatives = weight_cn_derivatives;
  workspace.weight_charge_derivatives = weight_charge_derivatives;
  workspace.system_errors = system_errors;
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    prepare_atom_weights(batch, parameters, coordination_numbers, atomic_charges, zero_charges,
                         write_cn_derivatives, write_charge_derivatives, system, atom, workspace,
                         device_error);
  }
}

struct PairCoefficient {
  double c6;
  double first_cn;
  double second_cn;
  double first_charge;
  double second_charge;
};

__device__ PairCoefficient pair_coefficient(Gfn2D4DeviceBatch batch,
                                            Gfn2D4DeviceParameters parameters,
                                            Gfn2D4DeviceWorkspace workspace, std::int64_t first,
                                            std::int64_t second) {
  const Gfn2D4DeviceElementData first_element =
      parameters.elements[batch.atomic_numbers[first] - 1];
  const Gfn2D4DeviceElementData second_element =
      parameters.elements[batch.atomic_numbers[second] - 1];
  const std::int64_t first_weight = first * kGfn2D4MaximumReferences;
  const std::int64_t second_weight = second * kGfn2D4MaximumReferences;
  PairCoefficient result{0.0, 0.0, 0.0, 0.0, 0.0};
  for (std::int64_t first_ref = 0; first_ref < first_element.reference_count; ++first_ref) {
    const std::int64_t global_first = first_element.reference_offset + first_ref;
    const double first_value = workspace.weights[first_weight + first_ref];
    const double first_cn_derivative = workspace.weight_cn_derivatives[first_weight + first_ref];
    const double first_derivative = workspace.weight_charge_derivatives[first_weight + first_ref];
    for (std::int64_t second_ref = 0; second_ref < second_element.reference_count; ++second_ref) {
      const std::int64_t global_second = second_element.reference_offset + second_ref;
      const double reference_c6 =
          parameters.reference_c6[global_first * parameters.reference_count + global_second];
      const double second_value = workspace.weights[second_weight + second_ref];
      const double second_cn_derivative =
          workspace.weight_cn_derivatives[second_weight + second_ref];
      const double second_derivative =
          workspace.weight_charge_derivatives[second_weight + second_ref];
      result.c6 += first_value * second_value * reference_c6;
      result.first_cn += first_cn_derivative * second_value * reference_c6;
      result.second_cn += first_value * second_cn_derivative * reference_c6;
      result.first_charge += first_derivative * second_value * reference_c6;
      result.second_charge += first_value * second_derivative * reference_c6;
    }
  }
  return result;
}

struct PairChargeDerivative {
  double first;
  double second;
};

/* SCC potential mode deliberately reads no CN-derivative workspace. */
__device__ PairChargeDerivative pair_charge_derivative(Gfn2D4DeviceBatch batch,
                                                       Gfn2D4DeviceParameters parameters,
                                                       Gfn2D4DeviceWorkspace workspace,
                                                       std::int64_t first, std::int64_t second) {
  const Gfn2D4DeviceElementData first_element =
      parameters.elements[batch.atomic_numbers[first] - 1];
  const Gfn2D4DeviceElementData second_element =
      parameters.elements[batch.atomic_numbers[second] - 1];
  const std::int64_t first_weight = first * kGfn2D4MaximumReferences;
  const std::int64_t second_weight = second * kGfn2D4MaximumReferences;
  PairChargeDerivative result{0.0, 0.0};
  for (std::int64_t first_ref = 0; first_ref < first_element.reference_count; ++first_ref) {
    const std::int64_t global_first = first_element.reference_offset + first_ref;
    const double first_value = workspace.weights[first_weight + first_ref];
    const double first_derivative = workspace.weight_charge_derivatives[first_weight + first_ref];
    for (std::int64_t second_ref = 0; second_ref < second_element.reference_count; ++second_ref) {
      const std::int64_t global_second = second_element.reference_offset + second_ref;
      const double reference_c6 =
          parameters.reference_c6[global_first * parameters.reference_count + global_second];
      const double second_value = workspace.weights[second_weight + second_ref];
      const double second_derivative =
          workspace.weight_charge_derivatives[second_weight + second_ref];
      result.first += first_derivative * second_value * reference_c6;
      result.second += first_value * second_derivative * reference_c6;
    }
  }
  return result;
}

/* SCC energy mode reads only charge-dependent weights and immutable tables. */
__device__ double pair_c6_coefficient(Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters,
                                      Gfn2D4DeviceWorkspace workspace, std::int64_t first,
                                      std::int64_t second) {
  const Gfn2D4DeviceElementData first_element =
      parameters.elements[batch.atomic_numbers[first] - 1];
  const Gfn2D4DeviceElementData second_element =
      parameters.elements[batch.atomic_numbers[second] - 1];
  const std::int64_t first_weight = first * kGfn2D4MaximumReferences;
  const std::int64_t second_weight = second * kGfn2D4MaximumReferences;
  double c6 = 0.0;
  for (std::int64_t first_ref = 0; first_ref < first_element.reference_count; ++first_ref) {
    const std::int64_t global_first = first_element.reference_offset + first_ref;
    const double first_value = workspace.weights[first_weight + first_ref];
    for (std::int64_t second_ref = 0; second_ref < second_element.reference_count; ++second_ref) {
      const std::int64_t global_second = second_element.reference_offset + second_ref;
      const double reference_c6 =
          parameters.reference_c6[global_first * parameters.reference_count + global_second];
      c6 += first_value * workspace.weights[second_weight + second_ref] * reference_c6;
    }
  }
  return c6;
}

__global__ void scc_potential_kernel(Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters,
                                     Gfn2D4DeviceCache cache,
                                     Gfn2SccIterationDeviceActivity activity,
                                     Gfn2D4DeviceWorkspace workspace, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!scc_system_is_active(activity, system)) {
    return;
  }
  __shared__ std::uint32_t shared_error;
  if (!block_system_is_valid(workspace, system, device_error, shared_error)) {
    return;
  }
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
    double potential = 0.0;
    for (std::int64_t other = begin; other < end; ++other) {
      if (other == atom) {
        continue;
      }
      const std::int64_t first = atom < other ? atom : other;
      const std::int64_t second = atom < other ? other : atom;
      const std::int64_t local_first = first - begin;
      const std::int64_t local_second = second - begin;
      const std::int64_t packed_pair =
          batch.pair_offsets[system] + local_second * (local_second - 1) / 2 + local_first;
      const double damping = cache.pair_data[packed_pair * kGfn2D4PairDataElements + 3];
      if (!isfinite(damping) || damping < 0.0) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kInvalidDamping);
        return;
      }
      if (damping == 0.0) {
        continue;
      }
      const PairChargeDerivative derivative =
          pair_charge_derivative(batch, parameters, workspace, first, second);
      potential -= (atom == first ? derivative.first : derivative.second) * damping;
      if (!isfinite(potential)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
        return;
      }
    }
    workspace.atom_scratch[atom] = potential;
  }
}

__global__ void scc_energy_kernel(Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters,
                                  Gfn2D4DeviceCache cache, Gfn2SccIterationDeviceActivity activity,
                                  Gfn2D4DeviceWorkspace workspace, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!scc_system_is_active(activity, system)) {
    return;
  }
  __shared__ double partial[kThreadsPerBlock];
  __shared__ std::uint32_t shared_error;
  if (!block_system_is_valid(workspace, system, device_error, shared_error)) {
    return;
  }
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  const std::int64_t atom_count = end - begin;
  const std::int64_t pair_begin = batch.pair_offsets[system];
  const std::int64_t pair_end = batch.pair_offsets[system + 1];
  double local_energy = 0.0;
  for (std::int64_t local_pair = threadIdx.x; local_pair < pair_end - pair_begin;
       local_pair += blockDim.x) {
    std::int64_t local_second =
        static_cast<std::int64_t>((1.0 + sqrt(1.0 + 8.0 * static_cast<double>(local_pair))) * 0.5);
    while (local_second * (local_second - 1) / 2 > local_pair) {
      --local_second;
    }
    while (local_second + 1 < atom_count && (local_second + 1) * local_second / 2 <= local_pair) {
      ++local_second;
    }
    const std::int64_t local_first = local_pair - local_second * (local_second - 1) / 2;
    const std::int64_t first = begin + local_first;
    const std::int64_t second = begin + local_second;
    const double damping = cache.pair_data[(pair_begin + local_pair) * kGfn2D4PairDataElements + 3];
    if (!isfinite(damping) || damping < 0.0) {
      record_system_error(workspace, system, Gfn2D4DeviceError::kInvalidDamping);
      local_energy = 0.0;
      break;
    }
    if (damping != 0.0) {
      local_energy -= pair_c6_coefficient(batch, parameters, workspace, first, second) * damping;
      if (!isfinite(local_energy)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
        local_energy = 0.0;
        break;
      }
    }
  }
  partial[threadIdx.x] = local_energy;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && sequence_is_valid(device_error) && system_is_valid(workspace, system)) {
    if (isfinite(partial[0])) {
      workspace.batch_scratch[system] = partial[0];
    } else {
      record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
    }
  }
}

__global__ void scc_publish_potential_kernel(Gfn2D4DeviceBatch batch,
                                             Gfn2SccIterationDeviceActivity activity,
                                             const double* scratch, double* output,
                                             Gfn2D4DeviceWorkspace workspace,
                                             const std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!scc_system_is_active(activity, system) || !sequence_is_valid(device_error) ||
      !system_is_valid(workspace, system)) {
    return;
  }
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
    output[atom] = scratch[atom];
  }
}

__global__ void scc_publish_energy_kernel(std::int64_t batch_size,
                                          Gfn2SccIterationDeviceActivity activity,
                                          const double* scratch, double* output,
                                          Gfn2D4DeviceWorkspace workspace,
                                          const std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system < batch_size && scc_system_is_active(activity, system) &&
      sequence_is_valid(device_error) && system_is_valid(workspace, system)) {
    output[system] = scratch[system];
  }
}

__global__ void potential_kernel(Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters,
                                 Gfn2D4DeviceCache cache, Gfn2D4DeviceWorkspace workspace,
                                 std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ std::uint32_t shared_error;
  if (!block_system_is_valid(workspace, system, device_error, shared_error)) {
    return;
  }
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
    double potential = 0.0;
    for (std::int64_t other = begin; other < end; ++other) {
      if (other == atom) {
        continue;
      }
      const std::int64_t first = atom < other ? atom : other;
      const std::int64_t second = atom < other ? other : atom;
      const std::int64_t local_first = first - begin;
      const std::int64_t local_second = second - begin;
      const std::int64_t packed_pair =
          batch.pair_offsets[system] + local_second * (local_second - 1) / 2 + local_first;
      const double damping = cache.pair_data[packed_pair * kGfn2D4PairDataElements + 3];
      if (!isfinite(damping) || damping < 0.0) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kInvalidDamping);
        return;
      }
      if (damping == 0.0) {
        continue;
      }
      const PairCoefficient coefficient =
          pair_coefficient(batch, parameters, workspace, first, second);
      const double derivative =
          atom == first ? coefficient.first_charge : coefficient.second_charge;
      potential -= derivative * damping;
      if (!isfinite(potential)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
        return;
      }
    }
    workspace.atom_scratch[atom] = potential;
  }
}

__device__ void commit_energy_reduction(double reduced_energy, Gfn2D4DeviceWorkspace workspace,
                                        std::int64_t system, std::uint32_t* device_error) {
  if (!sequence_is_valid(device_error) || !system_is_valid(workspace, system)) {
    return;
  }
  if (isfinite(reduced_energy)) {
    workspace.batch_scratch[system] = reduced_energy;
  } else {
    record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
  }
}

__global__ void energy_kernel(Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters,
                              Gfn2D4DeviceCache cache, Gfn2D4DeviceWorkspace workspace,
                              std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ double partial[kThreadsPerBlock];
  __shared__ std::uint32_t shared_error;
  if (!block_system_is_valid(workspace, system, device_error, shared_error)) {
    return;
  }
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  const std::int64_t atom_count = end - begin;
  const std::int64_t pair_begin = batch.pair_offsets[system];
  const std::int64_t pair_end = batch.pair_offsets[system + 1];
  double local_energy = 0.0;
  for (std::int64_t local_pair = threadIdx.x; local_pair < pair_end - pair_begin;
       local_pair += blockDim.x) {
    std::int64_t local_second =
        static_cast<std::int64_t>((1.0 + sqrt(1.0 + 8.0 * static_cast<double>(local_pair))) * 0.5);
    while (local_second * (local_second - 1) / 2 > local_pair) {
      --local_second;
    }
    while (local_second + 1 < atom_count && (local_second + 1) * local_second / 2 <= local_pair) {
      ++local_second;
    }
    const std::int64_t local_first = local_pair - local_second * (local_second - 1) / 2;
    const std::int64_t first = begin + local_first;
    const std::int64_t second = begin + local_second;
    const double damping = cache.pair_data[(pair_begin + local_pair) * kGfn2D4PairDataElements + 3];
    if (!isfinite(damping) || damping < 0.0) {
      record_system_error(workspace, system, Gfn2D4DeviceError::kInvalidDamping);
      local_energy = 0.0;
      break;
    }
    if (damping != 0.0) {
      const PairCoefficient coefficient =
          pair_coefficient(batch, parameters, workspace, first, second);
      local_energy -= coefficient.c6 * damping;
      if (!isfinite(local_energy)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
        local_energy = 0.0;
        break;
      }
    }
  }
  partial[threadIdx.x] = local_energy;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    commit_energy_reduction(partial[0], workspace, system, device_error);
  }
}

__device__ std::int64_t pair_index(Gfn2D4DeviceBatch batch, std::int64_t system, std::int64_t first,
                                   std::int64_t second) {
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t local_first = first - begin;
  const std::int64_t local_second = second - begin;
  return batch.pair_offsets[system] + local_second * (local_second - 1) / 2 + local_first;
}

__device__ double pair_damping_radius(Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters,
                                      std::int64_t first, std::int64_t second) {
  const Gfn2D4DeviceElementData first_element =
      parameters.elements[batch.atomic_numbers[first] - 1];
  const Gfn2D4DeviceElementData second_element =
      parameters.elements[batch.atomic_numbers[second] - 1];
  const double rrij = 3.0 * first_element.r4r2 * second_element.r4r2;
  return kDispersionA1 * sqrt(rrij) + kDispersionA2;
}

__global__ void pairlist_two_body_potential_kernel(
    Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters, Gfn2D4PairListDeviceCache cache,
    const std::uint8_t* external_active_mask, const std::uint32_t* external_sequence_active,
    Gfn2D4DeviceWorkspace workspace, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if ((external_sequence_active != nullptr &&
       atomicAdd(const_cast<std::uint32_t*>(external_sequence_active), 0u) != 1u) ||
      (external_active_mask != nullptr && external_active_mask[system] != 1u) ||
      !sequence_is_valid(device_error) || !system_is_valid(workspace, system) ||
      !d4_pairlist_role_active(cache.two_body_pairs, system)) {
    return;
  }
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    double potential = 0.0;
    const std::int64_t begin = cache.two_body_pairs.neighbor_offsets[atom];
    const std::int64_t end = begin + cache.two_body_pairs.neighbor_counts[atom];
    for (std::int64_t index = begin; index < end; ++index) {
      const std::int64_t other = cache.two_body_pairs.neighbors[index];
      const std::int64_t first = atom < other ? atom : other;
      const std::int64_t second = atom < other ? other : atom;
      D4PairGeometry pair{};
      if (!evaluate_d4_pair_geometry(batch, parameters, cache.positions, first, second, &pair)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
        return;
      }
      if (pair.distance_squared > kTwoBodyCutoffSquared) {
        continue;
      }
      const PairCoefficient coefficient =
          pair_coefficient(batch, parameters, workspace, first, second);
      const double derivative =
          atom == first ? coefficient.first_charge : coefficient.second_charge;
      potential -= derivative * pair.damping;
      if (!isfinite(potential)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
        return;
      }
    }
    workspace.atom_scratch[atom] = potential;
  }
}

__global__ void pairlist_two_body_energy_kernel(
    Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters, Gfn2D4PairListDeviceCache cache,
    const std::uint8_t* external_active_mask, const std::uint32_t* external_sequence_active,
    Gfn2D4DeviceWorkspace workspace, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ double partial[kThreadsPerBlock];
  if ((external_sequence_active != nullptr &&
       atomicAdd(const_cast<std::uint32_t*>(external_sequence_active), 0u) != 1u) ||
      (external_active_mask != nullptr && external_active_mask[system] != 1u) ||
      !sequence_is_valid(device_error) || !system_is_valid(workspace, system) ||
      !d4_pairlist_role_active(cache.two_body_pairs, system)) {
    return;
  }
  const std::int64_t pair_begin = cache.two_body_pairs.pair_offsets[system];
  const std::int64_t pair_count = cache.two_body_pairs.pair_counts[system];
  double local_energy = 0.0;
  for (std::int64_t local = threadIdx.x; local < pair_count; local += blockDim.x) {
    const Gfn2AtomPair pair_indices = cache.two_body_pairs.pairs[pair_begin + local];
    D4PairGeometry pair{};
    if (!evaluate_d4_pair_geometry(batch, parameters, cache.positions, pair_indices.first,
                                   pair_indices.second, &pair)) {
      record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
      local_energy = 0.0;
      break;
    }
    if (pair.distance_squared <= kTwoBodyCutoffSquared) {
      local_energy -= pair_c6_coefficient(batch, parameters, workspace, pair_indices.first,
                                          pair_indices.second) *
                      pair.damping;
      if (!isfinite(local_energy)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
        local_energy = 0.0;
        break;
      }
    }
  }
  partial[threadIdx.x] = local_energy;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    commit_energy_reduction(partial[0], workspace, system, device_error);
  }
}

__global__ void pairlist_two_body_cn_adjoint_kernel(Gfn2D4DeviceBatch batch,
                                                    Gfn2D4DeviceParameters parameters,
                                                    Gfn2D4PairListDeviceCache cache,
                                                    Gfn2D4DeviceWorkspace workspace,
                                                    std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!sequence_is_valid(device_error) || !system_is_valid(workspace, system) ||
      !d4_pairlist_role_active(cache.two_body_pairs, system)) {
    return;
  }
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    double adjoint = 0.0;
    const std::int64_t begin = cache.two_body_pairs.neighbor_offsets[atom];
    const std::int64_t end = begin + cache.two_body_pairs.neighbor_counts[atom];
    for (std::int64_t index = begin; index < end; ++index) {
      const std::int64_t other = cache.two_body_pairs.neighbors[index];
      const std::int64_t first = atom < other ? atom : other;
      const std::int64_t second = atom < other ? other : atom;
      D4PairGeometry pair{};
      if (!evaluate_d4_pair_geometry(batch, parameters, cache.positions, first, second, &pair)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
        return;
      }
      if (pair.distance_squared <= kTwoBodyCutoffSquared) {
        const PairCoefficient coefficient =
            pair_coefficient(batch, parameters, workspace, first, second);
        adjoint -= (atom == first ? coefficient.first_cn : coefficient.second_cn) * pair.damping;
      }
    }
    if (!isfinite(adjoint)) {
      record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
      return;
    }
    workspace.coordination_adjoints[atom] = adjoint;
  }
}

__global__ void pairlist_two_body_gradient_kernel(Gfn2D4DeviceBatch batch,
                                                  Gfn2D4DeviceParameters parameters,
                                                  Gfn2D4PairListDeviceCache cache,
                                                  Gfn2D4DeviceWorkspace workspace,
                                                  std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!sequence_is_valid(device_error) || !system_is_valid(workspace, system) ||
      !d4_pairlist_role_active(cache.two_body_pairs, system)) {
    return;
  }
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    double gradient[3]{0.0, 0.0, 0.0};
    const std::int64_t begin = cache.two_body_pairs.neighbor_offsets[atom];
    const std::int64_t end = begin + cache.two_body_pairs.neighbor_counts[atom];
    for (std::int64_t index = begin; index < end; ++index) {
      const std::int64_t other = cache.two_body_pairs.neighbors[index];
      const std::int64_t first = atom < other ? atom : other;
      const std::int64_t second = atom < other ? other : atom;
      const double sign = atom == first ? 1.0 : -1.0;
      D4PairGeometry pair{};
      if (!evaluate_d4_pair_geometry(batch, parameters, cache.positions, first, second, &pair)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
        return;
      }
      if (pair.distance_squared <= kTwoBodyCutoffSquared) {
        const PairCoefficient coefficient =
            pair_coefficient(batch, parameters, workspace, first, second);
        const double radial_scale = sign * -coefficient.c6 * pair.damping_derivative;
        for (int axis = 0; axis < 3; ++axis) {
          gradient[axis] += radial_scale * pair.vector[axis];
        }
      }
      if (pair.distance_squared <= kCoordinationCutoffSquared) {
        const Gfn2D4DeviceElementData first_element =
            parameters.elements[batch.atomic_numbers[first] - 1];
        const Gfn2D4DeviceElementData second_element =
            parameters.elements[batch.atomic_numbers[second] - 1];
        const double radius = first_element.covalent_radius + second_element.covalent_radius;
        const double en_delta =
            fabs(first_element.electronegativity - second_element.electronegativity);
        const double en_factor = kEnK4 * exp(-((en_delta + kEnK5) * (en_delta + kEnK5)) / kEnK6);
        const double distance = sqrt(pair.distance_squared);
        const double exponent = kCoordinationSteepness * (distance - radius) / radius;
        const double derivative = -en_factor * kCoordinationSteepness * exp(-exponent * exponent) *
                                  kInverseSqrtPi / radius;
        const double scale =
            sign * derivative *
            (workspace.coordination_adjoints[first] + workspace.coordination_adjoints[second]) /
            distance;
        for (int axis = 0; axis < 3; ++axis) {
          gradient[axis] += scale * pair.vector[axis];
        }
      }
    }
    if (!isfinite(gradient[0]) || !isfinite(gradient[1]) || !isfinite(gradient[2])) {
      record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
      return;
    }
    for (int axis = 0; axis < 3; ++axis) {
      workspace.gradient_scratch[atom * 3 + axis] = gradient[axis];
    }
  }
}

__global__ void two_body_cn_adjoint_kernel(Gfn2D4DeviceBatch batch,
                                           Gfn2D4DeviceParameters parameters,
                                           Gfn2D4DeviceCache cache, Gfn2D4DeviceWorkspace workspace,
                                           std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ std::uint32_t shared_error;
  if (!block_system_is_valid(workspace, system, device_error, shared_error)) {
    return;
  }
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
    double adjoint = 0.0;
    for (std::int64_t other = begin; other < end; ++other) {
      if (atom == other) {
        continue;
      }
      const std::int64_t first = atom < other ? atom : other;
      const std::int64_t second = atom < other ? other : atom;
      const double damping =
          cache.pair_data[pair_index(batch, system, first, second) * kGfn2D4PairDataElements + 3];
      if (damping == 0.0) {
        continue;
      }
      const PairCoefficient coefficient =
          pair_coefficient(batch, parameters, workspace, first, second);
      adjoint -= (atom == first ? coefficient.first_cn : coefficient.second_cn) * damping;
    }
    if (!isfinite(adjoint)) {
      record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
      return;
    }
    workspace.coordination_adjoints[atom] = adjoint;
  }
}

__global__ void two_body_gradient_kernel(Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters,
                                         Gfn2D4DeviceCache cache, Gfn2D4DeviceWorkspace workspace,
                                         std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ std::uint32_t shared_error;
  if (!block_system_is_valid(workspace, system, device_error, shared_error)) {
    return;
  }
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
    double gradient[3]{0.0, 0.0, 0.0};
    for (std::int64_t other = begin; other < end; ++other) {
      if (atom == other) {
        continue;
      }
      const std::int64_t first = atom < other ? atom : other;
      const std::int64_t second = atom < other ? other : atom;
      const double sign = atom == first ? 1.0 : -1.0;
      const double* pair =
          cache.pair_data + pair_index(batch, system, first, second) * kGfn2D4PairDataElements;
      if (pair[3] != 0.0) {
        const PairCoefficient coefficient =
            pair_coefficient(batch, parameters, workspace, first, second);
        const double radial_scale = sign * -coefficient.c6 * pair[4];
        for (int axis = 0; axis < 3; ++axis) {
          gradient[axis] += radial_scale * pair[axis];
        }
      }

      const double distance_squared = pair[0] * pair[0] + pair[1] * pair[1] + pair[2] * pair[2];
      if (distance_squared <= kCoordinationCutoffSquared) {
        const Gfn2D4DeviceElementData first_element =
            parameters.elements[batch.atomic_numbers[first] - 1];
        const Gfn2D4DeviceElementData second_element =
            parameters.elements[batch.atomic_numbers[second] - 1];
        const double radius = first_element.covalent_radius + second_element.covalent_radius;
        const double en_delta =
            fabs(first_element.electronegativity - second_element.electronegativity);
        const double en_factor = kEnK4 * exp(-((en_delta + kEnK5) * (en_delta + kEnK5)) / kEnK6);
        const double distance = sqrt(distance_squared);
        const double exponent = kCoordinationSteepness * (distance - radius) / radius;
        const double derivative = -en_factor * kCoordinationSteepness * exp(-exponent * exponent) *
                                  kInverseSqrtPi / radius;
        const double scale =
            sign * derivative *
            (workspace.coordination_adjoints[first] + workspace.coordination_adjoints[second]) /
            distance;
        for (int axis = 0; axis < 3; ++axis) {
          gradient[axis] += scale * pair[axis];
        }
      }
    }
    if (!isfinite(gradient[0]) || !isfinite(gradient[1]) || !isfinite(gradient[2])) {
      record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
      return;
    }
    for (int axis = 0; axis < 3; ++axis) {
      workspace.gradient_scratch[atom * 3 + axis] = gradient[axis];
    }
  }
}

__global__ void atm_energy_kernel(Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters,
                                  Gfn2D4DeviceCache cache, Gfn2D4DeviceWorkspace workspace,
                                  std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ double partial[kThreadsPerBlock];
  __shared__ std::uint32_t shared_error;
  if (!block_system_is_valid(workspace, system, device_error, shared_error)) {
    return;
  }
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  constexpr double exponent_third = kAtmExponent / 3.0;
  double energy = 0.0;
  std::int64_t outer_pair = 0;
  for (std::int64_t i = begin + 2; i < end; ++i) {
    for (std::int64_t j = begin + 1; j < i; ++j) {
      if (outer_pair++ % blockDim.x != threadIdx.x) {
        continue;
      }
      const double* vij =
          cache.pair_data + pair_index(batch, system, j, i) * kGfn2D4PairDataElements;
      const double r2ij = vij[0] * vij[0] + vij[1] * vij[1] + vij[2] * vij[2];
      if (r2ij > kAtmCutoffSquared) {
        continue;
      }
      const PairCoefficient c6ij = pair_coefficient(batch, parameters, workspace, i, j);
      for (std::int64_t k = begin; k < j; ++k) {
        const double* vik =
            cache.pair_data + pair_index(batch, system, k, i) * kGfn2D4PairDataElements;
        const double* vjk =
            cache.pair_data + pair_index(batch, system, k, j) * kGfn2D4PairDataElements;
        const double r2ik = vik[0] * vik[0] + vik[1] * vik[1] + vik[2] * vik[2];
        const double r2jk = vjk[0] * vjk[0] + vjk[1] * vjk[1] + vjk[2] * vjk[2];
        if (r2ik > kAtmCutoffSquared || r2jk > kAtmCutoffSquared) {
          continue;
        }
        const PairCoefficient c6ik = pair_coefficient(batch, parameters, workspace, i, k);
        const PairCoefficient c6jk = pair_coefficient(batch, parameters, workspace, j, k);
        const double r0ij = pair_damping_radius(batch, parameters, j, i);
        const double r0ik = pair_damping_radius(batch, parameters, k, i);
        const double r0jk = pair_damping_radius(batch, parameters, k, j);
        const double r2_product = r2ij * r2ik * r2jk;
        const double r1_product = sqrt(r2_product);
        const double r3_product = r2_product * r1_product;
        const double r5_product = r3_product * r2_product;
        const double damping =
            1.0 / (1.0 + 6.0 * pow((r0ij * r0ik * r0jk) / r1_product, exponent_third));
        const double angle = 0.375 * (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) *
                                 (-r2ij + r2jk + r2ik) / r5_product +
                             1.0 / r3_product;
        const double c9 = -kDispersionS9 * sqrt(fabs(c6ij.c6 * c6ik.c6 * c6jk.c6));
        energy -= angle * damping * c9;
      }
    }
  }
  if (!isfinite(energy)) {
    record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
    energy = 0.0;
  }
  partial[threadIdx.x] = energy;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    commit_energy_reduction(partial[0], workspace, system, device_error);
  }
}

#if defined(XTBLOOM_CUDA_TEST_HOOKS)
/* White-box regression kernel for the reduction/transaction boundary shared by energy paths. */
__global__ void atm_reduction_test_kernel(Gfn2D4DeviceBatch batch, const double* values,
                                          Gfn2D4DeviceWorkspace workspace,
                                          std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ double partial[kThreadsPerBlock];
  __shared__ std::uint32_t shared_error;
  if (!block_system_is_valid(workspace, system, device_error, shared_error)) {
    return;
  }
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  double local = 0.0;
  for (std::int64_t index = begin + threadIdx.x; index < end; index += blockDim.x) {
    local += values[index];
    if (!isfinite(local)) {
      record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
      local = 0.0;
      break;
    }
  }
  partial[threadIdx.x] = local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    commit_energy_reduction(partial[0], workspace, system, device_error);
  }
}
#endif

__device__ void atm_distance_gradient(double target, double other_first, double other_second,
                                      const double* vector, double r5_product, double c9,
                                      double angle, double damping, double damping_derivative,
                                      double* output) {
  const double angle_derivative =
      -0.375 *
      (target * target * target + target * target * (other_first + other_second) +
       target * (3.0 * other_first * other_first + 2.0 * other_first * other_second +
                 3.0 * other_second * other_second) -
       5.0 * (other_first - other_second) * (other_first - other_second) *
           (other_first + other_second)) /
      r5_product;
  const double scale = c9 * (-angle_derivative * damping + angle * damping_derivative) / target;
  for (int axis = 0; axis < 3; ++axis) {
    output[axis] = scale * vector[axis];
  }
}

__global__ void atm_gradient_kernel(Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters,
                                    Gfn2D4DeviceCache cache, Gfn2D4DeviceWorkspace workspace,
                                    std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ std::uint32_t shared_error;
  if (!block_system_is_valid(workspace, system, device_error, shared_error)) {
    return;
  }
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  constexpr double exponent_third = kAtmExponent / 3.0;
  std::int64_t outer_pair = 0;
  for (std::int64_t i = begin + 2; i < end; ++i) {
    for (std::int64_t j = begin + 1; j < i; ++j) {
      if (outer_pair++ % blockDim.x != threadIdx.x) {
        continue;
      }
      const double* vij =
          cache.pair_data + pair_index(batch, system, j, i) * kGfn2D4PairDataElements;
      const double r2ij = vij[0] * vij[0] + vij[1] * vij[1] + vij[2] * vij[2];
      if (r2ij > kAtmCutoffSquared) {
        continue;
      }
      const PairCoefficient c6ij = pair_coefficient(batch, parameters, workspace, i, j);
      for (std::int64_t k = begin; k < j; ++k) {
        const double* vik =
            cache.pair_data + pair_index(batch, system, k, i) * kGfn2D4PairDataElements;
        const double* vjk =
            cache.pair_data + pair_index(batch, system, k, j) * kGfn2D4PairDataElements;
        const double r2ik = vik[0] * vik[0] + vik[1] * vik[1] + vik[2] * vik[2];
        const double r2jk = vjk[0] * vjk[0] + vjk[1] * vjk[1] + vjk[2] * vjk[2];
        if (r2ik > kAtmCutoffSquared || r2jk > kAtmCutoffSquared) {
          continue;
        }
        const PairCoefficient c6ik = pair_coefficient(batch, parameters, workspace, i, k);
        const PairCoefficient c6jk = pair_coefficient(batch, parameters, workspace, j, k);
        if (!(c6ij.c6 > 0.0) || !(c6ik.c6 > 0.0) || !(c6jk.c6 > 0.0)) {
          record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
          return;
        }
        const double r0ij = pair_damping_radius(batch, parameters, j, i);
        const double r0ik = pair_damping_radius(batch, parameters, k, i);
        const double r0jk = pair_damping_radius(batch, parameters, k, j);
        const double r2_product = r2ij * r2ik * r2jk;
        const double r1_product = sqrt(r2_product);
        const double r3_product = r2_product * r1_product;
        const double r5_product = r3_product * r2_product;
        const double ratio = (r0ij * r0ik * r0jk) / r1_product;
        const double ratio_power = pow(ratio, exponent_third);
        const double damping = 1.0 / (1.0 + 6.0 * ratio_power);
        const double angle = 0.375 * (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) *
                                 (-r2ij + r2jk + r2ik) / r5_product +
                             1.0 / r3_product;
        const double c9 = -kDispersionS9 * sqrt(c6ij.c6 * c6ik.c6 * c6jk.c6);
        const double rr = angle * damping;
        const double damping_derivative = -2.0 * kAtmExponent * ratio_power * damping * damping;
        double dgij[3];
        double dgik[3];
        double dgjk[3];
        atm_distance_gradient(r2ij, r2jk, r2ik, vij, r5_product, c9, angle, damping,
                              damping_derivative, dgij);
        atm_distance_gradient(r2ik, r2jk, r2ij, vik, r5_product, c9, angle, damping,
                              damping_derivative, dgik);
        atm_distance_gradient(r2jk, r2ik, r2ij, vjk, r5_product, c9, angle, damping,
                              damping_derivative, dgjk);
        const double i_adjoint =
            -0.5 * rr * c9 * (c6ij.first_cn / c6ij.c6 + c6ik.first_cn / c6ik.c6);
        const double j_adjoint =
            -0.5 * rr * c9 * (c6ij.second_cn / c6ij.c6 + c6jk.first_cn / c6jk.c6);
        const double k_adjoint =
            -0.5 * rr * c9 * (c6ik.second_cn / c6ik.c6 + c6jk.second_cn / c6jk.c6);
        if (!isfinite(dgij[0]) || !isfinite(dgij[1]) || !isfinite(dgij[2]) || !isfinite(dgik[0]) ||
            !isfinite(dgik[1]) || !isfinite(dgik[2]) || !isfinite(dgjk[0]) || !isfinite(dgjk[1]) ||
            !isfinite(dgjk[2]) || !isfinite(i_adjoint) || !isfinite(j_adjoint) ||
            !isfinite(k_adjoint)) {
          record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
          return;
        }
        for (int axis = 0; axis < 3; ++axis) {
          atomic_add_fp64(workspace.gradient_scratch + i * 3 + axis, -dgij[axis] - dgik[axis]);
          atomic_add_fp64(workspace.gradient_scratch + j * 3 + axis, dgij[axis] - dgjk[axis]);
          atomic_add_fp64(workspace.gradient_scratch + k * 3 + axis, dgik[axis] + dgjk[axis]);
        }
        atomic_add_fp64(workspace.coordination_adjoints + i, i_adjoint);
        atomic_add_fp64(workspace.coordination_adjoints + j, j_adjoint);
        atomic_add_fp64(workspace.coordination_adjoints + k, k_adjoint);
      }
    }
  }
}

/*
 * Decode one flat ATM outer-pair index into its atom pair (j, i) with
 * begin+1 <= j and begin+2 <= i. The pair space is exactly the triangular
 * enumeration of energy_kernel over the atoms {begin+1, ..., end-1}, so the
 * decode mirrors that kernel's proven inverse (a sqrt estimate plus two exact
 * boundary corrections). Returns j (the smaller) and i (the larger).
 */
__device__ void atm_pair_from_flat(std::int64_t pair, std::int64_t begin, std::int64_t* j,
                                   std::int64_t* i) {
  std::int64_t second =
      static_cast<std::int64_t>((1.0 + sqrt(1.0 + 8.0 * static_cast<double>(pair))) * 0.5);
  while (second * (second - 1) / 2 > pair) {
    --second;
  }
  while ((second + 1) * second / 2 <= pair) {
    ++second;
  }
  const std::int64_t first = pair - second * (second - 1) / 2;
  *j = begin + 1 + first;
  *i = begin + 1 + second;
}

/*
 * Multi-block D4 ATM energy variant. gridDim.x = blocks per system,
 * gridDim.y = system. Each block owns a contiguous slice of that system's
 * flat outer-pair space and emits one deterministic partial into
 * workspace.atom_scratch[system * gridDim.x + blockIdx.x]; a separate
 * reduction kernel sums the partials in fixed block order. Only used when the
 * host-side split gate decides a single block per system would under-use the
 * device; otherwise the single-block kernel above preserves the exact
 * historical behavior.
 */
__global__ void atm_energy_split_kernel(Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters,
                                        Gfn2D4DeviceCache cache, Gfn2D4DeviceWorkspace workspace,
                                        std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.y);
  const std::int64_t slice = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t slices = static_cast<std::int64_t>(gridDim.x);
  __shared__ double partial[kThreadsPerBlock];
  __shared__ std::uint32_t shared_error;
  if (!block_system_is_valid(workspace, system, device_error, shared_error)) {
    return;
  }
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  const std::int64_t atom_count = end - begin;
  /* Flat outer-pair count over the atoms {begin+1, ..., end-1}. */
  const std::int64_t pair_count = atom_count >= 3 ? (atom_count - 1) * (atom_count - 2) / 2 : 0;
  const std::int64_t slice_start = pair_count * slice / slices;
  const std::int64_t slice_end = pair_count * (slice + 1) / slices;
  constexpr double exponent_third = kAtmExponent / 3.0;
  double energy = 0.0;
  for (std::int64_t pair = slice_start + threadIdx.x; pair < slice_end; pair += blockDim.x) {
    std::int64_t j = 0;
    std::int64_t i = 0;
    atm_pair_from_flat(pair, begin, &j, &i);
    const double* vij = cache.pair_data + pair_index(batch, system, j, i) * kGfn2D4PairDataElements;
    const double r2ij = vij[0] * vij[0] + vij[1] * vij[1] + vij[2] * vij[2];
    if (r2ij > kAtmCutoffSquared) {
      continue;
    }
    const PairCoefficient c6ij = pair_coefficient(batch, parameters, workspace, i, j);
    for (std::int64_t k = begin; k < j; ++k) {
      const double* vik =
          cache.pair_data + pair_index(batch, system, k, i) * kGfn2D4PairDataElements;
      const double* vjk =
          cache.pair_data + pair_index(batch, system, k, j) * kGfn2D4PairDataElements;
      const double r2ik = vik[0] * vik[0] + vik[1] * vik[1] + vik[2] * vik[2];
      const double r2jk = vjk[0] * vjk[0] + vjk[1] * vjk[1] + vjk[2] * vjk[2];
      if (r2ik > kAtmCutoffSquared || r2jk > kAtmCutoffSquared) {
        continue;
      }
      const PairCoefficient c6ik = pair_coefficient(batch, parameters, workspace, i, k);
      const PairCoefficient c6jk = pair_coefficient(batch, parameters, workspace, j, k);
      const double r0ij = pair_damping_radius(batch, parameters, j, i);
      const double r0ik = pair_damping_radius(batch, parameters, k, i);
      const double r0jk = pair_damping_radius(batch, parameters, k, j);
      const double r2_product = r2ij * r2ik * r2jk;
      const double r1_product = sqrt(r2_product);
      const double r3_product = r2_product * r1_product;
      const double r5_product = r3_product * r2_product;
      const double damping =
          1.0 / (1.0 + 6.0 * pow((r0ij * r0ik * r0jk) / r1_product, exponent_third));
      const double angle =
          0.375 * (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) * (-r2ij + r2jk + r2ik) / r5_product +
          1.0 / r3_product;
      const double c9 = -kDispersionS9 * sqrt(fabs(c6ij.c6 * c6ik.c6 * c6jk.c6));
      energy -= angle * damping * c9;
    }
  }
  if (!isfinite(energy)) {
    record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
    energy = 0.0;
  }
  partial[threadIdx.x] = energy;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    workspace.atom_scratch[system * slices + slice] = partial[0];
  }
}

/*
 * Deterministic cross-block finish for atm_energy_split_kernel: one block per
 * system sums the per-slice partials in ascending slice order and commits via
 * the shared reduction helper, preserving the single-block failure semantics.
 */
__global__ void atm_energy_split_reduce_kernel(Gfn2D4DeviceBatch batch,
                                               Gfn2D4DeviceWorkspace workspace,
                                               std::int32_t blocks_per_system,
                                               const std::uint8_t* active_mask,
                                               std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ double partial[kThreadsPerBlock];
  __shared__ std::uint32_t shared_error;
  if ((active_mask != nullptr && active_mask[system] != 1u) ||
      !block_system_is_valid(workspace, system, device_error, shared_error)) {
    return;
  }
  double energy = 0.0;
  const double* const partials =
      workspace.atom_scratch + system * static_cast<std::int64_t>(blocks_per_system);
  for (std::int64_t index = threadIdx.x; index < blocks_per_system; index += blockDim.x) {
    energy += partials[index];
  }
  if (!isfinite(energy)) {
    record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
    energy = 0.0;
  }
  partial[threadIdx.x] = energy;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    commit_energy_reduction(partial[0], workspace, system, device_error);
  }
}

/*
 * Multi-block D4 ATM gradient variant. gridDim.x = blocks per system,
 * gridDim.y = system, so many blocks contribute atomically to the shared
 * gradient and coordination-adjoint accumulators for one large system. The
 * single-slice arithmetic is identical to atm_gradient_kernel; cross-block
 * atomic accumulation matches the kernel's existing per-thread atomics.
 */
__global__ void atm_gradient_split_kernel(Gfn2D4DeviceBatch batch,
                                          Gfn2D4DeviceParameters parameters,
                                          Gfn2D4DeviceCache cache, Gfn2D4DeviceWorkspace workspace,
                                          std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.y);
  const std::int64_t slice = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t slices = static_cast<std::int64_t>(gridDim.x);
  __shared__ std::uint32_t shared_error;
  if (!block_system_is_valid(workspace, system, device_error, shared_error)) {
    return;
  }
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  const std::int64_t atom_count = end - begin;
  const std::int64_t pair_count = atom_count >= 3 ? (atom_count - 1) * (atom_count - 2) / 2 : 0;
  const std::int64_t slice_start = pair_count * slice / slices;
  const std::int64_t slice_end = pair_count * (slice + 1) / slices;
  constexpr double exponent_third = kAtmExponent / 3.0;
  for (std::int64_t pair = slice_start + threadIdx.x; pair < slice_end; pair += blockDim.x) {
    std::int64_t j = 0;
    std::int64_t i = 0;
    atm_pair_from_flat(pair, begin, &j, &i);
    const double* vij = cache.pair_data + pair_index(batch, system, j, i) * kGfn2D4PairDataElements;
    const double r2ij = vij[0] * vij[0] + vij[1] * vij[1] + vij[2] * vij[2];
    if (r2ij > kAtmCutoffSquared) {
      continue;
    }
    const PairCoefficient c6ij = pair_coefficient(batch, parameters, workspace, i, j);
    for (std::int64_t k = begin; k < j; ++k) {
      const double* vik =
          cache.pair_data + pair_index(batch, system, k, i) * kGfn2D4PairDataElements;
      const double* vjk =
          cache.pair_data + pair_index(batch, system, k, j) * kGfn2D4PairDataElements;
      const double r2ik = vik[0] * vik[0] + vik[1] * vik[1] + vik[2] * vik[2];
      const double r2jk = vjk[0] * vjk[0] + vjk[1] * vjk[1] + vjk[2] * vjk[2];
      if (r2ik > kAtmCutoffSquared || r2jk > kAtmCutoffSquared) {
        continue;
      }
      const PairCoefficient c6ik = pair_coefficient(batch, parameters, workspace, i, k);
      const PairCoefficient c6jk = pair_coefficient(batch, parameters, workspace, j, k);
      if (!(c6ij.c6 > 0.0) || !(c6ik.c6 > 0.0) || !(c6jk.c6 > 0.0)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
        return;
      }
      const double r0ij = pair_damping_radius(batch, parameters, j, i);
      const double r0ik = pair_damping_radius(batch, parameters, k, i);
      const double r0jk = pair_damping_radius(batch, parameters, k, j);
      const double r2_product = r2ij * r2ik * r2jk;
      const double r1_product = sqrt(r2_product);
      const double r3_product = r2_product * r1_product;
      const double r5_product = r3_product * r2_product;
      const double ratio = (r0ij * r0ik * r0jk) / r1_product;
      const double ratio_power = pow(ratio, exponent_third);
      const double damping = 1.0 / (1.0 + 6.0 * ratio_power);
      const double angle =
          0.375 * (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) * (-r2ij + r2jk + r2ik) / r5_product +
          1.0 / r3_product;
      const double c9 = -kDispersionS9 * sqrt(c6ij.c6 * c6ik.c6 * c6jk.c6);
      const double rr = angle * damping;
      const double damping_derivative = -2.0 * kAtmExponent * ratio_power * damping * damping;
      double dgij[3];
      double dgik[3];
      double dgjk[3];
      atm_distance_gradient(r2ij, r2jk, r2ik, vij, r5_product, c9, angle, damping,
                            damping_derivative, dgij);
      atm_distance_gradient(r2ik, r2jk, r2ij, vik, r5_product, c9, angle, damping,
                            damping_derivative, dgik);
      atm_distance_gradient(r2jk, r2ik, r2ij, vjk, r5_product, c9, angle, damping,
                            damping_derivative, dgjk);
      const double i_adjoint = -0.5 * rr * c9 * (c6ij.first_cn / c6ij.c6 + c6ik.first_cn / c6ik.c6);
      const double j_adjoint =
          -0.5 * rr * c9 * (c6ij.second_cn / c6ij.c6 + c6jk.first_cn / c6jk.c6);
      const double k_adjoint =
          -0.5 * rr * c9 * (c6ik.second_cn / c6ik.c6 + c6jk.second_cn / c6jk.c6);
      if (!isfinite(dgij[0]) || !isfinite(dgij[1]) || !isfinite(dgij[2]) || !isfinite(dgik[0]) ||
          !isfinite(dgik[1]) || !isfinite(dgik[2]) || !isfinite(dgjk[0]) || !isfinite(dgjk[1]) ||
          !isfinite(dgjk[2]) || !isfinite(i_adjoint) || !isfinite(j_adjoint) ||
          !isfinite(k_adjoint)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
        return;
      }
      for (int axis = 0; axis < 3; ++axis) {
        atomic_add_fp64(workspace.gradient_scratch + i * 3 + axis, -dgij[axis] - dgik[axis]);
        atomic_add_fp64(workspace.gradient_scratch + j * 3 + axis, dgij[axis] - dgjk[axis]);
        atomic_add_fp64(workspace.gradient_scratch + k * 3 + axis, dgik[axis] + dgjk[axis]);
      }
      atomic_add_fp64(workspace.coordination_adjoints + i, i_adjoint);
      atomic_add_fp64(workspace.coordination_adjoints + j, j_adjoint);
      atomic_add_fp64(workspace.coordination_adjoints + k, k_adjoint);
    }
  }
}

/* Each canonical outer pair (j,i) intersects the two ascending neighbor
 * ranges.  A common k with k<j yields exactly one ordered triple k<j<i; no
 * triplet list or dense pair lookup is materialized.  Large systems use the
 * same arithmetic through a two-dimensional launch: blockIdx.y selects the
 * system while blockIdx.x owns one contiguous slice of its committed outer
 * pairs.  The one-block path retains the historical blockIdx.x system map. */
__global__ void pairlist_atm_energy_kernel(
    Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters, Gfn2D4PairListDeviceCache cache,
    Gfn2D4DeviceWorkspace workspace, std::int32_t blocks_per_system, std::uint32_t* device_error) {
  const bool split = blocks_per_system > 1;
  const std::int64_t system =
      split ? static_cast<std::int64_t>(blockIdx.y) : static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t slice = split ? static_cast<std::int64_t>(blockIdx.x) : 0;
  const std::int64_t slices = split ? static_cast<std::int64_t>(blocks_per_system) : 1;
  __shared__ double partial[kThreadsPerBlock];
  if (!sequence_is_valid(device_error) || !system_is_valid(workspace, system) ||
      !d4_pairlist_role_active(cache.atm_pairs, system)) {
    return;
  }
  constexpr double exponent_third = kAtmExponent / 3.0;
  const std::int64_t pair_begin = cache.atm_pairs.pair_offsets[system];
  const std::int64_t pair_count = cache.atm_pairs.pair_counts[system];
  /* Form floor(pair_count * slice / slices) without overflowing the signed
   * product for a hostile but structurally representable capacity. */
  const std::int64_t pairs_per_slice = pair_count / slices;
  const std::int64_t remainder = pair_count % slices;
  const std::int64_t slice_begin = pairs_per_slice * slice + remainder * slice / slices;
  const std::int64_t slice_end = pairs_per_slice * (slice + 1) + remainder * (slice + 1) / slices;
  double energy = 0.0;
  for (std::int64_t local = slice_begin + threadIdx.x; local < slice_end; local += blockDim.x) {
    const Gfn2AtomPair outer = cache.atm_pairs.pairs[pair_begin + local];
    const std::int64_t j = outer.first;
    const std::int64_t i = outer.second;
    D4PairGeometry ij{};
    if (!evaluate_d4_pair_geometry(batch, parameters, cache.positions, j, i, &ij)) {
      record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
      energy = 0.0;
      break;
    }
    if (ij.distance_squared > kAtmCutoffSquared) {
      continue;
    }
    const PairCoefficient c6ij = pair_coefficient(batch, parameters, workspace, i, j);
    std::int64_t i_cursor = cache.atm_pairs.neighbor_offsets[i];
    const std::int64_t i_end = i_cursor + cache.atm_pairs.neighbor_counts[i];
    std::int64_t j_cursor = cache.atm_pairs.neighbor_offsets[j];
    const std::int64_t j_end = j_cursor + cache.atm_pairs.neighbor_counts[j];
    while (i_cursor < i_end && j_cursor < j_end) {
      const std::int64_t i_neighbor = cache.atm_pairs.neighbors[i_cursor];
      const std::int64_t j_neighbor = cache.atm_pairs.neighbors[j_cursor];
      if (i_neighbor < j_neighbor) {
        ++i_cursor;
        continue;
      }
      if (j_neighbor < i_neighbor) {
        ++j_cursor;
        continue;
      }
      const std::int64_t k = i_neighbor;
      ++i_cursor;
      ++j_cursor;
      if (k >= j) {
        continue;
      }
      D4PairGeometry ik{};
      D4PairGeometry jk{};
      if (!evaluate_d4_pair_geometry(batch, parameters, cache.positions, k, i, &ik) ||
          !evaluate_d4_pair_geometry(batch, parameters, cache.positions, k, j, &jk)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
        energy = 0.0;
        break;
      }
      if (ik.distance_squared > kAtmCutoffSquared || jk.distance_squared > kAtmCutoffSquared) {
        continue;
      }
      const PairCoefficient c6ik = pair_coefficient(batch, parameters, workspace, i, k);
      const PairCoefficient c6jk = pair_coefficient(batch, parameters, workspace, j, k);
      const double r0ij = pair_damping_radius(batch, parameters, j, i);
      const double r0ik = pair_damping_radius(batch, parameters, k, i);
      const double r0jk = pair_damping_radius(batch, parameters, k, j);
      const double r2_product = ij.distance_squared * ik.distance_squared * jk.distance_squared;
      const double r1_product = sqrt(r2_product);
      const double r3_product = r2_product * r1_product;
      const double r5_product = r3_product * r2_product;
      const double damping =
          1.0 / (1.0 + 6.0 * pow((r0ij * r0ik * r0jk) / r1_product, exponent_third));
      const double angle =
          0.375 * (ij.distance_squared + jk.distance_squared - ik.distance_squared) *
              (ij.distance_squared - jk.distance_squared + ik.distance_squared) *
              (-ij.distance_squared + jk.distance_squared + ik.distance_squared) / r5_product +
          1.0 / r3_product;
      const double c9 = -kDispersionS9 * sqrt(fabs(c6ij.c6 * c6ik.c6 * c6jk.c6));
      energy -= angle * damping * c9;
      if (!isfinite(energy)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
        energy = 0.0;
        break;
      }
    }
  }
  partial[threadIdx.x] = energy;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (split) {
      /* atom_scratch is sized to total_atoms.  The host split gate proves
       * batch_size * blocks_per_system fits before selecting this path. */
      workspace.atom_scratch[system * slices + slice] = partial[0];
    } else {
      commit_energy_reduction(partial[0], workspace, system, device_error);
    }
  }
}

__global__ void pairlist_atm_gradient_kernel(
    Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters, Gfn2D4PairListDeviceCache cache,
    Gfn2D4DeviceWorkspace workspace, std::int32_t blocks_per_system, std::uint32_t* device_error) {
  const bool split = blocks_per_system > 1;
  const std::int64_t system =
      split ? static_cast<std::int64_t>(blockIdx.y) : static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t slice = split ? static_cast<std::int64_t>(blockIdx.x) : 0;
  const std::int64_t slices = split ? static_cast<std::int64_t>(blocks_per_system) : 1;
  if (!sequence_is_valid(device_error) || !system_is_valid(workspace, system) ||
      !d4_pairlist_role_active(cache.atm_pairs, system)) {
    return;
  }
  constexpr double exponent_third = kAtmExponent / 3.0;
  const std::int64_t pair_begin = cache.atm_pairs.pair_offsets[system];
  const std::int64_t pair_count = cache.atm_pairs.pair_counts[system];
  const std::int64_t pairs_per_slice = pair_count / slices;
  const std::int64_t remainder = pair_count % slices;
  const std::int64_t slice_begin = pairs_per_slice * slice + remainder * slice / slices;
  const std::int64_t slice_end = pairs_per_slice * (slice + 1) + remainder * (slice + 1) / slices;
  for (std::int64_t local = slice_begin + threadIdx.x; local < slice_end; local += blockDim.x) {
    const Gfn2AtomPair outer = cache.atm_pairs.pairs[pair_begin + local];
    const std::int64_t j = outer.first;
    const std::int64_t i = outer.second;
    D4PairGeometry ij{};
    if (!evaluate_d4_pair_geometry(batch, parameters, cache.positions, j, i, &ij)) {
      record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
      return;
    }
    if (ij.distance_squared > kAtmCutoffSquared) {
      continue;
    }
    const PairCoefficient c6ij = pair_coefficient(batch, parameters, workspace, i, j);
    std::int64_t i_cursor = cache.atm_pairs.neighbor_offsets[i];
    const std::int64_t i_end = i_cursor + cache.atm_pairs.neighbor_counts[i];
    std::int64_t j_cursor = cache.atm_pairs.neighbor_offsets[j];
    const std::int64_t j_end = j_cursor + cache.atm_pairs.neighbor_counts[j];
    while (i_cursor < i_end && j_cursor < j_end) {
      const std::int64_t i_neighbor = cache.atm_pairs.neighbors[i_cursor];
      const std::int64_t j_neighbor = cache.atm_pairs.neighbors[j_cursor];
      if (i_neighbor < j_neighbor) {
        ++i_cursor;
        continue;
      }
      if (j_neighbor < i_neighbor) {
        ++j_cursor;
        continue;
      }
      const std::int64_t k = i_neighbor;
      ++i_cursor;
      ++j_cursor;
      if (k >= j) {
        continue;
      }
      D4PairGeometry ik{};
      D4PairGeometry jk{};
      if (!evaluate_d4_pair_geometry(batch, parameters, cache.positions, k, i, &ik) ||
          !evaluate_d4_pair_geometry(batch, parameters, cache.positions, k, j, &jk)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
        return;
      }
      if (ik.distance_squared > kAtmCutoffSquared || jk.distance_squared > kAtmCutoffSquared) {
        continue;
      }
      const PairCoefficient c6ik = pair_coefficient(batch, parameters, workspace, i, k);
      const PairCoefficient c6jk = pair_coefficient(batch, parameters, workspace, j, k);
      if (!(c6ij.c6 > 0.0) || !(c6ik.c6 > 0.0) || !(c6jk.c6 > 0.0)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
        return;
      }
      const double r0ij = pair_damping_radius(batch, parameters, j, i);
      const double r0ik = pair_damping_radius(batch, parameters, k, i);
      const double r0jk = pair_damping_radius(batch, parameters, k, j);
      const double r2_product = ij.distance_squared * ik.distance_squared * jk.distance_squared;
      const double r1_product = sqrt(r2_product);
      const double r3_product = r2_product * r1_product;
      const double r5_product = r3_product * r2_product;
      const double ratio = (r0ij * r0ik * r0jk) / r1_product;
      const double ratio_power = pow(ratio, exponent_third);
      const double damping = 1.0 / (1.0 + 6.0 * ratio_power);
      const double angle =
          0.375 * (ij.distance_squared + jk.distance_squared - ik.distance_squared) *
              (ij.distance_squared - jk.distance_squared + ik.distance_squared) *
              (-ij.distance_squared + jk.distance_squared + ik.distance_squared) / r5_product +
          1.0 / r3_product;
      const double c9 = -kDispersionS9 * sqrt(c6ij.c6 * c6ik.c6 * c6jk.c6);
      const double rr = angle * damping;
      const double damping_derivative = -2.0 * kAtmExponent * ratio_power * damping * damping;
      double dgij[3];
      double dgik[3];
      double dgjk[3];
      atm_distance_gradient(ij.distance_squared, jk.distance_squared, ik.distance_squared,
                            ij.vector, r5_product, c9, angle, damping, damping_derivative, dgij);
      atm_distance_gradient(ik.distance_squared, jk.distance_squared, ij.distance_squared,
                            ik.vector, r5_product, c9, angle, damping, damping_derivative, dgik);
      atm_distance_gradient(jk.distance_squared, ik.distance_squared, ij.distance_squared,
                            jk.vector, r5_product, c9, angle, damping, damping_derivative, dgjk);
      const double i_adjoint = -0.5 * rr * c9 * (c6ij.first_cn / c6ij.c6 + c6ik.first_cn / c6ik.c6);
      const double j_adjoint =
          -0.5 * rr * c9 * (c6ij.second_cn / c6ij.c6 + c6jk.first_cn / c6jk.c6);
      const double k_adjoint =
          -0.5 * rr * c9 * (c6ik.second_cn / c6ik.c6 + c6jk.second_cn / c6jk.c6);
      if (!isfinite(dgij[0]) || !isfinite(dgij[1]) || !isfinite(dgij[2]) || !isfinite(dgik[0]) ||
          !isfinite(dgik[1]) || !isfinite(dgik[2]) || !isfinite(dgjk[0]) || !isfinite(dgjk[1]) ||
          !isfinite(dgjk[2]) || !isfinite(i_adjoint) || !isfinite(j_adjoint) ||
          !isfinite(k_adjoint)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
        return;
      }
      for (int axis = 0; axis < 3; ++axis) {
        atomic_add_fp64(workspace.gradient_scratch + i * 3 + axis, -dgij[axis] - dgik[axis]);
        atomic_add_fp64(workspace.gradient_scratch + j * 3 + axis, dgij[axis] - dgjk[axis]);
        atomic_add_fp64(workspace.gradient_scratch + k * 3 + axis, dgik[axis] + dgjk[axis]);
      }
      atomic_add_fp64(workspace.coordination_adjoints + i, i_adjoint);
      atomic_add_fp64(workspace.coordination_adjoints + j, j_adjoint);
      atomic_add_fp64(workspace.coordination_adjoints + k, k_adjoint);
    }
  }
}

__global__ void pairlist_d4_coordination_vjp_kernel(Gfn2D4DeviceBatch batch,
                                                    Gfn2D4DeviceParameters parameters,
                                                    Gfn2D4PairListDeviceCache cache,
                                                    Gfn2D4DeviceWorkspace workspace,
                                                    std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!sequence_is_valid(device_error) || !system_is_valid(workspace, system) ||
      !d4_pairlist_role_active(cache.coordination_pairs, system)) {
    return;
  }
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    double contribution[3]{0.0, 0.0, 0.0};
    const std::int64_t begin = cache.coordination_pairs.neighbor_offsets[atom];
    const std::int64_t end = begin + cache.coordination_pairs.neighbor_counts[atom];
    for (std::int64_t index = begin; index < end; ++index) {
      const std::int64_t other = cache.coordination_pairs.neighbors[index];
      const std::int64_t first = atom < other ? atom : other;
      const std::int64_t second = atom < other ? other : atom;
      const double sign = atom == first ? 1.0 : -1.0;
      D4PairGeometry pair{};
      if (!evaluate_d4_pair_geometry(batch, parameters, cache.positions, first, second, &pair)) {
        record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
        return;
      }
      if (pair.distance_squared > kCoordinationCutoffSquared) {
        continue;
      }
      const Gfn2D4DeviceElementData first_element =
          parameters.elements[batch.atomic_numbers[first] - 1];
      const Gfn2D4DeviceElementData second_element =
          parameters.elements[batch.atomic_numbers[second] - 1];
      const double radius = first_element.covalent_radius + second_element.covalent_radius;
      const double en_delta =
          fabs(first_element.electronegativity - second_element.electronegativity);
      const double en_factor = kEnK4 * exp(-((en_delta + kEnK5) * (en_delta + kEnK5)) / kEnK6);
      const double distance = sqrt(pair.distance_squared);
      const double exponent = kCoordinationSteepness * (distance - radius) / radius;
      const double derivative =
          -en_factor * kCoordinationSteepness * exp(-exponent * exponent) * kInverseSqrtPi / radius;
      const double scale =
          sign * derivative *
          (workspace.coordination_adjoints[first] + workspace.coordination_adjoints[second]) /
          distance;
      for (int axis = 0; axis < 3; ++axis) {
        contribution[axis] += scale * pair.vector[axis];
      }
    }
    for (int axis = 0; axis < 3; ++axis) {
      workspace.gradient_scratch[atom * 3 + axis] += contribution[axis];
    }
    if (!isfinite(workspace.gradient_scratch[atom * 3]) ||
        !isfinite(workspace.gradient_scratch[atom * 3 + 1]) ||
        !isfinite(workspace.gradient_scratch[atom * 3 + 2])) {
      record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
    }
  }
}

__global__ void coordination_vjp_kernel(Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters,
                                        Gfn2D4DeviceCache cache, Gfn2D4DeviceWorkspace workspace,
                                        std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ std::uint32_t shared_error;
  if (!block_system_is_valid(workspace, system, device_error, shared_error)) {
    return;
  }
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
    double contribution[3]{0.0, 0.0, 0.0};
    for (std::int64_t other = begin; other < end; ++other) {
      if (atom == other) {
        continue;
      }
      const std::int64_t first = atom < other ? atom : other;
      const std::int64_t second = atom < other ? other : atom;
      const double sign = atom == first ? 1.0 : -1.0;
      const double* pair =
          cache.pair_data + pair_index(batch, system, first, second) * kGfn2D4PairDataElements;
      const double distance_squared = pair[0] * pair[0] + pair[1] * pair[1] + pair[2] * pair[2];
      if (distance_squared > kCoordinationCutoffSquared) {
        continue;
      }
      const Gfn2D4DeviceElementData first_element =
          parameters.elements[batch.atomic_numbers[first] - 1];
      const Gfn2D4DeviceElementData second_element =
          parameters.elements[batch.atomic_numbers[second] - 1];
      const double radius = first_element.covalent_radius + second_element.covalent_radius;
      const double en_delta =
          fabs(first_element.electronegativity - second_element.electronegativity);
      const double en_factor = kEnK4 * exp(-((en_delta + kEnK5) * (en_delta + kEnK5)) / kEnK6);
      const double distance = sqrt(distance_squared);
      const double exponent = kCoordinationSteepness * (distance - radius) / radius;
      const double derivative =
          -en_factor * kCoordinationSteepness * exp(-exponent * exponent) * kInverseSqrtPi / radius;
      const double scale =
          sign * derivative *
          (workspace.coordination_adjoints[first] + workspace.coordination_adjoints[second]) /
          distance;
      for (int axis = 0; axis < 3; ++axis) {
        contribution[axis] += scale * pair[axis];
      }
    }
    for (int axis = 0; axis < 3; ++axis) {
      workspace.gradient_scratch[atom * 3 + axis] += contribution[axis];
    }
    if (!isfinite(workspace.gradient_scratch[atom * 3]) ||
        !isfinite(workspace.gradient_scratch[atom * 3 + 1]) ||
        !isfinite(workspace.gradient_scratch[atom * 3 + 2])) {
      record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
      return;
    }
  }
}

__global__ void gradient_input_preflight_kernel(Gfn2D4DeviceBatch batch, const double* gradients,
                                                Gfn2D4DeviceWorkspace workspace,
                                                std::uint32_t* device_error) {
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < batch.total_atoms * 3 && sequence_is_valid(device_error)) {
    const std::int64_t system = system_for_atom(batch, index / 3);
    if (system_is_valid(workspace, system) && !isfinite(gradients[index])) {
      record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
    }
  }
}

__global__ void clear_atom_kernel(Gfn2D4DeviceBatch batch, std::int64_t count,
                                  std::int64_t elements_per_atom, double* values,
                                  const Gfn2D4DeviceWorkspace workspace,
                                  const std::uint32_t* device_error) {
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count && sequence_is_valid(device_error)) {
    const std::int64_t system = system_for_atom(batch, index / elements_per_atom);
    if (!system_is_valid(workspace, system)) {
      return;
    }
    values[index] = 0.0;
  }
}

__global__ void add_publish_atom_kernel(Gfn2D4DeviceBatch batch, std::int64_t count,
                                        std::int64_t elements_per_atom, const double* scratch,
                                        double* output, const Gfn2D4DeviceWorkspace workspace,
                                        const std::uint32_t* device_error) {
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count && sequence_is_valid(device_error)) {
    const std::int64_t system = system_for_atom(batch, index / elements_per_atom);
    if (!system_is_valid(workspace, system)) {
      return;
    }
    output[index] += scratch[index];
  }
}

__global__ void add_output_preflight_kernel(Gfn2D4DeviceBatch batch, std::int64_t count,
                                            std::int64_t elements_per_atom, const double* scratch,
                                            const double* output,
                                            const Gfn2D4DeviceWorkspace workspace,
                                            const std::uint32_t* device_error) {
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count && sequence_is_valid(device_error)) {
    const std::int64_t system = system_for_atom(batch, index / elements_per_atom);
    if (system_is_valid(workspace, system) && !isfinite(output[index] + scratch[index])) {
      record_system_error(workspace, system, Gfn2D4DeviceError::kNonfiniteArithmetic);
    }
  }
}

__global__ void publish_atom_kernel(Gfn2D4DeviceBatch batch, const double* scratch, double* output,
                                    const Gfn2D4DeviceWorkspace workspace,
                                    const std::uint32_t* device_error) {
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < batch.total_atoms && sequence_is_valid(device_error)) {
    const std::int64_t system = system_for_atom(batch, index);
    if (!system_is_valid(workspace, system)) {
      return;
    }
    output[index] = scratch[index];
  }
}

__global__ void publish_batch_kernel(std::int64_t batch_size, const double* scratch, double* output,
                                     const Gfn2D4DeviceWorkspace workspace,
                                     const std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system < batch_size && sequence_is_valid(device_error) &&
      system_is_valid(workspace, system)) {
    output[system] = scratch[system];
  }
}

/* A committed role's optional active mask is caller intent.  Inactive peers
 * leave caller outputs untouched while their fixed-capacity physical list
 * storage remains committed for later transactions. */
__global__ void publish_pairlist_batch_kernel(std::int64_t batch_size,
                                              const std::uint8_t* active_mask,
                                              const double* scratch, double* output,
                                              const Gfn2D4DeviceWorkspace workspace,
                                              const std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system < batch_size && (active_mask == nullptr || active_mask[system] == 1u) &&
      sequence_is_valid(device_error) && system_is_valid(workspace, system)) {
    output[system] = scratch[system];
  }
}

bool is_aligned(const void* pointer, std::size_t alignment) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

/*
 * Form a half-open byte range without dereferencing the CUDA address. Empty
 * ranges are normalized so a NULL zero-pair cache does not alias anything.
 */
bool make_address_range(const void* pointer, std::int64_t elements, std::size_t element_size,
                        AddressRange& range) {
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

bool ranges_overlap(const AddressRange& first, const AddressRange& second) {
  return first.begin < second.end && second.begin < first.end;
}

template <std::size_t ReadCount, std::size_t WriteCount>
bool writable_ranges_are_disjoint(const std::array<AddressRange, ReadCount>& read_ranges,
                                  const std::array<AddressRange, WriteCount>& write_ranges) {
  for (std::size_t write = 0; write < WriteCount; ++write) {
    for (const AddressRange& read_range : read_ranges) {
      if (ranges_overlap(write_ranges[write], read_range)) {
        return false;
      }
    }
    for (std::size_t other = write + 1u; other < WriteCount; ++other) {
      if (ranges_overlap(write_ranges[write], write_ranges[other])) {
        return false;
      }
    }
  }
  return true;
}

bool valid_geometry_refresh_descriptors(const Gfn2D4DeviceBatch& batch,
                                        const Gfn2D4DeviceParameters& parameters,
                                        const double* positions, const Gfn2D4DeviceCache& cache,
                                        const Gfn2D4DeviceWorkspace& workspace,
                                        const std::uint32_t* device_error) {
  const bool extents_representable =
      batch.batch_size > 0 && batch.batch_size <= std::numeric_limits<int>::max() &&
      batch.total_atoms > 0 && batch.total_atoms <= std::numeric_limits<std::int64_t>::max() / 3 &&
      batch.total_pairs >= 0 &&
      batch.total_pairs <= std::numeric_limits<std::int64_t>::max() / kGfn2D4PairDataElements;
  if (!extents_representable) {
    return false;
  }
  const std::int64_t coordinate_elements = batch.total_atoms * 3;
  const std::int64_t pair_elements = batch.total_pairs * kGfn2D4PairDataElements;
  if (batch.plan_token == 0u || batch.atom_offsets == nullptr || batch.pair_offsets == nullptr ||
      batch.atomic_numbers == nullptr || parameters.elements == nullptr ||
      parameters.element_count <= 0 || parameters.reference_count <= 0 || positions == nullptr ||
      cache.plan_token != batch.plan_token || cache.geometry_generation == 0u ||
      cache.pair_data_elements != pair_elements || cache.coordination_numbers == nullptr ||
      cache.coordination_elements != batch.total_atoms ||
      (pair_elements != 0 && cache.pair_data == nullptr) ||
      workspace.pair_scratch_elements < pair_elements ||
      (pair_elements != 0 && workspace.pair_scratch == nullptr) ||
      workspace.coordination_scratch == nullptr ||
      workspace.coordination_scratch_elements < batch.total_atoms ||
      workspace.geometry_generations == nullptr ||
      workspace.geometry_generation_elements < batch.batch_size ||
      workspace.geometry_sequence_active == nullptr || workspace.geometry_sequence_elements < 1 ||
      workspace.system_errors == nullptr || workspace.system_error_elements < batch.batch_size ||
      device_error == nullptr || !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.pair_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.atomic_numbers, alignof(std::int32_t)) ||
      !is_aligned(parameters.elements, alignof(Gfn2D4DeviceElementData)) ||
      !is_aligned(positions, alignof(double)) ||
      (cache.pair_data != nullptr && !is_aligned(cache.pair_data, alignof(double))) ||
      !is_aligned(cache.coordination_numbers, alignof(double)) ||
      (workspace.pair_scratch != nullptr && !is_aligned(workspace.pair_scratch, alignof(double))) ||
      !is_aligned(workspace.coordination_scratch, alignof(double)) ||
      !is_aligned(workspace.geometry_generations, alignof(std::uint64_t)) ||
      !is_aligned(workspace.geometry_sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(workspace.system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return false;
  }

  std::array<AddressRange, 5> reads;
  std::array<AddressRange, 8> writes;
  return make_address_range(batch.atom_offsets, batch.batch_size + 1, sizeof(*batch.atom_offsets),
                            reads[0]) &&
         make_address_range(batch.pair_offsets, batch.batch_size + 1, sizeof(*batch.pair_offsets),
                            reads[1]) &&
         make_address_range(batch.atomic_numbers, batch.total_atoms, sizeof(*batch.atomic_numbers),
                            reads[2]) &&
         make_address_range(parameters.elements, parameters.element_count,
                            sizeof(*parameters.elements), reads[3]) &&
         make_address_range(positions, coordinate_elements, sizeof(*positions), reads[4]) &&
         make_address_range(cache.pair_data, pair_elements, sizeof(*cache.pair_data), writes[0]) &&
         make_address_range(cache.coordination_numbers, batch.total_atoms,
                            sizeof(*cache.coordination_numbers), writes[1]) &&
         make_address_range(workspace.pair_scratch, pair_elements, sizeof(*workspace.pair_scratch),
                            writes[2]) &&
         make_address_range(workspace.coordination_scratch, batch.total_atoms,
                            sizeof(*workspace.coordination_scratch), writes[3]) &&
         make_address_range(workspace.geometry_generations, batch.batch_size,
                            sizeof(*workspace.geometry_generations), writes[4]) &&
         make_address_range(workspace.geometry_sequence_active, 1,
                            sizeof(*workspace.geometry_sequence_active), writes[5]) &&
         make_address_range(workspace.system_errors, batch.batch_size,
                            sizeof(*workspace.system_errors), writes[6]) &&
         make_address_range(device_error, 1, sizeof(*device_error), writes[7]) &&
         writable_ranges_are_disjoint(reads, writes);
}

bool valid_common_descriptors(const Gfn2D4DeviceBatch& batch,
                              const Gfn2D4DeviceParameters& parameters,
                              const Gfn2D4DeviceCache& cache,
                              const Gfn2D4DeviceWorkspace& workspace,
                              const std::uint32_t* device_error) {
  const bool reference_square_representable =
      parameters.reference_count > 0 &&
      parameters.reference_count <=
          std::numeric_limits<std::int64_t>::max() / parameters.reference_count;
  const bool extents_representable =
      batch.total_atoms > 0 &&
      batch.total_atoms <= std::numeric_limits<std::int64_t>::max() / kGfn2D4MaximumReferences &&
      batch.total_atoms <= std::numeric_limits<std::int64_t>::max() / 3 && batch.total_pairs >= 0 &&
      batch.total_pairs <= std::numeric_limits<std::int64_t>::max() / kGfn2D4PairDataElements;
  if (!reference_square_representable || !extents_representable) {
    return false;
  }
  const std::int64_t pair_elements = batch.total_pairs * kGfn2D4PairDataElements;
  const std::int64_t weight_elements = batch.total_atoms * kGfn2D4MaximumReferences;
  const std::int64_t gradient_elements = batch.total_atoms * 3;
  const std::int64_t reference_c6_elements =
      parameters.reference_count * parameters.reference_count;
  return batch.batch_size > 0 && batch.batch_size <= std::numeric_limits<unsigned int>::max() &&
         batch.plan_token != 0u && batch.atom_offsets != nullptr && batch.pair_offsets != nullptr &&
         batch.atomic_numbers != nullptr && parameters.elements != nullptr &&
         parameters.element_count > 0 && parameters.references != nullptr &&
         parameters.reference_c6 != nullptr &&
         parameters.reference_c6_elements >= reference_c6_elements &&
         cache.plan_token == batch.plan_token && cache.geometry_generation != 0u &&
         (batch.total_pairs == 0 || cache.pair_data != nullptr) &&
         cache.pair_data_elements == pair_elements && cache.coordination_numbers != nullptr &&
         cache.coordination_elements == batch.total_atoms && workspace.weights != nullptr &&
         workspace.weight_cn_derivatives != nullptr &&
         workspace.weight_charge_derivatives != nullptr &&
         workspace.weight_elements >= weight_elements && workspace.atom_scratch != nullptr &&
         workspace.coordination_adjoints != nullptr &&
         workspace.atom_elements >= batch.total_atoms && workspace.batch_scratch != nullptr &&
         workspace.batch_elements >= batch.batch_size && workspace.gradient_scratch != nullptr &&
         workspace.gradient_elements >= gradient_elements && workspace.system_errors != nullptr &&
         workspace.system_error_elements >= batch.batch_size && device_error != nullptr &&
         is_aligned(batch.atom_offsets, alignof(std::int64_t)) &&
         is_aligned(batch.pair_offsets, alignof(std::int64_t)) &&
         is_aligned(batch.atomic_numbers, alignof(std::int32_t)) &&
         is_aligned(parameters.elements, alignof(Gfn2D4DeviceElementData)) &&
         is_aligned(parameters.references, alignof(Gfn2D4DeviceReferenceData)) &&
         is_aligned(parameters.reference_c6, alignof(double)) &&
         (cache.pair_data == nullptr || is_aligned(cache.pair_data, alignof(double))) &&
         is_aligned(cache.coordination_numbers, alignof(double)) &&
         is_aligned(workspace.weights, alignof(double)) &&
         is_aligned(workspace.weight_cn_derivatives, alignof(double)) &&
         is_aligned(workspace.weight_charge_derivatives, alignof(double)) &&
         is_aligned(workspace.atom_scratch, alignof(double)) &&
         is_aligned(workspace.coordination_adjoints, alignof(double)) &&
         is_aligned(workspace.batch_scratch, alignof(double)) &&
         is_aligned(workspace.gradient_scratch, alignof(double)) &&
         is_aligned(workspace.system_errors, alignof(std::uint32_t)) &&
         is_aligned(device_error, alignof(std::uint32_t));
}

bool valid_scc_descriptors(const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
                           const Gfn2D4DeviceCache& cache,
                           const Gfn2SccIterationDeviceActivity& activity,
                           const Gfn2D4DeviceWorkspace& workspace, bool potential_mode,
                           const std::uint32_t* device_error) {
  const bool reference_square_representable =
      parameters.reference_count > 0 &&
      parameters.reference_count <=
          std::numeric_limits<std::int64_t>::max() / parameters.reference_count;
  const bool extents_representable =
      batch.total_atoms > 0 &&
      batch.total_atoms <= std::numeric_limits<std::int64_t>::max() / kGfn2D4MaximumReferences &&
      batch.total_pairs >= 0 &&
      batch.total_pairs <= std::numeric_limits<std::int64_t>::max() / kGfn2D4PairDataElements;
  if (!reference_square_representable || !extents_representable) {
    return false;
  }
  const std::int64_t pair_elements = batch.total_pairs * kGfn2D4PairDataElements;
  const std::int64_t weight_elements = batch.total_atoms * kGfn2D4MaximumReferences;
  const std::int64_t reference_c6_elements =
      parameters.reference_count * parameters.reference_count;
  const bool mode_workspace_valid =
      potential_mode
          ? workspace.weight_charge_derivatives != nullptr && workspace.atom_scratch != nullptr &&
                workspace.atom_elements >= batch.total_atoms &&
                is_aligned(workspace.weight_charge_derivatives, alignof(double)) &&
                is_aligned(workspace.atom_scratch, alignof(double))
          : workspace.batch_scratch != nullptr && workspace.batch_elements >= batch.batch_size &&
                is_aligned(workspace.batch_scratch, alignof(double));
  return batch.batch_size > 0 && batch.batch_size <= std::numeric_limits<unsigned int>::max() &&
         batch.plan_token != 0u && batch.atom_offsets != nullptr && batch.pair_offsets != nullptr &&
         batch.atomic_numbers != nullptr && parameters.elements != nullptr &&
         parameters.element_count > 0 && parameters.references != nullptr &&
         parameters.reference_c6 != nullptr &&
         parameters.reference_c6_elements >= reference_c6_elements &&
         cache.plan_token == batch.plan_token &&
         (batch.total_pairs == 0 || cache.pair_data != nullptr) &&
         cache.pair_data_elements == pair_elements && cache.coordination_numbers != nullptr &&
         cache.coordination_elements == batch.total_atoms &&
         activity.plan_token == batch.plan_token && activity.batch_elements == batch.batch_size &&
         activity.sequence_elements == 1 && activity.active_mask != nullptr &&
         activity.sequence_active != nullptr && workspace.weights != nullptr &&
         workspace.weight_elements >= weight_elements && workspace.system_errors != nullptr &&
         workspace.system_error_elements >= batch.batch_size && mode_workspace_valid &&
         device_error != nullptr && is_aligned(batch.atom_offsets, alignof(std::int64_t)) &&
         is_aligned(batch.pair_offsets, alignof(std::int64_t)) &&
         is_aligned(batch.atomic_numbers, alignof(std::int32_t)) &&
         is_aligned(parameters.elements, alignof(Gfn2D4DeviceElementData)) &&
         is_aligned(parameters.references, alignof(Gfn2D4DeviceReferenceData)) &&
         is_aligned(parameters.reference_c6, alignof(double)) &&
         (cache.pair_data == nullptr || is_aligned(cache.pair_data, alignof(double))) &&
         is_aligned(cache.coordination_numbers, alignof(double)) &&
         is_aligned(activity.active_mask, alignof(std::uint8_t)) &&
         is_aligned(activity.sequence_active, alignof(std::uint32_t)) &&
         is_aligned(workspace.weights, alignof(double)) &&
         is_aligned(workspace.system_errors, alignof(std::uint32_t)) &&
         is_aligned(device_error, alignof(std::uint32_t));
}

bool valid_scc_potential_ranges(const Gfn2D4DeviceBatch& batch,
                                const Gfn2D4DeviceParameters& parameters,
                                const Gfn2D4DeviceCache& cache, const double* charges,
                                const Gfn2SccIterationDeviceActivity& activity, double* potentials,
                                const Gfn2D4DeviceWorkspace& workspace,
                                std::uint32_t* device_error) {
  const std::int64_t pair_elements = batch.total_pairs * kGfn2D4PairDataElements;
  const std::int64_t weight_elements = batch.total_atoms * kGfn2D4MaximumReferences;
  const std::int64_t reference_c6_elements =
      parameters.reference_count * parameters.reference_count;
  std::array<AddressRange, 11> reads;
  std::array<AddressRange, 6> writes;
  return make_address_range(batch.atom_offsets, batch.batch_size + 1, sizeof(*batch.atom_offsets),
                            reads[0]) &&
         make_address_range(batch.pair_offsets, batch.batch_size + 1, sizeof(*batch.pair_offsets),
                            reads[1]) &&
         make_address_range(batch.atomic_numbers, batch.total_atoms, sizeof(*batch.atomic_numbers),
                            reads[2]) &&
         make_address_range(parameters.elements, parameters.element_count,
                            sizeof(*parameters.elements), reads[3]) &&
         make_address_range(parameters.references, parameters.reference_count,
                            sizeof(*parameters.references), reads[4]) &&
         make_address_range(parameters.reference_c6, reference_c6_elements,
                            sizeof(*parameters.reference_c6), reads[5]) &&
         make_address_range(cache.pair_data, pair_elements, sizeof(*cache.pair_data), reads[6]) &&
         make_address_range(cache.coordination_numbers, batch.total_atoms,
                            sizeof(*cache.coordination_numbers), reads[7]) &&
         make_address_range(charges, batch.total_atoms, sizeof(*charges), reads[8]) &&
         make_address_range(activity.active_mask, batch.batch_size, sizeof(*activity.active_mask),
                            reads[9]) &&
         make_address_range(activity.sequence_active, 1, sizeof(*activity.sequence_active),
                            reads[10]) &&
         make_address_range(workspace.weights, weight_elements, sizeof(*workspace.weights),
                            writes[0]) &&
         make_address_range(workspace.weight_charge_derivatives, weight_elements,
                            sizeof(*workspace.weight_charge_derivatives), writes[1]) &&
         make_address_range(workspace.atom_scratch, batch.total_atoms,
                            sizeof(*workspace.atom_scratch), writes[2]) &&
         make_address_range(workspace.system_errors, batch.batch_size,
                            sizeof(*workspace.system_errors), writes[3]) &&
         make_address_range(potentials, batch.total_atoms, sizeof(*potentials), writes[4]) &&
         make_address_range(device_error, 1, sizeof(*device_error), writes[5]) &&
         writable_ranges_are_disjoint(reads, writes);
}

bool valid_scc_energy_ranges(const Gfn2D4DeviceBatch& batch,
                             const Gfn2D4DeviceParameters& parameters,
                             const Gfn2D4DeviceCache& cache, const double* charges,
                             const Gfn2SccIterationDeviceActivity& activity, double* energies,
                             const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error) {
  const std::int64_t pair_elements = batch.total_pairs * kGfn2D4PairDataElements;
  const std::int64_t weight_elements = batch.total_atoms * kGfn2D4MaximumReferences;
  const std::int64_t reference_c6_elements =
      parameters.reference_count * parameters.reference_count;
  std::array<AddressRange, 11> reads;
  std::array<AddressRange, 5> writes;
  return make_address_range(batch.atom_offsets, batch.batch_size + 1, sizeof(*batch.atom_offsets),
                            reads[0]) &&
         make_address_range(batch.pair_offsets, batch.batch_size + 1, sizeof(*batch.pair_offsets),
                            reads[1]) &&
         make_address_range(batch.atomic_numbers, batch.total_atoms, sizeof(*batch.atomic_numbers),
                            reads[2]) &&
         make_address_range(parameters.elements, parameters.element_count,
                            sizeof(*parameters.elements), reads[3]) &&
         make_address_range(parameters.references, parameters.reference_count,
                            sizeof(*parameters.references), reads[4]) &&
         make_address_range(parameters.reference_c6, reference_c6_elements,
                            sizeof(*parameters.reference_c6), reads[5]) &&
         make_address_range(cache.pair_data, pair_elements, sizeof(*cache.pair_data), reads[6]) &&
         make_address_range(cache.coordination_numbers, batch.total_atoms,
                            sizeof(*cache.coordination_numbers), reads[7]) &&
         make_address_range(charges, batch.total_atoms, sizeof(*charges), reads[8]) &&
         make_address_range(activity.active_mask, batch.batch_size, sizeof(*activity.active_mask),
                            reads[9]) &&
         make_address_range(activity.sequence_active, 1, sizeof(*activity.sequence_active),
                            reads[10]) &&
         make_address_range(workspace.weights, weight_elements, sizeof(*workspace.weights),
                            writes[0]) &&
         make_address_range(workspace.batch_scratch, batch.batch_size,
                            sizeof(*workspace.batch_scratch), writes[1]) &&
         make_address_range(workspace.system_errors, batch.batch_size,
                            sizeof(*workspace.system_errors), writes[2]) &&
         make_address_range(energies, batch.batch_size, sizeof(*energies), writes[3]) &&
         make_address_range(device_error, 1, sizeof(*device_error), writes[4]) &&
         writable_ranges_are_disjoint(reads, writes);
}

template <std::size_t ReadCount, std::size_t WriteCount>
bool make_common_ranges(const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
                        const Gfn2D4DeviceCache& cache, const Gfn2D4DeviceWorkspace& workspace,
                        std::array<AddressRange, ReadCount>& reads,
                        std::array<AddressRange, WriteCount>& writes) {
  static_assert(ReadCount >= 8u);
  static_assert(WriteCount >= 8u);
  return make_address_range(batch.atom_offsets, batch.batch_size + 1, sizeof(*batch.atom_offsets),
                            reads[0]) &&
         make_address_range(batch.pair_offsets, batch.batch_size + 1, sizeof(*batch.pair_offsets),
                            reads[1]) &&
         make_address_range(batch.atomic_numbers, batch.total_atoms, sizeof(*batch.atomic_numbers),
                            reads[2]) &&
         make_address_range(parameters.elements, parameters.element_count,
                            sizeof(*parameters.elements), reads[3]) &&
         make_address_range(parameters.references, parameters.reference_count,
                            sizeof(*parameters.references), reads[4]) &&
         make_address_range(parameters.reference_c6, parameters.reference_c6_elements,
                            sizeof(*parameters.reference_c6), reads[5]) &&
         make_address_range(cache.pair_data, cache.pair_data_elements, sizeof(*cache.pair_data),
                            reads[6]) &&
         make_address_range(cache.coordination_numbers, batch.total_atoms,
                            sizeof(*cache.coordination_numbers), reads[7]) &&
         make_address_range(workspace.weights, workspace.weight_elements,
                            sizeof(*workspace.weights), writes[0]) &&
         make_address_range(workspace.weight_cn_derivatives, workspace.weight_elements,
                            sizeof(*workspace.weight_cn_derivatives), writes[1]) &&
         make_address_range(workspace.weight_charge_derivatives, workspace.weight_elements,
                            sizeof(*workspace.weight_charge_derivatives), writes[2]) &&
         make_address_range(workspace.atom_scratch, workspace.atom_elements,
                            sizeof(*workspace.atom_scratch), writes[3]) &&
         make_address_range(workspace.coordination_adjoints, workspace.atom_elements,
                            sizeof(*workspace.coordination_adjoints), writes[4]) &&
         make_address_range(workspace.batch_scratch, workspace.batch_elements,
                            sizeof(*workspace.batch_scratch), writes[5]) &&
         make_address_range(workspace.gradient_scratch, workspace.gradient_elements,
                            sizeof(*workspace.gradient_scratch), writes[6]) &&
         make_address_range(workspace.system_errors, workspace.system_error_elements,
                            sizeof(*workspace.system_errors), writes[7]);
}

bool same_pairlist_storage(const Gfn2PairListConsumerView& first,
                           const Gfn2PairListConsumerView& second) {
  return first.memory_space == second.memory_space && first.state == second.state &&
         first.pair_map_kind == second.pair_map_kind && first.plan_token == second.plan_token &&
         first.list_builder_cutoff_bohr == second.list_builder_cutoff_bohr &&
         first.batch_size == second.batch_size && first.total_atoms == second.total_atoms &&
         first.max_pairs_per_system == second.max_pairs_per_system &&
         first.max_neighbors_per_atom == second.max_neighbors_per_atom &&
         first.pair_offset_count == second.pair_offset_count &&
         first.neighbor_offset_count == second.neighbor_offset_count &&
         first.pair_count == second.pair_count && first.neighbor_count == second.neighbor_count &&
         first.pair_offsets == second.pair_offsets && first.pairs == second.pairs &&
         first.pair_count_elements == second.pair_count_elements &&
         first.neighbor_count_elements == second.neighbor_count_elements &&
         first.pair_counts == second.pair_counts &&
         first.neighbor_counts == second.neighbor_counts &&
         first.neighbor_offsets == second.neighbor_offsets && first.neighbors == second.neighbors &&
         first.committed_generation_count == second.committed_generation_count &&
         first.eligible_mask_count == second.eligible_mask_count &&
         first.active_mask_count == second.active_mask_count &&
         first.committed_generations == second.committed_generations &&
         first.eligible_mask == second.eligible_mask && first.active_mask == second.active_mask;
}

bool valid_pairlist_role(const Gfn2PairListConsumerView& view, Gfn2PairListRole role, double cutoff,
                         const Gfn2D4DeviceBatch& batch) {
  if (view.memory_space != Gfn2PlanMemorySpace::kCudaDevice ||
      view.state != Gfn2PairListState::kCommitted || view.role != role ||
      view.pair_map_kind != Gfn2PairMapKind::kExplicit || view.plan_token != batch.plan_token ||
      view.cutoff_bohr != cutoff || view.list_builder_cutoff_bohr != kGfn2D4TwoBodyCutoffBohr ||
      view.batch_size != batch.batch_size || view.total_atoms != batch.total_atoms ||
      view.max_pairs_per_system <= 0 || view.max_neighbors_per_atom <= 0 ||
      view.max_pairs_per_system > kMaximumInt64 / batch.batch_size ||
      view.max_neighbors_per_atom > kMaximumInt64 / batch.total_atoms ||
      view.pair_count != batch.batch_size * view.max_pairs_per_system ||
      view.neighbor_count != batch.total_atoms * view.max_neighbors_per_atom ||
      view.pair_offset_count != batch.batch_size + 1 ||
      view.neighbor_offset_count != batch.total_atoms + 1 ||
      view.pair_count_elements != batch.batch_size ||
      view.neighbor_count_elements != batch.total_atoms ||
      view.committed_generation_count != batch.batch_size ||
      view.eligible_mask_count != batch.batch_size || view.pair_offsets == nullptr ||
      view.pairs == nullptr || view.pair_counts == nullptr || view.neighbor_offsets == nullptr ||
      view.neighbor_counts == nullptr || view.neighbors == nullptr ||
      view.committed_generations == nullptr || view.eligible_mask == nullptr ||
      !((view.active_mask == nullptr && view.active_mask_count == 0) ||
        (view.active_mask != nullptr && view.active_mask_count == batch.batch_size))) {
    return false;
  }
  return is_aligned(view.pair_offsets, alignof(std::int64_t)) &&
         is_aligned(view.pairs, alignof(Gfn2AtomPair)) &&
         is_aligned(view.pair_counts, alignof(std::int64_t)) &&
         is_aligned(view.neighbor_offsets, alignof(std::int64_t)) &&
         is_aligned(view.neighbor_counts, alignof(std::int64_t)) &&
         is_aligned(view.neighbors, alignof(std::int64_t)) &&
         is_aligned(view.committed_generations, alignof(std::uint64_t));
}

bool valid_pairlist_structural_aliases(const Gfn2D4DeviceBatch& batch,
                                       const Gfn2D4DeviceParameters& parameters,
                                       const Gfn2D4PairListDeviceCache& cache,
                                       const Gfn2GeometryEpochDevice* geometry_epoch) {
  const Gfn2PairListConsumerView& view = cache.two_body_pairs;
  std::array<AddressRange, 9> structural;
  std::array<AddressRange, 8> other;
  if (!make_address_range(view.pair_offsets, view.pair_offset_count, sizeof(*view.pair_offsets),
                          structural[0]) ||
      !make_address_range(view.pairs, view.pair_count, sizeof(*view.pairs), structural[1]) ||
      !make_address_range(view.pair_counts, view.pair_count_elements, sizeof(*view.pair_counts),
                          structural[2]) ||
      !make_address_range(view.neighbor_offsets, view.neighbor_offset_count,
                          sizeof(*view.neighbor_offsets), structural[3]) ||
      !make_address_range(view.neighbor_counts, view.neighbor_count_elements,
                          sizeof(*view.neighbor_counts), structural[4]) ||
      !make_address_range(view.neighbors, view.neighbor_count, sizeof(*view.neighbors),
                          structural[5]) ||
      !make_address_range(view.committed_generations, view.committed_generation_count,
                          sizeof(*view.committed_generations), structural[6]) ||
      !make_address_range(view.eligible_mask, view.eligible_mask_count, sizeof(*view.eligible_mask),
                          structural[7]) ||
      !make_address_range(view.active_mask, view.active_mask_count, sizeof(*view.active_mask),
                          structural[8]) ||
      !make_address_range(batch.atom_offsets, batch.batch_size + 1, sizeof(*batch.atom_offsets),
                          other[0]) ||
      !make_address_range(batch.pair_offsets, batch.batch_size + 1, sizeof(*batch.pair_offsets),
                          other[1]) ||
      !make_address_range(batch.atomic_numbers, batch.total_atoms, sizeof(*batch.atomic_numbers),
                          other[2]) ||
      !make_address_range(cache.positions, cache.position_elements, sizeof(*cache.positions),
                          other[3]) ||
      !make_address_range(cache.coordination_numbers, cache.coordination_elements,
                          sizeof(*cache.coordination_numbers), other[4]) ||
      !make_address_range(cache.coordination_generations, cache.coordination_generation_elements,
                          sizeof(*cache.coordination_generations), other[5]) ||
      !make_address_range(cache.coordination_eligible_mask, cache.coordination_eligible_elements,
                          sizeof(*cache.coordination_eligible_mask), other[6]) ||
      !make_address_range(geometry_epoch == nullptr ? nullptr : geometry_epoch->value,
                          geometry_epoch == nullptr ? 0 : 1, sizeof(std::uint64_t), other[7])) {
    return false;
  }
  for (std::size_t first = 0; first < structural.size(); ++first) {
    for (std::size_t second = first + 1; second < structural.size(); ++second) {
      if (ranges_overlap(structural[first], structural[second])) {
        return false;
      }
    }
    for (const AddressRange& range : other) {
      if (ranges_overlap(structural[first], range)) {
        return false;
      }
    }
  }
  /* The D4 final cache and its provenance are separate semantic arrays. */
  for (std::size_t first = 3; first < other.size(); ++first) {
    for (std::size_t second = first + 1; second < other.size(); ++second) {
      if (ranges_overlap(other[first], other[second])) {
        return false;
      }
    }
  }
  static_cast<void>(parameters);
  return true;
}

enum class D4PairlistWorkspaceMode {
  kRefresh,
  kFull,
  kSccPotential,
  kSccEnergy,
};

bool valid_pairlist_base_descriptors(const Gfn2D4DeviceBatch& batch,
                                     const Gfn2D4DeviceParameters& parameters,
                                     std::uint64_t scalar_generation,
                                     const Gfn2GeometryEpochDevice* geometry_epoch,
                                     const Gfn2D4PairListDeviceCache& cache,
                                     const Gfn2D4DeviceWorkspace& workspace,
                                     D4PairlistWorkspaceMode mode, std::uint32_t* device_error) {
  const bool reference_square_representable =
      parameters.reference_count > 0 &&
      parameters.reference_count <= kMaximumInt64 / parameters.reference_count;
  if (batch.batch_size <= 0 || batch.batch_size > std::numeric_limits<unsigned int>::max() ||
      batch.total_atoms <= 0 || batch.total_atoms > kMaximumInt64 / 3 || batch.total_pairs < 0 ||
      batch.plan_token == 0u || batch.atom_offsets == nullptr || batch.pair_offsets == nullptr ||
      batch.atomic_numbers == nullptr || parameters.elements == nullptr ||
      parameters.element_count <= 0 || parameters.references == nullptr ||
      parameters.reference_c6 == nullptr || !reference_square_representable ||
      parameters.reference_c6_elements < parameters.reference_count * parameters.reference_count ||
      cache.plan_token != batch.plan_token || cache.positions == nullptr ||
      cache.position_elements != batch.total_atoms * 3 || cache.coordination_numbers == nullptr ||
      cache.coordination_elements != batch.total_atoms ||
      cache.coordination_generations == nullptr ||
      cache.coordination_generation_elements != batch.batch_size ||
      cache.coordination_eligible_mask == nullptr ||
      cache.coordination_eligible_elements != batch.batch_size ||
      !valid_pairlist_role(cache.coordination_pairs, Gfn2PairListRole::kD4Coordination,
                           kGfn2D4CoordinationCutoffBohr, batch) ||
      !valid_pairlist_role(cache.two_body_pairs, Gfn2PairListRole::kD4TwoBody,
                           kGfn2D4TwoBodyCutoffBohr, batch) ||
      !valid_pairlist_role(cache.atm_pairs, Gfn2PairListRole::kD4Atm, kGfn2D4AtmCutoffBohr,
                           batch) ||
      !same_pairlist_storage(cache.coordination_pairs, cache.two_body_pairs) ||
      !same_pairlist_storage(cache.coordination_pairs, cache.atm_pairs) ||
      !((geometry_epoch == nullptr && scalar_generation != 0u) ||
        (geometry_epoch != nullptr && scalar_generation == 0u && geometry_epoch->value != nullptr &&
         geometry_epoch->value_elements == 1 && geometry_epoch->plan_token == batch.plan_token &&
         is_aligned(geometry_epoch->value, alignof(std::uint64_t)))) ||
      workspace.system_errors == nullptr || workspace.system_error_elements < batch.batch_size ||
      device_error == nullptr || !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.pair_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.atomic_numbers, alignof(std::int32_t)) ||
      !is_aligned(parameters.elements, alignof(Gfn2D4DeviceElementData)) ||
      !is_aligned(parameters.references, alignof(Gfn2D4DeviceReferenceData)) ||
      !is_aligned(parameters.reference_c6, alignof(double)) ||
      !is_aligned(cache.positions, alignof(double)) ||
      !is_aligned(cache.coordination_numbers, alignof(double)) ||
      !is_aligned(cache.coordination_generations, alignof(std::uint64_t)) ||
      !is_aligned(workspace.system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t)) ||
      !valid_pairlist_structural_aliases(batch, parameters, cache, geometry_epoch)) {
    return false;
  }
  if (mode == D4PairlistWorkspaceMode::kRefresh) {
    return workspace.coordination_scratch != nullptr &&
           workspace.coordination_scratch_elements >= batch.total_atoms &&
           is_aligned(workspace.coordination_scratch, alignof(double));
  }
  if (batch.total_atoms > kMaximumInt64 / kGfn2D4MaximumReferences) {
    return false;
  }
  const std::int64_t weight_elements = batch.total_atoms * kGfn2D4MaximumReferences;
  const bool weights_valid = workspace.weights != nullptr &&
                             workspace.weight_elements >= weight_elements &&
                             is_aligned(workspace.weights, alignof(double));
  if (mode == D4PairlistWorkspaceMode::kSccPotential) {
    return weights_valid && workspace.weight_charge_derivatives != nullptr &&
           is_aligned(workspace.weight_charge_derivatives, alignof(double)) &&
           workspace.atom_scratch != nullptr && workspace.atom_elements >= batch.total_atoms &&
           is_aligned(workspace.atom_scratch, alignof(double));
  }
  if (mode == D4PairlistWorkspaceMode::kSccEnergy) {
    return weights_valid && workspace.batch_scratch != nullptr &&
           workspace.batch_elements >= batch.batch_size &&
           is_aligned(workspace.batch_scratch, alignof(double));
  }
  return weights_valid && workspace.weight_cn_derivatives != nullptr &&
         workspace.weight_charge_derivatives != nullptr &&
         is_aligned(workspace.weight_cn_derivatives, alignof(double)) &&
         is_aligned(workspace.weight_charge_derivatives, alignof(double)) &&
         workspace.atom_scratch != nullptr && workspace.coordination_adjoints != nullptr &&
         workspace.atom_elements >= batch.total_atoms && workspace.batch_scratch != nullptr &&
         workspace.batch_elements >= batch.batch_size && workspace.gradient_scratch != nullptr &&
         workspace.gradient_elements >= batch.total_atoms * 3 &&
         is_aligned(workspace.atom_scratch, alignof(double)) &&
         is_aligned(workspace.coordination_adjoints, alignof(double)) &&
         is_aligned(workspace.batch_scratch, alignof(double)) &&
         is_aligned(workspace.gradient_scratch, alignof(double));
}

bool valid_pairlist_ranges(const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
                           const Gfn2D4PairListDeviceCache& cache,
                           const Gfn2GeometryEpochDevice* geometry_epoch,
                           const double* numerical_input, std::int64_t numerical_input_elements,
                           const Gfn2SccIterationDeviceActivity* activity, double* first_output,
                           std::int64_t first_output_elements, double* second_output,
                           std::int64_t second_output_elements,
                           const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error,
                           D4PairlistWorkspaceMode mode) {
  const Gfn2PairListConsumerView& view = cache.two_body_pairs;
  std::array<AddressRange, 21> reads;
  std::array<AddressRange, 13> writes;
  const std::int64_t reference_c6_elements =
      parameters.reference_count * parameters.reference_count;
  if (!make_address_range(batch.atom_offsets, batch.batch_size + 1, sizeof(*batch.atom_offsets),
                          reads[0]) ||
      !make_address_range(batch.pair_offsets, batch.batch_size + 1, sizeof(*batch.pair_offsets),
                          reads[1]) ||
      !make_address_range(batch.atomic_numbers, batch.total_atoms, sizeof(*batch.atomic_numbers),
                          reads[2]) ||
      !make_address_range(parameters.elements, parameters.element_count,
                          sizeof(*parameters.elements), reads[3]) ||
      !make_address_range(parameters.references, parameters.reference_count,
                          sizeof(*parameters.references), reads[4]) ||
      !make_address_range(parameters.reference_c6, reference_c6_elements,
                          sizeof(*parameters.reference_c6), reads[5]) ||
      !make_address_range(cache.positions, cache.position_elements, sizeof(*cache.positions),
                          reads[6]) ||
      !make_address_range(cache.coordination_numbers, cache.coordination_elements,
                          sizeof(*cache.coordination_numbers), reads[7]) ||
      !make_address_range(cache.coordination_generations, cache.coordination_generation_elements,
                          sizeof(*cache.coordination_generations), reads[8]) ||
      !make_address_range(cache.coordination_eligible_mask, cache.coordination_eligible_elements,
                          sizeof(*cache.coordination_eligible_mask), reads[9]) ||
      !make_address_range(view.pair_offsets, view.pair_offset_count, sizeof(*view.pair_offsets),
                          reads[10]) ||
      !make_address_range(view.pairs, view.pair_count, sizeof(*view.pairs), reads[11]) ||
      !make_address_range(view.pair_counts, view.pair_count_elements, sizeof(*view.pair_counts),
                          reads[12]) ||
      !make_address_range(view.neighbor_offsets, view.neighbor_offset_count,
                          sizeof(*view.neighbor_offsets), reads[13]) ||
      !make_address_range(view.neighbor_counts, view.neighbor_count_elements,
                          sizeof(*view.neighbor_counts), reads[14]) ||
      !make_address_range(view.neighbors, view.neighbor_count, sizeof(*view.neighbors),
                          reads[15]) ||
      !make_address_range(numerical_input, numerical_input_elements, sizeof(double), reads[16]) ||
      !make_address_range(geometry_epoch == nullptr ? nullptr : geometry_epoch->value,
                          geometry_epoch == nullptr ? 0 : 1, sizeof(std::uint64_t), reads[17]) ||
      !make_address_range(view.committed_generations, view.committed_generation_count,
                          sizeof(*view.committed_generations), reads[18]) ||
      !make_address_range(view.eligible_mask, view.eligible_mask_count, sizeof(*view.eligible_mask),
                          reads[19]) ||
      !make_address_range(view.active_mask, view.active_mask_count, sizeof(*view.active_mask),
                          reads[20])) {
    return false;
  }
  if (activity != nullptr) {
    if (activity->plan_token != batch.plan_token || activity->batch_elements != batch.batch_size ||
        activity->sequence_elements != 1 || activity->active_mask == nullptr ||
        activity->sequence_active == nullptr ||
        !is_aligned(activity->active_mask, alignof(std::uint8_t)) ||
        !is_aligned(activity->sequence_active, alignof(std::uint32_t))) {
      return false;
    }
    AddressRange active_range;
    AddressRange sequence_range;
    if (!make_address_range(activity->active_mask, batch.batch_size, sizeof(*activity->active_mask),
                            active_range) ||
        !make_address_range(activity->sequence_active, 1, sizeof(*activity->sequence_active),
                            sequence_range)) {
      return false;
    }
    for (const AddressRange& read : reads) {
      if (ranges_overlap(active_range, read) || ranges_overlap(sequence_range, read)) {
        return false;
      }
    }
    if (ranges_overlap(active_range, sequence_range)) {
      return false;
    }
  }

  const bool write_weights = mode != D4PairlistWorkspaceMode::kRefresh;
  const bool write_cn_derivatives = mode == D4PairlistWorkspaceMode::kFull;
  const bool write_charge_derivatives =
      mode == D4PairlistWorkspaceMode::kFull || mode == D4PairlistWorkspaceMode::kSccPotential;
  const bool write_atom =
      mode == D4PairlistWorkspaceMode::kFull || mode == D4PairlistWorkspaceMode::kSccPotential;
  const bool write_batch =
      mode == D4PairlistWorkspaceMode::kFull || mode == D4PairlistWorkspaceMode::kSccEnergy;
  const bool write_full = mode == D4PairlistWorkspaceMode::kFull;
  if (!make_address_range(workspace.weights, write_weights ? workspace.weight_elements : 0,
                          sizeof(*workspace.weights), writes[0]) ||
      !make_address_range(workspace.weight_cn_derivatives,
                          write_cn_derivatives ? workspace.weight_elements : 0,
                          sizeof(*workspace.weight_cn_derivatives), writes[1]) ||
      !make_address_range(workspace.weight_charge_derivatives,
                          write_charge_derivatives ? workspace.weight_elements : 0,
                          sizeof(*workspace.weight_charge_derivatives), writes[2]) ||
      !make_address_range(workspace.atom_scratch, write_atom ? workspace.atom_elements : 0,
                          sizeof(*workspace.atom_scratch), writes[3]) ||
      !make_address_range(workspace.coordination_adjoints, write_full ? workspace.atom_elements : 0,
                          sizeof(*workspace.coordination_adjoints), writes[4]) ||
      !make_address_range(workspace.batch_scratch, write_batch ? workspace.batch_elements : 0,
                          sizeof(*workspace.batch_scratch), writes[5]) ||
      !make_address_range(workspace.gradient_scratch, write_full ? workspace.gradient_elements : 0,
                          sizeof(*workspace.gradient_scratch), writes[6]) ||
      !make_address_range(
          workspace.coordination_scratch,
          mode == D4PairlistWorkspaceMode::kRefresh ? workspace.coordination_scratch_elements : 0,
          sizeof(*workspace.coordination_scratch), writes[7]) ||
      !make_address_range(workspace.system_errors, batch.batch_size,
                          sizeof(*workspace.system_errors), writes[8]) ||
      !make_address_range(nullptr, 0, sizeof(std::uint32_t), writes[9]) ||
      !make_address_range(first_output, first_output_elements, sizeof(double), writes[10]) ||
      !make_address_range(second_output, second_output_elements, sizeof(double), writes[11]) ||
      !make_address_range(device_error, 1, sizeof(*device_error), writes[12]) ||
      !writable_ranges_are_disjoint(reads, writes)) {
    return false;
  }
  if (activity != nullptr) {
    AddressRange active_range;
    AddressRange sequence_range;
    make_address_range(activity->active_mask, batch.batch_size, sizeof(*activity->active_mask),
                       active_range);
    make_address_range(activity->sequence_active, 1, sizeof(*activity->sequence_active),
                       sequence_range);
    for (const AddressRange& write : writes) {
      if (ranges_overlap(active_range, write) || ranges_overlap(sequence_range, write)) {
        return false;
      }
    }
  }
  return true;
}

cudaError_t launch_grid(std::int64_t count, unsigned int* blocks) {
  if (count <= 0 || blocks == nullptr) {
    return cudaErrorInvalidValue;
  }
  const std::uint64_t needed =
      (static_cast<std::uint64_t>(count) + kThreadsPerBlock - 1u) / kThreadsPerBlock;
  if (needed > std::numeric_limits<unsigned int>::max()) {
    return cudaErrorInvalidConfiguration;
  }
  *blocks = static_cast<unsigned int>(needed);
  return cudaSuccess;
}

cudaError_t check_launch() { return cudaPeekAtLastError(); }

/*
 * Host-side gate for the D4 ATM multi-block split. The sub-kernels exist so a
 * large single system is not limited to one 256-thread block (one SM) for its
 * O(N^3) triple loop. The split is enabled only when it cannot regress the
 * existing path: every ragged member must be big enough to make the parallelism
 * worth it. Setup records that minimum from the canonical host topology, so the
 * gate never needs to inspect device offsets or synchronize the stream.
 *
 * The returned value is blocks-per-system (>= 1). A value of 1 keeps the
 * historical single-block kernels byte-for-byte for ineligible topologies.
 * When blocks_per_system > 1 the launchers below use the deterministic
 * partial/reduce and atomic split kernels instead. The partial buffer lives in
 * workspace.atom_scratch, which is sized to total_atoms and is otherwise
 * unused by the ATM energy path, so batch_size * blocks_per_system must never
 * exceed atom_elements.
 */
std::int32_t atm_split_blocks_per_system(const Gfn2D4DeviceBatch& batch,
                                         const Gfn2D4DeviceWorkspace& workspace) noexcept {
  if (batch.batch_size < 1 || batch.total_atoms < 1 || workspace.atom_elements < 1 ||
      batch.batch_size > 65535) {
    return 1;
  }
  const std::int64_t minimum_atoms_per_system = batch.minimum_atoms_per_system;
  /* A 62-atom alkane already makes the triple loop the dominant D4 kernel;
   * this threshold keeps every existing small-system path unchanged. */
  constexpr std::int64_t kMinimumAtomsPerSystem = 40;
  /* Once blockIdx.y (system axis) is used, gridDim.y must remain in range;
   * launching more than this many blocks per system adds overhead faster than
   * it recovers parallelism for GFN2-sized molecules. */
  constexpr std::int32_t kMaximumBlocksPerSystem = 64;
  if (minimum_atoms_per_system < kMinimumAtomsPerSystem) {
    return 1;
  }
  /* Cap so the per-system partials fit inside the caller-owned atom scratch
   * (batch_size * blocks_per_system <= total_atoms) and the launch stays in
   * the 2D grid limits. The estimate spreads ~8 atoms per extra block. */
  constexpr std::int32_t kAtomsPerBlock = 8;
  const std::int64_t requested_blocks = minimum_atoms_per_system / kAtomsPerBlock +
                                        (minimum_atoms_per_system % kAtomsPerBlock == 0 ? 0 : 1);
  std::int32_t requested = static_cast<std::int32_t>(
      requested_blocks > kMaximumBlocksPerSystem ? kMaximumBlocksPerSystem : requested_blocks);
  const std::int64_t capacity = workspace.atom_elements / batch.batch_size;
  if (requested > capacity) {
    requested = static_cast<std::int32_t>(capacity);
  }
  if (requested > std::numeric_limits<std::int32_t>::max() / batch.batch_size) {
    requested = 1;
  }
  if (batch.batch_size * requested > std::numeric_limits<std::uint32_t>::max()) {
    requested = 1;
  }
  return requested < 1 ? 1 : requested;
}

cudaError_t validate_pairlist_refresh_impl(const Gfn2D4DeviceBatch& batch,
                                           const Gfn2D4DeviceParameters& parameters,
                                           std::uint64_t scalar_generation,
                                           const Gfn2GeometryEpochDevice* geometry_epoch,
                                           const Gfn2D4PairListDeviceCache& cache,
                                           const Gfn2D4DeviceWorkspace& workspace,
                                           std::uint32_t* device_error) {
  return valid_pairlist_base_descriptors(batch, parameters, scalar_generation, geometry_epoch,
                                         cache, workspace, D4PairlistWorkspaceMode::kRefresh,
                                         device_error) &&
                 valid_pairlist_ranges(batch, parameters, cache, geometry_epoch, nullptr, 0,
                                       nullptr, nullptr, 0, nullptr, 0, workspace, device_error,
                                       D4PairlistWorkspaceMode::kRefresh)
             ? cudaSuccess
             : cudaErrorInvalidValue;
}

cudaError_t validate_pairlist_full_impl(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    std::uint64_t scalar_generation, const Gfn2GeometryEpochDevice* geometry_epoch,
    const Gfn2D4PairListDeviceCache& cache, const double* numerical_input,
    std::int64_t numerical_input_elements, const Gfn2SccIterationDeviceActivity* activity,
    double* first_output, std::int64_t first_output_elements, double* second_output,
    std::int64_t second_output_elements, const Gfn2D4DeviceWorkspace& workspace,
    std::uint32_t* device_error, D4PairlistWorkspaceMode mode = D4PairlistWorkspaceMode::kFull) {
  if ((numerical_input_elements > 0 &&
       (numerical_input == nullptr || !is_aligned(numerical_input, alignof(double)))) ||
      (first_output_elements > 0 &&
       (first_output == nullptr || !is_aligned(first_output, alignof(double)))) ||
      (second_output_elements > 0 &&
       (second_output == nullptr || !is_aligned(second_output, alignof(double))))) {
    return cudaErrorInvalidValue;
  }
  return valid_pairlist_base_descriptors(batch, parameters, scalar_generation, geometry_epoch,
                                         cache, workspace, mode, device_error) &&
                 valid_pairlist_ranges(batch, parameters, cache, geometry_epoch, numerical_input,
                                       numerical_input_elements, activity, first_output,
                                       first_output_elements, second_output, second_output_elements,
                                       workspace, device_error, mode)
             ? cudaSuccess
             : cudaErrorInvalidValue;
}

cudaError_t launch_pairlist_standalone_preflight(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    const Gfn2D4PairListDeviceCache& cache, const Gfn2PairListConsumerView& view,
    D4GenerationSource generation_source, bool require_coordination,
    const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error, cudaStream_t stream) {
  topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, parameters, device_error);
  cudaError_t status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  d4_pairlist_role_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock,
                                      0, stream>>>(batch, cache, view, generation_source,
                                                   require_coordination, nullptr, nullptr,
                                                   workspace, device_error);
  return check_launch();
}

cudaError_t launch_pairlist_scc_preflight(const Gfn2D4DeviceBatch& batch,
                                          const Gfn2D4DeviceParameters& parameters,
                                          const Gfn2D4PairListDeviceCache& cache,
                                          D4GenerationSource generation_source,
                                          const Gfn2SccIterationDeviceActivity& activity,
                                          const Gfn2D4DeviceWorkspace& workspace,
                                          std::uint32_t* device_error, cudaStream_t stream) {
  scc_topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, parameters, activity,
                                                                    device_error);
  cudaError_t status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  d4_pairlist_role_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock,
                                      0, stream>>>(
      batch, cache, cache.two_body_pairs, generation_source, true, activity.active_mask,
      activity.sequence_active, workspace, device_error);
  return check_launch();
}

}  // namespace

cudaError_t reset_gfn2_d4_device_errors_cuda(std::int64_t batch_size, std::uint32_t* system_errors,
                                             std::uint32_t* device_error,
                                             cudaStream_t stream) noexcept {
  AddressRange system_range;
  AddressRange error_range;
  if (batch_size <= 0 || system_errors == nullptr || device_error == nullptr ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t)) ||
      !make_address_range(system_errors, batch_size, sizeof(*system_errors), system_range) ||
      !make_address_range(device_error, 1, sizeof(*device_error), error_range)) {
    return cudaErrorInvalidValue;
  }
  if (ranges_overlap(system_range, error_range)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(
      system_errors, 0, static_cast<std::size_t>(batch_size) * sizeof(*system_errors), stream);
  if (status != cudaSuccess) {
    return status;
  }
  return cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream);
}

cudaError_t update_gfn2_d4_geometry_cache_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    const double* positions, const Gfn2D4DeviceCache& cache, const Gfn2D4DeviceWorkspace& workspace,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (!valid_geometry_refresh_descriptors(batch, parameters, positions, cache, workspace,
                                          device_error)) {
    return cudaErrorInvalidValue;
  }
  topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, parameters, device_error);
  cudaError_t status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  capture_geometry_sequence_kernel<<<1, 1, 0, stream>>>(device_error,
                                                        workspace.geometry_sequence_active);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  const unsigned int blocks = static_cast<unsigned int>(batch.batch_size);
  build_d4_geometry_pairs_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(batch, parameters,
                                                                          positions, workspace);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  build_d4_coordination_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(batch, parameters,
                                                                        workspace);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_d4_geometry_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(batch, cache, workspace);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_d4_geometry_generation_kernel<<<blocks, 1, 0, stream>>>(batch, cache.geometry_generation,
                                                                  workspace);
  return check_launch();
}

cudaError_t evaluate_gfn2_d4_two_body_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    const Gfn2D4DeviceCache& cache, const double* atomic_charges, double* energies,
    double* atomic_potentials, const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  const bool reference_square_representable =
      parameters.reference_count > 0 &&
      parameters.reference_count <=
          std::numeric_limits<std::int64_t>::max() / parameters.reference_count;
  const bool workspace_extents_representable =
      batch.total_atoms > 0 &&
      batch.total_atoms <= std::numeric_limits<std::int64_t>::max() / kGfn2D4MaximumReferences &&
      batch.total_atoms <= std::numeric_limits<std::int64_t>::max() / 3 && batch.total_pairs >= 0 &&
      batch.total_pairs <= std::numeric_limits<std::int64_t>::max() / kGfn2D4PairDataElements;
  const std::int64_t expected_pair_elements =
      workspace_extents_representable ? batch.total_pairs * kGfn2D4PairDataElements : 0;
  const std::int64_t expected_weight_elements =
      workspace_extents_representable ? batch.total_atoms * kGfn2D4MaximumReferences : 0;
  const std::int64_t expected_gradient_elements =
      workspace_extents_representable ? batch.total_atoms * 3 : 0;
  const std::int64_t required_reference_c6_elements =
      reference_square_representable ? parameters.reference_count * parameters.reference_count : 0;
  if (batch.batch_size <= 0 || batch.total_atoms <= 0 || batch.total_pairs < 0 ||
      batch.batch_size > std::numeric_limits<unsigned int>::max() || batch.plan_token == 0u ||
      !workspace_extents_representable || batch.atom_offsets == nullptr ||
      batch.pair_offsets == nullptr || batch.atomic_numbers == nullptr ||
      parameters.elements == nullptr || parameters.element_count <= 0 ||
      parameters.references == nullptr || parameters.reference_c6 == nullptr ||
      !reference_square_representable ||
      parameters.reference_c6_elements < required_reference_c6_elements ||
      cache.plan_token != batch.plan_token || cache.geometry_generation == 0u ||
      (batch.total_pairs != 0 && cache.pair_data == nullptr) ||
      cache.pair_data_elements != expected_pair_elements || cache.coordination_numbers == nullptr ||
      cache.coordination_elements != batch.total_atoms || atomic_charges == nullptr ||
      energies == nullptr || atomic_potentials == nullptr || workspace.weights == nullptr ||
      workspace.weight_cn_derivatives == nullptr ||
      workspace.weight_charge_derivatives == nullptr ||
      workspace.weight_elements < expected_weight_elements || workspace.atom_scratch == nullptr ||
      workspace.coordination_adjoints == nullptr || workspace.atom_elements < batch.total_atoms ||
      workspace.batch_scratch == nullptr || workspace.batch_elements < batch.batch_size ||
      workspace.gradient_scratch == nullptr ||
      workspace.gradient_elements < expected_gradient_elements ||
      workspace.system_errors == nullptr || workspace.system_error_elements < batch.batch_size ||
      device_error == nullptr || !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.pair_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.atomic_numbers, alignof(std::int32_t)) ||
      !is_aligned(parameters.elements, alignof(Gfn2D4DeviceElementData)) ||
      !is_aligned(parameters.references, alignof(Gfn2D4DeviceReferenceData)) ||
      !is_aligned(parameters.reference_c6, alignof(double)) ||
      (cache.pair_data != nullptr && !is_aligned(cache.pair_data, alignof(double))) ||
      !is_aligned(cache.coordination_numbers, alignof(double)) ||
      !is_aligned(atomic_charges, alignof(double)) || !is_aligned(energies, alignof(double)) ||
      !is_aligned(atomic_potentials, alignof(double)) ||
      !is_aligned(workspace.weights, alignof(double)) ||
      !is_aligned(workspace.weight_cn_derivatives, alignof(double)) ||
      !is_aligned(workspace.weight_charge_derivatives, alignof(double)) ||
      !is_aligned(workspace.atom_scratch, alignof(double)) ||
      !is_aligned(workspace.coordination_adjoints, alignof(double)) ||
      !is_aligned(workspace.batch_scratch, alignof(double)) ||
      !is_aligned(workspace.gradient_scratch, alignof(double)) ||
      !is_aligned(workspace.system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }

  std::array<AddressRange, 9> read_ranges;
  std::array<AddressRange, 11> write_ranges;
  if (!make_address_range(batch.atom_offsets, batch.batch_size + 1, sizeof(*batch.atom_offsets),
                          read_ranges[0]) ||
      !make_address_range(batch.pair_offsets, batch.batch_size + 1, sizeof(*batch.pair_offsets),
                          read_ranges[1]) ||
      !make_address_range(batch.atomic_numbers, batch.total_atoms, sizeof(*batch.atomic_numbers),
                          read_ranges[2]) ||
      !make_address_range(parameters.elements, parameters.element_count,
                          sizeof(*parameters.elements), read_ranges[3]) ||
      !make_address_range(parameters.references, parameters.reference_count,
                          sizeof(*parameters.references), read_ranges[4]) ||
      !make_address_range(parameters.reference_c6, parameters.reference_c6_elements,
                          sizeof(*parameters.reference_c6), read_ranges[5]) ||
      !make_address_range(cache.pair_data, expected_pair_elements, sizeof(*cache.pair_data),
                          read_ranges[6]) ||
      !make_address_range(cache.coordination_numbers, batch.total_atoms,
                          sizeof(*cache.coordination_numbers), read_ranges[7]) ||
      !make_address_range(atomic_charges, batch.total_atoms, sizeof(*atomic_charges),
                          read_ranges[8]) ||
      !make_address_range(workspace.weights, workspace.weight_elements, sizeof(*workspace.weights),
                          write_ranges[0]) ||
      !make_address_range(workspace.weight_cn_derivatives, workspace.weight_elements,
                          sizeof(*workspace.weight_cn_derivatives), write_ranges[1]) ||
      !make_address_range(workspace.weight_charge_derivatives, workspace.weight_elements,
                          sizeof(*workspace.weight_charge_derivatives), write_ranges[2]) ||
      !make_address_range(workspace.atom_scratch, workspace.atom_elements,
                          sizeof(*workspace.atom_scratch), write_ranges[3]) ||
      !make_address_range(workspace.coordination_adjoints, workspace.atom_elements,
                          sizeof(*workspace.coordination_adjoints), write_ranges[4]) ||
      !make_address_range(workspace.batch_scratch, workspace.batch_elements,
                          sizeof(*workspace.batch_scratch), write_ranges[5]) ||
      !make_address_range(workspace.gradient_scratch, workspace.gradient_elements,
                          sizeof(*workspace.gradient_scratch), write_ranges[6]) ||
      !make_address_range(workspace.system_errors, workspace.system_error_elements,
                          sizeof(*workspace.system_errors), write_ranges[7]) ||
      !make_address_range(energies, batch.batch_size, sizeof(*energies), write_ranges[8]) ||
      !make_address_range(atomic_potentials, batch.total_atoms, sizeof(*atomic_potentials),
                          write_ranges[9]) ||
      !make_address_range(device_error, 1, sizeof(*device_error), write_ranges[10]) ||
      !writable_ranges_are_disjoint(read_ranges, write_ranges)) {
    return cudaErrorInvalidValue;
  }

  unsigned int atom_blocks = 0;
  unsigned int batch_blocks = 0;
  cudaError_t status = launch_grid(batch.total_atoms, &atom_blocks);
  if (status != cudaSuccess) {
    return status;
  }
  status = launch_grid(batch.batch_size, &batch_blocks);
  if (status != cudaSuccess) {
    return status;
  }

  topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, parameters, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  cache_preflight_kernel<<<atom_blocks, kThreadsPerBlock, 0, stream>>>(batch, cache, workspace,
                                                                       device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  prepare_weights_kernel<<<atom_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, parameters, cache, atomic_charges, false, true, true, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  potential_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0, stream>>>(
      batch, parameters, cache, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  energy_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0, stream>>>(
      batch, parameters, cache, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_atom_kernel<<<atom_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, workspace.atom_scratch, atomic_potentials, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_batch_kernel<<<batch_blocks, kThreadsPerBlock, 0, stream>>>(
      batch.batch_size, workspace.batch_scratch, energies, workspace, device_error);
  return check_launch();
}

cudaError_t evaluate_gfn2_d4_scc_potential_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    const Gfn2D4DeviceCache& cache, std::uint64_t expected_geometry_generation,
    const double* mixed_atomic_charges, const Gfn2SccIterationDeviceActivity& activity,
    double* atomic_potentials, const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  if (!valid_scc_descriptors(batch, parameters, cache, activity, workspace, true, device_error) ||
      mixed_atomic_charges == nullptr || atomic_potentials == nullptr ||
      !is_aligned(mixed_atomic_charges, alignof(double)) ||
      !is_aligned(atomic_potentials, alignof(double)) ||
      !valid_scc_potential_ranges(batch, parameters, cache, mixed_atomic_charges, activity,
                                  atomic_potentials, workspace, device_error)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaSuccess;
  scc_activity_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
      batch, cache, expected_geometry_generation, activity, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  scc_topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, parameters, activity,
                                                                    device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  scc_cache_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                               stream>>>(batch, cache, activity, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  scc_prepare_weights_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                               stream>>>(batch, parameters, cache, mixed_atomic_charges, false,
                                         false, true, activity, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  scc_potential_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                         stream>>>(batch, parameters, cache, activity, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  scc_publish_potential_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                                 stream>>>(batch, activity, workspace.atom_scratch,
                                           atomic_potentials, workspace, device_error);
  return check_launch();
}

cudaError_t evaluate_gfn2_d4_scc_energy_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    const Gfn2D4DeviceCache& cache, std::uint64_t expected_geometry_generation,
    const double* raw_atomic_charges, const Gfn2SccIterationDeviceActivity& activity,
    double* energies, const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  if (!valid_scc_descriptors(batch, parameters, cache, activity, workspace, false, device_error) ||
      raw_atomic_charges == nullptr || energies == nullptr ||
      !is_aligned(raw_atomic_charges, alignof(double)) || !is_aligned(energies, alignof(double)) ||
      !valid_scc_energy_ranges(batch, parameters, cache, raw_atomic_charges, activity, energies,
                               workspace, device_error)) {
    return cudaErrorInvalidValue;
  }
  unsigned int batch_blocks = 0;
  cudaError_t status = launch_grid(batch.batch_size, &batch_blocks);
  if (status != cudaSuccess) {
    return status;
  }
  scc_activity_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
      batch, cache, expected_geometry_generation, activity, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  scc_topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, parameters, activity,
                                                                    device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  scc_cache_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                               stream>>>(batch, cache, activity, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  scc_prepare_weights_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                               stream>>>(batch, parameters, cache, raw_atomic_charges, false, false,
                                         false, activity, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  scc_energy_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0, stream>>>(
      batch, parameters, cache, activity, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  scc_publish_energy_kernel<<<batch_blocks, kThreadsPerBlock, 0, stream>>>(
      batch.batch_size, activity, workspace.batch_scratch, energies, workspace, device_error);
  return check_launch();
}

cudaError_t add_gfn2_d4_two_body_gradient_cuda(const Gfn2D4DeviceBatch& batch,
                                               const Gfn2D4DeviceParameters& parameters,
                                               const Gfn2D4DeviceCache& cache,
                                               const double* atomic_charges, double* gradients,
                                               const Gfn2D4DeviceWorkspace& workspace,
                                               std::uint32_t* device_error,
                                               cudaStream_t stream) noexcept {
  if (!valid_common_descriptors(batch, parameters, cache, workspace, device_error) ||
      atomic_charges == nullptr || gradients == nullptr ||
      !is_aligned(atomic_charges, alignof(double)) || !is_aligned(gradients, alignof(double))) {
    return cudaErrorInvalidValue;
  }
  std::array<AddressRange, 9> reads;
  std::array<AddressRange, 10> writes;
  if (!make_common_ranges(batch, parameters, cache, workspace, reads, writes) ||
      !make_address_range(atomic_charges, batch.total_atoms, sizeof(*atomic_charges), reads[8]) ||
      !make_address_range(gradients, batch.total_atoms * 3, sizeof(*gradients), writes[8]) ||
      !make_address_range(device_error, 1, sizeof(*device_error), writes[9]) ||
      !writable_ranges_are_disjoint(reads, writes)) {
    return cudaErrorInvalidValue;
  }
  unsigned int atom_blocks = 0;
  unsigned int gradient_blocks = 0;
  cudaError_t status = launch_grid(batch.total_atoms, &atom_blocks);
  if (status != cudaSuccess) {
    return status;
  }
  status = launch_grid(batch.total_atoms * 3, &gradient_blocks);
  if (status != cudaSuccess) {
    return status;
  }
  topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, parameters, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  cache_preflight_kernel<<<atom_blocks, kThreadsPerBlock, 0, stream>>>(batch, cache, workspace,
                                                                       device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  gradient_input_preflight_kernel<<<gradient_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, gradients, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  prepare_weights_kernel<<<atom_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, parameters, cache, atomic_charges, false, true, true, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  two_body_cn_adjoint_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                               stream>>>(batch, parameters, cache, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  two_body_gradient_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                             stream>>>(batch, parameters, cache, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  add_output_preflight_kernel<<<gradient_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, batch.total_atoms * 3, 3, workspace.gradient_scratch, gradients, workspace,
      device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  add_publish_atom_kernel<<<gradient_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, batch.total_atoms * 3, 3, workspace.gradient_scratch, gradients, workspace,
      device_error);
  return check_launch();
}

cudaError_t evaluate_gfn2_d4_atm_cuda(const Gfn2D4DeviceBatch& batch,
                                      const Gfn2D4DeviceParameters& parameters,
                                      const Gfn2D4DeviceCache& cache, double* energies,
                                      const Gfn2D4DeviceWorkspace& workspace,
                                      std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (!valid_common_descriptors(batch, parameters, cache, workspace, device_error) ||
      energies == nullptr || !is_aligned(energies, alignof(double))) {
    return cudaErrorInvalidValue;
  }
  std::array<AddressRange, 8> reads;
  std::array<AddressRange, 10> writes;
  if (!make_common_ranges(batch, parameters, cache, workspace, reads, writes) ||
      !make_address_range(energies, batch.batch_size, sizeof(*energies), writes[8]) ||
      !make_address_range(device_error, 1, sizeof(*device_error), writes[9]) ||
      !writable_ranges_are_disjoint(reads, writes)) {
    return cudaErrorInvalidValue;
  }
  unsigned int atom_blocks = 0;
  unsigned int batch_blocks = 0;
  cudaError_t status = launch_grid(batch.total_atoms, &atom_blocks);
  if (status != cudaSuccess) {
    return status;
  }
  status = launch_grid(batch.batch_size, &batch_blocks);
  if (status != cudaSuccess) {
    return status;
  }
  topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, parameters, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  cache_preflight_kernel<<<atom_blocks, kThreadsPerBlock, 0, stream>>>(batch, cache, workspace,
                                                                       device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  prepare_weights_kernel<<<atom_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, parameters, cache, nullptr, true, true, true, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  const std::int32_t blocks_per_system = atm_split_blocks_per_system(batch, workspace);
  if (blocks_per_system > 1) {
    const dim3 grid(static_cast<unsigned int>(blocks_per_system),
                    static_cast<unsigned int>(batch.batch_size));
    atm_energy_split_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(batch, parameters, cache,
                                                                   workspace, device_error);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
    atm_energy_split_reduce_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock,
                                     0, stream>>>(batch, workspace, blocks_per_system, nullptr,
                                                  device_error);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
  } else {
    atm_energy_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0, stream>>>(
        batch, parameters, cache, workspace, device_error);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
  }
  publish_batch_kernel<<<batch_blocks, kThreadsPerBlock, 0, stream>>>(
      batch.batch_size, workspace.batch_scratch, energies, workspace, device_error);
  return check_launch();
}

cudaError_t add_gfn2_d4_atm_gradient_cuda(const Gfn2D4DeviceBatch& batch,
                                          const Gfn2D4DeviceParameters& parameters,
                                          const Gfn2D4DeviceCache& cache, double* gradients,
                                          const Gfn2D4DeviceWorkspace& workspace,
                                          std::uint32_t* device_error,
                                          cudaStream_t stream) noexcept {
  if (!valid_common_descriptors(batch, parameters, cache, workspace, device_error) ||
      gradients == nullptr || !is_aligned(gradients, alignof(double))) {
    return cudaErrorInvalidValue;
  }
  std::array<AddressRange, 8> reads;
  std::array<AddressRange, 10> writes;
  if (!make_common_ranges(batch, parameters, cache, workspace, reads, writes) ||
      !make_address_range(gradients, batch.total_atoms * 3, sizeof(*gradients), writes[8]) ||
      !make_address_range(device_error, 1, sizeof(*device_error), writes[9]) ||
      !writable_ranges_are_disjoint(reads, writes)) {
    return cudaErrorInvalidValue;
  }
  unsigned int atom_blocks = 0;
  unsigned int gradient_blocks = 0;
  cudaError_t status = launch_grid(batch.total_atoms, &atom_blocks);
  if (status != cudaSuccess) {
    return status;
  }
  status = launch_grid(batch.total_atoms * 3, &gradient_blocks);
  if (status != cudaSuccess) {
    return status;
  }
  topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, parameters, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  cache_preflight_kernel<<<atom_blocks, kThreadsPerBlock, 0, stream>>>(batch, cache, workspace,
                                                                       device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  gradient_input_preflight_kernel<<<gradient_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, gradients, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  prepare_weights_kernel<<<atom_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, parameters, cache, nullptr, true, true, true, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  clear_atom_kernel<<<atom_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, batch.total_atoms, 1, workspace.coordination_adjoints, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  clear_atom_kernel<<<gradient_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, batch.total_atoms * 3, 3, workspace.gradient_scratch, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  const std::int32_t blocks_per_system = atm_split_blocks_per_system(batch, workspace);
  if (blocks_per_system > 1) {
    const dim3 grid(static_cast<unsigned int>(blocks_per_system),
                    static_cast<unsigned int>(batch.batch_size));
    atm_gradient_split_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(batch, parameters, cache,
                                                                     workspace, device_error);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
  } else {
    atm_gradient_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                          stream>>>(batch, parameters, cache, workspace, device_error);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
  }
  coordination_vjp_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                            stream>>>(batch, parameters, cache, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  add_output_preflight_kernel<<<gradient_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, batch.total_atoms * 3, 3, workspace.gradient_scratch, gradients, workspace,
      device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  add_publish_atom_kernel<<<gradient_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, batch.total_atoms * 3, 3, workspace.gradient_scratch, gradients, workspace,
      device_error);
  return check_launch();
}

namespace {

cudaError_t update_d4_pairlist_impl(const Gfn2D4DeviceBatch& batch,
                                    const Gfn2D4DeviceParameters& parameters,
                                    std::uint64_t scalar_generation,
                                    const Gfn2GeometryEpochDevice* geometry_epoch,
                                    const Gfn2D4PairListDeviceCache& cache,
                                    const Gfn2D4DeviceWorkspace& workspace,
                                    std::uint32_t* device_error, cudaStream_t stream) {
  cudaError_t status = validate_pairlist_refresh_impl(
      batch, parameters, scalar_generation, geometry_epoch, cache, workspace, device_error);
  if (status != cudaSuccess) {
    return status;
  }
  const D4GenerationSource generation_source{
      scalar_generation, geometry_epoch == nullptr ? nullptr : geometry_epoch->value};
  status = launch_pairlist_standalone_preflight(batch, parameters, cache, cache.coordination_pairs,
                                                generation_source, false, workspace, device_error,
                                                stream);
  if (status != cudaSuccess) {
    return status;
  }
  build_d4_pairlist_coordination_kernel<<<static_cast<unsigned int>(batch.batch_size),
                                          kThreadsPerBlock, 0, stream>>>(
      batch, parameters, cache, generation_source, workspace, device_error);
  return check_launch();
}

cudaError_t evaluate_d4_two_body_pairlist_impl(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    std::uint64_t scalar_generation, const Gfn2GeometryEpochDevice* geometry_epoch,
    const Gfn2D4PairListDeviceCache& cache, const double* atomic_charges, double* energies,
    double* atomic_potentials, const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream) {
  cudaError_t status = validate_pairlist_full_impl(
      batch, parameters, scalar_generation, geometry_epoch, cache, atomic_charges,
      batch.total_atoms, nullptr, energies, batch.batch_size, atomic_potentials, batch.total_atoms,
      workspace, device_error);
  if (status != cudaSuccess) {
    return status;
  }
  const D4GenerationSource source{scalar_generation,
                                  geometry_epoch == nullptr ? nullptr : geometry_epoch->value};
  status = launch_pairlist_standalone_preflight(batch, parameters, cache, cache.two_body_pairs,
                                                source, true, workspace, device_error, stream);
  if (status != cudaSuccess) {
    return status;
  }
  pairlist_prepare_weights_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock,
                                    0, stream>>>(
      batch, parameters, cache.coordination_numbers, cache.coordination_pairs.active_mask,
      atomic_charges, false, true, true, nullptr, nullptr, workspace.weights,
      workspace.weight_cn_derivatives, workspace.weight_charge_derivatives, workspace.system_errors,
      device_error);
  status = check_launch();
  if (status != cudaSuccess) return status;
  pairlist_two_body_potential_kernel<<<static_cast<unsigned int>(batch.batch_size),
                                       kThreadsPerBlock, 0, stream>>>(
      batch, parameters, cache, nullptr, nullptr, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) return status;
  pairlist_two_body_energy_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock,
                                    0, stream>>>(batch, parameters, cache, nullptr, nullptr,
                                                 workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) return status;
  unsigned int atom_blocks = 0;
  unsigned int batch_blocks = 0;
  status = launch_grid(batch.total_atoms, &atom_blocks);
  if (status != cudaSuccess) return status;
  status = launch_grid(batch.batch_size, &batch_blocks);
  if (status != cudaSuccess) return status;
  publish_atom_kernel<<<atom_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, workspace.atom_scratch, atomic_potentials, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) return status;
  publish_batch_kernel<<<batch_blocks, kThreadsPerBlock, 0, stream>>>(
      batch.batch_size, workspace.batch_scratch, energies, workspace, device_error);
  return check_launch();
}

cudaError_t validate_d4_scc_potential_pairlist_impl(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    std::uint64_t scalar_generation, const Gfn2GeometryEpochDevice* geometry_epoch,
    const Gfn2D4PairListDeviceCache& cache, const double* charges,
    const Gfn2SccIterationDeviceActivity& activity, double* potentials,
    const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error) {
  return validate_pairlist_full_impl(batch, parameters, scalar_generation, geometry_epoch, cache,
                                     charges, batch.total_atoms, &activity, potentials,
                                     batch.total_atoms, nullptr, 0, workspace, device_error,
                                     D4PairlistWorkspaceMode::kSccPotential);
}

cudaError_t evaluate_d4_scc_potential_pairlist_impl(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    std::uint64_t scalar_generation, const Gfn2GeometryEpochDevice* geometry_epoch,
    const Gfn2D4PairListDeviceCache& cache, const double* charges,
    const Gfn2SccIterationDeviceActivity& activity, double* potentials,
    const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error, cudaStream_t stream) {
  cudaError_t status = validate_d4_scc_potential_pairlist_impl(
      batch, parameters, scalar_generation, geometry_epoch, cache, charges, activity, potentials,
      workspace, device_error);
  if (status != cudaSuccess) return status;
  const D4GenerationSource source{scalar_generation,
                                  geometry_epoch == nullptr ? nullptr : geometry_epoch->value};
  status = launch_pairlist_scc_preflight(batch, parameters, cache, source, activity, workspace,
                                         device_error, stream);
  if (status != cudaSuccess) return status;
  pairlist_prepare_weights_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock,
                                    0, stream>>>(
      batch, parameters, cache.coordination_numbers, cache.coordination_pairs.active_mask, charges,
      false, false, true, activity.active_mask, activity.sequence_active, workspace.weights,
      workspace.weight_cn_derivatives, workspace.weight_charge_derivatives, workspace.system_errors,
      device_error);
  status = check_launch();
  if (status != cudaSuccess) return status;
  pairlist_two_body_potential_kernel<<<static_cast<unsigned int>(batch.batch_size),
                                       kThreadsPerBlock, 0, stream>>>(
      batch, parameters, cache, activity.active_mask, activity.sequence_active, workspace,
      device_error);
  status = check_launch();
  if (status != cudaSuccess) return status;
  scc_publish_potential_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                                 stream>>>(batch, activity, workspace.atom_scratch, potentials,
                                           workspace, device_error);
  return check_launch();
}

cudaError_t evaluate_d4_scc_energy_pairlist_impl(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    std::uint64_t scalar_generation, const Gfn2GeometryEpochDevice* geometry_epoch,
    const Gfn2D4PairListDeviceCache& cache, const double* charges,
    const Gfn2SccIterationDeviceActivity& activity, double* energies,
    const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error, cudaStream_t stream) {
  cudaError_t status = validate_pairlist_full_impl(
      batch, parameters, scalar_generation, geometry_epoch, cache, charges, batch.total_atoms,
      &activity, energies, batch.batch_size, nullptr, 0, workspace, device_error,
      D4PairlistWorkspaceMode::kSccEnergy);
  if (status != cudaSuccess) return status;
  const D4GenerationSource source{scalar_generation,
                                  geometry_epoch == nullptr ? nullptr : geometry_epoch->value};
  status = launch_pairlist_scc_preflight(batch, parameters, cache, source, activity, workspace,
                                         device_error, stream);
  if (status != cudaSuccess) return status;
  pairlist_prepare_weights_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock,
                                    0, stream>>>(
      batch, parameters, cache.coordination_numbers, cache.coordination_pairs.active_mask, charges,
      false, false, false, activity.active_mask, activity.sequence_active, workspace.weights,
      workspace.weight_cn_derivatives, workspace.weight_charge_derivatives, workspace.system_errors,
      device_error);
  status = check_launch();
  if (status != cudaSuccess) return status;
  pairlist_two_body_energy_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock,
                                    0, stream>>>(batch, parameters, cache, activity.active_mask,
                                                 activity.sequence_active, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) return status;
  unsigned int batch_blocks = 0;
  status = launch_grid(batch.batch_size, &batch_blocks);
  if (status != cudaSuccess) return status;
  scc_publish_energy_kernel<<<batch_blocks, kThreadsPerBlock, 0, stream>>>(
      batch.batch_size, activity, workspace.batch_scratch, energies, workspace, device_error);
  return check_launch();
}

}  // namespace

cudaError_t update_gfn2_d4_pairlist_cache_cuda(const Gfn2D4DeviceBatch& batch,
                                               const Gfn2D4DeviceParameters& parameters,
                                               std::uint64_t generation,
                                               const Gfn2D4PairListDeviceCache& cache,
                                               const Gfn2D4DeviceWorkspace& workspace,
                                               std::uint32_t* device_error,
                                               cudaStream_t stream) noexcept {
  return update_d4_pairlist_impl(batch, parameters, generation, nullptr, cache, workspace,
                                 device_error, stream);
}

cudaError_t update_gfn2_d4_pairlist_cache_cuda(const Gfn2D4DeviceBatch& batch,
                                               const Gfn2D4DeviceParameters& parameters,
                                               const Gfn2GeometryEpochDevice& epoch,
                                               const Gfn2D4PairListDeviceCache& cache,
                                               const Gfn2D4DeviceWorkspace& workspace,
                                               std::uint32_t* device_error,
                                               cudaStream_t stream) noexcept {
  return update_d4_pairlist_impl(batch, parameters, 0u, &epoch, cache, workspace, device_error,
                                 stream);
}

cudaError_t evaluate_gfn2_d4_two_body_pairlist_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    std::uint64_t generation, const Gfn2D4PairListDeviceCache& cache, const double* charges,
    double* energies, double* potentials, const Gfn2D4DeviceWorkspace& workspace,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  return evaluate_d4_two_body_pairlist_impl(batch, parameters, generation, nullptr, cache, charges,
                                            energies, potentials, workspace, device_error, stream);
}

cudaError_t evaluate_gfn2_d4_two_body_pairlist_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    const Gfn2GeometryEpochDevice& epoch, const Gfn2D4PairListDeviceCache& cache,
    const double* charges, double* energies, double* potentials,
    const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  return evaluate_d4_two_body_pairlist_impl(batch, parameters, 0u, &epoch, cache, charges, energies,
                                            potentials, workspace, device_error, stream);
}

cudaError_t validate_gfn2_d4_scc_potential_pairlist_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    std::uint64_t generation, const Gfn2D4PairListDeviceCache& cache, const double* charges,
    const Gfn2SccIterationDeviceActivity& activity, double* potentials,
    const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error) noexcept {
  return validate_d4_scc_potential_pairlist_impl(batch, parameters, generation, nullptr, cache,
                                                 charges, activity, potentials, workspace,
                                                 device_error);
}

cudaError_t validate_gfn2_d4_scc_potential_pairlist_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    const Gfn2GeometryEpochDevice& epoch, const Gfn2D4PairListDeviceCache& cache,
    const double* charges, const Gfn2SccIterationDeviceActivity& activity, double* potentials,
    const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error) noexcept {
  return validate_d4_scc_potential_pairlist_impl(batch, parameters, 0u, &epoch, cache, charges,
                                                 activity, potentials, workspace, device_error);
}

cudaError_t evaluate_gfn2_d4_scc_potential_pairlist_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    std::uint64_t generation, const Gfn2D4PairListDeviceCache& cache, const double* charges,
    const Gfn2SccIterationDeviceActivity& activity, double* potentials,
    const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  return evaluate_d4_scc_potential_pairlist_impl(batch, parameters, generation, nullptr, cache,
                                                 charges, activity, potentials, workspace,
                                                 device_error, stream);
}

cudaError_t evaluate_gfn2_d4_scc_potential_pairlist_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    const Gfn2GeometryEpochDevice& epoch, const Gfn2D4PairListDeviceCache& cache,
    const double* charges, const Gfn2SccIterationDeviceActivity& activity, double* potentials,
    const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  return evaluate_d4_scc_potential_pairlist_impl(batch, parameters, 0u, &epoch, cache, charges,
                                                 activity, potentials, workspace, device_error,
                                                 stream);
}

cudaError_t evaluate_gfn2_d4_scc_energy_pairlist_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    std::uint64_t generation, const Gfn2D4PairListDeviceCache& cache, const double* charges,
    const Gfn2SccIterationDeviceActivity& activity, double* energies,
    const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  return evaluate_d4_scc_energy_pairlist_impl(batch, parameters, generation, nullptr, cache,
                                              charges, activity, energies, workspace, device_error,
                                              stream);
}

cudaError_t evaluate_gfn2_d4_scc_energy_pairlist_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    const Gfn2GeometryEpochDevice& epoch, const Gfn2D4PairListDeviceCache& cache,
    const double* charges, const Gfn2SccIterationDeviceActivity& activity, double* energies,
    const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  return evaluate_d4_scc_energy_pairlist_impl(batch, parameters, 0u, &epoch, cache, charges,
                                              activity, energies, workspace, device_error, stream);
}

namespace {

cudaError_t add_d4_two_body_gradient_pairlist_impl(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    std::uint64_t scalar_generation, const Gfn2GeometryEpochDevice* geometry_epoch,
    const Gfn2D4PairListDeviceCache& cache, const double* charges, double* gradients,
    const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error, cudaStream_t stream) {
  cudaError_t status = validate_pairlist_full_impl(
      batch, parameters, scalar_generation, geometry_epoch, cache, charges, batch.total_atoms,
      nullptr, gradients, batch.total_atoms * 3, nullptr, 0, workspace, device_error);
  if (status != cudaSuccess) return status;
  const D4GenerationSource source{scalar_generation,
                                  geometry_epoch == nullptr ? nullptr : geometry_epoch->value};
  status = launch_pairlist_standalone_preflight(batch, parameters, cache, cache.two_body_pairs,
                                                source, true, workspace, device_error, stream);
  if (status != cudaSuccess) return status;
  unsigned int gradient_blocks = 0;
  status = launch_grid(batch.total_atoms * 3, &gradient_blocks);
  if (status != cudaSuccess) return status;
  gradient_input_preflight_kernel<<<gradient_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, gradients, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) return status;
  pairlist_prepare_weights_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock,
                                    0, stream>>>(
      batch, parameters, cache.coordination_numbers, cache.coordination_pairs.active_mask, charges,
      false, true, true, nullptr, nullptr, workspace.weights, workspace.weight_cn_derivatives,
      workspace.weight_charge_derivatives, workspace.system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) return status;
  pairlist_two_body_cn_adjoint_kernel<<<static_cast<unsigned int>(batch.batch_size),
                                        kThreadsPerBlock, 0, stream>>>(batch, parameters, cache,
                                                                       workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) return status;
  pairlist_two_body_gradient_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock,
                                      0, stream>>>(batch, parameters, cache, workspace,
                                                   device_error);
  status = check_launch();
  if (status != cudaSuccess) return status;
  add_output_preflight_kernel<<<gradient_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, batch.total_atoms * 3, 3, workspace.gradient_scratch, gradients, workspace,
      device_error);
  status = check_launch();
  if (status != cudaSuccess) return status;
  add_publish_atom_kernel<<<gradient_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, batch.total_atoms * 3, 3, workspace.gradient_scratch, gradients, workspace,
      device_error);
  return check_launch();
}

cudaError_t validate_d4_atm_pairlist_impl(const Gfn2D4DeviceBatch& batch,
                                          const Gfn2D4DeviceParameters& parameters,
                                          std::uint64_t scalar_generation,
                                          const Gfn2GeometryEpochDevice* geometry_epoch,
                                          const Gfn2D4PairListDeviceCache& cache, double* energies,
                                          const Gfn2D4DeviceWorkspace& workspace,
                                          std::uint32_t* device_error) {
  return validate_pairlist_full_impl(batch, parameters, scalar_generation, geometry_epoch, cache,
                                     nullptr, 0, nullptr, energies, batch.batch_size, nullptr, 0,
                                     workspace, device_error);
}

cudaError_t evaluate_d4_atm_pairlist_impl(const Gfn2D4DeviceBatch& batch,
                                          const Gfn2D4DeviceParameters& parameters,
                                          std::uint64_t scalar_generation,
                                          const Gfn2GeometryEpochDevice* geometry_epoch,
                                          const Gfn2D4PairListDeviceCache& cache, double* energies,
                                          const Gfn2D4DeviceWorkspace& workspace,
                                          std::uint32_t* device_error, cudaStream_t stream) {
  cudaError_t status =
      validate_d4_atm_pairlist_impl(batch, parameters, scalar_generation, geometry_epoch, cache,
                                    energies, workspace, device_error);
  if (status != cudaSuccess) return status;
  const D4GenerationSource source{scalar_generation,
                                  geometry_epoch == nullptr ? nullptr : geometry_epoch->value};
  status = launch_pairlist_standalone_preflight(batch, parameters, cache, cache.atm_pairs, source,
                                                true, workspace, device_error, stream);
  if (status != cudaSuccess) return status;
  pairlist_prepare_weights_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock,
                                    0, stream>>>(
      batch, parameters, cache.coordination_numbers, cache.coordination_pairs.active_mask, nullptr,
      true, true, true, nullptr, nullptr, workspace.weights, workspace.weight_cn_derivatives,
      workspace.weight_charge_derivatives, workspace.system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) return status;
  const std::int32_t blocks_per_system = atm_split_blocks_per_system(batch, workspace);
  if (blocks_per_system > 1) {
    const dim3 grid(static_cast<unsigned int>(blocks_per_system),
                    static_cast<unsigned int>(batch.batch_size));
    pairlist_atm_energy_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
        batch, parameters, cache, workspace, blocks_per_system, device_error);
    status = check_launch();
    if (status != cudaSuccess) return status;
    atm_energy_split_reduce_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock,
                                     0, stream>>>(batch, workspace, blocks_per_system,
                                                  cache.atm_pairs.active_mask, device_error);
    status = check_launch();
    if (status != cudaSuccess) return status;
  } else {
    pairlist_atm_energy_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                                 stream>>>(batch, parameters, cache, workspace, 1, device_error);
    status = check_launch();
    if (status != cudaSuccess) return status;
  }
  unsigned int batch_blocks = 0;
  status = launch_grid(batch.batch_size, &batch_blocks);
  if (status != cudaSuccess) return status;
  publish_pairlist_batch_kernel<<<batch_blocks, kThreadsPerBlock, 0, stream>>>(
      batch.batch_size, cache.atm_pairs.active_mask, workspace.batch_scratch, energies, workspace,
      device_error);
  return check_launch();
}

cudaError_t add_d4_atm_gradient_pairlist_impl(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    std::uint64_t scalar_generation, const Gfn2GeometryEpochDevice* geometry_epoch,
    const Gfn2D4PairListDeviceCache& cache, double* gradients,
    const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error, cudaStream_t stream) {
  cudaError_t status = validate_pairlist_full_impl(
      batch, parameters, scalar_generation, geometry_epoch, cache, nullptr, 0, nullptr, gradients,
      batch.total_atoms * 3, nullptr, 0, workspace, device_error);
  if (status != cudaSuccess) return status;
  const D4GenerationSource source{scalar_generation,
                                  geometry_epoch == nullptr ? nullptr : geometry_epoch->value};
  status = launch_pairlist_standalone_preflight(batch, parameters, cache, cache.atm_pairs, source,
                                                true, workspace, device_error, stream);
  if (status != cudaSuccess) return status;
  unsigned int atom_blocks = 0;
  unsigned int gradient_blocks = 0;
  status = launch_grid(batch.total_atoms, &atom_blocks);
  if (status != cudaSuccess) return status;
  status = launch_grid(batch.total_atoms * 3, &gradient_blocks);
  if (status != cudaSuccess) return status;
  gradient_input_preflight_kernel<<<gradient_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, gradients, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) return status;
  pairlist_prepare_weights_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock,
                                    0, stream>>>(
      batch, parameters, cache.coordination_numbers, cache.coordination_pairs.active_mask, nullptr,
      true, true, true, nullptr, nullptr, workspace.weights, workspace.weight_cn_derivatives,
      workspace.weight_charge_derivatives, workspace.system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) return status;
  clear_atom_kernel<<<atom_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, batch.total_atoms, 1, workspace.coordination_adjoints, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) return status;
  clear_atom_kernel<<<gradient_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, batch.total_atoms * 3, 3, workspace.gradient_scratch, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) return status;
  const std::int32_t blocks_per_system = atm_split_blocks_per_system(batch, workspace);
  if (blocks_per_system > 1) {
    const dim3 grid(static_cast<unsigned int>(blocks_per_system),
                    static_cast<unsigned int>(batch.batch_size));
    pairlist_atm_gradient_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
        batch, parameters, cache, workspace, blocks_per_system, device_error);
    status = check_launch();
    if (status != cudaSuccess) return status;
  } else {
    pairlist_atm_gradient_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                                   stream>>>(batch, parameters, cache, workspace, 1, device_error);
    status = check_launch();
    if (status != cudaSuccess) return status;
  }
  pairlist_d4_coordination_vjp_kernel<<<static_cast<unsigned int>(batch.batch_size),
                                        kThreadsPerBlock, 0, stream>>>(batch, parameters, cache,
                                                                       workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) return status;
  add_output_preflight_kernel<<<gradient_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, batch.total_atoms * 3, 3, workspace.gradient_scratch, gradients, workspace,
      device_error);
  status = check_launch();
  if (status != cudaSuccess) return status;
  add_publish_atom_kernel<<<gradient_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, batch.total_atoms * 3, 3, workspace.gradient_scratch, gradients, workspace,
      device_error);
  return check_launch();
}

}  // namespace

cudaError_t add_gfn2_d4_two_body_gradient_pairlist_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    std::uint64_t generation, const Gfn2D4PairListDeviceCache& cache, const double* charges,
    double* gradients, const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  return add_d4_two_body_gradient_pairlist_impl(batch, parameters, generation, nullptr, cache,
                                                charges, gradients, workspace, device_error,
                                                stream);
}

cudaError_t add_gfn2_d4_two_body_gradient_pairlist_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    const Gfn2GeometryEpochDevice& epoch, const Gfn2D4PairListDeviceCache& cache,
    const double* charges, double* gradients, const Gfn2D4DeviceWorkspace& workspace,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  return add_d4_two_body_gradient_pairlist_impl(batch, parameters, 0u, &epoch, cache, charges,
                                                gradients, workspace, device_error, stream);
}

cudaError_t validate_gfn2_d4_atm_pairlist_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    std::uint64_t generation, const Gfn2D4PairListDeviceCache& cache, double* energies,
    const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error) noexcept {
  return validate_d4_atm_pairlist_impl(batch, parameters, generation, nullptr, cache, energies,
                                       workspace, device_error);
}

cudaError_t validate_gfn2_d4_atm_pairlist_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    const Gfn2GeometryEpochDevice& epoch, const Gfn2D4PairListDeviceCache& cache, double* energies,
    const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error) noexcept {
  return validate_d4_atm_pairlist_impl(batch, parameters, 0u, &epoch, cache, energies, workspace,
                                       device_error);
}

cudaError_t evaluate_gfn2_d4_atm_pairlist_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    std::uint64_t generation, const Gfn2D4PairListDeviceCache& cache, double* energies,
    const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  return evaluate_d4_atm_pairlist_impl(batch, parameters, generation, nullptr, cache, energies,
                                       workspace, device_error, stream);
}

cudaError_t evaluate_gfn2_d4_atm_pairlist_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    const Gfn2GeometryEpochDevice& epoch, const Gfn2D4PairListDeviceCache& cache, double* energies,
    const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  return evaluate_d4_atm_pairlist_impl(batch, parameters, 0u, &epoch, cache, energies, workspace,
                                       device_error, stream);
}

cudaError_t add_gfn2_d4_atm_gradient_pairlist_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    std::uint64_t generation, const Gfn2D4PairListDeviceCache& cache, double* gradients,
    const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  return add_d4_atm_gradient_pairlist_impl(batch, parameters, generation, nullptr, cache, gradients,
                                           workspace, device_error, stream);
}

cudaError_t add_gfn2_d4_atm_gradient_pairlist_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    const Gfn2GeometryEpochDevice& epoch, const Gfn2D4PairListDeviceCache& cache, double* gradients,
    const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  return add_d4_atm_gradient_pairlist_impl(batch, parameters, 0u, &epoch, cache, gradients,
                                           workspace, device_error, stream);
}

#if defined(XTBLOOM_CUDA_TEST_HOOKS)
std::int32_t test_gfn2_d4_atm_split_blocks_per_system(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceWorkspace& workspace) noexcept {
  return atm_split_blocks_per_system(batch, workspace);
}

cudaError_t test_gfn2_d4_atm_reduction_cuda(const Gfn2D4DeviceBatch& batch,
                                            const double* finite_values, double* energies,
                                            const Gfn2D4DeviceWorkspace& workspace,
                                            std::uint32_t* device_error,
                                            cudaStream_t stream) noexcept {
  if (batch.batch_size <= 0 || batch.total_atoms <= 0 ||
      batch.batch_size > std::numeric_limits<unsigned int>::max() ||
      batch.atom_offsets == nullptr || finite_values == nullptr || energies == nullptr ||
      workspace.batch_scratch == nullptr || workspace.batch_elements < batch.batch_size ||
      workspace.system_errors == nullptr || workspace.system_error_elements < batch.batch_size ||
      device_error == nullptr || !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(finite_values, alignof(double)) || !is_aligned(energies, alignof(double)) ||
      !is_aligned(workspace.batch_scratch, alignof(double)) ||
      !is_aligned(workspace.system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  std::array<AddressRange, 2> reads;
  std::array<AddressRange, 4> writes;
  if (!make_address_range(batch.atom_offsets, batch.batch_size + 1, sizeof(*batch.atom_offsets),
                          reads[0]) ||
      !make_address_range(finite_values, batch.total_atoms, sizeof(*finite_values), reads[1]) ||
      !make_address_range(workspace.batch_scratch, workspace.batch_elements,
                          sizeof(*workspace.batch_scratch), writes[0]) ||
      !make_address_range(workspace.system_errors, workspace.system_error_elements,
                          sizeof(*workspace.system_errors), writes[1]) ||
      !make_address_range(energies, batch.batch_size, sizeof(*energies), writes[2]) ||
      !make_address_range(device_error, 1, sizeof(*device_error), writes[3]) ||
      !writable_ranges_are_disjoint(reads, writes)) {
    return cudaErrorInvalidValue;
  }
  unsigned int batch_blocks = 0;
  cudaError_t status = launch_grid(batch.batch_size, &batch_blocks);
  if (status != cudaSuccess) {
    return status;
  }
  atm_reduction_test_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                              stream>>>(batch, finite_values, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_batch_kernel<<<batch_blocks, kThreadsPerBlock, 0, stream>>>(
      batch.batch_size, workspace.batch_scratch, energies, workspace, device_error);
  return check_launch();
}

cudaError_t test_gfn2_d4_atm_addition_cuda(const Gfn2D4DeviceBatch& batch,
                                           const double* finite_deltas, double* gradients,
                                           const Gfn2D4DeviceWorkspace& workspace,
                                           std::uint32_t* device_error,
                                           cudaStream_t stream) noexcept {
  if (batch.batch_size <= 0 || batch.total_atoms <= 0 ||
      batch.total_atoms > std::numeric_limits<std::int64_t>::max() / 3 ||
      finite_deltas == nullptr || gradients == nullptr || workspace.system_errors == nullptr ||
      workspace.system_error_elements < batch.batch_size || device_error == nullptr ||
      !is_aligned(finite_deltas, alignof(double)) || !is_aligned(gradients, alignof(double)) ||
      !is_aligned(workspace.system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  const std::int64_t gradient_elements = batch.total_atoms * 3;
  std::array<AddressRange, 1> reads;
  std::array<AddressRange, 3> writes;
  if (!make_address_range(finite_deltas, gradient_elements, sizeof(*finite_deltas), reads[0]) ||
      !make_address_range(gradients, gradient_elements, sizeof(*gradients), writes[0]) ||
      !make_address_range(workspace.system_errors, workspace.system_error_elements,
                          sizeof(*workspace.system_errors), writes[1]) ||
      !make_address_range(device_error, 1, sizeof(*device_error), writes[2]) ||
      !writable_ranges_are_disjoint(reads, writes)) {
    return cudaErrorInvalidValue;
  }
  unsigned int gradient_blocks = 0;
  cudaError_t status = launch_grid(gradient_elements, &gradient_blocks);
  if (status != cudaSuccess) {
    return status;
  }
  add_output_preflight_kernel<<<gradient_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, gradient_elements, 3, finite_deltas, gradients, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  add_publish_atom_kernel<<<gradient_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, gradient_elements, 3, finite_deltas, gradients, workspace, device_error);
  return check_launch();
}
#endif

}  // namespace xtbloom::detail::cuda
