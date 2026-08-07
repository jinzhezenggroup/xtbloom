#include <array>
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_pairlist.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr std::int64_t kInt64Maximum = 9223372036854775807LL;
constexpr double kMinimumDistanceSquared = 1.0e-12;

/* Logical bucket edge equals the sparse cutoff, so every contacting pair lies
 * in the same or an immediately neighboring bucket in each axis. */
constexpr double kDefaultCutoffBohr = 25.0;
/* Default dense fallback threshold, anchored by the batch 1/8/32/128 and
 * atoms-per-system sweep in benchmarks/evidence/issue-70: the dense all-pairs
 * fallback wins or ties up to about 32 atoms, the measured crossover sits
 * between 32 and 48 atoms, and the bucketed build clearly wins from 48 atoms.
 * The dense path is therefore retained at or below 40 atoms and the sparse
 * bucketed path is used above it. */
constexpr std::int64_t kSparseCrossoverAtoms = 40;

struct SystemRanges {
  std::int64_t atom_begin;
  std::int64_t atom_end;
  std::int64_t cell_base;
  std::int64_t cells;
};

/* Per-system strategy.  A non-null system_modes array overrides the batch-wide
 * `mode`, letting heterogeneous batches dispatch dense and bucketed peers
 * independently.  The two paths emit identical canonical lists. */
__device__ Gfn2PairListMode system_mode(const Gfn2PairListDeviceBatch& batch, std::int64_t system) {
  if (batch.system_modes != nullptr) {
    return static_cast<Gfn2PairListMode>(batch.system_modes[system]);
  }
  return batch.mode;
}

__device__ bool valid_system_mode_value(std::int32_t value) {
  return value == static_cast<std::int32_t>(Gfn2PairListMode::kSparse) ||
         value == static_cast<std::int32_t>(Gfn2PairListMode::kDense);
}

__device__ bool finite_position(const double* positions, std::int64_t coordinate) {
  return isfinite(positions[coordinate]) && isfinite(positions[coordinate + 1]) &&
         isfinite(positions[coordinate + 2]);
}

__device__ bool sequence_is_active(const std::uint32_t* sequence_active) {
  return atomicAdd(const_cast<std::uint32_t*>(sequence_active), 0u) == 1u;
}

/* int64_t is long on this LP64 platform; the CUDA intrinsic requires long long. */
__device__ std::int64_t atomic_add_int64(std::int64_t* address, std::int64_t value) {
  return atomicAdd(reinterpret_cast<unsigned long long*>(address),
                   static_cast<unsigned long long>(value));
}

__device__ bool system_is_valid(const std::uint32_t* system_errors, std::int64_t system) {
  return atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) ==
         static_cast<std::uint32_t>(Gfn2PairListDeviceError::kSuccess);
}

__device__ void record_error(std::uint32_t* device_error, Gfn2PairListDeviceError error) {
  atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2PairListDeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    std::uint32_t* device_error, Gfn2PairListDeviceError error) {
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  if (atomicCAS(system_errors + system,
                static_cast<std::uint32_t>(Gfn2PairListDeviceError::kSuccess),
                code) == static_cast<std::uint32_t>(Gfn2PairListDeviceError::kSuccess)) {
    atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2PairListDeviceError::kSuccess), code);
  }
}

/* Stable logistic form of 1/(1+exp(-argument)), matching the CPU reference. */
__device__ double logistic(double argument) {
  if (argument >= 0.0) {
    const double exponential = exp(-argument);
    return 1.0 / (1.0 + exponential);
  }
  const double exponential = exp(argument);
  return exponential / (1.0 + exponential);
}

/*
 * One pair's shared physical values used by the coordination consumer.  The
 * pair cache stores indices only; consumers recompute the distance quantities
 * from positions, so this module keeps one canonical evaluation that exactly
 * mirrors the reference (CPU and dense geometry) pair evaluation.
 */
struct PairValues {
  double distance;
  double inverse_distance;
  double count;
  double derivative_over_distance;
};

__device__ bool evaluate_pair(double dx, double dy, double dz, double radius, PairValues* values) {
  /* A non-positive/non-finite radius is invalid input, not a zero-weight pair:
   * silently accepting it would publish a successful but physically undefined
   * coordination result. */
  if (!isfinite(radius) || !(radius > 0.0)) {
    return false;
  }
  const double distance_squared = dx * dx + dy * dy + dz * dz;
  if (!isfinite(distance_squared) || distance_squared < kMinimumDistanceSquared) {
    return false;
  }
  values->distance = sqrt(distance_squared);
  values->inverse_distance = 1.0 / values->distance;
  values->count = 0.0;
  values->derivative_over_distance = 0.0;
  if (!(values->distance > 0.0) || !isfinite(values->distance) ||
      !(values->inverse_distance > 0.0) || !isfinite(values->inverse_distance)) {
    return false;
  }
  constexpr double kCutoffSquaredBohr = kDefaultCutoffBohr * kDefaultCutoffBohr;
  if (isfinite(radius) && radius > 0.0 && distance_squared <= kCutoffSquaredBohr) {
    constexpr double kFirstSteepness = 10.0;
    constexpr double kSecondSteepness = 20.0;
    constexpr double kSecondRadiusShiftBohr = 2.0;
    const double inverse_distance_squared = values->inverse_distance * values->inverse_distance;
    const double shifted_radius = radius + kSecondRadiusShiftBohr;
    const double first = logistic(kFirstSteepness * (radius * values->inverse_distance - 1.0));
    const double second =
        logistic(kSecondSteepness * (shifted_radius * values->inverse_distance - 1.0));
    values->count = first * second;
    const double derivative = -inverse_distance_squared *
                              (kFirstSteepness * radius * first * (1.0 - first) * second +
                               kSecondSteepness * shifted_radius * second * (1.0 - second) * first);
    values->derivative_over_distance = derivative * values->inverse_distance;
    return values->count >= 0.0 && values->count <= 1.0 && isfinite(values->count) &&
           isfinite(values->derivative_over_distance);
  }
  return values->distance > 0.0 && isfinite(values->distance) && isfinite(values->inverse_distance);
}

/* Validate all offset endpoints before any later kernel subtracts them. */
__global__ void topology_preflight_kernel(Gfn2PairListDeviceBatch batch,
                                          std::uint32_t* sequence_active,
                                          std::uint32_t* device_error,
                                          bool reject_preexisting_error) {
  /* Peer-local failures must not make the rest of the batch a no-op. */
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *sequence_active = (!reject_preexisting_error ||
                        atomicAdd(device_error, 0u) ==
                            static_cast<std::uint32_t>(Gfn2PairListDeviceError::kSuccess))
                           ? 1u
                           : 0u;
  }
  __syncthreads();
  if (atomicAdd(const_cast<std::uint32_t*>(sequence_active), 0u) == 0u) {
    return;
  }
  if (threadIdx.x == 0 &&
      (batch.atom_offsets[0] != 0 || batch.atom_offsets[batch.batch_size] != batch.total_atoms)) {
    atomicExch(sequence_active, 0u);
    record_error(device_error, Gfn2PairListDeviceError::kInvalidOffsets);
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    const std::int64_t begin = batch.atom_offsets[system];
    const std::int64_t end = batch.atom_offsets[system + 1];
    const bool valid = begin >= 0 && begin <= end && end <= batch.total_atoms;
    if (!valid) {
      atomicExch(sequence_active, 0u);
      record_error(device_error, Gfn2PairListDeviceError::kInvalidOffsets);
    }
    if (batch.system_modes != nullptr && !valid_system_mode_value(batch.system_modes[system])) {
      atomicExch(sequence_active, 0u);
      record_error(device_error, Gfn2PairListDeviceError::kInvalidMode);
    }
  }
}

__device__ bool load_system(const Gfn2PairListDeviceBatch& batch, std::int64_t system,
                            const std::uint32_t* sequence_active,
                            const std::uint32_t* system_errors, SystemRanges* ranges, int* valid) {
  if (threadIdx.x == 0) {
    *valid = sequence_is_active(sequence_active) && system_is_valid(system_errors, system) ? 1 : 0;
    if (*valid != 0) {
      ranges->atom_begin = batch.atom_offsets[system];
      ranges->atom_end = batch.atom_offsets[system + 1];
      ranges->cell_base = system * (batch.max_cells_per_system + 1);
      ranges->cells = 1;
    }
  }
  __syncthreads();
  return *valid != 0;
}

__device__ std::int64_t flat_cell(const Gfn2PairListSystemMeta& meta, std::int64_t cx,
                                  std::int64_t cy, std::int64_t cz) {
  return (cz * meta.ny + cy) * meta.nx + cx;
}

/*
 * Compute the uniform bucket grid for one system, zero its cell counts and
 * neighbor cursors, assign every atom to a bucket, and increment the per-bucket
 * counts.  Bucket edge equals the sparse cutoff, so any pair within the cutoff
 * lies in the same or an immediately neighboring bucket.  Empty and
 * single-atom systems still own exactly one bucket and publish an empty pair
 * list.
 */
__global__ void build_buckets_kernel(Gfn2PairListDeviceBatch batch, const double* positions,
                                     Gfn2PairListSystemMeta* system_meta, std::int64_t* atom_cells,
                                     std::int64_t* cell_counts,
                                     const std::uint32_t* sequence_active,
                                     std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  if (!load_system(batch, system, sequence_active, system_errors, &ranges, &valid)) {
    return;
  }
  __syncthreads();

  const std::int64_t atom_count = ranges.atom_end - ranges.atom_begin;
  Gfn2PairListSystemMeta meta;
  meta.cell_base = ranges.cell_base;
  meta.nx = 1;
  meta.ny = 1;
  meta.nz = 1;
  meta.cells = (system_mode(batch, system) == Gfn2PairListMode::kDense) ? 0 : 1;
  for (int axis = 0; axis < 3; ++axis) {
    meta.origin[axis] = 0.0;
  }
  /* Empty and failed systems still publish the single-bucket default meta so
   * later stages never read a stale value from a previous inference.  A
   * per-system dense-dispatch peer publishes cells==0 so build_neighbors runs
   * the deterministic all-pairs scan for that peer regardless of the batch
   * strategy; the two paths emit the same canonical lists. */
  if (threadIdx.x == 0) {
    system_meta[system] = meta;
  }
  __syncthreads();
  if (meta.cells == 0) {
    return;
  }

  if (atom_count > 0 && threadIdx.x == 0) {
    const double edge = batch.cutoff;
    double min_value[3] = {0.0, 0.0, 0.0};
    double max_value[3] = {0.0, 0.0, 0.0};
    const std::int64_t first_index = ranges.atom_begin;
    for (int axis = 0; axis < 3; ++axis) {
      min_value[axis] = positions[first_index * 3 + axis];
      max_value[axis] = positions[first_index * 3 + axis];
    }
    bool all_finite = true;
    for (std::int64_t atom = ranges.atom_begin; atom < ranges.atom_end; ++atom) {
      if (!finite_position(positions, atom * 3)) {
        all_finite = false;
        break;
      }
      for (int axis = 0; axis < 3; ++axis) {
        const double value = positions[atom * 3 + axis];
        min_value[axis] = fmin(min_value[axis], value);
        max_value[axis] = fmax(max_value[axis], value);
      }
    }
    if (!all_finite) {
      record_system_error(system_errors, system, device_error,
                          Gfn2PairListDeviceError::kNonfinitePosition);
      valid = 0;
    }
    if (valid != 0) {
      bool dense_fallback = false;
      for (int axis = 0; axis < 3; ++axis) {
        const double extent = max_value[axis] - min_value[axis];
        meta.origin[axis] = min_value[axis];
        const double bucket_count = extent / edge;
        if (!isfinite(bucket_count) || bucket_count > static_cast<double>(kInt64Maximum - 1)) {
          dense_fallback = true;
          break;
        }
        const std::int64_t buckets =
            extent <= 0.0 ? 1 : static_cast<std::int64_t>(bucket_count) + 1;
        if (axis == 0) {
          meta.nx = buckets;
        } else if (axis == 1) {
          meta.ny = buckets;
        } else {
          meta.nz = buckets;
        }
      }
      if (!dense_fallback &&
          (meta.nx > kInt64Maximum / meta.ny || (meta.nx * meta.ny) > kInt64Maximum / meta.nz)) {
        dense_fallback = true;
      }
      if (!dense_fallback) {
        const std::int64_t cells = meta.nx * meta.ny * meta.nz;
        if (cells > batch.max_cells_per_system) {
          dense_fallback = true;
        } else {
          meta.cells = cells;
          system_meta[system] = meta;
        }
      }
      if (dense_fallback) {
        if ((batch.flags & kGfn2PairListAllowDenseFallback) != 0u) {
          meta.cells = 0;
          system_meta[system] = meta;
        } else {
          record_system_error(system_errors, system, device_error,
                              Gfn2PairListDeviceError::kCellCapacityExceeded);
          valid = 0;
        }
      }
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  const Gfn2PairListSystemMeta settled = system_meta[system];
  if (settled.cells == 0) {
    return;
  }
  for (std::int64_t cell = threadIdx.x; cell < settled.cells; cell += blockDim.x) {
    cell_counts[settled.cell_base + cell] = 0;
  }
  __syncthreads();
  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    if (!finite_position(positions, coordinate)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2PairListDeviceError::kNonfinitePosition);
      atomicExch(&valid, 0);
      continue;
    }
    std::int64_t cx =
        static_cast<std::int64_t>((positions[coordinate] - settled.origin[0]) / batch.cutoff);
    std::int64_t cy =
        static_cast<std::int64_t>((positions[coordinate + 1] - settled.origin[1]) / batch.cutoff);
    std::int64_t cz =
        static_cast<std::int64_t>((positions[coordinate + 2] - settled.origin[2]) / batch.cutoff);
    cx = cx < 0 ? 0 : (cx >= settled.nx ? settled.nx - 1 : cx);
    cy = cy < 0 ? 0 : (cy >= settled.ny ? settled.ny - 1 : cy);
    cz = cz < 0 ? 0 : (cz >= settled.nz ? settled.nz - 1 : cz);
    const std::int64_t cell = flat_cell(settled, cx, cy, cz);
    atom_cells[atom] = cell;
    atomic_add_int64(cell_counts + settled.cell_base + cell, 1LL);
  }
}

/* Zero the per-atom neighbor append cursors before enumeration.  Runs in both
 * modes so the dense fallback never depends on bucket construction. */
__global__ void zero_neighbor_cursors_kernel(Gfn2PairListDeviceBatch batch,
                                             std::int64_t* neighbor_cursor,
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
  if (valid == 0) {
    return;
  }
  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    neighbor_cursor[atom] = 0;
  }
}

/*
 * Prefix the per-system cell counts into cell_offsets (exclusive, in the
 * global cell_atoms domain) and seed cell_fill with the same values so the
 * scatter step owns one slot per atom.  cell_offsets[cell_base + cells] is the
 * exclusive end of the system's atom range.
 */
__global__ void prefix_cells_kernel(Gfn2PairListDeviceBatch batch,
                                    const Gfn2PairListSystemMeta* system_meta,
                                    const std::int64_t* cell_counts, std::int64_t* cell_offsets,
                                    std::int64_t* cell_fill, const std::uint32_t* sequence_active,
                                    std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  if (!load_system(batch, system, sequence_active, system_errors, &ranges, &valid)) {
    return;
  }
  __syncthreads();
  if (valid == 0 || threadIdx.x != 0) {
    return;
  }
  const Gfn2PairListSystemMeta meta = system_meta[system];
  std::int64_t running = ranges.atom_begin;
  for (std::int64_t cell = meta.cell_base; cell < meta.cell_base + meta.cells; ++cell) {
    cell_offsets[cell] = running;
    cell_fill[cell] = running;
    running += cell_counts[cell];
  }
  cell_offsets[meta.cell_base + meta.cells] = running;
}

/* Scatter atoms into per-bucket lists (cell_atoms) using the seeded cursors. */
__global__ void scatter_atoms_kernel(Gfn2PairListDeviceBatch batch,
                                     const Gfn2PairListSystemMeta* system_meta,
                                     const std::int64_t* atom_cells, std::int64_t* cell_fill,
                                     std::int64_t* cell_atoms, const std::uint32_t* sequence_active,
                                     std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  if (!load_system(batch, system, sequence_active, system_errors, &ranges, &valid)) {
    return;
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  const Gfn2PairListSystemMeta meta = system_meta[system];
  if (meta.cells == 0) {
    return;
  }
  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    const std::int64_t cell = atom_cells[atom];
    const std::int64_t slot = atomic_add_int64(cell_fill + meta.cell_base + cell, 1LL);
    cell_atoms[slot] = atom;
  }
}

/*
 * Enumerate retained pairs per atom.  Every retained unordered pair is
 * discovered exactly once from its smaller endpoint; both endpoints record the
 * peer in their own neighbor list so the published per-atom ranges are
 * complete in both directions.  kSparse sweeps the 3x3x3 bucket neighborhood
 * within the cutoff; kDense scans the full triangle (deterministic fallback).
 */
__global__ void build_neighbors_kernel(
    Gfn2PairListDeviceBatch batch, const double* positions,
    const Gfn2PairListSystemMeta* system_meta, const std::int64_t* atom_cells,
    const std::int64_t* cell_offsets, const std::int64_t* cell_atoms, std::int64_t* neighbor_cursor,
    std::int64_t* neighbor_scratch, const std::uint32_t* sequence_active,
    std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  if (!load_system(batch, system, sequence_active, system_errors, &ranges, &valid)) {
    return;
  }
  __syncthreads();

  const bool mode_sparse = system_mode(batch, system) == Gfn2PairListMode::kSparse;
  Gfn2PairListSystemMeta meta{};
  if (mode_sparse) {
    meta = system_meta[system];
  }
  if (mode_sparse && meta.cells > 0) {
    /* Bucketed sweep: the per-system meta is only valid here because the
     * sparse launch runs the bucket construction before this kernel. */
    const double cutoff_squared = batch.cutoff * batch.cutoff;
    for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
         atom += blockDim.x) {
      const std::int64_t coordinate = atom * 3;
      const std::int64_t cell = atom_cells[atom];
      const std::int64_t cx = cell % meta.nx;
      const std::int64_t cy = (cell / meta.nx) % meta.ny;
      const std::int64_t cz = cell / (meta.nx * meta.ny);
      for (std::int64_t dz = -1; dz <= 1; ++dz) {
        const std::int64_t czz = cz + dz;
        if (czz < 0 || czz >= meta.nz) {
          continue;
        }
        for (std::int64_t dy = -1; dy <= 1; ++dy) {
          const std::int64_t cyy = cy + dy;
          if (cyy < 0 || cyy >= meta.ny) {
            continue;
          }
          for (std::int64_t dx = -1; dx <= 1; ++dx) {
            const std::int64_t cxx = cx + dx;
            if (cxx < 0 || cxx >= meta.nx) {
              continue;
            }
            const std::int64_t neighbor_cell = flat_cell(meta, cxx, cyy, czz);
            const std::int64_t begin = cell_offsets[meta.cell_base + neighbor_cell];
            const std::int64_t end = cell_offsets[meta.cell_base + neighbor_cell + 1];
            for (std::int64_t slot = begin; slot < end; ++slot) {
              const std::int64_t peer = cell_atoms[slot];
              if (peer <= atom) {
                continue;
              }
              const std::int64_t peer_coordinate = peer * 3;
              const double dx = positions[peer_coordinate] - positions[coordinate];
              const double dy = positions[peer_coordinate + 1] - positions[coordinate + 1];
              const double dz = positions[peer_coordinate + 2] - positions[coordinate + 2];
              const double distance_squared = dx * dx + dy * dy + dz * dz;
              if (!isfinite(distance_squared)) {
                record_system_error(system_errors, system, device_error,
                                    Gfn2PairListDeviceError::kNonfiniteArithmetic);
                atomicExch(&valid, 0);
                return;
              }
              if (distance_squared > cutoff_squared) {
                continue;
              }
              if (distance_squared < kMinimumDistanceSquared) {
                record_system_error(system_errors, system, device_error,
                                    Gfn2PairListDeviceError::kCoincidentAtoms);
                atomicExch(&valid, 0);
                return;
              }
              const std::int64_t slot_own = atomic_add_int64(neighbor_cursor + atom, 1LL);
              if (slot_own >= batch.max_neighbors_per_atom) {
                record_system_error(system_errors, system, device_error,
                                    Gfn2PairListDeviceError::kNeighborCapacityExceeded);
                atomicExch(&valid, 0);
                return;
              }
              neighbor_scratch[atom * batch.max_neighbors_per_atom + slot_own] = peer;
              const std::int64_t slot_peer = atomic_add_int64(neighbor_cursor + peer, 1LL);
              if (slot_peer >= batch.max_neighbors_per_atom) {
                record_system_error(system_errors, system, device_error,
                                    Gfn2PairListDeviceError::kNeighborCapacityExceeded);
                atomicExch(&valid, 0);
                return;
              }
              neighbor_scratch[peer * batch.max_neighbors_per_atom + slot_peer] = atom;
            }
          }
        }
      }
    }
    return;
  }

  /* Dense fallback: no bucket state is consulted.  Explicit kDense retains
   * the full triangle; an opted-in sparse overflow scans the same triangle but
   * still retains only cutoff pairs. */
  const double cutoff_squared = batch.cutoff * batch.cutoff;
  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    for (std::int64_t peer = atom + 1; peer < ranges.atom_end; ++peer) {
      const std::int64_t peer_coordinate = peer * 3;
      const double dx = positions[peer_coordinate] - positions[coordinate];
      const double dy = positions[peer_coordinate + 1] - positions[coordinate + 1];
      const double dz = positions[peer_coordinate + 2] - positions[coordinate + 2];
      const double distance_squared = dx * dx + dy * dy + dz * dz;
      if (!isfinite(distance_squared)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2PairListDeviceError::kNonfiniteArithmetic);
        atomicExch(&valid, 0);
        return;
      }
      if (mode_sparse && distance_squared > cutoff_squared) {
        continue;
      }
      if (distance_squared < kMinimumDistanceSquared) {
        record_system_error(system_errors, system, device_error,
                            Gfn2PairListDeviceError::kCoincidentAtoms);
        atomicExch(&valid, 0);
        return;
      }
      const std::int64_t slot_own = atomic_add_int64(neighbor_cursor + atom, 1LL);
      if (slot_own >= batch.max_neighbors_per_atom) {
        record_system_error(system_errors, system, device_error,
                            Gfn2PairListDeviceError::kNeighborCapacityExceeded);
        atomicExch(&valid, 0);
        return;
      }
      neighbor_scratch[atom * batch.max_neighbors_per_atom + slot_own] = peer;
      const std::int64_t slot_peer = atomic_add_int64(neighbor_cursor + peer, 1LL);
      if (slot_peer >= batch.max_neighbors_per_atom) {
        record_system_error(system_errors, system, device_error,
                            Gfn2PairListDeviceError::kNeighborCapacityExceeded);
        atomicExch(&valid, 0);
        return;
      }
      neighbor_scratch[peer * batch.max_neighbors_per_atom + slot_peer] = atom;
    }
  }
}

/* Sort each atom's neighbor scratch range ascending (small insertion sort). */
__global__ void sort_neighbors_kernel(Gfn2PairListDeviceBatch batch,
                                      const std::int64_t* neighbor_cursor,
                                      std::int64_t* neighbor_scratch,
                                      const std::uint32_t* sequence_active,
                                      std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  if (!load_system(batch, system, sequence_active, system_errors, &ranges, &valid)) {
    return;
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    const std::int64_t count = neighbor_cursor[atom];
    std::int64_t* const slice = neighbor_scratch + atom * batch.max_neighbors_per_atom;
    for (std::int64_t i = 1; i < count; ++i) {
      const std::int64_t key = slice[i];
      std::int64_t j = i - 1;
      while (j >= 0 && slice[j] > key) {
        slice[j + 1] = slice[j];
        --j;
      }
      slice[j + 1] = key;
    }
  }
}

/*
 * Per-system scan: count the retained pairs (total neighbor records / 2) and
 * store either the count or zero (on capacity failure) into pair_cursor.
 * Invalid systems always record zero so the pair-offset prefix stays correct.
 */
__global__ void count_pairs_kernel(Gfn2PairListDeviceBatch batch,
                                   const std::int64_t* neighbor_cursor, std::int64_t* pair_cursor,
                                   const std::uint32_t* sequence_active,
                                   std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  std::int64_t pair_count = 0;
  if (threadIdx.x == 0) {
    valid = sequence_is_active(sequence_active) && system_is_valid(system_errors, system) ? 1 : 0;
    if (valid != 0) {
      ranges.atom_begin = batch.atom_offsets[system];
      ranges.atom_end = batch.atom_offsets[system + 1];
    }
  }
  __syncthreads();
  if (valid != 0) {
    for (std::int64_t atom = ranges.atom_begin; atom < ranges.atom_end; ++atom) {
      pair_count += neighbor_cursor[atom];
    }
    pair_count /= 2;
    if (pair_count > batch.max_pairs_per_system) {
      record_system_error(system_errors, system, device_error,
                          Gfn2PairListDeviceError::kPairCapacityExceeded);
      pair_count = 0;
    }
  }
  if (threadIdx.x == 0) {
    pair_cursor[system] = pair_count;
  }
}

/*
 * Global exclusive prefix of the per-system pair counts into the public
 * pair_offsets.  Failed peers contribute a zero-length range and invalidate
 * their prior generation; the batch-wide CSR arrays are therefore repacked,
 * while publish_kernel only writes pair bytes for healthy peers.  The whole
 * call is a no-op when the sequence is inactive, so caller outputs are never
 * touched before publication.
 */
__global__ void prefix_pair_offsets_kernel(Gfn2PairListDeviceBatch batch,
                                           const std::int64_t* pair_cursor,
                                           Gfn2PairListDeviceCache cache,
                                           const std::uint32_t* sequence_active,
                                           std::uint32_t* system_errors,
                                           std::uint32_t* device_error) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (sequence_is_active(sequence_active)) {
      std::int64_t running = 0;
      for (std::int64_t system = 0; system < batch.batch_size; ++system) {
        cache.pair_offsets[system] = running;
        const std::int64_t count = system_is_valid(system_errors, system) ? pair_cursor[system] : 0;
        cache.pair_counts[system] = count;
        running += count;
        if (!system_is_valid(system_errors, system)) {
          cache.pair_generations[system] = 0u;
        }
      }
      cache.pair_offsets[batch.batch_size] = running;
    }
  }
}

/*
 * Global exclusive prefix of the per-atom neighbor counts into the public
 * neighbor_offsets, publishing the explicit per-atom neighbor_counts.  Failed
 * systems contribute a zero-length range so healthy peers keep contiguous,
 * correct slices.  The whole call is a no-op when the sequence is inactive, so
 * caller outputs are never touched before publication.
 */
__global__ void prefix_neighbor_offsets_kernel(Gfn2PairListDeviceBatch batch,
                                               const std::int64_t* neighbor_cursor,
                                               Gfn2PairListDeviceCache cache,
                                               const std::uint32_t* sequence_active,
                                               std::uint32_t* system_errors,
                                               std::uint32_t* device_error) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (sequence_is_active(sequence_active)) {
      std::int64_t running = 0;
      for (std::int64_t atom = 0; atom < batch.total_atoms; ++atom) {
        cache.neighbor_offsets[atom] = running;
        std::int64_t system = 0;
        while (system + 1 < batch.batch_size && atom >= batch.atom_offsets[system + 1]) {
          ++system;
        }
        const std::int64_t count =
            system_is_valid(system_errors, system) ? neighbor_cursor[atom] : 0;
        cache.neighbor_counts[atom] = count;
        running += count;
      }
      cache.neighbor_offsets[batch.total_atoms] = running;
    }
  }
}

/*
 * Stable publish pass.  Copies the sorted neighbor lists into the public
 * cache, emits the pair list in canonical (first < second, second ascending)
 * order from the sorted neighbor ranges of each atom, and commits the
 * per-system generation.  Only valid systems write public bytes.
 */
__global__ void publish_kernel(Gfn2PairListDeviceBatch batch, std::uint64_t pair_generation,
                               const std::int64_t* neighbor_cursor,
                               const std::int64_t* neighbor_scratch, Gfn2PairListDeviceCache cache,
                               const std::uint32_t* sequence_active, std::uint32_t* system_errors,
                               std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  if (!load_system(batch, system, sequence_active, system_errors, &ranges, &valid)) {
    return;
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  /* Copy the sorted neighbor lists. */
  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    const std::int64_t count = neighbor_cursor[atom];
    const std::int64_t base = cache.neighbor_offsets[atom];
    for (std::int64_t index = 0; index < count; ++index) {
      cache.neighbors[base + index] = neighbor_scratch[atom * batch.max_neighbors_per_atom + index];
    }
  }
  /* Emit pairs from the larger endpoint, in ascending neighbor order, into the
   * canonical pair stream.  This mirrors the dense packed-triangle order
   * (second ascending, first ascending within a second) so a consumer can
   * compare the sparse list index-by-index with the dense cache on the
   * retained subset.  Serial per system for deterministic order. */
  if (threadIdx.x == 0) {
    std::int64_t cursor = cache.pair_offsets[system];
    for (std::int64_t atom = ranges.atom_begin; atom < ranges.atom_end; ++atom) {
      const std::int64_t count = neighbor_cursor[atom];
      const std::int64_t* const slice = neighbor_scratch + atom * batch.max_neighbors_per_atom;
      for (std::int64_t index = 0; index < count; ++index) {
        const std::int64_t peer = slice[index];
        if (peer < atom) {
          Gfn2AtomPair pair;
          pair.first = peer;
          pair.second = atom;
          cache.pairs[cursor++] = pair;
        }
      }
    }
    cache.pair_generations[system] = pair_generation;
  }
}

/*
 * Coordination consumer over the published all-direction neighbor ranges.
 * Neighbor lists are canonical (ascending) so the per-atom accumulation order
 * matches the dense geometry cache exactly for retained pairs.
 */
__global__ void evaluate_coordination_kernel(
    Gfn2PairListDeviceBatch batch, const double* positions, const double* covalent_radii,
    std::uint64_t scalar_generation, const Gfn2PairListDeviceCache& cache, double* coordination,
    const std::uint32_t* sequence_active, std::uint32_t* system_errors,
    std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  if (!load_system(batch, system, sequence_active, system_errors, &ranges, &valid)) {
    return;
  }
  __syncthreads();
  if (threadIdx.x == 0 && cache.pair_generations[system] != scalar_generation) {
    record_system_error(system_errors, system, device_error,
                        Gfn2PairListDeviceError::kStaleGeometry);
    valid = 0;
  }
  __syncthreads();
  /* Validate all atom-local inputs before any thread publishes coordination.
   * This keeps an invalid radius (including one on an otherwise isolated atom)
   * peer-local and prevents a partial system slice from escaping. */
  if (valid != 0) {
    for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
         atom += blockDim.x) {
      if (!finite_position(positions, atom * 3) || !isfinite(covalent_radii[atom]) ||
          !(covalent_radii[atom] > 0.0)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2PairListDeviceError::kInvalidCache);
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
    const std::int64_t begin = cache.neighbor_offsets[atom];
    const std::int64_t end = cache.neighbor_offsets[atom + 1];
    double cn = 0.0;
    bool finite_result = true;
    const std::int64_t coordinate = atom * 3;
    for (std::int64_t index = begin; index < end; ++index) {
      const std::int64_t peer = cache.neighbors[index];
      const std::int64_t peer_coordinate = peer * 3;
      const double dx = positions[peer_coordinate] - positions[coordinate];
      const double dy = positions[peer_coordinate + 1] - positions[coordinate + 1];
      const double dz = positions[peer_coordinate + 2] - positions[coordinate + 2];
      PairValues values{};
      if (!finite_position(positions, coordinate) || !finite_position(positions, peer_coordinate) ||
          !evaluate_pair(dx, dy, dz, covalent_radii[atom] + covalent_radii[peer], &values)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2PairListDeviceError::kInvalidCache);
        finite_result = false;
        break;
      }
      cn += values.count;
      finite_result = finite_result && isfinite(cn);
      if (!finite_result) {
        record_system_error(system_errors, system, device_error,
                            Gfn2PairListDeviceError::kNonfiniteArithmetic);
        break;
      }
    }
    if (finite_result) {
      coordination[atom] = cn;
    }
  }
}

/*
 * Validate every cached pair before the consumer writes any coordination
 * output.  The cache is normally produced from the same positions/radii, but
 * callers may update device inputs between launches.  A pair-level failure
 * must therefore be discovered in a read-only pass; otherwise another thread
 * could publish a partial slice for the failed peer before this consumer sees
 * the bad distance or radius sum.
 */
__global__ void preflight_coordination_pairs_kernel(
    Gfn2PairListDeviceBatch batch, const double* positions, const double* covalent_radii,
    std::uint64_t scalar_generation, const Gfn2PairListDeviceCache& cache,
    const std::uint32_t* sequence_active, std::uint32_t* system_errors,
    std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  if (!load_system(batch, system, sequence_active, system_errors, &ranges, &valid)) {
    return;
  }
  __syncthreads();
  if (threadIdx.x == 0 && cache.pair_generations[system] != scalar_generation) {
    record_system_error(system_errors, system, device_error,
                        Gfn2PairListDeviceError::kStaleGeometry);
    valid = 0;
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  const std::int64_t neighbor_capacity = batch.total_atoms * batch.max_neighbors_per_atom;
  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    if (!finite_position(positions, coordinate) || !isfinite(covalent_radii[atom]) ||
        !(covalent_radii[atom] > 0.0)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2PairListDeviceError::kInvalidCache);
      continue;
    }
    const std::int64_t begin = cache.neighbor_offsets[atom];
    const std::int64_t end = cache.neighbor_offsets[atom + 1];
    const std::int64_t count = cache.neighbor_counts[atom];
    if (begin < 0 || end < begin || end > neighbor_capacity || count < 0 ||
        count > batch.max_neighbors_per_atom || end - begin != count) {
      record_system_error(system_errors, system, device_error,
                          Gfn2PairListDeviceError::kInvalidCache);
      continue;
    }
    for (std::int64_t index = begin; index < end; ++index) {
      const std::int64_t peer = cache.neighbors[index];
      if (peer < ranges.atom_begin || peer >= ranges.atom_end) {
        record_system_error(system_errors, system, device_error,
                            Gfn2PairListDeviceError::kInvalidCache);
        break;
      }
      const std::int64_t peer_coordinate = peer * 3;
      const double dx = positions[peer_coordinate] - positions[coordinate];
      const double dy = positions[peer_coordinate + 1] - positions[coordinate + 1];
      const double dz = positions[peer_coordinate + 2] - positions[coordinate + 2];
      PairValues values{};
      if (!finite_position(positions, peer_coordinate) ||
          !evaluate_pair(dx, dy, dz, covalent_radii[atom] + covalent_radii[peer], &values)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2PairListDeviceError::kInvalidCache);
        break;
      }
    }
  }
}

/*
 * Sparse coordination VJP: accumulate gradients += (dCN/dR)^T * dE_dcn over
 * the published sparse neighbor ranges.  The neighbor list is canonical
 * (ascending) so the per-atom accumulation order matches the dense geometry
 * VJP for retained pairs; beyond-cutoff dense pairs carry exact-zero
 * derivatives, so sparse and dense results agree bitwise.  A failed peer
 * publishes nothing and retains its input gradients.
 */
__global__ void coordination_vjp_preflight_kernel(
    Gfn2PairListDeviceBatch batch, const double* positions, const double* covalent_radii,
    std::uint64_t scalar_generation, Gfn2PairListDeviceCache cache, const double* dE_dcn,
    const double* gradients, const std::uint32_t* sequence_active, std::uint32_t* system_errors,
    std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  if (!load_system(batch, system, sequence_active, system_errors, &ranges, &valid)) {
    return;
  }
  __syncthreads();
  if (threadIdx.x == 0 && cache.pair_generations[system] != scalar_generation) {
    record_system_error(system_errors, system, device_error,
                        Gfn2PairListDeviceError::kStaleGeometry);
    valid = 0;
  }
  __syncthreads();
  if (valid != 0) {
    for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
         atom += blockDim.x) {
      const std::int64_t coordinate = atom * 3;
      if (!finite_position(positions, coordinate) || !isfinite(covalent_radii[atom]) ||
          !(covalent_radii[atom] > 0.0) || !isfinite(dE_dcn[atom]) ||
          !isfinite(gradients[coordinate]) || !isfinite(gradients[coordinate + 1]) ||
          !isfinite(gradients[coordinate + 2])) {
        record_system_error(system_errors, system, device_error,
                            Gfn2PairListDeviceError::kInvalidCache);
        atomicExch(&valid, 0);
      }
    }
  }
  __syncthreads();
  if (valid != 0) {
    const std::int64_t neighbor_capacity = batch.total_atoms * batch.max_neighbors_per_atom;
    for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
         atom += blockDim.x) {
      const std::int64_t begin = cache.neighbor_offsets[atom];
      const std::int64_t end = cache.neighbor_offsets[atom + 1];
      const std::int64_t count = cache.neighbor_counts[atom];
      if (begin < 0 || end < begin || end > neighbor_capacity || count < 0 ||
          count > batch.max_neighbors_per_atom || end - begin != count) {
        record_system_error(system_errors, system, device_error,
                            Gfn2PairListDeviceError::kInvalidCache);
        atomicExch(&valid, 0);
        continue;
      }
      const std::int64_t coordinate = atom * 3;
      for (std::int64_t index = begin; index < end; ++index) {
        const std::int64_t peer = cache.neighbors[index];
        if (peer < ranges.atom_begin || peer >= ranges.atom_end) {
          record_system_error(system_errors, system, device_error,
                              Gfn2PairListDeviceError::kInvalidCache);
          atomicExch(&valid, 0);
          break;
        }
        const double radius = covalent_radii[atom] + covalent_radii[peer];
        const std::int64_t target_is_upper = atom > peer;
        const std::int64_t upper = target_is_upper ? atom : peer;
        const std::int64_t lower = target_is_upper ? peer : atom;
        const double displacement[3] = {positions[upper * 3] - positions[lower * 3],
                                        positions[upper * 3 + 1] - positions[lower * 3 + 1],
                                        positions[upper * 3 + 2] - positions[lower * 3 + 2]};
        PairValues values{};
        if (!finite_position(positions, coordinate) || !finite_position(positions, peer * 3) ||
            !evaluate_pair(displacement[0], displacement[1], displacement[2], radius, &values)) {
          record_system_error(system_errors, system, device_error,
                              Gfn2PairListDeviceError::kInvalidCache);
          atomicExch(&valid, 0);
          break;
        }
      }
    }
  }
  __syncthreads();
  static_cast<void>(valid);
}

__global__ void coordination_vjp_accumulate_kernel(
    Gfn2PairListDeviceBatch batch, const double* positions, const double* covalent_radii,
    std::uint64_t scalar_generation, Gfn2PairListDeviceCache cache, const double* dE_dcn,
    const double* gradients, double* gradient_scratch, const std::uint32_t* sequence_active,
    std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  if (!load_system(batch, system, sequence_active, system_errors, &ranges, &valid)) {
    return;
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    double contribution[3] = {0.0, 0.0, 0.0};
    bool finite_result = true;
    const std::int64_t begin = cache.neighbor_offsets[atom];
    const std::int64_t end = cache.neighbor_offsets[atom + 1];
    for (std::int64_t index = begin; finite_result && index < end; ++index) {
      const std::int64_t peer = cache.neighbors[index];
      const std::int64_t target_is_upper = atom > peer;
      const std::int64_t upper = target_is_upper ? atom : peer;
      const std::int64_t lower = target_is_upper ? peer : atom;
      const double displacement[3] = {positions[upper * 3] - positions[lower * 3],
                                      positions[upper * 3 + 1] - positions[lower * 3 + 1],
                                      positions[upper * 3 + 2] - positions[lower * 3 + 2]};
      PairValues values{};
      const double radius = covalent_radii[atom] + covalent_radii[peer];
      if (!evaluate_pair(displacement[0], displacement[1], displacement[2], radius, &values)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2PairListDeviceError::kInvalidCache);
        finite_result = false;
        break;
      }
      const double scale = (dE_dcn[upper] + dE_dcn[lower]) * values.derivative_over_distance;
      const double sign = target_is_upper ? 1.0 : -1.0;
      for (int axis = 0; axis < 3; ++axis) {
        contribution[axis] += sign * scale * displacement[axis];
        finite_result = finite_result && isfinite(contribution[axis]);
      }
    }
    const std::int64_t coordinate = atom * 3;
    for (int axis = 0; axis < 3; ++axis) {
      const double updated = gradients[coordinate + axis] + contribution[axis];
      if (!finite_result || !isfinite(updated)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2PairListDeviceError::kNonfiniteArithmetic);
        finite_result = false;
      } else {
        gradient_scratch[coordinate + axis] = updated;
      }
    }
  }
}

__global__ void coordination_vjp_publish_kernel(Gfn2PairListDeviceBatch batch,
                                                const double* gradient_scratch, double* gradients,
                                                const std::uint32_t* sequence_active,
                                                std::uint32_t* system_errors,
                                                std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  if (!load_system(batch, system, sequence_active, system_errors, &ranges, &valid)) {
    return;
  }
  const std::int64_t atom_begin = ranges.atom_begin;
  const std::int64_t atom_end = ranges.atom_end;
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

template <std::size_t ReadCount, std::size_t WriteCount>
bool writable_ranges_are_disjoint(const std::array<AddressRange, ReadCount>& reads,
                                  const std::array<AddressRange, WriteCount>& writes) noexcept {
  for (std::size_t write = 0u; write < WriteCount; ++write) {
    for (const AddressRange& read : reads) {
      if (read.begin < writes[write].end && writes[write].begin < read.end) {
        return false;
      }
    }
    for (std::size_t other = write + 1u; other < WriteCount; ++other) {
      if (writes[write].begin < writes[other].end && writes[other].begin < writes[write].end) {
        return false;
      }
    }
  }
  return true;
}

cudaError_t validate_batch(const Gfn2PairListDeviceBatch& batch) noexcept {
  const bool per_system_dispatch = batch.system_modes != nullptr;
  if (batch.batch_size <= 0 || batch.total_atoms < 0 ||
      batch.batch_size == std::numeric_limits<std::int64_t>::max() ||
      batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      batch.total_atoms > kInt64Maximum / 3 || batch.atom_offset_elements != batch.batch_size + 1 ||
      batch.max_cells_per_system <= 0 || batch.max_cells_per_system == kInt64Maximum ||
      batch.max_neighbors_per_atom <= 0 || batch.max_pairs_per_system <= 0 ||
      !(batch.cutoff > 0.0) || !isfinite(batch.cutoff) ||
      (batch.mode != Gfn2PairListMode::kSparse && batch.mode != Gfn2PairListMode::kDense) ||
      (batch.flags & ~kGfn2PairListAllowDenseFallback) != 0u || batch.plan_token == 0u ||
      !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      (!per_system_dispatch && batch.system_mode_elements != 0) ||
      (per_system_dispatch && batch.system_mode_elements != batch.batch_size) ||
      (per_system_dispatch && !is_aligned(batch.system_modes, alignof(std::int32_t)))) {
    return batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max())
               ? cudaErrorInvalidConfiguration
               : cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t validate_update(const Gfn2PairListDeviceBatch& batch, const double* positions,
                            std::uint64_t pair_generation, const Gfn2PairListDeviceCache& cache,
                            const Gfn2PairListDeviceWorkspace& workspace,
                            std::uint32_t* system_errors, std::uint32_t* device_error) noexcept {
  cudaError_t status = validate_batch(batch);
  if (status != cudaSuccess) {
    return status;
  }
  if (pair_generation == 0u || !required_pointer(positions, batch.total_atoms * 3) ||
      cache.plan_token != batch.plan_token || workspace.plan_token != batch.plan_token ||
      batch.batch_size > kInt64Maximum / batch.max_pairs_per_system ||
      cache.pair_elements < batch.max_pairs_per_system * batch.batch_size ||
      cache.pair_offset_elements != batch.batch_size + 1 ||
      cache.pair_count_elements < batch.batch_size || batch.total_atoms == kInt64Maximum ||
      batch.total_atoms > kInt64Maximum / batch.max_neighbors_per_atom ||
      cache.neighbor_offset_elements != batch.total_atoms + 1 ||
      cache.neighbor_count_elements < batch.total_atoms ||
      cache.neighbor_elements < batch.total_atoms * batch.max_neighbors_per_atom ||
      cache.generation_elements < batch.batch_size ||
      !required_pointer(cache.pairs, cache.pair_elements) ||
      !required_pointer(cache.pair_offsets, cache.pair_offset_elements) ||
      !required_pointer(cache.pair_counts, cache.pair_count_elements) ||
      !required_pointer(cache.neighbor_offsets, cache.neighbor_offset_elements) ||
      !required_pointer(cache.neighbor_counts, cache.neighbor_count_elements) ||
      !required_pointer(cache.neighbors, cache.neighbor_elements) ||
      !required_pointer(cache.pair_generations, cache.generation_elements) ||
      workspace.system_meta_elements < batch.batch_size ||
      workspace.atom_cell_elements < batch.total_atoms ||
      batch.batch_size > kInt64Maximum / (batch.max_cells_per_system + 1) ||
      workspace.cell_count_elements < batch.batch_size * (batch.max_cells_per_system + 1) ||
      workspace.cell_offset_elements < batch.batch_size * (batch.max_cells_per_system + 1) ||
      workspace.cell_fill_elements < batch.batch_size * (batch.max_cells_per_system + 1) ||
      workspace.cell_atom_elements < batch.total_atoms ||
      workspace.neighbor_cursor_elements < batch.total_atoms ||
      workspace.neighbor_scratch_elements < batch.total_atoms * batch.max_neighbors_per_atom ||
      workspace.pair_cursor_elements < batch.batch_size || workspace.sequence_elements < 1 ||
      !required_pointer(workspace.system_meta, workspace.system_meta_elements) ||
      !required_pointer(workspace.atom_cells, workspace.atom_cell_elements) ||
      !required_pointer(workspace.cell_counts, workspace.cell_count_elements) ||
      !required_pointer(workspace.cell_offsets, workspace.cell_offset_elements) ||
      !required_pointer(workspace.cell_fill, workspace.cell_fill_elements) ||
      !required_pointer(workspace.cell_atoms, workspace.cell_atom_elements) ||
      !required_pointer(workspace.neighbor_cursor, workspace.neighbor_cursor_elements) ||
      !required_pointer(workspace.neighbor_scratch, workspace.neighbor_scratch_elements) ||
      !required_pointer(workspace.pair_cursor, workspace.pair_cursor_elements) ||
      !required_pointer(workspace.sequence_active, workspace.sequence_elements) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }

  const std::int64_t neighbor_capacity = batch.total_atoms * batch.max_neighbors_per_atom;
  const std::int64_t pair_capacity = batch.batch_size * batch.max_pairs_per_system;

  std::array<AddressRange, 3> reads;
  std::array<AddressRange, 19> writes;
  if (!make_address_range(batch.atom_offsets, batch.atom_offset_elements,
                          sizeof(*batch.atom_offsets), &reads[0]) ||
      !make_address_range(positions, batch.total_atoms * 3, sizeof(*positions), &reads[1]) ||
      !make_address_range(batch.system_modes, batch.system_mode_elements,
                          sizeof(*batch.system_modes), &reads[2]) ||
      !make_address_range(cache.pairs, pair_capacity, sizeof(*cache.pairs), &writes[0]) ||
      !make_address_range(cache.pair_offsets, cache.pair_offset_elements,
                          sizeof(*cache.pair_offsets), &writes[1]) ||
      !make_address_range(cache.pair_counts, cache.pair_count_elements, sizeof(*cache.pair_counts),
                          &writes[17]) ||
      !make_address_range(cache.neighbor_offsets, cache.neighbor_offset_elements,
                          sizeof(*cache.neighbor_offsets), &writes[2]) ||
      !make_address_range(cache.neighbor_counts, cache.neighbor_count_elements,
                          sizeof(*cache.neighbor_counts), &writes[18]) ||
      !make_address_range(cache.neighbors, neighbor_capacity, sizeof(*cache.neighbors),
                          &writes[3]) ||
      !make_address_range(cache.pair_generations, cache.generation_elements,
                          sizeof(*cache.pair_generations), &writes[4]) ||
      !make_address_range(workspace.atom_cells, workspace.atom_cell_elements,
                          sizeof(*workspace.atom_cells), &writes[5]) ||
      !make_address_range(workspace.cell_counts, workspace.cell_count_elements,
                          sizeof(*workspace.cell_counts), &writes[6]) ||
      !make_address_range(workspace.cell_offsets, workspace.cell_offset_elements,
                          sizeof(*workspace.cell_offsets), &writes[7]) ||
      !make_address_range(workspace.cell_fill, workspace.cell_fill_elements,
                          sizeof(*workspace.cell_fill), &writes[8]) ||
      !make_address_range(workspace.cell_atoms, workspace.cell_atom_elements,
                          sizeof(*workspace.cell_atoms), &writes[9]) ||
      !make_address_range(workspace.neighbor_cursor, workspace.neighbor_cursor_elements,
                          sizeof(*workspace.neighbor_cursor), &writes[10]) ||
      !make_address_range(workspace.neighbor_scratch, workspace.neighbor_scratch_elements,
                          sizeof(*workspace.neighbor_scratch), &writes[11]) ||
      !make_address_range(workspace.pair_cursor, workspace.pair_cursor_elements,
                          sizeof(*workspace.pair_cursor), &writes[12]) ||
      !make_address_range(workspace.sequence_active, workspace.sequence_elements,
                          sizeof(*workspace.sequence_active), &writes[13]) ||
      !make_address_range(system_errors, batch.batch_size, sizeof(*system_errors), &writes[14]) ||
      !make_address_range(device_error, 1, sizeof(*device_error), &writes[15]) ||
      !make_address_range(workspace.system_meta, batch.batch_size, sizeof(*workspace.system_meta),
                          &writes[16]) ||
      !writable_ranges_are_disjoint(reads, writes)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t check_launch() noexcept { return cudaPeekAtLastError(); }

}  // namespace

bool query_gfn2_pairlist_requirements_cuda(
    std::int64_t batch_size, std::int64_t total_atoms, std::int64_t max_cells_per_system,
    std::int64_t max_neighbors_per_atom, std::int64_t max_pairs_per_system,
    std::int64_t* cache_pairs, std::int64_t* cache_neighbor_offsets, std::int64_t* cache_neighbors,
    std::int64_t* cache_pair_offsets, std::int64_t* cache_generations, std::int64_t* ws_meta,
    std::int64_t* ws_atom_cells, std::int64_t* ws_cell_arrays, std::int64_t* ws_cell_atoms,
    std::int64_t* ws_neighbor_cursor, std::int64_t* ws_neighbor_scratch,
    std::int64_t* ws_pair_cursor, std::int64_t* cache_pair_counts,
    std::int64_t* cache_neighbor_counts) noexcept {
  if (batch_size <= 0 || total_atoms < 0 || max_cells_per_system <= 0 ||
      max_neighbors_per_atom <= 0 || max_pairs_per_system <= 0) {
    return false;
  }
  const auto safe_product = [](std::int64_t a, std::int64_t b, std::int64_t* out) {
    if (a < 0 || b < 0 || (a != 0 && b > kInt64Maximum / a)) {
      return false;
    }
    *out = a * b;
    return true;
  };
  std::int64_t pairs = 0;
  std::int64_t cells = 0;
  std::int64_t neighbors = 0;
  if (!safe_product(batch_size, max_cells_per_system, &cells) ||
      !safe_product(total_atoms, max_neighbors_per_atom, &neighbors)) {
    return false;
  }
  if (!safe_product(batch_size, max_pairs_per_system, &pairs)) {
    return false;
  }
  /* Every system owns a trailing end slot.  This keeps the prefix sentinel
   * disjoint from the next system even when it uses its full cell capacity. */
  if (cells > kInt64Maximum - batch_size) {
    return false;
  }
  cells += batch_size;
  if (cache_pairs != nullptr) {
    *cache_pairs = pairs;
  }
  if (cache_neighbor_offsets != nullptr) {
    if (total_atoms == kInt64Maximum) return false;
    *cache_neighbor_offsets = total_atoms + 1;
  }
  if (cache_neighbors != nullptr) {
    *cache_neighbors = neighbors;
  }
  if (cache_pair_offsets != nullptr) {
    if (batch_size == kInt64Maximum) return false;
    *cache_pair_offsets = batch_size + 1;
  }
  if (cache_pair_counts != nullptr) {
    *cache_pair_counts = batch_size;
  }
  if (cache_neighbor_counts != nullptr) {
    *cache_neighbor_counts = total_atoms;
  }
  if (cache_generations != nullptr) {
    *cache_generations = batch_size;
  }
  if (ws_meta != nullptr) {
    *ws_meta = batch_size;
  }
  if (ws_atom_cells != nullptr) {
    *ws_atom_cells = total_atoms;
  }
  if (ws_cell_arrays != nullptr) {
    *ws_cell_arrays = cells;
  }
  if (ws_cell_atoms != nullptr) {
    *ws_cell_atoms = total_atoms;
  }
  if (ws_neighbor_cursor != nullptr) {
    *ws_neighbor_cursor = total_atoms;
  }
  if (ws_neighbor_scratch != nullptr) {
    *ws_neighbor_scratch = neighbors;
  }
  if (ws_pair_cursor != nullptr) {
    *ws_pair_cursor = batch_size;
  }
  return true;
}

cudaError_t reset_gfn2_pairlist_device_errors_cuda(std::int64_t batch_size,
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
      (system_range.begin < device_range.end && device_range.begin < system_range.end)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(
      system_errors, 0, static_cast<std::size_t>(batch_size) * sizeof(*system_errors), stream);
  return status == cudaSuccess ? cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream)
                               : status;
}

cudaError_t update_gfn2_pairlist_cache_cuda(
    const Gfn2PairListDeviceBatch& batch, const double* positions, std::uint64_t pair_generation,
    const Gfn2PairListDeviceCache& cache, const Gfn2PairListDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  cudaError_t status = validate_update(batch, positions, pair_generation, cache, workspace,
                                       system_errors, device_error);
  if (status != cudaSuccess) {
    return status;
  }
  topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, workspace.sequence_active,
                                                                device_error, true);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  const unsigned int blocks = static_cast<unsigned int>(batch.batch_size);
  zero_neighbor_cursors_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, workspace.neighbor_cursor, workspace.sequence_active, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  /* Bucket construction runs when the batch is sparse-mode or per-system
   * dispatch is active (dense peers then publish a cells==0 meta and scan the
   * full triangle inside build_neighbors).  A pure kDense batch skips buckets
   * entirely. */
  if (batch.mode == Gfn2PairListMode::kSparse || batch.system_modes != nullptr) {
    build_buckets_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        batch, positions, workspace.system_meta, workspace.atom_cells, workspace.cell_counts,
        workspace.sequence_active, system_errors, device_error);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
    prefix_cells_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        batch, workspace.system_meta, workspace.cell_counts, workspace.cell_offsets,
        workspace.cell_fill, workspace.sequence_active, system_errors, device_error);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
    scatter_atoms_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        batch, workspace.system_meta, workspace.atom_cells, workspace.cell_fill,
        workspace.cell_atoms, workspace.sequence_active, system_errors, device_error);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
  }
  build_neighbors_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, positions, workspace.system_meta, workspace.atom_cells, workspace.cell_offsets,
      workspace.cell_atoms, workspace.neighbor_cursor, workspace.neighbor_scratch,
      workspace.sequence_active, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  sort_neighbors_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, workspace.neighbor_cursor, workspace.neighbor_scratch, workspace.sequence_active,
      system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  count_pairs_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, workspace.neighbor_cursor, workspace.pair_cursor, workspace.sequence_active,
      system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  prefix_pair_offsets_kernel<<<1, 1, 0, stream>>>(
      batch, workspace.pair_cursor, cache, workspace.sequence_active, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  prefix_neighbor_offsets_kernel<<<1, 1, 0, stream>>>(batch, workspace.neighbor_cursor, cache,
                                                      workspace.sequence_active, system_errors,
                                                      device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, pair_generation, workspace.neighbor_cursor, workspace.neighbor_scratch, cache,
      workspace.sequence_active, system_errors, device_error);
  return check_launch();
}

cudaError_t evaluate_gfn2_pairlist_coordination_cuda(
    const Gfn2PairListDeviceBatch& batch, const double* positions, const double* covalent_radii,
    std::uint64_t pair_generation, const Gfn2PairListDeviceCache& cache, double* coordination,
    const Gfn2PairListDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  cudaError_t status = validate_batch(batch);
  if (status != cudaSuccess) {
    return status;
  }
  if (pair_generation == 0u || batch.plan_token != cache.plan_token ||
      batch.plan_token != workspace.plan_token || batch.cutoff < kDefaultCutoffBohr ||
      batch.total_atoms > kInt64Maximum / batch.max_neighbors_per_atom ||
      !required_pointer(coordination, batch.total_atoms) ||
      !required_pointer(covalent_radii, batch.total_atoms) ||
      !required_pointer(positions, batch.total_atoms * 3) ||
      cache.generation_elements < batch.batch_size ||
      cache.neighbor_offset_elements != batch.total_atoms + 1 ||
      cache.neighbor_count_elements < batch.total_atoms ||
      cache.neighbor_elements < batch.total_atoms * batch.max_neighbors_per_atom ||
      !required_pointer(cache.pair_generations, cache.generation_elements) ||
      !required_pointer(cache.neighbor_offsets, cache.neighbor_offset_elements) ||
      !required_pointer(cache.neighbor_counts, cache.neighbor_count_elements) ||
      !required_pointer(cache.neighbors, cache.neighbor_elements) ||
      workspace.sequence_elements < 1 ||
      !required_pointer(workspace.sequence_active, workspace.sequence_elements) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  const std::int64_t neighbor_capacity = batch.total_atoms * batch.max_neighbors_per_atom;
  std::array<AddressRange, 8> reads;
  std::array<AddressRange, 3> writes;
  if (!make_address_range(batch.atom_offsets, batch.atom_offset_elements,
                          sizeof(*batch.atom_offsets), &reads[0]) ||
      !make_address_range(positions, batch.total_atoms * 3, sizeof(*positions), &reads[1]) ||
      !make_address_range(covalent_radii, batch.total_atoms, sizeof(*covalent_radii), &reads[2]) ||
      !make_address_range(cache.pair_generations, cache.generation_elements,
                          sizeof(*cache.pair_generations), &reads[3]) ||
      !make_address_range(cache.neighbor_offsets, cache.neighbor_offset_elements,
                          sizeof(*cache.neighbor_offsets), &reads[4]) ||
      !make_address_range(cache.neighbor_counts, cache.neighbor_count_elements,
                          sizeof(*cache.neighbor_counts), &reads[7]) ||
      !make_address_range(cache.neighbors, neighbor_capacity, sizeof(*cache.neighbors),
                          &reads[5]) ||
      !make_address_range(workspace.sequence_active, 1, sizeof(*workspace.sequence_active),
                          &reads[6]) ||
      !make_address_range(coordination, batch.total_atoms, sizeof(*coordination), &writes[0]) ||
      !make_address_range(system_errors, batch.batch_size, sizeof(*system_errors), &writes[1]) ||
      !make_address_range(device_error, 1, sizeof(*device_error), &writes[2]) ||
      !writable_ranges_are_disjoint(reads, writes)) {
    return cudaErrorInvalidValue;
  }
  /* Evaluation consumes the sequence snapshot produced by the preceding
   * update.  Re-running topology preflight here would erase a batch-wide
   * invalid-offset marker while peer-local errors are intentionally sticky. */
  const unsigned int blocks = static_cast<unsigned int>(batch.batch_size);
  preflight_coordination_pairs_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, positions, covalent_radii, pair_generation, cache, workspace.sequence_active,
      system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  evaluate_coordination_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, positions, covalent_radii, pair_generation, cache, coordination,
      workspace.sequence_active, system_errors, device_error);
  return check_launch();
}

cudaError_t add_gfn2_pairlist_coordination_vjp_cuda(
    const Gfn2PairListDeviceBatch& batch, const double* positions, const double* covalent_radii,
    std::uint64_t pair_generation, const Gfn2PairListDeviceCache& cache, const double* dE_dcn,
    double* gradients, double* gradient_scratch, std::int64_t gradient_elements,
    const Gfn2PairListDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  cudaError_t status = validate_batch(batch);
  if (status != cudaSuccess) {
    return status;
  }
  if (pair_generation == 0u || batch.plan_token != cache.plan_token ||
      batch.plan_token != workspace.plan_token || batch.cutoff < kDefaultCutoffBohr ||
      batch.total_atoms > kInt64Maximum / batch.max_neighbors_per_atom ||
      batch.total_atoms > kInt64Maximum / 3 || gradient_elements < batch.total_atoms * 3 ||
      !required_pointer(positions, batch.total_atoms * 3) ||
      !required_pointer(covalent_radii, batch.total_atoms) ||
      !required_pointer(dE_dcn, batch.total_atoms) ||
      !required_pointer(gradients, batch.total_atoms * 3) ||
      !required_pointer(gradient_scratch, gradient_elements) ||
      cache.generation_elements < batch.batch_size ||
      cache.neighbor_offset_elements != batch.total_atoms + 1 ||
      cache.neighbor_count_elements < batch.total_atoms ||
      cache.neighbor_elements < batch.total_atoms * batch.max_neighbors_per_atom ||
      !required_pointer(cache.pair_generations, cache.generation_elements) ||
      !required_pointer(cache.neighbor_offsets, cache.neighbor_offset_elements) ||
      !required_pointer(cache.neighbor_counts, cache.neighbor_count_elements) ||
      !required_pointer(cache.neighbors, cache.neighbor_elements) ||
      workspace.sequence_elements < 1 ||
      !required_pointer(workspace.sequence_active, workspace.sequence_elements) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  const std::int64_t neighbor_capacity = batch.total_atoms * batch.max_neighbors_per_atom;
  std::array<AddressRange, 9> reads;
  std::array<AddressRange, 4> writes;
  if (!make_address_range(batch.atom_offsets, batch.atom_offset_elements,
                          sizeof(*batch.atom_offsets), &reads[0]) ||
      !make_address_range(positions, batch.total_atoms * 3, sizeof(*positions), &reads[1]) ||
      !make_address_range(covalent_radii, batch.total_atoms, sizeof(*covalent_radii), &reads[2]) ||
      !make_address_range(dE_dcn, batch.total_atoms, sizeof(*dE_dcn), &reads[3]) ||
      !make_address_range(cache.pair_generations, cache.generation_elements,
                          sizeof(*cache.pair_generations), &reads[4]) ||
      !make_address_range(cache.neighbor_offsets, cache.neighbor_offset_elements,
                          sizeof(*cache.neighbor_offsets), &reads[5]) ||
      !make_address_range(cache.neighbor_counts, cache.neighbor_count_elements,
                          sizeof(*cache.neighbor_counts), &reads[8]) ||
      !make_address_range(cache.neighbors, neighbor_capacity, sizeof(*cache.neighbors),
                          &reads[6]) ||
      !make_address_range(workspace.sequence_active, 1, sizeof(*workspace.sequence_active),
                          &reads[7]) ||
      !make_address_range(gradients, batch.total_atoms * 3, sizeof(*gradients), &writes[0]) ||
      !make_address_range(gradient_scratch, gradient_elements, sizeof(*gradient_scratch),
                          &writes[1]) ||
      !make_address_range(system_errors, batch.batch_size, sizeof(*system_errors), &writes[2]) ||
      !make_address_range(device_error, 1, sizeof(*device_error), &writes[3]) ||
      !writable_ranges_are_disjoint(reads, writes)) {
    return cudaErrorInvalidValue;
  }
  const unsigned int blocks = static_cast<unsigned int>(batch.batch_size);
  coordination_vjp_preflight_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, positions, covalent_radii, pair_generation, cache, dE_dcn, gradients,
      workspace.sequence_active, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  coordination_vjp_accumulate_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, positions, covalent_radii, pair_generation, cache, dE_dcn, gradients, gradient_scratch,
      workspace.sequence_active, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  coordination_vjp_publish_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, gradient_scratch, gradients, workspace.sequence_active, system_errors, device_error);
  return check_launch();
}

/*
 * A zero-copy cache projection of a committed consumer view.  The VJP kernels
 * read neighbor_offsets/neighbor_counts/neighbors plus per-peer generations;
 * the committed view supplies committed_generations in the same slot.  The
 * projection aliases the committed arrays, never copies or repacks, and is
 * reconstructed per launch on the host like every other leaf descriptor.
 */
namespace {

/* Rebuild the kernel-level cache projection from a committed consumer view.
 * Returned by value so callers pass a true temporary into the kernel, exactly
 * like the established device.cache() path. */
Gfn2PairListDeviceCache pairlist_cache_from_consumer(
    const Gfn2PairListConsumerView& committed) noexcept {
  Gfn2PairListDeviceCache cache{};
  cache.pairs = const_cast<Gfn2AtomPair*>(committed.pairs);
  cache.pair_elements = committed.pair_count;
  cache.pair_offsets = const_cast<std::int64_t*>(committed.pair_offsets);
  cache.pair_offset_elements = committed.pair_offset_count;
  cache.pair_counts = const_cast<std::int64_t*>(committed.pair_counts);
  cache.pair_count_elements = committed.pair_count_elements;
  cache.neighbor_offsets = const_cast<std::int64_t*>(committed.neighbor_offsets);
  cache.neighbor_offset_elements = committed.neighbor_offset_count;
  cache.neighbor_counts = const_cast<std::int64_t*>(committed.neighbor_counts);
  cache.neighbor_count_elements = committed.neighbor_count_elements;
  cache.neighbors = const_cast<std::int64_t*>(committed.neighbors);
  cache.neighbor_elements = committed.neighbor_count;
  cache.pair_generations = const_cast<std::uint64_t*>(committed.committed_generations);
  cache.generation_elements = committed.committed_generation_count;
  cache.plan_token = committed.plan_token;
  return cache;
}

}  // namespace

cudaError_t add_gfn2_pairlist_consumer_coordination_vjp_cuda(
    const Gfn2PairListDeviceBatch& batch, const Gfn2PairListConsumerView& committed,
    const double* positions, const double* covalent_radii, std::uint64_t expected_generation,
    const double* dE_dcn, double* gradients, double* gradient_scratch,
    std::int64_t gradient_elements, const std::uint32_t* sequence_active,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (committed.plan_token == 0u || committed.plan_token != batch.plan_token ||
      committed.state != Gfn2PairListState::kCommitted ||
      committed.batch_size != batch.batch_size || committed.total_atoms != batch.total_atoms ||
      committed.max_neighbors_per_atom != batch.max_neighbors_per_atom ||
      committed.pair_offsets == nullptr || committed.pair_counts == nullptr ||
      committed.neighbor_offsets == nullptr || committed.neighbor_counts == nullptr ||
      committed.neighbors == nullptr || committed.committed_generations == nullptr ||
      !required_pointer(sequence_active, 1) || !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t)) ||
      expected_generation == 0u || gradient_elements < batch.total_atoms * 3) {
    return cudaErrorInvalidValue;
  }
  const Gfn2PairListDeviceCache cache = pairlist_cache_from_consumer(committed);

  if (expected_generation == 0u || batch.plan_token != cache.plan_token ||
      batch.total_atoms > kInt64Maximum / batch.max_neighbors_per_atom ||
      batch.total_atoms > kInt64Maximum / 3 ||
      batch.total_atoms > kInt64Maximum / 3 ||
      !required_pointer(positions, batch.total_atoms * 3) ||
      !required_pointer(covalent_radii, batch.total_atoms) ||
      !required_pointer(dE_dcn, batch.total_atoms) ||
      !required_pointer(gradients, batch.total_atoms * 3) ||
      !required_pointer(gradient_scratch, gradient_elements) ||
      cache.generation_elements < batch.batch_size ||
      cache.neighbor_offset_elements != batch.total_atoms + 1 ||
      cache.neighbor_count_elements < batch.total_atoms ||
      cache.neighbor_elements < batch.total_atoms * batch.max_neighbors_per_atom ||
      cache.pair_offsets == nullptr || cache.pair_counts == nullptr ||
      cache.neighbor_offsets == nullptr || cache.neighbor_counts == nullptr ||
      cache.neighbors == nullptr || cache.pair_generations == nullptr) {
    return cudaErrorInvalidValue;
  }
  const std::int64_t neighbor_capacity = batch.total_atoms * batch.max_neighbors_per_atom;
  std::array<AddressRange, 9> reads;
  std::array<AddressRange, 4> writes;
  if (!make_address_range(batch.atom_offsets, batch.atom_offset_elements,
                          sizeof(*batch.atom_offsets), &reads[0]) ||
      !make_address_range(positions, batch.total_atoms * 3, sizeof(*positions), &reads[1]) ||
      !make_address_range(covalent_radii, batch.total_atoms, sizeof(*covalent_radii), &reads[2]) ||
      !make_address_range(dE_dcn, batch.total_atoms, sizeof(*dE_dcn), &reads[3]) ||
      !make_address_range(cache.pair_generations, cache.generation_elements,
                          sizeof(*cache.pair_generations), &reads[4]) ||
      !make_address_range(cache.neighbor_offsets, cache.neighbor_offset_elements,
                          sizeof(*cache.neighbor_offsets), &reads[5]) ||
      !make_address_range(cache.neighbor_counts, cache.neighbor_count_elements,
                          sizeof(*cache.neighbor_counts), &reads[8]) ||
      !make_address_range(cache.neighbors, neighbor_capacity, sizeof(*cache.neighbors),
                          &reads[6]) ||
      !make_address_range(sequence_active, 1, sizeof(*sequence_active), &reads[7]) ||
      !make_address_range(gradients, batch.total_atoms * 3, sizeof(*gradients), &writes[0]) ||
      !make_address_range(gradient_scratch, gradient_elements, sizeof(*gradient_scratch),
                          &writes[1]) ||
      !make_address_range(system_errors, batch.batch_size, sizeof(*system_errors), &writes[2]) ||
      !make_address_range(device_error, 1, sizeof(*device_error), &writes[3]) ||
      !writable_ranges_are_disjoint(reads, writes)) {
    return cudaErrorInvalidValue;
  }
  const unsigned int blocks = static_cast<unsigned int>(batch.batch_size);
  coordination_vjp_preflight_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, positions, covalent_radii, expected_generation, cache, dE_dcn, gradients,
      sequence_active, system_errors, device_error);
  cudaError_t status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  coordination_vjp_accumulate_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, positions, covalent_radii, expected_generation, cache, dE_dcn, gradients,
      gradient_scratch, sequence_active, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  coordination_vjp_publish_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, gradient_scratch, gradients, sequence_active, system_errors, device_error);
  return check_launch();
}

bool gfn2_pairlist_use_sparse_for(std::int64_t atoms_per_system) noexcept {
  return atoms_per_system > kSparseCrossoverAtoms;
}

}  // namespace gpuxtb::detail::cuda
