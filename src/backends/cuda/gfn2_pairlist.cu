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
                                          std::uint32_t* device_error) {
  if (atomicAdd(device_error, 0u) !=
      static_cast<std::uint32_t>(Gfn2PairListDeviceError::kSuccess)) {
    return;
  }
  if (threadIdx.x == 0 &&
      (batch.atom_offsets[0] != 0 || batch.atom_offsets[batch.batch_size] != batch.total_atoms)) {
    record_error(device_error, Gfn2PairListDeviceError::kInvalidOffsets);
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    const std::int64_t begin = batch.atom_offsets[system];
    const std::int64_t end = batch.atom_offsets[system + 1];
    const bool valid = begin >= 0 && begin <= end && end <= batch.total_atoms;
    if (!valid) {
      record_error(device_error, Gfn2PairListDeviceError::kInvalidOffsets);
    }
  }
}

/* Snapshot topology/upstream validity before peer-local errors set device_error. */
__global__ void capture_sequence_kernel(const std::uint32_t* device_error,
                                        std::uint32_t* sequence_active) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *sequence_active = atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                               static_cast<std::uint32_t>(Gfn2PairListDeviceError::kSuccess)
                           ? 1u
                           : 0u;
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
      ranges->cell_base = system * batch.max_cells_per_system;
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
  meta.cells = 1;
  for (int axis = 0; axis < 3; ++axis) {
    meta.origin[axis] = 0.0;
  }
  /* Empty and failed systems still publish the single-bucket default meta so
   * later stages never read a stale value from a previous inference. */
  if (threadIdx.x == 0) {
    system_meta[system] = meta;
  }
  __syncthreads();

  if (atom_count > 0 && threadIdx.x == 0) {
    constexpr double edge = kDefaultCutoffBohr;
    double min_value[3] = {0.0, 0.0, 0.0};
    double max_value[3] = {0.0, 0.0, 0.0};
    const std::int64_t first_index = ranges.atom_begin;
    for (int axis = 0; axis < 3; ++axis) {
      min_value[axis] = positions[first_index * 3 + axis];
      max_value[axis] = positions[first_index * 3 + axis];
    }
    for (std::int64_t atom = ranges.atom_begin; atom < ranges.atom_end; ++atom) {
      for (int axis = 0; axis < 3; ++axis) {
        const double value = positions[atom * 3 + axis];
        min_value[axis] = fmin(min_value[axis], value);
        max_value[axis] = fmax(max_value[axis], value);
      }
    }
    if (!finite_position(positions, first_index * 3)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2PairListDeviceError::kNonfinitePosition);
      valid = 0;
    }
    if (valid != 0) {
      for (int axis = 0; axis < 3; ++axis) {
        const double extent = max_value[axis] - min_value[axis];
        meta.origin[axis] = min_value[axis];
        const std::int64_t buckets =
            extent <= 0.0 ? 1 : static_cast<std::int64_t>(extent / edge) + 1;
        if (axis == 0) {
          meta.nx = buckets;
        } else if (axis == 1) {
          meta.ny = buckets;
        } else {
          meta.nz = buckets;
        }
      }
      if (meta.nx > kInt64Maximum / meta.ny || (meta.nx * meta.ny) > kInt64Maximum / meta.nz) {
        record_system_error(system_errors, system, device_error,
                            Gfn2PairListDeviceError::kCellCapacityExceeded);
        valid = 0;
      } else {
        const std::int64_t cells = meta.nx * meta.ny * meta.nz;
        if (cells > batch.max_cells_per_system) {
          record_system_error(system_errors, system, device_error,
                              Gfn2PairListDeviceError::kCellCapacityExceeded);
          valid = 0;
        } else {
          meta.cells = cells;
          system_meta[system] = meta;
        }
      }
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  const Gfn2PairListSystemMeta settled = system_meta[system];
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
        static_cast<std::int64_t>((positions[coordinate] - settled.origin[0]) / kDefaultCutoffBohr);
    std::int64_t cy = static_cast<std::int64_t>((positions[coordinate + 1] - settled.origin[1]) /
                                                kDefaultCutoffBohr);
    std::int64_t cz = static_cast<std::int64_t>((positions[coordinate + 2] - settled.origin[2]) /
                                                kDefaultCutoffBohr);
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
  if (valid == 0) {
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

  const bool mode_sparse = batch.mode == Gfn2PairListMode::kSparse;
  if (mode_sparse) {
    /* Bucketed sweep: the per-system meta is only valid here because the
     * sparse launch runs the bucket construction before this kernel. */
    const Gfn2PairListSystemMeta meta = system_meta[system];
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

  /* Dense fallback: full triangle, no bucket state is consulted or required. */
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
 * pair_offsets.  The whole call is a no-op when the sequence is inactive, so
 * caller outputs are never touched before publication.
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
        running += pair_cursor[system];
      }
      cache.pair_offsets[batch.batch_size] = running;
    }
  }
}

/*
 * Global exclusive prefix of the per-atom neighbor counts into the public
 * neighbor_offsets.  Failed systems contribute a zero-length range so healthy
 * peers keep contiguous, correct slices.  The whole call is a no-op when the
 * sequence is inactive, so caller outputs are never touched before
 * publication.
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
        if (system_is_valid(system_errors, system)) {
          running += neighbor_cursor[atom];
        }
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
  if (batch.batch_size <= 0 || batch.total_atoms < 0 ||
      batch.batch_size == std::numeric_limits<std::int64_t>::max() ||
      batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      batch.total_atoms > kInt64Maximum / 3 || batch.atom_offset_elements != batch.batch_size + 1 ||
      batch.max_cells_per_system <= 0 || batch.max_neighbors_per_atom <= 0 ||
      batch.max_pairs_per_system <= 0 || !(batch.cutoff > 0.0) || !isfinite(batch.cutoff) ||
      batch.plan_token == 0u || !is_aligned(batch.atom_offsets, alignof(std::int64_t))) {
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
      cache.pair_elements < batch.max_pairs_per_system * batch.batch_size ||
      cache.pair_offset_elements != batch.batch_size + 1 ||
      cache.neighbor_offset_elements != batch.total_atoms + 1 ||
      cache.neighbor_elements < batch.total_atoms * batch.max_neighbors_per_atom ||
      cache.generation_elements < batch.batch_size ||
      !required_pointer(cache.pairs, cache.pair_elements) ||
      !required_pointer(cache.pair_offsets, cache.pair_offset_elements) ||
      !required_pointer(cache.neighbor_offsets, cache.neighbor_offset_elements) ||
      !required_pointer(cache.neighbors, cache.neighbor_elements) ||
      !required_pointer(cache.pair_generations, cache.generation_elements) ||
      workspace.system_meta_elements < batch.batch_size ||
      workspace.atom_cell_elements < batch.total_atoms ||
      batch.batch_size > kInt64Maximum / batch.max_cells_per_system ||
      workspace.cell_count_elements < batch.batch_size * batch.max_cells_per_system + 1 ||
      workspace.cell_offset_elements < batch.batch_size * batch.max_cells_per_system + 1 ||
      workspace.cell_fill_elements < batch.batch_size * batch.max_cells_per_system + 1 ||
      workspace.cell_atom_elements < batch.total_atoms ||
      batch.total_atoms > kInt64Maximum / batch.max_neighbors_per_atom ||
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
  std::array<AddressRange, 16> writes;
  if (!make_address_range(batch.atom_offsets, batch.atom_offset_elements,
                          sizeof(*batch.atom_offsets), &reads[0]) ||
      !make_address_range(positions, batch.total_atoms * 3, sizeof(*positions), &reads[1]) ||
      !make_address_range(workspace.system_meta, batch.batch_size, sizeof(*workspace.system_meta),
                          &reads[2]) ||
      !make_address_range(cache.pairs, pair_capacity, sizeof(*cache.pairs), &writes[0]) ||
      !make_address_range(cache.pair_offsets, cache.pair_offset_elements,
                          sizeof(*cache.pair_offsets), &writes[1]) ||
      !make_address_range(cache.neighbor_offsets, cache.neighbor_offset_elements,
                          sizeof(*cache.neighbor_offsets), &writes[2]) ||
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
    std::int64_t* ws_pair_cursor) noexcept {
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
  /* The three per-system cell arrays share one allocation; cell_offsets needs
   * a trailing end slot, so every cell array is sized cells + 1. */
  cells += 1;
  if (cache_pairs != nullptr) {
    *cache_pairs = pairs;
  }
  if (cache_neighbor_offsets != nullptr) {
    *cache_neighbor_offsets = total_atoms + 1;
  }
  if (cache_neighbors != nullptr) {
    *cache_neighbors = neighbors;
  }
  if (cache_pair_offsets != nullptr) {
    *cache_pair_offsets = batch_size + 1;
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
  zero_neighbor_cursors_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, workspace.neighbor_cursor, workspace.sequence_active, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  if (batch.mode == Gfn2PairListMode::kSparse) {
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
  if (batch.plan_token != cache.plan_token || batch.plan_token != workspace.plan_token ||
      batch.batch_size <= 0 || !required_pointer(coordination, batch.total_atoms) ||
      !required_pointer(covalent_radii, batch.total_atoms) ||
      !required_pointer(positions, batch.total_atoms * 3) ||
      cache.generation_elements < batch.batch_size ||
      cache.neighbor_offset_elements != batch.total_atoms + 1 ||
      cache.neighbor_elements < batch.total_atoms * batch.max_neighbors_per_atom ||
      cache.pair_generations == nullptr || cache.neighbor_offsets == nullptr ||
      cache.neighbors == nullptr || !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
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
  const unsigned int blocks = static_cast<unsigned int>(batch.batch_size);
  evaluate_coordination_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, positions, covalent_radii, pair_generation, cache, coordination,
      workspace.sequence_active, system_errors, device_error);
  return check_launch();
}

bool gfn2_pairlist_use_sparse_for(std::int64_t atoms_per_system) noexcept {
  return atoms_per_system > kSparseCrossoverAtoms;
}

}  // namespace gpuxtb::detail::cuda
