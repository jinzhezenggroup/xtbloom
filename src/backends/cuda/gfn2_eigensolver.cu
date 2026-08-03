#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_eigensolver.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kThreadsPerSystem = 128;
constexpr std::int64_t kMaximumInt64 = 9223372036854775807LL;
constexpr double kOne = 1.0;

struct GeometryGenerationSource {
  std::uint64_t scalar = 0u;
  const std::uint64_t* device = nullptr;
};

template <typename T>
bool is_aligned(const T* pointer, std::size_t alignment) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == static_cast<std::uintptr_t>(0u);
}

bool checked_add(std::int64_t first, std::int64_t second, std::int64_t* result) noexcept {
  if (first < 0 || second < 0 || first > kMaximumInt64 - second) {
    return false;
  }
  *result = first + second;
  return true;
}

bool checked_multiply(std::int64_t first, std::int64_t second, std::int64_t* result) noexcept {
  if (first < 0 || second < 0 || (first != 0 && second > kMaximumInt64 / first)) {
    return false;
  }
  *result = first * second;
  return true;
}

bool ranges_overlap(std::int64_t first_begin, std::int64_t first_end, std::int64_t second_begin,
                    std::int64_t second_end) noexcept {
  return first_begin < second_end && second_begin < first_end;
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

bool make_byte_range(const void* pointer, std::size_t bytes, AddressRange* range) noexcept {
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

Gfn2EigensolverLaunchResult invalid_argument() noexcept {
  Gfn2EigensolverLaunchResult result;
  result.status = Gfn2EigensolverLaunchStatus::kInvalidArgument;
  result.cuda_status = cudaErrorInvalidValue;
  return result;
}

Gfn2EigensolverLaunchResult cuda_failure(cudaError_t status) noexcept {
  Gfn2EigensolverLaunchResult result;
  result.status = Gfn2EigensolverLaunchStatus::kCudaError;
  result.cuda_status = status;
  return result;
}

Gfn2EigensolverLaunchResult cublas_failure(cublasStatus_t status) noexcept {
  Gfn2EigensolverLaunchResult result;
  result.status = Gfn2EigensolverLaunchStatus::kCublasError;
  result.cublas_status = status;
  return result;
}

Gfn2EigensolverLaunchResult cusolver_failure(cusolverStatus_t status) noexcept {
  Gfn2EigensolverLaunchResult result;
  result.status = Gfn2EigensolverLaunchStatus::kCusolverError;
  result.cusolver_status = status;
  return result;
}

Gfn2EigensolverLaunchResult launch_success() noexcept { return {}; }

bool valid_options(const Gfn2EigensolverOptions& options) noexcept {
  return std::isfinite(options.minimum_overlap_rcond) && options.minimum_overlap_rcond > 0.0 &&
         options.minimum_overlap_rcond <= 1.0 && std::isfinite(options.symmetry_tolerance) &&
         options.symmetry_tolerance >= 0.0;
}

bool valid_bucket_plan(const Gfn2EigensolverDeviceBatch& batch,
                       const Gfn2EigensolverBucket* buckets, std::int64_t bucket_count) noexcept {
  if (batch.batch_size <= 0 || batch.batch_size > std::numeric_limits<int>::max() ||
      batch.total_orbitals <= 0 || batch.total_matrix_elements <= 0 || bucket_count <= 0 ||
      bucket_count > std::numeric_limits<int>::max() || buckets == nullptr ||
      batch.plan_token == 0u || batch.orbital_offset_count != batch.batch_size + 1 ||
      batch.matrix_offset_count != batch.batch_size + 1 ||
      batch.bucket_system_count != batch.batch_size || batch.active_elements != batch.batch_size ||
      !is_aligned(batch.orbital_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.matrix_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.bucket_systems, alignof(std::int32_t)) ||
      !is_aligned(batch.active, alignof(std::uint8_t))) {
    return false;
  }

  std::int64_t counted_systems = 0;
  std::int64_t counted_orbitals = 0;
  std::int64_t counted_matrices = 0;
  for (std::int64_t bucket_index = 0; bucket_index < bucket_count; ++bucket_index) {
    const Gfn2EigensolverBucket& bucket = buckets[bucket_index];
    const std::int64_t n = bucket.orbital_count;
    const std::int64_t count = bucket.system_count;
    std::int64_t matrix_stride = 0;
    std::int64_t matrix_span = 0;
    std::int64_t orbital_span = 0;
    std::int64_t system_end = 0;
    std::int64_t matrix_end = 0;
    std::int64_t orbital_end = 0;
    if (n <= 0 || count <= 0 || !checked_multiply(n, n, &matrix_stride) ||
        !checked_multiply(matrix_stride, count, &matrix_span) ||
        !checked_multiply(n, count, &orbital_span) ||
        matrix_stride > std::numeric_limits<int>::max() ||
        matrix_span > std::numeric_limits<int>::max() ||
        orbital_span > std::numeric_limits<int>::max() ||
        !checked_add(bucket.system_index_offset, count, &system_end) ||
        !checked_add(bucket.matrix_scratch_offset, matrix_span, &matrix_end) ||
        !checked_add(bucket.orbital_scratch_offset, orbital_span, &orbital_end) ||
        system_end > batch.bucket_system_count || matrix_end > batch.total_matrix_elements ||
        orbital_end > batch.total_orbitals ||
        !checked_add(counted_systems, count, &counted_systems) ||
        !checked_add(counted_orbitals, orbital_span, &counted_orbitals) ||
        !checked_add(counted_matrices, matrix_span, &counted_matrices)) {
      return false;
    }
    for (std::int64_t other_index = 0; other_index < bucket_index; ++other_index) {
      const Gfn2EigensolverBucket& other = buckets[other_index];
      const std::int64_t other_n = other.orbital_count;
      const std::int64_t other_count = other.system_count;
      std::int64_t other_matrix_stride = 0;
      std::int64_t other_matrix_span = 0;
      std::int64_t other_orbital_span = 0;
      std::int64_t other_system_end = 0;
      std::int64_t other_matrix_end = 0;
      std::int64_t other_orbital_end = 0;
      if (!checked_multiply(other_n, other_n, &other_matrix_stride) ||
          !checked_multiply(other_matrix_stride, other_count, &other_matrix_span) ||
          !checked_multiply(other_n, other_count, &other_orbital_span) ||
          !checked_add(other.system_index_offset, other_count, &other_system_end) ||
          !checked_add(other.matrix_scratch_offset, other_matrix_span, &other_matrix_end) ||
          !checked_add(other.orbital_scratch_offset, other_orbital_span, &other_orbital_end)) {
        return false;
      }
      if (ranges_overlap(bucket.system_index_offset, system_end, other.system_index_offset,
                         other_system_end) ||
          ranges_overlap(bucket.matrix_scratch_offset, matrix_end, other.matrix_scratch_offset,
                         other_matrix_end) ||
          ranges_overlap(bucket.orbital_scratch_offset, orbital_end, other.orbital_scratch_offset,
                         other_orbital_end)) {
        return false;
      }
    }
  }
  return counted_systems == batch.batch_size && counted_orbitals == batch.total_orbitals &&
         counted_matrices == batch.total_matrix_elements;
}

bool valid_workspace(const Gfn2EigensolverDeviceBatch& batch,
                     const Gfn2EigensolverDeviceWorkspace& workspace) noexcept {
  return workspace.plan_token == batch.plan_token &&
         workspace.matrix_a_elements >= batch.total_matrix_elements &&
         workspace.matrix_b_elements >= batch.total_matrix_elements &&
         workspace.eigenvalue_elements >= batch.total_orbitals &&
         workspace.factor_pointer_elements >= batch.batch_size &&
         workspace.matrix_pointer_elements >= batch.batch_size &&
         workspace.info_a_elements >= batch.batch_size &&
         workspace.info_b_elements >= batch.batch_size &&
         workspace.eligible_elements >= batch.batch_size &&
         workspace.sequence_active_elements >= 1 &&
         is_aligned(workspace.matrix_scratch_a, alignof(double)) &&
         is_aligned(workspace.matrix_scratch_b, alignof(double)) &&
         is_aligned(workspace.eigenvalue_scratch, alignof(double)) &&
         is_aligned(workspace.factor_pointers, alignof(double*)) &&
         is_aligned(workspace.matrix_pointers, alignof(double*)) &&
         is_aligned(workspace.info_a, alignof(int)) && is_aligned(workspace.info_b, alignof(int)) &&
         is_aligned(workspace.eligible, alignof(std::uint8_t)) &&
         is_aligned(workspace.sequence_active, alignof(std::uint32_t)) &&
         (workspace.solver_device_workspace_bytes == 0u ||
          is_aligned(static_cast<const std::byte*>(workspace.solver_device_workspace), 256u)) &&
         (workspace.solver_host_workspace_bytes == 0u ||
          is_aligned(static_cast<const std::byte*>(workspace.solver_host_workspace),
                     alignof(std::max_align_t)));
}

bool valid_cache(const Gfn2EigensolverDeviceBatch& batch,
                 const Gfn2EigensolverOverlapCache& cache) noexcept {
  return cache.plan_token == batch.plan_token &&
         cache.factor_elements >= batch.total_matrix_elements &&
         cache.generation_elements >= batch.batch_size &&
         cache.status_elements >= batch.batch_size &&
         is_aligned(cache.cholesky_factors, alignof(double)) &&
         is_aligned(cache.geometry_generations, alignof(std::uint64_t)) &&
         is_aligned(cache.factor_statuses, alignof(std::uint32_t));
}

bool valid_factor_ranges(const Gfn2EigensolverDeviceBatch& batch, const double* overlap,
                         const std::uint64_t* geometry_epoch,
                         const Gfn2EigensolverDeviceWorkspace& workspace,
                         const Gfn2EigensolverOverlapCache& cache, std::uint32_t* system_errors,
                         std::uint32_t* device_error) noexcept {
  std::array<AddressRange, 6> reads{};
  std::array<AddressRange, 16> writes{};
  return make_range(batch.orbital_offsets, batch.batch_size + 1, sizeof(std::int64_t), &reads[0]) &&
         make_range(batch.matrix_offsets, batch.batch_size + 1, sizeof(std::int64_t), &reads[1]) &&
         make_range(batch.bucket_systems, batch.batch_size, sizeof(std::int32_t), &reads[2]) &&
         make_range(batch.active, batch.batch_size, sizeof(std::uint8_t), &reads[3]) &&
         make_range(overlap, batch.total_matrix_elements, sizeof(double), &reads[4]) &&
         make_range(geometry_epoch, geometry_epoch == nullptr ? 0 : 1, sizeof(std::uint64_t),
                    &reads[5]) &&
         make_range(workspace.matrix_scratch_a, batch.total_matrix_elements, sizeof(double),
                    &writes[0]) &&
         make_range(workspace.matrix_scratch_b, batch.total_matrix_elements, sizeof(double),
                    &writes[1]) &&
         make_range(workspace.eigenvalue_scratch, batch.total_orbitals, sizeof(double),
                    &writes[2]) &&
         make_range(workspace.factor_pointers, batch.batch_size, sizeof(double*), &writes[3]) &&
         make_range(workspace.matrix_pointers, batch.batch_size, sizeof(double*), &writes[4]) &&
         make_range(workspace.info_a, batch.batch_size, sizeof(int), &writes[5]) &&
         make_range(workspace.info_b, batch.batch_size, sizeof(int), &writes[6]) &&
         make_range(workspace.eligible, batch.batch_size, sizeof(std::uint8_t), &writes[7]) &&
         make_range(workspace.sequence_active, 1, sizeof(std::uint32_t), &writes[8]) &&
         make_byte_range(workspace.solver_device_workspace, workspace.solver_device_workspace_bytes,
                         &writes[9]) &&
         make_byte_range(workspace.solver_host_workspace, workspace.solver_host_workspace_bytes,
                         &writes[10]) &&
         make_range(cache.cholesky_factors, batch.total_matrix_elements, sizeof(double),
                    &writes[11]) &&
         make_range(cache.geometry_generations, batch.batch_size, sizeof(std::uint64_t),
                    &writes[12]) &&
         make_range(cache.factor_statuses, batch.batch_size, sizeof(std::uint32_t), &writes[13]) &&
         make_range(system_errors, batch.batch_size, sizeof(std::uint32_t), &writes[14]) &&
         make_range(device_error, 1, sizeof(std::uint32_t), &writes[15]) &&
         pairwise_disjoint(reads) && pairwise_disjoint(writes) && disjoint_sets(reads, writes);
}

bool valid_solve_ranges(const Gfn2EigensolverDeviceBatch& batch,
                        const Gfn2EigensolverOverlapCache& cache, const double* hamiltonians,
                        const Gfn2EigensolverDeviceWorkspace& workspace,
                        const Gfn2EigensolverDeviceResults& results, std::uint32_t* system_errors,
                        std::uint32_t* device_error) noexcept {
  std::array<AddressRange, 8> reads{};
  std::array<AddressRange, 15> writes{};
  return make_range(batch.orbital_offsets, batch.batch_size + 1, sizeof(std::int64_t), &reads[0]) &&
         make_range(batch.matrix_offsets, batch.batch_size + 1, sizeof(std::int64_t), &reads[1]) &&
         make_range(batch.bucket_systems, batch.batch_size, sizeof(std::int32_t), &reads[2]) &&
         make_range(batch.active, batch.batch_size, sizeof(std::uint8_t), &reads[3]) &&
         make_range(cache.cholesky_factors, batch.total_matrix_elements, sizeof(double),
                    &reads[4]) &&
         make_range(cache.geometry_generations, batch.batch_size, sizeof(std::uint64_t),
                    &reads[5]) &&
         make_range(cache.factor_statuses, batch.batch_size, sizeof(std::uint32_t), &reads[6]) &&
         make_range(hamiltonians, batch.total_matrix_elements, sizeof(double), &reads[7]) &&
         make_range(workspace.matrix_scratch_a, batch.total_matrix_elements, sizeof(double),
                    &writes[0]) &&
         make_range(workspace.matrix_scratch_b, batch.total_matrix_elements, sizeof(double),
                    &writes[1]) &&
         make_range(workspace.eigenvalue_scratch, batch.total_orbitals, sizeof(double),
                    &writes[2]) &&
         make_range(workspace.factor_pointers, batch.batch_size, sizeof(double*), &writes[3]) &&
         make_range(workspace.matrix_pointers, batch.batch_size, sizeof(double*), &writes[4]) &&
         make_range(workspace.info_a, batch.batch_size, sizeof(int), &writes[5]) &&
         make_range(workspace.info_b, batch.batch_size, sizeof(int), &writes[6]) &&
         make_range(workspace.eligible, batch.batch_size, sizeof(std::uint8_t), &writes[7]) &&
         make_range(workspace.sequence_active, 1, sizeof(std::uint32_t), &writes[8]) &&
         make_byte_range(workspace.solver_device_workspace, workspace.solver_device_workspace_bytes,
                         &writes[9]) &&
         make_byte_range(workspace.solver_host_workspace, workspace.solver_host_workspace_bytes,
                         &writes[10]) &&
         make_range(results.eigenvalues, batch.total_orbitals, sizeof(double), &writes[11]) &&
         make_range(results.coefficients, batch.total_matrix_elements, sizeof(double),
                    &writes[12]) &&
         make_range(system_errors, batch.batch_size, sizeof(std::uint32_t), &writes[13]) &&
         make_range(device_error, 1, sizeof(std::uint32_t), &writes[14]) &&
         pairwise_disjoint(reads) && pairwise_disjoint(writes) && disjoint_sets(reads, writes);
}

__device__ bool system_is_clear(const std::uint32_t* system_errors, std::int64_t system) {
  return atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) ==
         static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kSuccess);
}

__device__ std::uint64_t load_geometry_generation(GeometryGenerationSource source) {
  return source.device == nullptr
             ? source.scalar
             : atomicAdd(
                   reinterpret_cast<unsigned long long*>(const_cast<std::uint64_t*>(source.device)),
                   0ULL);
}

__device__ void record_global_error(std::uint32_t* device_error, Gfn2EigensolverDeviceError error) {
  atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    std::uint32_t* device_error, Gfn2EigensolverDeviceError error) {
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  if (atomicCAS(system_errors + system,
                static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kSuccess),
                code) == static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kSuccess)) {
    record_global_error(device_error, error);
  }
}

__device__ bool valid_closed_range(std::int64_t begin, std::int64_t end, std::int64_t total) {
  return begin >= 0 && end >= begin && end <= total;
}

__device__ bool symmetric_input_is_valid(const double* matrix, std::int64_t n, double tolerance,
                                         bool* finite) {
  bool symmetric = true;
  *finite = true;
  for (std::int64_t row = 0; row < n; ++row) {
    for (std::int64_t column = 0; column < n; ++column) {
      const double value = matrix[row * n + column];
      if (!isfinite(value)) {
        *finite = false;
      }
      if (column < row) {
        const double transpose = matrix[column * n + row];
        const double scale = fmax(1.0, fmax(fabs(value), fabs(transpose)));
        if (!isfinite(transpose)) {
          *finite = false;
        } else if (isfinite(value) && fabs(value - transpose) > tolerance * scale) {
          symmetric = false;
        }
      }
    }
  }
  return symmetric;
}

/*
 * Preserve the sticky-error state at call entry. Later numerical failures are
 * deliberately per system and must not suppress healthy peers in this call.
 */
__global__ void capture_sequence_kernel(const std::uint32_t* device_error,
                                        std::uint32_t* sequence_active) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    sequence_active[0] = atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                                 static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kSuccess)
                             ? 1u
                             : 0u;
  }
}

/*
 * The host plan validates bucket slices, while this device preflight validates
 * the device-resident permutation. A duplicate or out-of-range entry can cause
 * provider pointer races, so it is a global fail-closed plan error rather than
 * a recoverable per-system failure. info_a is unpublished scratch at this point.
 */
__global__ void validate_bucket_map_kernel(Gfn2EigensolverDeviceBatch batch, int* visit_counts,
                                           std::uint32_t* sequence_active,
                                           std::uint32_t* device_error) {
  const std::int64_t slot = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= batch.bucket_system_count || atomicAdd(sequence_active, 0u) == 0u) {
    return;
  }
  const std::int32_t system = batch.bucket_systems[slot];
  bool invalid = system < 0 || system >= batch.batch_size;
  if (!invalid && atomicAdd(visit_counts + system, 1) != 0) {
    invalid = true;
  }
  if (invalid) {
    record_global_error(device_error, Gfn2EigensolverDeviceError::kInvalidBucketMap);
    atomicExch(sequence_active, 0u);
  }
}

__global__ void prepare_overlap_bucket_kernel(Gfn2EigensolverDeviceBatch batch,
                                              Gfn2EigensolverBucket bucket, const double* overlap,
                                              double symmetry_tolerance,
                                              Gfn2EigensolverDeviceWorkspace workspace,
                                              std::uint32_t* system_errors,
                                              std::uint32_t* device_error) {
  const std::int64_t slot = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t bucket_slot = bucket.system_index_offset + slot;
  const std::int64_t matrix_stride = static_cast<std::int64_t>(bucket.orbital_count) *
                                     static_cast<std::int64_t>(bucket.orbital_count);
  const std::int64_t scratch_begin = bucket.matrix_scratch_offset + slot * matrix_stride;

  __shared__ std::int64_t system;
  __shared__ std::int64_t input_begin;
  __shared__ int valid;
  if (threadIdx.x == 0) {
    system = batch.bucket_systems[bucket_slot];
    input_begin = 0;
    valid = 0;
    workspace.eligible[bucket_slot] = 0u;
    workspace.factor_pointers[bucket_slot] = workspace.matrix_scratch_a + scratch_begin;
    workspace.matrix_pointers[bucket_slot] = workspace.matrix_scratch_b + scratch_begin;
    if (atomicAdd(workspace.sequence_active, 0u) == 0u) {
      system = -1;
    } else if (system < 0 || system >= batch.batch_size) {
      record_global_error(device_error, Gfn2EigensolverDeviceError::kInvalidBucketMap);
    } else if (system_is_clear(system_errors, system)) {
      const std::uint8_t active = batch.active[system];
      if (active != 0u && active != 1u) {
        record_system_error(system_errors, system, device_error,
                            Gfn2EigensolverDeviceError::kInvalidActiveMask);
      } else if (active == 1u) {
        const std::int64_t orbital_begin = batch.orbital_offsets[system];
        const std::int64_t orbital_end = batch.orbital_offsets[system + 1];
        const std::int64_t matrix_begin = batch.matrix_offsets[system];
        const std::int64_t matrix_end = batch.matrix_offsets[system + 1];
        if (!valid_closed_range(orbital_begin, orbital_end, batch.total_orbitals) ||
            !valid_closed_range(matrix_begin, matrix_end, batch.total_matrix_elements) ||
            orbital_end - orbital_begin != bucket.orbital_count ||
            matrix_end - matrix_begin != matrix_stride) {
          record_system_error(system_errors, system, device_error,
                              Gfn2EigensolverDeviceError::kInvalidOffsets);
        } else {
          bool finite = true;
          const bool symmetric = symmetric_input_is_valid(
              overlap + matrix_begin, bucket.orbital_count, symmetry_tolerance, &finite);
          if (!finite) {
            record_system_error(system_errors, system, device_error,
                                Gfn2EigensolverDeviceError::kNonfiniteOverlap);
          } else if (!symmetric) {
            record_system_error(system_errors, system, device_error,
                                Gfn2EigensolverDeviceError::kNonsymmetricOverlap);
          } else {
            input_begin = matrix_begin;
            valid = 1;
            workspace.eligible[bucket_slot] = 1u;
          }
        }
      }
    }
  }
  __syncthreads();

  for (std::int64_t index = threadIdx.x; index < matrix_stride; index += blockDim.x) {
    const std::int64_t row = index % bucket.orbital_count;
    const std::int64_t column = index / bucket.orbital_count;
    double value = row == column ? 1.0 : 0.0;
    if (valid != 0) {
      const double first = overlap[input_begin + row * bucket.orbital_count + column];
      const double second = overlap[input_begin + column * bucket.orbital_count + row];
      value = row == column ? first : 0.5 * first + 0.5 * second;
    }
    workspace.matrix_scratch_a[scratch_begin + index] = value;
    workspace.matrix_scratch_b[scratch_begin + index] = value;
  }
}

__global__ void finalize_overlap_bucket_kernel(
    Gfn2EigensolverDeviceBatch batch, Gfn2EigensolverBucket bucket, std::uint64_t scalar_generation,
    const std::uint64_t* device_generation, double minimum_overlap_rcond,
    Gfn2EigensolverDeviceWorkspace workspace, Gfn2EigensolverOverlapCache cache,
    std::uint32_t* system_errors, std::uint32_t* device_error,
    Gfn2EigensolverFactorCachePolicy cache_policy) {
  const std::int64_t slot = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t bucket_slot = bucket.system_index_offset + slot;
  const std::int64_t system = batch.bucket_systems[bucket_slot];
  if (atomicAdd(workspace.sequence_active, 0u) == 0u || system < 0 || system >= batch.batch_size ||
      batch.active[system] != 1u) {
    return;
  }
  const std::int64_t matrix_stride = static_cast<std::int64_t>(bucket.orbital_count) *
                                     static_cast<std::int64_t>(bucket.orbital_count);
  const std::int64_t matrix_begin = bucket.matrix_scratch_offset + slot * matrix_stride;
  const std::int64_t eigen_begin = bucket.orbital_scratch_offset + slot * bucket.orbital_count;

  __shared__ int publish;
  if (threadIdx.x == 0) {
    const std::uint64_t geometry_generation =
        load_geometry_generation({scalar_generation, device_generation});
    publish =
        workspace.eligible[bucket_slot] == 1u && system_is_clear(system_errors, system) ? 1 : 0;
    Gfn2EigensolverDeviceError failure = Gfn2EigensolverDeviceError::kSuccess;
    if (publish != 0 && geometry_generation == 0u) {
      failure = Gfn2EigensolverDeviceError::kStaleOverlapCache;
    } else if (publish != 0 && device_generation != nullptr &&
               geometry_generation <= cache.geometry_generations[system]) {
      /* A replay that did not advance the shared epoch must never relabel a
       * changed overlap with an already committed generation. */
      failure = Gfn2EigensolverDeviceError::kStaleOverlapCache;
    } else if (publish != 0 && workspace.info_a[bucket_slot] != 0) {
      failure = Gfn2EigensolverDeviceError::kOverlapConditionEstimateFailed;
    } else if (publish != 0 && workspace.info_b[bucket_slot] != 0) {
      failure = Gfn2EigensolverDeviceError::kOverlapNotPositiveDefinite;
    } else if (publish != 0) {
      const double minimum = workspace.eigenvalue_scratch[eigen_begin];
      const double maximum = workspace.eigenvalue_scratch[eigen_begin + bucket.orbital_count - 1];
      const double reciprocal_condition = minimum / maximum;
      if (!isfinite(minimum) || !isfinite(maximum) || !(minimum > 0.0) || !(maximum >= minimum)) {
        failure = Gfn2EigensolverDeviceError::kOverlapNotPositiveDefinite;
      } else if (!isfinite(reciprocal_condition) || reciprocal_condition < minimum_overlap_rcond) {
        failure = Gfn2EigensolverDeviceError::kOverlapIllConditioned;
      }
    }
    if (failure != Gfn2EigensolverDeviceError::kSuccess) {
      record_system_error(system_errors, system, device_error, failure);
      publish = 0;
    }
    if (publish != 0) {
      cache.geometry_generations[system] = geometry_generation;
      cache.factor_statuses[system] =
          static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kSuccess);
    } else if (cache_policy == Gfn2EigensolverFactorCachePolicy::kPublishFailure) {
      cache.geometry_generations[system] = geometry_generation;
      cache.factor_statuses[system] = atomicAdd(system_errors + system, 0u);
    }
  }
  __syncthreads();
  if (publish == 0) {
    return;
  }
  for (std::int64_t index = threadIdx.x; index < matrix_stride; index += blockDim.x) {
    const std::int64_t row = index % bucket.orbital_count;
    const std::int64_t column = index / bucket.orbital_count;
    cache.cholesky_factors[matrix_begin + index] =
        row >= column ? workspace.matrix_scratch_a[matrix_begin + index] : 0.0;
  }
}

__global__ void prepare_solve_bucket_kernel(
    Gfn2EigensolverDeviceBatch batch, Gfn2EigensolverBucket bucket,
    Gfn2EigensolverOverlapCache cache, std::uint64_t geometry_generation,
    const double* hamiltonians, double symmetry_tolerance, Gfn2EigensolverDeviceWorkspace workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t slot = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t bucket_slot = bucket.system_index_offset + slot;
  const std::int64_t matrix_stride = static_cast<std::int64_t>(bucket.orbital_count) *
                                     static_cast<std::int64_t>(bucket.orbital_count);
  const std::int64_t scratch_begin = bucket.matrix_scratch_offset + slot * matrix_stride;

  __shared__ std::int64_t system;
  __shared__ std::int64_t input_begin;
  __shared__ int valid;
  if (threadIdx.x == 0) {
    system = batch.bucket_systems[bucket_slot];
    input_begin = 0;
    valid = 0;
    workspace.eligible[bucket_slot] = 0u;
    workspace.factor_pointers[bucket_slot] = workspace.matrix_scratch_a + scratch_begin;
    workspace.matrix_pointers[bucket_slot] = workspace.matrix_scratch_b + scratch_begin;
    if (atomicAdd(workspace.sequence_active, 0u) == 0u) {
      system = -1;
    } else if (system < 0 || system >= batch.batch_size) {
      record_global_error(device_error, Gfn2EigensolverDeviceError::kInvalidBucketMap);
    } else if (system_is_clear(system_errors, system)) {
      const std::uint8_t active = batch.active[system];
      if (active != 0u && active != 1u) {
        record_system_error(system_errors, system, device_error,
                            Gfn2EigensolverDeviceError::kInvalidActiveMask);
      } else if (active == 1u) {
        const std::int64_t orbital_begin = batch.orbital_offsets[system];
        const std::int64_t orbital_end = batch.orbital_offsets[system + 1];
        const std::int64_t matrix_begin = batch.matrix_offsets[system];
        const std::int64_t matrix_end = batch.matrix_offsets[system + 1];
        if (!valid_closed_range(orbital_begin, orbital_end, batch.total_orbitals) ||
            !valid_closed_range(matrix_begin, matrix_end, batch.total_matrix_elements) ||
            orbital_end - orbital_begin != bucket.orbital_count ||
            matrix_end - matrix_begin != matrix_stride) {
          record_system_error(system_errors, system, device_error,
                              Gfn2EigensolverDeviceError::kInvalidOffsets);
        } else if (cache.geometry_generations[system] != geometry_generation ||
                   cache.factor_statuses[system] !=
                       static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kSuccess)) {
          record_system_error(system_errors, system, device_error,
                              Gfn2EigensolverDeviceError::kStaleOverlapCache);
        } else {
          bool finite = true;
          const bool symmetric = symmetric_input_is_valid(
              hamiltonians + matrix_begin, bucket.orbital_count, symmetry_tolerance, &finite);
          bool factor_finite = true;
          for (std::int64_t diagonal = 0; diagonal < bucket.orbital_count; ++diagonal) {
            const double value =
                cache.cholesky_factors[scratch_begin + diagonal * bucket.orbital_count + diagonal];
            factor_finite = factor_finite && isfinite(value) && value > 0.0;
          }
          if (!finite) {
            record_system_error(system_errors, system, device_error,
                                Gfn2EigensolverDeviceError::kNonfiniteHamiltonian);
          } else if (!symmetric) {
            record_system_error(system_errors, system, device_error,
                                Gfn2EigensolverDeviceError::kNonsymmetricHamiltonian);
          } else if (!factor_finite) {
            record_system_error(system_errors, system, device_error,
                                Gfn2EigensolverDeviceError::kStaleOverlapCache);
          } else {
            input_begin = matrix_begin;
            valid = 1;
            workspace.eligible[bucket_slot] = 1u;
          }
        }
      }
    }
  }
  __syncthreads();

  for (std::int64_t index = threadIdx.x; index < matrix_stride; index += blockDim.x) {
    const std::int64_t row = index % bucket.orbital_count;
    const std::int64_t column = index / bucket.orbital_count;
    double factor = row == column ? 1.0 : 0.0;
    double hamiltonian = row == column ? 1.0 : 0.0;
    if (valid != 0) {
      factor = cache.cholesky_factors[scratch_begin + index];
      const double first = hamiltonians[input_begin + row * bucket.orbital_count + column];
      const double second = hamiltonians[input_begin + column * bucket.orbital_count + row];
      hamiltonian = row == column ? first : 0.5 * first + 0.5 * second;
    }
    workspace.matrix_scratch_a[scratch_begin + index] = factor;
    workspace.matrix_scratch_b[scratch_begin + index] = hamiltonian;
  }
}

__global__ void symmetrize_transformed_bucket_kernel(Gfn2EigensolverDeviceBatch batch,
                                                     Gfn2EigensolverBucket bucket,
                                                     Gfn2EigensolverDeviceWorkspace workspace,
                                                     std::uint32_t* system_errors,
                                                     std::uint32_t* device_error) {
  const std::int64_t slot = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t bucket_slot = bucket.system_index_offset + slot;
  const std::int64_t system = batch.bucket_systems[bucket_slot];
  const std::int64_t matrix_stride = static_cast<std::int64_t>(bucket.orbital_count) *
                                     static_cast<std::int64_t>(bucket.orbital_count);
  const std::int64_t matrix_begin = bucket.matrix_scratch_offset + slot * matrix_stride;
  if (system < 0 || system >= batch.batch_size) {
    return;
  }

  __shared__ int valid;
  if (threadIdx.x == 0) {
    valid = workspace.eligible[bucket_slot] == 1u && system_is_clear(system_errors, system) ? 1 : 0;
    if (valid != 0) {
      for (std::int64_t index = 0; index < matrix_stride; ++index) {
        if (!isfinite(workspace.matrix_scratch_b[matrix_begin + index])) {
          record_system_error(system_errors, system, device_error,
                              Gfn2EigensolverDeviceError::kNonfiniteEigenpair);
          workspace.eligible[bucket_slot] = 0u;
          valid = 0;
          break;
        }
      }
    }
  }
  __syncthreads();
  /* One lower-triangle thread owns both members of each transpose pair. */
  for (std::int64_t index = threadIdx.x; index < matrix_stride; index += blockDim.x) {
    const std::int64_t row = index % bucket.orbital_count;
    const std::int64_t column = index / bucket.orbital_count;
    if (row < column) {
      continue;
    }
    double value = row == column ? 1.0 : 0.0;
    if (valid != 0) {
      const double first =
          workspace.matrix_scratch_b[matrix_begin + row + column * bucket.orbital_count];
      const double second =
          workspace.matrix_scratch_b[matrix_begin + column + row * bucket.orbital_count];
      value = row == column ? first : 0.5 * first + 0.5 * second;
    }
    workspace.matrix_scratch_b[matrix_begin + row + column * bucket.orbital_count] = value;
    workspace.matrix_scratch_b[matrix_begin + column + row * bucket.orbital_count] = value;
  }
}

__global__ void publish_eigensystems_bucket_kernel(Gfn2EigensolverDeviceBatch batch,
                                                   Gfn2EigensolverBucket bucket,
                                                   Gfn2EigensolverDeviceWorkspace workspace,
                                                   Gfn2EigensolverDeviceResults results,
                                                   std::uint32_t* system_errors,
                                                   std::uint32_t* device_error) {
  const std::int64_t slot = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t bucket_slot = bucket.system_index_offset + slot;
  const std::int64_t system = batch.bucket_systems[bucket_slot];
  if (system < 0 || system >= batch.batch_size || batch.active[system] != 1u) {
    return;
  }
  const std::int64_t matrix_stride = static_cast<std::int64_t>(bucket.orbital_count) *
                                     static_cast<std::int64_t>(bucket.orbital_count);
  const std::int64_t scratch_matrix_begin = bucket.matrix_scratch_offset + slot * matrix_stride;
  const std::int64_t scratch_orbital_begin =
      bucket.orbital_scratch_offset + slot * bucket.orbital_count;
  const std::int64_t output_matrix_begin = batch.matrix_offsets[system];
  const std::int64_t output_orbital_begin = batch.orbital_offsets[system];

  __shared__ int publish;
  if (threadIdx.x == 0) {
    publish =
        workspace.eligible[bucket_slot] == 1u && system_is_clear(system_errors, system) ? 1 : 0;
    if (publish != 0 && workspace.info_a[bucket_slot] != 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2EigensolverDeviceError::kEigensolverFailed);
      publish = 0;
    }
    if (publish != 0) {
      for (std::int64_t orbital = 0; orbital < bucket.orbital_count; ++orbital) {
        if (!isfinite(workspace.eigenvalue_scratch[scratch_orbital_begin + orbital])) {
          publish = 0;
          break;
        }
      }
      for (std::int64_t index = 0; publish != 0 && index < matrix_stride; ++index) {
        if (!isfinite(workspace.matrix_scratch_b[scratch_matrix_begin + index])) {
          publish = 0;
        }
      }
      if (publish == 0) {
        record_system_error(system_errors, system, device_error,
                            Gfn2EigensolverDeviceError::kNonfiniteEigenpair);
      }
    }
  }
  __syncthreads();
  if (publish == 0) {
    return;
  }
  for (std::int64_t orbital = threadIdx.x; orbital < bucket.orbital_count; orbital += blockDim.x) {
    results.eigenvalues[output_orbital_begin + orbital] =
        workspace.eigenvalue_scratch[scratch_orbital_begin + orbital];
  }
  for (std::int64_t index = threadIdx.x; index < matrix_stride; index += blockDim.x) {
    const std::int64_t ao = index / bucket.orbital_count;
    const std::int64_t orbital = index - ao * bucket.orbital_count;
    results.coefficients[output_matrix_begin + index] =
        workspace.matrix_scratch_b[scratch_matrix_begin + ao + orbital * bucket.orbital_count];
  }
}

Gfn2EigensolverLaunchResult check_kernel_launch() noexcept {
  const cudaError_t status = cudaGetLastError();
  return status == cudaSuccess ? launch_success() : cuda_failure(status);
}

Gfn2EigensolverLaunchResult prepare_launch_sequence(const Gfn2EigensolverDeviceBatch& batch,
                                                    const Gfn2EigensolverDeviceWorkspace& workspace,
                                                    std::uint32_t* device_error,
                                                    cudaStream_t stream) noexcept {
  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace.sequence_active);
  Gfn2EigensolverLaunchResult result = check_kernel_launch();
  if (!result.success()) {
    return result;
  }
  const cudaError_t clear_status = cudaMemsetAsync(
      workspace.info_a, 0, static_cast<std::size_t>(batch.batch_size) * sizeof(int), stream);
  if (clear_status != cudaSuccess) {
    return cuda_failure(clear_status);
  }
  constexpr int kMapThreads = 256;
  const std::int64_t block_count = (batch.batch_size + kMapThreads - 1) / kMapThreads;
  validate_bucket_map_kernel<<<static_cast<unsigned int>(block_count), kMapThreads, 0, stream>>>(
      batch, workspace.info_a, workspace.sequence_active, device_error);
  return check_kernel_launch();
}

Gfn2EigensolverLaunchResult configure_solver(cusolverDnHandle_t solver,
                                             cudaStream_t stream) noexcept {
  const cusolverStatus_t status = cusolverDnSetStream(solver, stream);
  return status == CUSOLVER_STATUS_SUCCESS ? launch_success() : cusolver_failure(status);
}

Gfn2EigensolverLaunchResult configure_blas(cublasHandle_t blas, cudaStream_t stream,
                                           bool deterministic_debug) noexcept {
  cublasStatus_t status = cublasSetStream(blas, stream);
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status =
        cublasSetMathMode(blas, deterministic_debug ? CUBLAS_PEDANTIC_MATH : CUBLAS_DEFAULT_MATH);
  }
  return status == CUBLAS_STATUS_SUCCESS ? launch_success() : cublas_failure(status);
}

Gfn2EigensolverLaunchResult symmetric_eigensolve(
    cusolverDnHandle_t solver, cusolverDnParams_t parameters, cusolverEigMode_t vectors,
    const Gfn2EigensolverBucket& bucket, double* matrices, double* eigenvalues,
    const Gfn2EigensolverDeviceWorkspace& workspace, int* info) noexcept {
  const cusolverStatus_t status = cusolverDnXsyevBatched(
      solver, parameters, vectors, CUBLAS_FILL_MODE_LOWER, bucket.orbital_count, CUDA_R_64F,
      matrices, bucket.orbital_count, CUDA_R_64F, eigenvalues, CUDA_R_64F,
      workspace.solver_device_workspace, workspace.solver_device_workspace_bytes,
      workspace.solver_host_workspace, workspace.solver_host_workspace_bytes, info,
      bucket.system_count);
  return status == CUSOLVER_STATUS_SUCCESS ? launch_success() : cusolver_failure(status);
}

Gfn2EigensolverLaunchResult triangular_solve(cublasHandle_t blas, cublasSideMode_t side,
                                             cublasOperation_t operation,
                                             const Gfn2EigensolverBucket& bucket,
                                             double** factor_pointers,
                                             double** matrix_pointers) noexcept {
  const auto factors = reinterpret_cast<const double* const*>(factor_pointers);
  const cublasStatus_t status = cublasDtrsmBatched(
      blas, side, CUBLAS_FILL_MODE_LOWER, operation, CUBLAS_DIAG_NON_UNIT, bucket.orbital_count,
      bucket.orbital_count, &kOne, factors, bucket.orbital_count, matrix_pointers,
      bucket.orbital_count, bucket.system_count);
  return status == CUBLAS_STATUS_SUCCESS ? launch_success() : cublas_failure(status);
}

}  // namespace

Gfn2EigensolverLaunchResult query_gfn2_eigensolver_bucket_workspace_cuda(
    cusolverDnHandle_t solver, cusolverDnParams_t parameters, const Gfn2EigensolverBucket& bucket,
    const double* device_matrix, const double* device_eigenvalues,
    Gfn2EigensolverWorkspaceRequirements& requirements) noexcept {
  if (solver == nullptr || parameters == nullptr || bucket.orbital_count <= 0 ||
      bucket.system_count <= 0 || !is_aligned(device_matrix, alignof(double)) ||
      !is_aligned(device_eigenvalues, alignof(double))) {
    return invalid_argument();
  }
  std::int64_t matrix_stride = 0;
  std::int64_t matrix_span = 0;
  if (!checked_multiply(bucket.orbital_count, bucket.orbital_count, &matrix_stride) ||
      !checked_multiply(matrix_stride, bucket.system_count, &matrix_span) ||
      matrix_stride > std::numeric_limits<int>::max() ||
      matrix_span > std::numeric_limits<int>::max()) {
    return invalid_argument();
  }
  std::size_t maximum_device = 0u;
  std::size_t maximum_host = 0u;
  for (const cusolverEigMode_t mode : {CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR}) {
    std::size_t device_bytes = 0u;
    std::size_t host_bytes = 0u;
    const cusolverStatus_t status = cusolverDnXsyevBatched_bufferSize(
        solver, parameters, mode, CUBLAS_FILL_MODE_LOWER, bucket.orbital_count, CUDA_R_64F,
        device_matrix, bucket.orbital_count, CUDA_R_64F, device_eigenvalues, CUDA_R_64F,
        &device_bytes, &host_bytes, bucket.system_count);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      return cusolver_failure(status);
    }
    maximum_device = std::max(maximum_device, device_bytes);
    maximum_host = std::max(maximum_host, host_bytes);
  }
  requirements.solver_device_workspace_bytes =
      std::max(requirements.solver_device_workspace_bytes, maximum_device);
  requirements.solver_host_workspace_bytes =
      std::max(requirements.solver_host_workspace_bytes, maximum_host);
  return launch_success();
}

cudaError_t reset_gfn2_eigensolver_device_errors_cuda(std::int64_t batch_size,
                                                      std::uint32_t* system_errors,
                                                      std::uint32_t* device_error,
                                                      cudaStream_t stream) noexcept {
  if (batch_size <= 0 || batch_size > std::numeric_limits<int>::max() ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
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
  const std::size_t bytes = static_cast<std::size_t>(batch_size) * sizeof(*system_errors);
  cudaError_t status = cudaMemsetAsync(system_errors, 0, bytes, stream);
  return status == cudaSuccess ? cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream)
                               : status;
}

Gfn2EigensolverLaunchResult factor_overlap_impl(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const double* overlap, GeometryGenerationSource generation_source,
    const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver, cusolverDnParams_t parameters,
    const Gfn2EigensolverDeviceWorkspace& workspace, const Gfn2EigensolverOverlapCache& cache,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream,
    Gfn2EigensolverFactorCachePolicy cache_policy) noexcept {
  if (!valid_bucket_plan(batch, buckets, bucket_count) || !valid_options(options) ||
      !valid_workspace(batch, workspace) || !valid_cache(batch, cache) ||
      !valid_factor_ranges(batch, overlap, generation_source.device, workspace, cache,
                           system_errors, device_error) ||
      (generation_source.device == nullptr ? generation_source.scalar == 0u
                                           : generation_source.scalar != 0u) ||
      (generation_source.device != nullptr &&
       !is_aligned(generation_source.device, alignof(std::uint64_t))) ||
      solver == nullptr || parameters == nullptr ||
      (cache_policy != Gfn2EigensolverFactorCachePolicy::kPublishFailure &&
       cache_policy != Gfn2EigensolverFactorCachePolicy::kPreservePriorOnFailure) ||
      !is_aligned(overlap, alignof(double)) || !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return invalid_argument();
  }
  Gfn2EigensolverLaunchResult result = configure_solver(solver, stream);
  if (!result.success()) {
    return result;
  }
  result = prepare_launch_sequence(batch, workspace, device_error, stream);
  if (!result.success()) {
    return result;
  }

  for (std::int64_t bucket_index = 0; bucket_index < bucket_count; ++bucket_index) {
    const Gfn2EigensolverBucket bucket = buckets[bucket_index];
    const std::int64_t matrix_begin = bucket.matrix_scratch_offset;
    const std::int64_t orbital_begin = bucket.orbital_scratch_offset;
    const std::int64_t info_begin = bucket.system_index_offset;
    prepare_overlap_bucket_kernel<<<static_cast<unsigned int>(bucket.system_count),
                                    kThreadsPerSystem, 0, stream>>>(
        batch, bucket, overlap, options.symmetry_tolerance, workspace, system_errors, device_error);
    result = check_kernel_launch();
    if (!result.success()) {
      return result;
    }
    result = symmetric_eigensolve(solver, parameters, CUSOLVER_EIG_MODE_NOVECTOR, bucket,
                                  workspace.matrix_scratch_b + matrix_begin,
                                  workspace.eigenvalue_scratch + orbital_begin, workspace,
                                  workspace.info_a + info_begin);
    if (!result.success()) {
      return result;
    }
    const cusolverStatus_t factor_status =
        cusolverDnDpotrfBatched(solver, CUBLAS_FILL_MODE_LOWER, bucket.orbital_count,
                                workspace.factor_pointers + info_begin, bucket.orbital_count,
                                workspace.info_b + info_begin, bucket.system_count);
    if (factor_status != CUSOLVER_STATUS_SUCCESS) {
      return cusolver_failure(factor_status);
    }
    finalize_overlap_bucket_kernel<<<static_cast<unsigned int>(bucket.system_count),
                                     kThreadsPerSystem, 0, stream>>>(
        batch, bucket, generation_source.scalar, generation_source.device,
        options.minimum_overlap_rcond, workspace, cache, system_errors, device_error, cache_policy);
    result = check_kernel_launch();
    if (!result.success()) {
      return result;
    }
  }
  return launch_success();
}

Gfn2EigensolverLaunchResult factor_gfn2_overlap_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const double* overlap, std::uint64_t geometry_generation,
    const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver, cusolverDnParams_t parameters,
    const Gfn2EigensolverDeviceWorkspace& workspace, const Gfn2EigensolverOverlapCache& cache,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream,
    Gfn2EigensolverFactorCachePolicy cache_policy) noexcept {
  return factor_overlap_impl(batch, buckets, bucket_count, overlap, {geometry_generation, nullptr},
                             options, solver, parameters, workspace, cache, system_errors,
                             device_error, stream, cache_policy);
}

Gfn2EigensolverLaunchResult factor_gfn2_overlap_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const double* overlap, const Gfn2GeometryEpochDevice& geometry_epoch,
    const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver, cusolverDnParams_t parameters,
    const Gfn2EigensolverDeviceWorkspace& workspace, const Gfn2EigensolverOverlapCache& cache,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream,
    Gfn2EigensolverFactorCachePolicy cache_policy) noexcept {
  if (geometry_epoch.value_elements != 1 || geometry_epoch.plan_token != batch.plan_token) {
    return invalid_argument();
  }
  return factor_overlap_impl(batch, buckets, bucket_count, overlap, {0u, geometry_epoch.value},
                             options, solver, parameters, workspace, cache, system_errors,
                             device_error, stream, cache_policy);
}

Gfn2EigensolverLaunchResult solve_gfn2_eigensystems_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const Gfn2EigensolverOverlapCache& cache,
    std::uint64_t geometry_generation, const double* hamiltonians,
    const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver, cusolverDnParams_t parameters,
    cublasHandle_t blas, const Gfn2EigensolverDeviceWorkspace& workspace,
    const Gfn2EigensolverDeviceResults& results, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (!valid_bucket_plan(batch, buckets, bucket_count) || !valid_options(options) ||
      !valid_workspace(batch, workspace) || !valid_cache(batch, cache) ||
      geometry_generation == 0u || solver == nullptr || parameters == nullptr || blas == nullptr ||
      results.plan_token != batch.plan_token ||
      results.eigenvalue_elements < batch.total_orbitals ||
      results.coefficient_elements < batch.total_matrix_elements ||
      !is_aligned(hamiltonians, alignof(double)) ||
      !is_aligned(results.eigenvalues, alignof(double)) ||
      !is_aligned(results.coefficients, alignof(double)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t)) ||
      !valid_solve_ranges(batch, cache, hamiltonians, workspace, results, system_errors,
                          device_error)) {
    return invalid_argument();
  }
  Gfn2EigensolverLaunchResult result = configure_solver(solver, stream);
  if (!result.success()) {
    return result;
  }
  result = prepare_launch_sequence(batch, workspace, device_error, stream);
  if (!result.success()) {
    return result;
  }
  result = configure_blas(blas, stream, options.deterministic_debug);
  if (!result.success()) {
    return result;
  }

  for (std::int64_t bucket_index = 0; bucket_index < bucket_count; ++bucket_index) {
    const Gfn2EigensolverBucket bucket = buckets[bucket_index];
    const std::int64_t matrix_begin = bucket.matrix_scratch_offset;
    const std::int64_t orbital_begin = bucket.orbital_scratch_offset;
    const std::int64_t info_begin = bucket.system_index_offset;
    prepare_solve_bucket_kernel<<<static_cast<unsigned int>(bucket.system_count), kThreadsPerSystem,
                                  0, stream>>>(batch, bucket, cache, geometry_generation,
                                               hamiltonians, options.symmetry_tolerance, workspace,
                                               system_errors, device_error);
    result = check_kernel_launch();
    if (!result.success()) {
      return result;
    }
    result = triangular_solve(blas, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, bucket,
                              workspace.factor_pointers + info_begin,
                              workspace.matrix_pointers + info_begin);
    if (!result.success()) {
      return result;
    }
    result = triangular_solve(blas, CUBLAS_SIDE_RIGHT, CUBLAS_OP_T, bucket,
                              workspace.factor_pointers + info_begin,
                              workspace.matrix_pointers + info_begin);
    if (!result.success()) {
      return result;
    }
    symmetrize_transformed_bucket_kernel<<<static_cast<unsigned int>(bucket.system_count),
                                           kThreadsPerSystem, 0, stream>>>(
        batch, bucket, workspace, system_errors, device_error);
    result = check_kernel_launch();
    if (!result.success()) {
      return result;
    }
    result = symmetric_eigensolve(solver, parameters, CUSOLVER_EIG_MODE_VECTOR, bucket,
                                  workspace.matrix_scratch_b + matrix_begin,
                                  workspace.eigenvalue_scratch + orbital_begin, workspace,
                                  workspace.info_a + info_begin);
    if (!result.success()) {
      return result;
    }
    result = triangular_solve(blas, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, bucket,
                              workspace.factor_pointers + info_begin,
                              workspace.matrix_pointers + info_begin);
    if (!result.success()) {
      return result;
    }
    publish_eigensystems_bucket_kernel<<<static_cast<unsigned int>(bucket.system_count),
                                         kThreadsPerSystem, 0, stream>>>(
        batch, bucket, workspace, results, system_errors, device_error);
    result = check_kernel_launch();
    if (!result.success()) {
      return result;
    }
  }
  return launch_success();
}

}  // namespace gpuxtb::detail::cuda
