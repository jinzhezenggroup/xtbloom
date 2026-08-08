#include <algorithm>
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <utility>

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

/* Validate one bucket borrowed from an already assembled batch. Unlike
 * valid_bucket_plan(), this deliberately does not require the bucket to cover
 * every system because the exact-capacity chain compacts buckets separately. */
bool valid_bucket_slice(const Gfn2EigensolverDeviceBatch& batch,
                        const Gfn2EigensolverBucket& bucket) noexcept {
  if (batch.batch_size <= 0 || batch.batch_size > std::numeric_limits<int>::max() ||
      batch.total_orbitals <= 0 || batch.total_matrix_elements <= 0 || batch.plan_token == 0u ||
      batch.orbital_offset_count != batch.batch_size + 1 ||
      batch.matrix_offset_count != batch.batch_size + 1 ||
      batch.bucket_system_count != batch.batch_size || batch.active_elements != batch.batch_size ||
      !is_aligned(batch.orbital_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.matrix_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.bucket_systems, alignof(std::int32_t)) ||
      !is_aligned(batch.active, alignof(std::uint8_t)) || bucket.orbital_count <= 0 ||
      bucket.system_count <= 0 || bucket.system_index_offset < 0 ||
      bucket.matrix_scratch_offset < 0 || bucket.orbital_scratch_offset < 0) {
    return false;
  }

  std::int64_t matrix_stride = 0;
  std::int64_t matrix_span = 0;
  std::int64_t orbital_span = 0;
  std::int64_t system_end = 0;
  std::int64_t matrix_end = 0;
  std::int64_t orbital_end = 0;
  return checked_multiply(bucket.orbital_count, bucket.orbital_count, &matrix_stride) &&
         checked_multiply(matrix_stride, bucket.system_count, &matrix_span) &&
         checked_multiply(bucket.orbital_count, bucket.system_count, &orbital_span) &&
         matrix_stride <= std::numeric_limits<int>::max() &&
         matrix_span <= std::numeric_limits<int>::max() &&
         orbital_span <= std::numeric_limits<int>::max() &&
         checked_add(bucket.system_index_offset, bucket.system_count, &system_end) &&
         checked_add(bucket.matrix_scratch_offset, matrix_span, &matrix_end) &&
         checked_add(bucket.orbital_scratch_offset, orbital_span, &orbital_end) &&
         system_end <= batch.bucket_system_count && matrix_end <= batch.total_matrix_elements &&
         orbital_end <= batch.total_orbitals;
}

bool valid_spin_bucket_plan(const Gfn2EigensolverDeviceBatch& batch,
                            const Gfn2WavefunctionLayoutView& layout,
                            const Gfn2EigensolverBucket* buckets,
                            std::int64_t bucket_count) noexcept {
  std::int64_t two_batch = 0;
  std::int64_t two_orbitals = 0;
  std::int64_t two_matrices = 0;
  if (!valid_bucket_plan(batch, buckets, bucket_count) ||
      !checked_multiply(batch.batch_size, 2, &two_batch) ||
      !checked_multiply(batch.total_orbitals, 2, &two_orbitals) ||
      !checked_multiply(batch.total_matrix_elements, 2, &two_matrices) ||
      layout.memory_space != Gfn2PlanMemorySpace::kCudaDevice ||
      layout.plan_token != batch.plan_token || layout.batch_size != batch.batch_size ||
      layout.total_spin_channels < batch.batch_size || layout.total_spin_channels > two_batch ||
      layout.total_spin_orbitals < batch.total_orbitals ||
      layout.total_spin_orbitals > two_orbitals ||
      layout.total_spin_matrix_elements < batch.total_matrix_elements ||
      layout.total_spin_matrix_elements > two_matrices ||
      layout.spin_channel_count != batch.batch_size ||
      layout.spin_channel_offset_count != batch.batch_size + 1 ||
      layout.spin_orbital_offset_count != batch.batch_size + 1 ||
      layout.spin_matrix_offset_count != batch.batch_size + 1 ||
      !is_aligned(layout.spin_channels, alignof(std::int32_t)) ||
      !is_aligned(layout.spin_channel_offsets, alignof(std::int64_t)) ||
      !is_aligned(layout.spin_orbital_offsets, alignof(std::int64_t)) ||
      !is_aligned(layout.spin_matrix_offsets, alignof(std::int64_t))) {
    return false;
  }

  std::int64_t counted_solves = 0;
  std::int64_t counted_orbitals = 0;
  std::int64_t counted_matrices = 0;
  for (std::int64_t bucket_index = 0; bucket_index < bucket_count; ++bucket_index) {
    const Gfn2EigensolverBucket& bucket = buckets[bucket_index];
    const std::int64_t n = bucket.orbital_count;
    const std::int64_t solve_count = bucket.solve_count;
    std::int64_t matrix_stride = 0;
    std::int64_t matrix_span = 0;
    std::int64_t orbital_span = 0;
    std::int64_t solve_end = 0;
    std::int64_t matrix_end = 0;
    std::int64_t orbital_end = 0;
    if (solve_count < bucket.system_count ||
        solve_count > 2LL * static_cast<std::int64_t>(bucket.system_count) ||
        !checked_multiply(n, n, &matrix_stride) ||
        !checked_multiply(matrix_stride, solve_count, &matrix_span) ||
        !checked_multiply(n, solve_count, &orbital_span) ||
        !checked_add(bucket.solve_index_offset, solve_count, &solve_end) ||
        !checked_add(bucket.spin_matrix_scratch_offset, matrix_span, &matrix_end) ||
        !checked_add(bucket.spin_orbital_scratch_offset, orbital_span, &orbital_end) ||
        solve_end > layout.total_spin_channels || matrix_end > layout.total_spin_matrix_elements ||
        orbital_end > layout.total_spin_orbitals ||
        !checked_add(counted_solves, solve_count, &counted_solves) ||
        !checked_add(counted_orbitals, orbital_span, &counted_orbitals) ||
        !checked_add(counted_matrices, matrix_span, &counted_matrices)) {
      return false;
    }
    for (std::int64_t other_index = 0; other_index < bucket_index; ++other_index) {
      const Gfn2EigensolverBucket& other = buckets[other_index];
      std::int64_t other_matrix_stride = 0;
      std::int64_t other_matrix_span = 0;
      std::int64_t other_orbital_span = 0;
      std::int64_t other_solve_end = 0;
      std::int64_t other_matrix_end = 0;
      std::int64_t other_orbital_end = 0;
      if (!checked_multiply(other.orbital_count, other.orbital_count, &other_matrix_stride) ||
          !checked_multiply(other_matrix_stride, other.solve_count, &other_matrix_span) ||
          !checked_multiply(other.orbital_count, other.solve_count, &other_orbital_span) ||
          !checked_add(other.solve_index_offset, other.solve_count, &other_solve_end) ||
          !checked_add(other.spin_matrix_scratch_offset, other_matrix_span, &other_matrix_end) ||
          !checked_add(other.spin_orbital_scratch_offset, other_orbital_span, &other_orbital_end)) {
        return false;
      }
      if (ranges_overlap(bucket.solve_index_offset, solve_end, other.solve_index_offset,
                         other_solve_end) ||
          ranges_overlap(bucket.spin_matrix_scratch_offset, matrix_end,
                         other.spin_matrix_scratch_offset, other_matrix_end) ||
          ranges_overlap(bucket.spin_orbital_scratch_offset, orbital_end,
                         other.spin_orbital_scratch_offset, other_orbital_end)) {
        return false;
      }
    }
  }
  return counted_solves == layout.total_spin_channels &&
         counted_orbitals == layout.total_spin_orbitals &&
         counted_matrices == layout.total_spin_matrix_elements;
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

bool valid_spin_workspace(const Gfn2EigensolverDeviceBatch& batch,
                          const Gfn2WavefunctionLayoutView& layout,
                          const Gfn2EigensolverDeviceWorkspace& workspace) noexcept {
  return valid_workspace(batch, workspace) &&
         workspace.matrix_a_elements >= layout.total_spin_matrix_elements &&
         workspace.matrix_b_elements >= layout.total_spin_matrix_elements &&
         workspace.eigenvalue_elements >= layout.total_spin_orbitals &&
         workspace.factor_pointer_elements >= layout.total_spin_channels &&
         workspace.matrix_pointer_elements >= layout.total_spin_channels &&
         workspace.info_a_elements >= layout.total_spin_channels &&
         workspace.info_b_elements >= layout.total_spin_channels &&
         workspace.eligible_elements >= layout.total_spin_channels;
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
  std::array<AddressRange, 19> writes{};
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
         make_range(workspace.compact_systems, workspace.compact_system_elements,
                    sizeof(std::int32_t), &writes[16]) &&
         make_range(workspace.compact_source_slots, workspace.compact_source_slot_elements,
                    sizeof(std::int32_t), &writes[17]) &&
         make_range(workspace.bucket_activity, workspace.bucket_activity_elements,
                    sizeof(Gfn2EigensolverBucketActivity), &writes[18]) &&
         pairwise_disjoint(reads) && pairwise_disjoint(writes) && disjoint_sets(reads, writes);
}

bool valid_solve_ranges(const Gfn2EigensolverDeviceBatch& batch,
                        const Gfn2EigensolverOverlapCache& cache, const double* hamiltonians,
                        const std::uint64_t* geometry_epoch,
                        const Gfn2EigensolverDeviceWorkspace& workspace,
                        const Gfn2EigensolverDeviceResults& results, std::uint32_t* system_errors,
                        std::uint32_t* device_error) noexcept {
  std::array<AddressRange, 9> reads{};
  std::array<AddressRange, 18> writes{};
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
         make_range(geometry_epoch, geometry_epoch == nullptr ? 0 : 1, sizeof(std::uint64_t),
                    &reads[8]) &&
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
         make_range(workspace.compact_systems, workspace.compact_system_elements,
                    sizeof(std::int32_t), &writes[15]) &&
         make_range(workspace.compact_source_slots, workspace.compact_source_slot_elements,
                    sizeof(std::int32_t), &writes[16]) &&
         make_range(workspace.bucket_activity, workspace.bucket_activity_elements,
                    sizeof(Gfn2EigensolverBucketActivity), &writes[17]) &&
         pairwise_disjoint(reads) && pairwise_disjoint(writes) && disjoint_sets(reads, writes);
}

bool valid_spin_solve_ranges(const Gfn2EigensolverDeviceBatch& batch,
                             const Gfn2WavefunctionLayoutView& layout,
                             const Gfn2EigensolverOverlapCache& cache, const double* hamiltonians,
                             const std::uint64_t* geometry_epoch,
                             const Gfn2EigensolverDeviceWorkspace& workspace,
                             const Gfn2EigensolverDeviceResults& results,
                             std::uint32_t* system_errors, std::uint32_t* device_error) noexcept {
  std::array<AddressRange, 13> reads{};
  std::array<AddressRange, 15> writes{};
  return make_range(batch.orbital_offsets, batch.batch_size + 1, sizeof(std::int64_t), &reads[0]) &&
         make_range(batch.matrix_offsets, batch.batch_size + 1, sizeof(std::int64_t), &reads[1]) &&
         make_range(batch.bucket_systems, batch.batch_size, sizeof(std::int32_t), &reads[2]) &&
         make_range(batch.active, batch.batch_size, sizeof(std::uint8_t), &reads[3]) &&
         make_range(layout.spin_channels, batch.batch_size, sizeof(std::int32_t), &reads[4]) &&
         make_range(layout.spin_channel_offsets, batch.batch_size + 1, sizeof(std::int64_t),
                    &reads[5]) &&
         make_range(layout.spin_orbital_offsets, batch.batch_size + 1, sizeof(std::int64_t),
                    &reads[6]) &&
         make_range(layout.spin_matrix_offsets, batch.batch_size + 1, sizeof(std::int64_t),
                    &reads[7]) &&
         make_range(cache.cholesky_factors, batch.total_matrix_elements, sizeof(double),
                    &reads[8]) &&
         make_range(cache.geometry_generations, batch.batch_size, sizeof(std::uint64_t),
                    &reads[9]) &&
         make_range(cache.factor_statuses, batch.batch_size, sizeof(std::uint32_t), &reads[10]) &&
         make_range(hamiltonians, layout.total_spin_matrix_elements, sizeof(double), &reads[11]) &&
         make_range(geometry_epoch, geometry_epoch == nullptr ? 0 : 1, sizeof(std::uint64_t),
                    &reads[12]) &&
         make_range(workspace.matrix_scratch_a, layout.total_spin_matrix_elements, sizeof(double),
                    &writes[0]) &&
         make_range(workspace.matrix_scratch_b, layout.total_spin_matrix_elements, sizeof(double),
                    &writes[1]) &&
         make_range(workspace.eigenvalue_scratch, layout.total_spin_orbitals, sizeof(double),
                    &writes[2]) &&
         make_range(workspace.factor_pointers, layout.total_spin_channels, sizeof(double*),
                    &writes[3]) &&
         make_range(workspace.matrix_pointers, layout.total_spin_channels, sizeof(double*),
                    &writes[4]) &&
         make_range(workspace.info_a, layout.total_spin_channels, sizeof(int), &writes[5]) &&
         make_range(workspace.info_b, layout.total_spin_channels, sizeof(int), &writes[6]) &&
         make_range(workspace.eligible, layout.total_spin_channels, sizeof(std::uint8_t),
                    &writes[7]) &&
         make_range(workspace.sequence_active, 1, sizeof(std::uint32_t), &writes[8]) &&
         make_byte_range(workspace.solver_device_workspace, workspace.solver_device_workspace_bytes,
                         &writes[9]) &&
         make_byte_range(workspace.solver_host_workspace, workspace.solver_host_workspace_bytes,
                         &writes[10]) &&
         make_range(results.eigenvalues, layout.total_spin_orbitals, sizeof(double), &writes[11]) &&
         make_range(results.coefficients, layout.total_spin_matrix_elements, sizeof(double),
                    &writes[12]) &&
         make_range(system_errors, batch.batch_size, sizeof(std::uint32_t), &writes[13]) &&
         make_range(device_error, 1, sizeof(std::uint32_t), &writes[14]) &&
         pairwise_disjoint(reads) && pairwise_disjoint(writes) && disjoint_sets(reads, writes);
}

bool valid_compaction_workspace(const Gfn2EigensolverDeviceBatch& batch, std::int64_t bucket_count,
                                const Gfn2EigensolverDeviceWorkspace& workspace) noexcept {
  return workspace.compact_system_elements >= batch.batch_size &&
         workspace.compact_source_slot_elements >= batch.batch_size &&
         workspace.bucket_activity_elements >= bucket_count &&
         is_aligned(workspace.compact_systems, alignof(std::int32_t)) &&
         is_aligned(workspace.compact_source_slots, alignof(std::int32_t)) &&
         is_aligned(workspace.bucket_activity, alignof(Gfn2EigensolverBucketActivity));
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
    Gfn2EigensolverOverlapCache cache, std::uint64_t scalar_generation,
    const std::uint64_t* device_generation, const double* hamiltonians, double symmetry_tolerance,
    Gfn2EigensolverDeviceWorkspace workspace, std::uint32_t* system_errors,
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
        const std::uint64_t geometry_generation =
            load_geometry_generation({scalar_generation, device_generation});
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

/* Resolve canonical bucket-local solve work without materializing another
 * device map. Topology setup fixes spin_channels for the plan lifetime, and
 * the scan order is deliberately system-major then spin-major. */
__device__ bool resolve_spin_solve_work(Gfn2EigensolverDeviceBatch batch,
                                        Gfn2WavefunctionLayoutView layout,
                                        Gfn2EigensolverBucket bucket, std::int64_t work_local,
                                        std::int32_t* system, std::int32_t* physical_local,
                                        std::uint8_t* spin) {
  for (std::int32_t local = 0; local < bucket.system_count; ++local) {
    const std::int32_t candidate =
        batch.bucket_systems[bucket.system_index_offset + static_cast<std::int64_t>(local)];
    if (candidate < 0 || candidate >= batch.batch_size) {
      return false;
    }
    const std::int32_t channels = layout.spin_channels[candidate];
    if (channels != 1 && channels != 2) {
      return false;
    }
    if (work_local < channels) {
      *system = candidate;
      *physical_local = local;
      *spin = static_cast<std::uint8_t>(work_local);
      return true;
    }
    work_local -= channels;
  }
  return false;
}

/* Provider batch counts are host metadata, so device topology must prove that
 * each count exactly matches sum(nspin) before any physical result can publish.
 * A mismatch is a plan-wide error: provider calls still receive identity
 * placeholders, but the complete sequence remains fail-closed. */
__global__ void validate_spin_bucket_layout_kernel(Gfn2EigensolverDeviceBatch batch,
                                                   Gfn2WavefunctionLayoutView layout,
                                                   Gfn2EigensolverBucket bucket,
                                                   std::uint32_t* sequence_active,
                                                   std::uint32_t* device_error) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || atomicAdd(sequence_active, 0u) == 0u) {
    return;
  }
  std::int64_t solve_count = 0;
  const std::int64_t n = bucket.orbital_count;
  const std::int64_t matrix_stride = n * n;
  for (std::int32_t local = 0; local < bucket.system_count; ++local) {
    const std::int32_t system =
        batch.bucket_systems[bucket.system_index_offset + static_cast<std::int64_t>(local)];
    if (system < 0 || system >= batch.batch_size) {
      record_global_error(device_error, Gfn2EigensolverDeviceError::kInvalidSpinLayout);
      atomicExch(sequence_active, 0u);
      return;
    }
    const std::int32_t channels = layout.spin_channels[system];
    const std::int64_t orbital_begin = batch.orbital_offsets[system];
    const std::int64_t orbital_end = batch.orbital_offsets[system + 1];
    const std::int64_t matrix_begin = batch.matrix_offsets[system];
    const std::int64_t matrix_end = batch.matrix_offsets[system + 1];
    const std::int64_t channel_begin = layout.spin_channel_offsets[system];
    const std::int64_t channel_end = layout.spin_channel_offsets[system + 1];
    const std::int64_t spin_orbital_begin = layout.spin_orbital_offsets[system];
    const std::int64_t spin_orbital_end = layout.spin_orbital_offsets[system + 1];
    const std::int64_t spin_matrix_begin = layout.spin_matrix_offsets[system];
    const std::int64_t spin_matrix_end = layout.spin_matrix_offsets[system + 1];
    if ((channels != 1 && channels != 2) || orbital_begin < 0 || orbital_end - orbital_begin != n ||
        orbital_end > batch.total_orbitals || matrix_begin < 0 ||
        matrix_end - matrix_begin != matrix_stride || matrix_end > batch.total_matrix_elements ||
        channel_begin < 0 || channel_end - channel_begin != channels ||
        channel_end > layout.total_spin_channels || spin_orbital_begin < 0 ||
        spin_orbital_end - spin_orbital_begin != static_cast<std::int64_t>(channels) * n ||
        spin_orbital_end > layout.total_spin_orbitals || spin_matrix_begin < 0 ||
        spin_matrix_end - spin_matrix_begin !=
            static_cast<std::int64_t>(channels) * matrix_stride ||
        spin_matrix_end > layout.total_spin_matrix_elements ||
        (system == 0 && (orbital_begin != 0 || matrix_begin != 0 || channel_begin != 0 ||
                         spin_orbital_begin != 0 || spin_matrix_begin != 0)) ||
        (system + 1 == batch.batch_size &&
         (orbital_end != batch.total_orbitals || matrix_end != batch.total_matrix_elements ||
          channel_end != layout.total_spin_channels ||
          spin_orbital_end != layout.total_spin_orbitals ||
          spin_matrix_end != layout.total_spin_matrix_elements))) {
      record_global_error(device_error, Gfn2EigensolverDeviceError::kInvalidSpinLayout);
      atomicExch(sequence_active, 0u);
      return;
    }
    solve_count += channels;
  }
  if (solve_count != bucket.solve_count) {
    record_global_error(device_error, Gfn2EigensolverDeviceError::kInvalidSpinLayout);
    atomicExch(sequence_active, 0u);
  }
}

__global__ void prepare_spin_solve_bucket_kernel(
    Gfn2EigensolverDeviceBatch batch, Gfn2WavefunctionLayoutView layout,
    Gfn2EigensolverBucket bucket, Gfn2EigensolverOverlapCache cache,
    std::uint64_t scalar_generation, const std::uint64_t* device_generation,
    const double* hamiltonians, double symmetry_tolerance, Gfn2EigensolverDeviceWorkspace workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t work_local = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t solve_slot = bucket.solve_index_offset + work_local;
  const std::int64_t matrix_stride = static_cast<std::int64_t>(bucket.orbital_count) *
                                     static_cast<std::int64_t>(bucket.orbital_count);
  const std::int64_t scratch_begin = bucket.spin_matrix_scratch_offset + work_local * matrix_stride;

  __shared__ std::int32_t system;
  __shared__ std::int32_t physical_local;
  __shared__ std::uint8_t spin;
  __shared__ std::int64_t input_begin;
  __shared__ std::int64_t factor_begin;
  __shared__ int valid;
  __shared__ int scan_enabled;
  __shared__ int finite_ok;
  __shared__ int symmetric_ok;
  if (threadIdx.x == 0) {
    system = -1;
    physical_local = -1;
    spin = 0u;
    input_begin = 0;
    factor_begin = 0;
    valid = 0;
    scan_enabled = 0;
    finite_ok = 1;
    symmetric_ok = 1;
    workspace.eligible[solve_slot] = 0u;
    workspace.factor_pointers[solve_slot] = workspace.matrix_scratch_a + scratch_begin;
    workspace.matrix_pointers[solve_slot] = workspace.matrix_scratch_b + scratch_begin;
    if (atomicAdd(workspace.sequence_active, 0u) != 0u &&
        resolve_spin_solve_work(batch, layout, bucket, work_local, &system, &physical_local,
                                &spin) &&
        system_is_clear(system_errors, system)) {
      const std::uint8_t active = batch.active[system];
      if (active != 0u && active != 1u) {
        record_system_error(system_errors, system, device_error,
                            Gfn2EigensolverDeviceError::kInvalidActiveMask);
      } else if (active == 1u) {
        const std::uint64_t geometry_generation =
            load_geometry_generation({scalar_generation, device_generation});
        factor_begin = bucket.matrix_scratch_offset +
                       static_cast<std::int64_t>(physical_local) * matrix_stride;
        input_begin =
            layout.spin_matrix_offsets[system] + static_cast<std::int64_t>(spin) * matrix_stride;
        if (cache.geometry_generations[system] != geometry_generation ||
            cache.factor_statuses[system] !=
                static_cast<std::uint32_t>(Gfn2EigensolverDeviceError::kSuccess)) {
          record_system_error(system_errors, system, device_error,
                              Gfn2EigensolverDeviceError::kStaleOverlapCache);
        } else {
          scan_enabled = 1;
        }
      }
    }
  }
  __syncthreads();
  /* The symmetric-input validation scans every matrix element; it was
   * previously one serial 15K-element chain on lane 0. The scan's conclusions
   * (any non-finite element, any out-of-tolerance symmetric pair) are
   * monotone, so lanes can clear shared flags concurrently and lane 0 keeps
   * the original error precedence afterwards. */
  if (scan_enabled != 0) {
    for (std::int64_t index = threadIdx.x; index < matrix_stride; index += blockDim.x) {
      const std::int64_t row = index % bucket.orbital_count;
      const std::int64_t column = index / bucket.orbital_count;
      const double value = hamiltonians[input_begin + index];
      if (!isfinite(value)) {
        atomicExch(&finite_ok, 0);
      }
      if (column < row) {
        const double transpose = hamiltonians[input_begin + column * bucket.orbital_count + row];
        if (!isfinite(transpose)) {
          atomicExch(&finite_ok, 0);
        } else if (isfinite(value)) {
          const double scale = fmax(1.0, fmax(fabs(value), fabs(transpose)));
          if (fabs(value - transpose) > symmetry_tolerance * scale) {
            atomicExch(&symmetric_ok, 0);
          }
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && scan_enabled != 0) {
    bool factor_finite = true;
    for (std::int64_t diagonal = 0; diagonal < bucket.orbital_count; ++diagonal) {
      const double value =
          cache.cholesky_factors[factor_begin + diagonal * bucket.orbital_count + diagonal];
      factor_finite = factor_finite && isfinite(value) && value > 0.0;
    }
    if (finite_ok == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2EigensolverDeviceError::kNonfiniteHamiltonian);
    } else if (symmetric_ok == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2EigensolverDeviceError::kNonsymmetricHamiltonian);
    } else if (!factor_finite) {
      record_system_error(system_errors, system, device_error,
                          Gfn2EigensolverDeviceError::kStaleOverlapCache);
    } else {
      valid = 1;
      workspace.eligible[solve_slot] = 1u;
    }
  }
  __syncthreads();

  for (std::int64_t column = 0; column < bucket.orbital_count; ++column) {
    for (std::int64_t row = threadIdx.x; row < bucket.orbital_count; row += blockDim.x) {
      const std::int64_t index = row + column * bucket.orbital_count;
      double factor = row == column ? 1.0 : 0.0;
      double hamiltonian = row == column ? 1.0 : 0.0;
      if (valid != 0) {
        factor = cache.cholesky_factors[factor_begin + index];
        const double first = hamiltonians[input_begin + row * bucket.orbital_count + column];
        const double second = hamiltonians[input_begin + column * bucket.orbital_count + row];
        hamiltonian = row == column ? first : 0.5 * first + 0.5 * second;
      }
      workspace.matrix_scratch_a[scratch_begin + index] = factor;
      workspace.matrix_scratch_b[scratch_begin + index] = hamiltonian;
    }
  }
}

__global__ void symmetrize_spin_transformed_bucket_kernel(Gfn2EigensolverDeviceBatch batch,
                                                          Gfn2WavefunctionLayoutView layout,
                                                          Gfn2EigensolverBucket bucket,
                                                          Gfn2EigensolverDeviceWorkspace workspace,
                                                          std::uint32_t* system_errors,
                                                          std::uint32_t* device_error) {
  const std::int64_t work_local = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t solve_slot = bucket.solve_index_offset + work_local;
  std::int32_t system = -1;
  std::int32_t physical_local = -1;
  std::uint8_t spin = 0u;
  if (!resolve_spin_solve_work(batch, layout, bucket, work_local, &system, &physical_local,
                               &spin)) {
    return;
  }
  const std::int64_t matrix_stride = static_cast<std::int64_t>(bucket.orbital_count) *
                                     static_cast<std::int64_t>(bucket.orbital_count);
  const std::int64_t matrix_begin = bucket.spin_matrix_scratch_offset + work_local * matrix_stride;
  __shared__ int valid;
  __shared__ int scan_ok;
  if (threadIdx.x == 0) {
    valid = workspace.eligible[solve_slot] == 1u && system_is_clear(system_errors, system) ? 1 : 0;
    scan_ok = 1;
  }
  __syncthreads();
  /* Formerly one thread serially scanned every matrix element for finiteness;
   * each lane now owns a strided slice so the 122-AO scan costs one pass
   * instead of a 15K-element serial chain. The error is still recorded exactly
   * once by lane 0 after the scan. */
  if (valid != 0) {
    for (std::int64_t index = threadIdx.x; index < matrix_stride; index += blockDim.x) {
      if (!isfinite(workspace.matrix_scratch_b[matrix_begin + index])) {
        atomicExch(&scan_ok, 0);
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && valid != 0 && scan_ok == 0) {
    record_system_error(system_errors, system, device_error,
                        Gfn2EigensolverDeviceError::kNonfiniteEigenpair);
    valid = 0;
  }
  __syncthreads();
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

/* This validation is intentionally a separate ordered stage. A failure in
 * either spin records the physical system error before any peer block can
 * publish the other spin. */
__global__ void validate_spin_eigenpairs_bucket_kernel(Gfn2EigensolverDeviceBatch batch,
                                                       Gfn2WavefunctionLayoutView layout,
                                                       Gfn2EigensolverBucket bucket,
                                                       Gfn2EigensolverDeviceWorkspace workspace,
                                                       std::uint32_t* system_errors,
                                                       std::uint32_t* device_error) {
  const std::int64_t work_local = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t solve_slot = bucket.solve_index_offset + work_local;
  __shared__ std::int32_t system;
  __shared__ int valid;
  __shared__ int failure;
  if (threadIdx.x == 0) {
    std::int32_t physical_local = -1;
    std::uint8_t spin = 0u;
    system = -1;
    valid = resolve_spin_solve_work(batch, layout, bucket, work_local, &system, &physical_local,
                                    &spin) &&
                    workspace.eligible[solve_slot] == 1u && system_is_clear(system_errors, system)
                ? 1
                : 0;
    failure = 0;
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  if (threadIdx.x == 0 && workspace.info_a[solve_slot] != 0) {
    record_system_error(system_errors, system, device_error,
                        Gfn2EigensolverDeviceError::kEigensolverFailed);
    valid = 0;
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  const std::int64_t matrix_stride = static_cast<std::int64_t>(bucket.orbital_count) *
                                     static_cast<std::int64_t>(bucket.orbital_count);
  const std::int64_t matrix_begin = bucket.spin_matrix_scratch_offset + work_local * matrix_stride;
  const std::int64_t orbital_begin =
      bucket.spin_orbital_scratch_offset + work_local * bucket.orbital_count;
  /* Formerly lane 0 alone scanned the eigenvalues and then every matrix
   * element serially; the strided lanes below replace that with one parallel
   * pass each. The failure is still recorded once by lane 0 afterwards. */
  for (std::int64_t orbital = threadIdx.x; orbital < bucket.orbital_count; orbital += blockDim.x) {
    if (!isfinite(workspace.eigenvalue_scratch[orbital_begin + orbital])) {
      atomicExch(&failure, 1);
    }
  }
  for (std::int64_t index = threadIdx.x; index < matrix_stride; index += blockDim.x) {
    if (!isfinite(workspace.matrix_scratch_b[matrix_begin + index])) {
      atomicExch(&failure, 1);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && failure != 0) {
    record_system_error(system_errors, system, device_error,
                        Gfn2EigensolverDeviceError::kNonfiniteEigenpair);
  }
}

__global__ void publish_spin_eigensystems_bucket_kernel(Gfn2EigensolverDeviceBatch batch,
                                                        Gfn2WavefunctionLayoutView layout,
                                                        Gfn2EigensolverBucket bucket,
                                                        Gfn2EigensolverDeviceWorkspace workspace,
                                                        Gfn2EigensolverDeviceResults results,
                                                        const std::uint32_t* system_errors) {
  const std::int64_t work_local = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t solve_slot = bucket.solve_index_offset + work_local;
  std::int32_t system = -1;
  std::int32_t physical_local = -1;
  std::uint8_t spin = 0u;
  if (!resolve_spin_solve_work(batch, layout, bucket, work_local, &system, &physical_local,
                               &spin) ||
      workspace.eligible[solve_slot] != 1u || batch.active[system] != 1u ||
      !system_is_clear(system_errors, system)) {
    return;
  }
  const std::int64_t matrix_stride = static_cast<std::int64_t>(bucket.orbital_count) *
                                     static_cast<std::int64_t>(bucket.orbital_count);
  const std::int64_t scratch_matrix_begin =
      bucket.spin_matrix_scratch_offset + work_local * matrix_stride;
  const std::int64_t scratch_orbital_begin =
      bucket.spin_orbital_scratch_offset + work_local * bucket.orbital_count;
  const std::int64_t output_matrix_begin =
      layout.spin_matrix_offsets[system] + static_cast<std::int64_t>(spin) * matrix_stride;
  const std::int64_t output_orbital_begin =
      layout.spin_orbital_offsets[system] + static_cast<std::int64_t>(spin) * bucket.orbital_count;
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

#if CUDART_VERSION >= 12080

/* A single ordered scan is intentional: bucket sizes are modest and this
 * makes compaction order independent of block scheduling and scan-library
 * implementation details. Expensive matrix work remains fully parallel. */
__global__ void compact_solve_bucket_kernel(Gfn2EigensolverDeviceBatch batch,
                                            Gfn2EigensolverBucket bucket, std::int64_t bucket_index,
                                            Gfn2EigensolverDeviceWorkspace workspace,
                                            const std::uint32_t* system_errors,
                                            cudaGraphConditionalHandle handle) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  Gfn2EigensolverBucketActivity activity{};
  for (std::int32_t local = 0; local < bucket.system_count; ++local) {
    const std::int64_t slot = bucket.system_index_offset + local;
    workspace.compact_systems[slot] = -1;
    workspace.compact_source_slots[slot] = -1;
  }
  if (atomicAdd(workspace.sequence_active, 0u) != 0u) {
    for (std::int32_t local = 0; local < bucket.system_count; ++local) {
      const std::int64_t canonical_slot = bucket.system_index_offset + local;
      const std::int32_t system = batch.bucket_systems[canonical_slot];
      if (system >= 0 && system < batch.batch_size && workspace.eligible[canonical_slot] == 1u &&
          system_is_clear(system_errors, system)) {
        const std::int64_t compact_slot =
            bucket.system_index_offset + static_cast<std::int64_t>(activity.active_count);
        workspace.compact_systems[compact_slot] = system;
        workspace.compact_source_slots[compact_slot] = local;
        ++activity.active_count;
      }
    }
  }
  workspace.bucket_activity[bucket_index] = activity;
  cudaGraphSetConditional(handle, activity.active_count);
}

/*
 * No-conditional compaction sibling used by the production device-dispatched
 * exact-capacity chain. It publishes bucket_activity.active_count and the
 * compacted system lists exactly like the conditional variant, but performs no
 * cudaGraphSetConditional, so the kernel can live inside a device-launchable
 * executable without any conditional node.
 */
__global__ void compact_solve_bucket_counts_kernel(Gfn2EigensolverDeviceBatch batch,
                                                   Gfn2EigensolverBucket bucket,
                                                   std::int64_t bucket_index,
                                                   Gfn2EigensolverDeviceWorkspace workspace,
                                                   const std::uint32_t* system_errors) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  Gfn2EigensolverBucketActivity activity{};
  for (std::int32_t local = 0; local < bucket.system_count; ++local) {
    const std::int64_t slot = bucket.system_index_offset + local;
    workspace.compact_systems[slot] = -1;
    workspace.compact_source_slots[slot] = -1;
  }
  if (atomicAdd(workspace.sequence_active, 0u) != 0u) {
    for (std::int32_t local = 0; local < bucket.system_count; ++local) {
      const std::int64_t canonical_slot = bucket.system_index_offset + local;
      const std::int32_t system = batch.bucket_systems[canonical_slot];
      if (system >= 0 && system < batch.batch_size && workspace.eligible[canonical_slot] == 1u &&
          system_is_clear(system_errors, system)) {
        const std::int64_t compact_slot =
            bucket.system_index_offset + static_cast<std::int64_t>(activity.active_count);
        workspace.compact_systems[compact_slot] = system;
        workspace.compact_source_slots[compact_slot] = local;
        ++activity.active_count;
      }
    }
  }
  workspace.bucket_activity[bucket_index] = activity;
}

__global__ void gather_compacted_solve_bucket_kernel(
    Gfn2EigensolverDeviceBatch batch, Gfn2EigensolverBucket bucket, std::int64_t bucket_index,
    std::uint32_t submission_count, Gfn2EigensolverOverlapCache cache, const double* hamiltonians,
    Gfn2EigensolverDeviceWorkspace workspace) {
  const std::int64_t compact_local = static_cast<std::int64_t>(blockIdx.x);
  if (compact_local >= submission_count) {
    return;
  }
  const std::int64_t compact_slot = bucket.system_index_offset + compact_local;
  const std::int32_t system = workspace.compact_systems[compact_slot];
  const std::int32_t source_local = workspace.compact_source_slots[compact_slot];
  const std::int64_t matrix_stride = static_cast<std::int64_t>(bucket.orbital_count) *
                                     static_cast<std::int64_t>(bucket.orbital_count);
  const std::int64_t source_begin =
      bucket.matrix_scratch_offset + static_cast<std::int64_t>(source_local) * matrix_stride;
  const std::int64_t destination_begin =
      bucket.matrix_scratch_offset + compact_local * matrix_stride;
  const std::int64_t hamiltonian_begin = batch.matrix_offsets[system];

  if (threadIdx.x == 0) {
    workspace.factor_pointers[compact_slot] = workspace.matrix_scratch_a + destination_begin;
    workspace.matrix_pointers[compact_slot] = workspace.matrix_scratch_b + destination_begin;
    if (blockIdx.x == 0) {
      workspace.bucket_activity[bucket_index].submitted_eigensolver_count = submission_count;
    }
  }
  for (std::int64_t index = threadIdx.x; index < matrix_stride; index += blockDim.x) {
    const std::int64_t row = index % bucket.orbital_count;
    const std::int64_t column = index / bucket.orbital_count;
    workspace.matrix_scratch_a[destination_begin + index] =
        cache.cholesky_factors[source_begin + index];
    const double first = hamiltonians[hamiltonian_begin + row * bucket.orbital_count + column];
    const double second = hamiltonians[hamiltonian_begin + column * bucket.orbital_count + row];
    workspace.matrix_scratch_b[destination_begin + index] =
        row == column ? first : 0.5 * first + 0.5 * second;
  }
}

__global__ void validate_compacted_transformed_bucket_kernel(
    Gfn2EigensolverBucket bucket, std::uint32_t submission_count,
    Gfn2EigensolverDeviceWorkspace workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error) {
  const std::int64_t compact_local = static_cast<std::int64_t>(blockIdx.x);
  if (compact_local >= submission_count || threadIdx.x != 0) {
    return;
  }
  const std::int64_t compact_slot = bucket.system_index_offset + compact_local;
  const std::int32_t system = workspace.compact_systems[compact_slot];
  const std::int64_t matrix_stride = static_cast<std::int64_t>(bucket.orbital_count) *
                                     static_cast<std::int64_t>(bucket.orbital_count);
  const std::int64_t matrix_begin = bucket.matrix_scratch_offset + compact_local * matrix_stride;
  if (!system_is_clear(system_errors, system)) {
    return;
  }
  for (std::int64_t index = 0; index < matrix_stride; ++index) {
    if (!isfinite(workspace.matrix_scratch_b[matrix_begin + index])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2EigensolverDeviceError::kNonfiniteEigenpair);
      return;
    }
  }
}

__global__ void symmetrize_compacted_bucket_kernel(Gfn2EigensolverBucket bucket,
                                                   std::uint32_t submission_count,
                                                   Gfn2EigensolverDeviceWorkspace workspace,
                                                   const std::uint32_t* system_errors) {
  const std::int64_t compact_local = static_cast<std::int64_t>(blockIdx.x);
  if (compact_local >= submission_count) {
    return;
  }
  const std::int64_t compact_slot = bucket.system_index_offset + compact_local;
  const std::int32_t system = workspace.compact_systems[compact_slot];
  const std::int64_t matrix_stride = static_cast<std::int64_t>(bucket.orbital_count) *
                                     static_cast<std::int64_t>(bucket.orbital_count);
  const std::int64_t matrix_begin = bucket.matrix_scratch_offset + compact_local * matrix_stride;
  const bool valid = system_is_clear(system_errors, system);
  for (std::int64_t index = threadIdx.x; index < matrix_stride; index += blockDim.x) {
    const std::int64_t row = index % bucket.orbital_count;
    const std::int64_t column = index / bucket.orbital_count;
    if (row < column) {
      continue;
    }
    double value = row == column ? 1.0 : 0.0;
    if (valid) {
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

/* Filter provider failures before the final TRSM. In-place pointer compaction
 * is safe because every destination index is less than or equal to its source
 * index, and the scan is serial and stable. */
__global__ void compact_successful_eigenpairs_kernel(Gfn2EigensolverBucket bucket,
                                                     std::int64_t bucket_index,
                                                     Gfn2EigensolverDeviceWorkspace workspace,
                                                     std::uint32_t* system_errors,
                                                     std::uint32_t* device_error,
                                                     cudaGraphConditionalHandle handle) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  Gfn2EigensolverBucketActivity& activity = workspace.bucket_activity[bucket_index];
  std::uint32_t completed = 0u;
  const std::int64_t matrix_stride = static_cast<std::int64_t>(bucket.orbital_count) *
                                     static_cast<std::int64_t>(bucket.orbital_count);
  for (std::uint32_t source = 0u; source < activity.active_count; ++source) {
    const std::int64_t source_slot = bucket.system_index_offset + source;
    const std::int32_t system = workspace.compact_systems[source_slot];
    bool healthy = system_is_clear(system_errors, system);
    if (healthy && workspace.info_a[source_slot] != 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2EigensolverDeviceError::kEigensolverFailed);
      healthy = false;
    }
    const std::int64_t matrix_begin =
        bucket.matrix_scratch_offset + static_cast<std::int64_t>(source) * matrix_stride;
    const std::int64_t orbital_begin =
        bucket.orbital_scratch_offset + static_cast<std::int64_t>(source) * bucket.orbital_count;
    for (std::int32_t orbital = 0; healthy && orbital < bucket.orbital_count; ++orbital) {
      healthy = isfinite(workspace.eigenvalue_scratch[orbital_begin + orbital]);
    }
    for (std::int64_t index = 0; healthy && index < matrix_stride; ++index) {
      healthy = isfinite(workspace.matrix_scratch_b[matrix_begin + index]);
    }
    if (!healthy) {
      if (system_is_clear(system_errors, system)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2EigensolverDeviceError::kNonfiniteEigenpair);
      }
      continue;
    }
    const std::int64_t destination_slot = bucket.system_index_offset + completed;
    workspace.compact_systems[destination_slot] = system;
    workspace.compact_source_slots[destination_slot] = static_cast<std::int32_t>(source);
    workspace.factor_pointers[destination_slot] = workspace.factor_pointers[source_slot];
    workspace.matrix_pointers[destination_slot] = workspace.matrix_pointers[source_slot];
    ++completed;
  }
  activity.completed_count = completed;
  cudaGraphSetConditional(handle, completed);
}

/*
 * No-conditional sibling of compact_successful_eigenpairs_kernel. Publishes
 * bucket_activity.completed_count and the re-compacted success lists without
 * touching a conditional handle, so the backtransform capacity can be selected
 * by a device dispatcher inside a device-launchable executable.
 */
__global__ void compact_successful_eigenpair_counts_kernel(Gfn2EigensolverBucket bucket,
                                                           std::int64_t bucket_index,
                                                           Gfn2EigensolverDeviceWorkspace workspace,
                                                           std::uint32_t* system_errors,
                                                           std::uint32_t* device_error) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  Gfn2EigensolverBucketActivity& activity = workspace.bucket_activity[bucket_index];
  std::uint32_t completed = 0u;
  const std::int64_t matrix_stride = static_cast<std::int64_t>(bucket.orbital_count) *
                                     static_cast<std::int64_t>(bucket.orbital_count);
  for (std::uint32_t source = 0u; source < activity.active_count; ++source) {
    const std::int64_t source_slot = bucket.system_index_offset + source;
    const std::int32_t system = workspace.compact_systems[source_slot];
    bool healthy = system_is_clear(system_errors, system);
    if (healthy && workspace.info_a[source_slot] != 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2EigensolverDeviceError::kEigensolverFailed);
      healthy = false;
    }
    const std::int64_t matrix_begin =
        bucket.matrix_scratch_offset + static_cast<std::int64_t>(source) * matrix_stride;
    const std::int64_t orbital_begin =
        bucket.orbital_scratch_offset + static_cast<std::int64_t>(source) * bucket.orbital_count;
    for (std::int32_t orbital = 0; healthy && orbital < bucket.orbital_count; ++orbital) {
      healthy = isfinite(workspace.eigenvalue_scratch[orbital_begin + orbital]);
    }
    for (std::int64_t index = 0; healthy && index < matrix_stride; ++index) {
      healthy = isfinite(workspace.matrix_scratch_b[matrix_begin + index]);
    }
    if (!healthy) {
      if (system_is_clear(system_errors, system)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2EigensolverDeviceError::kNonfiniteEigenpair);
      }
      continue;
    }
    const std::int64_t destination_slot = bucket.system_index_offset + completed;
    workspace.compact_systems[destination_slot] = system;
    workspace.compact_source_slots[destination_slot] = static_cast<std::int32_t>(source);
    workspace.factor_pointers[destination_slot] = workspace.factor_pointers[source_slot];
    workspace.matrix_pointers[destination_slot] = workspace.matrix_pointers[source_slot];
    ++completed;
  }
  activity.completed_count = completed;
}

__global__ void mark_backtransform_submission_kernel(std::int64_t bucket_index,
                                                     std::uint32_t submission_count,
                                                     Gfn2EigensolverDeviceWorkspace workspace) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    workspace.bucket_activity[bucket_index].submitted_backtransform_count = submission_count;
  }
}

__global__ void scatter_compacted_eigensystems_kernel(
    Gfn2EigensolverDeviceBatch batch, Gfn2EigensolverBucket bucket, std::uint32_t submission_count,
    Gfn2EigensolverDeviceWorkspace workspace, Gfn2EigensolverDeviceResults results,
    std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t compact_local = static_cast<std::int64_t>(blockIdx.x);
  if (compact_local >= submission_count) {
    return;
  }
  const std::int64_t compact_slot = bucket.system_index_offset + compact_local;
  const std::int32_t system = workspace.compact_systems[compact_slot];
  const std::int32_t source_local = workspace.compact_source_slots[compact_slot];
  const std::int64_t matrix_stride = static_cast<std::int64_t>(bucket.orbital_count) *
                                     static_cast<std::int64_t>(bucket.orbital_count);
  const std::int64_t scratch_matrix_begin =
      bucket.matrix_scratch_offset + static_cast<std::int64_t>(source_local) * matrix_stride;
  const std::int64_t scratch_orbital_begin =
      bucket.orbital_scratch_offset +
      static_cast<std::int64_t>(source_local) * bucket.orbital_count;
  const std::int64_t output_matrix_begin = batch.matrix_offsets[system];
  const std::int64_t output_orbital_begin = batch.orbital_offsets[system];

  __shared__ int publish;
  if (threadIdx.x == 0) {
    publish = system_is_clear(system_errors, system) ? 1 : 0;
    for (std::int32_t orbital = 0; publish != 0 && orbital < bucket.orbital_count; ++orbital) {
      publish = isfinite(workspace.eigenvalue_scratch[scratch_orbital_begin + orbital]) ? 1 : 0;
    }
    for (std::int64_t index = 0; publish != 0 && index < matrix_stride; ++index) {
      publish = isfinite(workspace.matrix_scratch_b[scratch_matrix_begin + index]) ? 1 : 0;
    }
    if (publish == 0 && system_is_clear(system_errors, system)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2EigensolverDeviceError::kNonfiniteEigenpair);
    }
  }
  __syncthreads();
  if (publish == 0) {
    return;
  }
  for (std::int32_t orbital = threadIdx.x; orbital < bucket.orbital_count; orbital += blockDim.x) {
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

#endif  // CUDART_VERSION >= 12080

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
                                           const Gfn2EigensolverDeviceWorkspace& workspace,
                                           bool deterministic_debug) noexcept {
  cublasStatus_t status = cublasSetStream(blas, stream);
  if (status == CUBLAS_STATUS_SUCCESS) {
    /* Batched TRSM otherwise asks cuBLAS to manage an internal allocation for
     * large matrices. Reuse the setup-owned solver arena: every provider stage
     * is ordered on one stream, so cuBLAS and cuSOLVER never use it concurrently. */
    status = cublasSetWorkspace(blas, workspace.solver_device_workspace,
                                workspace.solver_device_workspace_bytes);
  }
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

#if CUDART_VERSION >= 12080

void finish_or_abort_capture(cudaStream_t stream) noexcept {
  cudaGraph_t abandoned = nullptr;
  const cudaError_t status = cudaStreamEndCapture(stream, &abandoned);
  if (status != cudaSuccess) {
    (void)cudaGetLastError();
  }
  if (abandoned != nullptr) {
    (void)cudaGraphDestroy(abandoned);
  }
}

Gfn2EigensolverLaunchResult insert_switch_after_capture_dependencies(
    cudaStream_t root_stream, cudaGraph_t root_graph, cudaGraphConditionalHandle handle,
    std::uint32_t body_count, cudaGraphNode_t& node, cudaGraph_t*& bodies) noexcept {
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  cudaGraph_t captured_graph = nullptr;
  const cudaGraphNode_t* dependencies = nullptr;
  std::size_t dependency_count = 0u;
  cudaError_t status = cudaStreamGetCaptureInfo(root_stream, &capture_status, nullptr,
                                                &captured_graph, &dependencies, &dependency_count);
  if (status != cudaSuccess) {
    return cuda_failure(status);
  }
  if (capture_status == cudaStreamCaptureStatusNone || captured_graph != root_graph ||
      dependencies == nullptr || dependency_count == 0u || body_count == 0u) {
    return cuda_failure(cudaErrorStreamCaptureInvalidated);
  }

  cudaGraphNodeParams parameters{};
  parameters.type = cudaGraphNodeTypeConditional;
  parameters.conditional.handle = handle;
  parameters.conditional.type = cudaGraphCondTypeSwitch;
  parameters.conditional.size = body_count;
  status = cudaGraphAddNode(&node, root_graph, dependencies, dependency_count, &parameters);
  if (status == cudaSuccess) {
    status = cudaStreamUpdateCaptureDependencies(root_stream, &node, 1u,
                                                 cudaStreamSetCaptureDependencies);
  }
  if (status != cudaSuccess) {
    return cuda_failure(status);
  }
  bodies = parameters.conditional.phGraph_out;
  return launch_success();
}

Gfn2EigensolverLaunchResult begin_body_capture(cudaStream_t stream, cudaGraph_t graph) noexcept {
  const cudaError_t status = cudaStreamBeginCaptureToGraph(stream, graph, nullptr, nullptr, 0u,
                                                           cudaStreamCaptureModeThreadLocal);
  return status == cudaSuccess ? launch_success() : cuda_failure(status);
}

Gfn2EigensolverLaunchResult end_body_capture(cudaStream_t stream, cudaGraph_t expected) noexcept {
  cudaGraph_t ended = nullptr;
  const cudaError_t status = cudaStreamEndCapture(stream, &ended);
  if (status != cudaSuccess) {
    return cuda_failure(status);
  }
  return ended == expected ? launch_success() : cuda_failure(cudaErrorStreamCaptureInvalidated);
}

Gfn2EigensolverLaunchResult enqueue_capacity_eigensolver_body(
    cudaStream_t capture_stream, std::uint32_t capacity, const Gfn2EigensolverDeviceBatch& batch,
    const Gfn2EigensolverBucket& bucket, std::int64_t bucket_index,
    const Gfn2EigensolverOverlapCache& cache, const double* hamiltonians, cusolverDnHandle_t solver,
    cusolverDnParams_t parameters, cublasHandle_t blas,
    const Gfn2EigensolverDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, bool deterministic_debug) noexcept;

Gfn2EigensolverLaunchResult enqueue_capacity_backtransform_body(
    cudaStream_t capture_stream, std::uint32_t capacity, const Gfn2EigensolverDeviceBatch& batch,
    const Gfn2EigensolverBucket& bucket, std::int64_t bucket_index, cublasHandle_t blas,
    const Gfn2EigensolverDeviceWorkspace& workspace, const Gfn2EigensolverDeviceResults& results,
    std::uint32_t* system_errors, std::uint32_t* device_error, bool deterministic_debug) noexcept;

Gfn2EigensolverLaunchResult capture_eigensolver_capacity_body(
    cudaGraph_t body, cudaStream_t capture_stream, std::uint32_t capacity,
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket& bucket,
    std::int64_t bucket_index, const Gfn2EigensolverOverlapCache& cache, const double* hamiltonians,
    cusolverDnHandle_t solver, cusolverDnParams_t parameters, cublasHandle_t blas,
    const Gfn2EigensolverDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, bool deterministic_debug) noexcept {
  if (capacity == 0u) {
    return launch_success();
  }
  Gfn2EigensolverLaunchResult result = begin_body_capture(capture_stream, body);
  if (!result.success()) {
    return result;
  }
  result = enqueue_capacity_eigensolver_body(
      capture_stream, capacity, batch, bucket, bucket_index, cache, hamiltonians, solver,
      parameters, blas, workspace, system_errors, device_error, deterministic_debug);
  if (!result.success()) {
    finish_or_abort_capture(capture_stream);
    return result;
  }
  return end_body_capture(capture_stream, body);
}

/* Shared enqueue body: launches the provider kernels for an exact capacity into
 * an active capture. Separate from begin/end so the production dispatch chain
 * can interleave its dispatcher kernels with the provider body in one graph. */
Gfn2EigensolverLaunchResult enqueue_capacity_eigensolver_body(
    cudaStream_t capture_stream, std::uint32_t capacity, const Gfn2EigensolverDeviceBatch& batch,
    const Gfn2EigensolverBucket& bucket, std::int64_t bucket_index,
    const Gfn2EigensolverOverlapCache& cache, const double* hamiltonians, cusolverDnHandle_t solver,
    cusolverDnParams_t parameters, cublasHandle_t blas,
    const Gfn2EigensolverDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, bool deterministic_debug) noexcept {
  const Gfn2EigensolverBucket submission{bucket.orbital_count, static_cast<std::int32_t>(capacity),
                                         bucket.system_index_offset, bucket.matrix_scratch_offset,
                                         bucket.orbital_scratch_offset};
  gather_compacted_solve_bucket_kernel<<<capacity, kThreadsPerSystem, 0, capture_stream>>>(
      batch, bucket, bucket_index, capacity, cache, hamiltonians, workspace);
  Gfn2EigensolverLaunchResult result = check_kernel_launch();
  if (result.success()) {
    result = configure_blas(blas, capture_stream, workspace, deterministic_debug);
  }
  if (result.success()) {
    result = triangular_solve(blas, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, submission,
                              workspace.factor_pointers + bucket.system_index_offset,
                              workspace.matrix_pointers + bucket.system_index_offset);
  }
  if (result.success()) {
    result = triangular_solve(blas, CUBLAS_SIDE_RIGHT, CUBLAS_OP_T, submission,
                              workspace.factor_pointers + bucket.system_index_offset,
                              workspace.matrix_pointers + bucket.system_index_offset);
  }
  if (result.success()) {
    validate_compacted_transformed_bucket_kernel<<<capacity, 1, 0, capture_stream>>>(
        bucket, capacity, workspace, system_errors, device_error);
    result = check_kernel_launch();
  }
  if (result.success()) {
    symmetrize_compacted_bucket_kernel<<<capacity, kThreadsPerSystem, 0, capture_stream>>>(
        bucket, capacity, workspace, system_errors);
    result = check_kernel_launch();
  }
  if (result.success()) {
    result = configure_solver(solver, capture_stream);
  }
  if (result.success()) {
    result = symmetric_eigensolve(solver, parameters, CUSOLVER_EIG_MODE_VECTOR, submission,
                                  workspace.matrix_scratch_b + bucket.matrix_scratch_offset,
                                  workspace.eigenvalue_scratch + bucket.orbital_scratch_offset,
                                  workspace, workspace.info_a + bucket.system_index_offset);
  }
  return result;
}

Gfn2EigensolverLaunchResult capture_backtransform_capacity_body(
    cudaGraph_t body, cudaStream_t capture_stream, std::uint32_t capacity,
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket& bucket,
    std::int64_t bucket_index, cublasHandle_t blas, const Gfn2EigensolverDeviceWorkspace& workspace,
    const Gfn2EigensolverDeviceResults& results, std::uint32_t* system_errors,
    std::uint32_t* device_error, bool deterministic_debug) noexcept {
  if (capacity == 0u) {
    return launch_success();
  }
  Gfn2EigensolverLaunchResult result = begin_body_capture(capture_stream, body);
  if (!result.success()) {
    return result;
  }
  result = enqueue_capacity_backtransform_body(capture_stream, capacity, batch, bucket,
                                               bucket_index, blas, workspace, results,
                                               system_errors, device_error, deterministic_debug);
  if (!result.success()) {
    finish_or_abort_capture(capture_stream);
    return result;
  }
  return end_body_capture(capture_stream, body);
}

/* Shared enqueue body for the exact-capacity backtransform provider work. */
Gfn2EigensolverLaunchResult enqueue_capacity_backtransform_body(
    cudaStream_t capture_stream, std::uint32_t capacity, const Gfn2EigensolverDeviceBatch& batch,
    const Gfn2EigensolverBucket& bucket, std::int64_t bucket_index, cublasHandle_t blas,
    const Gfn2EigensolverDeviceWorkspace& workspace, const Gfn2EigensolverDeviceResults& results,
    std::uint32_t* system_errors, std::uint32_t* device_error, bool deterministic_debug) noexcept {
  const Gfn2EigensolverBucket submission{bucket.orbital_count, static_cast<std::int32_t>(capacity),
                                         bucket.system_index_offset, bucket.matrix_scratch_offset,
                                         bucket.orbital_scratch_offset};
  mark_backtransform_submission_kernel<<<1, 1, 0, capture_stream>>>(bucket_index, capacity,
                                                                    workspace);
  Gfn2EigensolverLaunchResult result = check_kernel_launch();
  if (result.success()) {
    result = configure_blas(blas, capture_stream, workspace, deterministic_debug);
  }
  if (result.success()) {
    result = triangular_solve(blas, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, submission,
                              workspace.factor_pointers + bucket.system_index_offset,
                              workspace.matrix_pointers + bucket.system_index_offset);
  }
  if (result.success()) {
    scatter_compacted_eigensystems_kernel<<<capacity, kThreadsPerSystem, 0, capture_stream>>>(
        batch, bucket, capacity, workspace, results, system_errors, device_error);
    result = check_kernel_launch();
  }
  return result;
}

#endif  // CUDART_VERSION >= 12080

}  // namespace

struct Gfn2EigensolverCompactedSolveGraph::Impl {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  std::uint64_t plan_token = 0u;

  ~Impl() {
    if (executable != nullptr) {
      (void)cudaGraphExecDestroy(executable);
    }
    if (graph != nullptr) {
      (void)cudaGraphDestroy(graph);
    }
  }
};

Gfn2EigensolverLaunchResult capture_gfn2_eigensolver_capacity_cuda(
    cudaGraph_t body, cudaStream_t capture_stream, std::uint32_t capacity,
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket& bucket,
    std::int64_t bucket_index, const Gfn2EigensolverOverlapCache& cache, const double* hamiltonians,
    cusolverDnHandle_t solver, cusolverDnParams_t parameters, cublasHandle_t blas,
    const Gfn2EigensolverDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, bool deterministic_debug) noexcept {
#if CUDART_VERSION >= 12080
  if (body == nullptr || capture_stream == nullptr || capacity == 0u ||
      capacity > static_cast<std::uint32_t>(bucket.system_count) || solver == nullptr ||
      parameters == nullptr || blas == nullptr) {
    return invalid_argument();
  }
  return capture_eigensolver_capacity_body(
      body, capture_stream, capacity, batch, bucket, bucket_index, cache, hamiltonians, solver,
      parameters, blas, workspace, system_errors, device_error, deterministic_debug);
#else
  (void)body;
  (void)capture_stream;
  (void)capacity;
  (void)batch;
  (void)bucket;
  (void)bucket_index;
  (void)cache;
  (void)hamiltonians;
  (void)solver;
  (void)parameters;
  (void)blas;
  (void)workspace;
  (void)system_errors;
  (void)device_error;
  (void)deterministic_debug;
  return cuda_failure(cudaErrorNotSupported);
#endif
}

Gfn2EigensolverLaunchResult capture_gfn2_backtransform_capacity_cuda(
    cudaGraph_t body, cudaStream_t capture_stream, std::uint32_t capacity,
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket& bucket,
    std::int64_t bucket_index, cublasHandle_t blas, const Gfn2EigensolverDeviceWorkspace& workspace,
    const Gfn2EigensolverDeviceResults& results, std::uint32_t* system_errors,
    std::uint32_t* device_error, bool deterministic_debug) noexcept {
#if CUDART_VERSION >= 12080
  if (body == nullptr || capture_stream == nullptr || capacity == 0u ||
      capacity > static_cast<std::uint32_t>(bucket.system_count) || blas == nullptr) {
    return invalid_argument();
  }
  return capture_backtransform_capacity_body(body, capture_stream, capacity, batch, bucket,
                                             bucket_index, blas, workspace, results, system_errors,
                                             device_error, deterministic_debug);
#else
  (void)body;
  (void)capture_stream;
  (void)capacity;
  (void)batch;
  (void)bucket;
  (void)bucket_index;
  (void)blas;
  (void)workspace;
  (void)results;
  (void)system_errors;
  (void)device_error;
  (void)deterministic_debug;
  return cuda_failure(cudaErrorNotSupported);
#endif
}

Gfn2EigensolverLaunchResult enqueue_gfn2_eigensolver_capacity_cuda(
    cudaStream_t capture_stream, std::uint32_t capacity, const Gfn2EigensolverDeviceBatch& batch,
    const Gfn2EigensolverBucket& bucket, std::int64_t bucket_index,
    const Gfn2EigensolverOverlapCache& cache, const double* hamiltonians, cusolverDnHandle_t solver,
    cusolverDnParams_t parameters, cublasHandle_t blas,
    const Gfn2EigensolverDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, bool deterministic_debug) noexcept {
#if CUDART_VERSION >= 12080
  if (capture_stream == nullptr || capacity > static_cast<std::uint32_t>(bucket.system_count) ||
      solver == nullptr || parameters == nullptr || blas == nullptr) {
    return invalid_argument();
  }
  if (capacity == 0u) {
    return launch_success();
  }
  return enqueue_capacity_eigensolver_body(capture_stream, capacity, batch, bucket, bucket_index,
                                           cache, hamiltonians, solver, parameters, blas, workspace,
                                           system_errors, device_error, deterministic_debug);
#else
  (void)capture_stream;
  (void)capacity;
  (void)batch;
  (void)bucket;
  (void)bucket_index;
  (void)cache;
  (void)hamiltonians;
  (void)solver;
  (void)parameters;
  (void)blas;
  (void)workspace;
  (void)system_errors;
  (void)device_error;
  (void)deterministic_debug;
  return cuda_failure(cudaErrorNotSupported);
#endif
}

Gfn2EigensolverLaunchResult enqueue_gfn2_backtransform_capacity_cuda(
    cudaStream_t capture_stream, std::uint32_t capacity, const Gfn2EigensolverDeviceBatch& batch,
    const Gfn2EigensolverBucket& bucket, std::int64_t bucket_index, cublasHandle_t blas,
    const Gfn2EigensolverDeviceWorkspace& workspace, const Gfn2EigensolverDeviceResults& results,
    std::uint32_t* system_errors, std::uint32_t* device_error, bool deterministic_debug) noexcept {
#if CUDART_VERSION >= 12080
  if (capture_stream == nullptr || capacity > static_cast<std::uint32_t>(bucket.system_count) ||
      blas == nullptr) {
    return invalid_argument();
  }
  if (capacity == 0u) {
    return launch_success();
  }
  return enqueue_capacity_backtransform_body(capture_stream, capacity, batch, bucket, bucket_index,
                                             blas, workspace, results, system_errors, device_error,
                                             deterministic_debug);
#else
  (void)capture_stream;
  (void)capacity;
  (void)batch;
  (void)bucket;
  (void)bucket_index;
  (void)blas;
  (void)workspace;
  (void)results;
  (void)system_errors;
  (void)device_error;
  (void)deterministic_debug;
  return cuda_failure(cudaErrorNotSupported);
#endif
}

Gfn2EigensolverCompactedSolveGraph::Gfn2EigensolverCompactedSolveGraph() noexcept = default;
Gfn2EigensolverCompactedSolveGraph::~Gfn2EigensolverCompactedSolveGraph() = default;
Gfn2EigensolverCompactedSolveGraph::Gfn2EigensolverCompactedSolveGraph(
    Gfn2EigensolverCompactedSolveGraph&&) noexcept = default;
Gfn2EigensolverCompactedSolveGraph& Gfn2EigensolverCompactedSolveGraph::operator=(
    Gfn2EigensolverCompactedSolveGraph&&) noexcept = default;

bool Gfn2EigensolverCompactedSolveGraph::valid() const noexcept {
  return impl_ != nullptr && impl_->graph != nullptr && impl_->executable != nullptr &&
         impl_->plan_token != 0u;
}

Gfn2EigensolverLaunchResult Gfn2EigensolverCompactedSolveGraph::launch(
    cudaStream_t stream) const noexcept {
  if (!valid()) {
    return invalid_argument();
  }
  const cudaError_t status = cudaGraphLaunch(impl_->executable, stream);
  return status == cudaSuccess ? launch_success() : cuda_failure(status);
}

#if CUDART_VERSION >= 12080

namespace {

struct ProviderHandleState {
  cudaStream_t solver_stream = nullptr;
  cudaStream_t blas_stream = nullptr;
  cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_HOST;
  cublasMath_t math_mode = CUBLAS_DEFAULT_MATH;
};

Gfn2EigensolverLaunchResult save_provider_state(cusolverDnHandle_t solver, cublasHandle_t blas,
                                                ProviderHandleState& state) noexcept {
  cusolverStatus_t solver_status = cusolverDnGetStream(solver, &state.solver_stream);
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    return cusolver_failure(solver_status);
  }
  cublasStatus_t blas_status = cublasGetStream(blas, &state.blas_stream);
  if (blas_status == CUBLAS_STATUS_SUCCESS) {
    blas_status = cublasGetPointerMode(blas, &state.pointer_mode);
  }
  if (blas_status == CUBLAS_STATUS_SUCCESS) {
    blas_status = cublasGetMathMode(blas, &state.math_mode);
  }
  return blas_status == CUBLAS_STATUS_SUCCESS ? launch_success() : cublas_failure(blas_status);
}

void restore_provider_state(cusolverDnHandle_t solver, cublasHandle_t blas,
                            const ProviderHandleState& state) noexcept {
  (void)cusolverDnSetStream(solver, state.solver_stream);
  (void)cublasSetStream(blas, state.blas_stream);
  (void)cublasSetPointerMode(blas, state.pointer_mode);
  (void)cublasSetMathMode(blas, state.math_mode);
}

Gfn2EigensolverLaunchResult build_compacted_eigensolver_graph_impl(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const Gfn2EigensolverOverlapCache& cache,
    std::uint64_t scalar_generation, const Gfn2GeometryEpochDevice* geometry_epoch,
    const double* hamiltonians, const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver,
    cusolverDnParams_t parameters, cublasHandle_t blas,
    const Gfn2EigensolverDeviceWorkspace& workspace, const Gfn2EigensolverDeviceResults& results,
    std::uint32_t* system_errors, std::uint32_t* device_error,
    std::unique_ptr<Gfn2EigensolverCompactedSolveGraph::Impl>& built) noexcept {
  const bool dynamic_epoch = geometry_epoch != nullptr;
  if (!valid_bucket_plan(batch, buckets, bucket_count) || !valid_options(options) ||
      !valid_workspace(batch, workspace) ||
      !valid_compaction_workspace(batch, bucket_count, workspace) || !valid_cache(batch, cache) ||
      ((!dynamic_epoch && scalar_generation == 0u) ||
       (dynamic_epoch &&
        (scalar_generation != 0u || geometry_epoch->value == nullptr ||
         geometry_epoch->value_elements != 1 || geometry_epoch->plan_token != batch.plan_token ||
         !is_aligned(geometry_epoch->value, alignof(std::uint64_t))))) ||
      solver == nullptr || parameters == nullptr || blas == nullptr ||
      results.plan_token != batch.plan_token ||
      results.eigenvalue_elements < batch.total_orbitals ||
      results.coefficient_elements < batch.total_matrix_elements ||
      !is_aligned(hamiltonians, alignof(double)) ||
      !is_aligned(results.eigenvalues, alignof(double)) ||
      !is_aligned(results.coefficients, alignof(double)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t)) ||
      !valid_solve_ranges(batch, cache, hamiltonians,
                          dynamic_epoch ? geometry_epoch->value : nullptr, workspace, results,
                          system_errors, device_error)) {
    return invalid_argument();
  }

  ProviderHandleState provider_state{};
  Gfn2EigensolverLaunchResult result = save_provider_state(solver, blas, provider_state);
  if (!result.success()) {
    return result;
  }
  const auto restore = [&]() noexcept { restore_provider_state(solver, blas, provider_state); };

  std::unique_ptr<Gfn2EigensolverCompactedSolveGraph::Impl> candidate(
      new (std::nothrow) Gfn2EigensolverCompactedSolveGraph::Impl());
  if (candidate == nullptr) {
    restore();
    return cuda_failure(cudaErrorMemoryAllocation);
  }
  cudaError_t cuda_status = cudaGraphCreate(&candidate->graph, 0u);
  if (cuda_status != cudaSuccess) {
    restore();
    return cuda_failure(cuda_status);
  }

  std::unique_ptr<cudaGraphConditionalHandle[]> eigensolver_handles(
      new (std::nothrow) cudaGraphConditionalHandle[static_cast<std::size_t>(bucket_count)]{});
  std::unique_ptr<cudaGraphConditionalHandle[]> backtransform_handles(
      new (std::nothrow) cudaGraphConditionalHandle[static_cast<std::size_t>(bucket_count)]{});
  std::unique_ptr<cudaGraph_t*[]> eigensolver_body_sets(
      new (std::nothrow) cudaGraph_t* [static_cast<std::size_t>(bucket_count)] {});
  std::unique_ptr<cudaGraph_t*[]> backtransform_body_sets(
      new (std::nothrow) cudaGraph_t* [static_cast<std::size_t>(bucket_count)] {});
  if (eigensolver_handles == nullptr || backtransform_handles == nullptr ||
      eigensolver_body_sets == nullptr || backtransform_body_sets == nullptr) {
    restore();
    return cuda_failure(cudaErrorMemoryAllocation);
  }
  for (std::int64_t bucket_index = 0; bucket_index < bucket_count; ++bucket_index) {
    cuda_status = cudaGraphConditionalHandleCreate(
        &eigensolver_handles[static_cast<std::size_t>(bucket_index)], candidate->graph, 0u, 0u);
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaGraphConditionalHandleCreate(
          &backtransform_handles[static_cast<std::size_t>(bucket_index)], candidate->graph, 0u, 0u);
    }
    if (cuda_status != cudaSuccess) {
      restore();
      return cuda_failure(cuda_status);
    }
  }

  cudaStream_t root_stream = nullptr;
  cudaStream_t body_stream = nullptr;
  cuda_status = cudaStreamCreateWithFlags(&root_stream, cudaStreamNonBlocking);
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaStreamCreateWithFlags(&body_stream, cudaStreamNonBlocking);
  }
  if (cuda_status != cudaSuccess) {
    if (root_stream != nullptr) {
      (void)cudaStreamDestroy(root_stream);
    }
    restore();
    return cuda_failure(cuda_status);
  }
  bool root_capture_active = false;
  const auto finish_streams = [&]() noexcept {
    if (root_capture_active) {
      finish_or_abort_capture(root_stream);
      root_capture_active = false;
    }
    (void)cudaStreamDestroy(body_stream);
    (void)cudaStreamDestroy(root_stream);
    restore();
  };

  cuda_status = cudaStreamBeginCaptureToGraph(root_stream, candidate->graph, nullptr, nullptr, 0u,
                                              cudaStreamCaptureModeThreadLocal);
  if (cuda_status != cudaSuccess) {
    finish_streams();
    return cuda_failure(cuda_status);
  }
  root_capture_active = true;
  result = prepare_launch_sequence(batch, workspace, device_error, root_stream);
  if (!result.success()) {
    finish_streams();
    return result;
  }

  for (std::int64_t bucket_index = 0; bucket_index < bucket_count; ++bucket_index) {
    const Gfn2EigensolverBucket bucket = buckets[bucket_index];
    prepare_solve_bucket_kernel<<<static_cast<unsigned int>(bucket.system_count), kThreadsPerSystem,
                                  0, root_stream>>>(
        batch, bucket, cache, scalar_generation, dynamic_epoch ? geometry_epoch->value : nullptr,
        hamiltonians, options.symmetry_tolerance, workspace, system_errors, device_error);
    result = check_kernel_launch();
    if (!result.success()) {
      finish_streams();
      return result;
    }
    compact_solve_bucket_kernel<<<1, 1, 0, root_stream>>>(
        batch, bucket, bucket_index, workspace, system_errors,
        eigensolver_handles[static_cast<std::size_t>(bucket_index)]);
    result = check_kernel_launch();
    if (!result.success()) {
      finish_streams();
      return result;
    }

    cudaGraphNode_t eigensolver_switch = nullptr;
    const std::uint32_t capacity_count = static_cast<std::uint32_t>(bucket.system_count) + 1u;
    result = insert_switch_after_capture_dependencies(
        root_stream, candidate->graph, eigensolver_handles[static_cast<std::size_t>(bucket_index)],
        capacity_count, eigensolver_switch,
        eigensolver_body_sets[static_cast<std::size_t>(bucket_index)]);
    if (!result.success()) {
      finish_streams();
      return result;
    }

    compact_successful_eigenpairs_kernel<<<1, 1, 0, root_stream>>>(
        bucket, bucket_index, workspace, system_errors, device_error,
        backtransform_handles[static_cast<std::size_t>(bucket_index)]);
    result = check_kernel_launch();
    if (!result.success()) {
      finish_streams();
      return result;
    }
    cudaGraphNode_t backtransform_switch = nullptr;
    result = insert_switch_after_capture_dependencies(
        root_stream, candidate->graph,
        backtransform_handles[static_cast<std::size_t>(bucket_index)], capacity_count,
        backtransform_switch, backtransform_body_sets[static_cast<std::size_t>(bucket_index)]);
    if (!result.success()) {
      finish_streams();
      return result;
    }
  }

  cudaGraph_t ended_graph = nullptr;
  cuda_status = cudaStreamEndCapture(root_stream, &ended_graph);
  root_capture_active = false;
  if (cuda_status != cudaSuccess || ended_graph != candidate->graph) {
    finish_streams();
    return cuda_failure(cuda_status == cudaSuccess ? cudaErrorStreamCaptureInvalidated
                                                   : cuda_status);
  }

  /* Conditional body graphs can only be populated after the root capture has
   * ended; one thread cannot own nested stream captures. The stored body-array
   * addresses are CUDA-owned and remain valid for the parent node lifetime. */
  for (std::int64_t bucket_index = 0; bucket_index < bucket_count; ++bucket_index) {
    const Gfn2EigensolverBucket bucket = buckets[bucket_index];
    const std::uint32_t capacity_count = static_cast<std::uint32_t>(bucket.system_count) + 1u;
    for (std::uint32_t capacity = 1u; capacity < capacity_count; ++capacity) {
      result = capture_eigensolver_capacity_body(
          eigensolver_body_sets[static_cast<std::size_t>(bucket_index)][capacity], body_stream,
          capacity, batch, bucket, bucket_index, cache, hamiltonians, solver, parameters, blas,
          workspace, system_errors, device_error, options.deterministic_debug);
      if (!result.success()) {
        finish_streams();
        return result;
      }
      result = capture_backtransform_capacity_body(
          backtransform_body_sets[static_cast<std::size_t>(bucket_index)][capacity], body_stream,
          capacity, batch, bucket, bucket_index, blas, workspace, results, system_errors,
          device_error, options.deterministic_debug);
      if (!result.success()) {
        finish_streams();
        return result;
      }
    }
  }
  finish_streams();
  cuda_status = cudaGraphInstantiate(&candidate->executable, candidate->graph, 0u);
  if (cuda_status != cudaSuccess) {
    return cuda_failure(cuda_status);
  }
  candidate->plan_token = batch.plan_token;
  built = std::move(candidate);
  return launch_success();
}

}  // namespace

#endif  // CUDART_VERSION >= 12080

Gfn2EigensolverLaunchResult build_gfn2_compacted_eigensolver_graph_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const Gfn2EigensolverOverlapCache& cache,
    std::uint64_t geometry_generation, const double* hamiltonians,
    const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver, cusolverDnParams_t parameters,
    cublasHandle_t blas, const Gfn2EigensolverDeviceWorkspace& workspace,
    const Gfn2EigensolverDeviceResults& results, std::uint32_t* system_errors,
    std::uint32_t* device_error, Gfn2EigensolverCompactedSolveGraph& output) noexcept {
#if CUDART_VERSION >= 12080
  std::unique_ptr<Gfn2EigensolverCompactedSolveGraph::Impl> candidate;
  const Gfn2EigensolverLaunchResult launch = build_compacted_eigensolver_graph_impl(
      batch, buckets, bucket_count, cache, geometry_generation, nullptr, hamiltonians, options,
      solver, parameters, blas, workspace, results, system_errors, device_error, candidate);
  if (launch.success()) {
    output.impl_ = std::move(candidate);
  }
  return launch;
#else
  (void)batch;
  (void)buckets;
  (void)bucket_count;
  (void)cache;
  (void)geometry_generation;
  (void)hamiltonians;
  (void)options;
  (void)solver;
  (void)parameters;
  (void)blas;
  (void)workspace;
  (void)results;
  (void)system_errors;
  (void)device_error;
  (void)output;
  return cuda_failure(cudaErrorNotSupported);
#endif
}

Gfn2EigensolverLaunchResult build_gfn2_compacted_eigensolver_graph_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const Gfn2EigensolverOverlapCache& cache,
    const Gfn2GeometryEpochDevice& geometry_epoch, const double* hamiltonians,
    const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver, cusolverDnParams_t parameters,
    cublasHandle_t blas, const Gfn2EigensolverDeviceWorkspace& workspace,
    const Gfn2EigensolverDeviceResults& results, std::uint32_t* system_errors,
    std::uint32_t* device_error, Gfn2EigensolverCompactedSolveGraph& output) noexcept {
#if CUDART_VERSION >= 12080
  std::unique_ptr<Gfn2EigensolverCompactedSolveGraph::Impl> candidate;
  const Gfn2EigensolverLaunchResult launch = build_compacted_eigensolver_graph_impl(
      batch, buckets, bucket_count, cache, 0u, &geometry_epoch, hamiltonians, options, solver,
      parameters, blas, workspace, results, system_errors, device_error, candidate);
  if (launch.success()) {
    output.impl_ = std::move(candidate);
  }
  return launch;
#else
  (void)batch;
  (void)buckets;
  (void)bucket_count;
  (void)cache;
  (void)geometry_epoch;
  (void)hamiltonians;
  (void)options;
  (void)solver;
  (void)parameters;
  (void)blas;
  (void)workspace;
  (void)results;
  (void)system_errors;
  (void)device_error;
  (void)output;
  return cuda_failure(cudaErrorNotSupported);
#endif
}

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
  /* Exact-capacity Graph bodies can submit any count in [1, system_count].
   * cuSOLVER does not promise monotonic workspace sizes across batch counts,
   * so setup queries every reachable body and retains the true maximum. */
  for (const cusolverEigMode_t mode : {CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR}) {
    for (int capacity = 1; capacity <= bucket.system_count; ++capacity) {
      std::size_t device_bytes = 0u;
      std::size_t host_bytes = 0u;
      const cusolverStatus_t status = cusolverDnXsyevBatched_bufferSize(
          solver, parameters, mode, CUBLAS_FILL_MODE_LOWER, bucket.orbital_count, CUDA_R_64F,
          device_matrix, bucket.orbital_count, CUDA_R_64F, device_eigenvalues, CUDA_R_64F,
          &device_bytes, &host_bytes, capacity);
      if (status != CUSOLVER_STATUS_SUCCESS) {
        return cusolver_failure(status);
      }
      maximum_device = std::max(maximum_device, device_bytes);
      maximum_host = std::max(maximum_host, host_bytes);
    }
  }
  requirements.solver_device_workspace_bytes =
      std::max(requirements.solver_device_workspace_bytes, maximum_device);
  requirements.solver_host_workspace_bytes =
      std::max(requirements.solver_host_workspace_bytes, maximum_host);
  return launch_success();
}

Gfn2EigensolverLaunchResult query_gfn2_spin_eigensolver_bucket_workspace_cuda(
    cusolverDnHandle_t solver, cusolverDnParams_t parameters, const Gfn2EigensolverBucket& bucket,
    const double* device_matrix, const double* device_eigenvalues,
    Gfn2EigensolverWorkspaceRequirements& requirements) noexcept {
  if (bucket.solve_count <= 0) {
    return invalid_argument();
  }
  const Gfn2EigensolverBucket submission{
      bucket.orbital_count, bucket.solve_count, bucket.solve_index_offset,
      bucket.spin_matrix_scratch_offset, bucket.spin_orbital_scratch_offset};
  return query_gfn2_eigensolver_bucket_workspace_cuda(solver, parameters, submission, device_matrix,
                                                      device_eigenvalues, requirements);
}

Gfn2EigensolverLaunchResult compact_gfn2_solve_bucket_counts_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket& bucket,
    std::int64_t bucket_index, const Gfn2EigensolverDeviceWorkspace& workspace,
    const std::uint32_t* system_errors, cudaStream_t stream) noexcept {
  if (!valid_bucket_slice(batch, bucket) || !valid_compaction_workspace(batch, 1, workspace) ||
      bucket_index < 0 || workspace.bucket_activity_elements <= bucket_index ||
      system_errors == nullptr || !is_aligned(system_errors, alignof(std::uint32_t))) {
    return invalid_argument();
  }
#if CUDART_VERSION >= 12080
  compact_solve_bucket_counts_kernel<<<1, 1, 0, stream>>>(batch, bucket, bucket_index, workspace,
                                                          system_errors);
  return check_kernel_launch();
#else
  (void)stream;
  return cuda_failure(cudaErrorNotSupported);
#endif
}

Gfn2EigensolverLaunchResult prepare_and_compact_gfn2_solve_buckets_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const Gfn2EigensolverOverlapCache& cache,
    std::uint64_t geometry_generation, const double* hamiltonians,
    const Gfn2EigensolverOptions& options, const Gfn2EigensolverDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (!valid_bucket_plan(batch, buckets, bucket_count) || !valid_options(options) ||
      !valid_workspace(batch, workspace) ||
      !valid_compaction_workspace(batch, bucket_count, workspace) || geometry_generation == 0u ||
      !is_aligned(hamiltonians, alignof(double)) || system_errors == nullptr ||
      device_error == nullptr || !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return invalid_argument();
  }
#if CUDART_VERSION >= 12080
  for (std::int64_t bucket_index = 0; bucket_index < bucket_count; ++bucket_index) {
    const Gfn2EigensolverBucket bucket = buckets[bucket_index];
    prepare_solve_bucket_kernel<<<static_cast<unsigned int>(bucket.system_count), kThreadsPerSystem,
                                  0, stream>>>(batch, bucket, cache, geometry_generation, nullptr,
                                               hamiltonians, options.symmetry_tolerance, workspace,
                                               system_errors, device_error);
    Gfn2EigensolverLaunchResult result = check_kernel_launch();
    if (!result.success()) {
      return result;
    }
    compact_solve_bucket_counts_kernel<<<1, 1, 0, stream>>>(batch, bucket, bucket_index, workspace,
                                                            system_errors);
    result = check_kernel_launch();
    if (!result.success()) {
      return result;
    }
  }
  return launch_success();
#else
  (void)stream;
  return cuda_failure(cudaErrorNotSupported);
#endif
}

Gfn2EigensolverLaunchResult prepare_and_compact_gfn2_solve_buckets_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const Gfn2EigensolverOverlapCache& cache,
    const Gfn2GeometryEpochDevice& geometry_epoch, const double* hamiltonians,
    const Gfn2EigensolverOptions& options, const Gfn2EigensolverDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (geometry_epoch.value_elements != 1 || geometry_epoch.plan_token != batch.plan_token ||
      geometry_epoch.value == nullptr ||
      !is_aligned(geometry_epoch.value, alignof(std::uint64_t))) {
    return invalid_argument();
  }
  if (!valid_bucket_plan(batch, buckets, bucket_count) || !valid_options(options) ||
      !valid_workspace(batch, workspace) ||
      !valid_compaction_workspace(batch, bucket_count, workspace) ||
      !is_aligned(hamiltonians, alignof(double)) || system_errors == nullptr ||
      device_error == nullptr || !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return invalid_argument();
  }
#if CUDART_VERSION >= 12080
  for (std::int64_t bucket_index = 0; bucket_index < bucket_count; ++bucket_index) {
    const Gfn2EigensolverBucket bucket = buckets[bucket_index];
    prepare_solve_bucket_kernel<<<static_cast<unsigned int>(bucket.system_count), kThreadsPerSystem,
                                  0, stream>>>(batch, bucket, cache, 0u, geometry_epoch.value,
                                               hamiltonians, options.symmetry_tolerance, workspace,
                                               system_errors, device_error);
    Gfn2EigensolverLaunchResult result = check_kernel_launch();
    if (!result.success()) {
      return result;
    }
    compact_solve_bucket_counts_kernel<<<1, 1, 0, stream>>>(batch, bucket, bucket_index, workspace,
                                                            system_errors);
    result = check_kernel_launch();
    if (!result.success()) {
      return result;
    }
  }
  return launch_success();
#else
  (void)stream;
  return cuda_failure(cudaErrorNotSupported);
#endif
}

Gfn2EigensolverLaunchResult compact_gfn2_successful_eigenpair_counts_cuda(
    const Gfn2EigensolverBucket& bucket, std::int64_t bucket_index,
    const Gfn2EigensolverDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (bucket.system_count <= 0 || bucket_index < 0 || workspace.compact_system_elements <= 0 ||
      workspace.compact_source_slot_elements <= 0 ||
      workspace.bucket_activity_elements <= bucket_index ||
      !is_aligned(workspace.compact_systems, alignof(std::int32_t)) ||
      !is_aligned(workspace.compact_source_slots, alignof(std::int32_t)) ||
      !is_aligned(workspace.bucket_activity, alignof(Gfn2EigensolverBucketActivity)) ||
      system_errors == nullptr || device_error == nullptr ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return invalid_argument();
  }
#if CUDART_VERSION >= 12080
  compact_successful_eigenpair_counts_kernel<<<1, 1, 0, stream>>>(bucket, bucket_index, workspace,
                                                                  system_errors, device_error);
  return check_kernel_launch();
#else
  (void)stream;
  return cuda_failure(cudaErrorNotSupported);
#endif
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

Gfn2EigensolverLaunchResult prepare_gfn2_eigensolver_launch_sequence_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverDeviceWorkspace& workspace,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (!valid_workspace(batch, workspace) || device_error == nullptr ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return invalid_argument();
  }
  return prepare_launch_sequence(batch, workspace, device_error, stream);
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

static Gfn2EigensolverLaunchResult solve_eigensystems_impl(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const Gfn2EigensolverOverlapCache& cache,
    std::uint64_t scalar_generation, const Gfn2GeometryEpochDevice* geometry_epoch,
    const double* hamiltonians, const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver,
    cusolverDnParams_t parameters, cublasHandle_t blas,
    const Gfn2EigensolverDeviceWorkspace& workspace, const Gfn2EigensolverDeviceResults& results,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  const bool dynamic_epoch = geometry_epoch != nullptr;
  if (!valid_bucket_plan(batch, buckets, bucket_count) || !valid_options(options) ||
      !valid_workspace(batch, workspace) || !valid_cache(batch, cache) ||
      ((!dynamic_epoch && scalar_generation == 0u) ||
       (dynamic_epoch &&
        (scalar_generation != 0u || geometry_epoch->value == nullptr ||
         geometry_epoch->value_elements != 1 || geometry_epoch->plan_token != batch.plan_token ||
         !is_aligned(geometry_epoch->value, alignof(std::uint64_t))))) ||
      solver == nullptr || parameters == nullptr || blas == nullptr ||
      results.plan_token != batch.plan_token ||
      results.eigenvalue_elements < batch.total_orbitals ||
      results.coefficient_elements < batch.total_matrix_elements ||
      !is_aligned(hamiltonians, alignof(double)) ||
      !is_aligned(results.eigenvalues, alignof(double)) ||
      !is_aligned(results.coefficients, alignof(double)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t)) ||
      !valid_solve_ranges(batch, cache, hamiltonians,
                          dynamic_epoch ? geometry_epoch->value : nullptr, workspace, results,
                          system_errors, device_error)) {
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
  result = configure_blas(blas, stream, workspace, options.deterministic_debug);
  if (!result.success()) {
    return result;
  }

  for (std::int64_t bucket_index = 0; bucket_index < bucket_count; ++bucket_index) {
    const Gfn2EigensolverBucket bucket = buckets[bucket_index];
    const std::int64_t matrix_begin = bucket.matrix_scratch_offset;
    const std::int64_t orbital_begin = bucket.orbital_scratch_offset;
    const std::int64_t info_begin = bucket.system_index_offset;
    prepare_solve_bucket_kernel<<<static_cast<unsigned int>(bucket.system_count), kThreadsPerSystem,
                                  0, stream>>>(
        batch, bucket, cache, scalar_generation, dynamic_epoch ? geometry_epoch->value : nullptr,
        hamiltonians, options.symmetry_tolerance, workspace, system_errors, device_error);
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

static Gfn2EigensolverLaunchResult solve_spin_eigensystems_impl(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2EigensolverBucket* buckets, std::int64_t bucket_count,
    const Gfn2EigensolverOverlapCache& cache, std::uint64_t scalar_generation,
    const Gfn2GeometryEpochDevice* geometry_epoch, const double* hamiltonians,
    const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver, cusolverDnParams_t parameters,
    cublasHandle_t blas, const Gfn2EigensolverDeviceWorkspace& workspace,
    const Gfn2EigensolverDeviceResults& results, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  const bool dynamic_epoch = geometry_epoch != nullptr;
  if (!valid_spin_bucket_plan(batch, layout, buckets, bucket_count) || !valid_options(options) ||
      !valid_spin_workspace(batch, layout, workspace) || !valid_cache(batch, cache) ||
      ((!dynamic_epoch && scalar_generation == 0u) ||
       (dynamic_epoch &&
        (scalar_generation != 0u || geometry_epoch->value == nullptr ||
         geometry_epoch->value_elements != 1 || geometry_epoch->plan_token != batch.plan_token ||
         !is_aligned(geometry_epoch->value, alignof(std::uint64_t))))) ||
      solver == nullptr || parameters == nullptr || blas == nullptr ||
      results.plan_token != batch.plan_token ||
      results.eigenvalue_elements < layout.total_spin_orbitals ||
      results.coefficient_elements < layout.total_spin_matrix_elements ||
      !is_aligned(hamiltonians, alignof(double)) ||
      !is_aligned(results.eigenvalues, alignof(double)) ||
      !is_aligned(results.coefficients, alignof(double)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t)) ||
      !valid_spin_solve_ranges(batch, layout, cache, hamiltonians,
                               dynamic_epoch ? geometry_epoch->value : nullptr, workspace, results,
                               system_errors, device_error)) {
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
  cudaError_t clear_status = cudaMemsetAsync(
      workspace.info_a, 0,
      static_cast<std::size_t>(layout.total_spin_channels) * sizeof(*workspace.info_a), stream);
  if (clear_status != cudaSuccess) {
    return cuda_failure(clear_status);
  }
  result = configure_blas(blas, stream, workspace, options.deterministic_debug);
  if (!result.success()) {
    return result;
  }

  /* Validate every bucket before launching any publication stage so a device-
   * resident topology mismatch is whole-call fail-closed. */
  for (std::int64_t bucket_index = 0; bucket_index < bucket_count; ++bucket_index) {
    validate_spin_bucket_layout_kernel<<<1, 1, 0, stream>>>(
        batch, layout, buckets[bucket_index], workspace.sequence_active, device_error);
    result = check_kernel_launch();
    if (!result.success()) {
      return result;
    }
  }

  for (std::int64_t bucket_index = 0; bucket_index < bucket_count; ++bucket_index) {
    const Gfn2EigensolverBucket bucket = buckets[bucket_index];
    const Gfn2EigensolverBucket submission{
        bucket.orbital_count, bucket.solve_count, bucket.solve_index_offset,
        bucket.spin_matrix_scratch_offset, bucket.spin_orbital_scratch_offset};
    prepare_spin_solve_bucket_kernel<<<static_cast<unsigned int>(bucket.solve_count), 256, 0,
                                       stream>>>(batch, layout, bucket, cache, scalar_generation,
                                                 dynamic_epoch ? geometry_epoch->value : nullptr,
                                                 hamiltonians, options.symmetry_tolerance,
                                                 workspace, system_errors, device_error);
    result = check_kernel_launch();
    if (!result.success()) {
      return result;
    }
    result = triangular_solve(blas, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, submission,
                              workspace.factor_pointers + bucket.solve_index_offset,
                              workspace.matrix_pointers + bucket.solve_index_offset);
    if (!result.success()) {
      return result;
    }
    result = triangular_solve(blas, CUBLAS_SIDE_RIGHT, CUBLAS_OP_T, submission,
                              workspace.factor_pointers + bucket.solve_index_offset,
                              workspace.matrix_pointers + bucket.solve_index_offset);
    if (!result.success()) {
      return result;
    }
    symmetrize_spin_transformed_bucket_kernel<<<static_cast<unsigned int>(bucket.solve_count),
                                                kThreadsPerSystem, 0, stream>>>(
        batch, layout, bucket, workspace, system_errors, device_error);
    result = check_kernel_launch();
    if (!result.success()) {
      return result;
    }
    result = symmetric_eigensolve(solver, parameters, CUSOLVER_EIG_MODE_VECTOR, submission,
                                  workspace.matrix_scratch_b + bucket.spin_matrix_scratch_offset,
                                  workspace.eigenvalue_scratch + bucket.spin_orbital_scratch_offset,
                                  workspace, workspace.info_a + bucket.solve_index_offset);
    if (!result.success()) {
      return result;
    }
    result = triangular_solve(blas, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, submission,
                              workspace.factor_pointers + bucket.solve_index_offset,
                              workspace.matrix_pointers + bucket.solve_index_offset);
    if (!result.success()) {
      return result;
    }
    validate_spin_eigenpairs_bucket_kernel<<<static_cast<unsigned int>(bucket.solve_count),
                                             kThreadsPerSystem, 0, stream>>>(
        batch, layout, bucket, workspace, system_errors, device_error);
    result = check_kernel_launch();
    if (!result.success()) {
      return result;
    }
    publish_spin_eigensystems_bucket_kernel<<<static_cast<unsigned int>(bucket.solve_count),
                                              kThreadsPerSystem, 0, stream>>>(
        batch, layout, bucket, workspace, results, system_errors);
    result = check_kernel_launch();
    if (!result.success()) {
      return result;
    }
  }
  return launch_success();
}

Gfn2EigensolverLaunchResult solve_gfn2_eigensystems_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const Gfn2EigensolverOverlapCache& cache,
    std::uint64_t geometry_generation, const double* hamiltonians,
    const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver, cusolverDnParams_t parameters,
    cublasHandle_t blas, const Gfn2EigensolverDeviceWorkspace& workspace,
    const Gfn2EigensolverDeviceResults& results, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  return solve_eigensystems_impl(batch, buckets, bucket_count, cache, geometry_generation, nullptr,
                                 hamiltonians, options, solver, parameters, blas, workspace,
                                 results, system_errors, device_error, stream);
}

Gfn2EigensolverLaunchResult solve_gfn2_spin_eigensystems_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2EigensolverBucket* buckets, std::int64_t bucket_count,
    const Gfn2EigensolverOverlapCache& cache, std::uint64_t geometry_generation,
    const double* hamiltonians, const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver,
    cusolverDnParams_t parameters, cublasHandle_t blas,
    const Gfn2EigensolverDeviceWorkspace& workspace, const Gfn2EigensolverDeviceResults& results,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  return solve_spin_eigensystems_impl(
      batch, layout, buckets, bucket_count, cache, geometry_generation, nullptr, hamiltonians,
      options, solver, parameters, blas, workspace, results, system_errors, device_error, stream);
}

Gfn2EigensolverLaunchResult solve_gfn2_eigensystems_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2EigensolverBucket* buckets,
    std::int64_t bucket_count, const Gfn2EigensolverOverlapCache& cache,
    const Gfn2GeometryEpochDevice& geometry_epoch, const double* hamiltonians,
    const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver, cusolverDnParams_t parameters,
    cublasHandle_t blas, const Gfn2EigensolverDeviceWorkspace& workspace,
    const Gfn2EigensolverDeviceResults& results, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  return solve_eigensystems_impl(batch, buckets, bucket_count, cache, 0u, &geometry_epoch,
                                 hamiltonians, options, solver, parameters, blas, workspace,
                                 results, system_errors, device_error, stream);
}

Gfn2EigensolverLaunchResult solve_gfn2_spin_eigensystems_cuda(
    const Gfn2EigensolverDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2EigensolverBucket* buckets, std::int64_t bucket_count,
    const Gfn2EigensolverOverlapCache& cache, const Gfn2GeometryEpochDevice& geometry_epoch,
    const double* hamiltonians, const Gfn2EigensolverOptions& options, cusolverDnHandle_t solver,
    cusolverDnParams_t parameters, cublasHandle_t blas,
    const Gfn2EigensolverDeviceWorkspace& workspace, const Gfn2EigensolverDeviceResults& results,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  return solve_spin_eigensystems_impl(
      batch, layout, buckets, bucket_count, cache, 0u, &geometry_epoch, hamiltonians, options,
      solver, parameters, blas, workspace, results, system_errors, device_error, stream);
}

}  // namespace gpuxtb::detail::cuda
