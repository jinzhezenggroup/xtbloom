#include <algorithm>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_scc_setup_eigensolver.cuh"

namespace xtbloom::detail::cuda {
namespace {

using SetupDiagnostic = Gfn2SccSetupEigensolverDiagnostic;
using SetupError = Gfn2SccSetupEigensolverError;
using SetupField = Gfn2SccSetupEigensolverField;

SetupDiagnostic failure(xtbloom_status_t status, SetupError error, SetupField field,
                        std::int64_t index = -1) noexcept {
  SetupDiagnostic result{};
  result.status = status;
  result.error = error;
  result.field = field;
  result.index = index;
  return result;
}

SetupDiagnostic capacity_failure(SetupError error, SetupField field, std::size_t required,
                                 std::size_t provided) noexcept {
  SetupDiagnostic result = failure(XTBLOOM_STATUS_INVALID_ARGUMENT, error, field);
  result.required_bytes = required;
  result.provided_bytes = provided;
  return result;
}

SetupDiagnostic cuda_failure(SetupError error, SetupField field, cudaError_t status) noexcept {
  SetupDiagnostic result =
      failure(error == SetupError::kAllocationFailed ? XTBLOOM_STATUS_ALLOCATION_FAILED
                                                     : XTBLOOM_STATUS_INTERNAL_ERROR,
              error, field);
  result.cuda_status = status;
  return result;
}

bool checked_add(std::size_t first, std::size_t second, std::size_t& result) noexcept {
  if (first > std::numeric_limits<std::size_t>::max() - second) {
    return false;
  }
  result = first + second;
  return true;
}

bool checked_multiply(std::size_t first, std::size_t second, std::size_t& result) noexcept {
  if (first != 0u && second > std::numeric_limits<std::size_t>::max() / first) {
    return false;
  }
  result = first * second;
  return true;
}

bool checked_add(std::int64_t first, std::int64_t second, std::int64_t& result) noexcept {
  if (first < 0 || second < 0 || first > std::numeric_limits<std::int64_t>::max() - second) {
    return false;
  }
  result = first + second;
  return true;
}

bool checked_multiply(std::int64_t first, std::int64_t second, std::int64_t& result) noexcept {
  if (first < 0 || second < 0 ||
      (first != 0 && second > std::numeric_limits<std::int64_t>::max() / first)) {
    return false;
  }
  result = first * second;
  return true;
}

bool align_up(std::size_t value, std::size_t alignment, std::size_t& result) noexcept {
  if (alignment == 0u) {
    return false;
  }
  const std::size_t remainder = value % alignment;
  if (remainder == 0u) {
    result = value;
    return true;
  }
  return checked_add(value, alignment - remainder, result);
}

template <typename T>
bool append_array(std::size_t elements, std::size_t& cursor, std::size_t& offset) noexcept {
  std::size_t aligned = 0u;
  std::size_t bytes = 0u;
  std::size_t end = 0u;
  if (!align_up(cursor, alignof(T), aligned) || !checked_multiply(elements, sizeof(T), bytes) ||
      !checked_add(aligned, bytes, end)) {
    return false;
  }
  offset = aligned;
  cursor = end;
  return true;
}

bool valid_options(const Gfn2EigensolverOptions& options) noexcept {
  const bool valid_strategy =
      options.strategy == Gfn2EigensolverStrategy::kAuto ||
      options.strategy == Gfn2EigensolverStrategy::kBatchedDivideAndConquer ||
      options.strategy == Gfn2EigensolverStrategy::kBatchedJacobi ||
      options.strategy == Gfn2EigensolverStrategy::kTridiagonalBisection;
  return std::isfinite(options.minimum_overlap_rcond) && options.minimum_overlap_rcond > 0.0 &&
         options.minimum_overlap_rcond <= 1.0 && std::isfinite(options.symmetry_tolerance) &&
         options.symmetry_tolerance >= 0.0 && valid_strategy &&
         (options.strategy != Gfn2EigensolverStrategy::kBatchedJacobi || options.jacobi != nullptr);
}

/* This version gate enables the Graph-capable production route, but it is not
 * a per-shape provider guarantee. CUDA 12.9 vector-mode XsyevBatched stops
 * being capturable above 512 orbitals, so kAuto diverts 513-1024-orbital
 * singletons to the device-only tridiagonal provider. Keep the decision
 * conservative for older or mismatched runtime libraries: callers can still
 * execute through the explicit uncaptured-segment contract instead of
 * discovering an unsupported provider call halfway through capture. */
Gfn2SccIterationProviderCaptureMode detect_provider_capture_mode(cublasHandle_t blas) noexcept {
#if CUDART_VERSION >= 12090 && CUBLAS_VERSION >= 120901 && CUSOLVER_VERSION >= 11705
  int runtime_version = 0;
  int blas_version = 0;
  int solver_major = 0;
  int solver_minor = 0;
  int solver_patch = 0;
  if (cudaRuntimeGetVersion(&runtime_version) != cudaSuccess ||
      cublasGetVersion(blas, &blas_version) != CUBLAS_STATUS_SUCCESS ||
      cusolverGetProperty(MAJOR_VERSION, &solver_major) != CUSOLVER_STATUS_SUCCESS ||
      cusolverGetProperty(MINOR_VERSION, &solver_minor) != CUSOLVER_STATUS_SUCCESS ||
      cusolverGetProperty(PATCH_LEVEL, &solver_patch) != CUSOLVER_STATUS_SUCCESS) {
    (void)cudaGetLastError();
    return Gfn2SccIterationProviderCaptureMode::kUncapturedSegmentRequired;
  }
  const int solver_version = solver_major * 1000 + solver_minor * 100 + solver_patch;
  if (runtime_version >= 12090 && blas_version >= 120901 && solver_version >= 11705) {
    return Gfn2SccIterationProviderCaptureMode::kGraphSupported;
  }
#else
  (void)blas;
#endif
  return Gfn2SccIterationProviderCaptureMode::kUncapturedSegmentRequired;
}

bool valid_buckets(const Gfn2RaggedTopologyView& topology,
                   const std::vector<Gfn2EigensolverBucket>& buckets) noexcept {
  if (topology.bucket_count <= 0 || static_cast<std::uint64_t>(topology.bucket_count) !=
                                        static_cast<std::uint64_t>(buckets.size())) {
    return false;
  }
  std::int64_t systems = 0;
  std::int64_t matrices = 0;
  std::int64_t orbitals = 0;
  for (const Gfn2EigensolverBucket& bucket : buckets) {
    std::int64_t matrix_stride = 0;
    std::int64_t matrix_span = 0;
    std::int64_t orbital_span = 0;
    if (bucket.orbital_count <= 0 || bucket.system_count <= 0 ||
        bucket.system_index_offset != systems || bucket.matrix_scratch_offset != matrices ||
        bucket.orbital_scratch_offset != orbitals ||
        !checked_multiply(static_cast<std::int64_t>(bucket.orbital_count),
                          static_cast<std::int64_t>(bucket.orbital_count), matrix_stride) ||
        !checked_multiply(matrix_stride, static_cast<std::int64_t>(bucket.system_count),
                          matrix_span) ||
        !checked_multiply(static_cast<std::int64_t>(bucket.orbital_count),
                          static_cast<std::int64_t>(bucket.system_count), orbital_span) ||
        !checked_add(systems, bucket.system_count, systems) ||
        !checked_add(matrices, matrix_span, matrices) ||
        !checked_add(orbitals, orbital_span, orbitals)) {
      return false;
    }
  }
  return systems == topology.batch_size && matrices == topology.total_matrix_elements &&
         orbitals == topology.total_orbitals;
}

void hash_append(std::uint64_t value, std::uint64_t& hash) noexcept {
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  for (unsigned int byte = 0u; byte < 8u; ++byte) {
    hash ^= (value >> (byte * 8u)) & 0xffu;
    hash *= kPrime;
  }
}

template <typename T>
void hash_pointer(const T* pointer, std::uint64_t& hash) noexcept {
  hash_append(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer)), hash);
}

std::uint64_t binding_provenance_seal(const Gfn2SccSetupEigensolverBinding& binding,
                                      std::uint64_t salt) noexcept {
  std::uint64_t hash = salt;
  const auto count = [&](auto value) { hash_append(static_cast<std::uint64_t>(value), hash); };
  const auto pointer = [&](const auto* value) { hash_pointer(value, hash); };

  count(binding.batch.batch_size);
  count(binding.batch.total_orbitals);
  count(binding.batch.total_matrix_elements);
  count(binding.batch.orbital_offset_count);
  count(binding.batch.matrix_offset_count);
  count(binding.batch.bucket_system_count);
  count(binding.batch.active_elements);
  count(binding.batch.plan_token);
  pointer(binding.batch.orbital_offsets);
  pointer(binding.batch.matrix_offsets);
  pointer(binding.batch.bucket_systems);
  pointer(binding.batch.active);

  pointer(binding.provider.buckets);
  count(binding.provider.bucket_count);
  pointer(binding.provider.solver);
  pointer(binding.provider.parameters);
  pointer(binding.provider.blas);
  pointer(binding.provider.device_workspace);
  count(binding.provider.device_workspace_bytes);
  pointer(binding.provider.host_workspace);
  count(binding.provider.host_workspace_bytes);
  count(binding.provider.requirements.solver_device_workspace_bytes);
  count(binding.provider.requirements.solver_host_workspace_bytes);
  count(binding.provider.capture_mode);
  count(binding.provider.reserved);
  count(binding.provider.plan_token);

  pointer(binding.cache.cholesky_factors);
  count(binding.cache.factor_elements);
  pointer(binding.cache.geometry_generations);
  count(binding.cache.generation_elements);
  pointer(binding.cache.factor_statuses);
  count(binding.cache.status_elements);
  count(binding.cache.plan_token);

  pointer(binding.workspace.matrix_scratch_a);
  count(binding.workspace.matrix_a_elements);
  pointer(binding.workspace.matrix_scratch_b);
  count(binding.workspace.matrix_b_elements);
  pointer(binding.workspace.eigenvalue_scratch);
  count(binding.workspace.eigenvalue_elements);
  pointer(binding.workspace.factor_pointers);
  count(binding.workspace.factor_pointer_elements);
  pointer(binding.workspace.matrix_pointers);
  count(binding.workspace.matrix_pointer_elements);
  pointer(binding.workspace.info_a);
  count(binding.workspace.info_a_elements);
  pointer(binding.workspace.info_b);
  count(binding.workspace.info_b_elements);
  pointer(binding.workspace.eligible);
  count(binding.workspace.eligible_elements);
  pointer(binding.workspace.sequence_active);
  count(binding.workspace.sequence_active_elements);
  pointer(static_cast<const std::byte*>(binding.workspace.solver_device_workspace));
  count(binding.workspace.solver_device_workspace_bytes);
  pointer(static_cast<const std::byte*>(binding.workspace.solver_host_workspace));
  count(binding.workspace.solver_host_workspace_bytes);
  count(binding.workspace.plan_token);
  pointer(binding.workspace.compact_systems);
  count(binding.workspace.compact_system_elements);
  pointer(binding.workspace.compact_source_slots);
  count(binding.workspace.compact_source_slot_elements);
  pointer(binding.workspace.bucket_activity);
  count(binding.workspace.bucket_activity_elements);

  pointer(binding.overlap_input);
  count(binding.overlap_elements);
  pointer(binding.setup_system_errors);
  count(binding.setup_system_error_elements);
  pointer(binding.setup_device_error);
  pointer(static_cast<const std::byte*>(binding.owner_identity));
  pointer(static_cast<const std::byte*>(binding.setup_device_arena));
  count(binding.setup_device_arena_bytes);
  pointer(binding.geometry_epoch);
  count(binding.geometry_epoch_elements);
  count(binding.geometry_generation);
  count(binding.iteration_layout_fingerprint);
  count(binding.plan_token);
  return hash == 0u ? 1u : hash;
}

std::uint64_t layout_fingerprint(const Gfn2SccSetupEigensolverRequirements& requirements,
                                 std::int64_t batch, std::int64_t orbitals,
                                 std::int64_t matrices) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  hash_append(static_cast<std::uint64_t>(batch), hash);
  hash_append(static_cast<std::uint64_t>(orbitals), hash);
  hash_append(static_cast<std::uint64_t>(matrices), hash);
  hash_append(static_cast<std::uint64_t>(requirements.setup_device_bytes), hash);
  hash_append(static_cast<std::uint64_t>(requirements.provider.solver_device_workspace_bytes),
              hash);
  hash_append(static_cast<std::uint64_t>(requirements.provider.solver_host_workspace_bytes), hash);
  hash_append(requirements.plan_token, hash);
  hash_append(requirements.geometry_generation, hash);
  return hash == 0u ? 1u : hash;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool make_range(const void* pointer, std::size_t bytes, AddressRange& range) noexcept {
  if (bytes == 0u) {
    range = {};
    return pointer == nullptr;
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

template <typename T>
bool make_elements_range(const T* pointer, std::int64_t elements, AddressRange& range) noexcept {
  if (elements < 0 ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(T))) {
    return false;
  }
  return make_range(pointer, static_cast<std::size_t>(elements) * sizeof(T), range);
}

bool overlaps(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin < second.end && second.begin < first.end;
}

bool contains(const AddressRange& outer, const AddressRange& inner) noexcept {
  return inner.begin >= outer.begin && inner.end <= outer.end;
}

template <std::size_t Count>
bool pairwise_disjoint(const std::array<AddressRange, Count>& ranges) noexcept {
  for (std::size_t first = 0u; first < Count; ++first) {
    for (std::size_t second = first + 1u; second < Count; ++second) {
      if (overlaps(ranges[first], ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

bool cuda_accessible(const void* pointer, cudaPointerAttributes& attributes,
                     cudaError_t& status) noexcept {
  status = cudaPointerGetAttributes(&attributes, pointer);
  if (status != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged;
}

__global__ void snapshot_refactor_activity_kernel(std::int64_t batch_size,
                                                  const std::uint8_t* source,
                                                  const std::uint32_t* request_error,
                                                  std::uint8_t* destination) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system < batch_size) {
    destination[system] = request_error != nullptr && *request_error != 0u ? 0u : source[system];
  }
}

const Gfn2SccSetupEigensolverRequirements& empty_requirements() noexcept {
  static const Gfn2SccSetupEigensolverRequirements empty{};
  return empty;
}

bool same_iteration_requirements(const Gfn2SccIterationArenaRequirements& first,
                                 const Gfn2SccIterationArenaRequirements& second) noexcept {
  return first.abi_version == second.abi_version && first.reserved == second.reserved &&
         first.alignment == second.alignment &&
         first.persistent_offset == second.persistent_offset &&
         first.persistent_bytes == second.persistent_bytes &&
         first.workspace_offset == second.workspace_offset &&
         first.workspace_bytes == second.workspace_bytes &&
         first.provider_device_offset == second.provider_device_offset &&
         first.provider_device_bytes == second.provider_device_bytes &&
         first.total_bytes == second.total_bytes && first.plan_token == second.plan_token &&
         first.layout_fingerprint == second.layout_fingerprint;
}

}  // namespace

struct Gfn2SccSetupEigensolver::Impl {
  cusolverDnHandle_t solver = nullptr;
  cusolverDnParams_t parameters = nullptr;
  cublasHandle_t blas = nullptr;
  std::vector<Gfn2EigensolverBucket> buckets;
  Gfn2EigensolverOptions options{};
  Gfn2SccIterationProviderCaptureMode capture_mode =
      Gfn2SccIterationProviderCaptureMode::kUncapturedSegmentRequired;
  Gfn2SccSetupEigensolverRequirements requirements{};
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t total_matrices = 0;
  std::int64_t total_spin_channels = 0;
  std::int64_t total_spin_orbitals = 0;
  std::int64_t total_spin_matrices = 0;
  std::uint64_t binding_salt = 0u;
  void* pinned_overlap = nullptr;

  ~Impl() {
    if (pinned_overlap != nullptr) {
      (void)cudaFreeHost(pinned_overlap);
    }
  }
};

Gfn2SccSetupEigensolver::Gfn2SccSetupEigensolver() noexcept = default;
Gfn2SccSetupEigensolver::~Gfn2SccSetupEigensolver() = default;
Gfn2SccSetupEigensolver::Gfn2SccSetupEigensolver(Gfn2SccSetupEigensolver&&) noexcept = default;
Gfn2SccSetupEigensolver& Gfn2SccSetupEigensolver::operator=(Gfn2SccSetupEigensolver&&) noexcept =
    default;

Gfn2SccSetupEigensolverDiagnostic Gfn2SccSetupEigensolver::create(
    const Gfn2SccSetupTopology& topology, const double* host_overlap,
    std::int64_t host_overlap_elements, std::uint64_t geometry_generation, std::uint64_t plan_token,
    cusolverDnHandle_t solver, cusolverDnParams_t parameters, cublasHandle_t blas,
    const Gfn2EigensolverOptions& options, Gfn2SccSetupEigensolver& output) noexcept {
  if (!topology.valid()) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidTopology,
                   SetupField::kTopology);
  }
  const Gfn2RaggedTopologyView& host_topology = topology.host_topology();
  const Gfn2WavefunctionLayoutView& host_wavefunction = topology.host_wavefunction_layout();
  if (plan_token == 0u || host_topology.plan_token != plan_token) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kCrossPlan, SetupField::kPlanToken);
  }
  if (geometry_generation == 0u) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidGeneration,
                   SetupField::kGeometryGeneration);
  }
  if (host_overlap == nullptr || host_overlap_elements != host_topology.total_matrix_elements ||
      reinterpret_cast<std::uintptr_t>(host_overlap) % alignof(double) != 0u) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidOverlap,
                   SetupField::kOverlap);
  }
  if (!valid_options(options)) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidOptions,
                   SetupField::kOptions);
  }
  if (solver == nullptr || parameters == nullptr || blas == nullptr) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidProvider,
                   SetupField::kHandles);
  }
  if (!valid_buckets(host_topology, topology.eigensolver_buckets())) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidTopology,
                   SetupField::kTopology);
  }

  try {
    std::unique_ptr<Impl> candidate(new (std::nothrow) Impl());
    if (candidate == nullptr) {
      return failure(XTBLOOM_STATUS_ALLOCATION_FAILED, SetupError::kAllocationFailed,
                     SetupField::kOverlap);
    }
    candidate->solver = solver;
    candidate->parameters = parameters;
    candidate->blas = blas;
    candidate->capture_mode = detect_provider_capture_mode(blas);
    candidate->buckets = topology.eigensolver_buckets();
    candidate->options = options;
    candidate->batch_size = host_topology.batch_size;
    candidate->total_atoms = host_topology.total_atoms;
    candidate->total_shells = host_topology.total_shells;
    candidate->total_orbitals = host_topology.total_orbitals;
    candidate->total_matrices = host_topology.total_matrix_elements;
    candidate->total_spin_channels = host_wavefunction.total_spin_channels;
    candidate->total_spin_orbitals = host_wavefunction.total_spin_orbitals;
    candidate->total_spin_matrices = host_wavefunction.total_spin_matrix_elements;

    std::size_t matrix_bytes = 0u;
    std::size_t query_matrix_bytes = 0u;
    std::size_t query_orbital_bytes = 0u;
    std::size_t query_orbital_offset = 0u;
    std::size_t query_bytes = 0u;
    if (!checked_multiply(static_cast<std::size_t>(candidate->total_matrices), sizeof(double),
                          matrix_bytes) ||
        !checked_multiply(static_cast<std::size_t>(candidate->total_spin_matrices), sizeof(double),
                          query_matrix_bytes) ||
        !checked_multiply(static_cast<std::size_t>(candidate->total_spin_orbitals), sizeof(double),
                          query_orbital_bytes) ||
        !align_up(query_matrix_bytes, kGfn2SccSetupEigensolverArenaAlignment,
                  query_orbital_offset) ||
        !checked_add(query_orbital_offset, query_orbital_bytes, query_bytes)) {
      return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kCountOverflow,
                     SetupField::kWorkspaceQuery);
    }

    void* query_storage = nullptr;
    cudaError_t cuda_status = cudaMalloc(&query_storage, query_bytes);
    if (cuda_status != cudaSuccess) {
      return cuda_failure(cuda_status == cudaErrorMemoryAllocation ? SetupError::kAllocationFailed
                                                                   : SetupError::kCudaError,
                          SetupField::kWorkspaceQuery, cuda_status);
    }
    auto* const query_bytes_base = static_cast<std::byte*>(query_storage);
    auto* const query_matrix = reinterpret_cast<double*>(query_bytes_base);
    auto* const query_eigenvalues =
        reinterpret_cast<double*>(query_bytes_base + query_orbital_offset);
    Gfn2EigensolverLaunchResult query_result{};
    for (const Gfn2EigensolverBucket& bucket : candidate->buckets) {
      query_result = query_gfn2_eigensolver_bucket_workspace_cuda(solver, parameters, bucket,
                                                                  query_matrix, query_eigenvalues,
                                                                  candidate->requirements.provider);
      if (!query_result.success()) {
        break;
      }
      /* Physical overlap factorization and spin-expanded Hamiltonian solves
       * share one provider workspace. Mixed batches can have solve_count >
       * system_count, so both reachable submission capacities must contribute
       * to the retained maximum. */
      query_result = query_gfn2_spin_eigensolver_bucket_workspace_cuda(
          solver, parameters, bucket, query_matrix, query_eigenvalues,
          candidate->requirements.provider);
      if (!query_result.success()) {
        break;
      }
      if (gfn2_eigensolver_uses_tridiagonal(candidate->options, bucket)) {
        query_result = query_gfn2_tridiagonal_bucket_workspace_cuda(
            solver, bucket, query_matrix, query_eigenvalues, candidate->requirements.provider);
        if (!query_result.success()) {
          break;
        }
      }
      if (gfn2_eigensolver_uses_jacobi(candidate->options, bucket.orbital_count)) {
        query_result = query_gfn2_jacobi_bucket_workspace_cuda(
            solver, candidate->options.jacobi, bucket, query_matrix, query_eigenvalues,
            candidate->requirements.provider);
        if (!query_result.success()) {
          break;
        }
        const Gfn2EigensolverBucket spin_submission{
            bucket.orbital_count, bucket.solve_count, bucket.solve_index_offset,
            bucket.spin_matrix_scratch_offset, bucket.spin_orbital_scratch_offset};
        query_result = query_gfn2_jacobi_bucket_workspace_cuda(
            solver, candidate->options.jacobi, spin_submission, query_matrix, query_eigenvalues,
            candidate->requirements.provider);
        if (!query_result.success()) {
          break;
        }
      }
    }
    const cudaError_t free_status = cudaFree(query_storage);
    if (!query_result.success()) {
      SetupDiagnostic diagnostic =
          failure(XTBLOOM_STATUS_EIGENSOLVER_FAILED, SetupError::kWorkspaceQueryFailed,
                  SetupField::kWorkspaceQuery);
      diagnostic.cuda_status = query_result.cuda_status;
      diagnostic.cublas_status = query_result.cublas_status;
      diagnostic.cusolver_status = query_result.cusolver_status;
      return diagnostic;
    }
    if (free_status != cudaSuccess) {
      return cuda_failure(SetupError::kCudaError, SetupField::kWorkspaceQuery, free_status);
    }

    candidate->requirements.plan_token = plan_token;
    candidate->requirements.geometry_generation = geometry_generation;
    candidate->requirements.provider_host_workspace_bytes =
        candidate->requirements.provider.solver_host_workspace_bytes;
    std::size_t cursor = 0u;
    const std::size_t matrices = static_cast<std::size_t>(candidate->total_matrices);
    const std::size_t batch = static_cast<std::size_t>(candidate->batch_size);
    if (!append_array<double>(matrices, cursor, candidate->requirements.overlap_input_offset) ||
        !append_array<std::uint8_t>(batch, cursor, candidate->requirements.active_offset) ||
        !append_array<std::uint32_t>(batch, cursor, candidate->requirements.system_error_offset) ||
        !append_array<std::uint32_t>(1u, cursor, candidate->requirements.device_error_offset) ||
        !append_array<double>(matrices, cursor, candidate->requirements.cache_factor_offset) ||
        !append_array<std::uint64_t>(batch, cursor,
                                     candidate->requirements.cache_generation_offset) ||
        !append_array<std::uint32_t>(batch, cursor, candidate->requirements.cache_status_offset) ||
        !align_up(cursor, kGfn2SccSetupEigensolverArenaAlignment, cursor)) {
      return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kCountOverflow,
                     SetupField::kSetupArena);
    }
    candidate->requirements.setup_device_bytes = cursor;
    candidate->requirements.layout_fingerprint =
        layout_fingerprint(candidate->requirements, candidate->batch_size,
                           candidate->total_orbitals, candidate->total_matrices);
    candidate->binding_salt = candidate->requirements.layout_fingerprint;
    hash_append(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(candidate.get())),
                candidate->binding_salt);
    if (candidate->binding_salt == 0u) {
      candidate->binding_salt = 1u;
    }

    cuda_status = cudaMallocHost(&candidate->pinned_overlap, matrix_bytes);
    if (cuda_status != cudaSuccess) {
      return cuda_failure(cuda_status == cudaErrorMemoryAllocation ? SetupError::kAllocationFailed
                                                                   : SetupError::kCudaError,
                          SetupField::kOverlap, cuda_status);
    }
    std::memcpy(candidate->pinned_overlap, host_overlap, matrix_bytes);

    Gfn2SccSetupEigensolver replacement;
    replacement.impl_ = std::move(candidate);
    output = std::move(replacement);
    return {};
  } catch (const std::bad_alloc&) {
    return failure(XTBLOOM_STATUS_ALLOCATION_FAILED, SetupError::kAllocationFailed,
                   SetupField::kTopology);
  } catch (...) {
    return failure(XTBLOOM_STATUS_INTERNAL_ERROR, SetupError::kCudaError,
                   SetupField::kWorkspaceQuery);
  }
}

bool Gfn2SccSetupEigensolver::valid() const noexcept { return impl_ != nullptr; }

std::size_t Gfn2SccSetupEigensolver::retained_host_bytes() const noexcept {
  if (impl_ == nullptr) return 0u;
  return sizeof(*impl_) + impl_->buckets.capacity() * sizeof(Gfn2EigensolverBucket) +
         (impl_->pinned_overlap == nullptr
              ? 0u
              : static_cast<std::size_t>(impl_->total_matrices) * sizeof(double));
}

const Gfn2SccSetupEigensolverRequirements& Gfn2SccSetupEigensolver::requirements() const noexcept {
  return impl_ == nullptr ? empty_requirements() : impl_->requirements;
}

Gfn2SccSetupEigensolverDiagnostic Gfn2SccSetupEigensolver::bind_and_factor_overlap_async(
    const Gfn2RaggedTopologyView& device_topology, const Gfn2SccIterationDevicePlan& iteration_plan,
    const Gfn2SccIterationArenaRequirements& iteration_requirements, void* iteration_arena,
    std::size_t iteration_arena_bytes, const Gfn2SccIterationDeviceWorkspace& iteration_workspace,
    void* provider_host_workspace, std::size_t provider_host_workspace_bytes,
    void* setup_device_arena, std::size_t setup_device_arena_bytes,
    Gfn2SccSetupEigensolverBinding& binding, cudaStream_t stream) const noexcept {
  if (impl_ == nullptr) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidProvider,
                   SetupField::kHandles);
  }
  const auto& own = impl_->requirements;
  Gfn2SccIterationArenaRequirements expected_iteration_requirements{};
  const Gfn2SccIterationArenaDiagnostic iteration_query =
      query_gfn2_scc_iteration_arena_requirements_cuda(iteration_plan, own.provider,
                                                       expected_iteration_requirements);
  if (!iteration_query.success() ||
      !same_iteration_requirements(iteration_requirements, expected_iteration_requirements)) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidIterationProvenance,
                   SetupField::kIterationArena);
  }
  if (device_topology.memory_space != Gfn2PlanMemorySpace::kCudaDevice ||
      device_topology.plan_token != own.plan_token ||
      device_topology.batch_size != impl_->batch_size ||
      device_topology.total_atoms != impl_->total_atoms ||
      device_topology.total_shells != impl_->total_shells ||
      device_topology.total_orbitals != impl_->total_orbitals ||
      device_topology.total_matrix_elements != impl_->total_matrices ||
      device_topology.bucket_count != static_cast<std::int64_t>(impl_->buckets.size())) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT,
                   device_topology.plan_token == own.plan_token ? SetupError::kInvalidTopology
                                                                : SetupError::kCrossPlan,
                   SetupField::kTopology);
  }
  if (iteration_plan.plan_token != own.plan_token ||
      iteration_plan.topology.plan_token != own.plan_token ||
      iteration_plan.wavefunction_layout.plan_token != own.plan_token ||
      iteration_plan.topology.batch_size != impl_->batch_size ||
      iteration_plan.topology.total_atoms != impl_->total_atoms ||
      iteration_plan.topology.total_shells != impl_->total_shells ||
      iteration_plan.topology.total_orbitals != impl_->total_orbitals ||
      iteration_plan.topology.total_matrix_elements != impl_->total_matrices ||
      iteration_plan.wavefunction_layout.batch_size != impl_->batch_size ||
      iteration_plan.wavefunction_layout.total_spin_channels != impl_->total_spin_channels ||
      iteration_plan.wavefunction_layout.total_spin_orbitals != impl_->total_spin_orbitals ||
      iteration_plan.wavefunction_layout.total_spin_matrix_elements != impl_->total_spin_matrices) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kCrossPlan,
                   SetupField::kIterationArena);
  }
  const Gfn2PlanSchemaDiagnostic topology_shape =
      validate_gfn2_topology_binding(device_topology, Gfn2PlanMemorySpace::kCudaDevice);
  if (topology_shape.error != Gfn2PlanSchemaError::kSuccess) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidTopology,
                   SetupField::kTopology, topology_shape.index);
  }

  if (setup_device_arena == nullptr) {
    return capacity_failure(SetupError::kNullArena, SetupField::kSetupArena, own.setup_device_bytes,
                            setup_device_arena_bytes);
  }
  if (reinterpret_cast<std::uintptr_t>(setup_device_arena) % own.alignment != 0u) {
    return capacity_failure(SetupError::kMisalignedArena, SetupField::kSetupArena,
                            own.setup_device_bytes, setup_device_arena_bytes);
  }
  if (setup_device_arena_bytes < own.setup_device_bytes) {
    return capacity_failure(SetupError::kInsufficientArena, SetupField::kSetupArena,
                            own.setup_device_bytes, setup_device_arena_bytes);
  }
  cudaPointerAttributes setup_attributes{};
  cudaError_t cuda_status = cudaSuccess;
  if (!cuda_accessible(setup_device_arena, setup_attributes, cuda_status)) {
    SetupDiagnostic diagnostic = failure(XTBLOOM_STATUS_INVALID_ARGUMENT,
                                         SetupError::kInvalidArenaMemory, SetupField::kSetupArena);
    diagnostic.cuda_status = cuda_status;
    return diagnostic;
  }

  if (iteration_arena == nullptr ||
      reinterpret_cast<std::uintptr_t>(iteration_arena) % kGfn2SccIterationArenaAlignment != 0u) {
    return capacity_failure(
        iteration_arena == nullptr ? SetupError::kNullArena : SetupError::kMisalignedArena,
        SetupField::kIterationArena, iteration_requirements.total_bytes, iteration_arena_bytes);
  }
  if (iteration_arena_bytes < iteration_requirements.total_bytes) {
    return capacity_failure(SetupError::kInsufficientArena, SetupField::kIterationArena,
                            iteration_requirements.total_bytes, iteration_arena_bytes);
  }
  cudaPointerAttributes iteration_attributes{};
  if (!cuda_accessible(iteration_arena, iteration_attributes, cuda_status)) {
    SetupDiagnostic diagnostic =
        failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidArenaMemory,
                SetupField::kIterationArena);
    diagnostic.cuda_status = cuda_status;
    return diagnostic;
  }

  std::size_t persistent_end = 0u;
  std::size_t workspace_end = 0u;
  std::size_t provider_end = 0u;
  const bool valid_iteration_requirements =
      iteration_requirements.abi_version == kGfn2SccIterationArenaAbiVersion &&
      iteration_requirements.alignment == kGfn2SccIterationArenaAlignment &&
      iteration_requirements.plan_token == own.plan_token &&
      iteration_requirements.layout_fingerprint != 0u &&
      iteration_requirements.provider_device_bytes == own.provider.solver_device_workspace_bytes &&
      checked_add(iteration_requirements.persistent_offset, iteration_requirements.persistent_bytes,
                  persistent_end) &&
      checked_add(iteration_requirements.workspace_offset, iteration_requirements.workspace_bytes,
                  workspace_end) &&
      checked_add(iteration_requirements.provider_device_offset,
                  iteration_requirements.provider_device_bytes, provider_end) &&
      persistent_end <= iteration_requirements.workspace_offset &&
      workspace_end <= iteration_requirements.provider_device_offset &&
      provider_end <= iteration_requirements.total_bytes &&
      iteration_requirements.provider_device_offset % kGfn2SccIterationArenaAlignment == 0u &&
      iteration_requirements.total_bytes % kGfn2SccIterationArenaAlignment == 0u;
  if (!valid_iteration_requirements) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidIterationProvenance,
                   SetupField::kIterationArena);
  }

  AddressRange setup_range{};
  AddressRange iteration_range{};
  if (!make_range(setup_device_arena, own.setup_device_bytes, setup_range) ||
      !make_range(iteration_arena, iteration_requirements.total_bytes, iteration_range) ||
      overlaps(setup_range, iteration_range)) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidIterationProvenance,
                   SetupField::kIterationArena);
  }

  const std::int64_t batch = impl_->batch_size;
  const std::int64_t matrices = impl_->total_matrices;
  const std::int64_t orbitals = impl_->total_orbitals;
  const std::int64_t spin_channels = impl_->total_spin_channels;
  const std::int64_t spin_matrices = impl_->total_spin_matrices;
  const std::int64_t spin_orbitals = impl_->total_spin_orbitals;
  const std::int64_t bucket_count = static_cast<std::int64_t>(impl_->buckets.size());
  const Gfn2EigensolverDeviceWorkspace& eigensolver_workspace =
      iteration_workspace.eigensolver_workspace;
  const bool valid_counts =
      iteration_workspace.plan_token == own.plan_token &&
      iteration_workspace.ledger.plan_token == own.plan_token &&
      iteration_workspace.ledger.batch_elements == batch &&
      iteration_workspace.ledger.active_mask != nullptr &&
      eigensolver_workspace.plan_token == own.plan_token &&
      eigensolver_workspace.matrix_a_elements == spin_matrices &&
      eigensolver_workspace.matrix_b_elements == spin_matrices &&
      eigensolver_workspace.eigenvalue_elements == spin_orbitals &&
      eigensolver_workspace.factor_pointer_elements == spin_channels &&
      eigensolver_workspace.matrix_pointer_elements == spin_channels &&
      eigensolver_workspace.info_a_elements == spin_channels &&
      eigensolver_workspace.info_b_elements == spin_channels &&
      eigensolver_workspace.eligible_elements == spin_channels &&
      eigensolver_workspace.sequence_active_elements == 1 &&
      eigensolver_workspace.compact_system_elements == spin_channels &&
      eigensolver_workspace.compact_source_slot_elements == spin_channels &&
      eigensolver_workspace.bucket_activity_elements == bucket_count &&
      eigensolver_workspace.solver_device_workspace_bytes ==
          own.provider.solver_device_workspace_bytes &&
      eigensolver_workspace.solver_host_workspace_bytes == own.provider.solver_host_workspace_bytes;
  if (!valid_counts) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidIterationWorkspace,
                   SetupField::kIterationWorkspace);
  }

  auto* const iteration_bytes = static_cast<std::byte*>(iteration_arena);
  AddressRange workspace_segment{};
  AddressRange provider_segment{};
  if (!make_range(iteration_bytes + iteration_requirements.workspace_offset,
                  iteration_requirements.workspace_bytes, workspace_segment) ||
      !make_range(iteration_requirements.provider_device_bytes == 0u
                      ? nullptr
                      : iteration_bytes + iteration_requirements.provider_device_offset,
                  iteration_requirements.provider_device_bytes, provider_segment)) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidIterationProvenance,
                   SetupField::kIterationArena);
  }

  std::array<AddressRange, 13> workspace_ranges{};
  if (!make_elements_range(iteration_workspace.ledger.active_mask, batch, workspace_ranges[0]) ||
      !make_elements_range(eigensolver_workspace.matrix_scratch_a, spin_matrices,
                           workspace_ranges[1]) ||
      !make_elements_range(eigensolver_workspace.matrix_scratch_b, spin_matrices,
                           workspace_ranges[2]) ||
      !make_elements_range(eigensolver_workspace.eigenvalue_scratch, spin_orbitals,
                           workspace_ranges[3]) ||
      !make_elements_range(eigensolver_workspace.factor_pointers, spin_channels,
                           workspace_ranges[4]) ||
      !make_elements_range(eigensolver_workspace.matrix_pointers, spin_channels,
                           workspace_ranges[5]) ||
      !make_elements_range(eigensolver_workspace.info_a, spin_channels, workspace_ranges[6]) ||
      !make_elements_range(eigensolver_workspace.info_b, spin_channels, workspace_ranges[7]) ||
      !make_elements_range(eigensolver_workspace.eligible, spin_channels, workspace_ranges[8]) ||
      !make_elements_range(eigensolver_workspace.sequence_active, 1, workspace_ranges[9]) ||
      !make_elements_range(eigensolver_workspace.compact_systems, spin_channels,
                           workspace_ranges[10]) ||
      !make_elements_range(eigensolver_workspace.compact_source_slots, spin_channels,
                           workspace_ranges[11]) ||
      !make_elements_range(eigensolver_workspace.bucket_activity, bucket_count,
                           workspace_ranges[12]) ||
      !pairwise_disjoint(workspace_ranges)) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidIterationWorkspace,
                   SetupField::kIterationWorkspace);
  }
  for (const AddressRange& range : workspace_ranges) {
    if (!contains(workspace_segment, range)) {
      return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidIterationProvenance,
                     SetupField::kIterationWorkspace);
    }
  }
  if ((iteration_requirements.provider_device_bytes == 0u &&
       eigensolver_workspace.solver_device_workspace != nullptr) ||
      (iteration_requirements.provider_device_bytes != 0u &&
       (eigensolver_workspace.solver_device_workspace !=
            iteration_bytes + iteration_requirements.provider_device_offset ||
        reinterpret_cast<std::uintptr_t>(eigensolver_workspace.solver_device_workspace) %
                kGfn2SccIterationArenaAlignment !=
            0u))) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidIterationProvenance,
                   SetupField::kIterationWorkspace);
  }
  AddressRange provider_workspace_range{};
  if (!make_range(eigensolver_workspace.solver_device_workspace,
                  eigensolver_workspace.solver_device_workspace_bytes, provider_workspace_range) ||
      (iteration_requirements.provider_device_bytes != 0u &&
       (!contains(provider_segment, provider_workspace_range) ||
        overlaps(workspace_segment, provider_workspace_range)))) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidIterationProvenance,
                   SetupField::kIterationWorkspace);
  }

  if (provider_host_workspace_bytes < own.provider.solver_host_workspace_bytes ||
      eigensolver_workspace.solver_host_workspace != provider_host_workspace) {
    return capacity_failure(SetupError::kInvalidHostWorkspace, SetupField::kProviderHostWorkspace,
                            own.provider.solver_host_workspace_bytes,
                            provider_host_workspace_bytes);
  }
  if (own.provider.solver_host_workspace_bytes == 0u) {
    if (provider_host_workspace != nullptr ||
        eigensolver_workspace.solver_host_workspace != nullptr) {
      return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidHostWorkspace,
                     SetupField::kProviderHostWorkspace);
    }
  } else {
    if (provider_host_workspace == nullptr ||
        reinterpret_cast<std::uintptr_t>(provider_host_workspace) % alignof(std::max_align_t) !=
            0u) {
      return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidHostWorkspace,
                     SetupField::kProviderHostWorkspace);
    }
    cudaPointerAttributes host_attributes{};
    cuda_status = cudaPointerGetAttributes(&host_attributes, provider_host_workspace);
    if (cuda_status != cudaSuccess || host_attributes.type != cudaMemoryTypeHost) {
      if (cuda_status != cudaSuccess) {
        (void)cudaGetLastError();
      }
      SetupDiagnostic diagnostic =
          failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidHostWorkspace,
                  SetupField::kProviderHostWorkspace);
      diagnostic.cuda_status = cuda_status;
      return diagnostic;
    }
  }

  auto* const setup = static_cast<std::byte*>(setup_device_arena);
  Gfn2SccSetupEigensolverBinding candidate{};
  candidate.batch = {batch,
                     orbitals,
                     matrices,
                     device_topology.batch_orbital_offset_count,
                     device_topology.matrix_offset_count,
                     device_topology.bucket_system_count,
                     batch,
                     own.plan_token,
                     device_topology.batch_orbital_offsets,
                     device_topology.matrix_offsets,
                     device_topology.bucket_systems,
                     iteration_workspace.ledger.active_mask};
  candidate.provider.buckets = impl_->buckets.data();
  candidate.provider.bucket_count = static_cast<std::int64_t>(impl_->buckets.size());
  candidate.provider.solver = impl_->solver;
  candidate.provider.parameters = impl_->parameters;
  candidate.provider.blas = impl_->blas;
  candidate.provider.device_workspace = eigensolver_workspace.solver_device_workspace;
  candidate.provider.device_workspace_bytes = eigensolver_workspace.solver_device_workspace_bytes;
  candidate.provider.host_workspace = provider_host_workspace;
  candidate.provider.host_workspace_bytes = own.provider.solver_host_workspace_bytes;
  candidate.provider.requirements = own.provider;
  candidate.provider.capture_mode = impl_->capture_mode;
  candidate.provider.plan_token = own.plan_token;
  candidate.cache = {reinterpret_cast<double*>(setup + own.cache_factor_offset),
                     matrices,
                     reinterpret_cast<std::uint64_t*>(setup + own.cache_generation_offset),
                     batch,
                     reinterpret_cast<std::uint32_t*>(setup + own.cache_status_offset),
                     batch,
                     own.plan_token};
  candidate.workspace = eigensolver_workspace;
  candidate.options = impl_->options;
  candidate.overlap_input = reinterpret_cast<const double*>(setup + own.overlap_input_offset);
  candidate.overlap_elements = matrices;
  candidate.setup_system_errors = reinterpret_cast<std::uint32_t*>(setup + own.system_error_offset);
  candidate.setup_system_error_elements = batch;
  candidate.setup_device_error = reinterpret_cast<std::uint32_t*>(setup + own.device_error_offset);
  candidate.owner_identity = impl_.get();
  candidate.setup_device_arena = setup_device_arena;
  candidate.setup_device_arena_bytes = own.setup_device_bytes;
  candidate.geometry_generation = own.geometry_generation;
  candidate.iteration_layout_fingerprint = iteration_requirements.layout_fingerprint;
  candidate.plan_token = own.plan_token;
  candidate.provenance_seal = binding_provenance_seal(candidate, impl_->binding_salt);

  std::size_t overlap_bytes = 0u;
  if (!checked_multiply(static_cast<std::size_t>(matrices), sizeof(double), overlap_bytes)) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kCountOverflow,
                   SetupField::kOverlap);
  }
  cuda_status = cudaMemcpyAsync(const_cast<double*>(candidate.overlap_input), impl_->pinned_overlap,
                                overlap_bytes, cudaMemcpyHostToDevice, stream);
  if (cuda_status == cudaSuccess) {
    cuda_status =
        cudaMemsetAsync(setup + own.active_offset, 1, static_cast<std::size_t>(batch), stream);
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = reset_gfn2_eigensolver_device_errors_cuda(batch, candidate.setup_system_errors,
                                                            candidate.setup_device_error, stream);
  }
  if (cuda_status != cudaSuccess) {
    return cuda_failure(SetupError::kCudaError, SetupField::kOverlapFactorization, cuda_status);
  }

  Gfn2EigensolverDeviceBatch setup_batch = candidate.batch;
  setup_batch.active = reinterpret_cast<const std::uint8_t*>(setup + own.active_offset);
  const Gfn2EigensolverLaunchResult launch = factor_gfn2_overlap_cuda(
      setup_batch, candidate.provider.buckets, candidate.provider.bucket_count,
      candidate.overlap_input, own.geometry_generation, impl_->options, impl_->solver,
      impl_->parameters, candidate.workspace, candidate.cache, candidate.setup_system_errors,
      candidate.setup_device_error, stream);
  if (!launch.success()) {
    SetupDiagnostic diagnostic =
        failure(XTBLOOM_STATUS_EIGENSOLVER_FAILED, SetupError::kProviderLaunchFailed,
                SetupField::kOverlapFactorization);
    diagnostic.cuda_status = launch.cuda_status;
    diagnostic.cublas_status = launch.cublas_status;
    diagnostic.cusolver_status = launch.cusolver_status;
    return diagnostic;
  }

  binding = candidate;
  return {};
}

Gfn2SccSetupEigensolverDiagnostic Gfn2SccSetupEigensolver::refactor_overlap_impl(
    void* setup_device_arena, std::size_t setup_device_arena_bytes,
    Gfn2SccSetupEigensolverBinding& binding, const double* device_overlap,
    std::int64_t device_overlap_elements, std::uint64_t geometry_generation,
    const Gfn2GeometryEpochDevice* geometry_epoch, const std::uint32_t* request_error,
    cudaStream_t stream) const noexcept {
  if (impl_ == nullptr) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidProvider,
                   SetupField::kHandles);
  }
  const auto& own = impl_->requirements;
  const bool dynamic_epoch = geometry_epoch != nullptr;
  if ((!dynamic_epoch && geometry_generation == 0u) ||
      (dynamic_epoch &&
       (geometry_generation != 0u || geometry_epoch->value == nullptr ||
        geometry_epoch->value_elements != 1 || geometry_epoch->plan_token != own.plan_token ||
        reinterpret_cast<std::uintptr_t>(geometry_epoch->value) % alignof(std::uint64_t) != 0u))) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidGeneration,
                   SetupField::kGeometryGeneration);
  }
  if (setup_device_arena == nullptr) {
    return capacity_failure(SetupError::kNullArena, SetupField::kSetupArena, own.setup_device_bytes,
                            setup_device_arena_bytes);
  }
  if (reinterpret_cast<std::uintptr_t>(setup_device_arena) % own.alignment != 0u) {
    return capacity_failure(SetupError::kMisalignedArena, SetupField::kSetupArena,
                            own.setup_device_bytes, setup_device_arena_bytes);
  }
  if (setup_device_arena_bytes < own.setup_device_bytes) {
    return capacity_failure(SetupError::kInsufficientArena, SetupField::kSetupArena,
                            own.setup_device_bytes, setup_device_arena_bytes);
  }

  cudaPointerAttributes setup_attributes{};
  cudaError_t cuda_status = cudaSuccess;
  if (!cuda_accessible(setup_device_arena, setup_attributes, cuda_status)) {
    SetupDiagnostic diagnostic = failure(XTBLOOM_STATUS_INVALID_ARGUMENT,
                                         SetupError::kInvalidArenaMemory, SetupField::kSetupArena);
    diagnostic.cuda_status = cuda_status;
    return diagnostic;
  }
  if (device_overlap == nullptr || device_overlap_elements != impl_->total_matrices ||
      reinterpret_cast<std::uintptr_t>(device_overlap) % alignof(double) != 0u) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidOverlap,
                   SetupField::kOverlap);
  }
  cudaPointerAttributes overlap_attributes{};
  if (!cuda_accessible(device_overlap, overlap_attributes, cuda_status)) {
    SetupDiagnostic diagnostic = failure(XTBLOOM_STATUS_INVALID_ARGUMENT,
                                         SetupError::kInvalidArenaMemory, SetupField::kOverlap);
    diagnostic.cuda_status = cuda_status;
    return diagnostic;
  }
  cudaPointerAttributes epoch_attributes{};
  if (dynamic_epoch && !cuda_accessible(geometry_epoch->value, epoch_attributes, cuda_status)) {
    SetupDiagnostic diagnostic =
        failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidArenaMemory,
                SetupField::kGeometryGeneration);
    diagnostic.cuda_status = cuda_status;
    return diagnostic;
  }
  cudaPointerAttributes request_error_attributes{};
  if (dynamic_epoch &&
      (request_error == nullptr ||
       reinterpret_cast<std::uintptr_t>(request_error) % alignof(std::uint32_t) != 0u ||
       !cuda_accessible(request_error, request_error_attributes, cuda_status))) {
    SetupDiagnostic diagnostic = failure(XTBLOOM_STATUS_INVALID_ARGUMENT,
                                         SetupError::kInvalidArenaMemory, SetupField::kAdmission);
    diagnostic.cuda_status = cuda_status;
    return diagnostic;
  }
  int current_device = -1;
  cuda_status = cudaGetDevice(&current_device);
  if (cuda_status != cudaSuccess) {
    return cuda_failure(SetupError::kCudaError, SetupField::kOverlap, cuda_status);
  }
  if (setup_attributes.device != current_device || overlap_attributes.device != current_device ||
      (dynamic_epoch && epoch_attributes.device != current_device)) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidArenaMemory,
                   SetupField::kOverlap);
  }
  if (dynamic_epoch && request_error_attributes.device != current_device) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidArenaMemory,
                   SetupField::kAdmission);
  }

  auto* const setup = static_cast<std::byte*>(setup_device_arena);
  const double* const expected_overlap =
      reinterpret_cast<const double*>(setup + own.overlap_input_offset);
  const Gfn2EigensolverOverlapCache expected_cache{
      reinterpret_cast<double*>(setup + own.cache_factor_offset),
      impl_->total_matrices,
      reinterpret_cast<std::uint64_t*>(setup + own.cache_generation_offset),
      impl_->batch_size,
      reinterpret_cast<std::uint32_t*>(setup + own.cache_status_offset),
      impl_->batch_size,
      own.plan_token,
  };
  std::uint32_t* const expected_system_errors =
      reinterpret_cast<std::uint32_t*>(setup + own.system_error_offset);
  std::uint32_t* const expected_device_error =
      reinterpret_cast<std::uint32_t*>(setup + own.device_error_offset);

  const auto same_provider_requirements =
      [](const Gfn2EigensolverWorkspaceRequirements& first,
         const Gfn2EigensolverWorkspaceRequirements& second) noexcept {
        return first.solver_device_workspace_bytes == second.solver_device_workspace_bytes &&
               first.solver_host_workspace_bytes == second.solver_host_workspace_bytes;
      };
  const bool valid_binding =
      binding.owner_identity == impl_.get() && binding.setup_device_arena == setup_device_arena &&
      binding.setup_device_arena_bytes == own.setup_device_bytes &&
      binding.plan_token == own.plan_token && binding.iteration_layout_fingerprint != 0u &&
      binding.batch.batch_size == impl_->batch_size &&
      binding.batch.total_orbitals == impl_->total_orbitals &&
      binding.batch.total_matrix_elements == impl_->total_matrices &&
      binding.batch.orbital_offset_count == impl_->batch_size + 1 &&
      binding.batch.matrix_offset_count == impl_->batch_size + 1 &&
      binding.batch.bucket_system_count == impl_->batch_size &&
      binding.batch.active_elements == impl_->batch_size &&
      binding.batch.plan_token == own.plan_token && binding.batch.orbital_offsets != nullptr &&
      binding.batch.matrix_offsets != nullptr && binding.batch.bucket_systems != nullptr &&
      binding.batch.active != nullptr && binding.provider.buckets == impl_->buckets.data() &&
      binding.provider.bucket_count == static_cast<std::int64_t>(impl_->buckets.size()) &&
      binding.provider.solver == impl_->solver &&
      binding.provider.parameters == impl_->parameters && binding.provider.blas == impl_->blas &&
      binding.provider.device_workspace == binding.workspace.solver_device_workspace &&
      binding.provider.device_workspace_bytes == binding.workspace.solver_device_workspace_bytes &&
      binding.provider.host_workspace == binding.workspace.solver_host_workspace &&
      binding.provider.host_workspace_bytes == binding.workspace.solver_host_workspace_bytes &&
      same_provider_requirements(binding.provider.requirements, own.provider) &&
      binding.provider.plan_token == own.plan_token &&
      binding.cache.cholesky_factors == expected_cache.cholesky_factors &&
      binding.cache.factor_elements == expected_cache.factor_elements &&
      binding.cache.geometry_generations == expected_cache.geometry_generations &&
      binding.cache.generation_elements == expected_cache.generation_elements &&
      binding.cache.factor_statuses == expected_cache.factor_statuses &&
      binding.cache.status_elements == expected_cache.status_elements &&
      binding.cache.plan_token == own.plan_token &&
      binding.workspace.plan_token == own.plan_token &&
      binding.workspace.matrix_a_elements == impl_->total_spin_matrices &&
      binding.workspace.matrix_b_elements == impl_->total_spin_matrices &&
      binding.workspace.eigenvalue_elements == impl_->total_spin_orbitals &&
      binding.workspace.factor_pointer_elements == impl_->total_spin_channels &&
      binding.workspace.matrix_pointer_elements == impl_->total_spin_channels &&
      binding.workspace.info_a_elements == impl_->total_spin_channels &&
      binding.workspace.info_b_elements == impl_->total_spin_channels &&
      binding.workspace.eligible_elements == impl_->total_spin_channels &&
      binding.workspace.sequence_active_elements == 1 &&
      binding.workspace.compact_system_elements == impl_->total_spin_channels &&
      binding.workspace.compact_source_slot_elements == impl_->total_spin_channels &&
      binding.workspace.bucket_activity_elements ==
          static_cast<std::int64_t>(impl_->buckets.size()) &&
      same_provider_requirements({binding.workspace.solver_device_workspace_bytes,
                                  binding.workspace.solver_host_workspace_bytes},
                                 own.provider) &&
      binding.options.minimum_overlap_rcond == impl_->options.minimum_overlap_rcond &&
      binding.options.symmetry_tolerance == impl_->options.symmetry_tolerance &&
      binding.options.deterministic_debug == impl_->options.deterministic_debug &&
      binding.options.strategy == impl_->options.strategy &&
      binding.options.jacobi == impl_->options.jacobi &&
      binding.overlap_input == expected_overlap &&
      binding.overlap_elements == impl_->total_matrices &&
      binding.setup_system_errors == expected_system_errors &&
      binding.setup_system_error_elements == impl_->batch_size &&
      binding.setup_device_error == expected_device_error &&
      ((binding.geometry_epoch == nullptr && binding.geometry_epoch_elements == 0) ||
       (binding.geometry_epoch != nullptr && binding.geometry_epoch_elements == 1)) &&
      binding.provenance_seal != 0u &&
      binding.provenance_seal == binding_provenance_seal(binding, impl_->binding_salt);
  if (!valid_binding) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidIterationProvenance,
                   SetupField::kOverlapFactorization);
  }
  if ((!dynamic_epoch && binding.geometry_epoch != nullptr) ||
      (dynamic_epoch && binding.geometry_epoch != nullptr &&
       (binding.geometry_epoch != geometry_epoch->value || binding.geometry_epoch_elements != 1)) ||
      (!dynamic_epoch && geometry_generation <= binding.geometry_generation)) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidGeneration,
                   SetupField::kGeometryGeneration);
  }

  AddressRange setup_range{};
  AddressRange input_range{};
  AddressRange expected_input_range{};
  AddressRange epoch_range{};
  AddressRange request_error_range{};
  if (!make_range(setup_device_arena, own.setup_device_bytes, setup_range) ||
      !make_elements_range(device_overlap, device_overlap_elements, input_range) ||
      !make_elements_range(expected_overlap, impl_->total_matrices, expected_input_range) ||
      !make_elements_range(dynamic_epoch ? geometry_epoch->value : nullptr, dynamic_epoch ? 1 : 0,
                           epoch_range) ||
      !make_elements_range(dynamic_epoch ? request_error : nullptr, dynamic_epoch ? 1 : 0,
                           request_error_range) ||
      (overlaps(setup_range, input_range) && (input_range.begin != expected_input_range.begin ||
                                              input_range.end != expected_input_range.end)) ||
      overlaps(epoch_range, setup_range) || overlaps(epoch_range, input_range) ||
      overlaps(request_error_range, setup_range) || overlaps(request_error_range, input_range) ||
      overlaps(request_error_range, epoch_range)) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidOverlap,
                   overlaps(request_error_range, setup_range) ||
                           overlaps(request_error_range, input_range) ||
                           overlaps(request_error_range, epoch_range)
                       ? SetupField::kAdmission
                       : SetupField::kOverlap);
  }

  std::array<AddressRange, 22> protected_ranges{};
  if (!make_elements_range(binding.batch.orbital_offsets, binding.batch.orbital_offset_count,
                           protected_ranges[0]) ||
      !make_elements_range(binding.batch.matrix_offsets, binding.batch.matrix_offset_count,
                           protected_ranges[1]) ||
      !make_elements_range(binding.batch.bucket_systems, binding.batch.bucket_system_count,
                           protected_ranges[2]) ||
      !make_elements_range(binding.batch.active, binding.batch.active_elements,
                           protected_ranges[3]) ||
      !make_elements_range(binding.cache.cholesky_factors, binding.cache.factor_elements,
                           protected_ranges[4]) ||
      !make_elements_range(binding.cache.geometry_generations, binding.cache.generation_elements,
                           protected_ranges[5]) ||
      !make_elements_range(binding.cache.factor_statuses, binding.cache.status_elements,
                           protected_ranges[6]) ||
      !make_elements_range(binding.workspace.matrix_scratch_a, binding.workspace.matrix_a_elements,
                           protected_ranges[7]) ||
      !make_elements_range(binding.workspace.matrix_scratch_b, binding.workspace.matrix_b_elements,
                           protected_ranges[8]) ||
      !make_elements_range(binding.workspace.eigenvalue_scratch,
                           binding.workspace.eigenvalue_elements, protected_ranges[9]) ||
      !make_elements_range(binding.workspace.factor_pointers,
                           binding.workspace.factor_pointer_elements, protected_ranges[10]) ||
      !make_elements_range(binding.workspace.matrix_pointers,
                           binding.workspace.matrix_pointer_elements, protected_ranges[11]) ||
      !make_elements_range(binding.workspace.info_a, binding.workspace.info_a_elements,
                           protected_ranges[12]) ||
      !make_elements_range(binding.workspace.info_b, binding.workspace.info_b_elements,
                           protected_ranges[13]) ||
      !make_elements_range(binding.workspace.eligible, binding.workspace.eligible_elements,
                           protected_ranges[14]) ||
      !make_elements_range(binding.workspace.sequence_active,
                           binding.workspace.sequence_active_elements, protected_ranges[15]) ||
      !make_range(binding.workspace.solver_device_workspace,
                  binding.workspace.solver_device_workspace_bytes, protected_ranges[16]) ||
      !make_elements_range(binding.setup_system_errors, binding.setup_system_error_elements,
                           protected_ranges[17]) ||
      !make_elements_range(binding.setup_device_error, 1, protected_ranges[18]) ||
      !make_elements_range(binding.workspace.compact_systems,
                           binding.workspace.compact_system_elements, protected_ranges[19]) ||
      !make_elements_range(binding.workspace.compact_source_slots,
                           binding.workspace.compact_source_slot_elements, protected_ranges[20]) ||
      !make_elements_range(binding.workspace.bucket_activity,
                           binding.workspace.bucket_activity_elements, protected_ranges[21]) ||
      !pairwise_disjoint(protected_ranges)) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidIterationProvenance,
                   SetupField::kOverlapFactorization);
  }
  for (const AddressRange& range : protected_ranges) {
    if (overlaps(input_range, range) || overlaps(epoch_range, range) ||
        overlaps(request_error_range, range)) {
      return failure(
          XTBLOOM_STATUS_INVALID_ARGUMENT, SetupError::kInvalidOverlap,
          overlaps(request_error_range, range) ? SetupField::kAdmission : SetupField::kOverlap);
    }
  }

  auto* const setup_active = reinterpret_cast<std::uint8_t*>(setup + own.active_offset);
  if (dynamic_epoch) {
    /* Snapshot the runtime's canonical eligibility ledger before any solver
     * kernel. This keeps peer activity coherent for the complete factorization
     * even if a downstream SCC stage later mutates its public ledger. */
    constexpr int kActivityThreads = 256;
    const auto activity_blocks =
        static_cast<unsigned int>((impl_->batch_size + kActivityThreads - 1) / kActivityThreads);
    snapshot_refactor_activity_kernel<<<activity_blocks, kActivityThreads, 0, stream>>>(
        impl_->batch_size, binding.batch.active, request_error, setup_active);
    cuda_status = cudaPeekAtLastError();
  } else {
    cuda_status =
        cudaMemsetAsync(setup_active, 1, static_cast<std::size_t>(impl_->batch_size), stream);
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = reset_gfn2_eigensolver_device_errors_cuda(
        impl_->batch_size, binding.setup_system_errors, binding.setup_device_error, stream);
  }
  if (cuda_status != cudaSuccess) {
    return cuda_failure(SetupError::kCudaError, SetupField::kOverlapFactorization, cuda_status);
  }

  Gfn2EigensolverDeviceBatch refactor_batch = binding.batch;
  /* The private snapshot is all-active for legacy scalar refreshes and a copy
   * of the preprocessing/core eligibility ledger for dynamic execution. */
  refactor_batch.active = setup_active;
  const Gfn2EigensolverLaunchResult launch =
      dynamic_epoch ? factor_gfn2_overlap_cuda(
                          refactor_batch, binding.provider.buckets, binding.provider.bucket_count,
                          device_overlap, *geometry_epoch, binding.options, binding.provider.solver,
                          binding.provider.parameters, binding.workspace, binding.cache,
                          binding.setup_system_errors, binding.setup_device_error, stream,
                          Gfn2EigensolverFactorCachePolicy::kPreservePriorOnFailure)
                    : factor_gfn2_overlap_cuda(
                          refactor_batch, binding.provider.buckets, binding.provider.bucket_count,
                          device_overlap, geometry_generation, binding.options,
                          binding.provider.solver, binding.provider.parameters, binding.workspace,
                          binding.cache, binding.setup_system_errors, binding.setup_device_error,
                          stream, Gfn2EigensolverFactorCachePolicy::kPreservePriorOnFailure);
  if (!launch.success()) {
    SetupDiagnostic diagnostic =
        failure(XTBLOOM_STATUS_EIGENSOLVER_FAILED, SetupError::kProviderLaunchFailed,
                SetupField::kOverlapFactorization);
    diagnostic.cuda_status = launch.cuda_status;
    diagnostic.cublas_status = launch.cublas_status;
    diagnostic.cusolver_status = launch.cusolver_status;
    return diagnostic;
  }

  if (dynamic_epoch) {
    binding.geometry_epoch = geometry_epoch->value;
    binding.geometry_epoch_elements = geometry_epoch->value_elements;
  } else {
    binding.geometry_generation = geometry_generation;
  }
  binding.provenance_seal = binding_provenance_seal(binding, impl_->binding_salt);
  return {};
}

Gfn2SccSetupEigensolverDiagnostic Gfn2SccSetupEigensolver::refactor_overlap_from_device_async(
    void* setup_device_arena, std::size_t setup_device_arena_bytes,
    Gfn2SccSetupEigensolverBinding& binding, const double* device_overlap,
    std::int64_t device_overlap_elements, std::uint64_t geometry_generation,
    cudaStream_t stream) const noexcept {
  return refactor_overlap_impl(setup_device_arena, setup_device_arena_bytes, binding,
                               device_overlap, device_overlap_elements, geometry_generation,
                               nullptr, nullptr, stream);
}

Gfn2SccSetupEigensolverDiagnostic Gfn2SccSetupEigensolver::refactor_overlap_from_device_epoch_async(
    void* setup_device_arena, std::size_t setup_device_arena_bytes,
    Gfn2SccSetupEigensolverBinding& binding, const double* device_overlap,
    std::int64_t device_overlap_elements, const Gfn2GeometryEpochDevice& geometry_epoch,
    const std::uint32_t* request_error, cudaStream_t stream) const noexcept {
  return refactor_overlap_impl(setup_device_arena, setup_device_arena_bytes, binding,
                               device_overlap, device_overlap_elements, 0u, &geometry_epoch,
                               request_error, stream);
}

}  // namespace xtbloom::detail::cuda
