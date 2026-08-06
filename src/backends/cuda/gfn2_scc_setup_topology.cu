#include <algorithm>
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_scc_setup_topology.hpp"

namespace gpuxtb::detail::cuda {
namespace {

using gfn2::BasisPlan;
using gfn2::IntegralPlan;
using gfn2::WavefunctionFieldLayout;
using gfn2::WavefunctionLayout;

Gfn2SccSetupTopologyDiagnostic failure(gpuxtb_status_t status, Gfn2SccSetupTopologyError error,
                                       Gfn2SccSetupTopologyField field,
                                       std::int64_t index = -1) noexcept {
  Gfn2SccSetupTopologyDiagnostic result{};
  result.status = status;
  result.error = error;
  result.field = field;
  result.index = index;
  return result;
}

Gfn2SccSetupTopologyDiagnostic arena_failure(Gfn2SccSetupTopologyError error,
                                             std::size_t required_bytes,
                                             cudaError_t cuda_status = cudaSuccess) noexcept {
  Gfn2SccSetupTopologyDiagnostic result =
      failure(error == Gfn2SccSetupTopologyError::kCudaError ? GPUXTB_STATUS_INTERNAL_ERROR
                                                             : GPUXTB_STATUS_INVALID_ARGUMENT,
              error, Gfn2SccSetupTopologyField::kArena);
  result.required_bytes = required_bytes;
  result.cuda_status = cuda_status;
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

bool checked_multiply(std::int64_t first, std::int64_t second, std::int64_t& result) noexcept {
  if (first < 0 || second < 0 ||
      (first != 0 && second > std::numeric_limits<std::int64_t>::max() / first)) {
    return false;
  }
  result = first * second;
  return true;
}

bool aligned_offset(std::size_t offset, std::size_t alignment, std::size_t& result) noexcept {
  const std::size_t remainder = offset % alignment;
  return remainder == 0u || checked_add(offset, alignment - remainder, result);
}

template <typename T>
bool exact_extent(const std::vector<T>& values, std::int64_t expected) noexcept {
  return expected >= 0 &&
         static_cast<std::uint64_t>(expected) <=
             static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) &&
         values.size() == static_cast<std::size_t>(expected);
}

bool valid_offsets(const std::vector<std::int64_t>& offsets, std::int64_t partitions,
                   std::int64_t endpoint) noexcept {
  if (partitions < 0 || partitions == std::numeric_limits<std::int64_t>::max() || endpoint < 0 ||
      !exact_extent(offsets, partitions + 1) || offsets.front() != 0 ||
      offsets.back() != endpoint) {
    return false;
  }
  for (std::int64_t index = 0; index < partitions; ++index) {
    const std::int64_t begin = offsets[static_cast<std::size_t>(index)];
    const std::int64_t end = offsets[static_cast<std::size_t>(index + 1)];
    if (begin < 0 || begin > end || end > endpoint) {
      return false;
    }
  }
  return true;
}

/*
 * Gfn2SccSetupTopology is a public setup boundary and therefore cannot assume
 * that a WavefunctionLayout came directly from make_wavefunction_layout().
 * Prove every field partition before create() copies offsets or calls back().
 */
bool valid_wavefunction_field_offsets(const BasisPlan& basis, const IntegralPlan& integrals,
                                      const WavefunctionLayout& wavefunction) noexcept {
  const WavefunctionFieldLayout* const fields[] = {
      &wavefunction.coefficients, &wavefunction.eigenvalues, &wavefunction.occupations,
      &wavefunction.density,      &wavefunction.qsh,         &wavefunction.qat,
      &wavefunction.dipole,       &wavefunction.quadrupole,  &wavefunction.energy_weighted_density,
  };
  for (const WavefunctionFieldLayout* field : fields) {
    if (!valid_offsets(field->system_offsets, basis.batch_size, field->element_count)) {
      return false;
    }
  }

  for (std::int64_t system = 0; system < basis.batch_size; ++system) {
    const std::size_t index = static_cast<std::size_t>(system);
    const std::int32_t spin_channels = wavefunction.spin_channels[index];
    const std::int64_t atoms = basis.atom_offsets[index + 1u] - basis.atom_offsets[index];
    const std::int64_t shells =
        basis.batch_shell_offsets[index + 1u] - basis.batch_shell_offsets[index];
    const std::int64_t orbitals =
        basis.batch_orbital_offsets[index + 1u] - basis.batch_orbital_offsets[index];
    const std::int64_t matrices =
        integrals.matrix_offsets[index + 1u] - integrals.matrix_offsets[index];
    std::int64_t spin_matrices = 0;
    std::int64_t spin_orbitals = 0;
    std::int64_t occupations = 0;
    std::int64_t spin_shells = 0;
    std::int64_t spin_atoms = 0;
    std::int64_t dipoles = 0;
    std::int64_t quadrupoles = 0;
    if ((spin_channels != 1 && spin_channels != 2) ||
        !checked_multiply(matrices, spin_channels, spin_matrices) ||
        !checked_multiply(orbitals, spin_channels, spin_orbitals) ||
        !checked_multiply(orbitals, 2, occupations) ||
        !checked_multiply(shells, spin_channels, spin_shells) ||
        !checked_multiply(atoms, spin_channels, spin_atoms) ||
        !checked_multiply(spin_atoms, 3, dipoles) ||
        !checked_multiply(spin_atoms, gfn2::kWavefunctionQuadrupoleComponents, quadrupoles)) {
      return false;
    }
    const std::int64_t expected[] = {spin_matrices, spin_orbitals, occupations,
                                     spin_matrices, spin_shells,   spin_atoms,
                                     dipoles,       quadrupoles,   spin_matrices};
    for (std::size_t field_index = 0; field_index < std::size(fields); ++field_index) {
      const auto& offsets = fields[field_index]->system_offsets;
      if (offsets[index + 1u] - offsets[index] != expected[field_index]) {
        return false;
      }
    }
  }
  return true;
}

Gfn2SccSetupTopologyDiagnostic validate_plan_compatibility(const BasisPlan& basis,
                                                           const IntegralPlan& integrals,
                                                           const WavefunctionLayout& wavefunction,
                                                           std::uint64_t plan_token) noexcept {
  if (plan_token == 0u) {
    return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Gfn2SccSetupTopologyError::kInvalidPlan,
                   Gfn2SccSetupTopologyField::kPlanToken);
  }
  if (basis.batch_size <= 0 || basis.batch_size > INT32_MAX || basis.total_atoms <= 0 ||
      basis.total_shells <= 0 || basis.total_orbitals <= 0 ||
      !valid_offsets(basis.atom_offsets, basis.batch_size, basis.total_atoms) ||
      !valid_offsets(basis.batch_shell_offsets, basis.batch_size, basis.total_shells) ||
      !valid_offsets(basis.batch_orbital_offsets, basis.batch_size, basis.total_orbitals) ||
      !valid_offsets(basis.atom_shell_offsets, basis.total_atoms, basis.total_shells) ||
      !valid_offsets(basis.shell_orbital_offsets, basis.total_shells, basis.total_orbitals) ||
      !exact_extent(basis.shell_to_atom, basis.total_shells)) {
    return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Gfn2SccSetupTopologyError::kInvalidPlan,
                   Gfn2SccSetupTopologyField::kBasis);
  }
  if (integrals.batch_size != basis.batch_size || integrals.total_matrix_elements <= 0 ||
      !valid_offsets(integrals.matrix_offsets, basis.batch_size, integrals.total_matrix_elements)) {
    return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Gfn2SccSetupTopologyError::kInvalidPlan,
                   Gfn2SccSetupTopologyField::kIntegrals);
  }
  if (wavefunction.batch_size != basis.batch_size ||
      wavefunction.total_atoms != basis.total_atoms ||
      wavefunction.total_shells != basis.total_shells ||
      wavefunction.total_orbitals != basis.total_orbitals ||
      wavefunction.atom_offsets != basis.atom_offsets ||
      wavefunction.batch_shell_offsets != basis.batch_shell_offsets ||
      wavefunction.batch_orbital_offsets != basis.batch_orbital_offsets ||
      !exact_extent(wavefunction.spin_channels, basis.batch_size)) {
    return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Gfn2SccSetupTopologyError::kInvalidPlan,
                   Gfn2SccSetupTopologyField::kWavefunction);
  }
  for (std::int64_t system = 0; system < basis.batch_size; ++system) {
    const std::int32_t channels = wavefunction.spin_channels[static_cast<std::size_t>(system)];
    if (channels != 1 && channels != 2) {
      return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Gfn2SccSetupTopologyError::kInvalidPlan,
                     Gfn2SccSetupTopologyField::kWavefunction, system);
    }
  }
  if (!valid_wavefunction_field_offsets(basis, integrals, wavefunction)) {
    return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Gfn2SccSetupTopologyError::kInvalidPlan,
                   Gfn2SccSetupTopologyField::kWavefunction);
  }
  return {};
}

Gfn2SccSetupTopologyDiagnostic build_buckets(const BasisPlan& basis, const IntegralPlan& integrals,
                                             std::vector<std::int64_t>& bucket_offsets,
                                             std::vector<std::int32_t>& bucket_systems,
                                             std::vector<std::int32_t>& bucket_orbital_counts,
                                             std::vector<Gfn2EigensolverBucket>& buckets) {
  std::vector<std::int32_t> dimensions;
  dimensions.reserve(static_cast<std::size_t>(basis.batch_size));
  for (std::int64_t system = 0; system < basis.batch_size; ++system) {
    const std::int64_t orbitals =
        basis.batch_orbital_offsets[static_cast<std::size_t>(system + 1)] -
        basis.batch_orbital_offsets[static_cast<std::size_t>(system)];
    std::int64_t matrix_elements = 0;
    if (orbitals <= 0 || orbitals > INT32_MAX ||
        !checked_multiply(orbitals, orbitals, matrix_elements) || matrix_elements > INT_MAX ||
        integrals.matrix_offsets[static_cast<std::size_t>(system + 1)] -
                integrals.matrix_offsets[static_cast<std::size_t>(system)] !=
            matrix_elements) {
      return failure(GPUXTB_STATUS_INVALID_ARGUMENT,
                     orbitals > INT32_MAX || matrix_elements > INT_MAX
                         ? Gfn2SccSetupTopologyError::kCountOverflow
                         : Gfn2SccSetupTopologyError::kInvalidPlan,
                     Gfn2SccSetupTopologyField::kBuckets, system);
    }
    dimensions.push_back(static_cast<std::int32_t>(orbitals));
  }

  std::vector<std::int32_t> unique_dimensions = dimensions;
  std::sort(unique_dimensions.begin(), unique_dimensions.end());
  unique_dimensions.erase(std::unique(unique_dimensions.begin(), unique_dimensions.end()),
                          unique_dimensions.end());

  bucket_offsets.reserve(unique_dimensions.size() + 1u);
  bucket_orbital_counts.reserve(unique_dimensions.size());
  buckets.reserve(unique_dimensions.size());
  bucket_systems.reserve(static_cast<std::size_t>(basis.batch_size));
  bucket_offsets.push_back(0);

  std::int64_t packed_matrix_offset = 0;
  std::int64_t packed_orbital_offset = 0;
  for (const std::int32_t dimension : unique_dimensions) {
    const std::int64_t system_index_offset = static_cast<std::int64_t>(bucket_systems.size());
    for (std::int64_t system = 0; system < basis.batch_size; ++system) {
      if (dimensions[static_cast<std::size_t>(system)] == dimension) {
        bucket_systems.push_back(static_cast<std::int32_t>(system));
      }
    }
    const std::int64_t system_count =
        static_cast<std::int64_t>(bucket_systems.size()) - system_index_offset;
    std::int64_t matrix_stride = 0;
    std::int64_t matrix_span = 0;
    std::int64_t orbital_span = 0;
    std::int64_t next_matrix = 0;
    std::int64_t next_orbital = 0;
    if (system_count <= 0 || system_count > INT32_MAX ||
        !checked_multiply(static_cast<std::int64_t>(dimension),
                          static_cast<std::int64_t>(dimension), matrix_stride) ||
        !checked_multiply(matrix_stride, system_count, matrix_span) || matrix_span > INT_MAX ||
        !checked_multiply(static_cast<std::int64_t>(dimension), system_count, orbital_span) ||
        orbital_span > INT_MAX ||
        packed_matrix_offset > std::numeric_limits<std::int64_t>::max() - matrix_span ||
        packed_orbital_offset > std::numeric_limits<std::int64_t>::max() - orbital_span) {
      return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Gfn2SccSetupTopologyError::kCountOverflow,
                     Gfn2SccSetupTopologyField::kBuckets,
                     static_cast<std::int64_t>(buckets.size()));
    }
    next_matrix = packed_matrix_offset + matrix_span;
    next_orbital = packed_orbital_offset + orbital_span;
    buckets.push_back({dimension, static_cast<std::int32_t>(system_count), system_index_offset,
                       packed_matrix_offset, packed_orbital_offset});
    bucket_orbital_counts.push_back(dimension);
    bucket_offsets.push_back(static_cast<std::int64_t>(bucket_systems.size()));
    packed_matrix_offset = next_matrix;
    packed_orbital_offset = next_orbital;
  }
  if (packed_matrix_offset != integrals.total_matrix_elements ||
      packed_orbital_offset != basis.total_orbitals) {
    return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Gfn2SccSetupTopologyError::kInvalidPlan,
                   Gfn2SccSetupTopologyField::kBuckets);
  }
  return {};
}

/*
 * Append the canonical spin solve projection without changing the physical
 * bucket permutation used by overlap factorization.  Each bucket walks its
 * systems in the existing deterministic order and expands only unrestricted
 * members to alpha then beta work items.
 */
Gfn2SccSetupTopologyDiagnostic configure_spin_buckets(
    const std::vector<std::int32_t>& spin_channels, const std::vector<std::int32_t>& bucket_systems,
    std::vector<Gfn2EigensolverBucket>& buckets, std::int64_t expected_spin_channels,
    std::int64_t expected_spin_orbitals, std::int64_t expected_spin_matrix_elements) noexcept {
  std::int64_t solve_offset = 0;
  std::int64_t orbital_offset = 0;
  std::int64_t matrix_offset = 0;
  for (std::size_t bucket_index = 0; bucket_index < buckets.size(); ++bucket_index) {
    Gfn2EigensolverBucket& bucket = buckets[bucket_index];
    std::int64_t solve_count = 0;
    const std::int64_t system_end =
        bucket.system_index_offset + static_cast<std::int64_t>(bucket.system_count);
    if (bucket.system_index_offset < 0 || system_end < bucket.system_index_offset ||
        static_cast<std::uint64_t>(system_end) > bucket_systems.size()) {
      return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Gfn2SccSetupTopologyError::kInvalidPlan,
                     Gfn2SccSetupTopologyField::kBuckets, static_cast<std::int64_t>(bucket_index));
    }
    for (std::int64_t position = bucket.system_index_offset; position < system_end; ++position) {
      const std::int32_t system = bucket_systems[static_cast<std::size_t>(position)];
      if (system < 0 || static_cast<std::uint64_t>(system) >= spin_channels.size()) {
        return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Gfn2SccSetupTopologyError::kInvalidPlan,
                       Gfn2SccSetupTopologyField::kBuckets, position);
      }
      const std::int32_t channels = spin_channels[static_cast<std::size_t>(system)];
      if ((channels != 1 && channels != 2) ||
          solve_count > std::numeric_limits<std::int64_t>::max() - channels) {
        return failure(GPUXTB_STATUS_INVALID_ARGUMENT,
                       channels == 1 || channels == 2 ? Gfn2SccSetupTopologyError::kCountOverflow
                                                      : Gfn2SccSetupTopologyError::kInvalidPlan,
                       Gfn2SccSetupTopologyField::kWavefunction, system);
      }
      solve_count += channels;
    }
    std::int64_t matrix_stride = 0;
    std::int64_t orbital_span = 0;
    std::int64_t matrix_span = 0;
    if (solve_count <= 0 || solve_count > INT32_MAX ||
        !checked_multiply(static_cast<std::int64_t>(bucket.orbital_count),
                          static_cast<std::int64_t>(bucket.orbital_count), matrix_stride) ||
        !checked_multiply(static_cast<std::int64_t>(bucket.orbital_count), solve_count,
                          orbital_span) ||
        !checked_multiply(matrix_stride, solve_count, matrix_span) ||
        solve_offset > std::numeric_limits<std::int64_t>::max() - solve_count ||
        orbital_offset > std::numeric_limits<std::int64_t>::max() - orbital_span ||
        matrix_offset > std::numeric_limits<std::int64_t>::max() - matrix_span) {
      return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Gfn2SccSetupTopologyError::kCountOverflow,
                     Gfn2SccSetupTopologyField::kBuckets, static_cast<std::int64_t>(bucket_index));
    }
    bucket.solve_count = static_cast<std::int32_t>(solve_count);
    bucket.solve_index_offset = solve_offset;
    bucket.spin_orbital_scratch_offset = orbital_offset;
    bucket.spin_matrix_scratch_offset = matrix_offset;
    solve_offset += solve_count;
    orbital_offset += orbital_span;
    matrix_offset += matrix_span;
  }
  if (solve_offset != expected_spin_channels || orbital_offset != expected_spin_orbitals ||
      matrix_offset != expected_spin_matrix_elements) {
    return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Gfn2SccSetupTopologyError::kInvalidPlan,
                   Gfn2SccSetupTopologyField::kBuckets);
  }
  return {};
}

template <typename T>
bool append_array_layout(std::size_t elements, std::size_t& cursor, std::size_t& offset) noexcept {
  std::size_t aligned = cursor;
  if (!aligned_offset(cursor, alignof(T), aligned)) {
    return false;
  }
  std::size_t bytes = 0u;
  std::size_t end = 0u;
  if (!checked_multiply(elements, sizeof(T), bytes) || !checked_add(aligned, bytes, end)) {
    return false;
  }
  offset = aligned;
  cursor = end;
  return true;
}

const Gfn2RaggedTopologyView& empty_topology() noexcept {
  static const Gfn2RaggedTopologyView empty{};
  return empty;
}

const Gfn2WavefunctionLayoutView& empty_wavefunction_layout() noexcept {
  static const Gfn2WavefunctionLayoutView empty{};
  return empty;
}

const std::vector<Gfn2EigensolverBucket>& empty_buckets() noexcept {
  static const std::vector<Gfn2EigensolverBucket> empty;
  return empty;
}

}  // namespace

struct Gfn2SccSetupTopology::Impl {
  struct ArenaLayout {
    std::size_t atom_offsets = 0u;
    std::size_t batch_shell_offsets = 0u;
    std::size_t batch_orbital_offsets = 0u;
    std::size_t matrix_offsets = 0u;
    std::size_t atom_shell_offsets = 0u;
    std::size_t shell_orbital_offsets = 0u;
    std::size_t shell_to_atom = 0u;
    std::size_t orbital_to_shell = 0u;
    std::size_t orbital_to_atom = 0u;
    std::size_t bucket_offsets = 0u;
    std::size_t bucket_systems = 0u;
    std::size_t bucket_orbital_counts = 0u;
    std::size_t spin_channels = 0u;
    std::size_t spin_channel_offsets = 0u;
    std::size_t spin_orbital_offsets = 0u;
    std::size_t spin_matrix_offsets = 0u;
    std::size_t spin_shell_offsets = 0u;
    std::size_t spin_atom_offsets = 0u;
    std::size_t total_bytes = 0u;
  } arena;

  std::uint64_t plan_token = 0u;
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t total_matrix_elements = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> batch_shell_offsets;
  std::vector<std::int64_t> batch_orbital_offsets;
  std::vector<std::int64_t> matrix_offsets;
  std::vector<std::int64_t> atom_shell_offsets;
  std::vector<std::int64_t> shell_orbital_offsets;
  std::vector<std::int64_t> shell_to_atom;
  std::vector<std::int64_t> orbital_to_shell;
  std::vector<std::int64_t> orbital_to_atom;
  std::vector<std::int64_t> bucket_offsets;
  std::vector<std::int32_t> bucket_systems;
  std::vector<std::int32_t> bucket_orbital_counts;
  std::vector<std::int32_t> spin_channels;
  std::vector<std::int64_t> spin_channel_offsets;
  std::vector<std::int64_t> spin_orbital_offsets;
  std::vector<std::int64_t> spin_matrix_offsets;
  std::vector<std::int64_t> spin_shell_offsets;
  std::vector<std::int64_t> spin_atom_offsets;
  std::vector<Gfn2EigensolverBucket> buckets;
  Gfn2RaggedTopologyView host{};
  Gfn2WavefunctionLayoutView host_wavefunction{};
  void* upload_image = nullptr;

  ~Impl() {
    if (upload_image != nullptr) {
      (void)cudaFreeHost(upload_image);
    }
  }

  bool make_arena_layout() noexcept {
    std::size_t cursor = 0u;
    return append_array_layout<std::int64_t>(atom_offsets.size(), cursor, arena.atom_offsets) &&
           append_array_layout<std::int64_t>(batch_shell_offsets.size(), cursor,
                                             arena.batch_shell_offsets) &&
           append_array_layout<std::int64_t>(batch_orbital_offsets.size(), cursor,
                                             arena.batch_orbital_offsets) &&
           append_array_layout<std::int64_t>(matrix_offsets.size(), cursor, arena.matrix_offsets) &&
           append_array_layout<std::int64_t>(atom_shell_offsets.size(), cursor,
                                             arena.atom_shell_offsets) &&
           append_array_layout<std::int64_t>(shell_orbital_offsets.size(), cursor,
                                             arena.shell_orbital_offsets) &&
           append_array_layout<std::int64_t>(shell_to_atom.size(), cursor, arena.shell_to_atom) &&
           append_array_layout<std::int64_t>(orbital_to_shell.size(), cursor,
                                             arena.orbital_to_shell) &&
           append_array_layout<std::int64_t>(orbital_to_atom.size(), cursor,
                                             arena.orbital_to_atom) &&
           append_array_layout<std::int64_t>(bucket_offsets.size(), cursor, arena.bucket_offsets) &&
           append_array_layout<std::int32_t>(bucket_systems.size(), cursor, arena.bucket_systems) &&
           append_array_layout<std::int32_t>(bucket_orbital_counts.size(), cursor,
                                             arena.bucket_orbital_counts) &&
           append_array_layout<std::int32_t>(spin_channels.size(), cursor, arena.spin_channels) &&
           append_array_layout<std::int64_t>(spin_channel_offsets.size(), cursor,
                                             arena.spin_channel_offsets) &&
           append_array_layout<std::int64_t>(spin_orbital_offsets.size(), cursor,
                                             arena.spin_orbital_offsets) &&
           append_array_layout<std::int64_t>(spin_matrix_offsets.size(), cursor,
                                             arena.spin_matrix_offsets) &&
           append_array_layout<std::int64_t>(spin_shell_offsets.size(), cursor,
                                             arena.spin_shell_offsets) &&
           append_array_layout<std::int64_t>(spin_atom_offsets.size(), cursor,
                                             arena.spin_atom_offsets) &&
           (arena.total_bytes = cursor, true);
  }

  cudaError_t make_upload_image() noexcept {
    if (arena.total_bytes == 0u) {
      return cudaErrorInvalidValue;
    }
    cudaError_t status = cudaMallocHost(&upload_image, arena.total_bytes);
    if (status != cudaSuccess) {
      upload_image = nullptr;
      return status;
    }

    /* Padding is initialized too, making the complete transfer image defined
     * under initcheck even if a future field introduces wider alignment. */
    std::memset(upload_image, 0, arena.total_bytes);
    auto* const image = static_cast<std::byte*>(upload_image);
    const auto pack = [image](const auto& source, std::size_t offset) noexcept {
      using Value = typename std::decay_t<decltype(source)>::value_type;
      std::memcpy(image + offset, source.data(), source.size() * sizeof(Value));
    };
    pack(atom_offsets, arena.atom_offsets);
    pack(batch_shell_offsets, arena.batch_shell_offsets);
    pack(batch_orbital_offsets, arena.batch_orbital_offsets);
    pack(matrix_offsets, arena.matrix_offsets);
    pack(atom_shell_offsets, arena.atom_shell_offsets);
    pack(shell_orbital_offsets, arena.shell_orbital_offsets);
    pack(shell_to_atom, arena.shell_to_atom);
    pack(orbital_to_shell, arena.orbital_to_shell);
    pack(orbital_to_atom, arena.orbital_to_atom);
    pack(bucket_offsets, arena.bucket_offsets);
    pack(bucket_systems, arena.bucket_systems);
    pack(bucket_orbital_counts, arena.bucket_orbital_counts);
    pack(spin_channels, arena.spin_channels);
    pack(spin_channel_offsets, arena.spin_channel_offsets);
    pack(spin_orbital_offsets, arena.spin_orbital_offsets);
    pack(spin_matrix_offsets, arena.spin_matrix_offsets);
    pack(spin_shell_offsets, arena.spin_shell_offsets);
    pack(spin_atom_offsets, arena.spin_atom_offsets);
    return cudaSuccess;
  }

  void bind_host_descriptor() noexcept {
    host = {};
    host.memory_space = Gfn2PlanMemorySpace::kHost;
    host.pair_map_kind = Gfn2PairMapKind::kNone;
    host.plan_token = plan_token;
    host.batch_size = batch_size;
    host.total_atoms = total_atoms;
    host.total_shells = total_shells;
    host.total_orbitals = total_orbitals;
    host.total_matrix_elements = total_matrix_elements;
    host.bucket_count = static_cast<std::int64_t>(buckets.size());
    host.atom_offset_count = static_cast<std::int64_t>(atom_offsets.size());
    host.batch_shell_offset_count = static_cast<std::int64_t>(batch_shell_offsets.size());
    host.batch_orbital_offset_count = static_cast<std::int64_t>(batch_orbital_offsets.size());
    host.matrix_offset_count = static_cast<std::int64_t>(matrix_offsets.size());
    host.atom_shell_offset_count = static_cast<std::int64_t>(atom_shell_offsets.size());
    host.shell_orbital_offset_count = static_cast<std::int64_t>(shell_orbital_offsets.size());
    host.shell_to_atom_count = static_cast<std::int64_t>(shell_to_atom.size());
    host.orbital_to_shell_count = static_cast<std::int64_t>(orbital_to_shell.size());
    host.orbital_to_atom_count = static_cast<std::int64_t>(orbital_to_atom.size());
    host.bucket_offset_count = static_cast<std::int64_t>(bucket_offsets.size());
    host.bucket_system_count = static_cast<std::int64_t>(bucket_systems.size());
    host.bucket_orbital_count = static_cast<std::int64_t>(bucket_orbital_counts.size());
    host.atom_offsets = atom_offsets.data();
    host.batch_shell_offsets = batch_shell_offsets.data();
    host.batch_orbital_offsets = batch_orbital_offsets.data();
    host.matrix_offsets = matrix_offsets.data();
    host.atom_shell_offsets = atom_shell_offsets.data();
    host.shell_orbital_offsets = shell_orbital_offsets.data();
    host.shell_to_atom = shell_to_atom.data();
    host.orbital_to_shell = orbital_to_shell.data();
    host.orbital_to_atom = orbital_to_atom.data();
    host.bucket_offsets = bucket_offsets.data();
    host.bucket_systems = bucket_systems.data();
    host.bucket_orbital_counts = bucket_orbital_counts.data();

    host_wavefunction = {};
    host_wavefunction.memory_space = Gfn2PlanMemorySpace::kHost;
    host_wavefunction.plan_token = plan_token;
    host_wavefunction.batch_size = batch_size;
    host_wavefunction.total_spin_channels = spin_channel_offsets.back();
    host_wavefunction.total_spin_orbitals = spin_orbital_offsets.back();
    host_wavefunction.total_spin_matrix_elements = spin_matrix_offsets.back();
    host_wavefunction.total_spin_shells = spin_shell_offsets.back();
    host_wavefunction.total_spin_atoms = spin_atom_offsets.back();
    host_wavefunction.spin_channel_count = static_cast<std::int64_t>(spin_channels.size());
    host_wavefunction.spin_channel_offset_count =
        static_cast<std::int64_t>(spin_channel_offsets.size());
    host_wavefunction.spin_orbital_offset_count =
        static_cast<std::int64_t>(spin_orbital_offsets.size());
    host_wavefunction.spin_matrix_offset_count =
        static_cast<std::int64_t>(spin_matrix_offsets.size());
    host_wavefunction.spin_shell_offset_count =
        static_cast<std::int64_t>(spin_shell_offsets.size());
    host_wavefunction.spin_atom_offset_count = static_cast<std::int64_t>(spin_atom_offsets.size());
    host_wavefunction.spin_channels = spin_channels.data();
    host_wavefunction.spin_channel_offsets = spin_channel_offsets.data();
    host_wavefunction.spin_orbital_offsets = spin_orbital_offsets.data();
    host_wavefunction.spin_matrix_offsets = spin_matrix_offsets.data();
    host_wavefunction.spin_shell_offsets = spin_shell_offsets.data();
    host_wavefunction.spin_atom_offsets = spin_atom_offsets.data();
    host_wavefunction.layout_fingerprint =
        gfn2_wavefunction_layout_fingerprint_host(host_wavefunction);
  }
};

Gfn2SccSetupTopology::Gfn2SccSetupTopology() noexcept = default;
Gfn2SccSetupTopology::~Gfn2SccSetupTopology() = default;
Gfn2SccSetupTopology::Gfn2SccSetupTopology(Gfn2SccSetupTopology&&) noexcept = default;
Gfn2SccSetupTopology& Gfn2SccSetupTopology::operator=(Gfn2SccSetupTopology&&) noexcept = default;

Gfn2SccSetupTopologyDiagnostic Gfn2SccSetupTopology::create(const BasisPlan& basis,
                                                            const IntegralPlan& integrals,
                                                            const WavefunctionLayout& wavefunction,
                                                            std::uint64_t plan_token,
                                                            Gfn2SccSetupTopology& output) noexcept {
  Gfn2SccSetupTopologyDiagnostic diagnostic =
      validate_plan_compatibility(basis, integrals, wavefunction, plan_token);
  if (!diagnostic.success()) {
    return diagnostic;
  }
  try {
    std::unique_ptr<Impl> candidate(new (std::nothrow) Impl());
    if (candidate == nullptr) {
      return failure(GPUXTB_STATUS_ALLOCATION_FAILED, Gfn2SccSetupTopologyError::kAllocationFailed,
                     Gfn2SccSetupTopologyField::kHostTopology);
    }
    candidate->plan_token = plan_token;
    candidate->batch_size = basis.batch_size;
    candidate->total_atoms = basis.total_atoms;
    candidate->total_shells = basis.total_shells;
    candidate->total_orbitals = basis.total_orbitals;
    candidate->total_matrix_elements = integrals.total_matrix_elements;
    candidate->atom_offsets = basis.atom_offsets;
    candidate->batch_shell_offsets = basis.batch_shell_offsets;
    candidate->batch_orbital_offsets = basis.batch_orbital_offsets;
    candidate->matrix_offsets = integrals.matrix_offsets;
    candidate->atom_shell_offsets = basis.atom_shell_offsets;
    candidate->shell_orbital_offsets = basis.shell_orbital_offsets;
    candidate->shell_to_atom = basis.shell_to_atom;
    candidate->spin_channels = wavefunction.spin_channels;
    candidate->spin_channel_offsets.resize(static_cast<std::size_t>(basis.batch_size) + 1u, 0);
    for (std::int64_t system = 0; system < basis.batch_size; ++system) {
      const std::int32_t channels = candidate->spin_channels[static_cast<std::size_t>(system)];
      if (channels != 1 && channels != 2) {
        return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Gfn2SccSetupTopologyError::kInvalidPlan,
                       Gfn2SccSetupTopologyField::kWavefunction, system);
      }
      const std::int64_t previous =
          candidate->spin_channel_offsets[static_cast<std::size_t>(system)];
      if (previous > std::numeric_limits<std::int64_t>::max() - channels) {
        return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Gfn2SccSetupTopologyError::kCountOverflow,
                       Gfn2SccSetupTopologyField::kWavefunction, system);
      }
      candidate->spin_channel_offsets[static_cast<std::size_t>(system + 1)] = previous + channels;
    }
    candidate->spin_orbital_offsets = wavefunction.eigenvalues.system_offsets;
    candidate->spin_matrix_offsets = wavefunction.density.system_offsets;
    candidate->spin_shell_offsets = wavefunction.qsh.system_offsets;
    candidate->spin_atom_offsets = wavefunction.qat.system_offsets;
    if (wavefunction.coefficients.system_offsets != candidate->spin_matrix_offsets ||
        wavefunction.energy_weighted_density.system_offsets != candidate->spin_matrix_offsets) {
      return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Gfn2SccSetupTopologyError::kInvalidPlan,
                     Gfn2SccSetupTopologyField::kWavefunction);
    }

    diagnostic =
        build_buckets(basis, integrals, candidate->bucket_offsets, candidate->bucket_systems,
                      candidate->bucket_orbital_counts, candidate->buckets);
    if (!diagnostic.success()) {
      return diagnostic;
    }
    diagnostic = configure_spin_buckets(candidate->spin_channels, candidate->bucket_systems,
                                        candidate->buckets, candidate->spin_channel_offsets.back(),
                                        candidate->spin_orbital_offsets.back(),
                                        candidate->spin_matrix_offsets.back());
    if (!diagnostic.success()) {
      return diagnostic;
    }

    candidate->orbital_to_shell.assign(static_cast<std::size_t>(basis.total_orbitals), -1);
    candidate->orbital_to_atom.assign(static_cast<std::size_t>(basis.total_orbitals), -1);
    for (std::int64_t shell = 0; shell < basis.total_shells; ++shell) {
      const std::int64_t atom = basis.shell_to_atom[static_cast<std::size_t>(shell)];
      if (atom < 0 || atom >= basis.total_atoms) {
        return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Gfn2SccSetupTopologyError::kInvalidPlan,
                       Gfn2SccSetupTopologyField::kOrbitalMap, shell);
      }
      const std::int64_t begin = basis.shell_orbital_offsets[static_cast<std::size_t>(shell)];
      const std::int64_t end = basis.shell_orbital_offsets[static_cast<std::size_t>(shell + 1)];
      for (std::int64_t orbital = begin; orbital < end; ++orbital) {
        candidate->orbital_to_shell[static_cast<std::size_t>(orbital)] = shell;
        candidate->orbital_to_atom[static_cast<std::size_t>(orbital)] = atom;
      }
    }
    if (!candidate->make_arena_layout()) {
      return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Gfn2SccSetupTopologyError::kCountOverflow,
                     Gfn2SccSetupTopologyField::kArena);
    }
    candidate->bind_host_descriptor();
    const Gfn2PlanSchemaDiagnostic schema = validate_gfn2_topology_host(candidate->host);
    if (schema.error != Gfn2PlanSchemaError::kSuccess) {
      diagnostic = failure(GPUXTB_STATUS_INVALID_ARGUMENT, Gfn2SccSetupTopologyError::kInvalidPlan,
                           Gfn2SccSetupTopologyField::kHostTopology, schema.index);
      diagnostic.schema = schema;
      return diagnostic;
    }
    const Gfn2PlanSchemaDiagnostic wavefunction_schema =
        validate_gfn2_wavefunction_layout_host(candidate->host, candidate->host_wavefunction);
    if (wavefunction_schema.error != Gfn2PlanSchemaError::kSuccess) {
      diagnostic = failure(GPUXTB_STATUS_INVALID_ARGUMENT, Gfn2SccSetupTopologyError::kInvalidPlan,
                           Gfn2SccSetupTopologyField::kWavefunction, wavefunction_schema.index);
      diagnostic.schema = wavefunction_schema;
      return diagnostic;
    }

    const cudaError_t upload_image_status = candidate->make_upload_image();
    if (upload_image_status != cudaSuccess) {
      Gfn2SccSetupTopologyDiagnostic upload_diagnostic =
          failure(upload_image_status == cudaErrorMemoryAllocation ? GPUXTB_STATUS_ALLOCATION_FAILED
                                                                   : GPUXTB_STATUS_INTERNAL_ERROR,
                  upload_image_status == cudaErrorMemoryAllocation
                      ? Gfn2SccSetupTopologyError::kAllocationFailed
                      : Gfn2SccSetupTopologyError::kCudaError,
                  Gfn2SccSetupTopologyField::kHostTopology);
      upload_diagnostic.required_bytes = candidate->arena.total_bytes;
      upload_diagnostic.cuda_status = upload_image_status;
      return upload_diagnostic;
    }

    Gfn2SccSetupTopology replacement;
    replacement.impl_ = std::move(candidate);
    output = std::move(replacement);
    return {};
  } catch (const std::bad_alloc&) {
    return failure(GPUXTB_STATUS_ALLOCATION_FAILED, Gfn2SccSetupTopologyError::kAllocationFailed,
                   Gfn2SccSetupTopologyField::kHostTopology);
  } catch (...) {
    return failure(GPUXTB_STATUS_INTERNAL_ERROR, Gfn2SccSetupTopologyError::kInvalidPlan,
                   Gfn2SccSetupTopologyField::kHostTopology);
  }
}

bool Gfn2SccSetupTopology::valid() const noexcept { return impl_ != nullptr; }

const Gfn2RaggedTopologyView& Gfn2SccSetupTopology::host_topology() const noexcept {
  return impl_ == nullptr ? empty_topology() : impl_->host;
}

const Gfn2WavefunctionLayoutView& Gfn2SccSetupTopology::host_wavefunction_layout() const noexcept {
  return impl_ == nullptr ? empty_wavefunction_layout() : impl_->host_wavefunction;
}

const std::vector<Gfn2EigensolverBucket>& Gfn2SccSetupTopology::eigensolver_buckets()
    const noexcept {
  return impl_ == nullptr ? empty_buckets() : impl_->buckets;
}

Gfn2SccSetupTopologyRequirements Gfn2SccSetupTopology::requirements() const noexcept {
  Gfn2SccSetupTopologyRequirements result{};
  if (impl_ != nullptr) {
    result.immutable_device_bytes = impl_->arena.total_bytes;
  }
  return result;
}

Gfn2SccSetupTopologyDiagnostic Gfn2SccSetupTopology::bind_device_arena_and_upload_async(
    void* device_arena, std::size_t device_arena_bytes, Gfn2RaggedTopologyView& device_topology,
    Gfn2WavefunctionLayoutView& device_wavefunction, cudaStream_t stream) const noexcept {
  if (impl_ == nullptr) {
    return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Gfn2SccSetupTopologyError::kInvalidPlan,
                   Gfn2SccSetupTopologyField::kHostTopology);
  }

  const std::size_t required_bytes = impl_->arena.total_bytes;
  if (device_arena == nullptr) {
    return arena_failure(Gfn2SccSetupTopologyError::kNullArena, required_bytes);
  }
  const std::uintptr_t arena_address = reinterpret_cast<std::uintptr_t>(device_arena);
  if (arena_address % requirements().device_alignment != 0u) {
    return arena_failure(Gfn2SccSetupTopologyError::kMisalignedArena, required_bytes);
  }
  if (device_arena_bytes < required_bytes) {
    return arena_failure(Gfn2SccSetupTopologyError::kInsufficientArena, required_bytes);
  }
  if (required_bytes != 0u &&
      required_bytes - 1u > std::numeric_limits<std::uintptr_t>::max() - arena_address) {
    return arena_failure(Gfn2SccSetupTopologyError::kCountOverflow, required_bytes);
  }

  /*
   * cudaMemcpyAsync can otherwise accept registered host or peer addresses.
   * This owner promises a CUDA-device descriptor, so reject anything that is
   * not directly device accessible before any transfer is enqueued.
   */
  cudaPointerAttributes attributes{};
  const cudaError_t attribute_status = cudaPointerGetAttributes(&attributes, device_arena);
  if (attribute_status != cudaSuccess) {
    /* cudaPointerGetAttributes records invalid host pointers in the runtime's
     * last-error slot. Consume that setup diagnostic so a later hot launch is
     * not poisoned after the caller handles this fail-closed result. */
    (void)cudaGetLastError();
    return arena_failure(Gfn2SccSetupTopologyError::kInvalidArenaMemory, required_bytes,
                         attribute_status);
  }
  if (attributes.type != cudaMemoryTypeDevice && attributes.type != cudaMemoryTypeManaged) {
    return arena_failure(Gfn2SccSetupTopologyError::kInvalidArenaMemory, required_bytes);
  }

  auto* const arena = static_cast<std::byte*>(device_arena);
  Gfn2RaggedTopologyView candidate = impl_->host;
  candidate.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
  candidate.atom_offsets = reinterpret_cast<const std::int64_t*>(arena + impl_->arena.atom_offsets);
  candidate.batch_shell_offsets =
      reinterpret_cast<const std::int64_t*>(arena + impl_->arena.batch_shell_offsets);
  candidate.batch_orbital_offsets =
      reinterpret_cast<const std::int64_t*>(arena + impl_->arena.batch_orbital_offsets);
  candidate.matrix_offsets =
      reinterpret_cast<const std::int64_t*>(arena + impl_->arena.matrix_offsets);
  candidate.atom_shell_offsets =
      reinterpret_cast<const std::int64_t*>(arena + impl_->arena.atom_shell_offsets);
  candidate.shell_orbital_offsets =
      reinterpret_cast<const std::int64_t*>(arena + impl_->arena.shell_orbital_offsets);
  candidate.shell_to_atom =
      reinterpret_cast<const std::int64_t*>(arena + impl_->arena.shell_to_atom);
  candidate.orbital_to_shell =
      reinterpret_cast<const std::int64_t*>(arena + impl_->arena.orbital_to_shell);
  candidate.orbital_to_atom =
      reinterpret_cast<const std::int64_t*>(arena + impl_->arena.orbital_to_atom);
  candidate.bucket_offsets =
      reinterpret_cast<const std::int64_t*>(arena + impl_->arena.bucket_offsets);
  candidate.bucket_systems =
      reinterpret_cast<const std::int32_t*>(arena + impl_->arena.bucket_systems);
  candidate.bucket_orbital_counts =
      reinterpret_cast<const std::int32_t*>(arena + impl_->arena.bucket_orbital_counts);

  Gfn2WavefunctionLayoutView wavefunction_candidate = impl_->host_wavefunction;
  wavefunction_candidate.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
  wavefunction_candidate.spin_channels =
      reinterpret_cast<const std::int32_t*>(arena + impl_->arena.spin_channels);
  wavefunction_candidate.spin_channel_offsets =
      reinterpret_cast<const std::int64_t*>(arena + impl_->arena.spin_channel_offsets);
  wavefunction_candidate.spin_orbital_offsets =
      reinterpret_cast<const std::int64_t*>(arena + impl_->arena.spin_orbital_offsets);
  wavefunction_candidate.spin_matrix_offsets =
      reinterpret_cast<const std::int64_t*>(arena + impl_->arena.spin_matrix_offsets);
  wavefunction_candidate.spin_shell_offsets =
      reinterpret_cast<const std::int64_t*>(arena + impl_->arena.spin_shell_offsets);
  wavefunction_candidate.spin_atom_offsets =
      reinterpret_cast<const std::int64_t*>(arena + impl_->arena.spin_atom_offsets);

  /* The private descriptor must remain structurally valid even if this layout
   * changes later. Semantic validation is unnecessary here: create() already
   * validated the identical host values that are being uploaded. */
  const Gfn2PlanSchemaDiagnostic schema =
      validate_gfn2_topology_binding(candidate, Gfn2PlanMemorySpace::kCudaDevice);
  if (schema.error != Gfn2PlanSchemaError::kSuccess) {
    Gfn2SccSetupTopologyDiagnostic diagnostic =
        failure(GPUXTB_STATUS_INTERNAL_ERROR, Gfn2SccSetupTopologyError::kInvalidPlan,
                Gfn2SccSetupTopologyField::kHostTopology, schema.index);
    diagnostic.schema = schema;
    return diagnostic;
  }
  const Gfn2PlanSchemaDiagnostic wavefunction_schema = validate_gfn2_wavefunction_layout_binding(
      candidate, wavefunction_candidate, Gfn2PlanMemorySpace::kCudaDevice);
  if (wavefunction_schema.error != Gfn2PlanSchemaError::kSuccess) {
    Gfn2SccSetupTopologyDiagnostic diagnostic =
        failure(GPUXTB_STATUS_INTERNAL_ERROR, Gfn2SccSetupTopologyError::kInvalidPlan,
                Gfn2SccSetupTopologyField::kWavefunction, wavefunction_schema.index);
    diagnostic.schema = wavefunction_schema;
    return diagnostic;
  }

  /*
   * Publication is descriptor-transactional: a runtime rejection leaves the
   * caller's previous binding intact. The immutable source image is pinned,
   * so this single submission neither allocates nor synchronizes the stream.
   */
  const cudaError_t status = cudaMemcpyAsync(device_arena, impl_->upload_image, required_bytes,
                                             cudaMemcpyHostToDevice, stream);
  if (status != cudaSuccess) {
    return arena_failure(Gfn2SccSetupTopologyError::kCudaError, required_bytes, status);
  }

  device_topology = candidate;
  device_wavefunction = wavefunction_candidate;
  return {};
}

Gfn2SccSetupTopologyDiagnostic Gfn2SccSetupTopology::bind_device_arena_and_upload_async(
    void* device_arena, std::size_t device_arena_bytes, Gfn2RaggedTopologyView& device_topology,
    cudaStream_t stream) const noexcept {
  Gfn2WavefunctionLayoutView ignored{};
  return bind_device_arena_and_upload_async(device_arena, device_arena_bytes, device_topology,
                                            ignored, stream);
}

}  // namespace gpuxtb::detail::cuda
