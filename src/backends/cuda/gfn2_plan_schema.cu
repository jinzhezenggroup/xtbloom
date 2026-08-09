#include <array>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_plan_schema.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr std::int64_t kInt64Maximum = 9223372036854775807LL;

bool host_add_one(std::int64_t value, std::int64_t& result) noexcept {
  if (value == kInt64Maximum) {
    return false;
  }
  result = value + 1;
  return true;
}

__host__ __device__ Gfn2PlanSchemaDiagnostic success() { return {}; }

__host__ __device__ Gfn2PlanSchemaDiagnostic failure(Gfn2PlanSchemaError error,
                                                     Gfn2PlanSchemaField field,
                                                     std::int64_t index = -1) {
  return {error, field, index};
}

__device__ bool square(std::int64_t value, std::int64_t* result) {
  if (value < 0 || (value != 0 && value > kInt64Maximum / value)) {
    return false;
  }
  *result = value * value;
  return true;
}

__device__ bool triangle(std::int64_t value, std::int64_t* result) {
  if (value < 0 || (value > 1 && value > kInt64Maximum / (value - 1))) {
    return false;
  }
  *result = (value & 1LL) == 0LL ? (value / 2LL) * (value - 1LL) : value * ((value - 1LL) / 2LL);
  return true;
}

__device__ Gfn2PlanSchemaDiagnostic validate_offsets(const std::int64_t* offsets,
                                                     std::int64_t partitions, std::int64_t endpoint,
                                                     Gfn2PlanSchemaField field) {
  if (offsets[0] != 0) {
    return failure(Gfn2PlanSchemaError::kInvalidOffsets, field, 0);
  }
  for (std::int64_t index = 0; index < partitions; ++index) {
    if (offsets[index] < 0 || offsets[index] > offsets[index + 1] ||
        offsets[index + 1] > endpoint) {
      return failure(Gfn2PlanSchemaError::kInvalidOffsets, field, index);
    }
  }
  return offsets[partitions] == endpoint
             ? success()
             : failure(Gfn2PlanSchemaError::kInvalidOffsets, field, partitions);
}

__device__ Gfn2PlanSchemaDiagnostic inspect_topology(Gfn2RaggedTopologyView topology) {
  Gfn2PlanSchemaDiagnostic diagnostic =
      validate_offsets(topology.atom_offsets, topology.batch_size, topology.total_atoms,
                       Gfn2PlanSchemaField::kAtomOffsets);
  if (diagnostic.error == Gfn2PlanSchemaError::kSuccess) {
    diagnostic = validate_offsets(topology.batch_shell_offsets, topology.batch_size,
                                  topology.total_shells, Gfn2PlanSchemaField::kBatchShellOffsets);
  }
  if (diagnostic.error == Gfn2PlanSchemaError::kSuccess) {
    diagnostic =
        validate_offsets(topology.batch_orbital_offsets, topology.batch_size,
                         topology.total_orbitals, Gfn2PlanSchemaField::kBatchOrbitalOffsets);
  }
  if (diagnostic.error == Gfn2PlanSchemaError::kSuccess) {
    diagnostic =
        validate_offsets(topology.matrix_offsets, topology.batch_size,
                         topology.total_matrix_elements, Gfn2PlanSchemaField::kMatrixOffsets);
  }
  if (diagnostic.error == Gfn2PlanSchemaError::kSuccess) {
    diagnostic = validate_offsets(topology.atom_shell_offsets, topology.total_atoms,
                                  topology.total_shells, Gfn2PlanSchemaField::kAtomShellOffsets);
  }
  if (diagnostic.error == Gfn2PlanSchemaError::kSuccess) {
    diagnostic =
        validate_offsets(topology.shell_orbital_offsets, topology.total_shells,
                         topology.total_orbitals, Gfn2PlanSchemaField::kShellOrbitalOffsets);
  }
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }

  for (std::int64_t system = 0; system < topology.batch_size; ++system) {
    const std::int64_t atom_begin = topology.atom_offsets[system];
    const std::int64_t atom_end = topology.atom_offsets[system + 1];
    const std::int64_t shell_begin = topology.batch_shell_offsets[system];
    const std::int64_t shell_end = topology.batch_shell_offsets[system + 1];
    const std::int64_t orbital_begin = topology.batch_orbital_offsets[system];
    const std::int64_t orbital_end = topology.batch_orbital_offsets[system + 1];
    std::int64_t matrix_count = 0;
    if (!square(orbital_end - orbital_begin, &matrix_count)) {
      return failure(Gfn2PlanSchemaError::kCountOverflow, Gfn2PlanSchemaField::kMatrixOffsets,
                     system);
    }
    if (topology.matrix_offsets[system + 1] - topology.matrix_offsets[system] != matrix_count) {
      return failure(Gfn2PlanSchemaError::kInvalidMatrixExtent, Gfn2PlanSchemaField::kMatrixOffsets,
                     system);
    }
    if (topology.atom_shell_offsets[atom_begin] != shell_begin ||
        topology.atom_shell_offsets[atom_end] != shell_end ||
        topology.shell_orbital_offsets[shell_begin] != orbital_begin ||
        topology.shell_orbital_offsets[shell_end] != orbital_end) {
      return failure(Gfn2PlanSchemaError::kInvalidOffsets, Gfn2PlanSchemaField::kAtomShellOffsets,
                     system);
    }
    if ((atom_end == atom_begin) != (shell_end == shell_begin) ||
        (shell_end == shell_begin) != (orbital_end == orbital_begin)) {
      return failure(Gfn2PlanSchemaError::kInvalidOffsets, Gfn2PlanSchemaField::kBatchShellOffsets,
                     system);
    }
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
      const std::int64_t owned_shell_begin = topology.atom_shell_offsets[atom];
      const std::int64_t owned_shell_end = topology.atom_shell_offsets[atom + 1];
      if (owned_shell_begin >= owned_shell_end) {
        return failure(Gfn2PlanSchemaError::kInvalidShellMap,
                       Gfn2PlanSchemaField::kAtomShellOffsets, atom);
      }
      for (std::int64_t shell = owned_shell_begin; shell < owned_shell_end; ++shell) {
        if (topology.shell_to_atom[shell] != atom) {
          return failure(Gfn2PlanSchemaError::kInvalidShellMap, Gfn2PlanSchemaField::kShellToAtom,
                         shell);
        }
      }
    }
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      const std::int64_t owned_orbital_begin = topology.shell_orbital_offsets[shell];
      const std::int64_t owned_orbital_end = topology.shell_orbital_offsets[shell + 1];
      if (owned_orbital_begin >= owned_orbital_end) {
        return failure(Gfn2PlanSchemaError::kInvalidOrbitalMap,
                       Gfn2PlanSchemaField::kShellOrbitalOffsets, shell);
      }
      const std::int64_t atom = topology.shell_to_atom[shell];
      if (atom < atom_begin || atom >= atom_end) {
        return failure(Gfn2PlanSchemaError::kInvalidShellMap, Gfn2PlanSchemaField::kShellToAtom,
                       shell);
      }
      for (std::int64_t orbital = owned_orbital_begin; orbital < owned_orbital_end; ++orbital) {
        if (topology.orbital_to_shell[orbital] != shell) {
          return failure(Gfn2PlanSchemaError::kInvalidOrbitalMap,
                         Gfn2PlanSchemaField::kOrbitalToShell, orbital);
        }
        if (topology.orbital_to_atom[orbital] != atom) {
          return failure(Gfn2PlanSchemaError::kInvalidOrbitalMap,
                         Gfn2PlanSchemaField::kOrbitalToAtom, orbital);
        }
      }
    }
  }

  if (topology.pair_map_kind != Gfn2PairMapKind::kNone) {
    diagnostic = validate_offsets(topology.pair_offsets, topology.batch_size, topology.total_pairs,
                                  Gfn2PlanSchemaField::kPairOffsets);
    if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
      return diagnostic;
    }
    for (std::int64_t system = 0; system < topology.batch_size; ++system) {
      const std::int64_t pair_begin = topology.pair_offsets[system];
      const std::int64_t pair_end = topology.pair_offsets[system + 1];
      const std::int64_t atom_begin = topology.atom_offsets[system];
      const std::int64_t atom_end = topology.atom_offsets[system + 1];
      if (topology.pair_map_kind == Gfn2PairMapKind::kPackedLowerTriangle) {
        std::int64_t expected_pairs = 0;
        if (!triangle(atom_end - atom_begin, &expected_pairs)) {
          return failure(Gfn2PlanSchemaError::kCountOverflow, Gfn2PlanSchemaField::kPairOffsets,
                         system);
        }
        if (pair_end - pair_begin != expected_pairs) {
          return failure(Gfn2PlanSchemaError::kInvalidPairMap, Gfn2PlanSchemaField::kPairOffsets,
                         system);
        }
      } else {
        Gfn2AtomPair previous{-1, -1};
        for (std::int64_t pair = pair_begin; pair < pair_end; ++pair) {
          const Gfn2AtomPair current = topology.atom_pairs[pair];
          const bool in_system = current.first >= atom_begin && current.first < current.second &&
                                 current.second < atom_end;
          const bool sorted = pair == pair_begin || previous.first < current.first ||
                              (previous.first == current.first && previous.second < current.second);
          if (!in_system || !sorted) {
            return failure(Gfn2PlanSchemaError::kInvalidPairMap, Gfn2PlanSchemaField::kAtomPairs,
                           pair);
          }
          previous = current;
        }
      }
    }
  }

  if (topology.bucket_count != 0) {
    diagnostic = validate_offsets(topology.bucket_offsets, topology.bucket_count,
                                  topology.batch_size, Gfn2PlanSchemaField::kBucketOffsets);
    if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
      return diagnostic;
    }
    for (std::int64_t bucket = 0; bucket < topology.bucket_count; ++bucket) {
      if (topology.bucket_offsets[bucket] == topology.bucket_offsets[bucket + 1]) {
        return failure(Gfn2PlanSchemaError::kInvalidBucketMap, Gfn2PlanSchemaField::kBucketOffsets,
                       bucket);
      }
      if (topology.bucket_orbital_counts[bucket] < 0) {
        return failure(Gfn2PlanSchemaError::kInvalidBucketMap,
                       Gfn2PlanSchemaField::kBucketOrbitalCounts, bucket);
      }
      for (std::int64_t position = topology.bucket_offsets[bucket];
           position < topology.bucket_offsets[bucket + 1]; ++position) {
        const std::int64_t system = topology.bucket_systems[position];
        if (system < 0 || system >= topology.batch_size) {
          return failure(Gfn2PlanSchemaError::kInvalidBucketMap,
                         Gfn2PlanSchemaField::kBucketSystems, position);
        }
        const std::int64_t orbitals =
            topology.batch_orbital_offsets[system + 1] - topology.batch_orbital_offsets[system];
        if (orbitals != topology.bucket_orbital_counts[bucket]) {
          return failure(Gfn2PlanSchemaError::kInvalidBucketMap,
                         Gfn2PlanSchemaField::kBucketOrbitalCounts, bucket);
        }
        for (std::int64_t earlier = 0; earlier < position; ++earlier) {
          if (topology.bucket_systems[earlier] == system) {
            return failure(Gfn2PlanSchemaError::kInvalidBucketMap,
                           Gfn2PlanSchemaField::kBucketSystems, position);
          }
        }
      }
    }
  }
  return success();
}

__global__ void topology_validation_kernel(Gfn2RaggedTopologyView topology,
                                           Gfn2PlanSchemaDiagnostic* diagnostic) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *diagnostic = inspect_topology(topology);
  }
}

__global__ void provenance_validation_kernel(Gfn2GeometryCacheProvenanceView provenance,
                                             std::uint64_t expected_geometry_generation,
                                             const std::uint8_t* active_mask,
                                             Gfn2PlanSchemaDiagnostic* diagnostic) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  if (expected_geometry_generation == 0u) {
    *diagnostic =
        failure(Gfn2PlanSchemaError::kStaleGeometry, Gfn2PlanSchemaField::kGeometryProvenance);
    return;
  }
  for (std::int64_t system = 0; active_mask != nullptr && system < provenance.batch_size;
       ++system) {
    if (active_mask[system] > 1u) {
      *diagnostic = failure(Gfn2PlanSchemaError::kInvalidActiveMask,
                            Gfn2PlanSchemaField::kActiveMask, system);
      return;
    }
  }
  if (provenance.generation_scope == Gfn2GenerationScope::kBatch) {
    *diagnostic = provenance.geometry_generation == expected_geometry_generation
                      ? success()
                      : failure(Gfn2PlanSchemaError::kStaleGeometry,
                                Gfn2PlanSchemaField::kGeometryProvenance);
    return;
  }
  for (std::int64_t system = 0; system < provenance.batch_size; ++system) {
    const bool active = active_mask == nullptr || active_mask[system] == 1u;
    if (active && provenance.system_geometry_generations[system] != expected_geometry_generation) {
      *diagnostic = failure(Gfn2PlanSchemaError::kStaleGeometry,
                            Gfn2PlanSchemaField::kSystemGeometryGenerations, system);
      return;
    }
  }
  *diagnostic = success();
}

bool pointer_is_cuda_accessible(const void* pointer) noexcept {
  if (pointer == nullptr) {
    return false;
  }
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  if (status != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged;
}

bool diagnostic_range_is_valid(const Gfn2PlanSchemaDiagnostic* pointer) noexcept {
  if (pointer == nullptr ||
      reinterpret_cast<std::uintptr_t>(pointer) % alignof(Gfn2PlanSchemaDiagnostic) != 0u) {
    return false;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  return begin <= std::numeric_limits<std::uintptr_t>::max() - sizeof(Gfn2PlanSchemaDiagnostic);
}

Gfn2PlanSchemaDiagnostic validate_cuda_topology_pointers(
    const Gfn2RaggedTopologyView& topology) noexcept {
  const void* pointers[] = {
      topology.atom_offsets,
      topology.batch_shell_offsets,
      topology.batch_orbital_offsets,
      topology.matrix_offsets,
      topology.atom_shell_offsets,
      topology.shell_orbital_offsets,
      topology.shell_to_atom,
      topology.orbital_to_shell,
      topology.orbital_to_atom,
      topology.pair_offsets,
      topology.atom_pairs,
      topology.bucket_offsets,
      topology.bucket_systems,
      topology.bucket_orbital_counts,
  };
  const std::int64_t counts[] = {
      topology.atom_offset_count,
      topology.batch_shell_offset_count,
      topology.batch_orbital_offset_count,
      topology.matrix_offset_count,
      topology.atom_shell_offset_count,
      topology.shell_orbital_offset_count,
      topology.shell_to_atom_count,
      topology.orbital_to_shell_count,
      topology.orbital_to_atom_count,
      topology.pair_offset_count,
      topology.atom_pair_count,
      topology.bucket_offset_count,
      topology.bucket_system_count,
      topology.bucket_orbital_count,
  };
  const Gfn2PlanSchemaField fields[] = {
      Gfn2PlanSchemaField::kAtomOffsets,         Gfn2PlanSchemaField::kBatchShellOffsets,
      Gfn2PlanSchemaField::kBatchOrbitalOffsets, Gfn2PlanSchemaField::kMatrixOffsets,
      Gfn2PlanSchemaField::kAtomShellOffsets,    Gfn2PlanSchemaField::kShellOrbitalOffsets,
      Gfn2PlanSchemaField::kShellToAtom,         Gfn2PlanSchemaField::kOrbitalToShell,
      Gfn2PlanSchemaField::kOrbitalToAtom,       Gfn2PlanSchemaField::kPairOffsets,
      Gfn2PlanSchemaField::kAtomPairs,           Gfn2PlanSchemaField::kBucketOffsets,
      Gfn2PlanSchemaField::kBucketSystems,       Gfn2PlanSchemaField::kBucketOrbitalCounts,
  };
  for (std::size_t index = 0u; index < std::size(pointers); ++index) {
    if (counts[index] != 0 && !pointer_is_cuda_accessible(pointers[index])) {
      return failure(Gfn2PlanSchemaError::kInvalidMemorySpace, fields[index]);
    }
  }
  return success();
}

bool pointer_range_overlaps(const void* first, std::size_t first_bytes, const void* second,
                            std::size_t second_bytes) noexcept {
  if (first == nullptr || second == nullptr || first_bytes == 0u || second_bytes == 0u) {
    return false;
  }
  const std::uintptr_t first_begin = reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t second_begin = reinterpret_cast<std::uintptr_t>(second);
  if (first_begin > std::numeric_limits<std::uintptr_t>::max() - first_bytes ||
      second_begin > std::numeric_limits<std::uintptr_t>::max() - second_bytes) {
    return true;
  }
  return first_begin < second_begin + second_bytes && second_begin < first_begin + first_bytes;
}

/* Structural validation establishes every input extent before this helper is called. */
bool diagnostic_aliases_valid_range(const Gfn2PlanSchemaDiagnostic* device_diagnostic,
                                    const void* input, std::int64_t input_elements,
                                    std::size_t element_size) noexcept {
  if (input_elements == 0) {
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(input_elements) * element_size;
  return pointer_range_overlaps(device_diagnostic, sizeof(Gfn2PlanSchemaDiagnostic), input, bytes);
}

Gfn2PlanSchemaDiagnostic diagnostic_topology_alias(
    const Gfn2RaggedTopologyView& topology,
    const Gfn2PlanSchemaDiagnostic* device_diagnostic) noexcept {
  const void* pointers[] = {
      topology.atom_offsets,
      topology.batch_shell_offsets,
      topology.batch_orbital_offsets,
      topology.matrix_offsets,
      topology.atom_shell_offsets,
      topology.shell_orbital_offsets,
      topology.shell_to_atom,
      topology.orbital_to_shell,
      topology.orbital_to_atom,
      topology.pair_offsets,
      topology.atom_pairs,
      topology.bucket_offsets,
      topology.bucket_systems,
      topology.bucket_orbital_counts,
  };
  const std::int64_t counts[] = {
      topology.atom_offset_count,
      topology.batch_shell_offset_count,
      topology.batch_orbital_offset_count,
      topology.matrix_offset_count,
      topology.atom_shell_offset_count,
      topology.shell_orbital_offset_count,
      topology.shell_to_atom_count,
      topology.orbital_to_shell_count,
      topology.orbital_to_atom_count,
      topology.pair_offset_count,
      topology.atom_pair_count,
      topology.bucket_offset_count,
      topology.bucket_system_count,
      topology.bucket_orbital_count,
  };
  const std::size_t element_sizes[] = {
      sizeof(std::int64_t), sizeof(std::int64_t), sizeof(std::int64_t), sizeof(std::int64_t),
      sizeof(std::int64_t), sizeof(std::int64_t), sizeof(std::int64_t), sizeof(std::int64_t),
      sizeof(std::int64_t), sizeof(std::int64_t), sizeof(Gfn2AtomPair), sizeof(std::int64_t),
      sizeof(std::int32_t), sizeof(std::int32_t),
  };
  const Gfn2PlanSchemaField fields[] = {
      Gfn2PlanSchemaField::kAtomOffsets,         Gfn2PlanSchemaField::kBatchShellOffsets,
      Gfn2PlanSchemaField::kBatchOrbitalOffsets, Gfn2PlanSchemaField::kMatrixOffsets,
      Gfn2PlanSchemaField::kAtomShellOffsets,    Gfn2PlanSchemaField::kShellOrbitalOffsets,
      Gfn2PlanSchemaField::kShellToAtom,         Gfn2PlanSchemaField::kOrbitalToShell,
      Gfn2PlanSchemaField::kOrbitalToAtom,       Gfn2PlanSchemaField::kPairOffsets,
      Gfn2PlanSchemaField::kAtomPairs,           Gfn2PlanSchemaField::kBucketOffsets,
      Gfn2PlanSchemaField::kBucketSystems,       Gfn2PlanSchemaField::kBucketOrbitalCounts,
  };
  for (std::size_t index = 0u; index < std::size(pointers); ++index) {
    if (diagnostic_aliases_valid_range(device_diagnostic, pointers[index], counts[index],
                                       element_sizes[index])) {
      return failure(Gfn2PlanSchemaError::kAliasedRange, fields[index]);
    }
  }
  return success();
}

Gfn2PlanSchemaDiagnostic diagnostic_provenance_alias(
    const Gfn2RaggedTopologyView& topology, const Gfn2GeometryCacheProvenanceView& provenance,
    const std::uint8_t* active_mask, std::int64_t active_mask_elements,
    const Gfn2PlanSchemaDiagnostic* device_diagnostic) noexcept {
  Gfn2PlanSchemaDiagnostic diagnostic = diagnostic_topology_alias(topology, device_diagnostic);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return diagnostic;
  }
  if (diagnostic_aliases_valid_range(device_diagnostic, provenance.system_geometry_generations,
                                     provenance.system_generation_count, sizeof(std::uint64_t))) {
    return failure(Gfn2PlanSchemaError::kAliasedRange,
                   Gfn2PlanSchemaField::kSystemGeometryGenerations);
  }
  if (diagnostic_aliases_valid_range(device_diagnostic, active_mask, active_mask_elements,
                                     sizeof(std::uint8_t))) {
    return failure(Gfn2PlanSchemaError::kAliasedRange, Gfn2PlanSchemaField::kActiveMask);
  }
  return success();
}

Gfn2PlanSchemaDiagnostic validate_cuda_active_mask(
    const Gfn2RaggedTopologyView& topology, const Gfn2GeometryCacheProvenanceView& provenance,
    const std::uint8_t* active_mask, std::int64_t active_mask_elements,
    bool inspect_address_space) noexcept {
  if (active_mask == nullptr) {
    return active_mask_elements == 0
               ? success()
               : failure(Gfn2PlanSchemaError::kInvalidActiveMask, Gfn2PlanSchemaField::kActiveMask);
  }
  if (active_mask_elements != topology.batch_size ||
      (inspect_address_space && !pointer_is_cuda_accessible(active_mask))) {
    return failure(Gfn2PlanSchemaError::kInvalidActiveMask, Gfn2PlanSchemaField::kActiveMask);
  }
  if (provenance.system_generation_count != 0 &&
      pointer_range_overlaps(
          active_mask, static_cast<std::size_t>(active_mask_elements),
          provenance.system_geometry_generations,
          static_cast<std::size_t>(provenance.system_generation_count) * sizeof(std::uint64_t))) {
    return failure(Gfn2PlanSchemaError::kAliasedRange, Gfn2PlanSchemaField::kActiveMask);
  }
  const void* topology_pointers[] = {
      topology.atom_offsets,
      topology.batch_shell_offsets,
      topology.batch_orbital_offsets,
      topology.matrix_offsets,
      topology.atom_shell_offsets,
      topology.shell_orbital_offsets,
      topology.shell_to_atom,
      topology.orbital_to_shell,
      topology.orbital_to_atom,
      topology.pair_offsets,
      topology.atom_pairs,
      topology.bucket_offsets,
      topology.bucket_systems,
      topology.bucket_orbital_counts,
  };
  const std::int64_t topology_counts[] = {
      topology.atom_offset_count,
      topology.batch_shell_offset_count,
      topology.batch_orbital_offset_count,
      topology.matrix_offset_count,
      topology.atom_shell_offset_count,
      topology.shell_orbital_offset_count,
      topology.shell_to_atom_count,
      topology.orbital_to_shell_count,
      topology.orbital_to_atom_count,
      topology.pair_offset_count,
      topology.atom_pair_count,
      topology.bucket_offset_count,
      topology.bucket_system_count,
      topology.bucket_orbital_count,
  };
  const std::size_t topology_element_sizes[] = {
      sizeof(std::int64_t), sizeof(std::int64_t), sizeof(std::int64_t), sizeof(std::int64_t),
      sizeof(std::int64_t), sizeof(std::int64_t), sizeof(std::int64_t), sizeof(std::int64_t),
      sizeof(std::int64_t), sizeof(std::int64_t), sizeof(Gfn2AtomPair), sizeof(std::int64_t),
      sizeof(std::int32_t), sizeof(std::int32_t),
  };
  for (std::size_t index = 0u; index < std::size(topology_pointers); ++index) {
    if (topology_counts[index] != 0 &&
        pointer_range_overlaps(
            active_mask, static_cast<std::size_t>(active_mask_elements), topology_pointers[index],
            static_cast<std::size_t>(topology_counts[index]) * topology_element_sizes[index])) {
      return failure(Gfn2PlanSchemaError::kAliasedRange, Gfn2PlanSchemaField::kActiveMask);
    }
  }
  return success();
}

cudaError_t copy_diagnostic_and_synchronize(Gfn2PlanSchemaDiagnostic* device_diagnostic,
                                            Gfn2PlanSchemaDiagnostic& diagnostic,
                                            cudaStream_t stream) noexcept {
  cudaError_t status = cudaMemcpyAsync(&diagnostic, device_diagnostic, sizeof(diagnostic),
                                       cudaMemcpyDeviceToHost, stream);
  return status == cudaSuccess ? cudaStreamSynchronize(stream) : status;
}

}  // namespace

cudaError_t validate_gfn2_topology_cuda_async(const Gfn2RaggedTopologyView& topology,
                                              Gfn2PlanSchemaDiagnostic* device_diagnostic,
                                              cudaStream_t stream) noexcept {
  if (!diagnostic_range_is_valid(device_diagnostic)) {
    return cudaErrorInvalidValue;
  }
  const Gfn2PlanSchemaDiagnostic structural =
      validate_gfn2_topology_binding(topology, Gfn2PlanMemorySpace::kCudaDevice);
  if (structural.error != Gfn2PlanSchemaError::kSuccess) {
    return cudaErrorInvalidValue;
  }
  if (diagnostic_topology_alias(topology, device_diagnostic).error !=
      Gfn2PlanSchemaError::kSuccess) {
    return cudaErrorInvalidValue;
  }
  topology_validation_kernel<<<1, 1, 0, stream>>>(topology, device_diagnostic);
  return cudaGetLastError();
}

cudaError_t bind_gfn2_topology_cuda(const Gfn2RaggedTopologyView& candidate,
                                    Gfn2RaggedTopologyView& binding,
                                    Gfn2PlanSchemaDiagnostic* device_diagnostic,
                                    Gfn2PlanSchemaDiagnostic& diagnostic,
                                    cudaStream_t stream) noexcept {
  binding = {};
  diagnostic = validate_gfn2_topology_binding(candidate, Gfn2PlanMemorySpace::kCudaDevice);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return cudaSuccess;
  }
  if (!diagnostic_range_is_valid(device_diagnostic) ||
      !pointer_is_cuda_accessible(device_diagnostic)) {
    return cudaErrorInvalidValue;
  }
  diagnostic = validate_cuda_topology_pointers(candidate);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return cudaSuccess;
  }
  diagnostic = diagnostic_topology_alias(candidate, device_diagnostic);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return cudaSuccess;
  }
  cudaError_t status = validate_gfn2_topology_cuda_async(candidate, device_diagnostic, stream);
  if (status == cudaSuccess) {
    status = copy_diagnostic_and_synchronize(device_diagnostic, diagnostic, stream);
  }
  if (status == cudaSuccess && diagnostic.error == Gfn2PlanSchemaError::kSuccess) {
    binding = candidate;
  }
  return status;
}

cudaError_t validate_gfn2_geometry_provenance_cuda_async(
    const Gfn2RaggedTopologyView& topology, const Gfn2GeometryCacheProvenanceView& provenance,
    std::uint64_t expected_geometry_generation, const std::uint8_t* active_mask,
    std::int64_t active_mask_elements, Gfn2PlanSchemaDiagnostic* device_diagnostic,
    cudaStream_t stream) noexcept {
  if (!diagnostic_range_is_valid(device_diagnostic)) {
    return cudaErrorInvalidValue;
  }
  const Gfn2PlanSchemaDiagnostic structural = validate_gfn2_geometry_provenance_binding(
      topology, provenance, Gfn2PlanMemorySpace::kCudaDevice);
  if (structural.error != Gfn2PlanSchemaError::kSuccess) {
    return cudaErrorInvalidValue;
  }
  const Gfn2PlanSchemaDiagnostic active_shape =
      validate_cuda_active_mask(topology, provenance, active_mask, active_mask_elements, false);
  if (active_shape.error != Gfn2PlanSchemaError::kSuccess) {
    return cudaErrorInvalidValue;
  }
  const Gfn2PlanSchemaDiagnostic alias = diagnostic_provenance_alias(
      topology, provenance, active_mask, active_mask_elements, device_diagnostic);
  if (alias.error != Gfn2PlanSchemaError::kSuccess) {
    return cudaErrorInvalidValue;
  }
  provenance_validation_kernel<<<1, 1, 0, stream>>>(provenance, expected_geometry_generation,
                                                    active_mask, device_diagnostic);
  return cudaGetLastError();
}

cudaError_t bind_gfn2_geometry_provenance_cuda(
    const Gfn2RaggedTopologyView& topology, const Gfn2GeometryCacheProvenanceView& candidate,
    std::uint64_t expected_geometry_generation, const std::uint8_t* active_mask,
    std::int64_t active_mask_elements, Gfn2GeometryCacheProvenanceView& binding,
    Gfn2PlanSchemaDiagnostic* device_diagnostic, Gfn2PlanSchemaDiagnostic& diagnostic,
    cudaStream_t stream) noexcept {
  binding = {};
  diagnostic = validate_gfn2_geometry_provenance_binding(topology, candidate,
                                                         Gfn2PlanMemorySpace::kCudaDevice);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return cudaSuccess;
  }
  if (!diagnostic_range_is_valid(device_diagnostic) ||
      !pointer_is_cuda_accessible(device_diagnostic)) {
    return cudaErrorInvalidValue;
  }
  if (candidate.system_generation_count != 0 &&
      !pointer_is_cuda_accessible(candidate.system_geometry_generations)) {
    diagnostic = failure(Gfn2PlanSchemaError::kInvalidMemorySpace,
                         Gfn2PlanSchemaField::kSystemGeometryGenerations);
    return cudaSuccess;
  }
  diagnostic =
      validate_cuda_active_mask(topology, candidate, active_mask, active_mask_elements, true);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return cudaSuccess;
  }
  diagnostic = diagnostic_provenance_alias(topology, candidate, active_mask, active_mask_elements,
                                           device_diagnostic);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return cudaSuccess;
  }
  cudaError_t status = validate_gfn2_geometry_provenance_cuda_async(
      topology, candidate, expected_geometry_generation, active_mask, active_mask_elements,
      device_diagnostic, stream);
  if (status == cudaSuccess) {
    status = copy_diagnostic_and_synchronize(device_diagnostic, diagnostic, stream);
  }
  if (status == cudaSuccess && diagnostic.error == Gfn2PlanSchemaError::kSuccess) {
    binding = candidate;
  }
  return status;
}

namespace {

/*
 * The projection binders below derive each field with exact pointer identity
 * from an already-bound CUDA master topology and re-run the common-schema
 * binding validator (which never dereferences device arrays), so the master
 * must already have passed its own CUDA binding.  No helper beyond the
 * validated candidate is required.
 */
bool is_cuda_memory(const void* pointer) noexcept {
  if (pointer == nullptr) {
    return true;
  }
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  if (status != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged;
}

}  // namespace

cudaError_t bind_gfn2_atom_projection_cuda(const Gfn2RaggedTopologyView& device_topology,
                                           Gfn2AtomProjectionView& binding) noexcept {
  binding = {};
  Gfn2AtomProjectionView candidate{};
  candidate.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
  candidate.plan_token = device_topology.plan_token;
  candidate.batch_size = device_topology.batch_size;
  candidate.total_atoms = device_topology.total_atoms;
  std::int64_t atom_offsets = 0;
  if (host_add_one(device_topology.batch_size, atom_offsets) &&
      device_topology.atom_offset_count == atom_offsets) {
    candidate.atom_offset_count = atom_offsets;
    candidate.atom_offsets = device_topology.atom_offsets;
  }
  const Gfn2PlanSchemaDiagnostic diagnostic = validate_gfn2_atom_projection_binding(
      device_topology, candidate, Gfn2PlanMemorySpace::kCudaDevice);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return cudaSuccess;
  }
  binding = candidate;
  return cudaSuccess;
}

cudaError_t bind_gfn2_shell_ownership_projection_cuda(
    const Gfn2RaggedTopologyView& device_topology,
    Gfn2ShellOwnershipProjectionView& binding) noexcept {
  binding = {};
  Gfn2ShellOwnershipProjectionView candidate{};
  candidate.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
  candidate.plan_token = device_topology.plan_token;
  candidate.batch_size = device_topology.batch_size;
  candidate.total_atoms = device_topology.total_atoms;
  candidate.total_shells = device_topology.total_shells;
  std::int64_t batch_offsets = 0;
  std::int64_t atom_offsets = 0;
  if (host_add_one(device_topology.batch_size, batch_offsets) &&
      host_add_one(device_topology.total_atoms, atom_offsets) &&
      device_topology.batch_shell_offset_count == batch_offsets &&
      device_topology.atom_shell_offset_count == atom_offsets &&
      device_topology.shell_to_atom_count == device_topology.total_shells) {
    candidate.batch_shell_offset_count = batch_offsets;
    candidate.atom_shell_offset_count = atom_offsets;
    candidate.shell_to_atom_count = device_topology.total_shells;
    candidate.batch_shell_offsets = device_topology.batch_shell_offsets;
    candidate.atom_shell_offsets = device_topology.atom_shell_offsets;
    candidate.shell_to_atom = device_topology.shell_to_atom;
  }
  const Gfn2PlanSchemaDiagnostic diagnostic = validate_gfn2_shell_ownership_projection_binding(
      device_topology, candidate, Gfn2PlanMemorySpace::kCudaDevice);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return cudaSuccess;
  }
  binding = candidate;
  return cudaSuccess;
}

cudaError_t bind_gfn2_ao_matrix_projection_cuda(const Gfn2RaggedTopologyView& device_topology,
                                                Gfn2AOMatrixProjectionView& binding) noexcept {
  binding = {};
  Gfn2AOMatrixProjectionView candidate{};
  candidate.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
  candidate.plan_token = device_topology.plan_token;
  candidate.batch_size = device_topology.batch_size;
  candidate.total_shells = device_topology.total_shells;
  candidate.total_orbitals = device_topology.total_orbitals;
  candidate.total_matrix_elements = device_topology.total_matrix_elements;
  std::int64_t batch_offsets = 0;
  std::int64_t shell_offsets = 0;
  if (host_add_one(device_topology.batch_size, batch_offsets) &&
      host_add_one(device_topology.total_shells, shell_offsets) &&
      device_topology.batch_orbital_offset_count == batch_offsets &&
      device_topology.matrix_offset_count == batch_offsets &&
      device_topology.shell_orbital_offset_count == shell_offsets &&
      device_topology.orbital_to_shell_count == device_topology.total_orbitals &&
      device_topology.orbital_to_atom_count == device_topology.total_orbitals) {
    candidate.batch_orbital_offset_count = batch_offsets;
    candidate.matrix_offset_count = batch_offsets;
    candidate.shell_orbital_offset_count = shell_offsets;
    candidate.orbital_to_shell_count = device_topology.total_orbitals;
    candidate.orbital_to_atom_count = device_topology.total_orbitals;
    candidate.batch_orbital_offsets = device_topology.batch_orbital_offsets;
    candidate.matrix_offsets = device_topology.matrix_offsets;
    candidate.shell_orbital_offsets = device_topology.shell_orbital_offsets;
    candidate.orbital_to_shell = device_topology.orbital_to_shell;
    candidate.orbital_to_atom = device_topology.orbital_to_atom;
  }
  const Gfn2PlanSchemaDiagnostic diagnostic = validate_gfn2_ao_matrix_projection_binding(
      device_topology, candidate, Gfn2PlanMemorySpace::kCudaDevice);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return cudaSuccess;
  }
  binding = candidate;
  return cudaSuccess;
}

cudaError_t bind_gfn2_packed_all_pair_projection_cuda(
    const Gfn2RaggedTopologyView& device_topology,
    Gfn2PackedAllPairProjectionView& binding) noexcept {
  binding = {};
  Gfn2PackedAllPairProjectionView candidate{};
  candidate.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
  candidate.plan_token = device_topology.plan_token;
  candidate.batch_size = device_topology.batch_size;
  candidate.total_pairs = device_topology.total_pairs;
  candidate.pair_offset_count = device_topology.pair_offset_count;
  candidate.pair_offsets = device_topology.pair_offsets;
  const Gfn2PlanSchemaDiagnostic diagnostic = validate_gfn2_packed_all_pair_projection_binding(
      device_topology, candidate, Gfn2PlanMemorySpace::kCudaDevice);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return cudaSuccess;
  }
  binding = candidate;
  return cudaSuccess;
}

cudaError_t bind_gfn2_ao_bucket_projection_cuda(const Gfn2RaggedTopologyView& device_topology,
                                                Gfn2AOBucketProjectionView& binding) noexcept {
  binding = {};
  Gfn2AOBucketProjectionView candidate{};
  candidate.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
  candidate.plan_token = device_topology.plan_token;
  candidate.batch_size = device_topology.batch_size;
  candidate.bucket_count = device_topology.bucket_count;
  std::int64_t bucket_offsets = 0;
  if (device_topology.bucket_count == 0 ||
      host_add_one(device_topology.bucket_count, bucket_offsets)) {
    candidate.bucket_offset_count = bucket_offsets;
    candidate.bucket_system_count = device_topology.bucket_system_count;
    candidate.bucket_orbital_count = device_topology.bucket_orbital_count;
    candidate.bucket_offsets = device_topology.bucket_offsets;
    candidate.bucket_systems = device_topology.bucket_systems;
    candidate.bucket_orbital_counts = device_topology.bucket_orbital_counts;
  }
  const Gfn2PlanSchemaDiagnostic diagnostic = validate_gfn2_ao_bucket_projection_binding(
      device_topology, candidate, Gfn2PlanMemorySpace::kCudaDevice);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return cudaSuccess;
  }
  binding = candidate;
  return cudaSuccess;
}

cudaError_t bind_gfn2_element_identity_projection_cuda(
    const Gfn2ElementIdentityProjectionView& host_projection,
    const std::int32_t* device_atomic_numbers,
    Gfn2ElementIdentityProjectionView& device_binding) noexcept {
  device_binding = {};
  /* The host projector already produced a nonzero order-sensitive seal over
   * the exact atomic-number ordering.  A CUDA descriptor must carry the same
   * token, the same counts, the same seal, and a pointer proven to be CUDA
   * accessible (the setup upload owns it). */
  if (host_projection.plan_token == 0u || host_projection.element_fingerprint == 0u ||
      host_projection.total_atoms != host_projection.atomic_number_count ||
      host_projection.total_atoms < 0) {
    return cudaSuccess;
  }
  if (host_projection.atomic_number_count != 0 && !is_cuda_memory(device_atomic_numbers)) {
    return cudaSuccess;
  }
  Gfn2ElementIdentityProjectionView candidate = host_projection;
  candidate.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
  candidate.atomic_numbers = device_atomic_numbers;
  const Gfn2PlanSchemaDiagnostic diagnostic = validate_gfn2_element_identity_projection_binding(
      candidate, Gfn2PlanMemorySpace::kCudaDevice);
  if (diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
    return cudaSuccess;
  }
  device_binding = candidate;
  return cudaSuccess;
}

}  // namespace xtbloom::detail::cuda
