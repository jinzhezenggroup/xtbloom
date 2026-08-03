#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_plan_schema.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr std::int64_t kInt64Maximum = 9223372036854775807LL;

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

}  // namespace gpuxtb::detail::cuda
