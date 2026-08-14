#include <algorithm>
#include <array>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "backends/cuda/gfn2_preprocessing.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
/* Primitive candidates are unpublished and need only a valid nonzero scalar;
 * the final composer publishes the authoritative device epoch per peer. */
constexpr std::uint64_t kUnpublishedPrimitiveGeneration = 1u;

struct GeometryGenerationSource {
  std::uint64_t scalar = 0u;
  const std::uint64_t* device = nullptr;
};

using BindingDiagnostic = Gfn2PreprocessingBindingDiagnostic;
using BindingError = Gfn2PreprocessingBindingError;
using BindingField = Gfn2PreprocessingBindingField;

BindingDiagnostic binding_failure(BindingError error, BindingField field,
                                  std::int64_t index = -1) noexcept {
  BindingDiagnostic result{};
  result.error = error;
  result.field = field;
  result.index = index;
  return result;
}

bool checked_multiply(std::int64_t first, std::int64_t second, std::int64_t& product) noexcept {
  if (first < 0 || second < 0 ||
      (first != 0 && second > std::numeric_limits<std::int64_t>::max() / first)) {
    return false;
  }
  product = first * second;
  return true;
}

template <typename T>
bool canonical_pointer(const T* pointer, std::int64_t elements) noexcept {
  if (elements < 0) {
    return false;
  }
  if (elements == 0) {
    return pointer == nullptr;
  }
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignof(T) == 0u;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

template <std::size_t Capacity>
class RangeList {
 public:
  template <typename T>
  bool add(const T* pointer, std::int64_t elements) noexcept {
    if (elements < 0 || size_ == Capacity ||
        static_cast<std::uint64_t>(elements) >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(T))) {
      return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(T);
    if (bytes == 0u) {
      ranges_[size_++] = {};
      return pointer == nullptr;
    }
    if (pointer == nullptr) {
      return false;
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
      return false;
    }
    ranges_[size_++] = {begin, begin + bytes};
    return true;
  }

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] const AddressRange& operator[](std::size_t index) const noexcept {
    return ranges_[index];
  }

 private:
  std::array<AddressRange, Capacity> ranges_{};
  std::size_t size_ = 0u;
};

bool overlaps(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin < second.end && second.begin < first.end;
}

template <std::size_t ReadCapacity, std::size_t WriteCapacity>
bool writes_are_disjoint(const RangeList<ReadCapacity>& reads,
                         const RangeList<WriteCapacity>& writes) noexcept {
  for (std::size_t write = 0u; write < writes.size(); ++write) {
    for (std::size_t read = 0u; read < reads.size(); ++read) {
      if (overlaps(writes[write], reads[read])) {
        return false;
      }
    }
    for (std::size_t peer = write + 1u; peer < writes.size(); ++peer) {
      if (overlaps(writes[write], writes[peer])) {
        return false;
      }
    }
  }
  return true;
}

bool all_tokens_match(const Gfn2PreprocessingDeviceBinding& binding) noexcept {
  const std::uint64_t token = binding.plan_token;
  const bool epoch_token_matches =
      (binding.geometry_epoch.value == nullptr && binding.geometry_epoch.value_elements == 0 &&
       binding.geometry_epoch.plan_token == 0u) ||
      binding.geometry_epoch.plan_token == token;
  /* The sparse pair-list leaf is optional.  A disabled leaf contributes no
   * pointers at all and must not participate in the plan-token proof; an
   * enabled leaf requires every sparse view wired to the same plan token. */
  const bool pairlist_enabled = binding.plan.pairlist.batch_size > 0;
  bool pairlist_matches = true;
  if (pairlist_enabled) {
    pairlist_matches = binding.plan.pairlist.plan_token == token &&
                       binding.plan.pairlist.atom_offsets == binding.plan.geometry.atom_offsets &&
                       binding.workspace.pairlist_candidate.plan_token == token &&
                       binding.workspace.pairlist.plan_token == token &&
                       binding.workspace.pairlist_candidate.pair_counts != nullptr &&
                       binding.workspace.pairlist_candidate.neighbor_counts != nullptr &&
                       binding.output.pairlist.plan_token == token &&
                       binding.output.pairlist.state == Gfn2PairListState::kCommitted &&
                       binding.output.pairlist.role == Gfn2PairListRole::kCoordination &&
                       binding.diagnostics.sparse_system_errors != nullptr &&
                       binding.diagnostics.sparse_device_error != nullptr;
  } else {
    const auto& batch = binding.plan.pairlist;
    const auto& candidate = binding.workspace.pairlist_candidate;
    const auto& workspace = binding.workspace.pairlist;
    const auto& output = binding.output.pairlist;
    pairlist_matches =
        batch.batch_size == 0 && batch.total_atoms == 0 && batch.atom_offset_elements == 0 &&
        batch.cutoff == 0.0 && batch.max_cells_per_system == 0 &&
        batch.max_neighbors_per_atom == 0 && batch.max_pairs_per_system == 0 &&
        batch.mode == Gfn2PairListMode::kSparse && batch.plan_token == 0u &&
        batch.atom_offsets == nullptr && batch.flags == 0u && batch.system_modes == nullptr &&
        batch.system_mode_elements == 0 && candidate.pairs == nullptr &&
        candidate.pair_elements == 0 && candidate.pair_offsets == nullptr &&
        candidate.pair_offset_elements == 0 && candidate.pair_counts == nullptr &&
        candidate.pair_count_elements == 0 && candidate.neighbor_offsets == nullptr &&
        candidate.neighbor_offset_elements == 0 && candidate.neighbor_counts == nullptr &&
        candidate.neighbor_count_elements == 0 && candidate.neighbors == nullptr &&
        candidate.neighbor_elements == 0 && candidate.pair_generations == nullptr &&
        candidate.generation_elements == 0 && candidate.plan_token == 0u &&
        workspace.system_meta == nullptr && workspace.system_meta_elements == 0 &&
        workspace.atom_cells == nullptr && workspace.atom_cell_elements == 0 &&
        workspace.cell_counts == nullptr && workspace.cell_count_elements == 0 &&
        workspace.cell_offsets == nullptr && workspace.cell_offset_elements == 0 &&
        workspace.cell_fill == nullptr && workspace.cell_fill_elements == 0 &&
        workspace.cell_atoms == nullptr && workspace.cell_atom_elements == 0 &&
        workspace.neighbor_cursor == nullptr && workspace.neighbor_cursor_elements == 0 &&
        workspace.neighbor_scratch == nullptr && workspace.neighbor_scratch_elements == 0 &&
        workspace.pair_cursor == nullptr && workspace.pair_cursor_elements == 0 &&
        workspace.sequence_active == nullptr && workspace.sequence_elements == 0 &&
        workspace.plan_token == 0u && binding.workspace.sparse_coordination == nullptr &&
        binding.workspace.sparse_coordination_elements == 0 &&
        output.memory_space == Gfn2PlanMemorySpace::kHost &&
        output.state == Gfn2PairListState::kCommitted &&
        output.role == Gfn2PairListRole::kCoordination &&
        output.pair_map_kind == Gfn2PairMapKind::kExplicit && output.plan_token == 0u &&
        output.cutoff_bohr == 0.0 && output.list_builder_cutoff_bohr == 0.0 &&
        output.batch_size == 0 && output.total_atoms == 0 && output.max_pairs_per_system == 0 &&
        output.max_neighbors_per_atom == 0 && output.pair_offset_count == 0 &&
        output.neighbor_offset_count == 0 && output.pair_count == 0 && output.neighbor_count == 0 &&
        output.pair_offsets == nullptr && output.pairs == nullptr &&
        output.pair_count_elements == 0 && output.neighbor_count_elements == 0 &&
        output.pair_counts == nullptr && output.neighbor_counts == nullptr &&
        output.neighbor_offsets == nullptr && output.neighbors == nullptr &&
        output.committed_generation_count == 0 && output.eligible_mask_count == 0 &&
        output.active_mask_count == 0 && output.committed_generations == nullptr &&
        output.eligible_mask == nullptr && output.active_mask == nullptr &&
        binding.diagnostics.sparse_system_errors == nullptr &&
        binding.diagnostics.sparse_system_elements == 0 &&
        binding.diagnostics.sparse_device_error == nullptr;
  }
  return token != 0u && binding.plan.plan_token == token &&
         binding.plan.geometry.plan_token == token && binding.plan.integrals.plan_token == token &&
         binding.plan.h0.plan_token == token && binding.plan.es2.plan_token == token &&
         binding.plan.aes2.plan_token == token && binding.input.plan_token == token &&
         binding.activity.plan_token == token && binding.output.plan_token == token &&
         binding.output.geometry.plan_token == token && binding.output.es2.plan_token == token &&
         binding.output.aes2.plan_token == token && binding.diagnostics.plan_token == token &&
         binding.workspace.plan_token == token &&
         binding.workspace.geometry_candidate.plan_token == token &&
         binding.workspace.geometry.plan_token == token &&
         binding.workspace.integrals.plan_token == token &&
         binding.workspace.es2_candidate.plan_token == token &&
         binding.workspace.aes2_candidate.plan_token == token && pairlist_matches &&
         epoch_token_matches;
}

/* Hash the byte-stable POD projection. Dynamic requested-generation metadata
 * is normalized so callers may advance it without rebuilding any descriptor. */
std::uint64_t binding_seal(const Gfn2PreprocessingDeviceBinding& binding) noexcept {
  Gfn2PreprocessingDeviceBinding normalized{};
  std::memcpy(&normalized, &binding, sizeof(normalized));
  normalized.binding_seal = 0u;
  normalized.output.es2.geometry_generation = 0u;
  normalized.output.aes2.geometry_generation = 0u;
  normalized.workspace.es2_candidate.geometry_generation = 0u;
  normalized.workspace.aes2_candidate.geometry_generation = 0u;

  constexpr std::uint64_t kOffsetBasis = 1469598103934665603ULL;
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  std::uint64_t hash = kOffsetBasis;
  const auto* bytes = reinterpret_cast<const unsigned char*>(&normalized);
  for (std::size_t index = 0u; index < sizeof(normalized); ++index) {
    hash ^= bytes[index];
    hash *= kPrime;
  }
  return hash == 0u ? 1u : hash;
}

BindingDiagnostic validate_structure(const Gfn2PreprocessingDeviceBinding& binding,
                                     bool require_seal) noexcept {
  if (binding.plan.abi_version != kGfn2PreprocessingAbiVersion || binding.plan.reserved != 0u) {
    return binding_failure(BindingError::kInvalidAbi, BindingField::kPlan);
  }
  if (binding.plan_token == 0u) {
    return binding_failure(BindingError::kInvalidPlanToken, BindingField::kBinding);
  }
  /* Once sealed, any descriptor mutation is stale before its new value is
   * interpreted.  seal_gfn2_preprocessing_binding_cuda still runs the full
   * structural/canonical validation with require_seal=false. */
  if (require_seal && binding.binding_seal != binding_seal(binding)) {
    return binding_failure(
        binding.binding_seal == 0u ? BindingError::kUnsealedBinding : BindingError::kStaleSeal,
        BindingField::kSeal);
  }
  if (!all_tokens_match(binding)) {
    return binding_failure(BindingError::kCrossPlan, BindingField::kPlan);
  }
  const bool epoch_disabled = binding.geometry_epoch.value == nullptr &&
                              binding.geometry_epoch.value_elements == 0 &&
                              binding.geometry_epoch.plan_token == 0u;
  const bool epoch_enabled = binding.geometry_epoch.value_elements == 1 &&
                             binding.geometry_epoch.plan_token == binding.plan_token &&
                             canonical_pointer(binding.geometry_epoch.value, 1);
  if (!epoch_disabled && !epoch_enabled) {
    return binding_failure(BindingError::kInvalidEpoch, BindingField::kEpoch);
  }
  const bool admission_disabled = binding.admission.error == nullptr &&
                                  binding.admission.error_elements == 0 &&
                                  binding.admission.plan_token == 0u;
  const bool admission_enabled = binding.admission.error_elements == 1 &&
                                 binding.admission.plan_token == binding.plan_token &&
                                 canonical_pointer(binding.admission.error, 1);
  if (!admission_disabled && !admission_enabled) {
    return binding_failure(BindingError::kInvalidExtent, BindingField::kActivity);
  }

  const Gfn2GeometryDeviceBatch& geometry = binding.plan.geometry;
  const Gfn2IntegralDeviceBatch& integrals = binding.plan.integrals;
  const Gfn2H0DevicePlan& h0 = binding.plan.h0;
  const Gfn2ES2DeviceBatch& es2 = binding.plan.es2;
  const Gfn2AES2DeviceBatch& aes2 = binding.plan.aes2;
  const std::int64_t batch = geometry.batch_size;
  const std::int64_t atoms = geometry.total_atoms;
  const std::int64_t pairs = geometry.total_pairs;
  const std::int64_t shells = integrals.total_shells;
  const std::int64_t matrices = integrals.total_matrix_elements;
  const std::int64_t shell_matrices = es2.total_matrix_elements;
  std::int64_t coordinates = 0;
  std::int64_t geometry_pair_elements = 0;
  std::int64_t dipole_elements = 0;
  std::int64_t quadrupole_elements = 0;
  std::int64_t aes2_pair_elements = 0;
  std::int64_t shell_grid_per_system = 0;
  std::int64_t shell_grid_blocks = 0;
  if (batch <= 0 || atoms <= 0 || pairs < 0 || shells <= 0 || matrices <= 0 ||
      shell_matrices <= 0 || integrals.total_orbitals <= 0 || integrals.total_primitives <= 0 ||
      integrals.total_shell_pair_elements <= 0 || integrals.maximum_system_shells <= 0 ||
      !(integrals.integral_cutoff > 0.0) || !std::isfinite(integrals.integral_cutoff) ||
      atoms == std::numeric_limits<std::int64_t>::max() ||
      shells == std::numeric_limits<std::int64_t>::max() ||
      batch > static_cast<std::int64_t>(std::numeric_limits<unsigned int>::max()) ||
      batch > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      !checked_multiply(integrals.maximum_system_shells, integrals.maximum_system_shells,
                        shell_grid_per_system) ||
      !checked_multiply(shell_grid_per_system, batch, shell_grid_blocks) ||
      shell_grid_blocks > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      !checked_multiply(atoms, 3, coordinates) ||
      !checked_multiply(pairs, kGfn2GeometryPairDataElements, geometry_pair_elements) ||
      !checked_multiply(matrices, kGfn2IntegralDipoleComponents, dipole_elements) ||
      !checked_multiply(matrices, kGfn2IntegralQuadrupoleComponents, quadrupole_elements) ||
      !checked_multiply(pairs, kGfn2AES2PairDataElements, aes2_pair_elements)) {
    return binding_failure(BindingError::kInvalidExtent, BindingField::kPlan);
  }

  const bool compatible_extents =
      geometry.atom_offset_elements == batch + 1 && geometry.pair_offset_elements == batch + 1 &&
      geometry.covalent_radius_elements == atoms && geometry.coordinate_elements == coordinates &&
      integrals.batch_size == batch && integrals.total_atoms == atoms &&
      integrals.atom_offset_count == batch + 1 && integrals.batch_shell_offset_count == batch + 1 &&
      integrals.batch_orbital_offset_count == batch + 1 &&
      integrals.matrix_offset_count == batch + 1 &&
      integrals.shell_pair_offset_count == batch + 1 &&
      integrals.atom_shell_offset_count == atoms + 1 &&
      integrals.shell_orbital_offset_count == shells + 1 &&
      integrals.shell_primitive_offset_count == shells + 1 &&
      integrals.shell_to_atom_count == shells && integrals.angular_momentum_count == shells &&
      integrals.primitive_exponent_count == integrals.total_primitives &&
      integrals.primitive_coefficient_count == integrals.total_primitives &&
      h0.atomic_radius_count == atoms && h0.shell_level_count == shells &&
      h0.shell_coordination_scale_count == shells && h0.shell_polynomial_count == shells &&
      h0.shell_pair_scale_count == integrals.total_shell_pair_elements && es2.batch_size == batch &&
      es2.total_atoms == atoms && es2.total_shells == shells &&
      es2.atom_offset_count == batch + 1 && es2.batch_shell_offset_count == batch + 1 &&
      es2.atom_shell_offset_count == atoms + 1 && es2.matrix_offset_count == batch + 1 &&
      es2.shell_to_atom_count == shells && es2.shell_hardness_count == shells &&
      aes2.batch_size == batch && aes2.total_atoms == atoms && aes2.total_pairs == pairs &&
      aes2.atom_offset_count == batch + 1 && aes2.pair_offset_count == batch + 1 &&
      aes2.dipole_kernel_count == atoms && aes2.quadrupole_kernel_count == atoms &&
      aes2.multipole_radius_count == atoms && aes2.multipole_valence_cn_count == atoms;
  if (!compatible_extents) {
    return binding_failure(BindingError::kInvalidExtent, BindingField::kPlan);
  }

  const bool canonical_topology = geometry.atom_offsets == integrals.atom_offsets &&
                                  geometry.atom_offsets == es2.atom_offsets &&
                                  geometry.atom_offsets == aes2.atom_offsets &&
                                  geometry.pair_offsets == aes2.pair_offsets &&
                                  integrals.batch_shell_offsets == es2.batch_shell_offsets &&
                                  integrals.atom_shell_offsets == es2.atom_shell_offsets &&
                                  integrals.shell_to_atom == es2.shell_to_atom;
  if (!canonical_topology) {
    return binding_failure(BindingError::kCrossPlan, BindingField::kPlan);
  }

  const bool plan_pointers =
      canonical_pointer(geometry.atom_offsets, batch + 1) &&
      canonical_pointer(geometry.pair_offsets, batch + 1) &&
      canonical_pointer(geometry.covalent_radii, atoms) &&
      canonical_pointer(integrals.batch_shell_offsets, batch + 1) &&
      canonical_pointer(integrals.batch_orbital_offsets, batch + 1) &&
      canonical_pointer(integrals.matrix_offsets, batch + 1) &&
      canonical_pointer(integrals.shell_pair_offsets, batch + 1) &&
      canonical_pointer(integrals.atom_shell_offsets, atoms + 1) &&
      canonical_pointer(integrals.shell_orbital_offsets, shells + 1) &&
      canonical_pointer(integrals.shell_primitive_offsets, shells + 1) &&
      canonical_pointer(integrals.shell_to_atom, shells) &&
      canonical_pointer(integrals.angular_momenta, shells) &&
      canonical_pointer(integrals.primitive_exponents, integrals.total_primitives) &&
      canonical_pointer(integrals.primitive_coefficients, integrals.total_primitives) &&
      canonical_pointer(h0.atomic_radii, atoms) && canonical_pointer(h0.shell_levels, shells) &&
      canonical_pointer(h0.shell_coordination_scale, shells) &&
      canonical_pointer(h0.shell_polynomial, shells) &&
      canonical_pointer(h0.shell_pair_scale, integrals.total_shell_pair_elements) &&
      canonical_pointer(es2.matrix_offsets, batch + 1) &&
      canonical_pointer(es2.shell_hardness, shells) &&
      canonical_pointer(aes2.dipole_kernel, atoms) &&
      canonical_pointer(aes2.quadrupole_kernel, atoms) &&
      canonical_pointer(aes2.multipole_radius, atoms) &&
      canonical_pointer(aes2.multipole_valence_cn, atoms);
  if (!plan_pointers) {
    return binding_failure(BindingError::kInvalidPointer, BindingField::kPlan);
  }

  if (binding.input.position_elements != coordinates ||
      !canonical_pointer(binding.input.positions, coordinates)) {
    return binding_failure(BindingError::kInvalidPointer, BindingField::kPositions);
  }
  if (binding.activity.requested_elements != batch ||
      binding.activity.published_elements != batch ||
      !canonical_pointer(binding.activity.requested_mask, batch) ||
      !canonical_pointer(binding.activity.published_mask, batch)) {
    return binding_failure(BindingError::kInvalidActivity, BindingField::kActivity);
  }

  const Gfn2PreprocessingDeviceOutput& output = binding.output;
  const bool output_valid =
      output.geometry.pair_data_elements == geometry_pair_elements &&
      output.geometry.coordination_elements == atoms &&
      output.geometry.generation_elements == batch &&
      canonical_pointer(output.geometry.pair_data, geometry_pair_elements) &&
      canonical_pointer(output.geometry.coordination_numbers, atoms) &&
      canonical_pointer(output.geometry.geometry_generations, batch) &&
      output.overlap_elements == matrices && canonical_pointer(output.overlap, matrices) &&
      output.dipole_elements == dipole_elements &&
      canonical_pointer(output.dipole_integrals, dipole_elements) &&
      output.quadrupole_elements == quadrupole_elements &&
      canonical_pointer(output.quadrupole_integrals, quadrupole_elements) &&
      output.h0_elements == matrices && canonical_pointer(output.h0, matrices) &&
      output.es2.matrix_elements == shell_matrices &&
      canonical_pointer(output.es2.coulomb_matrix, shell_matrices) &&
      output.aes2.pair_data_elements == aes2_pair_elements &&
      canonical_pointer(output.aes2.pair_data, aes2_pair_elements) &&
      output.generation_elements == batch && canonical_pointer(output.operator_generations, batch);
  if (!output_valid) {
    return binding_failure(BindingError::kInvalidExtent, BindingField::kOutput);
  }

  const Gfn2PreprocessingDeviceDiagnostics& diagnostics = binding.diagnostics;
  const bool diagnostics_valid = diagnostics.geometry_system_elements == batch &&
                                 canonical_pointer(diagnostics.geometry_system_errors, batch) &&
                                 canonical_pointer(diagnostics.geometry_device_error, 1) &&
                                 diagnostics.integral_system_elements == batch &&
                                 canonical_pointer(diagnostics.integral_system_errors, batch) &&
                                 canonical_pointer(diagnostics.integral_device_error, 1) &&
                                 canonical_pointer(diagnostics.es2_device_error, 1) &&
                                 diagnostics.aes2_system_elements == batch &&
                                 canonical_pointer(diagnostics.aes2_system_errors, batch) &&
                                 canonical_pointer(diagnostics.aes2_device_error, 1) &&
                                 diagnostics.system_stage_elements == batch &&
                                 canonical_pointer(diagnostics.system_stages, batch) &&
                                 canonical_pointer(diagnostics.plan_error, 1);
  if (!diagnostics_valid) {
    return binding_failure(BindingError::kInvalidDiagnostics, BindingField::kDiagnostics);
  }

  const Gfn2PreprocessingDeviceWorkspace& workspace = binding.workspace;
  const bool workspace_valid =
      workspace.position_elements == coordinates &&
      canonical_pointer(workspace.positions_scratch, coordinates) &&
      workspace.geometry_candidate.pair_data_elements == geometry_pair_elements &&
      workspace.geometry_candidate.coordination_elements == atoms &&
      workspace.geometry_candidate.generation_elements == batch &&
      canonical_pointer(workspace.geometry_candidate.pair_data, geometry_pair_elements) &&
      canonical_pointer(workspace.geometry_candidate.coordination_numbers, atoms) &&
      canonical_pointer(workspace.geometry_candidate.geometry_generations, batch) &&
      workspace.geometry.pair_elements == geometry_pair_elements &&
      workspace.geometry.coordination_elements == atoms &&
      workspace.geometry.gradient_elements == 0 && workspace.geometry.gradient_scratch == nullptr &&
      workspace.geometry.sequence_elements == 1 &&
      canonical_pointer(workspace.geometry.pair_scratch, geometry_pair_elements) &&
      canonical_pointer(workspace.geometry.coordination_scratch, atoms) &&
      canonical_pointer(workspace.geometry.sequence_active, 1) &&
      workspace.overlap_elements == matrices &&
      canonical_pointer(workspace.overlap_candidate, matrices) &&
      workspace.dipole_elements == dipole_elements &&
      canonical_pointer(workspace.dipole_candidate, dipole_elements) &&
      workspace.quadrupole_elements == quadrupole_elements &&
      canonical_pointer(workspace.quadrupole_candidate, quadrupole_elements) &&
      workspace.h0_elements == matrices && canonical_pointer(workspace.h0_candidate, matrices) &&
      workspace.integrals.overlap_elements == matrices &&
      canonical_pointer(workspace.integrals.overlap_scratch, matrices) &&
      workspace.integrals.dipole_elements == dipole_elements &&
      canonical_pointer(workspace.integrals.dipole_scratch, dipole_elements) &&
      workspace.integrals.quadrupole_elements == quadrupole_elements &&
      canonical_pointer(workspace.integrals.quadrupole_scratch, quadrupole_elements) &&
      workspace.integrals.h0_elements == matrices &&
      canonical_pointer(workspace.integrals.h0_scratch, matrices) &&
      workspace.integrals.sequence_elements == 1 &&
      canonical_pointer(workspace.integrals.sequence_active, 1) &&
      workspace.es2_candidate.matrix_elements == shell_matrices &&
      canonical_pointer(workspace.es2_candidate.coulomb_matrix, shell_matrices) &&
      workspace.es2.matrix_elements == shell_matrices &&
      canonical_pointer(workspace.es2.matrix_scratch, shell_matrices) &&
      workspace.es2.shell_elements == 0 && workspace.es2.shell_scratch == nullptr &&
      workspace.es2.batch_elements == 0 && workspace.es2.batch_scratch == nullptr &&
      workspace.es2.gradient_elements == 0 && workspace.es2.gradient_scratch == nullptr &&
      workspace.aes2_candidate.pair_data_elements == aes2_pair_elements &&
      canonical_pointer(workspace.aes2_candidate.pair_data, aes2_pair_elements) &&
      workspace.aes2.pair_elements == aes2_pair_elements &&
      canonical_pointer(workspace.aes2.pair_scratch, aes2_pair_elements) &&
      workspace.aes2.potential_elements == 0 && workspace.aes2.potential_scratch == nullptr &&
      workspace.aes2.batch_elements == 0 && workspace.aes2.batch_scratch == nullptr &&
      workspace.aes2.gradient_elements == 0 && workspace.aes2.gradient_scratch == nullptr &&
      workspace.aes2.coordination_elements == 0 && workspace.aes2.coordination_scratch == nullptr &&
      workspace.aes2.scc_peer_error_elements == 0 &&
      workspace.aes2.scc_peer_error_scratch == nullptr;
  if (!workspace_valid) {
    return binding_failure(BindingError::kInvalidWorkspace, BindingField::kWorkspace);
  }

  /* Optional sparse pair-list leaf.  A disabled leaf carries no pointers; an
   * enabled leaf must describe exactly the canonical geometry partition with
   * per-system capacities and fully wired candidate/workspace/output domains.
   * The host scheduler chooses the batch mode with gfn2_pairlist_use_sparse_for
   * and provisions the fixed-topology capacities up front. */
  const Gfn2PairListDeviceBatch& pairlist = binding.plan.pairlist;
  const bool pairlist_enabled = pairlist.batch_size > 0;
  if (pairlist_enabled) {
    std::int64_t pairlist_neighbors_capacity = 0;
    std::int64_t pairlist_cells_capacity = 0;
    std::int64_t pairlist_pairs_capacity = 0;
    const bool pairlist_capacity_products =
        checked_multiply(atoms, pairlist.max_neighbors_per_atom, pairlist_neighbors_capacity) &&
        checked_multiply(batch, pairlist.max_cells_per_system, pairlist_cells_capacity) &&
        checked_multiply(batch, pairlist.max_pairs_per_system, pairlist_pairs_capacity) &&
        pairlist.max_cells_per_system < std::numeric_limits<std::int64_t>::max() &&
        checked_multiply(batch, pairlist.max_cells_per_system + 1, pairlist_cells_capacity);
    std::int64_t pairlist_scratch = 0;
    const bool pairlist_extents =
        pairlist_capacity_products && pairlist.plan_token == binding.plan_token &&
        pairlist.batch_size == batch && pairlist.total_atoms == atoms &&
        pairlist.atom_offset_elements == batch + 1 &&
        pairlist.atom_offsets == geometry.atom_offsets && std::isfinite(pairlist.cutoff) &&
        pairlist.cutoff >= kDefaultPairlistCutoffBohr && pairlist.max_cells_per_system > 0 &&
        pairlist.max_neighbors_per_atom > 0 && pairlist.max_pairs_per_system > 0 &&
        (pairlist.mode == Gfn2PairListMode::kSparse || pairlist.mode == Gfn2PairListMode::kDense) &&
        (pairlist.flags & ~kGfn2PairListAllowDenseFallback) == 0u &&
        pairlist.system_modes != nullptr && pairlist.system_mode_elements == batch &&
        canonical_pointer(pairlist.system_modes, batch) &&
        checked_multiply(atoms, pairlist.max_neighbors_per_atom, pairlist_scratch) &&
        workspace.sparse_coordination_elements == atoms &&
        canonical_pointer(workspace.sparse_coordination, atoms) &&
        workspace.pairlist_candidate.pair_elements >= pairlist_pairs_capacity &&
        workspace.pairlist_candidate.pair_offset_elements == batch + 1 &&
        workspace.pairlist_candidate.pair_count_elements >= batch &&
        workspace.pairlist_candidate.neighbor_offset_elements == atoms + 1 &&
        workspace.pairlist_candidate.neighbor_count_elements >= atoms &&
        workspace.pairlist_candidate.neighbor_elements >= pairlist_neighbors_capacity &&
        workspace.pairlist_candidate.generation_elements >= batch &&
        workspace.pairlist_candidate.plan_token == binding.plan_token &&
        canonical_pointer(workspace.pairlist_candidate.pairs,
                          workspace.pairlist_candidate.pair_elements) &&
        canonical_pointer(workspace.pairlist_candidate.pair_offsets,
                          workspace.pairlist_candidate.pair_offset_elements) &&
        canonical_pointer(workspace.pairlist_candidate.pair_counts,
                          workspace.pairlist_candidate.pair_count_elements) &&
        canonical_pointer(workspace.pairlist_candidate.neighbor_offsets,
                          workspace.pairlist_candidate.neighbor_offset_elements) &&
        canonical_pointer(workspace.pairlist_candidate.neighbor_counts,
                          workspace.pairlist_candidate.neighbor_count_elements) &&
        canonical_pointer(workspace.pairlist_candidate.neighbors,
                          workspace.pairlist_candidate.neighbor_elements) &&
        canonical_pointer(workspace.pairlist_candidate.pair_generations,
                          workspace.pairlist_candidate.generation_elements) &&
        workspace.pairlist.system_meta_elements >= batch &&
        workspace.pairlist.atom_cell_elements >= atoms &&
        workspace.pairlist.cell_count_elements >= pairlist_cells_capacity &&
        workspace.pairlist.cell_offset_elements >= pairlist_cells_capacity &&
        workspace.pairlist.cell_fill_elements >= pairlist_cells_capacity &&
        workspace.pairlist.cell_atom_elements >= atoms &&
        workspace.pairlist.neighbor_cursor_elements >= atoms &&
        workspace.pairlist.neighbor_scratch_elements >= pairlist_neighbors_capacity &&
        workspace.pairlist.pair_cursor_elements >= batch &&
        workspace.pairlist.sequence_elements >= 1 &&
        workspace.pairlist.plan_token == binding.plan_token &&
        canonical_pointer(workspace.pairlist.system_meta, batch) &&
        canonical_pointer(workspace.pairlist.atom_cells, atoms) &&
        canonical_pointer(workspace.pairlist.cell_counts, pairlist_cells_capacity) &&
        canonical_pointer(workspace.pairlist.cell_offsets, pairlist_cells_capacity) &&
        canonical_pointer(workspace.pairlist.cell_fill, pairlist_cells_capacity) &&
        canonical_pointer(workspace.pairlist.cell_atoms, workspace.pairlist.cell_atom_elements) &&
        canonical_pointer(workspace.pairlist.neighbor_cursor, atoms) &&
        canonical_pointer(workspace.pairlist.neighbor_scratch, pairlist_neighbors_capacity) &&
        canonical_pointer(workspace.pairlist.pair_cursor, batch) &&
        canonical_pointer(workspace.pairlist.sequence_active, 1) &&
        diagnostics.sparse_system_elements == batch &&
        canonical_pointer(diagnostics.sparse_system_errors, batch) &&
        canonical_pointer(diagnostics.sparse_device_error, 1);
    if (!pairlist_extents) {
      return binding_failure(BindingError::kInvalidWorkspace, BindingField::kPairlist);
    }
    /* Committed output view (ABI step 4): the final per-system gate publishes
     * the candidate list into fixed-stride per-system slices with explicit
     * counts, per-system committed generations, and an eligibility mask, so a
     * failed peer is ineligible and never shifts a later peer's slice. */
    const Gfn2PairListConsumerView& committed = binding.output.pairlist;
    std::int64_t committed_pairs_capacity = 0;
    std::int64_t committed_neighbors_capacity = 0;
    const bool committed_extents =
        committed.memory_space == Gfn2PlanMemorySpace::kCudaDevice &&
        committed.state == Gfn2PairListState::kCommitted &&
        committed.role == Gfn2PairListRole::kCoordination &&
        committed.pair_map_kind == Gfn2PairMapKind::kExplicit &&
        committed.plan_token == binding.plan_token && committed.batch_size == batch &&
        committed.total_atoms == atoms &&
        committed.max_pairs_per_system == pairlist.max_pairs_per_system &&
        committed.max_neighbors_per_atom == pairlist.max_neighbors_per_atom &&
        committed.cutoff_bohr == kDefaultPairlistCutoffBohr &&
        committed.list_builder_cutoff_bohr == pairlist.cutoff &&
        committed.pair_offset_count == batch + 1 && committed.neighbor_offset_count == atoms + 1 &&
        committed.pair_count_elements == batch && committed.neighbor_count_elements == atoms &&
        committed.pair_count == pairlist_pairs_capacity &&
        committed.neighbor_count == pairlist_neighbors_capacity &&
        committed.committed_generation_count == batch && committed.eligible_mask_count == batch &&
        committed.active_mask_count == 0 &&
        checked_multiply(batch, pairlist.max_pairs_per_system, committed_pairs_capacity) &&
        checked_multiply(atoms, pairlist.max_neighbors_per_atom, committed_neighbors_capacity) &&
        committed.pair_count == committed_pairs_capacity &&
        committed.neighbor_count == committed_neighbors_capacity &&
        canonical_pointer(committed.pair_offsets, committed.pair_offset_count) &&
        canonical_pointer(committed.pairs, committed.pair_count) &&
        canonical_pointer(committed.pair_counts, committed.pair_count_elements) &&
        canonical_pointer(committed.neighbor_offsets, committed.neighbor_offset_count) &&
        canonical_pointer(committed.neighbor_counts, committed.neighbor_count_elements) &&
        canonical_pointer(committed.neighbors, committed.neighbor_count) &&
        canonical_pointer(committed.committed_generations, committed.committed_generation_count) &&
        canonical_pointer(committed.eligible_mask, committed.eligible_mask_count) &&
        committed.active_mask == nullptr;
    if (!committed_extents) {
      return binding_failure(BindingError::kInvalidExtent, BindingField::kOutput);
    }
  }

  RangeList<41> reads;
  RangeList<80> writes;
  const bool ranges_valid =
      reads.add(geometry.atom_offsets, batch + 1) && reads.add(geometry.pair_offsets, batch + 1) &&
      reads.add(geometry.covalent_radii, atoms) &&
      reads.add(integrals.batch_shell_offsets, batch + 1) &&
      reads.add(integrals.batch_orbital_offsets, batch + 1) &&
      reads.add(integrals.matrix_offsets, batch + 1) &&
      reads.add(integrals.shell_pair_offsets, batch + 1) &&
      reads.add(integrals.atom_shell_offsets, atoms + 1) &&
      reads.add(integrals.shell_orbital_offsets, shells + 1) &&
      reads.add(integrals.shell_primitive_offsets, shells + 1) &&
      reads.add(integrals.shell_to_atom, shells) && reads.add(integrals.angular_momenta, shells) &&
      reads.add(integrals.primitive_exponents, integrals.total_primitives) &&
      reads.add(integrals.primitive_coefficients, integrals.total_primitives) &&
      reads.add(h0.atomic_radii, atoms) && reads.add(h0.shell_levels, shells) &&
      reads.add(h0.shell_coordination_scale, shells) && reads.add(h0.shell_polynomial, shells) &&
      reads.add(h0.shell_pair_scale, integrals.total_shell_pair_elements) &&
      reads.add(es2.matrix_offsets, batch + 1) && reads.add(es2.shell_hardness, shells) &&
      reads.add(aes2.dipole_kernel, atoms) && reads.add(aes2.quadrupole_kernel, atoms) &&
      reads.add(aes2.multipole_radius, atoms) && reads.add(aes2.multipole_valence_cn, atoms) &&
      reads.add(binding.input.positions, coordinates) &&
      reads.add(binding.admission.error, binding.admission.error_elements) &&
      reads.add(binding.activity.requested_mask, batch) &&
      (!pairlist_enabled ||
       reads.add(binding.plan.pairlist.system_modes, binding.plan.pairlist.system_mode_elements)) &&
      writes.add(binding.activity.published_mask, batch) &&
      writes.add(output.geometry.pair_data, geometry_pair_elements) &&
      writes.add(output.geometry.coordination_numbers, atoms) &&
      writes.add(output.geometry.geometry_generations, batch) &&
      writes.add(output.overlap, matrices) &&
      writes.add(output.dipole_integrals, dipole_elements) &&
      writes.add(output.quadrupole_integrals, quadrupole_elements) &&
      writes.add(output.h0, matrices) && writes.add(output.es2.coulomb_matrix, shell_matrices) &&
      writes.add(output.aes2.pair_data, aes2_pair_elements) &&
      writes.add(output.operator_generations, batch) &&
      writes.add(workspace.positions_scratch, coordinates) &&
      writes.add(workspace.geometry_candidate.pair_data, geometry_pair_elements) &&
      writes.add(workspace.geometry_candidate.coordination_numbers, atoms) &&
      writes.add(workspace.geometry_candidate.geometry_generations, batch) &&
      writes.add(workspace.geometry.pair_scratch, geometry_pair_elements) &&
      writes.add(workspace.geometry.coordination_scratch, atoms) &&
      writes.add(workspace.geometry.sequence_active, 1) &&
      writes.add(workspace.overlap_candidate, matrices) &&
      writes.add(workspace.dipole_candidate, dipole_elements) &&
      writes.add(workspace.quadrupole_candidate, quadrupole_elements) &&
      writes.add(workspace.h0_candidate, matrices) &&
      writes.add(workspace.integrals.overlap_scratch, matrices) &&
      writes.add(workspace.integrals.dipole_scratch, dipole_elements) &&
      writes.add(workspace.integrals.quadrupole_scratch, quadrupole_elements) &&
      writes.add(workspace.integrals.h0_scratch, matrices) &&
      writes.add(workspace.integrals.sequence_active, 1) &&
      writes.add(workspace.es2_candidate.coulomb_matrix, shell_matrices) &&
      writes.add(workspace.es2.matrix_scratch, shell_matrices) &&
      writes.add(workspace.aes2_candidate.pair_data, aes2_pair_elements) &&
      writes.add(workspace.aes2.pair_scratch, aes2_pair_elements) &&
      writes.add(diagnostics.geometry_system_errors, batch) &&
      writes.add(diagnostics.geometry_device_error, 1) &&
      writes.add(diagnostics.integral_system_errors, batch) &&
      writes.add(diagnostics.integral_device_error, 1) &&
      writes.add(diagnostics.es2_device_error, 1) &&
      writes.add(diagnostics.aes2_system_errors, batch) &&
      writes.add(diagnostics.aes2_device_error, 1) &&
      writes.add(diagnostics.system_stages, batch) && writes.add(diagnostics.plan_error, 1);
  const bool sparse_ranges_valid =
      !pairlist_enabled ||
      (writes.add(workspace.sparse_coordination, workspace.sparse_coordination_elements) &&
       writes.add(workspace.pairlist_candidate.pairs, workspace.pairlist_candidate.pair_elements) &&
       writes.add(workspace.pairlist_candidate.pair_offsets,
                  workspace.pairlist_candidate.pair_offset_elements) &&
       writes.add(workspace.pairlist_candidate.pair_counts,
                  workspace.pairlist_candidate.pair_count_elements) &&
       writes.add(workspace.pairlist_candidate.neighbor_offsets,
                  workspace.pairlist_candidate.neighbor_offset_elements) &&
       writes.add(workspace.pairlist_candidate.neighbor_counts,
                  workspace.pairlist_candidate.neighbor_count_elements) &&
       writes.add(workspace.pairlist_candidate.neighbors,
                  workspace.pairlist_candidate.neighbor_elements) &&
       writes.add(workspace.pairlist_candidate.pair_generations,
                  workspace.pairlist_candidate.generation_elements) &&
       writes.add(workspace.pairlist.system_meta, workspace.pairlist.system_meta_elements) &&
       writes.add(workspace.pairlist.atom_cells, workspace.pairlist.atom_cell_elements) &&
       writes.add(workspace.pairlist.cell_counts, workspace.pairlist.cell_count_elements) &&
       writes.add(workspace.pairlist.cell_offsets, workspace.pairlist.cell_offset_elements) &&
       writes.add(workspace.pairlist.cell_fill, workspace.pairlist.cell_fill_elements) &&
       writes.add(workspace.pairlist.cell_atoms, workspace.pairlist.cell_atom_elements) &&
       writes.add(workspace.pairlist.neighbor_cursor,
                  workspace.pairlist.neighbor_cursor_elements) &&
       writes.add(workspace.pairlist.neighbor_scratch,
                  workspace.pairlist.neighbor_scratch_elements) &&
       writes.add(workspace.pairlist.pair_cursor, workspace.pairlist.pair_cursor_elements) &&
       writes.add(workspace.pairlist.sequence_active, workspace.pairlist.sequence_elements) &&
       writes.add(output.pairlist.pairs, output.pairlist.pair_count) &&
       writes.add(output.pairlist.pair_offsets, output.pairlist.pair_offset_count) &&
       writes.add(output.pairlist.pair_counts, output.pairlist.pair_count_elements) &&
       writes.add(output.pairlist.neighbor_offsets, output.pairlist.neighbor_offset_count) &&
       writes.add(output.pairlist.neighbor_counts, output.pairlist.neighbor_count_elements) &&
       writes.add(output.pairlist.neighbors, output.pairlist.neighbor_count) &&
       writes.add(output.pairlist.committed_generations,
                  output.pairlist.committed_generation_count) &&
       writes.add(output.pairlist.eligible_mask, output.pairlist.eligible_mask_count) &&
       writes.add(diagnostics.sparse_system_errors, diagnostics.sparse_system_elements) &&
       writes.add(diagnostics.sparse_device_error, 1));
  const bool epoch_range_valid =
      epoch_disabled ||
      writes.add(binding.geometry_epoch.value, binding.geometry_epoch.value_elements);
  if (!ranges_valid || !sparse_ranges_valid || !epoch_range_valid) {
    return binding_failure(BindingError::kInvalidExtent, BindingField::kWorkspace);
  }
  if (!writes_are_disjoint(reads, writes)) {
    return binding_failure(BindingError::kInvalidAlias, BindingField::kWorkspace);
  }

  return {};
}

__device__ std::uint32_t read_u32(const std::uint32_t* value) {
  return atomicAdd(const_cast<std::uint32_t*>(value), 0u);
}

__device__ void record_plan_error(std::uint32_t* plan_error, Gfn2PreprocessingDeviceError error) {
  atomicCAS(plan_error, static_cast<std::uint32_t>(Gfn2PreprocessingDeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ std::uint64_t load_geometry_generation(GeometryGenerationSource source) {
  return source.device == nullptr
             ? source.scalar
             : atomicAdd(
                   reinterpret_cast<unsigned long long*>(const_cast<std::uint64_t*>(source.device)),
                   0ULL);
}

/* The CAS loop prevents wraparound even if a caller violates the documented
 * single-flight contract. Concurrent use still has no inference-level
 * ordering guarantee, but it cannot publish a duplicate or zero epoch. */
__global__ void advance_geometry_epoch_kernel(Gfn2GeometryEpochDevice epoch,
                                              Gfn2DeviceAdmission admission,
                                              std::uint32_t* plan_error) {
  if (threadIdx.x != 0 || blockIdx.x != 0 || !gfn2_request_admitted(admission) ||
      read_u32(plan_error) != 0u) {
    return;
  }
  auto* const value = reinterpret_cast<unsigned long long*>(epoch.value);
  unsigned long long observed = atomicAdd(value, 0ULL);
  while (observed != ~0ULL) {
    const unsigned long long previous = atomicCAS(value, observed, observed + 1ULL);
    if (previous == observed) {
      return;
    }
    observed = previous;
  }
  record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kGeometryEpochOverflow);
}

__device__ void store_safe_positions(std::int64_t atom_begin, std::int64_t atom,
                                     double* positions) {
  const double local = static_cast<double>(atom - atom_begin);
  positions[atom * 3] = 2.0 * local;
  positions[atom * 3 + 1] = 0.125 * local;
  positions[atom * 3 + 2] = -0.0625 * local;
}

__global__ void prepare_positions_kernel(Gfn2GeometryDeviceBatch batch, const double* input,
                                         const std::uint8_t* requested, double* scratch,
                                         std::uint32_t* plan_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  const std::uint8_t active = requested[system];
  if (begin < 0 || begin > end || end > batch.total_atoms || (system == 0 && begin != 0) ||
      (system + 1 == batch.batch_size && end != batch.total_atoms)) {
    if (threadIdx.x == 0) {
      record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kInvalidComposerOffsets);
    }
    return;
  }
  if (active > 1u) {
    if (threadIdx.x == 0) {
      record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kInvalidActivity);
    }
  }
  for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
    if (active == 1u) {
      scratch[atom * 3] = input[atom * 3];
      scratch[atom * 3 + 1] = input[atom * 3 + 1];
      scratch[atom * 3 + 2] = input[atom * 3 + 2];
    } else {
      store_safe_positions(begin, atom, scratch);
    }
  }
}

__global__ void gate_composer_plan_kernel(
    std::int64_t batch_size, const std::uint32_t* plan_error, std::uint32_t* geometry_system_errors,
    std::uint32_t* geometry_device_error, std::uint32_t* integral_system_errors,
    std::uint32_t* integral_device_error, std::uint32_t* es2_device_error,
    std::uint32_t* aes2_system_errors, std::uint32_t* aes2_device_error) {
  if (read_u32(plan_error) == 0u) {
    return;
  }
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system < batch_size) {
    geometry_system_errors[system] =
        static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kInvalidOffsets);
    integral_system_errors[system] =
        static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kInvalidOffsets);
    aes2_system_errors[system] = static_cast<std::uint32_t>(Gfn2AES2DeviceError::kInvalidOffsets);
  }
  if (system == 0) {
    *geometry_device_error = static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kInvalidOffsets);
    *integral_device_error = static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kInvalidOffsets);
    *es2_device_error = static_cast<std::uint32_t>(Gfn2ES2DeviceError::kInvalidOffsets);
    *aes2_device_error = static_cast<std::uint32_t>(Gfn2AES2DeviceError::kInvalidOffsets);
  }
}

__global__ void gate_h0_kernel(std::int64_t batch_size, const std::uint8_t* requested,
                               const std::uint32_t* geometry_errors,
                               std::uint32_t* integral_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system < batch_size && requested[system] == 1u && geometry_errors[system] != 0u) {
    atomicCAS(integral_errors + system, 0u,
              static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kInvalidCoordination));
  }
}

/* evaluate_gfn2_h0_cuda snapshots the same sticky primitive-domain scalar as
 * evaluate_gfn2_integrals_cuda. A peer numerical error from S/D/Q must not be
 * reinterpreted as a plan-wide H0 gate, so clear only when the integral
 * topology snapshot proved healthy. Per-system codes remain authoritative. */
__global__ void prepare_h0_sequence_kernel(const std::uint32_t* integral_sequence,
                                           std::uint32_t* integral_device_error) {
  if (blockIdx.x == 0 && threadIdx.x == 0 && read_u32(integral_sequence) == 1u) {
    *integral_device_error = 0u;
  }
}

__global__ void prepare_late_stages_kernel(Gfn2GeometryDeviceBatch batch,
                                           const std::uint8_t* requested,
                                           const std::uint32_t* geometry_errors,
                                           const std::uint32_t* integral_errors, double* positions,
                                           std::uint32_t* aes2_errors, std::uint32_t* plan_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  if (begin < 0 || begin > end || end > batch.total_atoms) {
    if (threadIdx.x == 0) {
      record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kInvalidComposerOffsets);
    }
    return;
  }
  const bool failed =
      requested[system] != 1u || geometry_errors[system] != 0u || integral_errors[system] != 0u;
  if (threadIdx.x == 0 && requested[system] == 1u && failed) {
    atomicCAS(aes2_errors + system, 0u,
              static_cast<std::uint32_t>(Gfn2AES2DeviceError::kInvalidCoordination));
  }
  if (failed) {
    for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
      store_safe_positions(begin, atom, positions);
    }
  }
}

__device__ bool geometry_plan_code(std::uint32_t code) {
  return code == static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kInvalidOffsets) ||
         code == static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kInvalidCovalentRadius);
}

__device__ bool integral_plan_code(std::uint32_t code) {
  return code == static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kInvalidOffsets) ||
         code == static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kInvalidShellMetadata) ||
         code == static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kInvalidPrimitiveData) ||
         code == static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kInvalidH0Parameter);
}

__device__ bool aes2_plan_code(std::uint32_t code) {
  return code == static_cast<std::uint32_t>(Gfn2AES2DeviceError::kInvalidOffsets) ||
         code == static_cast<std::uint32_t>(Gfn2AES2DeviceError::kInvalidElementParameter);
}

/*
 * Bitwise dense/sparse CN consistency gate.  One block per system compares the
 * dense geometry-cache coordination numbers (the authoritative production
 * output) against the bucketed sparse pair-list coordination numbers for every
 * healthy, requested peer.  Any disagreement records kSparseCoordinationMismatch
 * in the peer's geometry error slot (and the sparse domain markers), so the
 * existing publication gate rejects that peer closed.  Failed or inactive
 * peers are intentionally skipped, preserving peer isolation.
 */
__global__ void gate_sparse_coordination_kernel(Gfn2GeometryDeviceBatch geometry,
                                                Gfn2PairListDeviceBatch pairlist,
                                                const double* dense_coordination,
                                                const double* sparse_coordination,
                                                const std::uint32_t* sparse_sequence_active,
                                                Gfn2PreprocessingDeviceActivity activity,
                                                Gfn2PreprocessingDeviceDiagnostics diagnostics) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const bool requested = activity.requested_mask[system] == 1u;
  const bool geometry_healthy =
      atomicAdd(const_cast<std::uint32_t*>(diagnostics.geometry_system_errors + system), 0u) ==
      static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess);
  const bool sparse_healthy =
      atomicAdd(const_cast<std::uint32_t*>(diagnostics.sparse_system_errors + system), 0u) ==
      static_cast<std::uint32_t>(Gfn2PairListDeviceError::kSuccess);
  if (!requested || !geometry_healthy) {
    return;
  }
  const bool sparse_sequence_ok =
      atomicAdd(const_cast<std::uint32_t*>(sparse_sequence_active), 0u) == 1u;
  if (!sparse_sequence_ok || !sparse_healthy) {
    /* A peer-local sparse error is folded into the existing geometry
     * publication gate; a topology-wide sequence failure closes every
     * requested healthy peer. */
    atomicCAS(diagnostics.geometry_system_errors + system,
              static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess),
              static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSparseCoordinationMismatch));
    return;
  }
  if (pairlist.batch_size != geometry.batch_size || pairlist.total_atoms != geometry.total_atoms ||
      pairlist.atom_offsets != geometry.atom_offsets) {
    atomicCAS(diagnostics.geometry_system_errors + system,
              static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess),
              static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSparseCoordinationMismatch));
    return;
  }
  const std::int64_t atom_begin = geometry.atom_offsets[system];
  const std::int64_t atom_end = geometry.atom_offsets[system + 1];
  __shared__ unsigned int shared_mismatch;
  if (threadIdx.x == 0) {
    shared_mismatch = 0u;
  }
  __syncthreads();
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const std::uint64_t dense_bits =
        *reinterpret_cast<const std::uint64_t*>(dense_coordination + atom);
    const std::uint64_t sparse_bits =
        *reinterpret_cast<const std::uint64_t*>(sparse_coordination + atom);
    if (dense_bits != sparse_bits) {
      atomicExch(&shared_mismatch, 1u);
    }
  }
  __syncthreads();
  if (shared_mismatch != 0u && threadIdx.x == 0) {
    atomicCAS(diagnostics.geometry_system_errors + system,
              static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess),
              static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSparseCoordinationMismatch));
    atomicCAS(diagnostics.sparse_system_errors + system,
              static_cast<std::uint32_t>(Gfn2PairListDeviceError::kSuccess),
              static_cast<std::uint32_t>(Gfn2PairListDeviceError::kInvalidCache));
    atomicCAS(diagnostics.sparse_device_error,
              static_cast<std::uint32_t>(Gfn2PairListDeviceError::kSuccess),
              static_cast<std::uint32_t>(Gfn2PairListDeviceError::kInvalidCache));
  }
}

/*
 * Make the sparse pair list the authoritative CN producer for the current
 * epoch.  The gate above proved the sparse coordination numbers equal the
 * dense geometry cache bitwise for every healthy requested peer; this kernel
 * then overwrites the dense CN in the geometry candidate with the sparse
 * value, so H0, AES2 geometry, and the publication path all consume the
 * sparse-produced CN.  The dense path remains as the differential reference
 * and its seven-value pair cache still feeds the coordination VJP until the
 * sparse VJP is wired into the force path.  Failed/inactive peers are skipped.
 */
__global__ void promote_sparse_coordination_kernel(Gfn2GeometryDeviceBatch geometry,
                                                   Gfn2PairListDeviceBatch pairlist,
                                                   const double* sparse_coordination,
                                                   double* dense_coordination,
                                                   Gfn2PreprocessingDeviceActivity activity,
                                                   Gfn2PreprocessingDeviceDiagnostics diagnostics) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const bool requested = activity.requested_mask[system] == 1u;
  if (!requested) {
    return;
  }
  const bool geometry_healthy =
      atomicAdd(const_cast<std::uint32_t*>(diagnostics.geometry_system_errors + system), 0u) ==
      static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess);
  if (!geometry_healthy) {
    return;
  }
  const std::int64_t atom_begin = geometry.atom_offsets[system];
  const std::int64_t atom_end = geometry.atom_offsets[system + 1];
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    dense_coordination[atom] = sparse_coordination[atom];
  }
  static_cast<void>(pairlist);
}

/*
 * Commit the sparse pair list into the stable public output view through the
 * final per-system gate.  Only requested, healthy peers publish: their
 * committed generation advances and their eligibility byte flips.  A failed or
 * inactive peer keeps its last good payload/counts/generation and receives an
 * ineligible byte for the attempted refresh.  Consumers read
 * only eligible peers and trust explicit per-peer counts + committed
 * generations (the consumer-view contract), so one peer's failure can never
 * publish a partial or shifted slice for another.
 *
 * Committed offsets address fixed-capacity slots.  Explicit counts delimit the
 * live prefix copied from the compact candidate, so a failed peer cannot move
 * another peer's bytes.  Fixed slot offsets may be rewritten after plan-wide
 * validation; the per-peer kernel changes persistent metadata and payload only
 * for a peer that passed the final publication gate.
 */
__global__ void initialize_committed_pairlist_metadata_kernel(Gfn2PairListDeviceBatch pairlist,
                                                              Gfn2PairListConsumerView committed,
                                                              Gfn2DeviceAdmission admission,
                                                              const std::uint32_t* plan_error) {
  if (!gfn2_request_admitted(admission) ||
      atomicAdd(const_cast<std::uint32_t*>(plan_error), 0u) != 0u) {
    return;
  }
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto* const pair_offsets = const_cast<std::int64_t*>(committed.pair_offsets);
  auto* const neighbor_offsets = const_cast<std::int64_t*>(committed.neighbor_offsets);
  if (index <= pairlist.batch_size) {
    pair_offsets[index] = index * pairlist.max_pairs_per_system;
  }
  if (index <= pairlist.total_atoms) {
    neighbor_offsets[index] = index * pairlist.max_neighbors_per_atom;
  }
}

__global__ void commit_pairlist_kernel(Gfn2PairListDeviceBatch pairlist,
                                       const Gfn2PairListDeviceCache& candidate,
                                       Gfn2PairListConsumerView committed,
                                       GeometryGenerationSource generation_source,
                                       Gfn2PreprocessingDeviceActivity activity,
                                       Gfn2DeviceAdmission admission,
                                       const std::uint32_t* plan_error) {
  /* A plan-wide failure is transactional for every committed array.  In
   * particular, do not clear generations or eligibility after the metadata
   * initializer has already declined to run. */
  if (!gfn2_request_admitted(admission) ||
      atomicAdd(const_cast<std::uint32_t*>(plan_error), 0u) != 0u) {
    return;
  }
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const std::uint64_t generation = load_geometry_generation(generation_source);
  /* published_mask is the single final transaction gate.  Geometry can be
   * healthy while integral/H0/AES2 or a plan-wide failure has already closed
   * publication, so re-deriving eligibility from the geometry error alone
   * would expose a partial operator transaction to sparse consumers. */
  const bool commit = activity.published_mask[system] == 1u && generation != 0u;
  /* The committed consumer view is a const projection; the output arrays are
   * the mutable publication target owned by the caller, so publish through
   * mutable local aliases. */
  auto* const committed_pairs = const_cast<xtbloom::detail::Gfn2AtomPair*>(committed.pairs);
  auto* const committed_pair_counts = const_cast<std::int64_t*>(committed.pair_counts);
  auto* const committed_neighbor_counts = const_cast<std::int64_t*>(committed.neighbor_counts);
  auto* const committed_neighbors = const_cast<std::int64_t*>(committed.neighbors);
  auto* const committed_generations = const_cast<std::uint64_t*>(committed.committed_generations);
  auto* const committed_eligible_mask = const_cast<std::uint8_t*>(committed.eligible_mask);
  if (threadIdx.x == 0) {
    if (committed_eligible_mask != nullptr) {
      committed_eligible_mask[system] = commit ? 1u : 0u;
    }
    if (commit && committed_generations != nullptr) {
      committed_generations[system] = generation;
    }
  }
  if (!commit) {
    return;
  }
  const std::int64_t atom_begin = pairlist.atom_offsets[system];
  const std::int64_t atom_end = pairlist.atom_offsets[system + 1];
  const std::int64_t candidate_pair_begin = candidate.pair_offsets[system];
  const std::int64_t pair_count = candidate.pair_counts[system];
  const std::int64_t committed_pair_begin = system * pairlist.max_pairs_per_system;
  for (std::int64_t index = threadIdx.x; index < pair_count; index += blockDim.x) {
    committed_pairs[committed_pair_begin + index] = candidate.pairs[candidate_pair_begin + index];
  }
  if (threadIdx.x == 0) {
    committed_pair_counts[system] = pair_count;
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const std::int64_t count = candidate.neighbor_counts[atom];
    const std::int64_t candidate_begin = candidate.neighbor_offsets[atom];
    const std::int64_t committed_begin = atom * pairlist.max_neighbors_per_atom;
    for (std::int64_t index = 0; index < count; ++index) {
      committed_neighbors[committed_begin + index] = candidate.neighbors[candidate_begin + index];
    }
    committed_neighbor_counts[atom] = count;
  }
}

__global__ void classify_plan_kernel(
    std::int64_t batch_size, const std::uint32_t* geometry_sequence,
    const std::uint32_t* sparse_sequence, const std::uint32_t* integral_sequence,
    const std::uint32_t* geometry_errors, const std::uint32_t* integral_errors,
    const std::uint32_t* es2_error, const std::uint32_t* aes2_errors, std::uint32_t* plan_error) {
  if (threadIdx.x != 0 || blockIdx.x != 0 || read_u32(plan_error) != 0u) {
    return;
  }
  if (read_u32(geometry_sequence) == 0u) {
    record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kGeometryPlanFailure);
    return;
  }
  if (sparse_sequence != nullptr && read_u32(sparse_sequence) == 0u) {
    /* Pair-list preflight validates one shared topology and dispatch vector.
     * A closed sequence is plan-wide and must stop committed metadata
     * initialization, rather than being reclassified as peer-local CN drift. */
    record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kSparsePairlistFailure);
    return;
  }
  if (read_u32(integral_sequence) == 0u) {
    record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kIntegralPlanFailure);
    return;
  }
  if (read_u32(es2_error) != 0u) {
    record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kEs2PlanFailure);
    return;
  }
  for (std::int64_t system = 0; system < batch_size; ++system) {
    if (geometry_plan_code(geometry_errors[system])) {
      record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kGeometryPlanFailure);
      return;
    }
    if (integral_plan_code(integral_errors[system])) {
      record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kIntegralPlanFailure);
      return;
    }
    if (aes2_plan_code(aes2_errors[system])) {
      record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kAes2PlanFailure);
      return;
    }
  }
}

__global__ void publish_preprocessing_kernel(Gfn2PreprocessingDevicePlan plan,
                                             Gfn2PreprocessingDeviceActivity activity,
                                             Gfn2PreprocessingDeviceOutput output,
                                             Gfn2PreprocessingDeviceWorkspace workspace,
                                             Gfn2PreprocessingDeviceDiagnostics diagnostics,
                                             Gfn2DeviceAdmission admission,
                                             GeometryGenerationSource generation_source) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!gfn2_request_admitted(admission)) return;
  const bool requested = activity.requested_mask[system] == 1u;
  const std::uint32_t geometry_error = diagnostics.geometry_system_errors[system];
  const std::uint32_t integral_error = diagnostics.integral_system_errors[system];
  const std::uint32_t aes2_error = diagnostics.aes2_system_errors[system];
  std::uint32_t stage = static_cast<std::uint32_t>(Gfn2PreprocessingSystemStage::kSuccess);
  if (requested) {
    if (geometry_error != 0u) {
      stage = static_cast<std::uint32_t>(Gfn2PreprocessingSystemStage::kGeometry);
    } else if (integral_error != 0u) {
      stage = static_cast<std::uint32_t>(Gfn2PreprocessingSystemStage::kIntegralsOrH0);
    } else if (aes2_error != 0u) {
      stage = static_cast<std::uint32_t>(Gfn2PreprocessingSystemStage::kAes2);
    }
  }
  if (threadIdx.x == 0) {
    diagnostics.system_stages[system] = stage;
  }

  const bool publish = requested && stage == 0u && read_u32(diagnostics.plan_error) == 0u;
  if (!publish) {
    if (threadIdx.x == 0) {
      activity.published_mask[system] = 0u;
    }
    return;
  }

  const std::uint64_t generation = load_geometry_generation(generation_source);
  if (generation == 0u) {
    if (threadIdx.x == 0) {
      activity.published_mask[system] = 0u;
      record_plan_error(diagnostics.plan_error,
                        Gfn2PreprocessingDeviceError::kGeometryEpochOverflow);
    }
    return;
  }

  const std::int64_t atom_begin = plan.geometry.atom_offsets[system];
  const std::int64_t atom_end = plan.geometry.atom_offsets[system + 1];
  const std::int64_t pair_begin = plan.geometry.pair_offsets[system];
  const std::int64_t pair_end = plan.geometry.pair_offsets[system + 1];
  const std::int64_t matrix_begin = plan.integrals.matrix_offsets[system];
  const std::int64_t matrix_end = plan.integrals.matrix_offsets[system + 1];
  const std::int64_t es2_begin = plan.es2.matrix_offsets[system];
  const std::int64_t es2_end = plan.es2.matrix_offsets[system + 1];

  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    output.geometry.coordination_numbers[atom] =
        workspace.geometry_candidate.coordination_numbers[atom];
  }
  for (std::int64_t pair = pair_begin + threadIdx.x; pair < pair_end; pair += blockDim.x) {
    for (std::int64_t component = 0; component < kGfn2GeometryPairDataElements; ++component) {
      output.geometry.pair_data[pair * kGfn2GeometryPairDataElements + component] =
          workspace.geometry_candidate.pair_data[pair * kGfn2GeometryPairDataElements + component];
    }
    for (std::int64_t component = 0; component < kGfn2AES2PairDataElements; ++component) {
      output.aes2.pair_data[pair * kGfn2AES2PairDataElements + component] =
          workspace.aes2_candidate.pair_data[pair * kGfn2AES2PairDataElements + component];
    }
  }
  for (std::int64_t element = matrix_begin + threadIdx.x; element < matrix_end;
       element += blockDim.x) {
    output.overlap[element] = workspace.overlap_candidate[element];
    output.h0[element] = workspace.h0_candidate[element];
    for (std::int64_t component = 0; component < kGfn2IntegralDipoleComponents; ++component) {
      output.dipole_integrals[component * plan.integrals.total_matrix_elements + element] =
          workspace.dipole_candidate[component * plan.integrals.total_matrix_elements + element];
    }
    for (std::int64_t component = 0; component < kGfn2IntegralQuadrupoleComponents; ++component) {
      output.quadrupole_integrals[component * plan.integrals.total_matrix_elements + element] =
          workspace
              .quadrupole_candidate[component * plan.integrals.total_matrix_elements + element];
    }
  }
  for (std::int64_t element = es2_begin + threadIdx.x; element < es2_end; element += blockDim.x) {
    output.es2.coulomb_matrix[element] = workspace.es2_candidate.coulomb_matrix[element];
  }
  /* The sparse list is committed by commit_pairlist_kernel after this final
   * operator gate has published the authoritative per-peer decision. */
  if (threadIdx.x == 0) {
    output.geometry.geometry_generations[system] = generation;
    output.operator_generations[system] = generation;
    activity.published_mask[system] = 1u;
  }
}

cudaError_t check_launch() noexcept { return cudaPeekAtLastError(); }

Gfn2PreprocessingLaunchDiagnostic launch_failure(const BindingDiagnostic& binding,
                                                 cudaError_t status) noexcept {
  Gfn2PreprocessingLaunchDiagnostic result{};
  result.binding = binding;
  result.cuda_status = status;
  return result;
}

}  // namespace

Gfn2PreprocessingBindingDiagnostic validate_gfn2_preprocessing_binding_cuda(
    const Gfn2PreprocessingDeviceBinding& binding) noexcept {
  return validate_structure(binding, true);
}

Gfn2PreprocessingBindingDiagnostic seal_gfn2_preprocessing_binding_cuda(
    Gfn2PreprocessingDeviceBinding& binding) noexcept {
  binding.binding_seal = 0u;
  const BindingDiagnostic diagnostic = validate_structure(binding, false);
  if (!diagnostic.success()) {
    return diagnostic;
  }
  binding.binding_seal = binding_seal(binding);
  return {};
}

namespace {

Gfn2PreprocessingLaunchDiagnostic compose_preprocessing_impl(
    Gfn2PreprocessingDeviceBinding& binding, GeometryGenerationSource generation_source,
    bool advance_epoch, cudaStream_t stream) noexcept {
  const BindingDiagnostic descriptor = validate_structure(binding, true);
  if (!descriptor.success()) {
    return launch_failure(descriptor, cudaErrorInvalidValue);
  }
  const bool epoch_enabled = binding.geometry_epoch.value != nullptr &&
                             binding.geometry_epoch.value_elements == 1 &&
                             binding.geometry_epoch.plan_token == binding.plan_token;
  if ((!advance_epoch && (generation_source.scalar == 0u || epoch_enabled)) ||
      (advance_epoch &&
       (!epoch_enabled || generation_source.device != binding.geometry_epoch.value))) {
    return launch_failure(
        binding_failure(
            advance_epoch ? BindingError::kInvalidEpoch : BindingError::kInvalidGeneration,
            advance_epoch ? BindingField::kEpoch : BindingField::kGeneration),
        cudaErrorInvalidValue);
  }
  const std::uint64_t primitive_generation =
      advance_epoch ? kUnpublishedPrimitiveGeneration : generation_source.scalar;

  const std::int64_t batch = binding.plan.geometry.batch_size;
  cudaError_t status =
      reset_gfn2_geometry_device_errors_cuda(batch, binding.diagnostics.geometry_system_errors,
                                             binding.diagnostics.geometry_device_error, stream);
  if (status != cudaSuccess) return launch_failure({}, status);
  status =
      reset_gfn2_integral_device_errors_cuda(batch, binding.diagnostics.integral_system_errors,
                                             binding.diagnostics.integral_device_error, stream);
  if (status != cudaSuccess) return launch_failure({}, status);
  status = reset_gfn2_es2_device_error_cuda(binding.diagnostics.es2_device_error, stream);
  if (status != cudaSuccess) return launch_failure({}, status);
  status = reset_gfn2_aes2_device_errors_cuda(batch, binding.diagnostics.aes2_system_errors,
                                              binding.diagnostics.aes2_device_error, stream);
  if (status != cudaSuccess) return launch_failure({}, status);
  status = cudaMemsetAsync(binding.diagnostics.system_stages, 0,
                           static_cast<std::size_t>(batch) * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess) return launch_failure({}, status);
  status = cudaMemsetAsync(binding.diagnostics.plan_error, 0, sizeof(std::uint32_t), stream);
  if (status != cudaSuccess) return launch_failure({}, status);
  if (advance_epoch) {
    advance_geometry_epoch_kernel<<<1, 1, 0, stream>>>(binding.geometry_epoch, binding.admission,
                                                       binding.diagnostics.plan_error);
    status = check_launch();
    if (status != cudaSuccess) return launch_failure({}, status);
  }

  prepare_positions_kernel<<<static_cast<unsigned int>(batch), kThreadsPerBlock, 0, stream>>>(
      binding.plan.geometry, binding.input.positions, binding.activity.requested_mask,
      binding.workspace.positions_scratch, binding.diagnostics.plan_error);
  status = check_launch();
  if (status != cudaSuccess) return launch_failure({}, status);
  const unsigned int gate_blocks = static_cast<unsigned int>(
      (static_cast<std::uint64_t>(batch) + kThreadsPerBlock - 1u) / kThreadsPerBlock);
  gate_composer_plan_kernel<<<gate_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, binding.diagnostics.plan_error, binding.diagnostics.geometry_system_errors,
      binding.diagnostics.geometry_device_error, binding.diagnostics.integral_system_errors,
      binding.diagnostics.integral_device_error, binding.diagnostics.es2_device_error,
      binding.diagnostics.aes2_system_errors, binding.diagnostics.aes2_device_error);
  status = check_launch();
  if (status != cudaSuccess) return launch_failure({}, status);

  status = update_gfn2_geometry_cache_cuda(
      binding.plan.geometry, binding.workspace.positions_scratch, primitive_generation,
      binding.workspace.geometry_candidate, binding.workspace.geometry,
      binding.diagnostics.geometry_system_errors, binding.diagnostics.geometry_device_error,
      stream);
  if (status != cudaSuccess) return launch_failure({}, status);

  /* Optional sparse pair-list consistency gate.  The host scheduler enables
   * plan.pairlist when the batch crosses the measured dense-fallback crossover;
   * the bucketed builder then reproduces coordination numbers that must match
   * the dense geometry cache bitwise.  A mismatch fails that peer closed, so a
   * sparse/dense regression can never silently publish different physics. */
  if (binding.plan.pairlist.batch_size > 0) {
    status =
        reset_gfn2_pairlist_device_errors_cuda(batch, binding.diagnostics.sparse_system_errors,
                                               binding.diagnostics.sparse_device_error, stream);
    if (status != cudaSuccess) return launch_failure({}, status);
    status = update_gfn2_pairlist_cache_cuda(
        binding.plan.pairlist, binding.workspace.positions_scratch, primitive_generation,
        binding.workspace.pairlist_candidate, binding.workspace.pairlist,
        binding.diagnostics.sparse_system_errors, binding.diagnostics.sparse_device_error, stream);
    if (status != cudaSuccess) return launch_failure({}, status);
    status = evaluate_gfn2_pairlist_coordination_cuda(
        binding.plan.pairlist, binding.workspace.positions_scratch,
        binding.plan.geometry.covalent_radii, primitive_generation,
        binding.workspace.pairlist_candidate, binding.workspace.sparse_coordination,
        binding.workspace.pairlist, binding.diagnostics.sparse_system_errors,
        binding.diagnostics.sparse_device_error, stream);
    if (status != cudaSuccess) return launch_failure({}, status);
    const unsigned int sparse_blocks = static_cast<unsigned int>(batch);
    gate_sparse_coordination_kernel<<<sparse_blocks, kThreadsPerBlock, 0, stream>>>(
        binding.plan.geometry, binding.plan.pairlist,
        binding.workspace.geometry_candidate.coordination_numbers,
        binding.workspace.sparse_coordination, binding.workspace.pairlist.sequence_active,
        binding.activity, binding.diagnostics);
    status = check_launch();
    if (status != cudaSuccess) return launch_failure({}, status);
    /* After the bitwise gate passed, the sparse list is the authoritative CN
     * producer: publish its values into the geometry candidate so H0, AES2,
     * and the public cache consume sparse-produced CN.  The dense path remains
     * the differential reference behind the gate. */
    promote_sparse_coordination_kernel<<<sparse_blocks, kThreadsPerBlock, 0, stream>>>(
        binding.plan.geometry, binding.plan.pairlist, binding.workspace.sparse_coordination,
        binding.workspace.geometry_candidate.coordination_numbers, binding.activity,
        binding.diagnostics);
    status = check_launch();
    if (status != cudaSuccess) return launch_failure({}, status);
  }
  status = evaluate_gfn2_integrals_cuda(
      binding.plan.integrals, binding.workspace.positions_scratch,
      binding.workspace.overlap_candidate, binding.workspace.dipole_candidate,
      binding.workspace.quadrupole_candidate, binding.workspace.integrals,
      binding.diagnostics.integral_system_errors, binding.diagnostics.integral_device_error,
      stream);
  if (status != cudaSuccess) return launch_failure({}, status);

  gate_h0_kernel<<<gate_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, binding.activity.requested_mask, binding.diagnostics.geometry_system_errors,
      binding.diagnostics.integral_system_errors);
  status = check_launch();
  if (status != cudaSuccess) return launch_failure({}, status);
  prepare_h0_sequence_kernel<<<1, 1, 0, stream>>>(binding.workspace.integrals.sequence_active,
                                                  binding.diagnostics.integral_device_error);
  status = check_launch();
  if (status != cudaSuccess) return launch_failure({}, status);
  status = evaluate_gfn2_h0_cuda(
      binding.plan.integrals, binding.plan.h0, binding.workspace.positions_scratch,
      binding.workspace.geometry_candidate.coordination_numbers,
      binding.workspace.overlap_candidate, binding.workspace.h0_candidate,
      binding.workspace.integrals, binding.diagnostics.integral_system_errors,
      binding.diagnostics.integral_device_error, stream);
  if (status != cudaSuccess) return launch_failure({}, status);

  prepare_late_stages_kernel<<<static_cast<unsigned int>(batch), kThreadsPerBlock, 0, stream>>>(
      binding.plan.geometry, binding.activity.requested_mask,
      binding.diagnostics.geometry_system_errors, binding.diagnostics.integral_system_errors,
      binding.workspace.positions_scratch, binding.diagnostics.aes2_system_errors,
      binding.diagnostics.plan_error);
  status = check_launch();
  if (status != cudaSuccess) return launch_failure({}, status);

  Gfn2ES2DeviceCache es2_candidate = binding.workspace.es2_candidate;
  es2_candidate.geometry_generation = primitive_generation;
  status = update_gfn2_es2_geometry_cache_cuda(
      binding.plan.es2, binding.workspace.positions_scratch, es2_candidate, binding.workspace.es2,
      binding.diagnostics.es2_device_error, stream);
  if (status != cudaSuccess) return launch_failure({}, status);

  Gfn2AES2DeviceCache aes2_candidate = binding.workspace.aes2_candidate;
  aes2_candidate.geometry_generation = primitive_generation;
  status = update_gfn2_aes2_geometry_cache_cuda(
      binding.plan.aes2, binding.workspace.positions_scratch,
      binding.workspace.geometry_candidate.coordination_numbers, aes2_candidate,
      binding.workspace.aes2, binding.diagnostics.aes2_system_errors,
      binding.diagnostics.aes2_device_error, stream);
  if (status != cudaSuccess) return launch_failure({}, status);

  classify_plan_kernel<<<1, 1, 0, stream>>>(
      batch, binding.workspace.geometry.sequence_active,
      binding.plan.pairlist.batch_size > 0 ? binding.workspace.pairlist.sequence_active : nullptr,
      binding.workspace.integrals.sequence_active, binding.diagnostics.geometry_system_errors,
      binding.diagnostics.integral_system_errors, binding.diagnostics.es2_device_error,
      binding.diagnostics.aes2_system_errors, binding.diagnostics.plan_error);
  status = check_launch();
  if (status != cudaSuccess) return launch_failure({}, status);
  publish_preprocessing_kernel<<<static_cast<unsigned int>(batch), kThreadsPerBlock, 0, stream>>>(
      binding.plan, binding.activity, binding.output, binding.workspace, binding.diagnostics,
      binding.admission, generation_source);
  status = check_launch();
  if (status != cudaSuccess) return launch_failure({}, status);

  /* Committed pair-list transaction: initialize fixed-capacity slot metadata
   * only after plan validation, then publish each healthy peer into its own
   * stable slot with the current generation and eligibility. */
  if (binding.plan.pairlist.batch_size > 0) {
    const std::int64_t metadata_elements =
        std::max<std::int64_t>(batch + 1, binding.plan.pairlist.total_atoms + 1);
    initialize_committed_pairlist_metadata_kernel<<<
        static_cast<unsigned int>((metadata_elements + kThreadsPerBlock - 1) / kThreadsPerBlock),
        kThreadsPerBlock, 0, stream>>>(binding.plan.pairlist, binding.output.pairlist,
                                       binding.admission, binding.diagnostics.plan_error);
    status = check_launch();
    if (status != cudaSuccess) return launch_failure({}, status);
    commit_pairlist_kernel<<<static_cast<unsigned int>(batch), kThreadsPerBlock, 0, stream>>>(
        binding.plan.pairlist, binding.workspace.pairlist_candidate, binding.output.pairlist,
        generation_source, binding.activity, binding.admission, binding.diagnostics.plan_error);
    status = check_launch();
    if (status != cudaSuccess) return launch_failure({}, status);
  }

  /* Only the legacy path can update host-side descriptor scalars. In either
   * mode, device-resident per-system generations plus published_mask are the
   * authoritative record that a peer's complete public cache was committed. */
  if (!advance_epoch) {
    binding.output.es2.geometry_generation = generation_source.scalar;
    binding.output.aes2.geometry_generation = generation_source.scalar;
    binding.workspace.es2_candidate.geometry_generation = generation_source.scalar;
    binding.workspace.aes2_candidate.geometry_generation = generation_source.scalar;
  }
  return {};
}

}  // namespace

Gfn2PreprocessingLaunchDiagnostic compose_gfn2_preprocessing_cuda(
    Gfn2PreprocessingDeviceBinding& binding, std::uint64_t geometry_generation,
    cudaStream_t stream) noexcept {
  return compose_preprocessing_impl(binding, {geometry_generation, nullptr}, false, stream);
}

Gfn2PreprocessingLaunchDiagnostic compose_gfn2_preprocessing_epoch_cuda(
    Gfn2PreprocessingDeviceBinding& binding, cudaStream_t stream) noexcept {
  return compose_preprocessing_impl(binding, {0u, binding.geometry_epoch.value}, true, stream);
}

Gfn2PreprocessingLaunchDiagnostic gate_gfn2_sparse_coordination_cuda(
    Gfn2PreprocessingDeviceBinding& binding, cudaStream_t stream) noexcept {
  const BindingDiagnostic descriptor = validate_structure(binding, true);
  if (descriptor.error != BindingError::kSuccess) {
    return launch_failure(descriptor, cudaErrorInvalidValue);
  }
  if (binding.plan.pairlist.batch_size <= 0) {
    return launch_failure(binding_failure(BindingError::kInvalidWorkspace, BindingField::kPairlist),
                          cudaErrorInvalidValue);
  }
  const std::int64_t batch = binding.plan.geometry.batch_size;
  const unsigned int blocks = static_cast<unsigned int>(batch);
  gate_sparse_coordination_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      binding.plan.geometry, binding.plan.pairlist,
      binding.workspace.geometry_candidate.coordination_numbers,
      binding.workspace.sparse_coordination, binding.workspace.pairlist.sequence_active,
      binding.activity, binding.diagnostics);
  return {binding_failure(BindingError::kSuccess, BindingField::kNone), check_launch()};
}

}  // namespace xtbloom::detail::cuda
