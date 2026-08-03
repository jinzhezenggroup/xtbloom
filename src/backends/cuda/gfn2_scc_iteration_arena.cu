#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_scc_iteration_arena.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr std::size_t kSliceAlignment = 64u;
constexpr std::uint32_t kMandatoryComponents =
    static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES2) |
    static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES3) |
    static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kAES2);

struct ArenaShape {
  std::int64_t batch = 0;
  std::int64_t atoms = 0;
  std::int64_t shells = 0;
  std::int64_t orbitals = 0;
  std::int64_t matrices = 0;
  std::int64_t dipoles = 0;
  std::int64_t quadrupoles = 0;
  std::int64_t two_batch = 0;
  std::int64_t two_orbitals = 0;
  std::int64_t mixer_vector = 0;
  std::int64_t mixer_history = 0;
  std::int64_t mixer_history_elements = 0;
  std::int64_t mixer_omega_elements = 0;
  std::int64_t mixer_beta_elements = 0;
  std::int64_t mixer_coefficient_elements = 0;
  std::int64_t geometry_pair_elements = 0;
  std::int64_t es2_matrix_elements = 0;
  std::int64_t aes2_pair_elements = 0;
  std::int64_t aes2_potential_elements = 0;
  std::int64_t d4_weight_elements = 0;
  std::int64_t classical_scratch_elements = 0;
  std::int64_t free_scratch_elements = 0;
  std::uint32_t enabled_components = 0u;
  std::uint64_t plan_token = 0u;
  std::size_t provider_device_bytes = 0u;
  std::size_t provider_host_bytes = 0u;
  Gfn2SccIterationReportStorageRequirements reports{};
};

[[nodiscard]] bool checked_add(std::int64_t first, std::int64_t second,
                               std::int64_t& result) noexcept {
  if (first < 0 || second < 0 || first > std::numeric_limits<std::int64_t>::max() - second) {
    return false;
  }
  result = first + second;
  return true;
}

[[nodiscard]] bool checked_multiply(std::int64_t first, std::int64_t second,
                                    std::int64_t& result) noexcept {
  if (first < 0 || second < 0 ||
      (first != 0 && second > std::numeric_limits<std::int64_t>::max() / first)) {
    return false;
  }
  result = first * second;
  return true;
}

[[nodiscard]] bool component_enabled(const ArenaShape& shape,
                                     Gfn2SccPotentialComponent component) noexcept {
  return (shape.enabled_components & static_cast<std::uint32_t>(component)) != 0u;
}

[[nodiscard]] Gfn2SccIterationArenaDiagnostic failure(Gfn2SccIterationArenaError error,
                                                      std::size_t required = 0u,
                                                      std::size_t provided = 0u) noexcept {
  return {error, required, provided};
}

[[nodiscard]] bool derive_shape(const Gfn2SccIterationDevicePlan& plan,
                                const Gfn2EigensolverWorkspaceRequirements& provider_requirements,
                                ArenaShape& shape, Gfn2SccIterationArenaError& error) noexcept {
  shape = {};
  error = Gfn2SccIterationArenaError::kInvalidPlan;
  const auto& topology = plan.topology;
  if (plan.abi_version != kGfn2SccIterationAbiVersion || plan.plan_token == 0u ||
      topology.plan_token != plan.plan_token || topology.batch_size <= 0 ||
      topology.total_atoms <= 0 || topology.total_shells <= 0 || topology.total_orbitals <= 0 ||
      topology.total_matrix_elements <= 0 || plan.mixer_policy.history_size <= 0 ||
      (plan.enabled_components & kMandatoryComponents) != kMandatoryComponents ||
      (plan.enabled_components & ~kGfn2SccPotentialAllComponents) != 0u ||
      plan.geometry_batch.total_pairs < 0 || plan.es2_batch.total_matrix_elements < 0 ||
      plan.aes2_batch.total_pairs < 0 || plan.d4_batch.total_pairs < 0) {
    return false;
  }
  if (provider_requirements.solver_device_workspace_bytes >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      provider_requirements.solver_host_workspace_bytes >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    error = Gfn2SccIterationArenaError::kInvalidProviderRequirements;
    return false;
  }

  shape.batch = topology.batch_size;
  shape.atoms = topology.total_atoms;
  shape.shells = topology.total_shells;
  shape.orbitals = topology.total_orbitals;
  shape.matrices = topology.total_matrix_elements;
  shape.mixer_history = plan.mixer_policy.history_size;
  shape.es2_matrix_elements = plan.es2_batch.total_matrix_elements;
  shape.enabled_components = plan.enabled_components;
  shape.plan_token = plan.plan_token;
  shape.provider_device_bytes = provider_requirements.solver_device_workspace_bytes;
  shape.provider_host_bytes = provider_requirements.solver_host_workspace_bytes;

  std::int64_t history_square = 0;
  if (!checked_multiply(shape.atoms, 3, shape.dipoles) ||
      !checked_multiply(shape.atoms, 6, shape.quadrupoles) ||
      !checked_multiply(shape.batch, 2, shape.two_batch) ||
      !checked_multiply(shape.orbitals, 2, shape.two_orbitals) ||
      !checked_multiply(shape.atoms, 9, shape.mixer_vector) ||
      !checked_add(shape.shells, shape.mixer_vector, shape.mixer_vector) ||
      !checked_multiply(shape.mixer_vector, shape.mixer_history, shape.mixer_history_elements) ||
      !checked_multiply(shape.batch, shape.mixer_history, shape.mixer_omega_elements) ||
      !checked_multiply(shape.mixer_history, shape.mixer_history, history_square) ||
      !checked_multiply(shape.batch, history_square, shape.mixer_beta_elements) ||
      !checked_multiply(shape.batch, shape.mixer_history, shape.mixer_coefficient_elements) ||
      !checked_multiply(plan.geometry_batch.total_pairs, kGfn2GeometryPairDataElements,
                        shape.geometry_pair_elements) ||
      !checked_multiply(plan.aes2_batch.total_pairs, kGfn2AES2PairDataElements,
                        shape.aes2_pair_elements) ||
      !checked_multiply(shape.atoms, kGfn2AES2PotentialElementsPerAtom,
                        shape.aes2_potential_elements) ||
      !checked_multiply(shape.atoms, kGfn2D4MaximumReferences, shape.d4_weight_elements) ||
      !checked_multiply(shape.batch, kGfn2SccClassicalDiagnosticComponents,
                        shape.classical_scratch_elements) ||
      !checked_multiply(shape.batch, kGfn2SccFreeEnergyDiagnosticComponents,
                        shape.free_scratch_elements)) {
    error = Gfn2SccIterationArenaError::kSizeOverflow;
    return false;
  }

  const auto report_diagnostic = query_gfn2_scc_iteration_report_storage_cuda(
      shape.enabled_components, shape.batch, shape.reports);
  if (report_diagnostic.error != Gfn2SccIterationBindingError::kSuccess) {
    error = report_diagnostic.error == Gfn2SccIterationBindingError::kAddressOverflow
                ? Gfn2SccIterationArenaError::kSizeOverflow
                : Gfn2SccIterationArenaError::kInvalidPlan;
    return false;
  }
  error = Gfn2SccIterationArenaError::kSuccess;
  return true;
}

class ArenaCursor {
 public:
  explicit ArenaCursor(std::byte* base) noexcept : base_(base) {}

  template <typename T>
  [[nodiscard]] T* take(std::int64_t elements) noexcept {
    if (elements < 0) {
      valid_ = false;
      return nullptr;
    }
    if (elements == 0) {
      return nullptr;
    }
    if (static_cast<std::uint64_t>(elements) >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(T))) {
      valid_ = false;
      return nullptr;
    }
    const std::size_t alignment = alignof(T) > kSliceAlignment ? alignof(T) : kSliceAlignment;
    if (!align(alignment)) {
      return nullptr;
    }
    const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(T);
    if (offset_ > std::numeric_limits<std::size_t>::max() - bytes) {
      valid_ = false;
      return nullptr;
    }
    T* const result = base_ == nullptr ? nullptr : reinterpret_cast<T*>(base_ + offset_);
    offset_ += bytes;
    return result;
  }

  [[nodiscard]] void* take_bytes(std::size_t bytes, std::size_t alignment) noexcept {
    if (bytes == 0u) {
      return nullptr;
    }
    if (!align(alignment) || offset_ > std::numeric_limits<std::size_t>::max() - bytes) {
      valid_ = false;
      return nullptr;
    }
    void* const result = base_ == nullptr ? nullptr : base_ + offset_;
    offset_ += bytes;
    return result;
  }

  [[nodiscard]] bool align(std::size_t alignment) noexcept {
    if (alignment == 0u || (alignment & (alignment - 1u)) != 0u) {
      valid_ = false;
      return false;
    }
    const std::size_t mask = alignment - 1u;
    if (offset_ > std::numeric_limits<std::size_t>::max() - mask) {
      valid_ = false;
      return false;
    }
    offset_ = (offset_ + mask) & ~mask;
    return true;
  }

  [[nodiscard]] std::size_t offset() const noexcept { return offset_; }
  [[nodiscard]] bool valid() const noexcept { return valid_; }

 private:
  std::byte* base_ = nullptr;
  std::size_t offset_ = 0u;
  bool valid_ = true;
};

[[nodiscard]] std::uint64_t mix_hash(std::uint64_t value) noexcept {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31u);
}

void hash_append(std::uint64_t value, std::uint64_t& hash) noexcept {
  hash = mix_hash(hash ^ mix_hash(value + 0x9e3779b97f4a7c15ULL));
}

[[nodiscard]] std::uint64_t shape_fingerprint(const ArenaShape& shape) noexcept {
  std::uint64_t hash = 0x101c0de89abcdef0ULL;
  hash_append(kGfn2SccIterationArenaAbiVersion, hash);
  hash_append(static_cast<std::uint64_t>(shape.batch), hash);
  hash_append(static_cast<std::uint64_t>(shape.atoms), hash);
  hash_append(static_cast<std::uint64_t>(shape.shells), hash);
  hash_append(static_cast<std::uint64_t>(shape.orbitals), hash);
  hash_append(static_cast<std::uint64_t>(shape.matrices), hash);
  hash_append(static_cast<std::uint64_t>(shape.mixer_history), hash);
  hash_append(static_cast<std::uint64_t>(shape.geometry_pair_elements), hash);
  hash_append(static_cast<std::uint64_t>(shape.es2_matrix_elements), hash);
  hash_append(static_cast<std::uint64_t>(shape.aes2_pair_elements), hash);
  hash_append(static_cast<std::uint64_t>(shape.d4_weight_elements), hash);
  hash_append(shape.enabled_components, hash);
  hash_append(shape.plan_token, hash);
  hash_append(static_cast<std::uint64_t>(shape.provider_device_bytes), hash);
  hash_append(static_cast<std::uint64_t>(shape.provider_host_bytes), hash);
  hash_append(static_cast<std::uint64_t>(shape.reports.report_count), hash);
  hash_append(static_cast<std::uint64_t>(shape.reports.system_error_elements), hash);
  return hash == 0u ? 1u : hash;
}

[[nodiscard]] bool same_requirements(const Gfn2SccIterationArenaRequirements& first,
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

/*
 * These projections are declared before the front-of-workspace builder so
 * that layout order remains readable: descriptor helpers describe one slice,
 * while project_workspace_front records the SCC dataflow aliases between
 * those slices.
 */
[[nodiscard]] Gfn2EigensolverDeviceResults take_eigenpairs(ArenaCursor& cursor,
                                                           const ArenaShape& shape) noexcept;
[[nodiscard]] Gfn2OccupationsDeviceResults take_occupations(ArenaCursor& cursor,
                                                            const ArenaShape& shape) noexcept;
[[nodiscard]] Gfn2DensityDeviceResults take_density(ArenaCursor& cursor,
                                                    const ArenaShape& shape) noexcept;
[[nodiscard]] Gfn2MullikenDevicePopulation take_population(ArenaCursor& cursor,
                                                           const ArenaShape& shape) noexcept;
[[nodiscard]] Gfn2SccDeviceMultipoles take_multipoles(ArenaCursor& cursor,
                                                      const ArenaShape& shape) noexcept;
[[nodiscard]] Gfn2SccMixerDeviceState take_mixer(ArenaCursor& cursor,
                                                 const ArenaShape& shape) noexcept;
void take_energy_trace(ArenaCursor& cursor, const ArenaShape& shape, double* entropy_alias,
                       bool alias_entropy, Gfn2SccClassicalEnergyDeviceDiagnostics& classical,
                       Gfn2SccFreeEnergyDeviceDiagnostics& free) noexcept;
void project_persistent(ArenaCursor& cursor, const ArenaShape& shape,
                        Gfn2SccIterationDeviceState& state) noexcept;
void project_workspace(ArenaCursor& cursor, const ArenaShape& shape,
                       const Gfn2SccIterationDevicePlan& plan,
                       const Gfn2SccIterationDeviceState& state,
                       Gfn2SccIterationDeviceWorkspace& workspace,
                       Gfn2SccIterationReportStorage& report_storage) noexcept;

}  // namespace

}  // namespace gpuxtb::detail::cuda

namespace gpuxtb::detail::cuda {
namespace {

void take_component_storage(ArenaCursor& cursor, const ArenaShape& shape,
                            Gfn2SccIterationDeviceComponentStorage& storage) noexcept {
  storage = {};
  storage.es2_shell_potential = cursor.take<double>(shape.shells);
  storage.es2_shell_elements = shape.shells;
  storage.es3_shell_potential = cursor.take<double>(shape.shells);
  storage.es3_shell_elements = shape.shells;
  storage.aes2_atomic_potential = cursor.take<double>(shape.atoms);
  storage.aes2_atomic_elements = shape.atoms;
  storage.aes2_dipole_potential = cursor.take<double>(shape.dipoles);
  storage.aes2_dipole_elements = shape.dipoles;
  storage.aes2_quadrupole_potential = cursor.take<double>(shape.quadrupoles);
  storage.aes2_quadrupole_elements = shape.quadrupoles;
  if (component_enabled(shape, Gfn2SccPotentialComponent::kD4TwoBody)) {
    storage.d4_atomic_potential = cursor.take<double>(shape.atoms);
    storage.d4_atomic_elements = shape.atoms;
  }
  if (component_enabled(shape, Gfn2SccPotentialComponent::kPeriodicEmbedding)) {
    storage.periodic_atomic_potential = cursor.take<double>(shape.atoms);
    storage.periodic_atomic_elements = shape.atoms;
  }

  storage.es2_energy = cursor.take<double>(shape.batch);
  storage.es2_energy_elements = shape.batch;
  storage.es3_energy = cursor.take<double>(shape.batch);
  storage.es3_energy_elements = shape.batch;
  storage.aes2_energy = cursor.take<double>(shape.batch);
  storage.aes2_energy_elements = shape.batch;
  if (component_enabled(shape, Gfn2SccPotentialComponent::kD4TwoBody)) {
    storage.d4_two_body_energy = cursor.take<double>(shape.batch);
    storage.d4_two_body_energy_elements = shape.batch;
  }
  if (component_enabled(shape, Gfn2SccPotentialComponent::kExplicitPointCharge)) {
    storage.explicit_point_charge_energy = cursor.take<double>(shape.batch);
    storage.explicit_point_charge_energy_elements = shape.batch;
  }
  if (component_enabled(shape, Gfn2SccPotentialComponent::kPeriodicEmbedding)) {
    storage.periodic_embedding_energy = cursor.take<double>(shape.batch);
    storage.periodic_embedding_energy_elements = shape.batch;
  }
  storage.core_energy = cursor.take<double>(shape.batch);
  storage.core_energy_elements = shape.batch;
  storage.electronic_free_energy = cursor.take<double>(shape.batch);
  storage.electronic_free_energy_elements = shape.batch;
  storage.plan_token = shape.plan_token;
}

void project_workspace_front(ArenaCursor& cursor, const ArenaShape& shape,
                             const Gfn2SccIterationDevicePlan& plan,
                             const Gfn2SccIterationDeviceState& state,
                             Gfn2SccIterationDeviceWorkspace& workspace) noexcept {
  workspace = {};
  workspace.plan_token = shape.plan_token;
  workspace.ledger.active_mask = cursor.take<std::uint8_t>(shape.batch);
  workspace.ledger.pending_statuses = cursor.take<gpuxtb_status_t>(shape.batch);
  workspace.ledger.system_failure_records = cursor.take<std::uint64_t>(shape.batch);
  workspace.ledger.plan_failure_record = cursor.take<std::uint64_t>(1);
  workspace.ledger.sequence_active = cursor.take<std::uint32_t>(1);
  workspace.ledger.batch_elements = shape.batch;
  workspace.ledger.scalar_elements = 1;
  workspace.ledger.plan_token = shape.plan_token;

  workspace.activity = {workspace.ledger.active_mask, workspace.ledger.sequence_active, shape.batch,
                        1, shape.plan_token};
  workspace.potential_activity = {workspace.ledger.active_mask, shape.batch, shape.plan_token};
  workspace.hamiltonian_activity = {workspace.ledger.active_mask, shape.batch, shape.plan_token};
  workspace.mulliken_activity = {workspace.ledger.active_mask, shape.batch, shape.plan_token};
  workspace.classical_energy_activity = {workspace.ledger.active_mask, shape.batch,
                                         shape.plan_token};
  workspace.free_energy_activity = {workspace.ledger.active_mask, shape.batch, shape.plan_token};

  workspace.staged_eigenpairs = take_eigenpairs(cursor, shape);
  workspace.staged_occupations = take_occupations(cursor, shape);
  workspace.staged_density = take_density(cursor, shape);
  workspace.staged_raw_population = take_population(cursor, shape);
  take_energy_trace(cursor, shape, workspace.staged_occupations.entropies, true,
                    workspace.staged_classical_energy, workspace.staged_free_energy);
  workspace.staged_mixer = take_mixer(cursor, shape);
  workspace.next_mixed = take_multipoles(cursor, shape);

  workspace.mixed_topology = {state.scc.current_inputs.shell_charges,
                              shape.shells,
                              cursor.take<double>(shape.atoms),
                              shape.atoms,
                              state.scc.current_inputs.atomic_dipoles,
                              shape.dipoles,
                              state.scc.current_inputs.atomic_quadrupoles,
                              shape.quadrupoles,
                              shape.plan_token};

  take_component_storage(cursor, shape, workspace.components);
  workspace.potential_components.enabled_components = shape.enabled_components;
  workspace.potential_components.es2_shell = workspace.components.es2_shell_potential;
  workspace.potential_components.es2_shell_elements = workspace.components.es2_shell_elements;
  workspace.potential_components.es3_shell = workspace.components.es3_shell_potential;
  workspace.potential_components.es3_shell_elements = workspace.components.es3_shell_elements;
  if (component_enabled(shape, Gfn2SccPotentialComponent::kExplicitPointCharge)) {
    workspace.potential_components.explicit_point_charge_shell =
        plan.explicit_point_charge_cache.shell_potentials;
    workspace.potential_components.explicit_point_charge_shell_elements = shape.shells;
  }
  workspace.potential_components.aes2_atomic = workspace.components.aes2_atomic_potential;
  workspace.potential_components.aes2_atomic_elements = workspace.components.aes2_atomic_elements;
  workspace.potential_components.aes2_dipole = workspace.components.aes2_dipole_potential;
  workspace.potential_components.aes2_dipole_elements = workspace.components.aes2_dipole_elements;
  workspace.potential_components.aes2_quadrupole = workspace.components.aes2_quadrupole_potential;
  workspace.potential_components.aes2_quadrupole_elements =
      workspace.components.aes2_quadrupole_elements;
  workspace.potential_components.d4_atomic = workspace.components.d4_atomic_potential;
  workspace.potential_components.d4_atomic_elements = workspace.components.d4_atomic_elements;
  workspace.potential_components.periodic_atomic = workspace.components.periodic_atomic_potential;
  workspace.potential_components.periodic_atomic_elements =
      workspace.components.periodic_atomic_elements;
  workspace.potential_components.plan_token = shape.plan_token;

  workspace.complete_potentials = {cursor.take<double>(shape.shells),
                                   shape.shells,
                                   cursor.take<double>(shape.atoms),
                                   shape.atoms,
                                   cursor.take<double>(shape.dipoles),
                                   shape.dipoles,
                                   cursor.take<double>(shape.quadrupoles),
                                   shape.quadrupoles,
                                   shape.plan_token};
  workspace.scalar_bridge.fields = {workspace.complete_potentials.shell, shape.shells,
                                    workspace.complete_potentials.atomic, shape.atoms,
                                    shape.plan_token};
  workspace.scalar_bridge.shell_scalar = cursor.take<double>(shape.shells);
  workspace.scalar_bridge.shell_elements = shape.shells;
  workspace.scalar_bridge.plan_token = shape.plan_token;
  workspace.hamiltonian = {cursor.take<double>(shape.matrices), shape.matrices, shape.plan_token};

  workspace.staged_publication.wavefunction = {
      workspace.staged_eigenpairs, workspace.staged_occupations, workspace.staged_density,
      workspace.staged_raw_population, shape.plan_token};
  workspace.staged_publication.energy = {workspace.staged_classical_energy,
                                         workspace.staged_free_energy, shape.plan_token};
  workspace.staged_publication.mixer = workspace.staged_mixer;
  workspace.staged_publication.next_mixed = {workspace.next_mixed.shell_charges,
                                             shape.shells,
                                             workspace.next_mixed.atomic_dipoles,
                                             shape.dipoles,
                                             workspace.next_mixed.atomic_quadrupoles,
                                             shape.quadrupoles,
                                             shape.plan_token};
  workspace.staged_publication.plan_token = shape.plan_token;
}

}  // namespace
}  // namespace gpuxtb::detail::cuda

namespace gpuxtb::detail::cuda {

Gfn2SccIterationArenaDiagnostic query_gfn2_scc_iteration_arena_requirements_cuda(
    const Gfn2SccIterationDevicePlan& plan,
    const Gfn2EigensolverWorkspaceRequirements& provider_requirements,
    Gfn2SccIterationArenaRequirements& requirements) noexcept {
  requirements = {};
  ArenaShape shape{};
  Gfn2SccIterationArenaError shape_error = Gfn2SccIterationArenaError::kInvalidPlan;
  if (!derive_shape(plan, provider_requirements, shape, shape_error)) {
    return failure(shape_error);
  }

  Gfn2SccIterationArenaRequirements candidate{};
  ArenaCursor cursor(nullptr);
  Gfn2SccIterationDeviceState state{};
  candidate.persistent_offset = cursor.offset();
  project_persistent(cursor, shape, state);
  if (!cursor.valid()) {
    return failure(Gfn2SccIterationArenaError::kSizeOverflow);
  }
  candidate.persistent_bytes = cursor.offset() - candidate.persistent_offset;

  if (!cursor.align(kGfn2SccIterationArenaAlignment)) {
    return failure(Gfn2SccIterationArenaError::kSizeOverflow);
  }
  candidate.workspace_offset = cursor.offset();
  Gfn2SccIterationDeviceWorkspace workspace{};
  Gfn2SccIterationReportStorage report_storage{};
  project_workspace(cursor, shape, plan, state, workspace, report_storage);
  if (!cursor.valid()) {
    return failure(Gfn2SccIterationArenaError::kSizeOverflow);
  }
  candidate.workspace_bytes = cursor.offset() - candidate.workspace_offset;

  if (shape.provider_device_bytes != 0u) {
    if (!cursor.align(kGfn2SccIterationArenaAlignment)) {
      return failure(Gfn2SccIterationArenaError::kSizeOverflow);
    }
    candidate.provider_device_offset = cursor.offset();
    (void)cursor.take_bytes(shape.provider_device_bytes, kGfn2SccIterationArenaAlignment);
    if (!cursor.valid()) {
      return failure(Gfn2SccIterationArenaError::kSizeOverflow);
    }
  } else {
    /* Empty segments consume no alignment padding but retain a stable end offset. */
    candidate.provider_device_offset = cursor.offset();
  }
  candidate.provider_device_bytes = shape.provider_device_bytes;
  if (!cursor.align(kGfn2SccIterationArenaAlignment)) {
    return failure(Gfn2SccIterationArenaError::kSizeOverflow);
  }
  candidate.total_bytes = cursor.offset();
  candidate.plan_token = shape.plan_token;
  candidate.layout_fingerprint = shape_fingerprint(shape);
  requirements = candidate;
  return {};
}

}  // namespace gpuxtb::detail::cuda

namespace gpuxtb::detail::cuda {
namespace {

[[nodiscard]] Gfn2EigensolverDeviceResults take_eigenpairs(ArenaCursor& cursor,
                                                           const ArenaShape& shape) noexcept {
  Gfn2EigensolverDeviceResults result{};
  result.eigenvalues = cursor.take<double>(shape.orbitals);
  result.eigenvalue_elements = shape.orbitals;
  result.coefficients = cursor.take<double>(shape.matrices);
  result.coefficient_elements = shape.matrices;
  result.plan_token = shape.plan_token;
  return result;
}

[[nodiscard]] Gfn2OccupationsDeviceResults take_occupations(ArenaCursor& cursor,
                                                            const ArenaShape& shape) noexcept {
  Gfn2OccupationsDeviceResults result{};
  result.occupations = cursor.take<double>(shape.two_orbitals);
  result.occupation_elements = shape.two_orbitals;
  result.chemical_potentials = cursor.take<double>(shape.two_batch);
  result.chemical_potential_elements = shape.two_batch;
  result.electron_sums = cursor.take<double>(shape.two_batch);
  result.electron_sum_elements = shape.two_batch;
  result.entropies = cursor.take<double>(shape.batch);
  result.entropy_elements = shape.batch;
  result.plan_token = shape.plan_token;
  return result;
}

[[nodiscard]] Gfn2DensityDeviceResults take_density(ArenaCursor& cursor,
                                                    const ArenaShape& shape) noexcept {
  Gfn2DensityDeviceResults result{};
  result.density = cursor.take<double>(shape.matrices);
  result.density_elements = shape.matrices;
  result.energy_weighted_density = cursor.take<double>(shape.matrices);
  result.weighted_density_elements = shape.matrices;
  result.band_energies = cursor.take<double>(shape.batch);
  result.band_energy_elements = shape.batch;
  result.occupation_sums = cursor.take<double>(shape.batch);
  result.occupation_sum_elements = shape.batch;
  result.density_traces = cursor.take<double>(shape.batch);
  result.density_trace_elements = shape.batch;
  result.weighted_density_traces = cursor.take<double>(shape.batch);
  result.weighted_density_trace_elements = shape.batch;
  result.plan_token = shape.plan_token;
  return result;
}

[[nodiscard]] Gfn2MullikenDevicePopulation take_population(ArenaCursor& cursor,
                                                           const ArenaShape& shape) noexcept {
  Gfn2MullikenDevicePopulation result{};
  result.qsh = cursor.take<double>(shape.shells);
  result.qsh_elements = shape.shells;
  result.qat = cursor.take<double>(shape.atoms);
  result.qat_elements = shape.atoms;
  result.dipole = cursor.take<double>(shape.dipoles);
  result.dipole_elements = shape.dipoles;
  result.quadrupole = cursor.take<double>(shape.quadrupoles);
  result.quadrupole_elements = shape.quadrupoles;
  result.plan_token = shape.plan_token;
  return result;
}

[[nodiscard]] Gfn2SccDeviceMultipoles take_multipoles(ArenaCursor& cursor,
                                                      const ArenaShape& shape) noexcept {
  Gfn2SccDeviceMultipoles result{};
  result.shell_charges = cursor.take<double>(shape.shells);
  result.shell_elements = shape.shells;
  result.atomic_dipoles = cursor.take<double>(shape.dipoles);
  result.dipole_elements = shape.dipoles;
  result.atomic_quadrupoles = cursor.take<double>(shape.quadrupoles);
  result.quadrupole_elements = shape.quadrupoles;
  result.plan_token = shape.plan_token;
  return result;
}

[[nodiscard]] Gfn2SccMixerDeviceState take_mixer(ArenaCursor& cursor,
                                                 const ArenaShape& shape) noexcept {
  Gfn2SccMixerDeviceState result{};
  result.current_inputs = cursor.take<double>(shape.mixer_vector);
  result.previous_inputs = cursor.take<double>(shape.mixer_vector);
  result.previous_residuals = cursor.take<double>(shape.mixer_vector);
  result.df_history = cursor.take<double>(shape.mixer_history_elements);
  result.u_history = cursor.take<double>(shape.mixer_history_elements);
  result.omega = cursor.take<double>(shape.mixer_omega_elements);
  result.residual_rms = cursor.take<double>(shape.batch);
  result.residual_maximum = cursor.take<double>(shape.batch);
  result.iterations = cursor.take<std::uint64_t>(shape.batch);
  result.restart_counts = cursor.take<std::uint64_t>(shape.batch);
  result.system_statuses = cursor.take<gpuxtb_status_t>(shape.batch);
  result.initialized = cursor.take<std::uint8_t>(shape.batch);
  result.residual_converged = cursor.take<std::uint8_t>(shape.batch);
  result.total_vector_elements = shape.mixer_vector;
  result.history_elements = shape.mixer_history_elements;
  result.omega_elements = shape.mixer_omega_elements;
  result.batch_elements = shape.batch;
  result.plan_token = shape.plan_token;
  return result;
}

void take_energy_trace(ArenaCursor& cursor, const ArenaShape& shape, double* entropy_alias,
                       bool alias_entropy, Gfn2SccClassicalEnergyDeviceDiagnostics& classical,
                       Gfn2SccFreeEnergyDeviceDiagnostics& free) noexcept {
  free = {};
  free.core = cursor.take<double>(shape.batch);
  free.core_elements = shape.batch;
  free.es2 = cursor.take<double>(shape.batch);
  free.es2_elements = shape.batch;
  free.es3 = cursor.take<double>(shape.batch);
  free.es3_elements = shape.batch;
  free.aes2 = cursor.take<double>(shape.batch);
  free.aes2_elements = shape.batch;
  free.d4_two_body = cursor.take<double>(shape.batch);
  free.d4_two_body_elements = shape.batch;
  free.explicit_point_charge = cursor.take<double>(shape.batch);
  free.explicit_point_charge_elements = shape.batch;
  free.periodic_embedding = cursor.take<double>(shape.batch);
  free.periodic_embedding_elements = shape.batch;
  /*
   * Sizing projections intentionally use a null arena base, so pointer
   * nullness cannot encode whether entropy owns storage or aliases the
   * occupation transaction. Keep that layout decision explicit.
   */
  free.entropy = alias_entropy ? entropy_alias : cursor.take<double>(shape.batch);
  free.entropy_elements = shape.batch;
  free.internal_energy = cursor.take<double>(shape.batch);
  free.internal_energy_elements = shape.batch;
  free.free_energy = cursor.take<double>(shape.batch);
  free.free_energy_elements = shape.batch;
  free.plan_token = shape.plan_token;

  classical = {};
  classical.es2 = free.es2;
  classical.es2_elements = shape.batch;
  classical.es3 = free.es3;
  classical.es3_elements = shape.batch;
  classical.aes2 = free.aes2;
  classical.aes2_elements = shape.batch;
  classical.d4_two_body = free.d4_two_body;
  classical.d4_two_body_elements = shape.batch;
  classical.explicit_point_charge = free.explicit_point_charge;
  classical.explicit_point_charge_elements = shape.batch;
  classical.periodic_embedding = free.periodic_embedding;
  classical.periodic_embedding_elements = shape.batch;
  classical.classical_total = cursor.take<double>(shape.batch);
  classical.classical_total_elements = shape.batch;
  classical.plan_token = shape.plan_token;
}

void project_persistent(ArenaCursor& cursor, const ArenaShape& shape,
                        Gfn2SccIterationDeviceState& state) noexcept {
  state = {};
  state.plan_token = shape.plan_token;
  state.eigenpairs = take_eigenpairs(cursor, shape);
  state.occupations = take_occupations(cursor, shape);
  state.density = take_density(cursor, shape);
  state.raw_population = take_population(cursor, shape);
  take_energy_trace(cursor, shape, nullptr, false, state.classical_energy, state.free_energy);
  state.mixer = take_mixer(cursor, shape);
  state.published = {state.raw_population.qsh,
                     shape.shells,
                     state.raw_population.dipole,
                     shape.dipoles,
                     state.raw_population.quadrupole,
                     shape.quadrupoles,
                     shape.plan_token};

  state.scc.current_inputs = take_multipoles(cursor, shape);
  state.scc.free_energies = cursor.take<double>(shape.batch);
  state.scc.previous_free_energies = cursor.take<double>(shape.batch);
  state.scc.free_energy_changes = cursor.take<double>(shape.batch);
  state.scc.residual_rms = cursor.take<double>(shape.batch);
  state.scc.iterations = cursor.take<std::uint64_t>(shape.batch);
  state.scc.system_statuses = cursor.take<gpuxtb_status_t>(shape.batch);
  state.scc.converged = cursor.take<std::uint8_t>(shape.batch);
  state.scc.batch_elements = shape.batch;
  state.scc.plan_token = shape.plan_token;

  state.publication.wavefunction = {state.eigenpairs, state.occupations, state.density,
                                    state.raw_population, shape.plan_token};
  state.publication.energy = {state.classical_energy, state.free_energy, shape.plan_token};
  state.publication.mixer = state.mixer;
  state.publication.published = state.published;
  state.publication.scc = state.scc;
  state.publication.plan_token = shape.plan_token;
}

void project_primitive_workspace_front(ArenaCursor& cursor, const ArenaShape& shape,
                                       Gfn2SccIterationDeviceWorkspace& workspace) noexcept {
  workspace.geometry_workspace = {cursor.take<double>(shape.geometry_pair_elements),
                                  shape.geometry_pair_elements,
                                  cursor.take<double>(shape.atoms),
                                  shape.atoms,
                                  cursor.take<double>(shape.dipoles),
                                  shape.dipoles,
                                  cursor.take<std::uint32_t>(1),
                                  1,
                                  shape.plan_token};

  workspace.es2_workspace = {cursor.take<double>(shape.es2_matrix_elements),
                             shape.es2_matrix_elements,
                             cursor.take<double>(shape.shells),
                             shape.shells,
                             cursor.take<double>(shape.batch),
                             shape.batch,
                             cursor.take<double>(shape.dipoles),
                             shape.dipoles};

  workspace.aes2_workspace = {cursor.take<double>(shape.aes2_pair_elements),
                              shape.aes2_pair_elements,
                              cursor.take<double>(shape.aes2_potential_elements),
                              shape.aes2_potential_elements,
                              cursor.take<double>(shape.batch),
                              shape.batch,
                              cursor.take<double>(shape.dipoles),
                              shape.dipoles,
                              cursor.take<double>(shape.atoms),
                              shape.atoms,
                              cursor.take<std::uint32_t>(1),
                              1};

  if (component_enabled(shape, Gfn2SccPotentialComponent::kD4TwoBody)) {
    auto& d4 = workspace.d4_workspace;
    d4.weights = cursor.take<double>(shape.d4_weight_elements);
    d4.weight_charge_derivatives = cursor.take<double>(shape.d4_weight_elements);
    d4.weight_elements = shape.d4_weight_elements;
    d4.atom_scratch = cursor.take<double>(shape.atoms);
    d4.atom_elements = shape.atoms;
    d4.batch_scratch = cursor.take<double>(shape.batch);
    d4.batch_elements = shape.batch;
    d4.system_errors = cursor.take<std::uint32_t>(shape.batch);
    d4.system_error_elements = shape.batch;
  }

  if (component_enabled(shape, Gfn2SccPotentialComponent::kPeriodicEmbedding)) {
    workspace.periodic_workspace = {cursor.take<double>(shape.atoms),
                                    cursor.take<double>(shape.atoms),
                                    cursor.take<std::uint32_t>(1),
                                    shape.atoms,
                                    1,
                                    shape.plan_token};
  }

  workspace.potential_workspace = {cursor.take<double>(shape.shells),
                                   shape.shells,
                                   cursor.take<double>(shape.atoms),
                                   shape.atoms,
                                   cursor.take<double>(shape.dipoles),
                                   shape.dipoles,
                                   cursor.take<double>(shape.quadrupoles),
                                   shape.quadrupoles,
                                   cursor.take<std::uint32_t>(1),
                                   1,
                                   shape.plan_token};
  workspace.scalar_bridge.workspace = {cursor.take<double>(shape.shells), shape.shells,
                                       cursor.take<std::uint32_t>(1), 1, shape.plan_token};
  workspace.hamiltonian_workspace = {cursor.take<double>(shape.matrices), shape.matrices,
                                     cursor.take<std::uint32_t>(1), 1, shape.plan_token};
}

void project_solver_and_wavefunction_workspace(
    ArenaCursor& cursor, const ArenaShape& shape, const Gfn2SccIterationDevicePlan& plan,
    Gfn2SccIterationDeviceWorkspace& workspace) noexcept {
  auto& eigensolver = workspace.eigensolver_workspace;
  eigensolver.matrix_scratch_a = cursor.take<double>(shape.matrices);
  eigensolver.matrix_a_elements = shape.matrices;
  eigensolver.matrix_scratch_b = cursor.take<double>(shape.matrices);
  eigensolver.matrix_b_elements = shape.matrices;
  eigensolver.eigenvalue_scratch = cursor.take<double>(shape.orbitals);
  eigensolver.eigenvalue_elements = shape.orbitals;
  eigensolver.factor_pointers = cursor.take<double*>(shape.batch);
  eigensolver.factor_pointer_elements = shape.batch;
  eigensolver.matrix_pointers = cursor.take<double*>(shape.batch);
  eigensolver.matrix_pointer_elements = shape.batch;
  eigensolver.info_a = cursor.take<int>(shape.batch);
  eigensolver.info_a_elements = shape.batch;
  eigensolver.info_b = cursor.take<int>(shape.batch);
  eigensolver.info_b_elements = shape.batch;
  eigensolver.eligible = cursor.take<std::uint8_t>(shape.batch);
  eigensolver.eligible_elements = shape.batch;
  eigensolver.sequence_active = cursor.take<std::uint32_t>(1);
  eigensolver.sequence_active_elements = 1;
  eigensolver.solver_device_workspace = plan.eigensolver_provider.device_workspace;
  eigensolver.solver_device_workspace_bytes = shape.provider_device_bytes;
  eigensolver.solver_host_workspace = plan.eigensolver_provider.host_workspace;
  eigensolver.solver_host_workspace_bytes = shape.provider_host_bytes;
  eigensolver.plan_token = shape.plan_token;

  workspace.occupations_workspace = {cursor.take<double>(shape.two_orbitals),
                                     shape.two_orbitals,
                                     cursor.take<double>(shape.two_batch),
                                     shape.two_batch,
                                     cursor.take<double>(shape.two_batch),
                                     shape.two_batch,
                                     cursor.take<double>(shape.batch),
                                     shape.batch,
                                     cursor.take<std::uint32_t>(1),
                                     1,
                                     shape.plan_token};
  workspace.density_workspace = {cursor.take<double>(shape.matrices),
                                 shape.matrices,
                                 cursor.take<double>(shape.matrices),
                                 shape.matrices,
                                 cursor.take<double>(shape.orbitals),
                                 shape.orbitals,
                                 cursor.take<double>(shape.orbitals),
                                 shape.orbitals,
                                 cursor.take<double>(shape.batch),
                                 shape.batch,
                                 cursor.take<double>(shape.batch),
                                 shape.batch,
                                 cursor.take<double>(shape.batch),
                                 shape.batch,
                                 cursor.take<double>(shape.batch),
                                 shape.batch,
                                 cursor.take<std::uint32_t>(1),
                                 1,
                                 shape.plan_token};
  workspace.mulliken_workspace = {cursor.take<double>(shape.shells),
                                  shape.shells,
                                  cursor.take<double>(shape.atoms),
                                  shape.atoms,
                                  cursor.take<double>(shape.dipoles),
                                  shape.dipoles,
                                  cursor.take<double>(shape.quadrupoles),
                                  shape.quadrupoles,
                                  cursor.take<std::uint32_t>(1),
                                  1,
                                  shape.plan_token};
}

void project_energy_mixer_and_publication_workspace(
    ArenaCursor& cursor, const ArenaShape& shape,
    Gfn2SccIterationDeviceWorkspace& workspace) noexcept {
  workspace.electronic_energy_workspace = {cursor.take<double>(shape.batch),
                                           cursor.take<double>(shape.batch),
                                           cursor.take<std::uint32_t>(1),
                                           shape.batch,
                                           1,
                                           shape.plan_token};
  workspace.classical_energy_workspace = {cursor.take<double>(shape.classical_scratch_elements),
                                          shape.classical_scratch_elements,
                                          cursor.take<std::uint32_t>(1), 1, shape.plan_token};
  workspace.free_energy_workspace = {cursor.take<double>(shape.free_scratch_elements),
                                     shape.free_scratch_elements, cursor.take<std::uint32_t>(1), 1,
                                     shape.plan_token};

  auto& mixer = workspace.mixer_workspace;
  mixer.residual = cursor.take<double>(shape.mixer_vector);
  mixer.mixed = cursor.take<double>(shape.mixer_vector);
  mixer.delta_f = cursor.take<double>(shape.mixer_vector);
  mixer.new_u = cursor.take<double>(shape.mixer_vector);
  mixer.beta = cursor.take<double>(shape.mixer_beta_elements);
  mixer.coefficients = cursor.take<double>(shape.mixer_coefficient_elements);
  mixer.sequence_active = cursor.take<std::uint32_t>(1);
  mixer.vector_elements = shape.mixer_vector;
  mixer.beta_elements = shape.mixer_beta_elements;
  mixer.coefficient_elements = shape.mixer_coefficient_elements;
  mixer.sequence_elements = 1;
  mixer.plan_token = shape.plan_token;
  workspace.mixer_device_error = cursor.take<std::uint32_t>(1);
  workspace.mixer_device_error_elements = 1;

  auto& publication = workspace.publication_workspace;
  publication.mixed_atomic_charges = workspace.mixed_topology.atomic_charges;
  publication.mixed_atomic_charge_elements = shape.atoms;
  publication.previous_free_energies = cursor.take<double>(shape.batch);
  publication.free_energy_changes = cursor.take<double>(shape.batch);
  publication.next_iterations = cursor.take<std::uint64_t>(shape.batch);
  publication.next_statuses = cursor.take<gpuxtb_status_t>(shape.batch);
  publication.next_converged = cursor.take<std::uint8_t>(shape.batch);
  publication.batch_elements = shape.batch;
  publication.system_errors = cursor.take<std::uint32_t>(shape.batch);
  publication.system_error_elements = shape.batch;
  publication.device_error = cursor.take<std::uint32_t>(1);
  publication.device_error_elements = 1;
  publication.sequence_active = cursor.take<std::uint32_t>(1);
  publication.sequence_elements = 1;
  publication.plan_token = shape.plan_token;
}

void project_workspace(ArenaCursor& cursor, const ArenaShape& shape,
                       const Gfn2SccIterationDevicePlan& plan,
                       const Gfn2SccIterationDeviceState& state,
                       Gfn2SccIterationDeviceWorkspace& workspace,
                       Gfn2SccIterationReportStorage& report_storage) noexcept {
  project_workspace_front(cursor, shape, plan, state, workspace);
  project_primitive_workspace_front(cursor, shape, workspace);
  project_solver_and_wavefunction_workspace(cursor, shape, plan, workspace);
  project_energy_mixer_and_publication_workspace(cursor, shape, workspace);

  /*
   * Report storage deliberately reserves one canonical slot per report even
   * where the report factory selects a primitive-owned diagnostic. This keeps
   * offsets stable across setup and makes the unused capacity explicit.
   */
  report_storage = {cursor.take<std::uint32_t>(shape.reports.system_error_elements),
                    shape.reports.system_error_elements,
                    cursor.take<std::uint32_t>(shape.reports.device_error_elements),
                    shape.reports.device_error_elements,
                    cursor.take<std::uint32_t>(shape.reports.sequence_latch_elements),
                    shape.reports.sequence_latch_elements,
                    shape.plan_token};
}

}  // namespace
}  // namespace gpuxtb::detail::cuda

namespace gpuxtb::detail::cuda {

Gfn2SccIterationArenaDiagnostic bind_gfn2_scc_iteration_arena_cuda(
    Gfn2SccIterationDevicePlan& plan,
    const Gfn2EigensolverWorkspaceRequirements& provider_requirements,
    const Gfn2SccIterationArenaRequirements& requirements, void* arena, std::size_t arena_bytes,
    void* provider_host_workspace, std::size_t provider_host_workspace_bytes,
    Gfn2SccIterationDeviceState& state, Gfn2SccIterationDeviceWorkspace& workspace,
    Gfn2SccIterationReportStorage& report_storage) noexcept {
  /*
   * Descriptor publication is transactional. In particular, a failed rebind
   * must not leave a mixture of old state and newly projected arena ranges.
   */
  state = {};
  workspace = {};
  report_storage = {};

  Gfn2SccIterationArenaRequirements current{};
  const auto query_diagnostic =
      query_gfn2_scc_iteration_arena_requirements_cuda(plan, provider_requirements, current);
  if (!query_diagnostic.success()) {
    return query_diagnostic;
  }
  if (!same_requirements(requirements, current)) {
    return failure(Gfn2SccIterationArenaError::kStaleRequirements, current.total_bytes,
                   requirements.total_bytes);
  }
  if (arena == nullptr) {
    return failure(Gfn2SccIterationArenaError::kNullArena, current.total_bytes, arena_bytes);
  }

  const auto arena_address = reinterpret_cast<std::uintptr_t>(arena);
  if (arena_address % kGfn2SccIterationArenaAlignment != 0u) {
    return failure(Gfn2SccIterationArenaError::kMisalignedArena, current.total_bytes, arena_bytes);
  }
  if (arena_bytes < current.total_bytes) {
    return failure(Gfn2SccIterationArenaError::kInsufficientArena, current.total_bytes,
                   arena_bytes);
  }
  if (current.total_bytes != 0u &&
      current.total_bytes - 1u > std::numeric_limits<std::uintptr_t>::max() - arena_address) {
    return failure(Gfn2SccIterationArenaError::kSizeOverflow, current.total_bytes, arena_bytes);
  }

  const std::size_t required_host_bytes = provider_requirements.solver_host_workspace_bytes;
  if ((required_host_bytes != 0u && provider_host_workspace == nullptr) ||
      provider_host_workspace_bytes < required_host_bytes ||
      (required_host_bytes != 0u &&
       reinterpret_cast<std::uintptr_t>(provider_host_workspace) % alignof(std::max_align_t) !=
           0u)) {
    return failure(Gfn2SccIterationArenaError::kInvalidProviderHostWorkspace, required_host_bytes,
                   provider_host_workspace_bytes);
  }

  /*
   * Bind provider workspaces in a private plan copy first. Handles, buckets,
   * and immutable cache pointers remain untouched and outside this arena.
   */
  Gfn2SccIterationDevicePlan candidate_plan = plan;
  candidate_plan.eigensolver_provider.requirements = provider_requirements;
  auto* const arena_bytes_base = static_cast<std::byte*>(arena);
  candidate_plan.eigensolver_provider.device_workspace =
      current.provider_device_bytes == 0u
          ? nullptr
          : static_cast<void*>(arena_bytes_base + current.provider_device_offset);
  candidate_plan.eigensolver_provider.device_workspace_bytes = current.provider_device_bytes;
  candidate_plan.eigensolver_provider.host_workspace =
      required_host_bytes == 0u ? nullptr : provider_host_workspace;
  candidate_plan.eigensolver_provider.host_workspace_bytes = required_host_bytes;

  Gfn2SccIterationDeviceState candidate_state{};
  Gfn2SccIterationDeviceWorkspace candidate_workspace{};
  Gfn2SccIterationReportStorage candidate_reports{};
  ArenaShape shape{};
  Gfn2SccIterationArenaError shape_error = Gfn2SccIterationArenaError::kInvalidPlan;
  if (!derive_shape(candidate_plan, provider_requirements, shape, shape_error)) {
    return failure(shape_error);
  }

  ArenaCursor cursor(arena_bytes_base);
  if (cursor.offset() != current.persistent_offset) {
    return failure(Gfn2SccIterationArenaError::kStaleRequirements, current.total_bytes,
                   requirements.total_bytes);
  }
  project_persistent(cursor, shape, candidate_state);
  if (!cursor.valid() || cursor.offset() - current.persistent_offset != current.persistent_bytes ||
      !cursor.align(kGfn2SccIterationArenaAlignment) ||
      cursor.offset() != current.workspace_offset) {
    return failure(Gfn2SccIterationArenaError::kStaleRequirements, current.total_bytes,
                   requirements.total_bytes);
  }
  project_workspace(cursor, shape, candidate_plan, candidate_state, candidate_workspace,
                    candidate_reports);
  if (!cursor.valid() || cursor.offset() - current.workspace_offset != current.workspace_bytes) {
    return failure(Gfn2SccIterationArenaError::kStaleRequirements, current.total_bytes,
                   requirements.total_bytes);
  }

  if (current.provider_device_bytes != 0u) {
    if (!cursor.align(kGfn2SccIterationArenaAlignment) ||
        cursor.offset() != current.provider_device_offset ||
        cursor.take_bytes(current.provider_device_bytes, kGfn2SccIterationArenaAlignment) !=
            candidate_plan.eigensolver_provider.device_workspace) {
      return failure(Gfn2SccIterationArenaError::kStaleRequirements, current.total_bytes,
                     requirements.total_bytes);
    }
  } else if (cursor.offset() != current.provider_device_offset) {
    return failure(Gfn2SccIterationArenaError::kStaleRequirements, current.total_bytes,
                   requirements.total_bytes);
  }
  if (!cursor.align(kGfn2SccIterationArenaAlignment) || !cursor.valid() ||
      cursor.offset() != current.total_bytes) {
    return failure(Gfn2SccIterationArenaError::kStaleRequirements, current.total_bytes,
                   requirements.total_bytes);
  }

  plan = candidate_plan;
  state = candidate_state;
  workspace = candidate_workspace;
  report_storage = candidate_reports;
  return {};
}

}  // namespace gpuxtb::detail::cuda
