#include <array>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "backends/cuda/gfn2_scc_iteration_test.cuh"

namespace xtbloom::detail::cuda {
namespace {

using BindingError = Gfn2SccIterationBindingError;
using BindingField = Gfn2SccIterationBindingField;

constexpr std::uint32_t kMandatoryPotentialComponents =
    static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES2) |
    static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES3) |
    static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kAES2);

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
  bool writable = false;
  std::uint32_t alias_group = 0u;
  BindingField field = BindingField::kNone;
  std::int64_t index = -1;
};

class Validator {
 public:
  [[nodiscard]] bool success() const noexcept {
    return diagnostic_.error == BindingError::kSuccess;
  }

  [[nodiscard]] Gfn2SccIterationBindingDiagnostic diagnostic() const noexcept {
    return diagnostic_;
  }

  bool fail(BindingError error, BindingField field, std::int64_t index = -1) noexcept {
    if (success()) {
      diagnostic_ = {error, field, index};
    }
    return false;
  }

  bool exact_count(std::int64_t actual, std::int64_t expected, BindingField field,
                   std::int64_t index = -1) noexcept {
    if (actual < 0 || expected < 0) {
      return fail(BindingError::kInvalidCount, field, index);
    }
    return actual == expected || fail(BindingError::kInvalidCount, field, index);
  }

  bool capacity(std::int64_t actual, std::int64_t required, BindingField field,
                std::int64_t index = -1) noexcept {
    if (actual < 0 || required < 0) {
      return fail(BindingError::kInvalidCount, field, index);
    }
    return actual >= required || fail(BindingError::kInsufficientCapacity, field, index);
  }

  bool token(std::uint64_t actual, std::uint64_t expected, BindingField field,
             std::int64_t index = -1) noexcept {
    return actual == expected || fail(BindingError::kCrossPlan, field, index);
  }

  bool pointer(const void* value, std::int64_t elements, std::size_t element_size,
               std::size_t alignment, BindingField field, std::int64_t index, bool writable,
               std::uint32_t alias_group = 0u) noexcept {
    if (elements < 0 || element_size == 0u) {
      return fail(BindingError::kInvalidCount, field, index);
    }
    if (elements == 0) {
      return value == nullptr || fail(BindingError::kNullPointer, field, index);
    }
    if (value == nullptr) {
      return fail(BindingError::kNullPointer, field, index);
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(value);
    if (alignment == 0u || begin % alignment != 0u) {
      return fail(BindingError::kMisalignedPointer, field, index);
    }
    if (static_cast<std::uint64_t>(elements) >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
      return fail(BindingError::kAddressOverflow, field, index);
    }
    const std::size_t bytes = static_cast<std::size_t>(elements) * element_size;
    if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
      return fail(BindingError::kAddressOverflow, field, index);
    }
    if (range_count_ >= ranges_.size()) {
      return fail(BindingError::kInvalidCount, BindingField::kWorkspace);
    }
    ranges_[range_count_++] = {begin, begin + bytes, writable, alias_group, field, index};
    return true;
  }

  bool aliases_valid() noexcept {
    for (std::size_t first = 0u; first < range_count_; ++first) {
      for (std::size_t second = first + 1u; second < range_count_; ++second) {
        const AddressRange& lhs = ranges_[first];
        const AddressRange& rhs = ranges_[second];
        if (!(lhs.begin < rhs.end && rhs.begin < lhs.end) || (!lhs.writable && !rhs.writable)) {
          continue;
        }
        const bool exact = lhs.begin == rhs.begin && lhs.end == rhs.end;
        const bool permitted_exact =
            exact && lhs.alias_group != 0u && lhs.alias_group == rhs.alias_group;
        if (!permitted_exact) {
          return fail(BindingError::kForbiddenAlias, rhs.field, rhs.index);
        }
      }
    }
    return true;
  }

 private:
  Gfn2SccIterationBindingDiagnostic diagnostic_{};
  std::array<AddressRange, 512> ranges_{};
  std::size_t range_count_ = 0u;
};

bool checked_add(std::int64_t first, std::int64_t second, std::int64_t& result) noexcept {
  if (first < 0 || second < 0 || first > std::numeric_limits<std::int64_t>::max() - second) {
    return false;
  }
  result = first + second;
  return true;
}

bool checked_multiply(std::int64_t value, std::int64_t factor, std::int64_t& result) noexcept {
  if (value < 0 || factor < 0 ||
      (value != 0 && factor > std::numeric_limits<std::int64_t>::max() / value)) {
    return false;
  }
  result = value * factor;
  return true;
}

template <typename T, typename U>
bool same_pointer(const T* first, const U* second) noexcept {
  return static_cast<const void*>(first) == static_cast<const void*>(second);
}

bool same_wavefunction_layout(const Gfn2WavefunctionLayoutView& first,
                              const Gfn2WavefunctionLayoutView& second) noexcept {
  return first.memory_space == second.memory_space && first.plan_token == second.plan_token &&
         first.layout_fingerprint == second.layout_fingerprint &&
         first.batch_size == second.batch_size &&
         first.total_spin_channels == second.total_spin_channels &&
         first.total_spin_orbitals == second.total_spin_orbitals &&
         first.total_spin_matrix_elements == second.total_spin_matrix_elements &&
         first.total_spin_shells == second.total_spin_shells &&
         first.total_spin_atoms == second.total_spin_atoms &&
         first.spin_channel_count == second.spin_channel_count &&
         first.spin_channel_offset_count == second.spin_channel_offset_count &&
         first.spin_orbital_offset_count == second.spin_orbital_offset_count &&
         first.spin_matrix_offset_count == second.spin_matrix_offset_count &&
         first.spin_shell_offset_count == second.spin_shell_offset_count &&
         first.spin_atom_offset_count == second.spin_atom_offset_count &&
         first.spin_channels == second.spin_channels &&
         first.spin_channel_offsets == second.spin_channel_offsets &&
         first.spin_orbital_offsets == second.spin_orbital_offsets &&
         first.spin_matrix_offsets == second.spin_matrix_offsets &&
         first.spin_shell_offsets == second.spin_shell_offsets &&
         first.spin_atom_offsets == second.spin_atom_offsets;
}

bool same_topology_view(const Gfn2RaggedTopologyView& first,
                        const Gfn2RaggedTopologyView& second) noexcept {
  return first.memory_space == second.memory_space && first.pair_map_kind == second.pair_map_kind &&
         first.plan_token == second.plan_token && first.batch_size == second.batch_size &&
         first.total_atoms == second.total_atoms && first.total_shells == second.total_shells &&
         first.total_orbitals == second.total_orbitals &&
         first.total_matrix_elements == second.total_matrix_elements &&
         first.total_pairs == second.total_pairs && first.bucket_count == second.bucket_count &&
         first.atom_offset_count == second.atom_offset_count &&
         first.batch_shell_offset_count == second.batch_shell_offset_count &&
         first.batch_orbital_offset_count == second.batch_orbital_offset_count &&
         first.matrix_offset_count == second.matrix_offset_count &&
         first.atom_shell_offset_count == second.atom_shell_offset_count &&
         first.shell_orbital_offset_count == second.shell_orbital_offset_count &&
         first.shell_to_atom_count == second.shell_to_atom_count &&
         first.orbital_to_shell_count == second.orbital_to_shell_count &&
         first.orbital_to_atom_count == second.orbital_to_atom_count &&
         first.pair_offset_count == second.pair_offset_count &&
         first.atom_pair_count == second.atom_pair_count &&
         first.bucket_offset_count == second.bucket_offset_count &&
         first.bucket_system_count == second.bucket_system_count &&
         first.bucket_orbital_count == second.bucket_orbital_count &&
         first.atom_offsets == second.atom_offsets &&
         first.batch_shell_offsets == second.batch_shell_offsets &&
         first.batch_orbital_offsets == second.batch_orbital_offsets &&
         first.matrix_offsets == second.matrix_offsets &&
         first.atom_shell_offsets == second.atom_shell_offsets &&
         first.shell_orbital_offsets == second.shell_orbital_offsets &&
         first.shell_to_atom == second.shell_to_atom &&
         first.orbital_to_shell == second.orbital_to_shell &&
         first.orbital_to_atom == second.orbital_to_atom &&
         first.pair_offsets == second.pair_offsets && first.atom_pairs == second.atom_pairs &&
         first.bucket_offsets == second.bucket_offsets &&
         first.bucket_systems == second.bucket_systems &&
         first.bucket_orbital_counts == second.bucket_orbital_counts;
}

bool same_eigenpairs(const Gfn2EigensolverDeviceResults& first,
                     const Gfn2EigensolverDeviceResults& second) noexcept {
  return first.eigenvalues == second.eigenvalues &&
         first.eigenvalue_elements == second.eigenvalue_elements &&
         first.coefficients == second.coefficients &&
         first.coefficient_elements == second.coefficient_elements &&
         first.plan_token == second.plan_token;
}

bool same_occupations(const Gfn2OccupationsDeviceResults& first,
                      const Gfn2OccupationsDeviceResults& second) noexcept {
  return first.occupations == second.occupations &&
         first.occupation_elements == second.occupation_elements &&
         first.chemical_potentials == second.chemical_potentials &&
         first.chemical_potential_elements == second.chemical_potential_elements &&
         first.electron_sums == second.electron_sums &&
         first.electron_sum_elements == second.electron_sum_elements &&
         first.entropies == second.entropies && first.entropy_elements == second.entropy_elements &&
         first.plan_token == second.plan_token;
}

bool same_density(const Gfn2DensityDeviceResults& first,
                  const Gfn2DensityDeviceResults& second) noexcept {
  return first.density == second.density && first.density_elements == second.density_elements &&
         first.energy_weighted_density == second.energy_weighted_density &&
         first.weighted_density_elements == second.weighted_density_elements &&
         first.band_energies == second.band_energies &&
         first.band_energy_elements == second.band_energy_elements &&
         first.occupation_sums == second.occupation_sums &&
         first.occupation_sum_elements == second.occupation_sum_elements &&
         first.density_traces == second.density_traces &&
         first.density_trace_elements == second.density_trace_elements &&
         first.weighted_density_traces == second.weighted_density_traces &&
         first.weighted_density_trace_elements == second.weighted_density_trace_elements &&
         first.channel_band_energies == second.channel_band_energies &&
         first.channel_band_energy_elements == second.channel_band_energy_elements &&
         first.channel_occupation_sums == second.channel_occupation_sums &&
         first.channel_occupation_sum_elements == second.channel_occupation_sum_elements &&
         first.channel_density_traces == second.channel_density_traces &&
         first.channel_density_trace_elements == second.channel_density_trace_elements &&
         first.channel_weighted_density_traces == second.channel_weighted_density_traces &&
         first.channel_weighted_density_trace_elements ==
             second.channel_weighted_density_trace_elements &&
         first.plan_token == second.plan_token;
}

bool same_population(const Gfn2MullikenDevicePopulation& first,
                     const Gfn2MullikenDevicePopulation& second) noexcept {
  return first.qsh == second.qsh && first.qsh_elements == second.qsh_elements &&
         first.qat == second.qat && first.qat_elements == second.qat_elements &&
         first.dipole == second.dipole && first.dipole_elements == second.dipole_elements &&
         first.quadrupole == second.quadrupole &&
         first.quadrupole_elements == second.quadrupole_elements &&
         first.plan_token == second.plan_token;
}

bool same_classical_diagnostics(const Gfn2SccClassicalEnergyDeviceDiagnostics& first,
                                const Gfn2SccClassicalEnergyDeviceDiagnostics& second) noexcept {
  return first.es2 == second.es2 && first.es2_elements == second.es2_elements &&
         first.es3 == second.es3 && first.es3_elements == second.es3_elements &&
         first.aes2 == second.aes2 && first.aes2_elements == second.aes2_elements &&
         first.d4_two_body == second.d4_two_body &&
         first.d4_two_body_elements == second.d4_two_body_elements &&
         first.explicit_point_charge == second.explicit_point_charge &&
         first.explicit_point_charge_elements == second.explicit_point_charge_elements &&
         first.periodic_embedding == second.periodic_embedding &&
         first.periodic_embedding_elements == second.periodic_embedding_elements &&
         first.classical_total == second.classical_total &&
         first.classical_total_elements == second.classical_total_elements &&
         first.plan_token == second.plan_token;
}

bool same_free_energy_diagnostics(const Gfn2SccFreeEnergyDeviceDiagnostics& first,
                                  const Gfn2SccFreeEnergyDeviceDiagnostics& second) noexcept {
  return first.core == second.core && first.core_elements == second.core_elements &&
         first.es2 == second.es2 && first.es2_elements == second.es2_elements &&
         first.es3 == second.es3 && first.es3_elements == second.es3_elements &&
         first.aes2 == second.aes2 && first.aes2_elements == second.aes2_elements &&
         first.spin == second.spin && first.spin_elements == second.spin_elements &&
         first.d4_two_body == second.d4_two_body &&
         first.d4_two_body_elements == second.d4_two_body_elements &&
         first.explicit_point_charge == second.explicit_point_charge &&
         first.explicit_point_charge_elements == second.explicit_point_charge_elements &&
         first.periodic_embedding == second.periodic_embedding &&
         first.periodic_embedding_elements == second.periodic_embedding_elements &&
         first.entropy == second.entropy && first.entropy_elements == second.entropy_elements &&
         first.internal_energy == second.internal_energy &&
         first.internal_energy_elements == second.internal_energy_elements &&
         first.free_energy == second.free_energy &&
         first.free_energy_elements == second.free_energy_elements &&
         first.plan_token == second.plan_token;
}

bool same_mixer_state(const Gfn2SccMixerDeviceState& first,
                      const Gfn2SccMixerDeviceState& second) noexcept {
  return first.current_inputs == second.current_inputs &&
         first.previous_inputs == second.previous_inputs &&
         first.previous_residuals == second.previous_residuals &&
         first.df_history == second.df_history && first.u_history == second.u_history &&
         first.omega == second.omega && first.residual_rms == second.residual_rms &&
         first.residual_maximum == second.residual_maximum &&
         first.iterations == second.iterations && first.restart_counts == second.restart_counts &&
         first.system_statuses == second.system_statuses &&
         first.initialized == second.initialized &&
         first.residual_converged == second.residual_converged &&
         first.total_vector_elements == second.total_vector_elements &&
         first.history_elements == second.history_elements &&
         first.omega_elements == second.omega_elements &&
         first.batch_elements == second.batch_elements && first.plan_token == second.plan_token;
}

bool same_multipoles(const Gfn2SccDeviceMultipoles& first,
                     const Gfn2SccDeviceMultipoles& second) noexcept {
  return first.shell_charges == second.shell_charges &&
         first.shell_elements == second.shell_elements &&
         first.atomic_dipoles == second.atomic_dipoles &&
         first.dipole_elements == second.dipole_elements &&
         first.atomic_quadrupoles == second.atomic_quadrupoles &&
         first.quadrupole_elements == second.quadrupole_elements &&
         first.plan_token == second.plan_token;
}

bool same_const_multipoles(const Gfn2SccDeviceConstMultipoles& first,
                           const Gfn2SccDeviceMultipoles& second) noexcept {
  return first.shell_charges == second.shell_charges &&
         first.shell_elements == second.shell_elements &&
         first.atomic_dipoles == second.atomic_dipoles &&
         first.dipole_elements == second.dipole_elements &&
         first.atomic_quadrupoles == second.atomic_quadrupoles &&
         first.quadrupole_elements == second.quadrupole_elements &&
         first.plan_token == second.plan_token;
}

bool same_scc_state(const Gfn2SccDeviceState& first, const Gfn2SccDeviceState& second) noexcept {
  return same_multipoles(first.current_inputs, second.current_inputs) &&
         first.free_energies == second.free_energies &&
         first.previous_free_energies == second.previous_free_energies &&
         first.free_energy_changes == second.free_energy_changes &&
         first.residual_rms == second.residual_rms && first.iterations == second.iterations &&
         first.system_statuses == second.system_statuses && first.converged == second.converged &&
         first.batch_elements == second.batch_elements && first.plan_token == second.plan_token;
}

bool component_enabled(const Gfn2SccIterationDevicePlan& plan,
                       Gfn2SccPotentialComponent component) noexcept {
  return (plan.enabled_components & static_cast<std::uint32_t>(component)) != 0u;
}

bool validate_plan_tokens(const Gfn2SccIterationDevicePlan& plan,
                          const Gfn2SccIterationDeviceInput& input,
                          const Gfn2SccIterationDeviceState& state,
                          const Gfn2SccIterationDeviceWorkspace& workspace,
                          Validator& validator) noexcept {
  const std::uint64_t token = plan.plan_token;
  if (token == 0u) {
    return validator.fail(BindingError::kInvalidPlanToken, BindingField::kPlan);
  }
#define XTBLOOM_CHECK_TOKEN(value, field) \
  if (!validator.token((value), token, (field))) return false
  XTBLOOM_CHECK_TOKEN(plan.topology.plan_token, BindingField::kTopology);
  XTBLOOM_CHECK_TOKEN(plan.wavefunction_layout.plan_token, BindingField::kSpin);
  XTBLOOM_CHECK_TOKEN(plan.activity_policy.plan_token, BindingField::kActivity);
  XTBLOOM_CHECK_TOKEN(plan.state_policy.plan_token, BindingField::kStatePublication);
  XTBLOOM_CHECK_TOKEN(plan.mixer_policy.plan_token, BindingField::kMixer);
  XTBLOOM_CHECK_TOKEN(plan.provenance.plan_token, BindingField::kActivity);
  XTBLOOM_CHECK_TOKEN(plan.geometry_batch.plan_token, BindingField::kGeometry);
  XTBLOOM_CHECK_TOKEN(plan.geometry_cache.plan_token, BindingField::kGeometry);
  XTBLOOM_CHECK_TOKEN(plan.scc_batch.plan_token, BindingField::kStatePublication);
  XTBLOOM_CHECK_TOKEN(plan.potential_batch.plan_token, BindingField::kPotential);
  XTBLOOM_CHECK_TOKEN(plan.spin_batch.plan_token, BindingField::kSpin);
  XTBLOOM_CHECK_TOKEN(plan.es2_batch.plan_token, BindingField::kES2);
  XTBLOOM_CHECK_TOKEN(plan.es2_cache.plan_token, BindingField::kES2);
  XTBLOOM_CHECK_TOKEN(plan.es3_batch.plan_token, BindingField::kES3);
  XTBLOOM_CHECK_TOKEN(plan.aes2_batch.plan_token, BindingField::kAES2);
  XTBLOOM_CHECK_TOKEN(plan.aes2_cache.plan_token, BindingField::kAES2);
  if (component_enabled(plan, Gfn2SccPotentialComponent::kD4TwoBody)) {
    XTBLOOM_CHECK_TOKEN(plan.d4_batch.plan_token, BindingField::kD4);
    XTBLOOM_CHECK_TOKEN(plan.d4_cache.plan_token, BindingField::kD4);
  }
  if (component_enabled(plan, Gfn2SccPotentialComponent::kExplicitPointCharge)) {
    XTBLOOM_CHECK_TOKEN(plan.explicit_point_charge_batch.plan_token,
                        BindingField::kExplicitPointCharge);
    XTBLOOM_CHECK_TOKEN(plan.explicit_point_charge_cache.plan_token,
                        BindingField::kExplicitPointCharge);
  }
  if (component_enabled(plan, Gfn2SccPotentialComponent::kPeriodicEmbedding)) {
    XTBLOOM_CHECK_TOKEN(plan.periodic_batch.plan_token, BindingField::kPeriodicEmbedding);
  }
  XTBLOOM_CHECK_TOKEN(plan.scalar_bridge_batch.topology.plan_token, BindingField::kScalarBridge);
  XTBLOOM_CHECK_TOKEN(plan.hamiltonian_batch.plan_token, BindingField::kHamiltonian);
  XTBLOOM_CHECK_TOKEN(plan.eigensolver_batch.plan_token, BindingField::kEigensolver);
  XTBLOOM_CHECK_TOKEN(plan.overlap_cache.plan_token, BindingField::kEigensolver);
  XTBLOOM_CHECK_TOKEN(plan.eigensolver_provider.plan_token, BindingField::kEigensolver);
  XTBLOOM_CHECK_TOKEN(plan.occupations_batch.plan_token, BindingField::kOccupations);
  XTBLOOM_CHECK_TOKEN(plan.density_batch.plan_token, BindingField::kDensity);
  XTBLOOM_CHECK_TOKEN(plan.mulliken_batch.plan_token, BindingField::kMulliken);
  XTBLOOM_CHECK_TOKEN(plan.electronic_energy_batch.plan_token, BindingField::kElectronicEnergy);
  XTBLOOM_CHECK_TOKEN(plan.classical_energy_batch.plan_token, BindingField::kClassicalEnergy);
  XTBLOOM_CHECK_TOKEN(plan.free_energy_batch.plan_token, BindingField::kFreeEnergy);
  XTBLOOM_CHECK_TOKEN(plan.publication_plan.plan_token, BindingField::kStatePublication);

  XTBLOOM_CHECK_TOKEN(input.plan_token, BindingField::kPlan);
  XTBLOOM_CHECK_TOKEN(input.activity_state.plan_token, BindingField::kActivity);
  XTBLOOM_CHECK_TOKEN(input.mixed_fields.plan_token, BindingField::kPotential);
  XTBLOOM_CHECK_TOKEN(input.mixed_spin.plan_token, BindingField::kSpin);
  XTBLOOM_CHECK_TOKEN(input.raw_spin.plan_token, BindingField::kSpin);
  XTBLOOM_CHECK_TOKEN(input.hamiltonian.plan_token, BindingField::kHamiltonian);
  XTBLOOM_CHECK_TOKEN(input.density.plan_token, BindingField::kDensity);
  XTBLOOM_CHECK_TOKEN(input.mulliken.plan_token, BindingField::kMulliken);
  XTBLOOM_CHECK_TOKEN(input.electronic_energy.plan_token, BindingField::kElectronicEnergy);
  XTBLOOM_CHECK_TOKEN(input.classical_energy.plan_token, BindingField::kClassicalEnergy);
  XTBLOOM_CHECK_TOKEN(input.free_energy.plan_token, BindingField::kFreeEnergy);
  XTBLOOM_CHECK_TOKEN(input.raw_multipoles.plan_token, BindingField::kMulliken);

  XTBLOOM_CHECK_TOKEN(state.plan_token, BindingField::kPlan);
  XTBLOOM_CHECK_TOKEN(state.eigenpairs.plan_token, BindingField::kEigensolver);
  XTBLOOM_CHECK_TOKEN(state.occupations.plan_token, BindingField::kOccupations);
  XTBLOOM_CHECK_TOKEN(state.density.plan_token, BindingField::kDensity);
  XTBLOOM_CHECK_TOKEN(state.raw_population.plan_token, BindingField::kMulliken);
  XTBLOOM_CHECK_TOKEN(state.classical_energy.plan_token, BindingField::kClassicalEnergy);
  XTBLOOM_CHECK_TOKEN(state.free_energy.plan_token, BindingField::kFreeEnergy);
  XTBLOOM_CHECK_TOKEN(state.mixer.plan_token, BindingField::kMixer);
  XTBLOOM_CHECK_TOKEN(state.published.plan_token, BindingField::kStatePublication);
  XTBLOOM_CHECK_TOKEN(state.scc.plan_token, BindingField::kStatePublication);
  XTBLOOM_CHECK_TOKEN(state.scc.current_inputs.plan_token, BindingField::kStatePublication);
  XTBLOOM_CHECK_TOKEN(state.publication.plan_token, BindingField::kStatePublication);
  XTBLOOM_CHECK_TOKEN(state.publication.wavefunction.plan_token, BindingField::kStatePublication);
  XTBLOOM_CHECK_TOKEN(state.publication.energy.plan_token, BindingField::kStatePublication);

  XTBLOOM_CHECK_TOKEN(workspace.plan_token, BindingField::kWorkspace);
  XTBLOOM_CHECK_TOKEN(workspace.ledger.plan_token, BindingField::kActivity);
  XTBLOOM_CHECK_TOKEN(workspace.activity.plan_token, BindingField::kActivity);
  XTBLOOM_CHECK_TOKEN(workspace.potential_activity.plan_token, BindingField::kPotential);
  XTBLOOM_CHECK_TOKEN(workspace.hamiltonian_activity.plan_token, BindingField::kHamiltonian);
  XTBLOOM_CHECK_TOKEN(workspace.mulliken_activity.plan_token, BindingField::kMulliken);
  XTBLOOM_CHECK_TOKEN(workspace.classical_energy_activity.plan_token,
                      BindingField::kClassicalEnergy);
  XTBLOOM_CHECK_TOKEN(workspace.free_energy_activity.plan_token, BindingField::kFreeEnergy);
  XTBLOOM_CHECK_TOKEN(workspace.mixed_topology.plan_token, BindingField::kPotential);
  XTBLOOM_CHECK_TOKEN(workspace.physical_topology.plan_token, BindingField::kPotential);
  XTBLOOM_CHECK_TOKEN(workspace.components.plan_token, BindingField::kPotential);
  XTBLOOM_CHECK_TOKEN(workspace.potential_components.plan_token, BindingField::kPotential);
  XTBLOOM_CHECK_TOKEN(workspace.complete_potentials.plan_token, BindingField::kPotential);
  XTBLOOM_CHECK_TOKEN(workspace.scalar_bridge.plan_token, BindingField::kScalarBridge);
  XTBLOOM_CHECK_TOKEN(workspace.scalar_bridge.fields.plan_token, BindingField::kScalarBridge);
  XTBLOOM_CHECK_TOKEN(workspace.scalar_bridge.workspace.plan_token, BindingField::kScalarBridge);
  XTBLOOM_CHECK_TOKEN(workspace.hamiltonian.plan_token, BindingField::kHamiltonian);
  XTBLOOM_CHECK_TOKEN(workspace.staged_eigenpairs.plan_token, BindingField::kEigensolver);
  XTBLOOM_CHECK_TOKEN(workspace.staged_occupations.plan_token, BindingField::kOccupations);
  XTBLOOM_CHECK_TOKEN(workspace.staged_density.plan_token, BindingField::kDensity);
  XTBLOOM_CHECK_TOKEN(workspace.staged_raw_population.plan_token, BindingField::kMulliken);
  XTBLOOM_CHECK_TOKEN(workspace.staged_classical_energy.plan_token, BindingField::kClassicalEnergy);
  XTBLOOM_CHECK_TOKEN(workspace.staged_free_energy.plan_token, BindingField::kFreeEnergy);
  XTBLOOM_CHECK_TOKEN(workspace.staged_mixer.plan_token, BindingField::kMixer);
  XTBLOOM_CHECK_TOKEN(workspace.next_mixed.plan_token, BindingField::kMixer);
  XTBLOOM_CHECK_TOKEN(workspace.staged_publication.plan_token, BindingField::kStatePublication);
  XTBLOOM_CHECK_TOKEN(workspace.staged_publication.wavefunction.plan_token,
                      BindingField::kStatePublication);
  XTBLOOM_CHECK_TOKEN(workspace.staged_publication.energy.plan_token,
                      BindingField::kStatePublication);
  XTBLOOM_CHECK_TOKEN(workspace.geometry_workspace.plan_token, BindingField::kGeometry);
  if (component_enabled(plan, Gfn2SccPotentialComponent::kPeriodicEmbedding)) {
    XTBLOOM_CHECK_TOKEN(workspace.periodic_workspace.plan_token, BindingField::kPeriodicEmbedding);
  }
  XTBLOOM_CHECK_TOKEN(workspace.potential_workspace.plan_token, BindingField::kPotential);
  XTBLOOM_CHECK_TOKEN(workspace.hamiltonian_workspace.plan_token, BindingField::kHamiltonian);
  XTBLOOM_CHECK_TOKEN(workspace.eigensolver_workspace.plan_token, BindingField::kEigensolver);
  XTBLOOM_CHECK_TOKEN(workspace.occupations_workspace.plan_token, BindingField::kOccupations);
  XTBLOOM_CHECK_TOKEN(workspace.density_workspace.plan_token, BindingField::kDensity);
  XTBLOOM_CHECK_TOKEN(workspace.mulliken_workspace.plan_token, BindingField::kMulliken);
  XTBLOOM_CHECK_TOKEN(workspace.spin_output.plan_token, BindingField::kSpin);
  XTBLOOM_CHECK_TOKEN(workspace.spin_workspace.plan_token, BindingField::kSpin);
  XTBLOOM_CHECK_TOKEN(workspace.electronic_energy_workspace.plan_token,
                      BindingField::kElectronicEnergy);
  XTBLOOM_CHECK_TOKEN(workspace.classical_energy_workspace.plan_token,
                      BindingField::kClassicalEnergy);
  XTBLOOM_CHECK_TOKEN(workspace.free_energy_workspace.plan_token, BindingField::kFreeEnergy);
  XTBLOOM_CHECK_TOKEN(workspace.mixer_workspace.plan_token, BindingField::kMixer);
  XTBLOOM_CHECK_TOKEN(workspace.publication_workspace.plan_token, BindingField::kStatePublication);
#undef XTBLOOM_CHECK_TOKEN
  return true;
}

bool validate_top_level_shape(const Gfn2SccIterationDevicePlan& plan, Validator& validator,
                              std::int64_t& dipoles, std::int64_t& quadrupoles,
                              std::int64_t& two_batch, std::int64_t& two_orbitals,
                              std::int64_t& mixer_vector) noexcept {
  if (plan.abi_version != kGfn2SccIterationAbiVersion) {
    return validator.fail(BindingError::kInvalidAbiVersion, BindingField::kPlan);
  }
  const std::uint32_t invalid_components =
      plan.enabled_components & ~kGfn2SccPotentialAllComponents;
  if (invalid_components != 0u ||
      (plan.enabled_components & kMandatoryPotentialComponents) != kMandatoryPotentialComponents) {
    return validator.fail(BindingError::kInvalidCount, BindingField::kPlan);
  }
  if (plan.topology.memory_space != Gfn2PlanMemorySpace::kCudaDevice ||
      plan.topology.batch_size <= 0 || plan.topology.total_atoms <= 0 ||
      plan.topology.total_shells <= 0 || plan.topology.total_orbitals <= 0 ||
      plan.topology.total_matrix_elements <= 0 || plan.geometry_generation == 0u) {
    return validator.fail(BindingError::kInvalidTopology, BindingField::kTopology);
  }
  const Gfn2PlanSchemaDiagnostic topology =
      validate_gfn2_topology_binding(plan.topology, Gfn2PlanMemorySpace::kCudaDevice);
  if (topology.error != Gfn2PlanSchemaError::kSuccess) {
    return validator.fail(BindingError::kInvalidTopology, BindingField::kTopology, topology.index);
  }
  const Gfn2PlanSchemaDiagnostic wavefunction = validate_gfn2_wavefunction_layout_binding(
      plan.topology, plan.wavefunction_layout, Gfn2PlanMemorySpace::kCudaDevice);
  if (wavefunction.error != Gfn2PlanSchemaError::kSuccess) {
    return validator.fail(BindingError::kInvalidTopology, BindingField::kSpin, wavefunction.index);
  }
  if (!checked_multiply(plan.topology.total_atoms, 3, dipoles) ||
      !checked_multiply(plan.topology.total_atoms, 6, quadrupoles) ||
      !checked_multiply(plan.topology.batch_size, 2, two_batch) ||
      !checked_multiply(plan.topology.total_orbitals, 2, two_orbitals)) {
    return validator.fail(BindingError::kInvalidCount, BindingField::kTopology);
  }
  std::int64_t atom_multipoles = 0;
  if (!checked_multiply(plan.wavefunction_layout.total_spin_atoms, 9, atom_multipoles) ||
      !checked_add(plan.wavefunction_layout.total_spin_shells, atom_multipoles, mixer_vector)) {
    return validator.fail(BindingError::kInvalidCount, BindingField::kMixer);
  }
  return true;
}

/*
 * ABI v3 projection authority gate.  The plan's sealed common projections are
 * the single borrowing authority for every topology-derived leaf.  This step
 * proves each projection is a valid exact-pointer projection of the master
 * topology (or, for element identity, a valid setup-owned seal), so a later
 * leaf that names a different-but-equal array is rejected here once instead of
 * every leaf validator re-deriving the master.  It never dereferences device
 * arrays and performs no CUDA work.
 */
bool validate_plan_projection_identity(const Gfn2SccIterationDevicePlan& plan,
                                       Validator& validator) noexcept {
  const auto packed_projection_is_empty = [](const Gfn2PackedAllPairProjectionView& projection) {
    return projection.memory_space == Gfn2PlanMemorySpace::kHost && projection.plan_token == 0u &&
           projection.batch_size == 0 && projection.total_pairs == 0 &&
           projection.pair_offset_count == 0 && projection.pair_offsets == nullptr;
  };
  const auto bucket_projection_is_empty = [](const Gfn2AOBucketProjectionView& projection) {
    return projection.memory_space == Gfn2PlanMemorySpace::kHost && projection.plan_token == 0u &&
           projection.batch_size == 0 && projection.bucket_count == 0 &&
           projection.bucket_offset_count == 0 && projection.bucket_system_count == 0 &&
           projection.bucket_orbital_count == 0 && projection.bucket_offsets == nullptr &&
           projection.bucket_systems == nullptr && projection.bucket_orbital_counts == nullptr;
  };
  const Gfn2PlanSchemaDiagnostic atom = validate_gfn2_atom_projection_binding(
      plan.topology, plan.atom_projection, Gfn2PlanMemorySpace::kCudaDevice);
  if (atom.error != Gfn2PlanSchemaError::kSuccess) {
    return validator.fail(BindingError::kInvalidTopology, BindingField::kTopology, atom.index);
  }
  const Gfn2PlanSchemaDiagnostic shell = validate_gfn2_shell_ownership_projection_binding(
      plan.topology, plan.shell_ownership_projection, Gfn2PlanMemorySpace::kCudaDevice);
  if (shell.error != Gfn2PlanSchemaError::kSuccess) {
    return validator.fail(BindingError::kInvalidTopology, BindingField::kTopology, shell.index);
  }
  const Gfn2PlanSchemaDiagnostic ao = validate_gfn2_ao_matrix_projection_binding(
      plan.topology, plan.ao_matrix_projection, Gfn2PlanMemorySpace::kCudaDevice);
  if (ao.error != Gfn2PlanSchemaError::kSuccess) {
    return validator.fail(BindingError::kInvalidTopology, BindingField::kTopology, ao.index);
  }
  /* Packed-all-pair and AO-bucket projections are consumed only by the dense
   * geometry/AES2 and eigensolver/P-W paths respectively.  They are required
   * exactly when the master topology declares that domain; otherwise the
   * projection must remain the canonical empty form so a stale foreign
   * projection can never be mistaken for this plan. */
  if (plan.topology.pair_map_kind == Gfn2PairMapKind::kPackedLowerTriangle) {
    const Gfn2PlanSchemaDiagnostic pairs = validate_gfn2_packed_all_pair_projection_binding(
        plan.topology, plan.packed_all_pair_projection, Gfn2PlanMemorySpace::kCudaDevice);
    if (pairs.error != Gfn2PlanSchemaError::kSuccess) {
      return validator.fail(BindingError::kInvalidTopology, BindingField::kTopology, pairs.index);
    }
  } else if (!packed_projection_is_empty(plan.packed_all_pair_projection)) {
    return validator.fail(BindingError::kInvalidTopology, BindingField::kTopology);
  }
  if (plan.topology.bucket_count != 0) {
    const Gfn2PlanSchemaDiagnostic buckets = validate_gfn2_ao_bucket_projection_binding(
        plan.topology, plan.ao_bucket_projection, Gfn2PlanMemorySpace::kCudaDevice);
    if (buckets.error != Gfn2PlanSchemaError::kSuccess) {
      return validator.fail(BindingError::kInvalidTopology, BindingField::kTopology, buckets.index);
    }
  } else if (!bucket_projection_is_empty(plan.ao_bucket_projection)) {
    return validator.fail(BindingError::kInvalidTopology, BindingField::kTopology);
  }
  const Gfn2PlanSchemaDiagnostic element = validate_gfn2_element_identity_projection_binding(
      plan.element_identity_projection, Gfn2PlanMemorySpace::kCudaDevice);
  if (element.error != Gfn2PlanSchemaError::kSuccess) {
    return validator.fail(BindingError::kInvalidTopology, BindingField::kD4, element.index);
  }
  if (!validator.token(plan.element_identity_projection.plan_token, plan.plan_token,
                       BindingField::kD4) ||
      !validator.exact_count(plan.element_identity_projection.total_atoms,
                             plan.topology.total_atoms, BindingField::kD4) ||
      !validator.exact_count(plan.element_identity_projection.atomic_number_count,
                             plan.topology.total_atoms, BindingField::kD4)) {
    return false;
  }
  return true;
}

bool validate_plan_shapes(const Gfn2SccIterationDevicePlan& plan, Validator& validator,
                          std::int64_t dipoles, std::int64_t quadrupoles, std::int64_t two_batch,
                          std::int64_t mixer_vector) noexcept {
  const std::int64_t batch = plan.topology.batch_size;
  const std::int64_t atoms = plan.topology.total_atoms;
  const std::int64_t shells = plan.topology.total_shells;
  const std::int64_t orbitals = plan.topology.total_orbitals;
  const std::int64_t matrices = plan.topology.total_matrix_elements;
  const auto& wavefunction = plan.wavefunction_layout;
  const auto exact = [&](std::int64_t actual, std::int64_t expected, BindingField field,
                         std::int64_t index = -1) {
    return validator.exact_count(actual, expected, field, index);
  };
  if (!exact(plan.activity_policy.batch_size, batch, BindingField::kActivity) ||
      plan.activity_policy.maximum_iterations == 0u ||
      plan.state_policy.maximum_iterations != plan.activity_policy.maximum_iterations ||
      !std::isfinite(plan.state_policy.residual_rms_tolerance) ||
      !(plan.state_policy.residual_rms_tolerance > 0.0) ||
      !std::isfinite(plan.state_policy.energy_tolerance) ||
      !(plan.state_policy.energy_tolerance > 0.0) || plan.mixer_policy.history_size <= 0 ||
      !std::isfinite(plan.mixer_policy.damping) || !(plan.mixer_policy.damping > 0.0) ||
      !std::isfinite(plan.mixer_policy.rms_tolerance) || !(plan.mixer_policy.rms_tolerance > 0.0) ||
      !std::isfinite(plan.mixer_policy.maximum_tolerance) ||
      !(plan.mixer_policy.maximum_tolerance > 0.0)) {
    return validator.fail(BindingError::kInvalidCount, BindingField::kPlan);
  }
  /* The CPU driver applies the mixer's RMS threshold at final SCC
   * publication. Retain the legacy state-policy field only as an exact mirror
   * so setup cannot silently define two terminal convergence criteria. */
  if (plan.state_policy.residual_rms_tolerance != plan.mixer_policy.rms_tolerance) {
    return validator.fail(BindingError::kInvalidCount, BindingField::kStatePublication);
  }
  if (!exact(plan.geometry_batch.batch_size, batch, BindingField::kGeometry) ||
      !exact(plan.geometry_batch.total_atoms, atoms, BindingField::kGeometry) ||
      !exact(plan.scc_batch.batch_size, batch, BindingField::kStatePublication) ||
      !exact(plan.scc_batch.total_atoms, atoms, BindingField::kStatePublication) ||
      !exact(plan.scc_batch.total_shells, shells, BindingField::kStatePublication) ||
      !exact(plan.scc_batch.shell_offset_count, batch + 1, BindingField::kStatePublication) ||
      !exact(plan.scc_batch.atom_offset_count, batch + 1, BindingField::kStatePublication) ||
      !exact(plan.potential_batch.batch_size, batch, BindingField::kPotential) ||
      !exact(plan.potential_batch.total_atoms, atoms, BindingField::kPotential) ||
      !exact(plan.potential_batch.total_shells, shells, BindingField::kPotential) ||
      !exact(plan.spin_batch.batch_size, batch, BindingField::kSpin) ||
      !exact(plan.spin_batch.total_atoms, atoms, BindingField::kSpin) ||
      !exact(plan.spin_batch.total_shells, shells, BindingField::kSpin) ||
      !exact(plan.spin_batch.shell_population_elements, wavefunction.total_spin_shells,
             BindingField::kSpin) ||
      !exact(plan.hamiltonian_batch.batch_size, batch, BindingField::kHamiltonian) ||
      !exact(plan.hamiltonian_batch.total_atoms, atoms, BindingField::kHamiltonian) ||
      !exact(plan.hamiltonian_batch.total_shells, shells, BindingField::kHamiltonian) ||
      !exact(plan.hamiltonian_batch.total_orbitals, orbitals, BindingField::kHamiltonian) ||
      !exact(plan.hamiltonian_batch.total_matrix_elements, matrices, BindingField::kHamiltonian) ||
      !exact(plan.eigensolver_batch.batch_size, batch, BindingField::kEigensolver) ||
      !exact(plan.eigensolver_batch.total_orbitals, orbitals, BindingField::kEigensolver) ||
      !exact(plan.eigensolver_batch.total_matrix_elements, matrices, BindingField::kEigensolver) ||
      !exact(plan.occupations_batch.batch_size, batch, BindingField::kOccupations) ||
      !exact(plan.occupations_batch.total_orbitals, orbitals, BindingField::kOccupations) ||
      !exact(plan.occupations_batch.electron_count_elements, two_batch,
             BindingField::kOccupations) ||
      !exact(plan.occupations_batch.temperature_elements, batch, BindingField::kOccupations) ||
      !exact(plan.density_batch.batch_size, batch, BindingField::kDensity) ||
      !exact(plan.density_batch.total_orbitals, orbitals, BindingField::kDensity) ||
      !exact(plan.density_batch.total_matrix_elements, matrices, BindingField::kDensity) ||
      !exact(plan.mulliken_batch.batch_size, batch, BindingField::kMulliken) ||
      !exact(plan.mulliken_batch.total_atoms, atoms, BindingField::kMulliken) ||
      !exact(plan.mulliken_batch.total_shells, shells, BindingField::kMulliken) ||
      !exact(plan.mulliken_batch.total_orbitals, orbitals, BindingField::kMulliken) ||
      !exact(plan.mulliken_batch.total_matrix_elements, matrices, BindingField::kMulliken) ||
      !exact(plan.electronic_energy_batch.batch_size, batch, BindingField::kElectronicEnergy) ||
      !exact(plan.electronic_energy_batch.total_matrix_elements, matrices,
             BindingField::kElectronicEnergy) ||
      !exact(plan.classical_energy_batch.batch_size, batch, BindingField::kClassicalEnergy) ||
      !exact(plan.free_energy_batch.batch_size, batch, BindingField::kFreeEnergy) ||
      !exact(plan.publication_plan.batch_size, batch, BindingField::kStatePublication) ||
      !exact(plan.publication_plan.total_atoms, atoms, BindingField::kStatePublication) ||
      !exact(plan.publication_plan.total_shells, shells, BindingField::kStatePublication) ||
      !exact(plan.publication_plan.total_orbitals, orbitals, BindingField::kStatePublication) ||
      !exact(plan.publication_plan.total_matrix_elements, matrices,
             BindingField::kStatePublication)) {
    return false;
  }
  if (plan.classical_energy_batch.enabled_components != plan.enabled_components ||
      plan.free_energy_batch.enabled_components != plan.enabled_components ||
      !std::isfinite(plan.free_energy_batch.electronic_temperature) ||
      plan.free_energy_batch.electronic_temperature < 0.0) {
    return validator.fail(BindingError::kInvalidCount, BindingField::kFreeEnergy);
  }
  if (plan.publication_plan.total_mixer_vector_elements != mixer_vector ||
      plan.publication_plan.history_size != plan.mixer_policy.history_size ||
      plan.publication_plan.maximum_iterations != plan.activity_policy.maximum_iterations ||
      plan.publication_plan.residual_rms_tolerance != plan.mixer_policy.rms_tolerance ||
      plan.publication_plan.energy_tolerance != plan.state_policy.energy_tolerance ||
      plan.publication_plan.atom_offset_count != batch + 1 ||
      plan.publication_plan.shell_offset_count != batch + 1 ||
      plan.publication_plan.orbital_offset_count != batch + 1 ||
      plan.publication_plan.matrix_offset_count != batch + 1 ||
      plan.publication_plan.shell_to_atom_count != shells ||
      plan.publication_plan.atom_offsets != plan.topology.atom_offsets ||
      plan.publication_plan.shell_offsets != plan.topology.batch_shell_offsets ||
      plan.publication_plan.orbital_offsets != plan.topology.batch_orbital_offsets ||
      plan.publication_plan.matrix_offsets != plan.topology.matrix_offsets ||
      plan.publication_plan.shell_to_atom != plan.topology.shell_to_atom ||
      !same_wavefunction_layout(plan.publication_plan.wavefunction_layout,
                                plan.wavefunction_layout)) {
    return validator.fail(BindingError::kInvalidZeroCopyView, BindingField::kStatePublication);
  }
  if (!exact(plan.es2_batch.batch_size, batch, BindingField::kES2) ||
      !exact(plan.es2_batch.total_atoms, atoms, BindingField::kES2) ||
      !exact(plan.es2_batch.total_shells, shells, BindingField::kES2) ||
      !exact(plan.es3_batch.batch_size, batch, BindingField::kES3) ||
      !exact(plan.es3_batch.total_shells, shells, BindingField::kES3) ||
      !exact(plan.aes2_batch.batch_size, batch, BindingField::kAES2) ||
      !exact(plan.aes2_batch.total_atoms, atoms, BindingField::kAES2)) {
    return false;
  }
  if (component_enabled(plan, Gfn2SccPotentialComponent::kD4TwoBody) &&
      (!exact(plan.d4_batch.batch_size, batch, BindingField::kD4) ||
       !exact(plan.d4_batch.total_atoms, atoms, BindingField::kD4))) {
    return false;
  }
  if (component_enabled(plan, Gfn2SccPotentialComponent::kExplicitPointCharge) &&
      (!exact(plan.explicit_point_charge_batch.batch_size, batch,
              BindingField::kExplicitPointCharge) ||
       !exact(plan.explicit_point_charge_batch.total_atoms, atoms,
              BindingField::kExplicitPointCharge) ||
       !exact(plan.explicit_point_charge_batch.total_shells, shells,
              BindingField::kExplicitPointCharge))) {
    return false;
  }
  if (component_enabled(plan, Gfn2SccPotentialComponent::kPeriodicEmbedding) &&
      (!exact(plan.periodic_batch.batch_size, batch, BindingField::kPeriodicEmbedding) ||
       !exact(plan.periodic_batch.total_atoms, atoms, BindingField::kPeriodicEmbedding))) {
    return false;
  }
  if (plan.scalar_bridge_batch.topology.memory_space != Gfn2PlanMemorySpace::kCudaDevice ||
      plan.scalar_bridge_batch.topology.batch_size != batch ||
      plan.scalar_bridge_batch.topology.total_atoms != atoms ||
      plan.scalar_bridge_batch.topology.total_shells != shells) {
    return validator.fail(BindingError::kInvalidTopology, BindingField::kScalarBridge);
  }
  (void)dipoles;
  (void)quadrupoles;
  return true;
}

bool validate_plan_pointer_shapes(const Gfn2SccIterationDevicePlan& plan,
                                  Validator& validator) noexcept {
  const std::int64_t batch = plan.topology.batch_size;
  const std::int64_t atoms = plan.topology.total_atoms;
  const std::int64_t shells = plan.topology.total_shells;
  const std::int64_t orbitals = plan.topology.total_orbitals;
  const std::int64_t matrices = plan.topology.total_matrix_elements;
  const auto exact = [&](std::int64_t actual, std::int64_t expected, BindingField field,
                         std::int64_t index = -1) {
    return validator.exact_count(actual, expected, field, index);
  };
  const auto aligned = [&](const void* pointer, std::int64_t elements, std::size_t element_size,
                           std::size_t alignment, BindingField field, std::int64_t index) {
    if (elements == 0) {
      return pointer == nullptr || validator.fail(BindingError::kNullPointer, field, index);
    }
    if (pointer == nullptr) {
      return validator.fail(BindingError::kNullPointer, field, index);
    }
    if (reinterpret_cast<std::uintptr_t>(pointer) % alignment != 0u) {
      return validator.fail(BindingError::kMisalignedPointer, field, index);
    }
    if (static_cast<std::uint64_t>(elements) >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
      return validator.fail(BindingError::kAddressOverflow, field, index);
    }
    return true;
  };

  const auto& provenance = plan.provenance;
  if (provenance.cache_binding_count < 0 ||
      provenance.expected_geometry_generation != plan.geometry_generation ||
      !aligned(provenance.cache_bindings, provenance.cache_binding_count,
               sizeof(Gfn2SccCacheProvenanceBinding), alignof(Gfn2SccCacheProvenanceBinding),
               BindingField::kActivity, 0) ||
      !((provenance.warm_start_generations == nullptr && provenance.warm_start_elements == 0 &&
         provenance.expected_warm_start_generation == 0u) ||
        (provenance.warm_start_generations != nullptr && provenance.warm_start_elements == batch &&
         provenance.expected_warm_start_generation != 0u &&
         reinterpret_cast<std::uintptr_t>(provenance.warm_start_generations) %
                 alignof(std::uint64_t) ==
             0u))) {
    return validator.fail(BindingError::kInvalidCount, BindingField::kActivity);
  }

  const auto& geometry = plan.geometry_batch;
  if (!exact(geometry.atom_offset_elements, batch + 1, BindingField::kGeometry) ||
      !exact(geometry.pair_offset_elements, batch + 1, BindingField::kGeometry) ||
      !exact(geometry.covalent_radius_elements, atoms, BindingField::kGeometry) ||
      !exact(geometry.coordinate_elements, 3 * atoms, BindingField::kGeometry) ||
      !aligned(geometry.atom_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
               BindingField::kGeometry, 0) ||
      !aligned(geometry.pair_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
               BindingField::kGeometry, 1) ||
      !aligned(geometry.covalent_radii, atoms, sizeof(double), alignof(double),
               BindingField::kGeometry, 2)) {
    return false;
  }

  const auto& potential = plan.potential_batch;
  if (!exact(potential.atom_offset_count, batch + 1, BindingField::kPotential) ||
      !exact(potential.batch_shell_offset_count, batch + 1, BindingField::kPotential) ||
      !exact(potential.qsh_offset_count, batch + 1, BindingField::kPotential) ||
      !exact(potential.qat_offset_count, batch + 1, BindingField::kPotential) ||
      !exact(potential.dipole_offset_count, batch + 1, BindingField::kPotential) ||
      !exact(potential.quadrupole_offset_count, batch + 1, BindingField::kPotential) ||
      !exact(potential.shell_to_atom_count, shells, BindingField::kPotential) ||
      !aligned(potential.atom_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
               BindingField::kPotential, 0) ||
      !aligned(potential.batch_shell_offsets, batch + 1, sizeof(std::int64_t),
               alignof(std::int64_t), BindingField::kPotential, 1) ||
      !aligned(potential.qsh_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
               BindingField::kPotential, 2) ||
      !aligned(potential.qat_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
               BindingField::kPotential, 3) ||
      !aligned(potential.dipole_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
               BindingField::kPotential, 4) ||
      !aligned(potential.quadrupole_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
               BindingField::kPotential, 5) ||
      !aligned(potential.shell_to_atom, shells, sizeof(std::int64_t), alignof(std::int64_t),
               BindingField::kPotential, 6)) {
    return false;
  }

  const auto& spin = plan.spin_batch;
  if (!exact(spin.atom_offset_count, batch + 1, BindingField::kSpin, 0) ||
      !exact(spin.batch_shell_offset_count, batch + 1, BindingField::kSpin, 1) ||
      !exact(spin.atom_shell_offset_count, atoms + 1, BindingField::kSpin, 2) ||
      !exact(spin.shell_population_offset_count, batch + 1, BindingField::kSpin, 3) ||
      !exact(spin.spin_channel_count, batch, BindingField::kSpin, 4) ||
      !exact(spin.coupling_offset_count, atoms + 1, BindingField::kSpin, 5) ||
      spin.coupling_matrix_count <= 0 ||
      !aligned(spin.atom_offsets, spin.atom_offset_count, sizeof(std::int64_t),
               alignof(std::int64_t), BindingField::kSpin, 0) ||
      !aligned(spin.batch_shell_offsets, spin.batch_shell_offset_count, sizeof(std::int64_t),
               alignof(std::int64_t), BindingField::kSpin, 1) ||
      !aligned(spin.atom_shell_offsets, spin.atom_shell_offset_count, sizeof(std::int64_t),
               alignof(std::int64_t), BindingField::kSpin, 2) ||
      !aligned(spin.shell_population_offsets, spin.shell_population_offset_count,
               sizeof(std::int64_t), alignof(std::int64_t), BindingField::kSpin, 3) ||
      !aligned(spin.spin_channels, spin.spin_channel_count, sizeof(std::int32_t),
               alignof(std::int32_t), BindingField::kSpin, 4) ||
      !aligned(spin.coupling_offsets, spin.coupling_offset_count, sizeof(std::int64_t),
               alignof(std::int64_t), BindingField::kSpin, 5) ||
      !aligned(spin.coupling_matrices, spin.coupling_matrix_count, sizeof(double), alignof(double),
               BindingField::kSpin, 6)) {
    return false;
  }

  const auto& es2 = plan.es2_batch;
  if (!exact(es2.atom_offset_count, batch + 1, BindingField::kES2) ||
      !exact(es2.batch_shell_offset_count, batch + 1, BindingField::kES2) ||
      !exact(es2.atom_shell_offset_count, atoms + 1, BindingField::kES2) ||
      !exact(es2.matrix_offset_count, batch + 1, BindingField::kES2) ||
      !exact(es2.shell_to_atom_count, shells, BindingField::kES2) ||
      !exact(es2.shell_hardness_count, shells, BindingField::kES2) ||
      !aligned(es2.atom_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
               BindingField::kES2, 0) ||
      !aligned(es2.batch_shell_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
               BindingField::kES2, 1) ||
      !aligned(es2.atom_shell_offsets, atoms + 1, sizeof(std::int64_t), alignof(std::int64_t),
               BindingField::kES2, 2) ||
      !aligned(es2.matrix_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
               BindingField::kES2, 3) ||
      !aligned(es2.shell_to_atom, shells, sizeof(std::int64_t), alignof(std::int64_t),
               BindingField::kES2, 4) ||
      !aligned(es2.shell_hardness, shells, sizeof(double), alignof(double), BindingField::kES2,
               5)) {
    return false;
  }
  const auto& es3 = plan.es3_batch;
  if (!exact(es3.batch_shell_offset_count, batch + 1, BindingField::kES3) ||
      !exact(es3.shell_gamma3_count, shells, BindingField::kES3) ||
      !aligned(es3.batch_shell_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
               BindingField::kES3, 0) ||
      !aligned(es3.shell_gamma3, shells, sizeof(double), alignof(double), BindingField::kES3, 1)) {
    return false;
  }

  const auto& aes2 = plan.aes2_batch;
  if (!exact(aes2.atom_offset_count, batch + 1, BindingField::kAES2) ||
      !exact(aes2.pair_offset_count, batch + 1, BindingField::kAES2) ||
      !exact(aes2.dipole_kernel_count, atoms, BindingField::kAES2) ||
      !exact(aes2.quadrupole_kernel_count, atoms, BindingField::kAES2) ||
      !exact(aes2.multipole_radius_count, atoms, BindingField::kAES2) ||
      !exact(aes2.multipole_valence_cn_count, atoms, BindingField::kAES2) ||
      !aligned(aes2.atom_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
               BindingField::kAES2, 0) ||
      !aligned(aes2.pair_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
               BindingField::kAES2, 1) ||
      !aligned(aes2.dipole_kernel, atoms, sizeof(double), alignof(double), BindingField::kAES2,
               2) ||
      !aligned(aes2.quadrupole_kernel, atoms, sizeof(double), alignof(double), BindingField::kAES2,
               3) ||
      !aligned(aes2.multipole_radius, atoms, sizeof(double), alignof(double), BindingField::kAES2,
               4) ||
      !aligned(aes2.multipole_valence_cn, atoms, sizeof(double), alignof(double),
               BindingField::kAES2, 5)) {
    return false;
  }

  if (component_enabled(plan, Gfn2SccPotentialComponent::kD4TwoBody)) {
    const auto& d4 = plan.d4_batch;
    const auto& parameters = plan.d4_parameters;
    std::int64_t reference_square = 0;
    if (!checked_multiply(parameters.reference_count, parameters.reference_count,
                          reference_square) ||
        parameters.element_count <= 0 || parameters.reference_count <= 0 ||
        parameters.reference_c6_elements < reference_square || d4.atomic_number_hash == 0u ||
        !aligned(d4.atom_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
                 BindingField::kD4, 0) ||
        !aligned(d4.pair_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
                 BindingField::kD4, 1) ||
        !aligned(d4.atomic_numbers, atoms, sizeof(std::int32_t), alignof(std::int32_t),
                 BindingField::kD4, 2) ||
        !aligned(parameters.elements, parameters.element_count, sizeof(Gfn2D4DeviceElementData),
                 alignof(Gfn2D4DeviceElementData), BindingField::kD4, 3) ||
        !aligned(parameters.references, parameters.reference_count,
                 sizeof(Gfn2D4DeviceReferenceData), alignof(Gfn2D4DeviceReferenceData),
                 BindingField::kD4, 4) ||
        !aligned(parameters.reference_c6, parameters.reference_c6_elements, sizeof(double),
                 alignof(double), BindingField::kD4, 5)) {
      return false;
    }
  }

  if (component_enabled(plan, Gfn2SccPotentialComponent::kExplicitPointCharge)) {
    const auto& point = plan.explicit_point_charge_batch;
    std::int64_t atom_coordinates = 0;
    std::int64_t point_coordinates = 0;
    if (!checked_multiply(atoms, 3, atom_coordinates) ||
        !checked_multiply(point.total_point_charges, 3, point_coordinates) ||
        point.total_point_charges < 0 ||
        !aligned(point.atom_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
                 BindingField::kExplicitPointCharge, 0) ||
        !aligned(point.batch_shell_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
                 BindingField::kExplicitPointCharge, 1) ||
        !aligned(point.point_charge_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
                 BindingField::kExplicitPointCharge, 2) ||
        !aligned(point.shell_to_atom, shells, sizeof(std::int64_t), alignof(std::int64_t),
                 BindingField::kExplicitPointCharge, 3) ||
        !aligned(point.shell_hardness, shells, sizeof(double), alignof(double),
                 BindingField::kExplicitPointCharge, 4) ||
        !aligned(point.qm_positions, atom_coordinates, sizeof(double), alignof(double),
                 BindingField::kExplicitPointCharge, 5) ||
        !aligned(point.point_positions, point_coordinates, sizeof(double), alignof(double),
                 BindingField::kExplicitPointCharge, 6) ||
        !aligned(point.point_charges, point.total_point_charges, sizeof(double), alignof(double),
                 BindingField::kExplicitPointCharge, 7) ||
        !aligned(point.point_hardnesses, point.total_point_charges, sizeof(double), alignof(double),
                 BindingField::kExplicitPointCharge, 8)) {
      return false;
    }
  }

  if (component_enabled(plan, Gfn2SccPotentialComponent::kPeriodicEmbedding)) {
    const auto& periodic = plan.periodic_batch;
    if (!exact(periodic.atom_offset_count, batch + 1, BindingField::kPeriodicEmbedding) ||
        !exact(periodic.matrix_offset_count, batch + 1, BindingField::kPeriodicEmbedding) ||
        !exact(periodic.shift_elements, atoms, BindingField::kPeriodicEmbedding) ||
        !exact(periodic.response_elements, periodic.total_matrix_elements,
               BindingField::kPeriodicEmbedding) ||
        periodic.geometry_generation != plan.geometry_generation ||
        !aligned(periodic.atom_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
                 BindingField::kPeriodicEmbedding, 0) ||
        !aligned(periodic.matrix_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
                 BindingField::kPeriodicEmbedding, 1) ||
        !aligned(periodic.shifts, atoms, sizeof(double), alignof(double),
                 BindingField::kPeriodicEmbedding, 2) ||
        !aligned(periodic.response_matrices, periodic.total_matrix_elements, sizeof(double),
                 alignof(double), BindingField::kPeriodicEmbedding, 3)) {
      return false;
    }
  }

  const auto& eigensolver = plan.eigensolver_batch;
  if (!exact(eigensolver.orbital_offset_count, batch + 1, BindingField::kEigensolver) ||
      !exact(eigensolver.matrix_offset_count, batch + 1, BindingField::kEigensolver) ||
      !exact(eigensolver.bucket_system_count, batch, BindingField::kEigensolver) ||
      !exact(eigensolver.active_elements, batch, BindingField::kEigensolver) ||
      !aligned(eigensolver.orbital_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
               BindingField::kEigensolver, 0) ||
      !aligned(eigensolver.matrix_offsets, batch + 1, sizeof(std::int64_t), alignof(std::int64_t),
               BindingField::kEigensolver, 1) ||
      !aligned(eigensolver.bucket_systems, batch, sizeof(std::int32_t), alignof(std::int32_t),
               BindingField::kEigensolver, 2) ||
      !exact(plan.overlap_cache.factor_elements, matrices, BindingField::kEigensolver) ||
      !exact(plan.overlap_cache.generation_elements, batch, BindingField::kEigensolver) ||
      !exact(plan.overlap_cache.status_elements, batch, BindingField::kEigensolver) ||
      !aligned(plan.overlap_cache.cholesky_factors, matrices, sizeof(double), alignof(double),
               BindingField::kEigensolver, 3) ||
      !aligned(plan.overlap_cache.geometry_generations, batch, sizeof(std::uint64_t),
               alignof(std::uint64_t), BindingField::kEigensolver, 4) ||
      !aligned(plan.overlap_cache.factor_statuses, batch, sizeof(std::uint32_t),
               alignof(std::uint32_t), BindingField::kEigensolver, 5)) {
    return false;
  }

  if (!exact(plan.occupations_batch.orbital_offset_count, batch + 1, BindingField::kOccupations) ||
      !exact(plan.occupations_batch.active_elements, batch, BindingField::kOccupations) ||
      !exact(plan.density_batch.orbital_offset_count, batch + 1, BindingField::kDensity) ||
      !exact(plan.density_batch.matrix_offset_count, batch + 1, BindingField::kDensity) ||
      !exact(plan.mulliken_batch.atom_offset_count, batch + 1, BindingField::kMulliken) ||
      !exact(plan.mulliken_batch.batch_shell_offset_count, batch + 1, BindingField::kMulliken) ||
      !exact(plan.mulliken_batch.batch_orbital_offset_count, batch + 1, BindingField::kMulliken) ||
      !exact(plan.mulliken_batch.matrix_offset_count, batch + 1, BindingField::kMulliken) ||
      !exact(plan.mulliken_batch.atom_shell_offset_count, atoms + 1, BindingField::kMulliken) ||
      !exact(plan.mulliken_batch.shell_orbital_offset_count, shells + 1, BindingField::kMulliken) ||
      !exact(plan.mulliken_batch.shell_to_atom_count, shells, BindingField::kMulliken) ||
      !exact(plan.mulliken_batch.reference_occupation_count, shells, BindingField::kMulliken) ||
      !exact(plan.electronic_energy_batch.matrix_offset_count, batch + 1,
             BindingField::kElectronicEnergy)) {
    return false;
  }
  (void)orbitals;
  return true;
}

/*
 * Leaf-vs-projection identity (ABI v3 Step 3).  Each topology-only consumer
 * leaf must name exactly the sealed projection arrays, so a later leaf can no
 * longer silently re-derive or copy an equal-sized-but-different offset field.
 * These pointer identity checks are the single place the plan proves that
 * "equal-sized offset/map fields are the same plan"; the kernels otherwise
 * consume the same arrays through their legacy descriptors.
 */
bool validate_leaf_projection_identity(const Gfn2SccIterationDevicePlan& plan,
                                       Validator& validator) noexcept {
  const auto& atom = plan.atom_projection;
  const auto& shell = plan.shell_ownership_projection;
  const auto& ao = plan.ao_matrix_projection;
  const auto& buckets = plan.ao_bucket_projection;
  const auto& pairs = plan.packed_all_pair_projection;

  const auto same = [&](const void* leaf_pointer, const void* projection_pointer,
                        BindingField field) {
    if (leaf_pointer != projection_pointer) {
      return validator.fail(BindingError::kInvalidZeroCopyView, field);
    }
    return true;
  };

  /* Persistent SCC state and geometry/AES2 atom domains. */
  if (!same(plan.scc_batch.shell_offsets, shell.batch_shell_offsets,
            BindingField::kStatePublication) ||
      !same(plan.scc_batch.atom_offsets, atom.atom_offsets, BindingField::kStatePublication)) {
    return false;
  }
  if (!same(plan.geometry_batch.atom_offsets, atom.atom_offsets, BindingField::kGeometry) ||
      !same(plan.aes2_batch.atom_offsets, atom.atom_offsets, BindingField::kAES2)) {
    return false;
  }

  /* The potential and scalar bridge use physical charge-channel partitions.
   * They therefore borrow the atom and shell projections directly instead of
   * setup-owned copies whose equal values could conceal a foreign plan. */
  if (!same(plan.potential_batch.atom_offsets, atom.atom_offsets, BindingField::kPotential) ||
      !same(plan.potential_batch.batch_shell_offsets, shell.batch_shell_offsets,
            BindingField::kPotential) ||
      !same(plan.potential_batch.qsh_offsets, shell.batch_shell_offsets,
            BindingField::kPotential) ||
      !same(plan.potential_batch.qat_offsets, atom.atom_offsets, BindingField::kPotential) ||
      !same(plan.potential_batch.shell_to_atom, shell.shell_to_atom, BindingField::kPotential)) {
    return false;
  }

  /* Spin topology comes from the physical projections, while its expanded
   * population partition and channel count come from WavefunctionLayout. */
  if (!same(plan.spin_batch.atom_offsets, atom.atom_offsets, BindingField::kSpin) ||
      !same(plan.spin_batch.batch_shell_offsets, shell.batch_shell_offsets, BindingField::kSpin) ||
      !same(plan.spin_batch.atom_shell_offsets, shell.atom_shell_offsets, BindingField::kSpin) ||
      !same(plan.spin_batch.shell_population_offsets, plan.wavefunction_layout.spin_shell_offsets,
            BindingField::kSpin) ||
      !same(plan.spin_batch.spin_channels, plan.wavefunction_layout.spin_channels,
            BindingField::kSpin)) {
    return false;
  }

  if (!same(plan.es2_batch.atom_offsets, atom.atom_offsets, BindingField::kES2) ||
      !same(plan.es2_batch.batch_shell_offsets, shell.batch_shell_offsets, BindingField::kES2) ||
      !same(plan.es2_batch.atom_shell_offsets, shell.atom_shell_offsets, BindingField::kES2) ||
      !same(plan.es2_batch.shell_to_atom, shell.shell_to_atom, BindingField::kES2) ||
      !same(plan.es3_batch.batch_shell_offsets, shell.batch_shell_offsets, BindingField::kES3)) {
    return false;
  }

  if (component_enabled(plan, Gfn2SccPotentialComponent::kD4TwoBody) &&
      (!same(plan.d4_batch.atom_offsets, atom.atom_offsets, BindingField::kD4) ||
       !same(plan.d4_batch.atomic_numbers, plan.element_identity_projection.atomic_numbers,
             BindingField::kD4))) {
    return false;
  }
  if (component_enabled(plan, Gfn2SccPotentialComponent::kExplicitPointCharge) &&
      (!same(plan.explicit_point_charge_batch.atom_offsets, atom.atom_offsets,
             BindingField::kExplicitPointCharge) ||
       !same(plan.explicit_point_charge_batch.batch_shell_offsets, shell.batch_shell_offsets,
             BindingField::kExplicitPointCharge) ||
       !same(plan.explicit_point_charge_batch.shell_to_atom, shell.shell_to_atom,
             BindingField::kExplicitPointCharge))) {
    return false;
  }
  if (component_enabled(plan, Gfn2SccPotentialComponent::kPeriodicEmbedding) &&
      !same(plan.periodic_batch.atom_offsets, atom.atom_offsets,
            BindingField::kPeriodicEmbedding)) {
    return false;
  }

  /* Reducers and density/P/W/occupations: AO/matrix projection. */
  if (!same(plan.density_batch.orbital_offsets, ao.batch_orbital_offsets, BindingField::kDensity) ||
      !same(plan.density_batch.matrix_offsets, ao.matrix_offsets, BindingField::kDensity) ||
      !same(plan.occupations_batch.orbital_offsets, ao.batch_orbital_offsets,
            BindingField::kOccupations) ||
      !same(plan.electronic_energy_batch.matrix_offsets, ao.matrix_offsets,
            BindingField::kElectronicEnergy) ||
      !same(plan.hamiltonian_batch.atom_offsets, atom.atom_offsets, BindingField::kHamiltonian) ||
      !same(plan.hamiltonian_batch.batch_shell_offsets, shell.batch_shell_offsets,
            BindingField::kHamiltonian) ||
      !same(plan.hamiltonian_batch.batch_orbital_offsets, ao.batch_orbital_offsets,
            BindingField::kHamiltonian) ||
      !same(plan.hamiltonian_batch.matrix_offsets, ao.matrix_offsets, BindingField::kHamiltonian) ||
      !same(plan.hamiltonian_batch.atom_shell_offsets, shell.atom_shell_offsets,
            BindingField::kHamiltonian) ||
      !same(plan.hamiltonian_batch.shell_orbital_offsets, ao.shell_orbital_offsets,
            BindingField::kHamiltonian) ||
      !same(plan.hamiltonian_batch.shell_to_atom, shell.shell_to_atom,
            BindingField::kHamiltonian) ||
      !same(plan.hamiltonian_batch.orbital_to_shell, ao.orbital_to_shell,
            BindingField::kHamiltonian) ||
      !same(plan.hamiltonian_batch.orbital_to_atom, ao.orbital_to_atom,
            BindingField::kHamiltonian)) {
    return false;
  }

  /* Eigensolver: AO/matrix plus AO-bucket projections. */
  if (!same(plan.eigensolver_batch.orbital_offsets, ao.batch_orbital_offsets,
            BindingField::kEigensolver) ||
      !same(plan.eigensolver_batch.matrix_offsets, ao.matrix_offsets, BindingField::kEigensolver) ||
      !same(plan.eigensolver_batch.bucket_systems, buckets.bucket_systems,
            BindingField::kEigensolver)) {
    return false;
  }

  /* Mulliken: shell ownership plus AO/matrix projections. */
  if (!same(plan.mulliken_batch.atom_offsets, atom.atom_offsets, BindingField::kMulliken) ||
      !same(plan.mulliken_batch.batch_shell_offsets, shell.batch_shell_offsets,
            BindingField::kMulliken) ||
      !same(plan.mulliken_batch.batch_orbital_offsets, ao.batch_orbital_offsets,
            BindingField::kMulliken) ||
      !same(plan.mulliken_batch.matrix_offsets, ao.matrix_offsets, BindingField::kMulliken) ||
      !same(plan.mulliken_batch.atom_shell_offsets, shell.atom_shell_offsets,
            BindingField::kMulliken) ||
      !same(plan.mulliken_batch.shell_orbital_offsets, ao.shell_orbital_offsets,
            BindingField::kMulliken) ||
      !same(plan.mulliken_batch.shell_to_atom, shell.shell_to_atom, BindingField::kMulliken)) {
    return false;
  }

  /* Scalar bridge owns an exact copy of the complete master topology leaf and
   * its charge partitions borrow the same physical projections. */
  if (!same_topology_view(plan.scalar_bridge_batch.topology, plan.topology) ||
      !same(plan.scalar_bridge_batch.qsh_offsets, shell.batch_shell_offsets,
            BindingField::kScalarBridge) ||
      !same(plan.scalar_bridge_batch.qat_offsets, atom.atom_offsets, BindingField::kScalarBridge)) {
    return validator.fail(BindingError::kInvalidZeroCopyView, BindingField::kScalarBridge);
  }

  /* Publication is another consumer of the sealed physical partitions.  Its
   * complete WavefunctionLayout equality is checked with the plan shapes. */
  if (!same(plan.publication_plan.atom_offsets, atom.atom_offsets,
            BindingField::kStatePublication) ||
      !same(plan.publication_plan.shell_offsets, shell.batch_shell_offsets,
            BindingField::kStatePublication) ||
      !same(plan.publication_plan.orbital_offsets, ao.batch_orbital_offsets,
            BindingField::kStatePublication) ||
      !same(plan.publication_plan.matrix_offsets, ao.matrix_offsets,
            BindingField::kStatePublication) ||
      !same(plan.publication_plan.shell_to_atom, shell.shell_to_atom,
            BindingField::kStatePublication)) {
    return false;
  }

  /* Packed all-pair: dense geometry and AES2 use the packed projection when the
   * master declares it (production kNone keeps this empty and geometry/AES2
   * keep their setup-owned distinct pair offsets). */
  if (plan.topology.pair_map_kind == Gfn2PairMapKind::kPackedLowerTriangle) {
    if (!same(plan.geometry_batch.pair_offsets, pairs.pair_offsets, BindingField::kGeometry) ||
        !same(plan.aes2_batch.pair_offsets, pairs.pair_offsets, BindingField::kAES2) ||
        (component_enabled(plan, Gfn2SccPotentialComponent::kD4TwoBody) &&
         !same(plan.d4_batch.pair_offsets, pairs.pair_offsets, BindingField::kD4))) {
      return false;
    }
  } else if (!same(plan.aes2_batch.pair_offsets, plan.geometry_batch.pair_offsets,
                   BindingField::kAES2) ||
             (component_enabled(plan, Gfn2SccPotentialComponent::kD4TwoBody) &&
              !same(plan.d4_batch.pair_offsets, plan.geometry_batch.pair_offsets,
                    BindingField::kD4))) {
    /* Production plans currently use kNone and a setup-owned dense pair
     * partition.  Geometry is the authority for that non-topology domain. */
    return false;
  }
  return true;
}

bool validate_optional_plan_canonicalization(const Gfn2SccIterationDevicePlan& plan,
                                             Validator& validator) noexcept {
  if (!component_enabled(plan, Gfn2SccPotentialComponent::kD4TwoBody)) {
    const auto& batch = plan.d4_batch;
    const auto& parameters = plan.d4_parameters;
    const auto& cache = plan.d4_cache;
    if (batch.batch_size != 0 || batch.total_atoms != 0 || batch.total_pairs != 0 ||
        batch.plan_token != 0u || batch.atomic_number_hash != 0u || batch.atom_offsets != nullptr ||
        batch.pair_offsets != nullptr || batch.atomic_numbers != nullptr ||
        parameters.elements != nullptr || parameters.element_count != 0 ||
        parameters.references != nullptr || parameters.reference_count != 0 ||
        parameters.reference_c6 != nullptr || parameters.reference_c6_elements != 0 ||
        cache.pair_data != nullptr || cache.pair_data_elements != 0 ||
        cache.coordination_numbers != nullptr || cache.coordination_elements != 0 ||
        cache.geometry_generation != 0u || cache.plan_token != 0u) {
      return validator.fail(BindingError::kInvalidCount, BindingField::kD4);
    }
  }
  if (!component_enabled(plan, Gfn2SccPotentialComponent::kExplicitPointCharge)) {
    const auto& batch = plan.explicit_point_charge_batch;
    const auto& cache = plan.explicit_point_charge_cache;
    if (batch.batch_size != 0 || batch.total_atoms != 0 || batch.total_shells != 0 ||
        batch.total_point_charges != 0 || batch.atom_offsets != nullptr ||
        batch.batch_shell_offsets != nullptr || batch.point_charge_offsets != nullptr ||
        batch.shell_to_atom != nullptr || batch.shell_hardness != nullptr ||
        batch.qm_positions != nullptr || batch.point_positions != nullptr ||
        batch.point_charges != nullptr || batch.point_hardnesses != nullptr ||
        batch.plan_token != 0u || cache.shell_potentials != nullptr || cache.shell_elements != 0 ||
        cache.geometry_generation != 0u || cache.plan_token != 0u) {
      return validator.fail(BindingError::kInvalidCount, BindingField::kExplicitPointCharge);
    }
  }
  if (!component_enabled(plan, Gfn2SccPotentialComponent::kPeriodicEmbedding)) {
    const auto& batch = plan.periodic_batch;
    if (batch.batch_size != 0 || batch.total_atoms != 0 || batch.total_matrix_elements != 0 ||
        batch.atom_offset_count != 0 || batch.matrix_offset_count != 0 ||
        batch.shift_elements != 0 || batch.response_elements != 0 || batch.plan_token != 0u ||
        batch.atom_offsets != nullptr || batch.matrix_offsets != nullptr ||
        batch.shifts != nullptr || batch.response_matrices != nullptr ||
        batch.geometry_generation != 0u) {
      return validator.fail(BindingError::kInvalidCount, BindingField::kPeriodicEmbedding);
    }
  }
  return true;
}

bool validate_provider(const Gfn2SccIterationDevicePlan& plan,
                       const Gfn2SccIterationDeviceWorkspace& workspace,
                       Validator& validator) noexcept {
  const auto& provider = plan.eigensolver_provider;
  const auto& solver_workspace = workspace.eigensolver_workspace;
  if (provider.bucket_count <= 0 || provider.buckets == nullptr || provider.solver == nullptr ||
      provider.parameters == nullptr || provider.blas == nullptr ||
      (provider.capture_mode != Gfn2SccIterationProviderCaptureMode::kGraphSupported &&
       provider.capture_mode != Gfn2SccIterationProviderCaptureMode::kUncapturedSegmentRequired)) {
    return validator.fail(BindingError::kInvalidProvider, BindingField::kEigensolver);
  }
  if (provider.device_workspace_bytes < provider.requirements.solver_device_workspace_bytes ||
      provider.host_workspace_bytes < provider.requirements.solver_host_workspace_bytes) {
    return validator.fail(BindingError::kInsufficientCapacity, BindingField::kEigensolver);
  }
  if ((provider.device_workspace_bytes != 0u && provider.device_workspace == nullptr) ||
      (provider.host_workspace_bytes != 0u && provider.host_workspace == nullptr)) {
    return validator.fail(BindingError::kNullPointer, BindingField::kEigensolver);
  }
  if (provider.device_workspace_bytes >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    return validator.fail(BindingError::kAddressOverflow, BindingField::kEigensolver);
  }
  if ((provider.device_workspace != nullptr &&
       reinterpret_cast<std::uintptr_t>(provider.device_workspace) % alignof(double) != 0u) ||
      (provider.host_workspace != nullptr &&
       reinterpret_cast<std::uintptr_t>(provider.host_workspace) % alignof(std::max_align_t) !=
           0u)) {
    return validator.fail(BindingError::kMisalignedPointer, BindingField::kEigensolver);
  }
  if (solver_workspace.solver_device_workspace != provider.device_workspace ||
      solver_workspace.solver_device_workspace_bytes != provider.device_workspace_bytes ||
      solver_workspace.solver_host_workspace != provider.host_workspace ||
      solver_workspace.solver_host_workspace_bytes != provider.host_workspace_bytes) {
    return validator.fail(BindingError::kInvalidZeroCopyView, BindingField::kEigensolver);
  }

  std::int64_t system_cursor = 0;
  std::int64_t matrix_cursor = 0;
  std::int64_t orbital_cursor = 0;
  std::int64_t solve_cursor = 0;
  std::int64_t spin_matrix_cursor = 0;
  std::int64_t spin_orbital_cursor = 0;
  for (std::int64_t index = 0; index < provider.bucket_count; ++index) {
    const Gfn2EigensolverBucket& bucket = provider.buckets[index];
    if (bucket.orbital_count <= 0 || bucket.system_count <= 0 ||
        bucket.system_index_offset != system_cursor ||
        bucket.matrix_scratch_offset != matrix_cursor ||
        bucket.orbital_scratch_offset != orbital_cursor || bucket.solve_count <= 0 ||
        bucket.solve_index_offset != solve_cursor ||
        bucket.spin_matrix_scratch_offset != spin_matrix_cursor ||
        bucket.spin_orbital_scratch_offset != spin_orbital_cursor) {
      return validator.fail(BindingError::kInvalidBucket, BindingField::kEigensolver, index);
    }
    std::int64_t matrix_per_system = 0;
    std::int64_t bucket_matrices = 0;
    std::int64_t bucket_orbitals = 0;
    std::int64_t bucket_spin_matrices = 0;
    std::int64_t bucket_spin_orbitals = 0;
    if (!checked_multiply(bucket.orbital_count, bucket.orbital_count, matrix_per_system) ||
        !checked_multiply(matrix_per_system, bucket.system_count, bucket_matrices) ||
        !checked_multiply(bucket.orbital_count, bucket.system_count, bucket_orbitals) ||
        !checked_multiply(matrix_per_system, bucket.solve_count, bucket_spin_matrices) ||
        !checked_multiply(bucket.orbital_count, bucket.solve_count, bucket_spin_orbitals) ||
        !checked_add(system_cursor, bucket.system_count, system_cursor) ||
        !checked_add(matrix_cursor, bucket_matrices, matrix_cursor) ||
        !checked_add(orbital_cursor, bucket_orbitals, orbital_cursor) ||
        !checked_add(solve_cursor, bucket.solve_count, solve_cursor) ||
        !checked_add(spin_matrix_cursor, bucket_spin_matrices, spin_matrix_cursor) ||
        !checked_add(spin_orbital_cursor, bucket_spin_orbitals, spin_orbital_cursor)) {
      return validator.fail(BindingError::kInvalidBucket, BindingField::kEigensolver, index);
    }
  }
  if (system_cursor != plan.topology.batch_size ||
      matrix_cursor != plan.topology.total_matrix_elements ||
      orbital_cursor != plan.topology.total_orbitals ||
      solve_cursor != plan.wavefunction_layout.total_spin_channels ||
      spin_matrix_cursor != plan.wavefunction_layout.total_spin_matrix_elements ||
      spin_orbital_cursor != plan.wavefunction_layout.total_spin_orbitals ||
      plan.eigensolver_batch.bucket_system_count != system_cursor) {
    return validator.fail(BindingError::kInvalidBucket, BindingField::kEigensolver);
  }
  if (!validator.capacity(solver_workspace.matrix_a_elements, spin_matrix_cursor,
                          BindingField::kEigensolver) ||
      !validator.capacity(solver_workspace.matrix_b_elements, spin_matrix_cursor,
                          BindingField::kEigensolver) ||
      !validator.capacity(solver_workspace.eigenvalue_elements, spin_orbital_cursor,
                          BindingField::kEigensolver) ||
      !validator.capacity(solver_workspace.factor_pointer_elements, solve_cursor,
                          BindingField::kEigensolver) ||
      !validator.capacity(solver_workspace.matrix_pointer_elements, solve_cursor,
                          BindingField::kEigensolver) ||
      !validator.capacity(solver_workspace.info_a_elements, solve_cursor,
                          BindingField::kEigensolver) ||
      !validator.capacity(solver_workspace.info_b_elements, solve_cursor,
                          BindingField::kEigensolver) ||
      !validator.capacity(solver_workspace.eligible_elements, solve_cursor,
                          BindingField::kEigensolver) ||
      !validator.capacity(solver_workspace.compact_system_elements, solve_cursor,
                          BindingField::kEigensolver) ||
      !validator.capacity(solver_workspace.compact_source_slot_elements, solve_cursor,
                          BindingField::kEigensolver) ||
      !validator.capacity(solver_workspace.bucket_activity_elements, provider.bucket_count,
                          BindingField::kEigensolver) ||
      !validator.exact_count(solver_workspace.sequence_active_elements, 1,
                             BindingField::kEigensolver)) {
    return false;
  }
  return true;
}

bool validate_stage_reports(const Gfn2SccIterationDevicePlan& plan,
                            const Gfn2SccIterationDeviceWorkspace& workspace,
                            Validator& validator) noexcept {
  struct ReportSpec {
    Gfn2SccStageCodeFormat format = Gfn2SccStageCodeFormat::kUint32Error;
    Gfn2SccStageDeviceCodeRole role = Gfn2SccStageDeviceCodeRole::kMixedFirstError;
    std::uint64_t mask = 0u;
    xtbloom_status_t failure = XTBLOOM_STATUS_INTERNAL_ERROR;
  };
  const auto spec_for = [&](Gfn2SccStageId stage, ReportSpec& spec) {
    switch (stage) {
      case Gfn2SccStageId::kMixedGather:
        spec.mask = 0xfcu;
        return true;
      case Gfn2SccStageId::kSpinPotential:
      case Gfn2SccStageId::kSpinRawEnergy:
        spec.mask = kGfn2SpinDevicePeerErrorMask;
        return true;
      case Gfn2SccStageId::kES2Potential:
        spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
        spec.mask = 0x380u;
        return true;
      case Gfn2SccStageId::kES3Potential:
        spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
        spec.mask = 0x1cu;
        return true;
      case Gfn2SccStageId::kAES2Potential:
        spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
        spec.mask = 0x706u;
        return true;
      case Gfn2SccStageId::kD4Potential:
        if (!component_enabled(plan, Gfn2SccPotentialComponent::kD4TwoBody)) return false;
        spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
        spec.mask = 0xf0u;
        return true;
      case Gfn2SccStageId::kPeriodicPotential:
        if (!component_enabled(plan, Gfn2SccPotentialComponent::kPeriodicEmbedding)) return false;
        spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
        spec.mask = 0xecu;
        return true;
      case Gfn2SccStageId::kPotentialCompose:
        spec.mask = 0xff0cu;
        return true;
      case Gfn2SccStageId::kScalarBridge:
        spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
        spec.mask = 0xf0u;
        return true;
      case Gfn2SccStageId::kHamiltonian:
        spec.mask = 0x1feu;
        return true;
      case Gfn2SccStageId::kEigensolver:
        spec.mask = 0x3e06u;
        spec.failure = XTBLOOM_STATUS_EIGENSOLVER_FAILED;
        return true;
      case Gfn2SccStageId::kOccupations:
        spec.mask = 0x3feu;
        spec.failure = XTBLOOM_STATUS_EIGENSOLVER_FAILED;
        return true;
      case Gfn2SccStageId::kDensity:
        spec.mask = 0x7feu;
        spec.failure = XTBLOOM_STATUS_EIGENSOLVER_FAILED;
        return true;
      case Gfn2SccStageId::kMulliken:
        spec.mask = 0x1feu;
        return true;
      case Gfn2SccStageId::kClassicalEnergy:
        spec.mask = 0x1feu;
        return true;
      case Gfn2SccStageId::kElectronicEnergy:
        spec.mask = 0xfcu;
        return true;
      case Gfn2SccStageId::kFreeEnergy:
        spec.mask = 0x1ffeu;
        return true;
      case Gfn2SccStageId::kMixer:
        spec.format = Gfn2SccStageCodeFormat::kXTBloomStatus;
        spec.mask = 0x40u;
        return true;
      case Gfn2SccStageId::kStatePublication:
        spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
        spec.mask = 0x1f8u;
        return true;
      case Gfn2SccStageId::kES2RawEnergy:
        spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
        spec.mask = 0x980u;
        return true;
      case Gfn2SccStageId::kES3RawEnergy:
        spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
        spec.mask = 0x4cu;
        return true;
      case Gfn2SccStageId::kAES2RawEnergy:
        spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
        spec.mask = 0x1b06u;
        return true;
      case Gfn2SccStageId::kD4RawEnergy:
        if (!component_enabled(plan, Gfn2SccPotentialComponent::kD4TwoBody)) return false;
        spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
        spec.mask = 0xf0u;
        return true;
      case Gfn2SccStageId::kExplicitPointChargeRawEnergy:
        if (!component_enabled(plan, Gfn2SccPotentialComponent::kExplicitPointCharge)) return false;
        spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
        spec.mask = 0x60u;
        return true;
      case Gfn2SccStageId::kPeriodicRawEnergy:
        if (!component_enabled(plan, Gfn2SccPotentialComponent::kPeriodicEmbedding)) return false;
        spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
        spec.mask = 0x174u;
        return true;
      default:
        return false;
    }
  };

  std::int64_t expected_count = kGfn2SccIterationBaseStageReportCount;
  if (component_enabled(plan, Gfn2SccPotentialComponent::kD4TwoBody)) expected_count += 2;
  if (component_enabled(plan, Gfn2SccPotentialComponent::kExplicitPointCharge)) expected_count += 1;
  if (component_enabled(plan, Gfn2SccPotentialComponent::kPeriodicEmbedding)) expected_count += 2;
  if (plan.report_count != expected_count ||
      plan.report_count > kGfn2SccIterationMaximumStageReportCount ||
      plan.report_count > kGfn2SccIterationStageReportCapacity) {
    return validator.fail(BindingError::kInvalidStageReport, BindingField::kStageReports);
  }
  const std::int64_t batch = plan.topology.batch_size;
  std::uint64_t seen = 0u;
  std::uint64_t expected = 0u;
  for (std::uint32_t raw = 1u; raw <= static_cast<std::uint32_t>(Gfn2SccStageId::kSpinRawEnergy);
       ++raw) {
    ReportSpec spec{};
    if (spec_for(static_cast<Gfn2SccStageId>(raw), spec))
      expected |= std::uint64_t{1} << (raw - 1u);
  }
  for (std::int64_t index = 0; index < plan.report_count; ++index) {
    const Gfn2SccStageDeviceReport& report = plan.reports[index];
    ReportSpec spec{};
    if (!spec_for(report.stage, spec)) {
      return validator.fail(BindingError::kInvalidStageReport, BindingField::kStageReports, index);
    }
    const bool mixer = report.stage == Gfn2SccStageId::kMixer;
    const std::uint32_t raw_stage = static_cast<std::uint32_t>(report.stage);
    if (!gfn2_scc_stage_id_is_valid(report.stage) || report.system_code_format != spec.format ||
        report.device_code_role != spec.role || report.peer_error_mask != spec.mask ||
        report.peer_failure_status != spec.failure || report.plan_token != plan.plan_token ||
        report.system_codes == nullptr || report.system_code_elements != batch ||
        (mixer ? report.device_error != nullptr || report.device_error_elements != 0
               : report.device_error == nullptr || report.device_error_elements != 1) ||
        report.stage_sequence_active == nullptr || report.stage_sequence_elements != 1 ||
        (report.peer_error_mask & 1u) != 0u) {
      return validator.fail(BindingError::kInvalidStageReport, BindingField::kStageReports, index);
    }
    const std::uint64_t bit = std::uint64_t{1} << (raw_stage - 1u);
    if ((seen & bit) != 0u) {
      return validator.fail(BindingError::kInvalidStageReport, BindingField::kStageReports, index);
    }
    seen |= bit;
    const std::size_t system_alignment =
        report.system_code_format == Gfn2SccStageCodeFormat::kUint32Error
            ? alignof(std::uint32_t)
            : alignof(xtbloom_status_t);
    if (reinterpret_cast<std::uintptr_t>(report.system_codes) % system_alignment != 0u ||
        (!mixer &&
         reinterpret_cast<std::uintptr_t>(report.device_error) % alignof(std::uint32_t) != 0u) ||
        reinterpret_cast<std::uintptr_t>(report.stage_sequence_active) % alignof(std::uint32_t) !=
            0u) {
      return validator.fail(BindingError::kMisalignedPointer, BindingField::kStageReports, index);
    }
  }
  if (seen != expected) {
    return validator.fail(BindingError::kInvalidStageReport, BindingField::kStageReports);
  }
  const auto find_report = [&](Gfn2SccStageId stage) -> const Gfn2SccStageDeviceReport* {
    for (std::int64_t index = 0; index < plan.report_count; ++index) {
      if (plan.reports[index].stage == stage) {
        return &plan.reports[index];
      }
    }
    return nullptr;
  };
  const auto* mixer_report = find_report(Gfn2SccStageId::kMixer);
  if (mixer_report == nullptr ||
      mixer_report->system_code_format != Gfn2SccStageCodeFormat::kXTBloomStatus ||
      !same_pointer(static_cast<const xtbloom_status_t*>(mixer_report->system_codes),
                    workspace.staged_mixer.system_statuses) ||
      mixer_report->device_error != nullptr || mixer_report->device_error_elements != 0) {
    /* The mixer report deliberately omits its different-domain tracing scalar. */
    return validator.fail(BindingError::kInvalidStageReport, BindingField::kMixer);
  }
  const auto* publication_report = find_report(Gfn2SccStageId::kStatePublication);
  if (publication_report == nullptr ||
      publication_report->system_code_format != Gfn2SccStageCodeFormat::kUint32Error ||
      publication_report->device_code_role != Gfn2SccStageDeviceCodeRole::kPlanOnly ||
      publication_report->peer_error_mask != 0x1f8u ||
      !same_pointer(static_cast<const std::uint32_t*>(publication_report->system_codes),
                    workspace.publication_workspace.system_errors) ||
      publication_report->device_error != workspace.publication_workspace.device_error ||
      publication_report->stage_sequence_active !=
          workspace.publication_workspace.sequence_active) {
    return validator.fail(BindingError::kInvalidStageReport, BindingField::kStatePublication);
  }
  for (std::int64_t index = 0; index < plan.report_count; ++index) {
    const auto& report = plan.reports[index];
    const std::uint32_t* expected_sequence = nullptr;
    switch (report.stage) {
      case Gfn2SccStageId::kMixedGather:
      case Gfn2SccStageId::kPotentialCompose:
        expected_sequence = workspace.potential_workspace.sequence_active;
        break;
      case Gfn2SccStageId::kScalarBridge:
        expected_sequence = workspace.scalar_bridge.workspace.sequence_active;
        break;
      case Gfn2SccStageId::kPeriodicPotential:
      case Gfn2SccStageId::kPeriodicRawEnergy:
        expected_sequence = workspace.periodic_workspace.sequence_active;
        break;
      case Gfn2SccStageId::kHamiltonian:
        expected_sequence = workspace.hamiltonian_workspace.sequence_active;
        break;
      case Gfn2SccStageId::kEigensolver:
        expected_sequence = workspace.eigensolver_workspace.sequence_active;
        break;
      case Gfn2SccStageId::kOccupations:
        expected_sequence = workspace.occupations_workspace.sequence_active;
        break;
      case Gfn2SccStageId::kDensity:
        expected_sequence = workspace.density_workspace.sequence_active;
        break;
      case Gfn2SccStageId::kMulliken:
        expected_sequence = workspace.mulliken_workspace.sequence_active;
        break;
      case Gfn2SccStageId::kSpinPotential:
      case Gfn2SccStageId::kSpinRawEnergy:
        expected_sequence = workspace.spin_workspace.sequence_active;
        break;
      case Gfn2SccStageId::kClassicalEnergy:
        expected_sequence = workspace.classical_energy_workspace.sequence_active;
        break;
      case Gfn2SccStageId::kElectronicEnergy:
        expected_sequence = workspace.electronic_energy_workspace.sequence_active;
        break;
      case Gfn2SccStageId::kFreeEnergy:
        expected_sequence = workspace.free_energy_workspace.sequence_active;
        break;
      case Gfn2SccStageId::kMixer:
        expected_sequence = workspace.mixer_workspace.sequence_active;
        break;
      case Gfn2SccStageId::kStatePublication:
        expected_sequence = workspace.publication_workspace.sequence_active;
        break;
      default:
        /* Split ES2/ES3/AES2/D4/PC stages have no primitive-owned sequence
         * field. The composer opens their report latch directly. */
        break;
    }
    if (expected_sequence != nullptr && report.stage_sequence_active != expected_sequence) {
      return validator.fail(BindingError::kInvalidStageReport, BindingField::kStageReports, index);
    }
    if ((report.stage == Gfn2SccStageId::kD4Potential ||
         report.stage == Gfn2SccStageId::kD4RawEnergy) &&
        !same_pointer(static_cast<const std::uint32_t*>(report.system_codes),
                      workspace.d4_workspace.system_errors)) {
      return validator.fail(BindingError::kInvalidStageReport, BindingField::kD4, index);
    }
  }
  return true;
}

bool validate_zero_copy_views(const Gfn2SccIterationDevicePlan& plan,
                              const Gfn2SccIterationDeviceInput& input,
                              const Gfn2SccIterationDeviceState& state,
                              const Gfn2SccIterationDeviceWorkspace& workspace,
                              Validator& validator) noexcept {
  const bool mixed_spin = plan.wavefunction_layout.total_spin_channels != plan.topology.batch_size;
  const auto equal = [&](bool condition, BindingField field, std::int64_t index = -1) {
    return condition || validator.fail(BindingError::kInvalidZeroCopyView, field, index);
  };
  const auto* active = workspace.ledger.active_mask;
  const auto* sequence = workspace.ledger.sequence_active;
  if (!equal(workspace.activity.active_mask == active &&
                 workspace.activity.sequence_active == sequence &&
                 workspace.potential_activity.active_mask == active &&
                 workspace.hamiltonian_activity.active_mask == active &&
                 workspace.mulliken_activity.active_mask == active &&
                 workspace.classical_energy_activity.active_mask == active &&
                 workspace.free_energy_activity.active_mask == active &&
                 plan.eigensolver_batch.active == active &&
                 plan.occupations_batch.active == active && input.density.active == active,
             BindingField::kActivity)) {
    return false;
  }
  if (!equal(input.activity_state.iterations == state.scc.iterations &&
                 input.activity_state.system_statuses == state.scc.system_statuses &&
                 input.activity_state.converged == state.scc.converged,
             BindingField::kActivity)) {
    return false;
  }
  if (!equal(input.mixed_fields.qsh == state.scc.current_inputs.shell_charges &&
                 input.mixed_fields.dipoles == state.scc.current_inputs.atomic_dipoles &&
                 input.mixed_fields.quadrupoles == state.scc.current_inputs.atomic_quadrupoles &&
                 input.mixed_spin.shell_populations == input.mixed_fields.qsh,
             BindingField::kPotential)) {
    return false;
  }
  if (!equal(workspace.mixed_topology.shell_charges == input.mixed_fields.qsh &&
                 workspace.mixed_topology.atomic_dipoles == input.mixed_fields.dipoles &&
                 workspace.mixed_topology.atomic_quadrupoles == input.mixed_fields.quadrupoles,
             BindingField::kPotential)) {
    /* Only qsh -> qat reduction needs conversion storage in the complete DAG. */
    return false;
  }

  const auto& storage = workspace.components;
  const auto& components = workspace.potential_components;
  if (!equal(components.enabled_components == plan.enabled_components &&
                 components.es2_shell == storage.es2_shell_potential &&
                 components.es3_shell == storage.es3_shell_potential &&
                 components.explicit_point_charge_shell ==
                     plan.explicit_point_charge_cache.shell_potentials &&
                 components.aes2_atomic == storage.aes2_atomic_potential &&
                 components.aes2_dipole == storage.aes2_dipole_potential &&
                 components.aes2_quadrupole == storage.aes2_quadrupole_potential &&
                 components.d4_atomic == storage.d4_atomic_potential &&
                 components.periodic_atomic == storage.periodic_atomic_potential,
             BindingField::kPotential)) {
    return false;
  }
  const double* expected_shell_potential =
      mixed_spin ? workspace.complete_potentials.shell : workspace.scalar_bridge.shell_scalar;
  if (!equal(
          workspace.scalar_bridge.fields.shell == workspace.complete_potentials.shell &&
              workspace.scalar_bridge.fields.atomic == workspace.complete_potentials.atomic &&
              input.hamiltonian.shell_scalar_potentials == expected_shell_potential &&
              input.hamiltonian.atomic_dipole_potentials == workspace.complete_potentials.dipole &&
              input.hamiltonian.atomic_quadrupole_potentials ==
                  workspace.complete_potentials.quadrupole &&
              workspace.hamiltonian.matrix == input.eigensolver_hamiltonians,
          BindingField::kHamiltonian)) {
    return false;
  }

  if (!equal(input.occupation_eigenvalues == workspace.staged_eigenpairs.eigenvalues &&
                 input.density.coefficients == workspace.staged_eigenpairs.coefficients &&
                 input.density.eigenvalues == workspace.staged_eigenpairs.eigenvalues &&
                 input.density.occupations == workspace.staged_occupations.occupations,
             BindingField::kDensity)) {
    return false;
  }
  if (!equal(input.mulliken.density == workspace.staged_density.density &&
                 input.mulliken.overlap == input.hamiltonian.overlap &&
                 input.mulliken.dipole_integrals == input.hamiltonian.dipole_integrals &&
                 input.mulliken.quadrupole_integrals == input.hamiltonian.quadrupole_integrals,
             BindingField::kMulliken)) {
    return false;
  }
  if (!equal(input.electronic_energy.density == workspace.staged_density.density &&
                 input.electronic_energy.h0 == input.hamiltonian.h0 &&
                 input.electronic_energy.entropies == workspace.staged_occupations.entropies,
             BindingField::kElectronicEnergy)) {
    return false;
  }
  if (!equal(input.raw_multipoles.shell_charges == workspace.staged_raw_population.qsh &&
                 input.raw_multipoles.atomic_dipoles == workspace.staged_raw_population.dipole &&
                 input.raw_multipoles.atomic_quadrupoles ==
                     workspace.staged_raw_population.quadrupole &&
                 input.raw_spin.shell_populations == workspace.staged_raw_population.qsh &&
                 workspace.spin_output.spin_energies == workspace.staged_spin_energies,
             BindingField::kMulliken)) {
    return false;
  }

  const auto& classical = input.classical_energy;
  const auto& free = input.free_energy;
  if (!equal(classical.es2 == storage.es2_energy && classical.es3 == storage.es3_energy &&
                 classical.aes2 == storage.aes2_energy &&
                 classical.d4_two_body == storage.d4_two_body_energy &&
                 classical.explicit_point_charge == storage.explicit_point_charge_energy &&
                 classical.periodic_embedding == storage.periodic_embedding_energy,
             BindingField::kClassicalEnergy) ||
      !equal(free.core == storage.core_energy &&
                 free.entropy == workspace.staged_occupations.entropies &&
                 free.es2 == storage.es2_energy && free.es3 == storage.es3_energy &&
                 free.aes2 == storage.aes2_energy && free.spin == workspace.staged_spin_energies &&
                 free.d4_two_body == storage.d4_two_body_energy &&
                 free.explicit_point_charge == storage.explicit_point_charge_energy &&
                 free.periodic_embedding == storage.periodic_embedding_energy &&
                 input.complete_free_energies == workspace.staged_free_energy.free_energy,
             BindingField::kFreeEnergy)) {
    return false;
  }

  if (!equal(state.published.shell_charges == state.raw_population.qsh &&
                 state.published.atomic_dipoles == state.raw_population.dipole &&
                 state.published.atomic_quadrupoles == state.raw_population.quadrupole,
             BindingField::kStatePublication)) {
    return false;
  }
  if (!equal(
          state.free_energy.es2 == state.classical_energy.es2 &&
              state.free_energy.es3 == state.classical_energy.es3 &&
              state.free_energy.aes2 == state.classical_energy.aes2 &&
              state.free_energy.spin == state.spin_energies &&
              state.free_energy.d4_two_body == state.classical_energy.d4_two_body &&
              state.free_energy.explicit_point_charge ==
                  state.classical_energy.explicit_point_charge &&
              state.free_energy.periodic_embedding == state.classical_energy.periodic_embedding &&
              workspace.staged_free_energy.es2 == workspace.staged_classical_energy.es2 &&
              workspace.staged_free_energy.es3 == workspace.staged_classical_energy.es3 &&
              workspace.staged_free_energy.aes2 == workspace.staged_classical_energy.aes2 &&
              workspace.staged_free_energy.spin == workspace.staged_spin_energies &&
              workspace.staged_free_energy.d4_two_body ==
                  workspace.staged_classical_energy.d4_two_body &&
              workspace.staged_free_energy.explicit_point_charge ==
                  workspace.staged_classical_energy.explicit_point_charge &&
              workspace.staged_free_energy.periodic_embedding ==
                  workspace.staged_classical_energy.periodic_embedding,
          BindingField::kFreeEnergy)) {
    return false;
  }

  const auto& public_view = state.publication;
  if (!equal(same_eigenpairs(public_view.wavefunction.eigenpairs, state.eigenpairs) &&
                 same_occupations(public_view.wavefunction.occupations, state.occupations) &&
                 same_density(public_view.wavefunction.density, state.density) &&
                 same_population(public_view.wavefunction.population, state.raw_population) &&
                 same_classical_diagnostics(public_view.energy.classical, state.classical_energy) &&
                 same_free_energy_diagnostics(public_view.energy.free_energy, state.free_energy) &&
                 public_view.energy.spin_energies == state.spin_energies &&
                 public_view.energy.spin_energy_elements == state.spin_energy_elements &&
                 same_mixer_state(public_view.mixer, state.mixer) &&
                 same_multipoles(public_view.published, state.published) &&
                 same_scc_state(public_view.scc, state.scc),
             BindingField::kStatePublication)) {
    return false;
  }

  const auto& staged_view = workspace.staged_publication;
  if (!equal(same_eigenpairs(staged_view.wavefunction.eigenpairs, workspace.staged_eigenpairs) &&
                 same_occupations(staged_view.wavefunction.occupations,
                                  workspace.staged_occupations) &&
                 same_density(staged_view.wavefunction.density, workspace.staged_density) &&
                 same_population(staged_view.wavefunction.population,
                                 workspace.staged_raw_population) &&
                 same_classical_diagnostics(staged_view.energy.classical,
                                            workspace.staged_classical_energy) &&
                 same_free_energy_diagnostics(staged_view.energy.free_energy,
                                              workspace.staged_free_energy) &&
                 staged_view.energy.spin_energies == workspace.staged_spin_energies &&
                 staged_view.energy.spin_energy_elements == workspace.staged_spin_energy_elements &&
                 same_mixer_state(staged_view.mixer, workspace.staged_mixer) &&
                 same_const_multipoles(staged_view.next_mixed, workspace.next_mixed),
             BindingField::kStatePublication)) {
    return false;
  }

  const auto& publication = workspace.publication_workspace;
  if (!equal(publication.mixed_atomic_charges == workspace.mixed_topology.atomic_charges &&
                 publication.mixed_atomic_charge_elements == workspace.mixed_topology.atom_elements,
             BindingField::kStatePublication)) {
    return false;
  }
  return true;
}

bool validate_core_buffers(const Gfn2SccIterationDevicePlan& plan,
                           const Gfn2SccIterationDeviceInput& input,
                           const Gfn2SccIterationDeviceState& state,
                           const Gfn2SccIterationDeviceWorkspace& workspace, std::int64_t dipoles,
                           std::int64_t quadrupoles, std::int64_t two_batch,
                           std::int64_t two_orbitals, std::int64_t mixer_vector,
                           Validator& validator) noexcept {
  const std::int64_t batch = plan.topology.batch_size;
  const std::int64_t shells = plan.topology.total_shells;
  const std::int64_t matrices = plan.topology.total_matrix_elements;
  const std::int64_t spin_shells = plan.wavefunction_layout.total_spin_shells;
  const std::int64_t spin_atoms = plan.wavefunction_layout.total_spin_atoms;
  const std::int64_t spin_orbitals = plan.wavefunction_layout.total_spin_orbitals;
  const std::int64_t spin_matrices = plan.wavefunction_layout.total_spin_matrix_elements;
  const bool mixed_spin = plan.wavefunction_layout.total_spin_channels != plan.topology.batch_size;
  std::int64_t spin_dipoles = 0;
  std::int64_t spin_quadrupoles = 0;
  if (!checked_multiply(spin_atoms, 3, spin_dipoles) ||
      !checked_multiply(spin_atoms, 6, spin_quadrupoles)) {
    return validator.fail(BindingError::kInvalidCount, BindingField::kSpin);
  }
  std::uint32_t group = 1u;
  const auto exact = [&](std::int64_t actual, std::int64_t expected, BindingField field,
                         std::int64_t index = -1) {
    return validator.exact_count(actual, expected, field, index);
  };
  const auto capacity = [&](std::int64_t actual, std::int64_t required, BindingField field,
                            std::int64_t index = -1) {
    return validator.capacity(actual, required, field, index);
  };
  const auto read = [&](const void* pointer, std::int64_t count, std::size_t size,
                        std::size_t alignment, BindingField field, std::int64_t index = -1) {
    return validator.pointer(pointer, count, size, alignment, field, index, false);
  };
  const auto write = [&](const void* pointer, std::int64_t count, std::size_t size,
                         std::size_t alignment, BindingField field, std::int64_t index = -1) {
    return validator.pointer(pointer, count, size, alignment, field, index, true, group++);
  };

  if (!exact(input.activity_state.batch_elements, batch, BindingField::kActivity) ||
      !exact(workspace.ledger.batch_elements, batch, BindingField::kActivity) ||
      !exact(workspace.ledger.scalar_elements, 1, BindingField::kActivity) ||
      !exact(workspace.activity.batch_elements, batch, BindingField::kActivity) ||
      !exact(workspace.activity.sequence_elements, 1, BindingField::kActivity) ||
      !exact(workspace.potential_activity.elements, batch, BindingField::kPotential) ||
      !exact(workspace.hamiltonian_activity.elements, batch, BindingField::kHamiltonian) ||
      !exact(workspace.mulliken_activity.elements, batch, BindingField::kMulliken) ||
      !exact(workspace.classical_energy_activity.elements, batch, BindingField::kClassicalEnergy) ||
      !exact(workspace.free_energy_activity.elements, batch, BindingField::kFreeEnergy)) {
    return false;
  }
  if (!write(workspace.ledger.active_mask, batch, sizeof(std::uint8_t), alignof(std::uint8_t),
             BindingField::kActivity, 0) ||
      !write(workspace.ledger.pending_statuses, batch, sizeof(xtbloom_status_t),
             alignof(xtbloom_status_t), BindingField::kActivity, 1) ||
      !write(workspace.ledger.system_failure_records, batch, sizeof(std::uint64_t),
             alignof(std::uint64_t), BindingField::kActivity, 2) ||
      !write(workspace.ledger.plan_failure_record, 1, sizeof(std::uint64_t), alignof(std::uint64_t),
             BindingField::kActivity, 3) ||
      !write(workspace.ledger.sequence_active, 1, sizeof(std::uint32_t), alignof(std::uint32_t),
             BindingField::kActivity, 4)) {
    return false;
  }

  if (!exact(input.mixed_fields.qsh_elements, spin_shells, BindingField::kPotential) ||
      !exact(input.mixed_fields.dipole_elements, spin_dipoles, BindingField::kPotential) ||
      !exact(input.mixed_fields.quadrupole_elements, spin_quadrupoles, BindingField::kPotential) ||
      !exact(input.mixed_spin.shell_population_elements, spin_shells, BindingField::kSpin) ||
      !exact(input.raw_spin.shell_population_elements, spin_shells, BindingField::kSpin) ||
      !exact(input.hamiltonian.h0_elements, matrices, BindingField::kHamiltonian) ||
      !exact(input.hamiltonian.overlap_elements, matrices, BindingField::kHamiltonian) ||
      !exact(input.hamiltonian.dipole_integral_elements, 3 * matrices,
             BindingField::kHamiltonian) ||
      !exact(input.hamiltonian.quadrupole_integral_elements, 6 * matrices,
             BindingField::kHamiltonian) ||
      !exact(input.hamiltonian.shell_scalar_elements, mixed_spin ? spin_shells : shells,
             BindingField::kHamiltonian) ||
      !exact(input.hamiltonian.atomic_dipole_elements, mixed_spin ? spin_dipoles : dipoles,
             BindingField::kHamiltonian) ||
      !exact(input.hamiltonian.atomic_quadrupole_elements,
             mixed_spin ? spin_quadrupoles : quadrupoles, BindingField::kHamiltonian) ||
      !exact(input.eigensolver_hamiltonian_elements, spin_matrices, BindingField::kEigensolver) ||
      !exact(input.occupation_eigenvalue_elements, spin_orbitals, BindingField::kOccupations) ||
      !exact(input.density.coefficient_elements, spin_matrices, BindingField::kDensity) ||
      !exact(input.density.eigenvalue_elements, spin_orbitals, BindingField::kDensity) ||
      !exact(input.density.occupation_elements, two_orbitals, BindingField::kDensity) ||
      !exact(input.density.active_elements, batch, BindingField::kDensity) ||
      !exact(input.mulliken.density_elements, spin_matrices, BindingField::kMulliken) ||
      !exact(input.mulliken.overlap_elements, matrices, BindingField::kMulliken) ||
      !exact(input.mulliken.dipole_integral_elements, 3 * matrices, BindingField::kMulliken) ||
      !exact(input.mulliken.quadrupole_integral_elements, 6 * matrices, BindingField::kMulliken) ||
      !exact(input.electronic_energy.density_elements, spin_matrices,
             BindingField::kElectronicEnergy) ||
      !exact(input.electronic_energy.h0_elements, matrices, BindingField::kElectronicEnergy) ||
      !exact(input.electronic_energy.entropy_elements, batch, BindingField::kElectronicEnergy) ||
      !exact(input.complete_free_energy_elements, batch, BindingField::kFreeEnergy) ||
      !exact(input.raw_multipoles.shell_elements, spin_shells, BindingField::kMulliken) ||
      !exact(input.raw_multipoles.dipole_elements, spin_dipoles, BindingField::kMulliken) ||
      !exact(input.raw_multipoles.quadrupole_elements, spin_quadrupoles, BindingField::kMulliken)) {
    return false;
  }

  /* Immutable numerical arrays that do not originate in staged storage. */
  if (!read(input.hamiltonian.h0, matrices, sizeof(double), alignof(double),
            BindingField::kHamiltonian, 0) ||
      !read(input.hamiltonian.overlap, matrices, sizeof(double), alignof(double),
            BindingField::kHamiltonian, 1) ||
      !read(input.hamiltonian.dipole_integrals, 3 * matrices, sizeof(double), alignof(double),
            BindingField::kHamiltonian, 2) ||
      !read(input.hamiltonian.quadrupole_integrals, 6 * matrices, sizeof(double), alignof(double),
            BindingField::kHamiltonian, 3) ||
      !read(plan.occupations_batch.electron_counts, two_batch, sizeof(double), alignof(double),
            BindingField::kOccupations, 0) ||
      !read(plan.occupations_batch.temperatures, batch, sizeof(double), alignof(double),
            BindingField::kOccupations, 1)) {
    return false;
  }

  const auto validate_eigenpairs = [&](const Gfn2EigensolverDeviceResults& results,
                                       BindingField field) {
    return exact(results.eigenvalue_elements, spin_orbitals, field) &&
           exact(results.coefficient_elements, spin_matrices, field) &&
           write(results.eigenvalues, spin_orbitals, sizeof(double), alignof(double), field, 0) &&
           write(results.coefficients, spin_matrices, sizeof(double), alignof(double), field, 1);
  };
  const auto validate_occupations = [&](const Gfn2OccupationsDeviceResults& results,
                                        BindingField field) {
    return exact(results.occupation_elements, two_orbitals, field) &&
           exact(results.chemical_potential_elements, two_batch, field) &&
           exact(results.electron_sum_elements, two_batch, field) &&
           exact(results.entropy_elements, batch, field) &&
           write(results.occupations, two_orbitals, sizeof(double), alignof(double), field, 0) &&
           write(results.chemical_potentials, two_batch, sizeof(double), alignof(double), field,
                 1) &&
           write(results.electron_sums, two_batch, sizeof(double), alignof(double), field, 2) &&
           write(results.entropies, batch, sizeof(double), alignof(double), field, 3);
  };
  const auto validate_density = [&](const Gfn2DensityDeviceResults& results, BindingField field) {
    return exact(results.density_elements, spin_matrices, field) &&
           exact(results.weighted_density_elements, spin_matrices, field) &&
           exact(results.band_energy_elements, batch, field) &&
           exact(results.occupation_sum_elements, batch, field) &&
           exact(results.density_trace_elements, batch, field) &&
           exact(results.weighted_density_trace_elements, batch, field) &&
           exact(results.channel_band_energy_elements, plan.wavefunction_layout.total_spin_channels,
                 field) &&
           exact(results.channel_occupation_sum_elements,
                 plan.wavefunction_layout.total_spin_channels, field) &&
           exact(results.channel_density_trace_elements,
                 plan.wavefunction_layout.total_spin_channels, field) &&
           exact(results.channel_weighted_density_trace_elements,
                 plan.wavefunction_layout.total_spin_channels, field) &&
           write(results.density, spin_matrices, sizeof(double), alignof(double), field, 0) &&
           write(results.energy_weighted_density, spin_matrices, sizeof(double), alignof(double),
                 field, 1) &&
           write(results.band_energies, batch, sizeof(double), alignof(double), field, 2) &&
           write(results.occupation_sums, batch, sizeof(double), alignof(double), field, 3) &&
           write(results.density_traces, batch, sizeof(double), alignof(double), field, 4) &&
           write(results.weighted_density_traces, batch, sizeof(double), alignof(double), field,
                 5) &&
           write(results.channel_band_energies, plan.wavefunction_layout.total_spin_channels,
                 sizeof(double), alignof(double), field, 6) &&
           write(results.channel_occupation_sums, plan.wavefunction_layout.total_spin_channels,
                 sizeof(double), alignof(double), field, 7) &&
           write(results.channel_density_traces, plan.wavefunction_layout.total_spin_channels,
                 sizeof(double), alignof(double), field, 8) &&
           write(results.channel_weighted_density_traces,
                 plan.wavefunction_layout.total_spin_channels, sizeof(double), alignof(double),
                 field, 9);
  };
  const auto validate_population = [&](const Gfn2MullikenDevicePopulation& population,
                                       BindingField field, bool include_public_alias) {
    if (!exact(population.qsh_elements, spin_shells, field) ||
        !exact(population.qat_elements, spin_atoms, field) ||
        !exact(population.dipole_elements, spin_dipoles, field) ||
        !exact(population.quadrupole_elements, spin_quadrupoles, field)) {
      return false;
    }
    if (include_public_alias) {
      /* qsh/d/Q are registered through state.published below. */
      return write(population.qat, spin_atoms, sizeof(double), alignof(double), field, 1);
    }
    return write(population.qsh, spin_shells, sizeof(double), alignof(double), field, 0) &&
           write(population.qat, spin_atoms, sizeof(double), alignof(double), field, 1) &&
           write(population.dipole, spin_dipoles, sizeof(double), alignof(double), field, 2) &&
           write(population.quadrupole, spin_quadrupoles, sizeof(double), alignof(double), field,
                 3);
  };
  if (!validate_eigenpairs(state.eigenpairs, BindingField::kStatePublication) ||
      !validate_occupations(state.occupations, BindingField::kStatePublication) ||
      !validate_density(state.density, BindingField::kStatePublication) ||
      !validate_population(state.raw_population, BindingField::kStatePublication, true) ||
      !validate_eigenpairs(workspace.staged_eigenpairs, BindingField::kEigensolver) ||
      !validate_occupations(workspace.staged_occupations, BindingField::kOccupations) ||
      !validate_density(workspace.staged_density, BindingField::kDensity) ||
      !validate_population(workspace.staged_raw_population, BindingField::kMulliken, false)) {
    return false;
  }
  if (!exact(state.spin_energy_elements, batch, BindingField::kStatePublication) ||
      !exact(workspace.staged_spin_energy_elements, batch, BindingField::kSpin) ||
      !exact(workspace.spin_output.spin_energy_elements, batch, BindingField::kSpin) ||
      !exact(workspace.spin_output.shell_potential_elements, spin_shells, BindingField::kSpin) ||
      !write(workspace.spin_output.shell_potentials, spin_shells, sizeof(double), alignof(double),
             BindingField::kSpin, 0)) {
    return false;
  }

  const auto validate_multipoles = [&](const Gfn2SccDeviceMultipoles& multipoles,
                                       BindingField field, bool register_ranges) {
    if (!exact(multipoles.shell_elements, spin_shells, field) ||
        !exact(multipoles.dipole_elements, spin_dipoles, field) ||
        !exact(multipoles.quadrupole_elements, spin_quadrupoles, field)) {
      return false;
    }
    return !register_ranges || (write(multipoles.shell_charges, spin_shells, sizeof(double),
                                      alignof(double), field, 0) &&
                                write(multipoles.atomic_dipoles, spin_dipoles, sizeof(double),
                                      alignof(double), field, 1) &&
                                write(multipoles.atomic_quadrupoles, spin_quadrupoles,
                                      sizeof(double), alignof(double), field, 2));
  };
  if (!validate_multipoles(state.published, BindingField::kStatePublication, true) ||
      !validate_multipoles(state.scc.current_inputs, BindingField::kStatePublication, true) ||
      !validate_multipoles(workspace.next_mixed, BindingField::kMixer, true)) {
    return false;
  }
  if (!exact(state.scc.batch_elements, batch, BindingField::kStatePublication) ||
      !write(state.scc.free_energies, batch, sizeof(double), alignof(double),
             BindingField::kStatePublication, 3) ||
      !write(state.scc.previous_free_energies, batch, sizeof(double), alignof(double),
             BindingField::kStatePublication, 4) ||
      !write(state.scc.free_energy_changes, batch, sizeof(double), alignof(double),
             BindingField::kStatePublication, 5) ||
      !write(state.scc.residual_rms, batch, sizeof(double), alignof(double),
             BindingField::kStatePublication, 6) ||
      !write(state.scc.iterations, batch, sizeof(std::uint64_t), alignof(std::uint64_t),
             BindingField::kStatePublication, 7) ||
      !write(state.scc.system_statuses, batch, sizeof(xtbloom_status_t), alignof(xtbloom_status_t),
             BindingField::kStatePublication, 8) ||
      !write(state.scc.converged, batch, sizeof(std::uint8_t), alignof(std::uint8_t),
             BindingField::kStatePublication, 9)) {
    return false;
  }

  const auto validate_mixer = [&](const Gfn2SccMixerDeviceState& mixer, BindingField field) {
    std::int64_t history = 0;
    std::int64_t omega = 0;
    if (!checked_multiply(mixer_vector, plan.mixer_policy.history_size, history) ||
        !checked_multiply(batch, plan.mixer_policy.history_size, omega) ||
        !exact(mixer.total_vector_elements, mixer_vector, field) ||
        !exact(mixer.history_elements, history, field) ||
        !exact(mixer.omega_elements, omega, field) || !exact(mixer.batch_elements, batch, field)) {
      return false;
    }
    return write(mixer.current_inputs, mixer_vector, sizeof(double), alignof(double), field, 0) &&
           write(mixer.previous_inputs, mixer_vector, sizeof(double), alignof(double), field, 1) &&
           write(mixer.previous_residuals, mixer_vector, sizeof(double), alignof(double), field,
                 2) &&
           write(mixer.df_history, history, sizeof(double), alignof(double), field, 3) &&
           write(mixer.u_history, history, sizeof(double), alignof(double), field, 4) &&
           write(mixer.omega, omega, sizeof(double), alignof(double), field, 5) &&
           write(mixer.residual_rms, batch, sizeof(double), alignof(double), field, 6) &&
           write(mixer.residual_maximum, batch, sizeof(double), alignof(double), field, 7) &&
           write(mixer.iterations, batch, sizeof(std::uint64_t), alignof(std::uint64_t), field,
                 8) &&
           write(mixer.restart_counts, batch, sizeof(std::uint64_t), alignof(std::uint64_t), field,
                 9) &&
           write(mixer.system_statuses, batch, sizeof(xtbloom_status_t), alignof(xtbloom_status_t),
                 field, 10) &&
           write(mixer.initialized, batch, sizeof(std::uint8_t), alignof(std::uint8_t), field,
                 11) &&
           write(mixer.residual_converged, batch, sizeof(std::uint8_t), alignof(std::uint8_t),
                 field, 12);
  };
  if (!validate_mixer(state.mixer, BindingField::kStatePublication) ||
      !validate_mixer(workspace.staged_mixer, BindingField::kMixer)) {
    return false;
  }

  (void)capacity;
  return true;
}

bool validate_component_and_energy_buffers(const Gfn2SccIterationDevicePlan& plan,
                                           const Gfn2SccIterationDeviceInput& input,
                                           const Gfn2SccIterationDeviceState& state,
                                           const Gfn2SccIterationDeviceWorkspace& workspace,
                                           std::int64_t dipoles, std::int64_t quadrupoles,
                                           Validator& validator) noexcept {
  const std::int64_t batch = plan.topology.batch_size;
  const std::int64_t atoms = plan.topology.total_atoms;
  const std::int64_t shells = plan.topology.total_shells;
  const std::int64_t spin_shells = plan.wavefunction_layout.total_spin_shells;
  const std::int64_t spin_atoms = plan.wavefunction_layout.total_spin_atoms;
  const std::int64_t spin_matrices = plan.wavefunction_layout.total_spin_matrix_elements;
  std::int64_t spin_dipoles = 0;
  std::int64_t spin_quadrupoles = 0;
  if (!checked_multiply(spin_atoms, 3, spin_dipoles) ||
      !checked_multiply(spin_atoms, 6, spin_quadrupoles)) {
    return validator.fail(BindingError::kInvalidCount, BindingField::kPotential);
  }
  std::uint32_t group = 1000u;
  const auto exact = [&](std::int64_t actual, std::int64_t expected, BindingField field,
                         std::int64_t index = -1) {
    return validator.exact_count(actual, expected, field, index);
  };
  const auto write = [&](const void* pointer, std::int64_t count, std::size_t size,
                         std::size_t alignment, BindingField field, std::int64_t index = -1) {
    return validator.pointer(pointer, count, size, alignment, field, index, true, group++);
  };

  const auto component = [&](Gfn2SccPotentialComponent bit, double* pointer, std::int64_t elements,
                             std::int64_t expected, BindingField field, std::int64_t index) {
    if (component_enabled(plan, bit)) {
      return exact(elements, expected, field, index) &&
             write(pointer, expected, sizeof(double), alignof(double), field, index);
    }
    return exact(elements, 0, field, index) && pointer == nullptr;
  };
  const auto& storage = workspace.components;
  if (!component(Gfn2SccPotentialComponent::kES2, storage.es2_shell_potential,
                 storage.es2_shell_elements, shells, BindingField::kES2, 0) ||
      !component(Gfn2SccPotentialComponent::kES3, storage.es3_shell_potential,
                 storage.es3_shell_elements, shells, BindingField::kES3, 0) ||
      !component(Gfn2SccPotentialComponent::kAES2, storage.aes2_atomic_potential,
                 storage.aes2_atomic_elements, atoms, BindingField::kAES2, 0) ||
      !component(Gfn2SccPotentialComponent::kAES2, storage.aes2_dipole_potential,
                 storage.aes2_dipole_elements, dipoles, BindingField::kAES2, 1) ||
      !component(Gfn2SccPotentialComponent::kAES2, storage.aes2_quadrupole_potential,
                 storage.aes2_quadrupole_elements, quadrupoles, BindingField::kAES2, 2) ||
      !component(Gfn2SccPotentialComponent::kD4TwoBody, storage.d4_atomic_potential,
                 storage.d4_atomic_elements, atoms, BindingField::kD4, 0) ||
      !component(Gfn2SccPotentialComponent::kPeriodicEmbedding, storage.periodic_atomic_potential,
                 storage.periodic_atomic_elements, atoms, BindingField::kPeriodicEmbedding, 0) ||
      !component(Gfn2SccPotentialComponent::kES2, storage.es2_energy, storage.es2_energy_elements,
                 batch, BindingField::kES2, 3) ||
      !component(Gfn2SccPotentialComponent::kES3, storage.es3_energy, storage.es3_energy_elements,
                 batch, BindingField::kES3, 3) ||
      !component(Gfn2SccPotentialComponent::kAES2, storage.aes2_energy,
                 storage.aes2_energy_elements, batch, BindingField::kAES2, 3) ||
      !component(Gfn2SccPotentialComponent::kD4TwoBody, storage.d4_two_body_energy,
                 storage.d4_two_body_energy_elements, batch, BindingField::kD4, 3) ||
      !component(Gfn2SccPotentialComponent::kExplicitPointCharge,
                 storage.explicit_point_charge_energy,
                 storage.explicit_point_charge_energy_elements, batch,
                 BindingField::kExplicitPointCharge, 3) ||
      !component(Gfn2SccPotentialComponent::kPeriodicEmbedding, storage.periodic_embedding_energy,
                 storage.periodic_embedding_energy_elements, batch,
                 BindingField::kPeriodicEmbedding, 3) ||
      !exact(storage.core_energy_elements, batch, BindingField::kElectronicEnergy) ||
      !write(storage.core_energy, batch, sizeof(double), alignof(double),
             BindingField::kElectronicEnergy, 0) ||
      !exact(storage.electronic_free_energy_elements, batch, BindingField::kElectronicEnergy) ||
      !write(storage.electronic_free_energy, batch, sizeof(double), alignof(double),
             BindingField::kElectronicEnergy, 1)) {
    return false;
  }
  const auto validate_component_projection =
      [&](const double* pointer, std::int64_t elements, Gfn2SccPotentialComponent bit,
          std::int64_t expected, BindingField field, std::int64_t index) {
        if (component_enabled(plan, bit)) {
          return exact(elements, expected, field, index) && pointer != nullptr;
        }
        return exact(elements, 0, field, index) && pointer == nullptr;
      };
  const auto& projection = workspace.potential_components;
  if (!validate_component_projection(projection.es2_shell, projection.es2_shell_elements,
                                     Gfn2SccPotentialComponent::kES2, shells, BindingField::kES2,
                                     4) ||
      !validate_component_projection(projection.es3_shell, projection.es3_shell_elements,
                                     Gfn2SccPotentialComponent::kES3, shells, BindingField::kES3,
                                     4) ||
      !validate_component_projection(projection.aes2_atomic, projection.aes2_atomic_elements,
                                     Gfn2SccPotentialComponent::kAES2, atoms, BindingField::kAES2,
                                     4) ||
      !validate_component_projection(projection.aes2_dipole, projection.aes2_dipole_elements,
                                     Gfn2SccPotentialComponent::kAES2, dipoles, BindingField::kAES2,
                                     5) ||
      !validate_component_projection(
          projection.aes2_quadrupole, projection.aes2_quadrupole_elements,
          Gfn2SccPotentialComponent::kAES2, quadrupoles, BindingField::kAES2, 6) ||
      !validate_component_projection(projection.d4_atomic, projection.d4_atomic_elements,
                                     Gfn2SccPotentialComponent::kD4TwoBody, atoms,
                                     BindingField::kD4, 4) ||
      !validate_component_projection(projection.explicit_point_charge_shell,
                                     projection.explicit_point_charge_shell_elements,
                                     Gfn2SccPotentialComponent::kExplicitPointCharge, shells,
                                     BindingField::kExplicitPointCharge, 4) ||
      !validate_component_projection(projection.periodic_atomic,
                                     projection.periodic_atomic_elements,
                                     Gfn2SccPotentialComponent::kPeriodicEmbedding, atoms,
                                     BindingField::kPeriodicEmbedding, 4)) {
    return false;
  }

  if (!exact(workspace.mixed_topology.shell_elements, spin_shells, BindingField::kPotential) ||
      !exact(workspace.mixed_topology.atom_elements, spin_atoms, BindingField::kPotential) ||
      !exact(workspace.mixed_topology.dipole_elements, spin_dipoles, BindingField::kPotential) ||
      !exact(workspace.mixed_topology.quadrupole_elements, spin_quadrupoles,
             BindingField::kPotential) ||
      !write(workspace.mixed_topology.atomic_charges, spin_atoms, sizeof(double), alignof(double),
             BindingField::kPotential, 10) ||
      !exact(workspace.physical_topology.shell_elements, shells, BindingField::kPotential) ||
      !exact(workspace.physical_topology.atom_elements, atoms, BindingField::kPotential) ||
      !exact(workspace.physical_topology.dipole_elements, dipoles, BindingField::kPotential) ||
      !exact(workspace.physical_topology.quadrupole_elements, quadrupoles,
             BindingField::kPotential) ||
      !write(workspace.physical_topology.shell_charges, shells, sizeof(double), alignof(double),
             BindingField::kPotential, 15) ||
      !write(workspace.physical_topology.atomic_charges, atoms, sizeof(double), alignof(double),
             BindingField::kPotential, 16) ||
      !write(workspace.physical_topology.atomic_dipoles, dipoles, sizeof(double), alignof(double),
             BindingField::kPotential, 17) ||
      !write(workspace.physical_topology.atomic_quadrupoles, quadrupoles, sizeof(double),
             alignof(double), BindingField::kPotential, 18) ||
      !exact(workspace.complete_potentials.shell_elements, spin_shells, BindingField::kPotential) ||
      !exact(workspace.complete_potentials.atom_elements, spin_atoms, BindingField::kPotential) ||
      !exact(workspace.complete_potentials.dipole_elements, spin_dipoles,
             BindingField::kPotential) ||
      !exact(workspace.complete_potentials.quadrupole_elements, spin_quadrupoles,
             BindingField::kPotential) ||
      !write(workspace.complete_potentials.shell, spin_shells, sizeof(double), alignof(double),
             BindingField::kPotential, 11) ||
      !write(workspace.complete_potentials.atomic, spin_atoms, sizeof(double), alignof(double),
             BindingField::kPotential, 12) ||
      !write(workspace.complete_potentials.dipole, spin_dipoles, sizeof(double), alignof(double),
             BindingField::kPotential, 13) ||
      !write(workspace.complete_potentials.quadrupole, spin_quadrupoles, sizeof(double),
             alignof(double), BindingField::kPotential, 14) ||
      !exact(workspace.scalar_bridge.fields.shell_elements, spin_shells,
             BindingField::kScalarBridge) ||
      !exact(workspace.scalar_bridge.fields.atom_elements, spin_atoms,
             BindingField::kScalarBridge) ||
      !exact(workspace.scalar_bridge.shell_elements, shells, BindingField::kScalarBridge) ||
      !write(workspace.scalar_bridge.shell_scalar, shells, sizeof(double), alignof(double),
             BindingField::kScalarBridge, 0) ||
      !exact(workspace.hamiltonian.elements, spin_matrices, BindingField::kHamiltonian) ||
      !write(workspace.hamiltonian.matrix, spin_matrices, sizeof(double), alignof(double),
             BindingField::kHamiltonian, 0)) {
    return false;
  }

  const auto validate_classical_input = [&](const Gfn2SccClassicalEnergyDeviceInput& values) {
    const auto check = [&](const double* pointer, std::int64_t elements,
                           Gfn2SccPotentialComponent bit, std::int64_t index) {
      return component_enabled(plan, bit)
                 ? exact(elements, batch, BindingField::kClassicalEnergy, index) &&
                       pointer != nullptr
                 : exact(elements, 0, BindingField::kClassicalEnergy, index) && pointer == nullptr;
    };
    return check(values.es2, values.es2_elements, Gfn2SccPotentialComponent::kES2, 0) &&
           check(values.es3, values.es3_elements, Gfn2SccPotentialComponent::kES3, 1) &&
           check(values.aes2, values.aes2_elements, Gfn2SccPotentialComponent::kAES2, 2) &&
           check(values.d4_two_body, values.d4_two_body_elements,
                 Gfn2SccPotentialComponent::kD4TwoBody, 3) &&
           check(values.explicit_point_charge, values.explicit_point_charge_elements,
                 Gfn2SccPotentialComponent::kExplicitPointCharge, 4) &&
           check(values.periodic_embedding, values.periodic_embedding_elements,
                 Gfn2SccPotentialComponent::kPeriodicEmbedding, 5);
  };
  if (!validate_classical_input(input.classical_energy)) {
    return false;
  }
  const auto validate_free_input = [&](const Gfn2SccFreeEnergyDeviceInput& values) {
    const auto check = [&](const double* pointer, std::int64_t elements,
                           Gfn2SccPotentialComponent bit, std::int64_t index) {
      return component_enabled(plan, bit)
                 ? exact(elements, batch, BindingField::kFreeEnergy, index) && pointer != nullptr
                 : exact(elements, 0, BindingField::kFreeEnergy, index) && pointer == nullptr;
    };
    return exact(values.core_elements, batch, BindingField::kFreeEnergy, 0) &&
           values.core != nullptr &&
           exact(values.entropy_elements, batch, BindingField::kFreeEnergy, 1) &&
           values.entropy != nullptr &&
           check(values.es2, values.es2_elements, Gfn2SccPotentialComponent::kES2, 2) &&
           check(values.es3, values.es3_elements, Gfn2SccPotentialComponent::kES3, 3) &&
           check(values.aes2, values.aes2_elements, Gfn2SccPotentialComponent::kAES2, 4) &&
           exact(values.spin_elements, batch, BindingField::kFreeEnergy, 5) &&
           values.spin != nullptr &&
           check(values.d4_two_body, values.d4_two_body_elements,
                 Gfn2SccPotentialComponent::kD4TwoBody, 6) &&
           check(values.explicit_point_charge, values.explicit_point_charge_elements,
                 Gfn2SccPotentialComponent::kExplicitPointCharge, 7) &&
           check(values.periodic_embedding, values.periodic_embedding_elements,
                 Gfn2SccPotentialComponent::kPeriodicEmbedding, 8);
  };
  if (!validate_free_input(input.free_energy)) {
    return false;
  }

  const auto validate_classical_diagnostics = [&](const Gfn2SccClassicalEnergyDeviceDiagnostics& d,
                                                  BindingField field) {
    const std::array<std::pair<double*, std::int64_t>, 7> fields{{
        {d.es2, d.es2_elements},
        {d.es3, d.es3_elements},
        {d.aes2, d.aes2_elements},
        {d.d4_two_body, d.d4_two_body_elements},
        {d.explicit_point_charge, d.explicit_point_charge_elements},
        {d.periodic_embedding, d.periodic_embedding_elements},
        {d.classical_total, d.classical_total_elements},
    }};
    for (std::size_t index = 0; index < fields.size(); ++index) {
      if (!exact(fields[index].second, batch, field, static_cast<std::int64_t>(index))) {
        return false;
      }
      /* Shared component arrays are registered through free diagnostics. */
      if (index == fields.size() - 1u &&
          !write(fields[index].first, batch, sizeof(double), alignof(double), field,
                 static_cast<std::int64_t>(index))) {
        return false;
      }
    }
    return true;
  };
  const auto validate_free_diagnostics = [&](const Gfn2SccFreeEnergyDeviceDiagnostics& d,
                                             BindingField field, bool staged) {
    const std::array<std::pair<double*, std::int64_t>, 11> fields{{
        {d.core, d.core_elements},
        {d.es2, d.es2_elements},
        {d.es3, d.es3_elements},
        {d.aes2, d.aes2_elements},
        {d.spin, d.spin_elements},
        {d.d4_two_body, d.d4_two_body_elements},
        {d.explicit_point_charge, d.explicit_point_charge_elements},
        {d.periodic_embedding, d.periodic_embedding_elements},
        {d.entropy, d.entropy_elements},
        {d.internal_energy, d.internal_energy_elements},
        {d.free_energy, d.free_energy_elements},
    }};
    for (std::size_t index = 0; index < fields.size(); ++index) {
      if (!exact(fields[index].second, batch, field, static_cast<std::int64_t>(index))) {
        return false;
      }
      /* #99 deliberately reuses the staged occupation entropy as the
       * complete free-energy entropy diagnostic. The free-energy primitive
       * reads it into scratch before publishing the identical value. */
      if (staged && index == 8u) {
        if (fields[index].first != workspace.staged_occupations.entropies) {
          return validator.fail(BindingError::kInvalidZeroCopyView, BindingField::kStatePublication,
                                7);
        }
        continue;
      }
      if (!write(fields[index].first, batch, sizeof(double), alignof(double), field,
                 static_cast<std::int64_t>(index))) {
        return false;
      }
    }
    return true;
  };
  if (!validate_classical_diagnostics(state.classical_energy, BindingField::kStatePublication) ||
      !validate_free_diagnostics(state.free_energy, BindingField::kStatePublication, false) ||
      !validate_classical_diagnostics(workspace.staged_classical_energy,
                                      BindingField::kClassicalEnergy) ||
      !validate_free_diagnostics(workspace.staged_free_energy, BindingField::kFreeEnergy, true)) {
    return false;
  }

  return true;
}

bool validate_workspace_buffers(const Gfn2SccIterationDevicePlan& plan,
                                const Gfn2SccIterationDeviceWorkspace& workspace,
                                std::int64_t dipoles, std::int64_t two_batch,
                                std::int64_t two_orbitals, std::int64_t mixer_vector,
                                Validator& validator) noexcept {
  const std::int64_t batch = plan.topology.batch_size;
  const std::int64_t atoms = plan.topology.total_atoms;
  const std::int64_t shells = plan.topology.total_shells;
  const std::int64_t spin_channels = plan.wavefunction_layout.total_spin_channels;
  const std::int64_t spin_shells = plan.wavefunction_layout.total_spin_shells;
  const std::int64_t spin_atoms = plan.wavefunction_layout.total_spin_atoms;
  const std::int64_t spin_orbitals = plan.wavefunction_layout.total_spin_orbitals;
  const std::int64_t spin_matrices = plan.wavefunction_layout.total_spin_matrix_elements;
  std::int64_t spin_dipoles = 0;
  std::int64_t spin_quadrupoles = 0;
  if (!checked_multiply(spin_atoms, 3, spin_dipoles) ||
      !checked_multiply(spin_atoms, 6, spin_quadrupoles)) {
    return validator.fail(BindingError::kInvalidCount, BindingField::kSpin);
  }
  std::uint32_t group = 2000u;
  const auto exact = [&](std::int64_t actual, std::int64_t expected, BindingField field,
                         std::int64_t index = -1) {
    return validator.exact_count(actual, expected, field, index);
  };
  const auto scratch = [&](void* pointer, std::int64_t capacity, std::int64_t required,
                           std::size_t size, std::size_t alignment, BindingField field,
                           std::int64_t index) {
    return validator.capacity(capacity, required, field, index) &&
           validator.pointer(pointer, required, size, alignment, field, index, true, group++);
  };
  const auto read = [&](const void* pointer, std::int64_t count, std::size_t size,
                        std::size_t alignment, BindingField field, std::int64_t index) {
    return validator.pointer(pointer, count, size, alignment, field, index, false);
  };

  /* Immutable common topology is read-only and may be projected by many stages. */
  const auto& topology = plan.topology;
  if (!read(topology.atom_offsets, topology.atom_offset_count, sizeof(std::int64_t),
            alignof(std::int64_t), BindingField::kTopology, 0) ||
      !read(topology.batch_shell_offsets, topology.batch_shell_offset_count, sizeof(std::int64_t),
            alignof(std::int64_t), BindingField::kTopology, 1) ||
      !read(topology.batch_orbital_offsets, topology.batch_orbital_offset_count,
            sizeof(std::int64_t), alignof(std::int64_t), BindingField::kTopology, 2) ||
      !read(topology.matrix_offsets, topology.matrix_offset_count, sizeof(std::int64_t),
            alignof(std::int64_t), BindingField::kTopology, 3) ||
      !read(topology.atom_shell_offsets, topology.atom_shell_offset_count, sizeof(std::int64_t),
            alignof(std::int64_t), BindingField::kTopology, 4) ||
      !read(topology.shell_orbital_offsets, topology.shell_orbital_offset_count,
            sizeof(std::int64_t), alignof(std::int64_t), BindingField::kTopology, 5) ||
      !read(topology.shell_to_atom, topology.shell_to_atom_count, sizeof(std::int64_t),
            alignof(std::int64_t), BindingField::kTopology, 6) ||
      !read(topology.orbital_to_shell, topology.orbital_to_shell_count, sizeof(std::int64_t),
            alignof(std::int64_t), BindingField::kTopology, 7) ||
      !read(topology.orbital_to_atom, topology.orbital_to_atom_count, sizeof(std::int64_t),
            alignof(std::int64_t), BindingField::kTopology, 8)) {
    return false;
  }

  std::int64_t geometry_pair_elements = 0;
  if (!checked_multiply(plan.geometry_batch.total_pairs, kGfn2GeometryPairDataElements,
                        geometry_pair_elements)) {
    return validator.fail(BindingError::kInvalidCount, BindingField::kGeometry);
  }
  if (!exact(plan.geometry_cache.pair_data_elements, geometry_pair_elements,
             BindingField::kGeometry) ||
      !exact(plan.geometry_cache.coordination_elements, atoms, BindingField::kGeometry) ||
      !exact(plan.geometry_cache.generation_elements, batch, BindingField::kGeometry) ||
      !read(plan.geometry_cache.pair_data, geometry_pair_elements, sizeof(double), alignof(double),
            BindingField::kGeometry, 0) ||
      !read(plan.geometry_cache.coordination_numbers, atoms, sizeof(double), alignof(double),
            BindingField::kGeometry, 1) ||
      !read(plan.geometry_cache.geometry_generations, batch, sizeof(std::uint64_t),
            alignof(std::uint64_t), BindingField::kGeometry, 2) ||
      !scratch(workspace.geometry_workspace.pair_scratch,
               workspace.geometry_workspace.pair_elements, geometry_pair_elements, sizeof(double),
               alignof(double), BindingField::kGeometry, 3) ||
      !scratch(workspace.geometry_workspace.coordination_scratch,
               workspace.geometry_workspace.coordination_elements, atoms, sizeof(double),
               alignof(double), BindingField::kGeometry, 4) ||
      !scratch(workspace.geometry_workspace.gradient_scratch,
               workspace.geometry_workspace.gradient_elements, dipoles, sizeof(double),
               alignof(double), BindingField::kGeometry, 5) ||
      !scratch(workspace.geometry_workspace.sequence_active,
               workspace.geometry_workspace.sequence_elements, 1, sizeof(std::uint32_t),
               alignof(std::uint32_t), BindingField::kGeometry, 6)) {
    return false;
  }
  if (!exact(plan.es2_cache.matrix_elements, plan.es2_batch.total_matrix_elements,
             BindingField::kES2) ||
      !read(plan.es2_cache.coulomb_matrix, plan.es2_cache.matrix_elements, sizeof(double),
            alignof(double), BindingField::kES2, 0) ||
      !scratch(workspace.es2_workspace.matrix_scratch, workspace.es2_workspace.matrix_elements,
               plan.es2_batch.total_matrix_elements, sizeof(double), alignof(double),
               BindingField::kES2, 1) ||
      !scratch(workspace.es2_workspace.shell_scratch, workspace.es2_workspace.shell_elements,
               shells, sizeof(double), alignof(double), BindingField::kES2, 2) ||
      !scratch(workspace.es2_workspace.batch_scratch, workspace.es2_workspace.batch_elements, batch,
               sizeof(double), alignof(double), BindingField::kES2, 3) ||
      !scratch(workspace.es2_workspace.gradient_scratch, workspace.es2_workspace.gradient_elements,
               dipoles, sizeof(double), alignof(double), BindingField::kES2, 4)) {
    return false;
  }

  std::int64_t aes2_pair_elements = 0;
  std::int64_t aes2_potential_elements = 0;
  if (!checked_multiply(plan.aes2_batch.total_pairs, kGfn2AES2PairDataElements,
                        aes2_pair_elements) ||
      !checked_multiply(atoms, kGfn2AES2PotentialElementsPerAtom, aes2_potential_elements) ||
      !exact(plan.aes2_cache.pair_data_elements, aes2_pair_elements, BindingField::kAES2) ||
      !read(plan.aes2_cache.pair_data, aes2_pair_elements, sizeof(double), alignof(double),
            BindingField::kAES2, 0) ||
      !scratch(workspace.aes2_workspace.pair_scratch, workspace.aes2_workspace.pair_elements,
               aes2_pair_elements, sizeof(double), alignof(double), BindingField::kAES2, 1) ||
      !scratch(workspace.aes2_workspace.potential_scratch,
               workspace.aes2_workspace.potential_elements, aes2_potential_elements, sizeof(double),
               alignof(double), BindingField::kAES2, 2) ||
      !scratch(workspace.aes2_workspace.batch_scratch, workspace.aes2_workspace.batch_elements,
               batch, sizeof(double), alignof(double), BindingField::kAES2, 3) ||
      !scratch(workspace.aes2_workspace.gradient_scratch,
               workspace.aes2_workspace.gradient_elements, dipoles, sizeof(double), alignof(double),
               BindingField::kAES2, 4) ||
      !scratch(workspace.aes2_workspace.coordination_scratch,
               workspace.aes2_workspace.coordination_elements, atoms, sizeof(double),
               alignof(double), BindingField::kAES2, 5) ||
      !scratch(workspace.aes2_workspace.scc_peer_error_scratch,
               workspace.aes2_workspace.scc_peer_error_elements, 1, sizeof(std::uint32_t),
               alignof(std::uint32_t), BindingField::kAES2, 6)) {
    return false;
  }

  if (component_enabled(plan, Gfn2SccPotentialComponent::kD4TwoBody)) {
    std::int64_t d4_pairs = 0;
    std::int64_t weights = 0;
    if (!checked_multiply(plan.d4_batch.total_pairs, kGfn2D4PairDataElements, d4_pairs) ||
        !checked_multiply(atoms, kGfn2D4MaximumReferences, weights) ||
        !exact(plan.d4_cache.pair_data_elements, d4_pairs, BindingField::kD4) ||
        !exact(plan.d4_cache.coordination_elements, atoms, BindingField::kD4) ||
        !read(plan.d4_cache.pair_data, d4_pairs, sizeof(double), alignof(double), BindingField::kD4,
              0) ||
        !read(plan.d4_cache.coordination_numbers, atoms, sizeof(double), alignof(double),
              BindingField::kD4, 1) ||
        !scratch(workspace.d4_workspace.weights, workspace.d4_workspace.weight_elements, weights,
                 sizeof(double), alignof(double), BindingField::kD4, 2) ||
        !scratch(workspace.d4_workspace.weight_charge_derivatives,
                 workspace.d4_workspace.weight_elements, weights, sizeof(double), alignof(double),
                 BindingField::kD4, 3) ||
        !scratch(workspace.d4_workspace.atom_scratch, workspace.d4_workspace.atom_elements, atoms,
                 sizeof(double), alignof(double), BindingField::kD4, 4) ||
        !scratch(workspace.d4_workspace.batch_scratch, workspace.d4_workspace.batch_elements, batch,
                 sizeof(double), alignof(double), BindingField::kD4, 5) ||
        !scratch(workspace.d4_workspace.system_errors, workspace.d4_workspace.system_error_elements,
                 batch, sizeof(std::uint32_t), alignof(std::uint32_t), BindingField::kD4, 6)) {
      return false;
    }
  } else {
    const auto& value = workspace.d4_workspace;
    if (value.weights != nullptr || value.weight_cn_derivatives != nullptr ||
        value.weight_charge_derivatives != nullptr || value.weight_elements != 0 ||
        value.atom_scratch != nullptr || value.coordination_adjoints != nullptr ||
        value.atom_elements != 0 || value.batch_scratch != nullptr || value.batch_elements != 0 ||
        value.gradient_scratch != nullptr || value.gradient_elements != 0 ||
        value.system_errors != nullptr || value.system_error_elements != 0) {
      return validator.fail(BindingError::kInvalidCount, BindingField::kD4);
    }
  }

  if (component_enabled(plan, Gfn2SccPotentialComponent::kExplicitPointCharge)) {
    if (!exact(plan.explicit_point_charge_cache.shell_elements, shells,
               BindingField::kExplicitPointCharge) ||
        !read(plan.explicit_point_charge_cache.shell_potentials, shells, sizeof(double),
              alignof(double), BindingField::kExplicitPointCharge, 0)) {
      return false;
    }
  }

  if (component_enabled(plan, Gfn2SccPotentialComponent::kPeriodicEmbedding)) {
    if (!scratch(workspace.periodic_workspace.potential_scratch,
                 workspace.periodic_workspace.atom_elements, atoms, sizeof(double), alignof(double),
                 BindingField::kPeriodicEmbedding, 0) ||
        !scratch(workspace.periodic_workspace.raw_response_scratch,
                 workspace.periodic_workspace.atom_elements, atoms, sizeof(double), alignof(double),
                 BindingField::kPeriodicEmbedding, 1) ||
        !scratch(workspace.periodic_workspace.sequence_active,
                 workspace.periodic_workspace.sequence_elements, 1, sizeof(std::uint32_t),
                 alignof(std::uint32_t), BindingField::kPeriodicEmbedding, 2)) {
      return false;
    }
  } else {
    const auto& value = workspace.periodic_workspace;
    if (value.potential_scratch != nullptr || value.raw_response_scratch != nullptr ||
        value.sequence_active != nullptr || value.atom_elements != 0 ||
        value.sequence_elements != 0 || value.plan_token != 0u) {
      return validator.fail(BindingError::kInvalidCount, BindingField::kPeriodicEmbedding);
    }
  }

  if (!scratch(workspace.potential_workspace.shell_scratch,
               workspace.potential_workspace.shell_elements, spin_shells, sizeof(double),
               alignof(double), BindingField::kPotential, 20) ||
      !scratch(workspace.potential_workspace.atom_scratch,
               workspace.potential_workspace.atom_elements, spin_atoms, sizeof(double),
               alignof(double), BindingField::kPotential, 21) ||
      !scratch(workspace.potential_workspace.dipole_scratch,
               workspace.potential_workspace.dipole_elements, spin_dipoles, sizeof(double),
               alignof(double), BindingField::kPotential, 22) ||
      !scratch(workspace.potential_workspace.quadrupole_scratch,
               workspace.potential_workspace.quadrupole_elements, spin_quadrupoles, sizeof(double),
               alignof(double), BindingField::kPotential, 23) ||
      !scratch(workspace.potential_workspace.sequence_active,
               workspace.potential_workspace.sequence_elements, 1, sizeof(std::uint32_t),
               alignof(std::uint32_t), BindingField::kPotential, 24) ||
      !scratch(workspace.scalar_bridge.workspace.shell_scratch,
               workspace.scalar_bridge.workspace.shell_elements, shells, sizeof(double),
               alignof(double), BindingField::kScalarBridge, 20) ||
      !scratch(workspace.scalar_bridge.workspace.sequence_active,
               workspace.scalar_bridge.workspace.sequence_elements, 1, sizeof(std::uint32_t),
               alignof(std::uint32_t), BindingField::kScalarBridge, 21) ||
      !scratch(workspace.hamiltonian_workspace.matrix_scratch,
               workspace.hamiltonian_workspace.matrix_elements, spin_matrices, sizeof(double),
               alignof(double), BindingField::kHamiltonian, 20) ||
      !scratch(workspace.hamiltonian_workspace.sequence_active,
               workspace.hamiltonian_workspace.sequence_elements, 1, sizeof(std::uint32_t),
               alignof(std::uint32_t), BindingField::kHamiltonian, 21)) {
    return false;
  }

  const auto& eig = workspace.eigensolver_workspace;
  if (!scratch(eig.matrix_scratch_a, eig.matrix_a_elements, spin_matrices, sizeof(double),
               alignof(double), BindingField::kEigensolver, 20) ||
      !scratch(eig.matrix_scratch_b, eig.matrix_b_elements, spin_matrices, sizeof(double),
               alignof(double), BindingField::kEigensolver, 21) ||
      !scratch(eig.eigenvalue_scratch, eig.eigenvalue_elements, spin_orbitals, sizeof(double),
               alignof(double), BindingField::kEigensolver, 22) ||
      !scratch(eig.factor_pointers, eig.factor_pointer_elements, spin_channels, sizeof(double*),
               alignof(double*), BindingField::kEigensolver, 23) ||
      !scratch(eig.matrix_pointers, eig.matrix_pointer_elements, spin_channels, sizeof(double*),
               alignof(double*), BindingField::kEigensolver, 24) ||
      !scratch(eig.info_a, eig.info_a_elements, spin_channels, sizeof(int), alignof(int),
               BindingField::kEigensolver, 25) ||
      !scratch(eig.info_b, eig.info_b_elements, spin_channels, sizeof(int), alignof(int),
               BindingField::kEigensolver, 26) ||
      !scratch(eig.eligible, eig.eligible_elements, spin_channels, sizeof(std::uint8_t),
               alignof(std::uint8_t), BindingField::kEigensolver, 27) ||
      !scratch(eig.sequence_active, eig.sequence_active_elements, 1, sizeof(std::uint32_t),
               alignof(std::uint32_t), BindingField::kEigensolver, 28) ||
      !scratch(eig.compact_systems, eig.compact_system_elements, spin_channels,
               sizeof(std::int32_t), alignof(std::int32_t), BindingField::kEigensolver, 30) ||
      !scratch(eig.compact_source_slots, eig.compact_source_slot_elements, spin_channels,
               sizeof(std::int32_t), alignof(std::int32_t), BindingField::kEigensolver, 31) ||
      !scratch(eig.bucket_activity, eig.bucket_activity_elements,
               plan.eigensolver_provider.bucket_count, sizeof(Gfn2EigensolverBucketActivity),
               alignof(Gfn2EigensolverBucketActivity), BindingField::kEigensolver, 32)) {
    return false;
  }
  if (plan.eigensolver_provider.device_workspace_bytes != 0u &&
      !validator.pointer(
          plan.eigensolver_provider.device_workspace,
          static_cast<std::int64_t>(plan.eigensolver_provider.device_workspace_bytes), 1u,
          alignof(double), BindingField::kEigensolver, 29, true, group++)) {
    return false;
  }

  const auto& occ = workspace.occupations_workspace;
  if (!scratch(occ.occupation_scratch, occ.occupation_elements, two_orbitals, sizeof(double),
               alignof(double), BindingField::kOccupations, 20) ||
      !scratch(occ.chemical_potential_scratch, occ.chemical_potential_elements, two_batch,
               sizeof(double), alignof(double), BindingField::kOccupations, 21) ||
      !scratch(occ.electron_sum_scratch, occ.electron_sum_elements, two_batch, sizeof(double),
               alignof(double), BindingField::kOccupations, 22) ||
      !scratch(occ.entropy_scratch, occ.entropy_elements, batch, sizeof(double), alignof(double),
               BindingField::kOccupations, 23) ||
      !scratch(occ.sequence_active, occ.sequence_active_elements, 1, sizeof(std::uint32_t),
               alignof(std::uint32_t), BindingField::kOccupations, 24)) {
    return false;
  }

  const auto& density = workspace.density_workspace;
  if (!scratch(density.density_scratch, density.density_elements, spin_matrices, sizeof(double),
               alignof(double), BindingField::kDensity, 20) ||
      !scratch(density.weighted_density_scratch, density.weighted_density_elements, spin_matrices,
               sizeof(double), alignof(double), BindingField::kDensity, 21) ||
      !scratch(density.weights, density.weight_elements, spin_orbitals, sizeof(double),
               alignof(double), BindingField::kDensity, 22) ||
      !scratch(density.energy_weights, density.energy_weight_elements, spin_orbitals,
               sizeof(double), alignof(double), BindingField::kDensity, 23) ||
      !scratch(density.band_energy_scratch, density.band_energy_elements, batch, sizeof(double),
               alignof(double), BindingField::kDensity, 24) ||
      !scratch(density.occupation_sum_scratch, density.occupation_sum_elements, batch,
               sizeof(double), alignof(double), BindingField::kDensity, 25) ||
      !scratch(density.density_trace_scratch, density.density_trace_elements, batch, sizeof(double),
               alignof(double), BindingField::kDensity, 26) ||
      !scratch(density.weighted_density_trace_scratch, density.weighted_density_trace_elements,
               batch, sizeof(double), alignof(double), BindingField::kDensity, 27) ||
      !scratch(density.sequence_active, density.sequence_active_elements, 1, sizeof(std::uint32_t),
               alignof(std::uint32_t), BindingField::kDensity, 28) ||
      !scratch(density.channel_band_energy_scratch, density.channel_band_energy_elements,
               spin_channels, sizeof(double), alignof(double), BindingField::kDensity, 29) ||
      !scratch(density.channel_occupation_sum_scratch, density.channel_occupation_sum_elements,
               spin_channels, sizeof(double), alignof(double), BindingField::kDensity, 30) ||
      !scratch(density.channel_density_trace_scratch, density.channel_density_trace_elements,
               spin_channels, sizeof(double), alignof(double), BindingField::kDensity, 31) ||
      !scratch(density.channel_weighted_density_trace_scratch,
               density.channel_weighted_density_trace_elements, spin_channels, sizeof(double),
               alignof(double), BindingField::kDensity, 32)) {
    return false;
  }

  const auto& mulliken = workspace.mulliken_workspace;
  if (!scratch(mulliken.qsh_scratch, mulliken.qsh_elements, spin_shells, sizeof(double),
               alignof(double), BindingField::kMulliken, 20) ||
      !scratch(mulliken.qat_scratch, mulliken.qat_elements, spin_atoms, sizeof(double),
               alignof(double), BindingField::kMulliken, 21) ||
      !scratch(mulliken.dipole_scratch, mulliken.dipole_elements, spin_dipoles, sizeof(double),
               alignof(double), BindingField::kMulliken, 22) ||
      !scratch(mulliken.quadrupole_scratch, mulliken.quadrupole_elements, spin_quadrupoles,
               sizeof(double), alignof(double), BindingField::kMulliken, 23) ||
      !scratch(mulliken.sequence_active, mulliken.sequence_elements, 1, sizeof(std::uint32_t),
               alignof(std::uint32_t), BindingField::kMulliken, 24)) {
    return false;
  }

  if (!scratch(workspace.spin_workspace.energy_scratch, workspace.spin_workspace.energy_elements,
               batch, sizeof(double), alignof(double), BindingField::kSpin, 20) ||
      !scratch(workspace.spin_workspace.potential_scratch,
               workspace.spin_workspace.potential_elements, spin_shells, sizeof(double),
               alignof(double), BindingField::kSpin, 21) ||
      !scratch(workspace.spin_workspace.sequence_active, workspace.spin_workspace.sequence_elements,
               1, sizeof(std::uint32_t), alignof(std::uint32_t), BindingField::kSpin, 22)) {
    return false;
  }

  const auto& electronic = workspace.electronic_energy_workspace;
  std::int64_t classical_scratch = 0;
  std::int64_t free_scratch = 0;
  if (!checked_multiply(batch, kGfn2SccClassicalDiagnosticComponents, classical_scratch) ||
      !checked_multiply(batch, kGfn2SccFreeEnergyDiagnosticComponents, free_scratch) ||
      !scratch(electronic.core_energy_scratch, electronic.batch_elements, batch, sizeof(double),
               alignof(double), BindingField::kElectronicEnergy, 20) ||
      !scratch(electronic.electronic_free_energy_scratch, electronic.batch_elements, batch,
               sizeof(double), alignof(double), BindingField::kElectronicEnergy, 21) ||
      !scratch(electronic.sequence_active, electronic.sequence_elements, 1, sizeof(std::uint32_t),
               alignof(std::uint32_t), BindingField::kElectronicEnergy, 22) ||
      !scratch(workspace.classical_energy_workspace.component_scratch,
               workspace.classical_energy_workspace.component_elements, classical_scratch,
               sizeof(double), alignof(double), BindingField::kClassicalEnergy, 20) ||
      !scratch(workspace.classical_energy_workspace.sequence_active,
               workspace.classical_energy_workspace.sequence_elements, 1, sizeof(std::uint32_t),
               alignof(std::uint32_t), BindingField::kClassicalEnergy, 21) ||
      !scratch(workspace.free_energy_workspace.diagnostic_scratch,
               workspace.free_energy_workspace.diagnostic_elements, free_scratch, sizeof(double),
               alignof(double), BindingField::kFreeEnergy, 20) ||
      !scratch(workspace.free_energy_workspace.sequence_active,
               workspace.free_energy_workspace.sequence_elements, 1, sizeof(std::uint32_t),
               alignof(std::uint32_t), BindingField::kFreeEnergy, 21)) {
    return false;
  }

  std::int64_t history = 0;
  std::int64_t beta = 0;
  std::int64_t coefficients = 0;
  std::int64_t history_square = 0;
  if (!checked_multiply(mixer_vector, plan.mixer_policy.history_size, history) ||
      !checked_multiply(plan.mixer_policy.history_size, plan.mixer_policy.history_size,
                        history_square) ||
      !checked_multiply(batch, history_square, beta) ||
      !checked_multiply(batch, plan.mixer_policy.history_size, coefficients)) {
    return validator.fail(BindingError::kInvalidCount, BindingField::kMixer);
  }
  const auto& mixer = workspace.mixer_workspace;
  if (!scratch(mixer.residual, mixer.vector_elements, mixer_vector, sizeof(double), alignof(double),
               BindingField::kMixer, 20) ||
      !scratch(mixer.mixed, mixer.vector_elements, mixer_vector, sizeof(double), alignof(double),
               BindingField::kMixer, 21) ||
      !scratch(mixer.delta_f, mixer.vector_elements, mixer_vector, sizeof(double), alignof(double),
               BindingField::kMixer, 22) ||
      !scratch(mixer.new_u, mixer.vector_elements, mixer_vector, sizeof(double), alignof(double),
               BindingField::kMixer, 23) ||
      !scratch(mixer.beta, mixer.beta_elements, beta, sizeof(double), alignof(double),
               BindingField::kMixer, 24) ||
      !scratch(mixer.coefficients, mixer.coefficient_elements, coefficients, sizeof(double),
               alignof(double), BindingField::kMixer, 25) ||
      !scratch(mixer.sequence_active, mixer.sequence_elements, 1, sizeof(std::uint32_t),
               alignof(std::uint32_t), BindingField::kMixer, 26) ||
      !scratch(workspace.mixer_device_error, workspace.mixer_device_error_elements, 1,
               sizeof(std::uint32_t), alignof(std::uint32_t), BindingField::kMixer, 27) ||
      !scratch(workspace.publication_workspace.system_errors,
               workspace.publication_workspace.system_error_elements, batch, sizeof(std::uint32_t),
               alignof(std::uint32_t), BindingField::kStatePublication, 21) ||
      !scratch(workspace.publication_workspace.device_error,
               workspace.publication_workspace.device_error_elements, 1, sizeof(std::uint32_t),
               alignof(std::uint32_t), BindingField::kStatePublication, 22) ||
      !scratch(workspace.publication_workspace.sequence_active,
               workspace.publication_workspace.sequence_elements, 1, sizeof(std::uint32_t),
               alignof(std::uint32_t), BindingField::kStatePublication, 23)) {
    return false;
  }
  const auto& publication = workspace.publication_workspace;
  if (!validator.exact_count(publication.mixed_atomic_charge_elements, spin_atoms,
                             BindingField::kStatePublication, 24) ||
      !validator.exact_count(publication.batch_elements, batch, BindingField::kStatePublication,
                             25) ||
      !scratch(publication.previous_free_energies, publication.batch_elements, batch,
               sizeof(double), alignof(double), BindingField::kStatePublication, 26) ||
      !scratch(publication.free_energy_changes, publication.batch_elements, batch, sizeof(double),
               alignof(double), BindingField::kStatePublication, 27) ||
      !scratch(publication.next_iterations, publication.batch_elements, batch,
               sizeof(std::uint64_t), alignof(std::uint64_t), BindingField::kStatePublication,
               28) ||
      !scratch(publication.next_converged, publication.batch_elements, batch, sizeof(std::uint8_t),
               alignof(std::uint8_t), BindingField::kStatePublication, 29) ||
      !scratch(publication.next_statuses, publication.batch_elements, batch,
               sizeof(xtbloom_status_t), alignof(xtbloom_status_t), BindingField::kStatePublication,
               30)) {
    return false;
  }
  (void)history;
  return true;
}

const Gfn2SccStageDeviceReport* find_stage_report(const Gfn2SccIterationDevicePlan& plan,
                                                  Gfn2SccStageId stage) noexcept {
  for (std::int64_t index = 0; index < plan.report_count; ++index) {
    if (plan.reports[index].stage == stage) {
      return &plan.reports[index];
    }
  }
  return nullptr;
}

std::uint32_t* mutable_system_codes(const Gfn2SccStageDeviceReport& report) noexcept {
  return const_cast<std::uint32_t*>(static_cast<const std::uint32_t*>(report.system_codes));
}

std::uint32_t* mutable_device_error(const Gfn2SccStageDeviceReport& report) noexcept {
  return const_cast<std::uint32_t*>(report.device_error);
}

Gfn2SccIterationLaunchResult invalid_launch_binding(
    Gfn2SccStageId stage = Gfn2SccStageId::kNone) noexcept {
  Gfn2SccIterationLaunchResult result{};
  result.status = Gfn2SccIterationLaunchStatus::kInvalidBinding;
  result.stage = stage;
  result.binding = {Gfn2SccIterationBindingError::kInvalidStageReport,
                    Gfn2SccIterationBindingField::kStageReports, -1};
  return result;
}

Gfn2SccIterationLaunchResult cuda_launch_failure(Gfn2SccStageId stage,
                                                 cudaError_t status) noexcept {
  Gfn2SccIterationLaunchResult result{};
  result.status = Gfn2SccIterationLaunchStatus::kCudaError;
  result.stage = stage;
  result.cuda_status = status;
  return result;
}

Gfn2SccIterationLaunchResult provider_launch_failure(
    Gfn2SccStageId stage, const Gfn2EigensolverLaunchResult& provider) noexcept {
  Gfn2SccIterationLaunchResult result{};
  result.stage = stage;
  result.cuda_status = provider.cuda_status;
  result.cublas_status = provider.cublas_status;
  result.cusolver_status = provider.cusolver_status;
  switch (provider.status) {
    case Gfn2EigensolverLaunchStatus::kCublasError:
      result.status = Gfn2SccIterationLaunchStatus::kCublasError;
      break;
    case Gfn2EigensolverLaunchStatus::kCusolverError:
      result.status = Gfn2SccIterationLaunchStatus::kCusolverError;
      break;
    case Gfn2EigensolverLaunchStatus::kCudaError:
      result.status = Gfn2SccIterationLaunchStatus::kCudaError;
      break;
    case Gfn2EigensolverLaunchStatus::kInvalidArgument:
      result.status = Gfn2SccIterationLaunchStatus::kInvalidBinding;
      result.binding = {Gfn2SccIterationBindingError::kInvalidProvider,
                        Gfn2SccIterationBindingField::kEigensolver, -1};
      break;
    case Gfn2EigensolverLaunchStatus::kSuccess:
      break;
  }
  return result;
}

__global__ void open_stage_sequence_kernel(std::uint32_t* sequence_active) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *sequence_active = 1u;
  }
}

cudaError_t open_stage_sequence(const Gfn2SccStageDeviceReport& report,
                                cudaStream_t stream) noexcept {
  open_stage_sequence_kernel<<<1, 1, 0, stream>>>(
      const_cast<std::uint32_t*>(report.stage_sequence_active));
  return cudaPeekAtLastError();
}

template <typename T>
__device__ void copy_device_range(const T* source, T* destination, std::int64_t begin,
                                  std::int64_t end) {
  for (std::int64_t index = begin + threadIdx.x; index < end; index += blockDim.x) {
    destination[index] = source[index];
  }
}

/*
 * The CPU driver clones mixer history before every transition. Mirroring that
 * transaction is essential after a failed publication or replay: staged
 * scratch may be dirty while the public history is still authoritative.
 */
__global__ void stage_active_mixer_kernel(Gfn2WavefunctionLayoutView wavefunction_layout,
                                          std::int64_t history_size,
                                          Gfn2SccIterationDeviceActivity activity,
                                          Gfn2SccMixerDeviceState source,
                                          Gfn2SccMixerDeviceState destination) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (*activity.sequence_active != 1u || activity.active_mask[system] != 1u) {
    return;
  }
  /* Mixer vectors contain every charge/magnetization qsh, dipole, and
   * quadrupole channel. Physical topology offsets are therefore insufficient
   * for a mixed-spin batch: use the canonical nspin-expanded partitions.
   * Restricted layouts carry identical physical and spin-aware offsets, so
   * this preserves the historical byte ranges exactly. */
  const std::int64_t vector_begin = wavefunction_layout.spin_shell_offsets[system] +
                                    9 * wavefunction_layout.spin_atom_offsets[system];
  const std::int64_t vector_end = wavefunction_layout.spin_shell_offsets[system + 1] +
                                  9 * wavefunction_layout.spin_atom_offsets[system + 1];
  const std::int64_t history_begin = vector_begin * history_size;
  const std::int64_t history_end = vector_end * history_size;
  const std::int64_t omega_begin = system * history_size;
  const std::int64_t omega_end = omega_begin + history_size;

  copy_device_range(source.current_inputs, destination.current_inputs, vector_begin, vector_end);
  copy_device_range(source.previous_inputs, destination.previous_inputs, vector_begin, vector_end);
  copy_device_range(source.previous_residuals, destination.previous_residuals, vector_begin,
                    vector_end);
  copy_device_range(source.df_history, destination.df_history, history_begin, history_end);
  copy_device_range(source.u_history, destination.u_history, history_begin, history_end);
  copy_device_range(source.omega, destination.omega, omega_begin, omega_end);
  __syncthreads();
  if (threadIdx.x == 0) {
    destination.residual_rms[system] = source.residual_rms[system];
    destination.residual_maximum[system] = source.residual_maximum[system];
    destination.iterations[system] = source.iterations[system];
    destination.restart_counts[system] = source.restart_counts[system];
    destination.system_statuses[system] = source.system_statuses[system];
    destination.initialized[system] = source.initialized[system];
    destination.residual_converged[system] = source.residual_converged[system];
  }
}

cudaError_t stage_active_mixer(const Gfn2SccIterationBinding& binding,
                               cudaStream_t stream) noexcept {
  stage_active_mixer_kernel<<<static_cast<unsigned int>(binding.plan.topology.batch_size), 256, 0,
                              stream>>>(
      binding.plan.wavefunction_layout, binding.plan.mixer_policy.history_size,
      binding.workspace.activity, binding.state.mixer, binding.workspace.staged_mixer);
  return cudaPeekAtLastError();
}

cudaError_t normalize_stage(const Gfn2SccStageDeviceReport& report,
                            const Gfn2SccIterationDeviceLedger& ledger,
                            cudaStream_t stream) noexcept {
  return normalize_gfn2_scc_stage_cuda(report, ledger, stream);
}

#if defined(XTBLOOM_CUDA_TEST_HOOKS)
__global__ void inject_test_stage_fault_kernel(Gfn2SccStageDeviceReport report, std::int64_t system,
                                               std::uint32_t raw_code) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  if (report.system_code_format == Gfn2SccStageCodeFormat::kXTBloomStatus) {
    auto* statuses =
        const_cast<xtbloom_status_t*>(static_cast<const xtbloom_status_t*>(report.system_codes));
    statuses[system] = static_cast<xtbloom_status_t>(raw_code);
  } else {
    auto* codes =
        const_cast<std::uint32_t*>(static_cast<const std::uint32_t*>(report.system_codes));
    codes[system] = raw_code;
  }
}

cudaError_t inject_test_stage_fault(const Gfn2SccStageDeviceReport& report,
                                    const Gfn2SccIterationTestFault& fault,
                                    cudaStream_t stream) noexcept {
  inject_test_stage_fault_kernel<<<1, 1, 0, stream>>>(report, fault.system, fault.raw_code);
  return cudaPeekAtLastError();
}
#endif

}  // namespace

Gfn2SccIterationBindingDiagnostic validate_gfn2_scc_iteration_binding_cuda(
    const Gfn2SccIterationDevicePlan& plan, const Gfn2SccIterationDeviceInput& input,
    const Gfn2SccIterationDeviceState& state,
    const Gfn2SccIterationDeviceWorkspace& workspace) noexcept {
  Validator validator;
  std::int64_t dipoles = 0;
  std::int64_t quadrupoles = 0;
  std::int64_t two_batch = 0;
  std::int64_t two_orbitals = 0;
  std::int64_t mixer_vector = 0;
  if (!validate_top_level_shape(plan, validator, dipoles, quadrupoles, two_batch, two_orbitals,
                                mixer_vector) ||
      !validate_plan_projection_identity(plan, validator) ||
      !validate_leaf_projection_identity(plan, validator) ||
      !validate_plan_tokens(plan, input, state, workspace, validator) ||
      !validate_plan_shapes(plan, validator, dipoles, quadrupoles, two_batch, mixer_vector) ||
      !validate_plan_pointer_shapes(plan, validator) ||
      !validate_optional_plan_canonicalization(plan, validator) ||
      !validate_provider(plan, workspace, validator) ||
      !validate_zero_copy_views(plan, input, state, workspace, validator) ||
      !validate_stage_reports(plan, workspace, validator) ||
      !validate_core_buffers(plan, input, state, workspace, dipoles, quadrupoles, two_batch,
                             two_orbitals, mixer_vector, validator) ||
      !validate_component_and_energy_buffers(plan, input, state, workspace, dipoles, quadrupoles,
                                             validator) ||
      !validate_workspace_buffers(plan, workspace, dipoles, two_batch, two_orbitals, mixer_vector,
                                  validator) ||
      !validator.aliases_valid()) {
    return validator.diagnostic();
  }
  return {};
}

Gfn2SccIterationBindingDiagnostic bind_gfn2_scc_iteration_cuda(
    const Gfn2SccIterationDevicePlan& plan, const Gfn2SccIterationDeviceInput& input,
    const Gfn2SccIterationDeviceState& state, const Gfn2SccIterationDeviceWorkspace& workspace,
    Gfn2SccIterationBinding& binding) noexcept {
  binding = {};
  const Gfn2SccIterationBindingDiagnostic diagnostic =
      validate_gfn2_scc_iteration_binding_cuda(plan, input, state, workspace);
  if (diagnostic.error == Gfn2SccIterationBindingError::kSuccess) {
    binding.plan = plan;
    binding.input = input;
    binding.state = state;
    binding.workspace = workspace;
  }
  return diagnostic;
}

template <bool EnableTestFault>
static Gfn2SccIterationLaunchResult launch_scc_iteration_impl(
    const Gfn2SccIterationBinding& binding, const Gfn2GeometryEpochConsumerDevice* geometry,
    cudaStream_t stream, bool derive_activity, bool launch_numerical_body,
    Gfn2SccIterationBodySegment segment = Gfn2SccIterationBodySegment::kFull,
    const Gfn2SccIterationTestFault* test_fault = nullptr) noexcept {
  const auto& plan = binding.plan;
  const auto& input = binding.input;
  const auto& state = binding.state;
  const auto& workspace = binding.workspace;
  const bool mixed_spin = plan.wavefunction_layout.total_spin_channels != plan.topology.batch_size;
  if (plan.abi_version != kGfn2SccIterationAbiVersion || plan.plan_token == 0u ||
      input.plan_token != plan.plan_token || state.plan_token != plan.plan_token ||
      workspace.plan_token != plan.plan_token) {
    return invalid_launch_binding();
  }
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
  if constexpr (EnableTestFault) {
    const Gfn2SccStageDeviceReport* fault_report =
        test_fault == nullptr ? nullptr : find_stage_report(plan, test_fault->stage);
    if (test_fault == nullptr || fault_report == nullptr || test_fault->system < 0 ||
        test_fault->system >= plan.topology.batch_size || test_fault->raw_code == 0u ||
        test_fault->raw_code >= 64u ||
        (fault_report->peer_error_mask & (std::uint64_t{1} << test_fault->raw_code)) == 0u ||
        fault_report->system_codes == nullptr ||
        fault_report->system_code_elements != plan.topology.batch_size) {
      return invalid_launch_binding(test_fault == nullptr ? Gfn2SccStageId::kNone
                                                          : test_fault->stage);
    }
  } else {
    (void)test_fault;
  }
#else
  static_assert(!EnableTestFault, "test-fault launchers require XTBLOOM_CUDA_TEST_HOOKS");
  (void)test_fault;
#endif
  if (geometry != nullptr &&
      (geometry->plan_token != plan.plan_token || geometry->epoch.plan_token != plan.plan_token ||
       geometry->epoch.value_elements != 1 ||
       geometry->batch_elements != plan.activity_policy.batch_size ||
       geometry->epoch.value == nullptr || geometry->committed_generations == nullptr ||
       geometry->eligible_mask == nullptr ||
       reinterpret_cast<std::uintptr_t>(geometry->epoch.value) % alignof(std::uint64_t) != 0u ||
       reinterpret_cast<std::uintptr_t>(geometry->committed_generations) % alignof(std::uint64_t) !=
           0u ||
       reinterpret_cast<std::uintptr_t>(geometry->eligible_mask) % alignof(std::uint8_t) != 0u)) {
    Gfn2SccIterationLaunchResult result = invalid_launch_binding(Gfn2SccStageId::kActivity);
    result.binding = {Gfn2SccIterationBindingError::kCrossPlan,
                      Gfn2SccIterationBindingField::kGeometry, -1};
    return result;
  }

  if (launch_numerical_body &&
      plan.eigensolver_provider.capture_mode ==
          Gfn2SccIterationProviderCaptureMode::kUncapturedSegmentRequired) {
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const cudaError_t capture_error = cudaStreamIsCapturing(stream, &capture_status);
    if (capture_error != cudaSuccess) {
      return cuda_launch_failure(Gfn2SccStageId::kEigensolver, capture_error);
    }
    if (capture_status != cudaStreamCaptureStatusNone) {
      Gfn2SccIterationLaunchResult result{};
      result.status = Gfn2SccIterationLaunchStatus::kProviderCaptureUnsupported;
      result.stage = Gfn2SccStageId::kEigensolver;
      return result;
    }
  }

  Gfn2SccIterationLaunchResult failure{};
  const auto check_cuda = [&](Gfn2SccStageId stage, cudaError_t status) {
    if (status == cudaSuccess) {
      return true;
    }
    failure = cuda_launch_failure(stage, status);
    return false;
  };
  const auto report = [&](Gfn2SccStageId stage) { return find_stage_report(plan, stage); };
  const auto begin_stage = [&](const Gfn2SccStageDeviceReport* stage_report) {
    if (stage_report == nullptr) {
      failure = invalid_launch_binding();
      return false;
    }
    return check_cuda(stage_report->stage, open_stage_sequence(*stage_report, stream));
  };
  const auto finish_stage = [&](const Gfn2SccStageDeviceReport& stage_report) {
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
    if constexpr (EnableTestFault) {
      if (stage_report.stage == test_fault->stage &&
          !check_cuda(stage_report.stage,
                      inject_test_stage_fault(stage_report, *test_fault, stream))) {
        return false;
      }
    }
#endif
    return check_cuda(stage_report.stage, normalize_stage(stage_report, workspace.ledger, stream));
  };

  if (derive_activity) {
    const cudaError_t activity_status =
        geometry == nullptr
            ? derive_gfn2_scc_iteration_activity_cuda(plan.activity_policy, input.activity_state,
                                                      plan.provenance, workspace.ledger, stream)
            : derive_gfn2_scc_iteration_activity_cuda(plan.activity_policy, input.activity_state,
                                                      plan.provenance, *geometry, workspace.ledger,
                                                      stream);
    if (!check_cuda(Gfn2SccStageId::kActivity, activity_status)) {
      return failure;
    }
  }
  if (!launch_numerical_body) {
    return {};
  }

  const Gfn2SccStageDeviceReport* stage_report = nullptr;
  if (segment != Gfn2SccIterationBodySegment::kPostEigensolver) {
    stage_report = report(Gfn2SccStageId::kMixedGather);
    if (!begin_stage(stage_report) ||
        !check_cuda(stage_report->stage,
                    reset_gfn2_scc_potential_device_errors_cuda(
                        plan.topology.batch_size, mutable_system_codes(*stage_report),
                        mutable_device_error(*stage_report), stream)) ||
        !check_cuda(stage_report->stage,
                    reduce_gfn2_scc_spin_atomic_charges_cuda(
                        plan.potential_batch, plan.wavefunction_layout, input.mixed_fields,
                        workspace.activity, workspace.mixed_topology, workspace.physical_topology,
                        workspace.potential_workspace, mutable_system_codes(*stage_report),
                        mutable_device_error(*stage_report), stream)) ||
        !finish_stage(*stage_report)) {
      return failure;
    }

    stage_report = report(Gfn2SccStageId::kSpinPotential);
    if (!begin_stage(stage_report) ||
        !check_cuda(stage_report->stage,
                    reset_gfn2_spin_device_errors_cuda(
                        plan.topology.batch_size, mutable_system_codes(*stage_report),
                        mutable_device_error(*stage_report), stream)) ||
        !check_cuda(stage_report->stage, evaluate_gfn2_spin_polarization_cuda(
                                             plan.spin_batch, input.mixed_spin, workspace.activity,
                                             workspace.spin_output, workspace.spin_workspace,
                                             mutable_system_codes(*stage_report),
                                             mutable_device_error(*stage_report), stream)) ||
        !finish_stage(*stage_report)) {
      return failure;
    }

    stage_report = report(Gfn2SccStageId::kES2Potential);
    if (!begin_stage(stage_report) ||
        !check_cuda(stage_report->stage,
                    reset_gfn2_es2_scc_errors_cuda(plan.topology.batch_size,
                                                   mutable_system_codes(*stage_report),
                                                   mutable_device_error(*stage_report), stream)) ||
        !check_cuda(
            stage_report->stage,
            evaluate_gfn2_es2_scc_potential_cuda(
                plan.es2_batch, plan.es2_cache, plan.geometry_generation, workspace.activity,
                workspace.physical_topology.shell_charges, workspace.components.es2_shell_potential,
                workspace.es2_workspace, mutable_system_codes(*stage_report),
                mutable_device_error(*stage_report), stream)) ||
        !finish_stage(*stage_report)) {
      return failure;
    }

    stage_report = report(Gfn2SccStageId::kES3Potential);
    if (!begin_stage(stage_report) ||
        !check_cuda(stage_report->stage,
                    reset_gfn2_es3_scc_errors_cuda(plan.topology.batch_size,
                                                   mutable_system_codes(*stage_report),
                                                   mutable_device_error(*stage_report), stream)) ||
        !check_cuda(
            stage_report->stage,
            evaluate_gfn2_es3_scc_potential_cuda(
                plan.es3_batch, workspace.activity, workspace.physical_topology.shell_charges,
                workspace.components.es3_shell_potential, mutable_system_codes(*stage_report),
                mutable_device_error(*stage_report), stream)) ||
        !finish_stage(*stage_report)) {
      return failure;
    }

    stage_report = report(Gfn2SccStageId::kAES2Potential);
    if (!begin_stage(stage_report) ||
        !check_cuda(stage_report->stage,
                    reset_gfn2_aes2_device_errors_cuda(
                        plan.topology.batch_size, mutable_system_codes(*stage_report),
                        mutable_device_error(*stage_report), stream)) ||
        !check_cuda(stage_report->stage,
                    evaluate_gfn2_aes2_scc_potential_cuda(
                        plan.aes2_batch, plan.aes2_cache, plan.geometry_generation,
                        workspace.activity, workspace.physical_topology.atomic_charges,
                        workspace.physical_topology.atomic_dipoles,
                        workspace.physical_topology.atomic_quadrupoles,
                        workspace.components.aes2_atomic_potential,
                        workspace.components.aes2_dipole_potential,
                        workspace.components.aes2_quadrupole_potential, workspace.aes2_workspace,
                        mutable_system_codes(*stage_report), mutable_device_error(*stage_report),
                        stream)) ||
        !finish_stage(*stage_report)) {
      return failure;
    }

    if (component_enabled(plan, Gfn2SccPotentialComponent::kD4TwoBody)) {
      stage_report = report(Gfn2SccStageId::kD4Potential);
      if (!begin_stage(stage_report) ||
          !check_cuda(stage_report->stage,
                      reset_gfn2_d4_device_errors_cuda(
                          plan.topology.batch_size, workspace.d4_workspace.system_errors,
                          mutable_device_error(*stage_report), stream)) ||
          !check_cuda(stage_report->stage,
                      evaluate_gfn2_d4_scc_potential_cuda(
                          plan.d4_batch, plan.d4_parameters, plan.d4_cache,
                          plan.geometry_generation, workspace.physical_topology.atomic_charges,
                          workspace.activity, workspace.components.d4_atomic_potential,
                          workspace.d4_workspace, mutable_device_error(*stage_report), stream)) ||
          !finish_stage(*stage_report)) {
        return failure;
      }
    }

    if (component_enabled(plan, Gfn2SccPotentialComponent::kPeriodicEmbedding)) {
      stage_report = report(Gfn2SccStageId::kPeriodicPotential);
      if (!begin_stage(stage_report) ||
          !check_cuda(stage_report->stage,
                      reset_gfn2_periodic_embedding_scc_device_errors_cuda(
                          plan.topology.batch_size, mutable_system_codes(*stage_report),
                          mutable_device_error(*stage_report), stream)) ||
          !check_cuda(stage_report->stage,
                      evaluate_gfn2_periodic_embedding_scc_potential_cuda(
                          plan.periodic_batch, plan.geometry_generation,
                          workspace.physical_topology.atomic_charges, workspace.activity,
                          workspace.components.periodic_atomic_potential,
                          workspace.periodic_workspace, mutable_system_codes(*stage_report),
                          mutable_device_error(*stage_report), stream)) ||
          !finish_stage(*stage_report)) {
        return failure;
      }
    }

    stage_report = report(Gfn2SccStageId::kPotentialCompose);
    if (!begin_stage(stage_report) ||
        !check_cuda(stage_report->stage,
                    reset_gfn2_scc_potential_device_errors_cuda(
                        plan.topology.batch_size, mutable_system_codes(*stage_report),
                        mutable_device_error(*stage_report), stream))) {
      return failure;
    }
    const cudaError_t compose_status =
        mixed_spin
            ? compose_gfn2_scc_spin_potentials_cuda(
                  plan.potential_batch, plan.wavefunction_layout, workspace.potential_components,
                  {workspace.spin_output.shell_potentials,
                   workspace.spin_output.shell_potential_elements, plan.plan_token},
                  workspace.potential_activity, workspace.complete_potentials,
                  workspace.potential_workspace, mutable_system_codes(*stage_report),
                  mutable_device_error(*stage_report), stream)
            : compose_gfn2_scc_potentials_cuda(
                  plan.potential_batch, workspace.potential_components, workspace.activity,
                  workspace.complete_potentials, workspace.potential_workspace,
                  mutable_system_codes(*stage_report), mutable_device_error(*stage_report), stream);
    if (!check_cuda(stage_report->stage, compose_status) || !finish_stage(*stage_report)) {
      return failure;
    }

    stage_report = report(Gfn2SccStageId::kScalarBridge);
    if (!begin_stage(stage_report) ||
        !check_cuda(stage_report->stage,
                    reset_gfn2_scc_bridge_device_errors_cuda(
                        plan.topology.batch_size, mutable_system_codes(*stage_report),
                        mutable_device_error(*stage_report),
                        workspace.scalar_bridge.workspace.sequence_active, stream))) {
      return failure;
    }
    if (!mixed_spin &&
        !check_cuda(
            stage_report->stage,
            collect_gfn2_scc_shell_scalar_potential_cuda(
                plan.scalar_bridge_batch, workspace.scalar_bridge.fields, workspace.activity,
                workspace.scalar_bridge.shell_scalar, workspace.scalar_bridge.shell_elements,
                workspace.scalar_bridge.workspace, mutable_system_codes(*stage_report),
                mutable_device_error(*stage_report), stream))) {
      return failure;
    }
    /* The mixed composer already applied the scalar bridge. Preserve the
     * canonical sequence latch for report normalization without collecting a
     * second time (which would double-add the atomic scalar potential). */
    if (mixed_spin &&
        !check_cuda(stage_report->stage,
                    cudaMemcpyAsync(workspace.scalar_bridge.workspace.sequence_active,
                                    workspace.ledger.sequence_active, sizeof(std::uint32_t),
                                    cudaMemcpyDeviceToDevice, stream))) {
      return failure;
    }
    if (!finish_stage(*stage_report)) {
      return failure;
    }

    stage_report = report(Gfn2SccStageId::kHamiltonian);
    if (!begin_stage(stage_report) ||
        !check_cuda(stage_report->stage,
                    reset_gfn2_hamiltonian_device_errors_cuda(
                        plan.topology.batch_size, mutable_system_codes(*stage_report),
                        mutable_device_error(*stage_report), stream))) {
      return failure;
    }
    const cudaError_t hamiltonian_status =
        mixed_spin
            ? assemble_gfn2_spin_hamiltonian_cuda(
                  plan.hamiltonian_batch, plan.wavefunction_layout, input.hamiltonian,
                  workspace.hamiltonian_activity, workspace.hamiltonian,
                  workspace.hamiltonian_workspace, mutable_system_codes(*stage_report),
                  mutable_device_error(*stage_report), stream)
            : assemble_gfn2_hamiltonian_cuda(
                  plan.hamiltonian_batch, input.hamiltonian, workspace.hamiltonian_activity,
                  workspace.hamiltonian, workspace.hamiltonian_workspace,
                  mutable_system_codes(*stage_report), mutable_device_error(*stage_report), stream);
    if (!check_cuda(stage_report->stage, hamiltonian_status) || !finish_stage(*stage_report)) {
      return failure;
    }
    if (segment == Gfn2SccIterationBodySegment::kPreEigensolver) {
      return {};
    }
  }

  if (segment == Gfn2SccIterationBodySegment::kFull) {
    stage_report = report(Gfn2SccStageId::kEigensolver);
    if (!begin_stage(stage_report) ||
        !check_cuda(stage_report->stage,
                    reset_gfn2_eigensolver_device_errors_cuda(
                        plan.topology.batch_size, mutable_system_codes(*stage_report),
                        mutable_device_error(*stage_report), stream))) {
      return failure;
    }
    const Gfn2EigensolverLaunchResult eigensolver =
        geometry == nullptr
            ? solve_gfn2_spin_eigensystems_cuda(
                  plan.eigensolver_batch, plan.wavefunction_layout,
                  plan.eigensolver_provider.buckets, plan.eigensolver_provider.bucket_count,
                  plan.overlap_cache, plan.geometry_generation, input.eigensolver_hamiltonians,
                  plan.eigensolver_options, plan.eigensolver_provider.solver,
                  plan.eigensolver_provider.parameters, plan.eigensolver_provider.blas,
                  workspace.eigensolver_workspace, workspace.staged_eigenpairs,
                  mutable_system_codes(*stage_report), mutable_device_error(*stage_report), stream)
            : solve_gfn2_spin_eigensystems_cuda(
                  plan.eigensolver_batch, plan.wavefunction_layout,
                  plan.eigensolver_provider.buckets, plan.eigensolver_provider.bucket_count,
                  plan.overlap_cache, geometry->epoch, input.eigensolver_hamiltonians,
                  plan.eigensolver_options, plan.eigensolver_provider.solver,
                  plan.eigensolver_provider.parameters, plan.eigensolver_provider.blas,
                  workspace.eigensolver_workspace, workspace.staged_eigenpairs,
                  mutable_system_codes(*stage_report), mutable_device_error(*stage_report), stream);
    if (!eigensolver.success()) {
      return provider_launch_failure(stage_report->stage, eigensolver);
    }
    if (!finish_stage(*stage_report)) {
      return failure;
    }
  }

  stage_report = report(Gfn2SccStageId::kOccupations);
  if (!begin_stage(stage_report) ||
      !check_cuda(stage_report->stage,
                  reset_gfn2_occupations_device_errors_cuda(
                      plan.topology.batch_size, mutable_system_codes(*stage_report),
                      mutable_device_error(*stage_report), stream)) ||
      !check_cuda(
          stage_report->stage,
          evaluate_gfn2_occupations_cuda(
              plan.occupations_batch, plan.wavefunction_layout, input.occupation_eigenvalues,
              input.occupation_eigenvalue_elements, workspace.staged_occupations,
              workspace.occupations_workspace, mutable_system_codes(*stage_report),
              mutable_device_error(*stage_report), stream)) ||
      !finish_stage(*stage_report)) {
    return failure;
  }

  stage_report = report(Gfn2SccStageId::kDensity);
  if (!begin_stage(stage_report) ||
      !check_cuda(stage_report->stage,
                  reset_gfn2_density_device_errors_cuda(
                      plan.topology.batch_size, mutable_system_codes(*stage_report),
                      mutable_device_error(*stage_report), stream)) ||
      !check_cuda(stage_report->stage,
                  evaluate_gfn2_spin_density_cuda(plan.density_batch, plan.wavefunction_layout,
                                                  input.density, workspace.staged_density,
                                                  workspace.density_workspace,
                                                  mutable_system_codes(*stage_report),
                                                  mutable_device_error(*stage_report), stream)) ||
      !finish_stage(*stage_report)) {
    return failure;
  }

  stage_report = report(Gfn2SccStageId::kMulliken);
  if (!begin_stage(stage_report) ||
      !check_cuda(stage_report->stage,
                  reset_gfn2_mulliken_device_errors_cuda(
                      plan.topology.batch_size, mutable_system_codes(*stage_report),
                      mutable_device_error(*stage_report), stream)) ||
      !check_cuda(stage_report->stage,
                  evaluate_gfn2_mulliken_population_spin_cuda(
                      plan.mulliken_batch, plan.wavefunction_layout, input.mulliken,
                      workspace.mulliken_activity, workspace.staged_raw_population,
                      workspace.mulliken_workspace, mutable_system_codes(*stage_report),
                      mutable_device_error(*stage_report), stream)) ||
      !finish_stage(*stage_report)) {
    return failure;
  }

  /* Classical SCC kernels still consume one dense physical charge channel.
   * Reuse the stable mixed-gather stage after Mulliken normalization so failed
   * peers remain inactive while successful unrestricted peers are projected
   * out of the spin-ragged population layout. */
  const Gfn2SccPotentialDeviceMixedFields raw_mixed_fields{input.raw_multipoles.shell_charges,
                                                           input.raw_multipoles.shell_elements,
                                                           input.raw_multipoles.atomic_dipoles,
                                                           input.raw_multipoles.dipole_elements,
                                                           input.raw_multipoles.atomic_quadrupoles,
                                                           input.raw_multipoles.quadrupole_elements,
                                                           plan.plan_token};
  const Gfn2SccPotentialDeviceTopologyMultipoles raw_spin_topology{
      workspace.staged_raw_population.qsh,
      workspace.staged_raw_population.qsh_elements,
      workspace.staged_raw_population.qat,
      workspace.staged_raw_population.qat_elements,
      workspace.staged_raw_population.dipole,
      workspace.staged_raw_population.dipole_elements,
      workspace.staged_raw_population.quadrupole,
      workspace.staged_raw_population.quadrupole_elements,
      plan.plan_token};
  stage_report = report(Gfn2SccStageId::kMixedGather);
  if (!begin_stage(stage_report) ||
      !check_cuda(stage_report->stage,
                  reset_gfn2_scc_potential_device_errors_cuda(
                      plan.topology.batch_size, mutable_system_codes(*stage_report),
                      mutable_device_error(*stage_report), stream)) ||
      !check_cuda(
          stage_report->stage,
          reduce_gfn2_scc_spin_atomic_charges_cuda(
              plan.potential_batch, plan.wavefunction_layout, raw_mixed_fields, workspace.activity,
              raw_spin_topology, workspace.physical_topology, workspace.potential_workspace,
              mutable_system_codes(*stage_report), mutable_device_error(*stage_report), stream)) ||
      !finish_stage(*stage_report)) {
    return failure;
  }

  stage_report = report(Gfn2SccStageId::kSpinRawEnergy);
  if (!begin_stage(stage_report) ||
      !check_cuda(stage_report->stage,
                  reset_gfn2_spin_device_errors_cuda(
                      plan.topology.batch_size, mutable_system_codes(*stage_report),
                      mutable_device_error(*stage_report), stream)) ||
      !check_cuda(stage_report->stage,
                  evaluate_gfn2_spin_polarization_cuda(
                      plan.spin_batch, input.raw_spin, workspace.activity, workspace.spin_output,
                      workspace.spin_workspace, mutable_system_codes(*stage_report),
                      mutable_device_error(*stage_report), stream)) ||
      !finish_stage(*stage_report)) {
    return failure;
  }

  stage_report = report(Gfn2SccStageId::kES2RawEnergy);
  if (!begin_stage(stage_report) ||
      !check_cuda(stage_report->stage,
                  reset_gfn2_es2_scc_errors_cuda(plan.topology.batch_size,
                                                 mutable_system_codes(*stage_report),
                                                 mutable_device_error(*stage_report), stream)) ||
      !check_cuda(stage_report->stage,
                  evaluate_gfn2_es2_scc_energy_cuda(
                      plan.es2_batch, plan.es2_cache, plan.geometry_generation, workspace.activity,
                      workspace.physical_topology.shell_charges, workspace.components.es2_energy,
                      workspace.es2_workspace, mutable_system_codes(*stage_report),
                      mutable_device_error(*stage_report), stream)) ||
      !finish_stage(*stage_report)) {
    return failure;
  }

  stage_report = report(Gfn2SccStageId::kES3RawEnergy);
  if (!begin_stage(stage_report) ||
      !check_cuda(stage_report->stage,
                  reset_gfn2_es3_scc_errors_cuda(plan.topology.batch_size,
                                                 mutable_system_codes(*stage_report),
                                                 mutable_device_error(*stage_report), stream)) ||
      !check_cuda(stage_report->stage,
                  evaluate_gfn2_es3_scc_energy_cuda(
                      plan.es3_batch, workspace.activity, workspace.physical_topology.shell_charges,
                      workspace.components.es3_energy, mutable_system_codes(*stage_report),
                      mutable_device_error(*stage_report), stream)) ||
      !finish_stage(*stage_report)) {
    return failure;
  }

  stage_report = report(Gfn2SccStageId::kAES2RawEnergy);
  if (!begin_stage(stage_report) ||
      !check_cuda(stage_report->stage,
                  reset_gfn2_aes2_device_errors_cuda(
                      plan.topology.batch_size, mutable_system_codes(*stage_report),
                      mutable_device_error(*stage_report), stream)) ||
      !check_cuda(
          stage_report->stage,
          evaluate_gfn2_aes2_scc_energy_cuda(
              plan.aes2_batch, plan.aes2_cache, plan.geometry_generation, workspace.activity,
              workspace.physical_topology.atomic_charges,
              workspace.physical_topology.atomic_dipoles,
              workspace.physical_topology.atomic_quadrupoles, workspace.components.aes2_energy,
              workspace.aes2_workspace, mutable_system_codes(*stage_report),
              mutable_device_error(*stage_report), stream)) ||
      !finish_stage(*stage_report)) {
    return failure;
  }

  if (component_enabled(plan, Gfn2SccPotentialComponent::kD4TwoBody)) {
    stage_report = report(Gfn2SccStageId::kD4RawEnergy);
    if (!begin_stage(stage_report) ||
        !check_cuda(stage_report->stage,
                    reset_gfn2_d4_device_errors_cuda(
                        plan.topology.batch_size, workspace.d4_workspace.system_errors,
                        mutable_device_error(*stage_report), stream)) ||
        !check_cuda(stage_report->stage,
                    evaluate_gfn2_d4_scc_energy_cuda(
                        plan.d4_batch, plan.d4_parameters, plan.d4_cache, plan.geometry_generation,
                        workspace.physical_topology.atomic_charges, workspace.activity,
                        workspace.components.d4_two_body_energy, workspace.d4_workspace,
                        mutable_device_error(*stage_report), stream)) ||
        !finish_stage(*stage_report)) {
      return failure;
    }
  }

  if (component_enabled(plan, Gfn2SccPotentialComponent::kExplicitPointCharge)) {
    stage_report = report(Gfn2SccStageId::kExplicitPointChargeRawEnergy);
    if (!begin_stage(stage_report) ||
        !check_cuda(stage_report->stage,
                    reset_gfn2_external_point_charge_scc_errors_cuda(
                        plan.topology.batch_size, mutable_system_codes(*stage_report),
                        mutable_device_error(*stage_report), stream)) ||
        !check_cuda(stage_report->stage,
                    evaluate_gfn2_external_point_charge_scc_energy_cuda(
                        plan.explicit_point_charge_batch, workspace.activity,
                        plan.explicit_point_charge_cache, plan.geometry_generation,
                        workspace.physical_topology.shell_charges,
                        workspace.components.explicit_point_charge_energy,
                        mutable_system_codes(*stage_report), mutable_device_error(*stage_report),
                        stream)) ||
        !finish_stage(*stage_report)) {
      return failure;
    }
  }

  if (component_enabled(plan, Gfn2SccPotentialComponent::kPeriodicEmbedding)) {
    stage_report = report(Gfn2SccStageId::kPeriodicRawEnergy);
    if (!begin_stage(stage_report) ||
        !check_cuda(stage_report->stage,
                    reset_gfn2_periodic_embedding_scc_device_errors_cuda(
                        plan.topology.batch_size, mutable_system_codes(*stage_report),
                        mutable_device_error(*stage_report), stream)) ||
        !check_cuda(stage_report->stage,
                    evaluate_gfn2_periodic_embedding_scc_energy_cuda(
                        plan.periodic_batch, plan.geometry_generation,
                        workspace.physical_topology.atomic_charges, workspace.activity,
                        workspace.components.periodic_embedding_energy,
                        workspace.periodic_workspace, mutable_system_codes(*stage_report),
                        mutable_device_error(*stage_report), stream)) ||
        !finish_stage(*stage_report)) {
      return failure;
    }
  }

  stage_report = report(Gfn2SccStageId::kClassicalEnergy);
  if (!begin_stage(stage_report) ||
      !check_cuda(stage_report->stage,
                  reset_gfn2_scc_classical_energy_device_errors_cuda(
                      plan.topology.batch_size, mutable_system_codes(*stage_report),
                      mutable_device_error(*stage_report), stream)) ||
      !check_cuda(stage_report->stage,
                  evaluate_gfn2_scc_classical_energy_cuda(
                      plan.classical_energy_batch, input.classical_energy,
                      workspace.classical_energy_activity, workspace.staged_classical_energy,
                      workspace.classical_energy_workspace, mutable_system_codes(*stage_report),
                      mutable_device_error(*stage_report), stream)) ||
      !finish_stage(*stage_report)) {
    return failure;
  }

  stage_report = report(Gfn2SccStageId::kElectronicEnergy);
  if (!begin_stage(stage_report) ||
      !check_cuda(stage_report->stage,
                  reset_gfn2_scc_energy_device_errors_cuda(
                      plan.topology.batch_size, mutable_system_codes(*stage_report),
                      mutable_device_error(*stage_report), stream)) ||
      !check_cuda(stage_report->stage,
                  evaluate_gfn2_scc_electronic_energy_spin_cuda(
                      plan.electronic_energy_batch, plan.wavefunction_layout,
                      input.electronic_energy.density, input.electronic_energy.density_elements,
                      input.electronic_energy.h0, input.electronic_energy.entropies,
                      plan.free_energy_batch.electronic_temperature, workspace.activity,
                      workspace.components.core_energy, workspace.components.electronic_free_energy,
                      workspace.electronic_energy_workspace, mutable_system_codes(*stage_report),
                      mutable_device_error(*stage_report), stream)) ||
      !finish_stage(*stage_report)) {
    return failure;
  }

  stage_report = report(Gfn2SccStageId::kFreeEnergy);
  if (!begin_stage(stage_report) ||
      !check_cuda(stage_report->stage,
                  reset_gfn2_scc_free_energy_device_errors_cuda(
                      plan.topology.batch_size, mutable_system_codes(*stage_report),
                      mutable_device_error(*stage_report), stream)) ||
      !check_cuda(
          stage_report->stage,
          compose_gfn2_scc_free_energy_cuda(
              plan.free_energy_batch, input.free_energy, workspace.free_energy_activity,
              workspace.staged_free_energy, workspace.free_energy_workspace,
              mutable_system_codes(*stage_report), mutable_device_error(*stage_report), stream)) ||
      !finish_stage(*stage_report)) {
    return failure;
  }

  stage_report = report(Gfn2SccStageId::kMixer);
  if (!begin_stage(stage_report) ||
      !check_cuda(stage_report->stage, stage_active_mixer(binding, stream)) ||
      !check_cuda(stage_report->stage,
                  cudaMemsetAsync(workspace.mixer_device_error, 0,
                                  sizeof(*workspace.mixer_device_error), stream)) ||
      !check_cuda(
          stage_report->stage,
          mix_gfn2_scc_broyden_cuda(plan.scc_batch, plan.wavefunction_layout, plan.mixer_policy,
                                    workspace.activity, input.raw_multipoles, workspace.next_mixed,
                                    workspace.staged_mixer, workspace.mixer_workspace,
                                    workspace.mixer_device_error, stream)) ||
      !finish_stage(*stage_report)) {
    return failure;
  }

  stage_report = report(Gfn2SccStageId::kStatePublication);
  if (!begin_stage(stage_report) ||
      !check_cuda(stage_report->stage,
                  reset_gfn2_scc_publication_errors_cuda(
                      plan.publication_plan, workspace.publication_workspace, stream)) ||
      !check_cuda(stage_report->stage,
                  preflight_gfn2_scc_publication_cuda(
                      plan.publication_plan, workspace.activity, workspace.ledger,
                      workspace.staged_publication, state.publication,
                      workspace.publication_workspace, stream)) ||
      !finish_stage(*stage_report) ||
      !check_cuda(stage_report->stage,
                  commit_gfn2_scc_publication_cuda(plan.publication_plan, workspace.activity,
                                                   workspace.ledger, workspace.staged_publication,
                                                   state.publication,
                                                   workspace.publication_workspace, stream))) {
    return failure;
  }

  return {};
}

Gfn2SccIterationLaunchResult launch_gfn2_scc_iteration_cuda(const Gfn2SccIterationBinding& binding,
                                                            cudaStream_t stream) noexcept {
  return launch_scc_iteration_impl<false>(binding, nullptr, stream, true, true);
}

Gfn2SccIterationLaunchResult launch_gfn2_scc_iteration_cuda(
    const Gfn2SccIterationBinding& binding, const Gfn2GeometryEpochConsumerDevice& geometry,
    cudaStream_t stream) noexcept {
  return launch_scc_iteration_impl<false>(binding, &geometry, stream, true, true);
}

#if defined(XTBLOOM_CUDA_TEST_HOOKS)
Gfn2SccIterationLaunchResult launch_gfn2_scc_iteration_test_fault_cuda(
    const Gfn2SccIterationBinding& binding, const Gfn2SccIterationTestFault& fault,
    cudaStream_t stream) noexcept {
  return launch_scc_iteration_impl<true>(binding, nullptr, stream, true, true,
                                         Gfn2SccIterationBodySegment::kFull, &fault);
}
#endif

Gfn2SccIterationLaunchResult launch_gfn2_scc_activity_cuda(const Gfn2SccIterationBinding& binding,
                                                           cudaStream_t stream) noexcept {
  return launch_scc_iteration_impl<false>(binding, nullptr, stream, true, false);
}

Gfn2SccIterationLaunchResult launch_gfn2_scc_activity_cuda(
    const Gfn2SccIterationBinding& binding, const Gfn2GeometryEpochConsumerDevice& geometry,
    cudaStream_t stream) noexcept {
  return launch_scc_iteration_impl<false>(binding, &geometry, stream, true, false);
}

Gfn2SccIterationLaunchResult launch_gfn2_scc_numerical_body_cuda(
    const Gfn2SccIterationBinding& binding, cudaStream_t stream) noexcept {
  return launch_scc_iteration_impl<false>(binding, nullptr, stream, false, true);
}

Gfn2SccIterationLaunchResult launch_gfn2_scc_numerical_body_cuda(
    const Gfn2SccIterationBinding& binding, const Gfn2GeometryEpochConsumerDevice& geometry,
    cudaStream_t stream) noexcept {
  return launch_scc_iteration_impl<false>(binding, &geometry, stream, false, true);
}

Gfn2SccIterationLaunchResult launch_gfn2_scc_pre_eigensolver_cuda(
    const Gfn2SccIterationBinding& binding, cudaStream_t stream) noexcept {
  return launch_scc_iteration_impl<false>(binding, nullptr, stream, false, true,
                                          Gfn2SccIterationBodySegment::kPreEigensolver);
}

Gfn2SccIterationLaunchResult launch_gfn2_scc_pre_eigensolver_cuda(
    const Gfn2SccIterationBinding& binding, const Gfn2GeometryEpochConsumerDevice& geometry,
    cudaStream_t stream) noexcept {
  return launch_scc_iteration_impl<false>(binding, &geometry, stream, false, true,
                                          Gfn2SccIterationBodySegment::kPreEigensolver);
}

Gfn2SccIterationLaunchResult launch_gfn2_scc_post_eigensolver_cuda(
    const Gfn2SccIterationBinding& binding, cudaStream_t stream) noexcept {
  return launch_scc_iteration_impl<false>(binding, nullptr, stream, false, true,
                                          Gfn2SccIterationBodySegment::kPostEigensolver);
}

Gfn2SccIterationLaunchResult launch_gfn2_scc_post_eigensolver_cuda(
    const Gfn2SccIterationBinding& binding, const Gfn2GeometryEpochConsumerDevice& geometry,
    cudaStream_t stream) noexcept {
  return launch_scc_iteration_impl<false>(binding, &geometry, stream, false, true,
                                          Gfn2SccIterationBodySegment::kPostEigensolver);
}

Gfn2SccIterationLaunchResult launch_gfn2_restricted_scc_iteration_cuda(
    const Gfn2SccIterationBinding& binding, cudaStream_t stream) noexcept {
  return launch_gfn2_scc_iteration_cuda(binding, stream);
}

Gfn2SccIterationLaunchResult launch_gfn2_restricted_scc_iteration_cuda(
    const Gfn2SccIterationBinding& binding, const Gfn2GeometryEpochConsumerDevice& geometry,
    cudaStream_t stream) noexcept {
  return launch_gfn2_scc_iteration_cuda(binding, geometry, stream);
}

Gfn2SccIterationLaunchResult launch_gfn2_restricted_scc_activity_cuda(
    const Gfn2SccIterationBinding& binding, cudaStream_t stream) noexcept {
  return launch_gfn2_scc_activity_cuda(binding, stream);
}

Gfn2SccIterationLaunchResult launch_gfn2_restricted_scc_activity_cuda(
    const Gfn2SccIterationBinding& binding, const Gfn2GeometryEpochConsumerDevice& geometry,
    cudaStream_t stream) noexcept {
  return launch_gfn2_scc_activity_cuda(binding, geometry, stream);
}

Gfn2SccIterationLaunchResult launch_gfn2_restricted_scc_numerical_body_cuda(
    const Gfn2SccIterationBinding& binding, cudaStream_t stream) noexcept {
  return launch_gfn2_scc_numerical_body_cuda(binding, stream);
}

Gfn2SccIterationLaunchResult launch_gfn2_restricted_scc_numerical_body_cuda(
    const Gfn2SccIterationBinding& binding, const Gfn2GeometryEpochConsumerDevice& geometry,
    cudaStream_t stream) noexcept {
  return launch_gfn2_scc_numerical_body_cuda(binding, geometry, stream);
}

Gfn2SccIterationLaunchResult launch_gfn2_restricted_scc_pre_eigensolver_cuda(
    const Gfn2SccIterationBinding& binding, cudaStream_t stream) noexcept {
  return launch_gfn2_scc_pre_eigensolver_cuda(binding, stream);
}

Gfn2SccIterationLaunchResult launch_gfn2_restricted_scc_pre_eigensolver_cuda(
    const Gfn2SccIterationBinding& binding, const Gfn2GeometryEpochConsumerDevice& geometry,
    cudaStream_t stream) noexcept {
  return launch_gfn2_scc_pre_eigensolver_cuda(binding, geometry, stream);
}

Gfn2SccIterationLaunchResult launch_gfn2_restricted_scc_post_eigensolver_cuda(
    const Gfn2SccIterationBinding& binding, cudaStream_t stream) noexcept {
  return launch_gfn2_scc_post_eigensolver_cuda(binding, stream);
}

Gfn2SccIterationLaunchResult launch_gfn2_restricted_scc_post_eigensolver_cuda(
    const Gfn2SccIterationBinding& binding, const Gfn2GeometryEpochConsumerDevice& geometry,
    cudaStream_t stream) noexcept {
  return launch_gfn2_scc_post_eigensolver_cuda(binding, geometry, stream);
}

}  // namespace xtbloom::detail::cuda
