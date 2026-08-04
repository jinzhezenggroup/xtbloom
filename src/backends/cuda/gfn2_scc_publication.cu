#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_scc_publication.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr std::int64_t kDipoleComponents = 3;
constexpr std::int64_t kQuadrupoleComponents = 6;
constexpr std::int64_t kMultipoleAtomComponents = kDipoleComponents + kQuadrupoleComponents;

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

template <std::size_t Capacity>
struct RangeSet {
  std::array<AddressRange, Capacity> ranges{};
  std::size_t count = 0u;
};

bool checked_multiply(std::int64_t first, std::int64_t second, std::int64_t* product) noexcept {
  if (first < 0 || second < 0 ||
      (first != 0 && second > std::numeric_limits<std::int64_t>::max() / first)) {
    return false;
  }
  *product = first * second;
  return true;
}

bool checked_add(std::int64_t first, std::int64_t second, std::int64_t* sum) noexcept {
  if (first < 0 || second < 0 || second > std::numeric_limits<std::int64_t>::max() - first) {
    return false;
  }
  *sum = first + second;
  return true;
}

bool is_aligned(const void* pointer, std::size_t alignment) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

bool make_range(const void* pointer, std::int64_t elements, std::size_t element_size,
                AddressRange* range) noexcept {
  if (elements < 0 || element_size == 0u ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(elements) * element_size;
  if (bytes == 0u) {
    *range = {};
    return pointer == nullptr;
  }
  if (pointer == nullptr) {
    return false;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
    return false;
  }
  *range = {begin, begin + bytes};
  return true;
}

bool ranges_overlap(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin < second.end && second.begin < first.end;
}

template <std::size_t Capacity>
bool append_range(RangeSet<Capacity>* set, const void* pointer, std::int64_t elements,
                  std::size_t element_size) noexcept {
  if (set->count >= Capacity ||
      !make_range(pointer, elements, element_size, &set->ranges[set->count])) {
    return false;
  }
  ++set->count;
  return true;
}

template <std::size_t Capacity>
bool pairwise_disjoint(const RangeSet<Capacity>& set) noexcept {
  for (std::size_t first = 0u; first < set.count; ++first) {
    for (std::size_t second = first + 1u; second < set.count; ++second) {
      if (ranges_overlap(set.ranges[first], set.ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

template <std::size_t FirstCapacity, std::size_t SecondCapacity>
bool disjoint_sets(const RangeSet<FirstCapacity>& first,
                   const RangeSet<SecondCapacity>& second) noexcept {
  for (std::size_t i = 0u; i < first.count; ++i) {
    for (std::size_t j = 0u; j < second.count; ++j) {
      if (ranges_overlap(first.ranges[i], second.ranges[j])) {
        return false;
      }
    }
  }
  return true;
}

bool same_pointer(const void* first, const void* second) noexcept { return first == second; }

bool valid_layout_binding(const Gfn2SccPublicationDevicePlan& plan) noexcept {
  const auto& layout = plan.wavefunction_layout;
  return layout.memory_space == Gfn2PlanMemorySpace::kCudaDevice &&
         layout.plan_token == plan.plan_token && layout.batch_size == plan.batch_size &&
         layout.total_spin_channels >= plan.batch_size &&
         layout.total_spin_channels <= 2 * plan.batch_size &&
         layout.total_spin_orbitals >= plan.total_orbitals &&
         layout.total_spin_orbitals - plan.total_orbitals <= plan.total_orbitals &&
         layout.total_spin_matrix_elements >= plan.total_matrix_elements &&
         layout.total_spin_matrix_elements - plan.total_matrix_elements <=
             plan.total_matrix_elements &&
         layout.total_spin_shells >= plan.total_shells &&
         layout.total_spin_shells - plan.total_shells <= plan.total_shells &&
         layout.total_spin_atoms >= plan.total_atoms &&
         layout.total_spin_atoms - plan.total_atoms <= plan.total_atoms &&
         layout.spin_channel_count == plan.batch_size &&
         layout.spin_channel_offset_count == plan.batch_size + 1 &&
         layout.spin_orbital_offset_count == plan.batch_size + 1 &&
         layout.spin_matrix_offset_count == plan.batch_size + 1 &&
         layout.spin_shell_offset_count == plan.batch_size + 1 &&
         layout.spin_atom_offset_count == plan.batch_size + 1 &&
         is_aligned(layout.spin_channels, alignof(std::int32_t)) &&
         is_aligned(layout.spin_channel_offsets, alignof(std::int64_t)) &&
         is_aligned(layout.spin_orbital_offsets, alignof(std::int64_t)) &&
         is_aligned(layout.spin_matrix_offsets, alignof(std::int64_t)) &&
         is_aligned(layout.spin_shell_offsets, alignof(std::int64_t)) &&
         is_aligned(layout.spin_atom_offsets, alignof(std::int64_t));
}

bool valid_plan(const Gfn2SccPublicationDevicePlan& plan, std::int64_t* dipoles,
                std::int64_t* quadrupoles, std::int64_t* two_orbitals, std::int64_t* two_batch,
                std::int64_t* history_elements, std::int64_t* omega_elements) noexcept {
  std::int64_t atom_components = 0;
  std::int64_t expected_vector = 0;
  return plan.batch_size > 0 && plan.batch_size <= std::numeric_limits<unsigned int>::max() &&
         plan.total_atoms > 0 && plan.total_shells > 0 && plan.total_orbitals > 0 &&
         plan.total_matrix_elements > 0 && plan.history_size > 0 && plan.maximum_iterations > 0u &&
         plan.plan_token != 0u && std::isfinite(plan.residual_rms_tolerance) &&
         plan.residual_rms_tolerance > 0.0 && std::isfinite(plan.energy_tolerance) &&
         plan.energy_tolerance > 0.0 && plan.atom_offset_count == plan.batch_size + 1 &&
         plan.shell_offset_count == plan.batch_size + 1 &&
         plan.orbital_offset_count == plan.batch_size + 1 &&
         plan.matrix_offset_count == plan.batch_size + 1 &&
         plan.shell_to_atom_count == plan.total_shells && valid_layout_binding(plan) &&
         is_aligned(plan.atom_offsets, alignof(std::int64_t)) &&
         is_aligned(plan.shell_offsets, alignof(std::int64_t)) &&
         is_aligned(plan.orbital_offsets, alignof(std::int64_t)) &&
         is_aligned(plan.matrix_offsets, alignof(std::int64_t)) &&
         is_aligned(plan.shell_to_atom, alignof(std::int64_t)) &&
         checked_multiply(plan.wavefunction_layout.total_spin_atoms, kDipoleComponents, dipoles) &&
         checked_multiply(plan.wavefunction_layout.total_spin_atoms, kQuadrupoleComponents,
                          quadrupoles) &&
         checked_multiply(plan.wavefunction_layout.total_spin_atoms, kMultipoleAtomComponents,
                          &atom_components) &&
         checked_add(plan.wavefunction_layout.total_spin_shells, atom_components,
                     &expected_vector) &&
         expected_vector == plan.total_mixer_vector_elements &&
         checked_multiply(plan.total_orbitals, 2, two_orbitals) &&
         checked_multiply(plan.batch_size, 2, two_batch) &&
         checked_multiply(plan.total_mixer_vector_elements, plan.history_size, history_elements) &&
         checked_multiply(plan.batch_size, plan.history_size, omega_elements);
}

bool valid_eigenpairs(const Gfn2EigensolverDeviceResults& values,
                      const Gfn2SccPublicationDevicePlan& plan) noexcept {
  return values.plan_token == plan.plan_token &&
         values.eigenvalue_elements == plan.wavefunction_layout.total_spin_orbitals &&
         values.coefficient_elements == plan.wavefunction_layout.total_spin_matrix_elements &&
         is_aligned(values.eigenvalues, alignof(double)) &&
         is_aligned(values.coefficients, alignof(double));
}

bool valid_occupations(const Gfn2OccupationsDeviceResults& values,
                       const Gfn2SccPublicationDevicePlan& plan, std::int64_t two_orbitals,
                       std::int64_t two_batch) noexcept {
  return values.plan_token == plan.plan_token && values.occupation_elements == two_orbitals &&
         values.chemical_potential_elements == two_batch &&
         values.electron_sum_elements == two_batch && values.entropy_elements == plan.batch_size &&
         is_aligned(values.occupations, alignof(double)) &&
         is_aligned(values.chemical_potentials, alignof(double)) &&
         is_aligned(values.electron_sums, alignof(double)) &&
         is_aligned(values.entropies, alignof(double));
}

bool valid_density(const Gfn2DensityDeviceResults& values,
                   const Gfn2SccPublicationDevicePlan& plan) noexcept {
  return values.plan_token == plan.plan_token &&
         values.density_elements == plan.wavefunction_layout.total_spin_matrix_elements &&
         values.weighted_density_elements == plan.wavefunction_layout.total_spin_matrix_elements &&
         values.band_energy_elements == plan.batch_size &&
         values.occupation_sum_elements == plan.batch_size &&
         values.density_trace_elements == plan.batch_size &&
         values.weighted_density_trace_elements == plan.batch_size &&
         values.channel_band_energy_elements == plan.wavefunction_layout.total_spin_channels &&
         values.channel_occupation_sum_elements == plan.wavefunction_layout.total_spin_channels &&
         values.channel_density_trace_elements == plan.wavefunction_layout.total_spin_channels &&
         values.channel_weighted_density_trace_elements ==
             plan.wavefunction_layout.total_spin_channels &&
         is_aligned(values.density, alignof(double)) &&
         is_aligned(values.energy_weighted_density, alignof(double)) &&
         is_aligned(values.band_energies, alignof(double)) &&
         is_aligned(values.occupation_sums, alignof(double)) &&
         is_aligned(values.density_traces, alignof(double)) &&
         is_aligned(values.weighted_density_traces, alignof(double)) &&
         is_aligned(values.channel_band_energies, alignof(double)) &&
         is_aligned(values.channel_occupation_sums, alignof(double)) &&
         is_aligned(values.channel_density_traces, alignof(double)) &&
         is_aligned(values.channel_weighted_density_traces, alignof(double));
}

bool valid_population(const Gfn2MullikenDevicePopulation& values,
                      const Gfn2SccPublicationDevicePlan& plan, std::int64_t dipoles,
                      std::int64_t quadrupoles) noexcept {
  return values.plan_token == plan.plan_token &&
         values.qsh_elements == plan.wavefunction_layout.total_spin_shells &&
         values.qat_elements == plan.wavefunction_layout.total_spin_atoms &&
         values.dipole_elements == dipoles && values.quadrupole_elements == quadrupoles &&
         is_aligned(values.qsh, alignof(double)) && is_aligned(values.qat, alignof(double)) &&
         is_aligned(values.dipole, alignof(double)) &&
         is_aligned(values.quadrupole, alignof(double));
}

bool valid_wavefunction(const Gfn2SccPublicationDeviceWavefunction& values,
                        const Gfn2SccPublicationDevicePlan& plan, std::int64_t dipoles,
                        std::int64_t quadrupoles, std::int64_t two_orbitals,
                        std::int64_t two_batch) noexcept {
  return values.plan_token == plan.plan_token && valid_eigenpairs(values.eigenpairs, plan) &&
         valid_occupations(values.occupations, plan, two_orbitals, two_batch) &&
         valid_density(values.density, plan) &&
         valid_population(values.population, plan, dipoles, quadrupoles);
}

bool valid_classical(const Gfn2SccClassicalEnergyDeviceDiagnostics& values,
                     const Gfn2SccPublicationDevicePlan& plan) noexcept {
  const std::array<std::pair<double*, std::int64_t>, 7> fields{{
      {values.es2, values.es2_elements},
      {values.es3, values.es3_elements},
      {values.aes2, values.aes2_elements},
      {values.d4_two_body, values.d4_two_body_elements},
      {values.explicit_point_charge, values.explicit_point_charge_elements},
      {values.periodic_embedding, values.periodic_embedding_elements},
      {values.classical_total, values.classical_total_elements},
  }};
  if (values.plan_token != plan.plan_token) {
    return false;
  }
  for (const auto& field : fields) {
    if (field.second != plan.batch_size || !is_aligned(field.first, alignof(double))) {
      return false;
    }
  }
  return true;
}

bool valid_free_energy(const Gfn2SccFreeEnergyDeviceDiagnostics& values,
                       const Gfn2SccPublicationDevicePlan& plan) noexcept {
  const std::array<std::pair<double*, std::int64_t>, 11> fields{{
      {values.core, values.core_elements},
      {values.es2, values.es2_elements},
      {values.es3, values.es3_elements},
      {values.aes2, values.aes2_elements},
      {values.spin, values.spin_elements},
      {values.d4_two_body, values.d4_two_body_elements},
      {values.explicit_point_charge, values.explicit_point_charge_elements},
      {values.periodic_embedding, values.periodic_embedding_elements},
      {values.entropy, values.entropy_elements},
      {values.internal_energy, values.internal_energy_elements},
      {values.free_energy, values.free_energy_elements},
  }};
  if (values.plan_token != plan.plan_token) {
    return false;
  }
  for (const auto& field : fields) {
    if (field.second != plan.batch_size || !is_aligned(field.first, alignof(double))) {
      return false;
    }
  }
  return true;
}

bool valid_energy(const Gfn2SccPublicationDeviceEnergyTrace& values,
                  const Gfn2SccPublicationDevicePlan& plan) noexcept {
  return values.plan_token == plan.plan_token && valid_classical(values.classical, plan) &&
         valid_free_energy(values.free_energy, plan) &&
         values.spin_energy_elements == plan.batch_size &&
         is_aligned(values.spin_energies, alignof(double)) &&
         same_pointer(values.spin_energies, values.free_energy.spin) &&
         same_pointer(values.classical.es2, values.free_energy.es2) &&
         same_pointer(values.classical.es3, values.free_energy.es3) &&
         same_pointer(values.classical.aes2, values.free_energy.aes2) &&
         same_pointer(values.classical.d4_two_body, values.free_energy.d4_two_body) &&
         same_pointer(values.classical.explicit_point_charge,
                      values.free_energy.explicit_point_charge) &&
         same_pointer(values.classical.periodic_embedding, values.free_energy.periodic_embedding);
}

bool valid_mixer(const Gfn2SccMixerDeviceState& values, const Gfn2SccPublicationDevicePlan& plan,
                 std::int64_t history_elements, std::int64_t omega_elements) noexcept {
  return values.plan_token == plan.plan_token &&
         values.total_vector_elements == plan.total_mixer_vector_elements &&
         values.history_elements == history_elements && values.omega_elements == omega_elements &&
         values.batch_elements == plan.batch_size &&
         is_aligned(values.current_inputs, alignof(double)) &&
         is_aligned(values.previous_inputs, alignof(double)) &&
         is_aligned(values.previous_residuals, alignof(double)) &&
         is_aligned(values.df_history, alignof(double)) &&
         is_aligned(values.u_history, alignof(double)) &&
         is_aligned(values.omega, alignof(double)) &&
         is_aligned(values.residual_rms, alignof(double)) &&
         is_aligned(values.residual_maximum, alignof(double)) &&
         is_aligned(values.iterations, alignof(std::uint64_t)) &&
         is_aligned(values.restart_counts, alignof(std::uint64_t)) &&
         is_aligned(values.system_statuses, alignof(gpuxtb_status_t)) &&
         is_aligned(values.initialized, alignof(std::uint8_t)) &&
         is_aligned(values.residual_converged, alignof(std::uint8_t));
}

bool valid_const_multipoles(const Gfn2SccDeviceConstMultipoles& values,
                            const Gfn2SccPublicationDevicePlan& plan, std::int64_t dipoles,
                            std::int64_t quadrupoles) noexcept {
  return values.plan_token == plan.plan_token &&
         values.shell_elements == plan.wavefunction_layout.total_spin_shells &&
         values.dipole_elements == dipoles && values.quadrupole_elements == quadrupoles &&
         is_aligned(values.shell_charges, alignof(double)) &&
         is_aligned(values.atomic_dipoles, alignof(double)) &&
         is_aligned(values.atomic_quadrupoles, alignof(double));
}

bool valid_multipoles(const Gfn2SccDeviceMultipoles& values,
                      const Gfn2SccPublicationDevicePlan& plan, std::int64_t dipoles,
                      std::int64_t quadrupoles) noexcept {
  return values.plan_token == plan.plan_token &&
         values.shell_elements == plan.wavefunction_layout.total_spin_shells &&
         values.dipole_elements == dipoles && values.quadrupole_elements == quadrupoles &&
         is_aligned(values.shell_charges, alignof(double)) &&
         is_aligned(values.atomic_dipoles, alignof(double)) &&
         is_aligned(values.atomic_quadrupoles, alignof(double));
}

bool valid_scc_state(const Gfn2SccDeviceState& values, const Gfn2SccPublicationDevicePlan& plan,
                     std::int64_t dipoles, std::int64_t quadrupoles) noexcept {
  return values.plan_token == plan.plan_token && values.batch_elements == plan.batch_size &&
         valid_multipoles(values.current_inputs, plan, dipoles, quadrupoles) &&
         is_aligned(values.free_energies, alignof(double)) &&
         is_aligned(values.previous_free_energies, alignof(double)) &&
         is_aligned(values.free_energy_changes, alignof(double)) &&
         is_aligned(values.residual_rms, alignof(double)) &&
         is_aligned(values.iterations, alignof(std::uint64_t)) &&
         is_aligned(values.system_statuses, alignof(gpuxtb_status_t)) &&
         is_aligned(values.converged, alignof(std::uint8_t));
}

bool valid_staged(const Gfn2SccPublicationDeviceStagedState& staged,
                  const Gfn2SccPublicationDevicePlan& plan, std::int64_t dipoles,
                  std::int64_t quadrupoles, std::int64_t two_orbitals, std::int64_t two_batch,
                  std::int64_t history_elements, std::int64_t omega_elements) noexcept {
  return staged.plan_token == plan.plan_token &&
         valid_wavefunction(staged.wavefunction, plan, dipoles, quadrupoles, two_orbitals,
                            two_batch) &&
         valid_energy(staged.energy, plan) &&
         same_pointer(staged.energy.free_energy.entropy,
                      staged.wavefunction.occupations.entropies) &&
         valid_mixer(staged.mixer, plan, history_elements, omega_elements) &&
         valid_const_multipoles(staged.next_mixed, plan, dipoles, quadrupoles);
}

bool valid_public(const Gfn2SccPublicationDevicePublicState& values,
                  const Gfn2SccPublicationDevicePlan& plan, std::int64_t dipoles,
                  std::int64_t quadrupoles, std::int64_t two_orbitals, std::int64_t two_batch,
                  std::int64_t history_elements, std::int64_t omega_elements) noexcept {
  return values.plan_token == plan.plan_token &&
         valid_wavefunction(values.wavefunction, plan, dipoles, quadrupoles, two_orbitals,
                            two_batch) &&
         valid_energy(values.energy, plan) &&
         valid_mixer(values.mixer, plan, history_elements, omega_elements) &&
         valid_multipoles(values.published, plan, dipoles, quadrupoles) &&
         valid_scc_state(values.scc, plan, dipoles, quadrupoles) &&
         same_pointer(values.wavefunction.population.qsh, values.published.shell_charges) &&
         same_pointer(values.wavefunction.population.dipole, values.published.atomic_dipoles) &&
         same_pointer(values.wavefunction.population.quadrupole,
                      values.published.atomic_quadrupoles);
}

bool valid_activity(const Gfn2SccIterationDeviceActivity& activity,
                    const Gfn2SccPublicationDevicePlan& plan) noexcept {
  return activity.plan_token == plan.plan_token && activity.batch_elements == plan.batch_size &&
         activity.sequence_elements == 1 &&
         is_aligned(activity.active_mask, alignof(std::uint8_t)) &&
         is_aligned(activity.sequence_active, alignof(std::uint32_t));
}

bool valid_ledger(const Gfn2SccIterationDeviceLedger& ledger,
                  const Gfn2SccIterationDeviceActivity& activity,
                  const Gfn2SccPublicationDevicePlan& plan) noexcept {
  return ledger.plan_token == plan.plan_token && ledger.batch_elements == plan.batch_size &&
         ledger.scalar_elements == 1 && is_aligned(ledger.active_mask, alignof(std::uint8_t)) &&
         is_aligned(ledger.pending_statuses, alignof(gpuxtb_status_t)) &&
         is_aligned(ledger.system_failure_records, alignof(std::uint64_t)) &&
         is_aligned(ledger.plan_failure_record, alignof(std::uint64_t)) &&
         is_aligned(ledger.sequence_active, alignof(std::uint32_t)) &&
         same_pointer(activity.active_mask, ledger.active_mask) &&
         same_pointer(activity.sequence_active, ledger.sequence_active);
}

bool valid_workspace(const Gfn2SccPublicationDeviceWorkspace& workspace,
                     const Gfn2SccPublicationDevicePlan& plan) noexcept {
  return workspace.plan_token == plan.plan_token &&
         workspace.mixed_atomic_charge_elements == plan.wavefunction_layout.total_spin_atoms &&
         workspace.batch_elements == plan.batch_size &&
         workspace.system_error_elements == plan.batch_size &&
         workspace.device_error_elements == 1 && workspace.sequence_elements == 1 &&
         is_aligned(workspace.mixed_atomic_charges, alignof(double)) &&
         is_aligned(workspace.previous_free_energies, alignof(double)) &&
         is_aligned(workspace.free_energy_changes, alignof(double)) &&
         is_aligned(workspace.next_iterations, alignof(std::uint64_t)) &&
         is_aligned(workspace.next_statuses, alignof(gpuxtb_status_t)) &&
         is_aligned(workspace.next_converged, alignof(std::uint8_t)) &&
         is_aligned(workspace.system_errors, alignof(std::uint32_t)) &&
         is_aligned(workspace.device_error, alignof(std::uint32_t)) &&
         is_aligned(workspace.sequence_active, alignof(std::uint32_t));
}

template <std::size_t Capacity>
bool append_wavefunction_ranges(RangeSet<Capacity>* set,
                                const Gfn2SccPublicationDeviceWavefunction& values,
                                const Gfn2SccPublicationDevicePlan& plan, std::int64_t dipoles,
                                std::int64_t quadrupoles, std::int64_t two_orbitals,
                                std::int64_t two_batch) noexcept {
  return append_range(set, values.eigenpairs.eigenvalues,
                      plan.wavefunction_layout.total_spin_orbitals, sizeof(double)) &&
         append_range(set, values.eigenpairs.coefficients,
                      plan.wavefunction_layout.total_spin_matrix_elements, sizeof(double)) &&
         append_range(set, values.occupations.occupations, two_orbitals, sizeof(double)) &&
         append_range(set, values.occupations.chemical_potentials, two_batch, sizeof(double)) &&
         append_range(set, values.occupations.electron_sums, two_batch, sizeof(double)) &&
         append_range(set, values.occupations.entropies, plan.batch_size, sizeof(double)) &&
         append_range(set, values.density.density,
                      plan.wavefunction_layout.total_spin_matrix_elements, sizeof(double)) &&
         append_range(set, values.density.energy_weighted_density,
                      plan.wavefunction_layout.total_spin_matrix_elements, sizeof(double)) &&
         append_range(set, values.density.band_energies, plan.batch_size, sizeof(double)) &&
         append_range(set, values.density.occupation_sums, plan.batch_size, sizeof(double)) &&
         append_range(set, values.density.density_traces, plan.batch_size, sizeof(double)) &&
         append_range(set, values.density.weighted_density_traces, plan.batch_size,
                      sizeof(double)) &&
         append_range(set, values.density.channel_band_energies,
                      plan.wavefunction_layout.total_spin_channels, sizeof(double)) &&
         append_range(set, values.density.channel_occupation_sums,
                      plan.wavefunction_layout.total_spin_channels, sizeof(double)) &&
         append_range(set, values.density.channel_density_traces,
                      plan.wavefunction_layout.total_spin_channels, sizeof(double)) &&
         append_range(set, values.density.channel_weighted_density_traces,
                      plan.wavefunction_layout.total_spin_channels, sizeof(double)) &&
         append_range(set, values.population.qsh, plan.wavefunction_layout.total_spin_shells,
                      sizeof(double)) &&
         append_range(set, values.population.qat, plan.wavefunction_layout.total_spin_atoms,
                      sizeof(double)) &&
         append_range(set, values.population.dipole, dipoles, sizeof(double)) &&
         append_range(set, values.population.quadrupole, quadrupoles, sizeof(double));
}

template <std::size_t Capacity>
bool append_energy_ranges(RangeSet<Capacity>* set,
                          const Gfn2SccPublicationDeviceEnergyTrace& values,
                          const Gfn2SccPublicationDevicePlan& plan,
                          bool entropy_already_registered = false) noexcept {
  const auto& free = values.free_energy;
  return append_range(set, free.core, plan.batch_size, sizeof(double)) &&
         append_range(set, free.es2, plan.batch_size, sizeof(double)) &&
         append_range(set, free.es3, plan.batch_size, sizeof(double)) &&
         append_range(set, free.aes2, plan.batch_size, sizeof(double)) &&
         append_range(set, values.spin_energies, plan.batch_size, sizeof(double)) &&
         append_range(set, free.d4_two_body, plan.batch_size, sizeof(double)) &&
         append_range(set, free.explicit_point_charge, plan.batch_size, sizeof(double)) &&
         append_range(set, free.periodic_embedding, plan.batch_size, sizeof(double)) &&
         (entropy_already_registered ||
          append_range(set, free.entropy, plan.batch_size, sizeof(double))) &&
         append_range(set, free.internal_energy, plan.batch_size, sizeof(double)) &&
         append_range(set, free.free_energy, plan.batch_size, sizeof(double)) &&
         append_range(set, values.classical.classical_total, plan.batch_size, sizeof(double));
}

template <std::size_t Capacity>
bool append_mixer_ranges(RangeSet<Capacity>* set, const Gfn2SccMixerDeviceState& values,
                         const Gfn2SccPublicationDevicePlan& plan, std::int64_t history_elements,
                         std::int64_t omega_elements) noexcept {
  return append_range(set, values.current_inputs, plan.total_mixer_vector_elements,
                      sizeof(double)) &&
         append_range(set, values.previous_inputs, plan.total_mixer_vector_elements,
                      sizeof(double)) &&
         append_range(set, values.previous_residuals, plan.total_mixer_vector_elements,
                      sizeof(double)) &&
         append_range(set, values.df_history, history_elements, sizeof(double)) &&
         append_range(set, values.u_history, history_elements, sizeof(double)) &&
         append_range(set, values.omega, omega_elements, sizeof(double)) &&
         append_range(set, values.residual_rms, plan.batch_size, sizeof(double)) &&
         append_range(set, values.residual_maximum, plan.batch_size, sizeof(double)) &&
         append_range(set, values.iterations, plan.batch_size, sizeof(std::uint64_t)) &&
         append_range(set, values.restart_counts, plan.batch_size, sizeof(std::uint64_t)) &&
         append_range(set, values.system_statuses, plan.batch_size, sizeof(gpuxtb_status_t)) &&
         append_range(set, values.initialized, plan.batch_size, sizeof(std::uint8_t)) &&
         append_range(set, values.residual_converged, plan.batch_size, sizeof(std::uint8_t));
}

bool transaction_ranges_are_valid(const Gfn2SccPublicationDevicePlan& plan,
                                  const Gfn2SccIterationDeviceLedger& ledger,
                                  const Gfn2SccPublicationDeviceStagedState& staged,
                                  const Gfn2SccPublicationDevicePublicState& public_state,
                                  const Gfn2SccPublicationDeviceWorkspace& workspace,
                                  std::int64_t dipoles, std::int64_t quadrupoles,
                                  std::int64_t two_orbitals, std::int64_t two_batch,
                                  std::int64_t history_elements,
                                  std::int64_t omega_elements) noexcept {
  RangeSet<48> staged_reads;
  RangeSet<64> public_writes;
  RangeSet<9> scratch_writes;
  RangeSet<5> control_reads;

  if (!append_range(&control_reads, ledger.active_mask, plan.batch_size, sizeof(std::uint8_t)) ||
      !append_range(&control_reads, ledger.pending_statuses, plan.batch_size,
                    sizeof(gpuxtb_status_t)) ||
      !append_range(&control_reads, ledger.system_failure_records, plan.batch_size,
                    sizeof(std::uint64_t)) ||
      !append_range(&control_reads, ledger.plan_failure_record, 1, sizeof(std::uint64_t)) ||
      !append_range(&control_reads, ledger.sequence_active, 1, sizeof(std::uint32_t))) {
    return false;
  }

  if (!append_wavefunction_ranges(&staged_reads, staged.wavefunction, plan, dipoles, quadrupoles,
                                  two_orbitals, two_batch) ||
      /* #96 binds staged free-energy entropy directly to occupation entropy. */
      !append_energy_ranges(&staged_reads, staged.energy, plan, true) ||
      !append_mixer_ranges(&staged_reads, staged.mixer, plan, history_elements, omega_elements) ||
      !append_range(&staged_reads, staged.next_mixed.shell_charges,
                    plan.wavefunction_layout.total_spin_shells, sizeof(double)) ||
      !append_range(&staged_reads, staged.next_mixed.atomic_dipoles, dipoles, sizeof(double)) ||
      !append_range(&staged_reads, staged.next_mixed.atomic_quadrupoles, quadrupoles,
                    sizeof(double))) {
    return false;
  }

  /* qsh/d/Q publication ranges are registered once through published. */
  if (!append_range(&public_writes, public_state.wavefunction.eigenpairs.eigenvalues,
                    plan.wavefunction_layout.total_spin_orbitals, sizeof(double)) ||
      !append_range(&public_writes, public_state.wavefunction.eigenpairs.coefficients,
                    plan.wavefunction_layout.total_spin_matrix_elements, sizeof(double)) ||
      !append_range(&public_writes, public_state.wavefunction.occupations.occupations, two_orbitals,
                    sizeof(double)) ||
      !append_range(&public_writes, public_state.wavefunction.occupations.chemical_potentials,
                    two_batch, sizeof(double)) ||
      !append_range(&public_writes, public_state.wavefunction.occupations.electron_sums, two_batch,
                    sizeof(double)) ||
      !append_range(&public_writes, public_state.wavefunction.occupations.entropies,
                    plan.batch_size, sizeof(double)) ||
      !append_range(&public_writes, public_state.wavefunction.density.density,
                    plan.wavefunction_layout.total_spin_matrix_elements, sizeof(double)) ||
      !append_range(&public_writes, public_state.wavefunction.density.energy_weighted_density,
                    plan.wavefunction_layout.total_spin_matrix_elements, sizeof(double)) ||
      !append_range(&public_writes, public_state.wavefunction.density.band_energies,
                    plan.batch_size, sizeof(double)) ||
      !append_range(&public_writes, public_state.wavefunction.density.occupation_sums,
                    plan.batch_size, sizeof(double)) ||
      !append_range(&public_writes, public_state.wavefunction.density.density_traces,
                    plan.batch_size, sizeof(double)) ||
      !append_range(&public_writes, public_state.wavefunction.density.weighted_density_traces,
                    plan.batch_size, sizeof(double)) ||
      !append_range(&public_writes, public_state.wavefunction.density.channel_band_energies,
                    plan.wavefunction_layout.total_spin_channels, sizeof(double)) ||
      !append_range(&public_writes, public_state.wavefunction.density.channel_occupation_sums,
                    plan.wavefunction_layout.total_spin_channels, sizeof(double)) ||
      !append_range(&public_writes, public_state.wavefunction.density.channel_density_traces,
                    plan.wavefunction_layout.total_spin_channels, sizeof(double)) ||
      !append_range(&public_writes,
                    public_state.wavefunction.density.channel_weighted_density_traces,
                    plan.wavefunction_layout.total_spin_channels, sizeof(double)) ||
      !append_range(&public_writes, public_state.wavefunction.population.qat,
                    plan.wavefunction_layout.total_spin_atoms, sizeof(double)) ||
      !append_range(&public_writes, public_state.published.shell_charges,
                    plan.wavefunction_layout.total_spin_shells, sizeof(double)) ||
      !append_range(&public_writes, public_state.published.atomic_dipoles, dipoles,
                    sizeof(double)) ||
      !append_range(&public_writes, public_state.published.atomic_quadrupoles, quadrupoles,
                    sizeof(double)) ||
      !append_energy_ranges(&public_writes, public_state.energy, plan) ||
      !append_mixer_ranges(&public_writes, public_state.mixer, plan, history_elements,
                           omega_elements) ||
      !append_range(&public_writes, public_state.scc.current_inputs.shell_charges,
                    plan.wavefunction_layout.total_spin_shells, sizeof(double)) ||
      !append_range(&public_writes, public_state.scc.current_inputs.atomic_dipoles, dipoles,
                    sizeof(double)) ||
      !append_range(&public_writes, public_state.scc.current_inputs.atomic_quadrupoles, quadrupoles,
                    sizeof(double)) ||
      !append_range(&public_writes, public_state.scc.free_energies, plan.batch_size,
                    sizeof(double)) ||
      !append_range(&public_writes, public_state.scc.previous_free_energies, plan.batch_size,
                    sizeof(double)) ||
      !append_range(&public_writes, public_state.scc.free_energy_changes, plan.batch_size,
                    sizeof(double)) ||
      !append_range(&public_writes, public_state.scc.residual_rms, plan.batch_size,
                    sizeof(double)) ||
      !append_range(&public_writes, public_state.scc.iterations, plan.batch_size,
                    sizeof(std::uint64_t)) ||
      !append_range(&public_writes, public_state.scc.system_statuses, plan.batch_size,
                    sizeof(gpuxtb_status_t)) ||
      !append_range(&public_writes, public_state.scc.converged, plan.batch_size,
                    sizeof(std::uint8_t))) {
    return false;
  }

  if (!append_range(&scratch_writes, workspace.mixed_atomic_charges,
                    plan.wavefunction_layout.total_spin_atoms, sizeof(double)) ||
      !append_range(&scratch_writes, workspace.previous_free_energies, plan.batch_size,
                    sizeof(double)) ||
      !append_range(&scratch_writes, workspace.free_energy_changes, plan.batch_size,
                    sizeof(double)) ||
      !append_range(&scratch_writes, workspace.next_iterations, plan.batch_size,
                    sizeof(std::uint64_t)) ||
      !append_range(&scratch_writes, workspace.next_statuses, plan.batch_size,
                    sizeof(gpuxtb_status_t)) ||
      !append_range(&scratch_writes, workspace.next_converged, plan.batch_size,
                    sizeof(std::uint8_t)) ||
      !append_range(&scratch_writes, workspace.system_errors, plan.batch_size,
                    sizeof(std::uint32_t)) ||
      !append_range(&scratch_writes, workspace.device_error, 1, sizeof(std::uint32_t)) ||
      !append_range(&scratch_writes, workspace.sequence_active, 1, sizeof(std::uint32_t))) {
    return false;
  }

  return pairwise_disjoint(control_reads) && pairwise_disjoint(staged_reads) &&
         pairwise_disjoint(public_writes) && pairwise_disjoint(scratch_writes) &&
         disjoint_sets(control_reads, staged_reads) &&
         disjoint_sets(control_reads, public_writes) &&
         disjoint_sets(control_reads, scratch_writes) &&
         disjoint_sets(staged_reads, public_writes) &&
         disjoint_sets(staged_reads, scratch_writes) &&
         disjoint_sets(public_writes, scratch_writes);
}

bool valid_bindings(const Gfn2SccPublicationDevicePlan& plan,
                    const Gfn2SccIterationDeviceActivity* activity,
                    const Gfn2SccIterationDeviceLedger* ledger,
                    const Gfn2SccPublicationDeviceStagedState* staged,
                    const Gfn2SccPublicationDevicePublicState* public_state,
                    const Gfn2SccPublicationDeviceWorkspace& workspace) noexcept {
  std::int64_t dipoles = 0;
  std::int64_t quadrupoles = 0;
  std::int64_t two_orbitals = 0;
  std::int64_t two_batch = 0;
  std::int64_t history_elements = 0;
  std::int64_t omega_elements = 0;
  if (!valid_plan(plan, &dipoles, &quadrupoles, &two_orbitals, &two_batch, &history_elements,
                  &omega_elements) ||
      !valid_workspace(workspace, plan)) {
    return false;
  }
  if (activity == nullptr && ledger == nullptr && staged == nullptr && public_state == nullptr) {
    return true;
  }
  return activity != nullptr && ledger != nullptr && staged != nullptr && public_state != nullptr &&
         valid_activity(*activity, plan) && valid_ledger(*ledger, *activity, plan) &&
         valid_staged(*staged, plan, dipoles, quadrupoles, two_orbitals, two_batch,
                      history_elements, omega_elements) &&
         valid_public(*public_state, plan, dipoles, quadrupoles, two_orbitals, two_batch,
                      history_elements, omega_elements) &&
         transaction_ranges_are_valid(plan, *ledger, *staged, *public_state, workspace, dipoles,
                                      quadrupoles, two_orbitals, two_batch, history_elements,
                                      omega_elements);
}

__device__ void record_plan_error(Gfn2SccPublicationDeviceWorkspace workspace,
                                  Gfn2SccPublicationDeviceError error) {
  atomicCAS(workspace.device_error,
            static_cast<std::uint32_t>(Gfn2SccPublicationDeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
  atomicExch(workspace.sequence_active, 0u);
}

__device__ void record_peer_error(Gfn2SccPublicationDeviceWorkspace workspace, std::int64_t system,
                                  Gfn2SccPublicationDeviceError error) {
  atomicCAS(workspace.system_errors + system,
            static_cast<std::uint32_t>(Gfn2SccPublicationDeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ bool valid_closed_range(std::int64_t begin, std::int64_t end, std::int64_t total,
                                   bool allow_empty) {
  return begin >= 0 && begin <= end && end <= total && (allow_empty || begin < end);
}

__global__ void publication_topology_preflight_kernel(Gfn2SccPublicationDevicePlan plan,
                                                      Gfn2SccIterationDeviceActivity activity,
                                                      Gfn2SccPublicationDeviceWorkspace workspace) {
  __shared__ int run;
  __shared__ int invalid_activity;
  if (threadIdx.x == 0) {
    const std::uint32_t sequence = *activity.sequence_active;
    run = sequence == 1u ? 1 : 0;
    invalid_activity = sequence > 1u ? 1 : 0;
    *workspace.sequence_active = run == 1 ? 1u : 0u;
  }
  __syncthreads();
  if (run == 0) {
    if (threadIdx.x == 0 && invalid_activity != 0) {
      record_plan_error(workspace, Gfn2SccPublicationDeviceError::kInvalidState);
    }
    return;
  }

  for (std::int64_t system = threadIdx.x; system < plan.batch_size; system += blockDim.x) {
    const std::uint8_t active = activity.active_mask[system];
    if (active > 1u) {
      atomicExch(&invalid_activity, 1);
      continue;
    }
    if (active == 0u) {
      continue;
    }

    const std::int64_t atom_begin = plan.atom_offsets[system];
    const std::int64_t atom_end = plan.atom_offsets[system + 1];
    const std::int64_t shell_begin = plan.shell_offsets[system];
    const std::int64_t shell_end = plan.shell_offsets[system + 1];
    const std::int64_t orbital_begin = plan.orbital_offsets[system];
    const std::int64_t orbital_end = plan.orbital_offsets[system + 1];
    const std::int64_t matrix_begin = plan.matrix_offsets[system];
    const std::int64_t matrix_end = plan.matrix_offsets[system + 1];
    const std::int32_t spin_channels = plan.wavefunction_layout.spin_channels[system];
    const std::int64_t spin_channel_begin = plan.wavefunction_layout.spin_channel_offsets[system];
    const std::int64_t spin_channel_end = plan.wavefunction_layout.spin_channel_offsets[system + 1];
    const std::int64_t spin_orbital_begin = plan.wavefunction_layout.spin_orbital_offsets[system];
    const std::int64_t spin_orbital_end = plan.wavefunction_layout.spin_orbital_offsets[system + 1];
    const std::int64_t spin_matrix_begin = plan.wavefunction_layout.spin_matrix_offsets[system];
    const std::int64_t spin_matrix_end = plan.wavefunction_layout.spin_matrix_offsets[system + 1];
    const std::int64_t spin_shell_begin = plan.wavefunction_layout.spin_shell_offsets[system];
    const std::int64_t spin_shell_end = plan.wavefunction_layout.spin_shell_offsets[system + 1];
    const std::int64_t spin_atom_begin = plan.wavefunction_layout.spin_atom_offsets[system];
    const std::int64_t spin_atom_end = plan.wavefunction_layout.spin_atom_offsets[system + 1];
    bool valid = valid_closed_range(atom_begin, atom_end, plan.total_atoms, false) &&
                 valid_closed_range(shell_begin, shell_end, plan.total_shells, false) &&
                 valid_closed_range(orbital_begin, orbital_end, plan.total_orbitals, false) &&
                 valid_closed_range(matrix_begin, matrix_end, plan.total_matrix_elements, false) &&
                 (spin_channels == 1 || spin_channels == 2) &&
                 valid_closed_range(spin_channel_begin, spin_channel_end,
                                    plan.wavefunction_layout.total_spin_channels, false) &&
                 valid_closed_range(spin_orbital_begin, spin_orbital_end,
                                    plan.wavefunction_layout.total_spin_orbitals, false) &&
                 valid_closed_range(spin_matrix_begin, spin_matrix_end,
                                    plan.wavefunction_layout.total_spin_matrix_elements, false) &&
                 valid_closed_range(spin_shell_begin, spin_shell_end,
                                    plan.wavefunction_layout.total_spin_shells, false) &&
                 valid_closed_range(spin_atom_begin, spin_atom_end,
                                    plan.wavefunction_layout.total_spin_atoms, false);
    if (valid) {
      const std::int64_t orbitals = orbital_end - orbital_begin;
      const std::int64_t shells = shell_end - shell_begin;
      const std::int64_t atoms = atom_end - atom_begin;
      const std::int64_t matrices = matrix_end - matrix_begin;
      valid = orbitals <= INT64_MAX / orbitals && matrices == orbitals * orbitals &&
              orbitals <= INT64_MAX / spin_channels && matrices <= INT64_MAX / spin_channels &&
              shells <= INT64_MAX / spin_channels && atoms <= INT64_MAX / spin_channels &&
              spin_channel_end - spin_channel_begin == spin_channels &&
              spin_orbital_end - spin_orbital_begin == spin_channels * orbitals &&
              spin_matrix_end - spin_matrix_begin == spin_channels * matrices &&
              spin_shell_end - spin_shell_begin == spin_channels * shells &&
              spin_atom_end - spin_atom_begin == spin_channels * atoms;
    }
    if (valid && system == 0) {
      valid = atom_begin == 0 && shell_begin == 0 && orbital_begin == 0 && matrix_begin == 0 &&
              spin_channel_begin == 0 && spin_orbital_begin == 0 && spin_matrix_begin == 0 &&
              spin_shell_begin == 0 && spin_atom_begin == 0;
    }
    if (valid && system + 1 == plan.batch_size) {
      valid = atom_end == plan.total_atoms && shell_end == plan.total_shells &&
              orbital_end == plan.total_orbitals && matrix_end == plan.total_matrix_elements &&
              spin_channel_end == plan.wavefunction_layout.total_spin_channels &&
              spin_orbital_end == plan.wavefunction_layout.total_spin_orbitals &&
              spin_matrix_end == plan.wavefunction_layout.total_spin_matrix_elements &&
              spin_shell_end == plan.wavefunction_layout.total_spin_shells &&
              spin_atom_end == plan.wavefunction_layout.total_spin_atoms;
    }
    if (valid) {
      for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
        const std::int64_t atom = plan.shell_to_atom[shell];
        if (atom < atom_begin || atom >= atom_end) {
          valid = false;
          break;
        }
      }
    }
    if (!valid) {
      record_plan_error(workspace, Gfn2SccPublicationDeviceError::kInvalidOffsets);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && invalid_activity != 0) {
    record_plan_error(workspace, Gfn2SccPublicationDeviceError::kInvalidState);
  }
}

__global__ void publication_numerical_preflight_kernel(
    Gfn2SccPublicationDevicePlan plan, Gfn2SccIterationDeviceActivity activity,
    Gfn2SccPublicationDeviceStagedState staged, Gfn2SccPublicationDevicePublicState public_state,
    Gfn2SccPublicationDeviceWorkspace workspace) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int active;
  __shared__ int valid;
  __shared__ int plan_valid;

  if (threadIdx.x == 0) {
    active = 0;
    valid = 1;
    plan_valid = 1;
    if (*activity.sequence_active == 1u && atomicAdd(workspace.sequence_active, 0u) == 1u &&
        activity.active_mask[system] == 1u) {
      active = 1;
      const gpuxtb_status_t public_status = public_state.scc.system_statuses[system];
      const std::uint8_t public_converged = public_state.scc.converged[system];
      const std::uint64_t public_iteration = public_state.scc.iterations[system];
      if (public_status != GPUXTB_STATUS_SUCCESS || public_converged != 0u ||
          public_iteration >= plan.maximum_iterations ||
          staged.mixer.system_statuses[system] != GPUXTB_STATUS_SUCCESS ||
          staged.mixer.initialized[system] != 1u) {
        record_plan_error(workspace, Gfn2SccPublicationDeviceError::kInvalidState);
        plan_valid = 0;
        active = 0;
      }
    }
  }
  __syncthreads();
  if (active == 0 || plan_valid == 0) {
    return;
  }

  const std::int64_t atom_begin = plan.atom_offsets[system];
  const std::int64_t atom_end = plan.atom_offsets[system + 1];
  const std::int64_t shell_begin = plan.shell_offsets[system];
  const std::int64_t shell_end = plan.shell_offsets[system + 1];
  const std::int64_t spin_shell_begin = plan.wavefunction_layout.spin_shell_offsets[system];
  const std::int64_t spin_shell_end = plan.wavefunction_layout.spin_shell_offsets[system + 1];
  const std::int64_t spin_atom_begin = plan.wavefunction_layout.spin_atom_offsets[system];
  const std::int64_t spin_atom_end = plan.wavefunction_layout.spin_atom_offsets[system + 1];
  const std::int64_t dipole_begin = spin_atom_begin * kDipoleComponents;
  const std::int64_t dipole_end = spin_atom_end * kDipoleComponents;
  const std::int64_t quadrupole_begin = spin_atom_begin * kQuadrupoleComponents;
  const std::int64_t quadrupole_end = spin_atom_end * kQuadrupoleComponents;

  for (std::int64_t atom = spin_atom_begin + threadIdx.x; atom < spin_atom_end;
       atom += blockDim.x) {
    workspace.mixed_atomic_charges[atom] = 0.0;
    const double raw_charge = staged.wavefunction.population.qat[atom];
    if (!isfinite(raw_charge)) {
      record_peer_error(workspace, system, Gfn2SccPublicationDeviceError::kNonfiniteRawMultipole);
      atomicExch(&valid, 0);
    }
  }

  for (std::int64_t shell = spin_shell_begin + threadIdx.x; shell < spin_shell_end;
       shell += blockDim.x) {
    const double mixed = staged.next_mixed.shell_charges[shell];
    const double raw = staged.wavefunction.population.qsh[shell];
    if (!isfinite(mixed)) {
      record_peer_error(workspace, system,
                        Gfn2SccPublicationDeviceError::kNonfiniteNextMixedMultipole);
      atomicExch(&valid, 0);
    }
    if (!isfinite(raw)) {
      record_peer_error(workspace, system, Gfn2SccPublicationDeviceError::kNonfiniteRawMultipole);
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t element = dipole_begin + threadIdx.x; element < dipole_end;
       element += blockDim.x) {
    const double mixed = staged.next_mixed.atomic_dipoles[element];
    const double raw = staged.wavefunction.population.dipole[element];
    if (!isfinite(mixed)) {
      record_peer_error(workspace, system,
                        Gfn2SccPublicationDeviceError::kNonfiniteNextMixedMultipole);
      atomicExch(&valid, 0);
    }
    if (!isfinite(raw)) {
      record_peer_error(workspace, system, Gfn2SccPublicationDeviceError::kNonfiniteRawMultipole);
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t element = quadrupole_begin + threadIdx.x; element < quadrupole_end;
       element += blockDim.x) {
    const double mixed = staged.next_mixed.atomic_quadrupoles[element];
    const double raw = staged.wavefunction.population.quadrupole[element];
    if (!isfinite(mixed)) {
      record_peer_error(workspace, system,
                        Gfn2SccPublicationDeviceError::kNonfiniteNextMixedMultipole);
      atomicExch(&valid, 0);
    }
    if (!isfinite(raw)) {
      record_peer_error(workspace, system, Gfn2SccPublicationDeviceError::kNonfiniteRawMultipole);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const std::uint64_t old_iteration = public_state.scc.iterations[system];
    const double old_energy = old_iteration == 0u ? 0.0 : public_state.scc.free_energies[system];
    const double new_energy = staged.energy.free_energy.free_energy[system];
    const double energy_change = new_energy - old_energy;
    const double residual = staged.mixer.residual_rms[system];
    if (!isfinite(old_energy)) {
      record_plan_error(workspace, Gfn2SccPublicationDeviceError::kInvalidState);
      valid = 0;
    } else if (!isfinite(new_energy)) {
      record_peer_error(workspace, system, Gfn2SccPublicationDeviceError::kNonfiniteFreeEnergy);
      valid = 0;
    } else if (!isfinite(energy_change)) {
      record_plan_error(workspace, Gfn2SccPublicationDeviceError::kNonfiniteEnergyDelta);
      valid = 0;
    } else if (!isfinite(residual)) {
      record_peer_error(workspace, system, Gfn2SccPublicationDeviceError::kNonfiniteResidual);
      valid = 0;
    } else {
      const std::uint64_t next_iteration = old_iteration + 1u;
      const bool converged =
          residual < plan.residual_rms_tolerance && fabs(energy_change) < plan.energy_tolerance;
      workspace.previous_free_energies[system] = old_energy;
      workspace.free_energy_changes[system] = energy_change;
      workspace.next_iterations[system] = next_iteration;
      workspace.next_converged[system] = converged ? 1u : 0u;
      workspace.next_statuses[system] = !converged && next_iteration >= plan.maximum_iterations
                                            ? GPUXTB_STATUS_SCC_NOT_CONVERGED
                                            : GPUXTB_STATUS_SUCCESS;
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  /* Preserve the CPU shell accumulation order for exact overflow behavior. */
  if (threadIdx.x == 0) {
    const std::int64_t physical_shells = shell_end - shell_begin;
    const std::int64_t physical_atoms = atom_end - atom_begin;
    for (std::int64_t shell = spin_shell_begin; shell < spin_shell_end; ++shell) {
      const std::int64_t local = shell - spin_shell_begin;
      const std::int64_t channel = local / physical_shells;
      const std::int64_t physical_shell = shell_begin + local % physical_shells;
      const std::int64_t physical_atom = plan.shell_to_atom[physical_shell];
      const std::int64_t atom =
          spin_atom_begin + channel * physical_atoms + (physical_atom - atom_begin);
      const double updated =
          workspace.mixed_atomic_charges[atom] + staged.next_mixed.shell_charges[shell];
      if (!isfinite(updated)) {
        record_plan_error(workspace, Gfn2SccPublicationDeviceError::kNonfiniteMixedAtomicCharge);
        break;
      }
      workspace.mixed_atomic_charges[atom] = updated;
    }
  }
}

template <typename T>
__device__ void copy_range(const T* source, T* destination, std::int64_t begin, std::int64_t end) {
  for (std::int64_t index = begin + threadIdx.x; index < end; index += blockDim.x) {
    destination[index] = source[index];
  }
}

__device__ bool peer_failure_counts_attempt(std::uint64_t failure_record) {
  /* Match iterate_scc_driver_batch_cpu: failures before the generalized
   * eigensolve do not count an SCC attempt; eigensolve and every downstream
   * numerical stage do. Raw-energy stage ids were appended to the enum, so an
   * explicit switch is required instead of relying on ordinal ranges. */
  switch (gfn2_scc_failure_stage(failure_record)) {
    case Gfn2SccStageId::kEigensolver:
    case Gfn2SccStageId::kOccupations:
    case Gfn2SccStageId::kDensity:
    case Gfn2SccStageId::kMulliken:
    case Gfn2SccStageId::kClassicalEnergy:
    case Gfn2SccStageId::kElectronicEnergy:
    case Gfn2SccStageId::kFreeEnergy:
    case Gfn2SccStageId::kMixer:
    case Gfn2SccStageId::kStatePublication:
    case Gfn2SccStageId::kES2RawEnergy:
    case Gfn2SccStageId::kES3RawEnergy:
    case Gfn2SccStageId::kAES2RawEnergy:
    case Gfn2SccStageId::kD4RawEnergy:
    case Gfn2SccStageId::kExplicitPointChargeRawEnergy:
    case Gfn2SccStageId::kPeriodicRawEnergy:
    case Gfn2SccStageId::kSpinRawEnergy:
      return true;
    default:
      return false;
  }
}

__device__ void publish_peer_failure_trace(std::int64_t system, std::uint64_t failure_record,
                                           Gfn2SccIterationDeviceLedger ledger,
                                           Gfn2SccPublicationDevicePublicState public_state) {
  const double nan = __longlong_as_double(0x7ff8000000000000ULL);
  public_state.energy.free_energy.core[system] = nan;
  public_state.energy.free_energy.es2[system] = nan;
  public_state.energy.free_energy.es3[system] = nan;
  public_state.energy.free_energy.aes2[system] = nan;
  public_state.energy.spin_energies[system] = nan;
  public_state.energy.free_energy.d4_two_body[system] = nan;
  public_state.energy.free_energy.explicit_point_charge[system] = nan;
  public_state.energy.free_energy.periodic_embedding[system] = nan;
  public_state.energy.free_energy.entropy[system] = nan;
  public_state.energy.free_energy.internal_energy[system] = nan;
  public_state.energy.free_energy.free_energy[system] = nan;
  public_state.energy.classical.classical_total[system] = nan;
  public_state.wavefunction.density.band_energies[system] = nan;

  public_state.scc.previous_free_energies[system] = nan;
  public_state.scc.free_energies[system] = nan;
  public_state.scc.free_energy_changes[system] = nan;
  public_state.scc.residual_rms[system] = nan;
  const Gfn2SccStageId failure_stage = gfn2_scc_failure_stage(failure_record);
  if (peer_failure_counts_attempt(failure_record)) {
    ++public_state.scc.iterations[system];
  }
  if (failure_stage == Gfn2SccStageId::kMixer) {
    /* The CPU staged mixer transaction changes only this diagnostic on a
     * numerical mixer failure; all history and residual bytes stay old. */
    public_state.mixer.system_statuses[system] = ledger.pending_statuses[system];
  }
  public_state.scc.system_statuses[system] = ledger.pending_statuses[system];
}

__global__ void publication_commit_kernel(Gfn2SccPublicationDevicePlan plan,
                                          Gfn2SccIterationDeviceActivity activity,
                                          Gfn2SccIterationDeviceLedger ledger,
                                          Gfn2SccPublicationDeviceStagedState staged,
                                          Gfn2SccPublicationDevicePublicState public_state,
                                          Gfn2SccPublicationDeviceWorkspace workspace) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (*activity.sequence_active != 1u || *ledger.plan_failure_record != 0u ||
      *workspace.sequence_active != 1u || *workspace.device_error != 0u) {
    return;
  }

  /* #87 disables a failed peer before commit. Publish the CPU-compatible
   * failure trace, but leave its numerical wavefunction/mixer transaction and
   * public multipoles untouched. */
  const std::uint64_t failure_record = ledger.system_failure_records[system];
  if (failure_record != 0u) {
    if (threadIdx.x == 0) {
      publish_peer_failure_trace(system, failure_record, ledger, public_state);
    }
    return;
  }
  if (activity.active_mask[system] != 1u || workspace.system_errors[system] != 0u) {
    return;
  }

  const std::int64_t physical_orbital_begin = plan.orbital_offsets[system];
  const std::int64_t physical_orbital_end = plan.orbital_offsets[system + 1];
  const std::int64_t spin_channel_begin = plan.wavefunction_layout.spin_channel_offsets[system];
  const std::int64_t spin_channel_end = plan.wavefunction_layout.spin_channel_offsets[system + 1];
  const std::int64_t orbital_begin = plan.wavefunction_layout.spin_orbital_offsets[system];
  const std::int64_t orbital_end = plan.wavefunction_layout.spin_orbital_offsets[system + 1];
  const std::int64_t matrix_begin = plan.wavefunction_layout.spin_matrix_offsets[system];
  const std::int64_t matrix_end = plan.wavefunction_layout.spin_matrix_offsets[system + 1];
  const std::int64_t shell_begin = plan.wavefunction_layout.spin_shell_offsets[system];
  const std::int64_t shell_end = plan.wavefunction_layout.spin_shell_offsets[system + 1];
  const std::int64_t spin_atom_begin = plan.wavefunction_layout.spin_atom_offsets[system];
  const std::int64_t spin_atom_end = plan.wavefunction_layout.spin_atom_offsets[system + 1];
  const std::int64_t dipole_begin = spin_atom_begin * kDipoleComponents;
  const std::int64_t dipole_end = spin_atom_end * kDipoleComponents;
  const std::int64_t quadrupole_begin = spin_atom_begin * kQuadrupoleComponents;
  const std::int64_t quadrupole_end = spin_atom_end * kQuadrupoleComponents;
  const std::int64_t occupation_begin = 2 * physical_orbital_begin;
  const std::int64_t occupation_end = 2 * physical_orbital_end;
  const std::int64_t vector_begin = shell_begin + kMultipoleAtomComponents * spin_atom_begin;
  const std::int64_t vector_end = shell_end + kMultipoleAtomComponents * spin_atom_end;
  const std::int64_t history_begin = vector_begin * plan.history_size;
  const std::int64_t history_end = vector_end * plan.history_size;
  const std::int64_t omega_begin = system * plan.history_size;
  const std::int64_t omega_end = omega_begin + plan.history_size;
  const bool converged = workspace.next_converged[system] == 1u;

  copy_range(staged.wavefunction.eigenpairs.eigenvalues,
             public_state.wavefunction.eigenpairs.eigenvalues, orbital_begin, orbital_end);
  copy_range(staged.wavefunction.eigenpairs.coefficients,
             public_state.wavefunction.eigenpairs.coefficients, matrix_begin, matrix_end);
  copy_range(staged.wavefunction.occupations.occupations,
             public_state.wavefunction.occupations.occupations, occupation_begin, occupation_end);
  copy_range(staged.wavefunction.density.density, public_state.wavefunction.density.density,
             matrix_begin, matrix_end);
  copy_range(staged.wavefunction.density.energy_weighted_density,
             public_state.wavefunction.density.energy_weighted_density, matrix_begin, matrix_end);

  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    public_state.published.shell_charges[shell] = converged
                                                      ? staged.wavefunction.population.qsh[shell]
                                                      : staged.next_mixed.shell_charges[shell];
    public_state.scc.current_inputs.shell_charges[shell] = staged.next_mixed.shell_charges[shell];
  }
  for (std::int64_t atom = spin_atom_begin + threadIdx.x; atom < spin_atom_end;
       atom += blockDim.x) {
    public_state.wavefunction.population.qat[atom] =
        converged ? staged.wavefunction.population.qat[atom] : workspace.mixed_atomic_charges[atom];
  }
  for (std::int64_t element = dipole_begin + threadIdx.x; element < dipole_end;
       element += blockDim.x) {
    public_state.published.atomic_dipoles[element] =
        converged ? staged.wavefunction.population.dipole[element]
                  : staged.next_mixed.atomic_dipoles[element];
    public_state.scc.current_inputs.atomic_dipoles[element] =
        staged.next_mixed.atomic_dipoles[element];
  }
  for (std::int64_t element = quadrupole_begin + threadIdx.x; element < quadrupole_end;
       element += blockDim.x) {
    public_state.published.atomic_quadrupoles[element] =
        converged ? staged.wavefunction.population.quadrupole[element]
                  : staged.next_mixed.atomic_quadrupoles[element];
    public_state.scc.current_inputs.atomic_quadrupoles[element] =
        staged.next_mixed.atomic_quadrupoles[element];
  }

  copy_range(staged.mixer.current_inputs, public_state.mixer.current_inputs, vector_begin,
             vector_end);
  copy_range(staged.mixer.previous_inputs, public_state.mixer.previous_inputs, vector_begin,
             vector_end);
  copy_range(staged.mixer.previous_residuals, public_state.mixer.previous_residuals, vector_begin,
             vector_end);
  copy_range(staged.mixer.df_history, public_state.mixer.df_history, history_begin, history_end);
  copy_range(staged.mixer.u_history, public_state.mixer.u_history, history_begin, history_end);
  copy_range(staged.mixer.omega, public_state.mixer.omega, omega_begin, omega_end);
  copy_range(staged.wavefunction.density.channel_band_energies,
             public_state.wavefunction.density.channel_band_energies, spin_channel_begin,
             spin_channel_end);
  copy_range(staged.wavefunction.density.channel_occupation_sums,
             public_state.wavefunction.density.channel_occupation_sums, spin_channel_begin,
             spin_channel_end);
  copy_range(staged.wavefunction.density.channel_density_traces,
             public_state.wavefunction.density.channel_density_traces, spin_channel_begin,
             spin_channel_end);
  copy_range(staged.wavefunction.density.channel_weighted_density_traces,
             public_state.wavefunction.density.channel_weighted_density_traces, spin_channel_begin,
             spin_channel_end);
  __syncthreads();

  if (threadIdx.x == 0) {
    const std::int64_t pair = 2 * system;
    public_state.wavefunction.occupations.chemical_potentials[pair] =
        staged.wavefunction.occupations.chemical_potentials[pair];
    public_state.wavefunction.occupations.chemical_potentials[pair + 1] =
        staged.wavefunction.occupations.chemical_potentials[pair + 1];
    public_state.wavefunction.occupations.electron_sums[pair] =
        staged.wavefunction.occupations.electron_sums[pair];
    public_state.wavefunction.occupations.electron_sums[pair + 1] =
        staged.wavefunction.occupations.electron_sums[pair + 1];
    public_state.wavefunction.occupations.entropies[system] =
        staged.wavefunction.occupations.entropies[system];

    public_state.wavefunction.density.band_energies[system] =
        staged.wavefunction.density.band_energies[system];
    public_state.wavefunction.density.occupation_sums[system] =
        staged.wavefunction.density.occupation_sums[system];
    public_state.wavefunction.density.density_traces[system] =
        staged.wavefunction.density.density_traces[system];
    public_state.wavefunction.density.weighted_density_traces[system] =
        staged.wavefunction.density.weighted_density_traces[system];

    public_state.energy.free_energy.core[system] = staged.energy.free_energy.core[system];
    public_state.energy.free_energy.es2[system] = staged.energy.free_energy.es2[system];
    public_state.energy.free_energy.es3[system] = staged.energy.free_energy.es3[system];
    public_state.energy.free_energy.aes2[system] = staged.energy.free_energy.aes2[system];
    public_state.energy.spin_energies[system] = staged.energy.spin_energies[system];
    public_state.energy.free_energy.d4_two_body[system] =
        staged.energy.free_energy.d4_two_body[system];
    public_state.energy.free_energy.explicit_point_charge[system] =
        staged.energy.free_energy.explicit_point_charge[system];
    public_state.energy.free_energy.periodic_embedding[system] =
        staged.energy.free_energy.periodic_embedding[system];
    public_state.energy.free_energy.entropy[system] = staged.energy.free_energy.entropy[system];
    public_state.energy.free_energy.internal_energy[system] =
        staged.energy.free_energy.internal_energy[system];
    public_state.energy.free_energy.free_energy[system] =
        staged.energy.free_energy.free_energy[system];
    public_state.energy.classical.classical_total[system] =
        staged.energy.classical.classical_total[system];

    public_state.mixer.residual_rms[system] = staged.mixer.residual_rms[system];
    public_state.mixer.residual_maximum[system] = staged.mixer.residual_maximum[system];
    public_state.mixer.iterations[system] = staged.mixer.iterations[system];
    public_state.mixer.restart_counts[system] = staged.mixer.restart_counts[system];
    public_state.mixer.system_statuses[system] = staged.mixer.system_statuses[system];
    public_state.mixer.initialized[system] = staged.mixer.initialized[system];
    public_state.mixer.residual_converged[system] = staged.mixer.residual_converged[system];

    public_state.scc.previous_free_energies[system] = workspace.previous_free_energies[system];
    public_state.scc.free_energies[system] = staged.energy.free_energy.free_energy[system];
    public_state.scc.free_energy_changes[system] = workspace.free_energy_changes[system];
    public_state.scc.residual_rms[system] = staged.mixer.residual_rms[system];
    public_state.scc.iterations[system] = workspace.next_iterations[system];
    public_state.scc.converged[system] = workspace.next_converged[system];
    public_state.scc.system_statuses[system] = workspace.next_statuses[system];
  }
}

}  // namespace

cudaError_t reset_gfn2_scc_publication_errors_cuda(
    const Gfn2SccPublicationDevicePlan& plan, const Gfn2SccPublicationDeviceWorkspace& workspace,
    cudaStream_t stream) noexcept {
  if (!valid_bindings(plan, nullptr, nullptr, nullptr, nullptr, workspace)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status =
      cudaMemsetAsync(workspace.system_errors, 0,
                      static_cast<std::size_t>(plan.batch_size) * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaMemsetAsync(workspace.device_error, 0, sizeof(std::uint32_t), stream);
  return status == cudaSuccess
             ? cudaMemsetAsync(workspace.sequence_active, 0, sizeof(std::uint32_t), stream)
             : status;
}

cudaError_t preflight_gfn2_scc_publication_cuda(
    const Gfn2SccPublicationDevicePlan& plan, const Gfn2SccIterationDeviceActivity& activity,
    const Gfn2SccIterationDeviceLedger& ledger, const Gfn2SccPublicationDeviceStagedState& staged,
    const Gfn2SccPublicationDevicePublicState& public_state,
    const Gfn2SccPublicationDeviceWorkspace& workspace, cudaStream_t stream) noexcept {
  if (!valid_bindings(plan, &activity, &ledger, &staged, &public_state, workspace)) {
    return cudaErrorInvalidValue;
  }
  publication_topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(plan, activity,
                                                                            workspace);
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  publication_numerical_preflight_kernel<<<static_cast<unsigned int>(plan.batch_size),
                                           kThreadsPerBlock, 0, stream>>>(plan, activity, staged,
                                                                          public_state, workspace);
  return cudaPeekAtLastError();
}

cudaError_t commit_gfn2_scc_publication_cuda(
    const Gfn2SccPublicationDevicePlan& plan, const Gfn2SccIterationDeviceActivity& activity,
    const Gfn2SccIterationDeviceLedger& ledger, const Gfn2SccPublicationDeviceStagedState& staged,
    const Gfn2SccPublicationDevicePublicState& public_state,
    const Gfn2SccPublicationDeviceWorkspace& workspace, cudaStream_t stream) noexcept {
  if (!valid_bindings(plan, &activity, &ledger, &staged, &public_state, workspace)) {
    return cudaErrorInvalidValue;
  }
  publication_commit_kernel<<<static_cast<unsigned int>(plan.batch_size), kThreadsPerBlock, 0,
                              stream>>>(plan, activity, ledger, staged, public_state, workspace);
  return cudaPeekAtLastError();
}

}  // namespace gpuxtb::detail::cuda
