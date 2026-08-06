#include "model/gfn2/scc_driver.hpp"
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace gpuxtb::detail::gfn2 {

struct SccDriverPlanData {
  WavefunctionLayout wavefunction;
  MullikenPlan mulliken;
  ES2Plan es2;
  ES3Plan es3;
  AES2Plan aes2;
  SpinPolarizationPlan spin;
  D4Plan d4;
  EigensolverPlan eigensolver;
  SccMixerPlan mixer;
  PeriodicEmbeddingPlan periodic_embedding;

  std::uint64_t maximum_iterations = 0u;
  double electronic_temperature = 0.0;
  double energy_tolerance = 0.0;

  std::size_t state_size_bytes = 0u;
  std::size_t state_free_energy_offset = 0u;
  std::size_t state_previous_free_energy_offset = 0u;
  std::size_t state_free_energy_change_offset = 0u;
  std::size_t state_entropy_offset = 0u;
  std::size_t state_band_energy_offset = 0u;
  std::size_t state_core_energy_offset = 0u;
  std::size_t state_es2_energy_offset = 0u;
  std::size_t state_es3_energy_offset = 0u;
  std::size_t state_aes2_energy_offset = 0u;
  std::size_t state_spin_energy_offset = 0u;
  std::size_t state_d4_energy_offset = 0u;
  std::size_t state_explicit_pc_energy_offset = 0u;
  std::size_t state_periodic_energy_offset = 0u;
  std::size_t state_internal_energy_offset = 0u;
  std::size_t state_iteration_offset = 0u;
  std::size_t state_status_offset = 0u;
  std::size_t state_initialized_offset = 0u;
  std::size_t state_converged_offset = 0u;

  std::size_t workspace_size_bytes = 0u;
  std::size_t staged_wavefunction_offset = 0u;
  std::size_t hamiltonian_offset = 0u;
  std::size_t shell_charge_offset = 0u;
  std::size_t atomic_charge_offset = 0u;
  std::size_t atomic_dipole_offset = 0u;
  std::size_t atomic_quadrupole_offset = 0u;
  std::size_t component_shell_offset = 0u;
  std::size_t component_atomic_offset = 0u;
  std::size_t component_dipole_offset = 0u;
  std::size_t component_quadrupole_offset = 0u;
  std::size_t atom_potential_offset = 0u;
  std::size_t shell_potential_offset = 0u;
  std::size_t dipole_potential_offset = 0u;
  std::size_t quadrupole_potential_offset = 0u;
  std::size_t spin_shell_potential_offset = 0u;
  std::size_t raw_qsh_offset = 0u;
  std::size_t raw_qat_offset = 0u;
  std::size_t raw_dipole_offset = 0u;
  std::size_t raw_quadrupole_offset = 0u;
  std::size_t core_energy_offset = 0u;
  std::size_t es2_energy_offset = 0u;
  std::size_t es3_energy_offset = 0u;
  std::size_t aes2_energy_offset = 0u;
  std::size_t spin_energy_offset = 0u;
  std::size_t explicit_pc_energy_offset = 0u;
  std::size_t internal_energy_offset = 0u;
  std::size_t free_energy_offset = 0u;
  std::size_t d4_potential_offset = 0u;
  std::size_t d4_energy_offset = 0u;
  std::size_t d4_scratch_offset = 0u;
  std::size_t periodic_potential_offset = 0u;
  std::size_t periodic_energy_offset = 0u;
  std::size_t periodic_status_offset = 0u;
  std::size_t periodic_scratch_offset = 0u;
  std::size_t es2_shell_scratch_offset = 0u;
  std::size_t aes2_potential_scratch_offset = 0u;
  std::size_t mulliken_scratch_offset = 0u;
  std::size_t eigensolver_scratch_offset = 0u;
  std::size_t staged_mixer_state_offset = 0u;
  std::size_t mixer_scratch_offset = 0u;
  std::size_t thermodynamic_status_offset = 0u;
  std::size_t chemical_potential_offset = 0u;
  std::size_t thermodynamic_entropy_offset = 0u;
  std::size_t thermodynamic_band_energy_offset = 0u;
  std::size_t thermodynamic_free_energy_offset = 0u;
  std::size_t active_system_offset = 0u;
};

namespace {

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool checked_add_size(std::size_t first, std::size_t second, std::size_t& result) {
  if (first > std::numeric_limits<std::size_t>::max() - second) {
    return false;
  }
  result = first + second;
  return true;
}

bool checked_multiply_size(std::size_t first, std::size_t second, std::size_t& result) {
  if (first != 0u && second > std::numeric_limits<std::size_t>::max() / first) {
    return false;
  }
  result = first * second;
  return true;
}

bool align_up(std::size_t value, std::size_t alignment, std::size_t& result) {
  if (alignment == 0u || (alignment & (alignment - 1u)) != 0u) {
    return false;
  }
  const std::size_t mask = alignment - 1u;
  if (value > std::numeric_limits<std::size_t>::max() - mask) {
    return false;
  }
  result = (value + mask) & ~mask;
  return true;
}

bool append_segment(std::size_t bytes, std::size_t alignment, std::size_t& cursor,
                    std::size_t& offset) {
  if (!align_up(cursor, alignment, offset)) {
    return false;
  }
  return checked_add_size(offset, bytes, cursor);
}

bool bytes_for(std::int64_t elements, std::size_t element_size, std::size_t& bytes) {
  return elements >= 0 &&
         static_cast<std::uint64_t>(elements) <=
             static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) &&
         checked_multiply_size(static_cast<std::size_t>(elements), element_size, bytes);
}

bool make_range(const void* pointer, std::size_t bytes, AddressRange& range) {
  if (bytes != 0u && pointer == nullptr) {
    return false;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  range = {begin, begin + bytes};
  return true;
}

bool ranges_overlap(const AddressRange& first, const AddressRange& second) {
  return first.begin < second.end && second.begin < first.end;
}

bool range_contains(const AddressRange& outer, const AddressRange& inner) {
  return outer.begin <= inner.begin && inner.end <= outer.end;
}

bool aligned(const void* pointer, std::size_t alignment) {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

template <typename T>
bool overlaps_vector(const AddressRange& range, const std::vector<T>& values) {
  AddressRange storage;
  std::size_t bytes = 0u;
  return checked_multiply_size(values.capacity(), sizeof(T), bytes) &&
         make_range(values.data(), bytes, storage) && ranges_overlap(range, storage);
}

bool overlaps_plan_storage(const SccDriverPlanData& data, const AddressRange& range) {
  AddressRange descriptor;
  if (!make_range(&data, sizeof(data), descriptor) || ranges_overlap(range, descriptor)) {
    return true;
  }
  const WavefunctionLayout& wavefunction = data.wavefunction;
  const std::array<const std::vector<std::int64_t>*, 19> integer_vectors{{
      &wavefunction.atom_offsets,
      &wavefunction.batch_shell_offsets,
      &wavefunction.batch_orbital_offsets,
      &wavefunction.coefficients.system_offsets,
      &wavefunction.eigenvalues.system_offsets,
      &wavefunction.occupations.system_offsets,
      &wavefunction.density.system_offsets,
      &wavefunction.qsh.system_offsets,
      &wavefunction.qat.system_offsets,
      &wavefunction.dipole.system_offsets,
      &wavefunction.quadrupole.system_offsets,
      &wavefunction.energy_weighted_density.system_offsets,
      &data.es3.batch_shell_offsets,
      &data.mixer.vector_offsets(),
      &data.spin.atom_offsets,
      &data.spin.batch_shell_offsets,
      &data.spin.atom_shell_offsets,
      &data.spin.shell_population_offsets,
      &data.spin.coupling_offsets,
  }};
  for (const std::vector<std::int64_t>* values : integer_vectors) {
    if (overlaps_vector(range, *values)) {
      return true;
    }
  }
  const std::array<const std::vector<std::int32_t>*, 4> int32_vectors{
      {&wavefunction.atomic_numbers, &wavefunction.unpaired_electrons, &wavefunction.spin_channels,
       &data.spin.spin_channels}};
  for (const std::vector<std::int32_t>* values : int32_vectors) {
    if (overlaps_vector(range, *values)) {
      return true;
    }
  }
  const std::array<const std::vector<double>*, 9> double_vectors{{
      &wavefunction.molecular_charges,
      &wavefunction.reference_atom_occupations,
      &wavefunction.reference_shell_occupations,
      &wavefunction.electron_counts,
      &wavefunction.alpha_electron_counts,
      &wavefunction.beta_electron_counts,
      &data.es3.shell_gamma3,
      &data.mulliken.reference_shell_occupations(),
      &data.spin.coupling_matrices,
  }};
  for (const std::vector<double>* values : double_vectors) {
    if (overlaps_vector(range, *values)) {
      return true;
    }
  }
  const std::size_t bytes = static_cast<std::size_t>(range.end - range.begin);
  const void* pointer = reinterpret_cast<const void*>(range.begin);
  return data.mulliken.overlaps_storage(pointer, bytes) ||
         data.es2.overlaps_storage(pointer, bytes) || data.aes2.overlaps_storage(pointer, bytes) ||
         (data.d4.sealed() && data.d4.overlaps_storage(pointer, bytes)) ||
         data.eigensolver.overlaps_storage(pointer, bytes) ||
         data.mixer.overlaps_storage(pointer, bytes) ||
         (data.periodic_embedding.sealed() &&
          data.periodic_embedding.overlaps_storage(pointer, bytes));
}

template <std::size_t N>
bool pairwise_disjoint(const std::array<AddressRange, N>& ranges) {
  for (std::size_t first = 0u; first < N; ++first) {
    for (std::size_t second = first + 1u; second < N; ++second) {
      if (ranges_overlap(ranges[first], ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

template <std::size_t N, std::size_t M>
bool disjoint_from_controls(const SccDriverPlanData& data,
                            const std::array<AddressRange, N>& numerical,
                            const std::array<AddressRange, M>& controls) {
  for (const AddressRange& range : numerical) {
    if (overlaps_plan_storage(data, range)) {
      return false;
    }
    for (const AddressRange& control : controls) {
      if (ranges_overlap(range, control)) {
        return false;
      }
    }
  }
  return true;
}

template <typename T>
T* offset_pointer(void* base, std::size_t offset) {
  return reinterpret_cast<T*>(static_cast<unsigned char*>(base) + offset);
}

template <typename T>
const T* offset_pointer(const void* base, std::size_t offset) {
  return reinterpret_cast<const T*>(static_cast<const unsigned char*>(base) + offset);
}

gpuxtb_status_t validate_plan(const SccDriverPlan& plan, std::string& error) {
  if (!plan.sealed() || plan.identity() == nullptr || plan.batch_size() <= 0 ||
      plan.maximum_iterations() == 0u || !std::isfinite(plan.electronic_temperature()) ||
      plan.electronic_temperature() < 0.0 || !std::isfinite(plan.energy_tolerance()) ||
      !(plan.energy_tolerance() > 0.0) || plan.state_size_bytes() == 0u ||
      plan.workspace_size_bytes() == 0u) {
    error = "SCC driver plan is not sealed or has invalid metadata";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return GPUXTB_STATUS_SUCCESS;
}

bool same_field_layout(const WavefunctionFieldLayout& first,
                       const WavefunctionFieldLayout& second) {
  return first.offset_bytes == second.offset_bytes && first.size_bytes == second.size_bytes &&
         first.element_count == second.element_count &&
         first.system_offsets == second.system_offsets;
}

bool same_wavefunction_layout(const WavefunctionLayout& first, const WavefunctionLayout& second) {
  return first.batch_size == second.batch_size && first.total_atoms == second.total_atoms &&
         first.total_shells == second.total_shells &&
         first.total_orbitals == second.total_orbitals &&
         first.workspace_size_bytes == second.workspace_size_bytes &&
         first.atom_offsets == second.atom_offsets &&
         first.batch_shell_offsets == second.batch_shell_offsets &&
         first.batch_orbital_offsets == second.batch_orbital_offsets &&
         first.atomic_numbers == second.atomic_numbers &&
         first.molecular_charges == second.molecular_charges &&
         first.unpaired_electrons == second.unpaired_electrons &&
         first.spin_channels == second.spin_channels &&
         first.reference_atom_occupations == second.reference_atom_occupations &&
         first.reference_shell_occupations == second.reference_shell_occupations &&
         first.electron_counts == second.electron_counts &&
         first.alpha_electron_counts == second.alpha_electron_counts &&
         first.beta_electron_counts == second.beta_electron_counts &&
         same_field_layout(first.coefficients, second.coefficients) &&
         same_field_layout(first.eigenvalues, second.eigenvalues) &&
         same_field_layout(first.occupations, second.occupations) &&
         same_field_layout(first.density, second.density) &&
         same_field_layout(first.qsh, second.qsh) && same_field_layout(first.qat, second.qat) &&
         same_field_layout(first.dipole, second.dipole) &&
         same_field_layout(first.quadrupole, second.quadrupole) &&
         same_field_layout(first.energy_weighted_density, second.energy_weighted_density);
}

bool same_mulliken_plan(const MullikenPlan& first, const MullikenPlan& second) {
  return first.batch_size() == second.batch_size() && first.total_atoms() == second.total_atoms() &&
         first.total_shells() == second.total_shells() &&
         first.total_orbitals() == second.total_orbitals() &&
         first.matrix_elements() == second.matrix_elements() &&
         first.density_elements() == second.density_elements() &&
         first.shell_population_elements() == second.shell_population_elements() &&
         first.atom_population_elements() == second.atom_population_elements() &&
         first.dipole_population_elements() == second.dipole_population_elements() &&
         first.quadrupole_population_elements() == second.quadrupole_population_elements() &&
         first.population_scratch_elements() == second.population_scratch_elements() &&
         first.hamiltonian_scratch_elements() == second.hamiltonian_scratch_elements() &&
         first.atom_offsets() == second.atom_offsets() &&
         first.batch_shell_offsets() == second.batch_shell_offsets() &&
         first.batch_orbital_offsets() == second.batch_orbital_offsets() &&
         first.matrix_offsets() == second.matrix_offsets() &&
         first.shell_orbital_offsets() == second.shell_orbital_offsets() &&
         first.shell_to_atom() == second.shell_to_atom() &&
         first.orbital_to_shell() == second.orbital_to_shell() &&
         first.orbital_to_atom() == second.orbital_to_atom() &&
         first.spin_channels() == second.spin_channels() &&
         first.reference_shell_occupations() == second.reference_shell_occupations();
}

bool same_es2_plan(const ES2Plan& first, const ES2Plan& second) {
  return first.batch_size() == second.batch_size() && first.total_atoms() == second.total_atoms() &&
         first.total_shells() == second.total_shells() &&
         first.total_matrix_elements() == second.total_matrix_elements() &&
         first.atom_offsets() == second.atom_offsets() &&
         first.batch_shell_offsets() == second.batch_shell_offsets() &&
         first.atom_shell_offsets() == second.atom_shell_offsets() &&
         first.matrix_offsets() == second.matrix_offsets() &&
         first.shell_to_atom() == second.shell_to_atom() &&
         first.shell_hardness() == second.shell_hardness();
}

bool same_es3_plan(const ES3Plan& first, const ES3Plan& second) {
  return first.batch_size == second.batch_size && first.total_shells == second.total_shells &&
         first.batch_shell_offsets == second.batch_shell_offsets &&
         first.shell_gamma3 == second.shell_gamma3;
}

bool same_aes2_plan(const AES2Plan& first, const AES2Plan& second) {
  return first.batch_size() == second.batch_size() && first.total_atoms() == second.total_atoms() &&
         first.total_pairs() == second.total_pairs() &&
         first.pair_data_elements() == second.pair_data_elements() &&
         first.potential_scratch_elements() == second.potential_scratch_elements() &&
         first.gradient_scratch_elements() == second.gradient_scratch_elements() &&
         first.coordination_scratch_elements() == second.coordination_scratch_elements() &&
         first.atom_offsets() == second.atom_offsets() &&
         first.pair_offsets() == second.pair_offsets() &&
         first.dipole_kernel() == second.dipole_kernel() &&
         first.quadrupole_kernel() == second.quadrupole_kernel() &&
         first.multipole_radius() == second.multipole_radius() &&
         first.multipole_valence_cn() == second.multipole_valence_cn();
}

bool same_eigensolver_plan(const EigensolverPlan& first, const EigensolverPlan& second) {
  return first.batch_size() == second.batch_size() &&
         first.total_matrix_elements() == second.total_matrix_elements() &&
         first.maximum_orbitals() == second.maximum_orbitals() &&
         first.minimum_overlap_rcond() == second.minimum_overlap_rcond() &&
         first.overlap_cache_size_bytes() == second.overlap_cache_size_bytes() &&
         first.worker_workspace_size_bytes() == second.worker_workspace_size_bytes() &&
         first.workspace_size_bytes() == second.workspace_size_bytes() &&
         first.matrix_offsets() == second.matrix_offsets() &&
         first.orbital_offsets() == second.orbital_offsets() &&
         first.spin_channels() == second.spin_channels() &&
         first.alpha_electron_counts() == second.alpha_electron_counts() &&
         first.beta_electron_counts() == second.beta_electron_counts();
}

bool same_mixer_plan(const SccMixerPlan& first, const SccMixerPlan& second) {
  return first.batch_size() == second.batch_size() &&
         first.history_size() == second.history_size() &&
         first.total_vector_elements() == second.total_vector_elements() &&
         first.maximum_vector_elements() == second.maximum_vector_elements() &&
         first.damping() == second.damping() && first.rms_tolerance() == second.rms_tolerance() &&
         first.maximum_tolerance() == second.maximum_tolerance() &&
         first.state_size_bytes() == second.state_size_bytes() &&
         first.workspace_size_bytes() == second.workspace_size_bytes() &&
         first.vector_offsets() == second.vector_offsets();
}

gpuxtb_status_t validate_wavefunction(const SccDriverPlanData& data,
                                      const WavefunctionView& wavefunction, std::string& error) {
  WavefunctionSystemView ignored;
  return make_wavefunction_system_view(data.wavefunction, wavefunction, 0, ignored, error);
}

bool exact_state_binding(const SccDriverPlanData& data, const SccDriverState& state) {
  return state.workspace_base != nullptr && state.workspace_size_bytes >= data.state_size_bytes &&
         state.plan_identity == &data &&
         state.free_energies ==
             offset_pointer<double>(state.workspace_base, data.state_free_energy_offset) &&
         state.previous_free_energies ==
             offset_pointer<double>(state.workspace_base, data.state_previous_free_energy_offset) &&
         state.free_energy_changes ==
             offset_pointer<double>(state.workspace_base, data.state_free_energy_change_offset) &&
         state.entropies ==
             offset_pointer<double>(state.workspace_base, data.state_entropy_offset) &&
         state.band_energies ==
             offset_pointer<double>(state.workspace_base, data.state_band_energy_offset) &&
         state.core_energies ==
             offset_pointer<double>(state.workspace_base, data.state_core_energy_offset) &&
         state.es2_energies ==
             offset_pointer<double>(state.workspace_base, data.state_es2_energy_offset) &&
         state.es3_energies ==
             offset_pointer<double>(state.workspace_base, data.state_es3_energy_offset) &&
         state.aes2_energies ==
             offset_pointer<double>(state.workspace_base, data.state_aes2_energy_offset) &&
         state.spin_energies ==
             offset_pointer<double>(state.workspace_base, data.state_spin_energy_offset) &&
         ((!data.d4.sealed() && state.d4_two_body_energies == nullptr) ||
          (data.d4.sealed() &&
           state.d4_two_body_energies ==
               offset_pointer<double>(state.workspace_base, data.state_d4_energy_offset))) &&
         state.explicit_point_charge_energies ==
             offset_pointer<double>(state.workspace_base, data.state_explicit_pc_energy_offset) &&
         ((!data.periodic_embedding.sealed() && state.periodic_embedding_energies == nullptr) ||
          (data.periodic_embedding.sealed() &&
           state.periodic_embedding_energies ==
               offset_pointer<double>(state.workspace_base, data.state_periodic_energy_offset))) &&
         state.internal_energies ==
             offset_pointer<double>(state.workspace_base, data.state_internal_energy_offset) &&
         state.iterations ==
             offset_pointer<std::uint64_t>(state.workspace_base, data.state_iteration_offset) &&
         state.system_statuses ==
             offset_pointer<gpuxtb_status_t>(state.workspace_base, data.state_status_offset) &&
         state.initialized ==
             offset_pointer<std::uint8_t>(state.workspace_base, data.state_initialized_offset) &&
         state.converged ==
             offset_pointer<std::uint8_t>(state.workspace_base, data.state_converged_offset);
}

bool same_d4_workspace_binding(const D4Workspace& first, const D4Workspace& second) {
  return first.workspace_base == second.workspace_base &&
         first.workspace_size_bytes == second.workspace_size_bytes &&
         first.pair_scratch == second.pair_scratch && first.pair_elements == second.pair_elements &&
         first.coordination_scratch == second.coordination_scratch &&
         first.coordination_elements == second.coordination_elements &&
         first.weights == second.weights &&
         first.weight_cn_derivatives == second.weight_cn_derivatives &&
         first.weight_charge_derivatives == second.weight_charge_derivatives &&
         first.weight_elements == second.weight_elements &&
         first.atom_scratch == second.atom_scratch &&
         first.coordination_adjoints == second.coordination_adjoints &&
         first.atom_elements == second.atom_elements &&
         first.batch_scratch == second.batch_scratch &&
         first.batch_elements == second.batch_elements &&
         first.gradient_scratch == second.gradient_scratch &&
         first.gradient_elements == second.gradient_elements &&
         first.plan_identity == second.plan_identity;
}

bool exact_workspace_binding(const SccDriverPlanData& data, const SccDriverWorkspace& workspace) {
  return workspace.workspace_base != nullptr &&
         workspace.workspace_size_bytes >= data.workspace_size_bytes &&
         workspace.plan_identity == &data &&
         workspace.staged_wavefunction.workspace_base ==
             offset_pointer<void>(workspace.workspace_base, data.staged_wavefunction_offset) &&
         workspace.hamiltonian ==
             offset_pointer<double>(workspace.workspace_base, data.hamiltonian_offset) &&
         workspace.shell_charges ==
             offset_pointer<double>(workspace.workspace_base, data.shell_charge_offset) &&
         workspace.atomic_charges ==
             offset_pointer<double>(workspace.workspace_base, data.atomic_charge_offset) &&
         workspace.atomic_dipoles ==
             offset_pointer<double>(workspace.workspace_base, data.atomic_dipole_offset) &&
         workspace.atomic_quadrupoles ==
             offset_pointer<double>(workspace.workspace_base, data.atomic_quadrupole_offset) &&
         workspace.component_shell_potential ==
             offset_pointer<double>(workspace.workspace_base, data.component_shell_offset) &&
         workspace.component_atomic_potential ==
             offset_pointer<double>(workspace.workspace_base, data.component_atomic_offset) &&
         workspace.component_dipole_potential ==
             offset_pointer<double>(workspace.workspace_base, data.component_dipole_offset) &&
         workspace.component_quadrupole_potential ==
             offset_pointer<double>(workspace.workspace_base, data.component_quadrupole_offset) &&
         workspace.atomic_potentials ==
             offset_pointer<double>(workspace.workspace_base, data.atom_potential_offset) &&
         workspace.shell_potentials ==
             offset_pointer<double>(workspace.workspace_base, data.shell_potential_offset) &&
         workspace.dipole_potentials ==
             offset_pointer<double>(workspace.workspace_base, data.dipole_potential_offset) &&
         workspace.quadrupole_potentials ==
             offset_pointer<double>(workspace.workspace_base, data.quadrupole_potential_offset) &&
         workspace.spin_shell_potentials ==
             offset_pointer<double>(workspace.workspace_base, data.spin_shell_potential_offset) &&
         workspace.raw_qsh ==
             offset_pointer<double>(workspace.workspace_base, data.raw_qsh_offset) &&
         workspace.raw_qat ==
             offset_pointer<double>(workspace.workspace_base, data.raw_qat_offset) &&
         workspace.raw_dipoles ==
             offset_pointer<double>(workspace.workspace_base, data.raw_dipole_offset) &&
         workspace.raw_quadrupoles ==
             offset_pointer<double>(workspace.workspace_base, data.raw_quadrupole_offset) &&
         workspace.core_energies ==
             offset_pointer<double>(workspace.workspace_base, data.core_energy_offset) &&
         workspace.es2_energies ==
             offset_pointer<double>(workspace.workspace_base, data.es2_energy_offset) &&
         workspace.es3_energies ==
             offset_pointer<double>(workspace.workspace_base, data.es3_energy_offset) &&
         workspace.aes2_energies ==
             offset_pointer<double>(workspace.workspace_base, data.aes2_energy_offset) &&
         workspace.spin_energies ==
             offset_pointer<double>(workspace.workspace_base, data.spin_energy_offset) &&
         workspace.explicit_point_charge_energies ==
             offset_pointer<double>(workspace.workspace_base, data.explicit_pc_energy_offset) &&
         workspace.internal_energies ==
             offset_pointer<double>(workspace.workspace_base, data.internal_energy_offset) &&
         workspace.free_energies ==
             offset_pointer<double>(workspace.workspace_base, data.free_energy_offset) &&
         ((!data.d4.sealed() && workspace.d4_atomic_potentials == nullptr &&
           workspace.d4_two_body_energies == nullptr &&
           workspace.d4_workspace.workspace_base == nullptr &&
           workspace.d4_workspace.workspace_size_bytes == 0u &&
           workspace.d4_workspace.plan_identity == nullptr) ||
          (data.d4.sealed() &&
           workspace.d4_atomic_potentials ==
               offset_pointer<double>(workspace.workspace_base, data.d4_potential_offset) &&
           workspace.d4_two_body_energies ==
               offset_pointer<double>(workspace.workspace_base, data.d4_energy_offset) &&
           workspace.d4_workspace.workspace_base ==
               offset_pointer<void>(workspace.workspace_base, data.d4_scratch_offset) &&
           workspace.d4_workspace.workspace_size_bytes >= data.d4.workspace_size_bytes() &&
           workspace.d4_workspace.plan_identity == data.d4.identity())) &&
         workspace.active_systems ==
             offset_pointer<std::uint8_t>(workspace.workspace_base, data.active_system_offset) &&
         workspace.es2_workspace.shell_scratch ==
             offset_pointer<double>(workspace.workspace_base, data.es2_shell_scratch_offset) &&
         workspace.es2_workspace.shell_elements == data.wavefunction.total_shells &&
         workspace.aes2_workspace.potential_scratch ==
             offset_pointer<double>(workspace.workspace_base, data.aes2_potential_scratch_offset) &&
         workspace.aes2_workspace.potential_elements == data.aes2.potential_scratch_elements() &&
         workspace.aes2_workspace.batch_scratch ==
             offset_pointer<double>(workspace.workspace_base, data.aes2_energy_offset) &&
         workspace.aes2_workspace.batch_elements == data.wavefunction.batch_size &&
         workspace.mulliken_workspace.scratch ==
             offset_pointer<double>(workspace.workspace_base, data.mulliken_scratch_offset) &&
         workspace.mulliken_workspace.elements ==
             std::max(data.mulliken.population_scratch_elements(),
                      data.mulliken.hamiltonian_scratch_elements()) &&
         ((!data.periodic_embedding.sealed() && workspace.periodic_atomic_potentials == nullptr &&
           workspace.periodic_embedding_energies == nullptr &&
           workspace.periodic_system_statuses == nullptr &&
           workspace.periodic_embedding_workspace.potential_scratch == nullptr &&
           workspace.periodic_embedding_workspace.potential_elements == 0 &&
           workspace.periodic_embedding_workspace.plan_identity == nullptr) ||
          (data.periodic_embedding.sealed() &&
           workspace.periodic_atomic_potentials ==
               offset_pointer<double>(workspace.workspace_base, data.periodic_potential_offset) &&
           workspace.periodic_embedding_energies ==
               offset_pointer<double>(workspace.workspace_base, data.periodic_energy_offset) &&
           workspace.periodic_system_statuses ==
               offset_pointer<gpuxtb_status_t>(workspace.workspace_base,
                                               data.periodic_status_offset) &&
           workspace.periodic_embedding_workspace.potential_scratch ==
               offset_pointer<double>(workspace.workspace_base, data.periodic_scratch_offset) &&
           workspace.periodic_embedding_workspace.potential_elements ==
               data.periodic_embedding.maximum_atoms() &&
           workspace.periodic_embedding_workspace.plan_identity ==
               data.periodic_embedding.identity())) &&
         workspace.eigensolver_workspace.workspace_base ==
             offset_pointer<void>(workspace.workspace_base, data.eigensolver_scratch_offset) &&
         workspace.staged_mixer_state.workspace_base ==
             offset_pointer<void>(workspace.workspace_base, data.staged_mixer_state_offset) &&
         workspace.mixer_workspace.workspace_base ==
             offset_pointer<void>(workspace.workspace_base, data.mixer_scratch_offset) &&
         workspace.eigensolver_workspace.plan_identity == data.eigensolver.identity() &&
         workspace.staged_mixer_state.plan_identity == data.mixer.identity() &&
         workspace.mixer_workspace.plan_identity == data.mixer.identity() &&
         workspace.thermodynamics.system_statuses ==
             offset_pointer<gpuxtb_status_t>(workspace.workspace_base,
                                             data.thermodynamic_status_offset) &&
         workspace.thermodynamics.chemical_potentials ==
             offset_pointer<double>(workspace.workspace_base, data.chemical_potential_offset) &&
         workspace.thermodynamics.entropies ==
             offset_pointer<double>(workspace.workspace_base, data.thermodynamic_entropy_offset) &&
         workspace.thermodynamics.band_energies ==
             offset_pointer<double>(workspace.workspace_base,
                                    data.thermodynamic_band_energy_offset) &&
         workspace.thermodynamics.free_energies ==
             offset_pointer<double>(workspace.workspace_base,
                                    data.thermodynamic_free_energy_offset) &&
         workspace.thermodynamics.system_status_capacity ==
             static_cast<std::size_t>(data.wavefunction.batch_size) &&
         workspace.thermodynamics.chemical_potential_capacity ==
             2u * static_cast<std::size_t>(data.wavefunction.batch_size) &&
         workspace.thermodynamics.entropy_capacity ==
             static_cast<std::size_t>(data.wavefunction.batch_size) &&
         workspace.thermodynamics.band_energy_capacity ==
             static_cast<std::size_t>(data.wavefunction.batch_size) &&
         workspace.thermodynamics.free_energy_capacity ==
             static_cast<std::size_t>(data.wavefunction.batch_size);
}

void copy_wavefunction(const WavefunctionLayout& layout, const WavefunctionView& source,
                       const WavefunctionView& destination) {
  const std::array<std::pair<const WavefunctionFieldLayout*, std::pair<const double*, double*>>, 10>
      fields{{{&layout.coefficients, {source.coefficients, destination.coefficients}},
              {&layout.eigenvalues, {source.eigenvalues, destination.eigenvalues}},
              {&layout.occupations, {source.occupations, destination.occupations}},
              {&layout.density, {source.density, destination.density}},
              {&layout.qsh, {source.qsh, destination.qsh}},
              {&layout.qat, {source.qat, destination.qat}},
              {&layout.dipole, {source.dipole, destination.dipole}},
              {&layout.quadrupole, {source.quadrupole, destination.quadrupole}},
              {&layout.energy_weighted_density,
               {source.energy_weighted_density, destination.energy_weighted_density}},
              {nullptr, {nullptr, nullptr}}}};
  for (const auto& field : fields) {
    if (field.first != nullptr && field.first->element_count > 0) {
      std::copy_n(field.second.first, static_cast<std::size_t>(field.first->element_count),
                  field.second.second);
    }
  }
}

void copy_system_field(const WavefunctionFieldLayout& layout, std::size_t system,
                       const double* source, double* destination) {
  const std::size_t begin = static_cast<std::size_t>(layout.system_offsets[system]);
  const std::size_t count =
      static_cast<std::size_t>(layout.system_offsets[system + 1u] - layout.system_offsets[system]);
  std::copy_n(source + begin, count, destination + begin);
}

void commit_system_wavefunction(const WavefunctionLayout& layout, std::size_t system,
                                const WavefunctionView& staged,
                                const WavefunctionView& destination) {
  copy_system_field(layout.coefficients, system, staged.coefficients, destination.coefficients);
  copy_system_field(layout.eigenvalues, system, staged.eigenvalues, destination.eigenvalues);
  copy_system_field(layout.occupations, system, staged.occupations, destination.occupations);
  copy_system_field(layout.density, system, staged.density, destination.density);
  copy_system_field(layout.qsh, system, staged.qsh, destination.qsh);
  copy_system_field(layout.qat, system, staged.qat, destination.qat);
  copy_system_field(layout.dipole, system, staged.dipole, destination.dipole);
  copy_system_field(layout.quadrupole, system, staged.quadrupole, destination.quadrupole);
  copy_system_field(layout.energy_weighted_density, system, staged.energy_weighted_density,
                    destination.energy_weighted_density);
}

gpuxtb_status_t validate_iteration_bindings(
    const SccDriverPlan& plan, const SccDriverPlanData& data, const SccDriverGeometryView& geometry,
    const CpuLinearAlgebraBackend& backend, const EigensolverOverlapCache& overlap_cache,
    const WavefunctionView& wavefunction, const SccMixerState& mixer_state,
    const SccDriverState& state, const SccDriverWorkspace& workspace, std::string& error);
gpuxtb_status_t prepare_potentials_and_hamiltonian(const SccDriverPlanData& data,
                                                   const SccDriverGeometryView& geometry,
                                                   const SccDriverWorkspace& workspace,
                                                   std::string& error);
void copy_raw_population_system(const WavefunctionLayout& layout, std::size_t system,
                                const SccDriverWorkspace& workspace);
gpuxtb_status_t rebuild_mixed_atomic_charges(const SccDriverPlanData& data, std::size_t system,
                                             const SccDriverWorkspace& workspace,
                                             std::string& error);
gpuxtb_status_t evaluate_scc_energy_system(const SccDriverPlanData& data,
                                           const SccDriverGeometryView& geometry,
                                           std::size_t system, const SccDriverWorkspace& workspace,
                                           std::string& error);

}  // namespace

gpuxtb_status_t iterate_scc_driver_batch_cpu(
    const SccDriverPlan& plan, const SccDriverGeometryView& geometry,
    const CpuLinearAlgebraBackend& backend, const EigensolverOverlapCache& overlap_cache,
    const WavefunctionView& wavefunction, const SccMixerState& mixer_state,
    const SccDriverState& state, const SccDriverWorkspace& workspace, std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const SccDriverPlanData& data = *plan.identity();
  status = validate_iteration_bindings(plan, data, geometry, backend, overlap_cache, wavefunction,
                                       mixer_state, state, workspace, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  const std::size_t batch = static_cast<std::size_t>(data.wavefunction.batch_size);
  bool any_active = false;
  for (std::size_t system = 0u; system < batch; ++system) {
    const bool active = state.system_statuses[system] == GPUXTB_STATUS_SUCCESS &&
                        state.converged[system] == 0u &&
                        state.iterations[system] < data.maximum_iterations;
    workspace.active_systems[system] = active ? 1u : 0u;
    any_active = any_active || active;
  }
  if (!any_active) {
    error.clear();
    return GPUXTB_STATUS_SUCCESS;
  }

  /* Every operation up to the mixer barrier publishes only into workspace. */
  copy_wavefunction(data.wavefunction, wavefunction, workspace.staged_wavefunction);
  status = prepare_potentials_and_hamiltonian(data, geometry, workspace, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::fill_n(workspace.thermodynamics.system_statuses, batch, GPUXTB_STATUS_INVALID_ARGUMENT);
  std::fill_n(workspace.thermodynamics.chemical_potentials, 2u * batch, nan);
  std::fill_n(workspace.thermodynamics.entropies, batch, nan);
  std::fill_n(workspace.thermodynamics.band_energies, batch, nan);
  std::fill_n(workspace.thermodynamics.free_energies, batch, nan);

  for (std::size_t system = 0u; system < batch; ++system) {
    if (workspace.active_systems[system] != 1u) {
      continue;
    }
    const std::int64_t hamiltonian_base = data.wavefunction.density.system_offsets[system];
    status = solve_eigensystem_cpu(
        data.eigensolver, static_cast<std::int64_t>(system), overlap_cache,
        geometry.geometry_generation, workspace.hamiltonian + hamiltonian_base,
        data.electronic_temperature, backend, workspace.eigensolver_workspace,
        workspace.staged_wavefunction, workspace.thermodynamics, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      /* Binding/backend contract failures are whole-call failures; all solved
       * peers still live only in the staged wavefunction at this point. */
      return status;
    }
    if (workspace.thermodynamics.system_statuses[system] != GPUXTB_STATUS_SUCCESS) {
      workspace.active_systems[system] = 2u;
      const std::int64_t density_begin = data.wavefunction.density.system_offsets[system];
      const std::int64_t density_end = data.wavefunction.density.system_offsets[system + 1u];
      std::fill_n(workspace.staged_wavefunction.density + density_begin,
                  static_cast<std::size_t>(density_end - density_begin), 0.0);
    }
  }

  bool any_solved = false;
  for (std::size_t system = 0u; system < batch; ++system) {
    any_solved = any_solved || workspace.active_systems[system] == 1u;
  }
  if (any_solved) {
    const MullikenDensityView density{workspace.staged_wavefunction.density,
                                      data.wavefunction.density.element_count,
                                      data.mulliken.identity()};
    const MullikenPopulationView population{
        workspace.raw_qsh,         data.wavefunction.qsh.element_count,
        workspace.raw_qat,         data.wavefunction.qat.element_count,
        workspace.raw_dipoles,     data.wavefunction.dipole.element_count,
        workspace.raw_quadrupoles, data.wavefunction.quadrupole.element_count,
        data.mulliken.identity()};
    status = evaluate_mulliken_population_cpu(data.mulliken, geometry.integrals, density,
                                              population, workspace.mulliken_workspace, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return status;
    }
  }

  /* Assemble the variational SCC energy from the new density and raw
   * Mulliken multipoles. The Hamiltonian was built from the previous mixed
   * input, so neither its band trace nor pre-H D4/periodic diagnostics are a
   * valid energy functional for this iteration. */
  for (std::size_t system = 0u; system < batch; ++system) {
    if (workspace.active_systems[system] != 1u) {
      continue;
    }
    status = evaluate_scc_energy_system(data, geometry, system, workspace, error);
    if (status == GPUXTB_STATUS_INVALID_ARGUMENT || status == GPUXTB_STATUS_NOT_SUPPORTED) {
      return status;
    }
    if (status != GPUXTB_STATUS_SUCCESS) {
      workspace.active_systems[system] = 6u;
    }
  }

  /* Run every mixer transition against an unpublished clone. This makes the
   * mixer history and wavefunction one transaction: even post-mix qsh->qat
   * validation can fail without advancing the public history. */
  std::memcpy(workspace.staged_mixer_state.workspace_base, mixer_state.workspace_base,
              data.mixer.state_size_bytes());
  for (std::size_t system = 0u; system < batch; ++system) {
    if (workspace.active_systems[system] != 1u) {
      continue;
    }

    copy_raw_population_system(data.wavefunction, system, workspace);
    status = mix_scc_broyden_system_cpu(data.mixer, static_cast<std::int64_t>(system),
                                        workspace.staged_wavefunction, workspace.staged_mixer_state,
                                        workspace.mixer_workspace, error);
    if (status == GPUXTB_STATUS_INVALID_ARGUMENT) {
      /* No public state has been changed, so a structural failure remains
       * whole-call atomic even after earlier staged peers succeeded. */
      return status;
    }
    if (status != GPUXTB_STATUS_SUCCESS) {
      workspace.active_systems[system] = 3u;
      continue;
    }
    const std::uint64_t old_iteration = state.iterations[system];
    const double old_free_energy = old_iteration == 0u ? 0.0 : state.free_energies[system];
    const double energy_change = workspace.free_energies[system] - old_free_energy;
    if (!std::isfinite(old_free_energy) || !std::isfinite(energy_change)) {
      error = "SCC driver energy convergence history is not finite";
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }
    const bool residual_converged =
        workspace.staged_mixer_state.residual_rms[system] < data.mixer.rms_tolerance();
    const bool energy_converged = std::abs(energy_change) < data.energy_tolerance;
    const bool converged = residual_converged && energy_converged;
    workspace.staged_mixer_state.converged[system] = converged ? 1u : 0u;
    if (converged) {
      /* The mixer stores its next-round input, but a terminal converged
       * wavefunction must expose the density-derived raw multipoles used by
       * tblite total-energy and force evaluation. Restart reinitializes the
       * mixer from this published raw state, so no stale mixed input leaks
       * into a resumed trajectory. */
      copy_raw_population_system(data.wavefunction, system, workspace);
    }
    status = rebuild_mixed_atomic_charges(data, system, workspace, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      /* Discard the entire staged mixer transaction. This path is expected
       * only for a finite-but-unrepresentable shell reduction, and it must
       * never leave public history ahead of the public wavefunction. */
      return status;
    }
    workspace.active_systems[system] = 4u;
  }

  /* Publication starts only after every active raw result and mixed atomic
   * charge has passed. Mixer numerical failures modify only their staged
   * status entry; their history arrays remain byte-identical to the input. */
  std::memcpy(mixer_state.workspace_base, workspace.staged_mixer_state.workspace_base,
              data.mixer.state_size_bytes());

  gpuxtb_status_t first_failure = GPUXTB_STATUS_SUCCESS;
  bool first_failure_was_periodic = false;
  for (std::size_t system = 0u; system < batch; ++system) {
    if (workspace.active_systems[system] == 2u || workspace.active_systems[system] == 3u ||
        workspace.active_systems[system] == 5u || workspace.active_systems[system] == 6u) {
      const gpuxtb_status_t failure =
          workspace.active_systems[system] == 2u
              ? GPUXTB_STATUS_EIGENSOLVER_FAILED
              : (workspace.active_systems[system] == 5u
                     ? GPUXTB_STATUS_INTERNAL_ERROR
                     : (workspace.active_systems[system] == 6u
                            ? GPUXTB_STATUS_INTERNAL_ERROR
                            : workspace.staged_mixer_state.system_statuses[system]));
      state.system_statuses[system] = failure;
      state.free_energies[system] = nan;
      state.previous_free_energies[system] = nan;
      state.free_energy_changes[system] = nan;
      state.entropies[system] = nan;
      state.band_energies[system] = nan;
      state.core_energies[system] = nan;
      state.es2_energies[system] = nan;
      state.es3_energies[system] = nan;
      state.aes2_energies[system] = nan;
      state.spin_energies[system] = nan;
      if (state.d4_two_body_energies != nullptr) {
        state.d4_two_body_energies[system] = nan;
      }
      state.explicit_point_charge_energies[system] = nan;
      if (state.periodic_embedding_energies != nullptr) {
        state.periodic_embedding_energies[system] = nan;
      }
      state.internal_energies[system] = nan;
      if (workspace.active_systems[system] != 5u) {
        ++state.iterations[system];
      }
      if (first_failure == GPUXTB_STATUS_SUCCESS) {
        first_failure = failure;
        first_failure_was_periodic = workspace.active_systems[system] == 5u;
      }
      continue;
    }
    if (workspace.active_systems[system] != 4u) {
      continue;
    }

    const double old_free_energy =
        state.iterations[system] == 0u ? 0.0 : state.free_energies[system];
    const double new_free_energy = workspace.free_energies[system];
    const std::uint64_t old_iteration = state.iterations[system];
    const double change = new_free_energy - old_free_energy;
    const bool converged = mixer_state.converged[system] == 1u;

    commit_system_wavefunction(data.wavefunction, system, workspace.staged_wavefunction,
                               wavefunction);
    state.previous_free_energies[system] = old_free_energy;
    state.free_energies[system] = new_free_energy;
    state.free_energy_changes[system] = change;
    state.entropies[system] = workspace.thermodynamics.entropies[system];
    state.band_energies[system] = workspace.thermodynamics.band_energies[system];
    state.core_energies[system] = workspace.core_energies[system];
    state.es2_energies[system] = workspace.es2_energies[system];
    state.es3_energies[system] = workspace.es3_energies[system];
    state.aes2_energies[system] = workspace.aes2_energies[system];
    state.spin_energies[system] = workspace.spin_energies[system];
    if (state.d4_two_body_energies != nullptr) {
      state.d4_two_body_energies[system] = workspace.d4_two_body_energies[system];
    }
    state.explicit_point_charge_energies[system] = workspace.explicit_point_charge_energies[system];
    if (state.periodic_embedding_energies != nullptr) {
      state.periodic_embedding_energies[system] = workspace.periodic_embedding_energies[system];
    }
    state.internal_energies[system] = workspace.internal_energies[system];
    state.iterations[system] = old_iteration + 1u;
    state.converged[system] = converged ? 1u : 0u;
    if (!converged && state.iterations[system] >= data.maximum_iterations) {
      state.system_statuses[system] = GPUXTB_STATUS_SCC_NOT_CONVERGED;
      if (first_failure == GPUXTB_STATUS_SUCCESS) {
        first_failure = GPUXTB_STATUS_SCC_NOT_CONVERGED;
      }
    } else {
      state.system_statuses[system] = GPUXTB_STATUS_SUCCESS;
    }
  }

  if (first_failure != GPUXTB_STATUS_SUCCESS) {
    if (first_failure == GPUXTB_STATUS_EIGENSOLVER_FAILED) {
      error = "one or more SCC systems failed during generalized eigensolve";
    } else if (first_failure == GPUXTB_STATUS_SCC_NOT_CONVERGED) {
      error = "one or more SCC systems reached the maximum iteration count";
    } else if (first_failure_was_periodic) {
      error = "one or more SCC systems failed during periodic charge embedding";
    } else if (first_failure == GPUXTB_STATUS_INTERNAL_ERROR) {
      error = "one or more SCC systems failed during energy assembly or mixing";
    } else {
      error = "one or more SCC systems failed during Broyden mixing";
    }
    return first_failure;
  }
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

SccDriverPlan::SccDriverPlan(std::shared_ptr<const SccDriverPlanData> data) noexcept
    : data_(std::move(data)) {}

bool SccDriverPlan::sealed() const noexcept { return data_ != nullptr; }
std::int64_t SccDriverPlan::batch_size() const noexcept {
  return data_ == nullptr ? 0 : data_->wavefunction.batch_size;
}
std::uint64_t SccDriverPlan::maximum_iterations() const noexcept {
  return data_ == nullptr ? 0u : data_->maximum_iterations;
}
double SccDriverPlan::electronic_temperature() const noexcept {
  return data_ == nullptr ? 0.0 : data_->electronic_temperature;
}
double SccDriverPlan::energy_tolerance() const noexcept {
  return data_ == nullptr ? 0.0 : data_->energy_tolerance;
}
bool SccDriverPlan::d4_enabled() const noexcept { return data_ != nullptr && data_->d4.sealed(); }
bool SccDriverPlan::periodic_embedding_enabled() const noexcept {
  return data_ != nullptr && data_->periodic_embedding.sealed();
}
std::size_t SccDriverPlan::state_size_bytes() const noexcept {
  return data_ == nullptr ? 0u : data_->state_size_bytes;
}
std::size_t SccDriverPlan::workspace_size_bytes() const noexcept {
  return data_ == nullptr ? 0u : data_->workspace_size_bytes;
}
const SccDriverPlanData* SccDriverPlan::identity() const noexcept { return data_.get(); }

gpuxtb_status_t make_scc_driver_plan(const WavefunctionLayout& wavefunction,
                                     const MullikenPlan& mulliken, const ES2Plan& es2,
                                     const ES3Plan& es3, const AES2Plan& aes2,
                                     const EigensolverPlan& eigensolver, const SccMixerPlan& mixer,
                                     std::uint64_t maximum_iterations,
                                     double electronic_temperature, SccDriverPlan& plan,
                                     std::string& error) {
  return make_scc_driver_plan(wavefunction, mulliken, es2, es3, aes2, eigensolver, mixer, nullptr,
                              nullptr, maximum_iterations, electronic_temperature,
                              kDefaultSccEnergyTolerance, plan, error);
}

gpuxtb_status_t make_scc_driver_plan(const WavefunctionLayout& wavefunction,
                                     const MullikenPlan& mulliken, const ES2Plan& es2,
                                     const ES3Plan& es3, const AES2Plan& aes2,
                                     const EigensolverPlan& eigensolver, const SccMixerPlan& mixer,
                                     const PeriodicEmbeddingPlan* periodic_embedding,
                                     std::uint64_t maximum_iterations,
                                     double electronic_temperature, SccDriverPlan& plan,
                                     std::string& error) {
  return make_scc_driver_plan(wavefunction, mulliken, es2, es3, aes2, eigensolver, mixer, nullptr,
                              periodic_embedding, maximum_iterations, electronic_temperature,
                              kDefaultSccEnergyTolerance, plan, error);
}

gpuxtb_status_t make_scc_driver_plan(
    const WavefunctionLayout& wavefunction, const MullikenPlan& mulliken, const ES2Plan& es2,
    const ES3Plan& es3, const AES2Plan& aes2, const EigensolverPlan& eigensolver,
    const SccMixerPlan& mixer, const D4Plan* d4, const PeriodicEmbeddingPlan* periodic_embedding,
    std::uint64_t maximum_iterations, double electronic_temperature, SccDriverPlan& plan,
    std::string& error) {
  return make_scc_driver_plan(wavefunction, mulliken, es2, es3, aes2, eigensolver, mixer, d4,
                              periodic_embedding, maximum_iterations, electronic_temperature,
                              kDefaultSccEnergyTolerance, plan, error);
}

gpuxtb_status_t make_scc_driver_plan(
    const WavefunctionLayout& wavefunction, const MullikenPlan& mulliken, const ES2Plan& es2,
    const ES3Plan& es3, const AES2Plan& aes2, const EigensolverPlan& eigensolver,
    const SccMixerPlan& mixer, const D4Plan* d4, const PeriodicEmbeddingPlan* periodic_embedding,
    std::uint64_t maximum_iterations, double electronic_temperature, double energy_tolerance,
    SccDriverPlan& plan, std::string& error) {
  WavefunctionWarmStartIdentity validated_wavefunction;
  gpuxtb_status_t status =
      make_wavefunction_warm_start_identity(wavefunction, 0u, validated_wavefunction, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (!mulliken.sealed() || !es2.sealed() || !aes2.sealed() || !eigensolver.sealed() ||
      !mixer.sealed() || maximum_iterations == 0u || !std::isfinite(electronic_temperature) ||
      electronic_temperature < 0.0 || !std::isfinite(energy_tolerance) ||
      !(energy_tolerance > 0.0) || (d4 != nullptr && !d4->sealed()) ||
      (periodic_embedding != nullptr && !periodic_embedding->sealed())) {
    error = "SCC driver components or numerical policy are invalid";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  /* Rebuild the canonical GFN2 component metadata from one chemical source
   * of truth. Aggregate extents are insufficient: same-sized molecules can
   * have different elements, electron counts, shell maps, and SCC parameters.
   * Plan construction is cold-path work, so exact reconstruction is preferred
   * over a collision-prone compatibility hash. */
  BasisPlan expected_basis;
  WavefunctionLayout expected_wavefunction;
  IntegralPlan expected_integrals;
  MullikenPlan expected_mulliken;
  ES2Plan expected_es2;
  ES3Plan expected_es3;
  AES2Plan expected_aes2;
  EigensolverPlan expected_eigensolver;
  SccMixerPlan expected_mixer;
  SpinPolarizationPlan expected_spin;
  status = make_basis_plan(wavefunction.batch_size, wavefunction.total_atoms,
                           wavefunction.atom_offsets.data(), wavefunction.atomic_numbers.data(),
                           expected_basis, error);
  if (status == GPUXTB_STATUS_SUCCESS) {
    status = make_wavefunction_layout(
        expected_basis, wavefunction.atomic_numbers.data(), wavefunction.molecular_charges.data(),
        wavefunction.unpaired_electrons.data(), wavefunction.spin_channels.data(),
        expected_wavefunction, error);
  }
  if (status == GPUXTB_STATUS_SUCCESS) {
    status = make_integral_plan(expected_basis, expected_integrals, error);
  }
  if (status == GPUXTB_STATUS_SUCCESS) {
    status = make_mulliken_plan(expected_basis, expected_integrals, expected_wavefunction,
                                expected_mulliken, error);
  }
  if (status == GPUXTB_STATUS_SUCCESS) {
    status = make_es2_plan(expected_basis, wavefunction.atomic_numbers.data(), expected_es2, error);
  }
  if (status == GPUXTB_STATUS_SUCCESS) {
    status = make_es3_plan(expected_basis, wavefunction.atomic_numbers.data(), expected_es3, error);
  }
  if (status == GPUXTB_STATUS_SUCCESS) {
    status =
        make_aes2_plan(expected_basis, wavefunction.atomic_numbers.data(), expected_aes2, error);
  }
  if (status == GPUXTB_STATUS_SUCCESS) {
    status = make_eigensolver_plan(expected_wavefunction, expected_eigensolver, error,
                                   eigensolver.minimum_overlap_rcond());
  }
  if (status == GPUXTB_STATUS_SUCCESS) {
    status = make_scc_mixer_plan(expected_wavefunction, mixer.history_size(), mixer.damping(),
                                 mixer.rms_tolerance(), mixer.maximum_tolerance(), expected_mixer,
                                 error);
  }
  if (status == GPUXTB_STATUS_SUCCESS) {
    status =
        make_spin_polarization_plan(expected_basis, expected_wavefunction, expected_spin, error);
  }
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (!same_wavefunction_layout(wavefunction, expected_wavefunction) ||
      !same_mulliken_plan(mulliken, expected_mulliken) || !same_es2_plan(es2, expected_es2) ||
      !same_es3_plan(es3, expected_es3) || !same_aes2_plan(aes2, expected_aes2) ||
      !same_eigensolver_plan(eigensolver, expected_eigensolver) ||
      !same_mixer_plan(mixer, expected_mixer) ||
      !mixer.matches_wavefunction_layout(expected_wavefunction)) {
    error = "SCC driver components do not share one canonical GFN2 chemical identity and layout";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  const std::size_t batch = static_cast<std::size_t>(wavefunction.batch_size);
  const bool common_extents =
      mulliken.batch_size() == wavefunction.batch_size &&
      mulliken.total_atoms() == wavefunction.total_atoms &&
      mulliken.total_shells() == wavefunction.total_shells &&
      mulliken.total_orbitals() == wavefunction.total_orbitals &&
      mulliken.density_elements() == wavefunction.density.element_count &&
      mulliken.shell_population_elements() == wavefunction.qsh.element_count &&
      mulliken.atom_population_elements() == wavefunction.qat.element_count &&
      mulliken.dipole_population_elements() == wavefunction.dipole.element_count &&
      mulliken.quadrupole_population_elements() == wavefunction.quadrupole.element_count &&
      es2.batch_size() == wavefunction.batch_size &&
      es2.total_atoms() == wavefunction.total_atoms &&
      es2.total_shells() == wavefunction.total_shells &&
      aes2.batch_size() == wavefunction.batch_size &&
      aes2.total_atoms() == wavefunction.total_atoms &&
      eigensolver.batch_size() == wavefunction.batch_size &&
      mixer.batch_size() == wavefunction.batch_size;
  if (!common_extents || mulliken.atom_offsets() != wavefunction.atom_offsets ||
      mulliken.batch_shell_offsets() != wavefunction.batch_shell_offsets ||
      mulliken.batch_orbital_offsets() != wavefunction.batch_orbital_offsets ||
      mulliken.spin_channels() != wavefunction.spin_channels ||
      es2.atom_offsets() != wavefunction.atom_offsets ||
      es2.batch_shell_offsets() != wavefunction.batch_shell_offsets ||
      aes2.atom_offsets() != wavefunction.atom_offsets ||
      eigensolver.orbital_offsets() != wavefunction.batch_orbital_offsets ||
      eigensolver.spin_channels() != wavefunction.spin_channels ||
      es3.batch_size != wavefunction.batch_size || es3.total_shells != wavefunction.total_shells ||
      es3.batch_shell_offsets != wavefunction.batch_shell_offsets ||
      es3.shell_gamma3.size() != static_cast<std::size_t>(wavefunction.total_shells)) {
    error = "SCC driver component plans describe different ragged topology";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (periodic_embedding != nullptr &&
      (periodic_embedding->batch_size() != wavefunction.batch_size ||
       periodic_embedding->total_atoms() != wavefunction.total_atoms ||
       periodic_embedding->atom_offsets() != wavefunction.atom_offsets)) {
    error = "SCC driver periodic embedding describes a different ragged atom topology";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (d4 != nullptr && (d4->batch_size() != wavefunction.batch_size ||
                        d4->total_atoms() != wavefunction.total_atoms ||
                        d4->atom_offsets() != wavefunction.atom_offsets ||
                        !d4->matches_atomic_numbers(wavefunction.atomic_numbers.data()))) {
    error = "SCC driver D4 plan describes a different chemical identity or ragged topology";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (mixer.vector_offsets().size() != batch + 1u) {
    error = "SCC driver mixer vector partition is malformed";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  std::int64_t expected_vector_offset = 0;
  for (std::size_t system = 0u; system < batch; ++system) {
    const std::int64_t qsh =
        wavefunction.qsh.system_offsets[system + 1u] - wavefunction.qsh.system_offsets[system];
    const std::int64_t dipole = wavefunction.dipole.system_offsets[system + 1u] -
                                wavefunction.dipole.system_offsets[system];
    const std::int64_t quadrupole = wavefunction.quadrupole.system_offsets[system + 1u] -
                                    wavefunction.quadrupole.system_offsets[system];
    if (qsh <= 0 || dipole <= 0 || quadrupole <= 0 ||
        expected_vector_offset > std::numeric_limits<std::int64_t>::max() - qsh) {
      error = "SCC driver mixed-vector dimensions are invalid";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    expected_vector_offset += qsh;
    if (expected_vector_offset > std::numeric_limits<std::int64_t>::max() - dipole) {
      error = "SCC driver mixed-vector dimensions are invalid";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    expected_vector_offset += dipole;
    if (expected_vector_offset > std::numeric_limits<std::int64_t>::max() - quadrupole) {
      error = "SCC driver mixed-vector dimensions are invalid";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    expected_vector_offset += quadrupole;
    if (mixer.vector_offsets()[system + 1u] != expected_vector_offset) {
      error = "SCC driver mixer was built for a different wavefunction layout";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  std::size_t batch_double_bytes = 0u;
  std::size_t batch_u64_bytes = 0u;
  std::size_t batch_status_bytes = 0u;
  std::size_t batch_byte_bytes = 0u;
  std::size_t chemical_potential_bytes = 0u;
  std::size_t hamiltonian_bytes = 0u;
  std::size_t shell_bytes = 0u;
  std::size_t atom_bytes = 0u;
  std::size_t atomic_dipole_bytes = 0u;
  std::size_t atomic_quadrupole_bytes = 0u;
  std::size_t shell_population_bytes = 0u;
  std::size_t atom_population_bytes = 0u;
  std::size_t dipole_population_bytes = 0u;
  std::size_t quadrupole_population_bytes = 0u;
  const std::int64_t mulliken_scratch_elements =
      std::max(mulliken.population_scratch_elements(), mulliken.hamiltonian_scratch_elements());
  std::size_t mulliken_scratch_bytes = 0u;
  std::size_t aes2_scratch_bytes = 0u;
  const std::size_t d4_scratch_bytes = d4 == nullptr ? 0u : d4->workspace_size_bytes();
  std::size_t periodic_scratch_bytes = 0u;
  if (!bytes_for(wavefunction.batch_size, sizeof(double), batch_double_bytes) ||
      !bytes_for(wavefunction.batch_size, sizeof(std::uint64_t), batch_u64_bytes) ||
      !bytes_for(wavefunction.batch_size, sizeof(gpuxtb_status_t), batch_status_bytes) ||
      !bytes_for(wavefunction.batch_size, sizeof(std::uint8_t), batch_byte_bytes) ||
      !bytes_for(wavefunction.density.element_count, sizeof(double), hamiltonian_bytes) ||
      !bytes_for(wavefunction.total_shells, sizeof(double), shell_bytes) ||
      !bytes_for(wavefunction.total_atoms, sizeof(double), atom_bytes) ||
      !bytes_for(wavefunction.qsh.element_count, sizeof(double), shell_population_bytes) ||
      !bytes_for(wavefunction.qat.element_count, sizeof(double), atom_population_bytes) ||
      !bytes_for(wavefunction.dipole.element_count, sizeof(double), dipole_population_bytes) ||
      !bytes_for(wavefunction.quadrupole.element_count, sizeof(double),
                 quadrupole_population_bytes) ||
      !bytes_for(mulliken_scratch_elements, sizeof(double), mulliken_scratch_bytes) ||
      !bytes_for(aes2.potential_scratch_elements(), sizeof(double), aes2_scratch_bytes) ||
      !bytes_for(periodic_embedding == nullptr ? 0 : periodic_embedding->maximum_atoms(),
                 sizeof(double), periodic_scratch_bytes) ||
      !checked_multiply_size(batch_double_bytes, 2u, chemical_potential_bytes) ||
      !checked_multiply_size(atom_bytes, 3u, atomic_dipole_bytes) ||
      !checked_multiply_size(atom_bytes, 6u, atomic_quadrupole_bytes)) {
    error = "SCC driver caller-owned storage exceeds addressable memory";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t periodic_state_bytes = periodic_embedding == nullptr ? 0u : batch_double_bytes;
  const std::size_t d4_state_bytes = d4 == nullptr ? 0u : batch_double_bytes;
  const std::size_t d4_potential_bytes = d4 == nullptr ? 0u : atom_bytes;
  const std::size_t d4_energy_bytes = d4 == nullptr ? 0u : batch_double_bytes;
  const std::size_t periodic_potential_bytes = periodic_embedding == nullptr ? 0u : atom_bytes;
  const std::size_t periodic_energy_bytes = periodic_embedding == nullptr ? 0u : batch_double_bytes;
  const std::size_t periodic_status_bytes = periodic_embedding == nullptr ? 0u : batch_status_bytes;

  try {
    SccDriverPlanData created;
    created.wavefunction = wavefunction;
    created.mulliken = mulliken;
    created.es2 = es2;
    created.es3 = es3;
    created.aes2 = aes2;
    created.spin = std::move(expected_spin);
    if (d4 != nullptr) {
      created.d4 = *d4;
    }
    created.eigensolver = eigensolver;
    created.mixer = mixer;
    if (periodic_embedding != nullptr) {
      created.periodic_embedding = *periodic_embedding;
    }
    created.maximum_iterations = maximum_iterations;
    created.electronic_temperature = electronic_temperature;
    created.energy_tolerance = energy_tolerance;

    std::size_t cursor = 0u;
    if (!append_segment(batch_double_bytes, alignof(double), cursor,
                        created.state_free_energy_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor,
                        created.state_previous_free_energy_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor,
                        created.state_free_energy_change_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor,
                        created.state_entropy_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor,
                        created.state_band_energy_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor,
                        created.state_core_energy_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor,
                        created.state_es2_energy_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor,
                        created.state_es3_energy_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor,
                        created.state_aes2_energy_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor,
                        created.state_spin_energy_offset) ||
        !append_segment(d4_state_bytes, alignof(double), cursor, created.state_d4_energy_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor,
                        created.state_explicit_pc_energy_offset) ||
        !append_segment(periodic_state_bytes, alignof(double), cursor,
                        created.state_periodic_energy_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor,
                        created.state_internal_energy_offset) ||
        !append_segment(batch_u64_bytes, alignof(std::uint64_t), cursor,
                        created.state_iteration_offset) ||
        !append_segment(batch_status_bytes, alignof(gpuxtb_status_t), cursor,
                        created.state_status_offset) ||
        !append_segment(batch_byte_bytes, alignof(std::uint8_t), cursor,
                        created.state_initialized_offset) ||
        !append_segment(batch_byte_bytes, alignof(std::uint8_t), cursor,
                        created.state_converged_offset) ||
        !align_up(cursor, kSccDriverWorkspaceAlignment, created.state_size_bytes)) {
      error = "SCC driver state layout overflows size_t";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }

    cursor = 0u;
    if (!append_segment(wavefunction.workspace_size_bytes, kWavefunctionWorkspaceAlignment, cursor,
                        created.staged_wavefunction_offset) ||
        !append_segment(hamiltonian_bytes, alignof(double), cursor, created.hamiltonian_offset) ||
        !append_segment(shell_bytes, alignof(double), cursor, created.shell_charge_offset) ||
        !append_segment(atom_bytes, alignof(double), cursor, created.atomic_charge_offset) ||
        !append_segment(atomic_dipole_bytes, alignof(double), cursor,
                        created.atomic_dipole_offset) ||
        !append_segment(atomic_quadrupole_bytes, alignof(double), cursor,
                        created.atomic_quadrupole_offset) ||
        !append_segment(shell_bytes, alignof(double), cursor, created.component_shell_offset) ||
        !append_segment(atom_bytes, alignof(double), cursor, created.component_atomic_offset) ||
        !append_segment(atomic_dipole_bytes, alignof(double), cursor,
                        created.component_dipole_offset) ||
        !append_segment(atomic_quadrupole_bytes, alignof(double), cursor,
                        created.component_quadrupole_offset) ||
        !append_segment(atom_population_bytes, alignof(double), cursor,
                        created.atom_potential_offset) ||
        !append_segment(shell_population_bytes, alignof(double), cursor,
                        created.shell_potential_offset) ||
        !append_segment(dipole_population_bytes, alignof(double), cursor,
                        created.dipole_potential_offset) ||
        !append_segment(quadrupole_population_bytes, alignof(double), cursor,
                        created.quadrupole_potential_offset) ||
        !append_segment(shell_population_bytes, alignof(double), cursor,
                        created.spin_shell_potential_offset) ||
        !append_segment(shell_population_bytes, alignof(double), cursor, created.raw_qsh_offset) ||
        !append_segment(atom_population_bytes, alignof(double), cursor, created.raw_qat_offset) ||
        !append_segment(dipole_population_bytes, alignof(double), cursor,
                        created.raw_dipole_offset) ||
        !append_segment(quadrupole_population_bytes, alignof(double), cursor,
                        created.raw_quadrupole_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor, created.core_energy_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor, created.es2_energy_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor, created.es3_energy_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor, created.aes2_energy_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor, created.spin_energy_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor,
                        created.explicit_pc_energy_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor,
                        created.internal_energy_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor, created.free_energy_offset) ||
        !append_segment(d4_potential_bytes, alignof(double), cursor, created.d4_potential_offset) ||
        !append_segment(d4_energy_bytes, alignof(double), cursor, created.d4_energy_offset) ||
        !append_segment(d4_scratch_bytes, d4 == nullptr ? alignof(double) : kD4WorkspaceAlignment,
                        cursor, created.d4_scratch_offset) ||
        !append_segment(periodic_potential_bytes, alignof(double), cursor,
                        created.periodic_potential_offset) ||
        !append_segment(periodic_energy_bytes, alignof(double), cursor,
                        created.periodic_energy_offset) ||
        !append_segment(periodic_status_bytes, alignof(gpuxtb_status_t), cursor,
                        created.periodic_status_offset) ||
        !append_segment(periodic_scratch_bytes, alignof(double), cursor,
                        created.periodic_scratch_offset) ||
        !append_segment(shell_bytes, alignof(double), cursor, created.es2_shell_scratch_offset) ||
        !append_segment(aes2_scratch_bytes, alignof(double), cursor,
                        created.aes2_potential_scratch_offset) ||
        !append_segment(mulliken_scratch_bytes, alignof(double), cursor,
                        created.mulliken_scratch_offset) ||
        !append_segment(eigensolver.worker_workspace_size_bytes(), kEigensolverWorkspaceAlignment,
                        cursor, created.eigensolver_scratch_offset) ||
        !append_segment(mixer.state_size_bytes(), kSccMixerWorkspaceAlignment, cursor,
                        created.staged_mixer_state_offset) ||
        !append_segment(mixer.workspace_size_bytes(), kSccMixerWorkspaceAlignment, cursor,
                        created.mixer_scratch_offset) ||
        !append_segment(batch_status_bytes, alignof(gpuxtb_status_t), cursor,
                        created.thermodynamic_status_offset) ||
        !append_segment(chemical_potential_bytes, alignof(double), cursor,
                        created.chemical_potential_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor,
                        created.thermodynamic_entropy_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor,
                        created.thermodynamic_band_energy_offset) ||
        !append_segment(batch_double_bytes, alignof(double), cursor,
                        created.thermodynamic_free_energy_offset) ||
        !append_segment(batch_byte_bytes, alignof(std::uint8_t), cursor,
                        created.active_system_offset) ||
        !align_up(cursor, kSccDriverWorkspaceAlignment, created.workspace_size_bytes)) {
      error = "SCC driver scratch layout overflows size_t";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }

    plan = SccDriverPlan(std::make_shared<const SccDriverPlanData>(std::move(created)));
    error.clear();
    return GPUXTB_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate immutable SCC driver metadata";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  }
}

gpuxtb_status_t bind_scc_driver_state(const SccDriverPlan& plan, void* workspace,
                                      std::size_t workspace_size, SccDriverState& state,
                                      std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const SccDriverPlanData& data = *plan.identity();
  AddressRange storage_range;
  AddressRange plan_range;
  AddressRange state_range;
  AddressRange error_range;
  if (!aligned(workspace, kSccDriverWorkspaceAlignment) || workspace_size < data.state_size_bytes ||
      !make_range(workspace, data.state_size_bytes, storage_range) ||
      !make_range(&plan, sizeof(plan), plan_range) ||
      !make_range(&state, sizeof(state), state_range) ||
      !make_range(&error, sizeof(error), error_range) ||
      ranges_overlap(storage_range, plan_range) || ranges_overlap(storage_range, state_range) ||
      ranges_overlap(storage_range, error_range) || overlaps_plan_storage(data, storage_range)) {
    error = "SCC driver state storage is invalid or overlaps control storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  SccDriverState created;
  created.workspace_base = workspace;
  created.workspace_size_bytes = workspace_size;
  created.free_energies = offset_pointer<double>(workspace, data.state_free_energy_offset);
  created.previous_free_energies =
      offset_pointer<double>(workspace, data.state_previous_free_energy_offset);
  created.free_energy_changes =
      offset_pointer<double>(workspace, data.state_free_energy_change_offset);
  created.entropies = offset_pointer<double>(workspace, data.state_entropy_offset);
  created.band_energies = offset_pointer<double>(workspace, data.state_band_energy_offset);
  created.core_energies = offset_pointer<double>(workspace, data.state_core_energy_offset);
  created.es2_energies = offset_pointer<double>(workspace, data.state_es2_energy_offset);
  created.es3_energies = offset_pointer<double>(workspace, data.state_es3_energy_offset);
  created.aes2_energies = offset_pointer<double>(workspace, data.state_aes2_energy_offset);
  created.spin_energies = offset_pointer<double>(workspace, data.state_spin_energy_offset);
  if (data.d4.sealed()) {
    created.d4_two_body_energies = offset_pointer<double>(workspace, data.state_d4_energy_offset);
  }
  created.explicit_point_charge_energies =
      offset_pointer<double>(workspace, data.state_explicit_pc_energy_offset);
  if (data.periodic_embedding.sealed()) {
    created.periodic_embedding_energies =
        offset_pointer<double>(workspace, data.state_periodic_energy_offset);
  }
  created.internal_energies = offset_pointer<double>(workspace, data.state_internal_energy_offset);
  created.iterations = offset_pointer<std::uint64_t>(workspace, data.state_iteration_offset);
  created.system_statuses = offset_pointer<gpuxtb_status_t>(workspace, data.state_status_offset);
  created.initialized = offset_pointer<std::uint8_t>(workspace, data.state_initialized_offset);
  created.converged = offset_pointer<std::uint8_t>(workspace, data.state_converged_offset);
  created.plan_identity = &data;

  std::memset(workspace, 0, data.state_size_bytes);
  const std::size_t batch = static_cast<std::size_t>(data.wavefunction.batch_size);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::fill_n(created.free_energies, batch, nan);
  std::fill_n(created.previous_free_energies, batch, nan);
  std::fill_n(created.free_energy_changes, batch, nan);
  std::fill_n(created.entropies, batch, nan);
  std::fill_n(created.band_energies, batch, nan);
  std::fill_n(created.core_energies, batch, nan);
  std::fill_n(created.es2_energies, batch, nan);
  std::fill_n(created.es3_energies, batch, nan);
  std::fill_n(created.aes2_energies, batch, nan);
  std::fill_n(created.spin_energies, batch, nan);
  if (created.d4_two_body_energies != nullptr) {
    std::fill_n(created.d4_two_body_energies, batch, nan);
  }
  std::fill_n(created.explicit_point_charge_energies, batch, nan);
  if (created.periodic_embedding_energies != nullptr) {
    std::fill_n(created.periodic_embedding_energies, batch, nan);
  }
  std::fill_n(created.internal_energies, batch, nan);
  std::fill_n(created.system_statuses, batch, GPUXTB_STATUS_INVALID_ARGUMENT);
  state = created;
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t bind_scc_driver_workspace(const SccDriverPlan& plan, void* workspace,
                                          std::size_t workspace_size, SccDriverWorkspace& view,
                                          std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const SccDriverPlanData& data = *plan.identity();
  AddressRange storage_range;
  AddressRange plan_range;
  AddressRange view_range;
  AddressRange error_range;
  if (!aligned(workspace, kSccDriverWorkspaceAlignment) ||
      workspace_size < data.workspace_size_bytes ||
      !make_range(workspace, data.workspace_size_bytes, storage_range) ||
      !make_range(&plan, sizeof(plan), plan_range) ||
      !make_range(&view, sizeof(view), view_range) ||
      !make_range(&error, sizeof(error), error_range) ||
      ranges_overlap(storage_range, plan_range) || ranges_overlap(storage_range, view_range) ||
      ranges_overlap(storage_range, error_range) || overlaps_plan_storage(data, storage_range)) {
    error = "SCC driver scratch storage is invalid or overlaps control storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  SccDriverWorkspace created;
  created.workspace_base = workspace;
  created.workspace_size_bytes = workspace_size;
  void* staged_base = offset_pointer<void>(workspace, data.staged_wavefunction_offset);
  status =
      bind_wavefunction_view(data.wavefunction, staged_base, data.wavefunction.workspace_size_bytes,
                             created.staged_wavefunction, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  created.hamiltonian = offset_pointer<double>(workspace, data.hamiltonian_offset);
  created.shell_charges = offset_pointer<double>(workspace, data.shell_charge_offset);
  created.atomic_charges = offset_pointer<double>(workspace, data.atomic_charge_offset);
  created.atomic_dipoles = offset_pointer<double>(workspace, data.atomic_dipole_offset);
  created.atomic_quadrupoles = offset_pointer<double>(workspace, data.atomic_quadrupole_offset);
  created.component_shell_potential =
      offset_pointer<double>(workspace, data.component_shell_offset);
  created.component_atomic_potential =
      offset_pointer<double>(workspace, data.component_atomic_offset);
  created.component_dipole_potential =
      offset_pointer<double>(workspace, data.component_dipole_offset);
  created.component_quadrupole_potential =
      offset_pointer<double>(workspace, data.component_quadrupole_offset);
  created.atomic_potentials = offset_pointer<double>(workspace, data.atom_potential_offset);
  created.shell_potentials = offset_pointer<double>(workspace, data.shell_potential_offset);
  created.dipole_potentials = offset_pointer<double>(workspace, data.dipole_potential_offset);
  created.quadrupole_potentials =
      offset_pointer<double>(workspace, data.quadrupole_potential_offset);
  created.spin_shell_potentials =
      offset_pointer<double>(workspace, data.spin_shell_potential_offset);
  created.raw_qsh = offset_pointer<double>(workspace, data.raw_qsh_offset);
  created.raw_qat = offset_pointer<double>(workspace, data.raw_qat_offset);
  created.raw_dipoles = offset_pointer<double>(workspace, data.raw_dipole_offset);
  created.raw_quadrupoles = offset_pointer<double>(workspace, data.raw_quadrupole_offset);
  created.core_energies = offset_pointer<double>(workspace, data.core_energy_offset);
  created.es2_energies = offset_pointer<double>(workspace, data.es2_energy_offset);
  created.es3_energies = offset_pointer<double>(workspace, data.es3_energy_offset);
  created.aes2_energies = offset_pointer<double>(workspace, data.aes2_energy_offset);
  created.spin_energies = offset_pointer<double>(workspace, data.spin_energy_offset);
  created.explicit_point_charge_energies =
      offset_pointer<double>(workspace, data.explicit_pc_energy_offset);
  created.internal_energies = offset_pointer<double>(workspace, data.internal_energy_offset);
  created.free_energies = offset_pointer<double>(workspace, data.free_energy_offset);
  created.active_systems = offset_pointer<std::uint8_t>(workspace, data.active_system_offset);

  created.es2_workspace.shell_scratch =
      offset_pointer<double>(workspace, data.es2_shell_scratch_offset);
  created.es2_workspace.shell_elements = data.wavefunction.total_shells;
  created.aes2_workspace.potential_scratch =
      offset_pointer<double>(workspace, data.aes2_potential_scratch_offset);
  created.aes2_workspace.potential_elements = data.aes2.potential_scratch_elements();
  /* Single-system AES2 energy publication needs one canonical batch staging
   * array. Reuse the driver's AES2 component trace: the helper writes exactly
   * the selected system's contribution, which is the value published there
   * after the remaining energy components have passed. */
  created.aes2_workspace.batch_scratch = created.aes2_energies;
  created.aes2_workspace.batch_elements = data.wavefunction.batch_size;
  created.mulliken_workspace.scratch =
      offset_pointer<double>(workspace, data.mulliken_scratch_offset);
  created.mulliken_workspace.elements = std::max(data.mulliken.population_scratch_elements(),
                                                 data.mulliken.hamiltonian_scratch_elements());
  if (data.d4.sealed()) {
    created.d4_atomic_potentials = offset_pointer<double>(workspace, data.d4_potential_offset);
    created.d4_two_body_energies = offset_pointer<double>(workspace, data.d4_energy_offset);
    void* d4_base = offset_pointer<void>(workspace, data.d4_scratch_offset);
    status = bind_d4_workspace(data.d4, d4_base, data.d4.workspace_size_bytes(),
                               created.d4_workspace, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return status;
    }
  }
  if (data.periodic_embedding.sealed()) {
    created.periodic_atomic_potentials =
        offset_pointer<double>(workspace, data.periodic_potential_offset);
    created.periodic_embedding_energies =
        offset_pointer<double>(workspace, data.periodic_energy_offset);
    created.periodic_system_statuses =
        offset_pointer<gpuxtb_status_t>(workspace, data.periodic_status_offset);
    created.periodic_embedding_workspace = {
        offset_pointer<double>(workspace, data.periodic_scratch_offset),
        data.periodic_embedding.maximum_atoms(), data.periodic_embedding.identity()};
  }

  void* eigensolver_base = offset_pointer<void>(workspace, data.eigensolver_scratch_offset);
  status = bind_eigensolver_worker_workspace(data.eigensolver, eigensolver_base,
                                             data.eigensolver.worker_workspace_size_bytes(),
                                             created.eigensolver_workspace, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  void* staged_mixer_base = offset_pointer<void>(workspace, data.staged_mixer_state_offset);
  status = bind_scc_mixer_state(data.mixer, staged_mixer_base, data.mixer.state_size_bytes(),
                                created.staged_mixer_state, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  void* mixer_base = offset_pointer<void>(workspace, data.mixer_scratch_offset);
  status = bind_scc_mixer_workspace(data.mixer, mixer_base, data.mixer.workspace_size_bytes(),
                                    created.mixer_workspace, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  const std::size_t batch = static_cast<std::size_t>(data.wavefunction.batch_size);
  created.thermodynamics = {
      offset_pointer<gpuxtb_status_t>(workspace, data.thermodynamic_status_offset),
      batch,
      offset_pointer<double>(workspace, data.chemical_potential_offset),
      2u * batch,
      offset_pointer<double>(workspace, data.thermodynamic_entropy_offset),
      batch,
      offset_pointer<double>(workspace, data.thermodynamic_band_energy_offset),
      batch,
      offset_pointer<double>(workspace, data.thermodynamic_free_energy_offset),
      batch};
  created.plan_identity = &data;
  view = created;
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t initialize_scc_driver_state_cpu(const SccDriverPlan& plan,
                                                const WavefunctionView& wavefunction,
                                                const SccMixerState& mixer_state,
                                                const SccDriverState& state, std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const SccDriverPlanData& data = *plan.identity();
  if (!exact_state_binding(data, state) || mixer_state.plan_identity != data.mixer.identity()) {
    error = "SCC driver initialization bindings do not belong to the sealed plan";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  status = validate_wavefunction(data, wavefunction, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  std::array<AddressRange, 3> numerical{};
  std::array<AddressRange, 5> controls{};
  if (!make_range(state.workspace_base, data.state_size_bytes, numerical[0]) ||
      !make_range(mixer_state.workspace_base, data.mixer.state_size_bytes(), numerical[1]) ||
      !make_range(wavefunction.workspace_base, data.wavefunction.workspace_size_bytes,
                  numerical[2]) ||
      !make_range(&plan, sizeof(plan), controls[0]) ||
      !make_range(&wavefunction, sizeof(wavefunction), controls[1]) ||
      !make_range(&mixer_state, sizeof(mixer_state), controls[2]) ||
      !make_range(&state, sizeof(state), controls[3]) ||
      !make_range(&error, sizeof(error), controls[4]) || !pairwise_disjoint(numerical) ||
      !pairwise_disjoint(controls) || !disjoint_from_controls(data, numerical, controls)) {
    error = "SCC driver initialization storage overlaps numerical, plan, or descriptor storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  /* The mixer validates every raw multipole before publishing its reset. */
  status = initialize_scc_mixer_state_cpu(data.mixer, wavefunction, mixer_state, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  std::memset(state.workspace_base, 0, data.state_size_bytes);
  const std::size_t batch = static_cast<std::size_t>(data.wavefunction.batch_size);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::fill_n(state.free_energies, batch, nan);
  std::fill_n(state.previous_free_energies, batch, nan);
  std::fill_n(state.free_energy_changes, batch, nan);
  std::fill_n(state.entropies, batch, nan);
  std::fill_n(state.band_energies, batch, nan);
  std::fill_n(state.core_energies, batch, nan);
  std::fill_n(state.es2_energies, batch, nan);
  std::fill_n(state.es3_energies, batch, nan);
  std::fill_n(state.aes2_energies, batch, nan);
  std::fill_n(state.spin_energies, batch, nan);
  if (state.d4_two_body_energies != nullptr) {
    std::fill_n(state.d4_two_body_energies, batch, nan);
  }
  std::fill_n(state.explicit_point_charge_energies, batch, nan);
  if (state.periodic_embedding_energies != nullptr) {
    std::fill_n(state.periodic_embedding_energies, batch, nan);
  }
  std::fill_n(state.internal_energies, batch, nan);
  std::fill_n(state.system_statuses, batch, GPUXTB_STATUS_SUCCESS);
  std::fill_n(state.initialized, batch, static_cast<std::uint8_t>(1u));
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t restart_scc_driver_system_cpu(const SccDriverPlan& plan, std::int64_t system,
                                              const WavefunctionView& wavefunction,
                                              const SccMixerState& mixer_state,
                                              const SccDriverState& state, std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const SccDriverPlanData& data = *plan.identity();
  if (!exact_state_binding(data, state) || mixer_state.plan_identity != data.mixer.identity() ||
      system < 0 || system >= data.wavefunction.batch_size) {
    error = "SCC driver restart bindings or system index are invalid";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  status = validate_wavefunction(data, wavefunction, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const std::size_t index = static_cast<std::size_t>(system);
  if (state.initialized[index] != 1u) {
    error = "SCC driver system must be initialized before restart";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  std::array<AddressRange, 3> numerical{};
  std::array<AddressRange, 5> controls{};
  if (!make_range(state.workspace_base, data.state_size_bytes, numerical[0]) ||
      !make_range(mixer_state.workspace_base, data.mixer.state_size_bytes(), numerical[1]) ||
      !make_range(wavefunction.workspace_base, data.wavefunction.workspace_size_bytes,
                  numerical[2]) ||
      !make_range(&plan, sizeof(plan), controls[0]) ||
      !make_range(&wavefunction, sizeof(wavefunction), controls[1]) ||
      !make_range(&mixer_state, sizeof(mixer_state), controls[2]) ||
      !make_range(&state, sizeof(state), controls[3]) ||
      !make_range(&error, sizeof(error), controls[4]) || !pairwise_disjoint(numerical) ||
      !pairwise_disjoint(controls) || !disjoint_from_controls(data, numerical, controls)) {
    error = "SCC driver restart storage overlaps numerical, plan, or descriptor storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  status = restart_scc_mixer_system_cpu(data.mixer, system, wavefunction, mixer_state, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const double nan = std::numeric_limits<double>::quiet_NaN();
  state.free_energies[index] = nan;
  state.previous_free_energies[index] = nan;
  state.free_energy_changes[index] = nan;
  state.entropies[index] = nan;
  state.band_energies[index] = nan;
  state.core_energies[index] = nan;
  state.es2_energies[index] = nan;
  state.es3_energies[index] = nan;
  state.aes2_energies[index] = nan;
  state.spin_energies[index] = nan;
  if (state.d4_two_body_energies != nullptr) {
    state.d4_two_body_energies[index] = nan;
  }
  state.explicit_point_charge_energies[index] = nan;
  if (state.periodic_embedding_energies != nullptr) {
    state.periodic_embedding_energies[index] = nan;
  }
  state.internal_energies[index] = nan;
  state.iterations[index] = 0u;
  state.system_statuses[index] = GPUXTB_STATUS_SUCCESS;
  state.converged[index] = 0u;
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

namespace {

gpuxtb_status_t validate_iteration_bindings(
    const SccDriverPlan& plan, const SccDriverPlanData& data, const SccDriverGeometryView& geometry,
    const CpuLinearAlgebraBackend& backend, const EigensolverOverlapCache& overlap_cache,
    const WavefunctionView& wavefunction, const SccMixerState& mixer_state,
    const SccDriverState& state, const SccDriverWorkspace& workspace, std::string& error) {
  if (!exact_state_binding(data, state) || !exact_workspace_binding(data, workspace) ||
      mixer_state.plan_identity != data.mixer.identity() ||
      overlap_cache.plan_identity != data.eigensolver.identity()) {
    error = "SCC driver runtime bindings do not belong to the sealed plan";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (data.d4.sealed()) {
    D4Workspace canonical_d4_workspace;
    gpuxtb_status_t d4_status =
        bind_d4_workspace(data.d4, workspace.d4_workspace.workspace_base,
                          data.d4.workspace_size_bytes(), canonical_d4_workspace, error);
    if (d4_status != GPUXTB_STATUS_SUCCESS ||
        !same_d4_workspace_binding(workspace.d4_workspace, canonical_d4_workspace)) {
      error = "SCC driver D4 workspace binding is not canonical";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  gpuxtb_status_t status = validate_wavefunction(data, wavefunction, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_wavefunction(data, workspace.staged_wavefunction, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (geometry.geometry_generation == 0u ||
      geometry.h0_elements != data.mulliken.matrix_elements() ||
      !aligned(geometry.h0, alignof(double)) ||
      geometry.integrals.matrix_elements != data.mulliken.matrix_elements() ||
      !aligned(geometry.integrals.overlap, alignof(double)) ||
      !aligned(geometry.integrals.dipole, alignof(double)) ||
      !aligned(geometry.integrals.quadrupole, alignof(double)) ||
      geometry.integrals.plan_identity != data.mulliken.identity() ||
      geometry.es2_cache.plan_identity != data.es2.identity() ||
      geometry.aes2_cache.plan_identity != data.aes2.identity() ||
      geometry.es2_cache.matrix_elements != data.es2.total_matrix_elements() ||
      (geometry.es2_cache.matrix_elements != 0 &&
       !aligned(geometry.es2_cache.coulomb_matrix, alignof(double))) ||
      geometry.aes2_cache.pair_data_elements != data.aes2.pair_data_elements() ||
      (geometry.aes2_cache.pair_data_elements != 0 &&
       !aligned(geometry.aes2_cache.pair_data, alignof(double))) ||
      geometry.es2_cache.geometry_generation != geometry.geometry_generation ||
      geometry.aes2_cache.geometry_generation != geometry.geometry_generation) {
    error = "SCC driver geometry view is stale or belongs to different component plans";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (data.d4.sealed()) {
    if (geometry.d4_cache.plan_identity != data.d4.identity() ||
        geometry.d4_cache.geometry_generation != geometry.geometry_generation ||
        geometry.d4_cache.pair_data_elements !=
            data.d4.total_pairs() * static_cast<std::int64_t>(kD4PairDataElements) ||
        geometry.d4_cache.coordination_elements != data.d4.total_atoms() ||
        !aligned(geometry.d4_cache.pair_data, alignof(double)) ||
        !aligned(geometry.d4_cache.coordination_numbers, alignof(double))) {
      error = "SCC driver D4 cache is stale, malformed, or belongs to another plan";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  } else if (geometry.d4_cache.pair_data != nullptr || geometry.d4_cache.pair_data_elements != 0 ||
             geometry.d4_cache.coordination_numbers != nullptr ||
             geometry.d4_cache.coordination_elements != 0 ||
             geometry.d4_cache.geometry_generation != 0u ||
             geometry.d4_cache.plan_identity != nullptr) {
    error = "SCC driver geometry supplies D4 data to a plan without D4";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (geometry.explicit_point_charge_shell_elements != 0 &&
      (geometry.explicit_point_charge_shell_elements != data.wavefunction.total_shells ||
       !aligned(geometry.explicit_point_charge_shell_potential, alignof(double)))) {
    error = "SCC driver explicit point-charge potential has invalid extent";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (data.periodic_embedding.sealed()) {
    if (geometry.periodic_shift_elements != data.periodic_embedding.total_atoms() ||
        geometry.periodic_response_elements != data.periodic_embedding.total_matrix_elements() ||
        (geometry.periodic_shift_elements != 0 &&
         !aligned(geometry.periodic_shifts, alignof(double))) ||
        (geometry.periodic_response_elements != 0 &&
         !aligned(geometry.periodic_response_matrices, alignof(double))) ||
        geometry.periodic_embedding_generation == 0u ||
        geometry.periodic_plan_identity != data.periodic_embedding.identity()) {
      error = "SCC driver periodic embedding is stale, malformed, or belongs to another plan";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  } else if (geometry.periodic_shifts != nullptr || geometry.periodic_shift_elements != 0 ||
             geometry.periodic_response_matrices != nullptr ||
             geometry.periodic_response_elements != 0 ||
             geometry.periodic_embedding_generation != 0u ||
             geometry.periodic_plan_identity != nullptr) {
    error = "SCC driver geometry supplies periodic data to a plan without periodic embedding";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  const std::size_t batch = static_cast<std::size_t>(data.wavefunction.batch_size);
  std::array<AddressRange, 5> principal{};
  std::array<AddressRange, 9> controls{};
  if (!make_range(state.workspace_base, data.state_size_bytes, principal[0]) ||
      !make_range(workspace.workspace_base, data.workspace_size_bytes, principal[1]) ||
      !make_range(mixer_state.workspace_base, data.mixer.state_size_bytes(), principal[2]) ||
      !make_range(wavefunction.workspace_base, data.wavefunction.workspace_size_bytes,
                  principal[3]) ||
      !make_range(overlap_cache.workspace_base, data.eigensolver.overlap_cache_size_bytes(),
                  principal[4]) ||
      !make_range(&plan, sizeof(plan), controls[0]) ||
      !make_range(&geometry, sizeof(geometry), controls[1]) ||
      !make_range(&backend, sizeof(backend), controls[2]) ||
      !make_range(&overlap_cache, sizeof(overlap_cache), controls[3]) ||
      !make_range(&wavefunction, sizeof(wavefunction), controls[4]) ||
      !make_range(&mixer_state, sizeof(mixer_state), controls[5]) ||
      !make_range(&state, sizeof(state), controls[6]) ||
      !make_range(&workspace, sizeof(workspace), controls[7]) ||
      !make_range(&error, sizeof(error), controls[8]) || !pairwise_disjoint(principal) ||
      !pairwise_disjoint(controls) || !disjoint_from_controls(data, principal, controls)) {
    error = "SCC driver runtime storage overlaps numerical, plan, or descriptor storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  AddressRange mixer_storage;
  AddressRange mixer_initialized;
  AddressRange mixer_converged;
  if (!make_range(mixer_state.workspace_base, data.mixer.state_size_bytes(), mixer_storage) ||
      !make_range(mixer_state.initialized, batch * sizeof(std::uint8_t), mixer_initialized) ||
      !make_range(mixer_state.converged, batch * sizeof(std::uint8_t), mixer_converged) ||
      !range_contains(mixer_storage, mixer_initialized) ||
      !range_contains(mixer_storage, mixer_converged)) {
    error = "SCC driver mixer state pointers are outside their caller-owned binding";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (std::size_t system = 0u; system < batch; ++system) {
    if (state.initialized[system] != 1u || mixer_state.initialized[system] != 1u ||
        state.converged[system] > 1u || mixer_state.converged[system] > 1u) {
      error = "SCC driver requires initialized canonical per-system state";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  std::size_t matrix_bytes = 0u;
  std::size_t dipole_integral_bytes = 0u;
  std::size_t quadrupole_integral_bytes = 0u;
  std::size_t es2_cache_bytes = 0u;
  std::size_t aes2_cache_bytes = 0u;
  std::size_t d4_pair_cache_bytes = 0u;
  std::size_t d4_coordination_cache_bytes = 0u;
  std::size_t point_potential_bytes = 0u;
  std::size_t periodic_shift_bytes = 0u;
  std::size_t periodic_response_bytes = 0u;
  if (!bytes_for(data.mulliken.matrix_elements(), sizeof(double), matrix_bytes) ||
      !checked_multiply_size(matrix_bytes, 3u, dipole_integral_bytes) ||
      !checked_multiply_size(matrix_bytes, 6u, quadrupole_integral_bytes) ||
      !bytes_for(data.es2.total_matrix_elements(), sizeof(double), es2_cache_bytes) ||
      !bytes_for(data.aes2.pair_data_elements(), sizeof(double), aes2_cache_bytes) ||
      !bytes_for(geometry.d4_cache.pair_data_elements, sizeof(double), d4_pair_cache_bytes) ||
      !bytes_for(geometry.d4_cache.coordination_elements, sizeof(double),
                 d4_coordination_cache_bytes) ||
      !bytes_for(geometry.explicit_point_charge_shell_elements, sizeof(double),
                 point_potential_bytes) ||
      !bytes_for(geometry.periodic_shift_elements, sizeof(double), periodic_shift_bytes) ||
      !bytes_for(geometry.periodic_response_elements, sizeof(double), periodic_response_bytes)) {
    error = "SCC driver geometry storage extents are not representable";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  std::array<AddressRange, 11> geometry_ranges{};
  if (!make_range(geometry.h0, matrix_bytes, geometry_ranges[0]) ||
      !make_range(geometry.integrals.overlap, matrix_bytes, geometry_ranges[1]) ||
      !make_range(geometry.integrals.dipole, dipole_integral_bytes, geometry_ranges[2]) ||
      !make_range(geometry.integrals.quadrupole, quadrupole_integral_bytes, geometry_ranges[3]) ||
      !make_range(geometry.es2_cache.coulomb_matrix, es2_cache_bytes, geometry_ranges[4]) ||
      !make_range(geometry.aes2_cache.pair_data, aes2_cache_bytes, geometry_ranges[5]) ||
      !make_range(geometry.d4_cache.pair_data, d4_pair_cache_bytes, geometry_ranges[6]) ||
      !make_range(geometry.d4_cache.coordination_numbers, d4_coordination_cache_bytes,
                  geometry_ranges[7]) ||
      !make_range(geometry.explicit_point_charge_shell_potential, point_potential_bytes,
                  geometry_ranges[8]) ||
      !make_range(geometry.periodic_shifts, periodic_shift_bytes, geometry_ranges[9]) ||
      !make_range(geometry.periodic_response_matrices, periodic_response_bytes,
                  geometry_ranges[10])) {
    error = "SCC driver geometry buffers have invalid address ranges";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (const AddressRange& input : geometry_ranges) {
    if (overlaps_plan_storage(data, input)) {
      error = "SCC driver geometry inputs must not overlap immutable plan storage";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t output = 0u; output < 4u; ++output) {
      if (ranges_overlap(input, principal[output])) {
        error = "SCC driver geometry inputs must not overlap mutable state or scratch";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
    }
    for (const AddressRange& control : controls) {
      if (ranges_overlap(input, control)) {
        error = "SCC driver geometry inputs must not overlap descriptor storage";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

bool add_finite(double contribution, double& target) {
  const double updated = target + contribution;
  if (!std::isfinite(contribution) || !std::isfinite(updated)) {
    return false;
  }
  target = updated;
  return true;
}

gpuxtb_status_t evaluate_scc_energy_system(const SccDriverPlanData& data,
                                           const SccDriverGeometryView& geometry,
                                           std::size_t system, const SccDriverWorkspace& workspace,
                                           std::string& error) {
  const WavefunctionLayout& layout = data.wavefunction;
  const std::int64_t atom_begin = layout.atom_offsets[system];
  const std::int64_t atom_end = layout.atom_offsets[system + 1u];
  const std::int64_t shell_begin = layout.batch_shell_offsets[system];
  const std::int64_t shell_end = layout.batch_shell_offsets[system + 1u];
  const std::int64_t atoms = atom_end - atom_begin;
  const std::int64_t shells = shell_end - shell_begin;
  const std::int64_t qsh_base = layout.qsh.system_offsets[system];
  const std::int64_t qat_base = layout.qat.system_offsets[system];
  const std::int64_t dipole_base = layout.dipole.system_offsets[system];
  const std::int64_t quadrupole_base = layout.quadrupole.system_offsets[system];

  /* Component APIs consume topology-major arrays. Mulliken outputs retain the
   * wavefunction field packing, so flatten only this target system after the
   * Hamiltonian no longer needs the mixed inputs. */
  for (std::int64_t local_shell = 0; local_shell < shells; ++local_shell) {
    workspace.shell_charges[static_cast<std::size_t>(shell_begin + local_shell)] =
        workspace.raw_qsh[static_cast<std::size_t>(qsh_base + local_shell)];
  }
  for (std::int64_t local_atom = 0; local_atom < atoms; ++local_atom) {
    const std::size_t atom = static_cast<std::size_t>(atom_begin + local_atom);
    workspace.atomic_charges[atom] =
        workspace.raw_qat[static_cast<std::size_t>(qat_base + local_atom)];
    std::copy_n(workspace.raw_dipoles + dipole_base + local_atom * 3, 3u,
                workspace.atomic_dipoles + atom * 3u);
    std::copy_n(workspace.raw_quadrupoles + quadrupole_base + local_atom * 6, 6u,
                workspace.atomic_quadrupoles + atom * 6u);
  }

  double core_energy = 0.0;
  const std::int64_t matrix_begin = data.mulliken.matrix_offsets()[system];
  const std::int64_t matrix_end = data.mulliken.matrix_offsets()[system + 1u];
  const std::int64_t matrix_elements = matrix_end - matrix_begin;
  const std::int64_t density_base = layout.density.system_offsets[system];
  for (std::int32_t spin = 0; spin < layout.spin_channels[system]; ++spin) {
    for (std::int64_t matrix = 0; matrix < matrix_elements; ++matrix) {
      core_energy =
          std::fma(geometry.h0[static_cast<std::size_t>(matrix_begin + matrix)],
                   workspace.staged_wavefunction.density[static_cast<std::size_t>(
                       density_base + static_cast<std::int64_t>(spin) * matrix_elements + matrix)],
                   core_energy);
      if (!std::isfinite(core_energy)) {
        error = "SCC driver H0 density contraction overflowed";
        return GPUXTB_STATUS_INTERNAL_ERROR;
      }
    }
  }

  double es2_energy = 0.0;
  gpuxtb_status_t status =
      add_es2_energy_system_cpu(data.es2, geometry.es2_cache, static_cast<std::int64_t>(system),
                                workspace.shell_charges, es2_energy, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  double es3_energy = 0.0;
  status = add_es3_energy_system_cpu(make_es3_view(data.es3), static_cast<std::int64_t>(system),
                                     workspace.shell_charges, es3_energy, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  double aes2_energy = 0.0;
  status = add_aes2_energy_system_cpu(data.aes2, geometry.aes2_cache,
                                      static_cast<std::int64_t>(system), workspace.atomic_charges,
                                      workspace.atomic_dipoles, workspace.atomic_quadrupoles,
                                      aes2_energy, workspace.aes2_workspace, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  double spin_energy = 0.0;
  status = add_spin_polarization_energy_system_cpu(make_spin_polarization_view(data.spin),
                                                   static_cast<std::int64_t>(system),
                                                   workspace.raw_qsh, spin_energy, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  double d4_energy = 0.0;
  if (data.d4.sealed()) {
    status = evaluate_d4_two_body_system_cpu(
        data.d4, geometry.d4_cache, static_cast<std::int64_t>(system), workspace.atomic_charges,
        d4_energy, nullptr, workspace.d4_workspace, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return status;
    }
  }

  double explicit_pc_energy = 0.0;
  if (geometry.explicit_point_charge_shell_elements != 0) {
    for (std::int64_t local_shell = 0; local_shell < shells; ++local_shell) {
      explicit_pc_energy =
          std::fma(workspace.shell_charges[static_cast<std::size_t>(shell_begin + local_shell)],
                   geometry.explicit_point_charge_shell_potential[static_cast<std::size_t>(
                       shell_begin + local_shell)],
                   explicit_pc_energy);
      if (!std::isfinite(explicit_pc_energy)) {
        error = "SCC driver explicit point-charge energy overflowed";
        return GPUXTB_STATUS_INTERNAL_ERROR;
      }
    }
  }

  double periodic_energy = 0.0;
  if (data.periodic_embedding.sealed()) {
    const PeriodicEmbeddingView periodic_view{geometry.periodic_shifts,
                                              geometry.periodic_shift_elements,
                                              geometry.periodic_response_matrices,
                                              geometry.periodic_response_elements,
                                              workspace.atomic_charges,
                                              layout.total_atoms,
                                              workspace.periodic_atomic_potentials,
                                              layout.total_atoms,
                                              workspace.periodic_embedding_energies,
                                              layout.batch_size,
                                              workspace.periodic_system_statuses,
                                              layout.batch_size,
                                              data.periodic_embedding.identity()};
    status = evaluate_periodic_embedding_system_cpu(
        data.periodic_embedding, static_cast<std::int64_t>(system), periodic_view,
        workspace.periodic_embedding_workspace, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return status;
    }
    periodic_energy = workspace.periodic_embedding_energies[system];
  }

  double internal_energy = core_energy;
  if (!add_finite(es2_energy, internal_energy) || !add_finite(es3_energy, internal_energy) ||
      !add_finite(aes2_energy, internal_energy) || !add_finite(spin_energy, internal_energy) ||
      !add_finite(d4_energy, internal_energy) || !add_finite(explicit_pc_energy, internal_energy) ||
      !add_finite(periodic_energy, internal_energy)) {
    error = "SCC driver complete internal energy overflowed";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  const double entropy = workspace.thermodynamics.entropies[system];
  const double free_energy = std::fma(-data.electronic_temperature, entropy, internal_energy);
  if (!std::isfinite(entropy) || !std::isfinite(free_energy)) {
    error = "SCC driver complete free energy is not finite";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }

  workspace.core_energies[system] = core_energy;
  workspace.es2_energies[system] = es2_energy;
  workspace.es3_energies[system] = es3_energy;
  workspace.aes2_energies[system] = aes2_energy;
  workspace.spin_energies[system] = spin_energy;
  if (workspace.d4_two_body_energies != nullptr) {
    workspace.d4_two_body_energies[system] = d4_energy;
  }
  workspace.explicit_point_charge_energies[system] = explicit_pc_energy;
  workspace.internal_energies[system] = internal_energy;
  workspace.free_energies[system] = free_energy;
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t gather_mixed_multipoles(const SccDriverPlanData& data,
                                        const SccDriverWorkspace& workspace, std::string& error) {
  const WavefunctionLayout& layout = data.wavefunction;
  const std::size_t batch = static_cast<std::size_t>(layout.batch_size);
  std::fill_n(workspace.staged_wavefunction.qat, static_cast<std::size_t>(layout.qat.element_count),
              0.0);
  for (std::size_t system = 0u; system < batch; ++system) {
    const std::int64_t atom_begin = layout.atom_offsets[system];
    const std::int64_t atom_end = layout.atom_offsets[system + 1u];
    const std::int64_t shell_begin = layout.batch_shell_offsets[system];
    const std::int64_t shell_end = layout.batch_shell_offsets[system + 1u];
    const std::int64_t atoms = atom_end - atom_begin;
    const std::int64_t shells = shell_end - shell_begin;
    const std::int32_t channels = layout.spin_channels[system];
    const std::int64_t qsh_base = layout.qsh.system_offsets[system];
    const std::int64_t qat_base = layout.qat.system_offsets[system];
    const std::int64_t dipole_base = layout.dipole.system_offsets[system];
    const std::int64_t quadrupole_base = layout.quadrupole.system_offsets[system];

    for (std::int32_t channel = 0; channel < channels; ++channel) {
      for (std::int64_t local_shell = 0; local_shell < shells; ++local_shell) {
        const std::int64_t shell = shell_begin + local_shell;
        const std::int64_t local_atom =
            data.mulliken.shell_to_atom()[static_cast<std::size_t>(shell)] - atom_begin;
        const double charge = workspace.staged_wavefunction.qsh[static_cast<std::size_t>(
            qsh_base + static_cast<std::int64_t>(channel) * shells + local_shell)];
        double& atomic_charge = workspace.staged_wavefunction.qat[static_cast<std::size_t>(
            qat_base + static_cast<std::int64_t>(channel) * atoms + local_atom)];
        if (!add_finite(charge, atomic_charge)) {
          error = "SCC driver mixed shell-to-atom reduction is not finite";
          return GPUXTB_STATUS_INTERNAL_ERROR;
        }
      }
    }

    for (std::int64_t local_shell = 0; local_shell < shells; ++local_shell) {
      const double value =
          workspace.staged_wavefunction.qsh[static_cast<std::size_t>(qsh_base + local_shell)];
      if (!std::isfinite(value)) {
        error = "SCC driver mixed shell charges contain NaN or infinity";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      workspace.shell_charges[static_cast<std::size_t>(shell_begin + local_shell)] = value;
    }
    for (std::int64_t local_atom = 0; local_atom < atoms; ++local_atom) {
      const std::size_t atom = static_cast<std::size_t>(atom_begin + local_atom);
      const double charge =
          workspace.staged_wavefunction.qat[static_cast<std::size_t>(qat_base + local_atom)];
      if (!std::isfinite(charge)) {
        error = "SCC driver mixed atomic charges contain NaN or infinity";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      workspace.atomic_charges[atom] = charge;
      for (std::size_t component = 0u; component < 3u; ++component) {
        const double value = workspace.staged_wavefunction.dipole[static_cast<std::size_t>(
            dipole_base + local_atom * 3 + static_cast<std::int64_t>(component))];
        if (!std::isfinite(value)) {
          error = "SCC driver mixed atomic dipoles contain NaN or infinity";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        workspace.atomic_dipoles[atom * 3u + component] = value;
      }
      for (std::size_t component = 0u; component < 6u; ++component) {
        const double value = workspace.staged_wavefunction.quadrupole[static_cast<std::size_t>(
            quadrupole_base + local_atom * 6 + static_cast<std::int64_t>(component))];
        if (!std::isfinite(value)) {
          error = "SCC driver mixed atomic quadrupoles contain NaN or infinity";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        workspace.atomic_quadrupoles[atom * 6u + component] = value;
      }
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t prepare_potentials_and_hamiltonian(const SccDriverPlanData& data,
                                                   const SccDriverGeometryView& geometry,
                                                   const SccDriverWorkspace& workspace,
                                                   std::string& error) {
  const WavefunctionLayout& layout = data.wavefunction;
  gpuxtb_status_t status = gather_mixed_multipoles(data, workspace, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  std::fill_n(workspace.atomic_potentials, static_cast<std::size_t>(layout.qat.element_count), 0.0);
  std::fill_n(workspace.shell_potentials, static_cast<std::size_t>(layout.qsh.element_count), 0.0);
  std::fill_n(workspace.dipole_potentials, static_cast<std::size_t>(layout.dipole.element_count),
              0.0);
  std::fill_n(workspace.quadrupole_potentials,
              static_cast<std::size_t>(layout.quadrupole.element_count), 0.0);
  const std::size_t batch = static_cast<std::size_t>(layout.batch_size);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  if (data.periodic_embedding.sealed()) {
    std::fill_n(workspace.periodic_atomic_potentials, static_cast<std::size_t>(layout.total_atoms),
                0.0);
    std::fill_n(workspace.periodic_embedding_energies, batch, nan);
    std::fill_n(workspace.periodic_system_statuses, batch, GPUXTB_STATUS_INVALID_ARGUMENT);
  }
  status = evaluate_spin_polarization_cpu(
      make_spin_polarization_view(data.spin), workspace.staged_wavefunction.qsh,
      workspace.spin_energies, workspace.spin_shell_potentials, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = evaluate_es2_potential_cpu(data.es2, geometry.es2_cache, workspace.shell_charges,
                                      workspace.component_shell_potential, workspace.es2_workspace,
                                      error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const ES3View es3_view = make_es3_view(data.es3);
  for (std::size_t system = 0u; system < batch; ++system) {
    const std::int64_t shell_begin = layout.batch_shell_offsets[system];
    const std::int64_t shell_end = layout.batch_shell_offsets[system + 1u];
    const std::int64_t shells = shell_end - shell_begin;
    const std::int64_t destination_base = layout.qsh.system_offsets[system];
    for (std::int64_t local_shell = 0; local_shell < shells; ++local_shell) {
      workspace.shell_potentials[static_cast<std::size_t>(destination_base + local_shell)] =
          workspace.component_shell_potential[static_cast<std::size_t>(shell_begin + local_shell)];
    }
  }
  status = evaluate_es3_potential_cpu(es3_view, workspace.shell_charges,
                                      workspace.component_shell_potential, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  for (std::size_t system = 0u; system < batch; ++system) {
    const std::int64_t shell_begin = layout.batch_shell_offsets[system];
    const std::int64_t shell_end = layout.batch_shell_offsets[system + 1u];
    const std::int64_t shells = shell_end - shell_begin;
    const std::int64_t destination_base = layout.qsh.system_offsets[system];
    for (std::int64_t local_shell = 0; local_shell < shells; ++local_shell) {
      double& target =
          workspace.shell_potentials[static_cast<std::size_t>(destination_base + local_shell)];
      if (!add_finite(
              workspace
                  .component_shell_potential[static_cast<std::size_t>(shell_begin + local_shell)],
              target)) {
        error = "SCC driver ES2+ES3 shell potential exceeded floating-point range";
        return GPUXTB_STATUS_INTERNAL_ERROR;
      }
      if (geometry.explicit_point_charge_shell_elements != 0 &&
          !add_finite(geometry.explicit_point_charge_shell_potential[static_cast<std::size_t>(
                          shell_begin + local_shell)],
                      target)) {
        error = "SCC driver explicit point-charge potential is not finite";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  for (std::int64_t element = 0; element < layout.qsh.element_count; ++element) {
    if (!add_finite(workspace.spin_shell_potentials[static_cast<std::size_t>(element)],
                    workspace.shell_potentials[static_cast<std::size_t>(element)])) {
      error = "SCC driver electrostatic+spin shell potential exceeded floating-point range";
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }
  }

  status = evaluate_aes2_potential_cpu(
      data.aes2, geometry.aes2_cache, workspace.atomic_charges, workspace.atomic_dipoles,
      workspace.atomic_quadrupoles, workspace.component_atomic_potential,
      workspace.component_dipole_potential, workspace.component_quadrupole_potential,
      workspace.aes2_workspace, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (data.d4.sealed()) {
    status = evaluate_d4_two_body_cpu(
        data.d4, geometry.d4_cache, workspace.atomic_charges, workspace.d4_two_body_energies,
        workspace.d4_atomic_potentials, workspace.d4_workspace, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return status;
    }
  }
  if (data.periodic_embedding.sealed()) {
    const PeriodicEmbeddingView periodic_view{geometry.periodic_shifts,
                                              geometry.periodic_shift_elements,
                                              geometry.periodic_response_matrices,
                                              geometry.periodic_response_elements,
                                              workspace.atomic_charges,
                                              layout.total_atoms,
                                              workspace.periodic_atomic_potentials,
                                              layout.total_atoms,
                                              workspace.periodic_embedding_energies,
                                              layout.batch_size,
                                              workspace.periodic_system_statuses,
                                              layout.batch_size,
                                              data.periodic_embedding.identity()};
    for (std::size_t system = 0u; system < batch; ++system) {
      if (workspace.active_systems[system] != 1u) {
        continue;
      }
      status = evaluate_periodic_embedding_system_cpu(
          data.periodic_embedding, static_cast<std::int64_t>(system), periodic_view,
          workspace.periodic_embedding_workspace, error);
      if (status == GPUXTB_STATUS_INVALID_ARGUMENT) {
        return status;
      }
      if (status != GPUXTB_STATUS_SUCCESS) {
        /* The component guarantees unchanged numerical outputs on failure.
         * Keep an explicit zero potential so the later whole-batch Mulliken
         * assembly remains safe while successful peers continue. */
        const std::int64_t atom_begin = layout.atom_offsets[system];
        const std::int64_t atom_end = layout.atom_offsets[system + 1u];
        std::fill_n(workspace.periodic_atomic_potentials + atom_begin,
                    static_cast<std::size_t>(atom_end - atom_begin), 0.0);
        workspace.periodic_embedding_energies[system] = nan;
        workspace.active_systems[system] = 5u;
      }
    }
  }
  for (std::size_t system = 0u; system < batch; ++system) {
    const std::int64_t atom_begin = layout.atom_offsets[system];
    const std::int64_t atom_end = layout.atom_offsets[system + 1u];
    const std::int64_t atoms = atom_end - atom_begin;
    const std::int64_t qat_base = layout.qat.system_offsets[system];
    const std::int64_t dipole_base = layout.dipole.system_offsets[system];
    const std::int64_t quadrupole_base = layout.quadrupole.system_offsets[system];
    for (std::int64_t local_atom = 0; local_atom < atoms; ++local_atom) {
      const std::size_t atom = static_cast<std::size_t>(atom_begin + local_atom);
      workspace.atomic_potentials[static_cast<std::size_t>(qat_base + local_atom)] =
          workspace.component_atomic_potential[atom];
      for (std::size_t component = 0u; component < 3u; ++component) {
        workspace.dipole_potentials[static_cast<std::size_t>(
            dipole_base + local_atom * 3 + static_cast<std::int64_t>(component))] =
            workspace.component_dipole_potential[atom * 3u + component];
      }
      for (std::size_t component = 0u; component < 6u; ++component) {
        workspace.quadrupole_potentials[static_cast<std::size_t>(
            quadrupole_base + local_atom * 6 + static_cast<std::int64_t>(component))] =
            workspace.component_quadrupole_potential[atom * 6u + component];
      }
    }
    if (data.periodic_embedding.sealed() && workspace.active_systems[system] == 1u) {
      bool finite = true;
      for (std::int64_t local_atom = 0; local_atom < atoms; ++local_atom) {
        const std::size_t atom = static_cast<std::size_t>(atom_begin + local_atom);
        double& target =
            workspace.atomic_potentials[static_cast<std::size_t>(qat_base + local_atom)];
        if (!add_finite(workspace.periodic_atomic_potentials[atom], target)) {
          finite = false;
          break;
        }
      }
      if (!finite) {
        /* Restore the complete AES2 atomic potential for this failed system;
         * Mulliken Hamiltonian construction is still a whole-batch primitive. */
        for (std::int64_t local_atom = 0; local_atom < atoms; ++local_atom) {
          const std::size_t atom = static_cast<std::size_t>(atom_begin + local_atom);
          workspace.atomic_potentials[static_cast<std::size_t>(qat_base + local_atom)] =
              workspace.component_atomic_potential[atom];
        }
        std::fill_n(workspace.periodic_atomic_potentials + atom_begin,
                    static_cast<std::size_t>(atoms), 0.0);
        workspace.periodic_embedding_energies[system] = nan;
        workspace.periodic_system_statuses[system] = GPUXTB_STATUS_INTERNAL_ERROR;
        workspace.active_systems[system] = 5u;
      }
    }
    if (data.d4.sealed() && workspace.active_systems[system] == 1u) {
      for (std::int64_t local_atom = 0; local_atom < atoms; ++local_atom) {
        const std::size_t atom = static_cast<std::size_t>(atom_begin + local_atom);
        double& target =
            workspace.atomic_potentials[static_cast<std::size_t>(qat_base + local_atom)];
        if (!add_finite(workspace.d4_atomic_potentials[atom], target)) {
          error = "SCC driver AES2+embedding+D4 atom potential exceeded floating-point range";
          return GPUXTB_STATUS_INTERNAL_ERROR;
        }
      }
    }
  }

  for (std::int64_t element = 0; element < geometry.h0_elements; ++element) {
    if (!std::isfinite(geometry.h0[static_cast<std::size_t>(element)])) {
      error = "SCC driver H0 contains NaN or infinity";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::size_t system = 0u; system < batch; ++system) {
    const std::int64_t matrix_begin = data.mulliken.matrix_offsets()[system];
    const std::int64_t matrix_end = data.mulliken.matrix_offsets()[system + 1u];
    const std::int64_t matrix_elements = matrix_end - matrix_begin;
    const std::int64_t hamiltonian_base = layout.density.system_offsets[system];
    for (std::int32_t spin = 0; spin < layout.spin_channels[system]; ++spin) {
      std::copy_n(geometry.h0 + matrix_begin, static_cast<std::size_t>(matrix_elements),
                  workspace.hamiltonian + hamiltonian_base +
                      static_cast<std::int64_t>(spin) * matrix_elements);
    }
  }

  const MullikenPotentialView potential{
      workspace.atomic_potentials,     layout.qat.element_count,        workspace.shell_potentials,
      layout.qsh.element_count,        workspace.dipole_potentials,     layout.dipole.element_count,
      workspace.quadrupole_potentials, layout.quadrupole.element_count, data.mulliken.identity()};
  const MullikenHamiltonianView hamiltonian{workspace.hamiltonian, layout.density.element_count,
                                            data.mulliken.identity()};
  status = add_mulliken_hamiltonian_cpu(data.mulliken, geometry.integrals, potential, hamiltonian,
                                        workspace.mulliken_workspace, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  /*
   * Mulliken converts charge/magnetization potentials with a one-half factor.
   * tblite subsequently doubles its complete converted Hamiltonian, whose core
   * channel entered that conversion only once. gpuxtb starts both spin channels
   * from H0, so reproduce the same physical operator as H0 + 2*(Htmp-H0);
   * doubling Htmp directly would incorrectly double the core Hamiltonian.
   */
  for (std::size_t system = 0u; system < batch; ++system) {
    if (layout.spin_channels[system] != 2) {
      continue;
    }
    const std::int64_t matrix_begin = data.mulliken.matrix_offsets()[system];
    const std::int64_t matrix_end = data.mulliken.matrix_offsets()[system + 1u];
    const std::int64_t matrix_elements = matrix_end - matrix_begin;
    const std::int64_t hamiltonian_base = layout.density.system_offsets[system];
    for (std::int32_t spin = 0; spin < 2; ++spin) {
      for (std::int64_t matrix = 0; matrix < matrix_elements; ++matrix) {
        const double h0 = geometry.h0[static_cast<std::size_t>(matrix_begin + matrix)];
        double& target = workspace.hamiltonian[static_cast<std::size_t>(
            hamiltonian_base + static_cast<std::int64_t>(spin) * matrix_elements + matrix)];
        const double physical = std::fma(2.0, target - h0, h0);
        if (!std::isfinite(physical)) {
          error = "SCC driver unrestricted Hamiltonian scaling overflowed";
          return GPUXTB_STATUS_INTERNAL_ERROR;
        }
        target = physical;
      }
    }
  }
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

void copy_raw_population_system(const WavefunctionLayout& layout, std::size_t system,
                                const SccDriverWorkspace& workspace) {
  copy_system_field(layout.qsh, system, workspace.raw_qsh, workspace.staged_wavefunction.qsh);
  copy_system_field(layout.qat, system, workspace.raw_qat, workspace.staged_wavefunction.qat);
  copy_system_field(layout.dipole, system, workspace.raw_dipoles,
                    workspace.staged_wavefunction.dipole);
  copy_system_field(layout.quadrupole, system, workspace.raw_quadrupoles,
                    workspace.staged_wavefunction.quadrupole);
}

gpuxtb_status_t rebuild_mixed_atomic_charges(const SccDriverPlanData& data, std::size_t system,
                                             const SccDriverWorkspace& workspace,
                                             std::string& error) {
  const WavefunctionLayout& layout = data.wavefunction;
  const std::int64_t atom_begin = layout.atom_offsets[system];
  const std::int64_t atom_end = layout.atom_offsets[system + 1u];
  const std::int64_t shell_begin = layout.batch_shell_offsets[system];
  const std::int64_t shell_end = layout.batch_shell_offsets[system + 1u];
  const std::int64_t atoms = atom_end - atom_begin;
  const std::int64_t shells = shell_end - shell_begin;
  const std::int64_t qsh_base = layout.qsh.system_offsets[system];
  const std::int64_t qat_base = layout.qat.system_offsets[system];
  std::fill_n(workspace.staged_wavefunction.qat + qat_base,
              static_cast<std::size_t>(layout.qat.system_offsets[system + 1u] - qat_base), 0.0);
  for (std::int32_t channel = 0; channel < layout.spin_channels[system]; ++channel) {
    for (std::int64_t local_shell = 0; local_shell < shells; ++local_shell) {
      const std::int64_t shell = shell_begin + local_shell;
      const std::int64_t local_atom =
          data.mulliken.shell_to_atom()[static_cast<std::size_t>(shell)] - atom_begin;
      const double charge = workspace.staged_wavefunction.qsh[static_cast<std::size_t>(
          qsh_base + static_cast<std::int64_t>(channel) * shells + local_shell)];
      double& target = workspace.staged_wavefunction.qat[static_cast<std::size_t>(
          qat_base + static_cast<std::int64_t>(channel) * atoms + local_atom)];
      if (!add_finite(charge, target)) {
        error = "SCC driver mixed atomic charge reconstruction is not finite";
        return GPUXTB_STATUS_INTERNAL_ERROR;
      }
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

}  // namespace

}  // namespace gpuxtb::detail::gfn2
