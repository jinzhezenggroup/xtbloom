// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn1/scc_driver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace xtbloom::detail::gfn1 {

struct SccDriverPlanData {
  WavefunctionLayout wavefunction;
  MullikenPlan mulliken;
  ES2Plan es2;
  ES3Plan es3;
  SpinPolarizationPlan spin;
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
  std::size_t state_spin_energy_offset = 0u;
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
  std::size_t component_atomic_offset = 0u;
  std::size_t component_shell_offset = 0u;
  std::size_t shell_potential_offset = 0u;
  std::size_t spin_shell_potential_offset = 0u;
  std::size_t raw_qsh_offset = 0u;
  std::size_t raw_qat_offset = 0u;
  std::size_t core_energy_offset = 0u;
  std::size_t es2_energy_offset = 0u;
  std::size_t es3_energy_offset = 0u;
  std::size_t spin_energy_offset = 0u;
  std::size_t explicit_pc_energy_offset = 0u;
  std::size_t periodic_energy_offset = 0u;
  std::size_t internal_energy_offset = 0u;
  std::size_t free_energy_offset = 0u;
  std::size_t periodic_potential_offset = 0u;
  std::size_t periodic_status_offset = 0u;
  std::size_t periodic_scratch_offset = 0u;
  std::size_t es2_shell_scratch_offset = 0u;
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

gfn2::EigensolverWavefunctionLayout make_eigensolver_wavefunction_layout(
    const WavefunctionLayout& layout) noexcept {
  gfn2::EigensolverWavefunctionLayout projected;
  projected.batch_size = layout.batch_size;
  projected.workspace_size_bytes = layout.workspace_size_bytes;
  projected.orbital_offsets = layout.batch_orbital_offsets.data();
  projected.orbital_offset_count = layout.batch_orbital_offsets.size();
  projected.spin_channels = layout.spin_channels.data();
  projected.spin_channel_count = layout.spin_channels.size();
  projected.alpha_electron_counts = layout.alpha_electron_counts.data();
  projected.beta_electron_counts = layout.beta_electron_counts.data();
  projected.electron_count_count = layout.alpha_electron_counts.size();
  const std::array<const WavefunctionFieldLayout*, 5> fields{
      &layout.coefficients, &layout.eigenvalues, &layout.occupations, &layout.density,
      &layout.energy_weighted_density};
  for (std::size_t index = 0u; index < fields.size(); ++index) {
    projected.fields[index] = {fields[index]->offset_bytes, fields[index]->element_count,
                               fields[index]->system_offsets.data(),
                               fields[index]->system_offsets.size()};
  }
  return projected;
}

gfn2::EigensolverWavefunctionView make_eigensolver_wavefunction_view(
    const WavefunctionView& view) noexcept {
  return {view.workspace_base,         view.workspace_size_bytes, view.coefficients,
          view.eigenvalues,            view.occupations,           view.density,
          view.energy_weighted_density};
}

namespace {

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool checked_add_size(std::size_t first, std::size_t second, std::size_t& result) {
  if (first > std::numeric_limits<std::size_t>::max() - second) return false;
  result = first + second;
  return true;
}

bool checked_multiply_size(std::size_t first, std::size_t second, std::size_t& result) {
  if (first != 0u && second > std::numeric_limits<std::size_t>::max() / first) return false;
  result = first * second;
  return true;
}

bool align_up(std::size_t value, std::size_t alignment, std::size_t& result) {
  if (alignment == 0u || (alignment & (alignment - 1u)) != 0u) return false;
  const std::size_t mask = alignment - 1u;
  if (value > std::numeric_limits<std::size_t>::max() - mask) return false;
  result = (value + mask) & ~mask;
  return true;
}

bool append_segment(std::size_t bytes, std::size_t alignment, std::size_t& cursor,
                    std::size_t& offset) {
  return align_up(cursor, alignment, offset) && checked_add_size(offset, bytes, cursor);
}

bool bytes_for(std::int64_t elements, std::size_t element_size, std::size_t& bytes) {
  return elements >= 0 &&
         static_cast<std::uint64_t>(elements) <=
             static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) &&
         checked_multiply_size(static_cast<std::size_t>(elements), element_size, bytes);
}

bool make_range(const void* pointer, std::size_t bytes, AddressRange& range) {
  if (bytes != 0u && pointer == nullptr) return false;
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) return false;
  range = {begin, begin + bytes};
  return true;
}

bool ranges_overlap(const AddressRange& first, const AddressRange& second) {
  return first.begin < second.end && second.begin < first.end;
}

bool range_contains(const AddressRange& outer, const AddressRange& inner) {
  return outer.begin <= inner.begin && inner.end <= outer.end;
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
  const std::array<const std::vector<std::int64_t>*, 17> integer_vectors{{
      &wavefunction.atom_offsets,
      &wavefunction.batch_shell_offsets,
      &wavefunction.batch_orbital_offsets,
      &wavefunction.coefficients.system_offsets,
      &wavefunction.eigenvalues.system_offsets,
      &wavefunction.occupations.system_offsets,
      &wavefunction.density.system_offsets,
      &wavefunction.qsh.system_offsets,
      &wavefunction.qat.system_offsets,
      &wavefunction.energy_weighted_density.system_offsets,
      &data.es3.atom_offsets,
      &data.mixer.vector_offsets(),
      &data.spin.atom_offsets,
      &data.spin.batch_shell_offsets,
      &data.spin.atom_shell_offsets,
      &data.spin.shell_population_offsets,
      &data.spin.coupling_offsets,
  }};
  for (const std::vector<std::int64_t>* values : integer_vectors) {
    if (overlaps_vector(range, *values)) return true;
  }
  const std::array<const std::vector<std::int32_t>*, 4> int32_vectors{{
      &wavefunction.atomic_numbers,
      &wavefunction.unpaired_electrons,
      &wavefunction.spin_channels,
      &data.spin.spin_channels,
  }};
  for (const std::vector<std::int32_t>* values : int32_vectors) {
    if (overlaps_vector(range, *values)) return true;
  }
  const std::array<const std::vector<double>*, 9> double_vectors{{
      &wavefunction.molecular_charges,
      &wavefunction.reference_atom_occupations,
      &wavefunction.reference_shell_occupations,
      &wavefunction.electron_counts,
      &wavefunction.alpha_electron_counts,
      &wavefunction.beta_electron_counts,
      &data.es3.atom_gamma3,
      &data.mulliken.reference_shell_occupations(),
      &data.spin.coupling_matrices,
  }};
  for (const std::vector<double>* values : double_vectors) {
    if (overlaps_vector(range, *values)) return true;
  }
  const std::size_t bytes = static_cast<std::size_t>(range.end - range.begin);
  const void* pointer = reinterpret_cast<const void*>(range.begin);
  return data.mulliken.overlaps_storage(pointer, bytes) ||
         data.es2.overlaps_storage(pointer, bytes) ||
         data.eigensolver.overlaps_storage(pointer, bytes) ||
         data.mixer.overlaps_storage(pointer, bytes) ||
         (data.periodic_embedding.sealed() &&
          data.periodic_embedding.overlaps_storage(pointer, bytes));
}

template <std::size_t N>
bool pairwise_disjoint(const std::array<AddressRange, N>& ranges) {
  for (std::size_t first = 0u; first < N; ++first) {
    for (std::size_t second = first + 1u; second < N; ++second) {
      if (ranges_overlap(ranges[first], ranges[second])) return false;
    }
  }
  return true;
}

template <std::size_t N, std::size_t M>
bool disjoint_from_controls(const SccDriverPlanData& data,
                            const std::array<AddressRange, N>& numerical,
                            const std::array<AddressRange, M>& controls) {
  for (const AddressRange& range : numerical) {
    if (overlaps_plan_storage(data, range)) return false;
    for (const AddressRange& control : controls) {
      if (ranges_overlap(range, control)) return false;
    }
  }
  return true;
}

bool aligned(const void* pointer, std::size_t alignment) {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

template <typename T>
T* offset_pointer(void* base, std::size_t offset) {
  return reinterpret_cast<T*>(static_cast<unsigned char*>(base) + offset);
}

template <typename T>
const T* offset_pointer(const void* base, std::size_t offset) {
  return reinterpret_cast<const T*>(static_cast<const unsigned char*>(base) + offset);
}

bool same_field(const WavefunctionFieldLayout& first, const WavefunctionFieldLayout& second) {
  return first.offset_bytes == second.offset_bytes && first.size_bytes == second.size_bytes &&
         first.element_count == second.element_count &&
         first.system_offsets == second.system_offsets;
}

bool same_wavefunction(const WavefunctionLayout& first, const WavefunctionLayout& second) {
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
         same_field(first.coefficients, second.coefficients) &&
         same_field(first.eigenvalues, second.eigenvalues) &&
         same_field(first.occupations, second.occupations) &&
         same_field(first.density, second.density) && same_field(first.qsh, second.qsh) &&
         same_field(first.qat, second.qat) &&
         same_field(first.energy_weighted_density, second.energy_weighted_density);
}

bool same_eigensolver(const EigensolverPlan& first, const EigensolverPlan& second) {
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

bool same_mixer(const SccMixerPlan& first, const SccMixerPlan& second) {
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

bool same_es3(const ES3Plan& first, const ES3Plan& second) {
  return first.batch_size == second.batch_size && first.total_atoms == second.total_atoms &&
         first.atom_offsets == second.atom_offsets && first.atom_gamma3 == second.atom_gamma3;
}

bool same_spin(const SpinPolarizationPlan& first, const SpinPolarizationPlan& second) {
  return first.batch_size == second.batch_size && first.total_atoms == second.total_atoms &&
         first.total_shells == second.total_shells &&
         first.shell_population_elements == second.shell_population_elements &&
         first.atom_offsets == second.atom_offsets &&
         first.batch_shell_offsets == second.batch_shell_offsets &&
         first.atom_shell_offsets == second.atom_shell_offsets &&
         first.shell_population_offsets == second.shell_population_offsets &&
         first.spin_channels == second.spin_channels &&
         first.coupling_offsets == second.coupling_offsets &&
         first.coupling_matrices == second.coupling_matrices;
}

xtbloom_status_t validate_plan(const SccDriverPlan& plan, std::string& error) {
  if (!plan.sealed() || plan.identity() == nullptr || plan.batch_size() <= 0 ||
      plan.maximum_iterations() == 0u || !std::isfinite(plan.electronic_temperature()) ||
      plan.electronic_temperature() < 0.0 || !std::isfinite(plan.energy_tolerance()) ||
      !(plan.energy_tolerance() > 0.0) || plan.state_size_bytes() == 0u ||
      plan.workspace_size_bytes() == 0u) {
    error = "GFN1 SCC driver plan is not sealed or has invalid metadata";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_wavefunction(const SccDriverPlanData& data,
                                       const WavefunctionView& wavefunction,
                                       std::string& error) {
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
         state.spin_energies ==
             offset_pointer<double>(state.workspace_base, data.state_spin_energy_offset) &&
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
             offset_pointer<xtbloom_status_t>(state.workspace_base, data.state_status_offset) &&
         state.initialized ==
             offset_pointer<std::uint8_t>(state.workspace_base, data.state_initialized_offset) &&
         state.converged ==
             offset_pointer<std::uint8_t>(state.workspace_base, data.state_converged_offset);
}

bool exact_workspace_binding(const SccDriverPlanData& data,
                             const SccDriverWorkspace& workspace) {
  const std::size_t batch = static_cast<std::size_t>(data.wavefunction.batch_size);
  const std::int64_t mulliken_elements =
      std::max(data.mulliken.population_scratch_elements(),
               data.mulliken.hamiltonian_scratch_elements());
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
         workspace.component_atomic_potential ==
             offset_pointer<double>(workspace.workspace_base, data.component_atomic_offset) &&
         workspace.component_shell_potential ==
             offset_pointer<double>(workspace.workspace_base, data.component_shell_offset) &&
         workspace.shell_potentials ==
             offset_pointer<double>(workspace.workspace_base, data.shell_potential_offset) &&
         workspace.spin_shell_potentials ==
             offset_pointer<double>(workspace.workspace_base, data.spin_shell_potential_offset) &&
         workspace.raw_qsh ==
             offset_pointer<double>(workspace.workspace_base, data.raw_qsh_offset) &&
         workspace.raw_qat ==
             offset_pointer<double>(workspace.workspace_base, data.raw_qat_offset) &&
         workspace.core_energies ==
             offset_pointer<double>(workspace.workspace_base, data.core_energy_offset) &&
         workspace.es2_energies ==
             offset_pointer<double>(workspace.workspace_base, data.es2_energy_offset) &&
         workspace.es3_energies ==
             offset_pointer<double>(workspace.workspace_base, data.es3_energy_offset) &&
         workspace.spin_energies ==
             offset_pointer<double>(workspace.workspace_base, data.spin_energy_offset) &&
         workspace.explicit_point_charge_energies ==
             offset_pointer<double>(workspace.workspace_base, data.explicit_pc_energy_offset) &&
         workspace.internal_energies ==
             offset_pointer<double>(workspace.workspace_base, data.internal_energy_offset) &&
         workspace.free_energies ==
             offset_pointer<double>(workspace.workspace_base, data.free_energy_offset) &&
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
               offset_pointer<xtbloom_status_t>(workspace.workspace_base,
                                                data.periodic_status_offset) &&
           workspace.periodic_embedding_workspace.potential_scratch ==
               offset_pointer<double>(workspace.workspace_base, data.periodic_scratch_offset) &&
           workspace.periodic_embedding_workspace.potential_elements ==
               data.periodic_embedding.maximum_atoms() &&
           workspace.periodic_embedding_workspace.plan_identity ==
               data.periodic_embedding.identity())) &&
         workspace.active_systems ==
             offset_pointer<std::uint8_t>(workspace.workspace_base, data.active_system_offset) &&
         workspace.es2_workspace.matrix_scratch == nullptr &&
         workspace.es2_workspace.matrix_elements == 0 &&
         workspace.es2_workspace.shell_scratch ==
             offset_pointer<double>(workspace.workspace_base, data.es2_shell_scratch_offset) &&
         workspace.es2_workspace.shell_elements == data.wavefunction.total_shells &&
         workspace.es2_workspace.batch_scratch == workspace.es2_energies &&
         workspace.es2_workspace.batch_elements == data.wavefunction.batch_size &&
         workspace.es2_workspace.gradient_scratch == nullptr &&
         workspace.es2_workspace.gradient_elements == 0 &&
         workspace.mulliken_workspace.scratch ==
             offset_pointer<double>(workspace.workspace_base, data.mulliken_scratch_offset) &&
         workspace.mulliken_workspace.elements == mulliken_elements &&
         workspace.eigensolver_workspace.workspace_base ==
             offset_pointer<void>(workspace.workspace_base, data.eigensolver_scratch_offset) &&
         workspace.staged_mixer_state.workspace_base ==
             offset_pointer<void>(workspace.workspace_base, data.staged_mixer_state_offset) &&
         workspace.mixer_workspace.workspace_base ==
             offset_pointer<void>(workspace.workspace_base, data.mixer_scratch_offset) &&
         workspace.thermodynamics.system_statuses ==
             offset_pointer<xtbloom_status_t>(workspace.workspace_base,
                                              data.thermodynamic_status_offset) &&
         workspace.thermodynamics.system_status_capacity == batch &&
         workspace.thermodynamics.chemical_potentials ==
             offset_pointer<double>(workspace.workspace_base, data.chemical_potential_offset) &&
         workspace.thermodynamics.chemical_potential_capacity == 2u * batch &&
         workspace.thermodynamics.entropies ==
             offset_pointer<double>(workspace.workspace_base, data.thermodynamic_entropy_offset) &&
         workspace.thermodynamics.entropy_capacity == batch &&
         workspace.thermodynamics.band_energies ==
             offset_pointer<double>(workspace.workspace_base,
                                    data.thermodynamic_band_energy_offset) &&
         workspace.thermodynamics.band_energy_capacity == batch &&
         workspace.thermodynamics.free_energies ==
             offset_pointer<double>(workspace.workspace_base,
                                    data.thermodynamic_free_energy_offset) &&
         workspace.thermodynamics.free_energy_capacity == batch;
}

void copy_field(const WavefunctionFieldLayout& layout, const double* source, double* target) {
  std::copy_n(source, static_cast<std::size_t>(layout.element_count), target);
}

void copy_wavefunction(const WavefunctionLayout& layout, const WavefunctionView& source,
                       const WavefunctionView& target) {
  copy_field(layout.coefficients, source.coefficients, target.coefficients);
  copy_field(layout.eigenvalues, source.eigenvalues, target.eigenvalues);
  copy_field(layout.occupations, source.occupations, target.occupations);
  copy_field(layout.density, source.density, target.density);
  copy_field(layout.qsh, source.qsh, target.qsh);
  copy_field(layout.qat, source.qat, target.qat);
  copy_field(layout.energy_weighted_density, source.energy_weighted_density,
             target.energy_weighted_density);
}

void copy_system_field(const WavefunctionFieldLayout& layout, std::size_t system,
                       const double* source, double* target) {
  const std::int64_t begin = layout.system_offsets[system];
  const std::int64_t end = layout.system_offsets[system + 1u];
  std::copy(source + begin, source + end, target + begin);
}

void commit_system_wavefunction(const WavefunctionLayout& layout, std::size_t system,
                                const WavefunctionView& source, const WavefunctionView& target) {
  copy_system_field(layout.coefficients, system, source.coefficients, target.coefficients);
  copy_system_field(layout.eigenvalues, system, source.eigenvalues, target.eigenvalues);
  copy_system_field(layout.occupations, system, source.occupations, target.occupations);
  copy_system_field(layout.density, system, source.density, target.density);
  copy_system_field(layout.qsh, system, source.qsh, target.qsh);
  copy_system_field(layout.qat, system, source.qat, target.qat);
  copy_system_field(layout.energy_weighted_density, system, source.energy_weighted_density,
                    target.energy_weighted_density);
}

bool add_finite(double contribution, double& target) {
  const double updated = target + contribution;
  if (!std::isfinite(contribution) || !std::isfinite(updated)) return false;
  target = updated;
  return true;
}

void rebuild_atomic_populations(const SccDriverPlanData& data, std::size_t system,
                                const double* shell_populations, double* atomic_populations,
                                bool& valid) {
  const auto& layout = data.wavefunction;
  const std::int64_t atom_begin = layout.atom_offsets[system];
  const std::int64_t atom_end = layout.atom_offsets[system + 1u];
  const std::int64_t shell_begin = layout.batch_shell_offsets[system];
  const std::int64_t shell_end = layout.batch_shell_offsets[system + 1u];
  const std::int64_t atoms = atom_end - atom_begin;
  const std::int64_t shells = shell_end - shell_begin;
  const std::int64_t qsh_begin = layout.qsh.system_offsets[system];
  const std::int64_t qat_begin = layout.qat.system_offsets[system];
  std::fill_n(atomic_populations + qat_begin,
              static_cast<std::size_t>(layout.qat.system_offsets[system + 1u] - qat_begin), 0.0);
  valid = true;
  for (std::int32_t channel = 0; channel < layout.spin_channels[system]; ++channel) {
    for (std::int64_t local_shell = 0; local_shell < shells; ++local_shell) {
      const std::int64_t shell = shell_begin + local_shell;
      const std::int64_t local_atom =
          data.mulliken.shell_to_atom()[static_cast<std::size_t>(shell)] - atom_begin;
      double& target = atomic_populations[qat_begin + channel * atoms + local_atom];
      if (!add_finite(shell_populations[qsh_begin + channel * shells + local_shell], target)) {
        valid = false;
        return;
      }
    }
  }
}

xtbloom_status_t prepare_system(const SccDriverPlanData& data,
                                const SccDriverGeometryView& geometry, std::size_t system,
                                const SccDriverWorkspace& workspace, std::string& error) {
  const auto& layout = data.wavefunction;
  bool valid = true;
  rebuild_atomic_populations(data, system, workspace.staged_wavefunction.qsh,
                             workspace.staged_wavefunction.qat, valid);
  if (!valid) {
    error = "GFN1 SCC mixed shell-to-atom reduction overflowed";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  const std::int64_t atom_begin = layout.atom_offsets[system];
  const std::int64_t atom_end = layout.atom_offsets[system + 1u];
  const std::int64_t shell_begin = layout.batch_shell_offsets[system];
  const std::int64_t shell_end = layout.batch_shell_offsets[system + 1u];
  const std::int64_t atoms = atom_end - atom_begin;
  const std::int64_t shells = shell_end - shell_begin;
  const std::int64_t qsh_begin = layout.qsh.system_offsets[system];
  const std::int64_t qat_begin = layout.qat.system_offsets[system];
  const std::int32_t channels = layout.spin_channels[system];
  const std::int64_t qsh_count = channels * shells;

  for (std::int64_t shell = 0; shell < shells; ++shell) {
    const double charge = workspace.staged_wavefunction.qsh[qsh_begin + shell];
    if (!std::isfinite(charge)) {
      error = "GFN1 SCC mixed shell charges contain NaN or infinity";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    workspace.shell_charges[shell_begin + shell] = charge;
  }
  for (std::int64_t atom = 0; atom < atoms; ++atom) {
    const double charge = workspace.staged_wavefunction.qat[qat_begin + atom];
    if (!std::isfinite(charge)) {
      error = "GFN1 SCC mixed atomic charges contain NaN or infinity";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    workspace.atomic_charges[atom_begin + atom] = charge;
  }

  std::fill_n(workspace.shell_potentials + qsh_begin, static_cast<std::size_t>(qsh_count), 0.0);
  std::fill_n(workspace.spin_shell_potentials + qsh_begin, static_cast<std::size_t>(qsh_count),
              0.0);
  double spin_energy = 0.0;
  xtbloom_status_t status = evaluate_spin_polarization_system_cpu(
      make_spin_polarization_view(data.spin), static_cast<std::int64_t>(system),
      workspace.staged_wavefunction.qsh, spin_energy, workspace.spin_shell_potentials, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  workspace.spin_energies[system] = spin_energy;

  status = evaluate_es2_potential_system_cpu(
      data.es2, geometry.es2_cache, static_cast<std::int64_t>(system), workspace.shell_charges,
      workspace.component_shell_potential, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  for (std::int64_t shell = 0; shell < shells; ++shell) {
    workspace.shell_potentials[qsh_begin + shell] =
        workspace.component_shell_potential[shell_begin + shell];
  }

  status = evaluate_es3_potential_system_cpu(
      make_es3_view(data.es3), static_cast<std::int64_t>(system), workspace.atomic_charges,
      workspace.component_atomic_potential, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  for (std::int64_t shell = 0; shell < shells; ++shell) {
    const std::int64_t atom = data.mulliken.shell_to_atom()[static_cast<std::size_t>(shell_begin + shell)];
    double& potential = workspace.shell_potentials[qsh_begin + shell];
    if (!add_finite(workspace.component_atomic_potential[atom], potential)) {
      error = "GFN1 SCC ES2+ES3 shell potential overflowed";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    if (geometry.explicit_point_charge_shell_elements != 0 &&
        !add_finite(geometry.explicit_point_charge_shell_potential[shell_begin + shell],
                    potential)) {
      error = "GFN1 SCC explicit point-charge potential is not finite";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
  }

  if (data.periodic_embedding.sealed()) {
    const gfn2::PeriodicEmbeddingView periodic_view{
        geometry.periodic_shifts,
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
    status = gfn2::evaluate_periodic_embedding_system_cpu(
        data.periodic_embedding, static_cast<std::int64_t>(system), periodic_view,
        workspace.periodic_embedding_workspace, error);
    if (status == XTBLOOM_STATUS_INVALID_ARGUMENT) return status;
    if (status != XTBLOOM_STATUS_SUCCESS) {
      /* Periodic numerical data belongs to one ragged member. Preserve the
       * component's unchanged-output guarantee and let successful peers
       * continue through the batch transaction. */
      std::fill_n(workspace.periodic_atomic_potentials + atom_begin,
                  static_cast<std::size_t>(atoms), 0.0);
      workspace.periodic_embedding_energies[system] =
          std::numeric_limits<double>::quiet_NaN();
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    for (std::int64_t shell = 0; shell < shells; ++shell) {
      const std::int64_t atom =
          data.mulliken.shell_to_atom()[static_cast<std::size_t>(shell_begin + shell)];
      if (!add_finite(workspace.periodic_atomic_potentials[atom],
                      workspace.shell_potentials[qsh_begin + shell])) {
        error = "GFN1 SCC periodic shell potential overflowed";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
    }
  }

  for (std::int64_t element = 0; element < qsh_count; ++element) {
    if (!add_finite(workspace.spin_shell_potentials[qsh_begin + element],
                    workspace.shell_potentials[qsh_begin + element])) {
      error = "GFN1 SCC scalar+spin shell potential overflowed";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
  }

  const std::int64_t matrix_begin = data.mulliken.matrix_offsets()[system];
  const std::int64_t matrix_end = data.mulliken.matrix_offsets()[system + 1u];
  const std::int64_t matrix_count = matrix_end - matrix_begin;
  const std::int64_t hamiltonian_begin = layout.density.system_offsets[system];
  for (std::int32_t spin = 0; spin < channels; ++spin) {
    std::copy_n(geometry.h0 + matrix_begin, static_cast<std::size_t>(matrix_count),
                workspace.hamiltonian + hamiltonian_begin + spin * matrix_count);
  }
  const MullikenPotentialView potential{workspace.shell_potentials, layout.qsh.element_count,
                                        data.mulliken.identity()};
  const MullikenHamiltonianView hamiltonian{workspace.hamiltonian, layout.density.element_count,
                                            data.mulliken.identity()};
  status = add_mulliken_hamiltonian_system_cpu(
      data.mulliken, geometry.integrals, potential, hamiltonian,
      static_cast<std::int64_t>(system), workspace.mulliken_workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  /* Mulliken uses tblite's half-valued charge/magnetization conversion.
   * Tblite then doubles the complete unrestricted Hamiltonian, whose core
   * channel entered that conversion once. Both xTBloom spin channels already
   * contain full-scale H0, so reproduce the physical operator as
   * H0 + 2*(Htmp-H0) instead of incorrectly doubling the core Hamiltonian. */
  if (channels == 2) {
    for (std::int32_t spin = 0; spin < 2; ++spin) {
      for (std::int64_t matrix = 0; matrix < matrix_count; ++matrix) {
        const double h0 = geometry.h0[matrix_begin + matrix];
        double& target = workspace.hamiltonian[hamiltonian_begin +
                                               static_cast<std::int64_t>(spin) * matrix_count +
                                               matrix];
        const double physical = std::fma(2.0, target - h0, h0);
        if (!std::isfinite(physical)) {
          error = "GFN1 SCC unrestricted Hamiltonian scaling overflowed";
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        target = physical;
      }
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t evaluate_energy(const SccDriverPlanData& data,
                                 const SccDriverGeometryView& geometry, std::size_t system,
                                 const SccDriverWorkspace& workspace, std::string& error) {
  const auto& layout = data.wavefunction;
  const std::int64_t atom_begin = layout.atom_offsets[system];
  const std::int64_t atom_end = layout.atom_offsets[system + 1u];
  const std::int64_t shell_begin = layout.batch_shell_offsets[system];
  const std::int64_t shell_end = layout.batch_shell_offsets[system + 1u];
  const std::int64_t atoms = atom_end - atom_begin;
  const std::int64_t shells = shell_end - shell_begin;
  const std::int64_t qsh_begin = layout.qsh.system_offsets[system];
  const std::int64_t qat_begin = layout.qat.system_offsets[system];
  for (std::int64_t shell = 0; shell < shells; ++shell) {
    workspace.shell_charges[shell_begin + shell] = workspace.raw_qsh[qsh_begin + shell];
  }
  for (std::int64_t atom = 0; atom < atoms; ++atom) {
    workspace.atomic_charges[atom_begin + atom] = workspace.raw_qat[qat_begin + atom];
  }

  double core_energy = 0.0;
  const std::int64_t matrix_begin = data.mulliken.matrix_offsets()[system];
  const std::int64_t matrix_end = data.mulliken.matrix_offsets()[system + 1u];
  const std::int64_t matrix_count = matrix_end - matrix_begin;
  const std::int64_t density_begin = layout.density.system_offsets[system];
  for (std::int32_t spin = 0; spin < layout.spin_channels[system]; ++spin) {
    for (std::int64_t matrix = 0; matrix < matrix_count; ++matrix) {
      core_energy = std::fma(geometry.h0[matrix_begin + matrix],
                             workspace.staged_wavefunction.density[density_begin +
                                                                  spin * matrix_count + matrix],
                             core_energy);
      if (!std::isfinite(core_energy)) {
        error = "GFN1 SCC H0 density contraction overflowed";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
    }
  }
  double es2_energy = 0.0;
  xtbloom_status_t status = add_es2_energy_system_cpu(
      data.es2, geometry.es2_cache, static_cast<std::int64_t>(system), workspace.shell_charges,
      es2_energy, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  double es3_energy = 0.0;
  status = add_es3_energy_system_cpu(make_es3_view(data.es3), static_cast<std::int64_t>(system),
                                     workspace.atomic_charges, es3_energy, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  double spin_energy = 0.0;
  status = add_spin_polarization_energy_system_cpu(
      make_spin_polarization_view(data.spin), static_cast<std::int64_t>(system), workspace.raw_qsh,
      spin_energy, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  double explicit_pc_energy = 0.0;
  if (geometry.explicit_point_charge_shell_elements != 0) {
    for (std::int64_t shell = 0; shell < shells; ++shell) {
      explicit_pc_energy = std::fma(
          workspace.shell_charges[shell_begin + shell],
          geometry.explicit_point_charge_shell_potential[shell_begin + shell],
          explicit_pc_energy);
    }
  }
  double periodic_energy = 0.0;
  if (data.periodic_embedding.sealed()) {
    const gfn2::PeriodicEmbeddingView periodic_view{
        geometry.periodic_shifts,
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
    status = gfn2::evaluate_periodic_embedding_system_cpu(
        data.periodic_embedding, static_cast<std::int64_t>(system), periodic_view,
        workspace.periodic_embedding_workspace, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    periodic_energy = workspace.periodic_embedding_energies[system];
  }

  double internal_energy = core_energy;
  if (!add_finite(es2_energy, internal_energy) || !add_finite(es3_energy, internal_energy) ||
      !add_finite(spin_energy, internal_energy) ||
      !add_finite(explicit_pc_energy, internal_energy) ||
      !add_finite(periodic_energy, internal_energy)) {
    error = "GFN1 SCC complete internal energy overflowed";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  const double entropy = workspace.thermodynamics.entropies[system];
  const double free_energy = std::fma(-data.electronic_temperature, entropy, internal_energy);
  if (!std::isfinite(entropy) || !std::isfinite(free_energy)) {
    error = "GFN1 SCC complete free energy is not finite";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  workspace.core_energies[system] = core_energy;
  workspace.es2_energies[system] = es2_energy;
  workspace.es3_energies[system] = es3_energy;
  workspace.spin_energies[system] = spin_energy;
  workspace.explicit_point_charge_energies[system] = explicit_pc_energy;
  if (workspace.periodic_embedding_energies != nullptr) {
    workspace.periodic_embedding_energies[system] = periodic_energy;
  }
  workspace.internal_energies[system] = internal_energy;
  workspace.free_energies[system] = free_energy;
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace

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
bool SccDriverPlan::periodic_embedding_enabled() const noexcept {
  return data_ != nullptr && data_->periodic_embedding.sealed();
}
std::size_t SccDriverPlan::state_size_bytes() const noexcept {
  return data_ == nullptr ? 0u : data_->state_size_bytes;
}
std::size_t SccDriverPlan::workspace_size_bytes() const noexcept {
  return data_ == nullptr ? 0u : data_->workspace_size_bytes;
}
std::size_t SccDriverPlan::resident_bytes() const noexcept {
  if (data_ == nullptr) return 0u;

  const auto vector_bytes = [](const auto& values) noexcept {
    using Value = typename std::decay_t<decltype(values)>::value_type;
    return values.capacity() * sizeof(Value);
  };
  const WavefunctionLayout& wavefunction = data_->wavefunction;
  std::size_t total = sizeof(*data_) + vector_bytes(wavefunction.atom_offsets) +
                      vector_bytes(wavefunction.batch_shell_offsets) +
                      vector_bytes(wavefunction.batch_orbital_offsets) +
                      vector_bytes(wavefunction.atomic_numbers) +
                      vector_bytes(wavefunction.molecular_charges) +
                      vector_bytes(wavefunction.unpaired_electrons) +
                      vector_bytes(wavefunction.spin_channels) +
                      vector_bytes(wavefunction.reference_atom_occupations) +
                      vector_bytes(wavefunction.reference_shell_occupations) +
                      vector_bytes(wavefunction.electron_counts) +
                      vector_bytes(wavefunction.alpha_electron_counts) +
                      vector_bytes(wavefunction.beta_electron_counts);
  total += vector_bytes(wavefunction.coefficients.system_offsets) +
           vector_bytes(wavefunction.eigenvalues.system_offsets) +
           vector_bytes(wavefunction.occupations.system_offsets) +
           vector_bytes(wavefunction.density.system_offsets) +
           vector_bytes(wavefunction.qsh.system_offsets) +
           vector_bytes(wavefunction.qat.system_offsets) +
           vector_bytes(wavefunction.energy_weighted_density.system_offsets);

  /* ES3 and spin are copied by value into the driver and therefore retain
   * allocations distinct from the executor's direct term plans. Mulliken,
   * ES2, eigensolver, mixer, and periodic handles share immutable backing. */
  total += vector_bytes(data_->es3.atom_offsets) + vector_bytes(data_->es3.atom_gamma3) +
           vector_bytes(data_->spin.atom_offsets) +
           vector_bytes(data_->spin.batch_shell_offsets) +
           vector_bytes(data_->spin.atom_shell_offsets) +
           vector_bytes(data_->spin.shell_population_offsets) +
           vector_bytes(data_->spin.spin_channels) +
           vector_bytes(data_->spin.coupling_offsets) +
           vector_bytes(data_->spin.coupling_matrices);
  return total;
}
const SccDriverPlanData* SccDriverPlan::identity() const noexcept { return data_.get(); }

xtbloom_status_t make_scc_driver_plan(
    const WavefunctionLayout& wavefunction, const MullikenPlan& mulliken, const ES2Plan& es2,
    const ES3Plan& es3, const SpinPolarizationPlan& spin, const EigensolverPlan& eigensolver,
    const SccMixerPlan& mixer, const PeriodicEmbeddingPlan* periodic_embedding,
    std::uint64_t maximum_iterations, double electronic_temperature, double energy_tolerance,
    SccDriverPlan& plan, std::string& error) {
  WavefunctionWarmStartIdentity validated;
  xtbloom_status_t status = make_wavefunction_warm_start_identity(wavefunction, 0u, validated, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (!mulliken.sealed() || !es2.sealed() || !eigensolver.sealed() || !mixer.sealed() ||
      maximum_iterations == 0u || !std::isfinite(electronic_temperature) ||
      electronic_temperature < 0.0 || !std::isfinite(energy_tolerance) ||
      !(energy_tolerance > 0.0) ||
      (periodic_embedding != nullptr && !periodic_embedding->sealed())) {
    error = "GFN1 SCC driver components or numerical policy are invalid";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  BasisPlan expected_basis;
  IntegralPlan expected_integrals;
  WavefunctionLayout expected_wavefunction;
  MullikenPlan expected_mulliken;
  ES2Plan expected_es2;
  ES3Plan expected_es3;
  SpinPopulationLayout expected_spin_layout;
  SpinPolarizationPlan expected_spin;
  EigensolverPlan expected_eigensolver;
  SccMixerPlan expected_mixer;
  status = make_basis_plan(wavefunction.batch_size, wavefunction.total_atoms,
                           wavefunction.atom_offsets.data(), wavefunction.atomic_numbers.data(),
                           expected_basis, error);
  if (status == XTBLOOM_STATUS_SUCCESS)
    status = make_integral_plan(expected_basis, expected_integrals, error);
  if (status == XTBLOOM_STATUS_SUCCESS)
    status = make_wavefunction_layout(expected_basis, wavefunction.atomic_numbers.data(),
                                      wavefunction.molecular_charges.data(),
                                      wavefunction.unpaired_electrons.data(),
                                      wavefunction.spin_channels.data(), expected_wavefunction,
                                      error);
  if (status == XTBLOOM_STATUS_SUCCESS)
    status = make_mulliken_plan(expected_basis, expected_integrals, expected_wavefunction,
                                expected_mulliken, error);
  if (status == XTBLOOM_STATUS_SUCCESS)
    status = gfn1::make_es2_plan(expected_basis, wavefunction.atomic_numbers.data(), expected_es2,
                                 error);
  if (status == XTBLOOM_STATUS_SUCCESS)
    status = make_es3_plan(expected_basis, wavefunction.atomic_numbers.data(), expected_es3, error);
  if (status == XTBLOOM_STATUS_SUCCESS)
    status = make_spin_population_layout(expected_basis, wavefunction.spin_channels.data(),
                                         expected_spin_layout, error);
  if (status == XTBLOOM_STATUS_SUCCESS)
    status = make_spin_polarization_plan(expected_basis, wavefunction.atomic_numbers.data(),
                                         expected_spin_layout, expected_spin, error);
  if (status == XTBLOOM_STATUS_SUCCESS)
    status = gfn2::make_eigensolver_plan(make_eigensolver_wavefunction_layout(expected_wavefunction),
                                         expected_eigensolver, error,
                                         eigensolver.minimum_overlap_rcond());
  if (status == XTBLOOM_STATUS_SUCCESS)
    status = make_scc_mixer_plan(expected_wavefunction, mixer.history_size(), mixer.damping(),
                                 mixer.rms_tolerance(), mixer.maximum_tolerance(), expected_mixer,
                                 error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  const bool component_match =
      same_wavefunction(wavefunction, expected_wavefunction) &&
      mulliken.atom_offsets() == expected_mulliken.atom_offsets() &&
      mulliken.batch_shell_offsets() == expected_mulliken.batch_shell_offsets() &&
      mulliken.batch_orbital_offsets() == expected_mulliken.batch_orbital_offsets() &&
      mulliken.matrix_offsets() == expected_mulliken.matrix_offsets() &&
      mulliken.shell_orbital_offsets() == expected_mulliken.shell_orbital_offsets() &&
      mulliken.shell_to_atom() == expected_mulliken.shell_to_atom() &&
      mulliken.orbital_to_shell() == expected_mulliken.orbital_to_shell() &&
      mulliken.spin_channels() == expected_mulliken.spin_channels() &&
      mulliken.reference_shell_occupations() ==
          expected_mulliken.reference_shell_occupations() &&
      mulliken.matrix_elements() == expected_mulliken.matrix_elements() &&
      mulliken.density_elements() == expected_mulliken.density_elements() &&
      mulliken.shell_population_elements() == expected_mulliken.shell_population_elements() &&
      mulliken.atom_population_elements() == expected_mulliken.atom_population_elements() &&
      mulliken.population_scratch_elements() == expected_mulliken.population_scratch_elements() &&
      mulliken.hamiltonian_scratch_elements() ==
          expected_mulliken.hamiltonian_scratch_elements() &&
      es2.atom_offsets() == expected_es2.atom_offsets() &&
      es2.batch_shell_offsets() == expected_es2.batch_shell_offsets() &&
      es2.atom_shell_offsets() == expected_es2.atom_shell_offsets() &&
      es2.matrix_offsets() == expected_es2.matrix_offsets() &&
      es2.shell_to_atom() == expected_es2.shell_to_atom() &&
      es2.shell_hardness() == expected_es2.shell_hardness() &&
      es2.hardness_average() == expected_es2.hardness_average() &&
      same_es3(es3, expected_es3) && same_spin(spin, expected_spin) &&
      same_eigensolver(eigensolver, expected_eigensolver) && same_mixer(mixer, expected_mixer) &&
      mixer.matches_wavefunction_layout(expected_wavefunction);
  if (!component_match) {
    error = "GFN1 SCC driver components do not share one canonical model identity";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (periodic_embedding != nullptr &&
      (periodic_embedding->batch_size() != wavefunction.batch_size ||
       periodic_embedding->total_atoms() != wavefunction.total_atoms ||
       periodic_embedding->atom_offsets() != wavefunction.atom_offsets)) {
    error = "GFN1 SCC periodic embedding describes a different atom topology";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  std::size_t batch_double = 0u;
  std::size_t batch_u64 = 0u;
  std::size_t batch_status = 0u;
  std::size_t batch_byte = 0u;
  std::size_t chemical = 0u;
  std::size_t matrix = 0u;
  std::size_t shell = 0u;
  std::size_t atom = 0u;
  std::size_t qsh = 0u;
  std::size_t qat = 0u;
  std::size_t mulliken_scratch = 0u;
  std::size_t periodic_scratch = 0u;
  const std::int64_t mulliken_elements =
      std::max(mulliken.population_scratch_elements(), mulliken.hamiltonian_scratch_elements());
  if (!bytes_for(wavefunction.batch_size, sizeof(double), batch_double) ||
      !bytes_for(wavefunction.batch_size, sizeof(std::uint64_t), batch_u64) ||
      !bytes_for(wavefunction.batch_size, sizeof(xtbloom_status_t), batch_status) ||
      !bytes_for(wavefunction.batch_size, sizeof(std::uint8_t), batch_byte) ||
      !checked_multiply_size(batch_double, 2u, chemical) ||
      !bytes_for(wavefunction.density.element_count, sizeof(double), matrix) ||
      !bytes_for(wavefunction.total_shells, sizeof(double), shell) ||
      !bytes_for(wavefunction.total_atoms, sizeof(double), atom) ||
      !bytes_for(wavefunction.qsh.element_count, sizeof(double), qsh) ||
      !bytes_for(wavefunction.qat.element_count, sizeof(double), qat) ||
      !bytes_for(mulliken_elements, sizeof(double), mulliken_scratch) ||
      !bytes_for(periodic_embedding == nullptr ? 0 : periodic_embedding->maximum_atoms(),
                 sizeof(double), periodic_scratch)) {
    error = "GFN1 SCC caller-owned storage exceeds addressable memory";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  try {
    SccDriverPlanData created;
    created.wavefunction = wavefunction;
    created.mulliken = mulliken;
    created.es2 = es2;
    created.es3 = es3;
    created.spin = spin;
    created.eigensolver = eigensolver;
    created.mixer = mixer;
    if (periodic_embedding != nullptr) created.periodic_embedding = *periodic_embedding;
    created.maximum_iterations = maximum_iterations;
    created.electronic_temperature = electronic_temperature;
    created.energy_tolerance = energy_tolerance;

    std::size_t cursor = 0u;
    if (!append_segment(batch_double, alignof(double), cursor, created.state_free_energy_offset) ||
        !append_segment(batch_double, alignof(double), cursor,
                        created.state_previous_free_energy_offset) ||
        !append_segment(batch_double, alignof(double), cursor,
                        created.state_free_energy_change_offset) ||
        !append_segment(batch_double, alignof(double), cursor, created.state_entropy_offset) ||
        !append_segment(batch_double, alignof(double), cursor, created.state_band_energy_offset) ||
        !append_segment(batch_double, alignof(double), cursor, created.state_core_energy_offset) ||
        !append_segment(batch_double, alignof(double), cursor, created.state_es2_energy_offset) ||
        !append_segment(batch_double, alignof(double), cursor, created.state_es3_energy_offset) ||
        !append_segment(batch_double, alignof(double), cursor, created.state_spin_energy_offset) ||
        !append_segment(batch_double, alignof(double), cursor,
                        created.state_explicit_pc_energy_offset) ||
        !append_segment(periodic_embedding == nullptr ? 0u : batch_double, alignof(double), cursor,
                        created.state_periodic_energy_offset) ||
        !append_segment(batch_double, alignof(double), cursor, created.state_internal_energy_offset) ||
        !append_segment(batch_u64, alignof(std::uint64_t), cursor, created.state_iteration_offset) ||
        !append_segment(batch_status, alignof(xtbloom_status_t), cursor,
                        created.state_status_offset) ||
        !append_segment(batch_byte, alignof(std::uint8_t), cursor,
                        created.state_initialized_offset) ||
        !append_segment(batch_byte, alignof(std::uint8_t), cursor,
                        created.state_converged_offset) ||
        !align_up(cursor, kSccDriverWorkspaceAlignment, created.state_size_bytes)) {
      error = "GFN1 SCC state layout overflows size_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    cursor = 0u;
    if (!append_segment(wavefunction.workspace_size_bytes, kWavefunctionWorkspaceAlignment, cursor,
                        created.staged_wavefunction_offset) ||
        !append_segment(matrix, alignof(double), cursor, created.hamiltonian_offset) ||
        !append_segment(shell, alignof(double), cursor, created.shell_charge_offset) ||
        !append_segment(atom, alignof(double), cursor, created.atomic_charge_offset) ||
        !append_segment(atom, alignof(double), cursor, created.component_atomic_offset) ||
        !append_segment(shell, alignof(double), cursor, created.component_shell_offset) ||
        !append_segment(qsh, alignof(double), cursor, created.shell_potential_offset) ||
        !append_segment(qsh, alignof(double), cursor, created.spin_shell_potential_offset) ||
        !append_segment(qsh, alignof(double), cursor, created.raw_qsh_offset) ||
        !append_segment(qat, alignof(double), cursor, created.raw_qat_offset) ||
        !append_segment(batch_double, alignof(double), cursor, created.core_energy_offset) ||
        !append_segment(batch_double, alignof(double), cursor, created.es2_energy_offset) ||
        !append_segment(batch_double, alignof(double), cursor, created.es3_energy_offset) ||
        !append_segment(batch_double, alignof(double), cursor, created.spin_energy_offset) ||
        !append_segment(batch_double, alignof(double), cursor,
                        created.explicit_pc_energy_offset) ||
        !append_segment(batch_double, alignof(double), cursor, created.internal_energy_offset) ||
        !append_segment(batch_double, alignof(double), cursor, created.free_energy_offset) ||
        !append_segment(periodic_embedding == nullptr ? 0u : atom, alignof(double), cursor,
                        created.periodic_potential_offset) ||
        !append_segment(periodic_embedding == nullptr ? 0u : batch_double, alignof(double), cursor,
                        created.periodic_energy_offset) ||
        !append_segment(periodic_embedding == nullptr ? 0u : batch_status,
                        alignof(xtbloom_status_t), cursor, created.periodic_status_offset) ||
        !append_segment(periodic_scratch, alignof(double), cursor,
                        created.periodic_scratch_offset) ||
        !append_segment(shell, alignof(double), cursor, created.es2_shell_scratch_offset) ||
        !append_segment(mulliken_scratch, alignof(double), cursor,
                        created.mulliken_scratch_offset) ||
        !append_segment(eigensolver.worker_workspace_size_bytes(),
                        gfn2::kEigensolverWorkspaceAlignment,
                        cursor, created.eigensolver_scratch_offset) ||
        !append_segment(mixer.state_size_bytes(), kSccMixerWorkspaceAlignment, cursor,
                        created.staged_mixer_state_offset) ||
        !append_segment(mixer.workspace_size_bytes(), kSccMixerWorkspaceAlignment, cursor,
                        created.mixer_scratch_offset) ||
        !append_segment(batch_status, alignof(xtbloom_status_t), cursor,
                        created.thermodynamic_status_offset) ||
        !append_segment(chemical, alignof(double), cursor, created.chemical_potential_offset) ||
        !append_segment(batch_double, alignof(double), cursor,
                        created.thermodynamic_entropy_offset) ||
        !append_segment(batch_double, alignof(double), cursor,
                        created.thermodynamic_band_energy_offset) ||
        !append_segment(batch_double, alignof(double), cursor,
                        created.thermodynamic_free_energy_offset) ||
        !append_segment(batch_byte, alignof(std::uint8_t), cursor,
                        created.active_system_offset) ||
        !align_up(cursor, kSccDriverWorkspaceAlignment, created.workspace_size_bytes)) {
      error = "GFN1 SCC workspace layout overflows size_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    plan = SccDriverPlan(std::make_shared<const SccDriverPlanData>(std::move(created)));
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate GFN1 SCC driver metadata";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

xtbloom_status_t bind_scc_driver_state(const SccDriverPlan& plan, void* workspace,
                                       std::size_t workspace_size, SccDriverState& state,
                                       std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  const auto& data = *plan.identity();
  AddressRange storage_range;
  AddressRange plan_range;
  AddressRange state_range;
  AddressRange error_range;
  if (!aligned(workspace, kSccDriverWorkspaceAlignment) ||
      workspace_size < data.state_size_bytes ||
      !make_range(workspace, data.state_size_bytes, storage_range) ||
      !make_range(&plan, sizeof(plan), plan_range) || !make_range(&state, sizeof(state), state_range) ||
      !make_range(&error, sizeof(error), error_range) || overlaps_plan_storage(data, storage_range) ||
      ranges_overlap(storage_range, plan_range) || ranges_overlap(storage_range, state_range) ||
      ranges_overlap(storage_range, error_range)) {
    error = "GFN1 SCC state storage is invalid or overlaps control storage";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  SccDriverState bound;
  bound.workspace_base = workspace;
  bound.workspace_size_bytes = workspace_size;
  bound.free_energies = offset_pointer<double>(workspace, data.state_free_energy_offset);
  bound.previous_free_energies =
      offset_pointer<double>(workspace, data.state_previous_free_energy_offset);
  bound.free_energy_changes =
      offset_pointer<double>(workspace, data.state_free_energy_change_offset);
  bound.entropies = offset_pointer<double>(workspace, data.state_entropy_offset);
  bound.band_energies = offset_pointer<double>(workspace, data.state_band_energy_offset);
  bound.core_energies = offset_pointer<double>(workspace, data.state_core_energy_offset);
  bound.es2_energies = offset_pointer<double>(workspace, data.state_es2_energy_offset);
  bound.es3_energies = offset_pointer<double>(workspace, data.state_es3_energy_offset);
  bound.spin_energies = offset_pointer<double>(workspace, data.state_spin_energy_offset);
  bound.explicit_point_charge_energies =
      offset_pointer<double>(workspace, data.state_explicit_pc_energy_offset);
  bound.periodic_embedding_energies = data.periodic_embedding.sealed()
                                           ? offset_pointer<double>(
                                                 workspace, data.state_periodic_energy_offset)
                                           : nullptr;
  bound.internal_energies = offset_pointer<double>(workspace, data.state_internal_energy_offset);
  bound.iterations = offset_pointer<std::uint64_t>(workspace, data.state_iteration_offset);
  bound.system_statuses =
      offset_pointer<xtbloom_status_t>(workspace, data.state_status_offset);
  bound.initialized = offset_pointer<std::uint8_t>(workspace, data.state_initialized_offset);
  bound.converged = offset_pointer<std::uint8_t>(workspace, data.state_converged_offset);
  bound.plan_identity = &data;
  state = bound;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t bind_scc_driver_workspace(const SccDriverPlan& plan, void* workspace,
                                           std::size_t workspace_size,
                                           SccDriverWorkspace& view, std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  const auto& data = *plan.identity();
  AddressRange storage_range;
  AddressRange plan_range;
  AddressRange view_range;
  AddressRange error_range;
  if (!aligned(workspace, kSccDriverWorkspaceAlignment) ||
      workspace_size < data.workspace_size_bytes ||
      !make_range(workspace, data.workspace_size_bytes, storage_range) ||
      !make_range(&plan, sizeof(plan), plan_range) || !make_range(&view, sizeof(view), view_range) ||
      !make_range(&error, sizeof(error), error_range) || overlaps_plan_storage(data, storage_range) ||
      ranges_overlap(storage_range, plan_range) || ranges_overlap(storage_range, view_range) ||
      ranges_overlap(storage_range, error_range)) {
    error = "GFN1 SCC workspace is invalid or overlaps control storage";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  SccDriverWorkspace bound;
  bound.workspace_base = workspace;
  bound.workspace_size_bytes = workspace_size;
  status = bind_wavefunction_view(
      data.wavefunction, offset_pointer<void>(workspace, data.staged_wavefunction_offset),
      data.wavefunction.workspace_size_bytes, bound.staged_wavefunction, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  bound.hamiltonian = offset_pointer<double>(workspace, data.hamiltonian_offset);
  bound.shell_charges = offset_pointer<double>(workspace, data.shell_charge_offset);
  bound.atomic_charges = offset_pointer<double>(workspace, data.atomic_charge_offset);
  bound.component_atomic_potential =
      offset_pointer<double>(workspace, data.component_atomic_offset);
  bound.component_shell_potential =
      offset_pointer<double>(workspace, data.component_shell_offset);
  bound.shell_potentials = offset_pointer<double>(workspace, data.shell_potential_offset);
  bound.spin_shell_potentials =
      offset_pointer<double>(workspace, data.spin_shell_potential_offset);
  bound.raw_qsh = offset_pointer<double>(workspace, data.raw_qsh_offset);
  bound.raw_qat = offset_pointer<double>(workspace, data.raw_qat_offset);
  bound.core_energies = offset_pointer<double>(workspace, data.core_energy_offset);
  bound.es2_energies = offset_pointer<double>(workspace, data.es2_energy_offset);
  bound.es3_energies = offset_pointer<double>(workspace, data.es3_energy_offset);
  bound.spin_energies = offset_pointer<double>(workspace, data.spin_energy_offset);
  bound.explicit_point_charge_energies =
      offset_pointer<double>(workspace, data.explicit_pc_energy_offset);
  bound.internal_energies = offset_pointer<double>(workspace, data.internal_energy_offset);
  bound.free_energies = offset_pointer<double>(workspace, data.free_energy_offset);
  bound.periodic_atomic_potentials = data.periodic_embedding.sealed()
                                         ? offset_pointer<double>(
                                               workspace, data.periodic_potential_offset)
                                         : nullptr;
  bound.periodic_embedding_energies = data.periodic_embedding.sealed()
                                          ? offset_pointer<double>(
                                                workspace, data.periodic_energy_offset)
                                          : nullptr;
  bound.periodic_system_statuses = data.periodic_embedding.sealed()
                                       ? offset_pointer<xtbloom_status_t>(
                                             workspace, data.periodic_status_offset)
                                       : nullptr;
  bound.active_systems = offset_pointer<std::uint8_t>(workspace, data.active_system_offset);
  bound.es2_workspace.shell_scratch =
      offset_pointer<double>(workspace, data.es2_shell_scratch_offset);
  bound.es2_workspace.shell_elements = data.wavefunction.total_shells;
  bound.es2_workspace.batch_scratch = bound.es2_energies;
  bound.es2_workspace.batch_elements = data.wavefunction.batch_size;
  bound.mulliken_workspace.scratch =
      offset_pointer<double>(workspace, data.mulliken_scratch_offset);
  bound.mulliken_workspace.elements =
      std::max(data.mulliken.population_scratch_elements(),
               data.mulliken.hamiltonian_scratch_elements());
  if (data.periodic_embedding.sealed()) {
    status = gfn2::bind_periodic_embedding_workspace(
        data.periodic_embedding, offset_pointer<double>(workspace, data.periodic_scratch_offset),
        static_cast<std::size_t>(data.periodic_embedding.maximum_atoms()),
        bound.periodic_embedding_workspace, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  }
  status = gfn2::bind_eigensolver_worker_workspace(
      data.eigensolver, offset_pointer<void>(workspace, data.eigensolver_scratch_offset),
      data.eigensolver.worker_workspace_size_bytes(), bound.eigensolver_workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = bind_scc_mixer_state(
      data.mixer, offset_pointer<void>(workspace, data.staged_mixer_state_offset),
      data.mixer.state_size_bytes(), bound.staged_mixer_state, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = bind_scc_mixer_workspace(
      data.mixer, offset_pointer<void>(workspace, data.mixer_scratch_offset),
      data.mixer.workspace_size_bytes(), bound.mixer_workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  bound.thermodynamics = {
      offset_pointer<xtbloom_status_t>(workspace, data.thermodynamic_status_offset),
      static_cast<std::size_t>(data.wavefunction.batch_size),
      offset_pointer<double>(workspace, data.chemical_potential_offset),
      2u * static_cast<std::size_t>(data.wavefunction.batch_size),
      offset_pointer<double>(workspace, data.thermodynamic_entropy_offset),
      static_cast<std::size_t>(data.wavefunction.batch_size),
      offset_pointer<double>(workspace, data.thermodynamic_band_energy_offset),
      static_cast<std::size_t>(data.wavefunction.batch_size),
      offset_pointer<double>(workspace, data.thermodynamic_free_energy_offset),
      static_cast<std::size_t>(data.wavefunction.batch_size)};
  bound.plan_identity = &data;
  view = bound;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t initialize_scc_driver_state_cpu(const SccDriverPlan& plan,
                                                 const WavefunctionView& wavefunction,
                                                 const SccMixerState& mixer_state,
                                                 const SccDriverState& state,
                                                 std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  const auto& data = *plan.identity();
  if (!exact_state_binding(data, state) ||
      validate_scc_mixer_state_binding(data.mixer, mixer_state, error) !=
          XTBLOOM_STATUS_SUCCESS) {
    error = "GFN1 SCC initialization bindings do not belong to the sealed plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  status = validate_wavefunction(data, wavefunction, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
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
    error = "GFN1 SCC initialization storage overlaps numerical, plan, or descriptor storage";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  status = initialize_scc_mixer_state_cpu(data.mixer, wavefunction, mixer_state, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
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
  std::fill_n(state.spin_energies, batch, nan);
  std::fill_n(state.explicit_point_charge_energies, batch, nan);
  if (state.periodic_embedding_energies != nullptr)
    std::fill_n(state.periodic_embedding_energies, batch, nan);
  std::fill_n(state.internal_energies, batch, nan);
  std::fill_n(state.system_statuses, batch, XTBLOOM_STATUS_SUCCESS);
  std::fill_n(state.initialized, batch, static_cast<std::uint8_t>(1u));
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t restart_scc_driver_system_cpu(const SccDriverPlan& plan, std::int64_t system,
                                               const WavefunctionView& wavefunction,
                                               const SccMixerState& mixer_state,
                                               const SccDriverState& state,
                                               std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  const auto& data = *plan.identity();
  if (system < 0 || system >= data.wavefunction.batch_size || !exact_state_binding(data, state) ||
      validate_scc_mixer_state_binding(data.mixer, mixer_state, error) !=
          XTBLOOM_STATUS_SUCCESS) {
    error = "GFN1 SCC restart bindings or system index are invalid";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  status = validate_wavefunction(data, wavefunction, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  const std::size_t index = static_cast<std::size_t>(system);
  if (state.initialized[index] != 1u || state.converged[index] > 1u ||
      mixer_state.initialized[index] != 1u || mixer_state.converged[index] > 1u ||
      state.converged[index] != mixer_state.converged[index] ||
      (state.system_statuses[index] == XTBLOOM_STATUS_SUCCESS &&
       (mixer_state.system_statuses[index] != XTBLOOM_STATUS_SUCCESS ||
        state.iterations[index] != mixer_state.iterations[index]))) {
    error = "GFN1 SCC system must have canonical initialized state before restart";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
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
    error = "GFN1 SCC restart storage overlaps numerical, plan, or descriptor storage";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  status = restart_scc_mixer_system_cpu(data.mixer, system, wavefunction, mixer_state, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  state.free_energies[index] = nan;
  state.previous_free_energies[index] = nan;
  state.free_energy_changes[index] = nan;
  state.entropies[index] = nan;
  state.band_energies[index] = nan;
  state.core_energies[index] = nan;
  state.es2_energies[index] = nan;
  state.es3_energies[index] = nan;
  state.spin_energies[index] = nan;
  state.explicit_point_charge_energies[index] = nan;
  if (state.periodic_embedding_energies != nullptr) state.periodic_embedding_energies[index] = nan;
  state.internal_energies[index] = nan;
  state.iterations[index] = 0u;
  state.system_statuses[index] = XTBLOOM_STATUS_SUCCESS;
  state.converged[index] = 0u;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

namespace {

xtbloom_status_t validate_iteration_bindings(
    const SccDriverPlan& plan, const SccDriverPlanData& data,
    const SccDriverGeometryView& geometry, const CpuLinearAlgebraBackend& backend,
    const EigensolverOverlapCache& overlap_cache, const WavefunctionView& wavefunction,
    const SccMixerState& mixer_state, const SccDriverState& state,
    const SccDriverWorkspace& workspace, std::string& error) {
  if (!backend.ready() || !exact_state_binding(data, state) ||
      !exact_workspace_binding(data, workspace)) {
    error = "GFN1 SCC runtime bindings do not belong to the sealed plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  xtbloom_status_t status = validate_wavefunction(data, wavefunction, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = validate_wavefunction(data, workspace.staged_wavefunction, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = validate_scc_mixer_state_binding(data.mixer, mixer_state, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = validate_scc_mixer_state_binding(data.mixer, workspace.staged_mixer_state, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = validate_scc_mixer_workspace_binding(data.mixer, workspace.mixer_workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = gfn2::validate_eigensolver_overlap_cache_binding(data.eigensolver, overlap_cache, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = gfn2::validate_eigensolver_worker_workspace_binding(
      data.eigensolver, workspace.eigensolver_workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  if (geometry.geometry_generation == 0u ||
      geometry.h0_elements != data.mulliken.matrix_elements() ||
      !aligned(geometry.h0, alignof(double)) ||
      geometry.integrals.elements != data.mulliken.matrix_elements() ||
      !aligned(geometry.integrals.overlap, alignof(double)) ||
      geometry.integrals.plan_identity != data.mulliken.identity() ||
      geometry.es2_cache.plan_identity != data.es2.identity() ||
      geometry.es2_cache.geometry_generation != geometry.geometry_generation ||
      geometry.es2_cache.matrix_elements != data.es2.total_matrix_elements() ||
      !aligned(geometry.es2_cache.coulomb_matrix, alignof(double))) {
    error = "GFN1 SCC geometry or geometry cache is stale, malformed, or incompatible";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const bool point_present = geometry.explicit_point_charge_shell_potential != nullptr ||
                             geometry.explicit_point_charge_shell_elements != 0;
  if (point_present &&
      (geometry.explicit_point_charge_shell_potential == nullptr ||
       geometry.explicit_point_charge_shell_elements != data.wavefunction.total_shells ||
       !aligned(geometry.explicit_point_charge_shell_potential, alignof(double)))) {
    error = "GFN1 SCC point-charge potential has invalid extent or alignment";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (data.periodic_embedding.sealed()) {
    if (geometry.periodic_shifts == nullptr || geometry.periodic_response_matrices == nullptr ||
        geometry.periodic_shift_elements != data.periodic_embedding.total_atoms() ||
        geometry.periodic_response_elements != data.periodic_embedding.total_matrix_elements() ||
        !aligned(geometry.periodic_shifts, alignof(double)) ||
        !aligned(geometry.periodic_response_matrices, alignof(double)) ||
        geometry.periodic_embedding_generation == 0u ||
        geometry.periodic_plan_identity != data.periodic_embedding.identity()) {
      error = "GFN1 SCC periodic embedding is stale, malformed, or belongs to another plan";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  } else if (geometry.periodic_shifts != nullptr || geometry.periodic_shift_elements != 0 ||
             geometry.periodic_response_matrices != nullptr ||
             geometry.periodic_response_elements != 0 ||
             geometry.periodic_embedding_generation != 0u ||
             geometry.periodic_plan_identity != nullptr) {
    error = "GFN1 SCC geometry supplies periodic data to a nonperiodic plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
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
    error = "GFN1 SCC runtime storage overlaps numerical, plan, or descriptor storage";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  AddressRange mixer_storage;
  AddressRange mixer_initialized;
  AddressRange mixer_converged;
  if (!make_range(mixer_state.workspace_base, data.mixer.state_size_bytes(), mixer_storage) ||
      !make_range(mixer_state.initialized, batch * sizeof(std::uint8_t), mixer_initialized) ||
      !make_range(mixer_state.converged, batch * sizeof(std::uint8_t), mixer_converged) ||
      !range_contains(mixer_storage, mixer_initialized) ||
      !range_contains(mixer_storage, mixer_converged)) {
    error = "GFN1 SCC mixer state pointers are outside their caller-owned binding";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::size_t system = 0u; system < batch; ++system) {
    if (state.initialized[system] != 1u || mixer_state.initialized[system] != 1u ||
        state.converged[system] > 1u || mixer_state.converged[system] > 1u ||
        state.converged[system] != mixer_state.converged[system] ||
        (state.system_statuses[system] == XTBLOOM_STATUS_SUCCESS &&
         (mixer_state.system_statuses[system] != XTBLOOM_STATUS_SUCCESS ||
          state.iterations[system] != mixer_state.iterations[system]))) {
      error = "GFN1 SCC requires initialized canonical per-system state";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  std::size_t matrix_bytes = 0u;
  std::size_t es2_cache_bytes = 0u;
  std::size_t point_potential_bytes = 0u;
  std::size_t periodic_shift_bytes = 0u;
  std::size_t periodic_response_bytes = 0u;
  if (!bytes_for(data.mulliken.matrix_elements(), sizeof(double), matrix_bytes) ||
      !bytes_for(data.es2.total_matrix_elements(), sizeof(double), es2_cache_bytes) ||
      !bytes_for(geometry.explicit_point_charge_shell_elements, sizeof(double),
                 point_potential_bytes) ||
      !bytes_for(geometry.periodic_shift_elements, sizeof(double), periodic_shift_bytes) ||
      !bytes_for(geometry.periodic_response_elements, sizeof(double), periodic_response_bytes)) {
    error = "GFN1 SCC geometry storage extents are not representable";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  std::array<AddressRange, 6> geometry_ranges{};
  if (!make_range(geometry.h0, matrix_bytes, geometry_ranges[0]) ||
      !make_range(geometry.integrals.overlap, matrix_bytes, geometry_ranges[1]) ||
      !make_range(geometry.es2_cache.coulomb_matrix, es2_cache_bytes, geometry_ranges[2]) ||
      !make_range(geometry.explicit_point_charge_shell_potential, point_potential_bytes,
                  geometry_ranges[3]) ||
      !make_range(geometry.periodic_shifts, periodic_shift_bytes, geometry_ranges[4]) ||
      !make_range(geometry.periodic_response_matrices, periodic_response_bytes,
                  geometry_ranges[5])) {
    error = "GFN1 SCC geometry buffers have invalid address ranges";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (const AddressRange& input : geometry_ranges) {
    if (overlaps_plan_storage(data, input)) {
      error = "GFN1 SCC geometry inputs must not overlap immutable plan storage";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t output = 0u; output < 4u; ++output) {
      if (ranges_overlap(input, principal[output])) {
        error = "GFN1 SCC geometry inputs must not overlap mutable state or scratch";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
    for (const AddressRange& control : controls) {
      if (ranges_overlap(input, control)) {
        error = "GFN1 SCC geometry inputs must not overlap descriptor storage";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace

xtbloom_status_t iterate_scc_driver_batch_cpu(
    const SccDriverPlan& plan, const SccDriverGeometryView& geometry,
    const CpuLinearAlgebraBackend& backend, const EigensolverOverlapCache& overlap_cache,
    const WavefunctionView& wavefunction, const SccMixerState& mixer_state,
    const SccDriverState& state, const SccDriverWorkspace& workspace, std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  const auto& data = *plan.identity();
  status = validate_iteration_bindings(plan, data, geometry, backend, overlap_cache, wavefunction,
                                       mixer_state, state, workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  const std::size_t batch = static_cast<std::size_t>(data.wavefunction.batch_size);
  bool any_active = false;
  for (std::size_t system = 0u; system < batch; ++system) {
    const bool active = state.initialized[system] == 1u &&
                        state.system_statuses[system] == XTBLOOM_STATUS_SUCCESS &&
                        state.converged[system] == 0u &&
                        state.iterations[system] < data.maximum_iterations;
    workspace.active_systems[system] = active ? 1u : 0u;
    any_active = any_active || active;
  }
  if (!any_active) {
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }
  copy_wavefunction(data.wavefunction, wavefunction, workspace.staged_wavefunction);
  for (std::size_t system = 0u; system < batch; ++system) {
    if (workspace.active_systems[system] != 1u) continue;
    status = prepare_system(data, geometry, system, workspace, error);
    if (status == XTBLOOM_STATUS_INVALID_ARGUMENT || status == XTBLOOM_STATUS_NOT_SUPPORTED)
      return status;
    if (status != XTBLOOM_STATUS_SUCCESS) workspace.active_systems[system] = 5u;
  }

  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::fill_n(workspace.thermodynamics.system_statuses, batch, XTBLOOM_STATUS_INVALID_ARGUMENT);
  std::fill_n(workspace.thermodynamics.chemical_potentials, 2u * batch, nan);
  std::fill_n(workspace.thermodynamics.entropies, batch, nan);
  std::fill_n(workspace.thermodynamics.band_energies, batch, nan);
  std::fill_n(workspace.thermodynamics.free_energies, batch, nan);
  const auto projected_wavefunction =
      make_eigensolver_wavefunction_view(workspace.staged_wavefunction);
  for (std::size_t system = 0u; system < batch; ++system) {
    if (workspace.active_systems[system] != 1u) continue;
    const std::int64_t hamiltonian_begin = data.wavefunction.density.system_offsets[system];
    status = gfn2::solve_eigensystem_cpu(
        data.eigensolver, static_cast<std::int64_t>(system), overlap_cache,
        geometry.geometry_generation, workspace.hamiltonian + hamiltonian_begin,
        data.electronic_temperature, backend, workspace.eigensolver_workspace,
        projected_wavefunction, workspace.thermodynamics, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    if (workspace.thermodynamics.system_statuses[system] != XTBLOOM_STATUS_SUCCESS)
      workspace.active_systems[system] = 2u;
  }

  const MullikenDensityView density{workspace.staged_wavefunction.density,
                                    data.wavefunction.density.element_count,
                                    data.mulliken.identity()};
  const MullikenPopulationView population{
      workspace.raw_qsh, data.wavefunction.qsh.element_count, workspace.raw_qat,
      data.wavefunction.qat.element_count, data.mulliken.identity()};
  for (std::size_t system = 0u; system < batch; ++system) {
    if (workspace.active_systems[system] != 1u) continue;
    status = evaluate_mulliken_population_system_cpu(
        data.mulliken, geometry.integrals, density, population, static_cast<std::int64_t>(system),
        workspace.mulliken_workspace, error);
    if (status == XTBLOOM_STATUS_INVALID_ARGUMENT) return status;
    if (status != XTBLOOM_STATUS_SUCCESS) {
      workspace.active_systems[system] = 6u;
      continue;
    }
    status = evaluate_energy(data, geometry, system, workspace, error);
    if (status == XTBLOOM_STATUS_INVALID_ARGUMENT) return status;
    if (status != XTBLOOM_STATUS_SUCCESS) workspace.active_systems[system] = 6u;
  }

  for (std::size_t system = 0u; system < batch; ++system) {
    if (workspace.active_systems[system] != 1u) continue;
    status = prepare_scc_mixer_system_transaction_cpu(
        data.mixer, static_cast<std::int64_t>(system), mixer_state,
        workspace.staged_mixer_state, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    copy_system_field(data.wavefunction.qsh, system, workspace.raw_qsh,
                      workspace.staged_wavefunction.qsh);
    copy_system_field(data.wavefunction.qat, system, workspace.raw_qat,
                      workspace.staged_wavefunction.qat);
    status = mix_scc_broyden_system_cpu(
        data.mixer, static_cast<std::int64_t>(system), workspace.staged_wavefunction,
        workspace.staged_mixer_state, workspace.mixer_workspace, error);
    if (status == XTBLOOM_STATUS_INVALID_ARGUMENT) return status;
    if (status != XTBLOOM_STATUS_SUCCESS) {
      /* Discard the staged history but retain the mixer's peer-local failure
       * status so the driver and mixer diagnostics describe the same failed
       * transaction. */
      mixer_state.system_statuses[system] =
          workspace.staged_mixer_state.system_statuses[system];
      workspace.active_systems[system] = 3u;
      continue;
    }
    const double old_energy = state.iterations[system] == 0u ? 0.0 : state.free_energies[system];
    const double change = workspace.free_energies[system] - old_energy;
    if (!std::isfinite(old_energy) || !std::isfinite(change)) {
      workspace.staged_mixer_state.system_statuses[system] = XTBLOOM_STATUS_INTERNAL_ERROR;
      mixer_state.system_statuses[system] =
          workspace.staged_mixer_state.system_statuses[system];
      workspace.active_systems[system] = 3u;
      continue;
    }
    const bool converged =
        workspace.staged_mixer_state.residual_rms[system] < data.mixer.rms_tolerance() &&
        std::abs(change) < data.energy_tolerance;
    workspace.staged_mixer_state.converged[system] = converged ? 1u : 0u;
    if (converged) {
      copy_system_field(data.wavefunction.qsh, system, workspace.raw_qsh,
                        workspace.staged_wavefunction.qsh);
      copy_system_field(data.wavefunction.qat, system, workspace.raw_qat,
                        workspace.staged_wavefunction.qat);
    } else {
      bool valid = true;
      rebuild_atomic_populations(data, system, workspace.staged_wavefunction.qsh,
                                 workspace.staged_wavefunction.qat, valid);
      if (!valid) {
        workspace.staged_mixer_state.system_statuses[system] = XTBLOOM_STATUS_INTERNAL_ERROR;
        mixer_state.system_statuses[system] =
            workspace.staged_mixer_state.system_statuses[system];
        workspace.active_systems[system] = 3u;
        continue;
      }
    }
    status = commit_scc_mixer_system_transaction_cpu(
        data.mixer, static_cast<std::int64_t>(system), workspace.staged_mixer_state, mixer_state,
        error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    workspace.active_systems[system] = 4u;
  }

  xtbloom_status_t first_failure = XTBLOOM_STATUS_SUCCESS;
  bool first_failure_was_preparation = false;
  for (std::size_t system = 0u; system < batch; ++system) {
    if (workspace.active_systems[system] == 0u) continue;
    if (workspace.active_systems[system] != 4u) {
      const xtbloom_status_t failure =
          workspace.active_systems[system] == 2u
              ? XTBLOOM_STATUS_EIGENSOLVER_FAILED
              : (workspace.active_systems[system] == 3u
                     ? workspace.staged_mixer_state.system_statuses[system]
                     : XTBLOOM_STATUS_INTERNAL_ERROR);
      state.system_statuses[system] = failure;
      state.free_energies[system] = nan;
      state.previous_free_energies[system] = nan;
      state.free_energy_changes[system] = nan;
      state.entropies[system] = nan;
      state.band_energies[system] = nan;
      state.core_energies[system] = nan;
      state.es2_energies[system] = nan;
      state.es3_energies[system] = nan;
      state.spin_energies[system] = nan;
      state.explicit_point_charge_energies[system] = nan;
      if (state.periodic_embedding_energies != nullptr)
        state.periodic_embedding_energies[system] = nan;
      state.internal_energies[system] = nan;
      /* Potential preparation happens before an eigensolve. Other numerical
       * failures occur after the iteration's eigensolve was attempted. */
      if (workspace.active_systems[system] != 5u) ++state.iterations[system];
      if (first_failure == XTBLOOM_STATUS_SUCCESS) {
        first_failure = failure;
        first_failure_was_preparation = workspace.active_systems[system] == 5u;
      }
      continue;
    }
    const double old_energy = state.iterations[system] == 0u ? 0.0 : state.free_energies[system];
    const double new_energy = workspace.free_energies[system];
    commit_system_wavefunction(data.wavefunction, system, workspace.staged_wavefunction,
                               wavefunction);
    state.previous_free_energies[system] = old_energy;
    state.free_energies[system] = new_energy;
    state.free_energy_changes[system] = new_energy - old_energy;
    state.entropies[system] = workspace.thermodynamics.entropies[system];
    state.band_energies[system] = workspace.thermodynamics.band_energies[system];
    state.core_energies[system] = workspace.core_energies[system];
    state.es2_energies[system] = workspace.es2_energies[system];
    state.es3_energies[system] = workspace.es3_energies[system];
    state.spin_energies[system] = workspace.spin_energies[system];
    state.explicit_point_charge_energies[system] =
        workspace.explicit_point_charge_energies[system];
    if (state.periodic_embedding_energies != nullptr)
      state.periodic_embedding_energies[system] = workspace.periodic_embedding_energies[system];
    state.internal_energies[system] = workspace.internal_energies[system];
    ++state.iterations[system];
    state.converged[system] = mixer_state.converged[system];
    if (state.converged[system] == 0u && state.iterations[system] >= data.maximum_iterations) {
      state.system_statuses[system] = XTBLOOM_STATUS_SCC_NOT_CONVERGED;
      if (first_failure == XTBLOOM_STATUS_SUCCESS) {
        first_failure = XTBLOOM_STATUS_SCC_NOT_CONVERGED;
      }
    } else {
      state.system_statuses[system] = XTBLOOM_STATUS_SUCCESS;
    }
  }
  /* Per-system transactions preserve successful peers. Returning the first
   * data-level failure lets the future CPU batch executor translate it into
   * NaN publication without hiding SCC/eigensolver diagnostics. */
  if (first_failure != XTBLOOM_STATUS_SUCCESS) {
    if (first_failure == XTBLOOM_STATUS_EIGENSOLVER_FAILED) {
      error = "one or more GFN1 SCC systems failed during generalized eigensolve";
    } else if (first_failure == XTBLOOM_STATUS_SCC_NOT_CONVERGED) {
      error = "one or more GFN1 SCC systems reached the maximum iteration count";
    } else if (first_failure_was_preparation) {
      error = "one or more GFN1 SCC systems failed during potential or Mulliken preparation";
    } else {
      error = "one or more GFN1 SCC systems failed during energy assembly or mixing";
    }
    return first_failure;
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t rebuild_scc_stationary_potentials_cpu(
    const SccDriverPlan& plan, const SccDriverGeometryView& geometry,
    const WavefunctionView& wavefunction, const SccDriverWorkspace& workspace,
    std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  const auto& data = *plan.identity();
  if (!exact_workspace_binding(data, workspace)) {
    error = "GFN1 stationary-potential workspace does not belong to the sealed driver plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  status = validate_wavefunction(data, wavefunction, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = validate_wavefunction(data, workspace.staged_wavefunction, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (geometry.geometry_generation == 0u ||
      geometry.h0_elements != data.mulliken.matrix_elements() ||
      !aligned(geometry.h0, alignof(double)) ||
      geometry.integrals.elements != data.mulliken.matrix_elements() ||
      !aligned(geometry.integrals.overlap, alignof(double)) ||
      geometry.integrals.plan_identity != data.mulliken.identity() ||
      geometry.es2_cache.plan_identity != data.es2.identity() ||
      geometry.es2_cache.geometry_generation != geometry.geometry_generation ||
      geometry.es2_cache.matrix_elements != data.es2.total_matrix_elements() ||
      !aligned(geometry.es2_cache.coulomb_matrix, alignof(double))) {
    error = "GFN1 stationary-potential geometry or cache is stale or incompatible";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const bool point_present = geometry.explicit_point_charge_shell_potential != nullptr ||
                             geometry.explicit_point_charge_shell_elements != 0;
  if (point_present &&
      (geometry.explicit_point_charge_shell_potential == nullptr ||
       geometry.explicit_point_charge_shell_elements != data.wavefunction.total_shells ||
       !aligned(geometry.explicit_point_charge_shell_potential, alignof(double)))) {
    error = "GFN1 stationary-potential point-charge input is malformed";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (data.periodic_embedding.sealed()) {
    if (geometry.periodic_shifts == nullptr || geometry.periodic_response_matrices == nullptr ||
        geometry.periodic_shift_elements != data.periodic_embedding.total_atoms() ||
        geometry.periodic_response_elements != data.periodic_embedding.total_matrix_elements() ||
        geometry.periodic_embedding_generation == 0u ||
        geometry.periodic_plan_identity != data.periodic_embedding.identity()) {
      error = "GFN1 stationary-potential periodic input is malformed or incompatible";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  } else if (geometry.periodic_shifts != nullptr || geometry.periodic_shift_elements != 0 ||
             geometry.periodic_response_matrices != nullptr ||
             geometry.periodic_response_elements != 0 ||
             geometry.periodic_embedding_generation != 0u ||
             geometry.periodic_plan_identity != nullptr) {
    error = "GFN1 stationary-potential input supplies periodic data to a nonperiodic plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  copy_wavefunction(data.wavefunction, wavefunction, workspace.staged_wavefunction);
  for (std::size_t system = 0u;
       system < static_cast<std::size_t>(data.wavefunction.batch_size); ++system) {
    status = prepare_system(data, geometry, system, workspace, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t project_scc_stationary_state_cpu(
    const WavefunctionLayout& layout, const WavefunctionView& wavefunction,
    const double* packed_shell_potentials, std::int64_t packed_shell_potential_elements,
    const SccStationaryProjection& projection, std::string& error) {
  WavefunctionSystemView ignored;
  xtbloom_status_t status =
      make_wavefunction_system_view(layout, wavefunction, 0, ignored, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  std::int64_t matrix_elements = 0;
  bool has_unrestricted = false;
  for (std::int64_t system = 0; system < layout.batch_size; ++system) {
    const std::int64_t orbitals =
        layout.batch_orbital_offsets[static_cast<std::size_t>(system) + 1u] -
        layout.batch_orbital_offsets[static_cast<std::size_t>(system)];
    if (orbitals <= 0 || orbitals > std::numeric_limits<std::int64_t>::max() / orbitals ||
        matrix_elements > std::numeric_limits<std::int64_t>::max() - orbitals * orbitals) {
      error = "GFN1 stationary projection matrix extent overflows signed dimensions";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    matrix_elements += orbitals * orbitals;
    has_unrestricted = has_unrestricted ||
                       layout.spin_channels[static_cast<std::size_t>(system)] == 2;
  }
  if (packed_shell_potentials == nullptr ||
      packed_shell_potential_elements != layout.qsh.element_count ||
      projection.matrix_elements != matrix_elements ||
      projection.shell_elements != layout.total_shells ||
      projection.atom_elements != layout.total_atoms || projection.density == nullptr ||
      projection.energy_weighted_density == nullptr || projection.shell_charges == nullptr ||
      projection.atomic_charges == nullptr || projection.scalar_shell_potentials == nullptr ||
      (has_unrestricted != (projection.spin_density != nullptr)) ||
      (has_unrestricted != (projection.spin_shell_potentials != nullptr))) {
    error = "GFN1 stationary projection binding does not match the wavefunction topology";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  std::size_t matrix_bytes = 0u;
  std::size_t shell_bytes = 0u;
  std::size_t atom_bytes = 0u;
  std::size_t packed_bytes = 0u;
  if (!bytes_for(matrix_elements, sizeof(double), matrix_bytes) ||
      !bytes_for(layout.total_shells, sizeof(double), shell_bytes) ||
      !bytes_for(layout.total_atoms, sizeof(double), atom_bytes) ||
      !bytes_for(layout.qsh.element_count, sizeof(double), packed_bytes)) {
    error = "GFN1 stationary projection byte extent overflows host dimensions";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  AddressRange wavefunction_range;
  AddressRange potential_range;
  if (!make_range(wavefunction.workspace_base, wavefunction.workspace_size_bytes,
                  wavefunction_range) ||
      !make_range(packed_shell_potentials, packed_bytes, potential_range)) {
    error = "GFN1 stationary projection input range is invalid";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  std::array<AddressRange, 7> outputs{};
  const std::array<std::pair<void*, std::size_t>, 7> output_bindings{{
      {projection.density, matrix_bytes},
      {projection.energy_weighted_density, matrix_bytes},
      {projection.spin_density, has_unrestricted ? matrix_bytes : 0u},
      {projection.shell_charges, shell_bytes},
      {projection.atomic_charges, atom_bytes},
      {projection.scalar_shell_potentials, shell_bytes},
      {projection.spin_shell_potentials, has_unrestricted ? shell_bytes : 0u},
  }};
  for (std::size_t index = 0u; index < outputs.size(); ++index) {
    if (!make_range(output_bindings[index].first, output_bindings[index].second,
                    outputs[index]) ||
        ranges_overlap(outputs[index], wavefunction_range) ||
        ranges_overlap(outputs[index], potential_range)) {
      error = "GFN1 stationary projection outputs overlap an input or have invalid ranges";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  if (!pairwise_disjoint(outputs)) {
    error = "GFN1 stationary projection outputs must be mutually disjoint";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  if (!std::all_of(packed_shell_potentials,
                   packed_shell_potentials + packed_shell_potential_elements,
                   [](double value) { return std::isfinite(value); })) {
    error = "GFN1 stationary projection shell potentials contain NaN or infinity";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::array<std::pair<const double*, std::int64_t>, 5> fields{{
      {wavefunction.density, layout.density.element_count},
      {wavefunction.energy_weighted_density, layout.energy_weighted_density.element_count},
      {wavefunction.qsh, layout.qsh.element_count},
      {wavefunction.qat, layout.qat.element_count},
      {packed_shell_potentials, packed_shell_potential_elements},
  }};
  for (const auto& field : fields) {
    if (!std::all_of(field.first, field.first + field.second,
                     [](double value) { return std::isfinite(value); })) {
      error = "GFN1 stationary projection input contains NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  /* Validate every alpha/beta reduction before touching caller outputs.  A
   * finite pair can still overflow on addition, and projection is used inside
   * a larger publication transaction whose unchanged-output guarantee must
   * survive that case. */
  for (std::int64_t system = 0; system < layout.batch_size; ++system) {
    const std::size_t index = static_cast<std::size_t>(system);
    if (layout.spin_channels[index] != 2) continue;
    const std::int64_t orbitals =
        layout.batch_orbital_offsets[index + 1u] - layout.batch_orbital_offsets[index];
    const std::int64_t matrices = orbitals * orbitals;
    const std::int64_t density_begin = layout.density.system_offsets[index];
    const std::int64_t weighted_begin = layout.energy_weighted_density.system_offsets[index];
    for (std::int64_t element = 0; element < matrices; ++element) {
      const double alpha_density = wavefunction.density[density_begin + element];
      const double beta_density = wavefunction.density[density_begin + matrices + element];
      const double alpha_weighted =
          wavefunction.energy_weighted_density[weighted_begin + element];
      const double beta_weighted =
          wavefunction.energy_weighted_density[weighted_begin + matrices + element];
      if (!std::isfinite(alpha_density + beta_density) ||
          !std::isfinite(alpha_density - beta_density) ||
          !std::isfinite(alpha_weighted + beta_weighted)) {
        error = "GFN1 stationary matrix projection overflowed";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
    }
  }

  std::int64_t matrix_output = 0;
  for (std::int64_t system = 0; system < layout.batch_size; ++system) {
    const std::size_t index = static_cast<std::size_t>(system);
    const std::int64_t orbitals =
        layout.batch_orbital_offsets[index + 1u] - layout.batch_orbital_offsets[index];
    const std::int64_t matrices = orbitals * orbitals;
    const std::int64_t density_begin = layout.density.system_offsets[index];
    const std::int64_t weighted_begin = layout.energy_weighted_density.system_offsets[index];
    const std::int32_t channels = layout.spin_channels[index];
    for (std::int64_t element = 0; element < matrices; ++element) {
      const double alpha_density = wavefunction.density[density_begin + element];
      const double alpha_weighted =
          wavefunction.energy_weighted_density[weighted_begin + element];
      double total_density = alpha_density;
      double total_weighted = alpha_weighted;
      double spin_density = 0.0;
      if (channels == 2) {
        const double beta_density = wavefunction.density[density_begin + matrices + element];
        const double beta_weighted =
            wavefunction.energy_weighted_density[weighted_begin + matrices + element];
        total_density += beta_density;
        total_weighted += beta_weighted;
        spin_density = alpha_density - beta_density;
      }
      if (!std::isfinite(total_density) || !std::isfinite(total_weighted) ||
          !std::isfinite(spin_density)) {
        error = "GFN1 stationary matrix projection overflowed";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      projection.density[matrix_output + element] = total_density;
      projection.energy_weighted_density[matrix_output + element] = total_weighted;
      if (has_unrestricted) projection.spin_density[matrix_output + element] = spin_density;
    }

    const std::int64_t shell_begin = layout.batch_shell_offsets[index];
    const std::int64_t shells = layout.batch_shell_offsets[index + 1u] - shell_begin;
    const std::int64_t qsh_begin = layout.qsh.system_offsets[index];
    for (std::int64_t shell = 0; shell < shells; ++shell) {
      projection.shell_charges[shell_begin + shell] = wavefunction.qsh[qsh_begin + shell];
      projection.scalar_shell_potentials[shell_begin + shell] =
          packed_shell_potentials[qsh_begin + shell];
      if (has_unrestricted) {
        projection.spin_shell_potentials[shell_begin + shell] =
            channels == 2 ? packed_shell_potentials[qsh_begin + shells + shell] : 0.0;
      }
    }

    const std::int64_t atom_begin = layout.atom_offsets[index];
    const std::int64_t atoms = layout.atom_offsets[index + 1u] - atom_begin;
    const std::int64_t qat_begin = layout.qat.system_offsets[index];
    std::copy_n(wavefunction.qat + qat_begin, static_cast<std::size_t>(atoms),
                projection.atomic_charges + atom_begin);
    matrix_output += matrices;
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn1
