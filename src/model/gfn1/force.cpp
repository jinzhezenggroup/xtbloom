// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn1/force.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "data/parameters/gfn1.hpp"

namespace xtbloom::detail::gfn1 {
namespace {

bool finite_array(const double* values, std::size_t count) {
  if (values == nullptr) return false;
  for (std::size_t index = 0u; index < count; ++index) {
    if (!std::isfinite(values[index])) return false;
  }
  return true;
}

bool valid_scratch(double* values, std::int64_t extent, std::int64_t required) {
  return required >= 0 && extent >= required &&
         (required == 0 ||
          (values != nullptr && reinterpret_cast<std::uintptr_t>(values) % alignof(double) == 0u));
}

bool checked_multiply(std::int64_t first, std::int64_t second, std::int64_t& result) {
  if (first < 0 || second < 0 ||
      (first != 0 && second > std::numeric_limits<std::int64_t>::max() / first)) {
    return false;
  }
  result = first * second;
  return true;
}

bool same_offsets(const std::vector<std::int64_t>& first, const std::vector<std::int64_t>& second) {
  return first == second;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool make_range(const void* pointer, std::size_t bytes, AddressRange& range) {
  if (bytes == 0u) {
    if (pointer != nullptr) return false;
    range = {};
    return true;
  }
  if (pointer == nullptr) return false;
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) return false;
  range = {begin, begin + bytes};
  return true;
}

bool make_double_range(const double* pointer, std::int64_t elements, AddressRange& range) {
  if (elements < 0 ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(double)) ||
      (elements != 0 && reinterpret_cast<std::uintptr_t>(pointer) % alignof(double) != 0u)) {
    return false;
  }
  return make_range(pointer, static_cast<std::size_t>(elements) * sizeof(double), range);
}

bool ranges_overlap(const AddressRange& first, const AddressRange& second) {
  return first.begin != first.end && second.begin != second.end && first.begin < second.end &&
         second.begin < first.end;
}

template <typename T>
bool overlaps_vector(const AddressRange& range, const std::vector<T>& values) {
  AddressRange storage;
  return make_range(values.data(), values.capacity() * sizeof(T), storage) &&
         ranges_overlap(range, storage);
}

bool overlaps_basis_storage(const AddressRange& range, const BasisPlan& basis) {
  return overlaps_vector(range, basis.atom_offsets) ||
         overlaps_vector(range, basis.batch_shell_offsets) ||
         overlaps_vector(range, basis.batch_orbital_offsets) ||
         overlaps_vector(range, basis.batch_cartesian_orbital_offsets) ||
         overlaps_vector(range, basis.batch_primitive_offsets) ||
         overlaps_vector(range, basis.atom_shell_offsets) ||
         overlaps_vector(range, basis.atom_orbital_offsets) ||
         overlaps_vector(range, basis.atom_cartesian_orbital_offsets) ||
         overlaps_vector(range, basis.atom_primitive_offsets) ||
         overlaps_vector(range, basis.shell_orbital_offsets) ||
         overlaps_vector(range, basis.shell_cartesian_orbital_offsets) ||
         overlaps_vector(range, basis.shell_primitive_offsets) ||
         overlaps_vector(range, basis.shell_to_atom) ||
         overlaps_vector(range, basis.principal_quantum_numbers) ||
         overlaps_vector(range, basis.angular_momenta) ||
         overlaps_vector(range, basis.shell_is_valence) ||
         overlaps_vector(range, basis.slater_exponents) ||
         overlaps_vector(range, basis.primitive_exponents) ||
         overlaps_vector(range, basis.primitive_coefficients);
}

bool overlaps_known_plan_storage(const AddressRange& range, const BasisPlan& basis,
                                 const IntegralPlan& integrals,
                                 const CoordinationPlan& coordination,
                                 const RepulsionPlan& repulsion, const H0Plan& h0,
                                 const MullikenPlan& mulliken, const ES2Plan& es2, const D3Plan& d3,
                                 const HalogenPlan& halogen,
                                 const ExternalPointChargePlan* external) {
  const std::size_t bytes = static_cast<std::size_t>(range.end - range.begin);
  const void* pointer = reinterpret_cast<const void*>(range.begin);
  if (overlaps_basis_storage(range, basis) || overlaps_vector(range, integrals.matrix_offsets) ||
      overlaps_vector(range, coordination.atom_offsets) ||
      overlaps_vector(range, coordination.covalent_radius) ||
      overlaps_vector(range, repulsion.atom_offsets) ||
      overlaps_vector(range, repulsion.sqrt_alpha) ||
      overlaps_vector(range, repulsion.effective_charge) ||
      overlaps_vector(range, h0.atom_offsets) || overlaps_vector(range, h0.batch_shell_offsets) ||
      overlaps_vector(range, h0.batch_orbital_offsets) ||
      overlaps_vector(range, h0.matrix_offsets) || overlaps_vector(range, h0.shell_pair_offsets) ||
      overlaps_vector(range, h0.atomic_radii) || overlaps_vector(range, h0.shell_levels) ||
      overlaps_vector(range, h0.shell_coordination_scale) ||
      overlaps_vector(range, h0.shell_polynomial) || overlaps_vector(range, h0.shell_pair_scale) ||
      overlaps_vector(range, mulliken.atom_offsets()) ||
      overlaps_vector(range, mulliken.batch_shell_offsets()) ||
      overlaps_vector(range, mulliken.batch_orbital_offsets()) ||
      overlaps_vector(range, mulliken.matrix_offsets()) ||
      overlaps_vector(range, mulliken.shell_orbital_offsets()) ||
      overlaps_vector(range, mulliken.shell_to_atom()) ||
      overlaps_vector(range, mulliken.orbital_to_shell()) ||
      overlaps_vector(range, mulliken.spin_channels()) ||
      overlaps_vector(range, mulliken.reference_shell_occupations()) ||
      es2.overlaps_storage(pointer, bytes) || d3.overlaps_storage(pointer, bytes) ||
      halogen.overlaps_storage(pointer, bytes)) {
    return true;
  }
  return external != nullptr && (overlaps_vector(range, external->atom_offsets) ||
                                 overlaps_vector(range, external->batch_shell_offsets) ||
                                 overlaps_vector(range, external->point_charge_offsets) ||
                                 overlaps_vector(range, external->shell_to_atom) ||
                                 overlaps_vector(range, external->shell_hardness));
}

bool exact_d3_workspace(const D3Plan& plan, const D3Workspace& workspace) {
  if (workspace.plan_identity != plan.identity() || workspace.workspace_base == nullptr ||
      workspace.workspace_size_bytes < plan.workspace_size_bytes()) {
    return false;
  }
  D3Workspace canonical;
  std::string ignored;
  if (bind_d3_workspace(plan, workspace.workspace_base, plan.workspace_size_bytes(), canonical,
                        ignored) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  return workspace.weights == canonical.weights &&
         workspace.weight_cn_derivatives == canonical.weight_cn_derivatives &&
         workspace.weight_elements == canonical.weight_elements &&
         workspace.coordination_adjoints == canonical.coordination_adjoints &&
         workspace.atom_elements == canonical.atom_elements &&
         workspace.batch_scratch == canonical.batch_scratch &&
         workspace.batch_elements == canonical.batch_elements &&
         workspace.gradient_scratch == canonical.gradient_scratch &&
         workspace.gradient_elements == canonical.gradient_elements;
}

bool exact_halogen_workspace(const HalogenPlan& plan, const HalogenWorkspace& workspace) {
  if (workspace.plan_identity != plan.identity() || workspace.workspace_base == nullptr ||
      workspace.workspace_size_bytes < plan.workspace_size_bytes()) {
    return false;
  }
  HalogenWorkspace canonical;
  std::string ignored;
  if (bind_halogen_workspace(plan, workspace.workspace_base, plan.workspace_size_bytes(), canonical,
                             ignored) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  return workspace.axis_neighbors == canonical.axis_neighbors &&
         workspace.axis_neighbor_elements == canonical.axis_neighbor_elements &&
         workspace.batch_scratch == canonical.batch_scratch &&
         workspace.batch_elements == canonical.batch_elements &&
         workspace.force_scratch == canonical.force_scratch &&
         workspace.force_elements == canonical.force_elements;
}

bool diagnostics_requested(const ComponentGradients& components) {
  return components.electronic != nullptr || components.es2 != nullptr ||
         components.d3 != nullptr || components.repulsion != nullptr ||
         components.halogen != nullptr || components.external_point_charge != nullptr;
}

bool add_component(const double* source, double* destination, std::size_t count) {
  for (std::size_t index = 0; index < count; ++index) {
    const double updated = destination[index] + source[index];
    if (!std::isfinite(updated)) return false;
    destination[index] = updated;
  }
  return true;
}

enum class ComponentIndex : std::size_t {
  kElectronic = 0u,
  kEs2 = 1u,
  kD3 = 2u,
  kRepulsion = 3u,
  kHalogen = 4u,
  kExternal = 5u,
  kCount = 6u,
};

double* staged_component(const ForceWorkspace& workspace, std::int64_t coordinates,
                         ComponentIndex component) {
  return workspace.component_gradient_staging +
         static_cast<std::size_t>(component) * static_cast<std::size_t>(coordinates);
}

bool same_basis_topology(const BasisPlan& basis, const H0Plan& h0) {
  return h0.atom_offsets == basis.atom_offsets &&
         h0.batch_shell_offsets == basis.batch_shell_offsets &&
         h0.batch_orbital_offsets == basis.batch_orbital_offsets;
}

bool exact_es2_topology(const BasisPlan& basis, const ES2Plan& es2) {
  return es2.hardness_average() == gfn2::ES2HardnessAverage::kHarmonic &&
         es2.atom_offsets() == basis.atom_offsets &&
         es2.batch_shell_offsets() == basis.batch_shell_offsets &&
         es2.atom_shell_offsets() == basis.atom_shell_offsets &&
         es2.shell_to_atom() == basis.shell_to_atom;
}

bool exact_mulliken_topology(const BasisPlan& basis, const IntegralPlan& integrals,
                             const MullikenPlan& mulliken) {
  std::int64_t expected_density = 0;
  std::int64_t expected_shell_population = 0;
  std::int64_t expected_atom_population = 0;
  if (mulliken.spin_channels().size() != static_cast<std::size_t>(basis.batch_size)) return false;
  for (std::int64_t system = 0; system < basis.batch_size; ++system) {
    const std::size_t index = static_cast<std::size_t>(system);
    const std::int64_t orbitals =
        basis.batch_orbital_offsets[index + 1u] - basis.batch_orbital_offsets[index];
    const std::int64_t shells =
        basis.batch_shell_offsets[index + 1u] - basis.batch_shell_offsets[index];
    const std::int64_t atoms = basis.atom_offsets[index + 1u] - basis.atom_offsets[index];
    const std::int32_t spin = mulliken.spin_channels()[index];
    std::int64_t system_matrix = 0;
    std::int64_t system_density = 0;
    std::int64_t system_shell_population = 0;
    std::int64_t system_atom_population = 0;
    if ((spin != 1 && spin != 2) || !checked_multiply(orbitals, orbitals, system_matrix) ||
        !checked_multiply(system_matrix, spin, system_density) ||
        !checked_multiply(shells, spin, system_shell_population) ||
        !checked_multiply(atoms, spin, system_atom_population) ||
        expected_density > std::numeric_limits<std::int64_t>::max() - system_density ||
        expected_shell_population >
            std::numeric_limits<std::int64_t>::max() - system_shell_population ||
        expected_atom_population >
            std::numeric_limits<std::int64_t>::max() - system_atom_population) {
      return false;
    }
    expected_density += system_density;
    expected_shell_population += system_shell_population;
    expected_atom_population += system_atom_population;
  }
  const auto& orbital_to_shell = mulliken.orbital_to_shell();
  if (orbital_to_shell.size() != static_cast<std::size_t>(basis.total_orbitals)) return false;
  for (std::int64_t shell = 0; shell < basis.total_shells; ++shell) {
    const std::int64_t begin = basis.shell_orbital_offsets[static_cast<std::size_t>(shell)];
    const std::int64_t end = basis.shell_orbital_offsets[static_cast<std::size_t>(shell) + 1u];
    for (std::int64_t orbital = begin; orbital < end; ++orbital) {
      if (orbital_to_shell[static_cast<std::size_t>(orbital)] != shell) return false;
    }
  }
  return mulliken.atom_offsets() == basis.atom_offsets &&
         mulliken.batch_shell_offsets() == basis.batch_shell_offsets &&
         mulliken.batch_orbital_offsets() == basis.batch_orbital_offsets &&
         mulliken.matrix_offsets() == integrals.matrix_offsets &&
         mulliken.shell_orbital_offsets() == basis.shell_orbital_offsets &&
         mulliken.shell_to_atom() == basis.shell_to_atom &&
         mulliken.density_elements() == expected_density &&
         mulliken.shell_population_elements() == expected_shell_population &&
         mulliken.atom_population_elements() == expected_atom_population &&
         mulliken.reference_shell_occupations().size() ==
             static_cast<std::size_t>(basis.total_shells);
}

xtbloom_status_t validate_canonical_model_parameters(
    const BasisPlan& basis, const CoordinationPlan& coordination, const RepulsionPlan& repulsion,
    const H0Plan& h0, const MullikenPlan& mulliken, const ES2Plan& es2,
    const std::int32_t* atomic_numbers, std::string& error) {
  constexpr double kElectronvoltToHartree = 1.0 / 27.21138505;
  if (atomic_numbers == nullptr ||
      coordination.covalent_radius.size() != static_cast<std::size_t>(basis.total_atoms) ||
      repulsion.sqrt_alpha.size() != static_cast<std::size_t>(basis.total_atoms) ||
      repulsion.effective_charge.size() != static_cast<std::size_t>(basis.total_atoms) ||
      h0.atomic_radii.size() != static_cast<std::size_t>(basis.total_atoms) ||
      h0.shell_levels.size() != static_cast<std::size_t>(basis.total_shells) ||
      h0.shell_coordination_scale.size() != static_cast<std::size_t>(basis.total_shells) ||
      h0.shell_polynomial.size() != static_cast<std::size_t>(basis.total_shells) ||
      es2.shell_hardness().size() != static_cast<std::size_t>(basis.total_shells) ||
      mulliken.reference_shell_occupations().size() !=
          static_cast<std::size_t>(basis.total_shells)) {
    error = "GFN1 force plans have noncanonical parameter extents";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t atom = 0; atom < basis.total_atoms; ++atom) {
    const auto* element = parameters::gfn1::find_element(
        static_cast<std::uint32_t>(atomic_numbers[static_cast<std::size_t>(atom)]));
    if (element == nullptr || element->atomic_number != atomic_numbers[atom] ||
        coordination.covalent_radius[static_cast<std::size_t>(atom)] !=
            element->covalent_radius_bohr ||
        repulsion.sqrt_alpha[static_cast<std::size_t>(atom)] != std::sqrt(element->arep) ||
        repulsion.effective_charge[static_cast<std::size_t>(atom)] != element->zeff ||
        h0.atomic_radii[static_cast<std::size_t>(atom)] != element->atomic_radius_bohr) {
      error = "GFN1 force plans do not share the canonical element parameters";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    const std::int64_t shell_begin = basis.atom_shell_offsets[static_cast<std::size_t>(atom)];
    const std::int64_t shell_end = basis.atom_shell_offsets[static_cast<std::size_t>(atom) + 1u];
    if (shell_end - shell_begin != element->shell_count) {
      error = "GFN1 force basis shell partition disagrees with the element identity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      const std::size_t shell_index = static_cast<std::size_t>(shell);
      const auto& shell_parameter =
          parameters::gfn1::kShells[static_cast<std::size_t>(element->shell_offset) + shell_index -
                                    static_cast<std::size_t>(shell_begin)];
      const bool basis_match =
          basis.principal_quantum_numbers[shell_index] ==
              shell_parameter.principal_quantum_number &&
          basis.angular_momenta[shell_index] == shell_parameter.angular_momentum &&
          basis.shell_is_valence[shell_index] ==
              static_cast<std::uint8_t>(shell_parameter.is_valence) &&
          basis.slater_exponents[shell_index] == shell_parameter.slater;
      if (!basis_match ||
          h0.shell_levels[shell_index] !=
              shell_parameter.level_electronvolt * kElectronvoltToHartree ||
          h0.shell_coordination_scale[shell_index] !=
              shell_parameter.coordination_number_scale_electronvolt * kElectronvoltToHartree ||
          h0.shell_polynomial[shell_index] != shell_parameter.shell_polynomial ||
          es2.shell_hardness()[shell_index] != element->gam * shell_parameter.shell_hubbard_scale ||
          mulliken.reference_shell_occupations()[shell_index] !=
              shell_parameter.reference_occupation) {
        error = "GFN1 force shell parameters are not one canonical model identity";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  if (h0.shell_pair_offsets.size() != static_cast<std::size_t>(basis.batch_size) + 1u ||
      h0.shell_pair_offsets.front() != 0 ||
      h0.shell_pair_scale.size() != static_cast<std::size_t>(h0.shell_pair_offsets.back())) {
    error = "GFN1 H0 shell-pair metadata is not canonical";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t system = 0; system < basis.batch_size; ++system) {
    const std::size_t system_index = static_cast<std::size_t>(system);
    const std::int64_t shell_begin = basis.batch_shell_offsets[system_index];
    const std::int64_t shell_end = basis.batch_shell_offsets[system_index + 1u];
    const std::int64_t shells = shell_end - shell_begin;
    const std::int64_t pair_begin = h0.shell_pair_offsets[system_index];
    if (h0.shell_pair_offsets[system_index + 1u] - pair_begin != shells * shells) {
      error = "GFN1 H0 shell-pair partition is incompatible with the basis";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (std::int64_t first = shell_begin; first < shell_end; ++first) {
      const std::size_t first_index = static_cast<std::size_t>(first);
      const std::int64_t first_atom = basis.shell_to_atom[first_index];
      const auto* first_element = parameters::gfn1::find_element(
          static_cast<std::uint32_t>(atomic_numbers[static_cast<std::size_t>(first_atom)]));
      for (std::int64_t second = shell_begin; second < shell_end; ++second) {
        const std::size_t second_index = static_cast<std::size_t>(second);
        const std::int64_t second_atom = basis.shell_to_atom[second_index];
        const auto* second_element = parameters::gfn1::find_element(
            static_cast<std::uint32_t>(atomic_numbers[static_cast<std::size_t>(second_atom)]));
        double expected = parameters::gfn1::kGlobal.hamiltonian_kpol;
        const bool first_valence = basis.shell_is_valence[first_index] != 0u;
        const bool second_valence = basis.shell_is_valence[second_index] != 0u;
        if (first_valence && second_valence) {
          const std::size_t angular =
              static_cast<std::size_t>(basis.angular_momenta[first_index]) * 3u +
              static_cast<std::size_t>(basis.angular_momenta[second_index]);
          const double delta = first_element->electronegativity - second_element->electronegativity;
          expected = parameters::gfn1::kGlobal.shell_pair_scale[angular] *
                     parameters::gfn1::pair_scale(first_element->atomic_number,
                                                  second_element->atomic_number) *
                     (1.0 + parameters::gfn1::kGlobal.hamiltonian_enscale * delta * delta);
        } else if (first_valence || second_valence) {
          const std::size_t active = first_valence ? first_index : second_index;
          const std::size_t angular = static_cast<std::size_t>(basis.angular_momenta[active]) * 4u;
          expected = 0.5 * (parameters::gfn1::kGlobal.shell_pair_scale[angular] +
                            parameters::gfn1::kGlobal.hamiltonian_kpol);
        }
        const std::size_t pair_index = static_cast<std::size_t>(
            pair_begin + (first - shell_begin) * shells + (second - shell_begin));
        if (h0.shell_pair_scale[pair_index] != expected) {
          error = "GFN1 H0 shell-pair parameters do not match the canonical model";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
      }
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t add_scalar_stationary_overlap_adjoint(const MullikenPlan& mulliken,
                                                       const double* density,
                                                       const double* shell_potentials,
                                                       double* overlap_adjoint,
                                                       bool unrestricted_only, std::string& error) {
  const auto& orbital_offsets = mulliken.batch_orbital_offsets();
  const auto& matrix_offsets = mulliken.matrix_offsets();
  const auto& orbital_to_shell = mulliken.orbital_to_shell();
  for (std::int64_t system = 0; system < mulliken.batch_size(); ++system) {
    /* The optional spin-density buffers are batch-wide because mixed ragged
     * batches share one descriptor. Restricted members own no magnetization
     * response, so ignore their slices even if a caller's reusable scratch
     * contains stale finite values there. */
    if (unrestricted_only && mulliken.spin_channels()[static_cast<std::size_t>(system)] != 2) {
      continue;
    }
    const std::int64_t orbital_begin = orbital_offsets[static_cast<std::size_t>(system)];
    const std::int64_t orbitals =
        orbital_offsets[static_cast<std::size_t>(system + 1)] - orbital_begin;
    const std::int64_t matrix_begin = matrix_offsets[static_cast<std::size_t>(system)];
    for (std::int64_t row = 0; row < orbitals; ++row) {
      const std::int64_t row_shell =
          orbital_to_shell[static_cast<std::size_t>(orbital_begin + row)];
      for (std::int64_t column = row; column < orbitals; ++column) {
        const std::int64_t column_shell =
            orbital_to_shell[static_cast<std::size_t>(orbital_begin + column)];
        const std::int64_t forward = matrix_begin + row * orbitals + column;
        const std::int64_t reverse = matrix_begin + column * orbitals + row;
        const double pair_density =
            density[forward] + (forward == reverse ? 0.0 : density[reverse]);
        const double contribution =
            -0.5 * pair_density * (shell_potentials[row_shell] + shell_potentials[column_shell]);
        const double updated = overlap_adjoint[forward] + contribution;
        if (!std::isfinite(updated)) {
          error = "GFN1 stationary scalar overlap response overflowed";
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        overlap_adjoint[forward] = updated;
      }
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_plans(const BasisPlan& basis, const IntegralPlan& integrals,
                                const CoordinationPlan& coordination,
                                const RepulsionPlan& repulsion, const H0Plan& h0,
                                const MullikenPlan& mulliken, const ES2Plan& es2, const D3Plan& d3,
                                const HalogenPlan& halogen, const ExternalPointChargePlan* external,
                                const std::int32_t* atomic_numbers, std::string& error) {
  const bool valid =
      basis.batch_size > 0 && basis.total_atoms > 0 && basis.total_shells > 0 &&
      basis.total_orbitals > 0 && integrals.batch_size == basis.batch_size &&
      integrals.total_matrix_elements > 0 && coordination.batch_size == basis.batch_size &&
      coordination.total_atoms == basis.total_atoms && repulsion.batch_size == basis.batch_size &&
      repulsion.total_atoms == basis.total_atoms && h0.batch_size == basis.batch_size &&
      h0.total_atoms == basis.total_atoms && h0.total_shells == basis.total_shells &&
      h0.total_orbitals == basis.total_orbitals &&
      h0.total_matrix_elements == integrals.total_matrix_elements && mulliken.sealed() &&
      mulliken.batch_size() == basis.batch_size && mulliken.total_atoms() == basis.total_atoms &&
      mulliken.total_shells() == basis.total_shells &&
      mulliken.total_orbitals() == basis.total_orbitals &&
      mulliken.matrix_elements() == integrals.total_matrix_elements && es2.sealed() &&
      es2.batch_size() == basis.batch_size && es2.total_atoms() == basis.total_atoms &&
      es2.total_shells() == basis.total_shells && atomic_numbers != nullptr && d3.sealed() &&
      d3.batch_size() == basis.batch_size && d3.total_atoms() == basis.total_atoms &&
      halogen.sealed() && halogen.batch_size() == basis.batch_size &&
      halogen.total_atoms() == basis.total_atoms && d3.matches_atomic_numbers(atomic_numbers) &&
      halogen.matches_atomic_numbers(atomic_numbers) && same_basis_topology(basis, h0) &&
      exact_mulliken_topology(basis, integrals, mulliken) && exact_es2_topology(basis, es2) &&
      es2.total_matrix_elements() > 0 &&
      same_offsets(basis.atom_offsets, coordination.atom_offsets) &&
      same_offsets(basis.atom_offsets, repulsion.atom_offsets) &&
      same_offsets(basis.atom_offsets, mulliken.atom_offsets()) &&
      same_offsets(basis.atom_offsets, es2.atom_offsets()) &&
      same_offsets(basis.atom_offsets, d3.atom_offsets()) &&
      same_offsets(basis.atom_offsets, halogen.atom_offsets()) &&
      same_offsets(basis.batch_shell_offsets, h0.batch_shell_offsets) &&
      same_offsets(basis.batch_shell_offsets, mulliken.batch_shell_offsets()) &&
      same_offsets(basis.batch_shell_offsets, es2.batch_shell_offsets()) &&
      same_offsets(integrals.matrix_offsets, h0.matrix_offsets) &&
      same_offsets(integrals.matrix_offsets, mulliken.matrix_offsets());
  if (!valid) {
    error = "GFN1 force plans do not describe one exact ragged topology";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  xtbloom_status_t parameter_status = validate_canonical_model_parameters(
      basis, coordination, repulsion, h0, mulliken, es2, atomic_numbers, error);
  if (parameter_status != XTBLOOM_STATUS_SUCCESS) return parameter_status;
  if (&d3.coordination_plan() != &coordination) {
    /* The accessor returns plan-owned storage. Requiring object identity is
     * intentionally strict: callers must pass d3.coordination_plan() itself
     * so one CN buffer drives both D3 interpolation and the H0/CN force chain. */
    error = "GFN1 force composition must use the D3 plan-owned coordination plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (external != nullptr &&
      (external->batch_size != basis.batch_size || external->total_atoms != basis.total_atoms ||
       external->total_shells != basis.total_shells ||
       !same_offsets(external->atom_offsets, basis.atom_offsets) ||
       !same_offsets(external->batch_shell_offsets, basis.batch_shell_offsets) ||
       external->shell_to_atom != basis.shell_to_atom ||
       external->shell_hardness != es2.shell_hardness())) {
    error = "GFN1 point-charge plan does not match the force topology";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_numerical_ranges(
    const BasisPlan& basis, const IntegralPlan& integrals, const CoordinationPlan& coordination,
    const RepulsionPlan& repulsion, const H0Plan& h0, const MullikenPlan& mulliken,
    const ES2Plan& es2, const ES2GeometryCache& es2_cache, const D3Plan& d3,
    const HalogenPlan& halogen, const ExternalPointChargePlan* external,
    const StationaryInput& input, double* energies, double* qm_forces, double* point_forces,
    const ComponentGradients& components, const ForceWorkspace& workspace, bool force_requested,
    std::int64_t coordinates, std::int64_t points, std::int64_t point_coordinates,
    std::string& error) {
  std::array<AddressRange, 16> reads{};
  std::array<AddressRange, 34> writes{};
  std::array<AddressRange, 16> controls{};
  std::size_t read_count = 0u;
  std::size_t write_count = 0u;
  std::size_t control_count = 0u;
  const auto add_read = [&](const double* pointer, std::int64_t elements) {
    return read_count < reads.size() && make_double_range(pointer, elements, reads[read_count++]);
  };
  const auto add_write = [&](double* pointer, std::int64_t elements) {
    return write_count < writes.size() &&
           make_double_range(pointer, elements, writes[write_count++]);
  };
  const auto add_control = [&](const void* pointer, std::size_t bytes) {
    return control_count < controls.size() && make_range(pointer, bytes, controls[control_count++]);
  };
  const auto add_component = [&](double* pointer) {
    return add_write(pointer, pointer == nullptr ? 0 : coordinates);
  };

  if (!add_read(input.positions, coordinates) ||
      !add_read(input.coordination_numbers, basis.total_atoms) ||
      !add_read(input.scc_free_energies, basis.batch_size) ||
      !add_write(energies, basis.batch_size) ||
      !add_write(workspace.energy_scratch, workspace.energy_elements) ||
      !add_write(workspace.component_energy_scratch, workspace.energy_elements)) {
    error = "GFN1 energy buffers have invalid ranges or alignment";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (force_requested &&
      (!add_read(input.overlap, integrals.total_matrix_elements) ||
       !add_read(input.density, integrals.total_matrix_elements) ||
       !add_read(input.energy_weighted_density, integrals.total_matrix_elements) ||
       !add_read(input.shell_charges, basis.total_shells) ||
       !add_read(input.scalar_shell_potentials, basis.total_shells) ||
       !add_read(input.spin_density,
                 input.spin_density == nullptr ? 0 : integrals.total_matrix_elements) ||
       !add_read(input.spin_shell_potentials,
                 input.spin_shell_potentials == nullptr ? 0 : basis.total_shells) ||
       !add_read(es2_cache.coulomb_matrix, es2.total_matrix_elements()) ||
       !add_read(input.point_positions, points == 0 ? 0 : point_coordinates) ||
       !add_read(input.point_charges, points) || !add_read(input.point_hardnesses, points) ||
       !add_write(qm_forces, coordinates) ||
       !add_write(point_forces, point_forces == nullptr ? 0 : point_coordinates) ||
       !add_write(workspace.total_gradient, workspace.coordinate_elements) ||
       !add_write(workspace.component_gradient, workspace.coordinate_elements) ||
       !add_write(workspace.component_gradient_staging,
                  workspace.component_gradient_staging_elements) ||
       !add_write(workspace.force_scratch, workspace.coordinate_elements) ||
       !add_write(point_forces == nullptr ? nullptr : workspace.point_force_scratch,
                  point_forces == nullptr ? 0 : workspace.point_force_elements) ||
       !add_write(workspace.overlap_adjoint, workspace.overlap_elements) ||
       !add_write(workspace.coordination_adjoint, workspace.atom_elements) ||
       !add_write(workspace.es2_workspace.matrix_scratch,
                  workspace.es2_workspace.matrix_elements) ||
       !add_write(workspace.es2_workspace.shell_scratch, workspace.es2_workspace.shell_elements) ||
       !add_write(workspace.es2_workspace.batch_scratch, workspace.es2_workspace.batch_elements) ||
       !add_write(workspace.es2_workspace.gradient_scratch,
                  workspace.es2_workspace.gradient_elements) ||
       !add_component(components.electronic) || !add_component(components.es2) ||
       !add_component(components.d3) || !add_component(components.repulsion) ||
       !add_component(components.halogen) || !add_component(components.external_point_charge))) {
    error = "GFN1 force buffers have invalid ranges or alignment";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  if (force_requested) {
    AddressRange integral_workspace;
    AddressRange d3_workspace;
    AddressRange halogen_workspace;
    if (!make_range(workspace.integral_workspace, integrals.workspace_size_bytes,
                    integral_workspace) ||
        !make_range(workspace.d3_workspace.workspace_base, d3.workspace_size_bytes(),
                    d3_workspace) ||
        !make_range(workspace.halogen_workspace.workspace_base, halogen.workspace_size_bytes(),
                    halogen_workspace) ||
        write_count + 3u > writes.size()) {
      error = "GFN1 nested force workspaces have invalid address ranges";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    writes[write_count++] = integral_workspace;
    writes[write_count++] = d3_workspace;
    writes[write_count++] = halogen_workspace;
  } else {
    AddressRange d3_workspace;
    AddressRange halogen_workspace;
    if (!make_range(workspace.d3_workspace.workspace_base, d3.workspace_size_bytes(),
                    d3_workspace) ||
        !make_range(workspace.halogen_workspace.workspace_base, halogen.workspace_size_bytes(),
                    halogen_workspace) ||
        write_count + 2u > writes.size()) {
      error = "GFN1 classical workspaces have invalid address ranges";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    writes[write_count++] = d3_workspace;
    writes[write_count++] = halogen_workspace;
  }

  if (!add_control(&basis, sizeof(basis)) || !add_control(&integrals, sizeof(integrals)) ||
      !add_control(&coordination, sizeof(coordination)) ||
      !add_control(&repulsion, sizeof(repulsion)) || !add_control(&h0, sizeof(h0)) ||
      !add_control(&mulliken, sizeof(mulliken)) || !add_control(&es2, sizeof(es2)) ||
      !add_control(&es2_cache, sizeof(es2_cache)) || !add_control(&d3, sizeof(d3)) ||
      !add_control(&halogen, sizeof(halogen)) || !add_control(&input, sizeof(input)) ||
      !add_control(&components, sizeof(components)) ||
      !add_control(&workspace, sizeof(workspace)) || !add_control(&error, sizeof(error)) ||
      (external != nullptr && !add_control(external, sizeof(*external)))) {
    error = "GFN1 force descriptors have invalid address ranges";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  for (std::size_t first = 0u; first < write_count; ++first) {
    if (overlaps_known_plan_storage(writes[first], basis, integrals, coordination, repulsion, h0,
                                    mulliken, es2, d3, halogen, external)) {
      error = "GFN1 force outputs or scratch overlap immutable plan storage";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t second = first + 1u; second < write_count; ++second) {
      if (ranges_overlap(writes[first], writes[second])) {
        error = "GFN1 force outputs and workspaces must be mutually disjoint";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
    for (std::size_t read = 0u; read < read_count; ++read) {
      if (ranges_overlap(writes[first], reads[read])) {
        error = "GFN1 force outputs/workspaces must not overlap inputs or caches";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
    for (std::size_t control = 0u; control < control_count; ++control) {
      if (ranges_overlap(writes[first], controls[control])) {
        error = "GFN1 force outputs/workspaces must not overlap descriptor storage";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  for (std::size_t read = 0u; read < read_count; ++read) {
    if (overlaps_known_plan_storage(reads[read], basis, integrals, coordination, repulsion, h0,
                                    mulliken, es2, d3, halogen, external)) {
      error = "GFN1 force inputs or caches overlap immutable plan storage";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t control = 0u; control < control_count; ++control) {
      if (ranges_overlap(reads[read], controls[control])) {
        error = "GFN1 force inputs/caches must not overlap descriptor storage";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  for (std::size_t first = 0u; first < control_count; ++first) {
    for (std::size_t second = first + 1u; second < control_count; ++second) {
      if (ranges_overlap(controls[first], controls[second])) {
        error = "GFN1 force descriptors must be mutually disjoint";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace

xtbloom_status_t evaluate_gfn1_energy_forces_cpu(
    const BasisPlan& basis, const IntegralPlan& integrals, const CoordinationPlan& coordination,
    const RepulsionPlan& repulsion, const H0Plan& h0, const MullikenPlan& mulliken,
    const ES2Plan& es2, const ES2GeometryCache& es2_cache, const D3Plan& d3,
    const HalogenPlan& halogen, const ExternalPointChargePlan* external,
    const StationaryInput& input, double* energies, double* qm_forces, double* point_forces,
    const ComponentGradients& components, const ForceWorkspace& workspace, std::string& error) {
  xtbloom_status_t status = validate_plans(basis, integrals, coordination, repulsion, h0, mulliken,
                                           es2, d3, halogen, external, input.atomic_numbers, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  const std::int64_t batch = basis.batch_size;
  const std::int64_t atoms = basis.total_atoms;
  const std::int64_t matrix = integrals.total_matrix_elements;
  std::int64_t coordinates = 0;
  std::int64_t point_coordinates = 0;
  if (!checked_multiply(atoms, 3, coordinates) ||
      !checked_multiply(external == nullptr ? 0 : external->total_point_charges, 3,
                        point_coordinates)) {
    error = "GFN1 force coordinate extents overflow signed dimensions";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::int64_t points = external == nullptr ? 0 : external->total_point_charges;
  const bool force_requested =
      qm_forces != nullptr || point_forces != nullptr || diagnostics_requested(components);
  const bool has_unrestricted =
      std::find(mulliken.spin_channels().begin(), mulliken.spin_channels().end(), 2) !=
      mulliken.spin_channels().end();
  if (energies == nullptr || input.positions == nullptr || input.coordination_numbers == nullptr ||
      input.scc_free_energies == nullptr ||
      !valid_scratch(workspace.energy_scratch, workspace.energy_elements, batch) ||
      !valid_scratch(workspace.component_energy_scratch, workspace.energy_elements, batch) ||
      !finite_array(input.positions, static_cast<std::size_t>(coordinates)) ||
      !finite_array(input.coordination_numbers, static_cast<std::size_t>(atoms)) ||
      !finite_array(input.scc_free_energies, static_cast<std::size_t>(batch))) {
    error = "GFN1 energy composition requires finite positions, SCC energy, and energy scratch";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (force_requested && qm_forces == nullptr) {
    error = "GFN1 force composition requires the qm_forces output";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if ((external == nullptr && (input.point_positions != nullptr || input.point_charges != nullptr ||
                               input.point_hardnesses != nullptr || point_forces != nullptr ||
                               components.external_point_charge != nullptr)) ||
      (external != nullptr && points > 0 &&
       (input.point_positions == nullptr || input.point_charges == nullptr ||
        input.point_hardnesses == nullptr)) ||
      (external != nullptr && points == 0 &&
       (input.point_positions != nullptr || input.point_charges != nullptr ||
        input.point_hardnesses != nullptr || point_forces != nullptr)) ||
      (has_unrestricted != (input.spin_density != nullptr)) ||
      ((input.spin_density == nullptr) != (input.spin_shell_potentials == nullptr))) {
    error = "GFN1 optional spin or point-charge force inputs are inconsistent with the topology";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (force_requested &&
      (es2_cache.plan_identity != es2.identity() || input.geometry_generation == 0u ||
       es2_cache.geometry_generation != input.geometry_generation ||
       es2_cache.matrix_elements != es2.total_matrix_elements() ||
       es2_cache.coulomb_matrix == nullptr)) {
    error = "GFN1 ES2 force cache is stale or belongs to another plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!exact_d3_workspace(d3, workspace.d3_workspace) ||
      !exact_halogen_workspace(halogen, workspace.halogen_workspace)) {
    error = "GFN1 classical force workspaces are not exact plan bindings";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (force_requested && (workspace.es2_workspace.gradient_scratch == nullptr ||
                          workspace.es2_workspace.gradient_elements != coordinates ||
                          workspace.es2_workspace.matrix_scratch != nullptr ||
                          workspace.es2_workspace.matrix_elements != 0 ||
                          workspace.es2_workspace.shell_scratch != nullptr ||
                          workspace.es2_workspace.shell_elements != 0 ||
                          workspace.es2_workspace.batch_scratch != nullptr ||
                          workspace.es2_workspace.batch_elements != 0)) {
    error = "GFN1 force ES2 workspace must be the canonical gradient-only binding";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  status = validate_numerical_ranges(basis, integrals, coordination, repulsion, h0, mulliken, es2,
                                     es2_cache, d3, halogen, external, input, energies, qm_forces,
                                     point_forces, components, workspace, force_requested,
                                     coordinates, points, point_coordinates, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  std::copy_n(input.scc_free_energies, static_cast<std::size_t>(batch), workspace.energy_scratch);
  std::fill_n(workspace.component_energy_scratch, static_cast<std::size_t>(batch), 0.0);
  status = add_d3_cpu(d3, input.positions, input.coordination_numbers,
                      workspace.component_energy_scratch, nullptr, workspace.d3_workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (!add_component(workspace.component_energy_scratch, workspace.energy_scratch,
                     static_cast<std::size_t>(batch))) {
    error = "GFN1 D3 energy accumulation overflowed";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  status = add_repulsion_cpu(repulsion, input.positions, workspace.energy_scratch, nullptr, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = add_halogen_cpu(halogen, input.positions, workspace.energy_scratch, nullptr,
                           workspace.halogen_workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  if (!force_requested) {
    std::copy_n(workspace.energy_scratch, static_cast<std::size_t>(batch), energies);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }
  if (input.coordination_numbers == nullptr || input.overlap == nullptr ||
      input.density == nullptr || input.energy_weighted_density == nullptr ||
      input.shell_charges == nullptr || input.scalar_shell_potentials == nullptr ||
      !valid_scratch(workspace.total_gradient, workspace.coordinate_elements, coordinates) ||
      !valid_scratch(workspace.component_gradient, workspace.coordinate_elements, coordinates) ||
      !valid_scratch(workspace.force_scratch, workspace.coordinate_elements, coordinates) ||
      !valid_scratch(workspace.overlap_adjoint, workspace.overlap_elements, matrix) ||
      !valid_scratch(workspace.coordination_adjoint, workspace.atom_elements, atoms) ||
      workspace.integral_workspace == nullptr ||
      workspace.integral_workspace_size < integrals.workspace_size_bytes ||
      (point_forces != nullptr &&
       !valid_scratch(workspace.point_force_scratch, workspace.point_force_elements,
                      point_coordinates))) {
    error = "GFN1 stationary force inputs or scratch are incomplete";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  std::int64_t component_staging_elements = 0;
  if (!checked_multiply(coordinates, static_cast<std::int64_t>(ComponentIndex::kCount),
                        component_staging_elements) ||
      !valid_scratch(workspace.component_gradient_staging,
                     workspace.component_gradient_staging_elements, component_staging_elements)) {
    error = "GFN1 component diagnostic staging is incomplete";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!finite_array(input.overlap, static_cast<std::size_t>(matrix)) ||
      !finite_array(input.density, static_cast<std::size_t>(matrix)) ||
      !finite_array(input.energy_weighted_density, static_cast<std::size_t>(matrix)) ||
      !finite_array(input.shell_charges, static_cast<std::size_t>(basis.total_shells)) ||
      !finite_array(input.scalar_shell_potentials, static_cast<std::size_t>(basis.total_shells)) ||
      (has_unrestricted && (!finite_array(input.spin_density, static_cast<std::size_t>(matrix)) ||
                            !finite_array(input.spin_shell_potentials,
                                          static_cast<std::size_t>(basis.total_shells))))) {
    error = "GFN1 stationary force inputs contain NaN or infinity";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  std::fill_n(workspace.total_gradient, static_cast<std::size_t>(coordinates), 0.0);
  std::fill_n(workspace.component_gradient, static_cast<std::size_t>(coordinates), 0.0);
  std::fill_n(
      workspace.component_gradient_staging,
      static_cast<std::size_t>(coordinates) * static_cast<std::size_t>(ComponentIndex::kCount),
      0.0);
  std::fill_n(workspace.overlap_adjoint, static_cast<std::size_t>(matrix), 0.0);
  std::fill_n(workspace.coordination_adjoint, static_cast<std::size_t>(atoms), 0.0);
  status = add_h0_vjp_cpu(basis, integrals, h0, input.positions, input.coordination_numbers,
                          input.overlap, input.density, workspace.overlap_adjoint,
                          workspace.coordination_adjoint, workspace.component_gradient, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  for (std::int64_t element = 0; element < matrix; ++element) {
    workspace.overlap_adjoint[element] -= input.energy_weighted_density[element];
  }
  status =
      add_scalar_stationary_overlap_adjoint(mulliken, input.density, input.scalar_shell_potentials,
                                            workspace.overlap_adjoint, false, error);
  if (status == XTBLOOM_STATUS_SUCCESS && has_unrestricted) {
    status = add_scalar_stationary_overlap_adjoint(mulliken, input.spin_density,
                                                   input.spin_shell_potentials,
                                                   workspace.overlap_adjoint, true, error);
  }
  if (status == XTBLOOM_STATUS_SUCCESS) {
    status = add_overlap_gradient_cpu(basis, integrals, input.positions, workspace.overlap_adjoint,
                                      workspace.component_gradient, workspace.integral_workspace,
                                      workspace.integral_workspace_size, error);
  }
  if (status == XTBLOOM_STATUS_SUCCESS) {
    status =
        add_coordination_gradient_cpu(coordination, input.positions, workspace.coordination_adjoint,
                                      workspace.component_gradient, error);
  }
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  std::copy_n(workspace.component_gradient, static_cast<std::size_t>(coordinates),
              staged_component(workspace, coordinates, ComponentIndex::kElectronic));
  if (!add_component(workspace.component_gradient, workspace.total_gradient,
                     static_cast<std::size_t>(coordinates))) {
    error = "GFN1 electronic gradient accumulation overflowed";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  std::fill_n(workspace.component_gradient, static_cast<std::size_t>(coordinates), 0.0);
  status = add_es2_gradient_cpu(es2, es2_cache, input.positions, input.geometry_generation,
                                input.shell_charges, workspace.component_gradient,
                                workspace.es2_workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  std::copy_n(workspace.component_gradient, static_cast<std::size_t>(coordinates),
              staged_component(workspace, coordinates, ComponentIndex::kEs2));
  if (!add_component(workspace.component_gradient, workspace.total_gradient,
                     static_cast<std::size_t>(coordinates))) {
    error = "GFN1 ES2 gradient accumulation overflowed";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  std::fill_n(workspace.component_gradient, static_cast<std::size_t>(coordinates), 0.0);
  std::fill_n(workspace.component_energy_scratch, static_cast<std::size_t>(batch), 0.0);
  status = add_d3_cpu(d3, input.positions, input.coordination_numbers,
                      workspace.component_energy_scratch, workspace.component_gradient,
                      workspace.d3_workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  std::copy_n(workspace.component_gradient, static_cast<std::size_t>(coordinates),
              staged_component(workspace, coordinates, ComponentIndex::kD3));
  if (!add_component(workspace.component_gradient, workspace.total_gradient,
                     static_cast<std::size_t>(coordinates))) {
    error = "GFN1 D3 gradient accumulation overflowed";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  std::fill_n(workspace.force_scratch, static_cast<std::size_t>(coordinates), 0.0);
  std::fill_n(workspace.component_energy_scratch, static_cast<std::size_t>(batch), 0.0);
  status = add_repulsion_cpu(repulsion, input.positions, workspace.component_energy_scratch,
                             workspace.force_scratch, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  for (std::int64_t coordinate = 0; coordinate < coordinates; ++coordinate) {
    workspace.component_gradient[coordinate] = -workspace.force_scratch[coordinate];
  }
  std::copy_n(workspace.component_gradient, static_cast<std::size_t>(coordinates),
              staged_component(workspace, coordinates, ComponentIndex::kRepulsion));
  if (!add_component(workspace.component_gradient, workspace.total_gradient,
                     static_cast<std::size_t>(coordinates))) {
    error = "GFN1 repulsion gradient accumulation overflowed";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  std::fill_n(workspace.force_scratch, static_cast<std::size_t>(coordinates), 0.0);
  std::fill_n(workspace.component_energy_scratch, static_cast<std::size_t>(batch), 0.0);
  status = add_halogen_cpu(halogen, input.positions, workspace.component_energy_scratch,
                           workspace.force_scratch, workspace.halogen_workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  for (std::int64_t coordinate = 0; coordinate < coordinates; ++coordinate) {
    workspace.component_gradient[coordinate] = -workspace.force_scratch[coordinate];
  }
  std::copy_n(workspace.component_gradient, static_cast<std::size_t>(coordinates),
              staged_component(workspace, coordinates, ComponentIndex::kHalogen));
  if (!add_component(workspace.component_gradient, workspace.total_gradient,
                     static_cast<std::size_t>(coordinates))) {
    error = "GFN1 halogen gradient accumulation overflowed";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  if (external != nullptr) {
    std::fill_n(workspace.force_scratch, static_cast<std::size_t>(coordinates), 0.0);
    if (point_forces != nullptr) {
      std::fill_n(workspace.point_force_scratch, static_cast<std::size_t>(point_coordinates), 0.0);
    }
    status = add_external_point_charge_forces_cpu(
        *external, input.positions, input.point_positions, input.point_charges,
        input.point_hardnesses, input.shell_charges, workspace.force_scratch,
        point_forces == nullptr ? nullptr : workspace.point_force_scratch, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    if (point_forces != nullptr) {
      for (std::int64_t coordinate = 0; coordinate < point_coordinates; ++coordinate) {
        if (!std::isfinite(workspace.point_force_scratch[coordinate])) {
          error = "GFN1 point-charge force arithmetic overflowed";
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
      }
    }
    for (std::int64_t coordinate = 0; coordinate < coordinates; ++coordinate) {
      workspace.component_gradient[coordinate] = -workspace.force_scratch[coordinate];
    }
    std::copy_n(workspace.component_gradient, static_cast<std::size_t>(coordinates),
                staged_component(workspace, coordinates, ComponentIndex::kExternal));
    if (!add_component(workspace.component_gradient, workspace.total_gradient,
                       static_cast<std::size_t>(coordinates))) {
      error = "GFN1 point-charge gradient accumulation overflowed";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
  }

  if (!finite_array(workspace.energy_scratch, static_cast<std::size_t>(batch)) ||
      !finite_array(workspace.total_gradient, static_cast<std::size_t>(coordinates))) {
    error = "GFN1 total energy or gradient overflowed";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  for (std::int64_t coordinate = 0; coordinate < coordinates; ++coordinate) {
    workspace.force_scratch[coordinate] = -workspace.total_gradient[coordinate];
  }
  std::copy_n(workspace.energy_scratch, static_cast<std::size_t>(batch), energies);
  std::copy_n(workspace.force_scratch, static_cast<std::size_t>(coordinates), qm_forces);
  if (point_forces != nullptr) {
    std::copy_n(workspace.point_force_scratch, static_cast<std::size_t>(point_coordinates),
                point_forces);
  }
  const std::array<std::pair<double*, ComponentIndex>, 6> publications{{
      {components.electronic, ComponentIndex::kElectronic},
      {components.es2, ComponentIndex::kEs2},
      {components.d3, ComponentIndex::kD3},
      {components.repulsion, ComponentIndex::kRepulsion},
      {components.halogen, ComponentIndex::kHalogen},
      {components.external_point_charge, ComponentIndex::kExternal},
  }};
  for (const auto& publication : publications) {
    if (publication.first != nullptr) {
      std::copy_n(staged_component(workspace, coordinates, publication.second),
                  static_cast<std::size_t>(coordinates), publication.first);
    }
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn1
