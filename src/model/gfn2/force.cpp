#include "model/gfn2/force.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace xtbloom::detail::gfn2 {
namespace {

bool same_offsets(const std::vector<std::int64_t>& first, const std::vector<std::int64_t>& second) {
  return first == second;
}

bool finite_array(const double* values, std::size_t count) {
  if (values == nullptr) {
    return false;
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      return false;
    }
  }
  return true;
}

bool valid_scratch(double* pointer, std::int64_t elements, std::int64_t required) {
  return required >= 0 && elements >= required && (required == 0 || pointer != nullptr);
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool make_range(const void* pointer, std::size_t bytes, AddressRange& range) {
  if (bytes == 0u) {
    if (pointer != nullptr) {
      return false;
    }
    range = {};
    return true;
  }
  if (pointer == nullptr) {
    return false;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  range = {begin, begin + bytes};
  return true;
}

bool make_double_range(const double* pointer, std::int64_t elements, AddressRange& range) {
  if (elements < 0 ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(double))) {
    return false;
  }
  return make_range(pointer, static_cast<std::size_t>(elements) * sizeof(double), range);
}

bool ranges_overlap(const AddressRange& first, const AddressRange& second) {
  return first.begin != first.end && second.begin != second.end && first.begin < second.end &&
         second.begin < first.end;
}

xtbloom_status_t validate_numerical_ranges(
    std::int64_t batch, std::int64_t atoms, std::int64_t shells, std::int64_t matrix,
    std::int64_t points, const ES2Plan& es2, const ES2GeometryCache& es2_cache,
    const AES2Plan& aes2, const AES2GeometryCache& aes2_cache, const D4Plan* d4,
    const D4GeometryCache* d4_cache, const RestrictedGfn2StationaryInput& input, double* energies,
    double* qm_forces, double* point_forces, const RestrictedGfn2ComponentGradients& components,
    const RestrictedGfn2ForceWorkspace& workspace, bool force_requested, std::string& error) {
  std::array<AddressRange, 24> reads{};
  std::array<AddressRange, 32> writes{};
  std::size_t read_count = 0u;
  std::size_t write_count = 0u;
  const auto add_read = [&](const double* pointer, std::int64_t elements) {
    return read_count < reads.size() && make_double_range(pointer, elements, reads[read_count++]);
  };
  const auto add_write = [&](double* pointer, std::int64_t elements) {
    return write_count < writes.size() &&
           make_double_range(pointer, elements, writes[write_count++]);
  };
  const auto add_optional_component = [&](double* pointer) {
    return add_write(pointer, pointer == nullptr ? 0 : atoms * 3);
  };

  if (!add_read(input.positions, atoms * 3) || !add_read(input.scc_energies, batch) ||
      (d4 != nullptr &&
       (!add_read(d4_cache->pair_data, d4_cache->pair_data_elements) ||
        !add_read(d4_cache->coordination_numbers, d4_cache->coordination_elements))) ||
      !add_write(energies, batch) ||
      !add_write(workspace.energy_scratch, workspace.energy_elements) ||
      !add_write(workspace.component_energy_scratch,
                 d4 == nullptr ? 0 : workspace.energy_elements)) {
    error = "restricted GFN2 energy buffers have invalid address ranges";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (force_requested) {
    if (!add_read(input.coordination_numbers, atoms) || !add_read(input.overlap, matrix) ||
        !add_read(input.density, matrix) || !add_read(input.energy_weighted_density, matrix) ||
        !add_read(input.spin_density, input.spin_density == nullptr ? 0 : matrix) ||
        !add_read(input.spin_scalar_shell_potentials,
                  input.spin_scalar_shell_potentials == nullptr ? 0 : shells) ||
        !add_read(input.shell_charges, shells) || !add_read(input.atomic_charges, atoms) ||
        !add_read(input.atomic_dipoles, atoms * 3) ||
        !add_read(input.atomic_quadrupoles, atoms * 6) ||
        !add_read(input.scalar_shell_potentials, shells) ||
        !add_read(input.atomic_dipole_potentials, atoms * 3) ||
        !add_read(input.atomic_quadrupole_potentials, atoms * 6) ||
        !add_read(es2_cache.coulomb_matrix, es2.total_matrix_elements()) ||
        !add_read(aes2_cache.pair_data, aes2.pair_data_elements()) ||
        (points > 0 &&
         (!add_read(input.point_positions, points * 3) || !add_read(input.point_charges, points) ||
          !add_read(input.point_hardnesses, points))) ||
        !add_write(qm_forces, atoms * 3) ||
        !add_write(point_forces, point_forces == nullptr ? 0 : points * 3) ||
        !add_write(workspace.total_gradient, workspace.coordinate_elements) ||
        !add_write(workspace.component_gradient, workspace.coordinate_elements) ||
        !add_write(workspace.force_scratch, workspace.coordinate_elements) ||
        !add_write(workspace.point_force_scratch,
                   point_forces == nullptr ? 0 : workspace.point_force_elements) ||
        !add_write(workspace.overlap_adjoint, workspace.overlap_adjoint_elements) ||
        !add_write(workspace.dipole_adjoint, workspace.dipole_adjoint_elements) ||
        !add_write(workspace.quadrupole_adjoint, workspace.quadrupole_adjoint_elements) ||
        !add_write(workspace.coordination_adjoint, workspace.coordination_adjoint_elements) ||
        !add_write(workspace.es2_workspace.matrix_scratch,
                   workspace.es2_workspace.matrix_elements) ||
        !add_write(workspace.es2_workspace.shell_scratch, workspace.es2_workspace.shell_elements) ||
        !add_write(workspace.es2_workspace.batch_scratch, workspace.es2_workspace.batch_elements) ||
        !add_write(workspace.es2_workspace.gradient_scratch,
                   workspace.es2_workspace.gradient_elements) ||
        !add_write(workspace.aes2_workspace.pair_scratch, workspace.aes2_workspace.pair_elements) ||
        !add_write(workspace.aes2_workspace.potential_scratch,
                   workspace.aes2_workspace.potential_elements) ||
        !add_write(workspace.aes2_workspace.batch_scratch,
                   workspace.aes2_workspace.batch_elements) ||
        !add_write(workspace.aes2_workspace.gradient_scratch,
                   workspace.aes2_workspace.gradient_elements) ||
        !add_write(workspace.aes2_workspace.coordination_scratch,
                   workspace.aes2_workspace.coordination_elements) ||
        !add_optional_component(components.electronic) ||
        !add_optional_component(components.repulsion) || !add_optional_component(components.es2) ||
        !add_optional_component(components.aes2) ||
        !add_optional_component(components.d4_two_body) ||
        !add_optional_component(components.d4_atm) ||
        !add_optional_component(components.external_point_charge)) {
      error = "restricted GFN2 force buffers have invalid address ranges";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    AddressRange integral_range;
    if (!make_range(workspace.integral_workspace, workspace.integral_workspace_size,
                    integral_range) ||
        write_count == writes.size()) {
      error = "restricted GFN2 integral workspace has an invalid address range";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    writes[write_count++] = integral_range;
  }
  if (d4 != nullptr) {
    AddressRange d4_range;
    if (!make_range(workspace.d4_workspace.workspace_base,
                    workspace.d4_workspace.workspace_size_bytes, d4_range) ||
        write_count == writes.size()) {
      error = "restricted GFN2 D4 workspace has an invalid address range";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    writes[write_count++] = d4_range;
  }
  for (std::size_t first = 0u; first < write_count; ++first) {
    for (std::size_t second = first + 1u; second < write_count; ++second) {
      if (ranges_overlap(writes[first], writes[second])) {
        error = "restricted GFN2 outputs and workspaces must be mutually disjoint";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
    for (std::size_t read = 0u; read < read_count; ++read) {
      if (ranges_overlap(writes[first], reads[read])) {
        error = "restricted GFN2 outputs/workspaces must not overlap numerical inputs or caches";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

bool diagnostics_requested(const RestrictedGfn2ComponentGradients& components) {
  return components.electronic != nullptr || components.repulsion != nullptr ||
         components.es2 != nullptr || components.aes2 != nullptr ||
         components.d4_two_body != nullptr || components.d4_atm != nullptr ||
         components.external_point_charge != nullptr;
}

xtbloom_status_t validate_plan_compatibility(
    const BasisPlan& basis, const IntegralPlan& integrals, const CoordinationPlan& coordination,
    const RepulsionPlan& repulsion, const H0Plan& h0, const MullikenPlan& mulliken,
    const ES2Plan& es2, const AES2Plan& aes2, const D4Plan* d4, const D4GeometryCache* d4_cache,
    const ExternalPointChargePlan* external, std::string& error) {
  const bool base_valid =
      basis.batch_size > 0 && basis.total_atoms > 0 && integrals.total_matrix_elements > 0 &&
      mulliken.sealed() && es2.sealed() && aes2.sealed() &&
      coordination.batch_size == basis.batch_size &&
      coordination.total_atoms == basis.total_atoms && repulsion.batch_size == basis.batch_size &&
      repulsion.total_atoms == basis.total_atoms && h0.batch_size == basis.batch_size &&
      h0.total_atoms == basis.total_atoms && h0.total_shells == basis.total_shells &&
      h0.total_orbitals == basis.total_orbitals &&
      h0.total_matrix_elements == integrals.total_matrix_elements &&
      mulliken.batch_size() == basis.batch_size && mulliken.total_atoms() == basis.total_atoms &&
      mulliken.total_shells() == basis.total_shells &&
      mulliken.total_orbitals() == basis.total_orbitals &&
      mulliken.matrix_elements() == integrals.total_matrix_elements &&
      es2.batch_size() == basis.batch_size && es2.total_atoms() == basis.total_atoms &&
      es2.total_shells() == basis.total_shells && aes2.batch_size() == basis.batch_size &&
      aes2.total_atoms() == basis.total_atoms &&
      same_offsets(basis.atom_offsets, coordination.atom_offsets) &&
      same_offsets(basis.atom_offsets, repulsion.atom_offsets) &&
      same_offsets(basis.atom_offsets, mulliken.atom_offsets()) &&
      same_offsets(basis.atom_offsets, es2.atom_offsets()) &&
      same_offsets(basis.atom_offsets, aes2.atom_offsets()) &&
      same_offsets(basis.batch_shell_offsets, h0.batch_shell_offsets) &&
      same_offsets(basis.batch_shell_offsets, mulliken.batch_shell_offsets()) &&
      same_offsets(basis.batch_shell_offsets, es2.batch_shell_offsets()) &&
      same_offsets(integrals.matrix_offsets, h0.matrix_offsets) &&
      same_offsets(integrals.matrix_offsets, mulliken.matrix_offsets());
  if (!base_valid) {
    error = "restricted GFN2 force plans do not describe one exact ragged topology";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int32_t spin_channels : mulliken.spin_channels()) {
    if (spin_channels != 1 && spin_channels != 2) {
      error = "GFN2 force composition requires one or two spin channels";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  if ((d4 == nullptr) != (d4_cache == nullptr)) {
    error = "D4 force plan and geometry cache must be enabled together";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (d4 != nullptr && (!d4->sealed() || d4->batch_size() != basis.batch_size ||
                        d4->total_atoms() != basis.total_atoms ||
                        !same_offsets(d4->atom_offsets(), basis.atom_offsets) ||
                        d4_cache->plan_identity != d4->identity())) {
    error = "D4 force inputs do not match the restricted GFN2 topology";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (external != nullptr &&
      (external->batch_size != basis.batch_size || external->total_atoms != basis.total_atoms ||
       external->total_shells != basis.total_shells ||
       !same_offsets(external->atom_offsets, basis.atom_offsets) ||
       !same_offsets(external->batch_shell_offsets, basis.batch_shell_offsets))) {
    error = "external point-charge force plan does not match the GFN2 topology";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t add_stationary_integral_adjoints(
    const MullikenPlan& mulliken, const double* density, const double* scalar_shell_potentials,
    const double* dipole_potentials, const double* quadrupole_potentials, double* overlap_adjoint,
    double* dipole_adjoint, double* quadrupole_adjoint, std::string& error) {
  const std::int64_t total_matrix = mulliken.matrix_elements();
  const auto& orbital_to_shell = mulliken.orbital_to_shell();
  const auto& orbital_to_atom = mulliken.orbital_to_atom();
  const auto& orbital_offsets = mulliken.batch_orbital_offsets();
  const auto& matrix_offsets = mulliken.matrix_offsets();
  for (std::int64_t system = 0; system < mulliken.batch_size(); ++system) {
    const std::int64_t orbital_begin = orbital_offsets[static_cast<std::size_t>(system)];
    const std::int64_t orbitals =
        orbital_offsets[static_cast<std::size_t>(system + 1)] - orbital_begin;
    const std::int64_t matrix_begin = matrix_offsets[static_cast<std::size_t>(system)];
    for (std::int64_t row_local = 0; row_local < orbitals; ++row_local) {
      const std::int64_t row = orbital_begin + row_local;
      const std::int64_t row_shell = orbital_to_shell[static_cast<std::size_t>(row)];
      const std::int64_t row_atom = orbital_to_atom[static_cast<std::size_t>(row)];
      for (std::int64_t column_local = row_local; column_local < orbitals; ++column_local) {
        const std::int64_t column = orbital_begin + column_local;
        const std::int64_t column_shell = orbital_to_shell[static_cast<std::size_t>(column)];
        const std::int64_t column_atom = orbital_to_atom[static_cast<std::size_t>(column)];
        const std::int64_t forward = matrix_begin + row_local * orbitals + column_local;
        const std::int64_t reverse = matrix_begin + column_local * orbitals + row_local;
        const double pair_density =
            density[forward] + (forward == reverse ? 0.0 : density[reverse]);
        const double scalar_factor =
            -0.5 * (scalar_shell_potentials[row_shell] + scalar_shell_potentials[column_shell]);
        overlap_adjoint[forward] += pair_density * scalar_factor;
        for (std::int64_t component = 0; dipole_potentials != nullptr && component < 3;
             ++component) {
          const std::int64_t forward_index = component * total_matrix + forward;
          const std::int64_t reverse_index = component * total_matrix + reverse;
          dipole_adjoint[forward_index] +=
              -0.5 * pair_density * dipole_potentials[column_atom * 3 + component];
          dipole_adjoint[reverse_index] +=
              -0.5 * pair_density * dipole_potentials[row_atom * 3 + component];
        }
        for (std::int64_t component = 0; quadrupole_potentials != nullptr && component < 6;
             ++component) {
          const std::int64_t forward_index = component * total_matrix + forward;
          const std::int64_t reverse_index = component * total_matrix + reverse;
          quadrupole_adjoint[forward_index] +=
              -0.5 * pair_density * quadrupole_potentials[column_atom * 6 + component];
          quadrupole_adjoint[reverse_index] +=
              -0.5 * pair_density * quadrupole_potentials[row_atom * 6 + component];
        }
      }
    }
  }
  if (!finite_array(overlap_adjoint, static_cast<std::size_t>(total_matrix)) ||
      !finite_array(dipole_adjoint, static_cast<std::size_t>(total_matrix) * 3u) ||
      !finite_array(quadrupole_adjoint, static_cast<std::size_t>(total_matrix) * 6u)) {
    error = "stationary Mulliken integral adjoints overflowed";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

void add_component(const double* component, double* total, std::size_t count) {
  for (std::size_t index = 0; index < count; ++index) {
    total[index] += component[index];
  }
}

void publish_component(const double* source, double* destination, std::size_t count) {
  if (destination != nullptr) {
    std::copy_n(source, count, destination);
  }
}

}  // namespace

xtbloom_status_t evaluate_restricted_gfn2_energy_forces_cpu(
    const BasisPlan& basis, const IntegralPlan& integrals, const CoordinationPlan& coordination,
    const RepulsionPlan& repulsion, const H0Plan& h0, const MullikenPlan& mulliken,
    const ES2Plan& es2, const ES2GeometryCache& es2_cache, const AES2Plan& aes2,
    const AES2GeometryCache& aes2_cache, const D4Plan* d4, const D4GeometryCache* d4_cache,
    const ExternalPointChargePlan* external, const RestrictedGfn2StationaryInput& input,
    double* energies, double* qm_forces, double* point_forces,
    const RestrictedGfn2ComponentGradients& components,
    const RestrictedGfn2ForceWorkspace& workspace, std::string& error) {
  xtbloom_status_t status =
      validate_plan_compatibility(basis, integrals, coordination, repulsion, h0, mulliken, es2,
                                  aes2, d4, d4_cache, external, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  const std::int64_t batch = basis.batch_size;
  const std::int64_t atoms = basis.total_atoms;
  const std::int64_t shells = basis.total_shells;
  const std::int64_t matrix = integrals.total_matrix_elements;
  const std::int64_t coordinates = atoms * 3;
  const std::int64_t points = external == nullptr ? 0 : external->total_point_charges;
  const bool force_requested =
      qm_forces != nullptr || point_forces != nullptr || diagnostics_requested(components);
  if (energies == nullptr || input.positions == nullptr || input.scc_energies == nullptr ||
      !valid_scratch(workspace.energy_scratch, workspace.energy_elements, batch) ||
      (d4 != nullptr &&
       !valid_scratch(workspace.component_energy_scratch, workspace.energy_elements, batch))) {
    error =
        "restricted GFN2 energy output, positions, SCC energies, and energy scratch are required";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!finite_array(input.positions, static_cast<std::size_t>(coordinates)) ||
      !finite_array(input.scc_energies, static_cast<std::size_t>(batch))) {
    error = "restricted GFN2 positions and SCC energies must be finite";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const bool external_enabled = external != nullptr;
  const bool has_unrestricted_system =
      std::find(mulliken.spin_channels().begin(), mulliken.spin_channels().end(), 2) !=
      mulliken.spin_channels().end();
  if (force_requested &&
      ((input.spin_density == nullptr) != (input.spin_scalar_shell_potentials == nullptr) ||
       (has_unrestricted_system && input.spin_density == nullptr) ||
       (!has_unrestricted_system && input.spin_density != nullptr))) {
    error = "spin density and magnetization shell potentials are inconsistent with the topology";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const bool point_data_required = force_requested && external_enabled && points > 0;
  if ((!external_enabled && (input.point_positions != nullptr || input.point_charges != nullptr ||
                             input.point_hardnesses != nullptr || point_forces != nullptr ||
                             components.external_point_charge != nullptr)) ||
      (point_data_required && (input.point_positions == nullptr || input.point_charges == nullptr ||
                               input.point_hardnesses == nullptr)) ||
      (force_requested && external_enabled && points == 0 &&
       (input.point_positions != nullptr || input.point_charges != nullptr ||
        input.point_hardnesses != nullptr)) ||
      (d4 == nullptr && (components.d4_two_body != nullptr || components.d4_atm != nullptr))) {
    error = "external point-charge plan, numerical inputs, and point-force output are inconsistent";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (point_forces != nullptr &&
      !valid_scratch(workspace.point_force_scratch, workspace.point_force_elements, points * 3)) {
    error = "point-force scratch is missing or too small";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (d4 != nullptr && workspace.d4_workspace.plan_identity != d4->identity()) {
    error = "D4 force workspace is not canonically bound to the enabled plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  status = validate_numerical_ranges(batch, atoms, shells, matrix, points, es2, es2_cache, aes2,
                                     aes2_cache, d4, d4_cache, input, energies, qm_forces,
                                     point_forces, components, workspace, force_requested, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }

  std::copy_n(input.scc_energies, static_cast<std::size_t>(batch), workspace.energy_scratch);
  double* repulsion_force = force_requested ? workspace.force_scratch : nullptr;
  if (force_requested) {
    if (!valid_scratch(workspace.total_gradient, workspace.coordinate_elements, coordinates) ||
        !valid_scratch(workspace.component_gradient, workspace.coordinate_elements, coordinates) ||
        !valid_scratch(workspace.force_scratch, workspace.coordinate_elements, coordinates)) {
      error = "restricted GFN2 force scratch is missing or too small";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    std::fill_n(repulsion_force, static_cast<std::size_t>(coordinates), 0.0);
  }
  status = add_repulsion_cpu(repulsion, input.positions, workspace.energy_scratch, repulsion_force,
                             error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (force_requested) {
    for (std::int64_t coordinate = 0; coordinate < coordinates; ++coordinate) {
      repulsion_force[coordinate] = -repulsion_force[coordinate];
    }
    publish_component(repulsion_force, components.repulsion, static_cast<std::size_t>(coordinates));
  }
  if (d4 != nullptr) {
    status = evaluate_d4_atm_cpu(*d4, *d4_cache, workspace.component_energy_scratch,
                                 workspace.d4_workspace, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
    for (std::int64_t system = 0; system < batch; ++system) {
      const double updated =
          workspace.energy_scratch[system] + workspace.component_energy_scratch[system];
      if (!std::isfinite(updated)) {
        error = "restricted GFN2 D4 ATM energy accumulation overflowed";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      workspace.energy_scratch[system] = updated;
    }
  }

  if (!force_requested) {
    std::copy_n(workspace.energy_scratch, static_cast<std::size_t>(batch), energies);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }
  if (qm_forces == nullptr) {
    error = "QM force output is required whenever force diagnostics or point forces are requested";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (input.coordination_numbers == nullptr || input.overlap == nullptr ||
      input.density == nullptr || input.energy_weighted_density == nullptr ||
      input.shell_charges == nullptr || input.atomic_charges == nullptr ||
      input.atomic_dipoles == nullptr || input.atomic_quadrupoles == nullptr ||
      input.scalar_shell_potentials == nullptr || input.atomic_dipole_potentials == nullptr ||
      input.atomic_quadrupole_potentials == nullptr || input.geometry_generation == 0u) {
    error = "restricted stationary force inputs are incomplete";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!valid_scratch(workspace.overlap_adjoint, workspace.overlap_adjoint_elements, matrix) ||
      !valid_scratch(workspace.dipole_adjoint, workspace.dipole_adjoint_elements, matrix * 3) ||
      !valid_scratch(workspace.quadrupole_adjoint, workspace.quadrupole_adjoint_elements,
                     matrix * 6) ||
      !valid_scratch(workspace.coordination_adjoint, workspace.coordination_adjoint_elements,
                     atoms) ||
      workspace.integral_workspace == nullptr ||
      workspace.integral_workspace_size < integrals.workspace_size_bytes) {
    error = "restricted stationary adjoint or integral scratch is missing or too small";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!finite_array(input.coordination_numbers, static_cast<std::size_t>(atoms)) ||
      !finite_array(input.overlap, static_cast<std::size_t>(matrix)) ||
      !finite_array(input.density, static_cast<std::size_t>(matrix)) ||
      !finite_array(input.energy_weighted_density, static_cast<std::size_t>(matrix)) ||
      (has_unrestricted_system &&
       (!finite_array(input.spin_density, static_cast<std::size_t>(matrix)) ||
        !finite_array(input.spin_scalar_shell_potentials, static_cast<std::size_t>(shells)))) ||
      !finite_array(input.shell_charges, static_cast<std::size_t>(shells)) ||
      !finite_array(input.atomic_charges, static_cast<std::size_t>(atoms)) ||
      !finite_array(input.atomic_dipoles, static_cast<std::size_t>(atoms) * 3u) ||
      !finite_array(input.atomic_quadrupoles, static_cast<std::size_t>(atoms) * 6u) ||
      !finite_array(input.scalar_shell_potentials, static_cast<std::size_t>(shells)) ||
      !finite_array(input.atomic_dipole_potentials, static_cast<std::size_t>(atoms) * 3u) ||
      !finite_array(input.atomic_quadrupole_potentials, static_cast<std::size_t>(atoms) * 6u)) {
    error = "restricted stationary force inputs contain NaN or infinity";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  std::fill_n(workspace.total_gradient, static_cast<std::size_t>(coordinates), 0.0);
  add_component(repulsion_force, workspace.total_gradient, static_cast<std::size_t>(coordinates));
  std::fill_n(workspace.component_gradient, static_cast<std::size_t>(coordinates), 0.0);
  std::fill_n(workspace.overlap_adjoint, static_cast<std::size_t>(matrix), 0.0);
  std::fill_n(workspace.dipole_adjoint, static_cast<std::size_t>(matrix) * 3u, 0.0);
  std::fill_n(workspace.quadrupole_adjoint, static_cast<std::size_t>(matrix) * 6u, 0.0);
  std::fill_n(workspace.coordination_adjoint, static_cast<std::size_t>(atoms), 0.0);

  status = add_h0_vjp_cpu(basis, integrals, h0, input.positions, input.coordination_numbers,
                          input.overlap, input.density, workspace.overlap_adjoint,
                          workspace.coordination_adjoint, workspace.component_gradient, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  for (std::int64_t element = 0; element < matrix; ++element) {
    workspace.overlap_adjoint[element] -= input.energy_weighted_density[element];
  }
  status = add_stationary_integral_adjoints(
      mulliken, input.density, input.scalar_shell_potentials, input.atomic_dipole_potentials,
      input.atomic_quadrupole_potentials, workspace.overlap_adjoint, workspace.dipole_adjoint,
      workspace.quadrupole_adjoint, error);
  if (status == XTBLOOM_STATUS_SUCCESS && has_unrestricted_system) {
    /* Spin polarization is geometry independent at fixed magnetization, but
     * magnetization itself is a Mulliken overlap population. Its stationary
     * response therefore contributes through P_alpha-P_beta and v_mag. */
    status = add_stationary_integral_adjoints(
        mulliken, input.spin_density, input.spin_scalar_shell_potentials, nullptr, nullptr,
        workspace.overlap_adjoint, workspace.dipole_adjoint, workspace.quadrupole_adjoint, error);
  }
  if (status == XTBLOOM_STATUS_SUCCESS) {
    status = add_overlap_gradient_cpu(basis, integrals, input.positions, workspace.overlap_adjoint,
                                      workspace.component_gradient, workspace.integral_workspace,
                                      workspace.integral_workspace_size, error);
  }
  if (status == XTBLOOM_STATUS_SUCCESS) {
    status = add_multipole_gradient_cpu(
        basis, integrals, input.positions, workspace.dipole_adjoint, workspace.quadrupole_adjoint,
        workspace.component_gradient, workspace.integral_workspace,
        workspace.integral_workspace_size, error, mulliken.cpu_isa());
  }
  if (status == XTBLOOM_STATUS_SUCCESS) {
    status =
        add_coordination_gradient_cpu(coordination, input.positions, workspace.coordination_adjoint,
                                      workspace.component_gradient, error);
  }
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  publish_component(workspace.component_gradient, components.electronic,
                    static_cast<std::size_t>(coordinates));
  add_component(workspace.component_gradient, workspace.total_gradient,
                static_cast<std::size_t>(coordinates));

  std::fill_n(workspace.component_gradient, static_cast<std::size_t>(coordinates), 0.0);
  status = add_es2_gradient_cpu(es2, es2_cache, input.positions, input.geometry_generation,
                                input.shell_charges, workspace.component_gradient,
                                workspace.es2_workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  publish_component(workspace.component_gradient, components.es2,
                    static_cast<std::size_t>(coordinates));
  add_component(workspace.component_gradient, workspace.total_gradient,
                static_cast<std::size_t>(coordinates));

  std::fill_n(workspace.component_gradient, static_cast<std::size_t>(coordinates), 0.0);
  std::fill_n(workspace.coordination_adjoint, static_cast<std::size_t>(atoms), 0.0);
  status = add_aes2_vjp_cpu(aes2, aes2_cache, input.positions, input.coordination_numbers,
                            input.geometry_generation, input.atomic_charges, input.atomic_dipoles,
                            input.atomic_quadrupoles, workspace.component_gradient,
                            workspace.coordination_adjoint, workspace.aes2_workspace, error);
  if (status == XTBLOOM_STATUS_SUCCESS) {
    status =
        add_coordination_gradient_cpu(coordination, input.positions, workspace.coordination_adjoint,
                                      workspace.component_gradient, error);
  }
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  publish_component(workspace.component_gradient, components.aes2,
                    static_cast<std::size_t>(coordinates));
  add_component(workspace.component_gradient, workspace.total_gradient,
                static_cast<std::size_t>(coordinates));

  if (d4 != nullptr) {
    std::fill_n(workspace.component_gradient, static_cast<std::size_t>(coordinates), 0.0);
    status =
        add_d4_two_body_gradient_cpu(*d4, *d4_cache, input.atomic_charges,
                                     workspace.component_gradient, workspace.d4_workspace, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
    publish_component(workspace.component_gradient, components.d4_two_body,
                      static_cast<std::size_t>(coordinates));
    add_component(workspace.component_gradient, workspace.total_gradient,
                  static_cast<std::size_t>(coordinates));

    std::fill_n(workspace.component_gradient, static_cast<std::size_t>(coordinates), 0.0);
    status = add_d4_atm_gradient_cpu(*d4, *d4_cache, workspace.component_gradient,
                                     workspace.d4_workspace, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
    publish_component(workspace.component_gradient, components.d4_atm,
                      static_cast<std::size_t>(coordinates));
    add_component(workspace.component_gradient, workspace.total_gradient,
                  static_cast<std::size_t>(coordinates));
  }

  if (external_enabled) {
    std::fill_n(workspace.force_scratch, static_cast<std::size_t>(coordinates), 0.0);
    if (point_forces != nullptr) {
      std::fill_n(workspace.point_force_scratch, static_cast<std::size_t>(points) * 3u, 0.0);
    }
    status = add_external_point_charge_forces_cpu(
        *external, input.positions, input.point_positions, input.point_charges,
        input.point_hardnesses, input.shell_charges, workspace.force_scratch,
        point_forces == nullptr ? nullptr : workspace.point_force_scratch, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
    for (std::int64_t coordinate = 0; coordinate < coordinates; ++coordinate) {
      workspace.component_gradient[coordinate] = -workspace.force_scratch[coordinate];
    }
    publish_component(workspace.component_gradient, components.external_point_charge,
                      static_cast<std::size_t>(coordinates));
    add_component(workspace.component_gradient, workspace.total_gradient,
                  static_cast<std::size_t>(coordinates));
  }

  if (!finite_array(workspace.total_gradient, static_cast<std::size_t>(coordinates)) ||
      !finite_array(workspace.energy_scratch, static_cast<std::size_t>(batch))) {
    error = "restricted GFN2 total energy or gradient overflowed";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  for (std::int64_t coordinate = 0; coordinate < coordinates; ++coordinate) {
    workspace.force_scratch[coordinate] = -workspace.total_gradient[coordinate];
  }
  std::copy_n(workspace.energy_scratch, static_cast<std::size_t>(batch), energies);
  std::copy_n(workspace.force_scratch, static_cast<std::size_t>(coordinates), qm_forces);
  if (point_forces != nullptr) {
    std::copy_n(workspace.point_force_scratch, static_cast<std::size_t>(points) * 3u, point_forces);
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn2
