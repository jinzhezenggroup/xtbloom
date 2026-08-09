#include "model/gfn2/external_point_charges.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <utility>

#include "data/parameters/gfn2.hpp"

namespace xtbloom::detail::gfn2 {
namespace {

bool representable_as_size(std::int64_t value) {
  return value >= 0 && static_cast<std::uint64_t>(value) <=
                           static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
}

bool representable_xyz_size(std::int64_t count) {
  if (!representable_as_size(count)) {
    return false;
  }
  const auto value = static_cast<std::uint64_t>(count);
  return value <= std::numeric_limits<std::size_t>::max() / 3u &&
         value <= static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / 3u;
}

xtbloom_status_t validate_plan(const ExternalPointChargePlan& plan, std::string& error) {
  if (plan.batch_size <= 0 || plan.total_atoms <= 0 || plan.total_shells <= 0 ||
      plan.total_point_charges < 0 || !representable_as_size(plan.batch_size) ||
      !representable_xyz_size(plan.total_atoms) || !representable_as_size(plan.total_shells) ||
      !representable_xyz_size(plan.total_point_charges) ||
      plan.atom_offsets.size() != static_cast<std::size_t>(plan.batch_size + 1) ||
      plan.batch_shell_offsets.size() != static_cast<std::size_t>(plan.batch_size + 1) ||
      plan.point_charge_offsets.size() != static_cast<std::size_t>(plan.batch_size + 1) ||
      plan.shell_to_atom.size() != static_cast<std::size_t>(plan.total_shells) ||
      plan.shell_hardness.size() != static_cast<std::size_t>(plan.total_shells)) {
    error = "external point-charge plan is incomplete or internally inconsistent";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (plan.atom_offsets.front() != 0 || plan.atom_offsets.back() != plan.total_atoms ||
      plan.batch_shell_offsets.front() != 0 ||
      plan.batch_shell_offsets.back() != plan.total_shells ||
      plan.point_charge_offsets.front() != 0 ||
      plan.point_charge_offsets.back() != plan.total_point_charges) {
    error = "external point-charge plan offsets do not span the stored data";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  for (std::int64_t batch = 0; batch < plan.batch_size; ++batch) {
    const std::size_t index = static_cast<std::size_t>(batch);
    const std::int64_t atom_begin = plan.atom_offsets[index];
    const std::int64_t atom_end = plan.atom_offsets[index + 1u];
    const std::int64_t shell_begin = plan.batch_shell_offsets[index];
    const std::int64_t shell_end = plan.batch_shell_offsets[index + 1u];
    const std::int64_t point_begin = plan.point_charge_offsets[index];
    const std::int64_t point_end = plan.point_charge_offsets[index + 1u];
    if (atom_begin < 0 || atom_begin > atom_end || atom_end > plan.total_atoms || shell_begin < 0 ||
        shell_begin > shell_end || shell_end > plan.total_shells || point_begin < 0 ||
        point_begin > point_end || point_end > plan.total_point_charges) {
      error = "external point-charge plan offsets are not valid ragged partitions";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      const std::size_t shell_index = static_cast<std::size_t>(shell);
      const std::int64_t atom = plan.shell_to_atom[shell_index];
      if (atom < atom_begin || atom >= atom_end || !(plan.shell_hardness[shell_index] > 0.0) ||
          !std::isfinite(plan.shell_hardness[shell_index])) {
        error = "external point-charge shell metadata is internally inconsistent";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }

  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_geometry_inputs(const ExternalPointChargePlan& plan,
                                          const double* qm_positions, const double* point_positions,
                                          const double* point_charges,
                                          const double* point_hardnesses, std::string& error) {
  if (qm_positions == nullptr) {
    error = "external point-charge QM positions must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t atom_count = static_cast<std::size_t>(plan.total_atoms);
  for (std::size_t coordinate = 0; coordinate < atom_count * 3u; ++coordinate) {
    if (!std::isfinite(qm_positions[coordinate])) {
      error = "external point-charge QM positions contain NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  if (plan.total_point_charges == 0) {
    return XTBLOOM_STATUS_SUCCESS;
  }
  if (point_positions == nullptr || point_charges == nullptr || point_hardnesses == nullptr) {
    error = "external point-charge site inputs must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t point_count = static_cast<std::size_t>(plan.total_point_charges);
  for (std::size_t coordinate = 0; coordinate < point_count * 3u; ++coordinate) {
    if (!std::isfinite(point_positions[coordinate])) {
      error = "external point-charge positions contain NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::size_t point = 0; point < point_count; ++point) {
    if (!std::isfinite(point_charges[point]) || !(point_hardnesses[point] > 0.0) ||
        !std::isfinite(point_hardnesses[point])) {
      error = "external point-charge values must be finite and hardnesses must be finite positive";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_shell_values(const ExternalPointChargePlan& plan, const double* values,
                                       const char* null_error, const char* finite_error,
                                       std::string& error) {
  if (values == nullptr) {
    error = null_error;
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t shell_count = static_cast<std::size_t>(plan.total_shells);
  for (std::size_t shell = 0; shell < shell_count; ++shell) {
    if (!std::isfinite(values[shell])) {
      error = finite_error;
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace

xtbloom_status_t make_external_point_charge_plan(
    const BasisPlan& basis, const std::int32_t* atomic_numbers, std::int64_t total_point_charges,
    const std::int64_t* point_charge_offsets, ExternalPointChargePlan& plan, std::string& error) {
  if (basis.batch_size <= 0 || basis.total_atoms <= 0 || basis.total_shells <= 0 ||
      total_point_charges < 0 || !representable_as_size(basis.batch_size) ||
      !representable_xyz_size(basis.total_atoms) || !representable_as_size(basis.total_shells) ||
      !representable_xyz_size(total_point_charges) ||
      static_cast<std::uint64_t>(basis.batch_size) >=
          static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    error = "external point-charge plan requires representable basis and point counts";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (atomic_numbers == nullptr || (total_point_charges != 0 && point_charge_offsets == nullptr)) {
    error = "external point-charge plan inputs must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  const std::size_t batch_count = static_cast<std::size_t>(basis.batch_size);
  const std::size_t atom_count = static_cast<std::size_t>(basis.total_atoms);
  const std::size_t shell_count = static_cast<std::size_t>(basis.total_shells);
  if (basis.atom_offsets.size() != batch_count + 1u ||
      basis.batch_shell_offsets.size() != batch_count + 1u ||
      basis.atom_shell_offsets.size() != atom_count + 1u ||
      basis.shell_to_atom.size() != shell_count ||
      basis.principal_quantum_numbers.size() != shell_count ||
      basis.angular_momenta.size() != shell_count || basis.slater_exponents.size() != shell_count ||
      basis.atom_offsets.front() != 0 || basis.atom_offsets.back() != basis.total_atoms ||
      basis.batch_shell_offsets.front() != 0 ||
      basis.batch_shell_offsets.back() != basis.total_shells ||
      basis.atom_shell_offsets.front() != 0 ||
      basis.atom_shell_offsets.back() != basis.total_shells) {
    error = "external point-charge plan received an inconsistent basis plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  for (std::size_t batch = 0; batch < batch_count; ++batch) {
    const std::int64_t atom_begin = basis.atom_offsets[batch];
    const std::int64_t atom_end = basis.atom_offsets[batch + 1u];
    const std::int64_t shell_begin = basis.batch_shell_offsets[batch];
    const std::int64_t shell_end = basis.batch_shell_offsets[batch + 1u];
    if (atom_begin < 0 || atom_begin > atom_end || atom_end > basis.total_atoms ||
        shell_begin < 0 || shell_begin > shell_end || shell_end > basis.total_shells ||
        shell_begin != basis.atom_shell_offsets[static_cast<std::size_t>(atom_begin)] ||
        shell_end != basis.atom_shell_offsets[static_cast<std::size_t>(atom_end)]) {
      error = "external point-charge basis offsets are not valid ragged partitions";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  if (point_charge_offsets != nullptr) {
    if (point_charge_offsets[0] != 0 ||
        point_charge_offsets[basis.batch_size] != total_point_charges) {
      error = "point-charge offsets must start at zero and end at the total point count";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (std::int64_t batch = 0; batch < basis.batch_size; ++batch) {
      if (point_charge_offsets[batch] < 0 ||
          point_charge_offsets[batch] > point_charge_offsets[batch + 1] ||
          point_charge_offsets[batch + 1] > total_point_charges) {
        error = "point-charge offsets must be a monotone ragged partition";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }

  try {
    ExternalPointChargePlan created;
    created.batch_size = basis.batch_size;
    created.total_atoms = basis.total_atoms;
    created.total_shells = basis.total_shells;
    created.total_point_charges = total_point_charges;
    created.atom_offsets = basis.atom_offsets;
    created.batch_shell_offsets = basis.batch_shell_offsets;
    if (point_charge_offsets == nullptr) {
      created.point_charge_offsets.assign(batch_count + 1u, 0);
    } else {
      created.point_charge_offsets.assign(point_charge_offsets,
                                          point_charge_offsets + basis.batch_size + 1);
    }
    created.shell_to_atom = basis.shell_to_atom;
    created.shell_hardness.resize(shell_count);

    for (std::int64_t atom = 0; atom < basis.total_atoms; ++atom) {
      const std::size_t atom_index = static_cast<std::size_t>(atom);
      const std::int32_t atomic_number = atomic_numbers[atom_index];
      const auto* element =
          parameters::gfn2::find_element(static_cast<std::uint32_t>(atomic_number));
      if (element == nullptr || element->atomic_number != atomic_number || !(element->gam > 0.0) ||
          !std::isfinite(element->gam)) {
        error = "external point-charge plan contains an unsupported element or hardness";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }

      const std::int64_t shell_begin = basis.atom_shell_offsets[atom_index];
      const std::int64_t shell_end = basis.atom_shell_offsets[atom_index + 1u];
      if (shell_begin < 0 || shell_begin > shell_end || shell_end > basis.total_shells ||
          shell_end - shell_begin != element->shell_count) {
        error = "external point-charge element list does not match the basis shell layout";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      const std::size_t parameter_begin = element->shell_offset;
      if (parameter_begin > parameters::gfn2::kShells.size() ||
          element->shell_count > parameters::gfn2::kShells.size() - parameter_begin) {
        error = "external point-charge generated shell parameters are inconsistent";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }

      for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
        const std::size_t shell_index = static_cast<std::size_t>(shell);
        const std::size_t local_shell = static_cast<std::size_t>(shell - shell_begin);
        const auto& shell_parameters = parameters::gfn2::kShells[parameter_begin + local_shell];
        if (basis.shell_to_atom[shell_index] != atom ||
            basis.principal_quantum_numbers[shell_index] !=
                shell_parameters.principal_quantum_number ||
            basis.angular_momenta[shell_index] != shell_parameters.angular_momentum ||
            basis.slater_exponents[shell_index] != shell_parameters.slater ||
            !(shell_parameters.shell_hubbard_scale > 0.0) ||
            !std::isfinite(shell_parameters.shell_hubbard_scale)) {
          error = "external point-charge element list does not match the basis shell metadata";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        const double shell_hardness = element->gam * shell_parameters.shell_hubbard_scale;
        if (!(shell_hardness > 0.0) || !std::isfinite(shell_hardness)) {
          error = "external point-charge shell hardness is invalid";
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        created.shell_hardness[shell_index] = shell_hardness;
      }
    }

    plan = std::move(created);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN2 external point-charge plan";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

xtbloom_status_t evaluate_external_point_charge_potential_cpu(
    const ExternalPointChargePlan& plan, const double* qm_positions, const double* point_positions,
    const double* point_charges, const double* point_hardnesses, double* shell_potentials,
    std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (shell_potentials == nullptr) {
    error = "external point-charge shell potential output must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  status = validate_geometry_inputs(plan, qm_positions, point_positions, point_charges,
                                    point_hardnesses, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }

  for (std::int64_t batch = 0; batch < plan.batch_size; ++batch) {
    const std::size_t batch_index = static_cast<std::size_t>(batch);
    const std::int64_t shell_begin = plan.batch_shell_offsets[batch_index];
    const std::int64_t shell_end = plan.batch_shell_offsets[batch_index + 1u];
    const std::int64_t point_begin = plan.point_charge_offsets[batch_index];
    const std::int64_t point_end = plan.point_charge_offsets[batch_index + 1u];
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      const std::size_t shell_index = static_cast<std::size_t>(shell);
      const std::size_t atom_index = static_cast<std::size_t>(plan.shell_to_atom[shell_index]);
      double potential = 0.0;
      for (std::int64_t point = point_begin; point < point_end; ++point) {
        const std::size_t point_index = static_cast<std::size_t>(point);
        const double dx = qm_positions[atom_index * 3u] - point_positions[point_index * 3u];
        const double dy =
            qm_positions[atom_index * 3u + 1u] - point_positions[point_index * 3u + 1u];
        const double dz =
            qm_positions[atom_index * 3u + 2u] - point_positions[point_index * 3u + 2u];
        if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz)) {
          error = "external point-charge coordinate differences overflow floating-point range";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        const double inverse_average_hardness =
            2.0 / (plan.shell_hardness[shell_index] + point_hardnesses[point_index]);
        const double softened_distance =
            std::hypot(std::hypot(dx, dy), std::hypot(dz, inverse_average_hardness));
        const double contribution = point_charges[point_index] / softened_distance;
        const double updated_potential = potential + contribution;
        if (!(softened_distance > 0.0) || !std::isfinite(softened_distance) ||
            !std::isfinite(contribution) || !std::isfinite(updated_potential)) {
          error = "external point-charge potential arithmetic exceeded floating-point range";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        potential = updated_potential;
      }
      shell_potentials[shell_index] = potential;
    }
  }

  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t add_external_point_charge_energy_cpu(const ExternalPointChargePlan& plan,
                                                      const double* shell_charges,
                                                      const double* shell_potentials,
                                                      double* energies, std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = validate_shell_values(
      plan, shell_charges, "external point-charge shell charges must not be NULL",
      "external point-charge shell charges contain NaN or infinity", error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = validate_shell_values(
      plan, shell_potentials, "external point-charge shell potentials must not be NULL",
      "external point-charge shell potentials contain NaN or infinity", error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (energies == nullptr) {
    error = "external point-charge energies must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  for (std::int64_t batch = 0; batch < plan.batch_size; ++batch) {
    const std::size_t batch_index = static_cast<std::size_t>(batch);
    const std::int64_t shell_begin = plan.batch_shell_offsets[batch_index];
    const std::int64_t shell_end = plan.batch_shell_offsets[batch_index + 1u];
    double energy = 0.0;
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      const std::size_t shell_index = static_cast<std::size_t>(shell);
      const double contribution = shell_charges[shell_index] * shell_potentials[shell_index];
      const double updated_energy = energy + contribution;
      if (!std::isfinite(contribution) || !std::isfinite(updated_energy)) {
        error = "external point-charge energy arithmetic exceeded floating-point range";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      energy = updated_energy;
    }
    const double updated_output = energies[batch_index] + energy;
    if (!std::isfinite(updated_output)) {
      error = "external point-charge accumulated energy exceeded floating-point range";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    energies[batch_index] = updated_output;
  }

  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t add_external_point_charge_forces_cpu(
    const ExternalPointChargePlan& plan, const double* qm_positions, const double* point_positions,
    const double* point_charges, const double* point_hardnesses, const double* shell_charges,
    double* qm_forces, double* point_forces, std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = validate_geometry_inputs(plan, qm_positions, point_positions, point_charges,
                                    point_hardnesses, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = validate_shell_values(
      plan, shell_charges, "external point-charge shell charges must not be NULL",
      "external point-charge shell charges contain NaN or infinity", error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (qm_forces == nullptr && point_forces == nullptr) {
    error = "at least one external point-charge force output must be provided";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  for (std::int64_t batch = 0; batch < plan.batch_size; ++batch) {
    const std::size_t batch_index = static_cast<std::size_t>(batch);
    const std::int64_t shell_begin = plan.batch_shell_offsets[batch_index];
    const std::int64_t shell_end = plan.batch_shell_offsets[batch_index + 1u];
    const std::int64_t point_begin = plan.point_charge_offsets[batch_index];
    const std::int64_t point_end = plan.point_charge_offsets[batch_index + 1u];
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      const std::size_t shell_index = static_cast<std::size_t>(shell);
      const std::size_t atom_index = static_cast<std::size_t>(plan.shell_to_atom[shell_index]);
      for (std::int64_t point = point_begin; point < point_end; ++point) {
        const std::size_t point_index = static_cast<std::size_t>(point);
        const double dx = qm_positions[atom_index * 3u] - point_positions[point_index * 3u];
        const double dy =
            qm_positions[atom_index * 3u + 1u] - point_positions[point_index * 3u + 1u];
        const double dz =
            qm_positions[atom_index * 3u + 2u] - point_positions[point_index * 3u + 2u];
        if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz)) {
          error = "external point-charge coordinate differences overflow floating-point range";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        const double inverse_average_hardness =
            2.0 / (plan.shell_hardness[shell_index] + point_hardnesses[point_index]);
        if (dx == 0.0 && dy == 0.0 && dz == 0.0) {
          continue;
        }
        const double softened_distance =
            std::hypot(std::hypot(dx, dy), std::hypot(dz, inverse_average_hardness));
        const double inverse_distance = 1.0 / softened_distance;
        const double force_scale = shell_charges[shell_index] * point_charges[point_index] *
                                   inverse_distance * inverse_distance * inverse_distance;
        const double fx = force_scale * dx;
        const double fy = force_scale * dy;
        const double fz = force_scale * dz;
        if (!(softened_distance > 0.0) || !std::isfinite(softened_distance) ||
            !std::isfinite(inverse_distance) || !std::isfinite(force_scale) || !std::isfinite(fx) ||
            !std::isfinite(fy) || !std::isfinite(fz)) {
          error = "external point-charge force arithmetic exceeded floating-point range";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        if (qm_forces != nullptr) {
          const std::size_t coordinate = atom_index * 3u;
          const double updated_x = qm_forces[coordinate] + fx;
          const double updated_y = qm_forces[coordinate + 1u] + fy;
          const double updated_z = qm_forces[coordinate + 2u] + fz;
          if (!std::isfinite(qm_forces[coordinate]) || !std::isfinite(qm_forces[coordinate + 1u]) ||
              !std::isfinite(qm_forces[coordinate + 2u]) || !std::isfinite(updated_x) ||
              !std::isfinite(updated_y) || !std::isfinite(updated_z)) {
            error = "external point-charge accumulated QM force exceeded floating-point range";
            return XTBLOOM_STATUS_INVALID_ARGUMENT;
          }
          qm_forces[coordinate] = updated_x;
          qm_forces[coordinate + 1u] = updated_y;
          qm_forces[coordinate + 2u] = updated_z;
        }
        if (point_forces != nullptr) {
          const std::size_t coordinate = point_index * 3u;
          const double updated_x = point_forces[coordinate] - fx;
          const double updated_y = point_forces[coordinate + 1u] - fy;
          const double updated_z = point_forces[coordinate + 2u] - fz;
          if (!std::isfinite(point_forces[coordinate]) ||
              !std::isfinite(point_forces[coordinate + 1u]) ||
              !std::isfinite(point_forces[coordinate + 2u]) || !std::isfinite(updated_x) ||
              !std::isfinite(updated_y) || !std::isfinite(updated_z)) {
            error = "external point-charge accumulated point force exceeded floating-point range";
            return XTBLOOM_STATUS_INVALID_ARGUMENT;
          }
          point_forces[coordinate] = updated_x;
          point_forces[coordinate + 1u] = updated_y;
          point_forces[coordinate + 2u] = updated_z;
        }
      }
    }
  }

  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn2
