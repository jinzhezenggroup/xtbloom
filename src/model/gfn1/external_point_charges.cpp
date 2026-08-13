// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn1/external_point_charges.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace xtbloom::detail::gfn1 {
namespace {

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
  bool active = false;
};

bool representable_count(std::int64_t value, std::size_t element_size, bool add_sentinel = false) {
  if (value < 0) {
    return false;
  }
  const auto count = static_cast<std::uint64_t>(value);
  const auto extra = add_sentinel ? std::uint64_t{1} : std::uint64_t{0};
  return count <= std::numeric_limits<std::uint64_t>::max() - extra &&
         count + extra <=
             static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) / element_size &&
         count + extra <= static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
}

bool representable_xyz_count(std::int64_t value) {
  if (value < 0) {
    return false;
  }
  const auto count = static_cast<std::uint64_t>(value);
  return count <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) /
                      (3u * sizeof(double)) &&
         count <= static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / 3u;
}

bool bytes_for(std::int64_t count, std::size_t element_size, std::size_t& bytes) {
  if (!representable_count(count, element_size)) {
    return false;
  }
  bytes = static_cast<std::size_t>(count) * element_size;
  return true;
}

bool xyz_bytes_for(std::int64_t count, std::size_t& bytes) {
  if (!representable_xyz_count(count)) {
    return false;
  }
  bytes = static_cast<std::size_t>(count) * 3u * sizeof(double);
  return true;
}

bool make_range(const void* pointer, std::size_t bytes, AddressRange& range) {
  if (bytes == 0u) {
    range = {};
    return true;
  }
  if (pointer == nullptr) {
    return false;
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  range = {begin, begin + bytes, true};
  return true;
}

bool ranges_overlap(const AddressRange& first, const AddressRange& second) {
  return first.active && second.active && first.begin < second.end && second.begin < first.end;
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

template <typename T>
bool overlaps_vector(const AddressRange& active, const std::vector<T>& values) {
  if (!active.active) {
    return false;
  }
  const std::size_t capacity = values.capacity();
  if (capacity > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    return true;
  }
  AddressRange storage;
  return make_range(values.data(), capacity * sizeof(T), storage) &&
         ranges_overlap(active, storage);
}

bool overlaps_plan_storage(const ExternalPointChargePlan& plan, const AddressRange& range) {
  if (!range.active) {
    return false;
  }
  AddressRange descriptor;
  if (!make_range(&plan, sizeof(plan), descriptor)) {
    return true;
  }
  return ranges_overlap(range, descriptor) || overlaps_vector(range, plan.atom_offsets) ||
         overlaps_vector(range, plan.batch_shell_offsets) ||
         overlaps_vector(range, plan.point_charge_offsets) ||
         overlaps_vector(range, plan.shell_to_atom) || overlaps_vector(range, plan.shell_hardness);
}

bool aligned(const void* pointer, std::size_t alignment) {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

xtbloom_status_t validate_plan(const ExternalPointChargePlan& plan, std::string& error) {
  if (plan.batch_size <= 0 || plan.total_atoms <= 0 || plan.total_shells <= 0 ||
      plan.total_point_charges < 0 ||
      !representable_count(plan.batch_size, sizeof(std::int64_t), true) ||
      !representable_xyz_count(plan.total_atoms) ||
      !representable_count(plan.total_shells, sizeof(double)) ||
      !representable_xyz_count(plan.total_point_charges)) {
    error = "GFN1 external point-charge plan has invalid or unrepresentable extents";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const auto batches = static_cast<std::size_t>(plan.batch_size);
  const auto shells = static_cast<std::size_t>(plan.total_shells);
  if (plan.atom_offsets.size() != batches + 1u || plan.batch_shell_offsets.size() != batches + 1u ||
      plan.point_charge_offsets.size() != batches + 1u || plan.shell_to_atom.size() != shells ||
      plan.shell_hardness.size() != shells || plan.atom_offsets.front() != 0 ||
      plan.atom_offsets.back() != plan.total_atoms || plan.batch_shell_offsets.front() != 0 ||
      plan.batch_shell_offsets.back() != plan.total_shells ||
      plan.point_charge_offsets.front() != 0 ||
      plan.point_charge_offsets.back() != plan.total_point_charges) {
    error = "GFN1 external point-charge plan storage is incomplete or inconsistent";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::size_t batch = 0u; batch < batches; ++batch) {
    const std::int64_t atom_begin = plan.atom_offsets[batch];
    const std::int64_t atom_end = plan.atom_offsets[batch + 1u];
    const std::int64_t shell_begin = plan.batch_shell_offsets[batch];
    const std::int64_t shell_end = plan.batch_shell_offsets[batch + 1u];
    const std::int64_t point_begin = plan.point_charge_offsets[batch];
    const std::int64_t point_end = plan.point_charge_offsets[batch + 1u];
    if (atom_begin < 0 || atom_begin > atom_end || atom_end > plan.total_atoms || shell_begin < 0 ||
        shell_begin > shell_end || shell_end > plan.total_shells || point_begin < 0 ||
        point_begin > point_end || point_end > plan.total_point_charges) {
      error = "GFN1 external point-charge offsets are not valid ragged partitions";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      const std::size_t index = static_cast<std::size_t>(shell);
      if (plan.shell_to_atom[index] < atom_begin || plan.shell_to_atom[index] >= atom_end ||
          !(plan.shell_hardness[index] > 0.0) || !std::isfinite(plan.shell_hardness[index])) {
        error = "GFN1 external point-charge shell metadata is inconsistent";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_geometry(const ExternalPointChargePlan& plan, const double* qm_positions,
                                   const double* point_positions, const double* point_charges,
                                   const double* point_hardnesses, std::string& error) {
  if (!aligned(qm_positions, alignof(double))) {
    error = "GFN1 external point-charge QM positions must not be NULL or misaligned";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::size_t coordinate = 0u; coordinate < static_cast<std::size_t>(plan.total_atoms) * 3u;
       ++coordinate) {
    if (!std::isfinite(qm_positions[coordinate])) {
      error = "GFN1 external point-charge QM positions contain NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  if (plan.total_point_charges == 0) {
    return XTBLOOM_STATUS_SUCCESS;
  }
  if (!aligned(point_positions, alignof(double)) || !aligned(point_charges, alignof(double)) ||
      !aligned(point_hardnesses, alignof(double))) {
    error = "GFN1 external point-charge site inputs must not be NULL or misaligned";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const auto points = static_cast<std::size_t>(plan.total_point_charges);
  for (std::size_t coordinate = 0u; coordinate < points * 3u; ++coordinate) {
    if (!std::isfinite(point_positions[coordinate])) {
      error = "GFN1 external point-charge positions contain NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::size_t point = 0u; point < points; ++point) {
    if (!std::isfinite(point_charges[point]) || !(point_hardnesses[point] > 0.0) ||
        !std::isfinite(point_hardnesses[point])) {
      error = "GFN1 point-charge values must be finite and hardnesses finite positive";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_finite_values(const double* values, std::size_t count,
                                        const char* diagnostic, std::string& error) {
  if (count == 0u) {
    return XTBLOOM_STATUS_SUCCESS;
  }
  if (!aligned(values, alignof(double))) {
    error = diagnostic;
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::size_t index = 0u; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      error = diagnostic;
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

template <std::size_t N>
xtbloom_status_t validate_ranges(const ExternalPointChargePlan& plan,
                                 const std::array<AddressRange, N>& numerical, std::string& error) {
  AddressRange descriptor_range;
  AddressRange error_range;
  if (!make_range(&numerical, sizeof(numerical), descriptor_range) ||
      !make_range(&error, sizeof(error), error_range) || !pairwise_disjoint(numerical) ||
      ranges_overlap(descriptor_range, error_range)) {
    error = "GFN1 external point-charge buffers or control objects overlap";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (const AddressRange& range : numerical) {
    if (overlaps_plan_storage(plan, range) || ranges_overlap(range, descriptor_range) ||
        ranges_overlap(range, error_range)) {
      error = "GFN1 external point-charge buffers overlap plan or control storage";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t make_numerical_ranges(const ExternalPointChargePlan& plan,
                                       const double* qm_positions, const double* point_positions,
                                       const double* point_charges, const double* point_hardnesses,
                                       AddressRange& qm_range, AddressRange& point_position_range,
                                       AddressRange& point_charge_range,
                                       AddressRange& point_hardness_range) {
  std::size_t qm_bytes = 0u;
  std::size_t point_position_bytes = 0u;
  std::size_t point_bytes = 0u;
  if (!xyz_bytes_for(plan.total_atoms, qm_bytes) ||
      !xyz_bytes_for(plan.total_point_charges, point_position_bytes) ||
      !bytes_for(plan.total_point_charges, sizeof(double), point_bytes) ||
      !make_range(qm_positions, qm_bytes, qm_range) ||
      !make_range(point_positions, point_position_bytes, point_position_range) ||
      !make_range(point_charges, point_bytes, point_charge_range) ||
      !make_range(point_hardnesses, point_bytes, point_hardness_range)) {
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

bool screened_pair(const ExternalPointChargePlan& plan, std::size_t shell, std::size_t point,
                   const double* qm_positions, const double* point_positions, double& dx,
                   double& dy, double& dz) {
  const std::size_t atom = static_cast<std::size_t>(plan.shell_to_atom[shell]);
  dx = qm_positions[atom * 3u] - point_positions[point * 3u];
  dy = qm_positions[atom * 3u + 1u] - point_positions[point * 3u + 1u];
  dz = qm_positions[atom * 3u + 2u] - point_positions[point * 3u + 2u];
  if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz)) {
    return false;
  }
  return true;
}

bool finish_screened_pair(double shell_hardness, double point_hardness, double dx, double dy,
                          double dz, double& inverse_distance) {
  /* x^-1 = 0.5 * (1/gamma_s + 1/gamma_p) for the GFN1 harmonic mean x. */
  const double inverse_harmonic = 0.5 / shell_hardness + 0.5 / point_hardness;
  const double distance = std::hypot(std::hypot(dx, dy), std::hypot(dz, inverse_harmonic));
  inverse_distance = 1.0 / distance;
  return distance > 0.0 && std::isfinite(distance) && std::isfinite(inverse_distance);
}

bool potential_value(const ExternalPointChargePlan& plan, std::int64_t shell,
                     std::int64_t point_begin, std::int64_t point_end, const double* qm_positions,
                     const double* point_positions, const double* point_charges,
                     const double* point_hardnesses, double& potential) {
  potential = 0.0;
  const auto shell_index = static_cast<std::size_t>(shell);
  for (std::int64_t point = point_begin; point < point_end; ++point) {
    const auto point_index = static_cast<std::size_t>(point);
    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
    double inverse_distance = 0.0;
    if (!screened_pair(plan, shell_index, point_index, qm_positions, point_positions, dx, dy, dz) ||
        !finish_screened_pair(plan.shell_hardness[shell_index], point_hardnesses[point_index], dx,
                              dy, dz, inverse_distance)) {
      return false;
    }
    const double contribution = point_charges[point_index] * inverse_distance;
    const double updated = potential + contribution;
    if (!std::isfinite(contribution) || !std::isfinite(updated)) {
      return false;
    }
    potential = updated;
  }
  return true;
}

bool force_term(const ExternalPointChargePlan& plan, std::size_t shell, std::size_t point,
                const double* qm_positions, const double* point_positions,
                const double* point_charges, const double* point_hardnesses,
                const double* shell_charges, std::size_t axis, double& term) {
  double dx = 0.0;
  double dy = 0.0;
  double dz = 0.0;
  double inverse_distance = 0.0;
  if (!screened_pair(plan, shell, point, qm_positions, point_positions, dx, dy, dz) ||
      !finish_screened_pair(plan.shell_hardness[shell], point_hardnesses[point], dx, dy, dz,
                            inverse_distance)) {
    return false;
  }
  const double component = axis == 0u ? dx : (axis == 1u ? dy : dz);
  if (component == 0.0) {
    term = 0.0;
    return true;
  }
  const double inverse_cube = inverse_distance * inverse_distance * inverse_distance;
  term = shell_charges[shell] * point_charges[point] * inverse_cube * component;
  return std::isfinite(inverse_cube) && std::isfinite(term);
}

bool qm_force_value(const ExternalPointChargePlan& plan, std::int64_t atom, std::size_t axis,
                    const double* qm_positions, const double* point_positions,
                    const double* point_charges, const double* point_hardnesses,
                    const double* shell_charges, double initial, double& value) {
  value = initial;
  std::size_t batch = 0u;
  while (plan.atom_offsets[batch + 1u] <= atom) {
    ++batch;
  }
  for (std::int64_t shell = plan.batch_shell_offsets[batch];
       shell < plan.batch_shell_offsets[batch + 1u]; ++shell) {
    if (plan.shell_to_atom[static_cast<std::size_t>(shell)] != atom) {
      continue;
    }
    for (std::int64_t point = plan.point_charge_offsets[batch];
         point < plan.point_charge_offsets[batch + 1u]; ++point) {
      double term = 0.0;
      if (!force_term(plan, static_cast<std::size_t>(shell), static_cast<std::size_t>(point),
                      qm_positions, point_positions, point_charges, point_hardnesses, shell_charges,
                      axis, term)) {
        return false;
      }
      const double updated = value + term;
      if (!std::isfinite(updated)) {
        return false;
      }
      value = updated;
    }
  }
  return true;
}

bool point_force_value(const ExternalPointChargePlan& plan, std::int64_t point, std::size_t axis,
                       const double* qm_positions, const double* point_positions,
                       const double* point_charges, const double* point_hardnesses,
                       const double* shell_charges, double initial, double& value) {
  value = initial;
  std::size_t batch = 0u;
  while (plan.point_charge_offsets[batch + 1u] <= point) {
    ++batch;
  }
  for (std::int64_t shell = plan.batch_shell_offsets[batch];
       shell < plan.batch_shell_offsets[batch + 1u]; ++shell) {
    double term = 0.0;
    if (!force_term(plan, static_cast<std::size_t>(shell), static_cast<std::size_t>(point),
                    qm_positions, point_positions, point_charges, point_hardnesses, shell_charges,
                    axis, term)) {
      return false;
    }
    const double updated = value - term;
    if (!std::isfinite(updated)) {
      return false;
    }
    value = updated;
  }
  return true;
}

}  // namespace

xtbloom_status_t make_external_point_charge_plan(const BasisPlan& basis, const ES2Plan& es2,
                                                 std::int64_t total_point_charges,
                                                 const std::int64_t* point_charge_offsets,
                                                 ExternalPointChargePlan& plan,
                                                 std::string& error) {
  if (basis.batch_size <= 0 || basis.total_atoms <= 0 || basis.total_shells <= 0 ||
      total_point_charges < 0 ||
      !representable_count(basis.batch_size, sizeof(std::int64_t), true) ||
      !representable_xyz_count(basis.total_atoms) ||
      !representable_count(basis.total_shells, sizeof(double)) ||
      !representable_xyz_count(total_point_charges) ||
      (total_point_charges != 0 && point_charge_offsets == nullptr)) {
    error = "GFN1 external point-charge plan requires representable basis and point counts";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const auto batches = static_cast<std::size_t>(basis.batch_size);
  const auto atoms = static_cast<std::size_t>(basis.total_atoms);
  const auto shells = static_cast<std::size_t>(basis.total_shells);
  if (basis.atom_offsets.size() != batches + 1u ||
      basis.batch_shell_offsets.size() != batches + 1u ||
      basis.atom_shell_offsets.size() != atoms + 1u || basis.shell_to_atom.size() != shells ||
      basis.atom_offsets.front() != 0 || basis.atom_offsets.back() != basis.total_atoms ||
      basis.batch_shell_offsets.front() != 0 ||
      basis.batch_shell_offsets.back() != basis.total_shells ||
      basis.atom_shell_offsets.front() != 0 ||
      basis.atom_shell_offsets.back() != basis.total_shells || !es2.sealed() ||
      es2.hardness_average() != gfn2::ES2HardnessAverage::kHarmonic ||
      es2.batch_size() != basis.batch_size || es2.total_atoms() != basis.total_atoms ||
      es2.total_shells() != basis.total_shells || es2.atom_offsets() != basis.atom_offsets ||
      es2.batch_shell_offsets() != basis.batch_shell_offsets ||
      es2.atom_shell_offsets() != basis.atom_shell_offsets ||
      es2.shell_to_atom() != basis.shell_to_atom || es2.shell_hardness().size() != shells) {
    error = "GFN1 point-charge plan requires one exact sealed harmonic ES2 topology";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::size_t batch = 0u; batch < batches; ++batch) {
    const std::int64_t atom_begin = basis.atom_offsets[batch];
    const std::int64_t atom_end = basis.atom_offsets[batch + 1u];
    const std::int64_t shell_begin = basis.batch_shell_offsets[batch];
    const std::int64_t shell_end = basis.batch_shell_offsets[batch + 1u];
    if (atom_begin < 0 || atom_begin > atom_end || atom_end > basis.total_atoms ||
        shell_begin < 0 || shell_begin > shell_end || shell_end > basis.total_shells ||
        shell_begin != basis.atom_shell_offsets[static_cast<std::size_t>(atom_begin)] ||
        shell_end != basis.atom_shell_offsets[static_cast<std::size_t>(atom_end)]) {
      error = "GFN1 point-charge basis offsets are not valid ragged partitions";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  if (point_charge_offsets != nullptr) {
    if (point_charge_offsets[0] != 0 ||
        point_charge_offsets[basis.batch_size] != total_point_charges) {
      error = "GFN1 point-charge offsets must span the complete point-site array";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (std::int64_t batch = 0; batch < basis.batch_size; ++batch) {
      if (point_charge_offsets[batch] < 0 ||
          point_charge_offsets[batch] > point_charge_offsets[batch + 1] ||
          point_charge_offsets[batch + 1] > total_point_charges) {
        error = "GFN1 point-charge offsets must be a monotone ragged partition";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  for (std::size_t shell = 0u; shell < shells; ++shell) {
    if (!(es2.shell_hardness()[shell] > 0.0) || !std::isfinite(es2.shell_hardness()[shell])) {
      error = "GFN1 point-charge ES2 shell hardness is invalid";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
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
    created.shell_to_atom = basis.shell_to_atom;
    created.shell_hardness = es2.shell_hardness();
    if (point_charge_offsets == nullptr) {
      created.point_charge_offsets.assign(batches + 1u, 0);
    } else {
      created.point_charge_offsets.assign(point_charge_offsets,
                                          point_charge_offsets + basis.batch_size + 1);
    }
    plan = std::move(created);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN1 external point-charge plan";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  } catch (const std::length_error&) {
    error = "GFN1 external point-charge dimensions exceed host container limits";
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
  status = validate_geometry(plan, qm_positions, point_positions, point_charges, point_hardnesses,
                             error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (!aligned(shell_potentials, alignof(double))) {
    error = "GFN1 point-charge shell potential output must not be NULL or misaligned";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  AddressRange qm_range;
  AddressRange point_position_range;
  AddressRange point_charge_range;
  AddressRange point_hardness_range;
  std::size_t shell_bytes = 0u;
  AddressRange output_range;
  if (make_numerical_ranges(plan, qm_positions, point_positions, point_charges, point_hardnesses,
                            qm_range, point_position_range, point_charge_range,
                            point_hardness_range) != XTBLOOM_STATUS_SUCCESS ||
      !bytes_for(plan.total_shells, sizeof(double), shell_bytes) ||
      !make_range(shell_potentials, shell_bytes, output_range)) {
    error = "GFN1 point-charge potential buffers exceed addressable storage";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::array ranges{qm_range, point_position_range, point_charge_range, point_hardness_range,
                          output_range};
  status = validate_ranges(plan, ranges, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }

  for (std::size_t batch = 0u; batch < static_cast<std::size_t>(plan.batch_size); ++batch) {
    for (std::int64_t shell = plan.batch_shell_offsets[batch];
         shell < plan.batch_shell_offsets[batch + 1u]; ++shell) {
      double potential = 0.0;
      if (!potential_value(plan, shell, plan.point_charge_offsets[batch],
                           plan.point_charge_offsets[batch + 1u], qm_positions, point_positions,
                           point_charges, point_hardnesses, potential)) {
        error = "GFN1 point-charge potential arithmetic exceeded floating-point range";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  for (std::size_t batch = 0u; batch < static_cast<std::size_t>(plan.batch_size); ++batch) {
    for (std::int64_t shell = plan.batch_shell_offsets[batch];
         shell < plan.batch_shell_offsets[batch + 1u]; ++shell) {
      double potential = 0.0;
      (void)potential_value(plan, shell, plan.point_charge_offsets[batch],
                            plan.point_charge_offsets[batch + 1u], qm_positions, point_positions,
                            point_charges, point_hardnesses, potential);
      shell_potentials[static_cast<std::size_t>(shell)] = potential;
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
  status = validate_finite_values(shell_charges, static_cast<std::size_t>(plan.total_shells),
                                  "GFN1 point-charge shell charges are invalid", error);
  if (status == XTBLOOM_STATUS_SUCCESS) {
    status = validate_finite_values(shell_potentials, static_cast<std::size_t>(plan.total_shells),
                                    "GFN1 point-charge shell potentials are invalid", error);
  }
  if (status == XTBLOOM_STATUS_SUCCESS) {
    status = validate_finite_values(energies, static_cast<std::size_t>(plan.batch_size),
                                    "GFN1 point-charge energy accumulators are invalid", error);
  }
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  std::size_t shell_bytes = 0u;
  std::size_t energy_bytes = 0u;
  std::array<AddressRange, 3> ranges;
  if (!bytes_for(plan.total_shells, sizeof(double), shell_bytes) ||
      !bytes_for(plan.batch_size, sizeof(double), energy_bytes) ||
      !make_range(shell_charges, shell_bytes, ranges[0]) ||
      !make_range(shell_potentials, shell_bytes, ranges[1]) ||
      !make_range(energies, energy_bytes, ranges[2])) {
    error = "GFN1 point-charge energy buffers exceed addressable storage";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  status = validate_ranges(plan, ranges, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }

  for (std::size_t batch = 0u; batch < static_cast<std::size_t>(plan.batch_size); ++batch) {
    double value = energies[batch];
    for (std::int64_t shell = plan.batch_shell_offsets[batch];
         shell < plan.batch_shell_offsets[batch + 1u]; ++shell) {
      const std::size_t index = static_cast<std::size_t>(shell);
      value = std::fma(shell_charges[index], shell_potentials[index], value);
      if (!std::isfinite(value)) {
        error = "GFN1 point-charge energy arithmetic exceeded floating-point range";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  for (std::size_t batch = 0u; batch < static_cast<std::size_t>(plan.batch_size); ++batch) {
    double value = energies[batch];
    for (std::int64_t shell = plan.batch_shell_offsets[batch];
         shell < plan.batch_shell_offsets[batch + 1u]; ++shell) {
      const std::size_t index = static_cast<std::size_t>(shell);
      value = std::fma(shell_charges[index], shell_potentials[index], value);
    }
    energies[batch] = value;
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
  status = validate_geometry(plan, qm_positions, point_positions, point_charges, point_hardnesses,
                             error);
  if (status == XTBLOOM_STATUS_SUCCESS) {
    status = validate_finite_values(shell_charges, static_cast<std::size_t>(plan.total_shells),
                                    "GFN1 point-charge shell charges are invalid", error);
  }
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (qm_forces == nullptr && point_forces == nullptr) {
    error = "at least one GFN1 external point-charge force output is required";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (qm_forces != nullptr) {
    status = validate_finite_values(qm_forces, static_cast<std::size_t>(plan.total_atoms) * 3u,
                                    "GFN1 point-charge QM force accumulators are invalid", error);
  }
  if (status == XTBLOOM_STATUS_SUCCESS && point_forces != nullptr) {
    status = validate_finite_values(point_forces,
                                    static_cast<std::size_t>(plan.total_point_charges) * 3u,
                                    "GFN1 point-charge site force accumulators are invalid", error);
  }
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }

  AddressRange qm_range;
  AddressRange point_position_range;
  AddressRange point_charge_range;
  AddressRange point_hardness_range;
  std::size_t shell_bytes = 0u;
  std::size_t qm_force_bytes = 0u;
  std::size_t point_force_bytes = 0u;
  std::array<AddressRange, 7> ranges;
  if (make_numerical_ranges(plan, qm_positions, point_positions, point_charges, point_hardnesses,
                            qm_range, point_position_range, point_charge_range,
                            point_hardness_range) != XTBLOOM_STATUS_SUCCESS ||
      !bytes_for(plan.total_shells, sizeof(double), shell_bytes) ||
      !xyz_bytes_for(plan.total_atoms, qm_force_bytes) ||
      !xyz_bytes_for(plan.total_point_charges, point_force_bytes) ||
      !make_range(shell_charges, shell_bytes, ranges[4]) ||
      !make_range(qm_forces, qm_forces == nullptr ? 0u : qm_force_bytes, ranges[5]) ||
      !make_range(point_forces, point_forces == nullptr ? 0u : point_force_bytes, ranges[6])) {
    error = "GFN1 point-charge force buffers exceed addressable storage";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  ranges[0] = qm_range;
  ranges[1] = point_position_range;
  ranges[2] = point_charge_range;
  ranges[3] = point_hardness_range;
  status = validate_ranges(plan, ranges, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }

  /* Preflight every accumulator in the same contribution order used below. */
  if (qm_forces != nullptr) {
    for (std::int64_t atom = 0; atom < plan.total_atoms; ++atom) {
      for (std::size_t axis = 0u; axis < 3u; ++axis) {
        double value = 0.0;
        if (!qm_force_value(plan, atom, axis, qm_positions, point_positions, point_charges,
                            point_hardnesses, shell_charges,
                            qm_forces[static_cast<std::size_t>(atom) * 3u + axis], value)) {
          error = "GFN1 point-charge QM force arithmetic exceeded floating-point range";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
      }
    }
  }
  if (point_forces != nullptr) {
    for (std::int64_t point = 0; point < plan.total_point_charges; ++point) {
      for (std::size_t axis = 0u; axis < 3u; ++axis) {
        double value = 0.0;
        if (!point_force_value(plan, point, axis, qm_positions, point_positions, point_charges,
                               point_hardnesses, shell_charges,
                               point_forces[static_cast<std::size_t>(point) * 3u + axis], value)) {
          error = "GFN1 point-charge site force arithmetic exceeded floating-point range";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
      }
    }
  }

  if (qm_forces != nullptr) {
    for (std::int64_t atom = 0; atom < plan.total_atoms; ++atom) {
      for (std::size_t axis = 0u; axis < 3u; ++axis) {
        double value = 0.0;
        (void)qm_force_value(plan, atom, axis, qm_positions, point_positions, point_charges,
                             point_hardnesses, shell_charges,
                             qm_forces[static_cast<std::size_t>(atom) * 3u + axis], value);
        qm_forces[static_cast<std::size_t>(atom) * 3u + axis] = value;
      }
    }
  }
  if (point_forces != nullptr) {
    for (std::int64_t point = 0; point < plan.total_point_charges; ++point) {
      for (std::size_t axis = 0u; axis < 3u; ++axis) {
        double value = 0.0;
        (void)point_force_value(plan, point, axis, qm_positions, point_positions, point_charges,
                                point_hardnesses, shell_charges,
                                point_forces[static_cast<std::size_t>(point) * 3u + axis], value);
        point_forces[static_cast<std::size_t>(point) * 3u + axis] = value;
      }
    }
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn1
