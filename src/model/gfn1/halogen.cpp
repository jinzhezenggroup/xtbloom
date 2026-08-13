// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn1/halogen.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "data/parameters/gfn1.hpp"

namespace xtbloom::detail::gfn1 {

struct HalogenPlanData {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> scaled_atomic_radius;
  std::vector<double> bond_strength;
  std::vector<std::uint8_t> donor;
  std::vector<std::uint8_t> acceptor;
  std::size_t workspace_size_bytes = 0u;
  std::size_t axis_neighbor_offset = 0u;
  std::size_t batch_scratch_offset = 0u;
  std::size_t force_scratch_offset = 0u;
};

namespace {

constexpr double kCutoffBohr = 20.0;

static_assert(parameters::gfn1::kGlobal.halogen_damping == 0.44,
              "GFN1 halogen damping must match the pinned tblite model");
static_assert(parameters::gfn1::kGlobal.halogen_radius_scale == 1.3,
              "GFN1 halogen radii must match the pinned tblite model");

bool is_donor(std::int32_t atomic_number) {
  return atomic_number == 17 || atomic_number == 35 || atomic_number == 53 || atomic_number == 85;
}

bool is_acceptor(std::int32_t atomic_number) {
  return atomic_number == 7 || atomic_number == 8 || atomic_number == 15 || atomic_number == 16;
}

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

bool valid_batch_count(std::int64_t value) {
  return value > 0 && static_cast<std::uint64_t>(value) <
                          static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
}

bool valid_atom_count(std::int64_t value) {
  if (value <= 0) {
    return false;
  }
  const auto count = static_cast<std::uint64_t>(value);
  return count <= static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / 3u &&
         count <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) / 3u;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

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

bool aligned(const void* pointer, std::size_t alignment) {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

template <typename T>
T* offset_pointer(void* base, std::size_t offset) {
  return reinterpret_cast<T*>(static_cast<std::byte*>(base) + offset);
}

template <typename T>
bool overlaps_vector(const AddressRange& active, const std::vector<T>& values) {
  std::size_t bytes = 0u;
  AddressRange storage;
  return checked_multiply_size(values.capacity(), sizeof(T), bytes) &&
         make_range(values.data(), bytes, storage) && ranges_overlap(active, storage);
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
  return align_up(cursor, alignment, offset) && checked_add_size(offset, bytes, cursor);
}

xtbloom_status_t validate_plan(const HalogenPlan& plan, std::string& error) {
  if (!plan.sealed() || plan.batch_size() <= 0 || plan.total_atoms() <= 0) {
    error = "GFN1 halogen plan is not sealed or has invalid extents";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_workspace(const HalogenPlan& plan, const HalogenWorkspace& workspace,
                                    std::string& error) {
  const HalogenPlanData& data = *plan.identity();
  if (workspace.plan_identity != plan.identity() ||
      !aligned(workspace.workspace_base, kHalogenWorkspaceAlignment) ||
      workspace.workspace_size_bytes < plan.workspace_size_bytes() ||
      workspace.axis_neighbors !=
          offset_pointer<std::int64_t>(workspace.workspace_base, data.axis_neighbor_offset) ||
      workspace.batch_scratch !=
          offset_pointer<double>(workspace.workspace_base, data.batch_scratch_offset) ||
      workspace.force_scratch !=
          offset_pointer<double>(workspace.workspace_base, data.force_scratch_offset) ||
      workspace.axis_neighbor_elements != plan.total_atoms() ||
      workspace.batch_elements != plan.batch_size() ||
      workspace.force_elements != plan.total_atoms() * 3) {
    error = "GFN1 halogen workspace is incomplete or belongs to another plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

bool valid_call_storage(const HalogenPlan& plan, const HalogenWorkspace& workspace,
                        const double* positions, double* energies, double* forces,
                        std::string& error) {
  std::size_t position_bytes = 0u;
  std::size_t energy_bytes = 0u;
  AddressRange position_range;
  AddressRange energy_range;
  AddressRange force_range;
  AddressRange workspace_range;
  std::array<AddressRange, 3> controls{};
  if (!checked_multiply_size(static_cast<std::size_t>(plan.total_atoms()), 3u * sizeof(double),
                             position_bytes) ||
      !checked_multiply_size(static_cast<std::size_t>(plan.batch_size()), sizeof(double),
                             energy_bytes) ||
      !make_range(positions, position_bytes, position_range) ||
      !make_range(energies, energy_bytes, energy_range) ||
      !make_range(workspace.workspace_base, plan.workspace_size_bytes(), workspace_range) ||
      !make_range(&plan, sizeof(plan), controls[0]) ||
      !make_range(&workspace, sizeof(workspace), controls[1]) ||
      !make_range(&error, sizeof(error), controls[2]) ||
      (forces != nullptr && !make_range(forces, position_bytes, force_range)) ||
      ranges_overlap(position_range, energy_range) ||
      (forces != nullptr && (ranges_overlap(position_range, force_range) ||
                             ranges_overlap(energy_range, force_range))) ||
      ranges_overlap(position_range, workspace_range) ||
      ranges_overlap(energy_range, workspace_range) ||
      (forces != nullptr && ranges_overlap(force_range, workspace_range)) ||
      plan.overlaps_storage(workspace.workspace_base, plan.workspace_size_bytes()) ||
      plan.overlaps_storage(positions, position_bytes) ||
      plan.overlaps_storage(energies, energy_bytes) ||
      (forces != nullptr && plan.overlaps_storage(forces, position_bytes))) {
    return false;
  }
  for (const AddressRange& control : controls) {
    if (ranges_overlap(workspace_range, control) || ranges_overlap(position_range, control) ||
        ranges_overlap(energy_range, control) ||
        (forces != nullptr && ranges_overlap(force_range, control))) {
      return false;
    }
  }
  return !ranges_overlap(controls[0], controls[1]) && !ranges_overlap(controls[0], controls[2]) &&
         !ranges_overlap(controls[1], controls[2]);
}

xtbloom_status_t validate_numerical_inputs(const HalogenPlan& plan, const double* positions,
                                           const double* energies, const double* forces,
                                           std::string& error) {
  const std::size_t coordinate_count = static_cast<std::size_t>(plan.total_atoms()) * 3u;
  for (std::size_t coordinate = 0; coordinate < coordinate_count; ++coordinate) {
    if (!std::isfinite(positions[coordinate])) {
      error = "GFN1 halogen positions contain NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if (forces != nullptr && !std::isfinite(forces[coordinate])) {
      error = "GFN1 halogen force accumulators contain NaN or infinity";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
  }
  for (std::int64_t batch = 0; batch < plan.batch_size(); ++batch) {
    if (!std::isfinite(energies[batch])) {
      error = "GFN1 halogen energy accumulators contain NaN or infinity";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t distance_between(const double* positions, std::size_t first, std::size_t second,
                                  double difference[3], double& distance, std::string& error) {
  for (std::size_t axis = 0; axis < 3u; ++axis) {
    difference[axis] = positions[second * 3u + axis] - positions[first * 3u + axis];
    if (!std::isfinite(difference[axis])) {
      error = "GFN1 halogen coordinate differences overflow floating-point range";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  distance = std::hypot(difference[0], difference[1], difference[2]);
  if (!std::isfinite(distance)) {
    error = "GFN1 halogen distance arithmetic overflowed";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t prepare_axis_neighbors(const HalogenPlanData& data, const double* positions,
                                        const HalogenWorkspace& workspace, std::string& error) {
  std::fill_n(workspace.axis_neighbors, static_cast<std::size_t>(data.total_atoms), -1);
  for (std::int64_t batch = 0; batch < data.batch_size; ++batch) {
    const std::int64_t begin = data.atom_offsets[static_cast<std::size_t>(batch)];
    const std::int64_t end = data.atom_offsets[static_cast<std::size_t>(batch + 1)];
    for (std::int64_t donor_atom = begin; donor_atom < end; ++donor_atom) {
      const auto donor = static_cast<std::size_t>(donor_atom);
      if (data.donor[donor] == 0u) {
        continue;
      }
      bool active = false;
      for (std::int64_t acceptor_atom = begin; acceptor_atom < end; ++acceptor_atom) {
        const auto acceptor = static_cast<std::size_t>(acceptor_atom);
        if (data.acceptor[acceptor] == 0u) {
          continue;
        }
        double difference[3]{};
        double distance = 0.0;
        const xtbloom_status_t status =
            distance_between(positions, donor, acceptor, difference, distance, error);
        if (status != XTBLOOM_STATUS_SUCCESS) {
          return status;
        }
        if (distance <= kCutoffBohr) {
          if (!(distance > 0.0)) {
            error = "GFN1 halogen donor and acceptor must not coincide";
            return XTBLOOM_STATUS_INVALID_ARGUMENT;
          }
          active = true;
        }
      }
      if (!active) {
        continue;
      }

      std::int64_t neighbor = -1;
      double nearest_distance = std::numeric_limits<double>::infinity();
      for (std::int64_t candidate_atom = begin; candidate_atom < end; ++candidate_atom) {
        double difference[3]{};
        double distance = 0.0;
        const xtbloom_status_t status =
            distance_between(positions, donor, static_cast<std::size_t>(candidate_atom), difference,
                             distance, error);
        if (status != XTBLOOM_STATUS_SUCCESS) {
          return status;
        }
        /* Strict comparison retains the first, therefore lowest-index, tie. */
        if (distance > 0.0 && distance < nearest_distance) {
          neighbor = candidate_atom;
          nearest_distance = distance;
        }
      }
      if (neighbor < 0 || !std::isfinite(nearest_distance)) {
        error = "GFN1 halogen donor has no positive-distance axis neighbor";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      workspace.axis_neighbors[donor] = neighbor;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

struct TripleContribution {
  double energy = 0.0;
  double donor_force[3]{};
  double acceptor_force[3]{};
  double neighbor_force[3]{};
};

xtbloom_status_t evaluate_triple(const HalogenPlanData& data, const double* positions,
                                 std::size_t donor, std::size_t acceptor, std::size_t neighbor,
                                 bool with_forces, TripleContribution& contribution,
                                 std::string& error) {
  /* If the acceptor defines the donor axis, the sixth-power angular factor and
   * its first derivative are exactly zero. Preserve that identity explicitly. */
  if (acceptor == neighbor) {
    return XTBLOOM_STATUS_SUCCESS;
  }

  double donor_acceptor[3]{};
  double donor_neighbor[3]{};
  double acceptor_distance = 0.0;
  double neighbor_distance = 0.0;
  xtbloom_status_t status =
      distance_between(positions, donor, acceptor, donor_acceptor, acceptor_distance, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = distance_between(positions, donor, neighbor, donor_neighbor, neighbor_distance, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (!(acceptor_distance > 0.0) || !(neighbor_distance > 0.0)) {
    error = "GFN1 halogen topology contains a coincident donor, acceptor, or axis neighbor";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  double acceptor_unit[3]{};
  double neighbor_unit[3]{};
  double cosine = 0.0;
  for (std::size_t axis = 0; axis < 3u; ++axis) {
    acceptor_unit[axis] = donor_acceptor[axis] / acceptor_distance;
    neighbor_unit[axis] = donor_neighbor[axis] / neighbor_distance;
    cosine += acceptor_unit[axis] * neighbor_unit[axis];
  }
  const double angular_base = 0.5 * (1.0 - cosine);
  const double angular_base_squared = angular_base * angular_base;
  const double angular = angular_base_squared * angular_base_squared * angular_base_squared;

  const double r0 = data.scaled_atomic_radius[donor] + data.scaled_atomic_radius[acceptor];
  const double ratio = acceptor_distance / r0;
  const double ratio_squared = ratio * ratio;
  const double ratio_sixth = ratio_squared * ratio_squared * ratio_squared;
  const double ratio_twelfth = ratio_sixth * ratio_sixth;
  const double denominator = 1.0 + ratio_twelfth;
  const double radial =
      (1.0 - parameters::gfn1::kGlobal.halogen_damping * ratio_sixth) / denominator;
  const double strength = data.bond_strength[donor];
  contribution.energy = strength * angular * radial;
  if (!std::isfinite(cosine) || !std::isfinite(angular) || !std::isfinite(r0) ||
      !std::isfinite(ratio_sixth) || !std::isfinite(ratio_twelfth) || !std::isfinite(radial) ||
      !std::isfinite(contribution.energy)) {
    error = "GFN1 halogen energy arithmetic exceeded floating-point range";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  if (!with_forces) {
    return XTBLOOM_STATUS_SUCCESS;
  }

  const double radial_distance_derivative =
      6.0 * ratio_sixth / acceptor_distance *
      (parameters::gfn1::kGlobal.halogen_damping * ratio_twelfth - 2.0 * ratio_sixth -
       parameters::gfn1::kGlobal.halogen_damping) /
      (denominator * denominator);
  const double angular_cosine_derivative =
      -3.0 * angular_base_squared * angular_base_squared * angular_base;
  if (!std::isfinite(radial_distance_derivative) || !std::isfinite(angular_cosine_derivative)) {
    error = "GFN1 halogen force arithmetic exceeded floating-point range";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  for (std::size_t axis = 0; axis < 3u; ++axis) {
    const double cosine_acceptor_derivative =
        (neighbor_unit[axis] - cosine * acceptor_unit[axis]) / acceptor_distance;
    const double cosine_neighbor_derivative =
        (acceptor_unit[axis] - cosine * neighbor_unit[axis]) / neighbor_distance;
    const double acceptor_gradient =
        strength * (radial * angular_cosine_derivative * cosine_acceptor_derivative +
                    angular * radial_distance_derivative * acceptor_unit[axis]);
    const double neighbor_gradient =
        strength * radial * angular_cosine_derivative * cosine_neighbor_derivative;
    contribution.acceptor_force[axis] = -acceptor_gradient;
    contribution.neighbor_force[axis] = -neighbor_gradient;
    contribution.donor_force[axis] = acceptor_gradient + neighbor_gradient;
    if (!std::isfinite(contribution.acceptor_force[axis]) ||
        !std::isfinite(contribution.neighbor_force[axis]) ||
        !std::isfinite(contribution.donor_force[axis])) {
      error = "GFN1 halogen force arithmetic exceeded floating-point range";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t evaluate_into_scratch(const HalogenPlanData& data, const double* positions,
                                       bool with_forces, const HalogenWorkspace& workspace,
                                       std::string& error) {
  for (std::int64_t batch = 0; batch < data.batch_size; ++batch) {
    const std::int64_t begin = data.atom_offsets[static_cast<std::size_t>(batch)];
    const std::int64_t end = data.atom_offsets[static_cast<std::size_t>(batch + 1)];
    for (std::int64_t donor_atom = begin; donor_atom < end; ++donor_atom) {
      const auto donor = static_cast<std::size_t>(donor_atom);
      const std::int64_t neighbor_atom = workspace.axis_neighbors[donor];
      if (data.donor[donor] == 0u || neighbor_atom < 0) {
        continue;
      }
      const auto neighbor = static_cast<std::size_t>(neighbor_atom);
      for (std::int64_t acceptor_atom = begin; acceptor_atom < end; ++acceptor_atom) {
        const auto acceptor = static_cast<std::size_t>(acceptor_atom);
        if (data.acceptor[acceptor] == 0u) {
          continue;
        }
        double difference[3]{};
        double distance = 0.0;
        xtbloom_status_t status =
            distance_between(positions, donor, acceptor, difference, distance, error);
        if (status != XTBLOOM_STATUS_SUCCESS) {
          return status;
        }
        if (distance > kCutoffBohr) {
          continue;
        }
        TripleContribution contribution;
        status = evaluate_triple(data, positions, donor, acceptor, neighbor, with_forces,
                                 contribution, error);
        if (status != XTBLOOM_STATUS_SUCCESS) {
          return status;
        }
        const double candidate_energy = workspace.batch_scratch[batch] + contribution.energy;
        if (!std::isfinite(candidate_energy)) {
          error = "GFN1 halogen energy accumulation exceeded floating-point range";
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        workspace.batch_scratch[batch] = candidate_energy;
        if (with_forces) {
          for (std::size_t axis = 0; axis < 3u; ++axis) {
            const auto accumulate_force = [&](std::size_t atom, double increment) {
              double& target = workspace.force_scratch[atom * 3u + axis];
              const double candidate = target + increment;
              if (!std::isfinite(candidate)) {
                return false;
              }
              target = candidate;
              return true;
            };
            if (!accumulate_force(donor, contribution.donor_force[axis]) ||
                !accumulate_force(acceptor, contribution.acceptor_force[axis]) ||
                !accumulate_force(neighbor, contribution.neighbor_force[axis])) {
              error = "GFN1 halogen force accumulation exceeded floating-point range";
              return XTBLOOM_STATUS_INTERNAL_ERROR;
            }
          }
        }
      }
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace

HalogenPlan::HalogenPlan(std::shared_ptr<const HalogenPlanData> data) noexcept
    : data_(std::move(data)) {}

bool HalogenPlan::sealed() const noexcept { return data_ != nullptr; }

std::int64_t HalogenPlan::batch_size() const noexcept { return data_ ? data_->batch_size : 0; }

std::int64_t HalogenPlan::total_atoms() const noexcept { return data_ ? data_->total_atoms : 0; }

const std::vector<std::int64_t>& HalogenPlan::atom_offsets() const noexcept {
  static const std::vector<std::int64_t> empty;
  return data_ ? data_->atom_offsets : empty;
}

bool HalogenPlan::matches_atomic_numbers(const std::int32_t* atomic_numbers) const noexcept {
  return data_ != nullptr && atomic_numbers != nullptr &&
         std::equal(data_->atomic_numbers.begin(), data_->atomic_numbers.end(), atomic_numbers);
}

bool HalogenPlan::overlaps_storage(const void* data, std::size_t size_bytes) const noexcept {
  AddressRange active;
  AddressRange descriptor;
  AddressRange plan_data;
  if (data_ == nullptr || !make_range(data, size_bytes, active) ||
      !make_range(this, sizeof(*this), descriptor) ||
      !make_range(data_.get(), sizeof(*data_), plan_data)) {
    return true;
  }
  if (ranges_overlap(active, descriptor) || ranges_overlap(active, plan_data)) {
    return true;
  }
  return overlaps_vector(active, data_->atom_offsets) ||
         overlaps_vector(active, data_->atomic_numbers) ||
         overlaps_vector(active, data_->scaled_atomic_radius) ||
         overlaps_vector(active, data_->bond_strength) || overlaps_vector(active, data_->donor) ||
         overlaps_vector(active, data_->acceptor);
}

std::size_t HalogenPlan::workspace_size_bytes() const noexcept {
  return data_ ? data_->workspace_size_bytes : 0u;
}

std::size_t HalogenPlan::resident_bytes() const noexcept {
  if (data_ == nullptr) {
    return 0u;
  }
  return sizeof(*data_) + data_->atom_offsets.capacity() * sizeof(std::int64_t) +
         data_->atomic_numbers.capacity() * sizeof(std::int32_t) +
         data_->scaled_atomic_radius.capacity() * sizeof(double) +
         data_->bond_strength.capacity() * sizeof(double) +
         data_->donor.capacity() * sizeof(std::uint8_t) +
         data_->acceptor.capacity() * sizeof(std::uint8_t);
}

const HalogenPlanData* HalogenPlan::identity() const noexcept { return data_.get(); }

xtbloom_status_t make_halogen_plan(std::int64_t batch_size, std::int64_t total_atoms,
                                   const std::int64_t* atom_offsets,
                                   const std::int32_t* atomic_numbers, HalogenPlan& plan,
                                   std::string& error) {
  if (!valid_batch_count(batch_size) || !valid_atom_count(total_atoms) || atom_offsets == nullptr ||
      atomic_numbers == nullptr || atom_offsets[0] != 0 ||
      atom_offsets[batch_size] != total_atoms) {
    error = "GFN1 halogen plan requires a valid positive ragged batch";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  try {
    auto created = std::make_shared<HalogenPlanData>();
    created->batch_size = batch_size;
    created->total_atoms = total_atoms;
    created->atom_offsets.assign(atom_offsets, atom_offsets + batch_size + 1);
    created->atomic_numbers.assign(atomic_numbers, atomic_numbers + total_atoms);
    const auto atom_count = static_cast<std::size_t>(total_atoms);
    created->scaled_atomic_radius.resize(atom_count);
    created->bond_strength.resize(atom_count);
    created->donor.resize(atom_count);
    created->acceptor.resize(atom_count);
    for (std::int64_t batch = 0; batch < batch_size; ++batch) {
      if (atom_offsets[batch] < 0 || atom_offsets[batch] > atom_offsets[batch + 1] ||
          atom_offsets[batch + 1] > total_atoms) {
        error = "GFN1 halogen atom offsets are not a valid ragged partition";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
    for (std::size_t atom = 0; atom < atom_count; ++atom) {
      const std::int32_t atomic_number = created->atomic_numbers[atom];
      const auto* element =
          parameters::gfn1::find_element(static_cast<std::uint32_t>(atomic_number));
      if (element == nullptr || element->atomic_number != atomic_number ||
          !(element->atomic_radius_bohr > 0.0) || !std::isfinite(element->atomic_radius_bohr) ||
          element->xbond < 0.0 || !std::isfinite(element->xbond)) {
        error = "GFN1 halogen plan contains an unsupported atomic number or invalid parameter";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      created->scaled_atomic_radius[atom] =
          parameters::gfn1::kGlobal.halogen_radius_scale * element->atomic_radius_bohr;
      created->bond_strength[atom] = element->xbond;
      created->donor[atom] = static_cast<std::uint8_t>(is_donor(atomic_number));
      created->acceptor[atom] = static_cast<std::uint8_t>(is_acceptor(atomic_number));
    }

    std::size_t axis_bytes = 0u;
    std::size_t batch_bytes = 0u;
    std::size_t force_elements = 0u;
    std::size_t force_bytes = 0u;
    std::size_t cursor = 0u;
    if (!checked_multiply_size(atom_count, sizeof(std::int64_t), axis_bytes) ||
        !checked_multiply_size(static_cast<std::size_t>(batch_size), sizeof(double), batch_bytes) ||
        !checked_multiply_size(atom_count, 3u, force_elements) ||
        !checked_multiply_size(force_elements, sizeof(double), force_bytes) ||
        !append_segment(axis_bytes, alignof(std::int64_t), cursor, created->axis_neighbor_offset) ||
        !append_segment(batch_bytes, alignof(double), cursor, created->batch_scratch_offset) ||
        !append_segment(force_bytes, alignof(double), cursor, created->force_scratch_offset) ||
        !align_up(cursor, kHalogenWorkspaceAlignment, created->workspace_size_bytes)) {
      error = "GFN1 halogen workspace byte count overflows";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    plan = HalogenPlan(std::move(created));
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN1 halogen plan";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  } catch (const std::length_error&) {
    error = "GFN1 halogen plan dimensions exceed host container limits";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

xtbloom_status_t bind_halogen_workspace(const HalogenPlan& plan, void* workspace,
                                        std::size_t workspace_size, HalogenWorkspace& view,
                                        std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  AddressRange workspace_range;
  AddressRange view_range;
  AddressRange error_range;
  if (!aligned(workspace, kHalogenWorkspaceAlignment) ||
      workspace_size < plan.workspace_size_bytes() ||
      !make_range(workspace, plan.workspace_size_bytes(), workspace_range) ||
      !make_range(&view, sizeof(view), view_range) ||
      !make_range(&error, sizeof(error), error_range) ||
      ranges_overlap(workspace_range, view_range) || ranges_overlap(workspace_range, error_range) ||
      ranges_overlap(view_range, error_range) ||
      plan.overlaps_storage(workspace, plan.workspace_size_bytes())) {
    error = "GFN1 halogen workspace must be sufficiently large and 64-byte aligned";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const HalogenPlanData& data = *plan.identity();
  HalogenWorkspace bound;
  bound.workspace_base = workspace;
  bound.workspace_size_bytes = workspace_size;
  bound.axis_neighbors = offset_pointer<std::int64_t>(workspace, data.axis_neighbor_offset);
  bound.axis_neighbor_elements = data.total_atoms;
  bound.batch_scratch = offset_pointer<double>(workspace, data.batch_scratch_offset);
  bound.batch_elements = data.batch_size;
  bound.force_scratch = offset_pointer<double>(workspace, data.force_scratch_offset);
  bound.force_elements = data.total_atoms * 3;
  bound.plan_identity = plan.identity();
  view = bound;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t add_halogen_cpu(const HalogenPlan& plan, const double* positions, double* energies,
                                 double* forces, const HalogenWorkspace& workspace,
                                 std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = validate_workspace(plan, workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (!aligned(positions, alignof(double)) || !aligned(energies, alignof(double)) ||
      (forces != nullptr && !aligned(forces, alignof(double)))) {
    error = "GFN1 halogen positions and outputs must not be NULL or misaligned";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!valid_call_storage(plan, workspace, positions, energies, forces, error)) {
    error = "GFN1 halogen buffers overlap inputs, outputs, plan, workspace, or descriptors";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  status = validate_numerical_inputs(plan, positions, energies, forces, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  const HalogenPlanData& data = *plan.identity();
  status = prepare_axis_neighbors(data, positions, workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }

  const std::size_t batch_count = static_cast<std::size_t>(data.batch_size);
  const std::size_t force_count = static_cast<std::size_t>(data.total_atoms) * 3u;
  std::memcpy(workspace.batch_scratch, energies, batch_count * sizeof(double));
  if (forces != nullptr) {
    std::memcpy(workspace.force_scratch, forces, force_count * sizeof(double));
  }
  status = evaluate_into_scratch(data, positions, forces != nullptr, workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  std::memcpy(energies, workspace.batch_scratch, batch_count * sizeof(double));
  if (forces != nullptr) {
    std::memcpy(forces, workspace.force_scratch, force_count * sizeof(double));
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn1
