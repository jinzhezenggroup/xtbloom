// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn1/coordination.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

#include "data/parameters/gfn1.hpp"

namespace xtbloom::detail::gfn1 {
namespace {

static_assert(parameters::gfn1::kGlobal.coordination_number_model == 1u,
              "the generated GFN1 parameters must select the exp CN model");
static_assert(!parameters::gfn1::kGlobal.coordination_has_maximum_cn_cutoff,
              "GFN1 exponential CN must not apply a maximum-CN cutoff");
static_assert(parameters::gfn1::kGlobal.coordination_directed_factor == 1.0,
              "the nonperiodic GFN1 pair loop assumes unit directed weight");
static_assert(parameters::gfn1::kGlobal.coordination_cutoff_inclusive,
              "GFN1 coordination includes pairs exactly at the cutoff");
static_assert(parameters::gfn1::kGlobal.coordination_coincident_cutoff_inclusive,
              "GFN1 coordination includes the coincident threshold itself");

bool representable_as_size(std::int64_t value) {
  return value >= 0 && static_cast<std::uint64_t>(value) <=
                           static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
}

bool representable_geometry_size(std::int64_t atom_count) {
  if (!representable_as_size(atom_count)) {
    return false;
  }
  const auto count = static_cast<std::uint64_t>(atom_count);
  return count <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / 3u) &&
         count <= static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / 3u;
}

xtbloom_status_t validate_plan(const CoordinationPlan& plan, std::string& error) {
  if (plan.batch_size <= 0 || plan.total_atoms <= 0 || !representable_as_size(plan.batch_size) ||
      !representable_geometry_size(plan.total_atoms) ||
      static_cast<std::uint64_t>(plan.batch_size) >=
          static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    error = "GFN1 coordination plan has invalid batch or atom counts";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  const auto atom_count = static_cast<std::size_t>(plan.total_atoms);
  if (plan.atom_offsets.size() != static_cast<std::size_t>(plan.batch_size + 1) ||
      plan.covalent_radius.size() != atom_count || plan.atom_offsets.front() != 0 ||
      plan.atom_offsets.back() != plan.total_atoms) {
    error = "GFN1 coordination plan is incomplete or internally inconsistent";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t batch = 0; batch < plan.batch_size; ++batch) {
    const std::int64_t begin = plan.atom_offsets[static_cast<std::size_t>(batch)];
    const std::int64_t end = plan.atom_offsets[static_cast<std::size_t>(batch + 1)];
    if (begin < 0 || begin > end || end > plan.total_atoms) {
      error = "GFN1 coordination plan offsets are not a valid ragged partition";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (double radius : plan.covalent_radius) {
    if (!(radius > 0.0) || !std::isfinite(radius)) {
      error = "GFN1 coordination plan contains an invalid covalent radius";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_positions(const CoordinationPlan& plan, const double* positions,
                                    std::string& error) {
  if (positions == nullptr) {
    error = "GFN1 coordination positions must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const auto coordinate_count = static_cast<std::size_t>(plan.total_atoms) * 3u;
  for (std::size_t coordinate = 0; coordinate < coordinate_count; ++coordinate) {
    if (!std::isfinite(positions[coordinate])) {
      error = "GFN1 coordination positions contain NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

bool aligned_double(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(double) == 0u;
}

bool count_bytes(std::size_t count, std::size_t element_size, std::size_t& bytes) {
  if (count > std::numeric_limits<std::size_t>::max() / element_size) {
    return false;
  }
  bytes = count * element_size;
  return true;
}

bool ranges_overlap(const void* first, std::size_t first_bytes, const void* second,
                    std::size_t second_bytes) {
  if (first_bytes == 0u || second_bytes == 0u) {
    return false;
  }
  const std::uintptr_t first_begin = reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t second_begin = reinterpret_cast<std::uintptr_t>(second);
  if (first_begin > std::numeric_limits<std::uintptr_t>::max() - first_bytes ||
      second_begin > std::numeric_limits<std::uintptr_t>::max() - second_bytes) {
    return true;
  }
  return first_begin < second_begin + second_bytes && second_begin < first_begin + first_bytes;
}

template <typename T>
bool overlaps_vector(const void* pointer, std::size_t bytes, const std::vector<T>& values) {
  std::size_t value_bytes = 0u;
  return !count_bytes(values.size(), sizeof(T), value_bytes) ||
         ranges_overlap(pointer, bytes, values.data(), value_bytes);
}

bool overlaps_plan_or_error(const void* pointer, std::size_t bytes, const CoordinationPlan& plan,
                            const std::string& error) {
  return ranges_overlap(pointer, bytes, &plan, sizeof(plan)) ||
         ranges_overlap(pointer, bytes, &error, sizeof(error)) ||
         overlaps_vector(pointer, bytes, plan.atom_offsets) ||
         overlaps_vector(pointer, bytes, plan.covalent_radius);
}

template <typename PairFunction>
xtbloom_status_t for_each_active_pair(const CoordinationPlan& plan, const double* positions,
                                      PairFunction&& function, std::string& error) {
  const double cutoff = parameters::gfn1::kGlobal.coordination_cutoff_bohr;
  const double cutoff_squared = cutoff * cutoff;
  const double minimum_squared =
      parameters::gfn1::kGlobal.coordination_coincident_distance_squared_cutoff_bohr2;

  for (std::int64_t batch = 0; batch < plan.batch_size; ++batch) {
    const std::int64_t begin = plan.atom_offsets[static_cast<std::size_t>(batch)];
    const std::int64_t end = plan.atom_offsets[static_cast<std::size_t>(batch + 1)];
    for (std::int64_t first = begin; first < end; ++first) {
      const std::size_t first_index = static_cast<std::size_t>(first);
      for (std::int64_t second = begin; second < first; ++second) {
        const std::size_t second_index = static_cast<std::size_t>(second);
        const double dx = positions[first_index * 3u] - positions[second_index * 3u];
        const double dy = positions[first_index * 3u + 1u] - positions[second_index * 3u + 1u];
        const double dz = positions[first_index * 3u + 2u] - positions[second_index * 3u + 2u];
        const double distance_squared = dx * dx + dy * dy + dz * dz;
        if (!std::isfinite(distance_squared)) {
          error = "GFN1 coordination coordinate differences overflow floating-point range";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        if (distance_squared < minimum_squared) {
          /* Preserve the reference's diagonal/self-image exclusion exactly. */
          continue;
        }
        if (distance_squared > cutoff_squared) {
          continue;
        }
        const xtbloom_status_t status =
            function(first_index, second_index, dx, dy, dz, distance_squared);
        if (status != XTBLOOM_STATUS_SUCCESS) {
          return status;
        }
      }
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

template <typename PairFunction>
xtbloom_status_t for_each_active_pair_of_atom(const CoordinationPlan& plan, const double* positions,
                                              std::size_t atom, PairFunction&& function,
                                              std::string& error) {
  const double cutoff = parameters::gfn1::kGlobal.coordination_cutoff_bohr;
  const double cutoff_squared = cutoff * cutoff;
  const double minimum_squared =
      parameters::gfn1::kGlobal.coordination_coincident_distance_squared_cutoff_bohr2;
  const auto upper = std::upper_bound(plan.atom_offsets.begin(), plan.atom_offsets.end(),
                                      static_cast<std::int64_t>(atom));
  const std::size_t batch = static_cast<std::size_t>(upper - plan.atom_offsets.begin() - 1);
  const std::int64_t begin = plan.atom_offsets[batch];
  const std::int64_t end = plan.atom_offsets[batch + 1u];
  for (std::int64_t other = begin; other < end; ++other) {
    const std::size_t other_index = static_cast<std::size_t>(other);
    if (other_index == atom) {
      continue;
    }
    const double dx = positions[atom * 3u] - positions[other_index * 3u];
    const double dy = positions[atom * 3u + 1u] - positions[other_index * 3u + 1u];
    const double dz = positions[atom * 3u + 2u] - positions[other_index * 3u + 2u];
    const double distance_squared = dx * dx + dy * dy + dz * dz;
    if (!std::isfinite(distance_squared)) {
      error = "GFN1 coordination coordinate differences overflow floating-point range";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if (distance_squared < minimum_squared || distance_squared > cutoff_squared) {
      continue;
    }
    const xtbloom_status_t status = function(other_index, dx, dy, dz, distance_squared);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

struct PairCount {
  double value;
  double distance_derivative;
};

PairCount exponential_count(double distance, double radius) {
  const double steepness = parameters::gfn1::kGlobal.coordination_steepness;
  const double inverse_distance = 1.0 / distance;
  const double argument = steepness * (radius * inverse_distance - 1.0);

  /*
   * Evaluate the logistic and its derivative from a nonpositive exponential.
   * This avoids overflow and retains a small derivative after the value has
   * rounded to one on the strongly bonded side of the switching function.
   */
  double value = 0.0;
  double logistic_derivative = 0.0;
  if (argument >= 0.0) {
    const double exponential = std::exp(-argument);
    const double denominator = 1.0 + exponential;
    value = 1.0 / denominator;
    logistic_derivative = exponential / (denominator * denominator);
  } else {
    const double exponential = std::exp(argument);
    const double denominator = 1.0 + exponential;
    value = exponential / denominator;
    logistic_derivative = exponential / (denominator * denominator);
  }

  return {value, -steepness * radius * inverse_distance * inverse_distance * logistic_derivative};
}

xtbloom_status_t pair_count(const CoordinationPlan& plan, std::size_t first, std::size_t second,
                            double distance_squared, double& count, std::string& error) {
  const double radius = plan.covalent_radius[first] + plan.covalent_radius[second];
  count = exponential_count(std::sqrt(distance_squared), radius).value;
  if (!std::isfinite(radius) || !std::isfinite(count)) {
    error = "GFN1 coordination arithmetic exceeded floating-point range";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t pair_gradient_increment(const CoordinationPlan& plan, const double* dE_dcn,
                                         std::size_t first, std::size_t second, double dx,
                                         double dy, double dz, double distance_squared,
                                         double increments[3], std::string& error) {
  const double distance = std::sqrt(distance_squared);
  const double radius = plan.covalent_radius[first] + plan.covalent_radius[second];
  const double derivative = exponential_count(distance, radius).distance_derivative;
  const double derivative_sum = dE_dcn[first] + dE_dcn[second];
  const double scale = derivative_sum * derivative / distance;
  increments[0] = scale * dx;
  increments[1] = scale * dy;
  increments[2] = scale * dz;
  if (!std::isfinite(radius) || !std::isfinite(derivative) || !std::isfinite(derivative_sum) ||
      !std::isfinite(scale) || !std::isfinite(increments[0]) || !std::isfinite(increments[1]) ||
      !std::isfinite(increments[2])) {
    error = "GFN1 coordination gradient arithmetic exceeded floating-point range";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace

xtbloom_status_t make_coordination_plan(std::int64_t batch_size, std::int64_t total_atoms,
                                        const std::int64_t* atom_offsets,
                                        const std::int32_t* atomic_numbers, CoordinationPlan& plan,
                                        std::string& error) {
  if (batch_size <= 0 || total_atoms <= 0 || !representable_as_size(batch_size) ||
      !representable_geometry_size(total_atoms) ||
      static_cast<std::uint64_t>(batch_size) >=
          static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    error = "GFN1 coordination plan requires positive, representable batch and atom counts";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (atom_offsets == nullptr || atomic_numbers == nullptr) {
    error = "GFN1 coordination plan offsets and atomic numbers must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (atom_offsets[0] != 0 || atom_offsets[batch_size] != total_atoms) {
    error = "GFN1 coordination plan offsets must start at zero and end at total_atoms";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    if (atom_offsets[batch] < 0 || atom_offsets[batch] > atom_offsets[batch + 1] ||
        atom_offsets[batch + 1] > total_atoms) {
      error = "GFN1 coordination plan offsets must be a monotone ragged partition";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  try {
    CoordinationPlan created;
    created.batch_size = batch_size;
    created.total_atoms = total_atoms;
    created.atom_offsets.assign(atom_offsets, atom_offsets + batch_size + 1);
    created.covalent_radius.resize(static_cast<std::size_t>(total_atoms));

    for (std::int64_t atom = 0; atom < total_atoms; ++atom) {
      const std::int32_t atomic_number = atomic_numbers[atom];
      const auto* element =
          parameters::gfn1::find_element(static_cast<std::uint32_t>(atomic_number));
      if (element == nullptr || element->atomic_number != atomic_number ||
          !(element->covalent_radius_bohr > 0.0) || !std::isfinite(element->covalent_radius_bohr)) {
        error = "GFN1 coordination plan contains an unsupported atomic number";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      created.covalent_radius[static_cast<std::size_t>(atom)] = element->covalent_radius_bohr;
    }

    plan = std::move(created);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN1 coordination plan";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  } catch (const std::length_error&) {
    error = "GFN1 coordination plan dimensions exceed host container limits";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

xtbloom_status_t evaluate_coordination_cpu(const CoordinationPlan& plan, const double* positions,
                                           double* coordination_numbers, std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (positions == nullptr || coordination_numbers == nullptr || !aligned_double(positions) ||
      !aligned_double(coordination_numbers)) {
    error = "GFN1 coordination positions and output must not be NULL or misaligned";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t atom_count = static_cast<std::size_t>(plan.total_atoms);
  std::size_t position_bytes = 0u;
  std::size_t output_bytes = 0u;
  if (!count_bytes(atom_count, 3u * sizeof(double), position_bytes) ||
      !count_bytes(atom_count, sizeof(double), output_bytes) ||
      ranges_overlap(positions, position_bytes, coordination_numbers, output_bytes) ||
      overlaps_plan_or_error(coordination_numbers, output_bytes, plan, error)) {
    error = "GFN1 coordination output must be disjoint from inputs and control storage";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  status = validate_positions(plan, positions, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }

  /* Simulate every atom's exact accumulation order before publishing output. */
  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    double candidate = 0.0;
    status = for_each_active_pair_of_atom(
        plan, positions, atom,
        [&](std::size_t other, double, double, double,
            double distance_squared) -> xtbloom_status_t {
          double count = 0.0;
          const xtbloom_status_t pair_status =
              pair_count(plan, atom, other, distance_squared, count, error);
          if (pair_status != XTBLOOM_STATUS_SUCCESS) {
            return pair_status;
          }
          candidate += count;
          if (!std::isfinite(candidate)) {
            error = "GFN1 coordination accumulation exceeded floating-point range";
            return XTBLOOM_STATUS_INTERNAL_ERROR;
          }
          return XTBLOOM_STATUS_SUCCESS;
        },
        error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
  }

  std::fill_n(coordination_numbers, atom_count, 0.0);
  status = for_each_active_pair(
      plan, positions,
      [&](std::size_t first, std::size_t second, double, double, double, double distance_squared) {
        double count = 0.0;
        const xtbloom_status_t pair_status =
            pair_count(plan, first, second, distance_squared, count, error);
        if (pair_status == XTBLOOM_STATUS_SUCCESS) {
          coordination_numbers[first] += count;
          coordination_numbers[second] += count;
        }
        return pair_status;
      },
      error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t add_coordination_gradient_cpu(const CoordinationPlan& plan,
                                               const double* positions, const double* dE_dcn,
                                               double* gradients, std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (positions == nullptr || dE_dcn == nullptr || gradients == nullptr ||
      !aligned_double(positions) || !aligned_double(dE_dcn) || !aligned_double(gradients)) {
    error = "GFN1 coordination derivative inputs and gradients must not be NULL or misaligned";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t atom_count = static_cast<std::size_t>(plan.total_atoms);
  std::size_t position_bytes = 0u;
  std::size_t atom_bytes = 0u;
  if (!count_bytes(atom_count, 3u * sizeof(double), position_bytes) ||
      !count_bytes(atom_count, sizeof(double), atom_bytes) ||
      ranges_overlap(positions, position_bytes, gradients, position_bytes) ||
      ranges_overlap(dE_dcn, atom_bytes, gradients, position_bytes) ||
      overlaps_plan_or_error(gradients, position_bytes, plan, error)) {
    error = "GFN1 coordination gradients must be disjoint from inputs and control storage";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  status = validate_positions(plan, positions, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  for (std::int64_t atom = 0; atom < plan.total_atoms; ++atom) {
    if (!std::isfinite(dE_dcn[static_cast<std::size_t>(atom)])) {
      error = "GFN1 coordination derivatives contain NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::size_t coordinate = 0; coordinate < atom_count * 3u; ++coordinate) {
    if (!std::isfinite(gradients[coordinate])) {
      error = "GFN1 coordination gradient accumulators contain NaN or infinity";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const std::size_t atom = coordinate / 3u;
    const std::size_t axis = coordinate % 3u;
    double candidate = gradients[coordinate];
    status = for_each_active_pair_of_atom(
        plan, positions, atom,
        [&](std::size_t other, double dx, double dy, double dz,
            double distance_squared) -> xtbloom_status_t {
          double increments[3]{};
          const xtbloom_status_t pair_status = pair_gradient_increment(
              plan, dE_dcn, atom, other, dx, dy, dz, distance_squared, increments, error);
          if (pair_status != XTBLOOM_STATUS_SUCCESS) {
            return pair_status;
          }
          candidate += increments[axis];
          if (!std::isfinite(candidate)) {
            error = "GFN1 coordination gradient accumulation exceeded floating-point range";
            return XTBLOOM_STATUS_INTERNAL_ERROR;
          }
          return XTBLOOM_STATUS_SUCCESS;
        },
        error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
  }

  status = for_each_active_pair(
      plan, positions,
      [&](std::size_t first, std::size_t second, double dx, double dy, double dz,
          double distance_squared) {
        double increments[3]{};
        const xtbloom_status_t pair_status = pair_gradient_increment(
            plan, dE_dcn, first, second, dx, dy, dz, distance_squared, increments, error);
        if (pair_status == XTBLOOM_STATUS_SUCCESS) {
          for (std::size_t axis = 0; axis < 3u; ++axis) {
            gradients[first * 3u + axis] += increments[axis];
            gradients[second * 3u + axis] -= increments[axis];
          }
        }
        return pair_status;
      },
      error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn1
