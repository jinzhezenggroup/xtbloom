// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn1/coordination.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
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

template <typename PairFunction>
void for_each_active_pair(const CoordinationPlan& plan, const double* positions,
                          PairFunction&& function) {
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
        if (distance_squared < minimum_squared) {
          /* Preserve the reference's diagonal/self-image exclusion exactly. */
          continue;
        }
        if (distance_squared > cutoff_squared) {
          continue;
        }
        function(first_index, second_index, dx, dy, dz, distance_squared);
      }
    }
  }
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
  }
}

xtbloom_status_t evaluate_coordination_cpu(const CoordinationPlan& plan, const double* positions,
                                           double* coordination_numbers, std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (coordination_numbers == nullptr) {
    error = "GFN1 coordination output must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  status = validate_positions(plan, positions, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  std::fill_n(coordination_numbers, static_cast<std::size_t>(plan.total_atoms), 0.0);
  for_each_active_pair(
      plan, positions,
      [&](std::size_t first, std::size_t second, double, double, double, double distance_squared) {
        const double radius = plan.covalent_radius[first] + plan.covalent_radius[second];
        const double count = exponential_count(std::sqrt(distance_squared), radius).value;
        coordination_numbers[first] += count;
        coordination_numbers[second] += count;
      });
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
  if (dE_dcn == nullptr || gradients == nullptr) {
    error = "GFN1 coordination derivative inputs and gradients must not be NULL";
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
  for_each_active_pair(
      plan, positions,
      [&](std::size_t first, std::size_t second, double dx, double dy, double dz,
          double distance_squared) {
        const double distance = std::sqrt(distance_squared);
        const double radius = plan.covalent_radius[first] + plan.covalent_radius[second];
        const double derivative = exponential_count(distance, radius).distance_derivative;
        const double scale = (dE_dcn[first] + dE_dcn[second]) * derivative / distance;
        const double gx = scale * dx;
        const double gy = scale * dy;
        const double gz = scale * dz;
        gradients[first * 3u] += gx;
        gradients[first * 3u + 1u] += gy;
        gradients[first * 3u + 2u] += gz;
        gradients[second * 3u] -= gx;
        gradients[second * 3u + 1u] -= gy;
        gradients[second * 3u + 2u] -= gz;
      });
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn1
