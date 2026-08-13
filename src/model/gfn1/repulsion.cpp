// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn1/repulsion.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <utility>

#include "data/parameters/gfn1.hpp"

namespace xtbloom::detail::gfn1 {
namespace {

constexpr double kCutoffBohr = 25.0;
constexpr double kMinimumDistanceSquared = 1.0e-12;
constexpr double kDistanceDenominatorExponent = 1.0;

static_assert(parameters::gfn1::kGlobal.repulsion_kexp == 1.5,
              "GFN1 repulsion requires the r^1.5 exponential kernel");
static_assert(parameters::gfn1::kGlobal.repulsion_klight == 1.5,
              "GFN1 H/He pairs use the same exponent as heavy-element pairs");

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

xtbloom_status_t validate_plan(const RepulsionPlan& plan, std::string& error) {
  if (plan.batch_size <= 0 || plan.total_atoms <= 0 || !representable_as_size(plan.batch_size) ||
      !representable_geometry_size(plan.total_atoms) ||
      static_cast<std::uint64_t>(plan.batch_size) >=
          static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    error = "GFN1 repulsion plan has invalid batch or atom counts";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  const auto atom_count = static_cast<std::size_t>(plan.total_atoms);
  if (plan.atom_offsets.size() != static_cast<std::size_t>(plan.batch_size + 1) ||
      plan.sqrt_alpha.size() != atom_count || plan.effective_charge.size() != atom_count ||
      plan.atom_offsets.front() != 0 || plan.atom_offsets.back() != plan.total_atoms) {
    error = "GFN1 repulsion plan is incomplete or internally inconsistent";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t batch = 0; batch < plan.batch_size; ++batch) {
    const std::int64_t begin = plan.atom_offsets[static_cast<std::size_t>(batch)];
    const std::int64_t end = plan.atom_offsets[static_cast<std::size_t>(batch + 1)];
    if (begin < 0 || begin > end || end > plan.total_atoms) {
      error = "GFN1 repulsion plan offsets are not a valid ragged partition";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    if (!(plan.sqrt_alpha[atom] > 0.0) || !std::isfinite(plan.sqrt_alpha[atom]) ||
        !(plan.effective_charge[atom] > 0.0) || !std::isfinite(plan.effective_charge[atom])) {
      error = "GFN1 repulsion plan contains an invalid atomic parameter";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_positions(const RepulsionPlan& plan, const double* positions,
                                    std::string& error) {
  if (positions == nullptr) {
    error = "GFN1 repulsion positions must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const auto coordinate_count = static_cast<std::size_t>(plan.total_atoms) * 3u;
  for (std::size_t coordinate = 0; coordinate < coordinate_count; ++coordinate) {
    if (!std::isfinite(positions[coordinate])) {
      error = "GFN1 repulsion positions contain NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

template <typename PairFunction>
void for_each_active_pair(const RepulsionPlan& plan, const double* positions,
                          PairFunction&& function) {
  constexpr double cutoff_squared = kCutoffBohr * kCutoffBohr;
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
        if (distance_squared < kMinimumDistanceSquared) {
          /* Preserve tblite's diagonal/self-image exclusion exactly. */
          continue;
        }
        if (distance_squared > cutoff_squared) {
          continue;
        }
        function(static_cast<std::size_t>(batch), first_index, second_index, dx, dy, dz,
                 distance_squared);
      }
    }
  }
}

}  // namespace

xtbloom_status_t make_repulsion_plan(std::int64_t batch_size, std::int64_t total_atoms,
                                     const std::int64_t* atom_offsets,
                                     const std::int32_t* atomic_numbers, RepulsionPlan& plan,
                                     std::string& error) {
  if (batch_size <= 0 || total_atoms <= 0 || !representable_as_size(batch_size) ||
      !representable_geometry_size(total_atoms) ||
      static_cast<std::uint64_t>(batch_size) >=
          static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    error = "GFN1 repulsion plan requires positive, representable batch and atom counts";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (atom_offsets == nullptr || atomic_numbers == nullptr) {
    error = "GFN1 repulsion plan offsets and atomic numbers must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (atom_offsets[0] != 0 || atom_offsets[batch_size] != total_atoms) {
    error = "GFN1 repulsion plan offsets must start at zero and end at total_atoms";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    if (atom_offsets[batch] < 0 || atom_offsets[batch] > atom_offsets[batch + 1] ||
        atom_offsets[batch + 1] > total_atoms) {
      error = "GFN1 repulsion plan offsets must be a monotone ragged partition";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  try {
    RepulsionPlan created;
    created.batch_size = batch_size;
    created.total_atoms = total_atoms;
    created.atom_offsets.assign(atom_offsets, atom_offsets + batch_size + 1);
    created.sqrt_alpha.resize(static_cast<std::size_t>(total_atoms));
    created.effective_charge.resize(static_cast<std::size_t>(total_atoms));

    for (std::int64_t atom = 0; atom < total_atoms; ++atom) {
      const std::int32_t atomic_number = atomic_numbers[atom];
      const auto* element =
          parameters::gfn1::find_element(static_cast<std::uint32_t>(atomic_number));
      if (element == nullptr || element->atomic_number != atomic_number || !(element->arep > 0.0) ||
          !std::isfinite(element->arep) || !(element->zeff > 0.0) ||
          !std::isfinite(element->zeff)) {
        error = "GFN1 repulsion plan contains an unsupported atomic number or invalid parameter";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      const auto index = static_cast<std::size_t>(atom);
      created.sqrt_alpha[index] = std::sqrt(element->arep);
      created.effective_charge[index] = element->zeff;
    }

    plan = std::move(created);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN1 repulsion plan";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

xtbloom_status_t add_repulsion_cpu(const RepulsionPlan& plan, const double* positions,
                                   double* energies, double* forces, std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (energies == nullptr) {
    error = "GFN1 repulsion energies must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  status = validate_positions(plan, positions, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  const double exponent = parameters::gfn1::kGlobal.repulsion_kexp;
  for_each_active_pair(
      plan, positions,
      [&](std::size_t batch, std::size_t first, std::size_t second, double dx, double dy, double dz,
          double distance_squared) {
        const double distance = std::sqrt(distance_squared);
        const double distance_power = distance * std::sqrt(distance);
        const double pair_alpha = plan.sqrt_alpha[first] * plan.sqrt_alpha[second];
        const double pair_charge = plan.effective_charge[first] * plan.effective_charge[second];
        const double pair_energy = pair_charge * std::exp(-pair_alpha * distance_power) / distance;
        energies[batch] += pair_energy;

        if (forces != nullptr) {
          const double force_scale =
              (pair_alpha * exponent * distance_power + kDistanceDenominatorExponent) *
              pair_energy / distance_squared;
          const double fx = force_scale * dx;
          const double fy = force_scale * dy;
          const double fz = force_scale * dz;
          forces[first * 3u] += fx;
          forces[first * 3u + 1u] += fy;
          forces[first * 3u + 2u] += fz;
          forces[second * 3u] -= fx;
          forces[second * 3u + 1u] -= fy;
          forces[second * 3u + 2u] -= fz;
        }
      });
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn1
