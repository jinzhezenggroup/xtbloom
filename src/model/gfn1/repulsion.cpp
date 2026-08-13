// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn1/repulsion.hpp"

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

bool overlaps_plan_or_error(const void* pointer, std::size_t bytes, const RepulsionPlan& plan,
                            const std::string& error) {
  return ranges_overlap(pointer, bytes, &plan, sizeof(plan)) ||
         ranges_overlap(pointer, bytes, &error, sizeof(error)) ||
         overlaps_vector(pointer, bytes, plan.atom_offsets) ||
         overlaps_vector(pointer, bytes, plan.sqrt_alpha) ||
         overlaps_vector(pointer, bytes, plan.effective_charge);
}

template <typename PairFunction>
xtbloom_status_t for_each_active_pair(const RepulsionPlan& plan, const double* positions,
                                      PairFunction&& function, std::string& error) {
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
        if (!std::isfinite(distance_squared)) {
          error = "GFN1 repulsion coordinate differences overflow floating-point range";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        if (distance_squared < kMinimumDistanceSquared) {
          /* Preserve tblite's diagonal/self-image exclusion exactly. */
          continue;
        }
        if (distance_squared > cutoff_squared) {
          continue;
        }
        const xtbloom_status_t status = function(static_cast<std::size_t>(batch), first_index,
                                                 second_index, dx, dy, dz, distance_squared);
        if (status != XTBLOOM_STATUS_SUCCESS) {
          return status;
        }
      }
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

template <typename PairFunction>
xtbloom_status_t for_each_active_pair_in_batch(const RepulsionPlan& plan, const double* positions,
                                               std::size_t batch, PairFunction&& function,
                                               std::string& error) {
  constexpr double cutoff_squared = kCutoffBohr * kCutoffBohr;
  const std::int64_t begin = plan.atom_offsets[batch];
  const std::int64_t end = plan.atom_offsets[batch + 1u];
  for (std::int64_t first = begin; first < end; ++first) {
    const std::size_t first_index = static_cast<std::size_t>(first);
    for (std::int64_t second = begin; second < first; ++second) {
      const std::size_t second_index = static_cast<std::size_t>(second);
      const double dx = positions[first_index * 3u] - positions[second_index * 3u];
      const double dy = positions[first_index * 3u + 1u] - positions[second_index * 3u + 1u];
      const double dz = positions[first_index * 3u + 2u] - positions[second_index * 3u + 2u];
      const double distance_squared = dx * dx + dy * dy + dz * dz;
      if (!std::isfinite(distance_squared)) {
        error = "GFN1 repulsion coordinate differences overflow floating-point range";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      if (distance_squared < kMinimumDistanceSquared || distance_squared > cutoff_squared) {
        continue;
      }
      const xtbloom_status_t status =
          function(first_index, second_index, dx, dy, dz, distance_squared);
      if (status != XTBLOOM_STATUS_SUCCESS) {
        return status;
      }
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

template <typename PairFunction>
xtbloom_status_t for_each_active_pair_of_atom(const RepulsionPlan& plan, const double* positions,
                                              std::size_t atom, PairFunction&& function,
                                              std::string& error) {
  constexpr double cutoff_squared = kCutoffBohr * kCutoffBohr;
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
      error = "GFN1 repulsion coordinate differences overflow floating-point range";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if (distance_squared < kMinimumDistanceSquared || distance_squared > cutoff_squared) {
      continue;
    }
    const xtbloom_status_t status = function(other_index, dx, dy, dz, distance_squared);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

struct PairContribution {
  double energy;
  double force[3];
};

xtbloom_status_t evaluate_pair(const RepulsionPlan& plan, std::size_t first, std::size_t second,
                               double dx, double dy, double dz, double distance_squared,
                               bool with_forces, PairContribution& contribution,
                               std::string& error) {
  const double distance = std::sqrt(distance_squared);
  const double distance_power = distance * std::sqrt(distance);
  const double pair_alpha = plan.sqrt_alpha[first] * plan.sqrt_alpha[second];
  const double pair_charge = plan.effective_charge[first] * plan.effective_charge[second];
  contribution.energy = pair_charge * std::exp(-pair_alpha * distance_power) / distance;
  if (!std::isfinite(distance) || !std::isfinite(distance_power) || !std::isfinite(pair_alpha) ||
      !std::isfinite(pair_charge) || !std::isfinite(contribution.energy)) {
    error = "GFN1 repulsion energy arithmetic exceeded floating-point range";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  if (with_forces) {
    const double force_scale =
        (pair_alpha * parameters::gfn1::kGlobal.repulsion_kexp * distance_power +
         kDistanceDenominatorExponent) *
        contribution.energy / distance_squared;
    contribution.force[0] = force_scale * dx;
    contribution.force[1] = force_scale * dy;
    contribution.force[2] = force_scale * dz;
    if (!std::isfinite(force_scale) || !std::isfinite(contribution.force[0]) ||
        !std::isfinite(contribution.force[1]) || !std::isfinite(contribution.force[2])) {
      error = "GFN1 repulsion force arithmetic exceeded floating-point range";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
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
  } catch (const std::length_error&) {
    error = "GFN1 repulsion plan dimensions exceed host container limits";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

xtbloom_status_t add_repulsion_cpu(const RepulsionPlan& plan, const double* positions,
                                   double* energies, double* forces, std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (positions == nullptr || energies == nullptr || !aligned_double(positions) ||
      !aligned_double(energies) || (forces != nullptr && !aligned_double(forces))) {
    error = "GFN1 repulsion positions and outputs must not be NULL or misaligned";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t batch_count = static_cast<std::size_t>(plan.batch_size);
  const std::size_t atom_count = static_cast<std::size_t>(plan.total_atoms);
  std::size_t position_bytes = 0u;
  std::size_t energy_bytes = 0u;
  if (!count_bytes(atom_count, 3u * sizeof(double), position_bytes) ||
      !count_bytes(batch_count, sizeof(double), energy_bytes) ||
      ranges_overlap(positions, position_bytes, energies, energy_bytes) ||
      overlaps_plan_or_error(energies, energy_bytes, plan, error)) {
    error = "GFN1 repulsion energies must be disjoint from inputs and control storage";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (forces != nullptr && (ranges_overlap(positions, position_bytes, forces, position_bytes) ||
                            ranges_overlap(energies, energy_bytes, forces, position_bytes) ||
                            overlaps_plan_or_error(forces, position_bytes, plan, error))) {
    error = "GFN1 repulsion forces must be disjoint from inputs and control storage";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  status = validate_positions(plan, positions, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  for (std::size_t batch = 0; batch < batch_count; ++batch) {
    if (!std::isfinite(energies[batch])) {
      error = "GFN1 repulsion energy accumulators contain NaN or infinity";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    double candidate = energies[batch];
    status = for_each_active_pair_in_batch(
        plan, positions, batch,
        [&](std::size_t first, std::size_t second, double dx, double dy, double dz,
            double distance_squared) -> xtbloom_status_t {
          PairContribution contribution{};
          const xtbloom_status_t pair_status =
              evaluate_pair(plan, first, second, dx, dy, dz, distance_squared, forces != nullptr,
                            contribution, error);
          if (pair_status != XTBLOOM_STATUS_SUCCESS) {
            return pair_status;
          }
          candidate += contribution.energy;
          if (!std::isfinite(candidate)) {
            error = "GFN1 repulsion energy accumulation exceeded floating-point range";
            return XTBLOOM_STATUS_INTERNAL_ERROR;
          }
          return XTBLOOM_STATUS_SUCCESS;
        },
        error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
  }
  if (forces != nullptr) {
    for (std::size_t coordinate = 0; coordinate < atom_count * 3u; ++coordinate) {
      if (!std::isfinite(forces[coordinate])) {
        error = "GFN1 repulsion force accumulators contain NaN or infinity";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
    }
    for (std::size_t atom = 0; atom < atom_count; ++atom) {
      double candidate[3]{forces[atom * 3u], forces[atom * 3u + 1u], forces[atom * 3u + 2u]};
      status = for_each_active_pair_of_atom(
          plan, positions, atom,
          [&](std::size_t other, double dx, double dy, double dz,
              double distance_squared) -> xtbloom_status_t {
            PairContribution contribution{};
            const xtbloom_status_t pair_status = evaluate_pair(
                plan, atom, other, dx, dy, dz, distance_squared, true, contribution, error);
            if (pair_status != XTBLOOM_STATUS_SUCCESS) {
              return pair_status;
            }
            /*
             * Evaluate the pair once for all axes while retaining the exact
             * per-component accumulation order used by publication.
             */
            for (std::size_t axis = 0; axis < 3u; ++axis) {
              candidate[axis] += contribution.force[axis];
              if (!std::isfinite(candidate[axis])) {
                error = "GFN1 repulsion force accumulation exceeded floating-point range";
                return XTBLOOM_STATUS_INTERNAL_ERROR;
              }
            }
            return XTBLOOM_STATUS_SUCCESS;
          },
          error);
      if (status != XTBLOOM_STATUS_SUCCESS) {
        return status;
      }
    }
  }

  status = for_each_active_pair(
      plan, positions,
      [&](std::size_t batch, std::size_t first, std::size_t second, double dx, double dy, double dz,
          double distance_squared) -> xtbloom_status_t {
        PairContribution contribution{};
        const xtbloom_status_t pair_status =
            evaluate_pair(plan, first, second, dx, dy, dz, distance_squared, forces != nullptr,
                          contribution, error);
        if (pair_status == XTBLOOM_STATUS_SUCCESS) {
          energies[batch] += contribution.energy;
          if (forces != nullptr) {
            for (std::size_t axis = 0; axis < 3u; ++axis) {
              forces[first * 3u + axis] += contribution.force[axis];
              forces[second * 3u + axis] -= contribution.force[axis];
            }
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
