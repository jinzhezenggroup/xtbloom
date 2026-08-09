#include "model/gfn2/repulsion.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <utility>

#include "data/parameters/gfn2.hpp"

namespace xtbloom::detail::gfn2 {
namespace {

constexpr double kCutoffBohr = 25.0;
constexpr double kMinimumDistanceSquared = 1.0e-24;

bool representable_as_size(std::int64_t value) {
  return value >= 0 && static_cast<std::uint64_t>(value) <= std::numeric_limits<std::size_t>::max();
}

bool representable_geometry_size(std::int64_t atom_count) {
  if (!representable_as_size(atom_count)) {
    return false;
  }
  const auto count = static_cast<std::uint64_t>(atom_count);
  return count <= std::numeric_limits<std::size_t>::max() / 3u &&
         count <= static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / 3u;
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
    error = "repulsion plan requires positive, representable batch and atom counts";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (atom_offsets == nullptr || atomic_numbers == nullptr) {
    error = "repulsion plan offsets and atomic numbers must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (atom_offsets[0] != 0 || atom_offsets[batch_size] != total_atoms) {
    error = "repulsion plan offsets must start at zero and end at total_atoms";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    if (atom_offsets[batch] > atom_offsets[batch + 1]) {
      error = "repulsion plan offsets must be monotonically nondecreasing";
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
    created.light_element.resize(static_cast<std::size_t>(total_atoms));

    for (std::int64_t atom = 0; atom < total_atoms; ++atom) {
      const std::int32_t atomic_number = atomic_numbers[atom];
      const auto* element =
          parameters::gfn2::find_element(static_cast<std::uint32_t>(atomic_number));
      if (element == nullptr || !(element->arep > 0.0) || !(element->zeff > 0.0) ||
          !std::isfinite(element->arep) || !std::isfinite(element->zeff)) {
        error = "repulsion plan contains an unsupported atomic number or invalid parameter";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }

      const std::size_t index = static_cast<std::size_t>(atom);
      created.sqrt_alpha[index] = std::sqrt(element->arep);
      created.effective_charge[index] = element->zeff;
      created.light_element[index] = atomic_number <= 2 ? 1u : 0u;
    }

    plan = std::move(created);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN2 repulsion plan";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

xtbloom_status_t add_repulsion_cpu(const RepulsionPlan& plan, const double* positions,
                                   double* energies, double* forces, std::string& error) {
  const auto atom_count = static_cast<std::size_t>(plan.total_atoms);
  if (plan.batch_size <= 0 || plan.total_atoms <= 0 ||
      !representable_geometry_size(plan.total_atoms) ||
      plan.atom_offsets.size() != static_cast<std::size_t>(plan.batch_size + 1) ||
      plan.sqrt_alpha.size() != atom_count || plan.effective_charge.size() != atom_count ||
      plan.light_element.size() != atom_count) {
    error = "repulsion plan is incomplete or internally inconsistent";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (plan.atom_offsets.front() != 0 || plan.atom_offsets.back() != plan.total_atoms) {
    error = "repulsion plan offsets do not span the stored atoms";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t batch = 0; batch < plan.batch_size; ++batch) {
    const std::int64_t begin = plan.atom_offsets[static_cast<std::size_t>(batch)];
    const std::int64_t end = plan.atom_offsets[static_cast<std::size_t>(batch + 1)];
    if (begin < 0 || begin > end || end > plan.total_atoms) {
      error = "repulsion plan offsets are not a valid ragged partition";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  if (positions == nullptr || energies == nullptr) {
    error = "repulsion positions and energies must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::size_t coordinate = 0; coordinate < atom_count * 3; ++coordinate) {
    if (!std::isfinite(positions[coordinate])) {
      error = "repulsion positions contain NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  constexpr double cutoff_squared = kCutoffBohr * kCutoffBohr;
  for (std::int64_t batch = 0; batch < plan.batch_size; ++batch) {
    const std::int64_t begin = plan.atom_offsets[static_cast<std::size_t>(batch)];
    const std::int64_t end = plan.atom_offsets[static_cast<std::size_t>(batch + 1)];
    for (std::int64_t first = begin; first < end; ++first) {
      const std::size_t first_index = static_cast<std::size_t>(first);
      for (std::int64_t second = begin; second < first; ++second) {
        const std::size_t second_index = static_cast<std::size_t>(second);
        const double dx = positions[first_index * 3] - positions[second_index * 3];
        const double dy = positions[first_index * 3 + 1] - positions[second_index * 3 + 1];
        const double dz = positions[first_index * 3 + 2] - positions[second_index * 3 + 2];
        const double distance_squared = dx * dx + dy * dy + dz * dz;
        if (distance_squared <= kMinimumDistanceSquared) {
          error = "repulsion is undefined for coincident atoms in one molecule";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        if (distance_squared > cutoff_squared) {
          continue;
        }

        const double distance = std::sqrt(distance_squared);
        const bool light_pair =
            plan.light_element[first_index] != 0u && plan.light_element[second_index] != 0u;
        const double exponent = light_pair ? parameters::gfn2::kGlobal.repulsion_klight
                                           : parameters::gfn2::kGlobal.repulsion_kexp;
        const double distance_power = light_pair ? distance : distance * std::sqrt(distance);
        const double pair_alpha = plan.sqrt_alpha[first_index] * plan.sqrt_alpha[second_index];
        const double pair_charge =
            plan.effective_charge[first_index] * plan.effective_charge[second_index];
        const double pair_energy = pair_charge * std::exp(-pair_alpha * distance_power) / distance;
        energies[batch] += pair_energy;

        if (forces != nullptr) {
          const double force_scale =
              (pair_alpha * exponent * distance_power + 1.0) * pair_energy / distance_squared;
          const double fx = force_scale * dx;
          const double fy = force_scale * dy;
          const double fz = force_scale * dz;
          forces[first_index * 3] += fx;
          forces[first_index * 3 + 1] += fy;
          forces[first_index * 3 + 2] += fz;
          forces[second_index * 3] -= fx;
          forces[second_index * 3 + 1] -= fy;
          forces[second_index * 3 + 2] -= fz;
        }
      }
    }
  }

  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn2
