#include "model/gfn2/coordination.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <utility>

#include "data/parameters/gfn2.hpp"

namespace gpuxtb::detail::gfn2 {
namespace {

constexpr double kCutoffBohr = 25.0;
constexpr double kMinimumDistanceSquared = 1.0e-12;
constexpr double kFirstSteepness = 10.0;
constexpr double kSecondSteepness = 20.0;
constexpr double kSecondRadiusShiftBohr = 2.0;

/*
 * mctc-lib v0.5.2 derives this conversion from its CODATA-2018 constants.
 * Keeping the evaluated binary64 value reproduces its GFN2 reference CNs.
 */
constexpr double kAngstromToBohr = 1.8897261246204404;

/*
 * Pyykko--Atsumi (2009) covalent radii in angstrom, with transition-metal
 * values reduced by 10%, as used by mctc-lib's get_covalent_rad. The GFN2
 * double-exponential CN uses 4/3 times these values.
 */
constexpr std::array<double, parameters::gfn2::kElementCount> kCovalentRadiiAngstrom{{
    0.32, 0.46, 1.20, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67, 1.40, 1.25, 1.13, 1.04, 1.10,
    1.02, 0.99, 0.96, 1.76, 1.54, 1.33, 1.22, 1.21, 1.10, 1.07, 1.04, 1.00, 0.99, 1.01, 1.09,
    1.12, 1.09, 1.15, 1.10, 1.14, 1.17, 1.89, 1.67, 1.47, 1.39, 1.32, 1.24, 1.15, 1.13, 1.13,
    1.08, 1.15, 1.23, 1.28, 1.26, 1.26, 1.23, 1.32, 1.31, 2.09, 1.76, 1.62, 1.47, 1.58, 1.57,
    1.56, 1.55, 1.51, 1.52, 1.51, 1.50, 1.49, 1.49, 1.48, 1.53, 1.46, 1.37, 1.31, 1.23, 1.18,
    1.16, 1.11, 1.12, 1.13, 1.32, 1.30, 1.30, 1.36, 1.31, 1.38, 1.42,
}};

static_assert(parameters::gfn2::kGlobal.coordination_number_model == 0u,
              "the generated GFN2 parameters must select the dexp CN model");

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

gpuxtb_status_t validate_plan(const CoordinationPlan& plan, std::string& error) {
  if (plan.batch_size <= 0 || plan.total_atoms <= 0 || !representable_as_size(plan.batch_size) ||
      !representable_geometry_size(plan.total_atoms) ||
      static_cast<std::uint64_t>(plan.batch_size) >=
          static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    error = "coordination plan has invalid batch or atom counts";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  const auto atom_count = static_cast<std::size_t>(plan.total_atoms);
  if (plan.atom_offsets.size() != static_cast<std::size_t>(plan.batch_size + 1) ||
      plan.covalent_radius.size() != atom_count || plan.atom_offsets.front() != 0 ||
      plan.atom_offsets.back() != plan.total_atoms) {
    error = "coordination plan is incomplete or internally inconsistent";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t batch = 0; batch < plan.batch_size; ++batch) {
    const std::int64_t begin = plan.atom_offsets[static_cast<std::size_t>(batch)];
    const std::int64_t end = plan.atom_offsets[static_cast<std::size_t>(batch + 1)];
    if (begin < 0 || begin > end || end > plan.total_atoms) {
      error = "coordination plan offsets are not a valid ragged partition";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  for (double radius : plan.covalent_radius) {
    if (!(radius > 0.0) || !std::isfinite(radius)) {
      error = "coordination plan contains an invalid covalent radius";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_positions(const CoordinationPlan& plan, const double* positions,
                                   std::string& error) {
  if (positions == nullptr) {
    error = "coordination positions must not be NULL";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const auto coordinate_count = static_cast<std::size_t>(plan.total_atoms) * 3u;
  for (std::size_t coordinate = 0; coordinate < coordinate_count; ++coordinate) {
    if (!std::isfinite(positions[coordinate])) {
      error = "coordination positions contain NaN or infinity";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

/* Stable logistic form of 1 / (1 + exp(-argument)). */
double logistic(double argument) {
  if (argument >= 0.0) {
    const double exponential = std::exp(-argument);
    return 1.0 / (1.0 + exponential);
  }
  const double exponential = std::exp(argument);
  return exponential / (1.0 + exponential);
}

struct PairCount {
  double value;
  double derivative;
};

PairCount double_exponential_count(double distance, double radius) {
  const double inverse_distance = 1.0 / distance;
  const double inverse_distance_squared = inverse_distance * inverse_distance;
  const double shifted_radius = radius + kSecondRadiusShiftBohr;
  const double first = logistic(kFirstSteepness * (radius * inverse_distance - 1.0));
  const double second = logistic(kSecondSteepness * (shifted_radius * inverse_distance - 1.0));

  PairCount result{};
  result.value = first * second;
  result.derivative = -inverse_distance_squared *
                      (kFirstSteepness * radius * first * (1.0 - first) * second +
                       kSecondSteepness * shifted_radius * second * (1.0 - second) * first);
  return result;
}

}  // namespace

gpuxtb_status_t make_coordination_plan(std::int64_t batch_size, std::int64_t total_atoms,
                                       const std::int64_t* atom_offsets,
                                       const std::int32_t* atomic_numbers, CoordinationPlan& plan,
                                       std::string& error) {
  if (batch_size <= 0 || total_atoms <= 0 || !representable_as_size(batch_size) ||
      !representable_geometry_size(total_atoms) ||
      static_cast<std::uint64_t>(batch_size) >=
          static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    error = "coordination plan requires positive, representable batch and atom counts";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (atom_offsets == nullptr || atomic_numbers == nullptr) {
    error = "coordination plan offsets and atomic numbers must not be NULL";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (atom_offsets[0] != 0 || atom_offsets[batch_size] != total_atoms) {
    error = "coordination plan offsets must start at zero and end at total_atoms";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    if (atom_offsets[batch] > atom_offsets[batch + 1]) {
      error = "coordination plan offsets must be monotonically nondecreasing";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  try {
    CoordinationPlan created;
    created.batch_size = batch_size;
    created.total_atoms = total_atoms;
    created.atom_offsets.assign(atom_offsets, atom_offsets + batch_size + 1);
    created.covalent_radius.resize(static_cast<std::size_t>(total_atoms));

    constexpr double radius_scale = (4.0 / 3.0) * kAngstromToBohr;
    for (std::int64_t atom = 0; atom < total_atoms; ++atom) {
      const std::int32_t atomic_number = atomic_numbers[atom];
      const auto* element =
          parameters::gfn2::find_element(static_cast<std::uint32_t>(atomic_number));
      if (element == nullptr || element->atomic_number != atomic_number) {
        error = "coordination plan contains an unsupported atomic number";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      created.covalent_radius[static_cast<std::size_t>(atom)] =
          radius_scale * kCovalentRadiiAngstrom[static_cast<std::size_t>(atomic_number - 1)];
    }

    plan = std::move(created);
    error.clear();
    return GPUXTB_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN2 coordination plan";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  }
}

gpuxtb_status_t evaluate_coordination_cpu(const CoordinationPlan& plan, const double* positions,
                                          double* coordination_numbers, std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (coordination_numbers == nullptr) {
    error = "coordination output must not be NULL";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  status = validate_positions(plan, positions, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  const auto atom_count = static_cast<std::size_t>(plan.total_atoms);
  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    coordination_numbers[atom] = 0.0;
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

        if (distance_squared < kMinimumDistanceSquared) {
          /*
           * mctc-lib needs this threshold to discard the iat == jat,
           * zero-translation self image in its periodic-capable loop. This
           * nonperiodic pair list already excludes self pairs, so reaching the
           * threshold means two distinct atoms are coincident or nearly so.
           */
          error = "coordination is undefined for coincident or near-coincident atoms";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        if (distance_squared > cutoff_squared) {
          continue;
        }

        const double radius =
            plan.covalent_radius[first_index] + plan.covalent_radius[second_index];
        const double count = double_exponential_count(std::sqrt(distance_squared), radius).value;
        coordination_numbers[first_index] += count;
        coordination_numbers[second_index] += count;
      }
    }
  }

  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t add_coordination_gradient_cpu(const CoordinationPlan& plan, const double* positions,
                                              const double* dE_dcn, double* gradients,
                                              std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (dE_dcn == nullptr || gradients == nullptr) {
    error = "coordination derivative inputs and gradients must not be NULL";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  status = validate_positions(plan, positions, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const auto atom_count = static_cast<std::size_t>(plan.total_atoms);
  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    if (!std::isfinite(dE_dcn[atom])) {
      error = "coordination derivatives contain NaN or infinity";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
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
        if (distance_squared < kMinimumDistanceSquared) {
          error = "coordination derivative is undefined for coincident or near-coincident atoms";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        if (distance_squared > cutoff_squared) {
          continue;
        }

        const double distance = std::sqrt(distance_squared);
        const double radius =
            plan.covalent_radius[first_index] + plan.covalent_radius[second_index];
        const double pair_derivative = double_exponential_count(distance, radius).derivative;
        const double weight = dE_dcn[first_index] + dE_dcn[second_index];
        const double scale = weight * pair_derivative / distance;
        const double gx = scale * dx;
        const double gy = scale * dy;
        const double gz = scale * dz;

        gradients[first_index * 3] += gx;
        gradients[first_index * 3 + 1] += gy;
        gradients[first_index * 3 + 2] += gz;
        gradients[second_index * 3] -= gx;
        gradients[second_index * 3 + 1] -= gy;
        gradients[second_index * 3 + 2] -= gz;
      }
    }
  }

  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

}  // namespace gpuxtb::detail::gfn2
