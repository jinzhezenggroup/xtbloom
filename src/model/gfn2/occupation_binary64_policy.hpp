#pragma once
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define GPUXTB_OCCUPATION_HD __host__ __device__
#else
#define GPUXTB_OCCUPATION_HD
#endif

namespace gpuxtb::detail::gfn2::binary64_policy {

constexpr double kMaximum = 1.79769313486231570814527423731704357e308;
constexpr double kEpsilon = 2.220446049250313080847263336181640625e-16;
constexpr double kRepresentableErrorScale = 2.0 * kEpsilon;
constexpr int kMaximumRootIterations = 4096;

GPUXTB_OCCUPATION_HD inline double maximum(double first, double second) {
  return first > second ? first : second;
}

GPUXTB_OCCUPATION_HD inline double absolute(double value) { return value < 0.0 ? -value : value; }

GPUXTB_OCCUPATION_HD inline bool finite(double value) {
#if defined(__CUDA_ARCH__)
  return isfinite(value);
#else
  return std::isfinite(value);
#endif
}

GPUXTB_OCCUPATION_HD inline bool negative_sign(double value) {
#if defined(__CUDA_ARCH__)
  return signbit(value);
#else
  return std::signbit(value);
#endif
}

GPUXTB_OCCUPATION_HD inline double exponential(double value) {
#if defined(__CUDA_ARCH__)
  return exp(value);
#else
  return std::exp(value);
#endif
}

GPUXTB_OCCUPATION_HD inline double logarithm(double value) {
#if defined(__CUDA_ARCH__)
  return log(value);
#else
  return std::log(value);
#endif
}

GPUXTB_OCCUPATION_HD inline double logarithm_one_plus(double value) {
#if defined(__CUDA_ARCH__)
  return log1p(value);
#else
  return std::log1p(value);
#endif
}

GPUXTB_OCCUPATION_HD inline double adjacent(double value, double direction) {
#if defined(__CUDA_ARCH__)
  return nextafter(value, direction);
#else
  return std::nextafter(value, direction);
#endif
}

GPUXTB_OCCUPATION_HD inline double saturated_add(double first, double second) {
  const double result = first + second;
  if (finite(result)) {
    return result;
  }
  return negative_sign(first) == negative_sign(second) && negative_sign(first) ? -kMaximum
                                                                               : kMaximum;
}

GPUXTB_OCCUPATION_HD inline double saturated_subtract(double first, double second) {
  const double result = first - second;
  if (finite(result)) {
    return result;
  }
  return first < second ? -kMaximum : kMaximum;
}

GPUXTB_OCCUPATION_HD inline double saturated_multiply_nonnegative(double first, double second) {
  if (first == 0.0 || second == 0.0) {
    return 0.0;
  }
  return first > kMaximum / second ? kMaximum : first * second;
}

GPUXTB_OCCUPATION_HD inline double saturated_affine(double reference, double multiplier,
                                                    double scale) {
  const double product = multiplier * scale;
  if (finite(product)) {
    return saturated_add(reference, product);
  }
  const double normalized = reference / kMaximum + multiplier * (scale / kMaximum);
  if (normalized >= 1.0) {
    return kMaximum;
  }
  if (normalized <= -1.0) {
    return -kMaximum;
  }
  return normalized * kMaximum;
}

GPUXTB_OCCUPATION_HD inline double scaled_energy_difference(double energy, double reference,
                                                            double temperature) {
  if (negative_sign(energy) == negative_sign(reference)) {
    const double result = (energy - reference) / temperature;
    if (finite(result)) {
      return result;
    }
    return energy < reference ? -kMaximum : kMaximum;
  }
  const double scaled_energy = energy / temperature;
  const double scaled_reference = reference / temperature;
  const double result = scaled_energy - scaled_reference;
  if (finite(result)) {
    return result;
  }
  return energy < reference ? -kMaximum : kMaximum;
}

GPUXTB_OCCUPATION_HD inline double stable_middle(double lower, double upper) {
  return 0.5 * lower + 0.5 * upper;
}

GPUXTB_OCCUPATION_HD inline double fermi_value(double scaled_energy, double scaled_mu) {
  const double argument = saturated_subtract(scaled_energy, scaled_mu);
  if (argument >= 0.0) {
    const double value = exponential(-argument);
    return value / (1.0 + value);
  }
  return 1.0 / (exponential(argument) + 1.0);
}

GPUXTB_OCCUPATION_HD inline double fermi_hole_value(double scaled_energy, double scaled_mu) {
  const double argument = saturated_subtract(scaled_energy, scaled_mu);
  if (argument >= 0.0) {
    return 1.0 / (exponential(-argument) + 1.0);
  }
  const double value = exponential(argument);
  return value / (1.0 + value);
}

struct CompensatedSum {
  double value = 0.0;
  double compensation = 0.0;
};

GPUXTB_OCCUPATION_HD inline void add_compensated(CompensatedSum& sum, double value) {
  const double corrected = value - sum.compensation;
  const double updated = sum.value + corrected;
  sum.compensation = (updated - sum.value) - corrected;
  sum.value = updated;
}

GPUXTB_OCCUPATION_HD inline double quantity(const double* eigenvalues, std::int64_t count,
                                            double energy_reference, double scaled_mu,
                                            double temperature, bool solve_holes) {
  CompensatedSum sum{};
  for (std::int64_t orbital = 0; orbital < count; ++orbital) {
    const double scaled_energy =
        scaled_energy_difference(eigenvalues[orbital], energy_reference, temperature);
    add_compensated(sum, solve_holes ? fermi_hole_value(scaled_energy, scaled_mu)
                                     : fermi_value(scaled_energy, scaled_mu));
  }
  return sum.value;
}

struct Root {
  double energy_reference = 0.0;
  double lower = 0.0;
  double upper = 0.0;
  double scaled_mu = 0.0;
  double quantity = 0.0;
  bool spacing_exhausted = false;
  bool retried_at_frontier = false;
  std::int64_t frontier_begin = -1;
  std::int64_t frontier_end = -1;
};

/*
 * Solve in a translated binary64 frame. CUDA necessarily uses this arithmetic,
 * and the CPU rare path deliberately reuses it so publication candidates and
 * chemical-potential tie breaking do not depend on wider host long double.
 */
GPUXTB_OCCUPATION_HD inline bool solve_root_once(const double* eigenvalues, std::int64_t count,
                                                 double quantity_target, double temperature,
                                                 bool solve_holes, double energy_reference,
                                                 Root& root) {
  const double log_fraction = logarithm(quantity_target) - logarithm(static_cast<double>(count));
  const double thermal_steps = maximum(64.0, -log_fraction + 8.0);
  const double scaled_minimum =
      scaled_energy_difference(eigenvalues[0], energy_reference, temperature);
  const double scaled_maximum =
      scaled_energy_difference(eigenvalues[count - 1], energy_reference, temperature);
  const double scaled_span = saturated_subtract(scaled_maximum, scaled_minimum);
  const double energy_scale = maximum(1.0, absolute(scaled_span));
  const double representation_margin =
      saturated_multiply_nonnegative(64.0 * kEpsilon, energy_scale);
  const double margin = saturated_add(thermal_steps, representation_margin);
  double lower = saturated_subtract(scaled_minimum, margin);
  double upper = saturated_add(scaled_maximum, margin);
  const double lower_quantity =
      quantity(eigenvalues, count, energy_reference, lower, temperature, solve_holes);
  const double upper_quantity =
      quantity(eigenvalues, count, energy_reference, upper, temperature, solve_holes);
  const bool bracketed =
      solve_holes ? lower_quantity >= quantity_target && upper_quantity <= quantity_target
                  : lower_quantity <= quantity_target && upper_quantity >= quantity_target;
  if (!finite(lower) || !finite(upper) || !(lower < upper) || !finite(lower_quantity) ||
      !finite(upper_quantity) || !bracketed) {
    return false;
  }

  const double tolerance = 64.0 * kEpsilon * quantity_target;
  bool spacing_exhausted = false;
  for (int iteration = 0; iteration < kMaximumRootIterations; ++iteration) {
    const double middle = stable_middle(lower, upper);
    const double middle_quantity =
        quantity(eigenvalues, count, energy_reference, middle, temperature, solve_holes);
    if (!finite(middle_quantity)) {
      return false;
    }
    if (absolute(middle_quantity - quantity_target) <= tolerance) {
      lower = middle;
      upper = middle;
      break;
    }
    if (middle == lower || middle == upper) {
      spacing_exhausted = true;
      break;
    }
    if ((!solve_holes && middle_quantity < quantity_target) ||
        (solve_holes && middle_quantity > quantity_target)) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  const double scaled_mu = stable_middle(lower, upper);
  const double solved_quantity =
      quantity(eigenvalues, count, energy_reference, scaled_mu, temperature, solve_holes);
  if (!finite(solved_quantity)) {
    return false;
  }
  root = {};
  root.energy_reference = energy_reference;
  root.lower = lower;
  root.upper = upper;
  root.scaled_mu = scaled_mu;
  root.quantity = solved_quantity;
  root.spacing_exhausted = spacing_exhausted;
  return true;
}

GPUXTB_OCCUPATION_HD inline std::int64_t unique_changing_degenerate_frontier(
    const double* eigenvalues, std::int64_t count, double temperature, bool solve_holes,
    const Root& root) {
  std::int64_t frontier = count;
  for (std::int64_t begin = 0; begin < count;) {
    std::int64_t end = begin + 1;
    while (end < count && eigenvalues[end] == eigenvalues[begin]) {
      ++end;
    }
    if (end - begin > 1) {
      const double scaled_energy =
          scaled_energy_difference(eigenvalues[begin], root.energy_reference, temperature);
      const double lower_value = solve_holes ? fermi_hole_value(scaled_energy, root.lower)
                                             : fermi_value(scaled_energy, root.lower);
      const double upper_value = solve_holes ? fermi_hole_value(scaled_energy, root.upper)
                                             : fermi_value(scaled_energy, root.upper);
      if (lower_value != upper_value) {
        if (frontier != count) {
          return -1;
        }
        frontier = begin;
      }
    }
    begin = end;
  }
  return frontier;
}

GPUXTB_OCCUPATION_HD inline bool root_straddles_target(const double* eigenvalues,
                                                       std::int64_t count, double target,
                                                       double temperature, bool solve_holes,
                                                       const Root& root) {
  const double lower_quantity =
      quantity(eigenvalues, count, root.energy_reference, root.lower, temperature, solve_holes);
  const double upper_quantity =
      quantity(eigenvalues, count, root.energy_reference, root.upper, temperature, solve_holes);
  return finite(lower_quantity) && finite(upper_quantity) &&
         (solve_holes ? lower_quantity >= target && upper_quantity <= target
                      : lower_quantity <= target && upper_quantity >= target);
}

GPUXTB_OCCUPATION_HD inline bool solve_root(const double* eigenvalues, std::int64_t count,
                                            double target, double temperature, bool solve_holes,
                                            Root& root) {
  const double reference = solve_holes ? eigenvalues[count - 1] : eigenvalues[0];
  if (!solve_root_once(eigenvalues, count, target, temperature, solve_holes, reference, root)) {
    return false;
  }
  const double tolerance = 64.0 * kEpsilon * target;
  if (!root.spacing_exhausted || absolute(root.quantity - target) <= tolerance ||
      !root_straddles_target(eigenvalues, count, target, temperature, solve_holes, root)) {
    return true;
  }
  const std::int64_t frontier =
      unique_changing_degenerate_frontier(eigenvalues, count, temperature, solve_holes, root);
  if (frontier < 0 || frontier >= count) {
    return true;
  }
  root.frontier_begin = frontier;
  root.frontier_end = frontier + 1;
  while (root.frontier_end < count && eigenvalues[root.frontier_end] == eigenvalues[frontier]) {
    ++root.frontier_end;
  }
  if (eigenvalues[frontier] == root.energy_reference) {
    return true;
  }
  Root retried{};
  if (!solve_root_once(eigenvalues, count, target, temperature, solve_holes, eigenvalues[frontier],
                       retried)) {
    return true;
  }
  const double original_error = absolute(root.quantity - target);
  const double retried_error = absolute(retried.quantity - target);
  if (retried_error < original_error) {
    retried.retried_at_frontier = true;
    retried.frontier_begin = root.frontier_begin;
    retried.frontier_end = root.frontier_end;
    root = retried;
  }
  return true;
}

GPUXTB_OCCUPATION_HD inline std::int64_t largest_degenerate_block(const double* eigenvalues,
                                                                  std::int64_t count) {
  std::int64_t largest = 0;
  for (std::int64_t begin = 0; begin < count;) {
    std::int64_t end = begin + 1;
    while (end < count && eigenvalues[end] == eigenvalues[begin]) {
      ++end;
    }
    if (end - begin > 1 && end - begin > largest) {
      largest = end - begin;
    }
    begin = end;
  }
  return largest;
}

GPUXTB_OCCUPATION_HD inline double baseline_occupation(const double* eigenvalues,
                                                       std::int64_t orbital, double temperature,
                                                       const Root& root) {
  return fermi_value(
      scaled_energy_difference(eigenvalues[orbital], root.energy_reference, temperature),
      root.scaled_mu);
}

struct Correction {
  std::int64_t begin = 0;
  std::int64_t end = 0;
  double occupation = 0.0;
};

struct Publication {
  Correction corrections[2]{};
  int correction_count = 0;
  bool relaxed = false;
  double relaxed_tolerance = 0.0;
  double compensated_error = 0.0;
};

GPUXTB_OCCUPATION_HD inline double published_occupation(const double* eigenvalues,
                                                        std::int64_t orbital, double temperature,
                                                        const Root& root,
                                                        const Publication& publication) {
  for (int correction = publication.correction_count; correction > 0; --correction) {
    const Correction& selected = publication.corrections[correction - 1];
    if (orbital >= selected.begin && orbital < selected.end) {
      return selected.occupation;
    }
  }
  return baseline_occupation(eigenvalues, orbital, temperature, root);
}

struct Candidate {
  bool found = false;
  double error = kMaximum;
  bool fractional = false;
  std::int64_t begin = 0;
  std::int64_t end = 0;
  double occupation = 0.0;
  double tolerance = 0.0;
};

GPUXTB_OCCUPATION_HD inline bool better_candidate(double error, bool fractional, std::int64_t begin,
                                                  double occupation, const Candidate& best) {
  return !best.found || error < best.error ||
         (error == best.error && fractional && !best.fractional) ||
         (error == best.error && fractional == best.fractional && occupation < best.occupation) ||
         (error == best.error && fractional == best.fractional && occupation == best.occupation &&
          begin < best.begin);
}

GPUXTB_OCCUPATION_HD inline double compensated_publication_error(
    const double* eigenvalues, std::int64_t count, double temperature, const Root& root,
    const Publication& publication, bool solve_holes, double target) {
  CompensatedSum sum{};
  for (std::int64_t orbital = 0; orbital < count; ++orbital) {
    const double occupation =
        published_occupation(eigenvalues, orbital, temperature, root, publication);
    add_compensated(sum, solve_holes ? 1.0 - occupation : occupation);
  }
  return absolute((sum.value - target) - sum.compensation);
}

struct DoubleDouble {
  double high = 0.0;
  double low = 0.0;
};

GPUXTB_OCCUPATION_HD inline void add_double_double(DoubleDouble& total, double value) {
  const double sum = total.high + value;
  const double virtual_value = sum - total.high;
  const double error = (total.high - (sum - virtual_value)) + (value - virtual_value) + total.low;
  const double high = sum + error;
  total.low = error - (high - sum);
  total.high = high;
}

GPUXTB_OCCUPATION_HD inline DoubleDouble audited_publication_residual(
    const double* eigenvalues, std::int64_t count, double temperature, const Root& root,
    const Publication& publication, bool solve_holes, double target) {
  DoubleDouble total{};
  for (std::int64_t orbital = 0; orbital < count; ++orbital) {
    const double occupation =
        published_occupation(eigenvalues, orbital, temperature, root, publication);
    add_double_double(total, solve_holes ? 1.0 - occupation : occupation);
  }
  add_double_double(total, -target);
  return total;
}

GPUXTB_OCCUPATION_HD inline int double_double_sign(const DoubleDouble& value) {
  if (value.high > 0.0) {
    return 1;
  }
  if (value.high < 0.0) {
    return -1;
  }
  return value.low > 0.0 ? 1 : (value.low < 0.0 ? -1 : 0);
}

GPUXTB_OCCUPATION_HD inline bool double_double_within(const DoubleDouble& residual,
                                                      double tolerance) {
  DoubleDouble below_upper = residual;
  DoubleDouble above_lower = residual;
  add_double_double(below_upper, -tolerance);
  add_double_double(above_lower, tolerance);
  return double_double_sign(below_upper) <= 0 && double_double_sign(above_lower) >= 0;
}

GPUXTB_OCCUPATION_HD inline bool audited_publication_within(
    const double* eigenvalues, std::int64_t count, double temperature, const Root& root,
    const Publication& publication, bool solve_holes, double target, double tolerance) {
  const DoubleDouble residual = audited_publication_residual(eigenvalues, count, temperature, root,
                                                             publication, solve_holes, target);
  return double_double_within(residual, tolerance);
}

GPUXTB_OCCUPATION_HD inline double publication_entropy(const double* eigenvalues,
                                                       std::int64_t count, double temperature,
                                                       const Root& root,
                                                       const Publication& publication) {
  CompensatedSum entropy{};
  for (std::int64_t orbital = 0; orbital < count; ++orbital) {
    const double occupation =
        published_occupation(eigenvalues, orbital, temperature, root, publication);
    if (occupation > 0.0 && occupation < 1.0) {
      const double hole = 1.0 - occupation;
      add_compensated(entropy, -occupation * logarithm(occupation) - hole * logarithm(hole));
    }
  }
  return entropy.value;
}

GPUXTB_OCCUPATION_HD inline bool select_publication(const double* eigenvalues, std::int64_t count,
                                                    double target, double temperature,
                                                    bool solve_holes, const Root& root,
                                                    Publication& publication) {
  publication = {};
  const double strict_tolerance = 64.0 * kEpsilon * target;
  publication.compensated_error = compensated_publication_error(
      eigenvalues, count, temperature, root, publication, solve_holes, target);
  if (audited_publication_within(eigenvalues, count, temperature, root, publication, solve_holes,
                                 target, strict_tolerance)) {
    return true;
  }

  for (int phase = 0; phase < 2; ++phase) {
    Candidate strict{};
    Candidate relaxed{};
    for (std::int64_t begin = 0; begin < count;) {
      std::int64_t end = begin + 1;
      while (end < count && eigenvalues[end] == eigenvalues[begin]) {
        ++end;
      }
      const std::int64_t block_count = end - begin;
      const double old_occupation =
          published_occupation(eigenvalues, begin, temperature, root, publication);
      CompensatedSum other{};
      for (std::int64_t orbital = 0; orbital < count; ++orbital) {
        if (orbital >= begin && orbital < end) {
          continue;
        }
        const double occupation =
            published_occupation(eigenvalues, orbital, temperature, root, publication);
        add_compensated(other, solve_holes ? 1.0 - occupation : occupation);
      }
      const double desired_block_quantity = (target - other.value) + other.compensation;
      const double desired_per_member = desired_block_quantity / static_cast<double>(block_count);
      const double nearest = solve_holes ? 1.0 - desired_per_member : desired_per_member;
      const double nearest_down = adjacent(nearest, -kMaximum);
      const double nearest_up = adjacent(nearest, kMaximum);
      const double candidates[8] = {
          nearest,
          nearest_down,
          nearest_up,
          adjacent(nearest_down, -kMaximum),
          adjacent(nearest_up, kMaximum),
          old_occupation,
          adjacent(old_occupation, -kMaximum),
          adjacent(old_occupation, kMaximum),
      };
      for (int candidate_index = 0; candidate_index < 8; ++candidate_index) {
        const double block_occupation = candidates[candidate_index];
        bool duplicate = false;
        for (int previous = 0; previous < candidate_index; ++previous) {
          duplicate = duplicate || block_occupation == candidates[previous];
        }
        if (!finite(block_occupation) || block_occupation < 0.0 || block_occupation > 1.0 ||
            duplicate) {
          continue;
        }
        Publication trial = publication;
        if (trial.correction_count >= 2) {
          continue;
        }
        trial.corrections[trial.correction_count++] = {begin, end, block_occupation};
        const double error = compensated_publication_error(eigenvalues, count, temperature, root,
                                                           trial, solve_holes, target);
        const bool fractional = block_occupation > 0.0 && block_occupation < 1.0;
        if (audited_publication_within(eigenvalues, count, temperature, root, trial, solve_holes,
                                       target, strict_tolerance) &&
            better_candidate(error, fractional, begin, block_occupation, strict)) {
          strict = {true, error, fractional, begin, end, block_occupation, strict_tolerance};
        }
        if (phase == 0 && block_count > 1) {
          const double relaxed_tolerance =
              kRepresentableErrorScale * static_cast<double>(block_count);
          if (audited_publication_within(eigenvalues, count, temperature, root, trial, solve_holes,
                                         target, relaxed_tolerance) &&
              better_candidate(error, fractional, begin, block_occupation, relaxed)) {
            relaxed = {true, error, fractional, begin, end, block_occupation, relaxed_tolerance};
          }
        }
      }
      begin = end;
    }
    const Candidate selected = strict.found ? strict : relaxed;
    if (!selected.found) {
      return publication.relaxed &&
             audited_publication_within(eigenvalues, count, temperature, root, publication,
                                        solve_holes, target, publication.relaxed_tolerance);
    }
    publication.corrections[publication.correction_count++] = {selected.begin, selected.end,
                                                               selected.occupation};
    publication.compensated_error = selected.error;
    if (strict.found) {
      publication.relaxed = false;
      publication.relaxed_tolerance = 0.0;
      break;
    }
    publication.relaxed = true;
    publication.relaxed_tolerance = selected.tolerance;
  }
  return audited_publication_within(eigenvalues, count, temperature, root, publication, solve_holes,
                                    target, strict_tolerance) ||
         (publication.relaxed &&
          audited_publication_within(eigenvalues, count, temperature, root, publication,
                                     solve_holes, target, publication.relaxed_tolerance));
}

GPUXTB_OCCUPATION_HD inline double root_acceptance_tolerance(double target, const Root& root) {
  double tolerance = 1024.0 * kEpsilon * target;
  if (root.frontier_begin >= 0 && root.frontier_end > root.frontier_begin) {
    const double frontier_floor =
        kRepresentableErrorScale * static_cast<double>(root.frontier_end - root.frontier_begin);
    tolerance = maximum(tolerance, frontier_floor);
  }
  return tolerance;
}

}  // namespace gpuxtb::detail::gfn2::binary64_policy

#undef GPUXTB_OCCUPATION_HD
