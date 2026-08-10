#include "model/gfn2/coordination.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

int test_mctc_reference() {
  /*
   * This structure and the reference values come from mctc-lib's
   * Apache-2.0 test/test_ncoord.f90 at the tblite-pinned v0.5.2 tag.
   * Its maximum interatomic distance is below both the reference's explicit
   * 30-bohr test cutoff and GFN2's 25-bohr default cutoff.
   */
  constexpr std::array<std::int32_t, 16> atomic_numbers{
      11, 1, 8, 1, 9, 1, 1, 8, 7, 1, 1, 17, 5, 5, 7, 13,
  };
  constexpr std::array<double, 48> positions{
      -1.85528263484662, 3.58670515364616,  -2.41763729306344, 4.40178023537845,  0.02338844412653,
      -4.95457749372945, -2.98706033463438, 4.76252065456814,  1.27043301573532,  0.79980886075526,
      1.41103455609189,  -5.04655321620119, -4.20647469409936, 1.84275767548460,  4.55038084858449,
      -3.54356121843970, -3.18835665176557, 1.46240021785588,  2.70032160109941,  1.06818452504054,
      -1.73234650374438, 3.73114088824361,  -2.07001543363453, 2.23160937604731,  -1.75306819230397,
      0.35951417150421,  1.05323406177129,  5.41755788583825,  -1.57881830078929, 1.75394002750038,
      -2.23462868255966, -2.13856505054269, 4.10922285746451,  1.01565866207568,  -3.21952154552768,
      -3.36050963020778, 2.42119255723593,  0.26626435093114,  -3.91862474360560, -3.02526098819107,
      2.53667889095925,  2.31664984740423,  -2.00438948664892, -2.29235136977220, 2.19782807357059,
      1.12226554109716,  -1.36942007032045, 0.48455055461782,
  };
  constexpr std::array<double, 16> expected{
      4.11453659059991, 0.932058998762811, 2.03554597140311, 1.42227835389358,
      1.12812426574031, 1.05491602558828,  1.52709064704269, 1.95070367247232,
      3.83759889196540, 1.09388314007182,  1.07090773695340, 2.00285254082830,
      4.36400837813955, 3.83469860546080,  3.91542517673963, 5.58571682419960,
  };
  constexpr std::array<std::int64_t, 2> offsets{0, 16};

  xtbloom::detail::gfn2::CoordinationPlan plan;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_coordination_plan(1, 16, offsets.data(), atomic_numbers.data(),
                                                      plan, error) == XTBLOOM_STATUS_SUCCESS);
  std::array<double, 16> coordination{};
  CHECK(xtbloom::detail::gfn2::evaluate_coordination_cpu(
            plan, positions.data(), coordination.data(), error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t atom = 0; atom < coordination.size(); ++atom) {
    CHECK(near(coordination[atom], expected[atom], 4.0e-13));
  }
  return 0;
}

int test_ragged_batch_and_invariance() {
  /* H2, one atom, an empty molecule, and another atom must remain independent. */
  constexpr std::array<std::int64_t, 5> offsets{0, 2, 3, 3, 4};
  constexpr std::array<std::int32_t, 4> atomic_numbers{1, 1, 1, 1};
  constexpr std::array<double, 12> positions{
      0.0, 0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4, 0.0, 0.0,
  };

  xtbloom::detail::gfn2::CoordinationPlan plan;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_coordination_plan(4, 4, offsets.data(), atomic_numbers.data(),
                                                      plan, error) == XTBLOOM_STATUS_SUCCESS);
  std::array<double, 4> coordination{};
  CHECK(xtbloom::detail::gfn2::evaluate_coordination_cpu(
            plan, positions.data(), coordination.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(near(coordination[0], 0.8202925308871246, 2.0e-15));
  CHECK(near(coordination[1], 0.8202925308871246, 2.0e-15));
  CHECK(coordination[2] == 0.0);
  CHECK(coordination[3] == 0.0);

  std::array<double, 12> transformed{};
  for (std::size_t atom = 0; atom < atomic_numbers.size(); ++atom) {
    const double x = positions[atom * 3];
    const double y = positions[atom * 3 + 1];
    const double z = positions[atom * 3 + 2];
    transformed[atom * 3] = -y + 3.25;
    transformed[atom * 3 + 1] = x - 1.75;
    transformed[atom * 3 + 2] = z + 0.5;
  }
  std::array<double, 4> transformed_coordination{};
  CHECK(xtbloom::detail::gfn2::evaluate_coordination_cpu(plan, transformed.data(),
                                                         transformed_coordination.data(),
                                                         error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t atom = 0; atom < coordination.size(); ++atom) {
    CHECK(near(transformed_coordination[atom], coordination[atom], 2.0e-15));
  }
  return 0;
}

bool weighted_coordination(const xtbloom::detail::gfn2::CoordinationPlan& plan,
                           const std::vector<double>& positions, const std::vector<double>& weights,
                           double& value, std::string& error) {
  std::vector<double> coordination(weights.size());
  if (xtbloom::detail::gfn2::evaluate_coordination_cpu(plan, positions.data(), coordination.data(),
                                                       error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  value = 0.0;
  for (std::size_t atom = 0; atom < weights.size(); ++atom) {
    value += weights[atom] * coordination[atom];
  }
  return true;
}

int test_analytic_gradient() {
  constexpr std::array<std::int64_t, 3> offsets{0, 3, 6};
  constexpr std::array<std::int32_t, 6> atomic_numbers{8, 1, 1, 6, 8, 8};
  std::vector<double> positions{
      0.1, -0.2, 0.3, 1.7, 0.1,  -0.1, -0.6, 1.5,  0.2,
      4.0, -1.0, 0.5, 1.8, -0.8, 0.4,  6.3,  -1.4, 0.8,
  };
  const std::vector<double> weights{0.3, -0.7, 1.1, -0.2, 0.8, 0.45};

  xtbloom::detail::gfn2::CoordinationPlan plan;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_coordination_plan(2, 6, offsets.data(), atomic_numbers.data(),
                                                      plan, error) == XTBLOOM_STATUS_SUCCESS);

  constexpr double baseline = 0.125;
  std::vector<double> gradients(positions.size(), baseline);
  CHECK(xtbloom::detail::gfn2::add_coordination_gradient_cpu(plan, positions.data(), weights.data(),
                                                             gradients.data(),
                                                             error) == XTBLOOM_STATUS_SUCCESS);

  constexpr double step = 1.0e-5;
  for (std::size_t coordinate = 0; coordinate < positions.size(); ++coordinate) {
    positions[coordinate] += step;
    double right = 0.0;
    CHECK(weighted_coordination(plan, positions, weights, right, error));
    positions[coordinate] -= 2.0 * step;
    double left = 0.0;
    CHECK(weighted_coordination(plan, positions, weights, left, error));
    positions[coordinate] += step;
    const double numerical_gradient = (right - left) / (2.0 * step);
    CHECK(near(gradients[coordinate] - baseline, numerical_gradient, 3.0e-9));
  }

  /* Translational invariance requires zero net gradient for each molecule. */
  for (std::size_t batch = 0; batch < 2; ++batch) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      double sum = 0.0;
      for (std::int64_t atom = offsets[batch]; atom < offsets[batch + 1]; ++atom) {
        sum += gradients[static_cast<std::size_t>(atom) * 3 + axis] - baseline;
      }
      CHECK(near(sum, 0.0, 3.0e-15));
    }
  }
  return 0;
}

int test_distance_boundaries() {
  constexpr std::array<std::int64_t, 2> offsets{0, 2};
  constexpr std::array<std::int32_t, 2> atomic_numbers{1, 8};
  xtbloom::detail::gfn2::CoordinationPlan plan;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_coordination_plan(1, 2, offsets.data(), atomic_numbers.data(),
                                                      plan, error) == XTBLOOM_STATUS_SUCCESS);

  std::array<double, 6> positions{};
  std::array<double, 2> coordination{};
  std::array<double, 2> weights{1.0, -0.5};
  std::array<double, 6> gradients{};
  CHECK(xtbloom::detail::gfn2::evaluate_coordination_cpu(
            plan, positions.data(), coordination.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::add_coordination_gradient_cpu(plan, positions.data(), weights.data(),
                                                             gradients.data(), error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);

  positions[3] = 5.0e-7;
  CHECK(xtbloom::detail::gfn2::evaluate_coordination_cpu(
            plan, positions.data(), coordination.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  /* Upstream includes a pair exactly at cutoff and excludes only r > cutoff. */
  positions[3] = 25.0;
  CHECK(xtbloom::detail::gfn2::evaluate_coordination_cpu(
            plan, positions.data(), coordination.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(coordination[0] > 0.0);
  CHECK(coordination[1] == coordination[0]);

  positions[3] = std::nextafter(25.0, std::numeric_limits<double>::infinity());
  CHECK(xtbloom::detail::gfn2::evaluate_coordination_cpu(
            plan, positions.data(), coordination.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(coordination[0] == 0.0);
  CHECK(coordination[1] == 0.0);
  return 0;
}

int test_validation() {
  xtbloom::detail::gfn2::CoordinationPlan plan;
  std::string error;
  constexpr std::array<std::int64_t, 2> offsets{0, 2};
  constexpr std::array<std::int64_t, 2> bad_start{1, 2};
  constexpr std::array<std::int64_t, 3> descending{0, 2, 1};
  constexpr std::array<std::int32_t, 2> atoms{1, 8};
  constexpr std::array<std::int32_t, 2> bad_atoms{1, 87};
  CHECK(xtbloom::detail::gfn2::make_coordination_plan(1, 2, nullptr, atoms.data(), plan, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::make_coordination_plan(1, 2, bad_start.data(), atoms.data(), plan,
                                                      error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::make_coordination_plan(2, 1, descending.data(), atoms.data(), plan,
                                                      error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::make_coordination_plan(1, 2, offsets.data(), bad_atoms.data(), plan,
                                                      error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::make_coordination_plan(1, 2, offsets.data(), atoms.data(), plan,
                                                      error) == XTBLOOM_STATUS_SUCCESS);

  std::array<double, 6> positions{0.0, 0.0, 0.0, 1.8, 0.0, 0.0};
  std::array<double, 2> coordination{};
  std::array<double, 2> weights{1.0, 1.0};
  std::array<double, 6> gradients{};
  CHECK(xtbloom::detail::gfn2::evaluate_coordination_cpu(plan, nullptr, coordination.data(),
                                                         error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::evaluate_coordination_cpu(plan, positions.data(), nullptr, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  positions[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK(xtbloom::detail::gfn2::evaluate_coordination_cpu(
            plan, positions.data(), coordination.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  positions[0] = 0.0;
  CHECK(xtbloom::detail::gfn2::add_coordination_gradient_cpu(plan, positions.data(), nullptr,
                                                             gradients.data(), error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  weights[1] = std::numeric_limits<double>::infinity();
  CHECK(xtbloom::detail::gfn2::add_coordination_gradient_cpu(plan, positions.data(), weights.data(),
                                                             gradients.data(), error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);

  std::vector<std::int64_t> all_offsets{0, 86};
  std::vector<std::int32_t> all_elements(86);
  for (std::int32_t atomic_number = 1; atomic_number <= 86; ++atomic_number) {
    all_elements[static_cast<std::size_t>(atomic_number - 1)] = atomic_number;
  }
  CHECK(xtbloom::detail::gfn2::make_coordination_plan(
            1, 86, all_offsets.data(), all_elements.data(), plan, error) == XTBLOOM_STATUS_SUCCESS);

  plan.atom_offsets.back() = 85;
  std::vector<double> all_positions(86 * 3, 0.0);
  std::vector<double> all_coordination(86);
  CHECK(xtbloom::detail::gfn2::evaluate_coordination_cpu(plan, all_positions.data(),
                                                         all_coordination.data(),
                                                         error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_mctc_reference(); status != 0) {
    return status;
  }
  if (const int status = test_ragged_batch_and_invariance(); status != 0) {
    return status;
  }
  if (const int status = test_analytic_gradient(); status != 0) {
    return status;
  }
  if (const int status = test_distance_boundaries(); status != 0) {
    return status;
  }
  return test_validation();
}
