#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "model/gfn1/coordination.hpp"

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

int test_independent_mindless01_fixture() {
  /*
   * Coordinates are mstore's Apache-2.0 MB16-43 structure 01. The expected
   * values were evaluated independently in 80-digit decimal arithmetic from
   * the pinned mctc-lib v0.5.2 exp-count equation, its reviewed GFN1 radii,
   * k=16, and the inclusive 25-bohr cutoff. No maximum-CN transform is used.
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
      4.1506636895139719, 0.9788680263897811, 2.0108098563385948, 1.4786569782781824,
      1.0357782244211671, 1.0120699431478062, 1.5032977712740092, 1.9985846827260887,
      3.8918192753932410, 1.0432337336073976, 1.0152658445063574, 1.9931521322735404,
      4.6352656088968285, 3.8731226063933475, 3.9931680067788431, 5.4506822690388841,
  };
  constexpr std::array<std::int64_t, 2> offsets{0, 16};

  xtbloom::detail::gfn1::CoordinationPlan plan;
  std::string error;
  CHECK(xtbloom::detail::gfn1::make_coordination_plan(1, 16, offsets.data(), atomic_numbers.data(),
                                                      plan, error) == XTBLOOM_STATUS_SUCCESS);
  std::array<double, 16> actual{};
  CHECK(xtbloom::detail::gfn1::evaluate_coordination_cpu(plan, positions.data(), actual.data(),
                                                         error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t atom = 0; atom < actual.size(); ++atom) {
    CHECK(near(actual[atom], expected[atom], 3.0e-14));
  }
  return 0;
}

int test_pair_values_ragged_batch_and_invariance() {
  /* H-H and C-O exercise two independently calculated pair counts. */
  constexpr std::array<std::int64_t, 5> offsets{0, 2, 4, 5, 5};
  constexpr std::array<std::int32_t, 5> atomic_numbers{1, 1, 6, 8, 86};
  constexpr std::array<double, 15> positions{
      0.0, 0.0, 0.0, 1.4, 0.0, 0.0, -0.4, 0.2, 0.1, 1.9, 0.2, 0.1, 5.0, -2.0, 1.0,
  };

  xtbloom::detail::gfn1::CoordinationPlan plan;
  std::string error;
  CHECK(xtbloom::detail::gfn1::make_coordination_plan(4, 5, offsets.data(), atomic_numbers.data(),
                                                      plan, error) == XTBLOOM_STATUS_SUCCESS);
  std::array<double, 5> coordination{};
  CHECK(xtbloom::detail::gfn1::evaluate_coordination_cpu(
            plan, positions.data(), coordination.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(near(coordination[0], 0.9190366235286143, 3.0e-16));
  CHECK(coordination[1] == coordination[0]);
  CHECK(near(coordination[2], 0.9997222452945929, 3.0e-16));
  CHECK(coordination[3] == coordination[2]);
  CHECK(coordination[4] == 0.0);

  std::array<double, 15> transformed{};
  for (std::size_t atom = 0; atom < atomic_numbers.size(); ++atom) {
    const double x = positions[atom * 3u];
    const double y = positions[atom * 3u + 1u];
    const double z = positions[atom * 3u + 2u];
    transformed[atom * 3u] = -y + 3.25;
    transformed[atom * 3u + 1u] = x - 1.75;
    transformed[atom * 3u + 2u] = z + 0.5;
  }
  std::array<double, 5> transformed_coordination{};
  CHECK(xtbloom::detail::gfn1::evaluate_coordination_cpu(plan, transformed.data(),
                                                         transformed_coordination.data(),
                                                         error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t atom = 0; atom < coordination.size(); ++atom) {
    CHECK(near(transformed_coordination[atom], coordination[atom], 4.0e-16));
  }
  return 0;
}

bool weighted_coordination(const xtbloom::detail::gfn1::CoordinationPlan& plan,
                           const std::vector<double>& positions, const std::vector<double>& weights,
                           double& value, std::string& error) {
  std::vector<double> coordination(weights.size());
  if (xtbloom::detail::gfn1::evaluate_coordination_cpu(plan, positions.data(), coordination.data(),
                                                       error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  value = 0.0;
  for (std::size_t atom = 0; atom < weights.size(); ++atom) {
    value += weights[atom] * coordination[atom];
  }
  return true;
}

int test_analytic_vjp_multiple_steps_and_conservation() {
  constexpr std::array<std::int64_t, 3> offsets{0, 3, 6};
  constexpr std::array<std::int32_t, 6> atomic_numbers{8, 1, 1, 6, 8, 86};
  std::vector<double> positions{
      0.1, -0.2, 0.3, 1.7, 0.1,  -0.1, -0.6, 1.5,  0.2,
      4.0, -1.0, 0.5, 1.8, -0.8, 0.4,  6.3,  -1.4, 0.8,
  };
  const std::vector<double> weights{0.3, -0.7, 1.1, -0.2, 0.8, 0.45};

  xtbloom::detail::gfn1::CoordinationPlan plan;
  std::string error;
  CHECK(xtbloom::detail::gfn1::make_coordination_plan(2, 6, offsets.data(), atomic_numbers.data(),
                                                      plan, error) == XTBLOOM_STATUS_SUCCESS);

  constexpr double baseline = 0.125;
  std::vector<double> gradients(positions.size(), baseline);
  CHECK(xtbloom::detail::gfn1::add_coordination_gradient_cpu(plan, positions.data(), weights.data(),
                                                             gradients.data(),
                                                             error) == XTBLOOM_STATUS_SUCCESS);

  constexpr std::array<double, 3> steps{1.0e-4, 2.0e-5, 4.0e-6};
  constexpr std::array<double, 3> tolerances{2.0e-7, 9.0e-9, 7.0e-9};
  for (std::size_t step_index = 0; step_index < steps.size(); ++step_index) {
    const double step = steps[step_index];
    for (std::size_t coordinate = 0; coordinate < positions.size(); ++coordinate) {
      positions[coordinate] += step;
      double right = 0.0;
      CHECK(weighted_coordination(plan, positions, weights, right, error));
      positions[coordinate] -= 2.0 * step;
      double left = 0.0;
      CHECK(weighted_coordination(plan, positions, weights, left, error));
      positions[coordinate] += step;
      const double numerical_gradient = (right - left) / (2.0 * step);
      CHECK(near(gradients[coordinate] - baseline, numerical_gradient, tolerances[step_index]));
    }
  }

  for (std::size_t batch = 0; batch < 2; ++batch) {
    std::array<double, 3> net_gradient{};
    std::array<double, 3> torque{};
    for (std::int64_t atom = offsets[batch]; atom < offsets[batch + 1]; ++atom) {
      const auto index = static_cast<std::size_t>(atom);
      const double gx = gradients[index * 3u] - baseline;
      const double gy = gradients[index * 3u + 1u] - baseline;
      const double gz = gradients[index * 3u + 2u] - baseline;
      net_gradient[0] += gx;
      net_gradient[1] += gy;
      net_gradient[2] += gz;
      const double x = positions[index * 3u];
      const double y = positions[index * 3u + 1u];
      const double z = positions[index * 3u + 2u];
      torque[0] += y * gz - z * gy;
      torque[1] += z * gx - x * gz;
      torque[2] += x * gy - y * gx;
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
      CHECK(near(net_gradient[axis], 0.0, 3.0e-15));
      CHECK(near(torque[axis], 0.0, 8.0e-15));
    }
  }
  return 0;
}

int test_distance_boundaries_and_transactional_failures() {
  constexpr std::array<std::int64_t, 2> offsets{0, 2};
  constexpr std::array<std::int32_t, 2> atomic_numbers{1, 8};
  xtbloom::detail::gfn1::CoordinationPlan plan;
  std::string error;
  CHECK(xtbloom::detail::gfn1::make_coordination_plan(1, 2, offsets.data(), atomic_numbers.data(),
                                                      plan, error) == XTBLOOM_STATUS_SUCCESS);

  std::array<double, 6> positions{};
  std::array<double, 2> coordination{7.0, 8.0};
  std::array<double, 2> weights{1.0, -0.5};
  std::array<double, 6> gradients{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  const auto original_coordination = coordination;
  const auto original_gradients = gradients;
  CHECK(xtbloom::detail::gfn1::evaluate_coordination_cpu(
            plan, positions.data(), coordination.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(coordination[0] == 0.0 && coordination[1] == 0.0);
  CHECK(xtbloom::detail::gfn1::add_coordination_gradient_cpu(plan, positions.data(), weights.data(),
                                                             gradients.data(),
                                                             error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(gradients == original_gradients);

  /* The upstream skip is strict: equality at r^2=1e-12 is evaluated. */
  positions[3] = 1.0e-6;
  CHECK(xtbloom::detail::gfn1::evaluate_coordination_cpu(
            plan, positions.data(), coordination.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(coordination[0] == 1.0 && coordination[1] == 1.0);
  positions[3] = std::nextafter(1.0e-6, 0.0);
  coordination = original_coordination;
  CHECK(xtbloom::detail::gfn1::evaluate_coordination_cpu(
            plan, positions.data(), coordination.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(coordination[0] == 0.0 && coordination[1] == 0.0);

  positions[3] = 25.0;
  CHECK(xtbloom::detail::gfn1::evaluate_coordination_cpu(
            plan, positions.data(), coordination.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(coordination[0] > 0.0 && coordination[1] == coordination[0]);
  positions[3] = std::nextafter(25.0, std::numeric_limits<double>::infinity());
  CHECK(xtbloom::detail::gfn1::evaluate_coordination_cpu(
            plan, positions.data(), coordination.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(coordination[0] == 0.0 && coordination[1] == 0.0);

  positions[0] = std::numeric_limits<double>::quiet_NaN();
  coordination = original_coordination;
  CHECK(xtbloom::detail::gfn1::evaluate_coordination_cpu(
            plan, positions.data(), coordination.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(coordination == original_coordination);
  return 0;
}

int test_plan_validation_and_element_boundaries() {
  xtbloom::detail::gfn1::CoordinationPlan plan;
  std::string error;
  constexpr std::array<std::int64_t, 2> offsets{0, 2};
  constexpr std::array<std::int64_t, 2> bad_start{1, 2};
  constexpr std::array<std::int64_t, 3> descending{0, 2, 1};
  constexpr std::array<std::int32_t, 2> atoms{1, 86};
  constexpr std::array<std::int32_t, 2> low_atom{0, 1};
  constexpr std::array<std::int32_t, 2> high_atom{1, 87};
  CHECK(xtbloom::detail::gfn1::make_coordination_plan(1, 2, nullptr, atoms.data(), plan, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn1::make_coordination_plan(1, 2, bad_start.data(), atoms.data(), plan,
                                                      error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn1::make_coordination_plan(2, 1, descending.data(), atoms.data(), plan,
                                                      error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn1::make_coordination_plan(1, 2, offsets.data(), low_atom.data(), plan,
                                                      error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn1::make_coordination_plan(1, 2, offsets.data(), high_atom.data(), plan,
                                                      error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  std::vector<std::int64_t> all_offsets{0, 86};
  std::vector<std::int32_t> all_elements(86);
  for (std::int32_t atomic_number = 1; atomic_number <= 86; ++atomic_number) {
    all_elements[static_cast<std::size_t>(atomic_number - 1)] = atomic_number;
  }
  CHECK(xtbloom::detail::gfn1::make_coordination_plan(
            1, 86, all_offsets.data(), all_elements.data(), plan, error) == XTBLOOM_STATUS_SUCCESS);
  plan.atom_offsets.back() = 85;
  std::vector<double> positions(86u * 3u, 0.0);
  std::vector<double> coordination(86u, 9.0);
  CHECK(xtbloom::detail::gfn1::evaluate_coordination_cpu(
            plan, positions.data(), coordination.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  for (double value : coordination) {
    CHECK(value == 9.0);
  }
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_independent_mindless01_fixture(); status != 0) {
    return status;
  }
  if (const int status = test_pair_values_ragged_batch_and_invariance(); status != 0) {
    return status;
  }
  if (const int status = test_analytic_vjp_multiple_steps_and_conservation(); status != 0) {
    return status;
  }
  if (const int status = test_distance_boundaries_and_transactional_failures(); status != 0) {
    return status;
  }
  return test_plan_validation_and_element_boundaries();
}
