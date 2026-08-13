#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "model/gfn1/repulsion.hpp"

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

int test_tblite_mindless01_golden() {
  /*
   * Structure and energy are tblite's LGPL-3.0-or-later GFN1 effective-
   * repulsion unit fixture at revision 133f91e. The source golden is
   * 0.16777923624986593 Eh; the positions are mstore's Apache-2.0 MB16-43 01.
   */
  constexpr std::array<std::int32_t, 16> atomic_numbers{
      11, 1, 8, 1, 9, 1, 1, 8, 7, 1, 1, 17, 5, 5, 7, 13,
  };
  std::vector<double> positions{
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
  constexpr std::array<std::int64_t, 2> offsets{0, 16};

  xtbloom::detail::gfn1::RepulsionPlan plan;
  std::string error;
  CHECK(xtbloom::detail::gfn1::make_repulsion_plan(1, 16, offsets.data(), atomic_numbers.data(),
                                                   plan, error) == XTBLOOM_STATUS_SUCCESS);
  std::array<double, 1> energy{};
  std::vector<double> forces(positions.size());
  CHECK(xtbloom::detail::gfn1::add_repulsion_cpu(plan, positions.data(), energy.data(),
                                                 forces.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(near(energy[0], 0.16777923624986593, 3.0e-16));

  constexpr std::array<double, 3> steps{1.0e-4, 2.0e-5, 4.0e-6};
  constexpr std::array<double, 3> tolerances{5.0e-9, 5.0e-10, 2.0e-9};
  for (std::size_t step_index = 0; step_index < steps.size(); ++step_index) {
    const double step = steps[step_index];
    for (std::size_t coordinate = 0; coordinate < positions.size(); ++coordinate) {
      positions[coordinate] += step;
      std::array<double, 1> right{};
      CHECK(xtbloom::detail::gfn1::add_repulsion_cpu(plan, positions.data(), right.data(), nullptr,
                                                     error) == XTBLOOM_STATUS_SUCCESS);
      positions[coordinate] -= 2.0 * step;
      std::array<double, 1> left{};
      CHECK(xtbloom::detail::gfn1::add_repulsion_cpu(plan, positions.data(), left.data(), nullptr,
                                                     error) == XTBLOOM_STATUS_SUCCESS);
      positions[coordinate] += step;
      const double numerical_force = -(right[0] - left[0]) / (2.0 * step);
      CHECK(near(forces[coordinate], numerical_force, tolerances[step_index]));
    }
  }

  std::array<double, 3> net_force{};
  std::array<double, 3> torque{};
  for (std::size_t atom = 0; atom < atomic_numbers.size(); ++atom) {
    const double fx = forces[atom * 3u];
    const double fy = forces[atom * 3u + 1u];
    const double fz = forces[atom * 3u + 2u];
    net_force[0] += fx;
    net_force[1] += fy;
    net_force[2] += fz;
    const double x = positions[atom * 3u];
    const double y = positions[atom * 3u + 1u];
    const double z = positions[atom * 3u + 2u];
    torque[0] += y * fz - z * fy;
    torque[1] += z * fx - x * fz;
    torque[2] += x * fy - y * fx;
  }
  for (std::size_t axis = 0; axis < 3; ++axis) {
    CHECK(near(net_force[axis], 0.0, 2.0e-16));
    CHECK(near(torque[axis], 0.0, 2.0e-15));
  }
  return 0;
}

int test_independent_pairs_ragged_accumulation_and_covariance() {
  constexpr std::array<std::int64_t, 5> offsets{0, 2, 4, 5, 5};
  constexpr std::array<std::int32_t, 5> atomic_numbers{1, 1, 6, 8, 86};
  constexpr std::array<double, 15> positions{
      0.0, 0.0, 0.0, 1.4, 0.0, 0.0, -0.4, 0.2, 0.1, 1.9, 0.2, 0.1, 5.0, -2.0, 1.0,
  };

  xtbloom::detail::gfn1::RepulsionPlan plan;
  std::string error;
  CHECK(xtbloom::detail::gfn1::make_repulsion_plan(4, 5, offsets.data(), atomic_numbers.data(),
                                                   plan, error) == XTBLOOM_STATUS_SUCCESS);
  std::array<double, 4> energies{0.25, 0.25, 0.25, 0.25};
  std::array<double, 15> forces{};
  CHECK(xtbloom::detail::gfn1::add_repulsion_cpu(plan, positions.data(), energies.data(),
                                                 forces.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(near(energies[0] - 0.25, 0.022893402746661507, 3.0e-17));
  CHECK(near(energies[1] - 0.25, 0.037151145741047484, 4.0e-17));
  CHECK(energies[2] == 0.25 && energies[3] == 0.25);
  CHECK(near(forces[0], -0.10613642871993496, 2.0e-16));
  CHECK(near(forces[3], 0.10613642871993496, 2.0e-16));
  CHECK(near(forces[6], -0.15162165837073494, 2.0e-16));
  CHECK(near(forces[9], 0.15162165837073494, 2.0e-16));

  std::array<double, 15> transformed{};
  for (std::size_t atom = 0; atom < atomic_numbers.size(); ++atom) {
    const double x = positions[atom * 3u];
    const double y = positions[atom * 3u + 1u];
    const double z = positions[atom * 3u + 2u];
    transformed[atom * 3u] = -y + 3.25;
    transformed[atom * 3u + 1u] = x - 1.75;
    transformed[atom * 3u + 2u] = z + 0.5;
  }
  std::array<double, 4> transformed_energies{};
  std::array<double, 15> transformed_forces{};
  CHECK(xtbloom::detail::gfn1::add_repulsion_cpu(
            plan, transformed.data(), transformed_energies.data(), transformed_forces.data(),
            error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t batch = 0; batch < transformed_energies.size(); ++batch) {
    CHECK(near(transformed_energies[batch], energies[batch] - 0.25, 5.0e-17));
  }
  for (std::size_t atom = 0; atom < atomic_numbers.size(); ++atom) {
    CHECK(near(transformed_forces[atom * 3u], -forces[atom * 3u + 1u], 3.0e-16));
    CHECK(near(transformed_forces[atom * 3u + 1u], forces[atom * 3u], 3.0e-16));
    CHECK(near(transformed_forces[atom * 3u + 2u], forces[atom * 3u + 2u], 3.0e-16));
  }
  return 0;
}

int test_distance_boundaries_and_transactional_failures() {
  constexpr std::array<std::int64_t, 2> offsets{0, 2};
  constexpr std::array<std::int32_t, 2> atomic_numbers{1, 8};
  xtbloom::detail::gfn1::RepulsionPlan plan;
  std::string error;
  CHECK(xtbloom::detail::gfn1::make_repulsion_plan(1, 2, offsets.data(), atomic_numbers.data(),
                                                   plan, error) == XTBLOOM_STATUS_SUCCESS);

  std::array<double, 6> positions{};
  std::array<double, 1> energy{7.0};
  std::array<double, 6> forces{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  const auto original_energy = energy;
  const auto original_forces = forces;
  CHECK(xtbloom::detail::gfn1::add_repulsion_cpu(plan, positions.data(), energy.data(),
                                                 forces.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(energy == original_energy && forces == original_forces);

  /* Effective repulsion shares tblite's strict r^2 < 1e-12 skip. */
  positions[3] = 1.0e-6;
  energy = {0.0};
  forces.fill(0.0);
  CHECK(xtbloom::detail::gfn1::add_repulsion_cpu(plan, positions.data(), energy.data(),
                                                 forces.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::isfinite(energy[0]) && energy[0] > 0.0);
  for (double force : forces) {
    CHECK(std::isfinite(force));
  }
  positions[3] = std::nextafter(1.0e-6, 0.0);
  energy = original_energy;
  forces = original_forces;
  CHECK(xtbloom::detail::gfn1::add_repulsion_cpu(plan, positions.data(), energy.data(),
                                                 forces.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(energy == original_energy && forces == original_forces);

  positions[3] = 25.0;
  energy = {0.0};
  CHECK(xtbloom::detail::gfn1::add_repulsion_cpu(plan, positions.data(), energy.data(), nullptr,
                                                 error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(energy[0] > 0.0);
  positions[3] = std::nextafter(25.0, std::numeric_limits<double>::infinity());
  energy = {0.0};
  CHECK(xtbloom::detail::gfn1::add_repulsion_cpu(plan, positions.data(), energy.data(), nullptr,
                                                 error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(energy[0] == 0.0);

  positions[0] = std::numeric_limits<double>::infinity();
  energy = original_energy;
  forces = original_forces;
  CHECK(xtbloom::detail::gfn1::add_repulsion_cpu(plan, positions.data(), energy.data(),
                                                 forces.data(),
                                                 error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energy == original_energy && forces == original_forces);
  return 0;
}

int test_plan_validation_and_element_boundaries() {
  xtbloom::detail::gfn1::RepulsionPlan plan;
  std::string error;
  constexpr std::array<std::int64_t, 2> offsets{0, 2};
  constexpr std::array<std::int64_t, 2> bad_start{1, 2};
  constexpr std::array<std::int32_t, 2> atoms{1, 86};
  constexpr std::array<std::int32_t, 2> low_atom{0, 1};
  constexpr std::array<std::int32_t, 2> high_atom{1, 87};
  CHECK(xtbloom::detail::gfn1::make_repulsion_plan(1, 2, bad_start.data(), atoms.data(), plan,
                                                   error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn1::make_repulsion_plan(1, 2, offsets.data(), low_atom.data(), plan,
                                                   error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn1::make_repulsion_plan(1, 2, offsets.data(), high_atom.data(), plan,
                                                   error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  std::array<double, 6> two_atom_positions{};
  std::array<double, 1> two_atom_energy{23.0};
  CHECK(xtbloom::detail::gfn1::make_repulsion_plan(1, 2, offsets.data(), atoms.data(), plan,
                                                   error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn1::add_repulsion_cpu(plan, nullptr, two_atom_energy.data(), nullptr,
                                                 error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(two_atom_energy[0] == 23.0);
  CHECK(xtbloom::detail::gfn1::add_repulsion_cpu(plan, two_atom_positions.data(), nullptr, nullptr,
                                                 error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn1::add_repulsion_cpu(plan, two_atom_positions.data(),
                                                 two_atom_positions.data(), nullptr,
                                                 error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn1::add_repulsion_cpu(plan, two_atom_positions.data(),
                                                 plan.sqrt_alpha.data(), nullptr,
                                                 error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  std::vector<std::int64_t> all_offsets{0, 86};
  std::vector<std::int32_t> all_elements(86);
  for (std::int32_t atomic_number = 1; atomic_number <= 86; ++atomic_number) {
    all_elements[static_cast<std::size_t>(atomic_number - 1)] = atomic_number;
  }
  CHECK(xtbloom::detail::gfn1::make_repulsion_plan(1, 86, all_offsets.data(), all_elements.data(),
                                                   plan, error) == XTBLOOM_STATUS_SUCCESS);
  plan.effective_charge[0] = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> positions(86u * 3u, 1.0);
  std::array<double, 1> energy{5.0};
  CHECK(xtbloom::detail::gfn1::add_repulsion_cpu(plan, positions.data(), energy.data(), nullptr,
                                                 error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energy[0] == 5.0);
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_tblite_mindless01_golden(); status != 0) {
    return status;
  }
  if (const int status = test_independent_pairs_ragged_accumulation_and_covariance(); status != 0) {
    return status;
  }
  if (const int status = test_distance_boundaries_and_transactional_failures(); status != 0) {
    return status;
  }
  return test_plan_validation_and_element_boundaries();
}
