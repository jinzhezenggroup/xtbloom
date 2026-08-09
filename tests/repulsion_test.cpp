#include "model/gfn2/repulsion.hpp"

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

int test_diatom_batch() {
  constexpr std::array<std::int64_t, 5> offsets{0, 2, 4, 6, 7};
  constexpr std::array<std::int32_t, 7> atomic_numbers{1, 1, 6, 6, 1, 6, 8};
  std::array<double, 21> positions{
      0.0, 0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 0.0,
      0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  };

  xtbloom::detail::gfn2::RepulsionPlan plan;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_repulsion_plan(4, 7, offsets.data(), atomic_numbers.data(),
                                                   plan, error) == XTBLOOM_STATUS_SUCCESS);

  std::array<double, 4> energies{};
  std::array<double, 21> forces{};
  CHECK(xtbloom::detail::gfn2::add_repulsion_cpu(plan, positions.data(), energies.data(),
                                                 forces.data(), error) == XTBLOOM_STATUS_SUCCESS);

  CHECK(near(energies[0], 0.039349058626104098, 2.0e-15));
  CHECK(near(energies[1], 0.051655090243274783, 2.0e-15));
  CHECK(near(energies[2], 0.021256469094772152, 2.0e-15));
  CHECK(energies[3] == 0.0);
  CHECK(near(forces[0], -0.11521415046182050, 2.0e-15));
  CHECK(near(forces[3], 0.11521415046182050, 2.0e-15));
  CHECK(near(forces[6], -0.17351325255415656, 2.0e-15));
  CHECK(near(forces[9], 0.17351325255415656, 2.0e-15));
  CHECK(near(forces[12], -0.085566854041681767, 2.0e-15));
  CHECK(near(forces[15], 0.085566854041681767, 2.0e-15));
  for (std::size_t coordinate = 0; coordinate < forces.size(); coordinate += 3) {
    CHECK(near(forces[coordinate + 1], 0.0, 0.0));
    CHECK(near(forces[coordinate + 2], 0.0, 0.0));
  }

  constexpr double step = 1.0e-5;
  positions[12] += step;
  std::array<double, 4> right{};
  CHECK(xtbloom::detail::gfn2::add_repulsion_cpu(plan, positions.data(), right.data(), nullptr,
                                                 error) == XTBLOOM_STATUS_SUCCESS);
  positions[12] -= 2.0 * step;
  std::array<double, 4> left{};
  CHECK(xtbloom::detail::gfn2::add_repulsion_cpu(plan, positions.data(), left.data(), nullptr,
                                                 error) == XTBLOOM_STATUS_SUCCESS);
  const double numerical_force = -(right[2] - left[2]) / (2.0 * step);
  CHECK(near(numerical_force, forces[12], 2.0e-10));
  return 0;
}

int test_xtb_cluster_golden() {
  /*
   * Coordinates and reference energy are from xtb's LGPL-3.0-or-later
   * test/unit/test_repulsion.f90 at revision b31754b. Coordinates use bohr.
   */
  constexpr std::array<std::int32_t, 24> atomic_numbers{
      6, 7, 6, 7, 6, 6, 6, 8, 7, 6, 8, 7, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  };
  std::array<double, 72> positions{
      2.02799738646442,  0.09231312124713,  -0.14310895950963, 4.75011007621000,  0.02373496014051,
      -0.14324124033844, 6.33434307654413,  2.07098865582721,  -0.14235306905930, 8.72860718071825,
      1.38002919517619,  -0.14265542523943, 8.65318821103610,  -1.19324866489847, -0.14231527453678,
      6.23857175648671,  -2.08353643730276, -0.14218299370797, 5.63266886875962,  -4.69950321056008,
      -0.13940509630299, 3.44931709749015,  -5.48092386085491, -0.14318454855466, 7.77508917214346,
      -6.24427872938674, -0.13107140408805, 10.30229550927022, -5.39739796609292, -0.13672168520430,
      12.07410272485492, -6.91573621641911, -0.13666499342053, 10.70038521493902, -2.79078533715849,
      -0.14148379504141, 13.24597858727017, -1.76969072232377, -0.14218299370797, 7.40891694074004,
      -8.95905928176407, -0.11636933482904, 1.38702118184179,  2.05575746325296,  -0.14178615122154,
      1.34622199478497,  -0.86356704498496, 1.55590600570783,  1.34624089204623,  -0.86133716815647,
      -1.84340893849267, 5.65596919189118,  4.00172183859480,  -0.14131371969009, 14.67430918222276,
      -3.26230980007732, -0.14344911021228, 13.50897177220290, -0.60815166181684, 1.54898960808727,
      13.50780014200488, -0.60614855212345, -1.83214617078268, 5.41408424778406,  -9.49239668625902,
      -0.11022772492007, 8.31919801555568,  -9.74947502841788, 1.56539243085954,  8.31511620712388,
      -9.76854236502758, -1.79108242206824,
  };
  constexpr std::array<std::int64_t, 2> offsets{0, 24};

  xtbloom::detail::gfn2::RepulsionPlan plan;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_repulsion_plan(1, 24, offsets.data(), atomic_numbers.data(),
                                                   plan, error) == XTBLOOM_STATUS_SUCCESS);
  std::array<double, 1> energy{};
  std::array<double, 72> forces{};
  CHECK(xtbloom::detail::gfn2::add_repulsion_cpu(plan, positions.data(), energy.data(),
                                                 forces.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(near(energy[0], 0.49222837261241, 1.0e-13));

  for (std::size_t axis = 0; axis < 3; ++axis) {
    double force_sum = 0.0;
    for (std::size_t atom = 0; atom < atomic_numbers.size(); ++atom) {
      force_sum += forces[atom * 3 + axis];
    }
    CHECK(near(force_sum, 0.0, 5.0e-15));
  }

  constexpr double step = 1.0e-5;
  for (std::size_t coordinate = 0; coordinate < positions.size(); ++coordinate) {
    positions[coordinate] += step;
    std::array<double, 1> right{};
    CHECK(xtbloom::detail::gfn2::add_repulsion_cpu(plan, positions.data(), right.data(), nullptr,
                                                   error) == XTBLOOM_STATUS_SUCCESS);
    positions[coordinate] -= 2.0 * step;
    std::array<double, 1> left{};
    CHECK(xtbloom::detail::gfn2::add_repulsion_cpu(plan, positions.data(), left.data(), nullptr,
                                                   error) == XTBLOOM_STATUS_SUCCESS);
    positions[coordinate] += step;
    const double numerical_force = -(right[0] - left[0]) / (2.0 * step);
    CHECK(near(numerical_force, forces[coordinate], 2.0e-9));
  }
  return 0;
}

int test_validation() {
  xtbloom::detail::gfn2::RepulsionPlan plan;
  std::string error;
  constexpr std::array<std::int64_t, 2> offsets{0, 2};
  constexpr std::array<std::int64_t, 2> bad_offsets{1, 2};
  constexpr std::array<std::int32_t, 2> atoms{1, 1};
  constexpr std::array<std::int32_t, 2> bad_atoms{1, 87};
  CHECK(xtbloom::detail::gfn2::make_repulsion_plan(1, 2, bad_offsets.data(), atoms.data(), plan,
                                                   error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::make_repulsion_plan(1, 2, offsets.data(), bad_atoms.data(), plan,
                                                   error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::make_repulsion_plan(1, 2, offsets.data(), atoms.data(), plan,
                                                   error) == XTBLOOM_STATUS_SUCCESS);

  xtbloom::detail::gfn2::RepulsionPlan corrupt_plan = plan;
  corrupt_plan.atom_offsets[1] = 1;
  std::array<double, 6> valid_positions{0.0, 0.0, 0.0, 1.4, 0.0, 0.0};
  std::array<double, 1> corrupt_energy{};
  CHECK(xtbloom::detail::gfn2::add_repulsion_cpu(corrupt_plan, valid_positions.data(),
                                                 corrupt_energy.data(), nullptr,
                                                 error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  std::array<double, 6> positions{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::array<double, 1> energy{};
  CHECK(xtbloom::detail::gfn2::add_repulsion_cpu(plan, positions.data(), energy.data(), nullptr,
                                                 error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  positions[3] = std::numeric_limits<double>::quiet_NaN();
  CHECK(xtbloom::detail::gfn2::add_repulsion_cpu(plan, positions.data(), energy.data(), nullptr,
                                                 error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  std::vector<std::int64_t> all_offsets{0, 86};
  std::vector<std::int32_t> all_elements(86);
  for (std::int32_t atomic_number = 1; atomic_number <= 86; ++atomic_number) {
    all_elements[static_cast<std::size_t>(atomic_number - 1)] = atomic_number;
  }
  CHECK(xtbloom::detail::gfn2::make_repulsion_plan(1, 86, all_offsets.data(), all_elements.data(),
                                                   plan, error) == XTBLOOM_STATUS_SUCCESS);
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_diatom_batch(); status != 0) {
    return status;
  }
  if (const int status = test_xtb_cluster_golden(); status != 0) {
    return status;
  }
  return test_validation();
}
