#include "model/gfn2/basis.hpp"

#include <algorithm>
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

bool near(double actual, double expected, double tolerance = 2.0e-14) {
  return std::abs(actual - expected) <= tolerance;
}

template <typename Actual, typename Expected>
bool equal_sequence(const Actual& actual, const Expected& expected) {
  return actual.size() == expected.size() &&
         std::equal(actual.begin(), actual.end(), expected.begin());
}

int test_hco_ragged_layout_and_primitives() {
  /*
   * H, C, and O cover STO-3G s plus STO-4G s/p shells. The primitive
   * goldens are the normalized output of tblite_basis_slater at the tblite
   * revision pinned in data/parameters/manifest.json. They were independently
   * cross-checked against the [GTO] section emitted by xtb 6.7.1 --molden and
   * are not values recomputed from xtbloom's implementation.
   */
  constexpr std::array<std::int64_t, 4> atom_offsets{0, 1, 1, 3};
  constexpr std::array<std::int32_t, 3> atomic_numbers{1, 6, 8};
  xtbloom::detail::gfn2::BasisPlan plan;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_basis_plan(3, 3, atom_offsets.data(), atomic_numbers.data(),
                                               plan, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(error.empty());
  CHECK(plan.batch_size == 3);
  CHECK(plan.total_atoms == 3);
  CHECK(plan.total_shells == 5);
  CHECK(plan.total_orbitals == 9);
  CHECK(plan.total_cartesian_orbitals == 9);
  CHECK(plan.total_primitives == 19);
  CHECK(plan.maximum_angular_momentum == 1u);

  constexpr std::array<std::int64_t, 4> expected_atom_shell{0, 1, 3, 5};
  constexpr std::array<std::int64_t, 4> expected_atom_orbital{0, 1, 5, 9};
  constexpr std::array<std::int64_t, 4> expected_atom_primitive{0, 3, 11, 19};
  constexpr std::array<std::int64_t, 4> expected_batch_shell{0, 1, 1, 5};
  constexpr std::array<std::int64_t, 4> expected_batch_orbital{0, 1, 1, 9};
  constexpr std::array<std::int64_t, 4> expected_batch_primitive{0, 3, 3, 19};
  constexpr std::array<std::int64_t, 6> expected_shell_orbital{0, 1, 2, 5, 6, 9};
  constexpr std::array<std::int64_t, 6> expected_shell_primitive{0, 3, 7, 11, 15, 19};
  constexpr std::array<std::int64_t, 5> expected_shell_to_atom{0, 1, 1, 2, 2};
  constexpr std::array<std::uint8_t, 5> expected_n{1, 2, 2, 2, 2};
  constexpr std::array<std::uint8_t, 5> expected_l{0, 0, 1, 0, 1};
  CHECK(equal_sequence(plan.atom_offsets, atom_offsets));
  CHECK(equal_sequence(plan.atom_shell_offsets, expected_atom_shell));
  CHECK(equal_sequence(plan.atom_orbital_offsets, expected_atom_orbital));
  CHECK(equal_sequence(plan.atom_cartesian_orbital_offsets, expected_atom_orbital));
  CHECK(equal_sequence(plan.atom_primitive_offsets, expected_atom_primitive));
  CHECK(equal_sequence(plan.batch_shell_offsets, expected_batch_shell));
  CHECK(equal_sequence(plan.batch_orbital_offsets, expected_batch_orbital));
  CHECK(equal_sequence(plan.batch_cartesian_orbital_offsets, expected_batch_orbital));
  CHECK(equal_sequence(plan.batch_primitive_offsets, expected_batch_primitive));
  CHECK(equal_sequence(plan.shell_orbital_offsets, expected_shell_orbital));
  CHECK(equal_sequence(plan.shell_cartesian_orbital_offsets, expected_shell_orbital));
  CHECK(equal_sequence(plan.shell_primitive_offsets, expected_shell_primitive));
  CHECK(equal_sequence(plan.shell_to_atom, expected_shell_to_atom));
  CHECK(equal_sequence(plan.principal_quantum_numbers, expected_n));
  CHECK(equal_sequence(plan.angular_momenta, expected_l));

  constexpr std::array<double, 5> expected_slater{1.23, 2.096432, 1.8, 2.439742, 2.137023};
  for (std::size_t shell = 0; shell < expected_slater.size(); ++shell) {
    CHECK(plan.slater_exponents[shell] == expected_slater[shell]);
  }

  constexpr std::array<double, 19> expected_alpha{
      3.3702276975336001, 0.61389118221497996, 0.16614291148415999, 51.049363095579913,
      8.7911227406887544, 0.70640422258929825, 0.26922813413411617, 5.8263656140799993,
      1.510689601872,     0.53256483288000001, 0.21202323690600003, 69.137961906196551,
      11.906129132707308, 0.95670827744340703, 0.36462520496073686, 8.2124193257781641,
      2.129357699709812,  0.75066448202360914, 0.29885246543283067,
  };
  constexpr std::array<double, 19> expected_coeff{
      0.27359110588908298,  0.26460540653860526, 0.082465947534082318, -0.16312063210127389,
      -0.1991102458100073,  0.31882047753655751, 0.1270650418424038,   0.73716451545209594,
      0.68216371614850246,  0.35783053929253711, 0.053983030275847881, -0.20478756370077797,
      -0.24997023136827334, 0.40025880240639616, 0.15952206667693555,  1.1321556799186829,
      1.0476840782254822,   0.54956508217158984, 0.082908486592766603,
  };
  CHECK(plan.primitive_exponents.size() == expected_alpha.size());
  CHECK(plan.primitive_coefficients.size() == expected_coeff.size());
  for (std::size_t primitive = 0; primitive < expected_alpha.size(); ++primitive) {
    CHECK(near(plan.primitive_exponents[primitive], expected_alpha[primitive]));
    CHECK(near(plan.primitive_coefficients[primitive], expected_coeff[primitive]));
  }
  CHECK(near(plan.minimum_primitive_exponent, expected_alpha[2]));
  return 0;
}

int test_all_gfn2_elements_and_sixth_row_expansion() {
  std::vector<std::int32_t> atomic_numbers(86);
  for (std::int32_t atomic_number = 1; atomic_number <= 86; ++atomic_number) {
    atomic_numbers[static_cast<std::size_t>(atomic_number - 1)] = atomic_number;
  }
  constexpr std::array<std::int64_t, 4> atom_offsets{0, 1, 85, 86};
  xtbloom::detail::gfn2::BasisPlan plan;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_basis_plan(3, 86, atom_offsets.data(), atomic_numbers.data(),
                                               plan, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(plan.total_shells == 237);
  CHECK(plan.total_orbitals == 671);
  CHECK(plan.total_cartesian_orbitals == 737);
  CHECK(plan.total_primitives == 1008);
  CHECK(plan.maximum_angular_momentum == 2u);
  CHECK(plan.atom_shell_offsets.back() == plan.total_shells);
  CHECK(plan.atom_orbital_offsets.back() == plan.total_orbitals);
  CHECK(plan.atom_cartesian_orbital_offsets.back() == plan.total_cartesian_orbitals);
  CHECK(plan.atom_primitive_offsets.back() == plan.total_primitives);
  CHECK(plan.batch_shell_offsets[1] == 1);
  CHECK(plan.batch_shell_offsets[2] == 234);
  CHECK(plan.batch_shell_offsets[3] == 237);

  /* Rn exercises tblite's special STO-6G 6s and 6p tables. */
  const std::size_t radon_shell = static_cast<std::size_t>(plan.atom_shell_offsets[85]);
  CHECK(plan.principal_quantum_numbers[radon_shell] == 6u);
  CHECK(plan.angular_momenta[radon_shell] == 0u);
  CHECK(plan.shell_primitive_offsets[radon_shell + 1] - plan.shell_primitive_offsets[radon_shell] ==
        6);
  const std::size_t first_primitive =
      static_cast<std::size_t>(plan.shell_primitive_offsets[radon_shell]);
  CHECK(near(plan.primitive_exponents[first_primitive], 5.800292686e-1 * 3.109394 * 3.109394));
  CHECK(near(plan.primitive_coefficients[first_primitive], 0.011828739512200757));
  CHECK(plan.principal_quantum_numbers[radon_shell + 1] == 6u);
  CHECK(plan.angular_momenta[radon_shell + 1] == 1u);
  CHECK(plan.shell_primitive_offsets[radon_shell + 2] -
            plan.shell_primitive_offsets[radon_shell + 1] ==
        6);
  return 0;
}

int test_validation_and_strong_failure_guarantee() {
  xtbloom::detail::gfn2::BasisPlan plan;
  plan.batch_size = 17;
  std::string error;
  constexpr std::array<std::int64_t, 2> offsets{0, 1};
  constexpr std::array<std::int64_t, 2> wrong_start{1, 1};
  constexpr std::array<std::int64_t, 3> nonmonotone{0, 2, 1};
  constexpr std::array<std::int32_t, 1> hydrogen{1};
  constexpr std::array<std::int32_t, 1> unsupported_low{0};
  constexpr std::array<std::int32_t, 1> unsupported_high{87};

  CHECK(xtbloom::detail::gfn2::make_basis_plan(0, 1, offsets.data(), hydrogen.data(), plan,
                                               error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::make_basis_plan(1, 1, nullptr, hydrogen.data(), plan, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::make_basis_plan(1, 1, wrong_start.data(), hydrogen.data(), plan,
                                               error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::make_basis_plan(2, 1, nonmonotone.data(), hydrogen.data(), plan,
                                               error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::make_basis_plan(1, 1, offsets.data(), unsupported_low.data(), plan,
                                               error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::make_basis_plan(1, 1, offsets.data(), unsupported_high.data(), plan,
                                               error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(plan.batch_size == 17);

  constexpr std::array<std::int64_t, 2> huge_offsets{0, std::numeric_limits<std::int64_t>::max()};
  CHECK(xtbloom::detail::gfn2::make_basis_plan(1, std::numeric_limits<std::int64_t>::max(),
                                               huge_offsets.data(), hydrogen.data(), plan,
                                               error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(plan.batch_size == 17);

  CHECK(xtbloom::detail::gfn2::make_basis_plan(1, 1, offsets.data(), hydrogen.data(), plan,
                                               error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(plan.batch_size == 1);
  CHECK(plan.total_shells == 1);
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_hco_ragged_layout_and_primitives(); status != 0) {
    return status;
  }
  if (const int status = test_all_gfn2_elements_and_sixth_row_expansion(); status != 0) {
    return status;
  }
  return test_validation_and_strong_failure_guarantee();
}
