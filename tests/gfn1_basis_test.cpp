#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "model/gfn1/basis.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

bool near(double actual, double expected, double tolerance = 3.0e-14) {
  return std::abs(actual - expected) <= tolerance;
}

double contracted_s_overlap(const xtbloom::detail::gfn1::BasisPlan& plan, std::size_t first_shell,
                            std::size_t second_shell) {
  constexpr double pi = 3.141592653589793238462643383279502884;
  const auto first_begin = static_cast<std::size_t>(plan.shell_primitive_offsets[first_shell]);
  const auto first_end = static_cast<std::size_t>(plan.shell_primitive_offsets[first_shell + 1u]);
  const auto second_begin = static_cast<std::size_t>(plan.shell_primitive_offsets[second_shell]);
  const auto second_end = static_cast<std::size_t>(plan.shell_primitive_offsets[second_shell + 1u]);
  double overlap = 0.0;
  for (std::size_t first = first_begin; first < first_end; ++first) {
    for (std::size_t second = second_begin; second < second_end; ++second) {
      overlap +=
          plan.primitive_coefficients[first] * plan.primitive_coefficients[second] *
          std::pow(
              std::sqrt(pi / (plan.primitive_exponents[first] + plan.primitive_exponents[second])),
              3.0);
    }
  }
  return overlap;
}

int test_complete_element_topology() {
  std::vector<std::int32_t> atomic_numbers(86);
  for (std::int32_t atomic_number = 1; atomic_number <= 86; ++atomic_number) {
    atomic_numbers[static_cast<std::size_t>(atomic_number - 1)] = atomic_number;
  }
  constexpr std::array<std::int64_t, 5> offsets{0, 1, 1, 85, 86};
  xtbloom::detail::gfn1::BasisPlan plan;
  std::string error;
  CHECK(xtbloom::detail::gfn1::make_basis_plan(4, 86, offsets.data(), atomic_numbers.data(), plan,
                                               error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(error.empty());
  CHECK(plan.total_shells == 237);
  CHECK(plan.total_orbitals == 669);
  CHECK(plan.total_cartesian_orbitals == 735);
  CHECK(plan.total_primitives == 1287);
  CHECK(plan.maximum_angular_momentum == 2u);
  CHECK(plan.shell_to_atom.size() == 237u);
  CHECK(plan.shell_is_valence.size() == 237u);
  CHECK(std::count(plan.shell_is_valence.begin(), plan.shell_is_valence.end(), 0u) == 1);
  CHECK(plan.batch_shell_offsets == std::vector<std::int64_t>({0, 2, 2, 234, 237}));
  CHECK(plan.batch_orbital_offsets == std::vector<std::int64_t>({0, 2, 2, 660, 669}));
  CHECK(plan.atom_shell_offsets.back() == plan.total_shells);
  CHECK(plan.atom_orbital_offsets.back() == plan.total_orbitals);
  CHECK(plan.atom_primitive_offsets.back() == plan.total_primitives);

  for (std::size_t atom = 0; atom < atomic_numbers.size(); ++atom) {
    for (std::int64_t shell = plan.atom_shell_offsets[atom];
         shell < plan.atom_shell_offsets[atom + 1u]; ++shell) {
      CHECK(plan.shell_to_atom[static_cast<std::size_t>(shell)] == static_cast<std::int64_t>(atom));
    }
  }

  /* Li 2s and C 2p pin ordinary STO-6G rows omitted by the GFN2-only table. */
  const std::size_t lithium_shell = static_cast<std::size_t>(plan.atom_shell_offsets[2]);
  CHECK(plan.principal_quantum_numbers[lithium_shell] == 2u);
  CHECK(plan.angular_momenta[lithium_shell] == 0u);
  CHECK(plan.shell_primitive_offsets[lithium_shell + 1u] -
            plan.shell_primitive_offsets[lithium_shell] ==
        6);
  const std::size_t lithium_primitive =
      static_cast<std::size_t>(plan.shell_primitive_offsets[lithium_shell]);
  CHECK(near(plan.primitive_exponents[lithium_primitive], 2.768496241e+1 * 0.743881 * 0.743881));

  const std::size_t carbon_p_shell = static_cast<std::size_t>(plan.atom_shell_offsets[5] + 1);
  CHECK(plan.principal_quantum_numbers[carbon_p_shell] == 2u);
  CHECK(plan.angular_momenta[carbon_p_shell] == 1u);
  const std::size_t carbon_p_primitive =
      static_cast<std::size_t>(plan.shell_primitive_offsets[carbon_p_shell]);
  CHECK(near(plan.primitive_exponents[carbon_p_primitive], 5.868285913 * 1.832096 * 1.832096));

  /* GFN1 retains legacy xTB's STO-6G 4s/4p rows. This is model-scoped:
   * common/GFN2 basis expansion continues to use tblite's newer Stewart rows. */
  const std::size_t bromine_s_shell = static_cast<std::size_t>(plan.atom_shell_offsets[34]);
  const std::size_t bromine_p_shell = bromine_s_shell + 1u;
  const std::size_t bromine_s_primitive =
      static_cast<std::size_t>(plan.shell_primitive_offsets[bromine_s_shell]);
  const std::size_t bromine_p_primitive =
      static_cast<std::size_t>(plan.shell_primitive_offsets[bromine_p_shell]);
  CHECK(plan.principal_quantum_numbers[bromine_s_shell] == 4u);
  CHECK(plan.angular_momenta[bromine_s_shell] == 0u);
  CHECK(plan.principal_quantum_numbers[bromine_p_shell] == 4u);
  CHECK(plan.angular_momenta[bromine_p_shell] == 1u);
  CHECK(plan.shell_primitive_offsets[bromine_s_shell + 1u] -
            plan.shell_primitive_offsets[bromine_s_shell] ==
        6);
  CHECK(plan.shell_primitive_offsets[bromine_p_shell + 1u] -
            plan.shell_primitive_offsets[bromine_p_shell] ==
        6);
  constexpr std::array<double, 6> legacy_alpha{
      1.365346e+00, 4.393213e-01, 1.877069e-01, 9.360270e-02, 5.052263e-02, 2.809354e-02,
  };
  constexpr std::array<double, 6> legacy_s_coefficients{
      3.775056e-03, -5.585965e-02, -3.192946e-01, -2.764780e-02, 9.049199e-01, 3.406258e-01,
  };
  constexpr std::array<double, 6> legacy_p_coefficients{
      -7.052075e-03, -5.259505e-02, -3.773450e-02, 3.874773e-01, 5.791672e-01, 1.221817e-01,
  };
  constexpr double pi = 3.141592653589793238462643383279502884;
  for (std::size_t primitive = 0u; primitive < legacy_alpha.size(); ++primitive) {
    const double s_exponent = legacy_alpha[primitive] * 2.886237 * 2.886237;
    const double p_exponent = legacy_alpha[primitive] * 2.190987 * 2.190987;
    CHECK(near(plan.primitive_exponents[bromine_s_primitive + primitive], s_exponent));
    CHECK(near(plan.primitive_exponents[bromine_p_primitive + primitive], p_exponent));
    const double s_normalization = std::pow(2.0 / pi * s_exponent, 0.75);
    const double p_normalization =
        std::pow(2.0 / pi * p_exponent, 0.75) * std::sqrt(4.0 * p_exponent);
    CHECK(near(plan.primitive_coefficients[bromine_s_primitive + primitive],
               legacy_s_coefficients[primitive] * s_normalization));
    CHECK(near(plan.primitive_coefficients[bromine_p_primitive + primitive],
               legacy_p_coefficients[primitive] * p_normalization));
  }
  return 0;
}

int test_hydrogen_repeated_shell_fixture() {
  constexpr std::array<std::int64_t, 2> offsets{0, 1};
  constexpr std::array<std::int32_t, 1> atomic_numbers{1};
  xtbloom::detail::gfn1::BasisPlan plan;
  std::string error;
  CHECK(xtbloom::detail::gfn1::make_basis_plan(1, 1, offsets.data(), atomic_numbers.data(), plan,
                                               error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(plan.total_shells == 2);
  CHECK(plan.total_orbitals == 2);
  CHECK(plan.total_primitives == 11);
  CHECK(plan.principal_quantum_numbers == std::vector<std::uint8_t>({1u, 2u}));
  CHECK(plan.angular_momenta == std::vector<std::uint8_t>({0u, 0u}));
  CHECK(plan.shell_is_valence == std::vector<std::uint8_t>({1u, 0u}));
  CHECK(plan.shell_primitive_offsets == std::vector<std::int64_t>({0, 4, 11}));

  /* The appended tail is exactly the first shell's exponents. */
  for (std::size_t primitive = 0; primitive < 4; ++primitive) {
    CHECK(plan.primitive_exponents[7u + primitive] == plan.primitive_exponents[primitive]);
  }
  CHECK(near(contracted_s_overlap(plan, 0, 0), 1.0, 2.0e-8));
  CHECK(near(contracted_s_overlap(plan, 1, 1), 1.0, 2.0e-8));
  CHECK(near(contracted_s_overlap(plan, 0, 1), 0.0, 1.0e-9));
  return 0;
}

int test_validation_and_strong_failure() {
  xtbloom::detail::gfn1::BasisPlan plan;
  plan.batch_size = 17;
  std::string error;
  constexpr std::array<std::int64_t, 2> offsets{0, 1};
  constexpr std::array<std::int64_t, 2> wrong_start{1, 1};
  constexpr std::array<std::int64_t, 3> descending{0, 2, 1};
  constexpr std::array<std::int32_t, 1> hydrogen{1};
  constexpr std::array<std::int32_t, 1> low{0};
  constexpr std::array<std::int32_t, 1> high{87};
  CHECK(xtbloom::detail::gfn1::make_basis_plan(0, 1, offsets.data(), hydrogen.data(), plan,
                                               error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn1::make_basis_plan(1, 1, nullptr, hydrogen.data(), plan, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn1::make_basis_plan(1, 1, wrong_start.data(), hydrogen.data(), plan,
                                               error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn1::make_basis_plan(2, 1, descending.data(), hydrogen.data(), plan,
                                               error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn1::make_basis_plan(1, 1, offsets.data(), low.data(), plan, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn1::make_basis_plan(1, 1, offsets.data(), high.data(), plan, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(plan.batch_size == 17);

  constexpr std::array<std::int64_t, 2> huge{0, std::numeric_limits<std::int64_t>::max()};
  CHECK(xtbloom::detail::gfn1::make_basis_plan(1, std::numeric_limits<std::int64_t>::max(),
                                               huge.data(), hydrogen.data(), plan,
                                               error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(plan.batch_size == 17);
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_complete_element_topology(); status != 0) return status;
  if (const int status = test_hydrogen_repeated_shell_fixture(); status != 0) return status;
  return test_validation_and_strong_failure();
}
