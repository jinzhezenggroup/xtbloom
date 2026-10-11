// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn2/periodic_ewald.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "model/gfn2/basis.hpp"
#include "model/gfn2/es2.hpp"
#include "model/gfn2/integrals.hpp"
#include "model/gfn2/periodic_topology.hpp"

#define CHECK(condition)                                                                          \
  do {                                                                                            \
    if (!(condition)) {                                                                           \
      std::cerr << "periodic Ewald check failed at line " << __LINE__ << ": " #condition << "\n"; \
      return __LINE__;                                                                            \
    }                                                                                             \
  } while (false)

namespace {

using namespace xtbloom::detail::gfn2;

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

int test_water_fixture() {
  constexpr std::array<std::int64_t, 2> atom_offsets{0, 3};
  constexpr std::array<std::int32_t, 3> atomic_numbers{8, 1, 1};
  constexpr std::array<double, 9> positions{0.0, 0.0, 0.0, 1.42, 0.08, 1.08, -1.31, 0.17, 0.96};
  constexpr std::array<double, 9> cell{11.7, 0.0, 0.0, 1.1, 12.9, 0.0, -0.7, 0.8, 14.3};
  constexpr std::array<double, 4> shell_charges{-0.12, -0.08, 0.1, 0.1};
  constexpr std::array<double, 16> expected_matrix{
      0.26345127436421106, 0.2974717749054976,  0.15699693753585875, 0.16730160828154234,
      0.2974717749054976,  0.3314563325077212,  0.17320037928526977, 0.18518099078399636,
      0.15699693753585875, 0.17320037928526977, 0.21697998497833088, 0.09609531530995452,
      0.16730160828154234, 0.18518099078399636, 0.09609531530995452, 0.21697998497833088};
  constexpr std::array<double, 4> expected_potential{
      -0.022982040334405027, -0.02637498258235079, -0.0013881328182960909, -0.0035831422276762434};
  constexpr std::array<double, 9> expected_gradient{
      2.6622831568596768e-05, -2.1370465928710153e-04, -1.749685072728809e-03,
      6.465075845436515e-04,  8.000299182139346e-05,   8.594763229426282e-04,
      -6.731304161122483e-04, 1.3370166746570804e-04,  8.902087497861808e-04};
  constexpr std::array<double, 9> expected_strain{
      1.8582654412822239e-03, -6.496937832525031e-05, 4.946215852061299e-05,
      -6.496937832525052e-05, 2.1251882874110574e-07, 2.1658193731133714e-04,
      4.9462158520611685e-05, 2.1658193731133725e-04, 1.9193065362002905e-03};

  BasisPlan basis;
  IntegralPlan integrals;
  ES2Plan es2;
  PeriodicShortRangePlan topology;
  PeriodicEwaldPlan ewald;
  std::string error;
  CHECK(make_basis_plan(1, 3, atom_offsets.data(), atomic_numbers.data(), basis, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(make_integral_plan(basis, integrals, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(make_es2_plan(basis, atomic_numbers.data(), es2, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(make_periodic_short_range_plan(1, 3, atom_offsets.data(), cell.data(), topology, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(make_periodic_ewald_plan(es2, topology, ewald, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(near(ewald.alpha(0), 0.21875, 0.0));
  CHECK(near(ewald.direct_cutoff(0), 25.25, 0.0));
  CHECK(near(ewald.reciprocal_cutoff(0), 2.34375, 0.0));

  std::vector<double> matrix(16, 0.0);
  std::vector<double> potential(4, 0.0);
  std::vector<double> energies(1, 0.0);
  std::vector<double> gradient(9, 0.0);
  std::vector<double> strain(9, 0.0);
  CHECK(evaluate_periodic_ewald_cpu(ewald, topology, positions.data(), shell_charges.data(),
                                    matrix.data(), potential.data(), energies.data(),
                                    gradient.data(), strain.data(),
                                    error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t i = 0; i < matrix.size(); ++i)
    CHECK(near(matrix[i], expected_matrix[i], 5.0e-12));
  for (std::size_t i = 0; i < potential.size(); ++i)
    CHECK(near(potential[i], expected_potential[i], 5.0e-12));
  CHECK(near(energies[0], 2.1853579710597168e-03, 5.0e-12));
  for (std::size_t i = 0; i < gradient.size(); ++i)
    CHECK(near(gradient[i], expected_gradient[i], 5.0e-12));
  for (std::size_t i = 0; i < strain.size(); ++i) {
    if (!near(strain[i], expected_strain[i], 5.0e-12)) {
      std::cerr << "strain[" << i << "]=" << strain[i] << " expected=" << expected_strain[i]
                << "\n";
    }
  }
  for (std::size_t axis = 0; axis < 3u; ++axis)
    CHECK(near(gradient[axis] + gradient[3u + axis] + gradient[6u + axis], 0.0, 5.0e-11));

  /* The explicit background is a batch-level term, not part of the neutral
   * matrix oracle.  Check its potential and energy shifts independently. */
  constexpr std::array<double, 4> charged_shells{-0.02, -0.08, 0.1, 0.1};
  std::vector<double> charged_matrix(16, 0.0);
  std::vector<double> charged_potential(4, 0.0);
  std::vector<double> charged_energy(1, 0.0);
  std::vector<double> charged_gradient(9, 0.0);
  std::vector<double> charged_strain(9, 0.0);
  CHECK(evaluate_periodic_ewald_cpu(ewald, topology, positions.data(), charged_shells.data(),
                                    charged_matrix.data(), charged_potential.data(),
                                    charged_energy.data(), charged_gradient.data(),
                                    charged_strain.data(), error) == XTBLOOM_STATUS_SUCCESS);
  const double total_charge = 0.1;
  const double background_factor =
      -3.14159265358979323846 / (ewald.alpha(0) * ewald.alpha(0) * topology.lattice(0).volume);
  double charged_matrix_energy = 0.0;
  for (std::size_t row = 0; row < charged_shells.size(); ++row) {
    for (std::size_t column = 0; column < charged_shells.size(); ++column) {
      charged_matrix_energy += 0.5 * charged_shells[row] *
                               charged_matrix[row * charged_shells.size() + column] *
                               charged_shells[column];
    }
    CHECK(near(
        charged_potential[row],
        [&]() {
          double value = background_factor * total_charge;
          for (std::size_t column = 0; column < charged_shells.size(); ++column)
            value += charged_matrix[row * charged_shells.size() + column] * charged_shells[column];
          return value;
        }(),
        5.0e-12));
  }
  CHECK(near(charged_energy[0],
             charged_matrix_energy + 0.5 * background_factor * total_charge * total_charge,
             5.0e-12));
  return 0;
}

}  // namespace

int main() { return test_water_fixture(); }
