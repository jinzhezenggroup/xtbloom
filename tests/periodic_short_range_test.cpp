#include "model/gfn2/coordination.hpp"
#include "model/gfn2/d4.hpp"
#include "model/gfn2/periodic_topology.hpp"
#include "model/gfn2/repulsion.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
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

using xtbloom::detail::gfn2::CoordinationPlan;
using xtbloom::detail::gfn2::D4Plan;
using xtbloom::detail::gfn2::D4Workspace;
using xtbloom::detail::gfn2::PeriodicShortRangeGeometry;
using xtbloom::detail::gfn2::PeriodicShortRangePlan;
using xtbloom::detail::gfn2::PeriodicShortRangeWorkspace;
using xtbloom::detail::gfn2::RepulsionPlan;

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

void affine_deformation(const std::array<double, 9>& reference_positions,
                        const std::array<double, 9>& reference_cell, std::size_t row,
                        std::size_t column, double amount, std::array<double, 9>& positions,
                        std::array<double, 9>& cell) {
  positions = reference_positions;
  cell = reference_cell;
  /* Match the public contract r'=(I+eps)r and H'=H(I+eps)^T. */
  for (std::size_t atom = 0; atom < 3u; ++atom) {
    positions[atom * 3u + row] += amount * reference_positions[atom * 3u + column];
  }
  for (std::size_t lattice_row = 0; lattice_row < 3u; ++lattice_row) {
    cell[lattice_row * 3u + row] += amount * reference_cell[lattice_row * 3u + column];
  }
}

struct AlignedWorkspace {
  explicit AlignedWorkspace(std::size_t size) : storage(size + 63u) {
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(storage.data());
    const std::uintptr_t aligned = (address + 63u) & ~std::uintptr_t{63u};
    data = reinterpret_cast<void*>(aligned);
  }

  std::vector<std::byte> storage;
  void* data = nullptr;
};

struct Fixture {
  std::array<std::int64_t, 2> offsets{0, 3};
  std::array<std::int32_t, 3> atomic_numbers{8, 1, 1};
  std::array<double, 9> positions{
      0.0, 0.0, 0.0,
      1.42, 0.08, 1.08,
      -1.31, 0.17, 0.96,
  };
  /* Public row-major direct vectors; the oracle stores their transpose as
   * Fortran columns, which has the same flat byte order. */
  std::array<double, 9> cell{
      11.7, 0.0, 0.0,
      1.1, 12.9, 0.0,
      -0.7, 0.8, 14.3,
  };
  PeriodicShortRangePlan periodic;
  CoordinationPlan coordination;
  RepulsionPlan repulsion;
  D4Plan d4;
  AlignedWorkspace storage{1u};
  AlignedWorkspace d4_storage{1u};
  PeriodicShortRangeWorkspace workspace;
  D4Workspace d4_workspace;
  PeriodicShortRangeGeometry geometry;

  bool initialize(std::string& error, std::uint64_t generation = 1u) {
    if (xtbloom::detail::gfn2::make_periodic_short_range_plan(
            1, 3, offsets.data(), cell.data(), periodic, error) != XTBLOOM_STATUS_SUCCESS ||
        xtbloom::detail::gfn2::make_coordination_plan(
            1, 3, offsets.data(), atomic_numbers.data(), coordination,
            error) != XTBLOOM_STATUS_SUCCESS ||
        xtbloom::detail::gfn2::make_repulsion_plan(
            1, 3, offsets.data(), atomic_numbers.data(), repulsion,
            error) != XTBLOOM_STATUS_SUCCESS ||
        xtbloom::detail::gfn2::make_d4_plan(
            1, 3, offsets.data(), atomic_numbers.data(), d4,
            error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    storage = AlignedWorkspace(periodic.workspace_size_bytes());
    d4_storage = AlignedWorkspace(d4.workspace_size_bytes());
    return xtbloom::detail::gfn2::bind_periodic_short_range_workspace(
               periodic, storage.data, periodic.workspace_size_bytes(), workspace,
               error) == XTBLOOM_STATUS_SUCCESS &&
           xtbloom::detail::gfn2::bind_d4_workspace(
               d4, d4_storage.data, d4.workspace_size_bytes(), d4_workspace,
               error) == XTBLOOM_STATUS_SUCCESS &&
           xtbloom::detail::gfn2::update_periodic_short_range_geometry_cpu(
               periodic, positions.data(), generation, workspace, geometry,
               error) == XTBLOOM_STATUS_SUCCESS;
  }
};

struct D4Values {
  std::array<double, 3> coordination{};
  std::array<double, 3> pair_energies{};
  std::array<double, 3> atm_energies{};
  std::array<double, 3> charge_potentials{};
  std::array<double, 9> gradient{};
  std::array<double, 9> strain{};

  [[nodiscard]] double total_energy() const {
    double total = 0.0;
    for (std::size_t atom = 0; atom < pair_energies.size(); ++atom) {
      total += pair_energies[atom] + atm_energies[atom];
    }
    return total;
  }
};

bool evaluate_d4_values(const D4Plan& d4, const PeriodicShortRangePlan& periodic,
                        const PeriodicShortRangeGeometry& geometry,
                        const std::array<double, 3>& charges, D4Workspace& d4_workspace,
                        const PeriodicShortRangeWorkspace& workspace, bool derivatives,
                        D4Values& values, std::string& error) {
  if (xtbloom::detail::gfn2::evaluate_periodic_d4_coordination_cpu(
          d4, periodic, geometry, values.coordination.data(), workspace,
          error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::evaluate_periodic_d4_two_body_cpu(
          d4, periodic, geometry, values.coordination.data(), charges.data(),
          values.pair_energies.data(), values.charge_potentials.data(), d4_workspace,
          workspace, error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::evaluate_periodic_d4_atm_cpu(
          d4, periodic, geometry, values.coordination.data(), values.atm_energies.data(),
          d4_workspace, workspace, error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  if (!derivatives) return true;
  if (xtbloom::detail::gfn2::add_periodic_d4_two_body_gradient_cpu(
          d4, periodic, geometry, values.coordination.data(), charges.data(),
          values.gradient.data(), values.strain.data(), d4_workspace, workspace,
          error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::add_periodic_d4_atm_gradient_cpu(
          d4, periodic, geometry, values.coordination.data(), values.gradient.data(),
          values.strain.data(), d4_workspace, workspace,
          error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  return true;
}

constexpr std::array<double, 3> kExpectedCoordination{
    1.9584219021853333,
    0.9840589791512316,
    1.0068492940904639,
};

/* Fortran order [axis, displaced_atom, coordination_atom]. */
constexpr std::array<double, 27> kExpectedCoordinationCartesian{
    0.1188823059441014, 0.019168864441299883, 0.19049028332152243,
    -0.18589286234017496, -0.010472837504321764, -0.1413833075537824,
    0.06701055639607356, -0.008696026936978124, -0.04910697576774006,
    0.18589286234017496, 0.010472837504321764, 0.1413833075537824,
    -0.22056352646326938, -0.00932984849119577, -0.14290729290008206,
    0.0346706641230945, -0.0011429890131259952, 0.0015239853462996923,
    -0.06701055639607356, 0.008696026936978124, 0.04910697576774006,
    -0.0346706641230945, 0.0011429890131259952, -0.0015239853462996923,
    0.10168122051916803, -0.009839015950104118, -0.04758299042144037,
};

/* Fortran order [strain_row, strain_column, coordination_atom]. */
constexpr std::array<double, 27> kExpectedCoordinationStrain{
    -0.3517520433288607, -0.0034796396168503226, -0.1364341561673653,
    -0.0034796396168503244, -0.002316244027322561, -0.019658851714438076,
    -0.1364341561673653, -0.019658851714438073, -0.19983670714813606,
    -0.35861886567392826, -0.0117510695924886, -0.20492477066612774,
    -0.0117510695924886, -0.0009407085783962702, -0.011173506057139461,
    -0.20492477066612774, -0.01117350605713946, -0.15287685769914705,
    -0.18243482390700236, 0.014512153801732649, 0.06016965441243299,
    0.014512153801732649, -0.0015812061724554273, -0.008211027394112029,
    0.06016965441243299, -0.008211027394112029, -0.0473255823299251,
};

int test_oracle_coordination_values_and_vjp() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(error));
  const auto translations = fixture.periodic.translations(
      0, xtbloom::detail::gfn2::PeriodicTranslationCutoff::kShortRange25);
  CHECK(translations.size == 175);

  std::array<double, 3> coordination{};
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_coordination_cpu(
            fixture.coordination, fixture.periodic, fixture.geometry,
            coordination.data(), fixture.workspace,
            error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t atom = 0; atom < coordination.size(); ++atom) {
    CHECK(near(coordination[atom], kExpectedCoordination[atom], 5.0e-12));
  }

  for (std::size_t target = 0; target < 3u; ++target) {
    std::array<double, 3> adjoint{};
    adjoint[target] = 1.0;
    std::array<double, 9> gradients{};
    std::array<double, 9> strain{};
    CHECK(xtbloom::detail::gfn2::add_periodic_coordination_gradient_cpu(
              fixture.coordination, fixture.periodic, fixture.geometry,
              adjoint.data(), gradients.data(), strain.data(), fixture.workspace,
              error) == XTBLOOM_STATUS_SUCCESS);
    for (std::size_t atom = 0; atom < 3u; ++atom) {
      for (std::size_t axis = 0; axis < 3u; ++axis) {
        const double expected =
            kExpectedCoordinationCartesian[axis + 3u * atom + 9u * target];
        CHECK(near(gradients[atom * 3u + axis], expected, 5.0e-12));
      }
    }
    for (std::size_t row = 0; row < 3u; ++row) {
      for (std::size_t column = 0; column < 3u; ++column) {
        const double expected =
            kExpectedCoordinationStrain[row + 3u * column + 9u * target];
        CHECK(near(strain[row * 3u + column], expected, 5.0e-12));
      }
    }
  }
  return 0;
}

int test_oracle_repulsion_values_and_derivatives() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(error));
  constexpr std::array<double, 3> expected_per_atom{
      0.0299248514281825,
      0.010153257561506077,
      0.020822630121013735,
  };
  constexpr std::array<double, 9> expected_gradient{
      -0.08084602403649756, 0.02459377252590664, 0.17240654162296565,
      -0.07848145084794492, -0.004179738922418421, -0.057750242003783286,
      0.1593274748844425, -0.02041403360348822, -0.11465629961918236,
  };
  constexpr std::array<double, 9> expected_strain{
      -0.32016266111962055, 0.02080715466233598, 0.06819440897329054,
      0.020807154662335983, -0.003804764829448717, -0.02411159029556844,
      0.06819440897329057, -0.02411159029556844, -0.17244030899864174,
  };
  std::array<double, 3> per_atom{};
  std::array<double, 9> gradient{};
  std::array<double, 9> strain{};
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_repulsion_cpu(
            fixture.repulsion, fixture.periodic, fixture.geometry, per_atom.data(),
            gradient.data(), strain.data(), fixture.workspace,
            error) == XTBLOOM_STATUS_SUCCESS);
  double total = 0.0;
  for (std::size_t atom = 0; atom < per_atom.size(); ++atom) {
    CHECK(near(per_atom[atom], expected_per_atom[atom], 5.0e-12));
    total += per_atom[atom];
  }
  CHECK(near(total, 0.06090073911070231, 5.0e-12));
  for (std::size_t index = 0; index < gradient.size(); ++index) {
    CHECK(near(gradient[index], expected_gradient[index], 5.0e-12));
    CHECK(near(strain[index], expected_strain[index], 5.0e-12));
  }
  for (std::size_t axis = 0; axis < 3u; ++axis) {
    CHECK(near(gradient[axis] + gradient[3u + axis] + gradient[6u + axis], 0.0,
               5.0e-11));
  }
  return 0;
}

int test_oracle_d4_values_and_derivatives() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(error));
  constexpr std::array<double, 3> expected_coordination{
      1.6158375019069111,
      0.8066388615192044,
      0.8091986403878657,
  };
  /* Fortran order [axis, displaced_atom, coordination_atom]. */
  constexpr std::array<double, 27> expected_coordination_cartesian{
      0.026346027842993884, 0.002213054518558942, 0.025885676563224984,
      -0.030261857702836705, -0.00170489339170911, -0.02301606078807299,
      0.003915829859842822, -0.0005081611268498319, -0.002869615775151992,
      0.030261857702836705, 0.00170489339170911, 0.02301606078807299,
      -0.030261857706752708, -0.0017048933915800111, -0.023016060788245123,
      3.9160031044731925e-12, -1.290990034441712e-13, 1.721320045922284e-13,
      -0.003915829859842822, 0.0005081611268498319, 0.002869615775151992,
      -3.9160031044731925e-12, 1.290990034441712e-13, -1.721320045922284e-13,
      0.003915829863758825, -0.0005081611269789309, -0.00286961577497986,
  };
  /* Fortran order [strain_row, strain_column, coordination_atom]. */
  constexpr std::array<double, 27> expected_coordination_strain{
      -0.04810157505442222, -0.0017552575400536564, -0.028923609653614537,
      -0.0017552575400536564, -0.00022277886290120023, -0.0023291195448216775,
      -0.028923609653614537, -0.002329119544821678, -0.027612176795264745,
      -0.04297183794871881, -0.002420948615874496, -0.03268280631953357,
      -0.002420948615874496, -0.00013639147134834772, -0.0018412848630303472,
      -0.03268280631953357, -0.0018412848630303474, -0.02485734565113949,
      -0.005129737127084786, 0.00066569107652572, 0.0037591966649791883,
      0.00066569107652572, -8.638739157609033e-05, -0.0004878346817603467,
      0.003759196664979189, -0.0004878346817603468, -0.002754831144166568,
  };
  constexpr std::array<double, 3> charges{-0.2, 0.1, 0.1};
  constexpr std::array<double, 3> expected_pair_energy{
      -0.000118362306827777,
      -6.952019497814451e-05,
      -6.937858851721287e-05,
  };
  constexpr std::array<double, 3> expected_atm_energy{
      1.0097168595210285e-06,
      5.953359747866779e-07,
      5.632598684868382e-07,
  };
  constexpr std::array<double, 3> expected_potential{
      0.00010750574671171566,
      0.0003550677349712526,
      0.00035434449353966425,
  };
  constexpr std::array<double, 9> expected_pair_gradient{
      7.62231978510579e-07, 1.666594767382636e-07, 3.69794384918713e-07,
      -5.624033264992141e-06, -1.7128259182630545e-07, -3.1387469286237045e-07,
      4.861801286481558e-06, 4.6231150880418185e-09, -5.591969205634247e-08,
  };
  constexpr std::array<double, 9> expected_atm_gradient{
      7.769914924130705e-07, -5.253893764943104e-07, 2.41558463677267e-06,
      4.7000842299628995e-07, 1.0929384843016767e-07, -1.2492896603565808e-06,
      -1.2469999154093688e-06, 4.160955280641409e-07, -1.1662949764160829e-06,
  };
  constexpr std::array<double, 9> expected_pair_strain{
      0.00020682687685794293, 2.8150225648908485e-06, -2.8706159281571715e-06,
      2.8150225648908477e-06, 0.00010365398809936643, 1.1137360264197635e-06,
      -2.8706159281571715e-06, 1.1137360264197635e-06, 6.969566028528545e-05,
  };
  constexpr std::array<double, 9> expected_atm_strain{
      -1.548431599986841e-05, 8.884752672517118e-07, -1.3481594416809572e-07,
      8.88475267251714e-07, 1.2997959426444695e-05, 2.1826708712329e-07,
      -1.3481594416809765e-07, 2.182670871232993e-07, -1.1841233881871497e-05,
  };

  std::array<double, 3> coordination{};
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_d4_coordination_cpu(
            fixture.d4, fixture.periodic, fixture.geometry, coordination.data(),
            fixture.workspace, error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t atom = 0; atom < 3u; ++atom) {
    CHECK(near(coordination[atom], expected_coordination[atom], 5.0e-12));
  }
  for (std::size_t target = 0; target < 3u; ++target) {
    std::array<double, 3> adjoint{};
    adjoint[target] = 1.0;
    std::array<double, 9> gradient{};
    std::array<double, 9> strain{};
    CHECK(xtbloom::detail::gfn2::add_periodic_d4_coordination_gradient_cpu(
              fixture.d4, fixture.periodic, fixture.geometry, adjoint.data(), gradient.data(),
              strain.data(), fixture.workspace, error) == XTBLOOM_STATUS_SUCCESS);
    for (std::size_t atom = 0; atom < 3u; ++atom) {
      for (std::size_t axis = 0; axis < 3u; ++axis) {
        CHECK(near(gradient[atom * 3u + axis],
                   expected_coordination_cartesian[axis + 3u * atom + 9u * target],
                   5.0e-12));
      }
    }
    for (std::size_t row = 0; row < 3u; ++row) {
      for (std::size_t column = 0; column < 3u; ++column) {
        CHECK(near(strain[row * 3u + column],
                   expected_coordination_strain[row + 3u * column + 9u * target],
                   5.0e-12));
      }
    }
  }

  std::array<double, 3> pair_energy{};
  std::array<double, 3> potential{};
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_d4_two_body_cpu(
            fixture.d4, fixture.periodic, fixture.geometry, coordination.data(), charges.data(),
            pair_energy.data(), potential.data(), fixture.d4_workspace, fixture.workspace,
            error) == XTBLOOM_STATUS_SUCCESS);
  std::array<double, 9> pair_gradient{};
  std::array<double, 9> pair_strain{};
  CHECK(xtbloom::detail::gfn2::add_periodic_d4_two_body_gradient_cpu(
            fixture.d4, fixture.periodic, fixture.geometry, coordination.data(), charges.data(),
            pair_gradient.data(), pair_strain.data(), fixture.d4_workspace, fixture.workspace,
            error) == XTBLOOM_STATUS_SUCCESS);

  std::array<double, 3> atm_energy{};
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_d4_atm_cpu(
            fixture.d4, fixture.periodic, fixture.geometry, coordination.data(), atm_energy.data(),
            fixture.d4_workspace, fixture.workspace, error) == XTBLOOM_STATUS_SUCCESS);
  std::array<double, 9> atm_gradient{};
  std::array<double, 9> atm_strain{};
  CHECK(xtbloom::detail::gfn2::add_periodic_d4_atm_gradient_cpu(
            fixture.d4, fixture.periodic, fixture.geometry, coordination.data(),
            atm_gradient.data(), atm_strain.data(), fixture.d4_workspace, fixture.workspace,
            error) == XTBLOOM_STATUS_SUCCESS);

  double total_energy = 0.0;
  for (std::size_t atom = 0; atom < 3u; ++atom) {
    CHECK(near(pair_energy[atom], expected_pair_energy[atom], 5.0e-12));
    CHECK(near(atm_energy[atom], expected_atm_energy[atom], 5.0e-12));
    CHECK(near(potential[atom], expected_potential[atom], 5.0e-12));
    total_energy += pair_energy[atom] + atm_energy[atom];
  }
  CHECK(near(total_energy, -0.00025509277762033987, 5.0e-12));
  for (std::size_t index = 0; index < 9u; ++index) {
    CHECK(near(pair_gradient[index], expected_pair_gradient[index], 5.0e-12));
    CHECK(near(atm_gradient[index], expected_atm_gradient[index], 5.0e-12));
    CHECK(near(pair_strain[index], expected_pair_strain[index], 5.0e-12));
    CHECK(near(atm_strain[index], expected_atm_strain[index], 5.0e-12));
  }
  for (std::size_t axis = 0; axis < 3u; ++axis) {
    const double net = pair_gradient[axis] + pair_gradient[3u + axis] +
                       pair_gradient[6u + axis] + atm_gradient[axis] +
                       atm_gradient[3u + axis] + atm_gradient[6u + axis];
    CHECK(near(net, 0.0, 5.0e-11));
  }
  return 0;
}

int test_wrapping_and_transactional_failures() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(error));
  constexpr std::array<double, 3> charges{-0.2, 0.1, 0.1};
  std::array<double, 3> reference{};
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_coordination_cpu(
            fixture.coordination, fixture.periodic, fixture.geometry, reference.data(),
            fixture.workspace, error) == XTBLOOM_STATUS_SUCCESS);
  D4Values reference_d4;
  CHECK(evaluate_d4_values(fixture.d4, fixture.periodic, fixture.geometry, charges,
                           fixture.d4_workspace, fixture.workspace, true, reference_d4,
                           error));

  auto shifted = fixture.positions;
  for (std::size_t axis = 0; axis < 3u; ++axis) {
    shifted[3u + axis] += fixture.cell[axis];
    shifted[6u + axis] -= fixture.cell[6u + axis];
  }
  CHECK(xtbloom::detail::gfn2::update_periodic_short_range_geometry_cpu(
            fixture.periodic, shifted.data(), 2u, fixture.workspace, fixture.geometry,
            error) == XTBLOOM_STATUS_SUCCESS);
  std::array<double, 3> translated{};
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_coordination_cpu(
            fixture.coordination, fixture.periodic, fixture.geometry, translated.data(),
            fixture.workspace, error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t atom = 0; atom < reference.size(); ++atom) {
    CHECK(near(translated[atom], reference[atom], 5.0e-13));
  }
  D4Values translated_d4;
  CHECK(evaluate_d4_values(fixture.d4, fixture.periodic, fixture.geometry, charges,
                           fixture.d4_workspace, fixture.workspace, true, translated_d4,
                           error));
  for (std::size_t atom = 0; atom < 3u; ++atom) {
    CHECK(near(translated_d4.coordination[atom], reference_d4.coordination[atom], 5.0e-13));
    CHECK(near(translated_d4.pair_energies[atom], reference_d4.pair_energies[atom], 5.0e-13));
    CHECK(near(translated_d4.atm_energies[atom], reference_d4.atm_energies[atom], 5.0e-13));
    CHECK(near(translated_d4.charge_potentials[atom], reference_d4.charge_potentials[atom],
               5.0e-13));
  }
  for (std::size_t component = 0; component < 9u; ++component) {
    CHECK(near(translated_d4.gradient[component], reference_d4.gradient[component], 5.0e-13));
    CHECK(near(translated_d4.strain[component], reference_d4.strain[component], 5.0e-13));
  }

  /* U has determinant +1, so U*H is a different basis for the same lattice. */
  std::array<double, 9> basis_cell{
      fixture.cell[0] + fixture.cell[3], fixture.cell[1] + fixture.cell[4],
      fixture.cell[2] + fixture.cell[5],
      fixture.cell[3], fixture.cell[4], fixture.cell[5],
      fixture.cell[6], fixture.cell[7], fixture.cell[8],
  };
  PeriodicShortRangePlan basis_plan;
  CHECK(xtbloom::detail::gfn2::make_periodic_short_range_plan(
            1, 3, fixture.offsets.data(), basis_cell.data(), basis_plan,
            error) == XTBLOOM_STATUS_SUCCESS);
  AlignedWorkspace basis_storage(basis_plan.workspace_size_bytes());
  PeriodicShortRangeWorkspace basis_workspace;
  PeriodicShortRangeGeometry basis_geometry;
  CHECK(xtbloom::detail::gfn2::bind_periodic_short_range_workspace(
            basis_plan, basis_storage.data, basis_plan.workspace_size_bytes(), basis_workspace,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::update_periodic_short_range_geometry_cpu(
            basis_plan, fixture.positions.data(), 1u, basis_workspace, basis_geometry,
            error) == XTBLOOM_STATUS_SUCCESS);
  std::array<double, 3> basis_coordination{};
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_coordination_cpu(
            fixture.coordination, basis_plan, basis_geometry, basis_coordination.data(),
            basis_workspace, error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t atom = 0; atom < reference.size(); ++atom) {
    CHECK(near(basis_coordination[atom], reference[atom], 5.0e-12));
  }
  D4Values basis_d4;
  CHECK(evaluate_d4_values(fixture.d4, basis_plan, basis_geometry, charges,
                           fixture.d4_workspace, basis_workspace, true, basis_d4, error));
  for (std::size_t atom = 0; atom < 3u; ++atom) {
    CHECK(near(basis_d4.coordination[atom], reference_d4.coordination[atom], 5.0e-12));
    CHECK(near(basis_d4.pair_energies[atom], reference_d4.pair_energies[atom], 5.0e-12));
    CHECK(near(basis_d4.atm_energies[atom], reference_d4.atm_energies[atom], 5.0e-12));
    CHECK(near(basis_d4.charge_potentials[atom], reference_d4.charge_potentials[atom],
               5.0e-12));
  }
  for (std::size_t component = 0; component < 9u; ++component) {
    CHECK(near(basis_d4.gradient[component], reference_d4.gradient[component], 5.0e-12));
    CHECK(near(basis_d4.strain[component], reference_d4.strain[component], 5.0e-12));
  }

  fixture.workspace.atom_scratch[0] = 17.0;
  fixture.workspace.atom_scratch[1] = 18.0;
  fixture.workspace.atom_scratch[2] = 19.0;
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_d4_coordination_cpu(
            fixture.d4, fixture.periodic, fixture.geometry, fixture.workspace.atom_scratch,
            fixture.workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.workspace.atom_scratch[0] == 17.0);
  CHECK(fixture.workspace.atom_scratch[1] == 18.0);
  CHECK(fixture.workspace.atom_scratch[2] == 19.0);

  alignas(double) std::array<std::byte, sizeof(double) * 3u + 1u> misaligned_storage{};
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_d4_coordination_cpu(
            fixture.d4, fixture.periodic, fixture.geometry,
            reinterpret_cast<double*>(misaligned_storage.data() + 1u), fixture.workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  std::array<double, 3> overlapping_pair_outputs{27.0, 28.0, 29.0};
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_d4_two_body_cpu(
            fixture.d4, fixture.periodic, fixture.geometry, reference_d4.coordination.data(),
            charges.data(), overlapping_pair_outputs.data(), overlapping_pair_outputs.data(),
            fixture.d4_workspace, fixture.workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK((overlapping_pair_outputs == std::array<double, 3>{27.0, 28.0, 29.0}));

  fixture.d4_workspace.atom_scratch[0] = 37.0;
  fixture.d4_workspace.atom_scratch[1] = 38.0;
  fixture.d4_workspace.atom_scratch[2] = 39.0;
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_d4_atm_cpu(
            fixture.d4, fixture.periodic, fixture.geometry, reference_d4.coordination.data(),
            fixture.d4_workspace.atom_scratch, fixture.d4_workspace, fixture.workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.d4_workspace.atom_scratch[0] == 37.0);
  CHECK(fixture.d4_workspace.atom_scratch[1] == 38.0);
  CHECK(fixture.d4_workspace.atom_scratch[2] == 39.0);

  std::array<double, 9> overlapping_derivatives{};
  overlapping_derivatives.fill(47.0);
  CHECK(xtbloom::detail::gfn2::add_periodic_d4_atm_gradient_cpu(
            fixture.d4, fixture.periodic, fixture.geometry, reference_d4.coordination.data(),
            overlapping_derivatives.data(), overlapping_derivatives.data(), fixture.d4_workspace,
            fixture.workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(overlapping_derivatives.begin(), overlapping_derivatives.end(),
                    [](double value) { return value == 47.0; }));

  /* Retained descriptors are immutable views. Reject a forged nested pointer
   * before clearing scratch or publishing any term result. */
  PeriodicShortRangeWorkspace forged_workspace = fixture.workspace;
  forged_workspace.atom_scratch = forged_workspace.secondary_atom_scratch;
  std::array<double, 3> forged_output{57.0, 58.0, 59.0};
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_coordination_cpu(
            fixture.coordination, fixture.periodic, fixture.geometry, forged_output.data(),
            forged_workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK((forged_output == std::array<double, 3>{57.0, 58.0, 59.0}));

  /* The 50-bohr translation vector is large enough to make a plausible
   * aligned arena. Both the periodic binder and D4 evaluator must still
   * reject it because it is immutable topology storage. */
  const auto translation_storage = fixture.periodic.translations(
      0, xtbloom::detail::gfn2::PeriodicTranslationCutoff::kD4TwoBody50);
  const std::uintptr_t translation_begin =
      reinterpret_cast<std::uintptr_t>(translation_storage.data);
  const std::uintptr_t aligned_translation_begin =
      (translation_begin + 63u) & ~std::uintptr_t{63u};
  const std::uintptr_t translation_end =
      translation_begin + static_cast<std::size_t>(translation_storage.size) *
                              sizeof(xtbloom::detail::gfn2::LatticeTranslation);
  CHECK(aligned_translation_begin + fixture.d4.workspace_size_bytes() <= translation_end);
  PeriodicShortRangeWorkspace rejected_workspace;
  CHECK(xtbloom::detail::gfn2::bind_periodic_short_range_workspace(
            fixture.periodic, reinterpret_cast<void*>(aligned_translation_begin),
            fixture.periodic.workspace_size_bytes(), rejected_workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  D4Workspace cross_plan_workspace = fixture.d4_workspace;
  const std::uintptr_t original_d4_base =
      reinterpret_cast<std::uintptr_t>(fixture.d4_workspace.workspace_base);
  const auto relocate_d4_pointer = [&](double* pointer) {
    return reinterpret_cast<double*>(aligned_translation_begin +
                                     (reinterpret_cast<std::uintptr_t>(pointer) - original_d4_base));
  };
  cross_plan_workspace.workspace_base = reinterpret_cast<void*>(aligned_translation_begin);
  cross_plan_workspace.pair_scratch = relocate_d4_pointer(fixture.d4_workspace.pair_scratch);
  cross_plan_workspace.coordination_scratch =
      relocate_d4_pointer(fixture.d4_workspace.coordination_scratch);
  cross_plan_workspace.weights = relocate_d4_pointer(fixture.d4_workspace.weights);
  cross_plan_workspace.weight_cn_derivatives =
      relocate_d4_pointer(fixture.d4_workspace.weight_cn_derivatives);
  cross_plan_workspace.weight_charge_derivatives =
      relocate_d4_pointer(fixture.d4_workspace.weight_charge_derivatives);
  cross_plan_workspace.atom_scratch = relocate_d4_pointer(fixture.d4_workspace.atom_scratch);
  cross_plan_workspace.coordination_adjoints =
      relocate_d4_pointer(fixture.d4_workspace.coordination_adjoints);
  cross_plan_workspace.batch_scratch = relocate_d4_pointer(fixture.d4_workspace.batch_scratch);
  cross_plan_workspace.gradient_scratch =
      relocate_d4_pointer(fixture.d4_workspace.gradient_scratch);
  std::array<double, 3> cross_plan_energy{67.0, 68.0, 69.0};
  std::array<double, 3> cross_plan_potential{77.0, 78.0, 79.0};
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_d4_two_body_cpu(
            fixture.d4, fixture.periodic, fixture.geometry, reference_d4.coordination.data(),
            charges.data(), cross_plan_energy.data(), cross_plan_potential.data(),
            cross_plan_workspace, fixture.workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK((cross_plan_energy == std::array<double, 3>{67.0, 68.0, 69.0}));
  CHECK((cross_plan_potential == std::array<double, 3>{77.0, 78.0, 79.0}));

  std::array<double, 3> reference_energy{};
  std::array<double, 9> reference_gradient{};
  std::array<double, 9> reference_strain{};
  CHECK(xtbloom::detail::gfn2::update_periodic_short_range_geometry_cpu(
            fixture.periodic, fixture.positions.data(), 4u, fixture.workspace, fixture.geometry,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_repulsion_cpu(
            fixture.repulsion, fixture.periodic, fixture.geometry, reference_energy.data(),
            reference_gradient.data(), reference_strain.data(), fixture.workspace,
            error) == XTBLOOM_STATUS_SUCCESS);
  std::array<double, 3> basis_energy{};
  std::array<double, 9> basis_gradient{};
  std::array<double, 9> basis_strain{};
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_repulsion_cpu(
            fixture.repulsion, basis_plan, basis_geometry, basis_energy.data(),
            basis_gradient.data(), basis_strain.data(), basis_workspace,
            error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t atom = 0; atom < basis_energy.size(); ++atom) {
    CHECK(near(basis_energy[atom], reference_energy[atom], 5.0e-12));
  }
  for (std::size_t component = 0; component < basis_gradient.size(); ++component) {
    CHECK(near(basis_gradient[component], reference_gradient[component], 5.0e-12));
    CHECK(near(basis_strain[component], reference_strain[component], 5.0e-12));
  }

  auto invalid_positions = fixture.positions;
  invalid_positions[4] = std::numeric_limits<double>::quiet_NaN();
  const PeriodicShortRangeGeometry sentinel = fixture.geometry;
  CHECK(xtbloom::detail::gfn2::update_periodic_short_range_geometry_cpu(
            fixture.periodic, invalid_positions.data(), 3u, fixture.workspace,
            fixture.geometry, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.geometry.wrapped_positions == sentinel.wrapped_positions);
  CHECK(fixture.geometry.geometry_generation == sentinel.geometry_generation);
  CHECK(fixture.geometry.plan_identity == sentinel.plan_identity);

  Fixture coincident;
  coincident.positions[3] = 0.0;
  coincident.positions[4] = 0.0;
  coincident.positions[5] = 0.0;
  CHECK(coincident.initialize(error));
  std::array<double, 3> coordination{91.0, 92.0, 93.0};
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_coordination_cpu(
            coincident.coordination, coincident.periodic, coincident.geometry,
            coordination.data(), coincident.workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK((coordination == std::array<double, 3>{91.0, 92.0, 93.0}));
  std::array<double, 3> energies{81.0, 82.0, 83.0};
  std::array<double, 9> gradients{};
  std::array<double, 9> strain{};
  gradients.fill(71.0);
  strain.fill(61.0);
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_repulsion_cpu(
            coincident.repulsion, coincident.periodic, coincident.geometry, energies.data(),
            gradients.data(), strain.data(), coincident.workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK((energies == std::array<double, 3>{81.0, 82.0, 83.0}));
  CHECK(std::all_of(gradients.begin(), gradients.end(), [](double value) { return value == 71.0; }));
  CHECK(std::all_of(strain.begin(), strain.end(), [](double value) { return value == 61.0; }));

  std::array<double, 3> d4_coordination{51.0, 52.0, 53.0};
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_d4_coordination_cpu(
            coincident.d4, coincident.periodic, coincident.geometry, d4_coordination.data(),
            coincident.workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK((d4_coordination == std::array<double, 3>{51.0, 52.0, 53.0}));

  constexpr std::array<double, 3> valid_d4_coordination{1.0, 1.0, 1.0};
  std::array<double, 3> pair_energies{41.0, 42.0, 43.0};
  std::array<double, 3> potentials{31.0, 32.0, 33.0};
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_d4_two_body_cpu(
            coincident.d4, coincident.periodic, coincident.geometry,
            valid_d4_coordination.data(), charges.data(), pair_energies.data(), potentials.data(),
            coincident.d4_workspace, coincident.workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK((pair_energies == std::array<double, 3>{41.0, 42.0, 43.0}));
  CHECK((potentials == std::array<double, 3>{31.0, 32.0, 33.0}));

  gradients.fill(21.0);
  strain.fill(11.0);
  CHECK(xtbloom::detail::gfn2::add_periodic_d4_two_body_gradient_cpu(
            coincident.d4, coincident.periodic, coincident.geometry,
            valid_d4_coordination.data(), charges.data(), gradients.data(), strain.data(),
            coincident.d4_workspace, coincident.workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(gradients.begin(), gradients.end(), [](double value) { return value == 21.0; }));
  CHECK(std::all_of(strain.begin(), strain.end(), [](double value) { return value == 11.0; }));

  std::array<double, 3> atm_energies{9.0, 8.0, 7.0};
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_d4_atm_cpu(
            coincident.d4, coincident.periodic, coincident.geometry,
            valid_d4_coordination.data(), atm_energies.data(), coincident.d4_workspace,
            coincident.workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK((atm_energies == std::array<double, 3>{9.0, 8.0, 7.0}));

  gradients.fill(6.0);
  strain.fill(5.0);
  CHECK(xtbloom::detail::gfn2::add_periodic_d4_atm_gradient_cpu(
            coincident.d4, coincident.periodic, coincident.geometry,
            valid_d4_coordination.data(), gradients.data(), strain.data(),
            coincident.d4_workspace, coincident.workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(gradients.begin(), gradients.end(), [](double value) { return value == 6.0; }));
  CHECK(std::all_of(strain.begin(), strain.end(), [](double value) { return value == 5.0; }));

  /* A representable cell whose cutoff repeat count exceeds int64 must fail
   * before replacing a previously sealed topology plan. */
  std::array<double, 9> hostile_cell{
      1.0e-100, 0.0, 0.0,
      0.0, 1.0e-100, 0.0,
      0.0, 0.0, 1.0e-100,
  };
  PeriodicShortRangePlan survivor = fixture.periodic;
  const auto* identity = survivor.identity();
  CHECK(xtbloom::detail::gfn2::make_periodic_short_range_plan(
            1, 3, fixture.offsets.data(), hostile_cell.data(), survivor,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(survivor.identity() == identity);

  constexpr std::array<std::int64_t, 3> ragged_offsets{0, 0, 1};
  constexpr std::array<double, 18> ragged_cells{
      12.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 12.0,
      13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0,
  };
  PeriodicShortRangePlan ragged;
  CHECK(xtbloom::detail::gfn2::make_periodic_short_range_plan(
            2, 1, ragged_offsets.data(), ragged_cells.data(), ragged,
            error) == XTBLOOM_STATUS_SUCCESS);
  return 0;
}

int test_complete_finite_differences() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(error));

  constexpr std::array<double, 2> steps{1.0e-4, 5.0e-5};
  constexpr std::array<std::array<std::size_t, 2>, 6> strain_modes{{
      {0u, 0u},
      {1u, 1u},
      {2u, 2u},
      {0u, 1u},
      {0u, 2u},
      {1u, 2u},
  }};

  auto evaluate_terms = [&](const std::array<double, 9>& positions,
                            const std::array<double, 9>& cell,
                            std::array<double, 3>* coordination, double* repulsion_energy) {
    PeriodicShortRangePlan periodic;
    if (xtbloom::detail::gfn2::make_periodic_short_range_plan(
            1, 3, fixture.offsets.data(), cell.data(), periodic,
            error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    AlignedWorkspace storage(periodic.workspace_size_bytes());
    PeriodicShortRangeWorkspace workspace;
    PeriodicShortRangeGeometry geometry;
    if (xtbloom::detail::gfn2::bind_periodic_short_range_workspace(
            periodic, storage.data, periodic.workspace_size_bytes(), workspace,
            error) != XTBLOOM_STATUS_SUCCESS ||
        xtbloom::detail::gfn2::update_periodic_short_range_geometry_cpu(
            periodic, positions.data(), 1u, workspace, geometry,
            error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    if (coordination != nullptr &&
        xtbloom::detail::gfn2::evaluate_periodic_coordination_cpu(
            fixture.coordination, periodic, geometry, coordination->data(), workspace,
            error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    if (repulsion_energy != nullptr) {
      std::array<double, 3> per_atom{};
      std::array<double, 9> gradient{};
      std::array<double, 9> strain{};
      if (xtbloom::detail::gfn2::evaluate_periodic_repulsion_cpu(
              fixture.repulsion, periodic, geometry, per_atom.data(), gradient.data(),
              strain.data(), workspace, error) != XTBLOOM_STATUS_SUCCESS) {
        return false;
      }
      *repulsion_energy = per_atom[0] + per_atom[1] + per_atom[2];
    }
    return true;
  };

  constexpr std::array<double, 3> charges{-0.2, 0.1, 0.1};
  auto evaluate_d4_at_geometry = [&](const std::array<double, 9>& positions,
                                     const std::array<double, 9>& cell,
                                     const std::array<double, 3>& local_charges,
                                     D4Values& values) {
    PeriodicShortRangePlan periodic;
    if (xtbloom::detail::gfn2::make_periodic_short_range_plan(
            1, 3, fixture.offsets.data(), cell.data(), periodic,
            error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    AlignedWorkspace storage(periodic.workspace_size_bytes());
    PeriodicShortRangeWorkspace workspace;
    PeriodicShortRangeGeometry geometry;
    if (xtbloom::detail::gfn2::bind_periodic_short_range_workspace(
            periodic, storage.data, periodic.workspace_size_bytes(), workspace,
            error) != XTBLOOM_STATUS_SUCCESS ||
        xtbloom::detail::gfn2::update_periodic_short_range_geometry_cpu(
            periodic, positions.data(), 1u, workspace, geometry,
            error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    return evaluate_d4_values(fixture.d4, periodic, geometry, local_charges,
                              fixture.d4_workspace, workspace, false, values, error);
  };

  for (std::size_t target = 0; target < 3u; ++target) {
    std::array<double, 3> adjoint{};
    adjoint[target] = 1.0;
    std::array<double, 9> analytic_gradient{};
    std::array<double, 9> analytic_strain{};
    CHECK(xtbloom::detail::gfn2::add_periodic_coordination_gradient_cpu(
              fixture.coordination, fixture.periodic, fixture.geometry, adjoint.data(),
              analytic_gradient.data(), analytic_strain.data(), fixture.workspace,
              error) == XTBLOOM_STATUS_SUCCESS);

    for (std::size_t atom = 0; atom < 3u; ++atom) {
      for (std::size_t axis = 0; axis < 3u; ++axis) {
        double previous_estimate = 0.0;
        for (std::size_t sample = 0; sample < steps.size(); ++sample) {
          const double step = steps[sample];
          auto plus_positions = fixture.positions;
          auto minus_positions = fixture.positions;
          plus_positions[atom * 3u + axis] += step;
          minus_positions[atom * 3u + axis] -= step;
          std::array<double, 3> plus_cn{};
          std::array<double, 3> minus_cn{};
          CHECK(evaluate_terms(plus_positions, fixture.cell, &plus_cn, nullptr));
          CHECK(evaluate_terms(minus_positions, fixture.cell, &minus_cn, nullptr));
          const double estimate = (plus_cn[target] - minus_cn[target]) / (2.0 * step);
          CHECK(near(estimate, analytic_gradient[atom * 3u + axis], 2.0e-7));
          if (sample != 0u) CHECK(near(estimate, previous_estimate, 2.0e-6));
          previous_estimate = estimate;
        }
      }
    }

    for (const auto& mode : strain_modes) {
      const std::size_t row = mode[0];
      const std::size_t column = mode[1];
      double previous_estimate = 0.0;
      for (std::size_t sample = 0; sample < steps.size(); ++sample) {
        const double step = steps[sample];
        std::array<double, 9> plus_positions{};
        std::array<double, 9> minus_positions{};
        std::array<double, 9> plus_cell{};
        std::array<double, 9> minus_cell{};
        affine_deformation(fixture.positions, fixture.cell, row, column, step, plus_positions,
                           plus_cell);
        affine_deformation(fixture.positions, fixture.cell, row, column, -step, minus_positions,
                           minus_cell);
        std::array<double, 3> plus_cn{};
        std::array<double, 3> minus_cn{};
        CHECK(evaluate_terms(plus_positions, plus_cell, &plus_cn, nullptr));
        CHECK(evaluate_terms(minus_positions, minus_cell, &minus_cn, nullptr));
        const double estimate = (plus_cn[target] - minus_cn[target]) / (2.0 * step);
        CHECK(near(estimate, analytic_strain[row * 3u + column], 2.0e-7));
        if (sample != 0u) CHECK(near(estimate, previous_estimate, 2.0e-6));
        previous_estimate = estimate;
      }
    }
  }

  std::array<double, 3> repulsion_per_atom{};
  std::array<double, 9> repulsion_gradient{};
  std::array<double, 9> repulsion_strain{};
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_repulsion_cpu(
            fixture.repulsion, fixture.periodic, fixture.geometry,
            repulsion_per_atom.data(), repulsion_gradient.data(), repulsion_strain.data(),
            fixture.workspace, error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t atom = 0; atom < 3u; ++atom) {
    for (std::size_t axis = 0; axis < 3u; ++axis) {
      double previous_estimate = 0.0;
      for (std::size_t sample = 0; sample < steps.size(); ++sample) {
        const double step = steps[sample];
        auto plus_positions = fixture.positions;
        auto minus_positions = fixture.positions;
        plus_positions[atom * 3u + axis] += step;
        minus_positions[atom * 3u + axis] -= step;
        double plus = 0.0;
        double minus = 0.0;
        CHECK(evaluate_terms(plus_positions, fixture.cell, nullptr, &plus));
        CHECK(evaluate_terms(minus_positions, fixture.cell, nullptr, &minus));
        const double estimate = (plus - minus) / (2.0 * step);
        CHECK(near(estimate, repulsion_gradient[atom * 3u + axis], 2.0e-7));
        if (sample != 0u) CHECK(near(estimate, previous_estimate, 2.0e-6));
        previous_estimate = estimate;
      }
    }
  }

  for (const auto& mode : strain_modes) {
    const std::size_t row = mode[0];
    const std::size_t column = mode[1];
    double previous_estimate = 0.0;
    for (std::size_t sample = 0; sample < steps.size(); ++sample) {
      const double step = steps[sample];
      std::array<double, 9> plus_positions{};
      std::array<double, 9> minus_positions{};
      std::array<double, 9> plus_cell{};
      std::array<double, 9> minus_cell{};
      affine_deformation(fixture.positions, fixture.cell, row, column, step, plus_positions,
                         plus_cell);
      affine_deformation(fixture.positions, fixture.cell, row, column, -step, minus_positions,
                         minus_cell);
      double plus = 0.0;
      double minus = 0.0;
      CHECK(evaluate_terms(plus_positions, plus_cell, nullptr, &plus));
      CHECK(evaluate_terms(minus_positions, minus_cell, nullptr, &minus));
      const double estimate = (plus - minus) / (2.0 * step);
      CHECK(near(estimate, repulsion_strain[row * 3u + column], 2.0e-7));
      if (sample != 0u) CHECK(near(estimate, previous_estimate, 2.0e-6));
      previous_estimate = estimate;
    }
  }

  D4Values analytic_d4;
  CHECK(evaluate_d4_values(fixture.d4, fixture.periodic, fixture.geometry, charges,
                           fixture.d4_workspace, fixture.workspace, true, analytic_d4,
                           error));
  std::array<std::array<double, 9>, 3> d4_coordination_gradients{};
  std::array<std::array<double, 9>, 3> d4_coordination_strains{};
  for (std::size_t target = 0; target < 3u; ++target) {
    std::array<double, 3> adjoint{};
    adjoint[target] = 1.0;
    CHECK(xtbloom::detail::gfn2::add_periodic_d4_coordination_gradient_cpu(
              fixture.d4, fixture.periodic, fixture.geometry, adjoint.data(),
              d4_coordination_gradients[target].data(),
              d4_coordination_strains[target].data(), fixture.workspace,
              error) == XTBLOOM_STATUS_SUCCESS);
  }

  for (std::size_t atom = 0; atom < 3u; ++atom) {
    for (std::size_t axis = 0; axis < 3u; ++axis) {
      std::array<double, 3> previous_cn{};
      double previous_energy = 0.0;
      for (std::size_t sample = 0; sample < steps.size(); ++sample) {
        const double step = steps[sample];
        auto plus_positions = fixture.positions;
        auto minus_positions = fixture.positions;
        plus_positions[atom * 3u + axis] += step;
        minus_positions[atom * 3u + axis] -= step;
        D4Values plus;
        D4Values minus;
        CHECK(evaluate_d4_at_geometry(plus_positions, fixture.cell, charges, plus));
        CHECK(evaluate_d4_at_geometry(minus_positions, fixture.cell, charges, minus));
        for (std::size_t target = 0; target < 3u; ++target) {
          const double estimate =
              (plus.coordination[target] - minus.coordination[target]) / (2.0 * step);
          CHECK(near(estimate, d4_coordination_gradients[target][atom * 3u + axis], 2.0e-7));
          if (sample != 0u) CHECK(near(estimate, previous_cn[target], 2.0e-6));
          previous_cn[target] = estimate;
        }
        const double energy_estimate = (plus.total_energy() - minus.total_energy()) / (2.0 * step);
        CHECK(near(energy_estimate, analytic_d4.gradient[atom * 3u + axis], 2.0e-7));
        if (sample != 0u) CHECK(near(energy_estimate, previous_energy, 2.0e-6));
        previous_energy = energy_estimate;
      }
    }
  }

  for (const auto& mode : strain_modes) {
    const std::size_t row = mode[0];
    const std::size_t column = mode[1];
    std::array<double, 3> previous_cn{};
    double previous_energy = 0.0;
    for (std::size_t sample = 0; sample < steps.size(); ++sample) {
      const double step = steps[sample];
      std::array<double, 9> plus_positions{};
      std::array<double, 9> minus_positions{};
      std::array<double, 9> plus_cell{};
      std::array<double, 9> minus_cell{};
      affine_deformation(fixture.positions, fixture.cell, row, column, step, plus_positions,
                         plus_cell);
      affine_deformation(fixture.positions, fixture.cell, row, column, -step, minus_positions,
                         minus_cell);
      D4Values plus;
      D4Values minus;
      CHECK(evaluate_d4_at_geometry(plus_positions, plus_cell, charges, plus));
      CHECK(evaluate_d4_at_geometry(minus_positions, minus_cell, charges, minus));
      for (std::size_t target = 0; target < 3u; ++target) {
        const double estimate =
            (plus.coordination[target] - minus.coordination[target]) / (2.0 * step);
        CHECK(near(estimate, d4_coordination_strains[target][row * 3u + column], 2.0e-7));
        if (sample != 0u) CHECK(near(estimate, previous_cn[target], 2.0e-6));
        previous_cn[target] = estimate;
      }
      const double energy_estimate = (plus.total_energy() - minus.total_energy()) / (2.0 * step);
      CHECK(near(energy_estimate, analytic_d4.strain[row * 3u + column], 2.0e-7));
      if (sample != 0u) CHECK(near(energy_estimate, previous_energy, 2.0e-6));
      previous_energy = energy_estimate;
    }
  }

  /* ATM is charge-independent, so the two-body energy derivative is the
   * complete D4 charge potential entering SCC. */
  for (std::size_t atom = 0; atom < 3u; ++atom) {
    double previous_estimate = 0.0;
    for (std::size_t sample = 0; sample < steps.size(); ++sample) {
      const double step = steps[sample];
      auto plus_charges = charges;
      auto minus_charges = charges;
      plus_charges[atom] += step;
      minus_charges[atom] -= step;
      D4Values plus;
      D4Values minus;
      CHECK(evaluate_d4_at_geometry(fixture.positions, fixture.cell, plus_charges, plus));
      CHECK(evaluate_d4_at_geometry(fixture.positions, fixture.cell, minus_charges, minus));
      const double plus_pair = plus.pair_energies[0] + plus.pair_energies[1] +
                               plus.pair_energies[2];
      const double minus_pair = minus.pair_energies[0] + minus.pair_energies[1] +
                                minus.pair_energies[2];
      const double estimate = (plus_pair - minus_pair) / (2.0 * step);
      CHECK(near(estimate, analytic_d4.charge_potentials[atom], 2.0e-7));
      if (sample != 0u) CHECK(near(estimate, previous_estimate, 2.0e-6));
      previous_estimate = estimate;
    }
  }
  return 0;
}

}  // namespace

int main() {
  if (const int line = test_oracle_coordination_values_and_vjp(); line != 0) return line;
  if (const int line = test_oracle_repulsion_values_and_derivatives(); line != 0) return line;
  if (const int line = test_oracle_d4_values_and_derivatives(); line != 0) return line;
  if (const int line = test_wrapping_and_transactional_failures(); line != 0) return line;
  if (const int line = test_complete_finite_differences(); line != 0) return line;
  return 0;
}
