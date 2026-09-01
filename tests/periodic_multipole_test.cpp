// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn2/periodic_multipole.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "model/gfn2/basis.hpp"
#include "model/gfn2/coordination.hpp"

#define CHECK(condition)                                                                     \
  do {                                                                                       \
    if (!(condition)) {                                                                      \
      std::cerr << "periodic multipole check failed at line " << __LINE__ << ": " #condition \
                << "\n";                                                                     \
      return __LINE__;                                                                       \
    }                                                                                        \
  } while (false)

namespace {

using namespace xtbloom::detail::gfn2;

constexpr std::array<std::int64_t, 2> kAtomOffsets{0, 3};
constexpr std::array<std::int32_t, 3> kAtomicNumbers{8, 1, 1};
constexpr std::array<double, 9> kPositions{0.0, 0.0, 0.0, 1.42, 0.08, 1.08, -1.31, 0.17, 0.96};
constexpr std::array<double, 9> kCell{11.7, 0.0, 0.0, 1.1, 12.9, 0.0, -0.7, 0.8, 14.3};
constexpr std::array<double, 3> kCoordination{1.9584219021853333, 0.9840589791512316,
                                              1.0068492940904639};
constexpr std::array<double, 3> kCharges{-0.2, 0.1, 0.1};
constexpr std::array<double, 9> kDipoles{0.012,  -0.004, 0.003, -0.005, 0.002,
                                         -0.001, -0.007, 0.002, -0.002};
constexpr std::array<double, 18> kQuadrupoles{0.01,   0.002,   -0.004,  -0.001, 0.003,   -0.006,
                                              0.004,  0.001,   -0.0015, 0.0,    -0.0005, -0.0025,
                                              -0.003, -0.0007, 0.001,   0.0004, 0.0002,  0.002};

/* Matrix values are converted from the Fortran fixture to the public
 * pair-major layout: [jat][iat][component], with components innermost. */
constexpr std::array<double, 27> kChargeDipoleMatrix{
    4.5263323118901674e-21,  4.9068353936645602e-21,  -1.6543612251060553e-24,
    0.041211267437392299,    0.0023105071540784634,   0.033350304579793558,
    -0.039767470677045558,   0.0053800066168654265,   0.030880859391349633,
    -0.041211267437392299,   -0.0023105071540784634,  -0.033350304579793558,
    -1.2890782666026383e-20, 1.3615392882622836e-21,  7.6141975385506197e-22,
    -0.06340349664015632,    0.0023552396163217209,   -0.0031312617702923405,
    0.039767470677045558,    -0.0053800066168654265,  -0.030880859391349633,
    0.06340349664015632,     -0.0023552396163217209,  0.0031312617702923405,
    1.0733495628488087e-20,  -1.0604455452929815e-21, 2.5295183131871586e-21};

constexpr std::array<double, 54> kChargeQuadrupoleMatrix{
    0.00096007423168759565,  3.6789858078767698e-05,  0.00063063758968710543,
    -1.1653923610147744e-05, 2.1155687139792318e-05,  0.00035007134912434529,
    0.011290705249277059,    0.0024615057815027003,   -0.010808423603570219,
    0.032892858693818451,    0.0018785575763173848,   0.0014585013962072385,
    0.011336817183002819,    -0.0055098990366506352,  -0.010152890422764122,
    -0.031294196050332131,   0.0040911774847671731,   0.00075685629211190626,
    0.011290705249277059,    0.0024615057815027003,   -0.010808423603570219,
    0.032892858693818451,    0.0018785575763173848,   0.0014585013962072381,
    0.0009612425103321864,   3.6916677148229697e-05,  0.00063041913593054905,
    -1.1683855316060284e-05, 2.1201399140411605e-05,  0.00034912152423631168,
    0.0237481240357259,      -0.0021913206937900515,  -0.010785329947630564,
    0.0029275772182810558,   -7.8147621958188113e-05, -0.011022011019041862,
    0.011336817183002819,    -0.0055098990366506352,  -0.010152890422764122,
    -0.031294196050332131,   0.0040911774847671731,   0.00075685629211190647,
    0.023748124035725904,    -0.0021913206937900515,  -0.010785329947630564,
    0.0029275772182810558,   -7.81476219581881e-05,   -0.011022011019041864,
    0.00096123731588377648,  3.6916113633387304e-05,  0.00063042010857215628,
    -1.1683722393441379e-05, 2.1201196173744034e-05,  0.00034912574604311296};

constexpr std::array<double, 81> kDipoleDipoleMatrix{
    -0.002880222695062779,   -5.5184787118151669e-05, 1.7480885415221622e-05,
    -5.5184787118151669e-05, -0.001891912769061315,   -3.1733530709688397e-05,
    1.7480885415221622e-05,  -3.1733530709688397e-05, -0.0010502140473730263,
    -0.033872115747831257,   -0.0036922586722540614,  -0.04933928804072818,
    -0.0036922586722540614,  0.032425270810710623,    -0.0028178363644760689,
    -0.04933928804072818,    -0.0028178363644760689,  -0.004375504188621755,
    -0.03401045154900828,    0.0082648485549758777,   0.046941294075498366,
    0.0082648485549758777,   0.030458671268292565,    -0.0061367662271508,
    0.046941294075498366,    -0.0061367662271508,     -0.0022705688763357437,
    -0.033872115747831257,   -0.0036922586722540614,  -0.04933928804072818,
    -0.0036922586722540614,  0.032425270810710623,    -0.0028178363644760689,
    -0.04933928804072818,    -0.0028178363644760689,  -0.0043755041886217559,
    -0.0028837275309965505,  -5.5375015722344526e-05, 1.7525782974090442e-05,
    -5.5375015722344526e-05, -0.0018912574077916454,  -3.1802098710617322e-05,
    1.7525782974090442e-05,  -3.1802098710617322e-05, -0.0010473645727089257,
    -0.071244372107177281,   0.0032869810406850873,   -0.004391365827421596,
    0.0032869810406850873,   0.032355989842891994,    0.00011722143293728377,
    -0.004391365827421596,   0.00011722143293728377,  0.033066033057125806,
    -0.03401045154900828,    0.0082648485549758777,   0.046941294075498366,
    0.0082648485549758777,   0.030458671268292565,    -0.0061367662271508,
    0.046941294075498366,    -0.0061367662271508,     -0.0022705688763357454,
    -0.071244372107177281,   0.0032869810406850873,   -0.004391365827421596,
    0.0032869810406850873,   0.032355989842891994,    0.00011722143293728377,
    -0.004391365827421596,   0.00011722143293728377,  0.033066033057125806,
    -0.0028837119476513222,  -5.5374170450080959e-05, 1.7525583590162085e-05,
    -5.5374170450080959e-05, -0.0018912603257164665,  -3.1801794260615959e-05,
    1.7525583590162085e-05,  -3.1801794260615959e-05, -0.00104737723812933};

constexpr std::array<double, 27> kDipolePotential{
    -0.0007024349952639715, 0.0012721818165863106, 0.006033277078153213,
    0.0013341583653827024,  0.000775714087941398,  0.00561746840433677,
    -0.002305222489404829,  0.0010667946758256613, 0.006838807093233201};
constexpr std::array<double, 18> kQuadrupolePotential{
    0.0020085717968904688,  -0.000337063537130547,  -0.002197392680570855,   0.0001746301690706618,
    0.0005554430086804973,  0.00018882085900704542, 0.00021499008475039689,  -0.0007066443179647222,
    0.0011453707095440424,  -0.0062869824024671915, -0.00038195475754525466, -0.0013603607787220026,
    0.00020192683856040441, 0.0008857712813144607,  0.0010156357206469835,   0.006550867455655187,
    -0.000823710691531879,  -0.0012175625457222564};
constexpr std::array<double, 3> kChargePotential{2.130078975303285e-05, 0.0001562706345055011,
                                                 0.00023575061083838764};

constexpr std::array<double, 9> kFullGradient{
    3.3795860807527104e-05, 2.17372607545797e-06,   4.633127564116167e-05,
    1.2724906923336276e-05, 7.77151785040006e-06,   -7.328535673281291e-06,
    -4.652076773086339e-05, -9.945243925858033e-06, -3.9002739967880376e-05};
constexpr std::array<double, 9> kFullStrain{
    7.749959022420015e-05, 8.489278595247059e-06,   4.815423885651584e-06,
    8.489278595247059e-06, -1.236659222077197e-06,  -4.1992305966866625e-06,
    4.815423885651583e-06, -4.1992305966866625e-06, -4.543710169800867e-05};

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

struct AlignedWorkspace {
  explicit AlignedWorkspace(std::size_t bytes = 1u) : storage(bytes + 63u) {
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(storage.data());
    data = reinterpret_cast<void*>((address + 63u) & ~std::uintptr_t{63u});
  }

  std::vector<std::byte> storage;
  void* data = nullptr;
};

struct Fixture {
  BasisPlan basis;
  AES2Plan aes2;
  PeriodicShortRangePlan topology;
  PeriodicMultipolePlan multipole;
  CoordinationPlan coordination;
  AlignedWorkspace topology_storage;
  PeriodicShortRangeWorkspace topology_workspace;
  PeriodicShortRangeGeometry geometry;

  bool initialize(std::string& error) {
    if (make_basis_plan(1, 3, kAtomOffsets.data(), kAtomicNumbers.data(), basis, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        make_aes2_plan(basis, kAtomicNumbers.data(), aes2, error) != XTBLOOM_STATUS_SUCCESS ||
        make_periodic_short_range_plan(1, 3, kAtomOffsets.data(), kCell.data(), topology, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        make_periodic_multipole_plan(aes2, topology, multipole, error) != XTBLOOM_STATUS_SUCCESS ||
        make_coordination_plan(1, 3, kAtomOffsets.data(), kAtomicNumbers.data(), coordination,
                               error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    topology_storage = AlignedWorkspace(topology.workspace_size_bytes());
    if (bind_periodic_short_range_workspace(topology, topology_storage.data,
                                            topology.workspace_size_bytes(), topology_workspace,
                                            error) != XTBLOOM_STATUS_SUCCESS ||
        update_periodic_short_range_geometry_cpu(topology, kPositions.data(), 1u,
                                                 topology_workspace, geometry,
                                                 error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    return true;
  }
};

struct Values {
  std::array<double, 27> charge_dipole{};
  std::array<double, 81> dipole_dipole{};
  std::array<double, 54> charge_quadrupole{};
  std::array<double, 3> charge_potential{};
  std::array<double, 9> dipole_potential{};
  std::array<double, 18> quadrupole_potential{};
  std::array<double, 1> energy{};
  std::array<double, 9> gradient{};
  std::array<double, 9> strain{};
  std::array<double, 3> coordination_adjoint{};
};

bool evaluate(const Fixture& fixture, const double* positions, const double* coordination,
              Values& values, std::string& error) {
  return evaluate_periodic_multipole_cpu(
             fixture.multipole, positions, coordination, kCharges.data(), kDipoles.data(),
             kQuadrupoles.data(), values.charge_dipole.data(), values.dipole_dipole.data(),
             values.charge_quadrupole.data(), values.charge_potential.data(),
             values.dipole_potential.data(), values.quadrupole_potential.data(),
             values.energy.data(), values.gradient.data(), values.strain.data(),
             values.coordination_adjoint.data(), error) == XTBLOOM_STATUS_SUCCESS;
}

int test_periodic_multipole_fixture() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(error));
  CHECK(near(fixture.multipole.alpha(0), 0.1875, 0.0));

  Values values;
  CHECK(evaluate(fixture, kPositions.data(), kCoordination.data(), values, error));
  CHECK(near(values.energy[0], 3.188846040354557e-05, 5.0e-12));
  for (std::size_t i = 0; i < values.charge_dipole.size(); ++i)
    CHECK(near(values.charge_dipole[i], kChargeDipoleMatrix[i], 5.0e-12));
  for (std::size_t i = 0; i < values.charge_quadrupole.size(); ++i)
    CHECK(near(values.charge_quadrupole[i], kChargeQuadrupoleMatrix[i], 5.0e-12));
  for (std::size_t i = 0; i < values.dipole_dipole.size(); ++i)
    CHECK(near(values.dipole_dipole[i], kDipoleDipoleMatrix[i], 5.0e-12));
  for (std::size_t i = 0; i < values.charge_potential.size(); ++i)
    CHECK(near(values.charge_potential[i], kChargePotential[i], 5.0e-12));
  for (std::size_t i = 0; i < values.dipole_potential.size(); ++i)
    CHECK(near(values.dipole_potential[i], kDipolePotential[i], 5.0e-12));
  for (std::size_t i = 0; i < values.quadrupole_potential.size(); ++i)
    CHECK(near(values.quadrupole_potential[i], kQuadrupolePotential[i], 5.0e-12));

  /* A matrix contraction is an independent check of the asymmetric charge
   * potential layout and the half-weighted d-d quadratic form. */
  double contracted_energy = 0.0;
  for (std::size_t row = 0; row < 3u; ++row) {
    for (std::size_t column = 0; column < 3u; ++column) {
      const std::size_t pair = row * 3u + column;
      double dipole_term = 0.0;
      double dd_term = 0.0;
      double quadrupole_term = 0.0;
      for (std::size_t component = 0; component < 3u; ++component) {
        dipole_term += kDipoles[row * 3u + component] * values.charge_dipole[pair * 3u + component];
        for (std::size_t other = 0; other < 3u; ++other) {
          dd_term += kDipoles[row * 3u + component] *
                     values.dipole_dipole[pair * 9u + component * 3u + other] *
                     kDipoles[column * 3u + other];
        }
      }
      for (std::size_t component = 0; component < 6u; ++component)
        quadrupole_term +=
            kQuadrupoles[row * 6u + component] * values.charge_quadrupole[pair * 6u + component];
      contracted_energy +=
          dipole_term * kCharges[column] + 0.5 * dd_term + quadrupole_term * kCharges[column];
    }
  }
  for (std::size_t atom = 0; atom < 3u; ++atom) {
    contracted_energy +=
        fixture.aes2.dipole_kernel()[atom] * (kDipoles[atom * 3u] * kDipoles[atom * 3u] +
                                              kDipoles[atom * 3u + 1u] * kDipoles[atom * 3u + 1u] +
                                              kDipoles[atom * 3u + 2u] * kDipoles[atom * 3u + 2u]);
    const double* quadrupole = kQuadrupoles.data() + atom * 6u;
    contracted_energy += fixture.aes2.quadrupole_kernel()[atom] *
                         (quadrupole[0] * quadrupole[0] + 2.0 * quadrupole[1] * quadrupole[1] +
                          quadrupole[2] * quadrupole[2] + 2.0 * quadrupole[3] * quadrupole[3] +
                          2.0 * quadrupole[4] * quadrupole[4] + quadrupole[5] * quadrupole[5]);
  }
  CHECK(near(values.energy[0], contracted_energy, 5.0e-12));

  /* The explicit derivative keeps CN fixed.  Central differences therefore
   * validate the fixed-radius gradient without relying on the same analytic
   * derivative routine. */
  constexpr double step = 1.0e-5;
  for (std::size_t coordinate = 0; coordinate < kPositions.size(); ++coordinate) {
    std::array<double, 9> plus = kPositions;
    std::array<double, 9> minus = kPositions;
    plus[coordinate] += step;
    minus[coordinate] -= step;
    Values plus_values;
    Values minus_values;
    CHECK(evaluate(fixture, plus.data(), kCoordination.data(), plus_values, error));
    CHECK(evaluate(fixture, minus.data(), kCoordination.data(), minus_values, error));
    const double finite_difference =
        (plus_values.energy[0] - minus_values.energy[0]) / (2.0 * step);
    CHECK(near(values.gradient[coordinate], finite_difference, 5.0e-9));
  }

  /* Rewrapping after a whole-cell translation must leave the matrix and
   * energy unchanged. */
  std::array<double, 9> translated = kPositions;
  for (std::size_t atom = 0; atom < 3u; ++atom) {
    for (std::size_t component = 0; component < 3u; ++component)
      translated[atom * 3u + component] += kCell[component];
  }
  Values translated_values;
  CHECK(evaluate(fixture, translated.data(), kCoordination.data(), translated_values, error));
  CHECK(near(translated_values.energy[0], values.energy[0], 5.0e-12));
  for (std::size_t i = 0; i < values.charge_dipole.size(); ++i)
    CHECK(near(translated_values.charge_dipole[i], values.charge_dipole[i], 5.0e-12));

  /* Contract the radius adjoint through the independent periodic CN model
   * and compare the complete tblite term oracle. */
  std::array<double, 3> coordination{};
  CHECK(evaluate_periodic_coordination_cpu(fixture.coordination, fixture.topology, fixture.geometry,
                                           coordination.data(), fixture.topology_workspace,
                                           error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t atom = 0; atom < coordination.size(); ++atom)
    CHECK(near(coordination[atom], kCoordination[atom], 5.0e-12));
  std::array<double, 9> coordination_gradient{};
  std::array<double, 9> coordination_strain{};
  CHECK(add_periodic_coordination_gradient_cpu(
            fixture.coordination, fixture.topology, fixture.geometry,
            values.coordination_adjoint.data(), coordination_gradient.data(),
            coordination_strain.data(), fixture.topology_workspace,
            error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t i = 0; i < values.gradient.size(); ++i)
    CHECK(near(values.gradient[i] + coordination_gradient[i], kFullGradient[i], 5.0e-11));
  for (std::size_t i = 0; i < values.strain.size(); ++i)
    CHECK(near(values.strain[i] + coordination_strain[i], kFullStrain[i], 5.0e-11));
  return 0;
}

}  // namespace

int main() { return test_periodic_multipole_fixture(); }
