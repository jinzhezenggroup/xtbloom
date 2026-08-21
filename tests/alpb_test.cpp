#include "model/gfn2/alpb.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

using xtbloom::detail::gfn2::AlpbBornKernel;
using xtbloom::detail::gfn2::AlpbPolarModel;
using xtbloom::detail::gfn2::AlpbPolarSettings;

constexpr std::int64_t kAtomCount = 3;
constexpr std::array<double, 9> kPositions{
    0.2, -0.4, 0.1, 1.7, 0.3, -0.2, -0.8, 1.4, 0.9,
};
constexpr std::array<double, 3> kBornRadii{2.1, 2.4, 1.8};
constexpr std::array<double, 3> kCavityRadii{2.7, 3.0, 2.2};
constexpr std::array<double, 3> kCharges{0.35, -0.55, 0.30};

bool near(double actual, double expected, double absolute_tolerance,
          double relative_tolerance = 0.0) {
  return std::abs(actual - expected) <=
         absolute_tolerance + relative_tolerance * std::abs(expected);
}

AlpbPolarSettings alpb_settings() { return {AlpbPolarModel::kAlpb, AlpbBornKernel::kP16, 37.5}; }

AlpbPolarSettings gbsa_settings() { return {AlpbPolarModel::kGbsa, AlpbBornKernel::kStill, 37.5}; }

std::array<AlpbPolarSettings, 4> all_model_kernel_combinations() {
  return {{
      {AlpbPolarModel::kAlpb, AlpbBornKernel::kP16, 37.5},
      {AlpbPolarModel::kAlpb, AlpbBornKernel::kStill, 37.5},
      {AlpbPolarModel::kGbsa, AlpbBornKernel::kP16, 37.5},
      {AlpbPolarModel::kGbsa, AlpbBornKernel::kStill, 37.5},
  }};
}

bool evaluate(const std::array<double, 9>& positions, const AlpbPolarSettings& settings,
              double& energy, std::array<double, 9>* matrix_out,
              std::array<double, 3>* potential_out, std::string& error) {
  std::array<double, 9> matrix{};
  std::array<double, 3> potential{};
  if (xtbloom::detail::gfn2::build_alpb_polar_matrix_cpu(
          kAtomCount, positions.data(), kBornRadii.data(), kCavityRadii.data(), settings,
          matrix.data(), error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::evaluate_alpb_polar_cpu(kAtomCount, matrix.data(), kCharges.data(),
                                                     potential.data(), &energy,
                                                     error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  if (matrix_out != nullptr) {
    *matrix_out = matrix;
  }
  if (potential_out != nullptr) {
    *potential_out = potential;
  }
  return true;
}

int test_high_precision_term_oracles() {
  /*
   * These binary64 references were independently evaluated at 80 decimal
   * digits from the equations in pinned tblite revision 133f91ef. They do not
   * depend on xTBloom or tblite binary64 arithmetic.
   */
  constexpr std::array<double, 9> expected_alpb_matrix{
      -0.46113082323943967324, -0.35446532979238465716, -0.34232019667939763223,
      -0.35446532979238465716, -0.40406388157678088825, -0.28166688030906519178,
      -0.34232019667939763223, -0.28166688030906519178, -0.53722007878965138657,
  };
  constexpr std::array<double, 3> expected_alpb_potential{
      -0.069135915751811613864,
      0.013672205347175300994,
      -0.12606130830469873177,
  };
  constexpr double expected_alpb_energy = -0.034767837972745049965;
  constexpr std::array<double, 9> expected_gbsa_matrix{
      -0.46349206349206349206, -0.36249043667099947933, -0.35266390786397574561,
      -0.36249043667099947933, -0.40555555555555555556, -0.28973300641666582735,
      -0.35266390786397574561, -0.28973300641666582735, -0.54074074074074074074,
  };
  constexpr std::array<double, 3> expected_gbsa_potential{
      -0.068651654412365232271,
      0.0092640007957059895841,
      -0.12630143644544752814,
  };
  constexpr double expected_gbsa_energy = -0.033506855207800192005;

  CHECK(xtbloom::detail::gfn2::kAlpbAlpha == 0.571412);
  for (const auto& oracle : std::array{
           std::tuple{alpb_settings(), expected_alpb_matrix, expected_alpb_potential,
                      expected_alpb_energy},
           std::tuple{gbsa_settings(), expected_gbsa_matrix, expected_gbsa_potential,
                      expected_gbsa_energy},
       }) {
    const auto& [settings, expected_matrix, expected_potential, expected_energy] = oracle;
    std::string error;
    std::array<double, 9> matrix{};
    std::array<double, 3> potential{};
    double energy = 0.0;
    CHECK(evaluate(kPositions, settings, energy, &matrix, &potential, error));
    CHECK(error.empty());
    for (std::size_t row = 0; row < 3u; ++row) {
      for (std::size_t column = 0; column < 3u; ++column) {
        CHECK(matrix[row * 3u + column] == matrix[column * 3u + row]);
        CHECK(near(matrix[row * 3u + column], expected_matrix[row * 3u + column], 3.0e-16));
      }
      CHECK(near(potential[row], expected_potential[row], 3.0e-16));
    }
    CHECK(near(energy, expected_energy, 3.0e-16));

    double identity_energy = 0.0;
    for (std::size_t atom = 0; atom < 3u; ++atom) {
      double identity_potential = 0.0;
      for (std::size_t other = 0; other < 3u; ++other) {
        identity_potential += matrix[atom * 3u + other] * kCharges[other];
      }
      CHECK(near(potential[atom], identity_potential, 2.0e-16));
      identity_energy += 0.5 * kCharges[atom] * potential[atom];
    }
    CHECK(near(energy, identity_energy, 2.0e-16));
  }
  return 0;
}

int test_charge_derivative_is_potential() {
  constexpr std::array<double, 3> steps{2.0e-4, 5.0e-5, 1.0e-5};
  for (const AlpbPolarSettings settings : all_model_kernel_combinations()) {
    std::string error;
    std::array<double, 9> matrix{};
    CHECK(xtbloom::detail::gfn2::build_alpb_polar_matrix_cpu(
              kAtomCount, kPositions.data(), kBornRadii.data(), kCavityRadii.data(), settings,
              matrix.data(), error) == XTBLOOM_STATUS_SUCCESS);
    std::array<double, 3> potential{};
    double energy = 0.0;
    CHECK(xtbloom::detail::gfn2::evaluate_alpb_polar_cpu(kAtomCount, matrix.data(), kCharges.data(),
                                                         potential.data(), &energy,
                                                         error) == XTBLOOM_STATUS_SUCCESS);
    for (const double step : steps) {
      for (std::size_t atom = 0; atom < 3u; ++atom) {
        auto plus_charges = kCharges;
        auto minus_charges = kCharges;
        plus_charges[atom] += step;
        minus_charges[atom] -= step;
        std::array<double, 3> scratch{};
        double plus = 0.0;
        double minus = 0.0;
        CHECK(xtbloom::detail::gfn2::evaluate_alpb_polar_cpu(
                  kAtomCount, matrix.data(), plus_charges.data(), scratch.data(), &plus, error) ==
              XTBLOOM_STATUS_SUCCESS);
        CHECK(xtbloom::detail::gfn2::evaluate_alpb_polar_cpu(
                  kAtomCount, matrix.data(), minus_charges.data(), scratch.data(), &minus, error) ==
              XTBLOOM_STATUS_SUCCESS);
        CHECK(near((plus - minus) / (2.0 * step), potential[atom], 2.0e-12));
      }
    }
  }
  return 0;
}

int test_analytic_coordinate_derivatives() {
  constexpr std::array<double, 3> steps{2.0e-4, 5.0e-5, 1.0e-5};
  for (const AlpbPolarSettings settings : all_model_kernel_combinations()) {
    std::string error;
    constexpr double baseline = 0.125;
    std::array<double, 9> gradients{};
    gradients.fill(baseline);
    CHECK(xtbloom::detail::gfn2::add_alpb_polar_gradient_cpu(
              kAtomCount, kPositions.data(), kBornRadii.data(), kCavityRadii.data(),
              kCharges.data(), settings, gradients.data(), error) == XTBLOOM_STATUS_SUCCESS);
    for (const double step : steps) {
      for (std::size_t coordinate = 0; coordinate < kPositions.size(); ++coordinate) {
        auto plus_positions = kPositions;
        auto minus_positions = kPositions;
        plus_positions[coordinate] += step;
        minus_positions[coordinate] -= step;
        double plus = 0.0;
        double minus = 0.0;
        CHECK(evaluate(plus_positions, settings, plus, nullptr, nullptr, error));
        CHECK(evaluate(minus_positions, settings, minus, nullptr, nullptr, error));
        CHECK(near((plus - minus) / (2.0 * step), gradients[coordinate] - baseline, 8.0e-9));
      }
    }
  }
  return 0;
}

int test_translation_rotation_and_net_gradient() {
  for (const AlpbPolarSettings settings : all_model_kernel_combinations()) {
    std::string error;
    std::array<double, 9> matrix{};
    std::array<double, 3> potential{};
    double energy = 0.0;
    CHECK(evaluate(kPositions, settings, energy, &matrix, &potential, error));
    std::array<double, 9> gradients{};
    CHECK(xtbloom::detail::gfn2::add_alpb_polar_gradient_cpu(
              kAtomCount, kPositions.data(), kBornRadii.data(), kCavityRadii.data(),
              kCharges.data(), settings, gradients.data(), error) == XTBLOOM_STATUS_SUCCESS);

    std::array<double, 9> transformed_positions{};
    for (std::size_t atom = 0; atom < 3u; ++atom) {
      const double x = kPositions[atom * 3u];
      const double y = kPositions[atom * 3u + 1u];
      const double z = kPositions[atom * 3u + 2u];
      transformed_positions[atom * 3u] = -y + 4.25;
      transformed_positions[atom * 3u + 1u] = x - 2.75;
      transformed_positions[atom * 3u + 2u] = z + 0.65;
    }
    std::array<double, 9> transformed_matrix{};
    std::array<double, 3> transformed_potential{};
    double transformed_energy = 0.0;
    CHECK(evaluate(transformed_positions, settings, transformed_energy, &transformed_matrix,
                   &transformed_potential, error));
    std::array<double, 9> transformed_gradients{};
    CHECK(xtbloom::detail::gfn2::add_alpb_polar_gradient_cpu(
              kAtomCount, transformed_positions.data(), kBornRadii.data(), kCavityRadii.data(),
              kCharges.data(), settings, transformed_gradients.data(),
              error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(near(transformed_energy, energy, 3.0e-17, 1.0e-14));
    for (std::size_t element = 0; element < matrix.size(); ++element) {
      CHECK(near(transformed_matrix[element], matrix[element], 3.0e-16));
    }
    for (std::size_t atom = 0; atom < 3u; ++atom) {
      CHECK(near(transformed_potential[atom], potential[atom], 3.0e-16));
      CHECK(near(transformed_gradients[atom * 3u], -gradients[atom * 3u + 1u], 3.0e-16));
      CHECK(near(transformed_gradients[atom * 3u + 1u], gradients[atom * 3u], 3.0e-16));
      CHECK(near(transformed_gradients[atom * 3u + 2u], gradients[atom * 3u + 2u], 3.0e-16));
    }

    for (std::size_t axis = 0; axis < 3u; ++axis) {
      double net_gradient = 0.0;
      for (std::size_t atom = 0; atom < 3u; ++atom) {
        net_gradient += gradients[atom * 3u + axis];
      }
      CHECK(near(net_gradient, 0.0, 3.0e-17));
    }
    std::array<double, 3> torque{};
    for (std::size_t atom = 0; atom < 3u; ++atom) {
      const double x = kPositions[atom * 3u];
      const double y = kPositions[atom * 3u + 1u];
      const double z = kPositions[atom * 3u + 2u];
      const double gx = gradients[atom * 3u];
      const double gy = gradients[atom * 3u + 1u];
      const double gz = gradients[atom * 3u + 2u];
      torque[0] += y * gz - z * gy;
      torque[1] += z * gx - x * gz;
      torque[2] += x * gy - y * gx;
    }
    for (const double component : torque) {
      CHECK(near(component, 0.0, 8.0e-17));
    }
  }
  return 0;
}

int test_single_atom_and_vacuum_limits() {
  constexpr std::array<double, 3> position{1.0, -2.0, 0.5};
  constexpr std::array<double, 1> born_radius{2.25};
  constexpr std::array<double, 1> cavity_radius{2.75};
  constexpr std::array<double, 1> charge{-0.4};

  for (AlpbPolarSettings settings : all_model_kernel_combinations()) {
    std::string error;
    std::array<double, 1> matrix{};
    CHECK(xtbloom::detail::gfn2::build_alpb_polar_matrix_cpu(
              1, position.data(), born_radius.data(), cavity_radius.data(), settings, matrix.data(),
              error) == XTBLOOM_STATUS_SUCCESS);
    std::array<double, 1> potential{};
    double energy = 0.0;
    CHECK(xtbloom::detail::gfn2::evaluate_alpb_polar_cpu(1, matrix.data(), charge.data(),
                                                         potential.data(), &energy,
                                                         error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(near(potential[0], matrix[0] * charge[0], 1.0e-16));
    CHECK(near(energy, 0.5 * charge[0] * potential[0], 1.0e-16));
    std::array<double, 3> gradient{};
    CHECK(xtbloom::detail::gfn2::add_alpb_polar_gradient_cpu(
              1, position.data(), born_radius.data(), cavity_radius.data(), charge.data(), settings,
              gradient.data(), error) == XTBLOOM_STATUS_SUCCESS);
    for (const double component : gradient) {
      CHECK(component == 0.0);
    }

    settings.dielectric_constant = 1.0;
    CHECK(xtbloom::detail::gfn2::build_alpb_polar_matrix_cpu(
              1, position.data(), born_radius.data(), cavity_radius.data(), settings, matrix.data(),
              error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(matrix[0] == 0.0);
  }
  return 0;
}

int test_empty_and_coincident_limits() {
  std::string error = "stale";
  double energy = 7.0;
  CHECK(xtbloom::detail::gfn2::build_alpb_polar_matrix_cpu(0, nullptr, nullptr, nullptr,
                                                           alpb_settings(), nullptr,
                                                           error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(error.empty());
  CHECK(xtbloom::detail::gfn2::evaluate_alpb_polar_cpu(0, nullptr, nullptr, nullptr, &energy,
                                                       error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(energy == 0.0);
  CHECK(xtbloom::detail::gfn2::add_alpb_polar_gradient_cpu(0, nullptr, nullptr, nullptr, nullptr,
                                                           alpb_settings(), nullptr,
                                                           error) == XTBLOOM_STATUS_SUCCESS);

  auto coincident = kPositions;
  coincident[3] = coincident[0];
  coincident[4] = coincident[1];
  coincident[5] = coincident[2];
  std::array<double, 9> matrix{};
  CHECK(xtbloom::detail::gfn2::build_alpb_polar_matrix_cpu(
            kAtomCount, coincident.data(), kBornRadii.data(), kCavityRadii.data(), alpb_settings(),
            matrix.data(), error) == XTBLOOM_STATUS_SUCCESS);
  for (const double element : matrix) {
    CHECK(std::isfinite(element));
  }

  std::array<double, 9> gradients{};
  CHECK(xtbloom::detail::gfn2::add_alpb_polar_gradient_cpu(
            kAtomCount, coincident.data(), kBornRadii.data(), nullptr, kCharges.data(),
            gbsa_settings(), gradients.data(), error) == XTBLOOM_STATUS_SUCCESS);
  for (const double component : gradients) {
    CHECK(std::isfinite(component));
  }
  gradients.fill(41.0);
  CHECK(xtbloom::detail::gfn2::add_alpb_polar_gradient_cpu(
            kAtomCount, coincident.data(), kBornRadii.data(), kCavityRadii.data(), kCharges.data(),
            alpb_settings(), gradients.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  for (const double component : gradients) {
    CHECK(component == 41.0);
  }

  auto near_coincident = coincident;
  near_coincident[3] += 1.0e-15;
  gradients.fill(0.0);
  CHECK(xtbloom::detail::gfn2::add_alpb_polar_gradient_cpu(
            kAtomCount, near_coincident.data(), kBornRadii.data(), kCavityRadii.data(),
            kCharges.data(), alpb_settings(), gradients.data(), error) == XTBLOOM_STATUS_SUCCESS);
  for (const double component : gradients) {
    CHECK(std::isfinite(component));
  }
  return 0;
}

int test_validation_is_non_mutating() {
  std::string error;
  std::array<double, 9> matrix{};
  matrix.fill(19.0);
  CHECK(xtbloom::detail::gfn2::build_alpb_polar_matrix_cpu(-1, nullptr, nullptr, nullptr,
                                                           alpb_settings(), nullptr, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::build_alpb_polar_matrix_cpu(
            kAtomCount, kPositions.data(), kBornRadii.data(), kCavityRadii.data(), alpb_settings(),
            nullptr, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::build_alpb_polar_matrix_cpu(
            kAtomCount, kPositions.data(), kBornRadii.data(), nullptr, alpb_settings(),
            matrix.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  auto bad_positions = kPositions;
  bad_positions[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK(xtbloom::detail::gfn2::build_alpb_polar_matrix_cpu(
            kAtomCount, bad_positions.data(), kBornRadii.data(), kCavityRadii.data(),
            alpb_settings(), matrix.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  for (const double element : matrix) {
    CHECK(element == 19.0);
  }

  auto bad_radii = kBornRadii;
  bad_radii[1] = 0.0;
  CHECK(xtbloom::detail::gfn2::build_alpb_polar_matrix_cpu(
            kAtomCount, kPositions.data(), bad_radii.data(), kCavityRadii.data(), alpb_settings(),
            matrix.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  auto bad_cavity_radii = kCavityRadii;
  bad_cavity_radii[2] = std::numeric_limits<double>::infinity();
  CHECK(xtbloom::detail::gfn2::build_alpb_polar_matrix_cpu(
            kAtomCount, kPositions.data(), kBornRadii.data(), bad_cavity_radii.data(),
            alpb_settings(), matrix.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  AlpbPolarSettings bad_settings = alpb_settings();
  bad_settings.dielectric_constant = 0.0;
  CHECK(xtbloom::detail::gfn2::build_alpb_polar_matrix_cpu(
            kAtomCount, kPositions.data(), kBornRadii.data(), kCavityRadii.data(), bad_settings,
            matrix.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  bad_settings = alpb_settings();
  bad_settings.kernel = static_cast<AlpbBornKernel>(99);
  CHECK(xtbloom::detail::gfn2::build_alpb_polar_matrix_cpu(
            kAtomCount, kPositions.data(), kBornRadii.data(), kCavityRadii.data(), bad_settings,
            matrix.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  bad_settings = alpb_settings();
  bad_settings.model = static_cast<AlpbPolarModel>(99);
  CHECK(xtbloom::detail::gfn2::build_alpb_polar_matrix_cpu(
            kAtomCount, kPositions.data(), kBornRadii.data(), kCavityRadii.data(), bad_settings,
            matrix.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  bad_settings = alpb_settings();
  bad_settings.dielectric_constant = std::numeric_limits<double>::denorm_min();
  CHECK(xtbloom::detail::gfn2::build_alpb_polar_matrix_cpu(
            kAtomCount, kPositions.data(), kBornRadii.data(), kCavityRadii.data(), bad_settings,
            matrix.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  CHECK(xtbloom::detail::gfn2::build_alpb_polar_matrix_cpu(
            kAtomCount, kPositions.data(), kBornRadii.data(), nullptr, gbsa_settings(),
            matrix.data(), error) == XTBLOOM_STATUS_SUCCESS);
  std::array<double, 3> potential{};
  potential.fill(23.0);
  double energy = 29.0;
  CHECK(xtbloom::detail::gfn2::evaluate_alpb_polar_cpu(-1, nullptr, nullptr, nullptr, &energy,
                                                       error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::evaluate_alpb_polar_cpu(kAtomCount, matrix.data(), kCharges.data(),
                                                       potential.data(), nullptr,
                                                       error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::evaluate_alpb_polar_cpu(kAtomCount, nullptr, kCharges.data(),
                                                       potential.data(), &energy,
                                                       error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  auto bad_charges = kCharges;
  bad_charges[0] = std::numeric_limits<double>::infinity();
  CHECK(xtbloom::detail::gfn2::evaluate_alpb_polar_cpu(
            kAtomCount, matrix.data(), bad_charges.data(), potential.data(), &energy, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energy == 29.0);
  for (const double value : potential) {
    CHECK(value == 23.0);
  }

  auto bad_matrix = matrix;
  bad_matrix[4] = std::numeric_limits<double>::quiet_NaN();
  CHECK(xtbloom::detail::gfn2::evaluate_alpb_polar_cpu(kAtomCount, bad_matrix.data(),
                                                       kCharges.data(), potential.data(), &energy,
                                                       error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energy == 29.0);
  for (const double value : potential) {
    CHECK(value == 23.0);
  }

  std::array<double, 1> unstable_matrix{std::numeric_limits<double>::max()};
  std::array<double, 1> unstable_charge{2.0};
  std::array<double, 1> unstable_potential{43.0};
  double unstable_energy = 47.0;
  CHECK(xtbloom::detail::gfn2::evaluate_alpb_polar_cpu(
            1, unstable_matrix.data(), unstable_charge.data(), unstable_potential.data(),
            &unstable_energy, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(unstable_potential[0] == 43.0);
  CHECK(unstable_energy == 47.0);

  constexpr std::array<double, 3> single_position{0.0, 0.0, 0.0};
  constexpr std::array<double, 1> unstable_radius{std::numeric_limits<double>::denorm_min()};
  std::array<double, 1> single_matrix{53.0};
  CHECK(xtbloom::detail::gfn2::build_alpb_polar_matrix_cpu(
            1, single_position.data(), unstable_radius.data(), nullptr, gbsa_settings(),
            single_matrix.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(single_matrix[0] == 53.0);

  std::array<double, 9> gradients{};
  gradients.fill(31.0);
  CHECK(xtbloom::detail::gfn2::add_alpb_polar_gradient_cpu(-1, nullptr, nullptr, nullptr, nullptr,
                                                           alpb_settings(), nullptr, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::add_alpb_polar_gradient_cpu(
            kAtomCount, kPositions.data(), kBornRadii.data(), kCavityRadii.data(), nullptr,
            alpb_settings(), gradients.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::add_alpb_polar_gradient_cpu(
            kAtomCount, kPositions.data(), kBornRadii.data(), kCavityRadii.data(), kCharges.data(),
            alpb_settings(), nullptr, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::add_alpb_polar_gradient_cpu(
            kAtomCount, kPositions.data(), kBornRadii.data(), kCavityRadii.data(),
            bad_charges.data(), alpb_settings(), gradients.data(),
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  for (const double value : gradients) {
    CHECK(value == 31.0);
  }

  gradients.fill(59.0);
  auto unstable_charges = kCharges;
  unstable_charges[0] = std::numeric_limits<double>::max();
  unstable_charges[1] = std::numeric_limits<double>::max();
  CHECK(xtbloom::detail::gfn2::add_alpb_polar_gradient_cpu(
            kAtomCount, kPositions.data(), kBornRadii.data(), kCavityRadii.data(),
            unstable_charges.data(), alpb_settings(), gradients.data(),
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  for (const double value : gradients) {
    CHECK(value == 59.0);
  }

  gradients.fill(61.0);
  gradients[4] = std::numeric_limits<double>::infinity();
  CHECK(xtbloom::detail::gfn2::add_alpb_polar_gradient_cpu(
            kAtomCount, kPositions.data(), kBornRadii.data(), kCavityRadii.data(), kCharges.data(),
            alpb_settings(), gradients.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::isinf(gradients[4]));
  CHECK(gradients[0] == 61.0);
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_high_precision_term_oracles(); status != 0) {
    return status;
  }
  if (const int status = test_charge_derivative_is_potential(); status != 0) {
    return status;
  }
  if (const int status = test_analytic_coordinate_derivatives(); status != 0) {
    return status;
  }
  if (const int status = test_translation_rotation_and_net_gradient(); status != 0) {
    return status;
  }
  if (const int status = test_single_atom_and_vacuum_limits(); status != 0) {
    return status;
  }
  if (const int status = test_empty_and_coincident_limits(); status != 0) {
    return status;
  }
  return test_validation_is_non_mutating();
}
