#include "model/gfn2/integrals.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "model/gfn2/basis.hpp"

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

struct Evaluation {
  gpuxtb::detail::gfn2::BasisPlan basis;
  gpuxtb::detail::gfn2::IntegralPlan integrals;
  std::vector<double> workspace;
  std::vector<double> overlap;
};

bool evaluate(std::int64_t batch_size, const std::vector<std::int64_t>& atom_offsets,
              const std::vector<std::int32_t>& atomic_numbers, const std::vector<double>& positions,
              Evaluation& evaluation, std::string& error) {
  if (gpuxtb::detail::gfn2::make_basis_plan(
          batch_size, static_cast<std::int64_t>(atomic_numbers.size()), atom_offsets.data(),
          atomic_numbers.data(), evaluation.basis, error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  if (gpuxtb::detail::gfn2::make_integral_plan(evaluation.basis, evaluation.integrals, error) !=
      GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  evaluation.workspace.resize((evaluation.integrals.workspace_size_bytes + sizeof(double) - 1u) /
                              sizeof(double));
  evaluation.overlap.resize(static_cast<std::size_t>(evaluation.integrals.total_matrix_elements));
  return gpuxtb::detail::gfn2::evaluate_overlap_cpu(
             evaluation.basis, evaluation.integrals, positions.data(), evaluation.overlap.data(),
             evaluation.workspace.data(), evaluation.workspace.size() * sizeof(double),
             error) == GPUXTB_STATUS_SUCCESS;
}

int test_tblite_si_reference_and_spherical_conventions() {
  const std::vector<std::int64_t> atom_offsets{0, 2};
  const std::vector<std::int32_t> atomic_numbers{14, 14};
  const std::vector<double> positions{0.0, 0.0, 0.0, 1.1, -0.7, 2.3};
  Evaluation evaluation;
  std::string error;
  CHECK(evaluate(1, atom_offsets, atomic_numbers, positions, evaluation, error));
  CHECK(error.empty());
  CHECK(evaluation.basis.total_orbitals == 18);
  CHECK(evaluation.integrals.matrix_offsets == std::vector<std::int64_t>({0, 324}));

  /*
   * Independent golden from tblite 0.6.0's exported overlap_cgto at commit
   * d4cac072c3e78cc27892d8b8d5101a3453ad2374. The overlap kernel is
   * numerically unchanged at the v0.7 commit pinned for GFN2 parameters. The
   * reference was fed tblite_basis_slater's normalized Si 3s/3p/3d
   * contractions, with the bra at the origin and ket at (1.1,-0.7,2.3) bohr.
   * This full 9-by-9 block covers every distinct off-center shell combination
   * and fixes tblite's real-spherical signs and [-l,...,+l] ordering.
   */
  constexpr std::array<double, 81> reference{
      0.54617816196523727,     0.13566337336125808,     -0.44575108390127643,
      -0.21318530099626273,    -0.067277171181478468,   -0.14067044883400037,
      0.22397489407821741,     0.22105356245342919,     0.031454261851080904,
      -0.13566337336125808,    0.35821558664914865,     0.15849301848283032,
      0.075801008839614506,    -0.20927692377472895,    -0.4375790224380694,
      -0.17241359780172544,    -0.084907871061414175,   -0.16170258406304963,
      0.44575108390127649,     0.15849301848283032,     -0.11431018274065598,
      -0.2490604576158762,     -0.084907871061414175,   -0.027913807832993843,
      -0.28499393479381679,    0.043864555166133211,    0.03969718647027163,
      0.21318530099626273,     0.075801008839614506,    -0.2490604576158762,
      0.28733672124067788,     0.10901271941229063,     -0.084907871061414161,
      0.27093565368842576,     -0.35818464949752632,    -0.21613283882937723,
      -0.067277171181478454,   0.20927692377472892,     0.084907871061414175,
      -0.10901271941229063,    0.35534386652711508,     -0.20455780787035371,
      -0.12086122992060848,    0.1066124139019308,      -0.0056340891776355791,
      -0.14067044883400037,    0.43757902243806951,     0.02791380783299385,
      0.084907871061414175,    -0.20455780787035371,    0.025464081502987471,
      0.0005290444284695231,   -0.012864464900685579,   -0.15798790890304715,
      0.22397489407821736,     0.17241359780172538,     0.28499393479381679,
      -0.27093565368842576,    -0.12086122992060847,    0.00052904442846953004,
      -0.060755054941781073,   -0.00083135553045211863, 0.056506549053791058,
      0.22105356245342925,     0.084907871061414175,    -0.043864555166133204,
      0.35818464949752626,     0.1066124139019308,      -0.012864464900685579,
      -0.00083135553045211863, 0.037493191539992185,    -0.21124269939461177,
      0.031454261851080856,    0.16170258406304963,     -0.039697186470271581,
      0.21613283882937723,     -0.0056340891776355791,  -0.15798790890304715,
      0.0565065490537911,      -0.21124269939461177,    0.34592729540157918,
  };

  constexpr std::size_t nao = 18;
  for (std::size_t row = 0; row < 9; ++row) {
    for (std::size_t column = 0; column < 9; ++column) {
      CHECK(
          near(evaluation.overlap[row * nao + 9u + column], reference[row * 9u + column], 7.0e-15));
    }
  }

  /* The same-center s/p/d blocks from the same tblite reference are diagonal. */
  constexpr std::array<double, 9> coincident_diagonal{
      1.0000000000136440,  0.99999999984975119, 0.99999999984975119,
      0.99999999984975119, 0.99999999984681553, 0.99999999984681553,
      0.99999999984681553, 0.99999999984681553, 0.99999999984681542,
  };
  for (std::size_t row = 0; row < 9; ++row) {
    for (std::size_t column = 0; column < 9; ++column) {
      const double expected = row == column ? coincident_diagonal[row] : 0.0;
      CHECK(near(evaluation.overlap[row * nao + column], expected, 2.0e-15));
      CHECK(near(evaluation.overlap[(9u + row) * nao + 9u + column], expected, 2.0e-15));
    }
  }

  /* The packed dense matrix is symmetric to roundoff. */
  for (std::size_t row = 0; row < nao; ++row) {
    for (std::size_t column = 0; column < nao; ++column) {
      CHECK(near(evaluation.overlap[row * nao + column], evaluation.overlap[column * nao + row],
                 2.0e-16));
    }
  }
  return 0;
}

int test_translation_and_rotation_covariance() {
  const std::vector<std::int64_t> atom_offsets{0, 2};
  const std::vector<std::int32_t> atomic_numbers{14, 14};
  const std::vector<double> positions{0.0, 0.0, 0.0, 1.1, -0.7, 2.3};
  Evaluation original;
  std::string error;
  CHECK(evaluate(1, atom_offsets, atomic_numbers, positions, original, error));

  std::vector<double> translated = positions;
  for (std::size_t atom = 0; atom < 2; ++atom) {
    translated[atom * 3u] += 4.25;
    translated[atom * 3u + 1u] -= 1.75;
    translated[atom * 3u + 2u] += 0.625;
  }
  Evaluation shifted;
  CHECK(evaluate(1, atom_offsets, atomic_numbers, translated, shifted, error));
  for (std::size_t element = 0; element < original.overlap.size(); ++element) {
    CHECK(near(shifted.overlap[element], original.overlap[element], 4.0e-16));
  }

  /* Rotate both centers by +90 degrees around z: (x,y,z) -> (-y,x,z). */
  std::vector<double> rotated = positions;
  for (std::size_t atom = 0; atom < 2; ++atom) {
    const double x = positions[atom * 3u];
    const double y = positions[atom * 3u + 1u];
    rotated[atom * 3u] = -y;
    rotated[atom * 3u + 1u] = x;
  }
  Evaluation turned;
  CHECK(evaluate(1, atom_offsets, atomic_numbers, rotated, turned, error));

  /*
   * q(Rr)=U q(r) for tblite's [s,y,z,x,d-2,...,d+2] real harmonics.
   * Therefore the interatomic block obeys B(Rr)=U B(r) U^T.
   */
  constexpr std::array<double, 81> rotation{
      1, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  0,  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0,  0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,  -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1,
  };
  constexpr std::size_t full_nao = 18;
  for (std::size_t row = 0; row < 9; ++row) {
    for (std::size_t column = 0; column < 9; ++column) {
      double expected = 0.0;
      for (std::size_t inner_row = 0; inner_row < 9; ++inner_row) {
        for (std::size_t inner_column = 0; inner_column < 9; ++inner_column) {
          expected += rotation[row * 9u + inner_row] *
                      original.overlap[inner_row * full_nao + 9u + inner_column] *
                      rotation[column * 9u + inner_column];
        }
      }
      CHECK(near(turned.overlap[row * full_nao + 9u + column], expected, 3.0e-15));
    }
  }
  return 0;
}

int test_ragged_batch_equals_sequential() {
  /* Si2, an empty molecule, Si, and OH exercise unequal dense matrix sizes. */
  const std::vector<std::int64_t> offsets{0, 2, 2, 3, 5};
  const std::vector<std::int32_t> atomic_numbers{14, 14, 14, 8, 1};
  const std::vector<double> positions{
      0.0, 0.0, 0.0, 1.1, -0.7, 2.3, -0.2, 0.4, 1.5, 2.0, -1.0, 0.5, 3.2, -0.8, 0.1,
  };
  Evaluation batch;
  std::string error;
  CHECK(evaluate(4, offsets, atomic_numbers, positions, batch, error));
  CHECK(batch.integrals.matrix_offsets == std::vector<std::int64_t>({0, 324, 324, 405, 430}));

  for (std::size_t molecule = 0; molecule < 4; ++molecule) {
    const std::int64_t atom_begin = offsets[molecule];
    const std::int64_t atom_end = offsets[molecule + 1u];
    if (atom_begin == atom_end) {
      continue;
    }
    const std::vector<std::int64_t> sequential_offsets{0, atom_end - atom_begin};
    const std::vector<std::int32_t> sequential_numbers(atomic_numbers.begin() + atom_begin,
                                                       atomic_numbers.begin() + atom_end);
    const std::vector<double> sequential_positions(positions.begin() + atom_begin * 3,
                                                   positions.begin() + atom_end * 3);
    Evaluation sequential;
    CHECK(evaluate(1, sequential_offsets, sequential_numbers, sequential_positions, sequential,
                   error));
    const std::size_t packed_begin =
        static_cast<std::size_t>(batch.integrals.matrix_offsets[molecule]);
    const std::size_t packed_end =
        static_cast<std::size_t>(batch.integrals.matrix_offsets[molecule + 1u]);
    CHECK(packed_end - packed_begin == sequential.overlap.size());
    for (std::size_t element = 0; element < sequential.overlap.size(); ++element) {
      CHECK(batch.overlap[packed_begin + element] == sequential.overlap[element]);
    }
  }
  return 0;
}

bool weighted_overlap(const gpuxtb::detail::gfn2::BasisPlan& basis,
                      const gpuxtb::detail::gfn2::IntegralPlan& integrals,
                      std::vector<double>& workspace, const std::vector<double>& positions,
                      const std::vector<double>& weights, double& value, std::string& error) {
  std::vector<double> overlap(static_cast<std::size_t>(integrals.total_matrix_elements));
  if (gpuxtb::detail::gfn2::evaluate_overlap_cpu(
          basis, integrals, positions.data(), overlap.data(), workspace.data(),
          workspace.size() * sizeof(double), error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  value = 0.0;
  for (std::size_t element = 0; element < overlap.size(); ++element) {
    value += weights[element] * overlap[element];
  }
  return true;
}

int test_analytic_vjp_against_finite_difference() {
  const std::vector<std::int64_t> atom_offsets{0, 2, 4};
  const std::vector<std::int32_t> atomic_numbers{14, 14, 6, 8};
  std::vector<double> positions{
      0.1, -0.2, 0.3, 1.4, -0.8, 2.1, -2.0, 0.7, -0.4, -0.5, 1.1, 0.9,
  };
  Evaluation evaluation;
  std::string error;
  CHECK(evaluate(2, atom_offsets, atomic_numbers, positions, evaluation, error));

  std::vector<double> weights(evaluation.overlap.size());
  for (std::size_t element = 0; element < weights.size(); ++element) {
    weights[element] = std::sin(0.37 * static_cast<double>(element + 1u)) +
                       0.2 * std::cos(0.11 * static_cast<double>(element + 3u));
  }
  constexpr double baseline = 0.125;
  std::vector<double> gradient(positions.size(), baseline);
  CHECK(gpuxtb::detail::gfn2::add_overlap_gradient_cpu(
            evaluation.basis, evaluation.integrals, positions.data(), weights.data(),
            gradient.data(), evaluation.workspace.data(),
            evaluation.workspace.size() * sizeof(double), error) == GPUXTB_STATUS_SUCCESS);

  constexpr double step = 2.0e-5;
  for (std::size_t coordinate = 0; coordinate < positions.size(); ++coordinate) {
    positions[coordinate] += step;
    double right = 0.0;
    CHECK(weighted_overlap(evaluation.basis, evaluation.integrals, evaluation.workspace, positions,
                           weights, right, error));
    positions[coordinate] -= 2.0 * step;
    double left = 0.0;
    CHECK(weighted_overlap(evaluation.basis, evaluation.integrals, evaluation.workspace, positions,
                           weights, left, error));
    positions[coordinate] += step;
    const double numerical = (right - left) / (2.0 * step);
    CHECK(near(gradient[coordinate] - baseline, numerical, 2.0e-9));
  }

  /* Translation invariance requires each molecule's net derivative to vanish. */
  for (std::size_t molecule = 0; molecule < 2; ++molecule) {
    for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
      double sum = 0.0;
      for (std::int64_t atom = atom_offsets[molecule]; atom < atom_offsets[molecule + 1u]; ++atom) {
        sum += gradient[static_cast<std::size_t>(atom) * 3u + coordinate] - baseline;
      }
      CHECK(near(sum, 0.0, 2.0e-15));
    }
  }
  return 0;
}

int test_validation_and_strong_plan_failure_guarantee() {
  constexpr std::array<std::int64_t, 2> offsets{0, 1};
  constexpr std::array<std::int32_t, 1> atomic_numbers{14};
  gpuxtb::detail::gfn2::BasisPlan basis;
  std::string error;
  CHECK(gpuxtb::detail::gfn2::make_basis_plan(1, 1, offsets.data(), atomic_numbers.data(), basis,
                                              error) == GPUXTB_STATUS_SUCCESS);

  gpuxtb::detail::gfn2::IntegralPlan plan;
  plan.batch_size = 17;
  CHECK(gpuxtb::detail::gfn2::make_integral_plan(basis, plan, error, 0.0) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(plan.batch_size == 17);
  CHECK(gpuxtb::detail::gfn2::make_integral_plan(basis, plan, error) == GPUXTB_STATUS_SUCCESS);

  std::array<double, 3> positions{};
  std::vector<double> overlap(static_cast<std::size_t>(plan.total_matrix_elements), 3.0);
  std::vector<double> workspace((plan.workspace_size_bytes + sizeof(double) - 1u) / sizeof(double));
  CHECK(gpuxtb::detail::gfn2::evaluate_overlap_cpu(basis, plan, positions.data(), overlap.data(),
                                                   workspace.data(), plan.workspace_size_bytes - 1u,
                                                   error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(overlap.begin(), overlap.end(), [](double value) { return value == 3.0; }));

  positions[1] = std::numeric_limits<double>::quiet_NaN();
  CHECK(gpuxtb::detail::gfn2::evaluate_overlap_cpu(
            basis, plan, positions.data(), overlap.data(), workspace.data(),
            workspace.size() * sizeof(double), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  positions[1] = 0.0;

  std::vector<double> weights(overlap.size(), 1.0);
  std::array<double, 3> gradients{};
  weights.back() = std::numeric_limits<double>::infinity();
  CHECK(gpuxtb::detail::gfn2::add_overlap_gradient_cpu(
            basis, plan, positions.data(), weights.data(), gradients.data(), workspace.data(),
            workspace.size() * sizeof(double), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK((gradients == std::array<double, 3>{}));
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_tblite_si_reference_and_spherical_conventions(); status != 0) {
    return status;
  }
  if (const int status = test_translation_and_rotation_covariance(); status != 0) {
    return status;
  }
  if (const int status = test_ragged_batch_equals_sequential(); status != 0) {
    return status;
  }
  if (const int status = test_analytic_vjp_against_finite_difference(); status != 0) {
    return status;
  }
  return test_validation_and_strong_plan_failure_guarantee();
}
