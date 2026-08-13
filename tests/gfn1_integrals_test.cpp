#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "model/gfn1/basis.hpp"
#include "model/gfn1/integrals.hpp"

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
  xtbloom::detail::gfn1::BasisPlan basis;
  xtbloom::detail::gfn1::IntegralPlan integrals;
  std::vector<double> overlap;
  std::vector<double> workspace;
};

bool evaluate(std::int64_t batch_size, const std::vector<std::int64_t>& atom_offsets,
              const std::vector<std::int32_t>& atomic_numbers, const std::vector<double>& positions,
              Evaluation& evaluation, std::string& error) {
  if (xtbloom::detail::gfn1::make_basis_plan(batch_size, atomic_numbers.size(), atom_offsets.data(),
                                             atomic_numbers.data(), evaluation.basis,
                                             error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn1::make_integral_plan(evaluation.basis, evaluation.integrals, error) !=
          XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  evaluation.overlap.resize(static_cast<std::size_t>(evaluation.integrals.total_matrix_elements));
  evaluation.workspace.resize((evaluation.integrals.workspace_size_bytes + sizeof(double) - 1u) /
                              sizeof(double));
  return xtbloom::detail::gfn1::evaluate_overlap_cpu(
             evaluation.basis, evaluation.integrals, positions.data(), evaluation.overlap.data(),
             evaluation.workspace.data(), evaluation.workspace.size() * sizeof(double),
             error) == XTBLOOM_STATUS_SUCCESS;
}

int test_hydrogen_oracle_and_same_center_orthogonality() {
  const std::vector<std::int64_t> offsets{0, 2};
  const std::vector<std::int32_t> atomic_numbers{1, 1};
  const std::vector<double> positions{0.0, 0.0, -0.70252931147690, 0.0, 0.0, 0.70252931147690};
  Evaluation evaluation;
  std::string error;
  CHECK(evaluate(1, offsets, atomic_numbers, positions, evaluation, error));
  CHECK(evaluation.integrals.matrix_offsets == std::vector<std::int64_t>({0, 16}));

  /*
   * Independent float32 overlap fixture from dxtb b529b5d's
   * test/test_overlap/overlap.npz, generated with tblite for GFN1 H2 at this
   * geometry. Tight double checks below additionally fix exact orthogonality.
   */
  constexpr std::array<double, 16> reference{
      1.0,          8.5040130e-10, 6.6998297e-1,  6.5205745e-2, 8.5039964e-10, 1.0,
      6.5205745e-2, 1.0264305e-1,  6.6998297e-1,  6.5205745e-2, 1.0,           8.5040130e-10,
      6.5205745e-2, 1.0264305e-1,  8.5039964e-10, 1.0,
  };
  for (std::size_t element = 0; element < reference.size(); ++element) {
    CHECK(near(evaluation.overlap[element], reference[element], 1.0e-6));
  }
  CHECK(near(evaluation.overlap[0], 1.0, 2.0e-8));
  CHECK(near(evaluation.overlap[5], 1.0, 2.0e-8));
  CHECK(near(evaluation.overlap[1], 0.0, 1.0e-9));
  CHECK(near(evaluation.overlap[4], 0.0, 1.0e-9));
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t column = 0; column < 4; ++column) {
      CHECK(near(evaluation.overlap[row * 4u + column], evaluation.overlap[column * 4u + row],
                 2.0e-16));
    }
  }
  return 0;
}

int test_sto6g_spd_ragged_and_translation() {
  const std::vector<std::int64_t> offsets{0, 2, 2, 3, 5};
  const std::vector<std::int32_t> atomic_numbers{14, 14, 6, 8, 1};
  const std::vector<double> positions{
      0.0, 0.0, 0.0, 1.1, -0.7, 2.3, -0.2, 0.4, 1.5, 2.0, -1.0, 0.5, 3.2, -0.8, 0.1,
  };
  Evaluation batch;
  std::string error;
  CHECK(evaluate(4, offsets, atomic_numbers, positions, batch, error));
  CHECK(batch.integrals.matrix_offsets == std::vector<std::int64_t>({0, 324, 324, 340, 376}));

  for (std::size_t molecule = 0; molecule < 4; ++molecule) {
    const std::int64_t begin = offsets[molecule];
    const std::int64_t end = offsets[molecule + 1u];
    if (begin == end) continue;
    Evaluation sequential;
    const std::vector<std::int64_t> local_offsets{0, end - begin};
    const std::vector<std::int32_t> local_numbers(atomic_numbers.begin() + begin,
                                                  atomic_numbers.begin() + end);
    const std::vector<double> local_positions(positions.begin() + begin * 3,
                                              positions.begin() + end * 3);
    CHECK(evaluate(1, local_offsets, local_numbers, local_positions, sequential, error));
    const auto packed_begin = static_cast<std::size_t>(batch.integrals.matrix_offsets[molecule]);
    for (std::size_t element = 0; element < sequential.overlap.size(); ++element) {
      CHECK(batch.overlap[packed_begin + element] == sequential.overlap[element]);
    }
  }

  std::vector<double> translated = positions;
  for (std::size_t atom = 0; atom < atomic_numbers.size(); ++atom) {
    translated[atom * 3u] += 3.5;
    translated[atom * 3u + 1u] -= 1.25;
    translated[atom * 3u + 2u] += 0.75;
  }
  Evaluation shifted;
  CHECK(evaluate(4, offsets, atomic_numbers, translated, shifted, error));
  for (std::size_t element = 0; element < batch.overlap.size(); ++element) {
    CHECK(near(shifted.overlap[element], batch.overlap[element], 2.0e-15));
  }
  return 0;
}

bool weighted_overlap(const Evaluation& evaluation, const std::vector<double>& positions,
                      const std::vector<double>& weights, double& value, std::string& error) {
  std::vector<double> overlap(weights.size());
  if (xtbloom::detail::gfn1::evaluate_overlap_cpu(
          evaluation.basis, evaluation.integrals, positions.data(), overlap.data(),
          const_cast<double*>(evaluation.workspace.data()),
          evaluation.workspace.size() * sizeof(double), error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  value = 0.0;
  for (std::size_t element = 0; element < weights.size(); ++element) {
    value += weights[element] * overlap[element];
  }
  return true;
}

int test_overlap_gradient_multiple_steps() {
  const std::vector<std::int64_t> offsets{0, 2, 4};
  const std::vector<std::int32_t> atomic_numbers{1, 1, 14, 8};
  std::vector<double> positions{0.0, 0.1, -0.7, 0.2, -0.3, 0.8, -1.4, 0.5, 0.2, 0.7, 1.1, -0.6};
  Evaluation evaluation;
  std::string error;
  CHECK(evaluate(2, offsets, atomic_numbers, positions, evaluation, error));
  std::vector<double> weights(evaluation.overlap.size());
  for (std::size_t element = 0; element < weights.size(); ++element) {
    weights[element] = std::sin(0.31 * static_cast<double>(element + 1u)) +
                       0.17 * std::cos(0.13 * static_cast<double>(element + 2u));
  }
  constexpr double baseline = -0.25;
  std::vector<double> gradients(positions.size(), baseline);
  CHECK(xtbloom::detail::gfn1::add_overlap_gradient_cpu(
            evaluation.basis, evaluation.integrals, positions.data(), weights.data(),
            gradients.data(), evaluation.workspace.data(),
            evaluation.workspace.size() * sizeof(double), error) == XTBLOOM_STATUS_SUCCESS);

  for (double step : {4.0e-5, 2.0e-5}) {
    for (std::size_t coordinate = 0; coordinate < positions.size(); ++coordinate) {
      positions[coordinate] += step;
      double right = 0.0;
      CHECK(weighted_overlap(evaluation, positions, weights, right, error));
      positions[coordinate] -= 2.0 * step;
      double left = 0.0;
      CHECK(weighted_overlap(evaluation, positions, weights, left, error));
      positions[coordinate] += step;
      const double numerical = (right - left) / (2.0 * step);
      CHECK(near(gradients[coordinate] - baseline, numerical, 4.0e-8));
    }
  }
  for (std::size_t molecule = 0; molecule < 2; ++molecule) {
    for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
      double sum = 0.0;
      for (std::int64_t atom = offsets[molecule]; atom < offsets[molecule + 1u]; ++atom) {
        sum += gradients[static_cast<std::size_t>(atom) * 3u + coordinate] - baseline;
      }
      CHECK(near(sum, 0.0, 5.0e-15));
    }
  }
  return 0;
}

int test_transactional_validation() {
  const std::vector<std::int64_t> offsets{0, 1};
  const std::vector<std::int32_t> atomic_numbers{1};
  const std::vector<double> positions(3, 0.0);
  Evaluation evaluation;
  std::string error;
  CHECK(evaluate(1, offsets, atomic_numbers, positions, evaluation, error));
  std::vector<double> overlap(evaluation.overlap.size(), 3.0);
  CHECK(xtbloom::detail::gfn1::evaluate_overlap_cpu(
            evaluation.basis, evaluation.integrals, positions.data(), overlap.data(),
            evaluation.workspace.data(), evaluation.integrals.workspace_size_bytes - 1u,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(overlap.begin(), overlap.end(), [](double value) { return value == 3.0; }));

  std::vector<double> bad_positions = positions;
  bad_positions[1] = std::numeric_limits<double>::quiet_NaN();
  CHECK(xtbloom::detail::gfn1::evaluate_overlap_cpu(
            evaluation.basis, evaluation.integrals, bad_positions.data(), overlap.data(),
            evaluation.workspace.data(), evaluation.workspace.size() * sizeof(double),
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_hydrogen_oracle_and_same_center_orthogonality(); status != 0)
    return status;
  if (const int status = test_sto6g_spd_ragged_and_translation(); status != 0) return status;
  if (const int status = test_overlap_gradient_multiple_steps(); status != 0) return status;
  return test_transactional_validation();
}
