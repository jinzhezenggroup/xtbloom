#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "data/parameters/gfn1.hpp"
#include "model/gfn1/coordination.hpp"
#include "model/gfn1/h0.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

bool near(double actual, double expected, double tolerance = 2.0e-13) {
  return std::abs(actual - expected) <= tolerance;
}

struct Evaluation {
  xtbloom::detail::gfn1::BasisPlan basis;
  xtbloom::detail::gfn1::IntegralPlan integrals;
  xtbloom::detail::gfn1::H0Plan h0;
  xtbloom::detail::gfn1::CoordinationPlan coordination;
  std::vector<double> integral_workspace;
};

bool make_evaluation(const std::vector<std::int64_t>& atom_offsets,
                     const std::vector<std::int32_t>& atomic_numbers, Evaluation& evaluation,
                     std::string& error) {
  const std::int64_t batch_size = static_cast<std::int64_t>(atom_offsets.size()) - 1;
  return xtbloom::detail::gfn1::make_basis_plan(
             batch_size, static_cast<std::int64_t>(atomic_numbers.size()), atom_offsets.data(),
             atomic_numbers.data(), evaluation.basis, error) == XTBLOOM_STATUS_SUCCESS &&
         xtbloom::detail::gfn1::make_integral_plan(evaluation.basis, evaluation.integrals, error) ==
             XTBLOOM_STATUS_SUCCESS &&
         xtbloom::detail::gfn1::make_h0_plan(evaluation.basis, evaluation.integrals,
                                             atomic_numbers.data(), evaluation.h0,
                                             error) == XTBLOOM_STATUS_SUCCESS &&
         xtbloom::detail::gfn1::make_coordination_plan(
             batch_size, static_cast<std::int64_t>(atomic_numbers.size()), atom_offsets.data(),
             atomic_numbers.data(), evaluation.coordination, error) == XTBLOOM_STATUS_SUCCESS &&
         (evaluation.integral_workspace.resize(
              (evaluation.integrals.workspace_size_bytes + sizeof(double) - 1u) / sizeof(double)),
          true);
}

bool evaluate(const Evaluation& evaluation, const std::vector<double>& positions,
              const std::vector<double>& coordination, const std::vector<double>& overlap,
              std::vector<double>& hamiltonian, std::string& error) {
  hamiltonian.resize(static_cast<std::size_t>(evaluation.integrals.total_matrix_elements));
  return xtbloom::detail::gfn1::evaluate_h0_cpu(
             evaluation.basis, evaluation.integrals, evaluation.h0, positions.data(),
             coordination.data(), overlap.data(), hamiltonian.data(),
             error) == XTBLOOM_STATUS_SUCCESS;
}

double contraction(const std::vector<double>& first, const std::vector<double>& second) {
  return std::inner_product(first.begin(), first.end(), second.begin(), 0.0);
}

bool evaluate_composed(Evaluation& evaluation, const std::vector<double>& positions,
                       std::vector<double>& coordination, std::vector<double>& overlap,
                       std::vector<double>& hamiltonian, std::string& error) {
  coordination.resize(static_cast<std::size_t>(evaluation.basis.total_atoms));
  overlap.resize(static_cast<std::size_t>(evaluation.integrals.total_matrix_elements));
  if (xtbloom::detail::gfn1::evaluate_coordination_cpu(evaluation.coordination, positions.data(),
                                                       coordination.data(),
                                                       error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn1::evaluate_overlap_cpu(
          evaluation.basis, evaluation.integrals, positions.data(), overlap.data(),
          evaluation.integral_workspace.data(),
          evaluation.integral_workspace.size() * sizeof(double), error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  return evaluate(evaluation, positions, coordination, overlap, hamiltonian, error);
}

int test_hydrogen_dxtb_tblite_matrix_fixture() {
  const std::vector<std::int64_t> atom_offsets{0, 2};
  const std::vector<std::int32_t> atomic_numbers{1, 1};
  const std::vector<double> positions{0.0, 0.0, -0.70252931147690, 0.0, 0.0, 0.70252931147690};
  Evaluation evaluation;
  std::string error;
  CHECK(make_evaluation(atom_offsets, atomic_numbers, evaluation, error));

  std::vector<double> overlap(16);
  CHECK(xtbloom::detail::gfn1::evaluate_overlap_cpu(
            evaluation.basis, evaluation.integrals, positions.data(), overlap.data(),
            evaluation.integral_workspace.data(),
            evaluation.integral_workspace.size() * sizeof(double),
            error) == XTBLOOM_STATUS_SUCCESS);
  const std::vector<double> coordination(2, 0.0);
  std::vector<double> hamiltonian;
  CHECK(evaluate(evaluation, positions, coordination, overlap, hamiltonian, error));

  /*
   * dxtb b529b5d retains this Apache-2.0 H2 fixture in
   * test/test_hamiltonian/test_gfn1.py and identifies tblite 0.3.0 as its
   * generator. CN is deliberately disabled so it independently cross-checks
   * basis, overlap, four H-shell branches, and the same-center rule. Its older
   * conversion constant makes this a diagnostic fixture; exact conversion is
   * pinned separately by the parameter-level assertions below.
   */
  constexpr std::array<double, 16> reference{{
      -0.40142945681830,
      0.0,
      -0.47765679842079,
      -0.03687145777483,
      0.0,
      -0.07981592633195,
      -0.03687145777483,
      -0.02334876845340,
      -0.47765679842079,
      -0.03687145777483,
      -0.40142945681830,
      0.0,
      -0.03687145777483,
      -0.02334876845340,
      0.0,
      -0.07981592633195,
  }};
  for (std::size_t index = 0; index < reference.size(); ++index) {
    CHECK(near(hamiltonian[index], reference[index], 3.0e-8));
  }
  return 0;
}

int test_hydrogen_four_branch_scales_and_same_atom_rule() {
  const std::vector<std::int64_t> atom_offsets{0, 2};
  const std::vector<std::int32_t> atomic_numbers{1, 1};
  Evaluation evaluation;
  std::string error;
  CHECK(make_evaluation(atom_offsets, atomic_numbers, evaluation, error));
  CHECK(evaluation.basis.total_shells == 4);
  CHECK(evaluation.basis.total_orbitals == 4);
  CHECK(evaluation.basis.shell_is_valence == std::vector<std::uint8_t>({1, 0, 1, 0}));
  CHECK(evaluation.h0.shell_pair_offsets == std::vector<std::int64_t>({0, 16}));

  const auto& scales = evaluation.h0.shell_pair_scale;
  CHECK(near(scales[0 * 4 + 2], 1.85 * 0.96));
  CHECK(near(scales[0 * 4 + 3], 0.5 * (1.85 + 2.85)));
  CHECK(near(scales[1 * 4 + 2], 0.5 * (1.85 + 2.85)));
  CHECK(near(scales[1 * 4 + 3], 2.85));

  const std::vector<double> positions{0.0, 0.0, -0.70252931147690, 0.0, 0.0, 0.70252931147690};
  const std::vector<double> coordination{0.0, 0.0};
  const std::vector<double> overlap(16, 1.0);
  std::vector<double> hamiltonian;
  CHECK(evaluate(evaluation, positions, coordination, overlap, hamiltonian, error));
  const double ev_to_hartree = 1.0 / 27.21138505;
  const double level_1s = -10.923452 * ev_to_hartree;
  const double level_2s = -2.171902 * ev_to_hartree;
  /* Same-atom blocks ignore hscale and the shell polynomial. */
  CHECK(near(hamiltonian[0 * 4 + 0], level_1s));
  CHECK(near(hamiltonian[0 * 4 + 1], 0.5 * (level_1s + level_2s)));
  CHECK(near(hamiltonian[1 * 4 + 0], 0.5 * (level_1s + level_2s)));
  CHECK(near(hamiltonian[1 * 4 + 1], level_2s));
  return 0;
}

int test_pair_override_default_and_dblock_boundaries() {
  struct PairCase {
    std::int32_t first;
    std::int32_t second;
    double pair_scale;
  };
  constexpr std::array<PairCase, 11> cases{{
      {1, 1, 0.96},
      {1, 5, 0.95},
      {1, 7, 1.04},
      {1, 28, 0.90},
      {1, 75, 0.80},
      {1, 78, 0.80},
      {5, 15, 0.97},
      {7, 14, 1.01},
      {6, 8, 1.0},
      {21, 21, 1.1},
      {21, 39, 1.15},
  }};
  for (const auto& item : cases) {
    const std::vector<std::int64_t> atom_offsets{0, 2};
    const std::vector<std::int32_t> atomic_numbers{item.first, item.second};
    Evaluation evaluation;
    std::string error;
    CHECK(make_evaluation(atom_offsets, atomic_numbers, evaluation, error));

    const std::size_t second_shell =
        static_cast<std::size_t>(evaluation.basis.atom_shell_offsets[1]);
    CHECK(evaluation.basis.shell_is_valence[0] == 1u);
    CHECK(evaluation.basis.shell_is_valence[second_shell] == 1u);
    const auto* first = xtbloom::parameters::gfn1::find_element(item.first);
    const auto* second = xtbloom::parameters::gfn1::find_element(item.second);
    const double difference = first->electronegativity - second->electronegativity;
    const std::size_t angular =
        static_cast<std::size_t>(evaluation.basis.angular_momenta[second_shell]);
    const double expected =
        xtbloom::parameters::gfn1::kGlobal
            .shell_pair_scale[static_cast<std::size_t>(evaluation.basis.angular_momenta[0]) * 3u +
                              angular] *
        item.pair_scale *
        (1.0 + xtbloom::parameters::gfn1::kGlobal.hamiltonian_enscale * difference * difference);
    CHECK(near(evaluation.h0.shell_pair_scale[second_shell], expected));
  }
  return 0;
}

int test_h0_vjp_at_multiple_steps() {
  const std::vector<std::int64_t> atom_offsets{0, 3};
  const std::vector<std::int32_t> atomic_numbers{8, 1, 1};
  std::vector<double> positions{0.13, -0.27, 0.08, 1.51, 0.31, -0.22, -0.44, 1.36, 0.57};
  std::vector<double> coordination{1.73, 0.91, 1.04};
  Evaluation evaluation;
  std::string error;
  CHECK(make_evaluation(atom_offsets, atomic_numbers, evaluation, error));
  const std::size_t matrix_size =
      static_cast<std::size_t>(evaluation.integrals.total_matrix_elements);
  std::vector<double> overlap(matrix_size);
  std::vector<double> weights(matrix_size);
  for (std::size_t index = 0; index < matrix_size; ++index) {
    overlap[index] = 0.07 + 0.013 * static_cast<double>((index * 7u) % 19u);
    weights[index] = -0.4 + 0.031 * static_cast<double>((index * 11u) % 23u);
  }

  std::vector<double> hamiltonian;
  CHECK(evaluate(evaluation, positions, coordination, overlap, hamiltonian, error));
  std::vector<double> dE_doverlap(matrix_size, 0.0);
  std::vector<double> dE_dcn(3, 0.0);
  std::vector<double> gradients(9, 0.0);
  CHECK(xtbloom::detail::gfn1::add_h0_vjp_cpu(evaluation.basis, evaluation.integrals, evaluation.h0,
                                              positions.data(), coordination.data(), overlap.data(),
                                              weights.data(), dE_doverlap.data(), dE_dcn.data(),
                                              gradients.data(), error) == XTBLOOM_STATUS_SUCCESS);

  for (double step : {2.0e-6, 8.0e-7}) {
    for (std::size_t atom = 0; atom < coordination.size(); ++atom) {
      coordination[atom] += step;
      std::vector<double> right;
      CHECK(evaluate(evaluation, positions, coordination, overlap, right, error));
      coordination[atom] -= 2.0 * step;
      std::vector<double> left;
      CHECK(evaluate(evaluation, positions, coordination, overlap, left, error));
      coordination[atom] += step;
      const double numerical =
          (contraction(right, weights) - contraction(left, weights)) / (2.0 * step);
      CHECK(near(dE_dcn[atom], numerical, 2.0e-8));
    }
    for (std::size_t coordinate = 0; coordinate < positions.size(); ++coordinate) {
      positions[coordinate] += step;
      std::vector<double> right;
      CHECK(evaluate(evaluation, positions, coordination, overlap, right, error));
      positions[coordinate] -= 2.0 * step;
      std::vector<double> left;
      CHECK(evaluate(evaluation, positions, coordination, overlap, left, error));
      positions[coordinate] += step;
      const double numerical =
          (contraction(right, weights) - contraction(left, weights)) / (2.0 * step);
      CHECK(near(gradients[coordinate], numerical, 3.0e-8));
    }
  }
  for (std::size_t axis = 0; axis < 3u; ++axis) {
    CHECK(near(gradients[axis] + gradients[3u + axis] + gradients[6u + axis], 0.0, 3.0e-13));
  }
  return 0;
}

int test_composed_overlap_cn_h0_gradient() {
  const std::vector<std::int64_t> atom_offsets{0, 3};
  const std::vector<std::int32_t> atomic_numbers{8, 1, 1};
  std::vector<double> positions{0.13, -0.27, 0.08, 1.51, 0.31, -0.22, -0.44, 1.36, 0.57};
  Evaluation evaluation;
  std::string error;
  CHECK(make_evaluation(atom_offsets, atomic_numbers, evaluation, error));

  std::vector<double> coordination;
  std::vector<double> overlap;
  std::vector<double> hamiltonian;
  CHECK(evaluate_composed(evaluation, positions, coordination, overlap, hamiltonian, error));
  std::vector<double> weights(hamiltonian.size());
  for (std::size_t index = 0; index < weights.size(); ++index) {
    weights[index] = std::sin(0.19 * static_cast<double>(index + 1u)) -
                     0.23 * std::cos(0.07 * static_cast<double>(index + 3u));
  }

  std::vector<double> overlap_adjoint(overlap.size(), 0.0);
  std::vector<double> cn_adjoint(coordination.size(), 0.0);
  std::vector<double> gradients(positions.size(), 0.0);
  CHECK(xtbloom::detail::gfn1::add_h0_vjp_cpu(
            evaluation.basis, evaluation.integrals, evaluation.h0, positions.data(),
            coordination.data(), overlap.data(), weights.data(), overlap_adjoint.data(),
            cn_adjoint.data(), gradients.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn1::add_overlap_gradient_cpu(
            evaluation.basis, evaluation.integrals, positions.data(), overlap_adjoint.data(),
            gradients.data(), evaluation.integral_workspace.data(),
            evaluation.integral_workspace.size() * sizeof(double),
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn1::add_coordination_gradient_cpu(
            evaluation.coordination, positions.data(), cn_adjoint.data(), gradients.data(),
            error) == XTBLOOM_STATUS_SUCCESS);

  for (double step : {2.0e-5, 5.0e-6}) {
    for (std::size_t coordinate = 0; coordinate < positions.size(); ++coordinate) {
      positions[coordinate] += step;
      std::vector<double> right_cn;
      std::vector<double> right_overlap;
      std::vector<double> right_hamiltonian;
      CHECK(evaluate_composed(evaluation, positions, right_cn, right_overlap, right_hamiltonian,
                              error));
      positions[coordinate] -= 2.0 * step;
      std::vector<double> left_cn;
      std::vector<double> left_overlap;
      std::vector<double> left_hamiltonian;
      CHECK(
          evaluate_composed(evaluation, positions, left_cn, left_overlap, left_hamiltonian, error));
      positions[coordinate] += step;
      const double numerical =
          (contraction(right_hamiltonian, weights) - contraction(left_hamiltonian, weights)) /
          (2.0 * step);
      CHECK(near(gradients[coordinate], numerical, 8.0e-8));
    }
  }
  for (std::size_t axis = 0; axis < 3u; ++axis) {
    CHECK(near(gradients[axis] + gradients[3u + axis] + gradients[6u + axis], 0.0, 5.0e-13));
  }
  return 0;
}

int test_validation_is_strong() {
  const std::vector<std::int64_t> atom_offsets{0, 2};
  const std::vector<std::int32_t> atomic_numbers{1, 1};
  Evaluation evaluation;
  std::string error;
  CHECK(make_evaluation(atom_offsets, atomic_numbers, evaluation, error));
  const std::vector<double> positions{0.0, 0.0, -0.7, 0.0, 0.0, 0.7};
  const std::vector<double> coordination(2, 0.8);
  std::vector<double> overlap(16, 1.0);
  std::vector<double> output(16, 77.0);

  auto corrupt_basis = evaluation.basis;
  corrupt_basis.shell_is_valence.pop_back();
  CHECK(xtbloom::detail::gfn1::evaluate_h0_cpu(corrupt_basis, evaluation.integrals, evaluation.h0,
                                               positions.data(), coordination.data(),
                                               overlap.data(), output.data(),
                                               error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(output.begin(), output.end(), [](double value) { return value == 77.0; }));

  corrupt_basis = evaluation.basis;
  corrupt_basis.principal_quantum_numbers.pop_back();
  CHECK(xtbloom::detail::gfn1::evaluate_h0_cpu(corrupt_basis, evaluation.integrals, evaluation.h0,
                                               positions.data(), coordination.data(),
                                               overlap.data(), output.data(),
                                               error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(output.begin(), output.end(), [](double value) { return value == 77.0; }));

  corrupt_basis = evaluation.basis;
  std::swap(corrupt_basis.shell_to_atom[0], corrupt_basis.shell_to_atom[2]);
  CHECK(xtbloom::detail::gfn1::evaluate_h0_cpu(corrupt_basis, evaluation.integrals, evaluation.h0,
                                               positions.data(), coordination.data(),
                                               overlap.data(), output.data(),
                                               error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(output.begin(), output.end(), [](double value) { return value == 77.0; }));

  overlap[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK(xtbloom::detail::gfn1::evaluate_h0_cpu(evaluation.basis, evaluation.integrals,
                                               evaluation.h0, positions.data(), coordination.data(),
                                               overlap.data(), output.data(),
                                               error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(output.begin(), output.end(), [](double value) { return value == 77.0; }));
  return 0;
}

int test_vjp_reference_distance_boundary_and_failure_atomicity() {
  const std::vector<std::int64_t> atom_offsets{0, 3};
  const std::vector<std::int32_t> atomic_numbers{1, 1, 1};
  Evaluation evaluation;
  std::string error;
  CHECK(make_evaluation(atom_offsets, atomic_numbers, evaluation, error));
  const std::size_t matrix_elements =
      static_cast<std::size_t>(evaluation.integrals.total_matrix_elements);
  const std::vector<double> coordination(3, 0.0);
  const std::vector<double> overlap(matrix_elements, 1.0);
  std::vector<double> weights(matrix_elements, 0.0);
  const std::int64_t orbitals = evaluation.basis.total_orbitals;
  for (std::int64_t first = evaluation.basis.atom_orbital_offsets[1];
       first < evaluation.basis.atom_orbital_offsets[2]; ++first) {
    for (std::int64_t second = evaluation.basis.atom_orbital_offsets[2];
         second < evaluation.basis.atom_orbital_offsets[3]; ++second) {
      weights[static_cast<std::size_t>(first * orbitals + second)] = 0.25;
      weights[static_cast<std::size_t>(second * orbitals + first)] = 0.25;
    }
  }

  const double boundary = std::sqrt(std::numeric_limits<double>::epsilon());
  std::vector<double> positions{-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, boundary, 0.0, 0.0};
  std::vector<double> overlap_adjoint(matrix_elements, 3.0);
  std::vector<double> cn_adjoint(3, 5.0);
  std::vector<double> gradients(9, 7.0);
  CHECK(xtbloom::detail::gfn1::add_h0_vjp_cpu(
            evaluation.basis, evaluation.integrals, evaluation.h0, positions.data(),
            coordination.data(), overlap.data(), weights.data(), overlap_adjoint.data(),
            cn_adjoint.data(), gradients.data(), error) == XTBLOOM_STATUS_SUCCESS);

  CHECK(std::all_of(overlap_adjoint.begin(), overlap_adjoint.end(),
                    [](double value) { return value == 3.0; }));
  CHECK(
      std::all_of(cn_adjoint.begin(), cn_adjoint.end(), [](double value) { return value == 5.0; }));
  CHECK(std::all_of(gradients.begin(), gradients.end(), [](double value) { return value == 7.0; }));

  /* A separation safely above the reference boundary is active. */
  positions[6] = 2.0 * boundary;
  CHECK(xtbloom::detail::gfn1::add_h0_vjp_cpu(
            evaluation.basis, evaluation.integrals, evaluation.h0, positions.data(),
            coordination.data(), overlap.data(), weights.data(), overlap_adjoint.data(),
            cn_adjoint.data(), gradients.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::any_of(overlap_adjoint.begin(), overlap_adjoint.end(),
                    [](double value) { return value != 3.0; }));

  const std::vector<double> overflowing_positions{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, std::numeric_limits<double>::max(), 0.0, 0.0,
  };
  overlap_adjoint.assign(matrix_elements, 11.0);
  cn_adjoint.assign(3, 13.0);
  gradients.assign(9, 17.0);
  CHECK(xtbloom::detail::gfn1::add_h0_vjp_cpu(
            evaluation.basis, evaluation.integrals, evaluation.h0, overflowing_positions.data(),
            coordination.data(), overlap.data(), weights.data(), overlap_adjoint.data(),
            cn_adjoint.data(), gradients.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(overlap_adjoint.begin(), overlap_adjoint.end(),
                    [](double value) { return value == 11.0; }));
  CHECK(std::all_of(cn_adjoint.begin(), cn_adjoint.end(),
                    [](double value) { return value == 13.0; }));
  CHECK(
      std::all_of(gradients.begin(), gradients.end(), [](double value) { return value == 17.0; }));
  return 0;
}

int test_ragged_batch_boundaries_are_tied_to_atom_topology() {
  const std::vector<std::int64_t> atom_offsets{0, 1, 2};
  const std::vector<std::int32_t> atomic_numbers{1, 1};
  Evaluation evaluation;
  std::string error;
  CHECK(make_evaluation(atom_offsets, atomic_numbers, evaluation, error));

  auto corrupt_basis = evaluation.basis;
  corrupt_basis.batch_shell_offsets[1] -= 1;
  xtbloom::detail::gfn1::H0Plan output = evaluation.h0;
  CHECK(xtbloom::detail::gfn1::make_h0_plan(corrupt_basis, evaluation.integrals,
                                            atomic_numbers.data(), output,
                                            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(output.batch_shell_offsets == evaluation.h0.batch_shell_offsets);
  return 0;
}

}  // namespace

int main() {
  if (const int line = test_hydrogen_dxtb_tblite_matrix_fixture(); line != 0) {
    return line;
  }
  if (const int line = test_hydrogen_four_branch_scales_and_same_atom_rule(); line != 0) {
    return line;
  }
  if (const int line = test_pair_override_default_and_dblock_boundaries(); line != 0) {
    return line;
  }
  if (const int line = test_h0_vjp_at_multiple_steps(); line != 0) {
    return line;
  }
  if (const int line = test_composed_overlap_cn_h0_gradient(); line != 0) {
    return line;
  }
  if (const int line = test_validation_is_strong(); line != 0) {
    return line;
  }
  if (const int line = test_vjp_reference_distance_boundary_and_failure_atomicity(); line != 0) {
    return line;
  }
  return test_ragged_batch_boundaries_are_tied_to_atom_topology();
}
