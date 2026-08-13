#include "model/gfn1/es2.hpp"
#include "model/gfn1/es3.hpp"
#include "model/gfn1/spin.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "data/parameters/gfn1.hpp"
#include "data/parameters/tblite_spin.hpp"
#include "model/gfn1/basis.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

using xtbloom::detail::gfn1::BasisPlan;

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

bool make_basis(const std::vector<std::int64_t>& atom_offsets,
                const std::vector<std::int32_t>& atomic_numbers, BasisPlan& basis,
                std::string& error) {
  return xtbloom::detail::gfn1::make_basis_plan(
             static_cast<std::int64_t>(atom_offsets.size() - 1u),
             static_cast<std::int64_t>(atomic_numbers.size()), atom_offsets.data(),
             atomic_numbers.data(), basis, error) == XTBLOOM_STATUS_SUCCESS;
}

struct ES2Evaluation {
  BasisPlan basis;
  xtbloom::detail::gfn1::ES2Plan plan;
  std::vector<double> matrix;
  std::vector<double> matrix_scratch;
  std::vector<double> shell_scratch;
  std::vector<double> batch_scratch;
  std::vector<double> gradient_scratch;
  xtbloom::detail::gfn1::ES2Workspace workspace;
  xtbloom::detail::gfn1::ES2GeometryCache cache;
};

bool make_es2(const std::vector<std::int64_t>& offsets,
              const std::vector<std::int32_t>& atomic_numbers,
              const std::vector<double>& positions, std::uint64_t generation,
              ES2Evaluation& evaluation, std::string& error) {
  if (!make_basis(offsets, atomic_numbers, evaluation.basis, error) ||
      xtbloom::detail::gfn1::make_es2_plan(evaluation.basis, atomic_numbers.data(),
                                           evaluation.plan, error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  evaluation.matrix.resize(static_cast<std::size_t>(evaluation.plan.total_matrix_elements()));
  evaluation.matrix_scratch.resize(evaluation.matrix.size());
  evaluation.shell_scratch.resize(static_cast<std::size_t>(evaluation.plan.total_shells()));
  evaluation.batch_scratch.resize(static_cast<std::size_t>(evaluation.plan.batch_size()));
  evaluation.gradient_scratch.resize(static_cast<std::size_t>(evaluation.plan.total_atoms()) * 3u);
  evaluation.workspace = {
      evaluation.matrix_scratch.data(),   evaluation.plan.total_matrix_elements(),
      evaluation.shell_scratch.data(),    evaluation.plan.total_shells(),
      evaluation.batch_scratch.data(),    evaluation.plan.batch_size(),
      evaluation.gradient_scratch.data(), evaluation.plan.total_atoms() * 3,
  };
  return xtbloom::detail::gfn1::update_es2_geometry_cache_cpu(
             evaluation.plan, positions.data(), generation, evaluation.matrix.data(),
             evaluation.matrix.size(), evaluation.workspace, evaluation.cache,
             error) == XTBLOOM_STATUS_SUCCESS;
}

double es2_energy(const xtbloom::detail::gfn1::ES2Plan& plan,
                  const xtbloom::detail::gfn1::ES2GeometryCache& cache,
                  const std::vector<double>& charges,
                  const xtbloom::detail::gfn1::ES2Workspace& workspace,
                  std::string& error) {
  std::vector<double> energies(static_cast<std::size_t>(plan.batch_size()), 0.0);
  if (xtbloom::detail::gfn1::add_es2_energy_cpu(plan, cache, charges.data(), energies.data(),
                                                workspace, error) != XTBLOOM_STATUS_SUCCESS) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double total = 0.0;
  for (double energy : energies) {
    total += energy;
  }
  return total;
}

int test_es2_harmonic_oracle_and_potential_derivative() {
  const std::vector<std::int64_t> offsets{0, 3};
  const std::vector<std::int32_t> numbers{8, 1, 6};
  const std::vector<double> positions{0.0, 0.0, 0.0, 1.7, -0.2, 0.4, -2.1, 0.3, 1.2};
  ES2Evaluation evaluation;
  std::string error;
  CHECK(make_es2(offsets, numbers, positions, 7u, evaluation, error));
  CHECK(evaluation.plan.hardness_average() ==
        xtbloom::detail::gfn2::ES2HardnessAverage::kHarmonic);

  /*
   * Independent literal expansion of tblite effective.f90 at pinned revision
   * 133f91e: g_s=gam_Z*shell_hubbard_scale and
   * g_st=2/(1/g_s+1/g_t), including onsite off-diagonal O(s,p).
   */
  std::vector<double> expected_hardness;
  for (std::int32_t number : numbers) {
    const auto& element = xtbloom::parameters::gfn1::kElements[number - 1];
    for (std::size_t local = 0; local < element.shell_count; ++local) {
      expected_hardness.push_back(
          element.gam * xtbloom::parameters::gfn1::kShells[element.shell_offset + local]
                            .shell_hubbard_scale);
    }
  }
  CHECK(evaluation.plan.shell_hardness() == expected_hardness);
  const std::int64_t shells = evaluation.plan.total_shells();
  CHECK(shells == 6);
  for (std::int64_t row = 0; row < shells; ++row) {
    for (std::int64_t column = 0; column < shells; ++column) {
      const double first = expected_hardness[static_cast<std::size_t>(row)];
      const double second = expected_hardness[static_cast<std::size_t>(column)];
      const double harmonic = 2.0 / (1.0 / first + 1.0 / second);
      const std::int64_t first_atom = evaluation.basis.shell_to_atom[static_cast<std::size_t>(row)];
      const std::int64_t second_atom =
          evaluation.basis.shell_to_atom[static_cast<std::size_t>(column)];
      double expected = harmonic;
      if (first_atom != second_atom) {
        const double dx = positions[static_cast<std::size_t>(first_atom) * 3u] -
                          positions[static_cast<std::size_t>(second_atom) * 3u];
        const double dy = positions[static_cast<std::size_t>(first_atom) * 3u + 1u] -
                          positions[static_cast<std::size_t>(second_atom) * 3u + 1u];
        const double dz = positions[static_cast<std::size_t>(first_atom) * 3u + 2u] -
                          positions[static_cast<std::size_t>(second_atom) * 3u + 2u];
        expected = 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + 1.0 / (harmonic * harmonic));
      }
      CHECK(near(evaluation.matrix[static_cast<std::size_t>(row * shells + column)], expected,
                 2.0e-15));
    }
  }
  CHECK(evaluation.matrix[1] !=
        0.5 * (expected_hardness[0] + expected_hardness[1]));

  std::vector<double> charges{0.32, -0.51, 0.17, -0.08, 0.24, -0.14};
  std::vector<double> potential(charges.size());
  CHECK(xtbloom::detail::gfn1::evaluate_es2_potential_cpu(
            evaluation.plan, evaluation.cache, charges.data(), potential.data(),
            evaluation.workspace, error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t shell = 0; shell < charges.size(); ++shell) {
    constexpr double step = 1.0e-6;
    charges[shell] += step;
    const double right = es2_energy(evaluation.plan, evaluation.cache, charges,
                                    evaluation.workspace, error);
    charges[shell] -= 2.0 * step;
    const double left = es2_energy(evaluation.plan, evaluation.cache, charges,
                                   evaluation.workspace, error);
    charges[shell] += step;
    CHECK(near((right - left) / (2.0 * step), potential[shell], 3.0e-11));
  }
  return 0;
}

int test_es2_coordinate_gradient_multistep_and_ragged_failure() {
  const std::vector<std::int64_t> offsets{0, 2, 5};
  const std::vector<std::int32_t> numbers{1, 8, 6, 1, 7};
  std::vector<double> positions{0.0, 0.0, 0.0, 1.4, -0.3, 0.7,
                                -1.1, 0.2, 0.4, 2.0, 0.5, -0.8,
                                0.6, -1.5, 1.1};
  ES2Evaluation evaluation;
  std::string error;
  CHECK(make_es2(offsets, numbers, positions, 19u, evaluation, error));
  std::vector<double> charges(static_cast<std::size_t>(evaluation.plan.total_shells()));
  for (std::size_t shell = 0; shell < charges.size(); ++shell) {
    charges[shell] = -0.36 + 0.11 * static_cast<double>(shell % 7u);
  }
  std::vector<double> gradient(positions.size(), 0.0);
  CHECK(xtbloom::detail::gfn1::add_es2_gradient_cpu(
            evaluation.plan, evaluation.cache, positions.data(), 19u, charges.data(),
            gradient.data(), evaluation.workspace, error) == XTBLOOM_STATUS_SUCCESS);

  for (double step : {2.0e-4, 7.0e-5, 2.0e-5}) {
    for (std::size_t coordinate : {std::size_t{0}, std::size_t{4}, std::size_t{8},
                                   std::size_t{13}}) {
      positions[coordinate] += step;
      ES2Evaluation right;
      CHECK(make_es2(offsets, numbers, positions, 31u, right, error));
      const double right_energy =
          es2_energy(right.plan, right.cache, charges, right.workspace, error);
      positions[coordinate] -= 2.0 * step;
      ES2Evaluation left;
      CHECK(make_es2(offsets, numbers, positions, 32u, left, error));
      const double left_energy = es2_energy(left.plan, left.cache, charges, left.workspace, error);
      positions[coordinate] += step;
      CHECK(near((right_energy - left_energy) / (2.0 * step), gradient[coordinate], 3.0e-9));
    }
  }
  for (std::size_t axis = 0; axis < 3u; ++axis) {
    double sum = 0.0;
    for (std::size_t atom = 0; atom < numbers.size(); ++atom) {
      sum += gradient[atom * 3u + axis];
    }
    CHECK(std::abs(sum) < 2.0e-16);
  }

  const double sentinel = 4.25;
  double energy = sentinel;
  charges[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK(xtbloom::detail::gfn1::add_es2_energy_system_cpu(
            evaluation.plan, evaluation.cache, 1, charges.data(), energy,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(energy != sentinel); /* A poisoned peer must not affect system 1. */
  const double saved = energy;
  CHECK(xtbloom::detail::gfn1::add_es2_energy_system_cpu(
            evaluation.plan, evaluation.cache, 0, charges.data(), energy,
            error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(energy == saved);
  return 0;
}

int test_es3_atomwise_oracle_derivative_and_failures() {
  const std::vector<std::int64_t> offsets{0, 2, 2, 5};
  const std::vector<std::int32_t> numbers{1, 8, 6, 24, 7};
  BasisPlan basis;
  std::string error;
  CHECK(make_basis(offsets, numbers, basis, error));
  xtbloom::detail::gfn1::ES3Plan plan;
  CHECK(xtbloom::detail::gfn1::make_es3_plan(basis, numbers.data(), plan,
                                             error) == XTBLOOM_STATUS_SUCCESS);
  const auto view = xtbloom::detail::gfn1::make_es3_view(plan);
  CHECK(plan.atom_offsets == offsets);
  for (std::size_t atom = 0; atom < numbers.size(); ++atom) {
    CHECK(plan.atom_gamma3[atom] ==
          xtbloom::parameters::gfn1::kElements[static_cast<std::size_t>(numbers[atom] - 1)].gam3);
  }
  std::vector<double> charges{0.31, -0.27, 0.42, -0.19, 0.08};
  std::vector<double> potentials(charges.size(), 91.0);
  CHECK(xtbloom::detail::gfn1::evaluate_es3_potential_cpu(
            view, charges.data(), potentials.data(), error) == XTBLOOM_STATUS_SUCCESS);
  std::vector<double> energies{0.1, -0.2, 0.3};
  CHECK(xtbloom::detail::gfn1::add_es3_energy_cpu(view, charges.data(), energies.data(),
                                                  error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(energies[1] == -0.2); /* Empty ragged member. */
  for (std::size_t atom = 0; atom < charges.size(); ++atom) {
    CHECK(potentials[atom] == plan.atom_gamma3[atom] * charges[atom] * charges[atom]);
    constexpr double step = 1.0e-6;
    const std::int64_t system = atom < 2u ? 0 : 2;
    charges[atom] += step;
    double right = 0.0;
    CHECK(xtbloom::detail::gfn1::add_es3_energy_system_cpu(
              view, system, charges.data(), right, error) == XTBLOOM_STATUS_SUCCESS);
    charges[atom] -= 2.0 * step;
    double left = 0.0;
    CHECK(xtbloom::detail::gfn1::add_es3_energy_system_cpu(
              view, system, charges.data(), left, error) == XTBLOOM_STATUS_SUCCESS);
    charges[atom] += step;
    CHECK(near((right - left) / (2.0 * step), potentials[atom], 2.0e-11));
  }

  charges[0] = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> target(charges.size(), -7.0);
  CHECK(xtbloom::detail::gfn1::evaluate_es3_potential_system_cpu(
            view, 2, charges.data(), target.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(target[0] == -7.0 && target[1] == -7.0);
  double accumulated = 4.0;
  CHECK(xtbloom::detail::gfn1::add_es3_energy_system_cpu(
            view, 0, charges.data(), accumulated, error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(accumulated == 4.0);

  auto bad = view;
  bad.atom_gamma3_count -= 1;
  const std::vector<double> sentinel = target;
  CHECK(xtbloom::detail::gfn1::evaluate_es3_potential_cpu(
            bad, charges.data(), target.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(target == sentinel);
  return 0;
}

int test_spin_repeated_hydrogen_and_derivative() {
  const std::vector<std::int64_t> offsets{0, 1, 2, 4};
  const std::vector<std::int32_t> numbers{1, 1, 24, 8};
  BasisPlan basis;
  std::string error;
  CHECK(make_basis(offsets, numbers, basis, error));
  const std::array<std::int32_t, 3> channels{{1, 2, 2}};
  xtbloom::detail::gfn1::SpinPopulationLayout layout;
  CHECK(xtbloom::detail::gfn1::make_spin_population_layout(
            basis, channels.data(), layout, error) == XTBLOOM_STATUS_SUCCESS);
  xtbloom::detail::gfn1::SpinPolarizationPlan plan;
  CHECK(xtbloom::detail::gfn1::make_spin_polarization_plan(
            basis, numbers.data(), layout, plan, error) == XTBLOOM_STATUS_SUCCESS);
  const auto view = xtbloom::detail::gfn1::make_spin_polarization_view(plan);

  CHECK(basis.atom_shell_offsets[1] - basis.atom_shell_offsets[0] == 2);
  CHECK(plan.coupling_offsets[1] - plan.coupling_offsets[0] == 4);
  const double wss = xtbloom::parameters::tblite::kSpinConstants[0][0];
  for (std::int64_t matrix = plan.coupling_offsets[0]; matrix < plan.coupling_offsets[1];
       ++matrix) {
    CHECK(plan.coupling_matrices[static_cast<std::size_t>(matrix)] == wss);
  }

  std::vector<double> populations(static_cast<std::size_t>(layout.element_count), 0.0);
  const std::int64_t restricted_shells = basis.batch_shell_offsets[1];
  CHECK(restricted_shells == 2);
  const std::int64_t second_shells = basis.batch_shell_offsets[2] - basis.batch_shell_offsets[1];
  const std::int64_t second_magnetization = layout.system_offsets[1] + second_shells;
  populations[static_cast<std::size_t>(second_magnetization)] = 0.37;
  populations[static_cast<std::size_t>(second_magnetization + 1)] = -0.21;
  const std::int64_t third_shells = basis.batch_shell_offsets[3] - basis.batch_shell_offsets[2];
  const std::int64_t third_magnetization = layout.system_offsets[2] + third_shells;
  for (std::int64_t shell = 0; shell < third_shells; ++shell) {
    populations[static_cast<std::size_t>(third_magnetization + shell)] =
        -0.3 + 0.13 * static_cast<double>(shell);
  }
  std::vector<double> energies(static_cast<std::size_t>(view.batch_size), 91.0);
  std::vector<double> potential(populations.size(), 91.0);
  CHECK(xtbloom::detail::gfn1::evaluate_spin_polarization_cpu(
            view, populations.data(), energies.data(), potential.data(),
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(energies[0] == 0.0);
  for (std::int64_t index = layout.system_offsets[0]; index < layout.system_offsets[1]; ++index) {
    CHECK(potential[static_cast<std::size_t>(index)] == 0.0);
  }
  CHECK(near(potential[static_cast<std::size_t>(second_magnetization)],
             wss * (0.37 - 0.21), 2.0e-17));
  CHECK(near(potential[static_cast<std::size_t>(second_magnetization + 1)],
             wss * (0.37 - 0.21), 2.0e-17));

  for (std::int64_t system = 1; system < view.batch_size; ++system) {
    const std::int64_t shell_begin = view.batch_shell_offsets[system];
    const std::int64_t shell_end = view.batch_shell_offsets[system + 1];
    const std::int64_t magnetization =
        view.shell_population_offsets[system] + shell_end - shell_begin;
    for (std::int64_t shell = 0; shell < shell_end - shell_begin; ++shell) {
      const std::size_t index = static_cast<std::size_t>(magnetization + shell);
      constexpr double step = 1.0e-6;
      populations[index] += step;
      double right = 0.0;
      CHECK(xtbloom::detail::gfn1::add_spin_polarization_energy_system_cpu(
                view, system, populations.data(), right, error) == XTBLOOM_STATUS_SUCCESS);
      populations[index] -= 2.0 * step;
      double left = 0.0;
      CHECK(xtbloom::detail::gfn1::add_spin_polarization_energy_system_cpu(
                view, system, populations.data(), left, error) == XTBLOOM_STATUS_SUCCESS);
      populations[index] += step;
      CHECK(near((right - left) / (2.0 * step), potential[index], 2.0e-11));
    }
  }
  return 0;
}

int test_spin_peer_local_failure_and_descriptor_validation() {
  const std::vector<std::int64_t> offsets{0, 1, 2};
  const std::vector<std::int32_t> numbers{1, 8};
  BasisPlan basis;
  std::string error;
  CHECK(make_basis(offsets, numbers, basis, error));
  const std::array<std::int32_t, 2> channels{{2, 2}};
  xtbloom::detail::gfn1::SpinPopulationLayout layout;
  CHECK(xtbloom::detail::gfn1::make_spin_population_layout(
            basis, channels.data(), layout, error) == XTBLOOM_STATUS_SUCCESS);
  xtbloom::detail::gfn1::SpinPolarizationPlan plan;
  CHECK(xtbloom::detail::gfn1::make_spin_polarization_plan(
            basis, numbers.data(), layout, plan, error) == XTBLOOM_STATUS_SUCCESS);
  auto view = xtbloom::detail::gfn1::make_spin_polarization_view(plan);
  std::vector<double> populations(static_cast<std::size_t>(layout.element_count), 0.0);
  populations[static_cast<std::size_t>(layout.system_offsets[0] + basis.batch_shell_offsets[1])] =
      std::numeric_limits<double>::quiet_NaN();
  const std::int64_t system_one_shells = basis.batch_shell_offsets[2] - basis.batch_shell_offsets[1];
  const std::int64_t system_one_magnetization = layout.system_offsets[1] + system_one_shells;
  for (std::int64_t shell = 0; shell < system_one_shells; ++shell) {
    populations[static_cast<std::size_t>(system_one_magnetization + shell)] = 0.2 - 0.1 * shell;
  }
  std::vector<double> potentials(populations.size(), -3.0);
  double energy = 0.0;
  CHECK(xtbloom::detail::gfn1::evaluate_spin_polarization_system_cpu(
            view, 1, populations.data(), energy, potentials.data(),
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::isfinite(energy));
  const double saved = 7.0;
  energy = saved;
  CHECK(xtbloom::detail::gfn1::add_spin_polarization_energy_system_cpu(
            view, 0, populations.data(), energy, error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(energy == saved);

  view.coupling_matrix_count -= 1;
  const std::vector<double> sentinel = potentials;
  std::vector<double> energies(2, 4.0);
  CHECK(xtbloom::detail::gfn1::evaluate_spin_polarization_cpu(
            view, populations.data(), energies.data(), potentials.data(),
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(potentials == sentinel && energies == std::vector<double>({4.0, 4.0}));
  return 0;
}

}  // namespace

int main() {
  if (const int line = test_es2_harmonic_oracle_and_potential_derivative(); line != 0) {
    return line;
  }
  if (const int line = test_es2_coordinate_gradient_multistep_and_ragged_failure(); line != 0) {
    return line;
  }
  if (const int line = test_es3_atomwise_oracle_derivative_and_failures(); line != 0) {
    return line;
  }
  if (const int line = test_spin_repeated_hydrogen_and_derivative(); line != 0) {
    return line;
  }
  if (const int line = test_spin_peer_local_failure_and_descriptor_validation(); line != 0) {
    return line;
  }
  return 0;
}
