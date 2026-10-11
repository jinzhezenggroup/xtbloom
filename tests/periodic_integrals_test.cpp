// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn2/periodic_integrals.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "model/gfn2/basis.hpp"
#include "model/gfn2/coordination.hpp"
#include "model/gfn2/h0.hpp"
#include "model/gfn2/integrals.hpp"
#include "model/gfn2/periodic_topology.hpp"

#define CHECK(condition)                                                                          \
  do {                                                                                            \
    if (!(condition)) {                                                                           \
      std::cerr << "periodic integral check failed at line " << __LINE__ << ": " #condition "\n"; \
      return __LINE__;                                                                            \
    }                                                                                             \
  } while (false)

namespace {

using xtbloom::detail::gfn2::BasisPlan;
using xtbloom::detail::gfn2::H0Plan;
using xtbloom::detail::gfn2::IntegralPlan;
using xtbloom::detail::gfn2::PeriodicIntegralPlan;
using xtbloom::detail::gfn2::PeriodicShortRangeGeometry;
using xtbloom::detail::gfn2::PeriodicShortRangePlan;
using xtbloom::detail::gfn2::PeriodicShortRangeWorkspace;

constexpr std::size_t kDipoleComponents = 3u;
constexpr std::size_t kQuadrupoleComponents = 6u;

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

struct RawArray {
  bool integer = false;
  std::vector<std::size_t> shape;
  std::vector<std::int64_t> integers;
  std::vector<double> reals;
};

struct RawFixture {
  std::unordered_map<std::string, RawArray> arrays;

  const RawArray* find(const std::string& name) const {
    const auto found = arrays.find(name);
    return found == arrays.end() ? nullptr : &found->second;
  }
};

bool read_raw_fixture(const std::string& path, RawFixture& fixture, std::string& error) {
  std::ifstream input(path);
  if (!input) {
    error = "failed to open periodic integral fixture: " + path;
    return false;
  }

  RawFixture parsed;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    std::istringstream header(line);
    std::string kind;
    header >> kind;
    if (kind == "SCHEMA" || kind == "MODE") continue;
    if (kind != "INTEGER" && kind != "REAL") {
      error = "unknown periodic integral fixture record: " + line;
      return false;
    }

    std::string name;
    std::size_t rank = 0u;
    if (!(header >> name >> rank)) {
      error = "malformed periodic integral fixture header: " + line;
      return false;
    }
    RawArray array;
    array.integer = kind == "INTEGER";
    array.shape.resize(rank);
    std::size_t count = 1u;
    for (std::size_t axis = 0; axis < rank; ++axis) {
      if (!(header >> array.shape[axis]) ||
          (array.shape[axis] != 0u &&
           count > std::numeric_limits<std::size_t>::max() / array.shape[axis])) {
        error = "invalid periodic integral fixture shape: " + line;
        return false;
      }
      count *= array.shape[axis];
    }
    if (array.integer) {
      array.integers.reserve(count);
    } else {
      array.reals.reserve(count);
    }
    for (std::size_t element = 0; element < count; ++element) {
      if (!std::getline(input, line)) {
        error = "truncated periodic integral fixture array: " + name;
        return false;
      }
      std::istringstream value(line);
      if (array.integer) {
        std::int64_t parsed_value = 0;
        if (!(value >> parsed_value)) {
          error = "invalid integer in periodic integral fixture array: " + name;
          return false;
        }
        array.integers.push_back(parsed_value);
      } else {
        double parsed_value = 0.0;
        if (!(value >> parsed_value) || !std::isfinite(parsed_value)) {
          error = "invalid real in periodic integral fixture array: " + name;
          return false;
        }
        array.reals.push_back(parsed_value);
      }
    }
    if (!parsed.arrays.emplace(name, std::move(array)).second) {
      error = "duplicate periodic integral fixture array: " + name;
      return false;
    }
  }

  fixture = std::move(parsed);
  error.clear();
  return true;
}

struct AlignedWorkspace {
  explicit AlignedWorkspace(std::size_t size = 1u) { reset(size); }

  void reset(std::size_t size) {
    storage.resize(size + 63u);
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(storage.data());
    data = reinterpret_cast<void*>((begin + 63u) & ~std::uintptr_t{63u});
  }

  std::vector<std::byte> storage;
  void* data = nullptr;
};

struct Model {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  BasisPlan basis;
  IntegralPlan integrals;
  xtbloom::detail::gfn2::CoordinationPlan coordination_plan;
  H0Plan h0;

  bool initialize(const RawFixture& raw, std::string& error) {
    const RawArray* numbers = raw.find("atomic_numbers");
    if (numbers == nullptr || !numbers->integer || numbers->integers.empty()) {
      error = "periodic integral fixture has no atomic numbers";
      return false;
    }
    atomic_numbers.resize(numbers->integers.size());
    for (std::size_t atom = 0; atom < numbers->integers.size(); ++atom) {
      atomic_numbers[atom] = static_cast<std::int32_t>(numbers->integers[atom]);
      if (atomic_numbers[atom] != numbers->integers[atom]) {
        error = "periodic integral fixture atomic number exceeds int32";
        return false;
      }
    }
    atom_offsets = {0, static_cast<std::int64_t>(atomic_numbers.size())};
    return xtbloom::detail::gfn2::make_basis_plan(
               1, static_cast<std::int64_t>(atomic_numbers.size()), atom_offsets.data(),
               atomic_numbers.data(), basis, error) == XTBLOOM_STATUS_SUCCESS &&
           xtbloom::detail::gfn2::make_integral_plan(basis, integrals, error) ==
               XTBLOOM_STATUS_SUCCESS &&
           xtbloom::detail::gfn2::make_coordination_plan(
               1, static_cast<std::int64_t>(atomic_numbers.size()), atom_offsets.data(),
               atomic_numbers.data(), coordination_plan, error) == XTBLOOM_STATUS_SUCCESS &&
           xtbloom::detail::gfn2::make_h0_plan(basis, integrals, atomic_numbers.data(), h0,
                                               error) == XTBLOOM_STATUS_SUCCESS;
  }
};

struct MatrixOutputs {
  std::vector<double> overlap;
  std::vector<double> dipole;
  std::vector<double> quadrupole;
  std::vector<double> hamiltonian;
};

struct MatrixAdjoints {
  std::vector<double> overlap;
  std::vector<double> dipole;
  std::vector<double> quadrupole;
  std::vector<double> hamiltonian;
};

struct PeriodicEvaluation {
  const Model* model = nullptr;
  PeriodicShortRangePlan topology;
  AlignedWorkspace geometry_storage;
  PeriodicShortRangeWorkspace geometry_workspace;
  PeriodicShortRangeGeometry geometry;
  PeriodicIntegralPlan periodic;
  AlignedWorkspace integral_workspace;

  bool initialize(const Model& input_model, const std::vector<double>& positions,
                  const std::vector<double>& cell, std::string& error) {
    model = &input_model;
    if (xtbloom::detail::gfn2::make_periodic_short_range_plan(
            input_model.basis.batch_size, input_model.basis.total_atoms,
            input_model.atom_offsets.data(), cell.data(), topology,
            error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    geometry_storage.reset(topology.workspace_size_bytes());
    if (xtbloom::detail::gfn2::bind_periodic_short_range_workspace(
            topology, geometry_storage.data, topology.workspace_size_bytes(), geometry_workspace,
            error) != XTBLOOM_STATUS_SUCCESS ||
        xtbloom::detail::gfn2::update_periodic_short_range_geometry_cpu(
            topology, positions.data(), 1u, geometry_workspace, geometry, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        xtbloom::detail::gfn2::make_periodic_integral_plan(input_model.basis, input_model.integrals,
                                                           topology, periodic,
                                                           error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    integral_workspace.reset(periodic.workspace_size_bytes());
    return true;
  }

  bool update_positions(const std::vector<double>& positions, std::uint64_t generation,
                        std::string& error) {
    return xtbloom::detail::gfn2::update_periodic_short_range_geometry_cpu(
               topology, positions.data(), generation, geometry_workspace, geometry, error) ==
           XTBLOOM_STATUS_SUCCESS;
  }

  bool evaluate(const std::vector<double>& coordination, MatrixOutputs& outputs,
                std::string& error) {
    const std::size_t matrix_elements =
        static_cast<std::size_t>(model->integrals.total_matrix_elements);
    outputs.overlap.resize(matrix_elements);
    outputs.dipole.resize(kDipoleComponents * matrix_elements);
    outputs.quadrupole.resize(kQuadrupoleComponents * matrix_elements);
    outputs.hamiltonian.resize(matrix_elements);
    return xtbloom::detail::gfn2::evaluate_periodic_integrals_h0_cpu(
               model->basis, model->integrals, model->h0, periodic, topology, geometry,
               geometry_workspace, coordination.data(), outputs.overlap.data(),
               outputs.dipole.data(), outputs.quadrupole.data(), outputs.hamiltonian.data(),
               integral_workspace.data, periodic.workspace_size_bytes(),
               error) == XTBLOOM_STATUS_SUCCESS;
  }
};

double contraction(const MatrixOutputs& outputs, const MatrixAdjoints& adjoints) {
  double value = 0.0;
  value = std::inner_product(outputs.overlap.begin(), outputs.overlap.end(),
                             adjoints.overlap.begin(), value);
  value = std::inner_product(outputs.dipole.begin(), outputs.dipole.end(), adjoints.dipole.begin(),
                             value);
  value = std::inner_product(outputs.quadrupole.begin(), outputs.quadrupole.end(),
                             adjoints.quadrupole.begin(), value);
  return std::inner_product(outputs.hamiltonian.begin(), outputs.hamiltonian.end(),
                            adjoints.hamiltonian.begin(), value);
}

MatrixAdjoints make_adjoint(const Model& model, std::size_t selected_term) {
  const std::size_t matrix_elements =
      static_cast<std::size_t>(model.integrals.total_matrix_elements);
  MatrixAdjoints adjoints;
  adjoints.overlap.assign(matrix_elements, 0.0);
  adjoints.dipole.assign(kDipoleComponents * matrix_elements, 0.0);
  adjoints.quadrupole.assign(kQuadrupoleComponents * matrix_elements, 0.0);
  adjoints.hamiltonian.assign(matrix_elements, 0.0);
  std::array<std::vector<double>*, 4> selected{&adjoints.overlap, &adjoints.dipole,
                                               &adjoints.quadrupole, &adjoints.hamiltonian};
  for (std::size_t index = 0; index < selected[selected_term]->size(); ++index) {
    (*selected[selected_term])[index] =
        -0.37 + 0.019 * static_cast<double>((index * 17u + selected_term * 11u) % 43u);
  }
  return adjoints;
}

bool evaluate_contraction(const Model& model, const std::vector<double>& positions,
                          const std::vector<double>& cell, const std::vector<double>& coordination,
                          const MatrixAdjoints& adjoints, double& value, std::string& error) {
  PeriodicEvaluation evaluation;
  MatrixOutputs outputs;
  if (!evaluation.initialize(model, positions, cell, error) ||
      !evaluation.evaluate(coordination, outputs, error)) {
    return false;
  }
  value = contraction(outputs, adjoints);
  return true;
}

void affine_deformation(const std::vector<double>& reference_positions,
                        const std::vector<double>& reference_cell, std::size_t row,
                        std::size_t column, double amount, std::vector<double>& positions,
                        std::vector<double>& cell) {
  positions = reference_positions;
  cell = reference_cell;
  /* Public row-vector convention: r'=r(I+eps)^T and H'=H(I+eps)^T. */
  for (std::size_t atom = 0; atom < reference_positions.size() / 3u; ++atom) {
    positions[atom * 3u + row] += amount * reference_positions[atom * 3u + column];
  }
  for (std::size_t lattice_row = 0; lattice_row < 3u; ++lattice_row) {
    cell[lattice_row * 3u + row] += amount * reference_cell[lattice_row * 3u + column];
  }
}

bool positive_definite(const std::vector<double>& matrix, std::size_t size) {
  std::vector<double> lower(size * size, 0.0);
  for (std::size_t row = 0; row < size; ++row) {
    for (std::size_t column = 0; column <= row; ++column) {
      double value = matrix[row * size + column];
      for (std::size_t inner = 0; inner < column; ++inner) {
        value -= lower[row * size + inner] * lower[column * size + inner];
      }
      if (row == column) {
        if (!(value > 1.0e-12) || !std::isfinite(value)) return false;
        lower[row * size + column] = std::sqrt(value);
      } else {
        lower[row * size + column] = value / lower[column * size + column];
      }
    }
  }
  return true;
}

int test_oracle_matrices_and_translation_order(const RawFixture& raw, const Model& model,
                                               const std::vector<double>& positions,
                                               const std::vector<double>& cell,
                                               const std::vector<double>& coordination) {
  std::string error;
  PeriodicEvaluation evaluation;
  CHECK(evaluation.initialize(model, positions, cell, error));

  const RawArray* cutoff = raw.find("integral_cutoff_bohr");
  const RawArray* translations = raw.find("integral_lattice_translations_bohr");
  const RawArray* shells = raw.find("number_of_shells");
  const RawArray* aos = raw.find("number_of_aos");
  const RawArray* angular = raw.find("shell_angular_momentum");
  const RawArray* levels = raw.find("shell_self_energies_hartree");
  CHECK(cutoff != nullptr && cutoff->reals.size() == 1u);
  CHECK(translations != nullptr && translations->reals.size() == 375u);
  CHECK(shells != nullptr && shells->integers.size() == 1u);
  CHECK(aos != nullptr && aos->integers.size() == 1u);
  CHECK(angular != nullptr && levels != nullptr);
  CHECK(model.basis.total_shells == shells->integers[0]);
  CHECK(model.basis.total_orbitals == aos->integers[0]);
  CHECK(near(evaluation.periodic.realspace_cutoff(0), cutoff->reals[0], 2.0e-15));
  CHECK(model.basis.angular_momenta.size() == angular->integers.size());
  CHECK(model.h0.shell_levels.size() == levels->reals.size());
  for (std::size_t shell = 0; shell < model.basis.angular_momenta.size(); ++shell) {
    CHECK(model.basis.angular_momenta[shell] == angular->integers[shell]);
    const auto atom = static_cast<std::size_t>(model.basis.shell_to_atom[shell]);
    const double self_energy = model.h0.shell_levels[shell] -
                               model.h0.shell_coordination_scale[shell] * coordination[atom];
    CHECK(near(self_energy, levels->reals[shell], 2.0e-15));
  }

  const auto plan_translations = evaluation.periodic.translations(0);
  CHECK(plan_translations.size == 125);
  std::size_t expected_index = 0u;
  for (std::int64_t nx = 0; nx <= 2; ++nx) {
    for (std::int64_t ny = 0; ny <= 2; ++ny) {
      for (std::int64_t nz = 0; nz <= 2; ++nz) {
        const std::size_t count_x = nx == 0 ? 1u : 2u;
        const std::size_t count_y = ny == 0 ? 1u : 2u;
        const std::size_t count_z = nz == 0 ? 1u : 2u;
        for (std::size_t sx = 0; sx < count_x; ++sx) {
          for (std::size_t sy = 0; sy < count_y; ++sy) {
            for (std::size_t sz = 0; sz < count_z; ++sz) {
              const std::array<std::int64_t, 3> index{sx == 0u ? nx : -nx, sy == 0u ? ny : -ny,
                                                      sz == 0u ? nz : -nz};
              CHECK(plan_translations.data[expected_index].index == index);
              for (std::size_t axis = 0; axis < 3u; ++axis) {
                const double actual = plan_translations.data[expected_index].cartesian[axis];
                const double expected = translations->reals[expected_index * 3u + axis];
                if (!near(actual, expected, 8.0e-15)) {
                  std::cerr << "translation mismatch at " << expected_index << "," << axis
                            << ": actual=" << actual << " expected=" << expected
                            << " difference=" << actual - expected << "\n";
                  CHECK(false);
                }
              }
              ++expected_index;
            }
          }
        }
      }
    }
  }
  CHECK(expected_index == 125u);

  MatrixOutputs outputs;
  CHECK(evaluation.evaluate(coordination, outputs, error));
  const RawArray* overlap = raw.find("overlap_matrix");
  const RawArray* dipole = raw.find("dipole_matrices_bohr");
  const RawArray* quadrupole = raw.find("quadrupole_matrices_bohr2");
  const RawArray* hamiltonian = raw.find("h0_matrix_hartree");
  CHECK(overlap != nullptr && dipole != nullptr && quadrupole != nullptr && hamiltonian != nullptr);
  CHECK(outputs.overlap.size() == overlap->reals.size());
  CHECK(outputs.dipole.size() == dipole->reals.size());
  CHECK(outputs.quadrupole.size() == quadrupole->reals.size());
  CHECK(outputs.hamiltonian.size() == hamiltonian->reals.size());
  for (std::size_t index = 0; index < outputs.overlap.size(); ++index) {
    if (!near(outputs.overlap[index], overlap->reals[index], 2.0e-13)) {
      std::cerr << "overlap mismatch at " << index << ": actual=" << outputs.overlap[index]
                << " expected=" << overlap->reals[index]
                << " difference=" << outputs.overlap[index] - overlap->reals[index] << "\n";
      CHECK(false);
    }
    if (!near(outputs.hamiltonian[index], hamiltonian->reals[index], 5.0e-13)) {
      std::cerr << "H0 mismatch at " << index << ": actual=" << outputs.hamiltonian[index]
                << " expected=" << hamiltonian->reals[index]
                << " difference=" << outputs.hamiltonian[index] - hamiltonian->reals[index] << "\n";
      CHECK(false);
    }
  }
  const std::size_t nao = static_cast<std::size_t>(model.basis.total_orbitals);
  const std::size_t matrix_elements = nao * nao;
  for (std::size_t component = 0; component < kDipoleComponents; ++component) {
    for (std::size_t row = 0; row < nao; ++row) {
      for (std::size_t column = 0; column < nao; ++column) {
        const std::size_t output_index = component * matrix_elements + row * nao + column;
        const std::size_t fixture_index =
            component + kDipoleComponents * row + kDipoleComponents * nao * column;
        if (!near(outputs.dipole[output_index], dipole->reals[fixture_index], 2.0e-13)) {
          std::cerr << "dipole mismatch at component,row,column=" << component << "," << row << ","
                    << column << ": actual=" << outputs.dipole[output_index]
                    << " expected=" << dipole->reals[fixture_index]
                    << " difference=" << outputs.dipole[output_index] - dipole->reals[fixture_index]
                    << "\n";
          CHECK(false);
        }
      }
    }
  }
  for (std::size_t component = 0; component < kQuadrupoleComponents; ++component) {
    for (std::size_t row = 0; row < nao; ++row) {
      for (std::size_t column = 0; column < nao; ++column) {
        const std::size_t output_index = component * matrix_elements + row * nao + column;
        const std::size_t fixture_index =
            component + kQuadrupoleComponents * row + kQuadrupoleComponents * nao * column;
        if (!near(outputs.quadrupole[output_index], quadrupole->reals[fixture_index], 5.0e-13)) {
          std::cerr << "quadrupole mismatch at component,row,column=" << component << "," << row
                    << "," << column << ": actual=" << outputs.quadrupole[output_index]
                    << " expected=" << quadrupole->reals[fixture_index] << " difference="
                    << outputs.quadrupole[output_index] - quadrupole->reals[fixture_index] << "\n";
          CHECK(false);
        }
      }
    }
  }

  for (std::size_t row = 0; row < nao; ++row) {
    for (std::size_t column = 0; column < nao; ++column) {
      CHECK(
          near(outputs.overlap[row * nao + column], outputs.overlap[column * nao + row], 2.0e-14));
      CHECK(near(outputs.hamiltonian[row * nao + column], outputs.hamiltonian[column * nao + row],
                 2.0e-14));
      CHECK(near(outputs.quadrupole[row * nao + column] +
                     outputs.quadrupole[2u * matrix_elements + row * nao + column] +
                     outputs.quadrupole[5u * matrix_elements + row * nao + column],
                 0.0, 2.0e-14));
    }
  }
  CHECK(positive_definite(outputs.overlap, nao));

  BasisPlan diffuse = model.basis;
  std::fill(diffuse.primitive_exponents.begin(), diffuse.primitive_exponents.end(), 1.0e-3);
  diffuse.minimum_primitive_exponent = 1.0e-3;
  PeriodicIntegralPlan capped;
  CHECK(xtbloom::detail::gfn2::make_periodic_integral_plan(diffuse, model.integrals,
                                                           evaluation.topology, capped,
                                                           error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(capped.realspace_cutoff(0) == xtbloom::detail::gfn2::kPeriodicIntegralMaximumCutoffBohr);
  return 0;
}

int test_complete_vjp_finite_differences(const Model& model, const std::vector<double>& positions,
                                         const std::vector<double>& cell,
                                         const std::vector<double>& coordination) {
  constexpr std::array<double, 2> steps{1.0e-4, 5.0e-5};
  constexpr std::array<std::array<std::size_t, 2>, 6> strain_modes{{
      {0u, 0u},
      {1u, 1u},
      {2u, 2u},
      {0u, 1u},
      {0u, 2u},
      {1u, 2u},
  }};
  std::string error;
  PeriodicEvaluation reference;
  CHECK(reference.initialize(model, positions, cell, error));

  for (std::size_t term = 0; term < 4u; ++term) {
    const MatrixAdjoints adjoints = make_adjoint(model, term);
    std::vector<double> dE_dcn(static_cast<std::size_t>(model.basis.total_atoms), 0.0);
    std::vector<double> gradients(static_cast<std::size_t>(model.basis.total_atoms) * 3u, 0.0);
    std::array<double, 9> strain{};
    CHECK(xtbloom::detail::gfn2::add_periodic_integrals_h0_vjp_cpu(
              model.basis, model.integrals, model.h0, reference.periodic, reference.topology,
              reference.geometry, reference.geometry_workspace, coordination.data(),
              adjoints.overlap.data(), adjoints.dipole.data(), adjoints.quadrupole.data(),
              adjoints.hamiltonian.data(), dE_dcn.data(), gradients.data(), strain.data(),
              reference.integral_workspace.data, reference.periodic.workspace_size_bytes(),
              error) == XTBLOOM_STATUS_SUCCESS);

    for (std::size_t coordinate = 0; coordinate < positions.size(); ++coordinate) {
      double previous = 0.0;
      for (std::size_t sample = 0; sample < steps.size(); ++sample) {
        auto plus_positions = positions;
        auto minus_positions = positions;
        plus_positions[coordinate] += steps[sample];
        minus_positions[coordinate] -= steps[sample];
        double plus = 0.0;
        double minus = 0.0;
        CHECK(
            evaluate_contraction(model, plus_positions, cell, coordination, adjoints, plus, error));
        CHECK(evaluate_contraction(model, minus_positions, cell, coordination, adjoints, minus,
                                   error));
        const double estimate = (plus - minus) / (2.0 * steps[sample]);
        CHECK(near(estimate, gradients[coordinate], 2.0e-7));
        if (sample != 0u) CHECK(near(estimate, previous, 2.0e-6));
        previous = estimate;
      }
    }
    for (std::size_t axis = 0; axis < 3u; ++axis) {
      double net = 0.0;
      for (std::size_t atom = 0; atom < positions.size() / 3u; ++atom) {
        net += gradients[atom * 3u + axis];
      }
      CHECK(near(net, 0.0, 2.0e-11));
    }

    for (const auto& mode : strain_modes) {
      const std::size_t row = mode[0];
      const std::size_t column = mode[1];
      double previous = 0.0;
      for (std::size_t sample = 0; sample < steps.size(); ++sample) {
        std::vector<double> plus_positions;
        std::vector<double> minus_positions;
        std::vector<double> plus_cell;
        std::vector<double> minus_cell;
        affine_deformation(positions, cell, row, column, steps[sample], plus_positions, plus_cell);
        affine_deformation(positions, cell, row, column, -steps[sample], minus_positions,
                           minus_cell);
        double plus = 0.0;
        double minus = 0.0;
        CHECK(evaluate_contraction(model, plus_positions, plus_cell, coordination, adjoints, plus,
                                   error));
        CHECK(evaluate_contraction(model, minus_positions, minus_cell, coordination, adjoints,
                                   minus, error));
        const double estimate = (plus - minus) / (2.0 * steps[sample]);
        CHECK(near(estimate, strain[row * 3u + column], 2.0e-7));
        if (sample != 0u) CHECK(near(estimate, previous, 2.0e-6));
        previous = estimate;
      }
    }

    if (term == 3u) {
      for (std::size_t atom = 0; atom < coordination.size(); ++atom) {
        double previous = 0.0;
        for (std::size_t sample = 0; sample < steps.size(); ++sample) {
          auto plus_cn = coordination;
          auto minus_cn = coordination;
          plus_cn[atom] += steps[sample];
          minus_cn[atom] -= steps[sample];
          double plus = 0.0;
          double minus = 0.0;
          CHECK(evaluate_contraction(model, positions, cell, plus_cn, adjoints, plus, error));
          CHECK(evaluate_contraction(model, positions, cell, minus_cn, adjoints, minus, error));
          const double estimate = (plus - minus) / (2.0 * steps[sample]);
          CHECK(near(estimate, dE_dcn[atom], 2.0e-7));
          if (sample != 0u) CHECK(near(estimate, previous, 2.0e-6));
          previous = estimate;
        }
      }
    } else {
      CHECK(std::all_of(dE_dcn.begin(), dE_dcn.end(), [](double value) { return value == 0.0; }));
    }
  }
  return 0;
}

int test_independent_displaced_oracle(const Model& model, const std::vector<double>& positions,
                                      const std::vector<double>& cell,
                                      const std::vector<double>& coordination) {
  /*
   * Pinned tblite 133f91ef retained these atom-1/x displacements for AO
   * element (bra,ket)=(0,4). They are independent of xTBloom's VJP and
   * therefore distinguish a shared evaluator/reverse-mode sign error from a
   * genuinely correct derivative. See integrals-h0.json finite_difference.
   */
  constexpr std::array<double, 2> steps{1.0e-4, 5.0e-5};
  constexpr std::array<double, 2> overlap_plus{0.45294706222908, 0.4529608655820777};
  constexpr std::array<double, 2> overlap_minus{0.4530022759638177, 0.4529884724494768};
  constexpr std::array<double, 2> overlap_estimate{-0.2760686736885676, -0.27606867399110335};
  constexpr std::array<double, 2> h0_plus{-0.4020967578524443, -0.40210982848312127};
  constexpr std::array<double, 2> h0_minus{-0.4021490409581941, -0.40213597003602397};
  constexpr std::array<double, 2> h0_estimate{0.2614155287489206, 0.26141552902703147};
  constexpr std::size_t matrix_index = 4u;
  constexpr std::size_t coordinate = 3u;

  std::string error;
  for (std::size_t sample = 0; sample < steps.size(); ++sample) {
    auto plus_positions = positions;
    auto minus_positions = positions;
    plus_positions[coordinate] += steps[sample];
    minus_positions[coordinate] -= steps[sample];
    PeriodicEvaluation plus_evaluation;
    PeriodicEvaluation minus_evaluation;
    MatrixOutputs plus;
    MatrixOutputs minus;
    CHECK(plus_evaluation.initialize(model, plus_positions, cell, error));
    CHECK(minus_evaluation.initialize(model, minus_positions, cell, error));
    std::vector<double> plus_cn(coordination.size());
    std::vector<double> minus_cn(coordination.size());
    CHECK(xtbloom::detail::gfn2::evaluate_periodic_coordination_cpu(
              model.coordination_plan, plus_evaluation.topology, plus_evaluation.geometry,
              plus_cn.data(), plus_evaluation.geometry_workspace, error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom::detail::gfn2::evaluate_periodic_coordination_cpu(
              model.coordination_plan, minus_evaluation.topology, minus_evaluation.geometry,
              minus_cn.data(), minus_evaluation.geometry_workspace,
              error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(plus_evaluation.evaluate(plus_cn, plus, error));
    CHECK(minus_evaluation.evaluate(minus_cn, minus, error));
    CHECK(near(plus.overlap[matrix_index], overlap_plus[sample], 2.0e-13));
    CHECK(near(minus.overlap[matrix_index], overlap_minus[sample], 2.0e-13));
    CHECK(near((plus.overlap[matrix_index] - minus.overlap[matrix_index]) / (2.0 * steps[sample]),
               overlap_estimate[sample], 5.0e-9));
    CHECK(near(plus.hamiltonian[matrix_index], h0_plus[sample], 5.0e-13));
    CHECK(near(minus.hamiltonian[matrix_index], h0_minus[sample], 5.0e-13));
    CHECK(near(
        (plus.hamiltonian[matrix_index] - minus.hamiltonian[matrix_index]) / (2.0 * steps[sample]),
        h0_estimate[sample], 5.0e-9));
  }

  PeriodicEvaluation reference;
  CHECK(reference.initialize(model, positions, cell, error));
  for (std::size_t term : {0u, 3u}) {
    MatrixAdjoints adjoints;
    const std::size_t matrix_elements =
        static_cast<std::size_t>(model.integrals.total_matrix_elements);
    adjoints.overlap.assign(matrix_elements, 0.0);
    adjoints.dipole.assign(kDipoleComponents * matrix_elements, 0.0);
    adjoints.quadrupole.assign(kQuadrupoleComponents * matrix_elements, 0.0);
    adjoints.hamiltonian.assign(matrix_elements, 0.0);
    (term == 0u ? adjoints.overlap : adjoints.hamiltonian)[matrix_index] = 1.0;
    std::vector<double> dE_dcn(coordination.size(), 0.0);
    std::vector<double> gradients(positions.size(), 0.0);
    std::array<double, 9> strain{};
    CHECK(xtbloom::detail::gfn2::add_periodic_integrals_h0_vjp_cpu(
              model.basis, model.integrals, model.h0, reference.periodic, reference.topology,
              reference.geometry, reference.geometry_workspace, coordination.data(),
              adjoints.overlap.data(), adjoints.dipole.data(), adjoints.quadrupole.data(),
              adjoints.hamiltonian.data(), dE_dcn.data(), gradients.data(), strain.data(),
              reference.integral_workspace.data, reference.periodic.workspace_size_bytes(),
              error) == XTBLOOM_STATUS_SUCCESS);
    if (term == 3u) {
      CHECK(xtbloom::detail::gfn2::add_periodic_coordination_gradient_cpu(
                model.coordination_plan, reference.topology, reference.geometry, dE_dcn.data(),
                gradients.data(), strain.data(), reference.geometry_workspace,
                error) == XTBLOOM_STATUS_SUCCESS);
    }
    const double expected = term == 0u ? overlap_estimate[1] : h0_estimate[1];
    CHECK(near(gradients[coordinate], expected, 5.0e-8));
  }
  return 0;
}

int test_molecular_limit_and_invariance(const Model& model, const std::vector<double>& positions,
                                        const std::vector<double>& cell,
                                        const std::vector<double>& coordination) {
  std::string error;
  std::vector<double> large_cell{100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0};
  PeriodicEvaluation periodic;
  MatrixOutputs periodic_outputs;
  CHECK(periodic.initialize(model, positions, large_cell, error));
  CHECK(periodic.evaluate(coordination, periodic_outputs, error));

  const std::size_t matrix_elements =
      static_cast<std::size_t>(model.integrals.total_matrix_elements);
  MatrixOutputs molecular;
  molecular.overlap.resize(matrix_elements);
  molecular.dipole.resize(kDipoleComponents * matrix_elements);
  molecular.quadrupole.resize(kQuadrupoleComponents * matrix_elements);
  molecular.hamiltonian.resize(matrix_elements);
  AlignedWorkspace molecular_workspace(model.integrals.workspace_size_bytes);
  CHECK(xtbloom::detail::gfn2::evaluate_overlap_cpu(
            model.basis, model.integrals, positions.data(), molecular.overlap.data(),
            molecular_workspace.data, model.integrals.workspace_size_bytes,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::evaluate_multipole_cpu(
            model.basis, model.integrals, positions.data(), molecular.dipole.data(),
            molecular.quadrupole.data(), molecular_workspace.data,
            model.integrals.workspace_size_bytes, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::evaluate_h0_cpu(
            model.basis, model.integrals, model.h0, positions.data(), coordination.data(),
            molecular.overlap.data(), molecular.hamiltonian.data(),
            error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t index = 0; index < molecular.overlap.size(); ++index) {
    CHECK(near(periodic_outputs.overlap[index], molecular.overlap[index], 2.0e-14));
    CHECK(near(periodic_outputs.hamiltonian[index], molecular.hamiltonian[index], 2.0e-14));
  }
  for (std::size_t index = 0; index < molecular.dipole.size(); ++index) {
    CHECK(near(periodic_outputs.dipole[index], molecular.dipole[index], 2.0e-14));
  }
  for (std::size_t index = 0; index < molecular.quadrupole.size(); ++index) {
    CHECK(near(periodic_outputs.quadrupole[index], molecular.quadrupole[index], 3.0e-14));
  }

  PeriodicEvaluation reference;
  MatrixOutputs reference_outputs;
  CHECK(reference.initialize(model, positions, cell, error));
  CHECK(reference.evaluate(coordination, reference_outputs, error));

  auto shifted_positions = positions;
  for (std::size_t axis = 0; axis < 3u; ++axis) {
    shifted_positions[3u + axis] += cell[axis];
    shifted_positions[6u + axis] -= cell[6u + axis];
  }
  PeriodicEvaluation shifted;
  MatrixOutputs shifted_outputs;
  CHECK(shifted.initialize(model, shifted_positions, cell, error));
  CHECK(shifted.evaluate(coordination, shifted_outputs, error));
  for (std::size_t index = 0; index < matrix_elements; ++index) {
    CHECK(near(shifted_outputs.overlap[index], reference_outputs.overlap[index], 5.0e-13));
    CHECK(near(shifted_outputs.hamiltonian[index], reference_outputs.hamiltonian[index], 5.0e-13));
  }
  for (std::size_t index = 0; index < shifted_outputs.dipole.size(); ++index) {
    CHECK(near(shifted_outputs.dipole[index], reference_outputs.dipole[index], 5.0e-13));
  }
  for (std::size_t index = 0; index < shifted_outputs.quadrupole.size(); ++index) {
    CHECK(near(shifted_outputs.quadrupole[index], reference_outputs.quadrupole[index], 8.0e-13));
  }

  std::vector<double> basis_cell{
      cell[0] + cell[3], cell[1] + cell[4], cell[2] + cell[5], cell[3], cell[4],
      cell[5],           cell[6],           cell[7],           cell[8],
  };
  PeriodicEvaluation basis_changed;
  MatrixOutputs basis_outputs;
  CHECK(basis_changed.initialize(model, positions, basis_cell, error));
  CHECK(basis_changed.evaluate(coordination, basis_outputs, error));
  for (std::size_t index = 0; index < matrix_elements; ++index) {
    CHECK(near(basis_outputs.overlap[index], reference_outputs.overlap[index], 5.0e-12));
    CHECK(near(basis_outputs.hamiltonian[index], reference_outputs.hamiltonian[index], 5.0e-12));
  }
  for (std::size_t index = 0; index < basis_outputs.dipole.size(); ++index) {
    CHECK(near(basis_outputs.dipole[index], reference_outputs.dipole[index], 5.0e-12));
  }
  for (std::size_t index = 0; index < basis_outputs.quadrupole.size(); ++index) {
    CHECK(near(basis_outputs.quadrupole[index], reference_outputs.quadrupole[index], 8.0e-12));
  }
  return 0;
}

int test_ragged_empty_member_matches_sequential(const Model& model,
                                                const std::vector<double>& positions,
                                                const std::vector<double>& cell,
                                                const std::vector<double>& coordination) {
  std::string error;
  constexpr std::array<std::int64_t, 3> offsets{0, 0, 3};
  BasisPlan basis;
  IntegralPlan integrals;
  H0Plan h0;
  CHECK(xtbloom::detail::gfn2::make_basis_plan(2, 3, offsets.data(), model.atomic_numbers.data(),
                                               basis, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::make_integral_plan(basis, integrals, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::make_h0_plan(basis, integrals, model.atomic_numbers.data(), h0,
                                            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(integrals.matrix_offsets == std::vector<std::int64_t>({0, 0, 36}));

  std::vector<double> cells = cell;
  cells.insert(cells.end(), cell.begin(), cell.end());
  PeriodicShortRangePlan topology;
  CHECK(xtbloom::detail::gfn2::make_periodic_short_range_plan(
            2, 3, offsets.data(), cells.data(), topology, error) == XTBLOOM_STATUS_SUCCESS);
  AlignedWorkspace geometry_storage(topology.workspace_size_bytes());
  PeriodicShortRangeWorkspace geometry_workspace;
  PeriodicShortRangeGeometry geometry;
  CHECK(xtbloom::detail::gfn2::bind_periodic_short_range_workspace(
            topology, geometry_storage.data, topology.workspace_size_bytes(), geometry_workspace,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::update_periodic_short_range_geometry_cpu(
            topology, positions.data(), 1u, geometry_workspace, geometry, error) ==
        XTBLOOM_STATUS_SUCCESS);
  PeriodicIntegralPlan periodic;
  CHECK(xtbloom::detail::gfn2::make_periodic_integral_plan(basis, integrals, topology, periodic,
                                                           error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(periodic.translations(0).size == 125);
  CHECK(periodic.translations(1).size == 125);
  AlignedWorkspace workspace(periodic.workspace_size_bytes());
  MatrixOutputs batched;
  batched.overlap.resize(36u);
  batched.dipole.resize(kDipoleComponents * 36u);
  batched.quadrupole.resize(kQuadrupoleComponents * 36u);
  batched.hamiltonian.resize(36u);
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_integrals_h0_cpu(
            basis, integrals, h0, periodic, topology, geometry, geometry_workspace,
            coordination.data(), batched.overlap.data(), batched.dipole.data(),
            batched.quadrupole.data(), batched.hamiltonian.data(), workspace.data,
            periodic.workspace_size_bytes(), error) == XTBLOOM_STATUS_SUCCESS);

  PeriodicEvaluation sequential;
  MatrixOutputs reference;
  CHECK(sequential.initialize(model, positions, cell, error));
  CHECK(sequential.evaluate(coordination, reference, error));
  CHECK(batched.overlap == reference.overlap);
  CHECK(batched.dipole == reference.dipole);
  CHECK(batched.quadrupole == reference.quadrupole);
  CHECK(batched.hamiltonian == reference.hamiltonian);
  return 0;
}

int test_transactional_failures(const Model& model, const std::vector<double>& positions,
                                const std::vector<double>& cell,
                                const std::vector<double>& coordination) {
  std::string error;
  PeriodicEvaluation evaluation;
  CHECK(evaluation.initialize(model, positions, cell, error));
  const std::size_t matrix_elements =
      static_cast<std::size_t>(model.integrals.total_matrix_elements);
  std::vector<double> overlap(matrix_elements, 11.0);
  std::vector<double> dipole(kDipoleComponents * matrix_elements, 12.0);
  std::vector<double> quadrupole(kQuadrupoleComponents * matrix_elements, 13.0);
  std::vector<double> hamiltonian(matrix_elements, 14.0);
  auto bad_coordination = coordination;
  bad_coordination[1] = std::numeric_limits<double>::quiet_NaN();
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_integrals_h0_cpu(
            model.basis, model.integrals, model.h0, evaluation.periodic, evaluation.topology,
            evaluation.geometry, evaluation.geometry_workspace, bad_coordination.data(),
            overlap.data(), dipole.data(), quadrupole.data(), hamiltonian.data(),
            evaluation.integral_workspace.data, evaluation.periodic.workspace_size_bytes(),
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(overlap.begin(), overlap.end(), [](double value) { return value == 11.0; }));
  CHECK(std::all_of(dipole.begin(), dipole.end(), [](double value) { return value == 12.0; }));
  CHECK(std::all_of(quadrupole.begin(), quadrupole.end(),
                    [](double value) { return value == 13.0; }));
  CHECK(std::all_of(hamiltonian.begin(), hamiltonian.end(),
                    [](double value) { return value == 14.0; }));

  const double wrapped_position = evaluation.geometry_workspace.wrapped_positions[0];
  evaluation.geometry_workspace.wrapped_positions[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_integrals_h0_cpu(
            model.basis, model.integrals, model.h0, evaluation.periodic, evaluation.topology,
            evaluation.geometry, evaluation.geometry_workspace, coordination.data(), overlap.data(),
            dipole.data(), quadrupole.data(), hamiltonian.data(),
            evaluation.integral_workspace.data, evaluation.periodic.workspace_size_bytes(),
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(overlap.begin(), overlap.end(), [](double value) { return value == 11.0; }));
  CHECK(std::all_of(dipole.begin(), dipole.end(), [](double value) { return value == 12.0; }));
  CHECK(std::all_of(quadrupole.begin(), quadrupole.end(),
                    [](double value) { return value == 13.0; }));
  CHECK(std::all_of(hamiltonian.begin(), hamiltonian.end(),
                    [](double value) { return value == 14.0; }));
  evaluation.geometry_workspace.wrapped_positions[0] = wrapped_position;

  CHECK(xtbloom::detail::gfn2::evaluate_periodic_integrals_h0_cpu(
            model.basis, model.integrals, model.h0, evaluation.periodic, evaluation.topology,
            evaluation.geometry, evaluation.geometry_workspace, coordination.data(), overlap.data(),
            dipole.data(), quadrupole.data(), hamiltonian.data(),
            evaluation.integral_workspace.data, evaluation.periodic.workspace_size_bytes() - 1u,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(overlap.begin(), overlap.end(), [](double value) { return value == 11.0; }));

  AlignedWorkspace misaligned_storage(evaluation.periodic.workspace_size_bytes() + 1u);
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_integrals_h0_cpu(
            model.basis, model.integrals, model.h0, evaluation.periodic, evaluation.topology,
            evaluation.geometry, evaluation.geometry_workspace, coordination.data(), overlap.data(),
            dipole.data(), quadrupole.data(), hamiltonian.data(),
            static_cast<std::byte*>(misaligned_storage.data) + 1u,
            evaluation.periodic.workspace_size_bytes(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(overlap.begin(), overlap.end(), [](double value) { return value == 11.0; }));

  CHECK(xtbloom::detail::gfn2::evaluate_periodic_integrals_h0_cpu(
            model.basis, model.integrals, model.h0, evaluation.periodic, evaluation.topology,
            evaluation.geometry, evaluation.geometry_workspace, coordination.data(), overlap.data(),
            dipole.data(), quadrupole.data(), overlap.data(), evaluation.integral_workspace.data,
            evaluation.periodic.workspace_size_bytes(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(overlap.begin(), overlap.end(), [](double value) { return value == 11.0; }));

  PeriodicShortRangeGeometry stale_geometry = evaluation.geometry;
  stale_geometry.geometry_generation = 0u;
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_integrals_h0_cpu(
            model.basis, model.integrals, model.h0, evaluation.periodic, evaluation.topology,
            stale_geometry, evaluation.geometry_workspace, coordination.data(), overlap.data(),
            dipole.data(), quadrupole.data(), hamiltonian.data(),
            evaluation.integral_workspace.data, evaluation.periodic.workspace_size_bytes(),
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  auto changed_cell = cell;
  changed_cell[0] += 0.25;
  PeriodicEvaluation changed;
  CHECK(changed.initialize(model, positions, changed_cell, error));
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_integrals_h0_cpu(
            model.basis, model.integrals, model.h0, evaluation.periodic, changed.topology,
            changed.geometry, changed.geometry_workspace, coordination.data(), overlap.data(),
            dipole.data(), quadrupole.data(), hamiltonian.data(),
            evaluation.integral_workspace.data, evaluation.periodic.workspace_size_bytes(),
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  const MatrixAdjoints adjoints = make_adjoint(model, 2u);
  auto bad_quadrupole = adjoints.quadrupole;
  bad_quadrupole[7] = std::numeric_limits<double>::infinity();
  std::vector<double> dE_dcn(coordination.size(), 21.0);
  std::vector<double> gradients(positions.size(), 22.0);
  std::array<double, 9> strain{};
  strain.fill(23.0);
  CHECK(xtbloom::detail::gfn2::add_periodic_integrals_h0_vjp_cpu(
            model.basis, model.integrals, model.h0, evaluation.periodic, evaluation.topology,
            evaluation.geometry, evaluation.geometry_workspace, coordination.data(),
            adjoints.overlap.data(), adjoints.dipole.data(), bad_quadrupole.data(),
            adjoints.hamiltonian.data(), dE_dcn.data(), gradients.data(), strain.data(),
            evaluation.integral_workspace.data, evaluation.periodic.workspace_size_bytes(),
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(dE_dcn.begin(), dE_dcn.end(), [](double value) { return value == 21.0; }));
  CHECK(
      std::all_of(gradients.begin(), gradients.end(), [](double value) { return value == 22.0; }));
  CHECK(std::all_of(strain.begin(), strain.end(), [](double value) { return value == 23.0; }));

  BasisPlan bad_basis = model.basis;
  bad_basis.minimum_primitive_exponent *= 2.0;
  PeriodicIntegralPlan survivor = evaluation.periodic;
  const auto* identity = survivor.identity();
  CHECK(xtbloom::detail::gfn2::make_periodic_integral_plan(bad_basis, model.integrals,
                                                           evaluation.topology, survivor, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(survivor.identity() == identity);

  BasisPlan stale_minimum_basis = model.basis;
  const auto minimum = std::min_element(stale_minimum_basis.primitive_exponents.begin(),
                                        stale_minimum_basis.primitive_exponents.end());
  CHECK(minimum != stale_minimum_basis.primitive_exponents.end());
  *minimum *= 0.5;
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_integrals_h0_cpu(
            stale_minimum_basis, model.integrals, model.h0, evaluation.periodic,
            evaluation.topology, evaluation.geometry, evaluation.geometry_workspace,
            coordination.data(), overlap.data(), dipole.data(), quadrupole.data(),
            hamiltonian.data(), evaluation.integral_workspace.data,
            evaluation.periodic.workspace_size_bytes(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(overlap.begin(), overlap.end(), [](double value) { return value == 11.0; }));

  MatrixOutputs reference;
  CHECK(evaluation.evaluate(coordination, reference, error));
  auto moved_positions = positions;
  moved_positions[3] += 0.05;
  CHECK(evaluation.update_positions(moved_positions, 2u, error));
  MatrixOutputs moved;
  CHECK(evaluation.evaluate(coordination, moved, error));
  CHECK(moved.overlap != reference.overlap);
  CHECK(evaluation.update_positions(positions, 3u, error));
  MatrixOutputs restored;
  CHECK(evaluation.evaluate(coordination, restored, error));
  for (std::size_t index = 0; index < reference.overlap.size(); ++index) {
    CHECK(near(restored.overlap[index], reference.overlap[index], 2.0e-14));
    CHECK(near(restored.hamiltonian[index], reference.hamiltonian[index], 2.0e-14));
  }
  return 0;
}

}  // namespace

int main() {
  RawFixture raw;
  std::string error;
  if (!read_raw_fixture("data/conformance/periodic/terms/raw/integrals-h0.txt", raw, error)) {
    return __LINE__;
  }
  const RawArray* positions_array = raw.find("positions_bohr");
  const RawArray* cell_array = raw.find("lattice_vectors_bohr_columns");
  const RawArray* coordination_array = raw.find("gfn2_dexp_cn");
  if (positions_array == nullptr || cell_array == nullptr || coordination_array == nullptr) {
    return __LINE__;
  }
  const std::vector<double> positions = positions_array->reals;
  const std::vector<double> cell = cell_array->reals;
  const std::vector<double> coordination = coordination_array->reals;

  Model model;
  if (!model.initialize(raw, error)) return __LINE__;
  if (const int line =
          test_oracle_matrices_and_translation_order(raw, model, positions, cell, coordination)) {
    return line;
  }
  if (const int line = test_complete_vjp_finite_differences(model, positions, cell, coordination)) {
    return line;
  }
  if (const int line = test_independent_displaced_oracle(model, positions, cell, coordination)) {
    return line;
  }
  if (const int line = test_molecular_limit_and_invariance(model, positions, cell, coordination)) {
    return line;
  }
  if (const int line =
          test_ragged_empty_member_matches_sequential(model, positions, cell, coordination)) {
    return line;
  }
  return test_transactional_failures(model, positions, cell, coordination);
}
