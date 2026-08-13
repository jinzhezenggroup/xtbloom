#include "model/gfn1/mulliken.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#define CHECK(condition)                                                                   \
  do {                                                                                     \
    if (!(condition)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      return __LINE__;                                                                     \
    }                                                                                      \
  } while (false)

namespace {

using namespace xtbloom::detail::gfn1;

bool near(double actual, double expected, double tolerance = 2.0e-13) {
  return std::abs(actual - expected) <=
         tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

struct Fixture {
  BasisPlan basis;
  IntegralPlan integrals;
  WavefunctionLayout wavefunction;
  MullikenPlan mulliken;
  std::vector<double> overlap;
  std::vector<double> density;
  std::vector<double> qsh;
  std::vector<double> qat;
  std::vector<double> potential;
  std::vector<double> hamiltonian;
  std::vector<double> scratch;
};

bool make_fixture(const std::vector<std::int64_t>& offsets,
                  const std::vector<std::int32_t>& numbers,
                  const std::vector<double>& charges,
                  const std::vector<std::int32_t>& unpaired,
                  const std::vector<std::int32_t>& spin_channels, Fixture& fixture,
                  std::string& error) {
  const auto batch = static_cast<std::int64_t>(offsets.size() - 1u);
  if (make_basis_plan(batch, static_cast<std::int64_t>(numbers.size()), offsets.data(),
                      numbers.data(), fixture.basis, error) != XTBLOOM_STATUS_SUCCESS ||
      make_integral_plan(fixture.basis, fixture.integrals, error) != XTBLOOM_STATUS_SUCCESS ||
      make_wavefunction_layout(fixture.basis, numbers.data(), charges.data(), unpaired.data(),
                               spin_channels.data(), fixture.wavefunction,
                               error) != XTBLOOM_STATUS_SUCCESS ||
      make_mulliken_plan(fixture.basis, fixture.integrals, fixture.wavefunction,
                         fixture.mulliken, error) != XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "GFN1 Mulliken fixture setup failed: %s\n", error.c_str());
    return false;
  }
  fixture.overlap.assign(static_cast<std::size_t>(fixture.mulliken.matrix_elements()), 0.0);
  fixture.density.assign(static_cast<std::size_t>(fixture.mulliken.density_elements()), 0.0);
  fixture.qsh.resize(static_cast<std::size_t>(fixture.mulliken.shell_population_elements()));
  fixture.qat.resize(static_cast<std::size_t>(fixture.mulliken.atom_population_elements()));
  fixture.potential.resize(fixture.qsh.size());
  fixture.hamiltonian.resize(fixture.density.size());
  fixture.scratch.resize(static_cast<std::size_t>(
      std::max(fixture.mulliken.population_scratch_elements(),
               fixture.mulliken.hamiltonian_scratch_elements())));
  return true;
}

int test_repeated_hydrogen_shell_population_and_reduction() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 1}, {1}, {0.0}, {1}, {1}, fixture, error));
  CHECK(fixture.basis.total_shells == 2);  // GFN1 hydrogen owns 1s and orthogonalized 2s.
  CHECK(fixture.wavefunction.reference_shell_occupations.size() == 2u);
  CHECK(fixture.wavefunction.reference_shell_occupations[0] == 1.0);
  CHECK(fixture.wavefunction.reference_shell_occupations[1] == 0.0);

  const std::int64_t orbitals = fixture.basis.total_orbitals;
  for (std::int64_t index = 0; index < orbitals; ++index) {
    fixture.overlap[static_cast<std::size_t>(index * orbitals + index)] = 1.0;
  }
  fixture.density[0] = 0.75;
  fixture.density[3] = 0.25;
  MullikenIntegralView integrals{fixture.overlap.data(), fixture.mulliken.matrix_elements(),
                                 fixture.mulliken.identity()};
  MullikenDensityView density{fixture.density.data(), fixture.mulliken.density_elements(),
                              fixture.mulliken.identity()};
  MullikenPopulationView population{fixture.qsh.data(),
                                    fixture.mulliken.shell_population_elements(),
                                    fixture.qat.data(),
                                    fixture.mulliken.atom_population_elements(),
                                    fixture.mulliken.identity()};
  MullikenWorkspace workspace{fixture.scratch.data(),
                              static_cast<std::int64_t>(fixture.scratch.size())};
  CHECK(evaluate_mulliken_population_cpu(fixture.mulliken, integrals, density, population,
                                         workspace, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(near(fixture.qsh[0], 0.25));
  CHECK(near(fixture.qsh[1], -0.25));
  CHECK(near(fixture.qat[0], 0.0));
  return 0;
}

int test_unrestricted_charge_magnetization_and_scalar_hamiltonian() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 1}, {1}, {0.0}, {1}, {2}, fixture, error));
  const std::int64_t orbitals = fixture.basis.total_orbitals;
  for (std::int64_t index = 0; index < orbitals; ++index) {
    fixture.overlap[static_cast<std::size_t>(index * orbitals + index)] = 1.0;
  }
  fixture.density[0] = 1.0;  // alpha
  MullikenIntegralView integrals{fixture.overlap.data(), fixture.mulliken.matrix_elements(),
                                 fixture.mulliken.identity()};
  MullikenDensityView density{fixture.density.data(), fixture.mulliken.density_elements(),
                              fixture.mulliken.identity()};
  MullikenPopulationView population{fixture.qsh.data(),
                                    fixture.mulliken.shell_population_elements(),
                                    fixture.qat.data(),
                                    fixture.mulliken.atom_population_elements(),
                                    fixture.mulliken.identity()};
  MullikenWorkspace workspace{fixture.scratch.data(),
                              static_cast<std::int64_t>(fixture.scratch.size())};
  CHECK(evaluate_mulliken_population_cpu(fixture.mulliken, integrals, density, population,
                                         workspace, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(near(fixture.qsh[0], 0.0));
  CHECK(near(fixture.qsh[2], -1.0));
  CHECK(near(fixture.qat[0], 0.0));
  CHECK(near(fixture.qat[1], -1.0));

  fixture.potential = {0.6, -0.2, 0.4, 0.8};
  MullikenPotentialView potential{fixture.potential.data(),
                                  fixture.mulliken.shell_population_elements(),
                                  fixture.mulliken.identity()};
  MullikenHamiltonianView hamiltonian{fixture.hamiltonian.data(),
                                      fixture.mulliken.density_elements(),
                                      fixture.mulliken.identity()};
  CHECK(add_mulliken_hamiltonian_system_cpu(fixture.mulliken, integrals, potential, hamiltonian,
                                            0, workspace, error) == XTBLOOM_STATUS_SUCCESS);
  /* The term-level primitive preserves tblite's half-valued conversion. The
   * SCC composition applies the later unrestricted-Hamiltonian factor. */
  CHECK(near(fixture.hamiltonian[0], -0.5));
  CHECK(near(fixture.hamiltonian[3], -0.3));
  CHECK(near(fixture.hamiltonian[4], -0.1));
  CHECK(near(fixture.hamiltonian[7], 0.5));
  return 0;
}

int test_peer_local_failure_leaves_target_unchanged() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 1, 2}, {1, 1}, {0.0, 0.0}, {1, 1}, {1, 1}, fixture, error));
  std::fill(fixture.overlap.begin(), fixture.overlap.end(), 0.0);
  for (std::size_t system = 0; system < 2u; ++system) {
    const std::int64_t matrix = fixture.integrals.matrix_offsets[system];
    fixture.overlap[static_cast<std::size_t>(matrix)] = 1.0;
    fixture.overlap[static_cast<std::size_t>(matrix + 3)] = 1.0;
  }
  fixture.density[0] = std::numeric_limits<double>::quiet_NaN();
  fixture.density[4] = 0.5;
  std::fill(fixture.qsh.begin(), fixture.qsh.end(), 77.0);
  std::fill(fixture.qat.begin(), fixture.qat.end(), 88.0);
  MullikenIntegralView integrals{fixture.overlap.data(), fixture.mulliken.matrix_elements(),
                                 fixture.mulliken.identity()};
  MullikenDensityView density{fixture.density.data(), fixture.mulliken.density_elements(),
                              fixture.mulliken.identity()};
  MullikenPopulationView population{fixture.qsh.data(),
                                    fixture.mulliken.shell_population_elements(),
                                    fixture.qat.data(),
                                    fixture.mulliken.atom_population_elements(),
                                    fixture.mulliken.identity()};
  MullikenWorkspace workspace{fixture.scratch.data(),
                              static_cast<std::int64_t>(fixture.scratch.size())};
  CHECK(evaluate_mulliken_population_system_cpu(fixture.mulliken, integrals, density, population,
                                                1, workspace,
                                                error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.qsh[0] == 77.0 && fixture.qsh[1] == 77.0);
  CHECK(fixture.qat[0] == 88.0);
  CHECK(evaluate_mulliken_population_system_cpu(fixture.mulliken, integrals, density, population,
                                                0, workspace,
                                                error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(fixture.qsh[0] == 77.0 && fixture.qsh[1] == 77.0 && fixture.qat[0] == 88.0);
  return 0;
}

int test_batch_hamiltonian_failure_is_transactional() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 1, 2}, {1, 1}, {0.0, 0.0}, {1, 1}, {1, 1}, fixture, error));
  std::fill(fixture.overlap.begin(), fixture.overlap.end(), 0.0);
  for (std::size_t system = 0; system < 2u; ++system) {
    const std::int64_t matrix = fixture.integrals.matrix_offsets[system];
    fixture.overlap[static_cast<std::size_t>(matrix)] = 1.0;
    fixture.overlap[static_cast<std::size_t>(matrix + 3)] = 1.0;
  }
  std::fill(fixture.hamiltonian.begin(), fixture.hamiltonian.end(), 4.0);
  std::fill(fixture.potential.begin(), fixture.potential.end(), 0.25);
  fixture.potential.back() = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> original = fixture.hamiltonian;

  MullikenIntegralView integrals{fixture.overlap.data(), fixture.mulliken.matrix_elements(),
                                 fixture.mulliken.identity()};
  MullikenPotentialView potential{fixture.potential.data(),
                                  fixture.mulliken.shell_population_elements(),
                                  fixture.mulliken.identity()};
  MullikenHamiltonianView hamiltonian{fixture.hamiltonian.data(),
                                      fixture.mulliken.density_elements(),
                                      fixture.mulliken.identity()};
  MullikenWorkspace workspace{fixture.scratch.data(),
                              static_cast<std::int64_t>(fixture.scratch.size())};
  CHECK(add_mulliken_hamiltonian_cpu(fixture.mulliken, integrals, potential, hamiltonian,
                                     workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.hamiltonian == original);
  return 0;
}

int test_aliasing_is_rejected_before_publication() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 1}, {1}, {0.0}, {1}, {1}, fixture, error));
  std::fill(fixture.overlap.begin(), fixture.overlap.end(), 0.0);
  fixture.overlap[0] = 1.0;
  fixture.overlap[3] = 1.0;
  fixture.density[0] = 0.25;
  fixture.density[3] = 0.75;

  MullikenIntegralView integrals{fixture.overlap.data(), fixture.mulliken.matrix_elements(),
                                 fixture.mulliken.identity()};
  MullikenDensityView density{fixture.density.data(), fixture.mulliken.density_elements(),
                              fixture.mulliken.identity()};
  MullikenPopulationView population{fixture.qsh.data(),
                                    fixture.mulliken.shell_population_elements(),
                                    fixture.qat.data(),
                                    fixture.mulliken.atom_population_elements(),
                                    fixture.mulliken.identity()};
  std::fill(fixture.qsh.begin(), fixture.qsh.end(), 71.0);
  std::fill(fixture.qat.begin(), fixture.qat.end(), 72.0);
  MullikenWorkspace population_alias{fixture.qsh.data(),
                                     fixture.mulliken.population_scratch_elements()};
  CHECK(evaluate_mulliken_population_cpu(fixture.mulliken, integrals, density, population,
                                         population_alias,
                                         error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(fixture.qsh.begin(), fixture.qsh.end(),
                    [](double value) { return value == 71.0; }));
  CHECK(std::all_of(fixture.qat.begin(), fixture.qat.end(),
                    [](double value) { return value == 72.0; }));

  std::fill(fixture.potential.begin(), fixture.potential.end(), 0.25);
  std::fill(fixture.hamiltonian.begin(), fixture.hamiltonian.end(), 4.0);
  const std::vector<double> original = fixture.hamiltonian;
  MullikenPotentialView potential{fixture.potential.data(),
                                  fixture.mulliken.shell_population_elements(),
                                  fixture.mulliken.identity()};
  MullikenHamiltonianView hamiltonian{fixture.hamiltonian.data(),
                                      fixture.mulliken.density_elements(),
                                      fixture.mulliken.identity()};
  MullikenWorkspace hamiltonian_alias{fixture.hamiltonian.data(),
                                      fixture.mulliken.hamiltonian_scratch_elements()};
  CHECK(add_mulliken_hamiltonian_cpu(fixture.mulliken, integrals, potential, hamiltonian,
                                     hamiltonian_alias,
                                     error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.hamiltonian == original);
  return 0;
}

}  // namespace

int main() {
  if (const int line = test_repeated_hydrogen_shell_population_and_reduction()) return line;
  if (const int line = test_unrestricted_charge_magnetization_and_scalar_hamiltonian()) return line;
  if (const int line = test_peer_local_failure_leaves_target_unchanged()) return line;
  if (const int line = test_batch_hamiltonian_failure_is_transactional()) return line;
  if (const int line = test_aliasing_is_rejected_before_publication()) return line;
  return 0;
}
