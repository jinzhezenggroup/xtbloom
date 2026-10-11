// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>
#include <string>
#include <vector>

#include "model/gfn1/force.hpp"
#include "model/gfn1/wavefunction.hpp"

#define CHECK(condition)                                                                   \
  do {                                                                                     \
    if (!(condition)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      return __LINE__;                                                                     \
    }                                                                                      \
  } while (false)

namespace {

using namespace xtbloom::detail::gfn1;

class AlignedStorage {
 public:
  AlignedStorage() = default;
  explicit AlignedStorage(std::size_t bytes) { reset(bytes); }
  ~AlignedStorage() { release(); }

  AlignedStorage(const AlignedStorage&) = delete;
  AlignedStorage& operator=(const AlignedStorage&) = delete;

  void reset(std::size_t bytes) {
    release();
    bytes_ = std::max<std::size_t>(bytes, 64u);
    data_ = ::operator new(bytes_, std::align_val_t(64u));
  }

  [[nodiscard]] void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

 private:
  void release() {
    if (data_ != nullptr) ::operator delete(data_, std::align_val_t(64u));
    data_ = nullptr;
    bytes_ = 0u;
  }

  void* data_ = nullptr;
  std::size_t bytes_ = 0u;
};

bool near(double actual, double expected, double tolerance = 5.0e-10) {
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::abs(actual - expected) <=
             tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

struct Fixture {
  std::vector<std::int64_t> atom_offsets{0, 2};
  std::vector<std::int32_t> atomic_numbers{1, 1};
  BasisPlan basis;
  IntegralPlan integrals;
  RepulsionPlan repulsion;
  H0Plan h0;
  WavefunctionLayout wavefunction;
  MullikenPlan mulliken;
  ES2Plan es2;
  ES2GeometryCache es2_cache;
  D3Plan d3;
  HalogenPlan halogen;
  ExternalPointChargePlan external;

  std::vector<double> positions{0.0, 0.0, -1.2, 0.0, 0.0, 1.2};
  std::vector<double> coordination;
  std::vector<double> overlap;
  std::vector<double> density;
  std::vector<double> spin_density;
  std::vector<double> energy_weighted_density;
  std::vector<double> shell_charges;
  std::vector<double> shell_potentials;
  std::vector<double> spin_shell_potentials;
  std::vector<double> scc_free_energy{-1.25};
  std::array<double, 3> point_position{2.5, 0.0, 0.0};
  std::array<double, 1> point_charge{0.75};
  std::array<double, 1> point_hardness{0.6};

  AlignedStorage integral_storage;
  AlignedStorage d3_storage;
  AlignedStorage halogen_storage;
  std::vector<double> es2_matrix;
  std::vector<double> es2_update_scratch;

  std::vector<double> energy_scratch;
  std::vector<double> component_energy_scratch;
  std::vector<double> total_gradient;
  std::vector<double> component_gradient;
  std::vector<double> component_staging;
  std::vector<double> force_scratch;
  std::vector<double> overlap_adjoint;
  std::vector<double> coordination_adjoint;
  std::vector<double> point_force_scratch;
  std::vector<double> es2_gradient_scratch;
  D3Workspace d3_workspace;
  HalogenWorkspace halogen_workspace;
  ForceWorkspace workspace;

  bool initialize(bool unrestricted, bool point_charges, std::string& error,
                  bool mixed_spin = false) {
    if (mixed_spin) {
      atom_offsets = {0, 2, 4};
      atomic_numbers = {1, 1, 1, 1};
      positions = {0.0, 0.0, -1.2, 0.0, 0.0, 1.2, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0};
      scc_free_energy = {-1.25, -1.15};
    }
    const std::int64_t batch = static_cast<std::int64_t>(atom_offsets.size() - 1u);
    const std::int64_t atoms = static_cast<std::int64_t>(atomic_numbers.size());
    std::vector<double> charges(static_cast<std::size_t>(batch), 0.0);
    std::vector<std::int32_t> unpaired(static_cast<std::size_t>(batch), unrestricted ? 2 : 0);
    std::vector<std::int32_t> spins(static_cast<std::size_t>(batch), unrestricted ? 2 : 1);
    if (mixed_spin) {
      unpaired = {0, 2};
      spins = {1, 2};
    }
    if (make_basis_plan(batch, atoms, atom_offsets.data(), atomic_numbers.data(), basis, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        make_integral_plan(basis, integrals, error) != XTBLOOM_STATUS_SUCCESS ||
        make_repulsion_plan(batch, atoms, atom_offsets.data(), atomic_numbers.data(), repulsion,
                            error) != XTBLOOM_STATUS_SUCCESS ||
        make_h0_plan(basis, integrals, atomic_numbers.data(), h0, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        make_wavefunction_layout(basis, atomic_numbers.data(), charges.data(), unpaired.data(),
                                 spins.data(), wavefunction, error) != XTBLOOM_STATUS_SUCCESS ||
        make_mulliken_plan(basis, integrals, wavefunction, mulliken, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        xtbloom::detail::gfn1::make_es2_plan(basis, atomic_numbers.data(), es2, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        make_d3_plan(batch, atoms, atom_offsets.data(), atomic_numbers.data(), d3, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        make_halogen_plan(batch, atoms, atom_offsets.data(), atomic_numbers.data(), halogen,
                          error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    if (point_charges) {
      std::vector<std::int64_t> point_offsets(static_cast<std::size_t>(batch) + 1u, 0);
      point_offsets.back() = 1;
      if (make_external_point_charge_plan(basis, es2, 1, point_offsets.data(), external, error) !=
          XTBLOOM_STATUS_SUCCESS) {
        return false;
      }
    }

    integral_storage.reset(integrals.workspace_size_bytes);
    d3_storage.reset(d3.workspace_size_bytes());
    halogen_storage.reset(halogen.workspace_size_bytes());
    if (bind_d3_workspace(d3, d3_storage.data(), d3_storage.size(), d3_workspace, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        bind_halogen_workspace(halogen, halogen_storage.data(), halogen_storage.size(),
                               halogen_workspace, error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }

    coordination.resize(static_cast<std::size_t>(basis.total_atoms));
    overlap.resize(static_cast<std::size_t>(integrals.total_matrix_elements));
    density.assign(overlap.size(), 0.0);
    const bool has_unrestricted = std::find(spins.begin(), spins.end(), 2) != spins.end();
    spin_density.assign(has_unrestricted ? overlap.size() : 0u, 0.0);
    energy_weighted_density.assign(overlap.size(), 0.0);
    shell_charges.assign(static_cast<std::size_t>(basis.total_shells), 0.0);
    shell_potentials.assign(shell_charges.size(), 0.0);
    spin_shell_potentials.assign(has_unrestricted ? shell_charges.size() : 0u, 0.0);
    if (evaluate_coordination_cpu(d3.coordination_plan(), positions.data(), coordination.data(),
                                  error) != XTBLOOM_STATUS_SUCCESS ||
        evaluate_overlap_cpu(basis, integrals, positions.data(), overlap.data(),
                             integral_storage.data(), integral_storage.size(),
                             error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    for (std::int64_t system = 0; system < basis.batch_size; ++system) {
      const std::int64_t begin = integrals.matrix_offsets[static_cast<std::size_t>(system)];
      const std::int64_t orbitals =
          basis.batch_orbital_offsets[static_cast<std::size_t>(system) + 1u] -
          basis.batch_orbital_offsets[static_cast<std::size_t>(system)];
      for (std::int64_t orbital = 0; orbital < orbitals; ++orbital) {
        density[static_cast<std::size_t>(begin + orbital * orbitals + orbital)] = 0.25;
        if (spins[static_cast<std::size_t>(system)] == 2) {
          spin_density[static_cast<std::size_t>(begin + orbital * orbitals + orbital)] = 0.05;
        }
      }
    }

    es2_matrix.resize(static_cast<std::size_t>(es2.total_matrix_elements()));
    es2_update_scratch.resize(es2_matrix.size());
    ES2Workspace update_workspace;
    update_workspace.matrix_scratch = es2_update_scratch.data();
    update_workspace.matrix_elements = es2.total_matrix_elements();
    if (update_es2_geometry_cache_cpu(es2, positions.data(), 1u, es2_matrix.data(),
                                      es2_matrix.size(), update_workspace, es2_cache,
                                      error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }

    const std::size_t coordinates = static_cast<std::size_t>(basis.total_atoms * 3);
    energy_scratch.resize(static_cast<std::size_t>(basis.batch_size));
    component_energy_scratch.resize(energy_scratch.size());
    total_gradient.resize(coordinates);
    component_gradient.resize(coordinates);
    component_staging.resize(6u * coordinates);
    force_scratch.resize(coordinates);
    overlap_adjoint.resize(overlap.size());
    coordination_adjoint.resize(static_cast<std::size_t>(basis.total_atoms));
    point_force_scratch.resize(point_charges ? 3u : 0u);
    es2_gradient_scratch.resize(coordinates);

    workspace.energy_scratch = energy_scratch.data();
    workspace.component_energy_scratch = component_energy_scratch.data();
    workspace.energy_elements = basis.batch_size;
    workspace.total_gradient = total_gradient.data();
    workspace.component_gradient = component_gradient.data();
    workspace.component_gradient_staging = component_staging.data();
    workspace.component_gradient_staging_elements =
        static_cast<std::int64_t>(component_staging.size());
    workspace.force_scratch = force_scratch.data();
    workspace.coordinate_elements = static_cast<std::int64_t>(coordinates);
    workspace.overlap_adjoint = overlap_adjoint.data();
    workspace.overlap_elements = integrals.total_matrix_elements;
    workspace.coordination_adjoint = coordination_adjoint.data();
    workspace.atom_elements = basis.total_atoms;
    workspace.point_force_scratch = point_charges ? point_force_scratch.data() : nullptr;
    workspace.point_force_elements = static_cast<std::int64_t>(point_force_scratch.size());
    workspace.integral_workspace = integral_storage.data();
    workspace.integral_workspace_size = integral_storage.size();
    workspace.es2_workspace.gradient_scratch = es2_gradient_scratch.data();
    workspace.es2_workspace.gradient_elements = static_cast<std::int64_t>(coordinates);
    workspace.d3_workspace = d3_workspace;
    workspace.halogen_workspace = halogen_workspace;
    return true;
  }

  StationaryInput input(bool unrestricted, bool point_charges) const {
    StationaryInput result;
    result.atomic_numbers = atomic_numbers.data();
    result.positions = positions.data();
    result.coordination_numbers = coordination.data();
    result.geometry_generation = 1u;
    result.overlap = overlap.data();
    result.density = density.data();
    result.energy_weighted_density = energy_weighted_density.data();
    result.shell_charges = shell_charges.data();
    result.scalar_shell_potentials = shell_potentials.data();
    result.scc_free_energies = scc_free_energy.data();
    result.spin_density = unrestricted ? spin_density.data() : nullptr;
    result.spin_shell_potentials = unrestricted ? spin_shell_potentials.data() : nullptr;
    result.point_positions = point_charges ? point_position.data() : nullptr;
    result.point_charges = point_charges ? point_charge.data() : nullptr;
    result.point_hardnesses = point_charges ? point_hardness.data() : nullptr;
    return result;
  }

  xtbloom_status_t evaluate(const StationaryInput& stationary, double* energy, double* qm_forces,
                            double* point_forces, const ComponentGradients& components,
                            std::string& error) const {
    return evaluate_gfn1_energy_forces_cpu(
        basis, integrals, d3.coordination_plan(), repulsion, h0, mulliken, es2, es2_cache, d3,
        halogen, external.batch_size == 0 ? nullptr : &external, stationary, energy, qm_forces,
        point_forces, components, workspace, error);
  }
};

int test_unrestricted_spin_summed_matrix_extent_and_component_sum() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(true, false, error));
  std::array<double, 1> energy{77.0};
  std::vector<double> forces(6u, 77.0);
  std::array<std::vector<double>, 6> diagnostics;
  for (auto& values : diagnostics) values.assign(6u, 77.0);
  ComponentGradients components{diagnostics[0].data(), diagnostics[1].data(), diagnostics[2].data(),
                                diagnostics[3].data(), diagnostics[4].data(), nullptr};
  CHECK(fixture.evaluate(fixture.input(true, false), energy.data(), forces.data(), nullptr,
                         components, error) == XTBLOOM_STATUS_SUCCESS ||
        (std::fprintf(stderr, "unrestricted force failure: %s\n", error.c_str()), false));
  for (std::size_t coordinate = 0u; coordinate < forces.size(); ++coordinate) {
    double gradient = 0.0;
    for (std::size_t component = 0u; component < 5u; ++component) {
      gradient += diagnostics[component][coordinate];
    }
    CHECK(near(forces[coordinate], -gradient, 3.0e-10));
  }
  return 0;
}

int test_qm_only_point_charge_force_and_conservation() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(false, true, error));
  std::fill(fixture.shell_charges.begin(), fixture.shell_charges.end(), 0.2);
  std::array<double, 1> energy{};
  std::array<double, 6> qm_forces{};
  ComponentGradients components;
  CHECK(fixture.evaluate(fixture.input(false, true), energy.data(), qm_forces.data(), nullptr,
                         components, error) == XTBLOOM_STATUS_SUCCESS ||
        (std::fprintf(stderr, "QM-only PC force failure: %s\n", error.c_str()), false));

  std::array<double, 3> point_forces{};
  CHECK(fixture.evaluate(fixture.input(false, true), energy.data(), qm_forces.data(),
                         point_forces.data(), components, error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t axis = 0u; axis < 3u; ++axis) {
    CHECK(near(qm_forces[axis] + qm_forces[axis + 3u] + point_forces[axis], 0.0, 2.0e-12));
  }

  /* The external potential has no one-half factor. Its fixed-charge force is
   * the negative derivative of sum_s q_s V_s, as in pinned xTB/tblite PCEM. */
  constexpr double step = 2.0e-5;
  const double original = fixture.positions[0];
  const auto external_energy = [&](double coordinate) -> double {
    fixture.positions[0] = coordinate;
    std::vector<double> potential(fixture.shell_charges.size());
    std::string local_error;
    if (evaluate_external_point_charge_potential_cpu(
            fixture.external, fixture.positions.data(), fixture.point_position.data(),
            fixture.point_charge.data(), fixture.point_hardness.data(), potential.data(),
            local_error) != XTBLOOM_STATUS_SUCCESS) {
      std::fprintf(stderr, "external potential finite-difference failure: %s\n",
                   local_error.c_str());
      return std::numeric_limits<double>::quiet_NaN();
    }
    std::array<double, 1> value{};
    if (add_external_point_charge_energy_cpu(fixture.external, fixture.shell_charges.data(),
                                             potential.data(), value.data(),
                                             local_error) != XTBLOOM_STATUS_SUCCESS) {
      std::fprintf(stderr, "external energy finite-difference failure: %s\n", local_error.c_str());
      return std::numeric_limits<double>::quiet_NaN();
    }
    return value[0];
  };
  const double right = external_energy(original + step);
  const double left = external_energy(original - step);
  fixture.positions[0] = original;
  CHECK(near(qm_forces[0], -(right - left) / (2.0 * step), 2.0e-9));
  return 0;
}

int test_alias_and_late_failure_leave_all_outputs_unchanged() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(false, false, error));
  std::array<double, 1> energy{91.0};
  std::vector<double> forces(6u, 92.0);
  std::array<std::vector<double>, 6> diagnostics;
  for (auto& values : diagnostics) values.assign(6u, 93.0);
  ComponentGradients components{diagnostics[0].data(), diagnostics[1].data(),
                                diagnostics[2].data(), diagnostics[3].data(),
                                diagnostics[4].data(), diagnostics[5].data()};
  StationaryInput input = fixture.input(false, false);
  ES2GeometryCache stale = fixture.es2_cache;
  stale.geometry_generation = 2u;
  CHECK(evaluate_gfn1_energy_forces_cpu(
            fixture.basis, fixture.integrals, fixture.d3.coordination_plan(), fixture.repulsion,
            fixture.h0, fixture.mulliken, fixture.es2, stale, fixture.d3, fixture.halogen, nullptr,
            input, energy.data(), forces.data(), nullptr, components, fixture.workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energy[0] == 91.0);
  CHECK(std::all_of(forces.begin(), forces.end(), [](double value) { return value == 92.0; }));
  for (const auto& values : diagnostics) {
    CHECK(std::all_of(values.begin(), values.end(), [](double value) { return value == 93.0; }));
  }

  ForceWorkspace alias = fixture.workspace;
  alias.total_gradient = forces.data();
  CHECK(evaluate_gfn1_energy_forces_cpu(
            fixture.basis, fixture.integrals, fixture.d3.coordination_plan(), fixture.repulsion,
            fixture.h0, fixture.mulliken, fixture.es2, fixture.es2_cache, fixture.d3,
            fixture.halogen, nullptr, input, energy.data(), forces.data(), nullptr, components,
            alias, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energy[0] == 91.0);
  CHECK(std::all_of(forces.begin(), forces.end(), [](double value) { return value == 92.0; }));
  return 0;
}

int test_external_diagnostic_requires_external_plan() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(false, false, error));
  std::array<double, 1> energy{31.0};
  std::array<double, 6> forces{};
  std::array<double, 6> diagnostic{};
  diagnostic.fill(32.0);
  ComponentGradients components;
  components.external_point_charge = diagnostic.data();
  CHECK(fixture.evaluate(fixture.input(false, false), energy.data(), forces.data(), nullptr,
                         components, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energy[0] == 31.0);
  CHECK(std::all_of(diagnostic.begin(), diagnostic.end(),
                    [](double value) { return value == 32.0; }));
  return 0;
}

int test_force_diagnostic_requires_qm_force_output() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(false, false, error));
  std::array<double, 1> energy{33.0};
  std::array<double, 6> diagnostic{};
  ComponentGradients components;
  components.electronic = diagnostic.data();
  CHECK(fixture.evaluate(fixture.input(false, false), energy.data(), nullptr, nullptr, components,
                         error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(error == "GFN1 force composition requires the qm_forces output");
  CHECK(energy[0] == 33.0);
  return 0;
}

int test_foreign_same_extent_plan_is_rejected() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(false, false, error));
  RepulsionPlan forged = fixture.repulsion;
  forged.effective_charge[0] += 0.25;
  std::array<double, 1> energy{41.0};
  std::array<double, 6> forces{};
  ComponentGradients components;
  CHECK(evaluate_gfn1_energy_forces_cpu(
            fixture.basis, fixture.integrals, fixture.d3.coordination_plan(), forged, fixture.h0,
            fixture.mulliken, fixture.es2, fixture.es2_cache, fixture.d3, fixture.halogen, nullptr,
            fixture.input(false, false), energy.data(), forces.data(), nullptr, components,
            fixture.workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energy[0] == 41.0);
  return 0;
}

int test_mixed_batch_ignores_restricted_spin_slices() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(true, false, error, true));
  const std::int64_t restricted_matrix_end = fixture.integrals.matrix_offsets[1];
  std::fill_n(fixture.spin_density.data(), static_cast<std::size_t>(restricted_matrix_end), 0.75);
  const std::int64_t restricted_shell_end = fixture.basis.batch_shell_offsets[1];
  std::fill_n(fixture.spin_shell_potentials.data(), static_cast<std::size_t>(restricted_shell_end),
              -0.4);
  std::vector<double> energy(static_cast<std::size_t>(fixture.basis.batch_size));
  std::vector<double> forces(static_cast<std::size_t>(3 * fixture.basis.total_atoms));
  ComponentGradients components;
  CHECK(fixture.evaluate(fixture.input(true, false), energy.data(), forces.data(), nullptr,
                         components, error) == XTBLOOM_STATUS_SUCCESS);
  const std::vector<double> poisoned_forces = forces;

  std::fill_n(fixture.spin_density.data(), static_cast<std::size_t>(restricted_matrix_end), 0.0);
  std::fill_n(fixture.spin_shell_potentials.data(), static_cast<std::size_t>(restricted_shell_end),
              0.0);
  CHECK(fixture.evaluate(fixture.input(true, false), energy.data(), forces.data(), nullptr,
                         components, error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t coordinate = 0u; coordinate < forces.size(); ++coordinate) {
    CHECK(near(poisoned_forces[coordinate], forces[coordinate], 2.0e-12));
  }
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_unrestricted_spin_summed_matrix_extent_and_component_sum())
    return status;
  if (const int status = test_qm_only_point_charge_force_and_conservation()) return status;
  if (const int status = test_alias_and_late_failure_leave_all_outputs_unchanged()) return status;
  if (const int status = test_external_diagnostic_requires_external_plan()) return status;
  if (const int status = test_force_diagnostic_requires_qm_force_output()) return status;
  if (const int status = test_foreign_same_extent_plan_is_rejected()) return status;
  return test_mixed_batch_ignores_restricted_spin_slices();
}
