#include "model/gfn2/force.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "model/gfn2/es3.hpp"
#include "model/gfn2/wavefunction.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

using namespace xtbloom::detail::gfn2;

constexpr std::uint64_t kGeneration = 7007u;

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <=
         tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

struct AlignedBuffer {
  void* data = nullptr;
  std::size_t size = 0u;

  explicit AlignedBuffer(std::size_t requested) {
    size = (std::max<std::size_t>(requested, 1u) + 63u) & ~std::size_t{63u};
    data = std::aligned_alloc(64u, size);
    if (data != nullptr) {
      std::memset(data, 0, size);
    }
  }
  ~AlignedBuffer() { std::free(data); }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
};

struct GeometryState {
  std::vector<double> positions;
  std::vector<double> point_positions;
  std::vector<double> coordination;
  std::vector<double> overlap;
  std::vector<double> dipole_integrals;
  std::vector<double> quadrupole_integrals;
  std::vector<double> h0;
  std::vector<double> shell_charges;
  std::vector<double> atomic_charges;
  std::vector<double> atomic_dipoles;
  std::vector<double> atomic_quadrupoles;
  std::vector<double> scalar_shell_potentials;
  std::vector<double> dipole_potentials;
  std::vector<double> quadrupole_potentials;
  std::vector<double> scc_energies;
  std::vector<double> total_energies;
  std::vector<double> stationary_lagrangian;
};

struct Fixture {
  std::vector<int> kinds;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> point_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> reference_positions;
  std::vector<double> reference_point_positions;
  std::vector<double> point_charges;
  std::vector<double> point_hardnesses;

  BasisPlan basis;
  IntegralPlan integrals;
  CoordinationPlan coordination_plan;
  RepulsionPlan repulsion;
  H0Plan h0_plan;
  WavefunctionLayout wavefunction_layout;
  MullikenPlan mulliken;
  ES2Plan es2;
  ES3Plan es3;
  AES2Plan aes2;
  D4Plan d4;
  ExternalPointChargePlan external;

  std::vector<double> density;
  std::vector<double> population_density;
  std::vector<double> energy_weighted_density;
  std::vector<double> spin_density;
  std::vector<double> spin_scalar_shell_potentials;

  std::vector<std::byte> integral_workspace;
  std::vector<double> mulliken_scratch;

  std::vector<double> es2_matrix;
  std::vector<double> es2_matrix_scratch;
  std::vector<double> es2_shell_scratch;
  std::vector<double> es2_batch_scratch;
  std::vector<double> es2_gradient_scratch;
  ES2Workspace es2_workspace{};
  ES2GeometryCache es2_cache{};

  std::vector<double> aes2_pairs;
  std::vector<double> aes2_pair_scratch;
  std::vector<double> aes2_potential_scratch;
  std::vector<double> aes2_batch_scratch;
  std::vector<double> aes2_gradient_scratch;
  std::vector<double> aes2_coordination_scratch;
  AES2Workspace aes2_workspace{};
  AES2GeometryCache aes2_cache{};

  AlignedBuffer* d4_storage = nullptr;
  D4Workspace d4_workspace{};
  std::vector<double> d4_pairs;
  std::vector<double> d4_coordination;
  D4GeometryCache d4_cache{};

  std::vector<double> energy_scratch;
  std::vector<double> component_energy_scratch;
  std::vector<double> total_gradient;
  std::vector<double> component_gradient;
  std::vector<double> force_scratch;
  std::vector<double> point_force_scratch;
  std::vector<double> overlap_adjoint;
  std::vector<double> dipole_adjoint;
  std::vector<double> quadrupole_adjoint;
  std::vector<double> coordination_adjoint;
  RestrictedGfn2ForceWorkspace force_workspace{};

  ~Fixture() { delete d4_storage; }
  Fixture() = default;
  Fixture(const Fixture&) = delete;
  Fixture& operator=(const Fixture&) = delete;

  void append_system(int kind, double shift) {
    if (kind == 0) {
      atomic_numbers.insert(atomic_numbers.end(), {1, 1});
      reference_positions.insert(reference_positions.end(),
                                 {shift - 0.71, 0.12, -0.08, shift + 0.71, -0.12, 0.08});
      reference_point_positions.insert(reference_point_positions.end(),
                                       {shift + 0.23, 1.77, -0.91});
      point_charges.push_back(0.37);
      point_hardnesses.push_back(0.82);
    } else if (kind == 1) {
      atomic_numbers.insert(atomic_numbers.end(), {8, 1, 1});
      reference_positions.insert(
          reference_positions.end(),
          {shift + 0.05, -0.11, 0.17, shift + 1.48, 0.26, -0.34, shift - 0.39, 1.42, 0.51});
      reference_point_positions.insert(reference_point_positions.end(),
                                       {shift - 1.21, -0.84, 1.37});
      point_charges.push_back(-0.29);
      point_hardnesses.push_back(0.91);
    } else {
      atomic_numbers.insert(atomic_numbers.end(), {6, 1, 1, 1, 1});
      reference_positions.insert(reference_positions.end(),
                                 {shift, 0.0, 0.0, shift + 1.18, 1.18, 1.18, shift - 1.18, -1.18,
                                  1.18, shift - 1.18, 1.18, -1.18, shift + 1.18, -1.18, -1.18});
      reference_point_positions.insert(reference_point_positions.end(),
                                       {shift + 0.77, -2.13, 1.48});
      point_charges.push_back(0.21);
      point_hardnesses.push_back(1.07);
    }
    atom_offsets.push_back(static_cast<std::int64_t>(atomic_numbers.size()));
    point_offsets.push_back(static_cast<std::int64_t>(point_charges.size()));
  }

  bool initialize(const std::vector<int>& requested_kinds, std::string& error,
                  const std::vector<std::int32_t>& requested_spin_channels = {},
                  const std::vector<std::int32_t>& requested_unpaired_electrons = {}) {
    kinds = requested_kinds;
    atom_offsets = {0};
    point_offsets = {0};
    for (std::size_t system = 0; system < kinds.size(); ++system) {
      append_system(kinds[system], 13.0 * static_cast<double>(system));
    }
    const std::int64_t batch = static_cast<std::int64_t>(kinds.size());
    const std::int64_t atoms = static_cast<std::int64_t>(atomic_numbers.size());
    std::vector<double> molecular_charges(kinds.size(), 0.0);
    std::vector<std::int32_t> unpaired = requested_unpaired_electrons.empty()
                                             ? std::vector<std::int32_t>(kinds.size(), 0)
                                             : requested_unpaired_electrons;
    std::vector<std::int32_t> spins = requested_spin_channels.empty()
                                          ? std::vector<std::int32_t>(kinds.size(), 1)
                                          : requested_spin_channels;
    if (unpaired.size() != kinds.size() || spins.size() != kinds.size()) {
      error = "force test spin topology does not match its batch size";
      return false;
    }
    if (make_basis_plan(batch, atoms, atom_offsets.data(), atomic_numbers.data(), basis, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        make_integral_plan(basis, integrals, error) != XTBLOOM_STATUS_SUCCESS ||
        make_coordination_plan(batch, atoms, atom_offsets.data(), atomic_numbers.data(),
                               coordination_plan, error) != XTBLOOM_STATUS_SUCCESS ||
        make_repulsion_plan(batch, atoms, atom_offsets.data(), atomic_numbers.data(), repulsion,
                            error) != XTBLOOM_STATUS_SUCCESS ||
        make_h0_plan(basis, integrals, atomic_numbers.data(), h0_plan, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        make_wavefunction_layout(basis, atomic_numbers.data(), molecular_charges.data(),
                                 unpaired.data(), spins.data(), wavefunction_layout,
                                 error) != XTBLOOM_STATUS_SUCCESS ||
        make_mulliken_plan(basis, integrals, wavefunction_layout, mulliken, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        make_es2_plan(basis, atomic_numbers.data(), es2, error) != XTBLOOM_STATUS_SUCCESS ||
        make_es3_plan(basis, atomic_numbers.data(), es3, error) != XTBLOOM_STATUS_SUCCESS ||
        make_aes2_plan(basis, atomic_numbers.data(), aes2, error) != XTBLOOM_STATUS_SUCCESS ||
        make_d4_plan(batch, atoms, atom_offsets.data(), atomic_numbers.data(), d4, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        make_external_point_charge_plan(basis, atomic_numbers.data(), batch, point_offsets.data(),
                                        external, error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }

    const std::size_t matrix = static_cast<std::size_t>(integrals.total_matrix_elements);
    density.assign(matrix, 0.0);
    energy_weighted_density.assign(matrix, 0.0);
    for (std::size_t system = 0; system < kinds.size(); ++system) {
      const std::int64_t orbital_begin = basis.batch_orbital_offsets[system];
      const std::int64_t orbital_end = basis.batch_orbital_offsets[system + 1u];
      const std::int64_t orbitals = orbital_end - orbital_begin;
      const std::int64_t matrix_begin = integrals.matrix_offsets[system];
      for (std::int64_t row = 0; row < orbitals; ++row) {
        const std::int64_t global_orbital = orbital_begin + row;
        const std::int64_t shell = [&] {
          for (std::int64_t candidate = basis.batch_shell_offsets[system];
               candidate < basis.batch_shell_offsets[system + 1u]; ++candidate) {
            if (global_orbital >=
                    basis.shell_orbital_offsets[static_cast<std::size_t>(candidate)] &&
                global_orbital <
                    basis.shell_orbital_offsets[static_cast<std::size_t>(candidate + 1)]) {
              return candidate;
            }
          }
          return basis.batch_shell_offsets[system];
        }();
        const std::int64_t shell_orbitals =
            basis.shell_orbital_offsets[static_cast<std::size_t>(shell + 1)] -
            basis.shell_orbital_offsets[static_cast<std::size_t>(shell)];
        const double diagonal =
            wavefunction_layout.reference_shell_occupations[static_cast<std::size_t>(shell)] /
            static_cast<double>(shell_orbitals);
        density[static_cast<std::size_t>(matrix_begin + row * orbitals + row)] = diagonal;
        energy_weighted_density[static_cast<std::size_t>(matrix_begin + row * orbitals + row)] =
            -0.31 * diagonal + 0.007 * static_cast<double>(row + 1);
        for (std::int64_t column = row + 1; column < orbitals; ++column) {
          const double value = 0.002 * static_cast<double>(((row + 2) * (column + 3)) % 7 - 3);
          const double weighted = -0.013 * static_cast<double>(((row + 1) * (column + 4)) % 5 - 2);
          density[static_cast<std::size_t>(matrix_begin + row * orbitals + column)] = value;
          density[static_cast<std::size_t>(matrix_begin + column * orbitals + row)] = value;
          energy_weighted_density[static_cast<std::size_t>(matrix_begin + row * orbitals +
                                                           column)] = weighted;
          energy_weighted_density[static_cast<std::size_t>(matrix_begin + column * orbitals +
                                                           row)] = weighted;
        }
      }
    }
    if (std::find(spins.begin(), spins.end(), 2) != spins.end()) {
      /* Mixed-spin composer tests use zero canonical slices for restricted
       * systems and deterministic finite magnetization data for unrestricted
       * systems. The values need not be self-consistent SCC solutions: this
       * fixture isolates the stationary overlap-response contraction. */
      spin_density.assign(matrix, 0.0);
      spin_scalar_shell_potentials.assign(static_cast<std::size_t>(basis.total_shells), 0.0);
      for (std::size_t system = 0; system < kinds.size(); ++system) {
        if (spins[system] != 2) {
          continue;
        }
        const std::int64_t orbital_begin = basis.batch_orbital_offsets[system];
        const std::int64_t orbital_end = basis.batch_orbital_offsets[system + 1u];
        const std::int64_t orbitals = orbital_end - orbital_begin;
        const std::int64_t matrix_begin = integrals.matrix_offsets[system];
        for (std::int64_t orbital = 0; orbital < orbitals; ++orbital) {
          spin_density[static_cast<std::size_t>(matrix_begin + orbital * orbitals + orbital)] =
              0.11 / static_cast<double>(orbital + 1);
          for (std::int64_t column = orbital + 1; column < orbitals; ++column) {
            const std::int64_t row_atom =
                mulliken.orbital_to_atom()[static_cast<std::size_t>(orbital_begin + orbital)];
            const std::int64_t column_atom =
                mulliken.orbital_to_atom()[static_cast<std::size_t>(orbital_begin + column)];
            if (row_atom == column_atom) {
              continue;
            }
            const double coupling =
                0.003 * static_cast<double>(((orbital + 2) * (column + 1)) % 5 + 1);
            spin_density[static_cast<std::size_t>(matrix_begin + orbital * orbitals + column)] =
                coupling;
            spin_density[static_cast<std::size_t>(matrix_begin + column * orbitals + orbital)] =
                coupling;
          }
        }
        const std::int64_t shell_begin = basis.batch_shell_offsets[system];
        const std::int64_t shell_end = basis.batch_shell_offsets[system + 1u];
        for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
          spin_scalar_shell_potentials[static_cast<std::size_t>(shell)] =
              0.025 * static_cast<double>(shell - shell_begin + 1);
        }
      }
    } else {
      spin_density.clear();
      spin_scalar_shell_potentials.clear();
    }
    population_density.clear();
    population_density.reserve(static_cast<std::size_t>(mulliken.density_elements()));
    for (std::size_t system = 0; system < kinds.size(); ++system) {
      const std::size_t matrix_begin = static_cast<std::size_t>(integrals.matrix_offsets[system]);
      const std::size_t matrix_end =
          static_cast<std::size_t>(integrals.matrix_offsets[system + 1u]);
      if (spins[system] == 1) {
        population_density.insert(population_density.end(), density.begin() + matrix_begin,
                                  density.begin() + matrix_end);
        continue;
      }
      for (std::size_t element = matrix_begin; element < matrix_end; ++element) {
        population_density.push_back(0.5 * (density[element] + spin_density[element]));
      }
      for (std::size_t element = matrix_begin; element < matrix_end; ++element) {
        population_density.push_back(0.5 * (density[element] - spin_density[element]));
      }
    }

    integral_workspace.resize(integrals.workspace_size_bytes);
    mulliken_scratch.resize(static_cast<std::size_t>(
        std::max(mulliken.population_scratch_elements(), mulliken.hamiltonian_scratch_elements())));

    es2_matrix.resize(static_cast<std::size_t>(es2.total_matrix_elements()));
    es2_matrix_scratch.resize(es2_matrix.size());
    es2_shell_scratch.resize(static_cast<std::size_t>(es2.total_shells()));
    es2_batch_scratch.resize(kinds.size());
    es2_gradient_scratch.resize(static_cast<std::size_t>(atoms) * 3u);
    es2_workspace = {es2_matrix_scratch.data(),   es2.total_matrix_elements(),
                     es2_shell_scratch.data(),    es2.total_shells(),
                     es2_batch_scratch.data(),    batch,
                     es2_gradient_scratch.data(), atoms * 3};

    aes2_pairs.resize(static_cast<std::size_t>(aes2.pair_data_elements()));
    aes2_pair_scratch.resize(aes2_pairs.size());
    aes2_potential_scratch.resize(static_cast<std::size_t>(aes2.potential_scratch_elements()));
    aes2_batch_scratch.resize(kinds.size());
    aes2_gradient_scratch.resize(static_cast<std::size_t>(atoms) * 3u);
    aes2_coordination_scratch.resize(static_cast<std::size_t>(atoms));
    aes2_workspace = {aes2_pair_scratch.data(),         aes2.pair_data_elements(),
                      aes2_potential_scratch.data(),    aes2.potential_scratch_elements(),
                      aes2_batch_scratch.data(),        batch,
                      aes2_gradient_scratch.data(),     atoms * 3,
                      aes2_coordination_scratch.data(), atoms};

    d4_storage = new AlignedBuffer(d4.workspace_size_bytes());
    if (d4_storage->data == nullptr ||
        bind_d4_workspace(d4, d4_storage->data, d4.workspace_size_bytes(), d4_workspace, error) !=
            XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    d4_pairs.resize(static_cast<std::size_t>(d4.total_pairs()) * kD4PairDataElements);
    d4_coordination.resize(static_cast<std::size_t>(atoms));

    energy_scratch.resize(kinds.size());
    component_energy_scratch.resize(kinds.size());
    total_gradient.resize(static_cast<std::size_t>(atoms) * 3u);
    component_gradient.resize(total_gradient.size());
    force_scratch.resize(total_gradient.size());
    point_force_scratch.resize(kinds.size() * 3u);
    overlap_adjoint.resize(matrix);
    dipole_adjoint.resize(matrix * 3u);
    quadrupole_adjoint.resize(matrix * 6u);
    coordination_adjoint.resize(static_cast<std::size_t>(atoms));
    force_workspace = {
        energy_scratch.data(),
        component_energy_scratch.data(),
        batch,
        total_gradient.data(),
        component_gradient.data(),
        force_scratch.data(),
        atoms * 3,
        overlap_adjoint.data(),
        static_cast<std::int64_t>(overlap_adjoint.size()),
        dipole_adjoint.data(),
        static_cast<std::int64_t>(dipole_adjoint.size()),
        quadrupole_adjoint.data(),
        static_cast<std::int64_t>(quadrupole_adjoint.size()),
        coordination_adjoint.data(),
        atoms,
        point_force_scratch.data(),
        static_cast<std::int64_t>(point_force_scratch.size()),
        integral_workspace.data(),
        integral_workspace.size(),
        es2_workspace,
        aes2_workspace,
        d4_workspace,
    };
    return true;
  }

  bool evaluate(const std::vector<double>& positions, const std::vector<double>& point_positions,
                GeometryState& state, std::string& error) {
    state.positions = positions;
    state.point_positions = point_positions;
    const std::size_t atoms = atomic_numbers.size();
    const std::size_t shells = static_cast<std::size_t>(basis.total_shells);
    const std::size_t matrix = static_cast<std::size_t>(integrals.total_matrix_elements);
    state.coordination.resize(atoms);
    state.overlap.resize(matrix);
    state.dipole_integrals.resize(matrix * 3u);
    state.quadrupole_integrals.resize(matrix * 6u);
    state.h0.resize(matrix);
    state.shell_charges.resize(shells);
    state.atomic_charges.resize(atoms);
    state.atomic_dipoles.resize(atoms * 3u);
    state.atomic_quadrupoles.resize(atoms * 6u);
    state.scalar_shell_potentials.resize(shells);
    state.dipole_potentials.resize(atoms * 3u);
    state.quadrupole_potentials.resize(atoms * 6u);
    state.scc_energies.assign(kinds.size(), 0.0);
    state.total_energies.assign(kinds.size(), 0.0);
    state.stationary_lagrangian.assign(kinds.size(), 0.0);
    if (evaluate_coordination_cpu(coordination_plan, positions.data(), state.coordination.data(),
                                  error) != XTBLOOM_STATUS_SUCCESS ||
        evaluate_overlap_cpu(basis, integrals, positions.data(), state.overlap.data(),
                             integral_workspace.data(), integral_workspace.size(),
                             error) != XTBLOOM_STATUS_SUCCESS ||
        evaluate_multipole_cpu(basis, integrals, positions.data(), state.dipole_integrals.data(),
                               state.quadrupole_integrals.data(), integral_workspace.data(),
                               integral_workspace.size(), error) != XTBLOOM_STATUS_SUCCESS ||
        evaluate_h0_cpu(basis, integrals, h0_plan, positions.data(), state.coordination.data(),
                        state.overlap.data(), state.h0.data(), error) != XTBLOOM_STATUS_SUCCESS ||
        update_es2_geometry_cache_cpu(es2, positions.data(), kGeneration, es2_matrix.data(),
                                      es2_matrix.size(), es2_workspace, es2_cache,
                                      error) != XTBLOOM_STATUS_SUCCESS ||
        update_aes2_geometry_cache_cpu(
            aes2, positions.data(), state.coordination.data(), kGeneration, aes2_pairs.data(),
            aes2_pairs.size(), aes2_workspace, aes2_cache, error) != XTBLOOM_STATUS_SUCCESS ||
        update_d4_geometry_cache_cpu(d4, positions.data(), kGeneration, d4_pairs.data(),
                                     d4_pairs.size(), d4_coordination.data(),
                                     d4_coordination.size(), d4_workspace, d4_cache,
                                     error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }

    const MullikenIntegralView integral_view{
        state.overlap.data(), state.dipole_integrals.data(), state.quadrupole_integrals.data(),
        static_cast<std::int64_t>(matrix), mulliken.identity()};
    const MullikenDensityView density_view{population_density.data(),
                                           static_cast<std::int64_t>(population_density.size()),
                                           mulliken.identity()};
    std::vector<double> packed_shell_charges(
        static_cast<std::size_t>(mulliken.shell_population_elements()));
    std::vector<double> packed_atomic_charges(
        static_cast<std::size_t>(mulliken.atom_population_elements()));
    std::vector<double> packed_atomic_dipoles(
        static_cast<std::size_t>(mulliken.dipole_population_elements()));
    std::vector<double> packed_atomic_quadrupoles(
        static_cast<std::size_t>(mulliken.quadrupole_population_elements()));
    const MullikenPopulationView population_view{
        packed_shell_charges.data(),
        static_cast<std::int64_t>(packed_shell_charges.size()),
        packed_atomic_charges.data(),
        static_cast<std::int64_t>(packed_atomic_charges.size()),
        packed_atomic_dipoles.data(),
        static_cast<std::int64_t>(packed_atomic_dipoles.size()),
        packed_atomic_quadrupoles.data(),
        static_cast<std::int64_t>(packed_atomic_quadrupoles.size()),
        mulliken.identity()};
    const MullikenWorkspace mulliken_workspace{mulliken_scratch.data(),
                                               static_cast<std::int64_t>(mulliken_scratch.size())};
    if (evaluate_mulliken_population_cpu(mulliken, integral_view, density_view, population_view,
                                         mulliken_workspace, error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    std::size_t shell_cursor = 0u;
    std::size_t atom_cursor = 0u;
    std::size_t dipole_cursor = 0u;
    std::size_t quadrupole_cursor = 0u;
    for (std::size_t system = 0; system < kinds.size(); ++system) {
      const std::size_t shell_begin = static_cast<std::size_t>(basis.batch_shell_offsets[system]);
      const std::size_t shell_count = static_cast<std::size_t>(
          basis.batch_shell_offsets[system + 1u] - basis.batch_shell_offsets[system]);
      const std::size_t atom_begin = static_cast<std::size_t>(atom_offsets[system]);
      const std::size_t atom_count =
          static_cast<std::size_t>(atom_offsets[system + 1u] - atom_offsets[system]);
      std::copy_n(packed_shell_charges.begin() + shell_cursor, shell_count,
                  state.shell_charges.begin() + shell_begin);
      std::copy_n(packed_atomic_charges.begin() + atom_cursor, atom_count,
                  state.atomic_charges.begin() + atom_begin);
      std::copy_n(packed_atomic_dipoles.begin() + dipole_cursor, atom_count * 3u,
                  state.atomic_dipoles.begin() + atom_begin * 3u);
      std::copy_n(packed_atomic_quadrupoles.begin() + quadrupole_cursor, atom_count * 6u,
                  state.atomic_quadrupoles.begin() + atom_begin * 6u);
      const std::size_t channels = static_cast<std::size_t>(mulliken.spin_channels()[system]);
      shell_cursor += channels * shell_count;
      atom_cursor += channels * atom_count;
      dipole_cursor += channels * atom_count * 3u;
      quadrupole_cursor += channels * atom_count * 6u;
    }

    std::vector<double> shell_component(shells);
    std::vector<double> atomic_potential(atoms);
    std::vector<double> d4_potential(atoms);
    std::vector<double> d4_energies(kinds.size());
    if (evaluate_es2_potential_cpu(es2, es2_cache, state.shell_charges.data(),
                                   shell_component.data(), es2_workspace,
                                   error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    state.scalar_shell_potentials = shell_component;
    if (evaluate_es3_potential_cpu(make_es3_view(es3), state.shell_charges.data(),
                                   shell_component.data(), error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    for (std::size_t shell = 0; shell < shells; ++shell) {
      state.scalar_shell_potentials[shell] += shell_component[shell];
    }
    if (evaluate_aes2_potential_cpu(aes2, aes2_cache, state.atomic_charges.data(),
                                    state.atomic_dipoles.data(), state.atomic_quadrupoles.data(),
                                    atomic_potential.data(), state.dipole_potentials.data(),
                                    state.quadrupole_potentials.data(), aes2_workspace,
                                    error) != XTBLOOM_STATUS_SUCCESS ||
        evaluate_d4_two_body_cpu(d4, d4_cache, state.atomic_charges.data(), d4_energies.data(),
                                 d4_potential.data(), d4_workspace,
                                 error) != XTBLOOM_STATUS_SUCCESS ||
        evaluate_external_point_charge_potential_cpu(
            external, positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), shell_component.data(), error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    for (std::size_t shell = 0; shell < shells; ++shell) {
      const std::size_t atom = static_cast<std::size_t>(basis.shell_to_atom[shell]);
      state.scalar_shell_potentials[shell] +=
          atomic_potential[atom] + d4_potential[atom] + shell_component[shell];
    }

    for (std::size_t system = 0; system < kinds.size(); ++system) {
      const std::int64_t matrix_begin = integrals.matrix_offsets[system];
      const std::int64_t matrix_end = integrals.matrix_offsets[system + 1u];
      double core = 0.0;
      double pulay_constraint = 0.0;
      for (std::int64_t element = matrix_begin; element < matrix_end; ++element) {
        core = std::fma(density[static_cast<std::size_t>(element)],
                        state.h0[static_cast<std::size_t>(element)], core);
        pulay_constraint =
            std::fma(energy_weighted_density[static_cast<std::size_t>(element)],
                     state.overlap[static_cast<std::size_t>(element)], pulay_constraint);
      }
      state.scc_energies[system] = core + d4_energies[system];
      state.stationary_lagrangian[system] = -pulay_constraint;
    }
    if (add_es2_energy_cpu(es2, es2_cache, state.shell_charges.data(), state.scc_energies.data(),
                           es2_workspace, error) != XTBLOOM_STATUS_SUCCESS ||
        add_es3_energy_cpu(make_es3_view(es3), state.shell_charges.data(),
                           state.scc_energies.data(), error) != XTBLOOM_STATUS_SUCCESS ||
        add_aes2_energy_cpu(aes2, aes2_cache, state.atomic_charges.data(),
                            state.atomic_dipoles.data(), state.atomic_quadrupoles.data(),
                            state.scc_energies.data(), aes2_workspace,
                            error) != XTBLOOM_STATUS_SUCCESS ||
        add_external_point_charge_energy_cpu(external, state.shell_charges.data(),
                                             shell_component.data(), state.scc_energies.data(),
                                             error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    state.total_energies = state.scc_energies;
    std::vector<double> atm(kinds.size());
    if (add_repulsion_cpu(repulsion, positions.data(), state.total_energies.data(), nullptr,
                          error) != XTBLOOM_STATUS_SUCCESS ||
        evaluate_d4_atm_cpu(d4, d4_cache, atm.data(), d4_workspace, error) !=
            XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    for (std::size_t system = 0; system < kinds.size(); ++system) {
      state.total_energies[system] += atm[system];
      state.stationary_lagrangian[system] += state.total_energies[system];
    }
    return true;
  }

  RestrictedGfn2StationaryInput stationary_input(const GeometryState& state) const {
    return {
        state.positions.data(),
        state.coordination.data(),
        kGeneration,
        state.overlap.data(),
        density.data(),
        energy_weighted_density.data(),
        state.shell_charges.data(),
        state.atomic_charges.data(),
        state.atomic_dipoles.data(),
        state.atomic_quadrupoles.data(),
        state.scalar_shell_potentials.data(),
        state.dipole_potentials.data(),
        state.quadrupole_potentials.data(),
        state.scc_energies.data(),
        state.point_positions.data(),
        point_charges.data(),
        point_hardnesses.data(),
        spin_density.empty() ? nullptr : spin_density.data(),
        spin_scalar_shell_potentials.empty() ? nullptr : spin_scalar_shell_potentials.data(),
    };
  }

  bool compose(const GeometryState& state, std::vector<double>& energies,
               std::vector<double>& qm_forces, std::vector<double>& point_forces,
               RestrictedGfn2ComponentGradients components, std::string& error) {
    energies.assign(kinds.size(), 0.0);
    qm_forces.assign(atomic_numbers.size() * 3u, 0.0);
    point_forces.assign(kinds.size() * 3u, 0.0);
    const RestrictedGfn2StationaryInput input = stationary_input(state);
    return evaluate_restricted_gfn2_energy_forces_cpu(
               basis, integrals, coordination_plan, repulsion, h0_plan, mulliken, es2, es2_cache,
               aes2, aes2_cache, &d4, &d4_cache, &external, input, energies.data(),
               qm_forces.data(), point_forces.data(), components, force_workspace,
               error) == XTBLOOM_STATUS_SUCCESS;
  }
};

double sum(const std::vector<double>& values) {
  double total = 0.0;
  for (double value : values) {
    total += value;
  }
  return total;
}

int test_complete_stationary_finite_difference() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize({1}, error));
  GeometryState baseline;
  CHECK(fixture.evaluate(fixture.reference_positions, fixture.reference_point_positions, baseline,
                         error));
  std::vector<double> electronic(baseline.positions.size());
  std::vector<double> repulsion(baseline.positions.size());
  std::vector<double> es2(baseline.positions.size());
  std::vector<double> aes2(baseline.positions.size());
  std::vector<double> d4_two(baseline.positions.size());
  std::vector<double> d4_atm(baseline.positions.size());
  std::vector<double> external(baseline.positions.size());
  const RestrictedGfn2ComponentGradients components{
      electronic.data(), repulsion.data(), es2.data(),     aes2.data(),
      d4_two.data(),     d4_atm.data(),    external.data()};
  std::vector<double> energies;
  std::vector<double> qm_forces;
  std::vector<double> point_forces;
  CHECK(fixture.compose(baseline, energies, qm_forces, point_forces, components, error));
  CHECK(near(energies[0], baseline.total_energies[0], 2.0e-13));
  for (std::size_t coordinate = 0; coordinate < qm_forces.size(); ++coordinate) {
    const double component_sum = electronic[coordinate] + repulsion[coordinate] + es2[coordinate] +
                                 aes2[coordinate] + d4_two[coordinate] + d4_atm[coordinate] +
                                 external[coordinate];
    CHECK(near(-qm_forces[coordinate], component_sum, 2.0e-12));
  }

  constexpr double step = 2.0e-5;
  for (std::size_t coordinate = 0; coordinate < baseline.positions.size(); ++coordinate) {
    std::vector<double> left_positions = baseline.positions;
    std::vector<double> right_positions = baseline.positions;
    left_positions[coordinate] -= step;
    right_positions[coordinate] += step;
    GeometryState left;
    GeometryState right;
    CHECK(fixture.evaluate(left_positions, baseline.point_positions, left, error));
    CHECK(fixture.evaluate(right_positions, baseline.point_positions, right, error));
    const double numerical_force =
        -(sum(right.stationary_lagrangian) - sum(left.stationary_lagrangian)) / (2.0 * step);
    CHECK(near(qm_forces[coordinate], numerical_force, 2.5e-5));
  }
  for (std::size_t coordinate = 0; coordinate < baseline.point_positions.size(); ++coordinate) {
    std::vector<double> left_points = baseline.point_positions;
    std::vector<double> right_points = baseline.point_positions;
    left_points[coordinate] -= step;
    right_points[coordinate] += step;
    GeometryState left;
    GeometryState right;
    CHECK(fixture.evaluate(baseline.positions, left_points, left, error));
    CHECK(fixture.evaluate(baseline.positions, right_points, right, error));
    const double numerical_force =
        -(sum(right.stationary_lagrangian) - sum(left.stationary_lagrangian)) / (2.0 * step);
    CHECK(near(point_forces[coordinate], numerical_force, 2.0e-8));
  }
  return 0;
}

using Rotation = std::array<double, 9>;

Rotation rotation() {
  constexpr double angle = 0.61;
  constexpr double inverse_norm = 0.40824829046386301637;
  const double x = inverse_norm;
  const double y = 2.0 * inverse_norm;
  const double z = -inverse_norm;
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  const double t = 1.0 - c;
  return {c + x * x * t,     x * y * t - z * s, x * z * t + y * s, y * x * t + z * s, c + y * y * t,
          y * z * t - x * s, z * x * t - y * s, z * y * t + x * s, c + z * z * t};
}

std::array<double, 3> rotate(const Rotation& matrix, const double* vector) {
  std::array<double, 3> result{};
  for (std::size_t row = 0; row < 3u; ++row) {
    for (std::size_t column = 0; column < 3u; ++column) {
      result[row] += matrix[row * 3u + column] * vector[column];
    }
  }
  return result;
}

int test_translation_rotation_and_total_force() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize({0}, error));
  GeometryState baseline;
  CHECK(fixture.evaluate(fixture.reference_positions, fixture.reference_point_positions, baseline,
                         error));
  std::vector<double> energies;
  std::vector<double> qm_forces;
  std::vector<double> point_forces;
  CHECK(fixture.compose(baseline, energies, qm_forces, point_forces, {}, error));

  for (std::size_t axis = 0; axis < 3u; ++axis) {
    double net = point_forces[axis];
    for (std::size_t atom = 0; atom < fixture.atomic_numbers.size(); ++atom) {
      net += qm_forces[atom * 3u + axis];
    }
    CHECK(std::abs(net) < 2.0e-11);
  }
  std::array<double, 3> torque{};
  for (std::size_t atom = 0; atom < fixture.atomic_numbers.size(); ++atom) {
    const double* position = baseline.positions.data() + atom * 3u;
    const double* force = qm_forces.data() + atom * 3u;
    torque[0] += position[1] * force[2] - position[2] * force[1];
    torque[1] += position[2] * force[0] - position[0] * force[2];
    torque[2] += position[0] * force[1] - position[1] * force[0];
  }
  torque[0] +=
      baseline.point_positions[1] * point_forces[2] - baseline.point_positions[2] * point_forces[1];
  torque[1] +=
      baseline.point_positions[2] * point_forces[0] - baseline.point_positions[0] * point_forces[2];
  torque[2] +=
      baseline.point_positions[0] * point_forces[1] - baseline.point_positions[1] * point_forces[0];
  for (double value : torque) {
    CHECK(std::abs(value) < 3.0e-10);
  }

  const Rotation transform = rotation();
  std::vector<double> rotated_positions = baseline.positions;
  std::vector<double> rotated_points = baseline.point_positions;
  for (std::size_t atom = 0; atom < fixture.atomic_numbers.size(); ++atom) {
    const auto value = rotate(transform, baseline.positions.data() + atom * 3u);
    std::copy(value.begin(), value.end(), rotated_positions.begin() + atom * 3u);
  }
  const auto point = rotate(transform, baseline.point_positions.data());
  std::copy(point.begin(), point.end(), rotated_points.begin());
  GeometryState rotated;
  CHECK(fixture.evaluate(rotated_positions, rotated_points, rotated, error));
  std::vector<double> rotated_energies;
  std::vector<double> rotated_qm_forces;
  std::vector<double> rotated_point_forces;
  CHECK(fixture.compose(rotated, rotated_energies, rotated_qm_forces, rotated_point_forces, {},
                        error));
  CHECK(near(rotated_energies[0], energies[0], 2.0e-11));
  for (std::size_t atom = 0; atom < fixture.atomic_numbers.size(); ++atom) {
    const auto expected = rotate(transform, qm_forces.data() + atom * 3u);
    for (std::size_t axis = 0; axis < 3u; ++axis) {
      CHECK(near(rotated_qm_forces[atom * 3u + axis], expected[axis], 2.0e-9));
    }
  }
  const auto expected_point = rotate(transform, point_forces.data());
  for (std::size_t axis = 0; axis < 3u; ++axis) {
    CHECK(near(rotated_point_forces[axis], expected_point[axis], 2.0e-9));
  }

  std::vector<double> translated_positions = baseline.positions;
  std::vector<double> translated_points = baseline.point_positions;
  constexpr std::array<double, 3> shift{3.4, -1.7, 2.2};
  for (std::size_t coordinate = 0; coordinate < translated_positions.size(); ++coordinate) {
    translated_positions[coordinate] += shift[coordinate % 3u];
  }
  for (std::size_t axis = 0; axis < 3u; ++axis) {
    translated_points[axis] += shift[axis];
  }
  GeometryState translated;
  CHECK(fixture.evaluate(translated_positions, translated_points, translated, error));
  std::vector<double> translated_energies;
  std::vector<double> translated_qm_forces;
  std::vector<double> translated_point_forces;
  CHECK(fixture.compose(translated, translated_energies, translated_qm_forces,
                        translated_point_forces, {}, error));
  CHECK(near(translated_energies[0], energies[0], 2.0e-11));
  for (std::size_t index = 0; index < qm_forces.size(); ++index) {
    CHECK(near(translated_qm_forces[index], qm_forces[index], 2.0e-10));
  }
  for (std::size_t index = 0; index < point_forces.size(); ++index) {
    CHECK(near(translated_point_forces[index], point_forces[index], 2.0e-10));
  }
  return 0;
}

int check_batch_matches_sequential(const std::vector<int>& kinds) {
  Fixture batch;
  std::string error;
  CHECK(batch.initialize(kinds, error));
  GeometryState batch_state;
  CHECK(batch.evaluate(batch.reference_positions, batch.reference_point_positions, batch_state,
                       error));
  std::vector<double> batch_energies;
  std::vector<double> batch_qm_forces;
  std::vector<double> batch_point_forces;
  CHECK(batch.compose(batch_state, batch_energies, batch_qm_forces, batch_point_forces, {}, error));

  std::size_t atom_cursor = 0u;
  for (std::size_t system = 0; system < kinds.size(); ++system) {
    Fixture sequential;
    CHECK(sequential.initialize({kinds[system]}, error));
    GeometryState state;
    CHECK(sequential.evaluate(sequential.reference_positions, sequential.reference_point_positions,
                              state, error));
    std::vector<double> energies;
    std::vector<double> qm_forces;
    std::vector<double> point_forces;
    CHECK(sequential.compose(state, energies, qm_forces, point_forces, {}, error));
    CHECK(near(batch_energies[system], energies[0], 3.0e-11));
    for (std::size_t coordinate = 0; coordinate < qm_forces.size(); ++coordinate) {
      CHECK(near(batch_qm_forces[atom_cursor * 3u + coordinate], qm_forces[coordinate], 3.0e-9));
    }
    for (std::size_t axis = 0; axis < 3u; ++axis) {
      CHECK(near(batch_point_forces[system * 3u + axis], point_forces[axis], 3.0e-10));
    }
    atom_cursor += sequential.atomic_numbers.size();
  }
  return 0;
}

int check_homogeneous_batch(std::size_t batch_size, int kind) {
  std::string error;
  Fixture batch;
  CHECK(batch.initialize(std::vector<int>(batch_size, kind), error));
  GeometryState batch_state;
  CHECK(batch.evaluate(batch.reference_positions, batch.reference_point_positions, batch_state,
                       error));
  std::vector<double> batch_energies;
  std::vector<double> batch_qm_forces;
  std::vector<double> batch_point_forces;
  CHECK(batch.compose(batch_state, batch_energies, batch_qm_forces, batch_point_forces, {}, error));

  Fixture sequential;
  CHECK(sequential.initialize({kind}, error));
  GeometryState sequential_state;
  CHECK(sequential.evaluate(sequential.reference_positions, sequential.reference_point_positions,
                            sequential_state, error));
  std::vector<double> energies;
  std::vector<double> qm_forces;
  std::vector<double> point_forces;
  CHECK(sequential.compose(sequential_state, energies, qm_forces, point_forces, {}, error));
  for (std::size_t system = 0; system < batch_size; ++system) {
    CHECK(near(batch_energies[system], energies[0], 3.0e-11));
    for (std::size_t coordinate = 0; coordinate < qm_forces.size(); ++coordinate) {
      CHECK(near(batch_qm_forces[system * qm_forces.size() + coordinate], qm_forces[coordinate],
                 3.0e-9));
    }
    for (std::size_t axis = 0; axis < 3u; ++axis) {
      CHECK(near(batch_point_forces[system * 3u + axis], point_forces[axis], 3.0e-10));
    }
  }
  return 0;
}

int test_ragged_batch_matches_sequential() {
  CHECK(check_batch_matches_sequential({0, 1, 2}) == 0);
  for (std::size_t batch_size : {1u, 8u, 32u, 128u}) {
    CHECK(check_homogeneous_batch(batch_size, 1) == 0);
  }
  return 0;
}

int test_energy_only_ignores_force_intermediates() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize({1}, error));
  GeometryState state;
  CHECK(fixture.evaluate(fixture.reference_positions, fixture.reference_point_positions, state,
                         error));
  RestrictedGfn2StationaryInput input{};
  input.positions = state.positions.data();
  input.scc_energies = state.scc_energies.data();
  RestrictedGfn2ForceWorkspace energy_workspace{};
  energy_workspace.energy_scratch = fixture.energy_scratch.data();
  energy_workspace.component_energy_scratch = fixture.component_energy_scratch.data();
  energy_workspace.energy_elements = 1;
  energy_workspace.d4_workspace = fixture.d4_workspace;
  std::array<double, 1> energy{-99.0};
  CHECK(evaluate_restricted_gfn2_energy_forces_cpu(
            fixture.basis, fixture.integrals, fixture.coordination_plan, fixture.repulsion,
            fixture.h0_plan, fixture.mulliken, fixture.es2, fixture.es2_cache, fixture.aes2,
            fixture.aes2_cache, &fixture.d4, &fixture.d4_cache, &fixture.external, input,
            energy.data(), nullptr, nullptr, {}, energy_workspace,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(near(energy[0], state.total_energies[0], 2.0e-13));
  return 0;
}

int test_unrestricted_spin_response_and_alias_atomicity() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize({0, 1}, error, {1, 2}, {0, 2}));
  GeometryState state;
  if (!fixture.evaluate(fixture.reference_positions, fixture.reference_point_positions, state,
                        error)) {
    std::cerr << "unrestricted composer fixture setup failed: " << error << '\n';
    return __LINE__;
  }

  const std::vector<double> physical_spin_density = fixture.spin_density;
  const std::vector<double> physical_spin_potential = fixture.spin_scalar_shell_potentials;
  std::fill(fixture.spin_density.begin(), fixture.spin_density.end(), 0.0);
  std::fill(fixture.spin_scalar_shell_potentials.begin(),
            fixture.spin_scalar_shell_potentials.end(), 0.0);
  std::vector<double> zero_spin_energies;
  std::vector<double> zero_spin_qm_forces;
  std::vector<double> zero_spin_point_forces;
  CHECK(fixture.compose(state, zero_spin_energies, zero_spin_qm_forces, zero_spin_point_forces, {},
                        error));

  fixture.spin_density = physical_spin_density;
  fixture.spin_scalar_shell_potentials = physical_spin_potential;
  std::vector<double> spin_energies;
  std::vector<double> spin_qm_forces;
  std::vector<double> spin_point_forces;
  CHECK(fixture.compose(state, spin_energies, spin_qm_forces, spin_point_forces, {}, error));
  CHECK(spin_energies == zero_spin_energies);
  CHECK(spin_point_forces == zero_spin_point_forces);

  const std::size_t restricted_coordinates = static_cast<std::size_t>(fixture.atom_offsets[1]) * 3u;
  for (std::size_t coordinate = 0; coordinate < restricted_coordinates; ++coordinate) {
    CHECK(spin_qm_forces[coordinate] == zero_spin_qm_forces[coordinate]);
  }
  bool unrestricted_force_changed = false;
  for (std::size_t coordinate = restricted_coordinates; coordinate < spin_qm_forces.size();
       ++coordinate) {
    unrestricted_force_changed |=
        std::abs(spin_qm_forces[coordinate] - zero_spin_qm_forces[coordinate]) > 1.0e-12;
  }
  CHECK(unrestricted_force_changed);

  RestrictedGfn2StationaryInput input = fixture.stationary_input(state);
  input.spin_scalar_shell_potentials = nullptr;
  std::vector<double> energies(fixture.kinds.size(), 91.0);
  std::vector<double> forces(state.positions.size(), 92.0);
  std::vector<double> points(state.point_positions.size(), 93.0);
  CHECK(evaluate_restricted_gfn2_energy_forces_cpu(
            fixture.basis, fixture.integrals, fixture.coordination_plan, fixture.repulsion,
            fixture.h0_plan, fixture.mulliken, fixture.es2, fixture.es2_cache, fixture.aes2,
            fixture.aes2_cache, &fixture.d4, &fixture.d4_cache, &fixture.external, input,
            energies.data(), forces.data(), points.data(), {}, fixture.force_workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energies == std::vector<double>(fixture.kinds.size(), 91.0));
  CHECK(forces == std::vector<double>(state.positions.size(), 92.0));
  CHECK(points == std::vector<double>(state.point_positions.size(), 93.0));

  input = fixture.stationary_input(state);
  const std::size_t alias_elements = std::max(
      static_cast<std::size_t>(fixture.integrals.total_matrix_elements), state.positions.size());
  std::vector<double> aliased_spin_and_force(alias_elements, 94.0);
  input.spin_density = aliased_spin_and_force.data();
  energies.assign(fixture.kinds.size(), 95.0);
  points.assign(state.point_positions.size(), 96.0);
  CHECK(evaluate_restricted_gfn2_energy_forces_cpu(
            fixture.basis, fixture.integrals, fixture.coordination_plan, fixture.repulsion,
            fixture.h0_plan, fixture.mulliken, fixture.es2, fixture.es2_cache, fixture.aes2,
            fixture.aes2_cache, &fixture.d4, &fixture.d4_cache, &fixture.external, input,
            energies.data(), aliased_spin_and_force.data(), points.data(), {},
            fixture.force_workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energies == std::vector<double>(fixture.kinds.size(), 95.0));
  CHECK(aliased_spin_and_force == std::vector<double>(alias_elements, 94.0));
  CHECK(points == std::vector<double>(state.point_positions.size(), 96.0));
  return 0;
}

int test_validation_and_progressive_diagnostics() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize({1}, error));
  GeometryState state;
  CHECK(fixture.evaluate(fixture.reference_positions, fixture.reference_point_positions, state,
                         error));
  const RestrictedGfn2StationaryInput input{
      state.positions.data(),
      state.coordination.data(),
      kGeneration,
      state.overlap.data(),
      fixture.density.data(),
      fixture.energy_weighted_density.data(),
      state.shell_charges.data(),
      state.atomic_charges.data(),
      state.atomic_dipoles.data(),
      state.atomic_quadrupoles.data(),
      state.scalar_shell_potentials.data(),
      state.dipole_potentials.data(),
      state.quadrupole_potentials.data(),
      state.scc_energies.data(),
      state.point_positions.data(),
      fixture.point_charges.data(),
      fixture.point_hardnesses.data(),
  };
  std::array<double, 1> energies{91.0};
  std::vector<double> points(3u, 92.0);
  CHECK(evaluate_restricted_gfn2_energy_forces_cpu(
            fixture.basis, fixture.integrals, fixture.coordination_plan, fixture.repulsion,
            fixture.h0_plan, fixture.mulliken, fixture.es2, fixture.es2_cache, fixture.aes2,
            fixture.aes2_cache, &fixture.d4, &fixture.d4_cache, &fixture.external, input,
            energies.data(), fixture.total_gradient.data(), points.data(), {},
            fixture.force_workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energies[0] == 91.0);
  CHECK(points == std::vector<double>(3u, 92.0));

  std::vector<double> unchanged_scc = state.scc_energies;
  CHECK(evaluate_restricted_gfn2_energy_forces_cpu(
            fixture.basis, fixture.integrals, fixture.coordination_plan, fixture.repulsion,
            fixture.h0_plan, fixture.mulliken, fixture.es2, fixture.es2_cache, fixture.aes2,
            fixture.aes2_cache, &fixture.d4, &fixture.d4_cache, &fixture.external, input,
            state.scc_energies.data(), fixture.force_scratch.data(), points.data(), {},
            fixture.force_workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(state.scc_energies == unchanged_scc);

  RestrictedGfn2ComponentGradients alias_component{};
  alias_component.electronic = fixture.density.data();
  energies[0] = 93.0;
  std::vector<double> forces(state.positions.size(), 41.0);
  CHECK(evaluate_restricted_gfn2_energy_forces_cpu(
            fixture.basis, fixture.integrals, fixture.coordination_plan, fixture.repulsion,
            fixture.h0_plan, fixture.mulliken, fixture.es2, fixture.es2_cache, fixture.aes2,
            fixture.aes2_cache, &fixture.d4, &fixture.d4_cache, &fixture.external, input,
            energies.data(), forces.data(), points.data(), alias_component, fixture.force_workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energies[0] == 93.0);

  std::vector<double> electronic(state.positions.size(), -73.0);
  points.assign(3u, 42.0);
  energies[0] = 43.0;
  RestrictedGfn2StationaryInput stale = input;
  stale.geometry_generation = kGeneration + 1u;
  const RestrictedGfn2ComponentGradients components{electronic.data(), nullptr, nullptr, nullptr,
                                                    nullptr,           nullptr, nullptr};
  CHECK(evaluate_restricted_gfn2_energy_forces_cpu(
            fixture.basis, fixture.integrals, fixture.coordination_plan, fixture.repulsion,
            fixture.h0_plan, fixture.mulliken, fixture.es2, fixture.es2_cache, fixture.aes2,
            fixture.aes2_cache, &fixture.d4, &fixture.d4_cache, &fixture.external, stale,
            energies.data(), forces.data(), points.data(), components, fixture.force_workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energies[0] == 43.0);
  CHECK(forces == std::vector<double>(state.positions.size(), 41.0));
  CHECK(points == std::vector<double>(3u, 42.0));
  CHECK(std::any_of(electronic.begin(), electronic.end(),
                    [](double value) { return value != -73.0; }));
  return 0;
}

}  // namespace

int main() {
  if (const int line = test_complete_stationary_finite_difference(); line != 0) {
    return line;
  }
  if (const int line = test_translation_rotation_and_total_force(); line != 0) {
    return line;
  }
  if (const int line = test_ragged_batch_matches_sequential(); line != 0) {
    return line;
  }
  if (const int line = test_energy_only_ignores_force_intermediates(); line != 0) {
    return line;
  }
  if (const int line = test_unrestricted_spin_response_and_alias_atomicity(); line != 0) {
    return line;
  }
  if (const int line = test_validation_and_progressive_diagnostics(); line != 0) {
    return line;
  }
  return 0;
}
