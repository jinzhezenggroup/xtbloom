#include "model/gfn2/external_point_charges.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <vector>

#include "model/gfn2/basis.hpp"

namespace allocation_test {
std::atomic<std::size_t> count{0};
std::atomic<bool> enabled{false};
}  // namespace allocation_test

void* operator new(std::size_t size) {
  if (allocation_test::enabled.load(std::memory_order_relaxed)) {
    allocation_test::count.fetch_add(1u, std::memory_order_relaxed);
  }
  if (void* pointer = std::malloc(size == 0u ? 1u : size); pointer != nullptr) {
    return pointer;
  }
  throw std::bad_alloc();
}

void* operator new[](std::size_t size) { return ::operator new(size); }

void operator delete(void* pointer) noexcept { std::free(pointer); }

void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }

void operator delete(void* pointer, std::size_t) noexcept { ::operator delete(pointer); }

void operator delete[](void* pointer, std::size_t) noexcept { ::operator delete[](pointer); }

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

using gpuxtb::detail::gfn2::BasisPlan;
using gpuxtb::detail::gfn2::ExternalPointChargePlan;

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

bool make_plan(const std::vector<std::int64_t>& atom_offsets,
               const std::vector<std::int32_t>& atomic_numbers,
               const std::vector<std::int64_t>* point_offsets, BasisPlan& basis,
               ExternalPointChargePlan& plan, std::string& error) {
  const std::int64_t batch_size = static_cast<std::int64_t>(atom_offsets.size() - 1u);
  if (gpuxtb::detail::gfn2::make_basis_plan(
          batch_size, static_cast<std::int64_t>(atomic_numbers.size()), atom_offsets.data(),
          atomic_numbers.data(), basis, error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  const std::int64_t point_count = point_offsets == nullptr ? 0 : point_offsets->back();
  const std::int64_t* offsets = point_offsets == nullptr ? nullptr : point_offsets->data();
  return gpuxtb::detail::gfn2::make_external_point_charge_plan(basis, atomic_numbers.data(),
                                                               point_count, offsets, plan,
                                                               error) == GPUXTB_STATUS_SUCCESS;
}

bool evaluate_energy(const ExternalPointChargePlan& plan, const std::vector<double>& qm_positions,
                     const std::vector<double>& point_positions,
                     const std::vector<double>& point_charges,
                     const std::vector<double>& point_hardnesses,
                     const std::vector<double>& shell_charges,
                     std::vector<double>& shell_potentials, std::vector<double>& energies,
                     std::string& error) {
  std::fill(energies.begin(), energies.end(), 0.0);
  const double* point_position_data = point_positions.empty() ? nullptr : point_positions.data();
  const double* point_charge_data = point_charges.empty() ? nullptr : point_charges.data();
  const double* point_hardness_data = point_hardnesses.empty() ? nullptr : point_hardnesses.data();
  if (gpuxtb::detail::gfn2::evaluate_external_point_charge_potential_cpu(
          plan, qm_positions.data(), point_position_data, point_charge_data, point_hardness_data,
          shell_potentials.data(), error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  return gpuxtb::detail::gfn2::add_external_point_charge_energy_cpu(
             plan, shell_charges.data(), shell_potentials.data(), energies.data(), error) ==
         GPUXTB_STATUS_SUCCESS;
}

double sum_values(const std::vector<double>& values) {
  double result = 0.0;
  for (double value : values) {
    result += value;
  }
  return result;
}

int test_xtb_671_water_golden() {
  const std::vector<std::int64_t> atom_offsets{0, 3};
  const std::vector<std::int32_t> atomic_numbers{8, 1, 1};
  const std::vector<std::int64_t> point_offsets{0, 1};
  BasisPlan basis;
  ExternalPointChargePlan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, atomic_numbers, &point_offsets, basis, plan, error));
  CHECK(error.empty());
  CHECK(plan.total_shells == 4);

  /*
   * The O 2p value locks down gamma_s = element gam * shell Hubbard scale;
   * the other three shells have scale one in the pinned GFN2 table.
   */
  CHECK(plan.shell_hardness[0] == 0.45189600000000002);
  CHECK(plan.shell_hardness[1] == 0.51954573499200007);
  CHECK(plan.shell_hardness[2] == 0.40577099999999999);
  CHECK(plan.shell_hardness[3] == 0.40577099999999999);

  const std::vector<double> qm_positions{
      0.0, 0.0, 0.0, 1.43233673, 0.0, 1.10715266, -1.43233673, 0.0, 1.10715266,
  };
  const std::vector<double> point_positions{4.0, 0.0, 0.0};
  const std::vector<double> point_charges{0.5};
  const std::vector<double> point_hardnesses{0.405771};

  /*
   * Converged qsh values were read from the Fortran unformatted xtbrestart
   * written by xTB 6.7.1 revision edcfbbe for docs/qmmm.md's water case.
   * They are ordered O(2s), O(2p), H(1s), H(1s).
   */
  const std::vector<double> shell_charges{
      0.26189717923223715,
      -0.8260775955268945,
      0.23530677010797196,
      0.3288736461866886,
  };
  std::vector<double> shell_potentials(4);
  std::vector<double> energies(1);
  CHECK(evaluate_energy(plan, qm_positions, point_positions, point_charges, point_hardnesses,
                        shell_charges, shell_potentials, energies, error));

  constexpr std::array<double, 4> expected_potential{
      0.10798911399580015,
      0.10997182482659495,
      0.13414824825075822,
      0.082411861188258814,
  };
  for (std::size_t shell = 0; shell < expected_potential.size(); ++shell) {
    CHECK(near(shell_potentials[shell], expected_potential[shell], 3.0e-16));
  }
  CHECK(near(energies[0], -0.003894135995627615, 4.0e-16));
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_energy_cpu(
            plan, shell_charges.data(), shell_potentials.data(), energies.data(), error) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(near(energies[0], -0.00778827199125523, 8.0e-16));

  std::vector<double> qm_forces(qm_positions.size());
  std::vector<double> point_forces(point_positions.size());
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), shell_charges.data(), qm_forces.data(), point_forces.data(),
            error) == GPUXTB_STATUS_SUCCESS);
  constexpr std::array<double, 9> expected_qm_forces{
      0.012301589617305368,   0.0, 0.0,
      -0.0058342831196417592, 0.0, 0.0025156889342053297,
      -0.0039998611358033368, 0.0, 0.00081520294419143728,
  };
  for (std::size_t coordinate = 0; coordinate < qm_forces.size(); ++coordinate) {
    CHECK(near(qm_forces[coordinate], expected_qm_forces[coordinate], 5.0e-16));
  }
  CHECK(near(point_forces[0], -0.0024674453618602722, 5.0e-16));
  CHECK(point_forces[1] == 0.0);
  CHECK(near(point_forces[2], -0.0033308918783967671, 5.0e-16));

  std::vector<double> qm_only(qm_positions.size());
  std::vector<double> point_only(point_positions.size());
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), shell_charges.data(), qm_only.data(), nullptr,
            error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), shell_charges.data(), nullptr, point_only.data(),
            error) == GPUXTB_STATUS_SUCCESS);
  CHECK(qm_only == qm_forces);
  CHECK(point_only == point_forces);

  /* docs/qmmm.md records the xTB gradient, which is minus our force. */
  CHECK(near(-point_forces[0], 0.0024674453618603, 3.0e-16));
  CHECK(near(-point_forces[2], 0.0033308918783968, 3.0e-16));
  for (std::size_t axis = 0; axis < 3u; ++axis) {
    double total_force = point_forces[axis];
    for (std::size_t atom = 0; atom < 3u; ++atom) {
      total_force += qm_forces[atom * 3u + axis];
    }
    CHECK(near(total_force, 0.0, 2.0e-18));
  }
  return 0;
}

int test_finite_difference_and_force_conservation() {
  const std::vector<std::int64_t> atom_offsets{0, 2, 4};
  const std::vector<std::int32_t> atomic_numbers{6, 8, 14, 1};
  const std::vector<std::int64_t> point_offsets{0, 2, 5};
  BasisPlan basis;
  ExternalPointChargePlan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, atomic_numbers, &point_offsets, basis, plan, error));

  std::vector<double> qm_positions{
      -0.4, 0.2, 1.1, 1.3, -0.7, 0.5, 2.0, 0.3, -1.2, 2.8, -0.9, 0.4,
  };
  std::vector<double> point_positions{
      3.1, 0.8, -0.2, -2.0, 1.4, 0.6, 0.2, -1.3, 2.2, 4.0, 0.1, -0.7, 2.5, 2.1, 1.4,
  };
  const std::vector<double> point_charges{0.7, -0.25, 0.1, -0.8, 0.35};
  const std::vector<double> point_hardnesses{0.42, 0.9, 1.7, 0.31, 999.0};
  std::vector<double> shell_charges(static_cast<std::size_t>(plan.total_shells));
  for (std::size_t shell = 0; shell < shell_charges.size(); ++shell) {
    shell_charges[shell] = 0.3 * std::sin(0.71 * static_cast<double>(shell + 1u)) - 0.11;
  }

  std::vector<double> shell_potentials(shell_charges.size());
  std::vector<double> energies(2);
  CHECK(evaluate_energy(plan, qm_positions, point_positions, point_charges, point_hardnesses,
                        shell_charges, shell_potentials, energies, error));
  std::vector<double> qm_forces(qm_positions.size());
  std::vector<double> point_forces(point_positions.size());
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), shell_charges.data(), qm_forces.data(), point_forces.data(),
            error) == GPUXTB_STATUS_SUCCESS);

  constexpr double step = 1.0e-5;
  for (std::size_t coordinate = 0; coordinate < qm_positions.size(); ++coordinate) {
    qm_positions[coordinate] += step;
    CHECK(evaluate_energy(plan, qm_positions, point_positions, point_charges, point_hardnesses,
                          shell_charges, shell_potentials, energies, error));
    const double right = sum_values(energies);
    qm_positions[coordinate] -= 2.0 * step;
    CHECK(evaluate_energy(plan, qm_positions, point_positions, point_charges, point_hardnesses,
                          shell_charges, shell_potentials, energies, error));
    const double left = sum_values(energies);
    qm_positions[coordinate] += step;
    CHECK(near(-(right - left) / (2.0 * step), qm_forces[coordinate], 3.0e-10));
  }
  for (std::size_t coordinate = 0; coordinate < point_positions.size(); ++coordinate) {
    point_positions[coordinate] += step;
    CHECK(evaluate_energy(plan, qm_positions, point_positions, point_charges, point_hardnesses,
                          shell_charges, shell_potentials, energies, error));
    const double right = sum_values(energies);
    point_positions[coordinate] -= 2.0 * step;
    CHECK(evaluate_energy(plan, qm_positions, point_positions, point_charges, point_hardnesses,
                          shell_charges, shell_potentials, energies, error));
    const double left = sum_values(energies);
    point_positions[coordinate] += step;
    CHECK(near(-(right - left) / (2.0 * step), point_forces[coordinate], 3.0e-10));
  }

  for (std::size_t batch = 0; batch < 2u; ++batch) {
    for (std::size_t axis = 0; axis < 3u; ++axis) {
      double total_force = 0.0;
      for (std::int64_t atom = atom_offsets[batch]; atom < atom_offsets[batch + 1u]; ++atom) {
        total_force += qm_forces[static_cast<std::size_t>(atom) * 3u + axis];
      }
      for (std::int64_t point = point_offsets[batch]; point < point_offsets[batch + 1u]; ++point) {
        total_force += point_forces[static_cast<std::size_t>(point) * 3u + axis];
      }
      CHECK(near(total_force, 0.0, 3.0e-17));
    }
  }
  return 0;
}

int test_ragged_matches_sequential_and_zero_point_sites() {
  /* The middle batch member has no QM atoms but owns one inert point site. */
  const std::vector<std::int64_t> atom_offsets{0, 1, 1, 3, 4};
  const std::vector<std::int32_t> atomic_numbers{1, 8, 6, 14};
  const std::vector<std::int64_t> point_offsets{0, 0, 1, 3, 6};
  const std::vector<double> qm_positions{
      0.2, -0.1, 0.7, 1.0, 0.4, -0.3, -0.8, 1.5, 0.2, 2.2, -1.1, 0.9,
  };
  const std::vector<double> point_positions{
      4.0, 4.0, 4.0, 2.4, -0.5, 1.0, -1.7, 0.8, 0.4, 0.0, 2.0, -1.0, 3.5, -2.2, 0.6, 1.1, -0.7, 2.9,
  };
  const std::vector<double> point_charges{9.0, 0.3, -0.4, 0.8, -0.2, 0.5};
  const std::vector<double> point_hardnesses{0.7, 0.4, 1.2, 999.0, 0.8, 0.25};

  BasisPlan basis;
  ExternalPointChargePlan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, atomic_numbers, &point_offsets, basis, plan, error));
  std::vector<double> shell_charges(static_cast<std::size_t>(plan.total_shells));
  for (std::size_t shell = 0; shell < shell_charges.size(); ++shell) {
    shell_charges[shell] = 0.04 * static_cast<double>(shell + 1u) - 0.17;
  }
  std::vector<double> potentials(shell_charges.size());
  std::vector<double> energies(4);
  CHECK(evaluate_energy(plan, qm_positions, point_positions, point_charges, point_hardnesses,
                        shell_charges, potentials, energies, error));
  std::vector<double> qm_forces(qm_positions.size());
  std::vector<double> point_forces(point_positions.size());
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), shell_charges.data(), qm_forces.data(), point_forces.data(),
            error) == GPUXTB_STATUS_SUCCESS);

  CHECK(energies[0] == 0.0);
  CHECK(energies[1] == 0.0);
  CHECK(point_forces[0] == 0.0 && point_forces[1] == 0.0 && point_forces[2] == 0.0);
  CHECK(qm_forces[0] == 0.0 && qm_forces[1] == 0.0 && qm_forces[2] == 0.0);

  for (std::size_t batch : {2u, 3u}) {
    const std::int64_t atom_begin = atom_offsets[batch];
    const std::int64_t atom_end = atom_offsets[batch + 1u];
    const std::int64_t point_begin = point_offsets[batch];
    const std::int64_t point_end = point_offsets[batch + 1u];
    const std::vector<std::int64_t> sequential_atom_offsets{0, atom_end - atom_begin};
    const std::vector<std::int64_t> sequential_point_offsets{0, point_end - point_begin};
    const std::vector<std::int32_t> sequential_numbers(atomic_numbers.begin() + atom_begin,
                                                       atomic_numbers.begin() + atom_end);
    const std::vector<double> sequential_qm_positions(qm_positions.begin() + atom_begin * 3,
                                                      qm_positions.begin() + atom_end * 3);
    const std::vector<double> sequential_point_positions(point_positions.begin() + point_begin * 3,
                                                         point_positions.begin() + point_end * 3);
    const std::vector<double> sequential_point_charges(point_charges.begin() + point_begin,
                                                       point_charges.begin() + point_end);
    const std::vector<double> sequential_point_hardnesses(point_hardnesses.begin() + point_begin,
                                                          point_hardnesses.begin() + point_end);

    BasisPlan sequential_basis;
    ExternalPointChargePlan sequential_plan;
    CHECK(make_plan(sequential_atom_offsets, sequential_numbers, &sequential_point_offsets,
                    sequential_basis, sequential_plan, error));
    const std::int64_t shell_begin = plan.batch_shell_offsets[batch];
    const std::int64_t shell_end = plan.batch_shell_offsets[batch + 1u];
    const std::vector<double> sequential_shell_charges(shell_charges.begin() + shell_begin,
                                                       shell_charges.begin() + shell_end);
    std::vector<double> sequential_potentials(sequential_shell_charges.size());
    std::vector<double> sequential_energies(1);
    CHECK(evaluate_energy(sequential_plan, sequential_qm_positions, sequential_point_positions,
                          sequential_point_charges, sequential_point_hardnesses,
                          sequential_shell_charges, sequential_potentials, sequential_energies,
                          error));
    std::vector<double> sequential_qm_forces(sequential_qm_positions.size());
    std::vector<double> sequential_point_forces(sequential_point_positions.size());
    CHECK(gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
              sequential_plan, sequential_qm_positions.data(), sequential_point_positions.data(),
              sequential_point_charges.data(), sequential_point_hardnesses.data(),
              sequential_shell_charges.data(), sequential_qm_forces.data(),
              sequential_point_forces.data(), error) == GPUXTB_STATUS_SUCCESS);

    CHECK(sequential_energies[0] == energies[batch]);
    for (std::size_t shell = 0; shell < sequential_potentials.size(); ++shell) {
      CHECK(sequential_potentials[shell] ==
            potentials[static_cast<std::size_t>(shell_begin) + shell]);
    }
    for (std::size_t coordinate = 0; coordinate < sequential_qm_forces.size(); ++coordinate) {
      CHECK(sequential_qm_forces[coordinate] ==
            qm_forces[static_cast<std::size_t>(atom_begin) * 3u + coordinate]);
    }
    for (std::size_t coordinate = 0; coordinate < sequential_point_forces.size(); ++coordinate) {
      CHECK(sequential_point_forces[coordinate] ==
            point_forces[static_cast<std::size_t>(point_begin) * 3u + coordinate]);
    }
  }
  return 0;
}

int test_coincident_and_large_hardness() {
  const std::vector<std::int64_t> atom_offsets{0, 1};
  const std::vector<std::int32_t> atomic_numbers{1};
  const std::vector<std::int64_t> point_offsets{0, 2};
  BasisPlan basis;
  ExternalPointChargePlan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, atomic_numbers, &point_offsets, basis, plan, error));

  const std::array<double, 3> qm_positions{1.25, -0.75, 0.5};
  const std::array<double, 6> point_positions{
      1.25, -0.75, 0.5, 4.25, -0.75, 4.5,
  };
  const std::array<double, 2> point_charges{0.6, -0.4};
  const std::array<double, 2> point_hardnesses{0.405771, 999.0};
  std::array<double, 1> potential{};
  CHECK(gpuxtb::detail::gfn2::evaluate_external_point_charge_potential_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), potential.data(), error) == GPUXTB_STATUS_SUCCESS);
  const double coincident = point_charges[0] * (plan.shell_hardness[0] + point_hardnesses[0]) / 2.0;
  const double softness = 2.0 / (plan.shell_hardness[0] + point_hardnesses[1]);
  const double large_gamma = point_charges[1] / std::sqrt(25.0 + softness * softness);
  CHECK(near(potential[0], coincident + large_gamma, 2.0e-16));

  constexpr std::array<double, 1> shell_charges{-0.3};
  std::array<double, 3> qm_forces{};
  std::array<double, 6> point_forces{};
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), shell_charges.data(), qm_forces.data(), point_forces.data(),
            error) == GPUXTB_STATUS_SUCCESS);
  CHECK(point_forces[0] == 0.0 && point_forces[1] == 0.0 && point_forces[2] == 0.0);
  for (std::size_t axis = 0; axis < 3u; ++axis) {
    CHECK(near(qm_forces[axis] + point_forces[3u + axis], 0.0, 0.0));
  }
  return 0;
}

int test_validation_and_strong_failure_guarantee() {
  const std::vector<std::int64_t> atom_offsets{0, 1};
  const std::vector<std::int32_t> atomic_numbers{8};
  const std::vector<std::int64_t> point_offsets{0, 1};
  BasisPlan basis;
  ExternalPointChargePlan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, atomic_numbers, &point_offsets, basis, plan, error));

  ExternalPointChargePlan sentinel;
  sentinel.batch_size = 17;
  constexpr std::array<std::int64_t, 2> bad_start{1, 1};
  constexpr std::array<std::int64_t, 2> bad_end{0, 0};
  constexpr std::array<std::int64_t, 2> descending{0, -1};
  constexpr std::array<std::int32_t, 1> mismatched_number{6};
  CHECK(gpuxtb::detail::gfn2::make_external_point_charge_plan(basis, atomic_numbers.data(), 1,
                                                              bad_start.data(), sentinel, error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(sentinel.batch_size == 17);
  CHECK(gpuxtb::detail::gfn2::make_external_point_charge_plan(basis, atomic_numbers.data(), 1,
                                                              bad_end.data(), sentinel, error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(gpuxtb::detail::gfn2::make_external_point_charge_plan(basis, atomic_numbers.data(), -1,
                                                              descending.data(), sentinel, error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(gpuxtb::detail::gfn2::make_external_point_charge_plan(
            basis, mismatched_number.data(), 1, point_offsets.data(), sentinel, error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(gpuxtb::detail::gfn2::make_external_point_charge_plan(
            basis, nullptr, 1, point_offsets.data(), sentinel, error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(gpuxtb::detail::gfn2::make_external_point_charge_plan(basis, atomic_numbers.data(), 1,
                                                              nullptr, sentinel, error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(sentinel.batch_size == 17);

  const std::array<double, 3> qm_positions{0.0, 0.0, 0.0};
  const std::array<double, 3> point_positions{2.0, -0.5, 1.0};
  const std::array<double, 1> point_charges{0.4};
  const std::array<double, 1> point_hardnesses{0.7};
  std::array<double, 2> potentials{8.0, 9.0};
  std::array<double, 2> shell_charges{0.2, -0.3};
  std::array<double, 1> energy{11.0};
  std::array<double, 3> qm_forces{1.0, 2.0, 3.0};
  std::array<double, 3> point_forces{4.0, 5.0, 6.0};

  ExternalPointChargePlan corrupt = plan;
  corrupt.point_charge_offsets[1] = 0;
  CHECK(gpuxtb::detail::gfn2::evaluate_external_point_charge_potential_cpu(
            corrupt, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), potentials.data(), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(potentials[0] == 8.0 && potentials[1] == 9.0);
  corrupt = plan;
  corrupt.shell_hardness[0] = 0.0;
  CHECK(gpuxtb::detail::gfn2::evaluate_external_point_charge_potential_cpu(
            corrupt, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), potentials.data(), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  corrupt = plan;
  corrupt.shell_to_atom[0] = 3;
  CHECK(gpuxtb::detail::gfn2::evaluate_external_point_charge_potential_cpu(
            corrupt, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), potentials.data(), error) == GPUXTB_STATUS_INVALID_ARGUMENT);

  CHECK(gpuxtb::detail::gfn2::evaluate_external_point_charge_potential_cpu(
            plan, nullptr, point_positions.data(), point_charges.data(), point_hardnesses.data(),
            potentials.data(), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(gpuxtb::detail::gfn2::evaluate_external_point_charge_potential_cpu(
            plan, qm_positions.data(), nullptr, point_charges.data(), point_hardnesses.data(),
            potentials.data(), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  std::array<double, 3> bad_qm_positions = qm_positions;
  bad_qm_positions[1] = std::numeric_limits<double>::quiet_NaN();
  CHECK(gpuxtb::detail::gfn2::evaluate_external_point_charge_potential_cpu(
            plan, bad_qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), potentials.data(), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  std::array<double, 3> bad_point_positions = point_positions;
  bad_point_positions[2] = std::numeric_limits<double>::infinity();
  CHECK(gpuxtb::detail::gfn2::evaluate_external_point_charge_potential_cpu(
            plan, qm_positions.data(), bad_point_positions.data(), point_charges.data(),
            point_hardnesses.data(), potentials.data(), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  std::array<double, 1> bad_point_charges{std::numeric_limits<double>::quiet_NaN()};
  CHECK(gpuxtb::detail::gfn2::evaluate_external_point_charge_potential_cpu(
            plan, qm_positions.data(), point_positions.data(), bad_point_charges.data(),
            point_hardnesses.data(), potentials.data(), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  for (double bad_hardness : {0.0, -1.0, std::numeric_limits<double>::infinity()}) {
    const std::array<double, 1> values{bad_hardness};
    CHECK(gpuxtb::detail::gfn2::evaluate_external_point_charge_potential_cpu(
              plan, qm_positions.data(), point_positions.data(), point_charges.data(),
              values.data(), potentials.data(), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  }
  CHECK(gpuxtb::detail::gfn2::evaluate_external_point_charge_potential_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), nullptr, error) == GPUXTB_STATUS_INVALID_ARGUMENT);

  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_energy_cpu(plan, nullptr, potentials.data(),
                                                                   energy.data(), error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_energy_cpu(plan, shell_charges.data(),
                                                                   nullptr, energy.data(), error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_energy_cpu(
            plan, shell_charges.data(), potentials.data(), nullptr, error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  std::array<double, 2> bad_shell_charges = shell_charges;
  bad_shell_charges[1] = std::numeric_limits<double>::quiet_NaN();
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_energy_cpu(
            plan, bad_shell_charges.data(), potentials.data(), energy.data(), error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(energy[0] == 11.0);

  const double maximum = std::numeric_limits<double>::max();
  const std::array<double, 2> overflow_shell_charges{1.0, 1.0};
  const std::array<double, 2> overflow_potentials{0.5 * maximum, 0.0};
  energy[0] = maximum;
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_energy_cpu(
            plan, overflow_shell_charges.data(), overflow_potentials.data(), energy.data(),
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(energy[0] == maximum);
  energy[0] = 11.0;

  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), shell_charges.data(), nullptr, nullptr,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), nullptr, qm_forces.data(), point_forces.data(),
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  constexpr std::array<double, 3> expected_qm_forces{1.0, 2.0, 3.0};
  constexpr std::array<double, 3> expected_point_forces{4.0, 5.0, 6.0};
  CHECK(qm_forces == expected_qm_forces);
  CHECK(point_forces == expected_point_forces);

  const std::array<double, 3> overflow_qm_position{1.0, 0.0, 0.0};
  const std::array<double, 3> overflow_point_position{0.0, 0.0, 0.0};
  const std::array<double, 1> overflow_point_charge{0.75 * maximum};
  const std::array<double, 1> overflow_point_hardness{maximum};
  const std::array<double, 2> overflow_force_shell_charges{1.0, 0.0};
  qm_forces = {maximum, 0.0, 0.0};
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
            plan, overflow_qm_position.data(), overflow_point_position.data(),
            overflow_point_charge.data(), overflow_point_hardness.data(),
            overflow_force_shell_charges.data(), qm_forces.data(), nullptr,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(qm_forces[0] == maximum);
  point_forces = {-maximum, 0.0, 0.0};
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
            plan, overflow_qm_position.data(), overflow_point_position.data(),
            overflow_point_charge.data(), overflow_point_hardness.data(),
            overflow_force_shell_charges.data(), nullptr, point_forces.data(),
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(point_forces[0] == -maximum);

  /* Zero-PC plans synthesize an all-zero ragged partition from NULL. */
  ExternalPointChargePlan zero_plan;
  CHECK(gpuxtb::detail::gfn2::make_external_point_charge_plan(
            basis, atomic_numbers.data(), 0, nullptr, zero_plan, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(zero_plan.point_charge_offsets == std::vector<std::int64_t>({0, 0}));
  std::array<double, 2> zero_potential{7.0, 7.0};
  CHECK(gpuxtb::detail::gfn2::evaluate_external_point_charge_potential_cpu(
            zero_plan, qm_positions.data(), nullptr, nullptr, nullptr, zero_potential.data(),
            error) == GPUXTB_STATUS_SUCCESS);
  CHECK(zero_potential[0] == 0.0 && zero_potential[1] == 0.0);
  return 0;
}

int test_no_steady_state_allocations() {
  const std::vector<std::int64_t> atom_offsets{0, 2};
  const std::vector<std::int32_t> atomic_numbers{8, 1};
  const std::vector<std::int64_t> point_offsets{0, 2};
  BasisPlan basis;
  ExternalPointChargePlan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, atomic_numbers, &point_offsets, basis, plan, error));
  const std::array<double, 6> qm_positions{0.0, 0.0, 0.0, 1.4, 0.0, 1.1};
  const std::array<double, 6> point_positions{3.0, 0.2, -0.4, -2.0, 1.0, 0.5};
  const std::array<double, 2> point_charges{0.5, -0.3};
  const std::array<double, 2> point_hardnesses{0.405771, 999.0};
  std::vector<double> shell_charges(static_cast<std::size_t>(plan.total_shells), 0.125);
  std::vector<double> shell_potentials(shell_charges.size());
  std::array<double, 1> energy{};
  std::array<double, 6> qm_forces{};
  std::array<double, 6> point_forces{};

  /* Warm up any implementation-independent C++ runtime paths first. */
  CHECK(gpuxtb::detail::gfn2::evaluate_external_point_charge_potential_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), shell_potentials.data(), error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_energy_cpu(
            plan, shell_charges.data(), shell_potentials.data(), energy.data(), error) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), shell_charges.data(), qm_forces.data(), point_forces.data(),
            error) == GPUXTB_STATUS_SUCCESS);

  const std::size_t before = allocation_test::count.load(std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const gpuxtb_status_t potential_status =
      gpuxtb::detail::gfn2::evaluate_external_point_charge_potential_cpu(
          plan, qm_positions.data(), point_positions.data(), point_charges.data(),
          point_hardnesses.data(), shell_potentials.data(), error);
  const gpuxtb_status_t energy_status = gpuxtb::detail::gfn2::add_external_point_charge_energy_cpu(
      plan, shell_charges.data(), shell_potentials.data(), energy.data(), error);
  const gpuxtb_status_t force_status = gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
      plan, qm_positions.data(), point_positions.data(), point_charges.data(),
      point_hardnesses.data(), shell_charges.data(), qm_forces.data(), point_forces.data(), error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  const std::size_t after = allocation_test::count.load(std::memory_order_relaxed);

  CHECK(potential_status == GPUXTB_STATUS_SUCCESS);
  CHECK(energy_status == GPUXTB_STATUS_SUCCESS);
  CHECK(force_status == GPUXTB_STATUS_SUCCESS);
  CHECK(after == before);
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_xtb_671_water_golden(); status != 0) {
    return status;
  }
  if (const int status = test_finite_difference_and_force_conservation(); status != 0) {
    return status;
  }
  if (const int status = test_ragged_matches_sequential_and_zero_point_sites(); status != 0) {
    return status;
  }
  if (const int status = test_coincident_and_large_hardness(); status != 0) {
    return status;
  }
  if (const int status = test_validation_and_strong_failure_guarantee(); status != 0) {
    return status;
  }
  return test_no_steady_state_allocations();
}
