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

#include "model/gfn1/external_point_charges.hpp"

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

using xtbloom::detail::gfn1::BasisPlan;
using xtbloom::detail::gfn1::ES2Plan;
using xtbloom::detail::gfn1::ExternalPointChargePlan;

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

bool make_plan(const std::vector<std::int64_t>& atom_offsets,
               const std::vector<std::int32_t>& atomic_numbers,
               const std::vector<std::int64_t>* point_offsets, BasisPlan& basis, ES2Plan& es2,
               ExternalPointChargePlan& external, std::string& error) {
  const auto batches = static_cast<std::int64_t>(atom_offsets.size() - 1u);
  if (xtbloom::detail::gfn1::make_basis_plan(
          batches, static_cast<std::int64_t>(atomic_numbers.size()), atom_offsets.data(),
          atomic_numbers.data(), basis, error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn1::make_es2_plan(basis, atomic_numbers.data(), es2, error) !=
          XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  const std::int64_t points = point_offsets == nullptr ? 0 : point_offsets->back();
  return xtbloom::detail::gfn1::make_external_point_charge_plan(
             basis, es2, points, point_offsets == nullptr ? nullptr : point_offsets->data(),
             external, error) == XTBLOOM_STATUS_SUCCESS;
}

bool evaluate_energy(const ExternalPointChargePlan& plan, const std::vector<double>& qm_positions,
                     const std::vector<double>& point_positions,
                     const std::vector<double>& point_charges,
                     const std::vector<double>& point_hardnesses,
                     const std::vector<double>& shell_charges,
                     std::vector<double>& shell_potentials, std::vector<double>& energies,
                     std::string& error) {
  std::fill(energies.begin(), energies.end(), 0.0);
  const double* positions = point_positions.empty() ? nullptr : point_positions.data();
  const double* charges = point_charges.empty() ? nullptr : point_charges.data();
  const double* hardnesses = point_hardnesses.empty() ? nullptr : point_hardnesses.data();
  return xtbloom::detail::gfn1::evaluate_external_point_charge_potential_cpu(
             plan, qm_positions.data(), positions, charges, hardnesses, shell_potentials.data(),
             error) == XTBLOOM_STATUS_SUCCESS &&
         xtbloom::detail::gfn1::add_external_point_charge_energy_cpu(
             plan, shell_charges.data(), shell_potentials.data(), energies.data(), error) ==
             XTBLOOM_STATUS_SUCCESS;
}

double sum(const std::vector<double>& values) {
  double result = 0.0;
  for (double value : values) {
    result += value;
  }
  return result;
}

int test_xtb_671_water_dimer_pcem_golden() {
  const std::vector<std::int64_t> atom_offsets{0, 6};
  const std::vector<std::int32_t> numbers{8, 1, 1, 8, 1, 1};
  const std::vector<std::int64_t> point_offsets{0, 6};
  BasisPlan basis;
  ES2Plan es2;
  ExternalPointChargePlan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, numbers, &point_offsets, basis, es2, plan, error));
  CHECK(plan.total_shells == 12);
  constexpr std::array<double, 12> expected_hardness{
      0.583349, 0.6052017202192, 0.470099, 0.470099, 0.470099, 0.470099,
      0.583349, 0.6052017202192, 0.470099, 0.470099, 0.470099, 0.470099,
  };
  for (std::size_t shell = 0u; shell < expected_hardness.size(); ++shell) {
    CHECK(near(plan.shell_hardness[shell], expected_hardness[shell], 2.0e-16));
  }

  const std::vector<double> qm_positions{
      -2.75237178376284, 2.43247309226225,  -0.01392519847964, -0.93157260886974, 2.7962140445859,
      -0.01863384029005, -3.43820531288547, 3.3058360842106,   1.42134539425148,  -2.43247309226225,
      -2.75237178376284, 0.01392519847964,  -2.7962140445859,  -0.93157260886974, 0.01863384029005,
      -3.3058360842106,  -3.43820531288547, -1.42134539425148,
  };
  const std::vector<double> point_positions{
      2.75237178376284,  -2.43247309226225, -0.01392519847964, 0.93157260886974, -2.7962140445859,
      -0.01863384029005, 3.43820531288547,  -3.3058360842106,  1.42134539425148, 2.43247309226225,
      2.75237178376284,  0.01392519847964,  2.7962140445859,   0.93157260886974, 0.01863384029005,
      3.3058360842106,   3.43820531288547,  -1.42134539425148,
  };
  const std::vector<double> point_charges{-0.69645733, 0.36031084, 0.33614649,
                                          -0.69645733, 0.36031084, 0.33614649};
  const std::vector<double> point_hardnesses{0.583349, 0.470099, 0.470099,
                                             0.583349, 0.470099, 0.470099};
  /* Converged qsh decoded from the pinned xTB 6.7.1 xtbrestart fixture. */
  const std::vector<double> shell_charges{
      0.2918603161097545, -0.9670712742098337,   0.4067325798143576, -0.031199713655994048,
      0.3735683251260489, -0.030135176913355523, 0.2915420241081715, -1.0145239222417881,
      0.392750474302224,  -0.04162990793699258,  0.3602210956660194, -0.03211482016861597,
  };
  std::vector<double> potentials(shell_charges.size());
  std::vector<double> energies(1);
  CHECK(evaluate_energy(plan, qm_positions, point_positions, point_charges, point_hardnesses,
                        shell_charges, potentials, energies, error));
  constexpr std::array<double, 12> expected_potential{
      -0.016891118400084802, -0.016934782641131192, -0.039774622488834135, -0.039774622488834135,
      -0.014518970192384413, -0.014518970192384413, 0.018484537242363576,  0.01872048321216087,
      0.004731997325812233,  0.004731997325812233,  0.01106932943484995,   0.01106932943484995,
  };
  /* xTB and C++ evaluate the same binary64 expression in a different order. */
  for (std::size_t shell = 0u; shell < potentials.size(); ++shell) {
    CHECK(near(potentials[shell], expected_potential[shell], 2.0e-16));
  }
  CHECK(near(energies[0], -0.016785619572167437, 2.0e-16));

  std::vector<double> qm_forces(qm_positions.size());
  std::vector<double> point_forces(point_positions.size());
  CHECK(xtbloom::detail::gfn1::add_external_point_charge_forces_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), shell_charges.data(), qm_forces.data(), point_forces.data(),
            error) == XTBLOOM_STATUS_SUCCESS);
  constexpr std::array<double, 18> expected_qm_forces{
      -0.005002548339982094,  -0.0026530209270549457, -0.0006893953440955792,
      0.0069997727216043895,  0.0016373545163549555,  0.001165753061055625,
      0.0018418106579806829,  0.0005562942141213919,  -0.00024254998393719536,
      0.00577830347232085,    -0.002725101933738746,  0.0007624094180657933,
      -0.000551592546709966,  0.0024560833796098205,  -0.00010776308549607173,
      -0.0011614865979287822, 0.00040546021025969617, -0.000775789040285691,
  };
  constexpr std::array<double, 18> expected_point_forces{
      0.005754382645828299,   0.003108280878440533,   -0.0006720991538233419,
      -0.007598129142914906,  -0.0017820248619566303, 0.0010870361586332777,
      -0.0020130657414842634, -0.0006369406139295026, -0.0002952943176660363,
      -0.006088864472756121,  0.0030013089601267783,  0.0007615822149723352,
      0.0006553261174369982,  -0.002864156853335547,  -0.00012635889921668782,
      0.0013860912266049118,  -0.0005035369688978033, -0.0008675310282064281,
  };
  for (std::size_t coordinate = 0u; coordinate < qm_forces.size(); ++coordinate) {
    CHECK(near(qm_forces[coordinate], expected_qm_forces[coordinate], 2.0e-16));
    CHECK(near(point_forces[coordinate], expected_point_forces[coordinate], 2.0e-16));
  }
  for (std::size_t axis = 0u; axis < 3u; ++axis) {
    double total = 0.0;
    for (std::size_t atom = 0u; atom < 6u; ++atom) {
      total += qm_forces[atom * 3u + axis];
      total += point_forces[atom * 3u + axis];
    }
    CHECK(std::abs(total) < 1.0e-17);
  }
  return 0;
}

int test_multistep_finite_difference_and_ragged_sequential() {
  const std::vector<std::int64_t> atom_offsets{0, 2, 4};
  const std::vector<std::int32_t> numbers{6, 8, 14, 1};
  const std::vector<std::int64_t> point_offsets{0, 2, 5};
  BasisPlan basis;
  ES2Plan es2;
  ExternalPointChargePlan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, numbers, &point_offsets, basis, es2, plan, error));
  std::vector<double> qm_positions{-0.4, 0.2, 1.1, 1.3, -0.7, 0.5, 2.0, 0.3, -1.2, 2.8, -0.9, 0.4};
  std::vector<double> point_positions{3.1, 0.8, -0.2, -2.0, 1.4, 0.6, 0.2, -1.3,
                                      2.2, 4.0, 0.1,  -0.7, 2.5, 2.1, 1.4};
  const std::vector<double> point_charges{0.7, -0.25, 0.1, -0.8, 0.35};
  const std::vector<double> point_hardnesses{0.42, 0.9, 1.7, 0.31, 999.0};
  std::vector<double> shell_charges(static_cast<std::size_t>(plan.total_shells));
  for (std::size_t shell = 0u; shell < shell_charges.size(); ++shell) {
    shell_charges[shell] = 0.3 * std::sin(0.71 * static_cast<double>(shell + 1u)) - 0.11;
  }
  std::vector<double> potentials(shell_charges.size());
  std::vector<double> energies(2);
  CHECK(evaluate_energy(plan, qm_positions, point_positions, point_charges, point_hardnesses,
                        shell_charges, potentials, energies, error));
  const std::vector<double> ragged_potentials = potentials;
  const std::vector<double> ragged_energies = energies;
  std::vector<double> qm_forces(qm_positions.size());
  std::vector<double> point_forces(point_positions.size());
  CHECK(xtbloom::detail::gfn1::add_external_point_charge_forces_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), shell_charges.data(), qm_forces.data(), point_forces.data(),
            error) == XTBLOOM_STATUS_SUCCESS);

  for (double step : {2.0e-4, 7.0e-5, 2.0e-5}) {
    for (std::size_t coordinate :
         {std::size_t{0}, std::size_t{4}, std::size_t{7}, std::size_t{11}}) {
      qm_positions[coordinate] += step;
      CHECK(evaluate_energy(plan, qm_positions, point_positions, point_charges, point_hardnesses,
                            shell_charges, potentials, energies, error));
      const double right = sum(energies);
      qm_positions[coordinate] -= 2.0 * step;
      CHECK(evaluate_energy(plan, qm_positions, point_positions, point_charges, point_hardnesses,
                            shell_charges, potentials, energies, error));
      const double left = sum(energies);
      qm_positions[coordinate] += step;
      CHECK(near(-(right - left) / (2.0 * step), qm_forces[coordinate], 2.0e-9));
    }
    for (std::size_t coordinate :
         {std::size_t{1}, std::size_t{6}, std::size_t{10}, std::size_t{14}}) {
      point_positions[coordinate] += step;
      CHECK(evaluate_energy(plan, qm_positions, point_positions, point_charges, point_hardnesses,
                            shell_charges, potentials, energies, error));
      const double right = sum(energies);
      point_positions[coordinate] -= 2.0 * step;
      CHECK(evaluate_energy(plan, qm_positions, point_positions, point_charges, point_hardnesses,
                            shell_charges, potentials, energies, error));
      const double left = sum(energies);
      point_positions[coordinate] += step;
      CHECK(near(-(right - left) / (2.0 * step), point_forces[coordinate], 2.0e-9));
    }
  }

  for (std::size_t batch = 0u; batch < 2u; ++batch) {
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
      double total = 0.0;
      for (std::int64_t atom = atom_offsets[batch]; atom < atom_offsets[batch + 1u]; ++atom) {
        total += qm_forces[static_cast<std::size_t>(atom) * 3u + axis];
      }
      for (std::int64_t point = point_offsets[batch]; point < point_offsets[batch + 1u]; ++point) {
        total += point_forces[static_cast<std::size_t>(point) * 3u + axis];
      }
      CHECK(std::abs(total) < 5.0e-17);
    }

    const std::int64_t atom_begin = atom_offsets[batch];
    const std::int64_t atom_end = atom_offsets[batch + 1u];
    const std::int64_t point_begin = point_offsets[batch];
    const std::int64_t point_end = point_offsets[batch + 1u];
    const std::vector<std::int64_t> sequential_atoms{0, atom_end - atom_begin};
    const std::vector<std::int64_t> sequential_points{0, point_end - point_begin};
    const std::vector<std::int32_t> sequential_numbers(numbers.begin() + atom_begin,
                                                       numbers.begin() + atom_end);
    BasisPlan sequential_basis;
    ES2Plan sequential_es2;
    ExternalPointChargePlan sequential_plan;
    CHECK(make_plan(sequential_atoms, sequential_numbers, &sequential_points, sequential_basis,
                    sequential_es2, sequential_plan, error));
    const std::int64_t shell_begin = plan.batch_shell_offsets[batch];
    const std::int64_t shell_end = plan.batch_shell_offsets[batch + 1u];
    const std::vector<double> sq(qm_positions.begin() + atom_begin * 3,
                                 qm_positions.begin() + atom_end * 3);
    const std::vector<double> sp(point_positions.begin() + point_begin * 3,
                                 point_positions.begin() + point_end * 3);
    const std::vector<double> sc(point_charges.begin() + point_begin,
                                 point_charges.begin() + point_end);
    const std::vector<double> sh(point_hardnesses.begin() + point_begin,
                                 point_hardnesses.begin() + point_end);
    const std::vector<double> ss(shell_charges.begin() + shell_begin,
                                 shell_charges.begin() + shell_end);
    std::vector<double> sequential_potential(ss.size());
    std::vector<double> sequential_energy(1);
    CHECK(evaluate_energy(sequential_plan, sq, sp, sc, sh, ss, sequential_potential,
                          sequential_energy, error));
    CHECK(sequential_energy[0] == ragged_energies[batch]);
    for (std::size_t shell = 0u; shell < sequential_potential.size(); ++shell) {
      CHECK(sequential_potential[shell] ==
            ragged_potentials[static_cast<std::size_t>(shell_begin) + shell]);
    }
  }
  return 0;
}

int test_zero_close_and_coincident_sites() {
  const std::vector<std::int64_t> atom_offsets{0, 1};
  const std::vector<std::int32_t> numbers{1};
  BasisPlan basis;
  ES2Plan es2;
  ExternalPointChargePlan zero;
  std::string error;
  CHECK(make_plan(atom_offsets, numbers, nullptr, basis, es2, zero, error));
  CHECK(zero.point_charge_offsets == std::vector<std::int64_t>({0, 0}));
  const std::array<double, 3> qm_positions{1.25, -0.75, 0.5};
  std::array<double, 2> zero_potential{7.0, 8.0};
  CHECK(xtbloom::detail::gfn1::evaluate_external_point_charge_potential_cpu(
            zero, qm_positions.data(), nullptr, nullptr, nullptr, zero_potential.data(), error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(zero_potential[0] == 0.0 && zero_potential[1] == 0.0);

  const std::vector<std::int64_t> point_offsets{0, 2};
  ExternalPointChargePlan plan;
  CHECK(xtbloom::detail::gfn1::make_external_point_charge_plan(
            basis, es2, 2, point_offsets.data(), plan, error) == XTBLOOM_STATUS_SUCCESS);
  const std::array<double, 6> point_positions{1.25, -0.75, 0.5, 1.25 + 1.0e-14, -0.75, 0.5};
  const std::array<double, 2> point_charges{0.6, -0.4};
  const std::array<double, 2> point_hardnesses{0.470099, 999.0};
  std::array<double, 2> potential{};
  CHECK(xtbloom::detail::gfn1::evaluate_external_point_charge_potential_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), potential.data(), error) == XTBLOOM_STATUS_SUCCESS);
  const double first_inverse = 0.5 / plan.shell_hardness[0] + 0.5 / point_hardnesses[0];
  const double second_inverse = 0.5 / plan.shell_hardness[0] + 0.5 / point_hardnesses[1];
  const double expected =
      point_charges[0] / first_inverse + point_charges[1] / std::hypot(1.0e-14, second_inverse);
  CHECK(near(potential[0], expected, 2.0e-16));
  CHECK(near(potential[1], expected, 2.0e-16));
  constexpr std::array<double, 2> shell_charges{-0.3, 0.2};
  std::array<double, 3> qm_forces{};
  std::array<double, 6> point_forces{};
  CHECK(xtbloom::detail::gfn1::add_external_point_charge_forces_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), shell_charges.data(), qm_forces.data(), point_forces.data(),
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(point_forces[0] == 0.0 && point_forces[1] == 0.0 && point_forces[2] == 0.0);
  for (std::size_t axis = 0u; axis < 3u; ++axis) {
    CHECK(near(qm_forces[axis] + point_forces[3u + axis], 0.0, 0.0));
  }
  return 0;
}

int test_validation_transactionality_and_no_allocations() {
  const std::vector<std::int64_t> atom_offsets{0, 1};
  const std::vector<std::int32_t> numbers{8};
  const std::vector<std::int64_t> point_offsets{0, 1};
  BasisPlan basis;
  ES2Plan es2;
  ExternalPointChargePlan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, numbers, &point_offsets, basis, es2, plan, error));
  ExternalPointChargePlan sentinel;
  sentinel.batch_size = 17;
  constexpr std::array<std::int64_t, 2> bad_start{1, 1};
  constexpr std::array<std::int64_t, 2> bad_end{0, 0};
  CHECK(xtbloom::detail::gfn1::make_external_point_charge_plan(
            basis, es2, 1, bad_start.data(), sentinel, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn1::make_external_point_charge_plan(
            basis, es2, 1, bad_end.data(), sentinel, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  ES2Plan unsealed;
  CHECK(xtbloom::detail::gfn1::make_external_point_charge_plan(
            basis, unsealed, 1, point_offsets.data(), sentinel, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  xtbloom::detail::gfn2::ES2Plan arithmetic;
  CHECK(xtbloom::detail::gfn2::make_es2_plan_from_shell_hardness(
            basis, xtbloom::detail::gfn2::ES2HardnessAverage::kArithmetic,
            es2.shell_hardness().data(), es2.total_shells(), arithmetic,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn1::make_external_point_charge_plan(
            basis, arithmetic, 1, point_offsets.data(), sentinel, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(sentinel.batch_size == 17);

  const std::array<double, 3> qm_positions{0.0, 0.0, 0.0};
  const std::array<double, 3> point_positions{2.0, -0.5, 1.0};
  const std::array<double, 1> point_charges{0.4};
  const std::array<double, 1> point_hardnesses{0.7};
  std::array<double, 2> potentials{8.0, 9.0};
  const std::array<double, 2> shell_charges{0.2, -0.3};
  std::array<double, 1> energy{11.0};
  std::array<double, 3> qm_forces{1.0, 2.0, 3.0};
  std::array<double, 3> point_forces{4.0, 5.0, 6.0};

  ExternalPointChargePlan corrupt = plan;
  corrupt.shell_hardness[0] = 0.0;
  CHECK(xtbloom::detail::gfn1::evaluate_external_point_charge_potential_cpu(
            corrupt, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), potentials.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  constexpr std::array<double, 2> expected_potentials{8.0, 9.0};
  CHECK(potentials == expected_potentials);
  for (double bad : {0.0, -1.0, std::numeric_limits<double>::infinity()}) {
    const std::array<double, 1> hardness{bad};
    CHECK(xtbloom::detail::gfn1::evaluate_external_point_charge_potential_cpu(
              plan, qm_positions.data(), point_positions.data(), point_charges.data(),
              hardness.data(), potentials.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  }
  const double maximum = std::numeric_limits<double>::max();
  const std::array<double, 6> overflow_positions{};
  const std::array<double, 2> overflow_charges{0.75 * maximum, 0.75 * maximum};
  const std::array<double, 2> overflow_hardness{maximum, maximum};
  ExternalPointChargePlan overflow_plan;
  constexpr std::array<std::int64_t, 2> two_points{0, 2};
  CHECK(xtbloom::detail::gfn1::make_external_point_charge_plan(
            basis, es2, 2, two_points.data(), overflow_plan, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn1::evaluate_external_point_charge_potential_cpu(
            overflow_plan, qm_positions.data(), overflow_positions.data(), overflow_charges.data(),
            overflow_hardness.data(), potentials.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(potentials == expected_potentials);

  energy[0] = maximum;
  constexpr std::array<double, 2> positive_shell_charges{1.0, 1.0};
  const std::array<double, 2> positive_potentials{maximum, 0.0};
  CHECK(xtbloom::detail::gfn1::add_external_point_charge_energy_cpu(
            plan, positive_shell_charges.data(), positive_potentials.data(), energy.data(),
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energy[0] == maximum);
  std::array<double, 2> aliased_shell_charges = shell_charges;
  CHECK(xtbloom::detail::gfn1::add_external_point_charge_energy_cpu(
            plan, aliased_shell_charges.data(), potentials.data(), aliased_shell_charges.data(),
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(aliased_shell_charges == shell_charges);

  constexpr std::array<double, 1> huge_point_charge{maximum};
  constexpr std::array<double, 2> huge_shell_charges{maximum, 1.0};
  CHECK(xtbloom::detail::gfn1::add_external_point_charge_forces_cpu(
            plan, qm_positions.data(), point_positions.data(), huge_point_charge.data(),
            point_hardnesses.data(), huge_shell_charges.data(), qm_forces.data(),
            point_forces.data(), error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  constexpr std::array<double, 3> expected_qm_forces{1.0, 2.0, 3.0};
  constexpr std::array<double, 3> expected_point_forces{4.0, 5.0, 6.0};
  CHECK(qm_forces == expected_qm_forces);
  CHECK(point_forces == expected_point_forces);
  CHECK(xtbloom::detail::gfn1::evaluate_external_point_charge_potential_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), const_cast<double*>(qm_positions.data()),
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  constexpr std::array<double, 3> expected_qm_positions{0.0, 0.0, 0.0};
  CHECK(qm_positions == expected_qm_positions);

  std::array<double, 2> warm_potentials{};
  energy[0] = 0.0;
  qm_forces.fill(0.0);
  point_forces.fill(0.0);
  CHECK(xtbloom::detail::gfn1::evaluate_external_point_charge_potential_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), warm_potentials.data(), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn1::add_external_point_charge_energy_cpu(
            plan, shell_charges.data(), warm_potentials.data(), energy.data(), error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn1::add_external_point_charge_forces_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), shell_charges.data(), qm_forces.data(), point_forces.data(),
            error) == XTBLOOM_STATUS_SUCCESS);
  const std::size_t before = allocation_test::count.load(std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const auto potential_status = xtbloom::detail::gfn1::evaluate_external_point_charge_potential_cpu(
      plan, qm_positions.data(), point_positions.data(), point_charges.data(),
      point_hardnesses.data(), warm_potentials.data(), error);
  const auto energy_status = xtbloom::detail::gfn1::add_external_point_charge_energy_cpu(
      plan, shell_charges.data(), warm_potentials.data(), energy.data(), error);
  const auto force_status = xtbloom::detail::gfn1::add_external_point_charge_forces_cpu(
      plan, qm_positions.data(), point_positions.data(), point_charges.data(),
      point_hardnesses.data(), shell_charges.data(), qm_forces.data(), point_forces.data(), error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  CHECK(potential_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(energy_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(force_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == before);
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_xtb_671_water_dimer_pcem_golden(); status != 0) {
    return status;
  }
  if (const int status = test_multistep_finite_difference_and_ragged_sequential(); status != 0) {
    return status;
  }
  if (const int status = test_zero_close_and_coincident_sites(); status != 0) {
    return status;
  }
  return test_validation_transactionality_and_no_allocations();
}
