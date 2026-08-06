#include "model/gfn2/es3.hpp"

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

#include "data/parameters/gfn2.hpp"
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
using gpuxtb::detail::gfn2::ES3Plan;
using gpuxtb::detail::gfn2::ES3View;

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

bool make_plan(const std::vector<std::int64_t>& atom_offsets,
               const std::vector<std::int32_t>& atomic_numbers, BasisPlan& basis, ES3Plan& es3,
               std::string& error) {
  const std::int64_t batch_size = static_cast<std::int64_t>(atom_offsets.size() - 1u);
  const std::int64_t total_atoms = static_cast<std::int64_t>(atomic_numbers.size());
  return gpuxtb::detail::gfn2::make_basis_plan(batch_size, total_atoms, atom_offsets.data(),
                                               atomic_numbers.data(), basis,
                                               error) == GPUXTB_STATUS_SUCCESS &&
         gpuxtb::detail::gfn2::make_es3_plan(basis, atomic_numbers.data(), es3, error) ==
             GPUXTB_STATUS_SUCCESS;
}

int test_all_elements_and_shell_angular_momenta() {
  std::vector<std::int32_t> atomic_numbers(gpuxtb::parameters::gfn2::kElementCount);
  for (std::size_t element = 0; element < atomic_numbers.size(); ++element) {
    atomic_numbers[element] = static_cast<std::int32_t>(element + 1u);
  }
  const std::vector<std::int64_t> atom_offsets{0, static_cast<std::int64_t>(atomic_numbers.size())};
  BasisPlan basis;
  ES3Plan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, atomic_numbers, basis, plan, error));
  CHECK(error.empty());
  CHECK(plan.batch_size == 1);
  CHECK(plan.total_shells == static_cast<std::int64_t>(gpuxtb::parameters::gfn2::kShellCount));
  CHECK(plan.batch_shell_offsets == basis.batch_shell_offsets);

  const ES3View view = gpuxtb::detail::gfn2::make_es3_view(plan);
  CHECK(view.batch_size == plan.batch_size);
  CHECK(view.total_shells == plan.total_shells);
  CHECK(view.batch_shell_offset_count ==
        static_cast<std::int64_t>(plan.batch_shell_offsets.size()));
  CHECK(view.shell_gamma3_count == static_cast<std::int64_t>(plan.shell_gamma3.size()));
  CHECK(view.batch_shell_offsets == plan.batch_shell_offsets.data());
  CHECK(view.shell_gamma3 == plan.shell_gamma3.data());

  std::array<bool, 3> seen_angular_momentum{};
  bool seen_positive_gamma3 = false;
  bool seen_negative_gamma3 = false;
  for (std::size_t atom = 0; atom < atomic_numbers.size(); ++atom) {
    const auto& element = gpuxtb::parameters::gfn2::kElements[atom];
    for (std::int64_t shell = basis.atom_shell_offsets[atom];
         shell < basis.atom_shell_offsets[atom + 1u]; ++shell) {
      const std::size_t shell_index = static_cast<std::size_t>(shell);
      const std::uint8_t angular_momentum = basis.angular_momenta[shell_index];
      CHECK(angular_momentum <= 2u);
      seen_angular_momentum[angular_momentum] = true;
      const double expected =
          element.gam3 * gpuxtb::parameters::gfn2::kGlobal.thirdorder_shell_scale[angular_momentum];
      CHECK(plan.shell_gamma3[shell_index] == expected);
      seen_positive_gamma3 = seen_positive_gamma3 || expected > 0.0;
      seen_negative_gamma3 = seen_negative_gamma3 || expected < 0.0;
    }
  }
  CHECK((seen_angular_momentum == std::array<bool, 3>{true, true, true}));
  CHECK(seen_positive_gamma3 && seen_negative_gamma3);

  const std::size_t shell_count = static_cast<std::size_t>(plan.total_shells);
  std::vector<double> charges(shell_count);
  for (std::size_t shell = 0; shell < shell_count; ++shell) {
    const double magnitude = 0.03 + 0.007 * static_cast<double>(shell % 31u);
    charges[shell] = shell % 2u == 0u ? magnitude : -magnitude;
  }
  std::vector<double> potentials(shell_count, -91.0);
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(view, charges.data(), potentials.data(),
                                                         error) == GPUXTB_STATUS_SUCCESS);
  double expected_energy = 0.0;
  for (std::size_t shell = 0; shell < shell_count; ++shell) {
    const double expected_potential = charges[shell] * charges[shell] * plan.shell_gamma3[shell];
    CHECK(potentials[shell] == expected_potential);
    expected_energy +=
        charges[shell] * charges[shell] * charges[shell] * plan.shell_gamma3[shell] / 3.0;
  }
  std::array<double, 1> energies{0.125};
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_cpu(view, charges.data(), energies.data(), error) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(near(energies[0], 0.125 + expected_energy, 2.0e-16));

  /* Every generated shell is also exercised at the opposite charge sign. */
  const std::vector<double> original_sign_potentials = potentials;
  for (double& charge : charges) {
    charge = -charge;
  }
  energies[0] = 0.0;
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(view, charges.data(), potentials.data(),
                                                         error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_cpu(view, charges.data(), energies.data(), error) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(potentials == original_sign_potentials);
  CHECK(near(energies[0], -expected_energy, 2.0e-16));
  return 0;
}

double total_energy(ES3View view, const std::vector<double>& charges, std::string& error) {
  std::vector<double> energies(static_cast<std::size_t>(view.batch_size));
  if (gpuxtb::detail::gfn2::add_es3_energy_cpu(view, charges.data(), energies.data(), error) !=
      GPUXTB_STATUS_SUCCESS) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double total = 0.0;
  for (double energy : energies) {
    total += energy;
  }
  return total;
}

int test_ragged_sequential_and_energy_potential_derivative() {
  /* Includes an empty molecule and s/p/d shells in nonempty members. */
  const std::vector<std::int64_t> atom_offsets{0, 2, 2, 5, 7};
  const std::vector<std::int32_t> atomic_numbers{1, 8, 6, 22, 7, 14, 79};
  BasisPlan basis;
  ES3Plan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, atomic_numbers, basis, plan, error));
  const ES3View view = gpuxtb::detail::gfn2::make_es3_view(plan);

  std::vector<double> charges(static_cast<std::size_t>(plan.total_shells));
  for (std::size_t shell = 0; shell < charges.size(); ++shell) {
    charges[shell] = -0.47 + 0.083 * static_cast<double>(shell % 12u);
  }
  std::vector<double> potentials(charges.size());
  std::vector<double> energies(4);
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(view, charges.data(), potentials.data(),
                                                         error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_cpu(view, charges.data(), energies.data(), error) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(energies[1] == 0.0);

  /* Serial one-system workers over the packed view reproduce the batch API. */
  for (std::int64_t system = 0; system < view.batch_size; ++system) {
    double system_energy = 0.0;
    CHECK(gpuxtb::detail::gfn2::add_es3_energy_system_cpu(
              view, system, charges.data(), system_energy, error) == GPUXTB_STATUS_SUCCESS);
    CHECK(system_energy == energies[static_cast<std::size_t>(system)]);
  }

  constexpr double step = 1.0e-6;
  for (std::size_t shell = 0; shell < charges.size(); ++shell) {
    charges[shell] += step;
    const double right = total_energy(view, charges, error);
    charges[shell] -= 2.0 * step;
    const double left = total_energy(view, charges, error);
    charges[shell] += step;
    CHECK(std::isfinite(right) && std::isfinite(left));
    CHECK(near((right - left) / (2.0 * step), potentials[shell], 3.0e-11));
  }

  /* Re-evaluate after perturbations; floating-point +/- steps need not restore bitwise q. */
  std::fill(energies.begin(), energies.end(), 0.0);
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(view, charges.data(), potentials.data(),
                                                         error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_cpu(view, charges.data(), energies.data(), error) ==
        GPUXTB_STATUS_SUCCESS);

  for (std::size_t batch : {0u, 2u, 3u}) {
    const std::int64_t atom_begin = atom_offsets[batch];
    const std::int64_t atom_end = atom_offsets[batch + 1u];
    const std::vector<std::int32_t> sequential_numbers(atomic_numbers.begin() + atom_begin,
                                                       atomic_numbers.begin() + atom_end);
    const std::vector<std::int64_t> sequential_offsets{0, atom_end - atom_begin};
    BasisPlan sequential_basis;
    ES3Plan sequential_plan;
    CHECK(make_plan(sequential_offsets, sequential_numbers, sequential_basis, sequential_plan,
                    error));
    const std::int64_t shell_begin = plan.batch_shell_offsets[batch];
    const std::int64_t shell_end = plan.batch_shell_offsets[batch + 1u];
    const std::vector<double> sequential_charges(charges.begin() + shell_begin,
                                                 charges.begin() + shell_end);
    std::vector<double> sequential_potentials(sequential_charges.size());
    std::array<double, 1> sequential_energy{};
    const ES3View sequential_view = gpuxtb::detail::gfn2::make_es3_view(sequential_plan);
    CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(
              sequential_view, sequential_charges.data(), sequential_potentials.data(), error) ==
          GPUXTB_STATUS_SUCCESS);
    CHECK(gpuxtb::detail::gfn2::add_es3_energy_cpu(sequential_view, sequential_charges.data(),
                                                   sequential_energy.data(),
                                                   error) == GPUXTB_STATUS_SUCCESS);
    CHECK(sequential_energy[0] == energies[batch]);
    for (std::size_t shell = 0; shell < sequential_potentials.size(); ++shell) {
      CHECK(sequential_potentials[shell] ==
            potentials[static_cast<std::size_t>(shell_begin) + shell]);
    }
  }
  return 0;
}

int test_system_energy_failure_isolation_and_binding() {
  const std::vector<std::int64_t> offsets{0, 1, 1, 3, 4};
  const std::vector<std::int32_t> atomic_numbers{1, 8, 1, 6};
  BasisPlan basis;
  ES3Plan plan;
  std::string error;
  CHECK(make_plan(offsets, atomic_numbers, basis, plan, error));
  const auto alias_at = [](const void* pointer) {
    return reinterpret_cast<double*>(reinterpret_cast<std::uintptr_t>(pointer));
  };
  ES3View view = gpuxtb::detail::gfn2::make_es3_view(plan);
  std::vector<double> charges(static_cast<std::size_t>(plan.total_shells));
  for (std::size_t shell = 0; shell < charges.size(); ++shell) {
    charges[shell] = 0.04 * static_cast<double>(shell + 1u) - 0.17;
  }

  constexpr std::int64_t target = 0;
  const std::int64_t target_shell = plan.batch_shell_offsets[0];
  const std::int64_t peer_shell = plan.batch_shell_offsets[2];
  double expected = 0.625;
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_system_cpu(view, target, charges.data(), expected,
                                                        error) == GPUXTB_STATUS_SUCCESS);

  const double saved_peer_charge = charges[static_cast<std::size_t>(peer_shell)];
  charges[static_cast<std::size_t>(peer_shell)] = std::numeric_limits<double>::quiet_NaN();
  double isolated = 0.625;
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_system_cpu(view, target, charges.data(), isolated,
                                                        error) == GPUXTB_STATUS_SUCCESS);
  CHECK(isolated == expected);
  charges[static_cast<std::size_t>(peer_shell)] = saved_peer_charge;

  std::vector<double> gamma3 = plan.shell_gamma3;
  gamma3[static_cast<std::size_t>(peer_shell)] = std::numeric_limits<double>::quiet_NaN();
  ES3View poisoned_peer_view = view;
  poisoned_peer_view.shell_gamma3 = gamma3.data();
  isolated = 0.625;
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_system_cpu(poisoned_peer_view, target, charges.data(),
                                                        isolated, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(isolated == expected);

  const double saved_target_charge = charges[static_cast<std::size_t>(target_shell)];
  charges[static_cast<std::size_t>(target_shell)] = std::numeric_limits<double>::infinity();
  double unchanged = -3.5;
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_system_cpu(view, target, charges.data(), unchanged,
                                                        error) == GPUXTB_STATUS_INTERNAL_ERROR);
  CHECK(unchanged == -3.5);
  charges[static_cast<std::size_t>(target_shell)] = saved_target_charge;

  gamma3 = plan.shell_gamma3;
  gamma3[static_cast<std::size_t>(target_shell)] = std::numeric_limits<double>::quiet_NaN();
  ES3View poisoned_target_view = view;
  poisoned_target_view.shell_gamma3 = gamma3.data();
  unchanged = -4.25;
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_system_cpu(poisoned_target_view, target,
                                                        charges.data(), unchanged,
                                                        error) == GPUXTB_STATUS_INTERNAL_ERROR);
  CHECK(unchanged == -4.25);

  unchanged = std::numeric_limits<double>::quiet_NaN();
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_system_cpu(view, target, charges.data(), unchanged,
                                                        error) == GPUXTB_STATUS_INTERNAL_ERROR);
  CHECK(std::isnan(unchanged));

  unchanged = 2.75;
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_system_cpu(view, -1, charges.data(), unchanged,
                                                        error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(unchanged == 2.75);
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_system_cpu(view, view.batch_size, charges.data(),
                                                        unchanged,
                                                        error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(unchanged == 2.75);

  ES3View bad_view = view;
  --bad_view.shell_gamma3_count;
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_system_cpu(bad_view, target, charges.data(), unchanged,
                                                        error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(unchanged == 2.75);
  std::vector<std::int64_t> bad_offsets = plan.batch_shell_offsets;
  bad_offsets[1] = plan.total_shells + 1;
  bad_view = view;
  bad_view.batch_shell_offsets = bad_offsets.data();
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_system_cpu(bad_view, target, charges.data(), unchanged,
                                                        error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(unchanged == 2.75);

  const std::vector<double> saved_charges = charges;
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_system_cpu(view, target, charges.data(), charges[0],
                                                        error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(charges == saved_charges);

  const std::vector<double> saved_gamma3 = plan.shell_gamma3;
  double& gamma_alias = plan.shell_gamma3[0];
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_system_cpu(view, target, charges.data(), gamma_alias,
                                                        error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(plan.shell_gamma3 == saved_gamma3);

  const std::vector<std::int64_t> saved_offsets = plan.batch_shell_offsets;
  double& offset_alias = *alias_at(plan.batch_shell_offsets.data());
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_system_cpu(view, target, charges.data(), offset_alias,
                                                        error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(plan.batch_shell_offsets == saved_offsets);
  return 0;
}

int test_system_potential_matches_batch_and_failure_isolation() {
  const std::vector<std::int64_t> offsets{0, 1, 1, 3, 4};
  const std::vector<std::int32_t> atomic_numbers{1, 8, 1, 6};
  BasisPlan basis;
  ES3Plan plan;
  std::string error;
  CHECK(make_plan(offsets, atomic_numbers, basis, plan, error));
  const ES3View view = gpuxtb::detail::gfn2::make_es3_view(plan);
  std::vector<double> charges(static_cast<std::size_t>(plan.total_shells));
  for (std::size_t shell = 0; shell < charges.size(); ++shell) {
    charges[shell] = 0.04 * static_cast<double>(shell + 1u) - 0.17;
  }

  std::vector<double> batch_potentials(charges.size());
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(
            view, charges.data(), batch_potentials.data(), error) == GPUXTB_STATUS_SUCCESS);

  std::vector<double> system_potentials(charges.size(), 0.0);
  for (std::int64_t system = 0; system < plan.batch_size; ++system) {
    const std::int64_t shell_begin = plan.batch_shell_offsets[system];
    const std::int64_t shell_end = plan.batch_shell_offsets[system + 1];
    CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_system_cpu(view, system, charges.data(),
                                                                  system_potentials.data(),
                                                                  error) == GPUXTB_STATUS_SUCCESS);
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      CHECK(system_potentials[static_cast<std::size_t>(shell)] ==
            batch_potentials[static_cast<std::size_t>(shell)]);
    }
  }

  constexpr std::int64_t target = 0;
  const std::int64_t target_shell = plan.batch_shell_offsets[0];
  const std::int64_t peer_shell = plan.batch_shell_offsets[2];
  const std::vector<double> expected(batch_potentials);

  /* A poisoned peer shell or gamma3 must not affect the target system. */
  const double saved_peer_charge = charges[static_cast<std::size_t>(peer_shell)];
  charges[static_cast<std::size_t>(peer_shell)] = std::numeric_limits<double>::quiet_NaN();
  std::fill(system_potentials.begin(), system_potentials.end(), 0.0);
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_system_cpu(view, target, charges.data(),
                                                                system_potentials.data(),
                                                                error) == GPUXTB_STATUS_SUCCESS);
  CHECK(system_potentials[static_cast<std::size_t>(target_shell)] ==
        expected[static_cast<std::size_t>(target_shell)]);
  charges[static_cast<std::size_t>(peer_shell)] = saved_peer_charge;

  std::vector<double> gamma3 = plan.shell_gamma3;
  gamma3[static_cast<std::size_t>(peer_shell)] = std::numeric_limits<double>::quiet_NaN();
  ES3View poisoned_peer_view = view;
  poisoned_peer_view.shell_gamma3 = gamma3.data();
  std::fill(system_potentials.begin(), system_potentials.end(), 0.0);
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_system_cpu(
            poisoned_peer_view, target, charges.data(), system_potentials.data(), error) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(system_potentials[static_cast<std::size_t>(target_shell)] ==
        expected[static_cast<std::size_t>(target_shell)]);

  /* Target poison is a target-only failure. */
  const double saved_target_charge = charges[static_cast<std::size_t>(target_shell)];
  charges[static_cast<std::size_t>(target_shell)] = std::numeric_limits<double>::quiet_NaN();
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_system_cpu(view, target, charges.data(),
                                                                system_potentials.data(), error) ==
        GPUXTB_STATUS_INTERNAL_ERROR);
  charges[static_cast<std::size_t>(target_shell)] = saved_target_charge;

  gamma3 = plan.shell_gamma3;
  gamma3[static_cast<std::size_t>(target_shell)] = std::numeric_limits<double>::quiet_NaN();
  ES3View poisoned_target_view = view;
  poisoned_target_view.shell_gamma3 = gamma3.data();
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_system_cpu(
            poisoned_target_view, target, charges.data(), system_potentials.data(), error) ==
        GPUXTB_STATUS_INTERNAL_ERROR);

  /* Out-of-range system and truncated view are structural failures. */
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_system_cpu(view, -1, charges.data(),
                                                                system_potentials.data(), error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_system_cpu(
            view, plan.batch_size, charges.data(), system_potentials.data(), error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  ES3View bad_view = view;
  --bad_view.shell_gamma3_count;
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_system_cpu(bad_view, target, charges.data(),
                                                                system_potentials.data(), error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);

  /* Output aliasing and NULL outputs are rejected without modifying charges. */
  const std::vector<double> saved_charges = charges;
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_system_cpu(
            view, target, charges.data(), charges.data(), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(charges == saved_charges);
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_system_cpu(
            view, target, charges.data(), nullptr, error) == GPUXTB_STATUS_INVALID_ARGUMENT);

  /* The one-system potential allocates nothing. */
  const std::size_t before = allocation_test::count.load(std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const gpuxtb_status_t status = gpuxtb::detail::gfn2::evaluate_es3_potential_system_cpu(
      view, target, charges.data(), system_potentials.data(), error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  CHECK(status == GPUXTB_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == before);
  return 0;
}

int test_extreme_arithmetic() {
  const std::vector<std::int64_t> offsets{0, 1};
  const std::vector<std::int32_t> atomic_numbers{1};
  BasisPlan basis;
  ES3Plan plan;
  std::string error;
  CHECK(make_plan(offsets, atomic_numbers, basis, plan, error));
  const ES3View view = gpuxtb::detail::gfn2::make_es3_view(plan);
  CHECK(plan.shell_gamma3[0] == 0.08);

  /* q^2 overflows in double, while Gamma3*q^2 remains representable. */
  std::array<double, 1> charges{2.0 * std::sqrt(std::numeric_limits<double>::max())};
  std::array<double, 1> potentials{-1.0};
  CHECK(std::isinf(charges[0] * charges[0]));
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(view, charges.data(), potentials.data(),
                                                         error) == GPUXTB_STATUS_SUCCESS);
  const double expected_potential = static_cast<double>(
      static_cast<long double>(charges[0]) * static_cast<long double>(charges[0]) * 0.08L);
  CHECK(std::isfinite(potentials[0]));
  CHECK(potentials[0] == expected_potential);

  /* q^3 likewise overflows before multiplication by Gamma3/3. */
  charges[0] = 1.5e103;
  CHECK(std::isinf(charges[0] * charges[0] * charges[0]));
  std::array<double, 1> energies{};
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_cpu(view, charges.data(), energies.data(), error) ==
        GPUXTB_STATUS_SUCCESS);
  const double expected_energy = static_cast<double>(
      static_cast<long double>(charges[0]) * static_cast<long double>(charges[0]) *
      static_cast<long double>(charges[0]) * 0.08L / 3.0L);
  CHECK(std::isfinite(energies[0]));
  CHECK(energies[0] == expected_energy);

  charges[0] = std::numeric_limits<double>::denorm_min();
  potentials[0] = 1.0;
  energies[0] = -2.0;
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(view, charges.data(), potentials.data(),
                                                         error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_cpu(view, charges.data(), energies.data(), error) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(potentials[0] == 0.0);
  CHECK(energies[0] == -2.0);

  /* Recover when q^2/q^3 underflows before a huge finite Gamma3 restores range. */
  std::array<double, 1> huge_gamma3{std::numeric_limits<double>::max()};
  ES3View restored_view = view;
  restored_view.shell_gamma3 = huge_gamma3.data();
  charges[0] = 1.0e-200;
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(
            restored_view, charges.data(), potentials.data(), error) == GPUXTB_STATUS_SUCCESS);
  const double restored_potential = static_cast<double>(static_cast<long double>(charges[0]) *
                                                        static_cast<long double>(charges[0]) *
                                                        static_cast<long double>(huge_gamma3[0]));
  CHECK(potentials[0] == restored_potential);
  CHECK(potentials[0] > 0.0);

  charges[0] = 1.0e-110;
  energies[0] = 0.0;
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_cpu(restored_view, charges.data(), energies.data(),
                                                 error) == GPUXTB_STATUS_SUCCESS);
  const double restored_energy = static_cast<double>(
      static_cast<long double>(charges[0]) * static_cast<long double>(charges[0]) *
      static_cast<long double>(charges[0]) * static_cast<long double>(huge_gamma3[0]) / 3.0L);
  CHECK(energies[0] == restored_energy);
  CHECK(energies[0] > 0.0);

  /* A zero Gamma3 is exactly zero even when q^2 or q^3 would overflow first. */
  std::array<double, 1> zero_gamma3{};
  ES3View zero_view = view;
  zero_view.shell_gamma3 = zero_gamma3.data();
  charges[0] = std::numeric_limits<double>::max();
  potentials[0] = -1.0;
  energies[0] = 0.75;
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(
            zero_view, charges.data(), potentials.data(), error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_cpu(zero_view, charges.data(), energies.data(),
                                                 error) == GPUXTB_STATUS_SUCCESS);
  CHECK(potentials[0] == 0.0);
  CHECK(energies[0] == 0.75);
  return 0;
}

int test_validation_and_failure_atomicity() {
  const std::vector<std::int64_t> offsets{0, 1, 2};
  const std::vector<std::int32_t> atomic_numbers{1, 8};
  BasisPlan basis;
  ES3Plan plan;
  std::string error;
  CHECK(make_plan(offsets, atomic_numbers, basis, plan, error));

  const ES3Plan saved_plan = plan;
  const std::array<std::int32_t, 2> mismatched_numbers{2, 8};
  CHECK(gpuxtb::detail::gfn2::make_es3_plan(basis, mismatched_numbers.data(), plan, error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(plan.batch_size == saved_plan.batch_size && plan.total_shells == saved_plan.total_shells &&
        plan.batch_shell_offsets == saved_plan.batch_shell_offsets &&
        plan.shell_gamma3 == saved_plan.shell_gamma3);
  CHECK(gpuxtb::detail::gfn2::make_es3_plan(basis, nullptr, plan, error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);

  BasisPlan corrupt_basis = basis;
  corrupt_basis.angular_momenta[0] = 3u;
  CHECK(gpuxtb::detail::gfn2::make_es3_plan(corrupt_basis, atomic_numbers.data(), plan, error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  corrupt_basis = basis;
  corrupt_basis.shell_to_atom.back() = 0;
  CHECK(gpuxtb::detail::gfn2::make_es3_plan(corrupt_basis, atomic_numbers.data(), plan, error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);

  plan = saved_plan;
  const ES3View valid_view = gpuxtb::detail::gfn2::make_es3_view(plan);
  std::vector<double> charges(static_cast<std::size_t>(plan.total_shells), 0.2);
  std::vector<double> potentials(charges.size(), 17.0);
  std::array<double, 2> energies{3.0, -4.0};
  const std::vector<double> saved_potentials = potentials;
  const std::array<double, 2> saved_energies = energies;

  ES3View bad_view = valid_view;
  bad_view.shell_gamma3 = nullptr;
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(
            bad_view, charges.data(), potentials.data(), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(potentials == saved_potentials);
  bad_view = valid_view;
  --bad_view.shell_gamma3_count;
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(
            bad_view, charges.data(), potentials.data(), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(potentials == saved_potentials);
  std::vector<std::int64_t> bad_offsets = plan.batch_shell_offsets;
  bad_offsets[1] = plan.total_shells + 1;
  bad_view = valid_view;
  bad_view.batch_shell_offsets = bad_offsets.data();
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_cpu(bad_view, charges.data(), energies.data(),
                                                 error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(energies == saved_energies);
  std::vector<double> bad_gamma3 = plan.shell_gamma3;
  bad_gamma3.back() = std::numeric_limits<double>::quiet_NaN();
  bad_view = valid_view;
  bad_view.shell_gamma3 = bad_gamma3.data();
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(
            bad_view, charges.data(), potentials.data(), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(potentials == saved_potentials);

  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(valid_view, nullptr, potentials.data(),
                                                         error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(valid_view, charges.data(), nullptr,
                                                         error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_cpu(valid_view, nullptr, energies.data(), error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_cpu(valid_view, charges.data(), nullptr, error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(potentials == saved_potentials && energies == saved_energies);

  charges.back() = std::numeric_limits<double>::infinity();
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(valid_view, charges.data(),
                                                         potentials.data(),
                                                         error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_cpu(valid_view, charges.data(), energies.data(),
                                                 error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(potentials == saved_potentials && energies == saved_energies);

  charges.assign(charges.size(), 0.2);
  charges.back() = std::numeric_limits<double>::max();
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(valid_view, charges.data(),
                                                         potentials.data(),
                                                         error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_cpu(valid_view, charges.data(), energies.data(),
                                                 error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(potentials == saved_potentials && energies == saved_energies);

  charges.assign(charges.size(), 0.2);
  energies[1] = std::numeric_limits<double>::infinity();
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_cpu(valid_view, charges.data(), energies.data(),
                                                 error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(energies[0] == saved_energies[0]);
  CHECK(std::isinf(energies[1]));

  energies = saved_energies;
  charges[0] = 1.0e103;
  energies[0] = std::numeric_limits<double>::max();
  const std::array<double, 2> overflow_seed = energies;
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_cpu(valid_view, charges.data(), energies.data(),
                                                 error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(energies == overflow_seed);

  std::vector<double> overlapping_charges(charges.size() + 1u, 0.2);
  const std::vector<double> saved_overlapping_charges = overlapping_charges;
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(valid_view, overlapping_charges.data(),
                                                         overlapping_charges.data(),
                                                         error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(overlapping_charges == saved_overlapping_charges);
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(valid_view, overlapping_charges.data(),
                                                         overlapping_charges.data() + 1,
                                                         error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(overlapping_charges == saved_overlapping_charges);

  CHECK(gpuxtb::detail::gfn2::add_es3_energy_cpu(valid_view, overlapping_charges.data(),
                                                 overlapping_charges.data() + 1,
                                                 error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(overlapping_charges == saved_overlapping_charges);

  const std::vector<double> saved_gamma3 = plan.shell_gamma3;
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(valid_view, overlapping_charges.data(),
                                                         plan.shell_gamma3.data(),
                                                         error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(plan.shell_gamma3 == saved_gamma3);
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_cpu(
            valid_view, overlapping_charges.data(),
            reinterpret_cast<double*>(plan.batch_shell_offsets.data()),
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(plan.batch_shell_offsets == saved_plan.batch_shell_offsets);
  return 0;
}

int test_zero_steady_state_allocations() {
  const std::vector<std::int64_t> offsets{0, 2, 5};
  const std::vector<std::int32_t> atomic_numbers{1, 8, 6, 22, 79};
  BasisPlan basis;
  ES3Plan plan;
  std::string error;
  CHECK(make_plan(offsets, atomic_numbers, basis, plan, error));
  const ES3View view = gpuxtb::detail::gfn2::make_es3_view(plan);
  std::vector<double> charges(static_cast<std::size_t>(plan.total_shells), -0.23);
  std::vector<double> potentials(charges.size());
  std::array<double, 2> energies{};

  /* Warm error-string and implementation paths before counting allocations. */
  CHECK(gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(view, charges.data(), potentials.data(),
                                                         error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_cpu(view, charges.data(), energies.data(), error) ==
        GPUXTB_STATUS_SUCCESS);
  double system_energy = 0.0;
  CHECK(gpuxtb::detail::gfn2::add_es3_energy_system_cpu(view, 0, charges.data(), system_energy,
                                                        error) == GPUXTB_STATUS_SUCCESS);
  energies.fill(0.0);
  allocation_test::count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  gpuxtb_status_t potential_status = GPUXTB_STATUS_INTERNAL_ERROR;
  gpuxtb_status_t energy_status = GPUXTB_STATUS_INTERNAL_ERROR;
  gpuxtb_status_t system_energy_status = GPUXTB_STATUS_INTERNAL_ERROR;
  for (int iteration = 0; iteration < 64; ++iteration) {
    potential_status = gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(view, charges.data(),
                                                                        potentials.data(), error);
    energy_status =
        gpuxtb::detail::gfn2::add_es3_energy_cpu(view, charges.data(), energies.data(), error);
    system_energy_status = gpuxtb::detail::gfn2::add_es3_energy_system_cpu(view, 0, charges.data(),
                                                                           system_energy, error);
  }
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  CHECK(potential_status == GPUXTB_STATUS_SUCCESS);
  CHECK(energy_status == GPUXTB_STATUS_SUCCESS);
  CHECK(system_energy_status == GPUXTB_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == 0u);
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_all_elements_and_shell_angular_momenta(); status != 0) {
    return status;
  }
  if (const int status = test_ragged_sequential_and_energy_potential_derivative(); status != 0) {
    return status;
  }
  if (const int status = test_system_energy_failure_isolation_and_binding(); status != 0) {
    return status;
  }
  if (const int status = test_system_potential_matches_batch_and_failure_isolation(); status != 0) {
    return status;
  }
  if (const int status = test_extreme_arithmetic(); status != 0) {
    return status;
  }
  if (const int status = test_validation_and_failure_atomicity(); status != 0) {
    return status;
  }
  return test_zero_steady_state_allocations();
}
