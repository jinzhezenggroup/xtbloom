#include "model/gfn2/spin.hpp"

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

using xtbloom::detail::gfn2::BasisPlan;
using xtbloom::detail::gfn2::SpinPolarizationPlan;
using xtbloom::detail::gfn2::SpinPolarizationView;
using xtbloom::detail::gfn2::WavefunctionLayout;

bool near(double actual, double expected, double tolerance = 2.0e-15) {
  return std::abs(actual - expected) <= tolerance;
}

bool make_plan(const std::vector<std::int64_t>& atom_offsets,
               const std::vector<std::int32_t>& atomic_numbers, const std::vector<double>& charges,
               const std::vector<std::int32_t>& unpaired,
               const std::vector<std::int32_t>& spin_channels, BasisPlan& basis,
               WavefunctionLayout& wavefunction, SpinPolarizationPlan& spin, std::string& error) {
  const std::int64_t batch = static_cast<std::int64_t>(charges.size());
  return xtbloom::detail::gfn2::make_basis_plan(
             batch, static_cast<std::int64_t>(atomic_numbers.size()), atom_offsets.data(),
             atomic_numbers.data(), basis, error) == XTBLOOM_STATUS_SUCCESS &&
         xtbloom::detail::gfn2::make_wavefunction_layout(
             basis, atomic_numbers.data(), charges.data(), unpaired.data(), spin_channels.data(),
             wavefunction, error) == XTBLOOM_STATUS_SUCCESS &&
         xtbloom::detail::gfn2::make_spin_polarization_plan(basis, wavefunction, spin, error) ==
             XTBLOOM_STATUS_SUCCESS;
}

int test_hydrogen_literal_and_chromium_shell_order() {
  const std::vector<std::int64_t> atom_offsets{0, 1, 2, 3};
  const std::vector<std::int32_t> atomic_numbers{1, 1, 24};
  const std::vector<double> charges{0.0, 0.0, 0.0};
  const std::vector<std::int32_t> unpaired{1, 1, 0};
  const std::vector<std::int32_t> channels{1, 2, 2};
  BasisPlan basis;
  WavefunctionLayout wavefunction;
  SpinPolarizationPlan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, atomic_numbers, charges, unpaired, channels, basis, wavefunction,
                  plan, error));
  CHECK(error.empty());
  CHECK(basis.angular_momenta.size() == 5u);
  CHECK((std::array<std::uint8_t, 3>{basis.angular_momenta[2], basis.angular_momenta[3],
                                     basis.angular_momenta[4]} ==
         std::array<std::uint8_t, 3>{2u, 0u, 1u}));

  /* Cr must use its actual d,s,p shell order, not an assumed s,p,d order. */
  const std::array<double, 9> expected_cr{{
      -0.015775,
      -0.003725,
      -0.001463,
      -0.003725,
      -0.014475,
      -0.011612,
      -0.001463,
      -0.011612,
      -0.016000,
  }};
  CHECK(plan.coupling_offsets == std::vector<std::int64_t>({0, 1, 2, 11}));
  CHECK(std::equal(expected_cr.begin(), expected_cr.end(), plan.coupling_matrices.begin() + 2));

  const SpinPolarizationView view = xtbloom::detail::gfn2::make_spin_polarization_view(plan);
  std::vector<double> populations(static_cast<std::size_t>(view.shell_population_elements), 0.0);
  CHECK(wavefunction.qsh.system_offsets == std::vector<std::int64_t>({0, 1, 3, 9}));
  populations[2] = -1.0; /* H: m=N_beta-N_alpha=-1 for an alpha doublet. */
  const std::array<double, 3> cr_magnetization{{0.2, -0.3, 0.4}};
  std::copy(cr_magnetization.begin(), cr_magnetization.end(), populations.begin() + 6);

  std::array<double, 3> energies{{91.0, 91.0, 91.0}};
  std::vector<double> potentials(populations.size(), 91.0);
  CHECK(xtbloom::detail::gfn2::evaluate_spin_polarization_cpu(view, populations.data(),
                                                              energies.data(), potentials.data(),
                                                              error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(energies[0] == 0.0);
  CHECK(energies[1] == -0.0358125);
  CHECK(potentials[0] == 0.0 && potentials[1] == 0.0);
  CHECK(potentials[2] == 0.071625);
  for (std::size_t index = 3u; index < 6u; ++index) {
    CHECK(potentials[index] == 0.0);
  }

  std::array<double, 3> expected_cr_potential{};
  double expected_cr_energy = 0.0;
  for (std::size_t row = 0; row < 3u; ++row) {
    for (std::size_t column = 0; column < 3u; ++column) {
      expected_cr_potential[row] = std::fma(expected_cr[row * 3u + column],
                                            cr_magnetization[column], expected_cr_potential[row]);
    }
    expected_cr_energy =
        std::fma(0.5 * cr_magnetization[row], expected_cr_potential[row], expected_cr_energy);
    CHECK(potentials[6u + row] == expected_cr_potential[row]);
  }
  CHECK(energies[2] == expected_cr_energy);
  return 0;
}

int test_energy_potential_derivative_and_zero_allocation() {
  const std::vector<std::int64_t> atom_offsets{0, 2};
  const std::vector<std::int32_t> atomic_numbers{8, 24};
  const std::vector<double> charges{0.0};
  const std::vector<std::int32_t> unpaired{0};
  const std::vector<std::int32_t> channels{2};
  BasisPlan basis;
  WavefunctionLayout wavefunction;
  SpinPolarizationPlan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, atomic_numbers, charges, unpaired, channels, basis, wavefunction,
                  plan, error));
  const SpinPolarizationView view = xtbloom::detail::gfn2::make_spin_polarization_view(plan);
  const std::int64_t shells = basis.total_shells;
  CHECK(view.shell_population_elements == 2 * shells);
  std::vector<double> populations(static_cast<std::size_t>(view.shell_population_elements), 0.0);
  for (std::int64_t shell = 0; shell < shells; ++shell) {
    populations[static_cast<std::size_t>(shells + shell)] =
        -0.51 + 0.17 * static_cast<double>(shell);
  }
  std::array<double, 1> energy{};
  std::vector<double> potentials(populations.size());

  allocation_test::count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const xtbloom_status_t status = xtbloom::detail::gfn2::evaluate_spin_polarization_cpu(
      view, populations.data(), energy.data(), potentials.data(), error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  CHECK(status == XTBLOOM_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == 0u);

  constexpr double step = 1.0e-6;
  std::array<double, 1> perturbed_energy{};
  std::vector<double> perturbed_potential(populations.size());
  for (std::int64_t shell = 0; shell < shells; ++shell) {
    const std::size_t index = static_cast<std::size_t>(shells + shell);
    populations[index] += step;
    CHECK(xtbloom::detail::gfn2::evaluate_spin_polarization_cpu(
              view, populations.data(), perturbed_energy.data(), perturbed_potential.data(),
              error) == XTBLOOM_STATUS_SUCCESS);
    const double right = perturbed_energy[0];
    populations[index] -= 2.0 * step;
    CHECK(xtbloom::detail::gfn2::evaluate_spin_polarization_cpu(
              view, populations.data(), perturbed_energy.data(), perturbed_potential.data(),
              error) == XTBLOOM_STATUS_SUCCESS);
    const double left = perturbed_energy[0];
    populations[index] += step;
    CHECK(near((right - left) / (2.0 * step), potentials[index], 2.0e-11));
  }
  return 0;
}

int test_failure_atomicity_and_descriptor_validation() {
  const std::vector<std::int64_t> atom_offsets{0, 1};
  const std::vector<std::int32_t> atomic_numbers{1};
  const std::vector<double> charges{0.0};
  const std::vector<std::int32_t> unpaired{1};
  const std::vector<std::int32_t> channels{2};
  BasisPlan basis;
  WavefunctionLayout wavefunction;
  SpinPolarizationPlan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, atomic_numbers, charges, unpaired, channels, basis, wavefunction,
                  plan, error));
  const SpinPolarizationView view = xtbloom::detail::gfn2::make_spin_polarization_view(plan);
  std::array<double, 2> populations{{0.0, -1.0}};
  std::array<double, 1> energies{{17.0}};
  std::array<double, 2> potentials{{19.0, 23.0}};
  const std::array<double, 2> potential_sentinel{{19.0, 23.0}};

  populations[1] = std::numeric_limits<double>::quiet_NaN();
  CHECK(xtbloom::detail::gfn2::evaluate_spin_polarization_cpu(
            view, populations.data(), energies.data(), potentials.data(), error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energies[0] == 17.0 && potentials == potential_sentinel);

  populations[1] = std::numeric_limits<double>::max();
  CHECK(xtbloom::detail::gfn2::evaluate_spin_polarization_cpu(
            view, populations.data(), energies.data(), potentials.data(), error) ==
        XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(energies[0] == 17.0 && potentials == potential_sentinel);

  populations[1] = -1.0;
  SpinPolarizationView malformed = view;
  malformed.coupling_matrix_count -= 1;
  CHECK(xtbloom::detail::gfn2::evaluate_spin_polarization_cpu(
            malformed, populations.data(), energies.data(), potentials.data(), error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energies[0] == 17.0 && potentials == potential_sentinel);

  CHECK(xtbloom::detail::gfn2::evaluate_spin_polarization_cpu(
            view, populations.data(), populations.data(), potentials.data(), error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  return 0;
}

int test_one_system_energy_potential_and_failure_isolation() {
  const std::vector<std::int64_t> atom_offsets{0, 1, 2};
  const std::vector<std::int32_t> atomic_numbers{1, 1};
  const std::vector<double> charges{0.0, 0.0};
  const std::vector<std::int32_t> unpaired{1, 1};
  const std::vector<std::int32_t> channels{2, 2};
  BasisPlan basis;
  WavefunctionLayout wavefunction;
  SpinPolarizationPlan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, atomic_numbers, charges, unpaired, channels, basis, wavefunction,
                  plan, error));
  const SpinPolarizationView view = xtbloom::detail::gfn2::make_spin_polarization_view(plan);
  std::vector<double> populations(static_cast<std::size_t>(view.shell_population_elements), 0.0);
  populations[1] = -1.0; /* first member: alpha doublet */
  populations[3] = 0.25; /* second member: arbitrary magnetization */
  std::vector<double> batch_potentials(populations.size(), 91.0);
  std::array<double, 2> batch_energies{{91.0, 91.0}};
  CHECK(xtbloom::detail::gfn2::evaluate_spin_polarization_cpu(
            view, populations.data(), batch_energies.data(), batch_potentials.data(), error) ==
        XTBLOOM_STATUS_SUCCESS);

  /* One-system potentials reproduce the batch potential slice. */
  std::vector<double> system_potentials(populations.size(), 0.0);
  for (std::int64_t system = 0; system < view.batch_size; ++system) {
    double energy = 0.0;
    CHECK(xtbloom::detail::gfn2::evaluate_spin_polarization_system_cpu(
              view, system, populations.data(), energy, system_potentials.data(), error) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(energy == batch_energies[static_cast<std::size_t>(system)]);
    const std::int64_t begin = view.shell_population_offsets[system];
    const std::int64_t end = view.shell_population_offsets[system + 1];
    for (std::int64_t element = begin; element < end; ++element) {
      CHECK(system_potentials[static_cast<std::size_t>(element)] ==
            batch_potentials[static_cast<std::size_t>(element)]);
    }
  }

  /* A poisoned peer population must not affect the target member. */
  const double saved_peer = populations[3];
  populations[3] = std::numeric_limits<double>::quiet_NaN();
  double isolated_energy = -3.25;
  std::fill(system_potentials.begin(), system_potentials.end(), 0.0);
  CHECK(xtbloom::detail::gfn2::evaluate_spin_polarization_system_cpu(
            view, 0, populations.data(), isolated_energy, system_potentials.data(), error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(isolated_energy == batch_energies[0]);
  CHECK(system_potentials[0] == batch_potentials[0]);
  CHECK(system_potentials[1] == batch_potentials[1]);
  populations[3] = saved_peer;

  /* Target numerical poison is a target-only failure. */
  populations[1] = std::numeric_limits<double>::quiet_NaN();
  double unchanged_energy = 1.75;
  std::fill(system_potentials.begin(), system_potentials.end(), 11.0);
  const std::vector<double> untouched(system_potentials);
  CHECK(xtbloom::detail::gfn2::evaluate_spin_polarization_system_cpu(
            view, 0, populations.data(), unchanged_energy, system_potentials.data(), error) ==
        XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(unchanged_energy == 1.75);
  CHECK(system_potentials == untouched);
  populations[1] = -1.0;

  CHECK(xtbloom::detail::gfn2::evaluate_spin_polarization_system_cpu(
            view, -1, populations.data(), unchanged_energy, system_potentials.data(), error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::evaluate_spin_polarization_system_cpu(
            view, view.batch_size, populations.data(), unchanged_energy, system_potentials.data(),
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  CHECK(xtbloom::detail::gfn2::evaluate_spin_polarization_system_cpu(
            view, 0, populations.data(), system_potentials[0], system_potentials.data(), error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::evaluate_spin_polarization_system_cpu(
            view, 0, populations.data(), populations[0], system_potentials.data(), error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);

  /* No per-call allocation. */
  const std::size_t before = allocation_test::count.load(std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const xtbloom_status_t status = xtbloom::detail::gfn2::evaluate_spin_polarization_system_cpu(
      view, 0, populations.data(), unchanged_energy, system_potentials.data(), error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  CHECK(status == XTBLOOM_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == before);
  return 0;
}

}  // namespace

int main() {
  if (const int line = test_hydrogen_literal_and_chromium_shell_order(); line != 0) {
    return line;
  }
  if (const int line = test_energy_potential_derivative_and_zero_allocation(); line != 0) {
    return line;
  }
  if (const int line = test_failure_atomicity_and_descriptor_validation(); line != 0) {
    return line;
  }
  if (const int line = test_one_system_energy_potential_and_failure_isolation(); line != 0) {
    return line;
  }
  return 0;
}
