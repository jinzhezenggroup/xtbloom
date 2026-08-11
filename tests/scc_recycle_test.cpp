#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "support/gfn2_scc_test_case.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

using xtbloom::detail::gfn2::EigensolverSolveMode;
using xtbloom::detail::gfn2::restart_scc_driver_system_cpu;
using xtbloom::detail::gfn2::SccDriverState;
using xtbloom::test::gfn2::HostSccCase;
using xtbloom::test::gfn2::HostSccCaseOptions;
using xtbloom::test::gfn2::SmallSystemKind;

struct SolveSequence {
  std::vector<EigensolverSolveMode> modes;
  std::vector<std::uint8_t> confirmations;
  std::vector<std::uint8_t> cooldowns;
};

int run_to_convergence(HostSccCase& calculation, bool force_dense, SolveSequence& sequence,
                       std::string& error) {
  SccDriverState& state = calculation.driver_state();
  while (state.converged[0] == 0u &&
         state.iterations[0] < calculation.options().maximum_iterations) {
    if (force_dense) {
      /* This test-only provenance mismatch disables recycle eligibility while
       * preserving the ordinary dense convergence criteria. */
      if (state.iterations[0] != 0u) {
        state.eigensolver_geometry_generations[0] = UINT64_MAX;
      }
    }
    const xtbloom_status_t status = calculation.run_one_iteration(error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      std::fprintf(stderr, "SCC iteration %llu (nao=%lld) failed with status %d: %s\n",
                   static_cast<unsigned long long>(state.iterations[0]),
                   static_cast<long long>(calculation.eigensolver_plan().maximum_orbitals()),
                   static_cast<int>(status), error.c_str());
      return __LINE__;
    }
    sequence.modes.push_back(state.last_eigensolver_modes[0]);
    sequence.confirmations.push_back(state.dense_confirmations_remaining[0]);
    sequence.cooldowns.push_back(state.recycle_cooldowns_remaining[0]);
  }
  return 0;
}

int test_production_recycle_schedule_and_dense_final_parity() {
  HostSccCaseOptions options;
  options.systems = {SmallSystemKind::kC20H42};
  options.enable_d4 = true;
  options.maximum_iterations = 100u;
  options.mixer_history = 8;
  options.residual_tolerance = 1.0e-10;
  options.energy_tolerance = 1.0e-12;
  options.electronic_temperature = XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE;
  options.use_production_cpu_backend = true;

  std::string error;
  HostSccCase adaptive;
  HostSccCase dense;
  CHECK(HostSccCase::create(options, adaptive, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(HostSccCase::create(options, dense, error) == XTBLOOM_STATUS_SUCCESS);
  SolveSequence adaptive_sequence;
  SolveSequence dense_sequence;
  CHECK(run_to_convergence(adaptive, false, adaptive_sequence, error) == 0);
  CHECK(run_to_convergence(dense, true, dense_sequence, error) == 0);

  const SccDriverState& adaptive_state = adaptive.driver_state();
  const SccDriverState& dense_state = dense.driver_state();
  CHECK(adaptive_state.converged[0] == 1u);
  CHECK(dense_state.converged[0] == 1u);
  CHECK(adaptive_state.recycled_eigensolves[0] >= 1u);
  CHECK(adaptive_state.recycled_eigensolves[0] == 5u);
  CHECK(adaptive_state.recycle_fallbacks[0] == 2u);
  CHECK(adaptive_state.full_eigensolves[0] == 29u);
  CHECK(adaptive_state.full_eigensolves[0] + adaptive_state.recycled_eigensolves[0] ==
        adaptive_state.iterations[0]);
  CHECK(adaptive_state.full_eigensolves[0] < dense_state.full_eigensolves[0]);
  CHECK(dense_state.recycled_eigensolves[0] == 0u);
  CHECK(dense_state.full_eigensolves[0] == dense_state.iterations[0]);

  bool saw_recycle_correction_confirmation = false;
  bool saw_fallback_cooldown = false;
  for (std::size_t iteration = 0u; iteration + 2u < adaptive_sequence.modes.size(); ++iteration) {
    if (adaptive_sequence.modes[iteration] == EigensolverSolveMode::kRecycleFallback) {
      CHECK(adaptive_sequence.cooldowns[iteration] == 1u);
      CHECK(adaptive_sequence.modes[iteration + 1u] == EigensolverSolveMode::kFull);
      CHECK(adaptive_sequence.cooldowns[iteration + 1u] == 0u);
      saw_fallback_cooldown = true;
    }
    if (adaptive_sequence.modes[iteration] != EigensolverSolveMode::kRecycled) {
      continue;
    }
    CHECK(adaptive_sequence.confirmations[iteration] == 2u);
    CHECK(adaptive_sequence.modes[iteration + 1u] == EigensolverSolveMode::kFull);
    CHECK(adaptive_sequence.confirmations[iteration + 1u] == 1u);
    CHECK(adaptive_sequence.modes[iteration + 2u] == EigensolverSolveMode::kFull);
    CHECK(adaptive_sequence.confirmations[iteration + 2u] == 0u);
    saw_recycle_correction_confirmation = true;
  }
  CHECK(saw_fallback_cooldown);
  CHECK(saw_recycle_correction_confirmation);
  CHECK(adaptive_sequence.modes.back() != EigensolverSolveMode::kRecycled);
  CHECK(adaptive_sequence.confirmations.back() == 0u);

  const auto& adaptive_layout = adaptive.wavefunction_layout();
  const auto& adaptive_wavefunction = adaptive.wavefunction();
  const auto& dense_wavefunction = dense.wavefunction();
  double maximum_density_difference = 0.0;
  for (std::int64_t element = 0; element < adaptive_layout.density.element_count; ++element) {
    maximum_density_difference = std::max(
        maximum_density_difference,
        std::abs(adaptive_wavefunction.density[element] - dense_wavefunction.density[element]));
  }
  double maximum_weighted_density_difference = 0.0;
  for (std::int64_t element = 0; element < adaptive_layout.energy_weighted_density.element_count;
       ++element) {
    maximum_weighted_density_difference =
        std::max(maximum_weighted_density_difference,
                 std::abs(adaptive_wavefunction.energy_weighted_density[element] -
                          dense_wavefunction.energy_weighted_density[element]));
  }
  double maximum_shell_charge_difference = 0.0;
  for (std::int64_t element = 0; element < adaptive_layout.qsh.element_count; ++element) {
    maximum_shell_charge_difference =
        std::max(maximum_shell_charge_difference,
                 std::abs(adaptive_wavefunction.qsh[element] - dense_wavefunction.qsh[element]));
  }
  double maximum_atomic_charge_difference = 0.0;
  for (std::int64_t element = 0; element < adaptive_layout.qat.element_count; ++element) {
    maximum_atomic_charge_difference =
        std::max(maximum_atomic_charge_difference,
                 std::abs(adaptive_wavefunction.qat[element] - dense_wavefunction.qat[element]));
  }
  const double energy_difference =
      std::abs(adaptive_state.free_energies[0] - dense_state.free_energies[0]);
  if (maximum_density_difference >= 5.0e-7 || maximum_weighted_density_difference >= 5.0e-7 ||
      maximum_shell_charge_difference >= 5.0e-7 || maximum_atomic_charge_difference >= 5.0e-7 ||
      energy_difference >= 5.0e-10) {
    std::fprintf(stderr,
                 "final parity: density=%g weighted=%g qsh=%g qat=%g energy=%g adaptive "
                 "iters=%llu dense=%llu\n",
                 maximum_density_difference, maximum_weighted_density_difference,
                 maximum_shell_charge_difference, maximum_atomic_charge_difference,
                 energy_difference, static_cast<unsigned long long>(adaptive_state.iterations[0]),
                 static_cast<unsigned long long>(dense_state.iterations[0]));
  }
  CHECK(maximum_density_difference < 5.0e-7);
  CHECK(maximum_weighted_density_difference < 5.0e-7);
  CHECK(maximum_shell_charge_difference < 5.0e-7);
  CHECK(maximum_atomic_charge_difference < 5.0e-7);
  CHECK(energy_difference < 5.0e-10);

  CHECK(restart_scc_driver_system_cpu(adaptive.driver_plan(), 0, adaptive.wavefunction(),
                                      adaptive.mixer_state(), adaptive.driver_state(),
                                      error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(adaptive.driver_state().iterations[0] == 0u);
  CHECK(adaptive.driver_state().eigensolver_geometry_generations[0] == 0u);
  CHECK(adaptive.driver_state().full_eigensolves[0] == 0u);
  CHECK(adaptive.driver_state().recycled_eigensolves[0] == 0u);
  CHECK(adaptive.driver_state().recycle_fallbacks[0] == 0u);
  CHECK(adaptive.driver_state().last_eigensolver_modes[0] == EigensolverSolveMode::kFull);
  CHECK(adaptive.driver_state().dense_confirmations_remaining[0] == 0u);
  CHECK(adaptive.driver_state().recycle_cooldowns_remaining[0] == 0u);
  return 0;
}

int test_small_and_unrestricted_systems_stay_dense() {
  HostSccCaseOptions small_options;
  small_options.systems = {SmallSystemKind::kC12H26};
  small_options.enable_d4 = true;
  small_options.maximum_iterations = 60u;
  small_options.mixer_history = 8;
  small_options.residual_tolerance = 2.0e-5;
  small_options.energy_tolerance = 1.0e-6;
  small_options.electronic_temperature = XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE;
  small_options.use_production_cpu_backend = true;

  std::string error;
  HostSccCase small;
  CHECK(HostSccCase::create(small_options, small, error) == XTBLOOM_STATUS_SUCCESS);
  SolveSequence small_sequence;
  CHECK(run_to_convergence(small, false, small_sequence, error) == 0);
  CHECK(small.driver_state().converged[0] == 1u);
  CHECK(small.driver_state().recycled_eigensolves[0] == 0u);
  CHECK(small.driver_state().recycle_fallbacks[0] == 0u);
  CHECK(small.driver_state().full_eigensolves[0] == small.driver_state().iterations[0]);

  HostSccCaseOptions unrestricted_options = small_options;
  unrestricted_options.systems = {SmallSystemKind::kC20H42};
  unrestricted_options.unpaired_electrons = {2};
  unrestricted_options.spin_channels = {2};
  unrestricted_options.maximum_iterations = 100u;
  unrestricted_options.residual_tolerance = 1.0e-8;
  unrestricted_options.energy_tolerance = 1.0e-10;
  HostSccCase unrestricted;
  CHECK(HostSccCase::create(unrestricted_options, unrestricted, error) == XTBLOOM_STATUS_SUCCESS);
  SolveSequence unrestricted_sequence;
  CHECK(run_to_convergence(unrestricted, false, unrestricted_sequence, error) == 0);
  CHECK(unrestricted.driver_state().converged[0] == 1u);
  CHECK(unrestricted.driver_state().recycled_eigensolves[0] == 0u);
  CHECK(unrestricted.driver_state().recycle_fallbacks[0] == 0u);
  CHECK(unrestricted.driver_state().full_eigensolves[0] ==
        unrestricted.driver_state().iterations[0]);
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_production_recycle_schedule_and_dense_final_parity(); status != 0) {
    return status;
  }
  return test_small_and_unrestricted_systems_stay_dense();
}
