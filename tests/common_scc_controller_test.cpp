#include <cmath>
#include <cstdint>
#include <limits>

#include "model/common/scc_controller.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

using xtbloom::detail::common::SccControllerDecision;
using xtbloom::detail::common::SccControllerObservation;
using xtbloom::detail::common::SccControllerState;

SccControllerObservation observation(double norm, double previous, double cosine) {
  return {norm, previous, cosine, true, true};
}

int test_contracts_then_recovers_conservative_level() {
  const auto config = xtbloom::detail::common::make_scc_controller_config(0.4);
  SccControllerState state;
  state.damping_level = 0u;
  SccControllerDecision decision;
  CHECK(xtbloom::detail::common::advance_scc_controller(config, state, observation(0.5, 1.0, 1.0),
                                                        decision));
  CHECK(!decision.restart_history);
  state = decision.next_state;
  CHECK(xtbloom::detail::common::advance_scc_controller(config, state, observation(0.2, 0.5, 1.0),
                                                        decision));
  CHECK(decision.restart_history);
  CHECK(decision.next_state.damping_level == 1u);
  CHECK(std::abs(decision.damping - 0.4) < 1.0e-15);
  return 0;
}

int test_detects_bounded_two_cycle() {
  const auto config = xtbloom::detail::common::make_scc_controller_config(0.4);
  SccControllerState state;
  SccControllerDecision decision;
  CHECK(xtbloom::detail::common::advance_scc_controller(config, state, observation(1.0, 1.0, -1.0),
                                                        decision));
  CHECK(!decision.restart_history);
  state = decision.next_state;
  CHECK(xtbloom::detail::common::advance_scc_controller(config, state, observation(1.0, 1.0, -1.0),
                                                        decision));
  CHECK(decision.restart_history);
  CHECK(decision.next_state.damping_level == 0u);
  CHECK(std::abs(decision.damping - 0.2) < 1.0e-15);
  return 0;
}

int test_distinguishes_slow_contraction_from_stagnation() {
  const auto config = xtbloom::detail::common::make_scc_controller_config(0.4);
  SccControllerState state;
  SccControllerDecision decision;
  for (int iteration = 0; iteration < 4; ++iteration) {
    CHECK(xtbloom::detail::common::advance_scc_controller(config, state,
                                                          observation(0.99, 1.0, -0.2), decision));
    CHECK(!decision.restart_history);
    state = decision.next_state;
  }
  CHECK(state.damping_level == 1u);
  return 0;
}

int test_restart_bound_falls_back_to_baseline() {
  auto config = xtbloom::detail::common::make_scc_controller_config(0.4);
  config.maximum_restarts = 1u;
  SccControllerState state;
  SccControllerDecision decision;
  CHECK(xtbloom::detail::common::advance_scc_controller(config, state, observation(2.0, 1.0, 1.0),
                                                        decision));
  CHECK(decision.restart_history);
  CHECK(decision.next_state.restart_count == 1u);
  state = decision.next_state;
  CHECK(xtbloom::detail::common::advance_scc_controller(config, state, observation(2.0, 1.0, 1.0),
                                                        decision));
  CHECK(decision.next_state.fallback_to_baseline);
  CHECK(decision.next_state.damping_level == 1u);
  CHECK(decision.restart_history);
  return 0;
}

int test_fallback_clears_history_even_at_baseline_damping() {
  auto config = xtbloom::detail::common::make_scc_controller_config(0.9);
  config.maximum_restarts = 0u;
  SccControllerState state;
  SccControllerDecision decision;
  CHECK(config.damping_levels[1] == 0.9);
  CHECK(xtbloom::detail::common::advance_scc_controller(config, state, observation(2.0, 1.0, 1.0),
                                                        decision));
  CHECK(decision.next_state.fallback_to_baseline);
  CHECK(decision.next_state.damping_level == 1u);
  CHECK(decision.damping == 0.9);
  CHECK(decision.restart_history);

  state = decision.next_state;
  CHECK(xtbloom::detail::common::advance_scc_controller(config, state, observation(2.0, 1.0, 1.0),
                                                        decision));
  CHECK(!decision.restart_history);
  CHECK(decision.damping == 0.9);
  return 0;
}

int test_thresholds_are_strict_and_deterministic() {
  const auto config = xtbloom::detail::common::make_scc_controller_config(0.4);
  SccControllerState state;
  SccControllerDecision decision;
  CHECK(xtbloom::detail::common::advance_scc_controller(
      config, state, observation(config.divergence_ratio, 1.0, 1.0), decision));
  CHECK(!decision.restart_history);
  CHECK(xtbloom::detail::common::advance_scc_controller(
      config, state,
      observation(std::nextafter(config.divergence_ratio, std::numeric_limits<double>::infinity()),
                  1.0, 1.0),
      decision));
  CHECK(decision.restart_history);
  return 0;
}

int test_zero_and_nonfinite_observations_are_safe() {
  const auto config = xtbloom::detail::common::make_scc_controller_config(0.4);
  SccControllerState state;
  state.stagnation_count = 2u;
  SccControllerDecision decision;
  SccControllerObservation zero{0.0, 0.0, 0.0, true, false};
  CHECK(xtbloom::detail::common::advance_scc_controller(config, state, zero, decision));
  CHECK(!decision.restart_history);
  CHECK(decision.next_state.stagnation_count == state.stagnation_count);

  SccControllerObservation nonfinite = zero;
  nonfinite.weighted_residual_norm = std::numeric_limits<double>::quiet_NaN();
  CHECK(!xtbloom::detail::common::advance_scc_controller(config, state, nonfinite, decision));
  CHECK(decision.next_state.stagnation_count == state.stagnation_count);
  CHECK(decision.next_state.damping_level == state.damping_level);
  return 0;
}

int test_cooldown_prevents_policy_chatter() {
  auto config = xtbloom::detail::common::make_scc_controller_config(0.4);
  config.restart_cooldown = 3u;
  SccControllerState state;
  SccControllerDecision decision;
  CHECK(xtbloom::detail::common::advance_scc_controller(config, state, observation(2.0, 1.0, 1.0),
                                                        decision));
  CHECK(decision.restart_history);
  state = decision.next_state;
  for (int iteration = 0; iteration < 2; ++iteration) {
    CHECK(xtbloom::detail::common::advance_scc_controller(config, state,
                                                          observation(1.0, 1.0, -1.0), decision));
    CHECK(!decision.restart_history);
    state = decision.next_state;
  }
  return 0;
}

int test_trust_radius_tracks_weighted_residual_with_a_floor() {
  const auto config = xtbloom::detail::common::make_scc_controller_config(0.4);
  SccControllerState state;
  SccControllerDecision decision;
  double recomputed = 0.0;
  CHECK(xtbloom::detail::common::compute_scc_controller_trust_radius(config, 0.25, recomputed));
  CHECK(recomputed == 0.5);
  CHECK(xtbloom::detail::common::advance_scc_controller(config, state,
                                                        {0.25, 0.0, 0.0, false, false}, decision));
  CHECK(decision.maximum_weighted_step_norm == 0.5);
  CHECK(xtbloom::detail::common::advance_scc_controller(config, state,
                                                        {0.0, 0.0, 0.0, false, false}, decision));
  CHECK(decision.maximum_weighted_step_norm == config.minimum_trust_radius);
  CHECK(!xtbloom::detail::common::compute_scc_controller_trust_radius(
      config, std::numeric_limits<double>::quiet_NaN(), recomputed));
  CHECK(recomputed == 0.0);
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_contracts_then_recovers_conservative_level(); status != 0) {
    return status;
  }
  if (const int status = test_detects_bounded_two_cycle(); status != 0) {
    return status;
  }
  if (const int status = test_distinguishes_slow_contraction_from_stagnation(); status != 0) {
    return status;
  }
  if (const int status = test_restart_bound_falls_back_to_baseline(); status != 0) {
    return status;
  }
  if (const int status = test_fallback_clears_history_even_at_baseline_damping(); status != 0) {
    return status;
  }
  if (const int status = test_thresholds_are_strict_and_deterministic(); status != 0) {
    return status;
  }
  if (const int status = test_zero_and_nonfinite_observations_are_safe(); status != 0) {
    return status;
  }
  if (const int status = test_cooldown_prevents_policy_chatter(); status != 0) {
    return status;
  }
  return test_trust_radius_tracks_weighted_residual_with_a_floor();
}
