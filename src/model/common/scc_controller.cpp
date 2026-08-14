#include "model/common/scc_controller.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <algorithm>
#include <cmath>
#include <limits>

namespace xtbloom::detail::common {
namespace {

constexpr std::uint8_t kBaselineDampingLevel = 1u;
constexpr std::uint8_t kMaximumDampingLevel = 2u;

std::uint8_t increment_saturating(std::uint8_t value) noexcept {
  return value == std::numeric_limits<std::uint8_t>::max() ? value
                                                           : static_cast<std::uint8_t>(value + 1u);
}

bool valid_config(const SccControllerConfig& config) noexcept {
  return std::isfinite(config.damping_levels[0]) && config.damping_levels[0] > 0.0 &&
         std::isfinite(config.damping_levels[1]) &&
         config.damping_levels[1] >= config.damping_levels[0] &&
         std::isfinite(config.damping_levels[2]) &&
         config.damping_levels[2] >= config.damping_levels[1] && config.damping_levels[2] <= 1.0 &&
         std::isfinite(config.contraction_ratio) && config.contraction_ratio > 0.0 &&
         std::isfinite(config.divergence_ratio) &&
         config.divergence_ratio > config.contraction_ratio &&
         std::isfinite(config.stagnation_ratio_minimum) &&
         std::isfinite(config.stagnation_ratio_maximum) &&
         config.stagnation_ratio_minimum <= config.stagnation_ratio_maximum &&
         std::isfinite(config.stagnation_cosine) && std::isfinite(config.two_cycle_ratio_minimum) &&
         std::isfinite(config.two_cycle_ratio_maximum) &&
         config.two_cycle_ratio_minimum <= config.two_cycle_ratio_maximum &&
         std::isfinite(config.two_cycle_cosine) && std::isfinite(config.angle_norm_floor) &&
         config.angle_norm_floor >= 0.0 && std::isfinite(config.trust_radius_multiplier) &&
         config.trust_radius_multiplier >= 1.0 && std::isfinite(config.minimum_trust_radius) &&
         config.minimum_trust_radius > 0.0 && config.contraction_window > 0u &&
         config.stagnation_window > 0u && config.two_cycle_window > 0u;
}

void reset_event_counters(SccControllerState& state) noexcept {
  state.contraction_count = 0u;
  state.stagnation_count = 0u;
  state.two_cycle_count = 0u;
}

void request_restart(const SccControllerConfig& config, SccControllerState& state,
                     std::uint8_t damping_level, SccControllerDecision& decision) noexcept {
  state.damping_level = damping_level;
  state.cooldown_remaining = config.restart_cooldown;
  ++state.restart_count;
  reset_event_counters(state);
  decision.restart_history = true;
}

}  // namespace

SccControllerConfig make_scc_controller_config(double baseline_damping) noexcept {
  SccControllerConfig config;
  if (!std::isfinite(baseline_damping) || baseline_damping <= 0.0 || baseline_damping > 1.0) {
    return config;
  }
  /* The middle level is the caller's exact mixer damping. The controller may
   * retreat below it and later recover to it, but must not silently reinterpret
   * a valid public damping value such as 0.9 or 1.0. */
  config.damping_levels =
      {{0.5 * baseline_damping, baseline_damping, std::min(1.0, 1.5 * baseline_damping)}};
  return config;
}

bool compute_scc_controller_trust_radius(const SccControllerConfig& config,
                                         double weighted_residual_norm,
                                         double& maximum_step_norm) noexcept {
  maximum_step_norm = 0.0;
  if (!valid_config(config) || !std::isfinite(weighted_residual_norm) ||
      weighted_residual_norm < 0.0) {
    return false;
  }
  if (weighted_residual_norm >
      std::numeric_limits<double>::max() / config.trust_radius_multiplier) {
    maximum_step_norm = std::numeric_limits<double>::max();
  } else {
    maximum_step_norm =
        std::max(config.minimum_trust_radius,
                 config.trust_radius_multiplier * weighted_residual_norm);
  }
  return true;
}

bool advance_scc_controller(const SccControllerConfig& config, const SccControllerState& state,
                            const SccControllerObservation& observation,
                            SccControllerDecision& decision) noexcept {
  decision.next_state = state;
  decision.restart_history = false;
  decision.maximum_weighted_step_norm = 0.0;
  const std::uint8_t level = std::min(state.damping_level, kMaximumDampingLevel);
  decision.damping = config.damping_levels[level];

  if (!valid_config(config) || state.damping_level > kMaximumDampingLevel ||
      !std::isfinite(observation.weighted_residual_norm) ||
      observation.weighted_residual_norm < 0.0 ||
      (observation.has_previous_residual &&
       (!std::isfinite(observation.previous_weighted_residual_norm) ||
        observation.previous_weighted_residual_norm < 0.0)) ||
      (observation.cosine_is_valid && (!std::isfinite(observation.weighted_residual_cosine) ||
                                       observation.weighted_residual_cosine < -1.0 ||
                                       observation.weighted_residual_cosine > 1.0))) {
    return false;
  }

  if (!compute_scc_controller_trust_radius(config, observation.weighted_residual_norm,
                                           decision.maximum_weighted_step_norm)) {
    return false;
  }

  SccControllerState& next = decision.next_state;
  if (next.cooldown_remaining > 0u) {
    --next.cooldown_remaining;
  }
  if (next.fallback_to_baseline || !observation.has_previous_residual ||
      observation.previous_weighted_residual_norm <= config.angle_norm_floor) {
    next.damping_level = next.fallback_to_baseline ? kBaselineDampingLevel : next.damping_level;
    decision.damping = config.damping_levels[next.damping_level];
    return true;
  }

  const double ratio =
      observation.weighted_residual_norm / observation.previous_weighted_residual_norm;
  if (!std::isfinite(ratio)) {
    decision.next_state = state;
    decision.damping = config.damping_levels[level];
    return false;
  }

  const bool angle_is_usable =
      observation.cosine_is_valid && observation.weighted_residual_norm > config.angle_norm_floor &&
      observation.previous_weighted_residual_norm > config.angle_norm_floor;
  const bool diverging = ratio > config.divergence_ratio;
  const bool two_cycle = angle_is_usable && ratio >= config.two_cycle_ratio_minimum &&
                         ratio <= config.two_cycle_ratio_maximum &&
                         observation.weighted_residual_cosine < config.two_cycle_cosine;
  const bool stagnating = angle_is_usable && ratio >= config.stagnation_ratio_minimum &&
                          ratio <= config.stagnation_ratio_maximum &&
                          observation.weighted_residual_cosine > config.stagnation_cosine;
  const bool contracting = ratio < config.contraction_ratio;

  next.two_cycle_count = two_cycle ? increment_saturating(next.two_cycle_count) : 0u;
  next.stagnation_count = stagnating ? increment_saturating(next.stagnation_count) : 0u;
  next.contraction_count = contracting ? increment_saturating(next.contraction_count) : 0u;

  const bool protected_by_cooldown = next.cooldown_remaining > 0u;
  const bool event_requests_restart =
      diverging ||
      (!protected_by_cooldown && (next.two_cycle_count >= config.two_cycle_window ||
                                  next.stagnation_count >= config.stagnation_window ||
                                  next.contraction_count >= config.contraction_window));
  if (!event_requests_restart) {
    decision.damping = config.damping_levels[next.damping_level];
    return true;
  }

  if (next.restart_count >= config.maximum_restarts) {
    next.damping_level = kBaselineDampingLevel;
    next.fallback_to_baseline = true;
    next.cooldown_remaining = 0u;
    reset_event_counters(next);
    /* The event that exhausts the controller budget still identified an
     * unusable secant history. Clear it once even when damping was already at
     * the baseline level, then keep the fixed baseline policy thereafter. */
    decision.restart_history = true;
    decision.damping = config.damping_levels[kBaselineDampingLevel];
    return true;
  }

  std::uint8_t requested_level = next.damping_level;
  if (!diverging && next.contraction_count >= config.contraction_window) {
    /* Strong contraction is evidence that a prior conservative response can
     * recover toward the user's baseline. It is not sufficient evidence for
     * exceeding that baseline: doing so paid an avoidable history restart on
     * otherwise well-behaved short trajectories. */
    requested_level = std::min<std::uint8_t>(kBaselineDampingLevel,
                                             static_cast<std::uint8_t>(requested_level + 1u));
  } else if (requested_level > 0u) {
    --requested_level;
  }

  if (requested_level != next.damping_level || diverging ||
      next.two_cycle_count >= config.two_cycle_window ||
      next.stagnation_count >= config.stagnation_window) {
    request_restart(config, next, requested_level, decision);
  }
  decision.damping = config.damping_levels[next.damping_level];
  return true;
}

}  // namespace xtbloom::detail::common
