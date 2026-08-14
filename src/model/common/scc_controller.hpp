#ifndef XTBLOOM_MODEL_COMMON_SCC_CONTROLLER_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_COMMON_SCC_CONTROLLER_HPP

#include <array>
#include <cstdint>

namespace xtbloom::detail::common {

/*
 * Finite-state policy for adapting a Broyden startup damping without changing
 * the residual that certifies SCC convergence. The caller owns the weighted
 * residual metric and supplies only invariant scalar observations here.
 *
 * Damping changes always request a history restart because the Johnson
 * modified-Broyden u vectors explicitly contain the damping used to create
 * them. Keeping this rule in the controller API prevents callers from
 * accidentally combining secant data from incompatible numerical policies.
 */
struct SccControllerConfig {
  std::array<double, 3u> damping_levels{{0.2, 0.4, 0.6}};
  double contraction_ratio = 0.70;
  double divergence_ratio = 1.50;
  double stagnation_ratio_minimum = 0.95;
  double stagnation_ratio_maximum = 1.05;
  double stagnation_cosine = 0.50;
  double two_cycle_ratio_minimum = 0.80;
  double two_cycle_ratio_maximum = 1.25;
  double two_cycle_cosine = -0.70;
  double angle_norm_floor = 1.0e-14;
  /* Bound the complete candidate step, including the Broyden history
   * correction, in the caller-provided weighted residual metric. */
  double trust_radius_multiplier = 2.0;
  double minimum_trust_radius = 1.0e-12;
  std::uint8_t contraction_window = 2u;
  std::uint8_t stagnation_window = 3u;
  std::uint8_t two_cycle_window = 2u;
  std::uint8_t restart_cooldown = 2u;
  std::uint32_t maximum_restarts = 6u;
};

struct SccControllerState {
  std::uint8_t damping_level = 1u;
  std::uint8_t contraction_count = 0u;
  std::uint8_t stagnation_count = 0u;
  std::uint8_t two_cycle_count = 0u;
  std::uint8_t cooldown_remaining = 0u;
  std::uint32_t restart_count = 0u;
  bool fallback_to_baseline = false;
};

struct SccControllerObservation {
  double weighted_residual_norm = 0.0;
  double previous_weighted_residual_norm = 0.0;
  double weighted_residual_cosine = 0.0;
  bool has_previous_residual = false;
  bool cosine_is_valid = false;
};

struct SccControllerDecision {
  SccControllerState next_state{};
  double damping = 0.0;
  double maximum_weighted_step_norm = 0.0;
  bool restart_history = false;
};

/* Build bounded damping levels around the user's original mixer damping. */
[[nodiscard]] SccControllerConfig make_scc_controller_config(double baseline_damping) noexcept;

/* Compute the bounded step norm for one residual without advancing controller
 * state. This is used when a safeguard switches residual maps after making a
 * restart decision and therefore needs a radius for the replacement map. */
[[nodiscard]] bool compute_scc_controller_trust_radius(const SccControllerConfig& config,
                                                       double weighted_residual_norm,
                                                       double& maximum_step_norm) noexcept;

/*
 * Advance one candidate copy of the state. Invalid/non-finite observations
 * return false and leave the returned state exactly equal to the input state.
 */
[[nodiscard]] bool advance_scc_controller(const SccControllerConfig& config,
                                          const SccControllerState& state,
                                          const SccControllerObservation& observation,
                                          SccControllerDecision& decision) noexcept;

}  // namespace xtbloom::detail::common

#endif  // XTBLOOM_MODEL_COMMON_SCC_CONTROLLER_HPP
