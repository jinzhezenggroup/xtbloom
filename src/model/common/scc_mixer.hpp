#ifndef XTBLOOM_MODEL_COMMON_SCC_MIXER_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_COMMON_SCC_MIXER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::common {

inline constexpr std::size_t kSccMixerWorkspaceAlignment = 64u;
inline constexpr std::size_t kSccMixerMaximumFields = 4u;

/*
 * One field in a model-owned ragged SCC vector. The common mixer copies the
 * offsets into its immutable plan, so these borrowed construction views need
 * to remain valid only for make_scc_mixer_plan.
 */
struct SccMixerFieldLayoutView {
  std::size_t offset_bytes = 0u;
  std::size_t size_bytes = 0u;
  std::int64_t element_count = 0;
  const std::int64_t* system_offsets = nullptr;
  std::size_t system_offset_count = 0u;
};

/*
 * Model-neutral description of fields concatenated into one ragged mixer
 * vector. Field order is significant and is retained exactly. A GFN1 plan
 * supplies only qsh; the GFN2 wrapper supplies qsh, dipole, and quadrupole.
 */
struct SccMixerVectorLayoutView {
  std::int64_t batch_size = 0;
  std::size_t workspace_size_bytes = 0u;
  std::size_t workspace_alignment = 0u;
  std::array<SccMixerFieldLayoutView, kSccMixerMaximumFields> fields{};
  std::size_t field_count = 0u;
};

/* Exact mutable binding corresponding to SccMixerVectorLayoutView. */
struct SccMixerVectorView {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0u;
  std::array<double*, kSccMixerMaximumFields> fields{};
  std::size_t field_count = 0u;
};

struct SccMixerPlanData;

/*
 * Immutable topology and Johnson modified-Broyden numerical policy.
 *
 * Construction may allocate metadata. All numerical transitions use only
 * caller-owned state and scratch, and successful steady-state calls allocate
 * nothing. The plan knows field packing but no model-specific multipoles.
 */
class SccMixerPlan {
 public:
  SccMixerPlan() noexcept = default;
  SccMixerPlan(const SccMixerPlan&) noexcept = default;
  SccMixerPlan(SccMixerPlan&&) noexcept = default;
  SccMixerPlan& operator=(const SccMixerPlan&) noexcept = default;
  SccMixerPlan& operator=(SccMixerPlan&&) noexcept = default;
  ~SccMixerPlan() = default;

  [[nodiscard]] bool sealed() const noexcept;
  [[nodiscard]] std::int64_t batch_size() const noexcept;
  [[nodiscard]] std::int64_t history_size() const noexcept;
  [[nodiscard]] std::int64_t total_vector_elements() const noexcept;
  [[nodiscard]] std::int64_t maximum_vector_elements() const noexcept;
  [[nodiscard]] double damping() const noexcept;
  [[nodiscard]] double rms_tolerance() const noexcept;
  [[nodiscard]] double maximum_tolerance() const noexcept;
  [[nodiscard]] std::size_t state_size_bytes() const noexcept;
  [[nodiscard]] std::size_t workspace_size_bytes() const noexcept;
  [[nodiscard]] std::size_t resident_bytes() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& vector_offsets() const noexcept;
  [[nodiscard]] bool matches_vector_layout(const SccMixerVectorLayoutView& layout) const noexcept;
  [[nodiscard]] bool overlaps_storage(const void* data, std::size_t size_bytes) const noexcept;
  [[nodiscard]] const SccMixerPlanData* identity() const noexcept;

 protected:
  explicit SccMixerPlan(std::shared_ptr<const SccMixerPlanData> data) noexcept;

 private:
  std::shared_ptr<const SccMixerPlanData> data_;

  friend xtbloom_status_t make_scc_mixer_plan(const SccMixerVectorLayoutView& layout,
                                              std::int64_t history_size, double damping,
                                              double rms_tolerance, double maximum_tolerance,
                                              SccMixerPlan& plan, std::string& error);
};

/* Persistent system-major history and convergence diagnostics. */
struct SccMixerState {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0u;

  double* current_inputs = nullptr;
  double* previous_inputs = nullptr;
  double* previous_residuals = nullptr;
  double* df_history = nullptr;
  double* u_history = nullptr;
  double* omega = nullptr;

  double* residual_rms = nullptr;
  double* residual_maximum = nullptr;
  std::uint64_t* iterations = nullptr;
  /* Number of successful transitions represented by the current secant
   * history. Unlike iterations, this counter may restart when a numerical
   * controller changes damping. */
  std::uint64_t* history_ages = nullptr;
  std::uint64_t* restart_counts = nullptr;
  xtbloom_status_t* system_statuses = nullptr;
  std::uint8_t* initialized = nullptr;
  std::uint8_t* converged = nullptr;

  const SccMixerPlanData* plan_identity = nullptr;
};

/*
 * Experimental internal seam for a model-provided effective residual.
 * Public convergence diagnostics remain the raw, unmodified values supplied
 * here. The effective residual must occupy the mixer's residual scratch and
 * may be preconditioned, but it must represent the same physical fixed point.
 */
struct SccMixerPreparedStepView {
  const double* effective_residual = nullptr;
  std::size_t residual_elements = 0u;
  double raw_residual_rms = 0.0;
  double raw_residual_maximum = 0.0;
  double runtime_damping = 0.0;
  bool restart_history = false;
  /* Positive per-component weights for the model's invariant step metric.
   * The complete damped-plus-Broyden candidate is scaled uniformly when its
   * weighted norm exceeds maximum_weighted_step_norm. */
  const double* step_metric_weights = nullptr;
  double maximum_weighted_step_norm = 0.0;
};

/* Compact scratch reusable by one worker or the serial batch wrapper. */
struct SccMixerWorkspace {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0u;

  double* residual = nullptr;
  double* mixed = nullptr;
  double* delta_f = nullptr;
  double* new_u = nullptr;
  double* beta = nullptr;
  double* coefficients = nullptr;
  std::int64_t* history_slots = nullptr;

  const SccMixerPlanData* plan_identity = nullptr;
};

xtbloom_status_t make_scc_mixer_plan(const SccMixerVectorLayoutView& layout,
                                     std::int64_t history_size, double damping,
                                     double rms_tolerance, double maximum_tolerance,
                                     SccMixerPlan& plan, std::string& error);

xtbloom_status_t bind_scc_mixer_state(const SccMixerPlan& plan, void* workspace,
                                      std::size_t workspace_size, SccMixerState& state,
                                      std::string& error);

xtbloom_status_t bind_scc_mixer_workspace(const SccMixerPlan& plan, void* workspace,
                                          std::size_t workspace_size, SccMixerWorkspace& view,
                                          std::string& error);

/* Read-only canonical-binding checks for higher-level allocation-free drivers. */
xtbloom_status_t validate_scc_mixer_state_binding(const SccMixerPlan& plan,
                                                  const SccMixerState& state, std::string& error);
xtbloom_status_t validate_scc_mixer_workspace_binding(const SccMixerPlan& plan,
                                                      const SccMixerWorkspace& workspace,
                                                      std::string& error);

/* Initialization is all-or-nothing across the complete ragged batch. */
xtbloom_status_t initialize_scc_mixer_state_cpu(const SccMixerPlan& plan,
                                                const SccMixerVectorView& vector,
                                                const SccMixerState& state, std::string& error);

/* Restart clears only the selected system after its new vector is validated. */
xtbloom_status_t restart_scc_mixer_system_cpu(const SccMixerPlan& plan, std::int64_t system,
                                              const SccMixerVectorView& vector,
                                              const SccMixerState& state, std::string& error);

/*
 * Mix one raw vector in place. Numerical failure changes only the selected
 * system status; raw values and all persistent numerical history stay intact.
 */
xtbloom_status_t mix_scc_broyden_system_cpu(const SccMixerPlan& plan, std::int64_t system,
                                            const SccMixerVectorView& vector,
                                            const SccMixerState& state,
                                            const SccMixerWorkspace& workspace, std::string& error);

xtbloom_status_t mix_scc_broyden_system_cpu_prepared(const SccMixerPlan& plan, std::int64_t system,
                                                     const SccMixerVectorView& vector,
                                                     const SccMixerState& state,
                                                     const SccMixerWorkspace& workspace,
                                                     const SccMixerPreparedStepView& prepared,
                                                     std::string& error);

/* Serial wrapper retaining peer-local numerical failure isolation. */
xtbloom_status_t mix_scc_broyden_batch_cpu(const SccMixerPlan& plan,
                                           const SccMixerVectorView& vector,
                                           const SccMixerState& state,
                                           const SccMixerWorkspace& workspace, std::string& error);

/* Copy exactly one system into or out of a disjoint full-layout binding. */
xtbloom_status_t prepare_scc_mixer_system_transaction_cpu(const SccMixerPlan& plan,
                                                          std::int64_t system,
                                                          const SccMixerState& source,
                                                          const SccMixerState& staged,
                                                          std::string& error);

xtbloom_status_t commit_scc_mixer_system_transaction_cpu(const SccMixerPlan& plan,
                                                         std::int64_t system,
                                                         const SccMixerState& staged,
                                                         const SccMixerState& destination,
                                                         std::string& error);

}  // namespace xtbloom::detail::common

#endif  // XTBLOOM_MODEL_COMMON_SCC_MIXER_HPP
