#ifndef GPUXTB_MODEL_GFN2_SCC_MIXER_HPP
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_MODEL_GFN2_SCC_MIXER_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gpuxtb/gpuxtb.h"
#include "model/gfn2/wavefunction.hpp"

namespace gpuxtb::detail::gfn2 {

inline constexpr std::size_t kSccMixerWorkspaceAlignment = 64u;

struct SccMixerPlanData;

/*
 * Immutable topology and numerical policy for the GFN2 SCC mixer.
 *
 * A system's mixed vector is the tblite-compatible concatenation
 *
 *   qsh, dipole, quadrupole,
 *
 * with each field retaining WavefunctionLayout's channel-major packing.
 * Construction may allocate metadata; all state transitions are performed in
 * caller-owned storage and allocate nothing on their successful hot paths.
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
  /* Exact field binding check used by higher-level SCC composers. */
  [[nodiscard]] bool matches_wavefunction_layout(const WavefunctionLayout& layout) const noexcept;
  /* True when a byte range aliases this plan's immutable object or backing storage. */
  [[nodiscard]] bool overlaps_storage(const void* data, std::size_t size_bytes) const noexcept;
  [[nodiscard]] const SccMixerPlanData* identity() const noexcept;

 private:
  explicit SccMixerPlan(std::shared_ptr<const SccMixerPlanData> data) noexcept;
  std::shared_ptr<const SccMixerPlanData> data_;

  friend gpuxtb_status_t make_scc_mixer_plan(const WavefunctionLayout& layout,
                                             std::int64_t history_size, double damping,
                                             double rms_tolerance, double maximum_tolerance,
                                             SccMixerPlan& plan, std::string& error);
};

/*
 * Persistent caller-owned history for every member of a ragged batch.
 *
 * History vectors are system-major, then circular-slot-major. Public pointers
 * are intended for driver diagnostics and checkpointing; numerical entry
 * points validate that the descriptor is the exact binding produced for its
 * plan. Concurrent workers may update different systems, never the same one.
 */
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
  std::uint64_t* restart_counts = nullptr;
  gpuxtb_status_t* system_statuses = nullptr;
  std::uint8_t* initialized = nullptr;
  std::uint8_t* converged = nullptr;

  const SccMixerPlanData* plan_identity = nullptr;
};

/*
 * Compact scratch shared by the serial batch entry point or owned by one
 * parallel worker. Its size depends only on the largest system and history
 * depth, not on the number of systems. All tentative numerical results live
 * here until a complete per-system update has been verified finite.
 */
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

/*
 * Build the finite-memory Johnson modified-Broyden policy used by tblite/xTB.
 * history_size must be positive. Damping and both convergence tolerances must
 * be finite and positive; damping is restricted to (0, 1].
 */
gpuxtb_status_t make_scc_mixer_plan(const WavefunctionLayout& layout, std::int64_t history_size,
                                    double damping, double rms_tolerance, double maximum_tolerance,
                                    SccMixerPlan& plan, std::string& error);

/* Bind and clear persistent state. The state is not usable until initialized. */
gpuxtb_status_t bind_scc_mixer_state(const SccMixerPlan& plan, void* workspace,
                                     std::size_t workspace_size, SccMixerState& state,
                                     std::string& error);

/* Bind compact allocation-free scratch usable by batch or one-system calls. */
gpuxtb_status_t bind_scc_mixer_workspace(const SccMixerPlan& plan, void* workspace,
                                         std::size_t workspace_size, SccMixerWorkspace& view,
                                         std::string& error);

/*
 * Capture the initial input multipoles for all systems and clear every
 * history, iteration, convergence, restart, and failure record atomically.
 */
gpuxtb_status_t initialize_scc_mixer_state_cpu(const SccMixerPlan& plan,
                                               const WavefunctionView& wavefunction,
                                               const SccMixerState& state, std::string& error);

/*
 * Independently restart one system from the supplied wavefunction values.
 * Other systems and their history are untouched. restart_counts[system] is
 * incremented only after all new values have been validated.
 */
gpuxtb_status_t restart_scc_mixer_system_cpu(const SccMixerPlan& plan, std::int64_t system,
                                             const WavefunctionView& wavefunction,
                                             const SccMixerState& state, std::string& error);

/*
 * Mix one raw SCC output in place.
 *
 * state.current_inputs contains the multipoles that produced this raw output.
 * On success, wavefunction receives the next mixed input and the corresponding
 * history/status is committed. A nonfinite residual or failed Broyden solve
 * leaves both the raw wavefunction values and all numerical history unchanged;
 * only system_statuses[system] records GPUXTB_STATUS_INTERNAL_ERROR.
 *
 * Different systems may be processed concurrently with distinct workspaces
 * and distinct std::string objects. Calling workers concurrently for the same
 * system is unsupported.
 */
gpuxtb_status_t mix_scc_broyden_system_cpu(const SccMixerPlan& plan, std::int64_t system,
                                           const WavefunctionView& wavefunction,
                                           const SccMixerState& state,
                                           const SccMixerWorkspace& workspace, std::string& error);

/*
 * Serial ragged-batch wrapper over the one-system primitive. Structural errors
 * are rejected before publication. Numerical failures remain per-system: all
 * other members are still advanced, and the call returns the first numerical
 * failure after every member has been attempted.
 */
gpuxtb_status_t mix_scc_broyden_batch_cpu(const SccMixerPlan& plan,
                                          const WavefunctionView& wavefunction,
                                          const SccMixerState& state,
                                          const SccMixerWorkspace& workspace, std::string& error);

/*
 * Stage exactly one system's mixer history and records into a separate
 * full-layout binding, leaving every other system untouched. Both states must
 * be exact bindings of the same plan and their storages must not overlap.
 *
 * The caller owns the transaction: it may run mix_scc_broyden_system_cpu (or
 * restart-style edits) against the staged binding and then either publish it
 * with commit_scc_mixer_system_transaction_cpu or discard it. Discarding a
 * staged system makes the public history byte-identical to before the prepare,
 * so higher-level drivers can isolate one failing system without copying the
 * full batch history. Concurrent workers must never prepare the same system
 * into the same staged binding.
 */
gpuxtb_status_t prepare_scc_mixer_system_transaction_cpu(const SccMixerPlan& plan,
                                                         std::int64_t system,
                                                         const SccMixerState& source,
                                                         const SccMixerState& staged,
                                                         std::string& error);

/*
 * Publish one system's staged mixer history and records back to its public
 * binding, leaving every other system untouched. This is the mirror of
 * prepare_scc_mixer_system_transaction_cpu; committing a system that was never
 * prepared is allowed and copies whatever the staged binding currently holds.
 * Neither function allocates on its successful path.
 */
gpuxtb_status_t commit_scc_mixer_system_transaction_cpu(const SccMixerPlan& plan,
                                                        std::int64_t system,
                                                        const SccMixerState& staged,
                                                        const SccMixerState& destination,
                                                        std::string& error);

}  // namespace gpuxtb::detail::gfn2

#endif  // GPUXTB_MODEL_GFN2_SCC_MIXER_HPP
