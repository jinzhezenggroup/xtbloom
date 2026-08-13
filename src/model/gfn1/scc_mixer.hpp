#ifndef XTBLOOM_MODEL_GFN1_SCC_MIXER_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN1_SCC_MIXER_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model/common/scc_mixer.hpp"
#include "model/gfn1/wavefunction.hpp"

namespace xtbloom::detail::gfn1 {

inline constexpr std::size_t kSccMixerWorkspaceAlignment =
    common::kSccMixerWorkspaceAlignment;

using SccMixerPlanData = common::SccMixerPlanData;
using SccMixerState = common::SccMixerState;
using SccMixerWorkspace = common::SccMixerWorkspace;

/*
 * GFN1 mixes only shell populations. For unrestricted systems qsh already
 * contains charge followed by magnetization through its nspin-expanded
 * ragged field offsets; atomic dipoles and quadrupoles never enter this plan.
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
  [[nodiscard]] bool matches_wavefunction_layout(const WavefunctionLayout& layout) const noexcept;
  [[nodiscard]] bool overlaps_storage(const void* data, std::size_t size_bytes) const noexcept;
  [[nodiscard]] const SccMixerPlanData* identity() const noexcept;

 private:
  common::SccMixerPlan engine_;

  friend xtbloom_status_t make_scc_mixer_plan(const WavefunctionLayout& layout,
                                              std::int64_t history_size, double damping,
                                              double rms_tolerance,
                                              double maximum_tolerance, SccMixerPlan& plan,
                                              std::string& error);
  friend const common::SccMixerPlan& common_plan(const SccMixerPlan& plan) noexcept;
};

xtbloom_status_t make_scc_mixer_plan(const WavefunctionLayout& layout,
                                     std::int64_t history_size, double damping,
                                     double rms_tolerance, double maximum_tolerance,
                                     SccMixerPlan& plan, std::string& error);

xtbloom_status_t bind_scc_mixer_state(const SccMixerPlan& plan, void* workspace,
                                      std::size_t workspace_size, SccMixerState& state,
                                      std::string& error);

xtbloom_status_t bind_scc_mixer_workspace(const SccMixerPlan& plan, void* workspace,
                                          std::size_t workspace_size,
                                          SccMixerWorkspace& view, std::string& error);

xtbloom_status_t initialize_scc_mixer_state_cpu(const SccMixerPlan& plan,
                                                const WavefunctionView& wavefunction,
                                                const SccMixerState& state,
                                                std::string& error);

xtbloom_status_t restart_scc_mixer_system_cpu(const SccMixerPlan& plan,
                                              std::int64_t system,
                                              const WavefunctionView& wavefunction,
                                              const SccMixerState& state,
                                              std::string& error);

xtbloom_status_t mix_scc_broyden_system_cpu(const SccMixerPlan& plan,
                                            std::int64_t system,
                                            const WavefunctionView& wavefunction,
                                            const SccMixerState& state,
                                            const SccMixerWorkspace& workspace,
                                            std::string& error);

xtbloom_status_t mix_scc_broyden_batch_cpu(const SccMixerPlan& plan,
                                           const WavefunctionView& wavefunction,
                                           const SccMixerState& state,
                                           const SccMixerWorkspace& workspace,
                                           std::string& error);

xtbloom_status_t prepare_scc_mixer_system_transaction_cpu(
    const SccMixerPlan& plan, std::int64_t system, const SccMixerState& source,
    const SccMixerState& staged, std::string& error);

xtbloom_status_t commit_scc_mixer_system_transaction_cpu(
    const SccMixerPlan& plan, std::int64_t system, const SccMixerState& staged,
    const SccMixerState& destination, std::string& error);

}  // namespace xtbloom::detail::gfn1

#endif  // XTBLOOM_MODEL_GFN1_SCC_MIXER_HPP
