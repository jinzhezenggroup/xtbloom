#include "model/gfn2/scc_mixer.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <array>
#include <utility>

namespace xtbloom::detail::gfn2 {
namespace {

common::SccMixerVectorLayoutView make_common_layout(const WavefunctionLayout& layout) {
  common::SccMixerVectorLayoutView common_layout;
  common_layout.batch_size = layout.batch_size;
  common_layout.workspace_size_bytes = layout.workspace_size_bytes;
  common_layout.workspace_alignment = kWavefunctionWorkspaceAlignment;
  common_layout.field_count = 3u;
  const std::array<const WavefunctionFieldLayout*, 3> fields{
      {&layout.qsh, &layout.dipole, &layout.quadrupole}};
  for (std::size_t field = 0u; field < fields.size(); ++field) {
    common_layout.fields[field] = {
        fields[field]->offset_bytes, fields[field]->size_bytes, fields[field]->element_count,
        fields[field]->system_offsets.data(), fields[field]->system_offsets.size()};
  }
  return common_layout;
}

common::SccMixerVectorView make_common_view(const WavefunctionView& wavefunction) {
  common::SccMixerVectorView vector;
  vector.workspace_base = wavefunction.workspace_base;
  vector.workspace_size_bytes = wavefunction.workspace_size_bytes;
  vector.fields = {{wavefunction.qsh, wavefunction.dipole, wavefunction.quadrupole, nullptr}};
  vector.field_count = 3u;
  return vector;
}

}  // namespace

const common::SccMixerPlan& common_plan(const SccMixerPlan& plan) noexcept { return plan.engine_; }

bool SccMixerPlan::sealed() const noexcept { return engine_.sealed(); }
std::int64_t SccMixerPlan::batch_size() const noexcept { return engine_.batch_size(); }
std::int64_t SccMixerPlan::history_size() const noexcept { return engine_.history_size(); }
std::int64_t SccMixerPlan::total_vector_elements() const noexcept {
  return engine_.total_vector_elements();
}
std::int64_t SccMixerPlan::maximum_vector_elements() const noexcept {
  return engine_.maximum_vector_elements();
}
double SccMixerPlan::damping() const noexcept { return engine_.damping(); }
double SccMixerPlan::rms_tolerance() const noexcept { return engine_.rms_tolerance(); }
double SccMixerPlan::maximum_tolerance() const noexcept { return engine_.maximum_tolerance(); }
std::size_t SccMixerPlan::state_size_bytes() const noexcept { return engine_.state_size_bytes(); }
std::size_t SccMixerPlan::workspace_size_bytes() const noexcept {
  return engine_.workspace_size_bytes();
}
std::size_t SccMixerPlan::resident_bytes() const noexcept { return engine_.resident_bytes(); }
const std::vector<std::int64_t>& SccMixerPlan::vector_offsets() const noexcept {
  return engine_.vector_offsets();
}
bool SccMixerPlan::matches_wavefunction_layout(const WavefunctionLayout& layout) const noexcept {
  return engine_.matches_vector_layout(make_common_layout(layout));
}
bool SccMixerPlan::overlaps_storage(const void* data, std::size_t size_bytes) const noexcept {
  return engine_.overlaps_storage(data, size_bytes);
}
const SccMixerPlanData* SccMixerPlan::identity() const noexcept { return engine_.identity(); }

xtbloom_status_t make_scc_mixer_plan(const WavefunctionLayout& layout, std::int64_t history_size,
                                     double damping, double rms_tolerance, double maximum_tolerance,
                                     SccMixerPlan& plan, std::string& error) {
  WavefunctionWarmStartIdentity validated_layout;
  xtbloom_status_t status =
      make_wavefunction_warm_start_identity(layout, 0u, validated_layout, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  common::SccMixerPlan created;
  status = common::make_scc_mixer_plan(make_common_layout(layout), history_size, damping,
                                       rms_tolerance, maximum_tolerance, created, error);
  if (status == XTBLOOM_STATUS_SUCCESS) {
    plan.engine_ = std::move(created);
  }
  return status;
}

xtbloom_status_t bind_scc_mixer_state(const SccMixerPlan& plan, void* workspace,
                                      std::size_t workspace_size, SccMixerState& state,
                                      std::string& error) {
  return common::bind_scc_mixer_state(common_plan(plan), workspace, workspace_size, state, error);
}

xtbloom_status_t bind_scc_mixer_workspace(const SccMixerPlan& plan, void* workspace,
                                          std::size_t workspace_size, SccMixerWorkspace& view,
                                          std::string& error) {
  return common::bind_scc_mixer_workspace(common_plan(plan), workspace, workspace_size, view,
                                          error);
}

xtbloom_status_t initialize_scc_mixer_state_cpu(const SccMixerPlan& plan,
                                                const WavefunctionView& wavefunction,
                                                const SccMixerState& state, std::string& error) {
  return common::initialize_scc_mixer_state_cpu(common_plan(plan), make_common_view(wavefunction),
                                                state, error);
}

xtbloom_status_t restart_scc_mixer_system_cpu(const SccMixerPlan& plan, std::int64_t system,
                                              const WavefunctionView& wavefunction,
                                              const SccMixerState& state, std::string& error) {
  return common::restart_scc_mixer_system_cpu(common_plan(plan), system,
                                              make_common_view(wavefunction), state, error);
}

xtbloom_status_t mix_scc_broyden_system_cpu(const SccMixerPlan& plan, std::int64_t system,
                                            const WavefunctionView& wavefunction,
                                            const SccMixerState& state,
                                            const SccMixerWorkspace& workspace,
                                            std::string& error) {
  return common::mix_scc_broyden_system_cpu(
      common_plan(plan), system, make_common_view(wavefunction), state, workspace, error);
}

xtbloom_status_t mix_scc_broyden_system_cpu_prepared(const SccMixerPlan& plan, std::int64_t system,
                                                     const WavefunctionView& wavefunction,
                                                     const SccMixerState& state,
                                                     const SccMixerWorkspace& workspace,
                                                     const SccMixerPreparedStepView& prepared,
                                                     std::string& error) {
  return common::mix_scc_broyden_system_cpu_prepared(
      common_plan(plan), system, make_common_view(wavefunction), state, workspace, prepared, error);
}

xtbloom_status_t mix_scc_broyden_batch_cpu(const SccMixerPlan& plan,
                                           const WavefunctionView& wavefunction,
                                           const SccMixerState& state,
                                           const SccMixerWorkspace& workspace, std::string& error) {
  return common::mix_scc_broyden_batch_cpu(common_plan(plan), make_common_view(wavefunction), state,
                                           workspace, error);
}

xtbloom_status_t prepare_scc_mixer_system_transaction_cpu(const SccMixerPlan& plan,
                                                          std::int64_t system,
                                                          const SccMixerState& source,
                                                          const SccMixerState& staged,
                                                          std::string& error) {
  return common::prepare_scc_mixer_system_transaction_cpu(common_plan(plan), system, source, staged,
                                                          error);
}

xtbloom_status_t commit_scc_mixer_system_transaction_cpu(const SccMixerPlan& plan,
                                                         std::int64_t system,
                                                         const SccMixerState& staged,
                                                         const SccMixerState& destination,
                                                         std::string& error) {
  return common::commit_scc_mixer_system_transaction_cpu(common_plan(plan), system, staged,
                                                         destination, error);
}

}  // namespace xtbloom::detail::gfn2
