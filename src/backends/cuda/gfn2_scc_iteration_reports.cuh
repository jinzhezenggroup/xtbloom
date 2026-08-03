#ifndef GPUXTB_BACKENDS_CUDA_GFN2_SCC_ITERATION_REPORTS_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_SCC_ITERATION_REPORTS_CUH

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_scc_iteration.cuh"

namespace gpuxtb::detail::cuda {

/*
 * Canonical arena projection for stage-local diagnostics not already owned by
 * a primitive descriptor. Slots follow the generated report order. Some slots
 * are intentionally unused because D4, mixer, publication, and primitive
 * sequence latches have stricter canonical owners; retaining one slot per
 * report keeps the arena layout deterministic for every enabled-component set.
 */
struct Gfn2SccIterationReportStorage {
  std::uint32_t* system_errors = nullptr;
  std::int64_t system_error_elements = 0;
  std::uint32_t* device_errors = nullptr;
  std::int64_t device_error_elements = 0;
  std::uint32_t* sequence_latches = nullptr;
  std::int64_t sequence_latch_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Exact element requirements for a report-storage arena projection. */
struct Gfn2SccIterationReportStorageRequirements {
  std::int64_t report_count = 0;
  std::int64_t system_error_elements = 0;
  std::int64_t device_error_elements = 0;
  std::int64_t sequence_latch_elements = 0;
};

/*
 * Unvalidated output of the deterministic projection pass. This type is
 * deliberately distinct from Gfn2SccIterationBinding: only the validated
 * builder below may publish a launchable binding.
 */
struct Gfn2SccIterationProjectedDescriptors {
  Gfn2SccIterationDevicePlan plan{};
  Gfn2SccIterationDeviceInput input{};
  Gfn2SccIterationDeviceState state{};
  Gfn2SccIterationDeviceWorkspace workspace{};
};

static_assert(std::is_trivially_copyable_v<Gfn2SccIterationReportStorage>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationReportStorage>);
static_assert(std::is_trivially_copyable_v<Gfn2SccIterationReportStorageRequirements>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationReportStorageRequirements>);
static_assert(std::is_trivially_copyable_v<Gfn2SccIterationProjectedDescriptors>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationProjectedDescriptors>);

/*
 * Compute deterministic report-arena element counts without inspecting any
 * pointer or touching CUDA state. Invalid component masks and overflow fail
 * synchronously and leave requirements cleared.
 */
[[nodiscard]] Gfn2SccIterationBindingDiagnostic query_gfn2_scc_iteration_report_storage_cuda(
    std::uint32_t enabled_components, std::int64_t batch_size,
    Gfn2SccIterationReportStorageRequirements& requirements) noexcept;

/*
 * Generate the canonical 19--24 reports and every exact alias derivable from
 * already-bound plan/state/workspace leaves. No allocation, transfer, CUDA API,
 * or arena-offset reconstruction occurs. On failure projected is cleared.
 * The result is intentionally not launchable until the complete #96 validator
 * accepts it through build_gfn2_scc_iteration_report_binding_cuda.
 */
[[nodiscard]] Gfn2SccIterationBindingDiagnostic project_gfn2_scc_iteration_reports_cuda(
    const Gfn2SccIterationReportStorage& report_storage,
    const Gfn2SccIterationDevicePlan& plan_seed, const Gfn2SccIterationDeviceInput& input_seed,
    const Gfn2SccIterationDeviceState& state_seed,
    const Gfn2SccIterationDeviceWorkspace& workspace_seed,
    Gfn2SccIterationProjectedDescriptors& projected) noexcept;

/*
 * Fail-closed production entry: project reports/aliases, run the complete #96
 * setup validator, and publish binding only after every contract succeeds.
 */
[[nodiscard]] Gfn2SccIterationBindingDiagnostic build_gfn2_scc_iteration_report_binding_cuda(
    const Gfn2SccIterationReportStorage& report_storage,
    const Gfn2SccIterationDevicePlan& plan_seed, const Gfn2SccIterationDeviceInput& input_seed,
    const Gfn2SccIterationDeviceState& state_seed,
    const Gfn2SccIterationDeviceWorkspace& workspace_seed,
    Gfn2SccIterationBinding& binding) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_SCC_ITERATION_REPORTS_CUH
