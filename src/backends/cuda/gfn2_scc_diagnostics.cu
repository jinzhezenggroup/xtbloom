#include "backends/cuda/gfn2_scc_diagnostics.cuh"

namespace xtbloom::detail::cuda {
namespace {

/* These observations share the existing CUDA execution/linking boundary.
 * xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION. */

__device__ std::uint64_t global_time_ns() {
  std::uint64_t timestamp = 0u;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp));
  return timestamp;
}

__global__ void reset_diagnostics_kernel(Gfn2SccDiagnosticsDevice diagnostics,
                                         std::uint32_t execution_mode) {
  *diagnostics.header = {};
  diagnostics.header->execution_mode = execution_mode;
}

__global__ void begin_diagnostics_kernel(Gfn2SccDiagnosticsDevice diagnostics,
                                         Gfn2SccIterationDeviceLedger ledger,
                                         const std::int32_t* bucket_systems,
                                         const std::int32_t* spin_channels,
                                         bool record_inactive_submissions) {
  auto& header = *diagnostics.header;
  header.current_iteration = kGfn2SccDiagnosticUnknown;
  bool active = false;
  const bool sequence_open = *ledger.sequence_active == 1u && *ledger.plan_failure_record == 0u;
  if (sequence_open) {
    for (std::int64_t system = 0; system < ledger.batch_elements; ++system) {
      active = active || ledger.active_mask[system] == 1u;
    }
  }
  /* Bounded fallback still queues fixed-capacity provider bodies after all
   * systems terminate. Do not mistake inactive physical work for no submission. */
  if (!active && !record_inactive_submissions) return;
  if (header.iteration_count >= diagnostics.maximum_iterations) {
    header.overflow = 1u;
    return;
  }
  const std::uint64_t iteration = header.iteration_count++;
  header.current_iteration = iteration;
  diagnostics.iterations[iteration] = {};
  for (std::int64_t bucket_index = 0; bucket_index < diagnostics.bucket_count; ++bucket_index) {
    const auto bucket = diagnostics.buckets[bucket_index];
    auto& row = diagnostics.rows[iteration * diagnostics.bucket_count + bucket_index];
    row = {};
    for (std::int32_t local = 0; local < bucket.system_count; ++local) {
      const auto system = bucket_systems[bucket.system_index_offset + local];
      if (sequence_open && ledger.active_mask[system] == 1u) {
        ++row.active_systems;
        row.active_channels += spin_channels == nullptr ? 1u : spin_channels[system];
      }
    }
  }
  diagnostics.iterations[iteration].start_ns = global_time_ns();
}

__global__ void finish_diagnostics_kernel(Gfn2SccDiagnosticsDevice diagnostics,
                                          Gfn2SccIterationDeviceLedger ledger,
                                          Gfn2SccIterationDeviceStateInput state,
                                          const std::int32_t* bucket_systems,
                                          const Gfn2EigensolverBucketActivity* activity,
                                          bool exact_capacity) {
  const auto iteration = diagnostics.header->current_iteration;
  if (iteration == kGfn2SccDiagnosticUnknown) return;
  auto& record = diagnostics.iterations[iteration];
  record.end_ns = global_time_ns();
  record.plan_failure_record = *ledger.plan_failure_record;
  for (std::int64_t bucket_index = 0; bucket_index < diagnostics.bucket_count; ++bucket_index) {
    const auto bucket = diagnostics.buckets[bucket_index];
    auto& row = diagnostics.rows[iteration * diagnostics.bucket_count + bucket_index];
    /* A plan failure may interrupt the chain before telemetry is refreshed.
     * Unknown is safer than publishing a previous iteration's slot counts. */
    if (record.plan_failure_record == 0u) {
      if (exact_capacity) {
        row.submitted_solver_slots = activity[bucket_index].submitted_eigensolver_count;
        row.submitted_backtransform_slots = activity[bucket_index].submitted_backtransform_count;
      } else {
        const auto capacity = bucket.solve_count > 0 ? bucket.solve_count : bucket.system_count;
        row.submitted_solver_slots = static_cast<std::uint64_t>(capacity);
        row.submitted_backtransform_slots = static_cast<std::uint64_t>(capacity);
      }
    }
    for (std::int32_t local = 0; local < bucket.system_count; ++local) {
      const auto system = bucket_systems[bucket.system_index_offset + local];
      const auto status = state.system_statuses[system];
      if (status == XTBLOOM_STATUS_SCC_NOT_CONVERGED ||
          (status == XTBLOOM_STATUS_SUCCESS && state.converged[system] == 0u &&
           state.iterations[system] >= diagnostics.maximum_iterations)) {
        ++row.exhausted_systems;
      } else if (status != XTBLOOM_STATUS_SUCCESS) {
        ++row.failed_systems;
      } else if (state.converged[system] != 0u) {
        ++row.converged_systems;
      }
    }
  }
}

}  // namespace

cudaError_t reset_gfn2_scc_diagnostics_cuda(Gfn2SccDiagnosticsDevice diagnostics,
                                            cudaStream_t stream,
                                            std::uint32_t execution_mode) noexcept {
  reset_diagnostics_kernel<<<1, 1, 0, stream>>>(diagnostics, execution_mode);
  return cudaPeekAtLastError();
}

cudaError_t begin_gfn2_scc_diagnostics_cuda(Gfn2SccDiagnosticsDevice diagnostics,
                                            const Gfn2SccIterationBinding& binding,
                                            cudaStream_t stream,
                                            bool record_inactive_submissions) noexcept {
  begin_diagnostics_kernel<<<1, 1, 0, stream>>>(
      diagnostics, binding.workspace.ledger, binding.plan.eigensolver_batch.bucket_systems,
      binding.plan.wavefunction_layout.spin_channels, record_inactive_submissions);
  return cudaPeekAtLastError();
}

cudaError_t finish_gfn2_scc_diagnostics_cuda(Gfn2SccDiagnosticsDevice diagnostics,
                                             const Gfn2SccIterationBinding& binding,
                                             bool exact_capacity, cudaStream_t stream) noexcept {
  finish_diagnostics_kernel<<<1, 1, 0, stream>>>(
      diagnostics, binding.workspace.ledger, binding.input.activity_state,
      binding.plan.eigensolver_batch.bucket_systems,
      binding.workspace.eigensolver_workspace.bucket_activity, exact_capacity);
  return cudaPeekAtLastError();
}

}  // namespace xtbloom::detail::cuda
