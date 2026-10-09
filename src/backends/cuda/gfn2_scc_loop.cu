#include <cuda_runtime.h>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <ostream>
#include <vector>

#include "backends/cuda/gfn2_scc_loop.cuh"

namespace xtbloom::detail::cuda {
namespace {

struct alignas(16) Gfn2SccDeviceLoopControl {
  std::uint32_t canonical_active_count = 0u;
  std::uint32_t device_launch_error = 0u;
  std::uint64_t numerical_body_count = 0u;
  std::uint64_t plan_failure_snapshot = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2SccDeviceLoopControl>);
static_assert(std::is_standard_layout_v<Gfn2SccDeviceLoopControl>);

Gfn2SccLoopLaunchResult reject_loop_binding(Gfn2SccIterationBindingError error) noexcept {
  Gfn2SccLoopLaunchResult result{};
  result.iteration.status = Gfn2SccIterationLaunchStatus::kInvalidBinding;
  result.iteration.binding.error = error;
  result.iteration.binding.field = Gfn2SccIterationBindingField::kPlan;
  return result;
}

Gfn2SccLoopLaunchResult validate_loop_plan(const Gfn2SccIterationBinding& binding) noexcept {
  const std::uint64_t submission_bound = binding.plan.activity_policy.maximum_iterations;
  if (binding.plan.abi_version != kGfn2SccIterationAbiVersion) {
    return reject_loop_binding(Gfn2SccIterationBindingError::kInvalidAbiVersion);
  }
  if (binding.plan.plan_token == 0u) {
    return reject_loop_binding(Gfn2SccIterationBindingError::kInvalidPlanToken);
  }
  if (submission_bound == 0u || binding.plan.state_policy.maximum_iterations != submission_bound ||
      binding.plan.publication_plan.maximum_iterations != submission_bound) {
    return reject_loop_binding(Gfn2SccIterationBindingError::kInvalidCount);
  }
  const Gfn2SccIterationBindingDiagnostic diagnostic = validate_gfn2_scc_iteration_binding_cuda(
      binding.plan, binding.input, binding.state, binding.workspace);
  if (diagnostic.error != Gfn2SccIterationBindingError::kSuccess) {
    Gfn2SccLoopLaunchResult result{};
    result.iteration.status = Gfn2SccIterationLaunchStatus::kInvalidBinding;
    result.iteration.binding = diagnostic;
    return result;
  }
  return {};
}

Gfn2SccLoopLaunchResult launch_restricted_scc_loop_impl(
    const Gfn2SccIterationBinding& binding, const Gfn2GeometryEpochConsumerDevice* geometry,
    cudaStream_t stream, Gfn2SccDiagnosticsDevice diagnostics = {}) noexcept {
  Gfn2SccLoopLaunchResult result = validate_loop_plan(binding);
  if (!result.success()) {
    return result;
  }

  const std::uint64_t submission_bound = binding.plan.activity_policy.maximum_iterations;
#ifdef XTBLOOM_CUDA_SCC_DIAGNOSTICS
  if (diagnostics.header != nullptr) {
    const cudaError_t status = reset_gfn2_scc_diagnostics_cuda(diagnostics, stream);
    if (status != cudaSuccess) {
      result.iteration.status = Gfn2SccIterationLaunchStatus::kCudaError;
      result.iteration.cuda_status = status;
      return result;
    }
  }
#endif
  for (std::uint64_t iteration = 0u; iteration < submission_bound; ++iteration) {
#ifdef XTBLOOM_CUDA_SCC_DIAGNOSTICS
    if (diagnostics.header != nullptr) {
      result.iteration = geometry == nullptr
                             ? launch_gfn2_restricted_scc_activity_cuda(binding, stream)
                             : launch_gfn2_restricted_scc_activity_cuda(binding, *geometry, stream);
      cudaError_t status = result.iteration.success()
                               ? begin_gfn2_scc_diagnostics_cuda(diagnostics, binding, stream, true)
                               : cudaSuccess;
      if (result.iteration.success() && status == cudaSuccess) {
        result.iteration =
            geometry == nullptr
                ? launch_gfn2_restricted_scc_numerical_body_cuda(binding, stream)
                : launch_gfn2_restricted_scc_numerical_body_cuda(binding, *geometry, stream);
      }
      if (result.iteration.success() && status == cudaSuccess) {
        status = finish_gfn2_scc_diagnostics_cuda(diagnostics, binding, false, stream);
      }
      if (status != cudaSuccess) {
        result.iteration.status = Gfn2SccIterationLaunchStatus::kCudaError;
        result.iteration.cuda_status = status;
      }
    } else
#endif
    {
      result.iteration =
          geometry == nullptr
              ? launch_gfn2_restricted_scc_iteration_cuda(binding, stream)
              : launch_gfn2_restricted_scc_iteration_cuda(binding, *geometry, stream);
    }
    if (!result.iteration.success()) {
      return result;
    }
    ++result.submitted_iterations;
  }
  return result;
}

#if CUDART_VERSION >= 12030

__global__ void reset_device_loop_control_kernel(Gfn2SccDeviceLoopControl* control) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    control->canonical_active_count = 0u;
    control->device_launch_error = 0u;
    control->numerical_body_count = 0u;
    control->plan_failure_snapshot = 0u;
  }
}

__global__ void count_device_loop_body_kernel(Gfn2SccDeviceLoopControl* control) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    ++control->numerical_body_count;
  }
}

__global__ void snapshot_device_loop_failure_kernel(Gfn2SccIterationDeviceLedger ledger,
                                                    Gfn2SccDeviceLoopControl* control) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    control->plan_failure_snapshot = *ledger.plan_failure_record;
  }
}

__device__ void close_device_loop_after_launch_failure(Gfn2SccIterationDeviceLedger ledger,
                                                       Gfn2SccDeviceLoopControl* control,
                                                       cudaError_t launch_error) {
  control->device_launch_error = static_cast<std::uint32_t>(launch_error);
  if (*ledger.plan_failure_record == 0u) {
    *ledger.plan_failure_record = gfn2_scc_stage_failure_record(
        Gfn2SccStageId::kActivity,
        static_cast<std::uint32_t>(Gfn2SccIterationControlCode::kDeviceGraphLaunchFailed));
  }
  *ledger.sequence_active = 0u;
  control->canonical_active_count = 0u;
  for (std::int64_t system = 0; system < ledger.batch_elements; ++system) {
    ledger.active_mask[system] = 0u;
  }
}

/*
 * Count canonical activity serially so the Graph controller itself needs no
 * block-shared scratch. Only this one thread performs the device Graph launch.
 * A launch error is preserved in both the stable SCC ledger and the controller
 * diagnostic instead of silently falling through to a partially executed SCC.
 */
__global__ void gate_device_loop_kernel(Gfn2SccIterationDeviceLedger ledger,
                                        Gfn2SccDeviceLoopControl* control, cudaGraphExec_t body) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const std::uint64_t failure_record = *ledger.plan_failure_record;
  const bool sequence_open = failure_record == 0u && *ledger.sequence_active == 1u;
  std::uint32_t active_count = 0u;
  for (std::int64_t system = 0; system < ledger.batch_elements; ++system) {
    if (sequence_open && ledger.active_mask[system] == 1u) {
      ++active_count;
    } else if (!sequence_open) {
      ledger.active_mask[system] = 0u;
    }
  }
  control->canonical_active_count = active_count;
  if (active_count == 0u) {
    return;
  }
  const cudaError_t status = cudaGraphLaunch(body, cudaStreamGraphFireAndForget);
  if (status != cudaSuccess) {
    close_device_loop_after_launch_failure(ledger, control, status);
  }
}

/*
 * The body snapshots a numerical plan failure before next activity derivation
 * resets the ledger. Restore that record first, then enqueue the currently
 * running device Graph on its tail stream only when canonical work remains.
 * Tail self-launch supplies exact device-resident early stop without any
 * conditional Graph node and therefore avoids conditional-node instrumentation.
 */
__global__ void tail_device_loop_kernel(Gfn2SccIterationDeviceLedger ledger,
                                        Gfn2SccDeviceLoopControl* control) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const std::uint64_t failure_record = control->plan_failure_snapshot != 0u
                                           ? control->plan_failure_snapshot
                                           : *ledger.plan_failure_record;
  const bool sequence_open = failure_record == 0u && *ledger.sequence_active == 1u;
  std::uint32_t active_count = 0u;
  for (std::int64_t system = 0; system < ledger.batch_elements; ++system) {
    if (sequence_open && ledger.active_mask[system] == 1u) {
      ++active_count;
    } else if (!sequence_open) {
      ledger.active_mask[system] = 0u;
    }
  }
  if (failure_record != 0u) {
    *ledger.plan_failure_record = failure_record;
    *ledger.sequence_active = 0u;
    active_count = 0u;
  }
  control->canonical_active_count = active_count;
  if (active_count == 0u) {
    return;
  }
  const cudaGraphExec_t current = cudaGetCurrentGraphExec();
  const cudaError_t status = current == nullptr
                                 ? cudaErrorInvalidResourceHandle
                                 : cudaGraphLaunch(current, cudaStreamGraphTailLaunch);
  if (status != cudaSuccess) {
    close_device_loop_after_launch_failure(ledger, control, status);
  }
}

cudaError_t check_kernel_launch() noexcept { return cudaPeekAtLastError(); }

/*
 * Device-resident dispatch state shared by every exact-capacity chain kernel
 * is declared in gfn2_scc_loop.cuh so State can reference it. The executable
 * table is filled once after all chain executables exist and is read only at
 * replay, so no per-iteration host transfer, allocation, polling, or
 * synchronization is introduced.
 */
__global__ void launch_scc_chain_tail_kernel(Gfn2SccIterationDeviceLedger ledger,
                                             Gfn2SccDeviceLoopControl* control,
                                             Gfn2SccDispatchChainDevice chain) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const std::uint64_t failure_record = control->plan_failure_snapshot != 0u
                                           ? control->plan_failure_snapshot
                                           : *ledger.plan_failure_record;
  const bool sequence_open = failure_record == 0u && *ledger.sequence_active == 1u;
  std::uint32_t active_count = 0u;
  for (std::int64_t system = 0; system < ledger.batch_elements; ++system) {
    if (sequence_open && ledger.active_mask[system] == 1u) {
      ++active_count;
    } else if (!sequence_open) {
      ledger.active_mask[system] = 0u;
    }
  }
  if (failure_record != 0u) {
    *ledger.plan_failure_record = failure_record;
    *ledger.sequence_active = 0u;
    active_count = 0u;
  }
  control->canonical_active_count = active_count;
  if (active_count == 0u) {
    return;
  }
  if (chain.pre_slot < 0 || chain.pre_slot >= chain.table_slots || chain.table == nullptr) {
    close_device_loop_after_launch_failure(ledger, control, cudaErrorInvalidResourceHandle);
    return;
  }
  const cudaError_t status =
      cudaGraphLaunch(chain.table[chain.pre_slot], cudaStreamGraphTailLaunch);
  if (status != cudaSuccess) {
    close_device_loop_after_launch_failure(ledger, control, status);
  }
}

/* Dispatches the exact-count eigensolver executable for one bucket based on the
 * device-published active count. Must be the final node of its graph so the
 * selected child runs after the current graph completes. When bucket_index is
 * at or past bucket_count (last backtransform body), it instead launches the
 * post-eigensolver executable at chain.post_slot. A zero active count skips
 * provider arithmetic entirely and chains directly to back_transform[count=0]. */
__global__ void dispatch_scc_eigensolver_kernel(Gfn2SccIterationDeviceLedger ledger,
                                                Gfn2SccDeviceLoopControl* control,
                                                Gfn2SccDispatchChainDevice chain,
                                                const Gfn2EigensolverBucketActivity* activity,
                                                std::int64_t bucket_index, std::int64_t capacity,
                                                std::int64_t eig_base, std::int64_t back_base) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const std::uint64_t failure_record = *ledger.plan_failure_record;
  if (failure_record != 0u || *ledger.sequence_active != 1u || chain.table == nullptr) {
    control->canonical_active_count = 0u;
    return;
  }
  if (bucket_index >= chain.bucket_count) {
    if (chain.post_slot < 0 || chain.post_slot >= chain.table_slots) {
      close_device_loop_after_launch_failure(ledger, control, cudaErrorInvalidResourceHandle);
      return;
    }
    const cudaError_t status =
        cudaGraphLaunch(chain.table[chain.post_slot], cudaStreamGraphTailLaunch);
    if (status != cudaSuccess) {
      close_device_loop_after_launch_failure(ledger, control, status);
    }
    return;
  }
  if (bucket_index < 0 || capacity < 0) {
    close_device_loop_after_launch_failure(ledger, control, cudaErrorInvalidResourceHandle);
    return;
  }
  const std::uint32_t count = activity[bucket_index].active_count;
  const std::int64_t zero_slot = back_base;
  const std::int64_t clamped =
      static_cast<std::int64_t>(count) <= capacity ? static_cast<std::int64_t>(count) : capacity;
  const std::int64_t count_slot = eig_base + (clamped > 0 ? clamped - 1 : clamped);
  const std::int64_t slot = count == 0u ? zero_slot : count_slot;
  if (slot < 0 || slot >= chain.table_slots) {
    close_device_loop_after_launch_failure(ledger, control, cudaErrorInvalidResourceHandle);
    return;
  }
  const cudaError_t status = cudaGraphLaunch(chain.table[slot], cudaStreamGraphTailLaunch);
  if (status != cudaSuccess) {
    close_device_loop_after_launch_failure(ledger, control, status);
  }
}

/* Dispatches the exact-count backtransform executable for one bucket based on
 * the completed-count published after the eigensolver body. */
__global__ void dispatch_scc_backtransform_kernel(Gfn2SccIterationDeviceLedger ledger,
                                                  Gfn2SccDeviceLoopControl* control,
                                                  Gfn2SccDispatchChainDevice chain,
                                                  const Gfn2EigensolverBucketActivity* activity,
                                                  std::int64_t bucket_index, std::int64_t capacity,
                                                  std::int64_t back_base) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const std::uint64_t failure_record = *ledger.plan_failure_record;
  if (failure_record != 0u || *ledger.sequence_active != 1u || chain.table == nullptr ||
      bucket_index < 0 || bucket_index >= chain.bucket_count || capacity < 0) {
    control->canonical_active_count = 0u;
    return;
  }
  const std::uint32_t count = activity[bucket_index].completed_count;
  const std::int64_t slot =
      back_base +
      (static_cast<std::int64_t>(count) <= capacity ? static_cast<std::int64_t>(count) : capacity);
  if (slot < 0 || slot >= chain.table_slots) {
    close_device_loop_after_launch_failure(ledger, control, cudaErrorInvalidResourceHandle);
    return;
  }
  const cudaError_t status = cudaGraphLaunch(chain.table[slot], cudaStreamGraphTailLaunch);
  if (status != cudaSuccess) {
    close_device_loop_after_launch_failure(ledger, control, status);
  }
}

void finish_or_abort_capture(cudaStream_t stream) noexcept {
  /* This helper is used only with cudaStreamBeginCapture(), never with
   * cudaStreamBeginCaptureToGraph(). Therefore any graph returned while
   * unwinding belongs to this setup path and must be destroyed here. */
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &capture_status) == cudaSuccess &&
      capture_status != cudaStreamCaptureStatusNone) {
    cudaGraph_t discarded = nullptr;
    (void)cudaStreamEndCapture(stream, &discarded);
    if (discarded != nullptr) {
      (void)cudaGraphDestroy(discarded);
    }
  }
  (void)cudaGetLastError();
}

#endif  // CUDART_VERSION >= 12030

}  // namespace

struct Gfn2SccLoopCudaGraphOwner::State {
  Gfn2SccIterationBinding binding{};
  Gfn2GeometryEpochConsumerDevice geometry{};
  bool dynamic_geometry = false;
  Gfn2SccLoopGraphFallbackReason fallback_reason = Gfn2SccLoopGraphFallbackReason::kNone;
  Gfn2SccLoopGraphPreference preference = Gfn2SccLoopGraphPreference::kAuto;
  cudaGraph_t root_graph = nullptr;
  cudaGraphExec_t root_executable = nullptr;
  cudaGraph_t body_graph = nullptr;
  cudaGraphExec_t body_executable = nullptr;
  Gfn2SccDeviceLoopControl* control = nullptr;
  Gfn2SccDiagnosticsDevice diagnostics{};
#ifdef XTBLOOM_CUDA_SCC_DIAGNOSTICS
  bool external_capture_uninstrumented = false;
#endif

  /* Production exact-capacity dispatch chain. When ready, launch() uses the
   * chain instead of the monolithic body_graph/body_executable. Every chain
   * executable is device-launchable and uploaded; the device table is read at
   * replay only. All members are setup-owned. */
  bool dispatch_chain_ready = false;
  cudaGraphExec_t* device_table = nullptr;
  std::int64_t table_slots = 0;
  /* Host mirrors for teardown; each entry is a distinct owned executable. */
  std::vector<cudaGraphExec_t> executables{};
  std::vector<cudaGraph_t> graph_owns{};
};

namespace {

cudaError_t allocate_diagnostics(Gfn2SccLoopCudaGraphOwner::State& state) noexcept {
#ifdef XTBLOOM_CUDA_SCC_DIAGNOSTICS
  auto& diagnostics = state.diagnostics;
  diagnostics.bucket_count = state.binding.plan.eigensolver_provider.bucket_count;
  diagnostics.maximum_iterations = state.binding.plan.activity_policy.maximum_iterations;
  const auto bucket_count = static_cast<std::size_t>(diagnostics.bucket_count);
  const auto limit = std::numeric_limits<std::size_t>::max();
  if (diagnostics.maximum_iterations > limit / sizeof(Gfn2SccDiagnosticIteration) ||
      bucket_count > limit / sizeof(Gfn2EigensolverBucket) ||
      (bucket_count != 0u &&
       diagnostics.maximum_iterations > limit / sizeof(Gfn2SccDiagnosticBucket) / bucket_count)) {
    return cudaErrorInvalidValue;
  }
  const auto iterations = static_cast<std::size_t>(diagnostics.maximum_iterations);
  cudaError_t status =
      cudaMalloc(reinterpret_cast<void**>(&diagnostics.header), sizeof(Gfn2SccDiagnosticHeader));
  if (status == cudaSuccess) {
    status = cudaMalloc(reinterpret_cast<void**>(&diagnostics.iterations),
                        iterations * sizeof(Gfn2SccDiagnosticIteration));
  }
  if (status == cudaSuccess && bucket_count != 0u) {
    status = cudaMalloc(reinterpret_cast<void**>(&diagnostics.rows),
                        iterations * bucket_count * sizeof(Gfn2SccDiagnosticBucket));
  }
  if (status == cudaSuccess && bucket_count != 0u) {
    status = cudaMalloc(reinterpret_cast<void**>(&diagnostics.buckets),
                        bucket_count * sizeof(Gfn2EigensolverBucket));
  }
  if (status == cudaSuccess && bucket_count != 0u) {
    status = cudaMemcpy(diagnostics.buckets, state.binding.plan.eigensolver_provider.buckets,
                        bucket_count * sizeof(Gfn2EigensolverBucket), cudaMemcpyHostToDevice);
  }
  if (status == cudaSuccess) {
    status = cudaMemset(diagnostics.iterations, 0, iterations * sizeof(Gfn2SccDiagnosticIteration));
  }
  if (status == cudaSuccess && bucket_count != 0u) {
    status = cudaMemset(diagnostics.rows, 0,
                        iterations * bucket_count * sizeof(Gfn2SccDiagnosticBucket));
  }
  if (status == cudaSuccess) {
    const Gfn2SccDiagnosticHeader initial{};
    status = cudaMemcpy(diagnostics.header, &initial, sizeof(initial), cudaMemcpyHostToDevice);
  }
  return status;
#else
  return cudaSuccess;
#endif
}

cudaError_t diagnostic_reset(const Gfn2SccLoopCudaGraphOwner::State& state, cudaStream_t stream,
                             Gfn2SccLoopExecutionMode mode) noexcept {
#ifdef XTBLOOM_CUDA_SCC_DIAGNOSTICS
  return reset_gfn2_scc_diagnostics_cuda(state.diagnostics, stream,
                                         static_cast<std::uint32_t>(mode));
#else
  return cudaSuccess;
#endif
}

cudaError_t diagnostic_begin(const Gfn2SccLoopCudaGraphOwner::State& state,
                             cudaStream_t stream) noexcept {
#ifdef XTBLOOM_CUDA_SCC_DIAGNOSTICS
  return begin_gfn2_scc_diagnostics_cuda(state.diagnostics, state.binding, stream);
#else
  return cudaSuccess;
#endif
}

cudaError_t diagnostic_finish(const Gfn2SccLoopCudaGraphOwner::State& state, bool exact_capacity,
                              cudaStream_t stream) noexcept {
#ifdef XTBLOOM_CUDA_SCC_DIAGNOSTICS
  return finish_gfn2_scc_diagnostics_cuda(state.diagnostics, state.binding, exact_capacity, stream);
#else
  return cudaSuccess;
#endif
}

void destroy_graph_state(Gfn2SccLoopCudaGraphOwner::State* state) noexcept {
  if (state == nullptr) {
    return;
  }
  if (state->root_executable != nullptr) {
    (void)cudaGraphExecDestroy(state->root_executable);
  }
  if (state->root_graph != nullptr) {
    (void)cudaGraphDestroy(state->root_graph);
  }
  if (state->body_executable != nullptr) {
    (void)cudaGraphExecDestroy(state->body_executable);
  }
  if (state->body_graph != nullptr) {
    (void)cudaGraphDestroy(state->body_graph);
  }
  for (cudaGraphExec_t executable : state->executables) {
    (void)cudaGraphExecDestroy(executable);
  }
  for (cudaGraph_t graph : state->graph_owns) {
    (void)cudaGraphDestroy(graph);
  }
  if (state->device_table != nullptr) {
    (void)cudaFree(state->device_table);
  }
  if (state->control != nullptr) {
    (void)cudaFree(state->control);
  }
  if (state->diagnostics.header != nullptr) (void)cudaFree(state->diagnostics.header);
  if (state->diagnostics.iterations != nullptr) (void)cudaFree(state->diagnostics.iterations);
  if (state->diagnostics.rows != nullptr) (void)cudaFree(state->diagnostics.rows);
  if (state->diagnostics.buckets != nullptr) (void)cudaFree(state->diagnostics.buckets);
  delete state;
}

Gfn2SccLoopGraphBuildResult invalid_graph_binding(
    const Gfn2SccIterationLaunchResult& iteration) noexcept {
  Gfn2SccLoopGraphBuildResult result{};
  result.status = Gfn2SccLoopGraphBuildStatus::kInvalidBinding;
  result.iteration = iteration;
  return result;
}

Gfn2SccLoopGraphBuildResult fallback_graph_build(
    Gfn2SccLoopCudaGraphOwner::State& state, Gfn2SccLoopGraphFallbackReason reason,
    cudaError_t cuda_status = cudaSuccess,
    const Gfn2SccIterationLaunchResult& iteration = {}) noexcept {
  state.fallback_reason = reason;
  if (state.root_executable != nullptr) {
    (void)cudaGraphExecDestroy(state.root_executable);
    state.root_executable = nullptr;
  }
  if (state.root_graph != nullptr) {
    (void)cudaGraphDestroy(state.root_graph);
    state.root_graph = nullptr;
  }
  if (state.body_executable != nullptr) {
    (void)cudaGraphExecDestroy(state.body_executable);
    state.body_executable = nullptr;
  }
  if (state.body_graph != nullptr) {
    (void)cudaGraphDestroy(state.body_graph);
    state.body_graph = nullptr;
  }
  if (state.control != nullptr) {
    (void)cudaFree(state.control);
    state.control = nullptr;
  }
  Gfn2SccLoopGraphBuildResult result{};
  result.status = Gfn2SccLoopGraphBuildStatus::kBoundedFallbackReady;
  result.fallback_reason = reason;
  result.cuda_status = cuda_status;
  result.iteration = iteration;
  (void)cudaGetLastError();
  return result;
}

#if CUDART_VERSION >= 12030

Gfn2SccIterationLaunchResult launch_graph_activity(const Gfn2SccLoopCudaGraphOwner::State& state,
                                                   cudaStream_t stream) noexcept {
  return state.dynamic_geometry
             ? launch_gfn2_restricted_scc_activity_cuda(state.binding, state.geometry, stream)
             : launch_gfn2_restricted_scc_activity_cuda(state.binding, stream);
}

Gfn2SccIterationLaunchResult launch_graph_numerical_body(
    const Gfn2SccLoopCudaGraphOwner::State& state, cudaStream_t stream) noexcept {
  return state.dynamic_geometry
             ? launch_gfn2_restricted_scc_numerical_body_cuda(state.binding, state.geometry, stream)
             : launch_gfn2_restricted_scc_numerical_body_cuda(state.binding, stream);
}

Gfn2SccIterationLaunchResult launch_graph_pre_eigensolver(
    const Gfn2SccLoopCudaGraphOwner::State& state, cudaStream_t stream) noexcept {
  return state.dynamic_geometry
             ? launch_gfn2_restricted_scc_pre_eigensolver_cuda(state.binding, state.geometry,
                                                               stream)
             : launch_gfn2_restricted_scc_pre_eigensolver_cuda(state.binding, stream);
}

Gfn2SccIterationLaunchResult launch_graph_post_eigensolver(
    const Gfn2SccLoopCudaGraphOwner::State& state, cudaStream_t stream) noexcept {
  return state.dynamic_geometry
             ? launch_gfn2_restricted_scc_post_eigensolver_cuda(state.binding, state.geometry,
                                                                stream)
             : launch_gfn2_restricted_scc_post_eigensolver_cuda(state.binding, stream);
}

struct DeviceGraphAuditResult {
  cudaError_t status = cudaSuccess;
  bool supported = true;
};

DeviceGraphAuditResult audit_device_launch_graph(cudaGraph_t graph) noexcept {
  std::size_t node_count = 0u;
  cudaError_t status = cudaGraphGetNodes(graph, nullptr, &node_count);
  if (status != cudaSuccess) {
    return {status, false};
  }
  if (node_count == 0u) {
    return {};
  }
  auto* const nodes = new (std::nothrow) cudaGraphNode_t[node_count];
  if (nodes == nullptr) {
    return {cudaErrorMemoryAllocation, false};
  }
  status = cudaGraphGetNodes(graph, nodes, &node_count);
  if (status != cudaSuccess) {
    delete[] nodes;
    return {status, false};
  }
  DeviceGraphAuditResult result{};
  for (std::size_t index = 0u; index < node_count; ++index) {
    cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
    status = cudaGraphNodeGetType(nodes[index], &type);
    if (status != cudaSuccess) {
      result = {status, false};
      break;
    }
    if (type == cudaGraphNodeTypeKernel || type == cudaGraphNodeTypeMemcpy ||
        type == cudaGraphNodeTypeMemset) {
      continue;
    }
    if (type == cudaGraphNodeTypeGraph) {
      cudaGraph_t child = nullptr;
      status = cudaGraphChildGraphNodeGetGraph(nodes[index], &child);
      if (status != cudaSuccess || child == nullptr) {
        result = {status == cudaSuccess ? cudaErrorInvalidResourceHandle : status, false};
        break;
      }
      result = audit_device_launch_graph(child);
      if (!result.supported) {
        break;
      }
      continue;
    }
    /* Device-launchable Graphs reject empty, event, external semaphore,
     * host, allocation/free, and conditional nodes. Audit recursively before
     * instantiation so fallback identity is deterministic across drivers. */
    result = {cudaErrorNotSupported, false};
    break;
  }
  delete[] nodes;
  return result;
}

Gfn2SccLoopGraphBuildResult build_device_tail_graph(
    Gfn2SccLoopCudaGraphOwner::State& state) noexcept {
  if (state.binding.plan.eigensolver_provider.capture_mode !=
      Gfn2SccIterationProviderCaptureMode::kGraphSupported) {
    return fallback_graph_build(state, Gfn2SccLoopGraphFallbackReason::kProviderCaptureUnsupported);
  }

  cudaError_t status =
      cudaMalloc(reinterpret_cast<void**>(&state.control), sizeof(Gfn2SccDeviceLoopControl));
  if (status != cudaSuccess) {
    return fallback_graph_build(state, Gfn2SccLoopGraphFallbackReason::kControlAllocationFailed,
                                status);
  }
  status = cudaMemset(state.control, 0, sizeof(Gfn2SccDeviceLoopControl));
  if (status != cudaSuccess) {
    return fallback_graph_build(state, Gfn2SccLoopGraphFallbackReason::kControlAllocationFailed,
                                status);
  }

  cudaStream_t capture_stream = nullptr;
  status = cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking);
  if (status != cudaSuccess) {
    return fallback_graph_build(state, Gfn2SccLoopGraphFallbackReason::kRootCaptureFailed, status);
  }
  const auto finish_stream = [&]() noexcept {
    finish_or_abort_capture(capture_stream);
    (void)cudaStreamDestroy(capture_stream);
    capture_stream = nullptr;
  };
  const auto capture_fallback = [&](Gfn2SccLoopGraphFallbackReason reason, cudaError_t error,
                                    const Gfn2SccIterationLaunchResult& iteration = {}) {
    finish_stream();
    return fallback_graph_build(state, reason, error, iteration);
  };

  /* Capture one complete numerical iteration as a flat device-launchable
   * Graph. The final kernel may enqueue this same executable on the tail
   * stream, which gives exact early stop without conditional nodes. */
  status = cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);
  if (status != cudaSuccess) {
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kNumericalBodyCaptureFailed, status);
  }
  count_device_loop_body_kernel<<<1, 1, 0, capture_stream>>>(state.control);
  status = check_kernel_launch();
  if (status == cudaSuccess) status = diagnostic_begin(state, capture_stream);
  const Gfn2SccIterationLaunchResult numerical =
      status == cudaSuccess ? launch_graph_numerical_body(state, capture_stream)
                            : Gfn2SccIterationLaunchResult{};
  if (status == cudaSuccess && !numerical.success()) {
    if (numerical.status == Gfn2SccIterationLaunchStatus::kInvalidBinding) {
      finish_stream();
      return invalid_graph_binding(numerical);
    }
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kNumericalBodyCaptureFailed,
                            numerical.cuda_status, numerical);
  }
  if (status == cudaSuccess) status = diagnostic_finish(state, false, capture_stream);
  if (status == cudaSuccess) {
    snapshot_device_loop_failure_kernel<<<1, 1, 0, capture_stream>>>(state.binding.workspace.ledger,
                                                                     state.control);
    status = check_kernel_launch();
  }
  const Gfn2SccIterationLaunchResult next_root = status == cudaSuccess
                                                     ? launch_graph_activity(state, capture_stream)
                                                     : Gfn2SccIterationLaunchResult{};
  if (status == cudaSuccess && !next_root.success()) {
    if (next_root.status == Gfn2SccIterationLaunchStatus::kInvalidBinding) {
      finish_stream();
      return invalid_graph_binding(next_root);
    }
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kNumericalBodyCaptureFailed,
                            next_root.cuda_status, next_root);
  }
  if (status == cudaSuccess) {
    tail_device_loop_kernel<<<1, 1, 0, capture_stream>>>(state.binding.workspace.ledger,
                                                         state.control);
    status = check_kernel_launch();
  }
  if (status != cudaSuccess) {
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kNumericalBodyCaptureFailed, status);
  }
  status = cudaStreamEndCapture(capture_stream, &state.body_graph);
  if (status != cudaSuccess || state.body_graph == nullptr) {
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kNumericalBodyCaptureFailed,
                            status == cudaSuccess ? cudaErrorStreamCaptureInvalidated : status);
  }

  const DeviceGraphAuditResult audit = audit_device_launch_graph(state.body_graph);
  if (!audit.supported) {
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kDeviceGraphNodeUnsupported,
                            audit.status);
  }

  status = cudaGraphInstantiate(&state.body_executable, state.body_graph,
                                cudaGraphInstantiateFlagDeviceLaunch);
  if (status != cudaSuccess) {
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kDeviceGraphInstantiationFailed,
                            status);
  }
  status = cudaGraphUpload(state.body_executable, capture_stream);
  if (status == cudaSuccess) {
    /* build() owns setup ordering. A successful return guarantees that a root
     * launch on any caller stream cannot race the asynchronous body upload. */
    status = cudaStreamSynchronize(capture_stream);
  }
  if (status != cudaSuccess) {
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kDeviceGraphUploadFailed, status);
  }

  status = cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);
  if (status != cudaSuccess) {
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kRootCaptureFailed, status);
  }
  reset_device_loop_control_kernel<<<1, 1, 0, capture_stream>>>(state.control);
  status = check_kernel_launch();
  if (status == cudaSuccess) {
    status = diagnostic_reset(state, capture_stream, Gfn2SccLoopExecutionMode::kDeviceTailGraph);
  }
  const Gfn2SccIterationLaunchResult root = status == cudaSuccess
                                                ? launch_graph_activity(state, capture_stream)
                                                : Gfn2SccIterationLaunchResult{};
  if (status == cudaSuccess && !root.success()) {
    if (root.status == Gfn2SccIterationLaunchStatus::kInvalidBinding) {
      finish_stream();
      return invalid_graph_binding(root);
    }
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kRootCaptureFailed, root.cuda_status,
                            root);
  }
  if (status == cudaSuccess) {
    gate_device_loop_kernel<<<1, 1, 0, capture_stream>>>(state.binding.workspace.ledger,
                                                         state.control, state.body_executable);
    status = check_kernel_launch();
  }
  if (status != cudaSuccess) {
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kRootCaptureFailed, status);
  }
  status = cudaStreamEndCapture(capture_stream, &state.root_graph);
  if (status != cudaSuccess || state.root_graph == nullptr) {
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kRootCaptureFailed,
                            status == cudaSuccess ? cudaErrorStreamCaptureInvalidated : status);
  }

  status = cudaGraphInstantiate(&state.root_executable, state.root_graph, 0u);
  if (status != cudaSuccess) {
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kInstantiationFailed, status);
  }
  (void)cudaStreamDestroy(capture_stream);
  capture_stream = nullptr;

  Gfn2SccLoopGraphBuildResult result{};
  result.status = Gfn2SccLoopGraphBuildStatus::kDeviceTailGraphReady;
  return result;
}

#endif  // CUDART_VERSION >= 12030

#if CUDART_VERSION >= 12080

const Gfn2SccStageDeviceReport* find_scc_stage_report(const Gfn2SccIterationDevicePlan& plan,
                                                      Gfn2SccStageId stage) noexcept {
  for (std::int64_t index = 0; index < plan.report_count; ++index) {
    if (plan.reports[index].stage == stage) {
      return &plan.reports[index];
    }
  }
  return nullptr;
}

std::uint32_t* mutable_stage_codes(const Gfn2SccStageDeviceReport& report) noexcept {
  return const_cast<std::uint32_t*>(static_cast<const std::uint32_t*>(report.system_codes));
}

std::uint32_t* mutable_stage_device_error(const Gfn2SccStageDeviceReport& report) noexcept {
  return const_cast<std::uint32_t*>(report.device_error);
}

void destroy_dispatch_chain_resources(Gfn2SccLoopCudaGraphOwner::State& state) noexcept {
  for (cudaGraphExec_t executable : state.executables) {
    (void)cudaGraphExecDestroy(executable);
  }
  state.executables.clear();
  for (cudaGraph_t graph : state.graph_owns) {
    (void)cudaGraphDestroy(graph);
  }
  state.graph_owns.clear();
  if (state.device_table != nullptr) {
    (void)cudaFree(state.device_table);
    state.device_table = nullptr;
  }
  state.table_slots = 0;
  state.dispatch_chain_ready = false;
}

/*
 * Device-dispatched exact-capacity chain builder for the production SCC loop.
 *
 * The monolithic numerical body submits the dense eigensolver at full bucket
 * capacity with identity placeholders for inactive peers. This path replaces
 * only the eigensolver stage with a chain of pre-built device-launchable
 * executables, one per (bucket, capacity), selected by a device dispatcher from
 * a device-resident executable table. No host decision, allocation, transfer,
 * polling, or synchronization happens on the hot path.
 *
 * Chain topology (all executables device-launchable and uploaded at build):
 *   root (host graph): reset control + activity + gate -> pre_exec
 *   pre_exec: count body + pre-eigensolver segment + open eigen stage + reset
 *             eigensolver errors + launch sequence + prepare/compact all
 *             buckets + dispatch eig[bucket 0]            (tail, LAST NODE)
 *   eig_exec[b][c>0]: enqueue eig body(c) + compact successful counts +
 *             dispatch back[b]                           (tail, LAST NODE)
 *   back_exec[b][c>=0]: enqueue back body(c) (skip when c==0) +
 *             dispatch eig[bucket b+1] (or post when past the last bucket)
 *   post_exec: normalize eigen stage + post-eigensolver segment + snapshot
 *             failure + next activity + relaunch pre_exec (tail, LAST NODE)
 *
 * table layout: slot 0 = pre_exec, slot 1 = post_exec, then per bucket b:
 * eig_base[b] holds eig_exec[b][1..cap_b]; back_base[b] holds
 * back_exec[b][0..cap_b]. dispatch_eig(b) uses eig_exec[b][active] when active>0
 * and back_exec[b][0] when active==0; dispatch_back(b) uses
 * back_exec[b][completed]. The final back_exec of the last bucket dispatches
 * slot 1 (post_exec).
 */
Gfn2SccLoopGraphBuildResult build_dispatch_chain(Gfn2SccLoopCudaGraphOwner::State& state) noexcept
    try {
  const Gfn2SccIterationBinding& binding = state.binding;
  const Gfn2SccIterationDevicePlan& plan = binding.plan;
  const auto& workspace = binding.workspace;
  const std::int64_t bucket_count = plan.eigensolver_provider.bucket_count;
  const Gfn2EigensolverBucket* buckets = plan.eigensolver_provider.buckets;

  if (plan.eigensolver_provider.capture_mode !=
      Gfn2SccIterationProviderCaptureMode::kGraphSupported) {
    return fallback_graph_build(state, Gfn2SccLoopGraphFallbackReason::kProviderCaptureUnsupported);
  }
  if (bucket_count <= 0 || buckets == nullptr) {
    return fallback_graph_build(state, Gfn2SccLoopGraphFallbackReason::kDispatchUnsupportedLayout);
  }
  /* The compaction kernels index physical systems. Production restricted
   * batches have nspin==1 everywhere, so solve_count == system_count and the
   * restricted compact map is valid. Mixed-spin layouts keep the monolithic
   * full-capacity body as the documented bounded fallback. */
  if (plan.wavefunction_layout.total_spin_channels != plan.topology.batch_size) {
    return fallback_graph_build(state, Gfn2SccLoopGraphFallbackReason::kDispatchUnsupportedLayout);
  }

  std::vector<std::int64_t> eig_base(static_cast<std::size_t>(bucket_count));
  std::vector<std::int64_t> back_base(static_cast<std::size_t>(bucket_count));
  std::vector<std::int64_t> capacity(static_cast<std::size_t>(bucket_count));
  std::int64_t next_slot = 2;
  for (std::int64_t b = 0; b < bucket_count; ++b) {
    const std::int64_t cap = buckets[b].system_count;
    if (cap <= 0 || cap > std::numeric_limits<std::int32_t>::max()) {
      return fallback_graph_build(state, Gfn2SccLoopGraphFallbackReason::kDispatchCapacityOverflow);
    }
    capacity[static_cast<std::size_t>(b)] = cap;
    eig_base[static_cast<std::size_t>(b)] = next_slot;
    next_slot += cap;
    back_base[static_cast<std::size_t>(b)] = next_slot;
    next_slot += cap + 1;
  }
  /* pre(0) + post(1) + eig slots + back slots. */
  const std::int64_t table_slots = next_slot;
  std::vector<cudaGraphExec_t> host_table(static_cast<std::size_t>(table_slots), cudaGraphExec_t{});
  state.executables.reserve(static_cast<std::size_t>(table_slots));
  state.graph_owns.reserve(static_cast<std::size_t>(table_slots));

  cudaError_t status =
      cudaMalloc(reinterpret_cast<void**>(&state.control), sizeof(Gfn2SccDeviceLoopControl));
  if (status != cudaSuccess) {
    return fallback_graph_build(state, Gfn2SccLoopGraphFallbackReason::kControlAllocationFailed,
                                status);
  }
  status = cudaMemset(state.control, 0, sizeof(Gfn2SccDeviceLoopControl));
  if (status != cudaSuccess) {
    destroy_dispatch_chain_resources(state);
    return fallback_graph_build(state, Gfn2SccLoopGraphFallbackReason::kControlAllocationFailed,
                                status);
  }
  status = cudaMalloc(reinterpret_cast<void**>(&state.device_table),
                      static_cast<std::size_t>(table_slots) * sizeof(cudaGraphExec_t));
  if (status != cudaSuccess) {
    destroy_dispatch_chain_resources(state);
    return fallback_graph_build(
        state, Gfn2SccLoopGraphFallbackReason::kDispatchTableAllocationFailed, status);
  }
  state.table_slots = table_slots;

  cudaStream_t capture_stream = nullptr;
  status = cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking);
  if (status != cudaSuccess) {
    destroy_dispatch_chain_resources(state);
    return fallback_graph_build(state, Gfn2SccLoopGraphFallbackReason::kRootCaptureFailed, status);
  }
  const auto finish_stream = [&]() noexcept {
    finish_or_abort_capture(capture_stream);
    (void)cudaStreamDestroy(capture_stream);
    capture_stream = nullptr;
  };
  const auto chain_fallback = [&](Gfn2SccLoopGraphFallbackReason reason, cudaError_t error,
                                  const Gfn2SccIterationLaunchResult& iteration = {}) {
    finish_stream();
    destroy_dispatch_chain_resources(state);
    return fallback_graph_build(state, reason, error, iteration);
  };

  const Gfn2SccStageDeviceReport* eigen_report =
      find_scc_stage_report(plan, Gfn2SccStageId::kEigensolver);
  if (eigen_report == nullptr || eigen_report->system_codes == nullptr ||
      eigen_report->device_error == nullptr) {
    return chain_fallback(Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed,
                          cudaErrorInvalidResourceHandle);
  }
  std::uint32_t* const eigen_codes = mutable_stage_codes(*eigen_report);
  std::uint32_t* const eigen_device = mutable_stage_device_error(*eigen_report);

  /* ---- pre_exec ---- */
  cudaGraph_t pre_graph = nullptr;
  status = cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);
  if (status != cudaSuccess) {
    return chain_fallback(Gfn2SccLoopGraphFallbackReason::kRootCaptureFailed, status);
  }
  count_device_loop_body_kernel<<<1, 1, 0, capture_stream>>>(state.control);
  status = check_kernel_launch();
  if (status == cudaSuccess) status = diagnostic_begin(state, capture_stream);
  const Gfn2SccIterationLaunchResult pre_segment =
      status == cudaSuccess ? launch_graph_pre_eigensolver(state, capture_stream)
                            : Gfn2SccIterationLaunchResult{};
  if (status == cudaSuccess && !pre_segment.success()) {
    if (pre_segment.status == Gfn2SccIterationLaunchStatus::kInvalidBinding) {
      finish_stream();
      return invalid_graph_binding(pre_segment);
    }
    return chain_fallback(Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed,
                          pre_segment.cuda_status, pre_segment);
  }
  /* Open the eigensolver stage, clear its error state, and open the launch
   * sequence before compaction. These nodes run unconditionally inside the
   * captured pre_exec graph, mirroring the monolithic begin_stage layout: a
   * runtime plan failure is latched by the earlier stages and the dispatch
   * kernels below gate on ledger.sequence_active, so the provider chain is
   * never entered after a failed pre-eigensolver stage. The per-stage
   * diagnostics remain open only because the aborted iteration is not
   * normalized, exactly as in the monolithic full-body capture. */
  if (status == cudaSuccess) {
    status = open_gfn2_scc_stage_cuda(*eigen_report, capture_stream);
  }
  if (status == cudaSuccess) {
    status = reset_gfn2_eigensolver_device_errors_cuda(plan.topology.batch_size, eigen_codes,
                                                       eigen_device, capture_stream);
  }
  Gfn2EigensolverLaunchResult sequence{};
  if (status == cudaSuccess) {
    sequence = prepare_gfn2_eigensolver_launch_sequence_cuda(
        plan.eigensolver_batch, workspace.eigensolver_workspace, eigen_device, capture_stream);
    status = sequence.cuda_status;
  }
  Gfn2EigensolverLaunchResult compact{};
  if (status == cudaSuccess) {
    compact = state.dynamic_geometry
                  ? prepare_and_compact_gfn2_solve_buckets_cuda(
                        plan.eigensolver_batch, buckets, bucket_count, plan.overlap_cache,
                        state.geometry.epoch, binding.input.eigensolver_hamiltonians,
                        plan.eigensolver_options, workspace.eigensolver_workspace, eigen_codes,
                        eigen_device, capture_stream)
                  : prepare_and_compact_gfn2_solve_buckets_cuda(
                        plan.eigensolver_batch, buckets, bucket_count, plan.overlap_cache,
                        plan.geometry_generation, binding.input.eigensolver_hamiltonians,
                        plan.eigensolver_options, workspace.eigensolver_workspace, eigen_codes,
                        eigen_device, capture_stream);
    status = compact.cuda_status;
  }
  if (status != cudaSuccess) {
    return chain_fallback(Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed, status);
  }
  Gfn2SccDispatchChainDevice chain{};
  chain.table = state.device_table;
  chain.table_slots = table_slots;
  chain.pre_slot = 0;
  chain.post_slot = 1;
  chain.bucket_count = static_cast<std::int32_t>(bucket_count);
  dispatch_scc_eigensolver_kernel<<<1, 1, 0, capture_stream>>>(
      workspace.ledger, state.control, chain, workspace.eigensolver_workspace.bucket_activity, 0,
      capacity[0], eig_base[0], back_base[0]);
  status = check_kernel_launch();
  if (status != cudaSuccess) {
    return chain_fallback(Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed, status);
  }
  status = cudaStreamEndCapture(capture_stream, &pre_graph);
  if (status != cudaSuccess || pre_graph == nullptr) {
    return chain_fallback(Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed,
                          status == cudaSuccess ? cudaErrorStreamCaptureInvalidated : status);
  }
  state.graph_owns.push_back(pre_graph);
  cudaGraphExec_t pre_exec = nullptr;
  status = cudaGraphInstantiate(&pre_exec, pre_graph, cudaGraphInstantiateFlagDeviceLaunch);
  if (status != cudaSuccess) {
    return chain_fallback(Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed, status);
  }
  state.executables.push_back(pre_exec);
  host_table[0] = pre_exec;

  /* ---- eig + back executables per bucket ---- */
  const double* const hamiltonians_device = binding.input.eigensolver_hamiltonians;
  for (std::int64_t b = 0; b < bucket_count; ++b) {
    const Gfn2EigensolverBucket bucket = buckets[b];
    const std::int64_t cap = capacity[static_cast<std::size_t>(b)];
    for (std::int64_t c = 1; c <= cap; ++c) {
      cudaGraph_t eig_graph = nullptr;
      status = cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);
      if (status != cudaSuccess) {
        return chain_fallback(Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed, status);
      }
      Gfn2EigensolverLaunchResult captured = enqueue_gfn2_eigensolver_capacity_cuda(
          capture_stream, static_cast<std::uint32_t>(c), plan.eigensolver_batch, bucket, b,
          plan.overlap_cache, hamiltonians_device, plan.eigensolver_provider.solver,
          plan.eigensolver_provider.parameters, plan.eigensolver_provider.blas,
          workspace.eigensolver_workspace, eigen_codes, eigen_device, plan.eigensolver_options);
      if (captured.success()) {
        captured = compact_gfn2_successful_eigenpair_counts_cuda(
            bucket, b, workspace.eigensolver_workspace, eigen_codes, eigen_device, capture_stream);
      }
      if (!captured.success()) {
        finish_stream();
        destroy_dispatch_chain_resources(state);
        return fallback_graph_build(state, Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed,
                                    static_cast<cudaError_t>(captured.cuda_status));
      }
      dispatch_scc_backtransform_kernel<<<1, 1, 0, capture_stream>>>(
          workspace.ledger, state.control, chain, workspace.eigensolver_workspace.bucket_activity,
          b, cap, back_base[static_cast<std::size_t>(b)]);
      status = check_kernel_launch();
      if (status != cudaSuccess) {
        finish_stream();
        destroy_dispatch_chain_resources(state);
        return fallback_graph_build(state, Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed,
                                    status);
      }
      status = cudaStreamEndCapture(capture_stream, &eig_graph);
      if (status != cudaSuccess || eig_graph == nullptr) {
        finish_stream();
        destroy_dispatch_chain_resources(state);
        return fallback_graph_build(
            state, Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed,
            status == cudaSuccess ? cudaErrorStreamCaptureInvalidated : status);
      }
      state.graph_owns.push_back(eig_graph);
      cudaGraphExec_t eig_exec = nullptr;
      status = cudaGraphInstantiate(&eig_exec, eig_graph, cudaGraphInstantiateFlagDeviceLaunch);
      if (status != cudaSuccess) {
        finish_stream();
        destroy_dispatch_chain_resources(state);
        return fallback_graph_build(state, Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed,
                                    status);
      }
      state.executables.push_back(eig_exec);
      host_table[static_cast<std::size_t>(eig_base[static_cast<std::size_t>(b)] + c - 1)] =
          eig_exec;
    }
    for (std::int64_t c = 0; c <= cap; ++c) {
      cudaGraph_t back_graph = nullptr;
      status = cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);
      if (status != cudaSuccess) {
        return chain_fallback(Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed, status);
      }
      if (c > 0) {
        Gfn2EigensolverLaunchResult captured = enqueue_gfn2_backtransform_capacity_cuda(
            capture_stream, static_cast<std::uint32_t>(c), plan.eigensolver_batch, bucket, b,
            plan.eigensolver_provider.blas, workspace.eigensolver_workspace,
            workspace.staged_eigenpairs, eigen_codes, eigen_device,
            plan.eigensolver_options.deterministic_debug);
        if (!captured.success()) {
          finish_stream();
          destroy_dispatch_chain_resources(state);
          return fallback_graph_build(state, Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed,
                                      static_cast<cudaError_t>(captured.cuda_status));
        }
      }
      if (b + 1 < bucket_count) {
        dispatch_scc_eigensolver_kernel<<<1, 1, 0, capture_stream>>>(
            workspace.ledger, state.control, chain, workspace.eigensolver_workspace.bucket_activity,
            b + 1, capacity[static_cast<std::size_t>(b + 1)],
            eig_base[static_cast<std::size_t>(b + 1)], back_base[static_cast<std::size_t>(b + 1)]);
      } else {
        dispatch_scc_eigensolver_kernel<<<1, 1, 0, capture_stream>>>(
            workspace.ledger, state.control, chain, workspace.eigensolver_workspace.bucket_activity,
            b + 1, 0, 0, 0);
      }
      status = check_kernel_launch();
      if (status != cudaSuccess) {
        finish_stream();
        destroy_dispatch_chain_resources(state);
        return fallback_graph_build(state, Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed,
                                    status);
      }
      status = cudaStreamEndCapture(capture_stream, &back_graph);
      if (status != cudaSuccess || back_graph == nullptr) {
        finish_stream();
        destroy_dispatch_chain_resources(state);
        return fallback_graph_build(
            state, Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed,
            status == cudaSuccess ? cudaErrorStreamCaptureInvalidated : status);
      }
      state.graph_owns.push_back(back_graph);
      cudaGraphExec_t back_exec = nullptr;
      status = cudaGraphInstantiate(&back_exec, back_graph, cudaGraphInstantiateFlagDeviceLaunch);
      if (status != cudaSuccess) {
        finish_stream();
        destroy_dispatch_chain_resources(state);
        return fallback_graph_build(state, Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed,
                                    status);
      }
      state.executables.push_back(back_exec);
      host_table[static_cast<std::size_t>(back_base[static_cast<std::size_t>(b)] + c)] = back_exec;
    }
  }

  /* ---- post_exec ---- */
  cudaGraph_t post_graph = nullptr;
  status = cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);
  if (status != cudaSuccess) {
    return chain_fallback(Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed, status);
  }
  status = normalize_gfn2_scc_stage_cuda(*eigen_report, workspace.ledger, capture_stream);
  const Gfn2SccIterationLaunchResult post_segment =
      status == cudaSuccess ? launch_graph_post_eigensolver(state, capture_stream)
                            : Gfn2SccIterationLaunchResult{};
  if (status == cudaSuccess && !post_segment.success()) {
    if (post_segment.status == Gfn2SccIterationLaunchStatus::kInvalidBinding) {
      finish_stream();
      return invalid_graph_binding(post_segment);
    }
    return chain_fallback(Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed,
                          post_segment.cuda_status, post_segment);
  }
  if (status == cudaSuccess) status = diagnostic_finish(state, true, capture_stream);
  if (status == cudaSuccess) {
    snapshot_device_loop_failure_kernel<<<1, 1, 0, capture_stream>>>(workspace.ledger,
                                                                     state.control);
    status = check_kernel_launch();
  }
  const Gfn2SccIterationLaunchResult next_root = status == cudaSuccess
                                                     ? launch_graph_activity(state, capture_stream)
                                                     : Gfn2SccIterationLaunchResult{};
  if (status == cudaSuccess && !next_root.success()) {
    if (next_root.status == Gfn2SccIterationLaunchStatus::kInvalidBinding) {
      finish_stream();
      return invalid_graph_binding(next_root);
    }
    return chain_fallback(Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed,
                          next_root.cuda_status, next_root);
  }
  if (status == cudaSuccess) {
    launch_scc_chain_tail_kernel<<<1, 1, 0, capture_stream>>>(workspace.ledger, state.control,
                                                              chain);
    status = check_kernel_launch();
  }
  if (status != cudaSuccess) {
    return chain_fallback(Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed, status);
  }
  status = cudaStreamEndCapture(capture_stream, &post_graph);
  if (status != cudaSuccess || post_graph == nullptr) {
    return chain_fallback(Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed,
                          status == cudaSuccess ? cudaErrorStreamCaptureInvalidated : status);
  }
  state.graph_owns.push_back(post_graph);
  cudaGraphExec_t post_exec = nullptr;
  status = cudaGraphInstantiate(&post_exec, post_graph, cudaGraphInstantiateFlagDeviceLaunch);
  if (status != cudaSuccess) {
    return chain_fallback(Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed, status);
  }
  state.executables.push_back(post_exec);
  host_table[1] = post_exec;

  /* ---- Device table upload + executable uploads ---- */
  status = cudaMemcpyAsync(state.device_table, host_table.data(),
                           host_table.size() * sizeof(cudaGraphExec_t), cudaMemcpyHostToDevice,
                           capture_stream);
  if (status == cudaSuccess) {
    for (cudaGraphExec_t executable : state.executables) {
      status = cudaGraphUpload(executable, capture_stream);
      if (status != cudaSuccess) {
        break;
      }
    }
  }
  if (status == cudaSuccess) {
    status = cudaStreamSynchronize(capture_stream);
  }
  if (status != cudaSuccess) {
    return chain_fallback(Gfn2SccLoopGraphFallbackReason::kDeviceGraphUploadFailed, status);
  }

  /* ---- root graph ---- */
  status = cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);
  if (status != cudaSuccess) {
    return chain_fallback(Gfn2SccLoopGraphFallbackReason::kRootCaptureFailed, status);
  }
  reset_device_loop_control_kernel<<<1, 1, 0, capture_stream>>>(state.control);
  status = check_kernel_launch();
  if (status == cudaSuccess) {
    status =
        diagnostic_reset(state, capture_stream, Gfn2SccLoopExecutionMode::kDeviceDispatchChain);
  }
  const Gfn2SccIterationLaunchResult root = status == cudaSuccess
                                                ? launch_graph_activity(state, capture_stream)
                                                : Gfn2SccIterationLaunchResult{};
  if (status == cudaSuccess && !root.success()) {
    if (root.status == Gfn2SccIterationLaunchStatus::kInvalidBinding) {
      finish_stream();
      return invalid_graph_binding(root);
    }
    return chain_fallback(Gfn2SccLoopGraphFallbackReason::kRootCaptureFailed, root.cuda_status,
                          root);
  }
  if (status == cudaSuccess) {
    gate_device_loop_kernel<<<1, 1, 0, capture_stream>>>(workspace.ledger, state.control, pre_exec);
    status = check_kernel_launch();
  }
  if (status != cudaSuccess) {
    return chain_fallback(Gfn2SccLoopGraphFallbackReason::kRootCaptureFailed, status);
  }
  status = cudaStreamEndCapture(capture_stream, &state.root_graph);
  if (status != cudaSuccess || state.root_graph == nullptr) {
    return chain_fallback(Gfn2SccLoopGraphFallbackReason::kRootCaptureFailed,
                          status == cudaSuccess ? cudaErrorStreamCaptureInvalidated : status);
  }
  status = cudaGraphInstantiate(&state.root_executable, state.root_graph, 0u);
  if (status != cudaSuccess) {
    return chain_fallback(Gfn2SccLoopGraphFallbackReason::kInstantiationFailed, status);
  }
  (void)cudaStreamDestroy(capture_stream);
  capture_stream = nullptr;

  state.dispatch_chain_ready = true;
  state.fallback_reason = Gfn2SccLoopGraphFallbackReason::kNone;
  Gfn2SccLoopGraphBuildResult result{};
  result.status = Gfn2SccLoopGraphBuildStatus::kDeviceDispatchChainReady;
  return result;
} catch (const std::bad_alloc&) {
  destroy_dispatch_chain_resources(state);
  return fallback_graph_build(state, Gfn2SccLoopGraphFallbackReason::kDispatchBuildFailed,
                              cudaErrorMemoryAllocation);
}

#endif  // CUDART_VERSION >= 12080

}  // namespace

Gfn2SccLoopLaunchResult launch_gfn2_restricted_scc_loop_cuda(const Gfn2SccIterationBinding& binding,
                                                             cudaStream_t stream) noexcept {
  return launch_restricted_scc_loop_impl(binding, nullptr, stream);
}

Gfn2SccLoopLaunchResult launch_gfn2_restricted_scc_loop_cuda(
    const Gfn2SccIterationBinding& binding, const Gfn2GeometryEpochConsumerDevice& geometry,
    cudaStream_t stream) noexcept {
  return launch_restricted_scc_loop_impl(binding, &geometry, stream);
}

Gfn2SccLoopLaunchResult launch_gfn2_restricted_scc_loop_cuda(
    const Gfn2SccIterationBinding& binding, const Gfn2GeometryEpochConsumerDevice& geometry,
    Gfn2SccDiagnosticsDevice diagnostics, cudaStream_t stream) noexcept {
  return launch_restricted_scc_loop_impl(binding, &geometry, stream, diagnostics);
}

Gfn2SccLoopCudaGraphOwner::~Gfn2SccLoopCudaGraphOwner() { reset(); }

Gfn2SccLoopGraphBuildResult Gfn2SccLoopCudaGraphOwner::build(
    const Gfn2SccIterationBinding& binding) noexcept {
  return build_impl(binding, nullptr, Gfn2SccLoopGraphPreference::kAuto);
}

Gfn2SccLoopGraphBuildResult Gfn2SccLoopCudaGraphOwner::build(
    const Gfn2SccIterationBinding& binding,
    const Gfn2GeometryEpochConsumerDevice& geometry) noexcept {
  return build_impl(binding, &geometry, Gfn2SccLoopGraphPreference::kAuto);
}

Gfn2SccLoopGraphBuildResult Gfn2SccLoopCudaGraphOwner::build(
    const Gfn2SccIterationBinding& binding, Gfn2SccLoopGraphPreference preference) noexcept {
  return build_impl(binding, nullptr, preference);
}

Gfn2SccLoopGraphBuildResult Gfn2SccLoopCudaGraphOwner::build(
    const Gfn2SccIterationBinding& binding, const Gfn2GeometryEpochConsumerDevice& geometry,
    Gfn2SccLoopGraphPreference preference) noexcept {
  return build_impl(binding, &geometry, preference);
}

Gfn2SccLoopGraphBuildResult Gfn2SccLoopCudaGraphOwner::build_impl(
    const Gfn2SccIterationBinding& binding, const Gfn2GeometryEpochConsumerDevice* geometry,
    Gfn2SccLoopGraphPreference preference) noexcept {
  reset();
  const Gfn2SccLoopLaunchResult validation = validate_loop_plan(binding);
  if (!validation.success()) {
    return invalid_graph_binding(validation.iteration);
  }

  auto* const state = new (std::nothrow) State;
  if (state == nullptr) {
    Gfn2SccLoopGraphBuildResult result{};
    result.status = Gfn2SccLoopGraphBuildStatus::kInvalidBinding;
    result.iteration.status = Gfn2SccIterationLaunchStatus::kCudaError;
    result.iteration.cuda_status = cudaErrorMemoryAllocation;
    return result;
  }
  state->binding = binding;
  state->preference = preference;
  if (geometry != nullptr) {
    state->geometry = *geometry;
    state->dynamic_geometry = true;
  }
  state_ = state;
  const cudaError_t diagnostic_status = allocate_diagnostics(*state);
  if (diagnostic_status != cudaSuccess) {
    Gfn2SccLoopGraphBuildResult result{};
    result.iteration.status = Gfn2SccIterationLaunchStatus::kCudaError;
    result.iteration.cuda_status = diagnostic_status;
    result.cuda_status = diagnostic_status;
    reset();
    return result;
  }

#if CUDART_VERSION >= 12030
#if CUDART_VERSION >= 12080
  /* A singleton batch can never expose a partial nonzero provider capacity, so
   * the measured dispatch overhead has no compensating compaction benefit.
   * Forced preferences remain exact; only kAuto applies this crossover. */
  const bool dispatch_requested = preference == Gfn2SccLoopGraphPreference::kDeviceDispatchChain;
  const bool dispatch_beneficial = preference == Gfn2SccLoopGraphPreference::kAuto &&
                                   binding.plan.topology.batch_size > 1 &&
                                   gfn2_scc_dispatch_chain_regime_applies(binding.plan);
  if (dispatch_requested || dispatch_beneficial) {
    Gfn2SccLoopGraphBuildResult dispatch = build_dispatch_chain(*state);
    if (dispatch.status == Gfn2SccLoopGraphBuildStatus::kDeviceDispatchChainReady) {
      return dispatch;
    }
    if (dispatch.status == Gfn2SccLoopGraphBuildStatus::kInvalidBinding) {
      reset();
      return dispatch;
    }
    if (preference == Gfn2SccLoopGraphPreference::kDeviceDispatchChain) {
      /* A forced dispatch-chain build that could not produce a chain returns
       * its bounded-fallback result directly instead of upgrading to the
       * monolithic device-tail graph, so the caller's requested family is
       * observable rather than silently replaced. */
      return dispatch;
    }
    /* dispatch is a bounded-fallback build; retain its fallback reason and fall
     * through to the monolithic device-tail graph. destroy_graph_state later
     * frees any partially built dispatch-chain resources. */
  }
#else
  if (preference == Gfn2SccLoopGraphPreference::kDeviceDispatchChain) {
    return fallback_graph_build(*state,
                                Gfn2SccLoopGraphFallbackReason::kDeviceGraphLaunchUnavailable,
                                cudaErrorNotSupported);
  }
#endif
  Gfn2SccLoopGraphBuildResult result = build_device_tail_graph(*state);
  if (result.status == Gfn2SccLoopGraphBuildStatus::kInvalidBinding) {
    reset();
  }
  if (state_ == nullptr) {
    return result;
  }
  if (result.device_tail_graph_ready()) {
    /* The monolithic device-tail graph is a healthy production mode, not a
     * fallback. Clear any reason latched by the earlier kAuto dispatch-chain
     * attempt (for example kDispatchUnsupportedLayout on a mixed-spin batch)
     * so the owner never misreports a release-visible fallback. */
    state_->fallback_reason = Gfn2SccLoopGraphFallbackReason::kNone;
  } else if (state_->fallback_reason == Gfn2SccLoopGraphFallbackReason::kNone) {
    state_->fallback_reason = result.fallback_reason;
  }
  return result;
#else
  return fallback_graph_build(*state, Gfn2SccLoopGraphFallbackReason::kDeviceGraphLaunchUnavailable,
                              cudaErrorNotSupported);
#endif
}

Gfn2SccLoopLaunchResult Gfn2SccLoopCudaGraphOwner::launch(cudaStream_t stream) const noexcept {
  if (state_ == nullptr) {
    return reject_loop_binding(Gfn2SccIterationBindingError::kInvalidPlanToken);
  }
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  const cudaError_t capture_query = cudaStreamIsCapturing(stream, &capture_status);
  if (capture_query != cudaSuccess) {
    Gfn2SccLoopLaunchResult result{};
    result.iteration.status = Gfn2SccIterationLaunchStatus::kCudaError;
    result.iteration.cuda_status = capture_query;
    return result;
  }
  /* Capturing the root Graph would make the outer executable depend on this
   * owner's body executable and control allocation. Capture the bounded DAG
   * instead so the outer executable remains valid after owner reset while the
   * caller-owned binding storage remains alive. */
#ifdef XTBLOOM_CUDA_SCC_DIAGNOSTICS
  state_->external_capture_uninstrumented = capture_status != cudaStreamCaptureStatusNone;
#endif
  if (state_->root_executable == nullptr || capture_status != cudaStreamCaptureStatusNone) {
    /* The independently captured DAG must not borrow this owner's ledger.
     * Runtime-owned captures use the explicit diagnostic overload instead and
     * retain the complete Prepared owner until their Graph is destroyed. */
    const auto diagnostics = capture_status == cudaStreamCaptureStatusNone
                                 ? state_->diagnostics
                                 : Gfn2SccDiagnosticsDevice{};
    Gfn2SccLoopLaunchResult result = launch_restricted_scc_loop_impl(
        state_->binding, state_->dynamic_geometry ? &state_->geometry : nullptr, stream,
        diagnostics);
    result.execution_mode = Gfn2SccLoopExecutionMode::kBoundedFallback;
    return result;
  }
  Gfn2SccLoopLaunchResult result{};
  result.execution_mode = state_->dispatch_chain_ready
                              ? Gfn2SccLoopExecutionMode::kDeviceDispatchChain
                              : Gfn2SccLoopExecutionMode::kDeviceTailGraph;
  const cudaError_t status = cudaGraphLaunch(state_->root_executable, stream);
  if (status != cudaSuccess) {
    result.iteration.status = Gfn2SccIterationLaunchStatus::kCudaError;
    result.iteration.cuda_status = status;
    return result;
  }
  result.submitted_graphs = 1u;
  return result;
}

Gfn2SccLoopLaunchResult Gfn2SccLoopCudaGraphOwner::append_root_child_to_capture(
    cudaStream_t stream) const noexcept {
  if (state_ == nullptr || state_->root_graph == nullptr || state_->root_executable == nullptr) {
    return reject_loop_binding(Gfn2SccIterationBindingError::kInvalidPlanToken);
  }

  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  cudaGraph_t captured_graph = nullptr;
  const cudaGraphNode_t* dependencies = nullptr;
  std::size_t dependency_count = 0u;
  cudaError_t status = cudaStreamGetCaptureInfo(stream, &capture_status, nullptr, &captured_graph,
                                                &dependencies, &dependency_count);
  if (status != cudaSuccess || capture_status == cudaStreamCaptureStatusNone ||
      captured_graph == nullptr || dependencies == nullptr || dependency_count == 0u) {
    Gfn2SccLoopLaunchResult result{};
    result.iteration.status = Gfn2SccIterationLaunchStatus::kCudaError;
    result.iteration.cuda_status =
        status == cudaSuccess ? cudaErrorStreamCaptureInvalidated : status;
    return result;
  }

  cudaGraphNode_t child = nullptr;
  Gfn2SccLoopLaunchResult result =
      append_root_child(captured_graph, dependencies, dependency_count, child);
  status = result.success() ? cudaSuccess : result.iteration.cuda_status;
  if (result.success()) {
    status =
        cudaStreamUpdateCaptureDependencies(stream, &child, 1u, cudaStreamSetCaptureDependencies);
  }
  if (status != cudaSuccess) {
    result.iteration.status = Gfn2SccIterationLaunchStatus::kCudaError;
    result.iteration.cuda_status = status;
    return result;
  }
  result.submitted_graphs = 1u;
  return result;
}

Gfn2SccLoopLaunchResult Gfn2SccLoopCudaGraphOwner::append_root_child(
    cudaGraph_t graph, const cudaGraphNode_t* dependencies, std::size_t dependency_count,
    cudaGraphNode_t& child) const noexcept {
  child = nullptr;
  if (state_ == nullptr || state_->root_graph == nullptr || state_->root_executable == nullptr) {
    return reject_loop_binding(Gfn2SccIterationBindingError::kInvalidPlanToken);
  }
  Gfn2SccLoopLaunchResult result{};
  result.execution_mode = state_->dispatch_chain_ready
                              ? Gfn2SccLoopExecutionMode::kDeviceDispatchChain
                              : Gfn2SccLoopExecutionMode::kDeviceTailGraph;
  cudaGraphNodeParams parameters{};
  parameters.type = cudaGraphNodeTypeGraph;
  parameters.graph.graph = state_->root_graph;
  const cudaError_t status =
      cudaGraphAddNode(&child, graph, dependencies, dependency_count, &parameters);
  if (status != cudaSuccess) {
    result.iteration.status = Gfn2SccIterationLaunchStatus::kCudaError;
    result.iteration.cuda_status = status;
    return result;
  }
  result.submitted_graphs = 1u;
  return result;
}

void Gfn2SccLoopCudaGraphOwner::reset() noexcept {
  destroy_graph_state(state_);
  state_ = nullptr;
}

bool Gfn2SccLoopCudaGraphOwner::ready() const noexcept { return state_ != nullptr; }

bool Gfn2SccLoopCudaGraphOwner::device_dispatch_chain_ready() const noexcept {
  return state_ != nullptr && state_->dispatch_chain_ready && state_->root_executable != nullptr &&
         state_->device_table != nullptr;
}

bool Gfn2SccLoopCudaGraphOwner::conditional_graph_ready() const noexcept {
  return device_tail_graph_ready() || device_dispatch_chain_ready();
}

bool Gfn2SccLoopCudaGraphOwner::device_tail_graph_ready() const noexcept {
  return state_ != nullptr && state_->root_executable != nullptr &&
         state_->body_executable != nullptr;
}

Gfn2SccLoopGraphFallbackReason Gfn2SccLoopCudaGraphOwner::fallback_reason() const noexcept {
  return state_ == nullptr ? Gfn2SccLoopGraphFallbackReason::kNone : state_->fallback_reason;
}

const std::uint32_t* Gfn2SccLoopCudaGraphOwner::canonical_active_count_device() const noexcept {
  return state_ == nullptr || state_->control == nullptr ? nullptr
                                                         : &state_->control->canonical_active_count;
}

const std::uint64_t* Gfn2SccLoopCudaGraphOwner::numerical_body_count_device() const noexcept {
  return state_ == nullptr || state_->control == nullptr ? nullptr
                                                         : &state_->control->numerical_body_count;
}

const std::uint32_t* Gfn2SccLoopCudaGraphOwner::device_launch_error_device() const noexcept {
  return state_ == nullptr || state_->control == nullptr ? nullptr
                                                         : &state_->control->device_launch_error;
}

std::size_t Gfn2SccLoopCudaGraphOwner::retained_device_bytes() const noexcept {
  if (state_ == nullptr) {
    return 0u;
  }
  std::size_t bytes = state_->control != nullptr ? sizeof(Gfn2SccDeviceLoopControl) : 0u;
  if (state_->device_table != nullptr) {
    bytes += static_cast<std::size_t>(state_->table_slots) * sizeof(cudaGraphExec_t);
  }
  const auto diagnostics = state_->diagnostics;
  if (diagnostics.header != nullptr) {
    bytes +=
        sizeof(Gfn2SccDiagnosticHeader) +
        diagnostics.maximum_iterations * sizeof(Gfn2SccDiagnosticIteration) +
        diagnostics.bucket_count * sizeof(Gfn2EigensolverBucket) +
        diagnostics.maximum_iterations * diagnostics.bucket_count * sizeof(Gfn2SccDiagnosticBucket);
  }
  return bytes;
}

Gfn2SccDiagnosticsDevice Gfn2SccLoopCudaGraphOwner::diagnostics_device() const noexcept {
  return state_ == nullptr ? Gfn2SccDiagnosticsDevice{} : state_->diagnostics;
}

cudaError_t Gfn2SccLoopCudaGraphOwner::write_diagnostics_json(std::ostream& output,
                                                              cudaStream_t stream,
                                                              std::uint32_t start_policy) const {
#ifndef XTBLOOM_CUDA_SCC_DIAGNOSTICS
  return cudaErrorNotSupported;
#else
  const auto diagnostics = diagnostics_device();
  if (diagnostics.header == nullptr || state_->external_capture_uninstrumented) {
    return cudaErrorNotSupported;
  }
  Gfn2SccDiagnosticHeader header{};
  std::vector<Gfn2SccDiagnosticIteration> iterations(diagnostics.maximum_iterations);
  std::vector<Gfn2SccDiagnosticBucket> rows(diagnostics.maximum_iterations *
                                            diagnostics.bucket_count);
  const auto systems = static_cast<std::size_t>(state_->binding.plan.topology.batch_size);
  std::vector<std::uint64_t> terminal_iterations(systems);
  std::vector<xtbloom_status_t> terminal_statuses(systems);
  std::vector<std::uint8_t> terminal_converged(systems);
  std::vector<std::int32_t> bucket_systems(systems);
  cudaError_t status =
      cudaMemcpyAsync(&header, diagnostics.header, sizeof(header), cudaMemcpyDeviceToHost, stream);
  if (status == cudaSuccess) {
    status =
        cudaMemcpyAsync(iterations.data(), diagnostics.iterations,
                        iterations.size() * sizeof(iterations[0]), cudaMemcpyDeviceToHost, stream);
  }
  if (status == cudaSuccess && !rows.empty()) {
    status = cudaMemcpyAsync(rows.data(), diagnostics.rows, rows.size() * sizeof(rows[0]),
                             cudaMemcpyDeviceToHost, stream);
  }
  const auto copy_terminal = [&](void* destination, const void* source, std::size_t bytes) {
    if (status == cudaSuccess && bytes != 0u) {
      status = cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToHost, stream);
    }
  };
  const auto terminal = state_->binding.input.activity_state;
  copy_terminal(terminal_iterations.data(), terminal.iterations,
                systems * sizeof(terminal_iterations[0]));
  copy_terminal(terminal_statuses.data(), terminal.system_statuses,
                systems * sizeof(terminal_statuses[0]));
  copy_terminal(terminal_converged.data(), terminal.converged,
                systems * sizeof(terminal_converged[0]));
  copy_terminal(bucket_systems.data(), state_->binding.plan.eigensolver_batch.bucket_systems,
                systems * sizeof(bucket_systems[0]));
  /* Settle every accepted readback even after a later copy fails, so local
   * destination lifetimes never end with an outstanding transfer. */
  const cudaError_t settled = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) return status;
  if (settled != cudaSuccess) return settled;
  if (header.overflow != 0u || header.iteration_count > diagnostics.maximum_iterations ||
      header.execution_mode > 2u) {
    return cudaErrorInvalidValue;
  }
  /* A corrupt readback must not index outside canonical state or emit a
   * partially valid trace. The downloaded bucket map is a complete permutation,
   * including peers that failed before executing a numerical body. */
  std::vector<bool> seen_systems(systems, false);
  for (std::int64_t bucket_index = 0; bucket_index < diagnostics.bucket_count; ++bucket_index) {
    const auto bucket = state_->binding.plan.eigensolver_provider.buckets[bucket_index];
    if (bucket.system_count < 0 || bucket.system_index_offset < 0 ||
        static_cast<std::uint64_t>(bucket.system_index_offset) > systems ||
        static_cast<std::uint64_t>(bucket.system_count) >
            systems - static_cast<std::uint64_t>(bucket.system_index_offset)) {
      return cudaErrorInvalidValue;
    }
    for (std::int32_t local = 0; local < bucket.system_count; ++local) {
      const auto system = bucket_systems[bucket.system_index_offset + local];
      if (system < 0 || static_cast<std::size_t>(system) >= systems || seen_systems[system]) {
        return cudaErrorInvalidValue;
      }
      seen_systems[system] = true;
    }
  }
  if (std::find(seen_systems.begin(), seen_systems.end(), false) != seen_systems.end()) {
    return cudaErrorInvalidValue;
  }
  const char* mode = header.execution_mode == 2u   ? "device_dispatch_chain"
                     : header.execution_mode == 1u ? "device_tail_graph"
                                                   : "bounded_fallback";
  output << "{\"schema\":\"xtbloom.cuda.scc_diagnostics.v1\",\"instrumented\":true,"
            "\"execution_mode\":\""
         << mode << "\",\"fallback_reason\":"
         << static_cast<std::uint32_t>(header.execution_mode == 0u && conditional_graph_ready()
                                           ? Gfn2SccLoopGraphFallbackReason::kRuntimeBoundedOverride
                                           : fallback_reason())
         << ",\"batch_size\":" << state_->binding.plan.topology.batch_size
         << ",\"maximum_iterations\":" << diagnostics.maximum_iterations
         << ",\"plan_token\":" << state_->binding.plan.plan_token
         << ",\"wavefunction_layout_fingerprint\":"
         << state_->binding.plan.wavefunction_layout.layout_fingerprint
         << ",\"start_policy\":" << start_policy
         << ",\"graph_preference\":" << static_cast<std::uint32_t>(state_->preference)
         << ",\"chain_measured_ao_bound\":" << kGfn2SccDispatchChainMeasuredOrbitalBound
         << ",\"diagnostic_device_bytes\":"
         << sizeof(Gfn2SccDiagnosticHeader) + iterations.size() * sizeof(iterations[0]) +
                rows.size() * sizeof(rows[0]) +
                diagnostics.bucket_count * sizeof(Gfn2EigensolverBucket)
         << ",\"chain_selection_constraints\":[";
  bool constraint_written = false;
  const auto constraint = [&](const char* reason) {
    if (constraint_written) output << ',';
    output << '"' << reason << '"';
    constraint_written = true;
  };
  if (state_->preference == Gfn2SccLoopGraphPreference::kDeviceTailGraph) {
    constraint("forced_device_tail_graph");
  } else if (state_->preference == Gfn2SccLoopGraphPreference::kAuto) {
    if (state_->binding.plan.topology.batch_size <= 1) constraint("singleton_no_capacity_gain");
    if (!gfn2_scc_dispatch_chain_regime_applies(state_->binding.plan)) {
      constraint("outside_measured_ao_regime");
    }
    const auto channels = state_->binding.plan.wavefunction_layout.total_spin_channels;
    if (channels > 0 && channels != state_->binding.plan.topology.batch_size) {
      constraint("mixed_spin_dispatch_unsupported");
    }
  }
  output << "],\"buckets\":[";
  for (std::int64_t bucket_index = 0; bucket_index < diagnostics.bucket_count; ++bucket_index) {
    const auto bucket = state_->binding.plan.eigensolver_provider.buckets[bucket_index];
    if (bucket_index != 0) output << ',';
    output << "{\"bucket_index\":" << bucket_index << ",\"ao\":" << bucket.orbital_count
           << ",\"system_capacity\":" << bucket.system_count << ",\"channel_capacity\":"
           << (bucket.solve_count > 0 ? bucket.solve_count : bucket.system_count) << '}';
  }
  output << "],\"terminal_buckets\":[";
  for (std::int64_t bucket_index = 0; bucket_index < diagnostics.bucket_count; ++bucket_index) {
    const auto bucket = state_->binding.plan.eigensolver_provider.buckets[bucket_index];
    Gfn2SccDiagnosticBucket counts{};
    for (std::int32_t local = 0; local < bucket.system_count; ++local) {
      const auto system = bucket_systems[bucket.system_index_offset + local];
      const auto status_code = terminal_statuses[system];
      if (status_code == XTBLOOM_STATUS_SCC_NOT_CONVERGED ||
          (status_code == XTBLOOM_STATUS_SUCCESS && terminal_converged[system] == 0u &&
           terminal_iterations[system] >= diagnostics.maximum_iterations)) {
        ++counts.exhausted_systems;
      } else if (status_code != XTBLOOM_STATUS_SUCCESS) {
        ++counts.failed_systems;
      } else if (terminal_converged[system] != 0u) {
        ++counts.converged_systems;
      }
    }
    if (bucket_index != 0) output << ',';
    output << "{\"bucket_index\":" << bucket_index
           << ",\"converged_systems\":" << counts.converged_systems
           << ",\"failed_systems\":" << counts.failed_systems
           << ",\"exhausted_systems\":" << counts.exhausted_systems << ",\"unfinished_systems\":"
           << bucket.system_count - counts.converged_systems - counts.failed_systems -
                  counts.exhausted_systems
           << '}';
  }
  output << "],\"terminal_systems\":[";
  for (std::size_t system = 0u; system < systems; ++system) {
    if (system != 0u) output << ',';
    output << "{\"system_index\":" << system << ",\"iterations\":" << terminal_iterations[system]
           << ",\"status\":" << terminal_statuses[system]
           << ",\"converged\":" << static_cast<unsigned>(terminal_converged[system]) << '}';
  }
  output << "],\"iterations\":[";
  for (std::uint64_t iteration = 0u; iteration < header.iteration_count; ++iteration) {
    if (iteration != 0u) output << ',';
    const auto record = iterations[iteration];
    output << "{\"iteration\":" << iteration << ",\"start_ns\":" << record.start_ns
           << ",\"end_ns\":" << record.end_ns
           << ",\"plan_failure_record\":" << record.plan_failure_record << ",\"buckets\":[";
    for (std::int64_t bucket_index = 0; bucket_index < diagnostics.bucket_count; ++bucket_index) {
      if (bucket_index != 0) output << ',';
      const auto row = rows[iteration * diagnostics.bucket_count + bucket_index];
      output << "{\"bucket_index\":" << bucket_index << ",\"active_systems\":" << row.active_systems
             << ",\"active_channels\":" << row.active_channels;
      const auto slots = [&](const char* name, std::uint64_t count) {
        output << ",\"" << name << "\":";
        if (count == kGfn2SccDiagnosticUnknown)
          output << "null";
        else
          output << count;
      };
      slots("submitted_solver_slots", row.submitted_solver_slots);
      slots("submitted_backtransform_slots", row.submitted_backtransform_slots);
      output << ",\"converged_systems\":" << row.converged_systems
             << ",\"failed_systems\":" << row.failed_systems
             << ",\"exhausted_systems\":" << row.exhausted_systems << '}';
    }
    output << "]}";
  }
  output << "]}\n";
  return output.good() ? cudaSuccess : cudaErrorUnknown;
#endif
}

std::size_t Gfn2SccLoopCudaGraphOwner::dispatch_chain_executable_count() const noexcept {
  return state_ == nullptr ? 0u : state_->executables.size();
}

}  // namespace xtbloom::detail::cuda
