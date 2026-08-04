#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <new>

#include "backends/cuda/gfn2_scc_loop.cuh"

namespace gpuxtb::detail::cuda {
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
    cudaStream_t stream) noexcept {
  Gfn2SccLoopLaunchResult result = validate_loop_plan(binding);
  if (!result.success()) {
    return result;
  }

  const std::uint64_t submission_bound = binding.plan.activity_policy.maximum_iterations;
  for (std::uint64_t iteration = 0u; iteration < submission_bound; ++iteration) {
    result.iteration = geometry == nullptr
                           ? launch_gfn2_restricted_scc_iteration_cuda(binding, stream)
                           : launch_gfn2_restricted_scc_iteration_cuda(binding, *geometry, stream);
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
  cudaGraph_t root_graph = nullptr;
  cudaGraphExec_t root_executable = nullptr;
  cudaGraph_t body_graph = nullptr;
  cudaGraphExec_t body_executable = nullptr;
  Gfn2SccDeviceLoopControl* control = nullptr;
};

namespace {

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
  if (state->control != nullptr) {
    (void)cudaFree(state->control);
  }
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

Gfn2SccLoopCudaGraphOwner::~Gfn2SccLoopCudaGraphOwner() { reset(); }

Gfn2SccLoopGraphBuildResult Gfn2SccLoopCudaGraphOwner::build(
    const Gfn2SccIterationBinding& binding) noexcept {
  return build_impl(binding, nullptr);
}

Gfn2SccLoopGraphBuildResult Gfn2SccLoopCudaGraphOwner::build(
    const Gfn2SccIterationBinding& binding,
    const Gfn2GeometryEpochConsumerDevice& geometry) noexcept {
  return build_impl(binding, &geometry);
}

Gfn2SccLoopGraphBuildResult Gfn2SccLoopCudaGraphOwner::build_impl(
    const Gfn2SccIterationBinding& binding,
    const Gfn2GeometryEpochConsumerDevice* geometry) noexcept {
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
  if (geometry != nullptr) {
    state->geometry = *geometry;
    state->dynamic_geometry = true;
  }
  state_ = state;

#if CUDART_VERSION >= 12030
  Gfn2SccLoopGraphBuildResult result = build_device_tail_graph(*state);
  if (result.status == Gfn2SccLoopGraphBuildStatus::kInvalidBinding) {
    reset();
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
  if (state_->root_executable == nullptr || capture_status != cudaStreamCaptureStatusNone) {
    Gfn2SccLoopLaunchResult result = launch_restricted_scc_loop_impl(
        state_->binding, state_->dynamic_geometry ? &state_->geometry : nullptr, stream);
    result.execution_mode = Gfn2SccLoopExecutionMode::kBoundedFallback;
    return result;
  }
  Gfn2SccLoopLaunchResult result{};
  result.execution_mode = Gfn2SccLoopExecutionMode::kDeviceTailGraph;
  const cudaError_t status = cudaGraphLaunch(state_->root_executable, stream);
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

bool Gfn2SccLoopCudaGraphOwner::conditional_graph_ready() const noexcept {
  return device_tail_graph_ready();
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

}  // namespace gpuxtb::detail::cuda
