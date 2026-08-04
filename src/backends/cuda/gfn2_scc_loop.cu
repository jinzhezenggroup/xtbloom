#include "backends/cuda/gfn2_scc_loop.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <new>

namespace gpuxtb::detail::cuda {
namespace {

struct alignas(16) Gfn2SccConditionalLoopDeviceControl {
  std::uint32_t canonical_active_count = 0u;
  std::uint32_t reserved = 0u;
  std::uint64_t numerical_body_count = 0u;
  std::uint64_t plan_failure_snapshot = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2SccConditionalLoopDeviceControl>);
static_assert(std::is_standard_layout_v<Gfn2SccConditionalLoopDeviceControl>);

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
  const Gfn2SccIterationBindingDiagnostic diagnostic =
      validate_gfn2_scc_iteration_binding_cuda(binding.plan, binding.input, binding.state,
                                                binding.workspace);
  if (diagnostic.error != Gfn2SccIterationBindingError::kSuccess) {
    Gfn2SccLoopLaunchResult result{};
    result.iteration.status = Gfn2SccIterationLaunchStatus::kInvalidBinding;
    result.iteration.binding = diagnostic;
    return result;
  }
  return {};
}

Gfn2SccLoopLaunchResult launch_restricted_scc_loop_impl(
    const Gfn2SccIterationBinding& binding,
    const Gfn2GeometryEpochConsumerDevice* geometry,
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

__global__ void reset_conditional_loop_control_kernel(
    Gfn2SccConditionalLoopDeviceControl* control) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    control->canonical_active_count = 0u;
    control->numerical_body_count = 0u;
    control->plan_failure_snapshot = 0u;
  }
}

__global__ void count_conditional_loop_body_kernel(
    Gfn2SccConditionalLoopDeviceControl* control) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    ++control->numerical_body_count;
  }
}

__global__ void snapshot_conditional_loop_failure_kernel(
    Gfn2SccIterationDeviceLedger ledger, Gfn2SccConditionalLoopDeviceControl* control) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    control->plan_failure_snapshot = *ledger.plan_failure_record;
  }
}

/*
 * Count the canonical activity mask and publish that exact count to the WHILE
 * handle. At a numerical-body tail, a plan failure is snapshotted before the
 * next root derivation resets its ledger. Restoring the first record here both
 * preserves the diagnostic and forces the device loop to terminate even when
 * the failed iteration could not advance public state.
 */
__global__ void publish_conditional_loop_condition_kernel(
    Gfn2SccIterationDeviceLedger ledger, Gfn2SccConditionalLoopDeviceControl* control,
    cudaGraphConditionalHandle handle, int restore_snapshotted_failure) {
  __shared__ std::uint32_t active_count;
  __shared__ std::uint64_t failure_record;
  __shared__ int sequence_open;
  if (threadIdx.x == 0) {
    active_count = 0u;
    failure_record = restore_snapshotted_failure != 0 && control->plan_failure_snapshot != 0u
                         ? control->plan_failure_snapshot
                         : *ledger.plan_failure_record;
    sequence_open = failure_record == 0u && *ledger.sequence_active == 1u ? 1 : 0;
  }
  __syncthreads();

  for (std::int64_t system = threadIdx.x; system < ledger.batch_elements;
       system += blockDim.x) {
    if (sequence_open != 0 && ledger.active_mask[system] == 1u) {
      atomicAdd(&active_count, 1u);
    } else if (sequence_open == 0) {
      ledger.active_mask[system] = 0u;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    if (failure_record != 0u) {
      *ledger.plan_failure_record = failure_record;
      *ledger.sequence_active = 0u;
      active_count = 0u;
    }
    control->canonical_active_count = active_count;
    cudaGraphSetConditional(handle, active_count);
  }
}

cudaError_t check_kernel_launch() noexcept {
  return cudaPeekAtLastError();
}

void finish_or_abort_capture(cudaStream_t stream) noexcept {
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &capture_status) == cudaSuccess &&
      capture_status != cudaStreamCaptureStatusNone) {
    cudaGraph_t discarded = nullptr;
    (void)cudaStreamEndCapture(stream, &discarded);
    /* A graph created implicitly by a failed ordinary capture is caller-owned.
     * begin-capture-to-graph returns the supplied graph and must not be
     * destroyed separately here. */
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
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  Gfn2SccConditionalLoopDeviceControl* control = nullptr;
#if CUDART_VERSION >= 12030
  cudaGraphConditionalHandle handle = 0u;
#endif
};

namespace {

void destroy_graph_state(Gfn2SccLoopCudaGraphOwner::State* state) noexcept {
  if (state == nullptr) {
    return;
  }
  if (state->executable != nullptr) {
    (void)cudaGraphExecDestroy(state->executable);
  }
  if (state->graph != nullptr) {
    (void)cudaGraphDestroy(state->graph);
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
  if (state.executable != nullptr) {
    (void)cudaGraphExecDestroy(state.executable);
    state.executable = nullptr;
  }
  if (state.graph != nullptr) {
    (void)cudaGraphDestroy(state.graph);
    state.graph = nullptr;
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

Gfn2SccIterationLaunchResult launch_graph_activity(
    const Gfn2SccLoopCudaGraphOwner::State& state, cudaStream_t stream) noexcept {
  return state.dynamic_geometry
             ? launch_gfn2_restricted_scc_activity_cuda(state.binding, state.geometry, stream)
             : launch_gfn2_restricted_scc_activity_cuda(state.binding, stream);
}

Gfn2SccIterationLaunchResult launch_graph_numerical_body(
    const Gfn2SccLoopCudaGraphOwner::State& state, cudaStream_t stream) noexcept {
  return state.dynamic_geometry
             ? launch_gfn2_restricted_scc_numerical_body_cuda(state.binding, state.geometry,
                                                               stream)
             : launch_gfn2_restricted_scc_numerical_body_cuda(state.binding, stream);
}

Gfn2SccLoopGraphBuildResult build_conditional_graph(
    Gfn2SccLoopCudaGraphOwner::State& state) noexcept {
  if (state.binding.plan.eigensolver_provider.capture_mode !=
      Gfn2SccIterationProviderCaptureMode::kGraphSupported) {
    return fallback_graph_build(state,
                                Gfn2SccLoopGraphFallbackReason::kProviderCaptureUnsupported);
  }

  cudaError_t status = cudaGraphCreate(&state.graph, 0u);
  if (status == cudaSuccess) {
    status = cudaGraphConditionalHandleCreate(&state.handle, state.graph, 0u, 0u);
  }
  if (status != cudaSuccess) {
    return fallback_graph_build(
        state, Gfn2SccLoopGraphFallbackReason::kConditionalNodesUnavailable, status);
  }
  status = cudaMalloc(reinterpret_cast<void**>(&state.control),
                      sizeof(Gfn2SccConditionalLoopDeviceControl));
  if (status != cudaSuccess) {
    return fallback_graph_build(state,
                                Gfn2SccLoopGraphFallbackReason::kControlAllocationFailed, status);
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

  status = cudaStreamBeginCaptureToGraph(capture_stream, state.graph, nullptr, nullptr, 0u,
                                         cudaStreamCaptureModeThreadLocal);
  if (status != cudaSuccess) {
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kRootCaptureFailed, status);
  }
  reset_conditional_loop_control_kernel<<<1, 1, 0, capture_stream>>>(state.control);
  status = check_kernel_launch();
  const Gfn2SccIterationLaunchResult root =
      status == cudaSuccess ? launch_graph_activity(state, capture_stream)
                            : Gfn2SccIterationLaunchResult{};
  if (status == cudaSuccess && !root.success()) {
    if (root.status == Gfn2SccIterationLaunchStatus::kInvalidBinding) {
      finish_stream();
      return invalid_graph_binding(root);
    }
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kRootCaptureFailed,
                            root.cuda_status, root);
  }
  if (status == cudaSuccess) {
    publish_conditional_loop_condition_kernel<<<1, 256, 0, capture_stream>>>(
        state.binding.workspace.ledger, state.control, state.handle, 0);
    status = check_kernel_launch();
  }
  if (status != cudaSuccess) {
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kRootCaptureFailed, status);
  }

  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  cudaGraph_t captured_graph = nullptr;
  const cudaGraphNode_t* dependencies = nullptr;
  std::size_t dependency_count = 0u;
  status = cudaStreamGetCaptureInfo(capture_stream, &capture_status, nullptr, &captured_graph,
                                    &dependencies, &dependency_count);
  if (status != cudaSuccess || capture_status == cudaStreamCaptureStatusNone ||
      captured_graph != state.graph || dependencies == nullptr || dependency_count == 0u) {
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kRootCaptureFailed,
                            status == cudaSuccess ? cudaErrorStreamCaptureInvalidated : status);
  }

  cudaGraphNode_t conditional_node = nullptr;
  cudaGraphNodeParams conditional_params{};
  conditional_params.type = cudaGraphNodeTypeConditional;
  conditional_params.conditional.handle = state.handle;
  conditional_params.conditional.type = cudaGraphCondTypeWhile;
  conditional_params.conditional.size = 1u;
  status = cudaGraphAddNode(&conditional_node, state.graph, dependencies, dependency_count,
                            &conditional_params);
  if (status == cudaSuccess) {
    status = cudaStreamUpdateCaptureDependencies(
        capture_stream, &conditional_node, 1u, cudaStreamSetCaptureDependencies);
  }
  if (status != cudaSuccess) {
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kRootCaptureFailed, status);
  }
  cudaGraph_t ended_graph = nullptr;
  status = cudaStreamEndCapture(capture_stream, &ended_graph);
  if (status != cudaSuccess || ended_graph != state.graph) {
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kRootCaptureFailed,
                            status == cudaSuccess ? cudaErrorStreamCaptureInvalidated : status);
  }

  cudaGraph_t body_graph = conditional_params.conditional.phGraph_out[0];
  status = cudaStreamBeginCaptureToGraph(capture_stream, body_graph, nullptr, nullptr, 0u,
                                         cudaStreamCaptureModeThreadLocal);
  if (status != cudaSuccess) {
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kNumericalBodyCaptureFailed, status);
  }
  count_conditional_loop_body_kernel<<<1, 1, 0, capture_stream>>>(state.control);
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
    snapshot_conditional_loop_failure_kernel<<<1, 1, 0, capture_stream>>>(
        state.binding.workspace.ledger, state.control);
    status = check_kernel_launch();
  }
  const Gfn2SccIterationLaunchResult next_root =
      status == cudaSuccess ? launch_graph_activity(state, capture_stream)
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
    publish_conditional_loop_condition_kernel<<<1, 256, 0, capture_stream>>>(
        state.binding.workspace.ledger, state.control, state.handle, 1);
    status = check_kernel_launch();
  }
  if (status != cudaSuccess) {
    return capture_fallback(Gfn2SccLoopGraphFallbackReason::kNumericalBodyCaptureFailed, status);
  }
  ended_graph = nullptr;
  status = cudaStreamEndCapture(capture_stream, &ended_graph);
  if (status != cudaSuccess || ended_graph != body_graph) {
    return capture_fallback(
        Gfn2SccLoopGraphFallbackReason::kNumericalBodyCaptureFailed,
        status == cudaSuccess ? cudaErrorStreamCaptureInvalidated : status);
  }
  (void)cudaStreamDestroy(capture_stream);
  capture_stream = nullptr;

  status = cudaGraphInstantiate(&state.executable, state.graph, 0u);
  if (status != cudaSuccess) {
    return fallback_graph_build(state, Gfn2SccLoopGraphFallbackReason::kInstantiationFailed,
                                status);
  }

  Gfn2SccLoopGraphBuildResult result{};
  result.status = Gfn2SccLoopGraphBuildStatus::kConditionalGraphReady;
  return result;
}

#endif  // CUDART_VERSION >= 12030

}  // namespace

Gfn2SccLoopLaunchResult launch_gfn2_restricted_scc_loop_cuda(
    const Gfn2SccIterationBinding& binding, cudaStream_t stream) noexcept {
  return launch_restricted_scc_loop_impl(binding, nullptr, stream);
}

Gfn2SccLoopLaunchResult launch_gfn2_restricted_scc_loop_cuda(
    const Gfn2SccIterationBinding& binding,
    const Gfn2GeometryEpochConsumerDevice& geometry,
    cudaStream_t stream) noexcept {
  return launch_restricted_scc_loop_impl(binding, &geometry, stream);
}

Gfn2SccLoopCudaGraphOwner::~Gfn2SccLoopCudaGraphOwner() {
  reset();
}

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
  Gfn2SccLoopGraphBuildResult result = build_conditional_graph(*state);
  if (result.status == Gfn2SccLoopGraphBuildStatus::kInvalidBinding) {
    reset();
  }
  return result;
#else
  return fallback_graph_build(
      *state, Gfn2SccLoopGraphFallbackReason::kConditionalNodesUnavailable,
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
  /* CUDA graphs containing conditional nodes cannot themselves become child
   * graph nodes. Preserve the established whole-pipeline capture contract by
   * capturing the bounded path when the caller is already recording a graph. */
  if (state_->executable == nullptr || capture_status != cudaStreamCaptureStatusNone) {
    Gfn2SccLoopLaunchResult result =
        launch_restricted_scc_loop_impl(state_->binding,
                                        state_->dynamic_geometry ? &state_->geometry : nullptr,
                                        stream);
    result.execution_mode = Gfn2SccLoopExecutionMode::kBoundedFallback;
    return result;
  }
  Gfn2SccLoopLaunchResult result{};
  result.execution_mode = Gfn2SccLoopExecutionMode::kConditionalGraph;
  const cudaError_t status = cudaGraphLaunch(state_->executable, stream);
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

bool Gfn2SccLoopCudaGraphOwner::ready() const noexcept {
  return state_ != nullptr;
}

bool Gfn2SccLoopCudaGraphOwner::conditional_graph_ready() const noexcept {
  return state_ != nullptr && state_->executable != nullptr;
}

Gfn2SccLoopGraphFallbackReason Gfn2SccLoopCudaGraphOwner::fallback_reason() const noexcept {
  return state_ == nullptr ? Gfn2SccLoopGraphFallbackReason::kNone : state_->fallback_reason;
}

const std::uint32_t* Gfn2SccLoopCudaGraphOwner::canonical_active_count_device() const noexcept {
  return state_ == nullptr || state_->control == nullptr
             ? nullptr
             : &state_->control->canonical_active_count;
}

const std::uint64_t* Gfn2SccLoopCudaGraphOwner::numerical_body_count_device() const noexcept {
  return state_ == nullptr || state_->control == nullptr ? nullptr
                                                         : &state_->control->numerical_body_count;
}

}  // namespace gpuxtb::detail::cuda
