#ifndef XTBLOOM_RUNTIME_GFN2_CUDA_EXECUTION_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_RUNTIME_GFN2_CUDA_EXECUTION_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "runtime/request.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail {

/*
 * Opaque identity for one runtime-owned numerical leaf.
 *
 * address is diagnostic metadata, not a host pointer.  Tests may pass it to
 * the selected accelerator backend's explicit download primitive, but code at
 * this CUDA-free boundary must never dereference it.  elements is the active
 * scalar extent (not bytes) and is zero exactly when the optional leaf is not
 * present for the prepared topology.
 */
struct Gfn2CudaOpaqueBufferIdentity {
  std::uintptr_t address = 0u;
  std::int64_t elements = 0;
};

/*
 * CUDA-free observability for the context-owned restricted GFN2 runtime.
 *
 * The addresses are opaque identities only: callers must never dereference
 * them. Tests use the snapshot to prove that a fixed topology reuses handles,
 * owners, descriptors, and arenas while a transactional rebuild replaces the
 * complete topology-scoped object at once. Keeping CUDA types out of this
 * header preserves the backend boundary required by a future HIP owner.
 */
struct Gfn2CudaExecutionIdentity {
  std::uint64_t topology_fingerprint = 0u;
  std::uint64_t plan_token = 0u;
  std::uint64_t iteration_layout_fingerprint = 0u;
  std::uint32_t enabled_component_mask = 0u;
  std::uint8_t scc_binding_ready = 0u;
  /* The six #67 descriptors are token-coherent for at least energy mode. */
  std::uint8_t energy_force_binding_ready = 0u;
  /* One only after every force-specific immutable/workspace leaf is bound. */
  std::uint8_t force_mode_ready = 0u;
  /* One only after a converged-state composed execution publishes finite output. */
  std::uint8_t energy_force_smoke_ready = 0u;
  /* One after the fixed-topology numerical refresh graph is sealed. */
  std::uint8_t numerical_refresh_ready = 0u;
  /* One after terminal energy/publication descriptors own stable storage. */
  std::uint8_t inference_ready = 0u;
  /* One after every peer published a consumable checkpoint and host aggregation completed. */
  std::uint8_t warm_checkpoint_ready = 0u;
  /* One when normal inference launches the context-owned conditional SCC Graph. */
  std::uint8_t scc_conditional_graph_ready = 0u;
  /* Gfn2SccLoopGraphFallbackReason encoded without exposing CUDA headers. */
  std::uint32_t scc_loop_fallback_reason = 0u;
  /* Native lattice identity is independent from caller-owned b + A*q
   * embedding.  The element counts live in the opaque buffer identities. */
  std::uint8_t native_lattice_enabled = 0u;

  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t total_point_charges = 0;
  std::int64_t native_lattice_systems = 0;

  std::uintptr_t solver_handle = 0u;
  std::uintptr_t solver_parameters = 0u;
  std::uintptr_t blas_handle = 0u;
  std::uintptr_t topology_owner = 0u;
  std::uintptr_t inputs_owner = 0u;
  std::uintptr_t eigensolver_owner = 0u;
  std::uintptr_t initializer_owner = 0u;
  std::uintptr_t scc_binding = 0u;
  /* Stable public SCC state leaves. Runtime tests sample these addresses to
   * prove rejected stream-ordered requests cannot mutate the prior attempt's
   * canonical state even when the bounded fallback DAG was already captured. */
  std::uintptr_t scc_state_iterations = 0u;
  std::uintptr_t scc_state_converged = 0u;
  std::uintptr_t scc_state_system_statuses = 0u;
  std::uintptr_t scc_loop_owner = 0u;
  std::uintptr_t scc_loop_active_count = 0u;
  std::uintptr_t scc_loop_numerical_body_count = 0u;
  std::uintptr_t scc_loop_device_launch_error = 0u;
  std::uintptr_t energy_force_descriptors = 0u;
  /* Stable completion owner and accepted single-flight submission count. */
  std::uintptr_t request_completion_owner = 0u;
  std::uint64_t request_submissions = 0u;
  std::uint8_t request_active = 0u;

  std::uintptr_t topology_arena = 0u;
  std::uintptr_t input_arena = 0u;
  std::uintptr_t iteration_arena = 0u;
  std::uintptr_t eigensolver_setup_arena = 0u;
  std::uintptr_t provider_host_workspace = 0u;
  std::uintptr_t native_lattice_host_staging = 0u;
  std::uintptr_t integral_task_arena = 0u;
  std::uintptr_t force_immutable_arena = 0u;
  std::uintptr_t force_execution_arena = 0u;
  std::uintptr_t numerical_refresh_arena = 0u;
  /* Fixed-capacity staging for ABI-v3 interaction bytes. Capacity is derived
   * from the prepared batch size, so every fixed-topology refresh preserves
   * both arena identities and performs no staging allocation. */
  std::uintptr_t interaction_device_staging_arena = 0u;
  std::uintptr_t interaction_host_staging_arena = 0u;
  std::uintptr_t numerical_refresh_binding = 0u;
  std::uintptr_t numerical_epoch = 0u;
  std::uintptr_t committed_generations = 0u;
  std::uintptr_t numerical_eligible_mask = 0u;
  std::uintptr_t overlap_factor_generations = 0u;
  std::uintptr_t overlap_factor_statuses = 0u;

  /*
   * Stable, runtime-owned committed numerical leaves.  These identities make
   * fixed-topology refresh publication directly testable without exposing a
   * CUDA type or promising that a future HIP backend uses the same address
   * representation.  Integral multipoles retain their production global
   * component-major layout.
   */
  Gfn2CudaOpaqueBufferIdentity committed_positions{};
  Gfn2CudaOpaqueBufferIdentity committed_geometry_pairs{};
  Gfn2CudaOpaqueBufferIdentity committed_coordination_numbers{};
  Gfn2CudaOpaqueBufferIdentity committed_overlap{};
  Gfn2CudaOpaqueBufferIdentity committed_dipole_integrals{};
  Gfn2CudaOpaqueBufferIdentity committed_quadrupole_integrals{};
  Gfn2CudaOpaqueBufferIdentity committed_h0{};
  Gfn2CudaOpaqueBufferIdentity committed_es2{};
  Gfn2CudaOpaqueBufferIdentity committed_aes2{};
  /* committed_d4_pairs is canonical empty after #220: D4 rebuilds
   * role-specific values directly from positions over the committed physical
   * pair-list superset. The committed coordination-number outlet remains
   * populated when D4 is enabled. */
  Gfn2CudaOpaqueBufferIdentity committed_d4_pairs{};
  Gfn2CudaOpaqueBufferIdentity committed_d4_coordination_numbers{};
  Gfn2CudaOpaqueBufferIdentity committed_point_charge_positions{};
  Gfn2CudaOpaqueBufferIdentity committed_point_charge_values{};
  Gfn2CudaOpaqueBufferIdentity committed_point_charge_gammas{};
  Gfn2CudaOpaqueBufferIdentity committed_point_charge_shell_potential{};
  Gfn2CudaOpaqueBufferIdentity committed_periodic_shifts{};
  Gfn2CudaOpaqueBufferIdentity committed_periodic_response{};
  Gfn2CudaOpaqueBufferIdentity committed_native_cell_matrices{};
  Gfn2CudaOpaqueBufferIdentity committed_native_periodic_axes{};

  std::int64_t committed_generation_elements = 0;
  std::int64_t numerical_eligible_elements = 0;
  std::int64_t overlap_factor_generation_elements = 0;
  std::int64_t overlap_factor_status_elements = 0;
  std::uintptr_t inference_arena = 0u;
  std::uintptr_t inference_epoch_consumer = 0u;
  std::uintptr_t inference_results = 0u;
  std::uintptr_t inference_energies = 0u;
  std::uintptr_t inference_qm_forces = 0u;
  std::uintptr_t inference_atomic_charges = 0u;
  std::uintptr_t inference_point_forces = 0u;
  std::uintptr_t inference_iterations = 0u;
  std::uintptr_t inference_converged = 0u;
  std::uintptr_t inference_system_statuses = 0u;
  /* Device publication provenance for the most recently submitted inference. */
  std::uintptr_t inference_publication_epoch_snapshot = 0u;
  std::uintptr_t inference_publication_system_errors = 0u;
  std::uintptr_t inference_publication_plan_error = 0u;
  std::uintptr_t warm_checkpoint_generations = 0u;

  std::size_t topology_arena_bytes = 0u;
  std::size_t input_arena_bytes = 0u;
  std::size_t iteration_arena_bytes = 0u;
  std::size_t eigensolver_setup_arena_bytes = 0u;
  std::size_t provider_host_workspace_bytes = 0u;
  std::size_t integral_task_arena_bytes = 0u;
  std::size_t force_immutable_arena_bytes = 0u;
  std::size_t force_execution_arena_bytes = 0u;
  std::size_t numerical_refresh_arena_bytes = 0u;
  std::size_t inference_arena_bytes = 0u;
  std::size_t numerical_host_staging_arena_bytes = 0u;
  std::size_t interaction_device_staging_arena_bytes = 0u;
  std::size_t interaction_descriptor_capacity_bytes = 0u;
  std::size_t interaction_payload_capacity_bytes = 0u;
  std::size_t public_result_device_arena_bytes = 0u;
  std::size_t public_result_host_arena_bytes = 0u;
  std::size_t candidate_validation_arena_bytes = 0u;
  std::size_t native_lattice_host_staging_bytes = 0u;
  std::size_t topology_staging_host_bytes = 0u;
  std::size_t topology_staging_device_bytes = 0u;
  /* Heap bodies for the prepared cache and its retained topology candidate. */
  std::size_t runtime_owner_host_bytes = 0u;
  std::size_t host_plans_bytes = 0u;
  /* Test-visible decomposition of host_plans_bytes. These fields are internal
   * runtime diagnostics, not part of the stable public C ABI. */
  std::size_t host_common_plan_vector_bytes = 0u;
  std::size_t host_gfn1_plan_vector_bytes = 0u;
  std::size_t host_common_model_plan_bytes = 0u;
  std::size_t host_gfn1_model_plan_bytes = 0u;
  std::size_t host_numerical_vector_bytes = 0u;
  std::size_t host_gfn1_expanded_parameter_bytes = 0u;
  std::size_t host_gfn2_wavefunction_arena_bytes = 0u;
  std::size_t host_gfn1_wavefunction_arena_bytes = 0u;
  std::size_t topology_setup_host_bytes = 0u;
  std::size_t inputs_setup_host_bytes = 0u;
  std::size_t eigensolver_setup_host_bytes = 0u;
  /* Host implementation record retained beside the device checkpoint. */
  std::size_t initializer_host_bytes = 0u;
  std::size_t initializer_device_checkpoint_bytes = 0u;
  std::size_t scc_loop_device_control_bytes = 0u;
  /* Complete plan-owned reusable workspace totals. These include every arena
   * and explicit owner allocation above plus mixed-memory topology staging. */
  std::size_t retained_host_workspace_bytes = 0u;
  std::size_t retained_device_workspace_bytes = 0u;
};

/* Runtime SCC initialization policy for one complete inference submission. */
enum class Gfn2CudaSccStartMode : std::uint32_t {
  kFresh = 1u,
  kWarm = 2u,
};

struct Gfn2CudaNativeLatticeTestIdentity {
  std::uintptr_t host_staging = 0u;
  std::size_t host_staging_bytes = 0u;
  bool pending = false;
  bool poisoned = false;
};

#if defined(XTBLOOM_CUDA_TEST_HOOKS)
/*
 * Test-only failure points for the context-owned native-lattice staging
 * transaction. Each armed fault is consumed once so a test can exercise the
 * original failure and then prove that the same cache remains retryable.
 */
enum class Gfn2CudaExecutionTestFault : std::uint32_t {
  kNone = 0u,
  kNativeLatticePinnedAllocation = 1u,
  kNativeLatticeCompletionWait = 2u,
  kNativeLatticeTeardownSettlement = 3u,
  kUnknownRequestValidationCode = 4u,
  kRequestPrepareSubmission = 5u,
  kRequestCommitSubmission = 6u,
  /* Force one newly built runtime to retain the production bounded SCC
   * fallback. This proves that the synchronous plan remains usable while the
   * narrower asynchronous Graph capability is reported as unavailable. */
  kSccProviderUncapturedFallback = 7u,
  kRequestSettlement = 8u,
  kRequestPrepareSubmissionAndSettlement = 9u,
  kRequestCommitSubmissionAndSettlement = 10u,
  /* Fail before Graph creation or after capture but before executable
   * publication. These prove that lazy setup leaves the request IDLE and no
   * caller-owned descriptor is retained by queued work. */
  kRequestGraphCreate = 11u,
  kRequestGraphInstantiate = 12u,
};

/* Read-only evidence for failure paths that intentionally cannot retain an
 * owner for later inspection. Resetting these counters never reclaims a
 * quarantined allocation because no reliable CUDA completion fence exists. */
struct Gfn2CudaExecutionTestStats {
  std::uint64_t native_lattice_allocation_faults = 0u;
  std::uint64_t native_lattice_completion_faults = 0u;
  std::uint64_t native_lattice_teardown_faults = 0u;
  std::uint64_t quarantined_native_lattice_arenas = 0u;
  std::uint64_t quarantined_native_lattice_bytes = 0u;
  std::uint64_t request_graph_build_attempts = 0u;
  std::uint64_t request_graph_build_successes = 0u;
};

void reset_gfn2_cuda_execution_test_state() noexcept;
void arm_gfn2_cuda_execution_test_fault(Gfn2CudaExecutionTestFault fault) noexcept;
[[nodiscard]] Gfn2CudaExecutionTestStats gfn2_cuda_execution_test_stats() noexcept;
#endif

/*
 * CUDA-free numerical view for a previously prepared fixed topology.
 *
 * Every nonempty buffer may reside in host or CUDA-device memory. The runtime
 * synchronously snapshots host values into runtime-owned pinned storage and
 * keeps device values on the caller stream; no queued work retains a caller
 * host pointer after return. Only one host snapshot may be in flight per fixed
 * topology; another host submission is rejected until a stream-ordered
 * completion releases that snapshot, while CUDA-device inputs remain direct.
 * The runtime never downloads numerical data, queries CUDA progress, or polls
 * asynchronous diagnostics. requested_mask is an optional
 * uint8_t[batch] view. An absent mask requests every member.
 *
 * CUDA Graph capture/replay requires every nonempty numerical buffer to be in
 * CUDA-device memory. A host-backed view is rejected before the runtime copies
 * or marks its fixed pinned staging image.
 *
 * The context-owned device epoch advances once per accepted refresh (and once
 * per CUDA Graph replay). Publication is transactional per ragged member: an unrequested or failing
 * member retains its prior numerical leaves, operators, caches, overlap
 * factor, and committed generation.
 */
struct Gfn2CudaNumericalInputView {
  xtbloom_const_buffer_t positions{};
  xtbloom_const_buffer_t point_charge_positions{};
  xtbloom_const_buffer_t point_charge_values{};
  xtbloom_const_buffer_t point_charge_gammas{};
  xtbloom_const_buffer_t atomic_potential_shifts{};
  xtbloom_const_buffer_t charge_response_matrix{};
  /* ABI-v3 interaction metadata remains numerical state: FRESH calls may
   * attach, change, or detach a field without rebuilding the fixed topology.
   * total_interactions is zero for short-prefix callers. */
  std::int64_t total_interactions = 0;
  xtbloom_const_buffer_t interaction_descriptors{};
  xtbloom_const_buffer_t interaction_payload{};
  xtbloom_const_buffer_t requested_mask{};
};

#ifdef XTBLOOM_CUDA_TEST_HOOKS
/* White-box construction faults used only by CUDA runtime-owner tests. The
 * next prepared candidate consumes the selected hook and then resets it. */
enum class Gfn2CudaAdmissionAliasTestHook : std::uint32_t {
  kNone = 0u,
  kNumericalCandidatePositions = 1u,
  kStationaryAtomicCharges = 2u,
};

void set_gfn2_cuda_admission_alias_test_hook(Gfn2CudaAdmissionAliasTestHook hook) noexcept;
#endif

/*
 * Context-owned cache for complete restricted CUDA GFN2 setup state.
 *
 * prepare_host consumes descriptor-validated host metadata. It constructs a
 * candidate off to the side, waits only for candidate setup work on the
 * caller stream, validates asynchronous factorization diagnostics, and then
 * atomically replaces the previous topology-scoped object. A same-topology
 * host request reuses every owner/address and enqueues the allocation-free
 * numerical refresh transaction on the context stream. Explicit host/device
 * refreshes use refresh_numerical_async and the same sealed device DAG.
 *
 * A Graph that captures a device-backed refresh retains fixed-topology arena
 * addresses. Until every such graph executable is destroyed, the cache and
 * its prepared topology must stay alive and unchanged, and replay is supported
 * only serially on the context owner stream. A future controlled launch API
 * can replace these caller-enforced lifetime and stream constraints.
 */
class Gfn2CudaExecutionCache : public RequestCompletion {
 public:
  Gfn2CudaExecutionCache(std::int32_t device_id, void* stream);
  ~Gfn2CudaExecutionCache() override;

  Gfn2CudaExecutionCache(const Gfn2CudaExecutionCache&) = delete;
  Gfn2CudaExecutionCache& operator=(const Gfn2CudaExecutionCache&) = delete;

  [[nodiscard]] xtbloom_status_t prepare_host(const xtbloom_batch_t& batch,
                                              const xtbloom_compute_options_t& options,
                                              bool& reused, std::string& error);

  /* Prepare a plan-owned runtime from topology metadata without reading the
   * caller's numerical buffers. This permits device-resident geometry during
   * plan creation; the first compute refreshes the prepared seed from the real
   * descriptor before executing or publishing results. */
  [[nodiscard]] xtbloom_status_t prepare_topology_only(const xtbloom_batch_t& batch,
                                                       const xtbloom_compute_options_t& options,
                                                       std::string& error);

  /* Enqueue one allocation-free fixed-topology numerical transaction. */
  [[nodiscard]] xtbloom_status_t refresh_numerical_async(const Gfn2CudaNumericalInputView& input,
                                                         std::string& error);

  /*
   * Enqueue SCC through internal result publication on the context stream.
   * Fresh mode restores the immutable SAD image. Warm mode reuses the prior
   * device wavefunction and mixer checkpoint only when its per-peer geometry
   * generation matches either the current numerical epoch or the immediately
   * preceding committed epoch accepted by the latest refresh transaction. It
   * still resets the driver-visible terminal trace for the new inference
   * attempt.
   */
  [[nodiscard]] xtbloom_status_t execute_inference_async(Gfn2CudaSccStartMode mode,
                                                         std::string& error);

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] Gfn2CudaExecutionIdentity identity() const noexcept;

  /* White-box staging entry points used to isolate allocation and teardown
   * contracts from SCC/request-graph construction. Definitions exist only in
   * test builds, but the declarations remain unconditional so every
   * translation unit sees the same internal class definition. */
  [[nodiscard]] xtbloom_status_t validate_native_lattice_test_only(const xtbloom_batch_t& batch,
                                                                   std::string& error);
  [[nodiscard]] Gfn2CudaNativeLatticeTestIdentity native_lattice_test_identity() const noexcept;

  /* The single-flight cache is its own preallocated completion owner, so
   * publishing it into a reusable request needs no per-enqueue allocation. */
  [[nodiscard]] xtbloom_status_t probe(bool wait,
                                       RequestCompletionResult& result) noexcept override;
  void settle_noexcept() noexcept override;

 private:
  friend xtbloom_status_t execute_restricted_gfn2_cuda_impl(
      Gfn2CudaExecutionCache& cache, const xtbloom_batch_t& batch,
      const xtbloom_compute_options_t& options, xtbloom_batch_result_t& result,
      bool require_prepared_topology, std::string& error);
  friend xtbloom_status_t execute_restricted_gfn2_cuda(Gfn2CudaExecutionCache& cache,
                                                       const xtbloom_batch_t& batch,
                                                       const xtbloom_compute_options_t& options,
                                                       xtbloom_batch_result_t& result,
                                                       std::string& error);
  friend xtbloom_status_t enqueue_restricted_gfn2_cuda_plan(
      const std::shared_ptr<Gfn2CudaExecutionCache>& cache, const xtbloom_batch_t& batch,
      const xtbloom_compute_options_t& options, const xtbloom_batch_result_t& result,
      RequestSubmission& submission, std::string& error);
  friend xtbloom_status_t enqueue_restricted_gfn2_cuda(
      const std::shared_ptr<Gfn2CudaExecutionCache>& cache, const xtbloom_batch_t& batch,
      const xtbloom_compute_options_t& options, const xtbloom_batch_result_t& result,
      RequestSubmission& submission, std::string& error);
  friend xtbloom_status_t enqueue_restricted_gfn2_cuda_impl(
      const std::shared_ptr<Gfn2CudaExecutionCache>& cache, const xtbloom_batch_t& batch,
      const xtbloom_compute_options_t& options, const xtbloom_batch_result_t& result,
      bool require_prepared_topology, RequestSubmission& submission, std::string& error);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/*
 * Execute one synchronous public CUDA request as a single cache transaction.
 * Validation, topology staging, candidate refresh, SCC, internal publication,
 * public result bridging, and completion all remain serialized by the cache
 * mutex. The owner drives prepare submission, completion observation, host
 * aggregate acceptance, caller-device commit, and host publication as
 * separate phases while preserving the synchronous wait points. Failures
 * detected before caller-output commit leave output bytes and result.flags
 * unchanged. Any later catastrophic failure may return
 * INTERNAL_ERROR after results were modified. Current-device restoration is a
 * separate exit boundary and may fail before or after output commit, as
 * documented by the public API.
 */
[[nodiscard]] xtbloom_status_t execute_restricted_gfn2_cuda(
    Gfn2CudaExecutionCache& cache, const xtbloom_batch_t& batch,
    const xtbloom_compute_options_t& options, xtbloom_batch_result_t& result, std::string& error);

/* Plan-owned variant of the public transaction. It uses the same pointer and
 * canonical topology staging as xtbloom_compute, but rejects a topology
 * candidate before numerical refresh instead of rebuilding the prepared
 * runtime. This is the fixed-topology corruption gate for device descriptors. */
[[nodiscard]] xtbloom_status_t execute_restricted_gfn2_cuda_plan(
    Gfn2CudaExecutionCache& cache, const xtbloom_batch_t& batch,
    const xtbloom_compute_options_t& options, xtbloom_batch_result_t& result, std::string& error);

/* Submit one fixed-topology CUDA plan transaction without waiting for
 * inference or caller-output publication. Descriptor structs are copied,
 * every host numerical leaf is snapshotted, host topology is compared before
 * return, and device topology is compared in owner-stream order before the
 * transactional result gate can commit any caller output. */
[[nodiscard]] xtbloom_status_t enqueue_restricted_gfn2_cuda_plan(
    const std::shared_ptr<Gfn2CudaExecutionCache>& cache, const xtbloom_batch_t& batch,
    const xtbloom_compute_options_t& options, const xtbloom_batch_result_t& result,
    RequestSubmission& submission, std::string& error);

/* Context-owned counterpart that transactionally prepares or reuses the
 * current topology before submitting the same asynchronous inference/result
 * protocol. Topology construction may perform bounded setup waits, but an
 * already prepared topology never waits for inference or caller publication. */
[[nodiscard]] xtbloom_status_t enqueue_restricted_gfn2_cuda(
    const std::shared_ptr<Gfn2CudaExecutionCache>& cache, const xtbloom_batch_t& batch,
    const xtbloom_compute_options_t& options, const xtbloom_batch_result_t& result,
    RequestSubmission& submission, std::string& error);

}  // namespace xtbloom::detail

#endif  // XTBLOOM_RUNTIME_GFN2_CUDA_EXECUTION_HPP
