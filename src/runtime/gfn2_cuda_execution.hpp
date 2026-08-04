#ifndef GPUXTB_RUNTIME_GFN2_CUDA_EXECUTION_HPP
#define GPUXTB_RUNTIME_GFN2_CUDA_EXECUTION_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "gpuxtb/gpuxtb.h"

namespace gpuxtb::detail {

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

  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t total_point_charges = 0;

  std::uintptr_t solver_handle = 0u;
  std::uintptr_t solver_parameters = 0u;
  std::uintptr_t blas_handle = 0u;
  std::uintptr_t topology_owner = 0u;
  std::uintptr_t inputs_owner = 0u;
  std::uintptr_t eigensolver_owner = 0u;
  std::uintptr_t initializer_owner = 0u;
  std::uintptr_t scc_binding = 0u;
  std::uintptr_t scc_loop_owner = 0u;
  std::uintptr_t scc_loop_active_count = 0u;
  std::uintptr_t scc_loop_numerical_body_count = 0u;
  std::uintptr_t energy_force_descriptors = 0u;

  std::uintptr_t topology_arena = 0u;
  std::uintptr_t input_arena = 0u;
  std::uintptr_t iteration_arena = 0u;
  std::uintptr_t eigensolver_setup_arena = 0u;
  std::uintptr_t provider_host_workspace = 0u;
  std::uintptr_t force_immutable_arena = 0u;
  std::uintptr_t force_execution_arena = 0u;
  std::uintptr_t numerical_refresh_arena = 0u;
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
  Gfn2CudaOpaqueBufferIdentity committed_d4_pairs{};
  Gfn2CudaOpaqueBufferIdentity committed_d4_coordination_numbers{};
  Gfn2CudaOpaqueBufferIdentity committed_point_charge_positions{};
  Gfn2CudaOpaqueBufferIdentity committed_point_charge_values{};
  Gfn2CudaOpaqueBufferIdentity committed_point_charge_gammas{};
  Gfn2CudaOpaqueBufferIdentity committed_point_charge_shell_potential{};
  Gfn2CudaOpaqueBufferIdentity committed_periodic_shifts{};
  Gfn2CudaOpaqueBufferIdentity committed_periodic_response{};

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
  std::size_t force_immutable_arena_bytes = 0u;
  std::size_t force_execution_arena_bytes = 0u;
  std::size_t numerical_refresh_arena_bytes = 0u;
  std::size_t inference_arena_bytes = 0u;
};

/* Runtime SCC initialization policy for one complete inference submission. */
enum class Gfn2CudaSccStartMode : std::uint32_t {
  kFresh = 1u,
  kWarm = 2u,
};

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
  gpuxtb_const_buffer_t positions{};
  gpuxtb_const_buffer_t point_charge_positions{};
  gpuxtb_const_buffer_t point_charge_values{};
  gpuxtb_const_buffer_t point_charge_gammas{};
  gpuxtb_const_buffer_t atomic_potential_shifts{};
  gpuxtb_const_buffer_t charge_response_matrix{};
  gpuxtb_const_buffer_t requested_mask{};
};

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
class Gfn2CudaExecutionCache {
 public:
  Gfn2CudaExecutionCache(std::int32_t device_id, void* stream);
  ~Gfn2CudaExecutionCache();

  Gfn2CudaExecutionCache(const Gfn2CudaExecutionCache&) = delete;
  Gfn2CudaExecutionCache& operator=(const Gfn2CudaExecutionCache&) = delete;

  [[nodiscard]] gpuxtb_status_t prepare_host(const gpuxtb_batch_t& batch,
                                             const gpuxtb_compute_options_t& options, bool& reused,
                                             std::string& error);

  /* Enqueue one allocation-free fixed-topology numerical transaction. */
  [[nodiscard]] gpuxtb_status_t refresh_numerical_async(const Gfn2CudaNumericalInputView& input,
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
  [[nodiscard]] gpuxtb_status_t execute_inference_async(Gfn2CudaSccStartMode mode,
                                                        std::string& error);

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] Gfn2CudaExecutionIdentity identity() const noexcept;

 private:
  friend gpuxtb_status_t execute_restricted_gfn2_cuda(
      Gfn2CudaExecutionCache& cache, const gpuxtb_batch_t& batch,
      const gpuxtb_compute_options_t& options, gpuxtb_batch_result_t& result,
      std::string& error);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/*
 * Execute one synchronous public CUDA request as a single cache transaction.
 * Validation, topology staging, candidate refresh, SCC, internal publication,
 * public result bridging, and completion all remain serialized by the cache
 * mutex. Failures detected before caller-output commit leave output bytes and
 * result.flags unchanged. Any later catastrophic failure may return
 * INTERNAL_ERROR after results were modified. Current-device restoration is a
 * separate exit boundary and may fail before or after output commit, as
 * documented by the public API.
 */
[[nodiscard]] gpuxtb_status_t execute_restricted_gfn2_cuda(
    Gfn2CudaExecutionCache& cache, const gpuxtb_batch_t& batch,
    const gpuxtb_compute_options_t& options, gpuxtb_batch_result_t& result,
    std::string& error);

}  // namespace gpuxtb::detail

#endif  // GPUXTB_RUNTIME_GFN2_CUDA_EXECUTION_HPP
