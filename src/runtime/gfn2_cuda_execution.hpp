#ifndef GPUXTB_RUNTIME_GFN2_CUDA_EXECUTION_HPP
#define GPUXTB_RUNTIME_GFN2_CUDA_EXECUTION_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "gpuxtb/gpuxtb.h"

namespace gpuxtb::detail {

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
  std::uintptr_t energy_force_descriptors = 0u;

  std::uintptr_t topology_arena = 0u;
  std::uintptr_t input_arena = 0u;
  std::uintptr_t iteration_arena = 0u;
  std::uintptr_t eigensolver_setup_arena = 0u;
  std::uintptr_t provider_host_workspace = 0u;
  std::uintptr_t force_immutable_arena = 0u;
  std::uintptr_t force_execution_arena = 0u;

  std::size_t topology_arena_bytes = 0u;
  std::size_t input_arena_bytes = 0u;
  std::size_t iteration_arena_bytes = 0u;
  std::size_t eigensolver_setup_arena_bytes = 0u;
  std::size_t provider_host_workspace_bytes = 0u;
  std::size_t force_immutable_arena_bytes = 0u;
  std::size_t force_execution_arena_bytes = 0u;
};

/*
 * Context-owned cache for complete restricted CUDA GFN2 setup state.
 *
 * prepare_host consumes descriptor-validated host metadata. It constructs a
 * candidate off to the side, waits only for candidate setup work on the
 * caller stream, validates asynchronous factorization diagnostics, and then
 * atomically replaces the previous topology-scoped object. A same-topology
 * request returns reused=true without allocation, transfer, or synchronization;
 * geometry-dependent refresh and execution are deliberately owned by #113.
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

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] Gfn2CudaExecutionIdentity identity() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpuxtb::detail

#endif  // GPUXTB_RUNTIME_GFN2_CUDA_EXECUTION_HPP
