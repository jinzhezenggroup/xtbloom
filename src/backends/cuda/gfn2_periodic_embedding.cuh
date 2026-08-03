#ifndef GPUXTB_BACKENDS_CUDA_GFN2_PERIODIC_EMBEDDING_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_PERIODIC_EMBEDDING_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "gpuxtb/gpuxtb.h"

namespace gpuxtb::detail::cuda {

/* First asynchronous semantic failure recorded by a periodic embedding call. */
enum class Gfn2PeriodicEmbeddingDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kNonfiniteShift = 2u,
  kNonfiniteMixedCharge = 3u,
  kNonfiniteRawCharge = 4u,
  kNonfiniteResponseMatrix = 5u,
  kNonsymmetricResponseMatrix = 6u,
  kNonfinitePotentialArithmetic = 7u,
  kNonfiniteEnergyArithmetic = 8u,
};

/*
 * Non-owning device view of a ragged periodic embedding batch. Atoms for
 * system i occupy [atom_offsets[i], atom_offsets[i + 1]); its dense row-major
 * response matrix occupies the corresponding matrix_offsets interval.
 * Empty systems are supported. A nonzero plan_token binds the topology and
 * caller-owned workspace without exposing a host plan object to device code.
 */
struct Gfn2PeriodicEmbeddingDeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_matrix_elements = 0;
  std::int64_t atom_offset_count = 0;
  std::int64_t matrix_offset_count = 0;
  std::int64_t shift_elements = 0;
  std::int64_t response_elements = 0;
  std::uint64_t plan_token = 0u;
  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* matrix_offsets = nullptr;
  const double* shifts = nullptr;
  const double* response_matrices = nullptr;
};

/*
 * Caller-owned scratch used to make publication transactional per system.
 * potential_scratch stores b + A*q_mixed and raw_response_scratch stores
 * A*q_raw. sequence_active snapshots batch-wide preflight success before a
 * numerical failure from one system can set the shared sticky error scalar.
 */
struct Gfn2PeriodicEmbeddingDeviceWorkspace {
  double* potential_scratch = nullptr;
  double* raw_response_scratch = nullptr;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t atom_elements = 0;
  std::int64_t sequence_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2PeriodicEmbeddingDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2PeriodicEmbeddingDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2PeriodicEmbeddingDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2PeriodicEmbeddingDeviceWorkspace>);

/* Clear the caller-owned sticky semantic error before a dependent sequence. */
cudaError_t reset_gfn2_periodic_embedding_device_error_cuda(std::uint32_t* device_error,
                                                            cudaStream_t stream = nullptr) noexcept;

/*
 * Queue the complete periodic charge embedding operation
 *
 *   V = b + A*q_mixed,
 *   E = q_raw^T*b + 0.5*q_raw^T*A*q_raw.
 *
 * Potentials and energies are overwritten. Each dense A must be finite and
 * exactly symmetric as doubles; +0.0 and -0.0 compare equal and are accepted.
 * A numerical failure leaves that member's V and E unchanged, publishes
 * GPUXTB_STATUS_INTERNAL_ERROR only to its status, and does not prevent
 * healthy peers from committing. A topology failure is whole-call atomic.
 * This kernel intentionally provides V and E only. Cartesian derivatives of
 * caller-supplied b or A remain the caller's responsibility; a CUDA force
 * path must neither treat their absence as zero nor fabricate those terms.
 *
 * Every pointer refers to caller-owned CUDA-accessible storage that remains
 * valid until work on stream completes. Writable active ranges must be
 * pairwise disjoint from inputs and one another. The launcher allocates
 * nothing, performs no synchronization or host transfer, supports custom
 * streams, and is CUDA Graph capture compatible.
 */
cudaError_t evaluate_gfn2_periodic_embedding_cuda(
    const Gfn2PeriodicEmbeddingDeviceBatch& batch, const double* mixed_atomic_charges,
    const double* raw_atomic_charges, double* atomic_potentials, double* energies,
    gpuxtb_status_t* system_statuses, const Gfn2PeriodicEmbeddingDeviceWorkspace& workspace,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_PERIODIC_EMBEDDING_CUH
