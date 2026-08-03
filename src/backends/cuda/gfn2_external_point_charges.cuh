#ifndef GPUXTB_BACKENDS_CUDA_GFN2_EXTERNAL_POINT_CHARGES_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_EXTERNAL_POINT_CHARGES_CUH

#include <cuda_runtime_api.h>

#include <cstdint>

#include "backends/cuda/gfn2_force_common.cuh"
#include "backends/cuda/gfn2_scc_iteration_control.cuh"

namespace gpuxtb::detail::cuda {

/* Semantic input errors detected asynchronously by external point-charge kernels. */
enum class Gfn2ExternalPointChargeDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kInvalidShellMetadata = 2u,
  kNonfiniteQmPosition = 3u,
  kInvalidPointChargeInput = 4u,
  kNonfiniteShellValue = 5u,
  kNonfinitePairArithmetic = 6u,
  kCacheMismatch = 7u,
  kInvalidForceRequest = 8u,
};

/*
 * A non-owning device view of a ragged GFN2 QM/external-point-charge batch.
 *
 * Coordinates are xyz-major within each atom/site and use bohr. Shells and
 * point charges belonging to system i occupy the half-open ranges described
 * by batch_shell_offsets and point_charge_offsets. shell_to_atom contains
 * global atom indices. All storage is owned by the caller and must remain
 * valid until work queued on the supplied stream has completed.
 *
 * The geometry and point-value pointers are only required by potential and
 * force evaluation. This permits the energy stage to consume independently
 * produced shell charges and potentials without retaining geometry buffers.
 */
struct Gfn2ExternalPointChargeDeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_point_charges = 0;

  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* batch_shell_offsets = nullptr;
  const std::int64_t* point_charge_offsets = nullptr;
  const std::int64_t* shell_to_atom = nullptr;
  const double* shell_hardness = nullptr;

  const double* qm_positions = nullptr;
  const double* point_positions = nullptr;
  const double* point_charges = nullptr;
  const double* point_hardnesses = nullptr;
  /* Optional setup identity used by SCC-specific cache consumers. */
  std::uint64_t plan_token = 0u;
};

/* Geometry-generation-scoped explicit point-charge shell-potential cache. */
struct Gfn2ExternalPointChargeDeviceCache {
  double* shell_potentials = nullptr;
  std::int64_t shell_elements = 0;
  std::uint64_t geometry_generation = 0u;
  std::uint64_t plan_token = 0u;
};

/* Caller-owned unpublished potential storage for transactional SCC updates. */
struct Gfn2ExternalPointChargeDeviceWorkspace {
  double* shell_scratch = nullptr;
  std::int64_t shell_elements = 0;
};

/*
 * Caller-owned scratch for transactionally reducing explicit-point-charge
 * forces. A scratch slice is required only when the matching public output is
 * non-NULL. The launcher overwrites eligible-system slices before use and
 * leaves inactive or previously failed slices untouched.
 */
struct Gfn2ExternalPointChargeForceDeviceWorkspace {
  double* qm_scratch = nullptr;
  std::int64_t qm_elements = 0;
  double* point_scratch = nullptr;
  std::int64_t point_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Queue Vpc_s = sum_p Q_p / sqrt(r_sp^2 + a_sp^2), where
 * a_sp = 2 / (gamma_s + gamma_p).
 *
 * shell_potentials is overwritten, including with zero for systems without
 * point charges. The return value only reports host argument, enqueue, and
 * launch errors.
 */
cudaError_t evaluate_gfn2_external_point_charge_potential_cuda(
    const Gfn2ExternalPointChargeDeviceBatch& batch, double* shell_potentials,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

/*
 * Queue E_pc += sum_s q_s Vpc_s for every system in the ragged batch.
 * Existing energy values are preserved and incremented.
 */
cudaError_t add_gfn2_external_point_charge_energy_cuda(
    const Gfn2ExternalPointChargeDeviceBatch& batch, const double* shell_charges,
    const double* shell_potentials, double* energies, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

/*
 * Queue equal-and-opposite QM and point-charge force contributions from the
 * converged shell charges. Either force output may be NULL, but not both.
 * Non-NULL force buffers are incremented rather than initialized.
 * Coincident QM/point sites are finite for positive hardness and contribute
 * exactly zero force.
 */
cudaError_t add_gfn2_external_point_charge_forces_cuda(
    const Gfn2ExternalPointChargeDeviceBatch& batch, const double* shell_charges, double* qm_forces,
    double* point_forces, std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

/*
 * Queue the same equal-and-opposite force reduction behind the common post-SCC
 * force gate. A member is eligible only when requested_mask is one, its
 * terminal system status is SUCCESS, and its incoming system_errors entry is
 * zero. Unrequested members do not read their status or numerical slices.
 *
 * Numerical failures are peer-local and sticky in system_errors. Contributions
 * are accumulated into caller-owned scratch first, so a failed member never
 * partially modifies either public force slice. Successful public buffers are
 * incremented rather than initialized. The caller may independently omit the
 * QM or point output, together with its corresponding scratch slice.
 *
 * The launcher performs no allocation, transfer, host polling, or
 * synchronization and is safe to enqueue on a caller stream or capture in a
 * CUDA Graph. device_error is a sticky sequence-wide first-error scalar;
 * workspace.sequence_active snapshots its incoming state so one peer failure
 * does not suppress healthy peers in the same launch. All plan tokens must
 * match batch.plan_token.
 */
cudaError_t add_gfn2_external_point_charge_gated_forces_cuda(
    const Gfn2ExternalPointChargeDeviceBatch& batch, const Gfn2ForceDeviceActivity& activity,
    const double* shell_charges, double* qm_forces, double* point_forces,
    const Gfn2ExternalPointChargeForceDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

/*
 * Asynchronously initialize a caller-owned error scalar at the beginning of
 * an external-point-charge compute sequence. The three stage launchers never
 * clear this scalar: the first semantic error remains sticky, and downstream
 * stages become no-ops when an upstream stage has failed. This is required
 * for a potential -> SCC -> energy/force pipeline without a host round trip.
 */
cudaError_t reset_gfn2_external_point_charge_device_error_cuda(
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

cudaError_t reset_gfn2_external_point_charge_scc_errors_cuda(
    std::int64_t batch_size, std::uint32_t* system_errors, std::uint32_t* plan_error,
    cudaStream_t stream = nullptr) noexcept;

/* Clear post-SCC force peer diagnostics and the sticky first-error scalar. */
cudaError_t reset_gfn2_external_point_charge_force_errors_cuda(
    std::int64_t batch_size, std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

/*
 * Build active members of the geometry-scoped Vpc cache. Inactive geometry
 * and point-charge arrays remain unread and their previous cache slices remain
 * untouched.
 */
cudaError_t update_gfn2_external_point_charge_scc_potential_cache_cuda(
    const Gfn2ExternalPointChargeDeviceBatch& batch, const Gfn2SccIterationDeviceActivity& activity,
    const Gfn2ExternalPointChargeDeviceCache& cache, std::uint64_t geometry_generation,
    const Gfn2ExternalPointChargeDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* plan_error, cudaStream_t stream = nullptr) noexcept;

/*
 * Overwrite raw Epc = sum_s q_raw,s Vpc,s in CPU shell order. The cache
 * generation is validated only when at least one member is active. No
 * one-half factor is applied and the public destination is never read.
 */
cudaError_t evaluate_gfn2_external_point_charge_scc_energy_cuda(
    const Gfn2ExternalPointChargeDeviceBatch& batch, const Gfn2SccIterationDeviceActivity& activity,
    const Gfn2ExternalPointChargeDeviceCache& cache, std::uint64_t geometry_generation,
    const double* raw_shell_charges, double* component_energies, std::uint32_t* system_errors,
    std::uint32_t* plan_error, cudaStream_t stream = nullptr) noexcept;

/*
 * Every launcher is allocation-free and performs no device-wide or stream
 * synchronization. A device_error scalar must not be shared concurrently
 * across streams. Call reset once before the first stage of a dependent
 * sequence. If a semantic error is reported, discard the sequence's outputs;
 * other valid systems in the failing stage may already have completed.
 */

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_EXTERNAL_POINT_CHARGES_CUH
