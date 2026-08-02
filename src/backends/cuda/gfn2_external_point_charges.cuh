#ifndef GPUXTB_BACKENDS_CUDA_GFN2_EXTERNAL_POINT_CHARGES_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_EXTERNAL_POINT_CHARGES_CUH

#include <cuda_runtime_api.h>

#include <cstdint>

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
 * Asynchronously initialize a caller-owned error scalar at the beginning of
 * an external-point-charge compute sequence. The three stage launchers never
 * clear this scalar: the first semantic error remains sticky, and downstream
 * stages become no-ops when an upstream stage has failed. This is required
 * for a potential -> SCC -> energy/force pipeline without a host round trip.
 */
cudaError_t reset_gfn2_external_point_charge_device_error_cuda(
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

/*
 * Every launcher is allocation-free and performs no device-wide or stream
 * synchronization. A device_error scalar must not be shared concurrently
 * across streams. Call reset once before the first stage of a dependent
 * sequence. If a semantic error is reported, discard the sequence's outputs;
 * other valid systems in the failing stage may already have completed.
 */

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_EXTERNAL_POINT_CHARGES_CUH
