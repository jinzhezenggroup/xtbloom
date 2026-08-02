#ifndef GPUXTB_BACKENDS_CUDA_GFN2_ES3_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_ES3_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

namespace gpuxtb::detail::cuda {

/* Semantic input and arithmetic errors detected asynchronously by ES3 kernels. */
enum class Gfn2ES3DeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kNonfiniteGamma3 = 2u,
  kNonfiniteShellCharge = 3u,
  kNonfinitePotentialArithmetic = 4u,
  kNonfiniteEnergySeed = 5u,
  kNonfiniteEnergyArithmetic = 6u,
};

/*
 * Non-owning device view of a ragged shell batch for the onsite GFN2 cubic
 * charge term. Shells for system i occupy
 * [batch_shell_offsets[i], batch_shell_offsets[i + 1]). The explicit counts
 * make truncated or stale owning plans a host-side error before launch.
 *
 * All pointers refer to caller-owned CUDA-accessible storage that must remain
 * valid until work queued on the supplied stream completes. The view contains
 * no backend-owned state and is suitable for fixed-topology reuse.
 */
struct Gfn2ES3DeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_shells = 0;
  std::int64_t batch_shell_offset_count = 0;
  std::int64_t shell_gamma3_count = 0;
  const std::int64_t* batch_shell_offsets = nullptr;
  const double* shell_gamma3 = nullptr;
};

static_assert(std::is_trivially_copyable_v<Gfn2ES3DeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2ES3DeviceBatch>);

/*
 * Queue v_s = Gamma3_s q_s^2 and overwrite every shell potential. Each
 * system is fully validated and arithmetic-preflighted before any potential
 * in that system is written. Input and output ranges must not overlap.
 */
cudaError_t evaluate_gfn2_es3_potential_cuda(const Gfn2ES3DeviceBatch& batch,
                                             const double* shell_charges, double* shell_potentials,
                                             std::uint32_t* device_error,
                                             cudaStream_t stream = nullptr) noexcept;

/*
 * Queue E3 += sum_s Gamma3_s q_s^3 / 3 in shell order for each ragged system.
 * Existing energies must be finite. A system's energy is written only after
 * its full accumulation has succeeded. Input and output ranges must not
 * overlap.
 */
cudaError_t add_gfn2_es3_energy_cuda(const Gfn2ES3DeviceBatch& batch, const double* shell_charges,
                                     double* energies, std::uint32_t* device_error,
                                     cudaStream_t stream = nullptr) noexcept;

/*
 * Asynchronously initialize a caller-owned error scalar before an ES3
 * sequence. Launchers never clear it: the first semantic error remains sticky
 * and dependent downstream launches become no-ops without a host round trip.
 */
cudaError_t reset_gfn2_es3_device_error_cuda(std::uint32_t* device_error,
                                             cudaStream_t stream = nullptr) noexcept;

/*
 * All operations are allocation-free, enqueue only on the supplied stream,
 * perform no hidden synchronization, and are CUDA Graph capture compatible.
 * A device_error scalar must not be shared by unordered streams. As with the
 * other native CUDA kernels, a batch-wide semantic error requires discarding
 * the sequence outputs because independently scheduled valid systems may have
 * completed before another system recorded the error.
 */

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_ES3_CUH
