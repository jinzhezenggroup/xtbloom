#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_SCC_ENERGY_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN2_SCC_ENERGY_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_scc_iteration_control.cuh"

namespace xtbloom::detail::cuda {

/* First asynchronous semantic or arithmetic failure in an energy sequence. */
enum class Gfn2SccEnergyDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kInvalidActiveState = 2u,
  kNonfiniteDensity = 3u,
  kNonfiniteH0 = 4u,
  kNonfiniteEntropy = 5u,
  kNonfiniteCoreArithmetic = 6u,
  kNonfiniteFreeEnergy = 7u,
  kInvalidSpinLayout = 8u,
};

/*
 * Restricted dense matrix layout for the electronic SCC energy reduction.
 * System s owns the full row-major matrix interval
 * [matrix_offsets[s], matrix_offsets[s+1]). The descriptor deliberately does
 * not contain CUDA allocation state so the same immutable plan schema can be
 * represented by a future HIP backend.
 */
struct Gfn2SccEnergyDeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_matrix_elements = 0;
  std::int64_t matrix_offset_count = 0;
  std::uint64_t plan_token = 0u;
  const std::int64_t* matrix_offsets = nullptr;
};

/* Caller-owned unpublished scalars used to make publication transactional. */
struct Gfn2SccEnergyDeviceWorkspace {
  double* core_energy_scratch = nullptr;
  double* electronic_free_energy_scratch = nullptr;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t batch_elements = 0;
  std::int64_t sequence_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2SccEnergyDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2SccEnergyDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2SccEnergyDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2SccEnergyDeviceWorkspace>);

/* Clear all per-system errors and the sticky first-error diagnostic. */
cudaError_t reset_gfn2_scc_energy_device_errors_cuda(std::int64_t batch_size,
                                                     std::uint32_t* system_errors,
                                                     std::uint32_t* device_error,
                                                     cudaStream_t stream = nullptr) noexcept;

/*
 * Evaluate Tr(P H0) and Tr(P H0) - temperature * entropy for each active
 * restricted system. P and H0 use complete dense row-major matrices; both
 * off-diagonal directions therefore participate exactly as in the CPU SCC
 * driver. entropy is dimensionless and temperature is in Hartree, so the
 * published electronic free-energy contribution is in Hartree.
 *
 * active_systems may be null to evaluate every system. Otherwise values must
 * be zero or one; inactive numerical matrices and entropy entries are not
 * inspected and retain their public output bytes. The shared immutable offset
 * partition is still validated as one plan-level contract before any system
 * arithmetic begins. Arithmetic failures are isolated per system. The
 * launcher allocates nothing, performs no transfer or synchronization,
 * supports custom streams and CUDA Graph capture, and publishes only after a
 * separate kernel boundary has validated the complete system reduction.
 */
cudaError_t evaluate_gfn2_scc_electronic_energy_cuda(
    const Gfn2SccEnergyDeviceBatch& batch, const double* density, const double* h0,
    const double* entropies, double electronic_temperature, const std::uint8_t* active_systems,
    double* core_energies, double* electronic_free_energies,
    const Gfn2SccEnergyDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

/* Canonical-ledger overload for one SCC iteration.  It checks the canonical
 * sequence before the active mask and validates only active matrix partitions,
 * so inactive topology and numerical slices may remain deliberately poisoned. */
cudaError_t evaluate_gfn2_scc_electronic_energy_cuda(
    const Gfn2SccEnergyDeviceBatch& batch, const double* density, const double* h0,
    const double* entropies, double electronic_temperature,
    const Gfn2SccIterationDeviceActivity& activity, double* core_energies,
    double* electronic_free_energies, const Gfn2SccEnergyDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

/*
 * Mixed-spin overload. Density matrices use WavefunctionLayout's system-major
 * spin packing, while H0 remains one physical matrix per system. Restricted
 * members therefore preserve the legacy contraction exactly; unrestricted
 * members accumulate alpha then beta Tr(P_sigma H0) in deterministic order.
 */
cudaError_t evaluate_gfn2_scc_electronic_energy_spin_cuda(
    const Gfn2SccEnergyDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const double* density, std::int64_t density_elements, const double* h0, const double* entropies,
    double electronic_temperature, const Gfn2SccIterationDeviceActivity& activity,
    double* core_energies, double* electronic_free_energies,
    const Gfn2SccEnergyDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_SCC_ENERGY_CUH
