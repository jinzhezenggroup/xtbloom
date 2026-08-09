#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_TOTAL_ENERGY_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN2_TOTAL_ENERGY_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::cuda {

/* Non-SCC terms admitted by the restricted GFN2 total-energy composer. */
enum class Gfn2TotalEnergyComponent : std::uint32_t {
  kD4Atm = 1u << 0u,
};

inline constexpr std::uint32_t kGfn2TotalEnergyAllComponents =
    static_cast<std::uint32_t>(Gfn2TotalEnergyComponent::kD4Atm);

/* First peer-local semantic or arithmetic failure. */
enum class Gfn2TotalEnergyDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidConvergenceFlag = 1u,
  kInconsistentSccStatus = 2u,
  kNonfiniteSccFreeEnergy = 3u,
  kNonfiniteRepulsion = 4u,
  kNonfiniteD4Atm = 5u,
  kNonfiniteSccRepulsionSum = 6u,
  kNonfiniteTotalArithmetic = 7u,
};

/* Immutable final-energy shape and optional-component contract. */
struct Gfn2TotalEnergyDeviceBatch {
  std::int64_t batch_size = 0;
  std::uint32_t enabled_components = 0u;
  std::uint64_t plan_token = 0u;
};

/*
 * Component-major inputs. scc_free_energy is the converged restricted SCC
 * free energy produced by #57 and already includes self-consistent D4
 * two-body when enabled. repulsion is mandatory and D4 ATM is optional.
 */
struct Gfn2TotalEnergyDeviceInput {
  const double* scc_free_energy = nullptr;
  std::int64_t scc_free_energy_elements = 0;
  const double* repulsion = nullptr;
  std::int64_t repulsion_elements = 0;
  const double* d4_atm = nullptr;
  std::int64_t d4_atm_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Exact projection of the terminal SCC state needed for publication gating.
 * Only converged==1 and status==SUCCESS members may publish a total energy.
 * converged==0 members are skipped without inspecting their status or energy
 * inputs, which preserves failed and max-iteration peers in a ragged batch.
 */
struct Gfn2TotalEnergyDeviceSccState {
  const xtbloom_status_t* system_statuses = nullptr;
  const std::uint8_t* converged = nullptr;
  std::int64_t elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Successful converged members overwrite one Hartree total energy each. */
struct Gfn2TotalEnergyDeviceResults {
  double* total_energy = nullptr;
  std::int64_t elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * One caller-owned scalar snapshots a clean incoming sticky diagnostic so a
 * peer-local failure cannot suppress healthy peers scheduled later.
 */
struct Gfn2TotalEnergyDeviceWorkspace {
  std::uint32_t* sequence_active = nullptr;
  std::int64_t elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2TotalEnergyDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2TotalEnergyDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2TotalEnergyDeviceInput>);
static_assert(std::is_standard_layout_v<Gfn2TotalEnergyDeviceInput>);
static_assert(std::is_trivially_copyable_v<Gfn2TotalEnergyDeviceSccState>);
static_assert(std::is_standard_layout_v<Gfn2TotalEnergyDeviceSccState>);
static_assert(std::is_trivially_copyable_v<Gfn2TotalEnergyDeviceResults>);
static_assert(std::is_standard_layout_v<Gfn2TotalEnergyDeviceResults>);
static_assert(std::is_trivially_copyable_v<Gfn2TotalEnergyDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2TotalEnergyDeviceWorkspace>);

/* Clear one total-energy diagnostic generation asynchronously. */
cudaError_t reset_gfn2_total_energy_device_errors_cuda(std::int64_t batch_size,
                                                       std::uint32_t* system_errors,
                                                       std::uint32_t* device_error,
                                                       cudaStream_t stream = nullptr) noexcept;

/*
 * Compose converged restricted GFN2 totals in the CPU arithmetic order
 *
 *   total = SCC free energy; total += repulsion; total += q=0 D4 ATM.
 *
 * The ATM addition is omitted when kD4Atm is disabled. The SCC free energy
 * must not be an electronic-only subtotal: it already contains ES2, ES3,
 * AES2, optional self-consistent D4 two-body, point-charge, periodic, and
 * entropy terms. This prevents double counting the D4 two-body contribution.
 *
 * Every active input and intermediate is finite-preflighted before the single
 * output store. Failed and nonconverged members preserve their output bytes;
 * healthy peers remain independent. The launcher allocates, transfers, and
 * synchronizes nothing, is custom-stream safe, and is CUDA Graph capturable.
 * All writable ranges must be disjoint from inputs and from one another.
 */
cudaError_t compose_gfn2_total_energy_cuda(
    const Gfn2TotalEnergyDeviceBatch& batch, const Gfn2TotalEnergyDeviceInput& input,
    const Gfn2TotalEnergyDeviceSccState& scc_state, const Gfn2TotalEnergyDeviceResults& results,
    const Gfn2TotalEnergyDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_TOTAL_ENERGY_CUH
