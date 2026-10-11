#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_TERMINAL_CLASSICAL_ENERGY_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN2_TERMINAL_CLASSICAL_ENERGY_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_d4.cuh"
#include "backends/cuda/gfn2_device_admission.cuh"
#include "backends/cuda/gfn2_geometry.cuh"
#include "backends/cuda/gfn2_native_periodic_d4.cuh"
#include "backends/cuda/gfn2_native_periodic_short_range.cuh"
#include "backends/cuda/gfn2_repulsion.cuh"

namespace xtbloom::detail::cuda {

/* ABI v2 replaces the legacy dense D4 cache with the committed pair-list
 * cache and its role-specific consumer views. */
inline constexpr std::uint32_t kGfn2TerminalClassicalEnergyAbiVersion = 2u;

enum class Gfn2TerminalClassicalEnergyComponent : std::uint32_t {
  kD4Atm = 1u << 0u,
};

inline constexpr std::uint32_t kGfn2TerminalClassicalEnergyAllComponents =
    static_cast<std::uint32_t>(Gfn2TerminalClassicalEnergyComponent::kD4Atm);

/* Stable plan-wide failures. A nonzero code suppresses every public write. */
enum class Gfn2TerminalClassicalEnergyPlanError : std::uint32_t {
  kSuccess = 0u,
  kInvalidRequestedMask = 1u,
  kInvalidEpoch = 2u,
  kRepulsionFailure = 3u,
  kD4AtmPlanFailure = 4u,
};

/* Per-member failures. Healthy peers still publish their complete component tuple. */
enum class Gfn2TerminalClassicalEnergySystemError : std::uint32_t {
  kSuccess = 0u,
  kStaleGeneration = 1u,
  kD4AtmFailure = 2u,
  kNonfiniteRepulsion = 3u,
  kNonfiniteD4Atm = 4u,
};

/*
 * Immutable terminal-classical plan.
 *
 * geometry_epoch and committed_generations form the runtime generation gate:
 * a requested member may publish only when its complete numerical transaction
 * was committed for the device epoch captured at the head of this launch.
 * The D4 descriptors are canonical zero when kD4Atm is disabled.
 */
struct Gfn2TerminalClassicalEnergyDevicePlan {
  std::uint32_t abi_version = kGfn2TerminalClassicalEnergyAbiVersion;
  std::uint32_t enabled_components = 0u;
  std::uint64_t plan_token = 0u;

  Gfn2RepulsionDeviceBatch repulsion{};
  Gfn2D4DeviceBatch d4_batch{};
  Gfn2D4DeviceParameters d4_parameters{};
  Gfn2D4PairListDeviceCache d4_cache{};

  Gfn2GeometryEpochDevice geometry_epoch{};
  const std::uint64_t* committed_generations = nullptr;
  std::int64_t generation_elements = 0;
  /* Native XYZ periodic repulsion is evaluated during numerical refresh,
   * alongside periodic CN, and consumed here as a read-only candidate.  A
   * null pointer retains the molecular repulsion leaf for non-periodic plans. */
  const double* native_repulsion_energies = nullptr;
  std::int64_t native_repulsion_elements = 0;
  /* Native XYZ periodic D4 uses the SCC arena's image-aware evaluator.  A
   * zero-token view keeps the legacy molecular pair-list route unchanged. */
  Gfn2NativePeriodicD4DeviceBatch native_d4_batch{};
};

/* One byte per system. Zero leaves that member's result tuple unchanged. */
struct Gfn2TerminalClassicalEnergyDeviceActivity {
  const std::uint8_t* requested_mask = nullptr;
  std::int64_t batch_elements = 0;
  std::uint64_t plan_token = 0u;
  Gfn2DeviceAdmission admission{};
};

/* Component values consumed by the total-energy composer. */
struct Gfn2TerminalClassicalEnergyDeviceResults {
  double* repulsion = nullptr;
  std::int64_t repulsion_elements = 0;
  double* d4_atm = nullptr;
  std::int64_t d4_atm_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Unpublished primitive outputs. Repulsion is additive and therefore is
 * zeroed before its primitive launch; D4 ATM overwrites its candidate slice.
 * The nested D4 workspace is reused later by the force chain only after this
 * stream-ordered launch has completed.
 */
struct Gfn2TerminalClassicalEnergyDeviceWorkspace {
  double* repulsion_candidate = nullptr;
  std::int64_t repulsion_elements = 0;
  double* d4_atm_candidate = nullptr;
  std::int64_t d4_atm_elements = 0;
  Gfn2D4DeviceWorkspace d4{};
  std::uint64_t* epoch_snapshot = nullptr;
  std::int64_t epoch_snapshot_elements = 0;
  std::uint64_t plan_token = 0u;
  Gfn2NativePeriodicD4DeviceWorkspace native_d4_workspace{};
};

/* Native primitive diagnostics plus normalized execution-level projections. */
struct Gfn2TerminalClassicalEnergyDeviceDiagnostics {
  std::uint32_t* repulsion_device_error = nullptr;
  std::uint32_t* d4_system_errors = nullptr;
  std::int64_t d4_system_error_elements = 0;
  std::uint32_t* d4_device_error = nullptr;
  std::uint32_t* system_errors = nullptr;
  std::int64_t system_error_elements = 0;
  std::uint32_t* plan_error = nullptr;
  std::int64_t plan_error_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2TerminalClassicalEnergyDevicePlan>);
static_assert(std::is_standard_layout_v<Gfn2TerminalClassicalEnergyDevicePlan>);
static_assert(std::is_trivially_copyable_v<Gfn2TerminalClassicalEnergyDeviceActivity>);
static_assert(std::is_standard_layout_v<Gfn2TerminalClassicalEnergyDeviceActivity>);
static_assert(std::is_trivially_copyable_v<Gfn2TerminalClassicalEnergyDeviceResults>);
static_assert(std::is_standard_layout_v<Gfn2TerminalClassicalEnergyDeviceResults>);
static_assert(std::is_trivially_copyable_v<Gfn2TerminalClassicalEnergyDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2TerminalClassicalEnergyDeviceWorkspace>);
static_assert(std::is_trivially_copyable_v<Gfn2TerminalClassicalEnergyDeviceDiagnostics>);
static_assert(std::is_standard_layout_v<Gfn2TerminalClassicalEnergyDeviceDiagnostics>);

/*
 * Evaluate nuclear repulsion and optional q=0 D4 ATM energy on stream.
 *
 * Both primitives write candidate storage. A final generation/error gate
 * publishes one complete pair of component values for each healthy requested
 * member. Peer-local D4 or nonfinite-output failures leave that member's
 * result bytes unchanged. Any plan-wide primitive or epoch failure suppresses
 * publication for the complete batch. The launcher allocates, transfers,
 * polls, and synchronizes nowhere and is CUDA Graph capture/replay safe.
 */
cudaError_t evaluate_gfn2_terminal_classical_energy_cuda(
    const Gfn2TerminalClassicalEnergyDevicePlan& plan,
    const Gfn2TerminalClassicalEnergyDeviceActivity& activity,
    const Gfn2TerminalClassicalEnergyDeviceResults& results,
    const Gfn2TerminalClassicalEnergyDeviceWorkspace& workspace,
    const Gfn2TerminalClassicalEnergyDeviceDiagnostics& diagnostics,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_TERMINAL_CLASSICAL_ENERGY_CUH
