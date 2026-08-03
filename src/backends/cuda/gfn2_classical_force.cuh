#ifndef GPUXTB_BACKENDS_CUDA_GFN2_CLASSICAL_FORCE_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_CLASSICAL_FORCE_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_aes2.cuh"
#include "backends/cuda/gfn2_d4.cuh"
#include "backends/cuda/gfn2_es2.cuh"
#include "backends/cuda/gfn2_force_common.cuh"
#include "backends/cuda/gfn2_geometry.cuh"

namespace gpuxtb::detail::cuda {

/* Fixed setup-time component mask for the non-electronic force reverse pass. */
enum class Gfn2ClassicalForceComponent : std::uint32_t {
  kRepulsion = 1u << 0u,
  kES2 = 1u << 1u,
  kAES2 = 1u << 2u,
  kD4TwoBody = 1u << 3u,
  kD4ATM = 1u << 4u,
};

inline constexpr std::uint32_t kGfn2ClassicalForceAllComponents =
    static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kRepulsion) |
    static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kES2) |
    static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kAES2) |
    static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4TwoBody) |
    static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4ATM);

/* Stable composer-local diagnostics; primitive error numbers remain private. */
enum class Gfn2ClassicalForceDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidActivity = 1u,
  kInvalidTopology = 2u,
  kNonfiniteForceSeed = 3u,
  kRepulsionFailure = 4u,
  kES2Failure = 5u,
  kAES2Failure = 6u,
  kAES2CoordinationFailure = 7u,
  kD4TwoBodyFailure = 8u,
  kD4ATMFailure = 9u,
  kNonfiniteForceArithmetic = 10u,
};

/*
 * Immutable post-SCC physics binding. The component descriptors are the same
 * generation-bound views used by the SCC iteration. atom_offsets and
 * atomic_numbers are repeated explicitly because repulsion is not an SCC
 * potential stage and therefore has no plan-token-bearing device descriptor.
 *
 * ES3 has no coordinate derivative. Externally supplied periodic b/A
 * operators are held fixed at this boundary, so their coordinate derivatives
 * are intentionally absent. Charge/multipole response is stationary at a
 * converged SCC solution and does not add a separate classical force term.
 */
struct Gfn2ClassicalForceDevicePlan {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::uint32_t enabled_components = 0u;
  std::uint64_t geometry_generation = 0u;
  std::uint64_t plan_token = 0u;

  const std::int64_t* atom_offsets = nullptr;
  const std::int32_t* atomic_numbers = nullptr;

  Gfn2GeometryDeviceBatch geometry_batch{};
  Gfn2GeometryDeviceCache geometry_cache{};
  Gfn2ES2DeviceBatch es2_batch{};
  Gfn2ES2DeviceCache es2_cache{};
  Gfn2AES2DeviceBatch aes2_batch{};
  Gfn2AES2DeviceCache aes2_cache{};
  Gfn2D4DeviceBatch d4_batch{};
  Gfn2D4DeviceParameters d4_parameters{};
  Gfn2D4DeviceCache d4_cache{};
};

/* Converged raw SCC multipoles and their generation-bound geometry. */
struct Gfn2ClassicalForceDeviceInput {
  const double* positions = nullptr;
  std::int64_t position_elements = 0;
  const double* coordination_numbers = nullptr;
  std::int64_t coordination_elements = 0;
  const double* shell_charges = nullptr;
  std::int64_t shell_elements = 0;
  const double* atomic_charges = nullptr;
  std::int64_t atom_elements = 0;
  const double* atomic_dipoles = nullptr;
  std::int64_t dipole_elements = 0;
  const double* atomic_quadrupoles = nullptr;
  std::int64_t quadrupole_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Existing public forces are finite-preflighted and incremented on success.
 * Values are physical forces F_classical = -dE_classical/dR in Hartree/bohr;
 * downstream composition must add them directly and must not negate again.
 */
struct Gfn2ClassicalForceDeviceOutput {
  double* forces = nullptr;
  std::int64_t force_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Caller-owned unpublished transaction storage. gradient_scratch accumulates
 * dE/dR; publication applies force = -gradient to the saved force seed.
 * primitive_system_errors and primitive_device_error are recycled between
 * component launches only after their result has been folded into the stable
 * composer diagnostics. selected_mask is the one-time terminal/request gate.
 *
 * The nested primitive workspaces must own storage disjoint from the composer
 * arrays and from one another. d4_workspace.system_errors must exactly alias
 * primitive_system_errors so failed/unrequested members can be pre-disabled
 * before a D4 primitive inspects any member-local numerical value.
 */
struct Gfn2ClassicalForceDeviceWorkspace {
  double* gradient_scratch = nullptr;
  std::int64_t gradient_elements = 0;
  double* force_scratch = nullptr;
  std::int64_t force_elements = 0;
  double* coordination_adjoints = nullptr;
  std::int64_t coordination_elements = 0;

  std::uint8_t* selected_mask = nullptr;
  std::int64_t selected_elements = 0;
  std::uint32_t* primitive_system_errors = nullptr;
  std::int64_t primitive_system_error_elements = 0;
  std::uint32_t* primitive_device_error = nullptr;
  std::int64_t primitive_device_error_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_elements = 0;

  Gfn2AES2DeviceWorkspace aes2_workspace{};
  Gfn2D4DeviceWorkspace d4_workspace{};
  Gfn2GeometryDeviceWorkspace geometry_workspace{};
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2ClassicalForceDevicePlan>);
static_assert(std::is_standard_layout_v<Gfn2ClassicalForceDevicePlan>);
static_assert(std::is_trivially_copyable_v<Gfn2ClassicalForceDeviceInput>);
static_assert(std::is_standard_layout_v<Gfn2ClassicalForceDeviceInput>);
static_assert(std::is_trivially_copyable_v<Gfn2ClassicalForceDeviceOutput>);
static_assert(std::is_standard_layout_v<Gfn2ClassicalForceDeviceOutput>);
static_assert(std::is_trivially_copyable_v<Gfn2ClassicalForceDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2ClassicalForceDeviceWorkspace>);

/* Clear public peer diagnostics and the sticky first-error scalar. */
cudaError_t reset_gfn2_classical_force_device_errors_cuda(std::int64_t batch_size,
                                                          std::uint32_t* system_errors,
                                                          std::uint32_t* device_error,
                                                          cudaStream_t stream = nullptr) noexcept;

/*
 * Accumulate all enabled non-electronic GFN2 forces for requested systems
 * whose terminal SCC status is SUCCESS. Unrequested and terminal-failed
 * systems are rejected before status-dependent numerical arrays are read and
 * retain their force bytes. A numerical component failure suppresses only the
 * affected member; a structural/plan failure suppresses the complete batch.
 *
 * Publication is transactional per system. The launcher allocates, transfers,
 * and synchronizes nothing, enqueues exclusively on stream, and is CUDA Graph
 * capture/replay compatible. Every pointer denotes caller-owned CUDA-accessible
 * storage whose lifetime extends through stream completion.
 */
cudaError_t add_gfn2_classical_forces_cuda(
    const Gfn2ClassicalForceDevicePlan& plan, const Gfn2ForceDeviceActivity& activity,
    const Gfn2ClassicalForceDeviceInput& input, const Gfn2ClassicalForceDeviceOutput& output,
    const Gfn2ClassicalForceDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

/* Replay-safe force reverse consuming geometry caches from the device epoch. */
cudaError_t add_gfn2_classical_forces_cuda(
    const Gfn2ClassicalForceDevicePlan& plan, const Gfn2ForceDeviceActivity& activity,
    const Gfn2ClassicalForceDeviceInput& input, const Gfn2ClassicalForceDeviceOutput& output,
    const Gfn2ClassicalForceDeviceWorkspace& workspace,
    const Gfn2GeometryEpochDevice& geometry_epoch, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_CLASSICAL_FORCE_CUH
