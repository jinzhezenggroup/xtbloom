#ifndef GPUXTB_BACKENDS_CUDA_GFN2_POST_SCC_POTENTIAL_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_POST_SCC_POTENTIAL_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_aes2.cuh"
#include "backends/cuda/gfn2_d4.cuh"
#include "backends/cuda/gfn2_es2.cuh"
#include "backends/cuda/gfn2_es3.cuh"
#include "backends/cuda/gfn2_external_point_charges.cuh"
#include "backends/cuda/gfn2_force_common.cuh"
#include "backends/cuda/gfn2_geometry.cuh"
#include "backends/cuda/gfn2_periodic_embedding.cuh"
#include "backends/cuda/gfn2_scc_bridge.cuh"
#include "backends/cuda/gfn2_scc_potential.cuh"

namespace gpuxtb::detail::cuda {

/* Stable stage identities encoded in post-SCC system/device diagnostics. */
enum class Gfn2PostSccPotentialStage : std::uint32_t {
  kSuccess = 0u,
  kActivity = 1u,
  kES2 = 2u,
  kES3 = 3u,
  kAES2 = 4u,
  kD4 = 5u,
  kPeriodicEmbedding = 6u,
  kComposition = 7u,
  kScalarBridge = 8u,
};

/* Pack a stage-local raw code without conflating primitive error domains. */
[[nodiscard]] inline __host__ __device__ constexpr std::uint32_t gfn2_post_scc_potential_error(
    Gfn2PostSccPotentialStage stage, std::uint32_t raw_code) noexcept {
  return (static_cast<std::uint32_t>(stage) << 24u) | (raw_code & 0x00ffffffu);
}

[[nodiscard]] inline __host__ __device__ constexpr Gfn2PostSccPotentialStage
gfn2_post_scc_potential_error_stage(std::uint32_t error) noexcept {
  return static_cast<Gfn2PostSccPotentialStage>(error >> 24u);
}

[[nodiscard]] inline __host__ __device__ constexpr std::uint32_t gfn2_post_scc_potential_raw_error(
    std::uint32_t error) noexcept {
  return error & 0x00ffffffu;
}

/*
 * Immutable bindings used to rebuild Hamiltonian potentials from the final
 * raw SCC state. ES2, ES3, and AES2 are mandatory GFN2 components.
 * Self-consistent D4, explicit point charges, and periodic embedding are
 * mask-controlled and must exactly follow the SCC/energy execution mask.
 */
struct Gfn2PostSccPotentialDevicePlan {
  std::uint32_t enabled_components = 0u;
  std::uint64_t geometry_generation = 0u;
  std::uint64_t plan_token = 0u;

  Gfn2SccPotentialDeviceBatch potential_batch{};
  Gfn2SccBridgeDeviceBatch scalar_bridge_batch{};
  Gfn2ES2DeviceBatch es2_batch{};
  Gfn2ES2DeviceCache es2_cache{};
  Gfn2ES3DeviceBatch es3_batch{};
  Gfn2AES2DeviceBatch aes2_batch{};
  Gfn2AES2DeviceCache aes2_cache{};
  Gfn2D4DeviceBatch d4_batch{};
  Gfn2D4DeviceParameters d4_parameters{};
  Gfn2D4DeviceCache d4_cache{};
  Gfn2ExternalPointChargeDeviceBatch external_point_charge_batch{};
  Gfn2ExternalPointChargeDeviceCache external_point_charge_cache{};
  Gfn2PeriodicEmbeddingDeviceBatch periodic_batch{};
};

/* Final raw multipoles and the common post-SCC requested/status gate. */
struct Gfn2PostSccPotentialDeviceInput {
  Gfn2ForceDeviceActivity activity{};
  const double* raw_shell_charges = nullptr;
  std::int64_t shell_elements = 0;
  const double* raw_atomic_charges = nullptr;
  std::int64_t atom_elements = 0;
  const double* raw_atomic_dipoles = nullptr;
  std::int64_t dipole_elements = 0;
  const double* raw_atomic_quadrupoles = nullptr;
  std::int64_t quadrupole_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * complete retains the composer field layout. shell_scalar additionally owns
 * complete_vsh = field_vsh + field_vat(shell_to_atom), which is the scalar
 * potential consumed by the post-SCC Hamiltonian force contraction.
 */
struct Gfn2PostSccPotentialDeviceResults {
  Gfn2SccPotentialDeviceResults complete{};
  double* shell_scalar = nullptr;
  std::int64_t shell_scalar_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Caller-owned component buffers; none are externally published. */
struct Gfn2PostSccPotentialDeviceIntermediates {
  double* es2_shell = nullptr;
  std::int64_t es2_shell_elements = 0;
  double* es3_shell = nullptr;
  std::int64_t es3_shell_elements = 0;
  double* aes2_atomic = nullptr;
  std::int64_t aes2_atomic_elements = 0;
  double* aes2_dipole = nullptr;
  std::int64_t aes2_dipole_elements = 0;
  double* aes2_quadrupole = nullptr;
  std::int64_t aes2_quadrupole_elements = 0;
  double* d4_atomic = nullptr;
  std::int64_t d4_atomic_elements = 0;
  double* periodic_atomic = nullptr;
  std::int64_t periodic_atomic_elements = 0;
  Gfn2SccPotentialDeviceResults complete{};
  double* shell_scalar = nullptr;
  std::int64_t shell_scalar_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Reusable storage for the component primitives and the real post-SCC
 * activity adapter. stage_system_errors/stage_device_error are reset and
 * folded after each primitive; d4.system_errors must alias stage_system_errors.
 */
struct Gfn2PostSccPotentialDeviceWorkspace {
  Gfn2ES2DeviceWorkspace es2{};
  Gfn2AES2DeviceWorkspace aes2{};
  Gfn2D4DeviceWorkspace d4{};
  Gfn2PeriodicEmbeddingDeviceWorkspace periodic{};
  Gfn2SccPotentialDeviceWorkspace composition{};
  Gfn2SccBridgeDeviceWorkspace scalar_bridge{};

  std::uint8_t* active_mask = nullptr;
  std::int64_t active_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_elements = 0;
  std::uint32_t* stage_system_errors = nullptr;
  std::int64_t stage_system_error_elements = 0;
  std::uint32_t* stage_device_error = nullptr;
  std::int64_t stage_device_error_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Stable execution-level error projection. Zero means success/unrequested. */
struct Gfn2PostSccPotentialDeviceDiagnostics {
  std::uint32_t* system_errors = nullptr;
  std::uint32_t* device_error = nullptr;
  std::int64_t batch_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2PostSccPotentialDevicePlan>);
static_assert(std::is_standard_layout_v<Gfn2PostSccPotentialDevicePlan>);
static_assert(std::is_trivially_copyable_v<Gfn2PostSccPotentialDeviceInput>);
static_assert(std::is_standard_layout_v<Gfn2PostSccPotentialDeviceInput>);
static_assert(std::is_trivially_copyable_v<Gfn2PostSccPotentialDeviceResults>);
static_assert(std::is_standard_layout_v<Gfn2PostSccPotentialDeviceResults>);
static_assert(std::is_trivially_copyable_v<Gfn2PostSccPotentialDeviceIntermediates>);
static_assert(std::is_standard_layout_v<Gfn2PostSccPotentialDeviceIntermediates>);
static_assert(std::is_trivially_copyable_v<Gfn2PostSccPotentialDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2PostSccPotentialDeviceWorkspace>);
static_assert(std::is_trivially_copyable_v<Gfn2PostSccPotentialDeviceDiagnostics>);
static_assert(std::is_standard_layout_v<Gfn2PostSccPotentialDeviceDiagnostics>);

/*
 * Rebuild all complete GFN2 Hamiltonian potentials from final raw converged
 * multipoles. The function never reads an SCC iteration's last-mixed fields,
 * allocates, polls, transfers, or synchronizes. Healthy ragged peers publish
 * transactionally on the caller stream and Graph replay uses changed inputs.
 */
cudaError_t refresh_gfn2_post_scc_potentials_cuda(
    const Gfn2PostSccPotentialDevicePlan& plan, const Gfn2PostSccPotentialDeviceInput& input,
    const Gfn2PostSccPotentialDeviceResults& results,
    const Gfn2PostSccPotentialDeviceIntermediates& intermediates,
    const Gfn2PostSccPotentialDeviceWorkspace& workspace,
    const Gfn2PostSccPotentialDeviceDiagnostics& diagnostics,
    cudaStream_t stream = nullptr) noexcept;

/* Replay-safe refresh gated by the runtime-owned geometry transaction. */
cudaError_t refresh_gfn2_post_scc_potentials_cuda(
    const Gfn2PostSccPotentialDevicePlan& plan, const Gfn2PostSccPotentialDeviceInput& input,
    const Gfn2PostSccPotentialDeviceResults& results,
    const Gfn2PostSccPotentialDeviceIntermediates& intermediates,
    const Gfn2PostSccPotentialDeviceWorkspace& workspace,
    const Gfn2PostSccPotentialDeviceDiagnostics& diagnostics,
    const Gfn2GeometryEpochConsumerDevice& geometry,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_POST_SCC_POTENTIAL_CUH
