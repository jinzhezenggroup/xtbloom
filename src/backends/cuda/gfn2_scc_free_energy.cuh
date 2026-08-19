#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_SCC_FREE_ENERGY_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN2_SCC_FREE_ENERGY_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_scc_classical_energy.cuh"

namespace xtbloom::detail::cuda {

inline constexpr std::int64_t kGfn2SccFreeEnergyInputComponents = 9;
inline constexpr std::int64_t kGfn2SccFreeEnergyDiagnosticComponents = 11;
inline constexpr std::int64_t kGfn2SccFreeEnergyStorageComponents = 12;

/* First asynchronous peer-local semantic or arithmetic failure. */
enum class Gfn2SccFreeEnergyDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidActiveMask = 1u,
  kNonfiniteCore = 2u,
  kNonfiniteEntropy = 3u,
  kNonfiniteES2 = 4u,
  kNonfiniteES3 = 5u,
  kNonfiniteAES2 = 6u,
  kNonfiniteD4TwoBody = 7u,
  kNonfiniteExplicitPointCharge = 8u,
  kNonfinitePeriodicEmbedding = 9u,
  kNonfiniteInternalArithmetic = 10u,
  kNonfiniteFreeEnergyArithmetic = 11u,
  /* Appended to preserve every pre-existing stage-local diagnostic value. */
  kNonfiniteSpin = 12u,
  kNonfiniteElectricField = 13u,
};

/*
 * Immutable restricted-SCC composition contract. enabled_components uses the
 * same bits as Gfn2SccClassicalEnergyDeviceBatch so #75 outputs can be bound
 * without translating a plan. Disabled classical terms are exact zeros.
 */
struct Gfn2SccFreeEnergyDeviceBatch {
  std::int64_t batch_size = 0;
  std::uint32_t enabled_components = 0u;
  double electronic_temperature = 0.0;
  std::uint64_t plan_token = 0u;
  XtbModelFlavor model = XtbModelFlavor::kGfn2;
};

/*
 * Per-system inputs from the native CUDA occupation/electronic-energy and
 * classical-energy stages. Core and entropy are mandatory. Each enabled
 * classical pointer has exactly batch_size elements; a disabled pointer must
 * be null with a zero extent. Read-only input arrays may alias one another.
 */
struct Gfn2SccFreeEnergyDeviceInput {
  const double* core = nullptr;
  std::int64_t core_elements = 0;
  const double* entropy = nullptr;
  std::int64_t entropy_elements = 0;
  const double* es2 = nullptr;
  std::int64_t es2_elements = 0;
  const double* es3 = nullptr;
  std::int64_t es3_elements = 0;
  const double* aes2 = nullptr;
  std::int64_t aes2_elements = 0;
  const double* spin = nullptr;
  std::int64_t spin_elements = 0;
  const double* d4_two_body = nullptr;
  std::int64_t d4_two_body_elements = 0;
  const double* explicit_point_charge = nullptr;
  std::int64_t explicit_point_charge_elements = 0;
  const double* periodic_embedding = nullptr;
  std::int64_t periodic_embedding_elements = 0;
  std::uint64_t plan_token = 0u;
  const double* electric_field = nullptr;
  std::int64_t electric_field_elements = 0;
};

/*
 * A null activity pointer with zero elements evaluates every system. Otherwise
 * mask values are exactly zero (skip without inspecting inputs) or one.
 */
struct Gfn2SccFreeEnergyDeviceActivity {
  const std::uint8_t* active_mask = nullptr;
  std::int64_t elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Complete CPU-compatible SCC diagnostics. Every pointer has exactly
 * batch_size elements. Successful active systems overwrite every diagnostic;
 * inactive or failed systems preserve all public bytes.
 */
struct Gfn2SccFreeEnergyDeviceDiagnostics {
  double* core = nullptr;
  std::int64_t core_elements = 0;
  double* es2 = nullptr;
  std::int64_t es2_elements = 0;
  double* es3 = nullptr;
  std::int64_t es3_elements = 0;
  double* aes2 = nullptr;
  std::int64_t aes2_elements = 0;
  double* spin = nullptr;
  std::int64_t spin_elements = 0;
  double* d4_two_body = nullptr;
  std::int64_t d4_two_body_elements = 0;
  double* explicit_point_charge = nullptr;
  std::int64_t explicit_point_charge_elements = 0;
  double* periodic_embedding = nullptr;
  std::int64_t periodic_embedding_elements = 0;
  double* entropy = nullptr;
  std::int64_t entropy_elements = 0;
  double* internal_energy = nullptr;
  std::int64_t internal_energy_elements = 0;
  double* free_energy = nullptr;
  std::int64_t free_energy_elements = 0;
  std::uint64_t plan_token = 0u;
  double* electric_field = nullptr;
  std::int64_t electric_field_elements = 0;
};

/*
 * Caller-owned unpublished storage. diagnostic_scratch is component-major in
 * public diagnostic order [core, ES2, ES3, AES2, spin, D4-2body,
 * explicit-PC, periodic, entropy, internal, free, field] and requires
 * 12*batch_size doubles. The appended field slot preserves every pre-field
 * diagnostic offset.
 * sequence_active snapshots a clean incoming sticky diagnostic before any
 * peer-local failure can update device_error.
 */
struct Gfn2SccFreeEnergyDeviceWorkspace {
  double* diagnostic_scratch = nullptr;
  std::int64_t diagnostic_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2SccFreeEnergyDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2SccFreeEnergyDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2SccFreeEnergyDeviceInput>);
static_assert(std::is_standard_layout_v<Gfn2SccFreeEnergyDeviceInput>);
static_assert(std::is_trivially_copyable_v<Gfn2SccFreeEnergyDeviceActivity>);
static_assert(std::is_standard_layout_v<Gfn2SccFreeEnergyDeviceActivity>);
static_assert(std::is_trivially_copyable_v<Gfn2SccFreeEnergyDeviceDiagnostics>);
static_assert(std::is_standard_layout_v<Gfn2SccFreeEnergyDeviceDiagnostics>);
static_assert(std::is_trivially_copyable_v<Gfn2SccFreeEnergyDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2SccFreeEnergyDeviceWorkspace>);

/* Clear every peer-local error and the sticky first-error diagnostic. */
cudaError_t reset_gfn2_scc_free_energy_device_errors_cuda(std::int64_t batch_size,
                                                          std::uint32_t* system_errors,
                                                          std::uint32_t* device_error,
                                                          cudaStream_t stream = nullptr) noexcept;

/*
 * Compose complete mixed-spin SCC energies in the exact CPU driver order:
 * core, ES2, ES3, AES2, spin, D4 two-body, explicit point charge, electric
 * field, periodic embedding, followed by one fma(-temperature, entropy,
 * internal_energy).
 * Every input and intermediate is finite-preflighted before transactional
 * publication. This intentionally does not consume electronic_free_energy or
 * classical_total because adding those rounded partials is not equivalent.
 *
 * The launcher allocates, transfers, and synchronizes nothing; it uses only
 * stream, supports CUDA Graph capture/replay, and isolates numerical failures
 * per system. Every writable range must be disjoint from all readable ranges
 * and from every other writable range, except diagnostics.spin/input.spin and
 * diagnostics.entropy/input.entropy and
 * diagnostics.electric_field/input.electric_field may exactly alias their
 * matching arrays. These zero-copy edges are safe because the reduction
 * snapshots every input in unpublished scratch before the later publication
 * kernel writes the identical values.
 */
cudaError_t compose_gfn2_scc_free_energy_cuda(const Gfn2SccFreeEnergyDeviceBatch& batch,
                                              const Gfn2SccFreeEnergyDeviceInput& input,
                                              const Gfn2SccFreeEnergyDeviceActivity& activity,
                                              const Gfn2SccFreeEnergyDeviceDiagnostics& diagnostics,
                                              const Gfn2SccFreeEnergyDeviceWorkspace& workspace,
                                              std::uint32_t* system_errors,
                                              std::uint32_t* device_error,
                                              cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_SCC_FREE_ENERGY_CUH
