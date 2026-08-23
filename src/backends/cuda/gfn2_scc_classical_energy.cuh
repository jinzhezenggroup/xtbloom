#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_SCC_CLASSICAL_ENERGY_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN2_SCC_CLASSICAL_ENERGY_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/common/xtb_model.hpp"
#include "backends/cuda/gfn2_electric_field.cuh"

namespace xtbloom::detail::cuda {

inline constexpr std::int64_t kGfn2SccClassicalInputComponents = 6;
inline constexpr std::int64_t kGfn2SccClassicalDiagnosticComponents = 7;
inline constexpr std::int64_t kGfn2SccClassicalStorageComponents = 8;

/* Bit positions used by Gfn2SccClassicalEnergyDeviceBatch::enabled_components. */
enum class Gfn2SccClassicalEnergyComponent : std::uint32_t {
  kES2 = 1u << 0u,
  kES3 = 1u << 1u,
  kAES2 = 1u << 2u,
  kD4TwoBody = 1u << 3u,
  kExplicitPointCharge = 1u << 4u,
  kPeriodicEmbedding = 1u << 5u,
};

inline constexpr std::uint32_t kGfn2SccClassicalAllComponents =
    static_cast<std::uint32_t>(Gfn2SccClassicalEnergyComponent::kES2) |
    static_cast<std::uint32_t>(Gfn2SccClassicalEnergyComponent::kES3) |
    static_cast<std::uint32_t>(Gfn2SccClassicalEnergyComponent::kAES2) |
    static_cast<std::uint32_t>(Gfn2SccClassicalEnergyComponent::kD4TwoBody) |
    static_cast<std::uint32_t>(Gfn2SccClassicalEnergyComponent::kExplicitPointCharge) |
    static_cast<std::uint32_t>(Gfn2SccClassicalEnergyComponent::kPeriodicEmbedding);

/* First asynchronous semantic or arithmetic failure in an aggregation sequence. */
enum class Gfn2SccClassicalEnergyDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidActiveMask = 1u,
  kNonfiniteES2 = 2u,
  kNonfiniteES3 = 3u,
  kNonfiniteAES2 = 4u,
  kNonfiniteD4TwoBody = 5u,
  kNonfiniteExplicitPointCharge = 6u,
  kNonfinitePeriodicEmbedding = 7u,
  kNonfiniteTotalArithmetic = 8u,
  kNonfiniteElectricField = 9u,
};

/*
 * Fixed-topology descriptor for one reusable restricted SCC batch. Optional
 * components are selected once by enabled_components. A disabled component
 * has no readable input range and publishes an exact zero diagnostic.
 */
struct Gfn2SccClassicalEnergyDeviceBatch {
  std::int64_t batch_size = 0;
  std::uint32_t enabled_components = 0u;
  std::uint64_t plan_token = 0u;
  /* Zero plan_token keeps legacy field-free bindings valid. */
  Gfn2ElectricFieldDeviceBatch electric_field{};
  XtbModelFlavor model = XtbModelFlavor::kGfn2;
};

/*
 * Pure per-system component energies produced by the CUDA physics kernels.
 * Each enabled pointer has exactly batch_size elements; disabled pointers must
 * be null with a zero extent. plan_token binds these arrays to the batch.
 *
 * The caller must bind density-derived raw Mulliken multipoles, not the mixed
 * Hamiltonian inputs. ES2, ES3, AES2, and explicit point-charge APIs accumulate
 * into an existing seed, so their destination arrays must be zeroed before the
 * physics launch when a pure component diagnostic is required. D4 is the
 * charge-dependent two-body energy evaluated at raw q; q=0 ATM is excluded.
 * Explicit point charge is sum(q_raw*V_pc), with no one-half. Periodic embedding
 * is q_raw^T*b + one-half*q_raw^T*A*q_raw.
 */
struct Gfn2SccClassicalEnergyDeviceInput {
  const double* es2 = nullptr;
  std::int64_t es2_elements = 0;
  const double* es3 = nullptr;
  std::int64_t es3_elements = 0;
  const double* aes2 = nullptr;
  std::int64_t aes2_elements = 0;
  const double* d4_two_body = nullptr;
  std::int64_t d4_two_body_elements = 0;
  const double* explicit_point_charge = nullptr;
  std::int64_t explicit_point_charge_elements = 0;
  const double* periodic_embedding = nullptr;
  std::int64_t periodic_embedding_elements = 0;
  std::uint64_t plan_token = 0u;
  Gfn2ElectricFieldDeviceMultipoles electric_field_multipoles{};
  Gfn2ElectricFieldDevicePotentialView electric_field_potentials{};
};

/*
 * Optional activity map. A null pointer with zero elements evaluates every
 * system. Otherwise values are exactly zero (inactive) or one (active).
 * Inactive systems are skipped before component inputs are inspected and keep
 * every public diagnostic byte unchanged.
 */
struct Gfn2SccClassicalEnergyDeviceActivity {
  const std::uint8_t* active_mask = nullptr;
  std::int64_t elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Every public array has exactly batch_size elements and is overwritten on success. */
struct Gfn2SccClassicalEnergyDeviceDiagnostics {
  double* es2 = nullptr;
  std::int64_t es2_elements = 0;
  double* es3 = nullptr;
  std::int64_t es3_elements = 0;
  double* aes2 = nullptr;
  std::int64_t aes2_elements = 0;
  double* d4_two_body = nullptr;
  std::int64_t d4_two_body_elements = 0;
  double* explicit_point_charge = nullptr;
  std::int64_t explicit_point_charge_elements = 0;
  double* periodic_embedding = nullptr;
  std::int64_t periodic_embedding_elements = 0;
  double* classical_total = nullptr;
  std::int64_t classical_total_elements = 0;
  std::uint64_t plan_token = 0u;
  double* electric_field = nullptr;
  std::int64_t electric_field_elements = 0;
};

/*
 * Caller-owned unpublished storage. component_scratch is component-major in
 * diagnostic order [ES2, ES3, AES2, D4-2body, explicit-PC, periodic, total,
 * field] and therefore requires 8*batch_size doubles. Keeping the field in the
 * appended slot preserves every pre-field diagnostic offset. sequence_active snapshots a
 * clean incoming device_error before peer-local arithmetic can make it sticky.
 */
struct Gfn2SccClassicalEnergyDeviceWorkspace {
  double* component_scratch = nullptr;
  std::int64_t component_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2SccClassicalEnergyDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2SccClassicalEnergyDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2SccClassicalEnergyDeviceInput>);
static_assert(std::is_standard_layout_v<Gfn2SccClassicalEnergyDeviceInput>);
static_assert(std::is_trivially_copyable_v<Gfn2SccClassicalEnergyDeviceActivity>);
static_assert(std::is_standard_layout_v<Gfn2SccClassicalEnergyDeviceActivity>);
static_assert(std::is_trivially_copyable_v<Gfn2SccClassicalEnergyDeviceDiagnostics>);
static_assert(std::is_standard_layout_v<Gfn2SccClassicalEnergyDeviceDiagnostics>);
static_assert(std::is_trivially_copyable_v<Gfn2SccClassicalEnergyDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2SccClassicalEnergyDeviceWorkspace>);

/* Clear all peer-local errors and the sticky first-error diagnostic asynchronously. */
cudaError_t reset_gfn2_scc_classical_energy_device_errors_cuda(
    std::int64_t batch_size, std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

/*
 * Validate, sum, and transactionally publish all enabled classical SCC energy
 * diagnostics. The total uses the CPU driver's fixed component order: ES2,
 * ES3, AES2, D4 two-body, explicit point charge, electric field, then
 * periodic embedding.
 * Component inputs are all finite-preflighted before the first addition. A
 * failed active member publishes nothing, while healthy peers commit normally.
 *
 * This hot launcher allocates nothing, transfers nothing, synchronizes nothing,
 * enqueues only on stream, and supports CUDA Graph capture. Read-only component
 * arrays may alias one another. Every writable range must be disjoint from all
 * readable ranges and from every other writable range. CUDA address-space
 * provenance is a setup/binding responsibility and is not queried here.
 */
cudaError_t evaluate_gfn2_scc_classical_energy_cuda(
    const Gfn2SccClassicalEnergyDeviceBatch& batch, const Gfn2SccClassicalEnergyDeviceInput& input,
    const Gfn2SccClassicalEnergyDeviceActivity& activity,
    const Gfn2SccClassicalEnergyDeviceDiagnostics& diagnostics,
    const Gfn2SccClassicalEnergyDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_SCC_CLASSICAL_ENERGY_CUH
