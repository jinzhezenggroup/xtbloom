#ifndef GPUXTB_BACKENDS_CUDA_GFN2_SCC_POTENTIAL_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_SCC_POTENTIAL_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_scc_iteration_control.cuh"

namespace gpuxtb::detail::cuda {

inline constexpr std::int64_t kGfn2SccPotentialDipoleComponents = 3;
inline constexpr std::int64_t kGfn2SccPotentialQuadrupoleComponents = 6;

enum class Gfn2SccPotentialComponent : std::uint32_t {
  kES2 = 1u << 0u,
  kES3 = 1u << 1u,
  kAES2 = 1u << 2u,
  kD4TwoBody = 1u << 3u,
  kExplicitPointCharge = 1u << 4u,
  kPeriodicEmbedding = 1u << 5u,
};

inline constexpr std::uint32_t kGfn2SccPotentialAllComponents =
    static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES2) |
    static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES3) |
    static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kAES2) |
    static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kD4TwoBody) |
    static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kExplicitPointCharge) |
    static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kPeriodicEmbedding);

/* First asynchronous semantic or arithmetic failure in a potential sequence. */
enum class Gfn2SccPotentialDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kInvalidShellMetadata = 2u,
  kInvalidActiveMask = 3u,
  kNonfiniteMixedShellCharge = 4u,
  kNonfiniteMixedDipole = 5u,
  kNonfiniteMixedQuadrupole = 6u,
  kNonfiniteAtomicChargeReduction = 7u,
  kNonfiniteES2Potential = 8u,
  kNonfiniteES3Potential = 9u,
  kNonfiniteAES2Potential = 10u,
  kNonfiniteD4Potential = 11u,
  kNonfiniteExplicitPointChargePotential = 12u,
  kNonfinitePeriodicPotential = 13u,
  kNonfiniteShellPotentialArithmetic = 14u,
  kNonfiniteAtomicPotentialArithmetic = 15u,
};

/*
 * Common ragged mapping between wavefunction field layouts and topology-major
 * component-kernel layouts. Dipole/quadrupole offsets count scalar elements,
 * not atoms. Empty systems are allowed. shell_to_atom contains global atoms.
 */
struct Gfn2SccPotentialDeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::uint64_t plan_token = 0u;

  std::int64_t atom_offset_count = 0;
  std::int64_t batch_shell_offset_count = 0;
  std::int64_t qsh_offset_count = 0;
  std::int64_t qat_offset_count = 0;
  std::int64_t dipole_offset_count = 0;
  std::int64_t quadrupole_offset_count = 0;
  std::int64_t shell_to_atom_count = 0;

  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* batch_shell_offsets = nullptr;
  const std::int64_t* qsh_offsets = nullptr;
  const std::int64_t* qat_offsets = nullptr;
  const std::int64_t* dipole_offsets = nullptr;
  const std::int64_t* quadrupole_offsets = nullptr;
  const std::int64_t* shell_to_atom = nullptr;
};

struct Gfn2SccPotentialDeviceActivity {
  const std::uint8_t* active_mask = nullptr;
  std::int64_t elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Mixed wavefunction fields used to build the next Hamiltonian. */
struct Gfn2SccPotentialDeviceMixedFields {
  const double* qsh = nullptr;
  std::int64_t qsh_elements = 0;
  const double* dipoles = nullptr;
  std::int64_t dipole_elements = 0;
  const double* quadrupoles = nullptr;
  std::int64_t quadrupole_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Topology-major multipoles consumed by ES2/ES3/AES2/D4/periodic kernels. */
struct Gfn2SccPotentialDeviceTopologyMultipoles {
  double* shell_charges = nullptr;
  std::int64_t shell_elements = 0;
  double* atomic_charges = nullptr;
  std::int64_t atom_elements = 0;
  double* atomic_dipoles = nullptr;
  std::int64_t dipole_elements = 0;
  double* atomic_quadrupoles = nullptr;
  std::int64_t quadrupole_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Pure component potentials produced by the CUDA physics kernels. */
struct Gfn2SccPotentialDeviceComponents {
  std::uint32_t enabled_components = 0u;
  const double* es2_shell = nullptr;
  std::int64_t es2_shell_elements = 0;
  const double* es3_shell = nullptr;
  std::int64_t es3_shell_elements = 0;
  const double* explicit_point_charge_shell = nullptr;
  std::int64_t explicit_point_charge_shell_elements = 0;
  const double* aes2_atomic = nullptr;
  std::int64_t aes2_atomic_elements = 0;
  const double* aes2_dipole = nullptr;
  std::int64_t aes2_dipole_elements = 0;
  const double* aes2_quadrupole = nullptr;
  std::int64_t aes2_quadrupole_elements = 0;
  const double* d4_atomic = nullptr;
  std::int64_t d4_atomic_elements = 0;
  const double* periodic_atomic = nullptr;
  std::int64_t periodic_atomic_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Complete field-layout potentials consumed by the Hamiltonian assembler. */
struct Gfn2SccPotentialDeviceResults {
  double* shell = nullptr;
  std::int64_t shell_elements = 0;
  double* atomic = nullptr;
  std::int64_t atom_elements = 0;
  double* dipole = nullptr;
  std::int64_t dipole_elements = 0;
  double* quadrupole = nullptr;
  std::int64_t quadrupole_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Caller-owned unpublished storage shared by gather and compose stages. */
struct Gfn2SccPotentialDeviceWorkspace {
  double* shell_scratch = nullptr;
  std::int64_t shell_elements = 0;
  double* atom_scratch = nullptr;
  std::int64_t atom_elements = 0;
  double* dipole_scratch = nullptr;
  std::int64_t dipole_elements = 0;
  double* quadrupole_scratch = nullptr;
  std::int64_t quadrupole_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2SccPotentialDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2SccPotentialDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2SccPotentialDeviceActivity>);
static_assert(std::is_trivially_copyable_v<Gfn2SccPotentialDeviceMixedFields>);
static_assert(std::is_trivially_copyable_v<Gfn2SccPotentialDeviceTopologyMultipoles>);
static_assert(std::is_trivially_copyable_v<Gfn2SccPotentialDeviceComponents>);
static_assert(std::is_trivially_copyable_v<Gfn2SccPotentialDeviceResults>);
static_assert(std::is_trivially_copyable_v<Gfn2SccPotentialDeviceWorkspace>);

cudaError_t reset_gfn2_scc_potential_device_errors_cuda(std::int64_t batch_size,
                                                        std::uint32_t* system_errors,
                                                        std::uint32_t* device_error,
                                                        cudaStream_t stream = nullptr) noexcept;

/*
 * Convert mixed field-layout qsh/dipole/quadrupole values to topology-major
 * arrays and reconstruct atomic charges by shell reduction. Publication is
 * transactional per system and inactive numerical fields are not inspected.
 */
cudaError_t gather_gfn2_scc_mixed_multipoles_cuda(
    const Gfn2SccPotentialDeviceBatch& batch, const Gfn2SccPotentialDeviceMixedFields& mixed,
    const Gfn2SccPotentialDeviceActivity& activity,
    const Gfn2SccPotentialDeviceTopologyMultipoles& topology,
    const Gfn2SccPotentialDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

/*
 * Project only mixed shell charges to topology-major atomic charges under the
 * canonical SCC ledger.  shell_charges, atomic_dipoles, and
 * atomic_quadrupoles in topology must be exact zero-copy aliases of mixed;
 * only atomic_charges is materialized through caller-owned scratch.  The
 * sequence gate is checked before the active mask, and inactive ragged slices
 * (including their offsets and numerical values) are never inspected.
 */
cudaError_t reduce_gfn2_scc_mixed_atomic_charges_cuda(
    const Gfn2SccPotentialDeviceBatch& batch, const Gfn2SccPotentialDeviceMixedFields& mixed,
    const Gfn2SccIterationDeviceActivity& activity,
    const Gfn2SccPotentialDeviceTopologyMultipoles& topology,
    const Gfn2SccPotentialDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

/*
 * Compose component potentials in CPU order and map topology-major arrays back
 * to field layouts. Shell order is ES2, ES3, explicit-PC; atomic order is AES2,
 * periodic, D4. Disabled components require null+zero extents and publish zero.
 */
cudaError_t compose_gfn2_scc_potentials_cuda(
    const Gfn2SccPotentialDeviceBatch& batch, const Gfn2SccPotentialDeviceComponents& components,
    const Gfn2SccPotentialDeviceActivity& activity, const Gfn2SccPotentialDeviceResults& results,
    const Gfn2SccPotentialDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

/* Canonical-ledger overload used by the SCC iteration composer.  Unlike the
 * compatibility entry above, its plan preflight reads offsets only for active
 * systems and never derives a second requested-activity policy. */
cudaError_t compose_gfn2_scc_potentials_cuda(
    const Gfn2SccPotentialDeviceBatch& batch, const Gfn2SccPotentialDeviceComponents& components,
    const Gfn2SccIterationDeviceActivity& activity, const Gfn2SccPotentialDeviceResults& results,
    const Gfn2SccPotentialDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_SCC_POTENTIAL_CUH
