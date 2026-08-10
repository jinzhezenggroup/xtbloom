#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_SCC_POTENTIAL_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN2_SCC_POTENTIAL_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/common/gfn2_plan_schema.hpp"
#include "backends/cuda/gfn2_scc_iteration_control.cuh"

namespace xtbloom::detail::cuda {

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
  kInvalidSpinLayout = 16u,
  kNonfiniteSpinPotential = 17u,
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

/* Spin-polarization shell potential in WavefunctionLayout charge/magnetization
 * packing. Restricted slices are not inspected; for unrestricted slices only
 * the magnetization channel contributes to the composed potential. */
struct Gfn2SccPotentialDeviceSpinComponent {
  const double* shell = nullptr;
  std::int64_t shell_elements = 0;
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
static_assert(std::is_trivially_copyable_v<Gfn2SccPotentialDeviceSpinComponent>);
static_assert(std::is_standard_layout_v<Gfn2SccPotentialDeviceSpinComponent>);
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
 * Mixed-spin canonical-ledger projection. spin_topology preserves the exact
 * WavefunctionLayout charge/magnetization packing and receives one shell-sum
 * per spin atom. physical_topology receives the dense charge-channel qsh,
 * qat, dipole, and quadrupole arrays consumed by legacy physical-topology
 * component kernels. Both destinations publish together per active system;
 * inactive and failed peers preserve every destination byte.
 */
cudaError_t reduce_gfn2_scc_spin_atomic_charges_cuda(
    const Gfn2SccPotentialDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2SccPotentialDeviceMixedFields& mixed, const Gfn2SccIterationDeviceActivity& activity,
    const Gfn2SccPotentialDeviceTopologyMultipoles& spin_topology,
    const Gfn2SccPotentialDeviceTopologyMultipoles& physical_topology,
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

/*
 * Compose potentials for a mixed one-/two-channel batch. Charge components
 * retain the established CPU association order and are written to channel
 * zero. The channel-zero shell field is Hamiltonian-ready: each raw shell
 * value receives its owning atom's scalar value with the literal legacy
 * `vsh + vat` bridge operation. Unrestricted channel one receives the
 * spin-polarization shell potential; its atom, dipole, and quadrupole fields
 * are exact +0.0. Restricted systems therefore match the legacy composer
 * followed by the scalar bridge, without a second collection pass.
 * Results/workspace extents follow layout.total_spin_shells and
 * layout.total_spin_atoms (with 3x/6x multipole extents).
 */
cudaError_t compose_gfn2_scc_spin_potentials_cuda(
    const Gfn2SccPotentialDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2SccPotentialDeviceComponents& components,
    const Gfn2SccPotentialDeviceSpinComponent& spin, const Gfn2SccPotentialDeviceActivity& activity,
    const Gfn2SccPotentialDeviceResults& results, const Gfn2SccPotentialDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

/* Canonical-ledger overload used by the SCC iteration composer.  Unlike the
 * compatibility entry above, its plan preflight reads offsets only for active
 * systems and never derives a second requested-activity policy. */
cudaError_t compose_gfn2_scc_potentials_cuda(
    const Gfn2SccPotentialDeviceBatch& batch, const Gfn2SccPotentialDeviceComponents& components,
    const Gfn2SccIterationDeviceActivity& activity, const Gfn2SccPotentialDeviceResults& results,
    const Gfn2SccPotentialDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_SCC_POTENTIAL_CUH
