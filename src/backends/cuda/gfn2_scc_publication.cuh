#ifndef GPUXTB_BACKENDS_CUDA_GFN2_SCC_PUBLICATION_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_SCC_PUBLICATION_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_density.cuh"
#include "backends/cuda/gfn2_eigensolver.cuh"
#include "backends/cuda/gfn2_mulliken.cuh"
#include "backends/cuda/gfn2_occupations.cuh"
#include "backends/cuda/gfn2_scc.cuh"
#include "backends/cuda/gfn2_scc_classical_energy.cuh"
#include "backends/cuda/gfn2_scc_free_energy.cuh"
#include "backends/cuda/gfn2_scc_iteration_control.cuh"
#include "backends/cuda/gfn2_scc_mixer.cuh"

namespace gpuxtb::detail::cuda {

/*
 * Stage-local publication diagnostics. Codes 1 and 2 are structural plan
 * failures. Codes 4 through 7 are peer-local numerical failures. Codes 3 and
 * 8 preserve the established trace ABI but are emitted through device_error as
 * CPU-compatible whole-call failures. The canonical #87 report retains the
 * frozen 0x1f8 mask; kPlanOnly keeps every device scalar plan-wide even when
 * its integer also lies in that historical peer mask.
 */
enum class Gfn2SccPublicationDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kInvalidState = 2u,
  kNonfiniteMixedAtomicCharge = 3u,
  kNonfiniteNextMixedMultipole = 4u,
  kNonfiniteRawMultipole = 5u,
  kNonfiniteResidual = 6u,
  kNonfiniteFreeEnergy = 7u,
  kNonfiniteEnergyDelta = 8u,
};

inline constexpr std::uint64_t kGfn2SccPublicationPeerErrorMask = 0x1f8u;

/*
 * Immutable mixed-spin publication topology. Physical offsets remain the
 * authority for occupations and shell-to-atom metadata. wavefunction_layout
 * supplies the nspin-expanded eigenpair, density, population, and mixer
 * partitions. The mixer vector for one system is the complete charge and
 * magnetization qsh followed by their 3 dipole and 6 quadrupole values.
 */
struct Gfn2SccPublicationDevicePlan {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t total_matrix_elements = 0;
  std::int64_t total_mixer_vector_elements = 0;
  std::int64_t history_size = 0;

  std::int64_t atom_offset_count = 0;
  std::int64_t shell_offset_count = 0;
  std::int64_t orbital_offset_count = 0;
  std::int64_t matrix_offset_count = 0;
  std::int64_t shell_to_atom_count = 0;

  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* shell_offsets = nullptr;
  const std::int64_t* orbital_offsets = nullptr;
  const std::int64_t* matrix_offsets = nullptr;
  const std::int64_t* shell_to_atom = nullptr;

  Gfn2WavefunctionLayoutView wavefunction_layout{};

  std::uint64_t maximum_iterations = 0u;
  double residual_rms_tolerance = 0.0;
  double energy_tolerance = 0.0;
  std::uint64_t plan_token = 0u;
};

/* Complete restricted wavefunction and thermodynamic output transaction. */
struct Gfn2SccPublicationDeviceWavefunction {
  Gfn2EigensolverDeviceResults eigenpairs{};
  Gfn2OccupationsDeviceResults occupations{};
  Gfn2DensityDeviceResults density{};
  Gfn2MullikenDevicePopulation population{};
  std::uint64_t plan_token = 0u;
};

/*
 * Component and complete free-energy traces committed with the wavefunction.
 * In staged state, free_energy.entropy exactly aliases the staged occupation
 * entropy produced by #76; public destinations remain separate transactions.
 */
struct Gfn2SccPublicationDeviceEnergyTrace {
  Gfn2SccClassicalEnergyDeviceDiagnostics classical{};
  Gfn2SccFreeEnergyDeviceDiagnostics free_energy{};
  /* Raw density-derived spin polarization, committed with the energy trace. */
  double* spin_energies = nullptr;
  std::int64_t spin_energy_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * All post-Mulliken results remain unpublished until commit. next_mixed has no
 * qat field; preflight reconstructs it deterministically into workspace by
 * summing qsh in shell order. staged_mixer is the complete tentative Broyden
 * state after the active transition.
 */
struct Gfn2SccPublicationDeviceStagedState {
  Gfn2SccPublicationDeviceWavefunction wavefunction{};
  Gfn2SccPublicationDeviceEnergyTrace energy{};
  Gfn2SccMixerDeviceState mixer{};
  Gfn2SccDeviceConstMultipoles next_mixed{};
  std::uint64_t plan_token = 0u;
};

/*
 * Persistent public transaction destination. wavefunction.population qsh,
 * dipole and quadrupole may exactly alias published field-by-field; qat remains
 * in population. On convergence the public population is raw, otherwise it is
 * next_mixed plus the derived atomic charges. scc.current_inputs and the mixer
 * private current input always receive next_mixed.
 */
struct Gfn2SccPublicationDevicePublicState {
  Gfn2SccPublicationDeviceWavefunction wavefunction{};
  Gfn2SccPublicationDeviceEnergyTrace energy{};
  Gfn2SccMixerDeviceState mixer{};
  Gfn2SccDeviceMultipoles published{};
  Gfn2SccDeviceState scc{};
  std::uint64_t plan_token = 0u;
};

/*
 * Caller-owned unpublished storage. The three diagnostic fields are the exact
 * pointers used by the kStatePublication #87 report. remaining arrays have one
 * element per atom or system and hold values already checked by preflight, so
 * commit performs copies only and cannot discover a semantic failure.
 */
struct Gfn2SccPublicationDeviceWorkspace {
  double* mixed_atomic_charges = nullptr;
  std::int64_t mixed_atomic_charge_elements = 0;
  double* previous_free_energies = nullptr;
  double* free_energy_changes = nullptr;
  std::uint64_t* next_iterations = nullptr;
  gpuxtb_status_t* next_statuses = nullptr;
  std::uint8_t* next_converged = nullptr;
  std::int64_t batch_elements = 0;

  std::uint32_t* system_errors = nullptr;
  std::int64_t system_error_elements = 0;
  std::uint32_t* device_error = nullptr;
  std::int64_t device_error_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2SccPublicationDevicePlan>);
static_assert(std::is_standard_layout_v<Gfn2SccPublicationDevicePlan>);
static_assert(std::is_trivially_copyable_v<Gfn2SccPublicationDeviceWavefunction>);
static_assert(std::is_standard_layout_v<Gfn2SccPublicationDeviceWavefunction>);
static_assert(std::is_trivially_copyable_v<Gfn2SccPublicationDeviceEnergyTrace>);
static_assert(std::is_standard_layout_v<Gfn2SccPublicationDeviceEnergyTrace>);
static_assert(std::is_trivially_copyable_v<Gfn2SccPublicationDeviceStagedState>);
static_assert(std::is_standard_layout_v<Gfn2SccPublicationDeviceStagedState>);
static_assert(std::is_trivially_copyable_v<Gfn2SccPublicationDevicePublicState>);
static_assert(std::is_standard_layout_v<Gfn2SccPublicationDevicePublicState>);
static_assert(std::is_trivially_copyable_v<Gfn2SccPublicationDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2SccPublicationDeviceWorkspace>);

/* Clear one publication diagnostic generation asynchronously. */
cudaError_t reset_gfn2_scc_publication_errors_cuda(
    const Gfn2SccPublicationDevicePlan& plan, const Gfn2SccPublicationDeviceWorkspace& workspace,
    cudaStream_t stream = nullptr) noexcept;

/*
 * Validate only canonically active members and prepare all convergence/state
 * values without touching public storage. Plan errors close sequence_active;
 * peer errors leave it open and are written only to system_errors. The caller
 * must normalize the resulting kStatePublication report through #87 before
 * enqueueing commit.
 */
cudaError_t preflight_gfn2_scc_publication_cuda(
    const Gfn2SccPublicationDevicePlan& plan, const Gfn2SccIterationDeviceActivity& activity,
    const Gfn2SccIterationDeviceLedger& ledger, const Gfn2SccPublicationDeviceStagedState& staged,
    const Gfn2SccPublicationDevicePublicState& public_state,
    const Gfn2SccPublicationDeviceWorkspace& workspace, cudaStream_t stream = nullptr) noexcept;

/*
 * Transactionally commit each member still active after #87 normalization.
 * A peer-failed member publishes its canonical pending SCC status, the CPU
 * attempt count, and a complete NaN energy/SCC trace while preserving its
 * wavefunction, mixer history, multipoles, and convergence flag. Canonically
 * inactive members are untouched, while a plan failure suppresses every
 * public write.
 */
cudaError_t commit_gfn2_scc_publication_cuda(
    const Gfn2SccPublicationDevicePlan& plan, const Gfn2SccIterationDeviceActivity& activity,
    const Gfn2SccIterationDeviceLedger& ledger, const Gfn2SccPublicationDeviceStagedState& staged,
    const Gfn2SccPublicationDevicePublicState& public_state,
    const Gfn2SccPublicationDeviceWorkspace& workspace, cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_SCC_PUBLICATION_CUH
