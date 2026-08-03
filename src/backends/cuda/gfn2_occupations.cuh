#ifndef GPUXTB_BACKENDS_CUDA_GFN2_OCCUPATIONS_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_OCCUPATIONS_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

namespace gpuxtb::detail::cuda {

/* Semantic or arithmetic failure reported per system and by canonical lowest index. */
enum class Gfn2OccupationsDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidActiveMask = 1u,
  kInvalidOffsets = 2u,
  kInvalidElectronCount = 3u,
  kInvalidTemperature = 4u,
  kNonfiniteEigenvalue = 5u,
  kUnsortedEigenvalues = 6u,
  kChemicalPotentialBracketFailure = 7u,
  kElectronConservationFailure = 8u,
  kNonfiniteEntropy = 9u,
};

/*
 * Restricted ragged spectra and their system-local electronic conditions.
 *
 * orbital_offsets is a zero-based half-open partition of total_orbitals.
 * electron_counts is interleaved [alpha,beta] for every system and each spin
 * count lies in [0, number of system orbitals]. temperatures contains k_B*T in
 * Hartree independently for every system. active values must be zero or one;
 * inactive systems are not inspected and all of their outputs remain untouched.
 */
struct Gfn2OccupationsDeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t orbital_offset_count = 0;
  std::int64_t electron_count_elements = 0;
  std::int64_t temperature_elements = 0;
  std::int64_t active_elements = 0;
  std::uint64_t plan_token = 0u;

  const std::int64_t* orbital_offsets = nullptr;
  const double* electron_counts = nullptr;
  const double* temperatures = nullptr;
  const std::uint8_t* active = nullptr;
};

/*
 * Transactional public outputs. For a system with orbital interval [b,e), its
 * 2*(e-b) occupations start at 2*b and are laid out as [alpha slice,beta slice],
 * exactly matching the CPU WavefunctionLayout. Chemical potentials and actual
 * published electron sums are interleaved [alpha,beta]. Entropy is the sum of
 * the two dimensionless per-spin entropies.
 */
struct Gfn2OccupationsDeviceResults {
  double* occupations = nullptr;
  std::int64_t occupation_elements = 0;
  double* chemical_potentials = nullptr;
  std::int64_t chemical_potential_elements = 0;
  double* electron_sums = nullptr;
  std::int64_t electron_sum_elements = 0;
  double* entropies = nullptr;
  std::int64_t entropy_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Caller-owned unpublished storage. Counts are numbers of elements, not bytes.
 * sequence_active snapshots whether the sticky device error was clear at entry,
 * so a failure in one member cannot suppress healthy peer publication.
 */
struct Gfn2OccupationsDeviceWorkspace {
  double* occupation_scratch = nullptr;
  std::int64_t occupation_elements = 0;
  double* chemical_potential_scratch = nullptr;
  std::int64_t chemical_potential_elements = 0;
  double* electron_sum_scratch = nullptr;
  std::int64_t electron_sum_elements = 0;
  double* entropy_scratch = nullptr;
  std::int64_t entropy_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_active_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2OccupationsDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2OccupationsDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2OccupationsDeviceResults>);
static_assert(std::is_standard_layout_v<Gfn2OccupationsDeviceResults>);
static_assert(std::is_trivially_copyable_v<Gfn2OccupationsDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2OccupationsDeviceWorkspace>);

/* Clear per-system failures and the sticky canonical diagnostic. */
cudaError_t reset_gfn2_occupations_device_errors_cuda(std::int64_t batch_size,
                                                      std::uint32_t* system_errors,
                                                      std::uint32_t* device_error,
                                                      cudaStream_t stream = nullptr) noexcept;

/*
 * Fill restricted alpha/beta occupations, chemical potentials, actual electron
 * sums, and total entropy from sorted finite eigenvalues.
 *
 * Zero-temperature Aufbau filling and finite-temperature Fermi filling follow
 * fill_occupations_cpu, including translated-energy bracketing, holes near full
 * capacity, equal correction across exactly degenerate blocks, and entropy of
 * the doubles actually published. A failed active system commits no public
 * output; healthy peers publish independently.
 *
 * The launcher allocates nothing, transfers nothing, never synchronizes, uses
 * only the supplied stream, and is CUDA Graph capture/replay compatible. Its
 * deterministic serial root solve per system is intentional: batch systems
 * execute in parallel while every member has reproducible reduction order.
 * Device/UVA pointer provenance is a setup-time binding contract.
 */
cudaError_t evaluate_gfn2_restricted_occupations_cuda(
    const Gfn2OccupationsDeviceBatch& batch, const double* eigenvalues,
    std::int64_t eigenvalue_elements, const Gfn2OccupationsDeviceResults& results,
    const Gfn2OccupationsDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_OCCUPATIONS_CUH
