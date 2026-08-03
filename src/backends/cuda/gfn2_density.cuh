#ifndef GPUXTB_BACKENDS_CUDA_GFN2_DENSITY_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_DENSITY_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

namespace gpuxtb::detail::cuda {

/* Per-system failure code; the batch diagnostic is canonicalized by system index. */
enum class Gfn2DensityDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidActiveMask = 1u,
  kInvalidOffsets = 2u,
  kNonfiniteCoefficient = 3u,
  kNonfiniteEigenvalue = 4u,
  kInvalidOccupation = 5u,
  kNonfiniteWeightArithmetic = 6u,
  kNonfiniteBandEnergy = 7u,
  kNonfiniteDensityArithmetic = 8u,
  kNonfiniteWeightedDensityArithmetic = 9u,
  kNonfiniteTrace = 10u,
};

/*
 * Restricted ragged AO topology. orbital_offsets partitions eigenvalue and
 * coefficient dimensions; matrix_offsets partitions complete dense row-major
 * nao*nao matrices. Every active member must have at least one orbital and
 * matrix_offsets[s+1]-matrix_offsets[s] == nao*nao.
 */
struct Gfn2DensityDeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t total_matrix_elements = 0;
  std::int64_t orbital_offset_count = 0;
  std::int64_t matrix_offset_count = 0;
  std::uint64_t plan_token = 0u;
  const std::int64_t* orbital_offsets = nullptr;
  const std::int64_t* matrix_offsets = nullptr;
};

/*
 * Restricted eigenpairs and occupations. Coefficients are system-major dense
 * row-major [AO,orbital]. Eigenvalues own one row per system. Occupations use
 * #76 packing: system s starts at 2*orbital_offsets[s], followed by complete
 * alpha and beta rows. Inactive members are skipped before numerical inputs.
 */
struct Gfn2DensityDeviceInput {
  const double* coefficients = nullptr;
  std::int64_t coefficient_elements = 0;
  const double* eigenvalues = nullptr;
  std::int64_t eigenvalue_elements = 0;
  const double* occupations = nullptr;
  std::int64_t occupation_elements = 0;
  const std::uint8_t* active = nullptr;
  std::int64_t active_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * P and W are complete dense row-major matrices in matrix_offsets packing.
 * Scalar diagnostics are one value per system: band energy sum(f*epsilon),
 * total alpha+beta occupation, Tr(P), and Tr(W).
 */
struct Gfn2DensityDeviceResults {
  double* density = nullptr;
  std::int64_t density_elements = 0;
  double* energy_weighted_density = nullptr;
  std::int64_t weighted_density_elements = 0;
  double* band_energies = nullptr;
  std::int64_t band_energy_elements = 0;
  double* occupation_sums = nullptr;
  std::int64_t occupation_sum_elements = 0;
  double* density_traces = nullptr;
  std::int64_t density_trace_elements = 0;
  double* weighted_density_traces = nullptr;
  std::int64_t weighted_density_trace_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Caller-owned unpublished matrices, orbital weights, scalars, and sequence state. */
struct Gfn2DensityDeviceWorkspace {
  double* density_scratch = nullptr;
  std::int64_t density_elements = 0;
  double* weighted_density_scratch = nullptr;
  std::int64_t weighted_density_elements = 0;
  double* weights = nullptr;
  std::int64_t weight_elements = 0;
  double* energy_weights = nullptr;
  std::int64_t energy_weight_elements = 0;
  double* band_energy_scratch = nullptr;
  std::int64_t band_energy_elements = 0;
  double* occupation_sum_scratch = nullptr;
  std::int64_t occupation_sum_elements = 0;
  double* density_trace_scratch = nullptr;
  std::int64_t density_trace_elements = 0;
  double* weighted_density_trace_scratch = nullptr;
  std::int64_t weighted_density_trace_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_active_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2DensityDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2DensityDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2DensityDeviceInput>);
static_assert(std::is_standard_layout_v<Gfn2DensityDeviceInput>);
static_assert(std::is_trivially_copyable_v<Gfn2DensityDeviceResults>);
static_assert(std::is_standard_layout_v<Gfn2DensityDeviceResults>);
static_assert(std::is_trivially_copyable_v<Gfn2DensityDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2DensityDeviceWorkspace>);

/* Clear per-system errors and the sticky canonical diagnostic. */
cudaError_t reset_gfn2_density_device_errors_cuda(std::int64_t batch_size,
                                                  std::uint32_t* system_errors,
                                                  std::uint32_t* device_error,
                                                  cudaStream_t stream = nullptr) noexcept;

/*
 * Assemble P=C(f_alpha+f_beta)C^T and
 * W=C((f_alpha+f_beta)*epsilon)C^T plus scalar diagnostics.
 *
 * Publication is atomic per system. Every product, accumulation, reduction,
 * and scalar is checked for finiteness before a separate publication kernel.
 * Matrix symmetry is exact because each lower-triangle contraction is written
 * to both dense directions. The launcher allocates nothing, synchronizes
 * nothing, uses only stream, and supports CUDA Graph capture/replay.
 */
cudaError_t evaluate_gfn2_restricted_density_cuda(const Gfn2DensityDeviceBatch& batch,
                                                  const Gfn2DensityDeviceInput& input,
                                                  const Gfn2DensityDeviceResults& results,
                                                  const Gfn2DensityDeviceWorkspace& workspace,
                                                  std::uint32_t* system_errors,
                                                  std::uint32_t* device_error,
                                                  cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_DENSITY_CUH
