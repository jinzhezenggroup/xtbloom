#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_ELECTRIC_FIELD_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN2_ELECTRIC_FIELD_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

namespace xtbloom::detail::cuda {

/* Fixed ragged topology shared by uniform-field potential, energy, and force
 * consumers. The runtime normalizes an absent attachment to an exact zero
 * three-vector, so changing field presence never changes this binding. */
struct Gfn2ElectricFieldDeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t atom_offset_count = 0;
  const std::int64_t* atom_offsets = nullptr;
  std::uint64_t plan_token = 0u;
};

/* Per-system E vectors and atom-major positions, both in atomic units. */
struct Gfn2ElectricFieldDeviceInput {
  const double* vectors = nullptr;
  std::int64_t vector_elements = 0;
  const double* positions = nullptr;
  std::int64_t position_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Generated SCC potentials: vat_i=-E.R_i and vdp_i=-E. */
struct Gfn2ElectricFieldDevicePotentials {
  double* atomic = nullptr;
  std::int64_t atom_elements = 0;
  double* dipole = nullptr;
  std::int64_t dipole_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Const projection used after the refresh transaction commits. */
struct Gfn2ElectricFieldDevicePotentialView {
  const double* atomic = nullptr;
  std::int64_t atom_elements = 0;
  const double* dipole = nullptr;
  std::int64_t dipole_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Raw stationary multipoles used by the field energy. */
struct Gfn2ElectricFieldDeviceMultipoles {
  const double* atomic_charges = nullptr;
  std::int64_t atom_elements = 0;
  const double* atomic_dipoles = nullptr;
  std::int64_t dipole_elements = 0;
  std::uint64_t plan_token = 0u;
};

enum class Gfn2ElectricFieldDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kNonfiniteVector = 2u,
  kNonfinitePosition = 3u,
  kNonfinitePotentialArithmetic = 4u,
};

static_assert(std::is_trivially_copyable_v<Gfn2ElectricFieldDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2ElectricFieldDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2ElectricFieldDeviceInput>);
static_assert(std::is_standard_layout_v<Gfn2ElectricFieldDeviceInput>);
static_assert(std::is_trivially_copyable_v<Gfn2ElectricFieldDevicePotentials>);
static_assert(std::is_standard_layout_v<Gfn2ElectricFieldDevicePotentials>);
static_assert(std::is_trivially_copyable_v<Gfn2ElectricFieldDeviceMultipoles>);
static_assert(std::is_standard_layout_v<Gfn2ElectricFieldDeviceMultipoles>);
static_assert(std::is_trivially_copyable_v<Gfn2ElectricFieldDevicePotentialView>);
static_assert(std::is_standard_layout_v<Gfn2ElectricFieldDevicePotentialView>);

/* Clear peer-local errors and the topology-only plan error asynchronously. */
cudaError_t reset_gfn2_electric_field_device_errors_cuda(std::int64_t batch_size,
                                                         std::uint32_t* system_errors,
                                                         std::uint32_t* plan_error,
                                                         cudaStream_t stream = nullptr) noexcept;

/*
 * Generate field potentials transactionally per system. A malformed ragged
 * topology suppresses the complete publication; a nonfinite field/position or
 * arithmetic failure preserves only that peer's prior potential slices.
 *
 * The launcher allocates, transfers, polls, and synchronizes nothing. It is
 * custom-stream safe and CUDA Graph capturable. Input and output ranges must
 * be disjoint; CUDA address-space provenance is a setup responsibility.
 */
cudaError_t refresh_gfn2_electric_field_potentials_cuda(
    const Gfn2ElectricFieldDeviceBatch& batch, const Gfn2ElectricFieldDeviceInput& input,
    const Gfn2ElectricFieldDevicePotentials& potentials, std::uint32_t* system_errors,
    std::uint32_t* plan_error, cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_ELECTRIC_FIELD_CUH
