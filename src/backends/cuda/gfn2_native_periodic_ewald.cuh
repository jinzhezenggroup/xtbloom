// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_NATIVE_PERIODIC_EWALD_CUH
#define XTBLOOM_BACKENDS_CUDA_GFN2_NATIVE_PERIODIC_EWALD_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/periodic_topology.cuh"

namespace xtbloom::detail::cuda {

/*
 * The Ewald primitive reports malformed numerical data per ragged peer.  A
 * peer-local failure leaves every caller output slice for that peer untouched;
 * only the diagnostic arrays are written.  Structural descriptor failures
 * remain synchronous launcher errors, matching the other native-periodic
 * primitives.
 */
enum class Gfn2NativePeriodicEwaldDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kNonfinitePosition = 2u,
  kNonfiniteCharge = 3u,
  kInvalidAlpha = 4u,
  kInvalidHardness = 5u,
  kCoincidentImage = 6u,
  kNonfiniteArithmetic = 7u,
  kInvalidTopology = 8u,
};

/*
 * All metadata is an immutable device mirror of PeriodicEwaldPlan.  The
 * direct list is also used for the tiny Wigner--Seitz image search: the
 * Ewald direct cutoff is a conservative superset of the translations needed
 * to identify the nearest image, so no second lattice enumeration is needed
 * in the kernel.
 */
struct Gfn2NativePeriodicEwaldDeviceBatch {
  Gfn2CudaPeriodicTopologyView topology{};
  const std::int64_t* batch_shell_offsets = nullptr;
  std::int64_t batch_shell_offset_elements = 0;
  const std::int64_t* atom_shell_offsets = nullptr;
  std::int64_t atom_shell_offset_elements = 0;
  const std::int64_t* matrix_offsets = nullptr;
  std::int64_t matrix_offset_elements = 0;
  std::int64_t matrix_elements = 0;
  const double* shell_hardness = nullptr;
  std::int64_t shell_hardness_elements = 0;
  const double* alphas = nullptr;
  std::int64_t alpha_elements = 0;
  const std::int64_t* direct_translation_offsets = nullptr;
  std::int64_t direct_translation_offset_elements = 0;
  const Gfn2CudaPeriodicTranslation* direct_translations = nullptr;
  std::int64_t direct_translation_elements = 0;
  const std::int64_t* reciprocal_translation_offsets = nullptr;
  std::int64_t reciprocal_translation_offset_elements = 0;
  const Gfn2CudaPeriodicTranslation* reciprocal_translations = nullptr;
  std::int64_t reciprocal_translation_elements = 0;
  const double* positions = nullptr;
  std::int64_t position_elements = 0;
  const double* shell_charges = nullptr;
  std::int64_t shell_charge_elements = 0;
  /* Optional peer-selection mask used by composed post-SCC force passes.
   * Standalone primitive callers leave it null, preserving the historical
   * all-peer evaluation contract. */
  const std::uint8_t* active_mask = nullptr;
  std::int64_t active_mask_elements = 0;
};

/*
 * Scratch is unpublished until the complete system calculation is finite.
 * This preserves peer-local transactional publication without requiring a
 * second kernel or host polling between the Ewald terms.
 */
struct Gfn2NativePeriodicEwaldDeviceWorkspace {
  double* wrapped_positions = nullptr;
  std::int64_t wrapped_position_elements = 0;
  double* matrix = nullptr;
  std::int64_t matrix_elements = 0;
  double* shell_potentials = nullptr;
  std::int64_t shell_potential_elements = 0;
  double* energies = nullptr;
  std::int64_t energy_elements = 0;
  double* gradients = nullptr;
  std::int64_t gradient_elements = 0;
  double* strain = nullptr;
  std::int64_t strain_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2NativePeriodicEwaldDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2NativePeriodicEwaldDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2NativePeriodicEwaldDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2NativePeriodicEwaldDeviceWorkspace>);

cudaError_t reset_gfn2_native_periodic_ewald_errors_cuda(std::int64_t batch_size,
                                                         std::uint32_t* system_errors,
                                                         std::uint32_t* device_error,
                                                         cudaStream_t stream = nullptr) noexcept;

/*
 * Evaluate shell-resolved periodic GFN2 Ewald electrostatics.  Matrix entries,
 * shell potentials, 1/2 q^T A q energies, Cartesian dE/dR, and affine 3x3
 * strain derivatives follow evaluate_periodic_ewald_cpu exactly, including
 * the Klopman--Ohno correction, self term, and charged-cell uniform
 * background.  Gradients are coordinate derivatives (the public force path
 * applies the conventional minus sign).
 *
 * The launcher performs only structural/range checks and enqueues one
 * allocation-free kernel.  Device inputs and all workspace/output pointers
 * must remain live until the supplied stream has completed.
 */
cudaError_t evaluate_gfn2_native_periodic_ewald_cuda(
    const Gfn2NativePeriodicEwaldDeviceBatch& batch,
    const Gfn2NativePeriodicEwaldDeviceWorkspace& workspace, double* coulomb_matrix,
    double* shell_potentials, double* energies, double* gradients, double* strain,
    std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_NATIVE_PERIODIC_EWALD_CUH
