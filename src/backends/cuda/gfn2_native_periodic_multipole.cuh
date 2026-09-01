// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_NATIVE_PERIODIC_MULTIPOLE_CUH
#define XTBLOOM_BACKENDS_CUDA_GFN2_NATIVE_PERIODIC_MULTIPOLE_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/periodic_topology.cuh"

namespace xtbloom::detail::cuda {

/*
 * A numerical failure is recorded against one ragged peer.  The launcher
 * still reports malformed descriptors synchronously; this distinction lets a
 * valid peer publish while another peer receives a complete sentinel slice.
 */
enum class Gfn2NativePeriodicMultipoleDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kNonfinitePosition = 2u,
  kNonfiniteCoordination = 3u,
  kNonfiniteCharge = 4u,
  kNonfiniteDipole = 5u,
  kNonfiniteQuadrupole = 6u,
  kInvalidAlpha = 7u,
  kInvalidVolume = 8u,
  kInvalidRadius = 9u,
  kInvalidKernel = 10u,
  kCoincidentImage = 11u,
  kNonfiniteArithmetic = 12u,
  kInvalidTopology = 13u,
};

/*
 * Immutable device mirror of PeriodicMultipolePlan.  The direct list is the
 * 100-bohr Wigner--Seitz/Ewald real-space list; the reciprocal list excludes
 * the zero vector.  Matrix offsets pack each ragged member as a local dense
 * atom-by-atom square, exactly like the CPU evaluator.
 */
struct Gfn2NativePeriodicMultipoleDeviceBatch {
  Gfn2CudaPeriodicTopologyView topology{};
  const std::int64_t* matrix_offsets = nullptr;
  std::int64_t matrix_offset_elements = 0;
  std::int64_t matrix_elements = 0;
  const double* volumes = nullptr;
  std::int64_t volume_elements = 0;
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
  const double* dipole_kernel = nullptr;
  std::int64_t dipole_kernel_elements = 0;
  const double* quadrupole_kernel = nullptr;
  std::int64_t quadrupole_kernel_elements = 0;
  const double* multipole_radius = nullptr;
  std::int64_t multipole_radius_elements = 0;
  const double* multipole_valence_cn = nullptr;
  std::int64_t multipole_valence_cn_elements = 0;
  const double* positions = nullptr;
  std::int64_t position_elements = 0;
  const double* coordination_numbers = nullptr;
  std::int64_t coordination_number_elements = 0;
  const double* atomic_charges = nullptr;
  std::int64_t atomic_charge_elements = 0;
  const double* atomic_dipoles = nullptr;
  std::int64_t atomic_dipole_elements = 0;
  const double* atomic_quadrupoles = nullptr;
  std::int64_t atomic_quadrupole_elements = 0;
  /* Optional peer-selection mask for a composed force transaction. */
  const std::uint8_t* active_mask = nullptr;
  std::int64_t active_mask_elements = 0;
};

/*
 * All numerical outputs are staged here before a peer commits to caller
 * storage.  Counts are expressed in doubles.  The matrix slices use the same
 * packed layouts as evaluate_periodic_multipole_cpu:
 * 3 values for q-d, 9 for d-d, and 6 for q-Q per ordered atom pair.
 */
struct Gfn2NativePeriodicMultipoleDeviceWorkspace {
  double* wrapped_positions = nullptr;
  std::int64_t wrapped_position_elements = 0;
  double* charge_dipole_matrix = nullptr;
  std::int64_t charge_dipole_matrix_elements = 0;
  double* dipole_dipole_matrix = nullptr;
  std::int64_t dipole_dipole_matrix_elements = 0;
  double* charge_quadrupole_matrix = nullptr;
  std::int64_t charge_quadrupole_matrix_elements = 0;
  double* charge_potentials = nullptr;
  std::int64_t charge_potential_elements = 0;
  double* dipole_potentials = nullptr;
  std::int64_t dipole_potential_elements = 0;
  double* quadrupole_potentials = nullptr;
  std::int64_t quadrupole_potential_elements = 0;
  double* energies = nullptr;
  std::int64_t energy_elements = 0;
  double* gradients = nullptr;
  std::int64_t gradient_elements = 0;
  double* strain = nullptr;
  std::int64_t strain_elements = 0;
  double* coordination_adjoint = nullptr;
  std::int64_t coordination_adjoint_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2NativePeriodicMultipoleDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2NativePeriodicMultipoleDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2NativePeriodicMultipoleDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2NativePeriodicMultipoleDeviceWorkspace>);

cudaError_t reset_gfn2_native_periodic_multipole_errors_cuda(
    std::int64_t batch_size, std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

/*
 * Evaluate the periodic q/d/Q AES2 primitive.  Matrix, potential, energy,
 * fixed-multipole Cartesian derivative, affine strain, and CN-adjoint outputs
 * follow evaluate_periodic_multipole_cpu.  Each block owns one ragged system;
 * the current correctness-first implementation intentionally uses one thread
 * per block so all sums have the CPU order and remain deterministic.
 */
cudaError_t evaluate_gfn2_native_periodic_multipole_cuda(
    const Gfn2NativePeriodicMultipoleDeviceBatch& batch,
    const Gfn2NativePeriodicMultipoleDeviceWorkspace& workspace, double* charge_dipole_matrix,
    double* dipole_dipole_matrix, double* charge_quadrupole_matrix, double* charge_potentials,
    double* dipole_potentials, double* quadrupole_potentials, double* energies, double* gradients,
    double* strain, double* coordination_adjoint, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_NATIVE_PERIODIC_MULTIPOLE_CUH
