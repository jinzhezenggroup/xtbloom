// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_NATIVE_PERIODIC_SHORT_RANGE_CUH
#define XTBLOOM_BACKENDS_CUDA_GFN2_NATIVE_PERIODIC_SHORT_RANGE_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/periodic_topology.cuh"

namespace xtbloom::detail::cuda {

/*
 * Errors local to one native-periodic short-range system are written to
 * system_errors.  The topology owner still reports malformed global device
 * views through device_error, but a bad atom or pair must not poison healthy
 * ragged peers.
 */
enum class Gfn2NativePeriodicShortRangeDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kInvalidAtomicNumber = 2u,
  kInvalidCovalentRadius = 3u,
  kNonfinitePosition = 4u,
  kCoincidentImage = 5u,
  kNonfiniteArithmetic = 6u,
  kInvalidTopology = 7u,
};

/*
 * Device inputs for the first CUDA-native periodic physics slice.  The
 * topology is the immutable complete 25-bohr translation superset from
 * periodic_topology.cuh.  Covalent radii are supplied by the model plan so
 * this primitive cannot silently drift from the CPU GFN2 coordination data.
 */
struct Gfn2NativePeriodicShortRangeDeviceBatch {
  Gfn2CudaPeriodicTopologyView topology{};
  const std::int32_t* atomic_numbers = nullptr;
  std::int64_t atomic_number_elements = 0;
  const double* positions = nullptr;
  std::int64_t position_elements = 0;
  const double* covalent_radii = nullptr;
  std::int64_t covalent_radius_elements = 0;
};

/*
 * Unpublished scratch makes the combined evaluator transactional per system:
 * a peer that encounters a non-finite input or coincident image never writes
 * partial coordination, energy, gradient, or strain output.
 */
struct Gfn2NativePeriodicShortRangeDeviceWorkspace {
  double* wrapped_positions = nullptr;
  std::int64_t wrapped_position_elements = 0;
  double* coordination = nullptr;
  std::int64_t coordination_elements = 0;
  double* repulsion_energies = nullptr;
  std::int64_t repulsion_energy_elements = 0;
  double* repulsion_gradients = nullptr;
  std::int64_t repulsion_gradient_elements = 0;
  double* repulsion_strain = nullptr;
  std::int64_t repulsion_strain_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2NativePeriodicShortRangeDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2NativePeriodicShortRangeDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2NativePeriodicShortRangeDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2NativePeriodicShortRangeDeviceWorkspace>);

/* Reset peer diagnostics and the sequence-wide device error asynchronously. */
cudaError_t reset_gfn2_native_periodic_short_range_errors_cuda(
    std::int64_t batch_size, std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

/*
 * Evaluate native GFN2 25-bohr coordination and screened nuclear repulsion.
 *
 * The outputs are overwritten only for peers whose complete calculation is
 * valid.  `repulsion_energies` contains one total energy per system (the CPU
 * term stores half-pair contributions per atom), repulsion gradients are
 * dE/dR (the public force path negates them), and strain is the row-major
 * affine dE/d(epsilon) convention used by the CPU periodic evaluator.  The
 * launcher performs structural checks only and enqueues one allocation-free
 * kernel; callers retain all pointers until the supplied stream has
 * completed.
 */
cudaError_t evaluate_gfn2_native_periodic_short_range_cuda(
    const Gfn2NativePeriodicShortRangeDeviceBatch& batch,
    const Gfn2NativePeriodicShortRangeDeviceWorkspace& workspace, double* coordination_numbers,
    double* repulsion_energies, double* repulsion_gradients, double* repulsion_strain,
    std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_NATIVE_PERIODIC_SHORT_RANGE_CUH
