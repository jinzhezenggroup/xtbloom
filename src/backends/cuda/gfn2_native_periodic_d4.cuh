// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_NATIVE_PERIODIC_D4_CUH
#define XTBLOOM_BACKENDS_CUDA_GFN2_NATIVE_PERIODIC_D4_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_d4.cuh"
#include "backends/cuda/periodic_topology.cuh"

namespace xtbloom::detail::cuda {

/*
 * Device view of the native XYZ D4 term.  The topology must be generated
 * with an image cutoff of at least 50 bohr.  Each evaluator applies its own
 * 30/50/25-bohr predicate, so one immutable translation superset can serve
 * D4-CN, two-body, and ATM without conflating their physical cutoffs.
 *
 * `coordination_numbers` and `atomic_charges` are consumed only by the
 * energy/potential/force entry points; the coordination entry point needs
 * only atomic_numbers and positions.  `active_mask` is optional and is used
 * by the composed force path to preserve peer-local publication semantics.
 */
struct Gfn2NativePeriodicD4DeviceBatch {
  Gfn2CudaPeriodicTopologyView topology{};
  Gfn2D4DeviceParameters parameters{};
  const std::int32_t* atomic_numbers = nullptr;
  std::int64_t atomic_number_elements = 0;
  const double* positions = nullptr;
  std::int64_t position_elements = 0;
  const double* coordination_numbers = nullptr;
  std::int64_t coordination_number_elements = 0;
  const double* atomic_charges = nullptr;
  std::int64_t atomic_charge_elements = 0;
  const std::uint8_t* active_mask = nullptr;
  std::int64_t active_mask_elements = 0;
  /* Setup identity and geometry epoch are checked before native consumers run. */
  std::uint64_t plan_token = 0u;
  std::uint64_t geometry_generation = 0u;
  double image_cutoff = 0.0;
};

/*
 * Caller-owned scratch.  The native implementation is deliberately one
 * block per ragged system and deterministic within a block.  Scratch is
 * written before any public output, which lets a failed system leave every
 * requested output slice untouched while healthy peers commit normally.
 */
struct Gfn2NativePeriodicD4DeviceWorkspace {
  double* wrapped_positions = nullptr;
  std::int64_t wrapped_position_elements = 0;
  double* weights = nullptr;
  double* weight_cn_derivatives = nullptr;
  double* weight_charge_derivatives = nullptr;
  std::int64_t weight_elements = 0;
  double* coordination = nullptr;
  std::int64_t coordination_elements = 0;
  double* atom_energy = nullptr;
  std::int64_t atom_energy_elements = 0;
  double* atom_potential = nullptr;
  std::int64_t atom_potential_elements = 0;
  double* gradient = nullptr;
  std::int64_t gradient_elements = 0;
  double* strain = nullptr;
  std::int64_t strain_elements = 0;
  double* coordination_adjoint = nullptr;
  std::int64_t coordination_adjoint_elements = 0;
  std::uint64_t plan_token = 0u;
};

enum class Gfn2NativePeriodicD4DeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kInvalidAtomicNumber = 2u,
  kInvalidParameterData = 3u,
  kNonfinitePosition = 4u,
  kNonfiniteCoordination = 5u,
  kNonfiniteCharge = 6u,
  kCoincidentImage = 7u,
  kNonfiniteArithmetic = 8u,
  kInvalidActivity = 9u,
  kInvalidTopology = 10u,
};

static_assert(std::is_trivially_copyable_v<Gfn2NativePeriodicD4DeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2NativePeriodicD4DeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2NativePeriodicD4DeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2NativePeriodicD4DeviceWorkspace>);

cudaError_t reset_gfn2_native_periodic_d4_errors_cuda(std::int64_t batch_size,
                                                      std::uint32_t* system_errors,
                                                      std::uint32_t* device_error,
                                                      cudaStream_t stream = nullptr) noexcept;

/* Build the image-aware D4 coordination numbers using the 30-bohr role. */
cudaError_t evaluate_gfn2_native_periodic_d4_coordination_cuda(
    const Gfn2NativePeriodicD4DeviceBatch& batch,
    const Gfn2NativePeriodicD4DeviceWorkspace& workspace, double* coordination_numbers,
    std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

/* Evaluate charge-dependent two-body D4 energy and dE/dq per atom. */
cudaError_t evaluate_gfn2_native_periodic_d4_two_body_cuda(
    const Gfn2NativePeriodicD4DeviceBatch& batch,
    const Gfn2NativePeriodicD4DeviceWorkspace& workspace, double* per_atom_energies,
    double* atomic_potentials, std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

/* Evaluate q=0 ATM energy and publish one per-atom contribution per peer. */
cudaError_t evaluate_gfn2_native_periodic_d4_atm_cuda(
    const Gfn2NativePeriodicD4DeviceBatch& batch,
    const Gfn2NativePeriodicD4DeviceWorkspace& workspace, double* per_atom_energies,
    std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

/*
 * Add the two-body and ATM native D4 coordinate/cell derivatives.  `gradients`
 * and `strain_derivatives` are existing dE/dR and dE/d(epsilon) accumulators;
 * only a complete finite peer is incremented.  The D4-CN VJP is included.
 */
cudaError_t add_gfn2_native_periodic_d4_gradients_cuda(
    const Gfn2NativePeriodicD4DeviceBatch& batch,
    const Gfn2NativePeriodicD4DeviceWorkspace& workspace, double* gradients,
    double* strain_derivatives, std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_NATIVE_PERIODIC_D4_CUH
