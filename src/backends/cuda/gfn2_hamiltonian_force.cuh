#ifndef GPUXTB_BACKENDS_CUDA_GFN2_HAMILTONIAN_FORCE_CUH
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_BACKENDS_CUDA_GFN2_HAMILTONIAN_FORCE_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_force_common.cuh"
#include "backends/cuda/gfn2_hamiltonian.cuh"

namespace gpuxtb::detail::cuda {

enum class Gfn2HamiltonianForceDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidActiveMask = 1u,
  kInvalidOffsets = 2u,
  kInvalidOrbitalMetadata = 3u,
  kNonfiniteInput = 4u,
  kNonfiniteOutputSeed = 5u,
  kNonfiniteArithmetic = 6u,
};

/* Converged physical density and SCC potentials used by Hamiltonian assembly. */
struct Gfn2HamiltonianForceDeviceInput {
  const double* density = nullptr;
  std::int64_t density_elements = 0;
  const double* shell_scalar_potentials = nullptr;
  std::int64_t shell_scalar_elements = 0;
  const double* atomic_dipole_potentials = nullptr;
  std::int64_t atomic_dipole_elements = 0;
  const double* atomic_quadrupole_potentials = nullptr;
  std::int64_t atomic_quadrupole_elements = 0;
  std::uint64_t plan_token = 0u;

  /*
   * Optional unrestricted overlap response in physical M/S packing. These
   * fields intentionally trail plan_token so existing restricted aggregate
   * initializers remain source compatible. The pair is canonical: either
   * both pointers are null with zero extents, or both physical arrays exist.
   */
  const double* spin_density = nullptr;
  std::int64_t spin_density_elements = 0;
  const double* spin_shell_scalar_potentials = nullptr;
  std::int64_t spin_shell_scalar_elements = 0;
};

/*
 * Adjoint accumulators in the exact directed integral packing consumed by
 * add_multipole_gradient_cpu and its CUDA successor. overlap_adjoint is M,
 * dipole_adjoint is 3*M, and quadrupole_adjoint is 6*M.
 */
struct Gfn2HamiltonianForceDeviceOutput {
  double* overlap_adjoint = nullptr;
  std::int64_t overlap_adjoint_elements = 0;
  double* dipole_adjoint = nullptr;
  std::int64_t dipole_adjoint_elements = 0;
  double* quadrupole_adjoint = nullptr;
  std::int64_t quadrupole_adjoint_elements = 0;
  std::uint64_t plan_token = 0u;
};

struct Gfn2HamiltonianForceDeviceWorkspace {
  double* overlap_adjoint_scratch = nullptr;
  std::int64_t overlap_adjoint_elements = 0;
  double* dipole_adjoint_scratch = nullptr;
  std::int64_t dipole_adjoint_elements = 0;
  double* quadrupole_adjoint_scratch = nullptr;
  std::int64_t quadrupole_adjoint_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2HamiltonianForceDeviceInput>);
static_assert(std::is_standard_layout_v<Gfn2HamiltonianForceDeviceInput>);
static_assert(std::is_trivially_copyable_v<Gfn2HamiltonianForceDeviceOutput>);
static_assert(std::is_standard_layout_v<Gfn2HamiltonianForceDeviceOutput>);
static_assert(std::is_trivially_copyable_v<Gfn2HamiltonianForceDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2HamiltonianForceDeviceWorkspace>);

cudaError_t reset_gfn2_hamiltonian_force_device_errors_cuda(std::int64_t batch_size,
                                                            std::uint32_t* system_errors,
                                                            std::uint32_t* device_error,
                                                            cudaStream_t stream = nullptr) noexcept;

/*
 * Reverse the SCC Hamiltonian's explicit S/D/Q dependence at fixed converged
 * density and potentials. An optional unrestricted P_alpha-P_beta and v_mag
 * pair contributes only to the overlap adjoint. The output is an integral
 * adjoint, not a Cartesian force; a subsequent integral reverse pass converts
 * it to dE/dR.
 *
 * Directed ket-origin multipoles obey the forward assembly exactly:
 * D_mn couples to the potential on atom(n), while D_nm couples to atom(m).
 * Publication is atomic per system; the launcher is allocation-free,
 * synchronization-free, stream ordered, and CUDA Graph compatible.
 */
cudaError_t add_gfn2_hamiltonian_integral_adjoints_cuda(
    const Gfn2HamiltonianDeviceBatch& batch, const Gfn2ForceDeviceActivity& activity,
    const Gfn2HamiltonianForceDeviceInput& input, const Gfn2HamiltonianForceDeviceOutput& output,
    const Gfn2HamiltonianForceDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_HAMILTONIAN_FORCE_CUH
