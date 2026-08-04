#ifndef GPUXTB_BACKENDS_CUDA_GFN2_HAMILTONIAN_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_HAMILTONIAN_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/common/gfn2_plan_schema.hpp"

namespace gpuxtb::detail::cuda {

inline constexpr std::int64_t kGfn2HamiltonianDipoleComponents = 3;
inline constexpr std::int64_t kGfn2HamiltonianQuadrupoleComponents = 6;

/* First asynchronous semantic or arithmetic failure in an assembly sequence. */
enum class Gfn2HamiltonianDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kInvalidOrbitalMetadata = 2u,
  kInvalidActiveMask = 3u,
  kNonfiniteH0 = 4u,
  kNonfiniteOverlap = 5u,
  kNonfiniteMultipoleIntegral = 6u,
  kNonfinitePotential = 7u,
  kNonfiniteAssemblyArithmetic = 8u,
  kInvalidSpinLayout = 9u,
  kNonfiniteSpinConversion = 10u,
};

/*
 * Restricted ragged AO topology. System matrices are complete row-major
 * squares. orbital_to_shell/orbital_to_atom make ownership explicit in the
 * hot matrix kernel; the remaining partitions validate those cached maps.
 */
struct Gfn2HamiltonianDeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t total_matrix_elements = 0;
  std::uint64_t plan_token = 0u;

  std::int64_t atom_offset_count = 0;
  std::int64_t batch_shell_offset_count = 0;
  std::int64_t batch_orbital_offset_count = 0;
  std::int64_t matrix_offset_count = 0;
  std::int64_t atom_shell_offset_count = 0;
  std::int64_t shell_orbital_offset_count = 0;
  std::int64_t shell_to_atom_count = 0;
  std::int64_t orbital_to_shell_count = 0;
  std::int64_t orbital_to_atom_count = 0;

  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* batch_shell_offsets = nullptr;
  const std::int64_t* batch_orbital_offsets = nullptr;
  const std::int64_t* matrix_offsets = nullptr;
  const std::int64_t* atom_shell_offsets = nullptr;
  const std::int64_t* shell_orbital_offsets = nullptr;
  const std::int64_t* shell_to_atom = nullptr;
  const std::int64_t* orbital_to_shell = nullptr;
  const std::int64_t* orbital_to_atom = nullptr;
};

/*
 * H0/overlap are M-element matrices. Directed ket-origin multipole integrals
 * are component-major 3*M and 6*M arrays in [x,y,z] and
 * [xx,xy,yy,xz,yz,zz] order.
 */
struct Gfn2HamiltonianDeviceInput {
  const double* h0 = nullptr;
  std::int64_t h0_elements = 0;
  const double* overlap = nullptr;
  std::int64_t overlap_elements = 0;
  const double* dipole_integrals = nullptr;
  std::int64_t dipole_integral_elements = 0;
  const double* quadrupole_integrals = nullptr;
  std::int64_t quadrupole_integral_elements = 0;
  const double* shell_scalar_potentials = nullptr;
  std::int64_t shell_scalar_elements = 0;
  const double* atomic_dipole_potentials = nullptr;
  std::int64_t atomic_dipole_elements = 0;
  const double* atomic_quadrupole_potentials = nullptr;
  std::int64_t atomic_quadrupole_elements = 0;
  std::uint64_t plan_token = 0u;
};

struct Gfn2HamiltonianDeviceOutput {
  double* matrix = nullptr;
  std::int64_t elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Zero skips a terminal member before its topology or numerical data is read. */
struct Gfn2HamiltonianDeviceActivity {
  const std::uint8_t* active_mask = nullptr;
  std::int64_t elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Caller-owned unpublished matrix plus one sticky-sequence snapshot scalar. */
struct Gfn2HamiltonianDeviceWorkspace {
  double* matrix_scratch = nullptr;
  std::int64_t matrix_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2HamiltonianDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2HamiltonianDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2HamiltonianDeviceInput>);
static_assert(std::is_standard_layout_v<Gfn2HamiltonianDeviceInput>);
static_assert(std::is_trivially_copyable_v<Gfn2HamiltonianDeviceOutput>);
static_assert(std::is_standard_layout_v<Gfn2HamiltonianDeviceOutput>);
static_assert(std::is_trivially_copyable_v<Gfn2HamiltonianDeviceActivity>);
static_assert(std::is_standard_layout_v<Gfn2HamiltonianDeviceActivity>);
static_assert(std::is_trivially_copyable_v<Gfn2HamiltonianDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2HamiltonianDeviceWorkspace>);

cudaError_t reset_gfn2_hamiltonian_device_errors_cuda(std::int64_t batch_size,
                                                      std::uint32_t* system_errors,
                                                      std::uint32_t* device_error,
                                                      cudaStream_t stream = nullptr) noexcept;

/*
 * Assemble H = H0 + dH for every active restricted member:
 *
 * dH_mn = -S_mn (u_m + u_n)/2
 *          -D_mn(A_n):v_D(A_n)/2 - D_nm(A_m):v_D(A_m)/2
 *          -Q_mn(A_n):v_Q(A_n)/2 - Q_nm(A_m):v_Q(A_m)/2.
 *
 * shell_scalar_potentials contains the already-collected complete scalar AO
 * potential by shell. Q uses the packed dual contraction with no extra factor
 * for off-diagonal components. One upper-triangle evaluation publishes the
 * identical shift into both stored matrix directions, matching the CPU path.
 *
 * Publication is atomic per system. The launcher allocates and synchronizes
 * nothing, enqueues only on stream, supports CUDA Graph capture, and accepts
 * device/UVA pointers whose provenance was validated by the binding layer.
 */
cudaError_t assemble_gfn2_hamiltonian_cuda(
    const Gfn2HamiltonianDeviceBatch& batch, const Gfn2HamiltonianDeviceInput& input,
    const Gfn2HamiltonianDeviceActivity& activity, const Gfn2HamiltonianDeviceOutput& output,
    const Gfn2HamiltonianDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream = nullptr) noexcept;

/*
 * Assemble a mixed restricted/unrestricted batch using the canonical
 * WavefunctionLayout projection. Potential arrays use system-major
 * charge/magnetization packing in spin_shell_offsets/spin_atom_offsets.
 * Restricted members retain the legacy arithmetic exactly. For two-channel
 * members, charge/magnetization potentials are converted half-before-add:
 *
 *   V_alpha = 0.5 V_charge + 0.5 V_magnetization
 *   V_beta  = 0.5 V_charge - 0.5 V_magnetization.
 *
 * Each channel first forms Htmp with the legacy assembler, then publishes the
 * literal CPU-compatible expression H = H0 + 2*(Htmp-H0). A failure in either
 * channel suppresses both channel matrices for that physical system.
 */
cudaError_t assemble_gfn2_spin_hamiltonian_cuda(
    const Gfn2HamiltonianDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2HamiltonianDeviceInput& input, const Gfn2HamiltonianDeviceActivity& activity,
    const Gfn2HamiltonianDeviceOutput& output, const Gfn2HamiltonianDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_HAMILTONIAN_CUH
