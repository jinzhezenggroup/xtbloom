#ifndef XTBLOOM_BACKENDS_CUDA_EXTERNAL_ENERGY_DEVICE_EVALUATOR_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_EXTERNAL_ENERGY_DEVICE_EVALUATOR_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/common/gfn2_plan_schema.hpp"
#include "backends/cuda/gfn2_density.cuh"
#include "backends/cuda/gfn2_hamiltonian.cuh"

namespace xtbloom::detail::cuda {

/*
 * Native CUDA representation of an external energy model. The evaluator uses
 * a documented flat parameter arena so uploads remain immutable and CUDA Graph
 * replay can reuse the exact same addresses.
 */
struct ExternalEnergyDeviceModel {
  std::int64_t geometry_dim = 0;
  std::int64_t electronic_dim = 0;
  std::int64_t hidden_dim = 0;
  std::int64_t max_atomic_number = 0;
  std::int64_t projection_width = 0;
  std::int64_t radial_count = 0;
  double output_scale = 1.0;
  const double* parameters = nullptr;
  std::int64_t parameter_elements = 0;
  std::uint32_t flags = 0u;
  std::uint64_t plan_token = 0u;
};

inline constexpr std::uint32_t kExternalEnergyDeviceModelNativeShell = 1u << 0u;
inline constexpr std::uint32_t kExternalEnergyDeviceModelDiagonalProjection = 1u << 1u;
inline constexpr std::uint32_t kExternalEnergyDeviceModelTrainingGradient = 1u << 2u;

struct ExternalEnergyDeviceInput {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t total_matrix_elements = 0;
  std::int64_t total_spin_matrix_elements = 0;
  std::uint64_t plan_token = 0u;

  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* batch_orbital_offsets = nullptr;
  const std::int64_t* matrix_offsets = nullptr;
  const std::int64_t* orbital_to_atom = nullptr;
  const std::int32_t* atomic_numbers = nullptr;
  const double* positions = nullptr;
  const double* molecular_charges = nullptr;
  const std::int32_t* unpaired_electrons = nullptr;
  const std::int32_t* spin_channels = nullptr;
  const double* overlap = nullptr;
  const double* current_density = nullptr;
  const double* staged_density = nullptr;
  Gfn2WavefunctionLayoutView wavefunction_layout{};
};

struct ExternalEnergyDeviceOutput {
  double* hamiltonian = nullptr;
  std::int64_t hamiltonian_elements = 0;
  double* energy = nullptr;
  std::int64_t energy_elements = 0;
  double* energy_accumulator = nullptr;
  std::int64_t energy_accumulator_elements = 0;
  double* parameter_gradient = nullptr;
  std::int64_t parameter_gradient_elements = 0;
  std::uint64_t plan_token = 0u;
  /* Optional stationary force seeds. overlap_adjoint is dE_external/dS in the
   * same packed AO matrix layout consumed by the existing integral reverse
   * pass; geometry_gradient is the explicit dE_external/dR contribution at fixed
   * converged density.  Keeping these as optional trailing fields preserves
   * the energy-only ABI while allowing the force composer to reuse its native
   * overlap/Pulay machinery. */
  double* overlap_adjoint = nullptr;
  std::int64_t overlap_adjoint_elements = 0;
  double* geometry_gradient = nullptr;
  std::int64_t geometry_gradient_elements = 0;
};

static_assert(std::is_trivially_copyable_v<ExternalEnergyDeviceModel>);
static_assert(std::is_standard_layout_v<ExternalEnergyDeviceModel>);
static_assert(std::is_trivially_copyable_v<ExternalEnergyDeviceInput>);
static_assert(std::is_standard_layout_v<ExternalEnergyDeviceInput>);
static_assert(std::is_trivially_copyable_v<ExternalEnergyDeviceOutput>);
static_assert(std::is_standard_layout_v<ExternalEnergyDeviceOutput>);

/* Validate the flat arena dimensions and the fixed native-shell contract. */
bool validate_external_energy_device_model(const ExternalEnergyDeviceModel& model) noexcept;

/* Add dE_external/dD_alpha,beta to the assembled device Hamiltonian. */
cudaError_t evaluate_external_energy_device_potential_cuda(const ExternalEnergyDeviceModel& model,
                                                           const ExternalEnergyDeviceInput& input,
                                                           const ExternalEnergyDeviceOutput& output,
                                                           const std::uint8_t* active_mask,
                                                           cudaStream_t stream = nullptr) noexcept;

/* Evaluate the external energy at the newly staged SCC density. */
cudaError_t evaluate_external_energy_device_energy_cuda(const ExternalEnergyDeviceModel& model,
                                                        const ExternalEnergyDeviceInput& input,
                                                        const ExternalEnergyDeviceOutput& output,
                                                        const std::uint8_t* active_mask,
                                                        cudaStream_t stream = nullptr) noexcept;

/* Evaluate stationary external-energy force seeds directly on the CUDA device.
 * This computes explicit geometry derivatives and dE_external/dS at the converged
 * density; no coordinate displacement or host staging is involved. */
cudaError_t evaluate_external_energy_device_force_cuda(const ExternalEnergyDeviceModel& model,
                                                       const ExternalEnergyDeviceInput& input,
                                                       const ExternalEnergyDeviceOutput& output,
                                                       const std::uint8_t* active_mask,
                                                       cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_EXTERNAL_ENERGY_DEVICE_EVALUATOR_CUH
