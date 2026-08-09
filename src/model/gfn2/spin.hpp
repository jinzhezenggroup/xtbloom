#ifndef XTBLOOM_MODEL_GFN2_SPIN_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_SPIN_HPP

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "model/gfn2/basis.hpp"
#include "model/gfn2/wavefunction.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {

/*
 * Geometry-independent, atom-local GFN2 spin-polarization parameters.
 *
 * coupling_offsets partitions coupling_matrices by atom. Atom A owns a dense
 * row-major [nsh_A, nsh_A] W matrix in the exact shell order of BasisPlan;
 * this is important for transition metals whose parameter order is d,s,p.
 * Population fields use WavefunctionLayout's per-system charge/magnetization
 * packing, so restricted and unrestricted systems may share one ragged plan.
 */
struct SpinPolarizationPlan {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t shell_population_elements = 0;

  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> batch_shell_offsets;
  std::vector<std::int64_t> atom_shell_offsets;
  std::vector<std::int64_t> shell_population_offsets;
  std::vector<std::int32_t> spin_channels;
  std::vector<std::int64_t> coupling_offsets;
  std::vector<double> coupling_matrices;
};

/*
 * Address-space-neutral view for CPU, CUDA, and future ROCm implementations.
 * Every pointer has an explicit element count so truncated staged descriptors
 * are rejected before numerical data is touched.
 */
struct SpinPolarizationView {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t shell_population_elements = 0;

  std::int64_t atom_offset_count = 0;
  std::int64_t batch_shell_offset_count = 0;
  std::int64_t atom_shell_offset_count = 0;
  std::int64_t shell_population_offset_count = 0;
  std::int64_t spin_channel_count = 0;
  std::int64_t coupling_offset_count = 0;
  std::int64_t coupling_matrix_count = 0;

  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* batch_shell_offsets = nullptr;
  const std::int64_t* atom_shell_offsets = nullptr;
  const std::int64_t* shell_population_offsets = nullptr;
  const std::int32_t* spin_channels = nullptr;
  const std::int64_t* coupling_offsets = nullptr;
  const double* coupling_matrices = nullptr;
};

static_assert(std::is_trivially_copyable_v<SpinPolarizationView>);
static_assert(std::is_standard_layout_v<SpinPolarizationView>);

/*
 * Pin tblite's element/angular-momentum spin constants to an exact GFN2 basis
 * and wavefunction packing. Plan construction may allocate; evaluation does
 * not. The supplied basis and wavefunction must describe one identical ragged
 * topology and chemical identity.
 */
xtbloom_status_t make_spin_polarization_plan(const BasisPlan& basis,
                                             const WavefunctionLayout& wavefunction,
                                             SpinPolarizationPlan& plan, std::string& error);

[[nodiscard]] SpinPolarizationView make_spin_polarization_view(
    const SpinPolarizationPlan& plan) noexcept;

/*
 * Overwrite one spin energy per system and one charge/magnetization shell
 * potential per packed wavefunction element:
 *
 *   E_spin = 1/2 sum_A m_A^T W_A m_A,    v_mag,A = W_A m_A.
 *
 * Restricted systems produce exact zeros. For unrestricted systems only the
 * magnetization channel is nonzero. All structure, aliasing, inputs, and
 * arithmetic are preflighted before either output is modified, providing
 * batch-atomic failure without dynamic allocation or caller scratch.
 */
xtbloom_status_t evaluate_spin_polarization_cpu(SpinPolarizationView view,
                                                const double* shell_populations,
                                                double* spin_energies, double* shell_potentials,
                                                std::string& error);

/*
 *  Compute the spin energy and magnetization shell potential of exactly one
 *  ragged batch member. shell_populations and shell_potentials retain the
 *  full packed layout, but only the selected system's population and shell
 *  potential slices are inspected or modified. Restricted systems produce a
 *  zero energy and write nothing (their potential slice must already be
 *  zeroed by the caller). This lets an SCC worker prepare the Hamiltonian for
 *  a successful member while peers may fail independently.
 *
 *  Structural and aliasing failures return INVALID_ARGUMENT. Invalid target
 *  numerical data or target arithmetic failure return INTERNAL_ERROR; the
 *  target potential slice may then be partially modified and the accumulated
 *  energy is unchanged, so callers must treat the whole target system as
 *  failed. The routine allocates no memory and needs no scratch.
 */
xtbloom_status_t evaluate_spin_polarization_system_cpu(
    SpinPolarizationView view, std::int64_t system, const double* shell_populations,
    double& spin_energy, double* shell_potentials, std::string& error);

/*
 * Accumulate the spin energy of one packed system. Only the target population
 * slice is inspected, allowing SCC workers to evaluate newly solved raw
 * multipoles while failed or inactive peers remain untouched. The accumulator
 * is unchanged on every failure.
 */
xtbloom_status_t add_spin_polarization_energy_system_cpu(SpinPolarizationView view,
                                                         std::int64_t system,
                                                         const double* shell_populations,
                                                         double& accumulated_energy,
                                                         std::string& error);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_SPIN_HPP
