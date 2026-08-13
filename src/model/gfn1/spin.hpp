#ifndef XTBLOOM_MODEL_GFN1_SPIN_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN1_SPIN_HPP

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "model/gfn1/basis.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn1 {

/*
 * Packed charge/magnetization shell layout used by the internal GFN1 spin
 * term before the full GFN1 wavefunction/SCC integration in issue #384.
 * For system i, element_count is nsh*nspin and the magnetization channel, if
 * present, follows the charge channel.
 */
struct SpinPopulationLayout {
  std::int64_t batch_size = 0;
  std::int64_t total_shells = 0;
  std::int64_t element_count = 0;
  std::vector<std::int64_t> system_offsets;
  std::vector<std::int32_t> spin_channels;
};

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

xtbloom_status_t make_spin_population_layout(const BasisPlan& basis,
                                             const std::int32_t* spin_channels,
                                             SpinPopulationLayout& layout, std::string& error);

/*
 * Expand tblite's common element/angular-momentum constants in exact basis
 * shell order. Repeated angular momenta are deliberately not collapsed: for
 * GFN1 hydrogen the 1s/2s atom owns a dense 2x2 matrix with Wss in every entry.
 */
xtbloom_status_t make_spin_polarization_plan(const BasisPlan& basis,
                                             const std::int32_t* atomic_numbers,
                                             const SpinPopulationLayout& populations,
                                             SpinPolarizationPlan& plan, std::string& error);

[[nodiscard]] SpinPolarizationView make_spin_polarization_view(
    const SpinPolarizationPlan& plan) noexcept;

xtbloom_status_t evaluate_spin_polarization_cpu(SpinPolarizationView view,
                                                const double* shell_populations,
                                                double* spin_energies,
                                                double* shell_potentials,
                                                std::string& error);

xtbloom_status_t evaluate_spin_polarization_system_cpu(
    SpinPolarizationView view, std::int64_t system, const double* shell_populations,
    double& spin_energy, double* shell_potentials, std::string& error);

xtbloom_status_t add_spin_polarization_energy_system_cpu(
    SpinPolarizationView view, std::int64_t system, const double* shell_populations,
    double& accumulated_energy, std::string& error);

}  // namespace xtbloom::detail::gfn1

#endif  // XTBLOOM_MODEL_GFN1_SPIN_HPP
