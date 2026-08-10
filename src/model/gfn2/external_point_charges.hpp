#ifndef XTBLOOM_MODEL_GFN2_EXTERNAL_POINT_CHARGES_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_EXTERNAL_POINT_CHARGES_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "model/gfn2/basis.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {

/*
 * Geometry-independent layout and shell hardnesses for GFN2 external point
 * charges. Point sites are a ragged, per-molecule partition and never interact
 * with sites or QM atoms from another batch member.
 *
 * shell_hardness stores the Hartree-valued parameter
 *
 *   element_hardness * shell_hubbard_scale
 *
 * in BasisPlan shell order. Plan construction may allocate. All evaluation
 * routines below only read this plan and do not allocate on successful calls.
 */
struct ExternalPointChargePlan {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_point_charges = 0;

  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> batch_shell_offsets;
  std::vector<std::int64_t> point_charge_offsets;
  std::vector<std::int64_t> shell_to_atom;
  std::vector<double> shell_hardness;
};

/*
 * Create a reusable external-charge plan for an existing BasisPlan.
 * point_charge_offsets has batch_size + 1 entries and ends at
 * total_point_charges. For a batch with no point sites, a NULL offsets pointer
 * is accepted and interpreted as an all-zero partition.
 *
 * atomic_numbers must describe the BasisPlan atoms in the same order. Besides
 * selecting generated GFN2 hardnesses, this is checked against the basis shell
 * layout so an accidentally mismatched element list fails during setup.
 */
xtbloom_status_t make_external_point_charge_plan(const BasisPlan& basis,
                                                 const std::int32_t* atomic_numbers,
                                                 std::int64_t total_point_charges,
                                                 const std::int64_t* point_charge_offsets,
                                                 ExternalPointChargePlan& plan, std::string& error);

/*
 * Overwrite shell_potentials with the xTB 6.7.1 GFN2 shell-monopole shift
 *
 *   V_s^PC = sum_p Q_p / sqrt(r_Ap^2 + (2/(gamma_s + gamma_p))^2).
 *
 * Positions are atom/site-major xyz in bohr, charges use elementary-charge
 * units, and point_hardnesses are finite positive gamma_p values in Hartree.
 * For zero point sites, all point-site pointers may be NULL and the shell
 * output is still overwritten with zero.
 */
xtbloom_status_t evaluate_external_point_charge_potential_cpu(
    const ExternalPointChargePlan& plan, const double* qm_positions, const double* point_positions,
    const double* point_charges, const double* point_hardnesses, double* shell_potentials,
    std::string& error);

/*
 * Accumulate the converged explicit embedding energy
 *
 *   E_PC = sum_s q_s V_s^PC
 *
 * into one Hartree value per batch member. shell_charges and shell_potentials
 * follow BasisPlan shell order. This routine deliberately consumes the
 * precomputed potential used by SCC instead of recomputing it.
 */
xtbloom_status_t add_external_point_charge_energy_cpu(const ExternalPointChargePlan& plan,
                                                      const double* shell_charges,
                                                      const double* shell_potentials,
                                                      double* energies, std::string& error);

/*
 * Accumulate explicit external-charge forces for fixed converged shell
 * charges. QM and point-site force buffers use atom/site-major xyz in
 * Hartree/bohr and may independently be NULL. There is no point-charge to
 * point-charge term. Coincident QM/site positions are valid for finite
 * positive hardnesses and contribute zero force.
 */
xtbloom_status_t add_external_point_charge_forces_cpu(
    const ExternalPointChargePlan& plan, const double* qm_positions, const double* point_positions,
    const double* point_charges, const double* point_hardnesses, const double* shell_charges,
    double* qm_forces, double* point_forces, std::string& error);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_EXTERNAL_POINT_CHARGES_HPP
