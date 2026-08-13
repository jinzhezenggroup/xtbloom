// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#ifndef XTBLOOM_MODEL_GFN1_EXTERNAL_POINT_CHARGES_HPP
#define XTBLOOM_MODEL_GFN1_EXTERNAL_POINT_CHARGES_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "model/gfn1/basis.hpp"
#include "model/gfn1/es2.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn1 {

/*
 * Geometry-independent GFN1 explicit point-charge layout. The shell
 * hardnesses are copied from the already sealed harmonic GFN1 ES2 plan, so
 * point-charge screening cannot drift from the SCC Coulomb parameterization.
 * Point sites form a ragged per-system partition and never couple different
 * batch members.
 */
struct ExternalPointChargePlan {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_point_charges = 0;

  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> batch_shell_offsets;
  std::vector<std::int64_t> atom_shell_offsets;
  std::vector<std::int64_t> point_charge_offsets;
  std::vector<std::int64_t> atom_to_batch;
  std::vector<std::int64_t> point_to_batch;
  std::vector<std::int64_t> shell_to_atom;
  std::vector<double> shell_hardness;
};

/*
 * Build a reusable point-charge plan from one exact GFN1 basis/ES2 topology.
 * The ES2 plan must be sealed with harmonic shell-hardness averaging.
 * point_charge_offsets contains batch_size + 1 values and ends at
 * total_point_charges. For zero point sites, NULL denotes an all-zero
 * partition. Construction is transactional: plan is unchanged on failure.
 */
xtbloom_status_t make_external_point_charge_plan(const BasisPlan& basis, const ES2Plan& es2,
                                                 std::int64_t total_point_charges,
                                                 const std::int64_t* point_charge_offsets,
                                                 ExternalPointChargePlan& plan, std::string& error);

/*
 * Overwrite the packed shell potential with the xTB GFN1 PCEM interaction
 *
 *   x_sp = 2 / (1/gamma_s + 1/gamma_p),
 *   V_s  = sum_p Q_p / sqrt(r_Ap^2 + x_sp^(-2)).
 *
 * Positions are atom/site-major xyz in bohr, charges use elementary-charge
 * units, and gamma values are finite positive Hartree parameters. Validation
 * and a complete arithmetic preflight occur before shell_potentials changes.
 */
xtbloom_status_t evaluate_external_point_charge_potential_cpu(
    const ExternalPointChargePlan& plan, const double* qm_positions, const double* point_positions,
    const double* point_charges, const double* point_hardnesses, double* shell_potentials,
    std::string& error);

/*
 * Accumulate E_PC = sum_s q_s V_s into one Hartree value per system. There is
 * no one-half factor because the external sites are fixed sources. Outputs
 * remain unchanged on validation or arithmetic failure.
 */
xtbloom_status_t add_external_point_charge_energy_cpu(const ExternalPointChargePlan& plan,
                                                      const double* shell_charges,
                                                      const double* shell_potentials,
                                                      double* energies, std::string& error);

/*
 * Accumulate analytic public forces (negative coordinate gradients) at fixed
 * converged shell charges. QM and point-site outputs are in Hartree/bohr and
 * may independently be NULL. Pair forces are equal and opposite; coincident
 * finite-hardness sites have finite energy and zero force. The complete call
 * is preflighted before either force domain is modified.
 */
xtbloom_status_t add_external_point_charge_forces_cpu(
    const ExternalPointChargePlan& plan, const double* qm_positions, const double* point_positions,
    const double* point_charges, const double* point_hardnesses, const double* shell_charges,
    double* qm_forces, double* point_forces, std::string& error);

}  // namespace xtbloom::detail::gfn1

#endif  // XTBLOOM_MODEL_GFN1_EXTERNAL_POINT_CHARGES_HPP
