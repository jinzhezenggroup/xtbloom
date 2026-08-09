#ifndef XTBLOOM_MODEL_GFN2_H0_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_H0_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "model/gfn2/basis.hpp"
#include "model/gfn2/integrals.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {

/*
 * Geometry-independent data for the GFN2 extended-Hueckel core
 * Hamiltonian. Shell-pair scaling factors are stored as one dense ragged
 * matrix per molecule. This avoids transcendental work during repeated
 * inference while keeping storage proportional to the much smaller shell
 * space rather than the orbital matrix space.
 *
 * Plan construction may allocate. Evaluation and reverse-mode contraction
 * only read the plan and do not allocate on successful steady-state calls.
 */
struct H0Plan {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t total_matrix_elements = 0;

  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> batch_shell_offsets;
  std::vector<std::int64_t> batch_orbital_offsets;
  std::vector<std::int64_t> matrix_offsets;
  std::vector<std::int64_t> shell_pair_offsets;

  std::vector<double> atomic_radii;
  std::vector<double> shell_levels;
  std::vector<double> shell_coordination_scale;
  std::vector<double> shell_polynomial;
  std::vector<double> shell_pair_scale;
};

/*
 * Build a reusable H0 plan for a BasisPlan/IntegralPlan pair. Atomic numbers
 * must be in the same batch-major order used to create the basis. GFN2 shell
 * levels and coordination shifts are converted from their parameter-file eV
 * convention to Hartree while constructing the plan.
 */
xtbloom_status_t make_h0_plan(const BasisPlan& basis, const IntegralPlan& integrals,
                              const std::int32_t* atomic_numbers, H0Plan& plan, std::string& error);

/*
 * Evaluate packed ragged-batch GFN2 core Hamiltonians.
 *
 * Positions use atom-major xyz coordinates in bohr. coordination_numbers has
 * one value per atom, overlap and hamiltonian use IntegralPlan's packed dense
 * row-major layout, and hamiltonian is overwritten. The caller may supply an
 * arbitrary overlap matrix; using an all-ones matrix exposes the shell-pair
 * H0 scaling factors used by tblite's Hamiltonian tests.
 */
xtbloom_status_t evaluate_h0_cpu(const BasisPlan& basis, const IntegralPlan& integrals,
                                 const H0Plan& plan, const double* positions,
                                 const double* coordination_numbers, const double* overlap,
                                 double* hamiltonian, std::string& error);

/*
 * Apply the reverse-mode derivative of evaluate_h0_cpu.
 *
 * For an incoming packed dE_dhamiltonian, this routine accumulates into the
 * three independent downstream adjoints:
 *
 *   dE_doverlap += (d H0 / d overlap)^T dE_dhamiltonian
 *   dE_dcn     += (d H0 / d CN)^T      dE_dhamiltonian
 *   gradients  += the direct shell-polynomial coordinate derivative
 *
 * The caller composes dE_doverlap with add_overlap_gradient_cpu and dE_dcn
 * with add_coordination_gradient_cpu. gradients are dE/dR, not forces.
 */
xtbloom_status_t add_h0_vjp_cpu(const BasisPlan& basis, const IntegralPlan& integrals,
                                const H0Plan& plan, const double* positions,
                                const double* coordination_numbers, const double* overlap,
                                const double* dE_dhamiltonian, double* dE_doverlap, double* dE_dcn,
                                double* gradients, std::string& error);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_H0_HPP
