#ifndef XTBLOOM_MODEL_GFN1_H0_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN1_H0_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "model/gfn1/basis.hpp"
#include "model/gfn1/integrals.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn1 {

/*
 * Geometry-independent GFN1 extended-Hueckel Hamiltonian data.
 *
 * GFN1 is kept separate from the GFN2 plan because its first-shell valence
 * mask changes the shell-pair scaling branches.  The dense ragged shell-pair
 * storage lets repeated calls evaluate H0 and its VJP without allocation.
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
 * Bind the canonical GFN1 parameter tables to one exact basis/integral
 * topology.  eV-valued shell levels and CN shifts use tblite's historical
 * 1/27.21138505 conversion, not a rounded physical constant.
 */
xtbloom_status_t make_h0_plan(const BasisPlan& basis, const IntegralPlan& integrals,
                              const std::int32_t* atomic_numbers, H0Plan& plan, std::string& error);

/* Overwrite packed dense H0 matrices for a ragged GFN1 batch. */
xtbloom_status_t evaluate_h0_cpu(const BasisPlan& basis, const IntegralPlan& integrals,
                                 const H0Plan& plan, const double* positions,
                                 const double* coordination_numbers, const double* overlap,
                                 double* hamiltonian, std::string& error);

/*
 * Accumulate the reverse-mode derivatives with respect to overlap, atomic CN,
 * and positions.  gradients are dE/dR, not forces; the caller composes the
 * overlap and exponential-CN VJPs separately.
 */
xtbloom_status_t add_h0_vjp_cpu(const BasisPlan& basis, const IntegralPlan& integrals,
                                const H0Plan& plan, const double* positions,
                                const double* coordination_numbers, const double* overlap,
                                const double* dE_dhamiltonian, double* dE_doverlap, double* dE_dcn,
                                double* gradients, std::string& error);

}  // namespace xtbloom::detail::gfn1

#endif  // XTBLOOM_MODEL_GFN1_H0_HPP
