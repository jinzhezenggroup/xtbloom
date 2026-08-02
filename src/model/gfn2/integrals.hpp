#ifndef GPUXTB_MODEL_GFN2_INTEGRALS_HPP
#define GPUXTB_MODEL_GFN2_INTEGRALS_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gpuxtb/gpuxtb.h"
#include "model/gfn2/basis.hpp"

namespace gpuxtb::detail::gfn2 {

/*
 * Geometry-independent layout for batched one-electron integral matrices.
 *
 * Every molecule owns a row-major, dense nao-by-nao matrix in the packed
 * output buffer. matrix_offsets is a zero-based half-open partition of that
 * buffer and is reused by overlap, dipole, and quadrupole evaluators.
 *
 * Plan construction may allocate. Evaluation requires a caller-owned scratch
 * buffer so successful steady-state calls do not allocate and the same layout
 * can be mapped to CUDA or a future ROCm backend.
 */
struct IntegralPlan {
  std::int64_t batch_size = 0;
  std::int64_t total_matrix_elements = 0;
  double integral_cutoff = 0.0;
  std::size_t workspace_size_bytes = 0;
  std::vector<std::int64_t> matrix_offsets;
};

/* tblite's GFN2 default at calculator accuracy 1.0. */
inline constexpr double kDefaultIntegralCutoff = 25.0;

/*
 * Create packed matrix offsets for an existing GFN2 BasisPlan. The cutoff is
 * the dimensionless Gaussian-product exponent threshold used by tblite; a
 * primitive pair is skipped when ai*aj*R^2/(ai+aj) exceeds this value.
 */
gpuxtb_status_t make_integral_plan(const BasisPlan& basis, IntegralPlan& plan, std::string& error,
                                   double integral_cutoff = kDefaultIntegralCutoff);

/*
 * Evaluate all ragged-batch overlap matrices.
 *
 * Positions use atom-major xyz coordinates in bohr. overlap contains
 * plan.total_matrix_elements doubles and is overwritten. workspace must be
 * aligned for double and contain at least plan.workspace_size_bytes bytes.
 */
gpuxtb_status_t evaluate_overlap_cpu(const BasisPlan& basis, const IntegralPlan& plan,
                                     const double* positions, double* overlap, void* workspace,
                                     std::size_t workspace_size, std::string& error);

/*
 * Evaluate GFN2 one-electron dipole and traceless quadrupole integrals.
 *
 * Each Cartesian component is a separate packed dense matrix. dipole contains
 * 3*plan.total_matrix_elements doubles in [x,y,z] component-major order;
 * quadrupole contains 6*plan.total_matrix_elements doubles in
 * [xx,xy,yy,xz,yz,zz] component-major order. Within every matrix, rows are
 * bra AOs and columns are ket AOs in tblite's real-spherical ordering.
 *
 * Following tblite/xtb's GFN2 convention, the operator origin of matrix
 * element (mu,nu) is the atom carrying the ket AO nu. Thus interatomic blocks
 * are related by the documented multipole translation identities rather than
 * by plain matrix symmetry. Quadrupoles are Q_ab = 3*r_a*r_b/2 -
 * delta_ab*r^2/2 relative to that ket origin.
 *
 * Positions are atom-major xyz coordinates in bohr. Both output buffers are
 * overwritten and must not overlap each other, positions, or workspace.
 * workspace is caller-owned, must be aligned for double, and must contain at
 * least plan.workspace_size_bytes bytes. Successful steady-state calls
 * perform no dynamic allocation.
 */
gpuxtb_status_t evaluate_multipole_cpu(const BasisPlan& basis, const IntegralPlan& plan,
                                       const double* positions, double* dipole, double* quadrupole,
                                       void* workspace, std::size_t workspace_size,
                                       std::string& error);

/*
 * Apply the analytic overlap reverse-mode derivative
 *
 *   gradients += (d overlap / d positions)^T * dE_doverlap.
 *
 * dE_doverlap has the same packed dense layout as evaluate_overlap_cpu.
 * Gradients are dE/dR, not forces, in atom-major xyz layout. The contraction
 * respects both matrix triangles even when the caller's adjoint is not
 * symmetric, and does not materialize a four-index derivative tensor.
 */
gpuxtb_status_t add_overlap_gradient_cpu(const BasisPlan& basis, const IntegralPlan& plan,
                                         const double* positions, const double* dE_doverlap,
                                         double* gradients, void* workspace,
                                         std::size_t workspace_size, std::string& error);

}  // namespace gpuxtb::detail::gfn2

#endif  // GPUXTB_MODEL_GFN2_INTEGRALS_HPP
