#ifndef XTBLOOM_MODEL_GFN2_INTEGRALS_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_INTEGRALS_HPP

#include "model/common/integrals.hpp"
#include "model/gfn2/basis.hpp"

namespace xtbloom::detail::gfn2 {

using IntegralPlan = common::IntegralPlan;
inline constexpr double kDefaultIntegralCutoff = common::kDefaultIntegralCutoff;
using common::add_multipole_gradient_cpu;
using common::add_overlap_gradient_cpu;
using common::evaluate_multipole_cpu;
using common::evaluate_overlap_cpu;
using common::make_integral_plan;

/*
 * Create packed matrix offsets for an existing basis plan. The cutoff is
 * the dimensionless Gaussian-product exponent threshold used by tblite; a
 * primitive pair is skipped when ai*aj*R^2/(ai+aj) exceeds this value.
 */
/*
 * Evaluate all ragged-batch overlap matrices.
 *
 * Positions use atom-major xyz coordinates in bohr. overlap contains
 * plan.total_matrix_elements doubles and is overwritten. workspace must be
 * aligned for double and contain at least plan.workspace_size_bytes bytes.
 */
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
/*
 * Apply the analytic reverse-mode derivative of evaluate_multipole_cpu.
 *
 * dE_ddipole and dE_dquadrupole use the same component-major packed layouts as
 * the corresponding evaluator outputs. Both complete matrix directions are
 * contracted independently, so adjoints need not be symmetric. The reverse
 * pass includes the explicit ket-origin translation used for interatomic
 * transpose blocks and accumulates atom-major dE/dR into gradients.
 *
 * Inputs and workspace must not overlap gradients. workspace has the same
 * alignment and size requirements as evaluate_multipole_cpu. Successful
 * steady-state calls allocate no dynamic memory.
 */
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
}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_INTEGRALS_HPP
