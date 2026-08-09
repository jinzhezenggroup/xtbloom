#ifndef XTBLOOM_MODEL_GFN2_COORDINATION_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_COORDINATION_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {

/*
 * Geometry-independent data for the GFN2 double-exponential coordination
 * number. The atom-wise radii are laid out like the input batch so this plan
 * can later be uploaded directly to an accelerator backend.
 */
struct CoordinationPlan {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<double> covalent_radius;
};

/* Build a reusable ragged-batch plan from atomic numbers and molecule offsets. */
xtbloom_status_t make_coordination_plan(std::int64_t batch_size, std::int64_t total_atoms,
                                        const std::int64_t* atom_offsets,
                                        const std::int32_t* atomic_numbers, CoordinationPlan& plan,
                                        std::string& error);

/*
 * Evaluate atom-wise GFN2 coordination numbers. Positions use atom-major xyz
 * layout in bohr. The output contains total_atoms values and is overwritten.
 * A successful steady-state call does not allocate.
 */
xtbloom_status_t evaluate_coordination_cpu(const CoordinationPlan& plan, const double* positions,
                                           double* coordination_numbers, std::string& error);

/*
 * Apply the analytic reverse-mode coordinate derivative
 *
 *   gradients += (d coordination_numbers / d positions)^T * dE_dcn.
 *
 * This is the form needed by the GFN2 force assembly and avoids materializing
 * an O(n_atom^2) Jacobian. Gradients are dE/dR (not forces), in atom-major xyz
 * layout, and are accumulated into the caller-owned buffer.
 */
xtbloom_status_t add_coordination_gradient_cpu(const CoordinationPlan& plan,
                                               const double* positions, const double* dE_dcn,
                                               double* gradients, std::string& error);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_COORDINATION_HPP
