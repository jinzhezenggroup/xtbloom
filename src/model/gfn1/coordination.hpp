#ifndef XTBLOOM_MODEL_GFN1_COORDINATION_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN1_COORDINATION_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn1 {

/*
 * Geometry-independent data for the GFN1 exponential coordination number.
 * The covalent radii are the generated GFN1 D3 radii in bohr, laid out like
 * the input ragged batch. Plan construction may allocate; evaluation does not.
 */
struct CoordinationPlan {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<double> covalent_radius;
};

/* Build a reusable GFN1 coordination plan from atomic numbers and offsets. */
xtbloom_status_t make_coordination_plan(std::int64_t batch_size, std::int64_t total_atoms,
                                        const std::int64_t* atom_offsets,
                                        const std::int32_t* atomic_numbers, CoordinationPlan& plan,
                                        std::string& error);

/*
 * Overwrite coordination_numbers with the atom-wise GFN1 exponential CNs.
 * Positions use atom-major xyz coordinates in bohr. The complete plan and
 * geometry are validated before the output is modified.
 */
xtbloom_status_t evaluate_coordination_cpu(const CoordinationPlan& plan, const double* positions,
                                           double* coordination_numbers, std::string& error);

/*
 * Apply the reverse-mode coordinate derivative
 *
 *   gradients += (d coordination_numbers / d positions)^T * dE_dcn.
 *
 * gradients are energy gradients, not forces, and use atom-major xyz layout.
 * Validation is transactional: failure leaves gradients unchanged.
 */
xtbloom_status_t add_coordination_gradient_cpu(const CoordinationPlan& plan,
                                               const double* positions, const double* dE_dcn,
                                               double* gradients, std::string& error);

}  // namespace xtbloom::detail::gfn1

#endif  // XTBLOOM_MODEL_GFN1_COORDINATION_HPP
