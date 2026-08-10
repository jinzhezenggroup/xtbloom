#ifndef XTBLOOM_MODEL_GFN2_REPULSION_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_REPULSION_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {

/*
 * Geometry-independent data for the GFN2 screened nuclear repulsion term.
 * Plan construction may allocate. Evaluation only reads these arrays and does
 * not allocate on a successful steady-state call.
 */
struct RepulsionPlan {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<double> sqrt_alpha;
  std::vector<double> effective_charge;
  std::vector<std::uint8_t> light_element;
};

/* Build a reusable ragged-batch plan from atomic numbers and molecule offsets. */
xtbloom_status_t make_repulsion_plan(std::int64_t batch_size, std::int64_t total_atoms,
                                     const std::int64_t* atom_offsets,
                                     const std::int32_t* atomic_numbers, RepulsionPlan& plan,
                                     std::string& error);

/*
 * Accumulate GFN2 repulsion energy and optional forces into caller-owned
 * buffers. Positions use atom-major xyz layout in bohr; output is Hartree and
 * Hartree/bohr. Forces may be NULL for an energy-only evaluation.
 */
xtbloom_status_t add_repulsion_cpu(const RepulsionPlan& plan, const double* positions,
                                   double* energies, double* forces, std::string& error);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_REPULSION_HPP
