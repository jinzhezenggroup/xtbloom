#ifndef XTBLOOM_MODEL_GFN1_REPULSION_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN1_REPULSION_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn1 {

/*
 * Geometry-independent data for GFN1 effective nuclear repulsion. Atom-wise
 * sqrt(arep) and zeff arrays avoid rebuilding symmetric pair parameters for
 * each geometry while retaining the exact geometric/product pair averages.
 */
struct RepulsionPlan {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<double> sqrt_alpha;
  std::vector<double> effective_charge;
};

/* Build a reusable GFN1 repulsion plan from atomic numbers and offsets. */
xtbloom_status_t make_repulsion_plan(std::int64_t batch_size, std::int64_t total_atoms,
                                     const std::int64_t* atom_offsets,
                                     const std::int32_t* atomic_numbers, RepulsionPlan& plan,
                                     std::string& error);

/*
 * Accumulate GFN1 effective-repulsion energy and optional forces. Positions
 * use bohr, energies use Hartree, and forces use Hartree/bohr. The complete
 * request is validated before either caller-owned output is modified.
 */
xtbloom_status_t add_repulsion_cpu(const RepulsionPlan& plan, const double* positions,
                                   double* energies, double* forces, std::string& error);

}  // namespace xtbloom::detail::gfn1

#endif  // XTBLOOM_MODEL_GFN1_REPULSION_HPP
