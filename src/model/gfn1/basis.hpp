#ifndef XTBLOOM_MODEL_GFN1_BASIS_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN1_BASIS_HPP

#include <string>

#include "model/common/basis.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn1 {

using BasisPlan = common::BasisPlan;

/*
 * Expand the canonical GFN1 element table into a reusable ragged basis plan.
 * The first shell at each angular momentum is valence; a later matching shell
 * is orthogonalized specifically to that first shell and marked non-valence.
 */
xtbloom_status_t make_basis_plan(std::int64_t batch_size, std::int64_t total_atoms,
                                 const std::int64_t* atom_offsets,
                                 const std::int32_t* atomic_numbers, BasisPlan& plan,
                                 std::string& error);

}  // namespace xtbloom::detail::gfn1

#endif  // XTBLOOM_MODEL_GFN1_BASIS_HPP
