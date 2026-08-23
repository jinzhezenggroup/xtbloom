#ifndef XTBLOOM_MODEL_GFN2_BASIS_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_BASIS_HPP

#include <string>

#include "model/common/basis.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {

using BasisPlan = common::BasisPlan;

/*
 * Expand the generated, pinned GFN2 element parameters into a reusable basis
 * plan. The STO-nG expansion and normalization follow tblite's
 * tblite_basis_slater implementation exactly, including its special 6s/6p
 * tables. The current GFN2 table contains no repeated angular momenta, so all
 * shell_is_valence entries are one.
 */
xtbloom_status_t make_basis_plan(std::int64_t batch_size, std::int64_t total_atoms,
                                 const std::int64_t* atom_offsets,
                                 const std::int32_t* atomic_numbers, BasisPlan& plan,
                                 std::string& error);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_BASIS_HPP
