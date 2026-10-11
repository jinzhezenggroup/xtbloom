#ifndef XTBLOOM_MODEL_COMMON_PROJECTION_BASIS_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_COMMON_PROJECTION_BASIS_HPP

#include <cstdint>
#include <string>

#include "model/common/basis.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::common {

/*
 * Construct the auxiliary projection basis used by the local external-energy
 * bridge. Every real atom receives twelve contracted shells for each
 * of l=0,1,2, giving 12 + 36 + 60 = 108 spherical projection functions.
 * The basis is deliberately independent of element type: atomic numbers only
 * affect the native GFN2 basis on the bra side of the cross overlap.
 */
xtbloom_status_t make_external_projection_basis(std::int64_t atom_count, BasisPlan& basis,
                                                std::string& error);

}  // namespace xtbloom::detail::common

#endif  // XTBLOOM_MODEL_COMMON_PROJECTION_BASIS_HPP
