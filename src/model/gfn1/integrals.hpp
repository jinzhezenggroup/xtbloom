#ifndef XTBLOOM_MODEL_GFN1_INTEGRALS_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN1_INTEGRALS_HPP

#include "model/common/integrals.hpp"
#include "model/gfn1/basis.hpp"

namespace xtbloom::detail::gfn1 {

using IntegralPlan = common::IntegralPlan;
inline constexpr double kDefaultIntegralCutoff = common::kDefaultIntegralCutoff;
using common::add_overlap_gradient_cpu;
using common::evaluate_overlap_cpu;
using common::make_integral_plan;

}  // namespace xtbloom::detail::gfn1

#endif  // XTBLOOM_MODEL_GFN1_INTEGRALS_HPP
