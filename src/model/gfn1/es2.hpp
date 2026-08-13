#ifndef XTBLOOM_MODEL_GFN1_ES2_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN1_ES2_HPP

#include <cstdint>
#include <string>

#include "model/gfn1/basis.hpp"
#include "model/gfn2/es2.hpp"

namespace xtbloom::detail::gfn1 {

/*
 * GFN1 uses the same isotropic gexp=2 effective-Coulomb kernel and cache
 * layout as GFN2, but with a distinct parameter expansion and harmonic shell
 * hardness averaging. Aliasing the sealed storage keeps the numerical kernel
 * genuinely shared without making GFN1 consume a GFN2 parameter family.
 */
using ES2Plan = gfn2::ES2Plan;
using ES2GeometryCache = gfn2::ES2GeometryCache;
using ES2Workspace = gfn2::ES2Workspace;

xtbloom_status_t make_es2_plan(const BasisPlan& basis, const std::int32_t* atomic_numbers,
                               ES2Plan& plan, std::string& error);

using gfn2::add_es2_energy_cpu;
using gfn2::add_es2_energy_system_cpu;
using gfn2::add_es2_gradient_cpu;
using gfn2::evaluate_es2_potential_cpu;
using gfn2::evaluate_es2_potential_system_cpu;
using gfn2::update_es2_geometry_cache_cpu;

}  // namespace xtbloom::detail::gfn1

#endif  // XTBLOOM_MODEL_GFN1_ES2_HPP
