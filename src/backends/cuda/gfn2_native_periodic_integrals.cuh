// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_NATIVE_PERIODIC_INTEGRALS_CUH
#define XTBLOOM_BACKENDS_CUDA_GFN2_NATIVE_PERIODIC_INTEGRALS_CUH

#include <cstdint>
#include <type_traits>

#include "backends/cuda/periodic_topology.cuh"

namespace xtbloom::detail::cuda {

/*
 * Immutable device projection of PeriodicIntegralPlan.  The one-electron
 * image list is intentionally separate from the 25-bohr short-range topology:
 * diffuse Gaussian functions can require the pinned 40-bohr image superset.
 * The setup owner uploads these arrays once and every numerical refresh only
 * changes the position/coordination views consumed by the evaluator.
 */
struct Gfn2NativePeriodicIntegralDeviceBatch {
  std::int64_t translation_offset_elements = 0;
  std::int64_t translation_elements = 0;
  std::int64_t max_translations_per_system = 0;
  double realspace_cutoff = 0.0;
  const std::int64_t* translation_offsets = nullptr;
  const Gfn2CudaPeriodicTranslation* translations = nullptr;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2NativePeriodicIntegralDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2NativePeriodicIntegralDeviceBatch>);

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_NATIVE_PERIODIC_INTEGRALS_CUH
