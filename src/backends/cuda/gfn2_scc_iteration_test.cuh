#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_SCC_ITERATION_TEST_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN2_SCC_ITERATION_TEST_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_scc_iteration.cuh"

namespace xtbloom::detail::cuda {

/*
 * Test-only fault descriptor for composed-DAG failure evidence. The dedicated
 * launcher below injects one classified peer code after the named stage has
 * executed and before its report is normalized. Production launchers compile
 * through a separate no-injection specialization and pay no branch or kernel
 * cost. This hook is internal and must never be used for public execution.
 */
struct Gfn2SccIterationTestFault {
  Gfn2SccStageId stage = Gfn2SccStageId::kNone;
  std::int64_t system = -1;
  std::uint32_t raw_code = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2SccIterationTestFault>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationTestFault>);

/* Execute one full iteration with the internal test fault described above. */
[[nodiscard]] Gfn2SccIterationLaunchResult launch_gfn2_scc_iteration_test_fault_cuda(
    const Gfn2SccIterationBinding& binding, const Gfn2SccIterationTestFault& fault,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_SCC_ITERATION_TEST_CUH
