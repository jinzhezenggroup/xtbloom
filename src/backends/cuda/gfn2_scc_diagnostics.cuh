#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_SCC_DIAGNOSTICS_CUH
#define XTBLOOM_BACKENDS_CUDA_GFN2_SCC_DIAGNOSTICS_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_scc_iteration.cuh"

namespace xtbloom::detail::cuda {

/* Diagnostic-only, setup-owned storage. Read after the containing public call
 * completes; these observations are not a synchronization or dispatch input.
 * xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION. */
inline constexpr std::uint64_t kGfn2SccDiagnosticUnknown =
    std::numeric_limits<std::uint64_t>::max();

struct Gfn2SccDiagnosticHeader {
  std::uint64_t iteration_count = 0u;
  std::uint64_t current_iteration = kGfn2SccDiagnosticUnknown;
  std::uint64_t overflow = 0u;
  std::uint64_t execution_mode = kGfn2SccDiagnosticUnknown;
};

struct Gfn2SccDiagnosticIteration {
  std::uint64_t start_ns = 0u;
  std::uint64_t end_ns = 0u;
  std::uint64_t plan_failure_record = 0u;
};

struct Gfn2SccDiagnosticBucket {
  std::uint64_t active_systems = 0u;
  std::uint64_t active_channels = 0u;
  std::uint64_t submitted_solver_slots = kGfn2SccDiagnosticUnknown;
  std::uint64_t submitted_backtransform_slots = kGfn2SccDiagnosticUnknown;
  std::uint64_t converged_systems = 0u;
  std::uint64_t failed_systems = 0u;
  std::uint64_t exhausted_systems = 0u;
};

/* Bucket metadata is uploaded once because the provider's original bucket
 * array is host storage. Rows use iteration-major, then exact-AO bucket order. */
struct Gfn2SccDiagnosticsDevice {
  Gfn2SccDiagnosticHeader* header = nullptr;
  Gfn2SccDiagnosticIteration* iterations = nullptr;
  Gfn2SccDiagnosticBucket* rows = nullptr;
  Gfn2EigensolverBucket* buckets = nullptr;
  std::int64_t bucket_count = 0;
  std::uint64_t maximum_iterations = 0u;
};

[[nodiscard]] cudaError_t reset_gfn2_scc_diagnostics_cuda(
    Gfn2SccDiagnosticsDevice diagnostics, cudaStream_t stream,
    std::uint32_t execution_mode = 0u) noexcept;

[[nodiscard]] cudaError_t begin_gfn2_scc_diagnostics_cuda(
    Gfn2SccDiagnosticsDevice diagnostics, const Gfn2SccIterationBinding& binding,
    cudaStream_t stream, bool record_inactive_submissions = false) noexcept;

[[nodiscard]] cudaError_t finish_gfn2_scc_diagnostics_cuda(Gfn2SccDiagnosticsDevice diagnostics,
                                                           const Gfn2SccIterationBinding& binding,
                                                           bool exact_capacity,
                                                           cudaStream_t stream) noexcept;

}  // namespace xtbloom::detail::cuda

#endif
