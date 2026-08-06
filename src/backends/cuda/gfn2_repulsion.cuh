#ifndef GPUXTB_BACKENDS_CUDA_GFN2_REPULSION_CUH
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_BACKENDS_CUDA_GFN2_REPULSION_CUH

#include <cuda_runtime_api.h>

#include <cstdint>

namespace gpuxtb::detail::cuda {

/*
 * Semantic input errors found asynchronously by the repulsion kernel. The
 * caller-owned device_error scalar is reset to kSuccess before every launch
 * and can be read after synchronizing the supplied stream.
 */
enum class Gfn2RepulsionDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidOffsets = 1u,
  kInvalidAtomicNumberOrParameter = 2u,
  kNonfinitePosition = 3u,
  kCoincidentAtoms = 4u,
};

/*
 * A non-owning view of a ragged batch resident in CUDA-accessible memory.
 * Positions are atom-major xyz coordinates in bohr. atom_offsets has
 * batch_size + 1 entries and atomic_numbers/positions span total_atoms.
 */
struct Gfn2RepulsionDeviceBatch {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  const std::int64_t* atom_offsets = nullptr;
  const std::int32_t* atomic_numbers = nullptr;
  const double* positions = nullptr;
};

/*
 * Queue native ragged-batch GFN2 screened nuclear repulsion on stream.
 *
 * All pointers refer to caller-owned CUDA-accessible memory. The kernel adds
 * Hartree energies and optional Hartree/bohr forces to the existing output
 * values; it never initializes or owns those result buffers. forces may be
 * NULL for energy-only evaluation. device_error is mandatory and is reset
 * asynchronously on stream before the kernel launch. Immutable element and
 * global parameters must already have been uploaded with
 * ensure_cuda_gfn2_parameters for the current device.
 *
 * Input arrays, result arrays, and device_error must not overlap. A
 * device_error scalar also must not be reused concurrently on independent
 * streams unless the caller supplies cross-stream ordering. If a semantic
 * error is reported, some otherwise-valid systems or earlier pairs may
 * already have been accumulated; callers must discard the affected launch's
 * outputs rather than treating the operation as transactional.
 *
 * The return value reports host-side argument, enqueue, and launch failures.
 * Execution failures and Gfn2RepulsionDeviceError require synchronization to
 * observe. No allocation or device-wide synchronization is performed.
 */
cudaError_t add_gfn2_repulsion_cuda(const Gfn2RepulsionDeviceBatch& batch, double* energies,
                                    double* forces, std::uint32_t* device_error,
                                    cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_REPULSION_CUH
