#ifndef GPUXTB_BACKENDS_CUDA_GFN2_ENERGY_FORCE_EXECUTION_TEST_CUH
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_BACKENDS_CUDA_GFN2_ENERGY_FORCE_EXECUTION_TEST_CUH

#include <cuda_runtime_api.h>

#include <cstdint>

namespace gpuxtb::detail::cuda {

/* Test-only entry through the production CN sequence-promotion/parity
 * launcher. It lets sanitizer tests supply hostile device offsets after a
 * sequence has failed without constructing an earlier failing SCC stage. */
cudaError_t test_gate_gfn2_cn_vjp_parity_cuda(
    std::int64_t batch_size, const std::int64_t* atom_offsets, const std::uint8_t* incoming_mask,
    const std::uint32_t* sparse_sequence_active, const std::uint32_t* dense_sequence_active,
    std::uint32_t* plan_failure, const double* dense_gradients, const double* sparse_gradients,
    double* production_gradients, std::uint32_t* coordination_errors,
    std::uint32_t* execution_device_error, cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif
