#ifndef GPUXTB_BACKENDS_CUDA_GFN2_PLAN_SCHEMA_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_PLAN_SCHEMA_CUH

#include <cuda_runtime_api.h>

#include <cstdint>

#include "backends/common/gfn2_plan_schema.hpp"

namespace gpuxtb::detail::cuda {

/*
 * Enqueue complete semantic validation of device-resident offsets and maps.
 * A host-visible structural failure returns cudaErrorInvalidValue synchronously,
 * enqueues nothing, and leaves the diagnostic untouched.  For a structurally
 * valid descriptor, the diagnostic must point to one disjoint device object and
 * is overwritten asynchronously with the device-resident semantic result.
 * Address-space provenance must already have been established by the setup
 * builder.  The launcher allocates nothing and is CUDA Graph compatible for
 * structurally valid descriptors.
 */
cudaError_t validate_gfn2_topology_cuda_async(const Gfn2RaggedTopologyView& topology,
                                              Gfn2PlanSchemaDiagnostic* device_diagnostic,
                                              cudaStream_t stream = nullptr) noexcept;

/*
 * Setup-time fail-closed builder.  Host-side structure and device-side values
 * are both checked.  This function synchronizes `stream`; it is not a hot-path
 * primitive.  `binding` is cleared unless the complete validation succeeds.
 */
cudaError_t bind_gfn2_topology_cuda(const Gfn2RaggedTopologyView& candidate,
                                    Gfn2RaggedTopologyView& binding,
                                    Gfn2PlanSchemaDiagnostic* device_diagnostic,
                                    Gfn2PlanSchemaDiagnostic& diagnostic,
                                    cudaStream_t stream = nullptr) noexcept;

/*
 * Validate cache identity and generation on the device.  A null active mask
 * means all systems are active.  Otherwise exactly batch_size zero/one bytes
 * are required.  Pointer provenance must already have passed setup binding;
 * inactive stale members are intentionally accepted.  Host-visible structural
 * failures, including active-mask shape errors, diagnostic/read alias, and an
 * unsafe diagnostic address, return cudaErrorInvalidValue without a launch and
 * leave the diagnostic untouched.  Device-resident active bytes and generation
 * values are validated asynchronously.  The call is Graph safe for
 * structurally valid descriptors.
 */
cudaError_t validate_gfn2_geometry_provenance_cuda_async(
    const Gfn2RaggedTopologyView& topology, const Gfn2GeometryCacheProvenanceView& provenance,
    std::uint64_t expected_geometry_generation, const std::uint8_t* active_mask,
    std::int64_t active_mask_elements, Gfn2PlanSchemaDiagnostic* device_diagnostic,
    cudaStream_t stream = nullptr) noexcept;

/* Synchronous setup-time counterpart with transactional publication. */
cudaError_t bind_gfn2_geometry_provenance_cuda(
    const Gfn2RaggedTopologyView& topology, const Gfn2GeometryCacheProvenanceView& candidate,
    std::uint64_t expected_geometry_generation, const std::uint8_t* active_mask,
    std::int64_t active_mask_elements, Gfn2GeometryCacheProvenanceView& binding,
    Gfn2PlanSchemaDiagnostic* device_diagnostic, Gfn2PlanSchemaDiagnostic& diagnostic,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_PLAN_SCHEMA_CUH
