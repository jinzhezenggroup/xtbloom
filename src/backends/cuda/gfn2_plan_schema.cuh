#ifndef GPUXTB_BACKENDS_CUDA_GFN2_PLAN_SCHEMA_CUH
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

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

/*
 * Setup-time fail-closed CUDA binders for the common topology projections.
 *
 * The projection types are standard-layout trivially-copyable PODs that are
 * copied byte-for-byte into device descriptors, so a CUDA consumer proves
 * identity through the exact pointer/count/token equality already established
 * by validate_gfn2_topology_cuda_async on the master view.  These binders are
 * therefore host-side and synchronous: they re-run the structural binding
 * validation of the CUDA master (which never dereferences), prove exact
 * pointer identity of each projection field with the master arrays, and only
 * then publish.  No kernel, allocation, transfer, or synchronization is
 * performed, and every binder clears `projection` on failure.
 */
cudaError_t bind_gfn2_atom_projection_cuda(const Gfn2RaggedTopologyView& device_topology,
                                           Gfn2AtomProjectionView& binding) noexcept;

cudaError_t bind_gfn2_shell_ownership_projection_cuda(
    const Gfn2RaggedTopologyView& device_topology,
    Gfn2ShellOwnershipProjectionView& binding) noexcept;

cudaError_t bind_gfn2_ao_matrix_projection_cuda(const Gfn2RaggedTopologyView& device_topology,
                                                Gfn2AOMatrixProjectionView& binding) noexcept;

cudaError_t bind_gfn2_packed_all_pair_projection_cuda(
    const Gfn2RaggedTopologyView& device_topology,
    Gfn2PackedAllPairProjectionView& binding) noexcept;

cudaError_t bind_gfn2_ao_bucket_projection_cuda(const Gfn2RaggedTopologyView& device_topology,
                                                Gfn2AOBucketProjectionView& binding) noexcept;

/*
 * Element identity is owned by setup, not the topology: bind a host-computed
 * projector (with its nonzero order-sensitive fingerprint) into a CUDA-space
 * descriptor that names the uploaded device atomic-number array, proving plan
 * token, counts, fingerprint identity, and CUDA accessibility.
 */
cudaError_t bind_gfn2_element_identity_projection_cuda(
    const Gfn2ElementIdentityProjectionView& host_projection,
    const std::int32_t* device_atomic_numbers,
    Gfn2ElementIdentityProjectionView& device_binding) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_PLAN_SCHEMA_CUH
