#ifndef XTBLOOM_RUNTIME_VALIDATION_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_RUNTIME_VALIDATION_HPP

#include <cstdint>
#include <string>

#include "xtbloom/xtbloom.h"

namespace xtbloom::detail {

/*
 * Host topology validation deliberately does not copy or dereference device
 * memory. A successful result with pending_offset_checks != 0 is therefore
 * provisional: the selected backend must stage the indicated topology metadata
 * and complete every deferred check before launching a physics kernel or
 * writing caller-owned output memory. The field name is retained temporarily
 * for source compatibility even though the mask now covers more than offsets.
 */
enum TopologyValidationRequirement : std::uint32_t {
  kNoTopologyValidationPending = 0,
  kNoOffsetValidationPending = kNoTopologyValidationPending,
  kAtomOffsetsNeedStaging = 1u << 0,
  kAtomicNumbersNeedStaging = 1u << 1,
  kMolecularChargesNeedStaging = 1u << 2,
  kUnpairedElectronsNeedStaging = 1u << 3,
  kPointChargeOffsetsNeedStaging = 1u << 4,
  kChargeResponseOffsetsNeedStaging = 1u << 5,
  kChargeResponseShapeNeedsStaging = 1u << 6,
  kSpinChannelsNeedStaging = 1u << 7,
  /* Interaction storage not readable on the host because it is
   * CUDA-device-resident. Intentionally not part of the topology staging mask:
   * interactions are external attachments, not topology metadata. */
  kInteractionDescriptorsNeedStaging = 1u << 8,
  kInteractionPayloadNeedsStaging = 1u << 9,
  kCellMatricesNeedStaging = 1u << 10,
  kPeriodicAxesNeedStaging = 1u << 11,
  kTopologyMetadataStagingMask = kAtomOffsetsNeedStaging | kAtomicNumbersNeedStaging |
                                 kMolecularChargesNeedStaging | kUnpairedElectronsNeedStaging |
                                 kPointChargeOffsetsNeedStaging |
                                 kChargeResponseOffsetsNeedStaging | kSpinChannelsNeedStaging,
  kAllTopologyValidationPending = kTopologyMetadataStagingMask | kChargeResponseShapeNeedsStaging,
};

struct DescriptorValidationResult {
  xtbloom_status_t status = XTBLOOM_STATUS_SUCCESS;
  std::uint32_t pending_offset_checks = kNoOffsetValidationPending;
  std::string error;

  [[nodiscard]] bool ok() const noexcept { return status == XTBLOOM_STATUS_SUCCESS; }
  [[nodiscard]] bool requires_backend_staging_validation() const noexcept {
    return ok() && pending_offset_checks != kNoOffsetValidationPending;
  }
};

/*
 * Validate the pointer-safe descriptor structure and aliases that must be
 * proven before reading the model tag for dispatch. This intentionally omits
 * backend feature availability: a structurally valid reserved model must be
 * reported as unsupported by model dispatch, rather than as an unavailable
 * output or interaction of some other model's executor.
 */
[[nodiscard]] DescriptorValidationResult validate_compute_descriptor_structure_for_dispatch(
    xtbloom_backend_t backend, const xtbloom_batch_t* batch,
    const xtbloom_compute_options_t* options, const xtbloom_batch_result_t* result);

/*
 * Validate ABI headers, inline fields, buffer extents/tags, and address ranges
 * without reading any buffer's pointed-to storage. `backend` must be the
 * resolved context backend (CPU or CUDA), never XTBLOOM_BACKEND_AUTO.
 *
 * This layer is safe for opaque pointers whose memory-space tags have not yet
 * been verified against a device runtime. In particular, it does not assume
 * that a HOST tag proves CPU accessibility. A CUDA bridge must perform pointer
 * ownership/type validation before staging or otherwise accessing such memory.
 */
[[nodiscard]] DescriptorValidationResult validate_compute_descriptor_structure(
    xtbloom_backend_t backend, const xtbloom_batch_t* batch,
    const xtbloom_compute_options_t* options, const xtbloom_batch_result_t* result);

/*
 * Validate semantic relationships that can be proven from host-resident
 * topology metadata and report device-resident metadata through the pending
 * mask. The request must first have valid descriptor headers, extents, tags,
 * and required capacities; calling validate_compute_descriptor_structure() is
 * one way to establish that precondition. Every HOST-tagged topology pointer
 * must also be known CPU-accessible (for a CUDA request, by runtime
 * pointer-attribute validation). This function relies on both invariants.
 *
 * Only HOST-tagged topology buffers are dereferenced. CUDA-device buffers are
 * opaque here, including in mixed-memory requests.
 */
[[nodiscard]] DescriptorValidationResult validate_host_topology_semantics(
    const xtbloom_batch_t& batch);

/*
 * Validate the ABI-v1 structural and host-topology sequence through model
 * dispatch. This entry point deliberately stops before backend execution
 * availability so a known reserved model can be refused without inheriting
 * GFN2-specific output diagnostics. Call it directly only for CPU requests or
 * after a CUDA bridge has verified that every HOST-tagged topology pointer is
 * CPU-accessible.
 */
[[nodiscard]] DescriptorValidationResult validate_compute_descriptors_for_dispatch(
    xtbloom_backend_t backend, const xtbloom_batch_t* batch,
    const xtbloom_compute_options_t* options, const xtbloom_batch_result_t* result);

/*
 * Validate the ABI-v4 native-cell contents after the caller's HOST pointers
 * are known to be CPU-accessible or after a CUDA bridge has staged them into
 * HOST storage.  This pass checks only released mask/cell semantics.  The
 * availability pass is separate so malformed requests retain precise errors
 * before a valid periodic request is refused.
 */
[[nodiscard]] DescriptorValidationResult validate_host_lattice_semantics(
    const xtbloom_batch_t& batch);

/*
 * Return SUCCESS for an absent suffix or an explicitly molecular V4 image
 * (all masks NONE with zero cells).  Any released XYZ item is currently
 * refused atomically because native periodic execution is not connected.
 * Call only after validate_host_lattice_semantics() has succeeded.
 */
[[nodiscard]] DescriptorValidationResult validate_host_lattice_execution_availability(
    const xtbloom_batch_t& batch, xtbloom_model_t model);

/*
 * Compatibility entry point for the complete ABI-v1 validation sequence. It
 * preserves the historical error order: descriptor headers/extents/tags,
 * host topology semantics, address-range and alias checks, then backend
 * execution availability.
 */
[[nodiscard]] DescriptorValidationResult validate_compute_descriptors(
    xtbloom_backend_t backend, const xtbloom_batch_t* batch,
    const xtbloom_compute_options_t* options, const xtbloom_batch_result_t* result);

/*
 * Validate batch and compute-options descriptors for fixed-topology plan
 * creation, which has no caller-owned result descriptor yet. The batch plus
 * the compute policy are checked with the same prefix, host-topology, and
 * alias rules as xtbloom_compute; result buffers are simply not required.
 */
[[nodiscard]] DescriptorValidationResult validate_plan_descriptor_structure_for_dispatch(
    xtbloom_backend_t backend, const xtbloom_batch_t* batch,
    const xtbloom_compute_options_t* options);

[[nodiscard]] DescriptorValidationResult validate_plan_descriptor_structure(
    xtbloom_backend_t backend, const xtbloom_batch_t* batch,
    const xtbloom_compute_options_t* options);

/*
 * Validate output and interaction execution availability after model dispatch
 * selected an implemented backend route. Keeping this phase separate prevents
 * a partial/reserved model from inheriting GFN2-specific NOT_IMPLEMENTED
 * diagnostics while preserving the established GFN2 error order.
 */
[[nodiscard]] DescriptorValidationResult validate_compute_execution_availability(
    xtbloom_backend_t backend, const xtbloom_batch_t& batch,
    const xtbloom_compute_options_t& options);

}  // namespace xtbloom::detail

#endif  // XTBLOOM_RUNTIME_VALIDATION_HPP
