#ifndef GPUXTB_RUNTIME_VALIDATION_HPP
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_RUNTIME_VALIDATION_HPP

#include <cstdint>
#include <string>

#include "gpuxtb/gpuxtb.h"

namespace gpuxtb::detail {

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
  kTopologyMetadataStagingMask = kAtomOffsetsNeedStaging | kAtomicNumbersNeedStaging |
                                 kMolecularChargesNeedStaging | kUnpairedElectronsNeedStaging |
                                 kPointChargeOffsetsNeedStaging |
                                 kChargeResponseOffsetsNeedStaging | kSpinChannelsNeedStaging,
  kAllTopologyValidationPending = kTopologyMetadataStagingMask | kChargeResponseShapeNeedsStaging,
};

struct DescriptorValidationResult {
  gpuxtb_status_t status = GPUXTB_STATUS_SUCCESS;
  std::uint32_t pending_offset_checks = kNoOffsetValidationPending;
  std::string error;

  [[nodiscard]] bool ok() const noexcept { return status == GPUXTB_STATUS_SUCCESS; }
  [[nodiscard]] bool requires_backend_staging_validation() const noexcept {
    return ok() && pending_offset_checks != kNoOffsetValidationPending;
  }
};

/*
 * Validate ABI headers, inline fields, buffer extents/tags, and address ranges
 * without reading any buffer's pointed-to storage. `backend` must be the
 * resolved context backend (CPU or CUDA), never GPUXTB_BACKEND_AUTO.
 *
 * This layer is safe for opaque pointers whose memory-space tags have not yet
 * been verified against a device runtime. In particular, it does not assume
 * that a HOST tag proves CPU accessibility. A CUDA bridge must perform pointer
 * ownership/type validation before staging or otherwise accessing such memory.
 */
[[nodiscard]] DescriptorValidationResult validate_compute_descriptor_structure(
    gpuxtb_backend_t backend, const gpuxtb_batch_t* batch, const gpuxtb_compute_options_t* options,
    const gpuxtb_batch_result_t* result);

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
    const gpuxtb_batch_t& batch);

/*
 * Compatibility entry point for the complete ABI-v1 validation sequence. It
 * preserves the historical error order: descriptor headers/extents/tags,
 * followed by host topology semantics, followed by address-range and alias
 * checks. Call it directly only for CPU requests or after a CUDA bridge has
 * verified that every HOST-tagged topology pointer is CPU-accessible.
 */
[[nodiscard]] DescriptorValidationResult validate_compute_descriptors(
    gpuxtb_backend_t backend, const gpuxtb_batch_t* batch, const gpuxtb_compute_options_t* options,
    const gpuxtb_batch_result_t* result);

}  // namespace gpuxtb::detail

#endif  // GPUXTB_RUNTIME_VALIDATION_HPP
