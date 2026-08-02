#ifndef GPUXTB_RUNTIME_VALIDATION_HPP
#define GPUXTB_RUNTIME_VALIDATION_HPP

#include <cstdint>
#include <string>

#include "gpuxtb/gpuxtb.h"

namespace gpuxtb::detail {

/*
 * Descriptor validation deliberately does not copy or dereference CUDA memory.
 * A successful result with pending_offset_checks != 0 is therefore provisional:
 * the selected backend must stage those small offset arrays and apply the same
 * monotonicity, endpoint, and response-matrix shape checks before launching any
 * physics kernel or writing caller-owned output memory.
 */
enum OffsetValidationRequirement : std::uint32_t {
  kNoOffsetValidationPending = 0,
  kAtomOffsetsNeedStaging = 1u << 0,
  kPointChargeOffsetsNeedStaging = 1u << 1,
  kChargeResponseOffsetsNeedStaging = 1u << 2,
  kChargeResponseShapeNeedsStaging = 1u << 3,
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
 * Validate the complete ABI-v1 compute request without modifying any descriptor
 * or pointed-to storage. `backend` must be the resolved context backend (CPU or
 * CUDA), never GPUXTB_BACKEND_AUTO. Host offsets are inspected immediately;
 * device offsets are reported through pending_offset_checks as described above.
 */
[[nodiscard]] DescriptorValidationResult validate_compute_descriptors(
    gpuxtb_backend_t backend, const gpuxtb_batch_t* batch, const gpuxtb_compute_options_t* options,
    const gpuxtb_batch_result_t* result);

}  // namespace gpuxtb::detail

#endif  // GPUXTB_RUNTIME_VALIDATION_HPP
