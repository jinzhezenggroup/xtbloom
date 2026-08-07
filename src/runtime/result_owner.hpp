#ifndef GPUXTB_RUNTIME_RESULT_OWNER_HPP
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_RUNTIME_RESULT_OWNER_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

#include "gpuxtb/gpuxtb.h"
#include "runtime/dlpack_layout.hpp"

namespace gpuxtb::detail {

/*
 * Ref-counted gpuxtb-owned result arena.
 *
 * One ResultOwner holds one contiguous allocation (host memory or CUDA device
 * memory). Public execution writes into it through caller-bound slices and a
 * DLPack producer hands slices to importing frameworks; the arena is freed
 * exactly once, when the last retained reference falls to zero.
 *
 * Reference accounting:
 * - create allocates the arena with an initial reference (the producer
 *   reference held by the Python ArrayBatchResult arena coordinator).
 * - Each exported capsule calls retain() once at export and its native deleter
 *   calls release() once when the importing framework is done. Every
 *   DLPackResultBuffer also retains on construction and releases on close or
 *   garbage collection.
 * - The final release frees the arena through the correct host/CUDA path.
 */
class ResultOwner {
 public:
  ResultOwner(gpuxtb_memory_space_t memory_space, std::int32_t device_id,
              std::size_t size_bytes, void* data)
      : memory_space_(memory_space),
        device_id_(device_id),
        size_bytes_(size_bytes),
        data_(data),
        refcount_(1u) {}
  ResultOwner(const ResultOwner&) = delete;
  ResultOwner& operator=(const ResultOwner&) = delete;

  gpuxtb_memory_space_t memory_space() const noexcept { return memory_space_; }
  std::int32_t device_id() const noexcept { return device_id_; }
  std::size_t size_bytes() const noexcept { return size_bytes_; }
  void* data() const noexcept { return data_; }

  void retain() noexcept { refcount_.fetch_add(1u, std::memory_order_relaxed); }

  /* Release one reference; the arena and this object are freed at zero.
   * Returns true when this was the final reference, so the caller may also
   * release the enclosing public wrapper handle. */
  bool release() noexcept {
    if (refcount_.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
      destroy();
      return true;
    }
    return false;
  }

 private:
  ~ResultOwner() = default;

  /* Free data_ through the correct allocation path, then delete this. */
  void destroy() noexcept;

  gpuxtb_memory_space_t memory_space_;
  std::int32_t device_id_;
  std::size_t size_bytes_;
  void* data_;
  std::atomic<std::uint32_t> refcount_;
};

/*
 * Host allocation used when memory_space == GPUXTB_MEMORY_HOST. The pointer is
 * 64-byte aligned so any gpuxtb scalar slice and any common DLPack consumer
 * alignment requirement is satisfied regardless of arena layout.
 */
gpuxtb_status_t allocate_host_result_arena(std::size_t size_bytes, void** data,
                                           std::string& error) noexcept;
void free_host_result_arena(void* data) noexcept;

#if defined(GPUXTB_HAS_CUDA)
/*
 * CUDA device allocation used when memory_space == GPUXTB_MEMORY_CUDA_DEVICE.
 * Implemented in result_owner_cuda.cu; every exit restores the caller's
 * current device, and a failure leaves *data untouched.
 */
gpuxtb_status_t allocate_cuda_result_arena(std::int32_t device_id, std::size_t size_bytes,
                                           void** data, std::string& error) noexcept;
/* Frees a CUDA arena, preserving the caller's current device on exit. */
void free_cuda_result_arena(std::int32_t device_id, void* data) noexcept;
#endif  // GPUXTB_HAS_CUDA

/* Scalar width and natural alignment for one supported DLPack dtype code/bits
 * pair, or zero when unsupported. */
std::size_t dlpack_dtype_size(std::int32_t code, std::int32_t bits) noexcept;

/* The DLPack device type reported for one gpuxtb memory space; -1 if unknown. */
std::int32_t dlpack_device_type(gpuxtb_memory_space_t memory_space) noexcept;

}  // namespace gpuxtb::detail

#endif /* GPUXTB_RUNTIME_RESULT_OWNER_HPP */