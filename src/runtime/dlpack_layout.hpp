#ifndef XTBLOOM_RUNTIME_DLPACK_LAYOUT_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_RUNTIME_DLPACK_LAYOUT_HPP

/*
 * Byte-exact mirrors of the DLPack 1.0 C managed-tensor layouts, used only to
 * produce capsules consumed by importing frameworks. No DLPack header is
 * bundled or linked: the layouts below are the public ABI described by the
 * DLPack specification (https://github.com/dmlc/dlpack), which is licensed
 * under the Apache License 2.0 and permits independent reimplementation of
 * its data-structure *layout*. Provenance: DLPack 1.0 spec / v0.9+ header
 * `include/dlpack/dlpack.h`, structs DLDataType, DLDevice, DLTensor,
 * DLManagedTensor, and DLManagedTensorVersioned.
 *
 * The xtbloom consumer bridge in `python/xtbloom/_dlpack.py` mirrors these same
 * layouts with the identical offsets, and the production Python test fakes
 * (`python/tests/_dlpack_fakes.py`) verify them byte-for-byte against the
 * "dltensor_versioned" capsules that NumPy produces.
 *
 * IMPORTANT: these mirrored layouts must never drift from the DLPack spec.
 * Offsets are asserted in `src/runtime/dlpack_layout.cpp` so the byte image
 * cannot silently change.
 */

#include <cstddef>
#include <cstdint>

namespace xtbloom::detail {

/* DLPack DLDataTypeCode values used by the producer. */
enum DlpackDtypeCode : std::uint8_t {
  kDlpackInt = 0,
  kDlpackUInt = 1,
  kDlpackFloat = 2,
  kDlpackBfloat = 4,
  kDlpackBool = 6,
};

/* DLPack DLDeviceType values used by the producer. */
enum DlpackDeviceType : std::int32_t {
  kDlpackCpu = 1,
  kDlpackCuda = 2,
};

/* Mirrors DLDataType: type code, bit width, and vector lanes. */
struct DlpackDataType {
  std::uint8_t code;
  std::uint8_t bits;
  std::uint16_t lanes;
};

/* Mirrors DLDevice: the device-kind enum and its ordinal. */
struct DlpackDevice {
  std::int32_t device_type;
  std::int32_t device_id;
};

/* Mirrors DLTensor. Pointer-bearing DLPack images follow the target ABI. */
struct DlpackTensor {
  void* data;
  DlpackDevice device;
  std::int32_t ndim;
  DlpackDataType dtype;
  std::int64_t* shape;
  std::int64_t* strides;
  std::uint64_t byte_offset;
};

struct DlpackManagedTensor;

/* Mirrors the unversioned DLManagedTensor. */
struct DlpackManagedTensor {
  DlpackTensor dl_tensor;
  void* manager_ctx;
  void (*deleter)(DlpackManagedTensor* self);
};

struct DlpackManagedTensorVersioned;

/* Mirrors DLManagedTensorVersioned from DLPack 1.0. */
struct DlpackManagedTensorVersioned {
  std::uint32_t version_major;
  std::uint32_t version_minor;
  void* manager_ctx;
  void (*deleter)(DlpackManagedTensorVersioned* self);
  std::uint64_t flags;
  DlpackTensor dl_tensor;
};

/* DLPack 1.0 flag bits. The producer always exports writable non-copied
 * compact slices, so it only ever emits zero unless the view requests it. */
constexpr std::uint64_t kDlpackFlagReadOnly = 1u << 0u;
constexpr std::uint64_t kDlpackFlagIsCopied = 1u << 1u;

/* DLPack versioned capsule version pair (1, 0). */
constexpr std::uint32_t kDlpackVersionMajor = 1u;
constexpr std::uint32_t kDlpackVersionMinor = 0u;

static_assert(sizeof(DlpackDataType) == 4u, "DLDataType must be 4 bytes");
static_assert(sizeof(DlpackDevice) == 8u, "DLDevice must be 8 bytes");
static_assert(offsetof(DlpackManagedTensor, dl_tensor) == 0u,
              "legacy DLManagedTensor must start with DLTensor");
#if UINTPTR_MAX == UINT64_MAX
static_assert(sizeof(DlpackTensor) == 48u, "LP64 DLTensor must be 48 bytes");
static_assert(sizeof(DlpackManagedTensor) == 64u, "LP64 legacy DLManagedTensor must be 64 bytes");
static_assert(sizeof(DlpackManagedTensorVersioned) == 80u,
              "LP64 versioned DLManagedTensorVersioned must be 80 bytes");
static_assert(offsetof(DlpackManagedTensor, manager_ctx) == 48u,
              "LP64 legacy manager_ctx must sit at byte 48");
static_assert(offsetof(DlpackManagedTensorVersioned, manager_ctx) == 8u,
              "LP64 versioned manager_ctx must sit at byte 8");
static_assert(offsetof(DlpackManagedTensorVersioned, deleter) == 16u,
              "LP64 versioned deleter must sit at byte 16");
static_assert(offsetof(DlpackManagedTensorVersioned, flags) == 24u,
              "LP64 versioned flags must sit at byte 24");
static_assert(offsetof(DlpackManagedTensorVersioned, dl_tensor) == 32u,
              "LP64 versioned DLTensor must sit at byte 32");
#elif UINTPTR_MAX == UINT32_MAX
static_assert(sizeof(DlpackTensor) == 40u, "ILP32 DLTensor must be 40 bytes");
static_assert(sizeof(DlpackManagedTensor) == 48u, "ILP32 legacy DLManagedTensor must be 48 bytes");
static_assert(sizeof(DlpackManagedTensorVersioned) == 64u,
              "ILP32 versioned DLManagedTensorVersioned must be 64 bytes");
static_assert(offsetof(DlpackManagedTensor, manager_ctx) == 40u,
              "ILP32 legacy manager_ctx must sit at byte 40");
static_assert(offsetof(DlpackManagedTensor, deleter) == 44u,
              "ILP32 legacy deleter must sit at byte 44");
static_assert(offsetof(DlpackManagedTensorVersioned, manager_ctx) == 8u,
              "ILP32 versioned manager_ctx must sit at byte 8");
static_assert(offsetof(DlpackManagedTensorVersioned, deleter) == 12u,
              "ILP32 versioned deleter must sit at byte 12");
static_assert(offsetof(DlpackManagedTensorVersioned, flags) == 16u,
              "ILP32 versioned flags must sit at byte 16");
static_assert(offsetof(DlpackManagedTensorVersioned, dl_tensor) == 24u,
              "ILP32 versioned DLTensor must sit at byte 24");
#endif

}  // namespace xtbloom::detail

#endif /* XTBLOOM_RUNTIME_DLPACK_LAYOUT_HPP */
