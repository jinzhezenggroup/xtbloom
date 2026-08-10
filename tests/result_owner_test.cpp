// Unit tests for the xtbloom-owned result arena and its DLPack export.
//
// xtbloom_result_owner_t allocates a host or CUDA device arena that the caller
// fills through a normal compute call and hands to an importing framework
// through the DLPack producer protocol. These tests prove the ref-counting
// contract, transactional failure behavior, and the byte-exact DLPack
// managed-tensor layout independently of Python.
//
// xtbloom's public CUDA compute is synchronous, so after xtbloom_compute
// returns the requested result bytes are fully committed and a producer
// export needs no additional device-wide synchronization or hidden host
// polling.

#include <cstdint>
#include <cstring>
#include <string>

#include "xtbloom/xtbloom.h"

namespace {

// Byte-exact mirrors of the DLPack 1.0 managed-tensor layouts (see
// src/runtime/dlpack_layout.hpp; DLPack spec, Apache-2.0).
struct DtTensor {
  void* data;
  struct {
    std::int32_t device_type;
    std::int32_t device_id;
  } device;
  std::int32_t ndim;
  struct {
    std::uint8_t code;
    std::uint8_t bits;
    std::uint16_t lanes;
  } dtype;
  std::int64_t* shape;
  std::int64_t* strides;
  std::uint64_t byte_offset;
};

struct DtManagedTensor {
  DtTensor dl_tensor;
  void* manager_ctx;
  void (*deleter)(DtManagedTensor*);
};

struct DtManagedTensorVersioned {
  std::uint32_t version_major;
  std::uint32_t version_minor;
  void* manager_ctx;
  void (*deleter)(DtManagedTensorVersioned*);
  std::uint64_t flags;
  DtTensor dl_tensor;
};

static_assert(sizeof(DtTensor) == 48u, "DLTensor must be 48 bytes");
static_assert(sizeof(DtManagedTensor) == 64u, "legacy DLManagedTensor must be 64 bytes");
static_assert(sizeof(DtManagedTensorVersioned) == 80u,
              "versioned DLManagedTensorVersioned must be 80 bytes");

int failures = 0;

void expect(bool condition, const std::string& message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++failures;
  }
}

// Allocate a host arena, bind it, export one float64 slice, and verify the
// managed tensor fields byte-by-byte without importing through a framework.
int test_host_arena_and_versioned_export() {
  xtbloom_result_owner_options_t options;
  xtbloom_status_t status = xtbloom_result_owner_options_init(&options, sizeof(options));
  expect(status == XTBLOOM_STATUS_SUCCESS, "result owner options init");
  expect(options.memory_space == XTBLOOM_MEMORY_HOST, "host default memory space");
  expect(options.device_id == -1, "host default device id");
  options.memory_space = XTBLOOM_MEMORY_HOST;
  options.device_id = -1;
  options.size_bytes = 4096;

  xtbloom_result_owner_t* owner = NULL;
  status = xtbloom_result_owner_create(&options, &owner);
  expect(status == XTBLOOM_STATUS_SUCCESS && owner != NULL, "host arena create");
  if (status != XTBLOOM_STATUS_SUCCESS || owner == NULL) {
    return 1;
  }

  xtbloom_buffer_t buffer;
  status = xtbloom_result_owner_buffer(owner, &buffer);
  expect(status == XTBLOOM_STATUS_SUCCESS, "result owner buffer");
  expect(buffer.data != NULL, "arena data is non-null");
  expect(buffer.size_bytes == 4096u, "arena size is preserved");
  expect(buffer.memory_space == XTBLOOM_MEMORY_HOST, "arena host memory space");
  expect(buffer.reserved == 0u, "arena buffer reserved is zero");

  const int64_t shape[2] = {8, 8};
  xtbloom_dlpack_view_t view;
  std::memset(&view, 0, sizeof(view));
  view.struct_size = sizeof(view);
  view.api_version = XTBLOOM_API_VERSION;
  view.byte_offset = 64;
  view.dtype_code = 2; /* float */
  view.dtype_bits = 64;
  view.dtype_lanes = 1;
  view.ndim = 2;
  view.shape = shape;

  void* managed = reinterpret_cast<void*>(0x1);
  status = xtbloom_result_owner_export_dltensor(owner, &view, 1, &managed);
  expect(status == XTBLOOM_STATUS_SUCCESS, "versioned export succeeds");
  expect(managed != NULL, "versioned export returns a managed tensor");
  if (status != XTBLOOM_STATUS_SUCCESS || managed == NULL) {
    return 1;
  }

  DtManagedTensorVersioned* versioned = static_cast<DtManagedTensorVersioned*>(managed);
  expect(versioned->version_major == 1u, "DLPack major version is 1");
  expect(versioned->version_minor == 0u, "DLPack minor version is 0");
  expect(versioned->flags == 0u, "no read-only/copied flag set");
  expect(versioned->dl_tensor.ndim == 2, "exported ndim is 2");
  expect(versioned->dl_tensor.device.device_type == 1, "host device type");
  expect(versioned->dl_tensor.device.device_id == 0, "host device id is 0");
  expect(versioned->dl_tensor.dtype.code == 2u && versioned->dl_tensor.dtype.bits == 64u &&
             versioned->dl_tensor.dtype.lanes == 1u,
         "exported dtype is float64 scalar");
  expect(versioned->dl_tensor.byte_offset == 0u, "byte_offset folded into data");
  expect(versioned->dl_tensor.data == static_cast<unsigned char*>(buffer.data) + 64,
         "exported data points into the arena slice");
  expect(versioned->dl_tensor.shape[0] == 8 && versioned->dl_tensor.shape[1] == 8,
         "exported shape equals the view");
  expect(versioned->dl_tensor.strides == NULL, "strides NULL means compact row-major");

  // A second export must be an independent single-use managed tensor.
  void* managed2 = NULL;
  status = xtbloom_result_owner_export_dltensor(owner, &view, 1, &managed2);
  expect(status == XTBLOOM_STATUS_SUCCESS && managed2 != NULL && managed2 != managed,
         "repeated export creates a fresh managed tensor");

  // Legacy export mirror.
  void* legacy = NULL;
  status = xtbloom_result_owner_export_dltensor(owner, &view, 0, &legacy);
  expect(status == XTBLOOM_STATUS_SUCCESS && legacy != NULL, "legacy export succeeds");
  if (legacy != NULL) {
    DtManagedTensor* legacy_tensor = static_cast<DtManagedTensor*>(legacy);
    expect(legacy_tensor->dl_tensor.ndim == 2, "legacy ndim is 2");
    expect(legacy_tensor->dl_tensor.device.device_type == 1, "legacy host device type");
    expect(legacy_tensor->manager_ctx != NULL, "legacy manager context is set");
  }

  // Producer close must not invalidate live exported tensors: each export
  // retains the arena independently, and each native deleter releases it
  // exactly once when an importing framework is done.
  xtbloom_result_owner_release(owner);

  // Consume all three exports through their native deleters. Each managed
  // tensor must be freed exactly once with no use-after-free or leak.
  versioned->deleter(versioned);
  versioned = NULL;
  DtManagedTensorVersioned* versioned2 = static_cast<DtManagedTensorVersioned*>(managed2);
  versioned2->deleter(versioned2);
  managed2 = NULL;
  DtManagedTensor* legacy_tensor = static_cast<DtManagedTensor*>(legacy);
  legacy_tensor->deleter(legacy_tensor);
  legacy = NULL;
  return 0;
}

int test_owner_lifetime_and_failures() {
  xtbloom_result_owner_options_t options;
  xtbloom_result_owner_options_init(&options, sizeof(options));
  options.memory_space = XTBLOOM_MEMORY_HOST;
  options.device_id = -1;
  options.size_bytes = 512;

  // Host creation and retain/release accounting.
  xtbloom_result_owner_t* owner = NULL;
  xtbloom_status_t status = xtbloom_result_owner_create(&options, &owner);
  expect(status == XTBLOOM_STATUS_SUCCESS && owner != NULL, "host arena create");
  if (owner == NULL) {
    return 1;
  }
  xtbloom_result_owner_retain(owner);
  xtbloom_result_owner_retain(owner);
  xtbloom_result_owner_release(owner); /* 2 refs remain */
  xtbloom_result_owner_release(owner); /* 1 ref remains */
  xtbloom_result_owner_release(owner); /* freed here */

  // NULL release is a no-op and must not crash.
  xtbloom_result_owner_buffer(NULL, NULL);
  const std::string diagnostic_before_release = xtbloom_get_last_error();
  xtbloom_result_owner_release(NULL);
  expect(std::string(xtbloom_get_last_error()) == diagnostic_before_release,
         "NULL release preserves the previous diagnostic");

  // Zero-size arena is rejected.
  options.size_bytes = 0;
  status = xtbloom_result_owner_create(&options, &owner);
  expect(status == XTBLOOM_STATUS_INVALID_ARGUMENT && owner == NULL, "zero-size arena rejected");
  options.size_bytes = 512;

  // Reserved field must be zero.
  options.reserved = 1;
  status = xtbloom_result_owner_create(&options, &owner);
  expect(status == XTBLOOM_STATUS_INVALID_ARGUMENT && owner == NULL, "nonzero reserved rejected");
  options.reserved = 0;

  // Host arenas use the sentinel device id -1; accepting a CUDA ordinal here
  // would produce a descriptor whose public metadata contradicts its storage.
  options.device_id = 0;
  status = xtbloom_result_owner_create(&options, &owner);
  expect(status == XTBLOOM_STATUS_INVALID_ARGUMENT && owner == NULL,
         "host arena rejects a non-sentinel device id");
  options.device_id = -1;

  owner = NULL;
  status = xtbloom_result_owner_create(&options, &owner);
  if (status != XTBLOOM_STATUS_SUCCESS || owner == NULL) {
    return 1;
  }

  // Export failure paths must leave *out_managed untouched and the arena ref
  // counting intact.
  const int64_t shape[2] = {8, 64}; /* 8*64*8 = 4096 > 512 arena */
  xtbloom_dlpack_view_t view;
  std::memset(&view, 0, sizeof(view));
  view.struct_size = sizeof(view);
  view.api_version = XTBLOOM_API_VERSION;
  view.dtype_code = 2;
  view.dtype_bits = 64;
  view.dtype_lanes = 1;
  view.ndim = 2;
  view.shape = shape;

  void* managed = reinterpret_cast<void*>(0x1234);
  status = xtbloom_result_owner_export_dltensor(owner, &view, 1, &managed);
  expect(status == XTBLOOM_STATUS_INVALID_ARGUMENT && managed == NULL,
         "out-of-range export fails transactionally");

  xtbloom_dlpack_view_t bad_dtype = view;
  bad_dtype.ndim = 1;
  const int64_t scalar_shape[1] = {16};
  bad_dtype.shape = scalar_shape;
  bad_dtype.dtype_code = 3; /* unsupported code */
  managed = reinterpret_cast<void*>(0x1234);
  status = xtbloom_result_owner_export_dltensor(owner, &bad_dtype, 1, &managed);
  expect(status == XTBLOOM_STATUS_INVALID_ARGUMENT && managed == NULL,
         "unsupported dtype rejected transactionally");

  // The public ABI intentionally exposes only scalar integer and float dtypes
  // documented in xtbloom.h.  Do not silently accept wider unsigned or special
  // DLPack encodings merely because they have a representable byte width.
  const int32_t unsupported_codes[][2] = {{1, 16}, {1, 32}, {1, 64}, {4, 16}, {6, 8}};
  for (const auto& dtype : unsupported_codes) {
    xtbloom_dlpack_view_t unsupported = bad_dtype;
    unsupported.dtype_code = dtype[0];
    unsupported.dtype_bits = dtype[1];
    managed = reinterpret_cast<void*>(0x1234);
    status = xtbloom_result_owner_export_dltensor(owner, &unsupported, 1, &managed);
    expect(status == XTBLOOM_STATUS_INVALID_ARGUMENT && managed == NULL,
           "undocumented DLPack dtype rejected transactionally");
  }

  // Empty tensors are valid compact views and must not trigger a zero-divisor
  // while validating subsequent shape extents.
  const int64_t empty_shape[2] = {0, 3};
  xtbloom_dlpack_view_t empty = view;
  empty.shape = empty_shape;
  empty.dtype_code = 2;
  empty.dtype_bits = 64;
  empty.byte_offset = 0;
  managed = NULL;
  status = xtbloom_result_owner_export_dltensor(owner, &empty, 1, &managed);
  expect(status == XTBLOOM_STATUS_SUCCESS && managed != NULL,
         "empty DLPack shape exports successfully");
  if (managed != NULL) {
    DtManagedTensorVersioned* empty_tensor = static_cast<DtManagedTensorVersioned*>(managed);
    expect(empty_tensor->dl_tensor.data == NULL && empty_tensor->dl_tensor.shape[0] == 0 &&
               empty_tensor->dl_tensor.shape[1] == 3,
           "empty DLPack shape publishes null data and copied extents");
    empty_tensor->deleter(empty_tensor);
    managed = NULL;
  }

  // Unaligned offset must be rejected for an 8-byte scalar.
  xtbloom_dlpack_view_t unaligned = view;
  unaligned.ndim = 1;
  unaligned.shape = scalar_shape;
  unaligned.byte_offset = 4;
  managed = reinterpret_cast<void*>(0x1234);
  status = xtbloom_result_owner_export_dltensor(owner, &unaligned, 1, &managed);
  expect(status == XTBLOOM_STATUS_INVALID_ARGUMENT && managed == NULL,
         "unaligned offset rejected transactionally");

  // Negative extents must be rejected.
  xtbloom_dlpack_view_t negative = view;
  negative.ndim = 1;
  const int64_t negative_shape[1] = {-1};
  negative.shape = negative_shape;
  negative.byte_offset = 0;
  managed = reinterpret_cast<void*>(0x1234);
  status = xtbloom_result_owner_export_dltensor(owner, &negative, 1, &managed);
  expect(status == XTBLOOM_STATUS_INVALID_ARGUMENT && managed == NULL,
         "negative extent rejected transactionally");

  // Bad version is rejected before any pointer is touched.
  managed = reinterpret_cast<void*>(0x1234);
  status = xtbloom_result_owner_export_dltensor(owner, &view, 7, &managed);
  expect(status == XTBLOOM_STATUS_INVALID_ARGUMENT && managed == NULL,
         "unknown export version rejected");

  // Buffer output must reflect the arena even with several retained exports.
  xtbloom_buffer_t arena_buffer;
  xtbloom_result_owner_buffer(owner, &arena_buffer);
  expect(arena_buffer.size_bytes == 512u, "buffer reflects arena after exports");

  xtbloom_result_owner_release(owner);
  return 0;
}

}  // namespace

int main() {
  // Short-structure init must fail so a future header suffix stays safe.
  {
    xtbloom_result_owner_options_t options;
    xtbloom_status_t status =
        xtbloom_result_owner_options_init(&options, XTBLOOM_RESULT_OWNER_OPTIONS_V1_SIZE - 1);
    expect(status == XTBLOOM_STATUS_INVALID_ARGUMENT, "short result-owner options rejected");
  }
  {
    xtbloom_dlpack_view_t view;
    xtbloom_status_t status = xtbloom_result_owner_export_dltensor(NULL, &view, 1, NULL);
    expect(status == XTBLOOM_STATUS_INVALID_ARGUMENT, "NULL owner export rejected");
  }

  test_host_arena_and_versioned_export();
  test_owner_lifetime_and_failures();

  if (failures != 0) {
    std::fprintf(stderr, "%d result-owner test failures\n", failures);
    return 1;
  }
  return 0;
}
