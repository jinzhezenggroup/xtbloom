#include <cuda_runtime_api.h>
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>

#include "runtime/cuda_descriptor_validation.hpp"
#include "runtime/gfn2_cuda_topology_staging.hpp"

namespace gpuxtb::detail {
namespace {

using Diagnostic = Gfn2CudaTopologyStagingDiagnostic;
using Error = Gfn2CudaTopologyStagingError;
using Field = Gfn2CudaTopologyStagingField;
using Disposition = Gfn2CudaTopologyStageDisposition;

Diagnostic failure(gpuxtb_status_t status, Error error, Field field, std::int64_t index = -1,
                   cudaError_t cuda_status = cudaSuccess) noexcept {
  Diagnostic result{};
  result.status = status;
  result.error = error;
  result.field = field;
  result.index = index;
  result.cuda_status = static_cast<std::int32_t>(cuda_status);
  return result;
}

Diagnostic cuda_failure(Field field, cudaError_t status) noexcept {
  const gpuxtb_status_t public_status = status == cudaErrorMemoryAllocation
                                            ? GPUXTB_STATUS_ALLOCATION_FAILED
                                            : GPUXTB_STATUS_INTERNAL_ERROR;
  return failure(public_status,
                 status == cudaErrorMemoryAllocation ? Error::kAllocationFailed : Error::kCudaError,
                 field, -1, status);
}

Diagnostic restore_boundary(ScopedCudaDevice& guard, Diagnostic intended,
                            std::string& error) noexcept {
  const gpuxtb_status_t status = guard.restore(error);
  return status == GPUXTB_STATUS_SUCCESS ? intended
                                         : failure(status, Error::kCudaError, Field::kNone);
}

bool checked_add(std::size_t first, std::size_t second, std::size_t& output) noexcept {
  if (first > std::numeric_limits<std::size_t>::max() - second) return false;
  output = first + second;
  return true;
}

bool checked_multiply(std::size_t first, std::size_t second, std::size_t& output) noexcept {
  if (first != 0u && second > std::numeric_limits<std::size_t>::max() / first) return false;
  output = first * second;
  return true;
}

bool align_cursor(std::size_t cursor, std::size_t alignment, std::size_t& output) noexcept {
  const std::size_t remainder = cursor % alignment;
  return remainder == 0u ? (output = cursor, true)
                         : checked_add(cursor, alignment - remainder, output);
}

template <typename T>
bool append_array(std::size_t count, std::size_t& cursor, std::size_t& offset) noexcept {
  std::size_t aligned = 0u;
  std::size_t bytes = 0u;
  if (!align_cursor(cursor, alignof(T), aligned) || !checked_multiply(count, sizeof(T), bytes) ||
      !checked_add(aligned, bytes, cursor)) {
    return false;
  }
  offset = aligned;
  return true;
}

bool to_size(std::int64_t value, std::size_t& output) noexcept {
  if (value < 0 || static_cast<std::uint64_t>(value) >
                       static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  output = static_cast<std::size_t>(value);
  return true;
}

struct PackedLayout {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_point_charges = 0;
  std::size_t atom_offsets = 0u;
  std::size_t atomic_numbers = 0u;
  std::size_t molecular_charges = 0u;
  std::size_t unpaired_electrons = 0u;
  std::size_t spin_channels = 0u;
  std::size_t point_offsets = 0u;
  std::size_t response_offsets = 0u;
  std::size_t bytes = 0u;

  [[nodiscard]] bool same_shape(const PackedLayout& other) const noexcept {
    return batch_size == other.batch_size && total_atoms == other.total_atoms &&
           total_point_charges == other.total_point_charges && bytes == other.bytes;
  }
};

bool make_layout(const gpuxtb_batch_t& batch, PackedLayout& layout) noexcept {
  if (batch.batch_size <= 0 || batch.batch_size == std::numeric_limits<std::int64_t>::max() ||
      batch.total_atoms <= 0 || batch.total_point_charges < 0 ||
      batch.total_charge_response_elements < 0) {
    return false;
  }
  std::size_t batch_count = 0u;
  std::size_t atom_count = 0u;
  std::size_t point_count = 0u;
  if (!to_size(batch.batch_size, batch_count) || !to_size(batch.total_atoms, atom_count) ||
      !to_size(batch.total_point_charges, point_count) || batch_count == SIZE_MAX) {
    return false;
  }
  const std::size_t offset_count = batch_count + 1u;
  std::size_t cursor = 0u;
  PackedLayout created{};
  created.batch_size = batch.batch_size;
  created.total_atoms = batch.total_atoms;
  created.total_point_charges = batch.total_point_charges;
  if (!append_array<std::int64_t>(offset_count, cursor, created.atom_offsets) ||
      !append_array<std::int32_t>(atom_count, cursor, created.atomic_numbers) ||
      !append_array<double>(batch_count, cursor, created.molecular_charges) ||
      !append_array<std::int32_t>(batch_count, cursor, created.unpaired_electrons) ||
      !append_array<std::int32_t>(batch_count, cursor, created.spin_channels) ||
      !append_array<std::int64_t>(offset_count, cursor, created.point_offsets) ||
      !append_array<std::int64_t>(offset_count, cursor, created.response_offsets)) {
    return false;
  }
  created.bytes = cursor;
  layout = created;
  return true;
}

template <typename T>
T* packed_pointer(void* base, std::size_t offset) noexcept {
  return reinterpret_cast<T*>(static_cast<unsigned char*>(base) + offset);
}

template <typename T>
const T* packed_pointer(const void* base, std::size_t offset) noexcept {
  return reinterpret_cast<const T*>(static_cast<const unsigned char*>(base) + offset);
}

struct DeviceKeyView {
  std::int64_t* atom_offsets = nullptr;
  std::int32_t* atomic_numbers = nullptr;
  double* molecular_charges = nullptr;
  std::int32_t* unpaired_electrons = nullptr;
  std::int32_t* spin_channels = nullptr;
  std::int64_t* point_offsets = nullptr;
  std::int64_t* response_offsets = nullptr;
};

DeviceKeyView device_view(void* base, const PackedLayout& layout) noexcept {
  return {packed_pointer<std::int64_t>(base, layout.atom_offsets),
          packed_pointer<std::int32_t>(base, layout.atomic_numbers),
          packed_pointer<double>(base, layout.molecular_charges),
          packed_pointer<std::int32_t>(base, layout.unpaired_electrons),
          packed_pointer<std::int32_t>(base, layout.spin_channels),
          packed_pointer<std::int64_t>(base, layout.point_offsets),
          packed_pointer<std::int64_t>(base, layout.response_offsets)};
}

Gfn2CudaTopologyDeviceKeyIdentity key_identity(void* base, const PackedLayout& layout) noexcept {
  if (base == nullptr) return {};
  const DeviceKeyView view = device_view(base, layout);
  return {reinterpret_cast<std::uintptr_t>(base),
          layout.bytes,
          reinterpret_cast<std::uintptr_t>(view.atom_offsets),
          reinterpret_cast<std::uintptr_t>(view.atomic_numbers),
          reinterpret_cast<std::uintptr_t>(view.molecular_charges),
          reinterpret_cast<std::uintptr_t>(view.unpaired_electrons),
          reinterpret_cast<std::uintptr_t>(view.spin_channels),
          reinterpret_cast<std::uintptr_t>(view.point_offsets),
          reinterpret_cast<std::uintptr_t>(view.response_offsets)};
}

/* The device writes exactly this small record for the ordinary fixed-topology path. */
struct DeviceReport {
  std::uint32_t error = 0u;
  std::uint32_t field = 0u;
  std::int64_t index = -1;
  std::uint8_t matches_committed = 0u;
  std::uint8_t reserved[7]{};
  std::int64_t normalized_response_elements = 0;
};

enum class DeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidMetadata = 1u,
  kCountOverflow = 2u,
};

__device__ void set_device_failure(DeviceReport* report, DeviceError error, Field field,
                                   std::int64_t index) {
  report->error = static_cast<std::uint32_t>(error);
  report->field = static_cast<std::uint32_t>(field);
  report->index = index;
  report->matches_committed = 0u;
}

template <typename T>
__device__ bool bitwise_equal(const T* first, const T* second, std::int64_t count) {
  for (std::int64_t index = 0; index < count; ++index) {
    if (first[index] != second[index]) return false;
  }
  return true;
}

template <>
__device__ bool bitwise_equal<double>(const double* first, const double* second,
                                      std::int64_t count) {
  for (std::int64_t index = 0; index < count; ++index) {
    if (__double_as_longlong(first[index]) != __double_as_longlong(second[index])) return false;
  }
  return true;
}

__global__ void validate_and_compare_topology_kernel(
    DeviceKeyView candidate, DeviceKeyView committed, std::int64_t batch_size,
    std::int64_t total_atoms, std::int64_t total_point_charges,
    std::int64_t declared_response_elements, bool point_offsets_supplied,
    bool response_offsets_supplied, bool spin_channels_supplied, bool compare_committed,
    DeviceReport* report) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  *report = {};
  report->index = -1;

  if (candidate.atom_offsets[0] != 0) {
    set_device_failure(report, DeviceError::kInvalidMetadata, Field::kAtomOffsets, 0);
    return;
  }
  if (candidate.atom_offsets[batch_size] != total_atoms) {
    set_device_failure(report, DeviceError::kInvalidMetadata, Field::kAtomOffsets, batch_size);
    return;
  }

  std::int64_t response_prefix = 0;
  if (!response_offsets_supplied) candidate.response_offsets[0] = 0;
  if (response_offsets_supplied && candidate.response_offsets[0] != 0) {
    set_device_failure(report, DeviceError::kInvalidMetadata, Field::kChargeResponseOffsets, 0);
    return;
  }
  for (std::int64_t system = 0; system < batch_size; ++system) {
    const std::int64_t begin = candidate.atom_offsets[system];
    const std::int64_t end = candidate.atom_offsets[system + 1];
    if (begin < 0 || end <= begin || end > total_atoms) {
      set_device_failure(report, DeviceError::kInvalidMetadata, Field::kAtomOffsets, system);
      return;
    }
    const std::int64_t atoms = end - begin;
    if (atoms > INT64_MAX / atoms) {
      set_device_failure(report, DeviceError::kCountOverflow, Field::kChargeResponseOffsets,
                         system);
      return;
    }
    const std::int64_t square = atoms * atoms;
    if (square > INT64_MAX - response_prefix) {
      set_device_failure(report, DeviceError::kCountOverflow, Field::kChargeResponseOffsets,
                         system);
      return;
    }
    const std::int64_t next = response_prefix + square;
    if (response_offsets_supplied && (candidate.response_offsets[system] != response_prefix ||
                                      candidate.response_offsets[system + 1] != next)) {
      set_device_failure(report, DeviceError::kInvalidMetadata, Field::kChargeResponseOffsets,
                         system);
      return;
    }
    if (!response_offsets_supplied) candidate.response_offsets[system + 1] = next;
    response_prefix = next;
  }
  if (response_offsets_supplied && declared_response_elements != response_prefix) {
    set_device_failure(report, DeviceError::kInvalidMetadata, Field::kChargeResponseOffsets,
                       batch_size);
    return;
  }
  if (!response_offsets_supplied && declared_response_elements != 0) {
    set_device_failure(report, DeviceError::kInvalidMetadata, Field::kChargeResponseOffsets,
                       batch_size);
    return;
  }
  report->normalized_response_elements = response_prefix;

  for (std::int64_t atom = 0; atom < total_atoms; ++atom) {
    const std::int32_t atomic_number = candidate.atomic_numbers[atom];
    if (atomic_number < 1 || atomic_number > 118) {
      set_device_failure(report, DeviceError::kInvalidMetadata, Field::kAtomicNumbers, atom);
      return;
    }
  }
  for (std::int64_t system = 0; system < batch_size; ++system) {
    if (!isfinite(candidate.molecular_charges[system])) {
      set_device_failure(report, DeviceError::kInvalidMetadata, Field::kMolecularCharges, system);
      return;
    }
    /* +0.0 and -0.0 are the same molecular charge in the public contract.
     * Canonicalize before the bitwise device-key comparison so alternating
     * signed-zero inputs cannot manufacture spurious topology candidates. */
    if (candidate.molecular_charges[system] == 0.0) {
      candidate.molecular_charges[system] = 0.0;
    }
    if (!spin_channels_supplied) candidate.spin_channels[system] = 1;
    if (candidate.spin_channels[system] != 1 && candidate.spin_channels[system] != 2) {
      set_device_failure(report, DeviceError::kInvalidMetadata, Field::kSpinChannels, system);
      return;
    }
  }

  if (!point_offsets_supplied) {
    for (std::int64_t index = 0; index <= batch_size; ++index) {
      candidate.point_offsets[index] = 0;
    }
  } else {
    if (candidate.point_offsets[0] != 0) {
      set_device_failure(report, DeviceError::kInvalidMetadata, Field::kPointChargeOffsets, 0);
      return;
    }
    for (std::int64_t system = 0; system < batch_size; ++system) {
      const std::int64_t begin = candidate.point_offsets[system];
      const std::int64_t end = candidate.point_offsets[system + 1];
      if (begin < 0 || end < begin || end > total_point_charges) {
        set_device_failure(report, DeviceError::kInvalidMetadata, Field::kPointChargeOffsets,
                           system);
        return;
      }
    }
    if (candidate.point_offsets[batch_size] != total_point_charges) {
      set_device_failure(report, DeviceError::kInvalidMetadata, Field::kPointChargeOffsets,
                         batch_size);
      return;
    }
  }

  bool equal = compare_committed;
  if (equal) {
    equal = bitwise_equal(candidate.atom_offsets, committed.atom_offsets, batch_size + 1) &&
            bitwise_equal(candidate.atomic_numbers, committed.atomic_numbers, total_atoms) &&
            bitwise_equal(candidate.molecular_charges, committed.molecular_charges, batch_size) &&
            bitwise_equal(candidate.unpaired_electrons, committed.unpaired_electrons, batch_size) &&
            bitwise_equal(candidate.spin_channels, committed.spin_channels, batch_size) &&
            bitwise_equal(candidate.point_offsets, committed.point_offsets, batch_size + 1) &&
            bitwise_equal(candidate.response_offsets, committed.response_offsets, batch_size + 1);
  }
  report->matches_committed = equal ? 1u : 0u;
}

struct Backing {
  std::int32_t device_id = -1;
  PackedLayout layout{};
  void* staging = nullptr;
  void* canonical = nullptr;
  void* pinned = nullptr;

  Backing() = default;
  Backing(const Backing&) = delete;
  Backing& operator=(const Backing&) = delete;
  ~Backing() { release(); }

  void release() noexcept {
    int previous = -1;
    const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
    if (device_id >= 0 && (!have_previous || previous != device_id)) (void)cudaSetDevice(device_id);
    if (staging != nullptr) (void)cudaFree(staging);
    if (canonical != nullptr) (void)cudaFree(canonical);
    if (pinned != nullptr) (void)cudaFreeHost(pinned);
    if (have_previous && previous != device_id) (void)cudaSetDevice(previous);
    staging = nullptr;
    canonical = nullptr;
    pinned = nullptr;
  }
};

Diagnostic allocate_backing(std::int32_t device_id, const PackedLayout& layout,
                            std::unique_ptr<Backing>& output) noexcept {
  std::unique_ptr<Backing> created(new (std::nothrow) Backing());
  if (!created) {
    return failure(GPUXTB_STATUS_ALLOCATION_FAILED, Error::kAllocationFailed, Field::kArena);
  }
  created->device_id = device_id;
  created->layout = layout;
  cudaError_t status = cudaMalloc(&created->staging, layout.bytes);
  if (status == cudaSuccess) status = cudaMalloc(&created->canonical, layout.bytes);
  if (status == cudaSuccess) status = cudaMallocHost(&created->pinned, layout.bytes);
  if (status != cudaSuccess) return cuda_failure(Field::kArena, status);
  output = std::move(created);
  return {};
}

std::size_t field_bytes(const PackedLayout& layout, Field field) noexcept {
  const std::size_t batch = static_cast<std::size_t>(layout.batch_size);
  const std::size_t atoms = static_cast<std::size_t>(layout.total_atoms);
  switch (field) {
    case Field::kAtomOffsets:
    case Field::kPointChargeOffsets:
    case Field::kChargeResponseOffsets:
      return (batch + 1u) * sizeof(std::int64_t);
    case Field::kAtomicNumbers:
      return atoms * sizeof(std::int32_t);
    case Field::kMolecularCharges:
      return batch * sizeof(double);
    case Field::kUnpairedElectrons:
    case Field::kSpinChannels:
      return batch * sizeof(std::int32_t);
    default:
      return 0u;
  }
}

std::size_t field_offset(const PackedLayout& layout, Field field) noexcept {
  switch (field) {
    case Field::kAtomOffsets:
      return layout.atom_offsets;
    case Field::kAtomicNumbers:
      return layout.atomic_numbers;
    case Field::kMolecularCharges:
      return layout.molecular_charges;
    case Field::kUnpairedElectrons:
      return layout.unpaired_electrons;
    case Field::kSpinChannels:
      return layout.spin_channels;
    case Field::kPointChargeOffsets:
      return layout.point_offsets;
    case Field::kChargeResponseOffsets:
      return layout.response_offsets;
    default:
      return 0u;
  }
}

const char* field_name(Field field) noexcept {
  switch (field) {
    case Field::kAtomOffsets:
      return "atom_offsets";
    case Field::kAtomicNumbers:
      return "atomic_numbers";
    case Field::kMolecularCharges:
      return "molecular_charges";
    case Field::kUnpairedElectrons:
      return "unpaired_electrons";
    case Field::kSpinChannels:
      return "spin_channels";
    case Field::kPointChargeOffsets:
      return "point_charge_offsets";
    case Field::kChargeResponseOffsets:
      return "charge_response_offsets";
    default:
      return "topology metadata";
  }
}

const gpuxtb_const_buffer_t& field_buffer(const gpuxtb_batch_t& batch, Field field) noexcept {
  switch (field) {
    case Field::kAtomOffsets:
      return batch.atom_offsets;
    case Field::kAtomicNumbers:
      return batch.atomic_numbers;
    case Field::kMolecularCharges:
      return batch.molecular_charges;
    case Field::kUnpairedElectrons:
      return batch.unpaired_electrons;
    case Field::kSpinChannels:
      return batch.spin_channels;
    case Field::kPointChargeOffsets:
      return batch.point_charge_offsets;
    case Field::kChargeResponseOffsets:
      return batch.charge_response_offsets;
    default:
      return batch.atom_offsets;
  }
}

std::size_t field_alignment(Field field) noexcept {
  switch (field) {
    case Field::kAtomOffsets:
    case Field::kPointChargeOffsets:
    case Field::kChargeResponseOffsets:
      return alignof(std::int64_t);
    case Field::kMolecularCharges:
      return alignof(double);
    default:
      return alignof(std::int32_t);
  }
}

Diagnostic diagnostic_from_device_report(const DeviceReport& report) noexcept {
  if (report.error == static_cast<std::uint32_t>(DeviceError::kSuccess)) return {};
  const Field field = static_cast<Field>(report.field);
  if (report.error == static_cast<std::uint32_t>(DeviceError::kCountOverflow)) {
    return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Error::kCountOverflow, field, report.index);
  }
  return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Error::kInvalidMetadata, field, report.index);
}

void set_semantic_error(const Diagnostic& diagnostic, std::string& error) {
  if (diagnostic.error == Error::kCountOverflow) {
    error = "the dense charge-response topology overflows int64_t";
  } else if (diagnostic.field == Field::kSpinChannels) {
    error = "spin_channels values must be one or two";
  } else {
    error = std::string(field_name(diagnostic.field)) + " contains invalid topology metadata";
  }
}

bool copy_snapshot(const Backing& backing, const DeviceReport& report, bool periodic_enabled,
                   Gfn2CudaTopologyHostSnapshot& snapshot) {
  const PackedLayout& layout = backing.layout;
  const std::size_t batch = static_cast<std::size_t>(layout.batch_size);
  const std::size_t atoms = static_cast<std::size_t>(layout.total_atoms);
  Gfn2CudaTopologyHostSnapshot created{};
  created.batch_size = layout.batch_size;
  created.total_atoms = layout.total_atoms;
  created.total_point_charges = layout.total_point_charges;
  created.total_charge_response_elements = report.normalized_response_elements;
  created.periodic_enabled = periodic_enabled;
  created.atom_offsets.assign(
      packed_pointer<const std::int64_t>(backing.pinned, layout.atom_offsets),
      packed_pointer<const std::int64_t>(backing.pinned, layout.atom_offsets) + batch + 1u);
  created.atomic_numbers.assign(
      packed_pointer<const std::int32_t>(backing.pinned, layout.atomic_numbers),
      packed_pointer<const std::int32_t>(backing.pinned, layout.atomic_numbers) + atoms);
  created.molecular_charges.assign(
      packed_pointer<const double>(backing.pinned, layout.molecular_charges),
      packed_pointer<const double>(backing.pinned, layout.molecular_charges) + batch);
  created.unpaired_electrons.assign(
      packed_pointer<const std::int32_t>(backing.pinned, layout.unpaired_electrons),
      packed_pointer<const std::int32_t>(backing.pinned, layout.unpaired_electrons) + batch);
  created.spin_channels.assign(
      packed_pointer<const std::int32_t>(backing.pinned, layout.spin_channels),
      packed_pointer<const std::int32_t>(backing.pinned, layout.spin_channels) + batch);
  created.point_charge_offsets.assign(
      packed_pointer<const std::int64_t>(backing.pinned, layout.point_offsets),
      packed_pointer<const std::int64_t>(backing.pinned, layout.point_offsets) + batch + 1u);
  created.charge_response_offsets.assign(
      packed_pointer<const std::int64_t>(backing.pinned, layout.response_offsets),
      packed_pointer<const std::int64_t>(backing.pinned, layout.response_offsets) + batch + 1u);
  snapshot = std::move(created);
  return true;
}

}  // namespace

struct Gfn2CudaTopologyStaging::Impl {
  std::int32_t device_id = -1;
  cudaStream_t stream = nullptr;
  cudaEvent_t event = nullptr;
  DeviceReport* device_report = nullptr;
  DeviceReport* pinned_report = nullptr;
  bool initialized = false;
  gpuxtb_status_t initialization_status = GPUXTB_STATUS_BACKEND_UNAVAILABLE;
  std::string initialization_error;
  std::unique_ptr<Backing> active;
  std::unique_ptr<Backing> pending;
  std::unique_ptr<Gfn2CudaTopologyHostSnapshot> committed_snapshot;
  std::unique_ptr<Gfn2CudaTopologyHostSnapshot> pending_snapshot;
  bool pending_ready = false;
  std::uint64_t committed_generation = 0u;
  std::uint64_t stage_submissions = 0u;
  std::uint64_t report_downloads = 0u;
  std::uint64_t full_metadata_downloads = 0u;

  Diagnostic wait_for_event(Field field, std::string& error) noexcept {
    cudaError_t status = cudaEventRecord(event, stream);
    if (status == cudaSuccess) status = cudaEventSynchronize(event);
    if (status != cudaSuccess) {
      error = std::string("CUDA topology staging event failed: ") + cudaGetErrorString(status);
      return cuda_failure(field, status);
    }
    return {};
  }

  /*
   * An enqueue can fail after earlier HOST-backed copies were accepted.  Drain
   * precisely that stream prefix before returning so a later call cannot
   * overwrite pinned staging while CUDA still reads it.
   */
  Diagnostic settle_submission_failure(Diagnostic original, std::string& error) noexcept {
    const std::string original_error = error;
    Diagnostic settled = wait_for_event(Field::kEvent, error);
    if (!settled.success()) return settled;
    error = original_error;
    return original;
  }
};

Gfn2CudaTopologyStaging::Gfn2CudaTopologyStaging(std::int32_t device_id, void* stream) noexcept
    : impl_(new (std::nothrow) Impl()) {
  if (!impl_) return;
  impl_->device_id = device_id;
  impl_->stream = reinterpret_cast<cudaStream_t>(stream);
  std::string error;
  ScopedCudaDevice guard(device_id, error);
  if (!guard.ok()) {
    impl_->initialization_status = guard.status();
    impl_->initialization_error = std::move(error);
    return;
  }
  gpuxtb_status_t status = validate_cuda_stream_owner(device_id, impl_->stream, true, error);
  if (status == GPUXTB_STATUS_SUCCESS) {
    cudaError_t cuda_status = cudaEventCreateWithFlags(&impl_->event, cudaEventDisableTiming);
    if (cuda_status == cudaSuccess) {
      cuda_status =
          cudaMalloc(reinterpret_cast<void**>(&impl_->device_report), sizeof(DeviceReport));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status =
          cudaMallocHost(reinterpret_cast<void**>(&impl_->pinned_report), sizeof(DeviceReport));
    }
    if (cuda_status != cudaSuccess) {
      status = cuda_status == cudaErrorMemoryAllocation ? GPUXTB_STATUS_ALLOCATION_FAILED
                                                        : GPUXTB_STATUS_INTERNAL_ERROR;
      error = std::string("failed to allocate CUDA topology staging diagnostics: ") +
              cudaGetErrorString(cuda_status);
    }
  }
  const gpuxtb_status_t restore_status = guard.restore(error);
  if (status == GPUXTB_STATUS_SUCCESS) status = restore_status;
  impl_->initialization_status = status;
  impl_->initialization_error = error;
  impl_->initialized = status == GPUXTB_STATUS_SUCCESS;
}

Gfn2CudaTopologyStaging::~Gfn2CudaTopologyStaging() {
  if (!impl_) return;
  impl_->pending.reset();
  impl_->active.reset();
  int previous = -1;
  const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
  if (impl_->device_id >= 0 && (!have_previous || previous != impl_->device_id)) {
    (void)cudaSetDevice(impl_->device_id);
  }
  if (impl_->device_report != nullptr) (void)cudaFree(impl_->device_report);
  if (impl_->pinned_report != nullptr) (void)cudaFreeHost(impl_->pinned_report);
  if (impl_->event != nullptr) (void)cudaEventDestroy(impl_->event);
  if (have_previous && previous != impl_->device_id) (void)cudaSetDevice(previous);
}

bool Gfn2CudaTopologyStaging::valid() const noexcept { return impl_ && impl_->initialized; }

Gfn2CudaTopologyStagingDiagnostic Gfn2CudaTopologyStaging::stage_and_validate(
    const gpuxtb_batch_t& batch, std::string& error) {
  error.clear();
  if (!impl_ || !impl_->initialized) {
    if (impl_) error = impl_->initialization_error;
    return failure(impl_ ? impl_->initialization_status : GPUXTB_STATUS_ALLOCATION_FAILED,
                   Error::kNotInitialized, Field::kNone);
  }
  if (impl_->pending) {
    error = "a CUDA topology candidate is already pending commit or abort";
    return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Error::kCandidatePending, Field::kNone);
  }

  ScopedCudaDevice guard(impl_->device_id, error);
  if (!guard.ok()) return failure(guard.status(), Error::kCudaError, Field::kNone);
  gpuxtb_status_t status = validate_cuda_stream_owner(impl_->device_id, impl_->stream, true, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return restore_boundary(guard, failure(status, Error::kInvalidDescriptor, Field::kStream),
                            error);
  }

  PackedLayout layout{};
  if (!make_layout(batch, layout)) {
    error = "CUDA topology dimensions are invalid or overflow host address space";
    return restore_boundary(
        guard,
        failure(GPUXTB_STATUS_INVALID_ARGUMENT, Error::kInvalidDimensions, Field::kDimensions),
        error);
  }
  const bool point_supplied = batch.total_point_charges != 0 ||
                              batch.point_charge_offsets.data != nullptr ||
                              batch.point_charge_offsets.size_bytes != 0u;
  const bool response_supplied =
      batch.total_charge_response_elements != 0 || batch.charge_response_offsets.data != nullptr ||
      batch.charge_response_offsets.size_bytes != 0u ||
      batch.charge_response_matrix.data != nullptr || batch.charge_response_matrix.size_bytes != 0u;
  const bool periodic_enabled = batch.atomic_potential_shifts.data != nullptr ||
                                batch.atomic_potential_shifts.size_bytes != 0u || response_supplied;
  /* The ABI-v2 suffix is optional. Avoid reading it for an ABI-v1 caller and
   * let the validation kernel materialize the canonical restricted default. */
  const bool spin_channels_supplied =
      batch.struct_size >= GPUXTB_BATCH_V2_SIZE &&
      (batch.spin_channels.data != nullptr || batch.spin_channels.size_bytes != 0u);

  struct Input {
    Field field = Field::kNone;
    CudaValidatedConstBuffer buffer{};
  };
  Input inputs[] = {{Field::kAtomOffsets, {}},          {Field::kAtomicNumbers, {}},
                    {Field::kMolecularCharges, {}},     {Field::kUnpairedElectrons, {}},
                    {Field::kSpinChannels, {}},         {Field::kPointChargeOffsets, {}},
                    {Field::kChargeResponseOffsets, {}}};
  for (std::size_t index = 0u; index < 7u; ++index) {
    Input& input = inputs[index];
    if (input.field == Field::kSpinChannels && !spin_channels_supplied) continue;
    if (input.field == Field::kPointChargeOffsets && !point_supplied) continue;
    if (input.field == Field::kChargeResponseOffsets && !response_supplied) continue;
    status = validate_cuda_const_buffer(
        impl_->device_id, field_name(input.field), field_buffer(batch, input.field),
        field_bytes(layout, input.field), field_alignment(input.field),
        CudaManagedMemoryPolicy::kReject, input.buffer, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return restore_boundary(guard, failure(status, Error::kInvalidDescriptor, input.field),
                              error);
    }
  }
  const bool same_layout = impl_->active && impl_->active->layout.same_shape(layout);
  std::unique_ptr<Backing> created;
  Backing* workspace = nullptr;
  if (same_layout) {
    workspace = impl_->active.get();
  } else {
    Diagnostic allocated = allocate_backing(impl_->device_id, layout, created);
    if (!allocated.success()) {
      error = "failed to allocate packed CUDA topology staging storage";
      return restore_boundary(guard, allocated, error);
    }
    workspace = created.get();
  }

  /* Canonicalize alignment padding as well as fields.  The kernel compares
   * typed leaves only, but whole-arena transactional copies and diagnostics
   * must never propagate indeterminate bytes. */
  cudaError_t cuda_status = cudaMemsetAsync(workspace->staging, 0, layout.bytes, impl_->stream);
  if (cuda_status != cudaSuccess) {
    error = std::string("failed to initialize packed CUDA topology staging: ") +
            cudaGetErrorString(cuda_status);
    return restore_boundary(guard, cuda_failure(Field::kArena, cuda_status), error);
  }
  for (const Input& input : inputs) {
    if (input.field == Field::kSpinChannels && !spin_channels_supplied) continue;
    if (input.field == Field::kPointChargeOffsets && !point_supplied) continue;
    if (input.field == Field::kChargeResponseOffsets && !response_supplied) continue;
    const std::size_t bytes = field_bytes(layout, input.field);
    const std::size_t offset = field_offset(layout, input.field);
    void* destination = static_cast<unsigned char*>(workspace->staging) + offset;
    cuda_status = cudaSuccess;
    if (input.buffer.memory_space == GPUXTB_MEMORY_HOST) {
      void* pinned = static_cast<unsigned char*>(workspace->pinned) + offset;
      std::memcpy(pinned, input.buffer.data, bytes);
      cuda_status =
          cudaMemcpyAsync(destination, pinned, bytes, cudaMemcpyHostToDevice, impl_->stream);
    } else {
      cuda_status = cudaMemcpyAsync(destination, input.buffer.data, bytes, cudaMemcpyDeviceToDevice,
                                    impl_->stream);
    }
    if (cuda_status != cudaSuccess) {
      error = std::string("failed to stage ") + field_name(input.field) + ": " +
              cudaGetErrorString(cuda_status);
      Diagnostic failed =
          impl_->settle_submission_failure(cuda_failure(input.field, cuda_status), error);
      return restore_boundary(guard, failed, error);
    }
  }

  const DeviceKeyView candidate = device_view(workspace->staging, layout);
  const DeviceKeyView committed =
      same_layout ? device_view(impl_->active->canonical, layout) : DeviceKeyView{};
  validate_and_compare_topology_kernel<<<1, 1, 0, impl_->stream>>>(
      candidate, committed, layout.batch_size, layout.total_atoms, layout.total_point_charges,
      batch.total_charge_response_elements, point_supplied, response_supplied,
      spin_channels_supplied, same_layout, impl_->device_report);
  cuda_status = cudaGetLastError();
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaMemcpyAsync(impl_->pinned_report, impl_->device_report, sizeof(DeviceReport),
                                  cudaMemcpyDeviceToHost, impl_->stream);
  }
  if (cuda_status != cudaSuccess) {
    error = std::string("failed to launch CUDA topology validation: ") +
            cudaGetErrorString(cuda_status);
    Diagnostic failed =
        impl_->settle_submission_failure(cuda_failure(Field::kArena, cuda_status), error);
    return restore_boundary(guard, failed, error);
  }
  ++impl_->stage_submissions;
  ++impl_->report_downloads;
  Diagnostic waited = impl_->wait_for_event(Field::kEvent, error);
  if (!waited.success()) {
    return restore_boundary(guard, waited, error);
  }
  Diagnostic semantic = diagnostic_from_device_report(*impl_->pinned_report);
  if (!semantic.success()) {
    set_semantic_error(semantic, error);
    return restore_boundary(guard, semantic, error);
  }

  const bool metadata_match = same_layout && impl_->pinned_report->matches_committed != 0u &&
                              impl_->committed_snapshot &&
                              impl_->committed_snapshot->periodic_enabled == periodic_enabled;
  if (metadata_match) {
    const gpuxtb_status_t restore_status = guard.restore(error);
    if (restore_status != GPUXTB_STATUS_SUCCESS) {
      return failure(restore_status, Error::kCudaError, Field::kNone);
    }
    Diagnostic result{};
    result.disposition = Disposition::kMatchesCommitted;
    return result;
  }

  if (same_layout) {
    Diagnostic allocated = allocate_backing(impl_->device_id, layout, created);
    if (!allocated.success()) {
      error = "failed to allocate a transactional CUDA topology candidate";
      return restore_boundary(guard, allocated, error);
    }
    cuda_status = cudaMemcpyAsync(created->staging, workspace->staging, layout.bytes,
                                  cudaMemcpyDeviceToDevice, impl_->stream);
    workspace = created.get();
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaMemcpyAsync(workspace->pinned, workspace->staging, layout.bytes,
                                  cudaMemcpyDeviceToHost, impl_->stream);
  }
  if (cuda_status != cudaSuccess) {
    error = std::string("failed to download the changed CUDA topology candidate: ") +
            cudaGetErrorString(cuda_status);
    Diagnostic failed =
        impl_->settle_submission_failure(cuda_failure(Field::kArena, cuda_status), error);
    return restore_boundary(guard, failed, error);
  }
  waited = impl_->wait_for_event(Field::kEvent, error);
  if (!waited.success()) {
    return restore_boundary(guard, waited, error);
  }
  ++impl_->full_metadata_downloads;

  std::unique_ptr<Gfn2CudaTopologyHostSnapshot> snapshot(new (std::nothrow)
                                                             Gfn2CudaTopologyHostSnapshot());
  if (!snapshot) {
    error = "failed to allocate the canonical host topology snapshot";
    return restore_boundary(
        guard, failure(GPUXTB_STATUS_ALLOCATION_FAILED, Error::kAllocationFailed, Field::kArena),
        error);
  }
  try {
    copy_snapshot(*workspace, *impl_->pinned_report, periodic_enabled, *snapshot);
  } catch (const std::bad_alloc&) {
    error = "failed to allocate arrays for the canonical host topology snapshot";
    return restore_boundary(
        guard, failure(GPUXTB_STATUS_ALLOCATION_FAILED, Error::kAllocationFailed, Field::kArena),
        error);
  }
  impl_->pending = std::move(created);
  impl_->pending_snapshot = std::move(snapshot);
  impl_->pending_ready = false;
  const gpuxtb_status_t restore_status = guard.restore(error);
  if (restore_status != GPUXTB_STATUS_SUCCESS) {
    impl_->pending.reset();
    impl_->pending_snapshot.reset();
    return failure(restore_status, Error::kCudaError, Field::kNone);
  }
  Diagnostic result{};
  result.disposition = Disposition::kCandidate;
  return result;
}

Gfn2CudaTopologyStagingDiagnostic Gfn2CudaTopologyStaging::prepare_candidate_commit(
    std::string& error) {
  error.clear();
  if (!impl_ || !impl_->initialized) {
    if (impl_) error = impl_->initialization_error;
    return failure(impl_ ? impl_->initialization_status : GPUXTB_STATUS_ALLOCATION_FAILED,
                   Error::kNotInitialized, Field::kNone);
  }
  if (!impl_->pending || !impl_->pending_snapshot) {
    error = "no CUDA topology candidate is pending";
    return failure(GPUXTB_STATUS_INVALID_ARGUMENT, Error::kNoCandidate, Field::kNone);
  }
  if (impl_->pending_ready) return {};
  ScopedCudaDevice guard(impl_->device_id, error);
  if (!guard.ok()) return failure(guard.status(), Error::kCudaError, Field::kNone);
  gpuxtb_status_t status = validate_cuda_stream_owner(impl_->device_id, impl_->stream, true, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return restore_boundary(guard, failure(status, Error::kInvalidDescriptor, Field::kStream),
                            error);
  }
  cudaError_t cuda_status =
      cudaMemcpyAsync(impl_->pending->canonical, impl_->pending->staging,
                      impl_->pending->layout.bytes, cudaMemcpyDeviceToDevice, impl_->stream);
  if (cuda_status != cudaSuccess) {
    error = std::string("failed to publish the CUDA topology candidate: ") +
            cudaGetErrorString(cuda_status);
    return restore_boundary(guard, cuda_failure(Field::kArena, cuda_status), error);
  }
  Diagnostic waited = impl_->wait_for_event(Field::kEvent, error);
  if (!waited.success()) {
    return restore_boundary(guard, waited, error);
  }
  const gpuxtb_status_t restore_status = guard.restore(error);
  if (restore_status != GPUXTB_STATUS_SUCCESS) {
    return failure(restore_status, Error::kCudaError, Field::kNone);
  }
  impl_->pending_ready = true;
  return {};
}

bool Gfn2CudaTopologyStaging::candidate_publishable() const noexcept {
  return impl_ && impl_->pending_ready && impl_->pending && impl_->pending_snapshot;
}

bool Gfn2CudaTopologyStaging::publish_candidate() noexcept {
  if (!candidate_publishable()) return false;
  impl_->active = std::move(impl_->pending);
  impl_->committed_snapshot = std::move(impl_->pending_snapshot);
  impl_->pending_ready = false;
  ++impl_->committed_generation;
  return true;
}

Gfn2CudaTopologyStagingDiagnostic Gfn2CudaTopologyStaging::commit_candidate(std::string& error) {
  Gfn2CudaTopologyStagingDiagnostic diagnostic = prepare_candidate_commit(error);
  if (!diagnostic.success()) return diagnostic;
  if (!publish_candidate()) {
    error = "prepared CUDA topology candidate violated the publication invariant";
    return failure(GPUXTB_STATUS_INTERNAL_ERROR, Error::kNoCandidate, Field::kNone);
  }
  return {};
}

void Gfn2CudaTopologyStaging::abort_candidate() noexcept {
  if (!impl_) return;
  impl_->pending_snapshot.reset();
  impl_->pending.reset();
  impl_->pending_ready = false;
}

const Gfn2CudaTopologyHostSnapshot* Gfn2CudaTopologyStaging::committed_snapshot() const noexcept {
  return impl_ ? impl_->committed_snapshot.get() : nullptr;
}

const Gfn2CudaTopologyHostSnapshot* Gfn2CudaTopologyStaging::candidate_snapshot() const noexcept {
  return impl_ ? impl_->pending_snapshot.get() : nullptr;
}

Gfn2CudaTopologyDeviceKeyIdentity Gfn2CudaTopologyStaging::committed_device_key() const noexcept {
  return impl_ && impl_->active ? key_identity(impl_->active->canonical, impl_->active->layout)
                                : Gfn2CudaTopologyDeviceKeyIdentity{};
}

Gfn2CudaTopologyDeviceKeyIdentity Gfn2CudaTopologyStaging::candidate_device_key() const noexcept {
  return impl_ && impl_->pending ? key_identity(impl_->pending->staging, impl_->pending->layout)
                                 : Gfn2CudaTopologyDeviceKeyIdentity{};
}

Gfn2CudaTopologyStagingIdentity Gfn2CudaTopologyStaging::identity() const noexcept {
  Gfn2CudaTopologyStagingIdentity result{};
  if (!impl_) return result;
  result.device_id = impl_->device_id;
  result.stream = reinterpret_cast<std::uintptr_t>(impl_->stream);
  result.completion_event = reinterpret_cast<std::uintptr_t>(impl_->event);
  result.pinned_report = reinterpret_cast<std::uintptr_t>(impl_->pinned_report);
  result.committed_owner = reinterpret_cast<std::uintptr_t>(impl_->active.get());
  result.pending_owner = reinterpret_cast<std::uintptr_t>(impl_->pending.get());
  if (impl_->active) {
    result.staging = key_identity(impl_->active->staging, impl_->active->layout);
    result.committed = key_identity(impl_->active->canonical, impl_->active->layout);
    result.pinned_staging = reinterpret_cast<std::uintptr_t>(impl_->active->pinned);
    result.pinned_staging_bytes = impl_->active->layout.bytes;
  }
  if (impl_->pending)
    result.pending = key_identity(impl_->pending->staging, impl_->pending->layout);
  result.committed_generation = impl_->committed_generation;
  result.stage_submissions = impl_->stage_submissions;
  result.report_downloads = impl_->report_downloads;
  result.full_metadata_downloads = impl_->full_metadata_downloads;
  result.completion_event_disable_timing = impl_->event != nullptr ? 1u : 0u;
  result.candidate_pending = impl_->pending ? 1u : 0u;
  return result;
}

}  // namespace gpuxtb::detail
