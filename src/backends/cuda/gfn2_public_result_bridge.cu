#include <cuda_runtime.h>
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_public_result_bridge.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaximumCopyBlocks = 256;
constexpr std::uint32_t kKnownProperties =
    static_cast<std::uint32_t>(GPUXTB_COMPUTE_ENERGY) |
    static_cast<std::uint32_t>(GPUXTB_COMPUTE_FORCES) |
    static_cast<std::uint32_t>(GPUXTB_COMPUTE_ATOMIC_CHARGES) |
    static_cast<std::uint32_t>(GPUXTB_COMPUTE_POINT_CHARGE_FORCES);
constexpr std::uint32_t kKnownResultFlags =
    static_cast<std::uint32_t>(GPUXTB_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES);

using BridgeError = Gfn2PublicResultBridgeError;
using Route = Gfn2PublicResultRoute;

struct ExpectedExtents {
  std::int64_t energies = 0;
  std::int64_t qm_forces = 0;
  std::int64_t atomic_charges = 0;
  std::int64_t point_forces = 0;
  std::int64_t diagnostics = 0;
};

__host__ __device__ bool property_requested(const Gfn2PublicResultBridgeDevicePlan& plan,
                                            gpuxtb_compute_flag_t property) noexcept {
  return (plan.requested_properties & static_cast<std::uint32_t>(property)) != 0u;
}

__host__ __device__ bool checked_times_three(std::int64_t count, std::int64_t& product) noexcept {
  constexpr std::int64_t kMaximum = INT64_MAX;
  if (count < 0 || count > kMaximum / 3) return false;
  product = count * 3;
  return true;
}

__host__ __device__ bool expected_extents(const Gfn2PublicResultBridgeDevicePlan& plan,
                                          ExpectedExtents& extents) noexcept {
  if (plan.batch_size <= 0 || plan.total_atoms < plan.batch_size || plan.total_point_charges < 0) {
    return false;
  }
  std::int64_t atom_coordinates = 0;
  std::int64_t point_coordinates = 0;
  if (!checked_times_three(plan.total_atoms, atom_coordinates) ||
      !checked_times_three(plan.total_point_charges, point_coordinates)) {
    return false;
  }
  extents.energies = property_requested(plan, GPUXTB_COMPUTE_ENERGY) ? plan.batch_size : 0;
  extents.qm_forces = property_requested(plan, GPUXTB_COMPUTE_FORCES) ? atom_coordinates : 0;
  extents.atomic_charges =
      property_requested(plan, GPUXTB_COMPUTE_ATOMIC_CHARGES) ? plan.total_atoms : 0;
  extents.point_forces =
      property_requested(plan, GPUXTB_COMPUTE_POINT_CHARGE_FORCES) ? point_coordinates : 0;
  extents.diagnostics = plan.batch_size;
  return true;
}

__host__ __device__ bool aligned_or_null(const void* pointer, std::size_t alignment) noexcept {
  return pointer == nullptr || reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

__host__ __device__ bool canonical(const void* pointer, std::int64_t elements,
                                   std::size_t alignment) noexcept {
  return elements == 0 ? pointer == nullptr
                       : elements > 0 && pointer != nullptr &&
                             reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

bool structurally_safe_buffer(const void* pointer, std::int64_t elements,
                              std::size_t alignment) noexcept {
  return elements >= 0 && aligned_or_null(pointer, alignment);
}

bool structurally_safe_destination(const Gfn2PublicResultBridgeDestination& destination,
                                   std::size_t alignment) noexcept {
  return destination.elements >= 0 && aligned_or_null(destination.device_data, alignment);
}

bool structurally_safe_staging(const Gfn2PublicResultBridgeHostBuffer& staging,
                               std::size_t alignment) noexcept {
  return staging.elements >= 0 && aligned_or_null(staging.data, alignment);
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool overlaps(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin < second.end && second.begin < first.end;
}

template <std::size_t Capacity>
class RangeList {
 public:
  bool add(const void* pointer, std::int64_t elements, std::size_t element_size) noexcept {
    if (size_ == Capacity || elements < 0 ||
        static_cast<std::uint64_t>(elements) >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
      return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(elements) * element_size;
    if (bytes == 0u || pointer == nullptr) {
      ranges_[size_++] = {};
      return true;
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) return false;
    ranges_[size_++] = {begin, begin + bytes};
    return true;
  }

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] const AddressRange& operator[](std::size_t index) const noexcept {
    return ranges_[index];
  }

 private:
  std::array<AddressRange, Capacity> ranges_{};
  std::size_t size_ = 0u;
};

template <std::size_t ReadCapacity, std::size_t WriteCapacity>
bool disjoint_writes(const RangeList<ReadCapacity>& reads,
                     const RangeList<WriteCapacity>& writes) noexcept {
  for (std::size_t write = 0u; write < writes.size(); ++write) {
    for (std::size_t read = 0u; read < reads.size(); ++read) {
      if (overlaps(writes[write], reads[read])) return false;
    }
    for (std::size_t peer = write + 1u; peer < writes.size(); ++peer) {
      if (overlaps(writes[write], writes[peer])) return false;
    }
  }
  return true;
}

template <std::size_t Capacity>
bool pairwise_disjoint(const RangeList<Capacity>& ranges) noexcept {
  for (std::size_t first = 0u; first < ranges.size(); ++first) {
    for (std::size_t second = first + 1u; second < ranges.size(); ++second) {
      if (overlaps(ranges[first], ranges[second])) return false;
    }
  }
  return true;
}

/*
 * This host check proves only that a malformed logical contract cannot make
 * preflight itself unsafe. Logical sealing remains device-authoritative so a
 * graph-ordered epoch/publication error can gate CUDA destinations atomically.
 */
bool valid_launch_binding(const Gfn2PublicResultBridgeDevicePlan& plan,
                          const Gfn2PublicResultBridgeDeviceInput& input,
                          const Gfn2PublicResultBridgeDeviceStaging& device_staging,
                          const Gfn2PublicResultBridgeDeviceDestinations& destinations,
                          const Gfn2PublicResultBridgeHostStaging& staging,
                          const Gfn2PublicResultBridgeDeviceDiagnostics& diagnostics) noexcept {
  (void)plan;
  const bool fields_safe =
      structurally_safe_buffer(input.energies, input.energy_elements, alignof(double)) &&
      structurally_safe_buffer(input.qm_forces, input.qm_force_elements, alignof(double)) &&
      structurally_safe_buffer(input.atomic_charges, input.atomic_charge_elements,
                               alignof(double)) &&
      structurally_safe_buffer(input.point_forces, input.point_force_elements, alignof(double)) &&
      structurally_safe_buffer(input.iterations, input.batch_elements, alignof(std::int32_t)) &&
      structurally_safe_buffer(input.converged, input.batch_elements, alignof(std::uint8_t)) &&
      structurally_safe_buffer(input.system_statuses, input.batch_elements,
                               alignof(gpuxtb_status_t)) &&
      canonical(input.publication_plan_error, 1, alignof(std::uint32_t)) &&
      canonical(input.request_topology_error, 1, alignof(std::uint32_t)) &&
      canonical(input.publication_epoch_snapshot, 1, alignof(std::uint64_t)) &&
      canonical(input.current_geometry_epoch, 1, alignof(std::uint64_t)) &&
      structurally_safe_buffer(device_staging.energies, device_staging.energy_elements,
                               alignof(double)) &&
      structurally_safe_buffer(device_staging.qm_forces, device_staging.qm_force_elements,
                               alignof(double)) &&
      structurally_safe_buffer(device_staging.atomic_charges, device_staging.atomic_charge_elements,
                               alignof(double)) &&
      structurally_safe_buffer(device_staging.point_forces, device_staging.point_force_elements,
                               alignof(double)) &&
      structurally_safe_buffer(device_staging.iterations, device_staging.batch_elements,
                               alignof(std::int32_t)) &&
      structurally_safe_buffer(device_staging.converged, device_staging.batch_elements,
                               alignof(std::uint8_t)) &&
      structurally_safe_buffer(device_staging.system_statuses, device_staging.batch_elements,
                               alignof(gpuxtb_status_t)) &&
      structurally_safe_destination(destinations.energies, alignof(double)) &&
      structurally_safe_destination(destinations.qm_forces, alignof(double)) &&
      structurally_safe_destination(destinations.atomic_charges, alignof(double)) &&
      structurally_safe_destination(destinations.point_forces, alignof(double)) &&
      structurally_safe_destination(destinations.iterations, alignof(std::int32_t)) &&
      structurally_safe_destination(destinations.converged, alignof(std::uint8_t)) &&
      structurally_safe_destination(destinations.system_statuses, alignof(gpuxtb_status_t)) &&
      structurally_safe_staging(staging.energies, alignof(double)) &&
      structurally_safe_staging(staging.qm_forces, alignof(double)) &&
      structurally_safe_staging(staging.atomic_charges, alignof(double)) &&
      structurally_safe_staging(staging.point_forces, alignof(double)) &&
      structurally_safe_staging(staging.iterations, alignof(std::int32_t)) &&
      structurally_safe_staging(staging.converged, alignof(std::uint8_t)) &&
      structurally_safe_staging(staging.system_statuses, alignof(gpuxtb_status_t)) &&
      canonical(staging.control, 1, alignof(Gfn2PublicResultBridgeControl)) &&
      staging.control_elements >= 0 &&
      canonical(staging.pending_result_flags, 1, alignof(std::uint32_t)) &&
      canonical(diagnostics.control, 1, alignof(Gfn2PublicResultBridgeControl)) &&
      diagnostics.control_elements >= 0;
  if (!fields_safe) return false;

  RangeList<11> device_reads;
  RangeList<8> device_writes;
  const bool device_ranges_valid =
      device_reads.add(input.energies, input.energy_elements, sizeof(double)) &&
      device_reads.add(input.qm_forces, input.qm_force_elements, sizeof(double)) &&
      device_reads.add(input.atomic_charges, input.atomic_charge_elements, sizeof(double)) &&
      device_reads.add(input.point_forces, input.point_force_elements, sizeof(double)) &&
      device_reads.add(input.iterations, input.batch_elements, sizeof(std::int32_t)) &&
      device_reads.add(input.converged, input.batch_elements, sizeof(std::uint8_t)) &&
      device_reads.add(input.system_statuses, input.batch_elements, sizeof(gpuxtb_status_t)) &&
      device_reads.add(input.publication_plan_error, 1, sizeof(std::uint32_t)) &&
      device_reads.add(input.request_topology_error, 1, sizeof(std::uint32_t)) &&
      device_reads.add(input.publication_epoch_snapshot, 1, sizeof(std::uint64_t)) &&
      device_reads.add(input.current_geometry_epoch, 1, sizeof(std::uint64_t)) &&
      device_writes.add(device_staging.energies, device_staging.energy_elements, sizeof(double)) &&
      device_writes.add(device_staging.qm_forces, device_staging.qm_force_elements,
                        sizeof(double)) &&
      device_writes.add(device_staging.atomic_charges, device_staging.atomic_charge_elements,
                        sizeof(double)) &&
      device_writes.add(device_staging.point_forces, device_staging.point_force_elements,
                        sizeof(double)) &&
      device_writes.add(device_staging.iterations, device_staging.batch_elements,
                        sizeof(std::int32_t)) &&
      device_writes.add(device_staging.converged, device_staging.batch_elements,
                        sizeof(std::uint8_t)) &&
      device_writes.add(device_staging.system_statuses, device_staging.batch_elements,
                        sizeof(gpuxtb_status_t)) &&
      device_writes.add(diagnostics.control, 1, sizeof(Gfn2PublicResultBridgeControl));
  if (!device_ranges_valid || !disjoint_writes(device_reads, device_writes)) return false;

  RangeList<9> host_writes;
  const bool host_ranges_valid =
      host_writes.add(staging.energies.data, staging.energies.elements, sizeof(double)) &&
      host_writes.add(staging.qm_forces.data, staging.qm_forces.elements, sizeof(double)) &&
      host_writes.add(staging.atomic_charges.data, staging.atomic_charges.elements,
                      sizeof(double)) &&
      host_writes.add(staging.point_forces.data, staging.point_forces.elements, sizeof(double)) &&
      host_writes.add(staging.iterations.data, staging.iterations.elements, sizeof(std::int32_t)) &&
      host_writes.add(staging.converged.data, staging.converged.elements, sizeof(std::uint8_t)) &&
      host_writes.add(staging.system_statuses.data, staging.system_statuses.elements,
                      sizeof(gpuxtb_status_t)) &&
      host_writes.add(staging.control, 1, sizeof(Gfn2PublicResultBridgeControl)) &&
      host_writes.add(staging.pending_result_flags, 1, sizeof(std::uint32_t));
  return host_ranges_valid && pairwise_disjoint(host_writes);
}

__host__ __device__ bool valid_input_buffer(const void* pointer, std::int64_t elements,
                                            std::int64_t expected, std::size_t alignment) noexcept {
  return elements == expected && canonical(pointer, expected, alignment);
}

__host__ __device__ bool valid_destination(const Gfn2PublicResultBridgeDestination& destination,
                                           const Gfn2PublicResultBridgeHostBuffer& staging,
                                           std::int64_t expected, bool requested,
                                           std::size_t alignment) noexcept {
  if (!requested) {
    return destination.route == Route::kAbsent && destination.device_data == nullptr &&
           destination.elements == 0 && staging.data == nullptr && staging.elements == 0;
  }
  if (destination.elements != expected) return false;
  if (destination.route == Route::kHost) {
    return destination.device_data == nullptr && staging.elements == expected &&
           canonical(staging.data, expected, alignment);
  }
  if (destination.route == Route::kCudaDevice) {
    return canonical(destination.device_data, expected, alignment) && staging.data == nullptr &&
           staging.elements == 0;
  }
  return false;
}

__host__ __device__ BridgeError static_contract_error(
    const Gfn2PublicResultBridgeDevicePlan& plan, const Gfn2PublicResultBridgeDeviceInput& input,
    const Gfn2PublicResultBridgeDeviceStaging& device_staging,
    const Gfn2PublicResultBridgeDeviceDestinations& destinations,
    const Gfn2PublicResultBridgeHostStaging& staging,
    const Gfn2PublicResultBridgeDeviceDiagnostics& diagnostics) noexcept {
  if (plan.plan_token == 0u || input.plan_token != plan.plan_token ||
      device_staging.plan_token != plan.plan_token || destinations.plan_token != plan.plan_token ||
      staging.plan_token != plan.plan_token || diagnostics.plan_token != plan.plan_token) {
    return BridgeError::kPlanTokenMismatch;
  }
  if (plan.abi_version != kGfn2PublicResultBridgeAbiVersion) {
    return BridgeError::kInvalidAbiVersion;
  }
  if (plan.reserved != 0u || plan.requested_properties == 0u ||
      (plan.requested_properties & ~kKnownProperties) != 0u ||
      (plan.result_flags & ~kKnownResultFlags) != 0u) {
    return BridgeError::kInvalidFlags;
  }

  ExpectedExtents extents;
  if (!expected_extents(plan, extents) ||
      !valid_input_buffer(input.energies, input.energy_elements, extents.energies,
                          alignof(double)) ||
      !valid_input_buffer(input.qm_forces, input.qm_force_elements, extents.qm_forces,
                          alignof(double)) ||
      !valid_input_buffer(input.atomic_charges, input.atomic_charge_elements,
                          extents.atomic_charges, alignof(double)) ||
      !valid_input_buffer(input.point_forces, input.point_force_elements, extents.point_forces,
                          alignof(double)) ||
      input.batch_elements != extents.diagnostics ||
      !canonical(input.iterations, extents.diagnostics, alignof(std::int32_t)) ||
      !canonical(input.converged, extents.diagnostics, alignof(std::uint8_t)) ||
      !canonical(input.system_statuses, extents.diagnostics, alignof(gpuxtb_status_t))) {
    return BridgeError::kInvalidExtents;
  }
  if (!valid_input_buffer(device_staging.energies, device_staging.energy_elements, extents.energies,
                          alignof(double)) ||
      !valid_input_buffer(device_staging.qm_forces, device_staging.qm_force_elements,
                          extents.qm_forces, alignof(double)) ||
      !valid_input_buffer(device_staging.atomic_charges, device_staging.atomic_charge_elements,
                          extents.atomic_charges, alignof(double)) ||
      !valid_input_buffer(device_staging.point_forces, device_staging.point_force_elements,
                          extents.point_forces, alignof(double)) ||
      device_staging.batch_elements != extents.diagnostics ||
      !canonical(device_staging.iterations, extents.diagnostics, alignof(std::int32_t)) ||
      !canonical(device_staging.converged, extents.diagnostics, alignof(std::uint8_t)) ||
      !canonical(device_staging.system_statuses, extents.diagnostics, alignof(gpuxtb_status_t))) {
    return BridgeError::kInvalidExtents;
  }

  const bool destinations_valid =
      valid_destination(destinations.energies, staging.energies, extents.energies,
                        property_requested(plan, GPUXTB_COMPUTE_ENERGY), alignof(double)) &&
      valid_destination(destinations.qm_forces, staging.qm_forces, extents.qm_forces,
                        property_requested(plan, GPUXTB_COMPUTE_FORCES), alignof(double)) &&
      valid_destination(destinations.atomic_charges, staging.atomic_charges, extents.atomic_charges,
                        property_requested(plan, GPUXTB_COMPUTE_ATOMIC_CHARGES), alignof(double)) &&
      valid_destination(destinations.point_forces, staging.point_forces, extents.point_forces,
                        property_requested(plan, GPUXTB_COMPUTE_POINT_CHARGE_FORCES),
                        alignof(double)) &&
      valid_destination(destinations.iterations, staging.iterations, extents.diagnostics, true,
                        alignof(std::int32_t)) &&
      valid_destination(destinations.converged, staging.converged, extents.diagnostics, true,
                        alignof(std::uint8_t)) &&
      valid_destination(destinations.system_statuses, staging.system_statuses, extents.diagnostics,
                        true, alignof(gpuxtb_status_t)) &&
      staging.control_elements == 1 && staging.control != nullptr &&
      staging.pending_result_flags != nullptr && diagnostics.control_elements == 1 &&
      diagnostics.control != nullptr;
  return destinations_valid ? BridgeError::kSuccess : BridgeError::kInvalidDestinations;
}

__global__ void public_result_preflight_kernel(
    Gfn2PublicResultBridgeDevicePlan plan, Gfn2PublicResultBridgeDeviceInput input,
    Gfn2PublicResultBridgeDeviceStaging device_staging,
    Gfn2PublicResultBridgeDeviceDestinations destinations,
    Gfn2PublicResultBridgeHostStaging staging,
    Gfn2PublicResultBridgeDeviceDiagnostics diagnostics) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  Gfn2PublicResultBridgeControl control{};
  control.plan_token = plan.plan_token;
  control.internal_publication_plan_error = *input.publication_plan_error;
  control.publication_epoch_snapshot = *input.publication_epoch_snapshot;
  control.current_geometry_epoch = *input.current_geometry_epoch;

  BridgeError error =
      static_contract_error(plan, input, device_staging, destinations, staging, diagnostics);
  if (error == BridgeError::kSuccess && control.internal_publication_plan_error != 0u) {
    error = BridgeError::kInternalPublicationFailure;
  }
  if (error == BridgeError::kSuccess && *input.request_topology_error != 0u) {
    error = BridgeError::kRequestTopologyMismatch;
  }
  if (error == BridgeError::kSuccess &&
      (control.publication_epoch_snapshot == 0u || control.current_geometry_epoch == 0u ||
       control.publication_epoch_snapshot != control.current_geometry_epoch)) {
    error = BridgeError::kInvalidEpoch;
  }
  if (error == BridgeError::kSuccess) {
    /* The public ABI exposes only data-level SCC/eigensolver failures. Any
     * internal or unknown status is an aggregate execution failure and must
     * gate every caller destination, matching the CPU backend contract. */
    for (std::int64_t system = 0; system < input.batch_elements; ++system) {
      const gpuxtb_status_t status = input.system_statuses[system];
      if (status != GPUXTB_STATUS_SUCCESS && status != GPUXTB_STATUS_SCC_NOT_CONVERGED &&
          status != GPUXTB_STATUS_EIGENSOLVER_FAILED) {
        error = BridgeError::kInternalPublicationFailure;
        break;
      }
    }
  }
  control.aggregate_error = static_cast<std::uint32_t>(error);
  *diagnostics.control = control;
}

template <typename T>
__device__ void copy_flat(const T* source, T* target, std::int64_t elements) {
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t element = index; element < elements; element += stride) {
    target[element] = source[element];
  }
}

__global__ void prepare_public_results_kernel(Gfn2PublicResultBridgeDevicePlan plan,
                                              Gfn2PublicResultBridgeDeviceInput input,
                                              Gfn2PublicResultBridgeDeviceStaging device_staging,
                                              Gfn2PublicResultBridgeDeviceDiagnostics diagnostics) {
  if (diagnostics.control->aggregate_error != static_cast<std::uint32_t>(BridgeError::kSuccess)) {
    return;
  }
  ExpectedExtents extents;
  if (!expected_extents(plan, extents)) return;
  copy_flat(input.energies, device_staging.energies, extents.energies);
  copy_flat(input.qm_forces, device_staging.qm_forces, extents.qm_forces);
  copy_flat(input.atomic_charges, device_staging.atomic_charges, extents.atomic_charges);
  copy_flat(input.point_forces, device_staging.point_forces, extents.point_forces);
  copy_flat(input.iterations, device_staging.iterations, extents.diagnostics);
  copy_flat(input.converged, device_staging.converged, extents.diagnostics);
  copy_flat(input.system_statuses, device_staging.system_statuses, extents.diagnostics);
}

template <typename T>
__device__ void commit_flat(const T* source, const Gfn2PublicResultBridgeDestination& destination,
                            std::int64_t elements) {
  if (destination.route != Route::kCudaDevice) return;
  copy_flat(source, static_cast<T*>(destination.device_data), elements);
}

__global__ void commit_public_results_kernel(Gfn2PublicResultBridgeDevicePlan plan,
                                             Gfn2PublicResultBridgeDeviceStaging device_staging,
                                             Gfn2PublicResultBridgeDeviceDestinations destinations,
                                             Gfn2PublicResultBridgeDeviceDiagnostics diagnostics) {
  if (diagnostics.control->aggregate_error != static_cast<std::uint32_t>(BridgeError::kSuccess)) {
    return;
  }
  ExpectedExtents extents;
  if (!expected_extents(plan, extents)) return;
  commit_flat(device_staging.energies, destinations.energies, extents.energies);
  commit_flat(device_staging.qm_forces, destinations.qm_forces, extents.qm_forces);
  commit_flat(device_staging.atomic_charges, destinations.atomic_charges, extents.atomic_charges);
  commit_flat(device_staging.point_forces, destinations.point_forces, extents.point_forces);
  commit_flat(device_staging.iterations, destinations.iterations, extents.diagnostics);
  commit_flat(device_staging.converged, destinations.converged, extents.diagnostics);
  commit_flat(device_staging.system_statuses, destinations.system_statuses, extents.diagnostics);
}

cudaError_t check_launch() noexcept { return cudaPeekAtLastError(); }

int copy_block_count(const Gfn2PublicResultBridgeDevicePlan& plan) noexcept {
  ExpectedExtents extents;
  if (!expected_extents(plan, extents)) return 1;
  std::int64_t maximum = extents.energies;
  if (extents.qm_forces > maximum) maximum = extents.qm_forces;
  if (extents.atomic_charges > maximum) maximum = extents.atomic_charges;
  if (extents.point_forces > maximum) maximum = extents.point_forces;
  if (extents.diagnostics > maximum) maximum = extents.diagnostics;
  const std::int64_t blocks =
      maximum / kThreadsPerBlock + (maximum % kThreadsPerBlock == 0 ? 0 : 1);
  if (blocks <= 1) return 1;
  return blocks > kMaximumCopyBlocks ? kMaximumCopyBlocks : static_cast<int>(blocks);
}

template <typename T>
cudaError_t stage_host_buffer(const T* source, const Gfn2PublicResultBridgeDestination& destination,
                              const Gfn2PublicResultBridgeHostBuffer& staging,
                              std::int64_t elements, cudaStream_t stream) noexcept {
  if (destination.route != Route::kHost || elements == 0) return cudaSuccess;
  return cudaMemcpyAsync(staging.data, source, static_cast<std::size_t>(elements) * sizeof(T),
                         cudaMemcpyDeviceToHost, stream);
}

__host__ __device__ bool valid_commit_destination(
    const Gfn2PublicResultBridgeDestination& destination, std::int64_t expected, bool requested,
    std::size_t alignment) noexcept {
  if (!requested) {
    return destination.route == Route::kAbsent && destination.device_data == nullptr &&
           destination.elements == 0;
  }
  if (destination.elements != expected) return false;
  if (destination.route == Route::kHost) return destination.device_data == nullptr;
  return destination.route == Route::kCudaDevice &&
         canonical(destination.device_data, expected, alignment);
}

bool valid_commit_binding(const Gfn2PublicResultBridgeDevicePlan& plan,
                          const Gfn2PublicResultBridgeDeviceStaging& device_staging,
                          const Gfn2PublicResultBridgeDeviceDestinations& destinations,
                          const Gfn2PublicResultBridgeDeviceDiagnostics& diagnostics) noexcept {
  ExpectedExtents extents;
  if (plan.plan_token == 0u || plan.abi_version != kGfn2PublicResultBridgeAbiVersion ||
      plan.reserved != 0u || plan.requested_properties == 0u ||
      (plan.requested_properties & ~kKnownProperties) != 0u ||
      (plan.result_flags & ~kKnownResultFlags) != 0u ||
      device_staging.plan_token != plan.plan_token || destinations.plan_token != plan.plan_token ||
      diagnostics.plan_token != plan.plan_token || diagnostics.control_elements != 1 ||
      !canonical(diagnostics.control, 1, alignof(Gfn2PublicResultBridgeControl)) ||
      !expected_extents(plan, extents) ||
      !valid_input_buffer(device_staging.energies, device_staging.energy_elements, extents.energies,
                          alignof(double)) ||
      !valid_input_buffer(device_staging.qm_forces, device_staging.qm_force_elements,
                          extents.qm_forces, alignof(double)) ||
      !valid_input_buffer(device_staging.atomic_charges, device_staging.atomic_charge_elements,
                          extents.atomic_charges, alignof(double)) ||
      !valid_input_buffer(device_staging.point_forces, device_staging.point_force_elements,
                          extents.point_forces, alignof(double)) ||
      device_staging.batch_elements != extents.diagnostics ||
      !canonical(device_staging.iterations, extents.diagnostics, alignof(std::int32_t)) ||
      !canonical(device_staging.converged, extents.diagnostics, alignof(std::uint8_t)) ||
      !canonical(device_staging.system_statuses, extents.diagnostics, alignof(gpuxtb_status_t)) ||
      !valid_commit_destination(destinations.energies, extents.energies,
                                property_requested(plan, GPUXTB_COMPUTE_ENERGY), alignof(double)) ||
      !valid_commit_destination(destinations.qm_forces, extents.qm_forces,
                                property_requested(plan, GPUXTB_COMPUTE_FORCES), alignof(double)) ||
      !valid_commit_destination(destinations.atomic_charges, extents.atomic_charges,
                                property_requested(plan, GPUXTB_COMPUTE_ATOMIC_CHARGES),
                                alignof(double)) ||
      !valid_commit_destination(destinations.point_forces, extents.point_forces,
                                property_requested(plan, GPUXTB_COMPUTE_POINT_CHARGE_FORCES),
                                alignof(double)) ||
      !valid_commit_destination(destinations.iterations, extents.diagnostics, true,
                                alignof(std::int32_t)) ||
      !valid_commit_destination(destinations.converged, extents.diagnostics, true,
                                alignof(std::uint8_t)) ||
      !valid_commit_destination(destinations.system_statuses, extents.diagnostics, true,
                                alignof(gpuxtb_status_t))) {
    return false;
  }

  RangeList<8> reads;
  RangeList<7> writes;
  const bool ranges_valid =
      reads.add(device_staging.energies, device_staging.energy_elements, sizeof(double)) &&
      reads.add(device_staging.qm_forces, device_staging.qm_force_elements, sizeof(double)) &&
      reads.add(device_staging.atomic_charges, device_staging.atomic_charge_elements,
                sizeof(double)) &&
      reads.add(device_staging.point_forces, device_staging.point_force_elements, sizeof(double)) &&
      reads.add(device_staging.iterations, device_staging.batch_elements, sizeof(std::int32_t)) &&
      reads.add(device_staging.converged, device_staging.batch_elements, sizeof(std::uint8_t)) &&
      reads.add(device_staging.system_statuses, device_staging.batch_elements,
                sizeof(gpuxtb_status_t)) &&
      reads.add(diagnostics.control, 1, sizeof(Gfn2PublicResultBridgeControl)) &&
      writes.add(destinations.energies.device_data, destinations.energies.elements,
                 sizeof(double)) &&
      writes.add(destinations.qm_forces.device_data, destinations.qm_forces.elements,
                 sizeof(double)) &&
      writes.add(destinations.atomic_charges.device_data, destinations.atomic_charges.elements,
                 sizeof(double)) &&
      writes.add(destinations.point_forces.device_data, destinations.point_forces.elements,
                 sizeof(double)) &&
      writes.add(destinations.iterations.device_data, destinations.iterations.elements,
                 sizeof(std::int32_t)) &&
      writes.add(destinations.converged.device_data, destinations.converged.elements,
                 sizeof(std::uint8_t)) &&
      writes.add(destinations.system_statuses.device_data, destinations.system_statuses.elements,
                 sizeof(gpuxtb_status_t));
  return ranges_valid && disjoint_writes(reads, writes);
}

}  // namespace

cudaError_t prepare_gfn2_public_results_cuda(
    const Gfn2PublicResultBridgeDevicePlan& plan, const Gfn2PublicResultBridgeDeviceInput& input,
    const Gfn2PublicResultBridgeDeviceStaging& device_staging,
    const Gfn2PublicResultBridgeDeviceDestinations& destinations,
    const Gfn2PublicResultBridgeHostStaging& staging,
    const Gfn2PublicResultBridgeDeviceDiagnostics& diagnostics, cudaStream_t stream) noexcept {
  if (!valid_launch_binding(plan, input, device_staging, destinations, staging, diagnostics)) {
    return cudaErrorInvalidValue;
  }

  public_result_preflight_kernel<<<1, 1, 0, stream>>>(plan, input, device_staging, destinations,
                                                      staging, diagnostics);
  cudaError_t status = check_launch();
  if (status != cudaSuccess) return status;

  ExpectedExtents extents;
  const bool static_contract_valid =
      static_contract_error(plan, input, device_staging, destinations, staging, diagnostics) ==
          BridgeError::kSuccess &&
      expected_extents(plan, extents);
  if (static_contract_valid) {
    prepare_public_results_kernel<<<copy_block_count(plan), kThreadsPerBlock, 0, stream>>>(
        plan, input, device_staging, diagnostics);
    status = check_launch();
    if (status != cudaSuccess) return status;
    status = stage_host_buffer(device_staging.energies, destinations.energies, staging.energies,
                               extents.energies, stream);
    if (status == cudaSuccess) {
      status = stage_host_buffer(device_staging.qm_forces, destinations.qm_forces,
                                 staging.qm_forces, extents.qm_forces, stream);
    }
    if (status == cudaSuccess) {
      status = stage_host_buffer(device_staging.atomic_charges, destinations.atomic_charges,
                                 staging.atomic_charges, extents.atomic_charges, stream);
    }
    if (status == cudaSuccess) {
      status = stage_host_buffer(device_staging.point_forces, destinations.point_forces,
                                 staging.point_forces, extents.point_forces, stream);
    }
    if (status == cudaSuccess) {
      status = stage_host_buffer(device_staging.iterations, destinations.iterations,
                                 staging.iterations, extents.diagnostics, stream);
    }
    if (status == cudaSuccess) {
      status = stage_host_buffer(device_staging.converged, destinations.converged,
                                 staging.converged, extents.diagnostics, stream);
    }
    if (status == cudaSuccess) {
      status = stage_host_buffer(device_staging.system_statuses, destinations.system_statuses,
                                 staging.system_statuses, extents.diagnostics, stream);
    }
    if (status != cudaSuccess) return status;
  }

  status = cudaMemcpyAsync(staging.control, diagnostics.control,
                           sizeof(Gfn2PublicResultBridgeControl), cudaMemcpyDeviceToHost, stream);
  if (status != cudaSuccess) return status;

  *staging.pending_result_flags = plan.result_flags;
  return cudaSuccess;
}

cudaError_t commit_gfn2_public_results_cuda(
    const Gfn2PublicResultBridgeDevicePlan& plan,
    const Gfn2PublicResultBridgeDeviceStaging& device_staging,
    const Gfn2PublicResultBridgeDeviceDestinations& destinations,
    const Gfn2PublicResultBridgeDeviceDiagnostics& diagnostics, cudaStream_t stream) noexcept {
  if (!valid_commit_binding(plan, device_staging, destinations, diagnostics)) {
    return cudaErrorInvalidValue;
  }
  commit_public_results_kernel<<<copy_block_count(plan), kThreadsPerBlock, 0, stream>>>(
      plan, device_staging, destinations, diagnostics);
  return check_launch();
}

}  // namespace gpuxtb::detail::cuda
