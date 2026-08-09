#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_public_result_bridge.cuh"

namespace {

using gpuxtb::detail::cuda::commit_gfn2_public_results_cuda;
using gpuxtb::detail::cuda::Gfn2PublicResultBridgeControl;
using gpuxtb::detail::cuda::Gfn2PublicResultBridgeDestination;
using gpuxtb::detail::cuda::Gfn2PublicResultBridgeDeviceDestinations;
using gpuxtb::detail::cuda::Gfn2PublicResultBridgeDeviceDiagnostics;
using gpuxtb::detail::cuda::Gfn2PublicResultBridgeDeviceInput;
using gpuxtb::detail::cuda::Gfn2PublicResultBridgeDevicePlan;
using gpuxtb::detail::cuda::Gfn2PublicResultBridgeDeviceStaging;
using gpuxtb::detail::cuda::Gfn2PublicResultBridgeError;
using gpuxtb::detail::cuda::Gfn2PublicResultBridgeHostBuffer;
using gpuxtb::detail::cuda::Gfn2PublicResultBridgeHostStaging;
using gpuxtb::detail::cuda::Gfn2PublicResultRoute;
using gpuxtb::detail::cuda::prepare_gfn2_public_results_cuda;

#define CHECK(condition)                                                                           \
  do {                                                                                             \
    if (!(condition)) {                                                                            \
      std::fprintf(stderr, "public result bridge check failed at %s:%d: %s\n", __FILE__, __LINE__, \
                   #condition);                                                                    \
      return false;                                                                                \
    }                                                                                              \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

constexpr std::uint64_t kPlanToken = UINT64_C(0x72f154e8a31bc906);
constexpr double kDoubleSentinel = -9137.625;
constexpr std::int32_t kIterationSentinel = -123456789;
constexpr std::uint8_t kConvergedSentinel = UINT8_C(0xa5);
constexpr gpuxtb_status_t kStatusSentinel = INT32_C(0x5a5a5a5a);
constexpr std::uint32_t kFlagsSentinel = UINT32_C(0xc35aa53c);
constexpr std::uint32_t kAllProperties = GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES |
                                         GPUXTB_COMPUTE_ATOMIC_CHARGES |
                                         GPUXTB_COMPUTE_POINT_CHARGE_FORCES;

__global__ void hold_result_stream_kernel(unsigned long long clock_cycles) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const unsigned long long start = clock64();
  while (clock64() - start < clock_cycles) {
    __nanosleep(1000u);
  }
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (data_ != nullptr) (void)cudaFree(data_);
  }

  bool allocate(std::size_t count) {
    count_ = count;
    return count == 0u ||
           cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)) == cudaSuccess;
  }

  bool upload(const std::vector<T>& values) {
    return values.size() == count_ &&
           (count_ == 0u || cudaMemcpy(data_, values.data(), count_ * sizeof(T),
                                       cudaMemcpyHostToDevice) == cudaSuccess);
  }

  bool upload_one(const T& value) {
    return count_ == 1u &&
           cudaMemcpy(data_, &value, sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
  }

  bool fill(const T& value) { return upload(std::vector<T>(count_, value)); }

  std::vector<T> download() const {
    std::vector<T> values(count_);
    if (count_ != 0u && cudaMemcpy(values.data(), data_, count_ * sizeof(T),
                                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      values.clear();
    }
    return values;
  }

  T* get() noexcept { return data_; }
  const T* get() const noexcept { return data_; }
  std::size_t size() const noexcept { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

template <typename T>
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  ~PinnedBuffer() {
    if (data_ != nullptr) (void)cudaFreeHost(data_);
  }

  bool allocate(std::size_t count) {
    count_ = count;
    return count == 0u ||
           cudaMallocHost(reinterpret_cast<void**>(&data_), count * sizeof(T)) == cudaSuccess;
  }

  void fill(const T& value) {
    if (count_ != 0u) std::fill(data_, data_ + count_, value);
  }

  std::vector<T> snapshot() const {
    return count_ == 0u ? std::vector<T>{} : std::vector<T>(data_, data_ + count_);
  }

  T* get() noexcept { return data_; }
  const T* get() const noexcept { return data_; }
  std::size_t size() const noexcept { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

template <typename T>
bool bit_equal(const std::vector<T>& first, const std::vector<T>& second) {
  if (first.size() != second.size()) return false;
  return first.empty() || std::memcmp(first.data(), second.data(), first.size() * sizeof(T)) == 0;
}

template <typename T>
bool all_equal(const std::vector<T>& values, const T& expected) {
  return std::all_of(values.begin(), values.end(),
                     [&](const T& value) { return value == expected; });
}

enum Field : std::size_t {
  kEnergy = 0,
  kForces = 1,
  kCharges = 2,
  kPointForces = 3,
  kIterations = 4,
  kConverged = 5,
  kStatuses = 6,
  kFieldCount = 7,
};

struct Fixture {
  explicit Fixture(std::int64_t batch, std::uint32_t flags) : batch_size(batch), flags(flags) {}

  std::int64_t batch_size;
  std::uint32_t flags;
  std::int64_t total_atoms = 0;
  std::int64_t total_points = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> point_offsets;

  std::vector<double> host_energies;
  std::vector<double> host_forces;
  std::vector<double> host_charges;
  std::vector<double> host_point_forces;
  std::vector<std::int32_t> host_iterations;
  std::vector<std::uint8_t> host_converged;
  std::vector<gpuxtb_status_t> host_statuses;

  DeviceBuffer<double> internal_energies;
  DeviceBuffer<double> internal_forces;
  DeviceBuffer<double> internal_charges;
  DeviceBuffer<double> internal_point_forces;
  DeviceBuffer<std::int32_t> internal_iterations;
  DeviceBuffer<std::uint8_t> internal_converged;
  DeviceBuffer<gpuxtb_status_t> internal_statuses;
  DeviceBuffer<std::uint32_t> publication_plan_error;
  DeviceBuffer<std::uint64_t> publication_epoch;
  DeviceBuffer<std::uint64_t> current_epoch;

  DeviceBuffer<double> shadow_energies;
  DeviceBuffer<double> shadow_forces;
  DeviceBuffer<double> shadow_charges;
  DeviceBuffer<double> shadow_point_forces;
  DeviceBuffer<std::int32_t> shadow_iterations;
  DeviceBuffer<std::uint8_t> shadow_converged;
  DeviceBuffer<gpuxtb_status_t> shadow_statuses;

  DeviceBuffer<double> output_energies;
  DeviceBuffer<double> output_forces;
  DeviceBuffer<double> output_charges;
  DeviceBuffer<double> output_point_forces;
  DeviceBuffer<std::int32_t> output_iterations;
  DeviceBuffer<std::uint8_t> output_converged;
  DeviceBuffer<gpuxtb_status_t> output_statuses;
  DeviceBuffer<Gfn2PublicResultBridgeControl> device_control;

  PinnedBuffer<double> staging_energies;
  PinnedBuffer<double> staging_forces;
  PinnedBuffer<double> staging_charges;
  PinnedBuffer<double> staging_point_forces;
  PinnedBuffer<std::int32_t> staging_iterations;
  PinnedBuffer<std::uint8_t> staging_converged;
  PinnedBuffer<gpuxtb_status_t> staging_statuses;
  PinnedBuffer<Gfn2PublicResultBridgeControl> host_control;
  PinnedBuffer<std::uint32_t> pending_flags;

  Gfn2PublicResultBridgeDevicePlan plan{};
  Gfn2PublicResultBridgeDeviceInput input{};
  Gfn2PublicResultBridgeDeviceStaging device_staging{};
  Gfn2PublicResultBridgeDeviceDestinations destinations{};
  Gfn2PublicResultBridgeHostStaging staging{};
  Gfn2PublicResultBridgeDeviceDiagnostics diagnostics{};
  std::array<Gfn2PublicResultRoute, kFieldCount> routes{};

  bool requested(gpuxtb_compute_flag_t property) const {
    return (flags & static_cast<std::uint32_t>(property)) != 0u;
  }

  bool initialize() {
    atom_offsets.assign(static_cast<std::size_t>(batch_size + 1), 0);
    point_offsets.assign(static_cast<std::size_t>(batch_size + 1), 0);
    for (std::int64_t system = 0; system < batch_size; ++system) {
      atom_offsets[static_cast<std::size_t>(system + 1)] =
          atom_offsets[static_cast<std::size_t>(system)] + 1 + system % 4;
      point_offsets[static_cast<std::size_t>(system + 1)] =
          point_offsets[static_cast<std::size_t>(system)] + system % 3;
    }
    total_atoms = atom_offsets.back();
    total_points = point_offsets.back();

    if (requested(GPUXTB_COMPUTE_ENERGY)) host_energies.resize(batch_size);
    if (requested(GPUXTB_COMPUTE_FORCES)) host_forces.resize(3 * total_atoms);
    if (requested(GPUXTB_COMPUTE_ATOMIC_CHARGES)) host_charges.resize(total_atoms);
    if (requested(GPUXTB_COMPUTE_POINT_CHARGE_FORCES)) {
      host_point_forces.resize(3 * total_points);
    }
    host_iterations.resize(batch_size);
    host_converged.resize(batch_size);
    host_statuses.resize(batch_size);

    for (std::size_t index = 0; index < host_energies.size(); ++index) {
      host_energies[index] = -0.25 - static_cast<double>(index) * 0.125;
    }
    for (std::size_t index = 0; index < host_forces.size(); ++index) {
      host_forces[index] = 0.001 * static_cast<double>(index + 1u);
    }
    for (std::size_t index = 0; index < host_charges.size(); ++index) {
      host_charges[index] = -0.4 + 0.003 * static_cast<double>(index);
    }
    for (std::size_t index = 0; index < host_point_forces.size(); ++index) {
      host_point_forces[index] = -0.002 * static_cast<double>(index + 1u);
    }
    for (std::int64_t system = 0; system < batch_size; ++system) {
      host_iterations[static_cast<std::size_t>(system)] =
          static_cast<std::int32_t>(4 + system % 17);
      host_converged[static_cast<std::size_t>(system)] = 1u;
      host_statuses[static_cast<std::size_t>(system)] = GPUXTB_STATUS_SUCCESS;
    }

    /* Internal publication already owns failed-peer NaN/status semantics. */
    if (batch_size > 2) {
      const std::int64_t failed = batch_size / 2;
      const double nan = std::numeric_limits<double>::quiet_NaN();
      if (!host_energies.empty()) host_energies[static_cast<std::size_t>(failed)] = nan;
      if (!host_forces.empty()) {
        std::fill(host_forces.begin() + 3 * atom_offsets[static_cast<std::size_t>(failed)],
                  host_forces.begin() + 3 * atom_offsets[static_cast<std::size_t>(failed + 1)],
                  nan);
      }
      if (!host_charges.empty()) {
        std::fill(host_charges.begin() + atom_offsets[static_cast<std::size_t>(failed)],
                  host_charges.begin() + atom_offsets[static_cast<std::size_t>(failed + 1)], nan);
      }
      if (!host_point_forces.empty()) {
        std::fill(
            host_point_forces.begin() + 3 * point_offsets[static_cast<std::size_t>(failed)],
            host_point_forces.begin() + 3 * point_offsets[static_cast<std::size_t>(failed + 1)],
            nan);
      }
      host_converged[static_cast<std::size_t>(failed)] = 0u;
      host_statuses[static_cast<std::size_t>(failed)] = GPUXTB_STATUS_SCC_NOT_CONVERGED;
    }

    const std::size_t atom_coordinates = static_cast<std::size_t>(3 * total_atoms);
    const std::size_t point_coordinates = static_cast<std::size_t>(3 * total_points);
    const bool allocated = internal_energies.allocate(host_energies.size()) &&
                           internal_forces.allocate(host_forces.size()) &&
                           internal_charges.allocate(host_charges.size()) &&
                           internal_point_forces.allocate(host_point_forces.size()) &&
                           internal_iterations.allocate(host_iterations.size()) &&
                           internal_converged.allocate(host_converged.size()) &&
                           internal_statuses.allocate(host_statuses.size()) &&
                           publication_plan_error.allocate(1) && publication_epoch.allocate(1) &&
                           current_epoch.allocate(1) &&
                           shadow_energies.allocate(host_energies.size()) &&
                           shadow_forces.allocate(host_forces.size()) &&
                           shadow_charges.allocate(host_charges.size()) &&
                           shadow_point_forces.allocate(host_point_forces.size()) &&
                           shadow_iterations.allocate(host_iterations.size()) &&
                           shadow_converged.allocate(host_converged.size()) &&
                           shadow_statuses.allocate(host_statuses.size()) &&
                           output_energies.allocate(static_cast<std::size_t>(batch_size)) &&
                           output_forces.allocate(atom_coordinates) &&
                           output_charges.allocate(static_cast<std::size_t>(total_atoms)) &&
                           output_point_forces.allocate(point_coordinates) &&
                           output_iterations.allocate(static_cast<std::size_t>(batch_size)) &&
                           output_converged.allocate(static_cast<std::size_t>(batch_size)) &&
                           output_statuses.allocate(static_cast<std::size_t>(batch_size)) &&
                           device_control.allocate(1) &&
                           staging_energies.allocate(static_cast<std::size_t>(batch_size)) &&
                           staging_forces.allocate(atom_coordinates) &&
                           staging_charges.allocate(static_cast<std::size_t>(total_atoms)) &&
                           staging_point_forces.allocate(point_coordinates) &&
                           staging_iterations.allocate(static_cast<std::size_t>(batch_size)) &&
                           staging_converged.allocate(static_cast<std::size_t>(batch_size)) &&
                           staging_statuses.allocate(static_cast<std::size_t>(batch_size)) &&
                           host_control.allocate(1) && pending_flags.allocate(1);
    if (!allocated) return false;

    if (!internal_energies.upload(host_energies) || !internal_forces.upload(host_forces) ||
        !internal_charges.upload(host_charges) ||
        !internal_point_forces.upload(host_point_forces) ||
        !internal_iterations.upload(host_iterations) ||
        !internal_converged.upload(host_converged) || !internal_statuses.upload(host_statuses) ||
        !set_control_values(0u, 37u, 37u) || !reset_outputs()) {
      return false;
    }

    plan.requested_properties = flags;
    plan.result_flags = GPUXTB_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES;
    plan.plan_token = kPlanToken;
    plan.batch_size = batch_size;
    plan.total_atoms = total_atoms;
    plan.total_point_charges = total_points;

    input.energies = internal_energies.get();
    input.energy_elements = static_cast<std::int64_t>(host_energies.size());
    input.qm_forces = internal_forces.get();
    input.qm_force_elements = static_cast<std::int64_t>(host_forces.size());
    input.atomic_charges = internal_charges.get();
    input.atomic_charge_elements = static_cast<std::int64_t>(host_charges.size());
    input.point_forces = internal_point_forces.get();
    input.point_force_elements = static_cast<std::int64_t>(host_point_forces.size());
    input.iterations = internal_iterations.get();
    input.converged = internal_converged.get();
    input.system_statuses = internal_statuses.get();
    input.batch_elements = batch_size;
    input.publication_plan_error = publication_plan_error.get();
    input.publication_epoch_snapshot = publication_epoch.get();
    input.current_geometry_epoch = current_epoch.get();
    input.plan_token = kPlanToken;

    device_staging.energies = shadow_energies.get();
    device_staging.energy_elements = static_cast<std::int64_t>(host_energies.size());
    device_staging.qm_forces = shadow_forces.get();
    device_staging.qm_force_elements = static_cast<std::int64_t>(host_forces.size());
    device_staging.atomic_charges = shadow_charges.get();
    device_staging.atomic_charge_elements = static_cast<std::int64_t>(host_charges.size());
    device_staging.point_forces = shadow_point_forces.get();
    device_staging.point_force_elements = static_cast<std::int64_t>(host_point_forces.size());
    device_staging.iterations = shadow_iterations.get();
    device_staging.converged = shadow_converged.get();
    device_staging.system_statuses = shadow_statuses.get();
    device_staging.batch_elements = batch_size;
    device_staging.plan_token = kPlanToken;

    staging.control = host_control.get();
    staging.control_elements = 1;
    staging.pending_result_flags = pending_flags.get();
    staging.plan_token = kPlanToken;
    diagnostics.control = device_control.get();
    diagnostics.control_elements = 1;
    diagnostics.plan_token = kPlanToken;
    return true;
  }

  bool set_control_values(std::uint32_t plan_error, std::uint64_t snapshot, std::uint64_t current) {
    return publication_plan_error.upload_one(plan_error) &&
           publication_epoch.upload_one(snapshot) && current_epoch.upload_one(current);
  }

  bool reset_outputs() {
    Gfn2PublicResultBridgeControl control_sentinel{};
    control_sentinel.aggregate_error = UINT32_MAX;
    control_sentinel.internal_publication_plan_error = UINT32_MAX;
    control_sentinel.publication_epoch_snapshot = UINT64_MAX;
    control_sentinel.current_geometry_epoch = UINT64_MAX;
    control_sentinel.plan_token = UINT64_MAX;
    if (!output_energies.fill(kDoubleSentinel) || !output_forces.fill(kDoubleSentinel) ||
        !output_charges.fill(kDoubleSentinel) || !output_point_forces.fill(kDoubleSentinel) ||
        !output_iterations.fill(kIterationSentinel) || !output_converged.fill(kConvergedSentinel) ||
        !output_statuses.fill(kStatusSentinel)) {
      return false;
    }
    if (!shadow_energies.fill(kDoubleSentinel) || !shadow_forces.fill(kDoubleSentinel) ||
        !shadow_charges.fill(kDoubleSentinel) || !shadow_point_forces.fill(kDoubleSentinel) ||
        !shadow_iterations.fill(kIterationSentinel) || !shadow_converged.fill(kConvergedSentinel) ||
        !shadow_statuses.fill(kStatusSentinel)) {
      return false;
    }
    if (!device_control.fill(control_sentinel)) return false;
    staging_energies.fill(kDoubleSentinel);
    staging_forces.fill(kDoubleSentinel);
    staging_charges.fill(kDoubleSentinel);
    staging_point_forces.fill(kDoubleSentinel);
    staging_iterations.fill(kIterationSentinel);
    staging_converged.fill(kConvergedSentinel);
    staging_statuses.fill(kStatusSentinel);
    host_control.get()->aggregate_error = UINT32_MAX;
    *pending_flags.get() = kFlagsSentinel;
    return true;
  }

  template <typename T>
  static void bind_field(Gfn2PublicResultBridgeDestination& destination,
                         Gfn2PublicResultBridgeHostBuffer& staging_binding,
                         Gfn2PublicResultRoute route, std::int64_t elements,
                         DeviceBuffer<T>& device_output, PinnedBuffer<T>& host_staging) {
    destination.route = route;
    destination.elements = elements;
    destination.device_data = route == Gfn2PublicResultRoute::kCudaDevice
                                  ? static_cast<void*>(device_output.get())
                                  : nullptr;
    staging_binding.data =
        route == Gfn2PublicResultRoute::kHost ? static_cast<void*>(host_staging.get()) : nullptr;
    staging_binding.elements = route == Gfn2PublicResultRoute::kHost ? elements : 0;
  }

  static void bind_absent(Gfn2PublicResultBridgeDestination& destination,
                          Gfn2PublicResultBridgeHostBuffer& staging_binding) {
    destination = {};
    staging_binding = {};
  }

  void configure(const std::array<Gfn2PublicResultRoute, kFieldCount>& requested_routes) {
    routes = requested_routes;
    if (requested(GPUXTB_COMPUTE_ENERGY)) {
      bind_field(destinations.energies, staging.energies, routes[kEnergy], batch_size,
                 output_energies, staging_energies);
    } else {
      bind_absent(destinations.energies, staging.energies);
    }
    if (requested(GPUXTB_COMPUTE_FORCES)) {
      bind_field(destinations.qm_forces, staging.qm_forces, routes[kForces], 3 * total_atoms,
                 output_forces, staging_forces);
    } else {
      bind_absent(destinations.qm_forces, staging.qm_forces);
    }
    if (requested(GPUXTB_COMPUTE_ATOMIC_CHARGES)) {
      bind_field(destinations.atomic_charges, staging.atomic_charges, routes[kCharges], total_atoms,
                 output_charges, staging_charges);
    } else {
      bind_absent(destinations.atomic_charges, staging.atomic_charges);
    }
    if (requested(GPUXTB_COMPUTE_POINT_CHARGE_FORCES)) {
      bind_field(destinations.point_forces, staging.point_forces, routes[kPointForces],
                 3 * total_points, output_point_forces, staging_point_forces);
    } else {
      bind_absent(destinations.point_forces, staging.point_forces);
    }
    bind_field(destinations.iterations, staging.iterations, routes[kIterations], batch_size,
               output_iterations, staging_iterations);
    bind_field(destinations.converged, staging.converged, routes[kConverged], batch_size,
               output_converged, staging_converged);
    bind_field(destinations.system_statuses, staging.system_statuses, routes[kStatuses], batch_size,
               output_statuses, staging_statuses);
    destinations.plan_token = kPlanToken;
  }

  template <typename T>
  bool verify_routed_field(Field field, const std::vector<T>& expected,
                           const DeviceBuffer<T>& device_output,
                           const PinnedBuffer<T>& host_staging, const T& sentinel) const {
    const Gfn2PublicResultRoute route = routes[field];
    const std::vector<T> device = device_output.download();
    const std::vector<T> host = host_staging.snapshot();
    if (route == Gfn2PublicResultRoute::kHost) {
      return bit_equal(std::vector<T>(host.begin(), host.begin() + expected.size()), expected) &&
             all_equal(device, sentinel);
    }
    if (route == Gfn2PublicResultRoute::kCudaDevice) {
      return bit_equal(std::vector<T>(device.begin(), device.begin() + expected.size()),
                       expected) &&
             all_equal(host, sentinel);
    }
    return all_equal(device, sentinel) && all_equal(host, sentinel);
  }

  bool verify_success() const {
    CHECK(host_control.get()->aggregate_error ==
          static_cast<std::uint32_t>(Gfn2PublicResultBridgeError::kSuccess));
    CHECK(host_control.get()->internal_publication_plan_error == 0u);
    CHECK(host_control.get()->publication_epoch_snapshot == 37u);
    CHECK(host_control.get()->current_geometry_epoch == 37u);
    CHECK(host_control.get()->plan_token == kPlanToken);
    CHECK(*pending_flags.get() == plan.result_flags);
    CHECK(verify_routed_field(kEnergy, host_energies, output_energies, staging_energies,
                              kDoubleSentinel));
    CHECK(
        verify_routed_field(kForces, host_forces, output_forces, staging_forces, kDoubleSentinel));
    CHECK(verify_routed_field(kCharges, host_charges, output_charges, staging_charges,
                              kDoubleSentinel));
    CHECK(verify_routed_field(kPointForces, host_point_forces, output_point_forces,
                              staging_point_forces, kDoubleSentinel));
    CHECK(verify_routed_field(kIterations, host_iterations, output_iterations, staging_iterations,
                              kIterationSentinel));
    CHECK(verify_routed_field(kConverged, host_converged, output_converged, staging_converged,
                              kConvergedSentinel));
    CHECK(verify_routed_field(kStatuses, host_statuses, output_statuses, staging_statuses,
                              kStatusSentinel));
    return true;
  }

  bool verify_prepared_shadow() const {
    return bit_equal(shadow_energies.download(), host_energies) &&
           bit_equal(shadow_forces.download(), host_forces) &&
           bit_equal(shadow_charges.download(), host_charges) &&
           bit_equal(shadow_point_forces.download(), host_point_forces) &&
           bit_equal(shadow_iterations.download(), host_iterations) &&
           bit_equal(shadow_converged.download(), host_converged) &&
           bit_equal(shadow_statuses.download(), host_statuses);
  }

  bool all_device_outputs_are_sentinels() const {
    return all_equal(output_energies.download(), kDoubleSentinel) &&
           all_equal(output_forces.download(), kDoubleSentinel) &&
           all_equal(output_charges.download(), kDoubleSentinel) &&
           all_equal(output_point_forces.download(), kDoubleSentinel) &&
           all_equal(output_iterations.download(), kIterationSentinel) &&
           all_equal(output_converged.download(), kConvergedSentinel) &&
           all_equal(output_statuses.download(), kStatusSentinel);
  }
};

std::array<Gfn2PublicResultRoute, kFieldCount> uniform_routes(Gfn2PublicResultRoute route) {
  std::array<Gfn2PublicResultRoute, kFieldCount> result{};
  result.fill(route);
  return result;
}

bool run_success_case(std::int64_t batch,
                      const std::array<Gfn2PublicResultRoute, kFieldCount>& routes,
                      cudaStream_t stream) {
  Fixture fixture(batch, kAllProperties);
  CHECK(fixture.initialize());
  fixture.configure(routes);
  CUDA_CHECK(prepare_gfn2_public_results_cuda(fixture.plan, fixture.input, fixture.device_staging,
                                              fixture.destinations, fixture.staging,
                                              fixture.diagnostics, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(fixture.verify_prepared_shadow());
  CHECK(fixture.all_device_outputs_are_sentinels());
  CUDA_CHECK(commit_gfn2_public_results_cuda(fixture.plan, fixture.device_staging,
                                             fixture.destinations, fixture.diagnostics, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(fixture.verify_success());
  return true;
}

template <typename Mutator>
bool run_aggregate_failure(Mutator&& mutate, Gfn2PublicResultBridgeError expected_error,
                           cudaStream_t stream) {
  Fixture fixture(8, kAllProperties);
  CHECK(fixture.initialize());
  fixture.configure({Gfn2PublicResultRoute::kHost, Gfn2PublicResultRoute::kCudaDevice,
                     Gfn2PublicResultRoute::kHost, Gfn2PublicResultRoute::kCudaDevice,
                     Gfn2PublicResultRoute::kHost, Gfn2PublicResultRoute::kCudaDevice,
                     Gfn2PublicResultRoute::kHost});
  CHECK(mutate(fixture));

  std::vector<double> host_caller_energy(static_cast<std::size_t>(fixture.batch_size),
                                         kDoubleSentinel);
  std::vector<double> host_caller_charges(static_cast<std::size_t>(fixture.total_atoms),
                                          kDoubleSentinel);
  std::vector<std::int32_t> host_caller_iterations(static_cast<std::size_t>(fixture.batch_size),
                                                   kIterationSentinel);
  std::vector<gpuxtb_status_t> host_caller_statuses(static_cast<std::size_t>(fixture.batch_size),
                                                    kStatusSentinel);
  std::uint32_t caller_flags = kFlagsSentinel;

  CUDA_CHECK(prepare_gfn2_public_results_cuda(fixture.plan, fixture.input, fixture.device_staging,
                                              fixture.destinations, fixture.staging,
                                              fixture.diagnostics, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (fixture.host_control.get()->aggregate_error ==
      static_cast<std::uint32_t>(Gfn2PublicResultBridgeError::kSuccess)) {
    std::copy_n(fixture.staging_energies.get(), fixture.batch_size, host_caller_energy.begin());
    std::copy_n(fixture.staging_charges.get(), fixture.total_atoms, host_caller_charges.begin());
    std::copy_n(fixture.staging_iterations.get(), fixture.batch_size,
                host_caller_iterations.begin());
    std::copy_n(fixture.staging_statuses.get(), fixture.batch_size, host_caller_statuses.begin());
    caller_flags = *fixture.pending_flags.get();
  }

  CHECK(fixture.host_control.get()->aggregate_error == static_cast<std::uint32_t>(expected_error));
  CHECK(fixture.all_device_outputs_are_sentinels());
  CHECK(all_equal(host_caller_energy, kDoubleSentinel));
  CHECK(all_equal(host_caller_charges, kDoubleSentinel));
  CHECK(all_equal(host_caller_iterations, kIterationSentinel));
  CHECK(all_equal(host_caller_statuses, kStatusSentinel));
  CHECK(caller_flags == kFlagsSentinel);
  return true;
}

bool test_success_matrix(cudaStream_t stream) {
  CHECK(run_success_case(1, uniform_routes(Gfn2PublicResultRoute::kHost), stream));
  CHECK(run_success_case(8, uniform_routes(Gfn2PublicResultRoute::kCudaDevice), stream));
  CHECK(run_success_case(32,
                         {Gfn2PublicResultRoute::kHost, Gfn2PublicResultRoute::kCudaDevice,
                          Gfn2PublicResultRoute::kHost, Gfn2PublicResultRoute::kCudaDevice,
                          Gfn2PublicResultRoute::kHost, Gfn2PublicResultRoute::kCudaDevice,
                          Gfn2PublicResultRoute::kHost},
                         stream));
  CHECK(run_success_case(128,
                         {Gfn2PublicResultRoute::kCudaDevice, Gfn2PublicResultRoute::kHost,
                          Gfn2PublicResultRoute::kCudaDevice, Gfn2PublicResultRoute::kHost,
                          Gfn2PublicResultRoute::kCudaDevice, Gfn2PublicResultRoute::kHost,
                          Gfn2PublicResultRoute::kCudaDevice},
                         stream));
  return true;
}

bool test_aggregate_gate(cudaStream_t stream) {
  CHECK(run_aggregate_failure(
      [](Fixture& fixture) {
        fixture.input.plan_token ^= UINT64_C(1);
        return true;
      },
      Gfn2PublicResultBridgeError::kPlanTokenMismatch, stream));
  CHECK(run_aggregate_failure(
      [](Fixture& fixture) { return fixture.set_control_values(73u, 37u, 37u); },
      Gfn2PublicResultBridgeError::kInternalPublicationFailure, stream));
  CHECK(run_aggregate_failure(
      [](Fixture& fixture) {
        fixture.host_statuses[0] = GPUXTB_STATUS_INTERNAL_ERROR;
        return fixture.internal_statuses.upload(fixture.host_statuses);
      },
      Gfn2PublicResultBridgeError::kInternalPublicationFailure, stream));
  CHECK(run_aggregate_failure(
      [](Fixture& fixture) { return fixture.set_control_values(0u, 36u, 37u); },
      Gfn2PublicResultBridgeError::kInvalidEpoch, stream));
  CHECK(
      run_aggregate_failure([](Fixture& fixture) { return fixture.set_control_values(0u, 0u, 0u); },
                            Gfn2PublicResultBridgeError::kInvalidEpoch, stream));
  CHECK(run_aggregate_failure(
      [](Fixture& fixture) {
        fixture.input.energy_elements -= 1;
        return true;
      },
      Gfn2PublicResultBridgeError::kInvalidExtents, stream));
  CHECK(run_aggregate_failure(
      [](Fixture& fixture) {
        fixture.plan.requested_properties |= UINT32_C(1) << 29;
        return true;
      },
      Gfn2PublicResultBridgeError::kInvalidFlags, stream));
  CHECK(run_aggregate_failure(
      [](Fixture& fixture) {
        fixture.destinations.energies.elements += 1;
        return true;
      },
      Gfn2PublicResultBridgeError::kInvalidDestinations, stream));
  return true;
}

bool test_unrequested_outputs(cudaStream_t stream) {
  Fixture fixture(8, GPUXTB_COMPUTE_ENERGY);
  CHECK(fixture.initialize());
  fixture.configure({Gfn2PublicResultRoute::kCudaDevice, Gfn2PublicResultRoute::kAbsent,
                     Gfn2PublicResultRoute::kAbsent, Gfn2PublicResultRoute::kAbsent,
                     Gfn2PublicResultRoute::kHost, Gfn2PublicResultRoute::kCudaDevice,
                     Gfn2PublicResultRoute::kHost});
  CUDA_CHECK(prepare_gfn2_public_results_cuda(fixture.plan, fixture.input, fixture.device_staging,
                                              fixture.destinations, fixture.staging,
                                              fixture.diagnostics, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(fixture.all_device_outputs_are_sentinels());
  CUDA_CHECK(commit_gfn2_public_results_cuda(fixture.plan, fixture.device_staging,
                                             fixture.destinations, fixture.diagnostics, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(fixture.verify_success());
  CHECK(all_equal(fixture.output_forces.download(), kDoubleSentinel));
  CHECK(all_equal(fixture.output_charges.download(), kDoubleSentinel));
  CHECK(all_equal(fixture.output_point_forces.download(), kDoubleSentinel));
  return true;
}

bool test_prepare_submission_precedes_host_acceptance(cudaStream_t stream) {
  Fixture fixture(1, kAllProperties);
  CHECK(fixture.initialize());
  fixture.configure(uniform_routes(Gfn2PublicResultRoute::kHost));

  int device = 0;
  int clock_rate_khz = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device));
  CHECK(clock_rate_khz > 0);
  const unsigned long long delay_cycles = static_cast<unsigned long long>(clock_rate_khz) * 250ULL;
  hold_result_stream_kernel<<<1, 1, 0, stream>>>(delay_cycles);
  CUDA_CHECK(cudaPeekAtLastError());

  CUDA_CHECK(prepare_gfn2_public_results_cuda(fixture.plan, fixture.input, fixture.device_staging,
                                              fixture.destinations, fixture.staging,
                                              fixture.diagnostics, stream));
  const cudaError_t query_status = cudaStreamQuery(stream);
  CHECK(query_status == cudaErrorNotReady || query_status == cudaSuccess);

  /* Submission owns the pending flag image immediately, but the downloaded
   * aggregate is not host-readable as an acceptance decision until the owner
   * observes stream completion. Instrumented CUDA runtimes may serialize the
   * delay; in an ordinary run the not-ready branch proves that separation. */
  CHECK(*fixture.pending_flags.get() == fixture.plan.result_flags);
  if (query_status == cudaErrorNotReady) {
    CHECK(fixture.host_control.get()->aggregate_error == UINT32_MAX);
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(fixture.host_control.get()->aggregate_error ==
        static_cast<std::uint32_t>(Gfn2PublicResultBridgeError::kSuccess));
  CHECK(fixture.verify_prepared_shadow());
  CHECK(fixture.all_device_outputs_are_sentinels());
  return true;
}

bool test_staging_enqueue_failure_precedes_device_writes(cudaStream_t stream) {
  Fixture fixture(8, kAllProperties);
  CHECK(fixture.initialize());
  fixture.configure({Gfn2PublicResultRoute::kHost, Gfn2PublicResultRoute::kCudaDevice,
                     Gfn2PublicResultRoute::kHost, Gfn2PublicResultRoute::kCudaDevice,
                     Gfn2PublicResultRoute::kHost, Gfn2PublicResultRoute::kCudaDevice,
                     Gfn2PublicResultRoute::kHost});

  /*
   * The binding is structurally aligned and logically sealed, but starts one
   * double into a pinned allocation while declaring its full original extent.
   * cudaMemcpyAsync rejects the transfer beyond the registered allocation
   * synchronously. No device caller copy may have entered the stream then.
   */
  fixture.staging.energies.data = fixture.staging_energies.get() + 1;
  const cudaError_t status = prepare_gfn2_public_results_cuda(
      fixture.plan, fixture.input, fixture.device_staging, fixture.destinations, fixture.staging,
      fixture.diagnostics, stream);
  CHECK(status != cudaSuccess);
  (void)cudaGetLastError();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const std::vector<Gfn2PublicResultBridgeControl> device_control =
      fixture.device_control.download();
  CHECK(device_control.size() == 1u);
  CHECK(device_control[0].aggregate_error ==
        static_cast<std::uint32_t>(Gfn2PublicResultBridgeError::kSuccess));
  CHECK(device_control[0].plan_token == kPlanToken);
  CHECK(fixture.all_device_outputs_are_sentinels());
  CHECK(*fixture.pending_flags.get() == kFlagsSentinel);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const bool skip_expected_api_error =
      argc == 2 && std::strcmp(argv[1], "--skip-expected-api-error") == 0;
  if (argc > 2 || (argc == 2 && !skip_expected_api_error)) return 2;
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::fprintf(stderr, "CUDA device unavailable\n");
    return 77;
  }
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return 1;
  const bool success =
      test_success_matrix(stream) && test_aggregate_gate(stream) &&
      test_unrequested_outputs(stream) &&
      test_prepare_submission_precedes_host_acceptance(stream) &&
      (skip_expected_api_error || test_staging_enqueue_failure_precedes_device_writes(stream));
  const cudaError_t destroy_status = cudaStreamDestroy(stream);
  return success && destroy_status == cudaSuccess ? 0 : 1;
}
