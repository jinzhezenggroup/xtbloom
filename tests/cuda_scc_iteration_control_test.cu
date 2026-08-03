#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_scc_iteration_control.cuh"

namespace {

using gpuxtb::detail::Gfn2GenerationScope;
using gpuxtb::detail::Gfn2GeometryCacheProvenanceView;
using gpuxtb::detail::Gfn2PlanMemorySpace;
using gpuxtb::detail::cuda::Gfn2SccCacheProvenanceBinding;
using gpuxtb::detail::cuda::Gfn2GeometryEpochConsumerDevice;
using gpuxtb::detail::cuda::Gfn2GeometryEpochDevice;
using gpuxtb::detail::cuda::Gfn2SccIterationControlCode;
using gpuxtb::detail::cuda::Gfn2SccIterationDeviceLedger;
using gpuxtb::detail::cuda::Gfn2SccIterationDevicePolicy;
using gpuxtb::detail::cuda::Gfn2SccIterationDeviceProvenance;
using gpuxtb::detail::cuda::Gfn2SccIterationDeviceStateInput;
using gpuxtb::detail::cuda::Gfn2SccStageCodeFormat;
using gpuxtb::detail::cuda::Gfn2SccStageDeviceCodeRole;
using gpuxtb::detail::cuda::Gfn2SccStageDeviceReport;
using gpuxtb::detail::cuda::Gfn2SccStageId;

constexpr std::uint64_t kPlanToken = 0x87c0ffee12345678ULL;
constexpr std::uint64_t kGeometryGeneration = 17u;
constexpr std::uint64_t kWarmStartGeneration = 29u;

// Stage-qualified failure records are persistent diagnostics, so extending
// the DAG must never renumber an identity already emitted by older builds.
static_assert(static_cast<std::uint32_t>(Gfn2SccStageId::kActivity) == 1u);
static_assert(static_cast<std::uint32_t>(Gfn2SccStageId::kStatePublication) == 23u);
static_assert(static_cast<std::uint32_t>(Gfn2SccStageId::kES2RawEnergy) == 24u);
static_assert(static_cast<std::uint32_t>(Gfn2SccStageId::kES3RawEnergy) == 25u);
static_assert(static_cast<std::uint32_t>(Gfn2SccStageId::kAES2RawEnergy) == 26u);
static_assert(static_cast<std::uint32_t>(Gfn2SccStageId::kD4RawEnergy) == 27u);
static_assert(static_cast<std::uint32_t>(Gfn2SccStageId::kExplicitPointChargeRawEnergy) == 28u);
static_assert(static_cast<std::uint32_t>(Gfn2SccStageId::kPeriodicRawEnergy) == 29u);
static_assert(gpuxtb::detail::cuda::gfn2_scc_stage_id_is_valid(Gfn2SccStageId::kPeriodicRawEnergy));
static_assert(!gpuxtb::detail::cuda::gfn2_scc_stage_id_in_domain(static_cast<Gfn2SccStageId>(30u)));

#define CHECK(condition)                                                                         \
  do {                                                                                           \
    if (!(condition)) {                                                                          \
      std::cerr << "CHECK failed at " << __FILE__ << ':' << __LINE__ << ": " #condition << '\n'; \
      return 1;                                                                                  \
    }                                                                                            \
  } while (false)

#define CUDA_CHECK(expression)                                               \
  do {                                                                       \
    const cudaError_t cuda_check_status = (expression);                      \
    if (cuda_check_status != cudaSuccess) {                                  \
      std::cerr << "CUDA failure at " << __FILE__ << ':' << __LINE__ << ": " \
                << cudaGetErrorString(cuda_check_status) << '\n';            \
      return 1;                                                              \
    }                                                                        \
  } while (false)

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { allocate(count); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      swap(other);
    }
    return *this;
  }
  ~DeviceBuffer() { release(); }

  void allocate(std::size_t count) {
    release();
    count_ = count;
    if (count != 0u) {
      if (cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)) != cudaSuccess) {
        pointer_ = nullptr;
        count_ = 0u;
      }
    }
  }

  [[nodiscard]] T* get() const { return pointer_; }
  [[nodiscard]] std::size_t size() const { return count_; }

  cudaError_t copy_from(const T* source, std::size_t count, cudaStream_t stream = nullptr) const {
    if (count > count_) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(pointer_, source, count * sizeof(T), cudaMemcpyHostToDevice, stream);
  }

  cudaError_t copy_to(T* destination, std::size_t count, cudaStream_t stream = nullptr) const {
    if (count > count_) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(destination, pointer_, count * sizeof(T), cudaMemcpyDeviceToHost,
                           stream);
  }

 private:
  void release() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
    pointer_ = nullptr;
    count_ = 0u;
  }

  void swap(DeviceBuffer& other) noexcept {
    std::swap(pointer_, other.pointer_);
    std::swap(count_, other.count_);
  }

  T* pointer_ = nullptr;
  std::size_t count_ = 0u;
};

struct Snapshot {
  std::vector<std::uint8_t> active;
  std::vector<gpuxtb_status_t> statuses;
  std::vector<std::uint64_t> failures;
  std::uint64_t plan_failure = 0u;
  std::uint32_t sequence_active = 0u;
};

struct Fixture {
  explicit Fixture(std::size_t batch)
      : batch_size(batch),
        iterations(batch),
        statuses(batch),
        converged(batch),
        geometry_epoch(1u),
        geometry_generations(batch),
        eligible(batch),
        warm_start_generations(batch),
        cache_bindings(1u),
        active(batch),
        pending_statuses(batch),
        failures(batch),
        plan_failure(1u),
        sequence_active(1u),
        stage_codes(batch),
        stage_status_codes(batch),
        device_error(1u),
        stage_sequence(1u) {
    policy = {static_cast<std::int64_t>(batch), 8u, kPlanToken};
    state = {iterations.get(), statuses.get(), converged.get(), static_cast<std::int64_t>(batch),
             kPlanToken};
    provenance = {nullptr, 0, 0u, nullptr, 0, 0u, kPlanToken};
    ledger = {active.get(),
              pending_statuses.get(),
              failures.get(),
              plan_failure.get(),
              sequence_active.get(),
              static_cast<std::int64_t>(batch),
              1,
              kPlanToken};
  }

  bool valid() const {
    return iterations.get() != nullptr && statuses.get() != nullptr && converged.get() != nullptr &&
           geometry_epoch.get() != nullptr && geometry_generations.get() != nullptr &&
           eligible.get() != nullptr && warm_start_generations.get() != nullptr &&
           cache_bindings.get() != nullptr && active.get() != nullptr &&
           pending_statuses.get() != nullptr && failures.get() != nullptr &&
           plan_failure.get() != nullptr && sequence_active.get() != nullptr &&
           stage_codes.get() != nullptr && stage_status_codes.get() != nullptr &&
           device_error.get() != nullptr && stage_sequence.get() != nullptr;
  }

  cudaError_t install_state(const std::vector<std::uint64_t>& host_iterations,
                            const std::vector<gpuxtb_status_t>& host_statuses,
                            const std::vector<std::uint8_t>& host_converged,
                            cudaStream_t stream = nullptr) const {
    cudaError_t status = iterations.copy_from(host_iterations.data(), batch_size, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status = statuses.copy_from(host_statuses.data(), batch_size, stream);
    if (status != cudaSuccess) {
      return status;
    }
    return converged.copy_from(host_converged.data(), batch_size, stream);
  }

  cudaError_t install_per_system_provenance(
      const std::vector<std::uint64_t>& host_geometry_generations,
      const std::vector<std::uint64_t>& host_warm_start_generations,
      cudaStream_t stream = nullptr) {
    cudaError_t status =
        geometry_generations.copy_from(host_geometry_generations.data(), batch_size, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status =
        warm_start_generations.copy_from(host_warm_start_generations.data(), batch_size, stream);
    if (status != cudaSuccess) {
      return status;
    }
    const Gfn2GeometryCacheProvenanceView view{
        Gfn2PlanMemorySpace::kCudaDevice,
        Gfn2GenerationScope::kPerSystem,
        kPlanToken,
        0u,
        static_cast<std::int64_t>(batch_size),
        static_cast<std::int64_t>(batch_size),
        geometry_generations.get(),
    };
    const Gfn2SccCacheProvenanceBinding binding{view, Gfn2SccStageId::kGeometry, 0u};
    status = cache_bindings.copy_from(&binding, 1u, stream);
    if (status != cudaSuccess) {
      return status;
    }
    provenance = {cache_bindings.get(),
                  1,
                  kGeometryGeneration,
                  warm_start_generations.get(),
                  static_cast<std::int64_t>(batch_size),
                  kWarmStartGeneration,
                  kPlanToken};
    return cudaSuccess;
  }

  cudaError_t install_geometry_transaction(
      std::uint64_t epoch, const std::vector<std::uint64_t>& committed,
      const std::vector<std::uint8_t>& host_eligible, cudaStream_t stream = nullptr) const {
    if (committed.size() != batch_size || host_eligible.size() != batch_size) {
      return cudaErrorInvalidValue;
    }
    cudaError_t status = geometry_epoch.copy_from(&epoch, 1u, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status = geometry_generations.copy_from(committed.data(), batch_size, stream);
    if (status != cudaSuccess) {
      return status;
    }
    return eligible.copy_from(host_eligible.data(), batch_size, stream);
  }

  [[nodiscard]] Gfn2GeometryEpochConsumerDevice geometry_consumer() const {
    return {Gfn2GeometryEpochDevice{geometry_epoch.get(), 1, kPlanToken},
            geometry_generations.get(), eligible.get(), static_cast<std::int64_t>(batch_size),
            kPlanToken};
  }

  cudaError_t install_batch_provenance(std::uint64_t generation, cudaStream_t stream = nullptr) {
    const Gfn2GeometryCacheProvenanceView view{
        Gfn2PlanMemorySpace::kCudaDevice,
        Gfn2GenerationScope::kBatch,
        kPlanToken,
        generation,
        static_cast<std::int64_t>(batch_size),
        0,
        nullptr,
    };
    const Gfn2SccCacheProvenanceBinding binding{view, Gfn2SccStageId::kGeometry, 0u};
    const cudaError_t status = cache_bindings.copy_from(&binding, 1u, stream);
    if (status != cudaSuccess) {
      return status;
    }
    provenance = {cache_bindings.get(), 1, kGeometryGeneration, nullptr, 0, 0u, kPlanToken};
    return cudaSuccess;
  }

  cudaError_t install_stage(const std::vector<std::uint32_t>& codes, std::uint32_t first_error,
                            std::uint32_t sequence, cudaStream_t stream = nullptr) const {
    cudaError_t status = stage_codes.copy_from(codes.data(), batch_size, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status = device_error.copy_from(&first_error, 1u, stream);
    if (status != cudaSuccess) {
      return status;
    }
    return stage_sequence.copy_from(&sequence, 1u, stream);
  }

  cudaError_t snapshot(Snapshot& result, cudaStream_t stream = nullptr) const {
    result.active.resize(batch_size);
    result.statuses.resize(batch_size);
    result.failures.resize(batch_size);
    cudaError_t status = active.copy_to(result.active.data(), batch_size, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status = pending_statuses.copy_to(result.statuses.data(), batch_size, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status = failures.copy_to(result.failures.data(), batch_size, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status = plan_failure.copy_to(&result.plan_failure, 1u, stream);
    if (status != cudaSuccess) {
      return status;
    }
    return sequence_active.copy_to(&result.sequence_active, 1u, stream);
  }

  std::size_t batch_size;
  DeviceBuffer<std::uint64_t> iterations;
  DeviceBuffer<gpuxtb_status_t> statuses;
  DeviceBuffer<std::uint8_t> converged;
  DeviceBuffer<std::uint64_t> geometry_epoch;
  DeviceBuffer<std::uint64_t> geometry_generations;
  DeviceBuffer<std::uint8_t> eligible;
  DeviceBuffer<std::uint64_t> warm_start_generations;
  DeviceBuffer<Gfn2SccCacheProvenanceBinding> cache_bindings;
  DeviceBuffer<std::uint8_t> active;
  DeviceBuffer<gpuxtb_status_t> pending_statuses;
  DeviceBuffer<std::uint64_t> failures;
  DeviceBuffer<std::uint64_t> plan_failure;
  DeviceBuffer<std::uint32_t> sequence_active;
  DeviceBuffer<std::uint32_t> stage_codes;
  DeviceBuffer<gpuxtb_status_t> stage_status_codes;
  DeviceBuffer<std::uint32_t> device_error;
  DeviceBuffer<std::uint32_t> stage_sequence;
  Gfn2SccIterationDevicePolicy policy{};
  Gfn2SccIterationDeviceStateInput state{};
  Gfn2SccIterationDeviceProvenance provenance{};
  Gfn2SccIterationDeviceLedger ledger{};
};

std::uint64_t record(Gfn2SccStageId stage, std::uint32_t code) {
  return gpuxtb::detail::cuda::gfn2_scc_stage_failure_record(stage, code);
}

int derive_and_snapshot(Fixture& fixture, Snapshot& snapshot, cudaStream_t stream = nullptr) {
  CUDA_CHECK(gpuxtb::detail::cuda::derive_gfn2_scc_iteration_activity_cuda(
      fixture.policy, fixture.state, fixture.provenance, fixture.ledger, stream));
  CUDA_CHECK(fixture.snapshot(snapshot, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return 0;
}

int test_activity_for_batch(std::size_t batch_size) {
  Fixture fixture(batch_size);
  CHECK(fixture.valid());
  std::vector<std::uint64_t> iterations(batch_size, 0u);
  std::vector<gpuxtb_status_t> statuses(batch_size, GPUXTB_STATUS_SUCCESS);
  std::vector<std::uint8_t> converged(batch_size, 0u);
  std::vector<std::uint64_t> geometry(batch_size, kGeometryGeneration);
  std::vector<std::uint64_t> warm(batch_size, kWarmStartGeneration);
  if (batch_size > 1u) {
    converged[1] = 1u;
  }
  if (batch_size > 2u) {
    statuses[2] = GPUXTB_STATUS_INTERNAL_ERROR;
  }
  if (batch_size > 3u) {
    iterations[3] = fixture.policy.maximum_iterations;
  }
  if (batch_size > 4u) {
    geometry[4] = kGeometryGeneration - 1u;
  }
  if (batch_size > 5u) {
    warm[5] = kWarmStartGeneration - 1u;
  }
  if (batch_size > 6u) {
    converged[6] = 1u;
    geometry[6] = std::numeric_limits<std::uint64_t>::max();
    warm[6] = std::numeric_limits<std::uint64_t>::max();
  }
  CUDA_CHECK(fixture.install_state(iterations, statuses, converged));
  CUDA_CHECK(fixture.install_per_system_provenance(geometry, warm));
  Snapshot snapshot;
  CHECK(derive_and_snapshot(fixture, snapshot) == 0);
  CHECK(snapshot.plan_failure == 0u);
  CHECK(snapshot.sequence_active == 1u);
  for (std::size_t system = 0u; system < batch_size; ++system) {
    const bool expected_active = system != 1u && system != 2u && system != 3u && system != 4u &&
                                 system != 5u && system != 6u;
    CHECK(snapshot.active[system] == (expected_active ? 1u : 0u));
    if (system == 4u) {
      CHECK(snapshot.failures[system] ==
            record(Gfn2SccStageId::kGeometry,
                   static_cast<std::uint32_t>(Gfn2SccIterationControlCode::kStaleGeneration)));
      CHECK(snapshot.statuses[system] == GPUXTB_STATUS_INTERNAL_ERROR);
    } else if (system == 5u) {
      CHECK(snapshot.failures[system] ==
            record(Gfn2SccStageId::kWarmStartProvenance,
                   static_cast<std::uint32_t>(Gfn2SccIterationControlCode::kStaleGeneration)));
      CHECK(snapshot.statuses[system] == GPUXTB_STATUS_INTERNAL_ERROR);
    } else {
      CHECK(snapshot.failures[system] == 0u);
    }
  }
  return 0;
}

int test_batch_provenance_and_embedded_alias() {
  Fixture fixture(8u);
  CHECK(fixture.valid());
  std::vector<std::uint64_t> iterations(8u, 0u);
  std::vector<gpuxtb_status_t> statuses(8u, GPUXTB_STATUS_SUCCESS);
  std::vector<std::uint8_t> converged(8u, 0u);
  CUDA_CHECK(fixture.install_state(iterations, statuses, converged));
  CUDA_CHECK(fixture.install_batch_provenance(kGeometryGeneration - 1u));
  Snapshot stale;
  CHECK(derive_and_snapshot(fixture, stale) == 0);
  CHECK(stale.sequence_active == 0u);
  CHECK(stale.plan_failure ==
        record(Gfn2SccStageId::kGeometry,
               static_cast<std::uint32_t>(Gfn2SccIterationControlCode::kStaleGeneration)));
  CHECK(std::all_of(stale.active.begin(), stale.active.end(),
                    [](std::uint8_t value) { return value == 0u; }));

  std::fill(converged.begin(), converged.end(), 1u);
  CUDA_CHECK(fixture.install_state(iterations, statuses, converged));
  Snapshot inactive_stale;
  CHECK(derive_and_snapshot(fixture, inactive_stale) == 0);
  CHECK(inactive_stale.sequence_active == 1u);
  CHECK(inactive_stale.plan_failure == 0u);

  std::fill(converged.begin(), converged.end(), 0u);
  CUDA_CHECK(fixture.install_state(iterations, statuses, converged));
  const Gfn2GeometryCacheProvenanceView aliased_view{
      Gfn2PlanMemorySpace::kCudaDevice, Gfn2GenerationScope::kPerSystem, kPlanToken, 0u, 8, 8,
      fixture.failures.get(),
  };
  const Gfn2SccCacheProvenanceBinding aliased_binding{aliased_view, Gfn2SccStageId::kGeometry, 0u};
  CUDA_CHECK(fixture.cache_bindings.copy_from(&aliased_binding, 1u));
  fixture.provenance = {
      fixture.cache_bindings.get(), 1, kGeometryGeneration, nullptr, 0, 0u, kPlanToken};
  Snapshot aliased;
  CHECK(derive_and_snapshot(fixture, aliased) == 0);
  CHECK(aliased.sequence_active == 0u);
  CHECK(aliased.plan_failure ==
        record(Gfn2SccStageId::kGeometry,
               static_cast<std::uint32_t>(Gfn2SccIterationControlCode::kInvalidProvenance)));

  const Gfn2GeometryCacheProvenanceView cross_plan_view{
      Gfn2PlanMemorySpace::kCudaDevice,
      Gfn2GenerationScope::kPerSystem,
      kPlanToken + 1u,
      0u,
      8,
      8,
      fixture.geometry_generations.get(),
  };
  const Gfn2SccCacheProvenanceBinding cross_plan_binding{cross_plan_view, Gfn2SccStageId::kGeometry,
                                                         0u};
  CUDA_CHECK(fixture.cache_bindings.copy_from(&cross_plan_binding, 1u));
  fixture.provenance = {
      fixture.cache_bindings.get(), 1, kGeometryGeneration, nullptr, 0, 0u, kPlanToken};
  Snapshot cross_plan;
  CHECK(derive_and_snapshot(fixture, cross_plan) == 0);
  CHECK(cross_plan.sequence_active == 0u);
  CHECK(cross_plan.plan_failure ==
        record(Gfn2SccStageId::kGeometry,
               static_cast<std::uint32_t>(Gfn2SccIterationControlCode::kCrossPlan)));

  const Gfn2GeometryCacheProvenanceView valid_view{
      Gfn2PlanMemorySpace::kCudaDevice,
      Gfn2GenerationScope::kBatch,
      kPlanToken,
      kGeometryGeneration,
      8,
      0,
      nullptr,
  };
  const Gfn2SccCacheProvenanceBinding invalid_owner_binding{
      valid_view,
      static_cast<Gfn2SccStageId>(static_cast<std::uint32_t>(Gfn2SccStageId::kPeriodicRawEnergy) +
                                  1u),
      0u};
  CUDA_CHECK(fixture.cache_bindings.copy_from(&invalid_owner_binding, 1u));
  Snapshot invalid_owner;
  CHECK(derive_and_snapshot(fixture, invalid_owner) == 0);
  CHECK(invalid_owner.sequence_active == 0u);
  CHECK(invalid_owner.plan_failure ==
        record(Gfn2SccStageId::kActivity,
               static_cast<std::uint32_t>(Gfn2SccIterationControlCode::kInvalidProvenance)));
  return 0;
}

int test_peer_then_plan_and_first_record_sticky() {
  Fixture fixture(8u);
  CHECK(fixture.valid());
  std::vector<std::uint64_t> iterations(8u, 0u);
  std::vector<gpuxtb_status_t> statuses(8u, GPUXTB_STATUS_SUCCESS);
  std::vector<std::uint8_t> converged(8u, 0u);
  CUDA_CHECK(fixture.install_state(iterations, statuses, converged));
  Snapshot initial;
  CHECK(derive_and_snapshot(fixture, initial) == 0);

  std::vector<std::uint32_t> codes(8u, 0u);
  codes[0] = 5u;
  CUDA_CHECK(fixture.install_stage(codes, 5u, 1u));
  const Gfn2SccStageDeviceReport peer_report{
      Gfn2SccStageId::kHamiltonian,
      Gfn2SccStageCodeFormat::kUint32Error,
      fixture.stage_codes.get(),
      8,
      fixture.device_error.get(),
      1,
      fixture.stage_sequence.get(),
      1,
      std::uint64_t{1} << 5u,
      GPUXTB_STATUS_INTERNAL_ERROR,
      kPlanToken,
  };
  CUDA_CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(peer_report, fixture.ledger));
  CUDA_CHECK(cudaDeviceSynchronize());
  Snapshot peer;
  CUDA_CHECK(fixture.snapshot(peer));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(peer.sequence_active == 1u);
  CHECK(peer.active[0] == 0u);
  CHECK(peer.failures[0] == record(Gfn2SccStageId::kHamiltonian, 5u));

  codes.assign(8u, 0u);
  codes[1] = 6u;
  codes[2] = 2u;
  CUDA_CHECK(fixture.install_stage(codes, 6u, 0u));
  const Gfn2SccStageDeviceReport plan_report{
      Gfn2SccStageId::kEigensolver,
      Gfn2SccStageCodeFormat::kUint32Error,
      fixture.stage_codes.get(),
      8,
      fixture.device_error.get(),
      1,
      fixture.stage_sequence.get(),
      1,
      std::uint64_t{1} << 6u,
      GPUXTB_STATUS_EIGENSOLVER_FAILED,
      kPlanToken,
  };
  CUDA_CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(plan_report, fixture.ledger));
  Snapshot plan;
  CUDA_CHECK(fixture.snapshot(plan));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(plan.sequence_active == 0u);
  CHECK(plan.plan_failure == record(Gfn2SccStageId::kEigensolver, 2u));
  CHECK(plan.failures[0] == record(Gfn2SccStageId::kHamiltonian, 5u));
  CHECK(std::all_of(plan.active.begin(), plan.active.end(),
                    [](std::uint8_t value) { return value == 0u; }));

  codes.assign(8u, 77u);
  CUDA_CHECK(fixture.install_stage(codes, 77u, 1u));
  CUDA_CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(peer_report, fixture.ledger));
  Snapshot still_first;
  CUDA_CHECK(fixture.snapshot(still_first));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(still_first.plan_failure == plan.plan_failure);
  return 0;
}

int test_device_first_plan_precedence() {
  Fixture fixture(8u);
  CHECK(fixture.valid());
  std::vector<std::uint64_t> iterations(8u, 0u);
  std::vector<gpuxtb_status_t> statuses(8u, GPUXTB_STATUS_SUCCESS);
  std::vector<std::uint8_t> converged(8u, 0u);
  std::vector<std::uint32_t> codes(8u, 0u);
  CUDA_CHECK(fixture.install_state(iterations, statuses, converged));
  Snapshot ignored;
  CHECK(derive_and_snapshot(fixture, ignored) == 0);

  // The device scalar is the sticky first plan error.  A lower indexed code
  // is a later diagnostic and must not replace it.
  codes[0] = 2u;
  CUDA_CHECK(fixture.install_stage(codes, 77u, 1u));
  const Gfn2SccStageDeviceReport report{
      Gfn2SccStageId::kEigensolver,
      Gfn2SccStageCodeFormat::kUint32Error,
      fixture.stage_codes.get(),
      8,
      fixture.device_error.get(),
      1,
      fixture.stage_sequence.get(),
      1,
      std::uint64_t{1} << 5u,
      GPUXTB_STATUS_EIGENSOLVER_FAILED,
      kPlanToken,
  };
  CUDA_CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(report, fixture.ledger));
  Snapshot indexed;
  CUDA_CHECK(fixture.snapshot(indexed));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(indexed.plan_failure == record(Gfn2SccStageId::kEigensolver, 77u));

  // A closed latch provides only less-specific plan evidence; retain the
  // concrete device code when both are present.
  CHECK(derive_and_snapshot(fixture, ignored) == 0);
  codes.assign(8u, 0u);
  CUDA_CHECK(fixture.install_stage(codes, 91u, 0u));
  CUDA_CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(report, fixture.ledger));
  Snapshot latched;
  CUDA_CHECK(fixture.snapshot(latched));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(latched.plan_failure == record(Gfn2SccStageId::kEigensolver, 91u));
  return 0;
}

int test_device_code_roles_and_raw_energy_stages() {
  Fixture fixture(8u);
  CHECK(fixture.valid());
  std::vector<std::uint64_t> iterations(8u, 0u);
  std::vector<gpuxtb_status_t> statuses(8u, GPUXTB_STATUS_SUCCESS);
  std::vector<std::uint8_t> converged(8u, 0u);
  CUDA_CHECK(fixture.install_state(iterations, statuses, converged));

  const Gfn2SccStageDeviceReport default_report{};
  CHECK(default_report.device_code_role == Gfn2SccStageDeviceCodeRole::kMixedFirstError);

  struct CollisionCase {
    Gfn2SccStageId stage;
    std::uint32_t code;
  };
  const CollisionCase collisions[] = {
      {Gfn2SccStageId::kES2Potential, 8u},
      {Gfn2SccStageId::kES2RawEnergy, 8u},
      {Gfn2SccStageId::kAES2Potential, 1u},
      {Gfn2SccStageId::kAES2RawEnergy, 1u},
  };
  for (const CollisionCase collision : collisions) {
    Snapshot ignored;
    CHECK(derive_and_snapshot(fixture, ignored) == 0);
    std::vector<std::uint32_t> codes(8u, 0u);
    codes[0] = collision.code;
    CUDA_CHECK(fixture.install_stage(codes, collision.code, 1u));
    const Gfn2SccStageDeviceReport report{
        collision.stage,
        Gfn2SccStageCodeFormat::kUint32Error,
        fixture.stage_codes.get(),
        8,
        fixture.device_error.get(),
        1,
        fixture.stage_sequence.get(),
        1,
        std::uint64_t{1} << collision.code,
        GPUXTB_STATUS_INTERNAL_ERROR,
        kPlanToken,
        Gfn2SccStageDeviceCodeRole::kPlanOnly,
    };
    CUDA_CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(report, fixture.ledger));
    Snapshot normalized;
    CUDA_CHECK(fixture.snapshot(normalized));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(normalized.sequence_active == 0u);
    CHECK(normalized.plan_failure == record(collision.stage, collision.code));
    CHECK(std::all_of(normalized.active.begin(), normalized.active.end(),
                      [](std::uint8_t active) { return active == 0u; }));
    CHECK(std::all_of(normalized.failures.begin(), normalized.failures.end(),
                      [](std::uint64_t failure) { return failure == 0u; }));
  }

  // The plan-only role applies only to the scalar. Per-system peer codes keep
  // their existing mask-based isolation and retain the raw-energy stage ID.
  Snapshot ignored;
  CHECK(derive_and_snapshot(fixture, ignored) == 0);
  std::vector<std::uint32_t> codes(8u, 0u);
  codes[2] = 8u;
  CUDA_CHECK(fixture.install_stage(codes, 0u, 1u));
  const Gfn2SccStageDeviceReport plan_only_system_peer{
      Gfn2SccStageId::kES2RawEnergy,
      Gfn2SccStageCodeFormat::kUint32Error,
      fixture.stage_codes.get(),
      8,
      fixture.device_error.get(),
      1,
      fixture.stage_sequence.get(),
      1,
      std::uint64_t{1} << 8u,
      GPUXTB_STATUS_INTERNAL_ERROR,
      kPlanToken,
      Gfn2SccStageDeviceCodeRole::kPlanOnly,
  };
  CUDA_CHECK(
      gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(plan_only_system_peer, fixture.ledger));
  Snapshot system_peer;
  CUDA_CHECK(fixture.snapshot(system_peer));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(system_peer.sequence_active == 1u);
  CHECK(system_peer.plan_failure == 0u);
  CHECK(system_peer.active[2] == 0u);
  CHECK(system_peer.failures[2] == record(Gfn2SccStageId::kES2RawEnergy, 8u));

  // An explicit mixed-first-error role is the legacy behavior: a matching
  // device scalar localizes the peer instead of closing the batch sequence.
  CHECK(derive_and_snapshot(fixture, ignored) == 0);
  codes.assign(8u, 0u);
  codes[3] = 5u;
  CUDA_CHECK(fixture.install_stage(codes, 5u, 1u));
  const Gfn2SccStageDeviceReport mixed_report{
      Gfn2SccStageId::kHamiltonian,
      Gfn2SccStageCodeFormat::kUint32Error,
      fixture.stage_codes.get(),
      8,
      fixture.device_error.get(),
      1,
      fixture.stage_sequence.get(),
      1,
      std::uint64_t{1} << 5u,
      GPUXTB_STATUS_INTERNAL_ERROR,
      kPlanToken,
      Gfn2SccStageDeviceCodeRole::kMixedFirstError,
  };
  CUDA_CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(mixed_report, fixture.ledger));
  Snapshot mixed;
  CUDA_CHECK(fixture.snapshot(mixed));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(mixed.sequence_active == 1u);
  CHECK(mixed.plan_failure == 0u);
  CHECK(mixed.active[3] == 0u);
  CHECK(mixed.failures[3] == record(Gfn2SccStageId::kHamiltonian, 5u));
  return 0;
}

int test_unknown_unlocalized_and_inactive_poison() {
  Fixture fixture(8u);
  CHECK(fixture.valid());
  std::vector<std::uint64_t> iterations(8u, 0u);
  std::vector<gpuxtb_status_t> statuses(8u, GPUXTB_STATUS_SUCCESS);
  std::vector<std::uint8_t> converged(8u, 0u);
  CUDA_CHECK(fixture.install_state(iterations, statuses, converged));
  Snapshot ignored;
  CHECK(derive_and_snapshot(fixture, ignored) == 0);

  std::vector<std::uint32_t> codes(8u, 0u);
  codes[0] = 65u;
  CUDA_CHECK(fixture.install_stage(codes, 65u, 1u));
  Gfn2SccStageDeviceReport report{
      Gfn2SccStageId::kDensity,
      Gfn2SccStageCodeFormat::kUint32Error,
      fixture.stage_codes.get(),
      8,
      fixture.device_error.get(),
      1,
      fixture.stage_sequence.get(),
      1,
      std::uint64_t{1} << 5u,
      GPUXTB_STATUS_INTERNAL_ERROR,
      kPlanToken,
  };
  CUDA_CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(report, fixture.ledger));
  Snapshot unknown;
  CUDA_CHECK(fixture.snapshot(unknown));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(unknown.plan_failure == record(Gfn2SccStageId::kDensity, 65u));

  CHECK(derive_and_snapshot(fixture, ignored) == 0);
  codes.assign(8u, 0u);
  codes[1] = 5u;
  CUDA_CHECK(fixture.install_stage(codes, 5u, 0u));
  CUDA_CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(report, fixture.ledger));
  Snapshot closed_latch;
  CUDA_CHECK(fixture.snapshot(closed_latch));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(closed_latch.plan_failure ==
        record(Gfn2SccStageId::kDensity, gpuxtb::detail::cuda::kGfn2SccStageSequenceClosedCode));

  CHECK(derive_and_snapshot(fixture, ignored) == 0);
  codes.assign(8u, 0u);
  CUDA_CHECK(fixture.install_stage(codes, 5u, 1u));
  CUDA_CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(report, fixture.ledger));
  Snapshot unlocalized;
  CUDA_CHECK(fixture.snapshot(unlocalized));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(unlocalized.plan_failure ==
        record(Gfn2SccStageId::kDensity, gpuxtb::detail::cuda::kGfn2SccStageUnlocalizedPeerCode));

  CHECK(derive_and_snapshot(fixture, ignored) == 0);
  codes.assign(8u, 0u);
  codes[2] = 6u;
  CUDA_CHECK(fixture.install_stage(codes, 5u, 1u));
  report.peer_error_mask = (std::uint64_t{1} << 5u) | (std::uint64_t{1} << 6u);
  CUDA_CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(report, fixture.ledger));
  Snapshot mismatched_peer;
  CUDA_CHECK(fixture.snapshot(mismatched_peer));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(mismatched_peer.plan_failure ==
        record(Gfn2SccStageId::kDensity, gpuxtb::detail::cuda::kGfn2SccStageUnlocalizedPeerCode));

  converged[0] = 1u;
  CUDA_CHECK(fixture.install_state(iterations, statuses, converged));
  CHECK(derive_and_snapshot(fixture, ignored) == 0);
  codes.assign(8u, 0u);
  codes[0] = 65u;
  CUDA_CHECK(fixture.install_stage(codes, 0u, 1u));
  CUDA_CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(report, fixture.ledger));
  Snapshot poison;
  CUDA_CHECK(fixture.snapshot(poison));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(poison.sequence_active == 1u);
  CHECK(poison.plan_failure == 0u);
  CHECK(poison.failures[0] == 0u);
  CHECK(poison.active[0] == 0u);
  CHECK(poison.active[1] == 1u);
  return 0;
}

int test_inactive_uninitialized_inputs_are_not_read() {
  Fixture fixture(8u);
  CHECK(fixture.valid());
  std::vector<std::uint64_t> iterations(8u, 0u);
  std::vector<gpuxtb_status_t> statuses(8u, GPUXTB_STATUS_SUCCESS);
  std::vector<std::uint8_t> converged(8u, 0u);
  converged[0] = 1u;
  CUDA_CHECK(fixture.install_state(iterations, statuses, converged));

  for (std::size_t system = 1u; system < 8u; ++system) {
    CUDA_CHECK(cudaMemcpy(fixture.geometry_generations.get() + system, &kGeometryGeneration,
                          sizeof(kGeometryGeneration), cudaMemcpyHostToDevice));
  }
  const Gfn2GeometryCacheProvenanceView view{
      Gfn2PlanMemorySpace::kCudaDevice,   Gfn2GenerationScope::kPerSystem, kPlanToken, 0u, 8, 8,
      fixture.geometry_generations.get(),
  };
  const Gfn2SccCacheProvenanceBinding binding{view, Gfn2SccStageId::kGeometry, 0u};
  CUDA_CHECK(fixture.cache_bindings.copy_from(&binding, 1u));
  fixture.provenance = {
      fixture.cache_bindings.get(), 1, kGeometryGeneration, nullptr, 0, 0u, kPlanToken};
  Snapshot derived;
  CHECK(derive_and_snapshot(fixture, derived) == 0);
  CHECK(derived.sequence_active == 1u);
  CHECK(derived.active[0] == 0u);
  CHECK(derived.failures[0] == 0u);
  for (std::size_t system = 1u; system < 8u; ++system) {
    CHECK(derived.active[system] == 1u);
  }

  const std::uint32_t zero = 0u;
  for (std::size_t system = 1u; system < 8u; ++system) {
    CUDA_CHECK(cudaMemcpy(fixture.stage_codes.get() + system, &zero, sizeof(zero),
                          cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(fixture.device_error.copy_from(&zero, 1u));
  const std::uint32_t one = 1u;
  CUDA_CHECK(fixture.stage_sequence.copy_from(&one, 1u));
  const Gfn2SccStageDeviceReport report{
      Gfn2SccStageId::kDensity,
      Gfn2SccStageCodeFormat::kUint32Error,
      fixture.stage_codes.get(),
      8,
      fixture.device_error.get(),
      1,
      fixture.stage_sequence.get(),
      1,
      std::uint64_t{1} << 5u,
      GPUXTB_STATUS_INTERNAL_ERROR,
      kPlanToken,
  };
  CUDA_CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(report, fixture.ledger));
  Snapshot normalized;
  CUDA_CHECK(fixture.snapshot(normalized));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(normalized.sequence_active == 1u);
  CHECK(normalized.plan_failure == 0u);
  CHECK(normalized.active[0] == 0u);
  return 0;
}

int test_status_code_format() {
  Fixture fixture(8u);
  CHECK(fixture.valid());
  std::vector<std::uint64_t> iterations(8u, 0u);
  std::vector<gpuxtb_status_t> statuses(8u, GPUXTB_STATUS_SUCCESS);
  std::vector<std::uint8_t> converged(8u, 0u);
  CUDA_CHECK(fixture.install_state(iterations, statuses, converged));
  Snapshot initial;
  CHECK(derive_and_snapshot(fixture, initial) == 0);
  std::vector<gpuxtb_status_t> codes(8u, GPUXTB_STATUS_SUCCESS);
  codes[3] = GPUXTB_STATUS_INTERNAL_ERROR;
  CUDA_CHECK(fixture.stage_status_codes.copy_from(codes.data(), codes.size()));
  const std::uint32_t device = GPUXTB_STATUS_INTERNAL_ERROR;
  const std::uint32_t sequence = 1u;
  CUDA_CHECK(fixture.device_error.copy_from(&device, 1u));
  CUDA_CHECK(fixture.stage_sequence.copy_from(&sequence, 1u));
  const Gfn2SccStageDeviceReport report{
      Gfn2SccStageId::kPeriodicPotential,
      Gfn2SccStageCodeFormat::kGpuxtbStatus,
      fixture.stage_status_codes.get(),
      8,
      fixture.device_error.get(),
      1,
      fixture.stage_sequence.get(),
      1,
      std::uint64_t{1} << GPUXTB_STATUS_INTERNAL_ERROR,
      GPUXTB_STATUS_INTERNAL_ERROR,
      kPlanToken,
  };
  CUDA_CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(report, fixture.ledger));
  Snapshot snapshot;
  CUDA_CHECK(fixture.snapshot(snapshot));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(snapshot.sequence_active == 1u);
  CHECK(snapshot.active[3] == 0u);
  CHECK(snapshot.failures[3] ==
        record(Gfn2SccStageId::kPeriodicPotential, GPUXTB_STATUS_INTERNAL_ERROR));
  return 0;
}

int test_graph_replay_resets_control() {
  Fixture fixture(8u);
  CHECK(fixture.valid());
  std::vector<std::uint64_t> iterations(8u, 0u);
  std::vector<gpuxtb_status_t> statuses(8u, GPUXTB_STATUS_SUCCESS);
  std::vector<std::uint8_t> converged(8u, 0u);
  std::vector<std::uint32_t> codes(8u, 0u);
  CUDA_CHECK(fixture.install_state(iterations, statuses, converged));
  CUDA_CHECK(fixture.install_stage(codes, 0u, 1u));

  const Gfn2SccStageDeviceReport report{
      Gfn2SccStageId::kMixer,
      Gfn2SccStageCodeFormat::kUint32Error,
      fixture.stage_codes.get(),
      8,
      fixture.device_error.get(),
      1,
      fixture.stage_sequence.get(),
      1,
      std::uint64_t{1} << 5u,
      GPUXTB_STATUS_INTERNAL_ERROR,
      kPlanToken,
  };

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(gpuxtb::detail::cuda::derive_gfn2_scc_iteration_activity_cuda(
      fixture.policy, fixture.state, fixture.provenance, fixture.ledger, stream));
  CUDA_CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(report, fixture.ledger, stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0u));

  codes[0] = 5u;
  CUDA_CHECK(fixture.install_stage(codes, 5u, 1u, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  Snapshot first;
  CUDA_CHECK(fixture.snapshot(first, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(first.sequence_active == 1u);
  CHECK(first.active[0] == 0u);
  CHECK(first.failures[0] == record(Gfn2SccStageId::kMixer, 5u));

  codes.assign(8u, 0u);
  converged[1] = 1u;
  CUDA_CHECK(fixture.install_state(iterations, statuses, converged, stream));
  CUDA_CHECK(fixture.install_stage(codes, 0u, 1u, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  Snapshot second;
  CUDA_CHECK(fixture.snapshot(second, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(second.sequence_active == 1u);
  CHECK(second.plan_failure == 0u);
  CHECK(second.active[0] == 1u);
  CHECK(second.failures[0] == 0u);
  CHECK(second.active[1] == 0u);

  CUDA_CHECK(fixture.install_stage(codes, 77u, 1u, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  Snapshot third;
  CUDA_CHECK(fixture.snapshot(third, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(third.sequence_active == 0u);
  CHECK(third.plan_failure == record(Gfn2SccStageId::kMixer, 77u));

  CUDA_CHECK(fixture.install_stage(codes, 0u, 1u, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  Snapshot fourth;
  CUDA_CHECK(fixture.snapshot(fourth, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(fourth.sequence_active == 1u);
  CHECK(fourth.plan_failure == 0u);
  CHECK(fourth.active[0] == 1u);

  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_device_epoch_graph_replay_and_fail_closed_gates() {
  constexpr std::size_t batch_size = 8u;
  Fixture fixture(batch_size);
  CHECK(fixture.valid());
  std::vector<std::uint64_t> iterations(batch_size, 0u);
  std::vector<gpuxtb_status_t> statuses(batch_size, GPUXTB_STATUS_SUCCESS);
  std::vector<std::uint8_t> converged(batch_size, 0u);
  std::vector<std::uint64_t> committed(batch_size, kGeometryGeneration);
  std::vector<std::uint64_t> warm(batch_size, kWarmStartGeneration);
  std::vector<std::uint8_t> eligible(batch_size, 1u);
  CUDA_CHECK(fixture.install_state(iterations, statuses, converged));
  CUDA_CHECK(fixture.install_per_system_provenance(committed, warm));
  CUDA_CHECK(fixture.install_geometry_transaction(kGeometryGeneration, committed, eligible));

  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(gpuxtb::detail::cuda::derive_gfn2_scc_iteration_activity_cuda(
      fixture.policy, fixture.state, fixture.provenance, fixture.geometry_consumer(),
      fixture.ledger, stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0u));

  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  Snapshot first;
  CUDA_CHECK(fixture.snapshot(first, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(first.sequence_active == 1u);
  CHECK(first.plan_failure == 0u);
  CHECK(std::all_of(first.active.begin(), first.active.end(),
                    [](std::uint8_t value) { return value == 1u; }));

  const std::uint64_t next_epoch = kGeometryGeneration + 1u;
  committed.assign(batch_size, next_epoch);
  committed[1] = kGeometryGeneration;
  eligible[2] = 0u;
  CUDA_CHECK(fixture.install_geometry_transaction(next_epoch, committed, eligible, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  Snapshot mixed;
  CUDA_CHECK(fixture.snapshot(mixed, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(mixed.sequence_active == 1u);
  CHECK(mixed.plan_failure == 0u);
  CHECK(mixed.active[0] == 1u);
  CHECK(mixed.active[1] == 0u);
  CHECK(mixed.active[2] == 0u);
  for (const std::size_t system : {1u, 2u}) {
    CHECK(mixed.failures[system] ==
          record(Gfn2SccStageId::kGeometry,
                 static_cast<std::uint32_t>(Gfn2SccIterationControlCode::kStaleGeneration)));
    CHECK(mixed.statuses[system] == GPUXTB_STATUS_INTERNAL_ERROR);
  }

  committed.assign(batch_size, next_epoch);
  eligible.assign(batch_size, 1u);
  CUDA_CHECK(fixture.install_geometry_transaction(next_epoch, committed, eligible, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  Snapshot refreshed;
  CUDA_CHECK(fixture.snapshot(refreshed, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(refreshed.sequence_active == 1u);
  CHECK(refreshed.plan_failure == 0u);
  CHECK(std::all_of(refreshed.active.begin(), refreshed.active.end(),
                    [](std::uint8_t value) { return value == 1u; }));
  CHECK(std::all_of(refreshed.failures.begin(), refreshed.failures.end(),
                    [](std::uint64_t value) { return value == 0u; }));

  /* Malformed eligibility is plan-wide even when every malformed byte belongs
   * to an otherwise inactive peer. */
  converged[3] = 1u;
  statuses[4] = GPUXTB_STATUS_INTERNAL_ERROR;
  eligible[3] = 2u;
  eligible[4] = 2u;
  CUDA_CHECK(fixture.install_state(iterations, statuses, converged, stream));
  CUDA_CHECK(fixture.install_geometry_transaction(next_epoch, committed, eligible, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  Snapshot malformed;
  CUDA_CHECK(fixture.snapshot(malformed, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(malformed.sequence_active == 0u);
  CHECK(malformed.plan_failure ==
        record(Gfn2SccStageId::kActivity,
               static_cast<std::uint32_t>(Gfn2SccIterationControlCode::kInvalidProvenance)));
  CHECK(std::all_of(malformed.active.begin(), malformed.active.end(),
                    [](std::uint8_t value) { return value == 0u; }));

  eligible.assign(batch_size, 1u);
  CUDA_CHECK(fixture.install_geometry_transaction(0u, committed, eligible, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  Snapshot zero_epoch;
  CUDA_CHECK(fixture.snapshot(zero_epoch, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(zero_epoch.sequence_active == 0u);
  CHECK(zero_epoch.plan_failure == malformed.plan_failure);

  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));

  Gfn2GeometryEpochConsumerDevice cross_plan = fixture.geometry_consumer();
  cross_plan.plan_token ^= 1u;
  CHECK(gpuxtb::detail::cuda::derive_gfn2_scc_iteration_activity_cuda(
            fixture.policy, fixture.state, fixture.provenance, cross_plan, fixture.ledger) ==
        cudaErrorInvalidValue);

  converged[3] = 0u;
  statuses[4] = GPUXTB_STATUS_SUCCESS;
  CUDA_CHECK(fixture.install_state(iterations, statuses, converged));
  CUDA_CHECK(fixture.install_geometry_transaction(next_epoch, committed, eligible));
  CUDA_CHECK(fixture.active.copy_from(eligible.data(), eligible.size()));
  Gfn2GeometryEpochConsumerDevice exact_alias = fixture.geometry_consumer();
  exact_alias.eligible_mask = fixture.active.get();
  CUDA_CHECK(gpuxtb::detail::cuda::derive_gfn2_scc_iteration_activity_cuda(
      fixture.policy, fixture.state, fixture.provenance, exact_alias, fixture.ledger));
  Snapshot alias;
  CUDA_CHECK(fixture.snapshot(alias));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(alias.sequence_active == 1u);
  CHECK(std::all_of(alias.active.begin(), alias.active.end(),
                    [](std::uint8_t value) { return value == 1u; }));
  return 0;
}

int test_host_validation() {
  Fixture fixture(1u);
  CHECK(fixture.valid());
  std::vector<std::uint64_t> iterations(1u, 0u);
  std::vector<gpuxtb_status_t> statuses(1u, GPUXTB_STATUS_SUCCESS);
  std::vector<std::uint8_t> converged(1u, 0u);
  CUDA_CHECK(fixture.install_state(iterations, statuses, converged));
  statuses[0] = 99;
  CUDA_CHECK(fixture.install_state(iterations, statuses, converged));
  Snapshot invalid_state;
  CHECK(derive_and_snapshot(fixture, invalid_state) == 0);
  CHECK(invalid_state.sequence_active == 0u);
  CHECK(invalid_state.plan_failure ==
        record(Gfn2SccStageId::kActivity,
               static_cast<std::uint32_t>(Gfn2SccIterationControlCode::kInvalidState)));
  statuses[0] = GPUXTB_STATUS_SUCCESS;
  CUDA_CHECK(fixture.install_state(iterations, statuses, converged));
  Snapshot sentinel_before;
  CHECK(derive_and_snapshot(fixture, sentinel_before) == 0);
  Gfn2SccIterationDeviceLedger bad_ledger = fixture.ledger;
  bad_ledger.sequence_active = reinterpret_cast<std::uint32_t*>(fixture.plan_failure.get());
  CHECK(gpuxtb::detail::cuda::derive_gfn2_scc_iteration_activity_cuda(
            fixture.policy, fixture.state, fixture.provenance, bad_ledger) ==
        cudaErrorInvalidValue);

  Gfn2SccStageDeviceReport bad_report{};
  bad_report.stage = Gfn2SccStageId::kDensity;
  bad_report.peer_error_mask = 1u;
  bad_report.peer_failure_status = GPUXTB_STATUS_INTERNAL_ERROR;
  bad_report.plan_token = kPlanToken;
  CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(bad_report, fixture.ledger) ==
        cudaErrorInvalidValue);

  const std::vector<std::uint32_t> codes(1u, 77u);
  CUDA_CHECK(fixture.install_stage(codes, 77u, 1u));
  Gfn2SccStageDeviceReport invalid_stage_report{
      static_cast<Gfn2SccStageId>(static_cast<std::uint32_t>(Gfn2SccStageId::kPeriodicRawEnergy) +
                                  1u),
      Gfn2SccStageCodeFormat::kUint32Error,
      fixture.stage_codes.get(),
      1,
      fixture.device_error.get(),
      1,
      fixture.stage_sequence.get(),
      1,
      0u,
      GPUXTB_STATUS_INTERNAL_ERROR,
      kPlanToken,
  };

  Gfn2SccStageDeviceReport invalid_role_report = invalid_stage_report;
  invalid_role_report.stage = Gfn2SccStageId::kES2RawEnergy;
  invalid_role_report.device_code_role =
      static_cast<Gfn2SccStageDeviceCodeRole>(std::numeric_limits<std::uint32_t>::max());
  CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(invalid_role_report, fixture.ledger) ==
        cudaErrorInvalidValue);

  Gfn2SccStageDeviceReport missing_plan_scalar = invalid_stage_report;
  missing_plan_scalar.stage = Gfn2SccStageId::kES2RawEnergy;
  missing_plan_scalar.device_error = nullptr;
  missing_plan_scalar.device_error_elements = 0;
  missing_plan_scalar.device_code_role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
  CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(missing_plan_scalar, fixture.ledger) ==
        cudaErrorInvalidValue);

  // Capture makes the no-launch contract observable: rejecting a malformed
  // host descriptor must leave an empty graph and preserve the ledger.
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(invalid_stage_report, fixture.ledger,
                                                            stream) == cudaErrorInvalidValue);
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  std::size_t node_count = std::numeric_limits<std::size_t>::max();
  CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &node_count));
  CHECK(node_count == 0u);
  Snapshot sentinel_after;
  CUDA_CHECK(fixture.snapshot(sentinel_after, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(sentinel_after.active == sentinel_before.active);
  CHECK(sentinel_after.statuses == sentinel_before.statuses);
  CHECK(sentinel_after.failures == sentinel_before.failures);
  CHECK(sentinel_after.plan_failure == sentinel_before.plan_failure);
  CHECK(sentinel_after.sequence_active == sentinel_before.sequence_active);
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));

  invalid_stage_report.stage =
      static_cast<Gfn2SccStageId>(std::numeric_limits<std::uint32_t>::max());
  CHECK(gpuxtb::detail::cuda::normalize_gfn2_scc_stage_cuda(invalid_stage_report, fixture.ledger) ==
        cudaErrorInvalidValue);
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return 77;
  }
  for (const std::size_t batch_size : {1u, 8u, 32u, 128u}) {
    const int status = test_activity_for_batch(batch_size);
    if (status != 0) {
      return status;
    }
  }
  if (const int status = test_batch_provenance_and_embedded_alias(); status != 0) {
    return status;
  }
  if (const int status = test_peer_then_plan_and_first_record_sticky(); status != 0) {
    return status;
  }
  if (const int status = test_device_first_plan_precedence(); status != 0) {
    return status;
  }
  if (const int status = test_device_code_roles_and_raw_energy_stages(); status != 0) {
    return status;
  }
  if (const int status = test_unknown_unlocalized_and_inactive_poison(); status != 0) {
    return status;
  }
  if (const int status = test_status_code_format(); status != 0) {
    return status;
  }
  if (const int status = test_inactive_uninitialized_inputs_are_not_read(); status != 0) {
    return status;
  }
  if (const int status = test_graph_replay_resets_control(); status != 0) {
    return status;
  }
  if (const int status = test_device_epoch_graph_replay_and_fail_closed_gates(); status != 0) {
    return status;
  }
  return test_host_validation();
}
