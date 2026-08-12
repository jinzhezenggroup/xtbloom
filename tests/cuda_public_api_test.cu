#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "tests/support/gfn2_scc_test_case.hpp"
#include "xtbloom/xtbloom.h"

/* Some minimal CUDA toolkit packages omit cuda_profiler_api.h while retaining
 * the stable cudart entry points. Keep this non-default profiling test usable
 * with those packages without changing the production library surface. */
extern "C" cudaError_t CUDARTAPI cudaProfilerStart(void);
extern "C" cudaError_t CUDARTAPI cudaProfilerStop(void);

/* End-to-end contract test for the public single-flight CUDA transaction. */
namespace {

using xtbloom::test::gfn2::HostSccCase;
using xtbloom::test::gfn2::HostSccCaseOptions;
using xtbloom::test::gfn2::SmallSystemKind;

constexpr std::uint32_t kRequestedProperties =
    XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES | XTBLOOM_COMPUTE_ATOMIC_CHARGES;
constexpr std::uint32_t kResultFlagsCanary = UINT32_C(0xa55a39c6);
constexpr double kEnergyAbsoluteTolerance = 3.0e-8;
constexpr double kEnergyRelativeTolerance = 3.0e-8;
constexpr double kChargeAbsoluteTolerance = 1.0e-7;
constexpr double kChargeRelativeTolerance = 1.0e-7;
constexpr double kForceAbsoluteTolerance = 3.0e-7;
constexpr double kForceRelativeTolerance = 3.0e-7;
/* A blocked-stream enqueue should return promptly, but Compute Sanitizer can
 * make otherwise cheap host admission substantially slower. Keep one generous
 * watchdog that still releases the gate before joining a regressed submitter. */
constexpr auto kBlockedEnqueueWatchdog = std::chrono::seconds(30);

const char* g_scenario = "uninitialized";

#define CHECK(condition)                                                                        \
  do {                                                                                          \
    if (!(condition)) {                                                                         \
      std::fprintf(stderr, "public CUDA API check failed in %s at %s:%d: %s; %s\n", g_scenario, \
                   __FILE__, __LINE__, #condition, xtbloom_get_last_error());                   \
      return __LINE__;                                                                          \
    }                                                                                           \
  } while (false)

#define CUDA_CHECK(expression)                                                                 \
  do {                                                                                         \
    const cudaError_t cuda_status_ = (expression);                                             \
    if (cuda_status_ != cudaSuccess) {                                                         \
      std::fprintf(stderr, "public CUDA API runtime failure in %s at %s:%d: %s\n", g_scenario, \
                   __FILE__, __LINE__, cudaGetErrorString(cuda_status_));                      \
      return __LINE__;                                                                         \
    }                                                                                          \
  } while (false)

struct ContextDeleter {
  void operator()(xtbloom_context_t* context) const noexcept { xtbloom_context_destroy(context); }
};

using ContextHandle = std::unique_ptr<xtbloom_context_t, ContextDeleter>;

class StreamOwner {
 public:
  StreamOwner() noexcept = default;
  ~StreamOwner() {
    if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
  }
  StreamOwner(const StreamOwner&) = delete;
  StreamOwner& operator=(const StreamOwner&) = delete;

  cudaError_t create() noexcept {
    if (stream_ != nullptr) return cudaErrorInvalidResourceHandle;
    return cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  }

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

class EventOwner {
 public:
  EventOwner() noexcept = default;
  ~EventOwner() {
    if (event_ != nullptr) (void)cudaEventDestroy(event_);
  }
  EventOwner(const EventOwner&) = delete;
  EventOwner& operator=(const EventOwner&) = delete;

  cudaError_t create() noexcept {
    return event_ == nullptr ? cudaEventCreateWithFlags(&event_, cudaEventDisableTiming)
                             : cudaErrorInvalidResourceHandle;
  }
  [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

void CUDART_CB hold_stream_until_released(void* user_data) {
  auto* const released = static_cast<std::atomic<bool>*>(user_data);
  while (!released->load(std::memory_order_acquire)) std::this_thread::yield();
}

class BlockingStreamGate {
 public:
  ~BlockingStreamGate() { release(); }
  BlockingStreamGate(const BlockingStreamGate&) = delete;
  BlockingStreamGate& operator=(const BlockingStreamGate&) = delete;
  BlockingStreamGate() = default;

  cudaError_t arm(cudaStream_t blocked_stream) {
    cudaError_t status = gate_stream_.create();
    if (status != cudaSuccess) return status;
    status = gate_event_.create();
    if (status != cudaSuccess) return status;
    status = cudaLaunchHostFunc(gate_stream_.get(), hold_stream_until_released, &released_);
    if (status != cudaSuccess) return status;
    status = cudaEventRecord(gate_event_.get(), gate_stream_.get());
    return status == cudaSuccess ? cudaStreamWaitEvent(blocked_stream, gate_event_.get(), 0u)
                                 : status;
  }

  void release() noexcept { released_.store(true, std::memory_order_release); }

 private:
  std::atomic<bool> released_{false};
  StreamOwner gate_stream_;
  EventOwner gate_event_;
};

/* Restore the test thread even when a CHECK returns early. The public API has
 * its own current-device preservation contract, verified separately below. */
class CurrentDeviceRestore {
 public:
  CurrentDeviceRestore() noexcept { valid_ = cudaGetDevice(&device_) == cudaSuccess; }
  ~CurrentDeviceRestore() {
    if (valid_) (void)cudaSetDevice(device_);
  }
  CurrentDeviceRestore(const CurrentDeviceRestore&) = delete;
  CurrentDeviceRestore& operator=(const CurrentDeviceRestore&) = delete;

 private:
  int device_ = 0;
  bool valid_ = false;
};

template <typename T>
xtbloom_const_buffer_t host_input(const std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), XTBLOOM_MEMORY_HOST,
          0u};
}

struct PublicBatch {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int32_t> spin_channels;
  std::vector<std::int64_t> point_charge_offsets;
  std::vector<double> point_charge_positions;
  std::vector<double> point_charge_values;
  std::vector<double> point_charge_gammas;
  std::vector<double> atomic_potential_shifts;
  std::vector<std::int64_t> charge_response_offsets;
  std::vector<double> charge_response_matrix;
  xtbloom_batch_t descriptor{};

  void bind() noexcept {
    (void)xtbloom_batch_init(&descriptor, sizeof(descriptor));
    descriptor.batch_size = static_cast<std::int64_t>(molecular_charges.size());
    descriptor.total_atoms = static_cast<std::int64_t>(atomic_numbers.size());
    descriptor.total_point_charges = static_cast<std::int64_t>(point_charge_values.size());
    descriptor.total_charge_response_elements =
        static_cast<std::int64_t>(charge_response_matrix.size());
    descriptor.atom_offsets = host_input(atom_offsets);
    descriptor.atomic_numbers = host_input(atomic_numbers);
    descriptor.positions = host_input(positions);
    descriptor.molecular_charges = host_input(molecular_charges);
    descriptor.unpaired_electrons = host_input(unpaired_electrons);
    if (!spin_channels.empty()) descriptor.spin_channels = host_input(spin_channels);
    if (!point_charge_values.empty()) {
      descriptor.point_charge_offsets = host_input(point_charge_offsets);
      descriptor.point_charge_positions = host_input(point_charge_positions);
      descriptor.point_charge_values = host_input(point_charge_values);
      descriptor.point_charge_gammas = host_input(point_charge_gammas);
    }
    if (!atomic_potential_shifts.empty()) {
      descriptor.atomic_potential_shifts = host_input(atomic_potential_shifts);
    }
    if (!charge_response_matrix.empty()) {
      descriptor.charge_response_offsets = host_input(charge_response_offsets);
      descriptor.charge_response_matrix = host_input(charge_response_matrix);
    }
  }

  static PublicBatch from_fixture(const HostSccCase& fixture) {
    PublicBatch batch;
    batch.atom_offsets = fixture.atom_offsets();
    batch.atomic_numbers = fixture.atomic_numbers();
    batch.positions = fixture.positions();
    batch.molecular_charges = fixture.molecular_charges();
    batch.unpaired_electrons = fixture.unpaired_electrons();
    batch.spin_channels = fixture.spin_channels();
    if (batch.spin_channels.empty()) {
      /* Some restricted fixtures intentionally exercise the ABI-v1 NULL
       * default. PublicBatch materializes that default so direct-device tests
       * can bind and mutate the ABI-v2 topology leaf explicitly. */
      batch.spin_channels.assign(batch.molecular_charges.size(), 1);
    }
    batch.point_charge_offsets = fixture.point_charge_offsets();
    batch.point_charge_positions = fixture.point_charge_positions();
    batch.point_charge_values = fixture.point_charge_charges();
    batch.point_charge_gammas = fixture.point_charge_hardnesses();
    batch.atomic_potential_shifts = fixture.periodic_shifts();
    batch.charge_response_matrix = fixture.periodic_response_matrices();
    if (!batch.charge_response_matrix.empty()) {
      batch.charge_response_offsets.reserve(batch.atom_offsets.size());
      batch.charge_response_offsets.push_back(0);
      for (std::size_t system = 0; system + 1u < batch.atom_offsets.size(); ++system) {
        const std::int64_t atoms = batch.atom_offsets[system + 1u] - batch.atom_offsets[system];
        batch.charge_response_offsets.push_back(batch.charge_response_offsets.back() +
                                                atoms * atoms);
      }
    }
    batch.bind();
    return batch;
  }

  /* Preserve topology while producing distinguishable requests for concurrency tests. */
  void perturb(double scale) noexcept {
    for (std::size_t system = 0; system + 1u < atom_offsets.size(); ++system) {
      const std::int64_t begin = atom_offsets[system];
      const std::int64_t end = atom_offsets[system + 1u];
      positions[3u * static_cast<std::size_t>(begin)] -= scale;
      if (end - begin > 1) {
        positions[3u * static_cast<std::size_t>(end - 1)] += 0.75 * scale;
      } else {
        positions[3u * static_cast<std::size_t>(begin) + 1u] += 0.5 * scale;
      }
    }
    for (std::size_t point = 0; point < point_charge_values.size(); ++point) {
      point_charge_positions[3u * point] += (point % 2u == 0u ? 0.4 : -0.3) * scale;
      point_charge_values[point] += (point % 2u == 0u ? 0.2 : -0.15) * scale;
      point_charge_gammas[point] += 0.1 * scale;
    }
    for (std::size_t atom = 0; atom < atomic_potential_shifts.size(); ++atom) {
      atomic_potential_shifts[atom] += (atom % 2u == 0u ? 0.25 : -0.2) * scale;
    }
    for (std::size_t system = 0; system + 1u < charge_response_offsets.size(); ++system) {
      const std::int64_t atoms = atom_offsets[system + 1u] - atom_offsets[system];
      const std::size_t base = static_cast<std::size_t>(charge_response_offsets[system]);
      for (std::int64_t atom = 0; atom < atoms; ++atom) {
        charge_response_matrix[base + static_cast<std::size_t>(atom * atoms + atom)] +=
            0.05 * scale;
      }
    }
    bind();
  }
};

xtbloom_compute_options_t make_compute_options() noexcept {
  xtbloom_compute_options_t options{};
  (void)xtbloom_compute_options_init(&options, sizeof(options));
  options.model = XTBLOOM_MODEL_GFN2_XTB;
  options.flags = kRequestedProperties;
  options.max_scc_iterations = 64;
  options.charge_tolerance = 1.0e-8;
  options.energy_tolerance = 1.0e-8;
  options.electronic_temperature = 0.0;
  return options;
}

ContextHandle make_context(xtbloom_backend_t backend, std::int32_t device_id, cudaStream_t stream,
                           xtbloom_status_t& status) {
  xtbloom_context_options_t options{};
  status = xtbloom_context_options_init(&options, sizeof(options));
  if (status != XTBLOOM_STATUS_SUCCESS) return {};
  options.backend = backend;
  options.device_id = device_id;
  options.stream = reinterpret_cast<void*>(stream);
  xtbloom_context_t* raw = nullptr;
  status = xtbloom_context_create(&options, &raw);
  return ContextHandle(raw);
}

template <typename T>
struct FieldCanaries;

template <>
struct FieldCanaries<double> {
  static constexpr double kFront = -81001.25;
  static constexpr double kPayload = 82002.5;
  static constexpr double kBack = -83003.75;
};

template <>
struct FieldCanaries<std::int32_t> {
  static constexpr std::int32_t kFront = INT32_C(0x13572468);
  static constexpr std::int32_t kPayload = INT32_C(0x24681357);
  static constexpr std::int32_t kBack = INT32_C(0x31415926);
};

template <>
struct FieldCanaries<std::uint8_t> {
  static constexpr std::uint8_t kFront = UINT8_C(0x5b);
  static constexpr std::uint8_t kPayload = UINT8_C(0xa7);
  static constexpr std::uint8_t kBack = UINT8_C(0xc3);
};

enum class Placement { kHost, kDevice };

enum class InputLayout { kHost, kDevice, kMixed };

/* Own one stable direct-device input allocation. upload() preserves the
 * address while the extent is unchanged, which lets the repeated-call test
 * distinguish fixed-topology refresh from an accidental descriptor rebuild. */
template <typename T>
class DeviceInputArray {
 public:
  DeviceInputArray() noexcept = default;
  ~DeviceInputArray() {
    if (data_ != nullptr) (void)cudaFree(data_);
  }
  DeviceInputArray(const DeviceInputArray&) = delete;
  DeviceInputArray& operator=(const DeviceInputArray&) = delete;

  cudaError_t upload(const std::vector<T>& values) noexcept {
    if (values.size() != count_) {
      if (data_ != nullptr) {
        const cudaError_t free_status = cudaFree(data_);
        if (free_status != cudaSuccess) return free_status;
        data_ = nullptr;
      }
      count_ = values.size();
      if (count_ != 0u) {
        const cudaError_t allocation_status =
            cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T));
        if (allocation_status != cudaSuccess) return allocation_status;
      }
    }
    return count_ == 0u
               ? cudaSuccess
               : cudaMemcpy(data_, values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice);
  }

  [[nodiscard]] xtbloom_const_buffer_t descriptor() const noexcept {
    return {data_, count_ * sizeof(T), XTBLOOM_MEMORY_CUDA_DEVICE, 0u};
  }

  [[nodiscard]] std::uintptr_t address() const noexcept {
    return reinterpret_cast<std::uintptr_t>(data_);
  }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

/* Mirrors every public input leaf so the end-to-end matrix can exercise all
 * seven topology buffers and every numerical QM/MM buffer without managed
 * memory obscuring the ABI memory-space contract. */
class DeviceBatchInputs {
 public:
  cudaError_t upload_topology(const PublicBatch& batch) noexcept {
    cudaError_t status = atom_offsets_.upload(batch.atom_offsets);
    if (status != cudaSuccess) return status;
    status = atomic_numbers_.upload(batch.atomic_numbers);
    if (status != cudaSuccess) return status;
    status = molecular_charges_.upload(batch.molecular_charges);
    if (status != cudaSuccess) return status;
    status = unpaired_electrons_.upload(batch.unpaired_electrons);
    if (status != cudaSuccess) return status;
    status = spin_channels_.upload(batch.spin_channels);
    if (status != cudaSuccess) return status;
    status = point_charge_offsets_.upload(batch.point_charge_offsets);
    if (status != cudaSuccess) return status;
    return charge_response_offsets_.upload(batch.charge_response_offsets);
  }

  cudaError_t upload_numerical(const PublicBatch& batch) noexcept {
    cudaError_t status = positions_.upload(batch.positions);
    if (status != cudaSuccess) return status;
    status = point_charge_positions_.upload(batch.point_charge_positions);
    if (status != cudaSuccess) return status;
    status = point_charge_values_.upload(batch.point_charge_values);
    if (status != cudaSuccess) return status;
    status = point_charge_gammas_.upload(batch.point_charge_gammas);
    if (status != cudaSuccess) return status;
    status = atomic_potential_shifts_.upload(batch.atomic_potential_shifts);
    if (status != cudaSuccess) return status;
    return charge_response_matrix_.upload(batch.charge_response_matrix);
  }

  cudaError_t upload_all(const PublicBatch& batch) noexcept {
    const cudaError_t status = upload_topology(batch);
    return status == cudaSuccess ? upload_numerical(batch) : status;
  }

  /* Pointer identity for every required or optional public input leaf. */
  [[nodiscard]] std::array<std::uintptr_t, 13> identity() const noexcept {
    return {atom_offsets_.address(),
            atomic_numbers_.address(),
            molecular_charges_.address(),
            unpaired_electrons_.address(),
            spin_channels_.address(),
            point_charge_offsets_.address(),
            charge_response_offsets_.address(),
            positions_.address(),
            point_charge_positions_.address(),
            point_charge_values_.address(),
            point_charge_gammas_.address(),
            atomic_potential_shifts_.address(),
            charge_response_matrix_.address()};
  }

  DeviceInputArray<std::int64_t> atom_offsets_;
  DeviceInputArray<std::int32_t> atomic_numbers_;
  DeviceInputArray<double> molecular_charges_;
  DeviceInputArray<std::int32_t> unpaired_electrons_;
  DeviceInputArray<std::int32_t> spin_channels_;
  DeviceInputArray<std::int64_t> point_charge_offsets_;
  DeviceInputArray<std::int64_t> charge_response_offsets_;
  DeviceInputArray<double> positions_;
  DeviceInputArray<double> point_charge_positions_;
  DeviceInputArray<double> point_charge_values_;
  DeviceInputArray<double> point_charge_gammas_;
  DeviceInputArray<double> atomic_potential_shifts_;
  DeviceInputArray<double> charge_response_matrix_;
};

void bind_inputs(PublicBatch& batch, const DeviceBatchInputs* device, InputLayout layout) noexcept {
  batch.bind();
  if (layout == InputLayout::kHost || device == nullptr) return;

  const bool all_device = layout == InputLayout::kDevice;
  /* Mixed topology: atom offsets, charge/spin metadata, and point offsets are
   * direct-device; atomic numbers and response offsets remain host-backed. */
  batch.descriptor.atom_offsets = device->atom_offsets_.descriptor();
  if (all_device) batch.descriptor.atomic_numbers = device->atomic_numbers_.descriptor();
  batch.descriptor.molecular_charges = device->molecular_charges_.descriptor();
  batch.descriptor.unpaired_electrons = device->unpaired_electrons_.descriptor();
  if (!batch.spin_channels.empty()) {
    batch.descriptor.spin_channels = device->spin_channels_.descriptor();
  }
  if (!batch.point_charge_offsets.empty()) {
    batch.descriptor.point_charge_offsets = device->point_charge_offsets_.descriptor();
  }
  if (all_device && !batch.charge_response_offsets.empty()) {
    batch.descriptor.charge_response_offsets = device->charge_response_offsets_.descriptor();
  }

  /* Mixed numerical leaves intentionally alternate device/host sources. */
  batch.descriptor.positions = device->positions_.descriptor();
  if (!batch.point_charge_positions.empty()) {
    if (all_device) {
      batch.descriptor.point_charge_positions = device->point_charge_positions_.descriptor();
    }
    batch.descriptor.point_charge_values = device->point_charge_values_.descriptor();
    if (all_device)
      batch.descriptor.point_charge_gammas = device->point_charge_gammas_.descriptor();
  }
  if (!batch.atomic_potential_shifts.empty()) {
    batch.descriptor.atomic_potential_shifts = device->atomic_potential_shifts_.descriptor();
  }
  if (all_device && !batch.charge_response_matrix.empty()) {
    batch.descriptor.charge_response_matrix = device->charge_response_matrix_.descriptor();
  }
}

/* Each public slice is surrounded by caller-owned guards. Besides checking
 * transactional preservation, this catches an off-by-one publication kernel
 * that ordinary value comparison would miss. */
template <typename T>
class GuardedOutput {
 public:
  GuardedOutput() noexcept = default;
  ~GuardedOutput() {
    if (device_ != nullptr) (void)cudaFree(device_);
  }
  GuardedOutput(const GuardedOutput&) = delete;
  GuardedOutput& operator=(const GuardedOutput&) = delete;

  cudaError_t initialize(std::size_t count, Placement placement) {
    if (device_ != nullptr) {
      const cudaError_t free_status = cudaFree(device_);
      if (free_status != cudaSuccess) return free_status;
      device_ = nullptr;
    }
    count_ = count;
    placement_ = placement;
    initial_.assign(count_ + 2u, FieldCanaries<T>::kPayload);
    initial_.front() = FieldCanaries<T>::kFront;
    initial_.back() = FieldCanaries<T>::kBack;
    host_ = initial_;
    if (placement_ == Placement::kHost) return cudaSuccess;
    cudaError_t status =
        cudaMalloc(reinterpret_cast<void**>(&device_), initial_.size() * sizeof(T));
    if (status != cudaSuccess) return status;
    status =
        cudaMemcpy(device_, initial_.data(), initial_.size() * sizeof(T), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
      (void)cudaFree(device_);
      device_ = nullptr;
    }
    return status;
  }

  [[nodiscard]] xtbloom_buffer_t descriptor() noexcept {
    void* payload = placement_ == Placement::kHost ? static_cast<void*>(host_.data() + 1u)
                                                   : static_cast<void*>(device_ + 1u);
    return {payload, count_ * sizeof(T),
            placement_ == Placement::kHost ? XTBLOOM_MEMORY_HOST : XTBLOOM_MEMORY_CUDA_DEVICE, 0u};
  }

  cudaError_t read_all(std::vector<T>& values) const {
    values.resize(count_ + 2u);
    if (placement_ == Placement::kHost) {
      values = host_;
      return cudaSuccess;
    }
    return cudaMemcpy(values.data(), device_, values.size() * sizeof(T), cudaMemcpyDeviceToHost);
  }

  cudaError_t read_payload(std::vector<T>& values) const {
    std::vector<T> all;
    const cudaError_t status = read_all(all);
    if (status != cudaSuccess) return status;
    values.assign(all.begin() + 1, all.end() - 1);
    return cudaSuccess;
  }

  cudaError_t guards_intact(bool& intact) const {
    std::vector<T> all;
    const cudaError_t status = read_all(all);
    if (status != cudaSuccess) return status;
    intact = all.front() == FieldCanaries<T>::kFront && all.back() == FieldCanaries<T>::kBack;
    return cudaSuccess;
  }

  cudaError_t unchanged(bool& same) const {
    std::vector<T> all;
    const cudaError_t status = read_all(all);
    if (status != cudaSuccess) return status;
    same = all == initial_;
    return cudaSuccess;
  }

 private:
  std::size_t count_ = 0u;
  Placement placement_ = Placement::kHost;
  std::vector<T> initial_;
  std::vector<T> host_;
  T* device_ = nullptr;
};

enum class ResultLayout {
  kHost,
  kDevice,
  kMixed,
  /* The Torch request path publishes primary tensors on device while keeping
   * atomic charges and convergence diagnostics in request-owned host slots. */
  kTorchRequest,
};

struct MaterializedResult {
  std::vector<double> energies;
  std::vector<double> forces;
  std::vector<double> atomic_charges;
  std::vector<double> point_charge_forces;
  std::vector<std::int32_t> iterations;
  std::vector<std::uint8_t> converged;
  std::vector<std::int32_t> statuses;
  std::uint32_t flags = 0u;
};

class ResultOwner {
 public:
  cudaError_t bind(const PublicBatch& batch, ResultLayout layout,
                   std::uint32_t flags = kRequestedProperties) {
    const std::size_t systems = static_cast<std::size_t>(batch.descriptor.batch_size);
    const std::size_t atoms = static_cast<std::size_t>(batch.descriptor.total_atoms);
    const std::size_t points = static_cast<std::size_t>(batch.descriptor.total_point_charges);
    const Placement all = layout == ResultLayout::kDevice ? Placement::kDevice : Placement::kHost;
    const Placement energies_placement =
        layout == ResultLayout::kDevice || layout == ResultLayout::kTorchRequest
            ? Placement::kDevice
            : Placement::kHost;
    const Placement forces_placement =
        layout == ResultLayout::kHost ? Placement::kHost : Placement::kDevice;
    const Placement iterations_placement =
        layout == ResultLayout::kMixed ? Placement::kDevice : all;
    const Placement statuses_placement = layout == ResultLayout::kMixed ? Placement::kDevice : all;

    cudaError_t status = energies_.initialize(systems, energies_placement);
    if (status != cudaSuccess) return status;
    status = forces_.initialize(3u * atoms, forces_placement);
    if (status != cudaSuccess) return status;
    status = charges_.initialize(atoms, all);
    if (status != cudaSuccess) return status;
    status = point_forces_.initialize(3u * points, all);
    if (status != cudaSuccess) return status;
    status = iterations_.initialize(systems, iterations_placement);
    if (status != cudaSuccess) return status;
    status = converged_.initialize(systems, all);
    if (status != cudaSuccess) return status;
    status = statuses_.initialize(systems, statuses_placement);
    if (status != cudaSuccess) return status;

    (void)xtbloom_batch_result_init(&descriptor, sizeof(descriptor));
    descriptor.flags = kResultFlagsCanary;
    descriptor.energies =
        (flags & XTBLOOM_COMPUTE_ENERGY) != 0u ? energies_.descriptor() : xtbloom_buffer_t{};
    descriptor.forces =
        (flags & XTBLOOM_COMPUTE_FORCES) != 0u ? forces_.descriptor() : xtbloom_buffer_t{};
    descriptor.atomic_charges =
        (flags & XTBLOOM_COMPUTE_ATOMIC_CHARGES) != 0u ? charges_.descriptor() : xtbloom_buffer_t{};
    descriptor.point_charge_forces = (flags & XTBLOOM_COMPUTE_POINT_CHARGE_FORCES) != 0u
                                         ? point_forces_.descriptor()
                                         : xtbloom_buffer_t{};
    descriptor.scc_iterations = iterations_.descriptor();
    descriptor.scc_converged = converged_.descriptor();
    descriptor.per_system_status = statuses_.descriptor();
    return cudaSuccess;
  }

  cudaError_t materialize(MaterializedResult& result) const {
    cudaError_t status = energies_.read_payload(result.energies);
    if (status != cudaSuccess) return status;
    status = forces_.read_payload(result.forces);
    if (status != cudaSuccess) return status;
    status = charges_.read_payload(result.atomic_charges);
    if (status != cudaSuccess) return status;
    status = point_forces_.read_payload(result.point_charge_forces);
    if (status != cudaSuccess) return status;
    status = iterations_.read_payload(result.iterations);
    if (status != cudaSuccess) return status;
    status = converged_.read_payload(result.converged);
    if (status != cudaSuccess) return status;
    status = statuses_.read_payload(result.statuses);
    result.flags = descriptor.flags;
    return status;
  }

  cudaError_t guards_intact(bool& intact) const {
    intact = true;
    bool field = false;
    cudaError_t status = energies_.guards_intact(field);
    if (status != cudaSuccess) return status;
    intact = intact && field;
    status = forces_.guards_intact(field);
    if (status != cudaSuccess) return status;
    intact = intact && field;
    status = charges_.guards_intact(field);
    if (status != cudaSuccess) return status;
    intact = intact && field;
    status = point_forces_.guards_intact(field);
    if (status != cudaSuccess) return status;
    intact = intact && field;
    status = iterations_.guards_intact(field);
    if (status != cudaSuccess) return status;
    intact = intact && field;
    status = converged_.guards_intact(field);
    if (status != cudaSuccess) return status;
    intact = intact && field;
    status = statuses_.guards_intact(field);
    if (status != cudaSuccess) return status;
    intact = intact && field;
    return cudaSuccess;
  }

  cudaError_t unchanged(bool& same) const {
    same = descriptor.flags == kResultFlagsCanary;
    bool field = false;
    cudaError_t status = energies_.unchanged(field);
    if (status != cudaSuccess) return status;
    same = same && field;
    status = forces_.unchanged(field);
    if (status != cudaSuccess) return status;
    same = same && field;
    status = charges_.unchanged(field);
    if (status != cudaSuccess) return status;
    same = same && field;
    status = point_forces_.unchanged(field);
    if (status != cudaSuccess) return status;
    same = same && field;
    status = iterations_.unchanged(field);
    if (status != cudaSuccess) return status;
    same = same && field;
    status = converged_.unchanged(field);
    if (status != cudaSuccess) return status;
    same = same && field;
    status = statuses_.unchanged(field);
    if (status != cudaSuccess) return status;
    same = same && field;
    return cudaSuccess;
  }

  cudaError_t torch_host_diagnostics_unchanged(bool& same) const {
    same = true;
    bool field = false;
    cudaError_t status = charges_.unchanged(field);
    if (status != cudaSuccess) return status;
    same = same && field;
    status = point_forces_.unchanged(field);
    if (status != cudaSuccess) return status;
    same = same && field;
    status = iterations_.unchanged(field);
    if (status != cudaSuccess) return status;
    same = same && field;
    status = converged_.unchanged(field);
    if (status != cudaSuccess) return status;
    same = same && field;
    status = statuses_.unchanged(field);
    if (status != cudaSuccess) return status;
    same = same && field;
    return cudaSuccess;
  }

  xtbloom_batch_result_t descriptor{};

 private:
  GuardedOutput<double> energies_;
  GuardedOutput<double> forces_;
  GuardedOutput<double> charges_;
  GuardedOutput<double> point_forces_;
  GuardedOutput<std::int32_t> iterations_;
  GuardedOutput<std::uint8_t> converged_;
  GuardedOutput<std::int32_t> statuses_;
};

bool near(double actual, double expected, double absolute_tolerance, double relative_tolerance) {
  return std::abs(actual - expected) <=
         absolute_tolerance + relative_tolerance * std::max(std::abs(actual), std::abs(expected));
}

bool is_quiet_nan(double value) noexcept {
  std::uint64_t bits = 0u;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  constexpr std::uint64_t kExponentMask = UINT64_C(0x7ff0000000000000);
  constexpr std::uint64_t kFractionMask = UINT64_C(0x000fffffffffffff);
  constexpr std::uint64_t kQuietBit = UINT64_C(0x0008000000000000);
  return (bits & kExponentMask) == kExponentMask && (bits & kFractionMask) != 0u &&
         (bits & kQuietBit) != 0u;
}

int compare_values(const char* name, const std::vector<double>& actual,
                   const std::vector<double>& expected, double absolute_tolerance,
                   double relative_tolerance) {
  CHECK(actual.size() == expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!near(actual[index], expected[index], absolute_tolerance, relative_tolerance)) {
      std::fprintf(stderr, "%s mismatch in %s at %zu: actual=%.17g expected=%.17g\n", name,
                   g_scenario, index, actual[index], expected[index]);
      return __LINE__;
    }
  }
  return 0;
}

int compare_result(const ResultOwner& owner, const MaterializedResult& actual,
                   const MaterializedResult& expected, const xtbloom_compute_options_t& options) {
  bool guards = false;
  CUDA_CHECK(owner.guards_intact(guards));
  CHECK(guards);
  CHECK(actual.flags == expected.flags);
  CHECK(actual.statuses.size() == expected.statuses.size());
  CHECK(actual.converged.size() == expected.converged.size());
  CHECK(actual.iterations.size() == expected.iterations.size());
  for (std::size_t system = 0; system < actual.statuses.size(); ++system) {
    CHECK(actual.statuses[system] == XTBLOOM_STATUS_SUCCESS);
    CHECK(actual.statuses[system] == expected.statuses[system]);
    CHECK(actual.converged[system] == 1u);
    CHECK(actual.converged[system] == expected.converged[system]);
    CHECK(actual.iterations[system] > 0);
    CHECK(actual.iterations[system] <= options.max_scc_iterations);
  }
  int line = compare_values("energy", actual.energies, expected.energies, kEnergyAbsoluteTolerance,
                            kEnergyRelativeTolerance);
  if (line != 0) return line;
  line = compare_values("atomic charge", actual.atomic_charges, expected.atomic_charges,
                        kChargeAbsoluteTolerance, kChargeRelativeTolerance);
  if (line != 0) return line;
  line = compare_values("force", actual.forces, expected.forces, kForceAbsoluteTolerance,
                        kForceRelativeTolerance);
  if (line != 0) return line;
  return compare_values("point-charge force", actual.point_charge_forces,
                        expected.point_charge_forces, kForceAbsoluteTolerance,
                        kForceRelativeTolerance);
}

int make_fixture_batch(std::size_t batch_size, bool enable_qmmm, PublicBatch& batch) {
  HostSccCaseOptions options{};
  constexpr std::array<SmallSystemKind, 4> kSystems = {
      SmallSystemKind::kH2, SmallSystemKind::kHe, SmallSystemKind::kLiH, SmallSystemKind::kCH2};
  options.systems.clear();
  options.systems.reserve(batch_size);
  for (std::size_t system = 0; system < batch_size; ++system) {
    options.systems.push_back(kSystems[system % kSystems.size()]);
  }
  options.enable_periodic_embedding = enable_qmmm;
  options.enable_explicit_point_charges = enable_qmmm;
  options.maximum_iterations = 64u;
  options.residual_tolerance = 1.0e-8;
  options.energy_tolerance = 1.0e-8;
  HostSccCase fixture;
  std::string error;
  const xtbloom_status_t status = HostSccCase::create(options, fixture, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "failed to build public CUDA API fixture: %s\n", error.c_str());
    return __LINE__;
  }
  batch = PublicBatch::from_fixture(fixture);
  return 0;
}

int make_fixture_batch(PublicBatch& batch) { return make_fixture_batch(4u, false, batch); }

PublicBatch make_representability_batch() {
  PublicBatch batch;
  batch.atom_offsets = {0, 2, 5};
  batch.atomic_numbers = {1, 1, 1, 1, 1};
  batch.positions = {
      -0.70, 0.0, 0.0, 0.70, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e20, 0.0, 0.0, 2.0e20, 0.0, 0.0,
  };
  /* The three remote hydrogens are physically independent at this scale, so
   * their one-center eigensystems are bitwise equal while every reciprocal or
   * overlap interaction safely tends to zero. The fractional charge produces
   * nextafter(3, 0) electrons in each restricted spin channel. */
  batch.molecular_charges = {0.0, 3.0 - 2.0 * std::nextafter(3.0, 0.0)};
  batch.unpaired_electrons = {0, 0};
  batch.spin_channels = {1, 1};
  batch.bind();
  return batch;
}

int run_cpu_reference(xtbloom_context_t* cpu_context, PublicBatch& batch,
                      const xtbloom_compute_options_t& options, MaterializedResult& reference) {
  ResultOwner result;
  bind_inputs(batch, nullptr, InputLayout::kHost);
  CUDA_CHECK(result.bind(batch, ResultLayout::kHost, options.flags));
  CHECK(xtbloom_compute(cpu_context, &batch.descriptor, &options, &result.descriptor) ==
        XTBLOOM_STATUS_SUCCESS);
  CUDA_CHECK(result.materialize(reference));
  bool guards = false;
  CUDA_CHECK(result.guards_intact(guards));
  CHECK(guards);
  for (std::size_t system = 0; system < reference.statuses.size(); ++system) {
    CHECK(reference.statuses[system] == XTBLOOM_STATUS_SUCCESS);
    CHECK(reference.converged[system] == 1u);
  }
  return 0;
}

int execute_cuda_and_compare(xtbloom_context_t* cuda_context, PublicBatch& batch,
                             const xtbloom_compute_options_t& options, ResultLayout layout,
                             const MaterializedResult& reference) {
  ResultOwner result;
  CUDA_CHECK(result.bind(batch, layout, options.flags));
  CHECK(xtbloom_compute(cuda_context, &batch.descriptor, &options, &result.descriptor) ==
        XTBLOOM_STATUS_SUCCESS);
  MaterializedResult actual;
  CUDA_CHECK(result.materialize(actual));
  return compare_result(result, actual, reference, options);
}

int test_public_mixer_controls_and_reproducibility(std::int32_t device,
                                                   xtbloom_context_t* cpu_context) {
  PublicBatch batch;
  /* One H2/He/LiH/CH2 cycle covers both restricted and unrestricted SCC. */
  CHECK(make_fixture_batch(4u, false, batch) == 0);
  xtbloom_compute_options_t explicit_defaults = make_compute_options();
  MaterializedResult reference;
  g_scenario = "mixer-controls/CPU-default-reference";
  CHECK(run_cpu_reference(cpu_context, batch, explicit_defaults, reference) == 0);

  const std::array<std::size_t, 3> legacy_sizes{XTBLOOM_COMPUTE_OPTIONS_V1_SIZE,
                                                XTBLOOM_COMPUTE_OPTIONS_V2_SIZE,
                                                XTBLOOM_COMPUTE_OPTIONS_V3_SIZE - 1u};
  const std::array<ResultLayout, 3> legacy_layouts{ResultLayout::kHost, ResultLayout::kDevice,
                                                   ResultLayout::kMixed};
  StreamOwner stream;
  CUDA_CHECK(stream.create());
  xtbloom_status_t context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle context = make_context(XTBLOOM_BACKEND_CUDA, device, stream.get(), context_status);
  CHECK(context_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(context != nullptr);
  for (std::size_t index = 0; index < legacy_sizes.size(); ++index) {
    std::string scenario = "mixer-controls/legacy-prefix/" + std::to_string(legacy_sizes[index]);
    g_scenario = scenario.c_str();
    xtbloom_compute_options_t legacy = explicit_defaults;
    legacy.struct_size = static_cast<std::uint32_t>(legacy_sizes[index]);
    CHECK(execute_cuda_and_compare(context.get(), batch, legacy, legacy_layouts[index],
                                   reference) == 0);
  }

  xtbloom_compute_options_t nondefault = explicit_defaults;
  nondefault.scc_mixer_history = 4;
  nondefault.scc_mixer_damping = 0.2;
  MaterializedResult nondefault_reference;
  g_scenario = "mixer-controls/CPU-nondefault-reference";
  CHECK(run_cpu_reference(cpu_context, batch, nondefault, nondefault_reference) == 0);
  for (const auto& [layout, name] :
       {std::pair{ResultLayout::kHost, "host"}, std::pair{ResultLayout::kDevice, "device"},
        std::pair{ResultLayout::kMixed, "mixed"}}) {
    std::string scenario = std::string("mixer-controls/nondefault/") + name;
    g_scenario = scenario.c_str();
    CHECK(execute_cuda_and_compare(context.get(), batch, nondefault, layout,
                                   nondefault_reference) == 0);
  }

  xtbloom_compute_options_t reproducible = explicit_defaults;
  reproducible.determinism = XTBLOOM_DETERMINISM_REPRODUCIBLE;
  ResultOwner replay_owner;
  CUDA_CHECK(replay_owner.bind(batch, ResultLayout::kMixed, reproducible.flags));
  g_scenario = "mixer-controls/reproducible-first";
  CHECK(xtbloom_compute(context.get(), &batch.descriptor, &reproducible,
                        &replay_owner.descriptor) == XTBLOOM_STATUS_SUCCESS);
  MaterializedResult first;
  CUDA_CHECK(replay_owner.materialize(first));
  for (int repetition = 0; repetition < 3; ++repetition) {
    g_scenario = "mixer-controls/reproducible-replay";
    CHECK(xtbloom_compute(context.get(), &batch.descriptor, &reproducible,
                          &replay_owner.descriptor) == XTBLOOM_STATUS_SUCCESS);
    MaterializedResult replay;
    CUDA_CHECK(replay_owner.materialize(replay));
    CHECK(replay.flags == first.flags);
    CHECK(replay.energies == first.energies);
    CHECK(replay.forces == first.forces);
    CHECK(replay.atomic_charges == first.atomic_charges);
    CHECK(replay.iterations == first.iterations);
    CHECK(replay.converged == first.converged);
    CHECK(replay.statuses == first.statuses);
  }

  /* The exact-replay contract includes the FRESH/WARM sequence. Re-seeding
   * the same cache with an identical FRESH frame must reproduce both that
   * frame and the following perturbed strict-WARM frame byte-for-byte. */
  const std::vector<double> base_positions = batch.positions;
  batch.positions[0] -= 0.002;
  batch.positions[3] += 0.002;
  reproducible.scc_start_mode = XTBLOOM_SCC_START_WARM;
  g_scenario = "mixer-controls/reproducible-warm-first";
  CHECK(xtbloom_compute(context.get(), &batch.descriptor, &reproducible,
                        &replay_owner.descriptor) == XTBLOOM_STATUS_SUCCESS);
  MaterializedResult warm_first;
  CUDA_CHECK(replay_owner.materialize(warm_first));

  std::copy(base_positions.begin(), base_positions.end(), batch.positions.begin());
  reproducible.scc_start_mode = XTBLOOM_SCC_START_FRESH;
  g_scenario = "mixer-controls/reproducible-sequence-reset";
  CHECK(xtbloom_compute(context.get(), &batch.descriptor, &reproducible,
                        &replay_owner.descriptor) == XTBLOOM_STATUS_SUCCESS);
  MaterializedResult fresh_replay;
  CUDA_CHECK(replay_owner.materialize(fresh_replay));
  CHECK(fresh_replay.flags == first.flags);
  CHECK(fresh_replay.energies == first.energies);
  CHECK(fresh_replay.forces == first.forces);
  CHECK(fresh_replay.atomic_charges == first.atomic_charges);
  CHECK(fresh_replay.iterations == first.iterations);
  CHECK(fresh_replay.converged == first.converged);
  CHECK(fresh_replay.statuses == first.statuses);

  batch.positions[0] -= 0.002;
  batch.positions[3] += 0.002;
  reproducible.scc_start_mode = XTBLOOM_SCC_START_WARM;
  g_scenario = "mixer-controls/reproducible-warm-replay";
  CHECK(xtbloom_compute(context.get(), &batch.descriptor, &reproducible,
                        &replay_owner.descriptor) == XTBLOOM_STATUS_SUCCESS);
  MaterializedResult warm_replay;
  CUDA_CHECK(replay_owner.materialize(warm_replay));
  CHECK(warm_replay.flags == warm_first.flags);
  CHECK(warm_replay.energies == warm_first.energies);
  CHECK(warm_replay.forces == warm_first.forces);
  CHECK(warm_replay.atomic_charges == warm_first.atomic_charges);
  CHECK(warm_replay.iterations == warm_first.iterations);
  CHECK(warm_replay.converged == warm_first.converged);
  CHECK(warm_replay.statuses == warm_first.statuses);
  std::copy(base_positions.begin(), base_positions.end(), batch.positions.begin());

  xtbloom_request_t* raw_request = nullptr;
  CHECK(xtbloom_request_create(context.get(), &raw_request) == XTBLOOM_STATUS_SUCCESS);
  const std::unique_ptr<xtbloom_request_t, void (*)(xtbloom_request_t*)> request(
      raw_request, xtbloom_request_destroy);
  ResultOwner context_async_owner;
  CUDA_CHECK(context_async_owner.bind(batch, ResultLayout::kMixed, nondefault.flags));
  g_scenario = "mixer-controls/context-async";
  CHECK(xtbloom_compute_enqueue(context.get(), &batch.descriptor, &nondefault,
                                &context_async_owner.descriptor,
                                request.get()) == XTBLOOM_STATUS_SUCCESS);
  xtbloom_request_info_t info{};
  CHECK(xtbloom_request_info_init(&info, sizeof(info)) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom_request_wait(request.get(), &info) == XTBLOOM_STATUS_SUCCESS);
  CHECK(info.completion_status == XTBLOOM_STATUS_SUCCESS);
  MaterializedResult context_async;
  CUDA_CHECK(context_async_owner.materialize(context_async));
  context_async.flags = info.result_flags;
  CHECK(compare_result(context_async_owner, context_async, nondefault_reference, nondefault) == 0);

  xtbloom_plan_t* raw_plan = nullptr;
  CHECK(xtbloom_plan_create(context.get(), &batch.descriptor, &nondefault, &raw_plan) ==
        XTBLOOM_STATUS_SUCCESS);
  const std::unique_ptr<xtbloom_plan_t, void (*)(xtbloom_plan_t*)> plan(raw_plan,
                                                                        xtbloom_plan_destroy);
  ResultOwner plan_sync_owner;
  CUDA_CHECK(plan_sync_owner.bind(batch, ResultLayout::kHost, nondefault.flags));
  g_scenario = "mixer-controls/plan-sync";
  CHECK(xtbloom_plan_compute(plan.get(), &batch.descriptor, &nondefault,
                             &plan_sync_owner.descriptor) == XTBLOOM_STATUS_SUCCESS);
  MaterializedResult plan_sync;
  CUDA_CHECK(plan_sync_owner.materialize(plan_sync));
  CHECK(compare_result(plan_sync_owner, plan_sync, nondefault_reference, nondefault) == 0);

  ResultOwner plan_async_owner;
  CUDA_CHECK(plan_async_owner.bind(batch, ResultLayout::kDevice, nondefault.flags));
  g_scenario = "mixer-controls/plan-async";
  CHECK(xtbloom_plan_compute_enqueue(plan.get(), &batch.descriptor, &nondefault,
                                     &plan_async_owner.descriptor,
                                     request.get()) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom_request_wait(request.get(), &info) == XTBLOOM_STATUS_SUCCESS);
  CHECK(info.completion_status == XTBLOOM_STATUS_SUCCESS);
  MaterializedResult plan_async;
  CUDA_CHECK(plan_async_owner.materialize(plan_async));
  plan_async.flags = info.result_flags;
  CHECK(compare_result(plan_async_owner, plan_async, nondefault_reference, nondefault) == 0);

  xtbloom_compute_options_t changed_plan_policy = nondefault;
  changed_plan_policy.scc_mixer_history = 8;
  changed_plan_policy.scc_start_mode = XTBLOOM_SCC_START_WARM;
  ResultOwner rejected_plan_owner;
  CUDA_CHECK(rejected_plan_owner.bind(batch, ResultLayout::kMixed, changed_plan_policy.flags));
  CHECK(xtbloom_plan_compute_enqueue(plan.get(), &batch.descriptor, &changed_plan_policy,
                                     &rejected_plan_owner.descriptor,
                                     request.get()) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  bool rejected_unchanged = false;
  CUDA_CHECK(rejected_plan_owner.unchanged(rejected_unchanged));
  CHECK(rejected_unchanged);
  return 0;
}

int verify_input_layout(const PublicBatch& batch, InputLayout layout, bool qmmm);

int test_public_representability_matrix(std::int32_t device, xtbloom_context_t* cpu_context,
                                        const xtbloom_compute_options_t& options) {
  PublicBatch batch = make_representability_batch();
  xtbloom_compute_options_t finite_temperature_options = options;
  finite_temperature_options.flags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_ATOMIC_CHARGES;
  finite_temperature_options.electronic_temperature = XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE;
  CHECK(finite_temperature_options.electronic_temperature > 0.0);
  const double corner_electrons = 3.0 - batch.molecular_charges[1];
  CHECK(0.5 * corner_electrons == std::nextafter(3.0, 0.0));
  MaterializedResult reference;
  g_scenario = "representability/CPU-reference";
  CHECK(run_cpu_reference(cpu_context, batch, finite_temperature_options, reference) == 0);
  CHECK(reference.statuses.size() == 2u);
  CHECK(reference.statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(reference.statuses[1] == XTBLOOM_STATUS_SUCCESS);
  CHECK(reference.converged[0] == 1u && reference.converged[1] == 1u);
  CHECK(reference.iterations[1] > 0 &&
        reference.iterations[1] <= finite_temperature_options.max_scc_iterations);
  CHECK(near(reference.energies[1], -1.8322400836158348, 2.0e-12, 2.0e-12));
  for (std::size_t atom = 2u; atom < 5u; ++atom) {
    CHECK(reference.atomic_charges[atom] == -1.0);
  }

  StreamOwner stream;
  CUDA_CHECK(stream.create());
  xtbloom_status_t context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle context = make_context(XTBLOOM_BACKEND_CUDA, device, stream.get(), context_status);
  CHECK(context_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(context != nullptr);

  DeviceBatchInputs device_inputs;
  CUDA_CHECK(device_inputs.upload_all(batch));
  const std::array<std::pair<InputLayout, ResultLayout>, 3> layouts{{
      {InputLayout::kHost, ResultLayout::kHost},
      {InputLayout::kDevice, ResultLayout::kDevice},
      {InputLayout::kMixed, ResultLayout::kMixed},
  }};
  const std::array<const char*, 3> names{{"host", "device", "mixed"}};
  for (std::size_t index = 0u; index < layouts.size(); ++index) {
    std::string scenario = std::string("representability/") + names[index];
    g_scenario = scenario.c_str();
    const auto [input_layout, result_layout] = layouts[index];
    bind_inputs(batch, input_layout == InputLayout::kHost ? nullptr : &device_inputs, input_layout);
    if (input_layout != InputLayout::kHost) {
      CHECK(verify_input_layout(batch, input_layout, false) == 0);
    }
    ResultOwner owner;
    CUDA_CHECK(owner.bind(batch, result_layout, finite_temperature_options.flags));
    CHECK(xtbloom_compute(context.get(), &batch.descriptor, &finite_temperature_options,
                          &owner.descriptor) == XTBLOOM_STATUS_SUCCESS);
    MaterializedResult actual;
    CUDA_CHECK(owner.materialize(actual));
    CHECK(compare_result(owner, actual, reference, finite_temperature_options) == 0);
    CHECK(actual.statuses[0] == XTBLOOM_STATUS_SUCCESS);
    CHECK(actual.statuses[1] == XTBLOOM_STATUS_SUCCESS);
    CHECK(actual.converged[0] == 1u && actual.converged[1] == 1u);
    CHECK(actual.iterations[1] > 0 &&
          actual.iterations[1] <= finite_temperature_options.max_scc_iterations);
    for (std::size_t atom = 2u; atom < 5u; ++atom) {
      CHECK(near(actual.atomic_charges[atom], -1.0, 2.0e-12, 2.0e-12));
    }
  }
  return 0;
}

int expect_strict_warm_rejection(xtbloom_context_t* context, PublicBatch& batch,
                                 const xtbloom_compute_options_t& options, ResultLayout layout) {
  bind_inputs(batch, nullptr, InputLayout::kHost);
  ResultOwner result;
  CUDA_CHECK(result.bind(batch, layout, options.flags));
  CHECK(xtbloom_compute(context, &batch.descriptor, &options, &result.descriptor) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  bool unchanged = false;
  CUDA_CHECK(result.unchanged(unchanged));
  CHECK(unchanged);
  bool guards = false;
  CUDA_CHECK(result.guards_intact(guards));
  CHECK(guards);
  return 0;
}

int test_public_warm_start_transactions(std::int32_t device) {
  StreamOwner stream;
  CUDA_CHECK(stream.create());
  xtbloom_status_t context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle context = make_context(XTBLOOM_BACKEND_CUDA, device, stream.get(), context_status);
  CHECK(context_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(context != nullptr);
  xtbloom_status_t reference_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle fresh_reference =
      make_context(XTBLOOM_BACKEND_CUDA, device, stream.get(), reference_status);
  CHECK(reference_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(fresh_reference != nullptr);

  PublicBatch batch_a;
  CHECK(make_fixture_batch(1u, false, batch_a) == 0);
  xtbloom_compute_options_t fresh_options = make_compute_options();
  fresh_options.scc_start_mode = XTBLOOM_SCC_START_FRESH;
  xtbloom_compute_options_t warm_options = fresh_options;
  warm_options.scc_start_mode = XTBLOOM_SCC_START_WARM;

  /* First-call WARM is rejected before any runtime or caller-output commit.
   * Exercise every public output placement because rollback is part of the ABI. */
  std::string scenario;
  for (const auto& [layout, name] :
       {std::pair{ResultLayout::kHost, "host"}, std::pair{ResultLayout::kDevice, "device"},
        std::pair{ResultLayout::kMixed, "mixed"}}) {
    scenario = std::string("warm/first-call-rejection/") + name;
    g_scenario = scenario.c_str();
    CHECK(expect_strict_warm_rejection(context.get(), batch_a, warm_options, layout) == 0);
  }

  /* A successful public call may still contain a data-level SCC failure. It
   * must not advertise a complete batch checkpoint to the next strict WARM
   * call merely because result publication itself succeeded. */
  g_scenario = "warm/reject-after-nonconverged-fresh";
  xtbloom_status_t nonconverged_context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle nonconverged_context =
      make_context(XTBLOOM_BACKEND_CUDA, device, stream.get(), nonconverged_context_status);
  CHECK(nonconverged_context_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(nonconverged_context != nullptr);
  PublicBatch nonconverged_batch;
  CHECK(make_fixture_batch(1u, false, nonconverged_batch) == 0);
  xtbloom_compute_options_t one_iteration_fresh = fresh_options;
  one_iteration_fresh.max_scc_iterations = 1;
  ResultOwner nonconverged_owner;
  CUDA_CHECK(
      nonconverged_owner.bind(nonconverged_batch, ResultLayout::kHost, one_iteration_fresh.flags));
  CHECK(xtbloom_compute(nonconverged_context.get(), &nonconverged_batch.descriptor,
                        &one_iteration_fresh,
                        &nonconverged_owner.descriptor) == XTBLOOM_STATUS_SUCCESS);
  MaterializedResult nonconverged;
  CUDA_CHECK(nonconverged_owner.materialize(nonconverged));
  CHECK(nonconverged.statuses[0] == XTBLOOM_STATUS_SCC_NOT_CONVERGED);
  CHECK(nonconverged.converged[0] == 0u);
  CHECK(nonconverged.iterations[0] == 1);
  xtbloom_compute_options_t one_iteration_warm = one_iteration_fresh;
  one_iteration_warm.scc_start_mode = XTBLOOM_SCC_START_WARM;
  CHECK(expect_strict_warm_rejection(nonconverged_context.get(), nonconverged_batch,
                                     one_iteration_warm, ResultLayout::kMixed) == 0);

  g_scenario = "warm/fresh-A";
  bind_inputs(batch_a, nullptr, InputLayout::kHost);
  ResultOwner fresh_a_owner;
  CUDA_CHECK(fresh_a_owner.bind(batch_a, ResultLayout::kHost, fresh_options.flags));
  CHECK(xtbloom_compute(context.get(), &batch_a.descriptor, &fresh_options,
                        &fresh_a_owner.descriptor) == XTBLOOM_STATUS_SUCCESS);
  MaterializedResult fresh_a;
  CUDA_CHECK(fresh_a_owner.materialize(fresh_a));
  CHECK(compare_result(fresh_a_owner, fresh_a, fresh_a, fresh_options) == 0);

  g_scenario = "warm/warm-A";
  ResultOwner warm_a_owner;
  CUDA_CHECK(warm_a_owner.bind(batch_a, ResultLayout::kDevice, warm_options.flags));
  CHECK(xtbloom_compute(context.get(), &batch_a.descriptor, &warm_options,
                        &warm_a_owner.descriptor) == XTBLOOM_STATUS_SUCCESS);
  MaterializedResult warm_a;
  CUDA_CHECK(warm_a_owner.materialize(warm_a));
  CHECK(compare_result(warm_a_owner, warm_a, fresh_a, warm_options) == 0);
  for (std::size_t system = 0; system < warm_a.iterations.size(); ++system) {
    CHECK(warm_a.iterations[system] <= fresh_a.iterations[system]);
  }

  /* Isolate the changed-geometry contract from the preceding same-geometry
   * warm probe: publish a fresh A checkpoint, then consume that checkpoint
   * exactly once for B. */
  g_scenario = "warm/reseed-fresh-A-for-B";
  ResultOwner reseeded_a_owner;
  CUDA_CHECK(reseeded_a_owner.bind(batch_a, ResultLayout::kHost, fresh_options.flags));
  CHECK(xtbloom_compute(context.get(), &batch_a.descriptor, &fresh_options,
                        &reseeded_a_owner.descriptor) == XTBLOOM_STATUS_SUCCESS);
  MaterializedResult reseeded_a;
  CUDA_CHECK(reseeded_a_owner.materialize(reseeded_a));
  CHECK(compare_result(reseeded_a_owner, reseeded_a, fresh_a, fresh_options) == 0);

  PublicBatch batch_b = batch_a;
  batch_b.perturb(0.002);
  g_scenario = "warm/independent-fresh-B";
  bind_inputs(batch_b, nullptr, InputLayout::kHost);
  ResultOwner fresh_b_owner;
  CUDA_CHECK(fresh_b_owner.bind(batch_b, ResultLayout::kHost, fresh_options.flags));
  CHECK(xtbloom_compute(fresh_reference.get(), &batch_b.descriptor, &fresh_options,
                        &fresh_b_owner.descriptor) == XTBLOOM_STATUS_SUCCESS);
  MaterializedResult fresh_b;
  CUDA_CHECK(fresh_b_owner.materialize(fresh_b));
  CHECK(compare_result(fresh_b_owner, fresh_b, fresh_b, fresh_options) == 0);

  g_scenario = "warm/warm-B";
  ResultOwner warm_b_owner;
  CUDA_CHECK(warm_b_owner.bind(batch_b, ResultLayout::kMixed, warm_options.flags));
  CHECK(xtbloom_compute(context.get(), &batch_b.descriptor, &warm_options,
                        &warm_b_owner.descriptor) == XTBLOOM_STATUS_SUCCESS);
  MaterializedResult warm_b;
  CUDA_CHECK(warm_b_owner.materialize(warm_b));
  CHECK(compare_result(warm_b_owner, warm_b, fresh_b, warm_options) == 0);
  for (std::size_t system = 0; system < warm_b.iterations.size(); ++system) {
    if (warm_b.iterations[system] > fresh_b.iterations[system]) {
      std::fprintf(stderr,
                   "changed-geometry WARM iteration regression in %s at system %zu: warm=%d "
                   "fresh=%d\n",
                   g_scenario, system, warm_b.iterations[system], fresh_b.iterations[system]);
    }
    CHECK(warm_b.iterations[system] <= fresh_b.iterations[system]);
  }

  /* Strict mode never constructs a replacement runtime. Each rejected request
   * must leave both caller outputs and the B checkpoint untouched. */
  PublicBatch topology_changed;
  CHECK(make_fixture_batch(1u, true, topology_changed) == 0);
  g_scenario = "warm/reject-topology";
  CHECK(expect_strict_warm_rejection(context.get(), topology_changed, warm_options,
                                     ResultLayout::kHost) == 0);

  PublicBatch charge_changed = batch_b;
  charge_changed.molecular_charges[0] += 0.125;
  charge_changed.bind();
  g_scenario = "warm/reject-charge";
  CHECK(expect_strict_warm_rejection(context.get(), charge_changed, warm_options,
                                     ResultLayout::kDevice) == 0);

  PublicBatch unpaired_changed = batch_b;
  unpaired_changed.unpaired_electrons[0] += 1;
  unpaired_changed.bind();
  g_scenario = "warm/reject-unpaired";
  CHECK(expect_strict_warm_rejection(context.get(), unpaired_changed, warm_options,
                                     ResultLayout::kMixed) == 0);

  xtbloom_compute_options_t temperature_changed = warm_options;
  temperature_changed.electronic_temperature = 300.0;
  g_scenario = "warm/reject-temperature";
  CHECK(expect_strict_warm_rejection(context.get(), batch_b, temperature_changed,
                                     ResultLayout::kHost) == 0);

  xtbloom_compute_options_t policy_changed = warm_options;
  policy_changed.max_scc_iterations -= 1;
  g_scenario = "warm/reject-policy";
  CHECK(expect_strict_warm_rejection(context.get(), batch_b, policy_changed,
                                     ResultLayout::kDevice) == 0);

  xtbloom_compute_options_t mixer_history_changed = warm_options;
  mixer_history_changed.scc_mixer_history = 4;
  g_scenario = "warm/reject-mixer-history";
  CHECK(expect_strict_warm_rejection(context.get(), batch_b, mixer_history_changed,
                                     ResultLayout::kHost) == 0);

  xtbloom_compute_options_t mixer_damping_changed = warm_options;
  mixer_damping_changed.scc_mixer_damping = 0.2;
  g_scenario = "warm/reject-mixer-damping";
  CHECK(expect_strict_warm_rejection(context.get(), batch_b, mixer_damping_changed,
                                     ResultLayout::kDevice) == 0);

  xtbloom_compute_options_t determinism_changed = warm_options;
  determinism_changed.determinism = XTBLOOM_DETERMINISM_REPRODUCIBLE;
  g_scenario = "warm/reject-determinism";
  CHECK(expect_strict_warm_rejection(context.get(), batch_b, determinism_changed,
                                     ResultLayout::kMixed) == 0);

  g_scenario = "warm/checkpoint-survives-rejections";
  ResultOwner preserved_owner;
  CUDA_CHECK(preserved_owner.bind(batch_b, ResultLayout::kMixed, warm_options.flags));
  CHECK(xtbloom_compute(context.get(), &batch_b.descriptor, &warm_options,
                        &preserved_owner.descriptor) == XTBLOOM_STATUS_SUCCESS);
  MaterializedResult preserved;
  CUDA_CHECK(preserved_owner.materialize(preserved));
  CHECK(compare_result(preserved_owner, preserved, fresh_b, warm_options) == 0);

  /* Once the caller explicitly requests FRESH, the previously rejected
   * topology may replace the runtime and converge normally. */
  g_scenario = "warm/fresh-recovery-after-rejection";
  bind_inputs(topology_changed, nullptr, InputLayout::kHost);
  ResultOwner recovered_owner;
  CUDA_CHECK(recovered_owner.bind(topology_changed, ResultLayout::kDevice, fresh_options.flags));
  CHECK(xtbloom_compute(context.get(), &topology_changed.descriptor, &fresh_options,
                        &recovered_owner.descriptor) == XTBLOOM_STATUS_SUCCESS);
  MaterializedResult recovered;
  CUDA_CHECK(recovered_owner.materialize(recovered));
  CHECK(compare_result(recovered_owner, recovered, recovered, fresh_options) == 0);
  return 0;
}

int test_host_device_mixed_and_streams(std::int32_t device, PublicBatch& batch,
                                       const xtbloom_compute_options_t& options,
                                       const MaterializedResult& reference) {
  g_scenario = "host-output/default-stream";
  xtbloom_status_t context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle default_context =
      make_context(XTBLOOM_BACKEND_CUDA, device, nullptr, context_status);
  CHECK(context_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(default_context != nullptr);
  CHECK(execute_cuda_and_compare(default_context.get(), batch, options, ResultLayout::kHost,
                                 reference) == 0);

  g_scenario = "host-output/custom-stream";
  StreamOwner stream;
  CUDA_CHECK(stream.create());
  ContextHandle custom_context =
      make_context(XTBLOOM_BACKEND_CUDA, device, stream.get(), context_status);
  CHECK(context_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(custom_context != nullptr);
  CHECK(execute_cuda_and_compare(custom_context.get(), batch, options, ResultLayout::kHost,
                                 reference) == 0);

  g_scenario = "device-output/custom-stream";
  CHECK(execute_cuda_and_compare(custom_context.get(), batch, options, ResultLayout::kDevice,
                                 reference) == 0);

  g_scenario = "mixed-output/custom-stream";
  CHECK(execute_cuda_and_compare(custom_context.get(), batch, options, ResultLayout::kMixed,
                                 reference) == 0);
  return 0;
}

struct RequestDeleter {
  void operator()(xtbloom_request_t* request) const noexcept { xtbloom_request_destroy(request); }
};

using RequestHandle = std::unique_ptr<xtbloom_request_t, RequestDeleter>;

/* A data-level numerical failure belongs to one ragged member, not the whole
 * public call. Use geometries from the pinned H3+ and OH trace corpus, then
 * establish the mixed terminal outcome independently through the public CPU
 * path under the exact public options used below. Exercise the caller-visible
 * transaction for every supported input/output placement: the healthy peer
 * must retain CPU-reference values, while the nonconverged peer publishes its
 * terminal status and complete quiet-NaN floating-point slices. Guard bytes
 * around every caller allocation must remain intact. */
int test_public_peer_failure_isolated(std::int32_t device, xtbloom_context_t* cpu_context,
                                      const xtbloom_compute_options_t& options) {
  xtbloom_compute_options_t failure_options = options;
  failure_options.max_scc_iterations = 3;
  failure_options.charge_tolerance = 2.0e-5;
  failure_options.energy_tolerance = 1.0e-6;
  failure_options.electronic_temperature = XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE;

  PublicBatch healthy;
  healthy.atom_offsets = {0, 3};
  healthy.atomic_numbers = {1, 1, 1};
  healthy.positions = {
      -0.47073898552969,
      0.81534384004086,
      0.0,
      -0.47073898552969,
      -0.81534384004086,
      0.0,
      0.94147797105939,
      0.0,
      0.0,
  };
  healthy.molecular_charges = {1.0};
  healthy.unpaired_electrons = {0};
  healthy.spin_channels = {1};
  healthy.bind();
  MaterializedResult healthy_reference;
  g_scenario = "peer-failure/CPU-reference";
  CHECK(run_cpu_reference(cpu_context, healthy, failure_options, healthy_reference) == 0);
  CHECK(healthy_reference.iterations[0] == failure_options.max_scc_iterations);

  PublicBatch batch;
  batch.atom_offsets = {0, 3, 5};
  batch.atomic_numbers = {1, 1, 1, 8, 1};
  batch.positions = {
      healthy.positions[0],
      healthy.positions[1],
      healthy.positions[2],
      healthy.positions[3],
      healthy.positions[4],
      healthy.positions[5],
      healthy.positions[6],
      healthy.positions[7],
      healthy.positions[8],
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      1.834,
  };
  batch.molecular_charges = {1.0, 0.0};
  batch.unpaired_electrons = {0, 1};
  batch.spin_channels = {1, 2};
  batch.bind();

  MaterializedResult failure_reference;
  {
    g_scenario = "peer-failure/CPU-mixed-reference";
    ResultOwner result;
    bind_inputs(batch, nullptr, InputLayout::kHost);
    CUDA_CHECK(result.bind(batch, ResultLayout::kHost, failure_options.flags));
    CHECK(xtbloom_compute(cpu_context, &batch.descriptor, &failure_options, &result.descriptor) ==
          XTBLOOM_STATUS_SUCCESS);
    CUDA_CHECK(result.materialize(failure_reference));
    bool guards = false;
    CUDA_CHECK(result.guards_intact(guards));
    CHECK(guards);
  }
  CHECK(failure_reference.statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(failure_reference.converged[0] == 1u);
  CHECK(failure_reference.iterations[0] == healthy_reference.iterations[0]);
  CHECK(near(failure_reference.energies[0], healthy_reference.energies[0], kEnergyAbsoluteTolerance,
             kEnergyRelativeTolerance));
  for (std::size_t atom = 0u; atom < 3u; ++atom) {
    CHECK(near(failure_reference.atomic_charges[atom], healthy_reference.atomic_charges[atom],
               kChargeAbsoluteTolerance, kChargeRelativeTolerance));
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
      const std::size_t coordinate = 3u * atom + axis;
      CHECK(near(failure_reference.forces[coordinate], healthy_reference.forces[coordinate],
                 kForceAbsoluteTolerance, kForceRelativeTolerance));
    }
  }
  CHECK(failure_reference.statuses[1] == XTBLOOM_STATUS_SCC_NOT_CONVERGED);
  CHECK(failure_reference.converged[1] == 0u);
  CHECK(failure_reference.iterations[1] == failure_options.max_scc_iterations);
  CHECK(is_quiet_nan(failure_reference.energies[1]));
  for (std::size_t atom = 3u; atom < 5u; ++atom) {
    CHECK(is_quiet_nan(failure_reference.atomic_charges[atom]));
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
      CHECK(is_quiet_nan(failure_reference.forces[3u * atom + axis]));
    }
  }

  StreamOwner stream;
  CUDA_CHECK(stream.create());
  xtbloom_status_t context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle context = make_context(XTBLOOM_BACKEND_CUDA, device, stream.get(), context_status);
  CHECK(context_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(context != nullptr);

  DeviceBatchInputs device_inputs;
  CUDA_CHECK(device_inputs.upload_all(batch));
  const std::array<std::pair<InputLayout, ResultLayout>, 3> layouts{{
      {InputLayout::kHost, ResultLayout::kHost},
      {InputLayout::kDevice, ResultLayout::kDevice},
      {InputLayout::kMixed, ResultLayout::kMixed},
  }};
  const std::array<const char*, 3> names{{"host", "device", "mixed"}};
  std::string scenario;
  for (std::size_t layout_index = 0u; layout_index < layouts.size(); ++layout_index) {
    scenario = std::string("peer-failure/") + names[layout_index];
    g_scenario = scenario.c_str();
    const auto [input_layout, result_layout] = layouts[layout_index];
    bind_inputs(batch, input_layout == InputLayout::kHost ? nullptr : &device_inputs, input_layout);
    if (input_layout != InputLayout::kHost) {
      CHECK(verify_input_layout(batch, input_layout, false) == 0);
    }

    ResultOwner owner;
    CUDA_CHECK(owner.bind(batch, result_layout, failure_options.flags));
    CHECK(xtbloom_compute(context.get(), &batch.descriptor, &failure_options, &owner.descriptor) ==
          XTBLOOM_STATUS_SUCCESS);
    MaterializedResult actual;
    CUDA_CHECK(owner.materialize(actual));
    bool guards = false;
    CUDA_CHECK(owner.guards_intact(guards));
    CHECK(guards);

    CHECK(actual.flags == failure_reference.flags);
    CHECK(actual.statuses == failure_reference.statuses);
    CHECK(actual.converged == failure_reference.converged);
    CHECK(actual.iterations == failure_reference.iterations);
    CHECK(near(actual.energies[0], failure_reference.energies[0], kEnergyAbsoluteTolerance,
               kEnergyRelativeTolerance));
    for (std::size_t atom = 0u; atom < 3u; ++atom) {
      CHECK(near(actual.atomic_charges[atom], failure_reference.atomic_charges[atom],
                 kChargeAbsoluteTolerance, kChargeRelativeTolerance));
      for (std::size_t axis = 0u; axis < 3u; ++axis) {
        const std::size_t coordinate = 3u * atom + axis;
        CHECK(near(actual.forces[coordinate], failure_reference.forces[coordinate],
                   kForceAbsoluteTolerance, kForceRelativeTolerance));
      }
    }

    CHECK(is_quiet_nan(actual.energies[1]));
    for (std::size_t atom = 3u; atom < 5u; ++atom) {
      CHECK(is_quiet_nan(actual.atomic_charges[atom]));
      for (std::size_t axis = 0u; axis < 3u; ++axis) {
        CHECK(is_quiet_nan(actual.forces[3u * atom + axis]));
      }
    }
  }

  /* The context request path preserves the same data-level failure contract:
   * completion succeeds as a call, diagnostics identify the failed peer, and
   * every requested floating-point slice for that peer is a complete qNaN. */
  g_scenario = "peer-failure/context-enqueue-mixed";
  bind_inputs(batch, &device_inputs, InputLayout::kMixed);
  ResultOwner async_owner;
  CUDA_CHECK(async_owner.bind(batch, ResultLayout::kMixed, failure_options.flags));
  xtbloom_request_t* raw_request = nullptr;
  CHECK(xtbloom_request_create(context.get(), &raw_request) == XTBLOOM_STATUS_SUCCESS);
  RequestHandle request(raw_request);
  CHECK(xtbloom_compute_enqueue(context.get(), &batch.descriptor, &failure_options,
                                &async_owner.descriptor, request.get()) == XTBLOOM_STATUS_SUCCESS);
  xtbloom_request_info_t info{};
  CHECK(xtbloom_request_info_init(&info, sizeof(info)) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom_request_wait(request.get(), &info) == XTBLOOM_STATUS_SUCCESS);
  CHECK(info.completion_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(info.result_flags == failure_reference.flags);
  MaterializedResult async_actual;
  CUDA_CHECK(async_owner.materialize(async_actual));
  CHECK(async_actual.statuses == failure_reference.statuses);
  CHECK(async_actual.converged == failure_reference.converged);
  CHECK(async_actual.iterations == failure_reference.iterations);
  CHECK(near(async_actual.energies[0], failure_reference.energies[0], kEnergyAbsoluteTolerance,
             kEnergyRelativeTolerance));
  CHECK(is_quiet_nan(async_actual.energies[1]));
  for (std::size_t atom = 3u; atom < 5u; ++atom) {
    CHECK(is_quiet_nan(async_actual.atomic_charges[atom]));
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
      CHECK(is_quiet_nan(async_actual.forces[3u * atom + axis]));
    }
  }
  bool async_guards = false;
  CUDA_CHECK(async_owner.guards_intact(async_guards));
  CHECK(async_guards);
  CHECK(async_owner.descriptor.flags == kResultFlagsCanary);

  /* A call-level successful request with one nonconverged peer cannot publish
   * a complete-batch checkpoint. Reusing the same request for strict WARM
   * therefore rejects at admission, preserves the completed request snapshot,
   * and leaves every new result byte untouched. */
  xtbloom_compute_options_t warm_failure_options = failure_options;
  warm_failure_options.scc_start_mode = XTBLOOM_SCC_START_WARM;
  ResultOwner rejected_warm_owner;
  CUDA_CHECK(rejected_warm_owner.bind(batch, ResultLayout::kDevice, warm_failure_options.flags));
  CHECK(xtbloom_compute_enqueue(context.get(), &batch.descriptor, &warm_failure_options,
                                &rejected_warm_owner.descriptor,
                                request.get()) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::strstr(xtbloom_get_last_error(), "preceding successful public checkpoint") != nullptr);
  xtbloom_request_info_t preserved_info{};
  CHECK(xtbloom_request_info_init(&preserved_info, sizeof(preserved_info)) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom_request_query(request.get(), &preserved_info) == XTBLOOM_STATUS_SUCCESS);
  CHECK(preserved_info.state == XTBLOOM_REQUEST_COMPLETE);
  CHECK(preserved_info.completion_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(preserved_info.result_flags == failure_reference.flags);
  bool rejected_warm_unchanged = false;
  CUDA_CHECK(rejected_warm_owner.unchanged(rejected_warm_unchanged));
  CHECK(rejected_warm_unchanged);
  return 0;
}

bool is_cuda_buffer(const xtbloom_const_buffer_t& buffer) noexcept {
  return buffer.data != nullptr && buffer.memory_space == XTBLOOM_MEMORY_CUDA_DEVICE;
}

bool is_host_buffer(const xtbloom_const_buffer_t& buffer) noexcept {
  return buffer.data != nullptr && buffer.memory_space == XTBLOOM_MEMORY_HOST;
}

struct PlanDeleter {
  void operator()(xtbloom_plan_t* plan) const noexcept { xtbloom_plan_destroy(plan); }
};

using PlanHandle = std::unique_ptr<xtbloom_plan_t, PlanDeleter>;

enum class PlanTestMode { kFull, kRequestOnly, kSanitizer, kProfileSteadyState };

/* The context convenience entry point owns the same request/publication
 * protocol as fixed plans, but may build a topology runtime on its first call.
 * Prewarm synchronously so the blocked-stream check isolates steady-state
 * enqueue rather than setup admission. */
int test_cuda_context_enqueue(std::int32_t device, xtbloom_context_t* cpu_context,
                              const xtbloom_compute_options_t& base_options,
                              PlanTestMode mode = PlanTestMode::kFull) {
  g_scenario = "context-enqueue";
  StreamOwner stream;
  CUDA_CHECK(stream.create());
  xtbloom_status_t context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle context = make_context(XTBLOOM_BACKEND_CUDA, device, stream.get(), context_status);
  CHECK(context_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(context != nullptr);

  PublicBatch batch;
  CHECK(make_fixture_batch(4u, true, batch) == 0);
  batch.spin_channels.back() = 2;
  batch.unpaired_electrons.back() = 2;
  batch.bind();
  xtbloom_compute_options_t options = base_options;
  options.flags |= XTBLOOM_COMPUTE_POINT_CHARGE_FORCES;
  MaterializedResult reference;
  CHECK(run_cpu_reference(cpu_context, batch, options, reference) == 0);

  xtbloom_compute_options_t first_warm_options = options;
  first_warm_options.scc_start_mode = XTBLOOM_SCC_START_WARM;
  g_scenario = "context-enqueue/first-warm-rejection";
  ResultOwner first_warm_result;
  CUDA_CHECK(first_warm_result.bind(batch, ResultLayout::kMixed, first_warm_options.flags));
  xtbloom_request_t* raw_first_warm_request = nullptr;
  CHECK(xtbloom_request_create(context.get(), &raw_first_warm_request) == XTBLOOM_STATUS_SUCCESS);
  RequestHandle first_warm_request(raw_first_warm_request);
  CHECK(xtbloom_compute_enqueue(context.get(), &batch.descriptor, &first_warm_options,
                                &first_warm_result.descriptor,
                                first_warm_request.get()) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::strstr(xtbloom_get_last_error(), "existing compatible prepared runtime") != nullptr);
  xtbloom_request_info_t first_warm_info{};
  CHECK(xtbloom_request_info_init(&first_warm_info, sizeof(first_warm_info)) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom_request_query(first_warm_request.get(), &first_warm_info) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(first_warm_info.state == XTBLOOM_REQUEST_IDLE);
  bool first_warm_unchanged = false;
  CUDA_CHECK(first_warm_result.unchanged(first_warm_unchanged));
  CHECK(first_warm_unchanged);

  if (mode == PlanTestMode::kProfileSteadyState) {
    DeviceBatchInputs inputs;
    CUDA_CHECK(inputs.upload_all(batch));
    bind_inputs(batch, &inputs, InputLayout::kMixed);
    const auto input_identity = inputs.identity();
    ResultOwner result;
    CUDA_CHECK(result.bind(batch, ResultLayout::kTorchRequest, options.flags));
    xtbloom_request_t* raw_request = nullptr;
    CHECK(xtbloom_request_create(context.get(), &raw_request) == XTBLOOM_STATUS_SUCCESS);
    RequestHandle request(raw_request);
    xtbloom_request_info_t info{};
    CHECK(xtbloom_request_info_init(&info, sizeof(info)) == XTBLOOM_STATUS_SUCCESS);

    /* Prepare the context cache and consume one asynchronous warmup before the
     * profiler range. The measured iterations therefore isolate reusable
     * same-topology admission, request completion, and caller publication. */
    CHECK(xtbloom_compute(context.get(), &batch.descriptor, &options, &result.descriptor) ==
          XTBLOOM_STATUS_SUCCESS);
    /* Synchronous compute publishes through the descriptor. Restore the
     * canary before profiling asynchronous calls so this mode specifically
     * proves that enqueue completion uses request_info.result_flags instead. */
    result.descriptor.flags = kResultFlagsCanary;
    xtbloom_compute_options_t warm_options = options;
    warm_options.scc_start_mode = XTBLOOM_SCC_START_WARM;
    CHECK(xtbloom_compute_enqueue(context.get(), &batch.descriptor, &warm_options,
                                  &result.descriptor, request.get()) == XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom_request_wait(request.get(), &info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(info.completion_status == XTBLOOM_STATUS_SUCCESS);

    constexpr int kProfileIterations = 10;
    CUDA_CHECK(cudaProfilerStart());
    for (int iteration = 0; iteration < kProfileIterations; ++iteration) {
      CHECK(xtbloom_compute_enqueue(context.get(), &batch.descriptor, &warm_options,
                                    &result.descriptor, request.get()) == XTBLOOM_STATUS_SUCCESS);
      CHECK(xtbloom_request_wait(request.get(), &info) == XTBLOOM_STATUS_SUCCESS);
      CHECK(info.state == XTBLOOM_REQUEST_COMPLETE);
      CHECK(info.completion_status == XTBLOOM_STATUS_SUCCESS);
      CHECK(info.result_flags == reference.flags);
    }
    CUDA_CHECK(cudaProfilerStop());
    CHECK(inputs.identity() == input_identity);
    MaterializedResult actual;
    CUDA_CHECK(result.materialize(actual));
    actual.flags = info.result_flags;
    CHECK(compare_result(result, actual, reference, warm_options) == 0);
    CHECK(result.descriptor.flags == kResultFlagsCanary);
    std::printf("context_request_profile_iterations=%d\n", kProfileIterations);
    return 0;
  }

  /* A fresh context may allocate and perform bounded setup admission, but the
   * accepted request still owns completion and publishes flags separately. */
  {
    StreamOwner first_stream;
    CUDA_CHECK(first_stream.create());
    xtbloom_status_t first_status = XTBLOOM_STATUS_INTERNAL_ERROR;
    ContextHandle first_context =
        make_context(XTBLOOM_BACKEND_CUDA, device, first_stream.get(), first_status);
    CHECK(first_status == XTBLOOM_STATUS_SUCCESS);
    ResultOwner first_result;
    CUDA_CHECK(first_result.bind(batch, ResultLayout::kHost, options.flags));
    xtbloom_batch_result_t first_submitted_result = first_result.descriptor;
    xtbloom_request_t* raw_first_request = nullptr;
    CHECK(xtbloom_request_create(first_context.get(), &raw_first_request) ==
          XTBLOOM_STATUS_SUCCESS);
    RequestHandle first_request(raw_first_request);
    CHECK(xtbloom_compute_enqueue(first_context.get(), &batch.descriptor, &options,
                                  &first_submitted_result,
                                  first_request.get()) == XTBLOOM_STATUS_SUCCESS);
    xtbloom_request_info_t first_info{};
    CHECK(xtbloom_request_info_init(&first_info, sizeof(first_info)) == XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom_request_wait(first_request.get(), &first_info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(first_info.completion_status == XTBLOOM_STATUS_SUCCESS);
    MaterializedResult first_actual;
    CUDA_CHECK(first_result.materialize(first_actual));
    first_actual.flags = first_info.result_flags;
    CHECK(compare_result(first_result, first_actual, reference, options) == 0);
    CHECK(first_submitted_result.flags == kResultFlagsCanary);
  }

  ResultOwner prewarm_result;
  CUDA_CHECK(prewarm_result.bind(batch, ResultLayout::kDevice, options.flags));
  CHECK(xtbloom_compute(context.get(), &batch.descriptor, &options, &prewarm_result.descriptor) ==
        XTBLOOM_STATUS_SUCCESS);
  xtbloom_compute_options_t warm_options = options;
  warm_options.scc_start_mode = XTBLOOM_SCC_START_WARM;

  for (const auto [layout, name] :
       {std::pair{InputLayout::kDevice, "device"}, std::pair{InputLayout::kMixed, "mixed"}}) {
    std::string scenario = std::string("context-enqueue/") + name;
    g_scenario = scenario.c_str();
    PublicBatch placed = batch;
    DeviceBatchInputs placed_inputs;
    CUDA_CHECK(placed_inputs.upload_all(placed));
    bind_inputs(placed, &placed_inputs, layout);
    ResultOwner placed_result;
    CUDA_CHECK(placed_result.bind(placed, ResultLayout::kDevice, options.flags));
    xtbloom_request_t* raw_placed_request = nullptr;
    CHECK(xtbloom_request_create(context.get(), &raw_placed_request) == XTBLOOM_STATUS_SUCCESS);
    RequestHandle placed_request(raw_placed_request);
    CHECK(xtbloom_compute_enqueue(context.get(), &placed.descriptor, &options,
                                  &placed_result.descriptor,
                                  placed_request.get()) == XTBLOOM_STATUS_SUCCESS);
    xtbloom_request_info_t placed_info{};
    CHECK(xtbloom_request_info_init(&placed_info, sizeof(placed_info)) == XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom_request_wait(placed_request.get(), &placed_info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(placed_info.completion_status == XTBLOOM_STATUS_SUCCESS);
    MaterializedResult placed_actual;
    CUDA_CHECK(placed_result.materialize(placed_actual));
    placed_actual.flags = placed_info.result_flags;
    CHECK(compare_result(placed_result, placed_actual, reference, options) == 0);
  }
  g_scenario = "context-enqueue";

  /* Topology ownership and numerical placement are independent. A changed
   * host topology with device numerical leaves must build a new context
   * runtime instead of being mistaken for an invalid fixed-topology reuse. */
  if (mode != PlanTestMode::kSanitizer) {
    PublicBatch hybrid = batch;
    /* Keep every declared extent and compute-policy field unchanged so the
     * old numerical-placement-sensitive probe would incorrectly select the
     * fixed-topology path instead of rebuilding this legitimate topology. */
    hybrid.molecular_charges.front() += 2.0;
    hybrid.bind();
    xtbloom_compute_options_t hybrid_options = options;
    MaterializedResult hybrid_reference;
    CHECK(run_cpu_reference(cpu_context, hybrid, hybrid_options, hybrid_reference) == 0);
    DeviceBatchInputs hybrid_inputs;
    CUDA_CHECK(hybrid_inputs.upload_all(hybrid));
    bind_inputs(hybrid, &hybrid_inputs, InputLayout::kMixed);
    /* Mixed placement already leaves some numerical fields on host. Force the
     * principal geometry leaf to device while restoring every topology leaf
     * to its host descriptor image. */
    hybrid.descriptor.atom_offsets = host_input(hybrid.atom_offsets);
    hybrid.descriptor.atomic_numbers = host_input(hybrid.atomic_numbers);
    hybrid.descriptor.molecular_charges = host_input(hybrid.molecular_charges);
    hybrid.descriptor.unpaired_electrons = host_input(hybrid.unpaired_electrons);
    hybrid.descriptor.spin_channels = host_input(hybrid.spin_channels);
    hybrid.descriptor.point_charge_offsets = host_input(hybrid.point_charge_offsets);
    hybrid.descriptor.charge_response_offsets = host_input(hybrid.charge_response_offsets);
    ResultOwner hybrid_result;
    CUDA_CHECK(hybrid_result.bind(hybrid, ResultLayout::kMixed, hybrid_options.flags));
    xtbloom_request_t* raw_hybrid_request = nullptr;
    CHECK(xtbloom_request_create(context.get(), &raw_hybrid_request) == XTBLOOM_STATUS_SUCCESS);
    RequestHandle hybrid_request(raw_hybrid_request);
    CHECK(xtbloom_compute_enqueue(context.get(), &hybrid.descriptor, &hybrid_options,
                                  &hybrid_result.descriptor,
                                  hybrid_request.get()) == XTBLOOM_STATUS_SUCCESS);
    xtbloom_request_info_t hybrid_info{};
    CHECK(xtbloom_request_info_init(&hybrid_info, sizeof(hybrid_info)) == XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom_request_wait(hybrid_request.get(), &hybrid_info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(hybrid_info.completion_status == XTBLOOM_STATUS_SUCCESS);
    MaterializedResult hybrid_actual;
    CUDA_CHECK(hybrid_result.materialize(hybrid_actual));
    hybrid_actual.flags = hybrid_info.result_flags;
    CHECK(compare_result(hybrid_result, hybrid_actual, hybrid_reference, hybrid_options) == 0);

    /* Restore the original topology for the same-shape device comparison and
     * blocked-stream steady-state cases below. */
    CHECK(xtbloom_compute(context.get(), &batch.descriptor, &options, &prewarm_result.descriptor) ==
          XTBLOOM_STATUS_SUCCESS);
  }

  if (mode != PlanTestMode::kSanitizer) {
    /* A device descriptor with a declared response extent that differs from
     * the prepared dense topology must enter bounded staging validation, not
     * the deferred same-shape comparison. Rejection remains pre-admission and
     * preserves the reusable request's public snapshot and every output. */
    PublicBatch response_changed = batch;
    CHECK(response_changed.descriptor.total_charge_response_elements > 1);
    --response_changed.descriptor.total_charge_response_elements;
    response_changed.charge_response_matrix.pop_back();
    response_changed.charge_response_offsets.back() =
        response_changed.descriptor.total_charge_response_elements;
    response_changed.bind();
    DeviceBatchInputs response_inputs;
    CUDA_CHECK(response_inputs.upload_all(response_changed));
    bind_inputs(response_changed, &response_inputs, InputLayout::kDevice);
    ResultOwner response_result;
    CUDA_CHECK(response_result.bind(response_changed, ResultLayout::kDevice, options.flags));
    xtbloom_request_t* raw_response_request = nullptr;
    CHECK(xtbloom_request_create(context.get(), &raw_response_request) == XTBLOOM_STATUS_SUCCESS);
    RequestHandle response_request(raw_response_request);
    CHECK(xtbloom_compute_enqueue(context.get(), &response_changed.descriptor, &options,
                                  &response_result.descriptor,
                                  response_request.get()) == XTBLOOM_STATUS_INVALID_ARGUMENT);
    xtbloom_request_info_t response_info{};
    CHECK(xtbloom_request_info_init(&response_info, sizeof(response_info)) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom_request_query(response_request.get(), &response_info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(response_info.state == XTBLOOM_REQUEST_IDLE);
    bool response_unchanged = false;
    CUDA_CHECK(response_result.unchanged(response_unchanged));
    CHECK(response_unchanged);
  }

  /* Same-shape device topology is compared on the owner stream. Mutating one
   * immutable field therefore admits successfully and completes with a
   * deferred INVALID_ARGUMENT while every result sentinel remains untouched. */
  {
    PublicBatch mismatch_batch = batch;
    DeviceBatchInputs mismatch_inputs;
    CUDA_CHECK(mismatch_inputs.upload_all(mismatch_batch));
    bind_inputs(mismatch_batch, &mismatch_inputs, InputLayout::kDevice);
    ResultOwner mismatch_result;
    CUDA_CHECK(mismatch_result.bind(mismatch_batch, ResultLayout::kDevice, options.flags));
    xtbloom_request_t* raw_mismatch_request = nullptr;
    CHECK(xtbloom_request_create(context.get(), &raw_mismatch_request) == XTBLOOM_STATUS_SUCCESS);
    RequestHandle mismatch_request(raw_mismatch_request);
    CUDA_CHECK(cudaMemsetAsync(const_cast<void*>(mismatch_batch.descriptor.spin_channels.data), 0,
                               mismatch_batch.descriptor.spin_channels.size_bytes, stream.get()));
    CHECK(xtbloom_compute_enqueue(context.get(), &mismatch_batch.descriptor, &warm_options,
                                  &mismatch_result.descriptor,
                                  mismatch_request.get()) == XTBLOOM_STATUS_SUCCESS);
    xtbloom_request_info_t mismatch_info{};
    CHECK(xtbloom_request_info_init(&mismatch_info, sizeof(mismatch_info)) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom_request_wait(mismatch_request.get(), &mismatch_info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(mismatch_info.completion_status == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(std::strstr(xtbloom_request_get_error(mismatch_request.get()),
                      "does not match the fixed CUDA plan topology") != nullptr);
    bool mismatch_unchanged = false;
    CUDA_CHECK(mismatch_result.unchanged(mismatch_unchanged));
    CHECK(mismatch_unchanged);
  }

  batch.perturb(0.003);
  CHECK(run_cpu_reference(cpu_context, batch, options, reference) == 0);
  ResultOwner result;
  CUDA_CHECK(result.bind(batch, ResultLayout::kDevice, options.flags));
  xtbloom_batch_result_t submitted_result = result.descriptor;
  /* These owners must outlive the reusable request because an early test
   * return after enqueue is settled by the request destructor. */
  ResultOwner warm_result;
  ResultOwner second_warm_result;
  ResultOwner post_destroy_warm_result;
  xtbloom_batch_result_t submitted_warm_result{};
  xtbloom_request_t* raw_request = nullptr;
  CHECK(xtbloom_request_create(context.get(), &raw_request) == XTBLOOM_STATUS_SUCCESS);
  RequestHandle request(raw_request);

  /* A request is permanently bound to its creating context. Rejection occurs
   * before either request or output state changes. */
  {
    StreamOwner other_stream;
    CUDA_CHECK(other_stream.create());
    xtbloom_status_t other_status = XTBLOOM_STATUS_INTERNAL_ERROR;
    ContextHandle other_context =
        make_context(XTBLOOM_BACKEND_CUDA, device, other_stream.get(), other_status);
    CHECK(other_status == XTBLOOM_STATUS_SUCCESS);
    xtbloom_request_t* raw_other_request = nullptr;
    CHECK(xtbloom_request_create(other_context.get(), &raw_other_request) ==
          XTBLOOM_STATUS_SUCCESS);
    RequestHandle other_request(raw_other_request);
    CHECK(xtbloom_compute_enqueue(context.get(), &batch.descriptor, &options, &submitted_result,
                                  other_request.get()) == XTBLOOM_STATUS_INVALID_ARGUMENT);
    xtbloom_request_info_t other_info{};
    CHECK(xtbloom_request_info_init(&other_info, sizeof(other_info)) == XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom_request_query(other_request.get(), &other_info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(other_info.state == XTBLOOM_REQUEST_IDLE);
  }

  xtbloom_request_info_t info{};
  CHECK(xtbloom_request_info_init(&info, sizeof(info)) == XTBLOOM_STATUS_SUCCESS);

  /* The deferred topology mismatch above consumed the preceding checkpoint.
   * Strict WARM therefore rejects at admission and leaves the reusable request
   * and every output sentinel untouched. */
  CHECK(xtbloom_compute_enqueue(context.get(), &batch.descriptor, &warm_options, &submitted_result,
                                request.get()) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::strstr(xtbloom_get_last_error(), "preceding successful public checkpoint") != nullptr);
  CHECK(xtbloom_request_query(request.get(), &info) == XTBLOOM_STATUS_SUCCESS);
  CHECK(info.state == XTBLOOM_REQUEST_IDLE);
  CHECK(submitted_result.flags == kResultFlagsCanary);

  /* Exact ABI-v1 storage ensures asynchronous settlement cannot read the
   * optional suffixes after enqueue has returned. */
  alignas(xtbloom_compute_options_t) std::array<unsigned char, XTBLOOM_COMPUTE_OPTIONS_V1_SIZE>
      short_options_storage{};
  std::memcpy(short_options_storage.data(), &options, short_options_storage.size());
  auto* const short_options =
      reinterpret_cast<xtbloom_compute_options_t*>(short_options_storage.data());
  short_options->struct_size = XTBLOOM_COMPUTE_OPTIONS_V1_SIZE;
  alignas(xtbloom_batch_result_t) std::array<unsigned char, XTBLOOM_BATCH_RESULT_V1_SIZE>
      short_result_storage{};
  std::memcpy(short_result_storage.data(), &submitted_result, short_result_storage.size());
  auto* const short_result = reinterpret_cast<xtbloom_batch_result_t*>(short_result_storage.data());
  short_result->struct_size = XTBLOOM_BATCH_RESULT_V1_SIZE;

  BlockingStreamGate gate;
  if (mode != PlanTestMode::kSanitizer) CUDA_CHECK(gate.arm(stream.get()));
  xtbloom_status_t enqueue_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  if (mode == PlanTestMode::kSanitizer) {
    enqueue_status = xtbloom_compute_enqueue(context.get(), &batch.descriptor, short_options,
                                             short_result, request.get());
  } else {
    bool enqueue_returned = false;
    std::mutex enqueue_mutex;
    std::condition_variable enqueue_changed;
    std::thread submitter([&] {
      enqueue_status = xtbloom_compute_enqueue(context.get(), &batch.descriptor, short_options,
                                               short_result, request.get());
      {
        std::lock_guard<std::mutex> lock(enqueue_mutex);
        enqueue_returned = true;
      }
      enqueue_changed.notify_one();
    });
    bool returned_while_blocked = false;
    {
      std::unique_lock<std::mutex> lock(enqueue_mutex);
      returned_while_blocked =
          enqueue_changed.wait_for(lock, kBlockedEnqueueWatchdog, [&] { return enqueue_returned; });
    }
    if (!returned_while_blocked) gate.release();
    submitter.join();
    CHECK(returned_while_blocked);
  }
  CHECK(enqueue_status == XTBLOOM_STATUS_SUCCESS);
  short_options->flags = 0u;
  short_result->energies.data = nullptr;
  CHECK(xtbloom_request_query(request.get(), &info) == XTBLOOM_STATUS_SUCCESS);
  if (mode == PlanTestMode::kSanitizer) {
    CHECK(info.state == XTBLOOM_REQUEST_PENDING || info.state == XTBLOOM_REQUEST_COMPLETE);
  } else {
    CHECK(info.state == XTBLOOM_REQUEST_PENDING);
  }
  CHECK(short_result->flags == kResultFlagsCanary);

  if (mode != PlanTestMode::kSanitizer) {
    ResultOwner busy_result;
    CUDA_CHECK(busy_result.bind(batch, ResultLayout::kHost, options.flags));
    /* Static request validation precedes reservation. Even while this request
     * is pending, malformed descriptors retain their own diagnostics instead
     * of being masked by the busy state. A valid WARM reaches request/cache
     * single-flight admission and reports the pending operation. */
    PublicBatch pending_invalid = batch;
    pending_invalid.bind();
    pending_invalid.descriptor.atom_offsets.size_bytes = 0u;
    ResultOwner pending_invalid_result;
    CUDA_CHECK(pending_invalid_result.bind(pending_invalid, ResultLayout::kHost, options.flags));
    CHECK(xtbloom_compute_enqueue(context.get(), &pending_invalid.descriptor, &options,
                                  &pending_invalid_result.descriptor,
                                  request.get()) == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(std::strstr(xtbloom_get_last_error(), "atom_offsets") != nullptr);
    bool pending_invalid_unchanged = false;
    CUDA_CHECK(pending_invalid_result.unchanged(pending_invalid_unchanged));
    CHECK(pending_invalid_unchanged);
    CHECK(xtbloom_compute_enqueue(context.get(), &batch.descriptor, &warm_options,
                                  &busy_result.descriptor,
                                  request.get()) == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(std::strstr(xtbloom_get_last_error(), "pending or submitting") != nullptr);
    bool pending_warm_unchanged = false;
    CUDA_CHECK(busy_result.unchanged(pending_warm_unchanged));
    CHECK(pending_warm_unchanged);
    CHECK(xtbloom_request_query(request.get(), &info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(info.state == XTBLOOM_REQUEST_PENDING);

    xtbloom_request_t* raw_busy_request = nullptr;
    CHECK(xtbloom_request_create(context.get(), &raw_busy_request) == XTBLOOM_STATUS_SUCCESS);
    RequestHandle busy_request(raw_busy_request);
    CHECK(xtbloom_compute_enqueue(context.get(), &batch.descriptor, &options,
                                  &busy_result.descriptor,
                                  busy_request.get()) == XTBLOOM_STATUS_INVALID_ARGUMENT);
    xtbloom_request_info_t busy_info{};
    CHECK(xtbloom_request_info_init(&busy_info, sizeof(busy_info)) == XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom_request_query(busy_request.get(), &busy_info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(busy_info.state == XTBLOOM_REQUEST_IDLE);
    CHECK(xtbloom_compute(context.get(), &batch.descriptor, &options, &busy_result.descriptor) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    bool busy_unchanged = false;
    CUDA_CHECK(busy_result.unchanged(busy_unchanged));
    CHECK(busy_unchanged);
  }

  if (mode != PlanTestMode::kSanitizer) gate.release();
  CHECK(xtbloom_request_wait(request.get(), &info) == XTBLOOM_STATUS_SUCCESS);
  CHECK(info.state == XTBLOOM_REQUEST_COMPLETE);
  CHECK(info.completion_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(info.result_flags == reference.flags);
  MaterializedResult actual;
  CUDA_CHECK(result.materialize(actual));
  actual.flags = info.result_flags;
  CHECK(compare_result(result, actual, reference, options) == 0);
  CHECK(short_result->flags == kResultFlagsCanary);

  /* A successful async FRESH publishes one strict token. Consume it on the
   * blocked owner stream at changed geometry, prove nonblocking admission and
   * request-only flag publication, then compare to an independent FRESH/CPU
   * result. */
  batch.perturb(0.0015);
  CHECK(run_cpu_reference(cpu_context, batch, options, reference) == 0);
  StreamOwner independent_stream;
  CUDA_CHECK(independent_stream.create());
  xtbloom_status_t independent_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle independent_context =
      make_context(XTBLOOM_BACKEND_CUDA, device, independent_stream.get(), independent_status);
  CHECK(independent_status == XTBLOOM_STATUS_SUCCESS);
  ResultOwner independent_fresh_result;
  CUDA_CHECK(independent_fresh_result.bind(batch, ResultLayout::kMixed, options.flags));
  CHECK(xtbloom_compute(independent_context.get(), &batch.descriptor, &options,
                        &independent_fresh_result.descriptor) == XTBLOOM_STATUS_SUCCESS);
  MaterializedResult independent_fresh;
  CUDA_CHECK(independent_fresh_result.materialize(independent_fresh));
  CHECK(compare_result(independent_fresh_result, independent_fresh, reference, options) == 0);
  CUDA_CHECK(warm_result.bind(batch, ResultLayout::kTorchRequest, warm_options.flags));
  submitted_warm_result = warm_result.descriptor;
  BlockingStreamGate warm_gate;
  if (mode != PlanTestMode::kSanitizer) CUDA_CHECK(warm_gate.arm(stream.get()));
  CHECK(xtbloom_compute_enqueue(context.get(), &batch.descriptor, &warm_options,
                                &submitted_warm_result, request.get()) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom_request_query(request.get(), &info) == XTBLOOM_STATUS_SUCCESS);
  if (mode == PlanTestMode::kSanitizer) {
    CHECK(info.state == XTBLOOM_REQUEST_PENDING || info.state == XTBLOOM_REQUEST_COMPLETE);
  } else {
    CHECK(info.state == XTBLOOM_REQUEST_PENDING);
    bool warm_host_unchanged = false;
    CUDA_CHECK(warm_result.torch_host_diagnostics_unchanged(warm_host_unchanged));
    CHECK(warm_host_unchanged);
    warm_gate.release();
  }
  CHECK(submitted_warm_result.flags == kResultFlagsCanary);
  CHECK(xtbloom_request_wait(request.get(), &info) == XTBLOOM_STATUS_SUCCESS);
  CHECK(info.state == XTBLOOM_REQUEST_COMPLETE);
  CHECK(info.completion_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(info.result_flags == reference.flags);
  MaterializedResult warm_actual;
  CUDA_CHECK(warm_result.materialize(warm_actual));
  warm_actual.flags = info.result_flags;
  independent_fresh.flags = info.result_flags;
  CHECK(compare_result(warm_result, warm_actual, independent_fresh, warm_options) == 0);
  for (std::size_t system = 0; system < warm_actual.iterations.size(); ++system) {
    CHECK(warm_actual.iterations[system] <= independent_fresh.iterations[system]);
  }
  CHECK(submitted_warm_result.flags == kResultFlagsCanary);

  /* A successful WARM publishes the next token, so the same request can
   * consume a second changed-geometry checkpoint without a FRESH reseed. */
  batch.perturb(0.0005);
  CHECK(run_cpu_reference(cpu_context, batch, options, reference) == 0);
  CUDA_CHECK(second_warm_result.bind(batch, ResultLayout::kDevice, warm_options.flags));
  CHECK(xtbloom_compute_enqueue(context.get(), &batch.descriptor, &warm_options,
                                &second_warm_result.descriptor,
                                request.get()) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom_request_wait(request.get(), &info) == XTBLOOM_STATUS_SUCCESS);
  CHECK(info.completion_status == XTBLOOM_STATUS_SUCCESS);
  MaterializedResult second_warm_actual;
  CUDA_CHECK(second_warm_result.materialize(second_warm_actual));
  second_warm_actual.flags = info.result_flags;
  CHECK(compare_result(second_warm_result, second_warm_actual, reference, warm_options) == 0);

  /* A host-side call-level validation failure rolls the reusable request back
   * to its preceding COMPLETE snapshot and preserves every caller byte. */
  PublicBatch invalid = batch;
  invalid.bind();
  invalid.descriptor.atom_offsets.size_bytes = 0u;
  ResultOwner invalid_result;
  CUDA_CHECK(invalid_result.bind(invalid, ResultLayout::kHost, options.flags));
  CHECK(xtbloom_compute_enqueue(context.get(), &invalid.descriptor, &options,
                                &invalid_result.descriptor,
                                request.get()) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  xtbloom_request_info_t preserved_info{};
  CHECK(xtbloom_request_info_init(&preserved_info, sizeof(preserved_info)) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom_request_query(request.get(), &preserved_info) == XTBLOOM_STATUS_SUCCESS);
  CHECK(preserved_info.state == XTBLOOM_REQUEST_COMPLETE);
  CHECK(preserved_info.completion_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(preserved_info.result_flags == reference.flags);
  bool invalid_unchanged = false;
  CUDA_CHECK(invalid_result.unchanged(invalid_unchanged));
  CHECK(invalid_unchanged);

  /* A host topology change is synchronously canonicalized and replaces the
   * context convenience runtime before the accepted request is published. */
  PublicBatch changed;
  CHECK(make_fixture_batch(2u, false, changed) == 0);
  MaterializedResult changed_reference;
  CHECK(run_cpu_reference(cpu_context, changed, base_options, changed_reference) == 0);
  ResultOwner changed_result;
  CUDA_CHECK(changed_result.bind(changed, ResultLayout::kMixed, base_options.flags));
  {
    /* Keep the request inside the result owner's lifetime so every early
     * return settles native work before the borrowed output storage dies. */
    xtbloom_request_t* raw_changed_request = nullptr;
    CHECK(xtbloom_request_create(context.get(), &raw_changed_request) == XTBLOOM_STATUS_SUCCESS);
    RequestHandle changed_request(raw_changed_request);
    CHECK(xtbloom_compute_enqueue(context.get(), &changed.descriptor, &base_options,
                                  &changed_result.descriptor,
                                  changed_request.get()) == XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom_request_wait(changed_request.get(), &info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(info.completion_status == XTBLOOM_STATUS_SUCCESS);
  }
  MaterializedResult changed_actual;
  CUDA_CHECK(changed_result.materialize(changed_actual));
  changed_actual.flags = info.result_flags;
  CHECK(compare_result(changed_result, changed_actual, changed_reference, base_options) == 0);

  if (mode != PlanTestMode::kSanitizer) {
    /* Request destruction is an exact completion boundary for a pending WARM
     * submission. Its successful settlement publishes the next checkpoint,
     * which a different reusable request can consume immediately. */
    xtbloom_compute_options_t changed_warm_options = base_options;
    changed_warm_options.scc_start_mode = XTBLOOM_SCC_START_WARM;
    changed.perturb(0.002);
    CHECK(run_cpu_reference(cpu_context, changed, base_options, changed_reference) == 0);
    ResultOwner destroy_result;
    CUDA_CHECK(destroy_result.bind(changed, ResultLayout::kDevice, changed_warm_options.flags));
    xtbloom_request_t* raw_destroy_request = nullptr;
    CHECK(xtbloom_request_create(context.get(), &raw_destroy_request) == XTBLOOM_STATUS_SUCCESS);
    RequestHandle destroy_request(raw_destroy_request);
    CHECK(xtbloom_compute_enqueue(context.get(), &changed.descriptor, &changed_warm_options,
                                  &destroy_result.descriptor,
                                  destroy_request.get()) == XTBLOOM_STATUS_SUCCESS);
    destroy_request.reset();
    MaterializedResult destroy_actual;
    CUDA_CHECK(destroy_result.materialize(destroy_actual));
    destroy_actual.flags = changed_reference.flags;
    CHECK(compare_result(destroy_result, destroy_actual, changed_reference, changed_warm_options) ==
          0);
    CHECK(destroy_result.descriptor.flags == kResultFlagsCanary);

    changed.perturb(0.0005);
    CHECK(run_cpu_reference(cpu_context, changed, base_options, changed_reference) == 0);
    CUDA_CHECK(
        post_destroy_warm_result.bind(changed, ResultLayout::kMixed, changed_warm_options.flags));
    CHECK(xtbloom_compute_enqueue(context.get(), &changed.descriptor, &changed_warm_options,
                                  &post_destroy_warm_result.descriptor,
                                  request.get()) == XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom_request_wait(request.get(), &info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(info.completion_status == XTBLOOM_STATUS_SUCCESS);
    MaterializedResult post_destroy_warm_actual;
    CUDA_CHECK(post_destroy_warm_result.materialize(post_destroy_warm_actual));
    post_destroy_warm_actual.flags = info.result_flags;
    CHECK(compare_result(post_destroy_warm_result, post_destroy_warm_actual, changed_reference,
                         changed_warm_options) == 0);
  }
  return 0;
}

/* Fixed-topology CUDA plans reuse the prepared runtime, expose host/device
 * workspace queries that differ by requested properties, and reject topology
 * mismatches before output mutation (matching the CPU plan contract). */
int test_cuda_plan_api(std::int32_t device, xtbloom_context_t* cpu_context,
                       const xtbloom_compute_options_t& base_options,
                       PlanTestMode mode = PlanTestMode::kFull) {
  g_scenario = "plan-api";
  StreamOwner stream;
  CUDA_CHECK(stream.create());
  xtbloom_status_t context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle context = make_context(XTBLOOM_BACKEND_CUDA, device, stream.get(), context_status);
  CHECK(context_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(context != nullptr);

  xtbloom_compute_options_t options = base_options;

  if (mode == PlanTestMode::kFull) {
    PublicBatch batch;
    CHECK(make_fixture_batch(4u, false, batch) == 0);

    MaterializedResult reference;
    CHECK(run_cpu_reference(cpu_context, batch, options, reference) == 0);

    /* Plan identity is canonicalized by the CUDA owner, so a plan accepts the
     * same all-device topology descriptors as xtbloom_compute. */
    DeviceBatchInputs device_inputs;
    CUDA_CHECK(device_inputs.upload_all(batch));
    bind_inputs(batch, &device_inputs, InputLayout::kDevice);
    CHECK(verify_input_layout(batch, InputLayout::kDevice, false) == 0);

    /* CUDA pointer ownership is checked before topology staging reads a buffer.
     * A device pointer tagged as HOST must fail plan creation transactionally. */
    xtbloom_batch_t mislabeled = batch.descriptor;
    mislabeled.atom_offsets.memory_space = XTBLOOM_MEMORY_HOST;
    xtbloom_plan_t* raw_mislabeled_plan = reinterpret_cast<xtbloom_plan_t*>(UINTPTR_MAX);
    CHECK(xtbloom_plan_create(context.get(), &mislabeled, &options, &raw_mislabeled_plan) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(raw_mislabeled_plan == nullptr);

    xtbloom_plan_t* raw_plan = nullptr;
    CHECK(xtbloom_plan_create(context.get(), &batch.descriptor, &options, &raw_plan) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(raw_plan != nullptr);
    PlanHandle plan(raw_plan);

    /* Host workspace is nonzero and well-aligned; the CUDA runtime reserves
     * device workspace that grows when forces are requested. */
    xtbloom_workspace_query_t query{};
    CHECK(xtbloom_workspace_query_init(&query, sizeof(query)) == XTBLOOM_STATUS_SUCCESS);
    query.compute_flags = options.flags;
    CHECK(xtbloom_plan_query_workspace(plan.get(), &query) == XTBLOOM_STATUS_SUCCESS);
    CHECK(query.host_required_bytes > 0u);
    CHECK(query.host_required_alignment >= 8u);
    CHECK(query.device_required_bytes > 0u);
    CHECK(query.device_required_alignment >= 8u);

    xtbloom_compute_options_t energy_options = options;
    energy_options.flags = XTBLOOM_COMPUTE_ENERGY;
    xtbloom_plan_t* raw_energy_plan = nullptr;
    CHECK(xtbloom_plan_create(context.get(), &batch.descriptor, &energy_options,
                              &raw_energy_plan) == XTBLOOM_STATUS_SUCCESS);
    CHECK(raw_energy_plan != nullptr);
    PlanHandle energy_plan(raw_energy_plan);

    xtbloom_workspace_query_t energy_query{};
    CHECK(xtbloom_workspace_query_init(&energy_query, sizeof(energy_query)) ==
          XTBLOOM_STATUS_SUCCESS);
    energy_query.compute_flags = XTBLOOM_COMPUTE_ENERGY;
    CHECK(xtbloom_plan_query_workspace(energy_plan.get(), &energy_query) == XTBLOOM_STATUS_SUCCESS);
    CHECK(energy_query.host_required_bytes > 0u);
    CHECK(query.device_required_bytes >= energy_query.device_required_bytes);

    /* Point-charge forces need the CUDA force arenas even when QM forces are
     * not requested. Compare two plans over the exact same embedded topology so
     * the property flag is the only source of the workspace-size difference. */
    PublicBatch qmmm_batch;
    CHECK(make_fixture_batch(4u, true, qmmm_batch) == 0);
    xtbloom_compute_options_t qmmm_energy_options = options;
    qmmm_energy_options.flags = XTBLOOM_COMPUTE_ENERGY;
    xtbloom_plan_t* raw_qmmm_energy_plan = nullptr;
    CHECK(xtbloom_plan_create(context.get(), &qmmm_batch.descriptor, &qmmm_energy_options,
                              &raw_qmmm_energy_plan) == XTBLOOM_STATUS_SUCCESS);
    CHECK(raw_qmmm_energy_plan != nullptr);
    PlanHandle qmmm_energy_plan(raw_qmmm_energy_plan);

    xtbloom_workspace_query_t qmmm_energy_query{};
    CHECK(xtbloom_workspace_query_init(&qmmm_energy_query, sizeof(qmmm_energy_query)) ==
          XTBLOOM_STATUS_SUCCESS);
    qmmm_energy_query.compute_flags = qmmm_energy_options.flags;
    CHECK(xtbloom_plan_query_workspace(qmmm_energy_plan.get(), &qmmm_energy_query) ==
          XTBLOOM_STATUS_SUCCESS);

    xtbloom_compute_options_t qmmm_point_force_options = options;
    qmmm_point_force_options.flags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_POINT_CHARGE_FORCES;
    xtbloom_plan_t* raw_qmmm_point_force_plan = nullptr;
    CHECK(xtbloom_plan_create(context.get(), &qmmm_batch.descriptor, &qmmm_point_force_options,
                              &raw_qmmm_point_force_plan) == XTBLOOM_STATUS_SUCCESS);
    CHECK(raw_qmmm_point_force_plan != nullptr);
    PlanHandle qmmm_point_force_plan(raw_qmmm_point_force_plan);

    xtbloom_workspace_query_t qmmm_point_force_query{};
    CHECK(xtbloom_workspace_query_init(&qmmm_point_force_query, sizeof(qmmm_point_force_query)) ==
          XTBLOOM_STATUS_SUCCESS);
    qmmm_point_force_query.compute_flags = qmmm_point_force_options.flags;
    CHECK(xtbloom_plan_query_workspace(qmmm_point_force_plan.get(), &qmmm_point_force_query) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(qmmm_point_force_query.device_required_bytes > qmmm_energy_query.device_required_bytes);

    /* Plan compute equals the CPU reference on the original and a changed
     * geometry, proving fixed-topology reuse. */
    {
      ResultOwner owner;
      CUDA_CHECK(owner.bind(batch, ResultLayout::kHost, options.flags));
      CHECK(xtbloom_plan_compute(plan.get(), &batch.descriptor, &options, &owner.descriptor) ==
            XTBLOOM_STATUS_SUCCESS);
      MaterializedResult actual;
      CUDA_CHECK(owner.materialize(actual));
      CHECK(compare_result(owner, actual, reference, options) == 0);
    }

    batch.perturb(0.003);
    MaterializedResult changed_reference;
    CHECK(run_cpu_reference(cpu_context, batch, options, changed_reference) == 0);
    CUDA_CHECK(device_inputs.upload_numerical(batch));
    bind_inputs(batch, &device_inputs, InputLayout::kDevice);
    {
      ResultOwner owner;
      CUDA_CHECK(owner.bind(batch, ResultLayout::kHost, options.flags));
      CHECK(xtbloom_plan_compute(plan.get(), &batch.descriptor, &options, &owner.descriptor) ==
            XTBLOOM_STATUS_SUCCESS);
      MaterializedResult actual;
      CUDA_CHECK(owner.materialize(actual));
      CHECK(compare_result(owner, actual, changed_reference, options) == 0);
    }

    /* A mislabeled device topology on compute is rejected before output
     * mutation, before host-side identity code could dereference it. */
    {
      ResultOwner owner;
      CUDA_CHECK(owner.bind(batch, ResultLayout::kHost, options.flags));
      xtbloom_batch_t mislabeled_compute = batch.descriptor;
      mislabeled_compute.atomic_numbers.memory_space = XTBLOOM_MEMORY_HOST;
      CHECK(xtbloom_plan_compute(plan.get(), &mislabeled_compute, &options, &owner.descriptor) ==
            XTBLOOM_STATUS_INVALID_ARGUMENT);
      bool unchanged = false;
      CUDA_CHECK(owner.unchanged(unchanged));
      CHECK(unchanged);
    }

    /* A mismatched all-device topology fails before output mutation (corrupted
     * plan) rather than rebuilding the plan-owned prepared runtime. */
    PublicBatch other;
    CHECK(make_fixture_batch(2u, false, other) == 0);
    DeviceBatchInputs other_inputs;
    CUDA_CHECK(other_inputs.upload_all(other));
    bind_inputs(other, &other_inputs, InputLayout::kDevice);
    ResultOwner other_result;
    CUDA_CHECK(other_result.bind(other, ResultLayout::kDevice, options.flags));
    CHECK(xtbloom_plan_compute(plan.get(), &other.descriptor, &options, &other_result.descriptor) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    bool unchanged = false;
    CUDA_CHECK(other_result.unchanged(unchanged));
    CHECK(unchanged);
  }

  if (mode != PlanTestMode::kSanitizer) {
    /* A fixed plan with host topology admits without fencing the owner stream.
     * Numerical HOST leaves are snapshotted, the device-gated commit is queued
     * before enqueue returns, and the request-owned cache survives plan destroy. */
    PublicBatch async_batch;
    CHECK(make_fixture_batch(2u, false, async_batch) == 0);
    bind_inputs(async_batch, nullptr, InputLayout::kHost);
    MaterializedResult async_reference;
    CHECK(run_cpu_reference(cpu_context, async_batch, options, async_reference) == 0);
    xtbloom_plan_t* raw_async_plan = nullptr;
    CHECK(xtbloom_plan_create(context.get(), &async_batch.descriptor, &options, &raw_async_plan) ==
          XTBLOOM_STATUS_SUCCESS);
    PlanHandle async_plan(raw_async_plan);

    /* Caller buffers must outlive every accepted request even when a later test
     * assertion returns early. Declare the result owner before request handles
     * so reverse-order cleanup settles native work first. */
    ResultOwner async_result;
    CUDA_CHECK(async_result.bind(async_batch, ResultLayout::kDevice, options.flags));

    xtbloom_request_t* raw_request = nullptr;
    CHECK(xtbloom_request_create(context.get(), &raw_request) == XTBLOOM_STATUS_SUCCESS);
    RequestHandle request(raw_request);
    xtbloom_request_t* raw_busy_request = nullptr;
    CHECK(xtbloom_request_create(context.get(), &raw_busy_request) == XTBLOOM_STATUS_SUCCESS);
    RequestHandle busy_request(raw_busy_request);

    xtbloom_batch_result_t enqueue_result = async_result.descriptor;

    /* Plan creation prepares topology but does not publish an electronic
     * checkpoint. A first strict WARM is rejected transactionally. */
    xtbloom_compute_options_t warm_options = options;
    warm_options.scc_start_mode = XTBLOOM_SCC_START_WARM;
    CHECK(xtbloom_plan_compute_enqueue(async_plan.get(), &async_batch.descriptor, &warm_options,
                                       &enqueue_result,
                                       request.get()) == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(std::strstr(xtbloom_get_last_error(), "preceding successful public checkpoint") !=
          nullptr);
    xtbloom_request_info_t info{};
    CHECK(xtbloom_request_info_init(&info, sizeof(info)) == XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom_request_query(request.get(), &info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(info.state == XTBLOOM_REQUEST_IDLE);
    CHECK(enqueue_result.flags == kResultFlagsCanary);

    BlockingStreamGate gate;
    CUDA_CHECK(gate.arm(stream.get()));

    /* Exact ABI-v1 storage catches accidental reads of the optional v2
     * suffixes after the public structural validator has accepted them. */
    alignas(xtbloom_compute_options_t) std::array<unsigned char, XTBLOOM_COMPUTE_OPTIONS_V1_SIZE>
        short_options_storage{};
    std::memcpy(short_options_storage.data(), &options, short_options_storage.size());
    auto* const short_options =
        reinterpret_cast<xtbloom_compute_options_t*>(short_options_storage.data());
    short_options->struct_size = XTBLOOM_COMPUTE_OPTIONS_V1_SIZE;
    alignas(xtbloom_batch_result_t) std::array<unsigned char, XTBLOOM_BATCH_RESULT_V1_SIZE>
        short_result_storage{};
    std::memcpy(short_result_storage.data(), &enqueue_result, short_result_storage.size());
    auto* const short_result =
        reinterpret_cast<xtbloom_batch_result_t*>(short_result_storage.data());
    short_result->struct_size = XTBLOOM_BATCH_RESULT_V1_SIZE;
    xtbloom_batch_t enqueue_batch = async_batch.descriptor;
    CHECK(xtbloom_plan_compute_enqueue(async_plan.get(), &enqueue_batch, short_options,
                                       short_result, request.get()) == XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom_request_query(request.get(), &info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(info.state == XTBLOOM_REQUEST_PENDING);
    CHECK(short_result->flags == kResultFlagsCanary);

    /* Cache and request single-flight gates precede topology staging. Neither a
     * second request nor the synchronous API may wait on the blocked stream. */
    CHECK(xtbloom_plan_compute_enqueue(async_plan.get(), &async_batch.descriptor, &options,
                                       &enqueue_result,
                                       busy_request.get()) == XTBLOOM_STATUS_INVALID_ARGUMENT);
    xtbloom_request_info_t busy_info{};
    CHECK(xtbloom_request_info_init(&busy_info, sizeof(busy_info)) == XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom_request_query(busy_request.get(), &busy_info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(busy_info.state == XTBLOOM_REQUEST_IDLE);
    ResultOwner busy_result;
    CUDA_CHECK(busy_result.bind(async_batch, ResultLayout::kHost, options.flags));
    CHECK(xtbloom_plan_compute(async_plan.get(), &async_batch.descriptor, &options,
                               &busy_result.descriptor) == XTBLOOM_STATUS_INVALID_ARGUMENT);
    bool busy_unchanged = false;
    CUDA_CHECK(busy_result.unchanged(busy_unchanged));
    CHECK(busy_unchanged);

    /* Descriptor structs may be reused immediately; host numerical bytes may
     * also be released or changed because enqueue copied them before return. */
    enqueue_batch = {};
    short_options->flags = 0u;
    short_result->energies.data = nullptr;
    async_batch.perturb(0.125);
    async_plan.reset();

    gate.release();
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    MaterializedResult async_actual;
    CUDA_CHECK(async_result.materialize(async_actual));
    CHECK(async_actual.flags == kResultFlagsCanary);
    async_actual.flags = async_reference.flags;
    CHECK(compare_result(async_result, async_actual, async_reference, options) == 0);
    CHECK(xtbloom_request_wait(request.get(), &info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(info.state == XTBLOOM_REQUEST_COMPLETE);
    CHECK(info.completion_status == XTBLOOM_STATUS_SUCCESS);
    CHECK(info.result_flags == 0u);
    CHECK(std::string(xtbloom_request_get_error(request.get())).empty());
    CHECK(short_result->flags == kResultFlagsCanary);
  }

  if (mode != PlanTestMode::kSanitizer) {
    PublicBatch warm_batch;
    CHECK(make_fixture_batch(4u, true, warm_batch) == 0);
    warm_batch.spin_channels.back() = 2;
    warm_batch.unpaired_electrons.back() = 2;
    warm_batch.bind();
    xtbloom_compute_options_t warm_fresh_options = options;
    warm_fresh_options.flags |= XTBLOOM_COMPUTE_POINT_CHARGE_FORCES;
    xtbloom_compute_options_t warm_options = warm_fresh_options;
    warm_options.scc_start_mode = XTBLOOM_SCC_START_WARM;
    MaterializedResult fresh_reference;
    CHECK(run_cpu_reference(cpu_context, warm_batch, warm_fresh_options, fresh_reference) == 0);

    xtbloom_plan_t* raw_warm_plan = nullptr;
    CHECK(xtbloom_plan_create(context.get(), &warm_batch.descriptor, &warm_fresh_options,
                              &raw_warm_plan) == XTBLOOM_STATUS_SUCCESS);
    PlanHandle warm_plan(raw_warm_plan);
    /* Keep every borrowed result owner alive until after the reusable request
     * has settled any pending submission during unwinding. */
    ResultOwner first_warm_result;
    ResultOwner seed_result;
    ResultOwner changed_fresh_result;
    ResultOwner changed_warm_result;
    ResultOwner next_warm_result;
    xtbloom_request_t* raw_warm_request = nullptr;
    CHECK(xtbloom_request_create(context.get(), &raw_warm_request) == XTBLOOM_STATUS_SUCCESS);
    RequestHandle warm_request(raw_warm_request);
    xtbloom_request_info_t warm_info{};
    CHECK(xtbloom_request_info_init(&warm_info, sizeof(warm_info)) == XTBLOOM_STATUS_SUCCESS);

    CUDA_CHECK(first_warm_result.bind(warm_batch, ResultLayout::kHost, warm_options.flags));
    CHECK(xtbloom_plan_compute_enqueue(warm_plan.get(), &warm_batch.descriptor, &warm_options,
                                       &first_warm_result.descriptor,
                                       warm_request.get()) == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(std::strstr(xtbloom_get_last_error(), "preceding successful public checkpoint") !=
          nullptr);
    CHECK(xtbloom_request_query(warm_request.get(), &warm_info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(warm_info.state == XTBLOOM_REQUEST_IDLE);
    bool first_warm_unchanged = false;
    CUDA_CHECK(first_warm_result.unchanged(first_warm_unchanged));
    CHECK(first_warm_unchanged);

    CUDA_CHECK(seed_result.bind(warm_batch, ResultLayout::kMixed, warm_fresh_options.flags));
    CHECK(xtbloom_plan_compute_enqueue(warm_plan.get(), &warm_batch.descriptor, &warm_fresh_options,
                                       &seed_result.descriptor,
                                       warm_request.get()) == XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom_request_wait(warm_request.get(), &warm_info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(warm_info.completion_status == XTBLOOM_STATUS_SUCCESS);
    MaterializedResult seed_actual;
    CUDA_CHECK(seed_result.materialize(seed_actual));
    seed_actual.flags = warm_info.result_flags;
    CHECK(compare_result(seed_result, seed_actual, fresh_reference, warm_fresh_options) == 0);

    warm_batch.perturb(0.00175);
    MaterializedResult changed_reference;
    CHECK(run_cpu_reference(cpu_context, warm_batch, warm_fresh_options, changed_reference) == 0);
    CUDA_CHECK(
        changed_fresh_result.bind(warm_batch, ResultLayout::kMixed, warm_fresh_options.flags));
    CHECK(xtbloom_compute(context.get(), &warm_batch.descriptor, &warm_fresh_options,
                          &changed_fresh_result.descriptor) == XTBLOOM_STATUS_SUCCESS);
    MaterializedResult changed_fresh_actual;
    CUDA_CHECK(changed_fresh_result.materialize(changed_fresh_actual));
    CHECK(compare_result(changed_fresh_result, changed_fresh_actual, changed_reference,
                         warm_fresh_options) == 0);
    CUDA_CHECK(
        changed_warm_result.bind(warm_batch, ResultLayout::kTorchRequest, warm_options.flags));
    CHECK(xtbloom_plan_compute_enqueue(warm_plan.get(), &warm_batch.descriptor, &warm_options,
                                       &changed_warm_result.descriptor,
                                       warm_request.get()) == XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom_request_wait(warm_request.get(), &warm_info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(warm_info.completion_status == XTBLOOM_STATUS_SUCCESS);
    MaterializedResult changed_warm_actual;
    CUDA_CHECK(changed_warm_result.materialize(changed_warm_actual));
    changed_warm_actual.flags = warm_info.result_flags;
    CHECK(compare_result(changed_warm_result, changed_warm_actual, changed_reference,
                         warm_options) == 0);
    for (std::size_t system = 0; system < changed_warm_actual.iterations.size(); ++system) {
      CHECK(changed_warm_actual.iterations[system] <= changed_fresh_actual.iterations[system]);
    }

    /* The successful WARM result publishes the next one-shot token. */
    warm_batch.perturb(0.00025);
    CHECK(run_cpu_reference(cpu_context, warm_batch, warm_fresh_options, changed_reference) == 0);
    CUDA_CHECK(next_warm_result.bind(warm_batch, ResultLayout::kDevice, warm_options.flags));
    CHECK(xtbloom_plan_compute_enqueue(warm_plan.get(), &warm_batch.descriptor, &warm_options,
                                       &next_warm_result.descriptor,
                                       warm_request.get()) == XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom_request_wait(warm_request.get(), &warm_info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(warm_info.completion_status == XTBLOOM_STATUS_SUCCESS);
    MaterializedResult next_warm_actual;
    CUDA_CHECK(next_warm_result.materialize(next_warm_actual));
    next_warm_actual.flags = warm_info.result_flags;
    CHECK(compare_result(next_warm_result, next_warm_actual, changed_reference, warm_options) == 0);
  }

  /* Device and mixed fixed topology is validated by a comparison kernel on
   * the owner stream. The ordering modes require enqueue to return while
   * preceding work is blocked; sanitizer mode runs the same production path
   * without a gate so instrumentation cost stays bounded. */
  for (const auto [layout, name] :
       {std::pair{InputLayout::kDevice, "device"}, std::pair{InputLayout::kMixed, "mixed"}}) {
    if (mode == PlanTestMode::kSanitizer && layout != InputLayout::kMixed) continue;
    std::string scenario = mode == PlanTestMode::kSanitizer ? "plan-async-sanitizer-topology/"
                                                            : "plan-async-blocked-topology/";
    scenario += name;
    g_scenario = scenario.c_str();

    PublicBatch device_async_batch;
    /* Default CTest retains the established batch-8 matrix. The focused
     * sanitizer/profile entry uses one four-system H2/He/LiH/CH2 cycle so it
     * still covers ragged restricted/unrestricted QM/MM and periodic work
     * without multiplying instruction-level sanitizer time unnecessarily. */
    const std::size_t request_batch_size = mode == PlanTestMode::kFull ? 8u : 4u;
    CHECK(make_fixture_batch(request_batch_size, true, device_async_batch) == 0);
    CHECK(device_async_batch.spin_channels.size() == request_batch_size);
    /* Exercise both explicit ABI-v2 modes in one fixed topology. CH2 has an
     * even electron count, so two unpaired electrons form a valid triplet. */
    device_async_batch.spin_channels.back() = 2;
    device_async_batch.unpaired_electrons.back() = 2;
    device_async_batch.bind();
    xtbloom_compute_options_t device_async_options = options;
    device_async_options.flags |= XTBLOOM_COMPUTE_POINT_CHARGE_FORCES;
    MaterializedResult device_async_reference;
    CHECK(run_cpu_reference(cpu_context, device_async_batch, device_async_options,
                            device_async_reference) == 0);
    DeviceBatchInputs device_async_inputs;
    CUDA_CHECK(device_async_inputs.upload_all(device_async_batch));
    bind_inputs(device_async_batch, &device_async_inputs, layout);

    xtbloom_plan_t* raw_device_async_plan = nullptr;
    CHECK(xtbloom_plan_create(context.get(), &device_async_batch.descriptor, &device_async_options,
                              &raw_device_async_plan) == XTBLOOM_STATUS_SUCCESS);
    PlanHandle device_async_plan(raw_device_async_plan);
    const ResultLayout request_result_layout =
        layout == InputLayout::kMixed ? ResultLayout::kTorchRequest : ResultLayout::kDevice;
    ResultOwner device_async_result;
    CUDA_CHECK(device_async_result.bind(device_async_batch, request_result_layout,
                                        device_async_options.flags));
    xtbloom_request_t* raw_device_async_request = nullptr;
    CHECK(xtbloom_request_create(context.get(), &raw_device_async_request) ==
          XTBLOOM_STATUS_SUCCESS);
    RequestHandle device_async_request(raw_device_async_request);

    xtbloom_status_t enqueue_status = XTBLOOM_STATUS_INTERNAL_ERROR;
    BlockingStreamGate device_gate;
    if (mode == PlanTestMode::kSanitizer) {
      enqueue_status = xtbloom_plan_compute_enqueue(
          device_async_plan.get(), &device_async_batch.descriptor, &device_async_options,
          &device_async_result.descriptor, device_async_request.get());
    } else {
      CUDA_CHECK(device_gate.arm(stream.get()));
      bool enqueue_returned = false;
      std::mutex enqueue_mutex;
      std::condition_variable enqueue_changed;
      std::thread submitter([&] {
        enqueue_status = xtbloom_plan_compute_enqueue(
            device_async_plan.get(), &device_async_batch.descriptor, &device_async_options,
            &device_async_result.descriptor, device_async_request.get());
        {
          std::lock_guard<std::mutex> lock(enqueue_mutex);
          enqueue_returned = true;
        }
        enqueue_changed.notify_one();
      });
      bool returned_while_blocked = false;
      {
        std::unique_lock<std::mutex> lock(enqueue_mutex);
        returned_while_blocked = enqueue_changed.wait_for(lock, kBlockedEnqueueWatchdog,
                                                          [&] { return enqueue_returned; });
      }
      if (!returned_while_blocked) device_gate.release();
      submitter.join();
      CHECK(returned_while_blocked);
    }
    CHECK(enqueue_status == XTBLOOM_STATUS_SUCCESS);

    xtbloom_request_info_t device_info{};
    CHECK(xtbloom_request_info_init(&device_info, sizeof(device_info)) == XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom_request_query(device_async_request.get(), &device_info) ==
          XTBLOOM_STATUS_SUCCESS);
    if (mode == PlanTestMode::kSanitizer) {
      CHECK(device_info.state == XTBLOOM_REQUEST_PENDING ||
            device_info.state == XTBLOOM_REQUEST_COMPLETE);
    } else {
      CHECK(device_info.state == XTBLOOM_REQUEST_PENDING);
    }
    /* Do not synchronously copy device sentinels while the owner stream is
     * deliberately blocked: CUDA may serialize that D2H read with the pending
     * stream. The copied descriptor itself must remain untouched, and final
     * correctness below proves publication happens only after release. */
    CHECK(device_async_result.descriptor.flags == kResultFlagsCanary);
    if (request_result_layout == ResultLayout::kTorchRequest &&
        device_info.state == XTBLOOM_REQUEST_PENDING) {
      bool host_diagnostics_unchanged = false;
      CUDA_CHECK(device_async_result.torch_host_diagnostics_unchanged(host_diagnostics_unchanged));
      CHECK(host_diagnostics_unchanged);
    }

    if (mode != PlanTestMode::kSanitizer) device_gate.release();
    CHECK(xtbloom_request_wait(device_async_request.get(), &device_info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(device_info.state == XTBLOOM_REQUEST_COMPLETE);
    CHECK(device_info.completion_status == XTBLOOM_STATUS_SUCCESS);
    CHECK(device_info.result_flags == device_async_reference.flags);
    MaterializedResult device_async_actual;
    CUDA_CHECK(device_async_result.materialize(device_async_actual));
    device_async_actual.flags = device_info.result_flags;
    CHECK(compare_result(device_async_result, device_async_actual, device_async_reference,
                         device_async_options) == 0);

    /* Every descriptor-placement coordinate also traverses strict WARM. This
     * includes the focused mixed sanitizer path and leaves a replacement token
     * for the steady-state WARM profiler below. */
    xtbloom_compute_options_t device_warm_options = device_async_options;
    device_warm_options.scc_start_mode = XTBLOOM_SCC_START_WARM;
    CHECK(xtbloom_plan_compute_enqueue(device_async_plan.get(), &device_async_batch.descriptor,
                                       &device_warm_options, &device_async_result.descriptor,
                                       device_async_request.get()) == XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom_request_wait(device_async_request.get(), &device_info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(device_info.completion_status == XTBLOOM_STATUS_SUCCESS);
    MaterializedResult device_warm_actual;
    CUDA_CHECK(device_async_result.materialize(device_warm_actual));
    device_warm_actual.flags = device_info.result_flags;
    CHECK(compare_result(device_async_result, device_warm_actual, device_async_reference,
                         device_warm_options) == 0);
    for (std::size_t system = 0; system < device_warm_actual.iterations.size(); ++system) {
      CHECK(device_warm_actual.iterations[system] <= device_async_actual.iterations[system]);
    }

    if (mode == PlanTestMode::kProfileSteadyState) {
      constexpr int kProfileIterations = 10;
      xtbloom_request_info_t profile_info{};
      CHECK(xtbloom_request_info_init(&profile_info, sizeof(profile_info)) ==
            XTBLOOM_STATUS_SUCCESS);
      CUDA_CHECK(cudaProfilerStart());
      for (int iteration = 0; iteration < kProfileIterations; ++iteration) {
        CHECK(xtbloom_plan_compute_enqueue(device_async_plan.get(), &device_async_batch.descriptor,
                                           &device_warm_options, &device_async_result.descriptor,
                                           device_async_request.get()) == XTBLOOM_STATUS_SUCCESS);
        CHECK(xtbloom_request_wait(device_async_request.get(), &profile_info) ==
              XTBLOOM_STATUS_SUCCESS);
        CHECK(profile_info.state == XTBLOOM_REQUEST_COMPLETE);
        CHECK(profile_info.completion_status == XTBLOOM_STATUS_SUCCESS);
        CHECK(profile_info.result_flags == device_async_reference.flags);
      }
      CUDA_CHECK(cudaProfilerStop());
      MaterializedResult profile_actual;
      CUDA_CHECK(device_async_result.materialize(profile_actual));
      profile_actual.flags = profile_info.result_flags;
      CHECK(compare_result(device_async_result, profile_actual, device_async_reference,
                           device_warm_options) == 0);
      CHECK(device_async_result.descriptor.flags == kResultFlagsCanary);
      std::printf("request_profile_iterations=%d\n", kProfileIterations);
      g_scenario = "plan-api-profile";
      return 0;
    }

    /* A topology mutation ordered before the comparison is a deferred
     * INVALID_ARGUMENT, not a host admission wait. The device commit remains
     * gated, so every caller output keeps its sentinel bytes. */
    ResultOwner mismatch_result;
    CUDA_CHECK(mismatch_result.bind(device_async_batch, request_result_layout,
                                    device_async_options.flags));
    xtbloom_request_t* raw_mismatch_request = nullptr;
    CHECK(xtbloom_request_create(context.get(), &raw_mismatch_request) == XTBLOOM_STATUS_SUCCESS);
    RequestHandle mismatch_request(raw_mismatch_request);
    BlockingStreamGate mismatch_gate;
    if (mode != PlanTestMode::kSanitizer) CUDA_CHECK(mismatch_gate.arm(stream.get()));
    /* spin_channels is device-backed in both layouts. A device-side memset is
     * guaranteed to enqueue behind the gate; a pageable host-to-device copy
     * is allowed to block in the CUDA runtime before the request is submitted. */
    CUDA_CHECK(cudaMemsetAsync(const_cast<void*>(device_async_batch.descriptor.spin_channels.data),
                               0, device_async_batch.descriptor.spin_channels.size_bytes,
                               stream.get()));
    xtbloom_status_t mismatch_enqueue_status = XTBLOOM_STATUS_INTERNAL_ERROR;
    if (mode == PlanTestMode::kSanitizer) {
      mismatch_enqueue_status = xtbloom_plan_compute_enqueue(
          device_async_plan.get(), &device_async_batch.descriptor, &device_warm_options,
          &mismatch_result.descriptor, mismatch_request.get());
    } else {
      bool mismatch_enqueue_returned = false;
      std::mutex mismatch_enqueue_mutex;
      std::condition_variable mismatch_enqueue_changed;
      std::thread mismatch_submitter([&] {
        mismatch_enqueue_status = xtbloom_plan_compute_enqueue(
            device_async_plan.get(), &device_async_batch.descriptor, &device_warm_options,
            &mismatch_result.descriptor, mismatch_request.get());
        {
          std::lock_guard<std::mutex> lock(mismatch_enqueue_mutex);
          mismatch_enqueue_returned = true;
        }
        mismatch_enqueue_changed.notify_one();
      });
      bool mismatch_returned_while_blocked = false;
      {
        std::unique_lock<std::mutex> lock(mismatch_enqueue_mutex);
        mismatch_returned_while_blocked = mismatch_enqueue_changed.wait_for(
            lock, kBlockedEnqueueWatchdog, [&] { return mismatch_enqueue_returned; });
      }
      if (!mismatch_returned_while_blocked) mismatch_gate.release();
      mismatch_submitter.join();
      CHECK(mismatch_returned_while_blocked);
    }
    CHECK(mismatch_enqueue_status == XTBLOOM_STATUS_SUCCESS);
    xtbloom_request_info_t mismatch_info{};
    CHECK(xtbloom_request_info_init(&mismatch_info, sizeof(mismatch_info)) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom_request_query(mismatch_request.get(), &mismatch_info) == XTBLOOM_STATUS_SUCCESS);
    if (mode == PlanTestMode::kSanitizer) {
      CHECK(mismatch_info.state == XTBLOOM_REQUEST_PENDING ||
            mismatch_info.state == XTBLOOM_REQUEST_COMPLETE);
    } else {
      CHECK(mismatch_info.state == XTBLOOM_REQUEST_PENDING);
      mismatch_gate.release();
    }
    CHECK(xtbloom_request_wait(mismatch_request.get(), &mismatch_info) == XTBLOOM_STATUS_SUCCESS);
    CHECK(mismatch_info.state == XTBLOOM_REQUEST_COMPLETE);
    CHECK(mismatch_info.completion_status == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(std::strstr(xtbloom_request_get_error(mismatch_request.get()),
                      "does not match the fixed CUDA plan topology") != nullptr);
    bool mismatch_unchanged = false;
    CUDA_CHECK(mismatch_result.unchanged(mismatch_unchanged));
    CHECK(mismatch_unchanged);

    /* Admission of this strict-WARM request consumed the preceding successful
     * checkpoint before its deferred topology failure became visible. Restore
     * the borrowed topology bytes, then prove another WARM rejects instead of
     * silently consuming stale electronic state from the earlier success. */
    CUDA_CHECK(device_async_inputs.spin_channels_.upload(device_async_batch.spin_channels));
    ResultOwner stale_warm_result;
    CUDA_CHECK(stale_warm_result.bind(device_async_batch, ResultLayout::kDevice,
                                      device_warm_options.flags));
    CHECK(xtbloom_plan_compute(device_async_plan.get(), &device_async_batch.descriptor,
                               &device_warm_options,
                               &stale_warm_result.descriptor) == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(std::strstr(xtbloom_get_last_error(), "preceding successful public checkpoint") !=
          nullptr);
    bool stale_warm_unchanged = false;
    CUDA_CHECK(stale_warm_result.unchanged(stale_warm_unchanged));
    CHECK(stale_warm_unchanged);
  }

  g_scenario = "plan-api";
  return 0;
}

int verify_input_layout(const PublicBatch& batch, InputLayout layout, bool qmmm) {
  if (layout == InputLayout::kDevice) {
    CHECK(is_cuda_buffer(batch.descriptor.atom_offsets));
    CHECK(is_cuda_buffer(batch.descriptor.atomic_numbers));
    CHECK(is_cuda_buffer(batch.descriptor.molecular_charges));
    CHECK(is_cuda_buffer(batch.descriptor.unpaired_electrons));
    CHECK(is_cuda_buffer(batch.descriptor.spin_channels));
    CHECK(is_cuda_buffer(batch.descriptor.positions));
    if (qmmm) {
      /* The seven topology fields include spin, point, and response partitions. */
      CHECK(is_cuda_buffer(batch.descriptor.point_charge_offsets));
      CHECK(is_cuda_buffer(batch.descriptor.charge_response_offsets));
      CHECK(is_cuda_buffer(batch.descriptor.point_charge_positions));
      CHECK(is_cuda_buffer(batch.descriptor.point_charge_values));
      CHECK(is_cuda_buffer(batch.descriptor.point_charge_gammas));
      CHECK(is_cuda_buffer(batch.descriptor.atomic_potential_shifts));
      CHECK(is_cuda_buffer(batch.descriptor.charge_response_matrix));
    }
    return 0;
  }

  CHECK(layout == InputLayout::kMixed);
  CHECK(is_cuda_buffer(batch.descriptor.atom_offsets));
  CHECK(is_host_buffer(batch.descriptor.atomic_numbers));
  CHECK(is_cuda_buffer(batch.descriptor.molecular_charges));
  CHECK(is_cuda_buffer(batch.descriptor.unpaired_electrons));
  CHECK(is_cuda_buffer(batch.descriptor.spin_channels));
  CHECK(is_cuda_buffer(batch.descriptor.positions));
  if (qmmm) {
    CHECK(is_cuda_buffer(batch.descriptor.point_charge_offsets));
    CHECK(is_host_buffer(batch.descriptor.charge_response_offsets));
    CHECK(is_host_buffer(batch.descriptor.point_charge_positions));
    CHECK(is_cuda_buffer(batch.descriptor.point_charge_values));
    CHECK(is_host_buffer(batch.descriptor.point_charge_gammas));
    CHECK(is_cuda_buffer(batch.descriptor.atomic_potential_shifts));
    CHECK(is_host_buffer(batch.descriptor.charge_response_matrix));
  }
  return 0;
}

/* Exercise public CUDA inference with genuine direct-device inputs across the
 * batch sizes where throughput acceleration matters. Each topology is first
 * checked against the CPU C API, then submitted as all-device and mixed
 * descriptors. A final numerical-only upload reuses every device address and
 * proves the fixed-topology direct path on a changed request. */
int test_input_descriptor_matrix(std::int32_t device, xtbloom_context_t* cpu_context,
                                 const xtbloom_compute_options_t& base_options) {
  StreamOwner stream;
  CUDA_CHECK(stream.create());
  xtbloom_status_t context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle context = make_context(XTBLOOM_BACKEND_CUDA, device, stream.get(), context_status);
  CHECK(context_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(context != nullptr);

  std::string scenario;
  for (bool qmmm : {false, true}) {
    xtbloom_compute_options_t options = base_options;
    if (qmmm) options.flags |= XTBLOOM_COMPUTE_POINT_CHARGE_FORCES;
    for (std::size_t batch_size : {1u, 8u, 32u, 128u}) {
      scenario = std::string(qmmm ? "qmmm" : "base") + "/batch-" + std::to_string(batch_size);
      g_scenario = scenario.c_str();
      PublicBatch batch;
      CHECK(make_fixture_batch(batch_size, qmmm, batch) == 0);

      MaterializedResult reference;
      CHECK(run_cpu_reference(cpu_context, batch, options, reference) == 0);

      DeviceBatchInputs device_inputs;
      CUDA_CHECK(device_inputs.upload_all(batch));
      const auto stable_addresses = device_inputs.identity();

      scenario += "/all-device";
      g_scenario = scenario.c_str();
      bind_inputs(batch, &device_inputs, InputLayout::kDevice);
      CHECK(verify_input_layout(batch, InputLayout::kDevice, qmmm) == 0);
      CHECK(execute_cuda_and_compare(context.get(), batch, options, ResultLayout::kDevice,
                                     reference) == 0);

      scenario.resize(scenario.rfind('/'));
      scenario += "/mixed";
      g_scenario = scenario.c_str();
      bind_inputs(batch, &device_inputs, InputLayout::kMixed);
      CHECK(verify_input_layout(batch, InputLayout::kMixed, qmmm) == 0);
      CHECK(execute_cuda_and_compare(context.get(), batch, options, ResultLayout::kMixed,
                                     reference) == 0);

      scenario.resize(scenario.rfind('/'));
      scenario += "/fixed-topology-repeat";
      g_scenario = scenario.c_str();
      batch.perturb(0.002 + 0.00001 * static_cast<double>(batch_size));
      MaterializedResult changed_reference;
      CHECK(run_cpu_reference(cpu_context, batch, options, changed_reference) == 0);
      CUDA_CHECK(device_inputs.upload_numerical(batch));
      CHECK(device_inputs.identity() == stable_addresses);
      bind_inputs(batch, &device_inputs, InputLayout::kDevice);
      CHECK(verify_input_layout(batch, InputLayout::kDevice, qmmm) == 0);
      CHECK(execute_cuda_and_compare(context.get(), batch, options, ResultLayout::kDevice,
                                     changed_reference) == 0);
    }
  }
  return 0;
}

int test_current_device_restored(int device_count, std::int32_t context_device, PublicBatch& batch,
                                 const xtbloom_compute_options_t& options,
                                 const MaterializedResult& reference) {
  if (device_count < 2) {
    std::puts("cuda_public_api_test: SKIP current-device restoration (requires two devices)");
    return 0;
  }
  g_scenario = "current-device-restoration";
  CurrentDeviceRestore restore;
  CUDA_CHECK(cudaSetDevice(context_device));
  StreamOwner stream;
  CUDA_CHECK(stream.create());
  xtbloom_status_t context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle context =
      make_context(XTBLOOM_BACKEND_CUDA, context_device, stream.get(), context_status);
  CHECK(context_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(context != nullptr);

  const int caller_device = context_device == 0 ? 1 : 0;
  CUDA_CHECK(cudaSetDevice(caller_device));
  int before = -1;
  CUDA_CHECK(cudaGetDevice(&before));
  CHECK(before == caller_device);

  ResultOwner result;
  CUDA_CHECK(result.bind(batch, ResultLayout::kHost));
  CHECK(xtbloom_compute(context.get(), &batch.descriptor, &options, &result.descriptor) ==
        XTBLOOM_STATUS_SUCCESS);
  int after = -1;
  CUDA_CHECK(cudaGetDevice(&after));
  CHECK(after == before);
  MaterializedResult actual;
  CUDA_CHECK(result.materialize(actual));
  CHECK(compare_result(result, actual, reference, options) == 0);

  /* Destroy on its own device so this test isolates compute-call restoration. */
  CUDA_CHECK(cudaSetDevice(context_device));
  context.reset();
  return 0;
}

/* Wrong-device determinism at the public ABI: a live CUDA allocation that
 * belongs to a different physical device than the context must be rejected
 * with XTBLOOM_STATUS_INVALID_ARGUMENT before any caller output byte is
 * touched, and the caller's thread-local current device must be restored.
 * Runs only when at least two GPUs are visible; on a single-GPU host the
 * mislabeled-space half of the contract (host-tagged device pointer and
 * device-tagged host pointer) is exercised by test_public_mislabeled_rejection
 * below, which shares this scenario code path through the same validator. */
int test_public_wrong_device_rejection(int device_count, std::int32_t context_device,
                                       PublicBatch& batch,
                                       const xtbloom_compute_options_t& options) {
  if (device_count < 2) {
    std::puts("cuda_public_api_test: SKIP wrong-device rejection (requires two devices)");
    return 0;
  }
  g_scenario = "wrong-device-rejection";
  CurrentDeviceRestore restore;
  CUDA_CHECK(cudaSetDevice(context_device));
  StreamOwner stream;
  CUDA_CHECK(stream.create());
  xtbloom_status_t context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle context =
      make_context(XTBLOOM_BACKEND_CUDA, context_device, stream.get(), context_status);
  CHECK(context_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(context != nullptr);

  const int foreign_device = context_device == 0 ? 1 : 0;
  /* Allocate the positions buffer on the foreign device so
   * cudaPointerGetAttributes reports a genuine cross-device ownership
   * mismatch instead of an ordinary stale or mislabeled address. */
  CUDA_CHECK(cudaSetDevice(foreign_device));
  double* foreign_positions = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&foreign_positions),
                        batch.positions.size() * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(foreign_positions, batch.positions.data(),
                        batch.positions.size() * sizeof(double), cudaMemcpyHostToDevice));

  /* The caller's current device differs from both the context device and the
   * foreign allocation device on one side, exercising isolation on the way in
   * and out. */
  CUDA_CHECK(cudaSetDevice(foreign_device));
  g_scenario = "wrong-device-rejection/xtbloom_compute";
  {
    PublicBatch wrong = batch;
    wrong.bind();
    wrong.descriptor.positions = {foreign_positions, wrong.positions.size() * sizeof(double),
                                  XTBLOOM_MEMORY_CUDA_DEVICE, 0u};
    ResultOwner result;
    CUDA_CHECK(result.bind(wrong, ResultLayout::kHost));
    CHECK(xtbloom_compute(context.get(), &wrong.descriptor, &options, &result.descriptor) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    bool unchanged = false;
    CUDA_CHECK(result.unchanged(unchanged));
    CHECK(unchanged);
    bool guards = false;
    CUDA_CHECK(result.guards_intact(guards));
    CHECK(guards);
    int after = -1;
    CUDA_CHECK(cudaGetDevice(&after));
    CHECK(after == foreign_device);
  }

  g_scenario = "wrong-device-rejection/plan-create";
  {
    PublicBatch wrong = batch;
    wrong.bind();
    wrong.descriptor.positions = {foreign_positions, wrong.positions.size() * sizeof(double),
                                  XTBLOOM_MEMORY_CUDA_DEVICE, 0u};
    xtbloom_plan_t* raw_plan = reinterpret_cast<xtbloom_plan_t*>(UINTPTR_MAX);
    CUDA_CHECK(cudaSetDevice(foreign_device));
    CHECK(xtbloom_plan_create(context.get(), &wrong.descriptor, &options, &raw_plan) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(raw_plan == nullptr);
    int after = -1;
    CUDA_CHECK(cudaGetDevice(&after));
    CHECK(after == foreign_device);
  }

  g_scenario = "wrong-device-rejection/plan-compute";
  {
    PublicBatch wrong = batch;
    wrong.bind();
    wrong.descriptor.positions = {foreign_positions, wrong.positions.size() * sizeof(double),
                                  XTBLOOM_MEMORY_CUDA_DEVICE, 0u};
    ResultOwner result;
    CUDA_CHECK(result.bind(wrong, ResultLayout::kHost));
    xtbloom_plan_t* raw_plan = nullptr;
    CUDA_CHECK(cudaSetDevice(foreign_device));
    CHECK(xtbloom_plan_create(context.get(), &batch.descriptor, &options, &raw_plan) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(raw_plan != nullptr);
    int after_create = -1;
    CUDA_CHECK(cudaGetDevice(&after_create));
    CHECK(after_create == foreign_device);
    PlanHandle plan(raw_plan);
    CUDA_CHECK(cudaSetDevice(foreign_device));
    CHECK(xtbloom_plan_compute(plan.get(), &wrong.descriptor, &options, &result.descriptor) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    int after_compute = -1;
    CUDA_CHECK(cudaGetDevice(&after_compute));
    CHECK(after_compute == foreign_device);
    bool unchanged = false;
    CUDA_CHECK(result.unchanged(unchanged));
    CHECK(unchanged);
    bool guards = false;
    CUDA_CHECK(result.guards_intact(guards));
    CHECK(guards);
  }

  /* Keep every input valid so validation reaches the writable-buffer path.
   * A requested result slice owned by another GPU must still fail before any
   * host or device output canary, including result.flags, is published. */
  g_scenario = "wrong-device-rejection/foreign-output";
  {
    PublicBatch valid = batch;
    valid.bind();
    ResultOwner result;
    CUDA_CHECK(cudaSetDevice(foreign_device));
    CUDA_CHECK(result.bind(valid, ResultLayout::kDevice));

    CUDA_CHECK(cudaSetDevice(foreign_device));
    CHECK(xtbloom_compute(context.get(), &valid.descriptor, &options, &result.descriptor) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    int after = -1;
    CUDA_CHECK(cudaGetDevice(&after));
    CHECK(after == foreign_device);
    bool unchanged = false;
    CUDA_CHECK(result.unchanged(unchanged));
    CHECK(unchanged);
    CHECK(result.descriptor.flags == kResultFlagsCanary);
    bool guards = false;
    CUDA_CHECK(result.guards_intact(guards));
    CHECK(guards);
  }

  CUDA_CHECK(cudaFree(foreign_positions));
  CUDA_CHECK(cudaSetDevice(context_device));
  context.reset();
  g_scenario = "wrong-device-rejection";
  return 0;
}

/* Mislabeled public descriptors must fail deterministically without output
 * mutation even on a single-GPU host: a device allocation tagged HOST and an
 * ordinary host buffer tagged CUDA_DEVICE cannot reach the physics kernels or
 * the caller-output commit. This directly exercises the same public
 * validate_public_request_pointers path that rejects genuine cross-device
 * pointers, so both halves of the wrong-device acceptance are covered on one
 * visible GPU. */
int test_public_mislabeled_rejection(std::int32_t device, PublicBatch& batch,
                                     const xtbloom_compute_options_t& options,
                                     const MaterializedResult& reference) {
  g_scenario = "mislabeled-rejection/host-tagged-device";
  StreamOwner stream;
  CUDA_CHECK(stream.create());
  xtbloom_status_t context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle context = make_context(XTBLOOM_BACKEND_CUDA, device, stream.get(), context_status);
  CHECK(context_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(context != nullptr);
  CHECK(execute_cuda_and_compare(context.get(), batch, options, ResultLayout::kHost, reference) ==
        0);

  /* A real device allocation mislabeled as HOST: the public request validator
   * rejects it before topology staging could dereference it as host memory,
   * and every caller output byte plus the flags canary must stay intact. */
  double* device_positions = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_positions),
                        batch.positions.size() * sizeof(double)));
  {
    PublicBatch mislabeled = batch;
    mislabeled.bind();
    mislabeled.descriptor.positions = {
        device_positions, mislabeled.positions.size() * sizeof(double), XTBLOOM_MEMORY_HOST, 0u};
    ResultOwner result;
    CUDA_CHECK(result.bind(mislabeled, ResultLayout::kDevice));
    g_scenario = "mislabeled-rejection/device-as-host";
    CHECK(xtbloom_compute(context.get(), &mislabeled.descriptor, &options, &result.descriptor) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    bool unchanged = false;
    CUDA_CHECK(result.unchanged(unchanged));
    CHECK(unchanged);
    bool guards = false;
    CUDA_CHECK(result.guards_intact(guards));
    CHECK(guards);
  }

  /* An ordinary host buffer mislabeled as CUDA_DEVICE must also fail closed
   * without reaching cudaMemcpy on a bogus address. */
  std::vector<double> host_positions(batch.positions.size(), 0.5);
  {
    PublicBatch mislabeled = batch;
    mislabeled.bind();
    mislabeled.descriptor.positions = {host_positions.data(),
                                       host_positions.size() * sizeof(double),
                                       XTBLOOM_MEMORY_CUDA_DEVICE, 0u};
    ResultOwner result;
    CUDA_CHECK(result.bind(mislabeled, ResultLayout::kHost));
    g_scenario = "mislabeled-rejection/host-as-device";
    CHECK(xtbloom_compute(context.get(), &mislabeled.descriptor, &options, &result.descriptor) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
    bool unchanged = false;
    CUDA_CHECK(result.unchanged(unchanged));
    CHECK(unchanged);
    bool guards = false;
    CUDA_CHECK(result.guards_intact(guards));
    CHECK(guards);
  }

  CUDA_CHECK(cudaFree(device_positions));
  g_scenario = "mislabeled-rejection";
  return 0;
}

int test_stream_capture_transactionality(std::int32_t device, PublicBatch& batch,
                                         const xtbloom_compute_options_t& options) {
  g_scenario = "stream-capture-transactionality";
  StreamOwner stream;
  CUDA_CHECK(stream.create());
  xtbloom_status_t context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle context = make_context(XTBLOOM_BACKEND_CUDA, device, stream.get(), context_status);
  CHECK(context_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(context != nullptr);
  ResultOwner result;
  CUDA_CHECK(result.bind(batch, ResultLayout::kHost));
  xtbloom_plan_t* raw_plan = nullptr;
  CHECK(xtbloom_plan_create(context.get(), &batch.descriptor, &options, &raw_plan) ==
        XTBLOOM_STATUS_SUCCESS);
  PlanHandle plan(raw_plan);
  xtbloom_request_t* raw_request = nullptr;
  CHECK(xtbloom_request_create(context.get(), &raw_request) == XTBLOOM_STATUS_SUCCESS);
  RequestHandle request(raw_request);

  CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
  const xtbloom_status_t compute_status =
      xtbloom_compute(context.get(), &batch.descriptor, &options, &result.descriptor);
  const xtbloom_status_t enqueue_status = xtbloom_plan_compute_enqueue(
      plan.get(), &batch.descriptor, &options, &result.descriptor, request.get());
  const xtbloom_status_t context_enqueue_status = xtbloom_compute_enqueue(
      context.get(), &batch.descriptor, &options, &result.descriptor, request.get());
  cudaGraph_t graph = nullptr;
  const cudaError_t end_status = cudaStreamEndCapture(stream.get(), &graph);
  if (graph != nullptr) (void)cudaGraphDestroy(graph);
  CUDA_CHECK(end_status);
  CHECK(compute_status == XTBLOOM_STATUS_NOT_SUPPORTED);
  CHECK(enqueue_status == XTBLOOM_STATUS_NOT_SUPPORTED);
  CHECK(context_enqueue_status == XTBLOOM_STATUS_NOT_SUPPORTED);
  bool unchanged = false;
  CUDA_CHECK(result.unchanged(unchanged));
  CHECK(unchanged);
  xtbloom_request_info_t info{};
  CHECK(xtbloom_request_info_init(&info, sizeof(info)) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom_request_query(request.get(), &info) == XTBLOOM_STATUS_SUCCESS);
  CHECK(info.state == XTBLOOM_REQUEST_IDLE);
  return 0;
}

struct ThreadCall {
  xtbloom_context_t* context = nullptr;
  PublicBatch* batch = nullptr;
  const xtbloom_compute_options_t* options = nullptr;
  ResultOwner* result = nullptr;
  xtbloom_status_t status = XTBLOOM_STATUS_INTERNAL_ERROR;
  std::string error;
};

void run_thread_call(ThreadCall& call, std::atomic<int>& ready, std::atomic<bool>& start) {
  ready.fetch_add(1, std::memory_order_release);
  while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
  call.status = xtbloom_compute(call.context, &call.batch->descriptor, call.options,
                                &call.result->descriptor);
  if (call.status != XTBLOOM_STATUS_SUCCESS) call.error = xtbloom_get_last_error();
}

int verify_thread_call(ThreadCall& call, const MaterializedResult& reference,
                       const xtbloom_compute_options_t& options) {
  if (call.status != XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "threaded public CUDA call failed in %s: status=%d error=%s\n", g_scenario,
                 static_cast<int>(call.status), call.error.c_str());
    return __LINE__;
  }
  MaterializedResult actual;
  CUDA_CHECK(call.result->materialize(actual));
  return compare_result(*call.result, actual, reference, options);
}

int test_same_context_serialization(std::int32_t device, xtbloom_context_t* cpu_context,
                                    const PublicBatch& seed,
                                    const xtbloom_compute_options_t& options) {
  g_scenario = "same-context/two-threads";
  PublicBatch first = seed;
  PublicBatch second = seed;
  first.perturb(0.004);
  second.perturb(0.011);
  MaterializedResult first_reference;
  MaterializedResult second_reference;
  CHECK(run_cpu_reference(cpu_context, first, options, first_reference) == 0);
  CHECK(run_cpu_reference(cpu_context, second, options, second_reference) == 0);

  StreamOwner stream;
  CUDA_CHECK(stream.create());
  xtbloom_status_t context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle context = make_context(XTBLOOM_BACKEND_CUDA, device, stream.get(), context_status);
  CHECK(context_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(context != nullptr);
  ResultOwner first_result;
  ResultOwner second_result;
  CUDA_CHECK(first_result.bind(first, ResultLayout::kHost));
  CUDA_CHECK(second_result.bind(second, ResultLayout::kHost));

  ThreadCall first_call{
      context.get(), &first, &options, &first_result, XTBLOOM_STATUS_INTERNAL_ERROR, {}};
  ThreadCall second_call{
      context.get(), &second, &options, &second_result, XTBLOOM_STATUS_INTERNAL_ERROR, {}};
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::thread first_thread(run_thread_call, std::ref(first_call), std::ref(ready), std::ref(start));
  std::thread second_thread(run_thread_call, std::ref(second_call), std::ref(ready),
                            std::ref(start));
  while (ready.load(std::memory_order_acquire) != 2) std::this_thread::yield();
  start.store(true, std::memory_order_release);
  first_thread.join();
  second_thread.join();
  CHECK(verify_thread_call(first_call, first_reference, options) == 0);
  CHECK(verify_thread_call(second_call, second_reference, options) == 0);
  return 0;
}

int test_independent_contexts(std::int32_t device, xtbloom_context_t* cpu_context,
                              const PublicBatch& seed, const xtbloom_compute_options_t& options) {
  g_scenario = "independent-contexts/two-threads";
  PublicBatch first = seed;
  PublicBatch second = seed;
  first.perturb(0.006);
  second.perturb(0.014);
  MaterializedResult first_reference;
  MaterializedResult second_reference;
  CHECK(run_cpu_reference(cpu_context, first, options, first_reference) == 0);
  CHECK(run_cpu_reference(cpu_context, second, options, second_reference) == 0);

  StreamOwner first_stream;
  StreamOwner second_stream;
  CUDA_CHECK(first_stream.create());
  CUDA_CHECK(second_stream.create());
  xtbloom_status_t first_context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  xtbloom_status_t second_context_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle first_context =
      make_context(XTBLOOM_BACKEND_CUDA, device, first_stream.get(), first_context_status);
  ContextHandle second_context =
      make_context(XTBLOOM_BACKEND_CUDA, device, second_stream.get(), second_context_status);
  CHECK(first_context_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(second_context_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(first_context != nullptr);
  CHECK(second_context != nullptr);
  ResultOwner first_result;
  ResultOwner second_result;
  CUDA_CHECK(first_result.bind(first, ResultLayout::kDevice));
  CUDA_CHECK(second_result.bind(second, ResultLayout::kMixed));

  ThreadCall first_call{first_context.get(),           &first, &options, &first_result,
                        XTBLOOM_STATUS_INTERNAL_ERROR, {}};
  ThreadCall second_call{second_context.get(),          &second, &options, &second_result,
                         XTBLOOM_STATUS_INTERNAL_ERROR, {}};
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::thread first_thread(run_thread_call, std::ref(first_call), std::ref(ready), std::ref(start));
  std::thread second_thread(run_thread_call, std::ref(second_call), std::ref(ready),
                            std::ref(start));
  while (ready.load(std::memory_order_acquire) != 2) std::this_thread::yield();
  start.store(true, std::memory_order_release);
  first_thread.join();
  second_thread.join();
  CHECK(verify_thread_call(first_call, first_reference, options) == 0);
  CHECK(verify_thread_call(second_call, second_reference, options) == 0);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const bool request_only = argc == 2 && std::strcmp(argv[1], "--request-only") == 0;
  const bool request_sanitizer = argc == 2 && std::strcmp(argv[1], "--request-sanitizer") == 0;
  const bool request_profile = argc == 2 && std::strcmp(argv[1], "--request-profile") == 0;
  const bool context_request_sanitizer =
      argc == 2 && std::strcmp(argv[1], "--context-request-sanitizer") == 0;
  const bool context_request_profile =
      argc == 2 && std::strcmp(argv[1], "--context-request-profile") == 0;
  const bool mixer_sanitizer = argc == 2 && std::strcmp(argv[1], "--mixer-sanitizer") == 0;
  if (argc != 1 && !request_only && !request_sanitizer && !request_profile &&
      !context_request_sanitizer && !context_request_profile && !mixer_sanitizer) {
    std::fprintf(stderr,
                 "usage: %s [--request-only|--request-sanitizer|--request-profile|"
                 "--context-request-sanitizer|--context-request-profile|"
                 "--mixer-sanitizer]\n",
                 argv[0]);
    return 2;
  }
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status == cudaErrorNoDevice || device_count == 0) {
    std::puts("cuda_public_api_test: SKIP (no CUDA device)");
    return 0;
  }
  if (count_status != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(count_status));
    return 1;
  }
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDevice failed\n");
    return 1;
  }

  PublicBatch batch;
  g_scenario = "fixture";
  if (const int line = make_fixture_batch(batch); line != 0) return line;
  const xtbloom_compute_options_t options = make_compute_options();
  xtbloom_status_t cpu_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  ContextHandle cpu_context = make_context(XTBLOOM_BACKEND_CPU, -1, nullptr, cpu_status);
  if (cpu_status != XTBLOOM_STATUS_SUCCESS || cpu_context == nullptr) {
    std::fprintf(stderr, "failed to create CPU reference context: %s\n", xtbloom_get_last_error());
    return 1;
  }
  MaterializedResult reference;
  g_scenario = "CPU-reference";
  if (const int line = run_cpu_reference(cpu_context.get(), batch, options, reference); line != 0) {
    return line;
  }

  /* Compute Sanitizer can make unrelated public matrices prohibitively slow
   * or invalidate real-time blocked-stream watchdogs through instrumentation
   * overhead. These opt-in entry points preserve default CTest coverage while
   * keeping each production path independently sanitizable. */
  if (request_only) {
    if (const int line = test_cuda_context_enqueue(device, cpu_context.get(), options); line != 0) {
      return line;
    }
    return test_cuda_plan_api(device, cpu_context.get(), options, PlanTestMode::kRequestOnly);
  }
  if (request_sanitizer) {
    return test_cuda_plan_api(device, cpu_context.get(), options, PlanTestMode::kSanitizer);
  }
  if (request_profile) {
    return test_cuda_plan_api(device, cpu_context.get(), options,
                              PlanTestMode::kProfileSteadyState);
  }
  if (context_request_sanitizer) {
    return test_cuda_context_enqueue(device, cpu_context.get(), options, PlanTestMode::kSanitizer);
  }
  if (context_request_profile) {
    return test_cuda_context_enqueue(device, cpu_context.get(), options,
                                     PlanTestMode::kProfileSteadyState);
  }
  if (mixer_sanitizer) {
    return test_public_mixer_controls_and_reproducibility(device, cpu_context.get());
  }

  if (const int line = test_host_device_mixed_and_streams(device, batch, options, reference);
      line != 0) {
    return line;
  }
  if (const int line = test_public_peer_failure_isolated(device, cpu_context.get(), options);
      line != 0) {
    return line;
  }
  if (const int line = test_cuda_plan_api(device, cpu_context.get(), options); line != 0) {
    return line;
  }
  if (const int line = test_cuda_context_enqueue(device, cpu_context.get(), options); line != 0) {
    return line;
  }
  if (const int line = test_public_representability_matrix(device, cpu_context.get(), options);
      line != 0) {
    return line;
  }
  if (const int line = test_public_mixer_controls_and_reproducibility(device, cpu_context.get());
      line != 0) {
    return line;
  }
  if (const int line = test_public_warm_start_transactions(device); line != 0) return line;
  if (const int line = test_input_descriptor_matrix(device, cpu_context.get(), options);
      line != 0) {
    return line;
  }
  if (const int line =
          test_current_device_restored(device_count, device, batch, options, reference);
      line != 0) {
    return line;
  }
  if (const int line = test_public_wrong_device_rejection(device_count, device, batch, options);
      line != 0) {
    return line;
  }
  if (const int line = test_public_mislabeled_rejection(device, batch, options, reference);
      line != 0) {
    return line;
  }
  if (const int line = test_stream_capture_transactionality(device, batch, options); line != 0) {
    return line;
  }
  if (const int line = test_same_context_serialization(device, cpu_context.get(), batch, options);
      line != 0) {
    return line;
  }
  if (const int line = test_independent_contexts(device, cpu_context.get(), batch, options);
      line != 0) {
    return line;
  }
  return 0;
}
