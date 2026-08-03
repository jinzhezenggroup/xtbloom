#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "gpuxtb/gpuxtb.h"
#include "tests/support/gfn2_scc_test_case.hpp"

/* End-to-end contract test for the public single-flight CUDA transaction. */
namespace {

using gpuxtb::test::gfn2::HostSccCase;
using gpuxtb::test::gfn2::HostSccCaseOptions;
using gpuxtb::test::gfn2::SmallSystemKind;

constexpr std::uint32_t kRequestedProperties =
    GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES;
constexpr std::uint32_t kResultFlagsCanary = UINT32_C(0xa55a39c6);
constexpr double kEnergyAbsoluteTolerance = 3.0e-8;
constexpr double kEnergyRelativeTolerance = 3.0e-8;
constexpr double kChargeAbsoluteTolerance = 1.0e-7;
constexpr double kChargeRelativeTolerance = 1.0e-7;
constexpr double kForceAbsoluteTolerance = 3.0e-7;
constexpr double kForceRelativeTolerance = 3.0e-7;

const char* g_scenario = "uninitialized";

#define CHECK(condition)                                                                        \
  do {                                                                                          \
    if (!(condition)) {                                                                         \
      std::fprintf(stderr, "public CUDA API check failed in %s at %s:%d: %s; %s\n", g_scenario, \
                   __FILE__, __LINE__, #condition, gpuxtb_get_last_error());                    \
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
  void operator()(gpuxtb_context_t* context) const noexcept { gpuxtb_context_destroy(context); }
};

using ContextHandle = std::unique_ptr<gpuxtb_context_t, ContextDeleter>;

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
gpuxtb_const_buffer_t host_input(const std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), GPUXTB_MEMORY_HOST,
          0u};
}

struct PublicBatch {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int64_t> point_charge_offsets;
  std::vector<double> point_charge_positions;
  std::vector<double> point_charge_values;
  std::vector<double> point_charge_gammas;
  std::vector<double> atomic_potential_shifts;
  std::vector<std::int64_t> charge_response_offsets;
  std::vector<double> charge_response_matrix;
  gpuxtb_batch_t descriptor{};

  void bind() noexcept {
    (void)gpuxtb_batch_init(&descriptor, sizeof(descriptor));
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

gpuxtb_compute_options_t make_compute_options() noexcept {
  gpuxtb_compute_options_t options{};
  (void)gpuxtb_compute_options_init(&options, sizeof(options));
  options.model = GPUXTB_MODEL_GFN2_XTB;
  options.flags = kRequestedProperties;
  options.max_scc_iterations = 64;
  options.charge_tolerance = 1.0e-8;
  options.energy_tolerance = 1.0e-8;
  options.electronic_temperature = 0.0;
  return options;
}

ContextHandle make_context(gpuxtb_backend_t backend, std::int32_t device_id, cudaStream_t stream,
                           gpuxtb_status_t& status) {
  gpuxtb_context_options_t options{};
  status = gpuxtb_context_options_init(&options, sizeof(options));
  if (status != GPUXTB_STATUS_SUCCESS) return {};
  options.backend = backend;
  options.device_id = device_id;
  options.stream = reinterpret_cast<void*>(stream);
  gpuxtb_context_t* raw = nullptr;
  status = gpuxtb_context_create(&options, &raw);
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

  [[nodiscard]] gpuxtb_const_buffer_t descriptor() const noexcept {
    return {data_, count_ * sizeof(T), GPUXTB_MEMORY_CUDA_DEVICE, 0u};
  }

  [[nodiscard]] std::uintptr_t address() const noexcept {
    return reinterpret_cast<std::uintptr_t>(data_);
  }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

/* Mirrors every public input leaf so the end-to-end matrix can exercise all
 * six topology buffers and every numerical QM/MM buffer without managed
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
  [[nodiscard]] std::array<std::uintptr_t, 12> identity() const noexcept {
    return {atom_offsets_.address(),
            atomic_numbers_.address(),
            molecular_charges_.address(),
            unpaired_electrons_.address(),
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

  [[nodiscard]] gpuxtb_buffer_t descriptor() noexcept {
    void* payload = placement_ == Placement::kHost ? static_cast<void*>(host_.data() + 1u)
                                                   : static_cast<void*>(device_ + 1u);
    return {payload, count_ * sizeof(T),
            placement_ == Placement::kHost ? GPUXTB_MEMORY_HOST : GPUXTB_MEMORY_CUDA_DEVICE, 0u};
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

enum class ResultLayout { kHost, kDevice, kMixed };

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
    const Placement forces_placement =
        layout == ResultLayout::kHost ? Placement::kHost : Placement::kDevice;
    const Placement iterations_placement =
        layout == ResultLayout::kMixed ? Placement::kDevice : all;
    const Placement statuses_placement = layout == ResultLayout::kMixed ? Placement::kDevice : all;

    cudaError_t status = energies_.initialize(systems, all);
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

    (void)gpuxtb_batch_result_init(&descriptor, sizeof(descriptor));
    descriptor.flags = kResultFlagsCanary;
    descriptor.energies =
        (flags & GPUXTB_COMPUTE_ENERGY) != 0u ? energies_.descriptor() : gpuxtb_buffer_t{};
    descriptor.forces =
        (flags & GPUXTB_COMPUTE_FORCES) != 0u ? forces_.descriptor() : gpuxtb_buffer_t{};
    descriptor.atomic_charges =
        (flags & GPUXTB_COMPUTE_ATOMIC_CHARGES) != 0u ? charges_.descriptor() : gpuxtb_buffer_t{};
    descriptor.point_charge_forces = (flags & GPUXTB_COMPUTE_POINT_CHARGE_FORCES) != 0u
                                         ? point_forces_.descriptor()
                                         : gpuxtb_buffer_t{};
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

  gpuxtb_batch_result_t descriptor{};

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
                   const MaterializedResult& expected, const gpuxtb_compute_options_t& options) {
  bool guards = false;
  CUDA_CHECK(owner.guards_intact(guards));
  CHECK(guards);
  CHECK(actual.flags == expected.flags);
  CHECK(actual.statuses.size() == expected.statuses.size());
  CHECK(actual.converged.size() == expected.converged.size());
  CHECK(actual.iterations.size() == expected.iterations.size());
  for (std::size_t system = 0; system < actual.statuses.size(); ++system) {
    CHECK(actual.statuses[system] == GPUXTB_STATUS_SUCCESS);
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
  const gpuxtb_status_t status = HostSccCase::create(options, fixture, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    std::fprintf(stderr, "failed to build public CUDA API fixture: %s\n", error.c_str());
    return __LINE__;
  }
  batch = PublicBatch::from_fixture(fixture);
  return 0;
}

int make_fixture_batch(PublicBatch& batch) { return make_fixture_batch(4u, false, batch); }

int run_cpu_reference(gpuxtb_context_t* cpu_context, PublicBatch& batch,
                      const gpuxtb_compute_options_t& options, MaterializedResult& reference) {
  ResultOwner result;
  bind_inputs(batch, nullptr, InputLayout::kHost);
  CUDA_CHECK(result.bind(batch, ResultLayout::kHost, options.flags));
  CHECK(gpuxtb_compute(cpu_context, &batch.descriptor, &options, &result.descriptor) ==
        GPUXTB_STATUS_SUCCESS);
  CUDA_CHECK(result.materialize(reference));
  bool guards = false;
  CUDA_CHECK(result.guards_intact(guards));
  CHECK(guards);
  for (std::size_t system = 0; system < reference.statuses.size(); ++system) {
    CHECK(reference.statuses[system] == GPUXTB_STATUS_SUCCESS);
    CHECK(reference.converged[system] == 1u);
  }
  return 0;
}

int execute_cuda_and_compare(gpuxtb_context_t* cuda_context, PublicBatch& batch,
                             const gpuxtb_compute_options_t& options, ResultLayout layout,
                             const MaterializedResult& reference) {
  ResultOwner result;
  CUDA_CHECK(result.bind(batch, layout, options.flags));
  CHECK(gpuxtb_compute(cuda_context, &batch.descriptor, &options, &result.descriptor) ==
        GPUXTB_STATUS_SUCCESS);
  MaterializedResult actual;
  CUDA_CHECK(result.materialize(actual));
  return compare_result(result, actual, reference, options);
}

int test_host_device_mixed_and_streams(std::int32_t device, PublicBatch& batch,
                                       const gpuxtb_compute_options_t& options,
                                       const MaterializedResult& reference) {
  g_scenario = "host-output/default-stream";
  gpuxtb_status_t context_status = GPUXTB_STATUS_INTERNAL_ERROR;
  ContextHandle default_context =
      make_context(GPUXTB_BACKEND_CUDA, device, nullptr, context_status);
  CHECK(context_status == GPUXTB_STATUS_SUCCESS);
  CHECK(default_context != nullptr);
  CHECK(execute_cuda_and_compare(default_context.get(), batch, options, ResultLayout::kHost,
                                 reference) == 0);

  g_scenario = "host-output/custom-stream";
  StreamOwner stream;
  CUDA_CHECK(stream.create());
  ContextHandle custom_context =
      make_context(GPUXTB_BACKEND_CUDA, device, stream.get(), context_status);
  CHECK(context_status == GPUXTB_STATUS_SUCCESS);
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

bool is_cuda_buffer(const gpuxtb_const_buffer_t& buffer) noexcept {
  return buffer.data != nullptr && buffer.memory_space == GPUXTB_MEMORY_CUDA_DEVICE;
}

bool is_host_buffer(const gpuxtb_const_buffer_t& buffer) noexcept {
  return buffer.data != nullptr && buffer.memory_space == GPUXTB_MEMORY_HOST;
}

int verify_input_layout(const PublicBatch& batch, InputLayout layout, bool qmmm) {
  if (layout == InputLayout::kDevice) {
    CHECK(is_cuda_buffer(batch.descriptor.atom_offsets));
    CHECK(is_cuda_buffer(batch.descriptor.atomic_numbers));
    CHECK(is_cuda_buffer(batch.descriptor.molecular_charges));
    CHECK(is_cuda_buffer(batch.descriptor.unpaired_electrons));
    CHECK(is_cuda_buffer(batch.descriptor.positions));
    if (qmmm) {
      /* The six topology fields include point and dense-response partitions. */
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
int test_input_descriptor_matrix(std::int32_t device, gpuxtb_context_t* cpu_context,
                                 const gpuxtb_compute_options_t& base_options) {
  StreamOwner stream;
  CUDA_CHECK(stream.create());
  gpuxtb_status_t context_status = GPUXTB_STATUS_INTERNAL_ERROR;
  ContextHandle context = make_context(GPUXTB_BACKEND_CUDA, device, stream.get(), context_status);
  CHECK(context_status == GPUXTB_STATUS_SUCCESS);
  CHECK(context != nullptr);

  std::string scenario;
  for (bool qmmm : {false, true}) {
    gpuxtb_compute_options_t options = base_options;
    if (qmmm) options.flags |= GPUXTB_COMPUTE_POINT_CHARGE_FORCES;
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
                                 const gpuxtb_compute_options_t& options,
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
  gpuxtb_status_t context_status = GPUXTB_STATUS_INTERNAL_ERROR;
  ContextHandle context =
      make_context(GPUXTB_BACKEND_CUDA, context_device, stream.get(), context_status);
  CHECK(context_status == GPUXTB_STATUS_SUCCESS);
  CHECK(context != nullptr);

  const int caller_device = context_device == 0 ? 1 : 0;
  CUDA_CHECK(cudaSetDevice(caller_device));
  int before = -1;
  CUDA_CHECK(cudaGetDevice(&before));
  CHECK(before == caller_device);

  ResultOwner result;
  CUDA_CHECK(result.bind(batch, ResultLayout::kHost));
  CHECK(gpuxtb_compute(context.get(), &batch.descriptor, &options, &result.descriptor) ==
        GPUXTB_STATUS_SUCCESS);
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

int test_stream_capture_transactionality(std::int32_t device, PublicBatch& batch,
                                         const gpuxtb_compute_options_t& options) {
  g_scenario = "stream-capture-transactionality";
  StreamOwner stream;
  CUDA_CHECK(stream.create());
  gpuxtb_status_t context_status = GPUXTB_STATUS_INTERNAL_ERROR;
  ContextHandle context = make_context(GPUXTB_BACKEND_CUDA, device, stream.get(), context_status);
  CHECK(context_status == GPUXTB_STATUS_SUCCESS);
  CHECK(context != nullptr);
  ResultOwner result;
  CUDA_CHECK(result.bind(batch, ResultLayout::kHost));

  CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
  const gpuxtb_status_t status =
      gpuxtb_compute(context.get(), &batch.descriptor, &options, &result.descriptor);
  cudaGraph_t graph = nullptr;
  const cudaError_t end_status = cudaStreamEndCapture(stream.get(), &graph);
  if (graph != nullptr) (void)cudaGraphDestroy(graph);
  CUDA_CHECK(end_status);
  CHECK(status == GPUXTB_STATUS_NOT_SUPPORTED);
  bool unchanged = false;
  CUDA_CHECK(result.unchanged(unchanged));
  CHECK(unchanged);
  return 0;
}

struct ThreadCall {
  gpuxtb_context_t* context = nullptr;
  PublicBatch* batch = nullptr;
  const gpuxtb_compute_options_t* options = nullptr;
  ResultOwner* result = nullptr;
  gpuxtb_status_t status = GPUXTB_STATUS_INTERNAL_ERROR;
  std::string error;
};

void run_thread_call(ThreadCall& call, std::atomic<int>& ready, std::atomic<bool>& start) {
  ready.fetch_add(1, std::memory_order_release);
  while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
  call.status =
      gpuxtb_compute(call.context, &call.batch->descriptor, call.options, &call.result->descriptor);
  if (call.status != GPUXTB_STATUS_SUCCESS) call.error = gpuxtb_get_last_error();
}

int verify_thread_call(ThreadCall& call, const MaterializedResult& reference,
                       const gpuxtb_compute_options_t& options) {
  if (call.status != GPUXTB_STATUS_SUCCESS) {
    std::fprintf(stderr, "threaded public CUDA call failed in %s: status=%d error=%s\n", g_scenario,
                 static_cast<int>(call.status), call.error.c_str());
    return __LINE__;
  }
  MaterializedResult actual;
  CUDA_CHECK(call.result->materialize(actual));
  return compare_result(*call.result, actual, reference, options);
}

int test_same_context_serialization(std::int32_t device, gpuxtb_context_t* cpu_context,
                                    const PublicBatch& seed,
                                    const gpuxtb_compute_options_t& options) {
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
  gpuxtb_status_t context_status = GPUXTB_STATUS_INTERNAL_ERROR;
  ContextHandle context = make_context(GPUXTB_BACKEND_CUDA, device, stream.get(), context_status);
  CHECK(context_status == GPUXTB_STATUS_SUCCESS);
  CHECK(context != nullptr);
  ResultOwner first_result;
  ResultOwner second_result;
  CUDA_CHECK(first_result.bind(first, ResultLayout::kHost));
  CUDA_CHECK(second_result.bind(second, ResultLayout::kHost));

  ThreadCall first_call{
      context.get(), &first, &options, &first_result, GPUXTB_STATUS_INTERNAL_ERROR, {}};
  ThreadCall second_call{
      context.get(), &second, &options, &second_result, GPUXTB_STATUS_INTERNAL_ERROR, {}};
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

int test_independent_contexts(std::int32_t device, gpuxtb_context_t* cpu_context,
                              const PublicBatch& seed, const gpuxtb_compute_options_t& options) {
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
  gpuxtb_status_t first_context_status = GPUXTB_STATUS_INTERNAL_ERROR;
  gpuxtb_status_t second_context_status = GPUXTB_STATUS_INTERNAL_ERROR;
  ContextHandle first_context =
      make_context(GPUXTB_BACKEND_CUDA, device, first_stream.get(), first_context_status);
  ContextHandle second_context =
      make_context(GPUXTB_BACKEND_CUDA, device, second_stream.get(), second_context_status);
  CHECK(first_context_status == GPUXTB_STATUS_SUCCESS);
  CHECK(second_context_status == GPUXTB_STATUS_SUCCESS);
  CHECK(first_context != nullptr);
  CHECK(second_context != nullptr);
  ResultOwner first_result;
  ResultOwner second_result;
  CUDA_CHECK(first_result.bind(first, ResultLayout::kDevice));
  CUDA_CHECK(second_result.bind(second, ResultLayout::kMixed));

  ThreadCall first_call{first_context.get(),          &first, &options, &first_result,
                        GPUXTB_STATUS_INTERNAL_ERROR, {}};
  ThreadCall second_call{second_context.get(),         &second, &options, &second_result,
                         GPUXTB_STATUS_INTERNAL_ERROR, {}};
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

int main() {
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
  const gpuxtb_compute_options_t options = make_compute_options();
  gpuxtb_status_t cpu_status = GPUXTB_STATUS_INTERNAL_ERROR;
  ContextHandle cpu_context = make_context(GPUXTB_BACKEND_CPU, -1, nullptr, cpu_status);
  if (cpu_status != GPUXTB_STATUS_SUCCESS || cpu_context == nullptr) {
    std::fprintf(stderr, "failed to create CPU reference context: %s\n", gpuxtb_get_last_error());
    return 1;
  }
  MaterializedResult reference;
  g_scenario = "CPU-reference";
  if (const int line = run_cpu_reference(cpu_context.get(), batch, options, reference); line != 0) {
    return line;
  }

  if (const int line = test_host_device_mixed_and_streams(device, batch, options, reference);
      line != 0) {
    return line;
  }
  if (const int line = test_input_descriptor_matrix(device, cpu_context.get(), options);
      line != 0) {
    return line;
  }
  if (const int line =
          test_current_device_restored(device_count, device, batch, options, reference);
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
