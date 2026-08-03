#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_scc_energy.cuh"

namespace {

using gpuxtb::detail::cuda::evaluate_gfn2_scc_electronic_energy_cuda;
using gpuxtb::detail::cuda::Gfn2SccEnergyDeviceBatch;
using gpuxtb::detail::cuda::Gfn2SccEnergyDeviceError;
using gpuxtb::detail::cuda::Gfn2SccEnergyDeviceWorkspace;
using gpuxtb::detail::cuda::Gfn2SccIterationDeviceActivity;
using gpuxtb::detail::cuda::reset_gfn2_scc_energy_device_errors_cuda;

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t elements) { CHECK_CUDA(allocate(elements)); }
  ~DeviceBuffer() { cudaFree(pointer_); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : pointer_(std::exchange(other.pointer_, nullptr)), elements_(other.elements_) {
    other.elements_ = 0u;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      cudaFree(pointer_);
      pointer_ = std::exchange(other.pointer_, nullptr);
      elements_ = other.elements_;
      other.elements_ = 0u;
    }
    return *this;
  }

  cudaError_t allocate(std::size_t elements) {
    cudaFree(pointer_);
    pointer_ = nullptr;
    elements_ = elements;
    return elements == 0u ? cudaSuccess
                          : cudaMalloc(reinterpret_cast<void**>(&pointer_), elements * sizeof(T));
  }

  cudaError_t upload(const std::vector<T>& host) {
    if (host.size() != elements_) {
      return cudaErrorInvalidValue;
    }
    return elements_ == 0u
               ? cudaSuccess
               : cudaMemcpy(pointer_, host.data(), elements_ * sizeof(T), cudaMemcpyHostToDevice);
  }

  cudaError_t download(std::vector<T>& host) const {
    host.resize(elements_);
    return elements_ == 0u
               ? cudaSuccess
               : cudaMemcpy(host.data(), pointer_, elements_ * sizeof(T), cudaMemcpyDeviceToHost);
  }

  cudaError_t fill_bytes(int value) {
    return elements_ == 0u ? cudaSuccess : cudaMemset(pointer_, value, elements_ * sizeof(T));
  }

  T* get() const { return pointer_; }
  std::size_t size() const { return elements_; }

 private:
  static void CHECK_CUDA(cudaError_t status) {
    if (status != cudaSuccess) {
      std::fprintf(stderr, "CUDA allocation failed: %s\n", cudaGetErrorString(status));
      std::abort();
    }
  }

  T* pointer_ = nullptr;
  std::size_t elements_ = 0u;
};

struct HostCase {
  std::vector<std::int64_t> matrix_offsets;
  std::vector<double> density;
  std::vector<double> h0;
  std::vector<double> entropies;
  std::vector<std::uint8_t> active;
  std::vector<double> expected_core;
  std::vector<double> expected_free;
  double temperature = 0.0035;
  std::uint64_t plan_token = 0x534343454e455247ULL;
};

HostCase make_case(std::size_t batch_size) {
  HostCase data;
  data.matrix_offsets.resize(batch_size + 1u, 0);
  data.entropies.resize(batch_size);
  data.active.assign(batch_size, 1u);
  data.expected_core.resize(batch_size, 0.0);
  data.expected_free.resize(batch_size, 0.0);

  for (std::size_t system = 0; system < batch_size; ++system) {
    const std::int64_t orbitals =
        batch_size == 1u ? 3
                         : (system % 11u == 7u ? 0 : 1 + static_cast<std::int64_t>(system % 5u));
    const std::int64_t elements = orbitals * orbitals;
    data.matrix_offsets[system + 1u] = data.matrix_offsets[system] + elements;
    data.entropies[system] = 0.02 * static_cast<double>((system * 13u) % 17u);
    double core = 0.0;
    for (std::int64_t local = 0; local < elements; ++local) {
      const double density =
          0.03 * static_cast<double>(1 + ((system + static_cast<std::size_t>(local)) % 19u));
      const double h0 = -0.11 + 0.007 * static_cast<double>(
                                            (3u * system + static_cast<std::size_t>(local)) % 23u);
      data.density.push_back(density);
      data.h0.push_back(h0);
      core = std::fma(h0, density, core);
    }
    data.expected_core[system] = core;
    data.expected_free[system] = std::fma(-data.temperature, data.entropies[system], core);
  }
  return data;
}

struct DeviceCase {
  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<double> density;
  DeviceBuffer<double> h0;
  DeviceBuffer<double> entropies;
  DeviceBuffer<std::uint8_t> active;
  DeviceBuffer<double> core;
  DeviceBuffer<double> free;
  DeviceBuffer<double> core_scratch;
  DeviceBuffer<double> free_scratch;
  DeviceBuffer<std::uint32_t> sequence_active;
  DeviceBuffer<std::uint32_t> canonical_sequence;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;
  Gfn2SccEnergyDeviceBatch batch;
  Gfn2SccEnergyDeviceWorkspace workspace;

  explicit DeviceCase(const HostCase& host)
      : matrix_offsets(host.matrix_offsets.size()),
        density(host.density.size()),
        h0(host.h0.size()),
        entropies(host.entropies.size()),
        active(host.active.size()),
        core(host.entropies.size()),
        free(host.entropies.size()),
        core_scratch(host.entropies.size()),
        free_scratch(host.entropies.size()),
        sequence_active(1u),
        canonical_sequence(1u),
        system_errors(host.entropies.size()),
        device_error(1u) {
    CHECK_CUDA(matrix_offsets.upload(host.matrix_offsets));
    CHECK_CUDA(density.upload(host.density));
    CHECK_CUDA(h0.upload(host.h0));
    CHECK_CUDA(entropies.upload(host.entropies));
    CHECK_CUDA(active.upload(host.active));
    batch = {static_cast<std::int64_t>(host.entropies.size()),
             static_cast<std::int64_t>(host.density.size()),
             static_cast<std::int64_t>(host.matrix_offsets.size()), host.plan_token,
             matrix_offsets.get()};
    workspace = {core_scratch.get(),
                 free_scratch.get(),
                 sequence_active.get(),
                 static_cast<std::int64_t>(host.entropies.size()),
                 1,
                 host.plan_token};
  }

 private:
  static void CHECK_CUDA(cudaError_t status) {
    if (status != cudaSuccess) {
      std::fprintf(stderr, "CUDA fixture setup failed: %s\n", cudaGetErrorString(status));
      std::abort();
    }
  }
};

bool near(double actual, double expected, double tolerance = 2.0e-13) {
  return std::abs(actual - expected) <=
         tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

int launch(DeviceCase& device, const HostCase& host, cudaStream_t stream = nullptr,
           const std::uint8_t* active = nullptr) {
  CHECK(reset_gfn2_scc_energy_device_errors_cuda(device.batch.batch_size,
                                                 device.system_errors.get(),
                                                 device.device_error.get(), stream) == cudaSuccess);
  CHECK(evaluate_gfn2_scc_electronic_energy_cuda(
            device.batch, device.density.get(), device.h0.get(), device.entropies.get(),
            host.temperature, active == nullptr ? device.active.get() : active, device.core.get(),
            device.free.get(), device.workspace, device.system_errors.get(),
            device.device_error.get(), stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  return 0;
}

Gfn2SccIterationDeviceActivity canonical_activity(const DeviceCase& device) {
  return {device.active.get(), device.canonical_sequence.get(), device.batch.batch_size, 1,
          device.batch.plan_token};
}

int test_batch_parity_and_custom_stream() {
  for (const std::size_t batch_size : std::array<std::size_t, 4>{{1u, 8u, 32u, 128u}}) {
    HostCase host = make_case(batch_size);
    DeviceCase device(host);
    const double sentinel = 91.25;
    std::vector<double> initial(batch_size, sentinel);
    CHECK(device.core.upload(initial) == cudaSuccess);
    CHECK(device.free.upload(initial) == cudaSuccess);
    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    const int status = launch(device, host, stream);
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    CHECK(status == 0);

    std::vector<double> core;
    std::vector<double> free;
    std::vector<std::uint32_t> system_errors;
    std::vector<std::uint32_t> device_error;
    CHECK(device.core.download(core) == cudaSuccess);
    CHECK(device.free.download(free) == cudaSuccess);
    CHECK(device.system_errors.download(system_errors) == cudaSuccess);
    CHECK(device.device_error.download(device_error) == cudaSuccess);
    CHECK(device_error[0] == static_cast<std::uint32_t>(Gfn2SccEnergyDeviceError::kSuccess));
    for (std::size_t system = 0; system < batch_size; ++system) {
      CHECK(system_errors[system] ==
            static_cast<std::uint32_t>(Gfn2SccEnergyDeviceError::kSuccess));
      CHECK(near(core[system], host.expected_core[system]));
      CHECK(near(free[system], host.expected_free[system]));
    }
  }
  return 0;
}

int test_all_empty_batch() {
  HostCase host;
  host.matrix_offsets.assign(9u, 0);
  host.entropies.resize(8u);
  host.active.assign(8u, 1u);
  host.expected_core.assign(8u, 0.0);
  host.expected_free.resize(8u);
  for (std::size_t system = 0; system < 8u; ++system) {
    host.entropies[system] = 0.125 * static_cast<double>(system);
    host.expected_free[system] = -host.temperature * host.entropies[system];
  }
  DeviceCase device(host);
  CHECK(launch(device, host) == 0);
  std::vector<double> core;
  std::vector<double> free;
  CHECK(device.core.download(core) == cudaSuccess);
  CHECK(device.free.download(free) == cudaSuccess);
  for (std::size_t system = 0; system < 8u; ++system) {
    CHECK(core[system] == 0.0);
    CHECK(near(free[system], host.expected_free[system]));
  }
  return 0;
}

int test_peer_isolation_and_inactive_skip() {
  HostCase host = make_case(4u);
  const std::size_t bad = 1u;
  const std::size_t inactive = 2u;
  host.density[static_cast<std::size_t>(host.matrix_offsets[bad])] =
      std::numeric_limits<double>::quiet_NaN();
  host.active[inactive] = 0u;
  host.h0[static_cast<std::size_t>(host.matrix_offsets[inactive])] =
      std::numeric_limits<double>::quiet_NaN();
  DeviceCase device(host);
  constexpr double sentinel = -731.5;
  std::vector<double> initial(4u, sentinel);
  CHECK(device.core.upload(initial) == cudaSuccess);
  CHECK(device.free.upload(initial) == cudaSuccess);
  CHECK(launch(device, host) == 0);

  std::vector<double> core;
  std::vector<double> free;
  std::vector<std::uint32_t> errors;
  CHECK(device.core.download(core) == cudaSuccess);
  CHECK(device.free.download(free) == cudaSuccess);
  CHECK(device.system_errors.download(errors) == cudaSuccess);
  CHECK(core[bad] == sentinel && free[bad] == sentinel);
  CHECK(errors[bad] == static_cast<std::uint32_t>(Gfn2SccEnergyDeviceError::kNonfiniteDensity));
  CHECK(core[inactive] == sentinel && free[inactive] == sentinel);
  CHECK(errors[inactive] == static_cast<std::uint32_t>(Gfn2SccEnergyDeviceError::kSuccess));
  CHECK(near(core[0], host.expected_core[0]) && near(free[0], host.expected_free[0]));
  CHECK(near(core[3], host.expected_core[3]) && near(free[3], host.expected_free[3]));
  return 0;
}

int test_reduction_and_addition_overflow() {
  HostCase host;
  host.matrix_offsets = {0, 256, 257};
  host.density.assign(257u, 1.0);
  host.h0.assign(257u, 0.0);
  host.h0[0] = 0.75 * std::numeric_limits<double>::max();
  host.h0[128] = 0.75 * std::numeric_limits<double>::max();
  host.h0[256] = 0.25;
  host.density[256] = 0.5;
  host.entropies = {0.0, 0.1};
  host.active = {1u, 1u};
  host.expected_core = {0.0, 0.125};
  host.expected_free = {0.0, 0.125};
  host.temperature = 0.0;
  DeviceCase reduction(host);
  std::vector<double> sentinel(2u, 88.0);
  CHECK(reduction.core.upload(sentinel) == cudaSuccess);
  CHECK(reduction.free.upload(sentinel) == cudaSuccess);
  CHECK(launch(reduction, host) == 0);
  std::vector<double> core;
  std::vector<double> free;
  std::vector<std::uint32_t> errors;
  CHECK(reduction.core.download(core) == cudaSuccess);
  CHECK(reduction.free.download(free) == cudaSuccess);
  CHECK(reduction.system_errors.download(errors) == cudaSuccess);
  CHECK(core[0] == 88.0 && free[0] == 88.0);
  CHECK(errors[0] ==
        static_cast<std::uint32_t>(Gfn2SccEnergyDeviceError::kNonfiniteCoreArithmetic));
  CHECK(near(core[1], 0.125) && near(free[1], 0.125));

  host.matrix_offsets = {0, 1, 2};
  host.density = {1.0, 0.5};
  host.h0 = {0.5 * std::numeric_limits<double>::max(), 0.25};
  host.entropies = {-0.5 * std::numeric_limits<double>::max(), 0.1};
  host.temperature = 2.0;
  DeviceCase addition(host);
  CHECK(addition.core.upload(sentinel) == cudaSuccess);
  CHECK(addition.free.upload(sentinel) == cudaSuccess);
  CHECK(launch(addition, host) == 0);
  CHECK(addition.core.download(core) == cudaSuccess);
  CHECK(addition.free.download(free) == cudaSuccess);
  CHECK(addition.system_errors.download(errors) == cudaSuccess);
  CHECK(core[0] == 88.0 && free[0] == 88.0);
  CHECK(errors[0] == static_cast<std::uint32_t>(Gfn2SccEnergyDeviceError::kNonfiniteFreeEnergy));
  CHECK(near(core[1], 0.125));
  CHECK(near(free[1], std::fma(-2.0, 0.1, 0.125)));
  return 0;
}

int test_first_error_diagnostics_are_consistent() {
  HostCase host;
  host.matrix_offsets = {0, 256};
  host.density.assign(256u, 0.0);
  host.h0.assign(256u, 0.0);
  host.density[0] = std::numeric_limits<double>::quiet_NaN();
  host.h0[128] = std::numeric_limits<double>::quiet_NaN();
  host.entropies = {0.0};
  host.active = {1u};
  host.expected_core = {0.0};
  host.expected_free = {0.0};
  DeviceCase device(host);
  CHECK(launch(device, host) == 0);
  std::vector<std::uint32_t> system_errors;
  std::vector<std::uint32_t> device_error;
  CHECK(device.system_errors.download(system_errors) == cudaSuccess);
  CHECK(device.device_error.download(device_error) == cudaSuccess);
  CHECK(system_errors[0] == device_error[0]);
  CHECK(system_errors[0] ==
            static_cast<std::uint32_t>(Gfn2SccEnergyDeviceError::kNonfiniteDensity) ||
        system_errors[0] == static_cast<std::uint32_t>(Gfn2SccEnergyDeviceError::kNonfiniteH0));
  return 0;
}

int test_offset_active_and_sticky_fail_closed() {
  HostCase host = make_case(4u);
  DeviceCase device(host);
  const std::vector<double> sentinel(4u, 17.0);
  CHECK(device.core.upload(sentinel) == cudaSuccess);
  CHECK(device.free.upload(sentinel) == cudaSuccess);
  std::vector<std::int64_t> invalid_offsets = host.matrix_offsets;
  invalid_offsets.back() -= 1;
  CHECK(device.matrix_offsets.upload(invalid_offsets) == cudaSuccess);
  CHECK(launch(device, host) == 0);
  std::vector<double> core;
  std::vector<std::uint32_t> first_error;
  CHECK(device.core.download(core) == cudaSuccess);
  CHECK(device.device_error.download(first_error) == cudaSuccess);
  CHECK(core == sentinel);
  CHECK(first_error[0] == static_cast<std::uint32_t>(Gfn2SccEnergyDeviceError::kInvalidOffsets));

  CHECK(device.matrix_offsets.upload(host.matrix_offsets) == cudaSuccess);
  host.active[1] = 2u;
  CHECK(device.active.upload(host.active) == cudaSuccess);
  CHECK(device.core.upload(sentinel) == cudaSuccess);
  CHECK(device.free.upload(sentinel) == cudaSuccess);
  CHECK(launch(device, host) == 0);
  std::vector<std::uint32_t> errors;
  CHECK(device.core.download(core) == cudaSuccess);
  CHECK(device.system_errors.download(errors) == cudaSuccess);
  CHECK(core[1] == sentinel[1]);
  CHECK(errors[1] == static_cast<std::uint32_t>(Gfn2SccEnergyDeviceError::kInvalidActiveState));
  CHECK(near(core[0], host.expected_core[0]));

  const std::uint32_t sticky = 91u;
  CHECK(cudaMemcpy(device.device_error.get(), &sticky, sizeof(sticky), cudaMemcpyHostToDevice) ==
        cudaSuccess);
  CHECK(device.core.upload(sentinel) == cudaSuccess);
  CHECK(evaluate_gfn2_scc_electronic_energy_cuda(
            device.batch, device.density.get(), device.h0.get(), device.entropies.get(),
            host.temperature, device.active.get(), device.core.get(), device.free.get(),
            device.workspace, device.system_errors.get(),
            device.device_error.get()) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(device.core.download(core) == cudaSuccess);
  CHECK(core == sentinel);
  return 0;
}

int test_host_validation_and_aliases() {
  HostCase host = make_case(2u);
  DeviceCase device(host);
  Gfn2SccEnergyDeviceBatch bad_batch = device.batch;
  bad_batch.matrix_offsets = reinterpret_cast<const std::int64_t*>(
      reinterpret_cast<const unsigned char*>(device.matrix_offsets.get()) + 1u);
  CHECK(evaluate_gfn2_scc_electronic_energy_cuda(
            bad_batch, device.density.get(), device.h0.get(), device.entropies.get(),
            host.temperature, device.active.get(), device.core.get(), device.free.get(),
            device.workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  CHECK(evaluate_gfn2_scc_electronic_energy_cuda(
            device.batch, device.density.get(), device.h0.get(), device.entropies.get(),
            host.temperature, device.active.get(), device.core.get(), device.core.get(),
            device.workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  Gfn2SccEnergyDeviceWorkspace bad_workspace = device.workspace;
  bad_workspace.core_energy_scratch = device.core.get();
  CHECK(evaluate_gfn2_scc_electronic_energy_cuda(
            device.batch, device.density.get(), device.h0.get(), device.entropies.get(),
            host.temperature, device.active.get(), device.core.get(), device.free.get(),
            bad_workspace, device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  CHECK(evaluate_gfn2_scc_electronic_energy_cuda(
            device.batch, device.density.get(), device.h0.get(), device.entropies.get(), -1.0,
            device.active.get(), device.core.get(), device.free.get(), device.workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  CHECK(reset_gfn2_scc_energy_device_errors_cuda(
            device.batch.batch_size, device.system_errors.get(), device.system_errors.get()) ==
        cudaErrorInvalidValue);
  CHECK(reset_gfn2_scc_energy_device_errors_cuda(
            std::numeric_limits<std::int64_t>::max(), device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  auto* misaligned_errors = reinterpret_cast<std::uint32_t*>(
      reinterpret_cast<unsigned char*>(device.system_errors.get()) + 1u);
  CHECK(reset_gfn2_scc_energy_device_errors_cuda(device.batch.batch_size, misaligned_errors,
                                                 device.device_error.get()) ==
        cudaErrorInvalidValue);
  return 0;
}

int test_cuda_graph_replay() {
  HostCase host = make_case(8u);
  DeviceCase device(host);
  cudaStream_t stream = nullptr;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess);
  CHECK(reset_gfn2_scc_energy_device_errors_cuda(device.batch.batch_size,
                                                 device.system_errors.get(),
                                                 device.device_error.get(), stream) == cudaSuccess);
  CHECK(evaluate_gfn2_scc_electronic_energy_cuda(
            device.batch, device.density.get(), device.h0.get(), device.entropies.get(),
            host.temperature, device.active.get(), device.core.get(), device.free.get(),
            device.workspace, device.system_errors.get(), device.device_error.get(),
            stream) == cudaSuccess);
  CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
  CHECK(cudaGraphInstantiate(&executable, graph, 0) == cudaSuccess);
  CHECK(cudaGraphLaunch(executable, stream) == cudaSuccess);
  CHECK(cudaGraphLaunch(executable, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  std::vector<double> core;
  std::vector<double> free;
  CHECK(device.core.download(core) == cudaSuccess);
  CHECK(device.free.download(free) == cudaSuccess);
  for (std::size_t system = 0; system < host.entropies.size(); ++system) {
    CHECK(near(core[system], host.expected_core[system]));
    CHECK(near(free[system], host.expected_free[system]));
  }
  CHECK(cudaGraphExecDestroy(executable) == cudaSuccess);
  CHECK(cudaGraphDestroy(graph) == cudaSuccess);
  CHECK(cudaStreamDestroy(stream) == cudaSuccess);
  return 0;
}

int test_canonical_activity_and_poisoned_inactive_topology() {
  constexpr double sentinel = -981.75;
  for (const std::size_t batch_size : std::array<std::size_t, 4>{{1u, 8u, 32u, 128u}}) {
    HostCase host = make_case(batch_size);
    DeviceCase device(host);
    CHECK(device.canonical_sequence.upload(std::vector<std::uint32_t>{1u}) == cudaSuccess);
    CHECK(device.core.upload(std::vector<double>(batch_size, sentinel)) == cudaSuccess);
    CHECK(device.free.upload(std::vector<double>(batch_size, sentinel)) == cudaSuccess);
    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    CHECK(reset_gfn2_scc_energy_device_errors_cuda(
              device.batch.batch_size, device.system_errors.get(), device.device_error.get(),
              stream) == cudaSuccess);
    CHECK(evaluate_gfn2_scc_electronic_energy_cuda(
              device.batch, device.density.get(), device.h0.get(), device.entropies.get(),
              host.temperature, canonical_activity(device), device.core.get(), device.free.get(),
              device.workspace, device.system_errors.get(), device.device_error.get(),
              stream) == cudaSuccess);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    std::vector<double> core;
    std::vector<double> free;
    CHECK(device.core.download(core) == cudaSuccess);
    CHECK(device.free.download(free) == cudaSuccess);
    for (std::size_t system = 0; system < batch_size; ++system) {
      CHECK(near(core[system], host.expected_core[system]));
      CHECK(near(free[system], host.expected_free[system]));
    }
  }

  {
    HostCase graph_host = make_case(8u);
    DeviceCase graph_device(graph_host);
    CHECK(graph_device.canonical_sequence.upload(std::vector<std::uint32_t>{1u}) == cudaSuccess);
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess);
    CHECK(reset_gfn2_scc_energy_device_errors_cuda(
              graph_device.batch.batch_size, graph_device.system_errors.get(),
              graph_device.device_error.get(), stream) == cudaSuccess);
    CHECK(evaluate_gfn2_scc_electronic_energy_cuda(
              graph_device.batch, graph_device.density.get(), graph_device.h0.get(),
              graph_device.entropies.get(), graph_host.temperature,
              canonical_activity(graph_device), graph_device.core.get(), graph_device.free.get(),
              graph_device.workspace, graph_device.system_errors.get(),
              graph_device.device_error.get(), stream) == cudaSuccess);
    CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
    CHECK(cudaGraphInstantiate(&executable, graph, 0) == cudaSuccess);
    CHECK(cudaGraphLaunch(executable, stream) == cudaSuccess);
    CHECK(cudaGraphLaunch(executable, stream) == cudaSuccess);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    std::vector<double> core;
    CHECK(graph_device.core.download(core) == cudaSuccess);
    for (std::size_t system = 0; system < core.size(); ++system) {
      CHECK(near(core[system], graph_host.expected_core[system]));
    }
    CHECK(cudaGraphExecDestroy(executable) == cudaSuccess);
    CHECK(cudaGraphDestroy(graph) == cudaSuccess);
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
  }

  HostCase host = make_case(8u);
  for (std::size_t system = 4u; system < 8u; ++system) {
    host.active[system] = 0u;
    host.entropies[system] = std::numeric_limits<double>::quiet_NaN();
    for (std::int64_t element = host.matrix_offsets[system];
         element < host.matrix_offsets[system + 1u]; ++element) {
      host.density[static_cast<std::size_t>(element)] = std::numeric_limits<double>::quiet_NaN();
      host.h0[static_cast<std::size_t>(element)] = std::numeric_limits<double>::quiet_NaN();
    }
  }
  DeviceCase device(host);
  CHECK(device.canonical_sequence.upload(std::vector<std::uint32_t>{1u}) == cudaSuccess);
  CHECK(device.core.upload(std::vector<double>(8u, sentinel)) == cudaSuccess);
  CHECK(device.free.upload(std::vector<double>(8u, sentinel)) == cudaSuccess);
  std::vector<std::int64_t> poisoned_offsets = host.matrix_offsets;
  for (std::size_t boundary = 5u; boundary < poisoned_offsets.size(); ++boundary) {
    poisoned_offsets[boundary] = -77;
  }
  CHECK(device.matrix_offsets.upload(poisoned_offsets) == cudaSuccess);
  CHECK(reset_gfn2_scc_energy_device_errors_cuda(device.batch.batch_size,
                                                 device.system_errors.get(),
                                                 device.device_error.get()) == cudaSuccess);
  CHECK(evaluate_gfn2_scc_electronic_energy_cuda(
            device.batch, device.density.get(), device.h0.get(), device.entropies.get(),
            host.temperature, canonical_activity(device), device.core.get(), device.free.get(),
            device.workspace, device.system_errors.get(),
            device.device_error.get()) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  std::vector<double> core;
  std::vector<double> free;
  std::vector<std::uint32_t> errors;
  std::vector<std::uint32_t> device_error;
  std::vector<std::uint32_t> sequence;
  CHECK(device.core.download(core) == cudaSuccess);
  CHECK(device.free.download(free) == cudaSuccess);
  CHECK(device.system_errors.download(errors) == cudaSuccess);
  CHECK(device.device_error.download(device_error) == cudaSuccess);
  CHECK(device.sequence_active.download(sequence) == cudaSuccess);
  for (std::size_t system = 0; system < 8u; ++system) {
    if (system < 4u) {
      CHECK(near(core[system], host.expected_core[system]));
      CHECK(near(free[system], host.expected_free[system]));
    } else {
      CHECK(core[system] == sentinel && free[system] == sentinel);
    }
    CHECK(errors[system] == 0u);
  }
  CHECK(device_error[0] == 0u && sequence[0] == 1u);

  /* Sequence closure dominates deliberately invalid active topology. */
  CHECK(device.canonical_sequence.upload(std::vector<std::uint32_t>{0u}) == cudaSuccess);
  poisoned_offsets[0] = -1;
  CHECK(device.matrix_offsets.upload(poisoned_offsets) == cudaSuccess);
  CHECK(device.core.upload(std::vector<double>(8u, sentinel)) == cudaSuccess);
  CHECK(reset_gfn2_scc_energy_device_errors_cuda(device.batch.batch_size,
                                                 device.system_errors.get(),
                                                 device.device_error.get()) == cudaSuccess);
  CHECK(evaluate_gfn2_scc_electronic_energy_cuda(
            device.batch, device.density.get(), device.h0.get(), device.entropies.get(),
            host.temperature, canonical_activity(device), device.core.get(), device.free.get(),
            device.workspace, device.system_errors.get(),
            device.device_error.get()) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(device.core.download(core) == cudaSuccess);
  CHECK(device.device_error.download(device_error) == cudaSuccess);
  CHECK(device.sequence_active.download(sequence) == cudaSuccess);
  CHECK(std::all_of(core.begin(), core.end(),
                    [sentinel](double value) { return value == sentinel; }));
  CHECK(device_error[0] == 0u && sequence[0] == 0u);
  return 0;
}

}  // namespace

int main() {
  const std::array<int (*)(), 9> tests{
      {test_batch_parity_and_custom_stream, test_all_empty_batch,
       test_peer_isolation_and_inactive_skip, test_reduction_and_addition_overflow,
       test_first_error_diagnostics_are_consistent, test_offset_active_and_sticky_fail_closed,
       test_host_validation_and_aliases, test_cuda_graph_replay,
       test_canonical_activity_and_poisoned_inactive_topology}};
  for (const auto test : tests) {
    const int line = test();
    if (line != 0) {
      std::fprintf(stderr, "CUDA SCC energy test failed at line %d\n", line);
      return 1;
    }
  }
  std::puts("CUDA SCC electronic energy tests passed");
  return 0;
}
