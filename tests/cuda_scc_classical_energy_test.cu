#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_scc_classical_energy.cuh"

namespace {

using xtbloom::detail::cuda::evaluate_gfn2_scc_classical_energy_cuda;
using xtbloom::detail::cuda::Gfn2SccClassicalEnergyComponent;
using xtbloom::detail::cuda::Gfn2SccClassicalEnergyDeviceActivity;
using xtbloom::detail::cuda::Gfn2SccClassicalEnergyDeviceBatch;
using xtbloom::detail::cuda::Gfn2SccClassicalEnergyDeviceDiagnostics;
using xtbloom::detail::cuda::Gfn2SccClassicalEnergyDeviceError;
using xtbloom::detail::cuda::Gfn2SccClassicalEnergyDeviceInput;
using xtbloom::detail::cuda::Gfn2SccClassicalEnergyDeviceWorkspace;
using xtbloom::detail::cuda::kGfn2SccClassicalAllComponents;
using xtbloom::detail::cuda::kGfn2SccClassicalDiagnosticComponents;
using xtbloom::detail::cuda::kGfn2SccClassicalInputComponents;
using xtbloom::detail::cuda::reset_gfn2_scc_classical_energy_device_errors_cuda;

#define CHECK(condition)                                                                   \
  do {                                                                                     \
    if (!(condition)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      return __LINE__;                                                                     \
    }                                                                                      \
  } while (false)

constexpr std::uint64_t kPlanToken = 0x434c415353454e45ULL;

constexpr std::uint32_t bit(Gfn2SccClassicalEnergyComponent component) {
  return static_cast<std::uint32_t>(component);
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t elements) { require(allocate(elements)); }
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

  cudaError_t upload(const std::vector<T>& values, cudaStream_t stream = nullptr) {
    if (values.size() != elements_) {
      return cudaErrorInvalidValue;
    }
    return elements_ == 0u ? cudaSuccess
                           : cudaMemcpyAsync(pointer_, values.data(), elements_ * sizeof(T),
                                             cudaMemcpyHostToDevice, stream);
  }

  cudaError_t download(std::vector<T>& values, cudaStream_t stream = nullptr) const {
    values.resize(elements_);
    return elements_ == 0u ? cudaSuccess
                           : cudaMemcpyAsync(values.data(), pointer_, elements_ * sizeof(T),
                                             cudaMemcpyDeviceToHost, stream);
  }

  T* get() const { return pointer_; }
  std::size_t size() const { return elements_; }

 private:
  static void require(cudaError_t status) {
    if (status != cudaSuccess) {
      std::fprintf(stderr, "CUDA fixture allocation failed: %s\n", cudaGetErrorString(status));
      std::abort();
    }
  }

  T* pointer_ = nullptr;
  std::size_t elements_ = 0u;
};

struct HostCase {
  std::array<std::vector<double>, kGfn2SccClassicalInputComponents> components;
  std::vector<std::uint8_t> active;
  std::vector<double> total;
};

HostCase make_case(std::size_t batch_size) {
  HostCase data;
  for (auto& component : data.components) {
    component.resize(batch_size);
  }
  data.active.assign(batch_size, 1u);
  data.total.resize(batch_size, 0.0);
  for (std::size_t system = 0; system < batch_size; ++system) {
    const double scale = 0.001 * static_cast<double>(1u + (system * 17u) % 31u);
    data.components[0][system] = 11.0 * scale;
    data.components[1][system] = -7.0 * scale;
    data.components[2][system] = 5.0 * scale;
    data.components[3][system] = -3.0 * scale;
    data.components[4][system] = 13.0 * scale;
    data.components[5][system] = -2.0 * scale;
    double total = 0.0;
    for (const auto& component : data.components) {
      total += component[system];
    }
    data.total[system] = total;
  }
  return data;
}

struct DeviceCase {
  std::array<DeviceBuffer<double>, kGfn2SccClassicalInputComponents> inputs;
  DeviceBuffer<std::uint8_t> active;
  std::array<DeviceBuffer<double>, kGfn2SccClassicalDiagnosticComponents> outputs;
  DeviceBuffer<double> scratch;
  DeviceBuffer<std::uint32_t> sequence_active;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;

  Gfn2SccClassicalEnergyDeviceBatch batch;
  Gfn2SccClassicalEnergyDeviceInput input;
  Gfn2SccClassicalEnergyDeviceActivity activity;
  Gfn2SccClassicalEnergyDeviceDiagnostics diagnostics;
  Gfn2SccClassicalEnergyDeviceWorkspace workspace;

  explicit DeviceCase(const HostCase& host)
      : active(host.active.size()),
        scratch(host.active.size() *
                static_cast<std::size_t>(kGfn2SccClassicalDiagnosticComponents)),
        sequence_active(1u),
        system_errors(host.active.size()),
        device_error(1u) {
    for (std::size_t component = 0; component < inputs.size(); ++component) {
      require(inputs[component].allocate(host.active.size()));
      require(inputs[component].upload(host.components[component]));
    }
    for (auto& output : outputs) {
      require(output.allocate(host.active.size()));
    }
    require(active.upload(host.active));
    bind(kGfn2SccClassicalAllComponents, true);
    require(cudaDeviceSynchronize());
  }

  void bind(std::uint32_t enabled_components, bool use_activity) {
    const std::int64_t count = static_cast<std::int64_t>(active.size());
    batch = {count, enabled_components, kPlanToken};
    input = {
        enabled_components & bit(Gfn2SccClassicalEnergyComponent::kES2) ? inputs[0].get() : nullptr,
        enabled_components & bit(Gfn2SccClassicalEnergyComponent::kES2) ? count : 0,
        enabled_components & bit(Gfn2SccClassicalEnergyComponent::kES3) ? inputs[1].get() : nullptr,
        enabled_components & bit(Gfn2SccClassicalEnergyComponent::kES3) ? count : 0,
        enabled_components & bit(Gfn2SccClassicalEnergyComponent::kAES2) ? inputs[2].get()
                                                                         : nullptr,
        enabled_components & bit(Gfn2SccClassicalEnergyComponent::kAES2) ? count : 0,
        enabled_components & bit(Gfn2SccClassicalEnergyComponent::kD4TwoBody) ? inputs[3].get()
                                                                              : nullptr,
        enabled_components & bit(Gfn2SccClassicalEnergyComponent::kD4TwoBody) ? count : 0,
        enabled_components & bit(Gfn2SccClassicalEnergyComponent::kExplicitPointCharge)
            ? inputs[4].get()
            : nullptr,
        enabled_components & bit(Gfn2SccClassicalEnergyComponent::kExplicitPointCharge) ? count : 0,
        enabled_components & bit(Gfn2SccClassicalEnergyComponent::kPeriodicEmbedding)
            ? inputs[5].get()
            : nullptr,
        enabled_components & bit(Gfn2SccClassicalEnergyComponent::kPeriodicEmbedding) ? count : 0,
        kPlanToken};
    activity = {use_activity ? active.get() : nullptr, use_activity ? count : 0, kPlanToken};
    diagnostics = {outputs[0].get(), count, outputs[1].get(), count, outputs[2].get(), count,
                   outputs[3].get(), count, outputs[4].get(), count, outputs[5].get(), count,
                   outputs[6].get(), count, kPlanToken};
    workspace = {scratch.get(), static_cast<std::int64_t>(scratch.size()), sequence_active.get(), 1,
                 kPlanToken};
  }

  cudaError_t fill_outputs(double value, cudaStream_t stream = nullptr) {
    const std::vector<double> values(active.size(), value);
    for (auto& output : outputs) {
      const cudaError_t status = output.upload(values, stream);
      if (status != cudaSuccess) {
        return status;
      }
    }
    return cudaSuccess;
  }

 private:
  static void require(cudaError_t status) {
    if (status != cudaSuccess) {
      std::fprintf(stderr, "CUDA fixture setup failed: %s\n", cudaGetErrorString(status));
      std::abort();
    }
  }
};

bool near(double actual, double expected, double tolerance = 2.0e-15) {
  return std::abs(actual - expected) <=
         tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

int launch(DeviceCase& device, cudaStream_t stream = nullptr, bool reset = true) {
  if (reset) {
    CHECK(reset_gfn2_scc_classical_energy_device_errors_cuda(
              device.batch.batch_size, device.system_errors.get(), device.device_error.get(),
              stream) == cudaSuccess);
  }
  CHECK(evaluate_gfn2_scc_classical_energy_cuda(
            device.batch, device.input, device.activity, device.diagnostics, device.workspace,
            device.system_errors.get(), device.device_error.get(), stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  return 0;
}

int download_outputs(const DeviceCase& device,
                     std::array<std::vector<double>, kGfn2SccClassicalDiagnosticComponents>& values,
                     cudaStream_t stream = nullptr) {
  for (std::size_t component = 0; component < values.size(); ++component) {
    CHECK(device.outputs[component].download(values[component], stream) == cudaSuccess);
  }
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  return 0;
}

int test_batch_parity_and_custom_stream() {
  for (const std::size_t batch_size : std::array<std::size_t, 4>{{1u, 8u, 32u, 128u}}) {
    const HostCase host = make_case(batch_size);
    DeviceCase device(host);
    CHECK(device.fill_outputs(81.25) == cudaSuccess);
    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    CHECK(launch(device, stream) == 0);
    std::array<std::vector<double>, kGfn2SccClassicalDiagnosticComponents> output;
    CHECK(download_outputs(device, output, stream) == 0);
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);

    for (std::size_t system = 0; system < batch_size; ++system) {
      for (std::size_t component = 0; component < host.components.size(); ++component) {
        CHECK(output[component][system] == host.components[component][system]);
      }
      CHECK(near(output[6][system], host.total[system]));
    }
  }
  return 0;
}

int test_component_contract_and_disabled_publication() {
  HostCase host = make_case(1u);
  /* Distinct literals catch accidental one-half scaling or inclusion of ATM-like extras. */
  host.components[0][0] = 0.11;
  host.components[1][0] = -0.07;
  host.components[2][0] = 0.05;
  host.components[3][0] = -0.31;  // raw-q D4 two-body only
  host.components[4][0] = 1.25;   // sum(q_raw*Vpc), no one-half
  host.components[5][0] = -0.75;  // q*b + one-half*q*A*q, already reduced
  host.total[0] = 0.0;
  for (const auto& component : host.components) {
    host.total[0] += component[0];
  }
  DeviceCase device(host);
  CHECK(launch(device) == 0);
  std::array<std::vector<double>, kGfn2SccClassicalDiagnosticComponents> output;
  CHECK(download_outputs(device, output) == 0);
  CHECK(output[4][0] == 1.25);
  CHECK(output[5][0] == -0.75);
  CHECK(near(output[6][0], host.total[0]));

  const std::uint32_t mandatory = bit(Gfn2SccClassicalEnergyComponent::kES2) |
                                  bit(Gfn2SccClassicalEnergyComponent::kES3) |
                                  bit(Gfn2SccClassicalEnergyComponent::kAES2);
  device.bind(mandatory, false);
  CHECK(device.fill_outputs(-99.0) == cudaSuccess);
  CHECK(launch(device) == 0);
  CHECK(download_outputs(device, output) == 0);
  const double expected = host.components[0][0] + host.components[1][0] + host.components[2][0];
  CHECK(output[3][0] == 0.0 && output[4][0] == 0.0 && output[5][0] == 0.0);
  CHECK(near(output[6][0], expected));

  device.bind(0u, false);
  CHECK(device.fill_outputs(-101.0) == cudaSuccess);
  CHECK(launch(device) == 0);
  CHECK(download_outputs(device, output) == 0);
  for (const auto& component : output) {
    CHECK(component[0] == 0.0);
  }
  return 0;
}

int test_nonfinite_peer_isolation_and_inactive_skip() {
  HostCase host = make_case(8u);
  const std::array<Gfn2SccClassicalEnergyDeviceError, 6> expected_errors{
      Gfn2SccClassicalEnergyDeviceError::kNonfiniteES2,
      Gfn2SccClassicalEnergyDeviceError::kNonfiniteES3,
      Gfn2SccClassicalEnergyDeviceError::kNonfiniteAES2,
      Gfn2SccClassicalEnergyDeviceError::kNonfiniteD4TwoBody,
      Gfn2SccClassicalEnergyDeviceError::kNonfiniteExplicitPointCharge,
      Gfn2SccClassicalEnergyDeviceError::kNonfinitePeriodicEmbedding};
  for (std::size_t component = 0; component < expected_errors.size(); ++component) {
    host.components[component][component] = component % 2u == 0u
                                                ? std::numeric_limits<double>::quiet_NaN()
                                                : std::numeric_limits<double>::infinity();
  }
  host.active[6] = 0u;
  for (auto& component : host.components) {
    component[6] = std::numeric_limits<double>::quiet_NaN();
  }
  host.active[7] = 2u;

  DeviceCase device(host);
  constexpr double sentinel = -713.25;
  CHECK(device.fill_outputs(sentinel) == cudaSuccess);
  CHECK(launch(device) == 0);
  std::array<std::vector<double>, kGfn2SccClassicalDiagnosticComponents> output;
  CHECK(download_outputs(device, output) == 0);
  std::vector<std::uint32_t> errors;
  CHECK(device.system_errors.download(errors) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  for (std::size_t system = 0; system < 6u; ++system) {
    CHECK(errors[system] == static_cast<std::uint32_t>(expected_errors[system]));
    for (const auto& component : output) {
      CHECK(component[system] == sentinel);
    }
  }
  CHECK(errors[6] == static_cast<std::uint32_t>(Gfn2SccClassicalEnergyDeviceError::kSuccess));
  CHECK(errors[7] ==
        static_cast<std::uint32_t>(Gfn2SccClassicalEnergyDeviceError::kInvalidActiveMask));
  for (const auto& component : output) {
    CHECK(component[6] == sentinel && component[7] == sentinel);
  }
  return 0;
}

int test_total_addition_overflow_and_sticky_sequence() {
  HostCase host = make_case(2u);
  host.components[0][0] = 0.75 * std::numeric_limits<double>::max();
  host.components[1][0] = 0.75 * std::numeric_limits<double>::max();
  for (std::size_t component = 2; component < host.components.size(); ++component) {
    host.components[component][0] = 0.0;
  }
  DeviceCase device(host);
  constexpr double sentinel = 917.5;
  CHECK(device.fill_outputs(sentinel) == cudaSuccess);
  CHECK(launch(device) == 0);
  std::array<std::vector<double>, kGfn2SccClassicalDiagnosticComponents> output;
  CHECK(download_outputs(device, output) == 0);
  std::vector<std::uint32_t> errors;
  CHECK(device.system_errors.download(errors) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(errors[0] ==
        static_cast<std::uint32_t>(Gfn2SccClassicalEnergyDeviceError::kNonfiniteTotalArithmetic));
  for (const auto& component : output) {
    CHECK(component[0] == sentinel);
  }
  for (std::size_t component = 0; component < host.components.size(); ++component) {
    CHECK(output[component][1] == host.components[component][1]);
  }
  CHECK(near(output[6][1], host.total[1]));

  const std::uint32_t sticky = 0x51u;
  CHECK(cudaMemcpy(device.device_error.get(), &sticky, sizeof(sticky), cudaMemcpyHostToDevice) ==
        cudaSuccess);
  CHECK(device.fill_outputs(sentinel) == cudaSuccess);
  CHECK(launch(device, nullptr, false) == 0);
  CHECK(download_outputs(device, output) == 0);
  for (const auto& component : output) {
    CHECK(std::all_of(component.begin(), component.end(),
                      [sentinel](double value) { return value == sentinel; }));
  }
  return 0;
}

int test_hostile_metadata_aliases_and_reset_validation() {
  const HostCase host = make_case(2u);
  DeviceCase device(host);
  const auto valid = [&]() {
    return evaluate_gfn2_scc_classical_energy_cuda(
        device.batch, device.input, device.activity, device.diagnostics, device.workspace,
        device.system_errors.get(), device.device_error.get());
  };
  CHECK(reset_gfn2_scc_classical_energy_device_errors_cuda(
            device.batch.batch_size, device.system_errors.get(), device.device_error.get()) ==
        cudaSuccess);
  CHECK(valid() == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);

  Gfn2SccClassicalEnergyDeviceBatch bad_batch = device.batch;
  bad_batch.enabled_components = kGfn2SccClassicalAllComponents | (1u << 17u);
  CHECK(evaluate_gfn2_scc_classical_energy_cuda(
            bad_batch, device.input, device.activity, device.diagnostics, device.workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_batch = device.batch;
  bad_batch.batch_size = std::numeric_limits<std::int64_t>::max();
  CHECK(evaluate_gfn2_scc_classical_energy_cuda(
            bad_batch, device.input, device.activity, device.diagnostics, device.workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);

  Gfn2SccClassicalEnergyDeviceInput bad_input = device.input;
  bad_input.es2_elements -= 1;
  CHECK(evaluate_gfn2_scc_classical_energy_cuda(
            device.batch, bad_input, device.activity, device.diagnostics, device.workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_input = device.input;
  bad_input.es2 = reinterpret_cast<const double*>(
      reinterpret_cast<const unsigned char*>(device.inputs[0].get()) + 1u);
  CHECK(evaluate_gfn2_scc_classical_energy_cuda(
            device.batch, bad_input, device.activity, device.diagnostics, device.workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_input = device.input;
  bad_input.plan_token += 1u;
  CHECK(evaluate_gfn2_scc_classical_energy_cuda(
            device.batch, bad_input, device.activity, device.diagnostics, device.workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);

  Gfn2SccClassicalEnergyDeviceActivity bad_activity = device.activity;
  bad_activity.elements -= 1;
  CHECK(evaluate_gfn2_scc_classical_energy_cuda(
            device.batch, device.input, bad_activity, device.diagnostics, device.workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);

  Gfn2SccClassicalEnergyDeviceDiagnostics bad_diagnostics = device.diagnostics;
  bad_diagnostics.classical_total = bad_diagnostics.es2;
  CHECK(evaluate_gfn2_scc_classical_energy_cuda(
            device.batch, device.input, device.activity, bad_diagnostics, device.workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_diagnostics = device.diagnostics;
  bad_diagnostics.es2 = device.inputs[0].get();
  CHECK(evaluate_gfn2_scc_classical_energy_cuda(
            device.batch, device.input, device.activity, bad_diagnostics, device.workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_diagnostics = device.diagnostics;
  bad_diagnostics.aes2_elements -= 1;
  CHECK(evaluate_gfn2_scc_classical_energy_cuda(
            device.batch, device.input, device.activity, bad_diagnostics, device.workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);

  Gfn2SccClassicalEnergyDeviceWorkspace bad_workspace = device.workspace;
  bad_workspace.component_scratch = device.outputs[0].get();
  CHECK(evaluate_gfn2_scc_classical_energy_cuda(
            device.batch, device.input, device.activity, device.diagnostics, bad_workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_workspace = device.workspace;
  bad_workspace.component_elements -= 1;
  CHECK(evaluate_gfn2_scc_classical_energy_cuda(
            device.batch, device.input, device.activity, device.diagnostics, bad_workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);

  device.bind(bit(Gfn2SccClassicalEnergyComponent::kES2), true);
  bad_input = device.input;
  bad_input.es3 = device.inputs[1].get();
  CHECK(evaluate_gfn2_scc_classical_energy_cuda(
            device.batch, bad_input, device.activity, device.diagnostics, device.workspace,
            device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  device.bind(kGfn2SccClassicalAllComponents, true);

  CHECK(reset_gfn2_scc_classical_energy_device_errors_cuda(
            device.batch.batch_size, device.system_errors.get(), device.system_errors.get()) ==
        cudaErrorInvalidValue);
  CHECK(reset_gfn2_scc_classical_energy_device_errors_cuda(
            0, device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  return 0;
}

int test_cuda_graph_replay() {
  const HostCase host = make_case(32u);
  DeviceCase device(host);
  CHECK(device.fill_outputs(-41.0) == cudaSuccess);
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess);
  CHECK(reset_gfn2_scc_classical_energy_device_errors_cuda(
            device.batch.batch_size, device.system_errors.get(), device.device_error.get(),
            stream) == cudaSuccess);
  CHECK(evaluate_gfn2_scc_classical_energy_cuda(
            device.batch, device.input, device.activity, device.diagnostics, device.workspace,
            device.system_errors.get(), device.device_error.get(), stream) == cudaSuccess);
  CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
  CHECK(cudaGraphInstantiate(&executable, graph, 0) == cudaSuccess);
  CHECK(cudaGraphLaunch(executable, stream) == cudaSuccess);
  CHECK(cudaGraphLaunch(executable, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);

  std::array<std::vector<double>, kGfn2SccClassicalDiagnosticComponents> output;
  CHECK(download_outputs(device, output, stream) == 0);
  for (std::size_t system = 0; system < host.active.size(); ++system) {
    CHECK(near(output[6][system], host.total[system]));
  }
  CHECK(cudaGraphExecDestroy(executable) == cudaSuccess);
  CHECK(cudaGraphDestroy(graph) == cudaSuccess);
  CHECK(cudaStreamDestroy(stream) == cudaSuccess);
  return 0;
}

}  // namespace

int main() {
  const std::array<int (*)(), 6> tests{
      {test_batch_parity_and_custom_stream, test_component_contract_and_disabled_publication,
       test_nonfinite_peer_isolation_and_inactive_skip,
       test_total_addition_overflow_and_sticky_sequence,
       test_hostile_metadata_aliases_and_reset_validation, test_cuda_graph_replay}};
  for (const auto test : tests) {
    const int status = test();
    if (status != 0) {
      return status;
    }
  }
  return 0;
}
