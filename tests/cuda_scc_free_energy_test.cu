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
#include "backends/cuda/gfn2_scc_energy.cuh"
#include "backends/cuda/gfn2_scc_free_energy.cuh"

namespace {

using xtbloom::detail::cuda::compose_gfn2_scc_free_energy_cuda;
using xtbloom::detail::cuda::evaluate_gfn2_scc_classical_energy_cuda;
using xtbloom::detail::cuda::evaluate_gfn2_scc_electronic_energy_cuda;
using xtbloom::detail::cuda::Gfn2SccClassicalEnergyComponent;
using xtbloom::detail::cuda::Gfn2SccClassicalEnergyDeviceActivity;
using xtbloom::detail::cuda::Gfn2SccClassicalEnergyDeviceBatch;
using xtbloom::detail::cuda::Gfn2SccClassicalEnergyDeviceDiagnostics;
using xtbloom::detail::cuda::Gfn2SccClassicalEnergyDeviceInput;
using xtbloom::detail::cuda::Gfn2SccClassicalEnergyDeviceWorkspace;
using xtbloom::detail::cuda::Gfn2SccEnergyDeviceBatch;
using xtbloom::detail::cuda::Gfn2SccEnergyDeviceWorkspace;
using xtbloom::detail::cuda::Gfn2SccFreeEnergyDeviceActivity;
using xtbloom::detail::cuda::Gfn2SccFreeEnergyDeviceBatch;
using xtbloom::detail::cuda::Gfn2SccFreeEnergyDeviceDiagnostics;
using xtbloom::detail::cuda::Gfn2SccFreeEnergyDeviceError;
using xtbloom::detail::cuda::Gfn2SccFreeEnergyDeviceInput;
using xtbloom::detail::cuda::Gfn2SccFreeEnergyDeviceWorkspace;
using xtbloom::detail::cuda::kGfn2SccClassicalAllComponents;
using xtbloom::detail::cuda::kGfn2SccFreeEnergyDiagnosticComponents;
using xtbloom::detail::cuda::kGfn2SccFreeEnergyInputComponents;
using xtbloom::detail::cuda::kGfn2SccFreeEnergyStorageComponents;
using xtbloom::detail::cuda::reset_gfn2_scc_classical_energy_device_errors_cuda;
using xtbloom::detail::cuda::reset_gfn2_scc_energy_device_errors_cuda;
using xtbloom::detail::cuda::reset_gfn2_scc_free_energy_device_errors_cuda;

#define CHECK(condition)                                                                   \
  do {                                                                                     \
    if (!(condition)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      return __LINE__;                                                                     \
    }                                                                                      \
  } while (false)

constexpr std::uint64_t kPlanToken = 0x46524545454e4552ULL;
constexpr double kSentinel = -913.625;

constexpr std::uint32_t bit(Gfn2SccClassicalEnergyComponent component) {
  return static_cast<std::uint32_t>(component);
}

double cpu_add(double lhs, double rhs) {
  /* Force one binary64 addition, matching add_finite in the CPU SCC driver. */
  volatile double result = lhs + rhs;
  return result;
}

double cpu_multiply(double lhs, double rhs) {
  /* Keep this product observably separate from the following addition so the
   * FMA regression test cannot be contracted by the host compiler. */
  volatile double result = lhs * rhs;
  return result;
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
      std::fprintf(stderr, "CUDA allocation failed: %s\n", cudaGetErrorString(status));
      std::abort();
    }
  }

  T* pointer_ = nullptr;
  std::size_t elements_ = 0u;
};

struct HostCase {
  std::array<std::vector<double>, kGfn2SccFreeEnergyInputComponents> inputs;
  std::array<std::vector<double>, kGfn2SccFreeEnergyDiagnosticComponents> expected;
  std::vector<std::uint8_t> active;
  std::uint32_t enabled_components = kGfn2SccClassicalAllComponents;
  double temperature = 0.019;

  void recompute() {
    const std::size_t batch_size = active.size();
    for (auto& values : expected) {
      values.assign(batch_size, 0.0);
    }
    for (std::size_t system = 0; system < batch_size; ++system) {
      const std::array<bool, 6> enabled{
          (enabled_components & bit(Gfn2SccClassicalEnergyComponent::kES2)) != 0u,
          (enabled_components & bit(Gfn2SccClassicalEnergyComponent::kES3)) != 0u,
          (enabled_components & bit(Gfn2SccClassicalEnergyComponent::kAES2)) != 0u,
          (enabled_components & bit(Gfn2SccClassicalEnergyComponent::kD4TwoBody)) != 0u,
          (enabled_components & bit(Gfn2SccClassicalEnergyComponent::kExplicitPointCharge)) != 0u,
          (enabled_components & bit(Gfn2SccClassicalEnergyComponent::kPeriodicEmbedding)) != 0u};
      expected[0][system] = inputs[0][system];
      expected[8][system] = inputs[1][system];
      double internal = inputs[0][system];
      constexpr std::array<std::size_t, 6> input_indices{{2u, 3u, 4u, 6u, 7u, 8u}};
      constexpr std::array<std::size_t, 6> output_indices{{1u, 2u, 3u, 5u, 6u, 7u}};
      for (std::size_t component = 0; component < enabled.size(); ++component) {
        const double value = enabled[component] ? inputs[input_indices[component]][system] : 0.0;
        expected[output_indices[component]][system] = value;
        internal = cpu_add(internal, value);
        if (component == 2u) {
          expected[4][system] = inputs[5][system];
          internal = cpu_add(internal, inputs[5][system]);
        }
      }
      expected[9][system] = internal;
      expected[10][system] = std::fma(-temperature, inputs[1][system], internal);
    }
  }
};

HostCase make_case(std::size_t batch_size) {
  HostCase host;
  for (auto& values : host.inputs) {
    values.resize(batch_size);
  }
  host.active.assign(batch_size, 1u);
  for (std::size_t system = 0; system < batch_size; ++system) {
    const double scale = 0.0005 * static_cast<double>(1u + (system * 23u) % 37u);
    host.inputs[0][system] = -1.75 + 3.0 * scale;
    host.inputs[1][system] = 0.25 + 2.0 * scale;
    host.inputs[2][system] = 11.0 * scale;
    host.inputs[3][system] = -7.0 * scale;
    host.inputs[4][system] = 5.0 * scale;
    host.inputs[5][system] = -3.0 * scale;
    host.inputs[6][system] = 13.0 * scale;
    host.inputs[7][system] = -2.0 * scale;
    host.inputs[8][system] = 17.0 * scale;
  }
  host.recompute();
  return host;
}

struct DeviceCase {
  std::array<DeviceBuffer<double>, kGfn2SccFreeEnergyInputComponents> inputs;
  DeviceBuffer<std::uint8_t> active;
  std::array<DeviceBuffer<double>, kGfn2SccFreeEnergyDiagnosticComponents> outputs;
  DeviceBuffer<double> scratch;
  DeviceBuffer<std::uint32_t> sequence_active;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;

  Gfn2SccFreeEnergyDeviceBatch batch;
  Gfn2SccFreeEnergyDeviceInput input;
  Gfn2SccFreeEnergyDeviceActivity activity;
  Gfn2SccFreeEnergyDeviceDiagnostics diagnostics;
  Gfn2SccFreeEnergyDeviceWorkspace workspace;

  explicit DeviceCase(const HostCase& host)
      : active(host.active.size()),
        scratch(host.active.size() * static_cast<std::size_t>(kGfn2SccFreeEnergyStorageComponents)),
        sequence_active(1u),
        system_errors(host.active.size()),
        device_error(1u) {
    for (std::size_t component = 0; component < inputs.size(); ++component) {
      require(inputs[component].allocate(host.active.size()));
      require(inputs[component].upload(host.inputs[component]));
    }
    for (auto& output : outputs) {
      require(output.allocate(host.active.size()));
    }
    require(active.upload(host.active));
    bind(host.enabled_components, host.temperature, true);
    require(cudaDeviceSynchronize());
  }

  void bind(std::uint32_t enabled_components, double temperature, bool use_activity) {
    const std::int64_t count = static_cast<std::int64_t>(active.size());
    const auto enabled = [enabled_components](Gfn2SccClassicalEnergyComponent component) {
      return (enabled_components & bit(component)) != 0u;
    };
    batch = {count, enabled_components, temperature, kPlanToken};
    input = {};
    input.core = inputs[0].get();
    input.core_elements = count;
    input.entropy = inputs[1].get();
    input.entropy_elements = count;
    input.es2 = enabled(Gfn2SccClassicalEnergyComponent::kES2) ? inputs[2].get() : nullptr;
    input.es2_elements = enabled(Gfn2SccClassicalEnergyComponent::kES2) ? count : 0;
    input.es3 = enabled(Gfn2SccClassicalEnergyComponent::kES3) ? inputs[3].get() : nullptr;
    input.es3_elements = enabled(Gfn2SccClassicalEnergyComponent::kES3) ? count : 0;
    input.aes2 = enabled(Gfn2SccClassicalEnergyComponent::kAES2) ? inputs[4].get() : nullptr;
    input.aes2_elements = enabled(Gfn2SccClassicalEnergyComponent::kAES2) ? count : 0;
    input.spin = inputs[5].get();
    input.spin_elements = count;
    input.d4_two_body =
        enabled(Gfn2SccClassicalEnergyComponent::kD4TwoBody) ? inputs[6].get() : nullptr;
    input.d4_two_body_elements = enabled(Gfn2SccClassicalEnergyComponent::kD4TwoBody) ? count : 0;
    input.explicit_point_charge =
        enabled(Gfn2SccClassicalEnergyComponent::kExplicitPointCharge) ? inputs[7].get() : nullptr;
    input.explicit_point_charge_elements =
        enabled(Gfn2SccClassicalEnergyComponent::kExplicitPointCharge) ? count : 0;
    input.periodic_embedding =
        enabled(Gfn2SccClassicalEnergyComponent::kPeriodicEmbedding) ? inputs[8].get() : nullptr;
    input.periodic_embedding_elements =
        enabled(Gfn2SccClassicalEnergyComponent::kPeriodicEmbedding) ? count : 0;
    input.plan_token = kPlanToken;
    activity = {use_activity ? active.get() : nullptr, use_activity ? count : 0, kPlanToken};
    diagnostics = {};
    diagnostics.core = outputs[0].get();
    diagnostics.core_elements = count;
    diagnostics.es2 = outputs[1].get();
    diagnostics.es2_elements = count;
    diagnostics.es3 = outputs[2].get();
    diagnostics.es3_elements = count;
    diagnostics.aes2 = outputs[3].get();
    diagnostics.aes2_elements = count;
    diagnostics.spin = outputs[4].get();
    diagnostics.spin_elements = count;
    diagnostics.d4_two_body = outputs[5].get();
    diagnostics.d4_two_body_elements = count;
    diagnostics.explicit_point_charge = outputs[6].get();
    diagnostics.explicit_point_charge_elements = count;
    diagnostics.periodic_embedding = outputs[7].get();
    diagnostics.periodic_embedding_elements = count;
    diagnostics.entropy = outputs[8].get();
    diagnostics.entropy_elements = count;
    diagnostics.internal_energy = outputs[9].get();
    diagnostics.internal_energy_elements = count;
    diagnostics.free_energy = outputs[10].get();
    diagnostics.free_energy_elements = count;
    diagnostics.plan_token = kPlanToken;
    workspace = {scratch.get(), count * kGfn2SccFreeEnergyDiagnosticComponents,
                 sequence_active.get(), 1, kPlanToken};
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

int launch(DeviceCase& device, cudaStream_t stream = nullptr, bool reset = true) {
  if (reset) {
    CHECK(reset_gfn2_scc_free_energy_device_errors_cuda(
              device.batch.batch_size, device.system_errors.get(), device.device_error.get(),
              stream) == cudaSuccess);
  }
  CHECK(compose_gfn2_scc_free_energy_cuda(
            device.batch, device.input, device.activity, device.diagnostics, device.workspace,
            device.system_errors.get(), device.device_error.get(), stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  return 0;
}

int download_outputs(
    const DeviceCase& device,
    std::array<std::vector<double>, kGfn2SccFreeEnergyDiagnosticComponents>& output,
    cudaStream_t stream = nullptr) {
  for (std::size_t component = 0; component < output.size(); ++component) {
    CHECK(device.outputs[component].download(output[component], stream) == cudaSuccess);
  }
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  return 0;
}

int test_batch_parity_custom_stream_and_disabled_terms() {
  for (const std::size_t batch_size : std::array<std::size_t, 4>{{1u, 8u, 32u, 128u}}) {
    HostCase host = make_case(batch_size);
    DeviceCase device(host);
    CHECK(device.fill_outputs(kSentinel) == cudaSuccess);
    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    CHECK(launch(device, stream) == 0);
    std::array<std::vector<double>, kGfn2SccFreeEnergyDiagnosticComponents> output;
    CHECK(download_outputs(device, output, stream) == 0);
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    for (std::size_t component = 0; component < output.size(); ++component) {
      CHECK(output[component] == host.expected[component]);
    }
  }

  HostCase disabled = make_case(1u);
  disabled.enabled_components =
      bit(Gfn2SccClassicalEnergyComponent::kES2) | bit(Gfn2SccClassicalEnergyComponent::kAES2);
  disabled.recompute();
  DeviceCase device(disabled);
  device.bind(disabled.enabled_components, disabled.temperature, false);
  CHECK(device.fill_outputs(kSentinel) == cudaSuccess);
  CHECK(launch(device) == 0);
  std::array<std::vector<double>, kGfn2SccFreeEnergyDiagnosticComponents> output;
  CHECK(download_outputs(device, output) == 0);
  for (std::size_t component = 0; component < output.size(); ++component) {
    CHECK(output[component] == disabled.expected[component]);
  }
  CHECK(output[2][0] == 0.0 && output[5][0] == 0.0 && output[6][0] == 0.0 && output[7][0] == 0.0);
  CHECK(output[4][0] == disabled.inputs[5][0]);
  return 0;
}

int test_exact_cpu_association_and_subtotal_counterexamples() {
  HostCase host = make_case(2u);
  host.temperature = 1.0;
  for (auto& input : host.inputs) {
    std::fill(input.begin(), input.end(), 0.0);
  }

  host.inputs[0][0] = 1.0e16;
  host.inputs[2][0] = -1.0e16;
  host.inputs[3][0] = 1.0;
  host.inputs[0][1] = 1.0e16;
  host.inputs[1][1] = 1.0;
  host.inputs[2][1] = -1.0e16;
  host.inputs[3][1] = 2.0;
  host.recompute();

  const double reordered =
      cpu_add(cpu_add(host.inputs[0][0], host.inputs[3][0]), host.inputs[2][0]);
  double classical_total = 0.0;
  for (std::size_t component = 2; component < host.inputs.size(); ++component) {
    classical_total = cpu_add(classical_total, host.inputs[component][0]);
  }
  const double subtotal_internal = cpu_add(host.inputs[0][0], classical_total);
  CHECK(host.expected[9][0] == 1.0);
  CHECK(reordered == 0.0 && subtotal_internal == 0.0);

  double second_classical_total = 0.0;
  for (std::size_t component = 2; component < host.inputs.size(); ++component) {
    second_classical_total = cpu_add(second_classical_total, host.inputs[component][1]);
  }
  const double electronic_free = std::fma(-host.temperature, host.inputs[1][1], host.inputs[0][1]);
  const double tempting_shortcut = cpu_add(electronic_free, second_classical_total);
  CHECK(host.expected[9][1] == 2.0 && host.expected[10][1] == 1.0);
  CHECK(tempting_shortcut == 2.0 && tempting_shortcut != host.expected[10][1]);

  DeviceCase device(host);
  CHECK(launch(device) == 0);
  std::array<std::vector<double>, kGfn2SccFreeEnergyDiagnosticComponents> output;
  CHECK(download_outputs(device, output) == 0);
  CHECK(output[9][0] == 1.0);
  CHECK(output[9][1] == 2.0);
  CHECK(output[10][1] == 1.0);
  return 0;
}

int test_final_fma_rounding_counterexample() {
  HostCase host = make_case(1u);
  for (auto& input : host.inputs) {
    input[0] = 0.0;
  }

  /* These exact binary64 values make the required final FMA differ by one ULP
   * from an implementation that rounds -temperature*entropy before adding the
   * internal energy. Keep hexadecimal literals so the counterexample does not
   * depend on decimal parsing or host extended precision. */
  host.temperature = 0x1.7bd44722789bbp+0;
  host.inputs[0][0] = 0x1.45ee763914c7ep+4;
  host.inputs[1][0] = 0x1.133944b58c733p+5;
  host.recompute();

  const double unfused_product = cpu_multiply(-host.temperature, host.inputs[1][0]);
  const double unfused_free_energy = cpu_add(unfused_product, host.inputs[0][0]);
  CHECK(host.expected[10][0] == -0x1.eac58b06286d3p+4);
  CHECK(unfused_free_energy == -0x1.eac58b06286d4p+4);
  CHECK(std::nextafter(host.expected[10][0], -std::numeric_limits<double>::infinity()) ==
        unfused_free_energy);

  DeviceCase device(host);
  CHECK(device.fill_outputs(kSentinel) == cudaSuccess);
  CHECK(launch(device) == 0);
  std::array<std::vector<double>, kGfn2SccFreeEnergyDiagnosticComponents> output;
  CHECK(download_outputs(device, output) == 0);
  CHECK(output[9][0] == host.inputs[0][0]);
  CHECK(output[10][0] == host.expected[10][0]);
  CHECK(output[10][0] != unfused_free_energy);
  return 0;
}

int test_electric_field_component_order_and_nonfinite_peer() {
  HostCase host = make_case(2u);
  DeviceCase device(host);
  const std::vector<double> field{0x1.8p-3, -0x1.4p-4};
  DeviceBuffer<double> device_field(field.size());
  DeviceBuffer<double> field_output(field.size());
  CHECK(device_field.upload(field) == cudaSuccess);
  CHECK(field_output.upload(std::vector<double>(2, kSentinel)) == cudaSuccess);
  device.input.electric_field = device_field.get();
  device.input.electric_field_elements = 2;
  device.diagnostics.electric_field = field_output.get();
  device.diagnostics.electric_field_elements = 2;
  device.workspace.diagnostic_elements =
      2 * xtbloom::detail::cuda::kGfn2SccFreeEnergyStorageComponents;
  CHECK(device.fill_outputs(kSentinel) == cudaSuccess);
  CHECK(launch(device) == 0);

  std::array<std::vector<double>, kGfn2SccFreeEnergyDiagnosticComponents> output;
  CHECK(download_outputs(device, output) == 0);
  std::vector<double> field_diagnostic;
  CHECK(field_output.download(field_diagnostic) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  for (std::size_t system = 0; system < 2u; ++system) {
    double internal = host.inputs[0][system];
    internal = cpu_add(internal, host.inputs[2][system]);
    internal = cpu_add(internal, host.inputs[3][system]);
    internal = cpu_add(internal, host.inputs[4][system]);
    internal = cpu_add(internal, host.inputs[5][system]);
    internal = cpu_add(internal, host.inputs[6][system]);
    internal = cpu_add(internal, host.inputs[7][system]);
    internal = cpu_add(internal, field[system]);
    internal = cpu_add(internal, host.inputs[8][system]);
    const double expected_free = std::fma(-host.temperature, host.inputs[1][system], internal);
    CHECK(field_diagnostic[system] == field[system]);
    CHECK(output[9][system] == internal);
    CHECK(output[10][system] == expected_free);
  }

  const std::vector<double> bad_field{field[0], std::numeric_limits<double>::quiet_NaN()};
  CHECK(device_field.upload(bad_field) == cudaSuccess);
  CHECK(device.fill_outputs(kSentinel) == cudaSuccess);
  CHECK(field_output.upload(std::vector<double>(2, kSentinel)) == cudaSuccess);
  CHECK(launch(device) == 0);
  std::vector<std::uint32_t> errors;
  CHECK(device.system_errors.download(errors) == cudaSuccess);
  CHECK(field_output.download(field_diagnostic) == cudaSuccess);
  CHECK(download_outputs(device, output) == 0);
  CHECK(errors[0] == 0u);
  CHECK(errors[1] ==
        static_cast<std::uint32_t>(Gfn2SccFreeEnergyDeviceError::kNonfiniteElectricField));
  CHECK(field_diagnostic[0] == field[0]);
  CHECK(field_diagnostic[1] == kSentinel && output[9][1] == kSentinel &&
        output[10][1] == kSentinel);
  return 0;
}

int test_existing_cuda_stage_output_binding() {
  constexpr std::size_t batch_size = 8u;
  HostCase host = make_case(batch_size);
  DeviceCase device(host);

  std::vector<std::int64_t> host_offsets(batch_size + 1u);
  std::vector<double> host_density(batch_size, 1.0);
  for (std::size_t system = 0; system <= batch_size; ++system) {
    host_offsets[system] = static_cast<std::int64_t>(system);
  }
  DeviceBuffer<std::int64_t> offsets(batch_size + 1u);
  DeviceBuffer<double> density(batch_size);
  DeviceBuffer<double> h0(batch_size);
  DeviceBuffer<double> electronic_free(batch_size);
  DeviceBuffer<double> electronic_core_scratch(batch_size);
  DeviceBuffer<double> electronic_free_scratch(batch_size);
  DeviceBuffer<std::uint32_t> electronic_sequence(1u);
  DeviceBuffer<std::uint32_t> electronic_system_errors(batch_size);
  DeviceBuffer<std::uint32_t> electronic_device_error(1u);
  CHECK(offsets.upload(host_offsets) == cudaSuccess);
  CHECK(density.upload(host_density) == cudaSuccess);
  CHECK(h0.upload(host.inputs[0]) == cudaSuccess);

  const Gfn2SccEnergyDeviceBatch electronic_batch{
      static_cast<std::int64_t>(batch_size), static_cast<std::int64_t>(batch_size),
      static_cast<std::int64_t>(batch_size + 1u), kPlanToken, offsets.get()};
  const Gfn2SccEnergyDeviceWorkspace electronic_workspace{electronic_core_scratch.get(),
                                                          electronic_free_scratch.get(),
                                                          electronic_sequence.get(),
                                                          static_cast<std::int64_t>(batch_size),
                                                          1,
                                                          kPlanToken};
  CHECK(reset_gfn2_scc_energy_device_errors_cuda(static_cast<std::int64_t>(batch_size),
                                                 electronic_system_errors.get(),
                                                 electronic_device_error.get()) == cudaSuccess);
  CHECK(evaluate_gfn2_scc_electronic_energy_cuda(
            electronic_batch, density.get(), h0.get(), device.inputs[1].get(), host.temperature,
            device.active.get(), device.inputs[0].get(), electronic_free.get(),
            electronic_workspace, electronic_system_errors.get(),
            electronic_device_error.get()) == cudaSuccess);

  std::array<DeviceBuffer<double>, 7> classical_outputs;
  for (auto& output : classical_outputs) {
    CHECK(output.allocate(batch_size) == cudaSuccess);
  }
  DeviceBuffer<double> classical_scratch(7u * batch_size);
  DeviceBuffer<std::uint32_t> classical_sequence(1u);
  DeviceBuffer<std::uint32_t> classical_system_errors(batch_size);
  DeviceBuffer<std::uint32_t> classical_device_error(1u);
  const std::int64_t count = static_cast<std::int64_t>(batch_size);
  const Gfn2SccClassicalEnergyDeviceBatch classical_batch{count, kGfn2SccClassicalAllComponents,
                                                          kPlanToken};
  const Gfn2SccClassicalEnergyDeviceInput classical_input{device.inputs[2].get(),
                                                          count,
                                                          device.inputs[3].get(),
                                                          count,
                                                          device.inputs[4].get(),
                                                          count,
                                                          device.inputs[6].get(),
                                                          count,
                                                          device.inputs[7].get(),
                                                          count,
                                                          device.inputs[8].get(),
                                                          count,
                                                          kPlanToken};
  const Gfn2SccClassicalEnergyDeviceActivity classical_activity{device.active.get(), count,
                                                                kPlanToken};
  const Gfn2SccClassicalEnergyDeviceDiagnostics classical_diagnostics{classical_outputs[0].get(),
                                                                      count,
                                                                      classical_outputs[1].get(),
                                                                      count,
                                                                      classical_outputs[2].get(),
                                                                      count,
                                                                      classical_outputs[3].get(),
                                                                      count,
                                                                      classical_outputs[4].get(),
                                                                      count,
                                                                      classical_outputs[5].get(),
                                                                      count,
                                                                      classical_outputs[6].get(),
                                                                      count,
                                                                      kPlanToken};
  const Gfn2SccClassicalEnergyDeviceWorkspace classical_workspace{
      classical_scratch.get(), static_cast<std::int64_t>(classical_scratch.size()),
      classical_sequence.get(), 1, kPlanToken};
  CHECK(reset_gfn2_scc_classical_energy_device_errors_cuda(
            count, classical_system_errors.get(), classical_device_error.get()) == cudaSuccess);
  CHECK(evaluate_gfn2_scc_classical_energy_cuda(classical_batch, classical_input,
                                                classical_activity, classical_diagnostics,
                                                classical_workspace, classical_system_errors.get(),
                                                classical_device_error.get()) == cudaSuccess);

  /* Bind the actual public #81 core and #75 component arrays without staging
   * them through the host or translating their plan identity. */
  device.input.es2 = classical_outputs[0].get();
  device.input.es3 = classical_outputs[1].get();
  device.input.aes2 = classical_outputs[2].get();
  device.input.d4_two_body = classical_outputs[3].get();
  device.input.explicit_point_charge = classical_outputs[4].get();
  device.input.periodic_embedding = classical_outputs[5].get();
  CHECK(device.fill_outputs(kSentinel) == cudaSuccess);
  CHECK(launch(device) == 0);

  std::array<std::vector<double>, kGfn2SccFreeEnergyDiagnosticComponents> output;
  CHECK(download_outputs(device, output) == 0);
  for (std::size_t component = 0; component < output.size(); ++component) {
    CHECK(output[component] == host.expected[component]);
  }
  return 0;
}

int test_nonfinite_peer_isolation_inactive_and_upstream_system_error() {
  HostCase host = make_case(13u);
  const std::array<Gfn2SccFreeEnergyDeviceError, kGfn2SccFreeEnergyInputComponents> errors{
      Gfn2SccFreeEnergyDeviceError::kNonfiniteCore,
      Gfn2SccFreeEnergyDeviceError::kNonfiniteEntropy,
      Gfn2SccFreeEnergyDeviceError::kNonfiniteES2,
      Gfn2SccFreeEnergyDeviceError::kNonfiniteES3,
      Gfn2SccFreeEnergyDeviceError::kNonfiniteAES2,
      Gfn2SccFreeEnergyDeviceError::kNonfiniteSpin,
      Gfn2SccFreeEnergyDeviceError::kNonfiniteD4TwoBody,
      Gfn2SccFreeEnergyDeviceError::kNonfiniteExplicitPointCharge,
      Gfn2SccFreeEnergyDeviceError::kNonfinitePeriodicEmbedding};
  for (std::size_t input = 0; input < host.inputs.size(); ++input) {
    host.inputs[input][input] = input % 2u == 0u ? std::numeric_limits<double>::quiet_NaN()
                                                 : std::numeric_limits<double>::infinity();
  }
  host.active[9] = 0u;
  for (auto& input : host.inputs) {
    input[9] = std::numeric_limits<double>::quiet_NaN();
  }
  host.active[10] = 2u;
  host.recompute();

  DeviceCase device(host);
  CHECK(device.fill_outputs(kSentinel) == cudaSuccess);
  CHECK(reset_gfn2_scc_free_energy_device_errors_cuda(device.batch.batch_size,
                                                      device.system_errors.get(),
                                                      device.device_error.get()) == cudaSuccess);
  std::vector<std::uint32_t> upstream_errors(host.active.size(), 0u);
  upstream_errors[11] = 0x77u;
  CHECK(cudaMemcpy(device.system_errors.get(), upstream_errors.data(),
                   upstream_errors.size() * sizeof(std::uint32_t),
                   cudaMemcpyHostToDevice) == cudaSuccess);
  CHECK(launch(device, nullptr, false) == 0);

  std::array<std::vector<double>, kGfn2SccFreeEnergyDiagnosticComponents> output;
  CHECK(download_outputs(device, output) == 0);
  std::vector<std::uint32_t> actual_errors;
  CHECK(device.system_errors.download(actual_errors) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  for (std::size_t system = 0; system < errors.size(); ++system) {
    if (actual_errors[system] != static_cast<std::uint32_t>(errors[system])) {
      std::fprintf(stderr, "free-energy system %zu: expected error %u, observed %u\n", system,
                   static_cast<unsigned int>(errors[system]),
                   static_cast<unsigned int>(actual_errors[system]));
    }
    CHECK(actual_errors[system] == static_cast<std::uint32_t>(errors[system]));
    for (const auto& values : output) {
      CHECK(values[system] == kSentinel);
    }
  }
  CHECK(actual_errors[9] == 0u);
  CHECK(actual_errors[10] ==
        static_cast<std::uint32_t>(Gfn2SccFreeEnergyDeviceError::kInvalidActiveMask));
  CHECK(actual_errors[11] == 0x77u);
  for (const auto& values : output) {
    CHECK(values[9] == kSentinel && values[10] == kSentinel && values[11] == kSentinel);
  }
  for (std::size_t component = 0; component < output.size(); ++component) {
    CHECK(output[component][12] == host.expected[component][12]);
  }
  return 0;
}

int test_intermediate_and_final_fma_overflow_and_sticky_plan_error() {
  HostCase host = make_case(3u);
  host.temperature = 1.0;
  for (std::size_t component = 2; component < host.inputs.size(); ++component) {
    host.inputs[component][0] = 0.0;
    host.inputs[component][1] = 0.0;
  }
  const double maximum = std::numeric_limits<double>::max();
  host.inputs[0][0] = 0.75 * maximum;
  host.inputs[2][0] = 0.75 * maximum;
  host.inputs[0][1] = -maximum;
  host.inputs[1][1] = maximum;
  host.recompute();

  DeviceCase device(host);
  CHECK(device.fill_outputs(kSentinel) == cudaSuccess);
  CHECK(launch(device) == 0);
  std::array<std::vector<double>, kGfn2SccFreeEnergyDiagnosticComponents> output;
  CHECK(download_outputs(device, output) == 0);
  std::vector<std::uint32_t> errors;
  CHECK(device.system_errors.download(errors) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(errors[0] ==
        static_cast<std::uint32_t>(Gfn2SccFreeEnergyDeviceError::kNonfiniteInternalArithmetic));
  CHECK(errors[1] ==
        static_cast<std::uint32_t>(Gfn2SccFreeEnergyDeviceError::kNonfiniteFreeEnergyArithmetic));
  for (const auto& values : output) {
    CHECK(values[0] == kSentinel && values[1] == kSentinel);
  }
  for (std::size_t component = 0; component < output.size(); ++component) {
    CHECK(output[component][2] == host.expected[component][2]);
  }

  const std::uint32_t sticky = 0x51u;
  CHECK(cudaMemset(device.system_errors.get(), 0, host.active.size() * sizeof(std::uint32_t)) ==
        cudaSuccess);
  CHECK(cudaMemcpy(device.device_error.get(), &sticky, sizeof(sticky), cudaMemcpyHostToDevice) ==
        cudaSuccess);
  CHECK(device.fill_outputs(kSentinel) == cudaSuccess);
  CHECK(launch(device, nullptr, false) == 0);
  CHECK(download_outputs(device, output) == 0);
  for (const auto& values : output) {
    CHECK(
        std::all_of(values.begin(), values.end(), [](double value) { return value == kSentinel; }));
  }
  return 0;
}

int test_hostile_metadata_alias_token_misalignment_and_reset() {
  HostCase host = make_case(2u);
  DeviceCase device(host);
  const auto evaluate = [&](const Gfn2SccFreeEnergyDeviceBatch& batch,
                            const Gfn2SccFreeEnergyDeviceInput& input,
                            const Gfn2SccFreeEnergyDeviceActivity& activity,
                            const Gfn2SccFreeEnergyDeviceDiagnostics& diagnostics,
                            const Gfn2SccFreeEnergyDeviceWorkspace& workspace,
                            std::uint32_t* system_errors, std::uint32_t* device_error) {
    return compose_gfn2_scc_free_energy_cuda(batch, input, activity, diagnostics, workspace,
                                             system_errors, device_error);
  };
  CHECK(reset_gfn2_scc_free_energy_device_errors_cuda(device.batch.batch_size,
                                                      device.system_errors.get(),
                                                      device.device_error.get()) == cudaSuccess);
  CHECK(evaluate(device.batch, device.input, device.activity, device.diagnostics, device.workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);

  Gfn2SccFreeEnergyDeviceBatch bad_batch = device.batch;
  bad_batch.enabled_components |= 1u << 20u;
  CHECK(evaluate(bad_batch, device.input, device.activity, device.diagnostics, device.workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_batch = device.batch;
  bad_batch.electronic_temperature = std::numeric_limits<double>::quiet_NaN();
  CHECK(evaluate(bad_batch, device.input, device.activity, device.diagnostics, device.workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_batch.electronic_temperature = -1.0;
  CHECK(evaluate(bad_batch, device.input, device.activity, device.diagnostics, device.workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_batch = device.batch;
  bad_batch.batch_size = std::numeric_limits<std::int64_t>::max();
  CHECK(evaluate(bad_batch, device.input, device.activity, device.diagnostics, device.workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_batch = device.batch;
  bad_batch.plan_token = 0u;
  CHECK(evaluate(bad_batch, device.input, device.activity, device.diagnostics, device.workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);

  Gfn2SccFreeEnergyDeviceInput bad_input = device.input;
  bad_input.core_elements -= 1;
  CHECK(evaluate(device.batch, bad_input, device.activity, device.diagnostics, device.workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_input = device.input;
  bad_input.core = reinterpret_cast<const double*>(
      reinterpret_cast<const unsigned char*>(device.inputs[0].get()) + 1u);
  CHECK(evaluate(device.batch, bad_input, device.activity, device.diagnostics, device.workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_input = device.input;
  bad_input.core = reinterpret_cast<const double*>(std::numeric_limits<std::uintptr_t>::max() -
                                                   sizeof(double) + 1u);
  CHECK(evaluate(device.batch, bad_input, device.activity, device.diagnostics, device.workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_input = device.input;
  bad_input.plan_token ^= 1u;
  CHECK(evaluate(device.batch, bad_input, device.activity, device.diagnostics, device.workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);

  device.bind(bit(Gfn2SccClassicalEnergyComponent::kES2), host.temperature, true);
  bad_input = device.input;
  bad_input.es3 = device.inputs[3].get();
  CHECK(evaluate(device.batch, bad_input, device.activity, device.diagnostics, device.workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  device.bind(kGfn2SccClassicalAllComponents, host.temperature, true);

  Gfn2SccFreeEnergyDeviceActivity bad_activity = device.activity;
  bad_activity.elements -= 1;
  CHECK(evaluate(device.batch, device.input, bad_activity, device.diagnostics, device.workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_activity = device.activity;
  bad_activity.active_mask = reinterpret_cast<const std::uint8_t*>(device.inputs[0].get()) + 1u;
  bad_activity.plan_token ^= 1u;
  CHECK(evaluate(device.batch, device.input, bad_activity, device.diagnostics, device.workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);

  Gfn2SccFreeEnergyDeviceDiagnostics bad_diagnostics = device.diagnostics;
  bad_diagnostics.free_energy = bad_diagnostics.internal_energy;
  CHECK(evaluate(device.batch, device.input, device.activity, bad_diagnostics, device.workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_diagnostics = device.diagnostics;
  bad_diagnostics.core = const_cast<double*>(device.input.core);
  CHECK(evaluate(device.batch, device.input, device.activity, bad_diagnostics, device.workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_diagnostics = device.diagnostics;
  bad_diagnostics.entropy_elements -= 1;
  CHECK(evaluate(device.batch, device.input, device.activity, bad_diagnostics, device.workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_diagnostics = device.diagnostics;
  bad_diagnostics.plan_token ^= 1u;
  CHECK(evaluate(device.batch, device.input, device.activity, bad_diagnostics, device.workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);

  Gfn2SccFreeEnergyDeviceWorkspace bad_workspace = device.workspace;
  bad_workspace.diagnostic_scratch = device.outputs[0].get();
  CHECK(evaluate(device.batch, device.input, device.activity, device.diagnostics, bad_workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_workspace = device.workspace;
  bad_workspace.diagnostic_elements -= 1;
  CHECK(evaluate(device.batch, device.input, device.activity, device.diagnostics, bad_workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_workspace = device.workspace;
  bad_workspace.diagnostic_elements = std::numeric_limits<std::int64_t>::max();
  CHECK(evaluate(device.batch, device.input, device.activity, device.diagnostics, bad_workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_workspace = device.workspace;
  bad_workspace.sequence_active = device.device_error.get();
  CHECK(evaluate(device.batch, device.input, device.activity, device.diagnostics, bad_workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  bad_workspace = device.workspace;
  bad_workspace.plan_token ^= 1u;
  CHECK(evaluate(device.batch, device.input, device.activity, device.diagnostics, bad_workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);

  auto* misaligned_error = reinterpret_cast<std::uint32_t*>(
      reinterpret_cast<unsigned char*>(device.system_errors.get()) + 1u);
  CHECK(evaluate(device.batch, device.input, device.activity, device.diagnostics, device.workspace,
                 misaligned_error, device.device_error.get()) == cudaErrorInvalidValue);
  CHECK(evaluate(device.batch, device.input, device.activity, device.diagnostics, device.workspace,
                 device.system_errors.get(), misaligned_error) == cudaErrorInvalidValue);

  /* Read-only inputs may alias: host validation must not reject this legal binding. */
  Gfn2SccFreeEnergyDeviceInput aliased_reads = device.input;
  aliased_reads.es2 = aliased_reads.core;
  CHECK(reset_gfn2_scc_free_energy_device_errors_cuda(device.batch.batch_size,
                                                      device.system_errors.get(),
                                                      device.device_error.get()) == cudaSuccess);
  CHECK(evaluate(device.batch, aliased_reads, device.activity, device.diagnostics, device.workspace,
                 device.system_errors.get(), device.device_error.get()) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);

  CHECK(reset_gfn2_scc_free_energy_device_errors_cuda(
            0, device.system_errors.get(), device.device_error.get()) == cudaErrorInvalidValue);
  CHECK(reset_gfn2_scc_free_energy_device_errors_cuda(
            std::numeric_limits<std::int64_t>::max(), device.system_errors.get(),
            device.device_error.get()) == cudaErrorInvalidValue);
  CHECK(reset_gfn2_scc_free_energy_device_errors_cuda(
            device.batch.batch_size, device.system_errors.get(), device.system_errors.get()) ==
        cudaErrorInvalidValue);
  CHECK(reset_gfn2_scc_free_energy_device_errors_cuda(device.batch.batch_size, misaligned_error,
                                                      device.device_error.get()) ==
        cudaErrorInvalidValue);
  return 0;
}

int test_staged_spin_entropy_exact_alias_and_overlap_rejection() {
  HostCase host = make_case(8u);
  DeviceCase device(host);

  /* The iteration composer keeps the staged occupation entropy as the
   * canonical storage for the free-energy entropy diagnostic. Exact aliasing
   * of that input/output pair is therefore both intentional and safe: the
   * composition kernel snapshots the input in unpublished scratch before the
   * publication kernel writes the identical value back. */
  device.diagnostics.spin = const_cast<double*>(device.input.spin);
  device.diagnostics.entropy = const_cast<double*>(device.input.entropy);
  CHECK(device.fill_outputs(kSentinel) == cudaSuccess);
  CHECK(launch(device) == 0);

  std::array<std::vector<double>, kGfn2SccFreeEnergyDiagnosticComponents> output;
  CHECK(download_outputs(device, output) == 0);
  for (std::size_t component = 0; component < output.size(); ++component) {
    if (component == 4u || component == 8u) {
      CHECK(std::all_of(output[component].begin(), output[component].end(),
                        [](double value) { return value == kSentinel; }));
    } else {
      CHECK(output[component] == host.expected[component]);
    }
  }
  std::vector<double> aliased_entropy;
  std::vector<double> aliased_spin;
  CHECK(device.inputs[5].download(aliased_spin) == cudaSuccess);
  CHECK(device.inputs[1].download(aliased_entropy) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(aliased_spin == host.expected[4]);
  CHECK(aliased_entropy == host.expected[8]);

  HostCase overlap_host = make_case(2u);
  DeviceCase overlap_device(overlap_host);
  DeviceBuffer<double> overlap_storage(3u);
  CHECK(overlap_storage.upload({overlap_host.inputs[1][0], overlap_host.inputs[1][1], kSentinel}) ==
        cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);

  /* A shifted output shares only one element with the entropy input. Accepting
   * it would let publication overwrite a peer input, so only exact full-range
   * equality is legal. */
  Gfn2SccFreeEnergyDeviceInput partial_input = overlap_device.input;
  partial_input.entropy = overlap_storage.get();
  Gfn2SccFreeEnergyDeviceDiagnostics partial_diagnostics = overlap_device.diagnostics;
  partial_diagnostics.entropy = overlap_storage.get() + 1;
  CHECK(compose_gfn2_scc_free_energy_cuda(
            overlap_device.batch, partial_input, overlap_device.activity, partial_diagnostics,
            overlap_device.workspace, overlap_device.system_errors.get(),
            overlap_device.device_error.get()) == cudaErrorInvalidValue);

  /* Exact overlap is special only for occupation entropy -> entropy trace.
   * Binding the entropy destination to any other readable component remains a
   * forbidden write-after-read hazard. */
  Gfn2SccFreeEnergyDeviceDiagnostics wrong_input_alias = overlap_device.diagnostics;
  wrong_input_alias.entropy = const_cast<double*>(overlap_device.input.core);
  CHECK(compose_gfn2_scc_free_energy_cuda(
            overlap_device.batch, overlap_device.input, overlap_device.activity, wrong_input_alias,
            overlap_device.workspace, overlap_device.system_errors.get(),
            overlap_device.device_error.get()) == cudaErrorInvalidValue);

  Gfn2SccFreeEnergyDeviceDiagnostics wrong_output_alias = overlap_device.diagnostics;
  wrong_output_alias.internal_energy = const_cast<double*>(overlap_device.input.entropy);
  CHECK(compose_gfn2_scc_free_energy_cuda(
            overlap_device.batch, overlap_device.input, overlap_device.activity, wrong_output_alias,
            overlap_device.workspace, overlap_device.system_errors.get(),
            overlap_device.device_error.get()) == cudaErrorInvalidValue);
  return 0;
}

int test_cuda_graph_replay() {
  HostCase host = make_case(32u);
  DeviceCase device(host);
  CHECK(device.fill_outputs(kSentinel) == cudaSuccess);
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess);
  CHECK(reset_gfn2_scc_free_energy_device_errors_cuda(
            device.batch.batch_size, device.system_errors.get(), device.device_error.get(),
            stream) == cudaSuccess);
  CHECK(compose_gfn2_scc_free_energy_cuda(
            device.batch, device.input, device.activity, device.diagnostics, device.workspace,
            device.system_errors.get(), device.device_error.get(), stream) == cudaSuccess);
  CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
  CHECK(cudaGraphInstantiate(&executable, graph, 0) == cudaSuccess);
  CHECK(cudaGraphLaunch(executable, stream) == cudaSuccess);
  CHECK(cudaGraphLaunch(executable, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);

  std::array<std::vector<double>, kGfn2SccFreeEnergyDiagnosticComponents> output;
  CHECK(download_outputs(device, output, stream) == 0);
  for (std::size_t component = 0; component < output.size(); ++component) {
    CHECK(output[component] == host.expected[component]);
  }
  CHECK(cudaGraphExecDestroy(executable) == cudaSuccess);
  CHECK(cudaGraphDestroy(graph) == cudaSuccess);
  CHECK(cudaStreamDestroy(stream) == cudaSuccess);
  return 0;
}

}  // namespace

int main() {
  const std::array<int (*)(), 10> tests{
      {test_batch_parity_custom_stream_and_disabled_terms,
       test_exact_cpu_association_and_subtotal_counterexamples,
       test_final_fma_rounding_counterexample,
       test_electric_field_component_order_and_nonfinite_peer,
       test_existing_cuda_stage_output_binding,
       test_nonfinite_peer_isolation_inactive_and_upstream_system_error,
       test_intermediate_and_final_fma_overflow_and_sticky_plan_error,
       test_hostile_metadata_alias_token_misalignment_and_reset,
       test_staged_spin_entropy_exact_alias_and_overlap_rejection, test_cuda_graph_replay}};
  for (const auto test : tests) {
    const int status = test();
    if (status != 0) {
      return status;
    }
  }
  return 0;
}
