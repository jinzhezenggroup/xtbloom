#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "backends/cuda/gfn2_force_composition.cuh"

#define CHECK(condition)                                                              \
  do {                                                                                \
    if (!(condition)) {                                                               \
      std::cerr << "check failed at line " << __LINE__ << ": " << #condition << '\n'; \
      return __LINE__;                                                                \
    }                                                                                 \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using xtbloom::detail::cuda::compose_gfn2_forces_cuda;
using xtbloom::detail::cuda::Gfn2ForceCompositionComponent;
using xtbloom::detail::cuda::Gfn2ForceCompositionDeviceBatch;
using xtbloom::detail::cuda::Gfn2ForceCompositionDeviceError;
using xtbloom::detail::cuda::Gfn2ForceCompositionDeviceInput;
using xtbloom::detail::cuda::Gfn2ForceCompositionDeviceOutput;
using xtbloom::detail::cuda::Gfn2ForceCompositionDeviceWorkspace;
using xtbloom::detail::cuda::Gfn2ForceDeviceActivity;
using xtbloom::detail::cuda::reset_gfn2_force_composition_device_errors_cuda;

constexpr std::uint64_t kPlanToken = 0x67f0ce55ULL;
constexpr double kSentinel = 731.25;

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  cudaError_t allocate(std::size_t count) {
    count_ = count;
    return count == 0u ? cudaSuccess
                       : cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
  }

  cudaError_t upload(const std::vector<T>& values, cudaStream_t stream = nullptr) {
    if (values.size() != count_) {
      return cudaErrorInvalidValue;
    }
    return values.empty() ? cudaSuccess
                          : cudaMemcpyAsync(data_, values.data(), values.size() * sizeof(T),
                                            cudaMemcpyHostToDevice, stream);
  }

  cudaError_t download(std::vector<T>& values, cudaStream_t stream = nullptr) const {
    values.resize(count_);
    return values.empty() ? cudaSuccess
                          : cudaMemcpyAsync(values.data(), data_, values.size() * sizeof(T),
                                            cudaMemcpyDeviceToHost, stream);
  }

  T* get() noexcept { return data_; }
  const T* get() const noexcept { return data_; }
  std::size_t size() const noexcept { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

struct HostCase {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> point_offsets;
  std::vector<std::uint8_t> requested;
  std::vector<xtbloom_status_t> statuses;
  std::vector<double> electronic;
  std::vector<double> classical;
  std::vector<double> explicit_qm;
  std::vector<double> explicit_point;
  std::vector<double> expected_qm;
  std::vector<double> expected_point;
};

HostCase make_case(std::size_t batch_size) {
  HostCase host;
  host.atom_offsets.resize(batch_size + 1u, 0);
  host.point_offsets.resize(batch_size + 1u, 0);
  for (std::size_t system = 0; system < batch_size; ++system) {
    host.atom_offsets[system + 1u] =
        host.atom_offsets[system] + 1 + static_cast<std::int64_t>(system % 3u);
    host.point_offsets[system + 1u] =
        host.point_offsets[system] + 1 + static_cast<std::int64_t>(system % 3u);
  }
  const std::size_t qm_elements = static_cast<std::size_t>(host.atom_offsets.back()) * 3u;
  const std::size_t point_elements = static_cast<std::size_t>(host.point_offsets.back()) * 3u;
  host.requested.assign(batch_size, 1u);
  host.statuses.assign(batch_size, XTBLOOM_STATUS_SUCCESS);
  host.electronic.resize(qm_elements);
  host.classical.resize(qm_elements);
  host.explicit_qm.resize(qm_elements);
  host.expected_qm.resize(qm_elements);
  host.explicit_point.resize(point_elements);
  host.expected_point.resize(point_elements);
  for (std::size_t coordinate = 0; coordinate < qm_elements; ++coordinate) {
    host.electronic[coordinate] = 0.001 * static_cast<double>(coordinate + 1u);
    host.classical[coordinate] = -0.0003 * static_cast<double>((coordinate % 17u) + 1u);
    host.explicit_qm[coordinate] = 0.0002 * static_cast<double>((coordinate % 11u) - 5.0);
    double force = -host.electronic[coordinate];
    force += host.classical[coordinate];
    force += host.explicit_qm[coordinate];
    host.expected_qm[coordinate] = force;
  }
  for (std::size_t coordinate = 0; coordinate < point_elements; ++coordinate) {
    host.explicit_point[coordinate] = -0.002 * static_cast<double>((coordinate % 13u) + 1u);
    host.expected_point[coordinate] = host.explicit_point[coordinate];
  }
  return host;
}

struct DeviceCase {
  explicit DeviceCase(const HostCase& host) {
    const std::int64_t batch_size = static_cast<std::int64_t>(host.requested.size());
    const std::int64_t total_atoms = host.atom_offsets.back();
    const std::int64_t total_points = host.point_offsets.back();
    require(atom_offsets.allocate(host.atom_offsets.size()));
    require(point_offsets.allocate(host.point_offsets.size()));
    require(requested.allocate(host.requested.size()));
    require(statuses.allocate(host.statuses.size()));
    require(electronic.allocate(host.electronic.size()));
    require(classical.allocate(host.classical.size()));
    require(explicit_qm.allocate(host.explicit_qm.size()));
    require(explicit_point.allocate(host.explicit_point.size()));
    require(qm_output.allocate(host.expected_qm.size()));
    require(point_output.allocate(host.expected_point.size()));
    require(qm_scratch.allocate(host.expected_qm.size()));
    require(point_scratch.allocate(host.expected_point.size()));
    require(system_errors.allocate(host.requested.size()));
    require(plan_error.allocate(1u));
    require(sequence_active.allocate(1u));

    batch = {
        batch_size,
        total_atoms,
        total_points,
        batch_size + 1,
        batch_size + 1,
        atom_offsets.get(),
        point_offsets.get(),
        static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kElectronicGradient) |
            static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kClassicalForce) |
            static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kExplicitPointChargeForce),
        kPlanToken};
    activity = {requested.get(), statuses.get(), batch_size, kPlanToken};
    input = {electronic.get(),
             static_cast<std::int64_t>(host.electronic.size()),
             classical.get(),
             static_cast<std::int64_t>(host.classical.size()),
             explicit_qm.get(),
             static_cast<std::int64_t>(host.explicit_qm.size()),
             explicit_point.get(),
             static_cast<std::int64_t>(host.explicit_point.size()),
             kPlanToken};
    output = {qm_output.get(), static_cast<std::int64_t>(host.expected_qm.size()),
              point_output.get(), static_cast<std::int64_t>(host.expected_point.size()),
              kPlanToken};
    workspace = {qm_scratch.get(),
                 static_cast<std::int64_t>(host.expected_qm.size()),
                 point_scratch.get(),
                 static_cast<std::int64_t>(host.expected_point.size()),
                 sequence_active.get(),
                 1,
                 kPlanToken};
  }

  cudaError_t upload(const HostCase& host, cudaStream_t stream) {
    cudaError_t status = atom_offsets.upload(host.atom_offsets, stream);
    if (status != cudaSuccess) return status;
    status = point_offsets.upload(host.point_offsets, stream);
    if (status != cudaSuccess) return status;
    status = requested.upload(host.requested, stream);
    if (status != cudaSuccess) return status;
    status = statuses.upload(host.statuses, stream);
    if (status != cudaSuccess) return status;
    status = electronic.upload(host.electronic, stream);
    if (status != cudaSuccess) return status;
    status = classical.upload(host.classical, stream);
    if (status != cudaSuccess) return status;
    status = explicit_qm.upload(host.explicit_qm, stream);
    if (status != cudaSuccess) return status;
    status = explicit_point.upload(host.explicit_point, stream);
    if (status != cudaSuccess) return status;
    status = qm_output.upload(std::vector<double>(host.expected_qm.size(), kSentinel), stream);
    if (status != cudaSuccess) return status;
    return point_output.upload(std::vector<double>(host.expected_point.size(), kSentinel), stream);
  }

  Gfn2ForceCompositionDeviceBatch batch{};
  Gfn2ForceDeviceActivity activity{};
  Gfn2ForceCompositionDeviceInput input{};
  Gfn2ForceCompositionDeviceOutput output{};
  Gfn2ForceCompositionDeviceWorkspace workspace{};
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> point_offsets;
  DeviceBuffer<std::uint8_t> requested;
  DeviceBuffer<xtbloom_status_t> statuses;
  DeviceBuffer<double> electronic;
  DeviceBuffer<double> classical;
  DeviceBuffer<double> explicit_qm;
  DeviceBuffer<double> explicit_point;
  DeviceBuffer<double> qm_output;
  DeviceBuffer<double> point_output;
  DeviceBuffer<double> qm_scratch;
  DeviceBuffer<double> point_scratch;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> plan_error;
  DeviceBuffer<std::uint32_t> sequence_active;

 private:
  static void require(cudaError_t status) {
    if (status != cudaSuccess) {
      std::cerr << "CUDA fixture setup failed: " << cudaGetErrorString(status) << '\n';
      std::abort();
    }
  }
};

int run_case(std::size_t batch_size) {
  HostCase host = make_case(batch_size);
  DeviceCase device(host);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(device.upload(host, stream));
  CUDA_CHECK(reset_gfn2_force_composition_device_errors_cuda(
      device.batch.batch_size, device.system_errors.get(), device.plan_error.get(), stream));
  CUDA_CHECK(compose_gfn2_forces_cuda(device.batch, device.activity, device.input, device.output,
                                      device.workspace, device.system_errors.get(),
                                      device.plan_error.get(), stream));
  std::vector<double> qm;
  std::vector<double> point;
  std::vector<std::uint32_t> errors;
  std::vector<std::uint32_t> plan;
  CUDA_CHECK(device.qm_output.download(qm, stream));
  CUDA_CHECK(device.point_output.download(point, stream));
  CUDA_CHECK(device.system_errors.download(errors, stream));
  CUDA_CHECK(device.plan_error.download(plan, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CHECK(qm == host.expected_qm);
  CHECK(point == host.expected_point);
  CHECK(errors == std::vector<std::uint32_t>(batch_size, 0u));
  CHECK(plan == std::vector<std::uint32_t>{0u});
  return 0;
}

int test_batch_matrix() {
  for (const std::size_t batch_size : std::array<std::size_t, 4>{1u, 8u, 32u, 128u}) {
    CHECK(run_case(batch_size) == 0);
  }
  return 0;
}

int test_terminal_gate_and_transactionality() {
  HostCase host = make_case(4u);
  host.requested[1] = 0u;
  host.statuses[2] = XTBLOOM_STATUS_SCC_NOT_CONVERGED;
  const auto poison_system = [&](std::size_t system) {
    const std::size_t atom_begin = static_cast<std::size_t>(host.atom_offsets[system]) * 3u;
    const std::size_t atom_end = static_cast<std::size_t>(host.atom_offsets[system + 1u]) * 3u;
    std::fill(host.electronic.begin() + static_cast<std::ptrdiff_t>(atom_begin),
              host.electronic.begin() + static_cast<std::ptrdiff_t>(atom_end),
              std::numeric_limits<double>::quiet_NaN());
    std::fill(host.classical.begin() + static_cast<std::ptrdiff_t>(atom_begin),
              host.classical.begin() + static_cast<std::ptrdiff_t>(atom_end),
              std::numeric_limits<double>::quiet_NaN());
    std::fill(host.explicit_qm.begin() + static_cast<std::ptrdiff_t>(atom_begin),
              host.explicit_qm.begin() + static_cast<std::ptrdiff_t>(atom_end),
              std::numeric_limits<double>::quiet_NaN());
    const std::size_t point_begin = static_cast<std::size_t>(host.point_offsets[system]) * 3u;
    const std::size_t point_end = static_cast<std::size_t>(host.point_offsets[system + 1u]) * 3u;
    std::fill(host.explicit_point.begin() + static_cast<std::ptrdiff_t>(point_begin),
              host.explicit_point.begin() + static_cast<std::ptrdiff_t>(point_end),
              std::numeric_limits<double>::quiet_NaN());
  };
  poison_system(1u);
  poison_system(2u);
  const std::size_t failed_coordinate = static_cast<std::size_t>(host.atom_offsets[3]) * 3u;
  host.classical[failed_coordinate] = std::numeric_limits<double>::quiet_NaN();

  DeviceCase device(host);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(device.upload(host, stream));
  CUDA_CHECK(reset_gfn2_force_composition_device_errors_cuda(
      device.batch.batch_size, device.system_errors.get(), device.plan_error.get(), stream));
  CUDA_CHECK(compose_gfn2_forces_cuda(device.batch, device.activity, device.input, device.output,
                                      device.workspace, device.system_errors.get(),
                                      device.plan_error.get(), stream));
  std::vector<double> qm;
  std::vector<double> point;
  std::vector<std::uint32_t> errors;
  CUDA_CHECK(device.qm_output.download(qm, stream));
  CUDA_CHECK(device.point_output.download(point, stream));
  CUDA_CHECK(device.system_errors.download(errors, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));

  const auto check_system = [&](std::size_t system, bool published) {
    for (std::int64_t atom = host.atom_offsets[system]; atom < host.atom_offsets[system + 1u];
         ++atom) {
      for (std::int64_t coordinate = atom * 3; coordinate < atom * 3 + 3; ++coordinate) {
        if (published) {
          if (qm[static_cast<std::size_t>(coordinate)] !=
              host.expected_qm[static_cast<std::size_t>(coordinate)]) {
            return false;
          }
        } else if (qm[static_cast<std::size_t>(coordinate)] != kSentinel) {
          return false;
        }
      }
    }
    for (std::int64_t point_index = host.point_offsets[system];
         point_index < host.point_offsets[system + 1u]; ++point_index) {
      for (std::int64_t coordinate = point_index * 3; coordinate < point_index * 3 + 3;
           ++coordinate) {
        if (published) {
          if (point[static_cast<std::size_t>(coordinate)] !=
              host.expected_point[static_cast<std::size_t>(coordinate)]) {
            return false;
          }
        } else if (point[static_cast<std::size_t>(coordinate)] != kSentinel) {
          return false;
        }
      }
    }
    return true;
  };
  CHECK(check_system(0u, true));
  CHECK(check_system(1u, false));
  CHECK(check_system(2u, false));
  CHECK(check_system(3u, false));
  CHECK(errors[0] == 0u && errors[1] == 0u && errors[2] == 0u);
  CHECK(errors[3] ==
        static_cast<std::uint32_t>(Gfn2ForceCompositionDeviceError::kNonfiniteClassicalForce));
  return 0;
}

int test_invalid_request_and_peer_overflow() {
  HostCase host = make_case(3u);
  host.requested[1] = 2u;
  const std::size_t overflow_coordinate = static_cast<std::size_t>(host.atom_offsets[2]) * 3u;
  host.electronic[overflow_coordinate] = -std::numeric_limits<double>::max();
  host.classical[overflow_coordinate] = std::numeric_limits<double>::max();
  host.explicit_qm[overflow_coordinate] = 0.0;

  DeviceCase device(host);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(device.upload(host, stream));
  CUDA_CHECK(reset_gfn2_force_composition_device_errors_cuda(
      device.batch.batch_size, device.system_errors.get(), device.plan_error.get(), stream));
  CUDA_CHECK(compose_gfn2_forces_cuda(device.batch, device.activity, device.input, device.output,
                                      device.workspace, device.system_errors.get(),
                                      device.plan_error.get(), stream));
  std::vector<double> qm;
  std::vector<std::uint32_t> errors;
  CUDA_CHECK(device.qm_output.download(qm, stream));
  CUDA_CHECK(device.system_errors.download(errors, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));

  for (std::int64_t coordinate = host.atom_offsets[0] * 3; coordinate < host.atom_offsets[1] * 3;
       ++coordinate) {
    CHECK(qm[static_cast<std::size_t>(coordinate)] ==
          host.expected_qm[static_cast<std::size_t>(coordinate)]);
  }
  for (std::size_t system = 1; system < 3u; ++system) {
    for (std::int64_t coordinate = host.atom_offsets[system] * 3;
         coordinate < host.atom_offsets[system + 1u] * 3; ++coordinate) {
      CHECK(qm[static_cast<std::size_t>(coordinate)] == kSentinel);
    }
  }
  CHECK(errors[0] == 0u);
  CHECK(errors[1] ==
        static_cast<std::uint32_t>(Gfn2ForceCompositionDeviceError::kInvalidRequestedMask));
  CHECK(errors[2] ==
        static_cast<std::uint32_t>(Gfn2ForceCompositionDeviceError::kNonfiniteForceArithmetic));
  return 0;
}

int test_plan_failure_is_whole_call_atomic() {
  HostCase host = make_case(8u);
  DeviceCase device(host);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(device.upload(host, stream));
  std::vector<std::int64_t> invalid_offsets = host.atom_offsets;
  invalid_offsets.back() -= 1;
  CUDA_CHECK(device.atom_offsets.upload(invalid_offsets, stream));
  CUDA_CHECK(reset_gfn2_force_composition_device_errors_cuda(
      device.batch.batch_size, device.system_errors.get(), device.plan_error.get(), stream));
  CUDA_CHECK(compose_gfn2_forces_cuda(device.batch, device.activity, device.input, device.output,
                                      device.workspace, device.system_errors.get(),
                                      device.plan_error.get(), stream));
  std::vector<double> qm;
  std::vector<double> point;
  std::vector<std::uint32_t> plan;
  CUDA_CHECK(device.qm_output.download(qm, stream));
  CUDA_CHECK(device.point_output.download(point, stream));
  CUDA_CHECK(device.plan_error.download(plan, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CHECK(std::all_of(qm.begin(), qm.end(), [](double value) { return value == kSentinel; }));
  CHECK(std::all_of(point.begin(), point.end(), [](double value) { return value == kSentinel; }));
  CHECK(plan == std::vector<std::uint32_t>{
                    static_cast<std::uint32_t>(Gfn2ForceCompositionDeviceError::kInvalidOffsets)});
  return 0;
}

int test_changed_input_graph_replay() {
  HostCase first = make_case(8u);
  HostCase second = first;
  for (std::size_t coordinate = 0; coordinate < second.electronic.size(); ++coordinate) {
    second.electronic[coordinate] += 0.125;
    double force = -second.electronic[coordinate];
    force += second.classical[coordinate];
    force += second.explicit_qm[coordinate];
    second.expected_qm[coordinate] = force;
  }
  DeviceCase device(first);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(device.upload(first, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(reset_gfn2_force_composition_device_errors_cuda(
      device.batch.batch_size, device.system_errors.get(), device.plan_error.get(), stream));
  CUDA_CHECK(compose_gfn2_forces_cuda(device.batch, device.activity, device.input, device.output,
                                      device.workspace, device.system_errors.get(),
                                      device.plan_error.get(), stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(device.electronic.upload(second.electronic, stream));
  CUDA_CHECK(
      device.qm_output.upload(std::vector<double>(second.expected_qm.size(), kSentinel), stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  std::vector<double> qm;
  CUDA_CHECK(device.qm_output.download(qm, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CHECK(qm == second.expected_qm);
  return 0;
}

int test_host_validation() {
  HostCase host = make_case(1u);
  DeviceCase device(host);
  Gfn2ForceCompositionDeviceOutput alias = device.output;
  alias.qm_forces = device.electronic.get();
  CHECK(compose_gfn2_forces_cuda(device.batch, device.activity, device.input, alias,
                                 device.workspace, device.system_errors.get(),
                                 device.plan_error.get()) == cudaErrorInvalidValue);
  Gfn2ForceCompositionDeviceBatch invalid = device.batch;
  invalid.enabled_components = 0x80000000u;
  CHECK(compose_gfn2_forces_cuda(invalid, device.activity, device.input, device.output,
                                 device.workspace, device.system_errors.get(),
                                 device.plan_error.get()) == cudaErrorInvalidValue);
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status == cudaErrorNoDevice || count_status == cudaErrorInsufficientDriver ||
      device_count == 0) {
    cudaGetLastError();
    std::cout << "CUDA GFN2 force-composition test skipped: no CUDA device\n";
    return 0;
  }
  CUDA_CHECK(count_status);
  CHECK(test_batch_matrix() == 0);
  CHECK(test_terminal_gate_and_transactionality() == 0);
  CHECK(test_invalid_request_and_peer_overflow() == 0);
  CHECK(test_plan_failure_is_whole_call_atomic() == 0);
  CHECK(test_changed_input_graph_replay() == 0);
  CHECK(test_host_validation() == 0);
  std::cout << "CUDA GFN2 force-composition tests passed\n";
  return 0;
}
