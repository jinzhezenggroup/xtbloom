#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "backends/cuda/gfn2_electric_field.cuh"

#define CHECK(condition)                                                              \
  do {                                                                                \
    if (!(condition)) {                                                               \
      std::cerr << "check failed at line " << __LINE__ << ": " << #condition << '\n'; \
      return __LINE__;                                                                \
    }                                                                                 \
  } while (false)
#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using namespace xtbloom::detail::cuda;
constexpr std::uint64_t kToken = 0xef1e1d237ULL;
constexpr double kSentinel = 777.0;

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count != 0 &&
        cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)) != cudaSuccess)
      std::abort();
  }
  ~DeviceBuffer() { cudaFree(data_); }
  T* get() const { return data_; }
  cudaError_t upload(const std::vector<T>& values, cudaStream_t stream) {
    return cudaMemcpyAsync(data_, values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice,
                           stream);
  }
  cudaError_t download(std::vector<T>& values, cudaStream_t stream) const {
    values.resize(count_);
    return cudaMemcpyAsync(values.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost,
                           stream);
  }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

int test_potentials_and_peer_failure() {
  const std::vector<std::int64_t> offsets{0, 2, 3};
  const std::vector<double> positions{1.0, 2.0, 3.0, -2.0, 0.5, 4.0, 3.0, -1.0, 0.25};
  const std::vector<double> fields{0.2, -0.3, 0.4, 0.0, 0.0, 0.0};
  DeviceBuffer<std::int64_t> d_offsets(offsets.size());
  DeviceBuffer<double> d_positions(positions.size());
  DeviceBuffer<double> d_fields(fields.size());
  DeviceBuffer<double> d_atomic(3);
  DeviceBuffer<double> d_dipole(9);
  DeviceBuffer<std::uint32_t> d_system_errors(2);
  DeviceBuffer<std::uint32_t> d_plan_error(1);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(d_offsets.upload(offsets, stream));
  CUDA_CHECK(d_positions.upload(positions, stream));
  CUDA_CHECK(d_fields.upload(fields, stream));
  CUDA_CHECK(d_atomic.upload(std::vector<double>(3, kSentinel), stream));
  CUDA_CHECK(d_dipole.upload(std::vector<double>(9, kSentinel), stream));
  const Gfn2ElectricFieldDeviceBatch batch{2, 3, 3, d_offsets.get(), kToken};
  const Gfn2ElectricFieldDeviceInput input{d_fields.get(), 6, d_positions.get(), 9, kToken};
  const Gfn2ElectricFieldDevicePotentials output{d_atomic.get(), 3, d_dipole.get(), 9, kToken};
  CUDA_CHECK(reset_gfn2_electric_field_device_errors_cuda(2, d_system_errors.get(),
                                                          d_plan_error.get(), stream));
  CUDA_CHECK(refresh_gfn2_electric_field_potentials_cuda(
      batch, input, output, d_system_errors.get(), d_plan_error.get(), stream));
  std::vector<double> atomic;
  std::vector<double> dipole;
  std::vector<std::uint32_t> errors;
  CUDA_CHECK(d_atomic.download(atomic, stream));
  CUDA_CHECK(d_dipole.download(dipole, stream));
  CUDA_CHECK(d_system_errors.download(errors, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(atomic == std::vector<double>({-0.8, -1.05, -0.0}));
  CHECK(dipole == std::vector<double>({-0.2, 0.3, -0.4, -0.2, 0.3, -0.4, -0.0, -0.0, -0.0}));
  CHECK(errors == std::vector<std::uint32_t>({0u, 0u}));

  std::vector<double> bad_fields = fields;
  bad_fields[3] = std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(d_fields.upload(bad_fields, stream));
  CUDA_CHECK(d_atomic.upload(std::vector<double>(3, kSentinel), stream));
  CUDA_CHECK(d_dipole.upload(std::vector<double>(9, kSentinel), stream));
  CUDA_CHECK(reset_gfn2_electric_field_device_errors_cuda(2, d_system_errors.get(),
                                                          d_plan_error.get(), stream));
  CUDA_CHECK(refresh_gfn2_electric_field_potentials_cuda(
      batch, input, output, d_system_errors.get(), d_plan_error.get(), stream));
  CUDA_CHECK(d_atomic.download(atomic, stream));
  CUDA_CHECK(d_dipole.download(dipole, stream));
  CUDA_CHECK(d_system_errors.download(errors, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(atomic[0] == -0.8 && atomic[1] == -1.05 && atomic[2] == kSentinel);
  CHECK(dipole[6] == kSentinel && dipole[7] == kSentinel && dipole[8] == kSentinel);
  CHECK(errors[0] == 0u);
  CHECK(errors[1] == static_cast<std::uint32_t>(Gfn2ElectricFieldDeviceError::kNonfiniteVector));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_graph_replay_changed_field() {
  const std::vector<std::int64_t> offsets{0, 1};
  const std::vector<double> positions{2.0, -1.0, 0.5};
  DeviceBuffer<std::int64_t> d_offsets(2);
  DeviceBuffer<double> d_positions(3);
  DeviceBuffer<double> d_fields(3);
  DeviceBuffer<double> d_atomic(1);
  DeviceBuffer<double> d_dipole(3);
  DeviceBuffer<std::uint32_t> d_system_errors(1);
  DeviceBuffer<std::uint32_t> d_plan_error(1);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(d_offsets.upload(offsets, stream));
  CUDA_CHECK(d_positions.upload(positions, stream));
  CUDA_CHECK(d_fields.upload({0.1, 0.2, 0.3}, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const Gfn2ElectricFieldDeviceBatch batch{1, 1, 2, d_offsets.get(), kToken};
  const Gfn2ElectricFieldDeviceInput input{d_fields.get(), 3, d_positions.get(), 3, kToken};
  const Gfn2ElectricFieldDevicePotentials output{d_atomic.get(), 1, d_dipole.get(), 3, kToken};
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(reset_gfn2_electric_field_device_errors_cuda(1, d_system_errors.get(),
                                                          d_plan_error.get(), stream));
  CUDA_CHECK(refresh_gfn2_electric_field_potentials_cuda(
      batch, input, output, d_system_errors.get(), d_plan_error.get(), stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUDA_CHECK(d_fields.upload({-0.4, 0.0, 0.2}, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  std::vector<double> atomic;
  CUDA_CHECK(d_atomic.download(atomic, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(atomic.size() == 1);
  CHECK(std::abs(atomic[0] - 0.7) <= 1.0e-15);
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main() {
  int devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || devices == 0) {
    cudaGetLastError();
    std::cout << "CUDA electric-field test skipped: no device\n";
    return 0;
  }
  CUDA_CHECK(status);
  CHECK(test_potentials_and_peer_failure() == 0);
  CHECK(test_graph_replay_changed_field() == 0);
  std::cout << "CUDA electric-field tests passed\n";
  return 0;
}
