#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

#include "backends/cuda/cuda_atomics.cuh"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

__global__ void add_kernel(double* value, double increment) {
  gpuxtb::detail::cuda::atomic_add_fp64(value, increment);
}

}  // namespace

int main() {
  double* device_value = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_value), sizeof(double)));

  double host_value = 1.25;
  CUDA_CHECK(cudaMemcpy(device_value, &host_value, sizeof(double), cudaMemcpyHostToDevice));
  add_kernel<<<1, 64>>>(device_value, 0.5);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(&host_value, device_value, sizeof(double), cudaMemcpyDeviceToHost));
  CHECK(host_value == 33.25);

  /* Integer payload comparison must terminate even when the stored value is NaN. */
  host_value = NAN;
  CUDA_CHECK(cudaMemcpy(device_value, &host_value, sizeof(double), cudaMemcpyHostToDevice));
  add_kernel<<<1, 1>>>(device_value, 1.0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(&host_value, device_value, sizeof(double), cudaMemcpyDeviceToHost));
  CHECK(std::isnan(host_value));

  CUDA_CHECK(cudaFree(device_value));
  return 0;
}
